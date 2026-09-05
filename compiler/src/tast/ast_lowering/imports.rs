//! `import` and `using`, including import.hx.

use super::*;
use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

impl<'a> AstLowering<'a> {
    /// Initialize span converter for proper source location tracking
    pub fn initialize_span_converter(&mut self, file_id: u32, source_text: String) {
        self.context.initialize_span_converter(file_id, source_text);
    }

    /// Initialize span converter with specific filename for proper source location tracking
    pub fn initialize_span_converter_with_filename(
        &mut self,
        file_id: u32,
        source_text: String,
        file_name: String,
    ) {
        self.context
            .initialize_span_converter_with_filename(file_id, source_text, file_name);
    }

    /// Lower a complete Haxe file to TAST

    /// Process import.hx files in the directory hierarchy
    pub(crate) fn process_import_hx_files(
        &mut self,
        current_file: &HaxeFile,
    ) -> LoweringResult<()> {
        use crate::tast::stdlib_loader::{StdLibConfig, StdLibLoader};
        use std::path::PathBuf;

        // The directory of the file being lowered. This used to be a hardcoded
        // absolute path into one checkout, so import.hx was looked for in a
        // directory that exists on one machine and is not where the file lives
        // even there -- the feature was wired up and could never fire.
        // Absolute, because the rest of the search walks parents: a bare
        // "." has no ancestors to walk, so a file named without a directory
        // would only ever look in one place.
        let current_dir = match PathBuf::from(&current_file.filename).parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let current_dir = std::fs::canonicalize(&current_dir).unwrap_or(current_dir);

        // Create a loader for import.hx files
        let mut config = StdLibConfig::default();
        config.load_import_hx = true;
        config.std_paths = vec![]; // We're not loading std lib here
        config.default_imports = vec![]; // No default imports

        let mut loader = StdLibLoader::new(config);

        // Look for import.hx files in the current directory and parent directories
        let mut search_dir = current_dir.clone();
        let mut import_files = Vec::new();

        let probe = crate::debug_flags::wildcard_log();
        let dirs = Self::import_hx_search_dirs(current_file, &search_dir);
        if probe {
            eprintln!(
                "[import.hx] file={} dir={} searching {:?}",
                current_file.filename,
                search_dir.display(),
                dirs
            );
        }
        for dir in dirs {
            let found = loader.load_import_hx(&dir);
            if probe && !found.is_empty() {
                eprintln!("[import.hx] loaded {} from {}", found.len(), dir.display());
            }
            import_files.extend(found);
        }

        // Process import.hx files in reverse order (parent directories first)
        import_files.reverse();

        for import_file in import_files {
            // Process imports from import.hx
            for import in &import_file.imports {
                self.process_import_from_import_hx(import)?;
            }

            // Process using statements from import.hx
            for using in &import_file.using {
                self.process_using_from_import_hx(using)?;
            }

            // Process type declarations from import.hx
            // Pre-register first (creates symbols in scope), then fully lower enums
            // so their variant constructor types are properly set (not left as TypeId::invalid)
            for declaration in &import_file.declarations {
                self.pre_register_declaration(declaration)?;
            }
            for declaration in &import_file.declarations {
                if let TypeDeclaration::Enum(enum_decl) = declaration {
                    let _ = self.lower_enum_declaration(enum_decl);
                }
            }
        }

        Ok(())
    }

    /// Process an import from import.hx file
    fn process_import_from_import_hx(&mut self, import: &Import) -> LoweringResult<()> {
        // An import.hx entry is an ordinary import that happens to be written
        // in another file, so it goes through the ordinary path. Handling it
        // separately is what silently dropped every wildcard: this only ever
        // registered concrete symbols and never built an ImportEntry, so
        // `import utest.Assert.*` in an import.hx bound nothing whatsoever.
        //
        // Registered against the root scope, because an import.hx applies to
        // every module beneath it and not only to the file being lowered when
        // it happened to be found.
        let saved = self.context.current_scope;
        self.context.current_scope = ScopeId::first();
        let result = self.lower_import(import).map(|_| ());
        self.context.current_scope = saved;
        result
    }

    /// Directories an `import.hx` is honoured in, outermost first.
    ///
    /// Haxe applies an import.hx to every module at or below it, so three
    /// roots matter, and each is processed before the ones nearer the file so
    /// that a nearer import.hx is applied last:
    ///
    ///   1. the compiler's own stdlib, so rayzor can ship defaults with it;
    ///   2. the project root -- the nearest ancestor holding a rayzor.toml,
    ///      falling back to the working directory;
    ///   3. the module's class-path root down to the file's own directory.
    ///
    /// The package declaration bounds (3) exactly: a file in `package cases`
    /// sits one directory below its class-path root, so there are
    /// package-depth + 1 directories in that chain. Walking past the root
    /// would pick up an import.hx belonging to an unrelated tree.
    fn import_hx_search_dirs(
        file: &HaxeFile,
        file_dir: &std::path::Path,
    ) -> Vec<std::path::PathBuf> {
        let mut dirs: Vec<std::path::PathBuf> = Vec::new();
        let mut push = |d: std::path::PathBuf, dirs: &mut Vec<std::path::PathBuf>| {
            if d.is_dir() && !dirs.contains(&d) {
                dirs.push(d);
            }
        };

        // 1. The compiler's own stdlib.
        let std_roots = crate::compilation::CompilationConfig::discover_stdlib_paths();
        for p in std_roots.iter() {
            push(p.clone(), &mut dirs);
        }

        // A stdlib module gets the compiler's own import.hx and nothing else.
        // The project root below is found by walking up from the file, and for
        // a stdlib file that walk leaves the project entirely and lands on the
        // working directory -- which would inject whatever the user happens to
        // have in scope into every stdlib module.
        let in_stdlib = std_roots.iter().any(|root| file_dir.starts_with(root));
        if in_stdlib {
            return dirs;
        }

        // 2. The project root: the nearest ancestor with a manifest.
        let mut probe = Some(file_dir.to_path_buf());
        let mut project_root: Option<std::path::PathBuf> = None;
        while let Some(dir) = probe {
            if dir.join("rayzor.toml").is_file() {
                project_root = Some(dir.clone());
                break;
            }
            probe = dir.parent().map(|p| p.to_path_buf());
        }
        match project_root {
            Some(root) => push(root, &mut dirs),
            None => {
                if let Ok(cwd) = std::env::current_dir() {
                    push(cwd, &mut dirs);
                }
            }
        }

        // 3. The class-path root down to the file's own directory.
        let depth = file.package.as_ref().map(|p| p.path.len()).unwrap_or(0);
        let mut chain = Vec::new();
        let mut dir = file_dir.to_path_buf();
        for _ in 0..=depth {
            chain.push(dir.clone());
            match dir.parent() {
                Some(parent) => dir = parent.to_path_buf(),
                None => break,
            }
        }
        chain.reverse();
        for d in chain {
            push(d, &mut dirs);
        }

        dirs
    }

    /// Process a using statement from import.hx file
    fn process_using_from_import_hx(&mut self, using: &Using) -> LoweringResult<()> {
        // Register the using statement globally
        // In a full implementation, we'd track these for static extension resolution
        let _path = using.path.join(".");
        // TODO: Track global using statements for static extension resolution
        Ok(())
    }

    /// Extract module name from file
    fn extract_module_name(&self, file: &HaxeFile) -> String {
        if let Some(package) = &file.package {
            package.path.join(".")
        } else {
            "default".to_string()
        }
    }

    /// Lower an import declaration
    pub(crate) fn lower_import(&mut self, import: &Import) -> LoweringResult<TypedImport> {
        let imported_symbols = match &import.mode {
            parser::ImportMode::Normal => import
                .path
                .last()
                .map(|s| vec![self.context.intern_string(s)]),
            parser::ImportMode::Alias(alias) => Some(vec![self.context.intern_string(alias)]),
            parser::ImportMode::Field(field) => Some(vec![self.context.intern_string(field)]),
            parser::ImportMode::Wildcard => None,
            parser::ImportMode::WildcardWithExclusions(_) => None,
        };

        let alias = match &import.mode {
            parser::ImportMode::Alias(alias) => Some(self.context.intern_string(alias)),
            _ => None,
        };

        // Create import entry for the import resolver
        let package_path: Vec<_> = import
            .path
            .iter()
            .take(import.path.len().saturating_sub(1)) // All but last element are package path
            .map(|s| self.context.string_interner.intern(s))
            .collect();

        let type_name = import
            .path
            .last()
            .map(|s| self.context.string_interner.intern(s))
            .unwrap_or_else(|| self.context.string_interner.intern("Unknown"));

        let qualified_path = super::namespace::QualifiedPath::new(package_path, type_name);

        // println!("Debug qualified path: {:?}", import.path);

        let alias_interned = alias;

        let exclusions = match &import.mode {
            parser::ImportMode::WildcardWithExclusions(excl) => excl
                .iter()
                .map(|e| self.context.string_interner.intern(e))
                .collect(),
            _ => Vec::new(),
        };

        // Clone qualified_path before moving it into import_entry
        let qualified_path_for_lookup = qualified_path.clone();

        let import_entry = super::namespace::ImportEntry {
            package_path: qualified_path,
            alias: alias_interned,
            exclusions,
            is_wildcard: matches!(
                import.mode,
                parser::ImportMode::Wildcard | parser::ImportMode::WildcardWithExclusions(_)
            ),
            location: self.context.create_location_from_span(import.span),
        };

        // Add import to current scope
        self.context
            .import_resolver
            .add_import(self.context.current_scope, import_entry);

        // Register imported symbols in the symbol table for type resolution
        if let Some(ref symbols) = imported_symbols {
            for &symbol_name in symbols {
                // IMPORTANT: Check if this symbol was already pre-registered (e.g., from stdlib loading)
                // If so, reuse that symbol instead of creating a duplicate
                // First try to look up by the full qualified path from the import
                let imported_symbol = if let Some(existing_symbol) = self
                    .context
                    .namespace_resolver
                    .lookup_symbol(&qualified_path_for_lookup)
                    .or_else(|| self.lookup_module_subtype(&import.path))
                {
                    // Reuse the pre-registered symbol from namespace
                    existing_symbol
                } else if import.path.len() > 1 {
                    // Import has a package path (e.g., rayzor.concurrent.Thread).
                    // Search by qualified_name first to avoid bare-name collisions
                    // (e.g., sys.thread.Thread vs rayzor.concurrent.Thread).
                    let full_qualified_name = import.path.join(".");
                    let qn_interned = self.context.string_interner.intern(&full_qualified_name);
                    if let Some(existing_symbol) = self
                        .context
                        .symbol_table
                        .resolve_qualified_name(qn_interned)
                    {
                        // Found the correct symbol — remap the name in the symbol table
                        // so expression resolution (symbol_table.lookup_symbol) finds it
                        self.context.symbol_table.remap_symbol_in_scope(
                            ScopeId::first(),
                            symbol_name,
                            existing_symbol,
                        );
                        existing_symbol
                    } else if let Some(existing_symbol) = self
                        .resolve_symbol_in_scope_hierarchy(symbol_name)
                        .filter(|sid| {
                            // Only reuse if the existing symbol has no qualified_name OR its
                            // qualified_name matches the imported path. Otherwise it's a DIFFERENT
                            // class with the same simple name (e.g. stdlib `Json` vs user `tink.Json`)
                            // and reusing it would silently route user imports to stdlib.
                            self.context
                                .symbol_table
                                .get_symbol(*sid)
                                .map(|sym| {
                                    sym.qualified_name.is_none()
                                        || sym.qualified_name == Some(qn_interned)
                                })
                                .unwrap_or(false)
                        })
                    {
                        // Bare-name fallback — set qualified_name to match the import path
                        // so downstream code (e.g., Send/Sync validation) can identify the type
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(existing_symbol)
                        {
                            if sym.qualified_name.is_none() {
                                sym.qualified_name = Some(qn_interned);
                            }
                        }
                        existing_symbol
                    } else {
                        self.create_import_placeholder(symbol_name, &full_qualified_name)
                    }
                } else if let Some(existing_symbol) =
                    self.resolve_symbol_in_scope_hierarchy(symbol_name)
                {
                    // Symbol already exists (likely from pre-registration)
                    // Reuse the existing symbol regardless of its kind (Class, Enum, Interface, etc.)
                    // This preserves the correct type info from the compiled file
                    if let Some(sym) = self.context.symbol_table.get_symbol(existing_symbol) {
                        // CRITICAL FIX: If the symbol has an invalid type_id, create a type for it
                        // This happens for extern classes that were created as placeholders but
                        // never had their type assigned
                        if !sym.type_id.is_valid()
                            && sym.kind == crate::tast::symbols::SymbolKind::Class
                        {
                            let class_type = self.context.type_table.borrow_mut().create_type(
                                crate::tast::core::TypeKind::Class {
                                    symbol_id: existing_symbol,
                                    type_args: Vec::new(),
                                },
                            );
                            self.context
                                .symbol_table
                                .update_symbol_type(existing_symbol, class_type);
                            self.context
                                .symbol_table
                                .register_type_symbol_mapping(class_type, existing_symbol);
                        }
                        existing_symbol
                    } else {
                        // Symbol ID exists but can't get symbol data - create new
                        let new_sym = self
                            .context
                            .symbol_table
                            .create_class_in_scope(symbol_name, ScopeId::first());
                        // Use the full import path as the qualified name
                        let full_qualified_name = import.path.join(".");
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(new_sym) {
                            sym.qualified_name =
                                Some(self.context.string_interner.intern(&full_qualified_name));
                        }
                        new_sym
                    }
                } else if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), symbol_name)
                {
                    // Symbol exists in root scope (from a previously compiled file)
                    // Reuse it to preserve correct type kind (Abstract, Enum, etc.)
                    existing.id
                } else {
                    let full_qualified_name = import.path.join(".");
                    self.create_import_placeholder(symbol_name, &full_qualified_name)
                };

                // Add to root scope so it can be resolved
                // Note: If symbol was pre-registered, it should already be in the scope,
                // but adding it again is idempotent
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(imported_symbol, symbol_name);

                self.import_enum_constructors(imported_symbol);
            }
        }

        Ok(TypedImport {
            module_path: self.context.intern_string(&import.path.join(".")),
            imported_symbols,
            alias,
            source_location: self.context.create_location_from_span(import.span),
        })
    }

    /// Placeholder for an imported type nothing has registered yet. The source's
    /// declaring keyword picks the kind: an `interface` placeholder left as a
    /// Class makes receivers typed by it call the abstract method by name
    /// instead of dispatching through the fat pointer.
    fn create_import_placeholder(
        &mut self,
        name: InternedString,
        qualified_name: &str,
    ) -> crate::tast::SymbolId {
        let is_interface =
            self.context.namespace_resolver.declared_kind(qualified_name) == Some("interface");
        if std::env::var_os("RAYZOR_SYM_DEBUG").is_some() {
            eprintln!("[sym] import-placeholder {qualified_name} interface={is_interface}");
        }
        let sym = if is_interface {
            self.context
                .symbol_table
                .create_interface_in_scope(name, ScopeId::first())
        } else {
            self.context
                .symbol_table
                .create_class_in_scope(name, ScopeId::first())
        };
        let qn = self.context.string_interner.intern(qualified_name);
        if let Some(s) = self.context.symbol_table.get_symbol_mut(sym) {
            s.qualified_name = Some(qn);
        }
        // A typeless placeholder leaves `this` untyped once the class itself is
        // lowered against it, so the placeholder carries its type from the start.
        let kind = if is_interface {
            TypeKind::Interface { symbol_id: sym, type_args: Vec::new() }
        } else {
            TypeKind::Class { symbol_id: sym, type_args: Vec::new() }
        };
        let ty = self.context.type_table.borrow_mut().create_type(kind);
        self.context.symbol_table.update_symbol_type(sym, ty);
        self.context.symbol_table.register_type_symbol_mapping(ty, sym);
        sym
    }

    /// `pkg.Module.Name` names a sub-type of `Module`; the sub-type registers
    /// under `pkg` alone, so retry the lookup with the module segment dropped.
    fn lookup_module_subtype(&self, path: &[String]) -> Option<crate::tast::SymbolId> {
        if path.len() < 3 {
            return None;
        }
        let module = &path[path.len() - 2];
        if !module.chars().next().map_or(false, |c| c.is_uppercase()) {
            return None;
        }
        let package: Vec<InternedString> = path[..path.len() - 2]
            .iter()
            .map(|p| self.context.string_interner.intern(p))
            .collect();
        let name = self.context.string_interner.intern(&path[path.len() - 1]);
        let qp = super::namespace::QualifiedPath::new(package, name);
        self.context.namespace_resolver.lookup_symbol(&qp)
    }

    /// Lower a using declaration
    pub(crate) fn lower_using(&mut self, using: &Using) -> LoweringResult<TypedUsing> {
        let module_path_str = using.path.join(".");
        let module_path = self.context.intern_string(&module_path_str);

        // Try to resolve the using module to a class symbol for static extension resolution
        // The module path is typically just the class name (e.g., "StringTools")
        // or a qualified path (e.g., "haxe.StringTools")
        let class_name = using
            .path
            .last()
            .map(|s| s.as_str())
            .unwrap_or(&module_path_str);
        let class_name_interned = self.context.intern_string(class_name);

        // First try to find via namespace resolver (handles qualified paths)
        let package_path: Vec<_> = using
            .path
            .iter()
            .take(using.path.len().saturating_sub(1))
            .map(|s| self.context.string_interner.intern(s))
            .collect();
        let qualified_path =
            super::namespace::QualifiedPath::new(package_path, class_name_interned);

        let class_symbol_id = if let Some(symbol_id) = self
            .context
            .namespace_resolver
            .lookup_symbol(&qualified_path)
        {
            // Found via namespace resolver
            Some(symbol_id)
        } else if let Some(class_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), class_name_interned)
        {
            // Found in global scope - but check if this symbol was actually lowered
            // If scope_id is ScopeId(0), it was only pre-registered but not lowered
            // In that case, search for a symbol with the same name that WAS lowered
            if class_symbol.scope_id == ScopeId::first() {
                // This symbol wasn't lowered - search for one that was
                let mut found_lowered = None;
                for sym in self
                    .context
                    .symbol_table
                    .symbols_of_kind(crate::tast::symbols::SymbolKind::Class)
                {
                    if sym.name == class_name_interned && sym.scope_id != ScopeId::first() {
                        found_lowered = Some(sym.id);
                        break;
                    }
                }
                if found_lowered.is_some() {
                    found_lowered
                } else {
                    // No lowered symbol found, use pre-registered one (will trigger loading)
                    Some(class_symbol.id)
                }
            } else {
                Some(class_symbol.id)
            }
        } else {
            None
        };

        if let Some(symbol_id) = class_symbol_id {
            // Found the class - register it for static extension resolution
            // Check if the class has been fully compiled (scope_id should not be ScopeId::first() or ScopeId(0))
            let needs_loading = if let Some(sym) = self.context.symbol_table.get_symbol(symbol_id) {
                // If scope_id is still the root scope (ScopeId(0)), the class was only pre-registered
                // and not actually compiled with its method bodies
                sym.scope_id == ScopeId::first()
            } else {
                true
            };

            if needs_loading {
                // Queue the module for loading - the compilation unit will load it
                self.pending_usings.push(module_path_str.clone());
            }

            self.using_modules.push((class_name_interned, symbol_id));
        }
        // Note: If class not found, static extensions will still work through the
        // "LAST RESORT" mechanism in hir_to_mir.rs which searches all stdlib classes

        Ok(TypedUsing {
            module_path,
            target_type: None, // TODO: Handle target type if specified
            source_location: self.context.create_location_from_span(using.span),
        })
    }
}
