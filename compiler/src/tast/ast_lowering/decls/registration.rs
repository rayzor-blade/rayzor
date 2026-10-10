//! Pre-registration: the names a file declares before it is lowered.

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
    /// Pre-register all type declarations in a file (first pass only)
    /// This registers class/interface/enum/typedef/abstract names in the namespace
    /// without lowering their bodies. Used for multi-file compilation where all
    /// type names need to be available before any file is fully compiled.
    pub fn pre_register_file(&mut self, file: &HaxeFile) -> LoweringResult<()> {
        // Process package declaration to set up the namespace context
        self.context.current_package = None;
        if let Some(package) = &file.package {
            // Create or get package in namespace resolver
            let package_path: Vec<_> = package
                .path
                .iter()
                .map(|s| self.context.string_interner.intern(s))
                .collect();
            let package_id = self
                .context
                .namespace_resolver
                .get_or_create_package(package_path.clone());
            self.context.current_package = Some(package_id);
        }

        // Pre-register all type declarations
        for declaration in &file.declarations {
            if let Err(e) = self.pre_register_declaration(declaration) {
                self.collected_errors.push(e);
            }
        }

        // Reset package context for next file
        self.context.current_package = None;

        // Return any errors that occurred during pre-registration
        if !self.collected_errors.is_empty() {
            return Err(self.collected_errors.pop().unwrap());
        }

        Ok(())
    }

    /// Register top-level stdlib symbols (Math, Std, etc.) for implicit availability.
    ///
    /// In Haxe, these classes are always available without explicit imports.
    /// This method is called separately from load_standard_library() to support
    /// lazy stdlib loading where we want to skip parsing/processing stdlib files
    /// but still need these symbols to be resolvable.
    pub(crate) fn register_toplevel_stdlib_symbols(&mut self) {
        for type_name in TOPLEVEL_STDLIB_CLASSES {
            let interned_name = self.context.intern_string(type_name);

            // Check if already registered (avoid duplicates)
            if self
                .resolve_symbol_in_scope_hierarchy(interned_name)
                .is_some()
            {
                continue;
            }

            let builtin_symbol = self
                .context
                .symbol_table
                .create_class_in_scope(interned_name, ScopeId::first());

            // Update qualified name
            self.context.update_symbol_qualified_name(builtin_symbol);

            // Add to root scope for global resolution
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(builtin_symbol, interned_name);
        }

        // Register built-in global functions (trace)
        self.register_builtin_functions();
    }

    /// Register built-in global functions like trace()
    fn register_builtin_functions(&mut self) {
        let builtin_functions = [("trace", vec!["Dynamic"], "Void")];

        for (func_name, param_types, return_type) in builtin_functions {
            let func_name_interned = self.context.intern_string(func_name);

            // Check if already registered
            if self
                .resolve_symbol_in_scope_hierarchy(func_name_interned)
                .is_some()
            {
                continue;
            }

            // Create parameter types
            let mut param_type_ids = Vec::new();
            for param_type_name in param_types {
                let param_type_id = match param_type_name {
                    "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                    "Int" => self.context.type_table.borrow().int_type(),
                    "String" => self.context.type_table.borrow().string_type(),
                    "Float" => self.context.type_table.borrow().float_type(),
                    "Bool" => self.context.type_table.borrow().bool_type(),
                    "Void" => self.context.type_table.borrow().void_type(),
                    _ => self.context.type_table.borrow().dynamic_type(),
                };
                param_type_ids.push(param_type_id);
            }

            // Create return type
            let return_type_id = match return_type {
                "Dynamic" => self.context.type_table.borrow().dynamic_type(),
                "Int" => self.context.type_table.borrow().int_type(),
                "String" => self.context.type_table.borrow().string_type(),
                "Float" => self.context.type_table.borrow().float_type(),
                "Bool" => self.context.type_table.borrow().bool_type(),
                "Void" => self.context.type_table.borrow().void_type(),
                _ => self.context.type_table.borrow().dynamic_type(),
            };

            // Create function type
            let function_type_id = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(param_type_ids, return_type_id);

            // Create function symbol
            use crate::tast::{
                LifetimeId, Mutability, SourceLocation, Symbol, SymbolFlags, SymbolKind, Visibility,
            };

            let func_symbol_id = SymbolId::from_raw(self.context.symbol_table.len() as u32);
            let func_symbol = Symbol {
                id: func_symbol_id,
                name: func_name_interned,
                kind: SymbolKind::Function,
                type_id: function_type_id,
                scope_id: ScopeId::first(),
                lifetime_id: LifetimeId::invalid(),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                definition_location: SourceLocation::unknown(),
                is_used: false,
                is_exported: false,
                documentation: None,
                flags: SymbolFlags::NONE,
                package_id: None,
                qualified_name: None,
                native_name: None,
                frameworks: None,
                c_includes: None,
                c_sources: None,
                c_libs: None,
                js_import: None,
            };

            self.context.symbol_table.add_symbol(func_symbol);

            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(func_symbol_id, func_name_interned);
        }
    }

    /// Keep declarations in distinct packages from sharing a root name's symbol.
    pub(crate) fn root_slot_is_foreign_type(&self, name: InternedString) -> bool {
        let named_package = |package: Option<crate::tast::namespace::PackageId>| {
            package.filter(|&id| {
                self.context
                    .namespace_resolver
                    .get_package(id)
                    .is_some_and(|p| !p.full_path.is_empty())
            })
        };
        let pkg = named_package(self.context.current_package);
        self.context
            .symbol_table
            .lookup_symbol(ScopeId::first(), name)
            .is_some_and(|s| match s.kind {
                crate::tast::SymbolKind::Enum | crate::tast::SymbolKind::Abstract => {
                    named_package(s.package_id) != pkg
                }
                // Declarations have package identities; unresolved placeholders do not.
                crate::tast::SymbolKind::Class => {
                    let owner = named_package(s.package_id);
                    s.package_id.is_some() && owner != pkg
                }
                _ => false,
            })
    }

    /// The root slot for `name` holds a type of another named package. An
    /// unpackaged root symbol is a top-level placeholder the declaration
    /// claims, so it does not count.
    pub(crate) fn root_slot_is_other_package_type(&self, name: InternedString) -> bool {
        let Some(pkg) = self.context.current_package else {
            return false;
        };
        self.root_slot_is_foreign_type(name)
            && self
                .context
                .symbol_table
                .lookup_symbol(ScopeId::first(), name)
                .is_some_and(|s| s.package_id.is_some_and(|p| p != pkg))
    }

    /// The symbol of `kind` registered for `name` in the current package, if any.
    pub(crate) fn package_type_symbol(
        &self,
        name: InternedString,
        kind: crate::tast::SymbolKind,
    ) -> Option<SymbolId> {
        let pkg = self
            .context
            .current_package
            .unwrap_or(crate::tast::namespace::PackageId::root());
        let id = *self
            .context
            .namespace_resolver
            .get_package(pkg)?
            .symbols
            .get(&name)?;
        self.context
            .symbol_table
            .get_symbol(id)
            .filter(|s| s.kind == kind)
            .map(|s| s.id)
    }

    pub(crate) fn register_symbol_with_package(&mut self, symbol_id: SymbolId, name: &str) {
        let package_id = self
            .context
            .current_package
            .unwrap_or(crate::tast::namespace::PackageId::root());
        let interned_name = self.context.string_interner.intern(name);

        // Register symbol in namespace
        self.context
            .namespace_resolver
            .register_symbol(package_id, interned_name, symbol_id);

        // Update symbol with package info and qualified name
        if let Some(symbol) = self.context.symbol_table.get_symbol_mut(symbol_id) {
            symbol.package_id = Some(package_id);

            // Create qualified name
            if let Some(package) = self.context.namespace_resolver.get_package(package_id) {
                let qualified_name = if package.full_path.is_empty() {
                    name.to_string()
                } else {
                    format!(
                        "{}.{}",
                        package
                            .full_path
                            .iter()
                            .map(|&s| self.context.string_interner.get(s).unwrap_or("<unknown>"))
                            .collect::<Vec<_>>()
                            .join("."),
                        name
                    )
                };
                symbol.qualified_name = Some(self.context.string_interner.intern(&qualified_name));
            }
        }
    }

    /// Pre-register type declarations in the symbol table (first pass)
    /// The classes named by `@:using(..)` on a type: their static methods
    /// extend values of the type, as a module-level `using` would.
    fn record_type_usings(&mut self, type_name: InternedString, meta: &[parser::Metadata]) {
        fn last_segment(expr: &parser::Expr) -> Option<&str> {
            match &expr.kind {
                parser::ExprKind::Ident(name) => Some(name),
                parser::ExprKind::Field { field, .. } => Some(field),
                _ => None,
            }
        }
        let names: Vec<InternedString> = meta
            .iter()
            .filter(|m| m.name.trim_start_matches(':') == "using")
            .flat_map(|m| m.params.iter())
            .filter_map(last_segment)
            .map(str::to_string)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|name| self.context.intern_string(&name))
            .collect();
        if !names.is_empty() {
            self.type_usings.entry(type_name).or_default().extend(names);
        }
    }

    pub fn pre_register_declaration(
        &mut self,
        declaration: &TypeDeclaration,
    ) -> LoweringResult<()> {
        match declaration {
            TypeDeclaration::Class(class_decl) => {
                let class_name = self.context.intern_string(&class_decl.name);
                self.record_type_usings(class_name, &class_decl.meta);

                if self.root_slot_is_foreign_type(class_name) {
                    if self
                        .package_type_symbol(class_name, crate::tast::SymbolKind::Class)
                        .is_none()
                    {
                        let class_symbol = self
                            .context
                            .symbol_table
                            .create_class_in_scope(class_name, ScopeId::first());
                        self.register_symbol_with_package(class_symbol, &class_decl.name);
                        let class_type = self.context.type_table.borrow_mut().create_type(
                            crate::tast::core::TypeKind::Class {
                                symbol_id: class_symbol,
                                type_args: Vec::new(),
                            },
                        );
                        self.context
                            .symbol_table
                            .update_symbol_type(class_symbol, class_type);
                        self.context
                            .symbol_table
                            .register_type_symbol_mapping(class_type, class_symbol);
                    }
                    return Ok(());
                }

                // Check if this class already exists in the root scope (from a previous compilation)
                // If so, skip pre-registration to avoid creating duplicate symbols
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), class_name)
                    .is_some()
                {
                    // Class already pre-registered, skip
                    return Ok(());
                }

                let class_symbol = self
                    .context
                    .symbol_table
                    .create_class_in_scope(class_name, ScopeId::first());

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(class_symbol, &class_decl.name);

                // Create the corresponding type for this class
                let class_type = self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Class {
                        symbol_id: class_symbol,
                        type_args: Vec::new(), // Will be updated during full lowering
                    },
                );

                // Set the symbol's type_id to link it to the type
                self.context
                    .symbol_table
                    .update_symbol_type(class_symbol, class_type);

                // Register the type-to-symbol mapping so we can look up symbols from types
                self.context
                    .symbol_table
                    .register_type_symbol_mapping(class_type, class_symbol);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(class_symbol, class_name);
            }
            TypeDeclaration::Interface(interface_decl) => {
                let type_name = self.context.intern_string(&interface_decl.name);
                self.record_type_usings(type_name, &interface_decl.meta);
                let interface_name = self.context.intern_string(&interface_decl.name);
                let root_binding = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), interface_name)
                    .map(|s| s.id);

                // A same-named root symbol is this interface only when it is one
                // (an import placeholder minted as Interface). A typedef alias to
                // it, like root `IMap` for `haxe.Constraints.IMap`, keeps the root
                // slot and the interface registers under its package alone.
                let existing = self
                    .packaged_symbol(interface_name)
                    .and_then(|id| self.context.symbol_table.get_symbol(id))
                    .or_else(|| {
                        self.context
                            .symbol_table
                            .lookup_symbol(ScopeId::first(), interface_name)
                    })
                    .map(|s| (s.id, s.kind.clone(), s.package_id.is_none()));
                if std::env::var_os("RAYZOR_SYM_DEBUG").is_some() {
                    eprintln!(
                        "[sym] pre-register iface {} existing={existing:?}",
                        interface_decl.name
                    );
                }
                let root_taken =
                    match existing {
                        Some((id, crate::tast::SymbolKind::Interface, unpackaged))
                            if unpackaged
                                || self.context.symbol_table.get_symbol(id).is_some_and(|s| {
                                    s.package_id == self.context.current_package
                                }) =>
                        {
                            if unpackaged {
                                self.register_symbol_with_package(id, &interface_decl.name);
                            }
                            return Ok(());
                        }
                        Some(_) => true,
                        None => false,
                    };

                let interface_symbol = self
                    .context
                    .symbol_table
                    .create_interface_in_scope(interface_name, ScopeId::first());
                if root_taken {
                    if let Some(root) = root_binding {
                        self.context.symbol_table.remap_symbol_in_scope(
                            ScopeId::first(),
                            interface_name,
                            root,
                        );
                    }
                }

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(interface_symbol, &interface_decl.name);

                // Create the corresponding type for this interface
                let interface_type = self.context.type_table.borrow_mut().create_type(
                    crate::tast::core::TypeKind::Interface {
                        symbol_id: interface_symbol,
                        type_args: Vec::new(), // Will be updated during full lowering
                    },
                );

                // Set the symbol's type_id to link it to the type
                self.context
                    .symbol_table
                    .update_symbol_type(interface_symbol, interface_type);

                // Register the type-to-symbol mapping so we can look up symbols from types
                self.context
                    .symbol_table
                    .register_type_symbol_mapping(interface_type, interface_symbol);

                if !root_taken {
                    self.context
                        .scope_tree
                        .get_scope_mut(ScopeId::first())
                        .expect("Root scope should exist")
                        .add_symbol(interface_symbol, interface_name);
                }
            }
            TypeDeclaration::Enum(enum_decl) => {
                let type_name = self.context.intern_string(&enum_decl.name);
                self.record_type_usings(type_name, &enum_decl.meta);
                let enum_name = self.context.intern_string(&enum_decl.name);

                let existing_id = self
                    .package_type_symbol(enum_name, crate::tast::SymbolKind::Enum)
                    .or_else(|| {
                        if self.root_slot_is_foreign_type(enum_name) {
                            return None;
                        }
                        self.context
                            .symbol_table
                            .lookup_symbol(ScopeId::first(), enum_name)
                            .map(|symbol| symbol.id)
                    });
                if let Some(existing_id) = existing_id {
                    // The symbol may have been created as `SymbolKind::Class` by
                    // earlier import resolution (e.g. a sibling file imported
                    // `pkg.X.Y` before `Y`'s declaration was lowered, so the
                    // namespace resolver registered `Y` as a generic Class
                    // placeholder). Fix it to Enum now that we know the actual
                    // declaration kind — otherwise the second compile of the
                    // declaring file's class methods would resolve `Y` as a
                    // Class through this stale entry, returning `Ptr(Void)`
                    // from MIR `convert_type` instead of the boxed-enum I64
                    // discriminant the first compile produced. The two compiles
                    // would then have different signatures for the same method
                    // and cross-file callers would dispatch to the wrong one.
                    // Mirrors the Class-to-Abstract fixup a few cases below.
                    let needs_fix = self
                        .context
                        .symbol_table
                        .get_symbol(existing_id)
                        .map(|s| s.kind == crate::tast::SymbolKind::Class)
                        .unwrap_or(false);
                    if needs_fix {
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(existing_id) {
                            sym.kind = crate::tast::SymbolKind::Enum;
                        }
                        let enum_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_enum_type(existing_id, Vec::new());
                        self.context
                            .symbol_table
                            .update_symbol_type(existing_id, enum_type);
                        self.context
                            .symbol_table
                            .register_type_symbol_mapping(enum_type, existing_id);

                        // Register variants under the corrected enum symbol.
                        for variant in &enum_decl.constructors {
                            let variant_name = self.context.intern_string(&variant.name);
                            // Look at any same-named symbol already in root scope.
                            let existing_info = self
                                .context
                                .symbol_table
                                .lookup_symbol(ScopeId::first(), variant_name)
                                .map(|e| (e.id, e.kind));
                            // Reuse only a same-parent EnumVariant (avoid duplicates).
                            let is_same_parent_variant = match existing_info {
                                Some((eid, crate::tast::symbols::SymbolKind::EnumVariant)) => {
                                    self.context
                                        .symbol_table
                                        .find_parent_enum_for_constructor(eid)
                                        == Some(existing_id)
                                }
                                _ => false,
                            };
                            if !is_same_parent_variant {
                                // ALWAYS create the variant symbol so it is findable via
                                // all_symbols() and linked to its parent enum — even when its
                                // name collides with a builtin TYPE (e.g. variant `Bool` vs the
                                // builtin `Bool` Abstract). Previously this branch skipped on
                                // any collision, so the variant was never created at all and a
                                // `Bool(x)` constructor call silently resolved to the type,
                                // producing a value-less return (W0020 -> SIGILL in nue's
                                // GGUFReader.readValue). Only insert into the scope NAME-map
                                // when the slot is free, so the builtin isn't clobbered; the
                                // call-site collision fix redirects `Bool(args)` to the variant.
                                let variant_symbol =
                                    self.context.symbol_table.create_enum_variant_in_scope(
                                        variant_name,
                                        ScopeId::first(),
                                        existing_id,
                                    );
                                if existing_info.is_none() {
                                    self.context
                                        .scope_tree
                                        .get_scope_mut(ScopeId::first())
                                        .expect("Root scope should exist")
                                        .add_symbol(variant_symbol, variant_name);
                                }
                            }
                        }
                    }
                    return Ok(());
                }

                let enum_symbol = self
                    .context
                    .symbol_table
                    .create_enum_in_scope(enum_name, ScopeId::first());

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(enum_symbol, &enum_decl.name);

                // Create the Enum type now so anything that resolves this enum
                // during the first pass (e.g. a class field declared before the
                // enum body has been lowered, or a cross-file user resolving
                // `pkg.File.EnumName` via import) gets a real TypeId instead of
                // a placeholder. Classes/Interfaces above already do this; the
                // omission for enums made declaration order load-bearing — if
                // the enum appeared *after* the class that referenced it in
                // the same file, downstream method-dispatch lowering would
                // silently elide calls returning the enum type.
                let enum_type = self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_enum_type(enum_symbol, Vec::new());
                self.context
                    .symbol_table
                    .update_symbol_type(enum_symbol, enum_type);
                self.context
                    .symbol_table
                    .register_type_symbol_mapping(enum_type, enum_symbol);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(enum_symbol, enum_name);

                // IMPORTANT: Also pre-register enum variants so they can be resolved
                // during pattern matching even before the enum is fully lowered
                for variant in &enum_decl.constructors {
                    let variant_name = self.context.intern_string(&variant.name);
                    // Is the bare name already taken (e.g. by the builtin `Bool`
                    // Abstract type, or another enum's same-named arm)?
                    let slot_taken = self
                        .context
                        .symbol_table
                        .lookup_symbol(ScopeId::first(), variant_name)
                        .is_some();
                    let variant_symbol = self.context.symbol_table.create_enum_variant_in_scope(
                        variant_name,
                        ScopeId::first(),
                        enum_symbol,
                    );

                    // Standard Haxe keeps enum constructors in the ENUM's namespace, so
                    // an arm named like a type (`MetaValue.Bool` vs builtin `Bool`) is
                    // legal. Only insert the arm into the global root scope name-map when
                    // the slot is FREE — otherwise we'd clobber the builtin type (breaking
                    // `var x:Bool` / the arm's own `Bool` param type) which produced a
                    // value-less return / W0020 SIGILL. The arm stays findable via
                    // all_symbols() + parent-linked; a bare `Bool(x)` constructor call is
                    // redirected to it at the call site (lower_call_expression collision fix).
                    if !slot_taken {
                        self.context
                            .scope_tree
                            .get_scope_mut(ScopeId::first())
                            .expect("Root scope should exist")
                            .add_symbol(variant_symbol, variant_name);
                    }
                }
            }
            TypeDeclaration::Typedef(typedef_decl) => {
                let typedef_name = self.context.intern_string(&typedef_decl.name);
                self.record_type_usings(typedef_name, &typedef_decl.meta);

                // Check if this typedef already exists in the root scope
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), typedef_name)
                    .is_some()
                {
                    return Ok(());
                }

                let typedef_symbol = self
                    .context
                    .symbol_table
                    .create_class_in_scope(typedef_name, ScopeId::first()); // Reuse class for typedefs

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(typedef_symbol, &typedef_decl.name);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(typedef_symbol, typedef_name);
            }
            TypeDeclaration::Abstract(abstract_decl) => {
                let type_name = self.context.intern_string(&abstract_decl.name);
                self.record_type_usings(type_name, &abstract_decl.meta);
                let abstract_name = self.context.intern_string(&abstract_decl.name);

                // The underlying type, recorded now rather than when the
                // abstract's own declaration is reached. A class declared ABOVE
                // the abstract resolves against whatever the type table holds at
                // that moment, and an abstract with no underlying recorded falls
                // back to a 32-bit slot: a Float loses its fraction and a String
                // is truncated to half a pointer, silently.
                //
                // Best effort. An underlying naming a type not yet registered --
                // or a type parameter, which is not in scope this early -- stays
                // None, exactly as before.
                // The abstract's own type parameters have to be in scope for
                // this: a generic abstract writes its underlying AS one of them
                // (`abstract Val<T>(T)`), so without them `T` does not resolve,
                // the underlying stays None, and every instantiation is a 32-bit
                // slot again. Type resolution runs before the abstract's own
                // declaration is lowered, so recording it there is too late.
                let mut tp_map: std::collections::BTreeMap<InternedString, TypeId> =
                    std::collections::BTreeMap::new();
                for tp in &abstract_decl.type_params {
                    let tp_name = self.context.intern_string(&tp.name);
                    let tp_symbol = self
                        .context
                        .symbol_table
                        .create_type_parameter(tp_name, Vec::new());
                    // No constraints here: this exists only so the underlying's
                    // reference to the parameter RESOLVES. The constrained form is
                    // built properly when the declaration itself is lowered.
                    let tp_type = self.context.type_table.borrow_mut().create_type_parameter(
                        tp_symbol,
                        Vec::new(),
                        tp.variance.into(),
                    );
                    tp_map.insert(tp_name, tp_type);
                }
                let has_tps = !tp_map.is_empty();
                if has_tps {
                    self.context.push_type_parameters(tp_map);
                }
                let pre_underlying = abstract_decl
                    .underlying
                    .as_ref()
                    .and_then(|u| self.lower_type(u).ok());
                if has_tps {
                    self.context.pop_type_parameters();
                }

                // The root slot holds another package's type (`haxe.Int64`
                // beside a private `Int64`): register under this package only.
                if self.root_slot_is_other_package_type(abstract_name) {
                    if self
                        .package_type_symbol(abstract_name, crate::tast::SymbolKind::Abstract)
                        .is_none()
                    {
                        let abstract_symbol = self
                            .context
                            .symbol_table
                            .create_abstract_in_scope(abstract_name, ScopeId::first());
                        let abstract_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_abstract_type(abstract_symbol, pre_underlying, Vec::new());
                        self.context
                            .symbol_table
                            .update_symbol_type(abstract_symbol, abstract_type);
                        self.context
                            .symbol_table
                            .register_type_symbol_mapping(abstract_type, abstract_symbol);
                        self.register_symbol_with_package(abstract_symbol, &abstract_decl.name);
                    }
                    return Ok(());
                }

                // Check if this abstract already exists in the root scope
                if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), abstract_name)
                {
                    let existing_id = existing.id;
                    // A top-level placeholder this declaration claims takes its
                    // package, so a same-named type elsewhere sees it as foreign.
                    if existing.package_id.is_none() {
                        self.register_symbol_with_package(existing_id, &abstract_decl.name);
                    }
                    // The symbol may have been created as SymbolKind::Class by import resolution
                    // (which doesn't know the declaration kind). Fix it to Abstract now that we
                    // know the actual declaration type. We must fix BOTH:
                    // 1. The symbol kind (Class -> Abstract)
                    // 2. The type in the type table (create a new Abstract type, since types are immutable)
                    let needs_fix = self
                        .context
                        .symbol_table
                        .get_symbol(existing_id)
                        .map(|s| s.kind == crate::tast::SymbolKind::Class)
                        .unwrap_or(false);
                    if needs_fix {
                        if let Some(sym) = self.context.symbol_table.get_symbol_mut(existing_id) {
                            sym.kind = crate::tast::SymbolKind::Abstract;
                        }
                        // Create a proper Abstract type to replace the Class type
                        let abstract_type = self
                            .context
                            .type_table
                            .borrow_mut()
                            .create_abstract_type(existing_id, pre_underlying, Vec::new());
                        self.context
                            .symbol_table
                            .update_symbol_type(existing_id, abstract_type);
                        self.context
                            .symbol_table
                            .register_type_symbol_mapping(abstract_type, existing_id);
                    }
                    return Ok(());
                }

                let abstract_symbol = self
                    .context
                    .symbol_table
                    .create_abstract_in_scope(abstract_name, ScopeId::first());

                // create_abstract_in_scope leaves the symbol's type invalid, so
                // give it one here with the underlying already attached.
                let abstract_type = self.context.type_table.borrow_mut().create_abstract_type(
                    abstract_symbol,
                    pre_underlying,
                    Vec::new(),
                );
                self.context
                    .symbol_table
                    .update_symbol_type(abstract_symbol, abstract_type);
                self.context
                    .symbol_table
                    .register_type_symbol_mapping(abstract_type, abstract_symbol);

                // Register symbol with package information (also sets qualified name)
                self.register_symbol_with_package(abstract_symbol, &abstract_decl.name);

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(abstract_symbol, abstract_name);
            }
            TypeDeclaration::Conditional(_) => {
                // Skip conditional compilation blocks in pre-registration
            }
        }
        Ok(())
    }

    /// Recover function field signatures before callers in earlier classes are lowered.
    fn function_field_signature(&mut self, initializer: &parser::Expr) -> Option<TypeId> {
        let parser::ExprKind::Function(function) = &initializer.kind else {
            return None;
        };
        if !function.type_params.is_empty() {
            return None;
        }
        fn returned_constant(expr: &parser::Expr) -> Option<&parser::Expr> {
            match &expr.kind {
                parser::ExprKind::Return(Some(value)) => Some(value),
                parser::ExprKind::Paren(inner) => returned_constant(inner),
                parser::ExprKind::Block(elements) => match elements.as_slice() {
                    [parser::BlockElement::Expr(expr)] => returned_constant(expr),
                    _ => None,
                },
                _ => None,
            }
        }
        let constant = function
            .body
            .as_deref()
            .and_then(returned_constant)
            .and_then(|value| self.literal_type(value))
            .filter(|ty| {
                matches!(
                    self.context.type_table.borrow().get(*ty).map(|ty| &ty.kind),
                    Some(
                        crate::tast::TypeKind::Int
                            | crate::tast::TypeKind::Float
                            | crate::tast::TypeKind::Bool
                            | crate::tast::TypeKind::String
                    )
                )
            });
        let result = if let Some(annotation) = &function.return_type {
            self.lower_type(annotation).ok()?
        } else {
            constant?
        };
        let mut params = Vec::with_capacity(function.params.len());
        for parameter in &function.params {
            let ty = if let Some(annotation) = &parameter.type_hint {
                let ty = self.lower_type(annotation).ok()?;
                self.optional_param_type(parameter, ty)
            } else {
                let default = parameter
                    .default_value
                    .as_deref()
                    .and_then(|value| self.literal_type(value));
                if default.is_none() && constant.is_none() {
                    return None;
                }
                default.unwrap_or_else(|| self.context.type_table.borrow().dynamic_type())
            };
            params.push(ty);
        }
        Some(
            self.context
                .type_table
                .borrow_mut()
                .create_function_type(params, result),
        )
    }

    /// Pre-register class fields for forward reference resolution.
    /// This runs after pre_register_declaration (which creates class type entries)
    /// but before full lowering, so that field access on forward-referenced classes
    /// can resolve field names and types correctly.
    pub(crate) fn pre_register_class_fields(
        &mut self,
        class_decl: &parser::ClassDecl,
    ) -> LoweringResult<()> {
        let class_name = self.context.intern_string(&class_decl.name);

        // Look up the pre-registered class symbol
        let class_symbol = if self.root_slot_is_foreign_type(class_name) {
            match self.package_type_symbol(class_name, crate::tast::SymbolKind::Class) {
                Some(symbol) => symbol,
                None => return Ok(()),
            }
        } else {
            match self
                .context
                .symbol_table
                .lookup_symbol(ScopeId::first(), class_name)
            {
                Some(entry) => entry.id,
                None => return Ok(()), // Not pre-registered, skip
            }
        };

        // If class_fields already has entries for this class, skip (already registered)
        if self.class_fields.contains_key(&class_symbol) {
            return Ok(());
        }

        // Field annotations and parent arguments use the class's parameters,
        // including when a caller precedes this declaration.
        let type_param_map = self.function_type_parameter_map(&class_decl.type_params)?;
        let has_type_params = !type_param_map.is_empty();
        if has_type_params {
            let ordered = class_decl
                .type_params
                .iter()
                .filter_map(|p| {
                    let name = self.context.string_interner.get_id(&p.name)?;
                    type_param_map.get(&name).copied()
                })
                .collect::<Vec<_>>();
            self.context
                .symbol_table
                .set_class_type_params(class_symbol, ordered.clone());
            self.class_type_params.insert(class_symbol, ordered);
            self.context.push_type_parameters(type_param_map);
        }
        let parent = class_decl
            .extends
            .as_ref()
            .and_then(|parent| self.lower_type(parent).ok());
        self.context
            .symbol_table
            .set_class_super_type(class_symbol, parent);

        // Initialize the field list
        self.class_fields.insert(class_symbol, Vec::new());

        // Register each var/final/property field
        for field in &class_decl.fields {
            let (field_name, type_hint, initializer) = match &field.kind {
                parser::ClassFieldKind::Var {
                    name,
                    type_hint,
                    expr,
                } => (name.clone(), type_hint.as_ref(), expr.as_ref()),
                parser::ClassFieldKind::Final {
                    name,
                    type_hint,
                    expr,
                } => (name.clone(), type_hint.as_ref(), expr.as_ref()),
                parser::ClassFieldKind::Property {
                    name,
                    type_hint,
                    expr,
                    ..
                } => (name.clone(), type_hint.as_ref(), expr.as_ref()),
                parser::ClassFieldKind::Function(_) => continue, // Skip methods
            };

            let is_static = field
                .modifiers
                .iter()
                .any(|m| matches!(m, parser::Modifier::Static));

            // Resolve the field type from the type hint
            let field_type = if let Some(th) = type_hint {
                self.lower_type(th)
                    .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
            } else {
                initializer
                    .and_then(|expr| self.function_field_signature(expr))
                    .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type())
            };

            let interned_name = self.context.intern_string(&field_name);
            let field_symbol = self.context.symbol_table.create_variable(interned_name);

            // Set the field's type
            self.context
                .symbol_table
                .update_symbol_type(field_symbol, field_type);

            // Mark as field
            let visibility = if field.access.is_none()
                && class_decl.modifiers.contains(&parser::Modifier::Extern)
            {
                crate::tast::Visibility::Public
            } else {
                self.lower_access(&field.access)
            };
            if let Some(sym) = self.context.symbol_table.get_symbol_mut(field_symbol) {
                sym.visibility = visibility;
                sym.kind = crate::tast::SymbolKind::Field;
                if is_static {
                    sym.flags = sym.flags.union(crate::tast::SymbolFlags::STATIC);
                }
            }

            // Add to class_fields
            if let Some(field_list) = self.class_fields.get_mut(&class_symbol) {
                field_list.push((interned_name, field_symbol, is_static));
            }
        }

        if has_type_params {
            self.context.pop_type_parameters();
        }
        Ok(())
    }

    /// Pre-register enum-abstract constants before any declaration body is
    /// lowered. Haxe exposes these fields both as `Color.Red` and, within the
    /// declaring module, as bare `Red`. A class may precede the enum abstract
    /// in the source, so registering the aliases from `lower_abstract_declaration`
    /// is too late for that class's methods.
    /// Record an abstract's implicit casts before any body in the file
    /// lowers. `Context.unify` consults `abstract_casts` during deferred
    /// macro re-expansion, which happens while some EARLIER declaration's
    /// body is being lowered — waiting for the abstract's own declaration to
    /// lower leaves a later-in-file abstract castless at that moment
    /// (Issue10728's private-types-after-class layout). Best effort: an
    /// annotation that does not lower yet is skipped, and the full
    /// declaration lowering re-records the entry authoritatively.
    pub(crate) fn pre_register_abstract_casts(&mut self, abstract_decl: &parser::AbstractDecl) {
        let abstract_name = self.context.intern_string(&abstract_decl.name);
        let Some(abstract_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), abstract_name)
            .map(|entry| entry.id)
        else {
            return;
        };
        let mut from_types: Vec<TypeId> = Vec::new();
        let mut to_types: Vec<TypeId> = Vec::new();
        for ty in &abstract_decl.from {
            if let Ok(id) = self.lower_type(ty) {
                from_types.push(id);
            }
        }
        for ty in &abstract_decl.to {
            if let Ok(id) = self.lower_type(ty) {
                to_types.push(id);
            }
        }
        for field in &abstract_decl.fields {
            let parser::ClassFieldKind::Function(func) = &field.kind else {
                continue;
            };
            let has = |tag: &str| field.meta.iter().any(|m| m.name == tag);
            if has(":from") || has("from") {
                if let Some(param) = func.params.first() {
                    if let Some(annotation) = &param.type_hint {
                        if let Ok(id) = self.lower_type(annotation) {
                            from_types.push(id);
                        }
                    }
                }
            }
            if has(":to") || has("to") {
                if let Some(annotation) = &func.return_type {
                    if let Ok(id) = self.lower_type(annotation) {
                        to_types.push(id);
                    }
                } else if func.name == "toString" {
                    to_types.push(self.context.type_table.borrow().string_type());
                }
            }
        }
        self.abstract_casts
            .insert(abstract_symbol, (from_types, to_types));
    }

    pub(crate) fn pre_register_abstract_fields(
        &mut self,
        abstract_decl: &parser::AbstractDecl,
    ) -> LoweringResult<()> {
        let abstract_name = self.context.intern_string(&abstract_decl.name);
        let Some(abstract_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), abstract_name)
            .map(|entry| entry.id)
        else {
            return Ok(());
        };

        let underlying_type = abstract_decl
            .underlying
            .as_ref()
            .and_then(|ty| self.lower_type(ty).ok())
            .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type());

        self.class_fields.entry(abstract_symbol).or_default();
        for field in &abstract_decl.fields {
            if !abstract_decl.is_enum_abstract
                && !field.modifiers.contains(&parser::Modifier::Static)
            {
                continue;
            }
            let (name, type_hint) = match &field.kind {
                parser::ClassFieldKind::Var {
                    name, type_hint, ..
                }
                | parser::ClassFieldKind::Final {
                    name, type_hint, ..
                }
                | parser::ClassFieldKind::Property {
                    name, type_hint, ..
                } => (name, type_hint.as_ref()),
                parser::ClassFieldKind::Function(_) => continue,
            };
            let member_name = self.context.intern_string(name);
            if self
                .class_fields
                .get(&abstract_symbol)
                .is_some_and(|fields| fields.iter().any(|(n, _, _)| *n == member_name))
            {
                continue;
            }

            let field_type = type_hint
                .and_then(|ty| self.lower_type(ty).ok())
                .unwrap_or_else(|| {
                    if abstract_decl.is_enum_abstract {
                        underlying_type
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    }
                });
            let field_symbol = self.context.symbol_table.create_variable(member_name);
            self.context
                .symbol_table
                .update_symbol_type(field_symbol, field_type);
            if let Some(symbol) = self.context.symbol_table.get_symbol_mut(field_symbol) {
                symbol.kind = crate::tast::SymbolKind::Field;
                symbol.flags = symbol
                    .flags
                    .union(crate::tast::symbols::SymbolFlags::STATIC);
                let qualified = format!("{}.{}", abstract_decl.name, name);
                symbol.qualified_name = Some(self.context.string_interner.intern(&qualified));
            }
            self.class_fields
                .get_mut(&abstract_symbol)
                .expect("abstract field map was initialized")
                .push((member_name, field_symbol, true));

            if abstract_decl.is_enum_abstract {
                self.context.symbol_table.add_symbol_alias(
                    field_symbol,
                    ScopeId::first(),
                    member_name,
                );
                let root = self
                    .context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist");
                if !root.has_symbol(member_name) {
                    root.add_symbol(field_symbol, member_name);
                }
            }
        }

        Ok(())
    }
}
