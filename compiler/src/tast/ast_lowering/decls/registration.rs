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

    /// Register a symbol with package information
    fn register_symbol_with_package(&mut self, symbol_id: SymbolId, name: &str) {
        if let Some(package_id) = self.context.current_package {
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
                                .map(|&s| self
                                    .context
                                    .string_interner
                                    .get(s)
                                    .unwrap_or("<unknown>"))
                                .collect::<Vec<_>>()
                                .join("."),
                            name
                        )
                    };
                    symbol.qualified_name =
                        Some(self.context.string_interner.intern(&qualified_name));
                }
            }
        }
    }

    /// Pre-register type declarations in the symbol table (first pass)
    pub fn pre_register_declaration(
        &mut self,
        declaration: &TypeDeclaration,
    ) -> LoweringResult<()> {
        match declaration {
            TypeDeclaration::Class(class_decl) => {
                let class_name = self.context.intern_string(&class_decl.name);

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
                let interface_name = self.context.intern_string(&interface_decl.name);

                // Check if this interface already exists in the root scope
                if self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), interface_name)
                    .is_some()
                {
                    return Ok(());
                }

                let interface_symbol = self
                    .context
                    .symbol_table
                    .create_interface_in_scope(interface_name, ScopeId::first());

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

                // Add to root scope for global resolution
                self.context
                    .scope_tree
                    .get_scope_mut(ScopeId::first())
                    .expect("Root scope should exist")
                    .add_symbol(interface_symbol, interface_name);
            }
            TypeDeclaration::Enum(enum_decl) => {
                let enum_name = self.context.intern_string(&enum_decl.name);

                // Check if this enum already exists in the root scope
                if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), enum_name)
                {
                    let existing_id = existing.id;
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
                let abstract_name = self.context.intern_string(&abstract_decl.name);

                // Check if this abstract already exists in the root scope
                if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(ScopeId::first(), abstract_name)
                {
                    let existing_id = existing.id;
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
                            .create_abstract_type(existing_id, None, Vec::new());
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
        let class_symbol = match self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), class_name)
        {
            Some(entry) => entry.id,
            None => return Ok(()), // Not pre-registered, skip
        };

        // If class_fields already has entries for this class, skip (already registered)
        if self.class_fields.contains_key(&class_symbol) {
            return Ok(());
        }

        // Initialize the field list
        self.class_fields.insert(class_symbol, Vec::new());

        // Register each var/final/property field
        for field in &class_decl.fields {
            let (field_name, type_hint) = match &field.kind {
                parser::ClassFieldKind::Var {
                    name, type_hint, ..
                } => (name.clone(), type_hint.as_ref()),
                parser::ClassFieldKind::Final {
                    name, type_hint, ..
                } => (name.clone(), type_hint.as_ref()),
                parser::ClassFieldKind::Property {
                    name, type_hint, ..
                } => (name.clone(), type_hint.as_ref()),
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
                self.context.type_table.borrow().dynamic_type()
            };

            let interned_name = self.context.intern_string(&field_name);
            let field_symbol = self.context.symbol_table.create_variable(interned_name);

            // Set the field's type
            self.context
                .symbol_table
                .update_symbol_type(field_symbol, field_type);

            // Mark as field
            if let Some(sym) = self.context.symbol_table.get_symbol_mut(field_symbol) {
                sym.kind = crate::tast::SymbolKind::Field;
            }

            // Add to class_fields
            if let Some(field_list) = self.class_fields.get_mut(&class_symbol) {
                field_list.push((interned_name, field_symbol, is_static));
            }
        }

        Ok(())
    }

    /// Pre-register enum-abstract constants before any declaration body is
    /// lowered. Haxe exposes these fields both as `Color.Red` and, within the
    /// declaring module, as bare `Red`. A class may precede the enum abstract
    /// in the source, so registering the aliases from `lower_abstract_declaration`
    /// is too late for that class's methods.
    pub(crate) fn pre_register_enum_abstract_fields(
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
                .unwrap_or(underlying_type);
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
                .expect("enum abstract field map was initialized")
                .push((member_name, field_symbol, true));

            self.context
                .symbol_table
                .add_symbol_alias(field_symbol, ScopeId::first(), member_name);
            let root = self
                .context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist");
            if !root.has_symbol(member_name) {
                root.add_symbol(field_symbol, member_name);
            }
        }

        Ok(())
    }
}
