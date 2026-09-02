//! Declarations: dispatch, modifiers and access.

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
    pub fn lower_file(&mut self, file: &HaxeFile) -> LoweringResult<TypedFile> {
        // Optimizer barrier

        // Create TypedFile with the shared interner from the pipeline
        let mut typed_file = TypedFile::new(Rc::clone(&self.context.string_interner_rc));

        // Populate file metadata from the input HaxeFile so downstream
        // passes (ownership diagnostics, the RAYZOR_DEBUG_E0382 dump,
        // cross-file warning attribution) can identify which file the
        // typed_file came from. Without this the metadata stays at the
        // FileMetadata::default() empty-string value and every E0382
        // dump prints `typed_file=` blank.
        typed_file.metadata.file_path = file.filename.clone();
        let file_name = file
            .filename
            .rsplit('/')
            .next()
            .unwrap_or(&file.filename)
            .to_string();
        typed_file.metadata.file_name = Some(self.context.string_interner.intern(&file_name));

        // Process package declaration
        if let Some(package) = &file.package {
            typed_file.metadata.package_name = Some(package.path.join("."));

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

            // Create package scope with full qualified name
            let package_name_str = package.path.join(".");
            let package_name_interned = self.context.string_interner.intern(&package_name_str);
            let package_scope = self
                .context
                .enter_named_scope(ScopeKind::Package, package_name_interned);
        }

        // Set file metadata
        typed_file.metadata.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Set file name if available
        if !typed_file.metadata.file_path.is_empty() {
            if let Some(file_name) = typed_file.metadata.file_path.split('/').last() {
                typed_file.metadata.file_name =
                    Some(self.context.string_interner.intern(file_name));
            }
        }

        // Load standard library types from source files
        // (skip if CompilationUnit is handling stdlib loading)
        if !self.skip_stdlib_loading {
            self.load_standard_library()?;
        } else {
            // Even when skipping full stdlib loading, we MUST register top-level stdlib
            // symbols (Math, Std, etc.) so they can be resolved without explicit imports.
            // This is required for Haxe semantics where these classes are implicitly available.
            self.register_toplevel_stdlib_symbols();
        }

        // Process import.hx files in the current directory hierarchy
        if let Err(e) = self.process_import_hx_files(&file) {
            self.collected_errors.push(e);
        }

        // Process imports - collect errors but continue
        for import in &file.imports {
            match self.lower_import(import) {
                Ok(typed_import) => typed_file.imports.push(typed_import),
                Err(e) => self.collected_errors.push(e),
            }
        }

        // Process using statements - collect errors but continue
        for using in &file.using {
            match self.lower_using(using) {
                Ok(typed_using) => typed_file.using_statements.push(typed_using),
                Err(e) => self.collected_errors.push(e),
            }
        }

        // Process module-level fields - collect errors but continue
        for module_field in &file.module_fields {
            match self.lower_module_field(module_field) {
                Ok(typed_field) => typed_file.module_fields.push(typed_field),
                Err(e) => self.collected_errors.push(e),
            }
        }

        // First pass: Pre-register all type declarations in the symbol table
        // Skip this if CompilationUnit has already pre-registered all files
        if !self.skip_pre_registration {
            for declaration in &file.declarations {
                if let Err(e) = self.pre_register_declaration(declaration) {
                    self.collected_errors.push(e);
                }
            }
        }

        // Pass 1.5: Pre-register fields for ALL classes and enum abstracts before
        // method bodies are lowered. This enables forward references (e.g., NBody
        // referencing Body fields when Body is declared later in the file) and
        // bare enum-abstract constants used by a class declared above the abstract.
        // Without this, field resolution falls back to placeholders or reports an
        // unresolved constant purely because of declaration order.
        for declaration in &file.declarations {
            match declaration {
                TypeDeclaration::Class(class_decl) => {
                    if let Err(e) = self.pre_register_class_fields(class_decl) {
                        self.collected_errors.push(e);
                    }
                }
                TypeDeclaration::Abstract(abstract_decl) => {
                    if abstract_decl.is_enum_abstract {
                        if let Err(e) = self.pre_register_enum_abstract_fields(abstract_decl) {
                            self.collected_errors.push(e);
                        }
                    }
                    self.pre_register_abstract_casts(abstract_decl);
                }
                _ => {}
            }
        }

        // Second pass: Process declarations with full type resolution
        for declaration in &file.declarations {
            match self.lower_declaration(declaration) {
                Ok(typed_decl) => match typed_decl {
                    TypedDeclaration::Function(func) => typed_file.functions.push(func),
                    TypedDeclaration::Class(class) => typed_file.classes.push(class),
                    TypedDeclaration::Interface(interface) => typed_file.interfaces.push(interface),
                    TypedDeclaration::Enum(enum_decl) => typed_file.enums.push(enum_decl),
                    TypedDeclaration::TypeAlias(alias) => typed_file.type_aliases.push(alias),
                    TypedDeclaration::Abstract(abstract_decl) => {
                        typed_file.abstracts.push(abstract_decl);
                    }
                },
                Err(e) => self.context.add_error(e),
            }
        }

        // Resolve any deferred type references (second pass)
        if let Err(e) = self.resolve_deferred_types() {
            self.collected_errors.push(e);
        }

        // Combine all errors from both context and collected_errors
        let mut all_errors = Vec::new();
        all_errors.extend(self.context.errors.clone());
        all_errors.extend(self.collected_errors.clone());

        // Check for errors - if any, return the first one but all are collected
        if !all_errors.is_empty() {
            // Store all errors in context for pipeline to access
            self.context.errors = all_errors.clone();
            return Err(all_errors.into_iter().next().unwrap());
        }

        Ok(typed_file)
    }

    /// Lower a module field
    fn lower_module_field(
        &mut self,
        module_field: &ModuleField,
    ) -> LoweringResult<TypedModuleField> {
        let field_name = match &module_field.kind {
            parser::ModuleFieldKind::Var { name, .. } => name.clone(),
            parser::ModuleFieldKind::Final { name, .. } => name.clone(),
            parser::ModuleFieldKind::Function(func) => func.name.clone(),
        };

        let interned_name = self.context.intern_string(&field_name);
        let field_symbol = self.context.symbol_table.create_variable(interned_name);
        let mut field_flags = self.extract_metadata_flags(&module_field.meta, field_symbol);
        for modifier in &module_field.modifiers {
            use crate::tast::symbols::SymbolFlags;
            field_flags = field_flags.union(match modifier {
                parser::haxe_ast::Modifier::Static => SymbolFlags::STATIC,
                parser::haxe_ast::Modifier::Inline => SymbolFlags::INLINE,
                parser::haxe_ast::Modifier::Macro => SymbolFlags::MACRO,
                parser::haxe_ast::Modifier::Dynamic => SymbolFlags::DYNAMIC,
                parser::haxe_ast::Modifier::Override => SymbolFlags::OVERRIDE,
                parser::haxe_ast::Modifier::Final => SymbolFlags::FINAL,
                parser::haxe_ast::Modifier::Extern => SymbolFlags::EXTERN,
            });
        }
        if !field_flags.is_empty() {
            self.context
                .symbol_table
                .add_symbol_flags(field_symbol, field_flags);
        }

        let kind = match &module_field.kind {
            parser::ModuleFieldKind::Var {
                name: _,
                type_hint,
                expr,
            } => {
                let field_type = if let Some(type_hint) = type_hint {
                    self.lower_type(type_hint)?
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };

                let initializer = if let Some(expr) = expr {
                    Some(self.lower_expression(expr)?)
                } else {
                    None
                };

                TypedModuleFieldKind::Var {
                    field_type,
                    initializer,
                    mutability: crate::tast::Mutability::Mutable,
                }
            }
            parser::ModuleFieldKind::Final {
                name: _,
                type_hint,
                expr,
            } => {
                let field_type = if let Some(type_hint) = type_hint {
                    self.lower_type(type_hint)?
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };

                let initializer = if let Some(expr) = expr {
                    Some(self.lower_expression(expr)?)
                } else {
                    None
                };

                TypedModuleFieldKind::Final {
                    field_type,
                    initializer,
                }
            }
            parser::ModuleFieldKind::Function(func) => TypedModuleFieldKind::Function(
                self.lower_function_object(func, &module_field.meta, &module_field.modifiers)?,
            ),
        };

        Ok(TypedModuleField {
            symbol_id: field_symbol,
            name: interned_name,
            kind,
            visibility: self.lower_access(&module_field.access),
            source_location: self.context.create_location_from_span(module_field.span),
        })
    }

    /// Lower a declaration
    fn lower_declaration(
        &mut self,
        declaration: &TypeDeclaration,
    ) -> LoweringResult<TypedDeclaration> {
        match declaration {
            TypeDeclaration::Class(class_decl) => self.lower_class_declaration(class_decl),
            TypeDeclaration::Interface(interface_decl) => {
                self.lower_interface_declaration(interface_decl)
            }
            TypeDeclaration::Enum(enum_decl) => self.lower_enum_declaration(enum_decl),
            TypeDeclaration::Typedef(typedef_decl) => self.lower_typedef_declaration(typedef_decl),
            TypeDeclaration::Abstract(abstract_decl) => {
                self.lower_abstract_declaration(abstract_decl)
            }
            TypeDeclaration::Conditional(conditional) => {
                // Process conditional compilation by evaluating compile-time conditions
                // This requires compile-time flag evaluation which should be done in preprocessing
                return Err(LoweringError::IncompleteImplementation {
                    feature:
                        "Conditional compilation blocks should be expanded during preprocessing"
                            .to_string(),
                    location: self.context.create_location_from_span(conditional.span),
                });
            }
        }
    }

    /// Lower a class declaration

    /// Lower an interface declaration
    pub(crate) fn lower_interface_declaration(
        &mut self,
        interface_decl: &InterfaceDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let interface_name = self.context.intern_string(&interface_decl.name);

        // Look up the existing symbol that was created during pre-registration
        let interface_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), interface_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_interface_in_scope(interface_name, ScopeId::first());
            // Update qualified name (full path including class hierarchy)
            self.context.update_symbol_qualified_name(new_symbol);
            // Add interface to the root scope so it can be resolved for forward references
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, interface_name);
            new_symbol
        };

        // Enter interface scope with name
        let interface_scope = self
            .context
            .enter_named_scope(ScopeKind::Interface, interface_name);

        // Process type parameters
        let type_params = self.lower_type_parameters(&interface_decl.type_params)?;
        let mut type_param_map: BTreeMap<InternedString, TypeId> = BTreeMap::new();
        for tp in &type_params {
            let interned_name = tp.name;
            // Convert constraints to ConstraintKind for symbol table
            let constraint_kinds = tp
                .constraints
                .iter()
                .map(|_| {
                    crate::tast::type_checker::ConstraintKind::Implements {
                        interface_type: TypeId::invalid(), // Placeholder, will be resolved later
                    }
                })
                .collect();
            let symbol_id = self
                .context
                .symbol_table
                .create_type_parameter(interned_name, constraint_kinds);
            let type_id = self.context.type_table.borrow_mut().create_type_parameter(
                symbol_id,
                tp.constraints.clone(),
                tp.variance.into(),
            );
            type_param_map.insert(tp.name, type_id);
        }
        self.context.push_type_parameters(type_param_map);

        // Process extends clause
        let extends = interface_decl
            .extends
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;

        // Process fields - separate method signatures from other fields
        let mut method_signatures = Vec::with_capacity(interface_decl.fields.len());
        for field in &interface_decl.fields {
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    // Interface methods are just signatures, not full implementations
                    match self.lower_function_signature(field, func) {
                        Ok(method_sig) => method_signatures.push(method_sig),
                        Err(e) => self.context.add_error(e),
                    }
                }
                _ => {
                    // Interfaces can have property signatures too
                    // Interfaces can have property signatures and constants
                    // These are handled separately in the interface specification
                }
            }
        }

        // Process modifiers
        let modifiers = self.lower_modifiers(&interface_decl.modifiers)?;

        self.context.pop_type_parameters();
        self.context.exit_scope();

        // Mirror concurrency derives onto the interface SYMBOL (same mechanism
        // as classes) so an interface-typed value reads as Send/Sync at a
        // `Thread.spawn` capture site, cross-file. The trait checker reads the
        // symbol flags when the type isn't in the local file's list.
        {
            let mut sym_flags = crate::tast::symbols::SymbolFlags::NONE;
            for name in interface_decl.get_derive_traits() {
                match crate::tast::DerivedTrait::from_str(&name) {
                    Some(crate::tast::DerivedTrait::Send) => {
                        sym_flags.insert(crate::tast::symbols::SymbolFlags::DERIVE_SEND);
                    }
                    Some(crate::tast::DerivedTrait::Sync) => {
                        sym_flags.insert(crate::tast::symbols::SymbolFlags::DERIVE_SYNC);
                    }
                    _ => {}
                }
            }
            if !sym_flags.is_empty() {
                self.context
                    .symbol_table
                    .add_symbol_flags(interface_symbol, sym_flags);
            }
        }

        let typed_interface = TypedInterface {
            symbol_id: interface_symbol,
            name: interface_name,
            extends,
            methods: method_signatures,
            type_parameters: type_params,
            visibility: self.lower_access(&interface_decl.access),
            source_location: self.context.create_location_from_span(interface_decl.span),
        };

        Ok(TypedDeclaration::Interface(typed_interface))
    }

    /// Lower a typedef declaration
    pub(crate) fn lower_typedef_declaration(
        &mut self,
        typedef_decl: &TypedefDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let typedef_name = self.context.intern_string(&typedef_decl.name);

        // Look up existing symbol or create a new one
        let typedef_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), typedef_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_type_alias_in_scope(typedef_name, ScopeId::first());
            // Update qualified name (full path including package/module)
            self.context.update_symbol_qualified_name(new_symbol);
            // Add typedef to the root scope so it can be resolved
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, typedef_name);
            new_symbol
        };

        // Process type parameters FIRST and push them onto the stack
        let type_params = self.lower_type_parameters(&typedef_decl.type_params)?;

        // Build type parameter map for the stack
        let mut type_param_map = BTreeMap::new();
        for type_param in &type_params {
            // Type parameter already has a symbol_id from lower_type_parameters
            // Create a TypeId for this parameter
            let variance = match type_param.variance {
                TypeVariance::Covariant => Variance::Covariant,
                TypeVariance::Contravariant => Variance::Contravariant,
                TypeVariance::Invariant => Variance::Invariant,
            };
            let type_var = self.context.type_table.borrow_mut().create_type_parameter(
                type_param.symbol_id,
                type_param.constraints.clone(),
                variance,
            );
            type_param_map.insert(type_param.name, type_var);
        }

        // Push type parameters onto stack so they're available when lowering the typedef body
        self.context.push_type_parameters(type_param_map);

        // Now process target type (can reference type parameters)
        let mut target_type = self.lower_type(&typedef_decl.type_def)?;

        // Pop type parameters from stack
        self.context.pop_type_parameters();

        // If the target resolved to a Placeholder, check if the existing symbol (from
        // a prior compilation, e.g. rayzor.Bytes extern class) already has a valid Class type.
        // This handles `typedef Bytes = rayzor.Bytes;` where rayzor.Bytes was compiled first
        // and the symbol "Bytes" already exists as a Class in root scope.
        {
            let is_placeholder = {
                let type_table = self.context.type_table.borrow();
                matches!(
                    type_table.get(target_type).map(|ti| &ti.kind),
                    Some(crate::tast::core::TypeKind::Placeholder { .. })
                )
            };
            if is_placeholder {
                // The target couldn't be resolved by lower_type. Check if the existing symbol
                // (which we're about to overwrite as TypeAlias) already has the right type.
                if let Some(existing) = self.context.symbol_table.get_symbol(typedef_symbol) {
                    if existing.kind == crate::tast::SymbolKind::Class
                        && existing.type_id.is_valid()
                    {
                        // The existing symbol IS the target class. Use its type directly.
                        target_type = existing.type_id;
                    }
                }
            }
        }

        // Create the TypeAlias type in the type table and set it on the symbol
        let type_arg_ids: Vec<TypeId> = type_params
            .iter()
            .map(|tp| {
                self.context.type_table.borrow_mut().create_type_parameter(
                    tp.symbol_id,
                    tp.constraints.clone(),
                    tp.variance.into(),
                )
            })
            .collect();

        let typedef_type = self.context.type_table.borrow_mut().create_type(
            crate::tast::core::TypeKind::TypeAlias {
                symbol_id: typedef_symbol,
                target_type,
                type_args: type_arg_ids,
            },
        );

        // Set the type on the symbol so it can be resolved later
        self.context
            .symbol_table
            .update_symbol_type(typedef_symbol, typedef_type);

        let typed_typedef = TypedTypeAlias {
            symbol_id: typedef_symbol,
            name: typedef_name,
            target_type,
            type_parameters: type_params,
            visibility: self.lower_access(&typedef_decl.access),
            source_location: self.context.create_location_from_span(typedef_decl.span),
        };

        Ok(TypedDeclaration::TypeAlias(typed_typedef))
    }

    /// Lower modifiers and extract static, override, etc.
    fn lower_modifiers(&mut self, modifiers: &[Modifier]) -> LoweringResult<ModifierInfo> {
        let mut modifier_info = ModifierInfo::default();

        for modifier in modifiers {
            match modifier {
                parser::Modifier::Static => modifier_info.is_static = true,
                parser::Modifier::Override => modifier_info.is_override = true,
                parser::Modifier::Inline => modifier_info.is_inline = true,
                parser::Modifier::Dynamic => modifier_info.is_dynamic = true,
                parser::Modifier::Macro => modifier_info.is_macro = true,
                parser::Modifier::Final => modifier_info.is_final = true,
                parser::Modifier::Extern => modifier_info.is_extern = true,
            }
        }

        Ok(modifier_info)
    }

    /// Lower access modifiers (separate from other modifiers)
    fn lower_access(&mut self, access: &Option<parser::Access>) -> Visibility {
        match access {
            Some(parser::Access::Public) => Visibility::Public,
            Some(parser::Access::Private) => Visibility::Private,
            None => Visibility::Internal, // Default visibility
        }
    }
}

mod abstracts;
mod classes;
mod enums;
mod fields;
mod registration;
