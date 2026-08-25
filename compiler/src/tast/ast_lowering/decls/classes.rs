//! Class declarations, inheritance and field seeding.

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
    /// Get the current class symbol if we're in a class context
    fn get_current_class_symbol(&self) -> Option<SymbolId> {
        self.context.class_context_stack.last().copied()
    }

    /// Seed class_fields from previously compiled files so that cross-file
    /// static field access (e.g., BufferUsage.VERTEX from GraphicsTypes.hx) works.
    pub fn seed_class_fields(
        &mut self,
        fields: &BTreeMap<SymbolId, Vec<(InternedString, SymbolId, bool)>>,
    ) {
        for (class_sym, field_list) in fields {
            self.class_fields
                .entry(*class_sym)
                .or_insert_with(|| field_list.clone());
        }
    }

    /// Export class_fields for accumulation across file compilations.
    pub fn export_class_fields(
        &self,
    ) -> &BTreeMap<SymbolId, Vec<(InternedString, SymbolId, bool)>> {
        &self.class_fields
    }

    pub(crate) fn lower_class_declaration(
        &mut self,
        class_decl: &ClassDecl,
    ) -> LoweringResult<TypedDeclaration> {
        let class_name = self.context.intern_string(&class_decl.name);

        // Look up the existing symbol that was created during pre-registration
        let class_symbol = if let Some(existing_symbol) = self
            .context
            .symbol_table
            .lookup_symbol(ScopeId::first(), class_name)
        {
            existing_symbol.id
        } else {
            let new_symbol = self
                .context
                .symbol_table
                .create_class_in_scope(class_name, ScopeId::first());
            // Update qualified name (full path including class hierarchy)
            self.context.update_symbol_qualified_name(new_symbol);
            // Add class to the root scope so it can be resolved for forward references
            self.context
                .scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(new_symbol, class_name);
            new_symbol
        };

        // Enter class scope with name. On a re-entry (this file is being
        // compiled a second time after a prior attempt failed and was
        // queued for retry), reuse the class scope that the first attempt
        // already populated — class scopes are looked up by scope_id, and
        // a fresh per-attempt scope means every retry starts with no
        // visible members. Combined with the cross-file `stdlib_function_map`
        // not being populated until success, that caused dependent files
        // (Caller files compiled in the same pass) to bind to nothing and
        // emit silent-elide MIR. Sticky-scope keeps the SymbolIds stable so
        // a later retry can fill in the missing function_map entries by
        // the same key. Mirrors the class-symbol reuse a few lines above.
        let existing_scope = self
            .context
            .symbol_table
            .get_symbol(class_symbol)
            .map(|s| s.scope_id)
            .filter(|sid| {
                *sid != ScopeId::first()
                    && *sid != ScopeId::invalid()
                    && self
                        .context
                        .scope_tree
                        .get_scope(*sid)
                        .map(|sc| sc.kind == ScopeKind::Class)
                        .unwrap_or(false)
            });
        let (class_scope, reusing_scope) = match existing_scope {
            Some(sid) => {
                self.context.current_scope = sid;
                (sid, true)
            }
            None => (
                self.context.enter_named_scope(ScopeKind::Class, class_name),
                false,
            ),
        };

        // Note: The class symbol remains in the parent scope, while its members are in class_scope
        // This is correct because the class name should be accessible from outside

        // Update the class symbol's scope_id to point to the class scope where its methods are registered
        // This is crucial for static extension resolution to find methods by scope lookup
        if let Some(sym) = self.context.symbol_table.get_symbol_mut(class_symbol) {
            sym.scope_id = class_scope;
        }

        // Push class onto context stack for method resolution
        self.context.class_context_stack.push(class_symbol);

        // Initialize method and field tracking for this class. On a reused
        // class scope, keep any prior entries — the per-method lookup
        // below will find them and skip re-creating symbols.
        if !reusing_scope {
            self.class_methods.insert(class_symbol, Vec::new());
            self.class_fields.insert(class_symbol, Vec::new());
        } else {
            self.class_methods
                .entry(class_symbol)
                .or_insert_with(Vec::new);
            self.class_fields
                .entry(class_symbol)
                .or_insert_with(Vec::new);
        }

        // Extract metadata flags from @:generic, @:final, @:native, etc.
        let mut symbol_flags = self.extract_metadata_flags(&class_decl.meta, class_symbol);
        // Also check for modifiers (final, extern, etc)
        for modifier in &class_decl.modifiers {
            match modifier {
                parser::haxe_ast::Modifier::Final => {
                    symbol_flags = symbol_flags.union(crate::tast::symbols::SymbolFlags::FINAL);
                }
                parser::haxe_ast::Modifier::Extern => {
                    symbol_flags = symbol_flags.union(crate::tast::symbols::SymbolFlags::EXTERN);
                }
                _ => {}
            }
        }
        // Apply flags to the class symbol
        if !symbol_flags.is_empty() {
            self.context
                .symbol_table
                .add_symbol_flags(class_symbol, symbol_flags);
        }

        // Process type parameters
        let type_params = self.lower_type_parameters(&class_decl.type_params)?;
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
        self.context.push_type_parameters(type_param_map.clone());

        // Store ordered type parameter TypeIds for generic type inference.
        // Mirror to the shared SymbolTable so cross-file lookups
        // (e.g. user file calling `new Arc(value)`) can recover the
        // type-param TypeIds populated when Arc.hx was lowered.
        if !type_param_map.is_empty() {
            let ordered_tp_ids: Vec<TypeId> = type_params
                .iter()
                .filter_map(|tp| type_param_map.get(&tp.name).copied())
                .collect();
            self.context
                .symbol_table
                .set_class_type_params(class_symbol, ordered_tp_ids.clone());
            self.class_type_params.insert(class_symbol, ordered_tp_ids);
        }

        // Process extends clause
        let extends = if let Some(extends_type) = &class_decl.extends {
            Some(self.lower_type(extends_type)?)
        } else {
            None
        };

        // Copy parent FIELDS and METHODS before processing child's members
        // This ensures:
        // 1. Field inheritance works (constructor can access parent fields)
        // 2. Method inheritance works (child methods can call parent methods)
        // 3. Method overriding works (child methods will replace parent methods during processing)
        if let Some(parent_type_id) = extends {
            if let Some(parent_symbol) = self.resolve_type_to_class_symbol(parent_type_id) {
                self.class_parents.insert(class_symbol, parent_symbol);
            }
            self.copy_parent_fields(parent_type_id, class_symbol);
            self.copy_parent_methods(parent_type_id, class_symbol);
        }

        // Process implements clause
        let implements = class_decl
            .implements
            .iter()
            .map(|t| self.lower_type(t))
            .collect::<Result<Vec<_>, _>>()?;

        // PRE-REGISTER ALL METHODS before lowering any method bodies.
        // This is critical for intra-class method calls: when main() calls iterate(),
        // iterate must already be registered in the class scope, even if it's defined later.
        // We also pre-compute function types from signatures to enable forward references.
        for field in &class_decl.fields {
            if let ClassFieldKind::Function(func) = &field.kind {
                let method_name = self.context.intern_string(&func.name);
                let is_static = field
                    .modifiers
                    .iter()
                    .any(|m| matches!(m, parser::haxe_ast::Modifier::Static));

                // Get or create the method symbol
                let method_symbol = if let Some(existing) = self
                    .context
                    .symbol_table
                    .lookup_symbol(class_scope, method_name)
                {
                    existing.id
                } else {
                    // Create the function symbol in the class scope
                    let sym = self
                        .context
                        .symbol_table
                        .create_function_in_scope(method_name, class_scope);

                    // Add to the class scope so it can be resolved during method body lowering
                    if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                        scope.add_symbol(sym, method_name);
                    }

                    // Mark as static if applicable (needed for resolution)
                    if is_static {
                        self.context
                            .symbol_table
                            .add_symbol_flags(sym, crate::tast::symbols::SymbolFlags::STATIC);
                    }

                    // Set qualified_name NOW, at pre-registration, not only when
                    // the body is lowered later (`lower_function_from_field`
                    // does call `update_symbol_qualified_name`, but only for
                    // the method it's actively lowering — this pre-registration
                    // symbol is what cross-module CALL SITES resolve against
                    // first). Without this, two same-named static methods in
                    // different classes (e.g. `Linear.fromQuant` vs
                    // `Embedding.fromQuant`) both carry `qualified_name=None`
                    // when a caller in a THIRD module resolves them, so the
                    // HIR-to-MIR fallback (`resolve_method_function_id` /
                    // `stdlib_function_name_map`, both keyed by bare name when
                    // qualified_name is absent) collides on the bare name
                    // `fromQuant` — whichever module's real body was merged in
                    // LAST silently wins for BOTH call sites, so a caller of
                    // one class's method invokes the OTHER class's method
                    // with mismatched arguments (confirmed: `Linear.fromQuant`
                    // calls resolved to `Embedding.fromQuant`'s compiled body,
                    // producing garbage/misaligned fields on the returned
                    // object — silent corruption, not a crash at the call
                    // site itself).
                    self.context.update_symbol_qualified_name(sym);

                    sym
                };

                // Pre-compute function type from AST signature for forward reference resolution
                let param_types: Vec<TypeId> = func
                    .params
                    .iter()
                    .map(|p| {
                        if let Some(ref type_hint) = p.type_hint {
                            self.lower_type(type_hint)
                                .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
                        } else {
                            self.context.type_table.borrow().dynamic_type()
                        }
                    })
                    .collect();
                let return_type = if let Some(ref ret_type) = func.return_type {
                    self.lower_type(ret_type)
                        .unwrap_or_else(|_| self.context.type_table.borrow().dynamic_type())
                } else {
                    // An un-annotated return is only inferred when the body is
                    // lowered, which happens after this. A caller written ABOVE
                    // the method therefore resolves it as Dynamic, and Dynamic
                    // carries no field layout -- so a returned anonymous
                    // structure reads its fields back as null, and the same
                    // source says different things depending on the order the
                    // two methods appear in. Recover the shape here for the
                    // case that can be read straight off the syntax.
                    self.anonymous_return_type_from_ast(func)
                        .unwrap_or_else(|| self.context.type_table.borrow().dynamic_type())
                };
                let function_type = self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(param_types, return_type);
                self.context
                    .symbol_table
                    .update_symbol_type(method_symbol, function_type);

                // Add to class_methods for forward reference resolution in lower_call_expression
                if func.name != "new" {
                    if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                        if let Some(existing_idx) = methods_list
                            .iter()
                            .position(|(name, _, _)| *name == method_name)
                        {
                            // Override parent method entry with child's symbol
                            // (critical for constructor bodies that call overridden methods)
                            methods_list[existing_idx] = (method_name, method_symbol, is_static);
                        } else {
                            methods_list.push((method_name, method_symbol, is_static));
                        }
                    }
                } else {
                    // Store constructor symbol for generic type inference,
                    // mirroring to SymbolTable for cross-file lookups.
                    self.context
                        .symbol_table
                        .set_class_constructor(class_symbol, method_symbol);
                    self.class_constructor_symbols
                        .insert(class_symbol, method_symbol);
                }
            }
        }

        // Process fields, methods, and constructors separately
        let mut fields = Vec::with_capacity(class_decl.fields.len());
        let mut methods = Vec::with_capacity(class_decl.fields.len()); // Initially allocate for all fields
        let mut constructors = Vec::with_capacity(2); // Most classes have 0-2 constructors

        for (field_idx, field) in class_decl.fields.iter().enumerate() {
            match &field.kind {
                ClassFieldKind::Function(func) => {
                    // Handle functions as methods or constructors
                    match self.lower_function_from_field(field, func) {
                        Ok(typed_function) => {
                            if func.name == "new" {
                                constructors.push(typed_function);
                            } else {
                                // Track method name and symbol for resolution
                                let method_name = self.context.intern_string(&func.name);
                                let method_symbol = typed_function.symbol_id;
                                if let Some(methods_list) =
                                    self.class_methods.get_mut(&class_symbol)
                                {
                                    // Check if a method with this name already exists (from parent)
                                    // If so, replace it (method overriding)
                                    if let Some(existing_idx) = methods_list
                                        .iter()
                                        .position(|(name, _, _)| *name == method_name)
                                    {
                                        // Override parent method
                                        methods_list[existing_idx] =
                                            (method_name, method_symbol, typed_function.is_static);
                                    } else {
                                        // New method, add to list
                                        methods_list.push((
                                            method_name,
                                            method_symbol,
                                            typed_function.is_static,
                                        ));
                                    }
                                }
                                methods.push(typed_function);
                            }
                        }
                        Err(e) => self.context.add_error(e),
                    }
                }
                _ => {
                    // Handle regular fields (var, final, property)
                    match self.lower_field(field) {
                        Ok(typed_field) => fields.push(typed_field),
                        Err(e) => self.context.add_error(e),
                    }
                }
            }
        }

        // Note: Parent fields and methods were already copied before processing members
        // This ensures:
        // 1. Field/method inheritance works (child can access parent members)
        // 2. Method overriding works (child methods replace parent methods during processing)

        // Process modifiers
        let modifiers = self.lower_modifiers(&class_decl.modifiers)?;

        self.context.pop_type_parameters();

        // Pop class from context stack
        self.context.class_context_stack.pop();

        self.context.exit_scope();

        // Auto-inject synthetic cdef() static method for @:cstruct classes
        if symbol_flags.is_cstruct() {
            let cdef_name = self.context.intern_string("cdef");
            // Create a symbol for the synthetic method
            let cdef_symbol = self
                .context
                .symbol_table
                .create_function_in_scope(cdef_name, class_scope);
            self.context
                .symbol_table
                .add_symbol_flags(cdef_symbol, crate::tast::symbols::SymbolFlags::STATIC);
            if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                scope.add_symbol(cdef_symbol, cdef_name);
            }
            // Set return type to String
            let string_type = self.context.type_table.borrow().string_type();
            let fn_type = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(vec![], string_type);
            self.context
                .symbol_table
                .update_symbol_type(cdef_symbol, fn_type);

            // Register in class_methods
            if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                methods_list.push((cdef_name, cdef_symbol, true));
            }

            // Add synthetic TypedFunction (empty body — intercepted at MIR level)
            methods.push(crate::tast::node::TypedFunction {
                symbol_id: cdef_symbol,
                name: cdef_name,
                parameters: vec![],
                return_type: string_type,
                body: vec![],
                visibility: crate::tast::symbols::Visibility::Public,
                effects: crate::tast::node::FunctionEffects {
                    can_throw: false,
                    async_kind: crate::tast::node::AsyncKind::Sync,
                    is_pure: true,
                    is_inline: true,
                    exception_types: vec![],
                    memory_effects: crate::tast::node::MemoryEffects::default(),
                    resource_effects: crate::tast::node::ResourceEffects::default(),
                },
                type_parameters: vec![],
                is_static: true,
                source_location: crate::tast::symbols::SourceLocation {
                    file_id: 0,
                    line: 0,
                    column: 0,
                    byte_offset: 0,
                },
                metadata: crate::tast::node::FunctionMetadata {
                    complexity_score: 0,
                    statement_count: 0,
                    is_recursive: false,
                    call_count: 0,
                    is_override: false,
                    overload_signatures: vec![],
                    operator_metadata: vec![],
                    is_array_access: false,
                    is_from_conversion: false,
                    is_to_conversion: false,
                    memory_annotations: vec![],
                },
            });
        }

        // Auto-inject synthetic gpuDef(), gpuSize(), gpuAlignment() for @:gpuStruct classes
        if symbol_flags.is_gpu_struct() {
            let string_type = self.context.type_table.borrow().string_type();
            let int_type = self.context.type_table.borrow().int_type();

            // Helper to create a synthetic static method with empty body
            let synthetic_names = [
                ("gpuDef", string_type),
                ("gpuSize", int_type),
                ("gpuAlignment", int_type),
                ("gpuVertexLayout", string_type),
            ];
            for (name_str, ret_type) in &synthetic_names {
                let method_name = self.context.intern_string(name_str);
                let method_symbol = self
                    .context
                    .symbol_table
                    .create_function_in_scope(method_name, class_scope);
                self.context
                    .symbol_table
                    .add_symbol_flags(method_symbol, crate::tast::symbols::SymbolFlags::STATIC);
                if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                    scope.add_symbol(method_symbol, method_name);
                }
                let fn_type = self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(vec![], *ret_type);
                self.context
                    .symbol_table
                    .update_symbol_type(method_symbol, fn_type);

                if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                    methods_list.push((method_name, method_symbol, true));
                }

                methods.push(crate::tast::node::TypedFunction {
                    symbol_id: method_symbol,
                    name: method_name,
                    parameters: vec![],
                    return_type: *ret_type,
                    body: vec![],
                    visibility: crate::tast::symbols::Visibility::Public,
                    effects: crate::tast::node::FunctionEffects {
                        can_throw: false,
                        async_kind: crate::tast::node::AsyncKind::Sync,
                        is_pure: true,
                        is_inline: true,
                        exception_types: vec![],
                        memory_effects: crate::tast::node::MemoryEffects::default(),
                        resource_effects: crate::tast::node::ResourceEffects::default(),
                    },
                    type_parameters: vec![],
                    is_static: true,
                    source_location: crate::tast::symbols::SourceLocation {
                        file_id: 0,
                        line: 0,
                        column: 0,
                        byte_offset: 0,
                    },
                    metadata: crate::tast::node::FunctionMetadata {
                        complexity_score: 0,
                        statement_count: 0,
                        is_recursive: false,
                        call_count: 0,
                        is_override: false,
                        overload_signatures: vec![],
                        operator_metadata: vec![],
                        is_array_access: false,
                        is_from_conversion: false,
                        is_to_conversion: false,
                        memory_annotations: vec![],
                    },
                });
            }
        }

        // Auto-inject synthetic wgsl() for @:shader classes
        if symbol_flags.is_shader() {
            let string_type = self.context.type_table.borrow().string_type();
            let method_name = self.context.intern_string("wgsl");
            let method_symbol = self
                .context
                .symbol_table
                .create_function_in_scope(method_name, class_scope);
            self.context
                .symbol_table
                .add_symbol_flags(method_symbol, crate::tast::symbols::SymbolFlags::STATIC);
            if let Some(scope) = self.context.scope_tree.get_scope_mut(class_scope) {
                scope.add_symbol(method_symbol, method_name);
            }
            let fn_type = self
                .context
                .type_table
                .borrow_mut()
                .create_function_type(vec![], string_type);
            self.context
                .symbol_table
                .update_symbol_type(method_symbol, fn_type);

            if let Some(methods_list) = self.class_methods.get_mut(&class_symbol) {
                methods_list.push((method_name, method_symbol, true));
            }

            methods.push(crate::tast::node::TypedFunction {
                symbol_id: method_symbol,
                name: method_name,
                parameters: vec![],
                return_type: string_type,
                body: vec![],
                visibility: crate::tast::symbols::Visibility::Public,
                effects: Default::default(),
                type_parameters: vec![],
                is_static: true,
                source_location: crate::tast::SourceLocation {
                    file_id: 0,
                    line: 0,
                    column: 0,
                    byte_offset: 0,
                },
                metadata: crate::tast::node::FunctionMetadata {
                    complexity_score: 0,
                    statement_count: 0,
                    is_recursive: false,
                    call_count: 0,
                    is_override: false,
                    overload_signatures: vec![],
                    operator_metadata: vec![],
                    is_array_access: false,
                    is_from_conversion: false,
                    is_to_conversion: false,
                    memory_annotations: vec![],
                },
            });
        }

        // Extract memory safety annotations from metadata
        let memory_annotations = self.extract_memory_annotations(&class_decl.meta);

        // Extract derived traits from @:derive metadata
        let mut derived_traits = self.extract_derived_traits(class_decl);

        // Create typed class first (needed for validation)
        // Extract @:debugFormat("pattern") metadata
        let debug_format = class_decl
            .meta
            .iter()
            .find(|m| m.name == "debugFormat")
            .and_then(|m| m.params.first())
            .and_then(|expr| {
                if let parser::haxe_ast::ExprKind::String(s) = &expr.kind {
                    Some(s.clone())
                } else {
                    None
                }
            });

        // `@:keep` on a class covers everything the class declares. Reachability
        // is decided per function and never consults the owning class, so the
        // flag has to reach each member symbol to survive dead-code elimination.
        if symbol_flags.is_keep() {
            for member in methods.iter().chain(constructors.iter()) {
                self.context
                    .symbol_table
                    .add_symbol_flags(member.symbol_id, crate::tast::symbols::SymbolFlags::KEEP);
            }
        }

        let typed_class = TypedClass {
            symbol_id: class_symbol,
            name: class_name,
            super_class: extends,
            interfaces: implements,
            fields: fields.clone(),
            methods: methods,
            constructors: constructors,
            type_parameters: type_params,
            visibility: self.lower_access(&class_decl.access),
            source_location: self.context.create_location_from_span(class_decl.span),
            memory_annotations,
            derived_traits: derived_traits.clone(),
            debug_format,
        };

        // Validate derived traits against field types
        self.validate_derived_traits(&typed_class, &mut derived_traits, &class_decl.name);

        // Update derived_traits after validation (may have been modified)
        let mut typed_class = typed_class;
        typed_class.derived_traits = derived_traits;

        // Mirror the concurrency derives onto the class SYMBOL. The TypedClass
        // only lives in its own file's `classes` list, but the send/sync
        // validator runs per-file and looks types up cross-file through the
        // shared symbol table — so a `@:derive([Send])` extern (e.g.
        // sys.net.SocketOutput) must carry the fact on its symbol to read as
        // Send at a `Thread.spawn` capture site in another module.
        {
            let mut sym_flags = crate::tast::symbols::SymbolFlags::NONE;
            if typed_class
                .derived_traits
                .iter()
                .any(|t| matches!(t, crate::tast::DerivedTrait::Send))
            {
                sym_flags.insert(crate::tast::symbols::SymbolFlags::DERIVE_SEND);
            }
            if typed_class
                .derived_traits
                .iter()
                .any(|t| matches!(t, crate::tast::DerivedTrait::Sync))
            {
                sym_flags.insert(crate::tast::symbols::SymbolFlags::DERIVE_SYNC);
            }
            if !sym_flags.is_empty() {
                self.context
                    .symbol_table
                    .add_symbol_flags(class_symbol, sym_flags);
            }
        }

        // Synthesize hashCode():Int method for classes that derive Hash
        if typed_class
            .derived_traits
            .iter()
            .any(|t| matches!(t, crate::tast::DerivedTrait::Hash))
        {
            let hash_code_name = self.context.intern_string("hashCode");
            let has_hashcode = typed_class.methods.iter().any(|m| m.name == hash_code_name);
            if !has_hashcode {
                let int_type = self.context.type_table.borrow().int_type();
                let func_symbol_id = SymbolId::from_raw(self.context.symbol_table.len() as u32);
                let func_symbol = Symbol {
                    id: func_symbol_id,
                    name: hash_code_name,
                    kind: SymbolKind::Function,
                    type_id: int_type,
                    scope_id: ScopeId::first(),
                    lifetime_id: LifetimeId::invalid(),
                    visibility: Visibility::Public,
                    mutability: crate::tast::symbols::Mutability::Immutable,
                    definition_location: SourceLocation::unknown(),
                    is_used: true,
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

                // Stub body: return 0 (replaced at MIR level with actual hash computation)
                let return_expr = TypedExpression {
                    expr_type: int_type,
                    kind: TypedExpressionKind::Literal {
                        value: LiteralValue::Int(0),
                    },
                    usage: VariableUsage::Copy,
                    lifetime_id: crate::tast::LifetimeId::default(),
                    source_location: SourceLocation::default(),
                    metadata: ExpressionMetadata::default(),
                };

                typed_class.methods.push(TypedFunction {
                    symbol_id: func_symbol_id,
                    name: hash_code_name,
                    parameters: vec![],
                    return_type: int_type,
                    body: vec![TypedStatement::Return {
                        value: Some(return_expr),
                        source_location: SourceLocation::default(),
                    }],
                    visibility: Visibility::Public,
                    effects: crate::tast::node::FunctionEffects {
                        can_throw: false,
                        async_kind: AsyncKind::Sync,
                        is_pure: true,
                        is_inline: false,
                        exception_types: vec![],
                        memory_effects: crate::tast::node::MemoryEffects::default(),
                        resource_effects: ResourceEffects::default(),
                    },
                    type_parameters: vec![],
                    is_static: false,
                    source_location: SourceLocation::default(),
                    metadata: FunctionMetadata::default(),
                });
            }
        }

        Ok(TypedDeclaration::Class(typed_class))
    }

    /// Lower a parameter
    /// Types for parameters the source left unannotated, recovered from the
    /// field they are stored into.
    ///
    /// `function new(left, right, item) { this.left = left; ... }` is ordinary
    /// Haxe: the compiler is expected to unify each parameter with the field it
    /// feeds. Without that every one of them is Dynamic, and a Dynamic argument
    /// is boxed at the call -- one heap allocation per field per object, none
    /// of which the escape analysis can see. On a tree builder the boxes
    /// outweigh the objects.
    ///
    /// Only a direct `this.f = p` counts, and only when `p` reaches exactly one
    /// field. A parameter feeding two differently typed fields has no single
    /// answer, and anything less direct is left to the Dynamic path rather than
    /// guessed at.
    pub(crate) fn param_types_from_field_stores(
        &mut self,
        func: &Function,
    ) -> BTreeMap<InternedString, TypeId> {
        let unannotated: std::collections::BTreeSet<&str> = func
            .params
            .iter()
            .filter(|p| p.type_hint.is_none())
            .map(|p| p.name.as_str())
            .collect();
        if unannotated.is_empty() {
            return BTreeMap::new();
        }
        let Some(body) = func.body.as_deref() else {
            return BTreeMap::new();
        };

        // param -> the one field it feeds; `None` once a second one is seen.
        let mut targets: BTreeMap<&str, Option<&str>> = BTreeMap::new();
        collect_this_field_stores(body, &unannotated, &mut targets);
        if targets.is_empty() {
            return BTreeMap::new();
        }

        let Some(&class_symbol) = self.context.class_context_stack.last() else {
            return BTreeMap::new();
        };
        let pairs: Vec<(String, String)> = targets
            .into_iter()
            .filter_map(|(p, f)| f.map(|f| (p.to_string(), f.to_string())))
            .collect();

        let mut out = BTreeMap::new();
        for (param, field) in pairs {
            let param_key = self.context.intern_string(&param);
            let field_key = self.context.intern_string(&field);
            let field_symbol = self.class_fields.get(&class_symbol).and_then(|fields| {
                fields
                    .iter()
                    .find(|(name, _, is_static)| *name == field_key && !*is_static)
                    .map(|(_, sym, _)| *sym)
            });
            let Some(field_symbol) = field_symbol else {
                continue;
            };
            if let Some(sym) = self.context.symbol_table.get_symbol(field_symbol) {
                out.insert(param_key, sym.type_id);
            }
        }
        if std::env::var_os("RAYZOR_PARAM_INFER_LOG").is_some() && !out.is_empty() {
            eprintln!("[param-infer] {}: {} parameter(s)", func.name, out.len());
        }
        out
    }

    /// Resolve a TypePath to a TypeId for constructor calls
    /// Ensure a symbol has a valid class type. If its type_id is invalid or has
    /// no entry in the type_table, create a Class type and link it to the symbol.
    /// This handles extern classes that were pre-registered as placeholders.
    pub(crate) fn ensure_symbol_has_class_type(
        &mut self,
        sym_id: SymbolId,
        type_id: TypeId,
    ) -> TypeId {
        if type_id == TypeId::invalid() || self.context.type_table.borrow().get(type_id).is_none() {
            let class_type = self
                .context
                .type_table
                .borrow_mut()
                .create_class_type(sym_id, Vec::new());
            self.context
                .symbol_table
                .update_symbol_type(sym_id, class_type);
            self.context
                .symbol_table
                .register_type_symbol_mapping(class_type, sym_id);
            class_type
        } else {
            type_id
        }
    }

    /// Resolve a TypeId to the underlying class-like symbol for member lookup.
    pub(crate) fn resolve_type_to_class_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        let type_table = self.context.type_table.borrow();
        self.resolve_type_to_class_symbol_inner(&type_table, type_id)
    }

    /// Inner helper that takes a borrowed type table to allow recursive calls
    fn resolve_type_to_class_symbol_inner(
        &self,
        type_table: &std::cell::Ref<'_, crate::tast::TypeTable>,
        type_id: TypeId,
    ) -> Option<SymbolId> {
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                crate::tast::core::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                crate::tast::core::TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                // `@:coreType extern abstract` receivers (Box<Int>, Atomic<Int>, Ptr<Int>)
                // resolve to the abstract symbol so instance method calls (cell.asPtr(),
                // a.fetchAdd()) find the pre-pass-typed method in the abstract's scope and
                // keep their declared return type instead of decaying to Dynamic.
                crate::tast::core::TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    // For generic instances like Thread<Int>, resolve the base type
                    self.resolve_type_to_class_symbol_inner(type_table, *base_type)
                }
                crate::tast::core::TypeKind::TypeAlias { target_type, .. } => {
                    // For type aliases like `typedef Bytes = rayzor.Bytes`, follow the target type
                    self.resolve_type_to_class_symbol_inner(type_table, *target_type)
                }
                // `Null<C>` receiver — methods resolve against C (an explicit
                // Null<T> annotation must behave like the ?param sugar).
                crate::tast::core::TypeKind::Optional { inner_type, .. } => {
                    self.resolve_type_to_class_symbol_inner(type_table, *inner_type)
                }
                crate::tast::core::TypeKind::Placeholder { name } => {
                    // For extern classes (Placeholder types), look up by name in the symbol table
                    // These classes may have been compiled in a different unit (stdlib)
                    let placeholder_name = *name;

                    // Try exact match first (bare name like "Bytes")
                    let results = self.context.symbol_table.find_symbols(|sym| {
                        sym.name == placeholder_name
                            && matches!(
                                sym.kind,
                                crate::tast::symbols::SymbolKind::Class
                                    | crate::tast::symbols::SymbolKind::Interface
                            )
                    });
                    if let Some(sym) = results.first() {
                        return Some(sym.id);
                    }

                    // Try matching qualified placeholder name against symbol's qualified_name
                    // e.g., placeholder "rayzor.Bytes" matches symbol with qualified_name "rayzor.Bytes"
                    let results = self.context.symbol_table.find_symbols(|sym| {
                        matches!(
                            sym.kind,
                            crate::tast::symbols::SymbolKind::Class
                                | crate::tast::symbols::SymbolKind::Interface
                        ) && sym.qualified_name == Some(placeholder_name)
                    });
                    if let Some(sym) = results.first() {
                        return Some(sym.id);
                    }

                    // Try matching bare name extracted from qualified placeholder
                    // e.g., "rayzor.Bytes" -> try matching symbol name "Bytes"
                    let name_str = self
                        .context
                        .string_interner
                        .get(placeholder_name)
                        .unwrap_or("");
                    let bare_name = name_str.rsplit('.').next().unwrap_or(name_str);
                    if bare_name != name_str {
                        let bare_interned = self.context.string_interner.intern(bare_name);
                        let results = self.context.symbol_table.find_symbols(|sym| {
                            sym.name == bare_interned
                                && matches!(
                                    sym.kind,
                                    crate::tast::symbols::SymbolKind::Class
                                        | crate::tast::symbols::SymbolKind::Interface
                                )
                        });
                        if let Some(sym) = results.first() {
                            return Some(sym.id);
                        }
                    }

                    None
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Get the class name for a given TypeId, if it resolves to a class.
    /// Prefers qualified_name (e.g. "sys.io.FileOutput") over bare name.
    pub(crate) fn get_class_name_for_type(&self, type_id: TypeId) -> Option<String> {
        if let Some(class_symbol) = self.resolve_type_to_class_symbol(type_id) {
            if let Some(sym) = self.context.symbol_table.get_symbol(class_symbol) {
                // Prefer qualified name for disambiguation
                if let Some(qname) = sym.qualified_name {
                    if let Some(qname_str) = self.context.string_interner.get(qname) {
                        return Some(qname_str.to_string());
                    }
                }
                return self
                    .context
                    .string_interner
                    .get(sym.name)
                    .map(|s| s.to_string());
            }
        }
        // Fallback: check if the type is a Placeholder with a recognizable name
        let type_table = self.context.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            if let crate::tast::core::TypeKind::Placeholder { name } = &type_info.kind {
                return self
                    .context
                    .string_interner
                    .get(*name)
                    .map(|s| s.to_string());
            }
        }
        None
    }

    /// Copy parent class fields to child class for field resolution
    /// This ensures that inherited fields can be resolved correctly in the child class
    /// Call this BEFORE processing child's members so fields are available in constructors
    fn copy_parent_fields(&mut self, parent_type_id: TypeId, child_symbol: SymbolId) {
        // Get the parent class symbol from the type
        if let Some(parent_symbol) = self.resolve_type_to_class_symbol(parent_type_id) {
            // Copy this parent's fields
            // Note: If the parent itself inherits from a grandparent, its class_fields
            // will already contain the grandparent's fields (since we process classes in order)
            if let Some(parent_fields) = self.class_fields.get(&parent_symbol).cloned() {
                if let Some(child_fields) = self.class_fields.get_mut(&child_symbol) {
                    // Prepend parent fields: splice at front in one operation
                    // instead of O(n) insert(0) per field.
                    let mut merged = parent_fields;
                    merged.append(child_fields);
                    *child_fields = merged;
                }
            }
        }
    }

    /// Copy parent class methods to child class for method resolution
    /// This enables method inheritance and overriding
    /// Call this AFTER processing child's members so child methods come first for overriding
    fn copy_parent_methods(&mut self, parent_type_id: TypeId, child_symbol: SymbolId) {
        // `RAYZOR_INHERIT_DEBUG=1` reports what a child actually inherits.
        // A cross-module parent resolves to a Class symbol but carries no
        // methods in this context, so the copy below is a no-op and every
        // inherited name is unresolvable -- which reads identically to the
        // parent having no methods at all.
        let dbg = std::env::var("RAYZOR_INHERIT_DEBUG").is_ok();
        if dbg {
            let ps = self.resolve_type_to_class_symbol(parent_type_id);
            let kind = {
                let tt = self.context.type_table.borrow();
                tt.get(parent_type_id).map(|t| format!("{:?}", t.kind))
            };
            let child_name = self
                .context
                .symbol_table
                .get_symbol(child_symbol)
                .and_then(|s| self.context.string_interner.get(s.name))
                .unwrap_or("?")
                .to_string();
            eprintln!(
                "[inherit] child={} parent_type={:?} kind={:?} parent_symbol={:?} methods={:?}",
                child_name,
                parent_type_id,
                kind.map(|k| k.chars().take(70).collect::<String>()),
                ps,
                ps.and_then(|p| self.class_methods.get(&p)).map(|m| m.len()),
            );
        }
        // Get the parent class symbol from the type
        if let Some(parent_symbol) = self.resolve_type_to_class_symbol(parent_type_id) {
            // Copy this parent's methods
            // Parent methods are added to the child's method list before child methods are processed
            // When child methods are added, they will replace parent methods with the same name (override)
            if let Some(parent_methods) = self.class_methods.get(&parent_symbol).cloned() {
                // Clone to avoid borrow conflicts
                if let Some(child_methods) = self.class_methods.get_mut(&child_symbol) {
                    // Add parent methods to child's method list
                    // Child methods will override these when they're processed
                    for parent_method in parent_methods.iter() {
                        child_methods.push(*parent_method);
                    }
                }
            }
        }
    }

    /// The anonymous structure an un-annotated method returns, when the return
    /// expression is an object literal whose field values are literals.
    ///
    /// Read off the syntax rather than lowered: this runs while the class is
    /// still being registered, so lowering here would create symbols in the
    /// wrong scope and evaluate expressions twice. Anything less direct keeps
    /// today's `Dynamic`.
    fn anonymous_return_type_from_ast(
        &mut self,
        func: &parser::haxe_ast::Function,
    ) -> Option<TypeId> {
        use parser::haxe_ast::ExprKind;

        fn returned_expr(expr: &parser::haxe_ast::Expr) -> Option<&parser::haxe_ast::Expr> {
            match &expr.kind {
                ExprKind::Return(Some(inner)) => Some(inner),
                ExprKind::Block(elements) => {
                    elements.iter().rev().find_map(|element| match element {
                        parser::BlockElement::Expr(e) => returned_expr(e),
                        _ => None,
                    })
                }
                _ => None,
            }
        }

        let body = func.body.as_ref()?;
        let ExprKind::Object(fields) = &returned_expr(body)?.kind else {
            return None;
        };
        if fields.is_empty() {
            return None;
        }

        let mut field_types = Vec::with_capacity(fields.len());
        for field in fields {
            let type_id = {
                let table = self.context.type_table.borrow();
                match &field.expr.kind {
                    ExprKind::String(_) => table.string_type(),
                    ExprKind::Int(_) => table.int_type(),
                    ExprKind::Float(_) => table.float_type(),
                    ExprKind::Bool(_) => table.bool_type(),
                    // A field whose type needs the expression evaluated is not
                    // decidable from syntax; leaving the whole thing Dynamic is
                    // what already happens, and a wrong field type is worse
                    // than none -- it reads back as garbage instead of null.
                    _ => return None,
                }
            };
            let name = self.context.intern_string(&field.name);
            field_types.push((name, type_id));
        }

        Some(crate::tast::type_resolution::create_anonymous_object_type(
            &self.context.type_table,
            field_types,
        ))
    }
}
