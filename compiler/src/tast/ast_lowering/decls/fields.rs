//! Fields, functions, parameters and signatures.

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
    /// Lower a function declaration (not used anymore - functions are in module fields)
    fn lower_function_declaration(
        &mut self,
        function_decl: &Function,
    ) -> LoweringResult<TypedDeclaration> {
        let function_name = self.context.intern_string(&function_decl.name);
        let function_symbol = self.context.symbol_table.create_function(function_name);

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&function_decl.type_params)?;
        let type_param_map: BTreeMap<InternedString, TypeId> = type_params
            .iter()
            .map(|tp| (tp.name, TypeId::invalid()))
            .collect();
        self.context.push_type_parameters(type_param_map);

        // Process parameters
        let mut parameters = Vec::with_capacity(function_decl.params.len());
        for param in &function_decl.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Process return type
        let return_type = if let Some(ret_type) = &function_decl.return_type {
            self.lower_type(ret_type)?
        } else {
            self.context.type_table.borrow().void_type()
        };

        // Process body
        let body = if let Some(body_expr) = &function_decl.body {
            // Convert expression to statement
            vec![self.lower_expression_as_statement(body_expr)?]
        } else {
            Vec::new()
        };

        // Process modifiers - skip for now

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let typed_function = TypedFunction {
            symbol_id: function_symbol,
            name: function_name,
            parameters,
            return_type,
            body,
            visibility: Visibility::Public,
            effects: crate::tast::node::FunctionEffects::default(),
            type_parameters: type_params,
            is_static: false, // Top-level functions are not static
            source_location: self.context.create_location_from_span(function_decl.span),
            metadata: FunctionMetadata::default(),
        };

        Ok(TypedDeclaration::Function(typed_function))
    }

    /// Lower a field
    pub(crate) fn lower_field(&mut self, field: &ClassField) -> LoweringResult<TypedField> {
        self.lower_field_with_symbol(field, None)
    }

    /// Lower a field while optionally reusing a symbol created by a declaration
    /// pre-pass. Enum-abstract constants need stable SymbolIds: a method above
    /// the abstract resolves the pre-registered symbol, and the initializer
    /// lowered later must publish its value under that same symbol.
    pub(crate) fn lower_field_with_symbol(
        &mut self,
        field: &ClassField,
        pre_registered_symbol: Option<SymbolId>,
    ) -> LoweringResult<TypedField> {
        let (field_name, field_type, initializer, mutability, is_static, property_access) =
            match &field.kind {
                ClassFieldKind::Var {
                    name,
                    type_hint,
                    expr,
                } => {
                    // Lower initializer first so we can infer type from it
                    let initializer = if let Some(expr) = expr {
                        Some(self.lower_expression(expr)?)
                    } else {
                        None
                    };

                    let field_type = if let Some(type_hint) = type_hint {
                        self.lower_type(type_hint)?
                    } else if let Some(ref init_expr) = initializer {
                        // Infer type from initializer expression
                        init_expr.expr_type
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    (
                        name.clone(),
                        field_type,
                        initializer,
                        crate::tast::Mutability::Mutable,
                        is_static,
                        None, // No property access for regular var fields
                    )
                }
                ClassFieldKind::Final {
                    name,
                    type_hint,
                    expr,
                } => {
                    // Lower initializer first so we can infer type from it
                    let initializer = if let Some(expr) = expr {
                        Some(self.lower_expression(expr)?)
                    } else {
                        None
                    };

                    let field_type = if let Some(type_hint) = type_hint {
                        self.lower_type(type_hint)?
                    } else if let Some(ref init_expr) = initializer {
                        // Infer type from initializer expression
                        init_expr.expr_type
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };

                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    (
                        name.clone(),
                        field_type,
                        initializer,
                        crate::tast::Mutability::Immutable,
                        is_static,
                        None, // No property access for final fields
                    )
                }
                ClassFieldKind::Property {
                    name,
                    type_hint,
                    getter,
                    setter,
                } => {
                    // Handle property with getter/setter
                    let field_type = if let Some(type_hint) = type_hint {
                        self.lower_type(type_hint)?
                    } else {
                        self.context.type_table.borrow().dynamic_type()
                    };
                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    // Properties are generally mutable unless they only have getters
                    let mutability = match (getter, setter) {
                        (_, parser::PropertyAccess::Never) => crate::tast::Mutability::Immutable,
                        (_, parser::PropertyAccess::Null) => crate::tast::Mutability::Immutable,
                        _ => crate::tast::Mutability::Mutable,
                    };

                    // Convert parser PropertyAccess to TAST PropertyAccessor
                    // TODO: Resolve method names to SymbolIds in a second pass after all methods are lowered
                    let getter_accessor = self.convert_property_accessor(getter, name, true);
                    let setter_accessor = self.convert_property_accessor(setter, name, false);

                    let property_info = Some(crate::tast::PropertyAccessInfo {
                        getter: getter_accessor,
                        setter: setter_accessor,
                    });

                    (
                        name.clone(),
                        field_type,
                        None,
                        mutability,
                        is_static,
                        property_info,
                    )
                }
                ClassFieldKind::Function(func) => {
                    // Functions should be handled separately as methods, not fields
                    // Return placeholder for now
                    let field_type = self.context.type_table.borrow().dynamic_type();
                    let is_static = field
                        .modifiers
                        .iter()
                        .any(|m| matches!(m, parser::Modifier::Static));

                    (
                        func.name.clone(),
                        field_type,
                        None,
                        crate::tast::Mutability::Immutable,
                        is_static,
                        None, // No property access for function fields
                    )
                }
            };

        let interned_field_name = self.context.intern_string(&field_name);
        let field_symbol = pre_registered_symbol.unwrap_or_else(|| {
            self.context
                .symbol_table
                .create_variable(interned_field_name)
        });

        // Update the field symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(field_symbol, field_type);

        let mut field_flags = self.extract_metadata_flags(&field.meta, field_symbol);
        for modifier in &field.modifiers {
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

        // Add field symbol to current class scope for resolution
        if let Some(scope) = self
            .context
            .scope_tree
            .get_scope_mut(self.context.current_scope)
        {
            scope.add_symbol(field_symbol, interned_field_name);
        }

        // Track field in the current class for implicit this resolution
        if let Some(class_symbol) = self.context.class_context_stack.last() {
            if let Some(field_list) = self.class_fields.get_mut(class_symbol) {
                if let Some(entry) = field_list
                    .iter_mut()
                    .find(|(_, symbol, _)| *symbol == field_symbol)
                {
                    *entry = (interned_field_name, field_symbol, is_static);
                } else {
                    field_list.push((interned_field_name, field_symbol, is_static));
                }
            }
        }

        // Process modifiers and access separately
        let modifier_info = self.lower_modifiers(&field.modifiers)?;
        let visibility = self.lower_access(&field.access);

        // QUALIFY STATIC FIELD SYMBOLS. A static's symbol carried
        // `qualified_name = None`, so every cross-module lookup downstream had
        // nothing to match on. Use `Class.field`, matching exactly how the
        // global is named where it is created in hir_to_mir.
        if modifier_info.is_static {
            if let Some(&class_symbol) = self.context.class_context_stack.last() {
                let class_name = self
                    .context
                    .symbol_table
                    .get_symbol(class_symbol)
                    .and_then(|c| self.context.string_interner.get(c.name))
                    .map(|s| s.to_string());
                let field_name_str = self
                    .context
                    .string_interner
                    .get(interned_field_name)
                    .map(|s| s.to_string());
                if let (Some(cn), Some(fname)) = (class_name, field_name_str) {
                    let qn = format!("{}.{}", cn, fname);
                    let qn_interned = self.context.string_interner.intern(&qn);
                    if let Some(sym) = self.context.symbol_table.get_symbol_mut(field_symbol) {
                        if sym.qualified_name.is_none() {
                            sym.qualified_name = Some(qn_interned);
                        }
                    }
                }
            }
        }

        // Extract @:default(value) metadata for @:derive(Default)
        let metadata_default = field
            .meta
            .iter()
            .find(|m| m.name == "default")
            .and_then(|m| m.params.first())
            .and_then(|expr| self.lower_expression(expr).ok());

        Ok(TypedField {
            symbol_id: field_symbol,
            name: interned_field_name,
            field_type,
            initializer,
            mutability,
            visibility, // Use visibility from access keyword (public/private), not from modifiers
            is_static: modifier_info.is_static,
            property_access,
            metadata_default,
            source_location: self.context.create_location_from_span(field.span),
        })
    }

    /// Lower a function object
    pub(crate) fn lower_function_object(
        &mut self,
        func: &Function,
        meta: &[parser::Metadata],
        modifiers: &[parser::Modifier],
    ) -> LoweringResult<TypedFunction> {
        let function_name = self.context.intern_string(&func.name);
        // A module-level function belongs to the module's scope, the way a class
        // declared in the same file does, so the types in that file can call it by
        // its bare name. Creating the symbol unscoped leaves it findable only by a
        // whole-table scan, which no call site performs.
        let function_symbol = self
            .context
            .symbol_table
            .create_function_in_scope(function_name, self.context.current_scope);
        let mut symbol_flags = self.extract_metadata_flags(meta, function_symbol);
        for modifier in modifiers {
            use crate::tast::symbols::SymbolFlags;
            symbol_flags = symbol_flags.union(match modifier {
                parser::haxe_ast::Modifier::Static => SymbolFlags::STATIC,
                parser::haxe_ast::Modifier::Inline => SymbolFlags::INLINE,
                parser::haxe_ast::Modifier::Macro => SymbolFlags::MACRO,
                parser::haxe_ast::Modifier::Dynamic => SymbolFlags::DYNAMIC,
                parser::haxe_ast::Modifier::Override => SymbolFlags::OVERRIDE,
                parser::haxe_ast::Modifier::Final => SymbolFlags::FINAL,
                parser::haxe_ast::Modifier::Extern => SymbolFlags::EXTERN,
            });
        }
        if !symbol_flags.is_empty() {
            self.context
                .symbol_table
                .add_symbol_flags(function_symbol, symbol_flags);
        }
        let is_static = modifiers
            .iter()
            .any(|m| matches!(m, parser::haxe_ast::Modifier::Static));
        let is_inline = modifiers
            .iter()
            .any(|m| matches!(m, parser::haxe_ast::Modifier::Inline));

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&func.type_params)?;
        let type_param_map: BTreeMap<InternedString, TypeId> = type_params
            .iter()
            .map(|tp| (tp.name, TypeId::invalid()))
            .collect();
        self.context.push_type_parameters(type_param_map);

        // Process parameters
        let mut parameters = Vec::new();
        for param in &func.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Process return type
        let return_type = if let Some(ret_type) = &func.return_type {
            self.lower_type(ret_type)?
        } else {
            self.context.type_table.borrow().void_type()
        };

        // Process body
        let body = if let Some(body_expr) = &func.body {
            vec![self.lower_expression_as_statement(body_expr)?]
        } else {
            Vec::new()
        };

        self.context.pop_type_parameters();
        self.context.exit_scope();

        // Call sites type the result of a call from the callee symbol's type. Leaving
        // it invalid types the result as unknown, so a value returned into a Dynamic
        // parameter is passed without being boxed and is read back as a pointer.
        let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
        let function_type = self
            .context
            .type_table
            .borrow_mut()
            .create_function_type(param_types, return_type);
        self.context
            .symbol_table
            .update_symbol_type(function_symbol, function_type);

        Ok(TypedFunction {
            symbol_id: function_symbol,
            name: function_name,
            parameters,
            return_type,
            body,
            visibility: Visibility::Public,
            effects: crate::tast::node::FunctionEffects {
                is_inline,
                ..crate::tast::node::FunctionEffects::default()
            },
            type_parameters: Vec::new(), // TODO: Convert type parameters
            is_static,
            source_location: self.context.create_location(),
            metadata: FunctionMetadata::default(),
        })
    }

    pub(crate) fn lower_function_from_field(
        &mut self,
        field: &ClassField,
        func: &Function,
    ) -> LoweringResult<TypedFunction> {
        let function_name = self.context.intern_string(&func.name);

        // Get function symbol - may have been pre-registered during class declaration
        // This ensures the method is associated with its class
        let current_class = self.context.class_context_stack.last().copied();

        // Use the current scope as the class scope since we're inside the class
        // The class symbol itself is in the parent scope, but methods are in the class scope
        let class_scope = if current_class.is_some() {
            self.context.current_scope
        } else {
            ScopeId::first() // Fallback to root scope
        };

        // Look up the pre-registered function symbol, or create a new one if not found
        // (constructors named "new" are not pre-registered since they're handled specially)
        let function_symbol = if let Some(existing) = self
            .context
            .symbol_table
            .lookup_symbol(class_scope, function_name)
        {
            existing.id
        } else {
            // Create the function symbol in the class scope (e.g., for constructors)
            self.context
                .symbol_table
                .create_function_in_scope(function_name, class_scope)
        };

        let mut function_flags = self.extract_metadata_flags(&field.meta, function_symbol);
        for modifier in &field.modifiers {
            use crate::tast::symbols::SymbolFlags;
            function_flags = function_flags.union(match modifier {
                parser::haxe_ast::Modifier::Static => SymbolFlags::STATIC,
                parser::haxe_ast::Modifier::Inline => SymbolFlags::INLINE,
                parser::haxe_ast::Modifier::Macro => SymbolFlags::MACRO,
                parser::haxe_ast::Modifier::Dynamic => SymbolFlags::DYNAMIC,
                parser::haxe_ast::Modifier::Override => SymbolFlags::OVERRIDE,
                parser::haxe_ast::Modifier::Final => SymbolFlags::FINAL,
                parser::haxe_ast::Modifier::Extern => SymbolFlags::EXTERN,
            });
        }
        if !function_flags.is_empty() {
            self.context
                .symbol_table
                .add_symbol_flags(function_symbol, function_flags);
        }

        // Update qualified name (full path including class hierarchy)
        self.context.update_symbol_qualified_name(function_symbol);

        // DEBUG: Check if qualified name was set correctly
        if let Some(sym) = self.context.symbol_table.get_symbol(function_symbol) {
            let qname = sym
                .qualified_name
                .and_then(|qn| self.context.string_interner.get(qn))
                .unwrap_or("<none>");
        }

        // Also track this method in our class_fields for field resolution
        if let Some(class_symbol) = current_class {
            if let Some(fields_list) = self.class_fields.get_mut(&class_symbol) {
                // Check if field has static modifier
                let is_static = field
                    .modifiers
                    .iter()
                    .any(|m| matches!(m, Modifier::Static));
                fields_list.push((function_name, function_symbol, is_static));
            }
        }

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&func.type_params)?;
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

        // Process parameters. An unannotated one takes the type of the field
        // it is stored into, the way Haxe's own unification would.
        let inferred = self.param_types_from_field_stores(func);
        let mut parameters = Vec::new();
        for param in &func.params {
            let hint = if param.type_hint.is_none() {
                let key = self.context.intern_string(&param.name);
                inferred.get(&key).copied()
            } else {
                None
            };
            parameters.push(self.lower_parameter_with_hint(param, hint)?);
        }

        // Check if this is a static method BEFORE lowering the body, so that
        // the implicit `this` logic in identifier resolution knows whether
        // `this` is available.
        let is_static_method = field
            .modifiers
            .iter()
            .any(|m| matches!(m, parser::Modifier::Static));
        let prev_static = self.in_static_method;
        self.in_static_method = is_static_method;

        // Pre-lower an explicit return-type annotation (cheap — `lower_type`
        // just resolves a type reference) so bare enum-variant identifiers
        // in `return` statements can disambiguate against it (see
        // `expected_return_type` doc comment). Reused below as the final
        // `return_type` when present, so this isn't wasted work.
        let prev_expected_return = self.context.expected_return_type;
        let annotated_return_type = match &func.return_type {
            Some(ret_type) => Some(self.lower_type(ret_type)?),
            None => None,
        };
        self.context.expected_return_type = annotated_return_type;

        // Process body first (we need it to infer return type if not specified)
        let (body, body_statements_for_inference) = if let Some(body_expr) = &func.body {
            let typed_expr = self.lower_expression(body_expr)?;
            // Extract statements for return type inference
            let stmts_for_inference = match &typed_expr.kind {
                TypedExpressionKind::Block { statements, .. } => statements.clone(),
                _ => vec![],
            };
            let body = vec![TypedStatement::Expression {
                expression: typed_expr,
                source_location: self.context.span_to_location(&body_expr.span),
            }];
            (body, stmts_for_inference)
        } else {
            (Vec::new(), Vec::new())
        };
        self.context.expected_return_type = prev_expected_return;

        // Restore static method flag
        self.in_static_method = prev_static;

        // Process return type - if not specified, infer from body.
        // Reuse the pre-lowered annotation from above instead of calling
        // `lower_type` again (avoids doing the resolution twice).
        let return_type = if let Some(ret_type) = annotated_return_type {
            ret_type
        } else {
            // Try to infer return type from return statements in the body.
            // Use the unwrapped block statements for inference since the body
            // is now wrapped in an Expression(Block(...)) for consistency.
            if body_statements_for_inference.is_empty() {
                self.infer_return_type_from_body(&body)
            } else {
                self.infer_return_type_from_body(&body_statements_for_inference)
            }
        };

        // Create function type and update symbol
        let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
        let function_type = self
            .context
            .type_table
            .borrow_mut()
            .create_function_type(param_types, return_type);

        // Update the symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(function_symbol, function_type);

        // Process field modifiers and access
        let modifier_info = self.lower_modifiers(&field.modifiers)?;
        let visibility = self.lower_access(&field.access);

        // Process @:overload metadata
        let overload_signatures = self.process_overload_metadata(&field.meta)?;

        // Process @:op metadata for operator overloading
        let operator_metadata = self.process_operator_metadata(&field.meta)?;

        // Check for @:arrayAccess metadata
        let is_array_access = self.has_array_access_metadata(&field.meta);

        // Check for @:from / @:to metadata (abstract implicit conversions)
        let is_from_conversion = field.meta.iter().any(|m| m.name == "from");
        let is_to_conversion = field.meta.iter().any(|m| m.name == "to");

        self.context.pop_type_parameters();
        self.context.exit_scope();

        let body_len = body.len();

        Ok(TypedFunction {
            symbol_id: function_symbol,
            name: function_name,
            parameters,
            return_type,
            body,
            visibility,
            effects: crate::tast::node::FunctionEffects {
                can_throw: self.analyze_can_throw(&func.body),
                async_kind: self.detect_async_kind(&field.meta),
                is_pure: self.analyze_is_pure(&func.body),
                is_inline: modifier_info.is_inline,
                exception_types: vec![],
                memory_effects: MemoryEffects::default(),
                resource_effects: ResourceEffects::default(),
            },
            type_parameters: type_params,
            is_static: modifier_info.is_static,
            source_location: self.context.create_location_from_span(field.span),
            metadata: FunctionMetadata {
                complexity_score: self.calculate_complexity(&func.body),
                statement_count: body_len,
                is_recursive: false, // Recursion detection requires call graph analysis
                call_count: 0,
                is_override: modifier_info.is_override,
                overload_signatures,
                operator_metadata,
                is_array_access,
                is_from_conversion,
                is_to_conversion,
                memory_annotations: self.extract_memory_annotations(&field.meta),
            },
        })
    }

    /// Lower a function signature for interfaces (no body, just signature)
    pub(crate) fn lower_function_signature(
        &mut self,
        field: &ClassField,
        func: &Function,
    ) -> LoweringResult<TypedMethodSignature> {
        let function_name = self.context.intern_string(&func.name);
        let function_symbol = self.context.symbol_table.create_function(function_name);

        // Enter function scope
        let function_scope = self.context.enter_scope(ScopeKind::Function);

        // Process type parameters
        let type_params = self.lower_type_parameters(&func.type_params)?;
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

        // Process parameters
        let mut parameters = Vec::new();
        for param in &func.params {
            parameters.push(self.lower_parameter(param)?);
        }

        // Process return type
        let return_type = if let Some(ret_type) = &func.return_type {
            self.lower_type(ret_type)?
        } else {
            self.context.type_table.borrow().void_type()
        };

        // Create function type and update symbol
        let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
        let function_type = self
            .context
            .type_table
            .borrow_mut()
            .create_function_type(param_types, return_type);

        // Update the symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(function_symbol, function_type);

        // Interface methods have no body
        let body: Vec<TypedStatement> = Vec::new();

        // Process field modifiers and access
        let modifier_info = self.lower_modifiers(&field.modifiers)?;
        let visibility = self.lower_access(&field.access);

        self.context.pop_type_parameters();
        self.context.exit_scope();

        Ok(TypedMethodSignature {
            name: function_name,
            parameters,
            return_type,
            effects: crate::tast::node::FunctionEffects {
                can_throw: false,            // Interface methods are pure signatures
                async_kind: AsyncKind::Sync, // Async detection not needed for now
                is_pure: true,               // Interface methods are pure signatures
                is_inline: modifier_info.is_inline,
                exception_types: vec![],
                memory_effects: MemoryEffects::default(),
                resource_effects: ResourceEffects::default(),
            },
            source_location: self.context.create_location_from_span(field.span),
        })
    }

    pub(crate) fn lower_parameter(
        &mut self,
        parameter: &FunctionParam,
    ) -> LoweringResult<TypedParameter> {
        self.lower_parameter_with_hint(parameter, None)
    }

    /// As `lower_parameter`, with a type recovered from the body for a
    /// parameter the source did not annotate.
    fn lower_parameter_with_hint(
        &mut self,
        parameter: &FunctionParam,
        inferred: Option<TypeId>,
    ) -> LoweringResult<TypedParameter> {
        let param_name = self.context.intern_string(&parameter.name);
        // Create the parameter symbol with the current scope
        let param_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(param_name, self.context.current_scope);

        // Add parameter to the current (function) scope so it can be resolved
        if let Some(scope) = self
            .context
            .scope_tree
            .get_scope_mut(self.context.current_scope)
        {
            scope.add_symbol(param_symbol, param_name);
        }

        let param_type = if let Some(type_annotation) = &parameter.type_hint {
            self.lower_type(type_annotation)?
        } else if let Some(ty) = inferred {
            ty
        } else {
            self.context.type_table.borrow().dynamic_type()
        };

        // Update the parameter symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(param_symbol, param_type);

        let default_value = if let Some(default) = &parameter.default_value {
            Some(self.lower_expression(default)?)
        } else {
            None
        };

        Ok(TypedParameter {
            symbol_id: param_symbol,
            name: param_name,
            param_type: param_type,
            is_optional: parameter.optional,
            default_value,
            mutability: crate::tast::Mutability::Immutable,
            ownership: crate::tast::ParamOwnership::from_metadata(&parameter.meta),
            source_location: self.context.create_location_from_span(parameter.span),
        })
    }

    /// Lower a function parameter, forcing its type to `param_type` (used
    /// when the surrounding call supplies an expected lambda signature
    /// and the parameter itself has no `type_hint`). Other than the type
    /// substitution this mirrors `lower_function_param` exactly.
    pub(crate) fn lower_function_param_with_type(
        &mut self,
        param: &parser::FunctionParam,
        param_type: TypeId,
    ) -> Result<TypedParameter, LoweringError> {
        let param_name = self.context.string_interner.intern(&param.name);
        let param_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(param_name, self.context.current_scope);

        self.context
            .symbol_table
            .update_symbol_type(param_symbol, param_type);

        let default_value = if let Some(default_expr) = &param.default_value {
            Some(self.lower_expression(default_expr)?)
        } else {
            None
        };

        Ok(TypedParameter {
            symbol_id: param_symbol,
            name: param_name,
            param_type,
            is_optional: param.optional,
            default_value,
            mutability: crate::tast::symbols::Mutability::Immutable,
            ownership: crate::tast::ParamOwnership::from_metadata(&param.meta),
            source_location: self.context.span_to_location(&param.span),
        })
    }

    pub(crate) fn lower_function_param(
        &mut self,
        param: &parser::FunctionParam,
    ) -> Result<TypedParameter, LoweringError> {
        // Create symbol for parameter in the current scope
        let param_name = self.context.string_interner.intern(&param.name);
        let param_symbol = self
            .context
            .symbol_table
            .create_variable_in_scope(param_name, self.context.current_scope);

        // Resolve parameter type
        let param_type = if let Some(type_hint) = &param.type_hint {
            self.lower_type(type_hint)?
        } else {
            self.context.type_table.borrow().dynamic_type()
        };

        // Update the symbol with its type
        self.context
            .symbol_table
            .update_symbol_type(param_symbol, param_type);

        // Lower default value if present
        let default_value = if let Some(default_expr) = &param.default_value {
            Some(self.lower_expression(default_expr)?)
        } else {
            None
        };

        Ok(TypedParameter {
            symbol_id: param_symbol,
            name: self.context.string_interner.intern(&param.name),
            param_type,
            is_optional: param.optional,
            default_value,
            mutability: crate::tast::symbols::Mutability::Immutable, // Function parameters are immutable by default in Haxe
            ownership: crate::tast::ParamOwnership::from_metadata(&param.meta),
            source_location: self.context.span_to_location(&param.span),
        })
    }

    /// Lower a function body
    pub(crate) fn lower_function_body(
        &mut self,
        body: &parser::Expr,
    ) -> Result<Vec<TypedStatement>, LoweringError> {
        match &body.kind {
            parser::ExprKind::Block(elements) => {
                // Function body is a block - lower all elements with error recovery
                let mut statements = Vec::new();
                for element in elements {
                    match element {
                        parser::BlockElement::Expr(expr) => {
                            // Check if this is a variable declaration expression
                            match &expr.kind {
                                parser::ExprKind::Var { .. } | parser::ExprKind::Final { .. } => {
                                    // Variable declaration - lower as expression and convert to statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            // Extract the declaration info to create a proper statement
                                            if let TypedExpressionKind::VarDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Mutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else if let TypedExpressionKind::FinalDeclarationExpr {
                                                symbol_id,
                                                var_type,
                                                initializer,
                                            } = typed_expr.kind
                                            {
                                                statements.push(TypedStatement::VarDeclaration {
                                                    symbol_id,
                                                    var_type,
                                                    initializer: Some(*initializer),
                                                    mutability: crate::tast::symbols::Mutability::Immutable,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                parser::ExprKind::Return(_) => {
                                    // Return expression - convert to TypedStatement::Return
                                    // so infer_return_type_from_body can extract the return type
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            if let TypedExpressionKind::Return { value } =
                                                typed_expr.kind
                                            {
                                                statements.push(TypedStatement::Return {
                                                    value: value.map(|v| *v),
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            } else {
                                                statements.push(TypedStatement::Expression {
                                                    expression: typed_expr,
                                                    source_location: self
                                                        .context
                                                        .span_to_location(&expr.span),
                                                });
                                            }
                                        }
                                        Err(e) => {
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                                _ => {
                                    // Regular expression - lower and wrap in statement
                                    match self.lower_expression(expr) {
                                        Ok(typed_expr) => {
                                            statements.push(TypedStatement::Expression {
                                                expression: typed_expr,
                                                source_location: self
                                                    .context
                                                    .span_to_location(&expr.span),
                                            });
                                        }
                                        Err(e) => {
                                            // Collect error and continue processing other statements
                                            self.collected_errors.push(e);
                                        }
                                    }
                                }
                            }
                        }
                        parser::BlockElement::Import(_)
                        | parser::BlockElement::Using(_)
                        | parser::BlockElement::Conditional(_) => {
                            // Skip imports, using statements, and conditional compilation for now
                            // These should be handled at the module level
                        }
                    }
                }
                Ok(statements)
            }
            _ => {
                // Single expression body - wrap in expression statement
                match self.lower_expression(body) {
                    Ok(typed_expr) => Ok(vec![TypedStatement::Expression {
                        expression: typed_expr,
                        source_location: self.context.span_to_location(&body.span),
                    }]),
                    Err(e) => {
                        // Collect error and return empty statement list
                        self.collected_errors.push(e);
                        Ok(vec![])
                    }
                }
            }
        }
    }
}
