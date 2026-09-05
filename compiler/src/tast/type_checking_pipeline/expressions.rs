//! Type checking an expression, and the per-kind checks the big one delegates to.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Check an expression and return its type
    pub fn check_expression(&mut self, expr: &TypedExpression) -> Result<TypeId, String> {
        match &expr.kind {
            TypedExpressionKind::BinaryOp {
                left,
                right,
                operator: op,
            } => {
                self.check_binary_op_expr(left, right, op, expr.source_location)?;
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                type_arguments: _,
            } => {
                self.check_function_call_expr(function, arguments, expr.source_location)?;
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                let object_type = self.check_expression(object)?;

                // Check if field exists on the object type
                self.check_field_access(object_type, *field_symbol, expr.source_location, false)?;

                // Field access type checking completed - the actual type is already stored in the TAST node
            }
            TypedExpressionKind::StaticFieldAccess {
                class_symbol,
                field_symbol,
            } => {
                // Check if field exists on the class and is static
                if let Some(symbol) = self.type_checker.symbol_table.get_symbol(*class_symbol) {
                    self.check_field_access(
                        symbol.type_id,
                        *field_symbol,
                        expr.source_location,
                        true,
                    )?;
                }

                // Static field access type checking completed
            }
            TypedExpressionKind::StaticMethodCall {
                class_symbol,
                method_symbol,
                arguments,
                type_arguments: _,
            } => {
                // Check argument types
                let mut arg_types = Vec::new();
                for arg in arguments {
                    let arg_type = self.check_expression(arg)?;
                    arg_types.push(arg_type);
                }

                // Check if method exists on the class and is static
                let class_and_method_data =
                    self.find_class_by_symbol(*class_symbol)
                        .and_then(|class_def| {
                            class_def
                                .methods
                                .iter()
                                .find(|m| m.symbol_id == *method_symbol)
                                .map(|method| {
                                    (
                                        class_def.name,
                                        class_def.symbol_id,
                                        method.name,
                                        method.is_static,
                                    )
                                })
                        });

                if let Some((class_name, _class_id, method_name, is_static)) = class_and_method_data
                {
                    if !is_static {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::InstanceAccessFromStatic {
                                member_name: method_name,
                                class_name,
                            },
                            location: expr.source_location,
                            context: "Instance methods cannot be accessed statically".to_string(),
                            suggestion: Some(format!(
                                "Create an instance of {} to call this method",
                                self.string_interner.get(class_name).unwrap_or("<class>")
                            )),
                        });
                        return Err("Instance method accessed statically".to_string());
                    }
                }

                // Validate argument count and types against the method's function type
                let method_type_id = self
                    .type_checker
                    .symbol_table
                    .get_symbol(*method_symbol)
                    .map(|s| s.type_id);
                if let Some(method_type_id) = method_type_id {
                    let param_info = {
                        let type_table = self.type_checker.type_table.borrow();
                        type_table.get(method_type_id).and_then(|t| {
                            if let super::TypeKind::Function { params, .. } = &t.kind {
                                Some(params.clone())
                            } else {
                                None
                            }
                        })
                    };
                    if let Some(param_types) = param_info {
                        if param_types.len() != arg_types.len() {
                            self.emit_error(TypeCheckError {
                                kind: TypeErrorKind::TypeMismatch {
                                    expected: method_type_id,
                                    actual: method_type_id,
                                },
                                location: expr.source_location,
                                context: format!(
                                    "Method expects {} arguments but {} were provided",
                                    param_types.len(),
                                    arg_types.len()
                                ),
                                suggestion: None,
                            });
                        } else {
                            for (i, (expected, actual)) in
                                param_types.iter().zip(&arg_types).enumerate()
                            {
                                let compat =
                                    self.type_checker.check_compatibility(*actual, *expected);
                                if matches!(compat, TypeCompatibility::Incompatible) {
                                    self.emit_enhanced_type_error(
                                        *actual,
                                        *expected,
                                        arguments[i].source_location,
                                        &format!("Argument {} type mismatch", i + 1),
                                        &TypeErrorContext::FunctionCall {
                                            param_index: i,
                                            expected_type: *expected,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
            TypedExpressionKind::Block {
                statements,
                scope_id: _,
            } => {
                // Check all statements in the block
                for stmt in statements {
                    self.check_statement(stmt)?;
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Check if we're accessing an instance member from a static context
                if let Some((is_static_context, class_symbol_id)) = self.current_method_context {
                    if is_static_context {
                        // We're in a static method - check if the variable is an instance member
                        if let Some(symbol) = self.type_checker.symbol_table.get_symbol(*symbol_id)
                        {
                            // In Haxe, unqualified field names in methods resolve to this.field
                            // So we need to check if this variable name matches an instance field
                            if let Some(class_def) = self.find_class_by_symbol(class_symbol_id) {
                                // Check if this variable name matches any instance field name in the class
                                let matching_instance_field = class_def
                                    .fields
                                    .iter()
                                    .find(|f| f.name == symbol.name && !f.is_static);

                                if let Some(_field) = matching_instance_field {
                                    // This is an unqualified access to an instance field from a static method
                                    self.emit_error(TypeCheckError {
                                                kind: TypeErrorKind::InstanceAccessFromStatic {
                                                    member_name: symbol.name,
                                                    class_name: class_def.name,
                                                },
                                                location: expr.source_location,
                                                context: "Instance members cannot be accessed from static context".to_string(),
                                                suggestion: Some("Static methods cannot access instance fields without an explicit instance".to_string()),
                                            });
                                }
                            }
                        }
                    }
                }
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                let array_type = self.check_expression(array)?;
                let index_type = self.check_expression(index)?;

                // Check that index is a valid index type (Int)
                let type_table = self.type_checker.type_table.borrow();
                let int_type = type_table.int_type();
                drop(type_table);

                let index_compatibility =
                    self.type_checker.check_compatibility(index_type, int_type);
                if matches!(index_compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        index_type,
                        int_type,
                        expr.source_location,
                        "Array index must be Int",
                        &TypeErrorContext::ArrayAccess,
                    );
                }

                // Check that the array is actually an array type
                let is_valid_indexable = {
                    let type_table = self.type_checker.type_table.borrow();
                    if let Some(array_type_info) = type_table.get(array_type) {
                        matches!(
                            &array_type_info.kind,
                            super::TypeKind::Array { .. }
                                | super::TypeKind::String
                                | super::TypeKind::Dynamic
                        )
                    } else {
                        false
                    }
                };

                if !is_valid_indexable {
                    // Invalid array access
                    let dynamic_type = self.type_checker.type_table.borrow().dynamic_type();
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: dynamic_type, // Use dynamic as placeholder
                            actual: array_type,
                        },
                        location: expr.source_location,
                        context: "Cannot index non-array type".to_string(),
                        suggestion: Some(
                            "Only arrays, strings, and dynamic types can be indexed".to_string(),
                        ),
                    });
                }
            }
            TypedExpressionKind::Cast {
                expression,
                target_type,
                cast_kind,
            } => {
                let source_type = self.check_expression(expression)?;

                // Validate the cast based on cast kind and type compatibility
                match cast_kind {
                    CastKind::Explicit => {
                        // Check if explicit cast is valid
                        if !self.is_valid_explicit_cast(source_type, *target_type) {
                            self.emit_error(TypeCheckError {
                                kind: TypeErrorKind::InvalidCast {
                                    from_type: source_type,
                                    to_type: *target_type,
                                },
                                location: expr.source_location,
                                context: "Invalid explicit cast".to_string(),
                                suggestion: Some(
                                    "Check if the cast is supported or use safe conversion methods"
                                        .to_string(),
                                ),
                            });
                        }
                    }
                    CastKind::Implicit => {
                        // Implicit casts should always be compatible
                        let compatibility = self
                            .type_checker
                            .check_compatibility(source_type, *target_type);
                        if matches!(compatibility, TypeCompatibility::Incompatible) {
                            self.emit_enhanced_type_error(
                                source_type,
                                *target_type,
                                expr.source_location,
                                "Implicit cast failed - types are incompatible",
                                &TypeErrorContext::Assignment {
                                    target_type: *target_type,
                                },
                            );
                        }
                    }
                    CastKind::Checked => {
                        // Checked casts with runtime validation
                        // For now, allow them but could add warnings about potential runtime failures
                    }
                    CastKind::Unsafe => {
                        // Unsafe casts bypass type checking but we can still warn
                        // For now, we'll allow all unsafe casts but could add warnings
                    }
                }
            }
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                arguments,
                ..
            } => {
                self.check_method_call_expr(
                    receiver,
                    *method_symbol,
                    arguments,
                    expr.source_location,
                )?;
            }
            TypedExpressionKind::MethodReference {
                receiver,
                method_symbol: _,
            } => {
                // Only the receiver participates in type-checking; the
                // method-as-value's own Function<args -> ret> type is
                // set on the expression at lowering time.
                self.check_expression(receiver)?;
            }
            TypedExpressionKind::UnaryOp { operand, operator } => {
                let operand_type = self.check_expression(operand)?;

                // Check operand compatibility for the operator
                match operator {
                    super::node::UnaryOperator::Not => {
                        // ! operator expects boolean
                        let bool_type = self.type_checker.type_table.borrow().bool_type();
                        let compatibility = self
                            .type_checker
                            .check_compatibility(operand_type, bool_type);
                        if matches!(compatibility, TypeCompatibility::Incompatible) {
                            self.emit_enhanced_type_error(
                                operand_type,
                                bool_type,
                                expr.source_location,
                                "Logical NOT operator requires boolean operand",
                                &TypeErrorContext::UnaryOperation {
                                    operator: *operator,
                                },
                            );
                        }
                    }
                    super::node::UnaryOperator::Neg => {
                        // - operator expects numeric type
                        let type_table = self.type_checker.type_table.borrow();
                        let int_type = type_table.int_type();
                        let float_type = type_table.float_type();
                        drop(type_table);

                        let compat_int = self
                            .type_checker
                            .check_compatibility(operand_type, int_type);
                        let compat_float = self
                            .type_checker
                            .check_compatibility(operand_type, float_type);

                        let is_numeric = matches!(
                            compat_int,
                            TypeCompatibility::Identical | TypeCompatibility::Assignable
                        ) || matches!(
                            compat_float,
                            TypeCompatibility::Identical | TypeCompatibility::Assignable
                        );

                        if !is_numeric {
                            self.emit_enhanced_type_error(
                                operand_type,
                                int_type,
                                expr.source_location,
                                "Unary minus operator requires numeric operand",
                                &TypeErrorContext::UnaryOperation {
                                    operator: *operator,
                                },
                            );
                        }
                    }
                    _ => {
                        // Other unary operators (++, --, etc.)
                        // TODO: Add more specific checks
                    }
                }
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                // Check condition is boolean
                let condition_type = self.check_expression(condition)?;
                let bool_type = self.type_checker.type_table.borrow().bool_type();

                let compatibility = self
                    .type_checker
                    .check_compatibility(condition_type, bool_type);
                if matches!(compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        condition_type,
                        bool_type,
                        condition.source_location,
                        "Conditional expression requires boolean condition",
                        &TypeErrorContext::ConditionalExpression,
                    );
                }

                // Check then and else branches
                let then_type = self.check_expression(then_expr)?;

                if let Some(else_expr) = else_expr {
                    let else_type = self.check_expression(else_expr)?;

                    // Branches should have compatible types
                    let branch_compat = self.type_checker.check_compatibility(then_type, else_type);
                    if matches!(branch_compat, TypeCompatibility::Incompatible) {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: then_type,
                                actual: else_type,
                            },
                            location: expr.source_location,
                            context: "Conditional branches must have compatible types".to_string(),
                            suggestion: Some(
                                "Ensure both branches return the same type".to_string(),
                            ),
                        });
                    }
                }
            }
            TypedExpressionKind::Switch {
                discriminant,
                cases,
                default_case,
            } => {
                self.check_switch_expr(
                    discriminant,
                    cases,
                    default_case.as_deref(),
                    expr.expr_type,
                    expr.source_location,
                )?;
            }
            TypedExpressionKind::Try {
                try_expr,
                catch_clauses,
                finally_block,
            } => {
                // Check try block
                let try_type = self.check_expression(try_expr)?;

                // Check catch clauses and verify type consistency
                for catch in catch_clauses {
                    // Validate exception type
                    self.validate_exception_type(catch.exception_type, catch.source_location)?;

                    // Check exception variable is properly declared in symbol table
                    if self
                        .type_checker
                        .symbol_table
                        .get_symbol(catch.exception_variable)
                        .is_none()
                    {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::UndefinedSymbol {
                                name: self.string_interner.intern("<exception_var>"),
                            },
                            location: catch.source_location,
                            context: "Exception variable not found in symbol table".to_string(),
                            suggestion: Some(
                                "This is likely an internal compiler error".to_string(),
                            ),
                        });
                    }

                    // Check optional filter expression
                    if let Some(filter_expr) = &catch.filter {
                        let filter_type = self.check_expression(filter_expr)?;
                        let bool_type = self.type_checker.type_table.borrow().bool_type();

                        let compatibility = self
                            .type_checker
                            .check_compatibility(filter_type, bool_type);
                        if matches!(compatibility, TypeCompatibility::Incompatible) {
                            self.emit_enhanced_type_error(
                                filter_type,
                                bool_type,
                                filter_expr.source_location,
                                "Catch filter must be boolean",
                                &TypeErrorContext::CatchFilter,
                            );
                        }
                    }

                    // Check catch body and get its return type
                    self.check_statement(&catch.body)?;
                }

                // Check finally block if present
                if let Some(finally) = finally_block {
                    self.check_expression(finally)?;
                }
            }
            TypedExpressionKind::While {
                condition,
                then_expr,
            } => {
                // Check condition is boolean
                let condition_type = self.check_expression(condition)?;
                let bool_type = self.type_checker.type_table.borrow().bool_type();

                let compatibility = self
                    .type_checker
                    .check_compatibility(condition_type, bool_type);
                if matches!(compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        condition_type,
                        bool_type,
                        condition.source_location,
                        "While loop condition must be boolean",
                        &TypeErrorContext::LoopCondition,
                    );
                }

                // Check loop body
                self.check_expression(then_expr)?;
            }
            TypedExpressionKind::For {
                variable,
                iterable,
                body,
            } => {
                // Check iterable type
                let iterable_type = self.check_expression(iterable)?;

                // TODO: Check that iterable_type implements Iterable interface
                // For now, check if it's an array or string
                let is_iterable = {
                    let type_table = self.type_checker.type_table.borrow();
                    if let Some(type_info) = type_table.get(iterable_type) {
                        matches!(
                            &type_info.kind,
                            super::TypeKind::Array { .. }
                                | super::TypeKind::String
                                | super::TypeKind::Dynamic
                        )
                    } else {
                        false
                    }
                };

                if !is_iterable {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: self.type_checker.type_table.borrow().dynamic_type(),
                            actual: iterable_type,
                        },
                        location: iterable.source_location,
                        context: "For loop requires an iterable type".to_string(),
                        suggestion: Some(
                            "Use an array, string, or other iterable type".to_string(),
                        ),
                    });
                }

                // Check loop body
                self.check_expression(body)?;
            }
            TypedExpressionKind::ForIn {
                value_var,
                key_var,
                iterable,
                body,
            } => {
                // Check iterable type
                let iterable_type = self.check_expression(iterable)?;

                // TODO: Check that iterable_type implements Iterable interface
                // For now, check if it's an array or string
                let is_iterable = {
                    let type_table = self.type_checker.type_table.borrow();
                    if let Some(type_info) = type_table.get(iterable_type) {
                        matches!(
                            &type_info.kind,
                            super::TypeKind::Array { .. }
                                | super::TypeKind::String
                                | super::TypeKind::Dynamic
                        )
                    } else {
                        false
                    }
                };

                if !is_iterable {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: self.type_checker.type_table.borrow().dynamic_type(),
                            actual: iterable_type,
                        },
                        location: iterable.source_location,
                        context: "For-in loop requires an iterable type".to_string(),
                        suggestion: Some(
                            "Use an array, string, or other iterable type".to_string(),
                        ),
                    });
                }

                // Check loop body
                self.check_expression(body)?;
            }
            TypedExpressionKind::Throw { expression } => {
                // Check exception expression
                let exception_type = self.check_expression(expression)?;

                // Validate that thrown type is throwable
                self.validate_throwable_type(exception_type, expression.source_location)?;
            }
            TypedExpressionKind::ObjectLiteral { fields } => {
                // Track field names to detect duplicates
                let mut field_names = std::collections::BTreeSet::new();

                // Check each field
                for field in fields {
                    // Check for duplicate field names
                    let field_name_str = self
                        .string_interner
                        .get(field.name)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("field_{}", field.name.as_raw()));

                    if field_names.contains(&field.name) {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::UndefinedType { name: field.name },
                            location: field.source_location,
                            context: format!(
                                "Duplicate field name '{}' in object literal",
                                field_name_str
                            ),
                            suggestion: Some(
                                "Field names in object literals must be unique".to_string(),
                            ),
                        });
                    }
                    field_names.insert(field.name);

                    // Check field value type
                    let field_type = self.check_expression(&field.value)?;

                    // TODO: Could validate against expected object structure if available
                    // For now, we just ensure the field expressions are valid

                    // Validate field names are valid identifiers (already handled in parsing)
                    // but we could add additional validation here if needed
                }

                // TODO: Infer object type based on field types
                // This would create an anonymous structural type
            }
            TypedExpressionKind::MapLiteral { entries } => {
                self.check_map_literal_expr(entries, expr.source_location)?;
            }
            TypedExpressionKind::ArrayLiteral { elements } => {
                // Check each element
                if !elements.is_empty() {
                    // All elements should have compatible types
                    let first_type = self.check_expression(&elements[0])?;

                    for (i, element) in elements.iter().enumerate().skip(1) {
                        let element_type = self.check_expression(element)?;
                        let compatibility = self
                            .type_checker
                            .check_compatibility(element_type, first_type);
                        if matches!(compatibility, TypeCompatibility::Incompatible) {
                            self.emit_error(TypeCheckError {
                                kind: TypeErrorKind::TypeMismatch {
                                    expected: first_type,
                                    actual: element_type,
                                },
                                location: element.source_location,
                                context: format!("Array element {} has incompatible type", i),
                                suggestion: Some(
                                    "All array elements must have compatible types".to_string(),
                                ),
                            });
                        }
                    }
                }
            }
            TypedExpressionKind::StringInterpolation { parts } => {
                // Check each interpolated expression
                for part in parts {
                    match part {
                        StringInterpolationPart::Expression(expr) => {
                            self.check_expression(expr)?;
                        }
                        StringInterpolationPart::String(_) => {
                            // String literals are always valid
                        }
                    }
                }
            }
            TypedExpressionKind::Is {
                expression,
                check_type,
            } => {
                // Check the expression
                self.check_expression(expression)?;
                // Is expression always returns boolean, no additional checking needed
            }
            TypedExpressionKind::Literal { .. }
            | TypedExpressionKind::Null
            | TypedExpressionKind::This { .. }
            | TypedExpressionKind::Super { .. }
            | TypedExpressionKind::Break
            | TypedExpressionKind::Continue => {
                // These expressions don't need additional type checking
            }
            TypedExpressionKind::Return { value } => {
                if let Some(return_expr) = value {
                    let expr_type = self.check_expression(return_expr)?;

                    // Check against expected return type
                    if let Some(&expected_return) = self.expected_return_types.last() {
                        let compatibility = self
                            .type_checker
                            .check_compatibility(expr_type, expected_return);
                        if matches!(compatibility, TypeCompatibility::Incompatible) {
                            self.emit_enhanced_type_error(
                                expr_type,
                                expected_return,
                                return_expr.source_location,
                                "Return type mismatch",
                                &TypeErrorContext::ReturnStatement {
                                    expected_type: expected_return,
                                },
                            );
                        }
                    }
                }
            }
            TypedExpressionKind::FunctionLiteral {
                parameters,
                body,
                return_type,
            } => {
                // Push expected return type for nested function
                self.expected_return_types.push(*return_type);

                // Check function body statements
                for stmt in body {
                    self.check_statement(stmt)?;
                }

                // Pop expected return type
                self.expected_return_types.pop();
            }
            TypedExpressionKind::New {
                class_type,
                arguments,
                type_arguments,
                class_name: _,
            } => {
                // Check constructor arguments
                for arg in arguments {
                    self.check_expression(arg)?;
                }

                // Check generic constraints if type arguments are provided
                if !type_arguments.is_empty() {
                    // Get the class type information
                    let type_table = self.type_checker.type_table.borrow();
                    if let Some(class_type_info) = type_table.get(*class_type) {
                        if let super::TypeKind::Class {
                            symbol_id,
                            type_args: class_type_params,
                            ..
                        } = &class_type_info.kind
                        {
                            let symbol_id = *symbol_id; // Copy the SymbolId
                                                        // Get the class definition to check its type parameter constraints
                            if let Some(_class_symbol) =
                                self.type_checker.symbol_table.get_symbol(symbol_id)
                            {
                                // Find the class definition to get its type parameters
                                if let Some(class_def) = self.find_class_by_symbol(symbol_id) {
                                    // Collect constraint violations first to avoid borrow checker issues
                                    let mut violations = Vec::new();

                                    // Validate each type argument against its constraint
                                    for (i, type_arg) in type_arguments.iter().enumerate() {
                                        if i < class_def.type_parameters.len() {
                                            let type_param = &class_def.type_parameters[i];

                                            // Check each constraint for this type parameter
                                            for constraint_type_id in &type_param.constraints {
                                                if !self.validate_type_constraint(
                                                    *type_arg,
                                                    *constraint_type_id,
                                                ) {
                                                    violations
                                                        .push((*type_arg, *constraint_type_id));
                                                }
                                            }
                                        }
                                    }

                                    // Emit errors for all violations
                                    for (type_arg, constraint_type_id) in violations {
                                        self.emit_constraint_violation(
                                            type_arg,
                                            constraint_type_id,
                                            expr.source_location,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }

                // TODO: Check constructor signature matches arguments
            }
            TypedExpressionKind::VarDeclarationExpr {
                var_type,
                initializer,
                ..
            }
            | TypedExpressionKind::FinalDeclarationExpr {
                var_type,
                initializer,
                ..
            } => {
                let init_type = self.check_expression(initializer)?;

                // Check variable type matches initializer
                let compatibility = self.type_checker.check_compatibility(init_type, *var_type);
                if matches!(compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        init_type,
                        *var_type,
                        initializer.source_location,
                        "Variable initialization type mismatch",
                        &TypeErrorContext::Initialization,
                    );
                }
            }
            TypedExpressionKind::Meta { expression, .. } => {
                // Check the wrapped expression
                self.check_expression(expression)?;
            }
            TypedExpressionKind::DollarIdent { .. }
            | TypedExpressionKind::CompilerSpecific { .. }
            | TypedExpressionKind::MacroExpression { .. } => {
                // These are compiler-specific and don't need type checking
            }
            TypedExpressionKind::PatternPlaceholder { .. } => {
                // Pattern placeholders are handled in later compilation phases
                // They have a dynamic type until resolved
            }
            TypedExpressionKind::ArrayComprehension {
                for_parts,
                expression,
                ..
            } => {
                // Check the iterator expressions
                for part in for_parts {
                    self.check_expression(&part.iterator)?;
                }
                // Check the output expression
                self.check_expression(expression)?;
            }
            TypedExpressionKind::MapComprehension {
                for_parts,
                key_expr,
                value_expr,
                ..
            } => {
                // Check the iterator expressions
                for part in for_parts {
                    self.check_expression(&part.iterator)?;
                }
                // Check the key and value expressions
                self.check_expression(key_expr)?;
                self.check_expression(value_expr)?;
            }
            TypedExpressionKind::Await {
                expression,
                await_type,
            } => {
                self.check_expression(expression)?;
                // await_type is the inner type T from Future<T>
                return Ok(*await_type);
            }
        }

        // Return the annotated type
        let result_type = expr.expr_type;

        Ok(result_type)
    }


    /// Check a binary operation expression (extracted to reduce stack frame size)
    #[inline(never)]
    pub(crate) fn check_binary_op_expr(
        &mut self,
        left: &TypedExpression,
        right: &TypedExpression,
        op: &BinaryOperator,
        source_location: SourceLocation,
    ) -> Result<(), String> {
        let lhs_type = self.check_expression(left)?;
        let rhs_type = self.check_expression(right)?;

        // OPERATOR OVERLOADING: Check if left operand has an abstract type with @:op metadata
        if let Some((method_symbol, _abstract_symbol)) = self.find_operator_method(lhs_type, op) {
            // TODO: For now, operator overloading is detected - actual rewriting will be done in AST lowering
            // The HIR lowering will automatically inline the method call
        }

        // Check operand compatibility for the operator
        match op {
            BinaryOperator::Add => {
                // Add can be either numeric addition or string concatenation
                let type_table = self.type_checker.type_table.borrow();
                let int_type = type_table.int_type();
                let float_type = type_table.float_type();
                let string_type = type_table.string_type();
                drop(type_table);

                let lhs_compat_int = self.type_checker.check_compatibility(lhs_type, int_type);
                let lhs_compat_float = self.type_checker.check_compatibility(lhs_type, float_type);
                let lhs_compat_string =
                    self.type_checker.check_compatibility(lhs_type, string_type);

                let lhs_is_numeric = matches!(
                    lhs_compat_int,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                ) || matches!(
                    lhs_compat_float,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                );
                let lhs_is_string = matches!(
                    lhs_compat_string,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                );

                let rhs_compat_int = self.type_checker.check_compatibility(rhs_type, int_type);
                let rhs_compat_float = self.type_checker.check_compatibility(rhs_type, float_type);
                let rhs_compat_string =
                    self.type_checker.check_compatibility(rhs_type, string_type);

                let rhs_is_numeric = matches!(
                    rhs_compat_int,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                ) || matches!(
                    rhs_compat_float,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                );
                let rhs_is_string = matches!(
                    rhs_compat_string,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                );

                // Check if this is valid string concatenation or numeric addition
                if lhs_is_string || rhs_is_string {
                    // String concatenation - Haxe allows implicit conversion of any type to string
                    // when concatenating with +, so this is always valid
                } else if lhs_is_numeric && rhs_is_numeric {
                    // Numeric addition - both operands are numeric, this is valid
                } else {
                    // Neither string concatenation nor numeric addition
                    self.emit_enhanced_type_error(
                        lhs_type,
                        int_type,
                        source_location,
                        "Left operand of Add must be numeric",
                        &TypeErrorContext::BinaryOperation {
                            operator: *op,
                            other_type: rhs_type,
                        },
                    );
                    self.emit_enhanced_type_error(
                        rhs_type,
                        int_type,
                        source_location,
                        "Right operand of Add must be numeric",
                        &TypeErrorContext::BinaryOperation {
                            operator: *op,
                            other_type: lhs_type,
                        },
                    );
                }
            }
            BinaryOperator::Sub | BinaryOperator::Mul | BinaryOperator::Div => {
                // Purely numeric operations
                let type_table = self.type_checker.type_table.borrow();
                let int_type = type_table.int_type();
                let float_type = type_table.float_type();
                drop(type_table);

                let lhs_compat_int = self.type_checker.check_compatibility(lhs_type, int_type);
                let lhs_compat_float = self.type_checker.check_compatibility(lhs_type, float_type);

                let is_numeric = matches!(
                    lhs_compat_int,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                ) || matches!(
                    lhs_compat_float,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                );

                if !is_numeric {
                    self.emit_enhanced_type_error(
                        lhs_type,
                        int_type,
                        source_location,
                        &format!("Left operand of {:?} must be numeric", op),
                        &TypeErrorContext::BinaryOperation {
                            operator: *op,
                            other_type: rhs_type,
                        },
                    );
                }

                // Check right operand too
                let rhs_compat_int = self.type_checker.check_compatibility(rhs_type, int_type);
                let rhs_compat_float = self.type_checker.check_compatibility(rhs_type, float_type);

                let rhs_is_numeric = matches!(
                    rhs_compat_int,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                ) || matches!(
                    rhs_compat_float,
                    TypeCompatibility::Identical | TypeCompatibility::Assignable
                );

                if !rhs_is_numeric {
                    self.emit_enhanced_type_error(
                        rhs_type,
                        int_type,
                        source_location,
                        &format!("Right operand of {:?} must be numeric", op),
                        &TypeErrorContext::BinaryOperation {
                            operator: *op,
                            other_type: lhs_type,
                        },
                    );
                }
            }
            BinaryOperator::Eq | BinaryOperator::Ne => {
                // Equality - just check types are compatible
                let compatibility = self.type_checker.check_compatibility(lhs_type, rhs_type);
                if matches!(compatibility, TypeCompatibility::Incompatible) {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: lhs_type,
                            actual: rhs_type,
                        },
                        location: source_location,
                        context: "Cannot compare incompatible types".to_string(),
                        suggestion: Some("Ensure both operands have compatible types".to_string()),
                    });
                }
            }
            _ => {
                // TODO: Handle other operators
            }
        }
        Ok(())
    }


    /// Check a function call expression (extracted to reduce stack frame size)
    #[inline(never)]
    pub(crate) fn check_function_call_expr(
        &mut self,
        function: &TypedExpression,
        arguments: &[TypedExpression],
        source_location: SourceLocation,
    ) -> Result<(), String> {
        let callee_type = self.check_expression(function)?;

        // Check argument types first
        let mut arg_types = Vec::new();
        for arg in arguments {
            let arg_type = self.check_expression(arg)?;
            arg_types.push(arg_type);
        }

        // Check function signature matches arguments
        let (param_types, is_function) = {
            let type_table = self.type_checker.type_table.borrow();
            if let Some(function_type) = type_table.get(callee_type) {
                match &function_type.kind {
                    super::TypeKind::Function { params, .. } => (params.clone(), true),
                    _ => (Vec::new(), false),
                }
            } else {
                (Vec::new(), false)
            }
        };

        if is_function {
            // Check parameter count
            if param_types.len() != arg_types.len() {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: callee_type, // Not ideal but best we can do
                        actual: callee_type,
                    },
                    location: source_location,
                    context: format!(
                        "Function expects {} arguments but {} were provided",
                        param_types.len(),
                        arg_types.len()
                    ),
                    suggestion: Some(format!(
                        "Provide exactly {} argument{}",
                        param_types.len(),
                        if param_types.len() == 1 { "" } else { "s" }
                    )),
                });
            } else {
                // Check each parameter type
                for (i, (expected_type, actual_type)) in
                    param_types.iter().zip(&arg_types).enumerate()
                {
                    let compatibility = self
                        .type_checker
                        .check_compatibility(*actual_type, *expected_type);
                    if matches!(compatibility, TypeCompatibility::Incompatible) {
                        self.emit_enhanced_type_error(
                            *actual_type,
                            *expected_type,
                            source_location,
                            &format!("Argument {} type mismatch", i + 1),
                            &TypeErrorContext::FunctionCall {
                                param_index: i,
                                expected_type: *expected_type,
                            },
                        );
                    }
                }
            }
        } else {
            // Not a function type or type not found

            // Not a function type - always report error
            {
                // Create a function type for the expected type in the error message
                let expected_function_type = {
                    let mut type_table = self.type_checker.type_table.borrow_mut();
                    // Create a generic function type (args) -> return
                    let dynamic_type = type_table.dynamic_type();
                    type_table.create_function_type(vec![], dynamic_type)
                };

                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: expected_function_type,
                        actual: callee_type,
                    },
                    location: source_location,
                    context: "Cannot call non-function type".to_string(),
                    suggestion: Some("Ensure the expression evaluates to a function".to_string()),
                });
            }
        }
        Ok(())
    }


    /// Check a method call expression (extracted to reduce stack frame size)
    #[inline(never)]
    pub(crate) fn check_method_call_expr(
        &mut self,
        receiver: &TypedExpression,
        method_symbol: SymbolId,
        arguments: &[TypedExpression],
        source_location: SourceLocation,
    ) -> Result<(), String> {
        // Check receiver type
        let receiver_type = self.check_expression(receiver)?;

        // Check argument types
        let mut arg_types = Vec::new();
        for arg in arguments {
            let arg_type = self.check_expression(arg)?;
            arg_types.push(arg_type);
        }

        // Check if method is not static (instance method call)
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(receiver_type) {
            if let super::TypeKind::Class {
                symbol_id: class_symbol,
                ..
            } = &type_info.kind
            {
                // Copy the class data to avoid borrow checker issues
                let class_symbol_copy = *class_symbol;
                drop(type_table); // Release borrow before calling method

                // Check if method is static (instance method call)
                let class_and_method_data =
                    self.find_class_by_symbol(class_symbol_copy)
                        .and_then(|class_def| {
                            class_def
                                .methods
                                .iter()
                                .find(|m| m.symbol_id == method_symbol)
                                .map(|method| {
                                    (
                                        class_def.name,
                                        class_def.symbol_id,
                                        method.name,
                                        method.is_static,
                                    )
                                })
                        });

                if let Some((class_name, class_id, method_name, is_static)) = class_and_method_data
                {
                    if is_static {
                        // Accessing static method through instance
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::StaticAccessFromInstance {
                                member_name: method_name,
                                class_name,
                            },
                            location: source_location,
                            context: "Static methods should be accessed through the class, not an instance".to_string(),
                            suggestion: Some(format!("Use {}.{} instead",
                                self.string_interner.get(class_name).unwrap_or("<class>"),
                                self.string_interner.get(method_name).unwrap_or("<method>")
                            )),
                        });
                        // Don't return early - continue checking other expressions
                    }
                }
            }
        }

        // Look up the method's function type from the symbol
        if let Some(method_info) = self.type_checker.symbol_table.get_symbol(method_symbol) {
            // Get the method's function type
            let method_type = method_info.type_id;

            // Check if it's a function type
            let (param_types, is_function) = {
                let type_table = self.type_checker.type_table.borrow();
                if let Some(function_type) = type_table.get(method_type) {
                    match &function_type.kind {
                        super::TypeKind::Function { params, .. } => (params.clone(), true),
                        _ => (Vec::new(), false),
                    }
                } else {
                    (Vec::new(), false)
                }
            };

            if is_function {
                // First try the main signature
                let main_signature_matches =
                    self.check_signature_compatibility(&param_types, &arg_types);

                if !main_signature_matches {
                    // If main signature doesn't match, try overloads
                    let overload_match_found =
                        self.check_method_overloads(method_symbol, &arg_types, source_location);

                    if !overload_match_found {
                        // No overload matched, emit error
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: method_type,
                                actual: method_type,
                            },
                            location: source_location,
                            context: format!(
                                "Method call does not match any available signature. Expected {} arguments but {} were provided",
                                param_types.len(),
                                arg_types.len()
                            ),
                            suggestion: Some("Check method signature and available overloads".to_string()),
                        });
                    }
                }
            }
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::InferenceFailed {
                    reason: format!("Method symbol not found: {:?}", method_symbol),
                },
                location: source_location,
                context: "Method call on unknown method".to_string(),
                suggestion: None,
            });
        }
        Ok(())
    }


    /// Check a switch expression (extracted to reduce stack frame size)
    #[inline(never)]
    pub(crate) fn check_switch_expr(
        &mut self,
        discriminant: &TypedExpression,
        cases: &[TypedSwitchCase],
        default_case: Option<&TypedExpression>,
        expr_type: TypeId,
        source_location: SourceLocation,
    ) -> Result<(), String> {
        let discriminant_type = self.check_expression(discriminant)?;

        // For switch expressions, collect branch types to ensure they're compatible
        let mut branch_types = Vec::new();
        let mut branch_locations = Vec::new();

        // Check each case
        for case in cases {
            // Check case value
            let pattern = &case.case_value;
            let pattern_type = self.check_expression(pattern)?;
            // Pattern type should be compatible with discriminant
            let compatibility = self
                .type_checker
                .check_compatibility(pattern_type, discriminant_type);
            if matches!(compatibility, TypeCompatibility::Incompatible) {
                self.emit_enhanced_type_error(
                    pattern_type,
                    discriminant_type,
                    pattern.source_location,
                    "Switch pattern type must match discriminant type",
                    &TypeErrorContext::SwitchPattern,
                );
            }

            // Note: TypedSwitchCase doesn't have guards in the current implementation

            // Check case body
            self.check_statement(&case.body)?;

            // For switch expressions, extract the expression type from the body
            if let TypedStatement::Expression { expression, .. } = &case.body {
                branch_types.push(expression.expr_type);
                branch_locations.push(expression.source_location);
            }
        }

        // Check default case if present
        if let Some(default) = default_case {
            let default_type = self.check_expression(default)?;
            branch_types.push(default_type);
            branch_locations.push(default.source_location);
        }

        // For switch expressions, ensure all branches have compatible types
        if !branch_types.is_empty()
            && expr_type != self.type_checker.type_table.borrow().void_type()
        {
            let expected_type = branch_types[0];
            for (i, (&branch_type, &location)) in branch_types
                .iter()
                .zip(branch_locations.iter())
                .enumerate()
                .skip(1)
            {
                let compatibility = self
                    .type_checker
                    .check_compatibility(branch_type, expected_type);
                if matches!(compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        branch_type,
                        expected_type,
                        location,
                        "Switch expression branches must have compatible types",
                        &TypeErrorContext::SwitchExpression,
                    );
                }
            }
        }
        Ok(())
    }


    /// Check a map literal expression (extracted to reduce stack frame size)
    #[inline(never)]
    pub(crate) fn check_map_literal_expr(
        &mut self,
        entries: &[TypedMapEntry],
        source_location: SourceLocation,
    ) -> Result<(), String> {
        if !entries.is_empty() {
            // Check first entry to establish expected types
            let first_key_type = self.check_expression(&entries[0].key)?;
            let first_value_type = self.check_expression(&entries[0].value)?;

            // Track duplicate keys if they're compile-time constants
            let mut constant_keys = std::collections::BTreeSet::new();

            // Check remaining entries for type consistency
            for (index, entry) in entries.iter().enumerate() {
                let key_type = self.check_expression(&entry.key)?;
                let value_type = self.check_expression(&entry.value)?;

                // Check key type consistency
                let key_compatibility = self
                    .type_checker
                    .check_compatibility(key_type, first_key_type);
                if matches!(key_compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        key_type,
                        first_key_type,
                        entry.key.source_location,
                        &format!("Map key type mismatch at index {}", index),
                        &TypeErrorContext::ArrayAccess, // Reuse context
                    );
                }

                // Check value type consistency
                let value_compatibility = self
                    .type_checker
                    .check_compatibility(value_type, first_value_type);
                if matches!(value_compatibility, TypeCompatibility::Incompatible) {
                    self.emit_enhanced_type_error(
                        value_type,
                        first_value_type,
                        entry.value.source_location,
                        &format!("Map value type mismatch at index {}", index),
                        &TypeErrorContext::ArrayAccess, // Reuse context
                    );
                }

                // Check for duplicate literal keys (strings, numbers, etc.)
                if let TypedExpressionKind::Literal { value } = &entry.key.kind {
                    let key_str = match value {
                        super::node::LiteralValue::String(s) => Some(s.clone()),
                        super::node::LiteralValue::Int(i) => Some(i.to_string()),
                        super::node::LiteralValue::Float(f) => Some(f.to_string()),
                        super::node::LiteralValue::Bool(b) => Some(b.to_string()),
                        _ => None,
                    };

                    if let Some(key_str) = key_str {
                        if constant_keys.contains(&key_str) {
                            self.emit_error(TypeCheckError {
                                kind: TypeErrorKind::UndefinedType {
                                    name: self.string_interner.intern(&key_str),
                                },
                                location: entry.key.source_location,
                                context: format!("Duplicate map key '{}'", key_str),
                                suggestion: Some("Map keys must be unique".to_string()),
                            });
                        }
                        constant_keys.insert(key_str);
                    }
                }
            }

            // TODO: Validate map key types are valid (hashable/comparable)
            // In Haxe, most types can be map keys, but some restrictions apply
        }
        Ok(())
    }
}
