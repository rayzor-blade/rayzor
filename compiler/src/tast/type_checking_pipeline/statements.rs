//! Type checking a statement.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Check a statement
    pub fn check_statement(&mut self, stmt: &TypedStatement) -> Result<(), String> {
        match stmt {
            TypedStatement::Expression {
                expression,
                source_location: _,
            } => {
                if let Err(e) = self.check_expression(expression) {
                    // Continue checking even if expression has errors
                    eprintln!("Expression error: {}", e);
                }
            }
            TypedStatement::Return {
                value,
                source_location,
            } => {
                if let Some(return_expr) = value {
                    let expr_type = match self.check_expression(return_expr) {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("Return expression error: {}", e);
                            return Ok(());
                        }
                    };

                    // Check against expected return type
                    if let Some(&expected_return) = self.expected_return_types.last() {
                        let compatibility = self
                            .type_checker
                            .check_compatibility(expr_type, expected_return);
                        match compatibility {
                            TypeCompatibility::Incompatible => {
                                self.emit_enhanced_type_error(
                                    expr_type,
                                    expected_return,
                                    *source_location,
                                    "Return type mismatch",
                                    &TypeErrorContext::ReturnStatement {
                                        expected_type: expected_return,
                                    },
                                );
                            }
                            _ => {} // Compatible
                        }
                    }
                } else {
                    // Check for void return in non-void function
                    if let Some(&expected_return) = self.expected_return_types.last() {
                        let void_type = self.type_checker.type_table.borrow().void_type();
                        if expected_return != void_type {
                            self.emit_error(TypeCheckError {
                                kind: TypeErrorKind::TypeMismatch {
                                    expected: expected_return,
                                    actual: void_type,
                                },
                                location: *source_location,
                                context: "Function must return a value".to_string(),
                                suggestion: Some(
                                    "Add a return value or change function return type to Void"
                                        .to_string(),
                                ),
                            });
                        }
                    }
                }
            }
            TypedStatement::VarDeclaration {
                var_type,
                initializer,
                source_location,
                ..
            } => {
                if let Some(init_expr) = initializer.as_ref() {
                    let init_type = self.check_expression(init_expr)?;

                    // Check variable type matches initializer
                    let compatibility = self.type_checker.check_compatibility(init_type, *var_type);
                    match compatibility {
                        TypeCompatibility::Incompatible => {
                            // Use the initializer's location for the error, not the variable declaration location
                            self.emit_enhanced_type_error(
                                init_type,
                                *var_type,
                                init_expr.source_location,
                                "Variable initialization type mismatch",
                                &TypeErrorContext::Initialization,
                            );
                        }
                        _ => {} // Compatible
                    }
                }
            }
            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                source_location,
            } => {
                // Check try block
                self.check_statement(body)?;

                // Check catch clauses
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

                    // Check catch body
                    self.check_statement(&catch.body)?;
                }

                // Check finally block if present
                if let Some(finally_stmt) = finally_block {
                    self.check_statement(finally_stmt)?;
                }
            }
            TypedStatement::Throw {
                exception,
                source_location,
            } => {
                let exception_type = self.check_expression(exception)?;

                // Validate that thrown type is throwable
                self.validate_throwable_type(exception_type, *source_location)?;
            }
            TypedStatement::While {
                condition,
                body,
                source_location: _,
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
                self.check_statement(body)?;
            }
            TypedStatement::For {
                condition, body, ..
            } => {
                // Check optional condition is boolean
                if let Some(cond_expr) = condition {
                    let condition_type = self.check_expression(cond_expr)?;
                    let bool_type = self.type_checker.type_table.borrow().bool_type();

                    let compatibility = self
                        .type_checker
                        .check_compatibility(condition_type, bool_type);
                    if matches!(compatibility, TypeCompatibility::Incompatible) {
                        self.emit_enhanced_type_error(
                            condition_type,
                            bool_type,
                            cond_expr.source_location,
                            "For loop condition must be boolean",
                            &TypeErrorContext::LoopCondition,
                        );
                    }
                }

                // Check loop body
                self.check_statement(body)?;
            }
            TypedStatement::ForIn { iterable, body, .. } => {
                // Check iterable type
                let iterable_type = self.check_expression(iterable)?;

                // Validate iterable implements Iterable interface or is a known iterable type
                self.validate_iterable_type(iterable_type, iterable.source_location)?;

                // Check loop body
                self.check_statement(body)?;
            }
            TypedStatement::Break {
                target_loop,
                source_location,
            } => {
                // TODO: Validate break is within a loop context
                // For now, just check the target loop symbol if present
                if let Some(loop_symbol) = target_loop {
                    if self
                        .type_checker
                        .symbol_table
                        .get_symbol(*loop_symbol)
                        .is_none()
                    {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::UndefinedSymbol {
                                name: self.string_interner.intern("<loop_label>"),
                            },
                            location: *source_location,
                            context: "Break target loop not found".to_string(),
                            suggestion: None,
                        });
                    }
                }
            }
            TypedStatement::Continue {
                target_loop,
                source_location,
            } => {
                // TODO: Validate continue is within a loop context
                // For now, just check the target loop symbol if present
                if let Some(loop_symbol) = target_loop {
                    if self
                        .type_checker
                        .symbol_table
                        .get_symbol(*loop_symbol)
                        .is_none()
                    {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::UndefinedSymbol {
                                name: self.string_interner.intern("<loop_label>"),
                            },
                            location: *source_location,
                            context: "Continue target loop not found".to_string(),
                            suggestion: None,
                        });
                    }
                }
            }
            TypedStatement::Block {
                statements,
                scope_id: _,
                source_location: _,
            } => {
                // Check all statements in the block
                for stmt in statements {
                    self.check_statement(stmt)?;
                }
            }
            _ => {
                // TODO: Implement remaining statement kinds (Assignment, If, Switch, etc.)
            }
        }

        Ok(())
    }
}
