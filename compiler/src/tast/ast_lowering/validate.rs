//! TAST validation.

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
    /// Validate that the lowered TAST contains all necessary information for memory safety analysis
    pub fn validate_tast(&self, typed_file: &TypedFile) -> Vec<LoweringError> {
        let mut errors = Vec::new();

        // Validate functions have proper lifetime and ownership information
        for function in &typed_file.functions {
            if function
                .parameters
                .iter()
                .any(|p| p.symbol_id == SymbolId::invalid())
            {
                errors.push(LoweringError::IncompleteImplementation {
                    feature: format!(
                        "Function parameter symbol resolution for {:?}",
                        function.name
                    ),
                    location: function.source_location,
                });
            }

            // Validate expressions in function bodies
            for statement in &function.body {
                self.validate_statement(statement, &mut errors);
            }
        }

        // Validate classes have proper field information
        for class in &typed_file.classes {
            if class.symbol_id == SymbolId::invalid() {
                errors.push(LoweringError::IncompleteImplementation {
                    feature: format!("Class symbol resolution for {:?}", class.name),
                    location: class.source_location,
                });
            }

            for field in &class.fields {
                if field.symbol_id == SymbolId::invalid() {
                    errors.push(LoweringError::IncompleteImplementation {
                        feature: format!("Field symbol resolution for {:?}", field.name),
                        location: field.source_location,
                    });
                }
            }
        }

        errors
    }

    /// Validate a statement recursively
    fn validate_statement(&self, statement: &TypedStatement, errors: &mut Vec<LoweringError>) {
        match statement {
            TypedStatement::Expression {
                expression,
                source_location,
            } => {
                self.validate_expression(expression, errors);
            }
            TypedStatement::VarDeclaration {
                symbol_id,
                source_location,
                ..
            } => {
                if *symbol_id == SymbolId::invalid() {
                    errors.push(LoweringError::IncompleteImplementation {
                        feature: "Variable declaration symbol resolution".to_string(),
                        location: *source_location,
                    });
                }
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.validate_expression(condition, errors);
                self.validate_statement(then_branch, errors);
                if let Some(else_stmt) = else_branch {
                    self.validate_statement(else_stmt, errors);
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.validate_expression(condition, errors);
                self.validate_statement(body, errors);
            }
            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                if let Some(init_stmt) = init {
                    self.validate_statement(init_stmt, errors);
                }
                if let Some(cond_expr) = condition {
                    self.validate_expression(cond_expr, errors);
                }
                if let Some(update_expr) = update {
                    self.validate_expression(update_expr, errors);
                }
                self.validate_statement(body, errors);
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.validate_statement(stmt, errors);
                }
            }
            _ => {
                // Other statement types - validate as needed
            }
        }
    }

    /// Validate an expression recursively
    fn validate_expression(&self, expression: &TypedExpression, errors: &mut Vec<LoweringError>) {
        // Check for invalid type IDs
        if expression.expr_type == TypeId::invalid() {
            errors.push(LoweringError::TypeInferenceError {
                expression: format!("{:?}", expression.kind),
                location: expression.source_location,
            });
        }

        // Check for invalid lifetime IDs
        if expression.lifetime_id == LifetimeId::invalid() {
            errors.push(LoweringError::LifetimeError {
                message: format!("Invalid lifetime for expression: {:?}", expression.kind),
                location: expression.source_location,
            });
        }

        // Validate variable references have proper symbols
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                if *symbol_id == SymbolId::invalid() {
                    errors.push(LoweringError::UnresolvedSymbol {
                        name: "unknown_variable".to_string(),
                        location: expression.source_location,
                    });
                }
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                self.validate_expression(object, errors);
                if *field_symbol == SymbolId::invalid() {
                    errors.push(LoweringError::UnresolvedSymbol {
                        name: "unknown_field".to_string(),
                        location: expression.source_location,
                    });
                }
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.validate_expression(left, errors);
                self.validate_expression(right, errors);
            }
            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.validate_expression(operand, errors);
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.validate_expression(function, errors);
                for arg in arguments {
                    self.validate_expression(arg, errors);
                }
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                self.validate_expression(array, errors);
                self.validate_expression(index, errors);
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.validate_expression(condition, errors);
                self.validate_expression(then_expr, errors);
                if let Some(else_e) = else_expr {
                    self.validate_expression(else_e, errors);
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.validate_statement(stmt, errors);
                }
            }
            _ => {
                // Other expression types - validate as needed
            }
        }
    }
}
