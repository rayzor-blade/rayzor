//! Capture analysis for closures/function literals
//!
//! This module analyzes function literals to determine which variables from
//! outer scopes are captured. This is essential for validating Send/Sync
//! constraints when closures are passed to Thread::spawn or similar functions.

use crate::tast::{
    node::{TypedExpression, TypedExpressionKind, TypedStatement},
    ScopeId, SymbolId, TypeId,
};
use std::collections::{HashMap, HashSet};

/// Information about a captured variable
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedVariable {
    /// Symbol ID of the captured variable
    pub symbol_id: SymbolId,

    /// Type of the captured variable
    pub type_id: TypeId,

    /// Whether the variable is captured by mutable reference
    pub is_mutable_capture: bool,
}

/// Result of capture analysis
#[derive(Debug, Clone)]
pub struct CaptureAnalysis {
    /// Variables captured from outer scopes
    pub captures: Vec<CapturedVariable>,
}

impl CaptureAnalysis {
    /// Create an empty capture analysis
    pub fn empty() -> Self {
        Self {
            captures: Vec::new(),
        }
    }

    /// Check if any variables are captured
    pub fn has_captures(&self) -> bool {
        !self.captures.is_empty()
    }

    /// Get all captured variable types
    pub fn captured_types(&self) -> Vec<TypeId> {
        self.captures.iter().map(|c| c.type_id).collect()
    }
}

/// Analyzer for finding captured variables in closures
pub struct CaptureAnalyzer {
    /// Scope of the closure being analyzed
    closure_scope: ScopeId,
}

impl CaptureAnalyzer {
    /// Create a new capture analyzer for a closure
    pub fn new(closure_scope: ScopeId) -> Self {
        Self { closure_scope }
    }

    /// Analyze a function literal to find captured variables
    ///
    /// This walks the function body and identifies all variable references
    /// that refer to variables defined in outer scopes (not parameters or
    /// local variables). Types are resolved from the expression's `expr_type`.
    pub fn analyze_function_literal(
        &self,
        parameters: &[crate::tast::node::TypedParameter],
        body: &[TypedStatement],
    ) -> CaptureAnalysis {
        let mut referenced_symbols: HashMap<SymbolId, TypeId> = HashMap::new();
        let mut local_symbols = HashSet::new();

        // Parameters are local to the function
        for param in parameters {
            local_symbols.insert(param.symbol_id);
        }

        // Walk the body to find all variable references with their types
        for stmt in body {
            self.collect_variable_references(stmt, &mut referenced_symbols, &mut local_symbols);
        }

        // Captured variables are those referenced but not local
        let mut captures = Vec::new();
        for (symbol_id, type_id) in referenced_symbols {
            if !local_symbols.contains(&symbol_id) {
                captures.push(CapturedVariable {
                    symbol_id,
                    type_id,
                    is_mutable_capture: false,
                });
            }
        }

        CaptureAnalysis { captures }
    }

    /// Collect all variable references from a statement
    fn collect_variable_references(
        &self,
        stmt: &TypedStatement,
        referenced: &mut HashMap<SymbolId, TypeId>,
        locals: &mut HashSet<SymbolId>,
    ) {
        match stmt {
            TypedStatement::Expression { expression, .. } => {
                self.collect_from_expression(expression, referenced, locals);
            }

            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => {
                // This declares a new local variable
                locals.insert(*symbol_id);
                if let Some(init) = initializer {
                    self.collect_from_expression(init, referenced, locals);
                }
            }

            TypedStatement::Assignment { target, value, .. } => {
                self.collect_from_expression(target, referenced, locals);
                self.collect_from_expression(value, referenced, locals);
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_from_expression(condition, referenced, locals);
                self.collect_variable_references(then_branch, referenced, locals);
                if let Some(else_stmt) = else_branch {
                    self.collect_variable_references(else_stmt, referenced, locals);
                }
            }

            TypedStatement::While {
                condition, body, ..
            } => {
                self.collect_from_expression(condition, referenced, locals);
                self.collect_variable_references(body, referenced, locals);
            }

            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                if let Some(init_stmt) = init {
                    self.collect_variable_references(init_stmt, referenced, locals);
                }
                if let Some(cond) = condition {
                    self.collect_from_expression(cond, referenced, locals);
                }
                if let Some(upd) = update {
                    self.collect_from_expression(upd, referenced, locals);
                }
                self.collect_variable_references(body, referenced, locals);
            }

            TypedStatement::ForIn {
                value_var,
                key_var,
                iterable,
                body,
                ..
            } => {
                locals.insert(*value_var);
                if let Some(key) = key_var {
                    locals.insert(*key);
                }
                self.collect_from_expression(iterable, referenced, locals);
                self.collect_variable_references(body, referenced, locals);
            }

            TypedStatement::Return { value, .. } => {
                if let Some(expr) = value {
                    self.collect_from_expression(expr, referenced, locals);
                }
            }

            TypedStatement::Throw { exception, .. } => {
                self.collect_from_expression(exception, referenced, locals);
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                self.collect_variable_references(body, referenced, locals);
                for catch_clause in catch_clauses {
                    // Catch variable is local to the catch block
                    let mut catch_locals = locals.clone();
                    catch_locals.insert(catch_clause.exception_variable);
                    self.collect_variable_references(
                        &catch_clause.body,
                        referenced,
                        &mut catch_locals,
                    );
                }
                if let Some(finally) = finally_block {
                    self.collect_variable_references(finally, referenced, locals);
                }
            }

            TypedStatement::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => {
                self.collect_from_expression(discriminant, referenced, locals);
                for case in cases {
                    self.collect_from_expression(&case.case_value, referenced, locals);
                    self.collect_variable_references(&case.body, referenced, locals);
                }
                if let Some(default) = default_case {
                    self.collect_variable_references(default, referenced, locals);
                }
            }

            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_variable_references(stmt, referenced, locals);
                }
            }

            TypedStatement::PatternMatch {
                value, patterns, ..
            } => {
                self.collect_from_expression(value, referenced, locals);
                for pattern_case in patterns {
                    // Pattern bindings are local
                    // TODO: Extract pattern bindings from pattern_case.pattern
                    self.collect_variable_references(&pattern_case.body, referenced, locals);
                }
            }

            TypedStatement::MacroExpansion { .. } => {
                // TODO: Handle macro expansions
            }

            TypedStatement::Break { .. } | TypedStatement::Continue { .. } => {
                // No variables referenced
            }
        }
    }

    /// Collect variable references from an expression
    fn collect_from_expression(
        &self,
        expr: &TypedExpression,
        referenced: &mut HashMap<SymbolId, TypeId>,
        locals: &mut HashSet<SymbolId>,
    ) {
        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                // Use the expression's type — each TypedExpression carries its resolved type
                referenced.entry(*symbol_id).or_insert(expr.expr_type);
            }

            TypedExpressionKind::FieldAccess { object, .. } => {
                self.collect_from_expression(object, referenced, locals);
            }

            TypedExpressionKind::StaticFieldAccess { .. } => {
                // Static field access doesn't capture variables
            }

            TypedExpressionKind::ArrayAccess { array, index } => {
                self.collect_from_expression(array, referenced, locals);
                self.collect_from_expression(index, referenced, locals);
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.collect_from_expression(function, referenced, locals);
                for arg in arguments {
                    self.collect_from_expression(arg, referenced, locals);
                }
            }

            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.collect_from_expression(receiver, referenced, locals);
                for arg in arguments {
                    self.collect_from_expression(arg, referenced, locals);
                }
            }

            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                for arg in arguments {
                    self.collect_from_expression(arg, referenced, locals);
                }
            }

            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.collect_from_expression(left, referenced, locals);
                self.collect_from_expression(right, referenced, locals);
            }

            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.collect_from_expression(operand, referenced, locals);
            }

            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.collect_from_expression(condition, referenced, locals);
                self.collect_from_expression(then_expr, referenced, locals);
                if let Some(else_e) = else_expr {
                    self.collect_from_expression(else_e, referenced, locals);
                }
            }

            TypedExpressionKind::ArrayLiteral { elements } => {
                for elem in elements {
                    self.collect_from_expression(elem, referenced, locals);
                }
            }

            TypedExpressionKind::ObjectLiteral { fields } => {
                for field in fields {
                    self.collect_from_expression(&field.value, referenced, locals);
                }
            }

            TypedExpressionKind::FunctionLiteral {
                parameters, body, ..
            } => {
                // Nested closure - need to track its local scope separately
                let mut nested_locals = locals.clone();
                for param in parameters {
                    nested_locals.insert(param.symbol_id);
                }
                for stmt in body {
                    self.collect_variable_references(stmt, referenced, &mut nested_locals);
                }
            }

            TypedExpressionKind::Cast { expression, .. } => {
                self.collect_from_expression(expression, referenced, locals);
            }

            TypedExpressionKind::New { arguments, .. } => {
                for arg in arguments {
                    self.collect_from_expression(arg, referenced, locals);
                }
            }

            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_variable_references(stmt, referenced, locals);
                }
            }

            TypedExpressionKind::VarDeclarationExpr {
                symbol_id,
                initializer,
                ..
            } => {
                locals.insert(*symbol_id);
                self.collect_from_expression(initializer, referenced, locals);
            }

            TypedExpressionKind::FinalDeclarationExpr {
                symbol_id,
                initializer,
                ..
            } => {
                locals.insert(*symbol_id);
                self.collect_from_expression(initializer, referenced, locals);
            }

            TypedExpressionKind::Switch {
                discriminant,
                cases,
                default_case,
            } => {
                self.collect_from_expression(discriminant, referenced, locals);
                for case in cases {
                    self.collect_from_expression(&case.case_value, referenced, locals);
                    // case.body is a TypedStatement, not an expression
                    self.collect_variable_references(&case.body, referenced, locals);
                }
                if let Some(default) = default_case {
                    self.collect_from_expression(default, referenced, locals);
                }
            }

            TypedExpressionKind::Try {
                try_expr,
                catch_clauses,
                finally_block,
            } => {
                self.collect_from_expression(try_expr, referenced, locals);
                for catch in catch_clauses {
                    let mut catch_locals = locals.clone();
                    catch_locals.insert(catch.exception_variable);
                    // catch.body is a TypedStatement, not an expression
                    self.collect_variable_references(&catch.body, referenced, &mut catch_locals);
                }
                if let Some(finally) = finally_block {
                    self.collect_from_expression(finally, referenced, locals);
                }
            }

            TypedExpressionKind::Await { expression, .. } => {
                self.collect_from_expression(expression, referenced, locals);
            }

            // Expressions that don't reference variables
            TypedExpressionKind::Literal { .. }
            | TypedExpressionKind::This { .. }
            | TypedExpressionKind::Super { .. }
            | TypedExpressionKind::Null
            | TypedExpressionKind::Break
            | TypedExpressionKind::Continue => {}

            // TODO: Handle remaining expression kinds
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_closure() {
        let analyzer = CaptureAnalyzer::new(ScopeId::from_raw(1));
        let analysis = analyzer.analyze_function_literal(&[], &[]);

        assert!(!analysis.has_captures());
        assert_eq!(analysis.captures.len(), 0);
    }

    // TODO: Add comprehensive tests with actual TypedExpression/TypedStatement instances
    // These tests would require constructing full TAST nodes which is complex
}
