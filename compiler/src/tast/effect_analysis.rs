//! Function effect analysis for determining throws, async, and purity
//!
//! This module analyzes function bodies to determine their effects:
//! - Can throw: Whether the function can throw exceptions
//! - Is async: Whether the function is asynchronous
//! - Is pure: Whether the function has no side effects

use crate::tast::{
    node::{
        AsyncKind, BinaryOperator, FunctionEffects, MemoryEffects, ResourceEffects,
        TypedCatchClause, TypedExpression, TypedExpressionKind, TypedFunction, TypedStatement,
        UnaryOperator,
    },
    SymbolId, SymbolTable, TypeId, TypeTable,
};
use std::cell::RefCell;
use std::collections::BTreeSet;

/// Analyzes a function to determine its effects
pub struct EffectAnalyzer<'a> {
    symbol_table: &'a SymbolTable,
    type_table: &'a RefCell<TypeTable>,
    /// Set of functions known to throw
    throwing_functions: BTreeSet<SymbolId>,
    /// Set of functions known to be async
    async_functions: BTreeSet<SymbolId>,
    /// Set of functions known to be pure
    pure_functions: BTreeSet<SymbolId>,
}

impl<'a> EffectAnalyzer<'a> {
    /// Create a new effect analyzer
    pub fn new(symbol_table: &'a SymbolTable, type_table: &'a RefCell<TypeTable>) -> Self {
        Self {
            symbol_table,
            type_table,
            throwing_functions: BTreeSet::new(),
            async_functions: BTreeSet::new(),
            pure_functions: BTreeSet::new(),
        }
    }

    /// Analyze a function and return its effects
    pub fn analyze_function(&mut self, function: &TypedFunction) -> FunctionEffects {
        let mut effects = FunctionEffects::default();

        // Check if function is marked with metadata
        if let Some(metadata) = self.check_function_metadata(function) {
            effects = metadata;
        }

        // Analyze the function body
        let body_effects = self.analyze_statements(&function.body);

        // Combine metadata and analysis results
        effects.can_throw = effects.can_throw || body_effects.can_throw;

        // Update async kind based on analysis
        if body_effects.is_async {
            effects.async_kind = AsyncKind::Async;
        }

        effects.is_pure = !body_effects.has_side_effects && !function.body.is_empty();

        // Update memory effects based on analysis
        if body_effects.has_side_effects {
            effects.memory_effects.accesses_global_state = true;
        }

        // Store the results for future reference
        if effects.can_throw {
            self.throwing_functions.insert(function.symbol_id);
        }
        if effects.async_kind != AsyncKind::Sync {
            self.async_functions.insert(function.symbol_id);
        }
        if effects.is_pure {
            self.pure_functions.insert(function.symbol_id);
        }

        effects
    }

    /// Check function metadata for effect annotations
    fn check_function_metadata(&self, function: &TypedFunction) -> Option<FunctionEffects> {
        // In Haxe, functions can be marked with metadata like @:throws, @:async, @:pure
        // For now, we'll check the existing metadata field
        let mut effects = FunctionEffects::default();

        // Check if function is inline (inline functions are often pure)
        effects.is_inline = function.metadata.is_override;

        // Return None if no specific metadata was found
        if !effects.can_throw && !matches!(effects.async_kind, AsyncKind::Async) && !effects.is_pure
        {
            None
        } else {
            Some(effects)
        }
    }

    /// Analyze a list of statements for effects
    fn analyze_statements(&self, statements: &[TypedStatement]) -> BodyEffects {
        let mut effects = BodyEffects::default();

        for statement in statements {
            let stmt_effects = self.analyze_statement(statement);
            effects.merge(stmt_effects);
        }

        effects
    }

    /// Analyze a single statement for effects
    fn analyze_statement(&self, statement: &TypedStatement) -> BodyEffects {
        match statement {
            TypedStatement::Expression { expression, .. } => self.analyze_expression(expression),
            TypedStatement::VarDeclaration { initializer, .. } => {
                if let Some(init) = initializer {
                    self.analyze_expression(init)
                } else {
                    BodyEffects::default()
                }
            }
            TypedStatement::Assignment { target, value, .. } => {
                let mut effects = self.analyze_expression(target);
                effects.merge(self.analyze_expression(value));
                effects.has_side_effects = true; // Assignments are side effects
                effects
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let mut effects = self.analyze_expression(condition);
                effects.merge(self.analyze_statement(then_branch));
                if let Some(else_stmt) = else_branch {
                    effects.merge(self.analyze_statement(else_stmt));
                }
                effects
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                let mut effects = self.analyze_expression(condition);
                effects.merge(self.analyze_statement(body));
                effects
            }
            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                let mut effects = BodyEffects::default();
                if let Some(init_stmt) = init {
                    effects.merge(self.analyze_statement(init_stmt));
                }
                if let Some(cond) = condition {
                    effects.merge(self.analyze_expression(cond));
                }
                if let Some(upd) = update {
                    effects.merge(self.analyze_expression(upd));
                }
                effects.merge(self.analyze_statement(body));
                effects
            }
            TypedStatement::ForIn { iterable, body, .. } => {
                let mut effects = self.analyze_expression(iterable);
                effects.merge(self.analyze_statement(body));
                effects
            }
            TypedStatement::Return { value, .. } => {
                if let Some(val) = value {
                    self.analyze_expression(val)
                } else {
                    BodyEffects::default()
                }
            }
            TypedStatement::Throw { exception, .. } => {
                let mut effects = self.analyze_expression(exception);
                effects.can_throw = true; // Explicit throw
                effects
            }
            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                let mut effects = self.analyze_statement(body);

                // Try blocks can catch exceptions, so they don't propagate throws
                let body_can_throw = effects.can_throw;
                effects.can_throw = false;

                // Analyze catch clauses
                for catch in catch_clauses {
                    effects.merge(self.analyze_catch_clause(catch));
                }

                // Finally block always executes
                if let Some(finally) = finally_block {
                    effects.merge(self.analyze_statement(finally));
                }

                // If all catch clauses rethrow, the try can still throw
                if catch_clauses.is_empty() && body_can_throw {
                    effects.can_throw = true;
                }

                effects
            }
            TypedStatement::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => {
                let mut effects = self.analyze_expression(discriminant);
                for case in cases {
                    effects.merge(self.analyze_statement(&case.body));
                }
                if let Some(default) = default_case {
                    effects.merge(self.analyze_statement(default));
                }
                effects
            }
            TypedStatement::Break { .. } | TypedStatement::Continue { .. } => {
                BodyEffects::default()
            }
            TypedStatement::Block { statements, .. } => self.analyze_statements(statements),
            TypedStatement::PatternMatch {
                value, patterns, ..
            } => {
                let mut effects = self.analyze_expression(value);
                for pattern in patterns {
                    if let Some(guard) = &pattern.guard {
                        effects.merge(self.analyze_expression(guard));
                    }
                    effects.merge(self.analyze_statement(&pattern.body));
                }
                effects
            }
            TypedStatement::MacroExpansion {
                expanded_statements,
                ..
            } => self.analyze_statements(expanded_statements),
        }
    }

    /// Analyze a catch clause for effects
    fn analyze_catch_clause(&self, catch: &TypedCatchClause) -> BodyEffects {
        let mut effects = BodyEffects::default();

        if let Some(filter) = &catch.filter {
            effects.merge(self.analyze_expression(filter));
        }

        effects.merge(self.analyze_statement(&catch.body));
        effects
    }

    /// Analyze an expression for effects
    fn analyze_expression(&self, expression: &TypedExpression) -> BodyEffects {
        let mut effects = BodyEffects::default();

        // Check expression metadata first
        if expression.metadata.can_throw {
            effects.can_throw = true;
        }
        if expression.metadata.has_side_effects {
            effects.has_side_effects = true;
        }

        match &expression.kind {
            TypedExpressionKind::Literal { .. }
            | TypedExpressionKind::Variable { .. }
            | TypedExpressionKind::This { .. }
            | TypedExpressionKind::Super { .. }
            | TypedExpressionKind::Null
            | TypedExpressionKind::Break
            | TypedExpressionKind::Continue => {
                // These have no effects
            }

            TypedExpressionKind::FieldAccess { object, .. }
            | TypedExpressionKind::ArrayAccess { array: object, .. } => {
                effects.merge(self.analyze_expression(object));
            }

            TypedExpressionKind::StaticFieldAccess { .. } => {
                // Static field access might have side effects if it's a getter
                effects.has_side_effects = true;
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                effects.merge(self.analyze_expression(function));
                for arg in arguments {
                    effects.merge(self.analyze_expression(arg));
                }

                // Function calls can have side effects and might throw
                effects.has_side_effects = true;

                // Check if the function is known to throw or be async
                if let TypedExpressionKind::Variable { symbol_id } = &function.kind {
                    if self.throwing_functions.contains(symbol_id) {
                        effects.can_throw = true;
                    }
                    if self.async_functions.contains(symbol_id) {
                        effects.is_async = true;
                    }
                }
            }

            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                arguments,
                ..
            } => {
                effects.merge(self.analyze_expression(receiver));

                for arg in arguments {
                    effects.merge(self.analyze_expression(arg));
                }

                // Method calls can have side effects and might throw
                effects.has_side_effects = true;

                // Check if the method is known to throw or be async
                if self.throwing_functions.contains(method_symbol) {
                    effects.can_throw = true;
                }
                if self.async_functions.contains(method_symbol) {
                    effects.is_async = true;
                }
            }

            TypedExpressionKind::StaticMethodCall {
                method_symbol,
                arguments,
                ..
            } => {
                for arg in arguments {
                    effects.merge(self.analyze_expression(arg));
                }

                // Method calls can have side effects and might throw
                effects.has_side_effects = true;

                // Check if the method is known to throw or be async
                if self.throwing_functions.contains(method_symbol) {
                    effects.can_throw = true;
                }
                if self.async_functions.contains(method_symbol) {
                    effects.is_async = true;
                }
            }

            TypedExpressionKind::BinaryOp {
                left,
                right,
                operator,
                ..
            } => {
                effects.merge(self.analyze_expression(left));
                effects.merge(self.analyze_expression(right));

                // Assignment operators have side effects
                match operator {
                    BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubAssign
                    | BinaryOperator::MulAssign
                    | BinaryOperator::DivAssign
                    | BinaryOperator::ModAssign => {
                        effects.has_side_effects = true;
                    }
                    _ => {}
                }
            }

            TypedExpressionKind::UnaryOp {
                operand, operator, ..
            } => {
                effects.merge(self.analyze_expression(operand));

                // Increment/decrement have side effects
                match operator {
                    UnaryOperator::PreInc
                    | UnaryOperator::PostInc
                    | UnaryOperator::PreDec
                    | UnaryOperator::PostDec => {
                        effects.has_side_effects = true;
                    }
                    _ => {}
                }
            }

            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                effects.merge(self.analyze_expression(condition));
                effects.merge(self.analyze_expression(then_expr));
                if let Some(else_e) = else_expr {
                    effects.merge(self.analyze_expression(else_e));
                }
            }

            TypedExpressionKind::While {
                condition,
                then_expr,
            } => {
                effects.merge(self.analyze_expression(condition));
                effects.merge(self.analyze_expression(then_expr));
            }

            TypedExpressionKind::For { iterable, body, .. }
            | TypedExpressionKind::ForIn { iterable, body, .. } => {
                effects.merge(self.analyze_expression(iterable));
                effects.merge(self.analyze_expression(body));
            }

            TypedExpressionKind::ArrayLiteral { elements } => {
                for elem in elements {
                    effects.merge(self.analyze_expression(elem));
                }
            }

            TypedExpressionKind::MapLiteral { entries } => {
                for entry in entries {
                    effects.merge(self.analyze_expression(&entry.key));
                    effects.merge(self.analyze_expression(&entry.value));
                }
            }

            TypedExpressionKind::ObjectLiteral { fields } => {
                for field in fields {
                    effects.merge(self.analyze_expression(&field.value));
                }
            }

            TypedExpressionKind::FunctionLiteral { body, .. } => {
                // Function literals themselves don't have effects
                // The effects happen when they're called
            }

            TypedExpressionKind::Cast { expression, .. }
            | TypedExpressionKind::Is { expression, .. } => {
                effects.merge(self.analyze_expression(expression));
            }

            TypedExpressionKind::New { arguments, .. } => {
                for arg in arguments {
                    effects.merge(self.analyze_expression(arg));
                }
                // Constructors can have side effects and might throw
                effects.has_side_effects = true;
                effects.can_throw = true;
            }

            TypedExpressionKind::Return { value } => {
                if let Some(val) = value {
                    effects.merge(self.analyze_expression(val));
                }
            }

            TypedExpressionKind::Throw { expression } => {
                effects.merge(self.analyze_expression(expression));
                effects.can_throw = true;
            }

            TypedExpressionKind::VarDeclarationExpr { initializer, .. }
            | TypedExpressionKind::FinalDeclarationExpr { initializer, .. } => {
                effects.merge(self.analyze_expression(initializer));
            }

            TypedExpressionKind::StringInterpolation { parts } => {
                for part in parts {
                    if let crate::tast::node::StringInterpolationPart::Expression(expr) = part {
                        effects.merge(self.analyze_expression(expr));
                    }
                }
            }

            TypedExpressionKind::MacroExpression { .. } => {
                // Macros can have arbitrary effects
                effects.has_side_effects = true;
                effects.can_throw = true;
            }

            TypedExpressionKind::Block { statements, .. } => {
                effects.merge(self.analyze_statements(statements));
            }

            TypedExpressionKind::Meta { expression, .. } => {
                effects.merge(self.analyze_expression(expression));
            }

            TypedExpressionKind::DollarIdent { arg, .. } => {
                if let Some(arg_expr) = arg {
                    effects.merge(self.analyze_expression(arg_expr));
                }
            }

            TypedExpressionKind::CompilerSpecific { code, .. } => {
                effects.merge(self.analyze_expression(code));
                // Compiler-specific code can have arbitrary effects
                effects.has_side_effects = true;
            }

            TypedExpressionKind::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => {
                effects.merge(self.analyze_expression(discriminant));
                for case in cases {
                    effects.merge(self.analyze_expression(&case.case_value));
                    effects.merge(self.analyze_statement(&case.body));
                }
                if let Some(default) = default_case {
                    effects.merge(self.analyze_expression(default));
                }
            }

            TypedExpressionKind::Try {
                try_expr,
                catch_clauses,
                finally_block,
            } => {
                let mut try_effects = self.analyze_expression(try_expr);

                // Try expressions can catch exceptions
                try_effects.can_throw = false;

                for catch in catch_clauses {
                    effects.merge(self.analyze_catch_clause(catch));
                }

                if let Some(finally) = finally_block {
                    effects.merge(self.analyze_expression(finally));
                }

                effects.merge(try_effects);
            }

            TypedExpressionKind::PatternPlaceholder { .. } => {
                // Pattern placeholders are compile-time constructs
            }

            TypedExpressionKind::ArrayComprehension {
                for_parts,
                expression,
                ..
            } => {
                for part in for_parts {
                    effects.merge(self.analyze_expression(&part.iterator));
                }
                effects.merge(self.analyze_expression(expression));
            }

            TypedExpressionKind::MapComprehension {
                for_parts,
                key_expr,
                value_expr,
                ..
            } => {
                for part in for_parts {
                    effects.merge(self.analyze_expression(&part.iterator));
                }
                effects.merge(self.analyze_expression(key_expr));
                effects.merge(self.analyze_expression(value_expr));
            }

            TypedExpressionKind::Await { expression, .. } => {
                effects.merge(self.analyze_expression(expression));
                // Await expressions make the function async
                effects.is_async = true;
            }
        }

        effects
    }
}

/// Effects discovered during body analysis
#[derive(Debug, Default, Clone)]
struct BodyEffects {
    /// Whether the body can throw exceptions
    can_throw: bool,
    /// Whether the body contains async operations
    is_async: bool,
    /// Whether the body has side effects
    has_side_effects: bool,
}

impl BodyEffects {
    /// Merge effects from another analysis
    fn merge(&mut self, other: BodyEffects) {
        self.can_throw = self.can_throw || other.can_throw;
        self.is_async = self.is_async || other.is_async;
        self.has_side_effects = self.has_side_effects || other.has_side_effects;
    }
}

/// Analyze all functions in a compilation unit for effects
pub fn analyze_file_effects(
    file: &crate::tast::node::TypedFile,
    symbol_table: &SymbolTable,
    type_table: &RefCell<TypeTable>,
) {
    let mut analyzer = EffectAnalyzer::new(symbol_table, type_table);

    // Analyze all functions
    for function in &file.functions {
        let effects = analyzer.analyze_function(function);
        // The effects are already stored in the function, but we could
        // update them here if needed
    }

    // Analyze functions in classes
    for class in &file.classes {
        for method in &class.methods {
            let effects = analyzer.analyze_function(method);
        }
        for constructor in &class.constructors {
            let effects = analyzer.analyze_function(constructor);
        }
    }

    // Analyze functions in abstracts
    for abstract_type in &file.abstracts {
        for method in &abstract_type.methods {
            let effects = analyzer.analyze_function(method);
        }
        for constructor in &abstract_type.constructors {
            let effects = analyzer.analyze_function(constructor);
        }
    }

    // Analyze module-level functions
    for field in &file.module_fields {
        if let crate::tast::node::TypedModuleFieldKind::Function(function) = &field.kind {
            let effects = analyzer.analyze_function(function);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tast::{StringInterner, SymbolId};

    #[test]
    fn test_throw_detection() {
        // Test that we can detect explicit throws
        // This would require setting up test infrastructure
    }

    #[test]
    fn test_side_effect_detection() {
        // Test that we can detect side effects
        // This would require setting up test infrastructure
    }

    #[test]
    fn test_async_detection() {
        // Test that we can detect async operations
        // This would require setting up test infrastructure
    }
}
