//! Null safety analysis for preventing null pointer exceptions
//!
//! This module provides comprehensive null safety analysis by tracking nullable values
//! through control flow and detecting potential null dereferences.

use crate::tast::{
    control_flow_analysis::{BlockId, ControlFlowGraph, VariableState},
    core::TypeKind,
    node::{BinaryOperator, TypedExpression, TypedExpressionKind, TypedFunction, TypedStatement},
    symbols::SymbolFlags,
    SourceLocation, SymbolId, SymbolTable, TypeId, TypeTable,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Null state of a variable or expression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullState {
    /// Definitely null
    Null,
    /// Definitely not null
    NotNull,
    /// Maybe null (unknown nullability)
    MaybeNull,
    /// Uninitialized (treated as potentially null)
    Uninitialized,
}

/// Information about a null check in the code
#[derive(Debug, Clone)]
pub struct NullCheck {
    /// Variable being checked
    pub variable: SymbolId,
    /// Whether this is a null check (x == null) or non-null check (x != null)
    pub is_null_check: bool,
    /// Location of the check
    pub location: SourceLocation,
}

/// Null safety violation
#[derive(Debug, Clone)]
pub struct NullSafetyViolation {
    /// Variable that might be null
    pub variable: SymbolId,
    /// Type of violation
    pub violation_kind: NullViolationKind,
    /// Location where violation occurs
    pub location: SourceLocation,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Types of null safety violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NullViolationKind {
    /// Dereferencing a potentially null value
    PotentialNullDereference,
    /// Calling method on potentially null value
    PotentialNullMethodCall,
    /// Accessing field of potentially null value
    PotentialNullFieldAccess,
    /// Array access on potentially null array
    PotentialNullArrayAccess,
    /// Passing null to non-nullable parameter
    NullArgumentToNonNullable,
    /// Returning null from non-nullable function
    NullReturnFromNonNullable,
    /// Assigning null literal to @:notNull variable
    NullAssignedToNotNull,
    /// Assigning potentially-null value to @:notNull variable
    NullableAssignedToNotNull,
}

/// Null safety analyzer
pub struct NullSafetyAnalyzer<'a> {
    /// Type table for checking nullable types
    type_table: &'a RefCell<TypeTable>,
    /// Symbol table for variable information
    symbol_table: &'a SymbolTable,
    /// Control flow graph
    cfg: &'a ControlFlowGraph,
    /// Null states for each variable at each program point
    null_states: HashMap<BlockId, HashMap<SymbolId, NullState>>,
    /// Detected violations
    violations: Vec<NullSafetyViolation>,
    /// Null checks found in the code
    null_checks: HashMap<BlockId, Vec<NullCheck>>,
    /// Scoped null narrowing stack — branch-sensitive refinements.
    /// Each entry maps a variable to its narrowed NullState within the current scope.
    narrowing_stack: Vec<HashMap<SymbolId, NullState>>,
}

impl<'a> NullSafetyAnalyzer<'a> {
    /// Create a new null safety analyzer
    pub fn new(
        type_table: &'a RefCell<TypeTable>,
        symbol_table: &'a SymbolTable,
        cfg: &'a ControlFlowGraph,
    ) -> Self {
        Self {
            type_table,
            symbol_table,
            cfg,
            null_states: HashMap::new(),
            violations: Vec::new(),
            null_checks: HashMap::new(),
            narrowing_stack: Vec::new(),
        }
    }

    /// Analyze null safety for a function
    pub fn analyze_function(&mut self, function: &TypedFunction) -> Vec<NullSafetyViolation> {
        // Initialize null states for function parameters
        self.initialize_parameter_states(function);

        // Find all null checks in the function
        self.find_null_checks(function);

        // Perform data flow analysis to track null states
        self.analyze_null_flow();

        // Check for violations
        self.check_violations(function);

        std::mem::take(&mut self.violations)
    }

    /// Check if a symbol has @:notNull metadata
    fn is_symbol_not_null(&self, symbol_id: SymbolId) -> bool {
        if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
            sym.flags.contains(SymbolFlags::NOT_NULL)
        } else {
            false
        }
    }

    /// Initialize null states for function parameters
    fn initialize_parameter_states(&mut self, function: &TypedFunction) {
        let entry_block = self.cfg.entry_block;

        // Collect parameter states first
        let mut param_states = HashMap::new();
        for param in &function.parameters {
            let null_state = if self.is_symbol_not_null(param.symbol_id) {
                // @:notNull parameters are guaranteed non-null
                NullState::NotNull
            } else if self.is_nullable_type(param.param_type) {
                NullState::MaybeNull
            } else {
                NullState::NotNull
            };
            param_states.insert(param.symbol_id, null_state);
        }

        // Then insert them into null_states
        let entry_states = self
            .null_states
            .entry(entry_block)
            .or_insert_with(HashMap::new);
        entry_states.extend(param_states);
    }

    /// Check if a type is nullable, considering @:notNull flags
    fn is_nullable_type(&self, type_id: TypeId) -> bool {
        let type_table = self.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            // @:notNull types are never nullable
            if type_info.flags.is_non_null {
                return false;
            }
            match &type_info.kind {
                TypeKind::Optional { .. } => true,
                TypeKind::Dynamic => true, // Dynamic is always potentially null
                TypeKind::Class { .. } => true, // Class instances can be null in Haxe
                TypeKind::Interface { .. } => true,
                TypeKind::Array { .. } => true, // Arrays can be null
                TypeKind::Map { .. } => true,   // Maps can be null
                TypeKind::Function { .. } => true, // Functions can be null
                // Primitives are generally not nullable unless optional
                TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String => false,
                _ => false,
            }
        } else {
            true // Unknown types are assumed nullable for safety
        }
    }

    /// Find all null checks in the function
    fn find_null_checks(&mut self, function: &TypedFunction) {
        for statement in &function.body {
            self.find_null_checks_in_statement(statement);
        }
    }

    /// Find null checks in a statement
    fn find_null_checks_in_statement(&mut self, statement: &TypedStatement) {
        match statement {
            TypedStatement::Expression { expression, .. } => {
                self.find_null_checks_in_expression(expression);
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.find_null_checks_in_expression(condition);
                self.find_null_checks_in_statement(then_branch);
                if let Some(else_stmt) = else_branch {
                    self.find_null_checks_in_statement(else_stmt);
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.find_null_checks_in_expression(condition);
                self.find_null_checks_in_statement(body);
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.find_null_checks_in_statement(stmt);
                }
            }
            _ => {
                // Handle other statement types
            }
        }
    }

    /// Find null checks in an expression
    fn find_null_checks_in_expression(&mut self, expression: &TypedExpression) {
        match &expression.kind {
            TypedExpressionKind::BinaryOp {
                left,
                right,
                operator,
            } => {
                // Look for null equality checks: x == null, x != null
                if matches!(operator, BinaryOperator::Eq | BinaryOperator::Ne) {
                    if let (Some(var_id), true) = (
                        self.extract_variable_from_expression(left),
                        self.is_null_literal(right),
                    ) {
                        let null_check = NullCheck {
                            variable: var_id,
                            is_null_check: matches!(operator, BinaryOperator::Eq),
                            location: expression.source_location,
                        };

                        // For now, we associate null checks with the entry block
                        // A more sophisticated implementation would track which block the check is in
                        self.null_checks
                            .entry(self.cfg.entry_block)
                            .or_insert_with(Vec::new)
                            .push(null_check);
                    } else if let (Some(var_id), true) = (
                        self.extract_variable_from_expression(right),
                        self.is_null_literal(left),
                    ) {
                        let null_check = NullCheck {
                            variable: var_id,
                            is_null_check: matches!(operator, BinaryOperator::Eq),
                            location: expression.source_location,
                        };

                        self.null_checks
                            .entry(self.cfg.entry_block)
                            .or_insert_with(Vec::new)
                            .push(null_check);
                    }
                }

                // Recursively check operands
                self.find_null_checks_in_expression(left);
                self.find_null_checks_in_expression(right);
            }

            TypedExpressionKind::FieldAccess { object, .. } => {
                self.find_null_checks_in_expression(object);
            }

            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.find_null_checks_in_expression(receiver);
                for arg in arguments {
                    self.find_null_checks_in_expression(arg);
                }
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.find_null_checks_in_expression(function);
                for arg in arguments {
                    self.find_null_checks_in_expression(arg);
                }
            }

            _ => {
                // Handle other expression types
            }
        }
    }

    /// Check if an expression is a null literal
    fn is_null_literal(&self, expression: &TypedExpression) -> bool {
        matches!(expression.kind, TypedExpressionKind::Null)
    }

    /// Extract variable symbol from expression
    fn extract_variable_from_expression(&self, expression: &TypedExpression) -> Option<SymbolId> {
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => Some(*symbol_id),
            _ => None,
        }
    }

    /// Perform data flow analysis to track null states
    fn analyze_null_flow(&mut self) {
        // Iterative data flow analysis
        let mut changed = true;
        while changed {
            changed = false;

            let block_ids: Vec<_> = self.cfg.blocks.keys().copied().collect();
            for block_id in block_ids {
                if self.update_null_states(block_id) {
                    changed = true;
                }
            }
        }
    }

    /// Update null states for a block
    fn update_null_states(&mut self, block_id: BlockId) -> bool {
        let mut changed = false;

        // Get predecessors and merge their exit states
        if let Some(block) = self.cfg.blocks.get(&block_id) {
            let predecessors = block.predecessors.clone();
            let mut merged_states = HashMap::new();

            for &pred_id in &predecessors {
                if let Some(pred_states) = self.null_states.get(&pred_id) {
                    for (&var_id, &state) in pred_states {
                        let current_state = merged_states
                            .get(&var_id)
                            .cloned()
                            .unwrap_or(NullState::Uninitialized);
                        let merged_state = self.merge_null_states(current_state, state);
                        merged_states.insert(var_id, merged_state);
                    }
                }
            }

            // Apply null checks in this block
            if let Some(checks) = self.null_checks.get(&block_id) {
                for check in checks {
                    if check.is_null_check {
                        merged_states.insert(check.variable, NullState::Null);
                    } else {
                        merged_states.insert(check.variable, NullState::NotNull);
                    }
                }
            }

            // Check if states changed
            if self.null_states.get(&block_id) != Some(&merged_states) {
                self.null_states.insert(block_id, merged_states);
                changed = true;
            }
        }

        changed
    }

    /// Merge two null states
    fn merge_null_states(&self, state1: NullState, state2: NullState) -> NullState {
        match (state1, state2) {
            (NullState::Null, NullState::Null) => NullState::Null,
            (NullState::NotNull, NullState::NotNull) => NullState::NotNull,
            (NullState::Uninitialized, other) | (other, NullState::Uninitialized) => other,
            _ => NullState::MaybeNull, // Any mismatch becomes maybe null
        }
    }

    /// Check for null safety violations
    fn check_violations(&mut self, function: &TypedFunction) {
        for statement in &function.body {
            self.check_violations_in_statement(statement, function);
        }
    }

    /// Check for violations in a statement
    fn check_violations_in_statement(
        &mut self,
        statement: &TypedStatement,
        function: &TypedFunction,
    ) {
        match statement {
            TypedStatement::Expression { expression, .. } => {
                self.check_violations_in_expression(expression);
            }
            TypedStatement::Assignment { target, value, .. } => {
                // Check for null assignment to @:notNull variable
                if let Some(var_id) = self.extract_variable_from_expression(target) {
                    if self.is_symbol_not_null(var_id) {
                        if self.is_null_literal(value) {
                            self.violations.push(NullSafetyViolation {
                                variable: var_id,
                                violation_kind: NullViolationKind::NullAssignedToNotNull,
                                location: value.source_location,
                                suggestion: Some(
                                    "Cannot assign null to @:notNull variable".to_string(),
                                ),
                            });
                        } else if let Some(rhs_var) = self.extract_variable_from_expression(value) {
                            if self.is_potentially_null(rhs_var) {
                                self.violations.push(NullSafetyViolation {
                                    variable: var_id,
                                    violation_kind: NullViolationKind::NullableAssignedToNotNull,
                                    location: value.source_location,
                                    suggestion: Some(
                                        "Add null check before assigning to @:notNull variable"
                                            .to_string(),
                                    ),
                                });
                            }
                        }
                    }
                }
                self.check_violations_in_expression(target);
                self.check_violations_in_expression(value);
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.check_violations_in_expression(condition);

                // Branch-sensitive null narrowing:
                // if (x != null) { then: x is NotNull } else { else: x is Null }
                // if (x == null) { then: x is Null } else { else: x is NotNull }
                if let Some((var_id, is_not_null_check)) =
                    self.extract_null_check_from_condition(condition)
                {
                    // then-branch narrowing
                    let then_state = if is_not_null_check {
                        NullState::NotNull
                    } else {
                        NullState::Null
                    };
                    let mut then_refinements = HashMap::new();
                    then_refinements.insert(var_id, then_state);
                    self.push_narrowing(then_refinements);
                    self.check_violations_in_statement(then_branch, function);
                    self.pop_narrowing();

                    // else-branch narrowing (inverse)
                    if let Some(else_stmt) = else_branch {
                        let else_state = if is_not_null_check {
                            NullState::Null
                        } else {
                            NullState::NotNull
                        };
                        let mut else_refinements = HashMap::new();
                        else_refinements.insert(var_id, else_state);
                        self.push_narrowing(else_refinements);
                        self.check_violations_in_statement(else_stmt, function);
                        self.pop_narrowing();
                    }
                } else {
                    // No null check in condition — analyze branches without narrowing
                    self.check_violations_in_statement(then_branch, function);
                    if let Some(else_stmt) = else_branch {
                        self.check_violations_in_statement(else_stmt, function);
                    }
                }
            }
            TypedStatement::While {
                condition, body, ..
            } => {
                self.check_violations_in_expression(condition);
                self.check_violations_in_statement(body, function);
            }
            TypedStatement::Return { value, .. } => {
                if let Some(val) = value {
                    self.check_violations_in_expression(val);

                    // Check if returning null from non-nullable function
                    if self.is_null_literal(val) && !self.is_nullable_type(function.return_type) {
                        self.violations.push(NullSafetyViolation {
                            variable: SymbolId::from_raw(0), // Use dummy variable for return
                            violation_kind: NullViolationKind::NullReturnFromNonNullable,
                            location: val.source_location,
                            suggestion: Some(
                                "Change return type to optional or return a non-null value"
                                    .to_string(),
                            ),
                        });
                    }
                }
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.check_violations_in_statement(stmt, function);
                }
            }
            _ => {}
        }
    }

    /// Check for violations in an expression
    fn check_violations_in_expression(&mut self, expression: &TypedExpression) {
        match &expression.kind {
            TypedExpressionKind::FieldAccess { object, .. } => {
                if let Some(var_id) = self.extract_variable_from_expression(object) {
                    if self.is_potentially_null(var_id) {
                        self.violations.push(NullSafetyViolation {
                            variable: var_id,
                            violation_kind: NullViolationKind::PotentialNullFieldAccess,
                            location: expression.source_location,
                            suggestion: Some("Add null check before field access".to_string()),
                        });
                    }
                }
                self.check_violations_in_expression(object);
            }

            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                if let Some(var_id) = self.extract_variable_from_expression(receiver) {
                    if self.is_potentially_null(var_id) {
                        self.violations.push(NullSafetyViolation {
                            variable: var_id,
                            violation_kind: NullViolationKind::PotentialNullMethodCall,
                            location: expression.source_location,
                            suggestion: Some("Add null check before method call".to_string()),
                        });
                    }
                }
                self.check_violations_in_expression(receiver);
                for arg in arguments {
                    self.check_violations_in_expression(arg);
                }
            }

            TypedExpressionKind::ArrayAccess { array, index } => {
                if let Some(var_id) = self.extract_variable_from_expression(array) {
                    if self.is_potentially_null(var_id) {
                        self.violations.push(NullSafetyViolation {
                            variable: var_id,
                            violation_kind: NullViolationKind::PotentialNullArrayAccess,
                            location: expression.source_location,
                            suggestion: Some("Add null check before array access".to_string()),
                        });
                    }
                }
                self.check_violations_in_expression(array);
                self.check_violations_in_expression(index);
            }

            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.check_violations_in_expression(left);
                self.check_violations_in_expression(right);
            }

            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.check_violations_in_expression(operand);
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.check_violations_in_expression(function);
                for arg in arguments {
                    self.check_violations_in_expression(arg);
                }
            }

            _ => {
                // Handle other expression types
            }
        }
    }

    /// Push a narrowing scope with refined null states
    fn push_narrowing(&mut self, refinements: HashMap<SymbolId, NullState>) {
        self.narrowing_stack.push(refinements);
    }

    /// Pop the current narrowing scope
    fn pop_narrowing(&mut self) {
        self.narrowing_stack.pop();
    }

    /// Get the narrowed state for a variable, checking the narrowing stack top-down
    fn get_narrowed_state(&self, var_id: SymbolId) -> Option<NullState> {
        for scope in self.narrowing_stack.iter().rev() {
            if let Some(&state) = scope.get(&var_id) {
                return Some(state);
            }
        }
        None
    }

    /// Extract null check info from a condition expression.
    /// Returns (variable, is_not_null_check) — true if `x != null`, false if `x == null`.
    fn extract_null_check_from_condition(
        &self,
        condition: &TypedExpression,
    ) -> Option<(SymbolId, bool)> {
        match &condition.kind {
            TypedExpressionKind::BinaryOp {
                left,
                right,
                operator,
            } if matches!(operator, BinaryOperator::Eq | BinaryOperator::Ne) => {
                let is_ne = matches!(operator, BinaryOperator::Ne);
                // x != null or x == null
                if let (Some(var_id), true) = (
                    self.extract_variable_from_expression(left),
                    self.is_null_literal(right),
                ) {
                    return Some((var_id, is_ne));
                }
                // null != x or null == x
                if let (Some(var_id), true) = (
                    self.extract_variable_from_expression(right),
                    self.is_null_literal(left),
                ) {
                    return Some((var_id, is_ne));
                }
                None
            }
            _ => None,
        }
    }

    /// Check if a variable is potentially null, respecting narrowing scopes
    fn is_potentially_null(&self, var_id: SymbolId) -> bool {
        // First check narrowing stack — branch-sensitive refinements take priority
        if let Some(narrowed) = self.get_narrowed_state(var_id) {
            return matches!(
                narrowed,
                NullState::Null | NullState::MaybeNull | NullState::Uninitialized
            );
        }

        // Then check @:notNull flag on the symbol
        if self.is_symbol_not_null(var_id) {
            return false;
        }

        // Check across all blocks (simplified - should check current block context)
        for states in self.null_states.values() {
            if let Some(state) = states.get(&var_id) {
                match state {
                    NullState::Null | NullState::MaybeNull | NullState::Uninitialized => {
                        return true
                    }
                    NullState::NotNull => return false,
                }
            }
        }

        // If not found, assume potentially null for safety
        true
    }
}

/// Perform null safety analysis on a function
pub fn analyze_function_null_safety(
    function: &TypedFunction,
    cfg: &ControlFlowGraph,
    type_table: &RefCell<TypeTable>,
    symbol_table: &SymbolTable,
) -> Vec<NullSafetyViolation> {
    let mut analyzer = NullSafetyAnalyzer::new(type_table, symbol_table, cfg);
    analyzer.analyze_function(function)
}

/// Generate suggested fixes for null safety violations
pub fn suggest_null_safety_fixes(violations: &[NullSafetyViolation]) -> Vec<String> {
    let mut suggestions = Vec::new();

    for violation in violations {
        let suggestion = match violation.violation_kind {
            NullViolationKind::PotentialNullDereference => {
                format!("Add null check: if (variable != null) {{ /* safe access */ }}")
            }
            NullViolationKind::PotentialNullMethodCall => {
                format!("Use safe navigation: variable?.method() or add null check")
            }
            NullViolationKind::PotentialNullFieldAccess => {
                format!("Use safe navigation: variable?.field or add null check")
            }
            NullViolationKind::PotentialNullArrayAccess => {
                format!("Check array is not null before accessing: if (array != null) array[index]")
            }
            NullViolationKind::NullArgumentToNonNullable => {
                format!("Ensure argument is not null or change parameter type to optional")
            }
            NullViolationKind::NullReturnFromNonNullable => {
                format!("Return non-null value or change return type to optional")
            }
            NullViolationKind::NullAssignedToNotNull => {
                format!("Cannot assign null to @:notNull variable; use a non-null value")
            }
            NullViolationKind::NullableAssignedToNotNull => {
                format!("Add a null check before assigning to @:notNull variable")
            }
        };

        suggestions.push(suggestion);
    }

    suggestions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // fn test_null_state_merge() {
    //     let analyzer = NullSafetyAnalyzer::new(
    //         &RefCell::new(TypeTable::new()),
    //         &SymbolTable::new(),
    //         &ControlFlowGraph::new(),
    //     );

    //     assert_eq!(
    //         analyzer.merge_null_states(NullState::Null, NullState::Null),
    //         NullState::Null
    //     );

    //     assert_eq!(
    //         analyzer.merge_null_states(NullState::NotNull, NullState::Null),
    //         NullState::MaybeNull
    //     );
    // }
    #[test]
    fn test_null_violation_suggestion() {
        let violation = NullSafetyViolation {
            variable: SymbolId::from_raw(1),
            violation_kind: NullViolationKind::PotentialNullMethodCall,
            location: SourceLocation::unknown(),
            suggestion: None,
        };

        let suggestions = suggest_null_safety_fixes(&[violation]);
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("safe navigation"));
    }
}
