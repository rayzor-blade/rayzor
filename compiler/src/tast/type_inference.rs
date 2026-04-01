//! Type Inference System
//!
//! This module implements type inference to ensure that all expressions have concrete types
//! instead of defaulting to Dynamic. This is critical for Rust-style memory management
//! as we need to know the exact size and layout of all types at compile time.

use crate::tast::{
    core::*,
    node::*,
    TypeTable,
    TypeId,
    SymbolId,
    InternedString,
    SourceLocation,
};
use std::collections::BTreeMap;
use std::cell::RefCell;
use std::rc::Rc;

/// Type inference context
pub struct TypeInferenceContext<'a> {
    type_table: &'a Rc<RefCell<TypeTable>>,
    inference_variables: BTreeMap<InferenceVariableId, TypeId>,
    next_inference_var: u32,
    constraints: Vec<TypeConstraint>,
}

/// Unique identifier for type inference variables
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct InferenceVariableId(u32);

/// Type constraints generated during inference
#[derive(Debug, Clone)]
pub enum TypeConstraint {
    /// Two types must be equal
    Equality {
        left: TypeId,
        right: TypeId,
        reason: String,
    },
    /// A type must be assignable to another
    Assignable {
        from: TypeId,
        to: TypeId,
        reason: String,
    },
    /// A type must implement a specific trait/interface
    Implements {
        type_id: TypeId,
        interface_id: SymbolId,
        reason: String,
    },
    /// A type must be numeric (Int or Float)
    Numeric {
        type_id: TypeId,
        reason: String,
    },
    /// A type must be boolean
    Boolean {
        type_id: TypeId,
        reason: String,
    },
}

/// Type inference error
#[derive(Debug, Clone)]
pub enum InferenceError {
    CannotInfer {
        expression: String,
        reason: String,
    },
    TypeMismatch {
        expected: String,
        found: String,
        location: SourceLocation,
    },
    ConstraintUnsatisfiable {
        constraint: String,
        reason: String,
    },
}

impl<'a> TypeInferenceContext<'a> {
    pub fn new(type_table: &'a Rc<RefCell<TypeTable>>) -> Self {
        Self {
            type_table,
            inference_variables: BTreeMap::new(),
            next_inference_var: 0,
            constraints: Vec::new(),
        }
    }

    /// Create a new inference variable
    pub fn new_inference_variable(&mut self) -> InferenceVariableId {
        let id = InferenceVariableId(self.next_inference_var);
        self.next_inference_var += 1;
        id
    }

    /// Infer concrete types for a function
    pub fn infer_function_types(&mut self, function: &mut TypedFunction) -> Result<(), Vec<InferenceError>> {
        // Infer parameter types
        for parameter in &mut function.parameters {
            if self.is_dynamic_type(parameter.param_type) {
                parameter.param_type = self.infer_parameter_type(parameter);
            }
        }

        // Infer return type from function body
        if self.is_dynamic_type(function.return_type) {
            match self.infer_return_type(&function.body) {
                Ok(return_type) => function.return_type = return_type,
                Err(e) => return Err(vec![e]),
            }
        }

        // Infer types for all statements in the body
        for statement in &mut function.body {
            if let Err(e) = self.infer_statement_types(statement) {
                return Err(vec![e]);
            }
        }

        // Solve constraints
        self.solve_constraints()?;

        Ok(())
    }

    /// Infer concrete types for a class
    pub fn infer_class_types(&mut self, class: &mut TypedClass) -> Result<(), Vec<InferenceError>> {
        // Infer field types
        for field in &mut class.fields {
            if self.is_dynamic_type(field.field_type) {
                field.field_type = self.infer_field_type(field);
            }
        }

        // Infer method types
        for method in &mut class.methods {
            self.infer_function_types(method)?;
        }

        Ok(())
    }

    /// Check if a type is Dynamic (needs inference)
    fn is_dynamic_type(&self, type_id: TypeId) -> bool {
        let type_table = self.type_table.borrow();
        if let Some(type_info) = type_table.get_type_info(type_id) {
            matches!(type_info.kind, TypeKind::Dynamic)
        } else {
            // If we can't find the type, assume it needs inference
            true
        }
    }

    /// Infer parameter type from usage context
    fn infer_parameter_type(&mut self, parameter: &TypedParameter) -> TypeId {
        // For now, parameters without explicit types default to Dynamic
        // In a full implementation, we would analyze usage patterns
        self.type_table.borrow().dynamic_type()
    }

    /// Infer return type from function body
    fn infer_return_type(&mut self, body: &[TypedStatement]) -> Result<TypeId, InferenceError> {
        let mut return_types = Vec::new();

        for statement in body {
            self.collect_return_types(statement, &mut return_types);
        }

        if return_types.is_empty() {
            // No return statements, function returns void
            Ok(self.type_table.borrow().void_type())
        } else if return_types.len() == 1 {
            // Single return type
            Ok(return_types[0])
        } else {
            // Multiple return types - find common supertype
            self.find_common_supertype(&return_types)
        }
    }

    /// Collect all return types from a statement
    fn collect_return_types(&self, statement: &TypedStatement, return_types: &mut Vec<TypeId>) {
        match statement {
            TypedStatement::Return { value, .. } => {
                if let Some(expr) = value {
                    return_types.push(expr.expr_type);
                } else {
                    return_types.push(self.type_table.borrow().void_type());
                }
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_return_types(stmt, return_types);
                }
            }
            TypedStatement::If { then_branch, else_branch, .. } => {
                self.collect_return_types(then_branch, return_types);
                if let Some(else_stmt) = else_branch {
                    self.collect_return_types(else_stmt, return_types);
                }
            }
            TypedStatement::While { body, .. } => {
                self.collect_return_types(body, return_types);
            }
            TypedStatement::For { body, .. } => {
                self.collect_return_types(body, return_types);
            }
            // Expression statements don't contain returns
            _ => {}
        }
    }

    /// Find common supertype for multiple types
    fn find_common_supertype(&self, types: &[TypeId]) -> Result<TypeId, InferenceError> {
        if types.is_empty() {
            return Ok(self.type_table.borrow().void_type());
        }

        let first_type = types[0];

        // For now, if all types are the same, return that type
        // Otherwise, return the first type as a fallback
        for &type_id in types.iter().skip(1) {
            if type_id != first_type {
                // In a full implementation, we would compute the actual common supertype
                // For now, we'll just ensure it's a concrete type
                if self.is_dynamic_type(first_type) {
                    return Ok(self.type_table.borrow().string_type()); // Default fallback
                }
                return Ok(first_type);
            }
        }

        Ok(first_type)
    }

    /// Infer field type from initializer or usage
    fn infer_field_type(&mut self, field: &TypedField) -> TypeId {
        // If field has an initializer, use its type
        if let Some(init_expr) = &field.initializer {
            if !self.is_dynamic_type(init_expr.expr_type) {
                return init_expr.expr_type;
            }
        }

        // Default to String for now - in practice, we'd analyze usage patterns
        self.type_table.borrow().string_type()
    }

    /// Infer types for a statement
    fn infer_statement_types(&mut self, statement: &mut TypedStatement) -> Result<(), InferenceError> {
        match statement {
            TypedStatement::Expression { expression, .. } => {
                self.infer_expression_types(expression)?;
            }
            TypedStatement::VarDecl { var_type, initializer, .. } => {
                if let Some(init_expr) = initializer {
                    self.infer_expression_types(init_expr)?;

                    // If variable type is dynamic, infer from initializer
                    if self.is_dynamic_type(*var_type) && !self.is_dynamic_type(init_expr.expr_type) {
                        *var_type = init_expr.expr_type;
                    }
                }
            }
            TypedStatement::Assignment { target, value, .. } => {
                self.infer_expression_types(target)?;
                self.infer_expression_types(value)?;

                // Add constraint that value type must be assignable to target type
                self.constraints.push(TypeConstraint::Assignable {
                    from: value.expr_type,
                    to: target.expr_type,
                    reason: "assignment".to_string(),
                });
            }
            TypedStatement::Return { value, .. } => {
                if let Some(expr) = value {
                    self.infer_expression_types(expr)?;
                }
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.infer_statement_types(stmt)?;
                }
            }
            TypedStatement::If { condition, then_branch, else_branch, .. } => {
                self.infer_expression_types(condition)?;

                // Condition must be boolean
                self.constraints.push(TypeConstraint::Boolean {
                    type_id: condition.expr_type,
                    reason: "if condition".to_string(),
                });

                self.infer_statement_types(then_branch)?;
                if let Some(else_stmt) = else_branch {
                    self.infer_statement_types(else_stmt)?;
                }
            }
            TypedStatement::While { condition, body, .. } => {
                self.infer_expression_types(condition)?;

                // Condition must be boolean
                self.constraints.push(TypeConstraint::Boolean {
                    type_id: condition.expr_type,
                    reason: "while condition".to_string(),
                });

                self.infer_statement_types(body)?;
            }
            TypedStatement::For { iterable, body, .. } => {
                self.infer_expression_types(iterable)?;
                self.infer_statement_types(body)?;
            }
            TypedStatement::Break { .. } | TypedStatement::Continue { .. } => {
                // No type inference needed
            }
        }

        Ok(())
    }

    /// Infer types for an expression
    fn infer_expression_types(&mut self, expression: &mut TypedExpression) -> Result<(), InferenceError> {
        match &mut expression.kind {
            TypedExpressionKind::Literal { value } => {
                // Literals have known types
                expression.expr_type = match value {
                    LiteralValue::Bool(_) => self.type_table.borrow().bool_type(),
                    LiteralValue::Int(_) => self.type_table.borrow().int_type(),
                    LiteralValue::Float(_) => self.type_table.borrow().float_type(),
                    LiteralValue::String(_) => self.type_table.borrow().string_type(),
                    LiteralValue::Char(_) => self.type_table.borrow().char_type(),
                    LiteralValue::Null => self.type_table.borrow().dynamic_type(), // Null can be any reference type
                };
            }
            TypedExpressionKind::Identifier { symbol_id, .. } => {
                // Look up the symbol's type
                // For now, we'll keep the current type if it's not dynamic
                if self.is_dynamic_type(expression.expr_type) {
                    // Default to String for identifiers - would need proper symbol table lookup
                    expression.expr_type = self.type_table.borrow().string_type();
                }
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.infer_expression_types(left)?;
                self.infer_expression_types(right)?;

                // Infer result type based on operation
                expression.expr_type = self.infer_binary_op_result_type(left.expr_type, right.expr_type)?;
            }
            TypedExpressionKind::Call { function, args, .. } => {
                self.infer_expression_types(function)?;
                for arg in args {
                    self.infer_expression_types(arg)?;
                }

                // For now, assume String result - would need function signature lookup
                if self.is_dynamic_type(expression.expr_type) {
                    expression.expr_type = self.type_table.borrow().string_type();
                }
            }
            TypedExpressionKind::FieldAccess { object, .. } => {
                self.infer_expression_types(object)?;

                // Field access result depends on field type - default to String
                if self.is_dynamic_type(expression.expr_type) {
                    expression.expr_type = self.type_table.borrow().string_type();
                }
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                self.infer_expression_types(array)?;
                self.infer_expression_types(index)?;

                // Index must be Int
                self.constraints.push(TypeConstraint::Equality {
                    left: index.expr_type,
                    right: self.type_table.borrow().int_type(),
                    reason: "array index".to_string(),
                });

                // Result type is array element type - default to String
                if self.is_dynamic_type(expression.expr_type) {
                    expression.expr_type = self.type_table.borrow().string_type();
                }
            }
            TypedExpressionKind::Conditional { condition, then_expr, else_expr } => {
                self.infer_expression_types(condition)?;
                self.infer_expression_types(then_expr)?;

                // Condition must be boolean
                self.constraints.push(TypeConstraint::Boolean {
                    type_id: condition.expr_type,
                    reason: "conditional expression condition".to_string(),
                });

                if let Some(else_expr) = else_expr {
                    self.infer_expression_types(else_expr)?;

                    // Result type is common supertype of then and else branches
                    expression.expr_type = self.find_common_supertype(&[then_expr.expr_type, else_expr.expr_type])?;
                } else {
                    // No else branch, result is void
                    expression.expr_type = self.type_table.borrow().void_type();
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.infer_statement_types(stmt)?;
                }

                // Block result is void unless it ends with an expression
                if self.is_dynamic_type(expression.expr_type) {
                    expression.expr_type = self.type_table.borrow().void_type();
                }
            }
            // Handle other expression kinds...
            _ => {
                // Default fallback for unhandled expression kinds
                if self.is_dynamic_type(expression.expr_type) {
                    expression.expr_type = self.type_table.borrow().string_type();
                }
            }
        }

        Ok(())
    }

    /// Infer the result type of a binary operation
    fn infer_binary_op_result_type(&self, left_type: TypeId, right_type: TypeId) -> Result<TypeId, InferenceError> {
        let type_table = self.type_table.borrow();

        let int_type = type_table.int_type();
        let float_type = type_table.float_type();
        let bool_type = type_table.bool_type();
        let string_type = type_table.string_type();

        // Numeric operations
        if left_type == int_type && right_type == int_type {
            Ok(int_type)
        } else if (left_type == int_type && right_type == float_type) ||
                  (left_type == float_type && right_type == int_type) ||
                  (left_type == float_type && right_type == float_type) {
            Ok(float_type)
        } else if left_type == string_type || right_type == string_type {
            // String concatenation
            Ok(string_type)
        } else {
            // Comparison operations typically return bool
            Ok(bool_type)
        }
    }

    /// Solve all accumulated constraints
    fn solve_constraints(&mut self) -> Result<(), Vec<InferenceError>> {
        let mut errors = Vec::new();

        for constraint in &self.constraints {
            if let Err(error) = self.solve_constraint(constraint) {
                errors.push(error);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Solve a single constraint
    fn solve_constraint(&self, constraint: &TypeConstraint) -> Result<(), InferenceError> {
        match constraint {
            TypeConstraint::Equality { left, right, reason } => {
                if left != right {
                    return Err(InferenceError::TypeMismatch {
                        expected: format!("TypeId({:?})", right),
                        found: format!("TypeId({:?})", left),
                        location: SourceLocation::unknown(),
                    });
                }
            }
            TypeConstraint::Boolean { type_id, reason } => {
                let bool_type = self.type_table.borrow().bool_type();
                if *type_id != bool_type {
                    return Err(InferenceError::TypeMismatch {
                        expected: "Bool".to_string(),
                        found: format!("TypeId({:?})", type_id),
                        location: SourceLocation::unknown(),
                    });
                }
            }
            TypeConstraint::Numeric { type_id, reason } => {
                let type_table = self.type_table.borrow();
                let int_type = type_table.int_type();
                let float_type = type_table.float_type();

                if *type_id != int_type && *type_id != float_type {
                    return Err(InferenceError::TypeMismatch {
                        expected: "Int or Float".to_string(),
                        found: format!("TypeId({:?})", type_id),
                        location: SourceLocation::unknown(),
                    });
                }
            }
            // Handle other constraint types...
            _ => {
                // For now, assume other constraints are satisfied
            }
        }

        Ok(())
    }
}

/// Replace Dynamic types with concrete types throughout a TAST file
pub fn infer_concrete_types(tast_file: &mut TypedFile, type_table: &Rc<RefCell<TypeTable>>) -> Result<(), Vec<InferenceError>> {
    let mut inference_context = TypeInferenceContext::new(type_table);
    let mut all_errors = Vec::new();

    // Infer types for all functions
    for function in &mut tast_file.functions {
        if let Err(mut errors) = inference_context.infer_function_types(function) {
            all_errors.append(&mut errors);
        }
    }

    // Infer types for all classes
    for class in &mut tast_file.classes {
        if let Err(mut errors) = inference_context.infer_class_types(class) {
            all_errors.append(&mut errors);
        }
    }

    // TODO: Handle interfaces, enums, abstracts

    if all_errors.is_empty() {
        Ok(())
    } else {
        Err(all_errors)
    }
}