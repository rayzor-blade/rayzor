//! Send/Sync validation for thread safety
//!
//! This module integrates TraitChecker, CaptureAnalyzer, and CoreTypeChecker
//! to enforce Send/Sync constraints at compile time. This provides Rust-like
//! thread safety guarantees for Rayzor code.
//!
//! ## Validation Rules
//!
//! 1. **Thread::spawn(closure)** - All captured variables must be Send
//! 2. **Channel<T>** - T must be Send
//! 3. **Arc<T>** - T must be Send + Sync
//! 4. **Mutex<T>** - T can be any type (Mutex provides interior mutability)
//!
//! ## Example
//!
//! ```haxe
//! @:derive([Send])
//! class Data {
//!     var value: Int;
//! }
//!
//! var data = new Data();
//! var handle = Thread.spawn(() -> {
//!     // OK: Data is Send
//!     trace(data.value);
//! });
//! ```

use crate::tast::{
    capture_analyzer::{CaptureAnalyzer, CapturedVariable},
    core_types::CoreTypeChecker,
    node::{TypedClass, TypedExpression, TypedFunction, TypedStatement},
    trait_checker::TraitChecker,
    ScopeId, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeTable,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Validation error for Send/Sync constraints
#[derive(Debug, Clone)]
pub struct SendSyncError {
    /// Full error message for display
    pub message: String,
    /// Name of the type that failed validation (e.g., "Function", "MyClass")
    pub type_name: String,
    /// Reason the type is not Send/Sync (e.g., "type does not derive Send")
    pub reason: String,
    /// Type that failed validation
    pub type_id: TypeId,
    /// Symbol that failed validation (if applicable)
    pub symbol_id: Option<SymbolId>,
    /// Source location of the offending expression
    pub source_location: SourceLocation,
}

impl SendSyncError {
    pub fn new(type_name: String, reason: String, type_id: TypeId) -> Self {
        let message = format!(
            "Cannot send value of type `{}` across threads: {}",
            type_name, reason
        );
        Self {
            message,
            type_name,
            reason,
            type_id,
            symbol_id: None,
            source_location: SourceLocation::unknown(),
        }
    }

    pub fn with_symbol(mut self, symbol_id: SymbolId) -> Self {
        self.symbol_id = Some(symbol_id);
        self
    }

    pub fn with_location(mut self, loc: SourceLocation) -> Self {
        self.source_location = loc;
        self
    }
}

/// Result type for validation
pub type ValidationResult<T> = Result<T, SendSyncError>;

/// Unified validator for Send/Sync constraints
pub struct SendSyncValidator<'a> {
    trait_checker: TraitChecker<'a>,
    core_checker: CoreTypeChecker<'a>,
    type_table: &'a Rc<RefCell<TypeTable>>,
    symbol_table: &'a SymbolTable,
    string_interner: &'a StringInterner,
}

impl<'a> SendSyncValidator<'a> {
    /// Create a new Send/Sync validator
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        symbol_table: &'a SymbolTable,
        string_interner: &'a StringInterner,
        classes: &'a [TypedClass],
    ) -> Self {
        Self {
            trait_checker: TraitChecker::new(type_table, symbol_table, string_interner, classes),
            core_checker: CoreTypeChecker::new(type_table, symbol_table, string_interner),
            type_table,
            symbol_table,
            string_interner,
        }
    }

    /// Validate a function call expression
    ///
    /// Checks for:
    /// - Thread::spawn(closure) - validates closure captures are Send
    /// - Other thread-safety sensitive calls
    pub fn validate_call(&self, call_expr: &TypedExpression) -> ValidationResult<()> {
        // Check if this is Thread::spawn
        if let Some(closure_type) = self.core_checker.get_thread_spawn_closure(call_expr) {
            return self.validate_thread_spawn(call_expr, closure_type);
        }

        // Check if this is Future::create (same Send constraint as Thread::spawn)
        if let Some(closure_type) = self.core_checker.get_future_create_closure(call_expr) {
            return self.validate_thread_spawn(call_expr, closure_type);
        }

        Ok(())
    }

    /// Validate Thread::spawn call
    ///
    /// Ensures all captured variables are Send
    fn validate_thread_spawn(
        &self,
        call_expr: &TypedExpression,
        closure_type: TypeId,
    ) -> ValidationResult<()> {
        use crate::tast::node::TypedExpressionKind;

        // Extract the closure expression
        let closure_expr = match &call_expr.kind {
            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                arguments.first().ok_or_else(|| {
                    SendSyncError::new(
                        "closure".to_string(),
                        "Thread::spawn requires a closure argument".to_string(),
                        closure_type,
                    )
                })?
            }
            _ => return Ok(()),
        };

        // Check if it's a function literal
        if let TypedExpressionKind::FunctionLiteral {
            parameters, body, ..
        } = &closure_expr.kind
        {
            // Analyze captures — types resolved from expression's expr_type
            let analyzer = CaptureAnalyzer::new(ScopeId::invalid());
            let analysis = analyzer.analyze_function_literal(parameters, body);

            // Validate all captures are Send
            let call_loc = call_expr.source_location;
            for capture in &analysis.captures {
                self.validate_capture_is_send(capture, call_loc)?;
            }
        }

        Ok(())
    }

    /// Validate that a captured variable is Send
    fn validate_capture_is_send(
        &self,
        capture: &CapturedVariable,
        call_location: SourceLocation,
    ) -> ValidationResult<()> {
        let type_id = capture.type_id;

        if !type_id.is_valid() {
            // Type couldn't be resolved — skip (shouldn't happen with expr_type resolution)
            return Ok(());
        }

        if !self.trait_checker.is_send(type_id) {
            let type_name = self.get_type_name(type_id);
            return Err(SendSyncError::new(
                type_name,
                "type does not derive Send".to_string(),
                type_id,
            )
            .with_symbol(capture.symbol_id)
            .with_location(call_location));
        }

        Ok(())
    }

    /// Get a human-readable name for a type
    fn get_type_name(&self, type_id: TypeId) -> String {
        let type_table = self.type_table.borrow();
        match type_table.get(type_id) {
            Some(info) => {
                use crate::tast::core::TypeKind;
                match &info.kind {
                    TypeKind::Int => "Int".to_string(),
                    TypeKind::Float => "Float".to_string(),
                    TypeKind::Bool => "Bool".to_string(),
                    TypeKind::String => "String".to_string(),
                    TypeKind::Void => "Void".to_string(),
                    TypeKind::Dynamic => "Dynamic".to_string(),
                    TypeKind::Class { symbol_id, .. } => {
                        if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                            if let Some(qn) = sym.qualified_name {
                                return self
                                    .string_interner
                                    .get(qn)
                                    .unwrap_or("unknown")
                                    .to_string();
                            }
                            return self
                                .string_interner
                                .get(sym.name)
                                .unwrap_or("unknown")
                                .to_string();
                        }
                        "unknown class".to_string()
                    }
                    TypeKind::Function { .. } => "Function".to_string(),
                    TypeKind::Array { .. } => "Array".to_string(),
                    _ => format!("{:?}", type_id),
                }
            }
            None => format!("{:?}", type_id),
        }
    }

    /// Validate a type used in a Channel<T>
    ///
    /// Ensures T is Send
    pub fn validate_channel_type(
        &self,
        channel_type_id: TypeId,
        loc: SourceLocation,
    ) -> ValidationResult<()> {
        if let Some(element_type) = self.core_checker.get_channel_element_type(channel_type_id) {
            if !self.trait_checker.is_send(element_type) {
                let type_name = self.get_type_name(element_type);
                return Err(SendSyncError::new(
                    format!("Channel<{}>", type_name),
                    format!("element type `{}` does not derive Send", type_name),
                    element_type,
                )
                .with_location(loc));
            }
        }
        Ok(())
    }

    /// Validate a type used in Arc<T>
    ///
    /// Ensures T is Send + Sync
    pub fn validate_arc_type(
        &self,
        arc_type_id: TypeId,
        loc: SourceLocation,
    ) -> ValidationResult<()> {
        if let Some(element_type) = self.core_checker.get_arc_element_type(arc_type_id) {
            let type_name = self.get_type_name(element_type);

            if !self.trait_checker.is_send(element_type) {
                return Err(SendSyncError::new(
                    format!("Arc<{}>", type_name),
                    format!("inner type `{}` does not derive Send", type_name),
                    element_type,
                )
                .with_location(loc));
            }

            if !self.trait_checker.is_sync(element_type) {
                return Err(SendSyncError::new(
                    format!("Arc<{}>", type_name),
                    format!("inner type `{}` does not derive Sync", type_name),
                    element_type,
                )
                .with_location(loc));
            }
        }
        Ok(())
    }

    /// Validate all expressions in a statement
    ///
    /// This walks the AST and validates all thread-safety constraints
    pub fn validate_statement(&self, stmt: &TypedStatement) -> ValidationResult<()> {
        match stmt {
            TypedStatement::Expression { expression, .. } => {
                self.validate_expression(expression)?;
            }

            TypedStatement::VarDeclaration { initializer, .. } => {
                if let Some(init) = initializer {
                    self.validate_expression(init)?;
                }
            }

            TypedStatement::Assignment { target, value, .. } => {
                self.validate_expression(target)?;
                self.validate_expression(value)?;
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.validate_expression(condition)?;
                self.validate_statement(then_branch)?;
                if let Some(else_stmt) = else_branch {
                    self.validate_statement(else_stmt)?;
                }
            }

            TypedStatement::While {
                condition, body, ..
            } => {
                self.validate_expression(condition)?;
                self.validate_statement(body)?;
            }

            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                if let Some(init_stmt) = init {
                    self.validate_statement(init_stmt)?;
                }
                if let Some(cond) = condition {
                    self.validate_expression(cond)?;
                }
                if let Some(upd) = update {
                    self.validate_expression(upd)?;
                }
                self.validate_statement(body)?;
            }

            TypedStatement::ForIn { iterable, body, .. } => {
                self.validate_expression(iterable)?;
                self.validate_statement(body)?;
            }

            TypedStatement::Return { value, .. } => {
                if let Some(expr) = value {
                    self.validate_expression(expr)?;
                }
            }

            TypedStatement::Throw { exception, .. } => {
                self.validate_expression(exception)?;
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                self.validate_statement(body)?;
                for catch in catch_clauses {
                    self.validate_statement(&catch.body)?;
                }
                if let Some(finally) = finally_block {
                    self.validate_statement(finally)?;
                }
            }

            TypedStatement::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => {
                self.validate_expression(discriminant)?;
                for case in cases {
                    self.validate_expression(&case.case_value)?;
                    self.validate_statement(&case.body)?;
                }
                if let Some(default) = default_case {
                    self.validate_statement(default)?;
                }
            }

            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.validate_statement(stmt)?;
                }
            }

            TypedStatement::PatternMatch {
                value, patterns, ..
            } => {
                self.validate_expression(value)?;
                for pattern in patterns {
                    self.validate_statement(&pattern.body)?;
                }
            }

            TypedStatement::Break { .. }
            | TypedStatement::Continue { .. }
            | TypedStatement::MacroExpansion { .. } => {
                // No validation needed
            }
        }

        Ok(())
    }

    /// Validate an expression
    fn validate_expression(&self, expr: &TypedExpression) -> ValidationResult<()> {
        use crate::tast::node::TypedExpressionKind;

        match &expr.kind {
            // Function calls - check for Thread::spawn
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.validate_expression(function)?;
                for arg in arguments {
                    self.validate_expression(arg)?;
                }
                self.validate_call(expr)?;
            }

            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.validate_expression(receiver)?;
                for arg in arguments {
                    self.validate_expression(arg)?;
                }
            }

            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                for arg in arguments {
                    self.validate_expression(arg)?;
                }
                // Check if this is Thread::spawn or other core types
                self.validate_call(expr)?;
            }

            // New expressions - check for Channel<T>, Arc<T>
            TypedExpressionKind::New { arguments, .. } => {
                for arg in arguments {
                    self.validate_expression(arg)?;
                }

                // Validate Channel<T> and Arc<T> type constraints
                let loc = expr.source_location;
                if self.core_checker.is_channel(expr.expr_type) {
                    self.validate_channel_type(expr.expr_type, loc)?;
                }
                if self.core_checker.is_arc(expr.expr_type) {
                    self.validate_arc_type(expr.expr_type, loc)?;
                }
            }

            // Recurse into other expression types
            TypedExpressionKind::FieldAccess { object, .. } => {
                self.validate_expression(object)?;
            }

            TypedExpressionKind::ArrayAccess { array, index } => {
                self.validate_expression(array)?;
                self.validate_expression(index)?;
            }

            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.validate_expression(left)?;
                self.validate_expression(right)?;
            }

            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.validate_expression(operand)?;
            }

            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.validate_expression(condition)?;
                self.validate_expression(then_expr)?;
                if let Some(else_e) = else_expr {
                    self.validate_expression(else_e)?;
                }
            }

            TypedExpressionKind::ArrayLiteral { elements } => {
                for elem in elements {
                    self.validate_expression(elem)?;
                }
            }

            TypedExpressionKind::ObjectLiteral { fields } => {
                for field in fields {
                    self.validate_expression(&field.value)?;
                }
            }

            TypedExpressionKind::FunctionLiteral { body, .. } => {
                for stmt in body {
                    self.validate_statement(stmt)?;
                }
            }

            TypedExpressionKind::Block { statements, .. } => {
                for stmt in statements {
                    self.validate_statement(stmt)?;
                }
            }

            TypedExpressionKind::Switch {
                discriminant,
                cases,
                default_case,
            } => {
                self.validate_expression(discriminant)?;
                for case in cases {
                    self.validate_expression(&case.case_value)?;
                    self.validate_statement(&case.body)?;
                }
                if let Some(default) = default_case {
                    self.validate_expression(default)?;
                }
            }

            TypedExpressionKind::Try {
                try_expr,
                catch_clauses,
                finally_block,
            } => {
                self.validate_expression(try_expr)?;
                for catch in catch_clauses {
                    self.validate_statement(&catch.body)?;
                }
                if let Some(finally) = finally_block {
                    self.validate_expression(finally)?;
                }
            }

            // Expressions that don't need validation
            TypedExpressionKind::Literal { .. }
            | TypedExpressionKind::Variable { .. }
            | TypedExpressionKind::StaticFieldAccess { .. }
            | TypedExpressionKind::This { .. }
            | TypedExpressionKind::Super { .. }
            | TypedExpressionKind::Null
            | TypedExpressionKind::Break
            | TypedExpressionKind::Continue => {}

            _ => {
                // TODO: Handle remaining expression kinds
            }
        }

        Ok(())
    }

    /// Validate an entire function
    pub fn validate_function(&self, func: &TypedFunction) -> ValidationResult<()> {
        for stmt in &func.body {
            self.validate_statement(stmt)?;
        }
        Ok(())
    }

    /// Validate all functions in a class
    pub fn validate_class(&self, class: &TypedClass) -> ValidationResult<()> {
        for method in &class.methods {
            self.validate_function(method)?;
        }
        for constructor in &class.constructors {
            self.validate_function(constructor)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add comprehensive tests with actual TAST structures
    // These would test:
    // 1. Thread::spawn with Send captures (should pass)
    // 2. Thread::spawn with non-Send captures (should fail)
    // 3. Channel<T> where T is Send (should pass)
    // 4. Channel<T> where T is not Send (should fail)
    // 5. Arc<T> where T is Send+Sync (should pass)
    // 6. Arc<T> where T is not Send or not Sync (should fail)
}
