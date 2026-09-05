//! Emitting a type error in the shapes the reporter expects.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    pub fn emit_error(&mut self, error: TypeCheckError) {
        let diagnostic = self.diagnostic_emitter.emit_diagnostic(error);
        self.diagnostics.push(diagnostic);
    }


    /// Emit an enhanced type error with context-aware suggestions
    pub fn emit_enhanced_type_error(
        &mut self,
        actual_type: TypeId,
        expected_type: TypeId,
        location: SourceLocation,
        context: &str,
        error_context: &TypeErrorContext,
    ) {
        // Get suggestions from the diagnostic emitter
        let suggestions =
            self.diagnostic_emitter
                .get_suggestions(actual_type, expected_type, error_context);

        let suggestion = if !suggestions.is_empty() {
            Some(suggestions.join(". "))
        } else {
            None
        };

        let error = TypeCheckError {
            kind: TypeErrorKind::TypeMismatch {
                expected: expected_type,
                actual: actual_type,
            },
            location,
            context: context.to_string(),
            suggestion,
        };

        self.emit_error(error);
    }


    /// Emit a constraint violation error
    pub fn emit_constraint_violation(
        &mut self,
        violating_type: TypeId,
        constraint_type: TypeId,
        location: SourceLocation,
    ) {
        // Find the type parameter that has this constraint
        let type_param = constraint_type; // For now, use constraint as type param

        let error = TypeCheckError {
            kind: TypeErrorKind::ConstraintViolation {
                type_param,
                constraint: constraint_type,
                violating_type,
            },
            location,
            context: "Generic constraint validation".to_string(),
            suggestion: Some(
                "Ensure the type argument implements the required constraint".to_string(),
            ),
        };

        self.emit_error(error);
    }
}
