//! Flow analysis and the Send/Sync validation that rides with it, plus the
//! diagnostics both produce.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Run flow-sensitive safety analysis on the typed file
    pub(crate) fn run_flow_analysis(&mut self, typed_file: &TypedFile) -> Result<(), String> {
        // Initialize flow guard if not already done
        self.initialize_flow_guard();

        if let Some(ref mut flow_guard) = self.type_flow_guard {
            // Run flow analysis
            let results = flow_guard.analyze_file(typed_file);

            // Convert flow safety errors to diagnostics
            self.emit_flow_safety_diagnostics(&results);

            // Log performance metrics if in debug mode
            #[cfg(debug_assertions)]
            {
                eprintln!("Flow analysis metrics:");
                eprintln!(
                    "  Functions analyzed: {}",
                    results.metrics.functions_analyzed
                );
                eprintln!("  Blocks processed: {}", results.metrics.blocks_processed);
                eprintln!(
                    "  CFG construction time: {} μs",
                    results.metrics.cfg_construction_time_us
                );
                eprintln!(
                    "  Variable analysis time: {} μs",
                    results.metrics.variable_analysis_time_us
                );
                eprintln!(
                    "  Null safety time: {} μs",
                    results.metrics.null_safety_time_us
                );
                eprintln!("  Dead code time: {} μs", results.metrics.dead_code_time_us);
            }
        }

        Ok(())
    }


    /// Run Send/Sync validation for thread safety
    ///
    /// Validates that:
    /// - Thread::spawn captures only Send types
    /// - Channel<T> has T: Send
    /// - Arc<T> has T: Send + Sync
    pub(crate) fn run_send_sync_validation(&mut self, typed_file: &TypedFile) -> Result<(), String> {
        // Create the validator
        let validator = SendSyncValidator::new(
            self.type_table,
            self.symbol_table,
            self.string_interner,
            &typed_file.classes,
        );

        // Validate all classes
        for class in &typed_file.classes {
            if let Err(error) = validator.validate_class(class) {
                self.emit_send_sync_error(error);
            }
            // Soundness: a class explicitly deriving Send/Sync must have fields
            // that fulfill the trait (extern types skipped — opaque promise).
            for error in validator.validate_derive_soundness(class) {
                self.emit_send_sync_error(error);
            }
        }

        // Validate all module-level functions
        for function in &typed_file.functions {
            if let Err(error) = validator.validate_function(function) {
                self.emit_send_sync_error(error);
            }
        }

        Ok(())
    }


    /// Emit a Send/Sync validation error as a diagnostic
    pub(crate) fn emit_send_sync_error(&mut self, error: SendSyncError) {
        self.emit_error(TypeCheckError {
            kind: TypeErrorKind::SendSyncViolation {
                type_name: error.type_name.clone(),
                reason: error.reason.clone(),
            },
            location: error.source_location,
            context: error.message.clone(),
            suggestion: Some(
                "Add @:derive([Send]) or @:derive([Send, Sync]) to the type".to_string(),
            ),
        });
    }


    /// Convert flow safety results to diagnostics
    pub(crate) fn emit_flow_safety_diagnostics(&mut self, results: &FlowSafetyResults) {
        // Emit errors
        for error in &results.errors {
            self.emit_flow_safety_error(error);
        }

        // Emit warnings
        for warning in &results.warnings {
            self.emit_flow_safety_warning(warning);
        }
    }


    /// Emit a flow safety error as a diagnostic
    pub(crate) fn emit_flow_safety_error(&mut self, error: &FlowSafetyError) {
        match error {
            FlowSafetyError::UninitializedVariable { variable, location } => {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType {
                        name: self
                            .string_interner
                            .intern(&format!("uninitialized_var_{}", variable.as_raw())),
                    },
                    location: *location,
                    context: format!("Variable used before initialization"),
                    suggestion: Some("Initialize the variable before using it".to_string()),
                });
            }
            FlowSafetyError::NullDereference { variable, location } => {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType {
                        name: self
                            .string_interner
                            .intern(&format!("null_deref_{}", variable.as_raw())),
                    },
                    location: *location,
                    context: format!("Potential null dereference"),
                    suggestion: Some("Check for null before dereferencing".to_string()),
                });
            }
            FlowSafetyError::ResourceLeak { resource, location } => {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType {
                        name: self
                            .string_interner
                            .intern(&format!("resource_leak_{}", resource.as_raw())),
                    },
                    location: *location,
                    context: format!("Resource leak detected"),
                    suggestion: Some("Ensure resource is properly disposed".to_string()),
                });
            }
            _ => {
                // Handle other error types as warnings for now
                self.emit_flow_safety_warning(error);
            }
        }
    }


    /// Emit a flow safety warning as a diagnostic
    pub(crate) fn emit_flow_safety_warning(&mut self, warning: &FlowSafetyError) {
        match warning {
            FlowSafetyError::DeadCode { location } => {
                // For now, we'll emit dead code as a hint in diagnostics
                let start_pos = SourcePosition::new(
                    location.line as usize,
                    location.column as usize,
                    location.byte_offset as usize,
                );
                let end_pos = SourcePosition::new(
                    location.line as usize,
                    location.column as usize + 1,
                    location.byte_offset as usize + 1,
                );
                let span = SourceSpan::new(
                    start_pos,
                    end_pos,
                    source_map::FileId::new(location.file_id as usize),
                );
                let diagnostic =
                    diagnostics::DiagnosticBuilder::hint("Dead code detected", span).build();
                self.diagnostics.push(diagnostic);
            }
            _ => {
                // Other warnings can be added here
            }
        }
    }
}
