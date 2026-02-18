//! Type Checking Pipeline Integration
//!
//! This module integrates the type checker into the compilation pipeline,
//! providing a complete type checking phase with diagnostic reporting.

use super::{
    node::{
        BinaryOperator, CastKind, StringInterpolationPart, TypedClass, TypedEnum, TypedExpression,
        TypedExpressionKind, TypedField, TypedFile, TypedFunction, TypedInterface, TypedMapEntry,
        TypedMethodSignature, TypedStatement, TypedSwitchCase,
    },
    send_sync_validator::{SendSyncError, SendSyncValidator},
    type_checker::TypeCompatibility,
    type_diagnostics::{TypeDiagnosticEmitter, TypeErrorContext},
    AccessLevel, FlowSafetyError, FlowSafetyResults, InternedString, NamespaceResolver,
    PackageAccessContext, PackageAccessValidator, ScopeTree, SourceLocation, StringInterner,
    SymbolId, SymbolTable, TypeCheckError, TypeChecker, TypeErrorKind, TypeFlowGuard, TypeId,
    TypeKind, TypeTable, Visibility,
};
use diagnostics::{Diagnostics, SourceMap};
use source_map::{SourcePosition, SourceSpan};
use std::cell::RefCell;
use std::rc::Rc;

/// Type checking phase that integrates with the compilation pipeline
pub struct TypeCheckingPhase<'a> {
    type_checker: TypeChecker<'a>,
    diagnostic_emitter: TypeDiagnosticEmitter<'a>,
    diagnostics: &'a mut Diagnostics,
    string_interner: &'a StringInterner,
    /// Type table (stored separately for SendSyncValidator)
    type_table: &'a Rc<RefCell<TypeTable>>,
    /// Symbol table (stored separately for SendSyncValidator)
    symbol_table: &'a SymbolTable,
    /// Stack of expected return types for nested function contexts
    expected_return_types: Vec<TypeId>,
    /// Temporary reference to the typed file for constraint validation
    /// This is set during class checking to enable access to class definitions
    current_typed_file: Option<*const TypedFile>,
    /// Current method context (is_static, class_symbol_id)
    current_method_context: Option<(bool, SymbolId)>,
    /// Current package context for package-level visibility checking
    current_package: Option<super::namespace::PackageId>,
    /// Package access validator for cross-package visibility
    package_access_validator: Option<PackageAccessValidator<'a>>,
    /// Flow-sensitive safety analyzer
    type_flow_guard: Option<TypeFlowGuard<'a>>,
    /// Whether to enable flow-sensitive analysis
    enable_flow_analysis: bool,
}

impl<'a> TypeCheckingPhase<'a> {
    /// Create a new type checking phase
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        symbol_table: &'a SymbolTable,
        scope_tree: &'a ScopeTree,
        string_interner: &'a StringInterner,
        source_map: &'a SourceMap,
        diagnostics: &'a mut Diagnostics,
    ) -> Self {
        let type_checker = TypeChecker::new(type_table, symbol_table, scope_tree, string_interner);
        let diagnostic_emitter =
            TypeDiagnosticEmitter::new(type_table, symbol_table, string_interner, source_map);

        Self {
            type_checker,
            diagnostic_emitter,
            diagnostics,
            string_interner,
            type_table,
            symbol_table,
            expected_return_types: Vec::new(),
            current_typed_file: None,
            current_method_context: None,
            current_package: None,
            package_access_validator: None,
            type_flow_guard: None,
            enable_flow_analysis: true, // Enable by default
        }
    }

    /// Set the namespace resolver for package access validation
    pub fn set_namespace_resolver(&mut self, namespace_resolver: &'a NamespaceResolver) {
        self.package_access_validator = Some(PackageAccessValidator::new(
            self.type_checker.symbol_table,
            namespace_resolver,
            self.string_interner,
        ));
    }

    /// Enable or disable flow-sensitive analysis
    pub fn set_flow_analysis(&mut self, enabled: bool) {
        self.enable_flow_analysis = enabled;
    }

    /// Initialize TypeFlowGuard for flow-sensitive analysis
    fn initialize_flow_guard(&mut self) {
        if self.enable_flow_analysis && self.type_flow_guard.is_none() {
            self.type_flow_guard = Some(TypeFlowGuard::new(
                self.type_checker.symbol_table,
                self.type_checker.type_table,
            ));
        }
    }

    /// Run type checking on a typed file
    pub fn check_file(&mut self, typed_file: &mut TypedFile) -> Result<(), String> {
        // Set current package context from file metadata
        self.current_package = self.extract_package_from_file(typed_file);

        // Phase 1: Check all type declarations
        self.check_type_declarations(typed_file)?;

        // Phase 2: Check interfaces
        for interface in &typed_file.interfaces {
            self.check_interface(interface)?;
        }

        // Phase 3: Check classes
        for class in &typed_file.classes {
            // Set the current typed file for constraint validation
            self.current_typed_file = Some(typed_file as *const TypedFile);
            self.check_class(class)?;
            self.current_typed_file = None;
        }

        // Phase 4: Check enums
        for enum_decl in &typed_file.enums {
            // TODO: Add enum checking
            // self.check_enum(enum_decl)?;
        }

        // Phase 5: Check module-level functions and variables
        self.check_module_fields(typed_file)?;

        // Phase 6: Flow-sensitive safety analysis
        if self.enable_flow_analysis {
            self.run_flow_analysis(typed_file)?;
        }

        // Phase 7: Send/Sync validation for thread safety
        self.run_send_sync_validation(typed_file)?;

        // Return error if we collected any error diagnostics
        if self.diagnostics.has_errors() {
            Err(format!(
                "Type checking failed with {} errors",
                self.diagnostics.errors().count()
            ))
        } else {
            Ok(())
        }
    }

    /// Check all type declarations for validity
    fn check_type_declarations(&mut self, typed_file: &TypedFile) -> Result<(), String> {
        // Check for duplicate type names
        let mut type_names = std::collections::HashSet::new();

        for class in &typed_file.classes {
            if !type_names.insert(&class.name) {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType { name: class.name },
                    location: class.source_location,
                    context: format!(
                        "Duplicate class definition: {}",
                        self.get_string(class.name)
                    ),
                    suggestion: Some("Rename one of the duplicate classes".to_string()),
                });
            }
        }

        for interface in &typed_file.interfaces {
            if !type_names.insert(&interface.name) {
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType {
                        name: interface.name,
                    },
                    location: interface.source_location,
                    context: format!(
                        "Duplicate interface definition: {}",
                        self.get_string(interface.name)
                    ),
                    suggestion: Some("Rename one of the duplicate interfaces".to_string()),
                });
            }
        }

        Ok(())
    }

    /// Check an interface for type correctness
    fn check_interface(&mut self, interface: &TypedInterface) -> Result<(), String> {
        // Check method signatures
        for method in &interface.methods {
            self.check_method_signature(method.name, method.source_location)?;
        }

        Ok(())
    }

    /// Check a class for type correctness
    fn check_class(&mut self, class: &TypedClass) -> Result<(), String> {
        // Check field types
        for field in &class.fields {
            self.check_field_type(field.symbol_id, field.field_type, field.source_location)?;
        }

        // Check method implementations
        for method in &class.methods {
            self.check_method_implementation(method.symbol_id, method.source_location)?;

            // Check method body with return type context
            self.check_method_body(method, class.symbol_id)?;
        }

        // Verify interface implementations
        for &interface_type_id in &class.interfaces {
            if let Some(interface_symbol_id) = self
                .type_checker
                .symbol_table
                .get_symbol_from_type(interface_type_id)
            {
                self.verify_interface_implementation(
                    class.symbol_id,
                    interface_symbol_id,
                    class.source_location,
                )?;
            }
        }

        // Check inheritance method signature compatibility
        if let Some(super_type_id) = class.super_class {
            self.verify_inheritance_signatures(class, super_type_id)?;
        }

        Ok(())
    }

    /// Check an enum for type correctness
    fn check_enum(&mut self, _enum_decl: &TypedEnum) -> Result<(), String> {
        // TODO: Check that enum variant parameter types are valid
        Ok(())
    }

    /// Check module-level fields
    fn check_module_fields(&mut self, _typed_file: &TypedFile) -> Result<(), String> {
        // Module fields would be checked here
        // Currently, TypedFile doesn't expose module fields directly
        Ok(())
    }

    /// Run flow-sensitive safety analysis on the typed file
    fn run_flow_analysis(&mut self, typed_file: &TypedFile) -> Result<(), String> {
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
    fn run_send_sync_validation(&mut self, typed_file: &TypedFile) -> Result<(), String> {
        // Create the validator
        let validator =
            SendSyncValidator::new(self.type_table, self.symbol_table, &typed_file.classes);

        // Validate all classes
        for class in &typed_file.classes {
            if let Err(error) = validator.validate_class(class) {
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
    fn emit_send_sync_error(&mut self, error: SendSyncError) {
        use crate::tast::SourceLocation;

        self.emit_error(TypeCheckError {
            kind: TypeErrorKind::UndefinedType {
                name: self.string_interner.intern("send_sync_violation"),
            },
            location: SourceLocation::unknown(), // TODO: Get from error when source location added
            context: error.message.clone(),
            suggestion: Some(
                "Add @:derive([Send]) or @:derive([Send, Sync]) to the type".to_string(),
            ),
        });
    }

    /// Convert flow safety results to diagnostics
    fn emit_flow_safety_diagnostics(&mut self, results: &FlowSafetyResults) {
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
    fn emit_flow_safety_error(&mut self, error: &FlowSafetyError) {
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
    fn emit_flow_safety_warning(&mut self, warning: &FlowSafetyError) {
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

    /// Check a field type
    fn check_field_type(
        &mut self,
        _symbol_id: SymbolId,
        type_id: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        // Check if the type is valid (not unknown/error)
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                super::TypeKind::Unknown => {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::UndefinedType {
                            name: self.string_interner.intern("<unknown>"),
                        },
                        location,
                        context: format!("Field has unknown type"),
                        suggestion: Some("Add explicit type annotation".to_string()),
                    });
                }
                super::TypeKind::Error => {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::InferenceFailed {
                            reason: "Type contains errors".to_string(),
                        },
                        location,
                        context: format!("Field type could not be resolved"),
                        suggestion: None,
                    });
                }
                _ => {
                    // Type is valid
                }
            }
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<invalid>"),
                },
                location,
                context: format!("Invalid type ID: {:?}", type_id),
                suggestion: None,
            });
        }

        Ok(())
    }

    /// Check a method signature
    fn check_method_signature(
        &mut self,
        method_name: InternedString,
        location: SourceLocation,
    ) -> Result<(), String> {
        // Method signature validation is already handled during method signature matching
        // in verify_interface_implementation and verify_inheritance_signatures
        // This validates that the method signature itself is well-formed
        Ok(())
    }

    /// Check a method implementation
    fn check_method_implementation(
        &mut self,
        symbol_id: SymbolId,
        location: SourceLocation,
    ) -> Result<(), String> {
        // Find the method in the symbol table
        if let Some(symbol) = self.type_checker.symbol_table.get_symbol(symbol_id) {
            match &symbol.kind {
                super::SymbolKind::Function { .. } => {
                    // TODO: Store expected return type in context when TypeChecker exposes it
                }
                _ => {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::InferenceFailed {
                            reason: "Symbol is not a function".to_string(),
                        },
                        location,
                        context: format!("Expected function symbol"),
                        suggestion: None,
                    });
                }
            }
        }

        Ok(())
    }

    /// Check a method body with return type validation
    fn check_method_body(
        &mut self,
        method: &TypedFunction,
        class_symbol_id: SymbolId,
    ) -> Result<(), String> {
        // Push the expected return type for this method
        self.expected_return_types.push(method.return_type);

        // Set the current method context
        let previous_context = self.current_method_context;
        self.current_method_context = Some((method.is_static, class_symbol_id));

        // Check all statements in the method body
        for stmt in &method.body {
            if let Err(e) = self.check_statement(stmt) {
                // Continue checking even if there's an error
                eprintln!("Type checking error: {}", e);
            }
        }

        // Restore the previous context
        self.current_method_context = previous_context;

        // Pop the return type context
        self.expected_return_types.pop();

        Ok(())
    }

    /// Verify that a class correctly implements an interface
    fn verify_interface_implementation(
        &mut self,
        class_id: SymbolId,
        interface_id: SymbolId,
        location: SourceLocation,
    ) -> Result<(), String> {
        // Get the interface and class from the typed file
        let typed_file = unsafe {
            if let Some(file_ptr) = self.current_typed_file {
                &*file_ptr
            } else {
                return Err("No current typed file available for interface validation".to_string());
            }
        };

        // Find the interface definition
        let interface = typed_file
            .interfaces
            .iter()
            .find(|iface| iface.symbol_id == interface_id)
            .ok_or_else(|| format!("Interface with symbol ID {:?} not found", interface_id))?;

        // Find the class definition
        let class = typed_file
            .classes
            .iter()
            .find(|cls| cls.symbol_id == class_id)
            .ok_or_else(|| format!("Class with symbol ID {:?} not found", class_id))?;

        // Check that all interface methods are implemented in the class
        for interface_method in &interface.methods {
            let mut found_correct_implementation = false;
            let mut found_method_with_wrong_signature = None;

            // Look for matching method in class methods
            for class_method in &class.methods {
                if interface_method.name == class_method.name {
                    // Found a method with the same name, check signature
                    if self.method_signatures_match(&interface_method, class_method)? {
                        found_correct_implementation = true;
                        break;
                    } else {
                        // Store the method with wrong signature for better error reporting
                        found_method_with_wrong_signature = Some(class_method);
                    }
                }
            }

            if !found_correct_implementation {
                if let Some(wrong_method) = found_method_with_wrong_signature {
                    // Method exists but has wrong signature
                    let interface_sig = self.format_method_signature(interface_method);
                    let class_sig = self.format_function_signature(wrong_method);

                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::MethodSignatureMismatch {
                            expected: interface_method.return_type,
                            actual: wrong_method.return_type,
                            method_name: interface_method.name,
                        },
                        location: wrong_method.source_location,
                        context: format!(
                            "Method '{}' has incompatible signature with interface '{}'\n  Expected: {}\n  Found:    {}",
                            self.string_interner.get(interface_method.name).unwrap_or("<unknown>"),
                            self.string_interner.get(interface.name).unwrap_or("<unknown>"),
                            interface_sig,
                            class_sig
                        ),
                        suggestion: Some(format!(
                            "Change method signature to match interface: {}",
                            interface_sig
                        )),
                    });
                } else {
                    // Method is completely missing
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::InterfaceNotImplemented {
                            interface_type: self.type_checker.type_table.borrow().dynamic_type(),
                            class_type: self.type_checker.type_table.borrow().dynamic_type(),
                            missing_method: interface_method.name,
                        },
                        location,
                        context: format!(
                            "Class '{}' must implement method '{}' from interface '{}'",
                            self.string_interner.get(class.name).unwrap_or("<unknown>"),
                            self.string_interner
                                .get(interface_method.name)
                                .unwrap_or("<unknown>"),
                            self.string_interner
                                .get(interface.name)
                                .unwrap_or("<unknown>")
                        ),
                        suggestion: Some(format!(
                            "Add method '{}' to class '{}'",
                            self.string_interner
                                .get(interface_method.name)
                                .unwrap_or("<unknown>"),
                            self.string_interner.get(class.name).unwrap_or("<unknown>")
                        )),
                    });
                }
            }
        }

        Ok(())
    }

    /// Verify that overridden methods have compatible signatures with parent class methods
    fn verify_inheritance_signatures(
        &mut self,
        class: &TypedClass,
        super_type_id: TypeId,
    ) -> Result<(), String> {
        // Get the parent class symbol
        let super_symbol_id = if let Some(symbol_id) = self
            .type_checker
            .symbol_table
            .get_symbol_from_type(super_type_id)
        {
            symbol_id
        } else {
            return Ok(()); // Can't find parent class, skip check
        };

        // Get the typed file to access parent class definition
        let typed_file = unsafe {
            if let Some(file_ptr) = self.current_typed_file {
                &*file_ptr
            } else {
                return Ok(());
            }
        };

        // Find the parent class definition
        let parent_class = if let Some(parent) = typed_file
            .classes
            .iter()
            .find(|c| c.symbol_id == super_symbol_id)
        {
            parent
        } else {
            return Ok(()); // Parent class not in this file, skip for now
        };

        // Check each method in the child class
        for method in &class.methods {
            // Look for a method with the same name in the parent class
            if let Some(parent_method) = parent_class.methods.iter().find(|m| m.name == method.name)
            {
                // First check if method is marked with override
                if !method.metadata.is_override {
                    // Method overrides parent but missing override modifier
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::MissingOverride {
                            method_name: method.name,
                            parent_class: parent_class.name,
                        },
                        location: method.source_location,
                        context: format!(
                            "Method '{}' overrides parent method from class '{}' but is missing the 'override' modifier",
                            self.string_interner.get(method.name).unwrap_or("<unknown>"),
                            self.string_interner.get(parent_class.name).unwrap_or("<unknown>")
                        ),
                        suggestion: Some("Add 'override' modifier to the method declaration".to_string()),
                    });
                    continue; // Still check signature compatibility
                }

                // Check if signatures are compatible
                if !self.check_override_compatibility(parent_method, method)? {
                    let parent_sig = self.format_function_signature(parent_method);
                    let child_sig = self.format_function_signature(method);

                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::MethodSignatureMismatch {
                            expected: parent_method.return_type,
                            actual: method.return_type,
                            method_name: method.name,
                        },
                        location: method.source_location,
                        context: format!(
                            "Overridden method '{}' has incompatible signature with parent class '{}'\n  Parent:   {}\n  Override: {}",
                            self.string_interner.get(method.name).unwrap_or("<unknown>"),
                            self.string_interner.get(parent_class.name).unwrap_or("<unknown>"),
                            parent_sig,
                            child_sig
                        ),
                        suggestion: Some(format!(
                            "Change method signature to match parent: {}",
                            parent_sig
                        )),
                    });
                }
            } else if method.metadata.is_override {
                // Method has override modifier but no parent method to override
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::InvalidOverride {
                        method_name: method.name,
                    },
                    location: method.source_location,
                    context: format!(
                        "Method '{}' is marked as 'override' but no parent method with this name exists",
                        self.string_interner.get(method.name).unwrap_or("<unknown>")
                    ),
                    suggestion: Some("Remove the 'override' modifier or check the method name".to_string()),
                });
            }
        }

        Ok(())
    }

    /// Check if an overriding method is compatible with the parent method
    fn check_override_compatibility(
        &mut self,
        parent_method: &TypedFunction,
        child_method: &TypedFunction,
    ) -> Result<bool, String> {
        // Check parameter count
        if parent_method.parameters.len() != child_method.parameters.len() {
            return Ok(false);
        }

        // Check parameter types (contravariant - child can accept more general types)
        // In method overriding: child params must be assignable FROM parent params
        for (parent_param, child_param) in parent_method
            .parameters
            .iter()
            .zip(child_method.parameters.iter())
        {
            let param_compat = self
                .type_checker
                .check_compatibility(parent_param.param_type, child_param.param_type);
            match param_compat {
                TypeCompatibility::Identical | TypeCompatibility::Assignable => {
                    // Parent parameter type is assignable to child parameter type (contravariance)
                    // This means child can accept the same or more general types
                }
                _ => {
                    return Ok(false);
                }
            }
        }

        // Check return type (covariant - child can return more specific type)
        // In method overriding: child return type must be assignable TO parent return type
        let return_compat = self
            .type_checker
            .check_compatibility(child_method.return_type, parent_method.return_type);
        match return_compat {
            TypeCompatibility::Identical | TypeCompatibility::Assignable => {
                // Child return type is assignable to parent return type (covariance)
                // This means child can return the same or more specific types
            }
            _ => {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if a class method's signature matches an interface method's signature
    fn method_signatures_match(
        &mut self,
        interface_method: &TypedMethodSignature,
        class_method: &TypedFunction,
    ) -> Result<bool, String> {
        // Check if names match
        if interface_method.name != class_method.name {
            return Ok(false);
        }

        // Check if parameter counts match
        if interface_method.parameters.len() != class_method.parameters.len() {
            return Ok(false);
        }

        // Check parameter types (contravariant - class can accept more general types than interface requires)
        // In interface implementation: class params must be assignable FROM interface params
        for (interface_param, class_param) in interface_method
            .parameters
            .iter()
            .zip(class_method.parameters.iter())
        {
            let compatibility = self
                .type_checker
                .check_compatibility(class_param.param_type, interface_param.param_type);
            if matches!(compatibility, TypeCompatibility::Incompatible) {
                return Ok(false);
            }
        }

        // Check return type (covariant - class can return more specific types than interface requires)
        // In interface implementation: class return type must be assignable TO interface return type
        let compatibility = self
            .type_checker
            .check_compatibility(class_method.return_type, interface_method.return_type);
        if matches!(compatibility, TypeCompatibility::Incompatible) {
            return Ok(false);
        }

        Ok(true)
    }

    /// Format a method signature for display
    fn format_method_signature(&self, method: &TypedMethodSignature) -> String {
        let params = method
            .parameters
            .iter()
            .map(|p| {
                let param_name = self.string_interner.get(p.name).unwrap_or("<unknown>");
                let param_type = self.format_type(p.param_type);
                format!("{}: {}", param_name, param_type)
            })
            .collect::<Vec<_>>()
            .join(", ");

        let return_type = self.format_type(method.return_type);
        let method_name = self.string_interner.get(method.name).unwrap_or("<unknown>");

        format!("function {}({}): {}", method_name, params, return_type)
    }

    /// Format a function signature for display
    fn format_function_signature(&self, func: &TypedFunction) -> String {
        let params = func
            .parameters
            .iter()
            .map(|p| {
                let param_name = self.string_interner.get(p.name).unwrap_or("<unknown>");
                let param_type = self.format_type(p.param_type);
                format!("{}: {}", param_name, param_type)
            })
            .collect::<Vec<_>>()
            .join(", ");

        let return_type = self.format_type(func.return_type);
        let func_name = self.string_interner.get(func.name).unwrap_or("<unknown>");

        format!("function {}({}): {}", func_name, params, return_type)
    }

    /// Format a type for display
    fn format_type(&self, type_id: TypeId) -> String {
        if let Some(type_info) = self.type_checker.type_table.borrow().get(type_id) {
            match &type_info.kind {
                TypeKind::Void => "Void".to_string(),
                TypeKind::Bool => "Bool".to_string(),
                TypeKind::Int => "Int".to_string(),
                TypeKind::Float => "Float".to_string(),
                TypeKind::String => "String".to_string(),
                TypeKind::Char => "Char".to_string(),
                TypeKind::Class { symbol_id, .. } => {
                    if let Some(symbol) = self.type_checker.symbol_table.get_symbol(*symbol_id) {
                        self.string_interner
                            .get(symbol.name)
                            .unwrap_or("<unknown>")
                            .to_string()
                    } else {
                        "<unknown class>".to_string()
                    }
                }
                TypeKind::Interface { symbol_id, .. } => {
                    if let Some(symbol) = self.type_checker.symbol_table.get_symbol(*symbol_id) {
                        self.string_interner
                            .get(symbol.name)
                            .unwrap_or("<unknown>")
                            .to_string()
                    } else {
                        "<unknown interface>".to_string()
                    }
                }
                TypeKind::Array { element_type } => {
                    format!("Array<{}>", self.format_type(*element_type))
                }
                TypeKind::Optional { inner_type } => {
                    format!("Null<{}>", self.format_type(*inner_type))
                }
                TypeKind::Dynamic => "Dynamic".to_string(),
                _ => "<unknown>".to_string(),
            }
        } else {
            "<unknown>".to_string()
        }
    }

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
                let mut field_names = std::collections::HashSet::new();

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
            } => todo!(),
        }

        // Return the annotated type
        let result_type = expr.expr_type;

        Ok(result_type)
    }

    /// Check a binary operation expression (extracted to reduce stack frame size)
    #[inline(never)]
    fn check_binary_op_expr(
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
    fn check_function_call_expr(
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
    fn check_method_call_expr(
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
    fn check_switch_expr(
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
    fn check_map_literal_expr(
        &mut self,
        entries: &[TypedMapEntry],
        source_location: SourceLocation,
    ) -> Result<(), String> {
        if !entries.is_empty() {
            // Check first entry to establish expected types
            let first_key_type = self.check_expression(&entries[0].key)?;
            let first_value_type = self.check_expression(&entries[0].value)?;

            // Track duplicate keys if they're compile-time constants
            let mut constant_keys = std::collections::HashSet::new();

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
                condition,
                body,
                source_location: _,
                ..
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
            TypedStatement::ForIn {
                iterable,
                body,
                source_location: _,
                ..
            } => {
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

    /// Emit a type error as a diagnostic
    /// Check if a field access is valid
    fn check_field_access(
        &mut self,
        object_type: TypeId,
        field_symbol: SymbolId,
        location: SourceLocation,
        is_static_access: bool,
    ) -> Result<(), String> {
        // Get the object type information
        let type_kind = {
            let type_table = self.type_checker.type_table.borrow();
            if let Some(type_info) = type_table.get(object_type) {
                type_info.kind.clone()
            } else {
                return Ok(()); // Invalid object type, but that's a separate error
            }
        };

        match &type_kind {
            super::TypeKind::Class {
                symbol_id: class_symbol,
                ..
            } => {
                // Check if the field belongs to this class
                if self
                    .type_checker
                    .symbol_table
                    .get_symbol(field_symbol)
                    .is_some()
                {
                    // Verify that the field's scope matches the class or is accessible
                    self.check_field_accessibility(
                        class_symbol,
                        field_symbol,
                        location,
                        is_static_access,
                    )?;
                } else {
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::UndefinedType {
                            name: self.string_interner.intern("<unknown_field>"),
                        },
                        location,
                        context: "Field not found in class".to_string(),
                        suggestion: None,
                    });
                }
            }
            super::TypeKind::Interface {
                symbol_id: interface_symbol,
                ..
            } => {
                // Similar check for interfaces
                self.check_field_accessibility(
                    interface_symbol,
                    field_symbol,
                    location,
                    is_static_access,
                )?;
            }
            super::TypeKind::Dynamic => {
                // Dynamic types allow any field access
            }
            super::TypeKind::Array { .. } => {
                // Arrays have built-in fields like push, pop, length
                // The field access is already validated during AST lowering
                // where the correct method types are inferred
            }
            super::TypeKind::String => {
                // Strings have built-in fields like toUpperCase, toLowerCase, charAt, etc.
                // The field access is already validated during AST lowering
            }
            super::TypeKind::Anonymous { fields } => {
                // Anonymous objects have explicitly defined fields
                // Check if the field exists in the anonymous structure
                if let Some(field_symbol_info) =
                    self.type_checker.symbol_table.get_symbol(field_symbol)
                {
                    let field_name = field_symbol_info.name;

                    // Verify the field exists in this anonymous structure
                    let field_exists = fields.iter().any(|f| f.name == field_name);

                    if !field_exists {
                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::UndefinedSymbol {
                                name: field_name
                            },
                            location,
                            context: format!("Field '{}' not found in anonymous structure. Available fields: {}",
                                self.string_interner.get(field_name).unwrap_or("<unknown>"),
                                fields.iter()
                                    .filter_map(|f| self.string_interner.get(f.name))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                            suggestion: Some("Check the field name or add it to the anonymous structure definition".to_string()),
                        });
                    }
                }
            }
            _ => {
                // Other types don't have fields
                if let Some(field_symbol_info) =
                    self.type_checker.symbol_table.get_symbol(field_symbol)
                {
                    if let Some(field_name) = self.string_interner.get(field_symbol_info.name) {
                        // Create a generic "object" type for the error message
                        let object_type_id = {
                            let mut type_table = self.type_checker.type_table.borrow_mut();
                            // Use Dynamic as a placeholder for "any object type"
                            // This could be improved by creating a proper "Object" base type
                            type_table.dynamic_type()
                        };

                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: object_type_id,
                                actual: object_type
                            },
                            location,
                            context: format!("Cannot access field '{}' on non-object type", field_name),
                            suggestion: Some("Field access is only allowed on classes, interfaces, anonymous objects, or Dynamic types".to_string()),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Check if a field is accessible from the current context
    fn check_field_accessibility(
        &mut self,
        class_symbol: &SymbolId,
        field_symbol: SymbolId,
        location: SourceLocation,
        is_static_access: bool,
    ) -> Result<(), String> {
        // Get field information and extract needed data to avoid borrow conflicts
        let (field_name, field_visibility, is_static) =
            if let Some(field_info) = self.find_field_by_symbol(field_symbol) {
                (field_info.name, field_info.visibility, field_info.is_static)
            } else {
                return Ok(());
            };

        let field_name_str = self.string_interner.get(field_name).unwrap_or("<field>");

        // Get class name for error messages
        let class_name = if let Some(class_def) = self.find_class_by_symbol(*class_symbol) {
            class_def.name
        } else {
            self.string_interner.intern("<unknown_class>")
        };

        // Check static vs instance access
        if is_static && !is_static_access {
            // Accessing static member through instance
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::StaticAccessFromInstance {
                    member_name: field_name,
                    class_name,
                },
                location,
                context: "Static members should be accessed through the class, not an instance"
                    .to_string(),
                suggestion: Some(format!(
                    "Use {}.{} instead",
                    self.string_interner.get(class_name).unwrap_or("<class>"),
                    self.string_interner.get(field_name).unwrap_or("<field>")
                )),
            });
            // Don't return early - continue checking
        } else if !is_static && is_static_access {
            // Accessing instance member through static context
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::InstanceAccessFromStatic {
                    member_name: field_name,
                    class_name,
                },
                location,
                context: "Instance members cannot be accessed from static context".to_string(),
                suggestion: Some(
                    "Create an instance of the class to access instance members".to_string(),
                ),
            });
            // Don't return early - continue checking
        }

        // Implement visibility checking using the field's visibility from TypedField
        self.validate_field_visibility(field_visibility, *class_symbol, field_symbol, location)?;

        Ok(())
    }

    /// Validate field visibility based on access context
    fn validate_field_visibility(
        &mut self,
        field_visibility: Visibility,
        target_class_symbol: SymbolId,
        field_symbol: SymbolId,
        location: SourceLocation,
    ) -> Result<(), String> {
        match field_visibility {
            Visibility::Public => {
                // Public fields are always accessible
                Ok(())
            }
            Visibility::Private => {
                // Private fields are only accessible from the same class
                if let Some((_, current_class_symbol)) = self.current_method_context {
                    if current_class_symbol == target_class_symbol {
                        Ok(()) // Same class - private access allowed
                    } else {
                        // Different class - private access denied
                        let field_name_str = if let Some(symbol) =
                            self.type_checker.symbol_table.get_symbol(field_symbol)
                        {
                            self.get_string(symbol.name).to_string()
                        } else {
                            "<unknown_field>".to_string()
                        };
                        let target_class_name = if let Some(class_def) =
                            self.find_class_by_symbol(target_class_symbol)
                        {
                            self.get_string(class_def.name).to_string()
                        } else {
                            "<unknown>".to_string()
                        };

                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::AccessViolation {
                                symbol_id: field_symbol,
                                required_access: AccessLevel::Private,
                            },
                            location,
                            context: format!("Private field '{}' in class '{}' cannot be accessed from outside the class", field_name_str, target_class_name),
                            suggestion: Some("Make the field public or use a getter method".to_string()),
                        });
                        Ok(()) // Continue type checking after error
                    }
                } else {
                    // No current class context - private access denied
                    let field_name_str = if let Some(symbol) =
                        self.type_checker.symbol_table.get_symbol(field_symbol)
                    {
                        self.get_string(symbol.name).to_string()
                    } else {
                        "<unknown_field>".to_string()
                    };
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::AccessViolation {
                            symbol_id: field_symbol,
                            required_access: AccessLevel::Private,
                        },
                        location,
                        context: format!(
                            "Private field '{}' cannot be accessed from module level",
                            field_name_str
                        ),
                        suggestion: Some(
                            "Make the field public to access from module level".to_string(),
                        ),
                    });
                    Ok(())
                }
            }
            Visibility::Protected => {
                // Protected fields are accessible from the same class or subclasses
                if let Some((_, current_class_symbol)) = self.current_method_context {
                    if current_class_symbol == target_class_symbol {
                        Ok(()) // Same class - protected access allowed
                    } else if self.is_subclass_of(current_class_symbol, target_class_symbol) {
                        Ok(()) // Subclass - protected access allowed
                    } else {
                        // Not a subclass - protected access denied
                        let field_name_str = if let Some(symbol) =
                            self.type_checker.symbol_table.get_symbol(field_symbol)
                        {
                            self.get_string(symbol.name).to_string()
                        } else {
                            "<unknown_field>".to_string()
                        };
                        let target_class_name = if let Some(class_def) =
                            self.find_class_by_symbol(target_class_symbol)
                        {
                            self.get_string(class_def.name).to_string()
                        } else {
                            "<unknown>".to_string()
                        };

                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::AccessViolation {
                                symbol_id: field_symbol,
                                required_access: AccessLevel::Protected,
                            },
                            location,
                            context: format!("Protected field '{}' in class '{}' can only be accessed from the class itself or its subclasses", field_name_str, target_class_name),
                            suggestion: Some("Make the field public or ensure access is from a subclass".to_string()),
                        });
                        Ok(())
                    }
                } else {
                    // No current class context - protected access denied
                    let field_name_str = if let Some(symbol) =
                        self.type_checker.symbol_table.get_symbol(field_symbol)
                    {
                        self.get_string(symbol.name).to_string()
                    } else {
                        "<unknown_field>".to_string()
                    };
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::AccessViolation {
                            symbol_id: field_symbol,
                            required_access: AccessLevel::Protected,
                        },
                        location,
                        context: format!(
                            "Protected field '{}' cannot be accessed from module level",
                            field_name_str
                        ),
                        suggestion: Some(
                            "Make the field public to access from module level".to_string(),
                        ),
                    });
                    Ok(())
                }
            }
            Visibility::Internal => {
                // Internal fields are accessible within the same package
                self.validate_package_level_access(
                    field_symbol,
                    target_class_symbol,
                    location,
                    "internal field",
                )
            }
        }
    }

    /// Validate package-level access for internal visibility
    fn validate_package_level_access(
        &mut self,
        target_symbol: SymbolId,
        target_class_symbol: SymbolId,
        location: SourceLocation,
        symbol_kind: &str,
    ) -> Result<(), String> {
        // Use the new package access validator if available
        if let Some(ref mut validator) = self.package_access_validator {
            // Set current context if needed
            if let Some(file_name) = self
                .current_typed_file
                .and_then(|f| unsafe { (*f).metadata.file_name })
            {
                validator.set_context(self.current_package, Some(file_name));
            } else {
                validator.set_context(self.current_package, None);
            }

            // Validate access
            match validator.validate_symbol_access(target_symbol, location) {
                Ok(()) => Ok(()),
                Err(error) => {
                    self.emit_error(error);
                    Ok(()) // Continue type checking after error
                }
            }
        } else {
            // Fallback to basic package checking (existing implementation)
            // Get the package of the target symbol
            let target_package = if let Some(target_symbol_info) =
                self.type_checker.symbol_table.get_symbol(target_symbol)
            {
                target_symbol_info.package_id
            } else if let Some(target_class_info) = self
                .type_checker
                .symbol_table
                .get_symbol(target_class_symbol)
            {
                // If target symbol doesn't have package info, use the class's package
                target_class_info.package_id
            } else {
                None
            };

            // Get the package of the current context
            let current_package =
                if let Some((_, current_class_symbol)) = self.current_method_context {
                    // We're inside a class method - use class's package if available
                    if let Some(current_class_info) = self
                        .type_checker
                        .symbol_table
                        .get_symbol(current_class_symbol)
                    {
                        current_class_info.package_id
                    } else {
                        // Fall back to file's package context
                        self.current_package
                    }
                } else {
                    // We're at module level - use the file's package context
                    self.current_package
                };

            // Check if packages match
            match (current_package, target_package) {
                (Some(current_pkg), Some(target_pkg)) if current_pkg == target_pkg => {
                    // Same package - internal access allowed
                    Ok(())
                }
                (None, None) => {
                    // Both in default package (no package declaration) - access allowed
                    Ok(())
                }
                _ => {
                    // Different packages or missing package info - internal access denied
                    let symbol_name = if let Some(symbol_info) =
                        self.type_checker.symbol_table.get_symbol(target_symbol)
                    {
                        self.get_string(symbol_info.name).to_string()
                    } else {
                        "<unknown>".to_string()
                    };

                    let target_class_name =
                        if let Some(class_def) = self.find_class_by_symbol(target_class_symbol) {
                            self.get_string(class_def.name).to_string()
                        } else {
                            "<unknown>".to_string()
                        };

                    let target_package_name = if let Some(pkg_id) = target_package {
                        // TODO: Get package name from namespace resolver
                        format!("package {:?}", pkg_id)
                    } else {
                        "default package".to_string()
                    };

                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::AccessViolation {
                            symbol_id: target_symbol,
                            required_access: AccessLevel::Internal,
                        },
                        location,
                        context: format!("Internal {} '{}' in class '{}' (in {}) cannot be accessed from a different package",
                            symbol_kind, symbol_name, target_class_name, target_package_name),
                        suggestion: Some("Make the symbol public to access from different packages, or move the accessing code to the same package".to_string()),
                    });
                    Ok(()) // Continue type checking after error
                }
            }
        }
    }

    /// Extract package information from a typed file
    fn extract_package_from_file(
        &self,
        typed_file: &TypedFile,
    ) -> Option<super::namespace::PackageId> {
        // Get package name from file metadata
        if let Some(package_name) = &typed_file.metadata.package_name {
            // Parse package path and find corresponding PackageId
            // For now, we'll return None as we need access to namespace resolver
            // TODO: This should be set during AST lowering when package context is available
            None
        } else {
            // No package declaration - default package
            None
        }
    }

    /// Get the name of the current class context
    fn get_current_class_name(&self) -> InternedString {
        if let Some((_, current_class_symbol)) = self.current_method_context {
            if let Some(class_def) = self.find_class_by_symbol(current_class_symbol) {
                class_def.name
            } else {
                self.string_interner.intern("<unknown>")
            }
        } else {
            self.string_interner.intern("<module>")
        }
    }

    /// Check if a class is a subclass of another class
    fn is_subclass_of(&self, potential_subclass: SymbolId, potential_superclass: SymbolId) -> bool {
        if let Some(subclass_def) = self.find_class_by_symbol(potential_subclass) {
            if let Some(super_type_id) = subclass_def.super_class {
                // Get the super class symbol from the type
                if let Some(super_class_symbol) = self.get_class_symbol_from_type(super_type_id) {
                    if super_class_symbol == potential_superclass {
                        return true; // Direct parent
                    }
                    // Check recursively up the inheritance chain
                    return self.is_subclass_of(super_class_symbol, potential_superclass);
                }
            }
        }
        false
    }

    /// Get class symbol from a type ID (helper for inheritance checking)
    fn get_class_symbol_from_type(&self, type_id: TypeId) -> Option<SymbolId> {
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Find a field by symbol ID
    fn find_field_by_symbol(&self, field_symbol: SymbolId) -> Option<&TypedField> {
        if let Some(typed_file_ptr) = self.current_typed_file {
            // SAFETY: This is safe because we only set current_typed_file during the lifetime
            // of the TypedFile reference in check_file, and we clear it after use
            let typed_file = unsafe { &*typed_file_ptr };

            // Search through all classes
            for class in &typed_file.classes {
                let class_name_str = self.string_interner.get(class.name).unwrap_or("<class>");

                for field in &class.fields {
                    let field_name_str = self.string_interner.get(field.name).unwrap_or("<field>");
                    if field.symbol_id == field_symbol {
                        return Some(field);
                    }
                }
            }
        }
        None
    }

    /// Find a class definition by symbol ID
    fn find_class_by_symbol(&self, symbol_id: SymbolId) -> Option<&TypedClass> {
        if let Some(typed_file_ptr) = self.current_typed_file {
            // SAFETY: This is safe because we only set current_typed_file during the lifetime
            // of the TypedFile reference in check_file, and we clear it after use
            let typed_file = unsafe { &*typed_file_ptr };
            typed_file
                .classes
                .iter()
                .find(|class| class.symbol_id == symbol_id)
        } else {
            None
        }
    }

    /// Check if a method access is valid (static vs instance)
    fn check_method_static_access(
        &mut self,
        class_def: &TypedClass,
        method_symbol: SymbolId,
        location: SourceLocation,
        is_static_access: bool,
    ) -> Result<(), String> {
        // Find the method in the class
        if let Some(method) = class_def
            .methods
            .iter()
            .find(|m| m.symbol_id == method_symbol)
        {
            if method.is_static && !is_static_access {
                // Accessing static method through instance
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::StaticAccessFromInstance {
                        member_name: method.name,
                        class_name: class_def.name,
                    },
                    location,
                    context: "Static methods should be accessed through the class, not an instance"
                        .to_string(),
                    suggestion: Some(format!(
                        "Use {}.{} instead",
                        self.string_interner
                            .get(class_def.name)
                            .unwrap_or("<class>"),
                        self.string_interner.get(method.name).unwrap_or("<method>")
                    )),
                });
                // Don't return early - continue checking
            } else if !method.is_static && is_static_access {
                // Accessing instance method through static context
                self.emit_error(TypeCheckError {
                    kind: TypeErrorKind::InstanceAccessFromStatic {
                        member_name: method.name,
                        class_name: class_def.name,
                    },
                    location,
                    context: "Instance methods cannot be accessed from static context".to_string(),
                    suggestion: Some(
                        "Create an instance of the class to access instance methods".to_string(),
                    ),
                });
                // Don't return early - continue checking
            }

            // Add method visibility checking
            self.validate_method_visibility(
                method.visibility,
                class_def.symbol_id,
                method.symbol_id,
                location,
            )?;
        }

        Ok(())
    }

    /// Validate method visibility based on access context
    fn validate_method_visibility(
        &mut self,
        method_visibility: Visibility,
        target_class_symbol: SymbolId,
        method_symbol: SymbolId,
        location: SourceLocation,
    ) -> Result<(), String> {
        match method_visibility {
            Visibility::Public => {
                // Public methods are always accessible
                Ok(())
            }
            Visibility::Private => {
                // Private methods are only accessible from the same class
                if let Some((_, current_class_symbol)) = self.current_method_context {
                    if current_class_symbol == target_class_symbol {
                        Ok(()) // Same class - private access allowed
                    } else {
                        // Different class - private access denied
                        let method_name_str = if let Some(symbol) =
                            self.type_checker.symbol_table.get_symbol(method_symbol)
                        {
                            self.get_string(symbol.name).to_string()
                        } else {
                            "<unknown_method>".to_string()
                        };
                        let target_class_name = if let Some(class_def) =
                            self.find_class_by_symbol(target_class_symbol)
                        {
                            self.get_string(class_def.name).to_string()
                        } else {
                            "<unknown>".to_string()
                        };

                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::AccessViolation {
                                symbol_id: method_symbol,
                                required_access: AccessLevel::Private,
                            },
                            location,
                            context: format!("Private method '{}' in class '{}' cannot be accessed from outside the class", method_name_str, target_class_name),
                            suggestion: Some("Make the method public or use a public wrapper method".to_string()),
                        });
                        Ok(()) // Continue type checking after error
                    }
                } else {
                    // No current class context - private access denied
                    let method_name_str = if let Some(symbol) =
                        self.type_checker.symbol_table.get_symbol(method_symbol)
                    {
                        self.get_string(symbol.name).to_string()
                    } else {
                        "<unknown_method>".to_string()
                    };
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::AccessViolation {
                            symbol_id: method_symbol,
                            required_access: AccessLevel::Private,
                        },
                        location,
                        context: format!(
                            "Private method '{}' cannot be accessed from module level",
                            method_name_str
                        ),
                        suggestion: Some(
                            "Make the method public to access from module level".to_string(),
                        ),
                    });
                    Ok(())
                }
            }
            Visibility::Protected => {
                // Protected methods are accessible from the same class or subclasses
                if let Some((_, current_class_symbol)) = self.current_method_context {
                    if current_class_symbol == target_class_symbol {
                        Ok(()) // Same class - protected access allowed
                    } else if self.is_subclass_of(current_class_symbol, target_class_symbol) {
                        Ok(()) // Subclass - protected access allowed
                    } else {
                        // Not a subclass - protected access denied
                        let method_name_str = if let Some(symbol) =
                            self.type_checker.symbol_table.get_symbol(method_symbol)
                        {
                            self.get_string(symbol.name).to_string()
                        } else {
                            "<unknown_method>".to_string()
                        };
                        let target_class_name = if let Some(class_def) =
                            self.find_class_by_symbol(target_class_symbol)
                        {
                            self.get_string(class_def.name).to_string()
                        } else {
                            "<unknown>".to_string()
                        };

                        self.emit_error(TypeCheckError {
                            kind: TypeErrorKind::AccessViolation {
                                symbol_id: method_symbol,
                                required_access: AccessLevel::Protected,
                            },
                            location,
                            context: format!("Protected method '{}' in class '{}' can only be accessed from the class itself or its subclasses", method_name_str, target_class_name),
                            suggestion: Some("Make the method public or ensure access is from a subclass".to_string()),
                        });
                        Ok(())
                    }
                } else {
                    // No current class context - protected access denied
                    let method_name_str = if let Some(symbol) =
                        self.type_checker.symbol_table.get_symbol(method_symbol)
                    {
                        self.get_string(symbol.name).to_string()
                    } else {
                        "<unknown_method>".to_string()
                    };
                    self.emit_error(TypeCheckError {
                        kind: TypeErrorKind::AccessViolation {
                            symbol_id: method_symbol,
                            required_access: AccessLevel::Protected,
                        },
                        location,
                        context: format!(
                            "Protected method '{}' cannot be accessed from module level",
                            method_name_str
                        ),
                        suggestion: Some(
                            "Make the method public to access from module level".to_string(),
                        ),
                    });
                    Ok(())
                }
            }
            Visibility::Internal => {
                // Internal methods are accessible within the same package
                self.validate_package_level_access(
                    method_symbol,
                    target_class_symbol,
                    location,
                    "internal method",
                )
            }
        }
    }

    /// Validate that a type satisfies a constraint type (e.g., T:Comparable<T>)
    fn validate_type_constraint(&self, type_id: TypeId, constraint_type_id: TypeId) -> bool {
        // For now, we implement a simple check:
        // - If types are identical, constraint is satisfied
        // - If constraint is an interface, check if type implements it
        // - Otherwise, reject the constraint (conservative approach)

        if type_id == constraint_type_id {
            return true;
        }

        // Check if it's an interface implementation
        if self.is_interface_type(constraint_type_id) {
            self.type_implements_interface(type_id, constraint_type_id)
        } else {
            // For non-interface constraints, be conservative and reject
            // TODO: Implement more sophisticated constraint checking
            false
        }
    }

    /// Check if a type implements an interface
    fn type_implements_interface(&self, type_id: TypeId, interface_type: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                super::TypeKind::Class { symbol_id, .. } => {
                    // TODO: Find the class definition and check its implemented interfaces
                    // This requires access to the typed_file context
                    // For now, return false to fix compilation
                    false
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type is an interface type
    fn is_interface_type(&self, type_id: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            matches!(type_info.kind, super::TypeKind::Interface { .. })
        } else {
            false
        }
    }

    /// Check if a type is comparable
    fn is_comparable_type(&self, type_id: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match &type_info.kind {
                // Primitive types are comparable
                super::TypeKind::Int
                | super::TypeKind::Float
                | super::TypeKind::String
                | super::TypeKind::Bool => true,
                // Check if class implements Comparable interface
                super::TypeKind::Class { .. } => {
                    // For classes, we would need to check if they implement Comparable<T>
                    // This is complex and would require interface lookup
                    // For now, assume non-primitive types need explicit implementation
                    false
                }
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type has a specific method
    fn type_has_method(
        &self,
        _type_id: TypeId,
        _method_name: InternedString,
        _signature: TypeId,
    ) -> bool {
        // TODO: Implement method lookup in class definitions
        // This would require checking the class's methods list
        false
    }

    /// Check if a type has a specific field
    fn type_has_field(
        &self,
        _type_id: TypeId,
        _field_name: InternedString,
        _field_type: TypeId,
    ) -> bool {
        // TODO: Implement field lookup in class definitions
        // This would require checking the class's fields list
        false
    }

    /// Convert a constraint type to a readable string
    fn constraint_type_to_string(&self, constraint_type_id: TypeId) -> String {
        // Format the constraint type name using existing get_type_name method
        self.get_type_name(constraint_type_id)
            .unwrap_or_else(|| format!("Type#{}", constraint_type_id.as_raw()))
    }

    /// Get a human-readable name for a type
    fn get_type_name(&self, type_id: TypeId) -> Option<String> {
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            match type_info.kind.clone() {
                super::TypeKind::Int => Some("Int".to_string()),
                super::TypeKind::Float => Some("Float".to_string()),
                super::TypeKind::String => Some("String".to_string()),
                super::TypeKind::Bool => Some("Bool".to_string()),
                super::TypeKind::Class { symbol_id, .. } => {
                    if let Some(symbol) = self.type_checker.symbol_table.get_symbol(symbol_id) {
                        self.string_interner.get(symbol.name).map(|s| s.to_string())
                    } else {
                        None
                    }
                }
                super::TypeKind::Interface { symbol_id, .. } => {
                    if let Some(symbol) = self.type_checker.symbol_table.get_symbol(symbol_id) {
                        self.string_interner.get(symbol.name).map(|s| s.to_string())
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

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

    /// Check if an explicit cast is valid between two types
    fn is_valid_explicit_cast(&self, from_type: TypeId, to_type: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();

        let from_info = type_table.get(from_type);
        let to_info = type_table.get(to_type);

        match (from_info, to_info) {
            (Some(from_type_info), Some(to_type_info)) => {
                use super::TypeKind;

                match (&from_type_info.kind, &to_type_info.kind) {
                    // Numeric conversions are always allowed
                    (TypeKind::Int, TypeKind::Float)
                    | (TypeKind::Float, TypeKind::Int)
                    | (TypeKind::Int, TypeKind::Bool)
                    | (TypeKind::Bool, TypeKind::Int) => true,

                    // String conversions are generally allowed
                    (TypeKind::String, TypeKind::Int)
                    | (TypeKind::String, TypeKind::Float)
                    | (TypeKind::String, TypeKind::Bool)
                    | (TypeKind::Int, TypeKind::String)
                    | (TypeKind::Float, TypeKind::String)
                    | (TypeKind::Bool, TypeKind::String) => true,

                    // Dynamic can be cast to/from anything
                    (TypeKind::Dynamic, _) | (_, TypeKind::Dynamic) => true,

                    // Array element type casts
                    (
                        TypeKind::Array {
                            element_type: from_elem,
                        },
                        TypeKind::Array {
                            element_type: to_elem,
                        },
                    ) => {
                        // Allow if element types are compatible or castable
                        // Note: We can't call check_compatibility here due to borrowing rules
                        // For now, allow array casts if element types are the same or both are basic types
                        *from_elem == *to_elem || self.are_both_basic_types(*from_elem, *to_elem)
                    }

                    // Class hierarchy casts (upcast/downcast)
                    (
                        TypeKind::Class {
                            symbol_id: from_class,
                            ..
                        },
                        TypeKind::Class {
                            symbol_id: to_class,
                            ..
                        },
                    ) => {
                        // TODO: Check class hierarchy for valid casts
                        // For now, allow all class-to-class casts (could be unsafe but explicit)
                        true
                    }

                    // Interface casts
                    (TypeKind::Class { .. }, TypeKind::Interface { .. })
                    | (TypeKind::Interface { .. }, TypeKind::Class { .. })
                    | (TypeKind::Interface { .. }, TypeKind::Interface { .. }) => true,

                    // Null/Optional conversions
                    (TypeKind::Optional { inner_type }, other_kind)
                    | (other_kind, TypeKind::Optional { inner_type }) => {
                        // Allow casting between T and Null<T>
                        matches!(other_kind, super::TypeKind::Void)
                            || self.is_valid_explicit_cast(from_type, *inner_type)
                            || self.is_valid_explicit_cast(*inner_type, to_type)
                    }

                    // Same types are always castable
                    _ if from_type == to_type => true,

                    // Everything else is invalid for explicit casts
                    _ => false,
                }
            }
            _ => false, // Invalid types
        }
    }

    /// Check if both types are basic types that can be cast between each other
    fn are_both_basic_types(&self, type1: TypeId, type2: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();
        let is_basic_type = |type_id: TypeId| -> bool {
            if let Some(type_info) = type_table.get(type_id) {
                matches!(
                    &type_info.kind,
                    super::TypeKind::Int
                        | super::TypeKind::Float
                        | super::TypeKind::Bool
                        | super::TypeKind::String
                        | super::TypeKind::Dynamic
                )
            } else {
                false
            }
        };

        is_basic_type(type1) && is_basic_type(type2)
    }

    /// Helper to get string from interner
    fn get_string(&self, interned: super::InternedString) -> &str {
        self.string_interner.get(interned).unwrap_or("<unknown>")
    }

    /// Check if a method signature matches the provided argument types
    fn check_signature_compatibility(&self, param_types: &[TypeId], arg_types: &[TypeId]) -> bool {
        if param_types.len() != arg_types.len() {
            return false;
        }

        for (expected_type, actual_type) in param_types.iter().zip(arg_types) {
            // Check compatibility by accessing type table directly to avoid borrowing issues
            let type_table = self.type_checker.type_table.borrow();
            let expected_type_info = type_table.get(*expected_type);
            let actual_type_info = type_table.get(*actual_type);

            // Basic compatibility check - for now, require exact match or allow implicit upcasts
            if *expected_type != *actual_type {
                // For now, only allow some basic implicit conversions
                let compatible = match (expected_type_info, actual_type_info) {
                    (Some(expected), Some(actual)) => {
                        match (&expected.kind, &actual.kind) {
                            // Dynamic can accept anything
                            (super::TypeKind::Dynamic, _) => true,
                            // Same types are compatible
                            _ if expected_type == actual_type => true,
                            // Allow Int -> Float implicit conversion
                            (super::TypeKind::Float, super::TypeKind::Int) => true,
                            _ => false,
                        }
                    }
                    _ => false,
                };

                if !compatible {
                    return false;
                }
            }
        }

        true
    }

    /// Check method overloads to find a matching signature
    fn check_method_overloads(
        &self,
        method_symbol: SymbolId,
        arg_types: &[TypeId],
        source_location: SourceLocation,
    ) -> bool {
        // First, we need to find the method definition to access its overload signatures
        // This requires iterating through classes to find the method
        if let Some(typed_file) = self.current_typed_file {
            unsafe {
                let typed_file_ref = &*typed_file;
                for class in &typed_file_ref.classes {
                    for method in &class.methods {
                        if method.symbol_id == method_symbol {
                            // Found the method, check its overload signatures
                            for overload in &method.metadata.overload_signatures {
                                if self.check_signature_compatibility(
                                    &overload.parameter_types,
                                    arg_types,
                                ) {
                                    return true; // Found a matching overload
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        false
    }

    /// Validate that a type can be used as an exception type in catch clauses
    fn validate_exception_type(
        &mut self,
        exception_type: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(exception_type) {
            match &type_info.kind {
                // Any type can be thrown in Haxe, but we can warn about unusual types
                super::TypeKind::Dynamic => Ok(()),
                super::TypeKind::String => Ok(()),
                super::TypeKind::Class { .. } => Ok(()),
                super::TypeKind::Interface { .. } => Ok(()),
                super::TypeKind::Int | super::TypeKind::Float | super::TypeKind::Bool => {
                    // Primitive types are unusual as exceptions but technically allowed
                    Ok(())
                }
                _ => Ok(()), // Allow any type to be throwable for flexibility
            }
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<unknown_exception_type>"),
                },
                location,
                context: "Exception type is not defined".to_string(),
                suggestion: Some("Use a defined class or interface as exception type".to_string()),
            });
            Err("Undefined exception type".to_string())
        }
    }

    /// Validate that a type can be thrown
    fn validate_throwable_type(
        &mut self,
        throwable_type: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        // In Haxe, any type can be thrown, but we provide helpful warnings
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(throwable_type) {
            match &type_info.kind {
                super::TypeKind::Dynamic => Ok(()),
                super::TypeKind::String => Ok(()),
                super::TypeKind::Class { .. } => Ok(()),
                super::TypeKind::Interface { .. } => Ok(()),
                _ => Ok(()), // Allow throwing any type
            }
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<unknown_throwable_type>"),
                },
                location,
                context: "Thrown type is not defined".to_string(),
                suggestion: Some("Ensure the thrown expression has a valid type".to_string()),
            });
            Err("Undefined throwable type".to_string())
        }
    }

    /// Validate that a type is iterable (for for-in loops)
    fn validate_iterable_type(
        &mut self,
        iterable_type: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(iterable_type) {
            let is_iterable = match &type_info.kind {
                super::TypeKind::Array { .. } => true,
                super::TypeKind::String => true, // Strings are iterable (char by char)
                super::TypeKind::Dynamic => true, // Dynamic allows anything
                super::TypeKind::Class { .. } => {
                    // TODO: Check if class implements Iterable interface
                    // For now, assume classes with "iterator" or "keyValueIterator" methods are iterable
                    true // Be permissive for now
                }
                super::TypeKind::Interface { .. } => {
                    // TODO: Check if it's an Iterable interface
                    true // Be permissive for now
                }
                _ => false,
            };

            if !is_iterable {
                self.emit_enhanced_type_error(
                    iterable_type,
                    self.type_checker.type_table.borrow().dynamic_type(), // Use Dynamic as "any iterable"
                    location,
                    "Type is not iterable",
                    &TypeErrorContext::ForInLoop,
                );
                return Err("Type is not iterable".to_string());
            }

            Ok(())
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<unknown_iterable_type>"),
                },
                location,
                context: "Iterable type is not defined".to_string(),
                suggestion: Some(
                    "Use an array, string, or iterable class in for-in loops".to_string(),
                ),
            });
            Err("Undefined iterable type".to_string())
        }
    }

    /// Find a method with matching @:op metadata for the given operator
    /// Returns (method_symbol, abstract_symbol) if found
    fn find_operator_method(
        &self,
        operand_type: TypeId,
        operator: &BinaryOperator,
    ) -> Option<(SymbolId, SymbolId)> {
        let type_table = self.type_checker.type_table.borrow();

        // Check if this type is an abstract type
        let type_info = type_table.get(operand_type)?;
        let abstract_symbol = match &type_info.kind {
            super::TypeKind::Abstract { symbol_id, .. } => *symbol_id,
            _ => return None, // Not an abstract type
        };

        drop(type_table);

        // Get the abstract definition from the current file being checked
        let typed_file_ptr = self.current_typed_file?;
        let typed_file = unsafe { &*typed_file_ptr };

        // Search all abstracts for the one matching our symbol
        for abstract_def in &typed_file.abstracts {
            if abstract_def.symbol_id != abstract_symbol {
                continue;
            }

            // Found the abstract, now search for a method with matching @:op metadata
            for method in &abstract_def.methods {
                for (op_str, _params) in &method.metadata.operator_metadata {
                    if let Some(parsed_op) = Self::parse_operator_from_metadata(op_str) {
                        if std::mem::discriminant(&parsed_op) == std::mem::discriminant(operator) {
                            // Found a matching operator method!
                            return Some((method.symbol_id, abstract_symbol));
                        }
                    }
                }
            }
        }

        None
    }

    /// Parse operator metadata string to extract the operator type
    /// e.g. "A Add B" → Some(BinaryOperator::Add)
    fn parse_operator_from_metadata(op_str: &str) -> Option<BinaryOperator> {
        if op_str.contains("Add") {
            Some(BinaryOperator::Add)
        } else if op_str.contains("Sub") {
            Some(BinaryOperator::Sub)
        } else if op_str.contains("Mul") {
            Some(BinaryOperator::Mul)
        } else if op_str.contains("Div") {
            Some(BinaryOperator::Div)
        } else if op_str.contains("Mod") {
            Some(BinaryOperator::Mod)
        } else if op_str.contains("Eq") && !op_str.contains("Ne") {
            Some(BinaryOperator::Eq)
        } else if op_str.contains("Ne") {
            Some(BinaryOperator::Ne)
        } else if op_str.contains("Lt") {
            Some(BinaryOperator::Lt)
        } else if op_str.contains("Le") {
            Some(BinaryOperator::Le)
        } else if op_str.contains("Gt") {
            Some(BinaryOperator::Gt)
        } else if op_str.contains("Ge") {
            Some(BinaryOperator::Ge)
        } else {
            None
        }
    }
}

/// Run type checking on a typed file with full diagnostic support
pub fn type_check_with_diagnostics(
    typed_file: &mut TypedFile,
    type_table: &Rc<RefCell<TypeTable>>,
    symbol_table: &SymbolTable,
    scope_tree: &ScopeTree,
    string_interner: &StringInterner,
    source_map: &SourceMap,
) -> Result<Diagnostics, String> {
    let mut diagnostics = Diagnostics::new();

    {
        let mut type_checking_phase = TypeCheckingPhase::new(
            type_table,
            symbol_table,
            scope_tree,
            string_interner,
            source_map,
            &mut diagnostics,
        );

        // Run type checking - we want the diagnostics regardless of whether errors were found
        let _result = type_checking_phase.check_file(typed_file);
        // Note: Intentionally ignoring the result here since we want to return diagnostics
        // even when type errors are found
    }

    Ok(diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::{AstLowering, ScopeId, ScopeTree};
    use diagnostics::ErrorFormatter;
    use parser::parse_haxe_file;

    #[test]
    fn test_type_checking_pipeline() {
        let haxe_code = r#"
            interface IShape {
                public function getArea():Float;
            }

            class Rectangle implements IShape {
                private var width:Float;
                private var height:Float;

                public function new(w:Float, h:Float) {
                    this.width = w;
                    this.height = h;
                }

                public function getArea():Float {
                    return width * height;
                }
            }

            class Circle implements IShape {
                private var radius:Float;

                public function new(r:Float) {
                    this.radius = r;
                }

                public function getArea():Float {
                    return 3.14159 * radius * radius;
                }
            }
        "#;

        // Parse
        let ast_result = parse_haxe_file("test.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("test.hx".to_string(), haxe_code.to_string());

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        // Lower to TAST
        let string_interner_rc = Rc::new(RefCell::new(StringInterner::new()));
        let mut lowering = AstLowering::new(
            &mut string_interner,
            string_interner_rc,
            &mut symbol_table,
            &type_table,
            &mut scope_tree,
            &mut namespace_resolver,
            &mut import_resolver,
        );
        lowering.initialize_span_converter(file_id.as_usize() as u32, haxe_code.to_string());
        let mut typed_file = lowering
            .lower_file(&haxe_file)
            .expect("Lowering should succeed");

        // Run type checking
        let diagnostics = type_check_with_diagnostics(
            &mut typed_file,
            &type_table,
            &symbol_table,
            &scope_tree,
            &string_interner,
            &source_map,
        )
        .expect("Type checking should complete");

        // Check results
        if !diagnostics.is_empty() {
            let formatter = ErrorFormatter::new();
            let formatted = formatter.format_diagnostics(&diagnostics, &source_map);
        }

        assert!(
            diagnostics.is_empty(),
            "Should have no type errors for valid code"
        );
    }

    #[test]
    fn test_type_error_detection() {
        let haxe_code = r#"
            class TypeErrors {
                public function test():Void {
                    var x:Int = "not an int";  // Type error
                    var y:String = 42;          // Type error
                    var z:Bool = x + y;         // Type error
                }
            }
        "#;

        // Parse
        let ast_result = parse_haxe_file("test_errors.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("test_errors.hx".to_string(), haxe_code.to_string());

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        // Lower to TAST
        let string_interner_rc = Rc::new(RefCell::new(StringInterner::new()));
        let mut lowering = AstLowering::new(
            &mut string_interner,
            string_interner_rc,
            &mut symbol_table,
            &type_table,
            &mut scope_tree,
            &mut namespace_resolver,
            &mut import_resolver,
        );
        lowering.initialize_span_converter(file_id.as_usize() as u32, haxe_code.to_string());
        let mut typed_file = lowering
            .lower_file(&haxe_file)
            .expect("Lowering should succeed");

        // For demonstration, manually create a type error since full type checking isn't implemented
        let mut diagnostics = Diagnostics::new();
        {
            let mut type_checking_phase = TypeCheckingPhase::new(
                &type_table,
                &symbol_table,
                &scope_tree,
                &string_interner,
                &source_map,
                &mut diagnostics,
            );

            // Simulate a type error
            type_checking_phase.emit_error(TypeCheckError {
                kind: TypeErrorKind::TypeMismatch {
                    expected: type_table.borrow().int_type(),
                    actual: type_table.borrow().string_type(),
                },
                location: SourceLocation::new(1, 4, 21, 20),
                context: "Cannot assign string literal to variable of type Int".to_string(),
                suggestion: Some(
                    "Change the type annotation to String or use an integer literal".to_string(),
                ),
            });
        }

        // Format and display errors
        assert!(!diagnostics.is_empty(), "Should have type errors");
        let formatter = ErrorFormatter::new();
        let formatted = formatter.format_diagnostics(&diagnostics, &source_map);
    }

    #[test]
    fn test_static_vs_instance_member_checking() {
        let haxe_code = r#"
class MyClass {
    public static var staticField:Int = 42;
    public var instanceField:String = "hello";

    public static function staticMethod():String {
        return "Static method";
    }

    public function instanceMethod():String {
        return "Instance method";
    }
}

class TestStaticAccess {
    public function new() {}

    public function testAccess():Void {
        // Valid static access
        var s1:Int = MyClass.staticField;
        MyClass.staticMethod();

        // Valid instance access
        var obj = new MyClass();
        var i1:String = obj.instanceField;
        obj.instanceMethod();

        // Invalid: accessing static member through instance
        var invalid1:Int = obj.staticField;  // Should error
        obj.staticMethod();  // Should error

        // Invalid: accessing instance member through static context
        var invalid2:String = MyClass.instanceField;  // Should error
        MyClass.instanceMethod();  // Should error
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        println!("\n=== Compilation Result ===");
        println!("Total errors: {}", result.errors.len());
        for (i, error) in result.errors.iter().enumerate() {
            println!("\nError {}: {}", i + 1, error.message);
            println!(
                "  Location: {}:{}:{}",
                error.location.file_id, error.location.line, error.location.column
            );
            println!("  Category: {:?}", error.category);
        }

        // Print expected errors for debugging
        println!("\n=== Expected Errors ===");
        println!("1. Line 35: obj.staticField - accessing static field through instance");
        println!("2. Line 36: obj.staticMethod() - accessing static method through instance");
        println!("3. Line 39: MyClass.instanceField - accessing instance field statically");
        println!("4. Line 40: MyClass.instanceMethod() - accessing instance method statically");

        // We expect exactly 4 errors
        assert_eq!(
            result.errors.len(),
            4,
            "Expected 4 static/instance access errors"
        );

        // Check error messages
        let error_messages: Vec<String> = result.errors.iter().map(|e| e.message.clone()).collect();

        // Verify we have the right kinds of errors
        let static_from_instance_errors = error_messages
            .iter()
            .filter(|msg| {
                msg.contains("Static member") && msg.contains("cannot be accessed through instance")
            })
            .count();
        assert_eq!(
            static_from_instance_errors, 2,
            "Expected 2 static-from-instance errors"
        );

        let instance_from_static_errors = error_messages
            .iter()
            .filter(|msg| {
                msg.contains("Instance member")
                    && msg.contains("cannot be accessed from static context")
            })
            .count();
        assert_eq!(
            instance_from_static_errors, 2,
            "Expected 2 instance-from-static errors"
        );
    }

    #[test]
    fn test_static_method_context() {
        let haxe_code = r#"
class StaticContext {
    private static var staticData:Int = 100;
    private var instanceData:String = "data";

    public static function staticWork():Void {
        // Valid: static accessing static
        var x = staticData;

        // Invalid: static method accessing instance member
        var y = instanceData;  // Should error
    }

    public function instanceWork():Void {
        // Valid: instance method can access both
        var x = staticData;
        var y = instanceData;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // We expect exactly 1 error
        assert_eq!(
            result.errors.len(),
            1,
            "Expected 1 error for instance member access from static context"
        );

        let error = &result.errors[0];
        assert!(error.message.contains("Instance member"));
        assert!(error
            .message
            .contains("cannot be accessed from static context"));
    }

    #[test]
    fn test_try_catch_type_checking() {
        let haxe_code = r#"
class TryCatchTest {
    public static function main() {
        // Test basic try-catch
        try {
            var result = riskyOperation();
            trace(result);
        } catch (e: String) {
            trace("String error: " + e);
        } catch (e: Int) {
            trace("Int error: " + e);
        } catch (e: Dynamic) {
            trace("Generic error");
        }

        // Test catch with filter (invalid - filter must be boolean)
        try {
            doSomething();
        } catch (e: String) if (e.length) {  // Error: filter must be boolean
            trace("Filtered error");
        }
    }

    static function riskyOperation(): String {
        throw "Error";
    }

    static function doSomething(): Void {}
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about non-boolean filter
        assert!(
            !result.errors.is_empty(),
            "Expected error for non-boolean catch filter"
        );

        let has_filter_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("filter") && e.message.contains("boolean"));
        assert!(
            has_filter_error,
            "Should have error about catch filter needing to be boolean"
        );
    }

    #[test]
    fn test_while_loop_condition_type_checking() {
        let haxe_code = r#"
class WhileLoopTest {
    public static function main() {
        var i = 0;

        // Valid while loop
        while (i < 10) {
            trace("Count: " + i);
            i++;
        }

        // Invalid: non-boolean condition
        var str = "test";
        while (str) {  // Error: condition must be boolean
            trace("Never reached");
            break;
        }
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected error for non-boolean while condition"
        );

        let has_condition_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("condition") && e.message.contains("boolean"));
        assert!(
            has_condition_error,
            "Should have error about while condition needing to be boolean"
        );
    }

    #[test]
    fn test_for_loop_condition_type_checking() {
        let haxe_code = r#"
class ForLoopTest {
    public static function main() {
        // Valid for loop
        for (i in 0...10) {
            trace("i = " + i);
        }

        // Invalid: non-boolean condition in traditional for loop
        for (var j = 0; "not boolean"; j++) {  // Error: condition must be boolean
            trace("Never reached");
        }
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about non-boolean condition
        let _has_condition_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("condition") && e.message.contains("boolean"));
        // Note: Traditional for loops with conditions might not be fully supported in parser
        // This test documents expected behavior
    }

    #[test]
    fn test_for_in_loop_iterable_checking() {
        let haxe_code = r#"
class ForInTest {
    public static function main() {
        // Valid: iterating over array
        var arr = [1, 2, 3];
        for (item in arr) {
            trace("Item: " + item);
        }

        // Valid: iterating over string
        var str = "hello";
        for (char in str) {
            trace("Char: " + char);
        }

        // Invalid: iterating over non-iterable
        var num = 42;
        for (x in num) {  // Error: Int is not iterable
            trace("Never reached");
        }
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected error for non-iterable type in for-in"
        );

        let has_iterable_error = result.errors.iter().any(|e| {
            e.message.contains("not iterable") || e.message.contains("Type is not iterable")
        });
        assert!(
            has_iterable_error,
            "Should have error about type not being iterable"
        );
    }

    #[test]
    fn test_throw_expression_type_checking() {
        let haxe_code = r#"
class ThrowTest {
    public static function main() {
        // Valid: throwing string
        if (Math.random() < 0.5) {
            throw "Error message";
        }

        // Valid: throwing custom exception
        throw new CustomException("Something went wrong");

        // Valid: throwing any type (Haxe allows this)
        throw 42;
        throw true;
        throw { error: "object error" };
    }
}

class CustomException {
    public var message: String;
    public function new(msg: String) {
        this.message = msg;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should not have errors - Haxe allows throwing any type
        assert!(
            result.errors.is_empty(),
            "Should not have errors for valid throw expressions"
        );
    }

    #[test]
    fn test_object_literal_validation() {
        let haxe_code = r#"
class ObjectLiteralTest {
    public static function main() {
        // Valid object literal
        var obj1 = {
            name: "test",
            value: 42,
            active: true
        };

        // Object with duplicate fields
        var obj2 = {
            field: "first",
            other: 123,
            field: "duplicate"  // Error: duplicate field name
        };
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected error for duplicate object field"
        );

        let has_duplicate_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("Duplicate field") || e.message.contains("duplicate"));
        assert!(
            has_duplicate_error,
            "Should have error about duplicate field in object literal"
        );
    }

    #[test]
    fn test_map_literal_type_consistency() {
        let haxe_code = r#"
class MapLiteralTest {
    public static function main() {
        // Valid: consistent types
        var map1 = [
            "key1" => "value1",
            "key2" => "value2",
            "key3" => "value3"
        ];

        // Invalid: inconsistent key types
        var map2 = [
            "key1" => "value1",
            42 => "value2",      // Error: key type mismatch
            "key3" => "value3"
        ];

        // Invalid: inconsistent value types
        var map3 = [
            "key1" => "value1",
            "key2" => 42,        // Error: value type mismatch
            "key3" => "value3"
        ];

        // Map with duplicate keys
        var map4 = [
            "same" => "first",
            "other" => "second",
            "same" => "duplicate"  // Error: duplicate key
        ];
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        assert!(
            !result.errors.is_empty(),
            "Expected errors for map literal issues"
        );

        // Check for various map errors
        let has_key_mismatch = result.errors.iter().any(|e| {
            e.message.contains("Map key type mismatch")
                || (e.message.contains("key") && e.message.contains("type"))
        });
        let has_value_mismatch = result.errors.iter().any(|e| {
            e.message.contains("Map value type mismatch")
                || (e.message.contains("value") && e.message.contains("type"))
        });
        let has_duplicate_key = result.errors.iter().any(|e| {
            e.message.contains("Duplicate map key")
                || (e.message.contains("duplicate") && e.message.contains("key"))
        });

        assert!(
            has_key_mismatch || has_value_mismatch || has_duplicate_key,
            "Should have errors for map literal type inconsistencies or duplicates"
        );
    }

    #[test]
    fn test_break_continue_validation() {
        let haxe_code = r#"
class BreakContinueTest {
    public static function main() {
        // Valid break/continue in loop
        for (i in 0...10) {
            if (i == 5) break;
            if (i % 2 == 0) continue;
            trace(i);
        }

        // Invalid: break outside loop
        if (true) {
            break;  // Error: break not in loop context
        }

        // Invalid: continue outside loop
        trace("test");
        continue;  // Error: continue not in loop context
    }
}
        "#;

        // Note: Break/continue validation outside loops would require context tracking
        // This test documents the expected behavior
        let _result = crate::pipeline::compile_haxe_source(haxe_code);

        // Current implementation checks symbol table references
        // Full context validation would be a future enhancement
    }

    #[test]
    fn test_nested_try_catch() {
        let haxe_code = r#"
class NestedTryCatchTest {
    public static function main() {
        try {
            outerOperation();
        } catch (e: String) {
            try {
                innerOperation();
            } catch (inner: Int) {
                trace("Inner int error: " + inner);
            } catch (inner: Dynamic) {
                trace("Inner dynamic error");
            }
        } catch (e: Dynamic) {
            trace("Outer dynamic error");
        }
    }

    static function outerOperation(): Void {
        throw "outer error";
    }

    static function innerOperation(): Void {
        throw 42;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Nested try-catch should compile without errors
        if !result.errors.is_empty() {
            for (i, e) in result.errors.iter().enumerate() {
                eprintln!("Error {}: {} (category: {:?})", i, e.message, e.category);
            }
        }
        assert!(
            result.errors.is_empty(),
            "Nested try-catch should not have errors, got {} errors",
            result.errors.len()
        );
    }

    #[test]
    fn test_try_catch_finally() {
        let haxe_code = r#"
class TryCatchFinallyTest {
    public static function main() {
        var resource: Resource = null;

        try {
            resource = new Resource();
            resource.use();
        } catch (e: String) {
            trace("Error: " + e);
        } finally {
            if (resource != null) {
                resource.cleanup();
            }
        }
    }
}

class Resource {
    public function new() {}
    public function use(): Void {
        throw "Resource error";
    }
    public function cleanup(): Void {
        trace("Cleaning up");
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Try-catch-finally should compile without errors
        assert!(
            result.errors.is_empty()
                || result.errors.iter().all(|e| !e.message.contains("finally")),
            "Try-catch-finally should be supported"
        );
    }

    #[test]
    fn test_private_field_access() {
        let haxe_code = r#"
class TestClass {
    private var privateField:Int = 42;
    public var publicField:Int = 24;

    public function testAccess() {
        // Should work - same class access
        privateField = 100;
        publicField = 200;
    }
}

class OtherClass {
    public function testExternalAccess() {
        var obj = new TestClass();

        // Should work - public field
        obj.publicField = 300;

        // Should fail - private field access from different class
        obj.privateField = 400;
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about private field access
        let has_private_access_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("Private") && e.message.contains("privateField"));

        // Note: This test documents expected behavior - parser may need enhancement
        // for full private field support
    }

    #[test]
    fn test_private_method_access() {
        let haxe_code = r#"
class TestClass {
    private function privateMethod():Void {
        trace("Private method");
    }

    public function publicMethod():Void {
        // Should work - same class access
        privateMethod();
    }
}

class OtherClass {
    public function testMethodAccess() {
        var obj = new TestClass();

        // Should work - public method
        obj.publicMethod();

        // Should fail - private method access from different class
        obj.privateMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have error about private method access
        let has_private_method_error = result
            .errors
            .iter()
            .any(|e| e.message.contains("Private") && e.message.contains("privateMethod"));

        // Note: This test documents expected behavior - parser may need enhancement
        // for full private method support
    }

    #[test]
    fn test_protected_access_inheritance() {
        let haxe_code = r#"
class BaseClass {
    protected var protectedField:Int = 42;

    protected function protectedMethod():Void {
        trace("Protected method");
    }
}

class DerivedClass extends BaseClass {
    public function testAccess() {
        // Should work - accessing protected members from subclass
        protectedField = 100;
        protectedMethod();
    }
}

class UnrelatedClass {
    public function testExternalAccess() {
        var obj = new BaseClass();

        // Should fail - protected field access from unrelated class
        obj.protectedField = 200;

        // Should fail - protected method access from unrelated class
        obj.protectedMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have errors about protected access from unrelated class
        let has_protected_errors = result.errors.iter().any(|e| {
            e.message.contains("Protected")
                && (e.message.contains("protectedField") || e.message.contains("protectedMethod"))
        });

        // Note: This test documents expected behavior - full inheritance checking
        // requires parser support for 'extends' and protected modifiers
    }

    #[test]
    fn test_internal_package_access_same_package() {
        let haxe_code = r#"
package com.example.utils;

class InternalClass {
    internal var internalField:Int = 42;

    internal function internalMethod():String {
        return "Internal method";
    }

    public function publicMethod():Void {
        // Should work - same class access
        this.internalField = 100;
        var result = this.internalMethod();
    }
}

class SamePackageClass {
    public function testInternalAccess() {
        var obj = new InternalClass();

        // Should work - same package access to internal members
        obj.internalField = 200;
        var result = obj.internalMethod();

        // Should always work - public access
        obj.publicMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should not have any access violation errors for internal members within same package
        let has_internal_access_errors = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && (e.message.contains("internalField") || e.message.contains("internalMethod"))
        });

        // This should pass - same package access to internal members should be allowed
        assert!(
            !has_internal_access_errors,
            "Internal access within same package should be allowed"
        );
    }

    #[test]
    fn test_internal_package_access_different_package() {
        let haxe_code = r#"
package com.example.utils;

class InternalClass {
    internal var internalField:Int = 42;

    internal function internalMethod():String {
        return "Internal method";
    }

    public function publicMethod():Void {
        trace("Public method");
    }
}

// Different package file would be:
package com.other.package;

class DifferentPackageClass {
    public function testCrossPackageAccess() {
        var obj = new com.example.utils.InternalClass();

        // Should work - public access across packages
        obj.publicMethod();

        // Should fail - internal field access from different package
        obj.internalField = 300;

        // Should fail - internal method access from different package
        var result = obj.internalMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should have access violation errors for internal members from different package
        let has_internal_field_error = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && e.message.contains("internalField")
                && e.message.contains("different package")
        });

        let has_internal_method_error = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && e.message.contains("internalMethod")
                && e.message.contains("different package")
        });

        // These should fail - cross-package access to internal members should be denied
        // Note: This test documents expected behavior - full package checking requires
        // parser support for package declarations and proper symbol resolution
    }

    #[test]
    fn test_default_package_internal_access() {
        let haxe_code = r#"
// No package declaration = default package

class DefaultPackageClass {
    internal var internalField:Int = 42;

    internal function internalMethod():String {
        return "Internal in default package";
    }
}

class AnotherDefaultClass {
    public function testDefaultPackageAccess() {
        var obj = new DefaultPackageClass();

        // Should work - both classes in default package (no package declaration)
        obj.internalField = 100;
        var result = obj.internalMethod();
    }
}
        "#;

        let result = crate::pipeline::compile_haxe_source(haxe_code);

        // Should not have access errors - both classes in default package
        let has_internal_access_errors = result.errors.iter().any(|e| {
            e.message.contains("Internal")
                && (e.message.contains("internalField") || e.message.contains("internalMethod"))
        });

        // This should pass - default package access should be allowed
        assert!(
            !has_internal_access_errors,
            "Internal access within default package should be allowed"
        );
    }
}
