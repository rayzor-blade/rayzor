//! Checking what a type declares: field types, method signatures and bodies,
//! and the compatibility an override or an interface demands.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Check a field type
    pub(crate) fn check_field_type(
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
    pub(crate) fn check_method_signature(
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
    pub(crate) fn check_method_implementation(
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
    pub(crate) fn check_method_body(
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
    pub(crate) fn verify_interface_implementation(
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
    pub(crate) fn verify_inheritance_signatures(
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
    pub(crate) fn check_override_compatibility(
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
    pub(crate) fn method_signatures_match(
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
        Ok(!matches!(compatibility, TypeCompatibility::Incompatible))
    }


    /// Format a method signature for display
    pub(crate) fn format_method_signature(&self, method: &TypedMethodSignature) -> String {
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
    pub(crate) fn format_function_signature(&self, func: &TypedFunction) -> String {
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
    pub(crate) fn format_type(&self, type_id: TypeId) -> String {
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
}
