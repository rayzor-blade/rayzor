//! Whether a member may be reached from where it is named: visibility,
//! package rules, and static-versus-instance access.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Emit a type error as a diagnostic
    /// Check if a field access is valid
    pub(crate) fn check_field_access(
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
    pub(crate) fn check_field_accessibility(
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
    pub(crate) fn validate_field_visibility(
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
    pub(crate) fn validate_package_level_access(
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
    pub(crate) fn extract_package_from_file(
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
    pub(crate) fn get_current_class_name(&self) -> InternedString {
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
    pub(crate) fn is_subclass_of(&self, potential_subclass: SymbolId, potential_superclass: SymbolId) -> bool {
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
    pub(crate) fn get_class_symbol_from_type(&self, type_id: TypeId) -> Option<SymbolId> {
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
    pub(crate) fn find_field_by_symbol(&self, field_symbol: SymbolId) -> Option<&TypedField> {
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
    pub(crate) fn find_class_by_symbol(&self, symbol_id: SymbolId) -> Option<&TypedClass> {
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
    pub(crate) fn check_method_static_access(
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
    pub(crate) fn validate_method_visibility(
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
}
