//! Explicit casts, and whether two signatures accept each other.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Check if an explicit cast is valid between two types
    pub(crate) fn is_valid_explicit_cast(&self, from_type: TypeId, to_type: TypeId) -> bool {
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
    pub(crate) fn are_both_basic_types(&self, type1: TypeId, type2: TypeId) -> bool {
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
    pub(crate) fn get_string(&self, interned: super::InternedString) -> &str {
        self.string_interner.get(interned).unwrap_or("<unknown>")
    }


    /// Check if a method signature matches the provided argument types
    pub(crate) fn check_signature_compatibility(&self, param_types: &[TypeId], arg_types: &[TypeId]) -> bool {
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
    pub(crate) fn check_method_overloads(
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
}
