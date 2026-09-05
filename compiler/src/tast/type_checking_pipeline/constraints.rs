//! Type-parameter constraints and the predicates that answer them.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Validate that a type satisfies a constraint type (e.g., T:Comparable<T>)
    pub(crate) fn validate_type_constraint(&self, type_id: TypeId, constraint_type_id: TypeId) -> bool {
        if type_id == constraint_type_id {
            return true;
        }

        // Check if it's an interface implementation
        if self.is_interface_type(constraint_type_id) {
            return self.type_implements_interface(type_id, constraint_type_id);
        }

        // Abstract constraints (e.g., EnumValue, FlatEnum, NotVoid) are type-erasure
        // markers — they serve as semantic hints, not runtime constraints.
        // Accept any type for abstract constraints.
        if self.is_abstract_type(constraint_type_id) {
            return true;
        }

        false
    }


    pub(crate) fn is_abstract_type(&self, type_id: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();
        type_table.get(type_id).map_or(false, |ti| {
            matches!(ti.kind, super::TypeKind::Abstract { .. })
        })
    }


    /// Check if a type implements an interface
    pub(crate) fn type_implements_interface(&self, type_id: TypeId, interface_type: TypeId) -> bool {
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
    pub(crate) fn is_interface_type(&self, type_id: TypeId) -> bool {
        let type_table = self.type_checker.type_table.borrow();
        if let Some(type_info) = type_table.get(type_id) {
            matches!(type_info.kind, super::TypeKind::Interface { .. })
        } else {
            false
        }
    }


    /// Check if a type is comparable
    pub(crate) fn is_comparable_type(&self, type_id: TypeId) -> bool {
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
    pub(crate) fn type_has_method(
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
    pub(crate) fn type_has_field(
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
    pub(crate) fn constraint_type_to_string(&self, constraint_type_id: TypeId) -> String {
        // Format the constraint type name using existing get_type_name method
        self.get_type_name(constraint_type_id)
            .unwrap_or_else(|| format!("Type#{}", constraint_type_id.as_raw()))
    }


    /// Get a human-readable name for a type
    pub(crate) fn get_type_name(&self, type_id: TypeId) -> Option<String> {
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
}
