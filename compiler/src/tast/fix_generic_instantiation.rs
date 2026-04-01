use std::collections::BTreeMap;
use crate::tast::{TypeId, SymbolId};

/// This module provides a fix for the generic instantiation issue where
/// interfaces are created with 0 type parameters during pre-registration.
///
/// The issue:
/// 1. During pre-registration, `create_interface_type(symbol, Vec::new())` creates interfaces with 0 type parameters
/// 2. When lowering `interface Sortable<T>`, we properly process the type parameter
/// 3. But when validating `Container<String>` where Container requires `T:Sortable<T>`,
///    the instantiation validation checks the pre-registered type which has 0 parameters
///
/// The fix:
/// We need to store the actual type parameter counts separately and use them during validation.

/// Registry for tracking the actual type parameter counts of generic types
pub struct TypeParameterRegistry {
    /// Maps symbol IDs to their actual type parameter counts
    type_param_counts: BTreeMap<SymbolId, usize>,
}

impl TypeParameterRegistry {
    pub fn new() -> Self {
        Self {
            type_param_counts: BTreeMap::new(),
        }
    }

    /// Register a type with its parameter count
    pub fn register_type(&mut self, symbol_id: SymbolId, param_count: usize) {
        self.type_param_counts.insert(symbol_id, param_count);
    }

    /// Get the parameter count for a type
    pub fn get_param_count(&self, symbol_id: SymbolId) -> Option<usize> {
        self.type_param_counts.get(&symbol_id).copied()
    }
}

/// Alternative approach: Extract type parameter count from the TypedDeclaration
/// after the file has been fully lowered.
pub fn extract_type_param_counts(typed_file: &crate::tast::node::TypedFile) -> BTreeMap<SymbolId, usize> {
    let mut counts = BTreeMap::new();

    // Extract from classes
    for class in &typed_file.classes {
        counts.insert(class.symbol_id, class.type_parameters.len());
    }

    // Extract from interfaces
    for interface in &typed_file.interfaces {
        counts.insert(interface.symbol_id, interface.type_parameters.len());
    }

    // Extract from enums
    for enum_decl in &typed_file.enums {
        counts.insert(enum_decl.symbol_id, enum_decl.type_parameters.len());
    }

    // Extract from type aliases
    for type_alias in &typed_file.type_aliases {
        counts.insert(type_alias.symbol_id, type_alias.type_parameters.len());
    }

    counts
}