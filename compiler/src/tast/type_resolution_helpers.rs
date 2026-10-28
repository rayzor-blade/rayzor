//! Type Resolution Helpers for AST Lowering
//!
//! This module provides utilities to resolve concrete types during AST lowering,
//! replacing dynamic_type placeholders with actual types.

use super::{
    InternedString, ScopeId, ScopeTree, SourceLocation, StringInterner, SymbolId, SymbolTable,
    TypeId, TypeKind, TypeTable,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Type resolution context for AST lowering
pub struct TypeResolutionHelpers<'a> {
    type_table: &'a Rc<RefCell<TypeTable>>,
    symbol_table: &'a SymbolTable,
    string_interner: &'a StringInterner,
    scope_tree: &'a ScopeTree,
}

impl<'a> TypeResolutionHelpers<'a> {
    pub fn new(
        type_table: &'a Rc<RefCell<TypeTable>>,
        symbol_table: &'a SymbolTable,
        string_interner: &'a StringInterner,
        scope_tree: &'a ScopeTree,
    ) -> Self {
        TypeResolutionHelpers {
            type_table,
            symbol_table,
            string_interner,
            scope_tree,
        }
    }

    /// Resolve type alias to its target type
    pub fn resolve_type_alias(&self, symbol_id: SymbolId) -> TypeId {
        if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
            let type_table = self.type_table.borrow();

            // Follow alias chain
            let mut current_type = symbol.type_id;
            let mut visited = vec![current_type];

            while let Some(type_info) = type_table.get(current_type) {
                match &type_info.kind {
                    TypeKind::TypeAlias { target_type, .. } => {
                        let target_type = *target_type;
                        // Check for cycles
                        if visited.contains(&target_type) {
                            return type_table.dynamic_type();
                        }
                        visited.push(target_type);
                        current_type = target_type;
                    }
                    _ => return current_type,
                }
            }

            current_type
        } else {
            self.type_table.borrow().dynamic_type()
        }
    }

    /// Resolve abstract type to its underlying type
    pub fn resolve_abstract_type(&self, symbol_id: SymbolId) -> TypeId {
        if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
            let type_table = self.type_table.borrow();

            if let Some(type_info) = type_table.get(symbol.type_id) {
                match &type_info.kind {
                    TypeKind::Abstract { underlying, .. } => {
                        underlying.unwrap_or_else(|| type_table.dynamic_type())
                    }
                    _ => symbol.type_id,
                }
            } else {
                type_table.dynamic_type()
            }
        } else {
            self.type_table.borrow().dynamic_type()
        }
    }

    /// Resolve field type from a class
    pub fn resolve_field_type(
        &self,
        _class_symbol: SymbolId,
        _field_name: InternedString,
    ) -> TypeId {
        // TODO: Implement proper field resolution when class metadata is available
        // For now, return dynamic type
        self.type_table.borrow().dynamic_type()
    }

    /// Resolve method type from a class
    pub fn resolve_method_type(
        &self,
        _class_symbol: SymbolId,
        _method_name: InternedString,
    ) -> TypeId {
        // TODO: Implement proper method resolution when class metadata is available
        // For now, return dynamic type
        self.type_table.borrow().dynamic_type()
    }

    /// Resolve 'this' type from current class context
    pub fn resolve_this_type(&self, current_scope: ScopeId) -> TypeId {
        // Walk up scopes to find enclosing class
        let mut scope_id = current_scope;

        loop {
            if let Some(scope) = self.scope_tree.get_scope(scope_id) {
                // Check if this is a class scope
                use super::scopes::ScopeKind;
                if matches!(scope.kind, ScopeKind::Class) {
                    // Find the class symbol
                    for &symbol_id in &scope.symbols {
                        if let Some(symbol) = self.symbol_table.get_symbol(symbol_id) {
                            if matches!(symbol.kind, crate::tast::SymbolKind::Class) {
                                return symbol.type_id;
                            }
                        }
                    }
                }

                // Move to parent scope
                if let Some(parent) = scope.parent_id {
                    scope_id = parent;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        // No enclosing class found
        self.type_table.borrow().dynamic_type()
    }

    /// Resolve 'super' type from current class context
    pub fn resolve_super_type(&self, current_scope: ScopeId) -> TypeId {
        // First find the current class
        let class_type = self.resolve_this_type(current_scope);

        let type_table = self.type_table.borrow();
        if let Some(type_info) = type_table.get(class_type) {
            match &type_info.kind {
                // Note: The actual TypeKind::Class might not have base_class field
                // This would need to be looked up from class metadata
                _ => type_table.dynamic_type(),
            }
        } else {
            type_table.dynamic_type()
        }
    }

    /// Create a proper null type
    pub fn create_null_type(&self) -> TypeId {
        // There is no TypeKind::Null in the current type system
        // Use dynamic type as a placeholder for null values
        self.type_table.borrow().dynamic_type()
    }

    /// Create a proper regex type (EReg)
    pub fn create_regex_type(&self) -> TypeId {
        // TODO: Implement proper EReg type creation
        // For now, return dynamic as placeholder
        self.type_table.borrow().dynamic_type()
    }

    /// Create a map type with key and value types
    pub fn create_map_type(&self, key_type: TypeId, value_type: TypeId) -> TypeId {
        self.type_table
            .borrow_mut()
            .create_map_type(key_type, value_type)
    }

    /// Create an anonymous object type from fields
    pub fn create_anonymous_object_type(&self, fields: Vec<(InternedString, TypeId)>) -> TypeId {
        let anonymous_fields: Vec<_> = fields
            .into_iter()
            .map(|(name, type_id)| super::core::AnonymousField {
                name,
                type_id,
                is_public: true,
                optional: false,

                access: Default::default(),
            })
            .collect();

        self.type_table
            .borrow_mut()
            .create_type(TypeKind::Anonymous {
                fields: anonymous_fields,
            })
    }

    /// Infer type from object literal fields
    pub fn infer_object_literal_type(&self, fields: &[(InternedString, TypeId)]) -> TypeId {
        if fields.is_empty() {
            // Empty object
            self.create_anonymous_object_type(vec![])
        } else {
            // Create anonymous type with inferred fields
            self.create_anonymous_object_type(fields.to_vec())
        }
    }

    /// Create a union type from branch types (for conditionals/switch)
    pub fn create_union_type(&self, types: Vec<TypeId>) -> TypeId {
        if types.is_empty() {
            return self.type_table.borrow().void_type();
        }

        if types.len() == 1 {
            return types[0];
        }

        // Deduplicate and filter out void types
        let type_table = self.type_table.borrow();
        let void_type = type_table.void_type();

        let mut unique_types = Vec::new();
        for t in types {
            if t != void_type && !unique_types.contains(&t) {
                unique_types.push(t);
            }
        }

        if unique_types.is_empty() {
            return void_type;
        }

        if unique_types.len() == 1 {
            return unique_types[0];
        }

        drop(type_table);
        self.type_table.borrow_mut().create_union_type(unique_types)
    }
}
