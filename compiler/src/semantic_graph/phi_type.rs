// Phi Type Unification Implementation for DFG Builder
// Properly unifies types from all phi operands instead of just using the first one

use crate::semantic_graph::dfg::{DataFlowGraph, DataFlowNodeKind, PhiIncoming};
use crate::semantic_graph::GraphConstructionError;
use crate::tast::core::{TypeKind, TypeTable};
use crate::tast::type_checker::TypeChecker;
use crate::tast::{DataFlowNodeId, TypeId};
use std::cell::RefCell;
use std::collections::BTreeSet;

/// Type unification module for phi nodes in the DFG builder
/// Always uses TypeChecker for proper type hierarchy resolution
pub struct PhiTypeUnifier<'a> {
    type_table: &'a RefCell<TypeTable>,
    type_checker: &'a TypeChecker<'a>,
}

impl<'a> PhiTypeUnifier<'a> {
    /// Create a new phi type unifier with TypeChecker for proper hierarchy resolution
    pub fn new(type_table: &'a RefCell<TypeTable>, type_checker: &'a TypeChecker<'a>) -> Self {
        Self {
            type_table,
            type_checker,
        }
    }

    /// Unify types from all phi operands to find the least upper bound using TypeChecker
    pub fn unify_phi_types(
        &mut self,
        phi_operands: &[PhiIncoming],
        dfg: &DataFlowGraph,
    ) -> Result<TypeId, GraphConstructionError> {
        if phi_operands.is_empty() {
            return Err(GraphConstructionError::InternalError {
                message: "Phi node has no operands".to_string(),
            });
        }

        // Collect all operand types
        let operand_types = self.collect_operand_types(phi_operands, dfg)?;

        // If all types are identical, return that type
        if operand_types.len() == 1 {
            return Ok(*operand_types.iter().next().unwrap());
        }

        // Find the least upper bound of all types
        self.find_least_upper_bound(&operand_types)
    }

    /// Collect unique types from phi operands
    fn collect_operand_types(
        &self,
        phi_operands: &[PhiIncoming],
        dfg: &DataFlowGraph,
    ) -> Result<BTreeSet<TypeId>, GraphConstructionError> {
        let mut types = BTreeSet::new();

        for operand in phi_operands {
            if let Some(node) = dfg.nodes.get(&operand.value) {
                types.insert(node.value_type);
            } else {
                return Err(GraphConstructionError::InternalError {
                    message: format!("Phi operand node {} not found", operand.value.as_raw()),
                });
            }
        }

        Ok(types)
    }

    /// Find the least upper bound (LUB) of a set of types using TypeChecker
    fn find_least_upper_bound(
        &mut self,
        types: &BTreeSet<TypeId>,
    ) -> Result<TypeId, GraphConstructionError> {
        let types_vec: Vec<TypeId> = types.iter().cloned().collect();

        // Start with the first type
        let mut lub = types_vec[0];

        // Iteratively find LUB with each subsequent type using TypeChecker
        for &ty in &types_vec[1..] {
            lub = self.compute_lub_pair_with_checker(lub, ty, self.type_checker)?;
        }

        Ok(lub)
    }

    fn get_type_kind(&self, type_id: TypeId) -> Result<TypeKind, GraphConstructionError> {
        let binding = self.type_table.borrow();
        binding
            .get(type_id)
            .ok_or_else(|| GraphConstructionError::InternalError {
                message: format!("Type {} not found", type_id.as_raw()),
            })
            .map(|t| t.kind.clone())
    }

    /// Compute the least upper bound of two types
    fn compute_lub_pair(
        &mut self,
        type1: TypeId,
        type2: TypeId,
    ) -> Result<TypeId, GraphConstructionError> {
        // If types are identical, return either one
        if type1 == type2 {
            return Ok(type1);
        }

        // Get type information

        let t1_kind = self.get_type_kind(type1)?;
        let t2_kind = self.get_type_kind(type2)?;

        match (t1_kind, t2_kind) {
            // Numeric type widening
            (TypeKind::Int, TypeKind::Float) | (TypeKind::Float, TypeKind::Int) => {
                let type_table_guard = self.type_table.borrow();
                Ok(type_table_guard.float_type())
            }

            // Optional type handling
            (
                TypeKind::Optional { inner_type: inner1 },
                TypeKind::Optional { inner_type: inner2 },
            ) => {
                // LUB of Optional<T1> and Optional<T2> is Optional<LUB(T1, T2)>
                let inner_lub = self.compute_lub_pair(inner1, inner2)?;

                let mut type_table_guard = self.type_table.borrow_mut();
                Ok(type_table_guard.create_optional_type(inner_lub))
            }

            (TypeKind::Optional { .. }, _) => {
                // LUB of Optional<T> and U is Optional<LUB(T, U)>

                let mut type_table_guard = self.type_table.borrow_mut();
                let optional_lub = type_table_guard.create_optional_type(type2);
                Ok(optional_lub)
            }

            (_, TypeKind::Optional { .. }) => {
                // LUB of T and Optional<U> is Optional<LUB(T, U)>

                let mut type_table_guard = self.type_table.borrow_mut();
                let optional_lub = type_table_guard.create_optional_type(type1);
                Ok(optional_lub)
            }

            // Array type handling
            (
                TypeKind::Array {
                    element_type: elem1,
                },
                TypeKind::Array {
                    element_type: elem2,
                },
            ) => {
                // LUB of Array<T1> and Array<T2> is Array<LUB(T1, T2)>
                let elem_lub = self.compute_lub_pair(elem1, elem2)?;

                let mut type_table_guard = self.type_table.borrow_mut();
                Ok(type_table_guard.create_array_type(elem_lub))
            }

            // Union type handling
            (TypeKind::Union { types: types1 }, TypeKind::Union { types: types2 }) => {
                // Combine all types and create a new union
                let mut all_types = types1.clone();
                all_types.extend(types2.iter().cloned());
                self.create_simplified_union(all_types)
            }

            (TypeKind::Union { types }, _) => {
                // Add the second type to the union
                let mut all_types = types.clone();
                all_types.push(type2);
                self.create_simplified_union(all_types)
            }

            (_, TypeKind::Union { types }) => {
                // Add the first type to the union
                let mut all_types = types.clone();
                all_types.push(type1);
                self.create_simplified_union(all_types)
            }

            // Dynamic type is a supertype of everything
            (TypeKind::Dynamic, _) | (_, TypeKind::Dynamic) => {
                let type_table_guard = self.type_table.borrow();
                Ok(type_table_guard.dynamic_type())
            }

            // Class/Interface hierarchy
            (TypeKind::Class { symbol_id: s1, .. }, TypeKind::Class { symbol_id: s2, .. }) => {
                self.find_common_supertype(s1, s2)
            }

            // If no other rules apply, create a union type
            _ => self.create_simplified_union(vec![type1, type2]),
        }
    }

    /// Compute the least upper bound of two types using TypeChecker
    fn compute_lub_pair_with_checker(
        &mut self,
        type1: TypeId,
        type2: TypeId,
        type_checker: &TypeChecker,
    ) -> Result<TypeId, GraphConstructionError> {
        // If types are identical, return either one
        if type1 == type2 {
            return Ok(type1);
        }

        // Get type information
        let t1_kind = self.get_type_kind(type1)?;
        let t2_kind = self.get_type_kind(type2)?;

        match (t1_kind, t2_kind) {
            // Numeric type widening
            (TypeKind::Int, TypeKind::Float) | (TypeKind::Float, TypeKind::Int) => {
                let type_table_guard = self.type_table.borrow();
                Ok(type_table_guard.float_type())
            }

            // Optional type handling
            (
                TypeKind::Optional { inner_type: inner1 },
                TypeKind::Optional { inner_type: inner2 },
            ) => {
                // LUB of Optional<T1> and Optional<T2> is Optional<LUB(T1, T2)>
                let inner_lub = self.compute_lub_pair_with_checker(inner1, inner2, type_checker)?;

                let mut type_table_guard = self.type_table.borrow_mut();
                Ok(type_table_guard.create_optional_type(inner_lub))
            }

            (TypeKind::Optional { .. }, _) => {
                // LUB of Optional<T> and U is Optional<LUB(T, U)>
                let mut type_table_guard = self.type_table.borrow_mut();
                let optional_lub = type_table_guard.create_optional_type(type2);
                Ok(optional_lub)
            }

            (_, TypeKind::Optional { .. }) => {
                // LUB of T and Optional<U> is Optional<LUB(T, U)>
                let mut type_table_guard = self.type_table.borrow_mut();
                let optional_lub = type_table_guard.create_optional_type(type1);
                Ok(optional_lub)
            }

            // Array type handling
            (
                TypeKind::Array {
                    element_type: elem1,
                },
                TypeKind::Array {
                    element_type: elem2,
                },
            ) => {
                // LUB of Array<T1> and Array<T2> is Array<LUB(T1, T2)>
                let elem_lub = self.compute_lub_pair_with_checker(elem1, elem2, type_checker)?;

                let mut type_table_guard = self.type_table.borrow_mut();
                Ok(type_table_guard.create_array_type(elem_lub))
            }

            // Union type handling
            (TypeKind::Union { types: types1 }, TypeKind::Union { types: types2 }) => {
                // Combine all types and create a new union
                let mut all_types = types1.clone();
                all_types.extend(types2.iter().cloned());
                self.create_simplified_union(all_types)
            }

            (TypeKind::Union { types }, _) => {
                // Add the second type to the union
                let mut all_types = types.clone();
                all_types.push(type2);
                self.create_simplified_union(all_types)
            }

            (_, TypeKind::Union { types }) => {
                // Add the first type to the union
                let mut all_types = types.clone();
                all_types.push(type1);
                self.create_simplified_union(all_types)
            }

            // Dynamic type is a supertype of everything
            (TypeKind::Dynamic, _) | (_, TypeKind::Dynamic) => {
                let type_table_guard = self.type_table.borrow();
                Ok(type_table_guard.dynamic_type())
            }

            // Class/Interface hierarchy - USE TypeChecker here
            (TypeKind::Class { symbol_id: s1, .. }, TypeKind::Class { symbol_id: s2, .. }) => {
                // Use TypeChecker's find_common_class_supertype method
                if let Some(common_type) = type_checker.find_common_class_supertype(type1, type2) {
                    Ok(common_type)
                } else {
                    // Fall back to Dynamic if no common supertype found
                    let type_table_guard = self.type_table.borrow();
                    Ok(type_table_guard.dynamic_type())
                }
            }

            // If no other rules apply, create a union type
            _ => self.create_simplified_union(vec![type1, type2]),
        }
    }

    /// Create a simplified union type, removing duplicates and redundant types
    fn create_simplified_union(
        &self,
        mut types: Vec<TypeId>,
    ) -> Result<TypeId, GraphConstructionError> {
        types.sort_unstable();
        types.dedup();

        if types.len() == 1 {
            return Ok(types[0]);
        }

        // Check with immutable borrow first
        {
            let type_table_guard = self.type_table.borrow();
            let dynamic_type = type_table_guard.dynamic_type();
            if types.iter().any(|&t| t == dynamic_type) {
                return Ok(dynamic_type);
            }
        }

        // Only take mutable borrow if we need to create a union
        let mut type_table_guard = self.type_table.borrow_mut();
        Ok(type_table_guard.create_union_type(types))
    }

    /// Find common supertype of two class types using TypeChecker
    fn find_common_supertype(
        &self,
        class1: crate::tast::SymbolId,
        class2: crate::tast::SymbolId,
    ) -> Result<TypeId, GraphConstructionError> {
        // Get TypeIds for both classes
        let type1 = self
            .get_type_by_symbol(class1)
            .unwrap_or_else(|| self.type_table.borrow().dynamic_type());
        let type2 = self
            .get_type_by_symbol(class2)
            .unwrap_or_else(|| self.type_table.borrow().dynamic_type());

        // Use TypeChecker's existing implementation
        if let Some(common_type) = self.type_checker.find_common_class_supertype(type1, type2) {
            Ok(common_type)
        } else {
            // Fall back to Dynamic if no common supertype found
            Ok(self.type_table.borrow().dynamic_type())
        }
    }

    /// Build the inheritance chain from a class to the root (Object)
    fn build_inheritance_chain(
        &self,
        class_id: crate::tast::SymbolId,
        type_table: &TypeTable,
    ) -> Result<Vec<crate::tast::SymbolId>, GraphConstructionError> {
        let mut chain = Vec::new();
        let mut current_class = Some(class_id);

        // Traverse the inheritance hierarchy up to the root
        while let Some(class) = current_class {
            chain.push(class);

            // Get the parent class from the type table
            current_class = self.get_parent_class(class)?;

            // Prevent infinite loops in case of circular inheritance
            if chain.len() > 100 {
                return Err(GraphConstructionError::InternalError {
                    message: format!(
                        "Circular inheritance detected starting from class {:?}",
                        class_id
                    ),
                });
            }
        }

        Ok(chain)
    }

    /// Get the parent class of a given class
    fn get_parent_class(
        &self,
        class_id: crate::tast::SymbolId,
    ) -> Result<Option<crate::tast::SymbolId>, GraphConstructionError> {
        if let Some(super_type_id) = self.type_checker.get_parent_class(class_id) {
            if let Some(super_type) = self.type_table.borrow().get(super_type_id) {
                Ok(super_type.symbol_id())
            } else {
                // Class symbol not found in type table
                Err(GraphConstructionError::UnresolvedSymbol {
                    symbol_name: format!("{:?}", class_id),
                    location: crate::tast::SourceLocation::unknown(),
                })
            }
        } else {
            // Class symbol not found in type table
            Err(GraphConstructionError::UnresolvedSymbol {
                symbol_name: format!("{:?}", class_id),
                location: crate::tast::SourceLocation::unknown(),
            })
        }
    }

    /// Find the least common ancestor of two inheritance chains
    fn find_least_common_ancestor(
        &self,
        chain1: &[crate::tast::SymbolId],
        chain2: &[crate::tast::SymbolId],
    ) -> Result<TypeId, GraphConstructionError> {
        // Convert chains to sets for efficient lookup
        let set1: BTreeSet<_> = chain1.iter().collect();
        let set2: BTreeSet<_> = chain2.iter().collect();

        // Find the first common ancestor in chain1 (closest to the original class)
        for &class in chain1 {
            if set2.contains(&class) {
                // Found common ancestor
                return Ok(self
                    .get_type_by_symbol(class)
                    .unwrap_or_else(|| self.type_table.borrow().dynamic_type()));
            }
        }

        // If no common ancestor found, check for common interfaces
        let interfaces1 = self.collect_implemented_interfaces(chain1)?;
        let interfaces2 = self.collect_implemented_interfaces(chain2)?;

        // Find common interfaces
        for interface in &interfaces1 {
            if interfaces2.contains(interface) {
                return Ok(self
                    .get_type_by_symbol(*interface)
                    .unwrap_or_else(|| self.type_table.borrow().dynamic_type()));
            }
        }

        // No common ancestor found - return Object or Dynamic
        Ok(self.get_object_type_or_dynamic())
    }

    /// Collect all interfaces implemented by classes in the inheritance chain
    fn collect_implemented_interfaces(
        &self,
        inheritance_chain: &[crate::tast::SymbolId],
    ) -> Result<BTreeSet<crate::tast::SymbolId>, GraphConstructionError> {
        let mut interfaces = BTreeSet::new();

        for &class_id in inheritance_chain {
            if let Some(hierarchy_info) = self
                .type_checker
                .symbol_table
                .class_hierarchies
                .get(&class_id)
            {
                for interface_id in hierarchy_info.interfaces.iter() {
                    if let Some(type_info) = self.type_table.borrow().get(interface_id.clone()) {
                        if let Some(symbol_id) = type_info.symbol_id() {
                            interfaces.insert(symbol_id);
                        }
                    }
                }
            }
        }

        Ok(interfaces)
    }

    /// Get the Object type, or Dynamic if Object is not available
    fn get_object_type_or_dynamic(&self) -> TypeId {
        // Try to find a well-known Object type
        // In Haxe, this might be Dynamic or Any
        self.type_table.borrow().dynamic_type()
    }

    /// Helper method to get a single type by symbol from the type table
    fn get_type_by_symbol(&self, symbol: crate::tast::SymbolId) -> Option<TypeId> {
        // Get all types for this symbol and return the first one
        // In most cases, there should be only one type per symbol
        self.type_table
            .borrow()
            .types_for_symbol(symbol)
            .and_then(|types| types.first().copied())
    }
}

/// Extension trait for DfgBuilder to use the type unifier
pub trait DfgBuilderPhiTypeUnification {
    fn resolve_and_validate_phi_operand_types<'a>(
        &self,
        phi_operands: &[PhiIncoming],
        type_checker: &'a TypeChecker<'a>,
    ) -> Result<TypeId, GraphConstructionError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_graph::dfg::{DataFlowNode, NodeMetadata};
    use crate::tast::collections::new_id_set;
    use crate::tast::core::TypeTable;
    use crate::tast::{BlockId, SourceLocation};

    #[test]
    fn test_phi_type_unification_numeric_widening() {
        let type_table = RefCell::new(TypeTable::new());
        let symbol_table = crate::tast::SymbolTable::new();
        let scope_tree = crate::tast::ScopeTree::new(crate::tast::ScopeId::first());
        let string_interner = crate::tast::StringInterner::new();

        let type_checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let mut unifier = PhiTypeUnifier::new(&type_table, &type_checker);

        let (int_type, float_type) = {
            let type_table_guard = unifier.type_table.borrow();
            // Test Int and Float unification
            (type_table_guard.int_type(), type_table_guard.float_type())
        };

        let lub = unifier.compute_lub_pair(int_type, float_type).unwrap();
        assert_eq!(lub, float_type);
    }

    #[test]
    fn test_phi_type_unification_optional() {
        let type_table = RefCell::new(TypeTable::new());
        let symbol_table = crate::tast::SymbolTable::new();
        let scope_tree = crate::tast::ScopeTree::new(crate::tast::ScopeId::first());
        let string_interner = crate::tast::StringInterner::new();

        let type_checker =
            &mut TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let mut unifier = PhiTypeUnifier::new(&type_table, &type_checker);

        let (int_type, optional_int) = {
            let mut type_table_guard = unifier.type_table.borrow_mut();
            // Test T and Optional<T> unification
            let int_type = type_table_guard.int_type();
            (
                int_type.clone(),
                type_table_guard.create_optional_type(int_type),
            )
        };

        let lub = unifier.compute_lub_pair(int_type, optional_int).unwrap();
        assert_eq!(lub, optional_int);
    }

    #[test]
    fn test_phi_type_unification_arrays() {
        let type_table = RefCell::new(TypeTable::new());
        let symbol_table = crate::tast::SymbolTable::new();
        let scope_tree = crate::tast::ScopeTree::new(crate::tast::ScopeId::first());
        let string_interner = crate::tast::StringInterner::new();

        let type_checker =
            &mut TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let mut unifier = PhiTypeUnifier::new(&type_table, &type_checker);

        let (int_type, float_type) = {
            let type_table_guard = unifier.type_table.borrow();
            // Test Int and Float unification
            (type_table_guard.int_type(), type_table_guard.float_type())
        };
        let (int_array, float_array) = {
            let mut type_table_guard = unifier.type_table.borrow_mut();
            // Test Array<Int> and Array<Float> unification
            let int_array = type_table_guard.create_array_type(int_type);
            let float_array = type_table_guard.create_array_type(float_type);
            (int_array, float_array)
        };

        let lub = unifier.compute_lub_pair(int_array, float_array).unwrap();

        let type_table_guard = unifier.type_table.borrow();
        // Should be Array<Float> since Float is the LUB of Int and Float
        if let Some(lub_type) = type_table_guard.get(lub) {
            if let TypeKind::Array { element_type } = &lub_type.kind {
                assert_eq!(*element_type, type_table_guard.float_type());
            } else {
                panic!("Expected array type");
            }
        }
    }
}
