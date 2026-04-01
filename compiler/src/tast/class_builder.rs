use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
};

use crate::tast::{
    core::{TypeKind, TypeTable},
    type_checker::{ClassHierarchyInfo, ClassHierarchyRegistry},
    SymbolId, SymbolTable, TypeId,
};

/// Builder for constructing class hierarchies during semantic analysis
pub struct ClassHierarchyBuilder {
    /// Temporary storage during construction
    hierarchies: BTreeMap<SymbolId, ClassHierarchyInfo>,
}

impl ClassHierarchyBuilder {
    pub fn new() -> Self {
        Self {
            hierarchies: BTreeMap::new(), // Most projects have <32 classes
        }
    }

    /// Register a class with its direct superclass and interfaces
    pub fn register_class(
        &mut self,
        class_id: SymbolId,
        superclass: Option<TypeId>,
        interfaces: Vec<TypeId>,
    ) {
        let info = ClassHierarchyInfo {
            superclass,
            interfaces,
            all_supertypes: BTreeSet::new(), // Will be computed later
            depth: 0,                       // Will be computed later
            is_final: false,                // TODO: Extract from class metadata
            is_abstract: false,             // TODO: Extract from class metadata
            is_extern: false,               // TODO: Extract from class metadata
            is_interface: false,            // This is a class, not interface
            sealed_to: None,                // TODO: Extract from metadata
        };

        self.hierarchies.insert(class_id, info);
    }

    /// Compute transitive closure of supertypes for all classes
    pub fn compute_transitive_closure(&mut self, type_table: &RefCell<TypeTable>) {
        // First pass: collect all direct relationships
        let mut direct_supers: BTreeMap<SymbolId, Vec<TypeId>> =
            BTreeMap::new();

        for (&class_id, info) in &self.hierarchies {
            let mut supers = Vec::with_capacity(1 + info.interfaces.len()); // Superclass + interfaces

            if let Some(superclass) = info.superclass {
                supers.push(superclass);
            }

            supers.extend(info.interfaces.iter().cloned());

            direct_supers.insert(class_id, supers);
        }

        // Second pass: compute transitive closure using BFS
        for (&class_id, info) in self.hierarchies.iter_mut() {
            let mut visited = BTreeSet::new();
            let mut queue = VecDeque::new();
            let mut max_depth = 0;

            // Start with direct supertypes
            if let Some(supers) = direct_supers.get(&class_id) {
                for &super_type in supers {
                    queue.push_back((super_type, 1));
                    visited.insert(super_type);
                }
            }

            // BFS to find all supertypes
            while let Some((current_type, depth)) = queue.pop_front() {
                max_depth = max_depth.max(depth);

                // Get symbol for this type
                if let Some(ty) = type_table.borrow().get(current_type) {
                    if let Some(symbol_id) = match &ty.kind {
                        TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    } {
                        // Add supertypes of this type
                        if let Some(supers) = direct_supers.get(&symbol_id) {
                            for &super_type in supers {
                                if visited.insert(super_type) {
                                    queue.push_back((super_type, depth + 1));
                                }
                            }
                        }
                    }
                }
            }

            info.all_supertypes = visited;
            info.depth = max_depth;
        }
    }

    /// Finalize and transfer to symbol table
    pub fn finalize(self, symbol_table: &mut SymbolTable) {
        for (class_id, info) in self.hierarchies {
            symbol_table.register_class_hierarchy(class_id, info);
        }
    }
}

/// Builder pattern for constructing class hierarchies with validation
pub struct ClassHierarchyValidator<'a> {
    symbol_table: &'a SymbolTable,
    errors: Vec<String>,
}

impl<'a> ClassHierarchyValidator<'a> {
    pub fn new(symbol_table: &'a SymbolTable) -> Self {
        Self {
            symbol_table,
            errors: Vec::new(),
        }
    }

    /// Validate class hierarchy for cycles
    pub fn validate_no_cycles(&mut self) -> Result<(), Vec<String>> {
        let hierarchies = &self.symbol_table.class_hierarchies;

        for &class_id in hierarchies.keys() {
            if self.has_cycle_from(class_id) {
                self.errors.push(format!(
                    "Circular inheritance detected for class {:?}",
                    class_id
                ));
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    /// Check if there's a cycle starting from the given class
    fn has_cycle_from(&self, start: SymbolId) -> bool {
        let mut visited = BTreeSet::new();
        let mut current = start;

        loop {
            if !visited.insert(current) {
                // We've seen this before - cycle detected
                return true;
            }

            // Get superclass
            if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(current) {
                if let Some(superclass) = hierarchy.superclass {
                    // Get symbol from type
                    if let Some(super_sym) = self.symbol_table.get_symbol_from_type(superclass) {
                        current = super_sym;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        false
    }

    /// Validate that interfaces don't extend classes
    pub fn validate_interface_rules(&mut self) -> Result<(), Vec<String>> {
        let hierarchies = &self.symbol_table.class_hierarchies;

        for (&symbol_id, hierarchy) in hierarchies {
            // If it has a superclass, it's a class not an interface
            if hierarchy.superclass.is_none() && hierarchy.depth == 0 {
                // This is likely an interface
                // Check that all extended types are also interfaces
                for &extended in &hierarchy.interfaces {
                    if let Some(extended_sym) = self.symbol_table.get_symbol_from_type(extended) {
                        if !self.symbol_table.is_interface(extended_sym) {
                            self.errors.push(format!(
                                "Interface {:?} cannot extend class {:?}",
                                symbol_id, extended_sym
                            ));
                        }
                    }
                }
            }
        }

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tast::StringInterner;

    use super::*;

    #[test]
    fn test_class_hierarchy_registration() {
        let mut symbol_table = SymbolTable::new();
        let interner = StringInterner::new();

        // Create symbols
        let object_sym = symbol_table.create_class(interner.intern("Object"));
        let animal_sym = symbol_table.create_class(interner.intern("Animal"));

        // Register Object as root class
        let object_info = ClassHierarchyInfo {
            superclass: None,
            interfaces: vec![],
            all_supertypes: BTreeSet::new(),
            depth: 0,
            is_final: false,
            is_abstract: false,
            is_extern: false,
            is_interface: false,
            sealed_to: None,
        };
        symbol_table.register_class_hierarchy(object_sym, object_info);

        // Register Animal as subclass of Object
        let animal_info = ClassHierarchyInfo {
            superclass: Some(TypeId::from_raw(1)), // Assumes Object type ID
            interfaces: vec![],
            all_supertypes: BTreeSet::from_iter([TypeId::from_raw(1)]),
            depth: 1,
            is_final: false,
            is_abstract: false,
            is_extern: false,
            is_interface: false,
            sealed_to: None,
        };
        symbol_table.register_class_hierarchy(animal_sym, animal_info);

        // Verify registration
        assert!(symbol_table.get_class_hierarchy(object_sym).is_some());
        assert!(symbol_table.get_class_hierarchy(animal_sym).is_some());

        let animal_hierarchy = symbol_table.get_class_hierarchy(animal_sym).unwrap();
        assert_eq!(animal_hierarchy.depth, 1);
        assert!(animal_hierarchy.superclass.is_some());
    }

    #[test]
    fn test_interface_detection() {
        let mut symbol_table = SymbolTable::new();
        let interner = StringInterner::new();
        // Create interface symbol
        let comparable_sym = symbol_table.create_interface(interner.intern("Comparable"));

        // Register as interface (no superclass, depth 0)
        let interface_info = ClassHierarchyInfo {
            superclass: None,
            interfaces: vec![],
            all_supertypes: BTreeSet::new(),
            depth: 0,
            is_final: false,
            is_abstract: false,
            is_extern: false,
            is_interface: true, // This is an interface
            sealed_to: None,
        };
        symbol_table.register_class_hierarchy(comparable_sym, interface_info);

        assert!(symbol_table.is_interface(comparable_sym));
        assert!(!symbol_table.is_class(comparable_sym));
    }
}
