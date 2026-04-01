// Constraint Solver for Generic Type System
// Unification and constraint propagation

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use super::generic_instantiation::{GenericInstantiator, InstantiationError};
use super::type_checker::{
    ConstraintKind, ConstraintPriority, ConstraintSet, ConstraintValidation, ConstraintValidator,
    TypeConstraint,
};
use crate::tast::core::{Type, TypeKind, TypeTable, Variance};
use crate::tast::{SourceLocation, TypeId};

/// Unique identifier for unification variables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnificationVar(u32);

impl UnificationVar {
    fn new() -> Self {
        static COUNTER: AtomicU32 = AtomicU32::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    pub fn to_raw(self) -> u32 {
        self.0
    }

    pub fn from_raw(id: u32) -> Self {
        Self(id)
    }
}

/// Represents a unification constraint between two types
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnificationConstraint {
    pub left: TypeId,
    pub right: TypeId,
    pub kind: UnificationKind,
    pub location: SourceLocation,
    pub priority: ConstraintPriority,
}

/// Different kinds of unification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationKind {
    /// Exact equality: T = U
    Equality,

    /// Subtype relationship: T <: U
    Subtype,

    /// Supertype relationship: T :> U
    Supertype,

    /// Structural compatibility (for function types, etc.)
    Structural,

    /// Assignment compatibility (includes conversions)
    Assignment,
}

/// Result of a unification operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnificationResult {
    /// Unification succeeded with optional substitutions
    Success {
        substitutions: Vec<(TypeId, TypeId)>,
        additional_constraints: Vec<UnificationConstraint>,
    },

    /// Unification failed
    Failure {
        reason: String,
        conflicting_types: Vec<TypeId>,
    },

    /// Unification is pending (waiting for more information)
    Pending {
        waiting_for: Vec<TypeId>,
        partial_substitutions: Vec<(TypeId, TypeId)>,
    },
}

/// Unification table for managing type variable assignments
pub struct UnificationTable {
    /// Maps unification variables to their current assignments
    assignments: BTreeMap<UnificationVar, TypeId>,

    /// Maps type IDs to unification variables
    type_to_var: BTreeMap<TypeId, UnificationVar>,

    /// Tracks the "rank" of each variable for union-find optimization
    ranks: BTreeMap<UnificationVar, u32>,

    /// Parent pointers for union-find structure
    parents: BTreeMap<UnificationVar, UnificationVar>,

    /// Active constraints being solved
    active_constraints: VecDeque<UnificationConstraint>,

    /// Statistics for performance tracking
    stats: UnificationStats,
}

#[derive(Debug, Clone, Default)]
pub struct UnificationStats {
    pub unifications_attempted: usize,
    pub unifications_successful: usize,
    pub unifications_failed: usize,
    pub substitutions_made: usize,
    pub cycles_detected: usize,
    pub occurs_check_failures: usize,
    pub find_operations: usize,
    pub union_operations: usize,
}

impl UnificationTable {
    pub fn new() -> Self {
        Self {
            assignments: BTreeMap::new(),
            type_to_var: BTreeMap::new(),
            ranks: BTreeMap::new(),
            parents: BTreeMap::new(),
            active_constraints: VecDeque::new(),
            stats: UnificationStats::default(),
        }
    }

    /// Get or create a unification variable for a type
    pub fn var_for_type(&mut self, type_id: TypeId) -> UnificationVar {
        if let Some(&var) = self.type_to_var.get(&type_id) {
            var
        } else {
            let var = UnificationVar::new();
            self.type_to_var.insert(type_id, var);
            self.parents.insert(var, var); // Initialize as its own parent
            self.ranks.insert(var, 0);
            var
        }
    }

    /// Find the root representative of a unification variable (with path compression)
    pub fn find(&mut self, var: UnificationVar) -> UnificationVar {
        self.stats.find_operations += 1;

        if self.parents[&var] != var {
            // Path compression
            let root = self.find(self.parents[&var]);
            self.parents.insert(var, root);
            root
        } else {
            var
        }
    }

    /// Union two unification variables (with union by rank)
    pub fn union(&mut self, var1: UnificationVar, var2: UnificationVar) -> bool {
        self.stats.union_operations += 1;

        let root1 = self.find(var1);
        let root2 = self.find(var2);

        if root1 == root2 {
            return false; // Already unified
        }

        let rank1 = self.ranks[&root1];
        let rank2 = self.ranks[&root2];

        if rank1 < rank2 {
            self.parents.insert(root1, root2);
        } else if rank1 > rank2 {
            self.parents.insert(root2, root1);
        } else {
            self.parents.insert(root2, root1);
            self.ranks.insert(root1, rank1 + 1);
        }

        true
    }

    /// Assign a type to a unification variable
    pub fn assign(&mut self, var: UnificationVar, type_id: TypeId) -> Result<(), String> {
        let root = self.find(var);

        // Check for occurs check (prevent infinite types)
        if self.occurs_check(root, type_id) {
            self.stats.occurs_check_failures += 1;
            self.stats.cycles_detected += 1;
            return Err(format!(
                "Occurs check failed: variable {:?} occurs in type {:?}",
                root, type_id
            ));
        }

        self.assignments.insert(root, type_id);
        self.stats.substitutions_made += 1;
        Ok(())
    }

    /// Assign a type to a unification variable with enhanced occurs check
    pub fn assign_with_type_table(
        &mut self,
        var: UnificationVar,
        type_id: TypeId,
        type_table: &RefCell<TypeTable>,
    ) -> Result<(), String> {
        let root = self.find(var);

        // Enhanced occurs check using type table
        if self.occurs_check_with_type_table(root, type_id, type_table) {
            self.stats.occurs_check_failures += 1;
            self.stats.cycles_detected += 1;
            return Err(format!(
                "Cycle detected: variable {:?} occurs in type {:?}",
                root, type_id
            ));
        }

        self.assignments.insert(root, type_id);
        self.stats.substitutions_made += 1;
        Ok(())
    }

    /// Get the current assignment for a variable
    pub fn assignment(&mut self, var: UnificationVar) -> Option<TypeId> {
        let root = self.find(var);
        self.assignments.get(&root).copied()
    }

    /// Resolve a type by following unification variable assignments
    pub fn resolve_type(&mut self, type_id: TypeId) -> TypeId {
        if let Some(&var) = self.type_to_var.get(&type_id) {
            if let Some(assigned_type) = self.assignment(var) {
                // Recursively resolve in case of chains
                self.resolve_type(assigned_type)
            } else {
                type_id
            }
        } else {
            type_id
        }
    }

    /// Add a constraint to be solved
    pub fn add_constraint(&mut self, constraint: UnificationConstraint) {
        self.active_constraints.push_back(constraint);
    }

    /// Process all pending constraints
    pub fn solve_constraints(&mut self, type_table: &RefCell<TypeTable>) -> Vec<UnificationResult> {
        let mut results = Vec::new();

        while let Some(constraint) = self.active_constraints.pop_front() {
            self.stats.unifications_attempted += 1;

            let result = self.unify_constraint(&constraint, type_table);

            match &result {
                UnificationResult::Success {
                    additional_constraints,
                    ..
                } => {
                    self.stats.unifications_successful += 1;
                    // Add any additional constraints that were generated
                    for new_constraint in additional_constraints {
                        self.active_constraints.push_back(new_constraint.clone());
                    }
                }
                UnificationResult::Failure { .. } => {
                    self.stats.unifications_failed += 1;
                }
                UnificationResult::Pending { .. } => {
                    // Re-queue the constraint for later
                    self.active_constraints.push_back(constraint.clone());
                }
            }

            results.push(result);
        }

        results
    }

    /// Get statistics
    pub fn stats(&self) -> &UnificationStats {
        &self.stats
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.assignments.clear();
        self.type_to_var.clear();
        self.ranks.clear();
        self.parents.clear();
        self.active_constraints.clear();
        self.stats = UnificationStats::default();
    }

    // === Private helper methods ===

    fn occurs_check(&mut self, var: UnificationVar, type_id: TypeId) -> bool {
        self.occurs_check_recursive(var, type_id, &mut std::collections::BTreeSet::new())
    }

    fn occurs_check_recursive(
        &mut self,
        var: UnificationVar,
        type_id: TypeId,
        visited: &mut std::collections::BTreeSet<TypeId>,
    ) -> bool {
        // Prevent infinite recursion
        if visited.contains(&type_id) {
            return false;
        }
        visited.insert(type_id);

        // Direct variable check
        if let Some(&check_var) = self.type_to_var.get(&type_id) {
            if self.find(var) == self.find(check_var) {
                return true;
            }
        }

        // We need to get the TypeTable to traverse type structure
        // For now, just do basic check - this will be enhanced when we have type_table access
        false
    }

    fn occurs_check_with_type_table(
        &mut self,
        var: UnificationVar,
        type_id: TypeId,
        type_table: &RefCell<TypeTable>,
    ) -> bool {
        let mut visited = std::collections::BTreeSet::new();
        self.occurs_check_recursive_with_types(var, type_id, type_table, &mut visited)
    }

    fn occurs_check_recursive_with_types(
        &mut self,
        var: UnificationVar,
        type_id: TypeId,
        type_table: &RefCell<TypeTable>,
        visited: &mut std::collections::BTreeSet<TypeId>,
    ) -> bool {
        // Prevent infinite recursion
        if visited.contains(&type_id) {
            return false;
        }
        visited.insert(type_id);

        // Direct variable check
        if let Some(&check_var) = self.type_to_var.get(&type_id) {
            if self.find(var) == self.find(check_var) {
                return true;
            }
        }

        // Traverse type structure to detect cycles
        if let Some(type_obj) = type_table.borrow().get(type_id) {
            match &type_obj.kind {
                crate::tast::core::TypeKind::Array { element_type } => {
                    self.occurs_check_recursive_with_types(var, *element_type, type_table, visited)
                }
                crate::tast::core::TypeKind::Map {
                    key_type,
                    value_type,
                } => {
                    self.occurs_check_recursive_with_types(var, *key_type, type_table, visited)
                        || self.occurs_check_recursive_with_types(
                            var,
                            *value_type,
                            type_table,
                            visited,
                        )
                }
                crate::tast::core::TypeKind::Optional { inner_type } => {
                    self.occurs_check_recursive_with_types(var, *inner_type, type_table, visited)
                }
                crate::tast::core::TypeKind::Function {
                    params,
                    return_type,
                    ..
                } => {
                    params.iter().any(|&param_type| {
                        self.occurs_check_recursive_with_types(var, param_type, type_table, visited)
                    }) || self.occurs_check_recursive_with_types(
                        var,
                        *return_type,
                        type_table,
                        visited,
                    )
                }
                crate::tast::core::TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => {
                    self.occurs_check_recursive_with_types(var, *base_type, type_table, visited)
                        || type_args.iter().any(|&arg_type| {
                            self.occurs_check_recursive_with_types(
                                var, arg_type, type_table, visited,
                            )
                        })
                }
                crate::tast::core::TypeKind::Class { type_args, .. }
                | crate::tast::core::TypeKind::Interface { type_args, .. }
                | crate::tast::core::TypeKind::Enum { type_args, .. } => {
                    type_args.iter().any(|&arg_type| {
                        self.occurs_check_recursive_with_types(var, arg_type, type_table, visited)
                    })
                }
                crate::tast::core::TypeKind::Abstract {
                    underlying,
                    type_args,
                    ..
                } => {
                    underlying.map_or(false, |u| {
                        self.occurs_check_recursive_with_types(var, u, type_table, visited)
                    }) || type_args.iter().any(|&arg_type| {
                        self.occurs_check_recursive_with_types(var, arg_type, type_table, visited)
                    })
                }
                crate::tast::core::TypeKind::TypeAlias {
                    target_type,
                    type_args,
                    ..
                } => {
                    self.occurs_check_recursive_with_types(var, *target_type, type_table, visited)
                        || type_args.iter().any(|&arg_type| {
                            self.occurs_check_recursive_with_types(
                                var, arg_type, type_table, visited,
                            )
                        })
                }
                crate::tast::core::TypeKind::Union { types }
                | crate::tast::core::TypeKind::Intersection { types } => {
                    types.iter().any(|&union_type| {
                        self.occurs_check_recursive_with_types(var, union_type, type_table, visited)
                    })
                }
                crate::tast::core::TypeKind::Reference { target_type, .. } => {
                    self.occurs_check_recursive_with_types(var, *target_type, type_table, visited)
                }
                crate::tast::core::TypeKind::Anonymous { fields } => fields.iter().any(|field| {
                    self.occurs_check_recursive_with_types(var, field.type_id, type_table, visited)
                }),
                // Primitive types and type parameters don't contain other types
                _ => false,
            }
        } else {
            false
        }
    }

    fn unify_constraint(
        &mut self,
        constraint: &UnificationConstraint,
        type_table: &RefCell<TypeTable>,
    ) -> UnificationResult {
        match constraint.kind {
            UnificationKind::Equality => {
                self.unify_equality(constraint.left, constraint.right, type_table)
            }
            UnificationKind::Subtype => {
                self.unify_subtype(constraint.left, constraint.right, type_table)
            }
            UnificationKind::Supertype => {
                self.unify_subtype(constraint.right, constraint.left, type_table)
            }
            UnificationKind::Structural => {
                self.unify_structural(constraint.left, constraint.right, type_table)
            }
            UnificationKind::Assignment => {
                self.unify_assignment(constraint.left, constraint.right, type_table)
            }
        }
    }

    fn unify_equality(
        &mut self,
        left: TypeId,
        right: TypeId,
        _type_table: &RefCell<TypeTable>,
    ) -> UnificationResult {
        let left_resolved = self.resolve_type(left);
        let right_resolved = self.resolve_type(right);

        if left_resolved == right_resolved {
            return UnificationResult::Success {
                substitutions: vec![],
                additional_constraints: vec![],
            };
        }

        // Try to unify variables
        let left_var = self.var_for_type(left_resolved);
        let right_var = self.var_for_type(right_resolved);

        if self.union(left_var, right_var) {
            UnificationResult::Success {
                substitutions: vec![(left_resolved, right_resolved)],
                additional_constraints: vec![],
            }
        } else {
            UnificationResult::Failure {
                reason: format!("Cannot unify {:?} with {:?}", left_resolved, right_resolved),
                conflicting_types: vec![left_resolved, right_resolved],
            }
        }
    }

    fn unify_subtype(
        &mut self,
        sub: TypeId,
        sup: TypeId,
        type_table: &RefCell<TypeTable>,
    ) -> UnificationResult {
        let sub_resolved = self.resolve_type(sub);
        let sup_resolved = self.resolve_type(sup);

        // Check for equality first
        if sub_resolved == sup_resolved {
            return UnificationResult::Success {
                substitutions: vec![],
                additional_constraints: vec![],
            };
        }

        // Use the enhanced subtype checking
        if ConstraintPropagationEngine::check_subtype_relationship(
            sub_resolved,
            sup_resolved,
            type_table,
        ) {
            UnificationResult::Success {
                substitutions: vec![],
                additional_constraints: vec![],
            }
        } else {
            // If not a clear subtype, check if they can be unified through variables
            let sub_var = self.var_for_type(sub_resolved);
            let sup_var = self.var_for_type(sup_resolved);

            if self.union(sub_var, sup_var) {
                UnificationResult::Success {
                    substitutions: vec![(sub_resolved, sup_resolved)],
                    additional_constraints: vec![],
                }
            } else {
                UnificationResult::Failure {
                    reason: format!(
                        "Type {:?} is not a subtype of {:?}",
                        sub_resolved, sup_resolved
                    ),
                    conflicting_types: vec![sub_resolved, sup_resolved],
                }
            }
        }
    }

    fn unify_structural(
        &mut self,
        left: TypeId,
        right: TypeId,
        type_table: &RefCell<TypeTable>,
    ) -> UnificationResult {
        let left_resolved = self.resolve_type(left);
        let right_resolved = self.resolve_type(right);

        // Get type information
        let binding = type_table.borrow();
        let left_type = binding.get(left_resolved);
        let right_type = binding.get(right_resolved);

        match (left_type, right_type) {
            (Some(left_obj), Some(right_obj)) => {
                self.unify_type_structures(left_obj, right_obj, type_table)
            }
            _ => UnificationResult::Failure {
                reason: "Invalid types for structural unification".to_string(),
                conflicting_types: vec![left_resolved, right_resolved],
            },
        }
    }

    fn unify_assignment(
        &mut self,
        source: TypeId,
        target: TypeId,
        _type_table: &RefCell<TypeTable>,
    ) -> UnificationResult {
        // Assignment compatibility (includes implicit conversions)
        let source_resolved = self.resolve_type(source);
        let target_resolved = self.resolve_type(target);

        // For now, delegate to equality check
        // In full implementation, would check assignment compatibility
        self.unify_equality(source_resolved, target_resolved, _type_table)
    }

    fn unify_type_structures(
        &mut self,
        left: &Type,
        right: &Type,
        _type_table: &RefCell<TypeTable>,
    ) -> UnificationResult {
        use TypeKind;

        match (&left.kind, &right.kind) {
            (
                TypeKind::Function {
                    params: p1,
                    return_type: r1,
                    ..
                },
                TypeKind::Function {
                    params: p2,
                    return_type: r2,
                    ..
                },
            ) => {
                // Function types unify if parameters and return types unify
                if p1.len() != p2.len() {
                    return UnificationResult::Failure {
                        reason: "Function arity mismatch".to_string(),
                        conflicting_types: vec![],
                    };
                }

                let mut additional_constraints = Vec::new();

                // Contravariant parameter unification
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    additional_constraints.push(UnificationConstraint {
                        left: *param2, // Note: contravariant
                        right: *param1,
                        kind: UnificationKind::Subtype,
                        location: SourceLocation::default(),
                        priority: ConstraintPriority::TypeAnnotation,
                    });
                }

                // Covariant return type unification
                additional_constraints.push(UnificationConstraint {
                    left: *r1,
                    right: *r2,
                    kind: UnificationKind::Subtype,
                    location: SourceLocation::default(),
                    priority: ConstraintPriority::TypeAnnotation,
                });

                UnificationResult::Success {
                    substitutions: vec![],
                    additional_constraints,
                }
            }

            (TypeKind::Array { element_type: e1 }, TypeKind::Array { element_type: e2 }) => {
                // Array types unify if element types unify (covariant)
                let additional_constraints = vec![UnificationConstraint {
                    left: *e1,
                    right: *e2,
                    kind: UnificationKind::Equality,
                    location: SourceLocation::default(),
                    priority: ConstraintPriority::TypeAnnotation,
                }];

                UnificationResult::Success {
                    substitutions: vec![],
                    additional_constraints,
                }
            }

            (TypeKind::Optional { inner_type: i1 }, TypeKind::Optional { inner_type: i2 }) => {
                // Optional types unify if inner types unify
                let additional_constraints = vec![UnificationConstraint {
                    left: *i1,
                    right: *i2,
                    kind: UnificationKind::Equality,
                    location: SourceLocation::default(),
                    priority: ConstraintPriority::TypeAnnotation,
                }];

                UnificationResult::Success {
                    substitutions: vec![],
                    additional_constraints,
                }
            }

            // More structural unification cases would be added here...
            _ => UnificationResult::Failure {
                reason: "Incompatible type structures".to_string(),
                conflicting_types: vec![],
            },
        }
    }
}

impl Default for UnificationTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Advanced constraint solver that orchestrates the entire solving process
pub struct ConstraintSolver<'a> {
    /// Unification table for managing type variables
    unification_table: UnificationTable,

    /// Generic instantiation engine
    generic_instantiator: GenericInstantiator,

    /// Constraint propagation engine
    propagation_engine: ConstraintPropagationEngine,

    pub(crate) type_table: &'a RefCell<TypeTable>,

    /// Configuration settings
    config: SolverConfig,

    /// Overall solving statistics
    stats: SolverStats,
}

#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of solving iterations
    pub max_iterations: usize,

    /// Whether to perform aggressive constraint propagation
    pub aggressive_propagation: bool,

    /// Whether to cache intermediate results
    pub cache_intermediate_results: bool,

    /// Maximum time to spend solving (in milliseconds)
    pub max_solving_time_ms: u64,

    /// Whether to attempt error recovery
    pub error_recovery: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            aggressive_propagation: true,
            cache_intermediate_results: true,
            max_solving_time_ms: 5000, // 5 seconds
            error_recovery: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub total_constraints_processed: usize,
    pub successful_solutions: usize,
    pub failed_solutions: usize,
    pub iterations_performed: usize,
    pub propagation_rounds: usize,
    pub instantiations_performed: usize,
    pub total_solving_time_ms: u64,
    pub average_solving_time_ms: f64,
}

/// Constraint propagation engine for inferring additional constraints
pub struct ConstraintPropagationEngine {
    /// Rules for constraint propagation
    propagation_rules: Vec<PropagationRule>,

    /// Cache for propagation results (using content hash as key)
    propagation_cache: BTreeMap<u64, ConstraintSet>,

    /// Statistics
    stats: PropagationStats,
}

#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    pub propagations_performed: usize,
    pub rules_applied: usize,
    pub constraints_inferred: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// A rule for constraint propagation
pub struct PropagationRule {
    pub name: String,
    pub applies_to: fn(&ConstraintKind) -> bool,
    pub apply: fn(&TypeConstraint, &mut ConstraintSet, &RefCell<TypeTable>) -> Vec<TypeConstraint>,
}

impl ConstraintPropagationEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            propagation_rules: Vec::new(),
            propagation_cache: BTreeMap::new(),
            stats: PropagationStats::default(),
        };

        engine.register_default_rules();
        engine
    }

    /// Propagate constraints to infer additional ones
    pub fn propagate(
        &mut self,
        constraints: &mut ConstraintSet,
        type_table: &RefCell<TypeTable>,
    ) -> usize {
        self.stats.propagations_performed += 1;

        // Compute content hash for cache lookup
        let content_hash = constraints.compute_content_hash();

        // Check cache first
        if let Some(cached_result) = self.propagation_cache.get(&content_hash) {
            self.stats.cache_hits += 1;
            constraints.merge(cached_result.clone());
            return cached_result.stats().total_constraints;
        }

        self.stats.cache_misses += 1;

        let initial_count = constraints.stats().total_constraints;
        let mut new_constraints = ConstraintSet::new();

        // Apply propagation rules
        for constraint in self.collect_constraints(constraints) {
            for rule in &self.propagation_rules {
                if (rule.applies_to)(&constraint.kind) {
                    let inferred = (rule.apply)(&constraint, constraints, type_table);
                    self.stats.rules_applied += 1;

                    for new_constraint in inferred {
                        new_constraints.add_constraint(new_constraint);
                        self.stats.constraints_inferred += 1;
                    }
                }
            }
        }

        // Merge new constraints
        constraints.merge(new_constraints.clone());

        // Cache the result using the original content hash
        self.propagation_cache.insert(content_hash, new_constraints);

        constraints.stats().total_constraints - initial_count
    }

    pub fn stats(&self) -> &PropagationStats {
        &self.stats
    }

    fn register_default_rules(&mut self) {
        // Rule 1: Iterable-IndexAccess consistency
        self.propagation_rules.push(PropagationRule {
            name: "iterable-index-consistency".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Iterable { .. }),
            apply: |constraint, constraint_set, type_table| {
                let mut inferred = Vec::new();

                if let ConstraintKind::Iterable { element_type } = &constraint.kind {
                    // Find IndexAccess constraints on the same type
                    // Note: Need to implement constraints() method on ConstraintSet
                    // This is a simplified implementation for now
                    // In the full implementation, would iterate through constraint_set.constraints()
                }

                inferred
            },
        });

        // Rule 2: Method parameter type propagation
        self.propagation_rules.push(PropagationRule {
            name: "method-parameter-propagation".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::HasMethod { .. }),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                if let ConstraintKind::HasMethod {
                    method_name: _,
                    signature,
                    is_static: _,
                } = &constraint.kind
                {
                    if let Some(sig_type) = type_table.borrow().get(*signature) {
                        if let TypeKind::Function {
                            params,
                            return_type,
                            ..
                        } = &sig_type.kind
                        {
                            // Add constraints for parameter types
                            for (i, &param_type) in params.iter().enumerate() {
                                // Ensure parameters are properly typed
                                // Check if param_type is a type parameter
                                if type_table.borrow().is_type_parameter(param_type) {
                                    inferred.push(TypeConstraint {
                                        type_var: param_type,
                                        kind: ConstraintKind::Sized,
                                        location: constraint.location,
                                        priority: ConstraintPriority::Usage,
                                        is_soft: true,
                                    });
                                }
                            }

                            // Ensure return type is valid
                            // Check if return_type is a type parameter
                            if type_table.borrow().is_type_parameter(*return_type) {
                                inferred.push(TypeConstraint {
                                    type_var: *return_type,
                                    kind: ConstraintKind::Sized,
                                    location: constraint.location,
                                    priority: ConstraintPriority::Usage,
                                    is_soft: true,
                                });
                            }
                        }
                    }
                }

                inferred
            },
        });

        // Rule 3: Comparable implies HasMethod(compare)
        self.propagation_rules.push(PropagationRule {
            name: "comparable-method-inference".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Comparable),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                // Comparable types should have a compare method
                let int_type = type_table.borrow().int_type();
                let compare_signature = type_table.borrow_mut().create_function_type(
                    vec![constraint.type_var], // compare(other: Self) -> Int
                    int_type,
                );

                inferred.push(TypeConstraint {
                    type_var: constraint.type_var,
                    kind: ConstraintKind::HasMethod {
                        method_name: type_table.borrow_mut().intern_string("compare"),
                        signature: compare_signature,
                        is_static: false,
                    },
                    location: constraint.location,
                    priority: ConstraintPriority::Usage,
                    is_soft: true,
                });

                inferred
            },
        });

        // Rule 4: Arithmetic implies numeric operations
        self.propagation_rules.push(PropagationRule {
            name: "arithmetic-operations".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Arithmetic),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                // Arithmetic types must be Comparable
                inferred.push(TypeConstraint {
                    type_var: constraint.type_var,
                    kind: ConstraintKind::Comparable,
                    location: constraint.location,
                    priority: ConstraintPriority::Usage,
                    is_soft: false,
                });

                // They also need standard arithmetic methods
                let self_type = constraint.type_var;
                let add_signature = type_table
                    .borrow_mut()
                    .create_function_type(vec![self_type], self_type);

                for (method_name, signature) in [
                    ("add", add_signature.clone()),
                    ("sub", add_signature.clone()),
                    ("mul", add_signature.clone()),
                    ("div", add_signature.clone()),
                ] {
                    inferred.push(TypeConstraint {
                        type_var: constraint.type_var,
                        kind: ConstraintKind::HasMethod {
                            method_name: type_table.borrow_mut().intern_string(method_name),
                            signature,
                            is_static: false,
                        },
                        location: constraint.location,
                        priority: ConstraintPriority::Usage,
                        is_soft: true,
                    });
                }

                inferred
            },
        });

        // Rule 5: Copy implies value semantics
        self.propagation_rules.push(PropagationRule {
            name: "copy-value-semantics".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Copy),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                // Copy types must be Sized
                inferred.push(TypeConstraint {
                    type_var: constraint.type_var,
                    kind: ConstraintKind::Sized,
                    location: constraint.location,
                    priority: ConstraintPriority::Usage,
                    is_soft: false,
                });

                inferred
            },
        });

        // Rule 6: StringConvertible implies toString
        self.propagation_rules.push(PropagationRule {
            name: "string-convertible-tostring".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::StringConvertible),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                let string_type = type_table.borrow().string_type();
                let to_string_signature = type_table
                    .borrow_mut()
                    .create_function_type(vec![], string_type);

                inferred.push(TypeConstraint {
                    type_var: constraint.type_var,
                    kind: ConstraintKind::HasMethod {
                        method_name: type_table.borrow_mut().intern_string("toString"),
                        signature: to_string_signature,
                        is_static: false,
                    },
                    location: constraint.location,
                    priority: ConstraintPriority::Usage,
                    is_soft: false,
                });

                inferred
            },
        });

        // Rule 7: Field constraints propagation
        self.propagation_rules.push(PropagationRule {
            name: "field-type-propagation".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::HasField { .. }),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                if let ConstraintKind::HasField {
                    field_name: _,
                    field_type,
                    is_public: _,
                } = &constraint.kind
                {
                    // Field types should be Sized
                    // Check if field_type is a type parameter
                    if type_table.borrow().is_type_parameter(*field_type) {
                        inferred.push(TypeConstraint {
                            type_var: *field_type,
                            kind: ConstraintKind::Sized,
                            location: constraint.location,
                            priority: ConstraintPriority::Usage,
                            is_soft: true,
                        });
                    }
                }

                inferred
            },
        });

        // Rule 8: Callable implies function type structure
        self.propagation_rules.push(PropagationRule {
            name: "callable-function-type".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Callable { .. }),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                if let ConstraintKind::Callable {
                    params,
                    return_type,
                } = &constraint.kind
                {
                    // All parameter types must be Sized
                    for &param in params {
                        // Check if param is a type parameter
                        if type_table.borrow().is_type_parameter(param) {
                            inferred.push(TypeConstraint {
                                type_var: param,
                                kind: ConstraintKind::Sized,
                                location: constraint.location,
                                priority: ConstraintPriority::Usage,
                                is_soft: true,
                            });
                        }
                    }

                    // Return type must be valid
                    // Check if return_type is a type parameter
                    if type_table.borrow().is_type_parameter(*return_type) {
                        inferred.push(TypeConstraint {
                            type_var: *return_type,
                            kind: ConstraintKind::Sized,
                            location: constraint.location,
                            priority: ConstraintPriority::Usage,
                            is_soft: true,
                        });
                    }
                }

                inferred
            },
        });

        // Rule 9: Constructible type requirements
        self.propagation_rules.push(PropagationRule {
            name: "constructible-requirements".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Constructible { .. }),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                if let ConstraintKind::Constructible { constructor_args } = &constraint.kind {
                    // Constructor argument types must be Sized
                    for &arg_type in constructor_args {
                        // Check if arg_type is a type parameter
                        if type_table.borrow().is_type_parameter(arg_type) {
                            inferred.push(TypeConstraint {
                                type_var: arg_type,
                                kind: ConstraintKind::Sized,
                                location: constraint.location,
                                priority: ConstraintPriority::Usage,
                                is_soft: true,
                            });
                        }
                    }

                    // Constructible types must be concrete (not abstract/interface)
                    // This would require more context in a real implementation
                }

                inferred
            },
        });

        // Rule 10: Interface implementation requirements
        self.propagation_rules.push(PropagationRule {
            name: "interface-implementation".to_string(),
            applies_to: |kind| matches!(kind, ConstraintKind::Implements { .. }),
            apply: |constraint, _constraint_set, type_table| {
                let mut inferred = Vec::new();

                if let ConstraintKind::Implements { interface_type } = &constraint.kind {
                    // When implementing an interface, the type must have all
                    // methods and fields required by that interface
                    // This would require interface metadata lookup

                    // For now, just ensure the implementing type is Sized
                    inferred.push(TypeConstraint {
                        type_var: constraint.type_var,
                        kind: ConstraintKind::Sized,
                        location: constraint.location,
                        priority: ConstraintPriority::Usage,
                        is_soft: true,
                    });
                }

                inferred
            },
        });
    }

    /// Properly collect all constraints from a ConstraintSet
    fn collect_constraints(&self, constraint_set: &ConstraintSet) -> Vec<TypeConstraint> {
        // Use the constraints() method to get all constraints
        constraint_set.constraints().cloned().collect()
    }

    /// Enhanced constraint satisfaction checking
    pub fn check_constraint_satisfaction(
        type_id: TypeId,
        constraint: &ConstraintKind,
        type_table: &RefCell<TypeTable>,
    ) -> bool {
        if let Some(type_obj) = type_table.borrow().get(type_id) {
            match constraint {
                ConstraintKind::Sized => {
                    match &type_obj.kind {
                        TypeKind::Dynamic => false,              // Dynamic has unknown size
                        TypeKind::Function { .. } => false,      // Functions don't have static size
                        TypeKind::TypeParameter { .. } => false, // Unknown until instantiated
                        _ => true,                               // Most types are sized
                    }
                }

                ConstraintKind::Comparable => {
                    match &type_obj.kind {
                        TypeKind::Int
                        | TypeKind::Float
                        | TypeKind::String
                        | TypeKind::Bool
                        | TypeKind::Char => true,
                        TypeKind::Dynamic => false, // Can't guarantee comparison
                        TypeKind::Optional { inner_type } => {
                            // Optional is comparable if inner type is
                            Self::check_constraint_satisfaction(*inner_type, constraint, type_table)
                        }
                        TypeKind::Class { symbol_id, .. } => {
                            // Check if class implements Comparable interface
                            // This would need symbol table access in real implementation
                            false // Conservative default
                        }
                        _ => false,
                    }
                }

                ConstraintKind::Arithmetic => {
                    match &type_obj.kind {
                        TypeKind::Int | TypeKind::Float => true,
                        TypeKind::Dynamic => false, // Can't guarantee arithmetic ops
                        _ => false,
                    }
                }

                ConstraintKind::StringConvertible => {
                    match &type_obj.kind {
                        // All primitive types can convert to string
                        TypeKind::Int
                        | TypeKind::Float
                        | TypeKind::String
                        | TypeKind::Bool
                        | TypeKind::Char => true,
                        TypeKind::Dynamic => false, // Can't guarantee string conversion
                        TypeKind::Optional { inner_type } => {
                            // Optional is string convertible if inner is
                            Self::check_constraint_satisfaction(*inner_type, constraint, type_table)
                        }
                        TypeKind::Class { .. } | TypeKind::Interface { .. } => {
                            // Would need to check for toString method
                            true // Most classes have toString in Haxe
                        }
                        _ => false,
                    }
                }

                ConstraintKind::Copy => {
                    match &type_obj.kind {
                        // Primitive types are copyable
                        TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::Char => true,
                        // Strings are immutable and copyable in Haxe
                        TypeKind::String => true,
                        // References and complex types are not Copy
                        TypeKind::Class { .. }
                        | TypeKind::Interface { .. }
                        | TypeKind::Function { .. }
                        | TypeKind::Dynamic => false,
                        // Arrays are reference types
                        TypeKind::Array { .. } => false,
                        // Optional is Copy if inner type is Copy
                        TypeKind::Optional { inner_type } => {
                            Self::check_constraint_satisfaction(*inner_type, constraint, type_table)
                        }
                        _ => false,
                    }
                }

                ConstraintKind::Referenceable => {
                    match &type_obj.kind {
                        // Most types can be referenced except void
                        TypeKind::Void => false,
                        // Everything else can be referenced
                        _ => true,
                    }
                }

                ConstraintKind::Iterable { element_type } => {
                    match &type_obj.kind {
                        TypeKind::Array {
                            element_type: array_elem,
                        } => {
                            // Check element type compatibility if specified
                            if let Some(expected_elem) = element_type {
                                // Would need type compatibility check here
                                true // Simplified
                            } else {
                                true
                            }
                        }
                        TypeKind::Map {
                            key_type: _,
                            value_type,
                        } => {
                            // Map is iterable over key-value pairs
                            if let Some(expected_elem) = element_type {
                                // Would check if expected_elem matches Pair<K,V>
                                true // Simplified
                            } else {
                                true
                            }
                        }
                        TypeKind::String => {
                            // String is iterable over characters
                            if let Some(expected_elem) = element_type {
                                // Check if expected_elem is Char
                                type_table
                                    .borrow()
                                    .get(*expected_elem)
                                    .map(|t| matches!(t.kind, TypeKind::Char))
                                    .unwrap_or(false)
                            } else {
                                true
                            }
                        }
                        TypeKind::Dynamic => false, // Can't guarantee iteration
                        _ => false,
                    }
                }

                ConstraintKind::HasMethod {
                    method_name,
                    signature,
                    is_static,
                } => {
                    match &type_obj.kind {
                        TypeKind::Class { symbol_id, .. }
                        | TypeKind::Interface { symbol_id, .. } => {
                            // Would need symbol table to check methods
                            // For now, return false as we can't verify
                            false
                        }
                        TypeKind::Dynamic => false, // Can't verify methods on Dynamic
                        _ => false,
                    }
                }

                ConstraintKind::HasField {
                    field_name,
                    field_type,
                    is_public,
                } => {
                    match &type_obj.kind {
                        TypeKind::Class { symbol_id, .. }
                        | TypeKind::Interface { symbol_id, .. } => {
                            // Would need symbol table to check fields
                            false
                        }
                        TypeKind::Anonymous { fields } => {
                            // Check anonymous object fields
                            fields.iter().any(|f| {
                                f.name == *field_name &&
                                (!is_public || f.is_public) &&
                                // Would need type compatibility check for field_type
                                true
                            })
                        }
                        TypeKind::Dynamic => false, // Can't verify fields on Dynamic
                        _ => false,
                    }
                }

                ConstraintKind::IndexAccess {
                    key_type,
                    value_type,
                } => {
                    match &type_obj.kind {
                        TypeKind::Array { element_type } => {
                            // Arrays use Int keys
                            if !type_table
                                .borrow()
                                .get(*key_type)
                                .map(|t| matches!(t.kind, TypeKind::Int))
                                .unwrap_or(false)
                            {
                                return false;
                            }
                            // Check value type matches element type
                            // if let Some(v_type) = value_type {
                            //     // Would need type compatibility check
                            //     true
                            // } else {
                            //     true
                            // }
                            true
                        }
                        TypeKind::Map {
                            key_type: map_key,
                            value_type: map_value,
                        } => {
                            // Check key and value type compatibility
                            // let key_ok = key_type.map(|k| {
                            //     // Would need type compatibility check
                            //     true
                            // }).unwrap_or(true);

                            // let value_ok = value_type.map(|v| {
                            //     // Would need type compatibility check
                            //     true
                            // }).unwrap_or(true);

                            // key_ok && value_ok

                            true
                        }
                        TypeKind::Dynamic => false, // Can't verify index access on Dynamic
                        _ => false,
                    }
                }

                ConstraintKind::Excludes { excluded_types } => {
                    // Check that the type is not in the excluded list
                    !excluded_types.contains(&type_id)
                }

                ConstraintKind::OneOf { candidates } => {
                    // Type must be one of the candidates
                    candidates.contains(&type_id) ||
                    // Or convertible to one of them
                    candidates.iter().any(|&candidate| {
                        // Would need type compatibility check here
                        type_id == candidate // Simplified
                    })
                }

                ConstraintKind::Constructible { constructor_args } => {
                    match &type_obj.kind {
                        TypeKind::Class { .. } => {
                            // Would need to check if class has matching constructor
                            // For now, assume classes are constructible
                            true
                        }
                        TypeKind::Interface { .. } => {
                            // Interfaces cannot be constructed directly
                            false
                        }
                        TypeKind::Dynamic => false, // Can't construct Dynamic
                        _ => false,
                    }
                }

                ConstraintKind::Callable {
                    params,
                    return_type,
                } => {
                    match &type_obj.kind {
                        TypeKind::Function {
                            params: fn_params,
                            return_type: fn_return,
                            ..
                        } => {
                            // Check parameter count matches
                            if params.len() != fn_params.len() {
                                return false;
                            }
                            // Would need to check parameter and return type compatibility
                            true // Simplified
                        }
                        TypeKind::Dynamic => false, // Can't guarantee callability
                        _ => false,
                    }
                }

                ConstraintKind::Implements { interface_type } => {
                    match &type_obj.kind {
                        TypeKind::Class { .. } => {
                            // Would need to check class hierarchy for interface implementation
                            false // Conservative without symbol table access
                        }
                        TypeKind::Interface { .. } => {
                            // Interface can implement another interface if it extends it
                            type_id == *interface_type // Simplified - only exact match
                        }
                        _ => false,
                    }
                }

                ConstraintKind::Custom {
                    predicate_name,
                    args,
                } => {
                    // Custom predicates would need special handling based on predicate_name
                    // This is an extension point for domain-specific constraints
                    false // Conservative default
                }

                // Previously handled constraint kinds
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type is convertible to another type
    fn check_convertible(source: TypeId, target: TypeId, type_table: &RefCell<TypeTable>) -> bool {
        if source == target {
            return true;
        }

        if let (Some(source_type), Some(target_type)) = (
            type_table.borrow().get(source),
            type_table.borrow().get(target),
        ) {
            match (&source_type.kind, &target_type.kind) {
                // Numeric conversions
                (TypeKind::Int, TypeKind::Float) => true,

                // Dynamic can convert to anything (runtime check)
                (TypeKind::Dynamic, _) => true,

                // Optional conversions
                (inner, TypeKind::Optional { inner_type }) => {
                    Self::check_convertible(source, *inner_type, type_table)
                }

                // Array element conversions
                (TypeKind::Array { element_type: e1 }, TypeKind::Array { element_type: e2 }) => {
                    Self::check_convertible(*e1, *e2, type_table)
                }

                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if two types can be unified
    fn check_unifiable(left: TypeId, right: TypeId, type_table: &RefCell<TypeTable>) -> bool {
        if left == right {
            return true;
        }

        if let (Some(left_type), Some(right_type)) = (
            type_table.borrow().get(left),
            type_table.borrow().get(right),
        ) {
            match (&left_type.kind, &right_type.kind) {
                // Same type kinds can potentially unify
                (TypeKind::Array { element_type: e1 }, TypeKind::Array { element_type: e2 }) => {
                    Self::check_unifiable(*e1, *e2, type_table)
                }

                (TypeKind::Optional { inner_type: i1 }, TypeKind::Optional { inner_type: i2 }) => {
                    Self::check_unifiable(*i1, *i2, type_table)
                }

                // Function types unify if signatures match
                (
                    TypeKind::Function {
                        params: p1,
                        return_type: r1,
                        ..
                    },
                    TypeKind::Function {
                        params: p2,
                        return_type: r2,
                        ..
                    },
                ) => {
                    p1.len() == p2.len()
                        && p1
                            .iter()
                            .zip(p2.iter())
                            .all(|(t1, t2)| Self::check_unifiable(*t1, *t2, type_table))
                        && Self::check_unifiable(*r1, *r2, type_table)
                }

                // Type variables can unify with anything
                (TypeKind::TypeParameter { .. }, _) | (_, TypeKind::TypeParameter { .. }) => true,

                // Unknown types can unify with anything (for inference)
                (TypeKind::Unknown, _) | (_, TypeKind::Unknown) => true,

                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type is a subtype of another type (simplified implementation)
    fn check_subtype_relationship(
        subtype: TypeId,
        supertype: TypeId,
        type_table: &RefCell<TypeTable>,
    ) -> bool {
        if subtype == supertype {
            return true;
        }

        if let (Some(sub_type), Some(super_type)) = (
            type_table.borrow().get(subtype),
            type_table.borrow().get(supertype),
        ) {
            match (&sub_type.kind, &super_type.kind) {
                // Numeric subtyping: Int <: Float
                (TypeKind::Int, TypeKind::Float) => true,

                // Dynamic can be treated as subtype of anything (runtime check)
                (TypeKind::Dynamic, _) => true,

                // Array covariance: Array<String> <: Array<Object> (simplified)
                (TypeKind::Array { element_type: e1 }, TypeKind::Array { element_type: e2 }) => {
                    Self::check_subtype_relationship(*e1, *e2, type_table)
                }

                // Optional subtyping: T <: T?
                (_, TypeKind::Optional { inner_type }) => {
                    Self::check_subtype_relationship(subtype, *inner_type, type_table)
                }

                // Function subtyping (contravariant in params, covariant in return)
                (
                    TypeKind::Function {
                        params: p1,
                        return_type: r1,
                        ..
                    },
                    TypeKind::Function {
                        params: p2,
                        return_type: r2,
                        ..
                    },
                ) => {
                    p1.len() == p2.len() &&
                    // Parameters are contravariant
                    p1.iter().zip(p2.iter()).all(|(param1, param2)|
                        Self::check_subtype_relationship(*param2, *param1, type_table)
                    ) &&
                    // Return type is covariant
                    Self::check_subtype_relationship(*r1, *r2, type_table)
                }

                // Type parameters and unknowns are flexible
                (TypeKind::TypeParameter { .. }, _)
                | (_, TypeKind::TypeParameter { .. })
                | (TypeKind::Unknown, _)
                | (_, TypeKind::Unknown) => true,

                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a type implements an interface (simplified implementation)
    fn check_interface_implementation(
        type_id: TypeId,
        interface_id: TypeId,
        type_table: &RefCell<TypeTable>,
    ) -> bool {
        if type_id == interface_id {
            return true;
        }

        if let (Some(type_obj), Some(interface_obj)) = (
            type_table.borrow().get(type_id),
            type_table.borrow().get(interface_id),
        ) {
            match (&type_obj.kind, &interface_obj.kind) {
                // Class implementing interface (simplified - would need symbol table integration)
                (TypeKind::Class { .. }, TypeKind::Interface { .. }) => {
                    // For now, assume no implementation relationship without symbol table
                    false
                }

                // Dynamic types could implement any interface at runtime
                (TypeKind::Dynamic, TypeKind::Interface { .. }) => true,

                // Arrays and Maps implement common interfaces
                (TypeKind::Array { .. }, TypeKind::Interface { .. }) => {
                    // Arrays could implement Iterable<T>, etc.
                    true // Simplified - assume arrays implement common interfaces
                }

                (TypeKind::Map { .. }, TypeKind::Interface { .. }) => {
                    // Maps could implement various interfaces
                    true // Simplified
                }

                _ => false,
            }
        } else {
            false
        }
    }
}

impl Default for ConstraintPropagationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> ConstraintSolver<'a> {
    pub fn new(config: SolverConfig, type_table: &'a RefCell<TypeTable>) -> Self {
        Self {
            unification_table: UnificationTable::new(),
            generic_instantiator: GenericInstantiator::with_defaults(),
            propagation_engine: ConstraintPropagationEngine::new(),
            config,
            stats: SolverStats::default(),
            type_table,
        }
    }

    pub fn with_defaults(type_table: &'a RefCell<TypeTable>) -> Self {
        Self::new(SolverConfig::default(), type_table)
    }

    /// Solve a set of constraints
    pub fn solve(
        &mut self,
        mut constraints: ConstraintSet,
        type_table: &RefCell<TypeTable>,
    ) -> SolverResult {
        let start_time = std::time::Instant::now();

        // Convert constraints to unification constraints
        let unification_constraints = self.convert_to_unification_constraints(&constraints);

        for constraint in unification_constraints {
            self.unification_table.add_constraint(constraint);
        }

        let mut iteration = 0;
        let mut success = true;

        while iteration < self.config.max_iterations {
            // Check timeout
            if start_time.elapsed().as_millis() as u64 > self.config.max_solving_time_ms {
                success = false;
                break;
            }

            // Propagate constraints
            if self.config.aggressive_propagation {
                let inferred = self
                    .propagation_engine
                    .propagate(&mut constraints, type_table);
                self.stats.propagation_rounds += 1;

                if inferred == 0 {
                    break; // Fixed point reached
                }
            }

            // Solve unification constraints
            let results = self.unification_table.solve_constraints(type_table);

            // Check if we made progress
            let made_progress = results
                .iter()
                .any(|r| matches!(r, UnificationResult::Success { .. }));

            if !made_progress {
                break;
            }

            iteration += 1;
        }

        self.stats.iterations_performed = iteration;
        self.stats.total_solving_time_ms = start_time.elapsed().as_millis() as u64;
        self.stats.total_constraints_processed += constraints.stats().total_constraints;

        if success {
            self.stats.successful_solutions += 1;
        } else {
            self.stats.failed_solutions += 1;
        }

        // Update average solving time
        let total_solutions = self.stats.successful_solutions + self.stats.failed_solutions;
        if total_solutions > 0 {
            self.stats.average_solving_time_ms =
                self.stats.total_solving_time_ms as f64 / total_solutions as f64;
        }

        SolverResult {
            success,
            final_constraints: constraints,
            substitutions: self.extract_substitutions(),
            iterations: iteration,
            solving_time_ms: start_time.elapsed().as_millis() as u64,
        }
    }

    /// Get solver statistics
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Clear all solver state
    pub fn clear(&mut self) {
        self.unification_table.clear();
        self.generic_instantiator.clear_cache();
        self.propagation_engine.propagation_cache.clear();
    }

    /// Convert high-level constraints to unification constraints
    fn convert_to_unification_constraints(
        &self,
        constraint_set: &ConstraintSet,
    ) -> Vec<UnificationConstraint> {
        let mut unification_constraints = Vec::new();

        // Iterate through all constraints in the set
        // Collect all constraints from the constraint set
        for constraint in constraint_set.constraints() {
            match &constraint.kind {
                ConstraintKind::Equality { target_type } => {
                    unification_constraints.push(UnificationConstraint {
                        left: constraint.type_var,
                        right: target_type.clone(),
                        kind: UnificationKind::Equality,
                        location: constraint.location,
                        priority: constraint.priority,
                    });
                }

                ConstraintKind::Subtype { supertype } => {
                    unification_constraints.push(UnificationConstraint {
                        left: constraint.type_var,
                        right: supertype.clone(),
                        kind: UnificationKind::Subtype,
                        location: constraint.location,
                        priority: constraint.priority,
                    });
                }

                ConstraintKind::Supertype { subtype } => {
                    unification_constraints.push(UnificationConstraint {
                        left: constraint.type_var,
                        right: subtype.clone(),
                        kind: UnificationKind::Supertype,
                        location: constraint.location,
                        priority: constraint.priority,
                    });
                }

                ConstraintKind::Implements { interface_type } => {
                    // Interface implementation is a form of subtyping
                    unification_constraints.push(UnificationConstraint {
                        left: constraint.type_var,
                        right: interface_type.clone(),
                        kind: UnificationKind::Subtype,
                        location: constraint.location,
                        priority: constraint.priority,
                    });
                }

                ConstraintKind::Callable {
                    params,
                    return_type,
                } => {
                    // For callable constraints, create a function type and unify
                    let func_type = self
                        .type_table
                        .borrow_mut()
                        .create_function_type(params.clone(), return_type.clone());
                    unification_constraints.push(UnificationConstraint {
                        left: constraint.type_var,
                        right: func_type,
                        kind: UnificationKind::Structural,
                        location: constraint.location,
                        priority: constraint.priority,
                    });
                }

                ConstraintKind::OneOf { candidates } => {
                    // For OneOf constraints, we need special handling
                    // This might require multiple unification attempts
                    if candidates.len() == 1 {
                        unification_constraints.push(UnificationConstraint {
                            left: constraint.type_var,
                            right: candidates[0],
                            kind: UnificationKind::Equality,
                            location: constraint.location,
                            priority: constraint.priority,
                        });
                    }
                    // Multiple candidates need more complex handling
                }

                // Other constraint kinds are handled by the propagation engine
                ConstraintKind::HasMethod { .. }
                | ConstraintKind::HasField { .. }
                | ConstraintKind::IndexAccess { .. }
                | ConstraintKind::Iterable { .. }
                | ConstraintKind::Comparable
                | ConstraintKind::Arithmetic
                | ConstraintKind::StringConvertible
                | ConstraintKind::Constructible { .. }
                | ConstraintKind::Sized
                | ConstraintKind::Copy
                | ConstraintKind::Referenceable
                | ConstraintKind::Excludes { .. }
                | ConstraintKind::Custom { .. } => {
                    // These constraints are handled by propagation rules
                    // or need special checking beyond simple unification
                }
            }
        }

        unification_constraints
    }

    /// Extract type substitutions from the unification table
    fn extract_substitutions(&mut self) -> Vec<(TypeId, TypeId)> {
        let mut substitutions = Vec::new();
        let mut processed = BTreeSet::new();

        // First, resolve all variables to their roots
        let mut var_to_type = BTreeMap::new();
        let assignments = self.unification_table.assignments.clone();
        for (&var, &type_id) in &assignments {
            let root = self.unification_table.find(var);
            if let Some(&assigned_type) = self.unification_table.assignments.get(&root) {
                var_to_type.insert(var, assigned_type);
            }
        }

        // Then extract substitutions for original type IDs
        for (&original_type, &var) in &self.unification_table.type_to_var {
            if processed.insert(original_type) {
                if let Some(&substituted_type) = var_to_type.get(&var) {
                    if original_type != substituted_type {
                        substitutions.push((original_type, substituted_type));
                    }
                }
            }
        }

        substitutions
    }
}

/// Result of constraint solving
#[derive(Debug, Clone)]
pub struct SolverResult {
    pub success: bool,
    pub final_constraints: ConstraintSet,
    pub substitutions: Vec<(TypeId, TypeId)>,
    pub iterations: usize,
    pub solving_time_ms: u64,
}

// Display implementations
impl fmt::Display for UnificationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnificationKind::Equality => write!(f, "="),
            UnificationKind::Subtype => write!(f, "<:"),
            UnificationKind::Supertype => write!(f, ":>"),
            UnificationKind::Structural => write!(f, "~"),
            UnificationKind::Assignment => write!(f, ":="),
        }
    }
}

impl fmt::Display for UnificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnificationResult::Success { substitutions, .. } => {
                write!(f, "Success ({} substitutions)", substitutions.len())
            }
            UnificationResult::Failure { reason, .. } => {
                write!(f, "Failure: {}", reason)
            }
            UnificationResult::Pending { waiting_for, .. } => {
                write!(f, "Pending (waiting for {} types)", waiting_for.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unification_table() {
        let mut table = UnificationTable::new();

        let type1 = TypeId::from_raw(1);
        let type2 = TypeId::from_raw(2);

        let var1 = table.var_for_type(type1);
        let var2 = table.var_for_type(type2);

        assert_ne!(var1, var2);
        assert!(table.union(var1, var2));
        assert_eq!(table.find(var1), table.find(var2));
    }

    #[test]
    fn test_unification_constraint() {
        let constraint = UnificationConstraint {
            left: TypeId::from_raw(1),
            right: TypeId::from_raw(2),
            kind: UnificationKind::Equality,
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
        };

        assert_eq!(constraint.kind, UnificationKind::Equality);
    }

    #[test]
    fn test_constraint_solver() {
        let type_table = &RefCell::new(TypeTable::new());
        let mut solver = ConstraintSolver::with_defaults(type_table);
        let constraint_set = ConstraintSet::new();

        let result = solver.solve(constraint_set, &type_table);
        assert!(result.success); // Empty constraint set should always succeed
    }

    #[test]
    fn test_propagation_engine() {
        let type_table = &RefCell::new(TypeTable::new());
        let mut engine = ConstraintPropagationEngine::new();
        let mut constraints = ConstraintSet::new();

        let inferred = engine.propagate(&mut constraints, &type_table);
        assert_eq!(inferred, 0); // No constraints to propagate
        assert_eq!(engine.stats().propagations_performed, 1);
    }

    #[test]
    fn test_solver_stats() {
        let type_table = &RefCell::new(TypeTable::new());
        let solver = ConstraintSolver::with_defaults(type_table);
        let stats = solver.stats();
        assert_eq!(stats.total_constraints_processed, 0);
        assert_eq!(stats.successful_solutions, 0);
    }
}
