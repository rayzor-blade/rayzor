//! Type Checking Engine for TAST (Typed AST)
//!
//! This module provides comprehensive type checking capabilities including:
//! - Type compatibility and subtyping relationships
//! - Type inference with constraint solving
//! - Generic type checking and instantiation
//! - Error reporting with detailed diagnostics
//! - Integration with Symbol and Scope systems

use super::{
    collections::{new_id_map, new_id_set, IdMap, IdSet},
    core::{Type, TypeKind, TypeTable, TypeUsageStats},
    InternedString, Scope, ScopeId, ScopeKind, ScopeTree, SourceLocation, StringInterner, Symbol,
    SymbolId, SymbolKind, SymbolTable, TypeId,
};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

/// Result of type checking operations
pub type TypeCheckResult<T> = Result<T, TypeCheckError>;

/// Type checking error with detailed context
#[derive(Debug, Clone)]
pub struct TypeCheckError {
    /// Kind of type error
    pub kind: TypeErrorKind,
    /// Source location where error occurred
    pub location: SourceLocation,
    /// Additional context for error reporting
    pub context: String,
    /// Suggested fix, if available
    pub suggestion: Option<String>,
}

/// Different kinds of type checking errors
#[derive(Debug, Clone, PartialEq)]
pub enum TypeErrorKind {
    /// Type mismatch: expected vs actual
    TypeMismatch { expected: TypeId, actual: TypeId },

    /// Undefined type reference
    UndefinedType { name: InternedString },

    /// Undefined symbol reference
    UndefinedSymbol { name: InternedString },

    /// Invalid type arguments for generic type
    InvalidTypeArguments {
        base_type: TypeId,
        expected_count: usize,
        actual_count: usize,
    },

    /// Type constraint violation
    ConstraintViolation {
        type_param: TypeId,
        constraint: TypeId,
        violating_type: TypeId,
    },

    /// Circular type dependency
    CircularDependency { types: Vec<TypeId> },

    /// Invalid cast/conversion
    InvalidCast { from_type: TypeId, to_type: TypeId },

    /// Function signature mismatch
    SignatureMismatch {
        expected_params: Vec<TypeId>,
        actual_params: Vec<TypeId>,
        expected_return: TypeId,
        actual_return: TypeId,
    },

    /// Access violation (private/protected access)
    AccessViolation {
        symbol_id: SymbolId,
        required_access: AccessLevel,
    },

    /// Generic type inference failed
    InferenceFailed { reason: String },

    /// Interface implementation is missing or incorrect
    InterfaceNotImplemented {
        interface_type: TypeId,
        class_type: TypeId,
        missing_method: InternedString,
    },

    /// Method signature doesn't match interface requirement
    MethodSignatureMismatch {
        expected: TypeId,
        actual: TypeId,
        method_name: InternedString,
    },

    /// Missing override modifier on overriding method
    MissingOverride {
        method_name: InternedString,
        parent_class: InternedString,
    },

    /// Invalid override - no parent method to override
    InvalidOverride { method_name: InternedString },

    /// Accessing static member through instance
    StaticAccessFromInstance {
        member_name: InternedString,
        class_name: InternedString,
    },

    /// Accessing instance member through static context
    InstanceAccessFromStatic {
        member_name: InternedString,
        class_name: InternedString,
    },

    /// Import-related errors
    ImportError { message: String },

    /// Unknown symbol reference
    UnknownSymbol { name: String },
}

/// Access levels for visibility checking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessLevel {
    Private,
    Protected,
    Internal,
    Public,
}

/// Type compatibility relationship
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeCompatibility {
    /// Types are identical
    Identical,
    /// Source type is assignable to target type
    Assignable,
    /// Types are compatible with implicit conversion
    Convertible,
    /// Types are incompatible
    Incompatible,
}

/// Type inference constraint
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypeConstraint {
    /// Type variable being constrained
    pub type_var: TypeId,
    /// Constraint kind
    pub kind: ConstraintKind,
    /// Source location for error reporting
    pub location: SourceLocation,

    /// Priority for constraint solving (higher = more important)
    pub priority: ConstraintPriority,

    /// Whether this constraint can be relaxed during solving
    pub is_soft: bool,
}

/// Priority levels for constraint solving
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConstraintPriority {
    /// User-specified explicit constraints (highest priority)
    Explicit = 100,

    /// Inferred from type annotations
    TypeAnnotation = 80,

    /// Inferred from constraint propagation
    Inferred = 70,

    /// Inferred from method/field usage
    Usage = 60,

    /// Inferred from context
    Contextual = 40,

    /// Default/fallback constraints (lowest priority)
    Default = 20,
}

/// Different kinds of type constraints
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Type variable must equal this type
    Equality { target_type: TypeId },

    /// Type variable must be subtype of this type
    Subtype { supertype: TypeId },

    /// Type variable must be supertype of this type
    Supertype { subtype: TypeId },

    /// Type variable must implement this interface
    Implements { interface_type: TypeId },

    /// Method signature constraint: T has method(args) -> return_type
    HasMethod {
        method_name: InternedString,
        signature: TypeId,
        /// Whether the method must be static
        is_static: bool,
    },

    /// Field access constraint: T has field of type
    HasField {
        field_name: InternedString,
        field_type: TypeId,
        /// Whether field must be public
        is_public: bool,
    },

    /// Index access constraint: T[K] -> V
    IndexAccess {
        key_type: TypeId,
        value_type: TypeId,
    },

    /// Iterator constraint: T is iterable over element_type
    Iterable { element_type: Option<TypeId> },

    /// Comparable constraint: T can be compared (has comparison operators)
    Comparable,

    /// Arithmetic constraint: T supports arithmetic operations
    Arithmetic,

    /// String convertible: T can be converted to string
    StringConvertible,

    /// Constructible constraint: T can be constructed with given arguments
    Constructible { constructor_args: Vec<TypeId> },

    /// Function call constraint: T can be called as function
    Callable {
        params: Vec<TypeId>,
        return_type: TypeId,
    },

    /// Size constraint: T has compile-time size information
    Sized,

    /// Copy constraint: T can be copied (not moved)
    Copy,

    /// Reference constraint: T can be referenced
    Referenceable,

    /// Exclusion constraint: T cannot be certain types (negative constraint)
    Excludes { excluded_types: Vec<TypeId> },

    /// Type variable must be one of these types
    OneOf { candidates: Vec<TypeId> },

    /// Custom predicate constraint (for advanced cases)
    Custom {
        predicate_name: InternedString,
        args: Vec<TypeId>,
    },
}

/// A set of constraints with efficient operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintSet {
    /// All constraints indexed by type variable
    constraints_by_var: HashMap<TypeId, Vec<TypeConstraint>>,

    /// Constraints indexed by kind for fast lookups
    constraints_by_kind: HashMap<std::mem::Discriminant<ConstraintKind>, Vec<TypeConstraint>>,

    /// Dependency graph for constraint propagation
    dependencies: HashMap<TypeId, HashSet<TypeId>>,

    /// Statistics for performance tracking
    stats: ConstraintStats,
}

/// Statistics for constraint set operations
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConstraintStats {
    pub total_constraints: usize,
    pub constraints_by_priority: HashMap<ConstraintPriority, usize>,
    pub constraints_by_kind: HashMap<String, usize>,
    pub dependency_edges: usize,
    pub propagation_rounds: usize,
    pub constraints_added: usize,
    pub constraints_removed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl ConstraintSet {
    /// Create a new empty constraint set
    pub fn new() -> Self {
        Self {
            constraints_by_var: HashMap::new(),
            constraints_by_kind: HashMap::new(),
            dependencies: HashMap::new(),
            stats: ConstraintStats::default(),
        }
    }

    /// Add a constraint to the set
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.stats.constraints_added += 1;
        self.stats.total_constraints += 1;

        // Update priority statistics
        *self
            .stats
            .constraints_by_priority
            .entry(constraint.priority)
            .or_insert(0) += 1;

        // Update kind statistics
        let kind_name = format!("{:?}", std::mem::discriminant(&constraint.kind));
        *self.stats.constraints_by_kind.entry(kind_name).or_insert(0) += 1;

        // Add to variable index
        self.constraints_by_var
            .entry(constraint.type_var)
            .or_insert_with(Vec::new)
            .push(constraint.clone());

        // Add to kind index
        let discriminant = std::mem::discriminant(&constraint.kind);
        self.constraints_by_kind
            .entry(discriminant)
            .or_insert_with(Vec::new)
            .push(constraint.clone());

        // Update dependencies based on constraint
        self.update_dependencies(&constraint);
    }

    /// Get all constraints for a type variable
    pub fn constraints_for(&self, type_var: TypeId) -> &[TypeConstraint] {
        self.constraints_by_var
            .get(&type_var)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get constraints of a specific kind
    pub fn constraints_of_kind(&self, kind: &ConstraintKind) -> Vec<&TypeConstraint> {
        let discriminant = std::mem::discriminant(kind);
        self.constraints_by_kind
            .get(&discriminant)
            .map(|constraints| {
                constraints
                    .iter()
                    .filter(|c| std::mem::discriminant(&c.kind) == discriminant)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a type variable has constraints
    pub fn has_constraints(&self, type_var: TypeId) -> bool {
        self.constraints_by_var.contains_key(&type_var)
    }

    /// Get all constraints as an iterator
    pub fn constraints(&self) -> impl Iterator<Item = &TypeConstraint> {
        self.constraints_by_var.values().flatten()
    }

    /// Remove all constraints for a type variable
    pub fn remove_constraints_for(&mut self, type_var: TypeId) -> usize {
        if let Some(constraints) = self.constraints_by_var.remove(&type_var) {
            let count = constraints.len();
            self.stats.constraints_removed += count;
            self.stats.total_constraints = self.stats.total_constraints.saturating_sub(count);

            // Update kind index
            for constraint in &constraints {
                let discriminant = std::mem::discriminant(&constraint.kind);
                if let Some(kind_constraints) = self.constraints_by_kind.get_mut(&discriminant) {
                    kind_constraints.retain(|c| c.type_var != type_var);
                    if kind_constraints.is_empty() {
                        self.constraints_by_kind.remove(&discriminant);
                    }
                }
            }

            // Update dependencies
            self.dependencies.remove(&type_var);
            for deps in self.dependencies.values_mut() {
                deps.remove(&type_var);
            }

            count
        } else {
            0
        }
    }

    /// Get dependencies for a type variable
    pub fn dependencies_of(&self, type_var: TypeId) -> &HashSet<TypeId> {
        static EMPTY_SET: LazyLock<HashSet<TypeId>> = LazyLock::new(|| HashSet::new());
        self.dependencies.get(&type_var).unwrap_or(&EMPTY_SET)
    }

    /// Get all type variables that depend on the given variable
    pub fn dependents_of(&self, type_var: TypeId) -> Vec<TypeId> {
        self.dependencies
            .iter()
            .filter(|(_, deps)| deps.contains(&type_var))
            .map(|(var, _)| *var)
            .collect()
    }

    /// Check if there's a dependency cycle
    pub fn has_dependency_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut recursion_stack = HashSet::new();

        for &type_var in self.dependencies.keys() {
            if !visited.contains(&type_var) {
                if self.has_cycle_dfs(type_var, &mut visited, &mut recursion_stack) {
                    return true;
                }
            }
        }
        false
    }

    /// Get constraint solving order (topological sort)
    pub fn solving_order(&self) -> Result<Vec<TypeId>, Vec<TypeId>> {
        let mut in_degree: HashMap<TypeId, usize> = HashMap::new();
        let mut out_edges: HashMap<TypeId, Vec<TypeId>> = HashMap::new();

        // Build degree counts
        for (&var, deps) in &self.dependencies {
            in_degree.entry(var).or_insert(0);
            for &dep in deps {
                in_degree.entry(dep).or_insert(0);
                out_edges.entry(dep).or_insert_with(Vec::new).push(var);
                *in_degree.get_mut(&var).unwrap() += 1;
            }
        }

        // Kahn's algorithm for topological sorting
        let mut queue: Vec<TypeId> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&var, _)| var)
            .collect();

        let mut result = Vec::new();

        while let Some(var) = queue.pop() {
            result.push(var);

            if let Some(neighbors) = out_edges.get(&var) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(neighbor);
                    }
                }
            }
        }

        if result.len() == in_degree.len() {
            Ok(result)
        } else {
            // Return variables involved in cycles
            let cycle_vars: Vec<TypeId> = in_degree
                .iter()
                .filter(|(_, &degree)| degree > 0)
                .map(|(&var, _)| var)
                .collect();
            Err(cycle_vars)
        }
    }

    /// Get statistics about the constraint set
    pub fn stats(&self) -> &ConstraintStats {
        &self.stats
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraints_by_var.clear();
        self.constraints_by_kind.clear();
        self.dependencies.clear();
        self.stats = ConstraintStats::default();
    }

    /// Merge another constraint set into this one
    pub fn merge(&mut self, other: ConstraintSet) {
        for constraints in other.constraints_by_var.into_values() {
            for constraint in constraints {
                self.add_constraint(constraint);
            }
        }
    }

    /// Compute a stable content-based hash for this constraint set
    /// This hash is based on the semantic content of the constraints, not their container
    pub fn compute_content_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash the total number of constraints for quick differentiation
        self.stats.total_constraints.hash(&mut hasher);

        // Collect and sort constraints for deterministic hashing
        let mut all_constraints: Vec<&TypeConstraint> = Vec::new();
        for constraints in self.constraints_by_var.values() {
            all_constraints.extend(constraints.iter());
        }

        // Sort constraints by type_var first, then by kind discriminant for determinism
        all_constraints.sort_by(|a, b| {
            a.type_var
                .as_raw()
                .cmp(&b.type_var.as_raw())
                .then_with(|| {
                    let a_discriminant = std::mem::discriminant(&a.kind);
                    let b_discriminant = std::mem::discriminant(&b.kind);
                    format!("{:?}", a_discriminant).cmp(&format!("{:?}", b_discriminant))
                })
                .then_with(|| a.priority.cmp(&b.priority))
        });

        // Hash each constraint
        for constraint in all_constraints {
            constraint.hash(&mut hasher);
        }

        // Hash dependency information (sorted for determinism)
        let mut sorted_deps: Vec<_> = self.dependencies.iter().collect();
        sorted_deps.sort_by_key(|(type_var, _)| type_var.as_raw());

        for (type_var, deps) in sorted_deps {
            type_var.hash(&mut hasher);
            let mut sorted_dep_list: Vec<_> = deps.iter().collect();
            sorted_dep_list.sort_by_key(|dep| dep.as_raw());
            for dep in sorted_dep_list {
                dep.hash(&mut hasher);
            }
        }

        // Hash key statistics that affect constraint solving behavior
        self.stats.total_constraints.hash(&mut hasher);
        self.stats.dependency_edges.hash(&mut hasher);

        hasher.finish()
    }

    // === Private helper methods ===

    fn update_dependencies(&mut self, constraint: &TypeConstraint) {
        let deps = self
            .dependencies
            .entry(constraint.type_var)
            .or_insert_with(HashSet::new);

        // Add dependencies based on constraint kind
        match &constraint.kind {
            ConstraintKind::Equality { target_type } => {
                deps.insert(*target_type);
            }
            ConstraintKind::Subtype { supertype } => {
                deps.insert(*supertype);
            }
            ConstraintKind::Implements { interface_type } => {
                deps.insert(*interface_type);
            }
            ConstraintKind::HasMethod { signature, .. } => {
                deps.insert(*signature);
            }
            ConstraintKind::HasField { field_type, .. } => {
                deps.insert(*field_type);
            }
            ConstraintKind::IndexAccess {
                key_type,
                value_type,
            } => {
                deps.insert(*key_type);
                deps.insert(*value_type);
            }
            ConstraintKind::Iterable {
                element_type: Some(elem_type),
            } => {
                deps.insert(*elem_type);
            }
            ConstraintKind::Constructible { constructor_args } => {
                deps.extend(constructor_args.iter());
            }
            ConstraintKind::Callable {
                params,
                return_type,
            } => {
                deps.extend(params.iter());
                deps.insert(*return_type);
            }
            ConstraintKind::Excludes { excluded_types } => {
                deps.extend(excluded_types.iter());
            }
            ConstraintKind::OneOf { candidates } => {
                deps.extend(candidates.iter());
            }
            ConstraintKind::Custom { args, .. } => {
                deps.extend(args.iter());
            }
            // Other constraints don't introduce type dependencies
            _ => {}
        }

        if !deps.is_empty() {
            self.stats.dependency_edges += deps.len();
        }
    }

    fn has_cycle_dfs(
        &self,
        var: TypeId,
        visited: &mut HashSet<TypeId>,
        recursion_stack: &mut HashSet<TypeId>,
    ) -> bool {
        visited.insert(var);
        recursion_stack.insert(var);

        if let Some(deps) = self.dependencies.get(&var) {
            for &dep in deps {
                if !visited.contains(&dep) {
                    if self.has_cycle_dfs(dep, visited, recursion_stack) {
                        return true;
                    }
                } else if recursion_stack.contains(&dep) {
                    return true;
                }
            }
        }

        recursion_stack.remove(&var);
        false
    }
}

impl Default for ConstraintSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Constraint validation result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintValidation {
    /// Constraint is satisfied
    Satisfied,

    /// Constraint is violated
    Violated {
        reason: String,
        suggestion: Option<String>,
    },

    /// Constraint cannot be checked yet (depends on unresolved types)
    Pending { waiting_for: Vec<TypeId> },

    /// Constraint is conditionally satisfied
    Conditional {
        condition: String,
        alternative: Box<ConstraintValidation>,
    },
}

/// Enhanced constraint validator for sophisticated type checking
pub struct ConstraintValidator {
    /// Cache for constraint validation results
    validation_cache: HashMap<(TypeId, ConstraintKind), ConstraintValidation>,

    /// Statistics tracking
    stats: ValidationStats,
}

#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    pub validations_performed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub pending_constraints: usize,
    pub violated_constraints: usize,
    pub satisfied_constraints: usize,
}

impl ConstraintValidator {
    pub fn new() -> Self {
        Self {
            validation_cache: HashMap::new(),
            stats: ValidationStats::default(),
        }
    }

    /// Validate a constraint against a concrete type
    pub fn validate_constraint(
        &mut self,
        type_id: TypeId,
        constraint: &ConstraintKind,
        type_table: &RefCell<TypeTable>,
    ) -> ConstraintValidation {
        self.stats.validations_performed += 1;

        // Check cache first
        let cache_key = (type_id, constraint.clone());
        if let Some(cached) = self.validation_cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }

        self.stats.cache_misses += 1;

        let result = self.validate_constraint_impl(type_id, constraint, type_table);

        // Update statistics
        match &result {
            ConstraintValidation::Satisfied => self.stats.satisfied_constraints += 1,
            ConstraintValidation::Violated { .. } => self.stats.violated_constraints += 1,
            ConstraintValidation::Pending { .. } => self.stats.pending_constraints += 1,
            ConstraintValidation::Conditional { .. } => {} // Count as neither satisfied nor violated
        }

        // Cache the result
        self.validation_cache.insert(cache_key, result.clone());
        result
    }

    /// Clear the validation cache
    pub fn clear_cache(&mut self) {
        self.validation_cache.clear();
    }

    /// Get validation statistics
    pub fn stats(&self) -> &ValidationStats {
        &self.stats
    }

    // Implementation of actual constraint validation logic
    fn validate_constraint_impl(
        &self,
        type_id: TypeId,
        constraint: &ConstraintKind,
        type_table: &RefCell<TypeTable>,
    ) -> ConstraintValidation {
        // This would integrate with the existing type system from Phase 1
        // For now, we'll provide the structure and some basic implementations

        match constraint {
            ConstraintKind::Equality { target_type } => {
                if type_id == *target_type {
                    ConstraintValidation::Satisfied
                } else {
                    ConstraintValidation::Violated {
                        reason: format!("Type {:?} is not equal to {:?}", type_id, target_type),
                        suggestion: Some("Check type annotation".to_string()),
                    }
                }
            }

            ConstraintKind::Comparable => {
                // Check if type has comparison operators
                if self.type_has_comparison_ops(type_id, type_table) {
                    ConstraintValidation::Satisfied
                } else {
                    ConstraintValidation::Violated {
                        reason: "Type does not implement comparison operators".to_string(),
                        suggestion: Some("Implement comparable interface".to_string()),
                    }
                }
            }

            ConstraintKind::Arithmetic => {
                if self.type_has_arithmetic_ops(type_id, type_table) {
                    ConstraintValidation::Satisfied
                } else {
                    ConstraintValidation::Violated {
                        reason: "Type does not support arithmetic operations".to_string(),
                        suggestion: Some(
                            "Use numeric type or implement arithmetic operators".to_string(),
                        ),
                    }
                }
            }

            ConstraintKind::Sized => {
                if self.type_has_known_size(type_id, type_table) {
                    ConstraintValidation::Satisfied
                } else {
                    ConstraintValidation::Violated {
                        reason: "Type does not have known size at compile time".to_string(),
                        suggestion: Some("Use sized type".to_string()),
                    }
                }
            }

            // More sophisticated validations would be implemented here
            // integrating with the Phase 1 type system and Phase 2 type checker
            _ => ConstraintValidation::Pending {
                waiting_for: vec![type_id],
            },
        }
    }

    // Helper methods for type property checking
    fn type_has_comparison_ops(&self, type_id: TypeId, type_table: &RefCell<TypeTable>) -> bool {
        // Check if type supports comparison operators
        if let Some(type_obj) = type_table.borrow().get(type_id) {
            match &type_obj.kind {
                // Primitive types generally support comparison
                super::core::TypeKind::Bool
                | super::core::TypeKind::Int
                | super::core::TypeKind::Float
                | super::core::TypeKind::String => true,

                // Dynamic doesn't guarantee comparison support
                super::core::TypeKind::Dynamic => false,

                // Void and Error types don't support comparison
                super::core::TypeKind::Void | super::core::TypeKind::Error => false,

                // Arrays and other types would need specific checks
                _ => true, // Assume others support it for now
            }
        } else {
            false
        }
    }

    fn type_has_arithmetic_ops(&self, type_id: TypeId, type_table: &RefCell<TypeTable>) -> bool {
        // Check if type supports arithmetic operators
        if let Some(type_obj) = type_table.borrow().get(type_id) {
            match &type_obj.kind {
                // Only numeric types support arithmetic
                super::core::TypeKind::Int | super::core::TypeKind::Float => true,

                // String supports + for concatenation
                super::core::TypeKind::String => true,

                // Other types don't support arithmetic
                _ => false,
            }
        } else {
            false
        }
    }

    fn type_has_known_size(&self, type_id: TypeId, type_table: &RefCell<TypeTable>) -> bool {
        // Check if type has compile-time known size
        if let Some(type_obj) = type_table.borrow().get(type_id) {
            match &type_obj.kind {
                // Primitive types have known size
                super::core::TypeKind::Void
                | super::core::TypeKind::Bool
                | super::core::TypeKind::Int
                | super::core::TypeKind::Float
                | super::core::TypeKind::String => true,

                // Dynamic type does NOT have known size at compile time
                super::core::TypeKind::Dynamic => false,

                // Error and Unknown types don't have known size
                super::core::TypeKind::Error | super::core::TypeKind::Unknown => false,

                // Other types would need specific analysis
                super::core::TypeKind::Array { .. } | super::core::TypeKind::Map { .. } => true, // Known structure

                super::core::TypeKind::Optional { .. } => true, // Wrapper type

                // Classes and interfaces have known layout
                super::core::TypeKind::Class { .. }
                | super::core::TypeKind::Interface { .. }
                | super::core::TypeKind::Enum { .. } => true,

                // Functions have known signature size
                super::core::TypeKind::Function { .. } => true,

                _ => true, // Assume others are sized for now
            }
        } else {
            false
        }
    }
}

impl Default for ConstraintValidator {
    fn default() -> Self {
        Self::new()
    }
}

// Display implementations for better debugging
impl fmt::Display for ConstraintKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintKind::Equality { target_type } => write!(f, "= {:?}", target_type),
            ConstraintKind::Subtype { supertype } => write!(f, "<: {:?}", supertype),
            ConstraintKind::Implements { interface_type } => write!(f, ": {:?}", interface_type),
            ConstraintKind::HasMethod { method_name, .. } => {
                write!(f, "has method {}", method_name)
            }
            ConstraintKind::HasField { field_name, .. } => write!(f, "has field {}", field_name),
            ConstraintKind::Comparable => write!(f, "Comparable"),
            ConstraintKind::Arithmetic => write!(f, "Arithmetic"),
            ConstraintKind::StringConvertible => write!(f, "StringConvertible"),
            ConstraintKind::Sized => write!(f, "Sized"),
            ConstraintKind::Copy => write!(f, "Copy"),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl fmt::Display for TypeConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} {}", self.type_var, self.kind)
    }
}

/// Type inference context and constraint solver
#[derive(Debug)]
pub struct TypeInference {
    /// Active type variables and their constraints
    type_variables: IdMap<TypeId, Vec<TypeConstraint>>,

    /// Solved type variables
    solutions: IdMap<TypeId, TypeId>,

    /// Constraint propagation queue
    constraint_queue: Vec<TypeConstraint>,

    /// Inference statistics
    stats: InferenceStats,
}

/// Statistics about type inference performance
#[derive(Debug, Default, Clone)]
pub struct InferenceStats {
    /// Number of type variables created
    pub type_variables_created: usize,
    /// Number of constraints generated
    pub constraints_generated: usize,
    /// Number of constraints solved
    pub constraints_solved: usize,
    /// Number of inference failures
    pub inference_failures: usize,
}

/// Class hierarchy information stored in the symbol table
#[derive(Debug, Clone)]
pub struct ClassHierarchyInfo {
    /// Direct superclass (if any)
    pub superclass: Option<TypeId>,

    /// Directly implemented interfaces
    pub interfaces: Vec<TypeId>,

    /// All supertypes (transitive closure) for fast lookup
    pub all_supertypes: HashSet<TypeId>,

    /// Depth in hierarchy (Object = 0, direct subclasses = 1, etc.)
    pub depth: usize,

    /// Whether the class is final (cannot be extended)
    pub is_final: bool,

    /// Whether the class is abstract (cannot be instantiated)
    pub is_abstract: bool,

    /// Whether the class is extern (defined externally)
    pub is_extern: bool,

    /// Whether this is an interface (not a class)
    pub is_interface: bool,

    /// Types this abstract/enum is sealed to (if any)
    pub sealed_to: Option<Vec<TypeId>>,
}

/// Extension to SymbolTable for class hierarchy
pub trait ClassHierarchyRegistry {
    fn register_class_hierarchy(&mut self, class_id: SymbolId, info: ClassHierarchyInfo);
    fn get_class_hierarchy(&self, class_id: SymbolId) -> Option<&ClassHierarchyInfo>;
    fn compute_all_supertypes(&self, class_id: SymbolId) -> HashSet<TypeId>;
}

/// Main type checking engine
pub struct TypeChecker<'a> {
    /// Type table for type storage and lookup
    pub(crate) type_table: &'a RefCell<TypeTable>,

    /// Symbol table for symbol resolution
    pub(crate) symbol_table: &'a SymbolTable,

    /// Scope tree for context resolution
    scope_tree: &'a ScopeTree,

    /// String interner for names
    string_interner: &'a StringInterner,

    /// Type inference engine
    inference: TypeInference,

    /// Current checking context
    context: TypeCheckContext,

    /// Error accumulation
    errors: Vec<TypeCheckError>,

    /// Subtype relationship cache for performance
    subtype_cache: HashMap<(TypeId, TypeId), bool>,

    /// Generic instantiation context
    generic_context: GenericContext,

    /// Type checking statistics
    stats: TypeCheckerStats,
}

/// Context for type checking operations
#[derive(Debug, Clone)]
pub struct TypeCheckContext {
    /// Current scope being checked
    pub current_scope: ScopeId,

    /// Current function being checked (if any)
    pub current_function: Option<SymbolId>,

    /// Current class being checked (if any)
    pub current_class: Option<SymbolId>,

    /// Generic type parameters in scope
    pub type_parameters: IdSet<TypeId>,

    /// Expected return type for current function
    pub expected_return_type: Option<TypeId>,
}

/// Context for generic type operations
#[derive(Debug, Default)]
pub struct GenericContext {
    /// Type parameter substitutions
    substitutions: IdMap<TypeId, TypeId>,

    /// Active generic instantiations (to detect recursion)
    active_instantiations: IdSet<TypeId>,
}

impl<'a> TypeChecker<'a> {
    /// Create a new type checker
    pub fn new(
        type_table: &'a RefCell<TypeTable>,
        symbol_table: &'a SymbolTable,
        scope_tree: &'a ScopeTree,
        string_interner: &'a StringInterner,
    ) -> Self {
        Self {
            type_table,
            symbol_table,
            scope_tree,
            string_interner,
            inference: TypeInference::new(),
            context: TypeCheckContext {
                current_scope: ScopeId::invalid(),
                current_function: None,
                current_class: None,
                type_parameters: new_id_set(),
                expected_return_type: None,
            },
            errors: Vec::new(),
            subtype_cache: HashMap::new(),
            generic_context: GenericContext::default(),
            stats: TypeCheckerStats {
                types_checked: 0,
                errors_found: 0,
                inference_stats: InferenceStats::default(),
                cache_hits: 0,
            },
        }
    }

    /// Set the current checking context
    pub fn with_context(&mut self, context: TypeCheckContext) -> &mut Self {
        self.context = context;
        self
    }

    /// Check if two types are compatible
    pub fn check_compatibility(&mut self, source: TypeId, target: TypeId) -> TypeCompatibility {
        // Increment stats counter
        self.stats.types_checked += 1;

        // Fast path: identical types
        if source == target {
            return TypeCompatibility::Identical;
        }

        // Get the actual type objects
        let source_type = match self.type_table.borrow().get(source) {
            Some(t) => t.kind.clone(),
            None => return TypeCompatibility::Incompatible,
        };

        let target_type = match self.type_table.borrow().get(target) {
            Some(t) => t.kind.clone(),
            None => return TypeCompatibility::Incompatible,
        };

        // Check various compatibility rules
        self.check_compatibility_impl(&source_type, &target_type, source, target)
    }

    /// Internal compatibility checking implementation
    fn check_compatibility_impl(
        &mut self,
        source: &super::core::TypeKind,
        target: &super::core::TypeKind,
        source_id: TypeId,
        _target_id: TypeId,
    ) -> TypeCompatibility {
        match (&source, &target) {
            // === Identity Cases ===
            (TypeKind::Void, TypeKind::Void)
            | (TypeKind::Bool, TypeKind::Bool)
            | (TypeKind::Int, TypeKind::Int)
            | (TypeKind::Float, TypeKind::Float)
            | (TypeKind::String, TypeKind::String) => TypeCompatibility::Identical,

            // === Numeric Conversions ===
            (TypeKind::Int, TypeKind::Float) => TypeCompatibility::Convertible,

            // === Dynamic Type (compatible with everything) ===
            (_, TypeKind::Dynamic) | (TypeKind::Dynamic, _) => TypeCompatibility::Assignable,

            // === Null Safety ===
            (_, TypeKind::Optional { inner_type }) => {
                // T is assignable to T?
                let inner_type_id = *inner_type; // Copy the TypeId
                                                 // Now we can make the mutable call without holding references
                let inner_compat = self.check_compatibility(source_id, inner_type_id);
                match inner_compat {
                    TypeCompatibility::Identical | TypeCompatibility::Assignable => {
                        TypeCompatibility::Assignable
                    }
                    _ => TypeCompatibility::Incompatible,
                }
            }

            (TypeKind::Optional { inner_type }, _) => {
                // T? is assignable to T only with explicit null check
                TypeCompatibility::Incompatible // Requires explicit unwrapping
            }

            (TypeKind::Union { .. }, _) => {
                self.check_compatibility_with_unions(source_id, _target_id)
            }

            (_, TypeKind::Union { .. }) => {
                self.check_compatibility_with_unions(source_id, _target_id)
            }

            // === Array Types ===
            (
                TypeKind::Array {
                    element_type: source_elem,
                },
                TypeKind::Array {
                    element_type: target_elem,
                },
            ) => {
                // Arrays are covariant in Haxe
                let source_elem_id = *source_elem; // Copy the TypeIds
                let target_elem_id = *target_elem;
                let elem_compat = self.check_compatibility(source_elem_id, target_elem_id);
                match elem_compat {
                    TypeCompatibility::Identical => TypeCompatibility::Identical,
                    TypeCompatibility::Assignable => TypeCompatibility::Assignable,
                    _ => TypeCompatibility::Incompatible,
                }
            }

            // === Function Types ===
            (
                TypeKind::Function {
                    params: source_params,
                    return_type: source_ret,
                    ..
                },
                TypeKind::Function {
                    params: target_params,
                    return_type: target_ret,
                    ..
                },
            ) => {
                // Copy the data we need before making mutable calls
                let source_params = source_params.clone();
                let target_params = target_params.clone();
                let source_ret = *source_ret;
                let target_ret = *target_ret;
                self.check_function_compatibility(
                    &source_params,
                    source_ret,
                    &target_params,
                    target_ret,
                )
            }

            // === Named Types ===
            (TypeKind::Class { .. }, TypeKind::Class { .. }) => {
                self.check_inheritance_compatibility(source_id, _target_id)
            }

            (TypeKind::Class { .. }, TypeKind::Interface { .. }) => {
                self.check_inheritance_compatibility(source_id, _target_id)
            }

            // === Enum Types ===
            (
                TypeKind::Enum {
                    symbol_id: source_sym,
                    type_args: source_args,
                    ..
                },
                TypeKind::Enum {
                    symbol_id: target_sym,
                    type_args: target_args,
                    ..
                },
            ) => {
                if source_sym == target_sym {
                    // Same base enum: always compatible. Different constructors of the
                    // same enum (e.g., Ok(1) and Err("failed")) may have different
                    // partially-inferred type arguments but are the same enum type.
                    if source_args == target_args {
                        TypeCompatibility::Identical
                    } else {
                        TypeCompatibility::Assignable
                    }
                } else {
                    TypeCompatibility::Incompatible
                }
            }

            // === Generic Instances ===
            (
                TypeKind::GenericInstance {
                    base_type: source_base,
                    type_args: source_args,
                    ..
                },
                TypeKind::GenericInstance {
                    base_type: target_base,
                    type_args: target_args,
                    ..
                },
            ) => {
                if source_base == target_base && source_args.len() == target_args.len() {
                    // Clone the args before making mutable calls
                    let source_args = source_args.clone();
                    let target_args = target_args.clone();
                    self.check_generic_compatibility(&source_args, &target_args)
                } else {
                    TypeCompatibility::Incompatible
                }
            }

            // === Error Recovery ===
            (TypeKind::Error, _) | (_, TypeKind::Error) => TypeCompatibility::Assignable,
            // Unknown types should trigger type errors to help find bugs
            (TypeKind::Unknown, _) | (_, TypeKind::Unknown) => TypeCompatibility::Incompatible,

            // === Structural Subtyping: Anonymous ← Anonymous (width subtyping) ===
            (
                TypeKind::Anonymous {
                    fields: src_fields, ..
                },
                TypeKind::Anonymous {
                    fields: tgt_fields, ..
                },
            ) => {
                // All target fields must exist in source (by name)
                let all_found = tgt_fields
                    .iter()
                    .all(|tf| src_fields.iter().any(|sf| sf.name == tf.name));
                if all_found {
                    if src_fields.len() == tgt_fields.len() {
                        TypeCompatibility::Identical
                    } else {
                        TypeCompatibility::Assignable
                    }
                } else {
                    TypeCompatibility::Incompatible
                }
            }

            // === Structural Subtyping: Anonymous ← Class ===
            (TypeKind::Class { .. }, TypeKind::Anonymous { .. }) => {
                // Class instances can satisfy anonymous structural types.
                // Actual field checking is done at the MIR level during materialization.
                TypeCompatibility::Assignable
            }

            // === Default: Incompatible ===
            _ => TypeCompatibility::Incompatible,
        }
    }

    /// Enhanced compatibility checking that handles union types
    pub fn check_compatibility_with_unions(
        &mut self,
        source: TypeId,
        target: TypeId,
    ) -> TypeCompatibility {
        // Fast path: identical types
        if source == target {
            return TypeCompatibility::Identical;
        }

        // Get the actual type objects
        let source_type = match self.type_table.borrow().get(source) {
            Some(t) => t.kind.clone(),
            None => return TypeCompatibility::Incompatible,
        };

        let target_type = match self.type_table.borrow().get(target) {
            Some(t) => t.kind.clone(),
            None => return TypeCompatibility::Incompatible,
        };

        // Check union type compatibility
        match (&source_type, &target_type) {
            // Both are unions
            (
                TypeKind::Union {
                    types: source_types,
                },
                TypeKind::Union {
                    types: target_types,
                },
            ) => {
                // Every source type must be compatible with some target type
                for &source_member in source_types {
                    let mut found_compatible = false;

                    for &target_member in target_types {
                        let compat = self.check_compatibility(source_member, target_member);
                        if compat != TypeCompatibility::Incompatible {
                            found_compatible = true;
                            break;
                        }
                    }

                    if !found_compatible {
                        return TypeCompatibility::Incompatible;
                    }
                }

                TypeCompatibility::Assignable
            }
            // Source is union, target is not
            (
                TypeKind::Union {
                    types: source_types,
                },
                _,
            ) => {
                // A union is compatible with target if ANY member is compatible
                let mut best_compat = TypeCompatibility::Incompatible;

                for &member_type in source_types {
                    let compat = self.check_compatibility(member_type, target);
                    match compat {
                        TypeCompatibility::Identical => return TypeCompatibility::Identical,
                        TypeCompatibility::Assignable => {
                            best_compat = TypeCompatibility::Assignable
                        }
                        TypeCompatibility::Convertible => {
                            if best_compat == TypeCompatibility::Incompatible {
                                best_compat = TypeCompatibility::Convertible;
                            }
                        }
                        TypeCompatibility::Incompatible => {}
                    }
                }

                best_compat
            }

            // Target is union, source is not
            (
                _,
                TypeKind::Union {
                    types: target_types,
                },
            ) => {
                // Source is compatible with union if compatible with ANY member
                for &member_type in target_types {
                    let compat = self.check_compatibility(source, member_type);
                    if compat != TypeCompatibility::Incompatible {
                        return compat;
                    }
                }

                TypeCompatibility::Incompatible
            }

            // Neither is union - use existing logic
            _ => self.check_compatibility_impl(&source_type, &target_type, source, target),
        }
    }

    /// Check function type compatibility (contravariant in parameters, covariant in return)
    fn check_function_compatibility(
        &mut self,
        source_params: &[TypeId],
        source_return: TypeId,
        target_params: &[TypeId],
        target_return: TypeId,
    ) -> TypeCompatibility {
        // Parameter count must match
        if source_params.len() != target_params.len() {
            return TypeCompatibility::Incompatible;
        }

        // Check parameter compatibility (contravariant)
        let mut param_compat = TypeCompatibility::Identical;
        for (source_param, target_param) in source_params.iter().zip(target_params.iter()) {
            let compat = self.check_compatibility(*target_param, *source_param); // Note: reversed for contravariance
            match compat {
                TypeCompatibility::Incompatible => return TypeCompatibility::Incompatible,
                TypeCompatibility::Convertible => param_compat = TypeCompatibility::Assignable,
                TypeCompatibility::Assignable if param_compat == TypeCompatibility::Identical => {
                    param_compat = TypeCompatibility::Assignable;
                }
                _ => {}
            }
        }

        // Check return type compatibility (covariant)
        let return_compat = self.check_compatibility(source_return, target_return);
        match return_compat {
            TypeCompatibility::Incompatible => TypeCompatibility::Incompatible,
            TypeCompatibility::Identical if param_compat == TypeCompatibility::Identical => {
                TypeCompatibility::Identical
            }
            _ => TypeCompatibility::Assignable,
        }
    }

    /// Check generic type argument compatibility
    fn check_generic_compatibility(
        &mut self,
        source_args: &[TypeId],
        target_args: &[TypeId],
    ) -> TypeCompatibility {
        let mut overall_compat = TypeCompatibility::Identical;

        for (source_arg, target_arg) in source_args.iter().zip(target_args.iter()) {
            let arg_compat = self.check_compatibility(*source_arg, *target_arg);
            match arg_compat {
                TypeCompatibility::Incompatible => return TypeCompatibility::Incompatible,
                TypeCompatibility::Assignable if overall_compat == TypeCompatibility::Identical => {
                    overall_compat = TypeCompatibility::Assignable;
                }
                TypeCompatibility::Convertible => overall_compat = TypeCompatibility::Assignable,
                _ => {}
            }
        }

        overall_compat
    }

    pub(crate) fn get_parent_class(&self, class: SymbolId) -> Option<TypeId> {
        self.symbol_table.get_class_hierarchy(class)?.superclass
    }

    /// Find common supertype of two class types
    pub fn find_common_class_supertype(&self, type1: TypeId, type2: TypeId) -> Option<TypeId> {
        let class1 = self
            .type_table
            .borrow()
            .get_type_symbol(type1)
            .filter(|ty| ty.is_valid());
        if class1.is_none() {
            return None;
        }
        let class2 = self
            .type_table
            .borrow()
            .get_type_symbol(type2)
            .filter(|ty| ty.is_valid());
        if class2.is_none() {
            return None;
        }
        // Get hierarchy info for both classes
        let hierarchy1 = self.symbol_table.get_class_hierarchy(class1.unwrap())?;
        let hierarchy2 = self.symbol_table.get_class_hierarchy(class2.unwrap())?;

        if hierarchy1.all_supertypes.contains(&type2) {
            return Some(type2);
        }
        if hierarchy2.all_supertypes.contains(&type1) {
            return Some(type1);
        }

        // Find lowest common ancestor using depth information
        let mut ancestors1: Vec<TypeId> = hierarchy1.all_supertypes.iter().cloned().collect();
        let mut ancestors2: Vec<TypeId> = hierarchy2.all_supertypes.iter().cloned().collect();

        // Sort by depth (deeper classes first)
        ancestors1.sort_by_key(|&t| {
            if let Some(class_sym) = self.get_class_symbol_from_type(t) {
                if let Some(h) = self.symbol_table.get_class_hierarchy(class_sym) {
                    return std::cmp::Reverse(h.depth);
                }
            }
            std::cmp::Reverse(0)
        });

        // Find first common ancestor
        for ancestor in ancestors1 {
            if hierarchy2.all_supertypes.contains(&ancestor) {
                return Some(ancestor);
            }
        }

        // No common supertype found (shouldn't happen if Object is root)
        None
    }

    /// Check if source class is subtype of target class
    pub fn is_class_subtype_of(&self, source: TypeId, target: TypeId) -> bool {
        if source == target {
            return true;
        }

        if let Some(source_sym) = self
            .type_table
            .borrow()
            .get_type_symbol(source)
            .filter(|ty| ty.is_valid())
        {
            if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(source_sym) {
                return hierarchy.all_supertypes.contains(&target);
            }
        }

        false
    }

    /// Check if a class implements an interface
    pub fn implements_interface(&self, class_id: TypeId, interface_id: TypeId) -> bool {
        if let Some(class_sym) = self
            .type_table
            .borrow()
            .get_type_symbol(class_id)
            .filter(|ty| ty.is_valid())
        {
            if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(class_sym) {
                return hierarchy.all_supertypes.contains(&interface_id);
            }
        }

        false
    }

    /// Get class symbol from a type ID
    fn get_class_symbol_from_type(&self, type_id: TypeId) -> Option<SymbolId> {
        if let Some(ty) = self.type_table.borrow().get(type_id) {
            match &ty.kind {
                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                _ => None,
            }
        } else {
            None
        }
    }

    /// Check if source type is compatible with target type through inheritance
    /// This handles class/interface hierarchies and generic type arguments
    pub fn check_inheritance_compatibility(
        &mut self,
        source: TypeId,
        target: TypeId,
    ) -> TypeCompatibility {
        // Fast path: identical types
        if source == target {
            return TypeCompatibility::Identical;
        }

        // Get type information
        let binding = self.type_table.borrow();
        let source_type = match binding.get(source) {
            Some(t) => t,
            None => return TypeCompatibility::Incompatible,
        };

        let binding = self.type_table.borrow();
        let target_type = match binding.get(target) {
            Some(t) => t,
            None => return TypeCompatibility::Incompatible,
        };

        match (&source_type.kind, &target_type.kind) {
            // Class to Class
            (
                TypeKind::Class {
                    symbol_id: source_sym,
                    type_args: source_args,
                },
                TypeKind::Class {
                    symbol_id: target_sym,
                    type_args: target_args,
                },
            ) => self.check_class_to_class_compatibility(
                source_sym.clone(),
                source,
                source_args.as_slice(),
                target_sym.clone(),
                target,
                target_args.as_slice(),
            ),

            // Class to Interface
            (
                TypeKind::Class {
                    symbol_id: class_sym,
                    type_args: class_args,
                },
                TypeKind::Interface {
                    symbol_id: interface_sym,
                    type_args: interface_args,
                },
            ) => self.check_class_to_interface_compatibility(
                class_sym.clone(),
                source,
                class_args.as_slice(),
                interface_sym.clone(),
                target,
                interface_args.as_slice(),
            ),

            // Interface to Interface
            (
                TypeKind::Interface {
                    symbol_id: source_sym,
                    type_args: source_args,
                },
                TypeKind::Interface {
                    symbol_id: target_sym,
                    type_args: target_args,
                },
            ) => self.check_interface_to_interface_compatibility(
                *source_sym,
                source,
                source_args,
                *target_sym,
                target,
                target_args,
            ),

            // Interface to Class (always incompatible)
            (TypeKind::Interface { .. }, TypeKind::Class { .. }) => TypeCompatibility::Incompatible,

            // Not inheritance-related
            _ => TypeCompatibility::Incompatible,
        }
    }

    /// Check class to class compatibility
    fn check_class_to_class_compatibility(
        &mut self,
        source_class: SymbolId,
        source_type_id: TypeId,
        source_args: &[TypeId],
        target_class: SymbolId,
        target_type_id: TypeId,
        target_args: &[TypeId],
    ) -> TypeCompatibility {
        // Check if source is subclass of target
        if !self.is_class_subtype_of(source_type_id, target_type_id) {
            return TypeCompatibility::Incompatible;
        }

        // If same class, check type arguments
        if source_class == target_class {
            return self.check_type_arguments_compatibility(source_args, target_args);
        }

        // Source is subclass of target, need to check type arguments
        // with proper variance and substitution
        self.check_inherited_type_arguments(source_class, source_args, target_class, target_args)
    }

    /// Check class to interface compatibility
    fn check_class_to_interface_compatibility(
        &mut self,
        class_sym: SymbolId,
        class_id: TypeId,
        class_args: &[TypeId],
        interface_sym: SymbolId,
        interface_id: TypeId,
        interface_args: &[TypeId],
    ) -> TypeCompatibility {
        // Check if class implements interface
        if !self.implements_interface(class_id, interface_id) {
            return TypeCompatibility::Incompatible;
        }

        // Check type arguments with proper substitution
        self.check_interface_implementation_args(
            class_sym,
            class_args,
            interface_sym,
            interface_args,
        )
    }

    /// Check interface to interface compatibility
    fn check_interface_to_interface_compatibility(
        &mut self,
        source_interface: SymbolId,
        source_id: TypeId,
        source_args: &[TypeId],
        target_interface: SymbolId,
        target_id: TypeId,
        target_args: &[TypeId],
    ) -> TypeCompatibility {
        if source_interface != target_interface {
            return TypeCompatibility::Incompatible;
        }
        // Check if source interface extends target
        if !self.interface_extends(source_interface, target_id) {
            return TypeCompatibility::Incompatible;
        }

        // Check type arguments
        if source_interface == target_interface {
            self.check_type_arguments_compatibility(source_args, target_args)
        } else {
            self.check_inherited_type_arguments(
                source_interface,
                source_args,
                target_interface,
                target_args,
            )
        }
    }

    /// Check if one interface extends another
    fn interface_extends(&self, source: SymbolId, target: TypeId) -> bool {
        // Use class hierarchy info (interfaces use the same system)
        if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(source) {
            return hierarchy.all_supertypes.contains(&target);
        }

        false
    }

    /// Check type arguments compatibility with variance
    fn check_type_arguments_compatibility(
        &mut self,
        source_args: &[TypeId],
        target_args: &[TypeId],
    ) -> TypeCompatibility {
        if source_args.len() != target_args.len() {
            return TypeCompatibility::Incompatible;
        }

        // Apply variance checking for each type argument
        let mut overall_compat = TypeCompatibility::Identical;

        for (i, (source_arg, target_arg)) in source_args.iter().zip(target_args.iter()).enumerate()
        {
            // For now, use conservative variance rules:
            // - Most generic types use covariance for "output" types (like Array<T>, return types)
            // - Function parameter types are contravariant
            // - In absence of explicit variance annotations, use sound defaults

            let variance = self.infer_type_parameter_variance(i, source_args.len());

            let arg_compat = match variance {
                super::core::Variance::Covariant => {
                    // Source type argument can be more specific than target
                    // T <: U means Generic<T> <: Generic<U>
                    self.check_compatibility(*source_arg, *target_arg)
                }
                super::core::Variance::Contravariant => {
                    // Source type argument can be more general than target
                    // T :> U means Generic<T> <: Generic<U>
                    self.check_compatibility(*target_arg, *source_arg)
                }
                super::core::Variance::Invariant => {
                    // Type arguments must be exactly the same
                    if source_arg == target_arg {
                        TypeCompatibility::Identical
                    } else {
                        TypeCompatibility::Incompatible
                    }
                }
            };

            match arg_compat {
                TypeCompatibility::Incompatible => return TypeCompatibility::Incompatible,
                TypeCompatibility::Identical if overall_compat == TypeCompatibility::Identical => {
                    // Keep as Identical
                }
                TypeCompatibility::Assignable | TypeCompatibility::Convertible => {
                    overall_compat = TypeCompatibility::Assignable;
                }
                _ => {
                    overall_compat = TypeCompatibility::Assignable;
                }
            }
        }

        overall_compat
    }

    /// Infer variance for a type parameter position based on common patterns
    fn infer_type_parameter_variance(
        &self,
        parameter_index: usize,
        total_params: usize,
    ) -> super::core::Variance {
        // Use safe defaults that are common in object-oriented languages:
        // - For most generic types (Array<T>, List<T>, etc.), use covariance
        // - This allows Array<Dog> to be treated as Array<Animal> when Dog extends Animal
        // - This is sound for immutable/read-only access patterns

        // In the future, this could be improved by:
        // 1. Looking up actual variance annotations from type definitions
        // 2. Using variance inference based on how type parameters are used
        // 3. Supporting explicit variance annotations in Haxe syntax

        super::core::Variance::Covariant
    }

    /// Check type arguments when inheriting (with substitution)
    fn check_inherited_type_arguments(
        &mut self,
        source_class: SymbolId,
        source_args: &[TypeId],
        target_class: SymbolId,
        target_args: &[TypeId],
    ) -> TypeCompatibility {
        // Get the inheritance chain from source to target
        let chain = self.get_inheritance_chain(source_class, target_class);

        if chain.is_empty() {
            return TypeCompatibility::Incompatible;
        }

        // Apply type substitutions along the chain
        let substituted_args = self.substitute_type_arguments_along_chain(source_args, &chain);

        // Check final substituted arguments against target
        self.check_type_arguments_compatibility(&substituted_args, target_args)
    }

    /// Get inheritance chain from source to target class
    fn get_inheritance_chain(&self, source: SymbolId, target: SymbolId) -> Vec<SymbolId> {
        let mut chain = Vec::new();
        let mut current = source;

        // Walk up the inheritance hierarchy
        while current != target {
            chain.push(current);

            // Get superclass
            if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(current) {
                if let Some(superclass_type) = hierarchy.superclass {
                    if let Some(superclass_sym) = self.get_class_symbol_from_type(superclass_type) {
                        current = superclass_sym;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }

            // Prevent infinite loop
            if chain.len() > 100 {
                return Vec::new();
            }
        }

        // Check if we reached the target
        if current == target {
            chain.push(target);
            chain
        } else {
            Vec::new()
        }
    }

    /// Substitute type arguments along inheritance chain
    fn substitute_type_arguments_along_chain(
        &self,
        initial_args: &[TypeId],
        chain: &[SymbolId],
    ) -> Vec<TypeId> {
        if chain.is_empty() {
            return initial_args.to_vec();
        }

        let mut current_args = initial_args.to_vec();

        // Walk through the inheritance chain and apply type substitutions
        for i in 0..chain.len() - 1 {
            let current_class = chain[i];
            let next_class = chain[i + 1];

            // Apply type substitution from current_class to next_class
            current_args =
                self.substitute_type_arguments_to_parent(current_class, &current_args, next_class);
        }

        current_args
    }

    /// Substitute type arguments from a child class to its direct parent
    fn substitute_type_arguments_to_parent(
        &self,
        child_class: SymbolId,
        child_args: &[TypeId],
        parent_class: SymbolId,
    ) -> Vec<TypeId> {
        // Get the hierarchy info for the child class
        if let Some(hierarchy) = self.symbol_table.get_class_hierarchy(child_class) {
            if let Some(superclass_type) = hierarchy.superclass {
                // Get the parent class symbol and type arguments from the superclass type
                if let Some(parent_sym) = self.symbol_table.get_symbol_from_type(superclass_type) {
                    if parent_sym == parent_class {
                        // This is the direct parent - extract its type arguments
                        return self.extract_type_arguments_from_type(superclass_type);
                    }
                }

                // Check interfaces as well
                for &interface_type in &hierarchy.interfaces {
                    if let Some(interface_sym) =
                        self.symbol_table.get_symbol_from_type(interface_type)
                    {
                        if interface_sym == parent_class {
                            // This is the target interface - extract its type arguments
                            return self.extract_type_arguments_from_type(interface_type);
                        }
                    }
                }
            }
        }

        // Fallback: return child arguments unchanged
        // This happens when we can't find the exact inheritance relationship
        child_args.to_vec()
    }

    /// Extract type arguments from a generic type
    fn extract_type_arguments_from_type(&self, type_id: TypeId) -> Vec<TypeId> {
        if let Some(type_info) = self.type_table.borrow().get(type_id) {
            match &type_info.kind {
                super::core::TypeKind::Class { type_args, .. }
                | super::core::TypeKind::Interface { type_args, .. }
                | super::core::TypeKind::Enum { type_args, .. } => {
                    return type_args.clone();
                }
                _ => {}
            }
        }

        // No type arguments found
        vec![]
    }

    /// Check interface implementation with type arguments
    fn check_interface_implementation_args(
        &mut self,
        class_sym: SymbolId,
        class_args: &[TypeId],
        interface_sym: SymbolId,
        interface_args: &[TypeId],
    ) -> TypeCompatibility {
        // Find how the class implements the interface
        // This requires looking at the class definition to see
        // how it maps its type parameters to the interface's

        // For now, use simple compatibility check
        // TODO: Implement proper interface implementation checking
        self.check_type_arguments_compatibility(class_args, interface_args)
    }

    /// Check if two method signatures are compatible (implementation can be assigned to interface)
    ///
    /// **PLACEHOLDER**: This method is temporarily commented out until MethodSignature type is implemented
    #[allow(dead_code)]
    fn check_method_signature_compatibility(
        &self,
        _implementation_signature: &str, // Temporary placeholder - simplified until MethodSignature exists
        _interface_signature: &str, // Temporary placeholder - simplified until MethodSignature exists
    ) -> bool {
        // **TODO**: Implement proper method signature compatibility checking
        // when MethodSignature type is defined in the TAST system

        // For now, conservatively return true to avoid blocking compilation
        // This maintains type safety by being permissive rather than restrictive
        true
    }

    /// Check if a type is a subtype of another type
    pub fn is_subtype(&mut self, subtype: TypeId, supertype: TypeId) -> bool {
        // Check cache first
        let cache_key = (subtype, supertype);
        if let Some(&cached_result) = self.subtype_cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return cached_result;
        }

        let result = match self.check_compatibility(subtype, supertype) {
            TypeCompatibility::Identical | TypeCompatibility::Assignable => true,
            _ => false,
        };

        // Cache the result
        self.subtype_cache.insert(cache_key, result);
        result
    }

    /// Infer the type of an expression (simplified version)
    pub fn infer_expression_type(
        &mut self,
        expression_id: u32,
        scope_id: ScopeId,
    ) -> TypeCheckResult<TypeId> {
        // This would integrate with the actual AST expressions
        // For now, return unknown type
        Ok(self.type_table.borrow().unknown_type())
    }

    /// Check function call compatibility
    pub fn check_function_call(
        &mut self,
        function_type: TypeId,
        arg_types: &[TypeId],
        call_location: SourceLocation,
    ) -> TypeCheckResult<TypeId> {
        // Check if it's a function type and extract needed info
        let type_table = self.type_table.borrow();
        let func_type = match type_table.get(function_type) {
            Some(t) => t,
            None => {
                return Err(TypeCheckError {
                    kind: TypeErrorKind::UndefinedType {
                        name: self.string_interner.intern("<invalid-function>"),
                    },
                    location: call_location,
                    context: "Function type not found".to_string(),
                    suggestion: None,
                });
            }
        };

        match &func_type.kind {
            TypeKind::Function {
                params,
                return_type,
                ..
            } => {
                // Extract values we need before dropping the borrow
                let expected_params = params.clone();
                let expected_return = *return_type;
                let void_type = type_table.void_type();

                // Drop the borrow before doing more work
                drop(type_table);

                let len = expected_params.len();
                // Check parameter count
                if len != arg_types.len() {
                    return Err(TypeCheckError {
                        kind: TypeErrorKind::SignatureMismatch {
                            expected_params,
                            actual_params: arg_types.to_vec(),
                            expected_return,
                            actual_return: void_type,
                        },
                        location: call_location,
                        context: format!("Expected {} arguments, got {}", len, arg_types.len()),
                        suggestion: Some("Check the function signature".to_string()),
                    });
                }

                // Check each parameter type
                for (i, (expected_param, actual_arg)) in
                    expected_params.iter().zip(arg_types.iter()).enumerate()
                {
                    let compat = self.check_compatibility(*actual_arg, *expected_param);
                    if compat == TypeCompatibility::Incompatible {
                        return Err(TypeCheckError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: *expected_param,
                                actual: *actual_arg,
                            },
                            location: call_location,
                            context: format!("Argument {} type mismatch", i + 1),
                            suggestion: Some("Check the argument type".to_string()),
                        });
                    }
                }

                Ok(expected_return)
            }

            _ => Err(TypeCheckError {
                kind: TypeErrorKind::TypeMismatch {
                    expected: self
                        .type_table
                        .borrow_mut()
                        .create_function_type(vec![], self.type_table.borrow().void_type()),
                    actual: function_type,
                },
                location: call_location,
                context: "Expected function type".to_string(),
                suggestion: None,
            }),
        }
    }

    /// Validate generic type instantiation
    pub fn validate_generic_instantiation(
        &mut self,
        base_type: TypeId,
        type_args: &[TypeId],
        location: SourceLocation,
    ) -> TypeCheckResult<TypeId> {
        // Extract needed info in a scoped borrow so it's dropped before borrow_mut below
        let expected_param_count = {
            let binding = self.type_table.borrow();
            let base = binding.get(base_type).ok_or_else(|| TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<invalid-type>"),
                },
                location,
                context: "Base type not found".to_string(),
                suggestion: None,
            })?;

            match &base.kind {
                TypeKind::Class {
                    type_args: params, ..
                }
                | TypeKind::Interface {
                    type_args: params, ..
                }
                | TypeKind::Enum {
                    type_args: params, ..
                } => params.len(),
                _ => 0,
            }
        }; // binding dropped here

        if expected_param_count != type_args.len() {
            return Err(TypeCheckError {
                kind: TypeErrorKind::InvalidTypeArguments {
                    base_type,
                    expected_count: expected_param_count,
                    actual_count: type_args.len(),
                },
                location,
                context: "Generic type argument count mismatch".to_string(),
                suggestion: Some(format!("Expected {} type arguments", expected_param_count)),
            });
        }

        // TODO: Check type parameter constraints

        Ok(self
            .type_table
            .borrow_mut()
            .create_generic_instance(base_type, type_args.to_vec()))
    }

    /// Add a type checking error
    pub fn add_error(&mut self, error: TypeCheckError) {
        self.errors.push(error);
        self.stats.errors_found += 1;
    }

    /// Get all accumulated errors
    pub fn errors(&self) -> &[TypeCheckError] {
        &self.errors
    }

    /// Clear all errors
    pub fn clear_errors(&mut self) {
        self.errors.clear();
        self.stats.errors_found = 0;
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get type checking statistics
    pub fn stats(&self) -> TypeCheckerStats {
        TypeCheckerStats {
            types_checked: self.stats.types_checked,
            errors_found: self.stats.errors_found,
            inference_stats: self.inference.stats.clone(),
            cache_hits: self.stats.cache_hits,
        }
    }
}

impl TypeInference {
    /// Create a new type inference engine
    pub fn new() -> Self {
        Self {
            type_variables: new_id_map(),
            solutions: new_id_map(),
            constraint_queue: Vec::new(),
            stats: InferenceStats::default(),
        }
    }

    /// Create a new type variable
    pub fn create_type_variable(&mut self) -> TypeId {
        let var_id = TypeId::from_raw(self.stats.type_variables_created as u32);
        self.type_variables.insert(var_id, Vec::new());
        self.stats.type_variables_created += 1;
        var_id
    }

    /// Add a constraint for a type variable
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        if let Some(constraints) = self.type_variables.get_mut(&constraint.type_var) {
            constraints.push(constraint.clone());
        }
        self.constraint_queue.push(constraint);
        self.stats.constraints_generated += 1;
    }

    /// Solve all constraints
    pub fn solve_constraints(&mut self) -> Result<(), String> {
        while let Some(constraint) = self.constraint_queue.pop() {
            self.solve_constraint(constraint)?;
        }
        Ok(())
    }

    /// Solve a single constraint
    fn solve_constraint(&mut self, constraint: TypeConstraint) -> Result<(), String> {
        match constraint.kind {
            ConstraintKind::Equality { target_type } => {
                self.solutions.insert(constraint.type_var, target_type);
                self.stats.constraints_solved += 1;
                Ok(())
            }

            // TODO: Implement other constraint kinds
            _ => {
                self.stats.inference_failures += 1;
                Err("Constraint solving not fully implemented".to_string())
            }
        }
    }

    /// Get the solved type for a type variable
    pub fn get_solution(&self, type_var: TypeId) -> Option<TypeId> {
        self.solutions.get(&type_var).copied()
    }

    /// Reset the inference state
    pub fn reset(&mut self) {
        self.type_variables.clear();
        self.solutions.clear();
        self.constraint_queue.clear();
        self.stats = InferenceStats::default();
    }
}

/// Statistics about type checker performance
#[derive(Debug, Clone)]
pub struct TypeCheckerStats {
    /// Number of type compatibility checks performed
    pub types_checked: usize,
    /// Number of errors found
    pub errors_found: usize,
    /// Type inference statistics
    pub inference_stats: InferenceStats,
    /// Number of cache hits for performance
    pub cache_hits: usize,
}

/// Format a type for error messages
pub fn format_type_for_error(
    type_id: TypeId,
    type_table: &RefCell<TypeTable>,
    string_interner: &StringInterner,
) -> String {
    if let Some(type_obj) = type_table.borrow().get(type_id) {
        format_type_kind_for_error(&type_obj.kind, type_table, string_interner)
    } else {
        "<invalid-type>".to_string()
    }
}

/// Format a type kind for error messages
pub fn format_type_kind_for_error(
    kind: &TypeKind,
    type_table: &RefCell<TypeTable>,
    string_interner: &StringInterner,
) -> String {
    match kind {
        TypeKind::Void => "Void".to_string(),
        TypeKind::Bool => "Bool".to_string(),
        TypeKind::Int => "Int".to_string(),
        TypeKind::Float => "Float".to_string(),
        TypeKind::String => "String".to_string(),
        TypeKind::Dynamic => "Dynamic".to_string(),
        TypeKind::Unknown => "?".to_string(),
        TypeKind::Error => "<error>".to_string(),

        TypeKind::Array { element_type } => {
            format!(
                "Array<{}>",
                format_type_for_error(*element_type, type_table, string_interner)
            )
        }

        TypeKind::Optional { inner_type } => {
            format!(
                "{}?",
                format_type_for_error(*inner_type, type_table, string_interner)
            )
        }

        TypeKind::Function {
            params,
            return_type,
            ..
        } => {
            let param_strs: Vec<String> = params
                .iter()
                .map(|&p| format_type_for_error(p, type_table, string_interner))
                .collect();
            let return_str = format_type_for_error(*return_type, type_table, string_interner);
            format!("({}) -> {}", param_strs.join(", "), return_str)
        }

        _ => format!("{:?}", kind), // Fallback for complex types
    }
}

impl fmt::Display for TypeCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.context)
    }
}

impl std::error::Error for TypeCheckError {}

#[cfg(test)]
mod tests {
    use crate::tast::class_builder::ClassHierarchyBuilder;

    use super::*;

    fn create_test_setup() -> (RefCell<TypeTable>, SymbolTable, ScopeTree, StringInterner) {
        let type_table = RefCell::new(TypeTable::new());
        let symbol_table = SymbolTable::new();
        let scope_tree = ScopeTree::new(ScopeId::first());
        let string_interner = StringInterner::new();
        (type_table, symbol_table, scope_tree, string_interner)
    }

    #[test]
    fn test_type_checker_creation() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        assert!(!checker.has_errors());
        assert_eq!(checker.errors().len(), 0);
    }

    #[test]
    fn test_primitive_type_compatibility() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        // Same types should be identical
        let int_type = checker.type_table.borrow().int_type();
        assert_eq!(
            checker.check_compatibility(int_type, int_type),
            TypeCompatibility::Identical
        );

        // Int -> Float should be convertible
        let float_type = checker.type_table.borrow().float_type();
        assert_eq!(
            checker.check_compatibility(int_type, float_type),
            TypeCompatibility::Convertible
        );

        // Int -> String should be incompatible
        let string_type = checker.type_table.borrow().string_type();
        assert_eq!(
            checker.check_compatibility(int_type, string_type),
            TypeCompatibility::Incompatible
        );
    }

    #[test]
    fn test_dynamic_type_compatibility() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let dynamic_type = checker.type_table.borrow().dynamic_type();
        let int_type = checker.type_table.borrow().int_type();

        // Dynamic should be assignable to/from any type
        assert_eq!(
            checker.check_compatibility(int_type, dynamic_type),
            TypeCompatibility::Assignable
        );
        assert_eq!(
            checker.check_compatibility(dynamic_type, int_type),
            TypeCompatibility::Assignable
        );
    }

    #[test]
    fn test_optional_type_compatibility() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let int_type = checker.type_table.borrow().int_type();
        let optional_int = checker
            .type_table
            .borrow_mut()
            .create_optional_type(int_type);

        // Int should be assignable to Int?
        assert_eq!(
            checker.check_compatibility(int_type, optional_int),
            TypeCompatibility::Assignable
        );

        // Int? should NOT be assignable to Int (requires unwrapping)
        assert_eq!(
            checker.check_compatibility(optional_int, int_type),
            TypeCompatibility::Incompatible
        );
    }

    #[test]
    fn test_array_type_compatibility() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let int_type = checker.type_table.borrow().int_type();
        let string_type = checker.type_table.borrow().string_type();

        let int_array = checker.type_table.borrow_mut().create_array_type(int_type);
        let string_array = checker
            .type_table
            .borrow_mut()
            .create_array_type(string_type);

        // Same array types should be identical
        let int_array2 = checker.type_table.borrow_mut().create_array_type(int_type);
        assert_eq!(
            checker.check_compatibility(int_array, int_array2),
            TypeCompatibility::Identical
        );

        // Different array types should be incompatible
        assert_eq!(
            checker.check_compatibility(int_array, string_array),
            TypeCompatibility::Incompatible
        );
    }

    #[test]
    fn test_function_type_compatibility() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let int_type = checker.type_table.borrow().int_type();
        let string_type = checker.type_table.borrow().string_type();
        let bool_type = checker.type_table.borrow().bool_type();

        // (Int) -> String
        let func1 = checker
            .type_table
            .borrow_mut()
            .create_function_type(vec![int_type], string_type);

        // (Int) -> String (same)
        let func2 = checker
            .type_table
            .borrow_mut()
            .create_function_type(vec![int_type], string_type);

        // (Int) -> Bool (different return)
        let func3 = checker
            .type_table
            .borrow_mut()
            .create_function_type(vec![int_type], bool_type);

        assert_eq!(
            checker.check_compatibility(func1, func2),
            TypeCompatibility::Identical
        );
        assert_eq!(
            checker.check_compatibility(func1, func3),
            TypeCompatibility::Incompatible
        );
    }

    #[test]
    fn test_function_call_checking() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();
        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let int_type = checker.type_table.borrow().int_type();
        let string_type = checker.type_table.borrow().string_type();

        // Create function type: (Int, String) -> String
        let func_type = checker
            .type_table
            .borrow_mut()
            .create_function_type(vec![int_type, string_type], string_type);

        let location = SourceLocation::new(1, 1, 1, 0);

        // Valid call
        let result = checker.check_function_call(func_type, &[int_type, string_type], location);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), string_type);

        // Invalid call - wrong argument count
        let result = checker.check_function_call(func_type, &[int_type], location);
        assert!(result.is_err());

        // Invalid call - wrong argument type
        let result = checker.check_function_call(func_type, &[string_type, string_type], location);
        assert!(result.is_err());
    }

    #[test]
    fn test_subtype_caching() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let int_type = checker.type_table.borrow().int_type();
        let float_type = checker.type_table.borrow().float_type();

        // First call should compute and cache
        let result1 = checker.is_subtype(int_type, float_type);

        // Second call should use cache
        let result2 = checker.is_subtype(int_type, float_type);

        assert_eq!(result1, result2);
        assert!(checker.subtype_cache.contains_key(&(int_type, float_type)));
    }

    #[test]
    fn test_generic_type_validation() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let symbol_id = SymbolId::from_raw(1);
        let int_type = checker.type_table.borrow().int_type();
        let base_type = checker
            .type_table
            .borrow_mut()
            .create_class_type(symbol_id, vec![]);

        let location = SourceLocation::new(1, 1, 1, 0);

        // Valid instantiation (though simplified - real implementation would check constraints)
        let result = checker.validate_generic_instantiation(base_type, &[], location);
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_inference_basics() {
        let mut inference = TypeInference::new();

        // Create a type variable
        let type_var = inference.create_type_variable();
        assert_eq!(inference.stats.type_variables_created, 1);

        // Add an equality constraint
        let constraint = TypeConstraint {
            type_var,
            kind: ConstraintKind::Equality {
                target_type: TypeId::from_raw(42),
            },
            location: SourceLocation::new(1, 1, 1, 0),
            priority: ConstraintPriority::Default,
            is_soft: true,
        };

        inference.add_constraint(constraint);
        assert_eq!(inference.stats.constraints_generated, 1);

        // Solve constraints
        let result = inference.solve_constraints();
        assert!(result.is_ok());
        assert_eq!(inference.stats.constraints_solved, 1);

        // Check solution
        let solution = inference.get_solution(type_var);
        assert_eq!(solution, Some(TypeId::from_raw(42)));
    }

    #[test]
    fn test_error_accumulation() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        // Add some errors
        let error1 = TypeCheckError {
            kind: TypeErrorKind::TypeMismatch {
                expected: checker.type_table.borrow().int_type(),
                actual: checker.type_table.borrow().string_type(),
            },
            location: SourceLocation::new(1, 1, 1, 0),
            context: "Test error 1".to_string(),
            suggestion: None,
        };

        let error2 = TypeCheckError {
            kind: TypeErrorKind::UndefinedType {
                name: string_interner.intern("UnknownType"),
            },
            location: SourceLocation::new(2, 1, 1, 10),
            context: "Test error 2".to_string(),
            suggestion: Some("Define the type".to_string()),
        };

        checker.add_error(error1);
        checker.add_error(error2);

        assert!(checker.has_errors());
        assert_eq!(checker.errors().len(), 2);

        checker.clear_errors();
        assert!(!checker.has_errors());
        assert_eq!(checker.errors().len(), 0);
    }

    #[test]
    fn test_type_checker_stats() {
        let (mut type_table, symbol_table, scope_tree, string_interner) = create_test_setup();

        let mut checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        // Perform some type checking operations
        let int_type = checker.type_table.borrow().int_type();
        let float_type = checker.type_table.borrow().float_type();

        checker.check_compatibility(int_type, float_type);
        checker.check_compatibility(float_type, int_type);

        let stats = checker.stats();
        assert!(stats.types_checked > 0);
        assert_eq!(stats.errors_found, 0);
    }

    #[test]
    fn test_type_formatting() {
        let (mut type_table, _, _, string_interner) = create_test_setup();

        let int_type = type_table.borrow().int_type();
        let formatted = format_type_for_error(int_type, &type_table, &string_interner);
        assert_eq!(formatted, "Int");

        let array_type = type_table.borrow_mut().create_array_type(int_type);
        let formatted = format_type_for_error(array_type, &type_table, &string_interner);
        assert_eq!(formatted, "Array<Int>");
    }

    #[test]
    fn test_constraint_set_creation() {
        let cs = ConstraintSet::new();
        assert_eq!(cs.stats().total_constraints, 0);
        assert!(!cs.has_dependency_cycle());
    }

    #[test]
    fn test_constraint_addition() {
        let mut cs = ConstraintSet::new();
        let constraint = TypeConstraint {
            type_var: TypeId::from_raw(1),
            kind: ConstraintKind::Comparable,
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
            is_soft: false,
        };

        cs.add_constraint(constraint.clone());
        assert_eq!(cs.stats().total_constraints, 1);
        assert_eq!(cs.constraints_for(TypeId::from_raw(1)).len(), 1);
    }

    #[test]
    fn test_constraint_removal() {
        let mut cs = ConstraintSet::new();
        let type_var = TypeId::from_raw(1);
        let constraint = TypeConstraint {
            type_var,
            kind: ConstraintKind::Comparable,
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
            is_soft: false,
        };

        cs.add_constraint(constraint);
        assert_eq!(cs.stats().total_constraints, 1);

        let removed = cs.remove_constraints_for(type_var);
        assert_eq!(removed, 1);
        assert_eq!(cs.stats().total_constraints, 0);
    }

    #[test]
    fn test_dependency_tracking() {
        let mut cs = ConstraintSet::new();
        let type_var = TypeId::from_raw(1);
        let target_type = TypeId::from_raw(2);

        let constraint = TypeConstraint {
            type_var,
            kind: ConstraintKind::Equality { target_type },
            location: SourceLocation::default(),
            priority: ConstraintPriority::Explicit,
            is_soft: false,
        };

        cs.add_constraint(constraint);
        assert!(cs.dependencies_of(type_var).contains(&target_type));
    }

    #[test]
    fn test_constraint_validator() {
        let mut validator = ConstraintValidator::new();
        let type_table = RefCell::new(TypeTable::new());

        let result = validator.validate_constraint(
            TypeId::from_raw(1),
            &ConstraintKind::Comparable,
            &type_table,
        );

        assert!(matches!(result, ConstraintValidation::Satisfied));
        assert_eq!(validator.stats().validations_performed, 1);
    }

    #[test]
    fn test_class_hierarchy() {
        let type_table = RefCell::new(TypeTable::new());
        let mut symbol_table = SymbolTable::new();
        let interner = StringInterner::new();

        // Create class symbols
        let object_sym = symbol_table.create_class(interner.intern("Object"));
        let animal_sym = symbol_table.create_class(interner.intern("Animal"));
        let dog_sym = symbol_table.create_class(interner.intern("Dog"));
        let cat_sym = symbol_table.create_class(interner.intern("Cat"));

        // Create class types
        let object_type = type_table
            .borrow_mut()
            .create_class_type(object_sym, vec![]);
        let animal_type = type_table
            .borrow_mut()
            .create_class_type(animal_sym, vec![]);
        let dog_type = type_table.borrow_mut().create_class_type(dog_sym, vec![]);
        let cat_type = type_table.borrow_mut().create_class_type(cat_sym, vec![]);

        // Build hierarchy: Object <- Animal <- Dog
        //                                   <- Cat
        let mut builder = ClassHierarchyBuilder::new();
        builder.register_class(object_sym, None, vec![]);
        builder.register_class(animal_sym, Some(object_type), vec![]);
        builder.register_class(dog_sym, Some(animal_type), vec![]);
        builder.register_class(cat_sym, Some(animal_type), vec![]);

        builder.compute_transitive_closure(&type_table);
        builder.finalize(&mut symbol_table);

        // Create type checker
        let scope_tree = ScopeTree::new(ScopeId::first());
        let string_interner = StringInterner::new();
        let type_checker =
            TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        // Test subtype relationships
        assert!(type_checker.is_class_subtype_of(dog_type, animal_type));
        assert!(type_checker.is_class_subtype_of(dog_type, object_type));
        assert!(!type_checker.is_class_subtype_of(dog_type, cat_type));

        // Test common supertype
        let common = type_checker.find_common_class_supertype(dog_type, cat_type);
        assert_eq!(common, Some(animal_type));
    }
}
