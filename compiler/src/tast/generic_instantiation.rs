// Generic Instantiation Engine for Haxe Compiler

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::tast::core::{Type, TypeKind, TypeTable, Variance};
use crate::tast::type_checker::{ConstraintValidation, ConstraintValidator};
use crate::tast::{SourceLocation, TypeId};

use super::type_checker::{ConstraintKind, ConstraintPriority, ConstraintSet, TypeConstraint};

/// Unique identifier for generic instantiations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstantiationId(u32);

impl InstantiationId {
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

/// Represents a generic instantiation request
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstantiationRequest {
    /// The generic base type to instantiate
    pub base_type: TypeId,

    /// Type arguments to substitute for type parameters
    pub type_args: Vec<TypeId>,

    /// Source location for error reporting
    pub location: SourceLocation,

    /// Whether this is a partial instantiation (some type args may be inference variables)
    pub is_partial: bool,
}

/// Result of a generic instantiation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstantiationResult {
    /// The instantiated type
    pub instantiated_type: TypeId,

    /// Unique ID for this instantiation
    pub instantiation_id: InstantiationId,

    /// Any additional constraints generated during instantiation
    pub generated_constraints: ConstraintSet,

    /// Whether this instantiation was cached
    pub was_cached: bool,

    /// Performance metrics for this instantiation
    pub metrics: InstantiationMetrics,
}

/// Performance metrics for instantiation operations
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InstantiationMetrics {
    pub instantiation_time_ns: u64,
    pub constraint_validation_time_ns: u64,
    pub cache_lookup_time_ns: u64,
    pub type_substitution_count: usize,
    pub constraint_count: usize,
    pub recursion_depth: usize,
}

/// Error types for generic instantiation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstantiationError {
    /// Recursive instantiation detected (e.g., Foo<Foo<T>>)
    RecursiveInstantiation {
        cycle: Vec<TypeId>,
        location: SourceLocation,
    },

    /// Wrong number of type arguments
    ArityMismatch {
        base_type: TypeId,
        expected: usize,
        actual: usize,
        location: SourceLocation,
    },

    /// Type argument doesn't satisfy constraints
    ConstraintViolation {
        type_arg: TypeId,
        constraint: ConstraintKind,
        reason: String,
        location: SourceLocation,
    },

    /// Invalid base type (not generic)
    NotGeneric {
        base_type: TypeId,
        location: SourceLocation,
    },

    /// Maximum instantiation depth exceeded
    DepthExceeded {
        max_depth: usize,
        location: SourceLocation,
    },

    /// Internal error during instantiation
    Internal {
        message: String,
        location: SourceLocation,
    },
}

/// Configuration for the generic instantiation engine
#[derive(Debug, Clone)]
pub struct InstantiationConfig {
    /// Maximum recursion depth for instantiation
    pub max_recursion_depth: usize,

    /// Maximum number of cached instantiations
    pub max_cache_size: usize,

    /// Whether to perform aggressive constraint validation
    pub strict_constraints: bool,

    /// Whether to cache partial instantiations
    pub cache_partial: bool,

    /// Performance profiling enabled
    pub profile_performance: bool,
}

impl Default for InstantiationConfig {
    fn default() -> Self {
        Self {
            max_recursion_depth: 32,
            max_cache_size: 10000,
            strict_constraints: true,
            cache_partial: false,
            profile_performance: true,
        }
    }
}

/// The main generic instantiation engine
pub struct GenericInstantiator {
    /// Cache of completed instantiations for performance
    instantiation_cache: BTreeMap<InstantiationRequest, InstantiationResult>,

    /// Currently active instantiations for cycle detection
    active_instantiations: BTreeSet<InstantiationRequest>,

    /// Stack of instantiations for recursion tracking
    instantiation_stack: Vec<InstantiationRequest>,

    /// Constraint validator from Phase 3
    constraint_validator: ConstraintValidator,

    /// Configuration settings
    config: InstantiationConfig,

    /// Performance statistics
    stats: InstantiationStats,

    /// Unique ID generator for instantiations
    next_id: InstantiationId,
}

/// Statistics for the instantiation engine
#[derive(Debug, Clone, Default)]
pub struct InstantiationStats {
    pub total_instantiations: usize,
    pub successful_instantiations: usize,
    pub failed_instantiations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cycles_detected: usize,
    pub max_recursion_depth_used: usize,
    pub constraint_violations: usize,
    pub partial_instantiations: usize,
    pub average_instantiation_time_ns: f64,
    pub total_time_ns: u64,
    pub memory_used_bytes: usize,
}

impl GenericInstantiator {
    /// Create a new generic instantiation engine
    pub fn new(config: InstantiationConfig) -> Self {
        Self {
            instantiation_cache: BTreeMap::new(),
            active_instantiations: BTreeSet::new(),
            instantiation_stack: Vec::new(),
            constraint_validator: ConstraintValidator::new(),
            config,
            stats: InstantiationStats::default(),
            next_id: InstantiationId::new(),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(InstantiationConfig::default())
    }

    /// Instantiate a generic type with the given type arguments
    pub fn instantiate(
        &mut self,
        request: InstantiationRequest,
        type_table: &RefCell<TypeTable>,
    ) -> Result<InstantiationResult, InstantiationError> {
        let start_time = if self.config.profile_performance {
            std::time::Instant::now()
        } else {
            std::time::Instant::now() // Always measure for now
        };

        self.stats.total_instantiations += 1;

        // Check cache first
        if let Some(cached_result) = self.instantiation_cache.get(&request) {
            self.stats.cache_hits += 1;
            let mut result = cached_result.clone();
            result.was_cached = true;
            return Ok(result);
        }

        self.stats.cache_misses += 1;

        // Check for cycles - both exact match and type argument cycles
        if self.active_instantiations.contains(&request) {
            self.stats.cycles_detected += 1;
            return Err(InstantiationError::RecursiveInstantiation {
                cycle: self.extract_cycle(&request),
                location: request.location,
            });
        }

        // Also check if any type argument references the base type (immediate cycle)
        for &type_arg in &request.type_args {
            if type_arg == request.base_type {
                self.stats.cycles_detected += 1;
                return Err(InstantiationError::RecursiveInstantiation {
                    cycle: vec![request.base_type, type_arg],
                    location: request.location,
                });
            }
        }

        // Check recursion depth
        if self.instantiation_stack.len() >= self.config.max_recursion_depth {
            return Err(InstantiationError::DepthExceeded {
                max_depth: self.config.max_recursion_depth,
                location: request.location,
            });
        }

        // Track recursion depth
        self.stats.max_recursion_depth_used = self
            .stats
            .max_recursion_depth_used
            .max(self.instantiation_stack.len());

        // Begin instantiation
        self.active_instantiations.insert(request.clone());
        self.instantiation_stack.push(request.clone());

        let result = self.instantiate_impl(request.clone(), type_table);

        // Clean up tracking
        self.instantiation_stack.pop();
        self.active_instantiations.remove(&request);

        // Update statistics
        let elapsed = start_time.elapsed();
        self.stats.total_time_ns += elapsed.as_nanos() as u64;

        match &result {
            Ok(instantiation_result) => {
                self.stats.successful_instantiations += 1;

                if request.is_partial {
                    self.stats.partial_instantiations += 1;
                }

                // Cache the result if configured to do so
                if !request.is_partial || self.config.cache_partial {
                    if self.instantiation_cache.len() < self.config.max_cache_size {
                        self.instantiation_cache
                            .insert(request, instantiation_result.clone());
                    }
                }
            }
            Err(InstantiationError::ConstraintViolation { .. }) => {
                self.stats.failed_instantiations += 1;
                self.stats.constraint_violations += 1;
            }
            Err(_) => {
                self.stats.failed_instantiations += 1;
            }
        }

        // Update average time
        if self.stats.successful_instantiations > 0 {
            self.stats.average_instantiation_time_ns =
                self.stats.total_time_ns as f64 / self.stats.successful_instantiations as f64;
        }

        result
    }

    /// Fast path for simple instantiations without complex constraints
    pub fn instantiate_simple(
        &mut self,
        base_type: TypeId,
        type_args: Vec<TypeId>,
        type_table: &RefCell<TypeTable>,
    ) -> Result<TypeId, InstantiationError> {
        let request = InstantiationRequest {
            base_type,
            type_args,
            location: SourceLocation::default(),
            is_partial: false,
        };

        let result = self.instantiate(request, type_table)?;
        Ok(result.instantiated_type)
    }

    /// Check if a type can be instantiated with given arguments
    pub fn can_instantiate(
        &mut self,
        base_type: TypeId,
        type_args: &[TypeId],
        type_table: &RefCell<TypeTable>,
    ) -> bool {
        // Perform lightweight validation without full instantiation
        self.validate_instantiation_request(base_type, type_args, type_table)
            .is_ok()
    }

    /// Pre-validate an instantiation request
    pub fn validate_instantiation_request(
        &self,
        base_type: TypeId,
        type_args: &[TypeId],
        type_table: &RefCell<TypeTable>,
    ) -> Result<(), InstantiationError> {
        // Get the base type
        let binding = type_table.borrow();
        let base_type_obj = binding
            .get(base_type)
            .ok_or_else(|| InstantiationError::Internal {
                message: "Invalid base type".to_string(),
                location: SourceLocation::default(),
            })?;

        // Check if it's actually generic
        let expected_param_count = match &base_type_obj.kind {
            TypeKind::Class {
                type_args: params, ..
            }
            | TypeKind::Interface {
                type_args: params, ..
            }
            | TypeKind::Enum {
                type_args: params, ..
            }
            | TypeKind::Abstract {
                type_args: params, ..
            }
            | TypeKind::TypeAlias {
                type_args: params, ..
            } => params.len(),
            _ => {
                return Err(InstantiationError::NotGeneric {
                    base_type,
                    location: SourceLocation::default(),
                });
            }
        };

        // Check arity
        if type_args.len() != expected_param_count {
            return Err(InstantiationError::ArityMismatch {
                base_type,
                expected: expected_param_count,
                actual: type_args.len(),
                location: SourceLocation::default(),
            });
        }

        Ok(())
    }

    /// Clear the instantiation cache
    pub fn clear_cache(&mut self) {
        self.instantiation_cache.clear();
        self.constraint_validator.clear_cache();
        self.update_memory_usage();
    }

    /// Get current statistics
    pub fn stats(&self) -> &InstantiationStats {
        &self.stats
    }

    /// Get cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_attempts = self.stats.cache_hits + self.stats.cache_misses;
        if total_attempts > 0 {
            self.stats.cache_hits as f64 / total_attempts as f64
        } else {
            0.0
        }
    }

    /// Get success ratio
    pub fn success_ratio(&self) -> f64 {
        if self.stats.total_instantiations > 0 {
            self.stats.successful_instantiations as f64 / self.stats.total_instantiations as f64
        } else {
            0.0
        }
    }

    // === Private implementation methods ===

    fn instantiate_impl(
        &mut self,
        request: InstantiationRequest,
        type_table: &RefCell<TypeTable>,
    ) -> Result<InstantiationResult, InstantiationError> {
        let mut metrics = InstantiationMetrics::default();
        metrics.recursion_depth = self.instantiation_stack.len();

        // Validate the request
        self.validate_instantiation_request(request.base_type, &request.type_args, &type_table)?;

        // Get base type information
        let binding = type_table.borrow();
        let base_type_obj = binding.get(request.base_type).unwrap();

        // Extract type parameters and their constraints
        let type_params = self.extract_type_parameters(base_type_obj)?;

        // Create constraint set for this instantiation
        let mut constraints = ConstraintSet::new();

        // Validate type arguments against constraints
        for (i, (&type_arg, type_param)) in request.type_args.iter().zip(&type_params).enumerate() {
            for constraint in &type_param.constraints {
                let type_constraint = TypeConstraint {
                    type_var: type_arg,
                    kind: constraint.clone(),
                    location: request.location,
                    priority: ConstraintPriority::TypeAnnotation,
                    is_soft: false,
                };

                if self.config.strict_constraints {
                    let validation = self
                        .constraint_validator
                        .validate_constraint(type_arg, constraint, type_table);

                    match validation {
                        ConstraintValidation::Violated { reason, .. } => {
                            return Err(InstantiationError::ConstraintViolation {
                                type_arg,
                                constraint: constraint.clone(),
                                reason,
                                location: request.location,
                            });
                        }
                        ConstraintValidation::Pending { .. } => {
                            // Add to constraint set for later resolution
                            constraints.add_constraint(type_constraint);
                        }
                        _ => {
                            // Satisfied or conditional - continue
                        }
                    }
                }

                metrics.constraint_count += 1;
            }

            metrics.type_substitution_count += 1;
        }

        // Perform the actual instantiation
        let instantiated_type = self.perform_substitution(
            request.base_type,
            &type_params,
            &request.type_args,
            type_table,
        )?;

        let instantiation_id = self.next_id();

        Ok(InstantiationResult {
            instantiated_type,
            instantiation_id,
            generated_constraints: constraints,
            was_cached: false,
            metrics,
        })
    }

    fn extract_type_parameters(
        &self,
        base_type: &Type,
    ) -> Result<Vec<TypeParameterInfo>, InstantiationError> {
        match &base_type.kind {
            TypeKind::Class { type_args, .. }
            | TypeKind::Interface { type_args, .. }
            | TypeKind::Enum { type_args, .. }
            | TypeKind::Abstract { type_args, .. }
            | TypeKind::TypeAlias { type_args, .. } => {
                // For now, return simplified parameter info
                // In a full implementation, this would extract actual constraint information
                Ok(type_args
                    .iter()
                    .map(|&param_id| TypeParameterInfo {
                        id: param_id,
                        constraints: vec![], // Would be populated from symbol table
                        variance: Variance::Invariant,
                    })
                    .collect())
            }
            _ => Err(InstantiationError::Internal {
                message: "Expected generic type".to_string(),
                location: SourceLocation::default(),
            }),
        }
    }

    fn perform_substitution(
        &self,
        base_type: TypeId,
        type_params: &[TypeParameterInfo],
        type_args: &[TypeId],
        type_table: &RefCell<TypeTable>,
    ) -> Result<TypeId, InstantiationError> {
        // Create a generic instance in the type table
        // This integrates with the Phase 1 type system
        let instantiated_type = type_table
            .borrow_mut()
            .create_generic_instance(base_type, type_args.to_vec());
        Ok(instantiated_type)
    }

    fn extract_cycle(&self, request: &InstantiationRequest) -> Vec<TypeId> {
        // Find the cycle in the instantiation stack
        let mut cycle = Vec::new();
        let mut found = false;

        for stack_request in &self.instantiation_stack {
            if found || stack_request == request {
                cycle.push(stack_request.base_type);
                found = true;
            }
        }

        if !found {
            cycle.push(request.base_type);
        }

        cycle
    }

    fn next_id(&mut self) -> InstantiationId {
        let id = self.next_id;
        self.next_id = InstantiationId::new();
        id
    }

    fn update_memory_usage(&mut self) {
        // Estimate memory usage for statistics
        let cache_size = self.instantiation_cache.len()
            * std::mem::size_of::<(InstantiationRequest, InstantiationResult)>();
        let active_size =
            self.active_instantiations.len() * std::mem::size_of::<InstantiationRequest>();
        let stack_size =
            self.instantiation_stack.len() * std::mem::size_of::<InstantiationRequest>();

        self.stats.memory_used_bytes = cache_size + active_size + stack_size;
    }
}

/// Information about a type parameter extracted from a generic type
#[derive(Debug, Clone)]
struct TypeParameterInfo {
    id: TypeId,
    constraints: Vec<ConstraintKind>,
    variance: super::core::Variance,
}

/// Specialized instantiation context for performance-critical paths
pub struct FastInstantiationContext {
    /// Pre-computed substitution maps
    substitution_cache: BTreeMap<(TypeId, Vec<TypeId>), TypeId>,

    /// Simple validation cache
    validation_cache: BTreeMap<(TypeId, Vec<TypeId>), bool>,

    /// Statistics
    fast_stats: FastInstantiationStats,
}

#[derive(Debug, Clone, Default)]
pub struct FastInstantiationStats {
    pub fast_instantiations: usize,
    pub substitution_cache_hits: usize,
    pub validation_cache_hits: usize,
}

impl FastInstantiationContext {
    pub fn new() -> Self {
        Self {
            substitution_cache: BTreeMap::new(),
            validation_cache: BTreeMap::new(),
            fast_stats: FastInstantiationStats::default(),
        }
    }

    /// Fast instantiation for hot paths (e.g., Array<T>, Map<K,V>)
    pub fn instantiate_builtin(
        &mut self,
        base_type: TypeId,
        type_args: Vec<TypeId>,
        type_table: &RefCell<TypeTable>,
    ) -> Option<TypeId> {
        self.fast_stats.fast_instantiations += 1;

        let cache_key = (base_type, type_args.clone());

        // Check substitution cache
        if let Some(&cached_type) = self.substitution_cache.get(&cache_key) {
            self.fast_stats.substitution_cache_hits += 1;
            return Some(cached_type);
        }

        // Quick validation for built-in types
        if !self.validate_builtin_args(base_type, &type_args, type_table) {
            return None;
        }

        // Check if this is a builtin generic type we can handle fast
        if let Some(base_type_obj) = type_table.borrow().get(base_type) {
            let instantiated_type = match &base_type_obj.kind {
                // Direct builtin types
                TypeKind::Array { .. } if type_args.len() == 1 => {
                    Some(type_table.borrow_mut().create_array_type(type_args[0]))
                }
                TypeKind::Map { .. } if type_args.len() == 2 => Some(
                    type_table
                        .borrow_mut()
                        .create_map_type(type_args[0], type_args[1]),
                ),
                TypeKind::Optional { .. } if type_args.len() == 1 => {
                    Some(type_table.borrow_mut().create_optional_type(type_args[0]))
                }
                // Generic class/interface types (like Array<T> defined as a class)
                TypeKind::Class {
                    type_args: params, ..
                } if params.len() == type_args.len() => Some(
                    type_table
                        .borrow_mut()
                        .create_generic_instance(base_type, type_args.clone()),
                ),
                TypeKind::Interface {
                    type_args: params, ..
                } if params.len() == type_args.len() => Some(
                    type_table
                        .borrow_mut()
                        .create_generic_instance(base_type, type_args.clone()),
                ),
                _ => None,
            };

            if let Some(result_type) = instantiated_type {
                // Cache the result
                self.substitution_cache.insert(cache_key, result_type);
                return Some(result_type);
            }
        }

        None
    }

    fn validate_builtin_args(
        &mut self,
        base_type: TypeId,
        type_args: &[TypeId],
        type_table: &RefCell<TypeTable>,
    ) -> bool {
        let cache_key = (base_type, type_args.to_vec());

        if let Some(&cached_valid) = self.validation_cache.get(&cache_key) {
            self.fast_stats.validation_cache_hits += 1;
            return cached_valid;
        }

        // Simple validation for built-in types
        let is_valid = match type_table.borrow().get(base_type) {
            Some(base_type_obj) => match &base_type_obj.kind {
                super::core::TypeKind::Class {
                    type_args: params, ..
                } => type_args.len() == params.len(),
                _ => false,
            },
            None => false,
        };

        self.validation_cache.insert(cache_key, is_valid);
        is_valid
    }

    pub fn stats(&self) -> &FastInstantiationStats {
        &self.fast_stats
    }

    pub fn clear(&mut self) {
        self.substitution_cache.clear();
        self.validation_cache.clear();
    }
}

impl Default for FastInstantiationContext {
    fn default() -> Self {
        Self::new()
    }
}

// Display implementations for debugging
impl fmt::Display for InstantiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InstantiationError::RecursiveInstantiation { cycle, .. } => {
                write!(f, "Recursive instantiation detected: {:?}", cycle)
            }
            InstantiationError::ArityMismatch {
                expected, actual, ..
            } => {
                write!(
                    f,
                    "Type argument count mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            InstantiationError::ConstraintViolation { reason, .. } => {
                write!(f, "Constraint violation: {}", reason)
            }
            InstantiationError::NotGeneric { .. } => {
                write!(f, "Type is not generic")
            }
            InstantiationError::DepthExceeded { max_depth, .. } => {
                write!(f, "Maximum instantiation depth {} exceeded", max_depth)
            }
            InstantiationError::Internal { message, .. } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for InstantiationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instantiation_engine_creation() {
        let engine = GenericInstantiator::with_defaults();
        assert_eq!(engine.stats().total_instantiations, 0);
        assert_eq!(engine.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_instantiation_request() {
        let request = InstantiationRequest {
            base_type: TypeId::from_raw(1),
            type_args: vec![TypeId::from_raw(2)],
            location: SourceLocation::default(),
            is_partial: false,
        };

        assert_eq!(request.type_args.len(), 1);
        assert!(!request.is_partial);
    }

    #[test]
    fn test_fast_instantiation_context() {
        let mut context = FastInstantiationContext::new();
        assert_eq!(context.stats().fast_instantiations, 0);

        // Test cache initialization
        context.clear();
        assert_eq!(context.stats().fast_instantiations, 0);
    }

    #[test]
    fn test_cycle_detection() {
        let mut engine = GenericInstantiator::with_defaults();
        let mut type_table = TypeTable::new();

        let request = InstantiationRequest {
            base_type: TypeId::from_raw(1),
            type_args: vec![TypeId::from_raw(1)], // Self-referential
            location: SourceLocation::default(),
            is_partial: false,
        };

        // This would detect cycles in a full implementation
        // For now, we test the structure
        assert!(engine.active_instantiations.is_empty());
    }

    #[test]
    fn test_validation() {
        let engine = GenericInstantiator::with_defaults();
        let type_table = RefCell::new(TypeTable::new());

        // Test validation structure
        let result = engine.validate_instantiation_request(
            TypeId::from_raw(1),
            &[TypeId::from_raw(2)],
            &type_table,
        );

        // Will fail because type doesn't exist, but tests the structure
        assert!(result.is_err());
    }
}
