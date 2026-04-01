//! Global Lifetime Constraints System
//!
//! Handles inter-procedural lifetime analysis across the entire call graph.
//! Manages cross-function lifetime flows, recursion constraints, and
//! virtual method lifetime polymorphism.
//!
//! This module provides comprehensive analysis of lifetime relationships that
//! span function boundaries, ensuring memory safety across the entire program.

use crate::semantic_graph::analysis::lifetime_analyzer::{
    LifetimeAnalysisError, LifetimeConstraint,
};
use crate::semantic_graph::{CallGraph, CallSite, CallTarget};
use crate::tast::collections::{new_id_map, new_id_set, IdMap, IdSet};
use crate::tast::{CallSiteId, LifetimeId, SourceLocation, SymbolId, TypeId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Main structure containing all global lifetime constraints and analysis results
#[derive(Debug, Clone)]
pub struct GlobalLifetimeConstraints {
    /// Inter-function lifetime relationships at call sites
    pub call_site_constraints: IdMap<CallSiteId, CallSiteLifetimeConstraint>,

    /// Function signature lifetime mappings
    pub function_signatures: IdMap<SymbolId, FunctionLifetimeSignature>,

    /// Cross-function lifetime flows
    pub lifetime_flows: Vec<CrossFunctionFlow>,

    /// Recursion-specific constraint handling
    pub recursive_constraint_groups: Vec<RecursiveConstraintGroup>,

    /// Virtual method lifetime polymorphism
    pub virtual_method_constraints: IdMap<SymbolId, VirtualLifetimeConstraints>,

    /// Global constraint graph for unified solving
    pub constraint_graph: LifetimeConstraintGraph,

    /// Validation results and violations
    pub violations: Vec<GlobalLifetimeViolation>,

    /// Analysis statistics
    pub stats: GlobalAnalysisStats,
}

/// Lifetime constraints for a specific call site
#[derive(Debug, Clone)]
pub struct CallSiteLifetimeConstraint {
    /// Call site identifier
    pub call_site_id: CallSiteId,
    /// Function making the call
    pub caller_function: SymbolId,
    /// Function being called
    pub callee_function: SymbolId,

    /// Argument lifetime flows: caller lifetime -> callee parameter lifetime
    pub argument_flows: Vec<LifetimeFlow>,

    /// Return lifetime flows: callee return -> caller result lifetime
    pub return_flows: Vec<LifetimeFlow>,

    /// Borrowing constraints (arguments borrowed by callee)
    pub borrow_constraints: Vec<BorrowConstraint>,

    /// Source location for error reporting
    pub source_location: SourceLocation,
}

/// Lifetime signature extracted from a function
#[derive(Debug, Clone)]
pub struct FunctionLifetimeSignature {
    /// Function identifier
    pub function_id: SymbolId,
    /// Lifetime parameters in function signature
    pub parameter_lifetimes: Vec<LifetimeId>,
    /// Return value lifetime (if any)
    pub return_lifetime: Option<LifetimeId>,
    /// Generic lifetime parameters
    pub generic_lifetime_params: Vec<LifetimeId>,
    /// Lifetime bounds and constraints within the function
    pub lifetime_bounds: Vec<LifetimeBound>,
    /// Source location of function definition
    pub source_location: SourceLocation,
}

/// Cross-function lifetime flow relationship
#[derive(Debug, Clone)]
pub struct CrossFunctionFlow {
    /// Source function
    pub from_function: SymbolId,
    /// Target function
    pub to_function: SymbolId,
    /// Type of lifetime flow
    pub flow_type: LifetimeFlowType,
    /// Source lifetime
    pub source_lifetime: LifetimeId,
    /// Target lifetime
    pub target_lifetime: LifetimeId,
    /// Call site where this flow occurs
    pub call_site: CallSiteId,
}

/// Individual lifetime flow between two lifetimes
#[derive(Debug, Clone)]
pub struct LifetimeFlow {
    /// Source lifetime
    pub from: LifetimeId,
    /// Target lifetime
    pub to: LifetimeId,
    /// Type of flow relationship
    pub flow_kind: LifetimeFlowKind,
}

/// Types of lifetime flows
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifetimeFlowType {
    /// Argument passed to parameter
    ArgumentToParameter,
    /// Return value flows to caller
    ReturnToResult,
    /// Value escapes to global scope
    EscapeToGlobal,
    /// Borrowed reference captured
    BorrowCapture,
}

/// Kinds of lifetime flow relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifetimeFlowKind {
    /// Source lifetime must outlive target
    Outlives,
    /// Source and target have same lifetime
    Equal,
    /// Source is borrowed by target
    Borrow,
    /// Source is moved to target
    Move,
}

/// Borrowing constraint for function parameters
#[derive(Debug, Clone)]
pub struct BorrowConstraint {
    /// Parameter being borrowed
    pub parameter_lifetime: LifetimeId,
    /// Lifetime of the borrow
    pub borrow_lifetime: LifetimeId,
    /// Type of borrow (mutable/immutable)
    pub borrow_kind: BorrowKind,
    /// Source location
    pub source_location: SourceLocation,
}

/// Kind of borrowing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind {
    /// Immutable borrow
    Immutable,
    /// Mutable borrow
    Mutable,
}

/// Lifetime bound relationship
#[derive(Debug, Clone)]
pub struct LifetimeBound {
    /// Constrained lifetime
    pub lifetime: LifetimeId,
    /// Bounding lifetime
    pub bound: LifetimeId,
    /// Type of bound
    pub bound_kind: BoundKind,
}

/// Types of lifetime bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundKind {
    /// Lifetime must outlive bound
    Outlives,
    /// Lifetime equals bound
    Equal,
}

/// Constraints for recursive function groups
#[derive(Debug, Clone)]
pub struct RecursiveConstraintGroup {
    /// Functions in the recursive group (SCC)
    pub functions: Vec<SymbolId>,
    /// Fixed-point constraints for recursion
    pub fixed_point_constraints: Vec<FixedPointConstraint>,
    /// Convergence criteria for solving
    pub convergence_criteria: ConvergenceCriteria,
}

/// Fixed-point constraint for recursive analysis
#[derive(Debug, Clone)]
pub struct FixedPointConstraint {
    /// Constraint identifier
    pub id: u32,
    /// Constraint equation
    pub equation: ConstraintEquation,
    /// Iteration stability threshold
    pub stability_threshold: f64,
}

/// Constraint equation for fixed-point solving
#[derive(Debug, Clone)]
pub struct ConstraintEquation {
    /// Left-hand side lifetimes
    pub lhs: Vec<LifetimeId>,
    /// Right-hand side lifetimes
    pub rhs: Vec<LifetimeId>,
    /// Equation operator
    pub operator: ConstraintOperator,
}

/// Constraint operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintOperator {
    /// Outlives relationship
    Outlives,
    /// Equality relationship
    Equal,
    /// Union of lifetimes
    Union,
    /// Intersection of lifetimes
    Intersection,
}

/// Convergence criteria for fixed-point iteration
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Stability threshold
    pub stability_threshold: f64,
    /// Early termination conditions
    pub early_termination: bool,
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            stability_threshold: 0.001,
            early_termination: true,
        }
    }
}

/// Virtual method lifetime constraints
#[derive(Debug, Clone)]
pub struct VirtualLifetimeConstraints {
    /// Method name
    pub method_name: String,
    /// Possible implementations
    pub implementations: Vec<VirtualImplementationConstraint>,
    /// Unified constraint for all implementations
    pub unified_constraint: Option<CallSiteLifetimeConstraint>,
}

/// Constraint for a specific virtual method implementation
#[derive(Debug, Clone)]
pub struct VirtualImplementationConstraint {
    /// Implementation function
    pub implementation: SymbolId,
    /// Specific constraints for this implementation
    pub constraint: CallSiteLifetimeConstraint,
    /// Receiver type for this implementation
    pub receiver_type: TypeId,
}

/// Global constraint graph for unified solving
#[derive(Debug, Clone)]
pub struct LifetimeConstraintGraph {
    /// Constraint nodes (lifetime variables)
    pub nodes: IdMap<LifetimeId, ConstraintNode>,
    /// Constraint edges (relationships)
    pub edges: Vec<ConstraintEdge>,
    /// Strongly connected components
    pub sccs: Vec<Vec<LifetimeId>>,
}

/// Node in the constraint graph
#[derive(Debug, Clone)]
pub struct ConstraintNode {
    /// Lifetime identifier
    pub lifetime: LifetimeId,
    /// Node metadata
    pub metadata: NodeMetadata,
}

/// Metadata for constraint nodes
#[derive(Debug, Clone, Default)]
pub struct NodeMetadata {
    /// Source function
    pub source_function: Option<SymbolId>,
    /// Node type
    pub node_type: NodeType,
    /// Source location
    pub source_location: SourceLocation,
}

/// Types of constraint nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// Function parameter
    Parameter,
    /// Function return value
    Return,
    /// Local variable
    Local,
    /// Global/static lifetime
    Global,
}

impl Default for NodeType {
    fn default() -> Self {
        Self::Local
    }
}

/// Edge in the constraint graph
#[derive(Debug, Clone)]
pub struct ConstraintEdge {
    /// Source lifetime
    pub from: LifetimeId,
    /// Target lifetime
    pub to: LifetimeId,
    /// Edge type
    pub edge_type: EdgeType,
    /// Call site causing this edge (if any)
    pub call_site: Option<CallSiteId>,
}

/// Types of constraint edges
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Outlives constraint
    Outlives,
    /// Equality constraint
    Equal,
    /// Borrow constraint
    Borrow,
}

/// Global lifetime violations
#[derive(Debug, Clone)]
pub enum GlobalLifetimeViolation {
    /// Use after free across function boundaries
    CrossFunctionUseAfterFree {
        caller: SymbolId,
        callee: SymbolId,
        call_site: CallSiteId,
        violated_lifetime: LifetimeId,
        source_location: SourceLocation,
    },

    /// Invalid borrow across function call
    InvalidCrossFunctionBorrow {
        borrower: SymbolId,
        borrowed_from: SymbolId,
        call_site: CallSiteId,
        borrow_lifetime: LifetimeId,
        source_location: SourceLocation,
    },

    /// Recursion causes infinite lifetime extension
    RecursiveLifetimeExtension {
        recursive_group: Vec<SymbolId>,
        problematic_lifetime: LifetimeId,
        source_location: SourceLocation,
    },

    /// Virtual method lifetime mismatch
    VirtualMethodLifetimeMismatch {
        method_name: String,
        implementations: Vec<SymbolId>,
        conflicting_lifetimes: Vec<LifetimeId>,
        source_location: SourceLocation,
    },
}

/// Statistics about global analysis
#[derive(Debug, Clone, Default)]
pub struct GlobalAnalysisStats {
    /// Number of functions analyzed
    pub function_count: usize,
    /// Number of call sites analyzed
    pub call_site_count: usize,
    /// Number of cross-function flows
    pub cross_function_flows: usize,
    /// Number of recursive constraint groups
    pub recursive_groups: usize,
    /// Number of virtual method constraints
    pub virtual_constraints: usize,
    /// Total constraint graph nodes
    pub constraint_nodes: usize,
    /// Total constraint graph edges
    pub constraint_edges: usize,
    /// Analysis time in milliseconds
    pub analysis_time_ms: u64,
}

impl GlobalLifetimeConstraints {
    /// Create new empty global constraints
    pub fn new() -> Self {
        Self {
            call_site_constraints: new_id_map(),
            function_signatures: new_id_map(),
            lifetime_flows: vec![],
            recursive_constraint_groups: vec![],
            virtual_method_constraints: new_id_map(),
            constraint_graph: LifetimeConstraintGraph::new(),
            violations: vec![],
            stats: GlobalAnalysisStats::default(),
        }
    }

    /// Clear all constraints and reset state
    pub fn clear(&mut self) {
        self.call_site_constraints.clear();
        self.function_signatures.clear();
        self.lifetime_flows.clear();
        self.recursive_constraint_groups.clear();
        self.virtual_method_constraints.clear();
        self.constraint_graph.clear();
        self.violations.clear();
        self.stats = GlobalAnalysisStats::default();
    }

    /// Check if there are any global lifetime violations
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Check if global constraints are compatible with function-level constraints
    pub fn is_compatible(&self, function_constraints: &[LifetimeConstraint]) -> bool {
        // Check if function constraints conflict with any global constraints
        for constraint in function_constraints {
            if !self.is_constraint_compatible(constraint) {
                return false;
            }
        }
        true
    }

    /// Check if a specific constraint is compatible with global constraints
    fn is_constraint_compatible(&self, _constraint: &LifetimeConstraint) -> bool {
        // For now, assume compatibility
        // In a full implementation, this would check for conflicts
        true
    }

    /// Collect function signatures and their lifetime information
    pub fn collect_function_signatures(
        &mut self,
        call_graph: &CallGraph,
    ) -> Result<(), LifetimeAnalysisError> {
        self.stats.function_count = call_graph.functions.len();

        for &function_id in &call_graph.functions {
            let signature = self.extract_function_signature(function_id)?;
            self.function_signatures.insert(function_id, signature);
        }

        Ok(())
    }

    /// Extract lifetime signature from a function
    fn extract_function_signature(
        &self,
        function_id: SymbolId,
    ) -> Result<FunctionLifetimeSignature, LifetimeAnalysisError> {
        // For now, create a basic signature
        // In a full implementation, this would analyze the function's AST/TAST
        Ok(FunctionLifetimeSignature {
            function_id,
            parameter_lifetimes: vec![],
            return_lifetime: None,
            generic_lifetime_params: vec![],
            lifetime_bounds: vec![],
            source_location: SourceLocation::unknown(),
        })
    }

    /// Generate constraints for each call site
    pub fn generate_call_site_constraints(
        &mut self,
        call_graph: &CallGraph,
    ) -> Result<(), LifetimeAnalysisError> {
        self.stats.call_site_count = call_graph.call_sites.len();

        for call_site in call_graph.call_sites.values() {
            match &call_site.callee {
                CallTarget::Direct { function } => {
                    let constraint = self.generate_direct_call_constraint(call_site, *function)?;
                    self.call_site_constraints.insert(call_site.id, constraint);
                }
                CallTarget::Virtual {
                    possible_targets,
                    method_name,
                    ..
                } => {
                    self.generate_virtual_call_constraints(
                        call_site,
                        possible_targets,
                        method_name,
                    )?;
                }
                CallTarget::Dynamic {
                    possible_targets, ..
                } => {
                    self.generate_dynamic_call_constraints(call_site, possible_targets)?;
                }
                CallTarget::External { .. } | CallTarget::Unresolved { .. } => {
                    // Skip external and unresolved calls for now
                }
            }
        }

        Ok(())
    }

    /// Generate constraint for a direct function call
    fn generate_direct_call_constraint(
        &self,
        call_site: &CallSite,
        callee: SymbolId,
    ) -> Result<CallSiteLifetimeConstraint, LifetimeAnalysisError> {
        Ok(CallSiteLifetimeConstraint {
            call_site_id: call_site.id,
            caller_function: call_site.caller,
            callee_function: callee,
            argument_flows: vec![],
            return_flows: vec![],
            borrow_constraints: vec![],
            source_location: call_site.source_location,
        })
    }

    /// Generate constraints for virtual method calls
    fn generate_virtual_call_constraints(
        &mut self,
        call_site: &CallSite,
        possible_targets: &[SymbolId],
        method_name: &str,
    ) -> Result<(), LifetimeAnalysisError> {
        let mut implementations = vec![];

        for &target in possible_targets {
            let constraint = self.generate_direct_call_constraint(call_site, target)?;
            implementations.push(VirtualImplementationConstraint {
                implementation: target,
                constraint,
                receiver_type: TypeId::from_raw(0), // Placeholder
            });
        }

        let virtual_constraint = VirtualLifetimeConstraints {
            method_name: method_name.to_string(),
            implementations,
            unified_constraint: None,
        };

        self.virtual_method_constraints
            .insert(call_site.caller, virtual_constraint);
        self.stats.virtual_constraints += 1;

        Ok(())
    }

    /// Generate constraints for dynamic calls
    fn generate_dynamic_call_constraints(
        &mut self,
        call_site: &CallSite,
        possible_targets: &[SymbolId],
    ) -> Result<(), LifetimeAnalysisError> {
        // For dynamic calls, generate constraints for all possible targets
        for &target in possible_targets {
            let constraint = self.generate_direct_call_constraint(call_site, target)?;
            self.call_site_constraints.insert(call_site.id, constraint);
        }

        Ok(())
    }

    /// Analyze recursive functions (SCCs in call graph)
    pub fn analyze_recursive_functions(
        &mut self,
        call_graph: &CallGraph,
    ) -> Result<(), LifetimeAnalysisError> {
        for scc in &call_graph.recursion_info.scc_components {
            if scc.has_cycles && scc.functions.len() > 0 {
                let recursive_group = self.build_recursive_constraint_group(scc)?;
                self.recursive_constraint_groups.push(recursive_group);
            }
        }

        self.stats.recursive_groups = self.recursive_constraint_groups.len();
        Ok(())
    }

    /// Build constraint group for recursive functions
    fn build_recursive_constraint_group(
        &self,
        scc: &crate::semantic_graph::call_graph::StronglyConnectedComponent,
    ) -> Result<RecursiveConstraintGroup, LifetimeAnalysisError> {
        Ok(RecursiveConstraintGroup {
            functions: scc.functions.clone(),
            fixed_point_constraints: vec![],
            convergence_criteria: ConvergenceCriteria::default(),
        })
    }

    /// Resolve virtual method lifetime polymorphism
    pub fn resolve_virtual_methods(
        &mut self,
        _call_graph: &CallGraph,
    ) -> Result<(), LifetimeAnalysisError> {
        // Unify constraints for virtual methods
        for virtual_constraint in self.virtual_method_constraints.values_mut() {
            if let Some(unified) =
                Self::unify_virtual_implementations(&virtual_constraint.implementations)?
            {
                virtual_constraint.unified_constraint = Some(unified);
            }
        }

        Ok(())
    }

    /// Unify constraints across virtual method implementations
    fn unify_virtual_implementations(
        implementations: &[VirtualImplementationConstraint],
    ) -> Result<Option<CallSiteLifetimeConstraint>, LifetimeAnalysisError> {
        if implementations.is_empty() {
            return Ok(None);
        }

        // For now, use the first implementation as the unified constraint
        // In a full implementation, this would merge all constraints
        Ok(Some(implementations[0].constraint.clone()))
    }

    /// Build unified constraint graph
    pub fn build_constraint_graph(&mut self) -> Result<(), LifetimeAnalysisError> {
        self.constraint_graph.clear();

        // Add nodes for all lifetimes
        for signature in self.function_signatures.values() {
            for &lifetime in &signature.parameter_lifetimes {
                self.constraint_graph.add_node(
                    lifetime,
                    NodeType::Parameter,
                    signature.source_location,
                );
            }

            if let Some(return_lifetime) = signature.return_lifetime {
                self.constraint_graph.add_node(
                    return_lifetime,
                    NodeType::Return,
                    signature.source_location,
                );
            }
        }

        // Add edges for call site constraints
        for constraint in self.call_site_constraints.values() {
            self.constraint_graph.add_call_site_edges(constraint)?;
        }

        self.stats.constraint_nodes = self.constraint_graph.nodes.len();
        self.stats.constraint_edges = self.constraint_graph.edges.len();

        Ok(())
    }

    /// Solve the global constraint system using constraint propagation and unification
    pub fn solve_constraints(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Phase 1: Build constraint graph if not already built
        if self.constraint_graph.nodes.is_empty() {
            self.build_constraint_graph()?;
        }

        // Phase 2: Find strongly connected components for recursive constraints
        self.find_strongly_connected_components()?;

        // Phase 3: Solve constraints using iterative constraint propagation
        self.solve_constraint_propagation()?;

        // Phase 4: Handle recursive constraint groups with fixed-point iteration
        self.solve_recursive_constraints()?;

        // Phase 5: Validate solution and check for violations
        self.validate_constraint_solution()?;

        Ok(())
    }

    /// Find strongly connected components in the constraint graph
    fn find_strongly_connected_components(&mut self) -> Result<(), LifetimeAnalysisError> {
        let mut index_counter = 0;
        let mut stack = Vec::new();
        let mut indices: BTreeMap<LifetimeId, usize> = BTreeMap::new();
        let mut lowlinks: BTreeMap<LifetimeId, usize> = BTreeMap::new();
        let mut on_stack: BTreeSet<LifetimeId> = BTreeSet::new();

        self.constraint_graph.sccs.clear();

        // Run Tarjan's algorithm on each unvisited node
        let lifetimes: Vec<LifetimeId> = self.constraint_graph.nodes.keys().copied().collect();
        for lifetime in lifetimes {
            if !indices.contains_key(&lifetime) {
                self.tarjan_strongconnect(
                    lifetime,
                    &mut index_counter,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                )?;
            }
        }

        Ok(())
    }

    /// Tarjan's algorithm for finding strongly connected components
    fn tarjan_strongconnect(
        &mut self,
        v: LifetimeId,
        index_counter: &mut usize,
        stack: &mut Vec<LifetimeId>,
        indices: &mut BTreeMap<LifetimeId, usize>,
        lowlinks: &mut BTreeMap<LifetimeId, usize>,
        on_stack: &mut BTreeSet<LifetimeId>,
    ) -> Result<(), LifetimeAnalysisError> {
        indices.insert(v, *index_counter);
        lowlinks.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v);

        // Consider successors of v
        let edges: Vec<_> = self
            .constraint_graph
            .edges
            .iter()
            .filter(|edge| edge.from == v)
            .cloned()
            .collect();

        for edge in edges {
            let w = edge.to;
            if !indices.contains_key(&w) {
                // Successor w has not yet been visited; recurse on it
                self.tarjan_strongconnect(w, index_counter, stack, indices, lowlinks, on_stack)?;
                let w_lowlink = lowlinks[&w];
                let v_lowlink = lowlinks[&v].min(w_lowlink);
                lowlinks.insert(v, v_lowlink);
            } else if on_stack.contains(&w) {
                // Successor w is in stack and hence in the current SCC
                let w_index = indices[&w];
                let v_lowlink = lowlinks[&v].min(w_index);
                lowlinks.insert(v, v_lowlink);
            }
        }

        // If v is a root node, pop the stack and create an SCC
        if lowlinks[&v] == indices[&v] {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.push(w);
                if w == v {
                    break;
                }
            }
            self.constraint_graph.sccs.push(scc);
        }

        Ok(())
    }

    /// Solve constraints using iterative constraint propagation
    fn solve_constraint_propagation(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Track constraint solution state
        let mut solution_changed = true;
        let mut iteration_count = 0;
        const MAX_ITERATIONS: usize = 1000;

        // Lifetime bounds tracking: lifetime -> set of lifetimes it must outlive
        let mut outlives_constraints: BTreeMap<LifetimeId, BTreeSet<LifetimeId>> = BTreeMap::new();
        let mut equal_constraints: BTreeMap<LifetimeId, BTreeSet<LifetimeId>> = BTreeMap::new();

        // Initialize constraint sets
        for edge in &self.constraint_graph.edges {
            match edge.edge_type {
                EdgeType::Outlives => {
                    outlives_constraints
                        .entry(edge.from)
                        .or_insert_with(BTreeSet::new)
                        .insert(edge.to);
                }
                EdgeType::Equal => {
                    equal_constraints
                        .entry(edge.from)
                        .or_insert_with(BTreeSet::new)
                        .insert(edge.to);
                    equal_constraints
                        .entry(edge.to)
                        .or_insert_with(BTreeSet::new)
                        .insert(edge.from);
                }
                EdgeType::Borrow => {
                    // Borrow implies outlives
                    outlives_constraints
                        .entry(edge.from)
                        .or_insert_with(BTreeSet::new)
                        .insert(edge.to);
                }
            }
        }

        // Iterative constraint propagation
        while solution_changed && iteration_count < MAX_ITERATIONS {
            solution_changed = false;
            iteration_count += 1;

            // Propagate outlives constraints transitively
            for (&lifetime, outlives_set) in &outlives_constraints.clone() {
                for &outlived in outlives_set {
                    if let Some(transitive_outlives) = outlives_constraints.get(&outlived).cloned()
                    {
                        for &transitive in &transitive_outlives {
                            if outlives_constraints
                                .entry(lifetime)
                                .or_insert_with(BTreeSet::new)
                                .insert(transitive)
                            {
                                solution_changed = true;
                            }
                        }
                    }
                }
            }

            // Propagate equality constraints
            for (&lifetime, equal_set) in &equal_constraints.clone() {
                for &equal_lifetime in equal_set {
                    // If A == B and B outlives C, then A outlives C
                    if let Some(equal_outlives) = outlives_constraints.get(&equal_lifetime).cloned()
                    {
                        for &outlived in &equal_outlives {
                            if outlives_constraints
                                .entry(lifetime)
                                .or_insert_with(BTreeSet::new)
                                .insert(outlived)
                            {
                                solution_changed = true;
                            }
                        }
                    }

                    // If A == B and C outlives B, then C outlives A
                    for (&other_lifetime, other_outlives) in &outlives_constraints.clone() {
                        if other_outlives.contains(&equal_lifetime) {
                            if outlives_constraints
                                .entry(other_lifetime)
                                .or_insert_with(BTreeSet::new)
                                .insert(lifetime)
                            {
                                solution_changed = true;
                            }
                        }
                    }
                }
            }

            // Check for contradiction: if A outlives B and B outlives A, they must be equal
            for (&lifetime_a, outlives_set_a) in &outlives_constraints {
                for &lifetime_b in outlives_set_a {
                    if let Some(outlives_set_b) = outlives_constraints.get(&lifetime_b) {
                        if outlives_set_b.contains(&lifetime_a) {
                            // Contradiction found - A outlives B and B outlives A
                            // They must be equal
                            if equal_constraints
                                .entry(lifetime_a)
                                .or_insert_with(BTreeSet::new)
                                .insert(lifetime_b)
                            {
                                solution_changed = true;
                            }
                            if equal_constraints
                                .entry(lifetime_b)
                                .or_insert_with(BTreeSet::new)
                                .insert(lifetime_a)
                            {
                                solution_changed = true;
                            }
                        }
                    }
                }
            }
        }

        if iteration_count >= MAX_ITERATIONS {
            return Err(LifetimeAnalysisError::ConstraintSolvingTimeout);
        }

        // Store the solution back into the constraint graph
        self.store_constraint_solution(outlives_constraints, equal_constraints)?;

        Ok(())
    }

    /// Store the constraint solution back into the graph
    fn store_constraint_solution(
        &mut self,
        outlives_constraints: BTreeMap<LifetimeId, BTreeSet<LifetimeId>>,
        equal_constraints: BTreeMap<LifetimeId, BTreeSet<LifetimeId>>,
    ) -> Result<(), LifetimeAnalysisError> {
        // Create unified constraint representation
        // For now, we'll store this as additional metadata in the constraint graph

        // Create equivalence classes for equal lifetimes
        let mut equivalence_classes: BTreeMap<LifetimeId, Vec<LifetimeId>> = BTreeMap::new();
        let mut visited = BTreeSet::new();

        for (&lifetime, equal_set) in &equal_constraints {
            if visited.contains(&lifetime) {
                continue;
            }

            let mut equivalence_class = vec![lifetime];
            let mut to_visit = vec![lifetime];
            visited.insert(lifetime);

            while let Some(current) = to_visit.pop() {
                if let Some(equals) = equal_constraints.get(&current) {
                    for &equal_lifetime in equals {
                        if visited.insert(equal_lifetime) {
                            equivalence_class.push(equal_lifetime);
                            to_visit.push(equal_lifetime);
                        }
                    }
                }
            }

            let representative = equivalence_class[0];
            equivalence_classes.insert(representative, equivalence_class);
        }

        // Validate that the solution is consistent
        for (&lifetime, outlives_set) in &outlives_constraints {
            for &outlived in outlives_set {
                // Check that no lifetime outlives itself (would be a cycle)
                if lifetime == outlived {
                    return Err(LifetimeAnalysisError::CyclicLifetimeConstraint(lifetime));
                }

                // Check that equal lifetimes don't have outlives relationships
                if let Some(equal_set) = equal_constraints.get(&lifetime) {
                    if equal_set.contains(&outlived) {
                        return Err(LifetimeAnalysisError::ContradictoryConstraints(
                            lifetime, outlived,
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Solve recursive constraints using fixed-point iteration
    fn solve_recursive_constraints(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Clone the groups to avoid borrow conflicts
        let mut groups = self.recursive_constraint_groups.clone();
        for recursive_group in &mut groups {
            self.solve_recursive_group(recursive_group)?;
        }
        // Update the original groups with the results
        self.recursive_constraint_groups = groups;
        Ok(())
    }

    /// Solve a single recursive constraint group
    fn solve_recursive_group(
        &self,
        group: &mut RecursiveConstraintGroup,
    ) -> Result<(), LifetimeAnalysisError> {
        let mut iteration = 0;
        let mut converged = false;

        while !converged && iteration < group.convergence_criteria.max_iterations {
            converged = true;

            // Apply fixed-point constraints
            for constraint in &group.fixed_point_constraints {
                let previous_solution = self.evaluate_constraint_equation(&constraint.equation)?;

                // Apply the constraint equation
                let new_solution =
                    self.apply_constraint_equation(&constraint.equation, &previous_solution)?;

                // Check for convergence
                if !self.solutions_equal(
                    &previous_solution,
                    &new_solution,
                    constraint.stability_threshold,
                ) {
                    converged = false;
                }
            }

            iteration += 1;
        }

        if !converged {
            return Err(LifetimeAnalysisError::RecursiveConstraintNonConvergence(
                group.functions.clone(),
            ));
        }

        Ok(())
    }

    /// Evaluate a constraint equation
    fn evaluate_constraint_equation(
        &self,
        equation: &ConstraintEquation,
    ) -> Result<Vec<LifetimeId>, LifetimeAnalysisError> {
        match equation.operator {
            ConstraintOperator::Outlives => {
                // For outlives, return the union of all RHS lifetimes
                Ok(equation.rhs.clone())
            }
            ConstraintOperator::Equal => {
                // For equality, return all lifetimes
                let mut result = equation.lhs.clone();
                result.extend(equation.rhs.clone());
                Ok(result)
            }
            ConstraintOperator::Union => {
                let mut result = equation.lhs.clone();
                result.extend(equation.rhs.clone());
                Ok(result)
            }
            ConstraintOperator::Intersection => {
                let lhs_set: BTreeSet<_> = equation.lhs.iter().collect();
                let rhs_set: BTreeSet<_> = equation.rhs.iter().collect();
                Ok(lhs_set.intersection(&rhs_set).cloned().cloned().collect())
            }
        }
    }

    /// Apply a constraint equation to a solution
    fn apply_constraint_equation(
        &self,
        equation: &ConstraintEquation,
        solution: &[LifetimeId],
    ) -> Result<Vec<LifetimeId>, LifetimeAnalysisError> {
        // For now, return the solution unchanged
        // In a full implementation, this would apply the constraint transformation
        Ok(solution.to_vec())
    }

    /// Check if two solutions are equal within a threshold
    fn solutions_equal(
        &self,
        solution1: &[LifetimeId],
        solution2: &[LifetimeId],
        threshold: f64,
    ) -> bool {
        if solution1.len() != solution2.len() {
            return false;
        }

        let set1: BTreeSet<_> = solution1.iter().collect();
        let set2: BTreeSet<_> = solution2.iter().collect();

        let intersection_size = set1.intersection(&set2).count();
        let union_size = set1.union(&set2).count();

        if union_size == 0 {
            return true;
        }

        let similarity = intersection_size as f64 / union_size as f64;
        similarity >= (1.0 - threshold)
    }

    /// Validate the constraint solution for global violations
    fn validate_constraint_solution(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Check for impossible lifetime relationships
        for edge in &self.constraint_graph.edges {
            match edge.edge_type {
                EdgeType::Outlives => {
                    // Check if there's a path from 'to' back to 'from' (cycle)
                    if self.has_path_in_constraint_graph(edge.to, edge.from) {
                        self.violations
                            .push(GlobalLifetimeViolation::CrossFunctionUseAfterFree {
                                caller: SymbolId::from_raw(0), // Would need to track actual caller
                                callee: SymbolId::from_raw(0), // Would need to track actual callee
                                call_site: edge.call_site.unwrap_or(CallSiteId::from_raw(0)),
                                violated_lifetime: edge.from,
                                source_location: SourceLocation::unknown(),
                            });
                    }
                }
                EdgeType::Equal => {
                    // Equal lifetimes should not have outlives relationships
                    if self.has_outlives_relationship(edge.from, edge.to)
                        || self.has_outlives_relationship(edge.to, edge.from)
                    {
                        self.violations
                            .push(GlobalLifetimeViolation::InvalidCrossFunctionBorrow {
                                borrower: SymbolId::from_raw(0),
                                borrowed_from: SymbolId::from_raw(0),
                                call_site: edge.call_site.unwrap_or(CallSiteId::from_raw(0)),
                                borrow_lifetime: edge.from,
                                source_location: SourceLocation::unknown(),
                            });
                    }
                }
                EdgeType::Borrow => {
                    // Borrow relationships should be consistent with outlives
                    if !self.has_outlives_relationship(edge.from, edge.to) {
                        self.violations
                            .push(GlobalLifetimeViolation::InvalidCrossFunctionBorrow {
                                borrower: SymbolId::from_raw(0),
                                borrowed_from: SymbolId::from_raw(0),
                                call_site: edge.call_site.unwrap_or(CallSiteId::from_raw(0)),
                                borrow_lifetime: edge.to,
                                source_location: SourceLocation::unknown(),
                            });
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if there's a path from one lifetime to another in the constraint graph
    fn has_path_in_constraint_graph(&self, from: LifetimeId, to: LifetimeId) -> bool {
        let mut visited = BTreeSet::new();
        let mut stack = vec![from];

        while let Some(current) = stack.pop() {
            if current == to {
                return true;
            }

            if visited.insert(current) {
                for edge in &self.constraint_graph.edges {
                    if edge.from == current && edge.edge_type == EdgeType::Outlives {
                        stack.push(edge.to);
                    }
                }
            }
        }

        false
    }

    /// Check if there's an outlives relationship between two lifetimes
    fn has_outlives_relationship(&self, from: LifetimeId, to: LifetimeId) -> bool {
        self.constraint_graph
            .edges
            .iter()
            .any(|edge| edge.from == from && edge.to == to && edge.edge_type == EdgeType::Outlives)
    }

    /// Validate the solution for global violations
    pub fn validate_solution(&mut self) -> Result<(), LifetimeAnalysisError> {
        self.violations.clear();

        // Check for various types of violations
        self.check_cross_function_violations()?;
        self.check_recursive_violations()?;
        self.check_virtual_method_violations()?;

        if !self.violations.is_empty() {
            return Err(LifetimeAnalysisError::GlobalViolations(
                self.violations.clone(),
            ));
        }

        Ok(())
    }

    /// Check for cross-function lifetime violations
    fn check_cross_function_violations(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Placeholder implementation
        Ok(())
    }

    /// Check for recursive lifetime violations
    fn check_recursive_violations(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Placeholder implementation
        Ok(())
    }

    /// Check for virtual method lifetime violations
    fn check_virtual_method_violations(&mut self) -> Result<(), LifetimeAnalysisError> {
        // Placeholder implementation
        Ok(())
    }
}

impl LifetimeConstraintGraph {
    /// Create new empty constraint graph
    pub fn new() -> Self {
        Self {
            nodes: new_id_map(),
            edges: vec![],
            sccs: vec![],
        }
    }

    /// Clear the constraint graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.sccs.clear();
    }

    /// Add a node to the constraint graph
    pub fn add_node(
        &mut self,
        lifetime: LifetimeId,
        node_type: NodeType,
        source_location: SourceLocation,
    ) {
        let node = ConstraintNode {
            lifetime,
            metadata: NodeMetadata {
                source_function: None,
                node_type,
                source_location,
            },
        };
        self.nodes.insert(lifetime, node);
    }

    /// Add edges for a call site constraint
    pub fn add_call_site_edges(
        &mut self,
        constraint: &CallSiteLifetimeConstraint,
    ) -> Result<(), LifetimeAnalysisError> {
        // Add edges for argument flows
        for flow in &constraint.argument_flows {
            self.edges.push(ConstraintEdge {
                from: flow.from,
                to: flow.to,
                edge_type: match flow.flow_kind {
                    LifetimeFlowKind::Outlives => EdgeType::Outlives,
                    LifetimeFlowKind::Equal => EdgeType::Equal,
                    LifetimeFlowKind::Borrow => EdgeType::Borrow,
                    LifetimeFlowKind::Move => EdgeType::Outlives, // Move implies outlives
                },
                call_site: Some(constraint.call_site_id),
            });
        }

        // Add edges for return flows
        for flow in &constraint.return_flows {
            self.edges.push(ConstraintEdge {
                from: flow.from,
                to: flow.to,
                edge_type: match flow.flow_kind {
                    LifetimeFlowKind::Outlives => EdgeType::Outlives,
                    LifetimeFlowKind::Equal => EdgeType::Equal,
                    LifetimeFlowKind::Borrow => EdgeType::Borrow,
                    LifetimeFlowKind::Move => EdgeType::Outlives,
                },
                call_site: Some(constraint.call_site_id),
            });
        }

        Ok(())
    }
}

impl fmt::Display for GlobalLifetimeViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalLifetimeViolation::CrossFunctionUseAfterFree {
                caller,
                callee,
                call_site,
                ..
            } => {
                write!(
                    f,
                    "Cross-function use-after-free: {:?} -> {:?} at call site {:?}",
                    caller, callee, call_site
                )
            }
            GlobalLifetimeViolation::InvalidCrossFunctionBorrow {
                borrower,
                borrowed_from,
                call_site,
                ..
            } => {
                write!(
                    f,
                    "Invalid cross-function borrow: {:?} borrows from {:?} at call site {:?}",
                    borrower, borrowed_from, call_site
                )
            }
            GlobalLifetimeViolation::RecursiveLifetimeExtension {
                recursive_group, ..
            } => {
                write!(
                    f,
                    "Recursive lifetime extension in functions: {:?}",
                    recursive_group
                )
            }
            GlobalLifetimeViolation::VirtualMethodLifetimeMismatch {
                method_name,
                implementations,
                ..
            } => {
                write!(
                    f,
                    "Virtual method '{}' lifetime mismatch across implementations: {:?}",
                    method_name, implementations
                )
            }
        }
    }
}

// Add error types for global constraints
impl LifetimeAnalysisError {
    /// Create error for global violations
    pub fn global_violations(violations: Vec<GlobalLifetimeViolation>) -> Self {
        Self::GlobalViolations(violations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_constraints_creation() {
        let constraints = GlobalLifetimeConstraints::new();
        assert!(constraints.call_site_constraints.is_empty());
        assert!(constraints.function_signatures.is_empty());
        assert!(!constraints.has_violations());
    }

    #[test]
    fn test_constraint_graph() {
        let mut graph = LifetimeConstraintGraph::new();
        let lifetime = LifetimeId::from_raw(1);

        graph.add_node(lifetime, NodeType::Parameter, SourceLocation::unknown());
        assert!(graph.nodes.contains_key(&lifetime));
    }

    #[test]
    fn test_function_signature() {
        let signature = FunctionLifetimeSignature {
            function_id: SymbolId::from_raw(1),
            parameter_lifetimes: vec![LifetimeId::from_raw(1), LifetimeId::from_raw(2)],
            return_lifetime: Some(LifetimeId::from_raw(3)),
            generic_lifetime_params: vec![],
            lifetime_bounds: vec![],
            source_location: SourceLocation::unknown(),
        };

        assert_eq!(signature.parameter_lifetimes.len(), 2);
        assert!(signature.return_lifetime.is_some());
    }
}
