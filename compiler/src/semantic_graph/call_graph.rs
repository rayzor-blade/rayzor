//! Call Graph implementation for inter-procedural analysis
//!
//! The call graph represents the calling relationships between functions, enabling
//! inter-procedural optimization, recursion detection, and call-site analysis.
//! Supports Haxe-specific features like dynamic dispatch and method overriding.
//!
//! Key features:
//! - Direct and indirect call relationships
//! - Dynamic dispatch resolution
//! - Call site information with context
//! - Recursion and cycle detection
//! - Performance-critical call path analysis

use super::{SourceLocation, SymbolId};
use crate::semantic_graph::CallType;
use crate::tast::collections::{new_id_map, new_id_set, IdMap, IdSet};
use crate::tast::node::TypedExpression;
use crate::tast::{BlockId, CallSiteId, DataFlowNodeId, TypeId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

/// Complete call graph for a program or module
#[derive(Debug, Clone)]
pub struct CallGraph {
    /// All call sites indexed by ID
    pub call_sites: IdMap<CallSiteId, CallSite>,

    /// All functions that can be called
    pub functions: IdSet<SymbolId>,

    /// Direct call relationships: caller -> callees
    pub direct_calls: IdMap<SymbolId, Vec<CallSiteId>>,

    /// Reverse call relationships: callee -> callers
    pub callers: IdMap<SymbolId, Vec<CallSiteId>>,

    /// Virtual method call resolution
    pub virtual_calls: VirtualCallResolution,

    /// Recursion analysis information
    pub recursion_info: RecursionInfo,

    /// Call graph statistics
    pub statistics: CallGraphStatistics,
}

/// Information about a specific call site
#[derive(Debug, Clone)]
pub struct CallSite {
    /// Unique identifier
    pub id: CallSiteId,

    /// Function making the call (caller)
    pub caller: SymbolId,

    /// Function being called (callee) - may be resolved or unresolved
    pub callee: CallTarget,

    /// Type of call (direct, virtual, etc.)
    pub call_type: CallType,

    /// Basic block containing this call
    pub basic_block: BlockId,

    /// DFG node representing this call (if available)
    pub dfg_node: Option<DataFlowNodeId>,

    /// Source location of the call
    pub source_location: SourceLocation,

    /// Call arguments information
    pub arguments: Vec<ArgumentInfo>,

    /// Return type of the call
    pub return_type: TypeId,

    /// Call site metadata
    pub metadata: CallSiteMetadata,
}

/// Target of a function call
#[derive(Debug, Clone)]
pub enum CallTarget {
    /// Direct call to a known function
    Direct { function: SymbolId },

    /// Virtual method call that may resolve to multiple implementations
    Virtual {
        method_name: String,
        receiver_type: TypeId,
        possible_targets: Vec<SymbolId>,
    },

    /// Dynamic call through function pointer/delegate
    Dynamic {
        function_expr: TypedExpression,
        possible_targets: Vec<SymbolId>,
    },

    /// Call to external/builtin function
    External {
        function_name: String,
        module: Option<String>,
    },

    /// Unresolved call (error recovery)
    Unresolved { name: String, reason: String },
}

/// Information about call arguments
#[derive(Debug, Clone)]
pub struct ArgumentInfo {
    /// Argument expression
    pub expression: TypedExpression,

    /// Argument type
    pub arg_type: TypeId,

    /// Whether this argument is passed by reference
    pub by_reference: bool,

    /// Usage pattern (move, borrow, etc.)
    pub usage: ArgumentUsage,
}

/// How an argument is used in the call
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentUsage {
    /// Argument value is moved to callee
    Move,
    /// Argument is borrowed immutably
    ImmutableBorrow,
    /// Argument is borrowed mutably
    MutableBorrow,
    /// Argument is copied
    Copy,
}

/// Call site metadata for analysis
#[derive(Debug, Clone, Default)]
pub struct CallSiteMetadata {
    /// Whether this call can throw exceptions
    pub can_throw: bool,

    /// Whether this call has side effects
    pub has_side_effects: bool,

    /// Whether this call is tail-recursive
    pub is_tail_call: bool,

    /// Estimated execution frequency
    pub execution_frequency: f64,

    /// Inlining hints
    pub inlining_hint: InliningHint,

    /// Custom analysis annotations
    pub annotations: BTreeMap<String, String>,
}

/// Hints for function inlining
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InliningHint {
    /// No specific hint
    None,
    /// Suggest inlining this call
    SuggestInline,
    /// Avoid inlining this call
    AvoidInline,
    /// Always inline (forced)
    AlwaysInline,
    /// Never inline
    NeverInline,
}

impl Default for InliningHint {
    fn default() -> Self {
        Self::None
    }
}

/// Virtual method call resolution information
#[derive(Debug, Default, Clone)]
pub struct VirtualCallResolution {
    /// Virtual method table information
    pub vtables: IdMap<TypeId, VirtualMethodTable>,

    /// Method override relationships
    pub overrides: BTreeMap<SymbolId, Vec<SymbolId>>,

    /// Interface implementation mappings
    pub interface_impls: BTreeMap<(TypeId, SymbolId), SymbolId>,
}

/// Virtual method table for a type
#[derive(Debug, Clone)]
pub struct VirtualMethodTable {
    /// Type this vtable belongs to
    pub type_id: TypeId,

    /// Method entries in the vtable
    pub methods: Vec<VTableEntry>,

    /// Parent class vtable (for inheritance)
    pub parent: Option<TypeId>,
}

/// Entry in a virtual method table
#[derive(Debug, Clone)]
pub struct VTableEntry {
    /// Method symbol
    pub method: SymbolId,

    /// Vtable slot index
    pub slot_index: usize,

    /// Whether this method is abstract
    pub is_abstract: bool,

    /// Whether this method is overridden
    pub is_override: bool,
}

/// Recursion analysis information
#[derive(Debug, Default, Clone)]
pub struct RecursionInfo {
    /// Strongly connected components (recursive function groups)
    pub scc_components: Vec<StronglyConnectedComponent>,

    /// Functions that are directly recursive
    pub directly_recursive: IdSet<SymbolId>,

    /// Functions that are mutually recursive
    pub mutually_recursive: Vec<Vec<SymbolId>>,

    /// Maximum recursion depth detected
    pub max_recursion_depth: Option<u32>,
}

/// Strongly connected component in the call graph
#[derive(Debug, Clone)]
pub struct StronglyConnectedComponent {
    /// Functions in this SCC
    pub functions: Vec<SymbolId>,

    /// Whether this SCC contains cycles (recursion)
    pub has_cycles: bool,

    /// Entry points to this SCC
    pub entry_points: Vec<SymbolId>,

    /// Exit points from this SCC
    pub exit_points: Vec<SymbolId>,
}

/// Call graph statistics
#[derive(Debug, Clone, Default)]
pub struct CallGraphStatistics {
    /// Total number of functions
    pub function_count: usize,

    /// Total number of call sites
    pub call_site_count: usize,

    /// Number of direct calls
    pub direct_call_count: usize,

    /// Number of virtual calls
    pub virtual_call_count: usize,

    /// Number of dynamic calls
    pub dynamic_call_count: usize,

    /// Number of external calls
    pub external_call_count: usize,

    /// Number of recursive functions
    pub recursive_function_count: usize,

    /// Number of strongly connected components
    pub scc_count: usize,

    /// Maximum call depth
    pub max_call_depth: u32,

    /// Average calls per function
    pub avg_calls_per_function: f64,
}

impl CallGraph {
    /// Create a new empty call graph
    pub fn new() -> Self {
        Self {
            call_sites: new_id_map(),
            functions: new_id_set(),
            direct_calls: new_id_map(),
            callers: new_id_map(),
            virtual_calls: VirtualCallResolution::default(),
            recursion_info: RecursionInfo::default(),
            statistics: CallGraphStatistics::default(),
        }
    }

    /// Add a function to the call graph
    pub fn add_function(&mut self, function_id: SymbolId) {
        self.functions.insert(function_id);
        self.direct_calls
            .entry(function_id)
            .or_insert_with(Vec::new);
    }

    /// Add a call site to the call graph
    pub fn add_call_site(&mut self, call_site: CallSite) -> CallSiteId {
        let call_site_id = call_site.id;
        let caller = call_site.caller;

        // Update direct calls mapping
        self.direct_calls
            .entry(caller)
            .or_insert_with(Vec::new)
            .push(call_site_id);

        // Update reverse mapping (callers)
        if let Some(callee) = call_site.get_direct_callee() {
            self.callers
                .entry(callee)
                .or_insert_with(Vec::new)
                .push(call_site_id);
        }

        // Add to call sites collection
        self.call_sites.insert(call_site_id, call_site);

        call_site_id
    }

    /// Get all call sites from a function
    pub fn get_calls_from(&self, function: SymbolId) -> &[CallSiteId] {
        self.direct_calls
            .get(&function)
            .map_or(&[], |v| v.as_slice())
    }

    /// Get all call sites to a function
    pub fn get_calls_to(&self, function: SymbolId) -> &[CallSiteId] {
        self.callers.get(&function).map_or(&[], |v| v.as_slice())
    }

    /// Get a call site by ID
    pub fn get_call_site(&self, id: CallSiteId) -> Option<&CallSite> {
        self.call_sites.get(&id)
    }

    /// Check if a function is recursive
    pub fn is_recursive(&self, function: SymbolId) -> bool {
        self.recursion_info.directly_recursive.contains(&function)
            || self
                .recursion_info
                .mutually_recursive
                .iter()
                .any(|group| group.contains(&function))
    }

    /// Find all functions reachable from a given function
    pub fn reachable_functions(&self, start: SymbolId) -> BTreeSet<SymbolId> {
        let mut reachable = BTreeSet::new();
        let mut worklist = VecDeque::new();

        worklist.push_back(start);
        reachable.insert(start);

        while let Some(function) = worklist.pop_front() {
            for &call_site_id in self.get_calls_from(function) {
                if let Some(call_site) = self.get_call_site(call_site_id) {
                    // Follow all possible callees (direct, virtual, dynamic)
                    for callee in call_site.get_possible_callees() {
                        if reachable.insert(callee) {
                            worklist.push_back(callee);
                        }
                    }
                }
            }
        }

        reachable
    }

    /// Compute strongly connected components for recursion analysis
    pub fn compute_strongly_connected_components(&mut self) {
        let scc_components = self.tarjan_scc_algorithm();

        // Analyze each SCC for recursion
        for mut scc in scc_components {
            scc.has_cycles = self.scc_has_cycles(&scc.functions);

            if scc.has_cycles {
                if scc.functions.len() == 1 {
                    // Single function SCC with cycles = direct recursion
                    self.recursion_info
                        .directly_recursive
                        .insert(scc.functions[0]);
                } else {
                    // Multi-function SCC with cycles = mutual recursion
                    self.recursion_info
                        .mutually_recursive
                        .push(scc.functions.clone());
                }
            }

            self.recursion_info.scc_components.push(scc);
        }
    }

    /// Tarjan's algorithm for finding strongly connected components
    fn tarjan_scc_algorithm(&self) -> Vec<StronglyConnectedComponent> {
        let mut sccs = Vec::new();
        let mut index_counter = 0;
        let mut stack = Vec::new();
        let mut indices: BTreeMap<SymbolId, usize> = BTreeMap::new();
        let mut lowlinks: BTreeMap<SymbolId, usize> = BTreeMap::new();
        let mut on_stack: BTreeSet<SymbolId> = BTreeSet::new();

        for &function in &self.functions {
            if !indices.contains_key(&function) {
                self.tarjan_strongconnect(
                    function,
                    &mut index_counter,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut sccs,
                );
            }
        }

        sccs
    }

    /// Recursive helper for Tarjan's algorithm
    fn tarjan_strongconnect(
        &self,
        v: SymbolId,
        index_counter: &mut usize,
        stack: &mut Vec<SymbolId>,
        indices: &mut BTreeMap<SymbolId, usize>,
        lowlinks: &mut BTreeMap<SymbolId, usize>,
        on_stack: &mut BTreeSet<SymbolId>,
        sccs: &mut Vec<StronglyConnectedComponent>,
    ) {
        indices.insert(v, *index_counter);
        lowlinks.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v);

        // Consider successors of v (follow all possible callees)
        for &call_site_id in self.get_calls_from(v) {
            if let Some(call_site) = self.get_call_site(call_site_id) {
                for w in call_site.get_possible_callees() {
                    if !indices.contains_key(&w) {
                        // Successor w has not yet been visited; recurse on it
                        self.tarjan_strongconnect(
                            w,
                            index_counter,
                            stack,
                            indices,
                            lowlinks,
                            on_stack,
                            sccs,
                        );
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
            }
        }

        // If v is a root node, pop the stack and create an SCC
        if lowlinks[&v] == indices[&v] {
            let mut scc_functions = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc_functions.push(w);
                if w == v {
                    break;
                }
            }

            let scc = StronglyConnectedComponent {
                functions: scc_functions,
                has_cycles: false, // Will be determined later
                entry_points: vec![],
                exit_points: vec![],
            };
            sccs.push(scc);
        }
    }

    /// Check if an SCC has cycles (internal edges)
    fn scc_has_cycles(&self, functions: &[SymbolId]) -> bool {
        if functions.len() == 1 {
            // Single function SCC - check for self-calls
            let function = functions[0];
            for &call_site_id in self.get_calls_from(function) {
                if let Some(call_site) = self.get_call_site(call_site_id) {
                    if call_site.get_direct_callee() == Some(function) {
                        return true;
                    }
                }
            }
            false
        } else {
            // Multi-function SCC always has cycles by definition
            true
        }
    }

    /// Analyze call depth and frequency
    pub fn analyze_call_patterns(&mut self) {
        // Compute maximum call depth
        let mut max_depth = 0;
        let mut max_recursion_depth = 0;

        for &function in &self.functions {
            let depth = self.compute_call_depth(function, &mut BTreeSet::new());
            max_depth = max_depth.max(depth);

            // For recursive functions, compute recursion depth
            if self.is_recursive(function) {
                let recursion_depth = self.compute_recursion_depth(function);
                max_recursion_depth = max_recursion_depth.max(recursion_depth);
            }
        }

        self.statistics.max_call_depth = max_depth;

        // Set max recursion depth if we found any recursive functions
        if max_recursion_depth > 0 {
            self.recursion_info.max_recursion_depth = Some(max_recursion_depth);
        }

        // Compute average calls per function
        let total_calls: usize = self.direct_calls.values().map(|calls| calls.len()).sum();
        self.statistics.avg_calls_per_function = if self.functions.is_empty() {
            0.0
        } else {
            total_calls as f64 / self.functions.len() as f64
        };
    }

    /// Compute call depth from a function (with cycle detection)
    fn compute_call_depth(&self, function: SymbolId, visited: &mut BTreeSet<SymbolId>) -> u32 {
        if visited.contains(&function) {
            return 0; // Cycle detected, return 0 to avoid infinite recursion
        }

        visited.insert(function);

        let mut max_depth = 0;
        for &call_site_id in self.get_calls_from(function) {
            if let Some(call_site) = self.get_call_site(call_site_id) {
                if let Some(callee) = call_site.get_direct_callee() {
                    let depth = self.compute_call_depth(callee, visited);
                    max_depth = max_depth.max(depth + 1);
                }
            }
        }

        visited.remove(&function);
        max_depth
    }

    /// Compute recursion depth for a recursive function
    fn compute_recursion_depth(&self, function: SymbolId) -> u32 {
        // For directly recursive functions, check how many levels deep the recursion goes
        if self.recursion_info.directly_recursive.contains(&function) {
            // Simple heuristic: count the maximum depth in the SCC containing this function
            for scc in &self.recursion_info.scc_components {
                if scc.functions.contains(&function) && scc.has_cycles {
                    // For now, return a simple depth based on the SCC size
                    // In a more sophisticated implementation, we could analyze call patterns
                    return scc.functions.len() as u32;
                }
            }
        }

        // For mutually recursive functions, find the group and return its size
        for group in &self.recursion_info.mutually_recursive {
            if group.contains(&function) {
                return group.len() as u32;
            }
        }

        // Not actually recursive or no depth info available
        0
    }

    /// Update statistics
    pub fn update_statistics(&mut self) {
        self.statistics.function_count = self.functions.len();
        self.statistics.call_site_count = self.call_sites.len();

        // Count different call types
        for call_site in self.call_sites.values() {
            match &call_site.callee {
                CallTarget::Direct { .. } => self.statistics.direct_call_count += 1,
                CallTarget::Virtual { .. } => self.statistics.virtual_call_count += 1,
                CallTarget::Dynamic { .. } => self.statistics.dynamic_call_count += 1,
                CallTarget::External { .. } => self.statistics.external_call_count += 1,
                CallTarget::Unresolved { .. } => {}
            }
        }

        self.statistics.recursive_function_count = self.recursion_info.directly_recursive.len()
            + self
                .recursion_info
                .mutually_recursive
                .iter()
                .map(|group| group.len())
                .sum::<usize>();

        self.statistics.scc_count = self.recursion_info.scc_components.len();

        self.analyze_call_patterns();
    }
}

impl CallSite {
    /// Create a new call site
    pub fn new(
        id: CallSiteId,
        caller: SymbolId,
        callee: CallTarget,
        call_type: CallType,
        basic_block: BlockId,
        source_location: SourceLocation,
    ) -> Self {
        Self {
            id,
            caller,
            callee,
            call_type,
            basic_block,
            dfg_node: None,
            source_location,
            arguments: vec![],
            return_type: TypeId::from_raw(1), // Default to void
            metadata: CallSiteMetadata::default(),
        }
    }

    /// Get the direct callee if this is a direct call
    pub fn get_direct_callee(&self) -> Option<SymbolId> {
        match &self.callee {
            CallTarget::Direct { function } => Some(*function),
            _ => None,
        }
    }

    /// Get all possible callees (for virtual/dynamic calls)
    pub fn get_possible_callees(&self) -> Vec<SymbolId> {
        match &self.callee {
            CallTarget::Direct { function } => vec![*function],
            CallTarget::Virtual {
                possible_targets, ..
            } => possible_targets.clone(),
            CallTarget::Dynamic {
                possible_targets, ..
            } => possible_targets.clone(),
            CallTarget::External { .. } | CallTarget::Unresolved { .. } => vec![],
        }
    }

    /// Check if this call site is recursive
    pub fn is_recursive(&self) -> bool {
        self.get_possible_callees().contains(&self.caller)
    }

    /// Check if this call site might have side effects
    pub fn has_side_effects(&self) -> bool {
        self.metadata.has_side_effects
            || matches!(self.callee, CallTarget::External { .. })
            || matches!(self.callee, CallTarget::Dynamic { .. })
    }
}

impl fmt::Display for CallTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallTarget::Direct { function } => write!(f, "direct({:?})", function),
            CallTarget::Virtual {
                method_name,
                receiver_type,
                possible_targets,
            } => {
                write!(
                    f,
                    "virtual({}@{:?}, {} targets)",
                    method_name,
                    receiver_type,
                    possible_targets.len()
                )
            }
            CallTarget::Dynamic {
                possible_targets, ..
            } => {
                write!(f, "dynamic({} targets)", possible_targets.len())
            }
            CallTarget::External {
                function_name,
                module,
            } => {
                if let Some(module) = module {
                    write!(f, "external({}::{})", module, function_name)
                } else {
                    write!(f, "external({})", function_name)
                }
            }
            CallTarget::Unresolved { name, reason } => {
                write!(f, "unresolved({}: {})", name, reason)
            }
        }
    }
}

impl fmt::Display for InliningHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InliningHint::None => write!(f, "none"),
            InliningHint::SuggestInline => write!(f, "suggest"),
            InliningHint::AvoidInline => write!(f, "avoid"),
            InliningHint::AlwaysInline => write!(f, "always"),
            InliningHint::NeverInline => write!(f, "never"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_graph_creation() {
        let call_graph = CallGraph::new();

        assert!(call_graph.functions.is_empty());
        assert!(call_graph.call_sites.is_empty());
        assert_eq!(call_graph.statistics.function_count, 0);
    }

    #[test]
    fn test_add_function() {
        let mut call_graph = CallGraph::new();
        let function_id = SymbolId::from_raw(1);

        call_graph.add_function(function_id);

        assert!(call_graph.functions.contains(&function_id));
        assert!(call_graph.direct_calls.contains_key(&function_id));
    }

    #[test]
    fn test_add_call_site() {
        let mut call_graph = CallGraph::new();

        let caller = SymbolId::from_raw(1);
        let callee = SymbolId::from_raw(2);

        call_graph.add_function(caller);
        call_graph.add_function(callee);

        let call_site = CallSite::new(
            CallSiteId::from_raw(1),
            caller,
            CallTarget::Direct { function: callee },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );

        let call_site_id = call_graph.add_call_site(call_site);

        // Check call site was added
        assert!(call_graph.call_sites.contains_key(&call_site_id));

        // Check caller mapping
        let calls_from_caller = call_graph.get_calls_from(caller);
        assert_eq!(calls_from_caller.len(), 1);
        assert_eq!(calls_from_caller[0], call_site_id);

        // Check callee mapping
        let calls_to_callee = call_graph.get_calls_to(callee);
        assert_eq!(calls_to_callee.len(), 1);
        assert_eq!(calls_to_callee[0], call_site_id);
    }

    #[test]
    fn test_reachable_functions() {
        let mut call_graph = CallGraph::new();

        let func_a = SymbolId::from_raw(1);
        let func_b = SymbolId::from_raw(2);
        let func_c = SymbolId::from_raw(3);

        call_graph.add_function(func_a);
        call_graph.add_function(func_b);
        call_graph.add_function(func_c);

        // A calls B, B calls C
        let call_site_1 = CallSite::new(
            CallSiteId::from_raw(1),
            func_a,
            CallTarget::Direct { function: func_b },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(call_site_1);

        let call_site_2 = CallSite::new(
            CallSiteId::from_raw(2),
            func_b,
            CallTarget::Direct { function: func_c },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(call_site_2);

        let reachable = call_graph.reachable_functions(func_a);

        assert_eq!(reachable.len(), 3);
        assert!(reachable.contains(&func_a));
        assert!(reachable.contains(&func_b));
        assert!(reachable.contains(&func_c));
    }

    #[test]
    fn test_recursion_detection() {
        let mut call_graph = CallGraph::new();

        let func_a = SymbolId::from_raw(1);
        call_graph.add_function(func_a);

        // A calls itself (direct recursion)
        let call_site = CallSite::new(
            CallSiteId::from_raw(1),
            func_a,
            CallTarget::Direct { function: func_a },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(call_site);

        call_graph.compute_strongly_connected_components();

        assert!(call_graph.is_recursive(func_a));
        assert!(call_graph
            .recursion_info
            .directly_recursive
            .contains(&func_a));
    }

    #[test]
    fn test_mutual_recursion() {
        let mut call_graph = CallGraph::new();

        let func_a = SymbolId::from_raw(1);
        let func_b = SymbolId::from_raw(2);

        call_graph.add_function(func_a);
        call_graph.add_function(func_b);

        // A calls B, B calls A (mutual recursion)
        let call_site_1 = CallSite::new(
            CallSiteId::from_raw(1),
            func_a,
            CallTarget::Direct { function: func_b },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(call_site_1);

        let call_site_2 = CallSite::new(
            CallSiteId::from_raw(2),
            func_b,
            CallTarget::Direct { function: func_a },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(call_site_2);

        call_graph.compute_strongly_connected_components();

        assert!(call_graph.is_recursive(func_a));
        assert!(call_graph.is_recursive(func_b));

        // Should have one SCC with both functions
        assert_eq!(call_graph.recursion_info.scc_components.len(), 1);
        let scc = &call_graph.recursion_info.scc_components[0];
        assert_eq!(scc.functions.len(), 2);
        assert!(scc.has_cycles);
    }

    #[test]
    fn test_call_statistics() {
        let mut call_graph = CallGraph::new();

        let func_a = SymbolId::from_raw(1);
        let func_b = SymbolId::from_raw(2);

        call_graph.add_function(func_a);
        call_graph.add_function(func_b);

        // Add various types of calls
        let direct_call = CallSite::new(
            CallSiteId::from_raw(1),
            func_a,
            CallTarget::Direct { function: func_b },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(direct_call);

        let virtual_call = CallSite::new(
            CallSiteId::from_raw(2),
            func_a,
            CallTarget::Virtual {
                method_name: "test".to_string(),
                receiver_type: TypeId::from_raw(1),
                possible_targets: vec![func_b],
            },
            CallType::Virtual,
            BlockId::from_raw(1),
            SourceLocation::unknown(),
        );
        call_graph.add_call_site(virtual_call);

        call_graph.update_statistics();

        assert_eq!(call_graph.statistics.function_count, 2);
        assert_eq!(call_graph.statistics.call_site_count, 2);
        assert_eq!(call_graph.statistics.direct_call_count, 1);
        assert_eq!(call_graph.statistics.virtual_call_count, 1);
        assert_eq!(call_graph.statistics.avg_calls_per_function, 1.0);
    }
}
