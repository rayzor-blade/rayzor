//! Escape Analysis for Memory Allocation Optimization
//!
//! The EscapeAnalyzer determines which allocations can be optimized for stack allocation
//! vs heap allocation by tracking how allocated objects escape from their allocation site.
//! This analysis is crucial for performance optimization in the HIR lowering phase.
//!
//! Key features:
//! - Allocation site detection via DFG analysis
//! - Escape tracking through def-use chains
//! - Function call escape analysis
//! - Return value escape detection
//! - Stack allocation opportunity identification
//! - Integration with existing semantic graph infrastructure

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use crate::semantic_graph::analysis::ownership_analyzer::FunctionAnalysisContext;
use crate::semantic_graph::dfg::{AllocationKind, CallType, ConstantValue};
use crate::semantic_graph::{
    CallGraph, ControlFlowGraph, DataFlowGraph, DataFlowNode, DataFlowNodeKind, OwnershipGraph,
};
use crate::tast::{DataFlowNodeId, SourceLocation, SsaVariableId, SymbolId, TypeId};

/// **EscapeAnalyzer - Memory Allocation Optimization**
///
/// The EscapeAnalyzer determines which memory allocations can be optimized by
/// tracking how allocated objects "escape" their allocation scope. Objects that
/// don't escape can be allocated on the stack for better performance.
pub struct EscapeAnalyzer {
    /// Tracks allocation sites and their properties
    allocation_tracker: AllocationTracker,

    /// Analyzes escapes through function calls
    call_analyzer: CallEscapeAnalyzer,

    /// Generates optimization hints for HIR lowering
    optimization_generator: OptimizationHintGenerator,

    /// Performance and diagnostic tracking
    stats: EscapeAnalysisStats,
}

/// Tracks allocation sites within functions
#[derive(Debug, Default)]
pub struct AllocationTracker {
    /// Cache of detected allocation sites
    allocation_cache: BTreeMap<DataFlowNodeId, AllocationInfo>,

    /// Statistics for cache efficiency
    cache_hits: usize,
    cache_misses: usize,
}

/// Analyzes how objects escape through function calls
#[derive(Debug, Default)]
pub struct CallEscapeAnalyzer {
    /// Known escape patterns for function calls
    escape_patterns: BTreeMap<SymbolId, CallEscapePattern>,

    /// Cache for call analysis results
    call_analysis_cache: BTreeMap<DataFlowNodeId, CallEscapeResult>,
}

/// Generates optimization hints for HIR lowering
#[derive(Debug, Default)]
pub struct OptimizationHintGenerator {
    /// Functions safe for inlining
    inlinable_functions: BTreeSet<SymbolId>,

    /// Stack allocatable objects
    stack_allocatable: BTreeSet<DataFlowNodeId>,
}

/// Information about an allocation site
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Location of the allocation
    pub allocation_site: DataFlowNodeId,

    /// Type being allocated
    pub allocated_type: TypeId,

    /// Kind of allocation (stack, heap, etc.)
    pub allocation_kind: AllocationKind,

    /// Source location for diagnostics
    pub source_location: SourceLocation,

    /// Size of allocation (if known)
    pub allocation_size: Option<u64>,

    /// Whether allocation size is compile-time constant
    pub is_constant_size: bool,
}

/// Results from escape analysis
#[derive(Debug, Clone)]
pub struct EscapeAnalysisResults {
    /// All detected allocation sites
    pub allocation_sites: BTreeMap<DataFlowNodeId, AllocationInfo>,

    /// Escape status for each allocation
    pub escape_status: BTreeMap<DataFlowNodeId, EscapeStatus>,

    /// Allocations that can use stack allocation
    pub stack_allocatable: Vec<DataFlowNodeId>,

    /// Functions safe for inlining
    pub inlinable_functions: Vec<SymbolId>,

    /// Optimization hints for HIR lowering
    pub optimization_hints: Vec<OptimizationHint>,

    /// Analysis statistics
    pub stats: EscapeAnalysisStats,
}

/// Escape status of an allocation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscapeStatus {
    /// Object doesn't escape - can use stack allocation
    NoEscape,

    /// Object escapes via function return
    EscapesViaReturn,

    /// Object escapes via function call argument
    EscapesViaCall { call_site: DataFlowNodeId },

    /// Object escapes via storage in global variable
    EscapesViaGlobal { global: SymbolId },

    /// Object escapes via storage in another escaping object
    EscapesViaContainer { container: DataFlowNodeId },

    /// Conservative analysis - assume escapes
    Unknown,
}

/// How function calls affect escape analysis
#[derive(Debug, Clone)]
pub enum CallEscapePattern {
    /// Function doesn't cause arguments to escape
    DoesNotEscape,

    /// Function causes specific arguments to escape
    EscapesArguments { argument_indices: Vec<usize> },

    /// Function may escape any argument (conservative)
    MayEscapeAny,

    /// Function's escape behavior is unknown
    Unknown,
}

/// Result of analyzing a specific function call for escapes
#[derive(Debug, Clone)]
pub struct CallEscapeResult {
    /// Whether the call causes escapes
    pub causes_escape: bool,

    /// Which arguments escape (if any)
    pub escaped_arguments: Vec<DataFlowNodeId>,

    /// Reason for the escape (for diagnostics)
    pub escape_reason: String,
}

/// Optimization hints for HIR lowering
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Use stack allocation for this object
    StackAllocation {
        allocation_site: DataFlowNodeId,
        estimated_size: u64,
    },

    /// Function is safe to inline
    InlineFunction { function: SymbolId, reason: String },

    /// Remove unnecessary allocation
    RemoveAllocation {
        allocation_site: DataFlowNodeId,
        reason: String,
    },

    /// Combine multiple allocations
    CombineAllocations {
        allocations: Vec<DataFlowNodeId>,
        combined_size: u64,
    },
}

/// Performance statistics for escape analysis
#[derive(Debug, Clone, Default)]
pub struct EscapeAnalysisStats {
    /// Time spent in analysis
    pub analysis_time: Duration,

    /// Number of allocations analyzed
    pub allocations_analyzed: usize,

    /// Number of function calls analyzed
    pub calls_analyzed: usize,

    /// Number of stack allocation opportunities found
    pub stack_opportunities: usize,

    /// Number of inlinable functions found
    pub inlinable_functions: usize,

    /// Cache efficiency metrics
    pub cache_hit_ratio: f64,
}

/// Escape analysis error types
#[derive(Debug)]
pub enum EscapeAnalysisError {
    /// Internal analysis error
    InternalError(String),

    /// Graph integrity error
    GraphIntegrityError(String),

    /// Analysis timeout
    AnalysisTimeout,
}

impl EscapeAnalyzer {
    /// Create new escape analyzer
    pub fn new() -> Self {
        Self {
            allocation_tracker: AllocationTracker::default(),
            call_analyzer: CallEscapeAnalyzer::default(),
            optimization_generator: OptimizationHintGenerator::default(),
            stats: EscapeAnalysisStats::default(),
        }
    }

    /// Analyze escape behavior for a specific function
    pub fn analyze_function(
        &mut self,
        context: &FunctionAnalysisContext,
    ) -> Result<EscapeAnalysisResults, EscapeAnalysisError> {
        let start_time = Instant::now();

        // 1. Find all allocation sites in the function
        let allocations = self.find_allocation_sites(context.dfg)?;

        // 2. Analyze escape status for each allocation
        let mut escape_status = BTreeMap::new();
        let mut stack_allocatable = Vec::new();

        for allocation in &allocations {
            let status = self.analyze_allocation_escape(allocation, context)?;

            if status == EscapeStatus::NoEscape {
                stack_allocatable.push(allocation.allocation_site);
                self.optimization_generator
                    .stack_allocatable
                    .insert(allocation.allocation_site);
            }

            escape_status.insert(allocation.allocation_site, status);
        }

        // 3. Analyze function inlinability
        let inlinable = self.analyze_function_inlinability(context)?;
        if inlinable {
            self.optimization_generator
                .inlinable_functions
                .insert(context.function_id);
        }

        // 4. Generate optimization hints
        let optimization_hints = self.generate_optimization_hints(&allocations, &escape_status);

        // Update statistics
        self.stats.analysis_time += start_time.elapsed();
        self.stats.allocations_analyzed += allocations.len();
        self.stats.stack_opportunities += stack_allocatable.len();
        if inlinable {
            self.stats.inlinable_functions += 1;
        }

        // Calculate cache hit ratio
        let total_lookups =
            self.allocation_tracker.cache_hits + self.allocation_tracker.cache_misses;
        self.stats.cache_hit_ratio = if total_lookups > 0 {
            self.allocation_tracker.cache_hits as f64 / total_lookups as f64
        } else {
            0.0
        };

        let allocation_sites: BTreeMap<DataFlowNodeId, AllocationInfo> = allocations
            .into_iter()
            .map(|alloc| (alloc.allocation_site, alloc))
            .collect();

        Ok(EscapeAnalysisResults {
            allocation_sites,
            escape_status,
            stack_allocatable,
            inlinable_functions: if inlinable {
                vec![context.function_id]
            } else {
                vec![]
            },
            optimization_hints,
            stats: self.stats.clone(),
        })
    }

    /// Global escape analysis across multiple functions
    pub fn analyze_escapes(
        &mut self,
        call_graph: &CallGraph,
        ownership_graph: &OwnershipGraph,
    ) -> Result<EscapeAnalysisResults, EscapeAnalysisError> {
        let start_time = Instant::now();

        // For now, return basic results
        // This would be expanded to do inter-procedural analysis

        self.stats.analysis_time += start_time.elapsed();

        Ok(EscapeAnalysisResults {
            allocation_sites: BTreeMap::new(),
            escape_status: BTreeMap::new(),
            stack_allocatable: Vec::new(),
            inlinable_functions: self
                .optimization_generator
                .inlinable_functions
                .iter()
                .copied()
                .collect(),
            optimization_hints: Vec::new(),
            stats: self.stats.clone(),
        })
    }

    /// Get analysis statistics
    pub fn stats(&self) -> &EscapeAnalysisStats {
        &self.stats
    }

    // Private implementation methods

    /// Find all allocation sites in a function's DFG
    fn find_allocation_sites(
        &mut self,
        dfg: &DataFlowGraph,
    ) -> Result<Vec<AllocationInfo>, EscapeAnalysisError> {
        let mut allocations = Vec::new();

        for (node_id, node) in &dfg.nodes {
            // Check cache first
            if let Some(cached_alloc) = self.allocation_tracker.allocation_cache.get(node_id) {
                self.allocation_tracker.cache_hits += 1;
                allocations.push(cached_alloc.clone());
                continue;
            }

            self.allocation_tracker.cache_misses += 1;

            if let Some(allocation_info) = self.detect_allocation_from_node(node)? {
                // Cache the result
                self.allocation_tracker
                    .allocation_cache
                    .insert(*node_id, allocation_info.clone());
                allocations.push(allocation_info);
            }
        }

        Ok(allocations)
    }

    /// Detect if a DFG node represents an allocation
    fn detect_allocation_from_node(
        &self,
        node: &DataFlowNode,
    ) -> Result<Option<AllocationInfo>, EscapeAnalysisError> {
        match &node.kind {
            DataFlowNodeKind::Allocation {
                allocation_type,
                size,
                allocation_kind,
            } => {
                // Direct allocation node
                let allocation_size = if let Some(size_node_id) = size {
                    // Would need to resolve the size from the DFG
                    // For now, use a default size
                    Some(64) // Placeholder
                } else {
                    None
                };

                Ok(Some(AllocationInfo {
                    allocation_site: node.id,
                    allocated_type: *allocation_type,
                    allocation_kind: *allocation_kind,
                    source_location: node.source_location.clone(),
                    allocation_size,
                    is_constant_size: allocation_size.is_some(),
                }))
            }

            DataFlowNodeKind::Call {
                call_type: CallType::Constructor,
                ..
            } => {
                // Constructor calls are allocations
                Ok(Some(AllocationInfo {
                    allocation_site: node.id,
                    allocated_type: node.value_type,
                    allocation_kind: AllocationKind::Heap, // Constructors typically allocate on heap
                    source_location: node.source_location.clone(),
                    allocation_size: None, // Unknown size for constructor calls
                    is_constant_size: false,
                }))
            }

            DataFlowNodeKind::BinaryOp { operator, .. } => {
                // String concatenation and array operations may allocate
                // This is a simplified check - would need more sophisticated analysis
                if self.is_potentially_allocating_operation(operator) {
                    Ok(Some(AllocationInfo {
                        allocation_site: node.id,
                        allocated_type: node.value_type,
                        allocation_kind: AllocationKind::Heap,
                        source_location: node.source_location.clone(),
                        allocation_size: None,
                        is_constant_size: false,
                    }))
                } else {
                    Ok(None)
                }
            }

            _ => Ok(None),
        }
    }

    /// Check if a binary operation potentially allocates memory
    fn is_potentially_allocating_operation(
        &self,
        operator: &crate::tast::node::BinaryOperator,
    ) -> bool {
        use crate::tast::node::BinaryOperator;
        match operator {
            BinaryOperator::Add => true, // String concatenation
            _ => false,
        }
    }

    /// Analyze how an allocation escapes
    fn analyze_allocation_escape(
        &mut self,
        allocation: &AllocationInfo,
        context: &FunctionAnalysisContext,
    ) -> Result<EscapeStatus, EscapeAnalysisError> {
        // Use def-use chains to trace how the allocation is used
        let uses = context.dfg.get_uses(allocation.allocation_site);

        for &use_node_id in uses {
            if let Some(use_node) = context.dfg.get_node(use_node_id) {
                match self.analyze_node_for_escape(&use_node, context)? {
                    EscapeStatus::NoEscape => continue,
                    escape_status => return Ok(escape_status),
                }
            }
        }

        // If no uses cause escape, the allocation doesn't escape
        Ok(EscapeStatus::NoEscape)
    }

    /// Analyze a specific node to see if it causes an escape
    fn analyze_node_for_escape(
        &mut self,
        node: &DataFlowNode,
        context: &FunctionAnalysisContext,
    ) -> Result<EscapeStatus, EscapeAnalysisError> {
        match &node.kind {
            DataFlowNodeKind::Return { .. } => {
                // Returning an allocation causes it to escape
                Ok(EscapeStatus::EscapesViaReturn)
            }

            DataFlowNodeKind::Call { arguments, .. } => {
                // Check if any arguments are the allocation we're tracking
                for &arg in arguments {
                    // This is simplified - would need to trace through SSA variables
                    // to see if the argument references our allocation
                }

                // For now, conservatively assume calls cause escape
                Ok(EscapeStatus::EscapesViaCall { call_site: node.id })
            }

            DataFlowNodeKind::Store { address, .. } => {
                // Storing to global memory causes escape
                // Would need to analyze the address to determine if it's global
                // For now, conservatively assume escape
                Ok(EscapeStatus::EscapesViaGlobal {
                    global: SymbolId::from_raw(0),
                })
            }

            DataFlowNodeKind::FieldAccess { .. } | DataFlowNodeKind::ArrayAccess { .. } => {
                // Field and array accesses don't inherently cause escape
                Ok(EscapeStatus::NoEscape)
            }

            _ => {
                // For other node types, conservatively assume no escape
                Ok(EscapeStatus::NoEscape)
            }
        }
    }

    /// Analyze if a function is safe for inlining
    fn analyze_function_inlinability(
        &self,
        context: &FunctionAnalysisContext,
    ) -> Result<bool, EscapeAnalysisError> {
        // Simple heuristics for inlinability:
        // 1. Function should be small (few nodes)
        // 2. No complex control flow
        // 3. No function calls (or only simple calls)

        let node_count = context.dfg.nodes.len();
        let block_count = context.cfg.blocks.len();

        // Heuristic: functions with <10 nodes and single basic block are inlinable
        let is_small = node_count < 10;
        let is_simple = block_count <= 1;

        Ok(is_small && is_simple)
    }

    /// Generate optimization hints based on analysis results
    fn generate_optimization_hints(
        &self,
        allocations: &[AllocationInfo],
        escape_status: &BTreeMap<DataFlowNodeId, EscapeStatus>,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        for allocation in allocations {
            if let Some(status) = escape_status.get(&allocation.allocation_site) {
                match status {
                    EscapeStatus::NoEscape => {
                        // Suggest stack allocation
                        hints.push(OptimizationHint::StackAllocation {
                            allocation_site: allocation.allocation_site,
                            estimated_size: allocation.allocation_size.unwrap_or(64),
                        });
                    }
                    _ => {
                        // For escaping allocations, no optimization hints for now
                    }
                }
            }
        }

        // Add inlining hints
        for &function_id in &self.optimization_generator.inlinable_functions {
            hints.push(OptimizationHint::InlineFunction {
                function: function_id,
                reason: "Small function with simple control flow".to_string(),
            });
        }

        hints
    }
}

impl EscapeAnalysisResults {
    /// Create new empty results
    pub fn new() -> Self {
        Self {
            allocation_sites: BTreeMap::new(),
            escape_status: BTreeMap::new(),
            stack_allocatable: Vec::new(),
            inlinable_functions: Vec::new(),
            optimization_hints: Vec::new(),
            stats: EscapeAnalysisStats::default(),
        }
    }

    /// Clear all results
    pub fn clear(&mut self) {
        self.allocation_sites.clear();
        self.escape_status.clear();
        self.stack_allocatable.clear();
        self.inlinable_functions.clear();
        self.optimization_hints.clear();
    }

    /// Get functions suitable for inlining
    pub fn get_inlinable_functions(&self) -> Vec<SymbolId> {
        self.inlinable_functions.clone()
    }

    /// Get stack allocation opportunities
    pub fn get_stack_allocatable(&self) -> &[DataFlowNodeId] {
        &self.stack_allocatable
    }

    /// Get optimization hints for HIR lowering
    pub fn get_optimization_hints(&self) -> &[OptimizationHint] {
        &self.optimization_hints
    }
}

// Display implementations for error reporting

impl std::fmt::Display for EscapeAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalError(msg) => write!(f, "Internal escape analysis error: {}", msg),
            Self::GraphIntegrityError(msg) => write!(f, "Graph integrity error: {}", msg),
            Self::AnalysisTimeout => write!(f, "Escape analysis timed out"),
        }
    }
}

impl std::error::Error for EscapeAnalysisError {}

impl Default for EscapeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for EscapeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoEscape => write!(f, "No escape"),
            Self::EscapesViaReturn => write!(f, "Escapes via return"),
            Self::EscapesViaCall { call_site } => write!(f, "Escapes via call at {:?}", call_site),
            Self::EscapesViaGlobal { global } => write!(f, "Escapes via global {:?}", global),
            Self::EscapesViaContainer { container } => {
                write!(f, "Escapes via container {:?}", container)
            }
            Self::Unknown => write!(f, "Unknown escape"),
        }
    }
}

impl std::fmt::Display for OptimizationHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StackAllocation {
                allocation_site,
                estimated_size,
            } => {
                write!(
                    f,
                    "Use stack allocation for {:?} (size: {} bytes)",
                    allocation_site, estimated_size
                )
            }
            Self::InlineFunction { function, reason } => {
                write!(f, "Inline function {:?}: {}", function, reason)
            }
            Self::RemoveAllocation {
                allocation_site,
                reason,
            } => {
                write!(f, "Remove allocation {:?}: {}", allocation_site, reason)
            }
            Self::CombineAllocations {
                allocations,
                combined_size,
            } => {
                write!(
                    f,
                    "Combine {} allocations (total size: {} bytes)",
                    allocations.len(),
                    combined_size
                )
            }
        }
    }
}
