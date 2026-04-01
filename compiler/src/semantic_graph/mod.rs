//! Semantic Graph Construction
//!
//! This module transforms Typed AST (TAST) into semantic graphs optimized for advanced
//! static analysis including ownership checking, lifetime verification, and optimization.
//!
//! Architecture:
//! ```,ignore
//! TAST → Control Flow Graph (CFG) → Data Flow Graph (DFG) → Analysis Results
//!      → Call Graph              → Ownership Graph
//! ```

use std::fmt;

use crate::tast::collections::{new_id_map, IdMap};
use crate::tast::{BlockId, LifetimeId, SourceLocation, SymbolId};

// Re-export all graph types for convenience
pub use self::builder::*;
pub use self::call_graph::*;
pub use self::cfg::*;
pub use self::dfg::*;
pub use self::dominance::*;
pub use self::ownership_graph::*;
pub use self::source_location_tracking::*;
// pub use self::analysis::*;

pub mod analysis;
pub mod builder;
pub(crate) mod call_graph;

pub(crate) mod cfg;
pub(crate) mod dfg;
pub(crate) mod dfg_builder;
pub(crate) mod free_variables;
pub(crate) mod phi_type;

// mod dfg_test;
pub(crate) mod dominance;
pub(crate) mod ownership_graph;
pub(crate) mod source_location_tracking;
pub(crate) mod tast_cfg_mapping;
mod test;
pub(crate) mod validation;

impl LifetimeId {
    /// The global lifetime that outlives all other lifetimes
    pub fn global() -> Self {
        Self(0)
    }

    /// Static lifetime for compile-time constants
    pub fn static_lifetime() -> Self {
        Self(1)
    }
}

/// Complete semantic graphs for a compilation unit
#[derive(Debug, Clone)]
pub struct SemanticGraphs {
    /// Control Flow Graph for each function
    pub control_flow: IdMap<SymbolId, ControlFlowGraph>,

    /// Data Flow Graph for each function
    pub data_flow: IdMap<SymbolId, DataFlowGraph>,

    /// Inter-procedural call graph
    pub call_graph: CallGraph,

    /// Ownership and borrowing relationships
    pub ownership_graph: OwnershipGraph,

    /// Source location mapping for diagnostics
    pub source_locations: IdMap<BlockId, SourceLocation>,
}

impl SemanticGraphs {
    /// Create empty semantic graphs
    pub fn new() -> Self {
        Self {
            control_flow: new_id_map(),
            data_flow: new_id_map(),
            call_graph: CallGraph::new(),
            ownership_graph: OwnershipGraph::new(),
            source_locations: new_id_map(),
        }
    }

    /// Get CFG for a specific function
    pub fn cfg_for_function(&self, function_id: SymbolId) -> Option<&ControlFlowGraph> {
        self.control_flow.get(&function_id)
    }

    /// Get DFG for a specific function
    pub fn dfg_for_function(&self, function_id: SymbolId) -> Option<&DataFlowGraph> {
        self.data_flow.get(&function_id)
    }

    /// Check if graphs are consistent (for testing/debugging)
    pub fn validate_consistency(&self) -> Result<(), GraphValidationError> {
        // Validate CFG-DFG consistency
        for (function_id, cfg) in &self.control_flow {
            if let Some(dfg) = self.data_flow.get(function_id) {
                // Check that every CFG block has corresponding DFG nodes
                for block_id in cfg.blocks.keys() {
                    if !dfg.block_nodes.contains_key(block_id) {
                        return Err(GraphValidationError::OwnershipInconsistency {
                            message: format!(
                                "CFG block {:?} has no corresponding DFG nodes",
                                block_id
                            ),
                        });
                    }
                }

                // Check that DFG nodes reference valid CFG blocks
                for (block_id, _) in &dfg.block_nodes {
                    if !cfg.blocks.contains_key(block_id) {
                        return Err(GraphValidationError::OwnershipInconsistency {
                            message: format!(
                                "DFG references non-existent CFG block {:?}",
                                block_id
                            ),
                        });
                    }
                }
            }
        }

        // Validate call graph consistency
        for call_site in self.call_graph.call_sites.values() {
            // Check that direct called functions exist in CFG
            match &call_site.callee {
                crate::semantic_graph::CallTarget::Direct { function } => {
                    if !self.control_flow.contains_key(function) {
                        return Err(GraphValidationError::UndefinedVariables {
                            variables: vec![*function],
                        });
                    }
                }
                crate::semantic_graph::CallTarget::Virtual {
                    possible_targets, ..
                } => {
                    for &target in possible_targets {
                        if !self.control_flow.contains_key(&target) {
                            return Err(GraphValidationError::UndefinedVariables {
                                variables: vec![target],
                            });
                        }
                    }
                }
                crate::semantic_graph::CallTarget::Dynamic {
                    possible_targets, ..
                } => {
                    for &target in possible_targets {
                        if !self.control_flow.contains_key(&target) {
                            return Err(GraphValidationError::UndefinedVariables {
                                variables: vec![target],
                            });
                        }
                    }
                }
                // External and Unresolved calls don't need to be in CFG
                _ => {}
            }
        }

        // Validate ownership graph variables exist in DFG
        let mut all_original_symbols = std::collections::BTreeSet::new();
        for dfg in self.data_flow.values() {
            for ssa_var in dfg.ssa_variables.values() {
                all_original_symbols.insert(ssa_var.original_symbol);
            }
        }

        for node in self.ownership_graph.variables.values() {
            // Check that the variable's original symbol exists in the DFG
            if !all_original_symbols.contains(&node.variable) {
                return Err(GraphValidationError::UndefinedVariables {
                    variables: vec![node.variable],
                });
            }
        }

        Ok(())
    }
}

/// Errors that can occur during graph construction
#[derive(Debug, Clone)]
pub enum GraphConstructionError {
    /// Invalid TAST structure prevents graph construction
    InvalidTAST {
        message: String,
        location: SourceLocation,
    },

    TypeError {
        message: String,
    },

    InvalidCFG {
        message: String,
        location: SourceLocation,
    },

    /// Unresolved symbol reference in TAST
    UnresolvedSymbol {
        symbol_name: String,
        location: SourceLocation,
    },

    /// Type information missing from TAST node
    MissingTypeInfo {
        node_description: String,
        location: SourceLocation,
    },

    /// Graph construction internal error
    InternalError {
        message: String,
    },

    DominanceAnalysisFailed(String),
}

impl GraphConstructionError {
    pub fn dominance_analysis_failed(message: String) -> Self {
        Self::DominanceAnalysisFailed(message)
    }
}

impl fmt::Display for GraphConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphConstructionError::InvalidTAST { message, location } => {
                write!(f, "Invalid TAST at {}: {}", location, message)
            }
            GraphConstructionError::UnresolvedSymbol {
                symbol_name,
                location,
            } => {
                write!(f, "Unresolved symbol '{}' at {}", symbol_name, location)
            }
            GraphConstructionError::MissingTypeInfo {
                node_description,
                location,
            } => {
                write!(
                    f,
                    "Missing type info for {} at {}",
                    node_description, location
                )
            }
            GraphConstructionError::InternalError { message } => {
                write!(f, "Graph construction error: {}", message)
            }
            GraphConstructionError::DominanceAnalysisFailed(message) => {
                write!(f, "Dominance analysis failed: {}", message)
            }
            GraphConstructionError::InvalidCFG { message, location } => {
                write!(f, "Invalid CGF at {}: {}", location, message)
            }
            GraphConstructionError::TypeError { message } => write!(f, "Type Error: {}", message),
        }
    }
}

impl std::error::Error for GraphConstructionError {}

/// Errors during graph validation
#[derive(Debug, Clone)]
pub enum GraphValidationError {
    /// CFG has unreachable blocks
    UnreachableBlocks { block_ids: Vec<BlockId> },

    /// DFG has undefined variables
    UndefinedVariables { variables: Vec<SymbolId> },

    /// Call graph has cycles (for analysis)
    CallGraphCycles { cycle_description: String },

    /// Ownership graph inconsistency
    OwnershipInconsistency { message: String },
}

impl fmt::Display for GraphValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphValidationError::UnreachableBlocks { block_ids } => {
                write!(f, "Unreachable blocks: {:?}", block_ids)
            }
            GraphValidationError::UndefinedVariables { variables } => {
                write!(f, "Undefined variables: {:?}", variables)
            }
            GraphValidationError::CallGraphCycles { cycle_description } => {
                write!(f, "Call graph cycles: {}", cycle_description)
            }
            GraphValidationError::OwnershipInconsistency { message } => {
                write!(f, "Ownership inconsistency: {}", message)
            }
        }
    }
}

impl std::error::Error for GraphValidationError {}

/// Configuration options for semantic graph construction
#[derive(Debug, Clone)]
pub struct GraphConstructionOptions {
    /// Whether to build call graph for inter-procedural analysis
    pub build_call_graph: bool,

    /// Whether to build ownership graph for memory safety analysis
    pub build_ownership_graph: bool,

    /// Whether to perform SSA conversion on DFG
    pub convert_to_ssa: bool,

    /// Whether to eliminate dead code during construction
    pub eliminate_dead_code: bool,

    /// Maximum function size for detailed analysis (performance)
    pub max_function_size: usize,

    /// Whether to collect detailed statistics for profiling
    pub collect_statistics: bool,
}

impl Default for GraphConstructionOptions {
    fn default() -> Self {
        Self {
            build_call_graph: true,
            build_ownership_graph: true,
            convert_to_ssa: true,
            eliminate_dead_code: false, // Off by default for correctness
            max_function_size: 10_000,  // 10K statements max
            collect_statistics: false,
        }
    }
}

/// Performance statistics for semantic graph construction
#[derive(Debug, Clone, Default)]
pub struct GraphConstructionStats {
    /// Number of functions processed
    pub functions_processed: usize,

    /// Total number of basic blocks created
    pub total_basic_blocks: usize,

    /// Total number of data flow nodes created
    pub total_dfg_nodes: usize,

    /// Time spent on CFG construction (microseconds)
    pub cfg_construction_time_us: u64,

    /// Time spent on DFG construction (microseconds)
    pub dfg_construction_time_us: u64,

    /// Peak memory usage during construction (bytes)
    pub peak_memory_bytes: usize,

    /// Number of optimization opportunities found
    pub optimization_hints: usize,
}

impl GraphConstructionStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge statistics from multiple construction passes
    pub fn merge(&mut self, other: &GraphConstructionStats) {
        self.functions_processed += other.functions_processed;
        self.total_basic_blocks += other.total_basic_blocks;
        self.total_dfg_nodes += other.total_dfg_nodes;
        self.cfg_construction_time_us += other.cfg_construction_time_us;
        self.dfg_construction_time_us += other.dfg_construction_time_us;
        self.peak_memory_bytes = self.peak_memory_bytes.max(other.peak_memory_bytes);
        self.optimization_hints += other.optimization_hints;
    }

    /// Get total construction time in milliseconds
    pub fn total_time_ms(&self) -> f64 {
        (self.cfg_construction_time_us + self.dfg_construction_time_us) as f64 / 1000.0
    }

    /// Get average blocks per function
    pub fn avg_blocks_per_function(&self) -> f64 {
        if self.functions_processed == 0 {
            0.0
        } else {
            self.total_basic_blocks as f64 / self.functions_processed as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block_id() {
        let id = BlockId::from_raw(42);
        assert_eq!(id.as_raw(), 42);
        assert!(id.is_valid());

        let invalid = BlockId::invalid();
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_lifetime_id() {
        let global = LifetimeId::global();
        let static_lt = LifetimeId::static_lifetime();

        assert!(global.is_valid());
        assert!(static_lt.is_valid());
        assert_ne!(global, static_lt);
    }

    #[test]
    fn test_semantic_graphs_creation() {
        let graphs = SemanticGraphs::new();

        // Should be empty initially
        assert!(graphs.control_flow.is_empty());
        assert!(graphs.data_flow.is_empty());
    }

    #[test]
    fn test_construction_options() {
        let default_options = GraphConstructionOptions::default();
        assert!(default_options.build_call_graph);
        assert!(default_options.build_ownership_graph);
        assert!(default_options.convert_to_ssa);
        assert!(!default_options.eliminate_dead_code);

        let custom_options = GraphConstructionOptions {
            build_call_graph: false,
            max_function_size: 1000,
            ..Default::default()
        };
        assert!(!custom_options.build_call_graph);
        assert_eq!(custom_options.max_function_size, 1000);
    }

    #[test]
    fn test_stats_merging() {
        let mut stats1 = GraphConstructionStats {
            functions_processed: 5,
            total_basic_blocks: 100,
            cfg_construction_time_us: 1000,
            peak_memory_bytes: 1024,
            ..Default::default()
        };

        let stats2 = GraphConstructionStats {
            functions_processed: 3,
            total_basic_blocks: 50,
            cfg_construction_time_us: 500,
            peak_memory_bytes: 2048,
            ..Default::default()
        };

        stats1.merge(&stats2);

        assert_eq!(stats1.functions_processed, 8);
        assert_eq!(stats1.total_basic_blocks, 150);
        assert_eq!(stats1.cfg_construction_time_us, 1500);
        assert_eq!(stats1.peak_memory_bytes, 2048); // Takes maximum
        assert_eq!(stats1.avg_blocks_per_function(), 150.0 / 8.0);
    }
}
