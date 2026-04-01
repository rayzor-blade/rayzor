//! Control Flow Graph (CFG) implementation for Haxe semantic analysis
//!
//! The CFG represents the control flow structure of functions, enabling
//! analysis of execution paths, reachability, and control dependencies.
//!
//! Key features:
//! - Efficient representation with arena-allocated blocks
//! - Support for Haxe-specific control flow (pattern matching, exceptions)
//! - Optimized for common analysis patterns
//! - Comprehensive validation and debugging support

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use crate::tast::collections::{new_id_map, new_id_set, IdMap, IdSet};
use crate::tast::node::{MacroExpansionInfo, TypedExpression, TypedPattern};
use crate::tast::{BlockId, SourceLocation, StatementId, SymbolId, TypeId};

/// A basic block in the control flow graph
///
/// Basic blocks are maximal sequences of statements with a single entry point
/// (the first statement) and a single exit point (the last statement).
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Unique identifier for this block
    pub id: BlockId,

    /// Statements in this block (in execution order)
    pub statements: Vec<StatementId>,

    /// The terminating instruction (jump, return, etc.)
    pub terminator: Terminator,

    /// Blocks that can transfer control to this block
    pub predecessors: IdSet<BlockId>,

    /// Blocks that this block can transfer control to
    pub successors: Vec<BlockId>,

    /// Source location of the first statement in this block
    pub source_location: SourceLocation,

    /// Additional metadata for analysis
    pub metadata: BlockMetadata,
}

/// How control flow exits a basic block
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional jump to another block
    Jump { target: BlockId },

    /// Conditional branch based on expression
    Branch {
        condition: TypedExpression,
        true_target: BlockId,
        false_target: BlockId,
    },

    /// Multi-way branch (switch statement)
    Switch {
        discriminant: TypedExpression,
        targets: Vec<SwitchTarget>,
        default_target: Option<BlockId>,
    },

    /// Function return
    Return { value: Option<TypedExpression> },

    /// Exception throw
    Throw { exception: TypedExpression },

    /// Unreachable code (e.g., after throw in same block)
    Unreachable,

    /// Haxe-specific: pattern matching
    PatternMatch {
        value: TypedExpression,
        patterns: Vec<PatternTarget>,
        default_target: Option<BlockId>,
    },

    /// Haxe-specific: macro expansion boundary
    MacroExpansion {
        target: BlockId,
        macro_info: MacroExpansionInfo,
    },
}

/// Target for switch statements
#[derive(Debug, Clone)]
pub struct SwitchTarget {
    /// Value to match (constant expression)
    pub case_value: TypedExpression,
    /// Target block for this case
    pub target: BlockId,
}

/// Target for pattern matching
#[derive(Debug, Clone)]
pub struct PatternTarget {
    /// Pattern to match against
    pub pattern: TypedPattern,
    /// Target block if pattern matches
    pub target: BlockId,
    /// Variables bound by this pattern
    pub bound_variables: Vec<SymbolId>,
}

/// Additional metadata for basic blocks
#[derive(Debug, Clone, Default)]
pub struct BlockMetadata {
    /// Whether this block is reachable from function entry
    pub is_reachable: bool,

    /// Loop nesting depth (0 = not in loop)
    pub loop_depth: u32,

    /// Exception handler blocks that cover this block
    pub exception_handlers: Vec<BlockId>,

    /// Dominance information (for optimization)
    pub dominance_info: Option<DominanceInfo>,

    /// Analysis-specific annotations
    pub annotations: BTreeMap<String, AnalysisAnnotation>,
}

/// Dominance relationship information
#[derive(Debug, Clone)]
pub struct DominanceInfo {
    /// Immediate dominator of this block
    pub immediate_dominator: Option<BlockId>,

    /// Blocks that this block dominates
    pub dominates: IdSet<BlockId>,

    /// Dominance frontier (for SSA construction)
    pub dominance_frontier: IdSet<BlockId>,
}

/// Analysis-specific annotations that can be attached to blocks
#[derive(Debug, Clone)]
pub enum AnalysisAnnotation {
    /// Lifetime analysis annotation
    Lifetime {
        active_lifetimes: Vec<super::LifetimeId>,
    },

    /// Ownership analysis annotation
    Ownership {
        owned_variables: Vec<SymbolId>,
        borrowed_variables: Vec<SymbolId>,
    },

    /// Performance hint annotation
    Performance {
        hint_type: PerformanceHint,
        estimated_impact: f32,
    },

    /// Custom analysis annotation
    Custom {
        analysis_name: String,
        data: serde_json::Value,
    },
}

/// Performance optimization hints
#[derive(Debug, Clone)]
pub enum PerformanceHint {
    /// This block is rarely executed (cold path)
    ColdPath,

    /// This block is frequently executed (hot path)
    HotPath,

    /// Inlining opportunity
    InlineCandidate { estimated_benefit: f32 },

    /// Loop optimization opportunity
    LoopOptimization { optimization_type: String },

    /// Memory allocation optimization
    AllocationOptimization { can_stack_allocate: bool },
}

/// Complete control flow graph for a function
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// All basic blocks in this CFG
    pub blocks: IdMap<BlockId, BasicBlock>,

    /// Entry block (where function execution begins)
    pub entry_block: BlockId,

    /// Exit blocks (where function execution can end)
    pub exit_blocks: IdSet<BlockId>,

    /// Exception handler blocks
    pub exception_handlers: BTreeMap<BlockId, ExceptionHandlerInfo>,

    /// Function this CFG represents
    pub function_id: SymbolId,

    /// CFG construction metadata
    pub metadata: CfgMetadata,
}

/// Information about exception handlers
#[derive(Debug, Clone)]
pub struct ExceptionHandlerInfo {
    /// Type of exceptions this handler catches
    pub exception_types: Vec<TypeId>,

    /// Blocks covered by this handler
    pub covered_blocks: IdSet<BlockId>,

    /// Handler entry block
    pub handler_block: BlockId,

    /// Variable that receives the exception (if any)
    pub exception_variable: Option<SymbolId>,
}

/// Metadata about CFG construction and properties
#[derive(Debug, Clone, Default)]
pub struct CfgMetadata {
    /// Number of statements in the original function
    pub original_statement_count: usize,

    /// Whether this CFG has been optimized
    pub is_optimized: bool,

    /// Whether this CFG has unreachable blocks
    pub has_unreachable_blocks: bool,

    /// Maximum loop nesting depth
    pub max_loop_depth: u32,

    /// Number of exception handlers
    pub exception_handler_count: usize,

    /// Construction statistics
    pub construction_stats: CfgConstructionStats,
}

/// Statistics from CFG construction
#[derive(Debug, Clone, Default)]
pub struct CfgConstructionStats {
    /// Time taken to build CFG (microseconds)
    pub construction_time_us: u64,

    /// Number of blocks created
    pub blocks_created: usize,

    /// Number of edges created
    pub edges_created: usize,

    /// Memory allocated for this CFG (bytes)
    pub memory_bytes: usize,
}

impl BasicBlock {
    /// Create a new basic block
    pub fn new(id: BlockId, source_location: SourceLocation) -> Self {
        Self {
            id,
            statements: Vec::new(),
            terminator: Terminator::Unreachable,
            predecessors: new_id_set(),
            successors: Vec::new(),
            source_location,
            metadata: BlockMetadata::default(),
        }
    }

    /// Add a statement to this block
    pub fn add_statement(&mut self, statement: StatementId) {
        self.statements.push(statement);
    }

    /// Set the terminator for this block
    pub fn set_terminator(&mut self, terminator: Terminator) {
        // Update successors based on terminator
        self.successors.clear();
        match &terminator {
            Terminator::Jump { target } => {
                self.successors.push(*target);
            }
            Terminator::Branch {
                true_target,
                false_target,
                ..
            } => {
                self.successors.push(*true_target);
                self.successors.push(*false_target);
            }
            Terminator::Switch {
                targets,
                default_target,
                ..
            } => {
                for target in targets {
                    self.successors.push(target.target);
                }
                if let Some(default) = default_target {
                    self.successors.push(*default);
                }
            }
            Terminator::PatternMatch {
                patterns,
                default_target,
                ..
            } => {
                for pattern in patterns {
                    self.successors.push(pattern.target);
                }
                if let Some(default) = default_target {
                    self.successors.push(*default);
                }
            }
            Terminator::MacroExpansion { target, .. } => {
                self.successors.push(*target);
            }
            Terminator::Return { .. } | Terminator::Throw { .. } | Terminator::Unreachable => {
                // No successors
            }
        }

        self.terminator = terminator;
    }

    /// Check if this block has any statements
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty()
    }

    /// Get the number of statements in this block
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }

    /// Check if this block is an exit block (no successors)
    pub fn is_exit_block(&self) -> bool {
        self.successors.is_empty()
    }

    /// Check if this block ends with a return
    pub fn ends_with_return(&self) -> bool {
        matches!(self.terminator, Terminator::Return { .. })
    }

    /// Check if this block can throw exceptions
    pub fn can_throw(&self) -> bool {
        matches!(self.terminator, Terminator::Throw { .. })
    }
}

impl ControlFlowGraph {
    /// Create a new empty CFG
    pub fn new(function_id: SymbolId, entry_block: BlockId) -> Self {
        let mut blocks = new_id_map();
        let exit_blocks = new_id_set();
        let exception_handlers = BTreeMap::new();

        Self {
            blocks,
            entry_block,
            exit_blocks,
            exception_handlers,
            function_id,
            metadata: CfgMetadata::default(),
        }
    }

    /// Add a basic block to the CFG
    pub fn add_block(&mut self, block: BasicBlock) -> BlockId {
        let block_id = block.id;

        // Track exit blocks
        if block.is_exit_block() {
            self.exit_blocks.insert(block_id);
        }

        // Note: We don't update predecessor relationships here because
        // successor blocks may not exist yet. Edge consistency is maintained
        // by update_block_terminator when terminators are set.
        self.blocks.insert(block_id, block);
        block_id
    }

    /// Update a block's terminator while maintaining edge consistency
    pub fn update_block_terminator(&mut self, block_id: BlockId, new_terminator: Terminator) {
        // First, collect old successors and exit status to avoid borrow conflicts
        let (old_successors, was_exit_block) = if let Some(block) = self.blocks.get(&block_id) {
            (block.successors.clone(), block.is_exit_block())
        } else {
            return; // Block doesn't exist
        };

        // Remove this block from old successors' predecessor lists
        for old_successor_id in old_successors {
            if let Some(old_successor) = self.blocks.get_mut(&old_successor_id) {
                old_successor.predecessors.remove(&block_id);
            }
        }

        // Update exit blocks tracking for old state
        if was_exit_block {
            self.exit_blocks.remove(&block_id);
        }

        // Update the block's terminator (this updates successors)
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.set_terminator(new_terminator);
        }

        // Get new successors and exit status
        let (new_successors, is_exit_block) = if let Some(block) = self.blocks.get(&block_id) {
            (block.successors.clone(), block.is_exit_block())
        } else {
            return; // Block doesn't exist
        };

        // Add this block to new successors' predecessor lists
        for new_successor_id in new_successors {
            if let Some(new_successor) = self.blocks.get_mut(&new_successor_id) {
                new_successor.predecessors.insert(block_id);
            }
        }

        // Update exit blocks tracking for new state
        if is_exit_block {
            self.exit_blocks.insert(block_id);
        }
    }

    /// Get a block by ID
    pub fn get_block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(&id)
    }

    /// Perform depth-first traversal from entry block
    pub fn dfs_traversal(&self) -> Vec<BlockId> {
        let mut visited = BTreeSet::new();
        let mut result = Vec::new();
        let mut stack = vec![self.entry_block];

        while let Some(block_id) = stack.pop() {
            if visited.insert(block_id) {
                result.push(block_id);

                if let Some(block) = self.get_block(block_id) {
                    // Add successors in reverse order for consistent traversal
                    for &successor in block.successors.iter().rev() {
                        if !visited.contains(&successor) {
                            stack.push(successor);
                        }
                    }
                }
            }
        }

        result
    }

    /// Perform breadth-first traversal from entry block
    pub fn bfs_traversal(&self) -> Vec<BlockId> {
        let mut visited = BTreeSet::new();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back(self.entry_block);
        visited.insert(self.entry_block);

        while let Some(block_id) = queue.pop_front() {
            result.push(block_id);

            if let Some(block) = self.get_block(block_id) {
                for &successor in &block.successors {
                    if visited.insert(successor) {
                        queue.push_back(successor);
                    }
                }
            }
        }

        result
    }

    /// Find all unreachable blocks
    pub fn find_unreachable_blocks(&self) -> Vec<BlockId> {
        let reachable: BTreeSet<_> = self.dfs_traversal().into_iter().collect();

        self.blocks
            .keys()
            .filter(|&block_id| !reachable.contains(block_id))
            .copied()
            .collect()
    }

    /// Validate CFG structure and consistency
    pub fn validate(&self) -> Result<(), CfgValidationError> {
        self.validate_with_options(true)
    }

    /// Validate CFG structure with options for strictness
    pub fn validate_with_options(
        &self,
        strict_unreachable_check: bool,
    ) -> Result<(), CfgValidationError> {
        // Check that entry block exists
        if !self.blocks.contains_key(&self.entry_block) {
            return Err(CfgValidationError::MissingEntryBlock);
        }

        // Check successor/predecessor consistency
        for (&block_id, block) in &self.blocks {
            for &successor_id in &block.successors {
                if let Some(successor) = self.blocks.get(&successor_id) {
                    if !successor.predecessors.contains(&block_id) {
                        return Err(CfgValidationError::InconsistentEdges {
                            from: block_id,
                            to: successor_id,
                            description: "successor doesn't list block as predecessor".to_string(),
                        });
                    }
                } else {
                    return Err(CfgValidationError::InvalidSuccessor {
                        block_id,
                        successor_id,
                    });
                }
            }
        }

        // Check for unreachable blocks (optional for strict validation)
        if strict_unreachable_check {
            let unreachable = self.find_unreachable_blocks();
            if !unreachable.is_empty() {
                return Err(CfgValidationError::UnreachableBlocks {
                    block_ids: unreachable,
                });
            }
        }

        Ok(())
    }

    /// Get statistics about this CFG
    pub fn statistics(&self) -> CfgStatistics {
        let block_count = self.blocks.len();
        let edge_count: usize = self
            .blocks
            .values()
            .map(|block| block.successors.len())
            .sum();
        let statement_count: usize = self
            .blocks
            .values()
            .map(|block| block.statements.len())
            .sum();

        let max_loop_depth = self
            .blocks
            .values()
            .map(|block| block.metadata.loop_depth)
            .max()
            .unwrap_or(0);

        CfgStatistics {
            block_count,
            edge_count,
            statement_count,
            max_loop_depth,
            exception_handler_count: self.exception_handlers.len(),
            unreachable_block_count: self.find_unreachable_blocks().len(),
        }
    }
}

/// Statistics about a CFG
#[derive(Debug, Clone)]
pub struct CfgStatistics {
    pub block_count: usize,
    pub edge_count: usize,
    pub statement_count: usize,
    pub max_loop_depth: u32,
    pub exception_handler_count: usize,
    pub unreachable_block_count: usize,
}

/// Errors that can occur during CFG validation
#[derive(Debug, Clone)]
pub enum CfgValidationError {
    /// Entry block is missing from CFG
    MissingEntryBlock,

    /// Successor/predecessor edges are inconsistent
    InconsistentEdges {
        from: BlockId,
        to: BlockId,
        description: String,
    },

    /// Block references non-existent successor
    InvalidSuccessor {
        block_id: BlockId,
        successor_id: BlockId,
    },

    /// CFG contains unreachable blocks
    UnreachableBlocks { block_ids: Vec<BlockId> },
}

impl fmt::Display for CfgValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfgValidationError::MissingEntryBlock => {
                write!(f, "CFG missing entry block")
            }
            CfgValidationError::InconsistentEdges {
                from,
                to,
                description,
            } => {
                write!(
                    f,
                    "Inconsistent edge from {:?} to {:?}: {}",
                    from, to, description
                )
            }
            CfgValidationError::InvalidSuccessor {
                block_id,
                successor_id,
            } => {
                write!(
                    f,
                    "Block {:?} references invalid successor {:?}",
                    block_id, successor_id
                )
            }
            CfgValidationError::UnreachableBlocks { block_ids } => {
                write!(f, "CFG contains unreachable blocks: {:?}", block_ids)
            }
        }
    }
}

impl std::error::Error for CfgValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block_creation() {
        let id = BlockId::from_raw(1);
        let location = SourceLocation::new(1, 1, 1, 1);
        let block = BasicBlock::new(id, location);

        assert_eq!(block.id, id);
        assert!(block.statements.is_empty());
        assert!(block.is_empty());
        assert!(block.is_exit_block()); // No successors initially
    }

    #[test]
    fn test_basic_block_terminator() {
        let id = BlockId::from_raw(1);
        let location = SourceLocation::new(1, 1, 1, 1);
        let mut block = BasicBlock::new(id, location);

        let target = BlockId::from_raw(2);
        block.set_terminator(Terminator::Jump { target });

        assert_eq!(block.successors.len(), 1);
        assert_eq!(block.successors[0], target);
        assert!(!block.is_exit_block());
    }

    #[test]
    fn test_cfg_creation() {
        let function_id = SymbolId::from_raw(1);
        let entry_block = BlockId::from_raw(1);
        let cfg = ControlFlowGraph::new(function_id, entry_block);

        assert_eq!(cfg.function_id, function_id);
        assert_eq!(cfg.entry_block, entry_block);
        assert!(cfg.blocks.is_empty());
    }

    #[test]
    fn test_cfg_traversal() {
        let function_id = SymbolId::from_raw(1);
        let entry_id = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_id);

        // Create entry block
        let location = SourceLocation::new(1, 1, 1, 1);
        let mut entry_block = BasicBlock::new(entry_id, location);
        let target_id = BlockId::from_raw(2);
        entry_block.set_terminator(Terminator::Jump { target: target_id });
        cfg.add_block(entry_block);

        // Create target block
        let mut target_block = BasicBlock::new(target_id, location);
        target_block.set_terminator(Terminator::Return { value: None });
        cfg.add_block(target_block);

        let traversal = cfg.dfs_traversal();
        assert_eq!(traversal.len(), 2);
        assert_eq!(traversal[0], entry_id);
        assert_eq!(traversal[1], target_id);
    }

    #[test]
    fn test_cfg_validation() {
        let function_id = SymbolId::from_raw(1);
        let entry_id = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_id);

        // Empty CFG should fail validation
        assert!(cfg.validate().is_err());

        // Add entry block
        let location = SourceLocation::new(1, 1, 1, 1);
        let mut entry_block = BasicBlock::new(entry_id, location);
        entry_block.set_terminator(Terminator::Return { value: None });
        cfg.add_block(entry_block);

        // Should now pass validation
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_cfg_statistics() {
        let function_id = SymbolId::from_raw(1);
        let entry_id = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_id);

        let location = SourceLocation::new(1, 1, 1, 1);
        let mut block = BasicBlock::new(entry_id, location);
        block.add_statement(StatementId::from_raw(1));
        block.add_statement(StatementId::from_raw(2));
        block.set_terminator(Terminator::Return { value: None });
        cfg.add_block(block);

        let stats = cfg.statistics();
        assert_eq!(stats.block_count, 1);
        assert_eq!(stats.edge_count, 0); // Return has no successors
        assert_eq!(stats.statement_count, 2);
        assert_eq!(stats.unreachable_block_count, 0);
    }
}
