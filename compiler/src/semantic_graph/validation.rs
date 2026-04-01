//! Validation and consistency checking for Control Flow Graphs
//!
//! This module provides deep validation of CFG structure, ensuring:
//! - Graph connectivity and reachability
//! - Terminator consistency
//! - Exception handler validity
//! - Performance characteristics
//! - Haxe-specific semantic correctness

use super::cfg::*;
use super::{BlockId, GraphValidationError};
use crate::tast::{SourceLocation, SymbolId, TypeId};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Comprehensive CFG validator with detailed analysis
pub struct CfgValidator<'a> {
    cfg: &'a ControlFlowGraph,
    visited: BTreeSet<BlockId>,
    reachable: BTreeSet<BlockId>,
    dominance_info: BTreeMap<BlockId, DominanceInfo>,
    loop_info: LoopAnalysisInfo,
}

/// Loop analysis information
#[derive(Debug, Clone, Default)]
pub struct LoopAnalysisInfo {
    /// Natural loops found in the CFG
    pub natural_loops: Vec<NaturalLoop>,
    /// Loop headers (blocks with back edges)
    pub loop_headers: BTreeSet<BlockId>,
    /// Back edges in the CFG
    pub back_edges: Vec<(BlockId, BlockId)>,
}

/// Information about a natural loop
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Header block of the loop
    pub header: BlockId,
    /// All blocks in the loop body
    pub body: BTreeSet<BlockId>,
    /// Nesting depth
    pub depth: u32,
    /// Exit blocks (blocks that leave the loop)
    pub exits: Vec<BlockId>,
}

/// Dominance information for a block
#[derive(Debug, Clone)]
pub struct DominanceInfo {
    /// Immediate dominator
    pub immediate_dominator: Option<BlockId>,
    /// Blocks that this block dominates
    pub dominates: BTreeSet<BlockId>,
    /// Dominance frontier
    pub dominance_frontier: BTreeSet<BlockId>,
}

/// Detailed validation results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    /// Overall validation success
    pub is_valid: bool,
    /// Structural issues found
    pub structural_issues: Vec<StructuralIssue>,
    /// Performance warnings
    pub performance_warnings: Vec<PerformanceWarning>,
    /// Semantic issues
    pub semantic_issues: Vec<SemanticIssue>,
    /// Statistics about the CFG
    pub statistics: ValidationStatistics,
}

/// Structural issues in the CFG
#[derive(Debug, Clone)]
pub enum StructuralIssue {
    /// Unreachable blocks
    UnreachableBlocks { blocks: Vec<BlockId> },
    /// Inconsistent predecessor/successor relationships
    InconsistentEdges {
        from: BlockId,
        to: BlockId,
        issue: String,
    },
    /// Invalid terminator
    InvalidTerminator { block: BlockId, issue: String },
    /// Missing entry or exit blocks
    MissingCriticalBlocks { issue: String },
    /// Disconnected components
    DisconnectedComponents { components: Vec<Vec<BlockId>> },
}

/// Performance warnings
#[derive(Debug, Clone)]
pub enum PerformanceWarning {
    /// Excessive block count
    ExcessiveBlocks { count: usize, threshold: usize },
    /// Deep nesting
    DeepNesting { depth: u32, threshold: u32 },
    /// Large basic blocks
    LargeBlocks { blocks: Vec<(BlockId, usize)> },
    /// Inefficient control flow patterns
    InefficiientPatterns {
        pattern: String,
        blocks: Vec<BlockId>,
    },
}

/// Semantic issues specific to Haxe
#[derive(Debug, Clone)]
pub enum SemanticIssue {
    /// Invalid exception handler configuration
    InvalidExceptionHandler { handler: BlockId, issue: String },
    /// Pattern matching issues
    PatternMatchingIssues { block: BlockId, issue: String },
    /// Loop break/continue target issues
    LoopTargetIssues { block: BlockId, issue: String },
    /// Macro expansion boundary issues
    MacroExpansionIssues { block: BlockId, issue: String },
}

/// Validation statistics
#[derive(Debug, Clone, Default)]
pub struct ValidationStatistics {
    /// Total blocks analyzed
    pub total_blocks: usize,
    /// Reachable blocks
    pub reachable_blocks: usize,
    /// Unreachable blocks
    pub unreachable_blocks: usize,
    /// Total edges
    pub total_edges: usize,
    /// Critical edges (from block with multiple successors to block with multiple predecessors)
    pub critical_edges: usize,
    /// Maximum path length from entry
    pub max_path_length: u32,
    /// Loop nesting depth
    pub max_loop_depth: u32,
    /// Exception handler count
    pub exception_handlers: usize,
}

impl<'a> CfgValidator<'a> {
    /// Create a new validator for the given CFG
    pub fn new(cfg: &'a ControlFlowGraph) -> Self {
        Self {
            cfg,
            visited: BTreeSet::new(),
            reachable: BTreeSet::new(),
            dominance_info: BTreeMap::new(),
            loop_info: LoopAnalysisInfo::default(),
        }
    }

    /// Perform comprehensive validation
    pub fn validate(&mut self) -> ValidationResults {
        let mut results = ValidationResults {
            is_valid: true,
            structural_issues: Vec::new(),
            performance_warnings: Vec::new(),
            semantic_issues: Vec::new(),
            statistics: ValidationStatistics::default(),
        };

        // Basic structural validation
        self.validate_structure(&mut results);

        // Reachability analysis
        self.analyze_reachability(&mut results);

        // Dominance analysis
        self.compute_dominance(&mut results);

        // Loop analysis
        self.analyze_loops(&mut results);

        // Exception handler validation
        self.validate_exception_handlers(&mut results);

        // Performance analysis
        self.analyze_performance(&mut results);

        // Semantic validation
        self.validate_semantics(&mut results);

        // Compute final statistics
        self.compute_statistics(&mut results);

        // Overall validity
        results.is_valid =
            results.structural_issues.is_empty() && results.semantic_issues.is_empty();

        results
    }

    /// Validate basic CFG structure
    fn validate_structure(&mut self, results: &mut ValidationResults) {
        // Check entry block exists
        if !self.cfg.blocks.contains_key(&self.cfg.entry_block) {
            results
                .structural_issues
                .push(StructuralIssue::MissingCriticalBlocks {
                    issue: "Entry block missing from CFG".to_string(),
                });
            return;
        }

        // Check predecessor/successor consistency
        for (&block_id, block) in &self.cfg.blocks {
            for &successor_id in &block.successors {
                if let Some(successor) = self.cfg.blocks.get(&successor_id) {
                    if !successor.predecessors.contains(&block_id) {
                        results
                            .structural_issues
                            .push(StructuralIssue::InconsistentEdges {
                                from: block_id,
                                to: successor_id,
                                issue: "Successor doesn't list block as predecessor".to_string(),
                            });
                    }
                } else {
                    results
                        .structural_issues
                        .push(StructuralIssue::InconsistentEdges {
                            from: block_id,
                            to: successor_id,
                            issue: "Successor block doesn't exist".to_string(),
                        });
                }
            }

            // Validate terminator consistency with successors
            self.validate_terminator_consistency(block_id, block, results);
        }

        // Check for exit blocks
        if self.cfg.exit_blocks.is_empty() {
            // Look for blocks that actually exit
            let actual_exits: Vec<_> = self
                .cfg
                .blocks
                .iter()
                .filter(|(_, block)| block.is_exit_block())
                .map(|(id, _)| *id)
                .collect();

            if actual_exits.is_empty() {
                results
                    .structural_issues
                    .push(StructuralIssue::MissingCriticalBlocks {
                        issue: "No exit blocks found".to_string(),
                    });
            }
        }
    }

    /// Validate terminator consistency
    fn validate_terminator_consistency(
        &self,
        block_id: BlockId,
        block: &BasicBlock,
        results: &mut ValidationResults,
    ) {
        let expected_successors = match &block.terminator {
            Terminator::Jump { target } => vec![*target],
            Terminator::Branch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            Terminator::Switch {
                targets,
                default_target,
                ..
            } => {
                let mut successors: Vec<_> = targets.iter().map(|t| t.target).collect();
                if let Some(default) = default_target {
                    successors.push(*default);
                }
                successors
            }
            Terminator::PatternMatch {
                patterns,
                default_target,
                ..
            } => {
                let mut successors: Vec<_> = patterns.iter().map(|p| p.target).collect();
                if let Some(default) = default_target {
                    successors.push(*default);
                }
                successors
            }
            Terminator::MacroExpansion { target, .. } => vec![*target],
            Terminator::Return { .. } | Terminator::Throw { .. } | Terminator::Unreachable => {
                vec![]
            }
        };

        // Check that successors match terminator
        let mut actual_successors = block.successors.clone();
        actual_successors.sort();
        let mut expected_sorted = expected_successors;
        expected_sorted.sort();
        expected_sorted.dedup();

        if actual_successors != expected_sorted {
            results
                .structural_issues
                .push(StructuralIssue::InvalidTerminator {
                    block: block_id,
                    issue: format!(
                        "Successors {:?} don't match terminator {:?}",
                        actual_successors, expected_sorted
                    ),
                });
        }
    }

    /// Analyze reachability from entry block
    fn analyze_reachability(&mut self, results: &mut ValidationResults) {
        self.reachable.clear();
        let mut queue = VecDeque::new();

        // Only start reachability analysis if entry block exists
        if self.cfg.blocks.contains_key(&self.cfg.entry_block) {
            queue.push_back(self.cfg.entry_block);
            self.reachable.insert(self.cfg.entry_block);

            while let Some(block_id) = queue.pop_front() {
                if let Some(block) = self.cfg.blocks.get(&block_id) {
                    for &successor in &block.successors {
                        if self.reachable.insert(successor) {
                            queue.push_back(successor);
                        }
                    }
                }
            }
        }

        // Find unreachable blocks
        let unreachable: Vec<_> = self
            .cfg
            .blocks
            .keys()
            .filter(|&id| !self.reachable.contains(id))
            .copied()
            .collect();

        if !unreachable.is_empty() {
            results
                .structural_issues
                .push(StructuralIssue::UnreachableBlocks {
                    blocks: unreachable,
                });
        }
    }

    /// Compute dominance information
    fn compute_dominance(&mut self, _results: &mut ValidationResults) {
        // Simplified dominance computation for validation
        // In a full implementation, this would use the standard dominance algorithm

        for &block_id in &self.reachable {
            let dominance = DominanceInfo {
                immediate_dominator: if block_id == self.cfg.entry_block {
                    None
                } else {
                    Some(self.cfg.entry_block) // Simplified
                },
                dominates: BTreeSet::new(),
                dominance_frontier: BTreeSet::new(),
            };
            self.dominance_info.insert(block_id, dominance);
        }
    }

    /// Analyze loop structure
    fn analyze_loops(&mut self, results: &mut ValidationResults) {
        // Find back edges (edges that go to a dominator)
        for (&block_id, block) in &self.cfg.blocks {
            for &successor in &block.successors {
                if self.dominates(successor, block_id) {
                    self.loop_info.back_edges.push((block_id, successor));
                    self.loop_info.loop_headers.insert(successor);
                }
            }
        }

        // Build natural loops for each back edge
        for &(tail, header) in &self.loop_info.back_edges {
            let loop_body = self.compute_natural_loop(header, tail);
            let natural_loop = NaturalLoop {
                header,
                body: loop_body,
                depth: 1,      // Simplified
                exits: vec![], // Would compute actual exits
            };
            self.loop_info.natural_loops.push(natural_loop);
        }
    }

    /// Check if block1 dominates block2
    fn dominates(&self, block1: BlockId, block2: BlockId) -> bool {
        // Simplified dominance check
        // In practice, would use proper dominance computation
        block1 == self.cfg.entry_block || block1 == block2
    }

    /// Compute natural loop body
    fn compute_natural_loop(&self, header: BlockId, tail: BlockId) -> BTreeSet<BlockId> {
        let mut loop_body = BTreeSet::new();
        let mut worklist = VecDeque::new();

        loop_body.insert(header);
        loop_body.insert(tail);
        worklist.push_back(tail);

        while let Some(block_id) = worklist.pop_front() {
            if let Some(block) = self.cfg.blocks.get(&block_id) {
                for &pred in &block.predecessors {
                    if loop_body.insert(pred) && pred != header {
                        worklist.push_back(pred);
                    }
                }
            }
        }

        loop_body
    }

    /// Validate exception handlers
    fn validate_exception_handlers(&self, results: &mut ValidationResults) {
        for (&handler_block, handler_info) in &self.cfg.exception_handlers {
            // Check handler block exists
            if !self.cfg.blocks.contains_key(&handler_block) {
                results
                    .semantic_issues
                    .push(SemanticIssue::InvalidExceptionHandler {
                        handler: handler_block,
                        issue: "Handler block doesn't exist".to_string(),
                    });
                continue;
            }

            // Check covered blocks exist
            for &covered_block in &handler_info.covered_blocks {
                if !self.cfg.blocks.contains_key(&covered_block) {
                    results
                        .semantic_issues
                        .push(SemanticIssue::InvalidExceptionHandler {
                            handler: handler_block,
                            issue: format!("Covered block {:?} doesn't exist", covered_block),
                        });
                }
            }

            // Check exception types are valid
            if handler_info.exception_types.is_empty() {
                results
                    .semantic_issues
                    .push(SemanticIssue::InvalidExceptionHandler {
                        handler: handler_block,
                        issue: "No exception types specified".to_string(),
                    });
            }
        }
    }

    /// Analyze performance characteristics
    fn analyze_performance(&self, results: &mut ValidationResults) {
        const MAX_BLOCKS_WARNING: usize = 1000;
        const MAX_STATEMENTS_PER_BLOCK: usize = 100;
        const MAX_NESTING_WARNING: u32 = 10;

        // Check for excessive block count
        if self.cfg.blocks.len() > MAX_BLOCKS_WARNING {
            results
                .performance_warnings
                .push(PerformanceWarning::ExcessiveBlocks {
                    count: self.cfg.blocks.len(),
                    threshold: MAX_BLOCKS_WARNING,
                });
        }

        // Check for large blocks
        let large_blocks: Vec<_> = self
            .cfg
            .blocks
            .iter()
            .filter(|(_, block)| block.statement_count() > MAX_STATEMENTS_PER_BLOCK)
            .map(|(id, block)| (*id, block.statement_count()))
            .collect();

        if !large_blocks.is_empty() {
            results
                .performance_warnings
                .push(PerformanceWarning::LargeBlocks {
                    blocks: large_blocks,
                });
        }

        // Check for deep nesting
        let max_depth = self
            .cfg
            .blocks
            .values()
            .map(|block| block.metadata.loop_depth)
            .max()
            .unwrap_or(0);

        if max_depth > MAX_NESTING_WARNING {
            results
                .performance_warnings
                .push(PerformanceWarning::DeepNesting {
                    depth: max_depth,
                    threshold: MAX_NESTING_WARNING,
                });
        }
    }

    /// Validate Haxe-specific semantics
    fn validate_semantics(&self, results: &mut ValidationResults) {
        for (&block_id, block) in &self.cfg.blocks {
            match &block.terminator {
                Terminator::PatternMatch { patterns, .. } => {
                    if patterns.is_empty() {
                        results
                            .semantic_issues
                            .push(SemanticIssue::PatternMatchingIssues {
                                block: block_id,
                                issue: "Pattern match with no patterns".to_string(),
                            });
                    }
                }

                Terminator::Switch { targets, .. } => {
                    // Check for duplicate case values (simplified)
                    if targets.len() > 1 {
                        // In practice, would check for duplicate case values
                    }
                }

                Terminator::MacroExpansion { macro_info, .. } => {
                    // Validate macro expansion context
                    if macro_info.expansion_context.is_empty() {
                        results
                            .semantic_issues
                            .push(SemanticIssue::MacroExpansionIssues {
                                block: block_id,
                                issue: "Empty macro expansion context".to_string(),
                            });
                    }
                }

                _ => {}
            }
        }
    }

    /// Compute final validation statistics
    fn compute_statistics(&self, results: &mut ValidationResults) {
        results.statistics.total_blocks = self.cfg.blocks.len();
        results.statistics.reachable_blocks = self.reachable.len();
        results.statistics.unreachable_blocks = self.cfg.blocks.len() - self.reachable.len();

        results.statistics.total_edges = self
            .cfg
            .blocks
            .values()
            .map(|block| block.successors.len())
            .sum();

        results.statistics.max_loop_depth = self
            .cfg
            .blocks
            .values()
            .map(|block| block.metadata.loop_depth)
            .max()
            .unwrap_or(0);

        results.statistics.exception_handlers = self.cfg.exception_handlers.len();

        // Compute critical edges
        results.statistics.critical_edges = self.compute_critical_edges();

        // Compute maximum path length
        results.statistics.max_path_length = self.compute_max_path_length();
    }

    /// Compute number of critical edges
    fn compute_critical_edges(&self) -> usize {
        let mut critical_count = 0;

        for block in self.cfg.blocks.values() {
            if block.successors.len() > 1 {
                for &successor_id in &block.successors {
                    if let Some(successor) = self.cfg.blocks.get(&successor_id) {
                        if successor.predecessors.len() > 1 {
                            critical_count += 1;
                        }
                    }
                }
            }
        }

        critical_count
    }

    /// Compute maximum path length from entry
    fn compute_max_path_length(&self) -> u32 {
        let mut max_length = 0;
        let mut visited = BTreeSet::new();

        fn dfs(
            cfg: &ControlFlowGraph,
            block_id: BlockId,
            current_length: u32,
            visited: &mut BTreeSet<BlockId>,
            max_length: &mut u32,
        ) {
            if visited.contains(&block_id) {
                return; // Avoid cycles
            }

            visited.insert(block_id);
            *max_length = (*max_length).max(current_length);

            if let Some(block) = cfg.blocks.get(&block_id) {
                for &successor in &block.successors {
                    dfs(cfg, successor, current_length + 1, visited, max_length);
                }
            }

            visited.remove(&block_id);
        }

        dfs(
            self.cfg,
            self.cfg.entry_block,
            0,
            &mut visited,
            &mut max_length,
        );
        max_length
    }
}

/// Pretty-print validation results
impl ValidationResults {
    /// Print a comprehensive validation report
    pub fn print_report(&self) {
        println!("🔍 CFG Validation Report");
        println!("========================");

        if self.is_valid {
            println!("✅ CFG is structurally valid");
        } else {
            println!("❌ CFG has validation issues");
        }

        println!("\n📊 Statistics:");
        println!("   Total blocks: {}", self.statistics.total_blocks);
        println!("   Reachable blocks: {}", self.statistics.reachable_blocks);
        println!(
            "   Unreachable blocks: {}",
            self.statistics.unreachable_blocks
        );
        println!("   Total edges: {}", self.statistics.total_edges);
        println!("   Critical edges: {}", self.statistics.critical_edges);
        println!("   Max path length: {}", self.statistics.max_path_length);
        println!("   Max loop depth: {}", self.statistics.max_loop_depth);
        println!(
            "   Exception handlers: {}",
            self.statistics.exception_handlers
        );

        if !self.structural_issues.is_empty() {
            println!("\n🔧 Structural Issues:");
            for issue in &self.structural_issues {
                println!("   • {:?}", issue);
            }
        }

        if !self.semantic_issues.is_empty() {
            println!("\n🎯 Semantic Issues:");
            for issue in &self.semantic_issues {
                println!("   • {:?}", issue);
            }
        }

        if !self.performance_warnings.is_empty() {
            println!("\n⚡ Performance Warnings:");
            for warning in &self.performance_warnings {
                println!("   • {:?}", warning);
            }
        }

        println!();
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use crate::semantic_graph::cfg::*;
    use crate::tast::*;

    fn create_simple_cfg() -> ControlFlowGraph {
        let function_id = SymbolId::from_raw(1);
        let entry_block = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_block);

        let mut block = BasicBlock::new(entry_block, SourceLocation::unknown());
        block.set_terminator(Terminator::Return { value: None });
        cfg.add_block(block);

        cfg
    }

    #[test]
    fn test_valid_cfg() {
        let cfg = create_simple_cfg();
        let mut validator = CfgValidator::new(&cfg);
        let results = validator.validate();

        assert!(results.is_valid);
        assert!(results.structural_issues.is_empty());
        assert!(results.semantic_issues.is_empty());
        assert_eq!(results.statistics.total_blocks, 1);
        assert_eq!(results.statistics.reachable_blocks, 1);
        assert_eq!(results.statistics.unreachable_blocks, 0);
    }

    #[test]
    fn test_invalid_cfg_missing_entry() {
        let function_id = SymbolId::from_raw(1);
        let entry_block = BlockId::from_raw(999); // Non-existent
        let cfg = ControlFlowGraph::new(function_id, entry_block);

        let mut validator = CfgValidator::new(&cfg);
        let results = validator.validate();

        assert!(!results.is_valid);
        assert!(!results.structural_issues.is_empty());

        // Should detect missing entry block
        assert!(results
            .structural_issues
            .iter()
            .any(|issue| { matches!(issue, StructuralIssue::MissingCriticalBlocks { .. }) }));
    }

    #[test]
    fn test_unreachable_blocks() {
        let function_id = SymbolId::from_raw(1);
        let entry_block = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_block);

        // Add entry block
        let mut entry = BasicBlock::new(entry_block, SourceLocation::unknown());
        entry.set_terminator(Terminator::Return { value: None });
        cfg.add_block(entry);

        // Add unreachable block
        let unreachable_block = BlockId::from_raw(2);
        let mut unreachable = BasicBlock::new(unreachable_block, SourceLocation::unknown());
        unreachable.set_terminator(Terminator::Return { value: None });
        cfg.add_block(unreachable);

        let mut validator = CfgValidator::new(&cfg);
        let results = validator.validate();

        assert!(!results.is_valid);
        assert_eq!(results.statistics.unreachable_blocks, 1);

        // Should detect unreachable block
        assert!(results.structural_issues.iter().any(|issue| {
            matches!(issue, StructuralIssue::UnreachableBlocks { blocks } if blocks.contains(&unreachable_block))
        }));
    }

    #[test]
    fn test_performance_warnings() {
        let function_id = SymbolId::from_raw(1);
        let entry_block = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_block);

        let mut block = BasicBlock::new(entry_block, SourceLocation::unknown());

        // Add many statements to trigger large block warning
        for i in 0..150 {
            block.add_statement(crate::tast::id_types::StatementId::from_raw(i));
        }

        block.set_terminator(Terminator::Return { value: None });
        cfg.add_block(block);

        let mut validator = CfgValidator::new(&cfg);
        let results = validator.validate();

        // Should have performance warning for large block
        assert!(!results.performance_warnings.is_empty());
        assert!(results
            .performance_warnings
            .iter()
            .any(|warning| { matches!(warning, PerformanceWarning::LargeBlocks { .. }) }));
    }
}
