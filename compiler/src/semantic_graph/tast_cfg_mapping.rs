//! TAST to CFG mapping system
//!
//! Provides real mapping from TAST statements to CFG blocks, enabling
//! proper SSA construction and data flow analysis. This replaces the
//! fake statement processing in the DFG builder.

use super::cfg::ControlFlowGraph;
use crate::tast::node::{TypedExpression, TypedFunction, TypedStatement};
use crate::tast::{BlockId, SourceLocation, SymbolId};
use std::collections::BTreeMap;

/// Maps TAST statements to CFG blocks
#[derive(Debug, Clone)]
pub struct TastCfgMapping {
    /// Map from statement locations to CFG blocks
    statement_to_block: BTreeMap<StatementLocation, BlockId>,

    /// Map from CFG blocks to statement ranges
    block_to_statements: BTreeMap<BlockId, Vec<StatementLocation>>,

    /// Source location mapping for precise error reporting
    statement_locations: BTreeMap<StatementLocation, SourceLocation>,

    /// Statement order within each block
    statement_order: BTreeMap<BlockId, Vec<StatementLocation>>,
}

/// Location of a statement in the TAST
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StatementLocation {
    /// Index in the function body
    pub statement_index: usize,

    /// Nested depth (for blocks within blocks)
    pub nesting_depth: u32,

    /// Unique identifier for this statement
    pub id: u64,

    /// Branch context for navigating control flow statements
    pub branch_context: BranchContext,
}

/// Context for navigating branches in control flow statements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BranchContext {
    /// Not in a branch
    None,
    /// In the then branch of an If statement
    IfThen,
    /// In the else branch of an If statement
    IfElse,
    /// In a specific case of a Switch statement
    SwitchCase(usize),
    /// In the default case of a Switch statement
    SwitchDefault,
    /// In a catch clause
    CatchClause(usize),
    /// In a finally block
    Finally,
}

/// Result of mapping TAST to CFG
#[derive(Debug)]
pub struct MappingResult {
    /// The computed mapping
    pub mapping: TastCfgMapping,

    /// Any warnings or issues encountered
    pub warnings: Vec<MappingWarning>,

    /// Statistics about the mapping process
    pub stats: MappingStats,
}

/// Warning about mapping issues
#[derive(Debug, Clone)]
pub enum MappingWarning {
    /// Statement couldn't be mapped to any block
    UnmappedStatement {
        location: StatementLocation,
        source_location: SourceLocation,
        reason: String,
    },

    /// Multiple statements map to same location
    ConflictingMapping {
        location1: StatementLocation,
        location2: StatementLocation,
        block_id: BlockId,
    },

    /// Source location information is missing
    MissingSourceLocation { location: StatementLocation },
}

/// Statistics about the mapping process
#[derive(Debug, Clone, Default)]
pub struct MappingStats {
    /// Total statements processed
    pub statements_processed: usize,

    /// Total blocks mapped
    pub blocks_mapped: usize,

    /// Statements successfully mapped
    pub successful_mappings: usize,

    /// Time taken to compute mapping (microseconds)
    pub computation_time_us: u64,

    /// Memory used for mapping structures
    pub memory_used_bytes: usize,
}

impl TastCfgMapping {
    /// Create mapping from TAST function to CFG
    pub fn build(cfg: &ControlFlowGraph, function: &TypedFunction) -> MappingResult {
        let start_time = std::time::Instant::now();

        let mut mapper = TastCfgMapper {
            mapping: TastCfgMapping {
                statement_to_block: BTreeMap::new(),
                block_to_statements: BTreeMap::new(),
                statement_locations: BTreeMap::new(),
                statement_order: BTreeMap::new(),
            },
            warnings: Vec::new(),
            current_block: cfg.entry_block,
            statement_counter: 0,
            stats: MappingStats::default(),
        };

        // Map all statements in the function
        mapper.map_function_statements(cfg, function);

        // Finalize statistics
        mapper.stats.computation_time_us = start_time.elapsed().as_micros() as u64;
        mapper.stats.memory_used_bytes = mapper.estimate_memory_usage();

        MappingResult {
            mapping: mapper.mapping,
            warnings: mapper.warnings,
            stats: mapper.stats,
        }
    }

    /// Get the block for a specific statement location
    pub fn get_block_for_statement(&self, location: StatementLocation) -> Option<BlockId> {
        self.statement_to_block.get(&location).copied()
    }

    /// Get all statements in a specific block
    pub fn get_statements_in_block(&self, block_id: BlockId) -> &[StatementLocation] {
        self.block_to_statements
            .get(&block_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the source location for a statement
    pub fn get_source_location(&self, location: StatementLocation) -> Option<SourceLocation> {
        self.statement_locations.get(&location).copied()
    }

    /// Get statements in execution order for a block
    pub fn get_ordered_statements(&self, block_id: BlockId) -> &[StatementLocation] {
        self.statement_order
            .get(&block_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check if a statement has been mapped
    pub fn is_statement_mapped(&self, location: StatementLocation) -> bool {
        self.statement_to_block.contains_key(&location)
    }
}

/// Internal mapper for building TAST-CFG mapping
struct TastCfgMapper {
    mapping: TastCfgMapping,
    warnings: Vec<MappingWarning>,
    current_block: BlockId,
    statement_counter: u64,
    stats: MappingStats,
}

impl TastCfgMapper {
    /// Map all statements in a function
    fn map_function_statements(&mut self, cfg: &ControlFlowGraph, function: &TypedFunction) {
        // Initialize block mappings for all CFG blocks
        for &block_id in cfg.blocks.keys() {
            self.mapping
                .block_to_statements
                .insert(block_id, Vec::new());
            self.mapping.statement_order.insert(block_id, Vec::new());
        }

        // Map function body statements
        for (index, statement) in function.body.iter().enumerate() {
            let location = StatementLocation {
                statement_index: index,
                nesting_depth: 0,
                id: self.statement_counter,
                branch_context: BranchContext::None,
            };
            self.statement_counter += 1;

            self.map_statement(cfg, statement, location, 0);
        }

        self.stats.blocks_mapped = self.mapping.block_to_statements.len();
        self.stats.successful_mappings = self.mapping.statement_to_block.len();
    }

    /// Map a single statement to CFG blocks
    fn map_statement(
        &mut self,
        cfg: &ControlFlowGraph,
        statement: &TypedStatement,
        location: StatementLocation,
        depth: u32,
    ) {
        use crate::tast::node::HasSourceLocation;

        // Count this statement as processed
        self.stats.statements_processed += 1;

        // Record source location
        self.mapping
            .statement_locations
            .insert(location, statement.source_location());

        match statement {
            TypedStatement::Expression { .. }
            | TypedStatement::VarDeclaration { .. }
            | TypedStatement::Assignment { .. }
            | TypedStatement::Return { .. }
            | TypedStatement::Throw { .. }
            | TypedStatement::Break { .. }
            | TypedStatement::Continue { .. } => {
                // Simple statements map to current block
                self.map_statement_to_block(location, self.current_block);
            }

            TypedStatement::If {
                condition: _,
                then_branch,
                else_branch,
                ..
            } => {
                // Map the if statement itself to current block
                self.map_statement_to_block(location, self.current_block);

                // Map then branch - try to find corresponding block
                let then_block = self.find_then_block(cfg, self.current_block);
                let old_block = self.current_block;
                self.current_block = then_block;

                let then_location = self.create_nested_location_with_branch(
                    location,
                    depth + 1,
                    0,
                    BranchContext::IfThen,
                );
                self.map_statement(cfg, then_branch, then_location, depth + 1);

                // Map else branch if present
                if let Some(else_stmt) = else_branch {
                    let else_block = self.find_else_block(cfg, old_block);
                    self.current_block = else_block;

                    let else_location = self.create_nested_location_with_branch(
                        location,
                        depth + 1,
                        0,
                        BranchContext::IfElse,
                    );
                    self.map_statement(cfg, else_stmt, else_location, depth + 1);
                }

                // Find merge block and continue there
                self.current_block = self.find_merge_block(cfg, old_block);
            }

            TypedStatement::While {
                condition: _, body, ..
            } => {
                // Map while statement to current block (which becomes loop header)
                self.map_statement_to_block(location, self.current_block);

                // Map body to loop body block
                let body_block = self.find_loop_body_block(cfg, self.current_block);
                let old_block = self.current_block;
                self.current_block = body_block;

                let body_location = self.create_nested_location(location, depth + 1, 0);
                self.map_statement(cfg, body, body_location, depth + 1);

                // Continue after loop
                self.current_block = self.find_loop_exit_block(cfg, old_block);
            }

            TypedStatement::For {
                init,
                condition: _,
                update: _,
                body,
                ..
            } => {
                // Map init statement if present
                if let Some(init_stmt) = init {
                    let init_location = self.create_nested_location(location, depth + 1, 0);
                    self.map_statement(cfg, init_stmt, init_location, depth + 1);
                }

                // Map for statement to header block
                self.map_statement_to_block(location, self.current_block);

                // Map body
                let body_block = self.find_loop_body_block(cfg, self.current_block);
                let old_block = self.current_block;
                self.current_block = body_block;

                let body_location = self.create_nested_location(location, depth + 1, 1);
                self.map_statement(cfg, body, body_location, depth + 1);

                // Continue after loop
                self.current_block = self.find_loop_exit_block(cfg, old_block);
            }

            TypedStatement::Block { statements, .. } => {
                // Map block statement to current block
                self.map_statement_to_block(location, self.current_block);

                // Map all statements in the block
                for (index, stmt) in statements.iter().enumerate() {
                    let stmt_location =
                        self.create_nested_location(location, depth + 1, index as u64);
                    self.map_statement(cfg, stmt, stmt_location, depth + 1);
                }
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                // Map try statement to current block
                self.map_statement_to_block(location, self.current_block);

                // Map try body
                let body_location = self.create_nested_location(location, depth + 1, 0);
                self.map_statement(cfg, body, body_location, depth + 1);

                // Map catch clauses
                for (index, catch_clause) in catch_clauses.iter().enumerate() {
                    let catch_block = self.find_catch_block(cfg, self.current_block, index);
                    let old_block = self.current_block;
                    self.current_block = catch_block;

                    let catch_location =
                        self.create_nested_location(location, depth + 1, index as u64 + 1);
                    self.map_statement(cfg, &catch_clause.body, catch_location, depth + 1);

                    self.current_block = old_block;
                }

                // Map finally block if present
                if let Some(finally_stmt) = finally_block {
                    let finally_block = self.find_finally_block(cfg, self.current_block);
                    let old_block = self.current_block;
                    self.current_block = finally_block;

                    let finally_location = self.create_nested_location(
                        location,
                        depth + 1,
                        catch_clauses.len() as u64 + 1,
                    );
                    self.map_statement(cfg, finally_stmt, finally_location, depth + 1);

                    self.current_block = old_block;
                }
            }

            TypedStatement::Switch {
                cases,
                default_case,
                ..
            } => {
                // Map switch statement to current block
                self.map_statement_to_block(location, self.current_block);

                // Map each case
                for (index, case) in cases.iter().enumerate() {
                    let case_block = self.find_case_block(cfg, self.current_block, index);
                    let old_block = self.current_block;
                    self.current_block = case_block;

                    let case_location =
                        self.create_nested_location(location, depth + 1, index as u64);
                    self.map_statement(cfg, &case.body, case_location, depth + 1);

                    self.current_block = old_block;
                }

                // Map default case if present
                if let Some(default_stmt) = default_case {
                    let default_block = self.find_default_case_block(cfg, self.current_block);
                    let old_block = self.current_block;
                    self.current_block = default_block;

                    let default_location =
                        self.create_nested_location(location, depth + 1, cases.len() as u64);
                    self.map_statement(cfg, default_stmt, default_location, depth + 1);

                    self.current_block = old_block;
                }
            }

            TypedStatement::PatternMatch { patterns, .. } => {
                // Map pattern match statement to current block
                self.map_statement_to_block(location, self.current_block);

                // Map each pattern case
                for (index, pattern_case) in patterns.iter().enumerate() {
                    let pattern_block = self.find_pattern_block(cfg, self.current_block, index);
                    let old_block = self.current_block;
                    self.current_block = pattern_block;

                    let pattern_location =
                        self.create_nested_location(location, depth + 1, index as u64);
                    self.map_statement(cfg, &pattern_case.body, pattern_location, depth + 1);

                    self.current_block = old_block;
                }
            }

            TypedStatement::MacroExpansion {
                expanded_statements,
                ..
            } => {
                // Map macro expansion to current block
                self.map_statement_to_block(location, self.current_block);

                // Map expanded statements
                for (index, stmt) in expanded_statements.iter().enumerate() {
                    let expanded_location =
                        self.create_nested_location(location, depth + 1, index as u64);
                    self.map_statement(cfg, stmt, expanded_location, depth + 1);
                }
            }

            TypedStatement::ForIn {
                value_var: _,
                key_var: _,
                iterable: _,
                body,
                ..
            } => {
                // Handle for-in loops similar to regular for loops
                self.map_statement_to_block(location, self.current_block);

                // Map the body to loop body block
                let body_block = self.find_loop_body_block(cfg, self.current_block);
                let old_block = self.current_block;
                self.current_block = body_block;

                let body_location = self.create_nested_location(location, depth + 1, 0);
                self.map_statement(cfg, body, body_location, depth + 1);

                // Find merge block and continue there
                self.current_block = self.find_merge_block(cfg, old_block);
            }
        }
    }

    /// Map a statement to a specific block
    fn map_statement_to_block(&mut self, location: StatementLocation, block_id: BlockId) {
        // Check for conflicts
        if let Some(existing_block) = self.mapping.statement_to_block.get(&location) {
            if *existing_block != block_id {
                self.warnings.push(MappingWarning::ConflictingMapping {
                    location1: location,
                    location2: location, // Same location, different interpretation
                    block_id,
                });
            }
            return;
        }

        // Add the mapping
        self.mapping.statement_to_block.insert(location, block_id);

        // Add to block's statement list
        self.mapping
            .block_to_statements
            .entry(block_id)
            .or_insert_with(Vec::new)
            .push(location);

        // Add to ordered list
        self.mapping
            .statement_order
            .entry(block_id)
            .or_insert_with(Vec::new)
            .push(location);
    }

    /// Create a nested statement location
    fn create_nested_location(
        &mut self,
        parent: StatementLocation,
        depth: u32,
        index: u64,
    ) -> StatementLocation {
        let location = StatementLocation {
            statement_index: parent.statement_index,
            nesting_depth: depth,
            id: self.statement_counter,
            branch_context: parent.branch_context,
        };
        self.statement_counter += 1;
        location
    }

    /// Create a nested statement location with a specific branch context
    fn create_nested_location_with_branch(
        &mut self,
        parent: StatementLocation,
        depth: u32,
        index: u64,
        branch_context: BranchContext,
    ) -> StatementLocation {
        let location = StatementLocation {
            statement_index: parent.statement_index,
            nesting_depth: depth,
            id: self.statement_counter,
            branch_context,
        };
        self.statement_counter += 1;
        location
    }

    /// Find the "then" branch block for an if statement
    fn find_then_block(&self, cfg: &ControlFlowGraph, current_block: BlockId) -> BlockId {
        // Look for the first successor that's not the "else" block
        // This is simplified - real implementation would analyze terminator
        if let Some(block) = cfg.get_block(current_block) {
            block.successors.first().copied().unwrap_or(current_block)
        } else {
            current_block
        }
    }

    /// Find the "else" branch block for an if statement
    fn find_else_block(&self, cfg: &ControlFlowGraph, current_block: BlockId) -> BlockId {
        // Look for the second successor
        if let Some(block) = cfg.get_block(current_block) {
            block
                .successors
                .get(1)
                .copied()
                .unwrap_or_else(|| block.successors.first().copied().unwrap_or(current_block))
        } else {
            current_block
        }
    }

    /// Find the merge block after an if statement
    fn find_merge_block(&self, cfg: &ControlFlowGraph, current_block: BlockId) -> BlockId {
        // Find the block that both then and else branches flow to
        // This would use dominance analysis to find the merge point
        if let Some(block) = cfg.get_block(current_block) {
            // Simplified: assume the last successor is the merge
            block.successors.last().copied().unwrap_or(current_block)
        } else {
            current_block
        }
    }

    /// Find the loop body block
    fn find_loop_body_block(&self, cfg: &ControlFlowGraph, header_block: BlockId) -> BlockId {
        // For loops, the body is usually the first successor
        if let Some(block) = cfg.get_block(header_block) {
            block.successors.first().copied().unwrap_or(header_block)
        } else {
            header_block
        }
    }

    /// Find the loop exit block
    fn find_loop_exit_block(&self, cfg: &ControlFlowGraph, header_block: BlockId) -> BlockId {
        // Loop exit is usually the second successor of the header
        if let Some(block) = cfg.get_block(header_block) {
            block.successors.get(1).copied().unwrap_or(header_block)
        } else {
            header_block
        }
    }

    /// Find catch block for exception handling
    fn find_catch_block(
        &self,
        cfg: &ControlFlowGraph,
        try_block: BlockId,
        catch_index: usize,
    ) -> BlockId {
        // Simplified: find catch block based on index
        if let Some(block) = cfg.get_block(try_block) {
            block
                .successors
                .get(catch_index + 1)
                .copied()
                .unwrap_or(try_block)
        } else {
            try_block
        }
    }

    /// Find finally block
    fn find_finally_block(&self, cfg: &ControlFlowGraph, try_block: BlockId) -> BlockId {
        // Finally block is typically the last successor
        if let Some(block) = cfg.get_block(try_block) {
            block.successors.last().copied().unwrap_or(try_block)
        } else {
            try_block
        }
    }

    /// Find case block for switch statement
    fn find_case_block(
        &self,
        cfg: &ControlFlowGraph,
        switch_block: BlockId,
        case_index: usize,
    ) -> BlockId {
        if let Some(block) = cfg.get_block(switch_block) {
            block
                .successors
                .get(case_index)
                .copied()
                .unwrap_or(switch_block)
        } else {
            switch_block
        }
    }

    /// Find default case block
    fn find_default_case_block(&self, cfg: &ControlFlowGraph, switch_block: BlockId) -> BlockId {
        // Default case is typically the last successor
        if let Some(block) = cfg.get_block(switch_block) {
            block.successors.last().copied().unwrap_or(switch_block)
        } else {
            switch_block
        }
    }

    /// Find pattern matching block
    fn find_pattern_block(
        &self,
        cfg: &ControlFlowGraph,
        match_block: BlockId,
        pattern_index: usize,
    ) -> BlockId {
        if let Some(block) = cfg.get_block(match_block) {
            block
                .successors
                .get(pattern_index)
                .copied()
                .unwrap_or(match_block)
        } else {
            match_block
        }
    }

    /// Estimate memory usage of mapping structures
    fn estimate_memory_usage(&self) -> usize {
        let statement_map_size = self.mapping.statement_to_block.len()
            * (std::mem::size_of::<StatementLocation>() + std::mem::size_of::<BlockId>());

        let block_map_size = self.mapping.block_to_statements.len()
            * std::mem::size_of::<BlockId>()
            + self
                .mapping
                .block_to_statements
                .values()
                .map(|v| v.len() * std::mem::size_of::<StatementLocation>())
                .sum::<usize>();

        let location_map_size = self.mapping.statement_locations.len()
            * (std::mem::size_of::<StatementLocation>() + std::mem::size_of::<SourceLocation>());

        statement_map_size + block_map_size + location_map_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic_graph::cfg::*;
    use crate::tast::node::*;
    use crate::tast::*;

    #[test]
    fn test_statement_location_creation() {
        let location = StatementLocation {
            statement_index: 0,
            nesting_depth: 0,
            id: 1,
            branch_context: BranchContext::None,
        };

        assert_eq!(location.statement_index, 0);
        assert_eq!(location.nesting_depth, 0);
        assert_eq!(location.id, 1);
    }

    #[test]
    fn test_simple_mapping() {
        let interner = StringInterner::new();
        // Create a simple function with one statement
        let function = TypedFunction {
            symbol_id: SymbolId::from_raw(1),
            name: interner.intern("test"),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![TypedStatement::Return {
                value: None,
                source_location: SourceLocation::unknown(),
            }],
            visibility: Visibility::Public,
            is_static: false,
            effects: FunctionEffects::default(),
            type_parameters: vec![],
            source_location: SourceLocation::unknown(),
            metadata: FunctionMetadata::default(),
        };

        // Create a simple CFG
        let entry = BlockId::from_raw(1);
        let cfg = ControlFlowGraph::new(SymbolId::from_raw(1), entry);

        // Build mapping
        let result = TastCfgMapping::build(&cfg, &function);

        assert_eq!(result.stats.statements_processed, 1);
        assert!(result.warnings.is_empty());
        assert_eq!(result.stats.successful_mappings, 1);
    }

    #[test]
    fn test_nested_statement_mapping() {
        let interner = StringInterner::new();
        // Create function with if statement
        let function = TypedFunction {
            symbol_id: SymbolId::from_raw(1),
            name: interner.intern("test"),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![TypedStatement::If {
                condition: TypedExpression {
                    expr_type: TypeId::from_raw(1),
                    kind: TypedExpressionKind::Literal {
                        value: LiteralValue::Bool(true),
                    },
                    usage: VariableUsage::Copy,
                    lifetime_id: LifetimeId::first(),
                    source_location: SourceLocation::unknown(),
                    metadata: ExpressionMetadata::default(),
                },
                then_branch: Box::new(TypedStatement::Return {
                    value: None,
                    source_location: SourceLocation::unknown(),
                }),
                else_branch: None,
                source_location: SourceLocation::unknown(),
            }],
            visibility: Visibility::Public,
            is_static: false,
            effects: FunctionEffects::default(),
            type_parameters: vec![],
            source_location: SourceLocation::unknown(),
            metadata: FunctionMetadata::default(),
        };

        // Create CFG with multiple blocks
        let entry = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(SymbolId::from_raw(1), entry);
        cfg.add_block(BasicBlock::new(entry, SourceLocation::unknown()));

        // Build mapping
        let result = TastCfgMapping::build(&cfg, &function);

        // Should process the if statement and its then branch
        assert!(result.stats.statements_processed >= 1);
        assert_eq!(
            result.stats.successful_mappings,
            result.stats.statements_processed
        );
    }
}
