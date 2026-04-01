// compiler/src/tast/semantic_graph/builder.rs
//! Builder for constructing Control Flow Graphs from TAST
//!
//! Transforms typed AST statements and expressions into semantic control flow graphs
//! suitable for advanced static analysis. Handles Haxe-specific constructs including
//! pattern matching, exception handling, and macro expansion.

use crate::tast::collections::{new_id_map, IdMap};
use crate::tast::node::{
    MacroExpansionInfo, TypedCatchClause, TypedExpression, TypedFile, TypedFunction,
    TypedPatternCase, TypedStatement, TypedSwitchCase,
};
use crate::tast::{BlockId, StatementId, TypeId};

use super::cfg::*;
use super::{GraphConstructionError, GraphConstructionOptions, GraphConstructionStats};
use super::{SourceLocation, SymbolId};
use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

/// Builder for constructing CFGs from TAST
pub struct CfgBuilder {
    /// Configuration options
    options: GraphConstructionOptions,

    /// Current CFG being built
    current_cfg: Option<ControlFlowGraph>,

    /// Next available basic block ID
    next_block_id: u32,

    /// Next available statement ID
    next_statement_id: u32,

    /// Mapping from TAST statements to CFG statement IDs
    statement_mapping: BTreeMap<*const TypedStatement, StatementId>,

    /// Stack of break targets for nested loops
    break_targets: Vec<BlockId>,

    /// Stack of continue targets for nested loops
    continue_targets: Vec<BlockId>,

    /// Stack of exception handlers
    exception_handlers: Vec<ExceptionHandlerContext>,

    /// Current loop depth
    current_loop_depth: u32,

    /// Construction statistics
    stats: GraphConstructionStats,
}

/// Context for exception handling during CFG construction
#[derive(Debug, Clone)]
struct ExceptionHandlerContext {
    /// Types of exceptions this handler catches
    exception_types: Vec<TypeId>,

    /// Block where the handler code begins
    handler_block: BlockId,

    /// Variable that receives the exception
    exception_variable: Option<SymbolId>,

    /// Blocks covered by this handler
    covered_blocks: Vec<BlockId>,
}

/// Result of building a statement or expression
#[derive(Debug)]
struct BuildResult {
    /// Block where execution continues after this statement/expression
    continue_block: BlockId,

    /// Whether control flow definitely exits (return, throw, etc.)
    exits: bool,

    /// Blocks created during construction
    created_blocks: Vec<BlockId>,
}

impl CfgBuilder {
    /// Create a new CFG builder with the given options
    pub fn new(options: GraphConstructionOptions) -> Self {
        Self {
            options,
            current_cfg: None,
            next_block_id: 1, // 0 is reserved for invalid ID
            next_statement_id: 1,
            statement_mapping: BTreeMap::new(),
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            exception_handlers: Vec::new(),
            current_loop_depth: 0,
            stats: GraphConstructionStats::new(),
        }
    }

    /// Build CFG for a complete typed file
    pub fn build_file(
        &mut self,
        file: &TypedFile,
    ) -> Result<IdMap<SymbolId, ControlFlowGraph>, GraphConstructionError> {
        let start_time = Instant::now();
        let mut cfgs = new_id_map();

        // Build CFG for each function in the file
        for function in &file.functions {
            let cfg = self.build_function(function)?;
            cfgs.insert(function.symbol_id, cfg);
            self.stats.functions_processed += 1;
        }

        // Update timing statistics
        self.stats.cfg_construction_time_us += start_time.elapsed().as_micros() as u64;

        Ok(cfgs)
    }

    /// Build CFG for a single function
    pub fn build_function(
        &mut self,
        function: &TypedFunction,
    ) -> Result<ControlFlowGraph, GraphConstructionError> {
        // Reset builder state for new function
        self.reset_for_function();

        // Create entry block
        let entry_block_id = self.allocate_block_id();
        let cfg = ControlFlowGraph::new(function.symbol_id, entry_block_id);
        self.current_cfg = Some(cfg);

        // Create entry block
        let entry_block = BasicBlock::new(entry_block_id, function.source_location);
        self.add_block_to_cfg(entry_block);

        // Build CFG for function body
        let body_result = self.build_statement_list(&function.body, entry_block_id)?;

        // Handle implicit return if function doesn't end with explicit return
        if !body_result.exits {
            self.add_implicit_return(body_result.continue_block, function.return_type)?;
        }

        // Extract completed CFG
        let mut cfg = self.current_cfg.take().unwrap();

        // Update metadata
        cfg.metadata.original_statement_count = function.body.len();
        cfg.metadata.max_loop_depth = self.current_loop_depth;
        cfg.metadata.construction_stats.blocks_created = cfg.blocks.len();
        cfg.metadata.construction_stats.edges_created = cfg
            .blocks
            .values()
            .map(|block| block.successors.len())
            .sum();

        // Validate the CFG if enabled
        if self.options.collect_statistics {
            // Use relaxed validation that allows unreachable blocks
            // This is important for cases like loops with always-exiting bodies
            cfg.validate_with_options(false).map_err(|e| {
                GraphConstructionError::InternalError {
                    message: format!("CFG validation failed: {}", e),
                }
            })?;
        }

        Ok(cfg)
    }

    /// Reset builder state for a new function
    fn reset_for_function(&mut self) {
        self.next_block_id = 1;
        self.next_statement_id = 1;
        self.statement_mapping.clear();
        self.break_targets.clear();
        self.continue_targets.clear();
        self.exception_handlers.clear();
        self.current_loop_depth = 0;
    }

    /// Build CFG for a list of statements
    fn build_statement_list(
        &mut self,
        statements: &[TypedStatement],
        start_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let mut current_block = start_block;
        let mut created_blocks = vec![start_block];

        for statement in statements {
            let result = self.build_statement(statement, current_block)?;

            // If this statement exits (return, throw), stop processing
            if result.exits {
                return Ok(BuildResult {
                    continue_block: current_block,
                    exits: true,
                    created_blocks,
                });
            }

            current_block = result.continue_block;
            created_blocks.extend(result.created_blocks);
        }

        Ok(BuildResult {
            continue_block: current_block,
            exits: false,
            created_blocks,
        })
    }

    /// Build CFG for a single statement
    fn build_statement(
        &mut self,
        statement: &TypedStatement,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        match statement {
            TypedStatement::Expression { expression, .. } => {
                self.build_expression_statement(expression, current_block)
            }

            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => self.build_var_declaration(*symbol_id, initializer.as_ref(), current_block),

            TypedStatement::Assignment { target, value, .. } => {
                self.build_assignment(target, value, current_block)
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.build_if_statement(condition, then_branch, else_branch.as_ref(), current_block)
            }

            TypedStatement::While {
                condition, body, ..
            } => self.build_while_loop(condition, body, current_block),

            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => self.build_for_loop(
                init.as_ref(),
                condition.as_ref(),
                update.as_ref(),
                body,
                current_block,
            ),

            TypedStatement::Return { value, .. } => {
                self.build_return_statement(value.as_ref(), current_block)
            }

            TypedStatement::Throw { exception, .. } => {
                self.build_throw_statement(exception, current_block)
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => self.build_try_statement(
                body,
                catch_clauses.as_slice(),
                finally_block.as_ref(),
                current_block,
            ),

            TypedStatement::Switch {
                discriminant,
                cases,
                default_case,
                ..
            } => self.build_switch_statement(
                discriminant,
                cases.as_slice(),
                default_case.as_ref(),
                current_block,
            ),

            TypedStatement::Break { .. } => self.build_break_statement(current_block),

            TypedStatement::Continue { .. } => self.build_continue_statement(current_block),

            TypedStatement::Block { statements, .. } => {
                self.build_statement_list(statements, current_block)
            }

            // Haxe-specific statements
            TypedStatement::PatternMatch {
                value, patterns, ..
            } => self.build_pattern_match(value, patterns.as_slice(), current_block),

            TypedStatement::MacroExpansion {
                expansion_info,
                expanded_statements,
                ..
            } => self.build_macro_expansion(expansion_info, expanded_statements, current_block),

            TypedStatement::ForIn {
                value_var,
                key_var,
                iterable,
                body,
                ..
            } => {
                // Build for-in loop similar to regular for loop
                self.build_for_in_statement(*value_var, *key_var, iterable, body, current_block)
            }
        }
    }

    /// Build CFG for an expression statement
    fn build_expression_statement(
        &mut self,
        expression: &TypedExpression,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        // Add the expression as a statement to the current block
        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        // Expression statements don't change control flow
        Ok(BuildResult {
            continue_block: current_block,
            exits: false,
            created_blocks: vec![],
        })
    }

    /// Build CFG for variable declaration
    fn build_var_declaration(
        &mut self,
        symbol_id: SymbolId,
        initializer: Option<&TypedExpression>,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        // Variable declarations don't change control flow
        Ok(BuildResult {
            continue_block: current_block,
            exits: false,
            created_blocks: vec![],
        })
    }

    /// Build CFG for assignment statement
    fn build_assignment(
        &mut self,
        target: &TypedExpression,
        value: &TypedExpression,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        // Assignments don't change control flow
        Ok(BuildResult {
            continue_block: current_block,
            exits: false,
            created_blocks: vec![],
        })
    }

    /// Build CFG for if statement
    fn build_if_statement(
        &mut self,
        condition: &TypedExpression,
        then_branch: &TypedStatement,
        else_branch: Option<&Box<TypedStatement>>,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        // Create blocks for then and else branches
        let then_block = self.allocate_block_id();
        let else_block = self.allocate_block_id();
        let merge_block = self.allocate_block_id();

        // Create all blocks first before setting any terminators
        let then_bb = BasicBlock::new(then_block, then_branch.source_location());
        self.add_block_to_cfg(then_bb);

        let else_bb = BasicBlock::new(
            else_block,
            else_branch.map_or(condition.source_location(), |s| s.source_location()),
        );
        self.add_block_to_cfg(else_bb);

        let merge_bb = BasicBlock::new(merge_block, condition.source_location());
        self.add_block_to_cfg(merge_bb);

        // Now set up conditional branch in current block
        let branch_terminator = Terminator::Branch {
            condition: condition.clone(),
            true_target: then_block,
            false_target: else_block,
        };
        self.set_block_terminator(current_block, branch_terminator);

        // Build then branch
        let then_result = self.build_statement(then_branch, then_block)?;

        // Build else branch
        let else_result = if let Some(else_stmt) = else_branch {
            self.build_statement(else_stmt, else_block)?
        } else {
            // Empty else branch - just jump to merge
            self.set_block_terminator(
                else_block,
                Terminator::Jump {
                    target: merge_block,
                },
            );
            BuildResult {
                continue_block: else_block,
                exits: false,
                created_blocks: vec![],
            }
        };

        // Connect non-exiting branches to merge block
        if !then_result.exits {
            self.set_block_terminator(
                then_result.continue_block,
                Terminator::Jump {
                    target: merge_block,
                },
            );
        }
        if !else_result.exits {
            self.set_block_terminator(
                else_result.continue_block,
                Terminator::Jump {
                    target: merge_block,
                },
            );
        }

        // If both branches exit, the merge block is unreachable
        let exits = then_result.exits && else_result.exits;
        let continue_block = if exits { current_block } else { merge_block };

        let mut created_blocks = vec![then_block, else_block, merge_block];
        created_blocks.extend(then_result.created_blocks);
        created_blocks.extend(else_result.created_blocks);

        Ok(BuildResult {
            continue_block,
            exits,
            created_blocks,
        })
    }

    /// Build CFG for while loop
    fn build_while_loop(
        &mut self,
        condition: &TypedExpression,
        body: &TypedStatement,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        // Create blocks for loop header and body
        let header_block = self.allocate_block_id();
        let body_block = self.allocate_block_id();

        // Create all blocks first
        let header_bb = BasicBlock::new(header_block, condition.source_location());
        self.add_block_to_cfg(header_bb);

        // Set up loop context - use header as continue target
        self.current_loop_depth += 1;
        self.continue_targets.push(header_block);

        let body_bb = BasicBlock::new(body_block, body.source_location());
        let mut body_bb = body_bb;
        body_bb.metadata.loop_depth = self.current_loop_depth;
        self.add_block_to_cfg(body_bb);

        // Create exit block only if needed (when body doesn't always exit)
        let mut created_blocks = vec![header_block, body_block];

        // Jump from current block to header
        self.set_block_terminator(
            current_block,
            Terminator::Jump {
                target: header_block,
            },
        );

        // Build loop body first to know if it exits
        let body_result = self.build_statement(body, body_block)?;
        created_blocks.extend(body_result.created_blocks);

        // Determine if we need an exit block
        let (exit_block, loop_exits) = if body_result.exits {
            // Body always exits - loop can only exit via condition being false
            // Create exit block for when condition is false
            let exit_block = self.allocate_block_id();
            let exit_bb = BasicBlock::new(exit_block, condition.source_location());
            self.add_block_to_cfg(exit_bb);
            created_blocks.push(exit_block);
            (exit_block, false)
        } else {
            // Body can continue - create exit block for condition being false
            let exit_block = self.allocate_block_id();
            let exit_bb = BasicBlock::new(exit_block, condition.source_location());
            self.add_block_to_cfg(exit_bb);
            created_blocks.push(exit_block);

            // Connect body back to header
            self.set_block_terminator(
                body_result.continue_block,
                Terminator::Jump {
                    target: header_block,
                },
            );
            (exit_block, false)
        };

        // Set break target and set up conditional branch
        self.break_targets.push(exit_block);

        let branch_terminator = Terminator::Branch {
            condition: condition.clone(),
            true_target: body_block,
            false_target: exit_block,
        };
        self.set_block_terminator(header_block, branch_terminator);

        // Clean up loop context
        self.current_loop_depth -= 1;
        self.break_targets.pop();
        self.continue_targets.pop();

        Ok(BuildResult {
            continue_block: exit_block,
            exits: loop_exits,
            created_blocks,
        })
    }

    /// Build CFG for return statement
    fn build_return_statement(
        &mut self,
        value: Option<&TypedExpression>,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        let return_terminator = Terminator::Return {
            value: value.cloned(),
        };
        self.set_block_terminator(current_block, return_terminator);

        Ok(BuildResult {
            continue_block: current_block,
            exits: true,
            created_blocks: vec![],
        })
    }

    /// Build CFG for throw statement
    fn build_throw_statement(
        &mut self,
        exception: &TypedExpression,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        let throw_terminator = Terminator::Throw {
            exception: exception.clone(),
        };
        self.set_block_terminator(current_block, throw_terminator);

        Ok(BuildResult {
            continue_block: current_block,
            exits: true,
            created_blocks: vec![],
        })
    }

    /// Build CFG for break statement
    fn build_break_statement(
        &mut self,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let break_target =
            *self
                .break_targets
                .last()
                .ok_or_else(|| GraphConstructionError::InvalidTAST {
                    message: "break statement outside of loop".to_string(),
                    location: self.get_current_location(),
                })?;

        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        self.set_block_terminator(
            current_block,
            Terminator::Jump {
                target: break_target,
            },
        );

        Ok(BuildResult {
            continue_block: current_block,
            exits: true,
            created_blocks: vec![],
        })
    }

    /// Build CFG for continue statement
    fn build_continue_statement(
        &mut self,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        let continue_target =
            *self
                .continue_targets
                .last()
                .ok_or_else(|| GraphConstructionError::InvalidTAST {
                    message: "continue statement outside of loop".to_string(),
                    location: self.get_current_location(),
                })?;

        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);

        self.set_block_terminator(
            current_block,
            Terminator::Jump {
                target: continue_target,
            },
        );

        Ok(BuildResult {
            continue_block: current_block,
            exits: true,
            created_blocks: vec![],
        })
    }

    /// Build CFG for for-in statement
    fn build_for_in_statement(
        &mut self,
        value_var: SymbolId,
        key_var: Option<SymbolId>,
        iterable: &TypedExpression,
        body: &TypedStatement,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        // Create blocks for the for-in loop
        let loop_header = self.allocate_block_id();
        let loop_body = self.allocate_block_id();
        let loop_exit = self.allocate_block_id();

        // Add statement to current block and jump to loop header
        let statement_id = self.allocate_statement_id();
        self.add_statement_to_block(current_block, statement_id);
        self.set_block_terminator(
            current_block,
            Terminator::Jump {
                target: loop_header,
            },
        );

        // Set up loop header - evaluate iterable and check if iteration should continue
        let loop_header_block = BasicBlock::new(loop_header, iterable.source_location());
        self.add_block_to_cfg(loop_header_block);
        let header_stmt_id = self.allocate_statement_id();
        self.add_statement_to_block(loop_header, header_stmt_id);

        // Set loop targets for break/continue
        self.break_targets.push(loop_exit);
        self.continue_targets.push(loop_header);

        // Build loop body
        let loop_body_block = BasicBlock::new(loop_body, body.source_location());
        self.add_block_to_cfg(loop_body_block);
        let body_result = self.build_statement(body, loop_body)?;

        // If body doesn't exit, jump back to header
        if !body_result.exits {
            self.set_block_terminator(
                body_result.continue_block,
                Terminator::Jump {
                    target: loop_header,
                },
            );
        }

        // Set header terminator to conditionally branch to body or exit
        // For now, use a simplified condition - would need proper iterable evaluation
        let condition_expr = iterable.clone();
        self.set_block_terminator(
            loop_header,
            Terminator::Branch {
                condition: condition_expr,
                true_target: loop_body,
                false_target: loop_exit,
            },
        );

        // Clean up loop targets
        self.break_targets.pop();
        self.continue_targets.pop();

        // Set up exit block
        let loop_exit_block = BasicBlock::new(loop_exit, iterable.source_location());
        self.add_block_to_cfg(loop_exit_block);

        Ok(BuildResult {
            continue_block: loop_exit,
            exits: false,
            created_blocks: vec![loop_header, loop_body, loop_exit],
        })
    }

    /// Add implicit return for functions that don't end with explicit return
    fn add_implicit_return(
        &mut self,
        block: BlockId,
        return_type: TypeId,
    ) -> Result<(), GraphConstructionError> {
        // For void functions, add return with no value
        // For other functions, this would be a type error that should be caught earlier
        let return_terminator = Terminator::Return { value: None };
        self.set_block_terminator(block, return_terminator);
        Ok(())
    }

    /// Allocate a new basic block ID
    fn allocate_block_id(&mut self) -> BlockId {
        let id = BlockId::from_raw(self.next_block_id);
        self.next_block_id += 1;
        self.stats.total_basic_blocks += 1;
        id
    }

    /// Allocate a new statement ID
    fn allocate_statement_id(&mut self) -> StatementId {
        let id = StatementId::from_raw(self.next_statement_id);
        self.next_statement_id += 1;
        id
    }

    /// Add a block to the current CFG
    fn add_block_to_cfg(&mut self, block: BasicBlock) {
        if let Some(ref mut cfg) = self.current_cfg {
            cfg.add_block(block);
        }
    }

    /// Add a statement to a block
    fn add_statement_to_block(&mut self, block_id: BlockId, statement_id: StatementId) {
        if let Some(ref mut cfg) = self.current_cfg {
            if let Some(block) = cfg.get_block_mut(block_id) {
                block.add_statement(statement_id);
            }
        }
    }

    /// Set the terminator for a block
    fn set_block_terminator(&mut self, block_id: BlockId, terminator: Terminator) {
        if let Some(ref mut cfg) = self.current_cfg {
            cfg.update_block_terminator(block_id, terminator);
        }
    }

    /// Get construction statistics
    pub fn stats(&self) -> &GraphConstructionStats {
        &self.stats
    }

    /// Get the current source location (placeholder)
    fn get_current_location(&self) -> SourceLocation {
        // Return a default location - in a real implementation this would track the current statement
        SourceLocation::new(0, 0, 0, 0)
    }

    fn build_for_loop(
        &mut self,
        init: Option<&Box<TypedStatement>>,
        condition: Option<&TypedExpression>,
        update: Option<&TypedExpression>,
        body: &TypedStatement,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        // Create blocks for different parts of the loop
        let init_block = if init.is_some() {
            Some(self.allocate_block_id())
        } else {
            None
        };
        let condition_block = if condition.is_some() {
            self.allocate_block_id()
        } else {
            self.allocate_block_id() // Still need a header block
        };
        let body_block = self.allocate_block_id();
        let update_block = if update.is_some() {
            Some(self.allocate_block_id())
        } else {
            None
        };
        let exit_block = self.allocate_block_id();

        let mut created_blocks = vec![condition_block, body_block, exit_block];
        if let Some(init_id) = init_block {
            created_blocks.push(init_id);
        }
        if let Some(update_id) = update_block {
            created_blocks.push(update_id);
        }

        // Create all blocks first before setting any terminators
        if let Some(init_id) = init_block {
            let init_stmt = init.unwrap();
            let init_bb = BasicBlock::new(init_id, init_stmt.source_location());
            self.add_block_to_cfg(init_bb);
        }

        let condition_bb = BasicBlock::new(
            condition_block,
            condition.map_or_else(|| body.source_location(), |c| c.source_location()),
        );
        self.add_block_to_cfg(condition_bb);

        // Set up loop context for break/continue
        self.current_loop_depth += 1;
        self.break_targets.push(exit_block);
        let continue_target = update_block.unwrap_or(condition_block);
        self.continue_targets.push(continue_target);

        let body_bb = BasicBlock::new(body_block, body.source_location());
        let mut body_bb = body_bb;
        body_bb.metadata.loop_depth = self.current_loop_depth;
        self.add_block_to_cfg(body_bb);

        if let Some(update_id) = update_block {
            let update_expr = update.unwrap();
            let update_bb = BasicBlock::new(update_id, update_expr.source_location());
            let mut update_bb = update_bb;
            update_bb.metadata.loop_depth = self.current_loop_depth;
            self.add_block_to_cfg(update_bb);
        }

        let exit_bb = BasicBlock::new(
            exit_block,
            condition.map_or_else(|| body.source_location(), |c| c.source_location()),
        );
        self.add_block_to_cfg(exit_bb);

        // Now set terminators after all blocks exist
        if let Some(init_stmt) = init {
            let init_id = init_block.unwrap();

            // Jump from current block to init
            self.set_block_terminator(current_block, Terminator::Jump { target: init_id });

            let init_result = self.build_statement(init_stmt, init_id)?;
            created_blocks.extend(init_result.created_blocks);

            // Init flows to condition/header
            if !init_result.exits {
                self.set_block_terminator(
                    init_result.continue_block,
                    Terminator::Jump {
                        target: condition_block,
                    },
                );
            }
        } else {
            // No init - jump directly to condition
            self.set_block_terminator(
                current_block,
                Terminator::Jump {
                    target: condition_block,
                },
            );
        }

        // Set condition terminator
        if let Some(cond_expr) = condition {
            // Conditional loop: check condition and branch
            let branch_terminator = Terminator::Branch {
                condition: cond_expr.clone(),
                true_target: body_block,
                false_target: exit_block,
            };
            self.set_block_terminator(condition_block, branch_terminator);
        } else {
            // Infinite loop: always go to body
            self.set_block_terminator(condition_block, Terminator::Jump { target: body_block });
        }

        // Build loop body
        let body_result = self.build_statement(body, body_block)?;
        created_blocks.extend(body_result.created_blocks);

        // Handle body completion
        if !body_result.exits {
            if let Some(update_expr) = update {
                // Body flows to update block
                let update_id = update_block.unwrap();
                self.set_block_terminator(
                    body_result.continue_block,
                    Terminator::Jump { target: update_id },
                );

                // Add update expression as statement
                let update_stmt_id = self.allocate_statement_id();
                self.add_statement_to_block(update_id, update_stmt_id);

                // Update flows back to condition
                self.set_block_terminator(
                    update_id,
                    Terminator::Jump {
                        target: condition_block,
                    },
                );
            } else {
                // Body flows directly back to condition
                self.set_block_terminator(
                    body_result.continue_block,
                    Terminator::Jump {
                        target: condition_block,
                    },
                );
            }
        }

        // Clean up loop context
        self.current_loop_depth -= 1;
        self.break_targets.pop();
        self.continue_targets.pop();

        Ok(BuildResult {
            continue_block: exit_block,
            exits: false, // For loops can always exit via break or condition
            created_blocks,
        })
    }

    fn build_try_statement(
        &mut self,
        body: &TypedStatement,
        catch_clauses: &[TypedCatchClause],
        finally_block: Option<&Box<TypedStatement>>,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        // Create blocks for different execution paths
        let try_block = self.allocate_block_id();
        let finally_block_id = if finally_block.is_some() {
            Some(self.allocate_block_id())
        } else {
            None
        };

        let mut created_blocks = vec![try_block];
        if let Some(finally_id) = finally_block_id {
            created_blocks.push(finally_id);
        }

        // Create exception handler contexts for catch clauses
        let mut catch_handlers = Vec::new();
        for catch_clause in catch_clauses {
            let catch_block = self.allocate_block_id();
            created_blocks.push(catch_block);

            let handler_context = ExceptionHandlerContext {
                exception_types: vec![catch_clause.exception_type],
                handler_block: catch_block,
                exception_variable: Some(catch_clause.exception_variable),
                covered_blocks: vec![try_block], // Will be extended as we build
            };
            catch_handlers.push(handler_context);
        }

        // Create all blocks first before setting any terminators
        let try_bb = BasicBlock::new(try_block, body.source_location());
        self.add_block_to_cfg(try_bb);

        for (catch_clause, handler) in catch_clauses.iter().zip(catch_handlers.iter()) {
            let catch_bb = BasicBlock::new(handler.handler_block, catch_clause.source_location);
            self.add_block_to_cfg(catch_bb);
        }

        if let Some(finally_stmt) = finally_block {
            let finally_id = finally_block_id.unwrap();
            let finally_bb = BasicBlock::new(finally_id, finally_stmt.source_location());
            self.add_block_to_cfg(finally_bb);
        }

        // Now set terminators after all blocks exist
        self.set_block_terminator(current_block, Terminator::Jump { target: try_block });

        // Push exception handlers onto stack
        self.exception_handlers
            .extend(catch_handlers.iter().cloned());

        // Build try body
        let try_result = self.build_statement(body, try_block)?;
        created_blocks.extend(try_result.created_blocks.clone());

        // Update covered blocks for all exception handlers
        for handler in &mut catch_handlers {
            handler.covered_blocks.extend(&try_result.created_blocks);
        }

        // Pop exception handlers
        for _ in 0..catch_handlers.len() {
            self.exception_handlers.pop();
        }

        // Build catch blocks
        let mut catch_results = Vec::new();
        for (catch_clause, handler) in catch_clauses.iter().zip(catch_handlers.iter()) {
            // Build catch body
            let catch_result = self.build_statement(&catch_clause.body, handler.handler_block)?;
            created_blocks.extend(catch_result.created_blocks.clone());
            catch_results.push(catch_result);
        }

        // Build finally block if present
        let mut finally_exits = false;
        if let Some(finally_stmt) = finally_block {
            let finally_id = finally_block_id.unwrap();
            let finally_result = self.build_statement(finally_stmt, finally_id)?;
            created_blocks.extend(finally_result.created_blocks);
            finally_exits = finally_result.exits;
        }

        // Determine if we need a merge block
        let any_path_continues = !try_result.exits
            || catch_results.iter().any(|result| !result.exits)
            || (finally_block.is_some() && !finally_exits);

        let continue_block = if any_path_continues {
            // Create merge block only if needed
            let merge_block = self.allocate_block_id();
            created_blocks.push(merge_block);

            let merge_bb = BasicBlock::new(merge_block, body.source_location());
            self.add_block_to_cfg(merge_bb);

            // Connect non-exiting paths to merge block
            if !try_result.exits {
                let target = finally_block_id.unwrap_or(merge_block);
                self.set_block_terminator(try_result.continue_block, Terminator::Jump { target });
            }

            for catch_result in catch_results.iter() {
                if !catch_result.exits {
                    let target = finally_block_id.unwrap_or(merge_block);
                    self.set_block_terminator(
                        catch_result.continue_block,
                        Terminator::Jump { target },
                    );
                }
            }

            if let Some(_) = finally_block {
                let finally_id = finally_block_id.unwrap();
                if !finally_exits {
                    self.set_block_terminator(
                        finally_id,
                        Terminator::Jump {
                            target: merge_block,
                        },
                    );
                }
            }

            merge_block
        } else {
            // All paths exit - no merge needed, just set terminators correctly
            if !try_result.exits {
                if let Some(finally_id) = finally_block_id {
                    self.set_block_terminator(
                        try_result.continue_block,
                        Terminator::Jump { target: finally_id },
                    );
                }
            }

            for catch_result in catch_results.iter() {
                if !catch_result.exits {
                    if let Some(finally_id) = finally_block_id {
                        self.set_block_terminator(
                            catch_result.continue_block,
                            Terminator::Jump { target: finally_id },
                        );
                    }
                }
            }

            current_block
        };

        // Register exception handlers in CFG
        for (catch_clause, handler) in catch_clauses.iter().zip(catch_handlers.iter()) {
            let handler_info = ExceptionHandlerInfo {
                exception_types: vec![catch_clause.exception_type],
                covered_blocks: handler.covered_blocks.iter().copied().collect(),
                handler_block: handler.handler_block,
                exception_variable: Some(catch_clause.exception_variable),
            };

            if let Some(ref mut cfg) = self.current_cfg {
                cfg.exception_handlers
                    .insert(handler.handler_block, handler_info);
            }
        }

        // Determine if the entire try-catch-finally exits
        let exits = try_result.exits
            && catch_results.iter().all(|result| result.exits)
            && (finally_block.is_none() || finally_exits);

        Ok(BuildResult {
            continue_block,
            exits,
            created_blocks,
        })
    }

    /// Build CFG for switch statement
    fn build_switch_statement(
        &mut self,
        discriminant: &TypedExpression,
        cases: &[TypedSwitchCase],
        default_case: Option<&Box<TypedStatement>>,
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        if cases.is_empty() && default_case.is_none() {
            return Err(GraphConstructionError::InvalidTAST {
                message: "Switch statement with no cases or default".to_string(),
                location: discriminant.source_location(),
            });
        }

        // Create merge block for after switch
        let merge_block = self.allocate_block_id();
        let mut created_blocks = vec![merge_block];
        let mut switch_targets = Vec::new();
        let mut all_cases_exit = true;

        // Create blocks for each case
        let mut case_blocks = Vec::new();
        for case in cases {
            let case_block = self.allocate_block_id();
            created_blocks.push(case_block);
            case_blocks.push(case_block);

            // Create switch target
            let switch_target = SwitchTarget {
                case_value: case.case_value.clone(),
                target: case_block,
            };
            switch_targets.push(switch_target);
        }

        // Create default block if present
        let default_block = if let Some(_) = default_case {
            let default_block = self.allocate_block_id();
            created_blocks.push(default_block);
            Some(default_block)
        } else {
            None
        };

        // Create all blocks first before setting any terminators
        for (case, case_block) in cases.iter().zip(case_blocks.iter()) {
            let case_bb = BasicBlock::new(*case_block, case.source_location);
            self.add_block_to_cfg(case_bb);
        }

        if let Some(default_stmt) = default_case {
            let default_block_id = default_block.unwrap();
            let default_bb = BasicBlock::new(default_block_id, default_stmt.source_location());
            self.add_block_to_cfg(default_bb);
        }

        let merge_bb = BasicBlock::new(merge_block, discriminant.source_location());
        self.add_block_to_cfg(merge_bb);

        // Now set switch terminator on current block
        let default_target = default_block.or(Some(merge_block));
        let switch_terminator = Terminator::Switch {
            discriminant: discriminant.clone(),
            targets: switch_targets,
            default_target,
        };
        self.set_block_terminator(current_block, switch_terminator);

        // Build case bodies
        for (case, case_block) in cases.iter().zip(case_blocks.iter()) {
            let case_result = self.build_statement(&case.body, *case_block)?;
            created_blocks.extend(case_result.created_blocks);

            // Handle case completion
            if !case_result.exits {
                // In Haxe, switch cases don't fall through by default
                // Each case needs explicit break or return
                self.set_block_terminator(
                    case_result.continue_block,
                    Terminator::Jump {
                        target: merge_block,
                    },
                );
                all_cases_exit = false;
            }
        }

        // Build default case if present
        if let Some(default_stmt) = default_case {
            let default_block_id = default_block.unwrap();
            let default_result = self.build_statement(default_stmt, default_block_id)?;
            created_blocks.extend(default_result.created_blocks);

            // Handle default completion
            if !default_result.exits {
                self.set_block_terminator(
                    default_result.continue_block,
                    Terminator::Jump {
                        target: merge_block,
                    },
                );
                all_cases_exit = false;
            }
        } else {
            // No default case - unmatched values fall through to merge
            all_cases_exit = false;
        }

        // Update break/continue context if we're in a loop
        // Switch statements can be break targets in some languages
        let exits = all_cases_exit && default_case.is_some();
        let continue_block = if exits { current_block } else { merge_block };

        Ok(BuildResult {
            continue_block,
            exits,
            created_blocks,
        })
    }
    /// Build CFG for pattern matching statement
    fn build_pattern_match(
        &mut self,
        value: &TypedExpression,
        patterns: &[TypedPatternCase],
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        if patterns.is_empty() {
            return Err(GraphConstructionError::InvalidTAST {
                message: "Pattern match with no patterns".to_string(),
                location: value.source_location(),
            });
        }

        // Create blocks for each pattern and merge block
        let merge_block = self.allocate_block_id();
        let mut pattern_targets = Vec::new();
        let mut created_blocks = vec![merge_block];
        let mut all_patterns_exit = true;

        // Create pattern blocks and targets
        let mut pattern_blocks = Vec::new();
        for pattern_case in patterns {
            let pattern_block = self.allocate_block_id();
            created_blocks.push(pattern_block);
            pattern_blocks.push(pattern_block);

            // Create the pattern target for the terminator
            let pattern_target = PatternTarget {
                pattern: pattern_case.pattern.clone(),
                target: pattern_block,
                bound_variables: pattern_case.bound_variables.clone(),
            };
            pattern_targets.push(pattern_target);
        }

        // Create all blocks first before setting any terminators
        for (pattern_case, pattern_block) in patterns.iter().zip(pattern_blocks.iter()) {
            let pattern_bb = BasicBlock::new(*pattern_block, pattern_case.source_location);
            self.add_block_to_cfg(pattern_bb);
        }

        let merge_bb = BasicBlock::new(merge_block, value.source_location());
        self.add_block_to_cfg(merge_bb);

        // Set pattern match terminator on current block
        let pattern_terminator = Terminator::PatternMatch {
            value: value.clone(),
            patterns: pattern_targets,
            default_target: if all_patterns_exit {
                None
            } else {
                Some(merge_block)
            },
        };
        self.set_block_terminator(current_block, pattern_terminator);

        // Now build pattern bodies after all blocks exist
        for (pattern_case, pattern_block) in patterns.iter().zip(pattern_blocks.iter()) {
            let pattern_result = self.build_statement(&pattern_case.body, *pattern_block)?;
            created_blocks.extend(pattern_result.created_blocks);

            // If pattern doesn't exit, connect to merge block
            if !pattern_result.exits {
                self.set_block_terminator(
                    pattern_result.continue_block,
                    Terminator::Jump {
                        target: merge_block,
                    },
                );
                all_patterns_exit = false;
            }
        }

        // If all patterns exit, merge block is unreachable
        let continue_block = if all_patterns_exit {
            current_block
        } else {
            merge_block
        };

        Ok(BuildResult {
            continue_block,
            exits: all_patterns_exit,
            created_blocks,
        })
    }

    fn build_macro_expansion(
        &mut self,
        expansion_info: &MacroExpansionInfo,
        expanded_statements: &[TypedStatement],
        current_block: BlockId,
    ) -> Result<BuildResult, GraphConstructionError> {
        if expanded_statements.is_empty() {
            // Empty macro expansion - just continue
            return Ok(BuildResult {
                continue_block: current_block,
                exits: false,
                created_blocks: vec![],
            });
        }

        // Create a block for the expanded code
        let expansion_block = self.allocate_block_id();
        let mut created_blocks = vec![expansion_block];

        // Create expansion block first
        let expansion_bb = BasicBlock::new(expansion_block, expansion_info.original_location);
        self.add_block_to_cfg(expansion_bb);

        // Now create macro expansion terminator to jump to expanded code
        let macro_terminator = Terminator::MacroExpansion {
            target: expansion_block,
            macro_info: expansion_info.clone(),
        };
        self.set_block_terminator(current_block, macro_terminator);

        // Build the expanded statements
        let expansion_result = self.build_statement_list(expanded_statements, expansion_block)?;
        created_blocks.extend(expansion_result.created_blocks);

        Ok(BuildResult {
            continue_block: expansion_result.continue_block,
            exits: expansion_result.exits,
            created_blocks,
        })
    }
}

// Extension trait to add source_location to TypedStatement/Expression
trait HasSourceLocation {
    fn source_location(&self) -> SourceLocation;
}

impl HasSourceLocation for TypedStatement {
    fn source_location(&self) -> SourceLocation {
        /// **REAL IMPLEMENTATION**: Extract actual source location from TAST nodes
        match self {
            TypedStatement::Expression {
                source_location, ..
            } => *source_location,
            TypedStatement::VarDeclaration {
                source_location, ..
            } => *source_location,
            TypedStatement::Assignment {
                source_location, ..
            } => *source_location,
            TypedStatement::If {
                source_location, ..
            } => *source_location,
            TypedStatement::While {
                source_location, ..
            } => *source_location,
            TypedStatement::For {
                source_location, ..
            } => *source_location,
            TypedStatement::Return {
                source_location, ..
            } => *source_location,
            TypedStatement::Throw {
                source_location, ..
            } => *source_location,
            TypedStatement::Try {
                source_location, ..
            } => *source_location,
            TypedStatement::Switch {
                source_location, ..
            } => *source_location,
            TypedStatement::Break {
                source_location, ..
            } => *source_location,
            TypedStatement::Continue {
                source_location, ..
            } => *source_location,
            TypedStatement::Block {
                source_location, ..
            } => *source_location,
            TypedStatement::PatternMatch {
                source_location, ..
            } => *source_location,
            TypedStatement::MacroExpansion {
                source_location, ..
            } => *source_location,
            TypedStatement::ForIn {
                source_location, ..
            } => *source_location,
        }
    }
}

impl HasSourceLocation for TypedExpression {
    fn source_location(&self) -> SourceLocation {
        /// **REAL IMPLEMENTATION**: Extract actual source location from TAST expressions
        // TypedExpression should have a source_location field directly
        // This is the proper way to get source locations from expressions
        self.source_location
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfg_builder_creation() {
        let options = GraphConstructionOptions::default();
        let builder = CfgBuilder::new(options);

        assert_eq!(builder.next_block_id, 1);
        assert_eq!(builder.current_loop_depth, 0);
        assert!(builder.break_targets.is_empty());
    }

    #[test]
    fn test_block_allocation() {
        let options = GraphConstructionOptions::default();
        let mut builder = CfgBuilder::new(options);

        let block1 = builder.allocate_block_id();
        let block2 = builder.allocate_block_id();

        assert_eq!(block1.as_raw(), 1);
        assert_eq!(block2.as_raw(), 2);
        assert_eq!(builder.stats.total_basic_blocks, 2);
    }

    #[test]
    fn test_statement_allocation() {
        let options = GraphConstructionOptions::default();
        let mut builder = CfgBuilder::new(options);

        let stmt1 = builder.allocate_statement_id();
        let stmt2 = builder.allocate_statement_id();

        assert_eq!(stmt1.as_raw(), 1);
        assert_eq!(stmt2.as_raw(), 2);
    }
}
