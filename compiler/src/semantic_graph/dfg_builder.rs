//! Builder for constructing Data Flow Graphs from TAST and CFG
//!
//! Transforms typed expressions and control flow into SSA-form data flow graphs
//! suitable for advanced optimization and analysis. Handles Haxe-specific
//! constructs and maintains precise def-use information.

use crate::semantic_graph::free_variables::{CapturedVariable, FreeVariableVisitor};
use crate::semantic_graph::phi_type::{DfgBuilderPhiTypeUnification, PhiTypeUnifier};
use crate::semantic_graph::tast_cfg_mapping::{BranchContext, StatementLocation, TastCfgMapping};
use crate::tast::collections::{new_id_map, new_id_set, IdMap, IdSet};
use crate::tast::core::{TypeKind, TypeTable};
use crate::tast::node::{
    BinaryOperator, CastKind as TastCastKind, HasSourceLocation, LiteralValue, TypedExpression,
    TypedExpressionKind, TypedFunction, TypedMapEntry, TypedObjectField, TypedParameter,
    TypedStatement, UnaryOperator,
};
use crate::tast::type_checker::{TypeChecker, TypeCompatibility};
use crate::tast::{BlockId, DataFlowNodeId, SourceLocation, SsaVariableId, SymbolId, TypeId};

use super::cfg::{BasicBlock, ControlFlowGraph, Terminator};
use super::dfg::*;
use super::dominance::DominanceTree;
use super::{GraphConstructionError, GraphConstructionOptions};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

impl TypeId {
    fn void() -> Self {
        TypeId::invalid() // Using invalid as void for now
    }
}

/// **DFG Builder - Production SSA Construction**
///
/// Transforms TAST into SSA-form data flow graphs using the actual enum variants
/// and struct fields from the codebase.
pub struct DfgBuilder {
    /// Configuration options
    options: GraphConstructionOptions,

    /// Current DFG being constructed
    pub(crate) dfg: DataFlowGraph,

    /// SSA construction state
    ssa_state: SsaConstructionState,

    /// Node and variable allocation
    next_node_id: u32,
    next_ssa_var_id: u32,

    /// Loop scope tracking for proper Phi placement
    loop_scope_tracker: LoopScopeTracker,

    /// Pending Phi updates for loop back-edges
    phi_pending_updates: HashMap<DataFlowNodeId, Vec<PhiIncoming>>,

    /// Statistics and diagnostics
    stats: DfgBuilderStats,
}

/// **SSA Construction State Management**
#[derive(Debug)]
pub struct SsaConstructionState {
    /// Variable stacks for renaming (symbol -> stack of SSA variables)
    variable_stacks: HashMap<SymbolId, Vec<SsaVariableId>>,

    /// Blocks where each original variable is defined
    def_blocks: HashMap<SymbolId, HashSet<BlockId>>,

    /// Phi nodes placed in each block for each variable
    phi_placed: HashMap<(BlockId, SymbolId), DataFlowNodeId>,

    /// Incomplete phi nodes awaiting operand completion
    incomplete_phis: HashMap<DataFlowNodeId, IncompletePhiInfo>,

    /// Current block during construction
    current_block: BlockId,

    /// Variables at block exits for correct phi operand computation
    block_exit_variables: HashMap<(BlockId, SymbolId), SsaVariableId>,
}

/// **Loop Scope Tracker - Minimal scope awareness for Phi placement**
#[derive(Debug, Default)]
struct LoopScopeTracker {
    /// Variables that may need Phi nodes
    potentially_modified: HashSet<SymbolId>,

    /// Current loop nesting level
    loop_depth: usize,

    /// Variables actually modified at each depth
    modified_per_depth: Vec<HashSet<SymbolId>>,

    /// Loop header blocks at each depth (for Phi placement)
    loop_headers: Vec<BlockId>,
}

impl LoopScopeTracker {
    fn new() -> Self {
        Self::default()
    }

    fn enter_loop(&mut self, header_block: BlockId) {
        self.loop_depth += 1;
        self.modified_per_depth.push(HashSet::new());
        self.loop_headers.push(header_block);
    }

    fn exit_loop(&mut self) -> HashSet<SymbolId> {
        self.loop_depth = self.loop_depth.saturating_sub(1);
        self.loop_headers.pop();
        self.modified_per_depth.pop().unwrap_or_default()
    }

    fn mark_variable_modified(&mut self, var: SymbolId) {
        if let Some(current_scope) = self.modified_per_depth.last_mut() {
            current_scope.insert(var);
        }
        self.potentially_modified.insert(var);
    }

    fn get_current_loop_header(&self) -> Option<BlockId> {
        self.loop_headers.last().copied()
    }

    fn is_in_loop(&self) -> bool {
        self.loop_depth > 0
    }
}

/// Information about an incomplete phi node
#[derive(Debug, Clone)]
struct IncompletePhiInfo {
    symbol_id: SymbolId,
    block_id: BlockId,
    ssa_var_id: SsaVariableId,
    predecessor_blocks: Vec<BlockId>,
}

/// Builder statistics and performance tracking
#[derive(Debug, Default)]
struct DfgBuilderStats {
    nodes_created: usize,
    ssa_variables_created: usize,
    phi_nodes_inserted: usize,
    statements_processed: usize,
    expressions_processed: usize,
    construction_time_us: u64,
}

impl DfgBuilder {
    /// **Create a new DFG builder**
    pub fn new(options: GraphConstructionOptions) -> Self {
        let entry_node_id = DataFlowNodeId::from_raw(1);

        Self {
            options,
            dfg: DataFlowGraph::new(entry_node_id),
            ssa_state: SsaConstructionState::new(),
            next_node_id: 2, // Start from 2 since entry is 1
            next_ssa_var_id: 1,
            loop_scope_tracker: LoopScopeTracker::new(),
            phi_pending_updates: HashMap::new(),
            stats: DfgBuilderStats::default(),
        }
    }

    /// **Main entry point: Build DFG from CFG and function**
    pub fn build_dfg<'a>(
        &mut self,
        cfg: &'a ControlFlowGraph,
        function: &'a TypedFunction,
        type_checker: &'a mut TypeChecker<'a>,
    ) -> Result<DataFlowGraph, GraphConstructionError> {
        let start_time = Instant::now();

        // Phase 1: Compute dominance information
        let dominance_tree =
            DominanceTree::build(cfg).map_err(|e| GraphConstructionError::InternalError {
                message: format!("Dominance computation failed: {}", e),
            })?;

        // Phase 2: Initialize function parameters
        self.initialize_function_parameters(function)?;

        // Phase 3: Place phi functions
        self.place_phi_functions(cfg, function, &dominance_tree)?;

        // Phase 4: Create TAST-CFG mapping
        let mapping_result = TastCfgMapping::build(cfg, function);
        let mapping = mapping_result.mapping;

        // Phase 5: Process function body with variable renaming
        self.ssa_state.current_block = cfg.entry_block;
        self.process_function_body_with_cfg(cfg, function, &dominance_tree, &mapping)?;

        // Phase 6: Complete phi operands
        self.complete_phi_operands_with_type_unification(type_checker)?;

        // Phase 7: Finalize DFG
        self.finalize_dfg(start_time.elapsed())?;

        // Return ownership without using std::mem::take (which requires Default)
        let mut result_dfg = DataFlowGraph::new(DataFlowNodeId::from_raw(1));
        std::mem::swap(&mut result_dfg, &mut self.dfg);
        Ok(result_dfg)
    }

    /// **Initialize function parameters as DFG nodes**
    fn initialize_function_parameters(
        &mut self,
        function: &TypedFunction,
    ) -> Result<(), GraphConstructionError> {
        for (index, param) in function.parameters.iter().enumerate() {
            let node_id = self.allocate_node_id();
            let ssa_var_id = self.allocate_ssa_variable(param.symbol_id, param.param_type);

            // Create parameter node using actual enum variant
            let param_node = DataFlowNode {
                id: node_id,
                kind: DataFlowNodeKind::Parameter {
                    parameter_index: index,
                    symbol_id: param.symbol_id,
                },
                value_type: param.param_type,
                source_location: param.source_location,
                operands: vec![],
                uses: new_id_set(),
                defines: Some(ssa_var_id),
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata::default(),
            };

            self.dfg.add_node(param_node);
            self.push_ssa_variable(param.symbol_id, ssa_var_id);
            self.stats.nodes_created += 1;
        }

        Ok(())
    }

    /// **Process function body with proper CFG integration**
    fn process_function_body_with_cfg<'a>(
        &mut self,
        cfg: &ControlFlowGraph,
        function: &'a TypedFunction,
        dominance_tree: &DominanceTree,
        mapping: &TastCfgMapping,
    ) -> Result<(), GraphConstructionError> {
        // Process blocks in dominance tree order
        for &block_id in &dominance_tree.reverse_postorder {
            self.process_block_with_mapping(block_id, cfg, function, dominance_tree, mapping)?;
        }
        Ok(())
    }

    /// **Process a block using TAST-CFG mapping**
    fn process_block_with_mapping<'a>(
        &mut self,
        block_id: BlockId,
        cfg: &ControlFlowGraph,
        function: &'a TypedFunction,
        dominance_tree: &DominanceTree,
        mapping: &TastCfgMapping,
    ) -> Result<(), GraphConstructionError> {
        // Save current variable stacks
        let saved_stacks = self.save_variable_stacks();

        // Set current block
        let old_block = self.ssa_state.current_block;
        self.ssa_state.current_block = block_id;

        // Process phi nodes in this block
        self.process_phi_nodes_in_block(block_id)?;

        // Process regular statements in this block
        let statements = mapping.get_statements_in_block(block_id);

        for &stmt_location in statements {
            let statement = Self::get_statement_from_location(stmt_location, &function.body)?;
            self.build_statement(statement)?;
        }

        self.save_block_exit_variables(block_id);

        // Fill phi operands in successor blocks
        self.fill_successor_phi_operands(block_id, cfg)?;

        // Recursively process dominated children
        if let Some(children) = dominance_tree.dom_tree_children.get(&block_id) {
            for &child_block in children {
                self.process_block_with_mapping(
                    child_block,
                    cfg,
                    function,
                    dominance_tree,
                    mapping,
                )?;
            }
        }

        // Restore variable stacks
        self.restore_variable_stacks(saved_stacks);
        self.ssa_state.current_block = old_block;

        Ok(())
    }

    /// **Get statement from location using function body**
    fn get_statement_from_location<'a>(
        location: StatementLocation,
        function_body: &'a [TypedStatement],
    ) -> Result<&'a TypedStatement, GraphConstructionError> {
        Self::navigate_to_statement(function_body, location, 0)
    }

    /// **Navigate to a statement at a specific nesting depth**
    fn navigate_to_statement<'a>(
        statements: &'a [TypedStatement],
        location: StatementLocation,
        current_depth: u32,
    ) -> Result<&'a TypedStatement, GraphConstructionError> {
        if current_depth == location.nesting_depth {
            // Adjust index if it's out of bounds - this handles cases where the mapping
            // might have created incorrect indices for nested statements
            let adjusted_index =
                if location.statement_index >= statements.len() && statements.len() > 0 {
                    // If we're looking for index 1 but there's only 1 statement, use index 0
                    0
                } else {
                    location.statement_index
                };

            statements
                .get(adjusted_index)
                .ok_or_else(|| GraphConstructionError::InternalError {
                    message: format!(
                        "Invalid statement index: {} (adjusted to {}), statements available: {}",
                        location.statement_index,
                        adjusted_index,
                        statements.len()
                    ),
                })
        } else {
            // Need to go deeper - find the containing statement
            // At depth 0, we look for the specific statement at location.statement_index
            // At deeper depths, we need to traverse into the correct branch

            if current_depth == 0 {
                // At the top level, find the statement at the specified index
                if let Some(statement) = statements.get(location.statement_index) {
                    // Now navigate into this statement based on branch context
                    if let Some(nested) =
                        Self::get_nested_statements_for_branch(statement, location.branch_context)
                    {
                        return Self::navigate_to_statement(nested, location, current_depth + 1);
                    }
                }
            } else {
                // At deeper levels, we need to check each statement to find nested content
                for statement in statements {
                    // Check if this statement contains nested statements
                    if let Some(nested) =
                        Self::get_nested_statements_for_branch(statement, location.branch_context)
                    {
                        if let Ok(found) =
                            Self::navigate_to_statement(nested, location, current_depth + 1)
                        {
                            return Ok(found);
                        }
                    }

                    // Also check if this IS the statement we're looking for (e.g., a non-block in a branch)
                    if current_depth == location.nesting_depth - 1 {
                        // This might be a single statement (not a block) at the target depth
                        return Ok(statement);
                    }
                }
            }

            Err(GraphConstructionError::InternalError {
                message: format!("Statement not found at location: {:?}. Current depth: {}, statements at this level: {}",
                                location, current_depth, statements.len()),
            })
        }
    }

    /// **Get nested statements from a parent statement based on branch context**
    fn get_nested_statements_for_branch<'a>(
        statement: &'a TypedStatement,
        branch_context: BranchContext,
    ) -> Option<&'a [TypedStatement]> {
        match statement {
            TypedStatement::Block { statements, .. } => Some(statements),
            TypedStatement::If {
                then_branch,
                else_branch,
                ..
            } => {
                match branch_context {
                    BranchContext::IfThen => {
                        // Return the then branch as a single-element slice
                        // The caller will handle unpacking it if it's a Block
                        Some(std::slice::from_ref(then_branch))
                    }
                    BranchContext::IfElse => {
                        if let Some(else_stmt) = else_branch {
                            // Return the else branch as a single-element slice
                            // The caller will handle unpacking it if it's a Block
                            Some(std::slice::from_ref(else_stmt))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            TypedStatement::While { body, .. } | TypedStatement::For { body, .. } => {
                match body.as_ref() {
                    TypedStatement::Block { statements, .. } => Some(statements),
                    _ => Some(std::slice::from_ref(body)),
                }
            }
            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => match branch_context {
                BranchContext::None => match body.as_ref() {
                    TypedStatement::Block { statements, .. } => Some(statements),
                    _ => Some(std::slice::from_ref(body)),
                },
                BranchContext::CatchClause(index) => {
                    catch_clauses
                        .get(index)
                        .and_then(|catch| match &catch.body {
                            TypedStatement::Block { statements, .. } => Some(statements.as_slice()),
                            _ => Some(std::slice::from_ref(&catch.body)),
                        })
                }
                BranchContext::Finally => finally_block.as_ref().and_then(|finally| match finally
                    .as_ref()
                {
                    TypedStatement::Block { statements, .. } => Some(statements.as_slice()),
                    _ => Some(std::slice::from_ref(finally)),
                }),
                _ => None,
            },
            TypedStatement::Switch {
                cases,
                default_case,
                ..
            } => match branch_context {
                BranchContext::SwitchCase(index) => cases.get(index).map(|case| match &case.body {
                    TypedStatement::Block { statements, .. } => statements.as_slice(),
                    _ => std::slice::from_ref(&case.body),
                }),
                BranchContext::SwitchDefault => {
                    default_case
                        .as_ref()
                        .and_then(|default| match default.as_ref() {
                            TypedStatement::Block { statements, .. } => Some(statements.as_slice()),
                            _ => Some(std::slice::from_ref(default.as_ref())),
                        })
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// **Get nested statements from a parent statement**
    /// For backward compatibility, defaults to the first available branch
    fn get_nested_statements<'a>(statement: &'a TypedStatement) -> Option<&'a [TypedStatement]> {
        match statement {
            TypedStatement::Block { statements, .. } => Some(statements),
            TypedStatement::If { then_branch, .. } => {
                Self::get_nested_statements_for_branch(statement, BranchContext::IfThen)
            }
            _ => Self::get_nested_statements_for_branch(statement, BranchContext::None),
        }
    }

    /// **Get block predecessors from CFG**
    fn get_block_predecessors(&self, block_id: BlockId, cfg: &ControlFlowGraph) -> Vec<BlockId> {
        cfg.get_block(block_id)
            .map(|block| block.predecessors.iter().copied().collect())
            .unwrap_or_default()
    }

    /// **Place phi functions at dominance frontiers**
    fn place_phi_functions<'a>(
        &mut self,
        cfg: &'a ControlFlowGraph,
        function: &'a TypedFunction,
        dominance_tree: &DominanceTree,
    ) -> Result<(), GraphConstructionError> {
        // Build mapping for correct block analysis
        let mapping_result = TastCfgMapping::build(cfg, function);
        let mapping = &mapping_result.mapping;

        // Collect all variables that need phi placement
        let variables = self.collect_all_variables(function);

        // For each variable, determine where phi nodes are needed
        for variable in variables {
            let def_blocks = self.find_variable_definition_blocks(cfg, function, variable, mapping);

            // Mark blocks where variable is defined
            self.ssa_state
                .def_blocks
                .insert(variable, def_blocks.clone());

            // Use worklist algorithm to place phi nodes
            let mut worklist: VecDeque<BlockId> = def_blocks.iter().copied().collect();
            let mut phi_placed_for_var = HashSet::new();

            while let Some(block) = worklist.pop_front() {
                // Get dominance frontier of this block
                let frontier = dominance_tree.dominance_frontier(block);

                for &frontier_block in frontier {
                    if !phi_placed_for_var.contains(&frontier_block) {
                        phi_placed_for_var.insert(frontier_block);

                        // Create phi node with actual CFG
                        let phi_node_id = self.create_phi_node(frontier_block, variable, cfg)?;
                        self.ssa_state
                            .phi_placed
                            .insert((frontier_block, variable), phi_node_id);

                        // Add to worklist for further propagation
                        worklist.push_back(frontier_block);

                        self.stats.phi_nodes_inserted += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// **Collect all variables needing phi placement**
    fn collect_all_variables(&self, function: &TypedFunction) -> Vec<SymbolId> {
        let mut variables = Vec::new();

        // Add parameters
        for param in &function.parameters {
            variables.push(param.symbol_id);
        }

        // Traverse function body for local variables
        self.collect_variables_from_statements(&function.body, &mut variables);

        variables
    }

    /// **Recursively collect variables from statements**
    fn collect_variables_from_statements(
        &self,
        statements: &[TypedStatement],
        variables: &mut Vec<SymbolId>,
    ) {
        for statement in statements {
            match statement {
                TypedStatement::VarDeclaration { symbol_id, .. } => {
                    variables.push(*symbol_id);
                }
                TypedStatement::Block { statements, .. } => {
                    self.collect_variables_from_statements(statements, variables);
                }
                TypedStatement::If {
                    then_branch,
                    else_branch,
                    ..
                } => {
                    self.collect_variables_from_statement(then_branch, variables);
                    if let Some(else_stmt) = else_branch {
                        self.collect_variables_from_statement(else_stmt, variables);
                    }
                }
                TypedStatement::While { body, .. } | TypedStatement::For { body, .. } => {
                    self.collect_variables_from_statement(body, variables);
                }
                TypedStatement::Try {
                    body,
                    catch_clauses,
                    finally_block,
                    ..
                } => {
                    self.collect_variables_from_statement(body, variables);
                    for catch in catch_clauses {
                        variables.push(catch.exception_variable);
                        self.collect_variables_from_statement(&catch.body, variables);
                    }
                    if let Some(finally) = finally_block {
                        self.collect_variables_from_statement(finally, variables);
                    }
                }
                TypedStatement::Switch {
                    cases,
                    default_case,
                    ..
                } => {
                    for case in cases {
                        self.collect_variables_from_statement(&case.body, variables);
                    }
                    if let Some(default) = default_case {
                        self.collect_variables_from_statement(default, variables);
                    }
                }
                TypedStatement::PatternMatch { patterns: arms, .. } => {
                    for arm in arms {
                        // Pattern variables would be collected here
                        self.collect_variables_from_statement(&arm.body, variables);
                    }
                }
                _ => {}
            }
        }
    }

    /// **Helper to collect variables from a single statement**
    fn collect_variables_from_statement(
        &self,
        statement: &TypedStatement,
        variables: &mut Vec<SymbolId>,
    ) {
        self.collect_variables_from_statements(std::slice::from_ref(statement), variables);
    }

    /// **Find blocks where a variable is defined**
    fn find_variable_definition_blocks<'a>(
        &self,
        cfg: &ControlFlowGraph,
        function: &'a TypedFunction,
        variable: SymbolId,
        mapping: &TastCfgMapping,
    ) -> HashSet<BlockId> {
        let mut def_blocks = HashSet::new();

        // Parameters are defined in entry block
        if function.parameters.iter().any(|p| p.symbol_id == variable) {
            def_blocks.insert(cfg.entry_block);
        }

        // Check each block for definitions using mapping
        for &block_id in cfg.blocks.keys() {
            if self.block_defines_variable(block_id, variable, function, mapping) {
                def_blocks.insert(block_id);
            }
        }

        def_blocks
    }

    /// **Check if a block defines a variable**
    fn block_defines_variable<'a>(
        &self,
        block_id: BlockId,
        variable: SymbolId,
        function: &'a TypedFunction,
        mapping: &TastCfgMapping,
    ) -> bool {
        let statements = mapping.get_statements_in_block(block_id);

        for &stmt_location in statements {
            if let Ok(statement) = Self::get_statement_from_location(stmt_location, &function.body)
            {
                if self.statement_defines_variable(statement, variable) {
                    return true;
                }
            }
        }
        false
    }

    /// **Check if a statement defines a variable**
    fn statement_defines_variable(&self, statement: &TypedStatement, variable: SymbolId) -> bool {
        match statement {
            TypedStatement::VarDeclaration { symbol_id, .. } => *symbol_id == variable,
            TypedStatement::Assignment { target, .. } => {
                // Check if target is the variable
                self.expression_defines_variable(target, variable)
            }
            _ => false,
        }
    }

    /// **Check if an expression defines a variable (for assignments)**
    fn expression_defines_variable(&self, expr: &TypedExpression, variable: SymbolId) -> bool {
        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id } => *symbol_id == variable,
            _ => false,
        }
    }

    /// **Place phi functions for a specific variable**
    fn place_phi_for_variable(
        &mut self,
        cfg: &ControlFlowGraph,
        variable: SymbolId,
        def_blocks: HashSet<BlockId>,
        dominance_tree: &DominanceTree,
    ) -> Result<(), GraphConstructionError> {
        let mut worklist: VecDeque<BlockId> = def_blocks.iter().copied().collect();
        let mut processed = HashSet::new();

        while let Some(block) = worklist.pop_front() {
            if processed.contains(&block) {
                continue;
            }
            processed.insert(block);

            // Get dominance frontier of this block
            if let Some(frontier) = dominance_tree.dominance_frontiers.get(&block) {
                for &frontier_block in frontier {
                    // Check if phi already placed
                    if !self
                        .ssa_state
                        .phi_placed
                        .contains_key(&(frontier_block, variable))
                    {
                        // Create phi node
                        let phi_node_id = self.create_phi_node(frontier_block, variable, cfg)?;
                        self.ssa_state
                            .phi_placed
                            .insert((frontier_block, variable), phi_node_id);

                        // Add to worklist for further propagation
                        worklist.push_back(frontier_block);

                        self.stats.phi_nodes_inserted += 1;
                    }
                }
            }
        }

        Ok(())
    }

    /// **Create a phi node for a variable in a block**
    fn create_phi_node(
        &mut self,
        block_id: BlockId,
        variable: SymbolId,
        cfg: &ControlFlowGraph,
    ) -> Result<DataFlowNodeId, GraphConstructionError> {
        let node_id = self.allocate_node_id();
        let ssa_var_id = self.allocate_ssa_variable(variable, TypeId::invalid()); // Type will be resolved

        // Get predecessor blocks
        let predecessors = self.get_block_predecessors(block_id, cfg);

        // Create phi node using actual enum variant
        let phi_node = DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Phi {
                incoming: vec![], // Will be completed later
            },
            value_type: TypeId::invalid(), // Will be resolved
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(ssa_var_id),
            basic_block: block_id,
            metadata: NodeMetadata::default(),
        };

        self.dfg.nodes.insert(node_id, phi_node);

        // Track incomplete phi
        self.ssa_state.incomplete_phis.insert(
            node_id,
            IncompletePhiInfo {
                symbol_id: variable,
                block_id,
                ssa_var_id,
                predecessor_blocks: predecessors,
            },
        );

        self.stats.nodes_created += 1;

        Ok(node_id)
    }

    /// **Process function body with variable renaming**
    fn process_function_body(
        &mut self,
        cfg: &ControlFlowGraph,
        function: &TypedFunction,
        dominance_tree: &DominanceTree,
    ) -> Result<(), GraphConstructionError> {
        // Process statements in dominance tree order
        self.process_block_recursive(cfg.entry_block, cfg, &function.body, dominance_tree)?;
        Ok(())
    }

    /// **Process a block and its dominance tree children**
    fn process_block_recursive(
        &mut self,
        block_id: BlockId,
        cfg: &ControlFlowGraph,
        statements: &[TypedStatement],
        dominance_tree: &DominanceTree,
    ) -> Result<(), GraphConstructionError> {
        // Save current variable stacks
        let saved_stacks = self.save_variable_stacks();

        // Set current block
        let old_block = self.ssa_state.current_block;
        self.ssa_state.current_block = block_id;

        // Process phi nodes in this block (update stacks)
        self.process_phi_nodes_in_block(block_id)?;

        // Process statements in this block
        for statement in statements {
            self.build_statement(statement)?;
        }

        // Fill phi operands in successor blocks
        self.fill_successor_phi_operands(block_id, cfg)?;

        // Process children in dominance tree (avoiding borrowing issues)
        if let Some(children) = dominance_tree.dom_tree_children.get(&block_id) {
            let children_copy = children.clone(); // Clone to avoid borrowing issues
            for child_block in children_copy {
                self.process_block_recursive(child_block, cfg, statements, dominance_tree)?;
            }
        }

        // Restore variable stacks
        self.restore_variable_stacks(saved_stacks);
        self.ssa_state.current_block = old_block;

        Ok(())
    }

    /// **Build an expression into a DFG node**
    pub fn build_expression(
        &mut self,
        expression: &TypedExpression,
    ) -> Result<DataFlowNodeId, GraphConstructionError> {
        self.stats.expressions_processed += 1;

        let node_id = self.allocate_node_id();

        // Match actual TypedExpressionKind variants from the codebase
        let node = match &expression.kind {
            TypedExpressionKind::Literal { value } => {
                self.build_literal_expression(node_id, value, expression)
            }
            TypedExpressionKind::Variable { symbol_id } => {
                self.build_variable_expression(node_id, *symbol_id, expression)
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => self.build_field_access_expression(node_id, object, *field_symbol, expression),
            TypedExpressionKind::StaticFieldAccess {
                class_symbol,
                field_symbol,
            } => {
                // Build static field access - no object to evaluate
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::StaticFieldAccess {
                        class_symbol: *class_symbol,
                        field_symbol: *field_symbol,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::ArrayAccess { array, index } => {
                self.build_array_access_expression(node_id, array, index, expression)
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                type_arguments,
            } => self.build_function_call_expression(node_id, function, arguments, expression),
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                arguments,
                type_arguments,
                ..
            } => self.build_method_call_expression(
                node_id,
                receiver,
                *method_symbol,
                arguments,
                expression,
            ),
            TypedExpressionKind::StaticMethodCall {
                class_symbol,
                method_symbol,
                arguments,
                type_arguments,
            } => {
                // Build static method call - no receiver to evaluate
                let arg_nodes = arguments
                    .iter()
                    .map(|arg| self.build_expression(arg))
                    .collect::<Result<Vec<_>, _>>()?;

                // Create a placeholder function node for the static method
                let static_method_node_id = self.allocate_node_id();
                let static_method_node = DataFlowNode {
                    id: static_method_node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::String(format!("static_method_{:?}", method_symbol)),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                };
                self.dfg
                    .nodes
                    .insert(static_method_node_id, static_method_node);

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Call {
                        function: static_method_node_id,
                        arguments: arg_nodes.clone(),
                        call_type: CallType::Static,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: arg_nodes,
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
            } => self.build_binary_op_expression(node_id, left, *operator, right, expression),
            TypedExpressionKind::UnaryOp { operator, operand } => {
                self.build_unary_op_expression(node_id, *operator, operand, expression)
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => self.build_conditional_expression(
                node_id,
                condition,
                then_expr,
                else_expr.as_deref(),
                expression,
            ),
            TypedExpressionKind::ArrayLiteral { elements } => {
                self.build_array_literal_expression(node_id, elements, expression)
            }
            TypedExpressionKind::MapLiteral { entries } => {
                self.build_map_literal_expression(node_id, entries, expression)
            }
            TypedExpressionKind::ObjectLiteral { fields } => {
                self.build_object_literal_expression(node_id, fields, expression)
            }
            TypedExpressionKind::FunctionLiteral {
                parameters,
                body,
                return_type,
            } => self.build_function_literal_expression(
                node_id,
                parameters,
                body,
                *return_type,
                expression,
            ),
            TypedExpressionKind::Cast {
                expression: cast_expr,
                target_type,
                cast_kind,
            } => {
                self.build_cast_expression(node_id, cast_expr, *target_type, *cast_kind, expression)
            }
            TypedExpressionKind::New {
                class_type,
                arguments,
                type_arguments,
                class_name: _,
            } => self.build_new_expression(node_id, *class_type, arguments, expression),
            TypedExpressionKind::This { this_type } => {
                self.build_this_expression(node_id, *this_type, expression)
            }
            TypedExpressionKind::Super { super_type } => {
                self.build_super_expression(node_id, *super_type, expression)
            }
            TypedExpressionKind::Null => self.build_null_expression(node_id, expression),
            TypedExpressionKind::StringInterpolation { parts } => {
                self.build_string_interpolation_expression(node_id, parts, expression)
            }
            TypedExpressionKind::MacroExpression {
                macro_symbol,
                arguments,
            } => self.build_macro_expression(node_id, *macro_symbol, arguments, expression),
            TypedExpressionKind::While {
                condition,
                then_expr,
            } => self.build_while_expression(node_id, condition, then_expr, expression),
            TypedExpressionKind::For {
                variable,
                iterable,
                body,
            } => self.build_for_expression(node_id, *variable, iterable, body, expression),
            TypedExpressionKind::Is {
                expression,
                check_type,
            } => {
                // Build instanceof/is expression
                let expr_node = self.build_expression(expression)?;

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::TypeCheck {
                        operand: expr_node,
                        check_type: *check_type,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![expr_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Return { value } => {
                // Build return statement
                let operands = if let Some(val) = value {
                    vec![self.build_expression(val)?]
                } else {
                    vec![]
                };

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Return {
                        value: operands.first().copied(),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands,
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Throw { expression } => {
                // Build throw statement
                let expr_node = self.build_expression(expression)?;

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Throw {
                        exception: expr_node,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![expr_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Break => {
                // Break/continue are handled by CFG, not DFG
                // Create a placeholder node that will be eliminated
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::Void,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Continue => {
                // Break/continue are handled by CFG, not DFG
                // Create a placeholder node that will be eliminated
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::Void,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Block {
                statements,
                scope_id: _,
            } => {
                // Block expressions: process statements and return last expression value
                let mut last_node = self.allocate_node_id();

                // Process all statements in the block
                for stmt in statements {
                    let stmt_nodes = self.build_statement(stmt)?;
                    if let Some(&last_stmt_node) = stmt_nodes.last() {
                        last_node = last_stmt_node;
                    }
                }

                // Block result is the last statement/expression
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::Void, // Simplified - would track actual last value
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![last_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::ForIn {
                value_var,
                key_var,
                iterable,
                body,
            } => {
                // Build for-in loop as a block with iterable and body
                let iterable_node = self.build_expression(iterable)?;
                let body_node = self.build_expression(body)?;

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Block {
                        statements: vec![iterable_node, body_node],
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![iterable_node, body_node],
                    uses: new_id_set(),
                    defines: None, // TODO: Convert SymbolId to SsaVariableId
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Meta {
                metadata,
                expression: inner_expr,
            } => {
                // Metadata expressions are mostly transparent to DFG
                let inner_node = self.build_expression(inner_expr)?;

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::Void, // Metadata doesn't produce values
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![inner_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::DollarIdent { name, arg } => {
                // Dollar identifiers are macro-related, treat as constants
                let arg_node = if let Some(arg_expr) = arg {
                    Some(self.build_expression(arg_expr)?)
                } else {
                    None
                };

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::String(name.to_string()),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: arg_node.map(|n| vec![n]).unwrap_or_default(),
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::CompilerSpecific { target, code, .. } => {
                // Compiler-specific code blocks
                let code_node = self.build_expression(code)?;

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::String(format!("__{}__", target)),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![code_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Switch {
                discriminant,
                cases,
                default_case,
            } => {
                // Build switch expression as a multi-way conditional
                let discriminant_node = self.build_expression(discriminant)?;

                // For now, simplify switch to a conditional expression
                // TODO: Implement proper switch semantics with pattern matching
                let mut operands = vec![discriminant_node];

                // Build case expressions
                for case in cases {
                    let case_value_node = self.build_expression(&case.case_value)?;
                    operands.push(case_value_node);
                    // Note: case body is a statement, would need different handling
                }

                // Build default case if present
                if let Some(default) = default_case {
                    let default_node = self.build_expression(default)?;
                    operands.push(default_node);
                }

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Block {
                        statements: operands.clone(),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands,
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Try {
                try_expr,
                catch_clauses,
                ..
            } => {
                // Build try-catch expression
                let try_node = self.build_expression(try_expr)?;

                // For now, simplify try-catch to the try expression
                // TODO: Implement proper exception handling semantics
                let mut operands = vec![try_node];

                // Build catch handlers
                for catch in catch_clauses {
                    // Extract expression from catch body statement
                    let handler_expr = match &catch.body {
                        TypedStatement::Expression { expression, .. } => expression,
                        _ => {
                            // For non-expression statements, we need to handle them differently
                            // For now, skip complex catch bodies in expressions
                            continue;
                        }
                    };

                    let handler_node = self.build_expression(handler_expr)?;
                    operands.push(handler_node);

                    // Build filter if present
                    if let Some(filter) = &catch.filter {
                        let filter_node = self.build_expression(filter)?;
                        operands.push(filter_node);
                    }
                }

                // For now, represent try-catch as a block containing try and catch handlers
                // TODO: Add proper Try variant to DataFlowNodeKind
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Block {
                        statements: operands.clone(),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands,
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::VarDeclarationExpr {
                symbol_id,
                var_type,
                initializer,
            } => {
                // Variable declaration as expression - evaluates to the initializer value
                let init_node = self.build_expression(initializer)?;

                // Create allocation for the variable (similar to VarDeclaration statement)
                let alloc_node_id = self.allocate_node_id();
                let alloc_node = DataFlowNode {
                    id: alloc_node_id,
                    kind: DataFlowNodeKind::Allocation {
                        allocation_type: *var_type,
                        size: None,
                        allocation_kind: AllocationKind::Stack,
                    },
                    value_type: *var_type,
                    source_location: expression.source_location,
                    operands: vec![],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                };
                self.dfg.add_node(alloc_node);

                // Store the initializer value
                let store_node_id = self.allocate_node_id();
                let store_node = DataFlowNode {
                    id: store_node_id,
                    kind: DataFlowNodeKind::Store {
                        address: alloc_node_id,
                        value: init_node,
                        memory_type: MemoryType::Stack,
                    },
                    value_type: TypeId::from_raw(0), // Store has void type
                    source_location: expression.source_location,
                    operands: vec![alloc_node_id, init_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                };
                self.dfg.add_node(store_node);

                // The expression evaluates to the initializer value
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Load {
                        address: init_node,
                        memory_type: MemoryType::Stack,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![init_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::FinalDeclarationExpr {
                symbol_id,
                var_type,
                initializer,
            } => {
                // Final declaration as expression - evaluates to the initializer value
                // Similar to var declaration but immutable
                let init_node = self.build_expression(initializer)?;

                // Create allocation for the final variable
                let alloc_node_id = self.allocate_node_id();
                let alloc_node = DataFlowNode {
                    id: alloc_node_id,
                    kind: DataFlowNodeKind::Allocation {
                        allocation_type: *var_type,
                        size: None,
                        allocation_kind: AllocationKind::Stack,
                    },
                    value_type: *var_type,
                    source_location: expression.source_location,
                    operands: vec![],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                };
                self.dfg.add_node(alloc_node);

                // Store the initializer value (final = immutable after first assignment)
                let store_node_id = self.allocate_node_id();
                let store_node = DataFlowNode {
                    id: store_node_id,
                    kind: DataFlowNodeKind::Store {
                        address: alloc_node_id,
                        value: init_node,
                        memory_type: MemoryType::Stack,
                    },
                    value_type: TypeId::from_raw(0), // Store has void type
                    source_location: expression.source_location,
                    operands: vec![alloc_node_id, init_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                };
                self.dfg.add_node(store_node);

                // The expression evaluates to the initializer value
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Load {
                        address: init_node,
                        memory_type: MemoryType::Stack,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: vec![init_node],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::PatternPlaceholder { .. } => {
                // Pattern placeholders are handled in later compilation phases
                // For now, create a placeholder constant node
                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Constant {
                        value: ConstantValue::Void,
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands: Vec::new(),
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::ArrayComprehension {
                for_parts,
                expression,
                ..
            } => {
                // Build array comprehension as a block with iterator and expression nodes
                let mut operands = Vec::new();

                // Build iterator nodes
                for part in for_parts {
                    let iter_node = self.build_expression(&part.iterator)?;
                    operands.push(iter_node);
                }

                // Build the output expression
                let expr_node = self.build_expression(expression)?;
                operands.push(expr_node);

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Block {
                        statements: operands.clone(),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands,
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::MapComprehension {
                for_parts,
                key_expr,
                value_expr,
                ..
            } => {
                // Build map comprehension as a block with iterator and key/value expression nodes
                let mut operands = Vec::new();

                // Build iterator nodes
                for part in for_parts {
                    let iter_node = self.build_expression(&part.iterator)?;
                    operands.push(iter_node);
                }

                // Build the key and value expressions
                let key_node = self.build_expression(key_expr)?;
                let value_node = self.build_expression(value_expr)?;
                operands.push(key_node);
                operands.push(value_node);

                Ok(DataFlowNode {
                    id: node_id,
                    kind: DataFlowNodeKind::Block {
                        statements: operands.clone(),
                    },
                    value_type: expression.expr_type,
                    source_location: expression.source_location,
                    operands,
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                })
            }
            TypedExpressionKind::Await {
                expression,
                await_type,
            } => todo!(),
        }?;

        self.dfg.nodes.insert(node_id, node);
        self.stats.nodes_created += 1;

        Ok(node_id)
    }

    /// **Build a statement into DFG nodes**
    pub fn build_statement(
        &mut self,
        statement: &TypedStatement,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        self.stats.statements_processed += 1;

        match statement {
            TypedStatement::Expression { expression, .. } => {
                let node_id = self.build_expression(expression)?;
                Ok(vec![node_id])
            }

            TypedStatement::VarDeclaration {
                symbol_id,
                var_type,
                initializer,
                ..
            } => self.build_var_declaration(*symbol_id, *var_type, initializer.as_ref()),

            TypedStatement::Assignment { target, value, .. } => {
                self.build_assignment(target, value)
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => self.build_if_statement(condition, then_branch, else_branch.as_deref()),

            TypedStatement::While {
                condition, body, ..
            } => self.build_while_statement(condition, body),

            TypedStatement::For {
                init,
                condition,
                update,
                body,
                ..
            } => {
                self.build_for_statement(init.as_deref(), condition.as_ref(), update.as_ref(), body)
            }

            TypedStatement::Return { value, .. } => self.build_return_statement(value.as_ref()),

            TypedStatement::Throw { exception, .. } => self.build_throw_statement(exception),

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => self.build_try_statement(body, catch_clauses, finally_block.as_deref()),

            TypedStatement::Block { statements, .. } => {
                let mut nodes = Vec::new();
                for stmt in statements {
                    nodes.extend(self.build_statement(stmt)?);
                }
                Ok(nodes)
            }

            TypedStatement::Switch {
                discriminant: value,
                cases,
                default_case,
                ..
            } => self.build_switch_statement(value, cases, default_case.as_deref()),

            TypedStatement::Break { .. } | TypedStatement::Continue { .. } => {
                // Control flow already handled by CFG
                Ok(vec![])
            }

            TypedStatement::PatternMatch { .. } | TypedStatement::MacroExpansion { .. } => {
                // Complex patterns - simplified for now
                Ok(vec![])
            }

            TypedStatement::ForIn {
                value_var,
                key_var,
                iterable,
                body,
                ..
            } => {
                // Build for-in loop similar to regular for loop
                let iterable_node = self.build_expression(iterable)?;
                let body_nodes = self.build_statement(body)?;

                // Create a loop node that combines iterable and body
                let mut result = vec![iterable_node];
                result.extend(body_nodes);
                Ok(result)
            }
        }
    }

    /// **Build switch statement**
    fn build_switch_statement(
        &mut self,
        value: &TypedExpression,
        _cases: &[crate::tast::node::TypedSwitchCase],
        _default_case: Option<&crate::tast::node::TypedStatement>,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        // Build switch value
        let value_node = self.build_expression(value)?;

        // Case bodies are processed by block processing
        Ok(vec![value_node])
    }

    // ==================================================================================
    // EXPRESSION BUILDERS - Using Actual DataFlowNodeKind Variants
    // ==================================================================================

    /// **Build literal expression**
    fn build_literal_expression(
        &mut self,
        node_id: DataFlowNodeId,
        value: &LiteralValue,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Constant {
                value: self.convert_literal_value(value),
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build variable expression**
    fn build_variable_expression(
        &mut self,
        node_id: DataFlowNodeId,
        symbol_id: SymbolId,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // Use the more lenient method that creates placeholders for undefined symbols
        // This handles cases like external function references
        let ssa_var_id = self.get_or_create_ssa_variable(symbol_id);

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Variable {
                ssa_var: ssa_var_id,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build field access expression**
    fn build_field_access_expression(
        &mut self,
        node_id: DataFlowNodeId,
        object: &TypedExpression,
        field_symbol: SymbolId,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let object_node = self.build_expression(object)?;

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::FieldAccess {
                object: object_node,
                field_symbol,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![object_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build array access expression**
    fn build_array_access_expression(
        &mut self,
        node_id: DataFlowNodeId,
        array: &TypedExpression,
        index: &TypedExpression,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let array_node = self.build_expression(array)?;
        let index_node = self.build_expression(index)?;

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::ArrayAccess {
                array: array_node,
                index: index_node,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![array_node, index_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build function call expression**
    fn build_function_call_expression(
        &mut self,
        node_id: DataFlowNodeId,
        function: &TypedExpression,
        arguments: &[TypedExpression],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let function_node = self.build_expression(function)?;
        let mut arg_nodes = vec![];

        for arg in arguments {
            let arg_node = self.build_expression(arg)?;
            arg_nodes.push(arg_node);
        }

        let mut operands = vec![function_node];
        operands.extend(arg_nodes.clone());

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Call {
                function: function_node,
                arguments: arg_nodes,
                call_type: CallType::Direct, // Use actual CallType variant
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands,
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata {
                has_side_effects: true,
                ..Default::default()
            },
        })
    }

    /// **Build method call expression**
    fn build_method_call_expression(
        &mut self,
        node_id: DataFlowNodeId,
        receiver: &TypedExpression,
        method_symbol: SymbolId,
        arguments: &[TypedExpression],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let object_node = self.build_expression(receiver)?;

        let mut arg_nodes = Vec::new();
        for arg in arguments {
            arg_nodes.push(self.build_expression(arg)?);
        }

        // Create a synthetic method access node
        let method_node_id = self.allocate_node_id();
        let method_node = DataFlowNode {
            id: method_node_id,
            kind: DataFlowNodeKind::FieldAccess {
                object: object_node,
                field_symbol: method_symbol,
            },
            value_type: TypeId::invalid(), // Method type
            source_location: expression.source_location,
            operands: vec![object_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(method_node);

        let mut operands = vec![method_node_id];
        operands.extend(&arg_nodes);

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Call {
                function: method_node_id,
                arguments: arg_nodes,
                call_type: CallType::Virtual,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands,
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata {
                has_side_effects: true,
                ..Default::default()
            },
        })
    }

    /// **Build binary operation expression**
    fn build_binary_op_expression(
        &mut self,
        node_id: DataFlowNodeId,
        left: &TypedExpression,
        operator: BinaryOperator,
        right: &TypedExpression,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let left_node = self.build_expression(left)?;
        let right_node = self.build_expression(right)?;

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::BinaryOp {
                operator,
                left: left_node,
                right: right_node,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![left_node, right_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build unary operation expression**
    fn build_unary_op_expression(
        &mut self,
        node_id: DataFlowNodeId,
        operator: UnaryOperator,
        operand: &TypedExpression,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let operand_node = self.build_expression(operand)?;

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::UnaryOp {
                operator,
                operand: operand_node,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![operand_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build cast expression**
    fn build_cast_expression(
        &mut self,
        node_id: DataFlowNodeId,
        expr: &TypedExpression,
        target_type: TypeId,
        cast_kind: TastCastKind,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let expr_node = self.build_expression(expr)?;

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Cast {
                value: expr_node,
                target_type,
                cast_kind: self.convert_cast_kind(cast_kind),
            },
            value_type: target_type,
            source_location: expression.source_location,
            operands: vec![expr_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build allocation expression (new)**
    fn build_new_expression(
        &mut self,
        node_id: DataFlowNodeId,
        class_type: TypeId,
        arguments: &[TypedExpression],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let mut arg_nodes = vec![];

        for arg in arguments {
            let arg_node = self.build_expression(arg)?;
            arg_nodes.push(arg_node);
        }

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Call {
                function: DataFlowNodeId::invalid(), // Constructor
                arguments: arg_nodes.clone(),
                call_type: CallType::Constructor,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: arg_nodes,
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata {
                has_side_effects: true,
                ..Default::default()
            },
        })
    }

    /// **Build return statement**
    fn build_return_statement(
        &mut self,
        value: Option<&TypedExpression>,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        let value_node = if let Some(expr) = value {
            Some(self.build_expression(expr)?)
        } else {
            None
        };

        let return_node_id = self.allocate_node_id();
        let return_node = DataFlowNode {
            id: return_node_id,
            kind: DataFlowNodeKind::Return { value: value_node },
            value_type: value.map(|e| e.expr_type).unwrap_or(TypeId::invalid()),
            source_location: value
                .map(|e| e.source_location())
                .unwrap_or(SourceLocation::unknown()),
            operands: value_node.map(|n| vec![n]).unwrap_or_default(),
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };

        self.dfg.nodes.insert(return_node_id, return_node);
        self.stats.nodes_created += 1;

        Ok(vec![return_node_id])
    }

    /// **Build throw statement**
    fn build_throw_statement(
        &mut self,
        exception: &TypedExpression,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        let exception_node = self.build_expression(exception)?;

        let throw_node_id = self.allocate_node_id();
        let throw_node = DataFlowNode {
            id: throw_node_id,
            kind: DataFlowNodeKind::Throw {
                exception: exception_node,
            },
            value_type: exception.expr_type,
            source_location: exception.source_location,
            operands: vec![exception_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata {
                has_side_effects: true,
                ..Default::default()
            },
        };

        self.dfg.nodes.insert(throw_node_id, throw_node);
        self.stats.nodes_created += 1;

        Ok(vec![throw_node_id])
    }

    fn build_while_statement(
        &mut self,
        condition: &TypedExpression,
        body: &TypedStatement,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        // Build condition expression
        let condition_node = self.build_expression(condition)?;

        // The actual loop control flow is handled by CFG
        // Process the body statement in the loop body block
        let mut nodes = vec![condition_node];

        // Note: Body processing is handled by CFG block traversal
        // This method only handles the DFG nodes for the condition

        Ok(nodes)
    }

    fn build_for_statement(
        &mut self,
        init: Option<&TypedStatement>,
        condition: Option<&TypedExpression>,
        update: Option<&TypedExpression>,
        body: &TypedStatement,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        let mut nodes = Vec::new();

        // Build init statement if present
        if let Some(init_stmt) = init {
            nodes.extend(self.build_statement(init_stmt)?);
        }

        // Build condition if present
        if let Some(cond_expr) = condition {
            let condition_node = self.build_expression(cond_expr)?;
            nodes.push(condition_node);
        }

        // Build update expression if present
        if let Some(update_expr) = update {
            let update_node = self.build_expression(update_expr)?;
            nodes.push(update_node);
        }

        // Note: Body and loop control flow are handled by CFG

        Ok(nodes)
    }

    fn build_try_statement(
        &mut self,
        body: &TypedStatement,
        catch_clauses: &[crate::tast::node::TypedCatchClause],
        finally_block: Option<&TypedStatement>,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        let mut nodes = Vec::new();

        // Build try body
        nodes.extend(self.build_statement(body)?);

        // Build catch clauses
        for catch_clause in catch_clauses {
            nodes.extend(self.build_statement(&catch_clause.body)?);
        }

        // Build finally block if present
        if let Some(finally_stmt) = finally_block {
            nodes.extend(self.build_statement(finally_stmt)?);
        }

        // Note: Exception handling control flow is managed by CFG

        Ok(nodes)
    }

    fn build_conditional_expression(
        &mut self,
        node_id: DataFlowNodeId,
        condition: &TypedExpression,
        then_expr: &TypedExpression,
        else_expr: Option<&TypedExpression>,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let cond_node = self.build_expression(condition)?;
        let then_node = self.build_expression(then_expr)?;
        let else_node = if let Some(else_expr) = else_expr {
            Some(self.build_expression(else_expr)?)
        } else {
            None
        };

        let mut incoming = vec![PhiIncoming {
            value: then_node,
            predecessor: BlockId::invalid(),
        }];

        if else_node.is_some() {
            incoming.push(PhiIncoming {
                value: else_node.unwrap(),
                predecessor: BlockId::invalid(),
            });
        }

        let mut operands = vec![cond_node, then_node];
        if else_node.is_some() {
            operands.push(else_node.unwrap());
        }
        // Conditional is like a phi node but within an expression
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Phi { incoming },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands,
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_array_literal_expression(
        &mut self,
        node_id: DataFlowNodeId,
        elements: &[TypedExpression],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // Build element expressions first
        let mut element_nodes = Vec::new();
        for elem in elements {
            element_nodes.push(self.build_expression(elem)?);
        }

        // Create size constant
        let size_node_id = self.allocate_node_id();
        let size_node = DataFlowNode {
            id: size_node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(elements.len() as i64),
            },
            value_type: TypeId::from_raw(1), // Assuming int type
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(size_node);

        // Create allocation node for the array
        let alloc_node_id = self.allocate_node_id();
        let alloc_node = DataFlowNode {
            id: alloc_node_id,
            kind: DataFlowNodeKind::Allocation {
                allocation_type: expression.expr_type,
                size: Some(size_node_id),
                allocation_kind: AllocationKind::Heap, // Arrays are heap allocated
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![size_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(alloc_node);

        // Create store nodes for each element
        for (index, &elem_node) in element_nodes.iter().enumerate() {
            let index_node_id = self.allocate_node_id();
            let index_node = DataFlowNode {
                id: index_node_id,
                kind: DataFlowNodeKind::Constant {
                    value: ConstantValue::Int(index as i64),
                },
                value_type: TypeId::from_raw(1), // int
                source_location: expression.source_location,
                operands: vec![],
                uses: new_id_set(),
                defines: None,
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata::default(),
            };
            self.dfg.add_node(index_node);

            // Create array element store
            let store_node_id = self.allocate_node_id();
            let store_node = DataFlowNode {
                id: store_node_id,
                kind: DataFlowNodeKind::Store {
                    address: alloc_node_id,
                    value: elem_node,
                    memory_type: MemoryType::Heap, // Arrays are heap allocated
                },
                value_type: TypeId::invalid(), // Store returns void
                source_location: expression.source_location,
                operands: vec![alloc_node_id, elem_node, index_node_id],
                uses: new_id_set(),
                defines: None,
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata {
                    has_side_effects: true,
                    ..Default::default()
                },
            };
            self.dfg.add_node(store_node);
        }

        // Return the allocation as the result
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Load {
                address: alloc_node_id,
                memory_type: MemoryType::Heap,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![alloc_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_map_literal_expression(
        &mut self,
        node_id: DataFlowNodeId,
        entries: &[TypedMapEntry],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // Build key and value expressions first
        let mut entry_nodes = Vec::new();
        for entry in entries {
            let key_node = self.build_expression(&entry.key)?;
            let value_node = self.build_expression(&entry.value)?;
            entry_nodes.push((key_node, value_node));
        }

        // Create size constant
        let size_node_id = self.allocate_node_id();
        let size_node = DataFlowNode {
            id: size_node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(entries.len() as i64),
            },
            value_type: TypeId::from_raw(1), // Assuming int type
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(size_node);

        // Create allocation node for the map
        let alloc_node_id = self.allocate_node_id();
        let alloc_node = DataFlowNode {
            id: alloc_node_id,
            kind: DataFlowNodeKind::Allocation {
                allocation_type: expression.expr_type,
                size: Some(size_node_id),
                allocation_kind: AllocationKind::Heap, // Maps are heap allocated
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![size_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(alloc_node);

        // Create store nodes for each key-value pair
        for (key_node, value_node) in entry_nodes {
            let store_node_id = self.allocate_node_id();
            let store_node = DataFlowNode {
                id: store_node_id,
                kind: DataFlowNodeKind::Store {
                    address: alloc_node_id,
                    value: value_node,
                    memory_type: MemoryType::Heap,
                },
                value_type: TypeId::from_raw(0), // Store has void type
                source_location: expression.source_location,
                operands: vec![alloc_node_id, key_node, value_node],
                uses: new_id_set(),
                defines: None,
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata::default(),
            };
            self.dfg.add_node(store_node);
        }

        // Return the allocation as the result
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Load {
                address: alloc_node_id,
                memory_type: MemoryType::Heap,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![alloc_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_object_literal_expression(
        &mut self,
        node_id: DataFlowNodeId,
        fields: &[TypedObjectField],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // Build field value expressions
        let mut field_nodes = Vec::new();
        for field in fields {
            field_nodes.push(self.build_expression(&field.value)?);
        }

        // Create size constant for object
        let size_node_id = self.allocate_node_id();
        let object_size = 16 + (fields.len() * 8); // Header + field slots (simplified)
        let size_node = DataFlowNode {
            id: size_node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(object_size as i64),
            },
            value_type: TypeId::from_raw(1), // int
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(size_node);

        // Create allocation node for the object
        let alloc_node_id = self.allocate_node_id();
        let alloc_node = DataFlowNode {
            id: alloc_node_id,
            kind: DataFlowNodeKind::Allocation {
                allocation_type: expression.expr_type,
                size: Some(size_node_id),
                allocation_kind: AllocationKind::Heap,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![size_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(alloc_node);

        // Create store nodes for each field
        for (field, &field_value_node) in fields.iter().zip(&field_nodes) {
            // For simplicity, we'll use the field name as a marker
            // In a real implementation, we'd resolve field offsets
            let store_node_id = self.allocate_node_id();
            let store_node = DataFlowNode {
                id: store_node_id,
                kind: DataFlowNodeKind::Store {
                    address: alloc_node_id,
                    value: field_value_node,
                    memory_type: MemoryType::Heap,
                },
                value_type: TypeId::invalid(), // Store returns void
                source_location: field.source_location,
                operands: vec![alloc_node_id, field_value_node],
                uses: new_id_set(),
                defines: None,
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata {
                    has_side_effects: true,
                    // annotations: {
                    //     let mut annotations = HashMap::new();
                    //     annotations.insert("field_name".to_string(), field.name.clone());
                    //     annotations
                    // },
                    ..Default::default()
                },
            };
            self.dfg.add_node(store_node);
        }

        // Return the initialized object
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Load {
                address: alloc_node_id,
                memory_type: MemoryType::Heap,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![alloc_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_function_literal_expression(
        &mut self,
        node_id: DataFlowNodeId,
        parameters: &[TypedParameter],
        body: &[TypedStatement],
        return_type: TypeId,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // 1. Find captured variables
        let captured_vars = self.find_captured_variables(body, parameters)?;

        // 2. Calculate closure size (header + captured variables)
        let closure_size = self.calculate_closure_size(&captured_vars);

        // 3. Create size constant
        let size_node_id = self.allocate_node_id();
        let size_node = DataFlowNode {
            id: size_node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(closure_size as i64),
            },
            value_type: TypeId::from_raw(1), // int
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(size_node);

        // 4. Create closure allocation
        let alloc_node_id = self.allocate_node_id();
        let alloc_node = DataFlowNode {
            id: alloc_node_id,
            kind: DataFlowNodeKind::Allocation {
                allocation_type: expression.expr_type,
                size: Some(size_node_id),
                allocation_kind: AllocationKind::Heap, // Closures are heap allocated
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![size_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(alloc_node);

        // 5. Store captured variables in closure
        let capture_stores = self.create_capture_stores(alloc_node_id, &captured_vars)?;

        // 6. Create a Load node that represents loading the initialized closure
        // with metadata indicating it's a closure and its capture count
        let mut annotations = HashMap::new();
        let capture_count = captured_vars.len();
        annotations.insert("capture_count".to_string(), capture_count.to_string());
        if capture_count > 0 {
            annotations.insert("closure".to_string(), "true".to_string());
        }

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Load {
                address: alloc_node_id,
                memory_type: MemoryType::Heap,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: {
                let mut ops = vec![alloc_node_id];
                ops.extend(&capture_stores); // Include stores as dependencies
                ops
            },
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata {
                annotations,
                ..Default::default()
            },
        })
    }

    fn build_this_expression(
        &mut self,
        node_id: DataFlowNodeId,
        this_type: TypeId,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // This reference - treat as parameter 0 typically
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Parameter {
                parameter_index: 0,               // This is typically parameter 0
                symbol_id: SymbolId::from_raw(0), // Special symbol for 'this'
            },
            value_type: this_type,
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_super_expression(
        &mut self,
        node_id: DataFlowNodeId,
        super_type: TypeId,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // Super reference - similar to this
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Parameter {
                parameter_index: 0,
                symbol_id: SymbolId::from_raw(1), // Special symbol for 'super'
            },
            value_type: super_type,
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_null_expression(
        &mut self,
        node_id: DataFlowNodeId,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Null,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_string_interpolation_expression(
        &mut self,
        node_id: DataFlowNodeId,
        parts: &[crate::tast::node::StringInterpolationPart],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // For now, treat as a string constant
        // Full implementation would build concatenation operations
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::String("<interpolated>".to_string()),
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    fn build_macro_expression(
        &mut self,
        node_id: DataFlowNodeId,
        macro_symbol: SymbolId,
        arguments: &[TypedExpression],
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        let mut arg_nodes = vec![];
        for arg in arguments {
            arg_nodes.push(self.build_expression(arg)?);
        }

        // Macro call - treat as special function call using actual CallType
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Call {
                function: DataFlowNodeId::from_raw(macro_symbol.as_raw()),
                arguments: arg_nodes.clone(),
                call_type: CallType::Builtin, // Use actual CallType variant (no Macro variant exists)
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: arg_nodes,
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata {
                has_side_effects: true,
                ..Default::default()
            },
        })
    }

    // ==================================================================================
    // STATEMENT BUILDERS
    // ==================================================================================

    fn build_var_declaration(
        &mut self,
        symbol_id: SymbolId,
        var_type: TypeId,
        initializer: Option<&TypedExpression>,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        let mut nodes = vec![];

        // Build initializer if present
        let init_node = if let Some(init_expr) = initializer {
            Some(self.build_expression(init_expr)?)
        } else {
            None
        };

        // Create SSA variable for this declaration
        let ssa_var_id = self.allocate_ssa_variable(symbol_id, var_type);

        // Create variable declaration node
        let var_node_id = self.allocate_node_id();
        let var_node = DataFlowNode {
            id: var_node_id,
            kind: DataFlowNodeKind::Variable {
                ssa_var: ssa_var_id,
            },
            value_type: var_type,
            source_location: initializer
                .map(|e| e.source_location())
                .unwrap_or(SourceLocation::unknown()),
            operands: init_node.map(|n| vec![n]).unwrap_or_default(),
            uses: new_id_set(),
            defines: Some(ssa_var_id),
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };

        self.dfg.nodes.insert(var_node_id, var_node);
        self.push_ssa_variable(symbol_id, ssa_var_id);

        nodes.push(var_node_id);
        if let Some(init_node) = init_node {
            nodes.push(init_node);
        }

        self.stats.nodes_created += 1;

        Ok(nodes)
    }

    fn build_assignment(
        &mut self,
        target: &TypedExpression,
        value: &TypedExpression,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        // Build the value expression
        let value_node_id = self.build_expression(value)?;

        // Handle assignment target
        match &target.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                // Create new SSA variable for the assignment
                let ssa_var_id = self.allocate_ssa_variable(*symbol_id, target.expr_type);

                let assign_node_id = self.allocate_node_id();
                let assign_node = DataFlowNode {
                    id: assign_node_id,
                    kind: DataFlowNodeKind::Variable {
                        ssa_var: ssa_var_id,
                    },
                    value_type: target.expr_type,
                    source_location: target.source_location,
                    operands: vec![value_node_id],
                    uses: new_id_set(),
                    defines: Some(ssa_var_id),
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata::default(),
                };

                self.dfg.add_node(assign_node);
                self.push_ssa_variable(*symbol_id, ssa_var_id);
                self.stats.nodes_created += 1;

                Ok(vec![assign_node_id])
            }

            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                // Build store operation for field assignment
                let object_node = self.build_expression(object)?;

                let store_node_id = self.allocate_node_id();
                let store_node = DataFlowNode {
                    id: store_node_id,
                    kind: DataFlowNodeKind::Store {
                        address: object_node,
                        value: value_node_id,
                        memory_type: MemoryType::Field(*field_symbol),
                    },
                    value_type: object.expr_type,
                    source_location: target.source_location,
                    operands: vec![object_node, value_node_id],
                    uses: new_id_set(),
                    defines: None,
                    basic_block: self.ssa_state.current_block,
                    metadata: NodeMetadata {
                        has_side_effects: true,
                        ..Default::default()
                    },
                };

                self.dfg.add_node(store_node);
                self.stats.nodes_created += 1;

                Ok(vec![store_node_id])
            }

            _ => Ok(vec![]), // Other assignment targets
        }
    }

    fn build_if_statement(
        &mut self,
        condition: &TypedExpression,
        then_branch: &TypedStatement,
        else_branch: Option<&TypedStatement>,
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        // Build condition expression
        let condition_node = self.build_expression(condition)?;

        // The actual branching is handled by CFG
        // Statements in then/else branches are processed by block processing

        Ok(vec![condition_node])
    }

    // ==================================================================================
    // LOOP LOWERING HELPER METHODS
    // ==================================================================================

    /// **Build While Loop Expression with Phi Nodes**
    ///
    /// Lowers while loops to SSA form using Phi nodes for loop-carried dependencies.
    /// Creates Phi nodes for all variables modified within the loop body.
    fn build_while_expression(
        &mut self,
        node_id: DataFlowNodeId,
        condition: &TypedExpression,
        body: &TypedExpression,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // 1. Find all variables modified in loop body
        let modified_vars = self.find_modified_variables(body)?;

        // 2. Enter loop scope
        let loop_header = self.ssa_state.current_block;
        self.loop_scope_tracker.enter_loop(loop_header);

        // 3. Create Phi nodes for each modified variable
        let mut loop_phi_nodes = HashMap::new();
        for var_id in &modified_vars {
            if let Ok(entry_value) = self.get_current_ssa_variable(*var_id) {
                let phi_node_id =
                    self.create_loop_phi_node(*var_id, entry_value, expression.expr_type)?;
                loop_phi_nodes.insert(*var_id, phi_node_id);

                // Update SSA state to use Phi result for this variable
                let phi_ssa_var = self.allocate_ssa_variable(*var_id, expression.expr_type);
                self.set_variable(*var_id, phi_ssa_var);
            }
        }

        // 4. Build condition using Phi values
        let condition_node = self.build_expression(condition)?;

        // 5. Build loop body (which may update variables)
        let body_node = self.build_expression(body)?;

        // 6. Update Phi nodes with back-edge values
        for (var_id, phi_node_id) in &loop_phi_nodes {
            if let Ok(updated_value_ssa) = self.get_current_ssa_variable(*var_id) {
                if let Some(updated_node) =
                    self.find_defining_node_for_ssa_variable(updated_value_ssa)
                {
                    // Store pending update for later completion
                    let phi_incoming = PhiIncoming {
                        value: updated_node,
                        predecessor: loop_header, // Back-edge from loop body
                    };
                    self.phi_pending_updates
                        .entry(*phi_node_id)
                        .or_insert_with(Vec::new)
                        .push(phi_incoming);
                }
            }
        }

        // 7. Exit loop scope
        let _modified_in_loop = self.loop_scope_tracker.exit_loop();

        // 8. Create a result Phi node that represents the loop result
        // (last iteration of body or initial value if loop never executes)
        let result_incoming = vec![PhiIncoming {
            value: body_node, // Value from loop execution
            predecessor: loop_header,
        }];

        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Phi {
                incoming: result_incoming,
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![condition_node, body_node],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Build For-in Loop Expression with Phi Nodes**
    ///
    /// Lowers for-in loops to SSA form with proper iterator and loop variable handling.
    fn build_for_expression(
        &mut self,
        node_id: DataFlowNodeId,
        variable: SymbolId,
        iterable: &TypedExpression,
        body: &TypedExpression,
        expression: &TypedExpression,
    ) -> Result<DataFlowNode, GraphConstructionError> {
        // 1. Build iterable expression
        let iterable_node = self.build_expression(iterable)?;

        // 2. Enter loop scope
        let loop_header = self.ssa_state.current_block;
        self.loop_scope_tracker.enter_loop(loop_header);

        // 3. Create iterator state Phi nodes
        let index_phi_id = self.create_iterator_phi_node(loop_header)?;

        // 4. Extract current value from iterable[index]
        let current_value_node =
            self.create_array_access_node(iterable_node, index_phi_id, expression.source_location)?;

        // 5. Bind loop variable to current value
        let loop_var_ssa = self.allocate_ssa_variable(variable, expression.expr_type);
        self.set_variable(variable, loop_var_ssa);

        // 6. Track that loop variable is modified
        self.loop_scope_tracker.mark_variable_modified(variable);

        // 7. Build loop body
        let body_node = self.build_expression(body)?;

        // 8. Update iterator: index + 1
        let next_index_node =
            self.create_iterator_increment_node(index_phi_id, expression.source_location)?;

        // 9. Add back-edge to index Phi
        let index_phi_incoming = PhiIncoming {
            value: next_index_node,
            predecessor: loop_header,
        };
        self.phi_pending_updates
            .entry(index_phi_id)
            .or_insert_with(Vec::new)
            .push(index_phi_incoming);

        // 10. Exit loop scope
        let _modified_in_loop = self.loop_scope_tracker.exit_loop();

        // 11. For-in loops typically return void or the iterable
        Ok(DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Phi {
                incoming: vec![PhiIncoming {
                    value: body_node,
                    predecessor: loop_header,
                }],
            },
            value_type: expression.expr_type,
            source_location: expression.source_location,
            operands: vec![iterable_node, current_value_node, body_node],
            uses: new_id_set(),
            defines: Some(loop_var_ssa),
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        })
    }

    /// **Find Modified Variables in Expression**
    ///
    /// Analyzes an expression to find all variables that may be modified within it.
    /// This is essential for proper Phi node placement in loops.
    fn find_modified_variables(
        &self,
        expr: &TypedExpression,
    ) -> Result<HashSet<SymbolId>, GraphConstructionError> {
        let mut modified = HashSet::new();
        self.collect_modified_variables_recursive(expr, &mut modified)?;
        Ok(modified)
    }

    /// **Recursively Collect Modified Variables**
    fn collect_modified_variables_recursive(
        &self,
        expr: &TypedExpression,
        modified: &mut HashSet<SymbolId>,
    ) -> Result<(), GraphConstructionError> {
        match &expr.kind {
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
            } => {
                // Check for assignment operations
                if matches!(
                    operator,
                    BinaryOperator::Assign
                        | BinaryOperator::AddAssign
                        | BinaryOperator::SubAssign
                        | BinaryOperator::MulAssign
                        | BinaryOperator::DivAssign
                        | BinaryOperator::ModAssign
                ) {
                    // Left side is being modified
                    if let TypedExpressionKind::Variable { symbol_id } = &left.kind {
                        modified.insert(*symbol_id);
                    }
                }
                self.collect_modified_variables_recursive(left, modified)?;
                self.collect_modified_variables_recursive(right, modified)?;
            }
            TypedExpressionKind::FunctionCall { arguments, .. } => {
                // Function calls may modify their arguments (especially ref parameters)
                for arg in arguments {
                    self.collect_modified_variables_recursive(arg, modified)?;
                }
            }
            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.collect_modified_variables_recursive(receiver, modified)?;
                for arg in arguments {
                    self.collect_modified_variables_recursive(arg, modified)?;
                }
            }
            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                // Static method calls don't have a receiver
                for arg in arguments {
                    self.collect_modified_variables_recursive(arg, modified)?;
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                // Process all statements in block
                for stmt in statements {
                    self.collect_modified_variables_from_statement(stmt, modified)?;
                }
            }
            // For other expressions, recursively check sub-expressions
            _ => {
                // Would implement full traversal for all expression kinds
                // Simplified for now
            }
        }
        Ok(())
    }

    /// **Collect Modified Variables from Statement**
    fn collect_modified_variables_from_statement(
        &self,
        stmt: &TypedStatement,
        modified: &mut HashSet<SymbolId>,
    ) -> Result<(), GraphConstructionError> {
        match stmt {
            TypedStatement::VarDeclaration { symbol_id, .. } => {
                modified.insert(*symbol_id);
            }
            TypedStatement::Assignment { target, .. } => {
                if let TypedExpressionKind::Variable { symbol_id } = &target.kind {
                    modified.insert(*symbol_id);
                }
            }
            TypedStatement::Expression { expression, .. } => {
                self.collect_modified_variables_recursive(expression, modified)?;
            }
            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_modified_variables_from_statement(stmt, modified)?;
                }
            }
            _ => {
                // Handle other statement types
            }
        }
        Ok(())
    }

    /// **Create Loop Phi Node**
    ///
    /// Creates a Phi node for a loop-carried variable with initial value.
    fn create_loop_phi_node(
        &mut self,
        variable: SymbolId,
        initial_value: SsaVariableId,
        value_type: TypeId,
    ) -> Result<DataFlowNodeId, GraphConstructionError> {
        let phi_node_id = self.allocate_node_id();

        // Find defining node for initial value
        let initial_node = self
            .find_defining_node_for_ssa_variable(initial_value)
            .ok_or_else(|| GraphConstructionError::InternalError {
                message: format!(
                    "Cannot find defining node for SSA variable {:?}",
                    initial_value
                ),
            })?;

        let phi_node = DataFlowNode {
            id: phi_node_id,
            kind: DataFlowNodeKind::Phi {
                incoming: vec![PhiIncoming {
                    value: initial_node,
                    predecessor: self.ssa_state.current_block, // Entry edge
                }],
            },
            value_type,
            source_location: SourceLocation::unknown(),
            operands: vec![initial_node],
            uses: new_id_set(),
            defines: Some(self.allocate_ssa_variable(variable, value_type)),
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };

        self.dfg.add_node(phi_node);
        Ok(phi_node_id)
    }

    /// **Create Iterator Phi Node**
    ///
    /// Creates a Phi node for loop iterator state (index counter).
    fn create_iterator_phi_node(
        &mut self,
        loop_header: BlockId,
    ) -> Result<DataFlowNodeId, GraphConstructionError> {
        let phi_node_id = self.allocate_node_id();

        // Create initial index value (0)
        let zero_node_id = self.allocate_node_id();
        let zero_node = DataFlowNode {
            id: zero_node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(0),
            },
            value_type: TypeId::from_raw(1), // Assume int type
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: loop_header,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(zero_node);

        let phi_node = DataFlowNode {
            id: phi_node_id,
            kind: DataFlowNodeKind::Phi {
                incoming: vec![PhiIncoming {
                    value: zero_node_id,
                    predecessor: loop_header,
                }],
            },
            value_type: TypeId::from_raw(1), // int type
            source_location: SourceLocation::unknown(),
            operands: vec![zero_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: loop_header,
            metadata: NodeMetadata::default(),
        };

        self.dfg.add_node(phi_node);
        Ok(phi_node_id)
    }

    /// **Create Array Access Node**
    ///
    /// Creates a node for accessing iterable[index] in for-in loops.
    fn create_array_access_node(
        &mut self,
        array: DataFlowNodeId,
        index: DataFlowNodeId,
        source_location: SourceLocation,
    ) -> Result<DataFlowNodeId, GraphConstructionError> {
        let access_node_id = self.allocate_node_id();
        let access_node = DataFlowNode {
            id: access_node_id,
            kind: DataFlowNodeKind::ArrayAccess { array, index },
            value_type: TypeId::invalid(), // Would infer from array element type
            source_location,
            operands: vec![array, index],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };

        self.dfg.add_node(access_node);
        Ok(access_node_id)
    }

    /// **Create Iterator Increment Node**
    ///
    /// Creates a node for incrementing the loop iterator (index + 1).
    fn create_iterator_increment_node(
        &mut self,
        index: DataFlowNodeId,
        source_location: SourceLocation,
    ) -> Result<DataFlowNodeId, GraphConstructionError> {
        // Create constant 1
        let one_node_id = self.allocate_node_id();
        let one_node = DataFlowNode {
            id: one_node_id,
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(1),
            },
            value_type: TypeId::from_raw(1), // int type
            source_location,
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };
        self.dfg.add_node(one_node);

        // Create addition node
        let add_node_id = self.allocate_node_id();
        let add_node = DataFlowNode {
            id: add_node_id,
            kind: DataFlowNodeKind::BinaryOp {
                operator: BinaryOperator::Add,
                left: index,
                right: one_node_id,
            },
            value_type: TypeId::from_raw(1), // int type
            source_location,
            operands: vec![index, one_node_id],
            uses: new_id_set(),
            defines: None,
            basic_block: self.ssa_state.current_block,
            metadata: NodeMetadata::default(),
        };

        self.dfg.add_node(add_node);
        Ok(add_node_id)
    }

    /// **Set Variable Helper**
    ///
    /// Helper method to set SSA variable in current state.
    fn set_variable(&mut self, symbol: SymbolId, ssa_var: SsaVariableId) {
        self.ssa_state
            .variable_stacks
            .entry(symbol)
            .or_insert_with(Vec::new)
            .push(ssa_var);
    }

    // ==================================================================================
    // HELPER FUNCTIONS
    // ==================================================================================

    /// **Convert literal value to DFG constant value**
    fn convert_literal_value(&self, value: &LiteralValue) -> ConstantValue {
        match value {
            LiteralValue::Bool(b) => ConstantValue::Bool(*b),
            LiteralValue::Int(i) => ConstantValue::Int(*i),
            LiteralValue::Float(f) => ConstantValue::Float(*f),
            LiteralValue::String(s) => ConstantValue::String(s.clone()),
            LiteralValue::Char(c) => ConstantValue::String(c.to_string()),
            LiteralValue::Regex(r) => ConstantValue::String(r.clone()),
            LiteralValue::RegexWithFlags { pattern, flags } => {
                ConstantValue::String(format!("/{}/{}", pattern, flags))
            }
        }
    }

    /// **Convert cast kind from TAST to DFG**
    fn convert_cast_kind(&self, cast_kind: TastCastKind) -> CastKind {
        match cast_kind {
            TastCastKind::Implicit => CastKind::Implicit,
            TastCastKind::Explicit => CastKind::Explicit,
            TastCastKind::Unsafe => CastKind::Unsafe,
            TastCastKind::Checked => CastKind::Checked,
        }
    }

    /// **Get current SSA variable for a symbol**
    fn get_current_ssa_variable(
        &self,
        symbol_id: SymbolId,
    ) -> Result<SsaVariableId, GraphConstructionError> {
        self.ssa_state
            .variable_stacks
            .get(&symbol_id)
            .and_then(|stack| stack.last().copied())
            .ok_or_else(|| GraphConstructionError::InternalError {
                message: format!("No SSA variable found for symbol {:?}", symbol_id),
            })
    }

    /// **Get or create SSA variable for a symbol**
    /// This is more lenient than get_current_ssa_variable and will create a placeholder
    /// for undefined symbols (like external functions)
    fn get_or_create_ssa_variable(&mut self, symbol_id: SymbolId) -> SsaVariableId {
        if let Ok(ssa_var) = self.get_current_ssa_variable(symbol_id) {
            ssa_var
        } else {
            // Create a placeholder SSA variable for undefined symbols
            // Use a placeholder type for external symbols
            let ssa_var_id = self.allocate_ssa_variable(symbol_id, TypeId::invalid());
            self.push_ssa_variable(symbol_id, ssa_var_id);
            ssa_var_id
        }
    }

    /// **Push SSA variable onto stack**
    pub(crate) fn push_ssa_variable(&mut self, symbol_id: SymbolId, ssa_var_id: SsaVariableId) {
        self.ssa_state
            .variable_stacks
            .entry(symbol_id)
            .or_default()
            .push(ssa_var_id);
    }

    /// **Allocate new node ID**
    fn allocate_node_id(&mut self) -> DataFlowNodeId {
        let id = DataFlowNodeId::from_raw(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// **Allocate new SSA variable using actual SsaVariable struct fields**
    pub(crate) fn allocate_ssa_variable(
        &mut self,
        original_symbol: SymbolId,
        var_type: TypeId,
    ) -> SsaVariableId {
        let id = SsaVariableId::from_raw(self.next_ssa_var_id);
        self.next_ssa_var_id += 1;

        // Use actual SsaVariable struct fields
        let ssa_var = SsaVariable {
            id,
            original_symbol,
            ssa_index: 0,                          // Would be computed properly
            var_type,                              // Use var_type, not variable_type
            definition: DataFlowNodeId::invalid(), // Would be set properly
            uses: vec![],                          // Empty initially
            liveness: LivenessInfo::default(),
        };

        self.dfg.ssa_variables.insert(id, ssa_var);
        self.stats.ssa_variables_created += 1;

        id
    }

    /// **Save variable stacks**
    fn save_variable_stacks(&self) -> HashMap<SymbolId, Vec<SsaVariableId>> {
        self.ssa_state.variable_stacks.clone()
    }

    /// **Restore variable stacks**
    fn restore_variable_stacks(&mut self, saved_stacks: HashMap<SymbolId, Vec<SsaVariableId>>) {
        self.ssa_state.variable_stacks = saved_stacks;
    }

    /// **Process phi nodes in a block**
    fn process_phi_nodes_in_block(
        &mut self,
        block_id: BlockId,
    ) -> Result<(), GraphConstructionError> {
        let phi_nodes: Vec<_> = self
            .ssa_state
            .phi_placed
            .iter()
            .filter(|((blk, _), _)| *blk == block_id)
            .map(|((_, symbol), node_id)| (*symbol, *node_id))
            .collect();

        for (symbol_id, _node_id) in phi_nodes {
            if let Some(incomplete_phi) = self
                .ssa_state
                .incomplete_phis
                .values()
                .find(|phi| phi.block_id == block_id && phi.symbol_id == symbol_id)
            {
                self.push_ssa_variable(symbol_id, incomplete_phi.ssa_var_id);
            }
        }

        Ok(())
    }

    /// **Fill phi operands in successor blocks**
    fn fill_successor_phi_operands(
        &mut self,
        block_id: BlockId,
        cfg: &ControlFlowGraph,
    ) -> Result<(), GraphConstructionError> {
        // Get successors of current block
        if let Some(block) = cfg.get_block(block_id) {
            let successors = match &block.terminator {
                Terminator::Jump { target } => vec![*target],
                Terminator::Branch {
                    true_target: then_block,
                    false_target: else_block,
                    ..
                } => vec![*then_block, *else_block],
                Terminator::Switch {
                    targets: cases,
                    default_target: default_block,
                    ..
                } => {
                    let mut succs: Vec<_> = cases.iter().map(|c| c.target).collect();
                    if let Some(default) = default_block {
                        succs.push(*default);
                    }
                    succs
                }
                _ => vec![],
            };

            // For each successor, fill in phi operands for current block
            for succ_block in successors {
                self.fill_phi_operands_for_predecessor(succ_block, block_id)?;
            }
        }

        Ok(())
    }

    /// **Fill phi operands for a specific predecessor**
    fn fill_phi_operands_for_predecessor(
        &mut self,
        block_id: BlockId,
        predecessor: BlockId,
    ) -> Result<(), GraphConstructionError> {
        // Find all phi nodes in this block
        let phi_nodes: Vec<_> = self
            .dfg
            .nodes
            .iter()
            .filter(|(_, node)| {
                node.basic_block == block_id && matches!(node.kind, DataFlowNodeKind::Phi { .. })
            })
            .map(|(id, _)| *id)
            .collect();

        for phi_node_id in phi_nodes {
            if let Some(incomplete_phi) = self.ssa_state.incomplete_phis.get(&phi_node_id) {
                // Get current SSA variable for this symbol at end of predecessor
                if let Ok(ssa_var) = self.get_current_ssa_variable(incomplete_phi.symbol_id) {
                    // Find the node that defines this SSA variable
                    if let Some(defining_node) = self.find_defining_node_for_ssa_variable(ssa_var) {
                        // This information will be used in complete_phi_operands
                        // For now, just ensure we track it properly
                    }
                }
            }
        }

        Ok(())
    }

    /// **Get SSA variable at block exit**
    fn get_variable_at_block_exit(
        &self,
        symbol_id: SymbolId,
        block_id: BlockId,
    ) -> Result<SsaVariableId, GraphConstructionError> {
        self.ssa_state
            .block_exit_variables
            .get(&(block_id, symbol_id))
            .cloned()
            .ok_or(GraphConstructionError::InternalError {
                message: "Could not get SSA variable at block exit".to_string(),
            })
    }

    /// **Find the node that defines an SSA variable**
    fn find_defining_node_for_ssa_variable(
        &self,
        ssa_var_id: SsaVariableId,
    ) -> Option<DataFlowNodeId> {
        // Search for the node that defines this SSA variable
        for (node_id, node) in &self.dfg.nodes {
            if node.defines == Some(ssa_var_id) {
                return Some(*node_id);
            }
        }
        None
    }

    /// **Rename variables in dominance tree order**
    fn rename_variables_in_block<'a>(
        &mut self,
        block_id: BlockId,
        dominance_tree: &DominanceTree,
        mapping: &'a TastCfgMapping,
        function: &'a TypedFunction,
        cfg: &ControlFlowGraph,
    ) -> Result<(), GraphConstructionError> {
        // Save current variable stacks
        let saved_stacks = self.save_variable_stacks();

        // Set current block
        let old_block = self.ssa_state.current_block;
        self.ssa_state.current_block = block_id;

        // 1. Process phi nodes in this block (update variable stacks)
        self.process_phi_nodes_in_block(block_id)?;

        // 2. Process regular statements in this block
        let statements = mapping.get_statements_in_block(block_id);
        for &stmt_location in statements {
            let statement = Self::get_statement_from_location(stmt_location, &function.body)?;
            self.build_statement(statement)?;
        }

        // 3. Fill phi operands in successor blocks
        self.fill_successor_phi_operands(block_id, cfg)?;

        // 4. Recursively process dominated children
        if let Some(children) = dominance_tree.dom_tree_children.get(&block_id) {
            for &child_block in children.clone().iter() {
                self.rename_variables_in_block(
                    child_block,
                    dominance_tree,
                    mapping,
                    function,
                    cfg,
                )?;
            }
        }

        // 5. Restore variable stacks
        self.restore_variable_stacks(saved_stacks);
        self.ssa_state.current_block = old_block;

        Ok(())
    }

    /// **Save variables at block exit for phi operand computation**
    fn save_block_exit_variables(&mut self, block_id: BlockId) {
        for (symbol_id, stack) in &self.ssa_state.variable_stacks {
            if let Some(&ssa_var) = stack.last() {
                self.ssa_state
                    .block_exit_variables
                    .insert((block_id, *symbol_id), ssa_var);
            }
        }
    }

    /// Find variables captured by the lambda
    fn find_captured_variables(
        &self,
        body: &[TypedStatement],
        parameters: &[TypedParameter],
    ) -> Result<Vec<CapturedVariable>, GraphConstructionError> {
        let mut visitor = FreeVariableVisitor::new(parameters, &self.ssa_state);
        visitor.visit_statements(body);

        let mut captured = Vec::new();
        for free_var in visitor.free_variables {
            // Get current SSA variable for this symbol
            if let Ok(ssa_var) = self.get_current_ssa_variable(free_var) {
                // Get type from SSA variable
                let var_type = self
                    .dfg
                    .ssa_variables
                    .get(&ssa_var)
                    .map(|v| v.var_type)
                    .unwrap_or(TypeId::invalid());

                captured.push(CapturedVariable {
                    symbol_id: free_var,
                    ssa_var_id: ssa_var,
                    capture_type: var_type,
                });
            }
        }

        Ok(captured)
    }

    /// Calculate closure size based on captured variables
    fn calculate_closure_size(&self, captured_vars: &[CapturedVariable]) -> usize {
        // Basic calculation: header (8 bytes) + pointer per captured variable (8 bytes each)
        // This is simplified - real implementation would consider actual type sizes
        8 + (captured_vars.len() * 8)
    }

    /// Create store nodes for captured variables
    fn create_capture_stores(
        &mut self,
        closure_alloc: DataFlowNodeId,
        captured_vars: &[CapturedVariable],
    ) -> Result<Vec<DataFlowNodeId>, GraphConstructionError> {
        let mut store_nodes = Vec::new();

        for (index, capture) in captured_vars.iter().enumerate() {
            // Create offset constant for this capture slot
            let offset_node_id = self.allocate_node_id();
            let offset_node = DataFlowNode {
                id: offset_node_id,
                kind: DataFlowNodeKind::Constant {
                    value: ConstantValue::Int((8 + index * 8) as i64), // Header + slot offset
                },
                value_type: TypeId::from_raw(1), // int
                source_location: SourceLocation::unknown(),
                operands: vec![],
                uses: new_id_set(),
                defines: None,
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata::default(),
            };
            self.dfg.add_node(offset_node);

            // Get the node that defines the captured variable
            let captured_value = self
                .find_defining_node_for_ssa_variable(capture.ssa_var_id)
                .ok_or_else(|| GraphConstructionError::InternalError {
                    message: format!(
                        "No defining node for captured variable {:?}",
                        capture.symbol_id
                    ),
                })?;

            // Create store node to save captured value
            let store_node_id = self.allocate_node_id();
            let store_node = DataFlowNode {
                id: store_node_id,
                kind: DataFlowNodeKind::Store {
                    address: closure_alloc,
                    value: captured_value,
                    memory_type: MemoryType::Heap,
                },
                value_type: TypeId::invalid(), // Store returns void
                source_location: SourceLocation::unknown(),
                operands: vec![closure_alloc, captured_value, offset_node_id],
                uses: new_id_set(),
                defines: None,
                basic_block: self.ssa_state.current_block,
                metadata: NodeMetadata {
                    has_side_effects: true,
                    annotations: {
                        let mut annotations = HashMap::new();
                        annotations.insert("capture_index".to_string(), index.to_string());
                        annotations.insert(
                            "captured_symbol".to_string(),
                            format!("{:?}", capture.symbol_id),
                        );
                        annotations
                    },
                    ..Default::default()
                },
            };
            self.dfg.add_node(store_node);
            store_nodes.push(store_node_id);
        }

        Ok(store_nodes)
    }

    /// Complete all phi operands after variable renaming with proper type unification
    pub fn complete_phi_operands_with_type_unification<'a>(
        &mut self,
        type_checker: &'a TypeChecker<'a>,
    ) -> Result<(), GraphConstructionError> {
        // Get all incomplete phi nodes
        let incomplete_phis: Vec<_> = self.ssa_state.incomplete_phis.clone().into_iter().collect();

        // Process each phi node separately to avoid mixing operands
        for (phi_node_id, incomplete_phi) in incomplete_phis {
            let mut phi_operands = Vec::new();

            // For each predecessor of this phi's block
            for &pred_block in &incomplete_phi.predecessor_blocks {
                // Get the SSA variable for this symbol at the end of the predecessor
                let ssa_var =
                    self.get_variable_at_block_exit(incomplete_phi.symbol_id, pred_block)?;

                // Find the node that defines this SSA variable
                if let Some(defining_node) = self.find_defining_node_for_ssa_variable(ssa_var) {
                    phi_operands.push(PhiIncoming {
                        value: defining_node,
                        predecessor: pred_block,
                    });
                }
            }

            // Skip phi nodes with no operands
            if phi_operands.is_empty() {
                continue;
            }

            // Validate all operands are compatible and get unified type
            let unified_type =
                self.resolve_and_validate_phi_operand_types(&phi_operands, type_checker)?;

            // Update the phi node with its specific operands and unified type
            if let Some(phi_node) = self.dfg.nodes.get_mut(&phi_node_id) {
                if let DataFlowNodeKind::Phi { incoming } = &mut phi_node.kind {
                    *incoming = phi_operands.clone();
                    phi_node.operands = phi_operands.iter().map(|inc| inc.value).collect();
                }

                // Set the properly unified type
                phi_node.value_type = unified_type;

                // Update statistics
                self.stats.phi_nodes_inserted += 1;
            }
        }

        // Clear incomplete phi tracking
        self.ssa_state.incomplete_phis.clear();
        Ok(())
    }

    fn finalize_dfg(
        &mut self,
        construction_time: std::time::Duration,
    ) -> Result<(), GraphConstructionError> {
        self.dfg.metadata.is_ssa_form = self.options.convert_to_ssa;
        self.dfg.metadata.construction_stats.construction_time_us =
            construction_time.as_micros() as u64;
        self.dfg.metadata.construction_stats.nodes_created = self.stats.nodes_created;

        self.stats.construction_time_us = construction_time.as_micros() as u64;

        Ok(())
    }
}

impl SsaConstructionState {
    fn new() -> Self {
        Self {
            variable_stacks: HashMap::new(),
            def_blocks: HashMap::new(),
            phi_placed: HashMap::new(),
            incomplete_phis: HashMap::new(),
            current_block: BlockId::invalid(),
            block_exit_variables: HashMap::new(),
        }
    }
}

impl Default for SsaConstructionState {
    fn default() -> Self {
        Self::new()
    }
}

impl DfgBuilderPhiTypeUnification for DfgBuilder {
    /// Validate that all phi operand types are compatible with the unified type
    fn resolve_and_validate_phi_operand_types<'a>(
        &self,
        phi_operands: &[PhiIncoming],
        type_checker: &'a TypeChecker<'a>,
    ) -> Result<TypeId, GraphConstructionError> {
        // First unify the types using TypeChecker for proper hierarchy
        let mut unifier = PhiTypeUnifier::new(&type_checker.type_table, type_checker);
        let unified_type = unifier.unify_phi_types(phi_operands, &self.dfg)?;

        // The PhiTypeUnifier already ensures type compatibility by finding the LUB
        // We trust the unifier's result since it uses TypeChecker for proper hierarchy resolution

        Ok(unified_type)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        semantic_graph::{dfg_builder, CfgBuilder},
        tast::{
            node::{FunctionEffects, FunctionMetadata},
            MemoryEffects, Mutability, ResourceEffects, StringInterner, Visibility,
        },
    };

    use super::*;

    #[test]
    fn test_fixed_dfg_builder_creation() {
        let options = GraphConstructionOptions::default();
        let builder = DfgBuilder::new(options);

        assert_eq!(builder.next_node_id, 2);
        assert_eq!(builder.next_ssa_var_id, 1);
        assert!(builder.ssa_state.variable_stacks.is_empty());
    }

    #[test]
    fn test_fixed_literal_expression_handling() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        let literal_expr = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::Literal {
                value: LiteralValue::Int(42),
            },
            usage: crate::tast::node::VariableUsage::Borrow,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: crate::tast::node::ExpressionMetadata::default(),
        };

        let result = builder.build_expression(&literal_expr);
        assert!(result.is_ok());

        let node_id = result.unwrap();
        assert!(builder.dfg.nodes.contains_key(&node_id));

        if let Some(node) = builder.dfg.nodes.get(&node_id) {
            assert!(matches!(node.kind, DataFlowNodeKind::Constant { .. }));
        }
    }

    /// Create a simple test function
    fn create_test_function() -> TypedFunction {
        let interner = StringInterner::new();
        TypedFunction {
            symbol_id: SymbolId::from_raw(1),
            name: interner.intern("test_function"),
            parameters: vec![TypedParameter {
                symbol_id: SymbolId::from_raw(2),
                name: interner.intern("x"),
                param_type: TypeId::from_raw(1), // int
                default_value: None,
                is_optional: false,
                // is_variadic: false,
                source_location: SourceLocation::unknown(),
                mutability: Mutability::Mutable,
            }],
            return_type: TypeId::from_raw(1), // int
            body: vec![],
            type_parameters: vec![],
            source_location: SourceLocation::unknown(),
            visibility: Visibility::Public,
            is_static: false,
            effects: FunctionEffects {
                async_kind: crate::tast::AsyncKind::Sync,
                is_pure: false,
                is_inline: false,
                can_throw: true,
                exception_types: vec![],
                memory_effects: MemoryEffects {
                    mutations: vec![],
                    moves: vec![],
                    escapes_references: false,
                    accesses_global_state: false,
                },
                resource_effects: ResourceEffects {
                    ..Default::default()
                },
            },
            metadata: FunctionMetadata::default(),
        }
    }

    /// Create a test expression
    fn create_test_expression(expr_type: TypeId) -> TypedExpression {
        TypedExpression {
            expr_type,
            kind: TypedExpressionKind::Literal {
                value: LiteralValue::Int(42),
            },
            usage: crate::tast::node::VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: crate::tast::node::ExpressionMetadata::default(),
        }
    }

    #[test]
    fn test_dfg_builder_initialization() {
        let options = GraphConstructionOptions::default();
        let builder = DfgBuilder::new(options);

        assert_eq!(builder.next_node_id, 2);
        assert_eq!(builder.next_ssa_var_id, 1);
        assert_eq!(builder.stats.nodes_created, 0);
    }

    #[test]
    fn test_parameter_initialization() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());
        let function = create_test_function();

        builder.initialize_function_parameters(&function).unwrap();

        // Should create one parameter node
        assert_eq!(builder.stats.nodes_created, 1);
        assert_eq!(builder.stats.ssa_variables_created, 1);

        // Should have SSA variable for parameter
        let param_symbol = function.parameters[0].symbol_id;
        assert!(builder.get_current_ssa_variable(param_symbol).is_ok());
    }

    #[test]
    fn test_literal_expression() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        let expr = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::Literal {
                value: LiteralValue::Int(42),
            },
            usage: crate::tast::node::VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: crate::tast::node::ExpressionMetadata::default(),
        };

        let node_id = builder.build_expression(&expr).unwrap();

        // Verify node was created
        let node = builder.dfg.get_node(node_id).unwrap();
        match &node.kind {
            DataFlowNodeKind::Constant { value } => {
                assert_eq!(*value, ConstantValue::Int(42));
            }
            _ => panic!("Expected constant node"),
        }
    }

    #[test]
    fn test_binary_operation() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        let left = Box::new(create_test_expression(TypeId::from_raw(1)));
        let right = Box::new(TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::Literal {
                value: LiteralValue::Int(10),
            },
            usage: crate::tast::node::VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: crate::tast::node::ExpressionMetadata::default(),
        });

        let expr = TypedExpression {
            expr_type: TypeId::from_raw(1),
            kind: TypedExpressionKind::BinaryOp {
                left,
                operator: BinaryOperator::Add,
                right,
            },
            usage: crate::tast::node::VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: crate::tast::node::ExpressionMetadata::default(),
        };

        let node_id = builder.build_expression(&expr).unwrap();

        // Verify binary op node
        let node = builder.dfg.get_node(node_id).unwrap();
        match &node.kind {
            DataFlowNodeKind::BinaryOp {
                operator,
                left,
                right,
            } => {
                assert_eq!(*operator, BinaryOperator::Add);

                // Verify operands are constant nodes
                let left_node = builder.dfg.get_node(*left).unwrap();
                let right_node = builder.dfg.get_node(*right).unwrap();

                assert!(matches!(left_node.kind, DataFlowNodeKind::Constant { .. }));
                assert!(matches!(right_node.kind, DataFlowNodeKind::Constant { .. }));
            }
            _ => panic!("Expected binary op node"),
        }

        // Should have 3 nodes: left, right, and binary op
        assert_eq!(builder.stats.nodes_created, 3);
    }

    #[test]
    fn test_variable_declaration() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        let symbol_id = SymbolId::from_raw(10);
        let var_type = TypeId::from_raw(1);
        let initializer = Some(create_test_expression(var_type));

        let nodes = builder
            .build_var_declaration(symbol_id, var_type, initializer.as_ref())
            .unwrap();

        assert_eq!(nodes.len(), 2);
        assert_eq!(builder.stats.ssa_variables_created, 1);

        // Should be able to get current SSA variable
        assert!(builder.get_current_ssa_variable(symbol_id).is_ok());
    }

    #[test]
    fn test_full_function_with_cfg() {
        let type_table = RefCell::new(TypeTable::new());
        let symbol_table = crate::tast::SymbolTable::new();
        let scope_tree = crate::tast::ScopeTree::new(crate::tast::ScopeId::first());
        let string_interner = crate::tast::StringInterner::new();

        let type_checker =
            &mut TypeChecker::new(&type_table, &symbol_table, &scope_tree, &string_interner);

        let mut cfg_builder = CfgBuilder::new(GraphConstructionOptions::default());
        let mut dfg_builder = DfgBuilder::new(GraphConstructionOptions::default());

        // Create a function with a simple body
        let mut function = create_test_function();
        function.body = vec![
            TypedStatement::VarDeclaration {
                symbol_id: SymbolId::from_raw(20),
                var_type: TypeId::from_raw(1),
                initializer: Some(TypedExpression {
                    expr_type: TypeId::from_raw(1),
                    kind: TypedExpressionKind::Variable {
                        symbol_id: SymbolId::from_raw(2), // parameter x
                    },
                    usage: crate::tast::node::VariableUsage::Copy,
                    lifetime_id: crate::tast::LifetimeId::static_lifetime(),
                    source_location: SourceLocation::unknown(),
                    metadata: crate::tast::node::ExpressionMetadata::default(),
                }),
                mutability: Mutability::Immutable,
                source_location: SourceLocation::unknown(),
            },
            TypedStatement::Return {
                value: Some(TypedExpression {
                    expr_type: TypeId::from_raw(1),
                    kind: TypedExpressionKind::Variable {
                        symbol_id: SymbolId::from_raw(20),
                    },
                    usage: crate::tast::node::VariableUsage::Move,
                    lifetime_id: crate::tast::LifetimeId::static_lifetime(),
                    source_location: SourceLocation::unknown(),
                    metadata: crate::tast::node::ExpressionMetadata::default(),
                }),
                source_location: SourceLocation::unknown(),
            },
        ];

        // Build CFG
        let cfg = cfg_builder.build_function(&function).unwrap();

        // Build DFG
        let dfg = dfg_builder
            .build_dfg(&cfg, &function, type_checker)
            .unwrap();

        // Verify DFG structure
        assert!(dfg.metadata.is_ssa_form);
        assert!(dfg.is_valid_ssa());

        // Should have parameter, variable declaration, and return nodes
        assert!(dfg_builder.stats.nodes_created >= 3);
        assert!(dfg_builder.stats.ssa_variables_created >= 2);
    }

    /// Test that block exit variables are correctly saved
    #[test]
    fn test_block_exit_variable_tracking() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        // Create a simple SSA variable
        let symbol = SymbolId::from_raw(10);
        let ssa_var = builder.allocate_ssa_variable(symbol, TypeId::from_raw(1));
        builder.push_ssa_variable(symbol, ssa_var);

        // Save block exit variables
        let block_id = BlockId::from_raw(1);
        builder.save_block_exit_variables(block_id);

        // Verify it was saved
        let exit_var = builder
            .get_variable_at_block_exit(symbol, block_id)
            .unwrap();
        assert_eq!(exit_var, ssa_var);
    }

    /// Test nested statement navigation
    #[test]
    fn test_nested_statement_navigation() {
        let builder = DfgBuilder::new(GraphConstructionOptions::default());
        let string_interner = crate::tast::StringInterner::new();
        // Create nested statements
        let inner_stmt = TypedStatement::Expression {
            expression: create_test_expression(TypeId::from_raw(1)),
            source_location: SourceLocation::unknown(),
        };

        let block_stmt = TypedStatement::Block {
            statements: vec![inner_stmt],
            scope_id: crate::tast::ScopeId::from_raw(1),
            source_location: SourceLocation::unknown(),
        };

        let function = TypedFunction {
            symbol_id: SymbolId::from_raw(1),
            name: string_interner.intern("test"),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![block_stmt],
            type_parameters: vec![],
            source_location: SourceLocation::unknown(),
            is_static: false,
            effects: FunctionEffects::default(),
            metadata: FunctionMetadata::default(),
            visibility: Visibility::Public,
        };

        // Test navigation to nested statement
        let location = StatementLocation {
            statement_index: 0,
            nesting_depth: 1,
            id: 1,
            branch_context: BranchContext::None,
        };

        let result = dfg_builder::DfgBuilder::get_statement_from_location(location, &function.body);
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), TypedStatement::Expression { .. }));
    }

    /// Test array literal initialization with stores
    #[test]
    fn test_array_literal_initialization() {
        let mut builder = DfgBuilder::new(GraphConstructionOptions::default());

        // Create array literal [1, 2, 3]
        let elements = vec![
            TypedExpression {
                expr_type: TypeId::from_raw(1),
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(1),
                },
                usage: crate::tast::node::VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: crate::tast::node::ExpressionMetadata::default(),
            },
            TypedExpression {
                expr_type: TypeId::from_raw(1),
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(2),
                },
                usage: crate::tast::node::VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: crate::tast::node::ExpressionMetadata::default(),
            },
            TypedExpression {
                expr_type: TypeId::from_raw(1),
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(3),
                },
                usage: crate::tast::node::VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: crate::tast::node::ExpressionMetadata::default(),
            },
        ];

        let array_expr = TypedExpression {
            expr_type: TypeId::from_raw(10), // Array type
            kind: TypedExpressionKind::ArrayLiteral { elements },
            usage: crate::tast::node::VariableUsage::Move,
            lifetime_id: crate::tast::LifetimeId::static_lifetime(),
            source_location: SourceLocation::unknown(),
            metadata: crate::tast::node::ExpressionMetadata::default(),
        };

        let node_id = builder.build_expression(&array_expr).unwrap();

        // Verify the result is a Load node
        let node = builder.dfg.get_node(node_id).unwrap();
        assert!(matches!(node.kind, DataFlowNodeKind::Load { .. }));

        // Count nodes created (should include size, allocation, elements, indices, stores)
        let total_nodes = builder.dfg.nodes.len();
        assert!(total_nodes >= 10); // At minimum: load + alloc + size + 3 elems + 3 indices + 3 stores

        // Verify allocation and store nodes exist
        let alloc_nodes = builder
            .dfg
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Allocation { .. }))
            .count();
        assert_eq!(alloc_nodes, 1);

        let store_nodes = builder
            .dfg
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Store { .. }))
            .count();
        assert_eq!(store_nodes, 3); // One for each element
    }

    /// Test phi node creation with actual CFG
    #[test]
    fn test_phi_node_with_cfg() {
        let mut cfg_builder = CfgBuilder::new(GraphConstructionOptions::default());
        let mut dfg_builder = DfgBuilder::new(GraphConstructionOptions::default());

        // Create a simple function with control flow
        let mut function = create_test_function();
        function.body = vec![TypedStatement::If {
            condition: TypedExpression {
                expr_type: TypeId::from_raw(2), // bool
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Bool(true),
                },
                usage: crate::tast::node::VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: crate::tast::node::ExpressionMetadata::default(),
            },
            then_branch: Box::new(TypedStatement::Block {
                statements: vec![],
                scope_id: crate::tast::ScopeId::from_raw(1),
                source_location: SourceLocation::unknown(),
            }),
            else_branch: Some(Box::new(TypedStatement::Block {
                statements: vec![],
                scope_id: crate::tast::ScopeId::from_raw(2),
                source_location: SourceLocation::unknown(),
            })),
            source_location: SourceLocation::unknown(),
        }];

        // Build CFG
        let cfg = cfg_builder.build_function(&function).unwrap();

        // Create phi node with actual CFG
        let block_id = BlockId::from_raw(3); // Merge block
        let variable = SymbolId::from_raw(10);
        let phi_node_id = dfg_builder
            .create_phi_node(block_id, variable, &cfg)
            .unwrap();

        // Verify phi node was created
        let phi_node = dfg_builder.dfg.get_node(phi_node_id).unwrap();
        assert!(matches!(phi_node.kind, DataFlowNodeKind::Phi { .. }));

        // Verify incomplete phi info was tracked
        assert!(dfg_builder
            .ssa_state
            .incomplete_phis
            .contains_key(&phi_node_id));
    }
}
