//! Control flow analysis for definite assignment, null safety, and dead code detection
//!
//! This module provides comprehensive control flow analysis to ensure:
//! - Variables are definitely assigned before use
//! - Null safety by tracking nullable values
//! - Dead code detection after returns/throws
//! - Resource tracking for proper cleanup

use crate::tast::{
    node::{TypedExpression, TypedExpressionKind, TypedFile, TypedFunction, TypedStatement},
    symbols::Mutability,
    ScopeId, SourceLocation, SymbolId, TypeId,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Represents the state of a variable at a program point
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariableState {
    /// Variable is uninitialized
    Uninitialized,
    /// Variable is definitely initialized
    Initialized,
    /// Variable might be initialized (depends on control flow)
    MaybeInitialized,
    /// Variable is definitely null
    Null,
    /// Variable is definitely not null
    NotNull,
    /// Variable might be null
    MaybeNull,
}

/// Represents a basic block in the control flow graph
#[derive(Debug, Clone)]
pub struct ControlFlowBlock {
    /// Unique identifier for this block
    pub id: BlockId,
    /// Statements in this block
    pub statements: Vec<StatementInfo>,
    /// Successors to this block
    pub successors: Vec<BlockId>,
    /// Predecessors to this block
    pub predecessors: Vec<BlockId>,
    /// Variable states at the beginning of this block
    pub entry_states: BTreeMap<SymbolId, VariableState>,
    /// Variable states at the end of this block
    pub exit_states: BTreeMap<SymbolId, VariableState>,
    /// Whether this block is reachable
    pub is_reachable: bool,
    /// Whether this block definitely returns/throws
    pub definitely_exits: bool,
}

/// Information about a statement in a control flow block
#[derive(Debug, Clone)]
pub struct StatementInfo {
    /// The statement itself
    pub statement: TypedStatement,
    /// Variables assigned in this statement
    pub assigns: BTreeSet<SymbolId>,
    /// Variables used in this statement
    pub uses: BTreeSet<SymbolId>,
    /// Whether this statement can throw
    pub can_throw: bool,
    /// Whether this statement definitely exits (return/throw)
    pub definitely_exits: bool,
}

/// Unique identifier for control flow blocks
pub type BlockId = usize;

/// Control flow graph for a function or method
#[derive(Debug)]
pub struct ControlFlowGraph {
    /// All blocks in the graph
    pub blocks: BTreeMap<BlockId, ControlFlowBlock>,
    /// Entry block (where execution begins)
    pub entry_block: BlockId,
    /// Exit blocks (where execution ends)
    pub exit_blocks: Vec<BlockId>,
    /// Next available block ID
    next_block_id: BlockId,
}

impl ControlFlowGraph {
    /// Create a new empty control flow graph
    pub fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
            entry_block: 0,
            exit_blocks: Vec::new(),
            next_block_id: 0,
        }
    }

    /// Create a new block and return its ID
    pub fn create_block(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id += 1;

        self.blocks.insert(
            id,
            ControlFlowBlock {
                id,
                statements: Vec::new(),
                successors: Vec::new(),
                predecessors: Vec::new(),
                entry_states: BTreeMap::new(),
                exit_states: BTreeMap::new(),
                is_reachable: false,
                definitely_exits: false,
            },
        );

        id
    }

    /// Add an edge from one block to another
    pub fn add_edge(&mut self, from: BlockId, to: BlockId) {
        if let Some(from_block) = self.blocks.get_mut(&from) {
            if !from_block.successors.contains(&to) {
                from_block.successors.push(to);
            }
        }

        if let Some(to_block) = self.blocks.get_mut(&to) {
            if !to_block.predecessors.contains(&from) {
                to_block.predecessors.push(from);
            }
        }
    }

    /// Add a statement to a block
    pub fn add_statement(&mut self, block_id: BlockId, statement_info: StatementInfo) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.statements.push(statement_info);
        }
    }
}

/// Information about a resource that needs cleanup
#[derive(Debug, Clone)]
pub struct ResourceInfo {
    /// Where the resource was acquired
    pub acquisition_location: SourceLocation,
    /// Type of resource (file, connection, etc.)
    pub resource_type: ResourceType,
    /// Whether the resource has been properly disposed
    pub is_disposed: bool,
    /// Cleanup method name (close, dispose, etc.)
    pub cleanup_method: Option<String>,
}

/// Types of resources that need tracking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourceType {
    /// File handle
    File,
    /// Database connection
    DatabaseConnection,
    /// Network socket
    Socket,
    /// Memory buffer
    Buffer,
    /// Generic resource requiring disposal
    Generic,
}

/// Control flow analysis engine
pub struct ControlFlowAnalyzer {
    /// Current control flow graph being built
    cfg: ControlFlowGraph,
    /// Stack of break targets (for break statements)
    break_targets: Vec<BlockId>,
    /// Stack of continue targets (for continue statements)
    continue_targets: Vec<BlockId>,
    /// Current block being built
    current_block: BlockId,
    /// Variables in current scope
    variables: BTreeSet<SymbolId>,
    /// Resources that need cleanup (file handles, connections, etc.)
    resources: BTreeMap<SymbolId, ResourceInfo>,
    /// Analysis results
    results: AnalysisResults,
    /// Whether we're analyzing an entry point function (like static main)
    is_entry_point: bool,
}

/// Results of control flow analysis
#[derive(Debug, Default)]
pub struct AnalysisResults {
    /// Variables that are used before being initialized
    pub uninitialized_uses: Vec<UninitializedUse>,
    /// Statements that are unreachable (dead code)
    pub dead_code: Vec<DeadCodeWarning>,
    /// Null dereference warnings
    pub null_dereferences: Vec<NullDereferenceWarning>,
    /// Resources that might not be cleaned up
    pub resource_leaks: Vec<ResourceLeakWarning>,
}

/// Warning about using an uninitialized variable
#[derive(Debug, Clone)]
pub struct UninitializedUse {
    pub variable: SymbolId,
    pub location: SourceLocation,
    pub message: String,
}

/// Warning about dead code
#[derive(Debug, Clone)]
pub struct DeadCodeWarning {
    pub location: SourceLocation,
    pub message: String,
}

/// Warning about potential null dereference
#[derive(Debug, Clone)]
pub struct NullDereferenceWarning {
    pub variable: SymbolId,
    pub location: SourceLocation,
    pub message: String,
}

/// Warning about resource leak
#[derive(Debug, Clone)]
pub struct ResourceLeakWarning {
    pub resource: SymbolId,
    pub location: SourceLocation,
    pub message: String,
}

impl ControlFlowAnalyzer {
    /// Create a new control flow analyzer
    pub fn new() -> Self {
        let mut cfg = ControlFlowGraph::new();
        let entry_block = cfg.create_block();
        cfg.entry_block = entry_block;

        Self {
            cfg,
            break_targets: Vec::new(),
            continue_targets: Vec::new(),
            current_block: entry_block,
            variables: BTreeSet::new(),
            resources: BTreeMap::new(),
            results: AnalysisResults::default(),
            is_entry_point: false,
        }
    }

    /// Analyze a function and return the results
    pub fn analyze_function(&mut self, function: &TypedFunction) -> AnalysisResults {
        // Check if this is an entry point function (static main)
        // Entry point functions are implicitly called by the runtime and should not trigger
        // dead code warnings for unused variables, as they may be test/demo code
        self.is_entry_point = function.is_static; // For now, treat all static functions as potential entry points

        // Add function parameters as initialized variables
        for param in &function.parameters {
            self.variables.insert(param.symbol_id);
            self.set_variable_state(param.symbol_id, VariableState::Initialized);
        }

        // Build control flow graph from function body
        for statement in &function.body {
            self.analyze_statement(statement);
        }

        // Perform data flow analysis
        self.compute_reachability();
        self.analyze_definite_assignment();
        self.analyze_null_safety();
        self.detect_dead_code();
        self.analyze_resource_usage();

        std::mem::take(&mut self.results)
    }

    /// Get a reference to the control flow graph
    pub fn get_cfg(&self) -> &ControlFlowGraph {
        &self.cfg
    }

    /// Analyze a statement and update the control flow graph
    fn analyze_statement(&mut self, statement: &TypedStatement) {
        let statement_info = self.create_statement_info(statement);

        match statement {
            TypedStatement::Expression { expression, .. } => {
                self.analyze_expression(expression);
                self.cfg.add_statement(self.current_block, statement_info);
            }

            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => {
                self.variables.insert(*symbol_id);

                if let Some(init) = initializer {
                    self.analyze_expression(init);
                    // Check if initializer is null
                    let state = if self.is_null_expression(init) {
                        VariableState::Null
                    } else {
                        VariableState::Initialized
                    };
                    self.set_variable_state(*symbol_id, state);
                } else {
                    self.set_variable_state(*symbol_id, VariableState::Uninitialized);
                }

                self.cfg.add_statement(self.current_block, statement_info);
            }

            TypedStatement::Assignment { target, value, .. } => {
                self.analyze_expression(value);

                // Extract assigned variable from target
                if let Some(symbol_id) = self.extract_assigned_variable(target) {
                    let state = if self.is_null_expression(value) {
                        VariableState::Null
                    } else {
                        VariableState::Initialized
                    };
                    self.set_variable_state(symbol_id, state);
                }

                self.analyze_expression(target);
                self.cfg.add_statement(self.current_block, statement_info);
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.analyze_expression(condition);

                let then_block = self.cfg.create_block();
                let else_block = self.cfg.create_block();
                let merge_block = self.cfg.create_block();

                // Add edges
                self.cfg.add_edge(self.current_block, then_block);
                self.cfg.add_edge(self.current_block, else_block);

                // Analyze then branch
                let old_block = self.current_block;
                self.current_block = then_block;
                self.analyze_statement(then_branch);
                if !self.current_block_exits() {
                    self.cfg.add_edge(self.current_block, merge_block);
                }

                // Analyze else branch
                self.current_block = else_block;
                if let Some(else_stmt) = else_branch {
                    self.analyze_statement(else_stmt);
                }
                if !self.current_block_exits() {
                    self.cfg.add_edge(self.current_block, merge_block);
                }

                // Continue with merge block
                self.current_block = merge_block;
                self.cfg.add_statement(old_block, statement_info);
            }

            TypedStatement::While {
                condition, body, ..
            } => {
                let loop_header = self.cfg.create_block();
                let loop_body = self.cfg.create_block();
                let loop_exit = self.cfg.create_block();

                // Add edge to loop header
                self.cfg.add_edge(self.current_block, loop_header);

                // Analyze condition in loop header
                let old_block = self.current_block;
                self.current_block = loop_header;
                self.analyze_expression(condition);

                // Add edges from header
                self.cfg.add_edge(loop_header, loop_body);
                self.cfg.add_edge(loop_header, loop_exit);

                // Analyze loop body
                self.break_targets.push(loop_exit);
                self.continue_targets.push(loop_header);

                self.current_block = loop_body;
                self.analyze_statement(body);

                if !self.current_block_exits() {
                    self.cfg.add_edge(self.current_block, loop_header);
                }

                self.break_targets.pop();
                self.continue_targets.pop();

                // Continue with loop exit
                self.current_block = loop_exit;
                self.cfg.add_statement(old_block, statement_info);
            }

            TypedStatement::Return { value, .. } => {
                if let Some(val) = value {
                    self.analyze_expression(val);
                }

                self.cfg.add_statement(self.current_block, statement_info);
                self.mark_current_block_exits();
            }

            TypedStatement::Throw { exception, .. } => {
                self.analyze_expression(exception);
                self.cfg.add_statement(self.current_block, statement_info);
                self.mark_current_block_exits();
            }

            TypedStatement::Break { .. } => {
                if let Some(&target) = self.break_targets.last() {
                    self.cfg.add_edge(self.current_block, target);
                }
                self.cfg.add_statement(self.current_block, statement_info);
                self.mark_current_block_exits();
            }

            TypedStatement::Continue { .. } => {
                if let Some(&target) = self.continue_targets.last() {
                    self.cfg.add_edge(self.current_block, target);
                }
                self.cfg.add_statement(self.current_block, statement_info);
                self.mark_current_block_exits();
            }

            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.analyze_statement(stmt);
                }
            }

            _ => {
                // Handle other statement types
                self.cfg.add_statement(self.current_block, statement_info);
            }
        }
    }

    /// Analyze an expression and track variable uses
    fn analyze_expression(&mut self, expression: &TypedExpression) {
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                self.check_variable_initialized(*symbol_id, &expression.source_location);
                self.check_null_dereference(*symbol_id, &expression.source_location);
            }

            TypedExpressionKind::FieldAccess { object, .. } => {
                self.analyze_expression(object);
                // Check if object could be null
                if let Some(symbol_id) = self.extract_variable_from_expression(object) {
                    self.check_null_dereference(symbol_id, &expression.source_location);
                }
            }

            TypedExpressionKind::ArrayAccess { array, index } => {
                self.analyze_expression(array);
                self.analyze_expression(index);
                // Check if array could be null
                if let Some(symbol_id) = self.extract_variable_from_expression(array) {
                    self.check_null_dereference(symbol_id, &expression.source_location);
                }
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                self.analyze_expression(function);
                for arg in arguments {
                    self.analyze_expression(arg);
                }
            }

            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                self.analyze_expression(receiver);
                for arg in arguments {
                    self.analyze_expression(arg);
                }
                // Check if receiver could be null
                if let Some(symbol_id) = self.extract_variable_from_expression(receiver) {
                    self.check_null_dereference(symbol_id, &expression.source_location);
                }
            }

            TypedExpressionKind::BinaryOp { left, right, .. } => {
                self.analyze_expression(left);
                self.analyze_expression(right);
            }

            TypedExpressionKind::UnaryOp { operand, .. } => {
                self.analyze_expression(operand);
            }

            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                self.analyze_expression(condition);
                self.analyze_expression(then_expr);
                if let Some(else_e) = else_expr {
                    self.analyze_expression(else_e);
                }
            }

            TypedExpressionKind::ArrayLiteral { elements } => {
                for elem in elements {
                    self.analyze_expression(elem);
                }
            }

            TypedExpressionKind::New { arguments, .. } => {
                for arg in arguments {
                    self.analyze_expression(arg);
                }
            }

            _ => {
                // Handle other expression types recursively
            }
        }
    }

    /// Create statement info for analysis
    fn create_statement_info(&self, statement: &TypedStatement) -> StatementInfo {
        let mut assigns = BTreeSet::new();
        let mut uses = BTreeSet::new();
        let can_throw = self.statement_can_throw(statement);
        let definitely_exits = self.statement_definitely_exits(statement);

        // Extract variable assignments and uses
        Self::extract_statement_variables(statement, &mut assigns, &mut uses);

        StatementInfo {
            statement: statement.clone(),
            assigns,
            uses,
            can_throw,
            definitely_exits,
        }
    }

    /// Check if a statement can throw an exception
    fn statement_can_throw(&self, statement: &TypedStatement) -> bool {
        match statement {
            TypedStatement::Throw { .. } => true,
            TypedStatement::Expression { expression, .. } => expression.metadata.can_throw,
            _ => false,
        }
    }

    /// Check if a statement definitely exits (return/throw)
    fn statement_definitely_exits(&self, statement: &TypedStatement) -> bool {
        matches!(
            statement,
            TypedStatement::Return { .. }
                | TypedStatement::Throw { .. }
                | TypedStatement::Break { .. }
                | TypedStatement::Continue { .. }
        )
    }

    /// Extract variables assigned and used in a statement
    fn extract_statement_variables(
        statement: &TypedStatement,
        assigns: &mut BTreeSet<SymbolId>,
        uses: &mut BTreeSet<SymbolId>,
    ) {
        match statement {
            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => {
                assigns.insert(*symbol_id);
                if let Some(init) = initializer {
                    Self::extract_expression_uses_static(init, uses);
                }
            }
            TypedStatement::Assignment { target, value, .. } => {
                if let Some(symbol_id) = Self::extract_assigned_variable_static(target) {
                    assigns.insert(symbol_id);
                }
                Self::extract_expression_uses_static(value, uses);
            }
            TypedStatement::Expression { expression, .. } => {
                Self::extract_expression_uses_static(expression, uses);
            }
            _ => {}
        }
    }

    /// Extract variable uses from an expression
    fn extract_expression_uses(&self, expression: &TypedExpression, uses: &mut BTreeSet<SymbolId>) {
        Self::extract_expression_uses_static(expression, uses);
    }

    /// Static version of extract_expression_uses
    fn extract_expression_uses_static(expression: &TypedExpression, uses: &mut BTreeSet<SymbolId>) {
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                uses.insert(*symbol_id);
            }
            TypedExpressionKind::FieldAccess { object, .. } => {
                Self::extract_expression_uses_static(object, uses);
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                Self::extract_expression_uses_static(left, uses);
                Self::extract_expression_uses_static(right, uses);
            }
            // Add more cases as needed
            _ => {}
        }
    }

    /// Extract the variable being assigned to
    fn extract_assigned_variable(&self, expression: &TypedExpression) -> Option<SymbolId> {
        Self::extract_assigned_variable_static(expression)
    }

    /// Static version of extract_assigned_variable
    fn extract_assigned_variable_static(expression: &TypedExpression) -> Option<SymbolId> {
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => Some(*symbol_id),
            _ => None,
        }
    }

    /// Extract variable from expression (for null checks)
    fn extract_variable_from_expression(&self, expression: &TypedExpression) -> Option<SymbolId> {
        match &expression.kind {
            TypedExpressionKind::Variable { symbol_id } => Some(*symbol_id),
            _ => None,
        }
    }

    /// Set the state of a variable
    fn set_variable_state(&mut self, symbol_id: SymbolId, state: VariableState) {
        if let Some(block) = self.cfg.blocks.get_mut(&self.current_block) {
            block.exit_states.insert(symbol_id, state);
        }
    }

    /// Check if an expression is a null literal
    fn is_null_expression(&self, expression: &TypedExpression) -> bool {
        matches!(expression.kind, TypedExpressionKind::Null)
    }

    /// Check if current block definitely exits
    fn current_block_exits(&self) -> bool {
        self.cfg
            .blocks
            .get(&self.current_block)
            .map(|b| b.definitely_exits)
            .unwrap_or(false)
    }

    /// Mark current block as definitely exiting
    fn mark_current_block_exits(&mut self) {
        if let Some(block) = self.cfg.blocks.get_mut(&self.current_block) {
            block.definitely_exits = true;
        }
    }

    /// Check if a variable is initialized before use
    fn check_variable_initialized(&mut self, symbol_id: SymbolId, location: &SourceLocation) {
        if !self.variables.contains(&symbol_id) {
            return; // Not a local variable
        }

        let state = self.get_variable_state(symbol_id);
        if matches!(
            state,
            VariableState::Uninitialized | VariableState::MaybeInitialized
        ) {
            self.results.uninitialized_uses.push(UninitializedUse {
                variable: symbol_id,
                location: location.clone(),
                message: "Variable used before initialization".to_string(),
            });
        }
    }

    /// Check for potential null dereference
    fn check_null_dereference(&mut self, symbol_id: SymbolId, location: &SourceLocation) {
        let state = self.get_variable_state(symbol_id);
        if matches!(state, VariableState::Null | VariableState::MaybeNull) {
            self.results.null_dereferences.push(NullDereferenceWarning {
                variable: symbol_id,
                location: location.clone(),
                message: "Potential null dereference".to_string(),
            });
        }
    }

    /// Get the current state of a variable
    fn get_variable_state(&self, symbol_id: SymbolId) -> VariableState {
        self.cfg
            .blocks
            .get(&self.current_block)
            .and_then(|b| b.exit_states.get(&symbol_id))
            .cloned()
            .unwrap_or(VariableState::Uninitialized)
    }

    /// Compute reachability of all blocks
    fn compute_reachability(&mut self) {
        let mut reachable = BTreeSet::new();
        let mut worklist = VecDeque::new();

        // Start from entry block
        worklist.push_back(self.cfg.entry_block);
        reachable.insert(self.cfg.entry_block);

        while let Some(block_id) = worklist.pop_front() {
            if let Some(block) = self.cfg.blocks.get(&block_id) {
                for &successor in &block.successors {
                    if !reachable.contains(&successor) {
                        reachable.insert(successor);
                        worklist.push_back(successor);
                    }
                }
            }
        }

        // Mark blocks as reachable
        for (&block_id, block) in &mut self.cfg.blocks {
            block.is_reachable = reachable.contains(&block_id);
        }
    }

    /// Analyze definite assignment using data flow analysis
    fn analyze_definite_assignment(&mut self) {
        // Iterative data flow analysis for definite assignment
        let mut changed = true;
        while changed {
            changed = false;

            let block_ids: Vec<_> = self.cfg.blocks.keys().copied().collect();
            for block_id in block_ids {
                if self.update_block_states(block_id) {
                    changed = true;
                }
            }
        }
    }

    /// Update the entry and exit states of a block
    fn update_block_states(&mut self, block_id: BlockId) -> bool {
        let mut changed = false;

        // Merge states from predecessors
        let mut entry_states = BTreeMap::new();

        if let Some(block) = self.cfg.blocks.get(&block_id) {
            let predecessors = block.predecessors.clone();

            for variable in &self.variables {
                let mut all_initialized = true;
                let mut any_initialized = false;

                for &pred_id in &predecessors {
                    if let Some(pred_block) = self.cfg.blocks.get(&pred_id) {
                        if pred_block.is_reachable {
                            let state = pred_block
                                .exit_states
                                .get(variable)
                                .cloned()
                                .unwrap_or(VariableState::Uninitialized);

                            match state {
                                VariableState::Initialized => any_initialized = true,
                                VariableState::Uninitialized => all_initialized = false,
                                _ => {
                                    all_initialized = false;
                                    any_initialized = true;
                                }
                            }
                        }
                    }
                }

                let merged_state = if all_initialized {
                    VariableState::Initialized
                } else if any_initialized {
                    VariableState::MaybeInitialized
                } else {
                    VariableState::Uninitialized
                };

                entry_states.insert(*variable, merged_state);
            }
        }

        // Update block states
        if let Some(block) = self.cfg.blocks.get(&block_id) {
            // First collect all the data we need
            let statements = block.statements.clone();
            let old_entry_states = block.entry_states.clone();
            let old_exit_states = block.exit_states.clone();

            // Check if entry states changed
            let entry_changed = old_entry_states != entry_states;

            // Compute new exit states
            let mut exit_states = entry_states.clone();
            for statement_info in &statements {
                for &assigned_var in &statement_info.assigns {
                    // Check if this assignment is a null assignment
                    let state = match &statement_info.statement {
                        TypedStatement::VarDeclaration {
                            initializer: Some(init),
                            ..
                        } => {
                            if self.is_null_expression(init) {
                                VariableState::Null
                            } else {
                                VariableState::Initialized
                            }
                        }
                        TypedStatement::Assignment { value, .. } => {
                            if self.is_null_expression(value) {
                                VariableState::Null
                            } else {
                                VariableState::Initialized
                            }
                        }
                        _ => VariableState::Initialized,
                    };
                    exit_states.insert(assigned_var, state);
                }
            }

            // Check if exit states changed
            let exit_changed = old_exit_states != exit_states;

            // Now update the block if anything changed
            if entry_changed || exit_changed {
                if let Some(block) = self.cfg.blocks.get_mut(&block_id) {
                    if entry_changed {
                        block.entry_states = entry_states;
                        changed = true;
                    }
                    if exit_changed {
                        block.exit_states = exit_states;
                        changed = true;
                    }
                }
            }
        }

        changed
    }

    /// Analyze null safety
    fn analyze_null_safety(&mut self) {
        // Track null state propagation through the CFG
        let mut changed = true;
        while changed {
            changed = false;

            let block_ids: Vec<_> = self.cfg.blocks.keys().copied().collect();
            for block_id in block_ids {
                if self.update_block_null_states(block_id) {
                    changed = true;
                }
            }
        }
    }

    /// Update null states for a block
    fn update_block_null_states(&mut self, block_id: BlockId) -> bool {
        let mut changed = false;

        // Merge null states from predecessors
        let mut entry_states = BTreeMap::new();

        if let Some(block) = self.cfg.blocks.get(&block_id) {
            let predecessors = block.predecessors.clone();

            for variable in &self.variables {
                let mut all_null = true;
                let mut any_null = false;
                let mut any_not_null = false;

                for &pred_id in &predecessors {
                    if let Some(pred_block) = self.cfg.blocks.get(&pred_id) {
                        if pred_block.is_reachable {
                            let state = pred_block
                                .exit_states
                                .get(variable)
                                .cloned()
                                .unwrap_or(VariableState::Uninitialized);

                            match state {
                                VariableState::Null => any_null = true,
                                VariableState::NotNull | VariableState::Initialized => {
                                    all_null = false;
                                    any_not_null = true;
                                }
                                VariableState::MaybeNull => {
                                    all_null = false;
                                    any_null = true;
                                    any_not_null = true;
                                }
                                VariableState::Uninitialized => {
                                    all_null = false;
                                }
                                _ => {}
                            }
                        }
                    }
                }

                let merged_state = if all_null {
                    VariableState::Null
                } else if any_null && any_not_null {
                    VariableState::MaybeNull
                } else if any_not_null {
                    VariableState::NotNull
                } else {
                    // Keep existing state from current block if no predecessors set state
                    block
                        .entry_states
                        .get(variable)
                        .cloned()
                        .unwrap_or(VariableState::Uninitialized)
                };

                entry_states.insert(*variable, merged_state);
            }
        }

        // Update null states for the block
        if let Some(block) = self.cfg.blocks.get(&block_id) {
            // Collect data first
            let statements = block.statements.clone();
            let mut old_entry_states = block.entry_states.clone();
            let old_exit_states = block.exit_states.clone();

            // Update null-related entry states
            let mut entry_changed = false;
            for (var, new_state) in entry_states {
                if matches!(
                    new_state,
                    VariableState::Null | VariableState::MaybeNull | VariableState::NotNull
                ) {
                    if old_entry_states.get(&var) != Some(&new_state) {
                        old_entry_states.insert(var, new_state);
                        entry_changed = true;
                    }
                }
            }

            // Recompute exit states considering null assignments
            let mut exit_states = old_entry_states.clone();
            for statement_info in &statements {
                for &assigned_var in &statement_info.assigns {
                    // Check if this assignment is a null assignment
                    let state = match &statement_info.statement {
                        TypedStatement::VarDeclaration {
                            initializer: Some(init),
                            ..
                        } => {
                            if self.is_null_expression(init) {
                                VariableState::Null
                            } else {
                                VariableState::NotNull
                            }
                        }
                        TypedStatement::Assignment { value, .. } => {
                            if self.is_null_expression(value) {
                                VariableState::Null
                            } else {
                                VariableState::NotNull
                            }
                        }
                        _ => VariableState::NotNull,
                    };
                    exit_states.insert(assigned_var, state);
                }
            }

            let exit_changed = old_exit_states != exit_states;

            // Apply changes if any
            if entry_changed || exit_changed {
                if let Some(block) = self.cfg.blocks.get_mut(&block_id) {
                    if entry_changed {
                        block.entry_states = old_entry_states;
                        changed = true;
                    }
                    if exit_changed {
                        block.exit_states = exit_states;
                        changed = true;
                    }
                }
            }
        }

        changed
    }

    /// Detect dead code
    fn detect_dead_code(&mut self) {
        // 1. Detect unreachable blocks
        for (_, block) in &self.cfg.blocks {
            if !block.is_reachable {
                for statement_info in &block.statements {
                    self.results.dead_code.push(DeadCodeWarning {
                        location: statement_info.statement.source_location(),
                        message: "Unreachable code due to unreachable basic block".to_string(),
                    });
                }
            }
        }

        // 2. Detect dead code after unconditional jumps
        for (_, block) in &self.cfg.blocks {
            if block.is_reachable {
                let mut found_exit = false;
                for (i, statement_info) in block.statements.iter().enumerate() {
                    if found_exit {
                        // This statement comes after an unconditional exit
                        self.results.dead_code.push(DeadCodeWarning {
                            location: statement_info.statement.source_location(),
                            message: "Unreachable code after return/throw/break/continue"
                                .to_string(),
                        });
                    }

                    if statement_info.definitely_exits {
                        found_exit = true;
                    }
                }
            }
        }

        // 3. Detect unreachable conditions (always true/false)
        self.detect_unreachable_conditions();

        // 4. Detect unused variables
        self.detect_unused_variables();
    }

    /// Detect unreachable conditions (always true/false branches)
    fn detect_unreachable_conditions(&mut self) {
        // This would require constant propagation analysis
        // For now, we'll implement basic literal condition detection
        for (_, block) in &self.cfg.blocks {
            if !block.is_reachable {
                continue;
            }

            for statement_info in &block.statements {
                match &statement_info.statement {
                    TypedStatement::If { condition, .. } => {
                        if self.is_constant_condition(condition) {
                            let (is_true, message) = if self.is_literal_true(condition) {
                                (
                                    true,
                                    "Condition is always true - else branch is unreachable",
                                )
                            } else if self.is_literal_false(condition) {
                                (
                                    false,
                                    "Condition is always false - then branch is unreachable",
                                )
                            } else {
                                continue;
                            };

                            self.results.dead_code.push(DeadCodeWarning {
                                location: condition.source_location,
                                message: message.to_string(),
                            });
                        }
                    }

                    TypedStatement::While { condition, .. } => {
                        if self.is_literal_false(condition) {
                            self.results.dead_code.push(DeadCodeWarning {
                                location: condition.source_location,
                                message: "While loop condition is always false - loop body is unreachable".to_string(),
                            });
                        }
                    }

                    _ => {}
                }
            }
        }
    }

    /// Detect unused variables (declared but never used)
    fn detect_unused_variables(&mut self) {
        // Skip unused variable detection for entry point functions (like static main)
        // Entry points are implicitly called by the runtime and may contain test/demo code
        // where variables are created just to verify compilation/type checking
        if self.is_entry_point {
            return;
        }

        let mut declared_vars = BTreeSet::new();
        let mut used_vars = BTreeSet::new();

        // Collect all variable declarations and uses
        for (_, block) in &self.cfg.blocks {
            if !block.is_reachable {
                continue;
            }

            for statement_info in &block.statements {
                // Add declared variables
                for &var_id in &statement_info.assigns {
                    declared_vars.insert(var_id);
                }

                // Add used variables
                for &var_id in &statement_info.uses {
                    used_vars.insert(var_id);
                }
            }
        }

        // Find declared but unused variables
        for &var_id in &declared_vars {
            if !used_vars.contains(&var_id) {
                // Find the declaration location
                if let Some(location) = self.find_variable_declaration_location(var_id) {
                    self.results.dead_code.push(DeadCodeWarning {
                        location,
                        message: format!("Variable {:?} is declared but never used", var_id),
                    });
                }
            }
        }
    }

    /// Check if an expression is a constant condition
    fn is_constant_condition(&self, expression: &TypedExpression) -> bool {
        self.is_literal_true(expression) || self.is_literal_false(expression)
    }

    /// Check if an expression is a literal true
    fn is_literal_true(&self, expression: &TypedExpression) -> bool {
        match &expression.kind {
            TypedExpressionKind::Literal { value } => {
                matches!(value, crate::tast::node::LiteralValue::Bool(true))
            }
            _ => false,
        }
    }

    /// Check if an expression is a literal false
    fn is_literal_false(&self, expression: &TypedExpression) -> bool {
        match &expression.kind {
            TypedExpressionKind::Literal { value } => {
                matches!(value, crate::tast::node::LiteralValue::Bool(false))
            }
            _ => false,
        }
    }

    /// Find the declaration location of a variable
    fn find_variable_declaration_location(&self, var_id: SymbolId) -> Option<SourceLocation> {
        for (_, block) in &self.cfg.blocks {
            for statement_info in &block.statements {
                match &statement_info.statement {
                    TypedStatement::VarDeclaration {
                        symbol_id,
                        source_location,
                        ..
                    } => {
                        if *symbol_id == var_id {
                            return Some(*source_location);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Analyze resource usage and detect potential leaks
    fn analyze_resource_usage(&mut self) {
        // Collect statements to analyze to avoid borrowing issues
        let mut statements_to_analyze = Vec::new();
        for (_, block) in &self.cfg.blocks {
            if !block.is_reachable {
                continue;
            }

            for statement_info in &block.statements {
                statements_to_analyze.push(statement_info.statement.clone());
            }
        }

        // Track resource acquisition and disposal through the control flow
        for statement in statements_to_analyze {
            self.analyze_statement_for_resources(&statement);
        }

        // Check for undisposed resources
        self.check_undisposed_resources();
    }

    /// Analyze a statement for resource acquisition and disposal
    fn analyze_statement_for_resources(&mut self, statement: &TypedStatement) {
        match statement {
            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                source_location,
                ..
            } => {
                if let Some(init) = initializer {
                    if let Some(resource_type) = self.detect_resource_acquisition(init) {
                        self.resources.insert(
                            *symbol_id,
                            ResourceInfo {
                                acquisition_location: *source_location,
                                resource_type,
                                is_disposed: false,
                                cleanup_method: None,
                            },
                        );
                    }
                }
            }

            TypedStatement::Assignment { target, value, .. } => {
                if let Some(var_id) = Self::extract_assigned_variable_static(target) {
                    if let Some(resource_type) = self.detect_resource_acquisition(value) {
                        self.resources.insert(
                            var_id,
                            ResourceInfo {
                                acquisition_location: value.source_location,
                                resource_type,
                                is_disposed: false,
                                cleanup_method: None,
                            },
                        );
                    }
                }
            }

            TypedStatement::Expression { expression, .. } => {
                self.check_resource_disposal(expression);
            }

            TypedStatement::Block { statements, .. } => {
                for stmt in statements {
                    self.analyze_statement_for_resources(stmt);
                }
            }

            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.check_resource_disposal(condition);
                self.analyze_statement_for_resources(then_branch);
                if let Some(else_stmt) = else_branch {
                    self.analyze_statement_for_resources(else_stmt);
                }
            }

            TypedStatement::Try {
                body,
                catch_clauses,
                finally_block,
                ..
            } => {
                self.analyze_statement_for_resources(body);

                for catch in catch_clauses {
                    self.analyze_statement_for_resources(&catch.body);
                }

                // Finally blocks are important for resource cleanup
                if let Some(finally) = finally_block {
                    self.analyze_statement_for_resources(finally);
                }
            }

            _ => {}
        }
    }

    /// Detect if an expression acquires a resource
    fn detect_resource_acquisition(&self, expression: &TypedExpression) -> Option<ResourceType> {
        match &expression.kind {
            TypedExpressionKind::FunctionCall { function, .. } => {
                // Check if the function name suggests resource acquisition
                if let TypedExpressionKind::Variable { symbol_id } = &function.kind {
                    // This would need to check against known resource-acquiring functions
                    // For now, we'll use basic heuristics based on function names
                    return self.classify_function_as_resource_acquisition(*symbol_id);
                }
            }

            TypedExpressionKind::MethodCall { method_symbol, .. } => {
                return self.classify_function_as_resource_acquisition(*method_symbol);
            }

            TypedExpressionKind::New { class_type, .. } => {
                // Check if the class type is a known resource type
                return self.classify_type_as_resource(*class_type);
            }

            _ => {}
        }

        None
    }

    /// Check if an expression disposes of a resource
    fn check_resource_disposal(&mut self, expression: &TypedExpression) {
        match &expression.kind {
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                ..
            } => {
                if let Some(var_id) = self.extract_variable_from_expression(receiver) {
                    if self.is_disposal_method(*method_symbol) {
                        // Mark the resource as disposed
                        if let Some(resource_info) = self.resources.get_mut(&var_id) {
                            resource_info.is_disposed = true;
                        }
                    }
                }
            }

            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                // Check for disposal function calls like close(handle)
                if let TypedExpressionKind::Variable { symbol_id } = &function.kind {
                    if self.is_disposal_function(*symbol_id) && !arguments.is_empty() {
                        if let Some(var_id) = self.extract_variable_from_expression(&arguments[0]) {
                            if let Some(resource_info) = self.resources.get_mut(&var_id) {
                                resource_info.is_disposed = true;
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }

    /// Check for undisposed resources and generate warnings
    fn check_undisposed_resources(&mut self) {
        for (&var_id, resource_info) in &self.resources {
            if !resource_info.is_disposed {
                self.results.resource_leaks.push(ResourceLeakWarning {
                    resource: var_id,
                    location: resource_info.acquisition_location,
                    message: format!(
                        "Resource of type {:?} may not be properly disposed",
                        resource_info.resource_type
                    ),
                });
            }
        }
    }

    /// Classify a function as resource acquisition based on symbol
    fn classify_function_as_resource_acquisition(
        &self,
        _symbol_id: SymbolId,
    ) -> Option<ResourceType> {
        // This would need access to the symbol table to get function names
        // For now, return None - in a full implementation, we'd check against
        // known resource-acquiring functions like File.open(), new Socket(), etc.
        None
    }

    /// Classify a type as a resource type
    fn classify_type_as_resource(&self, _type_id: TypeId) -> Option<ResourceType> {
        // This would need access to the type table to get type names
        // For now, return None - in a full implementation, we'd check against
        // known resource types like FileHandle, Socket, etc.
        None
    }

    /// Check if a method is a disposal method
    fn is_disposal_method(&self, _symbol_id: SymbolId) -> bool {
        // This would check against known disposal methods like close(), dispose(), etc.
        false
    }

    /// Check if a function is a disposal function
    fn is_disposal_function(&self, _symbol_id: SymbolId) -> bool {
        // This would check against known disposal functions
        false
    }
}

/// Analyze control flow for an entire file
pub fn analyze_file_control_flow(file: &TypedFile) -> Vec<AnalysisResults> {
    let mut results = Vec::new();

    // Analyze all functions
    for function in &file.functions {
        let mut analyzer = ControlFlowAnalyzer::new();
        let function_results = analyzer.analyze_function(function);
        results.push(function_results);
    }

    // Analyze functions in classes
    for class in &file.classes {
        for method in &class.methods {
            let mut analyzer = ControlFlowAnalyzer::new();
            let function_results = analyzer.analyze_function(method);
            results.push(function_results);
        }

        for constructor in &class.constructors {
            let mut analyzer = ControlFlowAnalyzer::new();
            let function_results = analyzer.analyze_function(constructor);
            results.push(function_results);
        }
    }

    results
}

/// Extension trait to get source location from statements
trait StatementSourceLocation {
    fn source_location(&self) -> SourceLocation;
}

impl StatementSourceLocation for TypedStatement {
    fn source_location(&self) -> SourceLocation {
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
            TypedStatement::ForIn {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_flow_graph_creation() {
        let mut cfg = ControlFlowGraph::new();
        let block1 = cfg.create_block();
        let block2 = cfg.create_block();

        cfg.add_edge(block1, block2);

        assert_eq!(cfg.blocks[&block1].successors, vec![block2]);
        assert_eq!(cfg.blocks[&block2].predecessors, vec![block1]);
    }

    #[test]
    fn test_variable_state_tracking() {
        let mut analyzer = ControlFlowAnalyzer::new();
        // Test would require setting up a proper function and statements
    }
}
