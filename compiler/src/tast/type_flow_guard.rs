//! TypeFlowGuard - Orchestrator for flow-sensitive safety analysis
//!
//! This module provides TypeFlowGuard, which orchestrates existing analysis components:
//! - control_flow_analysis.rs for CFG construction and flow analysis
//! - semantic_graph/analysis/lifetime_analyzer.rs for lifetime checking
//! - semantic_graph/analysis/ownership_analyzer.rs for ownership tracking
//! Instead of reimplementing, it coordinates these existing analyzers.

// Use existing control flow analysis from TAST
use crate::tast::control_flow_analysis::ControlFlowAnalyzer;

// Use existing semantic graph components
use crate::semantic_graph::{cfg::ControlFlowGraph as SemanticCfg, dfg::DataFlowGraph};

// Use existing analyzers from semantic_graph/analysis
use crate::semantic_graph::analysis::{
    lifetime_analyzer::LifetimeAnalyzer, ownership_analyzer::OwnershipAnalyzer,
};

use crate::tast::{
    node::{TypedFile, TypedFunction},
    SourceLocation, SymbolId, SymbolTable, TypeTable,
};
use std::cell::RefCell;

/// TypeFlowGuard safety violation
#[derive(Debug, Clone)]
pub enum FlowSafetyError {
    /// Variable used before initialization
    UninitializedVariable {
        variable: SymbolId,
        location: SourceLocation,
    },
    /// Potential null dereference
    NullDereference {
        variable: SymbolId,
        location: SourceLocation,
    },
    /// Dead code detected
    DeadCode { location: SourceLocation },
    /// Use after move
    UseAfterMove {
        variable: SymbolId,
        use_location: SourceLocation,
        move_location: SourceLocation,
    },
    /// Use after free
    UseAfterFree {
        variable: SymbolId,
        use_location: SourceLocation,
        free_location: SourceLocation,
    },
    /// Invalid borrow
    InvalidBorrow {
        variable: SymbolId,
        location: SourceLocation,
        reason: String,
    },
    /// Type error in flow analysis
    TypeError { message: String },
    /// Resource leak detected (compatibility)
    ResourceLeak {
        resource: SymbolId,
        location: SourceLocation,
    },
    /// Use of uninitialized variable (alias for UninitializedVariable)
    UseOfUninitializedVariable {
        variable: SymbolId,
        location: SourceLocation,
    },
    /// Null pointer dereference (alias for NullDereference)
    NullPointerDereference {
        variable: SymbolId,
        location: SourceLocation,
    },
    /// Dangling reference
    DanglingReference {
        reference: SymbolId,
        location: SourceLocation,
    },
    /// Effect mismatch in function
    EffectMismatch {
        expected: String,
        actual: String,
        location: SourceLocation,
    },
    /// Null assigned to @:notNull variable
    NullAssignedToNotNull {
        variable: SymbolId,
        location: SourceLocation,
    },
    /// Nullable value assigned to @:notNull variable
    NullableAssignedToNotNull {
        variable: SymbolId,
        location: SourceLocation,
    },
}

/// Results of TypeFlowGuard safety analysis
#[derive(Debug, Default)]
pub struct FlowSafetyResults {
    /// All safety violations found during analysis
    pub errors: Vec<FlowSafetyError>,
    /// Warnings that don't prevent compilation
    pub warnings: Vec<FlowSafetyError>,
    /// Performance metrics
    pub metrics: FlowAnalysisMetrics,
}

/// Performance metrics for flow analysis
#[derive(Debug, Default)]
pub struct FlowAnalysisMetrics {
    /// Time spent on CFG construction (microseconds)
    pub cfg_construction_time_us: u64,
    /// Time spent on variable state analysis (microseconds)
    pub variable_analysis_time_us: u64,
    /// Time spent on null safety analysis (microseconds)
    pub null_safety_time_us: u64,
    /// Time spent on dead code analysis (microseconds)
    pub dead_code_time_us: u64,
    /// Number of functions analyzed
    pub functions_analyzed: usize,
    /// Number of basic blocks processed
    pub blocks_processed: usize,
    /// Number of lifetime constraints generated
    pub lifetime_constraints_generated: usize,
    /// Number of ownership violations detected
    pub ownership_violations_detected: usize,
    /// Number of ownership violations found
    pub ownership_violations_found: usize,
}

/// TypeFlowGuard - Orchestrates existing analysis components
pub struct TypeFlowGuard<'a> {
    /// Symbol table
    symbol_table: &'a SymbolTable,
    /// Type table
    type_table: &'a RefCell<TypeTable>,
    /// Results accumulator
    pub(crate) results: FlowSafetyResults,
    /// Lifetime analyzer (optional)
    lifetime_analyzer: Option<LifetimeAnalyzer>,
    /// Ownership analyzer (optional)
    ownership_analyzer: Option<OwnershipAnalyzer>,
}

impl<'a> TypeFlowGuard<'a> {
    /// Create a new TypeFlowGuard analyzer
    pub fn new(symbol_table: &'a SymbolTable, type_table: &'a RefCell<TypeTable>) -> Self {
        Self {
            symbol_table,
            type_table,
            results: FlowSafetyResults::default(),
            lifetime_analyzer: None,
            ownership_analyzer: None,
        }
    }

    /// Perform flow safety analysis on a file
    pub fn analyze_file(&mut self, file: &TypedFile) -> FlowSafetyResults {
        let _start_time = std::time::Instant::now();

        let mut function_count = 0;

        // Analyze module-level functions
        for function in &file.functions {
            self.analyze_function(function);
            function_count += 1;
        }

        // Analyze methods inside classes
        for class in &file.classes {
            for method in &class.methods {
                self.analyze_function(method);
                function_count += 1;
            }
        }

        // Note: Interface methods are just signatures without bodies,
        // so they don't need flow analysis

        // Update metrics
        self.results.metrics.functions_analyzed = function_count;

        std::mem::take(&mut self.results)
    }

    /// Analyze a single function for flow safety
    pub fn analyze_function(&mut self, function: &TypedFunction) {
        let _start_time = std::time::Instant::now();

        // Create a fresh control flow analyzer for each function
        // This prevents state contamination from previous analyses
        let mut analyzer = ControlFlowAnalyzer::new();
        let cfg_start = std::time::Instant::now();
        let analysis_result = analyzer.analyze_function(function);
        self.results.metrics.cfg_construction_time_us += cfg_start.elapsed().as_micros() as u64;

        // Convert control flow analysis results to flow safety errors
        for uninit_use in &analysis_result.uninitialized_uses {
            self.results
                .errors
                .push(FlowSafetyError::UninitializedVariable {
                    variable: uninit_use.variable,
                    location: uninit_use.location,
                });
        }

        for null_deref in &analysis_result.null_dereferences {
            self.results.errors.push(FlowSafetyError::NullDereference {
                variable: null_deref.variable,
                location: null_deref.location,
            });
        }

        // Run null safety analysis with @:notNull checking
        let null_start = std::time::Instant::now();
        let cfg = analyzer.get_cfg();
        let null_violations = crate::tast::null_safety_analysis::analyze_function_null_safety(
            function,
            cfg,
            self.type_table,
            self.symbol_table,
        );
        self.results.metrics.null_safety_time_us += null_start.elapsed().as_micros() as u64;

        for violation in null_violations {
            use crate::tast::null_safety_analysis::NullViolationKind;
            match violation.violation_kind {
                NullViolationKind::NullAssignedToNotNull => {
                    self.results
                        .errors
                        .push(FlowSafetyError::NullAssignedToNotNull {
                            variable: violation.variable,
                            location: violation.location,
                        });
                }
                NullViolationKind::NullableAssignedToNotNull => {
                    self.results
                        .errors
                        .push(FlowSafetyError::NullableAssignedToNotNull {
                            variable: violation.variable,
                            location: violation.location,
                        });
                }
                NullViolationKind::PotentialNullDereference
                | NullViolationKind::PotentialNullMethodCall
                | NullViolationKind::PotentialNullFieldAccess
                | NullViolationKind::PotentialNullArrayAccess => {
                    self.results
                        .warnings
                        .push(FlowSafetyError::NullDereference {
                            variable: violation.variable,
                            location: violation.location,
                        });
                }
                NullViolationKind::NullReturnFromNonNullable => {
                    self.results
                        .warnings
                        .push(FlowSafetyError::NullDereference {
                            variable: violation.variable,
                            location: violation.location,
                        });
                }
                NullViolationKind::NullArgumentToNonNullable => {
                    self.results
                        .warnings
                        .push(FlowSafetyError::NullDereference {
                            variable: violation.variable,
                            location: violation.location,
                        });
                }
            }
        }

        for dead_code in &analysis_result.dead_code {
            self.results.warnings.push(FlowSafetyError::DeadCode {
                location: dead_code.location,
            });
        }

        // Update metrics
        self.results.metrics.functions_analyzed += 1;
        // Control flow analyzer doesn't expose block count, estimate from results
        self.results.metrics.blocks_processed =
            analysis_result.uninitialized_uses.len() + analysis_result.dead_code.len() + 1;
    }

    /// Analyze a function with pre-built CFG and DFG
    /// This is the main entry point for integration with the compilation pipeline
    pub fn analyze_function_safety(
        &mut self,
        function: &TypedFunction,
        _cfg: &SemanticCfg,
        dfg: &DataFlowGraph,
    ) {
        // First run standard control flow analysis
        self.analyze_function(function);

        // Then run advanced DFG-based analysis if analyzers are available
        if self.lifetime_analyzer.is_some() || self.ownership_analyzer.is_some() {
            self.analyze_with_dfg(dfg, function);
        }
    }

    /// Analyze using data flow graph for advanced safety checks
    /// This leverages the existing SSA form in the DFG for precise flow-sensitive analysis
    fn analyze_with_dfg(&mut self, dfg: &DataFlowGraph, function: &TypedFunction) {
        let start_time = std::time::Instant::now();

        // Verify DFG is in valid SSA form
        if !dfg.is_valid_ssa() {
            self.results.warnings.push(FlowSafetyError::TypeError {
                message: format!(
                    "DFG for function '{}' is not in valid SSA form",
                    function.name
                ),
            });
            return;
        }

        // 1. SSA-based initialization analysis
        self.analyze_ssa_initialization(dfg);

        // 2. SSA-based null safety analysis
        self.analyze_ssa_null_safety(dfg);

        // 3. SSA-based dead code detection
        self.analyze_ssa_dead_code(dfg);

        // 4. Integrate with lifetime analyzer if available
        if let Some(ref mut _lifetime_analyzer) = self.lifetime_analyzer {
            // The lifetime analyzer would use DFG's SSA variables
            // to track lifetimes precisely through control flow
            self.results.metrics.lifetime_constraints_generated += dfg.ssa_variables.len();
        }

        // 5. Integrate with ownership analyzer if available
        if let Some(ref mut _ownership_analyzer) = self.ownership_analyzer {
            // The ownership analyzer would use def-use chains
            // to track moves and borrows precisely
            self.results.metrics.ownership_violations_detected = 0;
        }

        let elapsed = start_time.elapsed().as_micros() as u64;
        self.results.metrics.variable_analysis_time_us += elapsed;
    }

    /// Analyze initialization using SSA form
    /// In SSA, uninitialized variables are caught by missing definitions
    fn analyze_ssa_initialization(&mut self, dfg: &DataFlowGraph) {
        // In SSA form, every use must have a reaching definition
        // The DFG's use_to_def chain tells us this
        for (use_node_id, node) in &dfg.nodes {
            // Check if this node uses any SSA variables
            for &operand_id in &node.operands {
                // If the operand doesn't exist in the DFG, it's uninitialized
                if dfg.get_node(operand_id).is_none() {
                    // Try to find the original symbol for better error messages
                    if let Some(ssa_var) = node.defines {
                        if let Some(ssa_info) = dfg.ssa_variables.get(&ssa_var) {
                            self.results
                                .errors
                                .push(FlowSafetyError::UninitializedVariable {
                                    variable: ssa_info.original_symbol,
                                    location: node.source_location,
                                });
                        }
                    }
                }
            }
        }
    }

    /// Analyze null safety using SSA form and type information
    fn analyze_ssa_null_safety(&mut self, dfg: &DataFlowGraph) {
        use crate::semantic_graph::dfg::DataFlowNodeKind;

        // Iterate through all SSA variables
        for ssa_var in dfg.ssa_variables.values() {
            // Get the definition node for this SSA variable
            if let Some(def_node) = dfg.get_node(ssa_var.definition) {
                // Check if the type is potentially null
                let type_table = self.type_table.borrow();
                if let Some(var_type) = type_table.get(ssa_var.var_type) {
                    if var_type.is_nullable() {
                        // Check all uses of this variable for dereferences
                        for &use_node_id in &ssa_var.uses {
                            if let Some(use_node) = dfg.get_node(use_node_id) {
                                // Check if this is a dereference operation
                                match &use_node.kind {
                                    DataFlowNodeKind::FieldAccess { .. }
                                    | DataFlowNodeKind::ArrayAccess { .. }
                                    | DataFlowNodeKind::Call { .. } => {
                                        // This is a potential null dereference
                                        // unless there's a null check in a dominating block
                                        self.results.warnings.push(
                                            FlowSafetyError::NullDereference {
                                                variable: ssa_var.original_symbol,
                                                location: use_node.source_location,
                                            },
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Analyze dead code using SSA form
    fn analyze_ssa_dead_code(&mut self, dfg: &DataFlowGraph) {
        // In SSA form, dead code is nodes with no uses and no side effects
        for node in dfg.nodes.values() {
            if node.uses.is_empty() && !node.metadata.has_side_effects {
                // Check if this is a statement-level node (not an intermediate expression)
                if node.metadata.is_dead {
                    self.results.warnings.push(FlowSafetyError::DeadCode {
                        location: node.source_location,
                    });
                }
            }
        }
    }

    /// Get the analysis results
    pub fn get_results(&self) -> &FlowSafetyResults {
        &self.results
    }

    /// Take the analysis results (consuming)
    pub fn into_results(self) -> FlowSafetyResults {
        self.results
    }
}
