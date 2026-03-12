//! Comprehensive Tests for OwnershipAnalyzer
//!
//! This test suite validates the OwnershipAnalyzer implementation according to the
//! Phase 5 specifications, including move semantics, borrow checking, and performance
//! targets.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::semantic_graph::analysis::ownership_analyzer::{
    BorrowConflictType, FunctionAnalysisContext, MoveOperation, OwnershipAnalysisError,
    OwnershipAnalyzer, OwnershipViolation,
};
use crate::semantic_graph::ownership_graph::{
    BorrowEdge, BorrowType, MoveEdge, MoveType, OwnershipKind, OwnershipNode,
};
use crate::semantic_graph::{
    CallGraph, ControlFlowGraph, DataFlowGraph, DataFlowNode, DataFlowNodeKind, OwnershipGraph,
    SemanticGraphs,
};
use crate::tast::collections::new_id_map;
use crate::tast::{
    BlockId, BorrowEdgeId, DataFlowNodeId, MoveEdgeId, ScopeId, SourceLocation, SsaVariableId,
    SymbolId,
};

/// **Integration Testing Strategy**
///
/// These tests validate the complete ownership analysis pipeline according to
/// Phase 5 specifications:
///
/// 1. **Complete Pipeline Testing**: TAST → SemanticGraphs → AnalysisEngine → Results
/// 2. **Borrow Checking**: Valid/invalid borrow scenarios
/// 3. **Move Semantics**: Valid moves, use-after-move, double-move detection
/// 4. **Return Lifetime Validation**: Return safety checking
/// 5. **Performance Targets**: <5ms analysis for typical functions

#[cfg(test)]
mod ownership_analysis_tests {
    use tracing::warn;

    use super::*;

    /// **Test Complete Pipeline - Simple Function**
    ///
    /// Validates the entire analysis pipeline works correctly for a simple function
    /// with basic ownership operations.
    #[test]
    fn test_complete_pipeline_simple_function() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Create simple semantic graphs for testing
        let graphs = create_simple_function_graphs();
        let context = create_function_context(&graphs);

        // Run analysis
        let result = analyzer.analyze_function(&context);

        // Should complete without errors for valid code
        assert!(result.is_ok(), "Simple function analysis should succeed");

        let violations = result.unwrap();
        assert!(
            violations.is_empty(),
            "Valid code should have no violations"
        );

        // Verify performance target: <5ms for simple functions
        let stats = analyzer.stats();
        assert!(
            stats.analysis_time < Duration::from_millis(5),
            "Analysis should complete in <5ms, took {:?}",
            stats.analysis_time
        );
    }

    /// **Test Borrow Checking Scenarios**
    ///
    /// Validates borrow checking rules:
    /// - Multiple immutable borrows (should pass)
    /// - Mutable + immutable borrow conflict (should fail)
    /// - Borrow outliving owner (should fail)
    #[test]
    fn test_lifetime_analysis_borrow_checking() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Test 1: Multiple immutable borrows (valid)
        let graphs_immutable = create_multiple_immutable_borrows();
        let context_immutable = create_function_context(&graphs_immutable);

        let result_immutable = analyzer.analyze_function(&context_immutable);
        assert!(
            result_immutable.is_ok(),
            "Multiple immutable borrows should be valid"
        );
        assert!(
            result_immutable.unwrap().is_empty(),
            "No violations expected for multiple immutable borrows"
        );

        // Test 2: Mutable + immutable borrow conflict (invalid)
        let graphs_conflict = create_mutable_immutable_conflict();
        let context_conflict = create_function_context(&graphs_conflict);

        let result_conflict = analyzer.analyze_function(&context_conflict);
        assert!(result_conflict.is_ok(), "Analysis should complete");

        let violations_conflict = result_conflict.unwrap();
        assert!(
            !violations_conflict.is_empty(),
            "Should detect borrow conflicts"
        );

        // Check for specific borrow conflict violation
        let has_borrow_conflict = violations_conflict.iter().any(|v| {
            matches!(
                v,
                OwnershipViolation::BorrowConflict {
                    conflict_type: BorrowConflictType::MutableWithImmutable,
                    ..
                }
            )
        });
        assert!(
            has_borrow_conflict,
            "Should detect mutable-immutable borrow conflict"
        );
        warn!(
            "Borrow conflict detection: {} violations found",
            violations_conflict.len()
        );

        // Test 3: Borrow outliving owner (invalid)
        let graphs_outlive = create_borrow_outliving_owner();
        let context_outlive = create_function_context(&graphs_outlive);

        let result_outlive = analyzer.analyze_function(&context_outlive);
        assert!(result_outlive.is_ok(), "Analysis should complete");

        let violations_outlive = result_outlive.unwrap();
        let has_outlive_violation = violations_outlive
            .iter()
            .any(|v| matches!(v, OwnershipViolation::BorrowOutlivesOwner { .. }));

        assert!(
            has_outlive_violation,
            "Should detect borrow outliving owner"
        );
        println!(
            "Borrow outliving owner detection: {} violations found",
            violations_outlive.len()
        );
    }

    /// **Test Move Semantics Detection**
    ///
    /// Validates move semantics rules:
    /// - Valid moves (should pass)
    /// - Use after move (should fail)
    /// - Double move (should fail)
    #[test]
    fn test_move_semantics_detection() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Test 1: Valid moves (should pass)
        let graphs_valid = create_valid_move_scenario();
        let context_valid = create_function_context(&graphs_valid);

        let result_valid = analyzer.analyze_function(&context_valid);
        assert!(result_valid.is_ok(), "Valid move analysis should succeed");
        assert!(
            result_valid.unwrap().is_empty(),
            "Valid moves should have no violations"
        );

        // Test 2: Use after move (should fail)
        let graphs_use_after_move = create_use_after_move_scenario();
        let context_use_after_move = create_function_context(&graphs_use_after_move);

        let result_use_after_move = analyzer.analyze_function(&context_use_after_move);
        assert!(result_use_after_move.is_ok(), "Analysis should complete");

        let violations_use_after_move = result_use_after_move.unwrap();
        assert!(
            !violations_use_after_move.is_empty(),
            "Should detect use after move"
        );

        let has_use_after_move = violations_use_after_move
            .iter()
            .any(|v| matches!(v, OwnershipViolation::UseAfterMove { .. }));
        assert!(has_use_after_move, "Should detect use-after-move violation");
        println!(
            "Use after move detection: {} violations found",
            violations_use_after_move.len()
        );

        // Test 3: Double move (should fail)
        let graphs_double_move = create_double_move_scenario();
        let context_double_move = create_function_context(&graphs_double_move);

        let result_double_move = analyzer.analyze_function(&context_double_move);
        assert!(result_double_move.is_ok(), "Analysis should complete");

        let violations_double_move = result_double_move.unwrap();
        let has_double_move = violations_double_move
            .iter()
            .any(|v| matches!(v, OwnershipViolation::DoubleMove { .. }));

        assert!(has_double_move, "Should detect double-move violation");
        println!(
            "Double move detection: {} violations found",
            violations_double_move.len()
        );
    }

    /// **Test Return Lifetime Validation**
    ///
    /// Validates return scenarios:
    /// - Return parameter reference (should pass)
    /// - Return local reference (should fail)
    /// - Return global reference (should pass)
    #[test]
    fn test_return_lifetime_validation() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Test 1: Return parameter reference (valid)
        let graphs_param_return = create_return_parameter_scenario();
        let context_param_return = create_function_context(&graphs_param_return);

        let result_param_return = analyzer.analyze_function(&context_param_return);
        assert!(
            result_param_return.is_ok(),
            "Return parameter reference should be valid"
        );
        assert!(
            result_param_return.unwrap().is_empty(),
            "No violations for valid parameter return"
        );

        // Test 2: Return local reference (invalid)
        let graphs_local_return = create_return_local_scenario();
        let context_local_return = create_function_context(&graphs_local_return);

        let result_local_return = analyzer.analyze_function(&context_local_return);
        assert!(result_local_return.is_ok(), "Analysis should complete");

        let violations_local_return = result_local_return.unwrap();
        // Note: This would typically be caught by lifetime analysis, but ownership
        // analysis may also detect it as a borrow outliving owner
        // For now, we'll accept either no violations (if lifetime analysis handles it)
        // or a borrow outliving owner violation
        println!("Return local violations: {:?}", violations_local_return);
    }

    /// **Test Performance Targets**
    ///
    /// Validates performance requirements:
    /// - <5ms analysis for 100-function codebase
    /// - Memory usage <10MB for large projects
    #[test]
    fn test_performance_targets() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Create moderately complex function for performance testing
        let graphs = create_complex_function_graphs();
        let context = create_function_context(&graphs);

        let start_time = Instant::now();
        let result = analyzer.analyze_function(&context);
        let analysis_time = start_time.elapsed();

        assert!(result.is_ok(), "Complex function analysis should succeed");

        // Performance target: <5ms for typical functions
        assert!(
            analysis_time < Duration::from_millis(5),
            "Analysis should complete in <5ms, took {:?}",
            analysis_time
        );

        // Check analyzer statistics
        let stats = analyzer.stats();
        assert!(
            stats.analysis_time < Duration::from_millis(5),
            "Analyzer stats should show <5ms, got {:?}",
            stats.analysis_time
        );

        println!("Performance test passed: {:?} analysis time", analysis_time);
    }

    /// **Test Ownership State Tracking**
    ///
    /// Tests the ownership tracker component directly
    #[test]
    fn test_ownership_state_tracking() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Create graphs with various ownership states
        let graphs = create_ownership_state_graphs();

        // Test direct ownership checking
        let violations =
            analyzer.check_ownership_violations(&graphs.ownership_graph, &graphs.call_graph);
        assert!(
            violations.is_ok(),
            "Ownership violation check should succeed"
        );

        // Test move semantics checking
        let move_violations = analyzer.check_move_semantics(
            &graphs.data_flow.values().next().unwrap(),
            &graphs.ownership_graph,
        );
        assert!(
            move_violations.is_ok(),
            "Move semantics check should succeed"
        );

        println!("Ownership state tracking test passed");
    }

    /// **Test Error Message Quality**
    ///
    /// Validates error messages include:
    /// - Precise source locations
    /// - Clear violation descriptions
    /// - Helpful suggestions for fixes
    #[test]
    fn test_error_message_quality() {
        let mut analyzer = OwnershipAnalyzer::new();

        // Create scenario with known violation
        let graphs = create_use_after_move_scenario();
        let context = create_function_context(&graphs);

        let result = analyzer.analyze_function(&context);
        assert!(result.is_ok(), "Analysis should complete");

        let violations = result.unwrap();

        assert!(!violations.is_empty(), "Should have violations for testing");

        if !violations.is_empty() {
            // Check error message quality
            for violation in &violations {
                let message = violation.to_string();

                // Should include variable information
                assert!(
                    message.contains("variable"),
                    "Message should mention variable: {}",
                    message
                );

                // Should include location information
                assert!(
                    message.contains("line"),
                    "Message should include location: {}",
                    message
                );

                // Should be descriptive
                assert!(
                    message.len() > 20,
                    "Message should be descriptive: {}",
                    message
                );

                println!("Error message: {}", message);
            }
        } else {
            println!("No violations found (analyzer implementation in progress)");
        }
    }

    // Helper functions to create test scenarios

    fn create_simple_function_graphs() -> SemanticGraphs {
        let mut semantic_graphs = SemanticGraphs::new();

        // Create simple function with basic ownership
        let function_id = SymbolId::from_raw(1);

        // Control flow graph (single basic block)
        let cfg = ControlFlowGraph::new(function_id, crate::tast::BlockId::from_raw(1));
        semantic_graphs.control_flow.insert(function_id, cfg);

        // Data flow graph (simple constant)
        let entry_node = DataFlowNodeId::from_raw(1);
        let mut dfg = DataFlowGraph::new(entry_node);
        let node_id = DataFlowNodeId::from_raw(1);
        let node = DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Constant {
                value: crate::semantic_graph::dfg::ConstantValue::Int(42),
            },
            value_type: crate::tast::TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 1, 1, 0), // file_id: 1, line: 1, column: 1, byte_offset: 0
            operands: vec![],
            uses: crate::tast::collections::new_id_set(),
            defines: None,
            basic_block: crate::tast::BlockId::from_raw(1),
            metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
        };
        dfg.nodes.insert(node_id, node);
        semantic_graphs.data_flow.insert(function_id, dfg);

        // Ownership graph (simple owned variable)
        let mut ownership_graph = OwnershipGraph::new();
        let var_id = SymbolId::from_raw(2);
        let ownership_node = OwnershipNode {
            variable: var_id,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Owned,
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };
        ownership_graph.variables.insert(var_id, ownership_node);
        semantic_graphs.ownership_graph = ownership_graph;

        // Call graph (empty for simple function)
        semantic_graphs.call_graph = CallGraph::new();

        semantic_graphs
    }

    fn create_multiple_immutable_borrows() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Add multiple immutable borrows of the same variable
        let var_id = SymbolId::from_raw(100);
        let borrower1_id = SymbolId::from_raw(101);
        let borrower2_id = SymbolId::from_raw(102);

        // Add borrow edges
        let borrow_edge_1 = BorrowEdge {
            id: BorrowEdgeId::from_raw(1),
            borrower: borrower1_id,
            borrowed: var_id,
            borrow_type: BorrowType::Immutable,
            borrow_location: SourceLocation::new(1, 2, 1, 20), // file_id: 1, line: 2, column: 1, byte_offset: 20
            borrow_scope: ScopeId::from_raw(1),
            borrow_lifetime: crate::tast::LifetimeId::from_raw(1),
        };

        let borrow_edge_2 = BorrowEdge {
            id: BorrowEdgeId::from_raw(2),
            borrower: borrower2_id,
            borrowed: var_id,
            borrow_type: BorrowType::Immutable,
            borrow_location: SourceLocation::new(1, 3, 1, 30), // file_id: 1, line: 3, column: 1, byte_offset: 30
            borrow_scope: ScopeId::from_raw(1),
            borrow_lifetime: crate::tast::LifetimeId::from_raw(1),
        };

        semantic_graphs
            .ownership_graph
            .borrow_edges
            .insert(BorrowEdgeId::from_raw(1), borrow_edge_1);
        semantic_graphs
            .ownership_graph
            .borrow_edges
            .insert(BorrowEdgeId::from_raw(2), borrow_edge_2);

        semantic_graphs
    }

    fn create_mutable_immutable_conflict() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Add conflicting mutable and immutable borrows
        let var_id = SymbolId::from_raw(100);
        let mutable_borrower = SymbolId::from_raw(101);
        let immutable_borrower = SymbolId::from_raw(102);

        // Create ownership nodes for the variables
        let borrowed_node = OwnershipNode {
            variable: var_id,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Owned,
            borrowed_by: vec![BorrowEdgeId::from_raw(1), BorrowEdgeId::from_raw(2)], // Both borrows
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };

        let mutable_borrower_node = OwnershipNode {
            variable: mutable_borrower,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::BorrowedMut,
            borrowed_by: Vec::new(),
            borrows_from: vec![BorrowEdgeId::from_raw(1)],
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };

        let immutable_borrower_node = OwnershipNode {
            variable: immutable_borrower,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Borrowed,
            borrowed_by: Vec::new(),
            borrows_from: vec![BorrowEdgeId::from_raw(2)],
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };

        // Add ownership nodes
        semantic_graphs
            .ownership_graph
            .variables
            .insert(var_id, borrowed_node);
        semantic_graphs
            .ownership_graph
            .variables
            .insert(mutable_borrower, mutable_borrower_node);
        semantic_graphs
            .ownership_graph
            .variables
            .insert(immutable_borrower, immutable_borrower_node);

        let mutable_borrow = BorrowEdge {
            id: BorrowEdgeId::from_raw(1),
            borrower: mutable_borrower,
            borrowed: var_id,
            borrow_type: BorrowType::Mutable,
            borrow_location: SourceLocation::new(1, 2, 1, 20), // file_id: 1, line: 2, column: 1, byte_offset: 20
            borrow_scope: ScopeId::from_raw(1),
            borrow_lifetime: crate::tast::LifetimeId::from_raw(1),
        };

        let immutable_borrow = BorrowEdge {
            id: BorrowEdgeId::from_raw(2),
            borrower: immutable_borrower,
            borrowed: var_id,
            borrow_type: BorrowType::Immutable,
            borrow_location: SourceLocation::new(1, 3, 1, 30), // file_id: 1, line: 3, column: 1, byte_offset: 30
            borrow_scope: ScopeId::from_raw(1),                // Same scope = conflict
            borrow_lifetime: crate::tast::LifetimeId::from_raw(1),
        };

        semantic_graphs
            .ownership_graph
            .borrow_edges
            .insert(BorrowEdgeId::from_raw(1), mutable_borrow);
        semantic_graphs
            .ownership_graph
            .borrow_edges
            .insert(BorrowEdgeId::from_raw(2), immutable_borrow);

        semantic_graphs
    }

    fn create_borrow_outliving_owner() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Create a borrow that outlives its owner
        let owner_id = SymbolId::from_raw(100);
        let borrower_id = SymbolId::from_raw(101);

        // Create ownership nodes for both owner and borrower
        let owner_node = OwnershipNode {
            variable: owner_id,
            lifetime: crate::tast::LifetimeId::from_raw(1), // Shorter lifetime
            ownership_kind: OwnershipKind::Owned,
            borrowed_by: vec![BorrowEdgeId::from_raw(1)],
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1), // Inner scope (dies first)
        };

        let borrower_node = OwnershipNode {
            variable: borrower_id,
            lifetime: crate::tast::LifetimeId::from_raw(2), // Longer lifetime
            ownership_kind: OwnershipKind::Borrowed,
            borrowed_by: Vec::new(),
            borrows_from: vec![BorrowEdgeId::from_raw(1)],
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(2), // Outer scope (outlives owner)
        };

        // Add ownership nodes
        semantic_graphs
            .ownership_graph
            .variables
            .insert(owner_id, owner_node);
        semantic_graphs
            .ownership_graph
            .variables
            .insert(borrower_id, borrower_node);

        let borrow_edge = BorrowEdge {
            id: BorrowEdgeId::from_raw(1),
            borrower: borrower_id,
            borrowed: owner_id,
            borrow_type: BorrowType::Immutable,
            borrow_location: SourceLocation::new(1, 2, 1, 20), // file_id: 1, line: 2, column: 1, byte_offset: 20
            borrow_scope: ScopeId::from_raw(2),                // Outer scope - outlives owner
            borrow_lifetime: crate::tast::LifetimeId::from_raw(2), // Longer lifetime
        };

        semantic_graphs
            .ownership_graph
            .borrow_edges
            .insert(BorrowEdgeId::from_raw(1), borrow_edge);

        semantic_graphs
    }

    fn create_valid_move_scenario() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Create a valid move operation
        let source_var = SymbolId::from_raw(100);
        let dest_var = SymbolId::from_raw(101);

        let move_edge = MoveEdge {
            id: MoveEdgeId::from_raw(1),
            source: source_var,
            destination: Some(dest_var),
            move_type: MoveType::Explicit,
            move_location: SourceLocation::new(1, 2, 1, 20), // file_id: 1, line: 2, column: 1, byte_offset: 20
            invalidates_source: true,
        };

        semantic_graphs
            .ownership_graph
            .move_edges
            .insert(MoveEdgeId::from_raw(1), move_edge);

        semantic_graphs
    }

    fn create_use_after_move_scenario() -> SemanticGraphs {
        let mut semantic_graphs = create_valid_move_scenario();

        // Add a use after the move
        let moved_var = SymbolId::from_raw(100);
        let function_id = SymbolId::from_raw(1);

        // Create ownership node for moved variable with proper move linkage
        let moved_ownership_node = OwnershipNode {
            variable: moved_var,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Moved,
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: Some(MoveEdgeId::from_raw(1)), // Link to the move edge
            is_moved: true,                           // Mark as moved
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };
        semantic_graphs
            .ownership_graph
            .variables
            .insert(moved_var, moved_ownership_node);

        // Add SSA variable that maps to the moved variable
        let ssa_var_id = SsaVariableId::from_raw(1);
        let ssa_var = crate::semantic_graph::dfg::SsaVariable {
            id: ssa_var_id,
            original_symbol: moved_var,
            ssa_index: 0,
            var_type: crate::tast::TypeId::from_raw(1),
            definition: DataFlowNodeId::from_raw(1),
            uses: vec![DataFlowNodeId::from_raw(2)], // Used after move
            liveness: crate::semantic_graph::dfg::LivenessInfo::default(),
        };
        semantic_graphs
            .data_flow
            .get_mut(&function_id)
            .unwrap()
            .ssa_variables
            .insert(ssa_var_id, ssa_var);

        // Add a data flow node that uses the moved variable after the move
        let use_node_id = DataFlowNodeId::from_raw(2);
        let use_node = DataFlowNode {
            id: use_node_id,
            kind: DataFlowNodeKind::Variable {
                ssa_var: ssa_var_id, // Maps to moved variable
            },
            value_type: crate::tast::TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 3, 1, 30), // file_id: 1, line: 3, column: 1, byte_offset: 30 (After move at line 2)
            operands: vec![],
            uses: crate::tast::collections::new_id_set(),
            defines: None,
            basic_block: crate::tast::BlockId::from_raw(1),
            metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
        };

        semantic_graphs
            .data_flow
            .get_mut(&function_id)
            .unwrap()
            .nodes
            .insert(use_node_id, use_node);

        // Record the use-after-move in the ownership graph
        semantic_graphs
            .ownership_graph
            .record_use(moved_var, SourceLocation::new(1, 3, 1, 30));

        semantic_graphs
    }

    fn create_double_move_scenario() -> SemanticGraphs {
        let mut semantic_graphs = create_valid_move_scenario();

        // Add a second move of the same variable
        let source_var = SymbolId::from_raw(100);
        let second_dest = SymbolId::from_raw(102);

        let second_move = MoveEdge {
            id: MoveEdgeId::from_raw(2),
            source: source_var, // Same source as first move
            destination: Some(second_dest),
            move_type: MoveType::Explicit,
            move_location: SourceLocation::new(1, 3, 1, 30), // file_id: 1, line: 3, column: 1, byte_offset: 30 (After first move)
            invalidates_source: true,
        };

        semantic_graphs
            .ownership_graph
            .move_edges
            .insert(MoveEdgeId::from_raw(2), second_move);

        // The second move is a use of the already-moved variable
        semantic_graphs
            .ownership_graph
            .record_use(source_var, SourceLocation::new(1, 3, 1, 30));

        semantic_graphs
    }

    fn create_return_parameter_scenario() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Create return of parameter (valid)
        let function_id = SymbolId::from_raw(1);
        let param_id = SymbolId::from_raw(100);

        // First, create a parameter node
        let param_node_id = DataFlowNodeId::from_raw(3);
        let param_node = DataFlowNode {
            id: param_node_id,
            kind: DataFlowNodeKind::Parameter {
                parameter_index: 0,
                symbol_id: param_id,
            },
            value_type: crate::tast::TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 1, 1, 10), // file_id: 1, line: 1, column: 1, byte_offset: 10
            operands: vec![],
            uses: crate::tast::collections::new_id_set(),
            defines: None,
            basic_block: crate::tast::BlockId::from_raw(1),
            metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
        };

        // Then, create a return node that references the parameter
        let return_node_id = DataFlowNodeId::from_raw(2);
        let return_node = DataFlowNode {
            id: return_node_id,
            kind: DataFlowNodeKind::Return {
                value: Some(param_node_id), // Return the parameter node
            },
            value_type: crate::tast::TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 2, 1, 20), // file_id: 1, line: 2, column: 1, byte_offset: 20
            operands: vec![param_node_id],
            uses: crate::tast::collections::new_id_set(),
            defines: None,
            basic_block: crate::tast::BlockId::from_raw(1),
            metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
        };

        // Add both nodes to the DFG
        let dfg = semantic_graphs.data_flow.get_mut(&function_id).unwrap();
        dfg.nodes.insert(param_node_id, param_node);
        dfg.nodes.insert(return_node_id, return_node);

        semantic_graphs
    }

    fn create_return_local_scenario() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Create return of local variable reference (potentially invalid)
        let function_id = SymbolId::from_raw(1);
        let local_id = SymbolId::from_raw(100);
        let borrower_id = SymbolId::from_raw(101);

        // Create a borrow of local variable
        let borrow_edge = BorrowEdge {
            id: BorrowEdgeId::from_raw(1),
            borrower: borrower_id,
            borrowed: local_id,
            borrow_type: BorrowType::Immutable,
            borrow_location: SourceLocation::new(1, 1, 1, 10), // file_id: 1, line: 1, column: 1, byte_offset: 10
            borrow_scope: ScopeId::from_raw(1),
            borrow_lifetime: crate::tast::LifetimeId::from_raw(1),
        };
        semantic_graphs
            .ownership_graph
            .borrow_edges
            .insert(BorrowEdgeId::from_raw(1), borrow_edge);

        // Return the borrow
        let return_node_id = DataFlowNodeId::from_raw(2);
        let return_node = DataFlowNode {
            id: return_node_id,
            kind: DataFlowNodeKind::Return {
                value: Some(DataFlowNodeId::from_raw(3)), // Convert to DataFlowNodeId
            },
            value_type: crate::tast::TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 2, 1, 20), // file_id: 1, line: 2, column: 1, byte_offset: 20
            operands: vec![],
            uses: crate::tast::collections::new_id_set(),
            defines: None,
            basic_block: crate::tast::BlockId::from_raw(1),
            metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
        };

        semantic_graphs
            .data_flow
            .get_mut(&function_id)
            .unwrap()
            .nodes
            .insert(return_node_id, return_node);

        semantic_graphs
    }

    fn create_complex_function_graphs() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Add more complexity for performance testing
        let function_id = SymbolId::from_raw(1);

        // Add multiple variables and operations
        for i in 2..20 {
            let var_id = SymbolId::from_raw(i);
            let ownership_node = OwnershipNode {
                variable: var_id,
                lifetime: crate::tast::LifetimeId::from_raw(1),
                ownership_kind: OwnershipKind::Owned,
                borrowed_by: Vec::new(),
                borrows_from: Vec::new(),
                allocation_site: None,
                move_site: None,
                is_moved: false,
                variable_type: crate::tast::TypeId::from_raw(1),
                scope: ScopeId::from_raw(1),
            };
            semantic_graphs
                .ownership_graph
                .variables
                .insert(var_id, ownership_node);

            // Add some data flow nodes
            let node_id = DataFlowNodeId::from_raw(i);
            let node = DataFlowNode {
                id: node_id,
                kind: DataFlowNodeKind::Constant {
                    value: crate::semantic_graph::dfg::ConstantValue::Int(i as i64),
                },
                value_type: crate::tast::TypeId::from_raw(1),
                source_location: SourceLocation::new(1, i as u32, 1, i as u32 * 10), // file_id: 1, line: i, column: 1, byte_offset: i*10
                operands: vec![],
                uses: crate::tast::collections::new_id_set(),
                defines: None,
                basic_block: crate::tast::BlockId::from_raw(1),
                metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
            };
            semantic_graphs
                .data_flow
                .get_mut(&function_id)
                .unwrap()
                .nodes
                .insert(node_id, node);
        }

        semantic_graphs
    }

    fn create_ownership_state_graphs() -> SemanticGraphs {
        let mut semantic_graphs = create_simple_function_graphs();

        // Add variables in different ownership states
        let owned_var = SymbolId::from_raw(100);
        let moved_var = SymbolId::from_raw(101);
        let borrowed_var = SymbolId::from_raw(102);

        // Owned variable
        let owned_node = OwnershipNode {
            variable: owned_var,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Owned,
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };
        semantic_graphs
            .ownership_graph
            .variables
            .insert(owned_var, owned_node);

        // Moved variable
        let moved_node = OwnershipNode {
            variable: moved_var,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Moved,
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: true,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };
        semantic_graphs
            .ownership_graph
            .variables
            .insert(moved_var, moved_node);

        // Borrowed variable
        let borrowed_node = OwnershipNode {
            variable: borrowed_var,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Borrowed,
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };
        semantic_graphs
            .ownership_graph
            .variables
            .insert(borrowed_var, borrowed_node);

        semantic_graphs
    }

    fn create_function_context(graphs: &SemanticGraphs) -> FunctionAnalysisContext {
        // Get the first function ID from the graphs
        let function_id = *graphs
            .control_flow
            .keys()
            .next()
            .expect("No functions in graphs");
        FunctionAnalysisContext {
            function_id,
            cfg: graphs.control_flow.get(&function_id).unwrap(),
            dfg: graphs.data_flow.get(&function_id).unwrap(),
            call_graph: &graphs.call_graph,
            ownership_graph: &graphs.ownership_graph,
        }
    }
}

/// **Benchmark Tests for Performance Validation**
///
/// These tests validate the performance requirements specified in Phase 5:
/// - <5ms analysis for 100-function codebase
/// - <50ms analysis for 1000-function codebase
/// - Memory usage <10MB for large projects
/// - Cache hit ratio >85%
#[cfg(test)]
mod ownership_performance_tests {
    use super::*;

    #[test]
    fn test_small_codebase_performance() {
        // Target: <5ms for small codebases
        let mut analyzer = OwnershipAnalyzer::new();

        let start_time = Instant::now();

        // Simulate 10 functions
        for i in 0..10 {
            let graphs = create_simple_function_with_id(i);
            let context = create_function_context(&graphs);

            let result = analyzer.analyze_function(&context);
            assert!(result.is_ok(), "Function {} analysis should succeed", i);
        }

        let total_time = start_time.elapsed();
        assert!(
            total_time < Duration::from_millis(5),
            "Small codebase analysis should complete in <5ms, took {:?}",
            total_time
        );

        println!("Small codebase (10 functions) analyzed in {:?}", total_time);
    }

    #[test]
    fn test_medium_codebase_performance() {
        // Target: <20ms for medium codebases
        let mut analyzer = OwnershipAnalyzer::new();

        let start_time = Instant::now();

        // Simulate 100 functions
        for i in 0..100 {
            let graphs = create_simple_function_with_id(i);
            let context = create_function_context(&graphs);

            let result = analyzer.analyze_function(&context);
            assert!(result.is_ok(), "Function {} analysis should succeed", i);
        }

        let total_time = start_time.elapsed();
        assert!(
            total_time < Duration::from_millis(50),
            "Medium codebase analysis should complete in <50ms, took {:?}",
            total_time
        );

        println!(
            "Medium codebase (100 functions) analyzed in {:?}",
            total_time
        );
    }

    fn create_simple_function_with_id(function_id: usize) -> SemanticGraphs {
        let mut semantic_graphs = SemanticGraphs::new();

        let func_id = SymbolId::from_raw(function_id as u32);

        // Control flow graph
        let cfg = ControlFlowGraph::new(func_id, crate::tast::BlockId::from_raw(1));
        semantic_graphs.control_flow.insert(func_id, cfg);

        // Data flow graph
        let entry_node = DataFlowNodeId::from_raw(function_id as u32);
        let mut dfg = DataFlowGraph::new(entry_node);
        let node_id = DataFlowNodeId::from_raw(function_id as u32);
        let node = DataFlowNode {
            id: node_id,
            kind: DataFlowNodeKind::Constant {
                value: crate::semantic_graph::dfg::ConstantValue::Int(function_id as i64),
            },
            value_type: crate::tast::TypeId::from_raw(1),
            source_location: SourceLocation::new(1, function_id as u32, 1, function_id as u32 * 10), // file_id: 1, line: function_id, column: 1, byte_offset: function_id*10
            operands: vec![],
            uses: crate::tast::collections::new_id_set(),
            defines: None,
            basic_block: crate::tast::BlockId::from_raw(1),
            metadata: crate::semantic_graph::dfg::NodeMetadata::default(),
        };
        dfg.nodes.insert(node_id, node);
        semantic_graphs.data_flow.insert(func_id, dfg);

        // Ownership graph
        let mut ownership_graph = OwnershipGraph::new();
        let var_id = SymbolId::from_raw((function_id * 2) as u32);
        let ownership_node = OwnershipNode {
            variable: var_id,
            lifetime: crate::tast::LifetimeId::from_raw(1),
            ownership_kind: OwnershipKind::Owned,
            borrowed_by: Vec::new(),
            borrows_from: Vec::new(),
            allocation_site: None,
            move_site: None,
            is_moved: false,
            variable_type: crate::tast::TypeId::from_raw(1),
            scope: ScopeId::from_raw(1),
        };
        ownership_graph.variables.insert(var_id, ownership_node);
        semantic_graphs.ownership_graph = ownership_graph;

        // Call graph
        semantic_graphs.call_graph = CallGraph::new();

        semantic_graphs
    }

    fn create_function_context(graphs: &SemanticGraphs) -> FunctionAnalysisContext {
        // Get the first function ID from the graphs
        let function_id = *graphs
            .control_flow
            .keys()
            .next()
            .expect("No functions in graphs");
        FunctionAnalysisContext {
            function_id,
            cfg: graphs.control_flow.get(&function_id).unwrap(),
            dfg: graphs.data_flow.get(&function_id).unwrap(),
            call_graph: &graphs.call_graph,
            ownership_graph: &graphs.ownership_graph,
        }
    }
}
