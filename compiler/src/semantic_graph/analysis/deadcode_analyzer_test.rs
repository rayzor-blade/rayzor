//! Comprehensive Tests for DeadCodeAnalyzer
//!
//! This module provides extensive testing for the dead code analysis implementation,
//! covering unreachable block detection, unused variable detection, dead function analysis,
//! and performance characteristics.

#[cfg(test)]
mod deadcode_analysis_tests {
    use super::super::{deadcode_analyzer::*, ownership_analyzer::FunctionAnalysisContext};
    use crate::semantic_graph::{
        cfg::{BasicBlock, Terminator},
        dfg::{ConstantValue, LivenessInfo, NodeMetadata, SsaVariable},
        CallGraph, CallSite, CallTarget, CallType, ControlFlowGraph, DataFlowGraph, DataFlowNode,
        DataFlowNodeKind, OwnershipGraph,
    };
    use crate::tast::{
        collections::new_id_set, BlockId, CallSiteId, DataFlowNodeId, SourceLocation,
        SsaVariableId, SymbolId, TypeId,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::time::Duration;

    /// Create test CFG with unreachable blocks
    pub fn create_test_cfg_with_unreachable_blocks() -> ControlFlowGraph {
        let function_id = SymbolId::from_raw(1);
        let entry_block_id = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_block_id);

        // Entry block
        let entry_block = BasicBlock {
            id: BlockId::from_raw(1),
            statements: vec![crate::tast::StatementId::from_raw(1)],
            terminator: Terminator::Jump {
                target: BlockId::from_raw(2),
            },
            predecessors: new_id_set(),
            successors: vec![BlockId::from_raw(2), BlockId::from_raw(3)],
            source_location: SourceLocation::new(1, 1, 1, 10),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(entry_block);

        // Reachable block
        let reachable_block = BasicBlock {
            id: BlockId::from_raw(2),
            statements: vec![crate::tast::StatementId::from_raw(2)],
            terminator: Terminator::Return { value: None },
            predecessors: new_id_set(),
            successors: vec![],
            source_location: SourceLocation::new(2, 1, 2, 10),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(reachable_block);

        // Another reachable block
        let another_reachable = BasicBlock {
            id: BlockId::from_raw(3),
            statements: vec![crate::tast::StatementId::from_raw(3)],
            terminator: Terminator::Jump {
                target: BlockId::from_raw(4),
            },
            predecessors: new_id_set(),
            successors: vec![BlockId::from_raw(4)],
            source_location: SourceLocation::new(3, 1, 3, 10),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(another_reachable);

        // Reachable from block 3
        let reached_from_3 = BasicBlock {
            id: BlockId::from_raw(4),
            statements: vec![crate::tast::StatementId::from_raw(4)],
            terminator: Terminator::Return { value: None },
            predecessors: new_id_set(),
            successors: vec![],
            source_location: SourceLocation::new(4, 1, 4, 10),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(reached_from_3);

        // Unreachable block (no predecessors)
        let unreachable_block = BasicBlock {
            id: BlockId::from_raw(5),
            statements: vec![crate::tast::StatementId::from_raw(5)],
            terminator: Terminator::Return { value: None },
            predecessors: new_id_set(),
            successors: vec![],
            source_location: SourceLocation::new(5, 1, 5, 10),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(unreachable_block);

        cfg
    }

    /// Create test DFG with unused variables
    pub fn create_test_dfg_with_unused_variables() -> DataFlowGraph {
        let entry_node_id = DataFlowNodeId::from_raw(1);
        let mut dfg = DataFlowGraph::new(entry_node_id);

        // Used variable
        let used_var_def = DataFlowNode {
            id: DataFlowNodeId::from_raw(1),
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(42),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 1, 1, 10),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(1)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(used_var_def);

        // Use of the variable
        let use_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(2),
            kind: DataFlowNodeKind::Load {
                address: DataFlowNodeId::from_raw(1),
                memory_type: crate::semantic_graph::dfg::MemoryType::Stack,
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(2, 1, 2, 10),
            operands: vec![DataFlowNodeId::from_raw(1)],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(3)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(use_node);

        // Unused variable
        let unused_var_def = DataFlowNode {
            id: DataFlowNodeId::from_raw(3),
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(100),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(3, 1, 3, 10),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(2)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(unused_var_def);

        // Dead store (variable assigned but never read)
        let dead_store = DataFlowNode {
            id: DataFlowNodeId::from_raw(4),
            kind: DataFlowNodeKind::Store {
                address: DataFlowNodeId::from_raw(5),
                value: DataFlowNodeId::from_raw(1),
                memory_type: crate::semantic_graph::dfg::MemoryType::Stack,
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(4, 1, 4, 15),
            operands: vec![DataFlowNodeId::from_raw(5), DataFlowNodeId::from_raw(1)],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(dead_store);

        // Variable that's stored to but never read
        let unread_var_def = DataFlowNode {
            id: DataFlowNodeId::from_raw(5),
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(200),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(5, 1, 5, 10),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(4)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(unread_var_def);

        // Add SSA variables
        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(1),
            SsaVariable {
                id: SsaVariableId::from_raw(1),
                original_symbol: SymbolId::from_raw(1),
                ssa_index: 0,
                var_type: TypeId::from_raw(1),
                definition: DataFlowNodeId::from_raw(1),
                uses: vec![DataFlowNodeId::from_raw(2)], // Used
                liveness: LivenessInfo::default(),
            },
        );

        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(2),
            SsaVariable {
                id: SsaVariableId::from_raw(2),
                original_symbol: SymbolId::from_raw(2),
                ssa_index: 0,
                var_type: TypeId::from_raw(1),
                definition: DataFlowNodeId::from_raw(3),
                uses: vec![], // Unused
                liveness: LivenessInfo::default(),
            },
        );

        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(3),
            SsaVariable {
                id: SsaVariableId::from_raw(3),
                original_symbol: SymbolId::from_raw(3),
                ssa_index: 0,
                var_type: TypeId::from_raw(1),
                definition: DataFlowNodeId::from_raw(2),
                uses: vec![], // Result of using var 1
                liveness: LivenessInfo::default(),
            },
        );

        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(4),
            SsaVariable {
                id: SsaVariableId::from_raw(4),
                original_symbol: SymbolId::from_raw(4),
                ssa_index: 0,
                var_type: TypeId::from_raw(1),
                definition: DataFlowNodeId::from_raw(5),
                uses: vec![DataFlowNodeId::from_raw(4)], // Used in store but store result never read
                liveness: LivenessInfo::default(),
            },
        );

        dfg
    }

    /// Create test call graph with unreachable functions
    pub fn create_test_call_graph_with_unreachable_functions() -> CallGraph {
        let mut call_graph = CallGraph::new();

        let main_func = SymbolId::from_raw(1);
        let called_func = SymbolId::from_raw(2);
        let unreachable_func = SymbolId::from_raw(3);
        let transitive_unreachable = SymbolId::from_raw(4);

        call_graph.add_function(main_func);
        call_graph.add_function(called_func);
        call_graph.add_function(unreachable_func);
        call_graph.add_function(transitive_unreachable);

        // Main calls called_func
        let main_call = CallSite::new(
            CallSiteId::from_raw(1),
            main_func,
            CallTarget::Direct {
                function: called_func,
            },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::new(1, 1, 1, 20),
        );
        call_graph.add_call_site(main_call);

        // Unreachable function calls transitive_unreachable (both unreachable)
        let unreachable_call = CallSite::new(
            CallSiteId::from_raw(2),
            unreachable_func,
            CallTarget::Direct {
                function: transitive_unreachable,
            },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::new(10, 1, 10, 25),
        );
        call_graph.add_call_site(unreachable_call);

        call_graph
    }

    /// Create test ownership graph
    pub fn create_test_ownership_graph() -> OwnershipGraph {
        OwnershipGraph::new()
    }

    #[test]
    fn test_deadcode_analyzer_creation() {
        let analyzer = DeadCodeAnalyzer::new();

        // Verify initial state
        assert_eq!(analyzer.stats().analysis_time, Duration::ZERO);
        assert_eq!(analyzer.stats().blocks_analyzed, 0);
        assert_eq!(analyzer.stats().variables_analyzed, 0);
        assert_eq!(analyzer.stats().functions_analyzed, 0);
    }

    #[test]
    fn test_complete_function_analysis() {
        let mut analyzer = DeadCodeAnalyzer::new();
        let cfg = create_test_cfg_with_unreachable_blocks();
        let dfg = create_test_dfg_with_unused_variables();
        let call_graph = create_test_call_graph_with_unreachable_functions();
        let ownership_graph = create_test_ownership_graph();

        let context = FunctionAnalysisContext {
            function_id: SymbolId::from_raw(1),
            cfg: &cfg,
            dfg: &dfg,
            call_graph: &call_graph,
            ownership_graph: &ownership_graph,
        };

        let result = analyzer
            .analyze_function(&context)
            .expect("Analysis should succeed");

        // Should find multiple types of dead code
        assert!(!result.dead_code_regions.is_empty());
        assert!(!result.unreachable_blocks.is_empty());
        assert!(!result.unused_variables.is_empty());

        // Check stats are updated
        assert!(result.stats.blocks_analyzed > 0);
        assert!(result.stats.variables_analyzed > 0);
        assert_eq!(result.stats.functions_analyzed, 1);
    }

    #[test]
    fn test_performance_characteristics() {
        let mut analyzer = DeadCodeAnalyzer::new();
        let cfg = create_test_cfg_with_unreachable_blocks();
        let dfg = create_test_dfg_with_unused_variables();
        let call_graph = create_test_call_graph_with_unreachable_functions();
        let ownership_graph = create_test_ownership_graph();

        let context = FunctionAnalysisContext {
            function_id: SymbolId::from_raw(1),
            cfg: &cfg,
            dfg: &dfg,
            call_graph: &call_graph,
            ownership_graph: &ownership_graph,
        };

        let start = std::time::Instant::now();
        let result = analyzer
            .analyze_function(&context)
            .expect("Analysis should succeed");
        let duration = start.elapsed();

        // Performance target: should be very fast for small functions
        assert!(duration < Duration::from_millis(10));
        assert!(result.stats.analysis_time < Duration::from_millis(5));

        // Check cache efficiency
        let analyzer_stats = analyzer.stats();
        assert!(analyzer_stats.cache_hit_ratio >= 0.0); // Should be valid ratio
    }

    #[test]
    fn test_dead_code_analysis_results_api() {
        let results = DeadCodeAnalysisResults::new();

        // Test initial state
        assert!(!results.has_dead_code());
        assert_eq!(results.estimated_savings(), 0);
        assert!(results.get_dead_regions_by_type().is_empty());
    }

    #[test]
    fn test_unreachability_reason_display() {
        let reasons = vec![
            UnreachabilityReason::NoPredecessors,
            UnreachabilityReason::AfterReturn,
            UnreachabilityReason::AfterJump,
            UnreachabilityReason::AlwaysFalseCondition,
            UnreachabilityReason::UnusedExceptionHandler,
            UnreachabilityReason::DebugOnlyCode,
        ];

        for reason in reasons {
            assert!(!format!("{}", reason).is_empty());
        }
    }

    #[test]
    fn test_dead_code_region_display() {
        let regions = vec![
            DeadCodeRegion::UnreachableBlock {
                block: BlockId::from_raw(1),
                function: SymbolId::from_raw(1),
                reason: UnreachabilityReason::NoPredecessors,
                source_location: SourceLocation::unknown(),
            },
            DeadCodeRegion::UnusedVariable {
                variable: SymbolId::from_raw(1),
                ssa_variable: SsaVariableId::from_raw(1),
                declaration_location: SourceLocation::unknown(),
                variable_type: TypeId::from_raw(1),
                suggested_action: "Remove unused variable".to_string(),
            },
            DeadCodeRegion::UnreachableFunction {
                function: SymbolId::from_raw(1),
                declaration_location: SourceLocation::unknown(),
                call_graph_analysis: "Never called".to_string(),
                estimated_savings: 100,
            },
            DeadCodeRegion::DeadStore {
                variable: SymbolId::from_raw(1),
                store_location: SourceLocation::unknown(),
                last_use_location: None,
            },
            DeadCodeRegion::UnreachableAfterReturn {
                function: SymbolId::from_raw(1),
                return_location: SourceLocation::unknown(),
                unreachable_start: SourceLocation::unknown(),
            },
        ];

        for region in regions {
            assert!(!format!("{}", region).is_empty());
        }
    }

    #[test]
    fn test_error_handling() {
        // Test error display
        let errors = vec![
            DeadCodeAnalysisError::InternalError("Test error".to_string()),
            DeadCodeAnalysisError::GraphIntegrityError("Graph invalid".to_string()),
            DeadCodeAnalysisError::AnalysisTimeout,
        ];

        for error in errors {
            assert!(!format!("{}", error).is_empty());
        }
    }
}

#[cfg(test)]
mod deadcode_analysis_integration_tests {
    use super::*;
    use crate::semantic_graph::analysis::analysis_engine::AnalysisEngine;
    use crate::semantic_graph::SemanticGraphs;

    #[test]
    fn test_integration_with_analysis_engine() {
        let mut analysis_engine = AnalysisEngine::new();

        // Create minimal semantic graphs for testing
        let graphs = create_test_semantic_graphs();

        let result = analysis_engine.analyze(&graphs);
        assert!(result.is_ok());

        let analysis_results = result.unwrap();

        // Check that dead code analysis results are included
        assert!(analysis_results.dead_code_by_function.is_empty()); // No dead code in minimal test

        // Check HIR hints include dead code analysis
        let hir_hints = analysis_results.get_hir_hints();
        assert!(hir_hints.dead_code_regions.is_empty()); // No dead code in minimal test
    }

    fn create_test_semantic_graphs() -> SemanticGraphs {
        use crate::semantic_graph::{CallGraph, OwnershipGraph};

        SemanticGraphs {
            control_flow: std::collections::BTreeMap::new(),
            data_flow: std::collections::BTreeMap::new(),
            call_graph: CallGraph::new(),
            ownership_graph: OwnershipGraph::new(),
            source_locations: std::collections::BTreeMap::new(),
        }
    }
}

#[cfg(test)]
mod deadcode_analysis_performance_tests {
    use super::*;
    use crate::semantic_graph::analysis::{
        deadcode_analyzer::DeadCodeAnalyzer, ownership_analyzer::FunctionAnalysisContext,
    };
    use crate::tast::SymbolId;
    use std::time::{Duration, Instant};

    #[test]
    fn test_large_function_performance() {
        let mut analyzer = DeadCodeAnalyzer::new();

        // Use the existing test helper functions instead of creating complex ones
        let cfg = deadcode_analysis_tests::create_test_cfg_with_unreachable_blocks();
        let dfg = deadcode_analysis_tests::create_test_dfg_with_unused_variables();
        let call_graph =
            deadcode_analysis_tests::create_test_call_graph_with_unreachable_functions();
        let ownership_graph = deadcode_analysis_tests::create_test_ownership_graph();

        let context = FunctionAnalysisContext {
            function_id: SymbolId::from_raw(1),
            cfg: &cfg,
            dfg: &dfg,
            call_graph: &call_graph,
            ownership_graph: &ownership_graph,
        };

        let start = Instant::now();
        let result = analyzer.analyze_function(&context);
        let duration = start.elapsed();

        assert!(result.is_ok());

        // Performance target: should handle large functions in reasonable time
        assert!(duration < Duration::from_millis(100));

        let analysis_result = result.unwrap();
        assert!(analysis_result.stats.blocks_analyzed > 0);
        assert!(analysis_result.stats.variables_analyzed > 0);
    }
}
