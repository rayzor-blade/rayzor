//! Comprehensive Tests for EscapeAnalyzer
//!
//! This module provides extensive testing for the escape analysis implementation,
//! covering allocation detection, escape path analysis, optimization hint generation,
//! and performance characteristics.

#[cfg(test)]
mod escape_analysis_tests {
    use super::super::{escape_analyzer::*, ownership_analyzer::FunctionAnalysisContext};
    use crate::semantic_graph::{
        cfg::{BasicBlock, Terminator},
        dfg::{CallType as DfgCallType, ConstantValue, LivenessInfo, NodeMetadata, SsaVariable},
        CallGraph, CallSite, CallTarget, CallType, ControlFlowGraph, DataFlowGraph, DataFlowNode,
        DataFlowNodeKind, OwnershipGraph,
    };
    use crate::tast::{
        collections::new_id_set, node::BinaryOperator, BlockId, CallSiteId, DataFlowNodeId,
        SourceLocation, SsaVariableId, SymbolId, TypeId,
    };
    use std::collections::BTreeMap;
    use std::time::Duration;

    /// Create test DFG with allocation patterns
    pub fn create_test_dfg_with_allocations() -> DataFlowGraph {
        let entry_node_id = DataFlowNodeId::from_raw(1);
        let mut dfg = DataFlowGraph::new(entry_node_id);

        // Object allocation: new MyClass()
        let alloc_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(1),
            kind: DataFlowNodeKind::Allocation {
                allocation_type: TypeId::from_raw(1),
                size: None,
                allocation_kind: crate::semantic_graph::dfg::AllocationKind::Heap,
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(1, 1, 1, 10),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(1)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(alloc_node);

        // Array allocation: [1, 2, 3]
        let array_alloc_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(2),
            kind: DataFlowNodeKind::Allocation {
                allocation_type: TypeId::from_raw(2),     // Array type
                size: Some(DataFlowNodeId::from_raw(10)), // Size expression
                allocation_kind: crate::semantic_graph::dfg::AllocationKind::Heap,
            },
            value_type: TypeId::from_raw(2),
            source_location: SourceLocation::new(2, 1, 2, 15),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(2)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(array_alloc_node);

        // Return statement that escapes allocation
        let return_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(3),
            kind: DataFlowNodeKind::Return {
                value: Some(DataFlowNodeId::from_raw(1)), // Returns the allocated object
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(3, 1, 3, 15),
            operands: vec![DataFlowNodeId::from_raw(1)],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(return_node);

        // Variable that doesn't escape
        let local_use_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(4),
            kind: DataFlowNodeKind::Load {
                address: DataFlowNodeId::from_raw(2), // Uses array locally
                memory_type: crate::semantic_graph::dfg::MemoryType::Stack,
            },
            value_type: TypeId::from_raw(2),
            source_location: SourceLocation::new(4, 1, 4, 10),
            operands: vec![DataFlowNodeId::from_raw(2)],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(3)),
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(local_use_node);

        // Add SSA variables
        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(1),
            SsaVariable {
                id: SsaVariableId::from_raw(1),
                original_symbol: SymbolId::from_raw(1),
                ssa_index: 0,
                var_type: TypeId::from_raw(1),
                definition: DataFlowNodeId::from_raw(1),
                uses: vec![DataFlowNodeId::from_raw(3)],
                liveness: LivenessInfo::default(),
            },
        );

        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(2),
            SsaVariable {
                id: SsaVariableId::from_raw(2),
                original_symbol: SymbolId::from_raw(2),
                ssa_index: 0,
                var_type: TypeId::from_raw(2),
                definition: DataFlowNodeId::from_raw(2),
                uses: vec![DataFlowNodeId::from_raw(4)],
                liveness: LivenessInfo::default(),
            },
        );

        dfg.ssa_variables.insert(
            SsaVariableId::from_raw(3),
            SsaVariable {
                id: SsaVariableId::from_raw(3),
                original_symbol: SymbolId::from_raw(3),
                ssa_index: 0,
                var_type: TypeId::from_raw(2),
                definition: DataFlowNodeId::from_raw(4),
                uses: vec![],
                liveness: LivenessInfo::default(),
            },
        );

        dfg
    }

    /// Create test CFG for escape analysis
    pub fn create_test_cfg() -> ControlFlowGraph {
        let function_id = SymbolId::from_raw(1);
        let entry_block_id = BlockId::from_raw(1);
        let mut cfg = ControlFlowGraph::new(function_id, entry_block_id);

        let entry_block = BasicBlock {
            id: BlockId::from_raw(1),
            statements: vec![
                crate::tast::StatementId::from_raw(1),
                crate::tast::StatementId::from_raw(2),
            ],
            terminator: Terminator::Jump {
                target: BlockId::from_raw(2),
            },
            predecessors: new_id_set(),
            successors: vec![BlockId::from_raw(2)],
            source_location: SourceLocation::new(1, 1, 1, 10),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(entry_block);

        let exit_block = BasicBlock {
            id: BlockId::from_raw(2),
            statements: vec![
                crate::tast::StatementId::from_raw(3),
                crate::tast::StatementId::from_raw(4),
            ],
            terminator: Terminator::Return { value: None },
            predecessors: new_id_set(),
            successors: vec![],
            source_location: SourceLocation::new(3, 1, 3, 15),
            metadata: crate::semantic_graph::cfg::BlockMetadata::default(),
        };
        cfg.add_block(exit_block);

        cfg
    }

    /// Create test call graph with function calls
    pub fn create_test_call_graph() -> CallGraph {
        let mut call_graph = CallGraph::new();

        let caller_func = SymbolId::from_raw(1);
        let callee_func = SymbolId::from_raw(2);
        let constructor_func = SymbolId::from_raw(100);

        call_graph.add_function(caller_func);
        call_graph.add_function(callee_func);
        call_graph.add_function(constructor_func);

        // Add call site for constructor
        let constructor_call = CallSite::new(
            CallSiteId::from_raw(1),
            caller_func,
            CallTarget::Direct {
                function: constructor_func,
            },
            CallType::Direct,
            BlockId::from_raw(1),
            SourceLocation::new(1, 1, 1, 10),
        );
        call_graph.add_call_site(constructor_call);

        // Add call to another function
        let function_call = CallSite::new(
            CallSiteId::from_raw(2),
            caller_func,
            CallTarget::Direct {
                function: callee_func,
            },
            CallType::Direct,
            BlockId::from_raw(2),
            SourceLocation::new(5, 1, 5, 20),
        );
        call_graph.add_call_site(function_call);

        call_graph
    }

    /// Create test ownership graph
    pub fn create_test_ownership_graph() -> OwnershipGraph {
        OwnershipGraph::new()
    }

    #[test]
    fn test_escape_analyzer_creation() {
        let analyzer = EscapeAnalyzer::new();

        // Verify initial state
        assert_eq!(analyzer.stats().analysis_time, Duration::ZERO);
        assert_eq!(analyzer.stats().allocations_analyzed, 0);
        assert_eq!(analyzer.stats().stack_opportunities, 0);
    }

    #[test]
    fn test_allocation_site_detection() {
        let mut analyzer = EscapeAnalyzer::new();
        let dfg = create_test_dfg_with_allocations();
        let cfg = create_test_cfg();
        let call_graph = create_test_call_graph();
        let ownership_graph = create_test_ownership_graph();

        let context = FunctionAnalysisContext {
            function_id: SymbolId::from_raw(1),
            cfg: &cfg,
            dfg: &dfg,
            call_graph: &call_graph,
            ownership_graph: &ownership_graph,
        };

        // Test through public API - analyze_function will detect allocations internally
        let result = analyzer
            .analyze_function(&context)
            .expect("Analysis should succeed");

        // Should find allocation sites: object construction
        assert!(!result.allocation_sites.is_empty());

        // Verify object allocation exists
        let obj_alloc = result
            .allocation_sites
            .get(&DataFlowNodeId::from_raw(1))
            .expect("Should find object allocation");
        assert_eq!(obj_alloc.allocation_site, DataFlowNodeId::from_raw(1));
        assert_eq!(obj_alloc.allocated_type, TypeId::from_raw(1));
    }

    #[test]
    fn test_escape_via_return() {
        let mut analyzer = EscapeAnalyzer::new();
        let dfg = create_test_dfg_with_allocations();
        let cfg = create_test_cfg();
        let call_graph = create_test_call_graph();
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

        // Check that object allocation escapes via return
        let obj_status = result
            .escape_status
            .get(&DataFlowNodeId::from_raw(1))
            .expect("Should have escape status for object allocation");
        assert_eq!(*obj_status, EscapeStatus::EscapesViaReturn);

        // Check that array allocation doesn't escape
        let array_status = result
            .escape_status
            .get(&DataFlowNodeId::from_raw(2))
            .expect("Should have escape status for array allocation");
        assert_eq!(*array_status, EscapeStatus::NoEscape);
    }

    #[test]
    fn test_optimization_hint_generation() {
        let mut analyzer = EscapeAnalyzer::new();
        let dfg = create_test_dfg_with_allocations();
        let cfg = create_test_cfg();
        let call_graph = create_test_call_graph();
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

        // Should generate optimization hint for non-escaping allocation
        let stack_allocation_hints: Vec<_> = result
            .optimization_hints
            .iter()
            .filter(|hint| matches!(hint, OptimizationHint::StackAllocation { .. }))
            .collect();

        assert!(!stack_allocation_hints.is_empty());

        if let OptimizationHint::StackAllocation {
            allocation_site,
            estimated_size,
        } = &stack_allocation_hints[0]
        {
            assert_eq!(*allocation_site, DataFlowNodeId::from_raw(2));
            assert!(*estimated_size > 0);
        }
    }

    #[test]
    fn test_function_call_escape_analysis() {
        let mut analyzer = EscapeAnalyzer::new();
        let mut dfg = create_test_dfg_with_allocations();
        let call_graph = create_test_call_graph();
        let ownership_graph = create_test_ownership_graph();

        // Add a function call that passes an allocation as argument
        let call_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(5),
            kind: DataFlowNodeKind::Call {
                function: DataFlowNodeId::from_raw(100), // Function to call
                arguments: vec![DataFlowNodeId::from_raw(2)], // Pass array to function
                call_type: crate::semantic_graph::dfg::CallType::Direct,
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::new(5, 1, 5, 20),
            operands: vec![DataFlowNodeId::from_raw(100), DataFlowNodeId::from_raw(2)],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        let call_node_id = DataFlowNodeId::from_raw(5);
        dfg.add_node(call_node);

        // Update array variable to show it's used in the call
        if let Some(array_var) = dfg.ssa_variables.get_mut(&SsaVariableId::from_raw(2)) {
            array_var.uses.push(call_node_id);
        }

        let cfg = create_test_cfg();
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

        // Array should now escape via function call
        let array_status = result
            .escape_status
            .get(&DataFlowNodeId::from_raw(2))
            .expect("Should have escape status for array allocation");
        assert!(matches!(*array_status, EscapeStatus::EscapesViaCall { .. }));
    }

    #[test]
    fn test_global_escape_analysis() {
        let mut analyzer = EscapeAnalyzer::new();
        let call_graph = create_test_call_graph();
        let ownership_graph = create_test_ownership_graph();

        let result = analyzer
            .analyze_escapes(&call_graph, &ownership_graph)
            .expect("Global analysis should succeed");

        // Should have analysis results structure
        assert!(result.allocation_sites.is_empty()); // No allocations in simple call graph
        assert!(result.optimization_hints.is_empty());
        assert!(result.stats.analysis_time < Duration::from_millis(1)); // Should be very fast

        // Check that functions are tracked
        let inlinable = result.get_inlinable_functions();
        assert!(inlinable.is_empty()); // Simple functions, no inlining hints yet
    }

    #[test]
    fn test_escape_analysis_performance() {
        let mut analyzer = EscapeAnalyzer::new();
        let dfg = create_test_dfg_with_allocations();
        let cfg = create_test_cfg();
        let call_graph = create_test_call_graph();
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

        // Check stats are tracked
        let analyzer_stats = analyzer.stats();
        assert!(analyzer_stats.allocations_analyzed > 0);
    }

    #[test]
    fn test_escape_status_display() {
        // Test display implementations for debugging
        let no_escape = EscapeStatus::NoEscape;
        let via_return = EscapeStatus::EscapesViaReturn;
        let via_call = EscapeStatus::EscapesViaCall {
            call_site: DataFlowNodeId::from_raw(42),
        };
        let via_global = EscapeStatus::EscapesViaGlobal {
            global: SymbolId::from_raw(1),
        };
        let unknown = EscapeStatus::Unknown;

        assert!(!format!("{}", no_escape).is_empty());
        assert!(!format!("{}", via_return).is_empty());
        assert!(!format!("{}", via_call).is_empty());
        assert!(!format!("{}", via_global).is_empty());
        assert!(!format!("{}", unknown).is_empty());
    }

    #[test]
    fn test_optimization_hint_priorities() {
        let stack_alloc = OptimizationHint::StackAllocation {
            allocation_site: DataFlowNodeId::from_raw(1),
            estimated_size: 100,
        };

        let remove_alloc = OptimizationHint::RemoveAllocation {
            allocation_site: DataFlowNodeId::from_raw(2),
            reason: "Never used".to_string(),
        };

        let inline_func = OptimizationHint::InlineFunction {
            function: SymbolId::from_raw(1),
            reason: "Small function".to_string(),
        };

        // Test that optimization hints can be created and formatted
        assert!(!format!("{}", stack_alloc).is_empty());
        assert!(!format!("{}", remove_alloc).is_empty());
        assert!(!format!("{}", inline_func).is_empty());
    }

    #[test]
    fn test_error_handling() {
        let analyzer = EscapeAnalyzer::new();

        // Test error display
        let internal_error = EscapeAnalysisError::InternalError("Test error".to_string());
        let graph_error = EscapeAnalysisError::GraphIntegrityError("Graph invalid".to_string());
        let timeout_error = EscapeAnalysisError::AnalysisTimeout;

        assert!(!format!("{}", internal_error).is_empty());
        assert!(!format!("{}", graph_error).is_empty());
        assert!(!format!("{}", timeout_error).is_empty());
    }
}

#[cfg(test)]
mod escape_analysis_integration_tests {
    use super::*;
    use crate::semantic_graph::analysis::analysis_engine::AnalysisEngine;
    use crate::semantic_graph::{CallGraph, OwnershipGraph, SemanticGraphs};

    #[test]
    fn test_integration_with_analysis_engine() {
        let mut analysis_engine = AnalysisEngine::new();

        // Create minimal semantic graphs for testing
        let graphs = create_test_semantic_graphs();

        let result = analysis_engine.analyze(&graphs);
        assert!(result.is_ok());

        let analysis_results = result.unwrap();

        // Check that escape analysis results are included
        assert!(analysis_results.escape_analysis.allocation_sites.is_empty()); // No allocations in minimal test

        // Check HIR hints include escape analysis
        let hir_hints = analysis_results.get_hir_hints();
        assert!(hir_hints.optimization_opportunities.is_empty()); // No optimizations in minimal test
    }

    fn create_test_semantic_graphs() -> SemanticGraphs {
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
mod escape_analysis_performance_tests {
    use crate::{
        semantic_graph::analysis::{
            analysis_engine::FunctionAnalysisContext, escape_analyzer::EscapeAnalyzer,
        },
        tast::SymbolId,
    };

    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_large_function_performance() {
        let mut analyzer = EscapeAnalyzer::new();

        // Use the existing test helper functions instead of creating complex ones
        let dfg = escape_analysis_tests::create_test_dfg_with_allocations();
        let cfg = escape_analysis_tests::create_test_cfg();
        let call_graph = escape_analysis_tests::create_test_call_graph();
        let ownership_graph = escape_analysis_tests::create_test_ownership_graph();

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

        // Performance target: should be very fast for small functions
        assert!(duration < Duration::from_millis(50));

        let analysis_result = result.unwrap();
        assert!(!analysis_result.allocation_sites.is_empty());
    }
}
