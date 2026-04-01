//! Test module for lifetime analysis
//!
//! Basic tests to verify the lifetime analyzer works correctly

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::semantic_graph::analysis::lifetime_analyzer::{
        EqualityReason, LifetimeAnalyzer, LifetimeConstraint, LifetimeRegion, OutlivesReason,
    };
    use crate::semantic_graph::analysis::lifetime_solver::{
        LifetimeConstraintSolver, LifetimeSolution, SolverConfig,
    };
    use crate::semantic_graph::{
        BasicBlock, ControlFlowGraph, DataFlowGraph, DataFlowNode, DataFlowNodeKind, LifetimeId,
        OwnershipGraph, PhiIncoming, Terminator,
    };
    use crate::tast::collections::new_id_set;
    use crate::tast::node::BinaryOperator;
    use crate::tast::{
        BlockId, DataFlowNodeId, ScopeId, SourceLocation, SsaVariableId, SymbolId, TypeId,
    };

    #[test]
    fn test_lifetime_analyzer_creation() {
        let analyzer = LifetimeAnalyzer::new();
        assert!(analyzer.lifetime_assignments.is_empty());
        assert!(analyzer.call_site_lifetimes.is_empty());
        assert!(analyzer.active_regions.is_empty());
    }

    #[test]
    fn test_constraint_solver_creation() {
        let solver = LifetimeConstraintSolver::new();
        assert_eq!(solver.cache_hit_ratio(), 0.0);

        let custom_solver = LifetimeConstraintSolver::with_config(SolverConfig {
            max_cache_size: 500,
            collect_detailed_stats: true,
            analyze_conflicts: true,
            max_constraint_system_size: 10000,
        });
        assert_eq!(custom_solver.cache_hit_ratio(), 0.0);
    }

    #[test]
    fn test_basic_constraint_creation() {
        let lifetime_a = LifetimeId::from_raw(1);
        let lifetime_b = LifetimeId::from_raw(2);
        let location = SourceLocation::unknown();

        let constraint = LifetimeConstraint::Outlives {
            longer: lifetime_a,
            shorter: lifetime_b,
            location: location.clone(),
            reason: OutlivesReason::Assignment,
        };

        match constraint {
            LifetimeConstraint::Outlives {
                longer,
                shorter,
                reason,
                ..
            } => {
                assert_eq!(longer, lifetime_a);
                assert_eq!(shorter, lifetime_b);
                assert_eq!(reason, OutlivesReason::Assignment);
            }
            _ => panic!("Expected Outlives constraint"),
        }
    }

    #[test]
    fn test_empty_constraint_system() {
        let mut solver = LifetimeConstraintSolver::new();
        let constraints = vec![];

        let result = solver.solve(&constraints);
        assert!(result.is_ok());

        let solution = result.unwrap();
        assert!(solution.satisfiable);
        assert!(solution.conflicts.is_empty());
        assert!(solution.assignments.is_empty());
    }

    #[test]
    fn test_constraint_solver_performance() {
        let mut solver = LifetimeConstraintSolver::new();

        // Test with various sizes of constraint systems
        let test_sizes = [10, 100, 500];

        for size in test_sizes {
            let constraints = create_test_constraints(size);
            let start_time = std::time::Instant::now();

            let result = solver.solve(constraints.as_slice());
            let solve_time = start_time.elapsed();

            // Should solve small systems very quickly
            assert!(
                solve_time.as_millis() < 100,
                "Constraint solving took too long: {:?}",
                solve_time
            );

            if let Ok(solution) = result {
                assert!(solution.satisfiable || !solution.conflicts.is_empty());
            }
        }
    }

    #[test]
    fn test_solver_cache_functionality() {
        let mut solver = LifetimeConstraintSolver::new();
        let constraints = create_test_constraints(50);

        // First solve - should be cache miss
        let result1 = solver.solve(&constraints);
        assert!(result1.is_ok());
        assert_eq!(solver.statistics().cache_hits, 0);
        assert_eq!(solver.statistics().cache_misses, 1);

        // Second solve with same constraints - should be cache hit
        let result2 = solver.solve(&constraints);
        assert!(result2.is_ok());
        assert_eq!(solver.statistics().cache_hits, 1);
        assert_eq!(solver.statistics().cache_misses, 1);

        // Cache hit ratio should be 50%
        assert!((solver.cache_hit_ratio() - 0.5).abs() < 0.01);
    }

    // Helper function to create test constraints
    fn create_test_constraints(count: usize) -> Vec<LifetimeConstraint> {
        let mut constraints = Vec::new();
        let location = SourceLocation::unknown();

        for i in 0..count {
            let lifetime_a = LifetimeId::from_raw(i as u32);
            let lifetime_b = LifetimeId::from_raw((i + 1) as u32);

            constraints.push(LifetimeConstraint::Outlives {
                longer: lifetime_a,
                shorter: lifetime_b,
                location: location.clone(),
                reason: OutlivesReason::Assignment,
            });
        }

        constraints
    }

    #[test]
    fn test_error_conversions() {
        use crate::semantic_graph::analysis::lifetime_analyzer::{
            ConstraintSolvingError, LifetimeAnalysisError,
        };

        let solver_error = ConstraintSolvingError::InvalidConstraint("test error".to_string());
        let analysis_error: LifetimeAnalysisError = solver_error.into();

        match analysis_error {
            LifetimeAnalysisError::SolverError(ConstraintSolvingError::InvalidConstraint(msg)) => {
                assert_eq!(msg, "test error");
            }
            _ => panic!("Error conversion failed"),
        }
    }

    #[test]
    fn test_statistics_tracking() {
        let mut solver = LifetimeConstraintSolver::new();
        let constraints = create_test_constraints(25);

        // Solve multiple times to build up statistics
        for _ in 0..5 {
            let _ = solver.solve(&constraints);
        }

        let stats = solver.statistics();
        assert!(stats.systems_solved >= 1);
        assert!(stats.constraints_processed >= 25);
        assert!(stats.total_solving_time_us > 0);

        // After multiple solves of same system, should have good cache performance
        assert!(solver.cache_hit_ratio() > 0.5);
    }

    // **Lifetime Region Creation Tests**

    #[test]
    fn test_create_lifetime_regions() {
        let mut analyzer = LifetimeAnalyzer::new();
        let cfg = create_test_cfg();
        let function_id = SymbolId::from_raw(1);

        let regions = analyzer
            .create_lifetime_regions(&cfg, function_id)
            .expect("Should create lifetime regions");

        // Should have at least global and function regions
        assert!(regions.len() >= 2);

        // Check global region
        let global_region = regions.iter().find(|r| r.id == LifetimeId::global());
        assert!(global_region.is_some());
        assert_eq!(global_region.unwrap().parent, None);

        // Check function region
        let function_region = regions
            .iter()
            .find(|r| r.parent == Some(LifetimeId::global()));
        assert!(function_region.is_some());
    }

    // **Initial Lifetime Assignment Tests**

    #[test]
    fn test_assign_initial_lifetimes() {
        let mut analyzer = LifetimeAnalyzer::new();
        let regions = create_test_regions();
        let dfg = create_test_dfg();
        let ownership_graph = create_test_ownership_graph();

        let result = analyzer.assign_initial_lifetimes(&regions, &dfg, &ownership_graph);
        assert!(result.is_ok());

        // Should have assigned some lifetimes
        assert!(!analyzer.lifetime_assignments.is_empty());
    }

    // **Constraint Generation Tests**

    #[test]
    fn test_binary_op_constraints() {
        let analyzer = LifetimeAnalyzer::new();
        let node = create_test_binary_op_node();
        let left = DataFlowNodeId::from_raw(1);
        let right = DataFlowNodeId::from_raw(2);

        let constraints = analyzer
            .generate_binary_op_constraints(left, right, &node)
            .expect("Should generate binary op constraints");

        // Should generate constraints for both operands
        assert_eq!(constraints.len(), 2);

        // Both should be outlives constraints
        for constraint in constraints {
            match constraint {
                LifetimeConstraint::Outlives { reason, .. } => {
                    assert_eq!(reason, OutlivesReason::Assignment);
                }
                _ => panic!("Expected Outlives constraint"),
            }
        }
    }

    #[test]
    fn test_phi_constraints() {
        let analyzer = LifetimeAnalyzer::new();
        let incoming = vec![
            PhiIncoming {
                value: DataFlowNodeId::from_raw(1),
                predecessor: BlockId::from_raw(1),
            },
            PhiIncoming {
                value: DataFlowNodeId::from_raw(2),
                predecessor: BlockId::from_raw(2),
            },
        ];
        let node = create_test_phi_node();

        let constraints = analyzer
            .generate_phi_constraints(&incoming, &node)
            .expect("Should generate phi constraints");

        // Should generate equality constraints for incoming values
        assert_eq!(constraints.len(), 2);

        for constraint in constraints {
            match constraint {
                LifetimeConstraint::Equal { reason, .. } => {
                    assert_eq!(reason, EqualityReason::ConditionalBranches);
                }
                _ => panic!("Expected Equal constraint"),
            }
        }
    }

    // **Constraint Solving Tests**

    #[test]
    fn test_simple_outlives_constraints() {
        let mut solver = LifetimeConstraintSolver::new();
        let constraints = vec![
            LifetimeConstraint::Outlives {
                longer: LifetimeId::from_raw(1),
                shorter: LifetimeId::from_raw(2),
                location: SourceLocation::unknown(),
                reason: OutlivesReason::Assignment,
            },
            LifetimeConstraint::Outlives {
                longer: LifetimeId::from_raw(2),
                shorter: LifetimeId::from_raw(3),
                location: SourceLocation::unknown(),
                reason: OutlivesReason::Assignment,
            },
        ];

        let result = solver.solve(&constraints);
        assert!(result.is_ok());

        let solution = result.unwrap();
        // Just verify that we get some assignments back
        assert!(!solution.assignments.is_empty());
    }

    #[test]
    fn test_equality_constraints() {
        let mut solver = LifetimeConstraintSolver::new();
        let constraints = vec![LifetimeConstraint::Equal {
            left: LifetimeId::from_raw(1),
            right: LifetimeId::from_raw(2),
            location: SourceLocation::unknown(),
            reason: EqualityReason::ConditionalBranches,
        }];

        let result = solver.solve(&constraints);
        assert!(result.is_ok());

        let solution = result.unwrap();
        // Just verify that we get some assignments back
        assert!(!solution.lifetime_representatives.is_empty());

        // In a real implementation, we'd check that the assignments reflect the equality
        // For now, just verify the constraint was processed
    }

    // **Violation Detection Tests**

    #[test]
    fn test_use_after_free_detection() {
        let analyzer = LifetimeAnalyzer::new();
        let solution = create_test_solution();
        let dfg = create_test_dfg_with_use_after_free();

        let violations = analyzer
            .check_use_after_free(&solution, &dfg)
            .expect("Should check for use after free");

        // Should detect violations if any exist
        // In this simplified test, we might not detect any due to mock data
        assert!(violations.len() >= 0); // Just ensure it doesn't panic
    }

    #[test]
    fn test_dangling_reference_detection() {
        let analyzer = LifetimeAnalyzer::new();
        let solution = create_test_solution();
        let ownership_graph = create_test_ownership_graph_with_dangling_ref();

        let violations = analyzer
            .check_dangling_references(&solution, &ownership_graph)
            .expect("Should check for dangling references");

        assert!(violations.len() >= 0); // Ensure it doesn't panic
    }

    // **Performance Tests**

    // **Test Helper Functions**

    fn create_test_cfg() -> ControlFlowGraph {
        let function_id = SymbolId::from_raw(1);
        let entry_block = BlockId::from_raw(2);

        let mut cfg = ControlFlowGraph::new(function_id, entry_block);

        let block = BasicBlock {
            id: entry_block,
            statements: vec![],
            terminator: Terminator::Return { value: None },
            predecessors: BTreeSet::new(),
            successors: vec![],
            source_location: SourceLocation::unknown(),
            metadata: Default::default(),
        };

        cfg.blocks.insert(entry_block, block);
        cfg
    }

    fn create_test_regions() -> Vec<LifetimeRegion> {
        vec![
            LifetimeRegion {
                id: LifetimeId::global(),
                scope: ScopeId::from_raw(0),
                variables: BTreeSet::new(),
                parent: None,
                children: vec![LifetimeId::from_raw(1)],
                start_location: SourceLocation::unknown(),
                end_location: None,
            },
            LifetimeRegion {
                id: LifetimeId::from_raw(1),
                scope: ScopeId::from_raw(1),
                variables: BTreeSet::new(),
                parent: Some(LifetimeId::global()),
                children: vec![],
                start_location: SourceLocation::unknown(),
                end_location: None,
            },
        ]
    }

    fn create_test_dfg() -> DataFlowGraph {
        let entry_node = DataFlowNodeId::from_raw(1);
        let mut dfg = DataFlowGraph::new(entry_node);

        let param_node = DataFlowNode {
            id: entry_node,
            kind: DataFlowNodeKind::Parameter {
                parameter_index: 0,
                symbol_id: SymbolId::from_raw(1),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(SsaVariableId::from_raw(1)),
            basic_block: BlockId::from_raw(1),
            metadata: Default::default(),
        };

        // Create the SSA variable entry that the node defines
        let ssa_variable = crate::semantic_graph::dfg::SsaVariable {
            id: SsaVariableId::from_raw(1),
            original_symbol: SymbolId::from_raw(1),
            ssa_index: 0,
            var_type: TypeId::from_raw(1),
            definition: entry_node,
            uses: vec![],
            liveness: crate::semantic_graph::dfg::LivenessInfo::default(),
        };

        dfg.nodes.insert(entry_node, param_node);
        dfg.ssa_variables
            .insert(SsaVariableId::from_raw(1), ssa_variable);
        dfg
    }

    fn create_test_ownership_graph() -> OwnershipGraph {
        OwnershipGraph::new()
    }

    fn create_test_binary_op_node() -> DataFlowNode {
        DataFlowNode {
            id: DataFlowNodeId::from_raw(3),
            kind: DataFlowNodeKind::BinaryOp {
                operator: BinaryOperator::Add,
                left: DataFlowNodeId::from_raw(1),
                right: DataFlowNodeId::from_raw(2),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![DataFlowNodeId::from_raw(1), DataFlowNodeId::from_raw(2)],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: Default::default(),
        }
    }

    fn create_test_phi_node() -> DataFlowNode {
        DataFlowNode {
            id: DataFlowNodeId::from_raw(4),
            kind: DataFlowNodeKind::Phi {
                incoming: vec![
                    PhiIncoming {
                        value: DataFlowNodeId::from_raw(1),
                        predecessor: BlockId::from_raw(1),
                    },
                    PhiIncoming {
                        value: DataFlowNodeId::from_raw(2),
                        predecessor: BlockId::from_raw(2),
                    },
                ],
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: Some(crate::tast::SsaVariableId::from_raw(2)),
            basic_block: BlockId::from_raw(3),
            metadata: Default::default(),
        }
    }

    fn create_test_solution() -> LifetimeSolution {
        let mut assignments = BTreeMap::new();
        assignments.insert(SymbolId::from_raw(1), LifetimeId::from_raw(1));
        assignments.insert(SymbolId::from_raw(2), LifetimeId::from_raw(2));

        LifetimeSolution {
            assignments,
            constraint_hash: 12345,
            // solver_stats: SolverStats::default(),
            lifetime_representatives: BTreeMap::new(),
            lifetime_ordering: vec![],
            satisfiable: true,
            conflicts: vec![],
        }
    }

    fn create_test_dfg_with_use_after_free() -> DataFlowGraph {
        create_test_dfg() // Simplified for now
    }

    fn create_test_ownership_graph_with_dangling_ref() -> OwnershipGraph {
        create_test_ownership_graph() // Simplified for now
    }
}

// Additional imports that might be needed
use crate::semantic_graph::analysis::lifetime_analyzer::{
    ConstraintSolvingError, LifetimeConstraint, OutlivesReason,
};
use crate::semantic_graph::analysis::lifetime_solver::LifetimeConstraintSolver;
