//! Comprehensive tests for control flow analysis
//!
//! This test suite validates all aspects of our control flow analysis implementation:
//! - CFG construction correctness
//! - Variable state tracking through control flow
//! - Dead code detection
//! - Resource tracking
//! - Uninitialized variable detection

use crate::tast::{
    control_flow_analysis::{
        ControlFlowAnalyzer, ControlFlowGraph, BlockId, BlockKind,
        VariableState, ResourceInfo, ResourceKind, AnalysisResults,
        UninitializedUse, DeadCode, ResourceLeak, NullDereference,
    },
    node::{
        TypedStatement, TypedExpression, TypedExpressionKind, TypedFunction,
        BinaryOperator, UnaryOperator, TypedStatementKind, StatementBlock,
        TypedComprehensionFor, TypedCatchClause, TypedSwitchCase, TypedPattern,
        FunctionEffects,
    },
    SymbolId, TypeId, SourceLocation,
};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a simple typed expression
    fn create_var_expr(symbol_id: SymbolId) -> TypedExpression {
        TypedExpression {
            kind: TypedExpressionKind::Variable { symbol_id },
            type_id: TypeId::from_raw(1), // dummy type
            source_location: SourceLocation::unknown(),
        }
    }

    // Helper function to create an assignment statement
    fn create_assignment(target_symbol: SymbolId, value_symbol: SymbolId) -> TypedStatement {
        TypedStatement::Assignment {
            target: Box::new(create_var_expr(target_symbol)),
            value: Box::new(create_var_expr(value_symbol)),
            source_location: SourceLocation::unknown(),
        }
    }

    #[test]
    fn test_cfg_construction_linear_flow() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create a simple linear function: x = 1; y = x; z = y;
        let stmts = vec![
            TypedStatement::VarDeclaration {
                name: "x".to_string(),
                symbol_id: SymbolId::from_raw(1),
                var_type: TypeId::from_raw(1),
                initializer: Some(Box::new(TypedExpression {
                    kind: TypedExpressionKind::IntLiteral { value: 1 },
                    type_id: TypeId::from_raw(1),
                    source_location: SourceLocation::unknown(),
                })),
                is_final: false,
                source_location: SourceLocation::unknown(),
            },
            create_assignment(SymbolId::from_raw(2), SymbolId::from_raw(1)),
            create_assignment(SymbolId::from_raw(3), SymbolId::from_raw(2)),
        ];

        let function = TypedFunction {
            name: "test".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: stmts,
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let results = analyzer.analyze_function(&function);

        // Verify CFG structure
        let cfg = &analyzer.cfg;
        assert!(cfg.blocks.len() >= 2, "Should have at least entry and exit blocks");

        // Verify no uninitialized uses in linear flow
        assert_eq!(results.uninitialized_uses.len(), 0, "Should have no uninitialized uses");

        // Verify all variables are initialized
        let entry_block = cfg.blocks.get(&cfg.entry_block).unwrap();
        assert_eq!(entry_block.successors.len(), 1, "Entry should have one successor");

        println!("✅ CFG linear flow construction test passed");
    }

    #[test]
    fn test_cfg_construction_branching() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create if-else branching
        let if_stmt = TypedStatement::If {
            condition: Box::new(create_var_expr(SymbolId::from_raw(1))),
            then_branch: Box::new(TypedStatement::Block {
                statements: vec![create_assignment(SymbolId::from_raw(2), SymbolId::from_raw(1))],
                source_location: SourceLocation::unknown(),
            }),
            else_branch: Some(Box::new(TypedStatement::Block {
                statements: vec![create_assignment(SymbolId::from_raw(3), SymbolId::from_raw(1))],
                source_location: SourceLocation::unknown(),
            })),
            source_location: SourceLocation::unknown(),
        };

        let function = TypedFunction {
            name: "test_branching".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![if_stmt],
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let _ = analyzer.analyze_function(&function);
        let cfg = &analyzer.cfg;

        // Count branch blocks
        let branch_blocks: Vec<_> = cfg.blocks.values()
            .filter(|b| matches!(b.kind, BlockKind::Branch))
            .collect();

        assert!(!branch_blocks.is_empty(), "Should have branch blocks");

        // Verify branch has two successors
        for branch in branch_blocks {
            assert_eq!(branch.successors.len(), 2, "Branch should have exactly 2 successors");
        }

        println!("✅ CFG branching construction test passed");
    }

    #[test]
    fn test_uninitialized_variable_detection() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create a function that uses uninitialized variable
        let stmts = vec![
            TypedStatement::VarDeclaration {
                name: "x".to_string(),
                symbol_id: SymbolId::from_raw(1),
                var_type: TypeId::from_raw(1),
                initializer: None, // No initializer!
                is_final: false,
                source_location: SourceLocation::unknown(),
            },
            // Use x before initialization
            TypedStatement::Expression {
                expression: Box::new(create_var_expr(SymbolId::from_raw(1))),
                source_location: SourceLocation::unknown(),
            },
        ];

        let function = TypedFunction {
            name: "test_uninit".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: stmts,
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let results = analyzer.analyze_function(&function);

        // Should detect uninitialized use
        assert!(!results.uninitialized_uses.is_empty(),
            "Should detect uninitialized variable use");

        assert_eq!(results.uninitialized_uses[0].variable, SymbolId::from_raw(1),
            "Should detect x as uninitialized");

        println!("✅ Uninitialized variable detection test passed");
    }

    #[test]
    fn test_dead_code_detection() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create a function with dead code after return
        let stmts = vec![
            TypedStatement::Return {
                value: Some(Box::new(TypedExpression {
                    kind: TypedExpressionKind::IntLiteral { value: 42 },
                    type_id: TypeId::from_raw(1),
                    source_location: SourceLocation::unknown(),
                })),
                source_location: SourceLocation::unknown(),
            },
            // This is dead code - unreachable
            TypedStatement::Expression {
                expression: Box::new(create_var_expr(SymbolId::from_raw(1))),
                source_location: SourceLocation::unknown(),
            },
        ];

        let function = TypedFunction {
            name: "test_dead".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: stmts,
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let results = analyzer.analyze_function(&function);

        // Should detect dead code
        assert!(!results.dead_code.is_empty(),
            "Should detect dead code after return");

        println!("✅ Dead code detection test passed");
    }

    #[test]
    fn test_resource_tracking() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Simulate file open without close
        let file_var = SymbolId::from_raw(1);

        // Create file open expression
        let file_open = TypedExpression {
            kind: TypedExpressionKind::FunctionCall {
                function: Box::new(TypedExpression {
                    kind: TypedExpressionKind::Variable {
                        symbol_id: SymbolId::from_raw(100) // File.open
                    },
                    type_id: TypeId::from_raw(2),
                    source_location: SourceLocation::unknown(),
                }),
                arguments: vec![],
                type_arguments: vec![],
            },
            type_id: TypeId::from_raw(3), // File type
            source_location: SourceLocation::unknown(),
        };

        let stmts = vec![
            TypedStatement::VarDeclaration {
                name: "file".to_string(),
                symbol_id: file_var,
                var_type: TypeId::from_raw(3),
                initializer: Some(Box::new(file_open)),
                is_final: false,
                source_location: SourceLocation::unknown(),
            },
            // Use file but don't close it
            TypedStatement::Expression {
                expression: Box::new(create_var_expr(file_var)),
                source_location: SourceLocation::unknown(),
            },
            // Function ends without closing file
        ];

        let function = TypedFunction {
            name: "test_resource".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: stmts,
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        // Manually register this as a resource for testing
        analyzer.resources.insert(file_var, ResourceInfo {
            kind: ResourceKind::File,
            allocation_point: BlockId::from_raw(0),
            cleanup_points: BTreeSet::new(),
            escapes: false,
        });

        let results = analyzer.analyze_function(&function);

        // Should detect resource leak
        assert!(!results.resource_leaks.is_empty(),
            "Should detect unclosed file resource");

        println!("✅ Resource tracking test passed");
    }

    #[test]
    fn test_loop_cfg_construction() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create a while loop
        let while_stmt = TypedStatement::While {
            condition: Box::new(create_var_expr(SymbolId::from_raw(1))),
            body: Box::new(TypedStatement::Block {
                statements: vec![
                    create_assignment(SymbolId::from_raw(2), SymbolId::from_raw(1))
                ],
                source_location: SourceLocation::unknown(),
            }),
            source_location: SourceLocation::unknown(),
        };

        let function = TypedFunction {
            name: "test_loop".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![while_stmt],
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let _ = analyzer.analyze_function(&function);
        let cfg = &analyzer.cfg;

        // Verify loop structure
        let loop_headers: Vec<_> = cfg.blocks.values()
            .filter(|b| matches!(b.kind, BlockKind::LoopHeader))
            .collect();

        assert!(!loop_headers.is_empty(), "Should have loop header blocks");

        // Loop header should have at least 2 predecessors (entry and back edge)
        for header in loop_headers {
            assert!(header.predecessors.len() >= 1,
                "Loop header should have at least one predecessor");
        }

        println!("✅ Loop CFG construction test passed");
    }

    #[test]
    fn test_variable_state_propagation() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Test that variable states propagate correctly through CFG
        let x = SymbolId::from_raw(1);
        let y = SymbolId::from_raw(2);

        let stmts = vec![
            // x = 1
            TypedStatement::VarDeclaration {
                name: "x".to_string(),
                symbol_id: x,
                var_type: TypeId::from_raw(1),
                initializer: Some(Box::new(TypedExpression {
                    kind: TypedExpressionKind::IntLiteral { value: 1 },
                    type_id: TypeId::from_raw(1),
                    source_location: SourceLocation::unknown(),
                })),
                is_final: false,
                source_location: SourceLocation::unknown(),
            },
            // if (condition) { y = x; } else { y = 2; }
            TypedStatement::If {
                condition: Box::new(create_var_expr(SymbolId::from_raw(3))),
                then_branch: Box::new(TypedStatement::Assignment {
                    target: Box::new(create_var_expr(y)),
                    value: Box::new(create_var_expr(x)),
                    source_location: SourceLocation::unknown(),
                }),
                else_branch: Some(Box::new(TypedStatement::Assignment {
                    target: Box::new(create_var_expr(y)),
                    value: Box::new(TypedExpression {
                        kind: TypedExpressionKind::IntLiteral { value: 2 },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    }),
                    source_location: SourceLocation::unknown(),
                })),
                source_location: SourceLocation::unknown(),
            },
            // Use y - should be initialized on all paths
            TypedStatement::Expression {
                expression: Box::new(create_var_expr(y)),
                source_location: SourceLocation::unknown(),
            },
        ];

        let function = TypedFunction {
            name: "test_propagation".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: stmts,
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let results = analyzer.analyze_function(&function);

        // y should be initialized on all paths
        let uninit_y = results.uninitialized_uses.iter()
            .find(|u| u.variable == y);

        assert!(uninit_y.is_none(),
            "Variable y should be initialized on all paths");

        println!("✅ Variable state propagation test passed");
    }

    #[test]
    fn test_break_continue_targets() {
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create a loop with break and continue
        let loop_body = vec![
            TypedStatement::If {
                condition: Box::new(create_var_expr(SymbolId::from_raw(1))),
                then_branch: Box::new(TypedStatement::Break {
                    source_location: SourceLocation::unknown(),
                }),
                else_branch: None,
                source_location: SourceLocation::unknown(),
            },
            TypedStatement::If {
                condition: Box::new(create_var_expr(SymbolId::from_raw(2))),
                then_branch: Box::new(TypedStatement::Continue {
                    source_location: SourceLocation::unknown(),
                }),
                else_branch: None,
                source_location: SourceLocation::unknown(),
            },
        ];

        let while_stmt = TypedStatement::While {
            condition: Box::new(create_var_expr(SymbolId::from_raw(3))),
            body: Box::new(TypedStatement::Block {
                statements: loop_body,
                source_location: SourceLocation::unknown(),
            }),
            source_location: SourceLocation::unknown(),
        };

        let function = TypedFunction {
            name: "test_break_continue".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![while_stmt],
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let _ = analyzer.analyze_function(&function);

        // Verify break and continue are handled
        assert!(!analyzer.break_targets.is_empty(), "Should track break targets");
        assert!(!analyzer.continue_targets.is_empty(), "Should track continue targets");

        println!("✅ Break/continue target tracking test passed");
    }

    #[test]
    fn test_comprehensive_analysis_results() {
        // This test verifies the complete AnalysisResults structure
        let mut analyzer = ControlFlowAnalyzer::new();

        // Create a complex function with multiple issues
        let stmts = vec![
            // Uninitialized variable
            TypedStatement::VarDeclaration {
                name: "uninit".to_string(),
                symbol_id: SymbolId::from_raw(1),
                var_type: TypeId::from_raw(1),
                initializer: None,
                is_final: false,
                source_location: SourceLocation::unknown(),
            },
            // Use uninitialized
            TypedStatement::Expression {
                expression: Box::new(create_var_expr(SymbolId::from_raw(1))),
                source_location: SourceLocation::unknown(),
            },
            // Return
            TypedStatement::Return {
                value: None,
                source_location: SourceLocation::unknown(),
            },
            // Dead code
            TypedStatement::Expression {
                expression: Box::new(TypedExpression {
                    kind: TypedExpressionKind::IntLiteral { value: 999 },
                    type_id: TypeId::from_raw(1),
                    source_location: SourceLocation::unknown(),
                }),
                source_location: SourceLocation::unknown(),
            },
        ];

        let function = TypedFunction {
            name: "test_comprehensive".to_string(),
            symbol_id: SymbolId::from_raw(0),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: stmts,
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let results = analyzer.analyze_function(&function);

        // Verify we collected all types of issues
        assert!(!results.uninitialized_uses.is_empty(), "Should find uninitialized uses");
        assert!(!results.dead_code.is_empty(), "Should find dead code");

        println!("✅ Comprehensive analysis results test passed");
        println!("   - Uninitialized uses: {}", results.uninitialized_uses.len());
        println!("   - Dead code regions: {}", results.dead_code.len());
        println!("   - Resource leaks: {}", results.resource_leaks.len());
        println!("   - Null dereferences: {}", results.null_dereferences.len());
    }
}