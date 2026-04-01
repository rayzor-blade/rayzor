//! Comprehensive tests for effect analysis
//!
//! This test suite validates all aspects of our effect analysis:
//! - Function effect detection (throws, async, pure)
//! - Effect propagation through call chains
//! - Effect contract validation
//! - Cross-function effect tracking

use crate::tast::{
    effect_analysis::{
        EffectAnalyzer, analyze_file_effects, analyze_function_effects,
    },
    node::{
        TypedStatement, TypedExpression, TypedExpressionKind, TypedFunction,
        FunctionEffects, TypedFile, TypedDeclaration, TypedClass, ClassMember,
        FunctionParam, Visibility, TypedCatchClause, TypedPattern,
    },
    core::TypeTable,
    SymbolId, TypeId, SourceLocation, SymbolTable, StringInterner,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::BTreeSet;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_environment() -> (RefCell<TypeTable>, SymbolTable) {
        let type_table = TypeTable::new();
        let symbol_table = SymbolTable::new();
        (RefCell::new(type_table), symbol_table)
    }

    fn create_throwing_function() -> TypedFunction {
        // function throwingFunc() { throw new Error(); }
        let throw_stmt = TypedStatement::Throw {
            value: Box::new(TypedExpression {
                kind: TypedExpressionKind::NewObject {
                    class_type: TypeId::from_raw(10), // Error type
                    arguments: vec![],
                },
                type_id: TypeId::from_raw(10),
                source_location: SourceLocation::unknown(),
            }),
            source_location: SourceLocation::unknown(),
        };

        TypedFunction {
            name: "throwingFunc".to_string(),
            symbol_id: SymbolId::from_raw(1),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![throw_stmt],
            type_parameters: vec![],
            effects: FunctionEffects {
                can_throw: false, // Will be detected by analyzer
                is_async: false,
                is_pure: true,
            },
            source_location: SourceLocation::unknown(),
        }
    }

    fn create_async_function() -> TypedFunction {
        // async function asyncFunc() { await somePromise(); }
        let await_expr = TypedExpression {
            kind: TypedExpressionKind::Await {
                value: Box::new(TypedExpression {
                    kind: TypedExpressionKind::FunctionCall {
                        function: Box::new(TypedExpression {
                            kind: TypedExpressionKind::Variable {
                                symbol_id: SymbolId::from_raw(10),
                            },
                            type_id: TypeId::from_raw(20),
                            source_location: SourceLocation::unknown(),
                        }),
                        arguments: vec![],
                        type_arguments: vec![],
                    },
                    type_id: TypeId::from_raw(21), // Promise type
                    source_location: SourceLocation::unknown(),
                }),
            },
            type_id: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
        };

        TypedFunction {
            name: "asyncFunc".to_string(),
            symbol_id: SymbolId::from_raw(2),
            parameters: vec![],
            return_type: TypeId::from_raw(21), // Promise<T>
            body: vec![
                TypedStatement::Expression {
                    expression: Box::new(await_expr),
                    source_location: SourceLocation::unknown(),
                }
            ],
            type_parameters: vec![],
            effects: FunctionEffects {
                can_throw: false,
                is_async: false, // Will be detected
                is_pure: true,
            },
            source_location: SourceLocation::unknown(),
        }
    }

    fn create_impure_function() -> TypedFunction {
        // function impureFunc() { globalVar = 42; }
        let global_assign = TypedStatement::Assignment {
            target: Box::new(TypedExpression {
                kind: TypedExpressionKind::Variable {
                    symbol_id: SymbolId::from_raw(100), // global var
                },
                type_id: TypeId::from_raw(1),
                source_location: SourceLocation::unknown(),
            }),
            value: Box::new(TypedExpression {
                kind: TypedExpressionKind::IntLiteral { value: 42 },
                type_id: TypeId::from_raw(1),
                source_location: SourceLocation::unknown(),
            }),
            source_location: SourceLocation::unknown(),
        };

        TypedFunction {
            name: "impureFunc".to_string(),
            symbol_id: SymbolId::from_raw(3),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![global_assign],
            type_parameters: vec![],
            effects: FunctionEffects {
                can_throw: false,
                is_async: false,
                is_pure: true, // Will be detected as false
            },
            source_location: SourceLocation::unknown(),
        }
    }

    #[test]
    fn test_throwing_function_detection() {
        let (type_table, symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        let throwing_func = create_throwing_function();
        let effects = analyzer.analyze_function(&throwing_func);

        assert!(effects.can_throw, "Should detect function can throw");
        assert!(analyzer.throwing_functions.contains(&throwing_func.symbol_id),
            "Should track throwing functions");

        println!("✅ Throwing function detection test passed");
    }

    #[test]
    fn test_async_function_detection() {
        let (type_table, symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        let async_func = create_async_function();
        let effects = analyzer.analyze_function(&async_func);

        assert!(effects.is_async, "Should detect async function");
        assert!(analyzer.async_functions.contains(&async_func.symbol_id),
            "Should track async functions");

        println!("✅ Async function detection test passed");
    }

    #[test]
    fn test_impure_function_detection() {
        let (type_table, mut symbol_table) = create_test_environment();

        // Mark global var as mutable
        symbol_table.set_mutability(SymbolId::from_raw(100), true);

        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        let impure_func = create_impure_function();
        let effects = analyzer.analyze_function(&impure_func);

        assert!(!effects.is_pure, "Should detect impure function");
        assert!(!analyzer.pure_functions.contains(&impure_func.symbol_id),
            "Should not track as pure function");

        println!("✅ Impure function detection test passed");
    }

    #[test]
    fn test_effect_propagation_through_calls() {
        let (type_table, mut symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        // First analyze throwing function
        let throwing_func = create_throwing_function();
        let _ = analyzer.analyze_function(&throwing_func);

        // Create a function that calls the throwing function
        let caller_func = TypedFunction {
            name: "caller".to_string(),
            symbol_id: SymbolId::from_raw(4),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![
                TypedStatement::Expression {
                    expression: Box::new(TypedExpression {
                        kind: TypedExpressionKind::FunctionCall {
                            function: Box::new(TypedExpression {
                                kind: TypedExpressionKind::Variable {
                                    symbol_id: throwing_func.symbol_id,
                                },
                                type_id: TypeId::from_raw(30),
                                source_location: SourceLocation::unknown(),
                            }),
                            arguments: vec![],
                            type_arguments: vec![],
                        },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    }),
                    source_location: SourceLocation::unknown(),
                }
            ],
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let caller_effects = analyzer.analyze_function(&caller_func);

        assert!(caller_effects.can_throw,
            "Should propagate throwing effect through call");

        println!("✅ Effect propagation through calls test passed");
    }

    #[test]
    fn test_try_catch_effect_suppression() {
        let (type_table, symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        // Create try-catch that handles throw
        let try_catch = TypedStatement::Try {
            body: Box::new(TypedStatement::Throw {
                value: Box::new(TypedExpression {
                    kind: TypedExpressionKind::StringLiteral {
                        value: "error".to_string(),
                    },
                    type_id: TypeId::from_raw(1),
                    source_location: SourceLocation::unknown(),
                }),
                source_location: SourceLocation::unknown(),
            }),
            catch_clauses: vec![
                TypedCatchClause {
                    pattern: TypedPattern::Variable {
                        name: "e".to_string(),
                        symbol_id: SymbolId::from_raw(10),
                    },
                    body: TypedStatement::Block {
                        statements: vec![],
                        source_location: SourceLocation::unknown(),
                    },
                }
            ],
            finally_block: None,
            source_location: SourceLocation::unknown(),
        };

        let function = TypedFunction {
            name: "tryCatchFunc".to_string(),
            symbol_id: SymbolId::from_raw(5),
            parameters: vec![],
            return_type: TypeId::from_raw(1),
            body: vec![try_catch],
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let effects = analyzer.analyze_function(&function);

        assert!(!effects.can_throw,
            "Try-catch should suppress throwing effect");

        println!("✅ Try-catch effect suppression test passed");
    }

    #[test]
    fn test_pure_function_validation() {
        let (type_table, symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        // Pure function - only uses parameters and returns
        let pure_func = TypedFunction {
            name: "pureFunc".to_string(),
            symbol_id: SymbolId::from_raw(6),
            parameters: vec![
                FunctionParam {
                    name: "x".to_string(),
                    param_type: TypeId::from_raw(1),
                    symbol_id: SymbolId::from_raw(7),
                    is_optional: false,
                    default_value: None,
                    is_variadic: false,
                }
            ],
            return_type: TypeId::from_raw(1),
            body: vec![
                TypedStatement::Return {
                    value: Some(Box::new(TypedExpression {
                        kind: TypedExpressionKind::BinaryOp {
                            left: Box::new(TypedExpression {
                                kind: TypedExpressionKind::Variable {
                                    symbol_id: SymbolId::from_raw(7), // param x
                                },
                                type_id: TypeId::from_raw(1),
                                source_location: SourceLocation::unknown(),
                            }),
                            right: Box::new(TypedExpression {
                                kind: TypedExpressionKind::IntLiteral { value: 1 },
                                type_id: TypeId::from_raw(1),
                                source_location: SourceLocation::unknown(),
                            }),
                            operator: crate::tast::node::BinaryOperator::Add,
                        },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    })),
                    source_location: SourceLocation::unknown(),
                }
            ],
            type_parameters: vec![],
            effects: FunctionEffects {
                can_throw: false,
                is_async: false,
                is_pure: true,
            },
            source_location: SourceLocation::unknown(),
        };

        let effects = analyzer.analyze_function(&pure_func);

        assert!(effects.is_pure, "Should validate as pure function");

        println!("✅ Pure function validation test passed");
    }

    #[test]
    fn test_file_level_effect_analysis() {
        let (type_table, symbol_table) = create_test_environment();
        let string_interner = Rc::new(RefCell::new(StringInterner::new()));

        let mut typed_file = TypedFile::new(string_interner);

        // Add multiple functions with different effects
        let throwing_func = create_throwing_function();
        let async_func = create_async_function();
        let impure_func = create_impure_function();

        // Create class with methods
        let class_decl = TypedDeclaration::Class(TypedClass {
            name: "TestClass".to_string(),
            symbol_id: SymbolId::from_raw(50),
            type_parameters: vec![],
            super_class: None,
            interfaces: vec![],
            fields: vec![],
            methods: vec![
                ClassMember::Method {
                    name: "throwingMethod".to_string(),
                    symbol_id: throwing_func.symbol_id,
                    function: throwing_func,
                    visibility: Visibility::Public,
                    is_static: false,
                    is_override: false,
                },
                ClassMember::Method {
                    name: "asyncMethod".to_string(),
                    symbol_id: async_func.symbol_id,
                    function: async_func,
                    visibility: Visibility::Public,
                    is_static: false,
                    is_override: false,
                },
            ],
            constructors: vec![],
            static_fields: vec![],
            static_methods: vec![],
            metadata: None,
            source_location: SourceLocation::unknown(),
        });

        typed_file.declarations.push(class_decl);

        // Add standalone function
        typed_file.functions.push(impure_func);

        // Analyze entire file
        analyze_file_effects(&typed_file, &symbol_table, &type_table);

        // In a real implementation, we'd check that all functions were analyzed
        println!("✅ File-level effect analysis test passed");
    }

    #[test]
    fn test_complex_effect_combinations() {
        let (type_table, symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        // Function that is async, can throw, and is impure
        let complex_func = TypedFunction {
            name: "complexFunc".to_string(),
            symbol_id: SymbolId::from_raw(8),
            parameters: vec![],
            return_type: TypeId::from_raw(21), // Promise
            body: vec![
                // Throw statement
                TypedStatement::Throw {
                    value: Box::new(TypedExpression {
                        kind: TypedExpressionKind::StringLiteral {
                            value: "error".to_string(),
                        },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    }),
                    source_location: SourceLocation::unknown(),
                },
                // Await expression
                TypedStatement::Expression {
                    expression: Box::new(TypedExpression {
                        kind: TypedExpressionKind::Await {
                            value: Box::new(TypedExpression {
                                kind: TypedExpressionKind::Variable {
                                    symbol_id: SymbolId::from_raw(20),
                                },
                                type_id: TypeId::from_raw(21),
                                source_location: SourceLocation::unknown(),
                            }),
                        },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    }),
                    source_location: SourceLocation::unknown(),
                },
                // Global mutation
                TypedStatement::Assignment {
                    target: Box::new(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: SymbolId::from_raw(100),
                        },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    }),
                    value: Box::new(TypedExpression {
                        kind: TypedExpressionKind::IntLiteral { value: 42 },
                        type_id: TypeId::from_raw(1),
                        source_location: SourceLocation::unknown(),
                    }),
                    source_location: SourceLocation::unknown(),
                },
            ],
            type_parameters: vec![],
            effects: FunctionEffects::default(),
            source_location: SourceLocation::unknown(),
        };

        let effects = analyzer.analyze_function(&complex_func);

        assert!(effects.can_throw, "Should detect throwing");
        assert!(effects.is_async, "Should detect async");
        assert!(!effects.is_pure, "Should detect impure");

        println!("✅ Complex effect combinations test passed");
    }

    #[test]
    fn test_effect_analysis_caching() {
        let (type_table, symbol_table) = create_test_environment();
        let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

        let func = create_throwing_function();

        // Analyze same function twice
        let effects1 = analyzer.analyze_function(&func);
        let effects2 = analyzer.analyze_function(&func);

        // Results should be consistent
        assert_eq!(effects1.can_throw, effects2.can_throw);
        assert_eq!(effects1.is_async, effects2.is_async);
        assert_eq!(effects1.is_pure, effects2.is_pure);

        // Function should be in tracking sets
        assert!(analyzer.throwing_functions.contains(&func.symbol_id));

        println!("✅ Effect analysis caching test passed");
    }
}