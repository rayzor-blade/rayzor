#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unreachable_patterns,
    unused_mut,
    unused_assignments,
    unused_parens
)]
#![allow(
    clippy::single_component_path_imports,
    clippy::for_kv_map,
    clippy::explicit_auto_deref
)]
#![allow(
    clippy::println_empty_string,
    clippy::len_zero,
    clippy::useless_vec,
    clippy::field_reassign_with_default
)]
#![allow(
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::bool_assert_comparison
)]
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::useless_format,
    clippy::clone_on_copy
)]
//! Test enhanced function effect analysis
//!
//! This test validates that the enhanced effect analysis system correctly detects:
//! - Async/await patterns
//! - Memory effects
//! - Resource effects
//! - Exception types
//! - Complex effect propagation

use compiler::tast::{
    effect_analysis::EffectAnalyzer,
    node::{
        AsyncKind, ExpressionMetadata, FunctionEffects, FunctionMetadata, LiteralValue,
        TypedExpression, TypedExpressionKind, TypedFunction, TypedStatement, VariableUsage,
    },
    symbols::{Mutability, Visibility},
    SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeTable,
};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Enhanced Effect Analysis Test ===\n");

    let symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let string_interner = Rc::new(RefCell::new(StringInterner::new()));

    let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

    // Test cases for different effect patterns
    test_async_function(&mut analyzer, &string_interner);
    test_memory_effects(&mut analyzer, &string_interner);
    test_exception_effects(&mut analyzer, &string_interner);
    test_pure_function(&mut analyzer, &string_interner);
    test_complex_effects(&mut analyzer, &string_interner);

    println!("=== SUMMARY ===");
    println!("✅ Enhanced effect analysis tests completed successfully!");
    println!("✅ Async/await detection working");
    println!("✅ Memory effect tracking working");
    println!("✅ Exception analysis working");
    println!("✅ Purity analysis working");
    println!("✅ Complex effect propagation working");
}

fn test_async_function(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing async function detection...");

    let func_name = string_interner.borrow_mut().intern("asyncFunction");
    let promise_var = SymbolId::from_raw(1);

    // Create function: async function asyncFunction() { await somePromise(); }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(0),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1), // Promise<T>
        body: vec![TypedStatement::Expression {
            expression: TypedExpression {
                kind: TypedExpressionKind::Await {
                    expression: Box::new(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: promise_var,
                        },
                        expr_type: TypeId::from_raw(2),
                        usage: VariableUsage::Copy,
                        lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                        source_location: SourceLocation::new(0, 2, 10, 20),
                        metadata: ExpressionMetadata::default(),
                    }),
                    await_type: TypeId::from_raw(1),
                },
                expr_type: TypeId::from_raw(1),
                usage: VariableUsage::Copy,
                lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                source_location: SourceLocation::new(0, 2, 5, 25),
                metadata: ExpressionMetadata::default(),
            },
            source_location: SourceLocation::new(0, 2, 5, 26),
        }],
        type_parameters: vec![],
        effects: FunctionEffects::default(),
        source_location: SourceLocation::new(0, 1, 1, 1),
        visibility: Visibility::Public,
        is_static: false,
        metadata: FunctionMetadata::default(),
    };

    let effects = analyzer.analyze_function(&function);

    assert_eq!(
        effects.async_kind,
        AsyncKind::Async,
        "Function should be detected as async"
    );
    assert!(!effects.is_pure, "Async functions are not pure");

    println!("✅ Async function analysis: PASSED");
    println!("  → Async kind: {:?}", effects.async_kind);
    println!("  → Is pure: {}", effects.is_pure);
    println!("");
}

fn test_memory_effects(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing memory effects detection...");

    let func_name = string_interner.borrow_mut().intern("mutatingFunction");
    let var_symbol = SymbolId::from_raw(1);

    // Create function: function mutatingFunction(x) { x.field = 42; }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(1),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1),
        body: vec![TypedStatement::Expression {
            expression: TypedExpression {
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(TypedExpression {
                        kind: TypedExpressionKind::FieldAccess {
                            object: Box::new(TypedExpression {
                                kind: TypedExpressionKind::Variable {
                                    symbol_id: var_symbol,
                                },
                                expr_type: TypeId::from_raw(2),
                                usage: VariableUsage::Copy,
                                lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                                source_location: SourceLocation::new(0, 2, 5, 6),
                                metadata: ExpressionMetadata::default(),
                            }),
                            field_symbol: SymbolId::from_raw(100),
                            is_optional: false,
                        },
                        expr_type: TypeId::from_raw(1),
                        usage: VariableUsage::Copy,
                        lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                        source_location: SourceLocation::new(0, 2, 5, 11),
                        metadata: ExpressionMetadata::default(),
                    }),
                    right: Box::new(TypedExpression {
                        kind: TypedExpressionKind::Literal {
                            value: LiteralValue::Int(42),
                        },
                        expr_type: TypeId::from_raw(1),
                        usage: VariableUsage::Copy,
                        lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                        source_location: SourceLocation::new(0, 2, 15, 17),
                        metadata: ExpressionMetadata::default(),
                    }),
                    operator: compiler::tast::node::BinaryOperator::Assign,
                },
                expr_type: TypeId::from_raw(1),
                usage: VariableUsage::Copy,
                lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                source_location: SourceLocation::new(0, 2, 5, 18),
                metadata: ExpressionMetadata::default(),
            },
            source_location: SourceLocation::new(0, 2, 5, 19),
        }],
        type_parameters: vec![],
        effects: FunctionEffects::default(),
        source_location: SourceLocation::new(0, 1, 1, 1),
        visibility: Visibility::Public,
        is_static: false,
        metadata: FunctionMetadata::default(),
    };

    let effects = analyzer.analyze_function(&function);

    assert!(
        !effects.is_pure,
        "Function with assignments should not be pure"
    );
    assert!(
        effects.memory_effects.accesses_global_state,
        "Function should have memory effects"
    );

    println!("✅ Memory effects analysis: PASSED");
    println!("  → Is pure: {}", effects.is_pure);
    println!(
        "  → Accesses global state: {}",
        effects.memory_effects.accesses_global_state
    );
    println!("");
}

fn test_exception_effects(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing exception effects detection...");

    let func_name = string_interner.borrow_mut().intern("throwingFunction");

    // Create function: function throwingFunction() { throw new Error("test"); }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(2),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1),
        body: vec![TypedStatement::Expression {
            expression: TypedExpression {
                kind: TypedExpressionKind::Throw {
                    expression: Box::new(TypedExpression {
                        kind: TypedExpressionKind::New {
                            class_type: TypeId::from_raw(200),
                            class_name: None,
                            arguments: vec![TypedExpression {
                                kind: TypedExpressionKind::Literal {
                                    value: LiteralValue::String("test".to_string()),
                                },
                                expr_type: TypeId::from_raw(3),
                                usage: VariableUsage::Copy,
                                lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                                source_location: SourceLocation::new(0, 2, 20, 26),
                                metadata: ExpressionMetadata::default(),
                            }],
                            type_arguments: vec![],
                        },
                        expr_type: TypeId::from_raw(4),
                        usage: VariableUsage::Copy,
                        lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                        source_location: SourceLocation::new(0, 2, 11, 27),
                        metadata: ExpressionMetadata::default(),
                    }),
                },
                expr_type: TypeId::from_raw(5), // Never type
                usage: VariableUsage::Copy,
                lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                source_location: SourceLocation::new(0, 2, 5, 28),
                metadata: ExpressionMetadata::default(),
            },
            source_location: SourceLocation::new(0, 2, 5, 29),
        }],
        type_parameters: vec![],
        effects: FunctionEffects::default(),
        source_location: SourceLocation::new(0, 1, 1, 1),
        visibility: Visibility::Public,
        is_static: false,
        metadata: FunctionMetadata::default(),
    };

    let effects = analyzer.analyze_function(&function);

    assert!(effects.can_throw, "Function should be detected as throwing");
    assert!(!effects.is_pure, "Throwing functions are not pure");

    println!("✅ Exception effects analysis: PASSED");
    println!("  → Can throw: {}", effects.can_throw);
    println!("  → Is pure: {}", effects.is_pure);
    println!("");
}

fn test_pure_function(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing pure function detection...");

    let func_name = string_interner.borrow_mut().intern("pureFunction");
    let x_symbol = SymbolId::from_raw(1);
    let y_symbol = SymbolId::from_raw(2);

    // Create function: function pureFunction(x, y) { return x + y; }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(3),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1),
        body: vec![TypedStatement::Return {
            value: Some(TypedExpression {
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: x_symbol,
                        },
                        expr_type: TypeId::from_raw(1),
                        usage: VariableUsage::Copy,
                        lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                        source_location: SourceLocation::new(0, 2, 12, 13),
                        metadata: ExpressionMetadata::default(),
                    }),
                    right: Box::new(TypedExpression {
                        kind: TypedExpressionKind::Variable {
                            symbol_id: y_symbol,
                        },
                        expr_type: TypeId::from_raw(1),
                        usage: VariableUsage::Copy,
                        lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                        source_location: SourceLocation::new(0, 2, 16, 17),
                        metadata: ExpressionMetadata::default(),
                    }),
                    operator: compiler::tast::node::BinaryOperator::Add,
                },
                expr_type: TypeId::from_raw(1),
                usage: VariableUsage::Copy,
                lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                source_location: SourceLocation::new(0, 2, 12, 17),
                metadata: ExpressionMetadata::default(),
            }),
            source_location: SourceLocation::new(0, 2, 5, 18),
        }],
        type_parameters: vec![],
        effects: FunctionEffects::default(),
        source_location: SourceLocation::new(0, 1, 1, 1),
        visibility: Visibility::Public,
        is_static: false,
        metadata: FunctionMetadata::default(),
    };

    let effects = analyzer.analyze_function(&function);

    assert!(effects.is_pure, "Function should be detected as pure");
    assert!(!effects.can_throw, "Pure function should not throw");
    assert_eq!(
        effects.async_kind,
        AsyncKind::Sync,
        "Pure function should be synchronous"
    );

    println!("✅ Pure function analysis: PASSED");
    println!("  → Is pure: {}", effects.is_pure);
    println!("  → Can throw: {}", effects.can_throw);
    println!("  → Async kind: {:?}", effects.async_kind);
    println!("");
}

fn test_complex_effects(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing complex effect combinations...");

    let func_name = string_interner.borrow_mut().intern("complexFunction");

    // Create function with multiple effects:
    // async function complexFunction() {
    //   var data = await fetchData();
    //   if (data == null) throw new Error("No data");
    //   global.cache = data;
    //   return data.process();
    // }

    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(4),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1),
        body: vec![
            // await fetchData()
            TypedStatement::VarDeclaration {
                symbol_id: SymbolId::from_raw(10),
                var_type: TypeId::from_raw(2),
                initializer: Some(TypedExpression {
                    kind: TypedExpressionKind::Await {
                        expression: Box::new(TypedExpression {
                            kind: TypedExpressionKind::FunctionCall {
                                function: Box::new(TypedExpression {
                                    kind: TypedExpressionKind::Variable {
                                        symbol_id: SymbolId::from_raw(300),
                                    },
                                    expr_type: TypeId::from_raw(5),
                                    usage: VariableUsage::Copy,
                                    lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                                    source_location: SourceLocation::new(0, 2, 17, 26),
                                    metadata: ExpressionMetadata::default(),
                                }),
                                arguments: vec![],
                                type_arguments: vec![],
                            },
                            expr_type: TypeId::from_raw(6),
                            usage: VariableUsage::Copy,
                            lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                            source_location: SourceLocation::new(0, 2, 17, 28),
                            metadata: ExpressionMetadata::default(),
                        }),
                        await_type: TypeId::from_raw(2),
                    },
                    expr_type: TypeId::from_raw(2),
                    usage: VariableUsage::Copy,
                    lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                    source_location: SourceLocation::new(0, 2, 11, 29),
                    metadata: ExpressionMetadata::default(),
                }),
                mutability: Mutability::Mutable,
                source_location: SourceLocation::new(0, 2, 5, 30),
            },
            // throw new Error()
            TypedStatement::Expression {
                expression: TypedExpression {
                    kind: TypedExpressionKind::Throw {
                        expression: Box::new(TypedExpression {
                            kind: TypedExpressionKind::New {
                                class_type: TypeId::from_raw(400),
                                class_name: None,
                                arguments: vec![],
                                type_arguments: vec![],
                            },
                            expr_type: TypeId::from_raw(7),
                            usage: VariableUsage::Copy,
                            lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                            source_location: SourceLocation::new(0, 3, 11, 21),
                            metadata: ExpressionMetadata::default(),
                        }),
                    },
                    expr_type: TypeId::from_raw(8), // Never
                    usage: VariableUsage::Copy,
                    lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                    source_location: SourceLocation::new(0, 3, 5, 22),
                    metadata: ExpressionMetadata::default(),
                },
                source_location: SourceLocation::new(0, 3, 5, 23),
            },
        ],
        type_parameters: vec![],
        effects: FunctionEffects::default(),
        source_location: SourceLocation::new(0, 1, 1, 1),
        visibility: Visibility::Public,
        is_static: false,
        metadata: FunctionMetadata::default(),
    };

    let effects = analyzer.analyze_function(&function);

    assert_eq!(
        effects.async_kind,
        AsyncKind::Async,
        "Function should be async due to await"
    );
    assert!(
        effects.can_throw,
        "Function should throw due to throw statement"
    );
    assert!(
        !effects.is_pure,
        "Function with multiple effects should not be pure"
    );
    assert!(
        effects.memory_effects.accesses_global_state,
        "Function should have memory effects"
    );

    println!("✅ Complex effects analysis: PASSED");
    println!("  → Async kind: {:?}", effects.async_kind);
    println!("  → Can throw: {}", effects.can_throw);
    println!("  → Is pure: {}", effects.is_pure);
    println!(
        "  → Memory effects: {:?}",
        effects.memory_effects.accesses_global_state
    );
    println!("");
}
