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
//! Test current function effect analysis system
//!
//! This test validates the existing effect analysis system and demonstrates
//! that the foundation is solid for building enhanced effect analysis.

use compiler::tast::{
    effect_analysis::EffectAnalyzer,
    node::{
        ExpressionMetadata, FunctionEffects, FunctionMetadata, LiteralValue, TypedExpression,
        TypedExpressionKind, TypedFunction, TypedStatement, VariableUsage,
    },
    symbols::{Mutability, Visibility},
    AsyncKind, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeTable,
};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Current Effect Analysis System Test ===\n");

    let symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let string_interner = Rc::new(RefCell::new(StringInterner::new()));

    let mut analyzer = EffectAnalyzer::new(&symbol_table, &type_table);

    // Test current effect analysis capabilities
    test_throwing_function(&mut analyzer, &string_interner);
    test_pure_function(&mut analyzer, &string_interner);
    test_side_effect_function(&mut analyzer, &string_interner);

    println!("=== CURRENT SYSTEM ASSESSMENT ===");
    println!("✅ Basic effect analysis: WORKING");
    println!("✅ Throw detection: WORKING");
    println!("✅ Purity analysis: WORKING");
    println!("✅ Side effect detection: WORKING");
    println!("");
    println!("📈 ENHANCEMENT OPPORTUNITIES:");
    println!("🔧 Async/await detection (syntax ready for implementation)");
    println!("🔧 Memory effect tracking (ownership system available)");
    println!("🔧 Resource effect analysis (I/O detection)");
    println!("🔧 Exception type tracking");
    println!("🔧 Effect contract validation");
}

fn test_throwing_function(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing exception throwing detection...");

    let func_name = string_interner.borrow_mut().intern("throwingFunction");

    // Create function: function throwingFunction() { throw new Error("test"); }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(0),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1),
        body: vec![TypedStatement::Expression {
            expression: TypedExpression {
                kind: TypedExpressionKind::Throw {
                    expression: Box::new(TypedExpression {
                        kind: TypedExpressionKind::New {
                            class_type: TypeId::from_raw(4),
                            type_arguments: vec![],
                            arguments: vec![],
                            class_name: None,
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

    println!("✅ Exception detection: PASSED");
    println!("  → Can throw: {}", effects.can_throw);
    println!("  → Is pure: {}", effects.is_pure);
    println!(
        "  → Is async: {}",
        matches!(effects.async_kind, AsyncKind::Sync)
    );
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
        symbol_id: SymbolId::from_raw(1),
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
    assert!(
        matches!(effects.async_kind, AsyncKind::Sync),
        "Pure function should be synchronous"
    );

    println!("✅ Pure function detection: PASSED");
    println!("  → Is pure: {}", effects.is_pure);
    println!("  → Can throw: {}", effects.can_throw);
    println!("  → Async kind: {:?}", effects.async_kind);
    println!("");
}

fn test_side_effect_function(
    analyzer: &mut EffectAnalyzer,
    string_interner: &Rc<RefCell<StringInterner>>,
) {
    println!("Testing side effect detection...");

    let func_name = string_interner.borrow_mut().intern("sideEffectFunction");
    let var_symbol = SymbolId::from_raw(1);

    // Create function: function sideEffectFunction(x) { x.field = 42; }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(2),
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
    assert!(!effects.can_throw, "Simple assignment should not throw");
    assert!(
        matches!(effects.async_kind, AsyncKind::Sync),
        "Simple assignment should be synchronous"
    );

    println!("✅ Side effect detection: PASSED");
    println!("  → Is pure: {}", effects.is_pure);
    println!("  → Can throw: {}", effects.can_throw);
    println!("  → Async kind: {:?}", effects.async_kind);
    println!("");
}
