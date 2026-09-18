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
//! Direct test of enhanced type checking integration gaps
//!
//! This directly calls the control flow analyzer to validate our fixes.

use compiler::tast::{
    SourceLocation, StringInterner, SymbolId, TypeId,
    control_flow_analysis::ControlFlowAnalyzer,
    node::{
        BinaryOperator, ExpressionMetadata, FunctionEffects, FunctionMetadata, LiteralValue,
        TypedExpression, TypedExpressionKind, TypedFunction, TypedStatement, VariableUsage,
    },
    symbols::{Mutability, Visibility},
};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Testing Enhanced Type Checking Integration Gaps ===");

    let string_interner = Rc::new(RefCell::new(StringInterner::new()));
    let func_name = string_interner.borrow_mut().intern("test");
    let x_symbol = SymbolId::from_raw(1);

    // Create function: function test(): Int { var x: Int; return x + 1; }
    let function = TypedFunction {
        symbol_id: SymbolId::from_raw(0),
        name: func_name,
        parameters: vec![],
        return_type: TypeId::from_raw(1), // Int
        body: vec![
            // var x: Int; (uninitialized)
            TypedStatement::VarDeclaration {
                symbol_id: x_symbol,
                var_type: TypeId::from_raw(1),
                initializer: None, // UNINITIALIZED!
                mutability: Mutability::Mutable,
                source_location: SourceLocation::new(0, 2, 5, 15),
            },
            // return x + 1; (use of uninitialized variable)
            TypedStatement::Return {
                value: Some(TypedExpression {
                    kind: TypedExpressionKind::BinaryOp {
                        left: Box::new(TypedExpression {
                            kind: TypedExpressionKind::Variable {
                                symbol_id: x_symbol,
                            },
                            expr_type: TypeId::from_raw(1),
                            usage: VariableUsage::Copy,
                            lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                            source_location: SourceLocation::new(0, 3, 12, 20),
                            metadata: ExpressionMetadata::default(),
                        }),
                        right: Box::new(TypedExpression {
                            kind: TypedExpressionKind::Literal {
                                value: LiteralValue::Int(1),
                            },
                            expr_type: TypeId::from_raw(1),
                            usage: VariableUsage::Copy,
                            lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                            source_location: SourceLocation::new(0, 3, 16, 24),
                            metadata: ExpressionMetadata::default(),
                        }),
                        operator: BinaryOperator::Add,
                    },
                    expr_type: TypeId::from_raw(1),
                    usage: VariableUsage::Copy,
                    lifetime_id: compiler::tast::LifetimeId::from_raw(0),
                    source_location: SourceLocation::new(0, 3, 10, 18),
                    metadata: ExpressionMetadata::default(),
                }),
                source_location: SourceLocation::new(0, 3, 5, 25),
            },
        ],
        type_parameters: vec![],
        effects: FunctionEffects::default(),
        source_location: SourceLocation::new(0, 1, 1, 1),
        visibility: Visibility::Public,
        is_static: false,
        metadata: FunctionMetadata::default(),
    };

    // Test control flow analyzer directly
    println!("Creating control flow analyzer...");
    let mut analyzer = ControlFlowAnalyzer::new();

    println!("Running analysis on function...");
    let results = analyzer.analyze_function(&function);

    println!("\n=== CONTROL FLOW ANALYSIS RESULTS ===");
    println!(
        "- Uninitialized uses found: {}",
        results.uninitialized_uses.len()
    );
    println!("- Dead code regions found: {}", results.dead_code.len());
    println!(
        "- Null dereferences found: {}",
        results.null_dereferences.len()
    );

    // Print detailed results
    for (i, uninit) in results.uninitialized_uses.iter().enumerate() {
        println!(
            "  Uninitialized Use #{}: Variable {:?} at {}:{}",
            i + 1,
            uninit.variable,
            uninit.location.line,
            uninit.location.column
        );
    }

    for (i, dead_code) in results.dead_code.iter().enumerate() {
        println!(
            "  Dead Code #{}: {} at {}:{}",
            i + 1,
            dead_code.message,
            dead_code.location.line,
            dead_code.location.column
        );
    }

    // CRITICAL TEST: Check if we detected the uninitialized variable
    if !results.uninitialized_uses.is_empty() {
        let uninit_use = &results.uninitialized_uses[0];
        if uninit_use.variable == x_symbol {
            println!(
                "\n🎉 SUCCESS: Detected uninitialized variable 'x' at line {}, column {}",
                uninit_use.location.line, uninit_use.location.column
            );
            println!("✅ Integration gap fixes are working correctly!");
            println!("✅ Variables are being properly registered and tracked");
            println!("✅ Uninitialized variable detection is functioning");
        } else {
            println!("\n⚠️  Detected uninitialized variable but wrong symbol");
            println!(
                "   Expected: {:?}, Got: {:?}",
                x_symbol, uninit_use.variable
            );
        }
    } else {
        println!("\nℹ️  Control flow analyzer ran but didn't detect uninitialized variable");
        println!("   This indicates the integration fixes may need further refinement");
        println!(
            "   Variables may not be properly registered or state propagation may be incomplete"
        );
    }

    println!("\n=== INTEGRATION GAP ANALYSIS COMPLETE ===");
}
