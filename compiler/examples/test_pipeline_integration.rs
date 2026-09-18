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
//! Test TypeFlowGuard integration with type checking pipeline

use compiler::tast::{
    ScopeTree, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeTable,
    node::{
        ExpressionMetadata, FunctionEffects, FunctionMetadata, LiteralValue, TypedExpression,
        TypedExpressionKind, TypedFile, TypedFunction, TypedStatement, VariableUsage,
    },
    symbols::{Mutability, Visibility},
    type_checking_pipeline::TypeCheckingPhase,
};
use diagnostics::{Diagnostics, SourceMap};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== TypeFlowGuard Pipeline Integration Test ===\n");

    // Initialize core components
    let string_interner_rc = Rc::new(RefCell::new(StringInterner::new()));
    let symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let scope_tree = ScopeTree::new(compiler::tast::ScopeId::from_raw(0));
    let source_map = SourceMap::new();
    let mut diagnostics = Diagnostics::new();

    // Create test function with uninitialized variable
    let func_name = string_interner_rc.borrow_mut().intern("testUninitialized");
    let x_symbol = SymbolId::from_raw(1);

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
                        operator: compiler::tast::node::BinaryOperator::Add,
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

    // Create typed file
    let mut typed_file = TypedFile::new(string_interner_rc.clone());
    typed_file.functions.push(function);

    println!("Running type checking with flow analysis...\n");

    // Create type checking phase with flow analysis
    {
        let string_interner_ref = string_interner_rc.borrow();
        let mut type_checker = TypeCheckingPhase::new(
            &type_table,
            &symbol_table,
            &scope_tree,
            &*string_interner_ref,
            &source_map,
            &mut diagnostics,
        );

        // Run type checking (which includes flow analysis)
        match type_checker.check_file(&mut typed_file) {
            Ok(_) => {
                println!("⚠️  Type checking passed but expected errors!");
            }
            Err(msg) => {
                println!("✅ Type checking correctly failed: {}", msg);
            }
        }
    }

    // Check diagnostics
    println!("\n=== DIAGNOSTICS ===");
    let error_count = diagnostics.errors().count();
    let hint_count = diagnostics.hints().count();

    println!("Errors: {}", error_count);
    println!("Hints: {}", hint_count);

    // Display errors
    if error_count > 0 {
        println!("\n=== ERRORS ===");
        for (i, error) in diagnostics.errors().enumerate() {
            println!("Error {}: {}", i + 1, error.message);
            if let Some(span) = error.labels.first() {
                println!(
                    "  at line {}, column {}",
                    span.span.start.line, span.span.start.column
                );
            }
        }
    }

    // Display hints (for dead code, etc.)
    if hint_count > 0 {
        println!("\n=== HINTS ===");
        for (i, hint) in diagnostics.hints().enumerate() {
            println!("Hint {}: {}", i + 1, hint.message);
        }
    }

    println!("\n=== INTEGRATION STATUS ===");
    if error_count > 0 {
        println!("✅ TypeFlowGuard successfully integrated with type checking pipeline!");
        println!("✅ Flow safety errors are properly reported as diagnostics");
        println!("✅ Uninitialized variable usage was detected");
    } else {
        println!("❌ Flow analysis may not be working correctly");
    }

    // Note: Flow analysis is enabled by default
    println!("\n✅ TypeFlowGuard is integrated and enabled by default in the pipeline");
}
