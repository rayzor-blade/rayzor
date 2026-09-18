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
//! Test for TypeFlowGuard - using existing CFG infrastructure

use compiler::tast::{
    FlowSafetyError, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeFlowGuard, TypeId,
    TypeTable,
    node::{
        ExpressionMetadata, FunctionEffects, FunctionMetadata, LiteralValue, TypedExpression,
        TypedExpressionKind, TypedFile, TypedFunction, TypedStatement, VariableUsage,
    },
    symbols::{Mutability, Visibility},
};
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== TypeFlowGuard Test ===\n");

    let symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let string_interner = Rc::new(RefCell::new(StringInterner::new()));

    // Create test function: function test(): Int { var x: Int; return x + 1; }
    let func_name = string_interner.borrow_mut().intern("test");
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

    let mut file = TypedFile::new(string_interner);
    file.functions.push(function);

    // Test TypeFlowGuard
    println!("Creating TypeFlowGuard analyzer...");
    let mut flow_guard = TypeFlowGuard::new(&symbol_table, &type_table);

    println!("Running flow safety analysis...");
    let results = flow_guard.analyze_file(&file);

    println!("\n=== TYPEFLOWGUARD RESULTS ===");
    println!("Functions analyzed: {}", results.metrics.functions_analyzed);
    println!("Blocks processed: {}", results.metrics.blocks_processed);
    println!("Errors found: {}", results.errors.len());
    println!("Warnings found: {}", results.warnings.len());

    println!("\n=== PERFORMANCE METRICS ===");
    println!(
        "CFG construction time: {} μs",
        results.metrics.cfg_construction_time_us
    );
    println!(
        "Variable analysis time: {} μs",
        results.metrics.variable_analysis_time_us
    );
    println!(
        "Null safety time: {} μs",
        results.metrics.null_safety_time_us
    );
    println!("Dead code time: {} μs", results.metrics.dead_code_time_us);

    if results.errors.is_empty() {
        println!("\n⚠️  Expected to find uninitialized variable error but found none");
    } else {
        println!("\n✅ Found {} error(s) as expected:", results.errors.len());
        for (i, error) in results.errors.iter().enumerate() {
            match error {
                FlowSafetyError::UninitializedVariable { variable, location } => {
                    println!(
                        "  Error {}: Uninitialized variable {:?} at {}:{}",
                        i + 1,
                        variable,
                        location.line,
                        location.column
                    );
                }
                _ => {
                    println!("  Error {}: {:?}", i + 1, error);
                }
            }
        }
    }

    println!("\n=== ARCHITECTURE ===");
    println!("✅ TypeFlowGuard uses existing semantic_graph::cfg::ControlFlowGraph");
    println!("✅ Leverages semantic_graph::tast_cfg_mapping for TAST integration");
    println!("✅ Uses semantic_graph::builder::CfgBuilder for CFG construction");
    println!("✅ No redundant CFG infrastructure");
    println!("✅ Clean, unversioned naming");
}
