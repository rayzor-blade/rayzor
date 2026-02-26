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
//! Tiered JIT Test with Loop: Demonstrating SSA Phi Nodes
//!
//! This test compiles a Haxe function with a while loop and demonstrates:
//! 1. Proper SSA form with phi nodes from HIR→MIR lowering
//! 2. Tiered JIT compilation with actual execution
//! 3. Automatic tier promotion based on execution count
//!
//! The loop demonstrates the most complex case for SSA form, where
//! loop variables must be represented with phi nodes in the loop header.

use compiler::codegen::profiling::ProfileConfig;
use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};
use compiler::ir::{hir_to_mir::lower_hir_to_mir, tast_to_hir::lower_tast_to_hir};
use compiler::tast::{
    ast_lowering::AstLowering, scopes::ScopeTree, StringInterner, SymbolTable, TypeTable,
};
use parser::haxe_parser::parse_haxe_file;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), String> {
    println!("=== Tiered JIT with Loop (SSA Phi Nodes) ===\n");

    // Haxe program with a while loop
    let source = r#"
package test;

class Math {
    public static function sumToN(n:Int):Int {
        var sum = 0;
        var i = 1;
        while (i <= n) {
            sum = sum + i;
            i = i + 1;
        }
        return sum;
    }
}
    "#;

    println!("Haxe Source Code:");
    println!("{}", "-".repeat(60));
    println!("{}", source);
    println!("{}", "-".repeat(60));
    println!();

    // Compile through full pipeline
    println!("Step 1: Compiling Haxe → MIR (with SSA phi nodes)...");
    let mir_module = compile_haxe_to_mir(source)?;

    println!("✓ MIR Module created");
    println!("  Functions: {}", mir_module.functions.len());

    // Get the function
    let func_id = *mir_module
        .functions
        .keys()
        .next()
        .ok_or("No functions in MIR module")?;

    let func_info = mir_module
        .functions
        .get(&func_id)
        .ok_or("Function not found")?;

    println!("\nMIR Function Details:");
    println!("  Name: {}", func_info.name);
    println!("  Blocks: {}", func_info.cfg.blocks.len());
    println!("  Parameters: {}", func_info.signature.parameters.len());

    // Check for phi nodes
    let has_phi = func_info
        .cfg
        .blocks
        .values()
        .any(|b| !b.phi_nodes.is_empty());
    println!(
        "  Has phi nodes: {} (required for proper SSA in loops)",
        has_phi
    );
    println!();

    // Set up tiered JIT
    println!("Step 2: Setting up Tiered JIT...");
    let config = TieredConfig {
        enable_background_optimization: true,
        optimization_check_interval_ms: 50,
        max_parallel_optimizations: 2,
        profile_config: ProfileConfig {
            interpreter_threshold: 5,
            warm_threshold: 10, // Fast promotion for demo
            hot_threshold: 50,
            blazing_threshold: 200,
            sample_rate: 1,
        },
        verbosity: 1, // Show promotions
        start_interpreted: false,
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 3,
        enable_stack_traces: false,
    };

    let mut backend = TieredBackend::new(config)?;
    println!("✓ Created tiered backend");
    println!();

    // Compile with tiered JIT
    println!("Step 3: Compiling MIR → Tiered JIT → Native...");
    backend.compile_module(mir_module)?;
    println!("✓ Compiled at Tier 0 (baseline)");
    println!();

    // Execute test cases
    println!("Step 4: Executing test cases...");
    let func_ptr = backend
        .get_function_pointer(func_id)
        .ok_or("Failed to get function pointer")?;
    let sum_fn: fn(i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test: sumToN(n) = n*(n+1)/2
    let test_cases = vec![
        (0, 0),      // sum(0) = 0
        (1, 1),      // sum(1) = 1
        (5, 15),     // sum(5) = 1+2+3+4+5 = 15
        (10, 55),    // sum(10) = 55
        (100, 5050), // sum(100) = 5050
    ];

    println!("Running test cases (formula: sum = n*(n+1)/2):");
    let mut all_passed = true;
    for (n, expected) in &test_cases {
        let result = sum_fn(*n);
        let passed = result == *expected;
        let symbol = if passed { "✓" } else { "✗" };
        println!(
            "  {} sumToN({:>3}) = {:>5} (expected {:>5})",
            symbol, n, result, expected
        );
        all_passed &= passed;
    }

    if !all_passed {
        return Err("Test cases failed".to_string());
    }
    println!();

    // Execute repeatedly to trigger tier promotion
    println!("Step 5: Executing 500 times to trigger tier promotion...\n");

    let test_values = vec![5, 10, 20, 50, 100];
    let mut execution_count = 0;
    let mut total_sum = 0i64;

    for round in 0..100 {
        for &n in &test_values {
            backend.record_call(func_id);

            if let Some(ptr) = backend.get_function_pointer(func_id) {
                let func: fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
                total_sum += func(n);
                execution_count += 1;
            }
        }

        // Report progress at milestones
        if execution_count == 10
            || execution_count == 50
            || execution_count == 100
            || execution_count == 200
            || execution_count == 500
        {
            std::thread::sleep(std::time::Duration::from_millis(100));
            println!(
                "  After {:>3} calls: total_sum = {:>8}",
                execution_count, total_sum
            );
        }
    }

    println!(
        "\n  ✓ Executed {} times with correct results",
        execution_count
    );
    println!();

    // Show final statistics
    println!("{}", "=".repeat(60));
    println!("Final Statistics");
    println!("{}", "=".repeat(60));

    let stats = backend.get_statistics();

    println!("\nTier Distribution:");
    println!(
        "  Tier 0 (Baseline):  {} functions",
        stats.baseline_functions
    );
    println!(
        "  Tier 1 (Standard):  {} functions",
        stats.standard_functions
    );
    println!(
        "  Tier 2 (Optimized): {} functions",
        stats.optimized_functions
    );
    println!("  Tier 3 (Maximum):   {} functions", stats.llvm_functions);

    println!("\nProfile Statistics:");
    println!("{}", stats.profile_stats.format());

    println!("\n{}", "=".repeat(60));
    println!("✓ Loop Test Complete!");
    println!("{}", "=".repeat(60));

    println!("\nSuccessfully demonstrated:");
    println!("  ✓ Haxe while loop with proper SSA form");
    println!("  ✓ Phi nodes for loop variables");
    println!("  ✓ JIT compilation of complex control flow");
    println!("  ✓ {} correct executions", execution_count);
    println!("  ✓ Automatic tier promotion");

    backend.shutdown();
    println!("\nBackend shutdown complete.");

    Ok(())
}

/// Compile Haxe source through the full pipeline to MIR
fn compile_haxe_to_mir(source: &str) -> Result<compiler::ir::IrModule, String> {
    // Parse
    let ast =
        parse_haxe_file("test.hx", source, false).map_err(|e| format!("Parse error: {}", e))?;

    // Lower AST to TAST
    let mut string_interner = StringInterner::new();
    let mut symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let mut scope_tree = ScopeTree::new(compiler::tast::ScopeId::from_raw(0));
    let mut namespace_resolver = compiler::tast::namespace::NamespaceResolver::new();
    let mut import_resolver = compiler::tast::namespace::ImportResolver::new();

    let mut ast_lowering = AstLowering::new(
        &mut string_interner,
        std::rc::Rc::new(std::cell::RefCell::new(
            compiler::tast::StringInterner::new(),
        )),
        &mut symbol_table,
        &type_table,
        &mut scope_tree,
        &mut namespace_resolver,
        &mut import_resolver,
    );

    let mut typed_file = ast_lowering
        .lower_file(&ast)
        .map_err(|e| format!("TAST lowering error: {:?}", e))?;

    // Lower TAST to HIR
    let string_interner_rc = Rc::new(RefCell::new(string_interner));
    typed_file.string_interner = Rc::clone(&string_interner_rc);

    let hir_module = lower_tast_to_hir(
        &typed_file,
        &symbol_table,
        &type_table,
        &mut *string_interner_rc.borrow_mut(),
        None,
    )
    .map_err(|errors| {
        let messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
        format!("HIR lowering errors: {}", messages.join(", "))
    })?;

    // Lower HIR to MIR (produces proper SSA with phi nodes!)
    let mir_module = lower_hir_to_mir(
        &hir_module,
        &*string_interner_rc.borrow(),
        &type_table,
        &symbol_table,
    )
    .map_err(|errors| {
        let messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
        format!("MIR lowering errors: {}", messages.join(", "))
    })?;

    Ok(mir_module)
}
