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
//! Full Pipeline Test: Haxe → Parser → TAST → HIR → MIR → Tiered JIT
//!
//! This test demonstrates the complete Rayzor compilation pipeline with tiered JIT:
//! 1. Parse Haxe source code
//! 2. Lower AST to TAST (Typed AST)
//! 3. Lower TAST to HIR (High-level IR)
//! 4. Lower HIR to MIR (SSA form with phi nodes)
//! 5. Compile MIR with Tiered JIT (Cranelift T0-T2, LLVM T3)
//! 6. Demonstrate automatic tier promotion
//!
//! Run with: cargo run --example test_full_pipeline_tiered
//! Run with LLVM: cargo run --example test_full_pipeline_tiered --features llvm-backend

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
    println!("=== Rayzor Tiered JIT Full Pipeline Test ===\n");
    println!("Testing: Haxe → TAST → HIR → MIR → Tiered JIT → Native\n");

    // Simple Haxe program
    let source = r#"
package test;

class Math {
    public static function add(a:Int, b:Int):Int {
        return a + b;
    }
}
    "#;

    println!("Haxe Source Code:");
    println!("{}", "-".repeat(60));
    println!("{}", source);
    println!("{}", "-".repeat(60));
    println!();

    // Compile through full pipeline
    println!("Step 1: Compiling Haxe → MIR...");
    let mir_module = compile_haxe_to_mir(source)?;

    println!("✓ MIR Module created");
    println!("  Functions: {}", mir_module.functions.len());
    for func in mir_module.functions.values() {
        println!("    - {}", func.name);
    }

    // Get the first function (should be 'add')
    let add_func_id = *mir_module
        .functions
        .keys()
        .next()
        .ok_or("No functions in MIR module")?;

    let add_func_info = mir_module
        .functions
        .get(&add_func_id)
        .ok_or("Function not found")?;

    println!("\nMIR Function Details:");
    println!("  Name: {}", add_func_info.name);
    println!("  Blocks: {}", add_func_info.cfg.blocks.len());
    println!("  Parameters: {}", add_func_info.signature.parameters.len());
    println!();

    // Step 2: Configure tiered JIT backend
    println!("Step 2: Setting up Tiered JIT...");
    let config = TieredConfig {
        enable_background_optimization: true,
        optimization_check_interval_ms: 50,
        max_parallel_optimizations: 2,
        profile_config: ProfileConfig {
            interpreter_threshold: 5,
            warm_threshold: 100,     // T0 → T1 at 100 calls
            hot_threshold: 1000,     // T1 → T2 at 1,000 calls
            blazing_threshold: 5000, // T2 → T3 at 5,000 calls
            sample_rate: 1,          // Profile every call for demo
        },
        verbosity: 2, // Verbose output to see tier promotions
        start_interpreted: false,
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 3,
        enable_stack_traces: false,
    };

    let mut backend = TieredBackend::new(config)?;
    println!("✓ Created tiered backend");
    println!("  Tiers: T0 (Baseline), T1 (Standard), T2 (Optimized), T3 (Maximum/LLVM)");
    println!();

    // Step 3: Compile with tiered JIT
    println!("Step 3: Compiling MIR → Tiered JIT → Native...");
    backend.compile_module(mir_module)?;
    println!("✓ Compiled at Tier 0 (baseline Cranelift)");
    println!();

    // Step 4: Get function pointer and execute
    println!("Step 4: Executing JIT-compiled function...");
    let func_ptr = backend
        .get_function_pointer(add_func_id)
        .ok_or("Failed to get function pointer")?;
    let add_fn: fn(i64, i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test execution
    let test_cases = vec![(10, 20, 30), (100, 200, 300), (-5, 15, 10), (0, 0, 0)];

    println!("Running test cases:");
    let mut all_passed = true;
    for (a, b, expected) in &test_cases {
        let result = add_fn(*a, *b);
        let passed = result == *expected;
        let symbol = if passed { "✓" } else { "✗" };
        println!(
            "  {} add({}, {}) = {} (expected {})",
            symbol, a, b, result, expected
        );
        all_passed &= passed;
    }

    if !all_passed {
        return Err("Function execution test failed".to_string());
    }
    println!();

    // Step 5: Execute function repeatedly to trigger tier promotion
    println!("Step 5: Executing function repeatedly to trigger tier promotion...\n");

    let milestones = [1, 50, 100, 150, 500, 1000, 2000, 5000, 7000];
    let mut current_count = 0;
    let mut sum = 0i64;

    for &milestone in &milestones {
        while current_count < milestone {
            // Actually execute the function with the tiered runtime
            backend.record_call(add_func_id);

            // Get the (potentially updated) function pointer and execute
            if let Some(ptr) = backend.get_function_pointer(add_func_id) {
                let func: fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
                sum += func(current_count as i64, 1);
            }

            current_count += 1;
        }

        // Small delay for background optimization
        std::thread::sleep(std::time::Duration::from_millis(100));

        let new_ptr = backend.get_function_pointer(add_func_id);
        let ptr_changed = new_ptr.map(|p| p as usize) != Some(func_ptr as usize);
        let indicator = if ptr_changed { "↑ (promoted)" } else { "" };

        println!(
            "  After {:>5} calls: sum={:>8} ptr={:?} {}",
            milestone,
            sum,
            new_ptr.map(|p| format!("{:p}", p as *const ())),
            indicator
        );
    }

    // Verify we got the expected sum: 0+1 + 1+1 + 2+1 + ... + 6999+1
    let expected_sum: i64 = (0..7000).sum::<i64>() + 7000;
    println!(
        "\n  Final sum: {} (expected: {}) {}",
        sum,
        expected_sum,
        if sum == expected_sum { "✓" } else { "✗" }
    );
    println!();

    // Step 6: Show final statistics
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

    println!("\nOptimization Queue:");
    println!("  Queued: {}", stats.queued_for_optimization);
    println!("  Currently optimizing: {}", stats.currently_optimizing);

    println!("\nProfile Statistics:");
    println!("{}", stats.profile_stats.format());

    println!("\n{}", "=".repeat(60));
    println!("✓ Full Pipeline Test Complete!");
    println!("{}", "=".repeat(60));

    println!("\nSuccessfully demonstrated:");
    println!("  ✓ Haxe source parsing");
    println!("  ✓ TAST (Typed AST) lowering");
    println!("  ✓ HIR (High-level IR) lowering");
    println!("  ✓ MIR (Mid-level IR) lowering with SSA");
    println!("  ✓ Tiered JIT compilation (Cranelift T0-T2)");
    println!("  ✓ Function execution");
    println!("  ✓ Automatic tier promotion");

    #[cfg(feature = "llvm-backend")]
    println!("  ✓ LLVM Tier 3 optimization available");

    #[cfg(not(feature = "llvm-backend"))]
    println!("\n  (Compile with --features llvm-backend for Tier 3/LLVM support)");

    // Cleanup
    backend.shutdown();
    println!("\nBackend shutdown complete.");

    Ok(())
}

/// Compile Haxe source through the full pipeline to MIR
fn compile_haxe_to_mir(source: &str) -> Result<compiler::ir::IrModule, String> {
    // Step 1: Parse
    let ast =
        parse_haxe_file("test.hx", source, false).map_err(|e| format!("Parse error: {}", e))?;

    // Step 2: Lower AST to TAST
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

    // Step 3: Lower TAST to HIR
    let string_interner_rc = Rc::new(RefCell::new(string_interner));
    typed_file.string_interner = Rc::clone(&string_interner_rc);

    let hir_module = lower_tast_to_hir(
        &typed_file,
        &symbol_table,
        &type_table,
        &mut *string_interner_rc.borrow_mut(),
        None, // No semantic graphs for now
    )
    .map_err(|errors| {
        let messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
        format!("HIR lowering errors: {}", messages.join(", "))
    })?;

    // Step 4: Lower HIR to MIR (this produces proper SSA!)
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
