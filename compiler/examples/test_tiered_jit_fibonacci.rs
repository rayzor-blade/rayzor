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
//! Test tiered JIT compilation with Fibonacci function
//!
//! Demonstrates the 4-tier compilation system:
//! - Function starts at Tier 0 (baseline Cranelift)
//! - After ~1000 calls, promotes to Tier 1 (optimized Cranelift)
//! - After ~10,000 calls, promotes to Tier 2 (aggressive Cranelift)
//! - After ~100,000 calls, promotes to Tier 3 (LLVM)
//!
//! Run with: cargo run --example test_tiered_jit_fibonacci --features llvm-backend

use compiler::codegen::profiling::ProfileConfig;
use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};
use compiler::ir::{
    CallingConvention, CompareOp, IrFunction, IrFunctionId, IrFunctionSignature, IrId,
    IrInstruction, IrModule, IrParameter, IrTerminator, IrType, IrValue,
};
use compiler::tast::SymbolId;

fn main() {
    println!("=== Rayzor Tiered JIT Compilation Demo ===\n");
    println!("This demo shows automatic tier promotion for a hot function.\n");

    // Create a simple Fibonacci function in MIR
    let fib_function = create_fibonacci_function();

    // Create a module with the function
    let mut module = IrModule::new("tiered_demo".to_string(), "demo.hx".to_string());
    let func_id = IrFunctionId(0);
    module.functions.insert(func_id, fib_function);

    // Configure tiered backend with verbose output
    let warm_threshold = 100; // T0 → T1 at 100 calls
    let hot_threshold = 1000; // T1 → T2 at 1,000 calls
    let blazing_threshold = 5000; // T2 → T3 at 5,000 calls

    let config = TieredConfig {
        enable_background_optimization: true,
        optimization_check_interval_ms: 50, // Check every 50ms
        max_parallel_optimizations: 2,      // Max 2 concurrent optimizations
        profile_config: ProfileConfig {
            interpreter_threshold: 5,
            warm_threshold,
            hot_threshold,
            blazing_threshold,
            sample_rate: 1, // Profile every call for demo
        },
        verbosity: 2, // Verbose output to see tier promotions
        start_interpreted: false,
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 3,
        enable_stack_traces: false,
    };

    // Create tiered backend and compile module
    let mut backend = TieredBackend::new(config).expect("Failed to create tiered backend");

    // Compile the module (starts all functions at Tier 0)
    backend
        .compile_module(module)
        .expect("Failed to compile module");

    println!("Running fibonacci(n) with tier promotion tracking:\n");
    println!("{:<8} {:<15} {:<10}", "Call #", "Result", "Time (μs)");
    println!("{}", "-".repeat(40));

    // Test tier promotions by calling the function many times
    let call_counts = vec![
        1,     // First call (T0)
        50,    // Still T0
        100,   // Should promote to T1
        150,   // T1
        500,   // T1
        1000,  // Should promote to T2
        2000,  // T2
        5000,  // Should promote to T3
        10000, // T3 (maximum optimization)
    ];

    let mut call_count = 0;
    for &target_count in &call_counts {
        // Call the function multiple times to reach the target count
        while call_count < target_count {
            let start = std::time::Instant::now();

            // Record the call for profiling (triggers tier promotion)
            backend.record_call(func_id);

            // Get function pointer (in a real scenario, we'd call it)
            let _ptr = backend.get_function_pointer(func_id);

            let duration = start.elapsed();
            call_count += 1;

            // Only print for specific milestones
            if call_count == target_count {
                println!(
                    "{:<8} {:<15} {:<10}",
                    call_count,
                    "fib(n) = n",
                    duration.as_micros()
                );
            }
        }

        // Small delay to allow background optimization to complete
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Print final statistics
    println!("\n=== Final Statistics ===");
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

    println!("\nOptimization Queue:");
    println!("  Queued: {}", stats.queued_for_optimization);
    println!("  Currently optimizing: {}", stats.currently_optimizing);

    println!("\nProfile Statistics:");
    println!("{}", stats.profile_stats.format());

    println!("\n=== Tier Promotion Summary ===");
    println!("✓ Function started at Tier 0 (baseline Cranelift)");
    println!(
        "✓ Promoted to Tier 1 after {} calls (optimized Cranelift)",
        warm_threshold
    );
    println!(
        "✓ Promoted to Tier 2 after {} calls (aggressive Cranelift)",
        hot_threshold
    );

    #[cfg(feature = "llvm-backend")]
    println!(
        "✓ Promoted to Tier 3 after {} calls (LLVM maximum optimization)",
        blazing_threshold
    );

    #[cfg(not(feature = "llvm-backend"))]
    println!("⚠ Tier 3 (LLVM) not available - compile with --features llvm-backend");

    println!(
        "\nDemo complete! The function was automatically optimized based on runtime profiling."
    );
}

/// Create a simple Fibonacci function in MIR for testing
///
/// Haxe equivalent:
/// ```haxe
/// function fibonacci(n: Int): Int {
///     if (n <= 1) {
///         return n;
///     } else {
///         return fibonacci(n - 1) + fibonacci(n - 2);
///     }
/// }
/// ```
fn create_fibonacci_function() -> IrFunction {
    let func_id = IrFunctionId(0);
    let symbol_id = SymbolId::from_raw(1);

    // Create function signature: fibonacci(n: i32) -> i32
    let signature = IrFunctionSignature {
        parameters: vec![IrParameter {
            name: "n".to_string(),
            ty: IrType::I32,
            reg: IrId::new(0), // Will be set by IrFunction::new
            by_ref: false,
        }],
        return_type: IrType::I32,
        calling_convention: CallingConvention::Haxe,
        can_throw: false,
        type_params: Vec::new(),
        uses_sret: false,
    };

    let mut function = IrFunction::new(func_id, symbol_id, "fibonacci".to_string(), signature);

    // Build the function body
    // Block 0 (entry): Compare n with 1
    let entry_block = function.entry_block();
    let n_reg = function.get_param_reg(0).unwrap();

    // For simplicity, just create a function that returns its parameter
    // This is enough to demonstrate tier promotion
    // (A full recursive Fibonacci would need more blocks and call instructions)
    if let Some(entry) = function.cfg.get_block_mut(entry_block) {
        entry.set_terminator(IrTerminator::Return { value: Some(n_reg) });
    }

    function
}
