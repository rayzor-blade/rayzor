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
//! Test loading and executing a .rzb bundle
//!
//! Compares bundle loading speed vs full compilation.
//!
//! Usage:
//!   # First create a bundle
//!   cargo run --release --package compiler --bin preblade -- --bundle /tmp/test.rzb /tmp/TestMain.hx
//!   # Then test loading it
//!   cargo run --release --package compiler --example test_bundle_loading -- /tmp/test.rzb

use compiler::codegen::profiling::ProfileConfig;
use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::blade::{load_bundle, RayzorBundle};
use compiler::ir::IrFunctionId;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: test_bundle_loading <bundle.rzb>");
        eprintln!();
        eprintln!("First create a bundle:");
        eprintln!(
            "  cargo run --release --package compiler --bin preblade -- --bundle app.rzb Main.hx"
        );
        std::process::exit(1);
    }

    let bundle_path = &args[1];

    println!("=== Bundle Loading Test ===\n");

    // Test 1: Load bundle
    println!("Test 1: Load bundle from disk");
    let t0 = Instant::now();
    let bundle = match load_bundle(bundle_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to load bundle: {}", e);
            std::process::exit(1);
        }
    };
    let load_time = t0.elapsed();
    println!("  Bundle loaded in {:?}", load_time);
    println!("  Modules: {}", bundle.module_count());
    println!(
        "  Entry: {}::{}",
        bundle
            .entry_module()
            .map(|m| m.name.as_str())
            .unwrap_or("?"),
        bundle.entry_function()
    );

    // Test 2: Execute via interpreter
    println!("\nTest 2: Execute via interpreter");
    match execute_bundle_interpreted(&bundle) {
        Ok(exec_time) => println!("  Execution time: {:?}", exec_time),
        Err(e) => println!("  Execution failed: {:?}", e),
    }

    // Test 3: Compare with full compilation
    println!("\nTest 3: Compare with full compilation");
    let source = std::fs::read_to_string(bundle_path.replace(".rzb", ".hx")).unwrap_or_else(|_| {
        // Use a simple test source if original not found
        r#"
class Main {
    static function main() {
        trace("Hello!");
    }
}
"#
        .to_string()
    });

    let t1 = Instant::now();
    match compile_and_run(&source) {
        Ok(()) => {
            let compile_time = t1.elapsed();
            println!("  Full compilation + execution: {:?}", compile_time);
            println!();
            println!("  Bundle load: {:?}", load_time);
            println!(
                "  Speedup: {:.1}x faster with bundle",
                compile_time.as_micros() as f64 / load_time.as_micros() as f64
            );
        }
        Err(e) => println!("  Compilation failed: {:?}", e),
    }

    println!("\n=== Test Complete ===");
}

fn execute_bundle_interpreted(bundle: &RayzorBundle) -> Result<std::time::Duration, String> {
    let t0 = Instant::now();

    // Get runtime symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Create tiered backend in interpreted mode
    let config = TieredConfig {
        profile_config: ProfileConfig {
            interpreter_threshold: 1000,
            warm_threshold: 10000,
            hot_threshold: 100000,
            blazing_threshold: 1000000,
            sample_rate: 1,
        },
        enable_background_optimization: false,
        optimization_check_interval_ms: 1000,
        max_parallel_optimizations: 1,
        verbosity: 0,
        start_interpreted: true,
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 0,
        enable_stack_traces: false,
    };

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)
        .map_err(|e| format!("Failed to create backend: {}", e))?;

    // Get entry module from bundle
    let entry_module = bundle.entry_module().ok_or("No entry module in bundle")?;

    // Find main function
    let main_func_id = entry_module
        .functions
        .iter()
        .find(|(_, f)| {
            f.name == bundle.entry_function()
                || f.name == "main"
                || f.name == "Main_main"
                || f.name.ends_with("_main")
        })
        .map(|(id, _)| *id)
        .ok_or("Main function not found")?;

    // Load module into backend
    backend
        .compile_module(entry_module.clone())
        .map_err(|e| format!("Failed to load module: {}", e))?;

    // Execute
    backend
        .execute_function(main_func_id, vec![])
        .map_err(|e| format!("Execution failed: {:?}", e))?;

    Ok(t0.elapsed())
}

fn compile_and_run(source: &str) -> Result<(), String> {
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;
    unit.add_file(source, "main.hx")
        .map_err(|e| format!("Failed to add file: {}", e))?;
    unit.lower_to_tast()
        .map_err(|errors| format!("TAST lowering failed: {:?}", errors))?;

    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Get runtime symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let config = TieredConfig {
        profile_config: ProfileConfig {
            interpreter_threshold: 1000,
            warm_threshold: 10000,
            hot_threshold: 100000,
            blazing_threshold: 1000000,
            sample_rate: 1,
        },
        enable_background_optimization: false,
        optimization_check_interval_ms: 1000,
        max_parallel_optimizations: 1,
        verbosity: 0,
        start_interpreted: true,
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 0,
        enable_stack_traces: false,
    };

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)
        .map_err(|e| format!("Failed to create backend: {}", e))?;

    // Find and execute main
    for module in mir_modules.iter().rev() {
        if let Some((func_id, _)) = module
            .functions
            .iter()
            .find(|(_, f)| f.name == "main" || f.name == "Main_main" || f.name.ends_with("_main"))
        {
            backend
                .compile_module((**module).clone())
                .map_err(|e| format!("Failed to load module: {}", e))?;
            backend
                .execute_function(*func_id, vec![])
                .map_err(|e| format!("Execution failed: {:?}", e))?;
            return Ok(());
        }
    }

    Err("No main function found".to_string())
}
