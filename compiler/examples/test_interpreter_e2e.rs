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
//! End-to-end test: Haxe source -> MIR -> Interpreter execution
//!
//! Demonstrates Phase 0 (Interpreter) instant startup with real Haxe code.
//!
//! Run with: cargo run --package compiler --example test_interpreter_e2e

use compiler::codegen::profiling::ProfileConfig;
use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};
use compiler::codegen::InterpValue;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::IrModule;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("=== Interpreter E2E Test ===\n");

    // Test 1: Simple arithmetic
    println!("Test 1: Simple arithmetic");
    test_simple_arithmetic();

    // Test 2: Control flow (if/else)
    println!("\nTest 2: Control flow");
    test_control_flow();

    // Test 3: For loop
    println!("\nTest 3: For loop");
    test_for_loop();

    // Test 4: Compare interpreter vs JIT startup time
    println!("\nTest 4: Startup time comparison");
    test_startup_comparison();

    // Test 5: Compare fast vs standard compilation
    println!("\nTest 5: Fast vs Standard compilation");
    test_fast_compilation();

    println!("\n=== All E2E Tests Complete ===");
}

fn test_simple_arithmetic() {
    let source = r#"
class Main {
    static function main() {
        var a = 10;
        var b = 20;
        var sum = a + b;
        trace(sum);

        var product = a * b;
        trace(product);
    }
}
"#;

    match compile_and_run_interpreted(source, "arithmetic") {
        Ok(()) => println!("  Simple arithmetic: PASSED"),
        Err(e) => println!("  Simple arithmetic: FAILED - {}", e),
    }
}

fn test_control_flow() {
    let source = r#"
class Main {
    static function main() {
        var x = 5;
        if (x > 3) {
            trace(1);  // Should print 1
        } else {
            trace(0);
        }

        var y = 2;
        if (y > 10) {
            trace(100);
        } else {
            trace(2);  // Should print 2
        }
    }
}
"#;

    match compile_and_run_interpreted(source, "control_flow") {
        Ok(()) => println!("  Control flow: PASSED"),
        Err(e) => println!("  Control flow: FAILED - {}", e),
    }
}

fn test_for_loop() {
    let source = r#"
class Main {
    static function main() {
        var sum = 0;
        for (i in 0...5) {
            sum = sum + i;
        }
        trace(sum);  // Should print 10 (0+1+2+3+4)
    }
}
"#;

    match compile_and_run_interpreted(source, "for_loop") {
        Ok(()) => println!("  For loop: PASSED"),
        Err(e) => println!("  For loop: FAILED - {}", e),
    }
}

fn test_startup_comparison() {
    let source = r#"
class Main {
    static function main() {
        trace(42);
    }
}
"#;

    // Test interpreter startup
    let t0 = Instant::now();
    let _ = compile_and_run_interpreted(source, "interp_startup");
    let interp_time = t0.elapsed();

    // Test JIT startup
    let t1 = Instant::now();
    let _ = compile_and_run_jit(source, "jit_startup");
    let jit_time = t1.elapsed();

    println!("  Interpreter total time: {:?}", interp_time);
    println!("  JIT total time: {:?}", jit_time);

    if interp_time < jit_time {
        println!(
            "  Interpreter was {:.1}x faster to start",
            jit_time.as_micros() as f64 / interp_time.as_micros() as f64
        );
    }
}

fn test_fast_compilation() {
    let source = r#"
class Main {
    static function main() {
        trace(42);
    }
}
"#;

    // Test standard compilation (eager symbol loading)
    println!("  Standard compilation (eager stdlib):");
    let t0 = Instant::now();
    let _ = compile_to_mir(source, "standard");
    let standard_time = t0.elapsed();

    // Test fast compilation (lazy stdlib loading)
    println!("  Fast compilation (lazy stdlib):");
    let t1 = Instant::now();
    let _ = compile_to_mir_fast(source, "fast");
    let fast_time = t1.elapsed();

    println!("\n  Standard compile: {:?}", standard_time);
    println!("  Fast compile: {:?}", fast_time);

    if fast_time < standard_time {
        let speedup = standard_time.as_micros() as f64 / fast_time.as_micros() as f64;
        println!("  Fast mode is {:.1}x faster!", speedup);
    }
}

fn compile_and_run_interpreted(source: &str, name: &str) -> Result<(), String> {
    let t0 = Instant::now();

    // Compile Haxe to MIR
    let mir_modules = compile_to_mir(source, name)?;
    let compile_time = t0.elapsed();

    let t1 = Instant::now();

    // Get runtime symbols for FFI
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Create tiered backend in interpreted mode
    let config = TieredConfig {
        profile_config: ProfileConfig {
            interpreter_threshold: 1000, // High threshold - stay interpreted
            warm_threshold: 10000,
            hot_threshold: 100000,
            blazing_threshold: 1000000,
            sample_rate: 1,
        },
        enable_background_optimization: false,
        optimization_check_interval_ms: 1000,
        max_parallel_optimizations: 1,
        verbosity: 0,
        start_interpreted: true, // Start in interpreter mode
        bailout_strategy: compiler::codegen::BailoutStrategy::Quick,
        max_tier_promotions: 0,
        enable_stack_traces: false,
    };

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)
        .map_err(|e| format!("Failed to create backend: {}", e))?;

    // Find the module containing main first, then load only that one
    // (since TieredBackend only stores one module at a time)
    let main_module = mir_modules
        .iter()
        .rev()
        .find(|m| {
            m.functions
                .values()
                .any(|f| f.name == "main" || f.name == "Main_main" || f.name.ends_with("_main"))
        })
        .ok_or("No module with main function found")?;

    let main_func_id = find_main_function(&main_module).ok_or("Main function not found")?;

    // Load the main module
    backend
        .compile_module((**main_module).clone())
        .map_err(|e| format!("Failed to load module: {}", e))?;

    let setup_time = t1.elapsed();

    let t2 = Instant::now();

    // Execute main
    let result = backend.execute_function(main_func_id, vec![]);
    match result {
        Ok(_) => {
            let exec_time = t2.elapsed();
            eprintln!(
                "  [{}] compile={:?}, setup={:?}, exec={:?}",
                name, compile_time, setup_time, exec_time
            );
            Ok(())
        }
        Err(e) => Err(format!("Execution failed: {:?}", e)),
    }
}

fn compile_and_run_jit(source: &str, name: &str) -> Result<(), String> {
    use compiler::codegen::CraneliftBackend;

    let t0 = Instant::now();

    // Compile Haxe to MIR
    let mir_modules = compile_to_mir(source, name)?;
    let compile_time = t0.elapsed();

    let t1 = Instant::now();

    // Get runtime symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Create Cranelift backend
    let mut backend = CraneliftBackend::with_symbols(&symbols_ref)
        .map_err(|e| format!("Failed to create backend: {}", e))?;

    // Compile all modules
    for module in &mir_modules {
        backend
            .compile_module(&module)
            .map_err(|e| format!("Failed to compile module: {}", e))?;
    }

    let jit_time = t1.elapsed();

    let t2 = Instant::now();

    // Execute main
    for module in mir_modules.iter().rev() {
        if backend.call_main(&module).is_ok() {
            let exec_time = t2.elapsed();
            eprintln!(
                "  [{}] compile={:?}, jit={:?}, exec={:?}",
                name, compile_time, jit_time, exec_time
            );
            return Ok(());
        }
    }

    Err("Failed to execute main".to_string())
}

fn compile_to_mir(source: &str, name: &str) -> Result<Vec<Arc<IrModule>>, String> {
    compile_to_mir_with_config(source, name, CompilationConfig::default())
}

fn compile_to_mir_fast(source: &str, name: &str) -> Result<Vec<Arc<IrModule>>, String> {
    compile_to_mir_with_config(source, name, CompilationConfig::fast())
}

fn compile_to_mir_with_config(
    source: &str,
    name: &str,
    config: CompilationConfig,
) -> Result<Vec<Arc<IrModule>>, String> {
    let t0 = Instant::now();
    let mut unit = CompilationUnit::new(config);

    // Load stdlib
    let t1 = Instant::now();
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;
    let stdlib_time = t1.elapsed();

    // Add source file
    let t2 = Instant::now();
    unit.add_file(source, &format!("{}.hx", name))
        .map_err(|e| format!("Failed to add file: {}", e))?;
    let add_file_time = t2.elapsed();

    // Lower to TAST
    let t3 = Instant::now();
    unit.lower_to_tast()
        .map_err(|errors| format!("TAST lowering failed: {:?}", errors))?;
    let tast_time = t3.elapsed();

    // Get MIR modules
    let t4 = Instant::now();
    let mir_modules = unit.get_mir_modules();
    let mir_time = t4.elapsed();

    eprintln!(
        "    [compile breakdown] stdlib={:?}, parse={:?}, tast={:?}, mir={:?}, total={:?}",
        stdlib_time,
        add_file_time,
        tast_time,
        mir_time,
        t0.elapsed()
    );

    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    Ok(mir_modules)
}

fn find_main_function(module: &IrModule) -> Option<compiler::ir::IrFunctionId> {
    for (func_id, func) in &module.functions {
        if func.name == "main" || func.name == "Main_main" || func.name.ends_with("_main") {
            return Some(*func_id);
        }
    }
    None
}
