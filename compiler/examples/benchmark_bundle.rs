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
#![allow(clippy::format_in_format_args)]
//! Benchmark: Bundle Loading vs Full Compilation
//!
//! This benchmark compares different execution paths:
//! 1. Full compilation + JIT execution
//! 2. Full compilation + Interpreter execution
//! 3. Bundle load + Interpreter execution
//! 4. Fast compilation + Interpreter execution
//!
//! Usage:
//!   # First create a bundle
//!   cargo run --release --package compiler --bin preblade -- --bundle /tmp/bench.rzb /tmp/BenchMain.hx
//!   # Then run benchmark
//!   cargo run --release --package compiler --example benchmark_bundle -- /tmp/bench.rzb

use compiler::codegen::profiling::ProfileConfig;
use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};
use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::blade::{load_bundle, RayzorBundle};
use compiler::ir::IrModule;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WARMUP_RUNS: usize = 2;
const BENCH_RUNS: usize = 5;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Rayzor Bundle & Interpreter Benchmark                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create test source if no bundle provided
    let (bundle_path, source) = if args.len() >= 2 {
        let bundle_path = args[1].clone();
        // Try to read corresponding .hx file
        let source_path = bundle_path.replace(".rzb", ".hx");
        let source = std::fs::read_to_string(&source_path).unwrap_or_else(|_| get_default_source());
        (Some(bundle_path), source)
    } else {
        println!("No bundle provided. Using in-memory benchmark.\n");
        println!("For full benchmark with bundle, run:");
        println!(
            "  cargo run --release --bin preblade -- --bundle /tmp/bench.rzb /tmp/BenchMain.hx"
        );
        println!("  cargo run --release --example benchmark_bundle -- /tmp/bench.rzb\n");
        (None, get_default_source())
    };

    // Run benchmarks
    println!(
        "Running {} warmup iterations, {} benchmark iterations...\n",
        WARMUP_RUNS, BENCH_RUNS
    );

    let mut results = Vec::new();

    // Benchmark 1: Full compilation + JIT
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 1: Full Compilation + JIT Execution");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let jit_result = bench_full_compile_jit(&source);
    results.push(("Full Compile + JIT", jit_result));

    // Benchmark 2: Full compilation + Interpreter
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 2: Full Compilation + Interpreter");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let interp_result = bench_full_compile_interp(&source);
    results.push(("Full Compile + Interp", interp_result));

    // Benchmark 3: Fast compilation + Interpreter
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 3: Fast Compilation + Interpreter");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let fast_result = bench_fast_compile_interp(&source);
    results.push(("Fast Compile + Interp", fast_result));

    // Benchmark 4: Bundle load + Interpreter (if bundle provided)
    if let Some(ref path) = bundle_path {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Benchmark 4: Bundle Load + Interpreter");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        let bundle_result = bench_bundle_interp(path);
        results.push(("Bundle + Interp", bundle_result));
    }

    // Print summary
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                ║");
    println!("╠════════════════════════════════════════════════════════════════╣");

    let baseline = results[0].1;
    for (name, time) in &results {
        let speedup = baseline.as_micros() as f64 / time.as_micros() as f64;
        let bar_len = (50.0 * (baseline.as_micros() as f64 / time.as_micros() as f64).min(10.0)
            / 10.0) as usize;
        let bar: String = "█".repeat(bar_len.min(50));

        println!("║ {:22} │ {:>10.2?} │ {:>5.1}x │", name, time, speedup);
        println!("║                        │ {} │", format!("{:50}", bar));
    }

    println!("╚════════════════════════════════════════════════════════════════╝");

    // Detailed breakdown
    if let Some(ref path) = bundle_path {
        println!("\n┌────────────────────────────────────────────────────────────────┐");
        println!("│                    DETAILED BREAKDOWN                          │");
        println!("├────────────────────────────────────────────────────────────────┤");
        detailed_breakdown(&source, path);
        println!("└────────────────────────────────────────────────────────────────┘");

        // Startup vs Execution breakdown for pre-bundled programs
        bundle_startup_execution_breakdown(path);
    }
}

fn get_default_source() -> String {
    r#"
class Main {
    static function main() {
        // Arithmetic test
        var sum = 0;
        for (i in 0...100) {
            sum = sum + i;
        }
        trace(sum);  // 4950

        // String test
        var msg = "Hello";
        msg = msg + " World";
        trace(msg);

        // Conditional test
        if (sum > 1000) {
            trace(1);
        } else {
            trace(0);
        }
    }
}
"#
    .to_string()
}

fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    symbols.iter().map(|(n, p)| (*n, *p)).collect()
}

fn create_interp_config() -> TieredConfig {
    TieredConfig {
        profile_config: ProfileConfig {
            interpreter_threshold: 1000000, // Stay in interpreter
            warm_threshold: 10000000,
            hot_threshold: 100000000,
            blazing_threshold: 1000000000,
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
    }
}

fn bench_full_compile_jit(source: &str) -> Duration {
    let symbols = get_runtime_symbols();
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let _ = run_full_compile_jit(source, &symbols);
    }

    // Benchmark
    for i in 0..BENCH_RUNS {
        let t0 = Instant::now();
        let result = run_full_compile_jit(source, &symbols);
        let elapsed = t0.elapsed();
        times.push(elapsed);

        match result {
            Ok(_) => println!("  Run {}: {:?}", i + 1, elapsed),
            Err(e) => println!("  Run {}: FAILED - {}", i + 1, e),
        }
    }

    median(&times)
}

fn run_full_compile_jit(source: &str, symbols: &[(&str, *const u8)]) -> Result<(), String> {
    let mut unit = CompilationUnit::new(CompilationConfig::default());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(source, "bench.hx")
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mir_modules = unit.get_mir_modules();

    let mut backend =
        CraneliftBackend::with_symbols(symbols).map_err(|e| format!("backend: {}", e))?;

    for module in &mir_modules {
        backend
            .compile_module(module)
            .map_err(|e| format!("compile: {}", e))?;
    }

    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            return Ok(());
        }
    }

    Err("No main found".to_string())
}

fn bench_full_compile_interp(source: &str) -> Duration {
    let symbols = get_runtime_symbols();
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let _ = run_full_compile_interp(source, &symbols);
    }

    // Benchmark
    for i in 0..BENCH_RUNS {
        let t0 = Instant::now();
        let result = run_full_compile_interp(source, &symbols);
        let elapsed = t0.elapsed();
        times.push(elapsed);

        match result {
            Ok(_) => println!("  Run {}: {:?}", i + 1, elapsed),
            Err(e) => println!("  Run {}: FAILED - {}", i + 1, e),
        }
    }

    median(&times)
}

fn run_full_compile_interp(source: &str, symbols: &[(&str, *const u8)]) -> Result<(), String> {
    let mut unit = CompilationUnit::new(CompilationConfig::default());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(source, "bench.hx")
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mir_modules = unit.get_mir_modules();

    let config = create_interp_config();
    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

    // Find main module
    let main_module = mir_modules
        .iter()
        .rev()
        .find(|m| {
            m.functions
                .values()
                .any(|f| f.name.ends_with("_main") || f.name == "main")
        })
        .ok_or("No main module")?;

    let main_id = main_module
        .functions
        .iter()
        .find(|(_, f)| f.name.ends_with("_main") || f.name == "main")
        .map(|(id, _)| *id)
        .ok_or("No main function")?;

    backend
        .compile_module((**main_module).clone())
        .map_err(|e| format!("load: {}", e))?;

    backend
        .execute_function(main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;

    Ok(())
}

fn bench_fast_compile_interp(source: &str) -> Duration {
    let symbols = get_runtime_symbols();
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let _ = run_fast_compile_interp(source, &symbols);
    }

    // Benchmark
    for i in 0..BENCH_RUNS {
        let t0 = Instant::now();
        let result = run_fast_compile_interp(source, &symbols);
        let elapsed = t0.elapsed();
        times.push(elapsed);

        match result {
            Ok(_) => println!("  Run {}: {:?}", i + 1, elapsed),
            Err(e) => println!("  Run {}: FAILED - {}", i + 1, e),
        }
    }

    median(&times)
}

fn run_fast_compile_interp(source: &str, symbols: &[(&str, *const u8)]) -> Result<(), String> {
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(source, "bench.hx")
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mir_modules = unit.get_mir_modules();

    let config = create_interp_config();
    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

    // Find main module
    let main_module = mir_modules
        .iter()
        .rev()
        .find(|m| {
            m.functions
                .values()
                .any(|f| f.name.ends_with("_main") || f.name == "main")
        })
        .ok_or("No main module")?;

    let main_id = main_module
        .functions
        .iter()
        .find(|(_, f)| f.name.ends_with("_main") || f.name == "main")
        .map(|(id, _)| *id)
        .ok_or("No main function")?;

    backend
        .compile_module((**main_module).clone())
        .map_err(|e| format!("load: {}", e))?;

    backend
        .execute_function(main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;

    Ok(())
}

fn bench_bundle_interp(bundle_path: &str) -> Duration {
    let symbols = get_runtime_symbols();
    let mut times = Vec::new();

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let _ = run_bundle_interp(bundle_path, &symbols);
    }

    // Benchmark
    for i in 0..BENCH_RUNS {
        let t0 = Instant::now();
        let result = run_bundle_interp(bundle_path, &symbols);
        let elapsed = t0.elapsed();
        times.push(elapsed);

        match result {
            Ok(_) => println!("  Run {}: {:?}", i + 1, elapsed),
            Err(e) => println!("  Run {}: FAILED - {}", i + 1, e),
        }
    }

    median(&times)
}

fn run_bundle_interp(bundle_path: &str, symbols: &[(&str, *const u8)]) -> Result<(), String> {
    let bundle = load_bundle(bundle_path).map_err(|e| format!("load bundle: {:?}", e))?;

    let config = create_interp_config();
    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

    let entry_module = bundle.entry_module().ok_or("No entry module")?;

    // Use pre-computed entry function ID (O(1) lookup)
    let main_id = bundle
        .entry_function_id()
        .ok_or("No entry function ID in bundle")?;

    backend
        .compile_module(entry_module.clone())
        .map_err(|e| format!("load: {}", e))?;

    backend
        .execute_function(main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;

    Ok(())
}

fn detailed_breakdown(source: &str, bundle_path: &str) {
    let symbols = get_runtime_symbols();

    // Full compile breakdown
    println!("│ Full Compilation Breakdown:                                    │");
    let t0 = Instant::now();
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    let t1 = Instant::now();
    let _ = unit.load_stdlib();
    let stdlib_time = t1.elapsed();

    let t2 = Instant::now();
    let _ = unit.add_file(source, "bench.hx");
    let parse_time = t2.elapsed();

    let t3 = Instant::now();
    let _ = unit.lower_to_tast();
    let tast_time = t3.elapsed();

    let t4 = Instant::now();
    let _ = unit.get_mir_modules();
    let mir_time = t4.elapsed();

    let total_compile = t0.elapsed();

    println!(
        "│   stdlib:    {:>10.2?}                                       │",
        stdlib_time
    );
    println!(
        "│   parse:     {:>10.2?}                                       │",
        parse_time
    );
    println!(
        "│   tast:      {:>10.2?}                                       │",
        tast_time
    );
    println!(
        "│   mir:       {:>10.2?}                                       │",
        mir_time
    );
    println!(
        "│   TOTAL:     {:>10.2?}                                       │",
        total_compile
    );
    println!("│                                                                │");

    // Bundle load breakdown
    println!("│ Bundle Load Breakdown:                                         │");
    let t5 = Instant::now();
    let bundle = load_bundle(bundle_path).unwrap();
    let load_time = t5.elapsed();

    let t6 = Instant::now();
    let config = create_interp_config();
    let mut backend = TieredBackend::with_symbols(config, &symbols).unwrap();
    let backend_time = t6.elapsed();

    let entry = bundle.entry_module().unwrap();
    let t7 = Instant::now();
    let _ = backend.compile_module(entry.clone());
    let module_load_time = t7.elapsed();

    let total_bundle = load_time + backend_time + module_load_time;

    println!(
        "│   file read: {:>10.2?}                                       │",
        load_time
    );
    println!(
        "│   backend:   {:>10.2?}                                       │",
        backend_time
    );
    println!(
        "│   module:    {:>10.2?}                                       │",
        module_load_time
    );
    println!(
        "│   TOTAL:     {:>10.2?}                                       │",
        total_bundle
    );
    println!("│                                                                │");

    // Speedup
    let speedup = total_compile.as_micros() as f64 / total_bundle.as_micros() as f64;
    println!(
        "│ Bundle is {:.1}x faster than full compilation                  │",
        speedup
    );
}

/// Detailed startup vs execution breakdown for pre-bundled programs
fn bundle_startup_execution_breakdown(bundle_path: &str) {
    let symbols = get_runtime_symbols();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║          BUNDLE STARTUP vs EXECUTION BREAKDOWN                 ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ For pre-bundled programs (.rzb files)                          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Run multiple times and collect detailed timings
    let mut startup_times = Vec::new();
    let mut exec_times = Vec::new();

    println!("Running {} iterations...\n", BENCH_RUNS);

    for i in 0..BENCH_RUNS {
        // ═══════════════════════════════════════════════════════════════════
        // STARTUP PHASE: Everything needed before first instruction executes
        // ═══════════════════════════════════════════════════════════════════
        let startup_start = Instant::now();

        // 1. Load bundle from disk (deserialize)
        let t_load = Instant::now();
        let bundle = load_bundle(bundle_path).unwrap();
        let load_time = t_load.elapsed();

        // 2. Create interpreter backend
        let t_backend = Instant::now();
        let config = create_interp_config();
        let mut backend = TieredBackend::with_symbols(config, &symbols).unwrap();
        let backend_time = t_backend.elapsed();

        // 3. Load module into interpreter
        let entry = bundle.entry_module().unwrap();
        let t_module = Instant::now();
        backend.compile_module(entry.clone()).unwrap();
        let module_time = t_module.elapsed();

        // 4. Get main function ID (O(1) - pre-computed in bundle)
        let t_find = Instant::now();
        let main_id = bundle.entry_function_id().unwrap_or_else(|| {
            // Fallback for older bundles without pre-computed ID
            entry
                .functions
                .iter()
                .find(|(_, f)| {
                    f.name.ends_with("_main")
                        || f.name == "main"
                        || f.name == bundle.entry_function()
                })
                .map(|(id, _)| *id)
                .unwrap()
        });
        let find_time = t_find.elapsed();

        let startup_total = startup_start.elapsed();

        // ═══════════════════════════════════════════════════════════════════
        // EXECUTION PHASE: Running the actual program
        // ═══════════════════════════════════════════════════════════════════
        let exec_start = Instant::now();
        let _ = backend.execute_function(main_id, vec![]);
        let exec_total = exec_start.elapsed();

        startup_times.push((
            startup_total,
            load_time,
            backend_time,
            module_time,
            find_time,
        ));
        exec_times.push(exec_total);

        println!(
            "  Run {}: startup={:>10.2?}  exec={:>10.2?}  total={:>10.2?}",
            i + 1,
            startup_total,
            exec_total,
            startup_total + exec_total
        );
    }

    // Calculate medians
    let startup_median = median(
        &startup_times
            .iter()
            .map(|(t, _, _, _, _)| *t)
            .collect::<Vec<_>>(),
    );
    let exec_median = median(&exec_times);
    let load_median = median(
        &startup_times
            .iter()
            .map(|(_, t, _, _, _)| *t)
            .collect::<Vec<_>>(),
    );
    let backend_median = median(
        &startup_times
            .iter()
            .map(|(_, _, t, _, _)| *t)
            .collect::<Vec<_>>(),
    );
    let module_median = median(
        &startup_times
            .iter()
            .map(|(_, _, _, t, _)| *t)
            .collect::<Vec<_>>(),
    );
    let find_median = median(
        &startup_times
            .iter()
            .map(|(_, _, _, _, t)| *t)
            .collect::<Vec<_>>(),
    );

    let total = startup_median + exec_median;
    let startup_pct = 100.0 * startup_median.as_nanos() as f64 / total.as_nanos() as f64;
    let exec_pct = 100.0 * exec_median.as_nanos() as f64 / total.as_nanos() as f64;

    println!("\n┌────────────────────────────────────────────────────────────────┐");
    println!("│                      MEDIAN RESULTS                            │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│                                                                │");
    println!("│  ┌─────────────────────────────────────────────────────────┐   │");
    println!("│  │ STARTUP (time to first instruction)                    │   │");
    println!("│  ├─────────────────────────────────────────────────────────┤   │");
    println!(
        "│  │   Bundle load (disk → memory):     {:>10.2?}          │   │",
        load_median
    );
    println!(
        "│  │   Backend init (interpreter):      {:>10.2?}          │   │",
        backend_median
    );
    println!(
        "│  │   Module load (into interpreter):  {:>10.2?}          │   │",
        module_median
    );
    println!(
        "│  │   Find main function:              {:>10.2?}          │   │",
        find_median
    );
    println!("│  │   ───────────────────────────────────────────────────   │   │");
    println!(
        "│  │   TOTAL STARTUP:                   {:>10.2?} ({:.1}%)  │   │",
        startup_median, startup_pct
    );
    println!("│  └─────────────────────────────────────────────────────────┘   │");
    println!("│                                                                │");
    println!("│  ┌─────────────────────────────────────────────────────────┐   │");
    println!("│  │ EXECUTION (running the program)                        │   │");
    println!("│  ├─────────────────────────────────────────────────────────┤   │");
    println!(
        "│  │   Program execution:               {:>10.2?} ({:.1}%)  │   │",
        exec_median, exec_pct
    );
    println!("│  └─────────────────────────────────────────────────────────┘   │");
    println!("│                                                                │");
    println!("│  ═══════════════════════════════════════════════════════════   │");
    println!(
        "│  TOTAL (startup + execution):         {:>10.2?}            │",
        total
    );
    println!("│  ═══════════════════════════════════════════════════════════   │");
    println!("│                                                                │");
    println!("└────────────────────────────────────────────────────────────────┘");

    // Visual bar chart
    println!("\n  Startup vs Execution:");
    let startup_bar = (50.0 * startup_pct / 100.0) as usize;
    let exec_bar = 50 - startup_bar;
    println!(
        "  [{:█<startup_bar$}{:░<exec_bar$}]",
        "",
        "",
        startup_bar = startup_bar,
        exec_bar = exec_bar
    );
    println!(
        "   {:^startup_bar$} {:^exec_bar$}",
        format!("{:.0}%", startup_pct),
        format!("{:.0}%", exec_pct),
        startup_bar = startup_bar,
        exec_bar = exec_bar
    );
    println!(
        "   {:^startup_bar$} {:^exec_bar$}",
        "startup",
        "execution",
        startup_bar = startup_bar,
        exec_bar = exec_bar
    );
}

fn median(times: &[Duration]) -> Duration {
    let mut sorted: Vec<_> = times.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}
