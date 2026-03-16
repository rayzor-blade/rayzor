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
#![allow(
    clippy::enum_variant_names,
    clippy::manual_is_variant_and,
    clippy::unnecessary_map_or
)]
//! Rayzor Benchmark Suite Runner
//!
//! Compares Rayzor execution modes against each other and external targets.
//!
//! Usage:
//!   cargo run --release --package compiler --example benchmark_runner
//!   cargo run --release --package compiler --example benchmark_runner -- mandelbrot
//!   cargo run --release --package compiler --example benchmark_runner -- --json

use compiler::codegen::tiered_backend::{TierPreset, TieredBackend, TieredConfig};
use compiler::codegen::CraneliftBackend;
use compiler::codegen::InterpValue;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::optimization::{strip_stack_trace_updates, OptimizationLevel, PassManager};
use compiler::ir::{load_bundle, IrFunctionId, IrModule, RayzorBundle};

#[cfg(feature = "llvm-backend")]
use compiler::codegen::init_llvm_once;
#[cfg(feature = "llvm-backend")]
use compiler::codegen::reset_llvm_global_state;
#[cfg(feature = "llvm-backend")]
use compiler::codegen::LLVMJitBackend;
#[cfg(feature = "llvm-backend")]
use inkwell::context::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const WARMUP_RUNS: usize = 15; // Increased to ensure LLVM promotion during warmup
const BENCH_RUNS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    name: String,
    target: String,
    compile_time_ms: f64,
    runtime_ms: f64,
    total_time_ms: f64,
    iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkSuite {
    date: String,
    #[serde(default)]
    system_info: Option<SystemInfo>,
    benchmarks: Vec<BenchmarkResults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemInfo {
    os: String,
    arch: String,
    cpu_cores: usize,
    ram_mb: u64,
    hostname: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResults {
    name: String,
    results: Vec<BenchmarkResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    RayzorInterpreter,
    RayzorCranelift,
    RayzorTiered,
    RayzorPrecompiled,       // .rzb pre-bundled MIR (skips parse/lower, still JITs)
    RayzorPrecompiledTiered, // .rzb pre-bundled MIR + tiered warmup + LLVM
    #[cfg(feature = "llvm-backend")]
    RayzorLLVM,
    HaxeInterp,    // haxe --interp
    HaxeHashLink,  // haxe -hl → hl
    HaxeHashLinkC, // haxe -hl → hlc (compile to C → gcc native)
    HaxeCpp,       // haxe -cpp → compiled binary
    HaxeJvm,       // haxe -java → JVM bytecode
    #[cfg(feature = "llvm-backend")]
    RayzorAOT, // AOT compile via LLVM → native executable
}

impl Target {
    fn name(&self) -> &'static str {
        match self {
            Target::RayzorInterpreter => "rayzor-interpreter",
            Target::RayzorCranelift => "rayzor-cranelift",
            Target::RayzorTiered => "rayzor-tiered",
            Target::RayzorPrecompiled => "rayzor-precompiled",
            Target::RayzorPrecompiledTiered => "rayzor-precompiled-tiered",
            #[cfg(feature = "llvm-backend")]
            Target::RayzorLLVM => "rayzor-llvm",
            Target::HaxeInterp => "haxe-interp",
            Target::HaxeHashLink => "haxe-hashlink",
            Target::HaxeHashLinkC => "haxe-hashlink-c",
            Target::HaxeCpp => "haxe-cpp",
            Target::HaxeJvm => "haxe-jvm",
            #[cfg(feature = "llvm-backend")]
            Target::RayzorAOT => "rayzor-aot",
        }
    }

    fn description(&self) -> &'static str {
        match self {
            Target::RayzorInterpreter => "MIR Interpreter (instant startup)",
            Target::RayzorCranelift => "Cranelift JIT (compile from source)",
            Target::RayzorTiered => "Tiered (source -> interp -> Cranelift)",
            Target::RayzorPrecompiled => "Pre-bundled MIR + JIT (skip parsing)",
            Target::RayzorPrecompiledTiered => "Pre-bundled MIR + tiered + LLVM",
            #[cfg(feature = "llvm-backend")]
            Target::RayzorLLVM => "LLVM JIT (-O3, maximum optimization)",
            Target::HaxeInterp => "Haxe --interp (eval interpreter)",
            Target::HaxeHashLink => "Haxe HashLink JIT",
            Target::HaxeHashLinkC => "Haxe HashLink/C (native via gcc)",
            Target::HaxeCpp => "Haxe C++ (hxcpp)",
            Target::HaxeJvm => "Haxe JVM (java bytecode)",
            #[cfg(feature = "llvm-backend")]
            Target::RayzorAOT => "Rayzor AOT (LLVM native executable)",
        }
    }

    fn is_haxe(&self) -> bool {
        matches!(
            self,
            Target::HaxeInterp
                | Target::HaxeHashLink
                | Target::HaxeHashLinkC
                | Target::HaxeCpp
                | Target::HaxeJvm
        )
    }
}

struct Benchmark {
    name: String,
    source: String,
}

fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    symbols.iter().map(|(n, p)| (*n, *p)).collect()
}

fn load_benchmark(name: &str) -> Option<Benchmark> {
    let base_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/src");
    let file_path = base_path.join(format!("{}.hx", name));

    if file_path.exists() {
        let source = fs::read_to_string(&file_path).ok()?;
        Some(Benchmark {
            name: name.to_string(),
            source,
        })
    } else {
        None
    }
}

/// Check if a precompiled .rzb bundle exists for this benchmark
fn has_precompiled_bundle(name: &str) -> bool {
    get_precompiled_path(name).exists()
}

/// Get the path to the precompiled .rzb bundle
fn get_precompiled_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/precompiled")
        .join(format!("{}.rzb", name))
}

/// Precompiled benchmark state - load once, run many iterations
struct PrecompiledState {
    backend: CraneliftBackend,
    main_module: std::sync::Arc<IrModule>,
    load_time: Duration,
}

/// Setup precompiled benchmark: load .rzb bundle and JIT compile once
fn setup_precompiled_benchmark(
    name: &str,
    symbols: &[(&str, *const u8)],
) -> Result<PrecompiledState, String> {
    let bundle_path = get_precompiled_path(name);

    let load_start = Instant::now();

    let bundle = load_bundle(&bundle_path).map_err(|e| format!("load bundle: {:?}", e))?;

    let mut backend =
        CraneliftBackend::with_symbols(symbols).map_err(|e| format!("backend: {}", e))?;

    // Load ALL modules from bundle (MIR already optimized at bundle creation time)
    for module in bundle.modules() {
        backend
            .compile_module(&std::sync::Arc::new(module.clone()))
            .map_err(|e| format!("load module: {}", e))?;
    }

    let load_time = load_start.elapsed();

    // Use the last module (same order as the old per-iteration approach)
    let main_module = bundle
        .modules()
        .last()
        .ok_or_else(|| "No modules in bundle".to_string())?;

    Ok(PrecompiledState {
        backend,
        main_module: std::sync::Arc::new(main_module.clone()),
        load_time,
    })
}

/// Run one iteration of a precompiled benchmark
fn run_precompiled_iteration(state: &mut PrecompiledState) -> Result<Duration, String> {
    let exec_start = Instant::now();
    state
        .backend
        .call_main(&state.main_module)
        .map_err(|e| format!("exec: {}", e))?;
    Ok(exec_start.elapsed())
}

/// Precompiled-tiered benchmark state - loads from .rzb then warms up through tiers
struct PrecompiledTieredState {
    backend: TieredBackend,
    main_id: IrFunctionId,
    load_time: Duration,
}

/// Setup precompiled-tiered benchmark: load .rzb bundle then warm up with tier promotion
fn setup_precompiled_tiered_benchmark(
    name: &str,
    symbols: &[(&str, *const u8)],
) -> Result<PrecompiledTieredState, String> {
    let bundle_path = get_precompiled_path(name);
    let load_start = Instant::now();

    let bundle = load_bundle(&bundle_path).map_err(|e| format!("load bundle: {:?}", e))?;

    // Get entry function ID
    let main_id = bundle
        .entry_function_id()
        .ok_or("No entry function ID in bundle")?;

    // Use Benchmark preset - fast tier promotion, immediate bailout
    let config = TierPreset::Benchmark.to_config();

    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

    // Load ALL modules from bundle
    for module in bundle.modules() {
        backend
            .compile_module(module.clone())
            .map_err(|e| format!("load module: {}", e))?;
    }

    let load_time = load_start.elapsed();

    Ok(PrecompiledTieredState {
        backend,
        main_id,
        load_time,
    })
}

fn run_precompiled_tiered_iteration(
    state: &mut PrecompiledTieredState,
) -> Result<Duration, String> {
    let exec_start = Instant::now();
    state
        .backend
        .execute_function(state.main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;
    Ok(exec_start.elapsed())
}

fn list_benchmarks() -> Vec<String> {
    let base_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/src");
    let mut benchmarks = Vec::new();

    if let Ok(entries) = fs::read_dir(&base_path) {
        for entry in entries.flatten() {
            if let Some(name) = entry.path().file_stem() {
                if entry.path().extension().map_or(false, |e| e == "hx") {
                    benchmarks.push(name.to_string_lossy().to_string());
                }
            }
        }
    }

    benchmarks.sort();
    benchmarks
}

fn run_benchmark_cranelift(
    bench: &Benchmark,
    symbols: &[(&str, *const u8)],
) -> Result<(Duration, Duration), String> {
    // Compile
    let compile_start = Instant::now();

    // Use fast() for lazy stdlib - avoids trace resolution issues
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(&bench.source, &format!("{}.hx", bench.name))
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mut mir_modules = unit.get_mir_modules();

    // Apply MIR optimizations (O2) for fair comparison with tiered backend
    // Tiered reaches "Optimized" tier which uses O2/O3 MIR opts + Cranelift "speed"
    let mut pass_manager = PassManager::for_level(OptimizationLevel::O2);
    for module in &mut mir_modules {
        // Get mutable access to the module through Arc::make_mut
        let module_mut = std::sync::Arc::make_mut(module);
        let _ = pass_manager.run(module_mut);
        let _ = strip_stack_trace_updates(module_mut);
    }

    let mut backend =
        CraneliftBackend::with_symbols(symbols).map_err(|e| format!("backend: {}", e))?;

    for module in &mir_modules {
        backend
            .compile_module(module)
            .map_err(|e| format!("compile: {}", e))?;
    }

    let compile_time = compile_start.elapsed();

    // Execute
    let exec_start = Instant::now();
    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            break;
        }
    }
    let exec_time = exec_start.elapsed();

    Ok((compile_time, exec_time))
}

fn run_benchmark_interpreter(
    bench: &Benchmark,
    symbols: &[(&str, *const u8)],
) -> Result<(Duration, Duration), String> {
    // Compile (to MIR only)
    let compile_start = Instant::now();

    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(&bench.source, &format!("{}.hx", bench.name))
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mir_modules = unit.get_mir_modules();

    // Use Embedded preset - interpreter only, never promotes to JIT
    // This measures pure MIR interpreter performance
    let config = TierPreset::Embedded.to_config();

    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

    // Load all modules so interpreter mode sees the same code graph as JIT modes.
    for module in &mir_modules {
        backend
            .compile_module((**module).clone())
            .map_err(|e| format!("load: {}", e))?;
    }

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

    let compile_time = compile_start.elapsed();

    // Execute
    let exec_start = Instant::now();
    backend
        .execute_function(main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;
    let exec_time = exec_start.elapsed();

    Ok((compile_time, exec_time))
}

/// Tiered benchmark state - persisted across iterations to allow JIT promotion
struct TieredBenchmarkState {
    backend: TieredBackend,
    main_id: IrFunctionId,
    compile_time: Duration,
}

/// Heavy benchmarks that should skip interpreter and start at Baseline (Cranelift)
fn is_heavy_benchmark(name: &str) -> bool {
    matches!(name, "mandelbrot" | "nbody")
}

fn setup_tiered_benchmark(
    bench: &Benchmark,
    symbols: &[(&str, *const u8)],
) -> Result<TieredBenchmarkState, String> {
    let compile_start = Instant::now();

    // Use fast() for lazy stdlib - avoids trace resolution issues
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(&bench.source, &format!("{}.hx", bench.name))
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mut mir_modules = unit.get_mir_modules();

    // Apply MIR optimization (O2) before loading into tiered backend.
    // Without this, SRA doesn't run, so 875K heap allocs/frame leak in mandelbrot.
    // Pre-compiled bundles (.rzb) are already optimized, which is why
    // rayzor-precompiled-tiered has no memory leak.
    let mut pass_manager = PassManager::for_level(OptimizationLevel::O2);
    for module in &mut mir_modules {
        let module_mut = std::sync::Arc::make_mut(module);
        let _ = pass_manager.run(module_mut);
        let _ = strip_stack_trace_updates(module_mut);
    }

    // Use Benchmark preset - optimized for performance testing
    // - Fast tier promotion (thresholds: 2, 3, 5)
    // - Immediate bailout from interpreter hot loops
    // - Synchronous optimization for deterministic results
    // - Manual LLVM upgrade after warmup (blazing_threshold = MAX)
    let config = TierPreset::Benchmark.to_config();

    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

    // Compile ALL modules (like the direct LLVM benchmark does)
    for module in &mir_modules {
        backend
            .compile_module((**module).clone())
            .map_err(|e| format!("load: {}", e))?;
    }

    // Find the main function ID
    let main_id = mir_modules
        .iter()
        .rev()
        .find_map(|m| {
            m.functions
                .iter()
                .find(|(_, f)| f.name.ends_with("_main") || f.name == "main")
                .map(|(id, _)| *id)
        })
        .ok_or("No main function")?;

    let compile_time = compile_start.elapsed();

    Ok(TieredBenchmarkState {
        backend,
        main_id,
        compile_time,
    })
}

fn run_tiered_iteration(state: &mut TieredBenchmarkState) -> Result<Duration, String> {
    let exec_start = Instant::now();
    state
        .backend
        .reset_loaded_modules_for_run()
        .map_err(|e| format!("startup: {}", e))?;
    state
        .backend
        .execute_function(state.main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;
    Ok(exec_start.elapsed())
}

fn run_benchmark_tiered(
    bench: &Benchmark,
    symbols: &[(&str, *const u8)],
) -> Result<(Duration, Duration), String> {
    // For single-iteration compatibility, create fresh backend
    let compile_start = Instant::now();

    // Use fast() for lazy stdlib - avoids trace resolution issues
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(&bench.source, &format!("{}.hx", bench.name))
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mir_modules = unit.get_mir_modules();

    // Use Application preset for single-iteration tiered benchmark
    // This is a legacy function - main benchmark uses setup_tiered_benchmark() with Benchmark preset
    let mut config = TierPreset::Application.to_config();
    config.start_interpreted = false; // Start at Baseline (Cranelift) for benchmarks
    config.verbosity = 0;

    let mut backend =
        TieredBackend::with_symbols(config, symbols).map_err(|e| format!("backend: {}", e))?;

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

    let compile_time = compile_start.elapsed();

    // Execute
    let exec_start = Instant::now();
    backend
        .reset_loaded_modules_for_run()
        .map_err(|e| format!("startup: {}", e))?;
    backend
        .execute_function(main_id, vec![])
        .map_err(|e| format!("exec: {}", e))?;
    let exec_time = exec_start.elapsed();

    Ok((compile_time, exec_time))
}

/// LLVM benchmark state - persisted across iterations (context must outlive backend)
#[cfg(feature = "llvm-backend")]
struct LLVMBenchmarkState<'ctx> {
    backend: LLVMJitBackend<'ctx>,
    mir_modules: Vec<std::sync::Arc<compiler::ir::IrModule>>,
    compile_time: Duration,
}

#[cfg(feature = "llvm-backend")]
fn setup_llvm_benchmark<'ctx>(
    bench: &Benchmark,
    symbols: &[(&str, *const u8)],
    context: &'ctx Context,
) -> Result<LLVMBenchmarkState<'ctx>, String> {
    let compile_start = Instant::now();

    // Use fast() for lazy stdlib like interpreter - avoids trace resolution issues
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().map_err(|e| format!("stdlib: {}", e))?;
    unit.add_file(&bench.source, &format!("{}.hx", bench.name))
        .map_err(|e| format!("parse: {}", e))?;
    unit.lower_to_tast().map_err(|e| format!("tast: {:?}", e))?;

    let mut mir_modules = unit.get_mir_modules();

    // Apply MIR optimizations (O2) before LLVM compilation.
    // MIR inlining + constant folding + DCE feed better IR to LLVM,
    // enabling LLVM O3 to produce tighter code (same as Cranelift path).
    let mut pass_manager = PassManager::for_level(OptimizationLevel::O2);
    for module in &mut mir_modules {
        let module_mut = std::sync::Arc::make_mut(module);
        let _ = pass_manager.run(module_mut);
        let _ = strip_stack_trace_updates(module_mut);
    }

    // Acquire LLVM lock for thread safety during compilation
    let _llvm_guard = compiler::codegen::llvm_lock();

    // Create LLVM backend (context is passed in and must outlive this)
    let mut backend =
        LLVMJitBackend::with_symbols(context, symbols).map_err(|e| format!("backend: {}", e))?;

    // Two-pass compilation for cross-module function references:
    // 1. First declare ALL functions from ALL modules
    for module in &mir_modules {
        backend
            .declare_module(module)
            .map_err(|e| format!("declare: {}", e))?;
    }
    // 2. Then compile all function bodies
    for module in &mir_modules {
        backend
            .compile_module_bodies(module)
            .map_err(|e| format!("compile: {}", e))?;
    }

    // IMPORTANT: Call finalize() ONCE to run LLVM optimization passes and create execution engine.
    // finalize() runs the LLVM optimizer (default<O3>) and creates the JIT execution engine.
    // This is the expensive part that should be counted as compile time, not execution time.
    backend.finalize().map_err(|e| format!("finalize: {}", e))?;

    let compile_time = compile_start.elapsed();

    Ok(LLVMBenchmarkState {
        backend,
        mir_modules,
        compile_time,
    })
}

#[cfg(feature = "llvm-backend")]
fn run_llvm_iteration(state: &mut LLVMBenchmarkState) -> Result<Duration, String> {
    let exec_start = Instant::now();
    for module in state.mir_modules.iter().rev() {
        if state.backend.call_main(module).is_ok() {
            break;
        }
    }
    Ok(exec_start.elapsed())
}

// Legacy function kept for compatibility - redirects to stateful approach
#[cfg(feature = "llvm-backend")]
fn run_benchmark_llvm(
    bench: &Benchmark,
    symbols: &[(&str, *const u8)],
) -> Result<(Duration, Duration), String> {
    let context = Context::create();
    let mut state = setup_llvm_benchmark(bench, symbols, &context)?;
    let exec_time = run_llvm_iteration(&mut state)?;
    Ok((state.compile_time, exec_time))
}

/// Map benchmark name to the Haxe-native source file and main class
fn get_haxe_source(bench_name: &str) -> Option<(&'static str, &'static str)> {
    match bench_name {
        "nbody" => Some(("BMNBodyCode.hx", "BMNBodyCode")),
        "mandelbrot" => Some(("BMMandelbrotCode.hx", "BMMandelbrotCode")),
        _ => None,
    }
}

/// Get the path to the haxe/ benchmark sources directory
fn get_haxe_bench_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/haxe")
}

/// Check if `haxe` CLI is available on the system
fn haxe_available() -> bool {
    Command::new("haxe")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Check if `hl` (HashLink) is available on the system
fn hashlink_available() -> bool {
    Command::new("hl")
        .arg("--version")
        .output()
        .map(|_| true) // hl --version may exit non-zero but still means it's installed
        .unwrap_or(false)
}

/// Check if HashLink/C compilation is available (requires gcc and hl --hlc support)
fn hashlink_c_available() -> bool {
    // HashLink/C requires gcc
    if !Command::new("gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return false;
    }
    // Also requires hl with hlc support - some builds may not have it
    hashlink_available()
}

fn java_available() -> bool {
    Command::new("java")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run a Haxe target benchmark via CLI. Returns (compile_time, exec_time).
/// For --interp, compile_time is 0 (interpreted). For C++/HL, compile is the haxe compile step.
fn run_haxe_benchmark(bench_name: &str, target: Target) -> Result<(Duration, Duration), String> {
    let (source_file, main_class) = get_haxe_source(bench_name)
        .ok_or_else(|| format!("No Haxe source for benchmark '{}'", bench_name))?;

    let haxe_dir = get_haxe_bench_dir();
    let source_path = haxe_dir.join(source_file);
    if !source_path.exists() {
        return Err(format!("Haxe source not found: {}", source_path.display()));
    }

    // Create a temp directory for compilation output
    let tmp_dir = std::env::temp_dir().join(format!("rayzor_haxe_bench_{}", bench_name));
    let _ = fs::create_dir_all(&tmp_dir);

    match target {
        Target::HaxeInterp => {
            // haxe --interp: compile + execute in one step
            let start = Instant::now();
            let output = Command::new("haxe")
                .arg("--main")
                .arg(main_class)
                .arg("-cp")
                .arg(&haxe_dir)
                .arg("--interp")
                .output()
                .map_err(|e| format!("Failed to run haxe: {}", e))?;
            let elapsed = start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("haxe --interp failed: {}", stderr));
            }

            // All time is execution for interp (no separate compile step)
            Ok((Duration::ZERO, elapsed))
        }

        Target::HaxeHashLink => {
            // Step 1: Compile to .hl bytecode
            let hl_path = tmp_dir.join(format!("{}.hl", bench_name));
            let compile_start = Instant::now();
            let output = Command::new("haxe")
                .arg("--main")
                .arg(main_class)
                .arg("-cp")
                .arg(&haxe_dir)
                .arg("-hl")
                .arg(&hl_path)
                .output()
                .map_err(|e| format!("Failed to compile to HL: {}", e))?;
            let compile_time = compile_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("haxe -hl compile failed: {}", stderr));
            }

            // Step 2: Run with hl
            let exec_start = Instant::now();
            let output = Command::new("hl")
                .arg(&hl_path)
                .output()
                .map_err(|e| format!("Failed to run hl: {}", e))?;
            let exec_time = exec_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("hl execution failed: {}", stderr));
            }

            Ok((compile_time, exec_time))
        }

        Target::HaxeHashLinkC => {
            // Step 1: Compile to .hl bytecode
            let hl_path = tmp_dir.join(format!("{}.hl", bench_name));
            let compile_start = Instant::now();
            let output = Command::new("haxe")
                .arg("--main")
                .arg(main_class)
                .arg("-cp")
                .arg(&haxe_dir)
                .arg("-hl")
                .arg(&hl_path)
                .output()
                .map_err(|e| format!("Failed to compile to HL: {}", e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("haxe -hl compile failed: {}", stderr));
            }

            // Step 2: Compile HL bytecode to C with hlc, then gcc
            let c_out_dir = tmp_dir.join("hlc_out");
            std::fs::create_dir_all(&c_out_dir).ok();

            let hlc_output = Command::new("hl")
                .arg(&hl_path)
                .arg("--hlc")
                .arg(&c_out_dir)
                .output()
                .map_err(|e| format!("Failed to run hlc: {}", e))?;

            if !hlc_output.status.success() {
                let stderr = String::from_utf8_lossy(&hlc_output.stderr);
                let stdout = String::from_utf8_lossy(&hlc_output.stdout);
                return Err(format!(
                    "hlc compilation failed:\nstderr: {}\nstdout: {}",
                    stderr, stdout
                ));
            }

            // Verify that main.c was created
            let main_c = c_out_dir.join("main.c");
            if !main_c.exists() {
                // List what files were actually created for debugging
                let files: Vec<_> = std::fs::read_dir(&c_out_dir)
                    .map(|rd| {
                        rd.filter_map(|e| e.ok())
                            .map(|e| e.file_name().to_string_lossy().to_string())
                            .collect()
                    })
                    .unwrap_or_default();
                return Err(format!(
                    "hlc did not create main.c. Files in {}: {:?}",
                    c_out_dir.display(),
                    files
                ));
            }

            // Step 3: Compile the generated C code with gcc
            let binary = tmp_dir.join("hlc_binary");
            let gcc_output = Command::new("gcc")
                .arg("-O2")
                .arg("-std=c11")
                .arg("-o")
                .arg(&binary)
                .arg(c_out_dir.join("main.c"))
                .arg("-I")
                .arg(&c_out_dir)
                .arg("-I")
                .arg("/usr/local/include")
                .arg("-lhl")
                .arg("-lm")
                .arg("-lpthread")
                .output()
                .map_err(|e| format!("Failed to gcc compile: {}", e))?;

            if !gcc_output.status.success() {
                let stderr = String::from_utf8_lossy(&gcc_output.stderr);
                return Err(format!("gcc compilation failed: {}", stderr));
            }

            let compile_time = compile_start.elapsed();

            // Step 4: Run the native binary
            let exec_start = Instant::now();
            let output = Command::new(&binary)
                .env("LD_LIBRARY_PATH", "/usr/local/lib")
                .output()
                .map_err(|e| format!("Failed to run hlc binary: {}", e))?;
            let exec_time = exec_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("hlc binary execution failed: {}", stderr));
            }

            Ok((compile_time, exec_time))
        }

        Target::HaxeCpp => {
            // Step 1: Compile to C++
            let cpp_dir = tmp_dir.join("cpp");
            let compile_start = Instant::now();
            let output = Command::new("haxe")
                .arg("--main")
                .arg(main_class)
                .arg("-cp")
                .arg(&haxe_dir)
                .arg("-cpp")
                .arg(&cpp_dir)
                .output()
                .map_err(|e| format!("Failed to compile to C++: {}", e))?;
            let compile_time = compile_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("haxe -cpp compile failed: {}", stderr));
            }

            // Step 2: Run the compiled binary
            let binary = cpp_dir.join(main_class);
            let exec_start = Instant::now();
            let output = Command::new(&binary)
                .output()
                .map_err(|e| format!("Failed to run C++ binary: {}", e))?;
            let exec_time = exec_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("C++ execution failed: {}", stderr));
            }

            Ok((compile_time, exec_time))
        }

        Target::HaxeJvm => {
            // Step 1: Compile to Java bytecode
            let java_dir = tmp_dir.join("java");
            let compile_start = Instant::now();
            let output = Command::new("haxe")
                .arg("--main")
                .arg(main_class)
                .arg("-cp")
                .arg(&haxe_dir)
                .arg("--jvm")
                .arg(java_dir.join(format!("{}.jar", main_class)))
                .output()
                .map_err(|e| format!("Failed to compile to JVM: {}", e))?;
            let compile_time = compile_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("haxe --jvm compile failed: {}", stderr));
            }

            // Step 2: Run with java
            let jar_path = java_dir.join(format!("{}.jar", main_class));
            let exec_start = Instant::now();
            let output = Command::new("java")
                .arg("-jar")
                .arg(&jar_path)
                .output()
                .map_err(|e| format!("Failed to run JVM: {}", e))?;
            let exec_time = exec_start.elapsed();

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("JVM execution failed: {}", stderr));
            }

            Ok((compile_time, exec_time))
        }

        _ => Err(format!("{} is not a Haxe target", target.name())),
    }
}

fn run_benchmark(bench: &Benchmark, target: Target) -> Result<BenchmarkResult, String> {
    let symbols = get_runtime_symbols();
    let mut compile_times = Vec::new();
    let mut exec_times = Vec::new();

    // Set trace prefix so output lines are tagged with the target name
    rayzor_runtime::haxe_sys::set_trace_prefix(&format!("[{}] ", target.name()));

    match target {
        // Tiered: stateful approach for JIT promotion across iterations
        Target::RayzorTiered => {
            let mut state = setup_tiered_benchmark(bench, &symbols)?;
            let compile_time = state.compile_time;

            // Warmup - runs accumulate, triggering tier promotion
            for _ in 0..WARMUP_RUNS {
                let _ = run_tiered_iteration(&mut state);
            }

            // Process optimization queue synchronously
            let optimized = state.backend.process_queue_sync();
            if optimized > 0 {
                for _ in 0..3 {
                    let _ = run_tiered_iteration(&mut state);
                }
                let _ = state.backend.process_queue_sync();
            }

            // Upgrade to LLVM tier for maximum performance
            #[cfg(feature = "llvm-backend")]
            {
                match state.backend.upgrade_to_llvm() {
                    Ok(()) => eprintln!("  [LLVM] Upgrade succeeded"),
                    Err(e) => eprintln!("  [LLVM] Upgrade FAILED: {}", e),
                }
            }

            // Benchmark runs
            for _ in 0..BENCH_RUNS {
                match run_tiered_iteration(&mut state) {
                    Ok(exec) => {
                        compile_times.push(compile_time);
                        exec_times.push(exec);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // LLVM: stateful approach - finalize() should only be called once
        #[cfg(feature = "llvm-backend")]
        Target::RayzorLLVM => {
            let context = Context::create();
            let mut state = setup_llvm_benchmark(bench, &symbols, &context)?;
            let compile_time = state.compile_time;

            // Warmup runs
            for _ in 0..WARMUP_RUNS {
                let _ = run_llvm_iteration(&mut state);
            }

            // Benchmark runs
            for _ in 0..BENCH_RUNS {
                match run_llvm_iteration(&mut state) {
                    Ok(exec) => {
                        compile_times.push(compile_time);
                        exec_times.push(exec);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Precompiled .rzb bundles: load + JIT once, then run warm iterations
        Target::RayzorPrecompiled => {
            let mut state = setup_precompiled_benchmark(&bench.name, &symbols)?;
            let load_time = state.load_time;

            // Warmup runs (same compiled code, warm caches)
            for _ in 0..WARMUP_RUNS {
                let _ = run_precompiled_iteration(&mut state);
            }

            // Benchmark runs
            for _ in 0..BENCH_RUNS {
                match run_precompiled_iteration(&mut state) {
                    Ok(exec) => {
                        compile_times.push(load_time);
                        exec_times.push(exec);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Precompiled + Tiered warmup: load .rzb, warm up, promote to LLVM
        Target::RayzorPrecompiledTiered => {
            let mut state = setup_precompiled_tiered_benchmark(&bench.name, &symbols)?;
            let load_time = state.load_time;

            // Warmup - runs accumulate, triggering tier promotion
            for _ in 0..WARMUP_RUNS {
                let _ = run_precompiled_tiered_iteration(&mut state);
            }

            // Process optimization queue synchronously
            let optimized = state.backend.process_queue_sync();
            if optimized > 0 {
                for _ in 0..3 {
                    let _ = run_precompiled_tiered_iteration(&mut state);
                }
                let _ = state.backend.process_queue_sync();
            }

            // Upgrade to LLVM tier for maximum performance
            #[cfg(feature = "llvm-backend")]
            {
                // "already done" is expected when multiple backends exist - silently ignore
                let _ = state.backend.upgrade_to_llvm();
            }

            // Benchmark runs at highest tier
            for _ in 0..BENCH_RUNS {
                match run_precompiled_tiered_iteration(&mut state) {
                    Ok(exec) => {
                        compile_times.push(load_time); // Load time = "compile time"
                        exec_times.push(exec);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // Haxe CLI targets: run via external haxe command
        Target::HaxeInterp
        | Target::HaxeHashLink
        | Target::HaxeHashLinkC
        | Target::HaxeCpp
        | Target::HaxeJvm => {
            // No warmup for Haxe targets — matches Haxe benchmark methodology
            for _ in 0..BENCH_RUNS {
                match run_haxe_benchmark(&bench.name, target) {
                    Ok((compile, exec)) => {
                        compile_times.push(compile);
                        exec_times.push(exec);
                    }
                    Err(e) => return Err(e),
                }
            }
        }

        // AOT: compile to native binary via LLVM, then run
        #[cfg(feature = "llvm-backend")]
        Target::RayzorAOT => {
            use compiler::codegen::aot_compiler::AotCompiler;
            use compiler::ir::optimization::OptimizationLevel as MirOpt;

            let tmp = std::env::temp_dir().join(format!("rayzor_aot_{}", bench.name));
            std::fs::create_dir_all(&tmp).ok();
            let binary_path = tmp.join("bench_aot");

            // Write source to a temp file for the AOT compiler
            let src_path = tmp.join(format!("{}.hx", bench.name));
            std::fs::write(&src_path, &bench.source).map_err(|e| format!("write source: {}", e))?;

            // Compile once (AOT)
            let compile_start = Instant::now();
            let compiler = AotCompiler {
                opt_level: MirOpt::O3,
                strip: true,
                verbose: false,
                ..Default::default()
            };
            compiler
                .compile(&[src_path.to_string_lossy().to_string()], &binary_path)
                .map_err(|e| format!("AOT compile: {}", e))?;
            let compile_time = compile_start.elapsed();

            // Warmup runs
            for _ in 0..WARMUP_RUNS {
                let _ = Command::new(&binary_path).output();
            }

            // Benchmark runs
            for _ in 0..BENCH_RUNS {
                let exec_start = Instant::now();
                let output = Command::new(&binary_path)
                    .output()
                    .map_err(|e| format!("AOT exec: {}", e))?;
                let exec_time = exec_start.elapsed();

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(format!("AOT execution failed: {}", stderr));
                }

                compile_times.push(compile_time);
                exec_times.push(exec_time);
            }
        }

        // Cranelift and Interpreter: each iteration is independent
        Target::RayzorCranelift | Target::RayzorInterpreter => {
            for _ in 0..WARMUP_RUNS {
                let _ = match target {
                    Target::RayzorCranelift => run_benchmark_cranelift(bench, &symbols),
                    Target::RayzorInterpreter => run_benchmark_interpreter(bench, &symbols),
                    _ => unreachable!(),
                };
            }

            for _ in 0..BENCH_RUNS {
                let result = match target {
                    Target::RayzorCranelift => run_benchmark_cranelift(bench, &symbols),
                    Target::RayzorInterpreter => run_benchmark_interpreter(bench, &symbols),
                    _ => unreachable!(),
                };

                match result {
                    Ok((compile, exec)) => {
                        compile_times.push(compile);
                        exec_times.push(exec);
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    // Calculate medians
    compile_times.sort();
    exec_times.sort();

    let median_compile = compile_times[BENCH_RUNS / 2];
    let median_exec = exec_times[BENCH_RUNS / 2];
    let total = median_compile + median_exec;

    Ok(BenchmarkResult {
        name: bench.name.clone(),
        target: target.name().to_string(),
        compile_time_ms: median_compile.as_secs_f64() * 1000.0,
        runtime_ms: median_exec.as_secs_f64() * 1000.0,
        total_time_ms: total.as_secs_f64() * 1000.0,
        iterations: BENCH_RUNS as u32,
    })
}

fn get_system_info() -> SystemInfo {
    let ram_mb = {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|l| {
                            l.split_whitespace()
                                .nth(1)
                                .and_then(|v| v.parse::<u64>().ok())
                        })
                })
                .map(|kb| kb / 1024)
                .unwrap_or(0)
        }
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.trim().parse::<u64>().ok())
                .map(|b| b / (1024 * 1024))
                .unwrap_or(0)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            0u64
        }
    };

    let hostname = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ram_mb,
        hostname,
    }
}

fn print_system_info(info: &SystemInfo) {
    println!("System Information:");
    println!("  OS:        {}", info.os);
    println!("  Arch:      {}", info.arch);
    println!("  CPU cores: {}", info.cpu_cores);
    println!("  RAM:       {:.1} GB", info.ram_mb as f64 / 1024.0);
    println!("  Host:      {}", info.hostname);
    println!();
}

fn print_results(results: &[BenchmarkResult]) {
    if results.is_empty() {
        return;
    }

    let bench_name = &results[0].name;
    let max_width = 50;

    println!("\n{}", "=".repeat(70));
    println!("  {} - Results", bench_name);
    println!("{}", "=".repeat(70));

    // Find baseline (cranelift) for speedup calculation
    let baseline = results
        .iter()
        .find(|r| r.target == "rayzor-cranelift")
        .map(|r| r.total_time_ms)
        .unwrap_or(results[0].total_time_ms);

    println!(
        "\n{:24} {:>12} {:>12} {:>12} {:>8}",
        "Target", "Compile", "Execute", "Total", "vs JIT"
    );
    println!("{}", "-".repeat(70));

    for result in results {
        let speedup = baseline / result.total_time_ms;
        let bar_len = ((result.total_time_ms / baseline) * 20.0).min(max_width as f64) as usize;
        let bar = "#".repeat(bar_len.max(1));

        println!(
            "{:24} {:>10.2}ms {:>10.2}ms {:>10.2}ms {:>7.2}x",
            result.target, result.compile_time_ms, result.runtime_ms, result.total_time_ms, speedup
        );
        println!("                         {}", bar);
    }

    println!();
}

fn save_results(suite: &BenchmarkSuite) -> Result<BenchmarkSuite, String> {
    let results_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/results");
    fs::create_dir_all(&results_dir).map_err(|e| format!("mkdir: {}", e))?;

    let filename = format!("results_{}.json", suite.date);
    let path = results_dir.join(&filename);

    // If a results file for today already exists, merge new benchmarks into it
    let merged = if path.exists() {
        let existing =
            fs::read_to_string(&path).map_err(|e| format!("read existing results: {}", e))?;
        let mut existing_suite: BenchmarkSuite = serde_json::from_str(&existing)
            .map_err(|e| format!("parse existing results: {}", e))?;

        for new_bench in &suite.benchmarks {
            if let Some(pos) = existing_suite
                .benchmarks
                .iter()
                .position(|b| b.name == new_bench.name)
            {
                // Replace existing benchmark with updated results
                existing_suite.benchmarks[pos] = new_bench.clone();
            } else {
                // Append new benchmark
                existing_suite.benchmarks.push(new_bench.clone());
            }
        }

        // Always update system info to current
        if suite.system_info.is_some() {
            existing_suite.system_info = suite.system_info.clone();
        }

        existing_suite
    } else {
        suite.clone()
    };

    let json = serde_json::to_string_pretty(&merged).map_err(|e| format!("serialize: {}", e))?;

    fs::write(&path, json).map_err(|e| format!("write: {}", e))?;
    println!("Results saved to: {}", path.display());

    Ok(merged)
}

fn generate_chart_html(suite: &BenchmarkSuite) -> Result<(), String> {
    let charts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks/charts");
    fs::create_dir_all(&charts_dir).map_err(|e| format!("mkdir: {}", e))?;

    let mut html = String::from(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Rayzor Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }
        .chart-container { width: 800px; height: 400px; margin: 20px auto; }
        h1 { text-align: center; }
        h2 { margin-top: 40px; }
        .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Rayzor Benchmark Results</h1>
    <p style="text-align: center">Generated: "#,
    );

    html.push_str(&suite.date);
    html.push_str(
        r#"</p>
"#,
    );

    if let Some(info) = &suite.system_info {
        html.push_str(&format!(
            r#"    <div class="summary">
        <h3>System</h3>
        <p><strong>OS:</strong> {} | <strong>Arch:</strong> {} | <strong>CPU cores:</strong> {} | <strong>RAM:</strong> {:.1} GB | <strong>Host:</strong> {}</p>
    </div>
"#,
            info.os,
            info.arch,
            info.cpu_cores,
            info.ram_mb as f64 / 1024.0,
            info.hostname
        ));
    }

    html.push_str(
        r#"    <div class="summary">
        <h3>Summary</h3>
        <ul>
"#,
    );

    for bench in &suite.benchmarks {
        html.push_str(&format!(
            "            <li><strong>{}</strong>: {} targets measured</li>\n",
            bench.name,
            bench.results.len()
        ));
    }

    html.push_str(
        r#"        </ul>
    </div>
    <div class="summary">
        <h3>Methodology</h3>
        <p>Each benchmark is run <strong>15 warmup iterations</strong> followed by <strong>10 measured iterations</strong>.
        Compile time and execution time are measured separately. Results show the <strong>mean</strong> of measured iterations.</p>
        <ul>
            <li><strong>rayzor-cranelift</strong> &mdash; Source &rarr; MIR (O2) &rarr; Cranelift JIT. Compile includes parsing, type-checking, MIR lowering, optimization, and JIT compilation.</li>
            <li><strong>rayzor-llvm</strong> &mdash; Source &rarr; MIR (O2) &rarr; LLVM MCJIT. Same frontend pipeline, LLVM backend for peak throughput.</li>
            <li><strong>rayzor-tiered</strong> &mdash; Source &rarr; interpreter &rarr; Cranelift JIT. Uses the <em>Benchmark</em> tier preset: interpreter thresholds (2/3/5), immediate bailout, synchronous optimization. Compile includes parsing + module loading; execution includes interpreter startup and JIT tier-up.</li>
            <li><strong>rayzor-precompiled</strong> &mdash; Pre-bundled .rzb (MIR already O2-optimized) &rarr; Cranelift JIT. Compile is bundle load + JIT only (no parsing/lowering).</li>
            <li><strong>rayzor-precompiled-tiered</strong> &mdash; Pre-bundled .rzb &rarr; tiered execution with LLVM upgrade after warmup.</li>
        </ul>
        <p>All targets share the same runtime (<code>librayzor_runtime</code>) and execute the same Haxe source code.
        MIR optimization level O2 includes: dead code elimination, constant folding, copy propagation, function inlining, LICM, and CSE.</p>
    </div>
"#,
    );

    for (i, bench) in suite.benchmarks.iter().enumerate() {
        let canvas_id = format!("chart_{}", i);

        html.push_str(&format!(
            r#"
    <h2>{}</h2>
    <div class="chart-container">
        <canvas id="{}"></canvas>
    </div>
    <script>
        new Chart(document.getElementById('{}'), {{
            type: 'bar',
            data: {{
                labels: [{}],
                datasets: [
                    {{
                        label: 'Compile (ms)',
                        data: [{}],
                        backgroundColor: 'rgba(54, 162, 235, 0.8)'
                    }},
                    {{
                        label: 'Execute (ms)',
                        data: [{}],
                        backgroundColor: 'rgba(255, 99, 132, 0.8)'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{ stacked: true }},
                    y: {{ stacked: true, title: {{ display: true, text: 'Time (ms)' }} }}
                }},
                plugins: {{
                    title: {{ display: true, text: '{}' }}
                }}
            }}
        }});
    </script>
"#,
            bench.name,
            canvas_id,
            canvas_id,
            bench
                .results
                .iter()
                .map(|r| format!("'{}'", r.target))
                .collect::<Vec<_>>()
                .join(", "),
            bench
                .results
                .iter()
                .map(|r| format!("{:.2}", r.compile_time_ms))
                .collect::<Vec<_>>()
                .join(", "),
            bench
                .results
                .iter()
                .map(|r| format!("{:.2}", r.runtime_ms))
                .collect::<Vec<_>>()
                .join(", "),
            bench.name
        ));
    }

    html.push_str(
        r#"
</body>
</html>
"#,
    );

    let path = charts_dir.join("index.html");
    fs::write(&path, html).map_err(|e| format!("write: {}", e))?;
    println!("Charts saved to: {}", path.display());

    Ok(())
}

fn main() {
    // IMPORTANT: Initialize LLVM on main thread BEFORE spawning any background threads
    // This prevents crashes due to LLVM's thread-unsafe global initialization
    #[cfg(feature = "llvm-backend")]
    init_llvm_once();

    let args: Vec<String> = std::env::args().collect();

    println!("{}", "=".repeat(70));
    println!("           Rayzor Benchmark Suite");
    println!("{}", "=".repeat(70));
    println!();

    // Parse arguments
    let json_output = args.iter().any(|a| a == "--json");
    let specific_target = args
        .iter()
        .position(|a| a == "--target")
        .and_then(|i| args.get(i + 1).cloned());
    let specific_bench = args
        .iter()
        .find(|a| {
            !a.starts_with("-") && *a != &args[0] && {
                // Skip the value after --target
                let idx = args.iter().position(|x| x == *a).unwrap_or(0);
                idx == 0 || args.get(idx - 1).map_or(true, |prev| prev != "--target")
            }
        })
        .cloned();

    // Get available benchmarks
    let available = list_benchmarks();
    println!("Available benchmarks: {}", available.join(", "));
    println!();

    // Select benchmarks to run
    let benchmarks_to_run: Vec<String> = if let Some(name) = specific_bench {
        if available.contains(&name) {
            vec![name]
        } else {
            eprintln!(
                "Unknown benchmark: {}. Available: {}",
                name,
                available.join(", ")
            );
            return;
        }
    } else {
        // Run all by default
        available
    };

    // Order targets so Cranelift-only targets run first, before LLVM-related
    // targets. This prevents LLVM compilation from corrupting shared heap state
    // that affects subsequent Cranelift targets.
    let mut all_targets_list = vec![
        // Cranelift-only targets first
        Target::RayzorCranelift,
        Target::RayzorPrecompiled,
        Target::RayzorInterpreter,
        // LLVM-related targets last (tiered upgrades to LLVM)
        Target::RayzorTiered,
        Target::RayzorPrecompiledTiered,
    ];
    #[cfg(feature = "llvm-backend")]
    all_targets_list.push(Target::RayzorLLVM);

    #[cfg(feature = "llvm-backend")]
    all_targets_list.push(Target::RayzorAOT);

    // Add Haxe targets if haxe CLI is available
    if haxe_available() {
        all_targets_list.push(Target::HaxeInterp);
        if hashlink_available() {
            all_targets_list.push(Target::HaxeHashLink);
            // HashLink/C requires gcc and proper hl --hlc support
            if hashlink_c_available() {
                all_targets_list.push(Target::HaxeHashLinkC);
            }
        }
        all_targets_list.push(Target::HaxeCpp);
        if java_available() {
            all_targets_list.push(Target::HaxeJvm);
        }
        println!("Haxe CLI detected — including Haxe targets");
    } else {
        println!("Haxe CLI not found — skipping Haxe targets (install haxe to enable)");
    }

    // Filter by --target flag if provided
    let all_targets: Vec<Target> = if let Some(ref target_name) = specific_target {
        let filtered: Vec<Target> = all_targets_list
            .iter()
            .filter(|t| t.name() == target_name.as_str())
            .copied()
            .collect();
        if filtered.is_empty() {
            eprintln!(
                "Unknown target: {}. Available: {}",
                target_name,
                all_targets_list
                    .iter()
                    .map(|t| t.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return;
        }
        filtered
    } else {
        all_targets_list
    };

    println!(
        "Running {} benchmarks x up to {} targets",
        benchmarks_to_run.len(),
        all_targets.len()
    );
    println!(
        "Warmup: {} runs, Benchmark: {} runs",
        WARMUP_RUNS, BENCH_RUNS
    );
    println!();

    let sys_info = get_system_info();
    print_system_info(&sys_info);

    let mut suite = BenchmarkSuite {
        date: chrono::Local::now().format("%Y-%m-%d").to_string(),
        system_info: Some(sys_info),
        benchmarks: Vec::new(),
    };

    for bench_name in &benchmarks_to_run {
        println!("{}", "-".repeat(70));
        println!("Benchmark: {}", bench_name);
        println!("{}", "-".repeat(70));

        let bench = match load_benchmark(bench_name) {
            Some(b) => b,
            None => {
                eprintln!("  Failed to load benchmark: {}", bench_name);
                continue;
            }
        };

        // Filter targets for this benchmark
        // Skip standalone interpreter for heavy benchmarks (millions of iterations too slow)
        // Tiered mode handles interpreter → Cranelift handoff automatically
        let is_heavy = is_heavy_benchmark(bench_name);
        let has_precompiled = has_precompiled_bundle(bench_name);

        let has_haxe_source = get_haxe_source(bench_name).is_some();

        let mut targets: Vec<Target> = if specific_target.is_some() {
            // When --target is specified, use exactly the requested targets
            all_targets.clone()
        } else {
            // Default: filter heavy benchmarks, filter Haxe/precompiled without sources/bundles
            let mut t: Vec<Target> = all_targets
                .iter()
                .filter(|t| !is_heavy || !matches!(t, Target::RayzorInterpreter))
                .filter(|t| !t.is_haxe() || has_haxe_source)
                .filter(|t| {
                    !matches!(
                        t,
                        Target::RayzorPrecompiled | Target::RayzorPrecompiledTiered
                    ) || has_precompiled
                })
                .copied()
                .collect();

            if has_precompiled {
                println!("  (Precompiled .rzb bundle found - testing AOT and AOT+tiered)\n");
            }

            if is_heavy {
                println!(
                    "  (Standalone interpreter skipped - tiered mode shows full progression)\n"
                );
            }
            t
        };

        // When --target is specified, filter to only that target
        if let Some(ref tn) = specific_target {
            targets.retain(|t| t.name() == tn.as_str());
            if targets.is_empty() {
                eprintln!(
                    "  Target '{}' not available for benchmark '{}'",
                    tn, bench_name
                );
                continue;
            }
        }

        // Run all targets sequentially to avoid CPU contention between benchmarks.
        // This ensures each target gets full CPU resources for accurate measurement.
        let mut results = Vec::new();

        for target in &targets {
            println!("  Running {} ...", target.name());

            // Reset LLVM global state before each target to ensure fresh compilation.
            // This prevents targets from reusing stale pointers from previous compilations
            // (e.g., precompiled-tiered reusing non-SRA pointers from rayzor-llvm).
            #[cfg(feature = "llvm-backend")]
            reset_llvm_global_state();

            let result = run_benchmark(&bench, *target);
            match result {
                Ok(bench_result) => {
                    println!("  [DONE] {} ({})", target.name(), target.description());
                    println!(
                        "         Compile: {:.2}ms, Execute: {:.2}ms, Total: {:.2}ms\n",
                        bench_result.compile_time_ms,
                        bench_result.runtime_ms,
                        bench_result.total_time_ms
                    );
                    results.push(bench_result);
                }
                Err(e) => {
                    eprintln!("  [FAIL] {}: {}\n", target.name(), e);
                }
            }

            // Brief pause between targets to let the allocator reclaim memory and
            // reduce fragmentation-related crashes (intermittent malloc errors).
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        // Sort results by target name for consistent ordering
        results.sort_by(|a, b| a.target.cmp(&b.target));

        print_results(&results);

        suite.benchmarks.push(BenchmarkResults {
            name: bench_name.clone(),
            results: results.clone(),
        });
    }

    // Save results
    println!("\n{}", "=".repeat(70));
    println!("Saving results...");

    let merged_suite = match save_results(&suite) {
        Ok(merged) => merged,
        Err(e) => {
            eprintln!("Failed to save results: {}", e);
            suite.clone()
        }
    };

    if let Err(e) = generate_chart_html(&merged_suite) {
        eprintln!("Failed to generate charts: {}", e);
    }

    if json_output {
        println!("\nJSON Output:");
        println!(
            "{}",
            serde_json::to_string_pretty(&merged_suite).unwrap_or_default()
        );
    }
}
