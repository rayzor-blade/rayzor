//! Measure source-to-main latency separately from kernel execution.
//! Each sample runs in a new process with an empty project cache.
//! `cargo run --release -p compiler --example benchmark_startup -- --repetitions 3`

use clap::Parser;
use compiler::codegen::tiered_backend::{TierPreset, TieredBackend};
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::optimization::{OptimizationLevel, PassManager, strip_stack_trace_updates};
use compiler::ir::tree_shake;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

const KERNELS: &[&str] = &[
    "nbody",
    "fibonacci",
    "deltablue",
    "mandelbrot",
    "binarytrees",
    "fannkuchredux",
    "spectralnorm",
    "fasta",
    "knucleotide",
    "revcomp",
    "pidigits",
    "regexredux",
];
const RESULT: &str = "__STARTUP_RESULT__";
const READY: &str = "__STARTUP_READY__";

struct ScratchCache(PathBuf);

impl Drop for ScratchCache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Parser)]
struct Cli {
    kernel: Option<String>,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    #[arg(long)]
    llvm: bool,
    /// Lower dependencies from source even when the binary embeds a snapshot.
    #[arg(long)]
    no_snapshot: bool,
    /// Retain the optimized MIR for comparisons between compiler builds.
    #[arg(long)]
    mir_dir: Option<PathBuf>,
    #[arg(long, hide = true)]
    child: bool,
}

#[derive(Serialize, Deserialize)]
struct Sample {
    kernel: String,
    build_id: String,
    llvm: bool,
    snapshot_entries: usize,
    stdlib_ms: f64,
    frontend_ms: f64,
    prepare_ms: f64,
    optimize_ms: f64,
    native_ms: f64,
    startup_ms: f64,
    process_startup_ms: f64,
    runtime_ms: f64,
    process_ms: f64,
    output_hash: u64,
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn measure(cli: &Cli) -> Result<Sample, String> {
    let start = Instant::now();
    let name = cli.kernel.as_deref().ok_or("missing kernel")?;
    if !KERNELS.contains(&name) {
        return Err(format!("unknown kernel: {name}"));
    }
    // Keep the recursive-call benchmark's work observable.
    if name == "fibonacci" {
        unsafe { std::env::set_var("RAYZOR_NO_PURE_REUSE", "1") };
    }
    if cli.no_snapshot {
        unsafe { std::env::set_var("RAYZOR_IGNORE_EMBEDDED_SNAPSHOT", "1") };
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/src")
        .join(format!("{name}.hx"));
    let source = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let cache = std::env::temp_dir().join(format!("rayzor-startup-{}", std::process::id()));
    std::fs::create_dir(&cache).map_err(|e| e.to_string())?;
    let _scratch = ScratchCache(cache.clone());
    let mut config = CompilationConfig {
        extra_defines: vec![if cli.llvm { "llvm" } else { "jit" }.into()],
        cache_dir: Some(cache.clone()),
        ..Default::default()
    };
    config.pipeline_config = config.pipeline_config.skip_analysis();
    let mut unit = CompilationUnit::new(config);
    let t = Instant::now();
    unit.load_stdlib()?;
    let stdlib_ms = ms(t);
    let t = Instant::now();
    unit.add_file(&source, path.to_str().ok_or("invalid source path")?)?;
    unit.lower_to_tast().map_err(|e| format!("{e:?}"))?;
    unit.finalize_mir_references();
    let frontend_ms = ms(t);
    let t = Instant::now();
    let lowered = unit.get_mir_modules();
    let mut modules = vec![(**lowered.last().ok_or("no MIR")?).clone()];
    let module = &modules[0];
    let entry = module.entry_function().ok_or("no main")?;
    let main_id = entry.id;
    let (module_name, entry_name) = (module.name.clone(), entry.name.clone());
    tree_shake::tree_shake_bundle(&mut modules, &module_name, &entry_name);
    let mut module = modules.pop().unwrap();
    module.link_selfcontained_wrapper_stubs();
    let prepare_ms = ms(t);
    let t = Instant::now();
    PassManager::for_level(OptimizationLevel::O2).run(&mut module);
    strip_stack_trace_updates(&mut module);
    let optimize_ms = ms(t);
    if let Some(dir) = &cli.mir_dir {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        let bytes = serde_json::to_vec(&module).map_err(|e| e.to_string())?;
        std::fs::write(dir.join(format!("{name}.json")), bytes).map_err(|e| e.to_string())?;
    }
    let t = Instant::now();
    let symbols = rayzor_runtime::plugin_impl::get_plugin().runtime_symbols();
    let mut config = TierPreset::Script.to_config();
    config.start_interpreted = false;
    config.enable_tier_promotion = false;
    config.enable_stack_traces = false;
    let mut backend = TieredBackend::with_symbols(config, &symbols)?;
    backend.set_osr_probes(false);
    backend.compile_module(module)?;
    backend.initialize_loaded_modules()?;
    if cli.llvm {
        backend.upgrade_to_llvm()?;
    }
    let native_ms = ms(t);
    let startup_ms = ms(start);
    println!("{READY}");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    let t = Instant::now();
    backend.execute_function(main_id, vec![])?;
    let runtime_ms = ms(t);
    Ok(Sample {
        kernel: name.into(),
        build_id: compiler::BUILD_ID.into(),
        llvm: cli.llvm,
        snapshot_entries: if cli.no_snapshot {
            0
        } else {
            compiler::ir::snapshot::installed().len()
        },
        stdlib_ms,
        frontend_ms,
        prepare_ms,
        optimize_ms,
        native_ms,
        startup_ms,
        process_startup_ms: 0.0,
        runtime_ms,
        process_ms: 0.0,
        output_hash: 0,
    })
}

fn run(cli: Cli) -> Result<(), String> {
    if cli.child {
        let sample = measure(&cli)?;
        println!(
            "{RESULT}{}",
            serde_json::to_string(&sample).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let names: Vec<&str> = cli
        .kernel
        .as_deref()
        .map_or_else(|| KERNELS.to_vec(), |n| vec![n]);
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    for name in names {
        for _ in 0..cli.repetitions {
            let start = Instant::now();
            let mut command = std::process::Command::new(&exe);
            command.args([name, "--child"]);
            if cli.llvm {
                command.arg("--llvm");
            }
            if cli.no_snapshot {
                command.arg("--no-snapshot");
            }
            if let Some(dir) = &cli.mir_dir {
                command.arg("--mir-dir").arg(dir);
            }
            let mut child = command
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .map_err(|e| e.to_string())?;
            let mut ready_ms = None;
            let mut result = None;
            let mut output_hash = std::collections::hash_map::DefaultHasher::new();
            for line in
                std::io::BufReader::new(child.stdout.take().ok_or("missing stdout")?).lines()
            {
                let line = line.map_err(|e| e.to_string())?;
                if line == READY {
                    ready_ms = Some(ms(start));
                } else if let Some(json) = line.strip_prefix(RESULT) {
                    result = Some(serde_json::from_str::<Sample>(json).map_err(|e| e.to_string())?);
                } else {
                    line.hash(&mut output_hash);
                    eprintln!("{line}");
                }
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            let process_ms = ms(start);
            if !status.success() {
                return Err(format!("{name}: {status}"));
            }
            let mut sample = result.ok_or("missing startup result")?;
            sample.process_startup_ms = ready_ms.ok_or("missing startup marker")?;
            sample.process_ms = process_ms;
            sample.output_hash = output_hash.finish();
            println!(
                "{}",
                serde_json::to_string(&sample).map_err(|e| e.to_string())?
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
