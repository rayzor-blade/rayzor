//! Rayzor - High-performance Haxe compiler with tiered JIT compilation
//!
//! # Usage
//!
//! ```bash
//! # Compile and run a Haxe file
//! rayzor run Main.hx
//!
//! # Use HXML build file (compatible with standard Haxe)
//! rayzor build.hxml
//!
//! # JIT compile with tier selection
//! rayzor jit --tier 2 MyApp.hx
//!
//! # Check syntax without executing
//! rayzor check Main.hx
//!
//! # Show compilation pipeline
//! rayzor compile --show-ir Main.hx
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::process;

#[derive(Parser)]
#[command(name = "rayzor")]
#[command(version = "0.1.0")]
#[command(about = "Rayzor - High-performance Haxe compiler with tiered JIT", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a Haxe file with JIT compilation
    Run {
        /// Path to the Haxe source file (reads from rayzor.toml if omitted)
        file: Option<PathBuf>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Show compilation statistics
        #[arg(long)]
        stats: bool,

        /// Starting optimization tier (0-3)
        #[arg(long, default_value = "0")]
        tier: u8,

        /// Enable LLVM Tier 3 optimization
        #[arg(long)]
        llvm: bool,

        /// Tier preset: script, application, server, benchmark, development, embedded
        #[arg(long, value_enum, default_value = "application")]
        preset: Preset,

        /// Enable BLADE cache for incremental compilation
        #[arg(long)]
        cache: bool,

        /// Cache directory (defaults to target/debug/cache or target/release/cache)
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Build with optimizations (uses target/release instead of target/debug)
        #[arg(long)]
        release: bool,

        /// Enable GPU compute support (loads rayzor-gpu dynamic library)
        #[arg(long)]
        compute: bool,

        /// Load .rpkg packages (repeatable)
        #[arg(long = "rpkg", value_name = "FILE")]
        rpkg_files: Vec<PathBuf>,
    },

    /// JIT compile with interactive REPL
    Jit {
        /// Path to the Haxe source file
        file: Option<PathBuf>,

        /// Target optimization tier (0=baseline, 1=standard, 2=optimized, 3=maximum/LLVM)
        #[arg(short, long, default_value = "2")]
        tier: u8,

        /// Show Cranelift IR
        #[arg(long)]
        show_cranelift: bool,

        /// Show MIR (Mid-level IR)
        #[arg(long)]
        show_mir: bool,

        /// Enable profiling for tier promotion
        #[arg(long)]
        profile: bool,
    },

    /// Check Haxe syntax and type checking
    Check {
        /// Path to the Haxe source file
        file: PathBuf,

        /// Show full type information
        #[arg(long)]
        show_types: bool,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Compile Haxe to intermediate representation
    Compile {
        /// Path to the Haxe source file
        file: PathBuf,

        /// Stop at compilation stage
        #[arg(long, value_enum, default_value = "native")]
        stage: CompileStage,

        /// Show intermediate representations
        #[arg(long)]
        show_ir: bool,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Enable BLADE cache for incremental compilation
        #[arg(long)]
        cache: bool,

        /// Cache directory (defaults to target/debug/cache or target/release/cache)
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Build with optimizations (uses target/release instead of target/debug)
        #[arg(long)]
        release: bool,
    },

    /// Build from HXML file or rayzor.toml
    Build {
        /// Path to HXML build file (auto-detects rayzor.toml if omitted)
        file: Option<PathBuf>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Override output path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Show what would be built without building
        #[arg(long)]
        dry_run: bool,
    },

    /// Show information about the compiler
    Info {
        /// Show detailed feature information
        #[arg(long)]
        features: bool,

        /// Show tiered JIT configuration
        #[arg(long)]
        tiers: bool,
    },

    /// Manage BLADE compilation cache
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// Create a .rzb bundle from source files
    Bundle {
        /// Source files to compile
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Output .rzb path
        #[arg(short, long)]
        output: PathBuf,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "2")]
        opt_level: u8,

        /// Tree-shake unreachable code (for AOT/size-optimized bundles)
        #[arg(long)]
        strip: bool,

        /// Disable zstd compression
        #[arg(long)]
        no_compress: bool,

        /// Enable BLADE incremental cache
        #[arg(long)]
        cache: bool,

        /// Custom BLADE cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Compile Haxe to a native executable via LLVM (AOT)
    Aot {
        /// Source files to compile
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Output path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Target triple for cross-compilation
        #[arg(long)]
        target: Option<String>,

        /// Output format: exe, obj, llvm-ir, llvm-bc, asm
        #[arg(long, default_value = "exe")]
        emit: String,

        /// Optimization level (0-3)
        #[arg(short = 'O', long, default_value = "2")]
        opt_level: u8,

        /// Tree-shake unreachable code
        #[arg(long, default_value = "true")]
        strip: bool,

        /// Strip debug symbols from binary
        #[arg(long)]
        strip_symbols: bool,

        /// Path to librayzor_runtime.a
        #[arg(long)]
        runtime_dir: Option<PathBuf>,

        /// Override linker path
        #[arg(long)]
        linker: Option<String>,

        /// Sysroot for cross-compilation
        #[arg(long)]
        sysroot: Option<PathBuf>,

        /// Enable BLADE incremental cache
        #[arg(long)]
        cache: bool,

        /// Custom BLADE cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Initialize a new Rayzor project or workspace
    Init {
        /// Project or workspace name (also used as directory name)
        #[arg(long)]
        name: Option<String>,

        /// Create a multi-project workspace instead of a single project
        #[arg(long)]
        workspace: bool,
    },

    /// Extract stdlib symbols to .bsym format (pre-BLADE)
    Preblade {
        /// Source files (if empty, uses stdlib)
        files: Vec<PathBuf>,

        /// Output directory for .bsym files
        #[arg(short, long)]
        out: Option<PathBuf>,

        /// List types without generating files
        #[arg(short, long)]
        list: bool,

        /// Custom BLADE cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Dump MIR (Mid-level IR) in LLVM-like textual format for debugging
    Dump {
        /// Path to the Haxe source file
        file: PathBuf,

        /// Output to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Optimization level (0-3, default: 2)
        #[arg(short = 'O', long, default_value = "2")]
        opt_level: u8,

        /// Show only specific function (by name)
        #[arg(long)]
        function: Option<String>,

        /// Show only CFG (control flow graph) without instructions
        #[arg(long)]
        cfg_only: bool,
    },

    /// Manage .rpkg packages (pack, inspect)
    Rpkg {
        #[command(subcommand)]
        action: RpkgAction,
    },
}

#[derive(Subcommand)]
enum RpkgAction {
    /// Pack Haxe sources (and optionally a native dylib) into an .rpkg file
    Pack {
        /// Path to a native library (.dylib/.so/.dll) — optional for pure Haxe packages
        #[arg(long)]
        dylib: Option<PathBuf>,

        /// Directory containing .hx source files to bundle
        #[arg(long)]
        haxe_dir: PathBuf,

        /// Output .rpkg path
        #[arg(short, long)]
        output: PathBuf,

        /// Package name (defaults to output filename without extension)
        #[arg(long)]
        name: Option<String>,
    },

    /// Inspect the contents of an .rpkg file
    Inspect {
        /// Path to the .rpkg file
        file: PathBuf,
    },
}

#[derive(Subcommand)]
enum CacheAction {
    /// Show cache statistics
    Stats {
        /// Cache directory (defaults to .rayzor-cache)
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },

    /// Clear all cached modules
    Clear {
        /// Cache directory (defaults to .rayzor-cache)
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
    Pretty,
}

#[derive(ValueEnum, Clone, Debug)]
enum CompileStage {
    /// Stop after parsing (AST)
    Ast,
    /// Stop after type checking (TAST)
    Tast,
    /// Stop after semantic analysis (HIR)
    Hir,
    /// Stop after MIR lowering
    Mir,
    /// Compile to native code (default)
    Native,
}

/// Tier preset for JIT compilation
#[derive(ValueEnum, Clone, Debug, Copy)]
enum Preset {
    /// CLI tools, one-shot scripts - instant startup, no tier promotion
    Script,
    /// Desktop apps, web servers - balanced tiering with LLVM (default)
    Application,
    /// Long-running services, APIs - aggressive optimization
    Server,
    /// Performance testing - immediate bailout, manual LLVM upgrade
    Benchmark,
    /// Development and debugging - verbose logging
    Development,
    /// Resource-constrained environments - interpreter only
    Embedded,
}

impl Preset {
    fn to_tier_preset(self) -> compiler::codegen::TierPreset {
        match self {
            Preset::Script => compiler::codegen::TierPreset::Script,
            Preset::Application => compiler::codegen::TierPreset::Application,
            Preset::Server => compiler::codegen::TierPreset::Server,
            Preset::Benchmark => compiler::codegen::TierPreset::Benchmark,
            Preset::Development => compiler::codegen::TierPreset::Development,
            Preset::Embedded => compiler::codegen::TierPreset::Embedded,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run {
            file,
            verbose,
            stats,
            tier,
            llvm,
            preset,
            cache,
            cache_dir,
            release,
            compute,
            rpkg_files,
        } => run_file(
            file, verbose, stats, tier, llvm, preset, cache, cache_dir, release, compute,
            rpkg_files,
        ),
        Commands::Jit {
            file,
            tier,
            show_cranelift,
            show_mir,
            profile,
        } => jit_compile(file, tier, show_cranelift, show_mir, profile),
        Commands::Check {
            file,
            show_types,
            format,
        } => check_file(file, show_types, format),
        Commands::Compile {
            file,
            stage,
            show_ir,
            output,
            cache,
            cache_dir,
            release,
        } => compile_file(file, stage, show_ir, output, cache, cache_dir, release),
        Commands::Build {
            file,
            verbose,
            output,
            dry_run,
        } => build_hxml(file, verbose, output, dry_run),
        Commands::Info { features, tiers } => {
            show_info(features, tiers);
            Ok(())
        }
        Commands::Cache { action } => match action {
            CacheAction::Stats { cache_dir } => cache_stats(cache_dir),
            CacheAction::Clear { cache_dir } => cache_clear(cache_dir),
        },
        Commands::Bundle {
            files,
            output,
            opt_level,
            strip,
            no_compress,
            cache,
            cache_dir,
            verbose,
        } => cmd_bundle(
            files,
            output,
            opt_level,
            strip,
            no_compress,
            cache,
            cache_dir,
            verbose,
        ),
        Commands::Aot {
            files,
            output,
            target,
            emit,
            opt_level,
            strip,
            strip_symbols,
            runtime_dir,
            linker,
            sysroot,
            cache,
            cache_dir,
            verbose,
        } => cmd_aot(
            files,
            output,
            target,
            emit,
            opt_level,
            strip,
            strip_symbols,
            runtime_dir,
            linker,
            sysroot,
            cache,
            cache_dir,
            verbose,
        ),
        Commands::Init { name, workspace } => cmd_init(name, workspace),
        Commands::Preblade {
            files,
            out,
            list,
            cache_dir,
            verbose,
        } => cmd_preblade(files, out, list, cache_dir, verbose),
        Commands::Dump {
            file,
            output,
            opt_level,
            function,
            cfg_only,
        } => cmd_dump(file, output, opt_level, function, cfg_only),
        Commands::Rpkg { action } => match action {
            RpkgAction::Pack {
                dylib,
                haxe_dir,
                output,
                name,
            } => cmd_rpkg_pack(dylib, haxe_dir, output, name),
            RpkgAction::Inspect { file } => cmd_rpkg_inspect(file),
        },
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

/// Helper function to compile Haxe source through the full pipeline to MIR
/// Uses CompilationUnit for proper multi-file, stdlib-aware compilation
/// Returns the primary MIR module (user code)
fn compile_haxe_to_mir(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
) -> Result<compiler::ir::IrModule, String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};

    // Create compilation unit with stdlib support
    let config = CompilationConfig {
        load_stdlib: true, // Enable stdlib for full Haxe compatibility
        ..Default::default()
    };

    let mut unit = CompilationUnit::new(config);

    // Register external plugins (e.g., GPU compute) before compilation
    for plugin in plugins {
        unit.register_compiler_plugin(plugin);
    }

    // Add extra source paths (e.g. from rpkg packages) for import resolution
    for dir in extra_source_dirs {
        unit.add_source_path(dir.clone());
    }

    // Load the standard library first
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add the source file to the compilation unit
    unit.add_file(source, filename)?;

    // Type-check pass — errors reported via diagnostics formatter
    if let Err(errors) = unit.lower_to_tast() {
        unit.print_compilation_errors(&errors);
        return Err(format!("Check failed with {} error(s)", errors.len()));
    }

    // Get all MIR modules (including stdlib)
    let mir_modules = unit.get_mir_modules();

    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Return the last module (user code - stdlib modules are compiled first)
    // Clone the inner IrModule from the Arc
    let module = (**mir_modules.last().unwrap()).clone();
    Ok(module)
}

/// Loaded GPU plugin — keeps the dylib alive and provides both runtime symbols
/// and a compiler plugin for method registration.
struct GpuPlugin {
    _lib: libloading::Library,
    symbols: Vec<(&'static str, *const u8)>,
    compiler_plugin: Option<compiler::compiler_plugin::NativePlugin>,
}

/// Try to load the GPU compute plugin from the rayzor-gpu dynamic library.
///
/// On success, returns a GpuPlugin containing:
/// - Runtime symbols for JIT linking
/// - A NativePlugin for compiler-side method registration
fn try_load_gpu_plugin() -> Option<GpuPlugin> {
    let lib_name = if cfg!(target_os = "macos") {
        "librayzor_gpu.dylib"
    } else if cfg!(target_os = "linux") {
        "librayzor_gpu.so"
    } else {
        return None;
    };

    // Try paths: next to executable, then current dir
    let search_paths = [
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(lib_name))),
        Some(std::path::PathBuf::from(lib_name)),
    ];

    for path in search_paths.iter().flatten() {
        if let Ok(lib) = unsafe { libloading::Library::new(path) } {
            let mut symbols = Vec::new();

            // Load runtime symbols for JIT linking
            type InitFn = unsafe extern "C" fn(*mut usize) -> *const u8;
            if let Ok(init_fn) = unsafe { lib.get::<InitFn>(b"rayzor_gpu_plugin_init") } {
                let mut count: usize = 0;
                let entries_ptr = unsafe { init_fn(&mut count) };
                if !entries_ptr.is_null() && count > 0 {
                    let entries = unsafe {
                        std::slice::from_raw_parts(
                            entries_ptr as *const (usize, usize, usize),
                            count,
                        )
                    };
                    for &(name_ptr, name_len, fn_ptr) in entries {
                        let name = unsafe {
                            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                                name_ptr as *const u8,
                                name_len,
                            ))
                        };
                        let name: &'static str = unsafe { std::mem::transmute(name) };
                        symbols.push((name, fn_ptr as *const u8));
                    }
                }
            }

            // Load method descriptors for compiler-side registration
            type DescribeFn =
                unsafe extern "C" fn(*mut usize) -> *const rayzor_plugin::NativeMethodDesc;
            let compiler_plugin = unsafe {
                if let Ok(describe_fn) = lib.get::<DescribeFn>(b"rayzor_gpu_plugin_describe") {
                    let mut count: usize = 0;
                    let descs = describe_fn(&mut count);
                    if !descs.is_null() && count > 0 {
                        Some(compiler::compiler_plugin::NativePlugin::from_descriptors(
                            "rayzor_gpu_compute",
                            descs,
                            count,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            return Some(GpuPlugin {
                _lib: lib,
                symbols,
                compiler_plugin,
            });
        }
    }
    None
}

fn run_bundle(file: &Path, verbose: bool, stats: bool, preset: Preset) -> Result<(), String> {
    use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};
    use compiler::ir::load_bundle;

    if !file.exists() {
        return Err(format!("Bundle not found: {}", file.display()));
    }

    let bundle = load_bundle(file).map_err(|e| format!("Failed to load bundle: {}", e))?;

    let entry_func_id = bundle
        .entry_function_id()
        .ok_or("Bundle has no entry function")?;

    if verbose {
        println!(
            "  bundle   {} modules, entry: {}",
            bundle.module_count(),
            bundle.entry_function()
        );
    }

    // Get runtime symbols
    let plugin = rayzor_runtime::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut config = TieredConfig::from_preset(preset.to_tier_preset());
    config.verbosity = if verbose { 2 } else { 0 };
    config.start_interpreted = false;

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)?;

    for module in bundle.modules().iter() {
        backend
            .compile_module(module.clone())
            .map_err(|e| format!("Failed to compile module '{}': {}", module.name, e))?;
    }

    if stats {
        let backend_stats = backend.get_statistics();
        println!("  tier 0   {} functions", backend_stats.baseline_functions);
        println!("  tier 1   {} functions", backend_stats.standard_functions);
        println!("  tier 2   {} functions", backend_stats.optimized_functions);
        println!("  tier 3   {} functions", backend_stats.llvm_functions);
    }

    backend
        .execute_function(entry_func_id, vec![])
        .map_err(|e| format!("Execution failed: {}", e))?;

    backend.shutdown();

    println!("✓ Complete");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_file(
    file_arg: Option<PathBuf>,
    verbose: bool,
    stats: bool,
    _tier: u8,
    _llvm: bool,
    preset: Preset,
    _cache: bool,
    _cache_dir: Option<PathBuf>,
    release: bool,
    compute: bool,
    rpkg_files: Vec<PathBuf>,
) -> Result<(), String> {
    use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};

    // Resolve file: from arg or rayzor.toml
    let file = match file_arg {
        Some(f) => f,
        None => resolve_entry_from_manifest()?,
    };

    let profile = if release { "release" } else { "debug" };
    println!(
        "🚀 Running {} [{}] [preset: {:?}]...",
        file.display(),
        profile,
        preset
    );

    // Handle precompiled .rzb bundles
    if file.extension().is_some_and(|ext| ext == "rzb") {
        return run_bundle(&file, verbose, stats, preset);
    }

    #[cfg(not(feature = "llvm-backend"))]
    if _llvm {
        return Err(
            "LLVM backend not available. Recompile with --features llvm-backend".to_string(),
        );
    }

    // Read source file
    if !file.exists() {
        return Err(format!("File not found: {}", file.display()));
    }
    let source =
        std::fs::read_to_string(&file).map_err(|e| format!("Failed to read file: {}", e))?;

    // Always try to load the GPU plugin — silently skip if the dylib isn't found.
    // The --compute flag upgrades a missing dylib from silent skip to a warning.
    let mut gpu_plugin = match try_load_gpu_plugin() {
        Some(gpu) => {
            if verbose {
                eprintln!(
                    "  gpu      loaded {} symbols from rayzor-gpu plugin",
                    gpu.symbols.len()
                );
            }
            Some(gpu)
        }
        None => {
            if compute {
                eprintln!("warning: --compute flag set but rayzor-gpu library not found");
            }
            None
        }
    };

    // Extract compiler plugin from GPU (moved into compiler during compilation)
    let mut compiler_plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>> = Vec::new();
    if let Some(ref mut gpu) = gpu_plugin {
        if let Some(cp) = gpu.compiler_plugin.take() {
            compiler_plugins.push(Box::new(cp));
        }
    }

    // Load .rpkg packages
    let mut loaded_rpkgs: Vec<compiler::rpkg::install::RpkgPlugin> = Vec::new();
    let mut rpkg_source_dirs: Vec<PathBuf> = Vec::new();
    for rpkg_path in &rpkg_files {
        match compiler::rpkg::install::RpkgPlugin::load(rpkg_path) {
            Ok(rpkg) => {
                if verbose {
                    eprintln!(
                        "  rpkg     loaded '{}' ({} methods, {} hx files)",
                        rpkg.package_name,
                        rpkg.runtime_symbols.len(),
                        rpkg.haxe_sources.len(),
                    );
                }
                // Write bundled .hx files to temp dir for import resolution
                if !rpkg.haxe_sources.is_empty() {
                    let tmp_dir = std::env::temp_dir().join(format!(
                        "rpkg_hx_{}_{}",
                        rpkg.package_name,
                        std::process::id()
                    ));
                    for (module_path, source) in &rpkg.haxe_sources {
                        let dest = tmp_dir.join(module_path);
                        if let Some(parent) = dest.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let _ = std::fs::write(&dest, source);
                    }
                    rpkg_source_dirs.push(tmp_dir);
                }
                loaded_rpkgs.push(rpkg);
            }
            Err(e) => {
                return Err(format!(
                    "failed to load rpkg {}: {}",
                    rpkg_path.display(),
                    e
                ));
            }
        }
    }

    // Extract compiler plugins from rpkg packages
    for rpkg in &mut loaded_rpkgs {
        if let Some(cp) = rpkg.compiler_plugin.take() {
            compiler_plugins.push(Box::new(cp));
        }
    }

    // Compile source file to MIR (with plugins registered)
    let mut mir_module = compile_haxe_to_mir(
        &source,
        file.to_str().unwrap_or("unknown"),
        compiler_plugins,
        &rpkg_source_dirs,
    )?;

    // Run O0 pass manager to expand Haxe `inline` functions and apply SRA
    if std::env::var("RAYZOR_RAW_MIR").is_err() {
        use compiler::ir::optimization::{OptimizationLevel, PassManager};
        let mut pass_manager = PassManager::for_level(OptimizationLevel::O0);
        let _ = pass_manager.run(&mut mir_module);
    }

    let total_functions = mir_module.functions.len();
    if verbose {
        println!("  parse    {} ({} decls)", file.display(), total_functions);
    }

    if total_functions == 0 {
        return Err("No functions found to execute".to_string());
    }

    // Find main function before consuming mir_module
    let main_func_id = mir_module
        .functions
        .iter()
        .find(|(_, f)| f.name == "main")
        .map(|(id, _)| *id)
        .ok_or("No main function found")?;

    // Find __vtable_init__ and __init__ functions (if present)
    let vtable_init_func_id = mir_module
        .functions
        .iter()
        .find(|(_, f)| f.name == "__vtable_init__")
        .map(|(id, _)| *id);
    let module_init_func_id = mir_module
        .functions
        .iter()
        .find(|(_, f)| f.name == "__init__")
        .map(|(id, _)| *id);

    // Get runtime symbols
    let plugin = rayzor_runtime::get_plugin();
    let mut symbols = plugin.runtime_symbols();

    // Merge GPU runtime symbols for JIT linking
    if let Some(ref gpu) = gpu_plugin {
        symbols.extend_from_slice(&gpu.symbols);
    }

    // Merge rpkg runtime symbols for JIT linking
    let rpkg_owned_symbols: Vec<(String, *const u8)> = loaded_rpkgs
        .iter()
        .flat_map(|r| r.runtime_symbols.clone())
        .collect();
    for (name, ptr) in &rpkg_owned_symbols {
        // Leak the string to get 'static lifetime (same pattern as GPU plugin)
        let name: &'static str = Box::leak(name.clone().into_boxed_str());
        symbols.push((name, *ptr));
    }

    // Keep dylibs alive until backend is done
    let _gpu_plugin = gpu_plugin;
    let _loaded_rpkgs = loaded_rpkgs;

    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Set up tiered JIT backend using the selected preset
    let mut config = TieredConfig::from_preset(preset.to_tier_preset());
    config.verbosity = if verbose { 2 } else { 0 };
    config.start_interpreted = false; // Start with JIT for immediate execution

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)?;

    // Compile module with tiered JIT
    backend.compile_module(mir_module)?;

    if verbose {
        let backend_stats = backend.get_statistics();
        let compiled = backend_stats.baseline_functions
            + backend_stats.standard_functions
            + backend_stats.optimized_functions
            + backend_stats.llvm_functions;
        println!(
            "  jit      {} functions compiled (preset: {:?})",
            compiled, preset
        );
    }

    // Show stats if requested
    if stats {
        let backend_stats = backend.get_statistics();
        println!("  tier 0   {} functions", backend_stats.baseline_functions);
        println!("  tier 1   {} functions", backend_stats.standard_functions);
        println!("  tier 2   {} functions", backend_stats.optimized_functions);
        println!("  tier 3   {} functions", backend_stats.llvm_functions);
    }

    // Execute init functions before main
    if let Some(vtable_init_id) = vtable_init_func_id {
        backend
            .execute_function(vtable_init_id, vec![])
            .map_err(|e| format!("vtable init failed: {}", e))?;
    }
    if let Some(init_id) = module_init_func_id {
        backend
            .execute_function(init_id, vec![])
            .map_err(|e| format!("module init failed: {}", e))?;
    }

    // Execute main function
    backend
        .execute_function(main_func_id, vec![])
        .map_err(|e| format!("Execution failed: {}", e))?;

    backend.shutdown();

    // Clean up temp dirs from rpkg haxe sources
    for dir in &rpkg_source_dirs {
        let _ = std::fs::remove_dir_all(dir);
    }

    println!("✓ Complete");
    Ok(())
}

fn jit_compile(
    file: Option<PathBuf>,
    tier: u8,
    show_cranelift: bool,
    show_mir: bool,
    profile: bool,
) -> Result<(), String> {
    if let Some(ref path) = file {
        println!("🔥 JIT compiling {} at Tier {}...", path.display(), tier);
    } else {
        println!("🔥 Starting Rayzor JIT REPL...");
        println!("   Type Haxe code or 'exit' to quit");
    }

    if show_cranelift {
        println!("  Will show Cranelift IR");
    }
    if show_mir {
        println!("  Will show MIR");
    }
    if profile {
        println!("  Profiling enabled for tier promotion");
    }

    // TODO: Implement JIT compilation
    Err(
        "JIT command not yet fully implemented. See compiler/examples/test_full_pipeline_tiered.rs"
            .to_string(),
    )
}

fn check_file(file: PathBuf, show_types: bool, format: OutputFormat) -> Result<(), String> {
    println!("✓ Checking {}...", file.display());

    if !file.exists() {
        return Err(format!("File not found: {}", file.display()));
    }

    let source =
        std::fs::read_to_string(&file).map_err(|e| format!("Failed to read file: {}", e))?;

    // Parse the file
    use parser::haxe_parser::parse_haxe_file;
    let ast = parse_haxe_file(file.to_str().unwrap_or("unknown"), &source, false)
        .map_err(|e| format!("Parse error: {}", e))?;

    match format {
        OutputFormat::Text => {
            println!("✓ Syntax: OK");
            println!("  Package: {:?}", ast.package);
            println!("  Declarations: {}", ast.declarations.len());
            println!("  Module fields: {}", ast.module_fields.len());
            println!("  Imports: {}", ast.imports.len());
        }
        OutputFormat::Json => {
            println!("{{");
            println!("  \"status\": \"ok\",");
            println!("  \"declarations\": {},", ast.declarations.len());
            println!("  \"module_fields\": {},", ast.module_fields.len());
            println!("  \"imports\": {}", ast.imports.len());
            println!("}}");
        }
        OutputFormat::Pretty => {
            println!("┌─ Syntax Check ─────────────────");
            println!("│ Status:       ✓ OK");
            println!("│ Package:      {:?}", ast.package);
            println!("│ Declarations: {}", ast.declarations.len());
            println!("│ Module fields: {}", ast.module_fields.len());
            println!("│ Imports:      {}", ast.imports.len());
            println!("└────────────────────────────────");
        }
    }

    if show_types {
        println!("\nType information:");
        println!("  (Full type checking not yet implemented)");
    }

    Ok(())
}

fn build_hxml(
    file_arg: Option<PathBuf>,
    verbose: bool,
    output_override: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), String> {
    // Auto-detect: if file is .hxml use HXML path, otherwise try rayzor.toml
    if let Some(ref file) = file_arg {
        if file.extension().map(|e| e == "hxml").unwrap_or(false) {
            return build_from_hxml(file, verbose, output_override, dry_run);
        }
    }

    // Try rayzor.toml
    let cwd = std::env::current_dir().map_err(|e| format!("Failed to get cwd: {}", e))?;
    if let Some(root) = compiler::workspace::find_project_root(&cwd) {
        return build_from_manifest(&root, verbose, output_override, dry_run);
    }

    // Fallback: if a file was provided, try it as HXML
    if let Some(file) = file_arg {
        return build_from_hxml(&file, verbose, output_override, dry_run);
    }

    Err("No rayzor.toml or .hxml build file found.\nRun `rayzor init` to create a project, or specify a .hxml file.".to_string())
}

fn build_from_manifest(
    root: &Path,
    verbose: bool,
    output_override: Option<PathBuf>,
    _dry_run: bool,
) -> Result<(), String> {
    use compiler::workspace::{self, RayzorManifest};

    let manifest = workspace::load_manifest(root)?;

    match manifest {
        RayzorManifest::SingleProject(pm) => {
            // Check for HXML delegation
            if let Some(hxml_path) = &pm.hxml {
                let hxml_file = root.join(hxml_path);
                return build_from_hxml(&hxml_file, verbose, output_override, _dry_run);
            }

            let project = workspace::Project {
                root: root.to_path_buf(),
                manifest: pm,
            };

            let entry = project
                .entry_path()
                .ok_or("No entry point in rayzor.toml. Set [project] entry = \"src/Main.hx\"")?;

            println!(
                "📦 Building {} ...",
                project.manifest.name.as_deref().unwrap_or("project")
            );

            if !entry.exists() {
                return Err(format!("Entry file not found: {}", entry.display()));
            }

            if verbose {
                println!("  entry    {}", entry.display());
                if let Some(out) = project.output_path() {
                    println!("  output   {}", out.display());
                }
                for cp in project.resolved_class_paths() {
                    println!("  classpath {}", cp.display());
                }
            }

            let output = output_override.or_else(|| project.output_path());

            // Compile via the standard pipeline
            let source = std::fs::read_to_string(&entry)
                .map_err(|e| format!("Failed to read {}: {}", entry.display(), e))?;
            let mir_module =
                compile_haxe_to_mir(&source, entry.to_str().unwrap_or("unknown"), vec![], &[])?;

            println!("  Compiled {} functions", mir_module.functions.len());

            if let Some(out) = output {
                println!(
                    "  Output: {} (binary serialization coming soon)",
                    out.display()
                );
            }

            println!("✓ Build complete");
            Ok(())
        }
        RayzorManifest::Workspace(wm) => {
            println!("📦 Building workspace ({} members)...", wm.members.len());
            for member in &wm.members {
                let member_dir = root.join(member);
                println!("\n  Building member: {}", member);
                build_from_manifest(&member_dir, verbose, None, _dry_run)?;
            }
            Ok(())
        }
    }
}

fn build_from_hxml(
    file: &Path,
    verbose: bool,
    output_override: Option<PathBuf>,
    dry_run: bool,
) -> Result<(), String> {
    use compiler::hxml::{HxmlConfig, RayzorMode};

    println!("📦 Building from HXML: {}", file.display());

    // Parse HXML file
    let config = HxmlConfig::from_file(&file.to_path_buf())?;

    if verbose {
        println!("\n{}", config.summary());
    }

    // Validate configuration
    config.validate()?;

    let output = output_override.or(config.output.clone());

    if dry_run {
        println!("\n🔍 Dry run - would build:");
        println!("  Main: {:?}", config.main_class);
        println!("  Mode: {:?}", config.mode);
        println!("  Output: {:?}", output);
        println!("  Class paths: {:?}", config.class_paths);
        println!("  Libraries: {}", config.libraries.join(", "));
        return Ok(());
    }

    // Extract main class
    if let Some(main_class) = config.main_class {
        println!("\n✓ Configuration loaded");
        println!("  Main class: {}", main_class);
        println!("  Mode: {:?}", config.mode);
        println!("  Libraries: {}", config.libraries.join(", "));

        // Find the main class file in class paths
        let mut main_file_path = None;
        for cp in &config.class_paths {
            let candidate = cp.join(format!("{}.hx", main_class.replace(".", "/")));
            if candidate.exists() {
                println!("  Found: {}", candidate.display());
                main_file_path = Some(candidate);
                break;
            }
        }

        let main_file = main_file_path
            .ok_or_else(|| format!("Main class file not found in class paths: {}", main_class))?;

        // Execute based on mode
        match config.mode {
            RayzorMode::Jit => {
                println!("\n🔥 JIT mode - compiling and executing...");
                println!("  (Full HXML JIT pipeline coming soon)");
                println!("  For now, use: rayzor jit {}", main_file.display());
                Ok(())
            }
            RayzorMode::Compile => {
                println!("\n🔨 Compile mode - generating native binary...");
                if let Some(out) = output {
                    println!("  Output: {}", out.display());
                    println!("  (Full HXML AOT pipeline coming soon)");
                    println!("  For now, use: rayzor compile {}", main_file.display());
                } else {
                    return Err(
                        "Compile mode requires output file. Use --rayzor-compile <output>"
                            .to_string(),
                    );
                }
                Ok(())
            }
        }
    } else {
        Err("No main class specified in HXML file".to_string())
    }
}

fn compile_file(
    file: PathBuf,
    stage: CompileStage,
    show_ir: bool,
    output: Option<PathBuf>,
    cache: bool,
    cache_dir: Option<PathBuf>,
    release: bool,
) -> Result<(), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};
    use parser::haxe_parser::parse_haxe_file;

    let profile = if release { "release" } else { "debug" };
    let target = CompilationConfig::get_target_triple();
    println!(
        "🔨 Compiling {} to {:?} [{}] [{}]...",
        file.display(),
        stage,
        profile,
        target
    );

    // Read source file
    if !file.exists() {
        return Err(format!("File not found: {}", file.display()));
    }

    let source =
        std::fs::read_to_string(&file).map_err(|e| format!("Failed to read file: {}", e))?;

    // Step 1: Parse
    let ast = parse_haxe_file(file.to_str().unwrap_or("unknown"), &source, false)
        .map_err(|e| format!("Parse error: {}", e))?;

    println!(
        "  parse    {} decls, {} imports",
        ast.declarations.len(),
        ast.imports.len()
    );

    if show_ir {
        println!("\n--- AST ---");
        println!("{:#?}", ast);
    }

    if matches!(stage, CompileStage::Ast) {
        if let Some(output_path) = output {
            let ast_json = format!("{:#?}", ast);
            std::fs::write(&output_path, ast_json)
                .map_err(|e| format!("Failed to write output: {}", e))?;
            println!("  write    {}", output_path.display());
        }
        println!("✓ Stopped at AST stage");
        return Ok(());
    }

    // Create compilation unit with cache configuration
    let cache_dir_resolved = if let Some(dir) = cache_dir {
        Some(dir)
    } else if cache {
        Some(CompilationConfig::get_profile_cache_dir(profile))
    } else {
        None
    };

    let config = CompilationConfig {
        load_stdlib: false,
        enable_cache: cache,
        cache_dir: cache_dir_resolved,
        ..Default::default()
    };

    let unit = CompilationUnit::new(config);

    // For stages beyond AST, compile using our helper with caching support
    let mir_module = if cache {
        if let Some(cached) = unit.try_load_cached(&file) {
            println!("  cache    hit (loaded from BLADE cache)");
            cached
        } else {
            println!("  cache    miss, compiling...");
            let module =
                compile_haxe_to_mir(&source, file.to_str().unwrap_or("unknown"), vec![], &[])?;
            unit.save_to_cache(&file, &module)?;
            module
        }
    } else {
        compile_haxe_to_mir(&source, file.to_str().unwrap_or("unknown"), vec![], &[])?
    };

    println!("  mir      {} functions", mir_module.functions.len());

    for func in mir_module.functions.values() {
        println!(
            "           - {} ({} blocks)",
            func.name,
            func.cfg.blocks.len()
        );
    }

    if show_ir {
        println!("\n--- MIR ---");
        println!("{:#?}", mir_module);
    }

    if matches!(stage, CompileStage::Mir)
        | matches!(stage, CompileStage::Tast)
        | matches!(stage, CompileStage::Hir)
    {
        if let Some(output_path) = output {
            let mir_json = format!("{:#?}", mir_module);
            std::fs::write(&output_path, mir_json)
                .map_err(|e| format!("Failed to write output: {}", e))?;
            println!("  write    {}", output_path.display());
        }
        println!("✓ Stopped at {:?} stage", stage);
        return Ok(());
    }

    // Step 2: Compile to native
    use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};

    let mut config = TieredConfig::from_preset(compiler::codegen::TierPreset::Script);
    config.enable_background_optimization = false;
    config.start_interpreted = false;

    let mut backend = TieredBackend::new(config)?;
    backend.compile_module(mir_module)?;

    println!("  native   code generated");

    if let Some(output_path) = output {
        println!(
            "  output   {} (binary serialization coming soon)",
            output_path.display()
        );
    }

    backend.shutdown();
    println!("✓ Compilation complete");
    Ok(())
}

fn show_info(features: bool, tiers: bool) {
    println!("Rayzor Compiler v0.1.0");
    println!("High-performance Haxe compiler with tiered JIT compilation\n");

    if features || !tiers {
        println!("Features:");
        println!("  ✓ Full Haxe parser");
        println!("  ✓ Type checker (TAST)");
        println!("  ✓ Semantic analysis (HIR)");
        println!("  ✓ SSA form with phi nodes (MIR)");
        println!("  ✓ Tiered JIT compilation (Cranelift)");

        #[cfg(feature = "llvm-backend")]
        println!("  ✓ LLVM backend (Tier 3)");

        #[cfg(not(feature = "llvm-backend"))]
        println!("  ✗ LLVM backend (not enabled)");

        println!();
    }

    if tiers || !features {
        println!("Tiered JIT System:");
        println!("  Tier 0 (Baseline)  - Cranelift 'none'          - ~3ms compile, 1.0x speed");
        println!("  Tier 1 (Standard)  - Cranelift 'speed'         - ~10ms compile, 1.5-3x speed");
        println!("  Tier 2 (Optimized) - Cranelift 'speed_and_size' - ~30ms compile, 3-5x speed");

        #[cfg(feature = "llvm-backend")]
        println!("  Tier 3 (Maximum)   - LLVM aggressive          - ~500ms compile, 5-20x speed");

        #[cfg(not(feature = "llvm-backend"))]
        println!("  Tier 3 (Maximum)   - LLVM (not available)");

        println!("\n  Functions automatically promote based on execution count:");
        println!("    • 100 calls   → Tier 1");
        println!("    • 1,000 calls → Tier 2");
        println!("    • 5,000 calls → Tier 3");
        println!();
    }

    println!("Examples:");
    println!("  cargo run --example test_full_pipeline_tiered");
    println!("  cargo run --example test_tiered_with_loop --features llvm-backend");
}

fn cache_stats(cache_dir: Option<PathBuf>) -> Result<(), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};

    let mut config = CompilationConfig::default();
    if let Some(dir) = cache_dir {
        config.cache_dir = Some(dir);
    }

    let unit = CompilationUnit::new(config);
    let stats = unit.get_cache_stats();

    println!("📊 BLADE Cache Statistics");
    println!("{}", "=".repeat(60));
    println!("Cache directory: {:?}", unit.config.get_cache_dir());
    println!("Cached modules:  {}", stats.cached_modules);
    println!("Total size:      {:.2} MB", stats.total_size_mb());
    println!();

    if stats.cached_modules == 0 {
        println!("No cached modules found.");
        println!("Use --cache flag with 'run' or 'compile' to enable caching.");
    } else {
        println!("Benefits:");
        println!("  • Incremental compilation: ~30x faster for unchanged files");
        println!("  • Dependency caching: Only recompile modified modules");
        println!("  • Version tracking: Automatic invalidation on compiler updates");
    }

    Ok(())
}

fn cache_clear(cache_dir: Option<PathBuf>) -> Result<(), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};

    let mut config = CompilationConfig::default();
    if let Some(dir) = cache_dir {
        config.cache_dir = Some(dir);
    }

    let unit = CompilationUnit::new(config);
    let cache_path = unit.config.get_cache_dir();

    println!("🗑️  Clearing BLADE cache...");
    println!("Cache directory: {:?}", cache_path);

    unit.clear_cache()?;

    println!("✓ Cache cleared successfully");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_bundle(
    files: Vec<PathBuf>,
    output: PathBuf,
    opt_level: u8,
    strip: bool,
    no_compress: bool,
    cache: bool,
    cache_dir: Option<PathBuf>,
    verbose: bool,
) -> Result<(), String> {
    use compiler::ir::optimization::OptimizationLevel;
    use compiler::tools::preblade::{create_bundle, BundleConfig};

    let opt = match opt_level {
        0 => Some(OptimizationLevel::O0),
        1 => Some(OptimizationLevel::O1),
        3 => Some(OptimizationLevel::O3),
        _ => Some(OptimizationLevel::O2),
    };

    let source_files: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().to_string())
        .collect();

    let config = BundleConfig {
        output: output.clone(),
        source_files,
        verbose,
        opt_level: opt,
        strip,
        compress: !no_compress,
        enable_cache: cache,
        cache_dir,
    };

    match create_bundle(&config) {
        Ok(module_count) => {
            println!();
            println!("Bundle created: {}", output.display());
            println!("  Modules: {}", module_count);
            Ok(())
        }
        Err(e) => Err(format!("Bundle creation failed: {}", e)),
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_aot(
    files: Vec<PathBuf>,
    output: Option<PathBuf>,
    target: Option<String>,
    emit: String,
    opt_level: u8,
    strip: bool,
    strip_symbols: bool,
    runtime_dir: Option<PathBuf>,
    linker: Option<String>,
    sysroot: Option<PathBuf>,
    _cache: bool,
    _cache_dir: Option<PathBuf>,
    verbose: bool,
) -> Result<(), String> {
    #[cfg(not(feature = "llvm-backend"))]
    {
        let _ = (
            &files,
            &output,
            &target,
            &emit,
            opt_level,
            strip,
            strip_symbols,
            &runtime_dir,
            &linker,
            &sysroot,
            verbose,
        );
        Err(
            "AOT compilation requires the LLVM backend. Recompile with --features llvm-backend"
                .to_string(),
        )
    }

    #[cfg(feature = "llvm-backend")]
    {
        use compiler::codegen::aot_compiler::OutputFormat;
        use compiler::ir::optimization::OptimizationLevel;
        use compiler::tools::aot_build::{run_aot, AotConfig};

        let output_format = match emit.as_str() {
            "exe" => OutputFormat::Executable,
            "obj" => OutputFormat::ObjectFile,
            "llvm-ir" => OutputFormat::LlvmIr,
            "llvm-bc" => OutputFormat::LlvmBitcode,
            "asm" => OutputFormat::Assembly,
            other => {
                return Err(format!(
                    "Unknown emit format: {}. Use: exe, obj, llvm-ir, llvm-bc, asm",
                    other
                ))
            }
        };

        let opt = match opt_level {
            0 => OptimizationLevel::O0,
            1 => OptimizationLevel::O1,
            3 => OptimizationLevel::O3,
            _ => OptimizationLevel::O2,
        };

        let source_files: Vec<String> = files
            .iter()
            .map(|f| f.to_string_lossy().to_string())
            .collect();

        let config = AotConfig {
            source_files,
            output,
            target_triple: target,
            output_format,
            opt_level: opt,
            strip,
            strip_symbols,
            verbose,
            linker,
            runtime_dir,
            sysroot,
            enable_cache: _cache,
            cache_dir: _cache_dir,
        };

        run_aot(config)
    }
}

fn cmd_preblade(
    _files: Vec<PathBuf>,
    out: Option<PathBuf>,
    list: bool,
    cache_dir: Option<PathBuf>,
    verbose: bool,
) -> Result<(), String> {
    use compiler::tools::preblade::{extract_stdlib_symbols, PrebladeConfig};

    let out_path = out.unwrap_or_else(|| PathBuf::from(".rayzor/blade/stdlib"));

    if !list {
        std::fs::create_dir_all(&out_path)
            .map_err(|e| format!("Error creating output directory: {}", e))?;
    }

    println!("Pre-BLADE: Extracting stdlib symbols");
    println!("  Output: {}", out_path.display());
    println!();

    let config = PrebladeConfig {
        out_path,
        list_only: list,
        verbose,
        cache_dir,
    };

    match extract_stdlib_symbols(&config) {
        Ok((classes, enums, aliases)) => {
            println!();
            println!("Pre-BLADE complete:");
            println!("  Classes: {}", classes);
            println!("  Enums:   {}", enums);
            println!("  Aliases: {}", aliases);
            Ok(())
        }
        Err(e) => Err(format!("Pre-BLADE failed: {}", e)),
    }
}

fn cmd_init(name: Option<String>, workspace: bool) -> Result<(), String> {
    let project_name = name.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "my-project".to_string())
    });

    let dir = PathBuf::from(&project_name);

    if dir.join("rayzor.toml").exists() {
        return Err(format!("rayzor.toml already exists in {}", dir.display()));
    }

    if workspace {
        compiler::workspace::init::init_workspace(&project_name, &dir)?;
        println!(
            "Initialized workspace '{}' at {}",
            project_name,
            dir.display()
        );
        println!("  Created: rayzor.toml, .rayzor/cache/, .gitignore");
        println!();
        println!("Add member projects:");
        println!("  cd {} && rayzor init --name my-lib", project_name);
        println!("  Then add \"my-lib\" to [workspace].members in rayzor.toml");
    } else {
        compiler::workspace::init::init_project(&project_name, &dir)?;
        println!(
            "Initialized project '{}' at {}",
            project_name,
            dir.display()
        );
        println!("  Created: rayzor.toml, src/Main.hx, .rayzor/cache/, .gitignore");
        println!();
        println!("Get started:");
        println!("  cd {} && rayzor run", project_name);
    }

    Ok(())
}

fn cmd_dump(
    file: PathBuf,
    output: Option<PathBuf>,
    opt_level: u8,
    function_filter: Option<String>,
    cfg_only: bool,
) -> Result<(), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};
    use compiler::ir::dump;
    use compiler::ir::optimization::{OptimizationLevel, PassManager};

    println!("🔍 Dumping MIR for {} (O{})...", file.display(), opt_level);

    if !file.exists() {
        return Err(format!("File not found: {}", file.display()));
    }

    let source =
        std::fs::read_to_string(&file).map_err(|e| format!("Failed to read file: {}", e))?;

    // Create compilation unit
    let config = CompilationConfig {
        load_stdlib: true,
        ..Default::default()
    };

    let mut unit = CompilationUnit::new(config);

    // Load stdlib
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add the source file
    unit.add_file(&source, file.to_str().unwrap_or("unknown"))?;

    // Type-check
    if let Err(errors) = unit.lower_to_tast() {
        unit.print_compilation_errors(&errors);
        return Err(format!("Compilation failed with {} error(s)", errors.len()));
    }

    // Get MIR modules
    let mir_modules = unit.get_mir_modules();

    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Get the user module (last one, after stdlib) and clone for optimization
    let mut module = (**mir_modules.last().unwrap()).clone();

    // Apply optimization if requested
    let opt = match opt_level {
        0 => OptimizationLevel::O0,
        1 => OptimizationLevel::O1,
        3 => OptimizationLevel::O3,
        _ => OptimizationLevel::O2,
    };

    // Always run the pass manager — even O0 has correctness passes
    // (InsertFreePass and forced inlining of Haxe `inline` functions)
    if std::env::var("RAYZOR_RAW_MIR").is_ok() {
        eprintln!("(skipping optimization passes — raw MIR dump)");
    } else if std::env::var("RAYZOR_PASS_DEBUG").is_ok() {
        // Debug mode: run passes one at a time and report
        use compiler::ir::optimization::OptimizationPass;
        let passes: Vec<Box<dyn OptimizationPass>> = match opt {
            OptimizationLevel::O0 => {
                let forced_inline_model = compiler::ir::inlining::InliningCostModel {
                    max_inline_size: 15,
                    ..Default::default()
                };
                vec![
                    Box::new(compiler::ir::inlining::InliningPass::with_cost_model(
                        forced_inline_model,
                    )),
                    Box::new(compiler::ir::optimization::DeadCodeEliminationPass::new()),
                    Box::new(compiler::ir::scalar_replacement::ScalarReplacementPass::new()),
                    Box::new(compiler::ir::optimization::CopyPropagationPass::new()),
                    Box::new(compiler::ir::optimization::DeadCodeEliminationPass::new()),
                ]
            }
            _ => {
                let mut pass_manager = PassManager::for_level(opt);
                let _ = pass_manager.run(&mut module);
                vec![]
            }
        };
        for mut pass in passes {
            let result = pass.run_on_module(&mut module);
            // Check main function after each pass for missing instructions
            for func in module.functions.values() {
                if func.name == "main" {
                    // Count total instructions
                    let total_insts: usize =
                        func.cfg.blocks.values().map(|b| b.instructions.len()).sum();
                    let total_blocks = func.cfg.blocks.len();
                    // Check if $4 is defined (second malloc result)
                    let has_ir4 = func.cfg.blocks.values().any(|b| {
                        b.instructions
                            .iter()
                            .any(|inst| inst.dest() == Some(compiler::ir::IrId::new(4)))
                    });
                    eprintln!(
                        "  After '{}': main has {} blocks, {} instructions, $4 defined: {}",
                        pass.name(),
                        total_blocks,
                        total_insts,
                        has_ir4
                    );
                }
            }
            for func in module.functions.values() {
                if func.name == "new" && func.signature.parameters.len() == 1 {
                    let has_unreachable = func.cfg.blocks.values().any(|b| {
                        matches!(
                            b.terminator,
                            compiler::ir::blocks::IrTerminator::Unreachable
                        )
                    });
                    if has_unreachable {
                        eprintln!(
                            "⚠ After pass '{}': new() has UNREACHABLE blocks! modified={}",
                            pass.name(),
                            result.modified
                        );
                        for (bid, b) in &func.cfg.blocks {
                            if matches!(
                                b.terminator,
                                compiler::ir::blocks::IrTerminator::Unreachable
                            ) {
                                eprintln!(
                                    "  {:?}: {} instructions, terminator=unreachable",
                                    bid,
                                    b.instructions.len()
                                );
                            }
                        }
                    } else {
                        eprintln!(
                            "✓ After pass '{}': new() OK (no unreachable blocks)",
                            pass.name()
                        );
                    }
                }
            }
        }
    } else {
        let mut pass_manager = PassManager::for_level(opt);
        let _ = pass_manager.run(&mut module);
    }

    // Generate MIR dump
    let mir_text = if cfg_only {
        // Dump only CFG structure
        let mut output_str = String::new();
        output_str.push_str(&format!("; Module: {}\n", module.name));
        output_str.push_str(&format!("; Functions: {}\n\n", module.functions.len()));

        for func in module.functions.values() {
            if let Some(ref filter) = function_filter {
                if !func.name.contains(filter) {
                    continue;
                }
            }
            output_str.push_str(&dump::dump_cfg(&func.cfg));
            output_str.push('\n');
        }
        output_str
    } else if let Some(ref filter) = function_filter {
        // Dump specific function
        let mut found = false;
        let mut output_str = String::new();

        for func in module.functions.values() {
            if func.name.contains(filter) {
                output_str.push_str(&dump::dump_function(func));
                output_str.push('\n');
                found = true;
            }
        }

        if !found {
            return Err(format!("Function '{}' not found in module", filter));
        }
        output_str
    } else {
        // Dump entire module
        dump::dump_module(&module)
    };

    // Output
    if let Some(output_path) = output {
        std::fs::write(&output_path, &mir_text)
            .map_err(|e| format!("Failed to write output: {}", e))?;
        println!("✓ MIR dumped to {}", output_path.display());
    } else {
        println!();
        println!("{}", mir_text);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// rpkg commands
// ---------------------------------------------------------------------------

fn cmd_rpkg_pack(
    dylib: Option<PathBuf>,
    haxe_dir: PathBuf,
    output: PathBuf,
    name: Option<String>,
) -> Result<(), String> {
    let package_name = name.unwrap_or_else(|| {
        output
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string())
    });

    if let Some(ref dylib_path) = dylib {
        println!(
            "Packing rpkg '{}' from {} + {}",
            package_name,
            dylib_path.display(),
            haxe_dir.display()
        );
        compiler::rpkg::pack::build_from_dylib(&package_name, dylib_path, &haxe_dir, &output)?;
    } else {
        println!(
            "Packing rpkg '{}' from {} (pure Haxe)",
            package_name,
            haxe_dir.display()
        );
        compiler::rpkg::pack::build_from_haxe_dir(&package_name, &haxe_dir, &output)?;
    }

    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "  wrote {} ({:.1} KB)",
        output.display(),
        size as f64 / 1024.0
    );

    Ok(())
}

fn cmd_rpkg_inspect(file: PathBuf) -> Result<(), String> {
    let loaded = compiler::rpkg::load_rpkg(&file)
        .map_err(|e| format!("failed to load {}: {}", file.display(), e))?;

    println!("RPKG: {}", file.display());
    println!("  package: {}", loaded.package_name);
    println!();

    if let Some(ref name) = loaded.plugin_name {
        println!("  Method Table (plugin: {})", name);
        for m in &loaded.methods {
            let kind = if m.is_static { "static" } else { "instance" };
            println!(
                "    {} {}.{}  →  {} (params: {}, ret: {})",
                kind, m.class_name, m.method_name, m.symbol_name, m.param_count, m.return_type
            );
        }
        println!();
    }

    if !loaded.haxe_sources.is_empty() {
        println!("  Haxe Sources ({}):", loaded.haxe_sources.len());
        for path in loaded.haxe_sources.keys() {
            println!("    {}", path);
        }
        println!();
    }

    if loaded.native_lib_bytes.is_some() {
        println!(
            "  Native Library: present for current platform ({}-{})",
            if cfg!(target_os = "macos") {
                "macos"
            } else if cfg!(target_os = "linux") {
                "linux"
            } else {
                "other"
            },
            if cfg!(target_arch = "aarch64") {
                "aarch64"
            } else {
                "x86_64"
            }
        );
    } else {
        println!("  Native Library: not available for current platform");
    }

    Ok(())
}

/// Resolve entry point from rayzor.toml in current or parent directories.
fn resolve_entry_from_manifest() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("Failed to get cwd: {}", e))?;

    let root = compiler::workspace::find_project_root(&cwd)
        .ok_or("No source file specified and no rayzor.toml found.\nRun `rayzor init` to create a project, or specify a .hx file.")?;

    let project = compiler::workspace::load_project(&root)?;

    project.entry_path().ok_or_else(|| {
        "No entry point in rayzor.toml. Set [project] entry = \"src/Main.hx\"".to_string()
    })
}
