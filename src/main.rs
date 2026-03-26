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

mod tui;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Mutex};

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

        /// Disable BLADE cache for incremental compilation
        #[arg(long)]
        no_cache: bool,

        /// Cache directory (defaults to target/debug/cache or target/release/cache)
        #[arg(long)]
        cache_dir: Option<PathBuf>,

        /// Build with optimizations (uses target/release instead of target/debug)
        #[arg(long)]
        release: bool,

        /// Load .rpkg packages (repeatable)
        #[arg(long = "rpkg", value_name = "FILE")]
        rpkg_files: Vec<PathBuf>,

        /// Enable or disable safety warnings (use-after-move, etc.)
        #[arg(long, default_value = "on")]
        safety_warnings: String,

        /// Open interactive TUI after execution (scrollable output, search)
        #[arg(short, long)]
        interactive: bool,

        /// Arguments to pass to the Haxe program (after --)
        #[arg(last = true)]
        program_args: Vec<String>,
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

        /// Disable BLADE cache for incremental compilation
        #[arg(long)]
        no_cache: bool,

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

        /// Strip debug symbols from output
        #[arg(long)]
        strip: bool,

        /// MIR optimization level (0-3)
        #[arg(long, default_value = "2")]
        opt_level: u8,

        /// Show what would be built without building
        #[arg(long)]
        dry_run: bool,

        /// Target platform: native (default), wasm, wasm-wasi
        #[arg(long, default_value = "native")]
        target: String,
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

        /// Disable BLADE incremental cache
        #[arg(long)]
        no_cache: bool,

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

        /// Disable BLADE incremental cache
        #[arg(long)]
        no_cache: bool,

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

        /// Project template: app (default), lib, benchmark, empty
        #[arg(long, default_value = "app")]
        template: String,

        /// Workspace member projects to create (comma-separated)
        #[arg(long, value_delimiter = ',')]
        members: Option<Vec<String>>,

        /// Generate rayzor.toml from an existing .hxml build file
        #[arg(long)]
        from_hxml: Option<PathBuf>,

        /// Overwrite existing rayzor.toml
        #[arg(long)]
        force: bool,
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

        /// Show before/after optimization diff
        #[arg(long)]
        diff: bool,

        /// Output format: text (default), dot (Graphviz)
        #[arg(long, default_value = "text")]
        format: String,

        /// Open interactive TUI viewer (scrollable, searchable, function list)
        #[arg(short, long)]
        interactive: bool,
    },

    /// Manage .rpkg packages (pack, inspect, install, add, remove, list)
    Rpkg {
        #[command(subcommand)]
        action: RpkgAction,
    },

    /// Start the Language Server Protocol server (for IDE integration)
    Lsp,
}

#[derive(Subcommand)]
enum RpkgAction {
    /// Pack Haxe sources (and optionally native dylibs) into an .rpkg file
    Pack {
        /// Native library to embed (repeatable for multi-platform).
        /// Each --dylib may be followed by --os and --arch to tag the platform.
        /// If --os/--arch are omitted, the current platform is assumed.
        #[arg(long, value_name = "FILE")]
        dylib: Vec<PathBuf>,

        /// OS for the preceding --dylib (macos, linux, windows). Repeatable.
        #[arg(long, value_name = "OS")]
        os: Vec<String>,

        /// Architecture for the preceding --dylib (aarch64, x86_64). Repeatable.
        #[arg(long, value_name = "ARCH")]
        arch: Vec<String>,

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

    /// Install an .rpkg file into the local package registry
    Install {
        /// Path to the .rpkg file
        file: PathBuf,
    },

    /// Add a package to the project's [dependencies] in rayzor.toml
    Add {
        /// Package name (must be installed in the registry)
        name: String,
    },

    /// Remove a package from the project's [dependencies] in rayzor.toml
    Remove {
        /// Package name to remove
        name: String,
    },

    /// List installed packages in the local registry
    List,

    /// Strip an .rpkg to keep only the native lib for a specific platform
    Strip {
        /// Input .rpkg file
        input: PathBuf,

        /// Target OS (defaults to current platform)
        #[arg(long)]
        os: Option<String>,

        /// Target architecture (defaults to current platform)
        #[arg(long)]
        arch: Option<String>,

        /// Output .rpkg path
        #[arg(short, long)]
        output: PathBuf,
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

    /// List all cached modules with sizes and timestamps
    List {
        /// Cache directory
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },

    /// Pre-compile stdlib to cache for faster first runs
    Warm {
        /// Cache directory
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
            no_cache,
            cache_dir,
            release,
            rpkg_files,
            safety_warnings,
            interactive,
            program_args,
        } => run_file(
            file,
            verbose,
            stats,
            tier,
            llvm,
            preset,
            !no_cache,
            cache_dir,
            release,
            rpkg_files,
            safety_warnings != "off",
            interactive,
            program_args,
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
            no_cache,
            cache_dir,
            release,
        } => compile_file(file, stage, show_ir, output, !no_cache, cache_dir, release),
        Commands::Build {
            file,
            verbose,
            output,
            strip,
            opt_level,
            dry_run,
            target,
        } => {
            if target == "wasm" || target == "wasm-wasi" || target == "wasm32" {
                cmd_build_wasm(file, output, target)
            } else {
                build_hxml(file, verbose, output, strip, opt_level, dry_run)
            }
        }
        Commands::Info { features, tiers } => {
            show_info(features, tiers);
            Ok(())
        }
        Commands::Cache { action } => match action {
            CacheAction::Stats { cache_dir } => cache_stats(cache_dir),
            CacheAction::List { cache_dir } => cache_list(cache_dir),
            CacheAction::Warm { cache_dir } => cache_warm(cache_dir),
            CacheAction::Clear { cache_dir } => cache_clear(cache_dir),
        },
        Commands::Bundle {
            files,
            output,
            opt_level,
            strip,
            no_compress,
            no_cache,
            cache_dir,
            verbose,
        } => cmd_bundle(
            files,
            output,
            opt_level,
            strip,
            no_compress,
            !no_cache,
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
            no_cache,
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
            !no_cache,
            cache_dir,
            verbose,
        ),
        Commands::Init {
            name,
            workspace,
            template,
            members,
            from_hxml,
            force,
        } => cmd_init(name, workspace, template, members, from_hxml, force),
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
            diff,
            format,
            interactive,
        } => cmd_dump(
            file,
            output,
            opt_level,
            function,
            cfg_only,
            diff,
            format,
            interactive,
        ),
        Commands::Rpkg { action } => match action {
            RpkgAction::Pack {
                dylib,
                os,
                arch,
                haxe_dir,
                output,
                name,
            } => cmd_rpkg_pack(dylib, os, arch, haxe_dir, output, name),
            RpkgAction::Inspect { file } => cmd_rpkg_inspect(file),
            RpkgAction::Install { file } => cmd_rpkg_install(file),
            RpkgAction::Add { name } => cmd_rpkg_add(name),
            RpkgAction::Remove { name } => cmd_rpkg_remove(name),
            RpkgAction::List => cmd_rpkg_list(),
            RpkgAction::Strip {
                input,
                os,
                arch,
                output,
            } => cmd_rpkg_strip(input, os, arch, output),
        },
        Commands::Lsp => rayzor_lsp::run_lsp(),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

/// Helper function to compile Haxe source through the full pipeline to MIR
/// Uses CompilationUnit for proper multi-file, stdlib-aware compilation
/// Returns the primary MIR module (user code) and any diagnostics (warnings)
fn compile_haxe_to_mir(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
    safety_warnings: bool,
) -> Result<(compiler::ir::IrModule, Vec<diagnostics::Diagnostic>), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};

    // Create compilation unit with stdlib support
    let mut config = CompilationConfig {
        load_stdlib: true, // Enable stdlib for full Haxe compatibility
        emit_safety_warnings: safety_warnings,
        ..Default::default()
    };
    config.pipeline_config = config.pipeline_config.skip_analysis();

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

    // Return the last module (user code). Import MIR modules are merged during
    // compilation (in compile_file_with_shared_state_ex's stdlib renumbering pass).
    let module = (**mir_modules.last().unwrap()).clone();
    let diagnostics = unit.collected_diagnostics.clone();
    Ok((module, diagnostics))
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

    // Execution complete — no banner needed, output speaks for itself
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
    cache_enabled: bool,
    _cache_dir: Option<PathBuf>,
    release: bool,
    rpkg_files: Vec<PathBuf>,
    safety_warnings: bool,
    interactive: bool,
    program_args: Vec<String>,
) -> Result<(), String> {
    use compiler::codegen::tiered_backend::{TieredBackend, TieredConfig};

    // Resolve file: from arg or rayzor.toml
    let (file, manifest_project) = match file_arg {
        Some(f) => {
            // Even with explicit file, try to load manifest from its parent directory
            // for class-paths, dependencies, and build settings
            let file_dir = f.parent().and_then(|p| {
                let abs = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    std::env::current_dir().unwrap_or_default().join(p)
                };
                compiler::workspace::find_project_root(&abs)
            });
            let project = file_dir.and_then(|root| compiler::workspace::load_project(&root).ok());
            (f, project)
        }
        None => resolve_from_manifest()?,
    };

    // Apply manifest config (class paths, cache settings) if resolved from rayzor.toml
    let extra_source_dirs_from_manifest: Vec<PathBuf> = manifest_project
        .as_ref()
        .map(|p| p.resolved_class_paths())
        .unwrap_or_default();

    let profile = if release { "release" } else { "debug" };

    // Handle precompiled .rzb bundles (no TUI for these)
    if file.extension().is_some_and(|ext| ext == "rzb") {
        tui::progress::print_run_banner(
            &file.display().to_string(),
            profile,
            &format!("{:?}", preset),
        );
        return run_bundle(&file, verbose, stats, preset);
    }

    // Handle .hxml build files (no TUI for these)
    if file.extension().is_some_and(|ext| ext == "hxml") {
        tui::progress::print_run_banner(
            &file.display().to_string(),
            profile,
            &format!("{:?}", preset),
        );
        return build_from_hxml(&file, verbose, None, false);
    }

    // TUI modes:
    // -i (interactive): full ratatui TUI with scrollable output, search (after execution)
    // -v (verbose):     spinner during compilation + inline stats after
    // default:          plain output, no TUI overhead
    let use_tui = (interactive || verbose) && tui::style::is_tty();
    let progress_tui = if use_tui {
        let tui = tui::progress::ProgressTui::new(
            &file.display().to_string(),
            profile,
            &format!("{:?}", preset),
        );
        Some(tui)
    } else {
        tui::progress::print_run_banner(
            &file.display().to_string(),
            profile,
            &format!("{:?}", preset),
        );
        None
    };
    let progress_tui_ref = progress_tui.map(Arc::new);
    let progress_handle = progress_tui_ref.as_ref().map(|t| t.handle());
    // Start spinner thread
    let tui_thread = progress_tui_ref.as_ref().map(|tui| {
        let tui = tui.clone();
        std::thread::spawn(move || {
            let _ = tui.run();
        })
    });

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

    // Compiler plugins (from rpkg packages with native libs)
    let mut compiler_plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>> = Vec::new();

    // Load .rpkg packages
    let mut loaded_rpkgs: Vec<compiler::rpkg::install::RpkgPlugin> = Vec::new();
    let mut rpkg_source_dirs: Vec<PathBuf> = Vec::new();
    let mut rpkg_temp_dirs: Vec<PathBuf> = Vec::new();
    // Manifest class paths go into source dirs but NOT temp dirs (they're real, not cleanup targets)
    // eprintln!("[DEBUG] manifest_project={}", manifest_project.is_some());
    // eprintln!("[DEBUG] extra_source_dirs={:?}", extra_source_dirs_from_manifest);
    let manifest_dirs = extra_source_dirs_from_manifest.clone();
    rpkg_source_dirs.extend(extra_source_dirs_from_manifest);
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
                    rpkg_source_dirs.push(tmp_dir.clone());
                    rpkg_temp_dirs.push(tmp_dir);
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

    // Check MIR cache: if source hash matches, skip compile+merge+shake entirely
    // Hash main source + all files in class paths for cache invalidation
    let source_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        source.hash(&mut h);
        // Include modification times of all .hx files in class paths
        for dir in &manifest_dirs {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("hx") {
                        if let Ok(meta) = path.metadata() {
                            if let Ok(modified) = meta.modified() {
                                modified.hash(&mut h);
                            }
                        }
                    }
                    // Also check subdirectories (packages)
                    if path.is_dir() {
                        if let Ok(sub_entries) = std::fs::read_dir(&path) {
                            for sub in sub_entries.flatten() {
                                if sub.path().extension().and_then(|e| e.to_str()) == Some("hx") {
                                    if let Ok(meta) = sub.path().metadata() {
                                        if let Ok(modified) = meta.modified() {
                                            modified.hash(&mut h);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        h.finish()
    };
    let mir_cache_path = {
        let cache_dir = std::path::PathBuf::from(".rayzor/cache");
        let _ = std::fs::create_dir_all(&cache_dir);
        let fname = file.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
        cache_dir.join(format!("{}.mir.cache", fname))
    };

    let (mir_module, _cache_hit) = 'load_mir: {
        // Try loading from MIR cache (source hash must match)
        // Cache includes pre-rendered diagnostic strings for replay.
        if cache_enabled {
            if let Ok(data) = std::fs::read(&mir_cache_path) {
                if data.len() >= 12 {
                    let cached_hash = u64::from_le_bytes(data[..8].try_into().unwrap());
                    let diag_len = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
                    if cached_hash == source_hash && data.len() >= 12 + diag_len {
                        // Replay cached diagnostic strings
                        if diag_len > 0 {
                            if let Ok(diag_strings) =
                                postcard::from_bytes::<Vec<String>>(&data[12..12 + diag_len])
                            {
                                for s in &diag_strings {
                                    eprint!("{}", s);
                                }
                            }
                        }
                        // Load MIR module
                        if let Ok(module) =
                            postcard::from_bytes::<compiler::ir::IrModule>(&data[12 + diag_len..])
                        {
                            break 'load_mir (module, true);
                        }
                    }
                }
            }
        }

        // Full compile pipeline
        if let Some(ref h) = progress_handle {
            h.begin_phase("compile");
        }
        let t_compile = std::time::Instant::now();
        let (mut mir_module, compile_diagnostics) = compile_haxe_to_mir(
            &source,
            file.to_str().unwrap_or("unknown"),
            compiler_plugins,
            &rpkg_source_dirs,
            safety_warnings,
        )?;
        if let Some(ref h) = progress_handle {
            h.end_phase("compile", t_compile.elapsed().as_secs_f64() * 1000.0);
        }

        // Tree-shake unused stdlib functions
        {
            if let Some(ref h) = progress_handle {
                h.begin_phase("tree-shake");
            }
            let t_shake = std::time::Instant::now();
            use compiler::ir::tree_shake;
            let before = mir_module.functions.len() + mir_module.extern_functions.len();
            let mut modules = vec![mir_module];
            if let Some((mod_name, func_name)) = modules.iter().rev().find_map(|m| {
                m.functions
                    .values()
                    .find(|f| f.name == "main" || f.name.ends_with("_main"))
                    .map(|f| (m.name.clone(), f.name.clone()))
            }) {
                tree_shake::tree_shake_bundle(&mut modules, &mod_name, &func_name);
            }
            mir_module = modules.into_iter().next().unwrap();
            let after = mir_module.functions.len() + mir_module.extern_functions.len();
            if let Some(ref h) = progress_handle {
                h.end_phase("shake", t_shake.elapsed().as_secs_f64() * 1000.0);
                h.set_shake_stats(before, after);
            }
        }

        // Run O0 pass manager to expand Haxe `inline` functions and apply SRA
        if std::env::var("RAYZOR_RAW_MIR").is_err() {
            if let Some(ref h) = progress_handle {
                h.begin_phase("optimize");
            }
            let t_opt = std::time::Instant::now();
            use compiler::ir::optimization::{OptimizationLevel, PassManager};
            let mut pass_manager = PassManager::for_level(OptimizationLevel::O0);
            let _ = pass_manager.run(&mut mir_module);
            if let Some(ref h) = progress_handle {
                h.end_phase("optimize", t_opt.elapsed().as_secs_f64() * 1000.0);
            }
        }

        if let Some(ref h) = progress_handle {
            h.set_functions(mir_module.functions.len());
        }

        // Save MIR cache with pre-rendered diagnostic strings
        if cache_enabled {
            // Render diagnostics to strings for cache replay
            let diag_strings: Vec<String> = if !compile_diagnostics.is_empty() {
                let mut source_map = diagnostics::SourceMap::new();
                source_map.add_file(
                    file.to_str().unwrap_or("unknown").to_string(),
                    source.clone(),
                );
                let formatter = diagnostics::ErrorFormatter::with_colors();
                compile_diagnostics
                    .iter()
                    .map(|d| formatter.format_diagnostic(d, &source_map))
                    .collect()
            } else {
                Vec::new()
            };
            let diag_bytes = postcard::to_allocvec(&diag_strings).unwrap_or_default();
            let mut cache_data = source_hash.to_le_bytes().to_vec();
            cache_data.extend((diag_bytes.len() as u32).to_le_bytes());
            cache_data.extend(&diag_bytes);
            if let Ok(serialized) = postcard::to_allocvec(&mir_module) {
                cache_data.extend(serialized);
                let _ = std::fs::write(&mir_cache_path, &cache_data);
            }
        }

        (mir_module, false)
    };

    let total_functions = mir_module.functions.len();

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

    // Keep rpkg dylibs alive until backend is done
    let _loaded_rpkgs = loaded_rpkgs;

    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Set up tiered JIT backend using the selected preset
    let mut config = TieredConfig::from_preset(preset.to_tier_preset());
    config.verbosity = if verbose { 2 } else { 0 };
    config.start_interpreted = false; // Start with JIT for immediate execution
    config.enable_stack_traces = false;

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)?;

    // Compile module with tiered JIT
    if let Some(ref h) = progress_handle {
        h.begin_phase("jit");
    }
    let t_jit = std::time::Instant::now();
    backend.compile_module(mir_module)?;
    if let Some(ref h) = progress_handle {
        h.end_phase("jit", t_jit.elapsed().as_secs_f64() * 1000.0);
        // Stop spinner before execution (output goes to stdout)
        h.finish();
    }
    // Wait for spinner thread to stop
    if let Some(handle) = tui_thread {
        let _ = handle.join();
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

    // Initialize Sys.args() before running Haxe code
    rayzor_runtime::haxe_sys::init_program_args(&program_args);

    // Capture program output by intercepting trace
    let output_capture: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    if progress_tui_ref.is_some() {
        let capture = output_capture.clone();
        rayzor_runtime::haxe_sys::set_trace_callback(Some(Box::new(move |msg: &str| {
            capture.lock().unwrap().push(msg.to_string());
        })));
    }

    // Execute main function
    backend
        .execute_function(main_func_id, vec![])
        .map_err(|e| format!("Execution failed: {}", e))?;

    // Remove trace callback
    rayzor_runtime::haxe_sys::set_trace_callback(None);

    backend.shutdown();

    // Render TUI
    if let Some(ref tui) = progress_tui_ref {
        let captured = output_capture.lock().unwrap();
        let handle = tui.handle();
        for line in captured.iter() {
            handle.add_output_line(line.clone());
        }
        if interactive {
            // Full interactive TUI — stays alive until user quits
            let _ = tui.run_interactive();
        } else {
            // One-shot inline render for -v mode
            let _ = tui.render_final();
        }
    }

    // Clean up temp dirs from rpkg haxe sources (NOT manifest class paths)
    for dir in &rpkg_temp_dirs {
        let _ = std::fs::remove_dir_all(dir);
    }

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
    _strip: bool,
    _opt_level: u8,
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

            let project_name = project.manifest.name.as_deref().unwrap_or("project");

            let entry = match project.entry_path() {
                Some(e) if e.exists() => e,
                Some(e) => return Err(format!("Entry file not found: {}", e.display())),
                None => {
                    // Library project — no entry point, skip build
                    if tui::style::is_tty() {
                        use crossterm::style::Stylize;
                        eprintln!(
                            "  {} {} (library, no entry point)",
                            "\u{2022}".with(crossterm::style::Color::DarkGrey),
                            project_name.with(crossterm::style::Color::DarkGrey),
                        );
                    } else {
                        println!("  {} (library, skipped)", project_name);
                    }
                    return Ok(());
                }
            };

            let class_paths = project.resolved_class_paths();
            let output = output_override.or_else(|| project.output_path());

            // Use TUI progress for build
            let use_tui = tui::style::is_tty();
            let tui_instance = if use_tui {
                let tui = tui::progress::ProgressTui::new(
                    &entry.display().to_string(),
                    "build",
                    project_name,
                );
                Some(std::sync::Arc::new(tui))
            } else {
                tui::progress::print_run_banner(
                    &entry.display().to_string(),
                    "build",
                    project_name,
                );
                None
            };
            let progress = tui_instance.as_ref().map(|t| t.handle());
            let tui_thread = tui_instance.as_ref().map(|t| {
                let t = t.clone();
                std::thread::spawn(move || {
                    let _ = t.run();
                })
            });

            // Compile
            if let Some(ref h) = progress {
                h.begin_phase("compile");
            }
            let t0 = std::time::Instant::now();
            let source = std::fs::read_to_string(&entry)
                .map_err(|e| format!("Failed to read {}: {}", entry.display(), e))?;
            let (mir_module, _compile_diags) = compile_haxe_to_mir(
                &source,
                entry.to_str().unwrap_or("unknown"),
                vec![],
                &class_paths.to_vec(),
                true,
            )?;
            if let Some(ref h) = progress {
                h.end_phase("compile", t0.elapsed().as_secs_f64() * 1000.0);
                h.set_functions(mir_module.functions.len());
            }
            let total_functions = mir_module.functions.len();

            // Produce output bundle (.rzb)
            if let Some(ref out) = output {
                if let Some(ref h) = progress {
                    h.begin_phase("bundle");
                }
                let t_bundle = std::time::Instant::now();

                // Ensure output directory exists
                if let Some(parent) = out.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }

                // Serialize MIR module as .rzb bundle
                let out_path = if out.extension().is_none() {
                    out.with_extension("rzb")
                } else {
                    out.clone()
                };
                let module_name = mir_module.name.clone();
                let bundle =
                    compiler::ir::RayzorBundle::new(vec![mir_module], &module_name, "main", None);
                compiler::ir::save_bundle(&out_path, &bundle)
                    .map_err(|e| format!("Failed to save bundle: {}", e))?;

                if let Some(ref h) = progress {
                    h.end_phase("bundle", t_bundle.elapsed().as_secs_f64() * 1000.0);
                }
            }

            // Stop spinner
            if let Some(ref h) = progress {
                h.finish();
            }
            if let Some(handle) = tui_thread {
                let _ = handle.join();
            }

            // Render final stats
            let func_count = total_functions;
            if let Some(ref tui) = tui_instance {
                if let Some(ref out) = output {
                    let out_path = if out.extension().is_none() {
                        out.with_extension("rzb")
                    } else {
                        out.clone()
                    };
                    let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
                    tui.handle().add_output_line(format!(
                        "{} ({} bytes)",
                        out_path.display(),
                        size
                    ));
                } else {
                    tui.handle()
                        .add_output_line(format!("{} functions compiled", func_count));
                }
                let _ = tui.render_final();
            } else if let Some(ref out) = output {
                let out_path = if out.extension().is_none() {
                    out.with_extension("rzb")
                } else {
                    out.clone()
                };
                let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
                println!("  output   {} ({} bytes)", out_path.display(), size);
            }

            Ok(())
        }
        RayzorManifest::Workspace(wm) => {
            if tui::style::is_tty() {
                use crossterm::style::Stylize;
                eprintln!(
                    " {} workspace ({} members)",
                    "\u{25B6}".with(crossterm::style::Color::Cyan),
                    wm.members.len()
                );
            } else {
                println!("Building workspace ({} members)...", wm.members.len());
            }
            for (i, member) in wm.members.iter().enumerate() {
                let member_dir = root.join(member);
                if tui::style::is_tty() {
                    use crossterm::style::Stylize;
                    eprintln!(
                        "  [{}/{}] {}",
                        (i + 1).to_string().with(crossterm::style::Color::Cyan),
                        wm.members.len(),
                        member.as_str().with(crossterm::style::Color::White).bold(),
                    );
                } else {
                    println!("  [{}/{}] {}", i + 1, wm.members.len(), member);
                }
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
                println!("  Compiling and executing via JIT...\n");
                run_file(
                    Some(main_file),
                    verbose,
                    false, // stats
                    0,     // tier
                    false, // llvm
                    Preset::Application,
                    false,      // cache flag
                    None,       // cache_dir
                    false,      // release
                    Vec::new(), // rpkg_files
                    false,      // safety_warnings
                    false,      // interactive
                    Vec::new(), // program_args
                )
            }
            RayzorMode::Compile => {
                let out = output.ok_or(
                    "Compile mode requires output file. Use --rayzor-compile <output>".to_string(),
                )?;
                println!("  Compiling to native binary: {}\n", out.display());
                use compiler::codegen::aot_compiler::{AotCompiler, OutputFormat};
                let compiler = AotCompiler {
                    output_format: OutputFormat::Executable,
                    verbose,
                    ..Default::default()
                };
                let sources: Vec<String> = vec![main_file.to_string_lossy().to_string()];
                let result = compiler.compile_c(&sources, &out)?;
                println!(
                    "  Compiled: {} ({} bytes)",
                    result.path.display(),
                    result.code_size
                );
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

    // Create compilation unit with cache configuration (cache on by default)
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
            let (module, _compile_diags) = compile_haxe_to_mir(
                &source,
                file.to_str().unwrap_or("unknown"),
                vec![],
                &[],
                true,
            )?;
            unit.save_to_cache(&file, &module)?;
            module
        }
    } else {
        compile_haxe_to_mir(
            &source,
            file.to_str().unwrap_or("unknown"),
            vec![],
            &[],
            true,
        )?
        .0
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

fn show_info(_features: bool, _tiers: bool) {
    if !tui::style::is_tty() {
        println!("rayzor v0.1.0");
        println!("  A next-generation Haxe compiler with 5-tier JIT,");
        println!("  ownership-based memory, and LLVM-powered native codegen");
        println!("  © rayzor-blade.com");
        return;
    }

    use ratatui::{
        backend::CrosstermBackend,
        layout::Constraint,
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{Block, Borders, Paragraph, Row, Table},
        Terminal,
    };

    // Orange color (RGB)
    let orange = Color::Rgb(255, 140, 0);

    let art_raw = include_str!("tui/art.txt");
    let art_lines: Vec<Line> = art_raw
        .lines()
        .map(|line| {
            let spans: Vec<Span> = line
                .chars()
                .map(|c| {
                    if c == '+' {
                        Span::styled("█", Style::default().fg(orange))
                    } else {
                        Span::styled(" ", Style::default())
                    }
                })
                .collect();
            Line::from(spans)
        })
        .collect();

    let info_rows = vec![
        Row::new(vec![
            Span::styled(" version", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "0.1.0",
                Style::default().fg(orange).add_modifier(Modifier::BOLD),
            ),
        ]),
        Row::new(vec![
            Span::styled("", Style::default()),
            Span::styled(
                "A next-generation Haxe compiler with 5-tier JIT,",
                Style::default().fg(Color::White),
            ),
        ]),
        Row::new(vec![
            Span::styled("", Style::default()),
            Span::styled(
                "ownership-based memory, and LLVM-powered native codegen",
                Style::default().fg(Color::White),
            ),
        ]),
        Row::new(vec![
            Span::styled("", Style::default()),
            Span::styled("", Style::default()),
        ]),
        Row::new(vec![
            Span::styled(" compile", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "50-200ms JIT vs 2-5s C++",
                Style::default().fg(Color::Green),
            ),
        ]),
        Row::new(vec![
            Span::styled(" safety", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Ownership and lifetimes safety model",
                Style::default().fg(Color::Green),
            ),
        ]),
        Row::new(vec![
            Span::styled(" concurrency", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Safe and fearless concurrency",
                Style::default().fg(Color::Green),
            ),
        ]),
        Row::new(vec![
            Span::styled(" simd", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "First-class SIMD support",
                Style::default().fg(Color::Green),
            ),
        ]),
        Row::new(vec![
            Span::styled(" embed", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "Embeddable C code via TinyCC",
                Style::default().fg(Color::Green),
            ),
        ]),
        Row::new(vec![
            Span::styled(" cache", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "BLADE incremental + .rzb bundles",
                Style::default().fg(Color::Green),
            ),
        ]),
    ];

    let art_height = art_lines.len() as u16;
    let info_height = info_rows.len() as u16 + 2; // +2 for borders
    let total_height = art_height + info_height;

    let _ = crossterm::terminal::enable_raw_mode();
    let backend = CrosstermBackend::new(std::io::stderr());
    if let Ok(mut terminal) = Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(total_height.min(30)),
        },
    ) {
        let _ = terminal.draw(|frame| {
            let area = frame.area();
            let chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    Constraint::Length(art_height),
                    Constraint::Length(info_height),
                ])
                .split(area);

            frame.render_widget(Paragraph::new(art_lines), chunks[0]);

            let table = Table::new(info_rows, [Constraint::Length(10), Constraint::Min(40)]).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title_bottom(
                        Line::from(Span::styled(
                            " © rayzor-blade.com ",
                            Style::default().fg(Color::DarkGray),
                        ))
                        .right_aligned(),
                    ),
            );
            frame.render_widget(table, chunks[1]);
        });
    }
    let _ = crossterm::terminal::disable_raw_mode();
    eprintln!();
}

fn cache_stats(cache_dir: Option<PathBuf>) -> Result<(), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};
    use ratatui::style::Color;
    use tui::panel::{render_info_panel, InfoRow};

    let mut config = CompilationConfig::default();
    if let Some(dir) = cache_dir {
        config.cache_dir = Some(dir);
    }

    let cache_path = config.get_cache_dir();
    let unit = CompilationUnit::new(config);
    let stats = unit.get_cache_stats();

    let rows = vec![
        InfoRow::new("directory", &cache_path.display().to_string()),
        InfoRow::colored("modules", &stats.cached_modules.to_string(), Color::Cyan),
        InfoRow::colored(
            "total size",
            &format!("{:.2} MB", stats.total_size_mb()),
            Color::Cyan,
        ),
    ];

    let footer = if stats.cached_modules == 0 {
        "run 'rayzor cache warm' to populate"
    } else {
        "incremental: ~30x faster for unchanged files"
    };

    render_info_panel("cache stats", &rows, Some(footer)).map_err(|e| e.to_string())?;

    Ok(())
}

fn cache_list(cache_dir: Option<PathBuf>) -> Result<(), String> {
    use compiler::compilation::CompilationConfig;

    let mut config = CompilationConfig::default();
    if let Some(dir) = cache_dir {
        config.cache_dir = Some(dir);
    }

    let cache_path = config.get_cache_dir();

    // Collect cache entries
    let mut entries: Vec<(String, u64, String)> = Vec::new();
    if cache_path.exists() {
        if let Ok(dir) = std::fs::read_dir(&cache_path) {
            for entry in dir.flatten() {
                let path = entry.path();
                let is_cache = path.extension().and_then(|e| e.to_str()) == Some("blade")
                    || path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.ends_with(".mir.cache"))
                        .unwrap_or(false);
                if is_cache {
                    if let Ok(meta) = path.metadata() {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("?")
                            .to_string();
                        let age = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.elapsed().ok())
                            .map(|d| {
                                if d.as_secs() < 60 {
                                    format!("{}s", d.as_secs())
                                } else if d.as_secs() < 3600 {
                                    format!("{}m", d.as_secs() / 60)
                                } else if d.as_secs() < 86400 {
                                    format!("{}h", d.as_secs() / 3600)
                                } else {
                                    format!("{}d", d.as_secs() / 86400)
                                }
                            })
                            .unwrap_or_else(|| "?".to_string());
                        entries.push((name, meta.len(), age));
                    }
                }
            }
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    if !tui::style::is_tty() {
        println!("BLADE Cache: {}", cache_path.display());
        for (name, size, age) in &entries {
            println!("  {:35} {:>6}KB  {}", name, size / 1024, age);
        }
        let total: u64 = entries.iter().map(|(_, s, _)| s).sum();
        println!("  {} entries, {}KB", entries.len(), total / 1024);
        return Ok(());
    }

    // Render in ratatui inline panel
    use ratatui::{
        backend::CrosstermBackend,
        layout::Constraint,
        style::{Color, Modifier, Style},
        text::{Line, Span},
        widgets::{Block, Borders, Row, Table},
        Terminal,
    };

    let total_size: u64 = entries.iter().map(|(_, s, _)| s).sum();
    let row_count = entries.len() as u16;
    let height = (row_count + 4).min(25); // +4 for borders + header + footer

    crossterm::terminal::enable_raw_mode().map_err(|e| e.to_string())?;
    let backend = CrosstermBackend::new(std::io::stderr());
    let mut terminal = Terminal::with_options(
        backend,
        ratatui::TerminalOptions {
            viewport: ratatui::Viewport::Inline(height),
        },
    )
    .map_err(|e| e.to_string())?;

    terminal
        .draw(|frame| {
            let rows: Vec<Row> = entries
                .iter()
                .map(|(name, size, age)| {
                    Row::new(vec![
                        Span::styled(format!(" {}", name), Style::default().fg(Color::White)),
                        Span::styled(
                            format!("{}KB", size / 1024),
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(age.as_str(), Style::default().fg(Color::DarkGray)),
                    ])
                })
                .collect();

            let table = Table::new(
                rows,
                [
                    Constraint::Min(30),
                    Constraint::Length(8),
                    Constraint::Length(6),
                ],
            )
            .block(
                Block::default()
                    .title(Span::styled(
                        " cache ",
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .title_bottom(
                        Line::from(vec![
                            Span::styled(
                                format!(" {} entries ", entries.len()),
                                Style::default().fg(Color::Cyan),
                            ),
                            Span::styled(
                                format!("{}KB ", total_size / 1024),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ])
                        .right_aligned(),
                    )
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            );

            frame.render_widget(table, frame.area());
        })
        .map_err(|e| e.to_string())?;

    crossterm::terminal::disable_raw_mode().map_err(|e| e.to_string())?;
    eprintln!();

    Ok(())
}

fn cache_warm(cache_dir: Option<PathBuf>) -> Result<(), String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};
    use crossterm::style::Stylize;

    let tty = tui::style::is_tty();

    if tty {
        eprintln!(
            " {} {}",
            "▶".with(crossterm::style::Color::Cyan),
            "Warming cache".with(crossterm::style::Color::White).bold(),
        );
    } else {
        println!("Warming BLADE cache...");
    }

    let mut config = CompilationConfig::default();
    if let Some(dir) = cache_dir {
        config.cache_dir = Some(dir);
    }

    // Step 1: Generate stdlib.bsym (suppress preblade's verbose output)
    let preblade_config = compiler::tools::preblade::PrebladeConfig {
        out_path: std::path::PathBuf::from(".rayzor/blade/stdlib"),
        list_only: false,
        verbose: false,
        cache_dir: None,
    };
    compiler::tools::preblade::extract_stdlib_symbols(&preblade_config)
        .map_err(|e| format!("preblade failed: {}", e))?;

    if tty {
        eprintln!(
            "  {} symbols extracted",
            "✓".with(crossterm::style::Color::Green),
        );
    }

    // Step 2: Compile a minimal file to trigger stdlib caching
    let mut unit = CompilationUnit::new(config);
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;
    let source = "class Main { static function main() {} }";
    unit.add_file(source, "warmup.hx")
        .map_err(|e| format!("warmup failed: {}", e))?;
    let _ = unit.lower_to_tast();

    let stats = unit.get_cache_stats();
    if tty {
        eprintln!(
            "  {} {} modules cached ({})",
            "✓".with(crossterm::style::Color::Green),
            stats
                .cached_modules
                .to_string()
                .with(crossterm::style::Color::Cyan),
            format!("{:.1}KB", stats.total_size_bytes as f64 / 1024.0)
                .with(crossterm::style::Color::DarkGrey),
        );
    } else {
        println!(
            "  {} modules cached ({:.1}KB)",
            stats.cached_modules,
            stats.total_size_bytes as f64 / 1024.0
        );
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
    // C backend does not require LLVM
    if emit == "c" || emit == "gcc" {
        use compiler::codegen::aot_compiler::{AotCompiler, OutputFormat};
        use compiler::ir::optimization::OptimizationLevel;

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

        let is_exe = emit == "gcc";
        let output_path = output.unwrap_or_else(|| {
            let base = std::path::PathBuf::from(&source_files[0])
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if is_exe {
                std::path::PathBuf::from(&base)
            } else {
                std::path::PathBuf::from(format!("{}.c", base))
            }
        });

        let compiler = AotCompiler {
            opt_level: opt,
            output_format: if is_exe {
                OutputFormat::Executable
            } else {
                OutputFormat::CSource
            },
            verbose,
            runtime_dir: runtime_dir.clone(),
            strip,
            ..Default::default()
        };

        println!("Rayzor C Backend");
        match compiler.compile_c(&source_files, &output_path) {
            Ok(result) => {
                println!(
                    "  emit     {} ({} bytes)",
                    result.path.display(),
                    result.code_size
                );
                println!("✓ Build succeeded");
                return Ok(());
            }
            Err(e) => return Err(format!("C compilation failed: {}", e)),
        }
    }

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
            "c" | "gcc" => OutputFormat::CSource,
            other => {
                return Err(format!(
                    "Unknown emit format: {}. Use: exe, obj, llvm-ir, llvm-bc, asm, c, gcc",
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

fn cmd_init(
    name: Option<String>,
    workspace: bool,
    template: String,
    members: Option<Vec<String>>,
    from_hxml: Option<PathBuf>,
    force: bool,
) -> Result<(), String> {
    use compiler::workspace::init::{self, ProjectTemplate};

    // Parse template
    let tmpl = ProjectTemplate::from_str(&template).ok_or_else(|| {
        format!(
            "Unknown template '{}'. Available: {}",
            template,
            ProjectTemplate::all_names().join(", ")
        )
    })?;

    // --from-hxml: generate rayzor.toml from HXML
    if let Some(ref hxml_path) = from_hxml {
        let dir = if let Some(ref n) = name {
            PathBuf::from(n)
        } else {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        };
        if !force && dir.join("rayzor.toml").exists() {
            return Err(format!(
                "rayzor.toml already exists in {}. Use --force to overwrite.",
                dir.display()
            ));
        }
        std::fs::create_dir_all(&dir).ok();
        init::init_from_hxml(hxml_path, &dir)?;
        println!(
            "Migrated {} to rayzor.toml at {}",
            hxml_path.display(),
            dir.display()
        );
        return Ok(());
    }

    // Determine project name and directory
    let project_name = name.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "my-project".to_string())
    });
    let dir = PathBuf::from(&project_name);

    if !force && dir.join("rayzor.toml").exists() {
        return Err(format!(
            "rayzor.toml already exists in {}. Use --force to overwrite.",
            dir.display()
        ));
    }

    std::fs::create_dir_all(&dir).ok();

    if workspace {
        let member_list = members.unwrap_or_default();
        init::init_workspace(&project_name, &dir, &member_list)?;

        let mut rows = vec![
            tui::panel::InfoRow::colored("type", "workspace", ratatui::style::Color::Cyan),
            tui::panel::InfoRow::new("path", &dir.display().to_string()),
            tui::panel::InfoRow::new("files", "rayzor.toml, .rayzor/cache/, .gitignore"),
        ];
        if !member_list.is_empty() {
            rows.push(tui::panel::InfoRow::colored(
                "members",
                &member_list.join(", "),
                ratatui::style::Color::Green,
            ));
        }
        let hint = if member_list.is_empty() {
            format!("cd {} && rayzor init --name my-app", project_name)
        } else {
            format!("cd {}/{} && rayzor run", project_name, member_list[0])
        };
        let _ = tui::panel::render_info_panel(&project_name, &rows, Some(&hint));
    } else {
        if let Some((entry, _)) = init::detect_existing_sources(&dir) {
            let _ = tui::panel::render_message_panel(
                "detected",
                &[&format!("Existing sources: {}", entry)],
            );
        }

        init::init_project(&project_name, &dir, tmpl)?;

        let files = match tmpl {
            ProjectTemplate::App | ProjectTemplate::Benchmark => {
                "rayzor.toml, src/Main.hx, .gitignore"
            }
            ProjectTemplate::Lib => "rayzor.toml, src/<Name>.hx, .gitignore",
            ProjectTemplate::Empty => "rayzor.toml, .gitignore",
        };
        let rows = vec![
            tui::panel::InfoRow::colored("type", &template, ratatui::style::Color::Cyan),
            tui::panel::InfoRow::new("path", &dir.display().to_string()),
            tui::panel::InfoRow::new("files", files),
        ];
        let _ = tui::panel::render_info_panel(
            &project_name,
            &rows,
            Some(&format!("cd {} && rayzor run", project_name)),
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_dump(
    file: PathBuf,
    output: Option<PathBuf>,
    opt_level: u8,
    function_filter: Option<String>,
    cfg_only: bool,
    diff: bool,
    format: String,
    interactive: bool,
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

    // Try loading manifest for class paths
    let manifest_class_paths: Vec<std::path::PathBuf> = {
        let cwd = std::env::current_dir().unwrap_or_default();
        compiler::workspace::find_project_root(&cwd)
            .and_then(|root| compiler::workspace::load_project(&root).ok())
            .map(|p| p.resolved_class_paths())
            .unwrap_or_default()
    };

    // Create compilation unit
    let mut config = CompilationConfig {
        load_stdlib: true,
        ..Default::default()
    };
    config.pipeline_config = config.pipeline_config.skip_analysis();

    let mut unit = CompilationUnit::new(config);

    // Add class paths from manifest
    for dir in &manifest_class_paths {
        unit.add_source_path(dir.clone());
    }

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

    // Save pre-optimization state for --diff
    let pre_opt_text = if diff {
        Some(dump::dump_module(&module))
    } else {
        None
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

    // Handle --format dot: emit Graphviz DOT
    if format == "dot" {
        let mut dot = String::from(
            "digraph MIR {\n  rankdir=TB;\n  node [shape=box, fontname=\"monospace\"];\n\n",
        );
        for func in module.functions.values() {
            if let Some(ref filter) = function_filter {
                if !func.name.contains(filter) {
                    continue;
                }
            }
            dot.push_str(&format!("  subgraph cluster_{} {{\n", func.id.0));
            dot.push_str(&format!("    label=\"{}\";\n", func.name));
            for (block_id, block) in &func.cfg.blocks {
                let inst_count = block.instructions.len();
                dot.push_str(&format!(
                    "    {} [label=\"{} ({} insts)\"];\n",
                    block_id, block_id, inst_count
                ));
                match &block.terminator {
                    compiler::ir::blocks::IrTerminator::Branch { target } => {
                        dot.push_str(&format!("    {} -> {};\n", block_id, target));
                    }
                    compiler::ir::blocks::IrTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        dot.push_str(&format!(
                            "    {} -> {} [label=\"T\"];\n",
                            block_id, true_target
                        ));
                        dot.push_str(&format!(
                            "    {} -> {} [label=\"F\"];\n",
                            block_id, false_target
                        ));
                    }
                    _ => {}
                }
            }
            dot.push_str("  }\n\n");
        }
        dot.push_str("}\n");

        if let Some(output_path) = output {
            std::fs::write(&output_path, &dot).map_err(|e| format!("Failed to write: {}", e))?;
            println!(
                "✓ DOT written to {} (pipe to: dot -Tpng -o graph.png)",
                output_path.display()
            );
        } else {
            println!("{}", dot);
        }
        return Ok(());
    }

    // Handle --diff: show before/after optimization
    if diff {
        let post_opt_text = dump::dump_module(&module);
        let pre = pre_opt_text.unwrap_or_default();

        let pre_lines: Vec<&str> = pre.lines().collect();
        let post_lines: Vec<&str> = post_opt_text.lines().collect();

        if interactive && tui::style::is_tty() {
            // Show diff in interactive TUI
            let diff_text = format_diff(&pre_lines, &post_lines);
            tui::mir_viewer::run_mir_viewer(
                &diff_text,
                &format!("{} (diff)", module.name),
                module.functions.len(),
            )
            .map_err(|e| format!("TUI error: {}", e))?;
        } else {
            // Print simple diff
            println!("; Before optimization (O0):");
            println!("; {} lines", pre_lines.len());
            println!("; After optimization (O{}):", opt_level);
            println!("; {} lines", post_lines.len());
            println!(
                "; Delta: {} lines",
                post_lines.len() as isize - pre_lines.len() as isize
            );
            println!();
            println!("{}", post_opt_text);
        }
        return Ok(());
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
    if interactive && tui::style::is_tty() {
        tui::mir_viewer::run_mir_viewer(&mir_text, &module.name, module.functions.len())
            .map_err(|e| format!("TUI error: {}", e))?;
    } else if let Some(output_path) = output {
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
// build command (AOT compilation to WASM, C, etc.)
// ---------------------------------------------------------------------------

fn cmd_build_wasm(
    file: Option<PathBuf>,
    output: Option<PathBuf>,
    target: String,
) -> Result<(), String> {
    let file = file.ok_or_else(|| "file path required for WASM build".to_string())?;
    let source = std::fs::read_to_string(&file)
        .map_err(|e| format!("failed to read {}: {}", file.display(), e))?;
    let filename = file.to_string_lossy().to_string();

    println!("Building {} [target: {}]...", file.display(), target);

    // Compile Haxe → MIR using the standard pipeline
    let config = compiler::compilation::CompilationConfig::default();
    let mut unit = compiler::compilation::CompilationUnit::new(config);
    unit.set_source_paths(vec![PathBuf::from("compiler/haxe-std")]);

    unit.add_file(&filename, &source)
        .map_err(|e| format!("parse error: {:?}", e))?;

    let _typed_files = unit.lower_to_tast().map_err(|errors| {
        for e in &errors {
            eprintln!("Error: {}", e.message);
        }
        format!("{} compilation error(s)", errors.len())
    })?;

    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        return Err("no MIR modules produced".to_string());
    }
    let mir_module = (**mir_modules.last().unwrap()).clone();

    // MIR → WASM via wasm-encoder
    let wasm_bytes = compiler::codegen::wasm_backend::WasmBackend::compile(
        &[&mir_module],
        Some("main"),
    )?;

    let out_path = output.unwrap_or_else(|| file.with_extension("wasm"));
    std::fs::write(&out_path, &wasm_bytes)
        .map_err(|e| format!("failed to write {}: {}", out_path.display(), e))?;

    println!(
        "  wrote {} ({:.1} KB, {} functions)",
        out_path.display(),
        wasm_bytes.len() as f64 / 1024.0,
        mir_module.functions.len()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// rpkg commands
// ---------------------------------------------------------------------------

fn cmd_rpkg_pack(
    dylibs: Vec<PathBuf>,
    os_tags: Vec<String>,
    arch_tags: Vec<String>,
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

    if dylibs.is_empty() {
        println!(
            "Packing rpkg '{}' from {} (pure Haxe)",
            package_name,
            haxe_dir.display()
        );
        compiler::rpkg::pack::build_from_haxe_dir(&package_name, &haxe_dir, &output)?;
    } else {
        // Build platform entries: pair each dylib with its os/arch tag
        let current_os = if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "windows"
        } else {
            "unknown"
        };
        let current_arch = if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else {
            "unknown"
        };

        let mut platform_dylibs = Vec::new();
        for (i, dylib_path) in dylibs.iter().enumerate() {
            let os = os_tags.get(i).map(|s| s.as_str()).unwrap_or(current_os);
            let arch = arch_tags.get(i).map(|s| s.as_str()).unwrap_or(current_arch);
            platform_dylibs.push((dylib_path.as_path(), os, arch));
        }

        println!(
            "Packing rpkg '{}' with {} native lib(s) + {}",
            package_name,
            platform_dylibs.len(),
            haxe_dir.display()
        );
        for (path, os, arch) in &platform_dylibs {
            println!("  {}-{}: {}", os, arch, path.display());
        }

        compiler::rpkg::pack::build_from_dylibs(
            &package_name,
            &platform_dylibs,
            &haxe_dir,
            &output,
        )?;
    }

    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "  wrote {} ({:.1} KB)",
        output.display(),
        size as f64 / 1024.0
    );

    Ok(())
}

fn cmd_rpkg_strip(
    input: PathBuf,
    os: Option<String>,
    arch: Option<String>,
    output: PathBuf,
) -> Result<(), String> {
    let target_os = os.unwrap_or_else(|| {
        if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else {
            "windows"
        }
        .to_string()
    });
    let target_arch = arch.unwrap_or_else(|| {
        if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "x86_64"
        }
        .to_string()
    });

    println!(
        "Stripping {} → {} (target: {}-{})",
        input.display(),
        output.display(),
        target_os,
        target_arch
    );

    compiler::rpkg::strip_rpkg(&input, &target_os, &target_arch, &output)
        .map_err(|e| format!("strip failed: {}", e))?;

    let size = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "  wrote {} ({:.1} KB)",
        output.display(),
        size as f64 / 1024.0
    );

    Ok(())
}

fn cmd_rpkg_install(file: PathBuf) -> Result<(), String> {
    use compiler::rpkg::registry::LocalRegistry;

    let mut registry = LocalRegistry::open_default()?;
    let entry = registry.install(&file)?;

    let rows = vec![
        tui::panel::InfoRow::colored("Package", &entry.name, ratatui::style::Color::Cyan),
        tui::panel::InfoRow::new("Size", &format_bytes(entry.size_bytes)),
        tui::panel::InfoRow::new("Haxe files", &entry.haxe_file_count.to_string()),
        tui::panel::InfoRow::new("Native", if entry.has_native { "yes" } else { "no" }),
        tui::panel::InfoRow::new("Location", &registry.root_dir().display().to_string()),
    ];
    let _ = tui::panel::render_info_panel("Package Installed", &rows, None);
    Ok(())
}

fn cmd_rpkg_add(name: String) -> Result<(), String> {
    use compiler::rpkg::registry::LocalRegistry;

    // Verify the package is installed in the registry
    let registry = LocalRegistry::open_default()?;
    if registry.get(&name).is_none() {
        return Err(format!(
            "Package '{}' is not installed. Run `rayzor rpkg install <file.rpkg>` first.",
            name
        ));
    }

    // Find rayzor.toml in current directory
    let manifest_path = std::path::PathBuf::from("rayzor.toml");
    if !manifest_path.exists() {
        return Err("No rayzor.toml found in current directory".to_string());
    }

    let content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read rayzor.toml: {}", e))?;

    // Check if dependency already exists
    if content.contains("[dependencies]") && content.contains(&format!("{} ", name)) {
        return Err(format!(
            "Dependency '{}' already exists in rayzor.toml",
            name
        ));
    }

    // Append [dependencies] section if missing, or add to existing
    let updated = if content.contains("[dependencies]") {
        // Add to existing section
        content.replace(
            "[dependencies]",
            &format!("[dependencies]\n{} = {{ rpkg = \"{}\" }}", name, name),
        )
    } else {
        format!(
            "{}\n[dependencies]\n{} = {{ rpkg = \"{}\" }}\n",
            content.trim_end(),
            name,
            name
        )
    };

    std::fs::write(&manifest_path, updated)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let rows = vec![
        tui::panel::InfoRow::colored("Added", &name, ratatui::style::Color::Green),
        tui::panel::InfoRow::new("Source", "rpkg registry"),
    ];
    let _ = tui::panel::render_info_panel("Dependency Added", &rows, None);
    Ok(())
}

fn cmd_rpkg_remove(name: String) -> Result<(), String> {
    let manifest_path = std::path::PathBuf::from("rayzor.toml");
    if !manifest_path.exists() {
        return Err("No rayzor.toml found in current directory".to_string());
    }

    let content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read rayzor.toml: {}", e))?;

    // Remove the dependency line
    let lines: Vec<&str> = content.lines().collect();
    let filtered: Vec<&str> = lines
        .into_iter()
        .filter(|line| {
            let trimmed = line.trim();
            // Remove lines like: mylib = { rpkg = "mylib" } or mylib = "1.0"
            !(trimmed.starts_with(&name) && (trimmed.contains("=") && !trimmed.starts_with("[")))
        })
        .collect();

    let updated = filtered.join("\n") + "\n";
    std::fs::write(&manifest_path, updated)
        .map_err(|e| format!("Failed to write rayzor.toml: {}", e))?;

    let rows = vec![tui::panel::InfoRow::colored(
        "Removed",
        &name,
        ratatui::style::Color::Yellow,
    )];
    let _ = tui::panel::render_info_panel("Dependency Removed", &rows, None);
    Ok(())
}

fn cmd_rpkg_list() -> Result<(), String> {
    use compiler::rpkg::registry::LocalRegistry;

    let registry = LocalRegistry::open_default()?;
    let packages = registry.list();

    if packages.is_empty() {
        let rows = vec![tui::panel::InfoRow::new("Status", "No packages installed")];
        let _ = tui::panel::render_info_panel(
            "Package Registry",
            &rows,
            Some("Install with: rayzor rpkg install <file.rpkg>"),
        );
        return Ok(());
    }

    let mut rows = Vec::new();
    for (name, entry) in packages {
        let info = format!(
            "{} | {} hx files{}",
            format_bytes(entry.size_bytes),
            entry.haxe_file_count,
            if entry.has_native { " | native" } else { "" }
        );
        rows.push(tui::panel::InfoRow::colored(
            name,
            &info,
            if entry.has_native {
                ratatui::style::Color::Magenta
            } else {
                ratatui::style::Color::Cyan
            },
        ));
    }

    let _ = tui::panel::render_info_panel(
        &format!("Package Registry ({} packages)", packages.len()),
        &rows,
        Some(&registry.root_dir().display().to_string()),
    );
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
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
/// Simple line-level diff for MIR before/after optimization.
fn format_diff(before: &[&str], after: &[&str]) -> String {
    let mut result = String::new();
    result.push_str("; === DIFF: before → after optimization ===\n");
    result.push_str(&format!(
        "; before: {} lines, after: {} lines\n\n",
        before.len(),
        after.len()
    ));

    // Show the optimized output with markers for new function headers
    for line in after {
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Resolve entry file and optional project config from rayzor.toml.
fn resolve_from_manifest() -> Result<(PathBuf, Option<compiler::workspace::Project>), String> {
    let cwd = std::env::current_dir().map_err(|e| format!("Failed to get cwd: {}", e))?;

    let root = compiler::workspace::find_project_root(&cwd)
        .ok_or("No source file specified and no rayzor.toml found.\nRun `rayzor init` to create a project, or specify a .hx file.")?;

    let project = compiler::workspace::load_project(&root)?;

    let entry = project.entry_path().ok_or_else(|| {
        "No entry point in rayzor.toml. Set [project] entry = \"src/Main.hx\"".to_string()
    })?;

    Ok((entry, Some(project)))
}
