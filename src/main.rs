// Doc prose uses `<file>` and `Vec<T>`-style shorthand that rustdoc
// would otherwise misparse.
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::broken_intra_doc_links)]

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

mod compile_helpers;
mod debug;
mod native_libs;
mod rpkg_cmd;
mod tui;
mod wasm_cmd;

// `--features profile` activates the runtime's TrackingAllocator +
// SIGPROF profiler. Off by default — adds ~5 atomic ops per alloc and
// a signal handler. See memory/project_debugger_feasibility.md.
#[cfg(feature = "profile")]
#[global_allocator]
static GLOBAL: rayzor_runtime::TrackingAllocator = rayzor_runtime::TrackingAllocator;

use clap::{Parser, Subcommand, ValueEnum};
use compiler::compiler_plugin::CompilerPlugin;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::{Arc, Mutex};

use compile_helpers::compile_haxe_to_mir;

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

        /// Let CLI --preset replace rayzor.toml [tier] instead of using manifest tier settings
        #[arg(long)]
        preset_override_toml: bool,

        /// Override tier thresholds as interpreter/warm/hot[/blazing], e.g. 1/15/5 or 1/15/5/max
        #[arg(long, value_name = "I/W/H[/B]")]
        tier_thresholds: Option<TierThresholds>,

        /// Override tier profiling sample rate
        #[arg(long, value_name = "N")]
        tier_sample_rate: Option<u64>,

        /// Override start_interpreted in the resolved tier config
        #[arg(long, value_name = "BOOL")]
        tier_start_interpreted: Option<bool>,

        /// Override enable_tier_promotion in the resolved tier config
        #[arg(long, value_name = "BOOL")]
        tier_promotion: Option<bool>,

        /// Disable entry MIR and BLADE caches for incremental compilation
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

        /// Load a raw native plugin dylib by path (repeatable). Supplies the
        /// runtime kernel symbols a `.rzb`/`.hx` needs without a rayzor.toml —
        /// the CLI equivalent of the manifest's `[build] native-libs`.
        #[arg(long = "native-lib", value_name = "FILE")]
        native_libs: Vec<PathBuf>,

        /// Enable or disable safety warnings (use-after-move, etc.)
        #[arg(long, default_value = "on")]
        safety_warnings: String,

        /// Open interactive TUI after execution (scrollable output, search)
        #[arg(short, long)]
        interactive: bool,

        /// Report phase timings as plain lines instead of drawing the TUI.
        /// Also enabled by RAYZOR_PROFILE_COMPILE/RAYZOR_PROFILE_TYPECHECK.
        #[arg(long)]
        plain: bool,

        /// Run in WASM sandbox (compile to WASM, execute via embedded wasmtime)
        #[arg(long)]
        wasm: bool,

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

        /// Generate browser HTML harness (for --target wasm)
        #[arg(long)]
        browser: bool,

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
        files: Vec<PathBuf>,

        /// Output .rzb path
        #[arg(short, long)]
        output: Option<PathBuf>,

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
        #[arg(num_args = 0..)]
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

    /// Investigative debugging toolkit: forensic run, multi-run bench, A/B
    /// compare across git refs, PC → Haxe function resolution, lldb wrapper,
    /// and a live metrics HTTP server with embedded browser dashboard.
    Debug {
        #[command(subcommand)]
        action: debug::DebugCommands,
    },
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

        /// WASM component to embed (universal fallback for all platforms).
        /// Pass a .wasm file path, or use --wasm alone to auto-compile from --haxe-dir.
        #[arg(long, value_name = "FILE", num_args = 0..=1, default_missing_value = "__auto__")]
        wasm: Option<PathBuf>,

        /// JS host module for WASM @:jsImport (repeatable).
        /// Format: MODULE=FILE, e.g. --js-host rayzor-gpu=wasm/gpu_host.js
        #[arg(long, value_name = "MODULE=FILE")]
        js_host: Vec<String>,

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

#[derive(Clone, Debug)]
struct TierThresholds {
    interpreter: u64,
    warm: u64,
    hot: u64,
    blazing: Option<u64>,
}

impl TierThresholds {
    fn apply_to(&self, config: &mut compiler::codegen::TieredConfig) {
        config.profile_config.interpreter_threshold = self.interpreter;
        config.profile_config.warm_threshold = self.warm;
        config.profile_config.hot_threshold = self.hot;
        if let Some(blazing) = self.blazing {
            config.profile_config.blazing_threshold = blazing;
        }
    }
}

impl std::str::FromStr for TierThresholds {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<_> = s
            .split(['/', ',', ':'])
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect();
        if !(parts.len() == 3 || parts.len() == 4) {
            return Err(
                "expected interpreter/warm/hot or interpreter/warm/hot/blazing".to_string(),
            );
        }

        Ok(Self {
            interpreter: parse_threshold_component(parts[0])?,
            warm: parse_threshold_component(parts[1])?,
            hot: parse_threshold_component(parts[2])?,
            blazing: if parts.len() == 4 {
                Some(parse_threshold_component(parts[3])?)
            } else {
                None
            },
        })
    }
}

fn parse_threshold_component(s: &str) -> Result<u64, String> {
    match s.to_ascii_lowercase().as_str() {
        "max" | "never" => Ok(u64::MAX),
        _ => s
            .parse::<u64>()
            .map_err(|_| format!("invalid threshold value `{s}`")),
    }
}

fn should_surface_compile_warning(d: &diagnostics::Diagnostic, verbose: bool) -> bool {
    if d.severity != diagnostics::DiagnosticSeverity::Warning {
        return false;
    }
    let Some(code) = d.code.as_deref() else {
        return false;
    };
    if !code.starts_with('W') {
        return false;
    }

    // W0014 is an advisory "cross-context iface return recovered late"
    // hint. It can be extremely noisy on large Nue graphs and does not
    // change execution; keep it available for annotation work without
    // making normal cold-start runs scroll through it.
    verbose || code != "W0014"
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

#[allow(clippy::too_many_arguments)]
fn resolve_tier_config(
    preset: Preset,
    preset_override_toml: bool,
    tier_thresholds: Option<&TierThresholds>,
    tier_sample_rate: Option<u64>,
    tier_start_interpreted: Option<bool>,
    tier_promotion: Option<bool>,
    manifest_project: Option<&compiler::workspace::Project>,
    verbose: bool,
    release: bool,
) -> compiler::codegen::tiered_backend::TieredConfig {
    // Config selection:
    //   * If the manifest carries an explicit `[tier]` block, it remains the
    //     default source of truth.
    //   * `--preset-override-toml` lets benchmarking runs start from a named
    //     CLI preset instead.
    //   * The narrow `--tier-*` overrides are then applied last, so one-off
    //     sweeps can manipulate thresholds without editing rayzor.toml.
    let manifest_tier_config = if preset_override_toml {
        None
    } else {
        manifest_project.and_then(|p| p.tier_config().cloned())
    };

    let mut config = match manifest_tier_config {
        Some(custom) => compiler::codegen::tiered_backend::TieredConfig::from_preset(
            compiler::codegen::TierPreset::Custom(custom),
        ),
        None => {
            let mut config = compiler::codegen::tiered_backend::TieredConfig::from_preset(
                preset.to_tier_preset(),
            );
            // Preserve the historical native CLI default for ad hoc runs
            // without a manifest-level `[tier]` block. Explicit manifest
            // configs must be able to tune `start_interpreted`.
            config.start_interpreted = false;
            config
        }
    };

    if let Some(thresholds) = tier_thresholds {
        thresholds.apply_to(&mut config);
    }
    if let Some(sample_rate) = tier_sample_rate {
        config.profile_config.sample_rate = sample_rate.max(1);
    }
    if let Some(start_interpreted) = tier_start_interpreted {
        config.start_interpreted = start_interpreted;
    }
    if let Some(enable_tier_promotion) = tier_promotion {
        config.enable_tier_promotion = enable_tier_promotion;
    }
    config.verbosity = if verbose { 2 } else { 0 };
    // In release mode, suppress stack-trace instrumentation overhead even if
    // the preset enables it. Debug runs honour the preset.
    if release {
        config.enable_stack_traces = false;
    }

    config
}

fn run_artifact_build_on_large_stack<F>(name: &'static str, f: F) -> Result<(), String>
where
    F: FnOnce() -> Result<(), String> + Send + 'static,
{
    let stack_size = std::env::var("RAYZOR_ARTIFACT_STACK_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(128)
        .saturating_mul(1024 * 1024);

    let handle = std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(stack_size)
        .spawn(f)
        .map_err(|e| format!("failed to start {name} worker: {e}"))?;

    match handle.join() {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(format!("{name} worker panicked: {message}"))
        }
    }
}

/// On Linux, transparently re-exec under jemalloc (system libjemalloc via
/// LD_PRELOAD) before doing any real work. This routes BOTH the compiler's
/// own Rust allocations AND the JIT'd guest's C `malloc`/`free` — which the
/// backend resolves by name from the process symbol table, so a Rust
/// `#[global_allocator]` alone would NOT cover them — through jemalloc.
/// glibc's default retains freed pages per-arena, inflating long-running RSS
/// (a bounded encode loop grew unbounded on the NUC); jemalloc returns them.
///
/// No-op when: not Linux, `RAYZOR_NO_JEMALLOC=1`, jemalloc is already
/// preloaded (the bench scripts do this), the lib is absent, or we've already
/// re-exec'd (guard env). `exec` replaces the image (same PID/argv); it only
/// returns on error, in which case we fall through on the default allocator.
#[cfg(target_os = "linux")]
fn ensure_jemalloc() {
    use std::os::unix::process::CommandExt;
    use std::path::Path;
    if std::env::var_os("RAYZOR_NO_JEMALLOC").is_some()
        || std::env::var_os("RAYZOR_JEMALLOC_ACTIVE").is_some()
    {
        return;
    }
    if std::env::var("LD_PRELOAD")
        .map(|v| v.contains("jemalloc"))
        .unwrap_or(false)
    {
        return;
    }
    let Some(lib) = [
        "/usr/lib/x86_64-linux-gnu/libjemalloc.so.2",
        "/usr/lib64/libjemalloc.so.2",
        "/usr/lib/libjemalloc.so.2",
    ]
    .into_iter()
    .find(|p| Path::new(p).exists()) else {
        return;
    };
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let preload = match std::env::var("LD_PRELOAD") {
        Ok(v) if !v.is_empty() => format!("{v}:{lib}"),
        _ => lib.to_string(),
    };
    let _ = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("LD_PRELOAD", preload)
        .env("RAYZOR_JEMALLOC_ACTIVE", "1")
        .exec();
}

#[cfg(not(target_os = "linux"))]
fn ensure_jemalloc() {}

fn main() {
    ensure_jemalloc();
    #[cfg(feature = "profile")]
    unsafe {
        rayzor_runtime::ensure_alloc_dump_hooks();
    }
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run {
            file,
            verbose,
            stats,
            tier,
            llvm,
            preset,
            preset_override_toml,
            tier_thresholds,
            tier_sample_rate,
            tier_start_interpreted,
            tier_promotion,
            no_cache,
            cache_dir,
            release,
            rpkg_files,
            native_libs,
            safety_warnings,
            interactive,
            plain,
            wasm,
            program_args,
        } => {
            if wasm {
                wasm_cmd::cmd_run_wasm(
                    file,
                    rpkg_files,
                    safety_warnings != "off",
                    no_cache,
                    program_args,
                )
            } else {
                run_file(
                    file,
                    verbose,
                    stats,
                    tier,
                    llvm,
                    preset,
                    preset_override_toml,
                    tier_thresholds,
                    tier_sample_rate,
                    tier_start_interpreted,
                    tier_promotion,
                    !no_cache,
                    cache_dir,
                    release,
                    rpkg_files,
                    native_libs,
                    safety_warnings != "off",
                    interactive,
                    plain,
                    program_args,
                )
            }
        }
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
            browser,
            target,
        } => {
            if target == "wasm" || target == "wasm-wasi" || target == "wasm32" {
                wasm_cmd::cmd_build_wasm(file, output, target, browser)
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
        } => run_artifact_build_on_large_stack("rayzor-bundle", move || {
            let out_path = output.unwrap_or_else(|| {
                let stem = if !files.is_empty() {
                    files[0]
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                } else if let Ok((entry, manifest)) = resolve_from_manifest() {
                    if let Some(p) = manifest.and_then(|m| m.output_path()) {
                        return p;
                    }
                    entry
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string()
                } else {
                    "bundle".to_string()
                };
                std::path::PathBuf::from(format!("{}.rzb", stem))
            });
            cmd_bundle(
                files,
                out_path,
                opt_level,
                strip,
                no_compress,
                !no_cache,
                cache_dir,
                verbose,
            )
        }),
        Commands::Aot {
            mut files,
            mut output,
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
        } => run_artifact_build_on_large_stack("rayzor-aot", move || {
            if files.is_empty() {
                if let Ok((entry, manifest)) = resolve_from_manifest() {
                    files.push(entry);
                    if output.is_none() {
                        if let Some(project) = manifest {
                            output = project.output_path().map(|p| {
                                p.with_extension(if emit == "gcc" || emit == "exe" {
                                    ""
                                } else {
                                    emit.as_str()
                                })
                            });
                        }
                    }
                }
            }
            if files.is_empty() {
                Err("No source files provided and no rayzor.toml found".to_string())
            } else {
                cmd_aot(
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
                )
            }
        }),
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
                wasm,
                js_host,
                haxe_dir,
                output,
                name,
            } => rpkg_cmd::cmd_rpkg_pack(dylib, os, arch, wasm, js_host, haxe_dir, output, name),
            RpkgAction::Inspect { file } => rpkg_cmd::cmd_rpkg_inspect(file),
            RpkgAction::Install { file } => rpkg_cmd::cmd_rpkg_install(file),
            RpkgAction::Add { name } => rpkg_cmd::cmd_rpkg_add(name),
            RpkgAction::Remove { name } => rpkg_cmd::cmd_rpkg_remove(name),
            RpkgAction::List => rpkg_cmd::cmd_rpkg_list(),
            RpkgAction::Strip {
                input,
                os,
                arch,
                output,
            } => rpkg_cmd::cmd_rpkg_strip(input, os, arch, output),
        },
        Commands::Lsp => rayzor_lsp::run_lsp(),
        Commands::Debug { action } => action.execute().map_err(|e| e.to_string()),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_bundle(
    file: &Path,
    verbose: bool,
    stats: bool,
    preset: Preset,
    llvm: bool,
    preset_override_toml: bool,
    tier_thresholds: Option<TierThresholds>,
    tier_sample_rate: Option<u64>,
    tier_start_interpreted: Option<bool>,
    tier_promotion: Option<bool>,
    release: bool,
    manifest_project: Option<compiler::workspace::Project>,
    cli_rpkg_files: Vec<PathBuf>,
    cli_native_libs: Vec<PathBuf>,
    program_args: Vec<String>,
) -> Result<(), String> {
    use compiler::codegen::tiered_backend::TieredBackend;
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

    let mut effective_rpkg_files = if let Some(project) = manifest_project.as_ref() {
        compiler::workspace::resolve_dependencies(&project.manifest, &project.root)?
    } else {
        Vec::new()
    };
    effective_rpkg_files.extend(cli_rpkg_files.iter().cloned());

    let mut loaded_rpkgs = Vec::new();
    for rpkg_path in &effective_rpkg_files {
        if let Ok(rpkg) = compiler::rpkg::install::RpkgPlugin::load(rpkg_path) {
            loaded_rpkgs.push(rpkg);
        }
    }

    // Runtime kernel symbols come from two sources: the manifest's
    // `[build] native-libs` (when a rayzor.toml is discoverable) and the CLI
    // `--native-lib` flag. The CLI path is what lets a bundle run with no
    // toml shipped — the deployment is `.rzb` + dylib, invoked as
    // `rayzor run app.rzb --native-lib libfoo.dylib --preset …`.
    let mut loaded_native_libs = Vec::new();
    let mut manifest_native_symbols = Vec::new();
    if let Some(project) = manifest_project.as_ref() {
        for lib_path in project.resolved_native_libs() {
            match crate::native_libs::load_manifest_native_lib(&lib_path) {
                Ok((lib, _plugin, runtime_symbols)) => {
                    loaded_native_libs.push(lib);
                    manifest_native_symbols.extend(runtime_symbols);
                }
                // A manifest-listed native-lib that fails to load leaves every
                // extern method it maps unresolved — the JIT then fails later
                // with an opaque "can't resolve symbol <leaf>" at finalize (or a
                // NULL call). Surface it here instead of degrading silently.
                Err(e) => {
                    eprintln!(
                        "  warning: native-lib '{}' failed to load: {} \
                         (its runtime kernels will be unresolved)",
                        lib_path.display(),
                        e
                    );
                }
            }
        }
    }
    for lib_path in &cli_native_libs {
        match crate::native_libs::load_manifest_native_lib(lib_path) {
            Ok((lib, _plugin, runtime_symbols)) => {
                if verbose {
                    println!(
                        "  native   loaded {} ({} symbols)",
                        lib_path.display(),
                        runtime_symbols.len()
                    );
                }
                loaded_native_libs.push(lib);
                manifest_native_symbols.extend(runtime_symbols);
            }
            Err(e) => {
                return Err(format!(
                    "failed to load --native-lib {}: {}",
                    lib_path.display(),
                    e
                ));
            }
        }
    }

    // Get runtime symbols
    let plugin = rayzor_runtime::get_plugin();
    let mut symbols = plugin.runtime_symbols();

    let rpkg_owned_symbols: Vec<(String, *const u8)> = loaded_rpkgs
        .iter()
        .flat_map(|r| r.runtime_symbols.clone())
        .collect();
    for (name, ptr) in &rpkg_owned_symbols {
        let name: &'static str = Box::leak(name.clone().into_boxed_str());
        symbols.push((name, *ptr));
    }
    for (name, ptr) in &manifest_native_symbols {
        let name: &'static str = Box::leak(name.clone().into_boxed_str());
        symbols.push((name, *ptr));
    }

    let _loaded_native_libs = loaded_native_libs;
    let _loaded_rpkgs = loaded_rpkgs;

    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let config = resolve_tier_config(
        preset,
        preset_override_toml,
        tier_thresholds.as_ref(),
        tier_sample_rate,
        tier_start_interpreted,
        tier_promotion,
        manifest_project.as_ref(),
        verbose,
        release,
    );
    // Keep bundle execution consistent with source execution: `--llvm`
    // is a direct request for whole-module LLVM even when the `.rzb` is
    // shipped without its original rayzor.toml.
    let auto_upgrade_to_llvm = config.auto_upgrade_to_llvm_after_main_entry || llvm;

    let mut backend = TieredBackend::with_symbols(config, &symbols_ref)?;

    for module in bundle.modules().iter() {
        backend
            .compile_module(module.clone())
            .map_err(|e| format!("Failed to compile module '{}': {}", module.name, e))?;
    }

    // Run static/vtable initializers BEFORE the LLVM upgrade — the same
    // sequence run_file uses. This is what makes the runner tier-aware:
    // `upgrade_to_llvm` only registers LLVM pointers for functions the tiered
    // backend has already materialized into `function_pointers` (see
    // compile_all_with_llvm's `needed_func_ids`). Running `__vtable_init__`
    // (which bakes method addresses into vtables) and `__init__` forces the
    // reachable set — including the hot decode path reached through vtables —
    // to materialize, so the upgrade actually promotes it. Without this the
    // bundle stayed 100% Cranelift baseline (0 LLVM in --stats) and decoded
    // ~3x slower than the source-JIT path. The backend owns the exactly-once
    // traversal of every same-named per-file hook; manually executing the
    // collected IDs here repeated the startup replay performed during native
    // baseline materialization.
    backend
        .initialize_loaded_modules()
        .map_err(|e| format!("module initialization failed: {}", e))?;

    if auto_upgrade_to_llvm {
        #[cfg(feature = "llvm-backend")]
        {
            let profile_load = verbose || std::env::var_os("RAYZOR_PROFILE_LOAD").is_some();
            let up_t = profile_load.then(std::time::Instant::now);
            if let Err(e) = backend.upgrade_to_llvm() {
                eprintln!(
                    "[tier] LLVM upgrade failed: {} (continuing on Cranelift)",
                    e
                );
            } else if let Some(up_t) = up_t {
                eprintln!(
                    "[tier] Upgraded to LLVM in {:.3}s",
                    up_t.elapsed().as_secs_f64()
                );
            }
        }
        #[cfg(not(feature = "llvm-backend"))]
        {
            // Presets default auto-upgrade on; stay quiet unless verbose.
            if verbose {
                eprintln!(
                    "[tier] auto-upgrade to LLVM unavailable (build lacks the `llvm-backend` feature); continuing on Cranelift"
                );
            }
        }
    }

    rayzor_runtime::haxe_sys::init_program_args(&program_args);

    backend
        .execute_function(entry_func_id, vec![])
        .map_err(|e| format!("Execution failed: {}", e))?;

    if stats {
        let backend_stats = backend.get_statistics();
        println!("{}", backend_stats.format());
        if std::env::var_os("RAYZOR_DUMP_TIERS").is_some() {
            println!("{}", backend.format_tier_listing());
        }
        let beadie_stats = backend.beadie_stats();
        println!(
            "Beadie: adapter={} routes={} (standard={} optimized={}) installs={} (standard={} optimized={}) beads={} (standard={} optimized={}) compiled={} (standard={} optimized={})",
            beadie_stats.adapter_enabled,
            beadie_stats.routes_attempted,
            beadie_stats.standard_routes_attempted,
            beadie_stats.optimized_routes_attempted,
            beadie_stats.installs,
            beadie_stats.standard_installs,
            beadie_stats.optimized_installs,
            beadie_stats.registered_beads,
            beadie_stats.standard_registered_beads,
            beadie_stats.optimized_registered_beads,
            beadie_stats.standard_compiled_beads + beadie_stats.optimized_compiled_beads,
            beadie_stats.standard_compiled_beads,
            beadie_stats.optimized_compiled_beads
        );
    }

    // Execution complete — no banner needed, output speaks for itself
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_file(
    file_arg: Option<PathBuf>,
    verbose: bool,
    stats: bool,
    tier: u8,
    llvm: bool,
    preset: Preset,
    preset_override_toml: bool,
    tier_thresholds: Option<TierThresholds>,
    tier_sample_rate: Option<u64>,
    tier_start_interpreted: Option<bool>,
    tier_promotion: Option<bool>,
    cache_enabled: bool,
    cache_dir: Option<PathBuf>,
    release: bool,
    rpkg_files: Vec<PathBuf>,
    native_libs: Vec<PathBuf>,
    safety_warnings: bool,
    interactive: bool,
    plain: bool,
    program_args: Vec<String>,
) -> Result<(), String> {
    use compiler::codegen::tiered_backend::TieredBackend;

    // Resolve file: from arg or rayzor.toml
    let (file, manifest_project) = match file_arg {
        Some(f) => {
            // Even with explicit file, try to load manifest from its parent directory
            // for class-paths, dependencies, and build settings
            let project_from_file = f.parent().and_then(|p| {
                let abs = if p.is_absolute() {
                    p.to_path_buf()
                } else {
                    std::env::current_dir().unwrap_or_default().join(p)
                };
                compiler::workspace::find_project_root(&abs)
                    .and_then(|root| compiler::workspace::load_project(&root).ok())
            });
            let project = project_from_file.or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|cwd| compiler::workspace::find_project_root(&cwd))
                    .and_then(|root| compiler::workspace::load_project(&root).ok())
            });
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
        return run_bundle(
            &file,
            verbose,
            stats,
            preset,
            llvm,
            preset_override_toml,
            tier_thresholds,
            tier_sample_rate,
            tier_start_interpreted,
            tier_promotion,
            release,
            manifest_project,
            rpkg_files,
            native_libs,
            program_args,
        );
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
    // default:          program output only, no profiling/reporting overhead
    // The reporter collects phase timings; the TUI is only one way to show
    // them. Gating its construction on a terminal meant a piped or CI run
    // recorded nothing at all, so `--verbose` there produced no timings.
    let profile_compile_env = std::env::var_os("RAYZOR_PROFILE_COMPILE").is_some()
        || std::env::var_os("RAYZOR_PROFILE_TYPECHECK").is_some();
    let is_tty = tui::style::is_tty();
    let plain = plain || profile_compile_env || (verbose && !is_tty);
    let want_report = interactive || verbose || plain || profile_compile_env;
    compile_helpers::set_compile_profiling_enabled(want_report);
    let progress_tui = if want_report {
        if plain {
            tui::progress::print_run_banner(
                &file.display().to_string(),
                profile,
                &format!("{:?}", preset),
            );
        }
        Some(tui::progress::ProgressTui::new_with_mode(
            &file.display().to_string(),
            profile,
            &format!("{:?}", preset),
            plain,
        ))
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
    if llvm {
        return Err(
            "LLVM backend not available. Recompile with --features llvm-backend".to_string(),
        );
    }

    let resolved_tier_config = resolve_tier_config(
        preset,
        preset_override_toml,
        tier_thresholds.as_ref(),
        tier_sample_rate,
        tier_start_interpreted,
        tier_promotion,
        manifest_project.as_ref(),
        verbose,
        release,
    );
    // Snapshot the upgrade flag before moving `resolved_tier_config` into the
    // backend. `--llvm` forces a whole-module LLVM compile.
    let auto_upgrade_to_llvm = resolved_tier_config.auto_upgrade_to_llvm_after_main_entry || llvm;

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
    let run_cache_dir = cache_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(".rayzor/blade/cache"));
    if std::env::var("RAYZOR_TRACE_CACHE").is_ok() {
        eprintln!(
            "[cache] run cache_enabled={} cache_dir={:?} entry_dir={:?}",
            cache_enabled, cache_dir, run_cache_dir
        );
    }

    // Resolve manifest [dependencies] → .rpkg paths and merge with any
    // explicit --rpkg flags. CLI flags take precedence by appearing later
    // in the resulting list, which matters when a project depends on a
    // version that the user wants to override locally.
    let mut effective_rpkg_files = if let Some(project) = manifest_project.as_ref() {
        compiler::workspace::resolve_dependencies(&project.manifest, &project.root)?
    } else {
        Vec::new()
    };
    effective_rpkg_files.extend(rpkg_files.iter().cloned());
    let rpkg_files = effective_rpkg_files;

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

    // Load native libs declared via `[build] native-libs = [...]` in
    // the manifest. Same shape as the rpkg native-lib path: dlopen the
    // dylib, call its `plugin_describe()` to get the method table, then
    // build a `NativePlugin`. The library handles are kept alive in
    // `loaded_native_libs` for the duration of compilation.
    //
    // Use case: in-tree native plugins (like nue-plugins/) that ship
    // with the source tree and don't need to go through pack/install.
    // The cdylib path is `[build] native-libs = [...]` in the project's
    // own rayzor.toml; resolution is relative to the manifest's root.
    let mut loaded_native_libs: Vec<libloading::Library> = Vec::new();
    let mut manifest_native_symbols: Vec<(String, *const u8)> = Vec::new();
    let mut manifest_native_lib_inputs: Vec<PathBuf> = Vec::new();
    if let Some(project) = manifest_project.as_ref() {
        for lib_path in project.resolved_native_libs() {
            manifest_native_lib_inputs.push(lib_path.clone());
            match native_libs::load_manifest_native_lib(&lib_path) {
                Ok((lib, plugin, runtime_symbols)) => {
                    loaded_native_libs.push(lib);
                    compiler_plugins.push(Box::new(plugin));
                    manifest_native_symbols.extend(runtime_symbols);
                }
                Err(e) => {
                    eprintln!(
                        "warning: failed to load native lib {}: {}",
                        lib_path.display(),
                        e
                    );
                }
            }
        }
    }
    // CLI `--native-lib` dylibs: same treatment as manifest native-libs, so a
    // source run can be configured without a rayzor.toml.
    for lib_path in &native_libs {
        manifest_native_lib_inputs.push(lib_path.clone());
        match native_libs::load_manifest_native_lib(lib_path) {
            Ok((lib, plugin, runtime_symbols)) => {
                loaded_native_libs.push(lib);
                compiler_plugins.push(Box::new(plugin));
                manifest_native_symbols.extend(runtime_symbols);
            }
            Err(e) => {
                return Err(format!(
                    "failed to load --native-lib {}: {}",
                    lib_path.display(),
                    e
                ));
            }
        }
    }
    let _loaded_native_libs = loaded_native_libs; // keep alive past compile

    // Extract compiler plugins from rpkg packages
    for rpkg in &mut loaded_rpkgs {
        if let Some(cp) = rpkg.compiler_plugin.take() {
            compiler_plugins.push(Box::new(cp));
        }
    }

    // Auto-define the execution tier so tests / code can conditionally
    // compile per backend with `#if jit` / `#if llvm` / `#if interp`.
    // rayzor's JIT is tiered (interp→cranelift→llvm at runtime); `--tier`
    // only sets the START tier, so the default `run` is still the JIT path.
    // The tier label therefore reflects the backend the command targets:
    // `--llvm`/`--tier 3` → llvm, the interpreter-only Embedded preset →
    // interp, otherwise the default JIT path.
    let tier_define: &str = if llvm || tier >= 3 {
        "llvm"
    } else if matches!(preset, Preset::Embedded) {
        "interp"
    } else {
        "jit"
    };
    let mut compile_defines: Vec<String> = vec![tier_define.to_string()];

    // Native source runs now honor manifest `[build.defines]`, matching the
    // WASM command. The current preprocessor stores define presence only, so
    // false/0 values are treated as disabled and other values as enabled.
    if let Some(project) = manifest_project.as_ref() {
        for (key, value) in project.defines() {
            let enabled = value
                .as_deref()
                .map(|v| {
                    let v = v.trim();
                    !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
                })
                .unwrap_or(true);
            if enabled {
                compile_defines.push(key);
            }
        }
    }

    // Auto-define a flag per loaded plugin so library code can guard an
    // optional backend with `#if gpu` and still compile when the plugin is
    // absent. Deriving it from what is actually loaded means the define can
    // never disagree with reality, unlike a user-supplied -D.
    for p in &compiler_plugins {
        let n = p.name();
        let short = n
            .trim_start_matches("lib")
            .trim_start_matches("rayzor_")
            .trim_end_matches("_plugins");
        if !short.is_empty() {
            compile_defines.push(short.to_string());
        }
    }
    compile_defines.sort();
    compile_defines.dedup();

    let raw_mir_requested = std::env::var_os("RAYZOR_RAW_MIR").is_some();
    let opt_level_requested = std::env::var_os("RAYZOR_OPT_LEVEL").is_some();
    let fast_interpreter_start =
        resolved_tier_config.start_interpreted && !auto_upgrade_to_llvm && !opt_level_requested;
    let skip_mir_opt = raw_mir_requested || fast_interpreter_start;

    // Check MIR cache: if source hash matches, skip compile+merge+shake entirely
    // Hash main source + all files in class paths for cache invalidation.
    //
    // Cache key components (folded into a single u64 via DefaultHasher):
    //   1. Entry source bytes (covers entry-file own imports & body).
    //   2. mtime of every `.hx` under manifest_dirs (depth-2 walk, legacy).
    //   3. **Transitive import set hash** — qnames + bytes of every
    //      `.hx` reachable from the entry file's imports. Without this,
    //      changing an imported file's *own* imports (e.g. `Foo.hx` adds
    //      `import nue.Bar;`) leaves the entry source unchanged and the
    //      cache replays a stale MIR graph against today's `.blade`s.
    //   4. **Compiler `BUILD_ID`** — content-derived cache ABI id matching
    //      the per-module `.blade` guard in `compilation.rs`. Without this,
    //      compiler/parser/stdlib changes can keep the top-level `.mir.cache`
    //      valid while every per-module `.blade` is invalidated, producing
    //      replay-vs-fresh-graph drift.
    let source_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        fn hash_file_fingerprint<H: Hasher>(path: &Path, h: &mut H) {
            path.to_string_lossy().hash(h);
            if let Ok(abs) = path.canonicalize() {
                abs.to_string_lossy().hash(h);
            }
            if let Ok(meta) = path.metadata() {
                meta.len().hash(h);
                if let Ok(modified) = meta.modified() {
                    if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                        dur.as_nanos().hash(h);
                    }
                }
            }
        }
        let mut h = DefaultHasher::new();
        "rayzor-run-entry-mir-cache-v2".hash(&mut h);
        source.hash(&mut h);
        file.to_string_lossy().hash(&mut h);
        if let Ok(abs) = file.canonicalize() {
            abs.to_string_lossy().hash(&mut h);
        }
        safety_warnings.hash(&mut h);
        std::env::var("RAYZOR_OPT_LEVEL").ok().hash(&mut h);
        std::env::var("RAYZOR_RAW_MIR").is_ok().hash(&mut h);
        skip_mir_opt.hash(&mut h);
        compile_defines.hash(&mut h);
        for dir in &manifest_dirs {
            dir.to_string_lossy().hash(&mut h);
            if let Ok(abs) = dir.canonicalize() {
                abs.to_string_lossy().hash(&mut h);
            }
        }
        for path in &rpkg_files {
            hash_file_fingerprint(path, &mut h);
        }
        for path in &manifest_native_lib_inputs {
            hash_file_fingerprint(path, &mut h);
        }
        let mut native_symbol_names: Vec<&str> = manifest_native_symbols
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        native_symbol_names.sort_unstable();
        native_symbol_names.hash(&mut h);
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
        // Fold in transitive import-set hash and compiler BUILD_ID.
        let import_hash = compiler::ir::blade::compute_import_set_hash(&source, &manifest_dirs);
        import_hash.hash(&mut h);
        compiler::BUILD_ID.hash(&mut h);
        h.finish()
    };
    let mir_cache_path = {
        if cache_enabled {
            let _ = std::fs::create_dir_all(&run_cache_dir);
        }
        let fname = file.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
        run_cache_dir.join(format!("{}.mir.cache", fname))
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
                        // Replay cached diagnostic strings — warnings and
                        // advice only. Error-severity strings describe the
                        // PREVIOUS compile and replaying them verbatim on a
                        // warm run made every cached start look like it had
                        // just failed (the "spurious cache errors" IMPORT
                        // storm users reported). The stored errors
                        // themselves are tracked by the blade-coherence
                        // F4/F7/F8 work; the replay was wrong independently.
                        // RAYZOR_REPLAY_CACHED_ERRORS=1 restores the old
                        // behavior for debugging.
                        if diag_len > 0 {
                            if let Ok(diag_strings) =
                                postcard::from_bytes::<Vec<String>>(&data[12..12 + diag_len])
                            {
                                let replay_errors =
                                    std::env::var_os("RAYZOR_REPLAY_CACHED_ERRORS").is_some();
                                for s in &diag_strings {
                                    if replay_errors || !s.contains("Error:") {
                                        eprint!("{}", s);
                                    }
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
        let t_compile = progress_handle.is_some().then(std::time::Instant::now);
        let define_refs: Vec<&str> = compile_defines.iter().map(|s| s.as_str()).collect();
        let compile_result = compile_helpers::compile_haxe_to_mir_with_defines_and_cache(
            &source,
            file.to_str().unwrap_or("unknown"),
            compiler_plugins,
            &rpkg_source_dirs,
            safety_warnings,
            &define_refs,
            cache_enabled,
            if cache_enabled {
                Some(run_cache_dir.clone())
            } else {
                None
            },
        )?;
        let mut mir_module = compile_result.module;
        let compile_diagnostics = compile_result.diagnostics;
        if let Some(ref h) = progress_handle {
            if let Some((stdlib, parse, tast, mir)) =
                compile_helpers::LAST_STAGE_MS.lock().ok().and_then(|s| *s)
            {
                h.end_phase("stdlib", stdlib);
                h.end_phase("parse", parse);
                h.end_phase("typecheck", tast);
                if std::env::var_os("RAYZOR_PROFILE_TYPECHECK").is_some() {
                    if let Some(detail) = compile_helpers::LAST_TYPECHECK_DETAIL
                        .lock()
                        .ok()
                        .and_then(|s| *s)
                    {
                        h.end_phase("tc.hdll", detail.hdll_ms);
                        h.end_phase("tc.deps", detail.dependency_ms);
                        h.end_phase("tc.import-scan", detail.import_scan_ms);
                        h.end_phase("tc.imports", detail.import_load_ms + detail.import_hx_ms);
                        h.end_phase("tc.imp-discover", detail.import_discover_ms);
                        h.end_phase("tc.imp-toposort", detail.import_toposort_ms);
                        h.end_phase("tc.imp-compile", detail.import_compile_ms);
                        h.end_phase("tc.imp-cache-load", detail.import_cache_load_ms);
                        h.end_phase("tc.imp-cache-save", detail.import_cache_save_ms);
                        h.end_phase("tc.imp-compile-call", detail.import_compile_call_ms);
                        h.end_phase("tc.user", detail.user_files_ms);
                        h.end_phase("tc.file-parse", detail.file_parse_ms);
                        h.end_phase("tc.macro", detail.macro_ms);
                        h.end_phase("tc.ast", detail.ast_lower_ms);
                        h.end_phase("tc.send-sync", detail.send_sync_ms);
                        h.end_phase("tc.ownership", detail.ownership_ms);
                        h.end_phase("tc.hir", detail.hir_ms);
                        h.end_phase("tc.extern", detail.extern_check_ms);
                        h.end_phase("tc.mir-prep", detail.mir_prep_ms);
                        h.end_phase("tc.mir", detail.mir_ms);
                        h.end_phase("tc.mir-core", detail.mir_lower_core_ms);
                        h.end_phase("tc.merge", detail.stdlib_merge_ms);
                        h.end_phase("tc.mono", detail.monomorphize_ms);
                        eprintln!(
                            "  typecheck-detail: files={} macro_skipped={} imports={} \
                             cache_hit={} cache_miss={} fresh={} typedef_fresh={} \
                             already={}",
                            detail.files_seen,
                            detail.macro_skipped_files,
                            detail.imports_collected,
                            detail.import_cache_hits,
                            detail.import_cache_misses,
                            detail.import_fresh_compiles,
                            detail.import_typedef_fresh,
                            detail.import_already_compiled
                        );
                    }
                }
                h.end_phase("mir", mir);
            }
            if let Some(t_compile) = t_compile {
                h.end_phase("compile", t_compile.elapsed().as_secs_f64() * 1000.0);
            }
        }
        // Surface non-fatal compile WARNINGS (errors already abort the compile).
        // e.g. the untyped-empty-array "uncertain element type" warning. Rendered
        // through the standard diagnostic formatter, like errors.
        {
            // W-prefixed codes only: these normal warnings have no other display
            // path. Safety/ownership warnings (E0382 etc.) are rendered by their
            // own pass — including them here would double-print them.
            let warns: Vec<&diagnostics::Diagnostic> = compile_diagnostics
                .iter()
                .filter(|d| should_surface_compile_warning(d, verbose))
                .collect();
            if !warns.is_empty() {
                let mut source_map = diagnostics::SourceMap::new();
                source_map.add_file(
                    file.to_str().unwrap_or("unknown").to_string(),
                    source.clone(),
                );
                let formatter = diagnostics::ErrorFormatter::with_colors();
                for d in warns {
                    eprintln!("{}", formatter.format_diagnostic(d, &source_map));
                }
            }
        }

        // Tree-shake unused stdlib functions
        {
            if let Some(ref h) = progress_handle {
                h.begin_phase("tree-shake");
            }
            let t_shake = progress_handle.is_some().then(std::time::Instant::now);
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
            // Link forward-declared SIMD MIR-wrapper stubs (cross-module imports
            // leave `unreachable` stubs whose calls otherwise run as no-ops).
            mir_module.link_selfcontained_wrapper_stubs();
            let after = mir_module.functions.len() + mir_module.extern_functions.len();
            if let Some(ref h) = progress_handle {
                if let Some(t_shake) = t_shake {
                    h.end_phase("shake", t_shake.elapsed().as_secs_f64() * 1000.0);
                }
                h.set_shake_stats(before, after);
            }
        }

        // Run O2 pass manager to expand Haxe `inline` functions, apply SRA, and
        // run mid-level MIR optimizations (DCE, const fold, copy prop, CSE,
        // LICM, CFG simplify). Phase 2 closed the last two known O2 regressions
        // (loop unrolling on multi-phi headers, InliningPass dropping a BinOp
        // dest type via phi-fed operands), so O2 is now the run-default.
        //
        // Exception: an interpreter-first, non-LLVM source run is asking for
        // first-instruction latency. In that mode the interpreter can execute
        // raw MIR immediately and the tier system can still promote later.
        // Spending ~1s+ optimizing before `main` defeats the tier contract, so
        // skip MIR optimization unless the user explicitly requested an opt
        // level. Eager LLVM runs keep optimizing up front because LLVM is the
        // point of that mode.
        if !skip_mir_opt {
            if let Some(ref h) = progress_handle {
                h.begin_phase("optimize");
            }
            let t_opt = progress_handle.is_some().then(std::time::Instant::now);
            use compiler::ir::optimization::{OptimizationLevel, PassManager};
            let level = match std::env::var("RAYZOR_OPT_LEVEL").as_deref() {
                Ok("0") => OptimizationLevel::O0,
                Ok("1") => OptimizationLevel::O1,
                Ok("3") => OptimizationLevel::O3,
                _ => OptimizationLevel::O2,
            };
            let mut pass_manager = PassManager::for_level(level);
            let _ = pass_manager.run(&mut mir_module);
            if let Some(ref h) = progress_handle {
                if let Some(t_opt) = t_opt {
                    h.end_phase("optimize", t_opt.elapsed().as_secs_f64() * 1000.0);
                }
            }
        } else {
            // Skipping optimization must not skip the passes correctness
            // depends on; see `PassManager::required_only`.
            use compiler::ir::optimization::PassManager;
            let mut pass_manager = PassManager::required_only();
            let _ = pass_manager.run(&mut mir_module);
            if fast_interpreter_start && verbose {
                eprintln!("[compile] MIR optimize reduced to required passes for startup");
            }
        }

        if let Some(ref h) = progress_handle {
            h.set_functions(mir_module.functions.len());
        }

        // TODO(blade-coherence F4): gate this write on
        // `compile_diagnostics` containing no Error-severity entries, so
        // an erroring compile can never poison later all-cached runs.
        // The abstract-member pre-pass cut the silently-stored errors
        // from ~19 to 8; the guard stays off until the remaining three
        // classes are fixed (else the warm path turns off entirely):
        //   1. "Int32: Invalid assignment target" x4 — abstract
        //      op-assign lowering.
        //   2. "BytesBuffer: Cannot access field 'high'/'low'" — the
        //      surviving Int64 receiver-property edge (second E0100
        //      emitter near hir_to_mir.rs:26514, not the instrumented
        //      one).
        //   3. "FPHelper: POSITIVE/NEGATIVE_INFINITY" — extern stdlib
        //      statics (Math constants) not registered as globals.
        // Diagnose with RAYZOR_DUMP_STORED_DIAGS=1.
        if cache_enabled && std::env::var_os("RAYZOR_DUMP_STORED_DIAGS").is_some() {
            for d in compile_diagnostics
                .iter()
                .filter(|d| d.severity == diagnostics::DiagnosticSeverity::Error)
                .take(8)
            {
                eprintln!("[stored-error] {}", d.message);
            }
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
                    .filter(|d| {
                        d.severity == diagnostics::DiagnosticSeverity::Error
                            || should_surface_compile_warning(d, verbose)
                    })
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
    // Merge runtime symbols from manifest-declared native libs (the
    // `[build] native-libs` entries that were dlopen'd earlier).
    for (name, ptr) in &manifest_native_symbols {
        let name: &'static str = Box::leak(name.clone().into_boxed_str());
        symbols.push((name, *ptr));
    }

    // Keep rpkg dylibs alive until backend is done
    let _loaded_rpkgs = loaded_rpkgs;

    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    // Set up tiered JIT backend.
    let mut backend = TieredBackend::with_symbols(resolved_tier_config, &symbols_ref)?;

    // Compile module with tiered JIT
    if let Some(ref h) = progress_handle {
        h.begin_phase("jit");
    }
    let t_jit = progress_handle.is_some().then(std::time::Instant::now);
    backend.compile_module(mir_module)?;
    if let Some(ref h) = progress_handle {
        if let Some(t_jit) = t_jit {
            h.end_phase("jit", t_jit.elapsed().as_secs_f64() * 1000.0);
        }
        // Stop spinner before execution (output goes to stdout)
        h.finish();
    }
    // Wait for spinner thread to stop
    if let Some(handle) = tui_thread {
        let _ = handle.join();
    }
    if let Some(ref tui) = progress_tui_ref {
        tui.report_plain();
    }

    // Initialize every per-file vtable/static hook exactly once at baseline.
    // LLVM shares the runtime-backed global slots and refreshes only vtable
    // addresses after promotion; repeating static initialization corrupts
    // observable side effects and made startup crashes process-layout-dependent.
    backend
        .initialize_loaded_modules()
        .map_err(|e| format!("module initialization failed: {}", e))?;

    // Once module init has populated globals/vtables, force every reachable
    // function up to LLVM (Maximum tier) before main runs.
    //
    // A failure is fatal when the user asked for `--llvm`. Continuing on
    // Cranelift there reports success for a run that never used the tier that
    // was requested: the program still works, so nothing looks wrong, and every
    // measurement taken from it is a Cranelift measurement wearing another
    // name. A preset that merely prefers LLVM keeps the old behaviour, since
    // nobody asked for the tier by name.
    if auto_upgrade_to_llvm {
        #[cfg(feature = "llvm-backend")]
        {
            let profile_load = verbose || std::env::var_os("RAYZOR_PROFILE_LOAD").is_some();
            let up_t = profile_load.then(std::time::Instant::now);
            if let Err(e) = backend.upgrade_to_llvm() {
                if llvm {
                    return Err(format!(
                        "--llvm was requested but the LLVM tier could not compile this program: {e}"
                    ));
                }
                eprintln!(
                    "[tier] LLVM upgrade failed: {} (continuing on Cranelift)",
                    e
                );
            } else if let Some(up_t) = up_t {
                eprintln!(
                    "[tier] Upgraded to LLVM in {:.3}s",
                    up_t.elapsed().as_secs_f64()
                );
            }
        }
        #[cfg(not(feature = "llvm-backend"))]
        {
            // Presets default auto-upgrade on; only warn when the user
            // explicitly asked for LLVM on a build that can't provide it.
            if llvm {
                eprintln!(
                    "[tier] --llvm requested, but this build was compiled without the `llvm-backend` feature; continuing on Cranelift"
                );
            }
        }
    }

    // Initialize Sys.args() before running Haxe code
    rayzor_runtime::haxe_sys::init_program_args(&program_args);

    // Capture program output by intercepting trace
    let output_capture: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    // Only when the TUI will draw the output itself. In plain mode the program
    // writes straight through, so installing this would swallow it.
    if progress_tui_ref.is_some() && !plain {
        let capture = output_capture.clone();
        rayzor_runtime::haxe_sys::set_trace_callback(Some(Box::new(move |msg: &str| {
            capture.lock().unwrap().push(msg.to_string());
        })));
    }

    // Execute main function
    backend
        .execute_function(main_func_id, vec![])
        .map_err(|e| format!("Execution failed: {}", e))?;

    // A published resume point proves only that one was compiled. Report the
    // transfers that actually happened, so a run where none did is not mistaken
    // for one where transferring was free.
    if compiler::ir::osr::osr_trace_enabled() {
        for (name, site, count) in compiler::ir::osr::transfers_taken() {
            eprintln!("[osr] transferred {name} site=0x{site:x} x{count}");
        }
    }

    if stats {
        let backend_stats = backend.get_statistics();
        let beadie_stats = backend.beadie_stats();
        eprintln!("{}", backend_stats.format());
        if std::env::var_os("RAYZOR_DUMP_TIERS").is_some() {
            eprintln!("{}", backend.format_tier_listing());
        }
        eprintln!(
            "Beadie: adapter={} routes={} (standard={} optimized={}) installs={} (standard={} optimized={}) beads={} (standard={} optimized={}) compiled={} (standard={} optimized={})",
            beadie_stats.adapter_enabled,
            beadie_stats.routes_attempted,
            beadie_stats.standard_routes_attempted,
            beadie_stats.optimized_routes_attempted,
            beadie_stats.installs,
            beadie_stats.standard_installs,
            beadie_stats.optimized_installs,
            beadie_stats.registered_beads,
            beadie_stats.standard_registered_beads,
            beadie_stats.optimized_registered_beads,
            beadie_stats.standard_compiled_beads + beadie_stats.optimized_compiled_beads,
            beadie_stats.standard_compiled_beads,
            beadie_stats.optimized_compiled_beads
        );
    }

    // Remove trace callback
    rayzor_runtime::haxe_sys::set_trace_callback(None);

    // Render TUI
    if let Some(tui) = progress_tui_ref.as_ref().filter(|_| !plain) {
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
                    false,      // preset_override_toml
                    None,       // tier_thresholds
                    None,       // tier_sample_rate
                    None,       // tier_start_interpreted
                    None,       // tier_promotion
                    false,      // cache flag
                    None,       // cache_dir
                    false,      // release
                    Vec::new(), // rpkg_files
                    Vec::new(), // native_libs
                    false,      // safety_warnings
                    false,      // interactive
                    false,      // plain
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

    // Warm into the shared prepared store unless a location was given, so the
    // standard library is lowered once per machine rather than once per
    // project.
    let warm_root = match cache_dir {
        Some(dir) => Some(dir),
        None => CompilationUnit::prepared_cache_root(),
    };

    // Reach the parts of the library a real program reaches. An empty `main`
    // lowers almost nothing, leaving the closure that actually costs a cold
    // compile - `Sys` and everything it pulls in - unprepared.
    let source = r#"
        class Main {
            static function main() {
                var m = new Map<String, Int>();
                m.set("k", 1);
                var b = new StringBuf();
                b.add(StringTools.trim(" v "));
                var xs = [1, 2, 3];
                var total = 0;
                for (x in xs) total += x + m.get("k") + Std.int(Math.sqrt(x));
                Sys.println(b.toString() + Std.string(total));
            }
        }
    "#;

    // An artifact is only found again under the same cache discriminator, and
    // the discriminator follows the defines. Prepare each tier a run can ask
    // for, or the store is written where nothing looks for it.
    struct WarmStats {
        cached_modules: usize,
        total_size_bytes: u64,
    }
    let mut stats = WarmStats {
        cached_modules: 0,
        total_size_bytes: 0,
    };
    for tier_define in ["jit", "llvm", "interp"] {
        let config = CompilationConfig {
            extra_defines: vec![tier_define.to_string()],
            cache_dir: warm_root.clone(),
            ..Default::default()
        };

        let mut unit = CompilationUnit::new(config);
        unit.load_stdlib()
            .map_err(|e| format!("Failed to load stdlib: {}", e))?;
        unit.add_file(source, "warmup.hx")
            .map_err(|e| format!("warmup failed: {}", e))?;
        let _ = unit.lower_to_tast();

        let tier_stats = unit.get_cache_stats();
        stats.cached_modules += tier_stats.cached_modules;
        stats.total_size_bytes += tier_stats.total_size_bytes;
    }
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
    let using_default_cache = cache_dir.is_none();
    if let Some(dir) = cache_dir {
        config.cache_dir = Some(dir);
    }

    let unit = CompilationUnit::new(config);
    let cache_path = unit.config.get_cache_dir();

    println!("🗑️  Clearing Rayzor cache...");
    println!("Cache directory: {:?}", cache_path);

    unit.clear_cache()?;
    if using_default_cache {
        let legacy_mir_cache = PathBuf::from(".rayzor/cache");
        if legacy_mir_cache.exists() {
            std::fs::remove_dir_all(&legacy_mir_cache)
                .map_err(|e| format!("Failed to clear legacy MIR cache: {}", e))?;
            std::fs::create_dir_all(&legacy_mir_cache)
                .map_err(|e| format!("Failed to recreate legacy MIR cache: {}", e))?;
            println!("Legacy MIR cache directory: {:?}", legacy_mir_cache);
        }
    }

    println!("✓ Cache cleared successfully");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_bundle(
    mut files: Vec<PathBuf>,
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

    // Resolve project config from the first file or current dir
    let mut manifest_project = None;
    if files.is_empty() {
        if let Ok((entry, manifest)) = resolve_from_manifest() {
            files.push(entry);
            manifest_project = manifest;
        }
    } else {
        let f = &files[0];
        let file_dir = f.parent().and_then(|p| {
            let abs = if p.is_absolute() {
                p.to_path_buf()
            } else {
                std::env::current_dir().unwrap_or_default().join(p)
            };
            compiler::workspace::find_project_root(&abs)
        });
        manifest_project = file_dir.and_then(|root| compiler::workspace::load_project(&root).ok());
    }

    if files.is_empty() {
        return Err("No source files provided and no rayzor.toml found".to_string());
    }

    let extra_source_dirs: Vec<PathBuf> = manifest_project
        .as_ref()
        .map(|p| p.resolved_class_paths())
        .unwrap_or_default();

    let mut plugins: Vec<Box<dyn CompilerPlugin + 'static>> = Vec::new();
    let mut _loaded_native_libs = Vec::new();
    if let Some(project) = manifest_project.as_ref() {
        let rpkgs = compiler::workspace::resolve_dependencies(&project.manifest, &project.root)
            .unwrap_or_default();
        for rpkg_path in rpkgs {
            if let Ok(mut rpkg) = compiler::rpkg::install::RpkgPlugin::load(&rpkg_path) {
                if let Some(plugin) = rpkg.compiler_plugin.take() {
                    plugins.push(Box::new(plugin));
                }
            }
        }
        for lib_path in project.resolved_native_libs() {
            match crate::native_libs::load_manifest_native_lib(&lib_path) {
                Ok((lib, plugin, _symbols)) => {
                    _loaded_native_libs.push(lib);
                    plugins.push(Box::new(plugin));
                }
                // Fatal for bundling: without the plugin's method mappings the
                // extern calls it backs (e.g. KvCacheQ8.alloc) bake into the
                // .rzb as bare leaf symbols that never resolve on any host.
                Err(e) => {
                    eprintln!(
                        "  warning: native-lib '{}' failed to load at bundle time: {} \
                         (extern kernels it maps will bake in as unresolved symbols)",
                        lib_path.display(),
                        e
                    );
                }
            }
        }
    }

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
        extra_source_dirs,
        plugins,
    };

    match create_bundle(config) {
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
    mut files: Vec<PathBuf>,
    mut output: Option<PathBuf>,
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
    // Resolve project config from the first file or current dir
    let mut manifest_project = None;
    if files.is_empty() {
        if let Ok((entry, manifest)) = resolve_from_manifest() {
            files.push(entry);
            manifest_project = manifest;
            if output.is_none() {
                if let Some(p) = manifest_project.as_ref().and_then(|m| m.output_path()) {
                    output = Some(p.with_extension(if emit == "gcc" || emit == "exe" {
                        ""
                    } else {
                        emit.as_str()
                    }));
                }
            }
        }
    } else {
        let f = &files[0];
        let file_dir = f.parent().and_then(|p| {
            let abs = if p.is_absolute() {
                p.to_path_buf()
            } else {
                std::env::current_dir().unwrap_or_default().join(p)
            };
            compiler::workspace::find_project_root(&abs)
        });
        manifest_project = file_dir.and_then(|root| compiler::workspace::load_project(&root).ok());
    }

    if files.is_empty() {
        return Err("No source files provided and no rayzor.toml found".to_string());
    }

    let extra_source_dirs: Vec<PathBuf> = manifest_project
        .as_ref()
        .map(|p| p.resolved_class_paths())
        .unwrap_or_default();
    let native_link_libs: Vec<PathBuf> = manifest_project
        .as_ref()
        .map(|p| p.resolved_native_libs())
        .unwrap_or_default();

    let mut plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>> = Vec::new();
    // Native libs have to live as long as compilation
    let mut _loaded_native_libs = Vec::new();
    if let Some(project) = manifest_project.as_ref() {
        let rpkgs = compiler::workspace::resolve_dependencies(&project.manifest, &project.root)
            .unwrap_or_default();
        for rpkg_path in rpkgs {
            if let Ok(mut rpkg) = compiler::rpkg::install::RpkgPlugin::load(&rpkg_path) {
                if let Some(plugin) = rpkg.compiler_plugin.take() {
                    plugins.push(Box::new(plugin));
                }
            }
        }
        for lib_path in &native_link_libs {
            match crate::native_libs::load_manifest_native_lib(lib_path) {
                Ok((lib, plugin, _symbols)) => {
                    _loaded_native_libs.push(lib);
                    plugins.push(Box::new(plugin));
                }
                Err(e) => {
                    eprintln!(
                        "warning: failed to load native lib {}: {}",
                        lib_path.display(),
                        e
                    );
                }
            }
        }
    }

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
            target_triple: target,
            strip_symbols,
            verbose,
            linker,
            sysroot,
            runtime_dir,
            extra_source_dirs,
            native_link_libs: native_link_libs.clone(),
            strip,
        };

        println!("Rayzor C Backend");
        match compiler.compile_c_with_plugins(&source_files, &output_path, plugins) {
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
            extra_source_dirs,
            native_link_libs,
        };

        run_aot(config, plugins)
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
