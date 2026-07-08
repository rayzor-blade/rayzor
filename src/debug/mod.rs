//! `rayzor debug` — investigative debugging toolkit.
//!
//! Promotes the forensic infrastructure we built while tracking down the
//! content-dependent SIGTRAP (commit 139c4f5) into named CLI commands so
//! the next investigation doesn't start from raw env vars and one-off
//! lldb scripts.
//!
//! Six subcommands, all gated behind `rayzor debug <name>`:
//!
//! - `run`     — drop-in for `rayzor run` with forensic env vars wired up
//! - `bench`   — run a Haxe file N times, capture per-run metrics, report
//!   min/median/mean/max with success rate (filters SIGTRAPs)
//! - `compare` — A/B between two git refs on the same input; auto-restores
//! - `resolve` — turn a list of hex PCs into Haxe function names + line
//!   numbers by joining against the per-PID JIT symbol map
//! - `lldb`    — launch the file under lldb with SIGTRAP/SIGSEGV pre-armed
//!   and a ready-to-use auto-backtrace command (macOS only)
//! - `server`  — start a small HTTP server that exposes live metrics +
//!   an embedded browser dashboard
//!
//! All subcommands use the existing runtime infrastructure (sigaction
//! crash handler in `runtime/src/profile.rs`, JIT map dump in
//! `compiler/src/codegen/cranelift_backend.rs`, tensor-data counters in
//! `runtime/src/tensor.rs`, `HeapCheckGuard` in `runtime/src/heap_check.rs`).
//! No new runtime knobs were added — this module just flips them on with
//! ergonomic flags.

use clap::Subcommand;
use std::path::PathBuf;

pub mod bench;
pub mod compare;
pub mod lldb_cmd;
pub mod resolve;
pub mod run;
pub mod server;

#[derive(Subcommand, Debug)]
pub enum DebugCommands {
    /// Run a Haxe file with forensic infrastructure pre-wired.
    ///
    /// Equivalent to `rayzor run` but with the env vars from
    /// `runtime/src/profile.rs` + `compiler/src/codegen/cranelift_backend.rs`
    /// set up for you. Crashes produce resolved Haxe function names + lines
    /// instead of raw exit codes.
    Run {
        /// Path to the Haxe source file or compiled .rzb bundle
        file: PathBuf,

        /// Disable the JIT symbol map dump (default ON; cheap)
        #[arg(long)]
        no_jit_map: bool,

        /// Disable the sigaction-based crash backtrace (default ON; cheap)
        #[arg(long)]
        no_backtrace: bool,

        /// Dump TrackingAllocator + tensor-data stats at exit. Requires the
        /// binary built with `--features profile`; this flag warns if not.
        #[arg(long)]
        alloc_stats: bool,

        /// Sample alloc backtraces at 1 in N rate. Implies `--alloc-stats`.
        /// Resolve via `rayzor debug resolve`.
        #[arg(long, value_name = "RATE")]
        alloc_graph: Option<u64>,

        /// Run SIGPROF profiler at the given microsecond interval. Useful
        /// for "where is the time going" surveys.
        #[arg(long, value_name = "MICROS")]
        cpu_profile: Option<u64>,

        /// Install `HeapCheckGuard` Drop guards on every suspect kernel
        /// (macOS only). Validates the malloc freelist at entry+exit of each
        /// kernel. Cost: microseconds per call; finds heap corruption EARLY.
        #[arg(long)]
        heap_check: bool,

        /// Set `MallocStackLogging=1` in the process env. Lets `malloc_history`
        /// query allocation traces post-mortem. macOS only; slows the process
        /// substantially.
        #[arg(long)]
        malloc_stack_log: bool,

        /// Tier preset to use (passed through to the run path).
        #[arg(long, default_value = "application")]
        preset: String,

        /// Args to forward to the Haxe program (after `--`).
        #[arg(last = true)]
        program_args: Vec<String>,
    },

    /// Run a file N times, report per-run metrics + aggregate stats.
    ///
    /// Captures the `[tensor-data] peak=` line from each successful run
    /// (requires `--features profile`). SIGTRAP / SIGSEGV failures are
    /// counted separately and excluded from the medians.
    Bench {
        /// Path to the Haxe source file
        file: PathBuf,

        /// Number of runs (default 5)
        #[arg(short = 'n', long, default_value_t = 5)]
        runs: usize,

        /// What to aggregate
        #[arg(long, value_enum, default_value_t = bench::Metric::PeakMem)]
        metric: bench::Metric,

        /// Timeout per run in seconds
        #[arg(long, default_value_t = 90)]
        timeout: u64,

        /// Scrub the .rayzor cache between each run (default ON)
        #[arg(long)]
        no_cache_scrub: bool,

        /// Pass --release to each child `rayzor run`
        #[arg(long)]
        release: bool,

        /// Pass --llvm to each child `rayzor run`
        #[arg(long)]
        llvm: bool,

        /// Load a raw native plugin dylib/so/dll in each child `rayzor run`
        #[arg(long = "native-lib", value_name = "FILE")]
        native_libs: Vec<PathBuf>,

        /// Enable debug-only JIT/alloc dump env vars in each child run
        #[arg(long)]
        debug_dumps: bool,

        /// Tier preset passed to each child `rayzor run`
        #[arg(long, default_value = "application")]
        preset: String,

        /// Repeatable tier threshold variant: interpreter/warm/hot[/blazing], e.g. 1/15/5
        #[arg(long, value_name = "I/W/H[/B]")]
        tier_thresholds: Vec<String>,

        /// Make each tier variant start from CLI --preset instead of rayzor.toml [tier]
        #[arg(long)]
        preset_override_toml: bool,

        /// Constant tier sample-rate override passed to each child run
        #[arg(long, value_name = "N")]
        tier_sample_rate: Option<u64>,

        /// Constant start_interpreted override passed to each child run
        #[arg(long, value_name = "BOOL")]
        tier_start_interpreted: Option<bool>,

        /// Constant enable_tier_promotion override passed to each child run
        #[arg(long, value_name = "BOOL")]
        tier_promotion: Option<bool>,

        /// Enable RAYZOR_PROFILE_DECODE=1 and surface per-token tail latency
        #[arg(long)]
        decode_profile: bool,

        /// Cooldown between subprocess runs, in milliseconds
        #[arg(long, default_value_t = 0)]
        cooldown_ms: u64,

        /// Args to forward to the Haxe program (after `--`)
        #[arg(last = true)]
        program_args: Vec<String>,
    },

    /// A/B compare two git refs by running a Haxe file at each and reporting
    /// the median delta. Auto-restores the working tree on exit.
    Compare {
        /// Baseline git ref (e.g. `HEAD~1`, `main`, `be797eb`)
        base_ref: String,

        /// Path to the Haxe source file
        file: PathBuf,

        /// Number of runs at each side (default 5)
        #[arg(short = 'n', long, default_value_t = 5)]
        runs: usize,

        /// What to aggregate
        #[arg(long, value_enum, default_value_t = bench::Metric::PeakMem)]
        metric: bench::Metric,

        /// Timeout per run in seconds
        #[arg(long, default_value_t = 90)]
        timeout: u64,

        /// Args to forward to the Haxe program (after `--`)
        #[arg(last = true)]
        program_args: Vec<String>,
    },

    /// Resolve hex PCs from a crash dump to Haxe function names + lines.
    ///
    /// Reads PCs from positional args (`rayzor debug resolve 0x10... 0x10...`)
    /// or from stdin. Joins against `/tmp/rayzor_jit_symbols.{pid}.csv` or
    /// the path passed via `--csv`.
    Resolve {
        /// Hex PCs to resolve (positional). If none given, reads from stdin.
        pcs: Vec<String>,

        /// Path to the JIT symbol CSV. Defaults to the most recent
        /// `/tmp/rayzor_jit_symbols.*.csv`.
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Path to the file table CSV. Defaults to `/tmp/rayzor_file_table.csv`.
        #[arg(long)]
        file_table: Option<PathBuf>,
    },

    /// Launch a Haxe file under lldb with crash handlers pre-armed.
    ///
    /// Auto-installs:
    /// - `process handle SIGTRAP -p true -s true -n true`
    /// - same for SIGSEGV/SIGABRT
    /// - `bt 20` + `register read x0..x28 fp sp pc lr` on stop
    /// - JIT map dump enabled
    ///
    /// macOS only.
    Lldb {
        /// Path to the Haxe source file
        file: PathBuf,

        /// Set `MallocStackLogging=1` so `malloc_history` queries work
        /// against the paused process.
        #[arg(long)]
        malloc_stack_log: bool,

        /// Args to forward to the Haxe program (after `--`)
        #[arg(last = true)]
        program_args: Vec<String>,
    },

    /// Start an HTTP server that exposes live metrics + a browser dashboard.
    ///
    /// Endpoints:
    /// - `GET /`             — embedded HTML dashboard
    /// - `GET /api/metrics`  — current allocator + tensor-data counters
    /// - `GET /api/jit-map`  — current JIT symbol map
    /// - `GET /api/file-table` — file-id → path table
    /// - `GET /api/crashes`  — recent crash backtraces from `/tmp`
    Server {
        /// Port to listen on
        #[arg(short, long, default_value_t = 7878)]
        port: u16,

        /// Bind address
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Auto-open the dashboard in the default browser
        #[arg(long)]
        open: bool,
    },
}

impl DebugCommands {
    pub fn execute(self) -> anyhow::Result<()> {
        match self {
            DebugCommands::Run { .. } => run::execute(self),
            DebugCommands::Bench { .. } => bench::execute(self),
            DebugCommands::Compare { .. } => compare::execute(self),
            DebugCommands::Resolve { .. } => resolve::execute(self),
            DebugCommands::Lldb { .. } => lldb_cmd::execute(self),
            DebugCommands::Server { .. } => server::execute(self),
        }
    }
}
