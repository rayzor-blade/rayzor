//! # Tiered Compilation Backend
//!
//! Implements multi-tier JIT compilation using Cranelift with different optimization levels.
//! Automatically recompiles hot functions with higher optimization based on runtime profiling.
//!
//! ## Optimization Tiers
//! - **Tier 0 (Baseline)**: Minimal optimization, fastest compilation (for cold code)
//! - **Tier 1 (Standard)**: Moderate optimization (for warm code)
//! - **Tier 2 (Optimized)**: Aggressive optimization (for hot code)
//!
//! ## How It Works
//! 1. All functions start at Tier 0 (baseline JIT)
//! 2. Execution counters track how often functions are called
//! 3. When a function crosses the "warm" threshold, it's recompiled at Tier 1
//! 4. When it crosses the "hot" threshold, it's recompiled at Tier 2
//! 5. Function pointers are atomically swapped after recompilation
//!
//! ## Architecture
//! - Main thread: Executes code, records profile data
//! - Background worker: Monitors hot functions, performs async recompilation
//! - Lock-free atomic counters: Minimal overhead profiling
//! - RwLock for function pointer map: Fast reads, infrequent writes

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

// ============================================================================
// Promotion Barrier - Safe Tier Promotion Mechanism
// ============================================================================
//
// Inspired by HotSpot JVM's safepoint mechanism, this barrier ensures safe
// code replacement during tier promotion. The key insight is that we cannot
// simply swap function pointers while code is executing - we need to ensure:
//
// 1. No thread is currently executing JIT code that might call replaced functions
// 2. All function pointers are replaced atomically (all-or-nothing)
// 3. Minimal performance impact during normal execution
//
// Protocol:
// - Main thread: Before executing JIT code, check barrier state and increment counter
// - Main thread: After executing, decrement counter
// - Background worker: Request promotion, wait for counter to reach 0
// - Background worker: Swap ALL function pointers atomically
// - Background worker: Release barrier, allowing execution to resume
//

/// State of the promotion barrier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PromotionState {
    /// Normal execution - no promotion pending
    Idle = 0,
    /// Promotion requested - new executions should wait
    PromotionRequested = 1,
    /// Promotion in progress - function pointers being swapped
    PromotionInProgress = 2,
}

impl From<u8> for PromotionState {
    fn from(v: u8) -> Self {
        match v {
            0 => PromotionState::Idle,
            1 => PromotionState::PromotionRequested,
            2 => PromotionState::PromotionInProgress,
            _ => PromotionState::Idle,
        }
    }
}

/// Barrier for safe tier promotion
///
/// This barrier coordinates between the main execution thread and background
/// compilation workers to ensure function pointers are only replaced when
/// no code is executing.
pub struct PromotionBarrier {
    /// Current barrier state
    state: AtomicU8,

    /// Number of JIT executions currently in flight
    /// This counts nested calls - a call from A to B to C increments 3 times
    execution_counter: AtomicU64,

    /// Mutex + Condvar for blocking when promotion is requested
    /// Main thread waits on this when state is PromotionRequested
    wait_mutex: Mutex<()>,
    wait_condvar: Condvar,

    /// Mutex + Condvar for background worker to wait for executions to drain
    drain_mutex: Mutex<()>,
    drain_condvar: Condvar,
}

impl PromotionBarrier {
    /// Create a new promotion barrier
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(PromotionState::Idle as u8),
            execution_counter: AtomicU64::new(0),
            wait_mutex: Mutex::new(()),
            wait_condvar: Condvar::new(),
            drain_mutex: Mutex::new(()),
            drain_condvar: Condvar::new(),
        }
    }

    /// Get current state
    #[inline]
    pub fn state(&self) -> PromotionState {
        PromotionState::from(self.state.load(Ordering::Acquire))
    }

    /// Get current execution count
    #[inline]
    pub fn execution_count(&self) -> u64 {
        self.execution_counter.load(Ordering::Acquire)
    }

    /// Enter JIT execution - called before executing JIT code
    ///
    /// Returns true if execution can proceed, false if caller should wait and retry
    #[inline]
    pub fn enter_execution(&self) -> bool {
        // Fast path: if state is Idle, just increment counter
        let state = self.state();
        if state == PromotionState::Idle {
            self.execution_counter.fetch_add(1, Ordering::AcqRel);
            // Double-check state didn't change (promotion might have started)
            if self.state() == PromotionState::Idle {
                return true;
            }
            // State changed, decrement and return false to wait
            self.execution_counter.fetch_sub(1, Ordering::AcqRel);
            // Notify drain condvar in case background worker is waiting
            self.drain_condvar.notify_all();
        }
        false
    }

    /// Wait for barrier to become idle, then enter execution
    pub fn wait_and_enter_execution(&self) {
        loop {
            // Try fast path first
            if self.enter_execution() {
                return;
            }

            // Slow path: wait for promotion to complete
            let guard = self.wait_mutex.lock().unwrap();
            // Re-check state after acquiring lock
            if self.state() == PromotionState::Idle {
                drop(guard);
                continue; // Retry fast path
            }
            // Wait for notification
            let _guard = self.wait_condvar.wait(guard).unwrap();
            // Loop and retry
        }
    }

    /// Exit JIT execution - called after executing JIT code
    #[inline]
    pub fn exit_execution(&self) {
        let prev = self.execution_counter.fetch_sub(1, Ordering::AcqRel);
        // If counter reached 0 and promotion is requested, notify the background worker
        if prev == 1 && self.state() != PromotionState::Idle {
            self.drain_condvar.notify_all();
        }
    }

    /// Request promotion - called by background worker before replacing function pointers
    ///
    /// This sets the state to PromotionRequested, preventing new executions.
    /// Returns immediately - caller should then wait_for_drain().
    pub fn request_promotion(&self) -> bool {
        // Try to transition from Idle to PromotionRequested
        self.state
            .compare_exchange(
                PromotionState::Idle as u8,
                PromotionState::PromotionRequested as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Wait for all executions to drain
    ///
    /// Called by background worker after request_promotion().
    /// Blocks until execution_counter reaches 0.
    pub fn wait_for_drain(&self, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;

        loop {
            // Check if already drained
            if self.execution_counter.load(Ordering::Acquire) == 0 {
                // Transition to PromotionInProgress
                self.state
                    .store(PromotionState::PromotionInProgress as u8, Ordering::Release);
                return true;
            }

            // Check timeout
            let now = std::time::Instant::now();
            if now >= deadline {
                return false;
            }

            // Wait for notification with remaining timeout
            let remaining = deadline - now;
            let guard = self.drain_mutex.lock().unwrap();
            let result = self.drain_condvar.wait_timeout(guard, remaining).unwrap();

            if result.1.timed_out() {
                return false;
            }
        }
    }

    /// Complete promotion - called after function pointers have been swapped
    ///
    /// This sets state back to Idle and wakes up any waiting execution threads.
    pub fn complete_promotion(&self) {
        self.state
            .store(PromotionState::Idle as u8, Ordering::Release);
        // Wake up all waiting execution threads
        self.wait_condvar.notify_all();
    }

    /// Cancel promotion request - called if promotion fails or is aborted
    pub fn cancel_promotion(&self) {
        // Only cancel if we're in PromotionRequested state
        let _ = self.state.compare_exchange(
            PromotionState::PromotionRequested as u8,
            PromotionState::Idle as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        // Also try to cancel from PromotionInProgress (in case of error during swap)
        let _ = self.state.compare_exchange(
            PromotionState::PromotionInProgress as u8,
            PromotionState::Idle as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        // Wake up waiting threads
        self.wait_condvar.notify_all();
    }
}

impl Default for PromotionBarrier {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard for JIT execution
///
/// Automatically decrements execution counter when dropped.
pub struct ExecutionGuard<'a> {
    barrier: &'a PromotionBarrier,
}

impl<'a> ExecutionGuard<'a> {
    /// Create a new execution guard
    ///
    /// Caller must have already called barrier.enter_execution() or wait_and_enter_execution()
    pub fn new(barrier: &'a PromotionBarrier) -> Self {
        Self { barrier }
    }
}

impl<'a> Drop for ExecutionGuard<'a> {
    fn drop(&mut self) {
        self.barrier.exit_execution();
    }
}

use super::cranelift_backend::CraneliftBackend;
use super::mir_interpreter::{InterpError, InterpValue, MirInterpreter};
use super::profiling::{ProfileConfig, ProfileData, ProfileStatistics};
use crate::ir::{IrFunction, IrFunctionId, IrInstruction, IrModule};

#[cfg(feature = "llvm-backend")]
use super::llvm_jit_backend::LLVMJitBackend;
#[cfg(feature = "llvm-backend")]
use inkwell::context::Context;
use tracing::debug;

/// Tiered compilation backend
pub struct TieredBackend {
    /// MIR interpreter for Phase 0 (instant startup)
    interpreter: Arc<Mutex<MirInterpreter>>,

    /// Primary Cranelift backend (used for Phase 1+ compilation)
    baseline_backend: Arc<Mutex<CraneliftBackend>>,

    /// Runtime profiling data
    profile_data: ProfileData,

    /// Current optimization tier for each function
    function_tiers: Arc<RwLock<BTreeMap<IrFunctionId, OptimizationTier>>>,

    /// Function pointers (usize for thread safety, cast to function type when needed)
    function_pointers: Arc<RwLock<BTreeMap<IrFunctionId, usize>>>,

    /// The MIR modules (needed for recompilation and interpretation)
    /// Multiple modules may be loaded (e.g., user code + stdlib)
    modules: Arc<RwLock<Vec<IrModule>>>,

    /// Configuration
    config: TieredConfig,

    /// Whether to start in interpreted mode (Phase 0)
    start_interpreted: bool,

    /// Runtime symbols for FFI (used by interpreter and LLVM backend)
    /// Stored as (name, pointer) pairs for thread-safe sharing
    runtime_symbols: Arc<Vec<(String, usize)>>,

    /// Queue of functions waiting for LLVM compilation on main thread
    /// LLVM's add_global_mapping requires main thread, so background workers
    /// queue requests here instead of downgrading to Cranelift
    llvm_queue: Arc<Mutex<VecDeque<IrFunctionId>>>,

    /// Whether LLVM compilation has already been performed
    /// Used to prevent unbounded LLVM context leaks from repeated compilations
    #[cfg(feature = "llvm-backend")]
    llvm_compiled: Arc<Mutex<bool>>,

    /// Promotion barrier for safe tier promotion
    /// Ensures no JIT code is executing when function pointers are replaced
    promotion_barrier: Arc<PromotionBarrier>,

    /// Number of loaded MIR modules that have already run startup hooks.
    initialized_module_count: Arc<Mutex<usize>>,

    /// Phase B (beadie integration): optional
    /// `BackendAdapter<BeadieJit>`, constructed only when
    /// [`TieredConfig::enable_beadie_adapter`] is true.
    ///
    /// `BeadieJit` builds a fresh `CraneliftBackend` per compile (the
    /// same `Box::leak` pattern legacy `compile_all_at_tier` uses).
    /// See [`super::beadie_jit`] for the rationale — short version:
    /// fresh-per-compile gets cross-module call resolution for free
    /// at the cost of the same per-promotion leak as the legacy path.
    /// What beadie brings is the *broker* (threshold + batched
    /// scheduling + broker-thread compile loop), not a different
    /// compile mechanism.
    ///
    /// When set, `record_call` routes Standard-tier promotion through
    /// this adapter. `beadie_adapter_optimized` does the same for
    /// Optimized-tier promotion.
    beadie_adapter: Arc<beadie::BackendAdapter<super::beadie_jit::BeadieJit>>,

    /// Phase B: lazy registry of one `BoundBead` per `IrFunctionId`
    /// that beadie's Standard-tier adapter is tracking. Populated on
    /// demand by [`TieredBackend::ensure_beadie_bead`]. Empty when
    /// `beadie_adapter` is `None`.
    beadie_beads:
        Arc<Mutex<BTreeMap<IrFunctionId, beadie::BoundBead<super::beadie_jit::BeadieJit>>>>,

    /// Phase B step 5: parallel adapter for Optimized-tier promotion.
    /// Compiles at cranelift opt-level `"speed_and_size"` (vs Standard's
    /// `"speed"`). Each tier needs its own adapter because `BeadieJit`
    /// carries a fixed opt-level; each tier needs its own bead
    /// registry because a bead's compile state is one-shot — once a
    /// function's Standard bead reaches `Compiled`, it doesn't
    /// recompile when the same function later crosses the Optimized
    /// threshold.
    beadie_adapter_optimized: Arc<beadie::BackendAdapter<super::beadie_jit::BeadieJit>>,

    /// Phase B step 5: bead registry for the Optimized adapter.
    /// Parallel to `beadie_beads` but tracks Optimized-tier promotion
    /// state.
    beadie_beads_optimized:
        Arc<Mutex<BTreeMap<IrFunctionId, beadie::BoundBead<super::beadie_jit::BeadieJit>>>>,

    /// Phase B: counter — number of times `record_call` attempted to
    /// route a promotion through beadie, summed across both Standard
    /// and Optimized tiers. Includes failures.
    beadie_routes_attempted: Arc<AtomicU64>,

    /// Phase B: route attempts split by Beadie target tier.
    beadie_standard_routes_attempted: Arc<AtomicU64>,
    beadie_optimized_routes_attempted: Arc<AtomicU64>,

    /// Phase B: counter — number of times `install_beadie_pointer`
    /// actually wrote a new pointer into `function_pointers`, summed
    /// across both Standard and Optimized tiers. Skips idempotent
    /// re-installs of an already-known pointer.
    beadie_installs: Arc<AtomicU64>,

    /// Phase B: pointer installs split by Beadie target tier.
    beadie_standard_installs: Arc<AtomicU64>,
    beadie_optimized_installs: Arc<AtomicU64>,
}

/// Optimization tier level (5-tier system with interpreter)
/// Note: All JIT tiers now use Cranelift. LLVM is available as a standalone backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimizationTier {
    Interpreted, // Phase 0: MIR interpreter (instant startup, ~5-10x native speed)
    Baseline,    // Phase 1: Cranelift, fast compilation, minimal optimization
    Standard,    // Phase 2: Cranelift, moderate optimization
    Optimized,   // Phase 3: Cranelift, aggressive optimization
    Maximum,     // Phase 4: Cranelift with maximum MIR optimizations
}

impl OptimizationTier {
    /// Get Cranelift optimization level for this tier (Phase 1-3 only)
    pub fn cranelift_opt_level(&self) -> &'static str {
        match self {
            OptimizationTier::Interpreted => "none", // P0: Not used (interpreter)
            OptimizationTier::Baseline => "none",    // P1: No optimization
            OptimizationTier::Standard => "speed",   // P2: Moderate
            // TODO: "speed_and_size" causes incorrect results in some cases (checksum halved)
            // Using "speed" until the root cause is identified
            OptimizationTier::Optimized => "speed", // P3: Was "speed_and_size"
            OptimizationTier::Maximum => "speed",   // P4: Maximum Cranelift optimization
        }
    }

    /// Get MIR optimization level for this tier
    pub fn mir_opt_level(&self) -> crate::ir::optimization::OptimizationLevel {
        use crate::ir::optimization::OptimizationLevel;
        match self {
            OptimizationTier::Interpreted => OptimizationLevel::O0, // No MIR opts for interpreter
            OptimizationTier::Baseline => OptimizationLevel::O0,    // Fast compilation
            OptimizationTier::Standard => OptimizationLevel::O1,    // Basic optimizations
            OptimizationTier::Optimized => OptimizationLevel::O2,   // Standard optimizations
            OptimizationTier::Maximum => OptimizationLevel::O3,     // Aggressive optimizations
        }
    }

    /// Check if this tier uses the interpreter
    pub fn uses_interpreter(&self) -> bool {
        matches!(self, OptimizationTier::Interpreted)
    }

    /// Check if this tier uses LLVM backend
    /// Note: LLVM is now a standalone backend, not part of tiered compilation
    pub fn uses_llvm(&self) -> bool {
        // Maximum tier uses LLVM for best optimization; other tiers use Cranelift
        matches!(self, OptimizationTier::Maximum)
    }

    /// Get the next higher tier (if any)
    pub fn next_tier(&self) -> Option<OptimizationTier> {
        match self {
            OptimizationTier::Interpreted => Some(OptimizationTier::Baseline),
            OptimizationTier::Baseline => Some(OptimizationTier::Standard),
            OptimizationTier::Standard => Some(OptimizationTier::Optimized),
            OptimizationTier::Optimized => Some(OptimizationTier::Maximum),
            OptimizationTier::Maximum => None, // Already at max
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            OptimizationTier::Interpreted => "Interpreted (P0/MIR)",
            OptimizationTier::Baseline => "Baseline (P1/Cranelift)",
            OptimizationTier::Standard => "Standard (P2/Cranelift)",
            OptimizationTier::Optimized => "Optimized (P3/Cranelift)",
            OptimizationTier::Maximum => "Maximum (P4/Cranelift+O3)",
        }
    }

    fn event_label(&self) -> &'static str {
        match self {
            OptimizationTier::Interpreted => "interpreted",
            OptimizationTier::Baseline => "baseline",
            OptimizationTier::Standard => "standard",
            OptimizationTier::Optimized => "optimized",
            OptimizationTier::Maximum => "maximum",
        }
    }
}

/// Interpreter bailout strategy - determines how quickly to switch from interpreter to JIT
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
pub enum BailoutStrategy {
    /// Immediate bailout after ~10 block executions
    /// Best for: Hot compute-intensive code, benchmarks
    Immediate,
    /// Quick bailout after ~100 block executions
    /// Best for: Most applications, good balance of startup and steady-state
    Quick,
    /// Normal bailout after ~1000 block executions
    /// Best for: Short-running scripts, startup-time sensitive apps
    Normal,
    /// Slow bailout after ~10000 block executions
    /// Best for: Very short scripts, one-shot programs
    Slow,
    /// Custom threshold value
    Custom(u64),
}

impl BailoutStrategy {
    /// Get the iteration threshold for this strategy
    pub fn threshold(&self) -> u64 {
        match self {
            BailoutStrategy::Immediate => 10,
            BailoutStrategy::Quick => 100,
            BailoutStrategy::Normal => 1000,
            BailoutStrategy::Slow => 10000,
            BailoutStrategy::Custom(n) => *n,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            BailoutStrategy::Immediate => "Immediate (~10 iterations)",
            BailoutStrategy::Quick => "Quick (~100 iterations)",
            BailoutStrategy::Normal => "Normal (~1000 iterations)",
            BailoutStrategy::Slow => "Slow (~10000 iterations)",
            BailoutStrategy::Custom(_) => "Custom",
        }
    }
}

impl Default for BailoutStrategy {
    fn default() -> Self {
        BailoutStrategy::Quick
    }
}

/// Predefined tier presets for common use cases
///
/// Use these presets to quickly configure the tiered backend for your application type.
/// Each preset is optimized for different performance characteristics:
///
/// | Preset      | Startup  | Peak Perf | Memory  | Best For                           |
/// |-------------|----------|-----------|---------|-----------------------------------|
/// | Script      | Instant  | Moderate  | Low     | CLI tools, one-shot scripts       |
/// | Application | Fast     | High      | Medium  | Desktop apps, web servers         |
/// | Server      | Slower   | Maximum   | Higher  | Long-running services, APIs       |
/// | Benchmark   | Slowest  | Maximum   | Highest | Performance testing, profiling    |
/// | Development | Instant  | Low       | Low     | Dev iteration, debugging          |
/// | Embedded    | Instant  | Moderate  | Minimal | Resource-constrained environments |
///
/// Note: `Copy` / `PartialEq` / `Eq` were dropped when the `Custom`
/// variant was added — `TieredConfig` carries non-`Copy` /
/// non-`PartialEq` fields, so the enum can't keep those auto-derives.
/// Existing callers all consume the preset by value via
/// [`TierPreset::to_config`], so the loss of `Copy` is benign.
#[derive(Debug, Clone)]
pub enum TierPreset {
    /// For short-running scripts and CLI tools
    /// - Instant startup via interpreter
    /// - Quick bailout to Cranelift JIT for hot loops
    /// - No LLVM tier (overhead not worth it for short runs)
    /// - Minimal memory usage
    Script,

    /// For typical applications (desktop apps, web servers)
    /// - Fast startup via interpreter
    /// - Balanced tier promotion thresholds
    /// - LLVM tier for sustained hot code
    /// - Background optimization enabled
    Application,

    /// For long-running servers and services
    /// - Startup time less critical
    /// - Aggressive optimization for hot paths
    /// - LLVM tier with lower threshold
    /// - Maximum parallel optimization
    Server,

    /// For benchmarks and performance testing
    /// - Immediate bailout from interpreter
    /// - Explicit LLVM upgrade after warmup
    /// - Maximum optimization at all tiers
    /// - Verbose tier transition logging
    Benchmark,

    /// For development and debugging
    /// - Instant startup
    /// - Verbose logging of tier transitions
    /// - Fast iteration over optimization
    /// - Useful for debugging tiered behavior
    Development,

    /// For resource-constrained environments
    /// - Interpreter-only, no JIT compilation
    /// - Minimal memory footprint
    /// - Predictable performance (no JIT spikes)
    /// - Suitable for embedded or WASM targets
    Embedded,

    /// User-supplied [`TieredConfig`], typically deserialized from a
    /// `[tier]` section in `rayzor.toml`. Bypasses all preset defaults
    /// and uses the inner config verbatim.
    Custom(TieredConfig),
}

impl TierPreset {
    /// Convert preset to a TieredConfig. Env overrides (e.g.
    /// `RAYZOR_USE_BEADIE=1`) are applied after the preset's defaults
    /// so benchmark harnesses can toggle Phase B features without
    /// recompiling.
    pub fn to_config(self) -> TieredConfig {
        self.to_config_without_env().apply_env_overrides()
    }

    /// Same as [`Self::to_config`] but without env-var overrides.
    /// Useful for tests that need deterministic config regardless of
    /// the test runner's environment.
    pub fn to_config_without_env(self) -> TieredConfig {
        match self {
            TierPreset::Script => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 5, // Quick promotion from interpreter
                    warm_threshold: 50,       // Don't optimize too aggressively
                    hot_threshold: 200,
                    blazing_threshold: u64::MAX, // No LLVM for short scripts
                    sample_rate: 1,
                },
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Quick,
                enable_tier_promotion: true,
                enable_stack_traces: true,
                auto_upgrade_to_llvm_after_main_entry: false,
            },

            TierPreset::Application => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 10,
                    warm_threshold: 100,
                    hot_threshold: 500,
                    blazing_threshold: 2000, // LLVM for sustained hot code
                    sample_rate: 1,
                },
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Quick,
                enable_tier_promotion: true,
                enable_stack_traces: true,
                // A monolithic `main` never re-enters the dispatcher, so
                // count-based promotion can't reach its inner functions;
                // install the LLVM tier before main runs.
                auto_upgrade_to_llvm_after_main_entry: true,
            },

            TierPreset::Server => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 5,
                    warm_threshold: 50,
                    hot_threshold: 200,
                    blazing_threshold: 500, // Lower LLVM threshold for servers
                    sample_rate: 1,
                },
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Immediate,
                enable_tier_promotion: true,
                enable_stack_traces: true,
                auto_upgrade_to_llvm_after_main_entry: true,
            },

            TierPreset::Benchmark => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 2,
                    warm_threshold: 3,
                    hot_threshold: 5,
                    blazing_threshold: u64::MAX, // Manual LLVM upgrade after warmup
                    sample_rate: 1,
                },
                verbosity: 1,            // Show tier transitions
                start_interpreted: true, // Start with interpreter for instant startup
                bailout_strategy: BailoutStrategy::Immediate,
                enable_tier_promotion: true,
                enable_stack_traces: false, // No instrumentation overhead in benchmarks
                auto_upgrade_to_llvm_after_main_entry: false,
            },

            TierPreset::Development => TieredConfig {
                profile_config: ProfileConfig::development(),
                verbosity: 2, // Detailed logging
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Immediate,
                enable_tier_promotion: true,
                enable_stack_traces: true,
                auto_upgrade_to_llvm_after_main_entry: false,
            },

            TierPreset::Embedded => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: u64::MAX, // Never promote
                    warm_threshold: u64::MAX,
                    hot_threshold: u64::MAX,
                    blazing_threshold: u64::MAX,
                    sample_rate: u64::MAX, // Disable profiling
                },
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Slow, // High threshold before bailout
                enable_tier_promotion: false,
                enable_stack_traces: false, // No stack traces for embedded
                auto_upgrade_to_llvm_after_main_entry: false,
            },

            TierPreset::Custom(cfg) => cfg.clone(),
        }
    }

    /// Get a human-readable description of the preset
    pub fn description(&self) -> &'static str {
        match self {
            TierPreset::Script => "Script - Fast startup, quick JIT, no LLVM",
            TierPreset::Application => "Application - Balanced tiering with LLVM",
            TierPreset::Server => "Server - Aggressive optimization, low LLVM threshold",
            TierPreset::Benchmark => "Benchmark - Immediate bailout, manual LLVM upgrade",
            TierPreset::Development => "Development - Verbose logging, fast iteration",
            TierPreset::Embedded => "Embedded - Interpreter only, minimal resources",
            TierPreset::Custom(_) => "Custom - user-defined config (rayzor.toml [tier])",
        }
    }

    /// Get recommended use cases for this preset
    pub fn use_cases(&self) -> &'static [&'static str] {
        match self {
            TierPreset::Script => &[
                "CLI tools",
                "Build scripts",
                "One-shot programs",
                "Shell utilities",
            ],
            TierPreset::Application => &[
                "Desktop apps",
                "Web servers",
                "GUI applications",
                "General purpose",
            ],
            TierPreset::Server => &[
                "API servers",
                "Microservices",
                "Background workers",
                "Long-running daemons",
            ],
            TierPreset::Benchmark => &["Performance testing", "Profiling", "Optimization analysis"],
            TierPreset::Development => &[
                "Debugging",
                "Development iteration",
                "Testing tier behavior",
            ],
            TierPreset::Embedded => &[
                "WebAssembly",
                "Embedded systems",
                "Memory-constrained",
                "Predictable latency",
            ],
            TierPreset::Custom(_) => &[],
        }
    }
}

impl std::fmt::Display for TierPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Configuration for tiered compilation
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct TieredConfig {
    /// Profiling configuration
    #[serde(flatten)]
    pub profile_config: ProfileConfig,

    /// Verbosity level (0 = silent, 1 = basic, 2 = detailed)
    pub verbosity: u8,

    /// Start in interpreted mode (Phase 0) for instant startup
    /// If false, functions are compiled to Baseline (Phase 1) immediately
    pub start_interpreted: bool,

    /// Interpreter bailout strategy — determines how quickly to switch
    /// from interpreter to JIT (Baseline tier). Use
    /// `BailoutStrategy::Immediate` for benchmarks, `Quick` for most apps.
    pub bailout_strategy: BailoutStrategy,

    /// Enable tier promotion (Standard / Optimized / Maximum). When
    /// `false`, functions stay at their initial tier (Interpreter or
    /// Baseline) forever — useful for embedded/script modes that don't
    /// want JIT spin-up cost.
    ///
    /// Step 6 renamed this from `max_tier_promotions: u64` (which was
    /// a Cranelift-leak budget) since beadie owns the per-promotion
    /// compile-and-leak now; the budget no longer applies.
    pub enable_tier_promotion: bool,

    /// Enable source-mapped stack traces for exceptions.
    /// When true, registers JIT function addresses with source metadata after compilation
    /// and captures Haxe-level stack traces on throw. Disabled in --release mode for
    /// zero overhead in production.
    pub enable_stack_traces: bool,

    /// After main-entry completes its tier-up, force-promote everything
    /// the tier scheduler reached at least once to the LLVM (Maximum)
    /// tier. Default `false`. Lets benchmark and "AOT-after-warmup"
    /// scenarios skip the per-call hot-threshold counter and jump
    /// straight to the high-tier backend once main has returned.
    pub auto_upgrade_to_llvm_after_main_entry: bool,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            profile_config: ProfileConfig::default(),
            verbosity: 0,
            start_interpreted: true, // Enable interpreter by default for instant startup
            bailout_strategy: BailoutStrategy::Quick, // Good balance for most apps
            enable_tier_promotion: true,
            enable_stack_traces: true, // Debug by default
            auto_upgrade_to_llvm_after_main_entry: false,
        }
    }
}

impl TieredConfig {
    /// Create a TieredConfig from a preset
    ///
    /// # Example
    /// ```
    /// use compiler::codegen::tiered_backend::{TieredConfig, TierPreset};
    ///
    /// // For a CLI tool
    /// let config = TieredConfig::from_preset(TierPreset::Script);
    ///
    /// // For a web server
    /// let config = TieredConfig::from_preset(TierPreset::Server);
    /// ```
    pub fn from_preset(preset: TierPreset) -> Self {
        preset.to_config()
    }

    /// Apply env-var overrides to the config.
    ///
    /// Step 6 removed the `RAYZOR_USE_BEADIE` override: beadie is now
    /// unconditional (it owns Standard + Optimized tier promotion as
    /// the single tier scheduler), so there's nothing to toggle. The
    /// method stays as a no-op extension point — kept on the public
    /// API so callers (`TierPreset::to_config`, benchmark harnesses)
    /// don't need to be touched if we later add new env-controlled
    /// options.
    pub fn apply_env_overrides(self) -> Self {
        self
    }

    /// Development configuration (aggressive optimization, verbose)
    pub fn development() -> Self {
        Self {
            profile_config: ProfileConfig::development(),
            verbosity: 2,
            start_interpreted: true, // Instant startup for quick iteration
            bailout_strategy: BailoutStrategy::Immediate, // Quick bailout for testing
            enable_tier_promotion: true,
            enable_stack_traces: true,
            auto_upgrade_to_llvm_after_main_entry: false,
        }
    }

    /// Production configuration (conservative, low overhead)
    pub fn production() -> Self {
        Self {
            profile_config: ProfileConfig::production(),
            verbosity: 0,
            start_interpreted: true, // Instant startup, then promote hot functions
            bailout_strategy: BailoutStrategy::Quick, // Quick bailout
            enable_tier_promotion: true,
            enable_stack_traces: false, // Production: no trace overhead
            auto_upgrade_to_llvm_after_main_entry: false,
        }
    }

    /// JIT-only configuration (skip interpreter, compile immediately)
    /// Use when startup time is less important than consistent performance
    pub fn jit_only() -> Self {
        Self {
            profile_config: ProfileConfig::default(),
            verbosity: 0,
            start_interpreted: false, // Skip interpreter, start at Phase 1
            bailout_strategy: BailoutStrategy::Quick, // Not used when start_interpreted=false
            enable_tier_promotion: true,
            enable_stack_traces: true,
            auto_upgrade_to_llvm_after_main_entry: false,
        }
    }
}

/// Snapshot of beadie-integration counters for observability /
/// benchmarking. See [`TieredBackend::beadie_stats`].
#[derive(Debug, Clone, Copy)]
pub struct BeadieStats {
    /// Whether the beadie adapter was constructed at all (true iff
    /// `TieredConfig::enable_beadie_adapter` was set).
    pub adapter_enabled: bool,
    /// Number of times `record_call` tried to route a Standard-tier
    /// promotion through beadie. Includes retries (the install may
    /// require multiple `record_call`s to land — the install observer
    /// runs inline with `record_call` rather than on a separate
    /// thread).
    pub routes_attempted: u64,
    /// Number of times a beadie-produced pointer was actually written
    /// into `function_pointers`. This counts real state transitions
    /// only (idempotent re-installs of the same pointer are skipped).
    pub installs: u64,
    /// Number of distinct functions registered as beads (one bead per
    /// promoted `IrFunctionId`). Reaches `installs` once every routed
    /// function has had its pointer installed.
    pub registered_beads: usize,
    /// Standard-tier route attempts.
    pub standard_routes_attempted: u64,
    /// Optimized-tier route attempts.
    pub optimized_routes_attempted: u64,
    /// Standard-tier pointer installs.
    pub standard_installs: u64,
    /// Optimized-tier pointer installs.
    pub optimized_installs: u64,
    /// Distinct Standard-tier beads.
    pub standard_registered_beads: usize,
    /// Distinct Optimized-tier beads.
    pub optimized_registered_beads: usize,
    /// Standard-tier beads whose broker compile has produced a pointer.
    pub standard_compiled_beads: usize,
    /// Optimized-tier beads whose broker compile has produced a pointer.
    pub optimized_compiled_beads: usize,
}

/// Per-tier beadie adapters: (Standard, Optimized).
type BeadieAdapters = (
    Arc<beadie::BackendAdapter<super::beadie_jit::BeadieJit>>, // standard
    Arc<beadie::BackendAdapter<super::beadie_jit::BeadieJit>>, // optimized
);

fn tier_event_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Value-gated: `=0`/`=false`/empty/unset all mean OFF (the old
        // `is_some()` treated `=0` as enabled).
        let on = |k: &str| {
            std::env::var(k)
                .map(|v| {
                    let v = v.trim();
                    !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
                })
                .unwrap_or(false)
        };
        on("RAYZOR_PROFILE_DECODE") || on("RAYZOR_PROFILE_TIER_EVENTS")
    })
}

fn tier_event_time_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn emit_tier_event(
    kind: &str,
    func_id: Option<IrFunctionId>,
    tier: Option<OptimizationTier>,
    detail: &str,
) {
    if !tier_event_enabled() {
        return;
    }
    match (func_id, tier) {
        (Some(func_id), Some(tier)) => eprintln!(
            "[tier-event] t={:.6} kind={} func={} tier={}{}{}",
            tier_event_time_s(),
            kind,
            func_id.0,
            tier.event_label(),
            if detail.is_empty() { "" } else { " " },
            detail
        ),
        (Some(func_id), None) => eprintln!(
            "[tier-event] t={:.6} kind={} func={}{}{}",
            tier_event_time_s(),
            kind,
            func_id.0,
            if detail.is_empty() { "" } else { " " },
            detail
        ),
        (None, Some(tier)) => eprintln!(
            "[tier-event] t={:.6} kind={} tier={}{}{}",
            tier_event_time_s(),
            kind,
            tier.event_label(),
            if detail.is_empty() { "" } else { " " },
            detail
        ),
        (None, None) => eprintln!(
            "[tier-event] t={:.6} kind={}{}{}",
            tier_event_time_s(),
            kind,
            if detail.is_empty() { "" } else { " " },
            detail
        ),
    }
}

fn sanitize_tier_event_detail(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

/// Build the per-tier beadie adapters. Step 6 removed the
/// `enable_beadie_adapter` toggle and the `Option`-of-tuple return:
/// beadie is now unconditional, the broker thread is always live, and
/// `record_call` always routes Standard/Optimized through it.
///
/// Each `BackendAdapter<BeadieJit>` is one broker thread + one bead
/// registry. `BeadieJit` itself is stateless aside from the runtime
/// symbol snapshot — it builds a fresh `CraneliftBackend` per compile
/// and `Box::leak`s it.
///
/// Beadie's policy threshold is hardcoded to 1 (immediate-on-route).
/// Rayzor's `ProfileData` already owns the "is this hot enough"
/// decision via tier-specific thresholds (`warm_threshold`,
/// `hot_threshold`); beadie's job once routed is to compile
/// immediately, not re-evaluate. The double-threshold issue this
/// avoids: under narrow promotion windows (Benchmark preset:
/// `warm_threshold=3`, `hot_threshold=5` — only 2 calls between
/// them), a non-1 beadie threshold would never let beadie's counter
/// cross before rayzor moves on to the next tier.
fn build_beadie_adapters(
    symbols: &[(&str, *const u8)],
    verbosity: u8,
) -> Result<BeadieAdapters, String> {
    let beadie_threshold: u32 = 1;
    let symbols_snapshot: super::beadie_jit::RuntimeSymbolTable = Arc::new(
        symbols
            .iter()
            .map(|(name, ptr)| ((*name).to_string(), *ptr as usize))
            .collect(),
    );
    let standard = super::beadie_jit::build_batched_adapter_at(
        Arc::clone(&symbols_snapshot),
        OptimizationTier::Standard.cranelift_opt_level(),
        beadie_threshold,
        /*capacity=*/ 64,
        /*batch_limit=*/ 16,
    );
    let optimized = super::beadie_jit::build_batched_adapter_at(
        symbols_snapshot,
        OptimizationTier::Optimized.cranelift_opt_level(),
        beadie_threshold,
        /*capacity=*/ 64,
        /*batch_limit=*/ 16,
    );
    // One-line construction announcement on stderr, gated by verbosity
    // so benchmark/silent callers can suppress it. Step 6 simplifies:
    // beadie is unconditional, no env-var to report.
    if verbosity >= 1 {
        eprintln!(
            "[beadie] adapters built — Standard ({:?}) + Optimized ({:?}); \
             threshold={} (immediate-on-route)",
            OptimizationTier::Standard.cranelift_opt_level(),
            OptimizationTier::Optimized.cranelift_opt_level(),
            beadie_threshold,
        );
    }
    Ok((standard, optimized))
}

impl TieredBackend {
    /// Create a tiered backend from a preset
    ///
    /// # Example
    /// ```ignore
    /// use compiler::codegen::tiered_backend::{TieredBackend, TierPreset};
    ///
    /// let backend = TieredBackend::from_preset(TierPreset::Script)?;
    /// ```
    pub fn from_preset(preset: TierPreset) -> Result<Self, String> {
        Self::new(preset.to_config())
    }

    /// Create a tiered backend from a preset with runtime symbols
    ///
    /// # Example
    /// ```ignore
    /// use compiler::codegen::tiered_backend::{TieredBackend, TierPreset};
    ///
    /// let symbols = get_runtime_symbols();
    /// let backend = TieredBackend::from_preset_with_symbols(TierPreset::Server, &symbols)?;
    /// ```
    pub fn from_preset_with_symbols(
        preset: TierPreset,
        symbols: &[(&str, *const u8)],
    ) -> Result<Self, String> {
        Self::with_symbols(preset.to_config(), symbols)
    }

    /// Create a new tiered backend
    pub fn new(config: TieredConfig) -> Result<Self, String> {
        // IMPORTANT: Initialize LLVM on the main thread BEFORE any background workers start.
        #[cfg(feature = "llvm-backend")]
        super::llvm_jit_backend::init_llvm_once();

        let baseline_backend = CraneliftBackend::new()?;
        let profile_data = ProfileData::new(config.profile_config);
        let start_interpreted = config.start_interpreted;

        // Create interpreter with configured bailout threshold
        let mut interp = MirInterpreter::new();
        let max_iterations = if !config.enable_tier_promotion {
            u64::MAX
        } else {
            config.bailout_strategy.threshold()
        };
        interp.set_max_iterations(max_iterations);

        let baseline_backend = Arc::new(Mutex::new(baseline_backend));
        // Constructor `new` has no runtime symbols — pass an empty
        // slice. A beadie-compiled function with no runtime symbols
        // can't call any `haxe_*` runtime helper, which is fine for
        // tests of pure-arithmetic kernels but useless for real code.
        // Production callers go through `with_symbols`.
        let (beadie_adapter, beadie_adapter_optimized) =
            build_beadie_adapters(&[], config.verbosity)?;

        Ok(Self {
            interpreter: Arc::new(Mutex::new(interp)),
            baseline_backend,
            profile_data,
            function_tiers: Arc::new(RwLock::new(BTreeMap::new())),
            function_pointers: Arc::new(RwLock::new(BTreeMap::new())),
            modules: Arc::new(RwLock::new(Vec::new())),
            config,
            start_interpreted,
            runtime_symbols: Arc::new(Vec::new()),
            llvm_queue: Arc::new(Mutex::new(VecDeque::new())),
            #[cfg(feature = "llvm-backend")]
            llvm_compiled: Arc::new(Mutex::new(false)),
            promotion_barrier: Arc::new(PromotionBarrier::new()),
            initialized_module_count: Arc::new(Mutex::new(0)),
            beadie_adapter,
            beadie_beads: Arc::new(Mutex::new(BTreeMap::new())),
            beadie_adapter_optimized,
            beadie_beads_optimized: Arc::new(Mutex::new(BTreeMap::new())),
            beadie_routes_attempted: Arc::new(AtomicU64::new(0)),
            beadie_standard_routes_attempted: Arc::new(AtomicU64::new(0)),
            beadie_optimized_routes_attempted: Arc::new(AtomicU64::new(0)),
            beadie_installs: Arc::new(AtomicU64::new(0)),
            beadie_standard_installs: Arc::new(AtomicU64::new(0)),
            beadie_optimized_installs: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a new tiered backend with runtime symbols for interpreter and LLVM FFI
    pub fn with_symbols(
        config: TieredConfig,
        symbols: &[(&str, *const u8)],
    ) -> Result<Self, String> {
        // IMPORTANT: Initialize LLVM on the main thread BEFORE any background workers start.
        // This prevents crashes from LLVM initialization racing with background compilation.
        #[cfg(feature = "llvm-backend")]
        super::llvm_jit_backend::init_llvm_once();

        // Create baseline backend WITH runtime symbols for extern function linking
        // This is required when start_interpreted=false (JIT-only mode)
        let baseline_backend = CraneliftBackend::with_symbols(symbols)?;
        let profile_data = ProfileData::new(config.profile_config);
        let start_interpreted = config.start_interpreted;

        // Store symbols for later LLVM backend use
        let runtime_symbols: Vec<(String, usize)> = symbols
            .iter()
            .map(|(name, ptr)| (name.to_string(), *ptr as usize))
            .collect();

        // Create interpreter with configured bailout threshold and register symbols
        let mut interp = MirInterpreter::new();
        let max_iterations = if !config.enable_tier_promotion {
            u64::MAX
        } else {
            config.bailout_strategy.threshold()
        };
        interp.set_max_iterations(max_iterations);
        for (name, ptr) in symbols {
            interp.register_symbol(name, *ptr);
        }

        let baseline_backend = Arc::new(Mutex::new(baseline_backend));
        let (beadie_adapter, beadie_adapter_optimized) =
            build_beadie_adapters(symbols, config.verbosity)?;

        Ok(Self {
            interpreter: Arc::new(Mutex::new(interp)),
            baseline_backend,
            profile_data,
            function_tiers: Arc::new(RwLock::new(BTreeMap::new())),
            function_pointers: Arc::new(RwLock::new(BTreeMap::new())),
            modules: Arc::new(RwLock::new(Vec::new())),
            config,
            start_interpreted,
            runtime_symbols: Arc::new(runtime_symbols),
            llvm_queue: Arc::new(Mutex::new(VecDeque::new())),
            #[cfg(feature = "llvm-backend")]
            llvm_compiled: Arc::new(Mutex::new(false)),
            promotion_barrier: Arc::new(PromotionBarrier::new()),
            initialized_module_count: Arc::new(Mutex::new(0)),
            beadie_adapter,
            beadie_beads: Arc::new(Mutex::new(BTreeMap::new())),
            beadie_adapter_optimized,
            beadie_beads_optimized: Arc::new(Mutex::new(BTreeMap::new())),
            beadie_routes_attempted: Arc::new(AtomicU64::new(0)),
            beadie_standard_routes_attempted: Arc::new(AtomicU64::new(0)),
            beadie_optimized_routes_attempted: Arc::new(AtomicU64::new(0)),
            beadie_installs: Arc::new(AtomicU64::new(0)),
            beadie_standard_installs: Arc::new(AtomicU64::new(0)),
            beadie_optimized_installs: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Borrow the Standard-tier beadie adapter. Always present (step
    /// 6: beadie is unconditional).
    pub fn beadie_adapter(&self) -> &Arc<beadie::BackendAdapter<super::beadie_jit::BeadieJit>> {
        &self.beadie_adapter
    }

    /// Borrow the beadie bead registry — one `BoundBead` per
    /// `IrFunctionId` that's been touched by the beadie path. Lazy:
    /// entries are only created on first call to
    /// [`Self::ensure_beadie_bead`].
    ///
    /// Returned as an `Arc<Mutex<_>>` so call-site code (eventually
    /// `record_call` in step 3) can clone it cheaply for use across
    /// the broker thread boundary.
    pub fn beadie_beads(
        &self,
    ) -> &Arc<Mutex<BTreeMap<IrFunctionId, beadie::BoundBead<super::beadie_jit::BeadieJit>>>> {
        &self.beadie_beads
    }

    /// Returns `true` if `func_id` is already registered in the bead
    /// registry. Cheap probe — used by callers that want to avoid
    /// allocating a new bead when the function has been seen before.
    pub fn has_beadie_bead(&self, func_id: IrFunctionId) -> bool {
        self.beadie_beads.lock().unwrap().contains_key(&func_id)
    }

    /// Phase B counters: how often `record_call` tried to route a
    /// Standard-tier promotion through beadie, and how many of those
    /// resulted in an actual pointer install. The first counter
    /// double-counts retry calls (each `record_call` past the
    /// threshold attempts again until the install happens); the
    /// second counter only fires on real state transitions.
    ///
    /// Used by benchmark harnesses to confirm the integration is
    /// active and to attribute compile counts.
    pub fn beadie_stats(&self) -> BeadieStats {
        let standard_beads = self.beadie_beads.lock().unwrap();
        let optimized_beads = self.beadie_beads_optimized.lock().unwrap();
        let standard_registered_beads = standard_beads.len();
        let optimized_registered_beads = optimized_beads.len();
        let standard_compiled_beads = standard_beads
            .values()
            .filter(|bound| bound.bead().compiled().is_some())
            .count();
        let optimized_compiled_beads = optimized_beads
            .values()
            .filter(|bound| bound.bead().compiled().is_some())
            .count();
        BeadieStats {
            adapter_enabled: true,
            routes_attempted: self.beadie_routes_attempted.load(Ordering::Relaxed),
            installs: self.beadie_installs.load(Ordering::Relaxed),
            registered_beads: standard_registered_beads + optimized_registered_beads,
            standard_routes_attempted: self
                .beadie_standard_routes_attempted
                .load(Ordering::Relaxed),
            optimized_routes_attempted: self
                .beadie_optimized_routes_attempted
                .load(Ordering::Relaxed),
            standard_installs: self.beadie_standard_installs.load(Ordering::Relaxed),
            optimized_installs: self.beadie_optimized_installs.load(Ordering::Relaxed),
            standard_registered_beads,
            optimized_registered_beads,
            standard_compiled_beads,
            optimized_compiled_beads,
        }
    }

    /// Snapshot of per-tier function counts + profiling stats.
    ///
    /// Step 6 dropped the `queued_for_optimization` /
    /// `currently_optimizing` fields from
    /// [`TieredStatistics`] because the legacy queue is gone; this
    /// method recomputes the per-tier counts from `function_tiers`
    /// and forwards `ProfileData::get_statistics`.
    pub fn get_statistics(&self) -> TieredStatistics {
        let tiers = self.function_tiers.read().unwrap();
        let mut interpreted = 0;
        let mut baseline = 0;
        let mut standard = 0;
        let mut optimized = 0;
        let mut maximum = 0;
        for tier in tiers.values() {
            match tier {
                OptimizationTier::Interpreted => interpreted += 1,
                OptimizationTier::Baseline => baseline += 1,
                OptimizationTier::Standard => standard += 1,
                OptimizationTier::Optimized => optimized += 1,
                OptimizationTier::Maximum => maximum += 1,
            }
        }
        TieredStatistics {
            profile_stats: self.profile_data.get_statistics(),
            interpreted_functions: interpreted,
            baseline_functions: baseline,
            standard_functions: standard,
            optimized_functions: optimized,
            llvm_functions: maximum,
        }
    }

    /// Read the installed JIT pointer for `func_id`, or `None` if no
    /// pointer has been installed yet. Used by tests to observe that
    /// the beadie path (or the legacy path) installed a callable
    /// pointer for a function.
    pub fn jit_pointer(&self, func_id: IrFunctionId) -> Option<usize> {
        self.function_pointers
            .read()
            .unwrap()
            .get(&func_id)
            .copied()
    }

    /// Read the current optimization tier for `func_id`. Used by
    /// tests to observe tier transitions across the beadie /
    /// legacy boundary.
    pub fn function_tier(&self, func_id: IrFunctionId) -> Option<OptimizationTier> {
        self.function_tiers.read().unwrap().get(&func_id).copied()
    }
}

impl TieredBackend {
    /// Register a bead for `func_id` against the beadie adapter,
    /// returning `Err` if Phase B is disabled. Idempotent: if the bead
    /// already exists, this is a no-op. Step 3 will call this lazily
    /// from `record_call` to register beads on first warm-tier
    /// promotion.
    ///
    /// The `core_ptr` argument is forwarded to
    /// [`beadie::BackendAdapter::register`]; rayzor passes
    /// `std::ptr::null_mut()` because rayzor's interpreter path is the
    /// "core" fallback and is dispatched through
    /// `TieredBackend::execute_function`, not through a raw pointer
    /// the bead would call.
    pub fn ensure_beadie_bead(&self, func_id: IrFunctionId) -> Result<(), &'static str> {
        self.ensure_beadie_bead_for(func_id, OptimizationTier::Standard)
    }

    /// Lazy-register a bead for `func_id` against the adapter for
    /// `tier` (Standard or Optimized). Idempotent. Errors when the
    /// requested tier's adapter is disabled or `tier` is not a beadie
    /// tier.
    fn ensure_beadie_bead_for(
        &self,
        func_id: IrFunctionId,
        tier: OptimizationTier,
    ) -> Result<(), &'static str> {
        let Some(adapter) = self.beadie_adapter_for(tier) else {
            return Err("beadie adapter is disabled for the requested tier");
        };
        let beads_arc = self
            .beadie_beads_for(tier)
            .ok_or("no bead registry for tier")?;
        let mut beads = beads_arc.lock().unwrap();
        if beads.contains_key(&func_id) {
            return Ok(());
        }
        let bound = adapter.register(std::ptr::null_mut(), None);
        beads.insert(func_id, bound);
        emit_tier_event("beadie_register", Some(func_id), Some(tier), "");
        Ok(())
    }

    /// Map a target tier to the corresponding beadie adapter, or
    /// `None` if the tier doesn't go through beadie / the adapter is
    /// disabled. Currently Standard + Optimized are the only beadie
    /// tiers; Baseline keeps the legacy big-bang and Maximum keeps
    /// LLVM.
    fn beadie_adapter_for(
        &self,
        tier: OptimizationTier,
    ) -> Option<&Arc<beadie::BackendAdapter<super::beadie_jit::BeadieJit>>> {
        match tier {
            OptimizationTier::Standard => Some(&self.beadie_adapter),
            OptimizationTier::Optimized => Some(&self.beadie_adapter_optimized),
            _ => None,
        }
    }

    /// Map a target tier to its bead registry. Returns `None` for
    /// non-beadie tiers.
    fn beadie_beads_for(
        &self,
        tier: OptimizationTier,
    ) -> Option<&Arc<Mutex<BTreeMap<IrFunctionId, beadie::BoundBead<super::beadie_jit::BeadieJit>>>>>
    {
        match tier {
            OptimizationTier::Standard => Some(&self.beadie_beads),
            OptimizationTier::Optimized => Some(&self.beadie_beads_optimized),
            _ => None,
        }
    }

    /// Phase B step 3 / 5: route a promotion to `target_tier` through
    /// beadie's broker instead of the legacy `enqueue_for_optimization`
    /// path. Returns `true` if forwarded; `false` if the caller should
    /// fall back to legacy (tier doesn't go through beadie, adapter
    /// disabled, or function not in any loaded module).
    fn route_through_beadie(&self, func_id: IrFunctionId, target_tier: OptimizationTier) -> bool {
        let Some(adapter) = self.beadie_adapter_for(target_tier) else {
            return false;
        };
        emit_tier_event("beadie_route", Some(func_id), Some(target_tier), "");
        if self.ensure_beadie_bead_for(func_id, target_tier).is_err() {
            return false;
        }

        // Verify the function actually belongs to a loaded module
        // before we hand its id to beadie — beadie's compile_outcome
        // would otherwise hit the "function not in any module" error
        // path on its own thread, where it can't surface a useful
        // diagnostic to the user.
        {
            let modules = self.modules.read().unwrap();
            if !modules.iter().any(|m| m.functions.contains_key(&func_id)) {
                return false;
            }
        }

        let modules_handle = Arc::clone(&self.modules);
        let beads_arc = match self.beadie_beads_for(target_tier) {
            Some(b) => b,
            None => return false,
        };
        // Hold the bead registry lock only for the brief
        // `on_invoke_outcome` call (which increments beadie's counter
        // + maybe queues a compile) and the post-call `compiled()`
        // probe. Both are short.
        let immediate_ptr = {
            let beads = beads_arc.lock().unwrap();
            let Some(bound) = beads.get(&func_id) else {
                return false;
            };
            let outcome =
                adapter.on_invoke_outcome(bound, move |_| super::beadie_jit::BeadieFunctionDef {
                    modules: modules_handle,
                    func_id,
                });
            outcome.or_else(|| bound.bead().compiled())
        };

        if let Some(ptr) = immediate_ptr {
            emit_tier_event(
                "beadie_ready_immediate",
                Some(func_id),
                Some(target_tier),
                "",
            );
            self.install_beadie_pointer(func_id, ptr as usize, target_tier);
        }
        true
    }

    /// Install a beadie-produced native pointer for `func_id` at
    /// `target_tier` under the `PromotionBarrier`. Idempotent; skips
    /// the barrier dance + writes when the pointer is already
    /// installed at this tier or when a higher tier is already in
    /// place (no-downgrade guard).
    fn install_beadie_pointer(
        &self,
        func_id: IrFunctionId,
        ptr: usize,
        target_tier: OptimizationTier,
    ) {
        // Short-circuit if already installed.
        {
            let fp = self.function_pointers.read().unwrap();
            if fp.get(&func_id).copied() == Some(ptr) {
                return;
            }
        }
        // Don't downgrade: drop the beadie pointer on the floor if a
        // higher tier is already in place. Benign — the beadie code
        // stays alive (Box::leak'd), it's just never called.
        {
            let ft = self.function_tiers.read().unwrap();
            if let Some(current) = ft.get(&func_id).copied() {
                if current as u8 > target_tier as u8 {
                    return;
                }
            }
        }
        if !self.promotion_barrier.request_promotion() {
            // Another promotion is already in flight — retry next call.
            emit_tier_event("install_barrier_busy", Some(func_id), Some(target_tier), "");
            return;
        }
        if !self
            .promotion_barrier
            .wait_for_drain(Duration::from_secs(1))
        {
            self.promotion_barrier.cancel_promotion();
            emit_tier_event(
                "install_barrier_timeout",
                Some(func_id),
                Some(target_tier),
                "",
            );
            return;
        }
        {
            let mut fp = self.function_pointers.write().unwrap();
            let mut ft = self.function_tiers.write().unwrap();
            fp.insert(func_id, ptr);
            ft.insert(func_id, target_tier);
        }
        self.promotion_barrier.complete_promotion();
        emit_tier_event("beadie_install", Some(func_id), Some(target_tier), "");
        self.beadie_installs.fetch_add(1, Ordering::Relaxed);
        match target_tier {
            OptimizationTier::Standard => {
                self.beadie_standard_installs
                    .fetch_add(1, Ordering::Relaxed);
            }
            OptimizationTier::Optimized => {
                self.beadie_optimized_installs
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        if self.config.verbosity >= 1 {
            eprintln!(
                "[beadie] installed pointer for {:?} at {} tier (total installs: {})",
                func_id,
                target_tier.description(),
                self.beadie_installs.load(Ordering::Relaxed)
            );
        }
    }

    /// Phase B step 3 / 5: poll bead registries (Standard + Optimized)
    /// for a newly-compiled pointer and install if available. Called
    /// from `execute_function` before tier dispatch so that a
    /// background compile finished between `record_call` and the next
    /// invocation gets observed promptly.
    ///
    /// Returns `true` if a pointer was just installed. Optimized is
    /// checked first; if it has a compiled pointer, it wins over
    /// Standard (same no-downgrade logic as the install path).
    fn maybe_install_compiled_beadie_pointer(&self, func_id: IrFunctionId) -> bool {
        // Optimized first — higher tier wins.
        for tier in [OptimizationTier::Optimized, OptimizationTier::Standard] {
            let Some(beads_arc) = self.beadie_beads_for(tier) else {
                continue;
            };
            let ptr_opt = {
                let beads = beads_arc.lock().unwrap();
                beads
                    .get(&func_id)
                    .and_then(|bound| bound.bead().compiled())
            };
            let Some(ptr) = ptr_opt else { continue };
            // Skip if already installed (avoid the barrier dance on
            // every hot call once beadie has handed us a pointer).
            {
                let fp = self.function_pointers.read().unwrap();
                if fp.get(&func_id).copied() == Some(ptr as usize) {
                    continue;
                }
            }
            self.install_beadie_pointer(func_id, ptr as usize, tier);
            return true;
        }
        false
    }

    fn ensure_loaded_modules_initialized(&mut self) -> Result<(), String> {
        let start_idx = *self.initialized_module_count.lock().unwrap();
        let modules_to_init: Vec<IrModule> = {
            let modules = self.modules.read().unwrap();
            if start_idx >= modules.len() {
                return Ok(());
            }
            modules[start_idx..].to_vec()
        };

        if modules_to_init.is_empty() {
            return Ok(());
        }

        let module_refs: Vec<_> = modules_to_init.iter().cloned().map(Arc::new).collect();
        CraneliftBackend::register_enum_rtti_from_modules(&module_refs);
        CraneliftBackend::register_class_rtti_from_modules(&module_refs);

        // `__vtable_init__` calls `haxe_iface_vtable_set_slot` with a
        // FUNCTION POINTER argument (the interface method's dispatch thunk)
        // that later gets called INDIRECTLY by compiled code rebuilding a
        // wider interface fat pointer (`haxe_iface_fat_ptr_build`, used by
        // an interface-to-interface cast whose target has methods the
        // source fat pointer's vtable doesn't cover). The interpreter's
        // `fn_ref` value is a tagged IrFunctionId (see `NanBoxedValue::
        // from_func_id` in mir_interpreter.rs), not a real machine-code
        // address — running `__vtable_init__` through the interpreter would
        // register that placeholder as if it were a callable pointer, and
        // compiled code would later `blr` through it and crash on a tiny
        // garbage address. The interpreter also has no dispatch arm for
        // `haxe_iface_vtable_set_slot` at all (falls through to the
        // "unknown extern → forgiving default" path and silently no-ops),
        // so today the registry entry is simply MISSING for any class whose
        // vtable-init only ever ran interpreted — same crash, different
        // proximate cause (null instead of garbage).
        //
        // In JIT mode (`!start_interpreted`, the default for `rayzor run`)
        // real Cranelift-compiled function pointers are available almost
        // immediately anyway (`execute_function` compiles all modules on
        // its very next step if this is the first call). Compile eagerly
        // here and run the NATIVE startup hooks instead, so `fn_ref`
        // resolves to the actual compiled thunk address before any
        // `__vtable_init__` body executes. Interpreter-first configs keep
        // the interpreted path — cross-interface casts needing a vtable
        // rebuild in a pure-interpreter run remain a known gap, not a new
        // regression from this fix.
        if !self.start_interpreted {
            if self.function_pointers.read().unwrap().is_empty() {
                self.compile_all_modules_jit()?;
            }
            self.run_startup_hooks_native(&modules_to_init)?;
        } else {
            self.run_startup_hooks_interpreter(&modules_to_init)?;
        }

        *self.initialized_module_count.lock().unwrap() = start_idx + modules_to_init.len();
        Ok(())
    }

    fn run_startup_hooks_interpreter(&self, modules: &[IrModule]) -> Result<(), String> {
        let normal_max_iterations = if !self.config.enable_tier_promotion {
            u64::MAX
        } else {
            self.config.bailout_strategy.threshold()
        };

        let mut interp = self.interpreter.lock().unwrap();
        // Module startup hooks should not trip hot-loop bailout. They are required
        // initialization work, not benchmark hot paths, and can legitimately exceed
        // the low iteration thresholds used by the Benchmark preset.
        interp.reset_iteration_count();
        interp.set_max_iterations(u64::MAX);
        for module in modules {
            // Run every `__vtable_init__` / `__init__` in the module
            // (see the native equivalent above for the why). The
            // interpreter is forgiving of missing externs (returns
            // default), so we don't need a trap-stub-skip here.
            for init_name in ["__vtable_init__", "__init__"] {
                let init_ids: Vec<_> = module
                    .functions
                    .values()
                    .filter(|f| f.name == init_name)
                    .map(|f| f.id)
                    .collect();
                for init_id in init_ids {
                    interp
                        .execute(module, init_id, vec![])
                        .map_err(|e| format!("Interpreter error in {}: {}", init_name, e))?;
                }
            }
        }
        interp.reset_iteration_count();
        interp.set_max_iterations(normal_max_iterations);
        Ok(())
    }

    fn run_startup_hooks_native(&self, modules: &[IrModule]) -> Result<(), String> {
        let function_pointers = self.function_pointers.read().unwrap();
        let trace_startup = std::env::var_os("RAYZOR_LLVM_TRACE_STARTUP").is_some()
            || std::env::var_os("RAYZOR_TIER_TRACE_STARTUP").is_some();

        for module in modules {
            // Every contributing file leaves its own `__vtable_init__` /
            // `__init__` with a distinct IrFunctionId. Running just the
            // first one silently dropped vtable + iface registrations
            // from every other file, which broke iface-to-iface casts
            // on classes defined outside the entry file (the
            // (class, iface) entry was never put in the runtime
            // `IFACE_VTABLE_REGISTRY`).
            //
            // Skip entries whose JIT compilation failed (no
            // function_pointers entry) — calling a trap-stub at
            // startup would SIGILL before user code begins. The
            // missing entries from a single skipped init are
            // recoverable, but a SIGILL is not.
            for init_name in ["__vtable_init__", "__init__"] {
                let init_ids: Vec<_> = module
                    .functions
                    .values()
                    .filter(|f| f.name == init_name)
                    .map(|f| f.id)
                    .collect();
                for init_id in init_ids {
                    let Some(&func_ptr) = function_pointers.get(&init_id) else {
                        continue;
                    };
                    if func_ptr == 0 {
                        if trace_startup {
                            eprintln!(
                                "[tier-startup] skipping zero pointer for {} {:?} in {}",
                                init_name, init_id, module.name
                            );
                        }
                        continue;
                    }
                    if trace_startup {
                        eprintln!(
                            "[tier-startup] calling {} {:?} in {} @ 0x{:x}",
                            init_name, init_id, module.name, func_ptr
                        );
                    }
                    unsafe {
                        let init_fn: extern "C" fn(i64) =
                            std::mem::transmute(func_ptr as *const u8);
                        init_fn(0);
                    }
                    if trace_startup {
                        eprintln!(
                            "[tier-startup] finished {} {:?} in {}",
                            init_name, init_id, module.name
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn rerun_loaded_modules_startup_native(&self) -> Result<(), String> {
        let modules = self.modules.read().unwrap();
        self.run_startup_hooks_native(&modules)
    }

    /// Reset loaded module startup state for the next top-level program run.
    ///
    /// Benchmarks reuse the same backend across warmups and measured iterations, so
    /// globals and runtime dispatch tables must be reinitialized before each run.
    pub fn reset_loaded_modules_for_run(&mut self) -> Result<(), String> {
        let loaded_module_count = self.modules.read().unwrap().len();
        let initialized_module_count = *self.initialized_module_count.lock().unwrap();
        if initialized_module_count < loaded_module_count {
            return self.ensure_loaded_modules_initialized();
        }

        let modules = self.modules.read().unwrap();
        if self.function_pointers.read().unwrap().is_empty() {
            self.run_startup_hooks_interpreter(&modules)
        } else {
            self.run_startup_hooks_native(&modules)
        }
    }

    /// Re-run startup hooks after compiling to native code.
    ///
    /// The interpreter maintains its own global state, while native tiers read/write
    /// backend storage and runtime registries. After switching execution tiers, we
    /// must replay startup so globals, vtables, and constructor thunks point at the
    /// newly active backend representation.
    fn rerun_loaded_modules_startup_jit(&self) -> Result<(), String> {
        self.rerun_loaded_modules_startup_native()?;
        Ok(())
    }

    /// Check if a function uses SIMD/vector instructions.
    /// The interpreter returns void for all vector ops, so these functions
    /// must skip Tier 0 (Interpreted) and start at Baseline (Cranelift JIT).
    fn function_uses_simd(func: &IrFunction) -> bool {
        for block in func.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::VectorLoad { .. }
                    | IrInstruction::VectorStore { .. }
                    | IrInstruction::VectorBinOp { .. }
                    | IrInstruction::VectorSplat { .. }
                    | IrInstruction::VectorExtract { .. }
                    | IrInstruction::VectorInsert { .. }
                    | IrInstruction::VectorReduce { .. }
                    | IrInstruction::VectorUnaryOp { .. }
                    | IrInstruction::VectorMinMax { .. }
                    | IrInstruction::VectorDot { .. }
                    // Atomics need real shared memory → force straight to JIT.
                    | IrInstruction::AtomicLoad { .. }
                    | IrInstruction::AtomicStore { .. }
                    | IrInstruction::AtomicRmw { .. }
                    | IrInstruction::AtomicCas { .. } => return true,
                    _ => {}
                }
            }
        }
        false
    }

    /// Compile/load a MIR module
    ///
    /// If `start_interpreted` is true:
    /// - Functions start at Phase 0 (Interpreted) for instant startup
    /// - Functions using SIMD instructions are promoted to Baseline immediately
    ///   (the interpreter doesn't support vector operations)
    /// - Background worker will JIT-compile functions as they get hot
    ///
    /// If `start_interpreted` is false:
    /// - Functions are compiled to Phase 1 (Baseline) immediately
    pub fn compile_module(&mut self, module: IrModule) -> Result<(), String> {
        let initial_tier = if self.start_interpreted {
            OptimizationTier::Interpreted
        } else {
            OptimizationTier::Baseline
        };

        if self.config.verbosity >= 1 {
            debug!(
                "[TieredBackend] Loading {} functions at {} ({})",
                module.functions.len(),
                initial_tier.description(),
                if self.start_interpreted {
                    "instant startup"
                } else {
                    "JIT compiled"
                }
            );
        }

        // Register function tiers (actual compilation deferred for JIT mode)
        // Functions using SIMD instructions skip the interpreter tier since
        // the interpreter returns void for all vector operations.
        for (func_id, func) in &module.functions {
            let tier = if initial_tier == OptimizationTier::Interpreted
                && Self::function_uses_simd(func)
            {
                if self.config.verbosity >= 1 {
                    debug!(
                        "[TieredBackend] Force-promoting {:?} to Baseline (uses SIMD)",
                        func_id
                    );
                }
                OptimizationTier::Baseline
            } else {
                initial_tier
            };
            self.function_tiers.write().unwrap().insert(*func_id, tier);
        }

        // Store module for later recompilation/interpretation
        self.modules.write().unwrap().push(module);

        Ok(())
    }

    /// Execute a function (interpreter or JIT based on current tier)
    ///
    /// Returns the result as an InterpValue, which can be converted to native types.
    pub fn execute_function(
        &mut self,
        func_id: IrFunctionId,
        args: Vec<InterpValue>,
    ) -> Result<InterpValue, String> {
        self.ensure_loaded_modules_initialized()?;

        // Record the call for profiling
        self.record_call(func_id);

        // Process LLVM queue (main thread compilation)
        // This is safe because execute_function runs on the main thread
        self.process_llvm_queue();

        // Phase B step 3: poll the beadie bead registry for a pointer
        // produced by a background compile since the last call. If one
        // is available, install it under the PromotionBarrier so the
        // tier read below sees the upgraded state. No-op when beadie
        // is disabled or the function has no registered bead.
        if true {
            self.maybe_install_compiled_beadie_pointer(func_id);
        }

        self.maybe_promote_interpreter_threshold_to_baseline(func_id)?;

        // Get current tier
        let tier = self
            .function_tiers
            .read()
            .unwrap()
            .get(&func_id)
            .copied()
            .unwrap_or(OptimizationTier::Interpreted);

        // Lazy JIT compilation. Trigger when:
        //   1. JIT mode (`!start_interpreted`) — original behaviour: every
        //      function compiles up front on first execute.
        //   2. Interpreter mode AND the requested function was *pre-promoted*
        //      to Baseline at compile_module time — currently this happens
        //      for SIMD-using functions (see `function_uses_simd`). Without
        //      this branch, the JIT-path lookup fails because no module has
        //      been compiled yet (interp mode normally defers compilation
        //      until a JitBailout fires).
        let pre_promoted_in_interp_mode =
            self.start_interpreted && tier == OptimizationTier::Baseline;
        if (!self.start_interpreted && tier == OptimizationTier::Baseline)
            || pre_promoted_in_interp_mode
        {
            // Check if we need to compile (no function pointers yet)
            let needs_compile = self.function_pointers.read().unwrap().is_empty();
            if needs_compile {
                self.compile_all_modules_jit()?;
            }
            if pre_promoted_in_interp_mode {
                self.promote_compiled_functions_to_baseline();
            }
        }

        // Debug: print current tier
        if self.config.verbosity >= 2 {
            let count = self.profile_data.get_function_count(func_id);
            debug!(
                "[TieredBackend] Executing {:?} at tier {:?} (count: {})",
                func_id, tier, count
            );
        }

        if tier.uses_interpreter() {
            // Execute via interpreter - find the module containing this function
            let modules = self.modules.read().unwrap();
            let module_ref = modules
                .iter()
                .find(|m| m.functions.contains_key(&func_id))
                .ok_or_else(|| format!("Function {:?} not found in any module", func_id))?;
            let mut interp = self.interpreter.lock().unwrap();
            let result = interp.execute(module_ref, func_id, args.clone());
            drop(interp); // Release lock before potential recompilation
            drop(modules);

            match result {
                Ok(value) => Ok(value),
                Err(InterpError::JitBailout(bailout_func_id)) => {
                    // Hot loop detected! Promote to JIT and re-execute
                    tracing::trace!(
                        "[TieredBackend] JIT bailout for {:?} - promoting to Baseline tier",
                        bailout_func_id
                    );

                    // Compile all modules with JIT if not already compiled
                    let needs_compile = self.function_pointers.read().unwrap().is_empty();
                    if needs_compile {
                        self.compile_all_modules_jit()?;
                    }

                    // Promote ALL compiled functions to Baseline tier
                    // This is crucial: without this, the recursive execute_function call
                    // would still think the functions are at Interpreted tier
                    self.promote_compiled_functions_to_baseline();

                    // Reset interpreter iteration counter
                    {
                        let mut interp = self.interpreter.lock().unwrap();
                        interp.reset_iteration_count();
                    }

                    // Re-execute using JIT (recursive call will use JIT path now)
                    self.execute_function(func_id, args)
                }
                Err(e) => Err(format!("Interpreter error: {}", e)),
            }
        } else {
            // JIT-compiled code - call via function pointer
            //
            // BARRIER PROTOCOL:
            // 1. Wait for any pending promotion to complete (blocks if PromotionRequested)
            // 2. Enter execution (increments counter, signals we're running JIT code)
            // 3. Get function pointer and execute
            // 4. Exit execution (decrements counter via RAII guard)
            //
            // This ensures the background worker can safely swap ALL function pointers
            // when no JIT code is executing.
            self.promotion_barrier.wait_and_enter_execution();
            let _exec_guard = ExecutionGuard::new(&self.promotion_barrier);

            // Now safe to get function pointer - no promotion can happen while we hold the guard
            let func_ptr = {
                let fp_guard = self.function_pointers.read().unwrap();
                fp_guard
                    .get(&func_id)
                    .map(|addr| *addr as *const u8)
                    .ok_or_else(|| {
                        format!("JIT function {:?} not found in function_pointers", func_id)
                    })?
            };

            // For functions with no args (like main), call directly
            // NOTE: Cranelift adds a hidden environment parameter (i64) to non-extern Haxe
            // functions. We must pass a null pointer for this parameter.
            // For functions with args, we'd need to marshal InterpValue -> native types
            if args.is_empty() {
                unsafe {
                    // Pass null environment pointer as required by Haxe calling convention
                    let jit_fn: extern "C" fn(i64) = std::mem::transmute(func_ptr);
                    jit_fn(0); // null environment pointer
                }
                // _exec_guard drops here, calling exit_execution()
                Ok(InterpValue::Void)
            } else {
                // TODO: Implement argument marshaling for JIT calls
                // For now, fall back to interpreter for functions with args
                // This is a limitation - in practice, hot inner functions often have args
                // Note: We keep the execution guard because the interpreter might call back into JIT
                if self.config.verbosity >= 1 {
                    debug!("[TieredBackend] JIT function with args - falling back to interpreter");
                }
                let modules = self.modules.read().unwrap();
                let module_ref = modules
                    .iter()
                    .find(|m| m.functions.contains_key(&func_id))
                    .ok_or_else(|| format!("Function {:?} not found in any module", func_id))?;
                let mut interp = self.interpreter.lock().unwrap();
                interp
                    .execute(module_ref, func_id, args)
                    .map_err(|e| format!("Interpreter error: {}", e))
            }
        }
    }

    /// Get a function pointer (for execution)
    pub fn get_function_pointer(&self, func_id: IrFunctionId) -> Option<*const u8> {
        self.function_pointers
            .read()
            .unwrap()
            .get(&func_id)
            .map(|addr| *addr as *const u8)
    }

    /// Get the current optimization tier for a function
    pub fn get_function_tier(&self, func_id: IrFunctionId) -> OptimizationTier {
        self.function_tiers
            .read()
            .unwrap()
            .get(&func_id)
            .copied()
            .unwrap_or(OptimizationTier::Interpreted)
    }

    /// Upgrade all functions to LLVM (Maximum tier) immediately
    ///
    /// This bypasses the normal tier promotion and compiles everything with LLVM.
    /// Useful for benchmarks where you want maximum performance from the start.
    ///
    /// Step 6 cleanup: no longer needs to stop the legacy background
    /// worker (it's gone). Beadie's broker threads run independently
    /// of LLVM compile; they may install Standard/Optimized pointers
    /// while LLVM is compiling, but `install_beadie_pointer`'s
    /// no-downgrade guard skips installs once Maximum lands.
    #[cfg(feature = "llvm-backend")]
    pub fn upgrade_to_llvm(&mut self) -> Result<(), String> {
        if self.config.verbosity >= 1 {
            debug!("[TieredBackend] Upgrading all functions to LLVM...");
        }

        match self.compile_all_with_llvm() {
            Ok(all_pointers) => {
                // Re-register source info at new LLVM addresses
                self.register_source_info_for_pointers(&all_pointers);

                let count = all_pointers.len();
                if std::env::var("RAYZOR_DUMP_FN_PTRS").is_ok() {
                    let modules = self.modules.read().unwrap();
                    for module in modules.iter() {
                        for (func_id, function) in &module.functions {
                            if let Some(ptr) = all_pointers.get(func_id) {
                                eprintln!(
                                    "[fn-ptr-llvm] {:?} {} -> 0x{:x}",
                                    func_id,
                                    function.qualified_name.as_deref().unwrap_or(&function.name),
                                    ptr
                                );
                            }
                        }
                    }
                }
                {
                    let mut fp_lock = self.function_pointers.write().unwrap();
                    let mut ft_lock = self.function_tiers.write().unwrap();
                    for (func_id, ptr) in all_pointers {
                        fp_lock.insert(func_id, ptr);
                        ft_lock.insert(func_id, OptimizationTier::Maximum);
                    }
                }

                if self.config.verbosity >= 1 {
                    debug!("[TieredBackend] Upgraded {} functions to LLVM", count);
                }
                self.rerun_loaded_modules_startup_jit()?;
                Ok(())
            }
            Err(e) => {
                if self.config.verbosity >= 1 {
                    debug!("[TieredBackend] LLVM upgrade failed: {}", e);
                }
                Err(e)
            }
        }
    }

    /// Stub for when LLVM backend is not enabled
    #[cfg(not(feature = "llvm-backend"))]
    pub fn upgrade_to_llvm(&mut self) -> Result<(), String> {
        Err("LLVM backend not enabled".to_string())
    }

    /// Record a function call (for profiling and tier promotion)
    /// This should be called before executing a function
    pub fn record_call(&self, func_id: IrFunctionId) {
        self.profile_data.record_function_call(func_id);
        let count = self.profile_data.get_function_count(func_id);

        if !self.config.enable_tier_promotion {
            return;
        }

        // Sample promotion checks, not the underlying execution counter.
        let sample_rate = self.profile_data.config().sample_rate.max(1);
        if count % sample_rate != 0 {
            return;
        }

        // Check if function should be promoted to a higher tier
        // Use count-based promotion that allows skipping tiers if count exceeds multiple thresholds
        let should_promote = {
            let tiers = self.function_tiers.read().unwrap();
            let current_tier = tiers
                .get(&func_id)
                .copied()
                .unwrap_or(OptimizationTier::Interpreted);

            let config = self.profile_data.config();

            // Determine target tier based on count (allows skipping tiers)
            let target_tier = if count >= config.blazing_threshold {
                OptimizationTier::Maximum
            } else if count >= config.hot_threshold {
                OptimizationTier::Optimized
            } else if count >= config.warm_threshold {
                OptimizationTier::Standard
            } else if count >= config.interpreter_threshold {
                OptimizationTier::Baseline
            } else {
                OptimizationTier::Interpreted
            };

            // Only promote if target tier is higher than current tier
            if target_tier as u8 > current_tier as u8 {
                Some(target_tier)
            } else {
                None
            }
        };

        if let Some(target_tier) = should_promote {
            emit_tier_event(
                "promote_target",
                Some(func_id),
                Some(target_tier),
                &format!("count={}", count),
            );
            // Phase B step 5/6: route Standard + Optimized through
            // beadie. The legacy queue + worker infrastructure is
            // gone (step 6 deletion), so there's no fallback for
            // those tiers — beadie owns them.
            //
            // Baseline is handled by `execute_function`, where we have
            // `&mut self` and can invoke `compile_all_modules_jit`.
            //
            // Maximum (LLVM) doesn't go through beadie either — it
            // must run on the main thread because of LLVM's
            // `add_global_mapping` constraint. Queue it here; the
            // queue is drained at the next `execute_function` entry
            // (a natural safe point on the main thread).
            //
            // ProfileData increment above stays in place so the tier
            // statistics report accurate hotness regardless of who
            // owns the compile.
            #[cfg(feature = "llvm-backend")]
            if matches!(target_tier, OptimizationTier::Maximum) {
                let mut queue = self.llvm_queue.lock().unwrap();
                if !queue.contains(&func_id) {
                    queue.push_back(func_id);
                }
            }
            if matches!(
                target_tier,
                OptimizationTier::Standard | OptimizationTier::Optimized
            ) {
                self.beadie_routes_attempted.fetch_add(1, Ordering::Relaxed);
                match target_tier {
                    OptimizationTier::Standard => {
                        self.beadie_standard_routes_attempted
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    OptimizationTier::Optimized => {
                        self.beadie_optimized_routes_attempted
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {}
                }
                self.route_through_beadie(func_id, target_tier);
            }
        }
    }

    fn maybe_promote_interpreter_threshold_to_baseline(
        &mut self,
        func_id: IrFunctionId,
    ) -> Result<(), String> {
        if !self.start_interpreted || !self.config.enable_tier_promotion {
            return Ok(());
        }

        let config = self.profile_data.config();
        if config.interpreter_threshold == u64::MAX
            || self.profile_data.get_function_count(func_id) < config.interpreter_threshold
        {
            return Ok(());
        }

        let current_tier = self
            .function_tiers
            .read()
            .unwrap()
            .get(&func_id)
            .copied()
            .unwrap_or(OptimizationTier::Interpreted);
        if current_tier != OptimizationTier::Interpreted {
            return Ok(());
        }

        if self.config.verbosity >= 1 {
            debug!(
                "[TieredBackend] {:?} crossed interpreter_threshold={} - compiling Baseline tier",
                func_id, config.interpreter_threshold
            );
        }

        let needs_compile = self.function_pointers.read().unwrap().is_empty();
        if needs_compile {
            emit_tier_event(
                "baseline_compile_start",
                Some(func_id),
                Some(OptimizationTier::Baseline),
                "",
            );
            self.compile_all_modules_jit()?;
            emit_tier_event(
                "baseline_compile_finish",
                Some(func_id),
                Some(OptimizationTier::Baseline),
                "",
            );
        }
        self.promote_compiled_functions_to_baseline();

        {
            let mut interp = self.interpreter.lock().unwrap();
            interp.reset_iteration_count();
        }

        Ok(())
    }

    fn promote_compiled_functions_to_baseline(&self) {
        let fp_lock = self.function_pointers.read().unwrap();
        let mut tiers = self.function_tiers.write().unwrap();
        for func_id in fp_lock.keys() {
            let current = tiers
                .get(func_id)
                .copied()
                .unwrap_or(OptimizationTier::Interpreted);
            if current < OptimizationTier::Baseline {
                tiers.insert(*func_id, OptimizationTier::Baseline);
            }
        }
    }

    /// Compile all modules to Cranelift in JIT mode (lazy compilation on first execution)
    ///
    /// This is called when `start_interpreted: false` and we need to compile all modules
    /// before executing any function. All modules are compiled to a single Cranelift backend
    /// to allow cross-module function calls.
    fn compile_all_modules_jit(&mut self) -> Result<(), String> {
        if self.config.verbosity >= 1 {
            debug!("[TieredBackend] JIT mode: compiling all modules to Cranelift");
        }

        // Instrument MIR functions with shadow call stack push/pop (debug mode only)
        if self.config.enable_stack_traces {
            Self::instrument_modules_for_stack_traces(&mut self.modules.write().unwrap());
        } else {
            // Strip update_call_frame_location calls that hir_to_mir emits unconditionally
            Self::strip_stack_trace_updates(&mut self.modules.write().unwrap());
        }

        // Convert runtime symbols to the format Cranelift expects
        let symbols: Vec<(&str, *const u8)> = self
            .runtime_symbols
            .iter()
            .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
            .collect();

        // Create a fresh Cranelift backend with runtime symbols
        let mut backend = CraneliftBackend::with_symbols(&symbols)?;

        // Compile all modules to the same backend WITHOUT finalizing between modules
        let modules = self.modules.read().unwrap();
        for module in modules.iter() {
            backend.compile_module_without_finalize(module)?;
        }

        // Register RTTI from MIR type definitions so that
        // getName()/getParameters()/trace/Type API work correctly at runtime
        for module in modules.iter() {
            Self::register_enum_rtti_from_module(module);
            Self::register_class_rtti_from_module(module);
        }

        // Finalize all modules at once (must be done before getting function pointers)
        backend.finalize()?;

        // Enable stack traces in the runtime if configured
        if self.config.enable_stack_traces {
            if let Some((_, ptr)) = self
                .runtime_symbols
                .iter()
                .find(|(name, _)| name == "rayzor_set_stack_traces_enabled")
            {
                let enable_fn: extern "C" fn(i32) = unsafe { std::mem::transmute(*ptr) };
                enable_fn(1);
            }
        }

        // Store function pointers for functions with bodies (non-extern)
        // Extern functions have empty CFGs and are imported, not compiled
        // Also register source info for debug-mode stack traces
        let register_fn: Option<
            extern "C" fn(u32, usize, *const u8, usize, *const u8, usize, u32, u32),
        > = if self.config.enable_stack_traces {
            self.runtime_symbols
                .iter()
                .find(|(name, _)| name == "rayzor_register_function_source")
                .map(|(_, ptr)| unsafe { std::mem::transmute(*ptr) })
        } else {
            None
        };

        for module in modules.iter() {
            for (func_id, function) in &module.functions {
                // Skip extern functions (no body to compile)
                if function.cfg.blocks.is_empty() {
                    continue;
                }
                if let Ok(ptr) = backend.get_function_ptr(*func_id) {
                    if std::env::var("RAYZOR_DUMP_FN_PTRS").is_ok() {
                        eprintln!(
                            "[fn-ptr] {:?} {} -> 0x{:x}",
                            func_id,
                            function.qualified_name.as_deref().unwrap_or(&function.name),
                            ptr as usize
                        );
                    }
                    self.function_pointers
                        .write()
                        .unwrap()
                        .insert(*func_id, ptr as usize);

                    // Register source info for stack traces (debug mode only)
                    if let Some(register) = register_fn {
                        let name = function.qualified_name.as_deref().unwrap_or(&function.name);
                        let source_file = &module.source_file;
                        register(
                            func_id.0,
                            ptr as usize,
                            name.as_ptr(),
                            name.len(),
                            source_file.as_ptr(),
                            source_file.len(),
                            function.source_location.line,
                            function.source_location.column,
                        );
                    }
                }
            }
        }
        drop(modules);

        // Keep the backend alive by storing it (replace the old baseline_backend)
        *self.baseline_backend.lock().unwrap() = backend;

        // Refresh backend-dependent startup state now that compiled function pointers exist.
        self.rerun_loaded_modules_startup_jit()?;

        if self.config.verbosity >= 1 {
            debug!("[TieredBackend] JIT compilation complete");
        }

        Ok(())
    }

    /// Instrument all MIR functions with shadow call stack push/pop instructions.
    /// In debug mode, this adds `rayzor_push_call_frame(func_id)` at each function's
    /// entry block and `rayzor_pop_call_frame()` before every return terminator.
    /// This enables source-mapped stack traces by maintaining a thread-local shadow stack.
    fn instrument_modules_for_stack_traces(modules: &mut [IrModule]) {
        use crate::ir::blocks::{IrBasicBlock, IrBlockId, IrTerminator};
        use crate::ir::functions::{IrFunctionSignature, IrParameter};
        use crate::ir::modules::IrExternFunction;
        use crate::ir::types::{IrType, IrValue};
        use crate::ir::{CallingConvention, IrId, OwnershipMode};
        use crate::tast::id_types::SymbolId;

        for module in modules.iter_mut() {
            // Allocate function IDs for the two extern functions in this module
            let push_fn_id = module.alloc_function_id();
            let pop_fn_id = module.alloc_function_id();

            // Register rayzor_push_call_frame(func_id: u32) -> void
            module.add_extern_function(IrExternFunction {
                id: push_fn_id,
                name: "rayzor_push_call_frame".to_string(),
                symbol_id: SymbolId::from_raw(u32::MAX - 10),
                signature: IrFunctionSignature {
                    parameters: vec![IrParameter {
                        name: "func_id".to_string(),
                        ty: IrType::U32,
                        reg: IrId::new(0),
                        by_ref: false,
                    }],
                    return_type: IrType::Void,
                    calling_convention: CallingConvention::C,
                    can_throw: false,
                    type_params: vec![],
                    uses_sret: false,
                },
                source: "runtime".to_string(),
            });

            // Register rayzor_pop_call_frame() -> void
            module.add_extern_function(IrExternFunction {
                id: pop_fn_id,
                name: "rayzor_pop_call_frame".to_string(),
                symbol_id: SymbolId::from_raw(u32::MAX - 11),
                signature: IrFunctionSignature {
                    parameters: vec![],
                    return_type: IrType::Void,
                    calling_convention: CallingConvention::C,
                    can_throw: false,
                    type_params: vec![],
                    uses_sret: false,
                },
                source: "runtime".to_string(),
            });

            // Collect function IDs that need instrumentation (non-extern functions with bodies)
            let func_ids: Vec<(crate::ir::IrFunctionId, u32)> = module
                .functions
                .iter()
                .filter(|(_, f)| !f.cfg.blocks.is_empty())
                .map(|(id, _)| (*id, id.0))
                .collect();

            for (func_id, raw_id) in func_ids {
                let function = match module.functions.get_mut(&func_id) {
                    Some(f) => f,
                    None => continue,
                };

                // Allocate registers for the push call
                let const_reg = IrId::new(function.next_reg_id);
                function.next_reg_id += 1;

                // Prepend to entry block: Const(reg, U32(func_id)) + CallDirect(push, [reg])
                let entry_block_id = function.cfg.entry_block;
                if let Some(entry_block) = function.cfg.get_block_mut(entry_block_id) {
                    let push_instructions = vec![
                        IrInstruction::Const {
                            dest: const_reg,
                            value: IrValue::U32(raw_id),
                        },
                        IrInstruction::CallDirect {
                            dest: None,
                            func_id: push_fn_id,
                            args: vec![const_reg],
                            arg_ownership: vec![OwnershipMode::Copy],
                            type_args: vec![],
                            is_tail_call: false,
                        },
                    ];

                    // Insert at the beginning of the entry block
                    let mut new_instructions = push_instructions;
                    new_instructions.append(&mut entry_block.instructions);
                    entry_block.instructions = new_instructions;
                }

                // For every block with a Return terminator, insert pop before it
                let return_block_ids: Vec<IrBlockId> = function
                    .cfg
                    .blocks
                    .iter()
                    .filter(|(_, block)| matches!(block.terminator, IrTerminator::Return { .. }))
                    .map(|(id, _)| *id)
                    .collect();

                for block_id in return_block_ids {
                    if let Some(block) = function.cfg.get_block_mut(block_id) {
                        block.instructions.push(IrInstruction::CallDirect {
                            dest: None,
                            func_id: pop_fn_id,
                            args: vec![],
                            arg_ownership: vec![],
                            type_args: vec![],
                            is_tail_call: false,
                        });
                    }
                }
            }
        }
    }

    /// Strip `rayzor_update_call_frame_location` calls from MIR when stack traces are disabled.
    ///
    /// Uses the shared MIR-level utility so behavior is consistent across backends.
    fn strip_stack_trace_updates(modules: &mut [IrModule]) {
        for module in modules.iter_mut() {
            let _ = crate::ir::optimization::strip_stack_trace_updates(module);
        }
    }

    /// Static version of source info registration for background worker thread.
    fn register_source_info_static(
        modules: &Arc<RwLock<Vec<IrModule>>>,
        pointers: &BTreeMap<IrFunctionId, usize>,
        runtime_symbols: &Arc<Vec<(String, usize)>>,
    ) {
        let register_fn: Option<
            extern "C" fn(u32, usize, *const u8, usize, *const u8, usize, u32, u32),
        > = runtime_symbols
            .iter()
            .find(|(name, _)| name == "rayzor_register_function_source")
            .map(|(_, ptr)| unsafe { std::mem::transmute(*ptr) });

        let register = match register_fn {
            Some(f) => f,
            None => return,
        };

        let modules_lock = modules.read().unwrap();
        for module in modules_lock.iter() {
            for (func_id, function) in &module.functions {
                if function.cfg.blocks.is_empty() {
                    continue;
                }
                if let Some(&ptr) = pointers.get(func_id) {
                    let name = function.qualified_name.as_deref().unwrap_or(&function.name);
                    let source_file = &module.source_file;
                    register(
                        func_id.0,
                        ptr,
                        name.as_ptr(),
                        name.len(),
                        source_file.as_ptr(),
                        source_file.len(),
                        function.source_location.line,
                        function.source_location.column,
                    );
                }
            }
        }
    }

    /// Register source info for a set of function pointers (used by LLVM/AOT paths).
    /// Iterates all modules, matching func_id → pointer in the given map.
    fn register_source_info_for_pointers(&self, pointers: &BTreeMap<IrFunctionId, usize>) {
        if !self.config.enable_stack_traces {
            return;
        }

        let register_fn: Option<
            extern "C" fn(u32, usize, *const u8, usize, *const u8, usize, u32, u32),
        > = self
            .runtime_symbols
            .iter()
            .find(|(name, _)| name == "rayzor_register_function_source")
            .map(|(_, ptr)| unsafe { std::mem::transmute(*ptr) });

        let register = match register_fn {
            Some(f) => f,
            None => return,
        };

        let modules = self.modules.read().unwrap();
        for module in modules.iter() {
            for (func_id, function) in &module.functions {
                if function.cfg.blocks.is_empty() {
                    continue;
                }
                if let Some(&ptr) = pointers.get(func_id) {
                    let name = function.qualified_name.as_deref().unwrap_or(&function.name);
                    let source_file = &module.source_file;
                    register(
                        func_id.0,
                        ptr,
                        name.as_ptr(),
                        name.len(),
                        source_file.as_ptr(),
                        source_file.len(),
                        function.source_location.line,
                        function.source_location.column,
                    );
                }
            }
        }
    }

    /// Register enum RTTI from a single MIR module's type definitions.
    /// This ensures getName()/getParameters()/trace work correctly at runtime.
    fn register_enum_rtti_from_module(module: &IrModule) {
        use crate::ir::modules::IrTypeDefinition;
        use rayzor_runtime::type_system::{register_enum_from_mir, ParamType};

        for (_id, typedef) in &module.types {
            if let IrTypeDefinition::Enum { variants, .. } = &typedef.definition {
                let variant_data: Vec<(String, usize, Vec<ParamType>)> = variants
                    .iter()
                    .map(|v| {
                        let param_types: Vec<ParamType> = v
                            .fields
                            .iter()
                            .map(|f| CraneliftBackend::ir_type_to_param_type(&f.ty))
                            .collect();
                        (v.name.clone(), v.fields.len(), param_types)
                    })
                    .collect();

                register_enum_from_mir(typedef.type_id.0, &typedef.name, &variant_data);
            }
        }
    }

    /// Register class RTTI from a single MIR module's type definitions.
    fn register_class_rtti_from_module(module: &IrModule) {
        use crate::ir::modules::IrTypeDefinition;
        use rayzor_runtime::type_system::register_class_from_mir;

        // Map raw TypeId → deterministic registry id so the registered super
        // pointer matches the registry key (see cranelift_backend for rationale).
        let mut typeid_to_rtti: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        for (_id, typedef) in &module.types {
            if let IrTypeDefinition::Struct { .. } = &typedef.definition {
                let rtti = typedef
                    .runtime_type_id
                    .map(|h| h as u32)
                    .unwrap_or(typedef.type_id.0);
                typeid_to_rtti.insert(typedef.type_id.0, rtti);
            }
        }

        for (_id, typedef) in &module.types {
            if let IrTypeDefinition::Struct { fields, .. } = &typedef.definition {
                // Separate instance and static fields
                // In MIR, all fields are in the struct — we treat them all as instance fields
                // Static fields would need separate tracking (future work)
                // Skip synthetic object header field from user-visible RTTI field list.
                // Object slot 0 is always reserved for __type_id at runtime.
                let instance_fields: Vec<String> = fields
                    .iter()
                    .filter(|f| f.name != "__type_id")
                    .map(|f| f.name.clone())
                    .collect();
                let instance_field_types: Vec<rayzor_runtime::type_system::ParamType> = fields
                    .iter()
                    .filter(|f| f.name != "__type_id")
                    .map(|f| CraneliftBackend::ir_type_to_param_type(&f.ty))
                    .collect();
                let static_fields: Vec<String> = Vec::new();

                let super_type_id = typedef
                    .super_type_id
                    .map(|t| typeid_to_rtti.get(&t.0).copied().unwrap_or(t.0));

                // Prefer the deterministic runtime_type_id when present so
                // the RTTI registry key matches what the New handler stores
                // in object headers across import orders.
                let rtti_key = typedef
                    .runtime_type_id
                    .map(|h| h as u32)
                    .unwrap_or(typedef.type_id.0);
                register_class_from_mir(
                    rtti_key,
                    &typedef.name,
                    super_type_id,
                    &instance_fields,
                    &instance_field_types,
                    &static_fields,
                );
            }
        }
    }

    /// Process pending LLVM compilations on the main thread
    ///
    /// LLVM's add_global_mapping requires main thread, so background workers
    /// queue requests and this function processes them during execute_function calls.
    fn process_llvm_queue(&mut self) {
        // Check if there are any pending LLVM compilations
        let pending: Vec<IrFunctionId> = {
            let mut queue = self.llvm_queue.lock().unwrap();
            queue.drain(..).collect()
        };

        if pending.is_empty() {
            return;
        }

        emit_tier_event(
            "llvm_queue_start",
            None,
            Some(OptimizationTier::Maximum),
            &format!("pending={}", pending.len()),
        );
        if self.config.verbosity >= 1 {
            debug!(
                "[TieredBackend] Processing {} LLVM compilation(s) on main thread",
                pending.len()
            );
        }

        // Compile with LLVM (this will compile ALL modules and return ALL function pointers)
        #[cfg(feature = "llvm-backend")]
        {
            match self.compile_all_with_llvm() {
                Ok(all_pointers) => {
                    // Re-register source info at new LLVM addresses
                    self.register_source_info_for_pointers(&all_pointers);

                    // Install ALL compiled function pointers (not just pending ones)
                    let mut fp_lock = self.function_pointers.write().unwrap();
                    let mut ft_lock = self.function_tiers.write().unwrap();

                    let installed_count = all_pointers.len();
                    for (func_id, ptr) in all_pointers {
                        fp_lock.insert(func_id, ptr);
                        ft_lock.insert(func_id, OptimizationTier::Maximum);
                    }

                    if self.config.verbosity >= 1 {
                        debug!(
                            "[TieredBackend] LLVM: Installed {} functions at Maximum tier",
                            installed_count
                        );
                    }
                    emit_tier_event(
                        "llvm_queue_finish",
                        None,
                        Some(OptimizationTier::Maximum),
                        &format!("installed={}", installed_count),
                    );
                }
                Err(e) => {
                    if self.config.verbosity >= 1 {
                        debug!("[TieredBackend] LLVM compilation failed: {}", e);
                    }
                    emit_tier_event(
                        "llvm_queue_error",
                        None,
                        Some(OptimizationTier::Maximum),
                        &format!("error={}", sanitize_tier_event_detail(&e)),
                    );
                }
            }
        }

        #[cfg(not(feature = "llvm-backend"))]
        {
            if self.config.verbosity >= 1 {
                debug!(
                    "[TieredBackend] LLVM backend not enabled, skipping {} requests",
                    pending.len()
                );
            }
            emit_tier_event(
                "llvm_queue_skipped",
                None,
                Some(OptimizationTier::Maximum),
                &format!("pending={}", pending.len()),
            );
        }
    }

    /// Compile all modules with LLVM backend (Tier 4/Maximum)
    ///
    /// Note: This compiles ALL modules because functions may call other
    /// functions across modules. Returns ALL function pointers.
    ///
    /// Compile all functions with LLVM
    ///
    /// Platform-specific behavior:
    /// - x86_64 Linux: Uses AOT compile-to-dylib. MCJIT produces ~2.2x worse
    ///   codegen on x86_64 (1649ms vs 763ms for nbody). AOT is stable here:
    ///   no icache coherence issues (x86 is coherent), simple trampoline asm
    ///   (movabsq+jmp), and fs races mitigated by fsync+fence.
    /// - Other platforms (aarch64 macOS, etc.): Uses MCJIT directly (stable and fast).
    ///
    /// The compiled code is leaked to ensure it remains valid for program lifetime.

    /// Clone stored modules and tree-shake to only reachable functions.
    /// This prevents LLVM from compiling unused stdlib wrappers (Tensor, GPU, etc.)
    /// that are included in the module but never called by the program.
    #[cfg(feature = "llvm-backend")]
    fn tree_shake_modules_for_llvm(&self) -> Vec<IrModule> {
        let modules_lock = self.modules.read().unwrap();
        let mut modules: Vec<IrModule> = modules_lock.iter().cloned().collect();
        drop(modules_lock);

        // Find the entry (main) function for tree-shaking
        let entry = modules.iter().enumerate().find_map(|(idx, m)| {
            m.functions
                .values()
                .find(|f| f.name.ends_with("_main") || f.name == "main")
                .map(|f| (m.name.clone(), f.name.clone()))
        });

        if let Some((entry_module, entry_function)) = entry {
            let stats = crate::ir::tree_shake::tree_shake_bundle(
                &mut modules,
                &entry_module,
                &entry_function,
            );
            tracing::trace!(
                "[LLVM] Tree-shaking: removed {} functions, {} externs (kept {} functions)",
                stats.functions_removed,
                stats.extern_functions_removed,
                stats.functions_kept,
            );
        }

        modules
    }

    #[cfg(feature = "llvm-backend")]
    #[allow(dead_code)]
    fn compile_all_with_llvm(&self) -> Result<BTreeMap<IrFunctionId, usize>, String> {
        // Tiered `rayzor run` must stay in-process. The AOT-to-dylib path shells
        // out to the system linker, so keep it behind an explicit escape hatch.
        if Self::llvm_tier_aot_enabled() {
            return self.compile_all_with_llvm_aot();
        }

        self.compile_all_with_llvm_mcjit()
    }

    /// Compile with MCJIT (for x86_64 and Linux)
    ///
    /// MCJIT is stable on these platforms and provides the best JIT experience.
    #[cfg(feature = "llvm-backend")]
    #[allow(dead_code)]
    fn compile_all_with_llvm_mcjit(&self) -> Result<BTreeMap<IrFunctionId, usize>, String> {
        // Check if THIS instance has already compiled with LLVM
        {
            let llvm_compiled = self.llvm_compiled.lock().unwrap();
            if *llvm_compiled {
                let fp_lock = self.function_pointers.read().unwrap();
                return Ok(fp_lock.iter().map(|(id, ptr)| (*id, *ptr)).collect());
            }
        }

        // Check GLOBAL flag - if already compiled, reuse global pointers
        if super::llvm_jit_backend::is_llvm_compiled_globally() {
            if let Some(global_ptrs) = super::llvm_jit_backend::get_global_llvm_pointers() {
                return self.map_global_pointers_to_ids(&global_ptrs);
            }
            return Err("LLVM compilation already done but pointers not available.".to_string());
        }

        let _llvm_guard = super::llvm_jit_backend::llvm_lock();

        // Double-check after lock
        if super::llvm_jit_backend::is_llvm_compiled_globally() {
            if let Some(global_ptrs) = super::llvm_jit_backend::get_global_llvm_pointers() {
                return self.map_global_pointers_to_ids(&global_ptrs);
            }
            return Err("LLVM compilation already done (race).".to_string());
        }

        // Create and leak context for stable JIT code
        let context = Box::leak(Box::new(Context::create()));

        let symbols: Vec<(&str, *const u8)> = self
            .runtime_symbols
            .iter()
            .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
            .collect();

        let mut backend = LLVMJitBackend::with_symbols(context, &symbols)?;

        // Tree-shake to remove unreachable stdlib wrappers (Tensor, GPU, etc.)
        // before LLVM compilation. Without this, LLVM compiles all functions
        // including unused wrappers, which can cause heap corruption on Linux.
        let modules = self.tree_shake_modules_for_llvm();
        for module in &modules {
            backend.declare_module(module)?;
        }
        for module in &modules {
            backend.compile_module_bodies(module)?;
        }

        // Get function symbols before finalize (we need names for global storage)
        let function_symbols = backend.get_function_symbols();

        backend.finalize()?;

        // IMPORTANT: Do NOT use get_all_function_pointers() here!
        // That would trigger MCJIT to bulk-compile ALL functions at once, which causes
        // intermittent segfaults (~40% failure rate). Instead, only resolve the function
        // pointers that the tiered backend actually needs for top-level dispatch.
        // Inner function calls go through LLVM's compiled code directly.
        //
        // We only resolve pointers for functions that already exist in our function_pointers
        // map (i.e., functions that Cranelift compiled and the tiered backend calls externally).
        let needed_func_ids: Vec<IrFunctionId> = {
            let fp_lock = self.function_pointers.read().unwrap();
            fp_lock.keys().cloned().collect()
        };

        let mut resolved_pointers = BTreeMap::new();
        for func_id in &needed_func_ids {
            if let Some(ptr) = backend.get_function_pointer_by_id(*func_id) {
                resolved_pointers.insert(*func_id, ptr);
            }
        }

        let _leaked_backend = Box::leak(Box::new(backend));

        // Build name -> pointer map for global storage (only resolved functions)
        let global_ptrs: BTreeMap<String, usize> = function_symbols
            .iter()
            .filter_map(|(id, name)| resolved_pointers.get(id).map(|ptr| (name.clone(), *ptr)))
            .collect();

        *self.llvm_compiled.lock().unwrap() = true;
        super::llvm_jit_backend::mark_llvm_compiled_globally_with_pointers(global_ptrs);

        Ok(resolved_pointers)
    }

    /// Compile with AOT to dylib.
    ///
    /// This path invokes the system linker and is opt-in for tiered execution via
    /// RAYZOR_LLVM_TIER_AOT=1. Normal `rayzor run` uses MCJIT to avoid spawning
    /// clang/gcc/cc during generation.
    ///
    /// Flow:
    /// 1. LLVM compiles to object file (.o)
    /// 2. System linker creates dylib (.dylib)
    /// 3. libloading loads the dylib
    /// 4. Function pointers extracted via dlsym
    #[cfg(feature = "llvm-backend")]
    #[allow(dead_code)]
    fn compile_all_with_llvm_aot(&self) -> Result<BTreeMap<IrFunctionId, usize>, String> {
        if !Self::llvm_tier_aot_enabled() {
            return Err(
                "LLVM AOT tier is disabled. Set RAYZOR_LLVM_TIER_AOT=1 to allow the tiered \
                 backend to invoke the system linker."
                    .to_string(),
            );
        }

        // Check if THIS instance has already compiled with LLVM
        {
            let llvm_compiled = self.llvm_compiled.lock().unwrap();
            if *llvm_compiled {
                // Already compiled - return existing pointers
                let fp_lock = self.function_pointers.read().unwrap();
                return Ok(fp_lock.iter().map(|(id, ptr)| (*id, *ptr)).collect());
            }
        }

        // Check GLOBAL flag - if already compiled, reuse global pointers
        if super::llvm_jit_backend::is_llvm_compiled_globally() {
            // Another backend already compiled - reuse their pointers
            if let Some(global_ptrs) = super::llvm_jit_backend::get_global_llvm_pointers() {
                return self.map_global_pointers_to_ids(&global_ptrs);
            }
            return Err("LLVM compilation already done but pointers not available.".to_string());
        }

        // Acquire global LLVM lock
        let _llvm_guard = super::llvm_jit_backend::llvm_lock();

        // Double-check after acquiring lock
        if super::llvm_jit_backend::is_llvm_compiled_globally() {
            return Err(
                "LLVM compilation already done by another backend instance (race).".to_string(),
            );
        }

        // Create temporary paths for object file and dylib
        // Use timestamp + process ID + random suffix to ensure uniqueness
        let temp_dir = std::env::temp_dir();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pid = std::process::id();
        let random_suffix: u32 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let obj_path = temp_dir.join(format!(
            "rayzor_llvm_{}_{}_{}.o",
            timestamp, pid, random_suffix
        ));

        let dylib_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };
        let dylib_path = temp_dir.join(format!(
            "rayzor_llvm_{}_{}_{}.{}",
            timestamp, pid, random_suffix, dylib_ext
        ));

        // Create LLVM context and backend
        let context = Box::leak(Box::new(Context::create()));

        let symbols: Vec<(&str, *const u8)> = self
            .runtime_symbols
            .iter()
            .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
            .collect();

        let mut backend = LLVMJitBackend::with_symbols(context, &symbols)?;

        // Tree-shake to remove unreachable stdlib wrappers (Tensor, GPU, etc.)
        // before LLVM compilation. Without this, LLVM compiles all functions
        // including unused wrappers, which can cause heap corruption on Linux.
        let modules = self.tree_shake_modules_for_llvm();
        for module in &modules {
            backend.declare_module(module)?;
        }
        for module in &modules {
            backend.compile_module_bodies(module)?;
        }

        // Get function symbols before compiling (we need the names for dlsym)
        let function_symbols = backend.get_function_symbols();

        // Compile to object file (AOT, not JIT!)
        tracing::trace!("[LLVM:AOT] Compiling to object file: {:?}", obj_path);
        backend.compile_to_object_file(&obj_path)?;

        // Link object file to dylib via system linker, then dlopen
        tracing::trace!("[LLVM:AOT] Linking with system linker");
        let all_pointers =
            self.link_and_load_with_system_linker(&obj_path, &dylib_path, &function_symbols)?;
        let _ = std::fs::remove_file(&obj_path);

        // Build name -> pointer map for global storage (so other backends can reuse)
        let global_ptrs: BTreeMap<String, usize> = function_symbols
            .iter()
            .filter_map(|(id, name)| all_pointers.get(id).map(|ptr| (name.clone(), *ptr)))
            .collect();

        // Mark LLVM compilation as done and store pointers globally
        *self.llvm_compiled.lock().unwrap() = true;
        super::llvm_jit_backend::mark_llvm_compiled_globally_with_pointers(global_ptrs);

        Ok(all_pointers)
    }

    /// Map globally stored LLVM pointers (by name) to this backend's function IDs
    #[cfg(feature = "llvm-backend")]
    fn map_global_pointers_to_ids(
        &self,
        global_ptrs: &BTreeMap<String, usize>,
    ) -> Result<BTreeMap<IrFunctionId, usize>, String> {
        let modules_lock = self.modules.read().unwrap();
        let mut result = BTreeMap::new();

        for module in modules_lock.iter() {
            for (func_id, func) in &module.functions {
                // Try to find this function in global pointers by name
                let mangled_name = LLVMJitBackend::mangle_function_name(&func.name);
                if let Some(&ptr) = global_ptrs.get(&mangled_name) {
                    result.insert(*func_id, ptr);
                }
            }
        }

        tracing::trace!(
            "[LLVM:AOT] Mapped {} function pointers from global cache",
            result.len()
        );

        // Mark this instance as using LLVM pointers
        *self.llvm_compiled.lock().unwrap() = true;

        Ok(result)
    }

    /// Link object file to dynamic library using system linker
    ///
    /// Link object file to dylib using system linker, load via dlopen,
    /// and extract function pointers.
    #[cfg(feature = "llvm-backend")]
    fn link_and_load_with_system_linker(
        &self,
        obj_path: &Path,
        dylib_path: &Path,
        function_symbols: &BTreeMap<IrFunctionId, String>,
    ) -> Result<BTreeMap<IrFunctionId, usize>, String> {
        self.link_to_dylib(obj_path, dylib_path, &self.runtime_symbols)?;

        // Ensure dylib is fully visible on disk before loading.
        // On some systems, the linker output may not be immediately visible.
        #[cfg(unix)]
        {
            if let Ok(f) = std::fs::OpenOptions::new().read(true).open(dylib_path) {
                // fsync to ensure file metadata is on disk
                let _ = f.sync_all();
            }
        }

        let lib = unsafe {
            #[cfg(unix)]
            {
                use libloading::os::unix::Library as UnixLibrary;
                let unix_lib =
                    UnixLibrary::open(Some(dylib_path), libc::RTLD_NOW | libc::RTLD_GLOBAL)
                        .map_err(|e| format!("Failed to load dylib: {}", e))?;
                libloading::Library::from(unix_lib)
            }
            #[cfg(not(unix))]
            {
                libloading::Library::new(dylib_path)
                    .map_err(|e| format!("Failed to load dylib: {}", e))?
            }
        };

        // Memory barrier to ensure all writes from dylib loading are visible.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);

        let mut all_pointers = BTreeMap::new();
        for (func_id, symbol_name) in function_symbols {
            let symbol_result: Result<libloading::Symbol<*const ()>, _> =
                unsafe { lib.get(symbol_name.as_bytes()) };
            if let Ok(symbol) = symbol_result {
                let ptr = *symbol as usize;
                if ptr != 0 {
                    all_pointers.insert(*func_id, ptr);
                }
            }
        }

        // On ARM64 macOS, explicitly invalidate instruction cache after loading JIT code.
        // This is critical for Apple Silicon where I-cache and D-cache are separate.
        // The dlopen call should handle this, but we do it again as a safety measure
        // for any code that we'll be calling via function pointers.
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            extern "C" {
                fn sys_icache_invalidate(start: *const std::ffi::c_void, size: usize);
            }
            // Invalidate instruction cache for each function entry point.
            // We use a 64KB range per function as a safe upper bound for typical
            // compiled function sizes. This ensures the icache is coherent
            // when we start executing the JIT-compiled code.
            const FUNC_SIZE_ESTIMATE: usize = 64 * 1024;
            for &ptr in all_pointers.values() {
                if ptr != 0 {
                    unsafe {
                        sys_icache_invalidate(ptr as *const std::ffi::c_void, FUNC_SIZE_ESTIMATE);
                    }
                }
            }
            // Additional memory barrier after icache invalidation
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        }

        tracing::trace!(
            "[LLVM:AOT] Loaded {} function pointers from dylib",
            all_pointers.len()
        );

        // Leak the library to keep code valid
        Box::leak(Box::new(lib));

        Ok(all_pointers)
    }

    /// On macOS: Uses clang (via cc wrapper)
    /// On Linux: Uses gcc or clang
    ///
    /// Returns error if no linker is available - caller should fall back to Cranelift
    #[cfg(feature = "llvm-backend")]
    fn link_to_dylib(
        &self,
        obj_path: &Path,
        dylib_path: &Path,
        runtime_symbols: &[(String, usize)],
    ) -> Result<(), String> {
        // Try to find a suitable linker
        let linker = Self::find_linker()?;

        // Generate trampoline stubs that jump to absolute runtime function addresses.
        // Unlike `.set` aliases, trampolines generate real executable code that works
        // correctly with PIC/PIE shared libraries on native x86_64 and ARM64.
        let stubs_asm_path = obj_path.with_extension("stubs.s");
        let stubs_obj_path = obj_path.with_extension("stubs.o");
        {
            let mut asm = String::new();
            asm.push_str(".text\n");
            for (name, addr) in runtime_symbols {
                // On macOS, C symbols have a leading underscore
                #[cfg(target_os = "macos")]
                let sym = format!("_{}", name);
                #[cfg(not(target_os = "macos"))]
                let sym = name.to_string();

                asm.push_str(&format!(".globl {}\n", sym));
                asm.push_str(&format!("{}:\n", sym));

                // Generate architecture-specific trampoline
                #[cfg(target_arch = "x86_64")]
                {
                    // Use Intel syntax for clarity - movabs loads 64-bit immediate, then indirect jump
                    // The .quad stores the absolute address, which we load and jump to.
                    // This avoids any ambiguity with AT&T indirect addressing syntax.
                    asm.push_str(&format!("  movabsq $0x{:x}, %rax\n", addr));
                    asm.push_str("  jmp *%rax\n");
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // Load 64-bit address into x16 (IP0 scratch register), then branch
                    let a = *addr;
                    asm.push_str(&format!("  movz x16, #0x{:x}\n", a & 0xFFFF));
                    asm.push_str(&format!(
                        "  movk x16, #0x{:x}, lsl #16\n",
                        (a >> 16) & 0xFFFF
                    ));
                    asm.push_str(&format!(
                        "  movk x16, #0x{:x}, lsl #32\n",
                        (a >> 32) & 0xFFFF
                    ));
                    asm.push_str(&format!(
                        "  movk x16, #0x{:x}, lsl #48\n",
                        (a >> 48) & 0xFFFF
                    ));
                    asm.push_str("  br x16\n");
                }
            }
            std::fs::write(&stubs_asm_path, &asm)
                .map_err(|e| format!("Failed to write stubs asm: {}", e))?;

            let output = std::process::Command::new(&linker)
                .args([
                    "-c",
                    stubs_asm_path.to_str().unwrap(),
                    "-o",
                    stubs_obj_path.to_str().unwrap(),
                ])
                .output()
                .map_err(|e| format!("Failed to assemble stubs: {}", e))?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("Stubs assembly failed: {}", stderr));
            }
        }

        // Build linker arguments based on platform
        #[cfg(target_os = "macos")]
        let args = vec![
            "-shared".to_string(),
            "-o".to_string(),
            dylib_path.to_str().ok_or("Invalid dylib path")?.to_string(),
            obj_path.to_str().ok_or("Invalid object path")?.to_string(),
            stubs_obj_path
                .to_str()
                .ok_or("Invalid stubs path")?
                .to_string(),
        ];

        #[cfg(target_os = "linux")]
        let args = vec![
            "-shared".to_string(),
            "-fPIC".to_string(),
            "-o".to_string(),
            dylib_path.to_str().ok_or("Invalid dylib path")?.to_string(),
            obj_path.to_str().ok_or("Invalid object path")?.to_string(),
            stubs_obj_path
                .to_str()
                .ok_or("Invalid stubs path")?
                .to_string(),
        ];

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        return Err("LLVM AOT compilation not supported on this platform".to_string());

        let output = std::process::Command::new(&linker)
            .args(&args)
            .output()
            .map_err(|e| format!("Failed to run linker '{}': {}", linker, e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Linker failed: {}", stderr));
        }

        // Clean up temp files
        let _ = std::fs::remove_file(&stubs_asm_path);
        let _ = std::fs::remove_file(&stubs_obj_path);

        Ok(())
    }

    /// Find a suitable linker on the system
    ///
    /// Searches for: clang, gcc, cc (in order of preference)
    /// Returns the path to the linker or an error if none found
    #[cfg(feature = "llvm-backend")]
    fn find_linker() -> Result<String, String> {
        // Check for linkers in order of preference
        let candidates = ["clang", "gcc", "cc"];

        for linker in &candidates {
            if let Ok(output) = std::process::Command::new(linker).arg("--version").output() {
                if output.status.success() {
                    return Ok(linker.to_string());
                }
            }
        }

        Err("No C compiler/linker found (tried: clang, gcc, cc). \
             LLVM tier 3 optimization requires a system linker. \
             Install Xcode (macOS) or gcc/clang (Linux), or use precompiled bundles."
            .to_string())
    }

    /// Check if LLVM AOT compilation is available on this system
    #[cfg(feature = "llvm-backend")]
    pub fn is_llvm_aot_available() -> bool {
        if !Self::llvm_tier_aot_enabled() {
            return false;
        }
        Self::find_linker().is_ok()
    }

    #[cfg(feature = "llvm-backend")]
    fn llvm_tier_aot_enabled() -> bool {
        match std::env::var("RAYZOR_LLVM_TIER_AOT") {
            Ok(value) => {
                let value = value.trim();
                !value.is_empty()
                    && value != "0"
                    && !value.eq_ignore_ascii_case("false")
                    && !value.eq_ignore_ascii_case("off")
            }
            // Keep AOT opt-in. The normal LLVM tier should exercise MCJIT so
            // beadie/tier promotion failures are visible instead of being
            // hidden behind a system linker hop.
            Err(_) => false,
        }
    }

    /// Legacy single-function compile (returns just one pointer)
    #[cfg(feature = "llvm-backend")]
    #[allow(dead_code)]
    fn compile_with_llvm(&self, func_id: IrFunctionId) -> Result<usize, String> {
        let all_pointers = self.compile_all_with_llvm()?;
        all_pointers
            .get(&func_id)
            .copied()
            .ok_or_else(|| format!("Function {:?} not found in LLVM compiled module", func_id))
    }

    /// Compile function with LLVM backend (Tier 3) - stub when LLVM not enabled
    #[cfg(not(feature = "llvm-backend"))]
    fn compile_with_llvm(&self, func_id: IrFunctionId) -> Result<usize, String> {
        if self.config.verbosity >= 1 {
            debug!(
                "[TieredBackend] LLVM backend not enabled, cannot compile {:?} at Tier 3",
                func_id
            );
        }
        Err("LLVM backend not enabled. Compile with --features llvm-backend".to_string())
    }
}

/// Statistics about the tiered backend.
///
/// Step 6 dropped the `queued_for_optimization` /
/// `currently_optimizing` fields — beadie owns the
/// Standard/Optimized scheduling; there's no rayzor-side queue to
/// report on. Use [`TieredBackend::beadie_stats`] for beadie's
/// per-tier counters.
#[derive(Debug, Clone)]
pub struct TieredStatistics {
    pub profile_stats: ProfileStatistics,
    pub interpreted_functions: usize,
    pub baseline_functions: usize,
    pub standard_functions: usize,
    pub optimized_functions: usize,
    pub llvm_functions: usize,
}

impl TieredStatistics {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Tiered Compilation: {} Interpreted (P0), {} Baseline (P1), {} Standard (P2), {} Optimized (P3), {} LLVM (P4)\n\
             {}",
            self.interpreted_functions,
            self.baseline_functions,
            self.standard_functions,
            self.optimized_functions,
            self.llvm_functions,
            self.profile_stats.format()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::mir_builder::MirBuilder;
    use crate::ir::{BinaryOp, IrType};

    fn build_add_module() -> (IrModule, IrFunctionId) {
        let mut builder = MirBuilder::new("tiered_threshold_test");
        let func_id = builder
            .begin_function("add")
            .param("a", IrType::I64)
            .param("b", IrType::I64)
            .returns(IrType::I64)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let a = builder.get_param(0);
        let b = builder.get_param(1);
        let sum = builder.bin_op(BinaryOp::Add, a, b);
        builder.ret(Some(sum));
        (builder.finish(), func_id)
    }

    fn build_two_add_module() -> (IrModule, IrFunctionId, IrFunctionId) {
        let mut builder = MirBuilder::new("tiered_pre_promote_test");

        let first_id = builder
            .begin_function("add_a")
            .param("a", IrType::I64)
            .param("b", IrType::I64)
            .returns(IrType::I64)
            .build();
        builder.set_current_function(first_id);
        let first_entry = builder.create_block("entry");
        builder.set_insert_point(first_entry);
        let a = builder.get_param(0);
        let b = builder.get_param(1);
        let sum = builder.bin_op(BinaryOp::Add, a, b);
        builder.ret(Some(sum));

        let second_id = builder
            .begin_function("add_b")
            .param("a", IrType::I64)
            .param("b", IrType::I64)
            .returns(IrType::I64)
            .build();
        builder.set_current_function(second_id);
        let second_entry = builder.create_block("entry");
        builder.set_insert_point(second_entry);
        let a = builder.get_param(0);
        let b = builder.get_param(1);
        let sum = builder.bin_op(BinaryOp::Add, a, b);
        builder.ret(Some(sum));

        (builder.finish(), first_id, second_id)
    }

    #[test]
    fn interpreter_threshold_promotes_to_baseline() {
        let mut cfg = TieredConfig::default();
        cfg.start_interpreted = true;
        cfg.enable_tier_promotion = true;
        cfg.enable_stack_traces = false;
        cfg.profile_config.interpreter_threshold = 1;
        cfg.profile_config.warm_threshold = u64::MAX;
        cfg.profile_config.hot_threshold = u64::MAX;
        cfg.profile_config.blazing_threshold = u64::MAX;

        let mut backend = TieredBackend::new(cfg).expect("tiered backend");
        let (module, func_id) = build_add_module();
        backend.compile_module(module).expect("module loaded");

        assert_eq!(
            backend.function_tier(func_id),
            Some(OptimizationTier::Interpreted)
        );

        let result = backend
            .execute_function(func_id, vec![InterpValue::I64(19), InterpValue::I64(23)])
            .expect("execute add");
        match result {
            InterpValue::I64(n) => assert_eq!(n, 42),
            other => panic!("unexpected result: {:?}", other),
        }

        assert_eq!(
            backend.function_tier(func_id),
            Some(OptimizationTier::Baseline)
        );
        assert!(
            backend.jit_pointer(func_id).is_some(),
            "threshold promotion should compile a Baseline pointer"
        );

        let stats = backend.get_statistics();
        assert_eq!(stats.interpreted_functions, 0);
        assert_eq!(stats.baseline_functions, 1);
    }

    #[test]
    fn sampled_record_call_still_counts_every_call() {
        let mut cfg = TieredConfig::default();
        cfg.profile_config.sample_rate = 10;
        cfg.profile_config.interpreter_threshold = u64::MAX;
        cfg.profile_config.warm_threshold = u64::MAX;
        cfg.profile_config.hot_threshold = u64::MAX;
        cfg.profile_config.blazing_threshold = u64::MAX;

        let backend = TieredBackend::new(cfg).expect("tiered backend");
        let func_id = IrFunctionId(crate::tast::SymbolId(99).into());

        for _ in 0..3 {
            backend.record_call(func_id);
        }

        assert_eq!(backend.get_statistics().profile_stats.total_executions, 3);
    }

    #[test]
    fn record_call_respects_disabled_tier_promotion() {
        let mut cfg = TieredConfig::default();
        cfg.enable_tier_promotion = false;
        cfg.profile_config.sample_rate = 1;
        cfg.profile_config.interpreter_threshold = 0;
        cfg.profile_config.warm_threshold = 1;
        cfg.profile_config.hot_threshold = 1;
        cfg.profile_config.blazing_threshold = u64::MAX;

        let backend = TieredBackend::new(cfg).expect("tiered backend");
        let func_id = IrFunctionId(99);

        for _ in 0..3 {
            backend.record_call(func_id);
        }

        assert_eq!(backend.get_statistics().profile_stats.total_executions, 3);
        let beadie = backend.beadie_stats();
        assert_eq!(beadie.routes_attempted, 0);
        assert_eq!(beadie.standard_routes_attempted, 0);
        assert_eq!(beadie.optimized_routes_attempted, 0);
        assert_eq!(beadie.installs, 0);
        assert_eq!(backend.function_tier(func_id), None);
    }

    #[test]
    fn pre_promoted_baseline_compile_updates_compiled_tiers() {
        let mut cfg = TieredConfig::default();
        cfg.start_interpreted = true;
        cfg.enable_tier_promotion = true;
        cfg.enable_stack_traces = false;
        cfg.profile_config.interpreter_threshold = u64::MAX;
        cfg.profile_config.warm_threshold = u64::MAX;
        cfg.profile_config.hot_threshold = u64::MAX;
        cfg.profile_config.blazing_threshold = u64::MAX;

        let mut backend = TieredBackend::new(cfg).expect("tiered backend");
        let (module, first_id, second_id) = build_two_add_module();
        backend.compile_module(module).expect("module loaded");

        backend
            .function_tiers
            .write()
            .unwrap()
            .insert(first_id, OptimizationTier::Baseline);
        assert_eq!(
            backend.function_tier(second_id),
            Some(OptimizationTier::Interpreted)
        );

        let result = backend
            .execute_function(first_id, vec![InterpValue::I64(10), InterpValue::I64(32)])
            .expect("execute pre-promoted add");
        match result {
            InterpValue::I64(n) => assert_eq!(n, 42),
            other => panic!("unexpected result: {:?}", other),
        }

        assert_eq!(
            backend.function_tier(second_id),
            Some(OptimizationTier::Baseline)
        );
    }
}
