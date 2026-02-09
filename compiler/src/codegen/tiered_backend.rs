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

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::time::Duration;

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
    function_tiers: Arc<RwLock<HashMap<IrFunctionId, OptimizationTier>>>,

    /// Function pointers (usize for thread safety, cast to function type when needed)
    function_pointers: Arc<RwLock<HashMap<IrFunctionId, usize>>>,

    /// Queue of functions waiting for recompilation at higher tier
    optimization_queue: Arc<Mutex<VecDeque<(IrFunctionId, OptimizationTier)>>>,

    /// Functions currently being optimized (prevents duplicate work)
    optimizing: Arc<Mutex<HashSet<IrFunctionId>>>,

    /// The MIR modules (needed for recompilation and interpretation)
    /// Multiple modules may be loaded (e.g., user code + stdlib)
    modules: Arc<RwLock<Vec<IrModule>>>,

    /// Configuration
    config: TieredConfig,

    /// Background optimization worker handle
    worker_handle: Option<thread::JoinHandle<()>>,

    /// Shutdown signal for background worker
    shutdown: Arc<Mutex<bool>>,

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

    /// Number of Cranelift tier promotions performed (each leaks a backend)
    promotion_count: Arc<AtomicU64>,

    /// The highest Cranelift tier currently compiled for all functions
    /// Used to skip redundant recompilations when multiple functions
    /// cross thresholds at the same tier level
    current_compiled_tier: Arc<AtomicU8>,
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
}

/// Interpreter bailout strategy - determines how quickly to switch from interpreter to JIT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

impl TierPreset {
    /// Convert preset to a TieredConfig
    pub fn to_config(self) -> TieredConfig {
        match self {
            TierPreset::Script => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 5, // Quick promotion from interpreter
                    warm_threshold: 50,       // Don't optimize too aggressively
                    hot_threshold: 200,
                    blazing_threshold: u64::MAX, // No LLVM for short scripts
                    sample_rate: 1,
                },
                enable_background_optimization: false, // Sync optimization for predictability
                optimization_check_interval_ms: 50,
                max_parallel_optimizations: 2,
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Quick,
                max_tier_promotions: 4,
            },

            TierPreset::Application => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 10,
                    warm_threshold: 100,
                    hot_threshold: 500,
                    blazing_threshold: 2000, // LLVM for sustained hot code
                    sample_rate: 1,
                },
                enable_background_optimization: true,
                optimization_check_interval_ms: 100,
                max_parallel_optimizations: 4,
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Quick,
                max_tier_promotions: 10,
            },

            TierPreset::Server => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 5,
                    warm_threshold: 50,
                    hot_threshold: 200,
                    blazing_threshold: 500, // Lower LLVM threshold for servers
                    sample_rate: 1,
                },
                enable_background_optimization: true,
                optimization_check_interval_ms: 50,
                max_parallel_optimizations: 8, // More parallel compilation
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Immediate,
                max_tier_promotions: 15,
            },

            TierPreset::Benchmark => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: 2,
                    warm_threshold: 3,
                    hot_threshold: 5,
                    blazing_threshold: u64::MAX, // Manual LLVM upgrade after warmup
                    sample_rate: 1,
                },
                enable_background_optimization: false, // Sync for deterministic results
                optimization_check_interval_ms: 1,
                max_parallel_optimizations: 4,
                verbosity: 1,            // Show tier transitions
                start_interpreted: true, // Start with interpreter for instant startup
                bailout_strategy: BailoutStrategy::Immediate,
                max_tier_promotions: 8,
            },

            TierPreset::Development => TieredConfig {
                profile_config: ProfileConfig::development(),
                enable_background_optimization: true,
                optimization_check_interval_ms: 50,
                max_parallel_optimizations: 2,
                verbosity: 2, // Detailed logging
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Immediate,
                max_tier_promotions: 6,
            },

            TierPreset::Embedded => TieredConfig {
                profile_config: ProfileConfig {
                    interpreter_threshold: u64::MAX, // Never promote
                    warm_threshold: u64::MAX,
                    hot_threshold: u64::MAX,
                    blazing_threshold: u64::MAX,
                    sample_rate: u64::MAX, // Disable profiling
                },
                enable_background_optimization: false,
                optimization_check_interval_ms: u64::MAX,
                max_parallel_optimizations: 0,
                verbosity: 0,
                start_interpreted: true,
                bailout_strategy: BailoutStrategy::Slow, // High threshold before bailout
                max_tier_promotions: 0,                  // Interpreter only
            },
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
        }
    }
}

impl std::fmt::Display for TierPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Configuration for tiered compilation
#[derive(Debug, Clone)]
pub struct TieredConfig {
    /// Profiling configuration
    pub profile_config: ProfileConfig,

    /// Enable background optimization (async optimization in separate thread)
    pub enable_background_optimization: bool,

    /// How often to check for hot functions (in milliseconds)
    pub optimization_check_interval_ms: u64,

    /// Maximum number of functions to optimize in parallel
    pub max_parallel_optimizations: usize,

    /// Verbosity level (0 = silent, 1 = basic, 2 = detailed)
    pub verbosity: u8,

    /// Start in interpreted mode (Phase 0) for instant startup
    /// If false, functions are compiled to Baseline (Phase 1) immediately
    pub start_interpreted: bool,

    /// Interpreter bailout strategy - determines how quickly to switch to JIT
    /// Use BailoutStrategy::Immediate for benchmarks, Quick for most apps
    pub bailout_strategy: BailoutStrategy,

    /// Maximum number of Cranelift tier promotions (each leaks a backend).
    /// Once exhausted, further Cranelift promotions are skipped (functions stay at current tier).
    /// LLVM promotion is not counted since it has its own singleton guard.
    /// Set to 0 to disable tier promotion entirely.
    pub max_tier_promotions: u64,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            profile_config: ProfileConfig::default(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 100,
            max_parallel_optimizations: 4,
            verbosity: 0,
            start_interpreted: true, // Enable interpreter by default for instant startup
            bailout_strategy: BailoutStrategy::Quick, // Good balance for most apps
            max_tier_promotions: 10,
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

    /// Development configuration (aggressive optimization, verbose)
    pub fn development() -> Self {
        Self {
            profile_config: ProfileConfig::development(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 50,
            max_parallel_optimizations: 2,
            verbosity: 2,
            start_interpreted: true, // Instant startup for quick iteration
            bailout_strategy: BailoutStrategy::Immediate, // Quick bailout for testing
            max_tier_promotions: 6,
        }
    }

    /// Production configuration (conservative, low overhead)
    pub fn production() -> Self {
        Self {
            profile_config: ProfileConfig::production(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 1000,
            max_parallel_optimizations: 8,
            verbosity: 0,
            start_interpreted: true, // Instant startup, then promote hot functions
            bailout_strategy: BailoutStrategy::Quick, // Quick bailout
            max_tier_promotions: 10,
        }
    }

    /// JIT-only configuration (skip interpreter, compile immediately)
    /// Use when startup time is less important than consistent performance
    pub fn jit_only() -> Self {
        Self {
            profile_config: ProfileConfig::default(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 100,
            max_parallel_optimizations: 4,
            verbosity: 0,
            start_interpreted: false, // Skip interpreter, start at Phase 1
            bailout_strategy: BailoutStrategy::Quick, // Not used when start_interpreted=false
            max_tier_promotions: 10,
        }
    }
}

impl TieredBackend {
    /// Create a tiered backend from a preset
    ///
    /// # Example
    /// ```
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
    /// ```
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
        interp.set_max_iterations(config.bailout_strategy.threshold());

        Ok(Self {
            interpreter: Arc::new(Mutex::new(interp)),
            baseline_backend: Arc::new(Mutex::new(baseline_backend)),
            profile_data,
            function_tiers: Arc::new(RwLock::new(HashMap::new())),
            function_pointers: Arc::new(RwLock::new(HashMap::new())),
            optimization_queue: Arc::new(Mutex::new(VecDeque::new())),
            optimizing: Arc::new(Mutex::new(HashSet::new())),
            modules: Arc::new(RwLock::new(Vec::new())),
            config,
            worker_handle: None,
            shutdown: Arc::new(Mutex::new(false)),
            start_interpreted,
            runtime_symbols: Arc::new(Vec::new()),
            llvm_queue: Arc::new(Mutex::new(VecDeque::new())),
            #[cfg(feature = "llvm-backend")]
            llvm_compiled: Arc::new(Mutex::new(false)),
            promotion_barrier: Arc::new(PromotionBarrier::new()),
            promotion_count: Arc::new(AtomicU64::new(0)),
            current_compiled_tier: Arc::new(AtomicU8::new(0)),
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
        interp.set_max_iterations(config.bailout_strategy.threshold());
        for (name, ptr) in symbols {
            interp.register_symbol(name, *ptr);
        }

        Ok(Self {
            interpreter: Arc::new(Mutex::new(interp)),
            baseline_backend: Arc::new(Mutex::new(baseline_backend)),
            profile_data,
            function_tiers: Arc::new(RwLock::new(HashMap::new())),
            function_pointers: Arc::new(RwLock::new(HashMap::new())),
            optimization_queue: Arc::new(Mutex::new(VecDeque::new())),
            optimizing: Arc::new(Mutex::new(HashSet::new())),
            modules: Arc::new(RwLock::new(Vec::new())),
            config,
            worker_handle: None,
            shutdown: Arc::new(Mutex::new(false)),
            start_interpreted,
            runtime_symbols: Arc::new(runtime_symbols),
            llvm_queue: Arc::new(Mutex::new(VecDeque::new())),
            #[cfg(feature = "llvm-backend")]
            llvm_compiled: Arc::new(Mutex::new(false)),
            promotion_barrier: Arc::new(PromotionBarrier::new()),
            promotion_count: Arc::new(AtomicU64::new(0)),
            current_compiled_tier: Arc::new(AtomicU8::new(0)),
        })
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
                    | IrInstruction::VectorMinMax { .. } => return true,
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

        // Start background optimization if enabled
        // NOTE: In JIT mode (start_interpreted=false), defer starting until after
        // compile_all_modules_jit() completes in execute_function(), otherwise the
        // background worker will try to recompile functions that haven't been compiled yet.
        if self.config.enable_background_optimization && self.start_interpreted {
            self.start_background_optimization();
        }

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
        // Record the call for profiling
        self.record_call(func_id);

        // Process LLVM queue (main thread compilation)
        // This is safe because execute_function runs on the main thread
        self.process_llvm_queue();

        // Get current tier
        let tier = self
            .function_tiers
            .read()
            .unwrap()
            .get(&func_id)
            .copied()
            .unwrap_or(OptimizationTier::Interpreted);

        // JIT mode: lazily compile all modules on first execution if not yet compiled
        if !self.start_interpreted && tier == OptimizationTier::Baseline {
            // Check if we need to compile (no function pointers yet)
            let needs_compile = self.function_pointers.read().unwrap().is_empty();
            if needs_compile {
                self.compile_all_modules_jit()?;
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
                    {
                        let fp_lock = self.function_pointers.read().unwrap();
                        let mut tiers = self.function_tiers.write().unwrap();
                        for func_id in fp_lock.keys() {
                            tiers.insert(*func_id, OptimizationTier::Baseline);
                        }
                    }

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
    #[cfg(feature = "llvm-backend")]
    pub fn upgrade_to_llvm(&mut self) -> Result<(), String> {
        if self.config.verbosity >= 1 {
            debug!("[TieredBackend] Upgrading all functions to LLVM...");
        }

        // Stop background optimization to prevent race conditions during LLVM compilation
        // The background worker might be in the middle of Cranelift compilation
        *self.shutdown.lock().unwrap() = true;
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }

        // Wait for any ongoing optimizations to complete
        loop {
            let optimizing_count = self.optimizing.lock().unwrap().len();
            if optimizing_count == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        match self.compile_all_with_llvm() {
            Ok(all_pointers) => {
                let mut fp_lock = self.function_pointers.write().unwrap();
                let mut ft_lock = self.function_tiers.write().unwrap();

                let count = all_pointers.len();
                for (func_id, ptr) in all_pointers {
                    fp_lock.insert(func_id, ptr);
                    ft_lock.insert(func_id, OptimizationTier::Maximum);
                }

                if self.config.verbosity >= 1 {
                    debug!("[TieredBackend] Upgraded {} functions to LLVM", count);
                }
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
        // Sample based on config to reduce overhead
        let count = self.profile_data.get_function_count(func_id);
        if count % self.profile_data.config().sample_rate != 0 {
            return;
        }

        self.profile_data.record_function_call(func_id);

        // Check if function should be promoted to a higher tier
        // Use count-based promotion that allows skipping tiers if count exceeds multiple thresholds
        let should_promote = {
            let tiers = self.function_tiers.read().unwrap();
            let current_tier = tiers
                .get(&func_id)
                .copied()
                .unwrap_or(OptimizationTier::Interpreted);

            let count = self.profile_data.get_function_count(func_id);
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
            self.enqueue_for_optimization(func_id, target_tier);
        }
    }

    /// Process the optimization queue synchronously (for when background optimization is disabled)
    /// Returns the number of functions optimized
    pub fn process_queue_sync(&mut self) -> usize {
        let mut optimized = 0;

        loop {
            // Take one item from the queue
            let item = {
                let mut queue = self.optimization_queue.lock().unwrap();
                queue.pop_front()
            };

            match item {
                Some((func_id, target_tier)) => {
                    // Skip LLVM tiers in sync mode (too slow)
                    if target_tier.uses_llvm() {
                        continue;
                    }

                    // Optimize the function
                    if let Err(e) = self.optimize_function_internal(func_id, target_tier) {
                        if self.config.verbosity >= 1 {
                            debug!(
                                "[TieredBackend] Sync optimization failed for {:?}: {}",
                                func_id, e
                            );
                        }
                    } else {
                        optimized += 1;
                    }
                }
                None => break, // Queue is empty
            }
        }

        optimized
    }

    /// Enqueue a function for optimization at a specific tier
    fn enqueue_for_optimization(&self, func_id: IrFunctionId, target_tier: OptimizationTier) {
        let mut queue = self.optimization_queue.lock().unwrap();
        let optimizing = self.optimizing.lock().unwrap();

        // Don't enqueue if already optimizing or already in queue at this tier
        if !optimizing.contains(&func_id)
            && !queue
                .iter()
                .any(|(id, tier)| *id == func_id && *tier == target_tier)
        {
            let count = self.profile_data.get_function_count(func_id);
            if self.config.verbosity >= 2 {
                debug!(
                    "[TieredBackend] Enqueuing {:?} for {} (count: {})",
                    func_id,
                    target_tier.description(),
                    count
                );
            }
            queue.push_back((func_id, target_tier));
        }
    }

    /// Manually trigger recompilation of a function at a specific tier
    pub fn optimize_function(
        &mut self,
        func_id: IrFunctionId,
        target_tier: OptimizationTier,
    ) -> Result<(), String> {
        self.optimize_function_internal(func_id, target_tier)
    }

    /// Internal: Recompile a single function at a specific tier
    fn optimize_function_internal(
        &mut self,
        func_id: IrFunctionId,
        target_tier: OptimizationTier,
    ) -> Result<(), String> {
        if self.config.verbosity >= 1 {
            let count = self.profile_data.get_function_count(func_id);
            debug!(
                "[TieredBackend] Recompiling {:?} at {} (count: {})",
                func_id,
                target_tier.description(),
                count
            );
        }

        // Verify the function exists
        let modules_lock = self.modules.read().unwrap();
        if !modules_lock
            .iter()
            .any(|m| m.functions.contains_key(&func_id))
        {
            return Err(format!("Function {:?} not found in any module", func_id));
        }

        // Choose backend based on tier
        // For tier promotion, we recompile ALL modules at the new tier because
        // functions may call each other across modules
        let all_pointers = if target_tier.uses_llvm() {
            // Tier 4 (Maximum): Use LLVM backend - compiles all modules
            drop(modules_lock); // Release lock before heavy work
            let ptr = self.compile_with_llvm(func_id)?;
            // LLVM compile_with_llvm returns a single pointer, but we need all
            // For now, just return the requested function
            let mut map = HashMap::new();
            map.insert(func_id, ptr);
            map
        } else {
            // Tier 1-3: Use Cranelift backend
            // Skip if already at this tier (dedup)
            let current_tier = self.current_compiled_tier.load(Ordering::Relaxed);
            if target_tier as u8 <= current_tier {
                drop(modules_lock);
                return Ok(());
            }

            // Check promotion budget
            let count = self.promotion_count.fetch_add(1, Ordering::Relaxed);
            if count >= self.config.max_tier_promotions {
                self.promotion_count.fetch_sub(1, Ordering::Relaxed);
                drop(modules_lock);
                return Ok(());
            }

            // Compile ALL modules at the new tier and get ALL function pointers
            let pointers = self.compile_all_at_tier(&modules_lock, target_tier)?;
            self.current_compiled_tier
                .store(target_tier as u8, Ordering::Relaxed);
            drop(modules_lock);
            pointers
        };

        // Atomically swap ALL function pointers from the new compilation
        {
            let mut fp_lock = self.function_pointers.write().unwrap();
            let mut ft_lock = self.function_tiers.write().unwrap();

            for (fid, ptr) in all_pointers {
                fp_lock.insert(fid, ptr);
                ft_lock.insert(fid, target_tier);
            }
        }

        if self.config.verbosity >= 1 {
            debug!(
                "[TieredBackend] Successfully recompiled {:?} at {}",
                func_id,
                target_tier.description()
            );
        }

        Ok(())
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

        // Register enum RTTI from MIR type definitions so that
        // getName()/getParameters()/trace work correctly at runtime
        for module in modules.iter() {
            Self::register_enum_rtti_from_module(module);
        }

        // Finalize all modules at once (must be done before getting function pointers)
        backend.finalize()?;

        // Store function pointers for functions with bodies (non-extern)
        // Extern functions have empty CFGs and are imported, not compiled
        for module in modules.iter() {
            for (func_id, function) in &module.functions {
                // Skip extern functions (no body to compile)
                if function.cfg.blocks.is_empty() {
                    continue;
                }
                if let Ok(ptr) = backend.get_function_ptr(*func_id) {
                    self.function_pointers
                        .write()
                        .unwrap()
                        .insert(*func_id, ptr as usize);
                }
            }
        }
        drop(modules);

        // Keep the backend alive by storing it (replace the old baseline_backend)
        *self.baseline_backend.lock().unwrap() = backend;

        if self.config.verbosity >= 1 {
            debug!("[TieredBackend] JIT compilation complete");
        }

        // Now start background optimization (was deferred in JIT mode)
        if self.config.enable_background_optimization {
            self.start_background_optimization();
        }

        Ok(())
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

    /// Compile ALL modules with Cranelift backend at the specified tier
    ///
    /// This method recompiles ALL modules at the target optimization tier and returns
    /// ALL function pointers. This is the correct approach for tier promotion because
    /// functions may call each other across modules.
    ///
    /// Returns: HashMap of (func_id -> function pointer) for all compiled functions
    fn compile_all_at_tier(
        &self,
        all_modules: &[IrModule],
        target_tier: OptimizationTier,
    ) -> Result<HashMap<IrFunctionId, usize>, String> {
        use crate::ir::optimization::PassManager;

        // Convert runtime symbols to the format Cranelift expects
        let symbols: Vec<(&str, *const u8)> = self
            .runtime_symbols
            .iter()
            .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
            .collect();

        // Create a new Cranelift backend with the target optimization level and runtime symbols
        let mut backend =
            CraneliftBackend::with_symbols_and_opt(target_tier.cranelift_opt_level(), &symbols)?;

        // Apply MIR-level optimizations for higher tiers
        let mir_opt_level = target_tier.mir_opt_level();
        let optimized_modules: Vec<IrModule>;
        let modules_to_compile: &[IrModule] =
            if mir_opt_level != crate::ir::optimization::OptimizationLevel::O0 {
                // Clone all modules and apply MIR optimizations
                optimized_modules = all_modules
                    .iter()
                    .map(|m| {
                        let mut module = m.clone();
                        let mut pass_manager = PassManager::for_level(mir_opt_level);
                        let _ = pass_manager.run(&mut module);
                        module
                    })
                    .collect();
                &optimized_modules
            } else {
                all_modules
            };

        // Compile all modules to the same backend WITHOUT finalizing between modules
        for module in modules_to_compile {
            backend.compile_module_without_finalize(module)?;
        }

        // Finalize all modules at once
        backend.finalize()?;

        // Collect function pointers for all functions with bodies
        let mut pointers = HashMap::new();
        for module in modules_to_compile {
            for (func_id, function) in &module.functions {
                // Skip extern functions (no body to compile)
                if function.cfg.blocks.is_empty() {
                    continue;
                }
                if let Ok(ptr) = backend.get_function_ptr(*func_id) {
                    pointers.insert(*func_id, ptr as usize);
                }
            }
        }

        // Leak the backend to keep the compiled code alive
        // This is necessary because the JIT code must remain valid for the program's lifetime
        Box::leak(Box::new(backend));

        Ok(pointers)
    }

    /// Apply MIR-level optimizations to a function
    fn apply_mir_optimizations(
        function: IrFunction,
        level: crate::ir::optimization::OptimizationLevel,
    ) -> IrFunction {
        use crate::ir::optimization::PassManager;

        // Create a temporary module containing just this function
        let mut temp_module = IrModule::new("temp_opt".to_string(), "temp".to_string());

        // Use a temporary function ID
        let temp_id = IrFunctionId(0);
        temp_module.functions.insert(temp_id, function);

        // Run optimization passes
        let mut pass_manager = PassManager::for_level(level);
        let _ = pass_manager.run(&mut temp_module);

        // Extract the optimized function
        temp_module.functions.remove(&temp_id).unwrap()
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
                }
                Err(e) => {
                    if self.config.verbosity >= 1 {
                        debug!("[TieredBackend] LLVM compilation failed: {}", e);
                    }
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
    fn compile_all_with_llvm(&self) -> Result<HashMap<IrFunctionId, usize>, String> {
        // On x86_64 Linux, use AOT-to-dylib for ~2x better codegen quality.
        // MCJIT's code generator on x86_64 produces significantly worse code
        // than the AOT path using the same LLVM IR and optimization passes.
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            return self.compile_all_with_llvm_aot();
        }

        // On all other platforms, use MCJIT (stable and performant).
        #[allow(unreachable_code)]
        self.compile_all_with_llvm_mcjit()
    }

    /// Compile with MCJIT (for x86_64 and Linux)
    ///
    /// MCJIT is stable on these platforms and provides the best JIT experience.
    #[cfg(feature = "llvm-backend")]
    #[allow(dead_code)]
    fn compile_all_with_llvm_mcjit(&self) -> Result<HashMap<IrFunctionId, usize>, String> {
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

        let mut resolved_pointers = HashMap::new();
        for func_id in &needed_func_ids {
            if let Some(ptr) = backend.get_function_pointer_by_id(*func_id) {
                resolved_pointers.insert(*func_id, ptr);
            }
        }

        let _leaked_backend = Box::leak(Box::new(backend));

        // Build name -> pointer map for global storage (only resolved functions)
        let global_ptrs: HashMap<String, usize> = function_symbols
            .iter()
            .filter_map(|(id, name)| resolved_pointers.get(id).map(|ptr| (name.clone(), *ptr)))
            .collect();

        *self.llvm_compiled.lock().unwrap() = true;
        super::llvm_jit_backend::mark_llvm_compiled_globally_with_pointers(global_ptrs);

        Ok(resolved_pointers)
    }

    /// Compile with AOT to dylib (for Apple Silicon)
    ///
    /// This avoids MCJIT's MAP_JIT issues on Apple Silicon by:
    /// 1. LLVM compiles to object file (.o)
    /// 2. System linker creates dylib (.dylib)
    /// 3. libloading loads the dylib
    /// 4. Function pointers extracted via dlsym
    #[cfg(feature = "llvm-backend")]
    #[allow(dead_code)]
    fn compile_all_with_llvm_aot(&self) -> Result<HashMap<IrFunctionId, usize>, String> {
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
        let global_ptrs: HashMap<String, usize> = function_symbols
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
        global_ptrs: &HashMap<String, usize>,
    ) -> Result<HashMap<IrFunctionId, usize>, String> {
        let modules_lock = self.modules.read().unwrap();
        let mut result = HashMap::new();

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
        function_symbols: &HashMap<IrFunctionId, String>,
    ) -> Result<HashMap<IrFunctionId, usize>, String> {
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

        let mut all_pointers = HashMap::new();
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
        Self::find_linker().is_ok()
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

    /// Start background optimization worker thread
    fn start_background_optimization(&mut self) {
        if self.worker_handle.is_some() {
            return; // Already started
        }

        let queue = Arc::clone(&self.optimization_queue);
        let optimizing = Arc::clone(&self.optimizing);
        let modules = Arc::clone(&self.modules);
        let function_pointers = Arc::clone(&self.function_pointers);
        let function_tiers = Arc::clone(&self.function_tiers);
        let shutdown = Arc::clone(&self.shutdown);
        let profile_data = self.profile_data.clone();
        let config = self.config.clone();
        let runtime_symbols = Arc::clone(&self.runtime_symbols);
        let llvm_queue = Arc::clone(&self.llvm_queue);
        let promotion_barrier = Arc::clone(&self.promotion_barrier);
        let promotion_count = Arc::clone(&self.promotion_count);
        let current_compiled_tier = Arc::clone(&self.current_compiled_tier);

        let handle = thread::spawn(move || {
            if config.verbosity >= 1 {
                debug!("[TieredBackend] Background optimization worker started");
            }

            loop {
                // Check for shutdown
                if *shutdown.lock().unwrap() {
                    if config.verbosity >= 1 {
                        debug!("[TieredBackend] Background worker shutting down");
                    }
                    break;
                }

                // Process optimization queue
                Self::background_worker_iteration(
                    &queue,
                    &optimizing,
                    &modules,
                    &function_pointers,
                    &function_tiers,
                    &profile_data,
                    &config,
                    &runtime_symbols,
                    &llvm_queue,
                    &promotion_barrier,
                    &promotion_count,
                    &current_compiled_tier,
                );

                // Sleep before next iteration
                thread::sleep(Duration::from_millis(config.optimization_check_interval_ms));
            }
        });

        self.worker_handle = Some(handle);
    }

    /// Background worker iteration - processes multiple functions in parallel using rayon
    ///
    /// This drains up to `max_parallel_optimizations` functions from the queue and
    /// compiles them concurrently using rayon's parallel iterators.
    ///
    /// ## Safe Promotion Protocol (Barrier-Based)
    /// 1. Compile new code in background (no barrier needed)
    /// 2. Request promotion via barrier (blocks new JIT executions)
    /// 3. Wait for all in-flight JIT executions to complete
    /// 4. Atomically swap ALL function pointers
    /// 5. Release barrier (allows JIT executions to resume)
    fn background_worker_iteration(
        queue: &Arc<Mutex<VecDeque<(IrFunctionId, OptimizationTier)>>>,
        optimizing: &Arc<Mutex<HashSet<IrFunctionId>>>,
        modules: &Arc<RwLock<Vec<IrModule>>>,
        function_pointers: &Arc<RwLock<HashMap<IrFunctionId, usize>>>,
        function_tiers: &Arc<RwLock<HashMap<IrFunctionId, OptimizationTier>>>,
        profile_data: &ProfileData,
        config: &TieredConfig,
        runtime_symbols: &Arc<Vec<(String, usize)>>,
        llvm_queue: &Arc<Mutex<VecDeque<IrFunctionId>>>,
        promotion_barrier: &Arc<PromotionBarrier>,
        promotion_count: &Arc<AtomicU64>,
        current_compiled_tier: &Arc<AtomicU8>,
    ) {
        // Drain batch of functions to compile in parallel
        let batch: Vec<(IrFunctionId, OptimizationTier)> = {
            let mut queue_lock = queue.lock().unwrap();
            let mut optimizing_lock = optimizing.lock().unwrap();

            // Calculate how many functions we can compile in parallel
            let available_slots = config
                .max_parallel_optimizations
                .saturating_sub(optimizing_lock.len());
            if available_slots == 0 {
                return;
            }

            // Drain up to available_slots functions from the queue
            let mut batch = Vec::with_capacity(available_slots);
            while batch.len() < available_slots {
                if let Some((func_id, target_tier)) = queue_lock.pop_front() {
                    optimizing_lock.insert(func_id);
                    batch.push((func_id, target_tier));
                } else {
                    break;
                }
            }
            batch
        };

        if batch.is_empty() {
            return;
        }

        // Get modules reference (read lock held during parallel compilation)
        let modules_lock = modules.read().unwrap();
        if modules_lock.is_empty() {
            // No modules, mark all as done and return
            let mut optimizing_lock = optimizing.lock().unwrap();
            for (func_id, _) in &batch {
                optimizing_lock.remove(func_id);
            }
            return;
        }

        // Separate LLVM and Cranelift compilations
        // LLVM requires main thread for symbol mapping, so queue those for main thread
        let (llvm_batch, cranelift_batch): (Vec<_>, Vec<_>) = batch
            .iter()
            .cloned() // Clone to get owned values
            .partition(|(_, tier)| tier.uses_llvm());

        // Queue LLVM requests for main thread compilation (instead of downgrading)
        // The main thread will process these during execute_function calls
        if !llvm_batch.is_empty() {
            let mut llvm_queue_lock = llvm_queue.lock().unwrap();
            let mut optimizing_lock = optimizing.lock().unwrap();
            for (func_id, _) in llvm_batch {
                // Remove from "optimizing" set since we're re-queuing for main thread
                optimizing_lock.remove(&func_id);
                // Add to LLVM queue for main thread
                if !llvm_queue_lock.iter().any(|id| *id == func_id) {
                    llvm_queue_lock.push_back(func_id);
                    if config.verbosity >= 1 {
                        debug!(
                            "[TieredBackend] Queued {:?} for LLVM compilation on main thread",
                            func_id
                        );
                    }
                }
            }
        }

        // For Cranelift tier promotion, we need to compile ALL modules at the target tier
        // Group by target tier to minimize recompilation
        if !cranelift_batch.is_empty() {
            // Find the highest tier requested in this batch
            let max_tier = cranelift_batch
                .iter()
                .map(|(_, tier)| *tier)
                .max()
                .unwrap_or(OptimizationTier::Baseline);

            if config.verbosity >= 1 {
                debug!(
                    "[TieredBackend] Background: compiling {} functions at {} tier",
                    cranelift_batch.len(),
                    max_tier.description()
                );
            }

            // Skip if we've already compiled at this tier or higher (dedup)
            let current_tier = current_compiled_tier.load(Ordering::Relaxed);
            if max_tier as u8 <= current_tier {
                if config.verbosity >= 2 {
                    debug!(
                        "[TieredBackend] Skipping recompilation: already at tier {} (requested {})",
                        current_tier, max_tier as u8
                    );
                }
                // Mark batch items as no longer optimizing
                let mut optimizing_lock = optimizing.lock().unwrap();
                for (func_id, _) in &cranelift_batch {
                    optimizing_lock.remove(func_id);
                }
                return;
            }

            // Check promotion budget
            let count = promotion_count.fetch_add(1, Ordering::Relaxed);
            if count >= config.max_tier_promotions {
                promotion_count.fetch_sub(1, Ordering::Relaxed);
                if config.verbosity >= 1 {
                    debug!(
                        "[TieredBackend] Tier promotion budget exhausted ({}/{}), skipping",
                        count, config.max_tier_promotions
                    );
                }
                let mut optimizing_lock = optimizing.lock().unwrap();
                for (func_id, _) in &cranelift_batch {
                    optimizing_lock.remove(func_id);
                }
                return;
            }

            // Compile ALL modules at the highest tier
            let compile_result =
                Self::compile_all_at_tier_static(&modules_lock[..], max_tier, runtime_symbols);

            // Drop modules lock before installing results
            drop(modules_lock);

            // Install results using the barrier-based safe promotion protocol
            match compile_result {
                Ok(all_pointers) => {
                    // Step 1: Request promotion - blocks new JIT executions
                    promotion_barrier.request_promotion();

                    if config.verbosity >= 2 {
                        debug!("[TieredBackend] Promotion requested, waiting for in-flight executions to drain");
                    }

                    // Step 2: Wait for all in-flight executions to complete (with timeout)
                    let drain_timeout = Duration::from_secs(5);
                    if !promotion_barrier.wait_for_drain(drain_timeout) {
                        // Timeout - cancel promotion and skip this batch
                        if config.verbosity >= 1 {
                            debug!(
                                "[TieredBackend] Promotion timed out waiting for drain, cancelling"
                            );
                        }
                        promotion_barrier.cancel_promotion();

                        // Mark batch items as no longer optimizing so they can be retried
                        let mut optimizing_lock = optimizing.lock().unwrap();
                        for (func_id, _) in &cranelift_batch {
                            optimizing_lock.remove(func_id);
                        }
                        return;
                    }

                    // Step 3: All executions drained - safe to install pointers atomically
                    {
                        let mut fp_lock = function_pointers.write().unwrap();
                        let mut ft_lock = function_tiers.write().unwrap();
                        let mut optimizing_lock = optimizing.lock().unwrap();

                        // Install ALL function pointers from the new compilation
                        let installed_count = all_pointers.len();
                        for (func_id, ptr) in all_pointers {
                            fp_lock.insert(func_id, ptr);
                            ft_lock.insert(func_id, max_tier);
                        }

                        // Mark all batch items as no longer optimizing
                        for (func_id, _) in &cranelift_batch {
                            optimizing_lock.remove(func_id);
                        }

                        if config.verbosity >= 1 {
                            debug!(
                                "[TieredBackend] Installed {} functions at {}",
                                installed_count,
                                max_tier.description()
                            );
                        }
                    }

                    // Step 4: Complete promotion - allow JIT executions to resume
                    promotion_barrier.complete_promotion();

                    // Track the tier we just compiled to (for dedup)
                    current_compiled_tier.store(max_tier as u8, Ordering::Relaxed);

                    if config.verbosity >= 2 {
                        debug!("[TieredBackend] Promotion complete, executions resumed");
                    }
                }
                Err(e) => {
                    if config.verbosity >= 1 {
                        debug!("[TieredBackend] Background compilation failed: {}", e);
                    }

                    // Mark all batch items as no longer optimizing
                    let mut optimizing_lock = optimizing.lock().unwrap();
                    for (func_id, _) in &cranelift_batch {
                        optimizing_lock.remove(func_id);
                    }
                }
            }
        } else {
            // No Cranelift batch, just drop the lock
            drop(modules_lock);
        }
    }

    /// Static version of compile_all_at_tier for use in worker thread
    ///
    /// Compiles ALL modules at the specified tier and returns ALL function pointers.
    /// This is necessary because functions may call each other across modules.
    fn compile_all_at_tier_static(
        all_modules: &[IrModule],
        target_tier: OptimizationTier,
        runtime_symbols: &Arc<Vec<(String, usize)>>,
    ) -> Result<HashMap<IrFunctionId, usize>, String> {
        use crate::ir::optimization::PassManager;

        // Convert runtime symbols to format expected by Cranelift
        let symbols: Vec<(&str, *const u8)> = runtime_symbols
            .iter()
            .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
            .collect();

        let mut backend =
            CraneliftBackend::with_symbols_and_opt(target_tier.cranelift_opt_level(), &symbols)?;

        // Apply MIR-level optimizations for higher tiers
        let mir_opt_level = target_tier.mir_opt_level();
        let optimized_modules: Vec<IrModule>;
        let modules_to_compile: &[IrModule] =
            if mir_opt_level != crate::ir::optimization::OptimizationLevel::O0 {
                // Clone all modules and apply MIR optimizations
                optimized_modules = all_modules
                    .iter()
                    .map(|m| {
                        let mut module = m.clone();
                        let mut pass_manager = PassManager::for_level(mir_opt_level);
                        let _ = pass_manager.run(&mut module);
                        module
                    })
                    .collect();
                &optimized_modules
            } else {
                all_modules
            };

        // Compile all modules to the same backend WITHOUT finalizing between modules
        for module in modules_to_compile {
            backend.compile_module_without_finalize(module)?;
        }

        // Finalize all modules at once
        backend.finalize()?;

        // Collect function pointers for all functions with bodies
        let mut pointers = HashMap::new();
        for module in modules_to_compile {
            for (func_id, function) in &module.functions {
                // Skip extern functions (no body to compile)
                if function.cfg.blocks.is_empty() {
                    continue;
                }
                if let Ok(ptr) = backend.get_function_ptr(*func_id) {
                    pointers.insert(*func_id, ptr as usize);
                }
            }
        }

        // Leak the backend to keep the compiled code alive
        Box::leak(Box::new(backend));

        Ok(pointers)
    }

    /// Static version of compile_with_llvm for use in worker thread
    ///
    /// Note: This intentionally leaks the LLVM context and backend to ensure
    /// JIT-compiled code remains valid for the program's lifetime.
    ///
    /// This compiles ALL modules because functions may call other functions
    /// across modules. The function pointer for the requested function is returned.
    #[cfg(feature = "llvm-backend")]
    #[allow(unused)] // Reserved for future main-thread LLVM compilation
    fn compile_with_llvm_static(
        func_id: IrFunctionId,
        modules: &[IrModule],
        runtime_symbols: &Arc<Vec<(String, usize)>>,
    ) -> Result<usize, String> {
        // Acquire global LLVM lock - LLVM is not thread-safe
        let _llvm_guard = super::llvm_jit_backend::llvm_lock();

        // Create context and backend, then leak them to ensure lifetime
        // This is intentional: JIT code must remain valid indefinitely
        let context = Box::leak(Box::new(Context::create()));

        // Convert symbols back to the format LLVMJitBackend expects
        let symbols: Vec<(&str, *const u8)> = runtime_symbols
            .iter()
            .map(|(name, ptr)| (name.as_str(), *ptr as *const u8))
            .collect();

        let mut backend = LLVMJitBackend::with_symbols(context, &symbols)?;

        // Compile ALL modules - functions may call across modules
        for module in modules {
            backend.compile_module(module)?;
        }

        // Finalize the module to create the execution engine
        backend.finalize()?;

        // Get the function pointer for the requested function
        let ptr = backend.get_function_ptr(func_id)?;

        // Leak the backend to keep the execution engine alive
        Box::leak(Box::new(backend));

        Ok(ptr as usize)
    }

    /// Static version of compile_with_llvm - stub when LLVM not enabled
    #[cfg(not(feature = "llvm-backend"))]
    fn compile_with_llvm_static(
        func_id: IrFunctionId,
        _modules: &[IrModule],
        _runtime_symbols: &Arc<Vec<(String, usize)>>,
    ) -> Result<usize, String> {
        Err(format!(
            "LLVM backend not enabled, cannot compile {:?} at Tier 3. Compile with --features llvm-backend",
            func_id
        ))
    }

    /// Get profiling and tiering statistics
    pub fn get_statistics(&self) -> TieredStatistics {
        let profile_stats = self.profile_data.get_statistics();
        let tiers = self.function_tiers.read().unwrap();

        // Debug: Print what tiers we actually have
        if self.config.verbosity >= 2 {
            debug!("[TieredBackend] Current function tiers:");
            for (func_id, tier) in tiers.iter() {
                debug!("  {:?} -> {:?}", func_id, tier);
            }
        }

        let interpreted_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Interpreted)
            .count();
        let baseline_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Baseline)
            .count();
        let standard_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Standard)
            .count();
        let optimized_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Optimized)
            .count();
        let maximum_count = tiers
            .values()
            .filter(|&&t| t == OptimizationTier::Maximum)
            .count();

        TieredStatistics {
            profile_stats,
            interpreted_functions: interpreted_count,
            baseline_functions: baseline_count,
            standard_functions: standard_count,
            optimized_functions: optimized_count,
            llvm_functions: maximum_count,
            queued_for_optimization: self.optimization_queue.lock().unwrap().len(),
            currently_optimizing: self.optimizing.lock().unwrap().len(),
        }
    }

    /// Shutdown the tiered backend (stops background worker)
    pub fn shutdown(&mut self) {
        *self.shutdown.lock().unwrap() = true;

        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TieredBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Statistics about the tiered backend
#[derive(Debug, Clone)]
pub struct TieredStatistics {
    pub profile_stats: ProfileStatistics,
    pub interpreted_functions: usize,
    pub baseline_functions: usize,
    pub standard_functions: usize,
    pub optimized_functions: usize,
    pub llvm_functions: usize,
    pub queued_for_optimization: usize,
    pub currently_optimizing: usize,
}

impl TieredStatistics {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        format!(
            "Tiered Compilation: {} Interpreted (P0), {} Baseline (P1), {} Standard (P2), {} Optimized (P3), {} LLVM (P4)\n\
             Queue: {} waiting, {} optimizing\n\
             {}",
            self.interpreted_functions,
            self.baseline_functions,
            self.standard_functions,
            self.optimized_functions,
            self.llvm_functions,
            self.queued_for_optimization,
            self.currently_optimizing,
            self.profile_stats.format()
        )
    }
}
