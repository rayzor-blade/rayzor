//! # Runtime Profiling Infrastructure
//!
//! Provides execution counters and profiling data for tiered compilation.
//! Tracks function execution frequencies to identify hot code paths.
//!
//! ## Design
//! - Lock-free atomic counters for minimal runtime overhead
//! - Configurable thresholds for warm/hot detection
//! - Sample-based profiling to reduce overhead
//! - Per-function execution tracking

use crate::ir::IrFunctionId;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Runtime profiling data collector
#[derive(Clone)]
pub struct ProfileData {
    /// Per-function execution counters (lock-free atomic)
    function_counts: Arc<RwLock<BTreeMap<IrFunctionId, Arc<AtomicU64>>>>,

    /// Configuration for hotness detection
    config: ProfileConfig,
}

/// Configuration for profiling and hotness detection (5-tier system with interpreter)
#[derive(Debug, Clone, Copy)]
pub struct ProfileConfig {
    /// Number of executions before JIT compiling (Phase 0 -> Phase 1)
    /// When a function is called this many times in interpreter mode,
    /// it gets compiled to native code for better performance.
    pub interpreter_threshold: u64,

    /// Number of executions before considering function "warm" (eligible for Phase 2)
    pub warm_threshold: u64,

    /// Number of executions before considering function "hot" (eligible for Phase 3)
    pub hot_threshold: u64,

    /// Number of executions before considering function "blazing" (eligible for Phase 4/LLVM)
    pub blazing_threshold: u64,

    /// Sample rate (1 = profile every call, 10 = every 10th call, etc.)
    /// Higher values reduce overhead but lower accuracy
    pub sample_rate: u64,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            interpreter_threshold: 10, // JIT compile after 10 interpreted calls
            warm_threshold: 100,       // Promote to P2 after 100 calls
            hot_threshold: 1000,       // Promote to P3 after 1000 calls
            blazing_threshold: 10000,  // Promote to P4/LLVM after 10000 calls
            sample_rate: 1,            // Profile every call by default
        }
    }
}

impl ProfileConfig {
    /// Development configuration (aggressive optimization, low thresholds)
    pub fn development() -> Self {
        Self {
            interpreter_threshold: 3, // JIT quickly for testing
            warm_threshold: 10,       // Quickly promote for testing
            hot_threshold: 100,       // P3 after 100 calls
            blazing_threshold: 500,   // P4/LLVM after 500 calls
            sample_rate: 1,           // Profile every call for accuracy
        }
    }

    /// Production configuration (conservative, high thresholds, low overhead)
    pub fn production() -> Self {
        Self {
            interpreter_threshold: 50, // Stay interpreted longer for cold code
            warm_threshold: 1000,      // Wait longer before optimizing
            hot_threshold: 10000,      // P3 after 10k calls
            blazing_threshold: 100000, // P4/LLVM after 100k calls (truly hot!)
            sample_rate: 10,           // Sample 1/10 calls to reduce overhead
        }
    }
}

impl ProfileData {
    /// Create a new profiling data collector
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            function_counts: Arc::new(RwLock::new(BTreeMap::new())),
            config,
        }
    }

    /// Record a function execution (called from generated code or runtime)
    pub fn record_function_call(&self, func_id: IrFunctionId) {
        let mut counts = self.function_counts.write().unwrap();
        let counter = counts
            .entry(func_id)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Get execution count for a function
    pub fn get_function_count(&self, func_id: IrFunctionId) -> u64 {
        let counts = self.function_counts.read().unwrap();
        counts
            .get(&func_id)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Check if a function should be JIT compiled (executed enough in interpreter)
    /// This promotes from Phase 0 (Interpreted) to Phase 1 (Baseline JIT)
    pub fn should_jit_compile(&self, func_id: IrFunctionId) -> bool {
        let count = self.get_function_count(func_id);
        count >= self.config.interpreter_threshold && count < self.config.warm_threshold
    }

    /// Check if a function is warm (executed moderately, eligible for Phase 2)
    pub fn is_warm(&self, func_id: IrFunctionId) -> bool {
        let count = self.get_function_count(func_id);
        count >= self.config.warm_threshold && count < self.config.hot_threshold
    }

    /// Check if a function is hot (executed frequently, eligible for Tier 2)
    pub fn is_hot(&self, func_id: IrFunctionId) -> bool {
        let count = self.get_function_count(func_id);
        count >= self.config.hot_threshold && count < self.config.blazing_threshold
    }

    /// Check if a function is blazing (ultra-hot, eligible for Tier 3/LLVM)
    pub fn is_blazing(&self, func_id: IrFunctionId) -> bool {
        self.get_function_count(func_id) >= self.config.blazing_threshold
    }

    /// Get the hotness level of a function
    pub fn get_hotness(&self, func_id: IrFunctionId) -> HotnessLevel {
        let count = self.get_function_count(func_id);

        if count >= self.config.blazing_threshold {
            HotnessLevel::Blazing
        } else if count >= self.config.hot_threshold {
            HotnessLevel::Hot
        } else if count >= self.config.warm_threshold {
            HotnessLevel::Warm
        } else if count >= self.config.interpreter_threshold {
            HotnessLevel::Cold
        } else {
            HotnessLevel::Interpreted
        }
    }

    /// Get all hot functions (sorted by execution count, descending)
    pub fn get_hot_functions(&self) -> Vec<(IrFunctionId, u64)> {
        let counts = self.function_counts.read().unwrap();
        let mut hot_funcs: Vec<_> = counts
            .iter()
            .map(|(id, counter)| (*id, counter.load(Ordering::Relaxed)))
            .filter(|(_, count)| *count >= self.config.hot_threshold)
            .collect();

        hot_funcs.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by count
        hot_funcs
    }

    /// Get all warm functions (sorted by execution count, descending)
    pub fn get_warm_functions(&self) -> Vec<(IrFunctionId, u64)> {
        let counts = self.function_counts.read().unwrap();
        let mut warm_funcs: Vec<_> = counts
            .iter()
            .map(|(id, counter)| (*id, counter.load(Ordering::Relaxed)))
            .filter(|(_, count)| {
                *count >= self.config.warm_threshold && *count < self.config.hot_threshold
            })
            .collect();

        warm_funcs.sort_by(|a, b| b.1.cmp(&a.1));
        warm_funcs
    }

    /// Reset all profiling counters (useful for testing)
    pub fn reset(&self) {
        let mut counts = self.function_counts.write().unwrap();
        counts.clear();
    }

    /// Get a function's counter reference for direct instrumentation
    /// This allows generated code to directly increment counters
    pub fn get_or_create_function_counter(&self, func_id: IrFunctionId) -> Arc<AtomicU64> {
        let mut counts = self.function_counts.write().unwrap();
        counts
            .entry(func_id)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    /// Get profiling statistics summary
    pub fn get_statistics(&self) -> ProfileStatistics {
        let func_counts = self.function_counts.read().unwrap();

        let total_functions = func_counts.len();
        let hot_count = func_counts
            .values()
            .filter(|c| c.load(Ordering::Relaxed) >= self.config.hot_threshold)
            .count();
        let warm_count = func_counts
            .values()
            .filter(|c| {
                let count = c.load(Ordering::Relaxed);
                count >= self.config.warm_threshold && count < self.config.hot_threshold
            })
            .count();
        let cold_count = total_functions - hot_count - warm_count;

        let total_executions: u64 = func_counts
            .values()
            .map(|c| c.load(Ordering::Relaxed))
            .sum();

        ProfileStatistics {
            total_functions,
            hot_functions: hot_count,
            warm_functions: warm_count,
            cold_functions: cold_count,
            total_executions,
        }
    }

    /// Get the profiling configuration
    pub fn config(&self) -> &ProfileConfig {
        &self.config
    }
}

/// Hotness level classification (5-tier system with interpreter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HotnessLevel {
    Interpreted, // Below interpreter threshold (Phase 0 - Interpreter)
    Cold,        // Below warm threshold (Phase 1 - Baseline JIT)
    Warm,        // Between warm and hot thresholds (Phase 2)
    Hot,         // Between hot and blazing thresholds (Phase 3)
    Blazing,     // Above blazing threshold (Phase 4/LLVM)
}

/// Profiling statistics summary
#[derive(Debug, Clone)]
pub struct ProfileStatistics {
    pub total_functions: usize,
    pub hot_functions: usize,
    pub warm_functions: usize,
    pub cold_functions: usize,
    pub total_executions: u64,
}

impl ProfileStatistics {
    /// Format as a human-readable string
    pub fn format(&self) -> String {
        format!(
            "Profile: {} functions ({} hot, {} warm, {} cold), {} total calls",
            self.total_functions,
            self.hot_functions,
            self.warm_functions,
            self.cold_functions,
            self.total_executions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tast::SymbolId;

    #[test]
    fn test_profile_data_basic() {
        let profile = ProfileData::new(ProfileConfig {
            interpreter_threshold: 3, // JIT after 3 calls
            warm_threshold: 5,
            hot_threshold: 10,
            blazing_threshold: 100,
            sample_rate: 1,
        });

        let func_id = IrFunctionId(SymbolId(42).into());

        // Initially interpreted (0 calls)
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Interpreted);
        assert_eq!(profile.get_function_count(func_id), 0);

        // Execute 3 times -> cold (eligible for JIT)
        for _ in 0..3 {
            profile.record_function_call(func_id);
        }
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Cold);
        assert!(profile.should_jit_compile(func_id));

        // Execute 4 more times (7 total) -> warm
        for _ in 0..4 {
            profile.record_function_call(func_id);
        }
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Warm);
        assert!(profile.is_warm(func_id));
        assert!(!profile.is_hot(func_id));

        // Execute 5 more times (12 total) -> hot
        for _ in 0..5 {
            profile.record_function_call(func_id);
        }
        assert_eq!(profile.get_hotness(func_id), HotnessLevel::Hot);
        assert!(!profile.is_warm(func_id));
        assert!(profile.is_hot(func_id));
        assert_eq!(profile.get_function_count(func_id), 12);
    }

    #[test]
    fn test_get_hot_functions() {
        let profile = ProfileData::new(ProfileConfig {
            interpreter_threshold: 2,
            warm_threshold: 5,
            hot_threshold: 10,
            blazing_threshold: 100,
            sample_rate: 1,
        });

        let func1 = IrFunctionId(SymbolId(1).into());
        let func2 = IrFunctionId(SymbolId(2).into());
        let func3 = IrFunctionId(SymbolId(3).into());

        // func1: cold (3 executions)
        for _ in 0..3 {
            profile.record_function_call(func1);
        }

        // func2: warm (7 executions)
        for _ in 0..7 {
            profile.record_function_call(func2);
        }

        // func3: hot (15 executions)
        for _ in 0..15 {
            profile.record_function_call(func3);
        }

        let hot_funcs = profile.get_hot_functions();
        assert_eq!(hot_funcs.len(), 1);
        assert_eq!(hot_funcs[0].0, func3);
        assert_eq!(hot_funcs[0].1, 15);

        let warm_funcs = profile.get_warm_functions();
        assert_eq!(warm_funcs.len(), 1);
        assert_eq!(warm_funcs[0].0, func2);
        assert_eq!(warm_funcs[0].1, 7);
    }

    #[test]
    fn test_statistics() {
        let profile = ProfileData::new(ProfileConfig::default());

        let func1 = IrFunctionId(SymbolId(10).into());
        let func2 = IrFunctionId(SymbolId(20).into());

        for _ in 0..50 {
            profile.record_function_call(func1);
        }

        for _ in 0..500 {
            profile.record_function_call(func2);
        }

        let stats = profile.get_statistics();
        assert_eq!(stats.total_functions, 2);
        assert_eq!(stats.cold_functions, 1); // func1 is cold (< 100)
        assert_eq!(stats.warm_functions, 1); // func2 is warm (>= 100, < 1000)
        assert_eq!(stats.hot_functions, 0);
        assert_eq!(stats.total_executions, 550);
    }

    #[test]
    fn test_atomic_counter_thread_safety() {
        let profile = ProfileData::new(ProfileConfig::default());
        let func_id = IrFunctionId(SymbolId(99).into());

        // Get counter and simulate concurrent increments
        let counter = profile.get_or_create_function_counter(func_id);

        // Simulate 100 concurrent calls
        for _ in 0..100 {
            counter.fetch_add(1, Ordering::Relaxed);
        }

        assert_eq!(profile.get_function_count(func_id), 100);
    }
}
