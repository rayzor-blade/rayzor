//! Per-kernel wall-time accumulators. When `RAYZOR_KERNEL_TIMING=1` is
//! set, every instrumented FFI entry wraps its body with `Instant::now()`
//! / `elapsed()` and adds to a static `KernelTimer` (atomic nanos + call
//! count). At process exit (atexit), all timers dump to stderr in
//! decreasing wall-time order.
//!
//! When the env var is NOT set, the macro path is a single
//! `AtomicU64::load(Relaxed)` + branch — zero clock work, no heap
//! allocation, no per-call cost worth measuring.
//!
//! Decoupled from the heavier `profile` feature (SIGPROF, alloc-graph)
//! because it is useful even on production builds and adds < 200 LOC.
//!
//! See `tools/profile_run.sh` for the SIGPROF-based alternative.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as MemOrdering};
use std::sync::Once;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

pub struct KernelTimer {
    pub name: &'static str,
    pub nanos: AtomicU64,
    pub calls: AtomicU64,
}

impl KernelTimer {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            nanos: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn record(&self, ns: u64) {
        self.nanos.fetch_add(ns, MemOrdering::Relaxed);
        self.calls.fetch_add(1, MemOrdering::Relaxed);
    }
}

/// All kernel timers, listed once at module load so `dump_all` can iterate.
/// Update this list when adding a new timer constant below.
const TIMERS: &[&KernelTimer] = &[
    &MATMUL_QT_T_F32_THREADED,
    &MATMUL_QKV_QT_T_F32_THREADED,
    &MATMUL_T_THREADED,
    &MATMUL,
    &MATMUL_T,
    &FLASH_ATTN_DECODE,
    &QTENSOR_MATMUL_F32,
    &MATMUL_QT_T_F32_CHUNK,
    &TOPK_SCAN,
    // Phase-2 (added 2026-06-07 per workflow wjxvc4laa adversarial verdict):
    // these previously-uninstrumented externs are suspect for the "257 ms
    // unattributed Haxe orchestration" bucket the per-phase profile
    // surfaced. They live in the LlamaModel forward path and run real
    // Rust kernel work (refcount atomics, vectorised norms, gather,
    // RoPE, in-place add). Without coverage, the per-phase profile
    // mis-attributes their wall to "pure Haxe overhead".
    &TENSOR_CLONE,
    &TENSOR_FREE,
    &TENSOR_ADD_INTO,
    &TENSOR_RMS_NORM,
    &TENSOR_SILU,
    &TENSOR_SOFTMAX,
    &TENSOR_ROPE,
    &TENSOR_GATHER_ROWS,
    &TENSOR_RESHAPE,
    &TENSOR_GET_FLAT,
];

pub static MATMUL_QT_T_F32_THREADED: KernelTimer = KernelTimer::new("matmul_qt_t_f32_threaded");
pub static MATMUL_QKV_QT_T_F32_THREADED: KernelTimer =
    KernelTimer::new("matmul_qkv_qt_t_f32_threaded");
pub static MATMUL_T_THREADED: KernelTimer = KernelTimer::new("matmul_t_threaded");
pub static MATMUL: KernelTimer = KernelTimer::new("matmul");
pub static MATMUL_T: KernelTimer = KernelTimer::new("matmul_t");
pub static FLASH_ATTN_DECODE: KernelTimer = KernelTimer::new("flash_attn_decode");
pub static QTENSOR_MATMUL_F32: KernelTimer = KernelTimer::new("qtensor_matmul_f32");
pub static MATMUL_QT_T_F32_CHUNK: KernelTimer = KernelTimer::new("matmul_qt_t_f32_chunk");
pub static TOPK_SCAN: KernelTimer = KernelTimer::new("topk_scan");
pub static TENSOR_CLONE: KernelTimer = KernelTimer::new("tensor_clone");
pub static TENSOR_FREE: KernelTimer = KernelTimer::new("tensor_free");
pub static TENSOR_ADD_INTO: KernelTimer = KernelTimer::new("tensor_add_into");
pub static TENSOR_RMS_NORM: KernelTimer = KernelTimer::new("tensor_rms_norm");
pub static TENSOR_SILU: KernelTimer = KernelTimer::new("tensor_silu");
pub static TENSOR_SOFTMAX: KernelTimer = KernelTimer::new("tensor_softmax");
pub static TENSOR_ROPE: KernelTimer = KernelTimer::new("tensor_rope");
pub static TENSOR_GATHER_ROWS: KernelTimer = KernelTimer::new("tensor_gather_rows");
pub static TENSOR_RESHAPE: KernelTimer = KernelTimer::new("tensor_reshape");
pub static TENSOR_GET_FLAT: KernelTimer = KernelTimer::new("tensor_get_flat");

/// Idempotent init. Reads `RAYZOR_KERNEL_TIMING` once, sets `ENABLED`,
/// registers an atexit dumper. Call from program start or lazily from
/// `time_kernel!` — `Once` makes both safe.
pub fn init() {
    INIT.call_once(|| {
        let on = std::env::var("RAYZOR_KERNEL_TIMING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        ENABLED.store(on, MemOrdering::Relaxed);
        if on {
            extern "C" fn dumper() {
                dump_all();
            }
            unsafe {
                libc::atexit(dumper);
            }
        }
    });
}

#[inline(always)]
pub fn is_enabled() -> bool {
    ENABLED.load(MemOrdering::Relaxed)
}

/// Print all timers to stderr in decreasing wall-time order. Safe to
/// call from anywhere (no locks taken beyond stderr's). Called from
/// atexit when `RAYZOR_KERNEL_TIMING=1`.
pub fn dump_all() {
    let mut rows: Vec<(&'static str, u64, u64)> = TIMERS
        .iter()
        .map(|t| {
            (
                t.name,
                t.nanos.load(MemOrdering::Relaxed),
                t.calls.load(MemOrdering::Relaxed),
            )
        })
        .filter(|&(_, ns, c)| ns > 0 || c > 0)
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    let total_ns: u64 = rows.iter().map(|r| r.1).sum();
    eprintln!(
        "[kernel-timing] total_wall_ns={} ({} kernels reported)",
        total_ns,
        rows.len()
    );
    for (name, ns, calls) in &rows {
        let pct = if total_ns > 0 {
            *ns as f64 / total_ns as f64 * 100.0
        } else {
            0.0
        };
        let ms = *ns as f64 / 1_000_000.0;
        let avg_us = if *calls > 0 {
            (*ns as f64) / (*calls as f64) / 1000.0
        } else {
            0.0
        };
        eprintln!(
            "[kernel-timing] {:>40}  {:>10.3} ms  ({:>5.1}%)  {:>8} calls  {:>10.3} us/call",
            name, ms, pct, calls, avg_us
        );
    }
}

/// RAII guard that times an FFI body. Drop records elapsed nanos into
/// the bound timer. Cheap when timing is disabled — `start` stays `None`
/// and `Drop` no-ops.
///
/// Example:
/// ```ignore
/// pub unsafe extern "C" fn rayzor_tensor_matmul(...) -> i64 {
///     crate::kernel_timing::init();
///     let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL);
///     // body — RAII records on every return path
/// }
/// ```
pub struct TimerGuard {
    start: Option<Instant>,
    timer: &'static KernelTimer,
}

impl TimerGuard {
    #[inline(always)]
    pub fn new(timer: &'static KernelTimer) -> Self {
        let start = if is_enabled() {
            Some(Instant::now())
        } else {
            None
        };
        Self { start, timer }
    }
}

impl Drop for TimerGuard {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            self.timer.record(start.elapsed().as_nanos() as u64);
        }
    }
}
