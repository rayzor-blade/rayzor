//! Persistent worker pool with **spin-wait fork-join** for sub-microsecond
//! dispatch latency.
//!
//! `parallel_rows(rows, threads, f)` writes one job slot per worker
//! (lo, hi, type-erased closure pointer + trampoline), wakes every worker
//! by flipping a per-worker `AtomicU8` flag, then spin-waits on a global
//! countdown. Each worker spin-waits on its own flag; once flipped, runs
//! the trampoline, decrements the global countdown, returns to spin.
//!
//! Why spin-wait: post c5ab136 (llama.cpp kernel port), per-worker
//! matmul work dropped 187.5 → 70.9 us. The earlier condvar-based wake
//! cost ~5-15 us per wake which was sub-noise relative to 187.5 us of
//! work but is now 7-20% of every per-worker call. Spin-wait wakes in
//! ~50-100 ns (atomic load loop with `spin_loop()` hint) — two orders
//! of magnitude faster.
//!
//! Empirical: long-form 807-token sustained decode hits ~64 tok/s with
//! spin pool vs ~50 tok/s with condvar pool. Short-form decode (80
//! tokens, low total fork-joins) shows higher variance because OS
//! scheduling effects don't get amortised; under contention from other
//! foreground apps short-prompt latency can degrade. The trade is the
//! right one for sustained inference workloads.
//!
//! Power: spin-wait burns power when idle. For LLM inference workers
//! are >95% active during decode; the idle case is a non-goal. After
//! `SPIN_LIMIT` iterations the worker checks the shutdown flag — no
//! condvar fallback because LLM decode is a steady pipeline of
//! fork-joins ≤ 1 ms apart.
//!
//! Closures must be `Fn + Send + Sync + 'static`. `parallel_rows` blocks
//! until every worker has marked done, so non-`'static` data the closure
//! references logically outlives the call — but the type system can't see
//! that, so callers either pass `'static` data or use `Arc`.
//!
//! Legacy condvar+mpsc path is reachable via `RAYZOR_LEGACY_POOL=1` for
//! A/B regression testing.

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;

/// Maximum workers we statically pre-allocate slot arrays for.
const MAX_WORKERS: usize = 16;

/// Pad each worker slot to 128 bytes so independent workers don't
/// false-share cache lines on the dispatch flag.
#[repr(C, align(128))]
struct CachelinePad<T> {
    inner: T,
}

impl<T> std::ops::Deref for CachelinePad<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner
    }
}

/// One per worker. Dispatcher writes (lo, hi, trampoline, fn_ptr) then
/// flips `state` 0→1. Worker spin-loads `state`, sees 1, runs the
/// trampoline, stores `state` ← 0, decrements `pending` on the parent.
struct WorkerSlot {
    /// 0 = idle, 1 = work assigned. Worker resets to 0 after running.
    state: AtomicU8,
    /// Closure-erased trampoline: takes a `*const ()` pointer to the
    /// real closure plus the (lo, hi) band.
    trampoline: AtomicPtr<()>,
    /// Type-erased pointer to the real closure. Dispatcher writes; the
    /// trampoline knows how to cast back.
    closure_ptr: AtomicPtr<()>,
    lo: AtomicUsize,
    hi: AtomicUsize,
}

impl WorkerSlot {
    const fn new() -> Self {
        Self {
            state: AtomicU8::new(0),
            trampoline: AtomicPtr::new(std::ptr::null_mut()),
            closure_ptr: AtomicPtr::new(std::ptr::null_mut()),
            lo: AtomicUsize::new(0),
            hi: AtomicUsize::new(0),
        }
    }
}

struct PoolInner {
    /// One slot per worker (worker index = position in array). Max
    /// `MAX_WORKERS`; if `n_workers` > MAX, we cap.
    slots: Vec<CachelinePad<WorkerSlot>>,
    /// Number of in-flight bands. Dispatcher initialises to `n` before
    /// signalling slots, then spin-waits for it to reach 0.
    pending: CachelinePad<AtomicUsize>,
    shutdown: AtomicBool,
    /// Legacy condvar-pool fallback (RAYZOR_LEGACY_POOL=1). Initialised
    /// only when the env var is set so the spin-wait path stays clean.
    legacy: Option<Arc<LegacyInner>>,
}

struct LegacyInner {
    queue: Mutex<VecDeque<Box<dyn FnOnce() + Send + 'static>>>,
    queue_cv: Condvar,
}

pub struct WorkerPool {
    inner: Arc<PoolInner>,
    handles: Vec<JoinHandle<()>>,
    n: usize,
    legacy_mode: bool,
}

/// Worker checks the shutdown flag every `SPIN_LIMIT` iterations. Below
/// the limit it spin-loads its state flag with `spin_loop()` hint —
/// ~50-100 ns per check on M1. Setting this high (8192) means workers
/// stay hot during a sustained decode pipeline (fork-joins ≤ 1 ms apart)
/// without burning a syscall on the common idle path.
const SPIN_LIMIT: u32 = 8192;

impl WorkerPool {
    pub fn new(n_workers: usize) -> Self {
        let n = n_workers.min(MAX_WORKERS);
        let legacy_mode = std::env::var("RAYZOR_LEGACY_POOL")
            .map(|v| v == "1")
            .unwrap_or(false);

        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(CachelinePad {
                inner: WorkerSlot::new(),
            });
        }

        let legacy = if legacy_mode {
            Some(Arc::new(LegacyInner {
                queue: Mutex::new(VecDeque::new()),
                queue_cv: Condvar::new(),
            }))
        } else {
            None
        };

        let inner = Arc::new(PoolInner {
            slots,
            pending: CachelinePad {
                inner: AtomicUsize::new(0),
            },
            shutdown: AtomicBool::new(false),
            legacy,
        });

        let mut handles = Vec::with_capacity(n);
        for w in 0..n {
            let inner_w = inner.clone();
            handles.push(std::thread::spawn(move || {
                bias_to_performance_core();
                if legacy_mode {
                    legacy_worker_loop(inner_w);
                } else {
                    spin_worker_loop(inner_w, w);
                }
            }));
        }

        Self {
            inner,
            handles,
            n,
            legacy_mode,
        }
    }

    /// Number of worker threads in this pool.
    pub fn workers(&self) -> usize {
        self.n
    }

    /// Dispatch `f(lo, hi)` over `threads` disjoint contiguous ranges that
    /// cover `[0, rows)`, wait for all to complete.
    ///
    /// When `threads <= 1` or `rows < threads`, runs inline on the calling
    /// thread (no enqueue, no wake, no wait). When `threads > workers()` it
    /// clamps to the available worker count.
    pub fn parallel_rows<F>(&self, rows: usize, threads: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        if rows == 0 {
            return;
        }
        // Caller participation: the calling thread computes band 0 while
        // workers run bands 1..n. Pre-change the caller pure-spun on the
        // join for the whole op, occupying a P-core without contributing —
        // on an 8-P-core M1 Pro that capped compute at `workers` threads
        // and made `workers = 8` catastrophically oversubscribed
        // (9 runnable → E-core straggler gates every join → 2.3x decode
        // collapse, measured). With participation, RAYZOR_WORKERS=7 means
        // 8 compute threads on 8 P-cores with zero oversubscription.
        // `RAYZOR_NO_CALLER_BAND=1` restores the old spin-only join.
        let caller_assists = caller_band_enabled() && !self.legacy_mode;
        let max_width = if caller_assists { self.n + 1 } else { self.n };
        let n = threads.min(max_width).min(rows);
        if n <= 1 {
            f(0, rows);
            return;
        }

        if self.legacy_mode {
            return self.parallel_rows_legacy(rows, n, f);
        }

        let chunk = rows.div_ceil(n);

        // Build bands. Any band whose lo >= rows is empty and skipped —
        // `pending` is set to the number of ACTUAL worker dispatches.
        let first_worker_band = if caller_assists { 1 } else { 0 };
        let mut dispatched = 0usize;
        for w in first_worker_band..n {
            let lo = w * chunk;
            if lo >= rows {
                break;
            }
            dispatched += 1;
        }

        if dispatched == 0 {
            // Tiny row count: everything fits in the caller's band.
            f(0, rows);
            return;
        }

        // Initialise pending BEFORE flipping any state bits so a worker
        // that wakes early sees the correct countdown when it decrements.
        self.inner.pending.store(dispatched, Ordering::Release);

        // Type-erase the closure to (*const F) + a trampoline that
        // knows how to cast back. The closure stays on this stack
        // frame for the duration of the call — we spin-wait below
        // until every worker has decremented `pending`.
        let f_ptr: *const F = &f;
        let trampoline = trampoline_for::<F> as *mut ();

        for (slot_idx, w) in (first_worker_band..first_worker_band + dispatched).enumerate() {
            let lo = w * chunk;
            let hi = (lo + chunk).min(rows);
            let slot = &self.inner.slots[slot_idx];
            slot.lo.store(lo, Ordering::Relaxed);
            slot.hi.store(hi, Ordering::Relaxed);
            slot.closure_ptr.store(f_ptr as *mut (), Ordering::Relaxed);
            slot.trampoline.store(trampoline, Ordering::Relaxed);
            // Release: workers must see all the Relaxed writes above
            // before they observe state == 1.
            slot.state.store(1, Ordering::Release);
        }

        if caller_assists {
            // Compute band 0 on this thread while workers run theirs.
            // Mark the thread as a worker for the duration so any nested
            // `parallel_rows_no_nest` from kernel code runs inline instead
            // of corrupting the busy worker slots.
            WORKER_THREAD.with(|on| *on.borrow_mut() = true);
            f(0, chunk.min(rows));
            WORKER_THREAD.with(|on| *on.borrow_mut() = false);
        }

        // Spin-wait for pending → 0. Acquire pairs with each worker's
        // Release fetch_sub, ensuring the closure's writes (to
        // `out_tensor`) are visible before we return.
        while self.inner.pending.load(Ordering::Acquire) > 0 {
            std::hint::spin_loop();
        }
    }

    fn parallel_rows_legacy<F>(&self, rows: usize, n: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        let legacy = self
            .inner
            .legacy
            .as_ref()
            .expect("legacy mode without legacy inner");
        let chunk = rows.div_ceil(n);
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let f = Arc::new(f);

        {
            let mut q = legacy.queue.lock();
            for w in 0..n {
                let lo = w * chunk;
                if lo >= rows {
                    break;
                }
                let hi = (lo + chunk).min(rows);
                let f = f.clone();
                let done_tx = done_tx.clone();
                q.push_back(Box::new(move || {
                    f(lo, hi);
                    let _ = done_tx.send(());
                }));
            }
            legacy.queue_cv.notify_all();
        }
        drop(done_tx);

        for _ in 0..n {
            if done_rx.recv().is_err() {
                break;
            }
        }
    }

    /// Like `parallel_rows` but runs on the caller's thread when called from
    /// inside an existing worker (to avoid pool-into-pool deadlock when
    /// nested work is dispatched recursively).
    pub fn parallel_rows_no_nest<F>(&self, rows: usize, threads: usize, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        if WORKER_THREAD.with(|on| *on.borrow()) {
            f(0, rows);
            return;
        }
        self.parallel_rows(rows, threads, f);
    }
}

/// Type-erased trampoline. Workers store `fn(*const (), usize, usize)`
/// in `trampoline` (cast from `trampoline_for::<F>`) and cast the
/// `closure_ptr` back to `*const F` inside this function.
unsafe fn trampoline_for<F: Fn(usize, usize) + Send + Sync>(
    closure_ptr: *const (),
    lo: usize,
    hi: usize,
) {
    let f = &*(closure_ptr as *const F);
    f(lo, hi);
}

type TrampolineFn = unsafe fn(*const (), usize, usize);

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        if let Some(legacy) = self.inner.legacy.as_ref() {
            legacy.queue_cv.notify_all();
        }
        // Spin workers: they observe shutdown == true on their next
        // spin iteration and return.
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

thread_local! {
    static WORKER_THREAD: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
}

fn spin_worker_loop(inner: Arc<PoolInner>, worker_idx: usize) {
    WORKER_THREAD.with(|on| *on.borrow_mut() = true);

    let slot = &inner.slots[worker_idx];
    let mut spin_count: u32 = 0;
    loop {
        // Spin-wait for work. We never sleep — for LLM decode the gap
        // between fork-joins is sub-millisecond. The shutdown flag is
        // checked at most every SPIN_LIMIT iterations.
        loop {
            let state = slot.state.load(Ordering::Acquire);
            if state == 1 {
                break;
            }
            spin_count = spin_count.wrapping_add(1);
            if spin_count >= SPIN_LIMIT {
                spin_count = 0;
                if inner.shutdown.load(Ordering::Acquire) {
                    return;
                }
            }
            std::hint::spin_loop();
        }

        let lo = slot.lo.load(Ordering::Relaxed);
        let hi = slot.hi.load(Ordering::Relaxed);
        let trampoline_raw = slot.trampoline.load(Ordering::Relaxed);
        let closure_ptr = slot.closure_ptr.load(Ordering::Relaxed);
        // SAFETY: the dispatcher wrote a valid (trampoline, closure)
        // pair before flipping `state` 1. The closure lives on the
        // dispatcher's stack until pending → 0; we mark our band done
        // BEFORE returning to the spin loop.
        unsafe {
            let trampoline: TrampolineFn = std::mem::transmute(trampoline_raw);
            trampoline(closure_ptr as *const (), lo, hi);
        }

        // Reset our slot — dispatcher won't see another flip until it
        // gets the green light from `pending == 0`.
        slot.state.store(0, Ordering::Release);
        // Release: the trampoline's writes (to out_tensor) must be
        // visible to the dispatcher before it observes the
        // decremented pending count.
        inner.pending.fetch_sub(1, Ordering::Release);
    }
}

fn legacy_worker_loop(inner: Arc<PoolInner>) {
    WORKER_THREAD.with(|on| *on.borrow_mut() = true);
    let legacy = inner
        .legacy
        .as_ref()
        .expect("legacy worker without legacy inner")
        .clone();

    loop {
        let job = {
            let mut q = legacy.queue.lock();
            loop {
                if let Some(j) = q.pop_front() {
                    break Some(j);
                }
                if inner.shutdown.load(Ordering::Acquire) {
                    break None;
                }
                legacy.queue_cv.wait(&mut q);
            }
        };
        match job {
            Some(j) => j(),
            None => break,
        }
    }
}

/// Apple-silicon scheduler hint: bias workers toward performance cores.
/// No-op on other platforms.
///
/// Uses `QOS_CLASS_USER_INTERACTIVE` (0x21) — the highest user QoS
/// class on macOS — instead of `QOS_CLASS_USER_INITIATED` (0x19).
/// INTERACTIVE biases the scheduler to keep these threads on P-cores
/// even under moderate thermal pressure; INITIATED can be demoted to
/// E-cores once the system warms. For matmul fork-join workloads
/// whose wall time is `max` across workers, even one E-core
/// straggler caps the achievable speedup, so the higher priority
/// pays for itself.
///
/// Set `RAYZOR_QOS=initiated` to fall back to USER_INITIATED — kept
/// as an escape hatch in case INTERACTIVE causes priority inversion
/// with other user-visible work (e.g. running rayzor alongside a
/// foreground GUI app where it should yield).
#[inline]
fn bias_to_performance_core() {
    #[cfg(target_os = "macos")]
    {
        const QOS_CLASS_USER_INTERACTIVE: std::ffi::c_uint = 0x21;
        const QOS_CLASS_USER_INITIATED: std::ffi::c_uint = 0x19;
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(
                qos_class: std::ffi::c_uint,
                relative_priority: std::ffi::c_int,
            ) -> std::ffi::c_int;
        }
        let qos = match std::env::var("RAYZOR_QOS").ok().as_deref() {
            Some("initiated") => QOS_CLASS_USER_INITIATED,
            _ => QOS_CLASS_USER_INTERACTIVE,
        };
        unsafe {
            let _ = pthread_set_qos_class_self_np(qos, 0);
        }
    }
}

/// Band count kernels should use for fork-join fan-out. Follows the global
/// pool's worker count, so `RAYZOR_WORKERS` sweeps the WHOLE dispatch width.
/// Historically five kernel call sites hardcoded 6 bands independently of
/// the pool size, which silently capped the dominant matmul at 6 threads on
/// 8-P-core machines regardless of the env knob.
///
/// With caller participation (the calling thread computes band 0), total
/// compute width is `workers + 1` — RAYZOR_WORKERS=7 → 8 compute threads.
pub fn auto_kernel_threads() -> usize {
    let w = global().workers();
    if caller_band_enabled() {
        w + 1
    } else {
        w
    }
}

/// Whether `parallel_rows` runs band 0 on the calling thread.
/// `RAYZOR_NO_CALLER_BAND=1` restores the pre-participation spin-only join.
fn caller_band_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RAYZOR_NO_CALLER_BAND").map_or(true, |v| v != "1"))
}

/// Process-wide singleton. Lazily constructed on first `global()` call with a
/// worker count picked from `RAYZOR_WORKERS` (or 6 by default on M-series).
pub fn global() -> &'static WorkerPool {
    static POOL: std::sync::OnceLock<WorkerPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        // `get_or_init` runs on the ORCHESTRATOR thread (first kernel
        // call site), so this QoS hint lands on the caller — the thread
        // that runs all sequential kernels, the Haxe decode loop, and
        // every fork-join wait. Workers get the same hint at spawn;
        // without this the caller competes at default QoS against
        // max-QoS spinners and can be demoted to an E-core under
        // thermal pressure, inflating the glue slice of every token.
        bias_to_performance_core();
        let n = std::env::var("RAYZOR_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(6);
        WorkerPool::new(n)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn inline_when_only_one_row() {
        let pool = WorkerPool::new(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        pool.parallel_rows(1, 4, move |_lo, _hi| {
            c.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn parallel_rows_covers_every_index_exactly_once() {
        let pool = WorkerPool::new(6);
        const N: usize = 1024;
        let touched: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
        let touched = Arc::new(touched);
        let t = touched.clone();
        pool.parallel_rows(N, 6, move |lo, hi| {
            for i in lo..hi {
                t[i].fetch_add(1, Ordering::SeqCst);
            }
        });
        for (i, c) in touched.iter().enumerate() {
            assert_eq!(c.load(Ordering::SeqCst), 1, "index {} touched != 1", i);
        }
    }

    #[test]
    fn parallel_rows_distributes_work() {
        let pool = WorkerPool::new(4);
        let counts: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        let counts = Arc::new(counts);
        let c = counts.clone();
        pool.parallel_rows(400, 4, move |lo, hi| {
            // Each worker hits at least one row — use lo/4 as worker idx.
            let worker_idx = lo / 100;
            c[worker_idx].fetch_add(hi - lo, Ordering::SeqCst);
        });
        let total: usize = counts.iter().map(|c| c.load(Ordering::SeqCst)).sum();
        assert_eq!(total, 400);
    }
}
