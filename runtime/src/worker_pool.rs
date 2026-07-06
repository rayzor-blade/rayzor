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
    /// Set true by the worker just before it `thread::park()`s after a long
    /// idle (see `PARK_AFTER_ROUNDS`); cleared on wake. The dispatcher reads
    /// it (under a paired SeqCst fence) to decide whether to `unpark`. Lets a
    /// genuinely-idle or orphaned pool release its cores instead of spinning
    /// them forever, without adding a syscall to the hot dispatch path.
    parked: AtomicBool,
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
            parked: AtomicBool::new(false),
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
    /// Count of workers currently parked (or transitioning to parked). The
    /// dispatcher reads this (after a SeqCst fence) to short-circuit the
    /// per-slot unpark scan: zero on every hot-path dispatch, non-zero only
    /// when the pool has been idle long enough to release cores.
    parked_count: AtomicUsize,
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

/// After this many consecutive `SPIN_LIMIT` rounds with no work, a worker
/// `thread::park()`s to release its core. An idle round spins on an
/// uncontended cache line, so ~2048 rounds is on the order of tens-to-low-
/// hundreds of ms of continuous idle before parking — far longer than the
/// sub-ms gap between decode fork-joins, so an actively-decoding pool resets
/// its idle counter long before parking and the spin-wait hot path is
/// unchanged. Even if a worker did park during an unusually long gap, the
/// next dispatch just pays a one-off ~µs unpark. Parking engages between
/// requests, when the process is idle, or when it has been orphaned —
/// exactly the cases where spinning a core is pure waste (and, for orphans,
/// a benchmark-contaminating hazard). `RAYZOR_NO_PARK=1` disables.
const PARK_AFTER_ROUNDS: u32 = 2048;

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
            parked_count: AtomicUsize::new(0),
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

    /// Wake any parked workers among the first `dispatched` slots. Called
    /// right after a dispatch loop flips their `state` to 1.
    ///
    /// Hot-path cost when nothing is parked (every dispatch during active
    /// decode): one `fence(SeqCst)` + one relaxed load of `parked_count`,
    /// then a short-circuit return — no per-slot scan, no syscall. The fence
    /// pairs with each worker's fence so a worker mid-transition-to-parked is
    /// never left asleep (the SeqCst total order forces at least one side to
    /// observe the other).
    #[inline]
    fn wake_parked(&self, dispatched: usize) {
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.inner.parked_count.load(Ordering::Relaxed) == 0 {
            return;
        }
        for slot_idx in 0..dispatched.min(self.n) {
            if self.inner.slots[slot_idx].parked.load(Ordering::Relaxed) {
                self.handles[slot_idx].thread().unpark();
            }
        }
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

        // Dynamic chunk stealing (default): all compute threads pull
        // fixed-size chunks from a shared atomic cursor instead of owning
        // one static band each. With static bands, ONE preempted thread
        // (tier-promotion compile, the streaming client waking per token,
        // any 9th runnable on 8 P-cores) stalls the join for its entire
        // band — measured as 10-20 tok/s dips on an otherwise ~90 tok/s
        // decode. With stealing, a preempted thread delays at most one
        // chunk. Each row is still computed wholly by one thread, so
        // outputs stay bit-identical. `RAYZOR_STATIC_BANDS=1` reverts.
        if chunk_stealing_enabled() {
            return self.parallel_rows_stealing(rows, n, caller_assists, f);
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
        // Wake any workers that parked during a prior idle window.
        self.wake_parked(dispatched);

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

    /// Chunk-stealing fork-join: dispatch a shared *driver* closure to the
    /// workers (and run it on the caller when assisting); every compute
    /// thread pulls `[lo, lo+chunk)` spans off one atomic cursor until the
    /// row space is drained. Reuses the existing slot/pending protocol —
    /// the driver simply ignores the static band written into its slot.
    fn parallel_rows_stealing<F>(&self, rows: usize, n: usize, caller_assists: bool, f: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        // Band-local stealing: each thread OWNS a contiguous band (so the
        // weight stream stays sequential for the hardware prefetcher —
        // pure shared-cursor stealing cost ~6 tok/s of median on a quiet
        // machine by scattering each thread's spans across the matrix)
        // and drains it in chunks through its own cache-line-private
        // cursor. Only threads that FINISH their band steal chunks from
        // the band with the most unclaimed rows, so a stalled thread
        // forfeits at most one chunk of its remainder to each helper
        // while the fast path keeps band locality.
        struct BandState {
            cursor: AtomicUsize,
            hi: usize,
        }
        let band = rows.div_ceil(n);
        // ~8 chunks per band: owner pays 8 uncontended fetch_adds on its
        // own line; straggler bound is band/8. Floor 8 rows.
        let chunk = band.div_ceil(8).max(8).min(rows);
        let bands: Vec<CachelinePad<BandState>> = (0..n)
            .map(|t| {
                let lo = (t * band).min(rows);
                let hi = ((t + 1) * band).min(rows);
                CachelinePad {
                    inner: BandState {
                        cursor: AtomicUsize::new(lo),
                        hi,
                    },
                }
            })
            .collect();
        let bands_ref = &bands;
        let f_ref = &f;

        // The slot's `lo` field carries the band index for this op.
        let driver = move |band_idx: usize, _hi: usize| {
            // Phase 1: drain the owned band front-to-back with GUIDED
            // claims — half the remaining band per claim, floored at
            // `chunk`. The first claims are huge contiguous spans (the
            // hardware prefetcher's favourite shape and only 2-4 atomics
            // on a private line); the tail claims shrink to `chunk` so a
            // straggling owner leaves fine-grained pieces for thieves.
            let mine = &bands_ref[band_idx].inner;
            loop {
                let cur = mine.cursor.load(Ordering::Relaxed);
                if cur >= mine.hi {
                    break;
                }
                let want = ((mine.hi - cur) / 2).max(chunk);
                let lo = mine.cursor.fetch_add(want, Ordering::Relaxed);
                if lo >= mine.hi {
                    break;
                }
                f_ref(lo, (lo + want).min(mine.hi));
            }
            // Phase 2: help stragglers. Pick the band with the most
            // unclaimed rows ONCE, then keep stealing `chunk`-sized
            // pieces from that victim until it drains; only then rescan.
            // A spin hint between empty scans keeps finished threads from
            // hammering the owners' cursor cache lines.
            loop {
                let mut victim: Option<usize> = None;
                let mut victim_left = 0usize;
                for (i, b) in bands_ref.iter().enumerate() {
                    if i == band_idx {
                        continue;
                    }
                    let cur = b.inner.cursor.load(Ordering::Relaxed);
                    let left = b.inner.hi.saturating_sub(cur);
                    if left > victim_left {
                        victim_left = left;
                        victim = Some(i);
                    }
                }
                let Some(v) = victim else { break };
                let vb = &bands_ref[v].inner;
                let mut stole_any = false;
                loop {
                    let lo = vb.cursor.fetch_add(chunk, Ordering::Relaxed);
                    if lo >= vb.hi {
                        break;
                    }
                    stole_any = true;
                    f_ref(lo, (lo + chunk).min(vb.hi));
                }
                if !stole_any {
                    // Lost the race for the last pieces — brief pause
                    // before rescanning so we don't ping-pong cursor
                    // lines under the remaining workers.
                    for _ in 0..32 {
                        std::hint::spin_loop();
                    }
                }
            }
        };

        let workers = if caller_assists { n - 1 } else { n };
        let workers = workers.min(self.n);

        self.inner.pending.store(workers, Ordering::Release);

        fn erase<D: Fn(usize, usize) + Send + Sync>(d: &D) -> (*const (), *mut ()) {
            (d as *const D as *const (), trampoline_for::<D> as *mut ())
        }
        let (d_ptr, trampoline) = erase(&driver);
        // Caller (when assisting) takes band 0; workers take 1..n.
        let first_worker_band = if caller_assists { 1 } else { 0 };
        for (slot_idx, slot) in self.inner.slots.iter().take(workers).enumerate() {
            slot.lo
                .store(first_worker_band + slot_idx, Ordering::Relaxed);
            slot.hi.store(0, Ordering::Relaxed);
            slot.closure_ptr.store(d_ptr as *mut (), Ordering::Relaxed);
            slot.trampoline.store(trampoline, Ordering::Relaxed);
            slot.state.store(1, Ordering::Release);
        }
        // Wake any workers that parked during a prior idle window.
        self.wake_parked(workers);

        if caller_assists {
            WORKER_THREAD.with(|on| *on.borrow_mut() = true);
            driver(0, 0);
            WORKER_THREAD.with(|on| *on.borrow_mut() = false);
        }

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
        // Wake any parked workers so they observe shutdown and exit; the
        // unparked spin workers see it on their next spin iteration.
        for h in &self.handles {
            h.thread().unpark();
        }
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

    let park_after = if park_enabled() {
        PARK_AFTER_ROUNDS
    } else {
        u32::MAX // never park
    };
    let slot = &inner.slots[worker_idx];
    let mut spin_count: u32 = 0;
    let mut idle_rounds: u32 = 0;
    loop {
        // Spin-wait for work. During a sustained decode pipeline the gap
        // between fork-joins is sub-millisecond, so we stay hot (the
        // shutdown flag is checked at most every SPIN_LIMIT iterations).
        // Only after ~0.5 s of *continuous* idle (PARK_AFTER_ROUNDS) do we
        // park to release the core — see PARK_AFTER_ROUNDS.
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
                idle_rounds = idle_rounds.wrapping_add(1);
                if idle_rounds >= park_after {
                    // Long idle: announce parked, then a SeqCst fence + a
                    // re-read of `state`. This pairs with the dispatcher's
                    // fence between its `state.store(1)` writes and its
                    // `parked` reads, so a dispatch that races our decision
                    // to sleep is never missed: either we observe state==1
                    // here and skip the park, or the dispatcher observes
                    // parked==true and unparks us.
                    slot.parked.store(true, Ordering::Relaxed);
                    inner.parked_count.fetch_add(1, Ordering::Relaxed);
                    std::sync::atomic::fence(Ordering::SeqCst);
                    if slot.state.load(Ordering::Relaxed) != 1
                        && !inner.shutdown.load(Ordering::Relaxed)
                    {
                        std::thread::park();
                    }
                    inner.parked_count.fetch_sub(1, Ordering::Relaxed);
                    slot.parked.store(false, Ordering::Relaxed);
                    idle_rounds = 0;
                    if inner.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                }
            }
            std::hint::spin_loop();
        }
        // Got work: reset the idle counter before running it.
        idle_rounds = 0;

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

/// Whether fork-joins use dynamic chunk stealing (default) instead of one
/// static band per thread. `RAYZOR_STATIC_BANDS=1` reverts.
fn chunk_stealing_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RAYZOR_STATIC_BANDS").map_or(true, |v| v != "1"))
}

/// Whether idle workers park after `PARK_AFTER_ROUNDS` (default on).
/// `RAYZOR_NO_PARK=1` keeps the pre-park behaviour (spin forever when idle).
fn park_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RAYZOR_NO_PARK").map_or(true, |v| v != "1"))
}

/// Matches the guest-side gate (`Linear.useHaxeMatmul`): set AND not
/// `"0"`/`""`/`"false"`. A bare `RAYZOR_HAXE_MATMUL=0` must behave like
/// unset — presence-only detection silently shrank the FFI-path pool to 1
/// worker and halved throughput.
fn haxe_matmul_active() -> bool {
    std::env::var("RAYZOR_HAXE_MATMUL")
        .map(|v| !v.is_empty() && v != "0" && v != "false")
        .unwrap_or(false)
}

/// Process-wide singleton. Lazily constructed on first `global()` call with a
/// worker count picked from `RAYZOR_WORKERS`, defaulting to a width derived
/// from the machine's performance-core count.
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
            .unwrap_or_else(|| {
                // On the pure-Haxe matmul path the guest SpinPool owns the
                // cores; a full-width Rust pool spin-waits against it and
                // caps the guest bands at ~2x scaling (measured: shrinking
                // this pool to 1 took llama-chat decode 65 -> 27.5 ms/token).
                // One worker (+ caller band) still covers the us-scale
                // attention/activation kernels.
                if haxe_matmul_active() {
                    1
                } else {
                    // Compute width = P-cores - 1 (caller participates, so
                    // workers = P - 2). E-cores are excluded — one straggler
                    // band on an E-core gates every fork-join — and one
                    // P-core stays free for the sequential glue between
                    // kernels (sampler, tokenizer, harness I/O).
                    (crate::topology::rayzor_topology_perf_core_count() as usize)
                        .saturating_sub(2)
                        .max(1)
                }
            });
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

    /// Workers must park after a long idle and then be correctly unparked by
    /// the next dispatch. A missed wakeup would leave a parked worker asleep,
    /// so its band's `pending` never decrements and `parallel_rows` would hang
    /// forever — completing the post-idle dispatch (covering every index once)
    /// proves the park→unpark protocol is race-free.
    #[test]
    fn parks_when_idle_then_unparks_on_next_dispatch() {
        let pool = WorkerPool::new(4);

        // Prime so workers run, finish, then go idle.
        let c = Arc::new(AtomicUsize::new(0));
        let cc = c.clone();
        pool.parallel_rows(64, 4, move |lo, hi| {
            cc.fetch_add(hi - lo, Ordering::SeqCst);
        });
        assert_eq!(c.load(Ordering::SeqCst), 64);

        // Idle well past the park threshold (tens-to-hundreds of ms).
        std::thread::sleep(std::time::Duration::from_millis(1500));
        assert!(
            pool.inner.parked_count.load(Ordering::SeqCst) > 0,
            "expected at least one worker to park after a long idle"
        );

        // Second dispatch must wake the parked workers and cover every index.
        const N: usize = 512;
        let touched: Vec<AtomicUsize> = (0..N).map(|_| AtomicUsize::new(0)).collect();
        let touched = Arc::new(touched);
        let t = touched.clone();
        pool.parallel_rows(N, 4, move |lo, hi| {
            for i in lo..hi {
                t[i].fetch_add(1, Ordering::SeqCst);
            }
        });
        for (i, cnt) in touched.iter().enumerate() {
            assert_eq!(
                cnt.load(Ordering::SeqCst),
                1,
                "index {i} touched != 1 after unpark"
            );
        }
    }
}
