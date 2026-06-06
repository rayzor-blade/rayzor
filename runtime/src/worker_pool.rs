//! Persistent worker pool for fork-join CPU parallelism.
//!
//! `parallel_rows(rows, threads, f)` enqueues `threads` jobs (one per
//! disjoint row band) into a shared queue, wakes the workers, and waits
//! for all to complete. Workers block on a condvar between jobs.
//!
//! Closures must be `Fn + Send + Sync + 'static`. `parallel_rows` blocks
//! until every dispatched job returns, so non-`'static` data the closure
//! references logically outlives the call — but the type system can't see
//! that, so callers either pass `'static` data or use `Arc`.

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct PoolInner {
    queue: Mutex<VecDeque<Job>>,
    queue_cv: Condvar,
    shutdown: AtomicBool,
}

pub struct WorkerPool {
    inner: Arc<PoolInner>,
    handles: Vec<JoinHandle<()>>,
    n: usize,
}

impl WorkerPool {
    pub fn new(n_workers: usize) -> Self {
        let inner = Arc::new(PoolInner {
            queue: Mutex::new(VecDeque::new()),
            queue_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });

        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let inner_w = inner.clone();
            handles.push(std::thread::spawn(move || {
                bias_to_performance_core();
                worker_loop(inner_w);
            }));
        }

        Self {
            inner,
            handles,
            n: n_workers,
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
        let n = threads.min(self.n).min(rows);
        if n <= 1 {
            f(0, rows);
            return;
        }

        let chunk = rows.div_ceil(n);
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let f = Arc::new(f);

        {
            let mut q = self.inner.queue.lock();
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
            self.inner.queue_cv.notify_all();
        }
        drop(done_tx);

        // Drain `n` completions. The channel's tx end is shared between
        // dispatched jobs and our own (now-dropped) handle, so the loop
        // exits cleanly when all workers have signalled.
        for _ in 0..n {
            // `recv()` errors when the last sender drops without sending
            // — in our pattern that only happens if a worker panics. Treat
            // the error as "the panicking worker counts as done"; the
            // panic itself surfaces on the worker's join handle.
            if done_rx.recv().is_err() {
                break;
            }
        }
    }

    /// Like `parallel_rows` but runs on the caller's thread when called from
    /// inside an existing worker (to avoid pool-into-pool deadlock when
    /// nested work is dispatched recursively).
    ///
    /// Today no caller nests. Provided as a hook for the future tile path
    /// where the outer matmul might dispatch per-block work via the same
    /// pool.
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

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        self.inner.queue_cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

thread_local! {
    static WORKER_THREAD: std::cell::RefCell<bool> = const { std::cell::RefCell::new(false) };
}

fn worker_loop(inner: Arc<PoolInner>) {
    WORKER_THREAD.with(|on| *on.borrow_mut() = true);

    loop {
        let job = {
            let mut q = inner.queue.lock();
            loop {
                if let Some(j) = q.pop_front() {
                    break Some(j);
                }
                if inner.shutdown.load(Ordering::Acquire) {
                    break None;
                }
                inner.queue_cv.wait(&mut q);
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

/// Process-wide singleton. Lazily constructed on first `global()` call with a
/// worker count picked from `RAYZOR_WORKERS` (or 6 by default on M-series).
pub fn global() -> &'static WorkerPool {
    static POOL: std::sync::OnceLock<WorkerPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
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
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn parallel_rows_covers_every_index_exactly_once() {
        let pool = WorkerPool::new(4);
        let touched: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0u8; 1000]));
        let t = touched.clone();
        pool.parallel_rows(1000, 4, move |lo, hi| {
            let mut v = t.lock();
            for i in lo..hi {
                v[i] += 1;
            }
        });
        let v = touched.lock();
        for (i, &n) in v.iter().enumerate() {
            assert_eq!(n, 1, "index {i} was visited {n} times");
        }
    }

    #[test]
    fn parallel_rows_distributes_work() {
        let pool = WorkerPool::new(4);
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        pool.parallel_rows(100, 4, move |_lo, _hi| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn multiple_dispatches_share_the_pool() {
        let pool = WorkerPool::new(4);
        for _ in 0..50 {
            let counter = Arc::new(AtomicUsize::new(0));
            let c = counter.clone();
            pool.parallel_rows(40, 4, move |lo, hi| {
                c.fetch_add(hi - lo, Ordering::Relaxed);
            });
            assert_eq!(counter.load(Ordering::Relaxed), 40);
        }
    }
}
