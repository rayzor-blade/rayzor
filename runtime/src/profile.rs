//! Heap + CPU profiler. Activated by the `profile` Cargo feature; the
//! binary crate then sets `#[global_allocator]` to [`TrackingAllocator`]
//! and calls [`ensure_alloc_dump_hooks`] from `main`.
//!
//! Env vars:
//! - `RAYZOR_DUMP_ALLOC_AT_EXIT=1` — print `[alloc-stats]` summary at exit
//!   (atexit + SIGTRAP/SIGSEGV/SIGABRT signal handlers).
//! - `RAYZOR_ALLOC_GRAPH=1` `RAYZOR_ALLOC_GRAPH_RATE=N` — sample 1 in N
//!   allocations, capture top-6 PCs via `backtrace::trace`, dump
//!   `/tmp/rayzor_alloc_graph.csv` at exit.
//! - `RAYZOR_CPU_PROFILE=1` `RAYZOR_CPU_PROFILE_US=N` — install SIGPROF
//!   handler firing every N microseconds (default 1000 = 1ms); samples
//!   share the GRAPH_SITES table with the alloc path.
//!
//! Resolve PCs offline by joining the alloc-graph CSV with
//! `/tmp/rayzor_jit_symbols.csv` + `/tmp/rayzor_file_table.csv`
//! (both written by the compiler when `RAYZOR_DUMP_JIT_MAP=1`) via
//! `tools/resolve_alloc_graph.py`.
//!
//! See `memory/project_debugger_feasibility.md` for the wider picture.

use std::sync::atomic::{AtomicU64, Ordering as MemOrdering};

pub static ALLOC_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static FREE_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
pub static FREE_COUNT: AtomicU64 = AtomicU64::new(0);
pub static LIVE_BYTES_PEAK: AtomicU64 = AtomicU64::new(0);

static GRAPH_RATE: AtomicU64 = AtomicU64::new(0);
static CPU_PROFILE_ACTIVE: AtomicU64 = AtomicU64::new(0);

use std::cell::Cell;
use std::sync::Mutex;

std::thread_local! {
    /// Reentrancy guard. Set whenever this thread is INSIDE the
    /// allocator's sample path OR the dump path OR the SIGPROF handler.
    /// A signal handler hitting this same thread will see the flag and
    /// skip — otherwise we'd recurse into the allocator (allocating
    /// inside an alloc) or deadlock on a Mutex we already hold.
    static IN_GRAPH: Cell<bool> = const { Cell::new(false) };
    static GRAPH_TICK: Cell<u64> = const { Cell::new(0) };
}

#[derive(Default)]
struct SiteStat {
    sampled_count: u64,
    pcs: [usize; 6],
}

pub static GRAPH_SITES: Mutex<Option<std::collections::HashMap<u64, SiteStat>>> = Mutex::new(None);

/// Hot path called from `TrackingAllocator::alloc` on every successful
/// allocation. Cheap when `RAYZOR_ALLOC_GRAPH` isn't set (single atomic
/// load + `IN_GRAPH` check + early return).
#[inline(always)]
fn record_sample() {
    let rate = GRAPH_RATE.load(MemOrdering::Relaxed);
    if rate == 0 || IN_GRAPH.with(|g| g.get()) {
        return;
    }
    let tick = GRAPH_TICK.with(|t| {
        let v = t.get().wrapping_add(1);
        t.set(v);
        v
    });
    if tick % rate != 0 {
        return;
    }
    IN_GRAPH.with(|g| g.set(true));
    let mut pcs = [0usize; 12];
    let mut idx = 0;
    backtrace::trace(|f| {
        if idx < pcs.len() {
            pcs[idx] = f.ip() as usize;
            idx += 1;
            true
        } else {
            false
        }
    });
    let skip = 2.min(idx);
    let n = (idx - skip).min(6);
    let mut top = [0usize; 6];
    top[..n].copy_from_slice(&pcs[skip..skip + n]);
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    top.hash(&mut h);
    if let Ok(mut g) = GRAPH_SITES.lock() {
        if let Some(map) = g.as_mut() {
            let e = map
                .entry(h.finish())
                .or_insert(SiteStat { sampled_count: 0, pcs: top });
            e.sampled_count += 1;
        }
    }
    IN_GRAPH.with(|g| g.set(false));
}

#[no_mangle]
pub extern "C" fn rayzor_dump_alloc_stats() {
    let a = ALLOC_BYTES_TOTAL.load(MemOrdering::Relaxed);
    let f = FREE_BYTES_TOTAL.load(MemOrdering::Relaxed);
    eprintln!(
        "[alloc-stats] allocs={} frees={} alloc_bytes={} free_bytes={} live={} peak={}",
        ALLOC_COUNT.load(MemOrdering::Relaxed),
        FREE_COUNT.load(MemOrdering::Relaxed),
        a,
        f,
        a.saturating_sub(f),
        LIVE_BYTES_PEAK.load(MemOrdering::Relaxed)
    );
}

#[no_mangle]
pub extern "C" fn rayzor_dump_alloc_graph() {
    // Disarm SIGPROF first if active — otherwise the handler can fire
    // mid-dump and either spin on `try_lock` (cheap) or interleave
    // async-signal-unsafe work like backtrace symbol resolution.
    if CPU_PROFILE_ACTIVE.load(MemOrdering::Relaxed) == 1 {
        #[repr(C)]
        struct Timeval {
            tv_sec: i64,
            tv_usec: i32,
        }
        #[repr(C)]
        struct Itimerval {
            it_interval: Timeval,
            it_value: Timeval,
        }
        extern "C" {
            fn setitimer(which: i32, new_value: *const Itimerval, old_value: *mut Itimerval)
                -> i32;
        }
        let zero = Itimerval {
            it_interval: Timeval { tv_sec: 0, tv_usec: 0 },
            it_value: Timeval { tv_sec: 0, tv_usec: 0 },
        };
        unsafe { setitimer(2, &zero, std::ptr::null_mut()) };
        CPU_PROFILE_ACTIVE.store(0, MemOrdering::Relaxed);
    }

    let rate = GRAPH_RATE.load(MemOrdering::Relaxed);
    // CRITICAL: set the thread-local guard before any operation that
    // touches GRAPH_SITES or allocates — otherwise our Vec/HashMap
    // operations during dump re-enter `record_sample` and deadlock on
    // the same Mutex.
    IN_GRAPH.with(|g| g.set(true));
    let snapshot: Vec<(u64, [usize; 6])> = match GRAPH_SITES.lock() {
        Ok(g) => match g.as_ref() {
            Some(m) => m.values().map(|v| (v.sampled_count, v.pcs)).collect(),
            None => {
                IN_GRAPH.with(|g| g.set(false));
                return;
            }
        },
        Err(_) => {
            IN_GRAPH.with(|g| g.set(false));
            return;
        }
    };
    let mut rows = snapshot;
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    // In CPU-profile-only mode rate is 0; surface the raw sample count.
    let scale = if rate == 0 { 1 } else { rate };
    let mut out = String::from("rank,sampled_allocs,est_total_allocs,pc1,pc2,pc3,pc4,pc5,pc6\n");
    let n = rows.len().min(120);
    for (rank, (sc, pcs)) in rows.iter().enumerate().take(120) {
        out.push_str(&format!(
            "{},{},{},0x{:x},0x{:x},0x{:x},0x{:x},0x{:x},0x{:x}\n",
            rank + 1,
            sc,
            sc * scale,
            pcs[0],
            pcs[1],
            pcs[2],
            pcs[3],
            pcs[4],
            pcs[5]
        ));
    }
    let _ = std::fs::write("/tmp/rayzor_alloc_graph.csv", out);
    eprintln!(
        "[alloc-graph] wrote {} top sites to /tmp/rayzor_alloc_graph.csv (rate 1/{})",
        n, scale
    );
    IN_GRAPH.with(|g| g.set(false));
}

extern "C" fn sigprof_handler(_sig: i32) {
    if IN_GRAPH.with(|g| g.get()) {
        return;
    }
    IN_GRAPH.with(|g| g.set(true));
    let mut pcs = [0usize; 12];
    let mut idx = 0;
    backtrace::trace(|f| {
        if idx < pcs.len() {
            pcs[idx] = f.ip() as usize;
            idx += 1;
            true
        } else {
            false
        }
    });
    let skip = 2.min(idx);
    let n = (idx - skip).min(6);
    let mut top = [0usize; 6];
    top[..n].copy_from_slice(&pcs[skip..skip + n]);
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    top.hash(&mut h);
    // try_lock — Mutex isn't reentrant and signal handlers run on the
    // interrupted thread; calling lock() here on a thread that holds
    // GRAPH_SITES deadlocks.
    if let Ok(mut g) = GRAPH_SITES.try_lock() {
        if let Some(map) = g.as_mut() {
            let e = map
                .entry(h.finish())
                .or_insert(SiteStat { sampled_count: 0, pcs: top });
            e.sampled_count += 1;
        }
    }
    IN_GRAPH.with(|g| g.set(false));
}

unsafe fn install_cpu_profiler(period_us: u64) {
    extern "C" {
        fn signal(sig: i32, h: extern "C" fn(i32)) -> *mut std::ffi::c_void;
    }
    signal(27, sigprof_handler); // SIGPROF = 27 on macOS + Linux

    #[repr(C)]
    struct Timeval {
        tv_sec: i64,
        tv_usec: i32,
    }
    #[repr(C)]
    struct Itimerval {
        it_interval: Timeval,
        it_value: Timeval,
    }
    extern "C" {
        fn setitimer(which: i32, new_value: *const Itimerval, old_value: *mut Itimerval) -> i32;
    }
    let sec = (period_us / 1_000_000) as i64;
    let usec = (period_us % 1_000_000) as i32;
    let tv = Itimerval {
        it_interval: Timeval { tv_sec: sec, tv_usec: usec },
        it_value: Timeval { tv_sec: sec, tv_usec: usec },
    };
    setitimer(2, &tv, std::ptr::null_mut()); // ITIMER_PROF
    CPU_PROFILE_ACTIVE.store(1, MemOrdering::Relaxed);
}

pub struct TrackingAllocator;

unsafe impl std::alloc::GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let p = std::alloc::System.alloc(layout);
        if !p.is_null() {
            let sz = layout.size() as u64;
            let prev = ALLOC_BYTES_TOTAL.fetch_add(sz, MemOrdering::Relaxed);
            ALLOC_COUNT.fetch_add(1, MemOrdering::Relaxed);
            let live = (prev + sz).saturating_sub(FREE_BYTES_TOTAL.load(MemOrdering::Relaxed));
            LIVE_BYTES_PEAK.fetch_max(live, MemOrdering::Relaxed);
            record_sample();
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        std::alloc::System.dealloc(ptr, layout);
        FREE_BYTES_TOTAL.fetch_add(layout.size() as u64, MemOrdering::Relaxed);
        FREE_COUNT.fetch_add(1, MemOrdering::Relaxed);
    }
}

/// Install atexit + signal-driven dumpers and (when requested) arm the
/// SIGPROF profiler. Idempotent — safe to call multiple times. Should
/// be called from `fn main` early.
pub unsafe fn ensure_alloc_dump_hooks() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("RAYZOR_ALLOC_GRAPH").as_deref() == Some(std::ffi::OsStr::new("1")) {
            let r = std::env::var("RAYZOR_ALLOC_GRAPH_RATE")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1024);
            GRAPH_RATE.store(r, MemOrdering::Relaxed);
            *GRAPH_SITES.lock().unwrap() = Some(std::collections::HashMap::new());
        }
        if std::env::var_os("RAYZOR_CPU_PROFILE").as_deref() == Some(std::ffi::OsStr::new("1")) {
            let period = std::env::var("RAYZOR_CPU_PROFILE_US")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .filter(|&n| n > 0)
                .unwrap_or(1000);
            {
                let mut g = GRAPH_SITES.lock().unwrap();
                if g.is_none() {
                    *g = Some(std::collections::HashMap::new());
                }
            }
            install_cpu_profiler(period);
            eprintln!("[cpu-profile] SIGPROF profiler armed (period={}us)", period);
        }
        if std::env::var_os("RAYZOR_DUMP_ALLOC_AT_EXIT").as_deref()
            == Some(std::ffi::OsStr::new("1"))
        {
            extern "C" {
                fn atexit(cb: extern "C" fn()) -> i32;
                fn signal(sig: i32, h: extern "C" fn(i32)) -> *mut std::ffi::c_void;
            }
            extern "C" fn dump_all() {
                rayzor_dump_alloc_stats();
                rayzor_dump_alloc_graph();
            }
            atexit(dump_all);
            extern "C" fn sig_dump(sig: i32) {
                rayzor_dump_alloc_stats();
                rayzor_dump_alloc_graph();
                unsafe {
                    extern "C" {
                        fn _exit(s: i32) -> !;
                    }
                    _exit(128 + sig);
                }
            }
            signal(5, sig_dump);  // SIGTRAP
            signal(11, sig_dump); // SIGSEGV
            signal(6, sig_dump);  // SIGABRT
        }
    });
}
