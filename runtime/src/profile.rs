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
            let e = map.entry(h.finish()).or_insert(SiteStat {
                sampled_count: 0,
                pcs: top,
            });
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
        let zero: libc::itimerval = unsafe { std::mem::zeroed() };
        unsafe { libc::setitimer(libc::ITIMER_PROF, &zero, std::ptr::null_mut()) };
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

/// Walks frame-pointer chain from `start_fp`, recording up to 6 PCs into
/// `out`. macOS arm64 ABI: each frame stores `[saved_fp, saved_lr]` at
/// the address pointed to by x29. Stops when fp becomes 0, unaligned,
/// or outside the conservative stack range (1 MB above where we
/// started). Returns the number of frames filled.
///
/// This bypasses libunwind: it works regardless of `.eh_frame` /
/// compact-unwind coverage, AS LONG AS the JIT code preserves frame
/// pointers (Cranelift's `preserve_frame_pointers = true` setting).
/// 8-slot variant for crash-handler use. Same algorithm.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn walk_fp_chain_8(initial_pc: u64, start_fp: u64, out: &mut [u64; 8]) -> usize {
    out[0] = initial_pc;
    let mut fp = start_fp;
    let stack_top_guess = fp.saturating_add(16 * 1024 * 1024);
    let mut i = 1usize;
    while i < out.len() {
        if fp == 0 || fp & 0xf != 0 || fp > stack_top_guess || fp < 0x1000 {
            break;
        }
        let saved_fp = *(fp as *const u64);
        let saved_lr = *((fp + 8) as *const u64);
        if saved_lr == 0 {
            break;
        }
        out[i] = saved_lr.saturating_sub(4);
        i += 1;
        if saved_fp <= fp {
            break;
        }
        fp = saved_fp;
    }
    i
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn walk_fp_chain(initial_pc: u64, start_fp: u64, out: &mut [u64; 6]) -> usize {
    out[0] = initial_pc;
    let mut fp = start_fp;
    let stack_top_guess = fp.saturating_add(16 * 1024 * 1024); // 16 MB upper
    let mut i = 1usize;
    while i < out.len() {
        if fp == 0 || fp & 0xf != 0 || fp > stack_top_guess || fp < 0x1000 {
            break;
        }
        let saved_fp = *(fp as *const u64);
        let saved_lr = *((fp + 8) as *const u64);
        if saved_lr == 0 {
            break;
        }
        // LR holds the address to RETURN to (next instruction). For a
        // sample PC we want the address of the CALL instruction, which
        // is lr - 4 on arm64 (4-byte instructions).
        out[i] = saved_lr.saturating_sub(4);
        i += 1;
        // Frame pointers should grow upward (toward stack top). If the
        // next FP is below the current one, the chain is bogus.
        if saved_fp <= fp {
            break;
        }
        fp = saved_fp;
    }
    i
}

#[cfg(target_arch = "aarch64")]
extern "C" fn sigprof_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    ctx: *mut libc::c_void,
) {
    if IN_GRAPH.with(|g| g.get()) {
        return;
    }
    IN_GRAPH.with(|g| g.set(true));
    let mut pcs64 = [0u64; 6];
    let n = unsafe {
        // ctx is *mut libc::ucontext_t. On macOS arm64 the libc crate
        // models it with uc_mcontext as an embedded struct; the PC + FP
        // live at __ss.__pc and __ss.__fp.
        let uc = ctx as *const libc::ucontext_t;
        if uc.is_null() {
            IN_GRAPH.with(|g| g.set(false));
            return;
        }
        let mc = &(*uc).uc_mcontext;
        // uc_mcontext is `*mut __darwin_mcontext64` per libc; deref it.
        if mc.is_null() {
            IN_GRAPH.with(|g| g.set(false));
            return;
        }
        let ss = &(**mc).__ss;
        walk_fp_chain(ss.__pc, ss.__fp, &mut pcs64)
    };
    let mut top = [0usize; 6];
    for i in 0..n.min(6) {
        top[i] = pcs64[i] as usize;
    }
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    top.hash(&mut h);
    if let Ok(mut g) = GRAPH_SITES.try_lock() {
        if let Some(map) = g.as_mut() {
            let e = map.entry(h.finish()).or_insert(SiteStat {
                sampled_count: 0,
                pcs: top,
            });
            e.sampled_count += 1;
        }
    }
    IN_GRAPH.with(|g| g.set(false));
}

#[cfg(not(target_arch = "aarch64"))]
extern "C" fn sigprof_handler(
    _sig: libc::c_int,
    _info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    // Fallback for non-aarch64: use backtrace::trace. Won't walk through
    // JIT frames without `.eh_frame` registration but at least captures
    // Rust frames.
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
    if let Ok(mut g) = GRAPH_SITES.try_lock() {
        if let Some(map) = g.as_mut() {
            let e = map.entry(h.finish()).or_insert(SiteStat {
                sampled_count: 0,
                pcs: top,
            });
            e.sampled_count += 1;
        }
    }
    IN_GRAPH.with(|g| g.set(false));
}

unsafe fn install_cpu_profiler(period_us: u64) {
    // SA_SIGINFO gives our handler the 3-arg signature
    // `(int, siginfo_t*, ucontext_t*)` instead of the legacy 1-arg one.
    // We need the ucontext to read the interrupted thread's PC + FP
    // (so our manual frame-pointer walker can start from the right
    // place; libunwind doesn't help in JIT frames).
    let mut sa: libc::sigaction = std::mem::zeroed();
    sa.sa_sigaction = sigprof_handler as usize;
    sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
    libc::sigemptyset(&mut sa.sa_mask);
    libc::sigaction(libc::SIGPROF, &sa, std::ptr::null_mut());

    let mut tv: libc::itimerval = std::mem::zeroed();
    tv.it_interval.tv_sec = (period_us / 1_000_000) as libc::time_t;
    tv.it_interval.tv_usec = (period_us % 1_000_000) as libc::suseconds_t;
    tv.it_value = tv.it_interval;
    libc::setitimer(libc::ITIMER_PROF, &tv, std::ptr::null_mut());
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

/// Arm the SIGPROF profiler explicitly. Callable from Haxe via the
/// extern declaration `@:native("rayzor_profile_start")`. Bypasses the
/// env-var path so users can place start/stop calls around a specific
/// region (e.g. inside `GenerationLoop.generate`). Period defaults to
/// 1000us; override via `RAYZOR_CPU_PROFILE_US` before calling.
///
/// Idempotent: calling twice has no extra effect, calling after env-var
/// based init is a no-op (CPU_PROFILE_ACTIVE flag).
#[no_mangle]
pub extern "C" fn rayzor_profile_start() {
    if CPU_PROFILE_ACTIVE.load(MemOrdering::Relaxed) == 1 {
        return;
    }
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
    unsafe { install_cpu_profiler(period) };
    eprintln!(
        "[cpu-profile] rayzor_profile_start: SIGPROF armed (period={}us)",
        period
    );
}

/// Disarm the SIGPROF profiler. Pair with `rayzor_profile_start`.
/// Subsequent calls to start re-arm with the same period; the
/// accumulated GRAPH_SITES table persists across start/stop windows.
#[no_mangle]
pub extern "C" fn rayzor_profile_stop() {
    if CPU_PROFILE_ACTIVE.load(MemOrdering::Relaxed) == 0 {
        return;
    }
    let zero: libc::itimerval = unsafe { std::mem::zeroed() };
    unsafe { libc::setitimer(libc::ITIMER_PROF, &zero, std::ptr::null_mut()) };
    CPU_PROFILE_ACTIVE.store(0, MemOrdering::Relaxed);
    eprintln!("[cpu-profile] rayzor_profile_stop: SIGPROF disarmed");
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
            // RAYZOR_CPU_PROFILE_DELAY_MS skips the first N ms of the run
            // so the compile-phase doesn't dominate the sample stream.
            // RAYZOR_CPU_PROFILE_DURATION_MS bounds how long sampling
            // stays armed; 0 / unset = forever. The combination lets a
            // user say "skip the first 2s of compile, then capture for
            // 30s of decode" without touching their Haxe code.
            let delay_ms = std::env::var("RAYZOR_CPU_PROFILE_DELAY_MS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            let duration_ms = std::env::var("RAYZOR_CPU_PROFILE_DURATION_MS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            {
                let mut g = GRAPH_SITES.lock().unwrap();
                if g.is_none() {
                    *g = Some(std::collections::HashMap::new());
                }
            }
            if delay_ms == 0 && duration_ms == 0 {
                install_cpu_profiler(period);
                eprintln!("[cpu-profile] SIGPROF profiler armed (period={}us)", period);
            } else {
                // Spawn a tiny scheduler thread. setitimer is per-thread on
                // some platforms, but on macOS+Linux ITIMER_PROF is per-
                // process and SIGPROF is delivered to whichever thread is
                // running — so it doesn't matter that this thread spawned
                // the timer.
                std::thread::spawn(move || {
                    if delay_ms > 0 {
                        eprintln!(
                            "[cpu-profile] arming SIGPROF in {}ms (period={}us{})",
                            delay_ms,
                            period,
                            if duration_ms > 0 {
                                format!(", duration={}ms", duration_ms)
                            } else {
                                String::new()
                            }
                        );
                        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
                    }
                    unsafe { install_cpu_profiler(period) };
                    eprintln!("[cpu-profile] SIGPROF profiler armed (period={}us)", period);
                    if duration_ms > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(duration_ms));
                        let zero: libc::itimerval = unsafe { std::mem::zeroed() };
                        unsafe { libc::setitimer(libc::ITIMER_PROF, &zero, std::ptr::null_mut()) };
                        CPU_PROFILE_ACTIVE.store(0, MemOrdering::Relaxed);
                        eprintln!(
                            "[cpu-profile] SIGPROF disarmed after {}ms duration",
                            duration_ms
                        );
                    }
                });
            }
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
                crate::tensor::rayzor_dump_tensor_alloc_stats();
            }
            atexit(dump_all);
            // SA_SIGINFO so the handler receives ucontext_t and we can
            // read the trapping thread's PC + FP directly. The legacy
            // signal(2) path only gives the signo, and `backtrace::trace`
            // from inside a Mach signal handler stops at _sigtramp on
            // macOS (no DWARF unwind info for the trampoline), so it
            // never reaches the actual crash frame.
            #[cfg(target_arch = "aarch64")]
            extern "C" fn sig_dump_si(
                sig: libc::c_int,
                _info: *mut libc::siginfo_t,
                ctx: *mut libc::c_void,
            ) {
                rayzor_dump_alloc_stats();
                rayzor_dump_alloc_graph();
                crate::tensor::rayzor_dump_tensor_alloc_stats();

                let mut pcs64 = [0u64; 8];
                let n = unsafe {
                    let uc = ctx as *const libc::ucontext_t;
                    if uc.is_null() {
                        0
                    } else {
                        let mc = &(*uc).uc_mcontext;
                        if mc.is_null() {
                            0
                        } else {
                            let ss = &(**mc).__ss;
                            walk_fp_chain_8(ss.__pc, ss.__fp, &mut pcs64)
                        }
                    }
                };

                #[repr(C)]
                struct DlInfo {
                    dli_fname: *const std::ffi::c_char,
                    dli_fbase: *mut std::ffi::c_void,
                    dli_sname: *const std::ffi::c_char,
                    dli_saddr: *mut std::ffi::c_void,
                }
                extern "C" {
                    fn dladdr(addr: *const std::ffi::c_void, info: *mut DlInfo) -> i32;
                }
                unsafe {
                    eprintln!("[sig{}-backtrace pid={}]", sig, std::process::id());
                    for pc in &pcs64[..n] {
                        let mut info: DlInfo = DlInfo {
                            dli_fname: std::ptr::null(),
                            dli_fbase: std::ptr::null_mut(),
                            dli_sname: std::ptr::null(),
                            dli_saddr: std::ptr::null_mut(),
                        };
                        let rc = dladdr(*pc as *const _, &mut info);
                        if rc != 0 && !info.dli_sname.is_null() {
                            let sname = std::ffi::CStr::from_ptr(info.dli_sname).to_string_lossy();
                            let off = (*pc as usize).saturating_sub(info.dli_saddr as usize);
                            let fname = if !info.dli_fname.is_null() {
                                let fp = std::ffi::CStr::from_ptr(info.dli_fname).to_string_lossy();
                                fp.rsplit('/').next().unwrap_or("?").to_string()
                            } else {
                                String::from("?")
                            };
                            eprintln!("  0x{:x}  {} +0x{:x}  ({})", pc, sname, off, fname);
                        } else {
                            eprintln!("  0x{:x}  (in JIT / unresolved)", pc);
                        }
                    }
                    extern "C" {
                        fn _exit(s: i32) -> !;
                    }
                    _exit(128 + sig);
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = sig_dump_si as usize;
                sa.sa_flags = libc::SA_SIGINFO;
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(5, &sa, std::ptr::null_mut()); // SIGTRAP
                libc::sigaction(11, &sa, std::ptr::null_mut()); // SIGSEGV
                libc::sigaction(6, &sa, std::ptr::null_mut()); // SIGABRT
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                extern "C" fn sig_dump(sig: i32) {
                    rayzor_dump_alloc_stats();
                    rayzor_dump_alloc_graph();
                    crate::tensor::rayzor_dump_tensor_alloc_stats();
                    unsafe {
                        extern "C" {
                            fn _exit(s: i32) -> !;
                        }
                        _exit(128 + sig);
                    }
                }
                signal(5, sig_dump);
                signal(11, sig_dump);
                signal(6, sig_dump);
            }
        }
    });
}
