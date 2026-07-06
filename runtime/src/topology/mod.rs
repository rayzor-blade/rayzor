//! CPU topology + thread-affinity primitives.
//!
//! Seven `extern "C"` symbols, mapped to the Haxe
//! `rayzor.concurrent.CpuTopology` static methods. `WorkerPool` (the
//! pure-Haxe pool) is built on top of these — see
//! `compiler/haxe-std/rayzor/concurrent/WorkerPool.hx`.
//!
//! ## Platform support
//!
//! | Platform | Topology source                                                 | Affinity                       |
//! |----------|-----------------------------------------------------------------|--------------------------------|
//! | Linux    | `/sys/devices/system/node/`                                     | `pthread_setaffinity_np`       |
//! | macOS    | Single node always (UMA — M1/M2/Intel laptop)                   | `thread_policy_set` (soft)     |
//! | Windows  | `GetNumaHighestNodeNumber` + `GetLogicalProcessorInformationEx` | `SetThreadGroupAffinity`       |
//! | other    | Degenerate stub: 1 node, all CPUs on it                         | No-op success                  |
//!
//! ## Calling-convention contract
//!
//! All functions are C-ABI and signal failure with sentinel return values
//! (never panic). Callers detect "this platform has no multi-node NUMA"
//! via `rayzor_topology_multi_node` returning `false`, and skip pinning
//! entirely.

#![allow(unsafe_op_in_unsafe_fn)]

mod platform;

// ---------------------------------------------------------------------------
// Public C-ABI surface — these are what the Haxe `CpuTopology` static
// methods route to via the runtime mapping table in
// `compiler/src/stdlib/runtime_mapping.rs`.
// ---------------------------------------------------------------------------

/// `true` iff the runtime discovered a multi-node NUMA topology.
///
/// Returns `false` on macOS, wasm, single-socket Linux, and Windows
/// systems with one NUMA node. The Haxe `WorkerPool` uses this to skip
/// affinity calls entirely on systems where they would be no-ops.
#[no_mangle]
pub extern "C" fn rayzor_topology_multi_node() -> bool {
    platform::available()
}

/// Number of NUMA nodes in the system. Always `>= 1`.
#[no_mangle]
pub extern "C" fn rayzor_topology_node_count() -> i32 {
    platform::node_count()
}

/// Total logical CPU count. Always `>= 1`.
#[no_mangle]
pub extern "C" fn rayzor_topology_cpu_count() -> i32 {
    platform::cpu_count()
}

/// Physical performance-core count on hybrid (big.LITTLE) parts. Falls
/// back to the logical CPU count when no hybrid split is exposed.
#[no_mangle]
pub extern "C" fn rayzor_topology_perf_core_count() -> i32 {
    platform::perf_core_count()
}

/// System-derived default for the guest SpinPool's CPU spin-wait relax
/// hint (PAUSE/YIELD each idle spin). 1 = emit the hint, 0 = bare spin.
/// The decision is purely a property of the platform, so it lives here
/// (one place) rather than a hand-set env: a bare spin loop wastes power
/// and heats the core on x86 and on thermally-tight ARM boards, but on a
/// thermally-generous Apple M-series desktop/laptop the hint's marginal
/// idle-branch cost is not worth it. `RAYZOR_HAXE_POOL_RELAX` overrides.
#[no_mangle]
pub extern "C" fn rayzor_pool_relax_default() -> i32 {
    // Always on. Emitting PAUSE/YIELD each idle spin is universally good in
    // a busy-wait: it drops core power, so less heat and less throttling —
    // which even a thermally-generous M-series hits under sustained serving
    // (measured: relax-on held drift +3.7 vs -17.4 off on M1, floor 68 vs
    // 57), and on x86 it also avoids memory-order machine-clears. Lowered
    // to a single inline instruction, so effectively free when idle-branch.
    1
}

/// Calibrate how many `spin_loop()` iterations run per microsecond on THIS
/// core — the whole reason a fixed iteration budget is meaningless across
/// CPUs (2000 iters is a different wall-time on M1 vs an i5).
fn spin_iters_per_us() -> f64 {
    use std::time::Instant;
    let n: u64 = 2_000_000;
    let t = Instant::now();
    let mut acc: u64 = 0;
    for i in 0..n {
        std::hint::spin_loop();
        acc = acc.wrapping_add(i);
    }
    std::hint::black_box(acc);
    let us = t.elapsed().as_nanos() as f64 / 1000.0;
    if us > 0.0 {
        n as f64 / us
    } else {
        100.0
    }
}

/// Measure the park -> unpark -> resume round-trip latency (median of a few
/// samples). This is the "cost of blocking" side of the spin-vs-block
/// break-even.
fn park_wake_latency_us() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    let mut samples: Vec<f64> = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let resumed_ns = Arc::new(AtomicU64::new(0));
        let r2 = resumed_ns.clone();
        let th = std::thread::spawn(move || {
            std::thread::park();
            r2.store(start.elapsed().as_nanos() as u64, Ordering::Release);
        });
        // Let it reach the park before we wake it.
        std::thread::sleep(Duration::from_micros(500));
        let unpark_ns = start.elapsed().as_nanos() as u64;
        th.thread().unpark();
        let _ = th.join();
        let resumed = resumed_ns.load(Ordering::Acquire);
        if resumed > unpark_ns {
            samples.push((resumed - unpark_ns) as f64 / 1000.0);
        }
    }
    if samples.is_empty() {
        return 20.0; // conservative fallback if measurement was too noisy
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// System-DERIVED (not hardcoded) tight-spin budget: keep spinning only
/// while spinning is cheaper than a park+wake cycle, then block so the core
/// can drop frequency. budget_iters = (iters/us on this core) x (park->wake
/// us) — the classic adaptive-spin break-even, with the CPU-speed term
/// measured so it is meaningful on any machine. Computed once (OnceLock);
/// `RAYZOR_HAXE_POOL_SPINS` still overrides. Clamped to a sane band.
#[no_mangle]
pub extern "C" fn rayzor_pool_spin_default() -> i32 {
    use std::sync::OnceLock;
    static CACHED: OnceLock<i32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let iters_per_us = spin_iters_per_us();
        let park_wake_us = park_wake_latency_us();
        let budget = (iters_per_us * park_wake_us).round();
        (budget as i64).clamp(200, 50_000) as i32
    })
}

/// NUMA node a given logical CPU belongs to. Returns `0` on no-NUMA
/// platforms; returns `-1` if `cpu` is out of range.
#[no_mangle]
pub extern "C" fn rayzor_topology_cpu_to_node(cpu: i32) -> i32 {
    platform::cpu_to_node(cpu)
}

/// Fill `out_buf` with up to `max` logical CPU IDs that belong to `node`.
/// Returns the number of CPU IDs written.
///
/// # Safety
/// `out_buf` must point to writable memory for at least `max` `i32` slots,
/// or be null when `max == 0`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_topology_node_cpus(node: i32, out_buf: *mut i32, max: i32) -> i32 {
    if max < 0 {
        return -1;
    }
    if max > 0 && out_buf.is_null() {
        return -1;
    }
    let slice = if max == 0 {
        &mut [][..]
    } else {
        std::slice::from_raw_parts_mut(out_buf, max as usize)
    };
    platform::node_cpus(node, slice)
}

/// Pin the calling thread to all CPUs on `node`.
///
/// Returns `0` on success (including no-NUMA platforms where the call
/// is a no-op), `-1` if the platform doesn't support thread affinity,
/// `-2` if `node` is out of range.
#[no_mangle]
pub extern "C" fn rayzor_topology_bind_to_node(node: i32) -> i32 {
    platform::bind_current_thread(node)
}

/// Clear any affinity hint on the calling thread.
#[no_mangle]
pub extern "C" fn rayzor_topology_unbind() -> i32 {
    platform::unbind_current_thread()
}

// ---------------------------------------------------------------------------
// Tests — exercise the platform impl this build targets.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_count_is_at_least_one() {
        assert!(rayzor_topology_node_count() >= 1);
    }

    #[test]
    fn cpu_count_is_at_least_one() {
        assert!(rayzor_topology_cpu_count() >= 1);
    }

    #[test]
    fn every_cpu_maps_to_a_valid_node() {
        let nodes = rayzor_topology_node_count();
        let cpus = rayzor_topology_cpu_count();
        for cpu in 0..cpus {
            let node = rayzor_topology_cpu_to_node(cpu);
            assert!((0..nodes).contains(&node));
        }
    }

    #[test]
    fn out_of_range_cpu_returns_negative() {
        assert!(rayzor_topology_cpu_to_node(-1) < 0);
        assert!(rayzor_topology_cpu_to_node(99_999) < 0);
    }

    #[test]
    fn node_cpus_fills_buffer_on_node_zero() {
        let cpus = rayzor_topology_cpu_count();
        let mut buf = vec![-1i32; cpus as usize];
        let n = unsafe { rayzor_topology_node_cpus(0, buf.as_mut_ptr(), cpus) };
        assert!(n >= 1);
    }

    #[test]
    fn bind_rejects_out_of_range_node() {
        let nodes = rayzor_topology_node_count();
        let r = rayzor_topology_bind_to_node(nodes + 100);
        assert_eq!(r, -2);
    }
}
