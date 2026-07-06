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
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        0
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        1
    }
}

/// System-derived default for the guest SpinPool tight-spin budget
/// (iterations before the yield-hold -> park descent). Throttle-prone
/// parts (x86, ARM boards) park sooner so idle cores can drop frequency;
/// M-series spins longer because park-wake latency, not power, is the
/// cost there. `RAYZOR_HAXE_POOL_SPINS` overrides. Iteration counts (not
/// wall-time) so the ratio, not the absolute, is what matters here.
#[no_mangle]
pub extern "C" fn rayzor_pool_spin_default() -> i32 {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        2000
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        800
    }
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
