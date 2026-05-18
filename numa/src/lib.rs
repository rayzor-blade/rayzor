//! # rayzor-numa
//!
//! NUMA topology + thread affinity primitives for Rayzor.
//!
//! This crate exposes seven `extern "C"` symbols. They are consumed by:
//! - The Haxe stdlib (`rayzor.concurrent.NumaTopology` / `NumaPool`) via the
//!   compiler's runtime mapping table.
//! - Any Rust code that wants topology queries directly (call via
//!   [`get_runtime_symbols`] or use [`rayzor_numa_*`](rayzor_numa_node_count)
//!   functions).
//!
//! ## Platform support
//!
//! | Platform | Topology source                            | Affinity                       |
//! |----------|--------------------------------------------|--------------------------------|
//! | Linux    | `/sys/devices/system/node/`                | `pthread_setaffinity_np`       |
//! | macOS    | No true NUMA — reports 1 node              | `thread_policy_set` (soft)     |
//! | Windows  | `GetNumaHighestNodeNumber` + `GetLogicalProcessorInformationEx` | `SetThreadGroupAffinity`       |
//! | other    | Degenerate stub: 1 node, all CPUs on it    | No-op success                  |
//!
//! ## Calling-convention contract
//!
//! All functions are C-ABI and signal failure with sentinel return values
//! (never panic). Callers can detect "this platform has no NUMA" via
//! [`rayzor_numa_available`] returning `false` and skip pinning entirely.
//!
//! ## v1 scope (Phase 1b-i)
//!
//! v1 ships topology + affinity only. Explicit node-local allocation
//! (`numa_alloc_onnode`-equivalent) is deferred — the Haxe `NumaPool` relies
//! on first-touch placement: each worker pinned to node N malloc()s its own
//! buffers, and the OS lands those pages on N's memory controller.

#![deny(unsafe_op_in_unsafe_fn)]

use rayzor_plugin::declare_native_methods;

mod platform;

// ---------------------------------------------------------------------------
// Plugin method descriptor table.
//
// Each entry maps a Haxe static method on `rayzor.concurrent.NumaTopology`
// to its runtime symbol. The compiler reads this table at plugin-load time
// and auto-registers the call sites — no manual MIR wrappers, no
// `runtime_mapping.rs` edits, no `plugin_impl.rs` edits. Same pattern as
// `rayzor-gpu`.
// ---------------------------------------------------------------------------

declare_native_methods! {
    NUMA_METHODS;
    // class,                                method,             kind,    symbol,                              params  => return
    "rayzor_concurrent_NumaTopology", "available",         static,  "rayzor_numa_available",             []      => Bool;
    "rayzor_concurrent_NumaTopology", "nodeCount",         static,  "rayzor_numa_node_count",            []      => I64;
    "rayzor_concurrent_NumaTopology", "cpuCount",          static,  "rayzor_numa_cpu_count",             []      => I64;
    "rayzor_concurrent_NumaTopology", "cpuToNode",         static,  "rayzor_numa_cpu_to_node",           [I64]   => I64;
    // node_cpus has a writable buffer parameter (Ptr to i32 slots) — Haxe
    // wraps it in a higher-level API on NumaTopology that hides the buffer.
    "rayzor_concurrent_NumaTopology", "_nodeCpusRaw",      static,  "rayzor_numa_node_cpus",             [I64, Ptr, I64] => I64;
    "rayzor_concurrent_NumaTopology", "bindCurrent",       static,  "rayzor_numa_bind_current_thread",   [I64]   => I64;
    "rayzor_concurrent_NumaTopology", "unbindCurrent",     static,  "rayzor_numa_unbind_current_thread", []      => I64;
}

// ---------------------------------------------------------------------------
// Public C-ABI surface
// ---------------------------------------------------------------------------

/// `true` iff the runtime discovered a multi-node NUMA topology.
///
/// Returns `false` on macOS, wasm, single-socket Linux, and Windows
/// systems with one NUMA node. The Haxe `NumaPool` uses this to skip
/// affinity calls entirely on systems where they would be no-ops.
#[no_mangle]
pub extern "C" fn rayzor_numa_available() -> bool {
    platform::available()
}

/// Number of NUMA nodes in the system. Always `>= 1`.
///
/// Returns `1` on platforms without NUMA. The matching CPU set for node 0
/// in that case is "all logical CPUs."
#[no_mangle]
pub extern "C" fn rayzor_numa_node_count() -> i32 {
    platform::node_count()
}

/// Total logical CPU count. Always `>= 1`.
#[no_mangle]
pub extern "C" fn rayzor_numa_cpu_count() -> i32 {
    platform::cpu_count()
}

/// NUMA node a given logical CPU belongs to.
///
/// Returns `0` on no-NUMA platforms. Returns `-1` if `cpu` is out of range.
#[no_mangle]
pub extern "C" fn rayzor_numa_cpu_to_node(cpu: i32) -> i32 {
    platform::cpu_to_node(cpu)
}

/// Fill `out_buf` with up to `max` logical CPU IDs that belong to `node`.
///
/// Returns the number of CPU IDs written (may be less than `max`). Returns
/// `-1` if `node` is out of range or `out_buf` is null with `max > 0`.
///
/// # Safety
/// `out_buf` must point to writable memory for at least `max` `i32` slots,
/// or be null when `max == 0`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_numa_node_cpus(node: i32, out_buf: *mut i32, max: i32) -> i32 {
    if max < 0 {
        return -1;
    }
    if max > 0 && out_buf.is_null() {
        return -1;
    }
    let slice = if max == 0 {
        &mut [][..]
    } else {
        // SAFETY: precondition above — out_buf is non-null and points to
        // `max` writable i32 slots when max > 0.
        unsafe { std::slice::from_raw_parts_mut(out_buf, max as usize) }
    };
    platform::node_cpus(node, slice)
}

/// Pin the calling thread to all CPUs on `node`.
///
/// Returns:
/// - `0` on success (including no-NUMA platforms where the call is a no-op).
/// - `-1` if the platform doesn't support thread affinity.
/// - `-2` if `node` is out of range.
#[no_mangle]
pub extern "C" fn rayzor_numa_bind_current_thread(node: i32) -> i32 {
    platform::bind_current_thread(node)
}

/// Clear any affinity hint on the calling thread (let it run anywhere).
///
/// Returns `0` on success (including no-NUMA platforms), `-1` if the platform
/// doesn't support unbinding.
#[no_mangle]
pub extern "C" fn rayzor_numa_unbind_current_thread() -> i32 {
    platform::unbind_current_thread()
}

// ---------------------------------------------------------------------------
// Rust-callable API for static linking into rayzor-runtime
// ---------------------------------------------------------------------------

/// Returns the table of `(symbol_name, function_pointer)` pairs that the
/// runtime registers into its symbol table.
///
/// `runtime::plugin_impl` merges this into its global symbol map alongside
/// stdlib symbols. Same shape as `rayzor_gpu::get_runtime_symbols()`.
pub fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("rayzor_numa_available", rayzor_numa_available as *const u8),
        (
            "rayzor_numa_node_count",
            rayzor_numa_node_count as *const u8,
        ),
        ("rayzor_numa_cpu_count", rayzor_numa_cpu_count as *const u8),
        (
            "rayzor_numa_cpu_to_node",
            rayzor_numa_cpu_to_node as *const u8,
        ),
        ("rayzor_numa_node_cpus", rayzor_numa_node_cpus as *const u8),
        (
            "rayzor_numa_bind_current_thread",
            rayzor_numa_bind_current_thread as *const u8,
        ),
        (
            "rayzor_numa_unbind_current_thread",
            rayzor_numa_unbind_current_thread as *const u8,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Universal rpkg entry point.
//
// Exported as `rayzor_rpkg_entry` — when the compiler dlopens the packaged
// `.rpkg` cdylib, it calls this single function to obtain both the runtime
// symbol table and the method descriptor table. The `rayzor-gpu` crate uses
// the same macro.
// ---------------------------------------------------------------------------

rayzor_plugin::rpkg_entry!(NUMA_METHODS, get_runtime_symbols);

/// Marker plugin type — mirrors `rayzor_gpu::GpuComputePlugin`. Implements the
/// `RuntimePlugin` trait so the inventory-based loader (when present) can
/// register this crate alongside dlopen'd packages.
pub struct NumaPlugin;

impl rayzor_plugin::RuntimePlugin for NumaPlugin {
    fn name(&self) -> &str {
        "rayzor_numa"
    }

    fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
        get_runtime_symbols()
    }
}

// ---------------------------------------------------------------------------
// Tests — exercise the platform impl that this build targets.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_count_is_at_least_one() {
        let n = rayzor_numa_node_count();
        assert!(n >= 1, "node count must be >= 1, got {n}");
    }

    #[test]
    fn cpu_count_is_at_least_one() {
        let n = rayzor_numa_cpu_count();
        assert!(n >= 1, "cpu count must be >= 1, got {n}");
    }

    #[test]
    fn every_cpu_maps_to_a_valid_node() {
        let nodes = rayzor_numa_node_count();
        let cpus = rayzor_numa_cpu_count();
        for cpu in 0..cpus {
            let node = rayzor_numa_cpu_to_node(cpu);
            assert!(
                (0..nodes).contains(&node),
                "cpu {cpu} mapped to node {node}, out of range [0, {nodes})"
            );
        }
    }

    #[test]
    fn out_of_range_cpu_returns_negative() {
        assert!(rayzor_numa_cpu_to_node(-1) < 0);
        assert!(rayzor_numa_cpu_to_node(99_999) < 0);
    }

    #[test]
    fn node_cpus_fills_buffer_on_node_zero() {
        let cpus = rayzor_numa_cpu_count();
        let mut buf = vec![-1i32; cpus as usize];
        let n = unsafe { rayzor_numa_node_cpus(0, buf.as_mut_ptr(), cpus) };
        assert!(n >= 1, "node 0 should own at least one CPU, got {n}");
        for &c in &buf[..n as usize] {
            assert!(
                (0..cpus).contains(&c),
                "node 0 CPU list contained {c}, out of [0, {cpus})"
            );
        }
    }

    #[test]
    fn node_cpus_rejects_invalid_args() {
        let mut buf = [0i32; 4];
        assert_eq!(
            unsafe { rayzor_numa_node_cpus(99_999, buf.as_mut_ptr(), 4) },
            -1
        );
        assert_eq!(
            unsafe { rayzor_numa_node_cpus(0, std::ptr::null_mut(), 4) },
            -1
        );
        // max == 0 with null buf is allowed (no write).
        assert!(unsafe { rayzor_numa_node_cpus(0, std::ptr::null_mut(), 0) } >= 0);
    }

    #[test]
    fn bind_unbind_round_trip_does_not_panic() {
        // On platforms without affinity, both calls return 0 or -1 — but they
        // must not panic, and unbind after bind must always succeed if bind did.
        let bind = rayzor_numa_bind_current_thread(0);
        let unbind = rayzor_numa_unbind_current_thread();
        if bind == 0 {
            assert!(unbind == 0 || unbind == -1);
        }
    }

    #[test]
    fn bind_rejects_out_of_range_node() {
        // node >= node_count() must yield -2 (invalid), not a crash.
        let nodes = rayzor_numa_node_count();
        let r = rayzor_numa_bind_current_thread(nodes + 100);
        assert_eq!(r, -2, "bind to invalid node should return -2, got {r}");
    }
}
