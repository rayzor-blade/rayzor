//! Degenerate single-node implementation.
//!
//! Used on wasm and any platform without a NUMA shim. Reports one node, owns
//! all CPUs, treats bind/unbind as no-op successes so callers can call them
//! unconditionally.

pub(super) fn available() -> bool {
    false
}

pub(super) fn node_count() -> i32 {
    1
}

pub(super) fn cpu_count() -> i32 {
    // std::thread::available_parallelism is best-effort and never panics.
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1)
        .max(1)
}

pub(super) fn perf_core_count() -> i32 {
    // No hybrid-core split exposed here — logical count is the best answer.
    cpu_count()
}

pub(super) fn cpu_to_node(_cpu: i32) -> i32 {
    0
}

pub(super) fn node_cpus(_node: i32, out: &mut [i32]) -> i32 {
    let total = cpu_count();
    let n = (out.len() as i32).min(total);
    for (i, slot) in out.iter_mut().enumerate().take(n as usize) {
        *slot = i as i32;
    }
    n
}

pub(super) fn bind_current_thread(_node: i32) -> i32 {
    0
}

pub(super) fn bind_current_thread_to_performance() -> i32 {
    0
}

pub(super) fn unbind_current_thread() -> i32 {
    0
}
