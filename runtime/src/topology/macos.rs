//! macOS implementation.
//!
//! Apple does not expose NUMA topology to userland (even on multi-socket Mac
//! Pros the kernel hides the affinity story behind the scheduler). We report
//! one node containing every logical CPU.
//!
//! For affinity we use the `THREAD_AFFINITY_POLICY` policy via
//! `thread_policy_set` — this is a *hint* (not enforced), but it lets the
//! kernel cluster threads onto cores that share an L2/L3, which is the
//! macOS-equivalent of the locality win NUMA provides on Linux.
//!
//! Each NUMA node maps to a distinct affinity tag (`node + 1`, so we never
//! pass the magic tag 0 = `THREAD_AFFINITY_TAG_NULL`). Since `node_count()`
//! is always 1 here, all threads end up on tag `1` — but the wiring is
//! ready for the day Apple ships multi-socket ARM.

use libc::{c_int, c_uint};
use std::os::raw::c_void;

const THREAD_AFFINITY_POLICY: c_int = 4;
const THREAD_AFFINITY_TAG_NULL: c_int = 0;

#[repr(C)]
struct ThreadAffinityPolicy {
    affinity_tag: c_int,
}

// Declared in <mach/thread_act.h>; not exposed by the libc crate, so we
// bring them in directly.
extern "C" {
    fn mach_thread_self() -> c_uint;
    fn thread_policy_set(
        thread: c_uint,
        flavor: c_int,
        policy_info: *mut c_void,
        count: c_uint,
    ) -> c_int;
}

const KERN_SUCCESS: c_int = 0;
const THREAD_AFFINITY_POLICY_COUNT: c_uint =
    (std::mem::size_of::<ThreadAffinityPolicy>() / std::mem::size_of::<c_int>()) as c_uint;

pub(super) fn available() -> bool {
    false
}

pub(super) fn node_count() -> i32 {
    1
}

pub(super) fn cpu_count() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1)
        .max(1)
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

pub(super) fn bind_current_thread(node: i32) -> i32 {
    // Apply a non-null affinity tag so the kernel groups threads with the
    // same tag onto shared-cache cores. Tag is node+1 to avoid the magic 0.
    let mut policy = ThreadAffinityPolicy {
        affinity_tag: node + 1,
    };
    // SAFETY: thread_policy_set is the documented mach API for this op;
    // policy is stack-allocated and outlives the call.
    let rc = unsafe {
        thread_policy_set(
            mach_thread_self(),
            THREAD_AFFINITY_POLICY,
            &mut policy as *mut _ as *mut c_void,
            THREAD_AFFINITY_POLICY_COUNT,
        )
    };
    if rc == KERN_SUCCESS {
        0
    } else {
        -1
    }
}

pub(super) fn unbind_current_thread() -> i32 {
    let mut policy = ThreadAffinityPolicy {
        affinity_tag: THREAD_AFFINITY_TAG_NULL,
    };
    let rc = unsafe {
        thread_policy_set(
            mach_thread_self(),
            THREAD_AFFINITY_POLICY,
            &mut policy as *mut _ as *mut c_void,
            THREAD_AFFINITY_POLICY_COUNT,
        )
    };
    if rc == KERN_SUCCESS {
        0
    } else {
        -1
    }
}
