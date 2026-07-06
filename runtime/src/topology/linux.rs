//! Linux implementation.
//!
//! Topology is read from `/sys/devices/system/node/` once at first use and
//! cached. Affinity uses `pthread_setaffinity_np` with a CPU set built from
//! the node's CPU list.
//!
//! Why not libnuma: the topology query is a handful of file reads, and we
//! don't ship `numa_alloc_onnode` in v1 (first-touch handles placement when
//! the worker thread is pinned). Skipping libnuma keeps the dependency
//! surface tiny and the binary smaller — drop in libnuma later if explicit
//! page-level placement becomes necessary.

use libc::{cpu_set_t, pthread_self, pthread_setaffinity_np, CPU_SET, CPU_ZERO};
use std::fs;
use std::mem::{size_of, MaybeUninit};
use std::sync::OnceLock;

/// Topology snapshot — populated once on first use.
struct Topology {
    /// Number of NUMA nodes.
    nodes: i32,
    /// Total logical CPU count (== `cpu_to_node.len()`).
    cpus: i32,
    /// `cpu_to_node[i]` = the NUMA node that logical CPU `i` belongs to.
    cpu_to_node: Vec<i32>,
    /// `node_cpus[n]` = the sorted list of logical CPU IDs on node `n`.
    node_cpus: Vec<Vec<i32>>,
}

fn read_topology() -> Topology {
    // /sys/devices/system/node/possible looks like "0-1" or "0".
    // /sys/devices/system/node/nodeN/cpulist looks like "0-3,8-11".
    let node_root = "/sys/devices/system/node";

    let possible = fs::read_to_string(format!("{node_root}/possible"))
        .ok()
        .map(|s| parse_cpu_list(s.trim()))
        .unwrap_or_default();

    if possible.is_empty() {
        return stub_topology();
    }

    let nodes = (possible.iter().copied().max().unwrap_or(0) + 1) as i32;
    let mut node_cpus: Vec<Vec<i32>> = vec![Vec::new(); nodes as usize];

    for &n in &possible {
        let path = format!("{node_root}/node{n}/cpulist");
        let mut cpus = match fs::read_to_string(&path) {
            Ok(s) => parse_cpu_list(s.trim()),
            Err(_) => continue,
        };
        cpus.sort_unstable();
        node_cpus[n as usize] = cpus;
    }

    let max_cpu = node_cpus
        .iter()
        .flat_map(|v| v.iter().copied())
        .max()
        .unwrap_or(-1);
    let cpus = (max_cpu + 1).max(0);

    let mut cpu_to_node = vec![-1i32; cpus as usize];
    for (node_idx, list) in node_cpus.iter().enumerate() {
        for &c in list {
            if (c as usize) < cpu_to_node.len() {
                cpu_to_node[c as usize] = node_idx as i32;
            }
        }
    }

    // Any CPU we couldn't map gets node 0 — defensive, shouldn't happen
    // under a valid sysfs layout but avoids `-1` leaks downstream.
    for slot in &mut cpu_to_node {
        if *slot < 0 {
            *slot = 0;
        }
    }

    if cpus == 0 {
        return stub_topology();
    }

    Topology {
        nodes,
        cpus,
        cpu_to_node,
        node_cpus,
    }
}

fn stub_topology() -> Topology {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1)
        .max(1);
    Topology {
        nodes: 1,
        cpus,
        cpu_to_node: vec![0; cpus as usize],
        node_cpus: vec![(0..cpus).collect()],
    }
}

/// Parse a Linux cpulist string: "0-3,8-11,15" → [0,1,2,3,8,9,10,11,15].
fn parse_cpu_list(s: &str) -> Vec<i32> {
    let mut out = Vec::new();
    for chunk in s.split(',') {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = chunk.split_once('-') {
            if let (Ok(a), Ok(b)) = (lo.trim().parse::<i32>(), hi.trim().parse::<i32>()) {
                for v in a..=b {
                    out.push(v);
                }
            }
        } else if let Ok(v) = chunk.parse::<i32>() {
            out.push(v);
        }
    }
    out
}

fn topology() -> &'static Topology {
    static TOPO: OnceLock<Topology> = OnceLock::new();
    TOPO.get_or_init(read_topology)
}

// ---------------------------------------------------------------------------
// Platform API
// ---------------------------------------------------------------------------

pub(super) fn available() -> bool {
    topology().nodes > 1
}

pub(super) fn node_count() -> i32 {
    topology().nodes
}

pub(super) fn cpu_count() -> i32 {
    topology().cpus
}

pub(super) fn perf_core_count() -> i32 {
    // No hybrid-core split exposed here — logical count is the best answer.
    cpu_count()
}

pub(super) fn cpu_to_node(cpu: i32) -> i32 {
    let t = topology();
    if (0..t.cpus).contains(&cpu) {
        t.cpu_to_node[cpu as usize]
    } else {
        -1
    }
}

pub(super) fn node_cpus(node: i32, out: &mut [i32]) -> i32 {
    let t = topology();
    if !(0..t.nodes).contains(&node) {
        return -1;
    }
    let src = &t.node_cpus[node as usize];
    let n = src.len().min(out.len());
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = src[i];
    }
    n as i32
}

pub(super) fn bind_current_thread(node: i32) -> i32 {
    let t = topology();
    if !(0..t.nodes).contains(&node) {
        return -2;
    }
    let cpus = &t.node_cpus[node as usize];
    if cpus.is_empty() {
        return -1;
    }

    let mut set: cpu_set_t = unsafe { MaybeUninit::zeroed().assume_init() };
    unsafe { CPU_ZERO(&mut set) };
    for &c in cpus {
        if c >= 0 {
            unsafe { CPU_SET(c as usize, &mut set) };
        }
    }
    let rc = unsafe { pthread_setaffinity_np(pthread_self(), size_of::<cpu_set_t>(), &set) };
    if rc == 0 {
        0
    } else {
        -1
    }
}

pub(super) fn unbind_current_thread() -> i32 {
    // Restore affinity to every CPU the topology knows about. This is the
    // "let the scheduler do whatever it wants" state — different from the
    // post-bind state where the thread is pinned to one node's CPUs.
    let t = topology();
    let mut set: cpu_set_t = unsafe { MaybeUninit::zeroed().assume_init() };
    unsafe { CPU_ZERO(&mut set) };
    for c in 0..t.cpus {
        unsafe { CPU_SET(c as usize, &mut set) };
    }
    let rc = unsafe { pthread_setaffinity_np(pthread_self(), size_of::<cpu_set_t>(), &set) };
    if rc == 0 {
        0
    } else {
        -1
    }
}
