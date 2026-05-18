//! Windows implementation.
//!
//! Topology comes from `GetLogicalProcessorInformationEx(RelationNumaNode)`,
//! which returns one record per NUMA node with the GROUP_AFFINITY mask of
//! processors on that node. Affinity uses `SetThreadGroupAffinity`.
//!
//! Windows separates logical processors into *processor groups* (max 64
//! processors per group) — a thread's affinity is `(group, mask)`. For
//! systems with > 64 CPUs, a single NUMA node may span multiple groups; we
//! pick the first group's mask for binding. This matches what `ProcessLasso`
//! and other affinity tools do for portability and is the right call for
//! ML inference workloads where all node-pinned threads end up adjacent.

use std::mem::{size_of, MaybeUninit};
use std::ptr;
use std::sync::OnceLock;

use windows_sys::Win32::System::SystemInformation::{
    GetLogicalProcessorInformationEx, RelationNumaNode, GROUP_AFFINITY,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};
use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};

struct NodeInfo {
    /// Processor group + mask of processors on this node.
    /// We pick the first group only — see module comment.
    group: u16,
    mask: u64,
    /// Cached list of logical-CPU IDs on this node, dense (`group * 64 + bit`).
    cpus: Vec<i32>,
}

struct Topology {
    nodes: Vec<NodeInfo>,
    cpus: i32,
    cpu_to_node: Vec<i32>,
}

fn read_topology() -> Topology {
    // First call: query buffer size.
    let mut needed: u32 = 0;
    let _ =
        unsafe { GetLogicalProcessorInformationEx(RelationNumaNode, ptr::null_mut(), &mut needed) };
    if needed == 0 {
        return stub_topology();
    }

    let mut buf = vec![0u8; needed as usize];
    let ok = unsafe {
        GetLogicalProcessorInformationEx(
            RelationNumaNode,
            buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut needed,
        )
    };
    if ok == 0 {
        return stub_topology();
    }

    let mut nodes: Vec<NodeInfo> = Vec::new();
    let mut cursor = 0usize;
    while cursor < needed as usize {
        let rec_ptr =
            unsafe { buf.as_ptr().add(cursor) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX };
        let size = unsafe { (*rec_ptr).Size } as usize;
        if size == 0 || cursor + size > needed as usize {
            break;
        }
        // SAFETY: the OS just told us this record is the NumaNode union arm.
        let numa = unsafe { (*rec_ptr).Anonymous.NumaNode };
        let aff: GROUP_AFFINITY = numa.GroupMask[0];

        let mut cpus = Vec::new();
        for bit in 0..64 {
            if (aff.Mask as u64) & (1u64 << bit) != 0 {
                cpus.push((aff.Group as i32) * 64 + bit);
            }
        }

        nodes.push(NodeInfo {
            group: aff.Group,
            mask: aff.Mask as u64,
            cpus,
        });

        cursor += size;
    }

    if nodes.is_empty() {
        return stub_topology();
    }

    let max_cpu = nodes
        .iter()
        .flat_map(|n| n.cpus.iter().copied())
        .max()
        .unwrap_or(-1);
    let cpus = (max_cpu + 1).max(0);

    let mut cpu_to_node = vec![0i32; cpus as usize];
    for (idx, n) in nodes.iter().enumerate() {
        for &c in &n.cpus {
            if (c as usize) < cpu_to_node.len() {
                cpu_to_node[c as usize] = idx as i32;
            }
        }
    }

    Topology {
        nodes,
        cpus,
        cpu_to_node,
    }
}

fn stub_topology() -> Topology {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1)
        .max(1);
    let mask = if cpus >= 64 {
        !0u64
    } else {
        (1u64 << cpus) - 1
    };
    Topology {
        nodes: vec![NodeInfo {
            group: 0,
            mask,
            cpus: (0..cpus).collect(),
        }],
        cpus,
        cpu_to_node: vec![0; cpus as usize],
    }
}

fn topology() -> &'static Topology {
    static TOPO: OnceLock<Topology> = OnceLock::new();
    TOPO.get_or_init(read_topology)
}

// ---------------------------------------------------------------------------
// Platform API
// ---------------------------------------------------------------------------

pub(super) fn available() -> bool {
    topology().nodes.len() > 1
}

pub(super) fn node_count() -> i32 {
    topology().nodes.len() as i32
}

pub(super) fn cpu_count() -> i32 {
    topology().cpus
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
    if !(0..(t.nodes.len() as i32)).contains(&node) {
        return -1;
    }
    let src = &t.nodes[node as usize].cpus;
    let n = src.len().min(out.len());
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = src[i];
    }
    n as i32
}

pub(super) fn bind_current_thread(node: i32) -> i32 {
    let t = topology();
    if !(0..(t.nodes.len() as i32)).contains(&node) {
        return -2;
    }
    let n = &t.nodes[node as usize];
    let mut new_aff = GROUP_AFFINITY {
        Mask: n.mask as usize,
        Group: n.group,
        Reserved: [0; 3],
    };
    let mut prev: MaybeUninit<GROUP_AFFINITY> = MaybeUninit::zeroed();
    let ok = unsafe { SetThreadGroupAffinity(GetCurrentThread(), &mut new_aff, prev.as_mut_ptr()) };
    if ok != 0 {
        0
    } else {
        -1
    }
}

pub(super) fn unbind_current_thread() -> i32 {
    // Build a "wide-open" affinity by setting every bit on group 0. Multi-group
    // systems will be partially unbound (group 0 only); the caller can iterate
    // node_count() if they need full unbind on those rare boxes.
    let mask = if topology().cpus >= 64 {
        !0usize
    } else {
        ((1u64 << topology().cpus) - 1) as usize
    };
    let mut new_aff = GROUP_AFFINITY {
        Mask: mask,
        Group: 0,
        Reserved: [0; 3],
    };
    let mut prev: MaybeUninit<GROUP_AFFINITY> = MaybeUninit::zeroed();
    let ok = unsafe { SetThreadGroupAffinity(GetCurrentThread(), &mut new_aff, prev.as_mut_ptr()) };
    if ok != 0 {
        0
    } else {
        -1
    }
}

#[allow(dead_code)]
fn _ensure_size_of_used() -> usize {
    size_of::<GROUP_AFFINITY>()
}
