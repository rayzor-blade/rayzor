//! Lazy evaluation for GPU kernel fusion.
//!
//! Instead of dispatching a separate kernel for each elementwise op,
//! lazy evaluation builds a DAG of pending operations. When materialization
//! is triggered (by `toTensor`, a reduction, or matmul), the DAG is fused
//! into a single kernel that performs all operations in one dispatch.
//!
//! Example: `gpu.relu(gpu.add(a, b).mul(c))` builds:
//! ```text
//! Relu(Mul(Add(Input(a), Input(b)), Input(c)))
//! ```
//! Which materializes as a single kernel:
//! ```metal
//! result[id] = max(0.0, (in0[id] + in1[id]) * in2[id]);
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

#[cfg(feature = "native")]
use crate::backend::NativeBuffer;
#[cfg(feature = "native")]
use crate::kernel_ir::KernelOp;

/// A node in the lazy computation DAG.
///
/// Input nodes hold a reference-counted NativeBuffer (keeping GPU memory alive
/// even if the original GpuBuffer is freed). Operation nodes compose their
/// inputs recursively via `Rc` for cheap sharing.
#[derive(Clone)]
pub enum LazyOp {
    /// Leaf: reference to an already-materialized GPU buffer.
    Input(Rc<NativeBuffer>),

    /// Unary elementwise operation.
    Unary { op: KernelOp, input: Rc<LazyOp> },

    /// Binary elementwise operation.
    Binary {
        op: KernelOp,
        lhs: Rc<LazyOp>,
        rhs: Rc<LazyOp>,
    },
}

/// Metadata for a lazy buffer.
pub struct LazyNode {
    pub op: Rc<LazyOp>,
    pub dtype: u8,
    pub numel: usize,
}

/// Compute a structural hash of a LazyOp tree.
///
/// Two trees with the same topology and operations but different input buffers
/// produce the same hash. This is used to cache compiled fused kernels — the
/// same op chain reuses the same compiled pipeline regardless of which buffers
/// are bound.
pub fn structural_hash(op: &LazyOp) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_op(op, &mut hasher);
    hasher.finish()
}

fn hash_op(op: &LazyOp, hasher: &mut impl Hasher) {
    match op {
        LazyOp::Input(_) => {
            0u8.hash(hasher);
        }
        LazyOp::Unary { op, input } => {
            1u8.hash(hasher);
            op.name().hash(hasher);
            hash_op(input, hasher);
        }
        LazyOp::Binary { op, lhs, rhs } => {
            2u8.hash(hasher);
            op.name().hash(hasher);
            hash_op(lhs, hasher);
            hash_op(rhs, hasher);
        }
    }
}

/// Result of `collect_inputs`: (input buffers, raw-ptr → binding index).
pub type CollectedInputs = (Vec<Rc<NativeBuffer>>, HashMap<usize, usize>);

/// Collect all unique Input buffer pointers from a LazyOp tree.
///
/// Returns the buffers in discovery order (left-to-right, depth-first),
/// deduplicating by `Rc` pointer identity. The returned indices map each
/// input buffer to its binding index.
pub fn collect_inputs(op: &LazyOp) -> CollectedInputs {
    let mut buffers: Vec<Rc<NativeBuffer>> = Vec::new();
    let mut ptr_to_idx: HashMap<usize, usize> = HashMap::new();
    collect_inputs_rec(op, &mut buffers, &mut ptr_to_idx);
    (buffers, ptr_to_idx)
}

fn collect_inputs_rec(
    op: &LazyOp,
    buffers: &mut Vec<Rc<NativeBuffer>>,
    ptr_to_idx: &mut HashMap<usize, usize>,
) {
    match op {
        LazyOp::Input(buf) => {
            let ptr = Rc::as_ptr(buf) as usize;
            if let std::collections::hash_map::Entry::Vacant(e) = ptr_to_idx.entry(ptr) {
                let idx = buffers.len();
                e.insert(idx);
                buffers.push(buf.clone());
            }
        }
        LazyOp::Unary { input, .. } => {
            collect_inputs_rec(input, buffers, ptr_to_idx);
        }
        LazyOp::Binary { lhs, rhs, .. } => {
            collect_inputs_rec(lhs, buffers, ptr_to_idx);
            collect_inputs_rec(rhs, buffers, ptr_to_idx);
        }
    }
}

/// Count the depth of a lazy op tree (for complexity estimation).
pub fn tree_depth(op: &LazyOp) -> usize {
    match op {
        LazyOp::Input(_) => 0,
        LazyOp::Unary { input, .. } => 1 + tree_depth(input),
        LazyOp::Binary { lhs, rhs, .. } => 1 + tree_depth(lhs).max(tree_depth(rhs)),
    }
}
