//! MSL code generation for reduction kernels (sum, max, min).
//!
//! Reductions use threadgroup shared memory and a two-pass strategy:
//! - Pass 1: Each threadgroup reduces a chunk via grid-stride loop + tree reduction
//! - Pass 2: Single threadgroup reduces partial results from pass 1
//!
//! The same compiled kernel is used for both passes (numel is a buffer parameter).

use crate::buffer;
use crate::kernel_ir::KernelOp;

use super::msl::dtype_to_msl;

/// Fixed threadgroup size for all reduction kernels.
pub const REDUCE_THREADGROUP_SIZE: usize = 256;

/// Generate MSL source for a reduction kernel.
pub fn emit_reduction(op: KernelOp, dtype: u8) -> String {
    let msl_type = dtype_to_msl(dtype);
    let fn_name = format!("rayzor_{}_{}", op.name(), msl_type);

    let (identity, accumulate, combine) = match op {
        KernelOp::ReduceSum => (
            "0",
            "acc = acc + input[i]",
            "shared_data[tid] = shared_data[tid] + shared_data[tid + s]",
        ),
        KernelOp::ReduceMax => {
            let id = match dtype {
                buffer::DTYPE_F32 => "-INFINITY",
                buffer::DTYPE_I32 => "-2147483647",
                _ => "-INFINITY",
            };
            (
                id,
                "acc = max(acc, input[i])",
                "shared_data[tid] = max(shared_data[tid], shared_data[tid + s])",
            )
        }
        KernelOp::ReduceMin => {
            let id = match dtype {
                buffer::DTYPE_F32 => "INFINITY",
                buffer::DTYPE_I32 => "2147483647",
                _ => "INFINITY",
            };
            (
                id,
                "acc = min(acc, input[i])",
                "shared_data[tid] = min(shared_data[tid], shared_data[tid + s])",
            )
        }
        _ => unreachable!("not a reduction op"),
    };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {fn_name}(
    device const {msl_type}* input [[buffer(0)]],
    device {msl_type}* output [[buffer(1)]],
    constant uint& numel [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint num_tgs [[threadgroups_per_grid]]
) {{
    threadgroup {msl_type} shared_data[{REDUCE_THREADGROUP_SIZE}];

    {msl_type} acc = {msl_type}({identity});
    uint stride = tg_size * num_tgs;
    for (uint i = gid; i < numel; i += stride) {{
        {accumulate};
    }}

    shared_data[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            {combine};
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid == 0) {{
        output[tgid] = shared_data[0];
    }}
}}
"#
    )
}

/// Kernel function name for a reduction.
pub fn reduction_fn_name(op: KernelOp, dtype: u8) -> String {
    format!("rayzor_{}_{}", op.name(), dtype_to_msl(dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum_f32() {
        let src = emit_reduction(KernelOp::ReduceSum, buffer::DTYPE_F32);
        assert!(src.contains("kernel void rayzor_reduce_sum_float"));
        assert!(src.contains("threadgroup float shared_data[256]"));
        assert!(src.contains("float acc = float(0)"));
        assert!(src.contains("constant uint& numel"));
        assert!(src.contains("threadgroup_barrier"));
    }

    #[test]
    fn test_reduce_max_f32() {
        let src = emit_reduction(KernelOp::ReduceMax, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_reduce_max_float"));
        assert!(src.contains("float(-INFINITY)"));
        assert!(src.contains("max("));
    }

    #[test]
    fn test_reduce_min_i32() {
        let src = emit_reduction(KernelOp::ReduceMin, buffer::DTYPE_I32);
        assert!(src.contains("rayzor_reduce_min_int"));
        assert!(src.contains("int(2147483647)"));
        assert!(src.contains("min("));
    }
}
