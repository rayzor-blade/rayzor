//! CUDA C code generation for reduction kernels.
//!
//! Two-pass tree reduction with shared memory, same strategy as MSL/WGSL:
//! Pass 1: Each block reduces a chunk → partial results
//! Pass 2: Single block reduces partials → final scalar

use super::cuda::dtype_to_cuda;
use crate::kernel_ir::KernelOp;

const BLOCK_SIZE: usize = 256;

/// Generate CUDA source for a reduction kernel.
pub fn emit_reduction(op: KernelOp, dtype: u8) -> String {
    let cuda_type = dtype_to_cuda(dtype);
    let fn_name = reduction_fn_name(op, dtype);

    let (identity, accumulate, combine) = match op {
        KernelOp::ReduceSum => (
            format!("({cuda_type})0"),
            "acc + input[i]".to_string(),
            "shared_data[tid] + shared_data[tid + s]".to_string(),
        ),
        KernelOp::ReduceMax => (
            "-1e30f".to_string(),
            "acc > input[i] ? acc : input[i]".to_string(),
            "shared_data[tid] > shared_data[tid + s] ? shared_data[tid] : shared_data[tid + s]"
                .to_string(),
        ),
        KernelOp::ReduceMin => (
            "1e30f".to_string(),
            "acc < input[i] ? acc : input[i]".to_string(),
            "shared_data[tid] < shared_data[tid + s] ? shared_data[tid] : shared_data[tid + s]"
                .to_string(),
        ),
        _ => unreachable!("not a reduction op"),
    };

    format!(
        r#"extern "C" __global__ void {fn_name}(
    const {cuda_type}* input,
    {cuda_type}* output,
    unsigned int numel
) {{
    __shared__ {cuda_type} shared_data[{BLOCK_SIZE}];

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    {cuda_type} acc = {identity};
    for (unsigned int i = gid; i < numel; i += stride) {{
        acc = {accumulate};
    }}

    shared_data[tid] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (tid < s) {{
            shared_data[tid] = {combine};
        }}
        __syncthreads();
    }}

    if (tid == 0) {{
        output[blockIdx.x] = shared_data[0];
    }}
}}
"#
    )
}

/// Kernel function name for reduction.
pub fn reduction_fn_name(op: KernelOp, dtype: u8) -> String {
    let type_str = dtype_to_cuda(dtype).replace(' ', "_");
    format!("rayzor_{}_{}", op.name(), type_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer;

    #[test]
    fn test_reduce_sum_f32() {
        let src = emit_reduction(KernelOp::ReduceSum, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_reduce_sum_float"));
        assert!(src.contains("__shared__"));
        assert!(src.contains("__syncthreads()"));
        assert!(src.contains("shared_data[tid] + shared_data[tid + s]"));
    }

    #[test]
    fn test_reduce_max_f32() {
        let src = emit_reduction(KernelOp::ReduceMax, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_reduce_max_float"));
    }

    #[test]
    fn test_reduce_min_f32() {
        let src = emit_reduction(KernelOp::ReduceMin, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_reduce_min_float"));
    }
}
