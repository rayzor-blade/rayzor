//! CUDA C code generation for GPU compute kernels.
//!
//! Generates CUDA kernel source strings that can be compiled via:
//! - NVRTC (runtime compilation on NVIDIA GPUs)
//! - TCC (CPU simulation with CUDA stub macros for testing)
//!
//! Each kernel follows standard CUDA conventions: `__global__` entry point,
//! `threadIdx`/`blockIdx`/`blockDim` for thread identification,
//! `__shared__` for threadgroup memory, `__syncthreads()` for barriers.

use crate::buffer;
use crate::kernel_ir::KernelOp;

/// Map a dtype tag to the corresponding CUDA C type string.
pub fn dtype_to_cuda(dtype: u8) -> &'static str {
    match dtype {
        buffer::DTYPE_F32 => "float",
        buffer::DTYPE_F64 => "double",
        buffer::DTYPE_I32 => "int",
        buffer::DTYPE_I64 => "long long",
        _ => "float",
    }
}

/// Returns the CUDA kernel function name for a given op and dtype.
pub fn kernel_fn_name(op: KernelOp, dtype: u8) -> String {
    if op == KernelOp::Matmul {
        return super::cuda_matmul::matmul_fn_name(dtype);
    }
    if op == KernelOp::BatchMatmul {
        return super::cuda_matmul::batch_matmul_fn_name(dtype);
    }
    format!("rayzor_{}_{}", op.name(), dtype_to_cuda(dtype).replace(' ', "_"))
}

/// Number of buffers a CUDA kernel needs.
pub fn kernel_num_buffers(op: KernelOp) -> usize {
    if op.is_reduction() {
        3 // input, output, numel
    } else if matches!(op, KernelOp::Matmul | KernelOp::BatchMatmul) {
        4 // A, B, C, dims
    } else {
        op.input_count() + 1
    }
}

/// Generate CUDA source for a binary elementwise operation.
pub fn emit_binary_elementwise(op: KernelOp, dtype: u8) -> String {
    let cuda_type = dtype_to_cuda(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Add => "a[id] + b[id]",
        KernelOp::Sub => "a[id] - b[id]",
        KernelOp::Mul => "a[id] * b[id]",
        KernelOp::Div => "a[id] / b[id]",
        _ => unreachable!("not a binary op"),
    };

    format!(
        r#"extern "C" __global__ void {fn_name}(
    const {cuda_type}* a,
    const {cuda_type}* b,
    {cuda_type}* result,
    unsigned int numel
) {{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numel) {{
        result[id] = {op_expr};
    }}
}}
"#
    )
}

/// Generate CUDA source for a unary elementwise operation.
pub fn emit_unary_elementwise(op: KernelOp, dtype: u8) -> String {
    let cuda_type = dtype_to_cuda(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Neg => "-a[id]".to_string(),
        KernelOp::Abs => format!("({cuda_type})fabs((double)a[id])"),
        KernelOp::Sqrt => format!("({cuda_type})sqrt((double)a[id])"),
        KernelOp::Exp => format!("({cuda_type})exp((double)a[id])"),
        KernelOp::Log => format!("({cuda_type})log((double)a[id])"),
        KernelOp::Relu => format!("a[id] > ({cuda_type})0 ? a[id] : ({cuda_type})0"),
        KernelOp::Sigmoid => format!("({cuda_type})(1.0 / (1.0 + exp(-(double)a[id])))"),
        KernelOp::Tanh => format!("({cuda_type})tanh((double)a[id])"),
        KernelOp::Gelu => {
            format!("({cuda_type})((double)a[id] * 0.5 * (1.0 + tanh(0.7978845608 * ((double)a[id] + 0.044715 * (double)a[id] * (double)a[id] * (double)a[id]))))")
        }
        KernelOp::Silu => format!("({cuda_type})((double)a[id] / (1.0 + exp(-(double)a[id])))"),
        _ => unreachable!("not a unary op"),
    };

    format!(
        r#"extern "C" __global__ void {fn_name}(
    const {cuda_type}* a,
    {cuda_type}* result,
    unsigned int numel
) {{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numel) {{
        result[id] = {op_expr};
    }}
}}
"#
    )
}

/// Generate CUDA source for any kernel op.
pub fn emit_kernel(op: KernelOp, dtype: u8) -> String {
    if op.is_reduction() {
        return super::cuda_reduction::emit_reduction(op, dtype);
    }
    if op == KernelOp::Matmul {
        return super::cuda_matmul::emit_matmul(dtype);
    }
    if op == KernelOp::BatchMatmul {
        return super::cuda_matmul::emit_batch_matmul(dtype);
    }
    match op.input_count() {
        2 => emit_binary_elementwise(op, dtype),
        1 => emit_unary_elementwise(op, dtype),
        _ => unreachable!(),
    }
}

/// CUDA stub macros for TCC compilation (CPU simulation).
/// Prepend this to kernel source before compiling with TCC.
pub const TCC_CUDA_STUBS: &str = r#"
#include <math.h>
#include <string.h>

// Stub CUDA qualifiers
#define __global__
#define __shared__ static
#define __device__

// Thread/block indices — set externally before each "thread" invocation
typedef struct { unsigned int x, y, z; } _dim3;
static _dim3 threadIdx, blockIdx, blockDim, gridDim;

// Barrier — no-op in single-threaded TCC simulation
#define __syncthreads()

// fma fallback
#ifndef fma
#define fma(a, b, c) ((a) * (b) + (c))
#endif
"#;

/// Wrap a CUDA kernel source with TCC stubs for CPU compilation.
/// Also wraps the kernel call in a C-callable runner function.
pub fn wrap_for_tcc(kernel_source: &str, fn_name: &str, num_inputs: usize) -> String {
    let mut runner_params = Vec::new();
    let mut kernel_args = Vec::new();

    for i in 0..num_inputs {
        runner_params.push(format!("const float* in{i}"));
        kernel_args.push(format!("in{i}"));
    }
    runner_params.push("float* result".to_string());
    kernel_args.push("result".to_string());
    runner_params.push("unsigned int numel".to_string());
    kernel_args.push("numel".to_string());

    format!(
        r#"{stubs}

// Strip extern "C" for TCC
#define extern

{kernel_source}

#undef extern

// Runner: simulates grid launch by iterating threads
void run_{fn_name}({params}, unsigned int block_size) {{
    unsigned int num_blocks = (numel + block_size - 1) / block_size;
    gridDim.x = num_blocks; gridDim.y = 1; gridDim.z = 1;
    blockDim.x = block_size; blockDim.y = 1; blockDim.z = 1;
    for (unsigned int b = 0; b < num_blocks; b++) {{
        blockIdx.x = b; blockIdx.y = 0; blockIdx.z = 0;
        for (unsigned int t = 0; t < block_size; t++) {{
            threadIdx.x = t; threadIdx.y = 0; threadIdx.z = 0;
            {fn_name}({args});
        }}
    }}
}}
"#,
        stubs = TCC_CUDA_STUBS,
        params = runner_params.join(", "),
        args = kernel_args.join(", "),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let src = emit_binary_elementwise(KernelOp::Add, buffer::DTYPE_F32);
        assert!(src.contains("__global__"));
        assert!(src.contains("rayzor_add_float"));
        assert!(src.contains("a[id] + b[id]"));
        assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
    }

    #[test]
    fn test_unary_relu_f32() {
        let src = emit_unary_elementwise(KernelOp::Relu, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_relu_float"));
        assert!(src.contains("a[id] >"));
    }

    #[test]
    fn test_unary_sigmoid_f32() {
        let src = emit_unary_elementwise(KernelOp::Sigmoid, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_sigmoid_float"));
        assert!(src.contains("exp("));
    }

    #[test]
    fn test_unary_gelu_f32() {
        let src = emit_unary_elementwise(KernelOp::Gelu, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_gelu_float"));
        assert!(src.contains("tanh("));
        assert!(src.contains("0.044715"));
    }

    #[test]
    fn test_all_ops_generate() {
        for op in [
            KernelOp::Add, KernelOp::Sub, KernelOp::Mul, KernelOp::Div,
            KernelOp::Neg, KernelOp::Abs, KernelOp::Sqrt, KernelOp::Exp,
            KernelOp::Log, KernelOp::Relu, KernelOp::Sigmoid, KernelOp::Tanh,
            KernelOp::Gelu, KernelOp::Silu,
        ] {
            let src = emit_kernel(op, buffer::DTYPE_F32);
            assert!(!src.is_empty(), "empty source for {:?}", op);
            assert!(src.contains("__global__"), "no __global__ for {:?}", op);
        }
    }

    #[test]
    fn test_tcc_wrap() {
        let src = emit_binary_elementwise(KernelOp::Add, buffer::DTYPE_F32);
        let fn_name = kernel_fn_name(KernelOp::Add, buffer::DTYPE_F32);
        let wrapped = wrap_for_tcc(&src, &fn_name, 2);
        assert!(wrapped.contains("#define __global__"));
        assert!(wrapped.contains("run_rayzor_add_float"));
        assert!(wrapped.contains("for (unsigned int b = 0;"));
        assert!(wrapped.contains("threadIdx.x = t;"));
    }
}
