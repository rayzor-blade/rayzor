//! WebGPU Shading Language (WGSL) code generation.
//!
//! Generates WGSL compute shader source strings for elementwise operations.
//! Each generated kernel uses `@compute @workgroup_size(256)` with buffer
//! bindings at `@group(0) @binding(N)`.

use crate::buffer;
use crate::kernel_ir::KernelOp;

/// Default workgroup size for elementwise kernels.
pub const WORKGROUP_SIZE: u32 = 256;

/// Map a dtype tag to the corresponding WGSL type string.
pub fn dtype_to_wgsl(dtype: u8) -> &'static str {
    match dtype {
        buffer::DTYPE_F32 => "f32",
        buffer::DTYPE_F64 => "f32", // WGSL has no f64; fall back to f32
        buffer::DTYPE_I32 => "i32",
        buffer::DTYPE_I64 => "i32", // WGSL has no i64; fall back to i32
        _ => "f32",
    }
}

/// Returns the WGSL kernel function name for a given op and dtype.
pub fn kernel_fn_name(op: KernelOp, dtype: u8) -> String {
    if op == KernelOp::Matmul {
        return super::wgsl_matmul::matmul_fn_name(dtype);
    }
    if op == KernelOp::BatchMatmul {
        return super::wgsl_matmul::batch_matmul_fn_name(dtype);
    }
    format!("rayzor_{}_{}", op.name(), dtype_to_wgsl(dtype))
}

/// Number of buffer bindings a kernel needs (inputs + output + optional uniforms).
pub fn kernel_num_buffers(op: KernelOp) -> usize {
    if op.is_reduction() {
        3 // input, output, numel uniform
    } else if matches!(op, KernelOp::Matmul | KernelOp::BatchMatmul) {
        4 // A, B, C, dims uniform
    } else {
        op.input_count() + 1 // inputs + result
    }
}

/// Generate WGSL source for a binary elementwise operation.
pub fn emit_binary_elementwise(op: KernelOp, dtype: u8) -> String {
    let wgsl_type = dtype_to_wgsl(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Add => "a[id] + b[id]",
        KernelOp::Sub => "a[id] - b[id]",
        KernelOp::Mul => "a[id] * b[id]",
        KernelOp::Div => "a[id] / b[id]",
        _ => unreachable!("not a binary op"),
    };

    format!(
        r#"@group(0) @binding(0) var<storage, read> a: array<{wgsl_type}>;
@group(0) @binding(1) var<storage, read> b: array<{wgsl_type}>;
@group(0) @binding(2) var<storage, read_write> result: array<{wgsl_type}>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn {fn_name}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let id = gid.x;
    if (id >= arrayLength(&a)) {{
        return;
    }}
    result[id] = {op_expr};
}}
"#
    )
}

/// Generate WGSL source for a unary elementwise operation.
pub fn emit_unary_elementwise(op: KernelOp, dtype: u8) -> String {
    let wgsl_type = dtype_to_wgsl(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Neg => "-a[id]".to_string(),
        KernelOp::Abs => "abs(a[id])".to_string(),
        KernelOp::Sqrt => "sqrt(a[id])".to_string(),
        KernelOp::Exp => "exp(a[id])".to_string(),
        KernelOp::Log => "log(a[id])".to_string(),
        KernelOp::Relu => format!("max({wgsl_type}(0), a[id])"),
        KernelOp::Sigmoid => format!("1.0 / (1.0 + exp(-a[id]))"),
        KernelOp::Tanh => "tanh(a[id])".to_string(),
        KernelOp::Gelu => {
            format!("a[id] * 0.5 * (1.0 + tanh(0.7978845608 * (a[id] + 0.044715 * a[id] * a[id] * a[id])))")
        }
        KernelOp::Silu => format!("a[id] / (1.0 + exp(-a[id]))"),
        _ => unreachable!("not a unary op"),
    };

    format!(
        r#"@group(0) @binding(0) var<storage, read> a: array<{wgsl_type}>;
@group(0) @binding(1) var<storage, read_write> result: array<{wgsl_type}>;

@compute @workgroup_size({WORKGROUP_SIZE})
fn {fn_name}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let id = gid.x;
    if (id >= arrayLength(&a)) {{
        return;
    }}
    result[id] = {op_expr};
}}
"#
    )
}

/// Generate WGSL source for any kernel op.
pub fn emit_kernel(op: KernelOp, dtype: u8) -> String {
    if op.is_reduction() {
        return super::wgsl_reduction::emit_reduction(op, dtype);
    }
    if op == KernelOp::Matmul {
        return super::wgsl_matmul::emit_matmul(dtype);
    }
    if op == KernelOp::BatchMatmul {
        return super::wgsl_matmul::emit_batch_matmul(dtype);
    }
    match op.input_count() {
        2 => emit_binary_elementwise(op, dtype),
        1 => emit_unary_elementwise(op, dtype),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let src = emit_binary_elementwise(KernelOp::Add, buffer::DTYPE_F32);
        assert!(src.contains("fn rayzor_add_f32"));
        assert!(src.contains("var<storage, read> a: array<f32>"));
        assert!(src.contains("var<storage, read> b: array<f32>"));
        assert!(src.contains("var<storage, read_write> result: array<f32>"));
        assert!(src.contains("a[id] + b[id]"));
    }

    #[test]
    fn test_binary_mul_i32() {
        let src = emit_binary_elementwise(KernelOp::Mul, buffer::DTYPE_I32);
        assert!(src.contains("fn rayzor_mul_i32"));
        assert!(src.contains("var<storage, read> a: array<i32>"));
        assert!(src.contains("a[id] * b[id]"));
    }

    #[test]
    fn test_unary_sqrt_f32() {
        let src = emit_unary_elementwise(KernelOp::Sqrt, buffer::DTYPE_F32);
        assert!(src.contains("fn rayzor_sqrt_f32"));
        assert!(src.contains("var<storage, read> a: array<f32>"));
        assert!(src.contains("sqrt(a[id])"));
    }

    #[test]
    fn test_unary_relu_f32() {
        let src = emit_unary_elementwise(KernelOp::Relu, buffer::DTYPE_F32);
        assert!(src.contains("max(f32(0), a[id])"));
    }

    #[test]
    fn test_emit_kernel_dispatches() {
        let src = emit_kernel(KernelOp::Add, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_add_f32"));

        let src = emit_kernel(KernelOp::Exp, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_exp_f32"));
    }

    #[test]
    fn test_all_ops_generate_valid_wgsl() {
        let ops = [
            KernelOp::Add,
            KernelOp::Sub,
            KernelOp::Mul,
            KernelOp::Div,
            KernelOp::Neg,
            KernelOp::Abs,
            KernelOp::Sqrt,
            KernelOp::Exp,
            KernelOp::Log,
            KernelOp::Relu,
        ];

        for op in ops {
            let src = emit_kernel(op, buffer::DTYPE_F32);
            assert!(
                src.contains("@compute @workgroup_size("),
                "op {:?} missing workgroup_size",
                op
            );
            assert!(
                src.contains("global_invocation_id"),
                "op {:?} missing thread id",
                op
            );
        }
    }

    #[test]
    fn test_kernel_num_buffers() {
        assert_eq!(kernel_num_buffers(KernelOp::Add), 3);
        assert_eq!(kernel_num_buffers(KernelOp::Neg), 2);
        assert_eq!(kernel_num_buffers(KernelOp::ReduceSum), 3);
        assert_eq!(kernel_num_buffers(KernelOp::Matmul), 4);
    }
}
