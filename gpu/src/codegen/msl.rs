//! Metal Shading Language (MSL) code generation.
//!
//! Generates MSL kernel source strings for elementwise operations.
//! Each generated kernel follows the pattern:
//!
//! ```metal
//! kernel void rayzor_add_float(
//!     device const float* a [[buffer(0)]],
//!     device const float* b [[buffer(1)]],
//!     device float* result   [[buffer(2)]],
//!     uint id [[thread_position_in_grid]]
//! ) {
//!     result[id] = a[id] + b[id];
//! }
//! ```

use crate::buffer;
use crate::kernel_ir::KernelOp;

/// Map a dtype tag to the corresponding MSL type string.
pub fn dtype_to_msl(dtype: u8) -> &'static str {
    match dtype {
        buffer::DTYPE_F32 => "float",
        buffer::DTYPE_F64 => "double", // Note: requires Metal GPU Family 5+
        buffer::DTYPE_I32 => "int",
        buffer::DTYPE_I64 => "long",
        _ => "float",
    }
}

/// Returns the MSL kernel function name for a given op and dtype.
pub fn kernel_fn_name(op: KernelOp, dtype: u8) -> String {
    if op == KernelOp::Matmul {
        return super::msl_matmul::matmul_fn_name(dtype);
    }
    if op == KernelOp::BatchMatmul {
        return super::msl_matmul::batch_matmul_fn_name(dtype);
    }
    format!("rayzor_{}_{}", op.name(), dtype_to_msl(dtype))
}

/// Generate MSL source for a binary elementwise operation.
///
/// Produces: `result[id] = a[id] OP b[id]`
pub fn emit_binary_elementwise(op: KernelOp, dtype: u8) -> String {
    let msl_type = dtype_to_msl(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Add => "a[id] + b[id]",
        KernelOp::Sub => "a[id] - b[id]",
        KernelOp::Mul => "a[id] * b[id]",
        KernelOp::Div => "a[id] / b[id]",
        _ => unreachable!("not a binary op"),
    };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {fn_name}(
    device const {msl_type}* a [[buffer(0)]],
    device const {msl_type}* b [[buffer(1)]],
    device {msl_type}* result   [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {{
    result[id] = {op_expr};
}}
"#
    )
}

/// Generate MSL source for a unary elementwise operation.
///
/// Produces: `result[id] = OP(a[id])`
pub fn emit_unary_elementwise(op: KernelOp, dtype: u8) -> String {
    let msl_type = dtype_to_msl(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Neg => "-a[id]".to_string(),
        KernelOp::Abs => "abs(a[id])".to_string(),
        KernelOp::Sqrt => "sqrt(a[id])".to_string(),
        KernelOp::Exp => "exp(a[id])".to_string(),
        KernelOp::Log => "log(a[id])".to_string(),
        KernelOp::Relu => format!("max(({msl_type})0, a[id])"),
        KernelOp::Sigmoid => "1.0 / (1.0 + exp(-a[id]))".to_string(),
        KernelOp::Tanh => "tanh(a[id])".to_string(),
        KernelOp::Gelu => {
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            "a[id] * 0.5 * (1.0 + tanh(0.7978845608 * (a[id] + 0.044715 * a[id] * a[id] * a[id])))".to_string()
        }
        KernelOp::Silu => "a[id] / (1.0 + exp(-a[id]))".to_string(),
        _ => unreachable!("not a unary op"),
    };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {fn_name}(
    device const {msl_type}* a [[buffer(0)]],
    device {msl_type}* result   [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {{
    result[id] = {op_expr};
}}
"#
    )
}

/// Generate MSL source for any kernel op.
pub fn emit_kernel(op: KernelOp, dtype: u8) -> String {
    if op.is_reduction() {
        return super::msl_reduction::emit_reduction(op, dtype);
    }
    if op == KernelOp::Matmul {
        return super::msl_matmul::emit_matmul(dtype);
    }
    if op == KernelOp::BatchMatmul {
        return super::msl_matmul::emit_batch_matmul(dtype);
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
        assert!(src.contains("kernel void rayzor_add_float"));
        assert!(src.contains("device const float* a"));
        assert!(src.contains("device const float* b"));
        assert!(src.contains("device float* result"));
        assert!(src.contains("result[id] = a[id] + b[id]"));
    }

    #[test]
    fn test_binary_mul_i32() {
        let src = emit_binary_elementwise(KernelOp::Mul, buffer::DTYPE_I32);
        assert!(src.contains("kernel void rayzor_mul_int"));
        assert!(src.contains("device const int* a"));
        assert!(src.contains("result[id] = a[id] * b[id]"));
    }

    #[test]
    fn test_unary_sqrt_f32() {
        let src = emit_unary_elementwise(KernelOp::Sqrt, buffer::DTYPE_F32);
        assert!(src.contains("kernel void rayzor_sqrt_float"));
        assert!(src.contains("device const float* a"));
        assert!(src.contains("result[id] = sqrt(a[id])"));
    }

    #[test]
    fn test_unary_relu_f32() {
        let src = emit_unary_elementwise(KernelOp::Relu, buffer::DTYPE_F32);
        assert!(src.contains("result[id] = max((float)0, a[id])"));
    }

    #[test]
    fn test_emit_kernel_dispatches() {
        // Binary ops go through emit_binary_elementwise
        let src = emit_kernel(KernelOp::Add, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_add_float"));

        // Unary ops go through emit_unary_elementwise
        let src = emit_kernel(KernelOp::Exp, buffer::DTYPE_F32);
        assert!(src.contains("rayzor_exp_float"));
    }

    #[test]
    fn test_all_ops_generate_valid_msl() {
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
                src.contains("#include <metal_stdlib>"),
                "op {:?} missing header",
                op
            );
            assert!(src.contains("kernel void"), "op {:?} missing kernel", op);
            assert!(
                src.contains("thread_position_in_grid"),
                "op {:?} missing thread id",
                op
            );
        }
    }
}
