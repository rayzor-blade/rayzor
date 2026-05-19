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
///
/// - F16 emits the `f16` type (requires `enable f16;` — see `wgsl_prelude`)
/// - BF16 has no native WGSL type → stored as `u32` and unpacked in-kernel
/// - FP8 dequants on load (Phase 3e) → kernel-visible type is `f32`
/// - I8/U8 stored as `u32`-packed; kernel works in i32 in-register
pub fn dtype_to_wgsl(dtype: u8) -> &'static str {
    match dtype {
        buffer::DTYPE_F32 => "f32",
        buffer::DTYPE_I32 => "i32",
        buffer::DTYPE_F16 => "f16",
        // BF16: u16 bit pattern stored as packed u32; in-kernel unpack to f32.
        buffer::DTYPE_BF16 => "f32",
        // FP8 storage formats — kernel-side dequant handled in Phase 3e.
        buffer::DTYPE_FP8_E4M3 | buffer::DTYPE_FP8_E5M2 => "f32",
        buffer::DTYPE_I8 | buffer::DTYPE_U8 => "i32",
        _ => "f32",
    }
}

/// WGSL helper functions for FP8 (E4M3 / E5M2) dequant-on-load. Returned as
/// a single string ready to splice into the top of a shader that wants to
/// read FP8 storage and compute in f32.
///
/// WebGPU has no native FP8 type, so storage is packed bytes inside `u32`
/// elements. The helpers below take one `u32` and an index `[0..4)` into
/// the four packed FP8 lanes and return the f32 value. Used by Phase 4's
/// quantised matmul + future FP8 weight pipelines.
pub fn wgsl_fp8_dequant_helpers() -> &'static str {
    r#"
fn rayzor_fp8_e4m3_dequant(byte: u32) -> f32 {
    let sign = (byte >> 7u) & 1u;
    let exp = (byte >> 3u) & 0xFu;
    let mant = byte & 0x7u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 0xFu && mant == 0x7u) { return bitcast<f32>(0x7FC00000u); }
    var mant_num: f32;
    var exp_val: i32;
    if (exp == 0u) {
        mant_num = f32(mant);
        exp_val = -9;
    } else {
        mant_num = f32(8u + mant);
        exp_val = i32(exp) - 10;
    }
    var mag = mant_num * pow(2.0, f32(exp_val));
    if (sign != 0u) { mag = -mag; }
    return mag;
}

fn rayzor_fp8_e5m2_dequant(byte: u32) -> f32 {
    let sign = (byte >> 7u) & 1u;
    let exp = (byte >> 2u) & 0x1Fu;
    let mant = byte & 0x3u;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 0x1Fu) {
        if (mant == 0u) {
            if (sign == 0u) { return bitcast<f32>(0x7F800000u); }
            return bitcast<f32>(0xFF800000u);
        }
        return bitcast<f32>(0x7FC00000u);
    }
    var mant_num: f32;
    var exp_val: i32;
    if (exp == 0u) {
        mant_num = f32(mant);
        exp_val = -16;
    } else {
        mant_num = f32(4u + mant);
        exp_val = i32(exp) - 17;
    }
    var mag = mant_num * pow(2.0, f32(exp_val));
    if (sign != 0u) { mag = -mag; }
    return mag;
}
"#
}

/// Optional WGSL prelude that must appear above any shader using a given
/// dtype. Returns `""` when no prelude is needed.
///
/// For F16, this emits `enable f16;` — a WGSL extension that gates the
/// `f16` type. Adapters announce support via the `shader-f16` feature; the
/// host must request it at device-init time. As of 2026 Chrome (Tint),
/// Safari (WebKit-WSL), and Firefox all ship this extension on hardware
/// that supports ARMv8.2-A FP16 / Vulkan VK_KHR_shader_float16_int8.
pub fn wgsl_prelude(dtype: u8) -> &'static str {
    match dtype {
        buffer::DTYPE_F16 => "enable f16;\n",
        _ => "",
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
    } else if op == KernelOp::RmsNorm {
        4 // x, weight, y, params uniform
    } else if op == KernelOp::Rope {
        5 // x, cos, sin, y, params uniform
    } else {
        op.input_count() + 1 // inputs + result
    }
}

/// Generate WGSL source for a binary elementwise operation.
pub fn emit_binary_elementwise(op: KernelOp, dtype: u8) -> String {
    let prelude = wgsl_prelude(dtype);
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
        r#"{prelude}@group(0) @binding(0) var<storage, read> a: array<{wgsl_type}>;
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
    let prelude = wgsl_prelude(dtype);
    let wgsl_type = dtype_to_wgsl(dtype);
    let fn_name = kernel_fn_name(op, dtype);
    let op_expr = match op {
        KernelOp::Neg => "-a[id]".to_string(),
        KernelOp::Abs => "abs(a[id])".to_string(),
        KernelOp::Sqrt => "sqrt(a[id])".to_string(),
        KernelOp::Exp => "exp(a[id])".to_string(),
        KernelOp::Log => "log(a[id])".to_string(),
        KernelOp::Relu => format!("max({wgsl_type}(0), a[id])"),
        KernelOp::Sigmoid => format!("{wgsl_type}(1) / ({wgsl_type}(1) + exp(-a[id]))"),
        KernelOp::Tanh => "tanh(a[id])".to_string(),
        KernelOp::Gelu => {
            // Constants are written as f32 literals; WGSL implicitly converts
            // when the surrounding expression is `f16`. We cast the leading
            // multiplier to `wgsl_type` to anchor type inference correctly.
            format!(
                "a[id] * {wgsl_type}(0.5) * ({wgsl_type}(1) + tanh({wgsl_type}(0.7978845608) * (a[id] + {wgsl_type}(0.044715) * a[id] * a[id] * a[id])))"
            )
        }
        KernelOp::Silu => format!("a[id] / ({wgsl_type}(1) + exp(-a[id]))"),
        _ => unreachable!("not a unary op"),
    };

    format!(
        r#"{prelude}@group(0) @binding(0) var<storage, read> a: array<{wgsl_type}>;
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
    if let Some(src) = super::wgsl_transformer::emit_transformer(op, dtype) {
        return src;
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
    fn f16_emits_enable_extension_and_native_type() {
        let src = emit_binary_elementwise(KernelOp::Add, buffer::DTYPE_F16);
        assert!(src.starts_with("enable f16;"));
        assert!(src.contains("array<f16>"));
        assert!(src.contains("a[id] + b[id]"));
    }

    #[test]
    fn f32_kernel_omits_f16_extension() {
        let src = emit_binary_elementwise(KernelOp::Add, buffer::DTYPE_F32);
        assert!(!src.contains("enable f16;"));
    }

    #[test]
    fn f16_unary_relu_uses_f16_zero() {
        let src = emit_unary_elementwise(KernelOp::Relu, buffer::DTYPE_F16);
        assert!(src.contains("enable f16;"));
        assert!(src.contains("max(f16(0), a[id])"));
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
