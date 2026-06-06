//! WGSL codegen for the transformer-primitive `KernelOp`s — `RmsNorm`
//! and `Rope`.
//!
//! These ops share the same row-shaped layout used by softmax /
//! layer_norm on the CPU side: the last dimension is the working axis
//! (`hidden_size` for RMSNorm, `head_dim` for RoPE) and one workgroup
//! handles one row. The `numel`-based grid that drives elementwise
//! kernels doesn't suit them — the per-row primitives dispatch
//! `groups × 1 × 1` instead, where `groups = total_numel / row_len`.
//!
//! F16 path: when the input dtype is F16 we still emit `enable f16;`
//! and use the native `f16` type. WebGPU adapters announce support via
//! the `shader-f16` feature; without it the host should request the
//! F32 variant. The reductions are accumulated in f32 either way (a
//! sum of squares would overflow F16 well before reaching standard
//! `hidden_size` like 4096).

use crate::buffer;
use crate::codegen::wgsl::{wgsl_prelude, WORKGROUP_SIZE};
use crate::kernel_ir::KernelOp;

/// RMS normalization shader. Each workgroup handles one row.
///
/// Bindings:
///   - 0: input  `array<{T}>` (row-major, [groups, row_len])
///   - 1: weight `array<{T}>` (per-channel gain, [row_len])
///   - 2: output `array<{T}>` (same shape as input)
///   - 3: uniform `{row_len: u32, eps: f32}`
///
/// Algorithm:
///   1. Each thread reads its share of the row, accumulates `x²` into a
///      thread-local f32.
///   2. Workgroup-shared reduction folds the per-thread sums.
///   3. Thread 0 computes `inv_rms = 1 / sqrt(sum / row_len + eps)`,
///      writes it to a shared scalar.
///   4. Every thread reads `inv_rms`, writes `input * inv_rms * weight[i]`.
pub fn emit_rms_norm(dtype: u8) -> String {
    let prelude = wgsl_prelude(dtype);
    let t = match dtype {
        buffer::DTYPE_F16 => "f16",
        _ => "f32",
    };
    let fn_name = format!("rayzor_rms_norm_{}", t);

    format!(
        r#"{prelude}struct RmsParams {{
    row_len: u32,
    eps: f32,
}};

@group(0) @binding(0) var<storage, read> x_in: array<{t}>;
@group(0) @binding(1) var<storage, read> weight: array<{t}>;
@group(0) @binding(2) var<storage, read_write> y_out: array<{t}>;
@group(0) @binding(3) var<uniform> params: RmsParams;

var<workgroup> partial_sums: array<f32, {WORKGROUP_SIZE}u>;
var<workgroup> shared_inv_rms: f32;

@compute @workgroup_size({WORKGROUP_SIZE})
fn {fn_name}(
    @builtin(workgroup_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {{
    let row = gid.x;
    let row_len = params.row_len;
    let base = row * row_len;
    let tid = lid.x;
    let lanes = {WORKGROUP_SIZE}u;

    // Sum of squares — accumulate in f32 regardless of storage dtype.
    var acc: f32 = 0.0;
    var i = tid;
    loop {{
        if (i >= row_len) {{ break; }}
        let v = f32(x_in[base + i]);
        acc = acc + v * v;
        i = i + lanes;
    }}
    partial_sums[tid] = acc;
    workgroupBarrier();

    // Workgroup-shared tree reduction.
    var s = lanes >> 1u;
    loop {{
        if (s == 0u) {{ break; }}
        if (tid < s) {{
            partial_sums[tid] = partial_sums[tid] + partial_sums[tid + s];
        }}
        workgroupBarrier();
        s = s >> 1u;
    }}

    if (tid == 0u) {{
        let mean_sq = partial_sums[0] / f32(row_len);
        shared_inv_rms = 1.0 / sqrt(mean_sq + params.eps);
    }}
    workgroupBarrier();
    let inv_rms = shared_inv_rms;

    // Apply: y = x * inv_rms * weight.
    i = tid;
    loop {{
        if (i >= row_len) {{ break; }}
        let xv = f32(x_in[base + i]);
        let wv = f32(weight[i]);
        y_out[base + i] = {t}(xv * inv_rms * wv);
        i = i + lanes;
    }}
}}
"#
    )
}

/// RoPE shader: rotates adjacent-pair lanes of `x [seq_len, num_heads,
/// head_dim]` against precomputed cos/sin tables.
///
/// Bindings:
///   - 0: x      `array<{T}>` (input)
///   - 1: cos    `array<f32>` (always f32 — LUT precision matters more
///     than memory for the rotation tables)
///   - 2: sin    `array<f32>`
///   - 3: out    `array<{T}>`
///   - 4: uniform `{seq_len, num_heads, head_dim, position_offset, cos_max_seq}`
///
/// One thread handles one `(seq_idx, head_idx, half_idx)` lane: it
/// reads x[2*half_idx] and x[2*half_idx+1], applies the rotation, and
/// writes both elements back.
pub fn emit_rope(dtype: u8) -> String {
    let prelude = wgsl_prelude(dtype);
    let t = match dtype {
        buffer::DTYPE_F16 => "f16",
        _ => "f32",
    };
    let fn_name = format!("rayzor_rope_{}", t);

    format!(
        r#"{prelude}struct RopeParams {{
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    position_offset: u32,
    cos_max_seq: u32,
}};

@group(0) @binding(0) var<storage, read> x_in: array<{t}>;
@group(0) @binding(1) var<storage, read> cos_tab: array<f32>;
@group(0) @binding(2) var<storage, read> sin_tab: array<f32>;
@group(0) @binding(3) var<storage, read_write> y_out: array<{t}>;
@group(0) @binding(4) var<uniform> params: RopeParams;

@compute @workgroup_size({WORKGROUP_SIZE})
fn {fn_name}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let half_dim = params.head_dim / 2u;
    let lane = gid.x;
    let total = params.seq_len * params.num_heads * half_dim;
    if (lane >= total) {{ return; }}

    let half_idx = lane % half_dim;
    let head_idx = (lane / half_dim) % params.num_heads;
    let seq_idx  = lane / (half_dim * params.num_heads);
    let pos = seq_idx + params.position_offset;

    let row_stride = params.num_heads * params.head_dim;
    let base = seq_idx * row_stride + head_idx * params.head_dim;
    let lo = base + 2u * half_idx;
    let hi = lo + 1u;

    if (pos >= params.cos_max_seq) {{
        // Out of LUT range — identity rotation (matches the CPU fallback).
        y_out[lo] = x_in[lo];
        y_out[hi] = x_in[hi];
        return;
    }}

    let cos_v = cos_tab[pos * half_dim + half_idx];
    let sin_v = sin_tab[pos * half_dim + half_idx];
    let xlo = f32(x_in[lo]);
    let xhi = f32(x_in[hi]);
    y_out[lo] = {t}(xlo * cos_v - xhi * sin_v);
    y_out[hi] = {t}(xlo * sin_v + xhi * cos_v);
}}
"#
    )
}

/// Dispatch helper: returns the shader source for the given
/// transformer KernelOp, or `None` if the op isn't a transformer
/// primitive (callers fall back to the standard `emit_kernel` path).
pub fn emit_transformer(op: KernelOp, dtype: u8) -> Option<String> {
    match op {
        KernelOp::RmsNorm => Some(emit_rms_norm(dtype)),
        KernelOp::Rope => Some(emit_rope(dtype)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_f32_has_expected_bindings() {
        let src = emit_rms_norm(buffer::DTYPE_F32);
        assert!(src.contains("fn rayzor_rms_norm_f32"));
        assert!(src.contains("var<storage, read> x_in: array<f32>"));
        assert!(src.contains("var<storage, read> weight: array<f32>"));
        assert!(src.contains("var<storage, read_write> y_out: array<f32>"));
        assert!(src.contains("var<uniform> params: RmsParams"));
        assert!(src.contains("inv_rms"));
        // No f16 prelude on the F32 path.
        assert!(!src.starts_with("enable f16;"));
    }

    #[test]
    fn rms_norm_f16_emits_extension() {
        let src = emit_rms_norm(buffer::DTYPE_F16);
        assert!(src.starts_with("enable f16;"));
        assert!(src.contains("array<f16>"));
        // Reduction accumulates in f32 even for f16 inputs.
        assert!(src.contains("var acc: f32"));
    }

    #[test]
    fn rope_f32_uses_cos_sin_tables() {
        let src = emit_rope(buffer::DTYPE_F32);
        assert!(src.contains("fn rayzor_rope_f32"));
        assert!(src.contains("cos_tab[pos * half_dim + half_idx]"));
        assert!(src.contains("sin_tab[pos * half_dim + half_idx]"));
        assert!(src.contains("xlo * cos_v - xhi * sin_v"));
    }

    #[test]
    fn dispatch_recognises_transformer_ops() {
        assert!(emit_transformer(KernelOp::RmsNorm, buffer::DTYPE_F32).is_some());
        assert!(emit_transformer(KernelOp::Rope, buffer::DTYPE_F32).is_some());
        // Non-transformer ops return None for the fallthrough.
        assert!(emit_transformer(KernelOp::Add, buffer::DTYPE_F32).is_none());
    }
}
