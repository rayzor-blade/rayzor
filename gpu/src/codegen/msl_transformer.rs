//! MSL (Metal Shading Language) codegen for the transformer KernelOps.
//!
//! Same row-shaped layout as the WGSL variants: each threadgroup handles
//! one row. Metal has native `half` so the F16 path is straightforward;
//! reductions still accumulate in `float` to dodge half-precision
//! overflow on `sum(x²)` for typical hidden_size values.

#[cfg(test)]
use crate::buffer;
use crate::codegen::msl::dtype_to_msl;
use crate::kernel_ir::KernelOp;

const TG_SIZE: usize = 256;

/// RMSNorm: one threadgroup per row, two-pass reduce inside the group.
pub fn emit_rms_norm(dtype: u8) -> String {
    let t = dtype_to_msl(dtype);
    let fn_name = format!("rayzor_rms_norm_{}", t);

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

struct RmsParams {{
    uint row_len;
    float eps;
}};

kernel void {fn_name}(
    device const {t}*    x_in    [[buffer(0)]],
    device const {t}*    weight  [[buffer(1)]],
    device {t}*          y_out   [[buffer(2)]],
    constant RmsParams&  params  [[buffer(3)]],
    uint                 gid     [[threadgroup_position_in_grid]],
    uint                 tid     [[thread_position_in_threadgroup]]
) {{
    constexpr uint LANES = {TG_SIZE};
    threadgroup float partial[LANES];
    threadgroup float shared_inv_rms;

    const uint row_len = params.row_len;
    const uint base = gid * row_len;

    // sum of squares in float32 even for half storage
    float acc = 0.0f;
    for (uint i = tid; i < row_len; i += LANES) {{
        float v = float(x_in[base + i]);
        acc += v * v;
    }}
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = LANES / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ partial[tid] += partial[tid + s]; }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (tid == 0) {{
        float mean_sq = partial[0] / float(row_len);
        shared_inv_rms = 1.0f / sqrt(mean_sq + params.eps);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_inv_rms;

    for (uint i = tid; i < row_len; i += LANES) {{
        float xv = float(x_in[base + i]);
        float wv = float(weight[i]);
        y_out[base + i] = ({t})(xv * inv_rms * wv);
    }}
}}
"#
    )
}

/// RoPE: one thread per `(seq_idx, head_idx, half_idx)` lane.
pub fn emit_rope(dtype: u8) -> String {
    let t = dtype_to_msl(dtype);
    let fn_name = format!("rayzor_rope_{}", t);

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

struct RopeParams {{
    uint seq_len;
    uint num_heads;
    uint head_dim;
    uint position_offset;
    uint cos_max_seq;
}};

kernel void {fn_name}(
    device const {t}*    x_in     [[buffer(0)]],
    device const float*  cos_tab  [[buffer(1)]],
    device const float*  sin_tab  [[buffer(2)]],
    device {t}*          y_out    [[buffer(3)]],
    constant RopeParams& params   [[buffer(4)]],
    uint                 lane     [[thread_position_in_grid]]
) {{
    const uint half_dim = params.head_dim / 2;
    const uint total = params.seq_len * params.num_heads * half_dim;
    if (lane >= total) {{ return; }}

    const uint half_idx = lane % half_dim;
    const uint head_idx = (lane / half_dim) % params.num_heads;
    const uint seq_idx  = lane / (half_dim * params.num_heads);
    const uint pos = seq_idx + params.position_offset;

    const uint row_stride = params.num_heads * params.head_dim;
    const uint base = seq_idx * row_stride + head_idx * params.head_dim;
    const uint lo = base + 2 * half_idx;
    const uint hi = lo + 1;

    if (pos >= params.cos_max_seq) {{
        y_out[lo] = x_in[lo];
        y_out[hi] = x_in[hi];
        return;
    }}

    float cos_v = cos_tab[pos * half_dim + half_idx];
    float sin_v = sin_tab[pos * half_dim + half_idx];
    float xlo = float(x_in[lo]);
    float xhi = float(x_in[hi]);
    y_out[lo] = ({t})(xlo * cos_v - xhi * sin_v);
    y_out[hi] = ({t})(xlo * sin_v + xhi * cos_v);
}}
"#
    )
}

/// Dispatch helper for transformer primitives. Returns `None` for ops
/// that aren't part of this module.
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
    fn rms_norm_float_has_metal_kernel() {
        let src = emit_rms_norm(buffer::DTYPE_F32);
        assert!(src.contains("#include <metal_stdlib>"));
        assert!(src.contains("kernel void rayzor_rms_norm_float"));
        assert!(src.contains("device const float*    x_in"));
        assert!(src.contains("threadgroup_barrier"));
    }

    #[test]
    fn rms_norm_half_uses_native_metal_half() {
        let src = emit_rms_norm(buffer::DTYPE_F16);
        assert!(src.contains("rayzor_rms_norm_half"));
        assert!(src.contains("device const half*"));
        // Accumulator stays float
        assert!(src.contains("float acc = 0.0f"));
    }

    #[test]
    fn rope_emits_adjacent_pair_rotation() {
        let src = emit_rope(buffer::DTYPE_F32);
        assert!(src.contains("kernel void rayzor_rope_float"));
        assert!(src.contains("cos_tab[pos * half_dim + half_idx]"));
        assert!(src.contains("xlo * cos_v - xhi * sin_v"));
    }
}
