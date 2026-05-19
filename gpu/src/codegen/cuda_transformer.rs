//! CUDA codegen for the transformer KernelOps.
//!
//! Same row-shaped layout — one block per row for RMSNorm, one thread
//! per `(seq, head, half_idx)` lane for RoPE. The half-precision path
//! pulls in `cuda_fp16.h` (sm_53+); reductions accumulate in `float`
//! to avoid f16 overflow on `sum(x²)`.

use crate::codegen::cuda::{cuda_prelude, dtype_to_cuda};
use crate::kernel_ir::KernelOp;

#[cfg(test)]
use crate::buffer;

const BLOCK_SIZE: usize = 256;

pub fn emit_rms_norm(dtype: u8) -> String {
    let prelude = cuda_prelude(dtype);
    let t = dtype_to_cuda(dtype);
    let t_suffix = t.replace(' ', "_");
    let fn_name = format!("rayzor_rms_norm_{}", t_suffix);

    format!(
        r#"{prelude}#include <math.h>

struct RmsParams {{
    unsigned int row_len;
    float eps;
}};

extern "C" __global__ void {fn_name}(
    const {t}* __restrict__ x_in,
    const {t}* __restrict__ weight,
    {t}* __restrict__ y_out,
    const RmsParams params
) {{
    constexpr unsigned int LANES = {BLOCK_SIZE};
    __shared__ float partial[LANES];
    __shared__ float shared_inv_rms;

    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int row_len = params.row_len;
    const unsigned int base = row * row_len;

    float acc = 0.0f;
    for (unsigned int i = tid; i < row_len; i += LANES) {{
        float v = (float)x_in[base + i];
        acc += v * v;
    }}
    partial[tid] = acc;
    __syncthreads();

    for (unsigned int s = LANES / 2; s > 0; s >>= 1) {{
        if (tid < s) {{ partial[tid] += partial[tid + s]; }}
        __syncthreads();
    }}

    if (tid == 0) {{
        float mean_sq = partial[0] / (float)row_len;
        shared_inv_rms = 1.0f / sqrtf(mean_sq + params.eps);
    }}
    __syncthreads();
    float inv_rms = shared_inv_rms;

    for (unsigned int i = tid; i < row_len; i += LANES) {{
        float xv = (float)x_in[base + i];
        float wv = (float)weight[i];
        y_out[base + i] = ({t})(xv * inv_rms * wv);
    }}
}}
"#
    )
}

pub fn emit_rope(dtype: u8) -> String {
    let prelude = cuda_prelude(dtype);
    let t = dtype_to_cuda(dtype);
    let t_suffix = t.replace(' ', "_");
    let fn_name = format!("rayzor_rope_{}", t_suffix);

    format!(
        r#"{prelude}struct RopeParams {{
    unsigned int seq_len;
    unsigned int num_heads;
    unsigned int head_dim;
    unsigned int position_offset;
    unsigned int cos_max_seq;
}};

extern "C" __global__ void {fn_name}(
    const {t}*    __restrict__ x_in,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    {t}*          __restrict__ y_out,
    const RopeParams params
) {{
    const unsigned int half_dim = params.head_dim / 2;
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = params.seq_len * params.num_heads * half_dim;
    if (lane >= total) {{ return; }}

    const unsigned int half_idx = lane % half_dim;
    const unsigned int head_idx = (lane / half_dim) % params.num_heads;
    const unsigned int seq_idx  = lane / (half_dim * params.num_heads);
    const unsigned int pos = seq_idx + params.position_offset;

    const unsigned int row_stride = params.num_heads * params.head_dim;
    const unsigned int base = seq_idx * row_stride + head_idx * params.head_dim;
    const unsigned int lo = base + 2 * half_idx;
    const unsigned int hi = lo + 1;

    if (pos >= params.cos_max_seq) {{
        y_out[lo] = x_in[lo];
        y_out[hi] = x_in[hi];
        return;
    }}

    float cos_v = cos_tab[pos * half_dim + half_idx];
    float sin_v = sin_tab[pos * half_dim + half_idx];
    float xlo = (float)x_in[lo];
    float xhi = (float)x_in[hi];
    y_out[lo] = ({t})(xlo * cos_v - xhi * sin_v);
    y_out[hi] = ({t})(xlo * sin_v + xhi * cos_v);
}}
"#
    )
}

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
    fn rms_norm_float_has_global_kernel() {
        let src = emit_rms_norm(buffer::DTYPE_F32);
        assert!(src.contains("__global__ void rayzor_rms_norm_float"));
        assert!(src.contains("__shared__ float partial"));
        assert!(src.contains("__syncthreads()"));
        assert!(src.contains("sqrtf("));
    }

    #[test]
    fn rms_norm_half_includes_cuda_fp16_header() {
        let src = emit_rms_norm(buffer::DTYPE_F16);
        assert!(src.contains("#include <cuda_fp16.h>"));
        assert!(src.contains("__half"));
        assert!(src.contains("float acc = 0.0f"));
    }

    #[test]
    fn rope_emits_adjacent_pair_rotation() {
        let src = emit_rope(buffer::DTYPE_F32);
        assert!(src.contains("__global__ void rayzor_rope_float"));
        assert!(src.contains("cos_tab[pos * half_dim + half_idx]"));
        assert!(src.contains("xlo * cos_v - xhi * sin_v"));
    }
}
