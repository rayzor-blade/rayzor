//! Higher-level F32 tensor kernel exports (flash attention, softmax,
//! RMSNorm). Thin wrappers around the runtime-core kernels with the wasm-
//! preferred math primitives (`libm::expf`, `libm::sqrtf`) bound in.

use core::slice;
use rayzor_runtime_core::tensor::{flash_attn, rms_norm, softmax};

/// `out[q_head, :] = Σ_l softmax((Q[h] · K[l, kv_head, :]) * scale) * V[l, kv_head, :]`.
///
/// All buffers are F32. Layout (mirrors the native `rayzor_tensor_flash_attn_decode`):
///   - `q_data`  : `[n_q_heads * head_dim]`
///   - `k_data`  : `[cache_len * kv_row_stride]` (typically
///                 `kv_row_stride = n_kv_heads * head_dim`)
///   - `v_data`  : same shape as `k_data`
///   - `out_data`: `[n_q_heads * head_dim]`
///
/// `group = n_q_heads / n_kv_heads` (GQA grouping factor; 4 for Llama 3 1B,
/// 8 for Llama 3 70B).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode_f32(
    q_data: i32,
    k_data: i32,
    v_data: i32,
    out_data: i32,
    n_q_heads: i32,
    group: i32,
    head_dim: i32,
    cache_len: i32,
    kv_row_stride: i32,
    scale: f32,
) {
    if q_data == 0 || k_data == 0 || v_data == 0 || out_data == 0 {
        return;
    }
    if n_q_heads <= 0 || group <= 0 || head_dim <= 0 || cache_len <= 0 {
        return;
    }
    let n_q_heads = n_q_heads as usize;
    let group = group as usize;
    let head_dim = head_dim as usize;
    let cache_len = cache_len as usize;
    let kv_row_stride = kv_row_stride as usize;

    let mut scores: std::vec::Vec<f32> = std::vec![0.0; cache_len];
    for q_head in 0..n_q_heads {
        flash_attn::flash_attn_decode_one_qhead(
            q_head,
            group,
            head_dim,
            cache_len,
            kv_row_stride,
            scale,
            q_data as *const f32,
            k_data as *const f32,
            v_data as *const f32,
            out_data as *mut f32,
            &mut scores,
            libm::expf,
        );
    }
}

/// In-place row-wise softmax over an `[n_rows, row_len]` F32 buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_softmax_inplace_f32(
    data: i32,
    n_rows: i32,
    row_len: i32,
) {
    if data == 0 || n_rows <= 0 || row_len <= 0 {
        return;
    }
    let row_len = row_len as usize;
    let p = data as *mut f32;
    for r in 0..n_rows as usize {
        let row = slice::from_raw_parts_mut(p.add(r * row_len), row_len);
        softmax::softmax_inplace_f32(row, libm::expf);
    }
}

/// Row-wise RMSNorm over an `[n_rows, hidden_dim]` F32 buffer, writing to
/// `out`. `weight` is the per-channel learnable gain (`[hidden_dim]`).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rms_norm_f32(
    out: i32,
    x: i32,
    weight: i32,
    eps: f32,
    n_rows: i32,
    hidden_dim: i32,
) {
    if out == 0 || x == 0 || weight == 0 || n_rows <= 0 || hidden_dim <= 0 {
        return;
    }
    let n = hidden_dim as usize;
    let out_p = out as *mut f32;
    let x_p = x as *const f32;
    let w_slice = slice::from_raw_parts(weight as *const f32, n);
    for r in 0..n_rows as usize {
        let x_row = slice::from_raw_parts(x_p.add(r * n), n);
        let out_row = slice::from_raw_parts_mut(out_p.add(r * n), n);
        rms_norm::rms_norm_row_f32(out_row, x_row, w_slice, eps, libm::sqrtf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_inplace_normalises_and_sums_to_one() {
        unsafe {
            let mut row = [1.0f32, 2.0, 3.0, 4.0];
            rayzor_tensor_softmax_inplace_f32(row.as_mut_ptr() as i32, 1, 4);
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
            // Monotonic — larger input ⇒ larger output.
            assert!(row[0] < row[1] && row[1] < row[2] && row[2] < row[3]);
        }
    }

    #[test]
    fn rms_norm_matches_scalar_reference() {
        unsafe {
            let x: [f32; 4] = [1.0, -2.0, 3.0, -4.0];
            let weight: [f32; 4] = [0.5, 0.25, 1.0, 2.0];
            let eps = 1e-6f32;
            let mut out = [0.0f32; 4];
            rayzor_tensor_rms_norm_f32(
                out.as_mut_ptr() as i32,
                x.as_ptr() as i32,
                weight.as_ptr() as i32,
                eps,
                1,
                4,
            );
            let ss: f32 = x.iter().map(|v| v * v).sum();
            let mean_sq = ss / 4.0;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..4 {
                let want = x[i] * weight[i] * inv;
                assert!((out[i] - want).abs() < 1e-5, "i={} got={} want={}", i, out[i], want);
            }
        }
    }

    #[test]
    fn flash_attn_decode_matches_reference_single_head() {
        unsafe {
            // 1 q_head, 1 kv_head, head_dim=4, cache_len=3.
            let q = [1.0f32, 2.0, -1.0, 0.5];
            let k = [
                0.5f32, -0.5, 1.0, 0.0, // K[0]
                -1.0, 1.0, 0.0, 0.5, // K[1]
                0.25, 0.25, 0.25, 0.25, // K[2]
            ];
            let v = [
                1.0f32, 0.0, 0.0, 0.0, // V[0]
                0.0, 1.0, 0.0, 0.0, // V[1]
                0.0, 0.0, 1.0, 0.0, // V[2]
            ];
            let mut out = [0.0f32; 4];
            let scale = 0.5f32;
            rayzor_tensor_flash_attn_decode_f32(
                q.as_ptr() as i32,
                k.as_ptr() as i32,
                v.as_ptr() as i32,
                out.as_mut_ptr() as i32,
                1,
                1,
                4,
                3,
                4,
                scale,
            );

            // Reference: scores, softmax, weighted V.
            let mut scores = [0.0f32; 3];
            let mut max = f32::NEG_INFINITY;
            for l in 0..3 {
                let mut dot = 0.0f32;
                for i in 0..4 {
                    dot += q[i] * k[l * 4 + i];
                }
                scores[l] = dot * scale;
                if scores[l] > max {
                    max = scores[l];
                }
            }
            let mut denom = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                denom += *s;
            }
            for s in scores.iter_mut() {
                *s /= denom;
            }
            let mut want = [0.0f32; 4];
            for l in 0..3 {
                for i in 0..4 {
                    want[i] += scores[l] * v[l * 4 + i];
                }
            }
            for i in 0..4 {
                assert!((out[i] - want[i]).abs() < 1e-5, "i={} got={} want={}", i, out[i], want[i]);
            }
        }
    }
}
