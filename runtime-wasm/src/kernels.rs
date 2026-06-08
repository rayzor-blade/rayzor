//! Higher-level F32 tensor kernel exports (flash attention, softmax, RoPE,
//! RMSNorm). Thin wrappers around the runtime-core kernels with the wasm-
//! preferred math primitives (`libm::expf`, `libm::sqrtf`) bound in.

use core::slice;
use rayzor_runtime_core::tensor::{flash_attn, rms_norm, rope, softmax};

use crate::tensor::{load_f32_at, store_f32_at, Tensor, DTYPE_F16, DTYPE_F32};

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
pub unsafe extern "C" fn rayzor_tensor_softmax_inplace_f32(data: i32, n_rows: i32, row_len: i32) {
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

/// Apply interleaved Llama/GGUF RoPE to an F32 tensor of shape
/// `[seq_len, num_heads, head_dim]` or `[seq_len, head_dim]`.
///
/// Returns a freshly allocated Tensor with the same shape. This export uses
/// the same public name as the native runtime so Haxe's existing
/// `Tensor.rope()` stdlib wrapper resolves on WASM too.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope(
    x: i32,
    cos: i32,
    sin: i32,
    position_offset: i32,
) -> i32 {
    if x == 0 || cos == 0 || sin == 0 {
        return 0;
    }
    let xr = &*(x as *const Tensor);
    let cr = &*(cos as *const Tensor);
    let sr = &*(sin as *const Tensor);
    if xr.dtype != DTYPE_F32 {
        return 0;
    }
    if xr.ndim < 2 || cr.ndim != 2 || sr.ndim != 2 {
        return 0;
    }

    let x_shape = slice::from_raw_parts(xr.shape, xr.ndim);
    let head_dim = x_shape[xr.ndim - 1];
    if !head_dim.is_multiple_of(2) {
        return 0;
    }
    let half = head_dim / 2;
    let num_heads = if xr.ndim >= 3 {
        x_shape[xr.ndim - 2]
    } else {
        1
    };
    let seq_len = if xr.ndim >= 3 {
        x_shape[..xr.ndim - 2].iter().product::<usize>().max(1)
    } else {
        x_shape[0]
    };

    let cos_shape = slice::from_raw_parts(cr.shape, 2);
    let sin_shape = slice::from_raw_parts(sr.shape, 2);
    if cos_shape[1] != half || sin_shape[1] != half || sin_shape[0] != cos_shape[0] {
        return 0;
    }

    let y = crate::tensor::rayzor_tensor_zeros(xr.shape as i32, xr.ndim as i32, DTYPE_F32 as i32);
    if y == 0 {
        return 0;
    }
    let yr = &*(y as *const Tensor);
    if cr.dtype == DTYPE_F32 && sr.dtype == DTYPE_F32 {
        let out = slice::from_raw_parts_mut(yr.data as *mut f32, yr.numel);
        let x_data = slice::from_raw_parts(xr.data as *const f32, xr.numel);
        let cos_data = slice::from_raw_parts(cr.data as *const f32, cr.numel);
        let sin_data = slice::from_raw_parts(sr.data as *const f32, sr.numel);
        rope::apply_interleaved_f32(
            out,
            x_data,
            cos_data,
            sin_data,
            seq_len,
            num_heads,
            head_dim,
            position_offset.max(0) as usize,
        );
    } else {
        let pos_off = position_offset.max(0) as usize;
        let elements_per_row = num_heads * head_dim;
        let x_data = xr.data as *const f32;
        let out_data = yr.data as *mut f32;
        for s in 0..seq_len {
            let pos = s + pos_off;
            let row_base = s * elements_per_row;
            if pos >= cos_shape[0] {
                core::ptr::copy_nonoverlapping(
                    x_data.add(row_base),
                    out_data.add(row_base),
                    elements_per_row,
                );
                continue;
            }
            for h in 0..num_heads {
                let head_base = row_base + h * head_dim;
                let table_base = pos * half;
                for i in 0..half {
                    let c = load_f32_at(cr.data, table_base + i, cr.dtype);
                    let sn = load_f32_at(sr.data, table_base + i, sr.dtype);
                    let lo = head_base + 2 * i;
                    let hi = lo + 1;
                    let xlo = *x_data.add(lo);
                    let xhi = *x_data.add(hi);
                    *out_data.add(lo) = xlo * c - xhi * sn;
                    *out_data.add(hi) = xlo * sn + xhi * c;
                }
            }
        }
    }
    y
}

/// Generate an F32 RoPE cosine table `[max_seq_len, head_dim / 2]`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, libm::cos)
}

/// Generate an F32 RoPE sine table `[max_seq_len, head_dim / 2]`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, libm::sin)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table_f16(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, libm::cos)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table_f16(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, libm::sin)
}

unsafe fn rope_table<F: Fn(f64) -> f64>(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
    dtype: u8,
    trig_fn: F,
) -> i32 {
    if head_dim <= 0 || max_seq_len <= 0 || head_dim % 2 != 0 {
        return 0;
    }
    let head_dim = head_dim as usize;
    let max_seq_len = max_seq_len as usize;
    let shape = [max_seq_len, head_dim / 2];
    let t = crate::tensor::rayzor_tensor_zeros(shape.as_ptr() as i32, 2, dtype as i32);
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    if dtype == DTYPE_F32 {
        let out = slice::from_raw_parts_mut(tr.data as *mut f32, tr.numel);
        rope::fill_table_f32(out, head_dim, max_seq_len, base, trig_fn);
    } else {
        let half = head_dim / 2;
        let mut scratch = std::vec![0.0f32; max_seq_len * half];
        rope::fill_table_f32(&mut scratch, head_dim, max_seq_len, base, trig_fn);
        for (i, &v) in scratch.iter().enumerate() {
            store_f32_at(tr.data, i, dtype, v);
        }
    }
    t
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
                assert!(
                    (out[i] - want).abs() < 1e-5,
                    "i={} got={} want={}",
                    i,
                    out[i],
                    want
                );
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
                assert!(
                    (out[i] - want[i]).abs() < 1e-5,
                    "i={} got={} want={}",
                    i,
                    out[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn rope_tables_and_apply_match_interleaved_reference() {
        unsafe {
            let cos = rayzor_tensor_rope_cos_table(4, 4, 10000.0);
            let sin = rayzor_tensor_rope_sin_table(4, 4, 10000.0);
            assert!(cos != 0);
            assert!(sin != 0);

            let x_data = [1.0f32, 2.0, 3.0, 4.0];
            let x_shape = [1usize, 1, 4];
            let x = crate::tensor::rayzor_tensor_from_floats(
                x_data.as_ptr() as i32,
                4,
                x_shape.as_ptr() as i32,
                3,
            );
            assert!(x != 0);

            let y = rayzor_tensor_rope(x, cos, sin, 1);
            assert!(y != 0);
            let yr = &*(y as *const Tensor);
            let got = slice::from_raw_parts(yr.data as *const f32, 4);

            let c0 = libm::cos(1.0) as f32;
            let s0 = libm::sin(1.0) as f32;
            let c1 = libm::cos(0.01) as f32;
            let s1 = libm::sin(0.01) as f32;
            let want = [
                x_data[0] * c0 - x_data[1] * s0,
                x_data[0] * s0 + x_data[1] * c0,
                x_data[2] * c1 - x_data[3] * s1,
                x_data[2] * s1 + x_data[3] * c1,
            ];
            for i in 0..4 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-6,
                    "i={} got={} want={}",
                    i,
                    got[i],
                    want[i]
                );
            }

            crate::tensor::rayzor_tensor_free(x);
            crate::tensor::rayzor_tensor_free(y);
            crate::tensor::rayzor_tensor_free(cos);
            crate::tensor::rayzor_tensor_free(sin);
        }
    }

    #[test]
    fn rope_f16_tables_apply_with_expected_precision() {
        unsafe {
            let cos = rayzor_tensor_rope_cos_table_f16(4, 4, 10000.0);
            let sin = rayzor_tensor_rope_sin_table_f16(4, 4, 10000.0);
            assert!(cos != 0);
            assert!(sin != 0);
            assert_eq!((*(cos as *const Tensor)).dtype, DTYPE_F16);
            assert_eq!((*(sin as *const Tensor)).dtype, DTYPE_F16);

            let x_data = [1.0f32, 2.0, 3.0, 4.0];
            let x_shape = [1usize, 1, 4];
            let x = crate::tensor::rayzor_tensor_from_floats(
                x_data.as_ptr() as i32,
                4,
                x_shape.as_ptr() as i32,
                3,
            );
            let y = rayzor_tensor_rope(x, cos, sin, 1);
            assert!(y != 0);

            let yr = &*(y as *const Tensor);
            let got = slice::from_raw_parts(yr.data as *const f32, 4);
            let c0 = libm::cos(1.0) as f32;
            let s0 = libm::sin(1.0) as f32;
            let c1 = libm::cos(0.01) as f32;
            let s1 = libm::sin(0.01) as f32;
            let want = [
                x_data[0] * c0 - x_data[1] * s0,
                x_data[0] * s0 + x_data[1] * c0,
                x_data[2] * c1 - x_data[3] * s1,
                x_data[2] * s1 + x_data[3] * c1,
            ];
            for i in 0..4 {
                assert!(
                    (got[i] - want[i]).abs() < 2e-3,
                    "i={} got={} want={}",
                    i,
                    got[i],
                    want[i]
                );
            }

            crate::tensor::rayzor_tensor_free(x);
            crate::tensor::rayzor_tensor_free(y);
            crate::tensor::rayzor_tensor_free(cos);
            crate::tensor::rayzor_tensor_free(sin);
        }
    }
}
