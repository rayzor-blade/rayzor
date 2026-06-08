//! Root-mean-square normalisation over the last dimension of a tensor row.

use crate::simd::tensor_f32::sum_of_squares;

/// Compute `out[i] = x[i] * weight[i] * 1 / sqrt(mean(x[i]^2) + eps)` for a
/// single row of length `hidden_dim`. `weight` is the per-channel learnable
/// gain. `sqrt_fn` is supplied by the caller — native passes `f32::sqrt`
/// (single-instruction on aarch64/x86 FPU); wasm passes `libm::sqrtf`
/// (LLVM lowers it to `f32.sqrt` under +simd128 / scalar).
#[inline(always)]
pub fn rms_norm_row_f32<F: Fn(f32) -> f32>(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    eps: f32,
    sqrt_fn: F,
) {
    let n = x.len();
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(weight.len(), n);
    if n == 0 {
        return;
    }
    let ss = sum_of_squares(x);
    let inv_rms = 1.0 / sqrt_fn(ss / (n as f32) + eps);
    for i in 0..n {
        out[i] = x[i] * weight[i] * inv_rms;
    }
}
