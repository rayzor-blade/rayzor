//! Numerically stabilised softmax over a contiguous f32 row.
//!
//! Generic over the exponential implementation so each crate passes its
//! preferred one (native: `|x| x.exp()` → libsystem_m / glibc; wasm:
//! `libm::expf`). Monomorphised + `#[inline(always)]` so the per-row inner
//! folds into the caller across the crate boundary.

use crate::simd::tensor_f32::max_slice;

/// In-place row softmax with explicit `exp_fn` provider.
///
/// ```text
///   max  = max(row)
///   row[i] = exp(row[i] - max)
///   denom = Σ_i row[i]
///   row[i] = row[i] / denom
/// ```
#[inline(always)]
pub fn softmax_inplace_f32<F: Fn(f32) -> f32>(row: &mut [f32], exp_fn: F) {
    if row.is_empty() {
        return;
    }
    let m = max_slice(row);
    let mut denom = 0.0f32;
    for v in row.iter_mut() {
        let e = exp_fn(*v - m);
        *v = e;
        denom += e;
    }
    if denom > 0.0 {
        let inv = 1.0 / denom;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}
