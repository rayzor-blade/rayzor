//! INT8 symmetric per-row quantisation kernels.

use crate::floats::roundf;
use crate::simd::tensor_f32::axpy_slice;

/// Quantise an f32 row into int8 + per-row scale.
/// Returns the scale; writes the quantised bytes into `dst`.
pub fn quantise_int8_row(src: &[f32], dst: &mut [i8]) -> f32 {
    debug_assert_eq!(src.len(), dst.len());
    let mut max_abs = 0.0f32;
    for &x in src {
        let a = x.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    // 127 = int8 max magnitude (-128..=127). Symmetric quant uses 127 to
    // keep the dequant formula sign-symmetric.
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let inv = 1.0 / scale;
    for (i, &x) in src.iter().enumerate() {
        let q = roundf(x * inv).clamp(-127.0, 127.0) as i8;
        dst[i] = q;
    }
    scale
}

/// Dequant + matmul kernel for INT8-quantised A times f32 B.
///
/// A is `[M, K]` stored as i8 with one scale per row.
/// B is `[K, N]` stored as f32, row-major.
/// C is `[M, N]` stored as f32, row-major.
///
/// Computes `c[m, n] = scale[m] * Σ_k a_i8[m, k] * b[k, n]`.
/// The kernel runs i32-accumulated dot product per (m, n) lane, then
/// multiplies by the row scale at the end. This is the canonical pattern —
/// the i8 × f32 mixed-precision product would lose information without
/// the i32 accumulator.
///
/// # Safety
/// All four pointers must reference live storage covering the indexed
/// regions: `a_data[..m*k]`, `scales[..m]`, `b_data[..k*n]`, `c_data[..m*n]`.
#[allow(clippy::needless_range_loop)]
pub unsafe fn int8_matmul_f32(
    a_data: *const i8,
    scales: *const f32,
    b_data: *const f32,
    c_data: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    for i in 0..m {
        let row_scale = *scales.add(i);
        let a_row = a_data.add(i * k);
        let c_row = c_data.add(i * n);
        // Initialise the result row to zero.
        core::ptr::write_bytes(c_row, 0, n * core::mem::size_of::<f32>());
        for p in 0..k {
            let a_ik = *a_row.add(p) as f32 * row_scale;
            let b_row = b_data.add(p * n);
            // Equivalent to axpy_slice(c_row, a_ik, b_row).
            let c_slice = core::slice::from_raw_parts_mut(c_row, n);
            let b_slice = core::slice::from_raw_parts(b_row, n);
            axpy_slice(c_slice, a_ik, b_slice);
        }
    }
}
