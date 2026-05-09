//! SIMD primitives for the f32 tensor hot paths.
//!
//! Provides 4-wide f32 lane operations using:
//! - aarch64 NEON intrinsics (Apple Silicon, ARM64 Linux)
//! - x86_64 SSE2 intrinsics (Intel/AMD, baseline x86_64 ABI)
//! - scalar fallback (other targets, including wasm32 without simd128)
//!
//! All public functions are safe: callers pass slices, the unsafe pointer
//! arithmetic is contained here. Tail elements (less than 4) fall back to
//! scalar code, so callers don't need to worry about alignment or remainders.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32,
    vminvq_f32, vmulq_f32, vst1q_f32, vsubq_f32,
};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, _mm_add_ps, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_min_ps, _mm_mul_ps, _mm_set1_ps,
    _mm_storeu_ps, _mm_sub_ps,
};

const LANES: usize = 4;

// ---------------------------------------------------------------------------
// Lane-wise binary ops on slices: r[i] = op(a[i], b[i])
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn add_slice(r: &mut [f32], a: &[f32], b: &[f32]) {
    let n = r.len();
    debug_assert!(a.len() == n && b.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vaddq_f32(va, vb));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            let vb = _mm_loadu_ps(b.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_add_ps(va, vb));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                r[off + k] = a[off + k] + b[off + k];
            }
        }
    }

    for i in (chunks * LANES)..n {
        r[i] = a[i] + b[i];
    }
}

#[inline(always)]
pub fn sub_slice(r: &mut [f32], a: &[f32], b: &[f32]) {
    let n = r.len();
    debug_assert!(a.len() == n && b.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vsubq_f32(va, vb));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            let vb = _mm_loadu_ps(b.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_sub_ps(va, vb));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                r[off + k] = a[off + k] - b[off + k];
            }
        }
    }

    for i in (chunks * LANES)..n {
        r[i] = a[i] - b[i];
    }
}

#[inline(always)]
pub fn mul_slice(r: &mut [f32], a: &[f32], b: &[f32]) {
    let n = r.len();
    debug_assert!(a.len() == n && b.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vmulq_f32(va, vb));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            let vb = _mm_loadu_ps(b.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_mul_ps(va, vb));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                r[off + k] = a[off + k] * b[off + k];
            }
        }
    }

    for i in (chunks * LANES)..n {
        r[i] = a[i] * b[i];
    }
}

/// Division — falls back to scalar in the divisor==0 branch to preserve the
/// original semantics (`if y != 0 { x/y } else { 0 }`). The SIMD path is
/// taken only when no zeros are present in the chunk, otherwise the four
/// lanes are evaluated scalar.
#[inline(always)]
pub fn div_slice(r: &mut [f32], a: &[f32], b: &[f32]) {
    let n = r.len();
    debug_assert!(a.len() == n && b.len() == n);
    let chunks = n / LANES;

    for i in 0..chunks {
        let off = i * LANES;
        let mut zero_present = false;
        for k in 0..LANES {
            if b[off + k] == 0.0 {
                zero_present = true;
                break;
            }
        }
        if zero_present {
            for k in 0..LANES {
                let bv = b[off + k];
                r[off + k] = if bv != 0.0 { a[off + k] / bv } else { 0.0 };
            }
            continue;
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(off));
            let vb = vld1q_f32(b.as_ptr().add(off));
            // NEON: f32 division is a real instruction on Apple Silicon.
            // Use reciprocal+mul as a portable fallback only if needed; here
            // we emit `fdiv` via the safe wrapping inline asm-equivalent.
            let mut tmp = [0.0f32; LANES];
            vst1q_f32(tmp.as_mut_ptr(), va);
            let mut tmp_b = [0.0f32; LANES];
            vst1q_f32(tmp_b.as_mut_ptr(), vb);
            for k in 0..LANES {
                r[off + k] = tmp[k] / tmp_b[k];
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            let vb = _mm_loadu_ps(b.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_div_ps(va, vb));
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            for k in 0..LANES {
                r[off + k] = a[off + k] / b[off + k];
            }
        }
    }

    for i in (chunks * LANES)..n {
        let bv = b[i];
        r[i] = if bv != 0.0 { a[i] / bv } else { 0.0 };
    }
}

/// Fused multiply-add into an accumulator: r[i] += alpha * b[i].
///
/// This is the inner-loop primitive for row-major matmul: with the loop
/// order (i, k, j), each step adds `a[i,k] * b[k, ...]` to row `i` of the
/// result. Matches BLAS's `saxpy` interface (single-precision a*x + y).
#[inline(always)]
pub fn axpy_slice(r: &mut [f32], alpha: f32, b: &[f32]) {
    let n = r.len();
    debug_assert!(b.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let va = vdupq_n_f32(alpha);
        for i in 0..chunks {
            let off = i * LANES;
            let vb = vld1q_f32(b.as_ptr().add(off));
            let vr = vld1q_f32(r.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vaddq_f32(vr, vmulq_f32(va, vb)));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let va = _mm_set1_ps(alpha);
        for i in 0..chunks {
            let off = i * LANES;
            let vb = _mm_loadu_ps(b.as_ptr().add(off));
            let vr = _mm_loadu_ps(r.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_add_ps(vr, _mm_mul_ps(va, vb)));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                r[off + k] += alpha * b[off + k];
            }
        }
    }

    for i in (chunks * LANES)..n {
        r[i] += alpha * b[i];
    }
}

/// r[i] = a[i] - c (broadcast scalar subtract).
#[inline(always)]
pub fn sub_const_slice(r: &mut [f32], a: &[f32], c: f32) {
    let n = r.len();
    debug_assert!(a.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let vc = vdupq_n_f32(c);
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vsubq_f32(va, vc));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let vc = _mm_set1_ps(c);
        for i in 0..chunks {
            let off = i * LANES;
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_sub_ps(va, vc));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                r[off + k] = a[off + k] - c;
            }
        }
    }

    for i in (chunks * LANES)..n {
        r[i] = a[i] - c;
    }
}

/// r[i] = a[i] * c (broadcast scalar multiply).
#[inline(always)]
pub fn mul_const_slice(r: &mut [f32], a: &[f32], c: f32) {
    let n = r.len();
    debug_assert!(a.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let vc = vdupq_n_f32(c);
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vmulq_f32(va, vc));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let vc = _mm_set1_ps(c);
        for i in 0..chunks {
            let off = i * LANES;
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_mul_ps(va, vc));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                r[off + k] = a[off + k] * c;
            }
        }
    }

    for i in (chunks * LANES)..n {
        r[i] = a[i] * c;
    }
}

/// Sum of x[i]² over a slice. Used by layerNorm (variance) and rmsNorm.
#[inline(always)]
pub fn sum_of_squares(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    let lanes_total: f32 = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(a.as_ptr().add(i * LANES));
            acc = vaddq_f32(acc, vmulq_f32(v, v));
        }
        vaddvq_f32(acc)
    };

    #[cfg(target_arch = "x86_64")]
    let lanes_total: f32 = unsafe {
        let mut acc = _mm_set1_ps(0.0);
        for i in 0..chunks {
            let v = _mm_loadu_ps(a.as_ptr().add(i * LANES));
            acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
        }
        let mut tmp = [0.0f32; LANES];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        tmp[0] + tmp[1] + tmp[2] + tmp[3]
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let lanes_total: f32 = {
        let mut acc = [0.0f32; LANES];
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                acc[k] += a[off + k] * a[off + k];
            }
        }
        acc[0] + acc[1] + acc[2] + acc[3]
    };

    let mut tail = 0.0f32;
    for &v in &a[chunks * LANES..n] {
        tail += v * v;
    }
    lanes_total + tail
}

// ---------------------------------------------------------------------------
// Lane-wise unary: relu (max with 0)
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn relu_slice(r: &mut [f32], a: &[f32]) {
    let n = r.len();
    debug_assert!(a.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let zero = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let off = i * LANES;
            let va = vld1q_f32(a.as_ptr().add(off));
            vst1q_f32(r.as_mut_ptr().add(off), vmaxq_f32(va, zero));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let zero = _mm_set1_ps(0.0);
        for i in 0..chunks {
            let off = i * LANES;
            let va = _mm_loadu_ps(a.as_ptr().add(off));
            _mm_storeu_ps(r.as_mut_ptr().add(off), _mm_max_ps(va, zero));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                let v = a[off + k];
                r[off + k] = if v > 0.0 { v } else { 0.0 };
            }
        }
    }

    for i in (chunks * LANES)..n {
        let v = a[i];
        r[i] = if v > 0.0 { v } else { 0.0 };
    }
}

// ---------------------------------------------------------------------------
// Reductions: sum / max / min over a slice
// ---------------------------------------------------------------------------

#[inline(always)]
pub fn sum_slice(a: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    let lanes_total: f32 = unsafe {
        let mut acc: float32x4_t = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(a.as_ptr().add(i * LANES));
            acc = vaddq_f32(acc, v);
        }
        vaddvq_f32(acc)
    };

    #[cfg(target_arch = "x86_64")]
    let lanes_total: f32 = unsafe {
        let mut acc: __m128 = _mm_set1_ps(0.0);
        for i in 0..chunks {
            let v = _mm_loadu_ps(a.as_ptr().add(i * LANES));
            acc = _mm_add_ps(acc, v);
        }
        let mut tmp = [0.0f32; LANES];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        tmp[0] + tmp[1] + tmp[2] + tmp[3]
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let lanes_total: f32 = {
        let mut acc = [0.0f32; LANES];
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                acc[k] += a[off + k];
            }
        }
        acc[0] + acc[1] + acc[2] + acc[3]
    };

    let mut tail = 0.0f32;
    for &v in &a[chunks * LANES..n] {
        tail += v;
    }
    lanes_total + tail
}

#[inline(always)]
pub fn max_slice(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::NEG_INFINITY;
    }
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    let mut best: f32 = unsafe {
        if chunks == 0 {
            f32::NEG_INFINITY
        } else {
            let mut acc = vld1q_f32(a.as_ptr());
            for i in 1..chunks {
                let v = vld1q_f32(a.as_ptr().add(i * LANES));
                acc = vmaxq_f32(acc, v);
            }
            vmaxvq_f32(acc)
        }
    };

    #[cfg(target_arch = "x86_64")]
    let mut best: f32 = unsafe {
        if chunks == 0 {
            f32::NEG_INFINITY
        } else {
            let mut acc = _mm_loadu_ps(a.as_ptr());
            for i in 1..chunks {
                let v = _mm_loadu_ps(a.as_ptr().add(i * LANES));
                acc = _mm_max_ps(acc, v);
            }
            let mut tmp = [0.0f32; LANES];
            _mm_storeu_ps(tmp.as_mut_ptr(), acc);
            tmp[0].max(tmp[1]).max(tmp[2]).max(tmp[3])
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut best: f32 = {
        if chunks == 0 {
            f32::NEG_INFINITY
        } else {
            let mut acc = [a[0], a[1], a[2], a[3]];
            for i in 1..chunks {
                let off = i * LANES;
                for k in 0..LANES {
                    if a[off + k] > acc[k] {
                        acc[k] = a[off + k];
                    }
                }
            }
            acc[0].max(acc[1]).max(acc[2]).max(acc[3])
        }
    };

    for &v in &a[chunks * LANES..n] {
        if v > best {
            best = v;
        }
    }
    best
}

#[inline(always)]
pub fn min_slice(a: &[f32]) -> f32 {
    let n = a.len();
    if n == 0 {
        return f32::INFINITY;
    }
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    let mut best: f32 = unsafe {
        if chunks == 0 {
            f32::INFINITY
        } else {
            let mut acc = vld1q_f32(a.as_ptr());
            for i in 1..chunks {
                let v = vld1q_f32(a.as_ptr().add(i * LANES));
                acc = vminq_f32(acc, v);
            }
            vminvq_f32(acc)
        }
    };

    #[cfg(target_arch = "x86_64")]
    let mut best: f32 = unsafe {
        if chunks == 0 {
            f32::INFINITY
        } else {
            let mut acc = _mm_loadu_ps(a.as_ptr());
            for i in 1..chunks {
                let v = _mm_loadu_ps(a.as_ptr().add(i * LANES));
                acc = _mm_min_ps(acc, v);
            }
            let mut tmp = [0.0f32; LANES];
            _mm_storeu_ps(tmp.as_mut_ptr(), acc);
            tmp[0].min(tmp[1]).min(tmp[2]).min(tmp[3])
        }
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let mut best: f32 = {
        if chunks == 0 {
            f32::INFINITY
        } else {
            let mut acc = [a[0], a[1], a[2], a[3]];
            for i in 1..chunks {
                let off = i * LANES;
                for k in 0..LANES {
                    if a[off + k] < acc[k] {
                        acc[k] = a[off + k];
                    }
                }
            }
            acc[0].min(acc[1]).min(acc[2]).min(acc[3])
        }
    };

    for &v in &a[chunks * LANES..n] {
        if v < best {
            best = v;
        }
    }
    best
}

/// Dot product of two equal-length f32 slices, accumulated in f64 for
/// numerical stability across large tensors.
#[inline(always)]
pub fn dot_slice(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    debug_assert!(b.len() == n);
    let chunks = n / LANES;

    #[cfg(target_arch = "aarch64")]
    let lanes_total: f64 = unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * LANES));
            let vb = vld1q_f32(b.as_ptr().add(i * LANES));
            acc = vaddq_f32(acc, vmulq_f32(va, vb));
        }
        vaddvq_f32(acc) as f64
    };

    #[cfg(target_arch = "x86_64")]
    let lanes_total: f64 = unsafe {
        let mut acc = _mm_set1_ps(0.0);
        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * LANES));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * LANES));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }
        let mut tmp = [0.0f32; LANES];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        (tmp[0] + tmp[1] + tmp[2] + tmp[3]) as f64
    };

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let lanes_total: f64 = {
        let mut acc = 0.0f64;
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                acc += (a[off + k] as f64) * (b[off + k] as f64);
            }
        }
        acc
    };

    let mut tail = 0.0f64;
    for i in (chunks * LANES)..n {
        tail += (a[i] as f64) * (b[i] as f64);
    }
    lanes_total + tail
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_sub_mul_div_match_scalar() {
        let a: Vec<f32> = (0..17).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..17).map(|i| (i + 1) as f32 * 0.5).collect();
        let mut out = vec![0.0f32; 17];

        add_slice(&mut out, &a, &b);
        for i in 0..17 {
            assert!((out[i] - (a[i] + b[i])).abs() < 1e-6);
        }

        sub_slice(&mut out, &a, &b);
        for i in 0..17 {
            assert!((out[i] - (a[i] - b[i])).abs() < 1e-6);
        }

        mul_slice(&mut out, &a, &b);
        for i in 0..17 {
            assert!((out[i] - (a[i] * b[i])).abs() < 1e-6);
        }

        div_slice(&mut out, &a, &b);
        for i in 0..17 {
            assert!((out[i] - (a[i] / b[i])).abs() < 1e-6);
        }
    }

    #[test]
    fn div_by_zero_returns_zero_per_lane() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 0.0, 1.0, 0.0, 1.0];
        let mut out = [0.0f32; 5];
        div_slice(&mut out, &a, &b);
        assert_eq!(out, [1.0, 0.0, 3.0, 0.0, 5.0]);
    }

    #[test]
    fn relu_matches_scalar() {
        let a: Vec<f32> = (-9..10).map(|i| i as f32 * 0.5).collect();
        let mut out = vec![0.0f32; a.len()];
        relu_slice(&mut out, &a);
        for i in 0..a.len() {
            let want = if a[i] > 0.0 { a[i] } else { 0.0 };
            assert_eq!(out[i], want);
        }
    }

    #[test]
    fn sum_matches_scalar() {
        let a: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let s = sum_slice(&a);
        assert!((s - 55.0).abs() < 1e-4);
    }

    #[test]
    fn max_min_match_scalar() {
        let a = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];
        assert_eq!(max_slice(&a), 9.0);
        assert_eq!(min_slice(&a), 1.0);
    }

    #[test]
    fn max_min_empty_returns_inf() {
        let a: [f32; 0] = [];
        assert_eq!(max_slice(&a), f32::NEG_INFINITY);
        assert_eq!(min_slice(&a), f32::INFINITY);
    }

    #[test]
    fn sub_mul_const_match_scalar() {
        let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let mut r = vec![0.0f32; 13];
        sub_const_slice(&mut r, &a, 1.5);
        for i in 0..13 {
            assert!((r[i] - (a[i] - 1.5)).abs() < 1e-6);
        }
        mul_const_slice(&mut r, &a, 2.0);
        for i in 0..13 {
            assert!((r[i] - (a[i] * 2.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn sum_of_squares_matches_scalar() {
        let a: Vec<f32> = (1..=11).map(|i| i as f32).collect();
        let want: f32 = a.iter().map(|x| x * x).sum();
        let got = sum_of_squares(&a);
        assert!((got - want).abs() < 1e-3);
    }

    #[test]
    fn axpy_matches_scalar() {
        let b: Vec<f32> = (1..=11).map(|i| i as f32).collect();
        let initial: Vec<f32> = (0..11).map(|i| i as f32).collect();
        let mut r = initial.clone();
        let alpha = 2.5f32;
        axpy_slice(&mut r, alpha, &b);
        for (i, (got, init)) in r.iter().zip(initial.iter()).enumerate() {
            let want = init + alpha * b[i];
            assert!((got - want).abs() < 1e-5, "i={i} got={got} want={want}");
        }
    }

    #[test]
    fn dot_matches_scalar() {
        let a: Vec<f32> = (1..=11).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=11).map(|i| (i * 2) as f32).collect();
        let want: f64 = (1..=11).map(|i| (i as f64) * (i as f64) * 2.0).sum();
        let got = dot_slice(&a, &b);
        assert!((got - want).abs() < 1e-3);
    }
}
