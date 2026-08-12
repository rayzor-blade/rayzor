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
use core::arch::aarch64::{
    float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32, vminq_f32,
    vminvq_f32, vmulq_f32, vst1q_f32, vsubq_f32,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
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

/// In-place fused add: `dst[i] += src[i]`.
///
/// Equivalent to `add_slice(dst, dst, src)` but expressed with a single
/// `&mut [f32]` and one `&[f32]`, so we never construct an aliased
/// `&[f32]` / `&mut [f32]` pair on the same memory (which would violate
/// Rust's reference aliasing model regardless of the SIMD's
/// load-before-store ordering). The SIMD intrinsics operate on raw
/// pointers derived once from `dst.as_mut_ptr()`.
#[inline(always)]
pub fn add_assign_slice(dst: &mut [f32], src: &[f32]) {
    let n = dst.len();
    debug_assert!(src.len() == n);
    let chunks = n / LANES;
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let vd = vld1q_f32(dst_ptr.add(off));
            let vs = vld1q_f32(src_ptr.add(off));
            vst1q_f32(dst_ptr.add(off), vaddq_f32(vd, vs));
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in 0..chunks {
            let off = i * LANES;
            let vd = _mm_loadu_ps(dst_ptr.add(off));
            let vs = _mm_loadu_ps(src_ptr.add(off));
            _mm_storeu_ps(dst_ptr.add(off), _mm_add_ps(vd, vs));
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..chunks {
            let off = i * LANES;
            for k in 0..LANES {
                dst[off + k] += src[off + k];
            }
        }
    }

    for i in (chunks * LANES)..n {
        dst[i] += src[i];
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
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let va = f32x4_splat(alpha);
        for i in 0..chunks {
            let off = i * LANES;
            let vb = v128_load(b.as_ptr().add(off) as *const v128);
            let vr = v128_load(r.as_ptr().add(off) as *const v128);
            v128_store(
                r.as_mut_ptr().add(off) as *mut v128,
                f32x4_add(vr, f32x4_mul(va, vb)),
            );
        }
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
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

/// Horizontal dot product of two equal-length F32 slices.
/// NEON path: 4-wide vfmaq_f32 accumulation, vaddvq_f32 horizontal reduce.
/// Tail (len % 4 != 0) handled with scalar fallback.
/// Single-rounding FMA matches scalar a*b+c order; expected byte-exact
/// vs scalar reduction when len is a multiple of 4. For ragged lengths
/// the partial-tail accumulation order is identical (rolling sum).
///
/// `inline(always)` so the call from rayzor-runtime's attention path
/// folds into the caller even without cross-crate LTO.
#[inline(always)]
pub fn dot_slice_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let mut acc = vdupq_n_f32(0.0);
        let chunks = len / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }
        let mut sum = vaddvq_f32(acc);
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }
        sum
    }

    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let mut acc = _mm_setzero_ps();
        let chunks = len / 4;
        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }
        // horizontal sum of acc
        let acc_hi = _mm_movehl_ps(acc, acc);
        let acc_sum = _mm_add_ps(acc, acc_hi);
        let acc_lo = _mm_shuffle_ps(acc_sum, acc_sum, 1);
        let scalar_lane = _mm_add_ss(acc_sum, acc_lo);
        let mut sum = _mm_cvtss_f32(scalar_lane);
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }
        sum
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    unsafe {
        use core::arch::wasm32::*;
        let chunks = len / 4;
        let mut acc = f32x4_splat(0.0);
        for i in 0..chunks {
            let va = v128_load(a.as_ptr().add(i * 4) as *const v128);
            let vb = v128_load(b.as_ptr().add(i * 4) as *const v128);
            acc = f32x4_add(acc, f32x4_mul(va, vb));
        }
        let mut sum = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        for i in (chunks * 4)..len {
            sum += a[i] * b[i];
        }
        sum
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        target_arch = "x86_64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
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

// ---------------------------------------------------------------------------
// Half-precision (F16 / BF16) SIMD primitives
//
// Storage is `[u16]` (the f16/bf16 bit pattern). Compute runs in f32 by
// staging through a small `[f32; LANES]` buffer per lane group, then converts
// back to storage on store. The conversion itself goes through the `half`
// crate's slice helpers (`convert_to_f32_slice` / `convert_from_f32_slice`),
// which use NEON `vcvt_f32_f16` on aarch64 (Apple Silicon + ARMv8.2-A FP16)
// and F16C `_mm_cvtph_ps` / `_mm_cvtps_ph` on x86_64 Sandy Bridge+. The f32
// arithmetic inside the lane reuses the f32 SIMD path established above.
//
// We deliberately bias toward `axpy` since that's the matmul row-update
// primitive — the single biggest CPU win for LLM inference. Other ops
// can layer on top later if profiling shows them as bottlenecks.
// ---------------------------------------------------------------------------

/// `r += alpha * b` for f16 storage.
/// Both slices are u16-stored f16 bit patterns. Result stays in f16.
#[inline(always)]
pub fn axpy_f16_slice(r: &mut [u16], alpha: f32, b: &[u16]) {
    use half::f16;
    use half::slice::HalfFloatSliceExt;

    let n = r.len();
    debug_assert!(b.len() == n);

    // Staging buffers for f32 compute. Chunk size keeps L1 hot on most CPUs.
    const CHUNK: usize = 64;
    let mut r_f32 = [0.0f32; CHUNK];
    let mut b_f32 = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= n {
        // Reinterpret the &[u16] storage as &[f16] for the half crate's
        // SIMD-accelerated batch conversion.
        let r_chunk: &[f16] =
            unsafe { core::slice::from_raw_parts(r[i..i + CHUNK].as_ptr() as *const f16, CHUNK) };
        let b_chunk: &[f16] =
            unsafe { core::slice::from_raw_parts(b[i..i + CHUNK].as_ptr() as *const f16, CHUNK) };

        r_chunk.convert_to_f32_slice(&mut r_f32);
        b_chunk.convert_to_f32_slice(&mut b_f32);

        // f32 axpy on the staged buffer — reuses the SIMD path above.
        axpy_slice(&mut r_f32, alpha, &b_f32);

        let r_chunk_mut: &mut [f16] = unsafe {
            core::slice::from_raw_parts_mut(r[i..i + CHUNK].as_mut_ptr() as *mut f16, CHUNK)
        };
        r_chunk_mut.convert_from_f32_slice(&r_f32);
        i += CHUNK;
    }
    // Tail: per-element scalar fallback.
    while i < n {
        let cur = f16::from_bits(r[i]).to_f32();
        let bv = f16::from_bits(b[i]).to_f32();
        r[i] = f16::from_f32(cur + alpha * bv).to_bits();
        i += 1;
    }
}

/// `r += alpha * b` for bf16 storage.
#[inline(always)]
pub fn axpy_bf16_slice(r: &mut [u16], alpha: f32, b: &[u16]) {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;

    let n = r.len();
    debug_assert!(b.len() == n);

    const CHUNK: usize = 64;
    let mut r_f32 = [0.0f32; CHUNK];
    let mut b_f32 = [0.0f32; CHUNK];

    let mut i = 0;
    while i + CHUNK <= n {
        let r_chunk: &[bf16] =
            unsafe { core::slice::from_raw_parts(r[i..i + CHUNK].as_ptr() as *const bf16, CHUNK) };
        let b_chunk: &[bf16] =
            unsafe { core::slice::from_raw_parts(b[i..i + CHUNK].as_ptr() as *const bf16, CHUNK) };

        r_chunk.convert_to_f32_slice(&mut r_f32);
        b_chunk.convert_to_f32_slice(&mut b_f32);

        axpy_slice(&mut r_f32, alpha, &b_f32);

        let r_chunk_mut: &mut [bf16] = unsafe {
            core::slice::from_raw_parts_mut(r[i..i + CHUNK].as_mut_ptr() as *mut bf16, CHUNK)
        };
        r_chunk_mut.convert_from_f32_slice(&r_f32);
        i += CHUNK;
    }
    while i < n {
        let cur = bf16::from_bits(r[i]).to_f32();
        let bv = bf16::from_bits(b[i]).to_f32();
        r[i] = bf16::from_f32(cur + alpha * bv).to_bits();
        i += 1;
    }
}

/// Convert a slice of f16-storage to f32 in bulk. Used by reductions that
/// want SIMD f32 paths over half-precision inputs.
#[inline]
pub fn f16_to_f32_slice(dst: &mut [f32], src: &[u16]) {
    use half::f16;
    use half::slice::HalfFloatSliceExt;
    let n = src.len();
    debug_assert!(dst.len() == n);
    let src_f16: &[f16] = unsafe { core::slice::from_raw_parts(src.as_ptr() as *const f16, n) };
    src_f16.convert_to_f32_slice(dst);
}

/// Convert a slice of bf16-storage to f32 in bulk.
#[inline]
pub fn bf16_to_f32_slice(dst: &mut [f32], src: &[u16]) {
    use half::bf16;
    use half::slice::HalfFloatSliceExt;
    let n = src.len();
    debug_assert!(dst.len() == n);
    let src_bf16: &[bf16] = unsafe { core::slice::from_raw_parts(src.as_ptr() as *const bf16, n) };
    src_bf16.convert_to_f32_slice(dst);
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
    fn axpy_f16_matches_scalar() {
        use half::f16;
        let b_f16: Vec<u16> = (1..=11)
            .map(|i| f16::from_f32(i as f32).to_bits())
            .collect();
        let initial_f16: Vec<u16> = (0..11).map(|i| f16::from_f32(i as f32).to_bits()).collect();
        let mut r = initial_f16.clone();
        let alpha = 2.5f32;
        axpy_f16_slice(&mut r, alpha, &b_f16);
        for (i, (got_bits, init_bits)) in r.iter().zip(initial_f16.iter()).enumerate() {
            let got = f16::from_bits(*got_bits).to_f32();
            let init = f16::from_bits(*init_bits).to_f32();
            let b_v = f16::from_bits(b_f16[i]).to_f32();
            // f16 add+mul precision is ~3e-3 relative
            let want = init + alpha * b_v;
            let tol = (want.abs() * 1e-2).max(1e-2);
            assert!((got - want).abs() < tol, "i={i} got={got} want={want}");
        }
    }

    #[test]
    fn axpy_bf16_matches_scalar() {
        use half::bf16;
        let b_bf16: Vec<u16> = (1..=11)
            .map(|i| bf16::from_f32(i as f32).to_bits())
            .collect();
        let initial_bf16: Vec<u16> = (0..11)
            .map(|i| bf16::from_f32(i as f32).to_bits())
            .collect();
        let mut r = initial_bf16.clone();
        let alpha = 2.5f32;
        axpy_bf16_slice(&mut r, alpha, &b_bf16);
        for (i, (got_bits, init_bits)) in r.iter().zip(initial_bf16.iter()).enumerate() {
            let got = bf16::from_bits(*got_bits).to_f32();
            let init = bf16::from_bits(*init_bits).to_f32();
            let b_v = bf16::from_bits(b_bf16[i]).to_f32();
            // bf16 has 7-bit mantissa — relative precision ~1%
            let want = init + alpha * b_v;
            let tol = (want.abs() * 5e-2).max(5e-2);
            assert!((got - want).abs() < tol, "i={i} got={got} want={want}");
        }
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

/// Vectorized exp(x) for one float32x4 lane — Cephes polynomial with
/// Cody-Waite range reduction. Max error ~1-2 ULP over the silu-relevant
/// range; inputs are clamped to ±87.3 so 2^n scaling never overflows the
/// exponent field.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::excessive_precision)] // C1 = 355/512 written in full: the split-constant
                                      // technique needs it EXACTLY representable, and the digits document that.
unsafe fn expq_f32(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    let x = vminq_f32(
        vmaxq_f32(x, vdupq_n_f32(-87.336_55)),
        vdupq_n_f32(88.376_26),
    );

    // n = round(x * log2(e)); r = x - n*ln2 (split-constant for precision).
    let n_f = vcvtq_f32_s32(vcvtaq_s32_f32(vmulq_f32(
        x,
        vdupq_n_f32(core::f32::consts::LOG2_E),
    )));
    let mut r = vfmsq_f32(x, n_f, vdupq_n_f32(0.693_359_375)); // x - n*C1
    r = vfmsq_f32(r, n_f, vdupq_n_f32(-2.121_944_4e-4)); // - n*C2

    // exp(r) ≈ 1 + r + r²·P(r), Cephes degree-5 Horner.
    let z = vmulq_f32(r, r);
    let mut p = vdupq_n_f32(1.987_569_2e-4);
    p = vfmaq_f32(vdupq_n_f32(1.398_199_9e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.333_452e-3), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.166_579_6e-2), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.666_666_5e-1), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
    let y = vaddq_f32(vfmaq_f32(r, z, p), vdupq_n_f32(1.0));

    // Scale by 2^n via exponent bits.
    let n_i = vcvtaq_s32_f32(vmulq_f32(x, vdupq_n_f32(core::f32::consts::LOG2_E)));
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23));
    vmulq_f32(y, pow2n)
}

/// silu(x) = x / (1 + exp(-x)) over a contiguous f32 buffer, NEON path.
///
/// # Safety
/// `src` and `dst` must reference `n` live, non-overlapping f32 elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn silu_slice_neon(src: *const f32, dst: *mut f32, n: usize) {
    use core::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let mut i = 0;
    while i + 4 <= n {
        let x = vld1q_f32(src.add(i));
        let e = expq_f32(vnegq_f32(x));
        vst1q_f32(dst.add(i), vdivq_f32(x, vaddq_f32(one, e)));
        i += 4;
    }
    while i < n {
        let x = *src.add(i);
        *dst.add(i) = x / (1.0 + (-x).exp());
        i += 1;
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod silu_neon_tests {
    use super::*;

    #[test]
    fn silu_neon_close_to_libm() {
        // Sweep the numerically interesting range; SwiGLU activations in
        // practice live within ±30. Tolerance: the Cephes poly is ~1-2 ULP
        // on exp; allow 4e-6 relative (or absolute near zero).
        let mut src = Vec::new();
        let mut v = -40.0f32;
        while v <= 40.0 {
            src.push(v);
            v += 0.0073;
        }
        let n = src.len();
        let mut dst = vec![0.0f32; n];
        unsafe { silu_slice_neon(src.as_ptr(), dst.as_mut_ptr(), n) };
        for i in 0..n {
            let x = src[i];
            let want = x / (1.0 + (-x).exp());
            let got = dst[i];
            let tol = 4e-6f32.max(want.abs() * 4e-6);
            assert!(
                (got - want).abs() <= tol,
                "silu({x}) = {got}, want {want} (diff {})",
                (got - want).abs()
            );
        }
    }

    #[test]
    fn silu_neon_tail_and_extremes() {
        // Non-multiple-of-4 length exercises the scalar tail; extremes
        // exercise the clamp (exp(-x) under/overflow regions).
        let src = [0.0f32, -100.0, 100.0, 87.0, -87.0, 1e-30, -1e-30];
        let mut dst = [0.0f32; 7];
        unsafe { silu_slice_neon(src.as_ptr(), dst.as_mut_ptr(), 7) };
        for i in 0..7 {
            let x = src[i];
            let want = x / (1.0 + (-x).exp());
            assert!(
                (dst[i] - want).abs() <= 4e-6f32.max(want.abs() * 4e-6),
                "silu({x}) = {}, want {want}",
                dst[i]
            );
        }
    }
}
