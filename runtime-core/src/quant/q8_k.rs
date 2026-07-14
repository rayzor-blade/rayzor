//! Q8_K activation quantisation — `quantize_x_block_q8` builds the internal
//! `Q8Block` shape used by the SDOT inner kernels; `quantize_row_q8_K` builds
//! the llama.cpp-canonical `Q8KBlock` shape used by the public
//! `vec_dot_q4_K_q8_K` API. Both encode the same per-block math; the public
//! variant uses i16 bsums and packs the per-16-elem sums directly.

use crate::floats::roundf;

use super::types::{Q8Block, Q8KBlock, Q4_K_M_BLOCK_SIZE};

/// Quantise a 256-element span of `x` to INT8 with one F32 scale + the 8
/// per-sub-block sums.
///
/// # Safety
/// `x` must reference 256 live f32 elements.
#[inline]
pub unsafe fn quantize_x_block_q8(x: *const f32) -> Q8Block {
    // Pass 1: find the absolute max over 256 elements (NEON 4×4-lane
    // unrolled). Used to pick a symmetric scale so quants land in
    // [-127, 127].
    #[allow(unused_assignments)] // overwritten in aarch64 path, read in fallback
    let mut max_abs = 0.0f32;
    #[cfg(target_arch = "aarch64")]
    {
        use core::arch::aarch64::*;
        let pa = x;
        let mut m0 = vdupq_n_f32(0.0);
        let mut m1 = vdupq_n_f32(0.0);
        let mut m2 = vdupq_n_f32(0.0);
        let mut m3 = vdupq_n_f32(0.0);
        let abs_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x7FFF_FFFF));
        let mut i = 0;
        while i < 256 {
            let v0 = vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(pa.add(i))),
                vreinterpretq_u32_f32(abs_mask),
            );
            let v1 = vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 4))),
                vreinterpretq_u32_f32(abs_mask),
            );
            let v2 = vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 8))),
                vreinterpretq_u32_f32(abs_mask),
            );
            let v3 = vandq_u32(
                vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 12))),
                vreinterpretq_u32_f32(abs_mask),
            );
            m0 = vmaxq_f32(m0, vreinterpretq_f32_u32(v0));
            m1 = vmaxq_f32(m1, vreinterpretq_f32_u32(v1));
            m2 = vmaxq_f32(m2, vreinterpretq_f32_u32(v2));
            m3 = vmaxq_f32(m3, vreinterpretq_f32_u32(v3));
            i += 16;
        }
        let m = vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3));
        max_abs = vmaxvq_f32(m);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..256 {
            let v = (*x.add(i)).abs();
            if v > max_abs {
                max_abs = v;
            }
        }
    }

    let mut block = Q8Block {
        quants: [0i8; 256],
        scale: 0.0,
        bsums: [0i32; 8],
        bsums_16: [0i32; 16],
    };

    if max_abs == 0.0 {
        block.scale = 1.0;
        return block;
    }
    block.scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;

    // Pass 2: quantise + accumulate per-16 sums; pair them up for the
    // Q4_K_M per-32 sums.
    for s16 in 0..16 {
        let mut sum: i32 = 0;
        for j in 0..16 {
            let v = *x.add(s16 * 16 + j) * inv_scale;
            // NO explicit clamp: the absmax scale already bounds
            // roundf(v) to [-127,127], and Rust's `as i8` is itself
            // saturating (NaN→0, out-of-range→clamp). A defensive
            // `.clamp(-128,127)` here is what BROKE wasm at opt>=2: it let
            // LLVM's vectorizer prove the range and downgrade the saturating
            // `fptosi.sat` to a plain `fptosi`, which is UB on the NaN lane
            // (vectorization evaluates all lanes, so a masked-out NaN still
            // traps via the planted `unreachable`). Bare `as i8` keeps the
            // total saturating lowering. Bit-identical for in-range inputs.
            let q = roundf(v) as i8;
            block.quants[s16 * 16 + j] = q;
            sum += q as i32;
        }
        block.bsums_16[s16] = sum;
    }
    for s in 0..8 {
        block.bsums[s] = block.bsums_16[2 * s] + block.bsums_16[2 * s + 1];
    }
    block
}

/// Lazy populator for the X→Q8 cache. Returns a borrowed reference to the
/// slot for block index `b_idx`, quantising the matching 256-element span
/// of X on first access.
///
/// # Safety
/// `x_ptr` must reference at least 256 f32 elements starting at the slice
/// the caller assigned to `b_idx`. `cache` and `init` must have at least
/// `b_idx + 1` entries.
#[inline]
pub unsafe fn x_q8_cache_get<'a>(
    cache: &'a mut [Q8Block],
    init: &mut [bool],
    b_idx: usize,
    x_ptr: *const f32,
) -> &'a Q8Block {
    if !init[b_idx] {
        cache[b_idx] = quantize_x_block_q8(x_ptr);
        init[b_idx] = true;
    }
    &cache[b_idx]
}

/// `quantize_row_q8_K` — convert a row of `x.len() / 256` super-blocks of
/// f32 into Q8_K blocks. Mirrors llama.cpp's `quantize_row_q8_K_ref`:
///
/// - Symmetric absmax scale: `d = absmax / 127`.
/// - Quants: `qs[i] = round(x[i] / d)`, saturating to `[-128, 127]`.
/// - 16 per-16-element sums: `bsums[s] = Σ_{j=0..16} qs[s*16 + j]`.
///
/// Identical scale convention to `quantize_x_block_q8` (which feeds the
/// internal SDOT path) — verified by sharing the inner kernel.
///
/// # Panics
/// Panics if `x.len()` is not a multiple of 256 or if `dest.len()` is smaller
/// than `x.len() / 256`.
#[allow(non_snake_case)] // matches llama.cpp's `quantize_row_q8_K` symbol
pub fn quantize_row_q8_K(x: &[f32], dest: &mut [Q8KBlock]) {
    assert!(
        x.len().is_multiple_of(Q4_K_M_BLOCK_SIZE),
        "quantize_row_q8_K: input length {} must be a multiple of {}",
        x.len(),
        Q4_K_M_BLOCK_SIZE
    );
    let nb = x.len() / Q4_K_M_BLOCK_SIZE;
    assert!(
        dest.len() >= nb,
        "quantize_row_q8_K: dest holds {} blocks but {} required",
        dest.len(),
        nb
    );

    for b in 0..nb {
        let src = &x[b * Q4_K_M_BLOCK_SIZE..(b + 1) * Q4_K_M_BLOCK_SIZE];
        dest[b] = quantize_block_q8_K(src);
    }
}

/// Quantise one 256-element super-block to the canonical `Q8KBlock`.
///
/// NEON path on aarch64, scalar elsewhere — bit-identical by construction:
/// `vcvtaq_s32_f32` (FCVTAS) rounds ties AWAY from zero, exactly matching
/// the scalar `roundf`, and `x*inv_d ∈ [-127, 127]` by the absmax scale so
/// the saturating narrows reproduce the defensive `[-128, 127]` clamp.
#[inline]
#[allow(non_snake_case)] // mirrors llama.cpp's quantize_row_q8_K naming
#[allow(clippy::needless_range_loop)] // kernel-style indexed loop
fn quantize_block_q8_K(src: &[f32]) -> Q8KBlock {
    debug_assert_eq!(src.len(), Q4_K_M_BLOCK_SIZE);

    // Pass 1: absolute max over the 256-element super-block.
    #[allow(unused_assignments)] // overwritten in aarch64 path, read in fallback
    let mut max_abs = 0.0f32;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let pa = src.as_ptr();
        let abs_mask = vdupq_n_u32(0x7FFF_FFFF);
        let mut m0 = vdupq_n_f32(0.0);
        let mut m1 = vdupq_n_f32(0.0);
        let mut m2 = vdupq_n_f32(0.0);
        let mut m3 = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < 256 {
            let v0 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i))), abs_mask);
            let v1 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 4))), abs_mask);
            let v2 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 8))), abs_mask);
            let v3 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 12))), abs_mask);
            m0 = vmaxq_f32(m0, vreinterpretq_f32_u32(v0));
            m1 = vmaxq_f32(m1, vreinterpretq_f32_u32(v1));
            m2 = vmaxq_f32(m2, vreinterpretq_f32_u32(v2));
            m3 = vmaxq_f32(m3, vreinterpretq_f32_u32(v3));
            i += 16;
        }
        max_abs = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
    }
    #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "wasm32")))]
    {
        for &v in src {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    // wasm SIMD128 abs-max: 4-lane parallel f32x4_max of |x|, then a lane fold.
    // `max` is associative/commutative (no NaN in activations), so the winning
    // f32 is bit-identical to the scalar sequential reduction.
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let pa = src.as_ptr();
        let mut m = f32x4_splat(0.0);
        let mut i = 0;
        while i < 256 {
            m = f32x4_max(m, f32x4_abs(v128_load(pa.add(i) as *const v128)));
            i += 4;
        }
        max_abs = f32x4_extract_lane::<0>(m)
            .max(f32x4_extract_lane::<1>(m))
            .max(f32x4_extract_lane::<2>(m))
            .max(f32x4_extract_lane::<3>(m));
    }

    // Degenerate block: zero quants, scale=0, bsums=0. Matches llama.cpp
    // which sets `d=0` and skips the encode loop.
    if max_abs == 0.0 {
        return Q8KBlock {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        };
    }

    let d = max_abs / 127.0;
    let inv_d = 127.0 / max_abs;

    let mut qs = [0i8; 256];
    let mut bsums = [0i16; 16];

    // Pass 2: quant + 16-elem partial sums.
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let pa = src.as_ptr();
        let inv = vdupq_n_f32(inv_d);
        for s in 0..16 {
            let base = s * 16;
            // 16 floats → 4 lanes of i32 (FCVTAS = round ties away, as roundf).
            let q0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(pa.add(base)), inv));
            let q1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(pa.add(base + 4)), inv));
            let q2 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(pa.add(base + 8)), inv));
            let q3 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(pa.add(base + 12)), inv));
            // Saturating narrows i32→i16→i8 (values are within ±127).
            let h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let b8 = vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1));
            vst1q_s8(qs.as_mut_ptr().add(base), b8);
            // Widening horizontal sum of 16 i8s — max |sum| 2032, fits i16.
            bsums[s] = vaddlvq_s8(b8);
        }
    }
    #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "wasm32")))]
    {
        for s in 0..16 {
            let mut sum: i32 = 0;
            for j in 0..16 {
                // No defensive clamp — see quantize_x_block_q8: the clamp
                // lets LLVM downgrade the saturating cast to a UB fptosi that
                // traps when vectorized for wasm. Bare saturating `as i8`.
                let q = roundf(src[s * 16 + j] * inv_d) as i8;
                qs[s * 16 + j] = q;
                sum += q as i32;
            }
            // Saturating cast to i16 keeps us compatible with the on-disk
            // representation (max |sum| = 16 × 127 = 2032 < 2^15, so this
            // never actually saturates for valid input but the cast keeps
            // the type clean).
            bsums[s] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }
    // wasm SIMD128 quant: mirrors the NEON pass. wasm SIMD has no round-ties-
    // away convert, so do it explicitly — trunc(x + copysign(0.5, x)) — which
    // is bit-identical to scalar `roundf` for |x*inv_d| <= 127. Saturating
    // i32->i16->i8 narrows mirror the NEON vqmovn chain; bsum sums the i32
    // lanes (== the i8 values, all <=127) before narrowing.
    #[cfg(target_arch = "wasm32")]
    unsafe {
        use core::arch::wasm32::*;
        let pa = src.as_ptr();
        let inv = f32x4_splat(inv_d);
        let half = f32x4_splat(0.5);
        let signmask = u32x4_splat(0x8000_0000);
        #[inline(always)]
        unsafe fn round_away(x: v128, half: v128, signmask: v128) -> v128 {
            // x + copysign(0.5, x), then truncate toward zero.
            let bias = v128_or(v128_and(x, signmask), half);
            i32x4_trunc_sat_f32x4(f32x4_add(x, bias))
        }
        let qp = qs.as_mut_ptr();
        for s in 0..16 {
            let base = s * 16;
            let q0 = round_away(
                f32x4_mul(v128_load(pa.add(base) as *const v128), inv),
                half,
                signmask,
            );
            let q1 = round_away(
                f32x4_mul(v128_load(pa.add(base + 4) as *const v128), inv),
                half,
                signmask,
            );
            let q2 = round_away(
                f32x4_mul(v128_load(pa.add(base + 8) as *const v128), inv),
                half,
                signmask,
            );
            let q3 = round_away(
                f32x4_mul(v128_load(pa.add(base + 12) as *const v128), inv),
                half,
                signmask,
            );
            let h0 = i16x8_narrow_i32x4(q0, q1);
            let h1 = i16x8_narrow_i32x4(q2, q3);
            let b8 = i8x16_narrow_i16x8(h0, h1);
            v128_store(qp.add(base) as *mut v128, b8);
            let ssum = i32x4_add(i32x4_add(q0, q1), i32x4_add(q2, q3));
            let sum = i32x4_extract_lane::<0>(ssum)
                + i32x4_extract_lane::<1>(ssum)
                + i32x4_extract_lane::<2>(ssum)
                + i32x4_extract_lane::<3>(ssum);
            bsums[s] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
    }

    Q8KBlock { d, qs, bsums }
}

#[cfg(test)]
mod q8k_neon_tests {
    use super::*;

    /// Scalar reference — the exact pre-NEON loop, kept verbatim so the
    /// vector path is checked against known-good semantics (roundf ties
    /// away from zero, defensive [-128,127] clamp, i16 bsums).
    fn quantize_block_q8_k_ref(src: &[f32]) -> Q8KBlock {
        let mut max_abs = 0.0f32;
        for &v in src {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs == 0.0 {
            return Q8KBlock {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            };
        }
        let d = max_abs / 127.0;
        let inv_d = 127.0 / max_abs;
        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];
        for s in 0..16 {
            let mut sum: i32 = 0;
            for j in 0..16 {
                // No defensive clamp — see quantize_x_block_q8: the clamp
                // lets LLVM downgrade the saturating cast to a UB fptosi that
                // traps when vectorized for wasm. Bare saturating `as i8`.
                let q = roundf(src[s * 16 + j] * inv_d) as i8;
                qs[s * 16 + j] = q;
                sum += q as i32;
            }
            bsums[s] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
        Q8KBlock { d, qs, bsums }
    }

    fn assert_block_eq(a: &Q8KBlock, b: &Q8KBlock, label: &str) {
        assert_eq!(a.d.to_bits(), b.d.to_bits(), "{label}: d differs");
        assert_eq!(a.qs, b.qs, "{label}: qs differ");
        assert_eq!(a.bsums, b.bsums, "{label}: bsums differ");
    }

    #[test]
    fn neon_matches_scalar_reference_exactly() {
        // Deterministic LCG so the test is reproducible.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as i32 as f32) / (1 << 20) as f32
        };
        for round in 0..64 {
            let src: Vec<f32> = (0..256).map(|_| next()).collect();
            assert_block_eq(
                &quantize_block_q8_K(&src),
                &quantize_block_q8_k_ref(&src),
                &format!("random round {round}"),
            );
        }
    }

    #[test]
    fn neon_matches_scalar_on_edges() {
        // All zeros (degenerate path).
        let zeros = vec![0.0f32; 256];
        assert_block_eq(
            &quantize_block_q8_K(&zeros),
            &quantize_block_q8_k_ref(&zeros),
            "zeros",
        );
        // Exact rounding ties: with max_abs = 127.0, inv_d = 1.0, so values
        // like 0.5, 1.5, -2.5 hit the round-half boundary exactly — the
        // case where vcvtnq (ties-to-even) would diverge from roundf
        // (ties-away). vcvtaq must match roundf on every one.
        let mut ties = vec![0.0f32; 256];
        ties[0] = 127.0; // pins max_abs so inv_d == 1.0
        for (i, slot) in ties.iter_mut().enumerate().skip(1) {
            let half = (i as f32) - 128.0 + 0.5;
            *slot = half.clamp(-127.0, 127.0);
        }
        assert_block_eq(
            &quantize_block_q8_K(&ties),
            &quantize_block_q8_k_ref(&ties),
            "rounding ties",
        );
        // Saturation guard: ±max everywhere.
        let mut sat = vec![127.5f32; 256];
        for v in sat.iter_mut().skip(128) {
            *v = -127.5;
        }
        assert_block_eq(
            &quantize_block_q8_K(&sat),
            &quantize_block_q8_k_ref(&sat),
            "saturation",
        );
    }
}
