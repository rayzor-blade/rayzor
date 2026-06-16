//! Q6_K (GGUF) dequantisation kernel.
//!
//! Layout (210 bytes per super-block of 256 weights):
//!   ql[128]    : lower 4 bits of each 6-bit quant
//!   qh[64]     : upper 2 bits of each 6-bit quant (4 quants packed per byte)
//!   scales[16] : i8 per-sub-block scales (16 sub-blocks × 16 weights each)
//!   d          : f16 super-block scale
//!
//! Final value = d * scale * (q - 32), where q is the recombined 6-bit value
//! (0..63) with a -32 bias to make it signed. Mirrors `ggml_dequant_row_q6_K`
//! in llama.cpp.

use half::f16;

use super::types::{Q8KBlock, Q6_K_BLOCK_SIZE};

/// Dispatch wrapper: wasm32+simd128 uses the vectorised Q6_K×Q8_K dot,
/// everything else uses the scalar reference. The dot prequantizes the
/// activation to Q8_K (`x`) exactly like the Q4_K_M path, so Q6_K matmuls
/// stop paying the per-column full f32 dequant.
///
/// # Safety
/// `block_ptr` must reference a live 210-byte Q6_K super-block.
#[inline]
#[allow(non_snake_case)]
pub unsafe fn vec_dot_q6_K_q8_K(block_ptr: *const u8, x: &Q8KBlock) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        unsafe { vec_dot_q6_K_q8_K_simd128(block_ptr, x) }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        unsafe { vec_dot_q6_K_q8_K_scalar(block_ptr, x) }
    }
}

/// Scalar reference Q6_K×Q8_K dot: dequant the weight block to f32 and dot
/// against the dequantized activation (`x.d * x.qs`). This is the proven
/// dequant-then-dot path; it's the wasm non-simd fallback and the test
/// oracle. (Native does not use this dispatcher — it has its own NEON
/// `dot_q6_k_q8`.)
///
/// # Safety
/// `block_ptr` must reference a live 210-byte Q6_K super-block.
#[allow(non_snake_case)]
pub unsafe fn vec_dot_q6_K_q8_K_scalar(block_ptr: *const u8, x: &Q8KBlock) -> f32 {
    let mut w = [0.0f32; Q6_K_BLOCK_SIZE];
    dequant_q6_k_block(block_ptr, &mut w);
    let mut acc = 0.0f32;
    for i in 0..Q6_K_BLOCK_SIZE {
        acc += w[i] * (x.d * (*x.qs.get_unchecked(i) as f32));
    }
    acc
}

/// SIMD128 (wasm) Q6_K×Q8_K dot — port of the native NEON `dot_q6_k_q8`
/// (sdot.rs). With `relaxed-simd` it uses `i32x4.relaxed_dot_i8x16_i7x16_add_s`
/// (one op, lowers to NEON SDOT on FEAT_DotProd hosts); otherwise it falls back
/// to the `dot16` extend + `i32x4.dot_i16x8` the Q4_K_M kernel uses. The 6-bit
/// weights (0..63 = valid i7) reconstruct from ql nibbles +
/// qh 2-bit pairs; the −32 bias folds into `32·Σx` via the per-16 bsums.
/// Bit-exact vs the scalar reference (integer dots are associative; float
/// accumulation order matches).
///
/// # Safety
/// `block_ptr` must reference a live 210-byte Q6_K super-block.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(non_snake_case)]
pub unsafe fn vec_dot_q6_K_q8_K_simd128(block_ptr: *const u8, x: &Q8KBlock) -> f32 {
    use core::arch::wasm32::*;

    #[allow(dead_code)] // unused when `relaxed-simd` is enabled (see below)
    #[inline(always)]
    fn dot16(a: v128, b: v128) -> v128 {
        let a_lo = i16x8_extend_low_i8x16(a);
        let a_hi = i16x8_extend_high_i8x16(a);
        let b_lo = i16x8_extend_low_i8x16(b);
        let b_hi = i16x8_extend_high_i8x16(b);
        i32x4_add(i32x4_dot_i16x8(a_lo, b_lo), i32x4_dot_i16x8(a_hi, b_hi))
    }
    #[inline(always)]
    fn hsum(v: v128) -> i32 {
        i32x4_extract_lane::<0>(v)
            + i32x4_extract_lane::<1>(v)
            + i32x4_extract_lane::<2>(v)
            + i32x4_extract_lane::<3>(v)
    }
    #[inline(always)]
    unsafe fn loadu(p: *const u8) -> v128 {
        core::ptr::read_unaligned(p as *const v128)
    }

    let ql_ptr = block_ptr;
    let qh_ptr = block_ptr.add(128);
    let scales_ptr = block_ptr.add(192) as *const i8;
    let d_bits = core::ptr::read_unaligned(block_ptr.add(208) as *const u16);
    let d = f16::from_bits(d_bits).to_f32();
    let x_base = x.qs.as_ptr() as *const u8;
    let mask = u8x16_splat(0x0F);
    let mask2 = u8x16_splat(0x03);

    let mut sum_term1 = 0.0f32;
    let mut sum_term2 = 0.0f32;
    for n in 0..2usize {
        let ql_off = n * 64;
        let qh_off = n * 32;
        let sc_off = n * 8;
        let out_off = n * 128;

        let ql0 = loadu(ql_ptr.add(ql_off));
        let ql1 = loadu(ql_ptr.add(ql_off + 16));
        let ql2 = loadu(ql_ptr.add(ql_off + 32));
        let ql3 = loadu(ql_ptr.add(ql_off + 48));
        let qh0 = loadu(qh_ptr.add(qh_off));
        let qh1 = loadu(qh_ptr.add(qh_off + 16));

        for j in 0..4usize {
            let (ql_p0, ql_p1) = match j {
                0 => (v128_and(ql0, mask), v128_and(ql1, mask)),
                1 => (v128_and(ql2, mask), v128_and(ql3, mask)),
                2 => (u8x16_shr(ql0, 4), u8x16_shr(ql1, 4)),
                _ => (u8x16_shr(ql2, 4), u8x16_shr(ql3, 4)),
            };
            let (qh_p0, qh_p1) = match j {
                0 => (v128_and(qh0, mask2), v128_and(qh1, mask2)),
                1 => (
                    v128_and(u8x16_shr(qh0, 2), mask2),
                    v128_and(u8x16_shr(qh1, 2), mask2),
                ),
                2 => (
                    v128_and(u8x16_shr(qh0, 4), mask2),
                    v128_and(u8x16_shr(qh1, 4), mask2),
                ),
                _ => (u8x16_shr(qh0, 6), u8x16_shr(qh1, 6)),
            };
            // q = ql_part | (qh_part << 4), range 0..63 (top bit clear → the
            // i8 sign-extend in dot16 reads it as the positive value).
            let q_lo = v128_or(ql_p0, u8x16_shl(qh_p0, 4));
            let q_hi = v128_or(ql_p1, u8x16_shl(qh_p1, 4));

            let x_span = out_off + j * 32;
            let x_lo = loadu(x_base.add(x_span));
            let x_hi = loadu(x_base.add(x_span + 16));

            // Activation (signed i8) × 6-bit weight (0..63 = valid i7),
            // accumulating from zero. relaxed_dot lowers to NEON SDOT; exact vs
            // dot16 (q*x products never saturate the i16 intermediate).
            #[cfg(target_feature = "relaxed-simd")]
            let (sdot_lo, sdot_hi) = (
                hsum(i32x4_relaxed_dot_i8x16_i7x16_add(
                    x_lo,
                    q_lo,
                    i32x4_splat(0),
                )),
                hsum(i32x4_relaxed_dot_i8x16_i7x16_add(
                    x_hi,
                    q_hi,
                    i32x4_splat(0),
                )),
            );
            #[cfg(not(target_feature = "relaxed-simd"))]
            let (sdot_lo, sdot_hi) = (hsum(dot16(x_lo, q_lo)), hsum(dot16(x_hi, q_hi)));

            let sc_lo = *scales_ptr.add(sc_off + 2 * j) as f32;
            let sc_hi = *scales_ptr.add(sc_off + 2 * j + 1) as f32;
            let d_sc_lo = d * sc_lo;
            let d_sc_hi = d * sc_hi;
            sum_term1 += d_sc_lo * sdot_lo as f32;
            sum_term1 += d_sc_hi * sdot_hi as f32;

            let bsum_lo = x.bsums[x_span / 16] as i32;
            let bsum_hi = x.bsums[x_span / 16 + 1] as i32;
            sum_term2 += 32.0 * d_sc_lo * bsum_lo as f32;
            sum_term2 += 32.0 * d_sc_hi * bsum_hi as f32;
        }
    }
    x.d * (sum_term1 - sum_term2)
}

/// Dequant a single Q6_K block into 256 f32 values.
///
/// 2026-06-04 perf note — a hand-written aarch64 NEON port of this kernel
/// (lane-loaded ql/qh bytes → vand/vshr unpack → widen i8 → i32 →
/// vcvtq_f32_s32 → vmulq_f32 by pre-broadcast `d * scale`) was tried and
/// measured 13-17% SLOWER than this scalar version on M1 Pro nue/llama-chat
/// 80-token warm cache (20.0 → 17.5 tok/s). LLVM's auto-vectorizer produces
/// tighter code than the hand-rolled NEON, likely due to better register
/// allocation across the strided stores at positions l, l+32, l+64, l+96 and
/// amortising the 4×16 scale broadcasts across the whole half. A future NEON
/// port that beats the scalar must also handle the FULL kernel including the
/// dot product (i.e. fused dequant + dot with SIMD throughout) — see the
/// failed scalar-fused attempt at commit 31c8a2d which captures why a
/// half-measure loses.
///
/// # Safety
/// `block_ptr` must reference a live 210-byte Q6_K super-block. `out` must
/// be exactly `Q6_K_BLOCK_SIZE` (256) f32 entries.
#[inline]
pub unsafe fn dequant_q6_k_block(block_ptr: *const u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), Q6_K_BLOCK_SIZE);
    let ql = core::slice::from_raw_parts(block_ptr, 128);
    let qh = core::slice::from_raw_parts(block_ptr.add(128), 64);
    let scales = core::slice::from_raw_parts(block_ptr.add(192) as *const i8, 16);
    let d_bits = core::ptr::read_unaligned(block_ptr.add(208) as *const u16);
    let d = f16::from_bits(d_bits).to_f32();

    // Two halves of 128 weights each. Per half:
    //   - ql_off advances by 64 bytes (lower nibbles span 128 weights)
    //   - qh_off advances by 32 bytes (2-bit quants span 128 weights)
    //   - sc_off advances by 8 (8 scales per half)
    for n in 0..2 {
        let ql_off = n * 64;
        let qh_off = n * 32;
        let sc_off = n * 8;
        let out_off = n * 128;
        for l in 0..32 {
            // 4 quants per (l, n) — at positions 0, 32, 64, 96 within the half.
            // Each takes 4 bits from ql + 2 bits from qh.
            let qh_byte = qh[qh_off + l];
            let q1 = ((ql[ql_off + l] & 0x0F) as i32) | (((qh_byte & 3) as i32) << 4);
            let q2 = ((ql[ql_off + l + 32] & 0x0F) as i32) | ((((qh_byte >> 2) & 3) as i32) << 4);
            let q3 = ((ql[ql_off + l] >> 4) as i32) | ((((qh_byte >> 4) & 3) as i32) << 4);
            let q4 = ((ql[ql_off + l + 32] >> 4) as i32) | ((((qh_byte >> 6) & 3) as i32) << 4);

            // Sub-block scale index: l < 16 → first scale slot, l >= 16 → second.
            let is_idx = l / 16;
            let s0 = scales[sc_off + is_idx] as i32;
            let s2 = scales[sc_off + 2 + is_idx] as i32;
            let s4 = scales[sc_off + 4 + is_idx] as i32;
            let s6 = scales[sc_off + 6 + is_idx] as i32;

            // Bias of -32 makes the unsigned 6-bit value (0..63) signed (-32..31).
            out[out_off + l] = d * (s0 as f32) * ((q1 - 32) as f32);
            out[out_off + l + 32] = d * (s2 as f32) * ((q2 - 32) as f32);
            out[out_off + l + 64] = d * (s4 as f32) * ((q3 - 32) as f32);
            out[out_off + l + 96] = d * (s6 as f32) * ((q4 - 32) as f32);
        }
    }
}
