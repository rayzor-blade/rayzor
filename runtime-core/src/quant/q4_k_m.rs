//! Q4_K_M (GGUF) decode / dequant / encode / dequant-fused-matmul kernels.
//!
//! Memory layout (a Q4_K_M super-block, 144 bytes):
//! ```text
//!   bytes  0..1     : f16 d        (super-block scale)
//!   bytes  2..3     : f16 dmin     (super-block min)
//!   bytes  4..15    : 8 × (6-bit scale, 6-bit min) packed in 12 bytes
//!   bytes 16..143   : 128 bytes of 4-bit quants (256 nibbles, low/high
//!                     bit-pairs interleaved across the 8 sub-blocks)
//! ```
//!
//! Dequant formula per element `i` (0..256):
//! ```text
//!   sub = i / 32                     // sub-block index 0..7
//!   q  = nibble[i]                   // 4-bit weight 0..15
//!   sc = d * (scales6[sub] / 63)     // effective scale
//!   mn = dmin * (mins6[sub] / 63)    // effective min
//!   w  = sc * q - mn
//! ```

use half::f16;

use crate::floats::roundf;
use crate::simd::tensor_f32::axpy_slice;

use super::types::{Q4KBlock, Q4KMBlock, Q8KBlock, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_SIZE};

/// Decode the 12-byte (scales, mins) header of a Q4_K_M block.
///
/// GGUF packs eight (6-bit scale, 6-bit min) pairs into 12 bytes via a
/// specific bit layout — this matches `llama.cpp/ggml-quants.c`
/// `get_scale_min_k4`.
#[inline]
pub fn q4_k_get_scale_min(j: usize, header: &[u8; 12]) -> (u8, u8) {
    // j is 0..8 sub-block index.
    // Layout per llama.cpp:
    //   scales[0..4] use low 6 bits of header[0..4]
    //   scales[4..8] use low 6 bits of header[8..12], lower 4 bits from
    //                bytes 8..12, upper 2 bits from bytes 0..4 high bits.
    if j < 4 {
        let sc = header[j] & 63;
        let mn = header[j + 4] & 63;
        (sc, mn)
    } else {
        let sc = (header[j + 4] & 0x0F) | ((header[j - 4] >> 6) << 4);
        let mn = (header[j + 4] >> 4) | ((header[j] >> 6) << 4);
        (sc, mn)
    }
}

/// Decode a 144-byte Q4_K_M super-block into a typed `Q4KBlock` with f32
/// effective scales + mins ready for the dequant kernel.
///
/// # Safety
/// `block_ptr` must reference a live 144-byte Q4_K_M super-block.
#[inline]
pub unsafe fn decode_q4_k_block(block_ptr: *const u8) -> Q4KBlock {
    let d_bits = *(block_ptr as *const u16);
    let dmin_bits = *(block_ptr.add(2) as *const u16);
    let d = f16::from_bits(d_bits).to_f32();
    let dmin = f16::from_bits(dmin_bits).to_f32();

    let mut header = [0u8; 12];
    for (i, slot) in header.iter_mut().enumerate() {
        *slot = *block_ptr.add(4 + i);
    }

    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    for j in 0..8 {
        let (sc6, mn6) = q4_k_get_scale_min(j, &header);
        scales[j] = d * (sc6 as f32);
        mins[j] = dmin * (mn6 as f32);
    }

    let mut quants = [0u8; 128];
    for (i, slot) in quants.iter_mut().enumerate() {
        *slot = *block_ptr.add(16 + i);
    }

    Q4KBlock {
        d,
        dmin,
        scales,
        mins,
        quants,
    }
}

/// Dequant a single Q4_K_M block into 256 f32 values.
///
/// Within a super-block, two adjacent sub-blocks (32 elements each) share
/// 32 bytes of quants: the low nibbles of bytes 0..32 hold sub-block 2*s,
/// the high nibbles hold sub-block 2*s+1.
#[inline]
pub fn dequant_q4_k_block(block: &Q4KBlock, out: &mut [f32]) {
    debug_assert_eq!(out.len(), Q4_K_M_BLOCK_SIZE);

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is unconditional on aarch64; we read 32 bytes per
    // `s` iteration from `block.quants` (size 128, indices 0..127) and
    // write to `out` strictly within `[0, 256)`.
    unsafe {
        use core::arch::aarch64::*;
        let q_ptr = block.quants.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for s in 0..4 {
            let sc_lo = vdupq_n_f32(block.scales[2 * s]);
            // Negate the min once so we can fold `out = sc*q - mn`
            // into `vfmaq_f32(neg_mn, sc, q)` — one FMA instead of
            // mul+sub. Single rounding, one fewer instruction in the
            // inner loop.
            let neg_mn_lo = vdupq_n_f32(-block.mins[2 * s]);
            let sc_hi = vdupq_n_f32(block.scales[2 * s + 1]);
            let neg_mn_hi = vdupq_n_f32(-block.mins[2 * s + 1]);

            // Process the 32 quant bytes in 2 chunks of 16. Each
            // byte's low nibble goes to sub-block 2s, high nibble to
            // sub-block 2s+1.
            for chunk in 0..2 {
                let bytes16 = vld1q_u8(q_ptr.add(s * 32 + chunk * 16));
                let mask = vdupq_n_u8(0x0F);
                let lo = vandq_u8(bytes16, mask);
                let hi = vshrq_n_u8::<4>(bytes16);

                // Widen lo to u16 then to two u32 vectors, convert to f32,
                // apply scale and min.
                let lo_lo16 = vmovl_u8(vget_low_u8(lo));
                let lo_hi16 = vmovl_u8(vget_high_u8(lo));
                let hi_lo16 = vmovl_u8(vget_low_u8(hi));
                let hi_hi16 = vmovl_u8(vget_high_u8(hi));

                let lo_q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_lo16)));
                let lo_q1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_lo16)));
                let lo_q2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo_hi16)));
                let lo_q3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo_hi16)));
                let hi_q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_lo16)));
                let hi_q1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_lo16)));
                let hi_q2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi_hi16)));
                let hi_q3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi_hi16)));

                // out = scale * q - min  ≡ fma(neg_min, scale, q).
                let lo_out0 = vfmaq_f32(neg_mn_lo, sc_lo, lo_q0);
                let lo_out1 = vfmaq_f32(neg_mn_lo, sc_lo, lo_q1);
                let lo_out2 = vfmaq_f32(neg_mn_lo, sc_lo, lo_q2);
                let lo_out3 = vfmaq_f32(neg_mn_lo, sc_lo, lo_q3);
                let hi_out0 = vfmaq_f32(neg_mn_hi, sc_hi, hi_q0);
                let hi_out1 = vfmaq_f32(neg_mn_hi, sc_hi, hi_q1);
                let hi_out2 = vfmaq_f32(neg_mn_hi, sc_hi, hi_q2);
                let hi_out3 = vfmaq_f32(neg_mn_hi, sc_hi, hi_q3);

                // Sub-block 2*s receives 16 outputs starting at
                // (2*s)*32 + chunk*16.
                let lo_dst = out_ptr.add((2 * s) * 32 + chunk * 16);
                vst1q_f32(lo_dst, lo_out0);
                vst1q_f32(lo_dst.add(4), lo_out1);
                vst1q_f32(lo_dst.add(8), lo_out2);
                vst1q_f32(lo_dst.add(12), lo_out3);

                let hi_dst = out_ptr.add((2 * s + 1) * 32 + chunk * 16);
                vst1q_f32(hi_dst, hi_out0);
                vst1q_f32(hi_dst.add(4), hi_out1);
                vst1q_f32(hi_dst.add(8), hi_out2);
                vst1q_f32(hi_dst.add(12), hi_out3);
            }
        }
        return;
    }

    // Scalar fallback — x86_64 (where llvm autovectorises this well
    // anyway), wasm, and the unreachable-after-aarch64 path.
    #[allow(unreachable_code)]
    {
        for s in 0..4 {
            let sc_lo = block.scales[2 * s];
            let mn_lo = block.mins[2 * s];
            let sc_hi = block.scales[2 * s + 1];
            let mn_hi = block.mins[2 * s + 1];
            for i in 0..32 {
                let byte = block.quants[s * 32 + i];
                let q_lo = byte & 0x0F;
                let q_hi = byte >> 4;
                out[(2 * s) * 32 + i] = sc_lo * (q_lo as f32) - mn_lo;
                out[(2 * s + 1) * 32 + i] = sc_hi * (q_hi as f32) - mn_hi;
            }
        }
    }
}

/// Pack 8 (sc6, mn6) pairs into the 12-byte llama.cpp-canonical scales
/// header. Inverse of `q4_k_get_scale_min`. Used by the F32 → Q4_K_M
/// encoder + verified by the round-trip unit test.
#[inline]
pub fn pack_q4_k_scales(sc6: &[u8; 8], mn6: &[u8; 8]) -> [u8; 12] {
    let mut h = [0u8; 12];
    // Indices 0..4: low 6 bits hold sc6 / mn6 directly.
    for j in 0..4 {
        h[j] = sc6[j] & 63;
        h[j + 4] = mn6[j] & 63;
    }
    // Indices 4..8: 6 bits split — low 4 bits in bytes 8..12, upper 2
    // bits stuffed into the upper 2 bits of bytes 0..4 (sc) / 4..8 (mn).
    for j in 4..8 {
        h[j + 4] = (sc6[j] & 0x0F) | ((mn6[j] & 0x0F) << 4);
        h[j - 4] |= ((sc6[j] >> 4) & 0x03) << 6;
        h[j] |= ((mn6[j] >> 4) & 0x03) << 6;
    }
    h
}

/// `vec_dot_q4_K_q8_K` scalar reference — single-super-block dot product
/// between a Q4_K_M weight block and a Q8_K activation block. Portable, no
/// SDOT, no architecture-specific intrinsics. Mirrors llama.cpp's
/// `ggml_vec_dot_q4_K_q8_K` arithmetic exactly:
///
/// ```text
/// Σ_i w[i] * x[i]
///   = Σ_{s=0..8} (d * sc6[s] * Σ_{i∈sub_s} q4[i] * x_q8[i]
///                 - dmin * mn6[s] * Σ_{i∈sub_s} x_q8[i])
///   = x.d * Σ_{s=0..8} (d * sc6[s] * sdot_s
///                       - dmin * mn6[s] * (bsums[2s] + bsums[2s+1]))
/// ```
///
/// On native aarch64+dotprod builds this is the slow path — the SDOT
/// kernels in `quant::sdot` are 2–3x faster on M1. It's the production
/// path for:
///   - wasm32 (no SDOT instruction available, no FMA-on-i8)
///   - aarch64 builds without `target-feature=+dotprod`
///   - x86_64 (until we wire AVX-VNNI)
///   - the per-ULP ground truth in the native unit tests
#[allow(non_snake_case)] // matches llama.cpp's `ggml_vec_dot_q4_K_q8_K` symbol
pub fn vec_dot_q4_K_q8_K_scalar(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    let mut acc = 0.0f32;
    // Decode the 12-byte header into 8 (sc6, mn6) pairs.
    let d = f16::from_bits(weight.d).to_f32();
    let dmin = f16::from_bits(weight.dmin).to_f32();
    let header = weight.scales;
    for s in 0..8 {
        let (sc6, mn6) = q4_k_get_scale_min(s, &header);
        let sub_scale = d * sc6 as f32;
        let sub_min = dmin * mn6 as f32;
        // Sub-block s spans elements s*32 .. (s+1)*32. Within the 128-byte
        // qs, the low-nibble/high-nibble pairing matches dequant_q4_k_block:
        // bytes [p*32 .. p*32+32] hold sub-blocks 2p (low nibbles) and
        // 2p+1 (high nibbles).
        let p = s / 2;
        let is_hi = s & 1 == 1;
        let mut sdot: i32 = 0;
        for i in 0..32 {
            let byte = weight.qs[p * 32 + i];
            let q = if is_hi { byte >> 4 } else { byte & 0x0F } as i32;
            sdot += q * x.qs[s * 32 + i] as i32;
        }
        // bsums[2s] + bsums[2s+1] == sum of 32 x-quants in sub_s.
        let bsum32 = x.bsums[2 * s] as i32 + x.bsums[2 * s + 1] as i32;
        acc += sub_scale * (sdot as f32) - sub_min * (bsum32 as f32);
    }
    x.d * acc
}

/// Dispatch wrapper: wasm32+simd128 uses the vectorised kernel, everything
/// else (native, where the AArch64 SDOT path lives upstream) uses scalar.
#[inline]
pub fn vec_dot_q4_K_q8_K(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        vec_dot_q4_K_q8_K_simd128(weight, x)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        vec_dot_q4_K_q8_K_scalar(weight, x)
    }
}

/// SIMD128 (wasm) Q4_K_M × Q8_K block dot. Bit-identical to the scalar
/// reference: the per-sub-block integer dot is summed via `i32x4.dot_i16x8`
/// + a horizontal add, and integer addition is associative, so `sdot` is
/// exact; the float accumulation order (sub-blocks 0..8) and the
/// `sub_scale*sdot - sub_min*bsum` expression form both match the scalar
/// path verbatim.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn vec_dot_q4_K_q8_K_simd128(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    use core::arch::wasm32::*;

    // i4×i8 dot of two 16-lane vectors → partial i32x4 (sign-extend both
    // halves to i16, two `i32x4.dot_i16x8`). Nibbles are 0..15 so the
    // sign-extend is a no-op on their value.
    // Fallback dot used when `relaxed-simd` is not enabled; the hot path below
    // prefers `i32x4_relaxed_dot_i8x16_i7x16_add` (one op, lowers to NEON SDOT
    // under FEAT_DotProd) when available.
    #[allow(dead_code)]
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

    // Fold x.d into d/dmin up front and accumulate the scaled dot in i32,
    // converting to f32 exactly ONCE at the end — this mirrors the native fast
    // kernel `dot_q4_k_q8_kblock_fast` (sdot.rs) and is bit-identical to it.
    // The previous structure converted each sub-block's sdot to f32 and folded
    // it into a serial f32 `acc` INSIDE the loop (8 i32->f32 converts + a
    // ~4-cycle FP dependency chain per super-block, the dominant ILP killer per
    // the bottleneck analysis). The i32 form exposes far more ILP. Correctness:
    // the per-sub-block i32 sdot is exact (relaxed_dot lowers to NEON SDOT),
    // integer sums are associative, and the final f32 fold order
    // (`-(dmin * Σ min)` then `+ (d * Σ scaled-dot)`) matches the native kernel
    // verbatim, so the result equals the in-production host-fallback path.
    let d = x.d * f16::from_bits(weight.d).to_f32();
    let dmin = x.d * f16::from_bits(weight.dmin).to_f32();
    let header = weight.scales;

    // Decode the 12-byte header ONCE into 8 (scale, min) pairs and index in the
    // loop — vs the old 8 in-loop `q4_k_get_scale_min` calls. Same llama.cpp
    // decode, so identical values.
    let mut scales_arr = [0u8; 8];
    let mut mins_arr = [0u8; 8];
    for s in 0..8 {
        let (sc6, mn6) = q4_k_get_scale_min(s, &header);
        scales_arr[s] = sc6;
        mins_arr[s] = mn6;
    }

    // Min term ONCE: Σ_s (bsum_s · min_s) in i32, folded as `-(dmin · Σ)`.
    let mut min_term: i32 = 0;
    for s in 0..8 {
        let bsum = x.bsums[2 * s] as i32 + x.bsums[2 * s + 1] as i32;
        min_term += bsum * mins_arr[s] as i32;
    }
    let mut sumf = -(dmin * min_term as f32);

    let mask = u8x16_splat(0x0F);
    // Raw pointers; both bases can be under-aligned for v128 (Q4KMBlock is
    // repr(packed); Q8KBlock.qs is 4-aligned). `read_unaligned` lowers to the
    // same wasm `v128.load` (the align is only a hint) with no precondition.
    let w_base = core::ptr::addr_of!(weight.qs) as *const u8;
    let x_base = x.qs.as_ptr() as *const u8;
    #[inline(always)]
    unsafe fn loadu(p: *const u8) -> v128 {
        core::ptr::read_unaligned(p as *const v128)
    }

    // Defer the horizontal reduction. Each sub-block's i32x4 sdot accumulator is
    // scaled by its 6-bit scale in the VECTOR domain and summed into `sumi_v`;
    // the single `hsum` happens once at the end instead of 8× per super-block.
    // wasm has no `addv`, so each in-loop `hsum` lowered to 4 vector→GPR
    // `mov`s + 3 scalar adds (Cranelift won't fuse a hand-written extract chain
    // into `addv`) — ~48 latency-bound micro-ops/super-block that also serialized
    // the sdot drain. Bit-identical: `hsum(Σ acc·scale) == Σ hsum(acc)·scale`
    // (integer mul distributes over the lane sum; both are exact i32). Magnitudes
    // stay well within i32 (≤ ~31M/super-block), same as the prior scalar `sumi`.
    let mut sumi_v = i32x4_splat(0);
    for p in 0..4usize {
        let s_lo = 2 * p; // low-nibble sub-block
        let s_hi = 2 * p + 1; // high-nibble sub-block
        let mut acc_lo = i32x4_splat(0);
        let mut acc_hi = i32x4_splat(0);
        for h in 0..2usize {
            // SAFETY: offsets stay within qs (128 bytes) / x.qs (256 bytes).
            let w = unsafe { loadu(w_base.add(p * 32 + h * 16)) };
            let low = v128_and(w, mask); // 0..15
            let high = u8x16_shr(w, 4); // logical >> 4 → 0..15
            let xl = unsafe { loadu(x_base.add(s_lo * 32 + h * 16)) };
            let xh = unsafe { loadu(x_base.add(s_hi * 32 + h * 16)) };
            // Activation is the signed-i8 operand; the 0..15 nibble is the i7
            // operand (top bit clear). Lowers to NEON SDOT under FEAT_DotProd.
            #[cfg(target_feature = "relaxed-simd")]
            {
                acc_lo = i32x4_relaxed_dot_i8x16_i7x16_add(xl, low, acc_lo);
                acc_hi = i32x4_relaxed_dot_i8x16_i7x16_add(xh, high, acc_hi);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                acc_lo = i32x4_add(acc_lo, dot16(low, xl));
                acc_hi = i32x4_add(acc_hi, dot16(high, xh));
            }
        }
        sumi_v = i32x4_add(
            sumi_v,
            i32x4_mul(acc_lo, i32x4_splat(scales_arr[s_lo] as i32)),
        );
        sumi_v = i32x4_add(
            sumi_v,
            i32x4_mul(acc_hi, i32x4_splat(scales_arr[s_hi] as i32)),
        );
    }
    sumf += d * hsum(sumi_v) as f32;
    sumf
}

/// Encode 256 f32 weights into a single Q4_K_M super-block.
///
/// Naive quantisation strategy — per sub-block (8 sub-blocks of 32), pick
/// scale + min directly from `(max - min)` and `-min`, then quantise those 8
/// (scale, min) pairs to 6-bit each via the super-block d / dmin scalars. No
/// iterative refinement like llama.cpp's `make_qkx2_quants`.
///
/// This is intentionally simpler than the canonical encoder — we lose a few
/// percent of quantisation quality vs llama.cpp's iterative scheme.
/// Acceptable for the lm_head re-quant use case where:
///   1. The original tensor was already quantised once (Q6_K) with the
///      careful encoder, so we're re-quantising a near-Q6_K-precise target;
///      small encoder loss compounds on a low-noise source.
///   2. Greedy / low-temp decode is robust to small per-logit ULPs — argmax
///      over a 128k vocab doesn't shift on 1-2 ULP perturbations to most
///      entries.
///
/// MATCH-on-canonical is the empirical check; commit only if it holds.
///
/// Encoding model (mirrors `dequant_q4_k_block`):
///   value = effective_sc[s] * q[s, i] - effective_mn[s]
///   where effective_sc[s] = d * sc6[s], effective_mn[s] = dmin * mn6[s].
pub fn quantize_block_q4_k_m(x: &[f32; 256]) -> Q4KMBlock {
    // 1. Per-sub-block (sc, mn). 8 sub-blocks of 32 elements each.
    let mut sub_sc = [0.0f32; 8];
    let mut sub_mn = [0.0f32; 8];
    for s in 0..8 {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..32 {
            let v = x[s * 32 + i];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if hi == lo {
            // Constant sub-block — quants all 0, encode the value as -mn.
            sub_sc[s] = 0.0;
            sub_mn[s] = -lo;
        } else {
            sub_sc[s] = (hi - lo) / 15.0;
            sub_mn[s] = -lo;
        }
    }

    // 2. Super-block d, dmin chosen so the largest sub-block sc / mn round
    // to the max 6-bit value (63). All other sub-blocks scale proportionally.
    let max_sc = sub_sc.iter().fold(0.0f32, |a, &b| a.max(b));
    let mut max_abs_mn = 0.0f32;
    let mut any_pos_mn = false;
    let mut any_neg_mn = false;
    for &m in &sub_mn {
        let am = m.abs();
        if am > max_abs_mn {
            max_abs_mn = am;
        }
        if m > 0.0 {
            any_pos_mn = true;
        }
        if m < 0.0 {
            any_neg_mn = true;
        }
    }

    let d_f32 = if max_sc == 0.0 { 0.0 } else { max_sc / 63.0 };
    // Pick dmin sign matching the majority. Mixed-sign case clips the
    // wrong-sign sub-blocks to mn6 = 0 below (some local error, but model
    // weights are typically symmetric so this is rare).
    let dmin_sign = if any_pos_mn && !any_neg_mn {
        1.0
    } else if any_neg_mn && !any_pos_mn {
        -1.0
    } else {
        // Mixed signs — pick the sign of the larger-magnitude side.
        let pos_mag = sub_mn
            .iter()
            .filter(|&&m| m > 0.0)
            .fold(0.0f32, |a, &b| a + b);
        let neg_mag: f32 = sub_mn.iter().filter(|&&m| m < 0.0).map(|&m| -m).sum();
        if pos_mag >= neg_mag {
            1.0
        } else {
            -1.0
        }
    };
    let dmin_f32 = if max_abs_mn == 0.0 {
        0.0
    } else {
        dmin_sign * max_abs_mn / 63.0
    };

    // 3. Quantise per-sub-block sc/mn to 6 bits each.
    let mut sc6 = [0u8; 8];
    let mut mn6 = [0u8; 8];
    if d_f32 != 0.0 {
        for s in 0..8 {
            sc6[s] = (roundf(sub_sc[s] / d_f32) as i32).clamp(0, 63) as u8;
        }
    }
    if dmin_f32 != 0.0 {
        for s in 0..8 {
            let target = sub_mn[s] / dmin_f32;
            // Negative targets (sub-block disagrees with dmin's sign) clip
            // to mn6 = 0 — sub-block decodes as `eff_sc * q` with no min
            // offset, which is the closest representable approximation.
            mn6[s] = if target < 0.0 {
                0
            } else {
                (roundf(target) as i32).clamp(0, 63) as u8
            };
        }
    }

    // 4. Recompute the EFFECTIVE per-sub-block (sc, mn) the decoder will see.
    // Quantising values through these (not the raw sub_sc/sub_mn) produces
    // lower error because the quant step uses the same (sc, mn) the dequant
    // will use.
    let mut eff_sc = [0.0f32; 8];
    let mut eff_mn = [0.0f32; 8];
    for s in 0..8 {
        eff_sc[s] = d_f32 * (sc6[s] as f32);
        eff_mn[s] = dmin_f32 * (mn6[s] as f32);
    }

    // 5. Quantise each value: q = round((x + eff_mn) / eff_sc), clamp [0, 15].
    // Pack two 4-bit quants per output byte. Layout mirrors
    // `dequant_q4_k_block`: bytes[p*32..p*32+32] hold sub-blocks 2p
    // (low nibble) and 2p+1 (high nibble).
    let mut qs = [0u8; 128];
    for p in 0..4 {
        let sc_lo = eff_sc[2 * p];
        let mn_lo = eff_mn[2 * p];
        let sc_hi = eff_sc[2 * p + 1];
        let mn_hi = eff_mn[2 * p + 1];
        for i in 0..32 {
            let v_lo = x[(2 * p) * 32 + i];
            let v_hi = x[(2 * p + 1) * 32 + i];
            let q_lo = if sc_lo == 0.0 {
                0
            } else {
                roundf((v_lo + mn_lo) / sc_lo).clamp(0.0, 15.0) as u8
            };
            let q_hi = if sc_hi == 0.0 {
                0
            } else {
                roundf((v_hi + mn_hi) / sc_hi).clamp(0.0, 15.0) as u8
            };
            qs[p * 32 + i] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
        }
    }

    Q4KMBlock {
        d: f16::from_f32(d_f32).to_bits(),
        dmin: f16::from_f32(dmin_f32).to_bits(),
        scales: pack_q4_k_scales(&sc6, &mn6),
        qs,
    }
}

/// Q4_K_M × f32 matmul. A is `[M, K]` quantised; B is `[K, N]` f32; C is
/// `[M, N]` f32. Dequant happens one block at a time into a small f32 stage
/// buffer, then reuses the existing SIMD axpy kernel for the row-update step.
///
/// # Safety
/// `a_data` must reference `m * (k / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES`
/// live bytes; `b_data[..k*n]`, `c_data[..m*n]` must be live f32 storage;
/// `k` must be a multiple of `Q4_K_M_BLOCK_SIZE`.
#[allow(clippy::needless_range_loop)]
pub unsafe fn q4_k_m_matmul_f32(
    a_data: *const u8,
    b_data: *const f32,
    c_data: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(
        k.is_multiple_of(Q4_K_M_BLOCK_SIZE),
        "Q4_K_M matmul: K must be a multiple of 256"
    );
    let blocks_per_row = k / Q4_K_M_BLOCK_SIZE;
    let mut stage = [0.0f32; Q4_K_M_BLOCK_SIZE];

    for i in 0..m {
        let c_row = c_data.add(i * n);
        core::ptr::write_bytes(c_row, 0, n * core::mem::size_of::<f32>());
        let row_ptr = a_data.add(i * blocks_per_row * Q4_K_M_BLOCK_BYTES);
        for b_idx in 0..blocks_per_row {
            let block_ptr = row_ptr.add(b_idx * Q4_K_M_BLOCK_BYTES);
            let block = decode_q4_k_block(block_ptr);
            dequant_q4_k_block(&block, &mut stage);
            // Now stage[0..256] holds the dequantised f32 weights for this
            // 256-element slice of A's row i. Update C's row i by axpy
            // against the matching 256 rows of B.
            let k_off = b_idx * Q4_K_M_BLOCK_SIZE;
            for p in 0..Q4_K_M_BLOCK_SIZE {
                let a_ik = stage[p];
                if a_ik == 0.0 {
                    continue;
                }
                let b_row = b_data.add((k_off + p) * n);
                let c_slice = core::slice::from_raw_parts_mut(c_row, n);
                let b_slice = core::slice::from_raw_parts(b_row, n);
                axpy_slice(c_slice, a_ik, b_slice);
            }
        }
    }
}

#[cfg(test)]
mod golden_gate {
    //! Golden reference for the pure-Haxe Q4_K_M×Q8_K dot kernel bit-exact gate
    //! (docs/plans/haxe-quant-matmul-vectordot.md, Phase 3). Constructs a fully
    //! deterministic, easily-Haxe-reproducible block pair and prints the scalar
    //! oracle's f32 result. The Haxe test rebuilds the SAME inputs and must match.
    use super::*;

    #[test]
    fn golden_q4_dot_for_haxe_gate() {
        // Weight: d=1.0 (0x3C00), dmin=0.5 (0x3800); sc6[s]=s+1, mn6[s]=1;
        // qs[i]=i (low nibble i&0x0F in 0..15, high nibble i>>4 in 0..7).
        let sc6: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let mn6: [u8; 8] = [1, 1, 1, 1, 1, 1, 1, 1];
        let scales = pack_q4_k_scales(&sc6, &mn6);
        let mut qs = [0u8; 128];
        for i in 0..128 {
            qs[i] = i as u8;
        }
        let weight = Q4KMBlock { d: 0x3C00, dmin: 0x3800, scales, qs };

        // Activation: d=0.01; qs[i]=(i%16)-8 → every 16-block sums to -8,
        // so bsums all -8. qs in -8..7 (well inside the dot's i7 operand range).
        let mut xqs = [0i8; 256];
        for i in 0..256 {
            xqs[i] = ((i % 16) as i32 - 8) as i8;
        }
        let mut bsums = [0i16; 16];
        for s in 0..16 {
            let mut sum = 0i32;
            for j in 0..16 {
                sum += xqs[s * 16 + j] as i32;
            }
            bsums[s] = sum as i16;
        }
        let x = Q8KBlock { d: 0.01, qs: xqs, bsums };

        let dot = vec_dot_q4_K_q8_K_scalar(&weight, &x);
        eprintln!("GOLDEN packed_scales={:?}", scales);
        eprintln!("GOLDEN bsums={:?}", bsums);
        eprintln!("GOLDEN dot_bits=0x{:08x}", dot.to_bits());
        eprintln!("GOLDEN1 packed_scales={:?}", scales);
        eprintln!("GOLDEN1 bsums={:?}", bsums);
        eprintln!("GOLDEN1 dot_bits=0x{:08x}", dot.to_bits());
        eprintln!("GOLDEN1 dot={:.10}", dot);

        // Case 2: fully varied — different d/dmin, per-sub-block scales/mins,
        // per-lane nibbles, and NEGATIVE activations, to exercise the high-
        // nibble path + sign handling that case 1's uniform-ish data misses.
        let sc6b: [u8; 8] = [3, 8, 13, 18, 23, 28, 33, 38];
        let mn6b: [u8; 8] = [2, 5, 8, 11, 14, 17, 20, 23];
        let scales2 = pack_q4_k_scales(&sc6b, &mn6b);
        let mut qs2 = [0u8; 128];
        for i in 0..128 {
            qs2[i] = ((i * 7 + 3) % 256) as u8;
        }
        let weight2 = Q4KMBlock { d: 0x2E66, dmin: 0x2C00, scales: scales2, qs: qs2 };

        let mut xqs2 = [0i8; 256];
        for i in 0..256 {
            xqs2[i] = (((i * 11 + 5) % 64) as i32 - 32) as i8; // -32..31, i7-valid
        }
        let mut bsums2 = [0i16; 16];
        for s in 0..16 {
            let mut sum = 0i32;
            for j in 0..16 {
                sum += xqs2[s * 16 + j] as i32;
            }
            bsums2[s] = sum as i16;
        }
        let x2 = Q8KBlock { d: 0.0375, qs: xqs2, bsums: bsums2 };

        let dot2 = vec_dot_q4_K_q8_K_scalar(&weight2, &x2);
        eprintln!("GOLDEN2 packed_scales={:?}", scales2);
        eprintln!("GOLDEN2 bsums={:?}", bsums2);
        eprintln!("GOLDEN2 dot_bits=0x{:08x}", dot2.to_bits());
        eprintln!("GOLDEN2 dot={:.10}", dot2);
    }
}
