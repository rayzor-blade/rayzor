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

use super::types::{Q4KBlock, Q4KMBlock, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_SIZE};

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
