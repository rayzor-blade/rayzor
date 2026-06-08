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

use super::types::Q6_K_BLOCK_SIZE;

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
