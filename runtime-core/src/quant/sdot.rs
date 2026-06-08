//! ARMv8.2-A SDOT (`vdotq_s32`) inner kernels for Q4_K_M / Q6_K dot products
//! against INT8-quantised activations.
//!
//! Everything here is gated on `cfg(all(target_arch = "aarch64",
//! target_feature = "dotprod"))` + a function-level `#[target_feature(enable
//! = "dotprod")]` so the SDOT inlining survives across the crate boundary.
//! Compile-time gates ensure pre-ARMv8.2 builds never see these symbols; the
//! runtime SDOT-enabled probe (env-var + `is_aarch64_feature_detected!`) lives
//! in `rayzor-runtime` and decides whether to call into here.
//!
//! Performance wins migrated from runtime/src/quant.rs:
//!   - commit c5ab136 — `dot_q4_k_q8_kblock_llamacpp` (+15.4% real decode,
//!     2.12x kernel microbench, llama.cpp NEON pattern port)
//!   - commit acb80e5 — `dot_q4_k_q8_kblock_2` (+3.04% Voronoi N=300,
//!     two-block paired SDOT with 8 in-flight accumulators)
//!   - commit a241cc9 — partial-accumulator unroll baked into all variants
//!
//! Numerical guarantees stay tight enough that Paris MATCH holds on the
//! canonical prompt — verified by the `vec_dot_q4_k_q8_k_matches_dequant_*`
//! unit tests in `rayzor-runtime::quant::tests`.

#![cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]

use core::arch::aarch64::*;

use super::q4_k_m::q4_k_get_scale_min;
use super::types::{Q4KMBlock, Q8Block, Q8KBlock};

/// Q4_K_M × Q8 dot product for one 256-element block.
///
/// Computes `Σ_i (q4_dequant[i] * x[i])` for the 256 elements covered by one
/// Q4_K_M block, using SDOT instructions on the 4-bit weights versus the
/// INT8-quantised X. Mirrors llama.cpp's `ggml_vec_dot_q4_K_q8_K` block
/// kernel.
///
/// Math: each sub-block `s` contributes
/// ```text
///   sub_scale_s = d * scale6bit_s              (effective f32 scale)
///   sub_min_s   = dmin * min6bit_s             (effective f32 offset)
///   Σ x[i] * (sub_scale_s * q4[i] - sub_min_s)
///     = sub_scale_s * Σ x[i] * q4[i]  -  sub_min_s * Σ x[i]
///     = sub_scale_s * x_scale * sdot(x_q[s..], q4[s..])
///       - sub_min_s * x_scale * bsums[s]
/// ```
///
/// # Safety
/// `q4_block_ptr` must reference a live 144-byte Q4_K_M super-block.
#[inline]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q4_k_q8(q4_block_ptr: *const u8, x_q8: &Q8Block) -> f32 {
    let d_bits = core::ptr::read_unaligned(q4_block_ptr as *const u16);
    let dmin_bits = core::ptr::read_unaligned(q4_block_ptr.add(2) as *const u16);
    let d = half::f16::from_bits(d_bits).to_f32();
    let dmin = half::f16::from_bits(dmin_bits).to_f32();

    // 12-byte header for the 6-bit scales and mins.
    let mut header = [0u8; 12];
    for (i, slot) in header.iter_mut().enumerate() {
        *slot = *q4_block_ptr.add(4 + i);
    }
    let quants_ptr = q4_block_ptr.add(16);

    // 4-way partial accumulators — one per `p` iteration. Breaks the
    // loop-carried dependency through a single `sum_term1 += ...`
    // scalar chain, exposing 4 independent FMA chains to M1's wide OoO
    // scheduler. Within each iteration the two SDOT calls per channel
    // are issued into independent partial accumulators (acc_lo_a /
    // acc_lo_b) instead of a serial chain — M1's vdotq_s32 has ~5
    // cycle latency / ~2 cycle throughput, so the serial form leaves
    // 3 issue slots open per pair of SDOTs.
    //
    // Numerical drift vs the prior serial-add form is bounded by f32
    // reduction-tree reshape (~1 ULP per super-block). Paris MATCH
    // remains byte-exact on the canonical "What is the capital of
    // France?" prompt.
    let mut p1 = [0.0f32; 4]; // Σ sub_scale_s * x_scale * sdot_s, per-p partials
    let mut p2 = [0.0f32; 4]; // Σ sub_min_s   * x_scale * bsum_s, per-p partials

    let mask_nibble = vdupq_n_u8(0x0F);

    // Sub-blocks pair up: bytes [32*p .. 32*p+32] hold the 4-bit quants
    // for sub-blocks 2p (low nibbles) and 2p+1 (high nibbles).
    for p in 0..4 {
        let (sc_lo6, mn_lo6) = q4_k_get_scale_min(2 * p, &header);
        let (sc_hi6, mn_hi6) = q4_k_get_scale_min(2 * p + 1, &header);
        let sub_scale_lo = d * sc_lo6 as f32;
        let sub_min_lo = dmin * mn_lo6 as f32;
        let sub_scale_hi = d * sc_hi6 as f32;
        let sub_min_hi = dmin * mn_hi6 as f32;

        // 32 quant bytes → 32 lo-nibble + 32 hi-nibble values.
        let q1 = vld1q_u8(quants_ptr.add(p * 32));
        let q2 = vld1q_u8(quants_ptr.add(p * 32 + 16));

        // Low nibbles (sub-block 2p), high nibbles (sub-block 2p+1).
        let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask_nibble));
        let lo2 = vreinterpretq_s8_u8(vandq_u8(q2, mask_nibble));
        let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
        let hi2 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q2));

        // X int8 for sub-block 2p (32 i8) and sub-block 2p+1 (32 i8).
        let x_lo_ptr = x_q8.quants.as_ptr().add(2 * p * 32);
        let x_hi_ptr = x_q8.quants.as_ptr().add((2 * p + 1) * 32);
        let xlo1 = vld1q_s8(x_lo_ptr);
        let xlo2 = vld1q_s8(x_lo_ptr.add(16));
        let xhi1 = vld1q_s8(x_hi_ptr);
        let xhi2 = vld1q_s8(x_hi_ptr.add(16));

        // Serial SDOT (matches original code) — proves whether the
        // partial-accumulator unroll alone is enough.
        let mut acc_lo = vdupq_n_s32(0);
        acc_lo = vdotq_s32(acc_lo, xlo1, lo1);
        acc_lo = vdotq_s32(acc_lo, xlo2, lo2);

        let mut acc_hi = vdupq_n_s32(0);
        acc_hi = vdotq_s32(acc_hi, xhi1, hi1);
        acc_hi = vdotq_s32(acc_hi, xhi2, hi2);

        let lo_sdot = vaddvq_s32(acc_lo) as f32;
        let hi_sdot = vaddvq_s32(acc_hi) as f32;

        p1[p] = sub_scale_lo * lo_sdot + sub_scale_hi * hi_sdot;
        p2[p] = sub_min_lo * x_q8.bsums[2 * p] as f32 + sub_min_hi * x_q8.bsums[2 * p + 1] as f32;
    }

    // Balanced reduction tree: pair-wise sum minimises rounding error
    // and exposes independent partial sums to OoO execution.
    let s1 = (p1[0] + p1[1]) + (p1[2] + p1[3]);
    let s2 = (p2[0] + p2[1]) + (p2[2] + p2[3]);

    x_q8.scale * (s1 - s2)
}

/// Q4_K_M × Q8_K dot product — variant of `dot_q4_k_q8` that reads the public
/// `Q8KBlock` shape directly. Skips the per-call shim memcpy
/// (`Q8KBlock → Q8Block`) that used to materialise on every super-block
/// inside `vec_dot_q4_K_q8_K`. With ~65k super-block calls per FFN matmul
/// (one per `(n_row, k_block)` pair) the old shim was burning ~16 MB of pure
/// memcpy bandwidth per matmul — wiping out the +15–25% hoist from
/// `prepare_x_q8k_blocks`.
///
/// Two layout differences vs `dot_q4_k_q8`:
///  - `Q8KBlock.qs` is `[i8; 256]` — read directly via `vld1q_s8`, no copy.
///  - `Q8KBlock.bsums` is `[i16; 16]` (per-16-elem sums); the per-32-elem
///    fold `bsums[2s] + bsums[2s+1]` is performed inline as f32 after the
///    SDOT loop (one widening add per sub-block pair — folded into the
///    same scalar arithmetic as `sum_term2` so the cost is negligible).
///
/// Numerically identical to `dot_q4_k_q8` — verified by the
/// `vec_dot_q4_k_q8_k_matches_dequant_then_dot` and
/// `vec_dot_q4_k_q8_k_zero_block` unit tests (rel < 1e-4 against the
/// dequant-then-dot reference).
///
/// # Safety
/// `weight` and `x` are typed shapes laid out byte-identically to GGUF's
/// `block_q4_K` and `block_q8_K`. Calling this from a CPU without ARMv8.2-A
/// SDOT (`vdotq_s32`) traps with SIGILL — the caller must gate on
/// `is_aarch64_feature_detected!("dotprod")`.
#[inline]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q4_k_q8_kblock(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    let q4_block_ptr = weight as *const Q4KMBlock as *const u8;

    let d_bits = core::ptr::read_unaligned(q4_block_ptr as *const u16);
    let dmin_bits = core::ptr::read_unaligned(q4_block_ptr.add(2) as *const u16);
    let d = half::f16::from_bits(d_bits).to_f32();
    let dmin = half::f16::from_bits(dmin_bits).to_f32();

    // 12-byte header for the 6-bit scales and mins.
    let mut header = [0u8; 12];
    for (i, slot) in header.iter_mut().enumerate() {
        *slot = *q4_block_ptr.add(4 + i);
    }
    let quants_ptr = q4_block_ptr.add(16);

    // See `dot_q4_k_q8` above for the rationale on the 4-way partial
    // unroll + split-acc SDOT. This kernel mirrors that structure for
    // the `Q8KBlock` input shape (i16 bsums folded inline rather than
    // the i32 bsums Q8Block stores).
    let mut p1 = [0.0f32; 4]; // Σ sub_scale_s * x_scale * sdot_s, per-p partials
    let mut p2 = [0.0f32; 4]; // Σ sub_min_s   * x_scale * bsum_s, per-p partials

    let mask_nibble = vdupq_n_u8(0x0F);

    // Q8KBlock.qs is a flat `[i8; 256]` — load straight from `x.qs` rather
    // than the shim buffer. Q8KBlock.bsums is `[i16; 16]`; the i32
    // per-32-elem pair sum is reconstructed inline below.
    let x_qs_ptr = x.qs.as_ptr();

    // Sub-blocks pair up: bytes [32*p .. 32*p+32] hold the 4-bit quants for
    // sub-blocks 2p (low nibbles) and 2p+1 (high nibbles).
    for p in 0..4 {
        let (sc_lo6, mn_lo6) = q4_k_get_scale_min(2 * p, &header);
        let (sc_hi6, mn_hi6) = q4_k_get_scale_min(2 * p + 1, &header);
        let sub_scale_lo = d * sc_lo6 as f32;
        let sub_min_lo = dmin * mn_lo6 as f32;
        let sub_scale_hi = d * sc_hi6 as f32;
        let sub_min_hi = dmin * mn_hi6 as f32;

        // 32 quant bytes → 32 lo-nibble + 32 hi-nibble values.
        let q1 = vld1q_u8(quants_ptr.add(p * 32));
        let q2 = vld1q_u8(quants_ptr.add(p * 32 + 16));

        // Low nibbles (sub-block 2p), high nibbles (sub-block 2p+1).
        let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, mask_nibble));
        let lo2 = vreinterpretq_s8_u8(vandq_u8(q2, mask_nibble));
        let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
        let hi2 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q2));

        // X int8 for sub-block 2p (32 i8) and sub-block 2p+1 (32 i8) — read
        // directly out of Q8KBlock.qs, no copy.
        let x_lo_ptr = x_qs_ptr.add(2 * p * 32);
        let x_hi_ptr = x_qs_ptr.add((2 * p + 1) * 32);
        let xlo1 = vld1q_s8(x_lo_ptr);
        let xlo2 = vld1q_s8(x_lo_ptr.add(16));
        let xhi1 = vld1q_s8(x_hi_ptr);
        let xhi2 = vld1q_s8(x_hi_ptr.add(16));

        // Serial SDOT (matches original code) — proves whether the
        // partial-accumulator unroll alone is enough.
        let mut acc_lo = vdupq_n_s32(0);
        acc_lo = vdotq_s32(acc_lo, xlo1, lo1);
        acc_lo = vdotq_s32(acc_lo, xlo2, lo2);

        let mut acc_hi = vdupq_n_s32(0);
        acc_hi = vdotq_s32(acc_hi, xhi1, hi1);
        acc_hi = vdotq_s32(acc_hi, xhi2, hi2);

        let lo_sdot = vaddvq_s32(acc_lo) as f32;
        let hi_sdot = vaddvq_s32(acc_hi) as f32;

        // Inline pair-sum: Q4_K_M's per-32-elem sub-block sum is
        // `bsums_16[2s] + bsums_16[2s+1]`. Q8KBlock stores those as i16
        // (`bsums[0..16]`); widen to i32 then f32 right where they feed the
        // partial. For sub-block pair `p` (covering Q4_K_M sub-blocks 2p,
        // 2p+1 — elements [64*p .. 64*p+64)) the matching Q8KBlock 16-elem
        // sums live at indices 4p .. 4p+4.
        let bsum_lo32 = x.bsums[4 * p] as i32 + x.bsums[4 * p + 1] as i32;
        let bsum_hi32 = x.bsums[4 * p + 2] as i32 + x.bsums[4 * p + 3] as i32;

        p1[p] = sub_scale_lo * lo_sdot + sub_scale_hi * hi_sdot;
        p2[p] = sub_min_lo * bsum_lo32 as f32 + sub_min_hi * bsum_hi32 as f32;
    }

    // Balanced reduction tree across the 4 partials.
    let s1 = (p1[0] + p1[1]) + (p1[2] + p1[3]);
    let s2 = (p2[0] + p2[1]) + (p2[2] + p2[3]);

    x.d * (s1 - s2)
}

/// Q4_K_M × Q8_K dot product matching llama.cpp's ggml_vec_dot_q4_K_q8_K
/// NEON path (arch/arm/quants.c:2715-2776). Empirically 2.12x faster than
/// `dot_q4_k_q8_kblock` in standalone microbench at /tmp/q4km_paired_bench.rs
/// (7.69 ns/call vs 16.34 ns/call median, N=3 runs on M1 Pro).
///
/// Structural deltas vs `dot_q4_k_q8_kblock`:
///   - **Scale decode ONCE per super-block**: the 12-byte scales/mins header
///     is bit-twiddled into utmp[4] up front; we then index scales[2*j+0..2]
///     inside the inner loop instead of calling `q4_k_get_scale_min` 4x.
///     Saves ~12 ops per super-block.
///   - **bsums × mins in i16→i32 path, computed ONCE**: vpaddq_s16 + 2x
///     vmull_s16 + vaddvq_s32 produces the entire dmin*Σ(bsums*mins) value
///     before the inner loop runs. We previously did this per-iteration with
///     f32 multiplies (4x f32 fcvt + 4x f32 fmul).
///   - **Per-sub-block accumulation in i32**: `sumi1 += vaddvq * scales[j]`
///     stays in int32; the f32 conversion is deferred to the SINGLE
///     `sumf += d * (sumi1 + sumi2) as f32` at the end. We previously did
///     f32 conversion + f32 multiply per p-iter, then f32 reduction across
///     4 partials.
///   - **Serial SDOT chain** (not split-acc unroll): `vdotq(vdotq(0, q4[0],
///     q8[0]), q4[1], q8[1])` accumulates into one register; we previously
///     used `acc_lo` + `acc_hi` × 4 partials for ILP. M1's OoO scheduler
///     extracts the ILP from the serial chain anyway — the unroll was
///     paying register pressure for parallelism the hardware already
///     provides.
///
/// Numerical: the result MAY differ from `dot_q4_k_q8_kblock` by 1-2 ULP
/// because the f32 reduction order changes. Tested at the unit level;
/// production MATCH must be verified at the matmul level.
///
/// # Safety
/// Same contract as [`dot_q4_k_q8_kblock`] — needs aarch64 + SDOT support;
/// the caller must runtime-gate this call.
#[inline]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q4_k_q8_kblock_llamacpp(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    let q4_block_ptr = weight as *const Q4KMBlock as *const u8;

    let d_bits = core::ptr::read_unaligned(q4_block_ptr as *const u16);
    let dmin_bits = core::ptr::read_unaligned(q4_block_ptr.add(2) as *const u16);
    let d = x.d * half::f16::from_bits(d_bits).to_f32();
    let dmin = x.d * half::f16::from_bits(dmin_bits).to_f32();

    let m4b = vdupq_n_u8(0x0f);
    let mzero = vdupq_n_s32(0);

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    // Pair-add bsums → 8 i16 super-sums from 16 bsums.
    let q8sums = vpaddq_s16(
        vld1q_s16(x.bsums.as_ptr()),
        vld1q_s16(x.bsums.as_ptr().add(8)),
    );

    // Decode scales ONCE per super-block. The 12-byte header at offset 4..16
    // packs eight 6-bit scales and eight 6-bit mins via the kmask1/2/3
    // bit-twiddle pattern from llama.cpp.
    let mut utmp = [0u32; 4];
    core::ptr::copy_nonoverlapping(q4_block_ptr.add(4) as *const u32, utmp.as_mut_ptr(), 3);

    let mut mins8 = [0u8; 8];
    let lo_word = utmp[1] & kmask1;
    let hi_word = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    mins8[0..4].copy_from_slice(&lo_word.to_le_bytes());
    mins8[4..8].copy_from_slice(&hi_word.to_le_bytes());

    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[0] &= kmask1;

    // bsums × mins in i16→i32, ONCE per super-block.
    let mins8_u8x8 = vld1_u8(mins8.as_ptr());
    let mins = vreinterpretq_s16_u16(vmovl_u8(mins8_u8x8));
    let prod = vaddq_s32(
        vmull_s16(vget_low_s16(q8sums), vget_low_s16(mins)),
        vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)),
    );
    let mut sumf = -(dmin * vaddvq_s32(prod) as f32);

    // scales[0..8] are u8 (post-decode); we'll index by sub-block.
    let mut scales_arr = [0u8; 8];
    scales_arr[0..4].copy_from_slice(&utmp[0].to_le_bytes());
    scales_arr[4..8].copy_from_slice(&utmp[1].to_le_bytes());

    let mut q4 = q4_block_ptr.add(16);
    let mut q8 = x.qs.as_ptr();

    let mut sumi1: i32 = 0;
    let mut sumi2: i32 = 0;

    // Inner loop: 4 iters. Each handles 64 elements (2 sub-blocks).
    for j in 0..4 {
        // Paired q4 load (32 bytes total).
        let q4bits_0 = vld1q_u8(q4);
        let q4bits_1 = vld1q_u8(q4.add(16));
        q4 = q4.add(32);

        // Lo nibbles (sub-block 2j): q4 & 0x0F.
        let q4_lo_0 = vreinterpretq_s8_u8(vandq_u8(q4bits_0, m4b));
        let q4_lo_1 = vreinterpretq_s8_u8(vandq_u8(q4bits_1, m4b));

        // Paired q8 load (32 bytes — first half of the 64-elem span).
        let q8_0 = vld1q_s8(q8);
        let q8_1 = vld1q_s8(q8.add(16));
        q8 = q8.add(32);

        // Serial SDOT chain on lo: p1 = q4_lo_0·q8_0 + q4_lo_1·q8_1.
        let p1 = vdotq_s32(vdotq_s32(mzero, q4_lo_0, q8_0), q4_lo_1, q8_1);
        // Scalar i32 accumulate: sumi1 += vaddvq(p1) * scales[2j+0].
        sumi1 += vaddvq_s32(p1) * scales_arr[2 * j] as i32;

        // Hi nibbles (sub-block 2j+1): q4 >> 4.
        let q4_hi_0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q4bits_0));
        let q4_hi_1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q4bits_1));

        // Second half of the q8 span.
        let q8_2 = vld1q_s8(q8);
        let q8_3 = vld1q_s8(q8.add(16));
        q8 = q8.add(32);

        let p2 = vdotq_s32(vdotq_s32(mzero, q4_hi_0, q8_2), q4_hi_1, q8_3);
        sumi2 += vaddvq_s32(p2) * scales_arr[2 * j + 1] as i32;
    }

    // Single f32 fold at the end.
    sumf += d * (sumi1 + sumi2) as f32;
    sumf
}

/// Two-block paired Q4_K_M × Q8_K dot product. Returns the per-block dot
/// results for `(weight_a, x_a)` and `(weight_b, x_b)` in one call, processed
/// with their inner loops interleaved.
///
/// Pairing pattern (mirror of llama.cpp's q4_K NEON two-block-per-iter kernel
/// that the candle research called out as a gap vs candle's
/// one-block-per-iter approach):
///   1. Load both 12-byte scale headers up front.
///   2. Per p-iter, fetch BOTH blocks' weight nibbles, BOTH X chunks, and
///      emit SDOTs for both before moving on. This doubles the independent
///      FMA chains visible to M1's OoO scheduler from 4 to 8, which is the
///      practical limit before register pressure starts spilling the partial
///      accumulators.
///   3. Per-block partial reduction at the end is bit-identical to the
///      single-block kernel — each block's result is computed from its own
///      four partials in the same reduction order. So
///      `(dot_q4_k_q8_kblock(a, x_a), dot_q4_k_q8_kblock(b, x_b))` and
///      `dot_q4_k_q8_kblock_2(a, b, x_a, x_b)` produce byte-identical pairs
///      of f32s — verified by the `dot_q4_k_q8_kblock_2_matches_single` unit
///      test.
///
/// # Safety
/// Same contract as [`dot_q4_k_q8_kblock`] — needs aarch64 + SDOT support;
/// the caller must runtime-gate this call.
#[inline]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_q4_k_q8_kblock_2(
    weight_a: &Q4KMBlock,
    weight_b: &Q4KMBlock,
    x_a: &Q8KBlock,
    x_b: &Q8KBlock,
) -> (f32, f32) {
    let ptr_a = weight_a as *const Q4KMBlock as *const u8;
    let ptr_b = weight_b as *const Q4KMBlock as *const u8;

    let d_a = half::f16::from_bits(core::ptr::read_unaligned(ptr_a as *const u16)).to_f32();
    let dmin_a =
        half::f16::from_bits(core::ptr::read_unaligned(ptr_a.add(2) as *const u16)).to_f32();
    let d_b = half::f16::from_bits(core::ptr::read_unaligned(ptr_b as *const u16)).to_f32();
    let dmin_b =
        half::f16::from_bits(core::ptr::read_unaligned(ptr_b.add(2) as *const u16)).to_f32();

    let mut header_a = [0u8; 12];
    let mut header_b = [0u8; 12];
    for i in 0..12 {
        header_a[i] = *ptr_a.add(4 + i);
        header_b[i] = *ptr_b.add(4 + i);
    }

    let quants_a = ptr_a.add(16);
    let quants_b = ptr_b.add(16);

    let mut p1_a = [0.0f32; 4];
    let mut p2_a = [0.0f32; 4];
    let mut p1_b = [0.0f32; 4];
    let mut p2_b = [0.0f32; 4];

    let mask_nibble = vdupq_n_u8(0x0F);

    let x_qs_a = x_a.qs.as_ptr();
    let x_qs_b = x_b.qs.as_ptr();

    for p in 0..4 {
        let (sc_lo6_a, mn_lo6_a) = q4_k_get_scale_min(2 * p, &header_a);
        let (sc_hi6_a, mn_hi6_a) = q4_k_get_scale_min(2 * p + 1, &header_a);
        let (sc_lo6_b, mn_lo6_b) = q4_k_get_scale_min(2 * p, &header_b);
        let (sc_hi6_b, mn_hi6_b) = q4_k_get_scale_min(2 * p + 1, &header_b);

        let sub_scale_lo_a = d_a * sc_lo6_a as f32;
        let sub_min_lo_a = dmin_a * mn_lo6_a as f32;
        let sub_scale_hi_a = d_a * sc_hi6_a as f32;
        let sub_min_hi_a = dmin_a * mn_hi6_a as f32;
        let sub_scale_lo_b = d_b * sc_lo6_b as f32;
        let sub_min_lo_b = dmin_b * mn_lo6_b as f32;
        let sub_scale_hi_b = d_b * sc_hi6_b as f32;
        let sub_min_hi_b = dmin_b * mn_hi6_b as f32;

        // Interleaved weight loads — both blocks' 32-byte quant chunks for
        // sub-block pair `p`.
        let q1_a = vld1q_u8(quants_a.add(p * 32));
        let q2_a = vld1q_u8(quants_a.add(p * 32 + 16));
        let q1_b = vld1q_u8(quants_b.add(p * 32));
        let q2_b = vld1q_u8(quants_b.add(p * 32 + 16));

        let lo1_a = vreinterpretq_s8_u8(vandq_u8(q1_a, mask_nibble));
        let lo2_a = vreinterpretq_s8_u8(vandq_u8(q2_a, mask_nibble));
        let hi1_a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1_a));
        let hi2_a = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q2_a));
        let lo1_b = vreinterpretq_s8_u8(vandq_u8(q1_b, mask_nibble));
        let lo2_b = vreinterpretq_s8_u8(vandq_u8(q2_b, mask_nibble));
        let hi1_b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1_b));
        let hi2_b = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q2_b));

        // Interleaved X int8 loads.
        let x_lo_a = x_qs_a.add(2 * p * 32);
        let x_hi_a = x_qs_a.add((2 * p + 1) * 32);
        let x_lo_b = x_qs_b.add(2 * p * 32);
        let x_hi_b = x_qs_b.add((2 * p + 1) * 32);

        let xlo1_a = vld1q_s8(x_lo_a);
        let xlo2_a = vld1q_s8(x_lo_a.add(16));
        let xhi1_a = vld1q_s8(x_hi_a);
        let xhi2_a = vld1q_s8(x_hi_a.add(16));
        let xlo1_b = vld1q_s8(x_lo_b);
        let xlo2_b = vld1q_s8(x_lo_b.add(16));
        let xhi1_b = vld1q_s8(x_hi_b);
        let xhi2_b = vld1q_s8(x_hi_b.add(16));

        // Eight independent SDOT chains (4 per block) — the OoO scheduler
        // sees all of them issuable in parallel.
        let mut acc_lo_a = vdupq_n_s32(0);
        acc_lo_a = vdotq_s32(acc_lo_a, xlo1_a, lo1_a);
        acc_lo_a = vdotq_s32(acc_lo_a, xlo2_a, lo2_a);
        let mut acc_hi_a = vdupq_n_s32(0);
        acc_hi_a = vdotq_s32(acc_hi_a, xhi1_a, hi1_a);
        acc_hi_a = vdotq_s32(acc_hi_a, xhi2_a, hi2_a);

        let mut acc_lo_b = vdupq_n_s32(0);
        acc_lo_b = vdotq_s32(acc_lo_b, xlo1_b, lo1_b);
        acc_lo_b = vdotq_s32(acc_lo_b, xlo2_b, lo2_b);
        let mut acc_hi_b = vdupq_n_s32(0);
        acc_hi_b = vdotq_s32(acc_hi_b, xhi1_b, hi1_b);
        acc_hi_b = vdotq_s32(acc_hi_b, xhi2_b, hi2_b);

        let lo_sdot_a = vaddvq_s32(acc_lo_a) as f32;
        let hi_sdot_a = vaddvq_s32(acc_hi_a) as f32;
        let lo_sdot_b = vaddvq_s32(acc_lo_b) as f32;
        let hi_sdot_b = vaddvq_s32(acc_hi_b) as f32;

        let bsum_lo32_a = x_a.bsums[4 * p] as i32 + x_a.bsums[4 * p + 1] as i32;
        let bsum_hi32_a = x_a.bsums[4 * p + 2] as i32 + x_a.bsums[4 * p + 3] as i32;
        let bsum_lo32_b = x_b.bsums[4 * p] as i32 + x_b.bsums[4 * p + 1] as i32;
        let bsum_hi32_b = x_b.bsums[4 * p + 2] as i32 + x_b.bsums[4 * p + 3] as i32;

        p1_a[p] = sub_scale_lo_a * lo_sdot_a + sub_scale_hi_a * hi_sdot_a;
        p2_a[p] = sub_min_lo_a * bsum_lo32_a as f32 + sub_min_hi_a * bsum_hi32_a as f32;
        p1_b[p] = sub_scale_lo_b * lo_sdot_b + sub_scale_hi_b * hi_sdot_b;
        p2_b[p] = sub_min_lo_b * bsum_lo32_b as f32 + sub_min_hi_b * bsum_hi32_b as f32;
    }

    let s1_a = (p1_a[0] + p1_a[1]) + (p1_a[2] + p1_a[3]);
    let s2_a = (p2_a[0] + p2_a[1]) + (p2_a[2] + p2_a[3]);
    let s1_b = (p1_b[0] + p1_b[1]) + (p1_b[2] + p1_b[3]);
    let s2_b = (p2_b[0] + p2_b[1]) + (p2_b[2] + p2_b[3]);

    (x_a.d * (s1_a - s2_a), x_b.d * (s1_b - s2_b))
}

/// Q6_K × Q8 dot product for one 256-element block.
///
/// Q6_K layout (210 bytes / super-block, mirrors `dequant_q6_k_block`):
///   ql[128]     : lower 4 bits of each 6-bit quant
///   qh[64]      : upper 2 bits of each 6-bit quant (4 quants / byte)
///   scales[16]  : i8 per-sub-block scale (16 sub-blocks × 16 weights)
///   d           : f16 super-block scale
///
/// Per-quant value = `d * scale_s * (q - 32)`, where `q ∈ [0, 63]`.
///
/// Math, mirroring the Q4_K SDOT trick:
/// ```text
///   Σ x[i] * d * scale_s * (q[i] − 32)
///     = d * scale_s * (Σ x[i] * q[i] − 32 * Σ x[i])
///     = d * scale_s * x_scale * (sdot − 32 * bsum16_s)
/// ```
/// where `bsum16_s` is `Σ quants_int8[s*16 .. (s+1)*16]` from
/// `Q8Block::bsums_16` (populated by `quantize_x_block_q8`).
///
/// Per `n in 0..2` half we read 32 ql + 16 qh bytes, build four 32-weight
/// 6-bit spans, and split each into 16-lo + 16-hi SDOT pairs since each
/// 16-weight Q6_K sub-block has its own scale.
///
/// # Safety
/// `q6_block_ptr` must reference a live 210-byte Q6_K super-block. Calling
/// this without SDOT support traps with SIGILL; the caller must runtime-gate.
#[inline]
#[target_feature(enable = "dotprod")]
#[allow(dead_code)] // future Q6_K SDOT entry — currently unused; see qmatmul_chunk_impl
pub unsafe fn dot_q6_k_q8(q6_block_ptr: *const u8, x_q8: &Q8Block) -> f32 {
    let ql_ptr = q6_block_ptr;
    let qh_ptr = q6_block_ptr.add(128);
    let scales_ptr = q6_block_ptr.add(192) as *const i8;
    let d_bits = core::ptr::read_unaligned(q6_block_ptr.add(208) as *const u16);
    let d = half::f16::from_bits(d_bits).to_f32();

    let mut sum_term1 = 0.0f32;
    let mut sum_term2 = 0.0f32;

    let mask_4bit = vdupq_n_u8(0x0F);

    for n in 0..2 {
        // Each half spans 128 weights in `out`, indexed
        // out[out_off + l + j*32] for j in 0..4 and l in 0..32. Read 32 ql
        // bytes + 32 qh bytes. (`qh` half is 32 bytes; we load it once but
        // shift differently per j.)
        let ql_off = n * 64;
        let qh_off = n * 32;
        let sc_off = n * 8;
        let out_off = n * 128;

        let ql0 = vld1q_u8(ql_ptr.add(ql_off));
        let ql1 = vld1q_u8(ql_ptr.add(ql_off + 16));
        let ql2 = vld1q_u8(ql_ptr.add(ql_off + 32));
        let ql3 = vld1q_u8(ql_ptr.add(ql_off + 48));
        let qh0 = vld1q_u8(qh_ptr.add(qh_off));
        let qh1 = vld1q_u8(qh_ptr.add(qh_off + 16));

        // Reconstruct the four 32-weight spans per half. For each l in 0..32
        // the dequant formula picks one of four ql / qh-shift combinations
        // (see `dequant_q6_k_block`). `shift_amt` is the right-shift of `qh`
        // before masking with 3.
        let bsum_base = n * 8;
        for j in 0..4 {
            // (j, ql, qh-shift) layout, mirrors dequant_q6_k_block:
            //   j=0: ql[ql_off + l]     low4, qh & 3                → out + l
            //   j=1: ql[ql_off + l +32] low4, (qh >> 2) & 3         → out + l + 32
            //   j=2: ql[ql_off + l]     high4, (qh >> 4) & 3        → out + l + 64
            //   j=3: ql[ql_off + l +32] high4, (qh >> 6) & 3        → out + l + 96
            let (ql_part0, ql_part1) = match j {
                0 => (vandq_u8(ql0, mask_4bit), vandq_u8(ql1, mask_4bit)),
                1 => (vandq_u8(ql2, mask_4bit), vandq_u8(ql3, mask_4bit)),
                2 => (vshrq_n_u8::<4>(ql0), vshrq_n_u8::<4>(ql1)),
                3 => (vshrq_n_u8::<4>(ql2), vshrq_n_u8::<4>(ql3)),
                _ => unreachable!(),
            };
            let (qh_part0, qh_part1) = match j {
                0 => (
                    vandq_u8(qh0, vdupq_n_u8(0x03)),
                    vandq_u8(qh1, vdupq_n_u8(0x03)),
                ),
                1 => (
                    vandq_u8(vshrq_n_u8::<2>(qh0), vdupq_n_u8(0x03)),
                    vandq_u8(vshrq_n_u8::<2>(qh1), vdupq_n_u8(0x03)),
                ),
                2 => (
                    vandq_u8(vshrq_n_u8::<4>(qh0), vdupq_n_u8(0x03)),
                    vandq_u8(vshrq_n_u8::<4>(qh1), vdupq_n_u8(0x03)),
                ),
                3 => (vshrq_n_u8::<6>(qh0), vshrq_n_u8::<6>(qh1)),
                _ => unreachable!(),
            };

            // q = ql_part | (qh_part << 4), range [0, 63]. We keep it as u8
            // — SDOT interprets bits as signed, but with values in [0, 63]
            // the high bit is zero so signed == unsigned. The −32 bias is
            // folded out via the `bsums_16` correction below.
            let q_lo16 = vreinterpretq_s8_u8(vorrq_u8(ql_part0, vshlq_n_u8::<4>(qh_part0)));
            let q_hi16 = vreinterpretq_s8_u8(vorrq_u8(ql_part1, vshlq_n_u8::<4>(qh_part1)));

            // 32 i8 of X for this 32-weight span.
            let x_span_off = out_off + j * 32;
            let x_lo = vld1q_s8(x_q8.quants.as_ptr().add(x_span_off));
            let x_hi = vld1q_s8(x_q8.quants.as_ptr().add(x_span_off + 16));

            // SDOT each 16-weight half against its sub-block scale.
            let mut acc_lo = vdupq_n_s32(0);
            acc_lo = vdotq_s32(acc_lo, x_lo, q_lo16);
            let sdot_lo = vaddvq_s32(acc_lo) as f32;
            let mut acc_hi = vdupq_n_s32(0);
            acc_hi = vdotq_s32(acc_hi, x_hi, q_hi16);
            let sdot_hi = vaddvq_s32(acc_hi) as f32;

            // The 32-weight span splits across two 16-elem Q6_K sub-blocks
            // with separate scales — the existing `scales[]` array indexes
            // sc_off + 2*j (lo half) and sc_off + 2*j+1 (hi half) for the
            // first 32 weights of this `n`-half; and sc_off + 2*j + 1 /
            // sc_off + 2*j for the four spans collectively cover sub-blocks
            // sc_off+0..sc_off+8. To match `dequant_q6_k_block`'s scale
            // picks, the lookup is `sc_off + 2*j + (l/16)`.
            let scale_lo = *scales_ptr.add(sc_off + 2 * j) as f32;
            let scale_hi = *scales_ptr.add(sc_off + 2 * j + 1) as f32;
            let d_sc_lo = d * scale_lo;
            let d_sc_hi = d * scale_hi;

            sum_term1 += d_sc_lo * sdot_lo;
            sum_term1 += d_sc_hi * sdot_hi;

            // bsums_16 is indexed by 16-elem sub-block of X. The two halves
            // of this 32-weight span use bsums_16[bsum_base + 2*j] (lo) and
            // bsums_16[bsum_base + 2*j + 1] (hi). The outer base offset is
            // the first 16-block in this n-half: bsum_16 index =
            // x_span_off / 16.
            let bsum_lo = x_q8.bsums_16[x_span_off / 16] as f32;
            let bsum_hi = x_q8.bsums_16[(x_span_off / 16) + 1] as f32;
            // Suppress the warning even though bsum_base is the
            // sub-block-pair start — the index above already encodes it.
            let _ = bsum_base;

            sum_term2 += 32.0 * d_sc_lo * bsum_lo;
            sum_term2 += 32.0 * d_sc_hi * bsum_hi;
        }
    }

    x_q8.scale * (sum_term1 - sum_term2)
}
