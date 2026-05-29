//! Quantised tensor runtime — INT8 + Q4_K_M storage with dequant-fused matmul.
//!
//! The two quantisation schemes shipped in Phase 4 target the two big LLM
//! inference use-cases:
//!
//! - **INT8 symmetric per-row**: 8-bit weights, one f32 scale per row. The
//!   simplest scheme that still cuts memory 4× vs F32. Used by quantisation
//!   pipelines that need accuracy headroom (most attention QKV projections).
//!
//! - **Q4_K_M** (llama.cpp / GGUF format): 4-bit weights packed in 256-element
//!   super-blocks. Each super-block carries one f16 scale, one f16 min, and
//!   eight 6-bit (scale, min) pairs — one pair per 32-element sub-block. The
//!   workhorse format for shipping Llama-class models to edge: 4.5 bits per
//!   weight (8 bits → 4 bits weight + amortised metadata).
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
//!
//! The runtime exposes a small FFI surface that the Haxe stdlib maps to
//! `rayzor.ds.QTensor`. All extern functions take/return i64 to match the
//! Haxe type system; pointers are reinterpreted on the way in.

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

use half::f16;

/// Quantisation scheme tag. Used by the Haxe-side `QScheme` enum.
pub const QSCHEME_INT8: u8 = 0;
pub const QSCHEME_Q4_K_M: u8 = 1;
pub const QSCHEME_Q6_K: u8 = 2;

/// Q4_K_M block dimensions. These are fixed by the GGUF spec.
pub const Q4_K_M_BLOCK_SIZE: usize = 256;
pub const Q4_K_M_BLOCK_BYTES: usize = 144;

/// Q6_K block dimensions. Despite the "Q4_K_M" model-suffix using both,
/// GGUF's dtype 14 is Q6_K (token_embd, attn_v, ffn_down in K_M variants).
/// Layout (210 bytes): ql[128] + qh[64] + scales[16, i8] + d[f16].
pub const Q6_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Internal opaque tensor representation. The layout depends on `scheme`:
///
/// - `INT8`: `data` is a packed `i8` array of `numel` elements; `meta` is a
///   `f32` array of `meta_len = numel / group_size` per-group scales. The
///   layout is per-row symmetric quant: each row of `group_size` elements
///   shares one scale. A 4096×4096 INT8 matrix with group_size=4096 has
///   one scale per row → 4096 scales.
///
/// - `Q4_K_M`: `data` is a contiguous array of `numel / 256` super-blocks
///   (each 144 bytes); `meta` is empty (None) since metadata is embedded
///   per super-block.
#[repr(C)]
struct RayzorQTensor {
    data: *mut u8,
    meta: *mut f32, // nullable; INT8 scales OR null for Q4_K_M
    numel: usize,
    group_size: usize, // INT8: elements per scale; Q4_K_M: fixed 256
    scheme: u8,
    owns_data: bool,
    // 2-D layout for matmul: stored as [rows, cols] row-major. For 1-D
    // tensors this is (1, numel).
    rows: usize,
    cols: usize,
}

impl RayzorQTensor {
    #[allow(dead_code)]
    #[inline]
    fn data_bytes(&self) -> usize {
        match self.scheme {
            QSCHEME_INT8 => self.numel,
            QSCHEME_Q4_K_M => (self.numel / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES,
            QSCHEME_Q6_K => (self.numel / Q6_K_BLOCK_SIZE) * Q6_K_BLOCK_BYTES,
            _ => 0,
        }
    }
}

// ============================================================================
// INT8 symmetric per-row quantisation
// ============================================================================

/// Quantise an f32 row into int8 + per-row scale.
/// Returns the scale; writes the quantised bytes into `dst`.
fn quantise_int8_row(src: &[f32], dst: &mut [i8]) -> f32 {
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
        let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
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
#[allow(clippy::needless_range_loop)]
unsafe fn int8_matmul_f32(
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
        std::ptr::write_bytes(c_row, 0, n * std::mem::size_of::<f32>());
        for p in 0..k {
            let a_ik = *a_row.add(p) as f32 * row_scale;
            let b_row = b_data.add(p * n);
            // Equivalent to axpy_slice(c_row, a_ik, b_row).
            let c_slice = std::slice::from_raw_parts_mut(c_row, n);
            let b_slice = std::slice::from_raw_parts(b_row, n);
            crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
        }
    }
}

// ============================================================================
// Q4_K_M (GGUF) quantisation
// ============================================================================

/// In-memory view of a 144-byte Q4_K_M super-block. Decoded once into f32
/// scales + mins at dequant time so the inner kernel can stay arithmetic.
/// `d` and `dmin` are kept on the struct for diagnostics + unit tests even
/// though the hot kernel only consumes the already-scaled `scales`/`mins`.
struct Q4KBlock {
    #[allow(dead_code)]
    d: f32,
    #[allow(dead_code)]
    dmin: f32,
    scales: [f32; 8],  // per-sub-block effective scale (already multiplied by d)
    mins: [f32; 8],    // per-sub-block effective min (already multiplied by dmin)
    quants: [u8; 128], // 256 nibbles
}

/// Decode the 12-byte (scales, mins) header of a Q4_K_M block.
/// GGUF packs eight (6-bit scale, 6-bit min) pairs into 12 bytes via a
/// specific bit layout — this matches `llama.cpp/ggml-quants.c`
/// `get_scale_min_k4`.
#[inline]
fn q4_k_get_scale_min(j: usize, header: &[u8; 12]) -> (u8, u8) {
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

#[inline]
unsafe fn decode_q4_k_block(block_ptr: *const u8) -> Q4KBlock {
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

/// Dequant a single Q6_K block into 256 f32 values.
///
/// Q6_K layout (210 bytes per super-block of 256 weights):
///   ql[128]    : lower 4 bits of each 6-bit quant
///   qh[64]     : upper 2 bits of each 6-bit quant (4 quants packed per byte)
///   scales[16] : i8 per-sub-block scales (16 sub-blocks × 16 weights each)
///   d          : f16 super-block scale
///
/// Final value = d * scale * (q - 32), where q is the recombined 6-bit value
/// (0..63) with a -32 bias to make it signed.
///
/// Mirrors `ggml_dequant_row_q6_K` in llama.cpp.
#[inline]
unsafe fn dequant_q6_k_block(block_ptr: *const u8, out: &mut [f32]) {
    debug_assert_eq!(out.len(), Q6_K_BLOCK_SIZE);
    let ql = std::slice::from_raw_parts(block_ptr, 128);
    let qh = std::slice::from_raw_parts(block_ptr.add(128), 64);
    let scales = std::slice::from_raw_parts(block_ptr.add(192) as *const i8, 16);
    let d_bits = std::ptr::read_unaligned(block_ptr.add(208) as *const u16);
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

/// Dequant a single Q4_K_M block into 256 f32 values.
///
/// Within a super-block, two adjacent sub-blocks (32 elements each) share
/// 32 bytes of quants: the low nibbles of bytes 0..32 hold sub-block 2*s,
/// the high nibbles hold sub-block 2*s+1.
#[inline]
fn dequant_q4_k_block(block: &Q4KBlock, out: &mut [f32]) {
    debug_assert_eq!(out.len(), Q4_K_M_BLOCK_SIZE);

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is unconditional on aarch64; we read 32 bytes per
    // `s` iteration from `block.quants` (size 128, indices 0..127) and
    // write to `out` strictly within `[0, 256)`.
    unsafe {
        use std::arch::aarch64::*;
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

/// Runtime toggle for the SDOT path. Defaults ON for aarch64 builds
/// (target-feature=+dotprod is set crate-wide via .cargo/config.toml).
/// Set `RAYZOR_USE_SDOT=0` to fall back to the F32 SIMD path for A/B
/// comparison.
#[cfg(target_arch = "aarch64")]
fn sdot_enabled() -> bool {
    use std::sync::atomic::{AtomicI8, Ordering};
    static CACHED: AtomicI8 = AtomicI8::new(-1);
    let cur = CACHED.load(Ordering::Relaxed);
    if cur >= 0 {
        return cur == 1;
    }
    let on = std::env::var("RAYZOR_USE_SDOT")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(true); // default ON
    CACHED.store(if on { 1 } else { 0 }, Ordering::Relaxed);
    on
}

// ============================================================================
// INT8 / SDOT inner path for Q4_K_M
//
// The F32 NEON path above hits ~4 cycles per output element — close to the
// peak NEON FMA throughput on M1. The next density jump is `vdotq_s32`
// (ARMv8.2-A SDOT / equivalent AVX-VNNI on x86_64), which does 16 INT8
// multiplies and four 32-bit accumulations in one instruction.
//
// To use it we (1) quantise X to INT8 with one F32 scale per 256-element
// super-block, (2) read the Q4_K_M block's 4-bit weights as INT8 (low/high
// nibbles), (3) SDOT the two INT8 vectors, (4) fold the result back through
// the (super-scale × X scale) and (super-min × Σx) into the final F32 sum.
//
// `quantize_x_block_q8` is called once per chunk call per batch row (cheap
// — typically 16 blocks for k=2048 = a few hundred SIMD ops). `dot_q4_k_q8`
// then does each per-row dot product in pure INT8 arithmetic.
// ============================================================================

/// Per-super-block scratch produced by `quantize_x_block_q8`. One entry
/// per 256-element block of X.
struct Q8Block {
    /// 256 INT8 quants.
    quants: [i8; 256],
    /// F32 quantisation scale (`x[i] ≈ scale * quants[i]`).
    scale: f32,
    /// Σ of `quants[s*32 .. (s+1)*32]` per sub-block (8 sub-blocks).
    /// Used to fold the Q4_K_M min term out of the SDOT result.
    bsums: [i32; 8],
}

/// Quantise a 256-element span of `x` to INT8 with one F32 scale + the
/// 8 per-sub-block sums.
#[inline]
unsafe fn quantize_x_block_q8(x: *const f32) -> Q8Block {
    // Pass 1: find the absolute max over 256 elements (NEON 4×4-lane
    // unrolled). Used to pick a symmetric scale so quants land in
    // [-127, 127].
    let mut max_abs = 0.0f32;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let pa = x;
        let mut m0 = vdupq_n_f32(0.0);
        let mut m1 = vdupq_n_f32(0.0);
        let mut m2 = vdupq_n_f32(0.0);
        let mut m3 = vdupq_n_f32(0.0);
        let abs_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x7FFF_FFFF));
        let mut i = 0;
        while i < 256 {
            let v0 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i))), vreinterpretq_u32_f32(abs_mask));
            let v1 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 4))), vreinterpretq_u32_f32(abs_mask));
            let v2 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 8))), vreinterpretq_u32_f32(abs_mask));
            let v3 = vandq_u32(vreinterpretq_u32_f32(vld1q_f32(pa.add(i + 12))), vreinterpretq_u32_f32(abs_mask));
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
    };

    if max_abs == 0.0 {
        block.scale = 1.0;
        return block;
    }
    block.scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;

    // Pass 2: quantise, accumulate sub-block sums.
    for s in 0..8 {
        let mut sum: i32 = 0;
        for j in 0..32 {
            let v = *x.add(s * 32 + j) * inv_scale;
            // Round-to-nearest, clamp.
            let q = v.round().clamp(-128.0, 127.0) as i8;
            block.quants[s * 32 + j] = q;
            sum += q as i32;
        }
        block.bsums[s] = sum;
    }
    block
}

/// Lazy populator for the X→Q8 cache. Returns a borrowed reference to
/// the slot for block index `b_idx`, quantising the matching 256-element
/// span of X on first access.
#[inline]
unsafe fn x_q8_cache_get<'a>(
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

/// Q4_K_M × Q8 dot product for one 256-element block.
///
/// Computes `Σ_i (q4_dequant[i] * x[i])` for the 256 elements covered
/// by one Q4_K_M block, using SDOT instructions on the 4-bit weights
/// versus the INT8-quantised X. Mirrors llama.cpp's
/// `ggml_vec_dot_q4_K_q8_K` block kernel.
///
/// Math: each sub-block `s` contributes
/// ```
///   sub_scale_s = d * scale6bit_s              (effective f32 scale)
///   sub_min_s   = dmin * min6bit_s             (effective f32 offset)
///   Σ x[i] * (sub_scale_s * q4[i] - sub_min_s)
///     = sub_scale_s * Σ x[i] * q4[i]  -  sub_min_s * Σ x[i]
///     = sub_scale_s * x_scale * sdot(x_q[s..], q4[s..])
///       - sub_min_s * x_scale * bsums[s]
/// ```
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_q4_k_q8(q4_block_ptr: *const u8, x_q8: &Q8Block) -> f32 {
    use std::arch::aarch64::*;

    let d_bits = std::ptr::read_unaligned(q4_block_ptr as *const u16);
    let dmin_bits = std::ptr::read_unaligned(q4_block_ptr.add(2) as *const u16);
    let d = half::f16::from_bits(d_bits).to_f32();
    let dmin = half::f16::from_bits(dmin_bits).to_f32();

    // 12-byte header for the 6-bit scales and mins.
    let mut header = [0u8; 12];
    for (i, slot) in header.iter_mut().enumerate() {
        *slot = *q4_block_ptr.add(4 + i);
    }
    let quants_ptr = q4_block_ptr.add(16);

    let mut sum_term1 = 0.0f32; // Σ sub_scale_s * x_scale * sdot_s
    let mut sum_term2 = 0.0f32; // Σ sub_min_s   * x_scale * bsum_s

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

        // SDOT: each call does Σ_{j in 16-lane group} a[j] * b[j] →
        // accumulated into 4 i32 lanes; we sum across lanes at the end.
        let mut acc_lo = vdupq_n_s32(0);
        acc_lo = vdotq_s32(acc_lo, xlo1, lo1);
        acc_lo = vdotq_s32(acc_lo, xlo2, lo2);

        let mut acc_hi = vdupq_n_s32(0);
        acc_hi = vdotq_s32(acc_hi, xhi1, hi1);
        acc_hi = vdotq_s32(acc_hi, xhi2, hi2);

        let lo_sdot = vaddvq_s32(acc_lo) as f32;
        let hi_sdot = vaddvq_s32(acc_hi) as f32;

        sum_term1 += sub_scale_lo * lo_sdot;
        sum_term1 += sub_scale_hi * hi_sdot;

        sum_term2 += sub_min_lo * x_q8.bsums[2 * p] as f32;
        sum_term2 += sub_min_hi * x_q8.bsums[2 * p + 1] as f32;
    }

    x_q8.scale * (sum_term1 - sum_term2)
}

/// Q4_K_M × f32 matmul. A is `[M, K]` quantised; B is `[K, N]` f32; C is
/// `[M, N]` f32. Dequant happens one block at a time into a small f32
/// stage buffer, then reuses the existing SIMD axpy kernel for the
/// row-update step.
#[allow(clippy::needless_range_loop)]
unsafe fn q4_k_m_matmul_f32(
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
        std::ptr::write_bytes(c_row, 0, n * std::mem::size_of::<f32>());
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
                let c_slice = std::slice::from_raw_parts_mut(c_row, n);
                let b_slice = std::slice::from_raw_parts(b_row, n);
                crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
            }
        }
    }
}

// ============================================================================
// Allocator helpers
// ============================================================================

unsafe fn alloc_qtensor(
    scheme: u8,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> *mut RayzorQTensor {
    let numel = rows * cols;
    let data_bytes = match scheme {
        QSCHEME_INT8 => numel,
        QSCHEME_Q4_K_M => (numel / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES,
        _ => return std::ptr::null_mut(),
    };

    let data = malloc(if data_bytes > 0 { data_bytes } else { 1 });
    if data.is_null() {
        return std::ptr::null_mut();
    }
    std::ptr::write_bytes(data, 0, data_bytes);

    // INT8 needs a per-row (or per-group) scale array. Q4_K_M embeds scales
    // in the data blocks so meta is null.
    let meta: *mut f32 = if scheme == QSCHEME_INT8 {
        let n_groups = numel / group_size;
        let scale_bytes = n_groups * std::mem::size_of::<f32>();
        let s = malloc(scale_bytes.max(4)) as *mut f32;
        if s.is_null() {
            free(data);
            return std::ptr::null_mut();
        }
        s
    } else {
        std::ptr::null_mut()
    };

    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        free(data);
        if !meta.is_null() {
            free(meta as *mut u8);
        }
        return std::ptr::null_mut();
    }

    *qt = RayzorQTensor {
        data,
        meta,
        numel,
        group_size,
        scheme,
        owns_data: true,
        rows,
        cols,
    };

    qt
}

// ============================================================================
// FFI surface
// ============================================================================

/// Create an INT8-quantised 2-D tensor `[rows, cols]` from an f32 source.
/// Each row of `cols` elements gets its own f32 scale. Returns the i64
/// pointer to the opaque `RayzorQTensor`; 0 on failure.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_f32_int8(src_ptr: i64, rows: i64, cols: i64) -> i64 {
    if src_ptr == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let qt_raw = alloc_qtensor(QSCHEME_INT8, rows, cols, cols);
    if qt_raw.is_null() {
        return 0;
    }
    let qt = &*qt_raw;
    let src = src_ptr as *const f32;
    for r in 0..rows {
        let row_src = std::slice::from_raw_parts(src.add(r * cols), cols);
        let row_dst = std::slice::from_raw_parts_mut(qt.data.add(r * cols) as *mut i8, cols);
        let scale = quantise_int8_row(row_src, row_dst);
        *qt.meta.add(r) = scale;
    }
    qt_raw as i64
}

/// Wrap a pre-quantised Q4_K_M byte buffer in a QTensor. The runtime takes
/// ownership of the bytes (i.e. they must come from malloc, OR the caller
/// must keep them alive and the QTensor will simply view them with
/// `owns_data=false`).
///
/// This is the intended GGUF integration point: the loader mmaps the
/// weights file and hands the runtime a raw block pointer + shape.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_wrap_q4_k_m(
    block_data_ptr: i64,
    rows: i64,
    cols: i64,
    take_ownership: i64,
) -> i64 {
    if block_data_ptr == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    if !(rows * cols).is_multiple_of(Q4_K_M_BLOCK_SIZE) {
        return 0;
    }
    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        return 0;
    }
    *qt = RayzorQTensor {
        data: block_data_ptr as *mut u8,
        meta: std::ptr::null_mut(),
        numel: rows * cols,
        group_size: Q4_K_M_BLOCK_SIZE,
        scheme: QSCHEME_Q4_K_M,
        owns_data: take_ownership != 0,
        rows,
        cols,
    };
    qt as i64
}

/// Copy a `haxe.io.Bytes` worth of pre-quantised Q4_K_M data into a fresh
/// owning `QTensor`. The intended caller is the GGUF loader handing the
/// runtime the raw byte slice returned by `GGUFReader.tensorBytes`.
///
/// **Zero-copy by default.** The QTensor points straight into the source
/// `HaxeBytes` buffer with `owns_data=false`. For the dominant use case
/// — mmap-backed GGUF files — the source Bytes lives for the program
/// lifetime so the alias is safe. For non-mmap sources (e.g. a temporary
/// `Bytes.alloc` buffer), callers must keep the source alive at least as
/// long as the QTensor or use `wrap_q4_k_m` with `take_ownership=1` to
/// transfer a malloc'd buffer's ownership in.
///
/// `bytes_handle` is a `*const HaxeBytes` interpreted as i64.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_bytes_q4_k_m(
    bytes_handle: i64,
    rows: i64,
    cols: i64,
) -> i64 {
    if bytes_handle == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let bytes = &*(bytes_handle as *const crate::haxe_sys::HaxeBytes);
    if bytes.ptr.is_null() {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    if !(rows * cols).is_multiple_of(Q4_K_M_BLOCK_SIZE) {
        return 0;
    }
    let expected = (rows * cols / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES;
    if bytes.len < expected {
        return 0;
    }

    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        return 0;
    }
    *qt = RayzorQTensor {
        data: bytes.ptr,
        meta: std::ptr::null_mut(),
        numel: rows * cols,
        group_size: Q4_K_M_BLOCK_SIZE,
        scheme: QSCHEME_Q4_K_M,
        owns_data: false,
        rows,
        cols,
    };
    qt as i64
}

/// Wrap a `HaxeBytes` slice as a Q6_K-backed QTensor. Same zero-copy
/// semantics as `rayzor_qtensor_from_bytes_q4_k_m`. Used for GGUF's
/// dtype 14 (token_embd, attn_v, ffn_down in Q4_K_M variants).
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_bytes_q6_k(
    bytes_handle: i64,
    rows: i64,
    cols: i64,
) -> i64 {
    if bytes_handle == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let bytes = &*(bytes_handle as *const crate::haxe_sys::HaxeBytes);
    if bytes.ptr.is_null() {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    if !(rows * cols).is_multiple_of(Q6_K_BLOCK_SIZE) {
        return 0;
    }
    let expected = (rows * cols / Q6_K_BLOCK_SIZE) * Q6_K_BLOCK_BYTES;
    if bytes.len < expected {
        return 0;
    }

    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        return 0;
    }
    *qt = RayzorQTensor {
        data: bytes.ptr,
        meta: std::ptr::null_mut(),
        numel: rows * cols,
        group_size: Q6_K_BLOCK_SIZE,
        scheme: QSCHEME_Q6_K,
        owns_data: false,
        rows,
        cols,
    };
    qt as i64
}

/// `qt.rows() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_rows(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).rows as i64
}

/// `qt.cols() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_cols(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).cols as i64
}

/// `qt.scheme() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_scheme(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).scheme as i64
}

/// `qt.numel() -> i64`
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_numel(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    (*(qt_ptr as *const RayzorQTensor)).numel as i64
}

/// Dequant the whole tensor into a fresh f32 Tensor (shape [rows, cols]).
/// Useful for debug / accuracy comparison; production code should prefer
/// the fused matmul path.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_dequant(qt_ptr: i64) -> i64 {
    if qt_ptr == 0 {
        return 0;
    }
    let qt = &*(qt_ptr as *const RayzorQTensor);

    // We need a fresh f32 Tensor allocation. Mirror tensor.rs's alloc
    // shape: [rows, cols], F32 dtype, no fill.
    let shape = [qt.rows, qt.cols];
    let out_tensor_ptr =
        crate::tensor::rayzor_tensor_zeros(shape.as_ptr() as i64, 2, 0 /* DTYPE_F32 */);
    if out_tensor_ptr == 0 {
        return 0;
    }
    // Reach into the freshly allocated tensor's data ptr. The tensor.rs
    // layout has `data` as the first field, so dereferencing as a struct
    // with `data: *mut u8` first is safe.
    #[repr(C)]
    struct TensorHead {
        data: *mut u8,
        shape: *mut usize,
        strides: *mut usize,
        ndim: usize,
        numel: usize,
        dtype: u8,
        owns_data: bool,
        device: u8,
        numa_node: i32,
    }
    let head = &*(out_tensor_ptr as *const TensorHead);
    let out = head.data as *mut f32;

    match qt.scheme {
        QSCHEME_INT8 => {
            for r in 0..qt.rows {
                let scale = *qt.meta.add(r);
                let row_src = qt.data.add(r * qt.cols) as *const i8;
                let row_dst = out.add(r * qt.cols);
                for c in 0..qt.cols {
                    *row_dst.add(c) = (*row_src.add(c) as f32) * scale;
                }
            }
        }
        QSCHEME_Q4_K_M => {
            let blocks_per_row = qt.cols / Q4_K_M_BLOCK_SIZE;
            let mut stage = [0.0f32; Q4_K_M_BLOCK_SIZE];
            for r in 0..qt.rows {
                let row_ptr = qt.data.add(r * blocks_per_row * Q4_K_M_BLOCK_BYTES);
                for b in 0..blocks_per_row {
                    let block = decode_q4_k_block(row_ptr.add(b * Q4_K_M_BLOCK_BYTES));
                    dequant_q4_k_block(&block, &mut stage);
                    let dst = out.add(r * qt.cols + b * Q4_K_M_BLOCK_SIZE);
                    std::ptr::copy_nonoverlapping(stage.as_ptr(), dst, Q4_K_M_BLOCK_SIZE);
                }
            }
        }
        QSCHEME_Q6_K => {
            let blocks_per_row = qt.cols / Q6_K_BLOCK_SIZE;
            let mut stage = [0.0f32; Q6_K_BLOCK_SIZE];
            for r in 0..qt.rows {
                let row_ptr = qt.data.add(r * blocks_per_row * Q6_K_BLOCK_BYTES);
                for b in 0..blocks_per_row {
                    dequant_q6_k_block(row_ptr.add(b * Q6_K_BLOCK_BYTES), &mut stage);
                    let dst = out.add(r * qt.cols + b * Q6_K_BLOCK_SIZE);
                    std::ptr::copy_nonoverlapping(stage.as_ptr(), dst, Q6_K_BLOCK_SIZE);
                }
            }
        }
        _ => {}
    }

    out_tensor_ptr
}

/// Fused dequant-matmul: A is quantised `[M, K]`, B is f32 `[K, N]`, out is
/// f32 `[M, N]`. Returns a fresh f32 Tensor; 0 on shape mismatch.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_matmul_f32(qt_a: i64, b_tensor: i64) -> i64 {
    if qt_a == 0 || b_tensor == 0 {
        return 0;
    }
    let qt = &*(qt_a as *const RayzorQTensor);

    // Pull B's shape + data.
    #[repr(C)]
    struct TensorHead {
        data: *mut u8,
        shape: *mut usize,
        strides: *mut usize,
        ndim: usize,
        numel: usize,
        dtype: u8,
        owns_data: bool,
        device: u8,
        numa_node: i32,
    }
    let b_head = &*(b_tensor as *const TensorHead);
    if b_head.ndim != 2 || b_head.dtype != 0
    /* DTYPE_F32 */
    {
        return 0;
    }
    let b_shape = std::slice::from_raw_parts(b_head.shape, 2);
    let k_b = b_shape[0];
    let n = b_shape[1];
    if k_b != qt.cols {
        return 0;
    }

    let out_shape = [qt.rows, n];
    let out_tensor = crate::tensor::rayzor_tensor_zeros(out_shape.as_ptr() as i64, 2, 0);
    if out_tensor == 0 {
        return 0;
    }
    let out_head = &*(out_tensor as *const TensorHead);
    let out_data = out_head.data as *mut f32;

    match qt.scheme {
        QSCHEME_INT8 => {
            int8_matmul_f32(
                qt.data as *const i8,
                qt.meta,
                b_head.data as *const f32,
                out_data,
                qt.rows,
                qt.cols,
                n,
            );
        }
        QSCHEME_Q4_K_M => {
            q4_k_m_matmul_f32(
                qt.data,
                b_head.data as *const f32,
                out_data,
                qt.rows,
                qt.cols,
                n,
            );
        }
        _ => return 0,
    }

    out_tensor
}

/// Compute `Y[B, N] = X[B, K] × Wq[N, K]^T`, with Wq quantised Q4_K_M.
///
/// This is the natural matmul for a PyTorch-style `Linear(in=K, out=N)` whose
/// weight is loaded directly from a GGUF Q4_K_M tensor: Wq has shape
/// `[out, in]` (rows=out, cols=in) with 256-element blocks along the inner
/// `in` (= K) dim — exactly what `GGUFLoader.decodeQ4KM` now produces.
///
/// Computes `y[b, n] = Σ_k x[b, k] * Wq[n, k]` without ever materialising
/// a dequant'd F32 copy of Wq. The kernel dequants each row of Wq once
/// into a small per-thread scratch buffer (K f32s) and then reuses it
/// across the batch — so the dequant cost amortises across B.
///
/// `x_tensor` and `qt_w` are taken as i64 pointers (matches the Haxe FFI).
/// Returns a fresh f32 Tensor on the heap; returns 0 on shape mismatch.
///
/// Single-threaded fallback. The Haxe path threads explicitly by
/// allocating Y first and dispatching `rayzor_tensor_matmul_qt_t_f32_chunk`
/// across workers via `rayzor.concurrent.NumaPool.parallelRows`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qt_t_f32(x_tensor: i64, qt_w: i64) -> i64 {
    if x_tensor == 0 || qt_w == 0 {
        return 0;
    }
    let qt = &*(qt_w as *const RayzorQTensor);

    let (batch, n, k, _block_size, _block_bytes) = match qmatmul_prep(x_tensor, qt) {
        Some(p) => p,
        None => return 0,
    };

    // Allocate Y[batch, N] f32.
    let out_shape = [batch, n];
    let out_tensor = crate::tensor::rayzor_tensor_zeros(out_shape.as_ptr() as i64, 2, 0);
    if out_tensor == 0 {
        return 0;
    }

    // Single-threaded fill of all rows.
    qmatmul_chunk_impl(x_tensor, qt_w, out_tensor, 0, n as i64);
    let _ = k; // K used inside the impl; suppress unused warning here.
    out_tensor
}

/// Threaded variant of `rayzor_tensor_matmul_qt_t_f32`.
///
/// Allocates `Y`, then dispatches `qmatmul_chunk_impl` across N OS
/// threads via `std::thread::scope`. Workers split the output-row
/// range `[0, N)` disjointly so the writes into `Y` need no
/// synchronisation. Joined at the scope boundary; no thread pool
/// outlives the call.
///
/// `threads = 0` picks a default (see implementation). `threads = 1`
/// falls through to the single-threaded path with no spawn overhead.
///
/// This lives next to the chunk entry point so the Haxe-side
/// `NumaPool.parallelRows` route stays available; the threaded entry
/// point exists because (a) importing `NumaPool` from `nue.Linear`
/// currently triggers a JIT cascade we haven't isolated, and
/// (b) for a single matmul-per-Linear-forward the fork-join cost
/// shape is the same either way.
///
/// Returns a fresh F32 tensor; returns 0 on shape mismatch.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qt_t_f32_threaded(
    x_tensor: i64,
    qt_w: i64,
    threads: i64,
) -> i64 {
    if x_tensor == 0 || qt_w == 0 {
        return 0;
    }
    let qt = &*(qt_w as *const RayzorQTensor);

    let (batch, n, _k, _block_size, _block_bytes) = match qmatmul_prep(x_tensor, qt) {
        Some(p) => p,
        None => return 0,
    };

    let out_shape = [batch, n];
    let out_tensor = crate::tensor::rayzor_tensor_zeros(out_shape.as_ptr() as i64, 2, 0);
    if out_tensor == 0 {
        return 0;
    }

    // Pick worker count: explicit > 0, or auto.
    // Auto: 6 workers. Tried 4, 6, 8 on M1 Pro / Llama 3.2 1B Q4_K_M;
    // all sit at ~19 s for a 24-token decode (≈2 effective cores).
    // The fork-join is invoked ~112 times per generated token (16
    // layers × 7 Linear projections), so spawn cost + memory-bandwidth
    // contention dominate; throwing more threads at it doesn't move
    // the wall time. Real per-core win lives in SIMD-tiled inner dot
    // (queued); P-core QoS hint also tested, ~0 effect on M1.
    let auto_threads: usize = 6;
    let mut t = if threads > 0 {
        (threads as usize).min(64)
    } else {
        auto_threads
    };
    if t > n {
        t = n.max(1);
    }
    if t <= 1 {
        qmatmul_chunk_impl(x_tensor, qt_w, out_tensor, 0, n as i64);
        return out_tensor;
    }

    // Bundle the raw i64 handles into `Send + Copy` wrappers so we can
    // capture them across `std::thread::scope`. The pointer-aliased
    // memory is read-only for X/Wq and disjoint per-worker for Y.
    let xh = x_tensor;
    let qh = qt_w;
    let yh = out_tensor;

    // Bias the spawning thread toward the performance cores. On macOS
    // the QoS class is inherited by threads spawned via `std::thread`,
    // so a single call here propagates to every worker without a
    // per-spawn syscall.
    bias_to_performance_core();

    let chunk = n.div_ceil(t);
    std::thread::scope(|s| {
        for w in 0..t {
            let lo = w * chunk;
            if lo >= n {
                break;
            }
            let hi = (lo + chunk).min(n);
            s.spawn(move || {
                // SAFETY: each worker writes Y[*, lo..hi); ranges are
                // disjoint, no aliasing across threads. X and Wq are
                // read-only.
                unsafe {
                    qmatmul_chunk_impl(xh, qh, yh, lo as i64, hi as i64);
                }
            });
        }
    });
    out_tensor
}

/// Bias the calling thread toward the performance cores on macOS.
///
/// Without this hint macOS's scheduler is free to land workers on the
/// E-cores, which are 3-4× slower than the P-cores on M1/M2 — and the
/// fork-join's wall time is `max` across workers, so even one E-core
/// straggler caps the speedup we can get. `QOS_CLASS_USER_INITIATED`
/// (the level used by foreground work in user-facing apps) tells the
/// scheduler we want this on a P-core unless the system is saturated.
///
/// Best-effort: failures are ignored. Other platforms get no-ops.
#[inline]
fn bias_to_performance_core() {
    #[cfg(target_os = "macos")]
    {
        // QOS_CLASS_USER_INITIATED — declared inline because libc
        // 0.2.186 doesn't re-export `QOS_CLASS_*` from the apple
        // module at the crate root.
        const QOS_CLASS_USER_INITIATED: std::ffi::c_uint = 0x19;
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(
                qos_class: std::ffi::c_uint,
                relative_priority: std::ffi::c_int,
            ) -> std::ffi::c_int;
        }
        // SAFETY: libSystem entry; no preconditions; result ignored.
        unsafe {
            let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        }
    }
}

/// Compute output rows `[n_start, n_end)` of `Y = X @ Wq.T` and store
/// them into a pre-allocated Y tensor.
///
/// This is the threading entry point. The caller (Haxe `NumaPool.parallelRows`)
/// allocates Y once, then fans out non-overlapping `[n_start, n_end)` ranges
/// to workers. Each call:
///   - dequants its own slice of `Wq` rows into a thread-local scratch,
///   - dots X against each dequanted row,
///   - stores into `y[b, n_start..n_end)`.
///
/// Memory safety: workers write to disjoint columns of Y, so no
/// synchronisation is needed beyond the standard fork-join barrier the
/// Haxe pool provides. Returns 1 on success, 0 on shape mismatch / null.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qt_t_f32_chunk(
    x_tensor: i64,
    qt_w: i64,
    y_tensor: i64,
    n_start: i64,
    n_end: i64,
) -> i64 {
    if x_tensor == 0 || qt_w == 0 || y_tensor == 0 {
        return 0;
    }
    let qt = &*(qt_w as *const RayzorQTensor);
    if qmatmul_prep(x_tensor, qt).is_none() {
        return 0;
    }
    qmatmul_chunk_impl(x_tensor, qt_w, y_tensor, n_start, n_end);
    1
}

/// Shape-validate `X` and a `Wq` QTensor for `Y = X @ Wq.T`. Returns
/// `(batch, n, k, block_size, block_bytes)` on success.
unsafe fn qmatmul_prep(
    x_tensor: i64,
    qt: &RayzorQTensor,
) -> Option<(usize, usize, usize, usize, usize)> {
    let (block_size, block_bytes) = match qt.scheme {
        QSCHEME_Q4_K_M => (Q4_K_M_BLOCK_SIZE, Q4_K_M_BLOCK_BYTES),
        QSCHEME_Q6_K => (Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES),
        _ => return None,
    };

    #[repr(C)]
    struct TensorHead {
        data: *mut u8,
        shape: *mut usize,
        strides: *mut usize,
        ndim: usize,
        numel: usize,
        dtype: u8,
        owns_data: bool,
        device: u8,
        numa_node: i32,
    }
    let x_head = &*(x_tensor as *const TensorHead);
    if x_head.ndim != 2 || x_head.dtype != 0
    /* DTYPE_F32 */
    {
        return None;
    }
    let x_shape = std::slice::from_raw_parts(x_head.shape, 2);
    let batch = x_shape[0];
    let k = x_shape[1];

    if k != qt.cols || !k.is_multiple_of(block_size) {
        return None;
    }
    Some((batch, qt.rows, k, block_size, block_bytes))
}

/// Inner kernel for both the single-threaded fallback and the
/// threaded-chunk entry point. Computes `y[b, n_idx] = X[b, :] · dequant(Wq[n_idx, :])`
/// for `n_idx in [n_start, n_end)`. Worker buffers live on the stack/
/// thread-local heap; cross-thread state is just the `*y` write band,
/// which workers split disjointly so this needs no synchronisation.
unsafe fn qmatmul_chunk_impl(
    x_tensor: i64,
    qt_w: i64,
    y_tensor: i64,
    n_start: i64,
    n_end: i64,
) {
    let qt = &*(qt_w as *const RayzorQTensor);
    let (block_size, block_bytes) = match qt.scheme {
        QSCHEME_Q4_K_M => (Q4_K_M_BLOCK_SIZE, Q4_K_M_BLOCK_BYTES),
        QSCHEME_Q6_K => (Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES),
        _ => return,
    };

    #[repr(C)]
    struct TensorHead {
        data: *mut u8,
        shape: *mut usize,
        strides: *mut usize,
        ndim: usize,
        numel: usize,
        dtype: u8,
        owns_data: bool,
        device: u8,
        numa_node: i32,
    }
    let x_head = &*(x_tensor as *const TensorHead);
    let y_head = &*(y_tensor as *const TensorHead);
    let x_shape = std::slice::from_raw_parts(x_head.shape, 2);
    let x_strides = std::slice::from_raw_parts(x_head.strides, 2);
    let batch = x_shape[0];
    let k = x_shape[1];
    let n = qt.rows;
    let blocks_per_row = k / block_size;

    let lo = (n_start.max(0) as usize).min(n);
    let hi = (n_end.max(0) as usize).min(n);
    if lo >= hi {
        return;
    }

    let y_data = y_head.data as *mut f32;
    let x_data = x_head.data as *const f32;
    let x_contig = x_strides[1] == 1;

    // Stage buffer for one dequanted block — 256 floats stays hot in
    // L1 across the (dequant, dot) pair, so we never write a
    // full-row scratch. Each `n_idx` iteration walks blocks_per_row
    // (8 for k=2048) blocks; each block is decoded, dotted against
    // the matching X chunk, sum accumulated, then discarded.
    let mut stage = [0.0f32; 256]; // Q4_K_M_BLOCK_SIZE == Q6_K_BLOCK_SIZE == 256

    // Per-row sums. Sized to `batch` so the general path can write
    // into it for any value of `batch`; the `batch == 1 && x_contig`
    // fast path below skips this entirely.
    let mut row_sums: Vec<f32> = vec![0.0; batch.max(1)];

    // Lazy cache of `quantize_x_block_q8` results for the SDOT path.
    // Each chunk call reuses the same X across every `n_idx` in
    // `[lo, hi)`, so quantising it once amortises across the whole
    // chunk. Populated on the first use of each block index.
    let mut x_q8_cache: Vec<Q8Block> = Vec::new();
    let mut x_q8_init: Vec<bool> = Vec::new();
    if cfg!(target_arch = "aarch64") && batch == 1 && x_contig && qt.scheme == QSCHEME_Q4_K_M {
        x_q8_cache.reserve_exact(blocks_per_row);
        for _ in 0..blocks_per_row {
            x_q8_cache.push(Q8Block {
                quants: [0i8; 256],
                scale: 0.0,
                bsums: [0i32; 8],
            });
        }
        x_q8_init = vec![false; blocks_per_row];
    }

    for n_idx in lo..hi {
        let row_ptr = qt.data.add(n_idx * blocks_per_row * block_bytes);

        if batch == 1 && x_contig {
            // Decode fast path: single batch row, contiguous X. Q4_K_M
            // can route through the SDOT kernel when `RAYZOR_USE_SDOT=1`
            // is set; the F32 path remains the default until A/B
            // measurement shows SDOT wins on this hardware.
            let mut sum = 0.0f32;
            #[cfg(target_arch = "aarch64")]
            let use_sdot = sdot_enabled();
            for b_idx in 0..blocks_per_row {
                let bp = row_ptr.add(b_idx * block_bytes);
                match qt.scheme {
                    QSCHEME_Q4_K_M => {
                        #[cfg(target_arch = "aarch64")]
                        if use_sdot {
                            let x_q8 = x_q8_cache_get(
                                &mut x_q8_cache,
                                &mut x_q8_init,
                                b_idx,
                                x_data.add(b_idx * block_size),
                            );
                            sum += dot_q4_k_q8(bp, x_q8);
                            continue;
                        }
                        let block = decode_q4_k_block(bp);
                        dequant_q4_k_block(&block, &mut stage);
                        let x_chunk = std::slice::from_raw_parts(
                            x_data.add(b_idx * block_size),
                            block_size,
                        );
                        sum += dot_f32_simd(x_chunk, &stage);
                    }
                    QSCHEME_Q6_K => {
                        dequant_q6_k_block(bp, &mut stage);
                        let x_chunk = std::slice::from_raw_parts(
                            x_data.add(b_idx * block_size),
                            block_size,
                        );
                        sum += dot_f32_simd(x_chunk, &stage);
                    }
                    _ => unreachable!(),
                }
            }
            *y_data.add(n_idx) = sum;
            continue;
        }

        // General path: batch > 1 or non-contiguous X. Accumulate one
        // running sum per batch row; flush at the end of the row.
        row_sums.iter_mut().for_each(|s| *s = 0.0);

        for b_idx in 0..blocks_per_row {
            let bp = row_ptr.add(b_idx * block_bytes);
            match qt.scheme {
                QSCHEME_Q4_K_M => {
                    let block = decode_q4_k_block(bp);
                    dequant_q4_k_block(&block, &mut stage);
                }
                QSCHEME_Q6_K => {
                    dequant_q6_k_block(bp, &mut stage);
                }
                _ => unreachable!(),
            }

            for b in 0..batch {
                let x_off = b * x_strides[0] + b_idx * block_size;
                if x_contig {
                    let x_chunk =
                        std::slice::from_raw_parts(x_data.add(x_off), block_size);
                    row_sums[b] += dot_f32_simd(x_chunk, &stage);
                } else {
                    let stride = x_strides[1];
                    let mut partial = 0.0f32;
                    let x_base = b * x_strides[0] + b_idx * block_size * stride;
                    for p in 0..block_size {
                        partial += *x_data.add(x_base + p * stride) * stage[p];
                    }
                    row_sums[b] += partial;
                }
            }
        }

        for b in 0..batch {
            *y_data.add(b * n + n_idx) = row_sums[b];
        }
    }
}

/// Vectorised `Σ a[i] * b[i]` over the common length of two F32 slices.
///
/// On aarch64 (Apple silicon, ARM servers) this uses 4×F32 NEON
/// fused multiply-add (`vfmaq_f32`) with 4-vector unrolling — 16
/// elements per inner iteration. On x86_64 (AVX2 / FMA when present)
/// it uses the equivalent 8×F32 path. Otherwise it falls through to
/// a scalar accumulator that LLVM will still autovectorise where it
/// can.
///
/// This is the per-core hot loop for Q4_K_M / Q6_K Linear projections
/// — `k = 2048` for the QKV / O projections, `k = 8192` for the FFN
/// gate/up/down. On M1 Pro the SIMD path turns the ~4 ms per dot
/// product into ~1 ms, which is the dominant CPU cost during decode.
#[inline]
fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on every aarch64 target; we
        // index strictly within `[0, n)` for both slices and bound-
        // check the unrolled tail.
        unsafe {
            use std::arch::aarch64::*;
            let pa = a.as_ptr();
            let pb = b.as_ptr();
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);
            let main = n & !15; // round down to multiple of 16
            let mut i = 0;
            while i < main {
                let va0 = vld1q_f32(pa.add(i));
                let vb0 = vld1q_f32(pb.add(i));
                let va1 = vld1q_f32(pa.add(i + 4));
                let vb1 = vld1q_f32(pb.add(i + 4));
                let va2 = vld1q_f32(pa.add(i + 8));
                let vb2 = vld1q_f32(pb.add(i + 8));
                let va3 = vld1q_f32(pa.add(i + 12));
                let vb3 = vld1q_f32(pb.add(i + 12));
                acc0 = vfmaq_f32(acc0, va0, vb0);
                acc1 = vfmaq_f32(acc1, va1, vb1);
                acc2 = vfmaq_f32(acc2, va2, vb2);
                acc3 = vfmaq_f32(acc3, va3, vb3);
                i += 16;
            }
            // 4-element trailing groups.
            let quad = n & !3;
            while i < quad {
                let va = vld1q_f32(pa.add(i));
                let vb = vld1q_f32(pb.add(i));
                acc0 = vfmaq_f32(acc0, va, vb);
                i += 4;
            }
            let sum_vec = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let mut sum = vaddvq_f32(sum_vec);
            // Scalar tail (n % 4 elements).
            while i < n {
                sum += *pa.add(i) * *pb.add(i);
                i += 1;
            }
            return sum;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            // SAFETY: feature-detect gate above guarantees AVX/FMA.
            return unsafe { dot_f32_avx2_fma(a, b, n) };
        }
    }

    #[allow(unreachable_code)]
    {
        // Scalar fallback. LLVM autovectorises well-aligned slices
        // here, so this is the path on wasm and pre-AVX2 x86.
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32], n: usize) -> f32 {
    use std::arch::x86_64::*;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let main = n & !31; // 4×AVX-256 lanes = 32 elements
    let mut i = 0;
    while i < main {
        let va0 = _mm256_loadu_ps(pa.add(i));
        let vb0 = _mm256_loadu_ps(pb.add(i));
        let va1 = _mm256_loadu_ps(pa.add(i + 8));
        let vb1 = _mm256_loadu_ps(pb.add(i + 8));
        let va2 = _mm256_loadu_ps(pa.add(i + 16));
        let vb2 = _mm256_loadu_ps(pb.add(i + 16));
        let va3 = _mm256_loadu_ps(pa.add(i + 24));
        let vb3 = _mm256_loadu_ps(pb.add(i + 24));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        i += 32;
    }
    let octo = n & !7;
    while i < octo {
        let va = _mm256_loadu_ps(pa.add(i));
        let vb = _mm256_loadu_ps(pb.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    let sum_vec = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    // Horizontal sum: hadd doesn't do across lanes, do it manually.
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum_vec);
    let mut sum = tmp.iter().sum::<f32>();
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

/// Release a QTensor. The runtime frees `data` and `meta` if `owns_data`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_free(qt_ptr: i64) {
    if qt_ptr == 0 {
        return;
    }
    let qt = &*(qt_ptr as *const RayzorQTensor);
    if qt.owns_data && !qt.data.is_null() {
        free(qt.data);
    }
    if !qt.meta.is_null() {
        free(qt.meta as *mut u8);
    }
    free(qt_ptr as *mut u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_quant_round_trip_close() {
        // Quantise a small row, then dequant, and check max abs error is
        // bounded by the per-row scale (1 ulp of int8).
        let row = [0.1f32, -0.5, 1.0, -1.0, 0.5, -0.1, 0.0, 2.0];
        let mut q = [0i8; 8];
        let scale = quantise_int8_row(&row, &mut q);
        for (i, &qv) in q.iter().enumerate() {
            let reconstructed = scale * qv as f32;
            assert!(
                (reconstructed - row[i]).abs() <= scale,
                "i={i} row={} got={} scale={}",
                row[i],
                reconstructed,
                scale
            );
        }
    }

    #[test]
    fn int8_matmul_close_to_f32() {
        // Build a small 2x4 × 4x3 matmul reference, quant A to int8, verify
        // the dequant matmul produces results within 5% relative error.
        let a_f32: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let b_f32: [f32; 12] = [
            0.5, 1.0, -0.5, 0.5, -1.0, 1.5, 1.0, 0.5, -1.0, -0.5, 1.0, 0.5,
        ];

        // Reference: A × B (2x4 × 4x3 → 2x3) in f32.
        let mut ref_c = [0.0f32; 6];
        for i in 0..2 {
            for j in 0..3 {
                let mut s = 0.0f32;
                for p in 0..4 {
                    s += a_f32[i * 4 + p] * b_f32[p * 3 + j];
                }
                ref_c[i * 3 + j] = s;
            }
        }

        // Quant A.
        unsafe {
            let qt = rayzor_qtensor_from_f32_int8(a_f32.as_ptr() as i64, 2, 4);
            assert!(qt != 0);

            // Allocate B as a Tensor and matmul.
            let b_shape = [4usize, 3];
            let b_tensor = crate::tensor::rayzor_tensor_zeros(b_shape.as_ptr() as i64, 2, 0);
            assert!(b_tensor != 0);
            // Copy b_f32 into the tensor's data.
            #[repr(C)]
            struct TensorHead {
                data: *mut u8,
                _shape: *mut usize,
                _strides: *mut usize,
                _ndim: usize,
                _numel: usize,
                _dtype: u8,
                _owns_data: bool,
                _device: u8,
                _numa_node: i32,
            }
            let b_head = &*(b_tensor as *const TensorHead);
            std::ptr::copy_nonoverlapping(b_f32.as_ptr(), b_head.data as *mut f32, 12);

            let out_tensor = rayzor_qtensor_matmul_f32(qt, b_tensor);
            assert!(out_tensor != 0);
            let out_head = &*(out_tensor as *const TensorHead);
            let out = std::slice::from_raw_parts(out_head.data as *const f32, 6);

            for i in 0..6 {
                let err = (out[i] - ref_c[i]).abs();
                let rel = if ref_c[i].abs() > 1e-6 {
                    err / ref_c[i].abs()
                } else {
                    err
                };
                assert!(
                    rel < 0.05,
                    "int8 matmul[{i}] = {} ref = {} rel_err = {}",
                    out[i],
                    ref_c[i],
                    rel
                );
            }

            rayzor_qtensor_free(qt);
        }
    }

    #[test]
    fn q6_k_block_dequant_matches_reference() {
        // Build a Q6_K block with known d, scales, and quants.
        // Layout: ql[128] | qh[64] | scales[16, i8] | d[f16, 2 bytes] = 210 bytes.
        //
        // We pick a single non-zero per-half slot and verify the produced f32
        // matches the spec formula: f = d * scale * (q - 32) where q is the
        // 6-bit value reconstructed from ql + qh.
        let mut block = [0u8; Q6_K_BLOCK_BYTES];

        // Quant slot (n=0, l=0, position 0 within the half).
        // We want q1 = (ql[0] & 0xF) | ((qh[0] & 3) << 4) = some known 6-bit value.
        // Set ql[0] = 0x0A (low nibble = 10), qh[0] bits 0..1 = 0b10 (=2).
        // → q1 = 10 | (2 << 4) = 10 | 32 = 42. After -32 bias: 10.
        block[0] = 0x0A;
        block[128] = 0b00000010;

        // scales[0] = 3 (i8). d = 0.25 (f16).
        block[192] = 3;
        let d_bits = f16::from_f32(0.25).to_bits();
        block[208..210].copy_from_slice(&d_bits.to_le_bytes());

        let mut out = [0.0f32; Q6_K_BLOCK_SIZE];
        unsafe {
            dequant_q6_k_block(block.as_ptr(), &mut out);
        }

        // Expected: out[0] = d * scales[0] * (q1 - 32) = 0.25 * 3 * 10 = 7.5.
        assert!(
            (out[0] - 7.5).abs() < 1e-3,
            "out[0] = {} (expected 7.5)",
            out[0]
        );
        // All other slots should be d * scale * (-32) = 0.25 * 3 * (-32) = -24
        // for the first 16 positions (which share scales[0]), then change.
        for i in 1..16 {
            assert!((out[i] - (-24.0)).abs() < 1e-3, "out[{i}] = {}", out[i]);
        }
    }

    #[test]
    fn q4_k_m_block_round_trip() {
        // Synthesise a known Q4_K_M block and verify the dequant produces
        // the expected values.
        // Block: d=2.0, dmin=1.0, sub-block 0 (scale=1, min=2), all-zeros
        // quants → expected output: q*sc - mn = 0*2*1/63 - 1*2/63 = -2/63
        // for elements 0..32.
        //
        // Use raw bytes: d=2.0 in f16 = 0x4000, dmin=1.0 in f16 = 0x3C00.
        let mut block = [0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(2.0).to_bits();
        let dmin = f16::from_f32(1.0).to_bits();
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        // Header: sub-block 0 scale=1, min=1. Bytes 4 and 8.
        block[4] = 1; // scales[0..4] low 6 bits → scale[0] = 1
        block[8] = 1; // mins[0..4] low 6 bits via byte[4..8] header bit-packed; see q4_k_get_scale_min

        // Decode and verify.
        unsafe {
            let decoded = decode_q4_k_block(block.as_ptr());
            // Per the spec: scales[0] = d * 1 = 2.0, mins[0] = dmin * 1 = 1.0
            // (using the j<4 branch: sc = header[0] & 63 = 1, mn = header[4] & 63 = 1).
            assert!((decoded.d - 2.0).abs() < 1e-3);
            assert!((decoded.dmin - 1.0).abs() < 1e-3);
            assert!((decoded.scales[0] - 2.0).abs() < 1e-3);
            assert!((decoded.mins[0] - 1.0).abs() < 1e-3);

            let mut out = [0.0f32; Q4_K_M_BLOCK_SIZE];
            dequant_q4_k_block(&decoded, &mut out);
            // sub-block 0 elements 0..32: q=0, sc=2, mn=1 → 0*2 - 1 = -1.
            for i in 0..32 {
                assert!((out[i] - (-1.0)).abs() < 1e-3, "out[{i}] = {}", out[i]);
            }
        }
    }

    #[test]
    fn from_bytes_q4_k_m_copies_and_wraps() {
        // Build a single-block Q4_K_M buffer in a HaxeBytes-shaped struct,
        // pass through the FFI, verify the resulting QTensor wraps an
        // owning copy with the correct shape.
        let mut block = vec![0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(1.0).to_bits();
        let dmin = f16::from_f32(0.0).to_bits();
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());

        let bytes = crate::haxe_sys::HaxeBytes::new_malloc(
            block.as_mut_ptr(),
            block.len(),
            block.capacity(),
        );
        let bytes_handle = &bytes as *const _ as i64;
        let qt = unsafe { rayzor_qtensor_from_bytes_q4_k_m(bytes_handle, 1, 256) };
        assert!(qt != 0);
        unsafe {
            let qt_ref = &*(qt as *const RayzorQTensor);
            assert_eq!(qt_ref.rows, 1);
            assert_eq!(qt_ref.cols, 256);
            assert_eq!(qt_ref.scheme, QSCHEME_Q4_K_M);
            // Zero-copy: QTensor aliases the source buffer, doesn't own it.
            assert!(!qt_ref.owns_data);
            assert_eq!(qt_ref.data, block.as_mut_ptr());
            assert_eq!(*qt_ref.data, d.to_le_bytes()[0]);
        }
        unsafe { rayzor_qtensor_free(qt) };
    }

    #[test]
    fn from_bytes_q4_k_m_rejects_bad_input() {
        // Misaligned (rows * cols not multiple of 256) → returns 0.
        let bytes = crate::haxe_sys::HaxeBytes::new_malloc(std::ptr::null_mut(), 0, 0);
        let handle = &bytes as *const _ as i64;
        assert_eq!(
            unsafe { rayzor_qtensor_from_bytes_q4_k_m(handle, 1, 100) },
            0
        );
        assert_eq!(unsafe { rayzor_qtensor_from_bytes_q4_k_m(0, 1, 256) }, 0);
    }

    /// Build a single Q4_K_M block whose dequant yields the constant `value`
    /// for every element. Sets d=value, every sub-block scale=1 / min=0,
    /// quants[..]=0x11 (q_lo=q_hi=1) → out = scale*q - min = value*1 - 0 = value.
    ///
    /// Header layout per `q4_k_get_scale_min`:
    ///   - sub-blocks 0..3: scale = block[4..8] & 63, min = block[8..12] & 63
    ///   - sub-blocks 4..7: scale = (block[12..16] & 0x0F) | ((block[4..8] >> 6) << 4),
    ///                      min   = (block[12..16] >> 4)   | ((block[8..12] >> 6) << 4)
    fn build_constant_block(value: f32) -> [u8; Q4_K_M_BLOCK_BYTES] {
        let mut block = [0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(value).to_bits();
        let dmin = f16::from_f32(0.0).to_bits();
        block[0..2].copy_from_slice(&d.to_le_bytes());
        block[2..4].copy_from_slice(&dmin.to_le_bytes());
        // Sub-blocks 0..3: low 6 bits of block[4..8] = scale = 1; block[8..12] = min = 0.
        for j in 0..4 {
            block[4 + j] = 1;
            block[8 + j] = 0;
        }
        // Sub-blocks 4..7: scale's low nibble in block[12..16] = 1, upper 2 bits from
        // block[4..8] high bits (already 0 since we wrote 1). Mins' low nibble (upper
        // half of block[12..16]) = 0. So block[12..16] = 0x01.
        for j in 0..4 {
            block[12 + j] = 0x01;
        }
        // Quants: 128 bytes, each holds two nibbles. q_lo=q_hi=1 → byte = 0x11.
        for i in 16..16 + 128 {
            block[i] = 0x11;
        }
        block
    }

    #[test]
    fn dequant_preserves_block_order_in_linear_memory() {
        // Critical test for GGUF Q4_K_M correctness: build 4 blocks with
        // distinct constant values (1.0, 2.0, 3.0, 4.0) and verify the
        // dequant output places them contiguously in linear memory in the
        // order they appear in the source buffer — independent of the
        // (rows, cols) shape interpretation. This is the invariant that
        // GGUFLoader's `decodeQ4KM` (which does rows=in, cols=out) relies
        // on for correctness.
        let blocks: Vec<u8> = (1..=4)
            .flat_map(|v| build_constant_block(v as f32).to_vec())
            .collect();
        assert_eq!(blocks.len(), 4 * Q4_K_M_BLOCK_BYTES);
        let total_elems = 4 * Q4_K_M_BLOCK_SIZE;

        // Try the interpretation GGUFLoader actually uses for a tensor whose
        // GGUF dims=[in=4, out=256] (i.e., 4 rows of 256-elem blocks each).
        let mut src = blocks.clone();
        let bytes =
            crate::haxe_sys::HaxeBytes::new_malloc(src.as_mut_ptr(), src.len(), src.capacity());
        let handle = &bytes as *const _ as i64;

        unsafe {
            // rows=4, cols=256 → 4 rows × 1 block per row = 4 blocks.
            // Linear output memory: [block0 (256), block1 (256), block2, block3].
            let qt = rayzor_qtensor_from_bytes_q4_k_m(handle, 4, 256);
            assert!(qt != 0);
            let dq = rayzor_qtensor_dequant(qt);
            assert!(dq != 0);

            #[repr(C)]
            struct TensorHead {
                data: *mut u8,
                _shape: *mut usize,
                _strides: *mut usize,
                _ndim: usize,
                _numel: usize,
                _dtype: u8,
            }
            let head = &*(dq as *const TensorHead);
            let out = std::slice::from_raw_parts(head.data as *const f32, total_elems);

            // Block 0 (value 1.0) in positions 0..256.
            for i in 0..256 {
                assert!(
                    (out[i] - 1.0).abs() < 1e-3,
                    "block0[{i}] = {} (expected 1.0)",
                    out[i]
                );
            }
            // Block 1 (value 2.0) in positions 256..512.
            for i in 0..256 {
                assert!(
                    (out[256 + i] - 2.0).abs() < 1e-3,
                    "block1[{i}] = {} (expected 2.0)",
                    out[256 + i]
                );
            }
            // Block 2 (value 3.0) in positions 512..768.
            for i in 0..256 {
                assert!(
                    (out[512 + i] - 3.0).abs() < 1e-3,
                    "block2[{i}] = {} (expected 3.0)",
                    out[512 + i]
                );
            }
            // Block 3 (value 4.0) in positions 768..1024.
            for i in 0..256 {
                assert!(
                    (out[768 + i] - 4.0).abs() < 1e-3,
                    "block3[{i}] = {} (expected 4.0)",
                    out[768 + i]
                );
            }

            rayzor_qtensor_free(qt);
        }

        // Same bytes, alternate shape (1, 1024). Output memory MUST be
        // identical — only the logical shape label changes.
        let mut src2 = blocks.clone();
        let bytes2 =
            crate::haxe_sys::HaxeBytes::new_malloc(src2.as_mut_ptr(), src2.len(), src2.capacity());
        let handle2 = &bytes2 as *const _ as i64;
        unsafe {
            let qt = rayzor_qtensor_from_bytes_q4_k_m(handle2, 1, 1024);
            assert!(qt != 0);
            let dq = rayzor_qtensor_dequant(qt);
            assert!(dq != 0);

            #[repr(C)]
            struct TensorHead {
                data: *mut u8,
                _shape: *mut usize,
                _strides: *mut usize,
                _ndim: usize,
                _numel: usize,
                _dtype: u8,
            }
            let head = &*(dq as *const TensorHead);
            let out = std::slice::from_raw_parts(head.data as *const f32, total_elems);
            // Same as above — linear memory has block0..block3 contiguously.
            for (b, expected) in (1..=4).enumerate() {
                for i in 0..256 {
                    assert!(
                        (out[b * 256 + i] - expected as f32).abs() < 1e-3,
                        "alt-shape block{b}[{i}] = {} (expected {})",
                        out[b * 256 + i],
                        expected
                    );
                }
            }
            rayzor_qtensor_free(qt);
        }
    }

    /// Sanity-check Linear-style matmul against a Q4_K_M weight constructed
    /// to mimic a PyTorch-style `[out=2, in=512]` matrix where:
    ///   row 0 (output 0) = all 1.0
    ///   row 1 (output 1) = all 2.0
    ///
    /// In a GGUF file, this matrix is stored row-major as PyTorch [out, in]
    /// (out outermost = slowest = physical rows; in innermost = fastest with
    /// 256-element Q4_K_M blocks). So the file's 4 blocks appear in order:
    ///   [block(1.0), block(1.0), block(2.0), block(2.0)]
    /// (output 0's two in-blocks, then output 1's two in-blocks).
    ///
    /// The current `GGUFReader.decodeQ4KM` would interpret dims=[in=512, out=2]
    /// with rows=in=512, cols=out=2. But cols=2 < 256 is INVALID for Q4_K_M,
    /// so the existing path can't even represent this small case. Use a
    /// 256×512 (or 512×512) shape for the realistic test below.
    ///
    /// This test EXPOSES the layout/shape mismatch when the weight is
    /// interpreted under different (rows, cols) conventions — confirming
    /// whether Linear-style `x @ dq` produces sensible outputs.
    #[test]
    fn dequant_shape_matches_pytorch_weight_layout() {
        // PyTorch weight w[out=256, in=512]:
        //   w[o, *] = (o + 1) as f32, for o in 0..256
        // File layout (PyTorch row-major): blocks along in (= innermost).
        //   - 256 rows × 2 blocks per row = 512 blocks total
        //   - Block order: row 0 b0 (=1), row 0 b1 (=1), row 1 b0 (=2), row 1 b1 (=2), ..., row 255 b0 (=256), row 255 b1 (=256)
        let mut blocks: Vec<u8> = Vec::with_capacity(512 * Q4_K_M_BLOCK_BYTES);
        for o in 0..256 {
            let v = (o + 1) as f32;
            // Output row o has 2 blocks of value v.
            blocks.extend_from_slice(&build_constant_block(v));
            blocks.extend_from_slice(&build_constant_block(v));
        }
        assert_eq!(blocks.len(), 512 * Q4_K_M_BLOCK_BYTES);

        // GGUF dims would be [in=512, out=256]. Existing decodeQ4KM does:
        //   rows = product of all-but-last = 512 = in
        //   cols = dims[last] = 256 = out
        // We mirror that here:
        let mut src = blocks.clone();
        let bytes =
            crate::haxe_sys::HaxeBytes::new_malloc(src.as_mut_ptr(), src.len(), src.capacity());
        let handle = &bytes as *const _ as i64;

        unsafe {
            // PROPOSED FIX convention: rows=out=256 (= dims[last]),
            //                          cols=in=512 (= product of all but last)
            // This matches the file's physical layout: 256 PyTorch-output
            // rows × 512 in-elements each.
            let qt = rayzor_qtensor_from_bytes_q4_k_m(handle, 256, 512);
            assert!(qt != 0);
            let dq = rayzor_qtensor_dequant(qt);
            assert!(dq != 0);

            #[repr(C)]
            struct TensorHead {
                data: *mut u8,
                _shape: *mut usize,
                _strides: *mut usize,
                _ndim: usize,
                _numel: usize,
                _dtype: u8,
            }
            let head = &*(dq as *const TensorHead);
            let dq_slice = std::slice::from_raw_parts(head.data as *const f32, 256 * 512);

            // With rows=out=256, cols=in=512: dq is PyTorch_w directly:
            // dq[o, i] = (o + 1) for all o in 0..256, i in 0..512.
            // dq_slice[o * 512 + i] must equal (o + 1).
            let mut mismatches = 0;
            let mut sample_wrong: Option<(usize, usize, f32, f32)> = None;
            for o in 0..256 {
                for i in 0..512 {
                    let expected = (o + 1) as f32;
                    let actual = dq_slice[o * 512 + i];
                    if (actual - expected).abs() > 1e-3 {
                        mismatches += 1;
                        if sample_wrong.is_none() {
                            sample_wrong = Some((o, i, actual, expected));
                        }
                    }
                }
            }
            if mismatches > 0 {
                let (o, i, a, e) = sample_wrong.unwrap();
                panic!(
                    "decodeQ4KM (proposed-fix convention rows=out, cols=in) STILL \
                     doesn't match PyTorch_w layout.\n  \
                     Mismatches: {} / {}.\n  \
                     Sample: dq[o={}, i={}] = {} (expected {}).",
                    mismatches,
                    256 * 512,
                    o,
                    i,
                    a,
                    e
                );
            }
            rayzor_qtensor_free(qt);
        }
    }

    /// Phase 4b correctness: `rayzor_tensor_matmul_qt_t_f32` must produce
    /// the same output as dequant'ing Wq to F32 and running a regular
    /// `y = x @ w.T` matmul.
    #[test]
    fn matmul_qt_t_f32_matches_dequant_then_matmul_t() {
        // Build a 256×512 Q4_K_M weight where output row o = constant (o+1).
        let mut blocks: Vec<u8> = Vec::with_capacity(512 * Q4_K_M_BLOCK_BYTES);
        for o in 0..256 {
            let v = (o + 1) as f32;
            blocks.extend_from_slice(&build_constant_block(v));
            blocks.extend_from_slice(&build_constant_block(v));
        }

        // Wrap as a QTensor with the PyTorch [out=256, in=512] convention.
        let mut src = blocks.clone();
        let bytes =
            crate::haxe_sys::HaxeBytes::new_malloc(src.as_mut_ptr(), src.len(), src.capacity());
        let handle = &bytes as *const _ as i64;

        // Build a 3-row batch of distinct inputs:
        //   x[0, :] = 1.0   (constant)
        //   x[1, :] = 0.5
        //   x[2, k] = (k as f32) / 256.0   (varies along k)
        let batch = 3usize;
        let k = 512usize;
        let mut x_data: Vec<f32> = Vec::with_capacity(batch * k);
        for _ in 0..k {
            x_data.push(1.0);
        }
        for _ in 0..k {
            x_data.push(0.5);
        }
        for kk in 0..k {
            x_data.push(kk as f32 / 256.0);
        }
        let x_shape = [batch, k];
        let x_tensor = unsafe {
            crate::tensor::rayzor_tensor_zeros(x_shape.as_ptr() as i64, 2, 0 /* F32 */)
        };
        assert!(x_tensor != 0);
        #[repr(C)]
        struct TensorHead {
            data: *mut u8,
            shape: *mut usize,
            strides: *mut usize,
            ndim: usize,
            numel: usize,
            dtype: u8,
            owns_data: bool,
            device: u8,
            numa_node: i32,
        }
        unsafe {
            let x_head = &*(x_tensor as *const TensorHead);
            std::ptr::copy_nonoverlapping(x_data.as_ptr(), x_head.data as *mut f32, batch * k);

            // Path A: dequant Wq to F32, then matmul_t.
            let qt_a = rayzor_qtensor_from_bytes_q4_k_m(handle, 256, 512);
            assert!(qt_a != 0);
            let dq = rayzor_qtensor_dequant(qt_a);
            assert!(dq != 0);
            let y_dq = crate::tensor::rayzor_tensor_matmul_t(x_tensor, dq);
            assert!(y_dq != 0);
            let y_dq_head = &*(y_dq as *const TensorHead);
            let y_dq_slice = std::slice::from_raw_parts(y_dq_head.data as *const f32, batch * 256);

            // Path B: the new fused kernel.
            let mut src2 = blocks.clone();
            let bytes2 = crate::haxe_sys::HaxeBytes::new_malloc(
                src2.as_mut_ptr(),
                src2.len(),
                src2.capacity(),
            );
            let handle2 = &bytes2 as *const _ as i64;
            let qt_b = rayzor_qtensor_from_bytes_q4_k_m(handle2, 256, 512);
            assert!(qt_b != 0);
            let y_fused = rayzor_tensor_matmul_qt_t_f32(x_tensor, qt_b);
            assert!(y_fused != 0);
            let y_fused_head = &*(y_fused as *const TensorHead);
            let y_fused_slice =
                std::slice::from_raw_parts(y_fused_head.data as *const f32, batch * 256);

            // Match element-wise. Q4_K_M dequant is deterministic so exact
            // equality should hold (or within 1e-4 for accumulated f32).
            for i in 0..(batch * 256) {
                let diff = (y_dq_slice[i] - y_fused_slice[i]).abs();
                // Relative tolerance — the two paths accumulate K=256
                // dot-product terms through different SIMD/scalar
                // sequences, so rounding can drift a few ULPs per
                // element. 1e-4 of the larger operand covers F32
                // accumulation noise; the end-to-end layer-diff
                // harness compares against llama.cpp directly for the
                // real bitwise sanity check.
                let tol = 1e-4 * y_dq_slice[i].abs().max(1.0);
                assert!(
                    diff < tol,
                    "y[{i}]: dequant-then-matmul_t={}, fused={}, diff={}, tol={}",
                    y_dq_slice[i],
                    y_fused_slice[i],
                    diff,
                    tol,
                );
            }

            rayzor_qtensor_free(qt_a);
            rayzor_qtensor_free(qt_b);
        }
    }
}
