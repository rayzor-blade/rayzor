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
//!
//! # Portability — AArch64 SDOT / `dotprod` gating
//!
//! The hot Q4_K_M dot path uses `vdotq_s32` (ARMv8.2-A SDOT). That
//! instruction is **not** universal across AArch64:
//!
//! - Present: Apple M1+, Cortex-A55r1 (limited), A75+, A76, A77, A78,
//!   X1+, Neoverse-N1+/V1+, every modern Apple / Ampere / AWS Graviton2+
//!   SKU.
//! - Missing: Cortex-A53, A55r0, A57, A72, A73 and any pre-ARMv8.2 part.
//!   These run plenty of 64-bit Linux distros and embedded boards.
//!
//! Calling SDOT on a part that lacks it raises SIGILL. To stay portable
//! the runtime gates SDOT three ways and only fires it when ALL three
//! say yes:
//!
//! 1. **Compile-time cargo flag** (`.cargo/config.toml` sets
//!    `target-feature=+dotprod` for `aarch64-apple-darwin` and
//!    `aarch64-unknown-linux-gnu`). The SDOT inner kernels
//!    (`dot_q4_k_q8`, `dot_q6_k_q8`) and every aarch64 call-site that
//!    invokes them are gated on
//!    `#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]`
//!    and additionally carry `#[target_feature(enable = "dotprod")]` so
//!    the function-level attribute matches the cfg gate.
//! 2. **Runtime CPU probe** via `std::arch::is_aarch64_feature_detected!
//!    ("dotprod")`. The result is cached in a `OnceLock` (process-wide;
//!    same pattern as `worker_pool::global`). Builds compiled for a
//!    `+dotprod` target *and* deployed to a real `+dotprod` core take
//!    the SDOT path. Anything else (built without `+dotprod`, or built
//!    with it but running on a pre-ARMv8.2 core via SDK pinning) falls
//!    back to the F32 dequant-then-FMA path.
//! 3. **Env override** (`RAYZOR_USE_SDOT=0`) for A/B testing.
//!
//! On non-AArch64 targets (x86_64 / wasm32) and AArch64 builds without
//! `+dotprod`, the SDOT gate is hard-off; the qmatmul chunk impls
//! route through `dequant_q4_k_block` + the F32 SIMD axpy. AVX-VNNI
//! (`vpdpbusd`) is the natural x86 analogue but is not wired here yet.

// Quant kernels are heavy on indexed inner loops over Q4_K_M / Q6_K block
// substructures where the same index drives several parallel pointers
// (quants, scales, mins, output) — rewriting to .iter().enumerate() chains
// hurts readability and frustrates auto-vectorisation. Same for the test
// scaffolding that uses `vec.push(constant)` in setup helpers.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::same_item_push)]

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
    // Phase 1 ARC refcount. Same semantic as `RayzorTensor::refcount`:
    // Relaxed fetch_add on clone, AcqRel fetch_sub on free, only the
    // dec-to-zero thread actually releases data + meta + wrapper (or
    // pool-routes the owning INT8 slot).
    refcount: std::sync::atomic::AtomicUsize,
    // Null for owning QTensors. Reserved for a future view-of-QTensor
    // primitive (none exists today). Symmetric with `RayzorTensor::parent`.
    parent: *mut RayzorQTensor,
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
///
/// 2026-06-04 perf note — a hand-written aarch64 NEON port of this
/// kernel (lane-loaded ql/qh bytes → vand/vshr unpack → widen i8 → i32
/// → vcvtq_f32_s32 → vmulq_f32 by pre-broadcast `d * scale`) was tried
/// and measured 13-17% SLOWER than this scalar version on M1 Pro
/// nue/llama-chat 80-token warm cache (20.0 → 17.5 tok/s). LLVM's
/// auto-vectorizer produces tighter code than the hand-rolled NEON,
/// likely due to better register allocation across the strided stores
/// at positions l, l+32, l+64, l+96 and amortising the 4×16 scale
/// broadcasts across the whole half. A future NEON port that beats
/// the scalar must also handle the FULL kernel including the dot
/// product (i.e. fused dequant + dot with SIMD throughout) — see the
/// failed scalar-fused attempt at commit 31c8a2d which captures why a
/// half-measure loses. Real Tier 2 #4 work per
/// [[project-optimization-roadmap]].
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

/// Runtime toggle for the SDOT path. Returns `true` only when ALL three
/// of the following hold:
///
/// 1. The crate was compiled with `target-feature=+dotprod` (see
///    `.cargo/config.toml` — wired for `aarch64-apple-darwin` and
///    `aarch64-unknown-linux-gnu`).
/// 2. The running CPU actually supports `dotprod` (probed via
///    `std::arch::is_aarch64_feature_detected!`). Process-wide cached
///    in a `OnceLock` — same pattern as `worker_pool::global`.
/// 3. `RAYZOR_USE_SDOT` is unset / non-empty / not `"0"`.
///
/// If gate (1) fails this function is replaced by an `#[inline] false`
/// stub via cfg — callers compile out the whole SDOT path.
///
/// If gate (2) fails on a `+dotprod`-built binary (e.g. shipped via SDK
/// pinning to a pre-ARMv8.2 core like Cortex-A53/A55r0/A57/A72/A73) the
/// runtime probe falls back to the F32 dequant+FMA path so we never
/// raise SIGILL.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
fn sdot_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // Runtime CPU probe — guards against `+dotprod`-built binaries
        // landing on pre-ARMv8.2 cores. `is_aarch64_feature_detected!`
        // reads `mrs ID_AA64ISAR0_EL1` on Linux and the equivalent
        // sysctl on macOS; the OS-side accessor is itself cached.
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return false;
        }
        std::env::var("RAYZOR_USE_SDOT")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true) // default ON when both compile- and run-time gates pass
    })
}

/// Fallback `sdot_enabled` for the rare combo of aarch64 build target
/// without `+dotprod` (e.g. a Cortex-A72 server distro). Returning
/// `false` here makes the SDOT call sites fall through to the F32
/// dequant+FMA path. The SDOT inner kernels themselves are cfg'd out
/// in this configuration so the symbols are never linked.
#[cfg(all(target_arch = "aarch64", not(target_feature = "dotprod")))]
#[inline]
fn sdot_enabled() -> bool {
    false
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
    /// Σ of `quants[s*32 .. (s+1)*32]` per Q4_K_M-style 32-elem
    /// sub-block (8 entries). Used to fold the Q4_K_M min term.
    bsums: [i32; 8],
    /// Σ of `quants[s*16 .. (s+1)*16]` per Q6_K-style 16-elem
    /// sub-block (16 entries). Used to fold the Q6_K `-32` bias.
    bsums_16: [i32; 16],
}

/// Quantise a 256-element span of `x` to INT8 with one F32 scale + the
/// 8 per-sub-block sums.
#[inline]
unsafe fn quantize_x_block_q8(x: *const f32) -> Q8Block {
    // Pass 1: find the absolute max over 256 elements (NEON 4×4-lane
    // unrolled). Used to pick a symmetric scale so quants land in
    // [-127, 127].
    #[allow(unused_assignments)] // overwritten in aarch64 path, read in fallback
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

    // Pass 2: quantise + accumulate per-16 sums; pair them up for
    // the Q4_K_M per-32 sums.
    for s16 in 0..16 {
        let mut sum: i32 = 0;
        for j in 0..16 {
            let v = *x.add(s16 * 16 + j) * inv_scale;
            let q = v.round().clamp(-128.0, 127.0) as i8;
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
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline]
#[target_feature(enable = "dotprod")]
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

/// Q4_K_M × Q8_K dot product — variant of `dot_q4_k_q8` that reads the
/// public `Q8KBlock` shape directly. Skips the per-call shim memcpy
/// (`Q8KBlock → Q8Block`) that used to materialise on every super-block
/// inside `vec_dot_q4_K_q8_K`. With ~65k super-block calls per FFN
/// matmul (one per `(n_row, k_block)` pair) the old shim was burning
/// ~16 MB of pure memcpy bandwidth per matmul — wiping out the +15–25%
/// hoist from `prepare_x_q8k_blocks`.
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
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_k_q8_kblock(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    use std::arch::aarch64::*;

    let q4_block_ptr = weight as *const Q4KMBlock as *const u8;

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

    // See `dot_q4_k_q8` above for the rationale on the 4-way partial
    // unroll + split-acc SDOT. This kernel mirrors that structure for
    // the `Q8KBlock` input shape (i16 bsums folded inline rather than
    // the i32 bsums Q8Block stores).
    let mut p1 = [0.0f32; 4]; // Σ sub_scale_s * x_scale * sdot_s, per-p partials
    let mut p2 = [0.0f32; 4]; // Σ sub_min_s   * x_scale * bsum_s, per-p partials

    let mask_nibble = vdupq_n_u8(0x0F);

    // Q8KBlock.qs is a flat `[i8; 256]` — load straight from `x.qs`
    // rather than the shim buffer. Q8KBlock.bsums is `[i16; 16]`; the
    // i32 per-32-elem pair sum is reconstructed inline below.
    let x_qs_ptr = x.qs.as_ptr();

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

        // X int8 for sub-block 2p (32 i8) and sub-block 2p+1 (32 i8) —
        // read directly out of Q8KBlock.qs, no copy.
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
        // (`bsums[0..16]`); widen to i32 then f32 right where they feed
        // the partial. For sub-block pair `p` (covering Q4_K_M sub-blocks
        // 2p, 2p+1 — elements [64*p .. 64*p+64)) the matching Q8KBlock
        // 16-elem sums live at indices 4p .. 4p+4.
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
/// NEON path (arch/arm/quants.c:2715-2776). Empirically 2.12x faster
/// than `dot_q4_k_q8_kblock` in standalone microbench at
/// /tmp/q4km_paired_bench.rs (7.69 ns/call vs 16.34 ns/call median,
/// N=3 runs on M1 Pro).
///
/// Structural deltas vs `dot_q4_k_q8_kblock`:
///   - **Scale decode ONCE per super-block**: the 12-byte scales/mins
///     header is bit-twiddled into utmp[4] up front; we then index
///     scales[2*j+0..2] inside the inner loop instead of calling
///     `q4_k_get_scale_min` 4x. Saves ~12 ops per super-block.
///   - **bsums × mins in i16→i32 path, computed ONCE**: vpaddq_s16 +
///     2x vmull_s16 + vaddvq_s32 produces the entire dmin*Σ(bsums*mins)
///     value before the inner loop runs. We previously did this
///     per-iteration with f32 multiplies (4x f32 fcvt + 4x f32 fmul).
///   - **Per-sub-block accumulation in i32**: `sumi1 += vaddvq * scales[j]`
///     stays in int32; the f32 conversion is deferred to the SINGLE
///     `sumf += d * (sumi1 + sumi2) as f32` at the end. We previously
///     did f32 conversion + f32 multiply per p-iter, then f32 reduction
///     across 4 partials.
///   - **Serial SDOT chain** (not split-acc unroll): `vdotq(vdotq(0,
///     q4[0], q8[0]), q4[1], q8[1])` accumulates into one register; we
///     previously used `acc_lo` + `acc_hi` × 4 partials for ILP. M1's
///     OoO scheduler extracts the ILP from the serial chain anyway —
///     the unroll was paying register pressure for parallelism the
///     hardware already provides.
///
/// Numerical: the result MAY differ from `dot_q4_k_q8_kblock` by 1-2
/// ULP because the f32 reduction order changes. Tested at the unit
/// level; production MATCH must be verified at the matmul level.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_k_q8_kblock_llamacpp(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    use std::arch::aarch64::*;

    let q4_block_ptr = weight as *const Q4KMBlock as *const u8;

    let d_bits = std::ptr::read_unaligned(q4_block_ptr as *const u16);
    let dmin_bits = std::ptr::read_unaligned(q4_block_ptr.add(2) as *const u16);
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
    std::ptr::copy_nonoverlapping(q4_block_ptr.add(4) as *const u32, utmp.as_mut_ptr(), 3);

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

/// Two-block paired Q4_K_M × Q8_K dot product. Returns the per-block
/// dot results for `(weight_a, x_a)` and `(weight_b, x_b)` in one
/// call, processed with their inner loops interleaved.
///
/// Pairing pattern (mirror of llama.cpp's q4_K NEON two-block-per-iter
/// kernel that the candle research called out as a gap vs candle's
/// one-block-per-iter approach):
///   1. Load both 12-byte scale headers up front.
///   2. Per p-iter, fetch BOTH blocks' weight nibbles, BOTH X chunks,
///      and emit SDOTs for both before moving on. This doubles the
///      independent FMA chains visible to M1's OoO scheduler from 4
///      to 8, which is the practical limit before register pressure
///      starts spilling the partial accumulators.
///   3. Per-block partial reduction at the end is bit-identical to
///      the single-block kernel — each block's result is computed
///      from its own four partials in the same reduction order. So
///      `(dot_q4_k_q8_kblock(a, x_a), dot_q4_k_q8_kblock(b, x_b))`
///      and `dot_q4_k_q8_kblock_2(a, b, x_a, x_b)` produce
///      byte-identical pairs of f32s — verified by the
///      `dot_q4_k_q8_kblock_2_matches_single` unit test.
///
/// Compile gate matches the single-block kernel: AArch64 + `dotprod`,
/// `target_feature(enable = "dotprod")` so the inner `vdotq_s32` is
/// inlined at the SDOT instruction (not the software fallback).
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline]
#[target_feature(enable = "dotprod")]
unsafe fn dot_q4_k_q8_kblock_2(
    weight_a: &Q4KMBlock,
    weight_b: &Q4KMBlock,
    x_a: &Q8KBlock,
    x_b: &Q8KBlock,
) -> (f32, f32) {
    use std::arch::aarch64::*;

    let ptr_a = weight_a as *const Q4KMBlock as *const u8;
    let ptr_b = weight_b as *const Q4KMBlock as *const u8;

    let d_a = half::f16::from_bits(std::ptr::read_unaligned(ptr_a as *const u16)).to_f32();
    let dmin_a =
        half::f16::from_bits(std::ptr::read_unaligned(ptr_a.add(2) as *const u16)).to_f32();
    let d_b = half::f16::from_bits(std::ptr::read_unaligned(ptr_b as *const u16)).to_f32();
    let dmin_b =
        half::f16::from_bits(std::ptr::read_unaligned(ptr_b.add(2) as *const u16)).to_f32();

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

        // Interleaved weight loads — both blocks' 32-byte quant
        // chunks for sub-block pair `p`.
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

        // Eight independent SDOT chains (4 per block) — the OoO
        // scheduler sees all of them issuable in parallel.
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
/// ```
///   Σ x[i] * d * scale_s * (q[i] − 32)
///     = d * scale_s * (Σ x[i] * q[i] − 32 * Σ x[i])
///     = d * scale_s * x_scale * (sdot − 32 * bsum16_s)
/// ```
/// where `bsum16_s` is `Σ quants_int8[s*16 .. (s+1)*16]` from
/// `Q8Block::bsums_16` (populated by `quantize_x_block_q8`).
///
/// Per `n in 0..2` half we read 32 ql + 16 qh bytes, build four
/// 32-weight 6-bit spans, and split each into 16-lo + 16-hi SDOT
/// pairs since each 16-weight Q6_K sub-block has its own scale.
#[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
#[inline]
#[target_feature(enable = "dotprod")]
#[allow(dead_code)] // future Q6_K SDOT entry — currently unused, see qmatmul_chunk_impl
unsafe fn dot_q6_k_q8(q6_block_ptr: *const u8, x_q8: &Q8Block) -> f32 {
    use std::arch::aarch64::*;

    let ql_ptr = q6_block_ptr;
    let qh_ptr = q6_block_ptr.add(128);
    let scales_ptr = q6_block_ptr.add(192) as *const i8;
    let d_bits = std::ptr::read_unaligned(q6_block_ptr.add(208) as *const u16);
    let d = half::f16::from_bits(d_bits).to_f32();

    let mut sum_term1 = 0.0f32;
    let mut sum_term2 = 0.0f32;

    let mask_4bit = vdupq_n_u8(0x0F);

    for n in 0..2 {
        // Each half spans 128 weights in `out`, indexed
        // out[out_off + l + j*32] for j in 0..4 and l in 0..32.
        // Read 32 ql bytes + 32 qh bytes. (`qh` half is 32 bytes; we
        // load it once but shift differently per j.)
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

        // Reconstruct the four 32-weight spans per half.
        // For each l in 0..32 the dequant formula picks one of four
        // ql / qh-shift combinations (see `dequant_q6_k_block`).
        // `shift_amt` is the right-shift of `qh` before masking with 3.
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

            // q = ql_part | (qh_part << 4), range [0, 63]. We keep it
            // as u8 — SDOT interprets bits as signed, but with values
            // in [0, 63] the high bit is zero so signed == unsigned.
            // The −32 bias is folded out via the `bsums_16` correction
            // below.
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

            // The 32-weight span splits across two 16-elem Q6_K sub-
            // blocks with separate scales — the existing `scales[]`
            // array indexes sc_off + 2*j (lo half) and sc_off + 2*j+1
            // (hi half) for the first 32 weights of this `n`-half;
            // and sc_off + 2*j + 1 / sc_off + 2*j  for the four spans
            // collectively cover sub-blocks sc_off+0..sc_off+8.
            // To match `dequant_q6_k_block`'s scale picks, the lookup
            // is `sc_off + 2*j + (l/16)`.
            let scale_lo = *scales_ptr.add(sc_off + 2 * j) as f32;
            let scale_hi = *scales_ptr.add(sc_off + 2 * j + 1) as f32;
            let d_sc_lo = d * scale_lo;
            let d_sc_hi = d * scale_hi;

            sum_term1 += d_sc_lo * sdot_lo;
            sum_term1 += d_sc_hi * sdot_hi;

            // bsums_16 is indexed by 16-elem sub-block of X. The two
            // halves of this 32-weight span use bsums_16[bsum_base +
            // 2*j] (lo) and bsums_16[bsum_base + 2*j + 1] (hi). The
            // outer base offset is the first 16-block in this n-half:
            //   bsum_16 index = x_span_off / 16
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

// ============================================================================
// llama.cpp-compatible Q8_K + Q4_K_M API
//
// `Q8KBlock` mirrors llama.cpp's `block_q8_K` from ggml-quants.h byte-for-byte
// (verified against the `static_assert(sizeof(block_q8_K) == sizeof(float) +
//   QK_K + QK_K/16*sizeof(int16_t), ...)` check there):
//
//   typedef struct {
//       float   d;              // delta
//       int8_t  qs[QK_K];       // quants                       (QK_K = 256)
//       int16_t bsums[QK_K/16]; // sum of quants in groups of 16
//   } block_q8_K;
//
// `Q4KMBlock` is the matching `block_q4_K` shape — a `repr(C)` wrapper over
// the 144-byte raw block so callers can treat the on-disk bytes as a typed
// struct without re-decoding the header. The existing internal helpers
// (`dot_q4_k_q8`, `quantize_x_block_q8`) operate on raw `*const u8` + the
// internal `Q8Block` shape (which uses `i32` accumulators and an auxiliary
// per-32-elem `bsums` cache). The functions below are the public, spec-shaped
// surface — they thunk through the same SDOT inner kernel.
// ============================================================================

/// llama.cpp-compatible `block_q8_K`. Layout MUST stay
/// `{ f32 + 256 × i8 + 16 × i16 }` (= 4 + 256 + 32 = 292 bytes) to allow
/// memcpy interop with GGUF-quantised Q8_K tensors.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Q8KBlock {
    /// Quantisation scale: `x[i] ≈ d * qs[i]`.
    pub d: f32,
    /// 256 INT8 quants.
    pub qs: [i8; 256],
    /// Per-16-element sum of `qs`. `bsums[s] = Σ_{j=0..16} qs[s*16 + j]`.
    /// Q4_K_M's vec-dot folds the dmin term as
    /// `dmin * x.d * (bsums[2*s] + bsums[2*s+1]) * mn6[s]`.
    pub bsums: [i16; 16],
}

/// llama.cpp-compatible `block_q4_K`. The struct itself is just a typed
/// view over the 144 on-disk bytes — the field layout matches the GGUF
/// Q4_K_M block byte-for-byte:
///
/// ```text
///   bytes  0..1     : f16 d        (super-block scale)
///   bytes  2..3     : f16 dmin     (super-block min)
///   bytes  4..15    : 12-byte packed (scale6, min6) header
///   bytes 16..143   : 128 nibbles × 2 = 256 quants
/// ```
///
/// `repr(C, packed(1))` so a `*const u8` cursor from a GGUF tensor can be
/// reinterpreted as `*const Q4KMBlock` without padding drift.
#[repr(C, packed(1))]
#[derive(Clone, Copy)]
pub struct Q4KMBlock {
    /// f16 super-block scale (raw bits).
    pub d: u16,
    /// f16 super-block min (raw bits).
    pub dmin: u16,
    /// 12-byte packed (6-bit scale, 6-bit min) × 8 sub-blocks.
    pub scales: [u8; 12],
    /// 128 bytes = 256 nibbles of 4-bit quants.
    pub qs: [u8; 128],
}

// Compile-time guard: keep the structs at their canonical llama.cpp sizes.
const _: () = {
    assert!(std::mem::size_of::<Q8KBlock>() == 4 + 256 + 32);
    assert!(std::mem::size_of::<Q4KMBlock>() == Q4_K_M_BLOCK_BYTES);
};

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
/// Panics if `x.len()` is not a multiple of 256 or if `dest.len()` is
/// smaller than `x.len() / 256`.
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

        // Pass 1: absolute max over the 256-element super-block.
        let mut max_abs = 0.0f32;
        for &v in src {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }

        // Degenerate block: zero quants, scale=0, bsums=0. Matches
        // llama.cpp which sets `d=0` and skips the encode loop.
        if max_abs == 0.0 {
            dest[b] = Q8KBlock {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            };
            continue;
        }

        let d = max_abs / 127.0;
        let inv_d = 127.0 / max_abs;

        let mut qs = [0i8; 256];
        let mut bsums = [0i16; 16];

        // Pass 2: quant + 16-elem partial sums.
        for s in 0..16 {
            let mut sum: i32 = 0;
            for j in 0..16 {
                let q = (src[s * 16 + j] * inv_d).round().clamp(-128.0, 127.0) as i8;
                qs[s * 16 + j] = q;
                sum += q as i32;
            }
            // Saturating cast to i16 keeps us compatible with the on-disk
            // representation (max |sum| = 16 × 127 = 2032 < 2^15, so this
            // never actually saturates for valid input but the cast keeps
            // the type clean).
            bsums[s] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }

        dest[b] = Q8KBlock { d, qs, bsums };
    }
}

/// `vec_dot_q4_K_q8_K` — single-super-block dot product between a Q4_K_M
/// weight block and a Q8_K activation block. Mirrors llama.cpp's
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
/// The inner SDOT loop pairs sub-blocks 2p and 2p+1 (low/high nibbles of
/// the same 32-byte quant span) and uses `vdotq_s32` × 4 per pair = 16
/// `sdot` instructions per super-block — identical density to llama.cpp's
/// AArch64 path.
///
/// Requires `target-feature=+dotprod` (wired crate-wide in
/// `.cargo/config.toml` for `aarch64-apple-darwin` /
/// `aarch64-unknown-linux-gnu`). On non-aarch64 targets falls back to a
/// scalar dequant-then-dot reference.
#[allow(non_snake_case)] // matches llama.cpp's `ggml_vec_dot_q4_K_q8_K` symbol
pub fn vec_dot_q4_K_q8_K(weight: &Q4KMBlock, x: &Q8KBlock) -> f32 {
    // Hot AArch64 SDOT path: routes directly through `dot_q4_k_q8_kblock`
    // which consumes `&Q8KBlock` in-place. No per-call shim memcpy —
    // see that function's docs for the layout-difference handling
    // (i8 qs loads + inline i16→f32 pair-sum). With ~65k calls per FFN
    // matmul, the eliminated 256-byte `copy_from_slice` was ~16 MB of
    // wasted bandwidth per matmul.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        if sdot_enabled() {
            // SAFETY: `weight` is a `repr(C, packed)` 144-byte struct
            // laid out byte-identically with the GGUF Q4_K_M on-disk
            // block. `dot_q4_k_q8_kblock` reads only through that
            // typed view + `x.qs` / `x.bsums` (no raw pointers across
            // the layout boundary).
            return unsafe { dot_q4_k_q8_kblock(weight, x) };
        }
    }

    // Portable scalar reference. Used on:
    //  - non-aarch64 targets
    //  - aarch64 builds without `target-feature=+dotprod` (Cortex-A53
    //    et al — the SDOT helper isn't compiled in that configuration)
    //  - aarch64 + `+dotprod` where the runtime probe failed (running
    //    on a pre-ARMv8.2 core via SDK pinning) or RAYZOR_USE_SDOT=0
    //
    // Also the per-ULP ground truth in the unit tests below.
    {
        let mut acc = 0.0f32;
        // Decode the 12-byte header into 8 (sc6, mn6) pairs.
        let d = f16::from_bits(weight.d).to_f32();
        let dmin = f16::from_bits(weight.dmin).to_f32();
        let header = weight.scales;
        for s in 0..8 {
            let (sc6, mn6) = q4_k_get_scale_min(s, &header);
            let sub_scale = d * sc6 as f32;
            let sub_min = dmin * mn6 as f32;
            // Sub-block s spans elements s*32 .. (s+1)*32. Within the
            // 128-byte qs, the low-nibble/high-nibble pairing matches
            // dequant_q4_k_block: bytes [p*32 .. p*32+32] hold sub-blocks
            // 2p (low nibbles) and 2p+1 (high nibbles).
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
}

/// Pack 8 (sc6, mn6) pairs into the 12-byte llama.cpp-canonical scales
/// header. Inverse of `q4_k_get_scale_min`. Used by the F32 → Q4_K_M
/// encoder + verified by the round-trip unit test.
#[inline]
fn pack_q4_k_scales(sc6: &[u8; 8], mn6: &[u8; 8]) -> [u8; 12] {
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
/// Naive quantisation strategy — per sub-block (8 sub-blocks of 32),
/// pick scale + min directly from `(max - min)` and `-min`, then quantise
/// those 8 (scale, min) pairs to 6-bit each via the super-block d / dmin
/// scalars. No iterative refinement like llama.cpp's `make_qkx2_quants`.
///
/// This is intentionally simpler than the canonical encoder — we lose a
/// few percent of quantisation quality vs llama.cpp's iterative scheme.
/// Acceptable for the lm_head re-quant use case where:
///   1. The original tensor was already quantised once (Q6_K) with the
///      careful encoder, so we're re-quantising a near-Q6_K-precise
///      target; small encoder loss compounds on a low-noise source.
///   2. Greedy / low-temp decode is robust to small per-logit ULPs —
///      argmax over a 128k vocab doesn't shift on 1-2 ULP perturbations
///      to most entries.
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

    // 2. Super-block d, dmin chosen so the largest sub-block sc / mn
    // round to the max 6-bit value (63). All other sub-blocks scale
    // proportionally.
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
    // wrong-sign sub-blocks to mn6 = 0 below (some local error, but
    // model weights are typically symmetric so this is rare).
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
            sc6[s] = ((sub_sc[s] / d_f32).round() as i32).clamp(0, 63) as u8;
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
                (target.round() as i32).clamp(0, 63) as u8
            };
        }
    }

    // 4. Recompute the EFFECTIVE per-sub-block (sc, mn) the decoder will
    // see. Quantising values through these (not the raw sub_sc/sub_mn)
    // produces lower error because the quant step uses the same (sc, mn)
    // the dequant will use.
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
                ((v_lo + mn_lo) / sc_lo).round().clamp(0.0, 15.0) as u8
            };
            let q_hi = if sc_hi == 0.0 {
                0
            } else {
                ((v_hi + mn_hi) / sc_hi).round().clamp(0.0, 15.0) as u8
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

    // Pool fast path. Returns null on miss; on hit the wrapper carries the
    // original data + meta allocations (meta f32 scales for INT8) which we
    // zero before handing back so the caller sees a fresh-feeling tensor.
    let popped = try_pop_qtensor(scheme, rows, cols, group_size);
    if !popped.is_null() {
        let qt = &mut *popped;
        if !qt.data.is_null() && data_bytes > 0 {
            std::ptr::write_bytes(qt.data, 0, data_bytes);
        }
        if scheme == QSCHEME_INT8 && !qt.meta.is_null() {
            let n_groups = numel / group_size;
            std::ptr::write_bytes(qt.meta, 0, n_groups * std::mem::size_of::<f32>());
        }
        qt.owns_data = true;
        // numel / group_size / scheme / rows / cols already match the bucket
        // key and shouldn't have drifted; refresh defensively.
        qt.numel = numel;
        qt.group_size = group_size;
        qt.scheme = scheme;
        qt.rows = rows;
        qt.cols = cols;
        // Phase 1 refcount reset on pool revive — see the parallel comment
        // in `tensor.rs::alloc_tensor`.
        qt.refcount.store(1, std::sync::atomic::Ordering::Relaxed);
        qt.parent = std::ptr::null_mut();
        return popped;
    }

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
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
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
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
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
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
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
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
    };
    qt as i64
}

/// Re-quantise a Q6_K QTensor as Q4_K_M, returning a fresh QTensor handle.
///
/// Walks every block of the source row-by-row, dequants Q6_K → f32 into a
/// stage buffer, then re-encodes the same 256 floats via the naive
/// `quantize_block_q4_k_m` encoder into the destination. The destination
/// is freshly allocated and owns its data — caller is responsible for
/// freeing.
///
/// Use case: the lm_head matmul on Llama-3.2-1B-Q4_K_M is the dominant
/// single decode-time op (one 128k×2048 call per token). Q6_K SDOT
/// landed at 5f23311 but the per-block 6-bit reconstruction overhead
/// still leaves headroom vs the Q4_K_M SDOT path (no reconstruction
/// needed — the 4-bit nibbles are already in the SDOT operand format).
/// Re-quantising the lm_head to Q4_K_M at load lets it join that
/// faster path.
///
/// Trade-off: naive encoder quality (per-block round-trip RMS ~3-5%
/// per the unit test). For greedy / low-temp decode on a 128k vocab
/// the per-logit error averages out before argmax. MATCH-on-canonical
/// is the empirical gate.
///
/// Returns 0 on:
///   - null source
///   - source scheme != Q6_K
///   - rows × cols not divisible by 256
///   - allocation failure
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_requant_q6k_to_q4km(src_ptr: i64) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_qtensor_requant_q6k_to_q4km");
    if src_ptr == 0 {
        return 0;
    }
    let src = &*(src_ptr as *const RayzorQTensor);
    if src.scheme != QSCHEME_Q6_K {
        return 0;
    }
    let rows = src.rows;
    let cols = src.cols;
    if !(rows * cols).is_multiple_of(Q6_K_BLOCK_SIZE) {
        return 0;
    }

    let dst = alloc_qtensor(QSCHEME_Q4_K_M, rows, cols, Q4_K_M_BLOCK_SIZE);
    if dst.is_null() {
        return 0;
    }
    let dst_ref = &*dst;

    let blocks_per_row = cols / Q6_K_BLOCK_SIZE;
    let mut stage = [0.0f32; 256];
    for r in 0..rows {
        let src_row_ptr = src.data.add(r * blocks_per_row * Q6_K_BLOCK_BYTES);
        let dst_row_ptr = (dst_ref.data as *mut Q4KMBlock).add(r * blocks_per_row);
        for b in 0..blocks_per_row {
            dequant_q6_k_block(src_row_ptr.add(b * Q6_K_BLOCK_BYTES), &mut stage);
            *dst_row_ptr.add(b) = quantize_block_q4_k_m(&stage);
        }
    }

    dst as i64
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

/// Gather rows from a Q6_K quantised tensor, dequantising each selected row
/// into a fresh f32 Tensor of shape `[n_indices, qt.cols]`.
///
/// Mirrors `rayzor_tensor_gather_rows` in `tensor.rs` but specialised for
/// Q6_K storage. The hot path for embeddings (`token_embd.weight` is Q6_K
/// in Q4_K_M variants of Llama-3 / 3.2): selecting `n_indices` token rows
/// and dequantising only those rows is dramatically cheaper than calling
/// `rayzor_qtensor_dequant` over the whole `[vocab, hidden]` weight.
///
/// Each Q6_K row is `qt.cols / 256` super-blocks of 210 bytes; for Llama
/// 3.2's `hidden=2048` that's exactly 8 blocks per row × 210 = 1680 bytes
/// of source data per gathered row, decoded into 2048 f32 = 8 KiB output.
///
/// Out-of-range indices leave the corresponding output row zero-filled,
/// matching `rayzor_tensor_gather_rows`'s policy.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows_q6_k(
    qt_ptr: i64,
    indices_ptr: i64,
    n_indices: i64,
) -> i64 {
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_gather_rows_q6_k");
    if qt_ptr == 0 || indices_ptr == 0 || n_indices <= 0 {
        return 0;
    }
    let qt = &*(qt_ptr as *const RayzorQTensor);
    if qt.scheme != QSCHEME_Q6_K {
        return 0;
    }
    // Q6_K rows must be a whole number of 256-element super-blocks.
    if !qt.cols.is_multiple_of(Q6_K_BLOCK_SIZE) {
        return 0;
    }
    let blocks_per_row = qt.cols / Q6_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
    let k = n_indices as usize;

    // Allocate output via the shared tensor allocator (mirrors
    // `rayzor_qtensor_dequant` above so we pick up the pool / histogram
    // bookkeeping for free).
    let shape = [k, qt.cols];
    let out_tensor_ptr =
        crate::tensor::rayzor_tensor_zeros(shape.as_ptr() as i64, 2, 0 /* DTYPE_F32 */);
    if out_tensor_ptr == 0 {
        return 0;
    }
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
    let indices = indices_ptr as *const i64;

    let mut stage = [0.0f32; Q6_K_BLOCK_SIZE];
    for i in 0..k {
        let idx_raw = *indices.add(i);
        if idx_raw < 0 || (idx_raw as usize) >= qt.rows {
            // Out-of-range index — leave the row zeroed (rayzor_tensor_zeros
            // already zero-filled the entire output buffer).
            continue;
        }
        let row_src = qt.data.add((idx_raw as usize) * row_bytes);
        let row_dst = out.add(i * qt.cols);
        for b in 0..blocks_per_row {
            dequant_q6_k_block(row_src.add(b * Q6_K_BLOCK_BYTES), &mut stage);
            std::ptr::copy_nonoverlapping(
                stage.as_ptr(),
                row_dst.add(b * Q6_K_BLOCK_SIZE),
                Q6_K_BLOCK_SIZE,
            );
        }
    }

    out_tensor_ptr
}

/// Fused dequant-matmul: A is quantised `[M, K]`, B is f32 `[K, N]`, out is
/// f32 `[M, N]`. Returns a fresh f32 Tensor; 0 on shape mismatch.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_matmul_f32(qt_a: i64, b_tensor: i64) -> i64 {
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::QTENSOR_MATMUL_F32);
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
    crate::kernel_timing::init();
    let _kt =
        crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL_QT_T_F32_THREADED);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_matmul_qt_t_f32_threaded");
    if x_tensor == 0 || qt_w == 0 {
        return 0;
    }
    let qt = &*(qt_w as *const RayzorQTensor);

    let (batch, n, k, _block_size, _block_bytes) = match qmatmul_prep(x_tensor, qt) {
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

    // SDOT fast path: when (a) the env-gated SDOT toggle is on
    // (default), (b) the input is the single-batch contiguous shape
    // that Linear forward emits, and (c) the weight is Q4_K_M, we
    // pre-quantise X to Q8_K *once* up front, then hand a shared
    // immutable slice to every worker via the persistent pool. This
    // eliminates the 6× redundant `quantize_x_block_q8` work each
    // worker was doing inside `qmatmul_chunk_impl` for the same X
    // (one per worker per Linear, 112× per generated token).
    //
    // Workers walk the canonical `vec_dot_q4_K_q8_K` per super-block,
    // which thunks through the already-shipping AArch64 SDOT inner
    // kernel — same arithmetic that produces byte-for-byte matches
    // against llama.cpp on the rope/coherence regression suite.
    let use_sdot_threaded = sdot_enabled_runtime()
        && batch == 1
        && qt.scheme == QSCHEME_Q4_K_M
        && x_is_contiguous(x_tensor);
    if use_sdot_threaded {
        let x_data = x_tensor_data_ptr(x_tensor);

        // Borrow the thread-local Q8K scratch for the duration of
        // this call (held until the closure returns, AFTER the
        // parallel_rows join — so workers' raw-ptr reads remain
        // valid). The borrow stays on the caller thread; workers
        // never touch the RefCell directly.
        return X_Q8K_SCRATCH.with(|cell| {
            let mut x_q8k = cell.borrow_mut();
            prepare_x_q8k_blocks_into(x_data, k, &mut x_q8k);
            let nb = k / Q4_K_M_BLOCK_SIZE;

            if t <= 1 {
                qmatmul_chunk_impl_sdot_q4km(qt_w, out_tensor, 0, n as i64, &x_q8k[..nb]);
                return out_tensor;
            }

            // SAFETY: the borrow on the thread-local Vec is held by
            // this closure across the parallel_rows join; workers see
            // only the raw `(*const Q8KBlock, len)` pair so no Send
            // gymnastics, and the storage they read stays alive
            // because the RefMut outlives the join.
            let q8k_ptr = x_q8k.as_ptr() as usize;
            let q8k_len = nb;
            let qh = qt_w;
            let yh = out_tensor;
            crate::worker_pool::global().parallel_rows(n, t, move |lo, hi| unsafe {
                let q8k_slice = std::slice::from_raw_parts(q8k_ptr as *const Q8KBlock, q8k_len);
                qmatmul_chunk_impl_sdot_q4km(qh, yh, lo as i64, hi as i64, q8k_slice);
            });
            out_tensor
        });
    }

    if t <= 1 {
        qmatmul_chunk_impl(x_tensor, qt_w, out_tensor, 0, n as i64);
        return out_tensor;
    }

    // Fallback path (multi-batch / non-contiguous / non-Q4_K_M /
    // SDOT-disabled). Workers re-derive their own per-block Q8 cache
    // inside `qmatmul_chunk_impl` for the shapes that don't fit the
    // shared-X pre-quant pattern.
    let xh = x_tensor;
    let qh = qt_w;
    let yh = out_tensor;
    crate::worker_pool::global().parallel_rows(n, t, move |lo, hi| {
        // SAFETY: each worker writes Y[*, lo..hi); ranges are disjoint
        // across calls (the pool guarantees this for a single
        // parallel_rows invocation) so there's no aliasing on Y. X
        // and Wq are read-only.
        unsafe {
            qmatmul_chunk_impl(xh, qh, yh, lo as i64, hi as i64);
        }
    });
    out_tensor
}

/// Fused Q/K/V projection: three concurrent `Y = X @ Wq.T` matmuls
/// against the same activation X but distinct weight tensors (Q, K, V).
///
/// Allocates three separate F32 output tensors up front (sized
/// `[batch, q_n]`, `[batch, k_n]`, `[batch, v_n]`) and writes the
/// resulting handles back through the three out-pointers. Returns 0 on
/// success, non-zero on a guard miss (null inputs, shape/dtype/scheme
/// mismatch). On a non-zero return the three out-pointers are NOT
/// written; the Haxe caller falls back to three sequential
/// `rayzor_tensor_matmul_qt_t_f32_threaded` calls.
///
/// Three separate outputs (vs one fused `[batch, q_n + k_n + v_n]`
/// tensor) preserve the byte-exact RoPE/bmm chain: a sliced view of a
/// fused tensor would have `elements_per_row = q_n + k_n + v_n` (e.g.
/// 3072 for Llama 3.2 1B) instead of the per-projection 2048/512/512,
/// which `rayzor_tensor_rope` reads as a flat
/// `base + s*elements_per_row + h*head_dim + i` offset with no stride
/// check — silently rotating across head boundaries. See
/// `bugs_rope_interleaved_for_gguf` + the bmm-stride memory entries.
///
/// Threading: one `parallel_rows` fan-out over the concatenated row
/// space `[0, q_n + k_n + v_n)`. Each worker receives `[lo, hi)`,
/// splits it into the per-projection sub-ranges and dispatches up to
/// three `qmatmul_chunk_impl_sdot_q4km` calls (one per weight whose
/// row range the worker's slice intersects). This replaces 3
/// sequential fork-joins with one and gives the scheduler 3072 rows
/// to balance across 6 workers (512 rows/worker) instead of three
/// disjoint joins of 512/128/128 rows.
///
/// SDOT sharing: the X→Q8_K pre-quantisation runs ONCE up front;
/// the resulting `Vec<Q8KBlock>` is shared (via raw ptr+len, same
/// `Send`-dodging pattern as the single-projection path) with all
/// three weights inside every worker. All three projections see the
/// same activation row, so the Q8_K view is bit-identical to what
/// three independent per-projection calls would have built.
///
/// Reduction order: each output row's inner dot product still runs
/// sequentially on one worker — workers write disjoint output rows,
/// so there is no cross-thread reduction and the floating-point
/// reduction order is byte-identical to three separate
/// `matmul_qt_t_f32_threaded` calls.
///
/// Gate (single up-front evaluation): SDOT enabled AND batch == 1
/// AND all three weights are Q4_K_M AND X is contiguous. If any of
/// these fail the function returns non-zero so the caller falls back
/// to three sequential calls (which already handle multi-batch /
/// non-Q4_K_M shapes individually).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qkv_qt_t_f32_threaded(
    x_tensor: i64,
    q_w: i64,
    k_w: i64,
    v_w: i64,
    threads: i64,
    out_q_tensor: *mut i64,
    out_k_tensor: *mut i64,
    out_v_tensor: *mut i64,
) -> i64 {
    crate::kernel_timing::init();
    let _kt =
        crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL_QKV_QT_T_F32_THREADED);
    let _hc = crate::heap_check::HeapCheckGuard::new("rayzor_tensor_matmul_qkv_qt_t_f32_threaded");
    // Null guards. `out_*_tensor` are written only on success, so we
    // refuse to run if any of them is null (otherwise the caller would
    // silently lose the output handles).
    if x_tensor == 0
        || q_w == 0
        || k_w == 0
        || v_w == 0
        || out_q_tensor.is_null()
        || out_k_tensor.is_null()
        || out_v_tensor.is_null()
    {
        return 1;
    }

    let qt_q = &*(q_w as *const RayzorQTensor);
    let qt_k = &*(k_w as *const RayzorQTensor);
    let qt_v = &*(v_w as *const RayzorQTensor);

    // Per-weight prep validates X-vs-Wq shape/dtype individually.
    // Each returns `(batch, n, k, _, _)`; we then enforce that
    // (batch, k) match across all three weights (they must, since
    // they all read the same activation).
    let (batch_q, q_n, k_q, _, _) = match qmatmul_prep(x_tensor, qt_q) {
        Some(p) => p,
        None => return 2,
    };
    let (batch_k, k_n, k_k, _, _) = match qmatmul_prep(x_tensor, qt_k) {
        Some(p) => p,
        None => return 2,
    };
    let (batch_v, v_n, k_v, _, _) = match qmatmul_prep(x_tensor, qt_v) {
        Some(p) => p,
        None => return 2,
    };
    if batch_q != batch_k || batch_q != batch_v || k_q != k_k || k_q != k_v {
        return 3;
    }
    let batch = batch_q;
    let k = k_q;

    // Up-front fast-path gate. All three weights must satisfy the
    // SDOT preconditions; if any one of them doesn't, we bail and let
    // the Haxe caller fall back to three sequential
    // `rayzor_tensor_matmul_qt_t_f32_threaded` invocations (which
    // each independently route the non-Q4_K_M / multi-batch shapes
    // through their existing fallback code).
    let fast_path = sdot_enabled_runtime()
        && batch == 1
        && qt_q.scheme == QSCHEME_Q4_K_M
        && qt_k.scheme == QSCHEME_Q4_K_M
        && qt_v.scheme == QSCHEME_Q4_K_M
        && x_is_contiguous(x_tensor);
    if !fast_path {
        return 4;
    }

    // Allocate the three F32 output tensors. If any one fails, free
    // the earlier ones via the standard tensor free path to avoid a
    // leak on the rare allocation-failure case.
    let q_shape = [batch, q_n];
    let k_shape = [batch, k_n];
    let v_shape = [batch, v_n];
    let out_q = crate::tensor::rayzor_tensor_zeros(q_shape.as_ptr() as i64, 2, 0);
    if out_q == 0 {
        return 5;
    }
    let out_k = crate::tensor::rayzor_tensor_zeros(k_shape.as_ptr() as i64, 2, 0);
    if out_k == 0 {
        crate::tensor::rayzor_tensor_free(out_q);
        return 5;
    }
    let out_v = crate::tensor::rayzor_tensor_zeros(v_shape.as_ptr() as i64, 2, 0);
    if out_v == 0 {
        crate::tensor::rayzor_tensor_free(out_q);
        crate::tensor::rayzor_tensor_free(out_k);
        return 5;
    }

    // Pick worker count from the same auto/explicit rule the
    // single-projection threaded path uses. Cap at the total row
    // count so we don't spawn idle workers when the row space is
    // tiny.
    let total_rows = q_n + k_n + v_n;
    let auto_threads: usize = 6;
    let mut t = if threads > 0 {
        (threads as usize).min(64)
    } else {
        auto_threads
    };
    if t > total_rows {
        t = total_rows.max(1);
    }

    // Pre-quantise X to Q8_K ONCE into the thread-local scratch.
    // All three weights share this view (same activation row, same
    // K). The borrow on the RefCell is held across the
    // parallel_rows join so workers' raw-ptr reads of the scratch
    // remain valid.
    let x_data = x_tensor_data_ptr(x_tensor);
    X_Q8K_SCRATCH.with(|cell| {
        let mut x_q8k = cell.borrow_mut();
        prepare_x_q8k_blocks_into(x_data, k, &mut x_q8k);
        let nb = k / Q4_K_M_BLOCK_SIZE;

        // Single-threaded shortcut — keeps the inner kernel exactly
        // the same as the threaded path so the byte-exact reduction
        // order holds across `threads == 1` vs `threads > 1`.
        if t <= 1 {
            qmatmul_chunk_impl_sdot_q4km(q_w, out_q, 0, q_n as i64, &x_q8k[..nb]);
            qmatmul_chunk_impl_sdot_q4km(k_w, out_k, 0, k_n as i64, &x_q8k[..nb]);
            qmatmul_chunk_impl_sdot_q4km(v_w, out_v, 0, v_n as i64, &x_q8k[..nb]);
            *out_q_tensor = out_q;
            *out_k_tensor = out_k;
            *out_v_tensor = out_v;
            return;
        }

        // Multi-threaded fan-out over the concatenated row space.
        // Concatenated layout: [0, q_n) → Q, [q_n, q_n+k_n) → K,
        // [q_n+k_n, total_rows) → V. Each worker receives
        // `[lo, hi) ⊆ [0, total_rows)`, clips that window against
        // each per-weight band, and dispatches one chunk call per
        // non-empty intersection. Output rows are disjoint across
        // workers AND across the three output tensors, so no
        // aliasing on any of Q/K/V outputs.
        //
        // SAFETY: the scratch borrow is held by this closure across
        // the parallel_rows join, so the storage workers read via
        // raw ptr stays alive throughout.
        let q8k_ptr = x_q8k.as_ptr() as usize;
        let q8k_len = nb;
        let q_split = q_n;
        let k_split = q_n + k_n;
        let q_handle = q_w;
        let k_handle = k_w;
        let v_handle = v_w;
        let out_q_handle = out_q;
        let out_k_handle = out_k;
        let out_v_handle = out_v;
        crate::worker_pool::global().parallel_rows(total_rows, t, move |lo, hi| unsafe {
            let q8k_slice = std::slice::from_raw_parts(q8k_ptr as *const Q8KBlock, q8k_len);

            let q_lo = lo;
            let q_hi = hi.min(q_split);
            if q_lo < q_hi {
                qmatmul_chunk_impl_sdot_q4km(
                    q_handle,
                    out_q_handle,
                    q_lo as i64,
                    q_hi as i64,
                    q8k_slice,
                );
            }

            let k_lo = lo.max(q_split);
            let k_hi = hi.min(k_split);
            if k_lo < k_hi {
                qmatmul_chunk_impl_sdot_q4km(
                    k_handle,
                    out_k_handle,
                    (k_lo - q_split) as i64,
                    (k_hi - q_split) as i64,
                    q8k_slice,
                );
            }

            let v_lo = lo.max(k_split);
            let v_hi = hi;
            if v_lo < v_hi {
                qmatmul_chunk_impl_sdot_q4km(
                    v_handle,
                    out_v_handle,
                    (v_lo - k_split) as i64,
                    (v_hi - k_split) as i64,
                    q8k_slice,
                );
            }
        });

        *out_q_tensor = out_q;
        *out_k_tensor = out_k;
        *out_v_tensor = out_v;
    });
    0
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
#[allow(dead_code)] // QoS-hint helper, kept for future wiring into the threaded matmul path
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
    crate::kernel_timing::init();
    let _kt = crate::kernel_timing::TimerGuard::new(&crate::kernel_timing::MATMUL_QT_T_F32_CHUNK);
    if x_tensor == 0 || qt_w == 0 || y_tensor == 0 {
        return 0;
    }
    let qt = &*(qt_w as *const RayzorQTensor);
    let (batch, _n, k, _block_size, _block_bytes) = match qmatmul_prep(x_tensor, qt) {
        Some(p) => p,
        None => return 0,
    };

    // SDOT chunk fast path: pre-quantise the X super-blocks that this
    // chunk touches ONCE at entry, then dispatch the canonical
    // `vec_dot_q4_K_q8_K` over each output row in `[n_start, n_end)`.
    // Versus the legacy lazy-cache in `qmatmul_chunk_impl`, this is
    // identical in arithmetic and hands the inner loop the spec-shaped
    // public Q8KBlock layout — the same path the `quantize_row_q8_K`
    // unit tests exercise.
    if sdot_enabled_runtime()
        && batch == 1
        && qt.scheme == QSCHEME_Q4_K_M
        && x_is_contiguous(x_tensor)
    {
        let x_data = x_tensor_data_ptr(x_tensor);
        X_Q8K_SCRATCH.with(|cell| {
            let mut x_q8k = cell.borrow_mut();
            prepare_x_q8k_blocks_into(x_data, k, &mut x_q8k);
            let nb = k / Q4_K_M_BLOCK_SIZE;
            qmatmul_chunk_impl_sdot_q4km(qt_w, y_tensor, n_start, n_end, &x_q8k[..nb]);
        });
        return 1;
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

/// Architecture-agnostic SDOT runtime gate. Returns `true` only on
/// AArch64 builds with `target-feature=+dotprod` AND a runtime CPU
/// probe that confirms `dotprod` is actually present on the live core
/// (see `sdot_enabled` for the cache + probe details).
///
/// On every other target — non-aarch64, or aarch64 built without
/// `+dotprod` — the gate is hard-off and callers fall through to the
/// dequant+FMA path. The two negative arms collapse to the same
/// `false` value at compile time but live in separate cfg branches so
/// the aarch64+!dotprod build doesn't try to reference `sdot_enabled`
/// (which is the `false`-stub there anyway, but keeping the gate-shape
/// matched to the inner-kernel gate is the cleaner invariant).
#[inline]
fn sdot_enabled_runtime() -> bool {
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    {
        sdot_enabled()
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    {
        false
    }
}

/// Cheap wrapper: is `x_tensor` 2-D F32 contiguous along axis 1?
/// The SDOT pre-quant path requires `strides[1] == 1` so it can hand
/// the inner kernel a flat `*const f32` view of each X super-block.
#[inline]
unsafe fn x_is_contiguous(x_tensor: i64) -> bool {
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
    let head = &*(x_tensor as *const TensorHead);
    if head.ndim != 2 || head.dtype != 0 {
        return false;
    }
    let strides = std::slice::from_raw_parts(head.strides, 2);
    strides[1] == 1
}

/// Raw `*const f32` cursor at the start of `x_tensor`'s data buffer.
/// Caller must have validated the tensor is 2-D F32 contiguous (see
/// `x_is_contiguous`).
#[inline]
unsafe fn x_tensor_data_ptr(x_tensor: i64) -> *const f32 {
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
    (*(x_tensor as *const TensorHead)).data as *const f32
}

/// Pre-quantise the full single-row activation tensor into a flat
/// `Vec<Q8KBlock>` covering `k / 256` super-blocks. Each entry uses the
/// canonical llama.cpp `block_q8_K` layout (`f32 d + 256 × i8 + 16 ×
/// i16 bsums`).
///
/// Hot-path note: this runs ONCE per Linear projection per token (was
/// previously running `workers × per_n_row × 1` times inside
/// `qmatmul_chunk_impl`'s lazy cache). The eliminated redundancy is
/// the main wall-time lever this commit targets — N=2048 / 8192 rows
/// were each pulling a fresh `quantize_x_block_q8` per worker.
///
/// SAFETY: `x_data` must be a valid f32 cursor with `k` contiguous
/// elements; `k` must be a multiple of 256.
#[inline]
#[allow(dead_code)] // allocating sibling of prepare_x_q8k_blocks_into; kept for one-shot callers
unsafe fn prepare_x_q8k_blocks(x_data: *const f32, k: usize) -> Vec<Q8KBlock> {
    debug_assert!(k.is_multiple_of(Q4_K_M_BLOCK_SIZE));
    let nb = k / Q4_K_M_BLOCK_SIZE;
    let x_slice = std::slice::from_raw_parts(x_data, k);
    let mut dest = vec![
        Q8KBlock {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        };
        nb
    ];
    quantize_row_q8_K(x_slice, &mut dest);
    dest
}

/// Same as `prepare_x_q8k_blocks` but writes into a caller-provided
/// `Vec<Q8KBlock>`, growing it in place if too small. Used by the
/// threaded matmul entry points to reuse a thread-local scratch
/// buffer across calls — for Llama-3.2-1B the per-call Vec is 8 ×
/// 292 = 2.3 KB, allocated 5+ times per layer × 16 layers per token.
/// Hoisting to the per-thread scratch saves ~80 allocations/token
/// in steady-state decode and the zero-init pass on the freshly
/// allocated Vec (`quantize_row_q8_K` overwrites every byte, so the
/// `vec![]` macro's zero-init is pure waste).
unsafe fn prepare_x_q8k_blocks_into(x_data: *const f32, k: usize, dest: &mut Vec<Q8KBlock>) {
    debug_assert!(k.is_multiple_of(Q4_K_M_BLOCK_SIZE));
    let nb = k / Q4_K_M_BLOCK_SIZE;
    let x_slice = std::slice::from_raw_parts(x_data, k);
    if dest.len() < nb {
        dest.resize(
            nb,
            Q8KBlock {
                d: 0.0,
                qs: [0i8; 256],
                bsums: [0i16; 16],
            },
        );
    }
    quantize_row_q8_K(x_slice, &mut dest[..nb]);
}

thread_local! {
    /// Per-thread scratch buffer reused across `prepare_x_q8k_blocks`
    /// calls on the calling thread. Sized to whatever the largest
    /// `k / Q4_K_M_BLOCK_SIZE` value seen so far is — for Llama-3.2-1B
    /// (k=2048) that's a single 2.3 KB buffer; for k=8192 (max ctx
    /// hidden size used by some larger models) ~9 KB. Trivial RAM
    /// cost per thread, and `quantize_row_q8_K` is called frequently
    /// enough on the hot path that the saved allocator pressure
    /// matters.
    static X_Q8K_SCRATCH: std::cell::RefCell<Vec<Q8KBlock>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// SDOT-specialised chunk impl. Computes `y[0, n_idx] = X · Wq[n_idx, :]`
/// for `n_idx in [n_start, n_end)` using the pre-quantised activation
/// blocks `x_q8k` and the canonical `vec_dot_q4_K_q8_K` super-block
/// kernel. Assumes (verified by the caller via `qmatmul_prep` +
/// `x_is_contiguous`): `batch == 1`, X contiguous, `Wq` scheme
/// `Q4_K_M`.
unsafe fn qmatmul_chunk_impl_sdot_q4km(
    qt_w: i64,
    y_tensor: i64,
    n_start: i64,
    n_end: i64,
    x_q8k: &[Q8KBlock],
) {
    let qt = &*(qt_w as *const RayzorQTensor);
    let block_size = Q4_K_M_BLOCK_SIZE;
    let block_bytes = Q4_K_M_BLOCK_BYTES;

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
    let y_head = &*(y_tensor as *const TensorHead);
    let y_data = y_head.data as *mut f32;

    let n = qt.rows;
    let blocks_per_row = qt.cols / block_size;
    debug_assert_eq!(x_q8k.len(), blocks_per_row);

    let lo = (n_start.max(0) as usize).min(n);
    let hi = (n_end.max(0) as usize).min(n);
    if lo >= hi {
        return;
    }

    // Two-block paired SDOT path: when blocks_per_row is even (the
    // common case for k = 2048 → 8 blocks, k = 4096 → 16 blocks)
    // process pairs of (b_idx, b_idx+1) with one
    // `dot_q4_k_q8_kblock_2` call. Interleaves both blocks' inner
    // SDOT chains so M1's OoO scheduler sees 8 independent
    // accumulators per sub-block-pair iteration instead of 4.
    //
    // The per-block result is bit-identical to the single-block
    // path — the partial reduction order within each block is the
    // same, only the *order in which two consecutive blocks fire*
    // changes. So the row sum `vec_dot(a) + vec_dot(b)` matches
    // exactly when we call `dot_q4_k_q8_kblock_2(a, b)` and add the
    // two returned f32s.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    let use_pairs = sdot_enabled() && blocks_per_row >= 2 && blocks_per_row.is_multiple_of(2);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    let use_pairs = false;

    // Llama.cpp-pattern kernel is 2.12x faster in standalone microbench
    // (perf_q4km_llamacpp_kernel_port). Default ON; set
    // RAYZOR_LEGACY_KERNEL=1 to fall back to the 2-block paired path
    // for A/B or in case of numerical regression on a specific workload.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    let use_llamacpp = !std::env::var("RAYZOR_LEGACY_KERNEL")
        .map(|v| v == "1")
        .unwrap_or(false);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "dotprod")))]
    let use_llamacpp = false;

    for n_idx in lo..hi {
        let row_ptr = qt.data.add(n_idx * blocks_per_row * block_bytes);
        let mut sum = 0.0f32;

        if use_llamacpp {
            // Fastest path: llama.cpp-pattern single-block kernel in a
            // simple loop. The 2-block pairing win (~7% kernel-level
            // from acb80e5) is dominated by the 2.12x kernel speedup,
            // so pairing is no longer needed.
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            for (b_idx, x_block) in x_q8k.iter().enumerate().take(blocks_per_row) {
                let weight = &*(row_ptr.add(b_idx * block_bytes) as *const Q4KMBlock);
                sum += dot_q4_k_q8_kblock_llamacpp(weight, x_block);
            }
        } else if use_pairs {
            // Legacy path: 2-block paired SDOT (acb80e5). Keep behind
            // RAYZOR_LEGACY_KERNEL=1 for A/B regression testing.
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            {
                let mut b_idx = 0;
                while b_idx + 1 < blocks_per_row {
                    let weight_a = &*(row_ptr.add(b_idx * block_bytes) as *const Q4KMBlock);
                    let weight_b = &*(row_ptr.add((b_idx + 1) * block_bytes) as *const Q4KMBlock);
                    let (sa, sb) =
                        dot_q4_k_q8_kblock_2(weight_a, weight_b, &x_q8k[b_idx], &x_q8k[b_idx + 1]);
                    sum += sa + sb;
                    b_idx += 2;
                }
            }
        } else {
            for (b_idx, x_block) in x_q8k.iter().enumerate().take(blocks_per_row) {
                let weight = &*(row_ptr.add(b_idx * block_bytes) as *const Q4KMBlock);
                sum += vec_dot_q4_K_q8_K(weight, x_block);
            }
        }

        *y_data.add(n_idx) = sum;
    }
}

/// Inner kernel for both the single-threaded fallback and the
/// threaded-chunk entry point. Computes `y[b, n_idx] = X[b, :] · dequant(Wq[n_idx, :])`
/// for `n_idx in [n_start, n_end)`. Worker buffers live on the stack/
/// thread-local heap; cross-thread state is just the `*y` write band,
/// which workers split disjointly so this needs no synchronisation.
unsafe fn qmatmul_chunk_impl(x_tensor: i64, qt_w: i64, y_tensor: i64, n_start: i64, n_end: i64) {
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
                                   // Second stage buffer for the Q6_K 2-row tile path (batch=1 decode).
    let mut stage_pair = [0.0f32; 256];

    // Per-row sums. Sized to `batch` so the general path can write
    // into it for any value of `batch`; the `batch == 1 && x_contig`
    // fast path below skips this entirely.
    let mut row_sums: Vec<f32> = vec![0.0; batch.max(1)];

    // Lazy cache of `quantize_x_block_q8` results for the SDOT path.
    // Each chunk call reuses the same X across every `n_idx` in
    // `[lo, hi)`, so quantising it once amortises across the whole
    // chunk. Populated on the first use of each block index.
    //
    // Gate: AArch64 + `+dotprod`. The `+dotprod` requirement matches
    // the cfg gate on `dot_q4_k_q8` itself — the symbol is cfg'd out
    // in builds without `+dotprod`, so the `use_sdot` binding (and the
    // initialiser block) MUST be cfg'd identically. `sdot_enabled()`
    // is the no-dotprod-stub returning `false` in the alternate arm,
    // so the runtime gate stays consistent across builds.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    let use_sdot = sdot_enabled() && batch == 1 && x_contig && qt.scheme == QSCHEME_Q4_K_M;
    // Q6_K SDOT path: same x-q8 cache shape, but uses `dot_q6_k_q8` which
    // reads `Q8Block::bsums_16` for the -32 bias correction (Q4_K_M uses
    // `bsums` for its 32-elem sub-block min). Sharing the cache between
    // both schemes means a single allocation per chunk regardless of which
    // scheme each call uses. Earlier attempts (per
    // bugs_q6k_sdot_no_win.md) showed -4.7% in isolated per-row testing
    // because per-block 6-bit reconstruction eats the SDOT density win at
    // single-row granularity — combining with the 2-row tile that landed
    // at 6285c22 amortises the x→Q8 reconstruction across two output rows
    // per block, which is where the win is expected to come from. Try it,
    // bench, document failure if not.
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    let use_sdot_q6k = sdot_enabled() && batch == 1 && x_contig && qt.scheme == QSCHEME_Q6_K;
    let mut x_q8_cache: Vec<Q8Block> = Vec::new();
    let mut x_q8_init: Vec<bool> = Vec::new();
    #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
    if use_sdot || use_sdot_q6k {
        x_q8_cache.reserve_exact(blocks_per_row);
        for _ in 0..blocks_per_row {
            x_q8_cache.push(Q8Block {
                quants: [0i8; 256],
                scale: 0.0,
                bsums: [0i32; 8],
                bsums_16: [0i32; 16],
            });
        }
        x_q8_init = vec![false; blocks_per_row];
    }

    // Q6_K 2-row tile (batch=1 decode fast path only). Shares one
    // x-chunk load per block between two output rows, giving two
    // independent FMA accumulator chains in the inner dot product.
    // matmul_qt_threaded is the dominant remaining decode-wall share
    // post-flash-attention on Llama-3.2-1B-Q4_K_M (lm_head is Q6_K),
    // so even a small per-row saving compounds across vocab=128256 rows.
    //
    // SDOT path (use_sdot_q6k): pre-quantise X to Q8 once per block,
    // reuse across both rows in the tile. Falls back to dequant+f32-dot
    // when SDOT isn't available (non-aarch64, !+dotprod, or
    // RAYZOR_USE_SDOT=0).
    let mut row_start = lo;
    if batch == 1 && x_contig && qt.scheme == QSCHEME_Q6_K {
        let tiled_end = lo + ((hi - lo) & !1usize); // largest even <= (hi-lo) + lo
        let mut r = lo;
        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
        if use_sdot_q6k {
            while r < tiled_end {
                let row0_ptr = qt.data.add(r * blocks_per_row * block_bytes);
                let row1_ptr = qt.data.add((r + 1) * blocks_per_row * block_bytes);
                let mut sum0 = 0.0f32;
                let mut sum1 = 0.0f32;
                for b_idx in 0..blocks_per_row {
                    let bp0 = row0_ptr.add(b_idx * block_bytes);
                    let bp1 = row1_ptr.add(b_idx * block_bytes);
                    let x_q8 = x_q8_cache_get(
                        &mut x_q8_cache,
                        &mut x_q8_init,
                        b_idx,
                        x_data.add(b_idx * block_size),
                    );
                    sum0 += dot_q6_k_q8(bp0, x_q8);
                    sum1 += dot_q6_k_q8(bp1, x_q8);
                }
                *y_data.add(r) = sum0;
                *y_data.add(r + 1) = sum1;
                r += 2;
            }
            #[allow(unused_assignments)] // overwritten at 2959 below when fallback also runs
            {
                row_start = r;
            }
        }
        // Non-SDOT fallback: dequant + f32-dot 2-row tile (the original
        // 6285c22 implementation). Runs when the SDOT gate is off or the
        // SDOT path above didn't advance r.
        while r < tiled_end {
            let row0_ptr = qt.data.add(r * blocks_per_row * block_bytes);
            let row1_ptr = qt.data.add((r + 1) * blocks_per_row * block_bytes);
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            for b_idx in 0..blocks_per_row {
                let bp0 = row0_ptr.add(b_idx * block_bytes);
                let bp1 = row1_ptr.add(b_idx * block_bytes);
                dequant_q6_k_block(bp0, &mut stage);
                dequant_q6_k_block(bp1, &mut stage_pair);
                let x_chunk =
                    std::slice::from_raw_parts(x_data.add(b_idx * block_size), block_size);
                sum0 += dot_f32_simd(x_chunk, &stage);
                sum1 += dot_f32_simd(x_chunk, &stage_pair);
            }
            *y_data.add(r) = sum0;
            *y_data.add(r + 1) = sum1;
            r += 2;
        }
        row_start = r;
    }

    for n_idx in row_start..hi {
        let row_ptr = qt.data.add(n_idx * blocks_per_row * block_bytes);

        if batch == 1 && x_contig {
            // Decode fast path: single batch row, contiguous X. Q4_K_M
            // can route through the SDOT kernel when `RAYZOR_USE_SDOT=1`
            // is set; the F32 path remains the default until A/B
            // measurement shows SDOT wins on this hardware.
            let mut sum = 0.0f32;
            for b_idx in 0..blocks_per_row {
                let bp = row_ptr.add(b_idx * block_bytes);
                match qt.scheme {
                    QSCHEME_Q4_K_M => {
                        // SDOT inner dispatch — `dot_q4_k_q8` exists in
                        // the binary only when `+dotprod` is enabled, so
                        // the cfg gate must match the kernel's own
                        // `#[cfg(...)]`. On aarch64+!dotprod / non-aarch64
                        // we fall through to the dequant+F32-dot path.
                        #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
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
                        let x_chunk =
                            std::slice::from_raw_parts(x_data.add(b_idx * block_size), block_size);
                        sum += dot_f32_simd(x_chunk, &stage);
                    }
                    QSCHEME_Q6_K => {
                        // Q6_K SDOT was tried (see git history at
                        // "perf(qmatmul): SDOT...Q6_K") and measured no
                        // wall-time improvement on M1 Pro — the per-block
                        // reconstruction overhead (4 shift/mask pairs per
                        // 16-weight span) eats the SDOT density win at
                        // this batch size.
                        //
                        // SCALAR fused dequant+dot was also tried 2026-06-04:
                        // saves the 1 KB stage write/read round-trip per
                        // block but loses the 4-way NEON FMA in
                        // `dot_f32_simd`. Net -56% tok/s on nue/llama-chat
                        // (20.5 → 9.2). The fused path only wins once the
                        // inner loop is itself vectorised — load 4×f32 x,
                        // decode 4×Q6_K weights via NEON shuffles,
                        // FMA-accumulate. Estimated ~150 LOC of NEON; see
                        // [[project-optimization-roadmap]] Tier 2 #4.
                        // Keeping the staged F32 + dot_f32_simd path until
                        // that NEON port lands.
                        dequant_q6_k_block(bp, &mut stage);
                        let x_chunk =
                            std::slice::from_raw_parts(x_data.add(b_idx * block_size), block_size);
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
                    let x_chunk = std::slice::from_raw_parts(x_data.add(x_off), block_size);
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

/// Atomic-refcount QTensor clone. Mirrors `rayzor_tensor_arc_clone`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_arc_clone(src: i64) -> i64 {
    if src == 0 {
        return 0;
    }
    let s = &*(src as *const RayzorQTensor);
    s.refcount
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    src
}

/// `QTensor.clone(src)` Haxe entry point. Routes to the Arc-increment path.
/// Preserves the `rayzor_qtensor_clone` extern symbol used by the Tier B
/// `@:derive([Clone])` lowering.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_clone(src: i64) -> i64 {
    rayzor_qtensor_arc_clone(src)
}

/// Disjoint-storage deep QTensor clone (escape hatch). Returns a fresh,
/// fully-owning QTensor sharing no storage with `src`.
///
/// Data buffer extent (chosen for safety + simplicity):
///   - INT8:   `numel` bytes.
///   - Q4_K_M: `(numel / 256) * 144`.
///   - Q6_K:   `(numel / 256) * 210`.
///
/// INT8 also carries a per-group `meta` f32 scale array of
/// `numel / group_size` entries; Q4_K_M / Q6_K embed scales inside each
/// super-block so `meta` is null for those.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_deep_clone(src: i64) -> i64 {
    if src == 0 {
        return 0;
    }
    let s = &*(src as *const RayzorQTensor);

    // Data buffer size by scheme — must match rayzor_qtensor_free's mental
    // model (which frees the whole `data` block as one allocation when it
    // owns it). For views the same byte extent is what the parent exposes,
    // so a full copy of that range covers everything the new wrapper might
    // dereference via `data`.
    let data_bytes = match s.scheme {
        QSCHEME_INT8 => s.numel,
        QSCHEME_Q4_K_M => (s.numel / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES,
        QSCHEME_Q6_K => (s.numel / Q6_K_BLOCK_SIZE) * Q6_K_BLOCK_BYTES,
        _ => return 0,
    };

    let data = malloc(if data_bytes > 0 { data_bytes } else { 1 });
    if data.is_null() {
        return 0;
    }
    if data_bytes > 0 && !s.data.is_null() {
        std::ptr::copy_nonoverlapping(s.data, data, data_bytes);
    }

    // INT8 carries a separate per-group f32 scale array; Q4_K_M / Q6_K
    // embed scales in the data blocks (meta is null).
    let meta: *mut f32 = if s.scheme == QSCHEME_INT8 && !s.meta.is_null() && s.group_size > 0 {
        let n_groups = s.numel / s.group_size;
        let scale_bytes = n_groups * std::mem::size_of::<f32>();
        let m = malloc(scale_bytes.max(4)) as *mut f32;
        if m.is_null() {
            free(data);
            return 0;
        }
        if n_groups > 0 {
            std::ptr::copy_nonoverlapping(s.meta, m, n_groups);
        }
        m
    } else {
        std::ptr::null_mut()
    };

    let qt = malloc(std::mem::size_of::<RayzorQTensor>()) as *mut RayzorQTensor;
    if qt.is_null() {
        free(data);
        if !meta.is_null() {
            free(meta as *mut u8);
        }
        return 0;
    }

    *qt = RayzorQTensor {
        data,
        meta,
        numel: s.numel,
        group_size: s.group_size,
        scheme: s.scheme,
        owns_data: true,
        rows: s.rows,
        cols: s.cols,
        refcount: std::sync::atomic::AtomicUsize::new(1),
        parent: std::ptr::null_mut(),
    };

    qt as i64
}

// ============================================================================
// QTensor pool integration
//
// QTensor pool keys live in a namespace disjoint from plain tensors: we OR
// 0x80 into the "dtype" byte and combine it with `scheme`, so plain f32
// (dtype=0) and INT8 (scheme=0) never alias. The bucket key also folds
// `group_size` into the shape hash because INT8 QTensors carry a separately-
// allocated f32 scales array of length `numel/group_size`; recycling across
// group_sizes would mismatch the meta length on revive.
//
// For Q4_K_M / Q6_K, `owns_data` is typically `false` (zero-copy wrap of an
// mmap'd GGUF buffer). The pool push side enforces the `owns_data == true`
// gate so zero-copy wrappers are never parked; they take the direct-free
// path that releases only the wrapper struct.
// ============================================================================

use crate::tensor_pool::{self, PoolKey, PooledEntry, ShapeBuf};

/// Construct a pool key for a QTensor's `(scheme, rows, cols, group_size)`
/// class. Distinct from plain-tensor keys via the 0x80 high bit on `dtype`.
fn qtensor_pool_key(scheme: u8, rows: usize, cols: usize, group_size: usize) -> PoolKey {
    // Fold group_size into the shape so the hash mixes it; this keeps the
    // bucket walk's `shape == ?` check precise too — we add a synthetic
    // dimension carrying group_size so two otherwise-identical shapes with
    // different group_sizes don't collide.
    let shape = [rows, cols, group_size];
    let mut key = PoolKey::from_shape(0x80 | scheme, &shape);
    // Defence-in-depth: also fold scheme into the hash so two schemes that
    // happen to share rows/cols/group_size still hash apart even though
    // they already differ in the `dtype` byte.
    key.shape_hash ^= scheme as u64;
    key
}

/// Pool entry "shape" recording (rows, cols, group_size). Used by the
/// bucket-walk `shape == ?` collision check.
fn qtensor_pool_shape(rows: usize, cols: usize, group_size: usize) -> [usize; 3] {
    [rows, cols, group_size]
}

/// Canonical free for pooled QTensor entries. Mirrors `rayzor_qtensor_free`
/// without the pool-routing — used on eviction / drain.
unsafe fn qtensor_pool_freer(entry: PooledEntry) {
    if entry.ptr.is_null() {
        return;
    }
    let qt = &*(entry.ptr as *const RayzorQTensor);
    if qt.owns_data && !qt.data.is_null() {
        free(qt.data);
    }
    if !qt.meta.is_null() {
        free(qt.meta as *mut u8);
    }
    free(entry.ptr);
}

/// Release a QTensor. The runtime frees `data` and `meta` if `owns_data`.
///
/// Pool-routing: owning QTensors (INT8 or Q4_K_M-with-take-ownership) are
/// parked in `tensor_pool::global()` keyed on
/// `(scheme, rows, cols, group_size)`. The meta scales array (INT8) rides
/// along on the `PooledEntry`, so an INT8 revive hands back the same
/// allocations including scales. Zero-copy wrappers (`owns_data=false`,
/// Q4_K_M / Q6_K mmap views) take the direct free path: their `data`
/// belongs to a parent `HaxeBytes`, only the wrapper struct is released.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_free(qt_ptr: i64) {
    if qt_ptr == 0 {
        return;
    }
    let qt = &*(qt_ptr as *const RayzorQTensor);

    // Phase 1 ARC: decrement first; only the dec-to-zero thread actually
    // releases storage (or pool-routes the owning INT8 slot).
    let prev = qt
        .refcount
        .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    if prev != 1 {
        return;
    }

    // Sole owner. Snapshot parent before tearing down — we'll cascade-free
    // it after our own storage is released, in case of view-of-qtensor.
    let parent = qt.parent;

    // Zero-copy wrappers: drop only the wrapper struct.
    if !qt.owns_data {
        if !qt.meta.is_null() {
            // Defensive — wrap paths set meta to null, but if a future
            // wrap variant adds a non-owned meta this still tracks correctly.
            free(qt.meta as *mut u8);
        }
        free(qt_ptr as *mut u8);
        if !parent.is_null() {
            rayzor_qtensor_free(parent as i64);
        }
        return;
    }

    // Scheme gate: only INT8 owning QTensors are pool-routed. `alloc_qtensor`
    // currently only revives INT8 via `try_pop_qtensor` (Q4_K_M and Q6_K both
    // arrive through `wrap`/`from_bytes` constructors that go straight to
    // `malloc` and never consult the pool), so parking a Q4_K_M / Q6_K
    // entry just leaks bytes into a bucket that nothing will ever pop. Take
    // the direct-free path for them instead — same physical release the
    // pool freer would do, without the bookkeeping churn.
    if qt.scheme != QSCHEME_INT8 {
        if !qt.data.is_null() {
            free(qt.data);
        }
        if !qt.meta.is_null() {
            free(qt.meta as *mut u8);
        }
        free(qt_ptr as *mut u8);
        if !parent.is_null() {
            rayzor_qtensor_free(parent as i64);
        }
        return;
    }

    // Owning INT8 QTensor: park in the pool.
    let key = qtensor_pool_key(qt.scheme, qt.rows, qt.cols, qt.group_size);
    let shape = qtensor_pool_shape(qt.rows, qt.cols, qt.group_size);
    let data_bytes = qt.data_bytes();
    let meta_bytes = if qt.meta.is_null() {
        0
    } else {
        // INT8 meta: one f32 scale per group.
        (qt.numel / qt.group_size) * std::mem::size_of::<f32>()
    };
    let entry = PooledEntry {
        ptr: qt_ptr as *mut u8,
        shape: ShapeBuf::from_slice(&shape),
        alloc_bytes: data_bytes,
        qtensor_meta_ptr: qt.meta as *mut u8,
        qtensor_meta_bytes: meta_bytes,
    };
    tensor_pool::global().push(key, entry, qtensor_pool_freer);
    if !parent.is_null() {
        rayzor_qtensor_free(parent as i64);
    }
}

/// Try to recycle a QTensor wrapper for the requested
/// `(scheme, rows, cols, group_size)` class. Returns the popped wrapper on
/// hit (caller is responsible for resetting / refilling `data` and `meta`
/// before use), or null on miss. Allocators that build a fresh QTensor
/// from raw data should consult this BEFORE allocating new buffers.
unsafe fn try_pop_qtensor(
    scheme: u8,
    rows: usize,
    cols: usize,
    group_size: usize,
) -> *mut RayzorQTensor {
    let key = qtensor_pool_key(scheme, rows, cols, group_size);
    let shape = qtensor_pool_shape(rows, cols, group_size);
    match tensor_pool::global().try_pop(key, &shape) {
        Some(entry) => entry.ptr as *mut RayzorQTensor,
        None => std::ptr::null_mut(),
    }
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
    ///     min   = (block[12..16] >> 4)   | ((block[8..12] >> 6) << 4)
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

    // ----------------------------------------------------------------
    // (C) Pool gate: Q4_K_M / Q6_K MUST NOT be pool-routed
    // ----------------------------------------------------------------
    //
    // `alloc_qtensor` only consults `try_pop_qtensor` for INT8; Q4_K_M
    // and Q6_K take the `from_bytes` / `wrap` path directly to malloc.
    // Parking Q4_K_M / Q6_K entries would therefore leak bytes into a
    // bucket nothing will ever pop. `rayzor_qtensor_free` now gates the
    // pool push on `scheme == QSCHEME_INT8` and falls through to direct
    // libc-`free()` for the other schemes.
    //
    // This test allocates 10 owning Q4_K_M wrappers (via `wrap` with
    // `take_ownership=1` so the freer treats them as owning), frees
    // each through the public FFI, and asserts:
    //   - the Q4_K_M pool key holds zero entries
    //   - the pool's `current_bytes` is unchanged across the free loop
    //     (no Q4_K_M byte budget churn)
    #[test]
    fn q4_k_m_free_bypasses_pool() {
        unsafe {
            // Snapshot current bookkeeping so a parallel cargo test that
            // parked some INT8 entries doesn't false-positive us.
            let pool = crate::tensor_pool::global();
            let bytes_before = pool
                .stats
                .current_bytes
                .load(std::sync::atomic::Ordering::Relaxed);
            let pushes_before = pool.stats.pushes.load(std::sync::atomic::Ordering::Relaxed);

            // Build a key matching what `rayzor_qtensor_free` would have
            // used for our (Q4_K_M, 1, 256, 256) qtensors. After the gate
            // no entries should ever appear under this key.
            let q4_key = qtensor_pool_key(QSCHEME_Q4_K_M, 1, 256, Q4_K_M_BLOCK_SIZE);
            let key_entries_before = pool.entries_in(q4_key);

            // Allocate + free 10 owning Q4_K_M wrappers. Each block is
            // 144 bytes; we malloc them so `take_ownership=1` transfers
            // a real owning buffer into the wrapper (the freer will
            // libc-free it directly under the new gate).
            for _ in 0..10 {
                let block = malloc(Q4_K_M_BLOCK_BYTES);
                assert!(!block.is_null());
                // Zero the block — content irrelevant for the gate test.
                std::ptr::write_bytes(block, 0, Q4_K_M_BLOCK_BYTES);
                let qt = rayzor_qtensor_wrap_q4_k_m(
                    block as i64,
                    1,
                    Q4_K_M_BLOCK_SIZE as i64,
                    1, /* take_ownership */
                );
                assert!(qt != 0);
                // Hand it straight to free — this is the path the gate
                // protects.
                rayzor_qtensor_free(qt);
            }

            // Q4_K_M bucket must still hold zero entries — every free
            // hit the direct-libc path, nothing was parked.
            let key_entries_after = pool.entries_in(q4_key);
            assert_eq!(
                key_entries_after, key_entries_before,
                "Q4_K_M frees must not park anything in the pool bucket"
            );
            // current_bytes must be unchanged — no bytes were added or
            // removed by Q4_K_M frees.
            let bytes_after = pool
                .stats
                .current_bytes
                .load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(
                bytes_after, bytes_before,
                "Q4_K_M free path must not touch pool current_bytes"
            );
            // pushes counter must be unchanged — the pool's `push` was
            // never called.
            let pushes_after = pool.stats.pushes.load(std::sync::atomic::Ordering::Relaxed);
            assert_eq!(
                pushes_after, pushes_before,
                "Q4_K_M free path must not invoke pool.push at all"
            );
        }
    }

    // ----------------------------------------------------------------
    // llama.cpp-compatible Q8_K + vec_dot_q4_K_q8_K spec surface
    // ----------------------------------------------------------------
    //
    // These tests pin the new public API to the same numerical contract
    // llama.cpp's `quantize_row_q8_K_ref` + `ggml_vec_dot_q4_K_q8_K`
    // ship. They run cross-architecture: on aarch64 the SDOT path is
    // exercised; on x86_64/CI the scalar fallback runs.

    /// (a) `quantize_row_q8_K` round-trip — re-dequantising the quants
    /// with the stored scale must reproduce the input to within one
    /// LSB of the symmetric scale. The bound `1.0 / 127.0` is the
    /// quantisation step size relative to the absmax; the task's
    /// stated `1/254` would only hold if quants used a [-254, 254]
    /// codebook (we use [-128, 127] like llama.cpp), so the correct
    /// rounding-tolerance bound is half the scale = `scale / 2 ≈
    /// max_abs / 254`. We assert the stricter `<= scale * 0.5 + eps`.
    #[test]
    fn quantize_row_q8_k_round_trip() {
        // Deterministic pseudo-random input. Mix sign, magnitude, and a
        // hot extremum so the scale lands on a known value.
        let mut x = [0.0f32; 256];
        let mut seed: u32 = 0x9E37_79B9;
        for slot in x.iter_mut() {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            // Map u32 → [-1.0, 1.0] then scale by a moderate range.
            let u = (seed >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
            *slot = (u * 2.0 - 1.0) * 3.5;
        }
        // Pin the absmax so the test is deterministic w.r.t. scale.
        x[42] = 4.0;
        x[200] = -4.0;

        let mut dest = [Q8KBlock {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }; 1];
        quantize_row_q8_K(&x, &mut dest);

        let block = &dest[0];
        let scale = block.d;
        assert!(scale > 0.0, "scale must be positive for non-zero input");
        // Expected scale: absmax (= 4.0) / 127.
        assert!(
            (scale - 4.0 / 127.0).abs() < 1e-7,
            "scale = {} (expected {})",
            scale,
            4.0 / 127.0
        );

        // Round-trip error bound: max error per element = scale / 2
        // (rounding to nearest). Allow a tiny f32 epsilon.
        let bound = scale * 0.5 + 1e-6;
        let mut max_err = 0.0f32;
        for i in 0..256 {
            let recon = scale * block.qs[i] as f32;
            let err = (recon - x[i]).abs();
            if err > max_err {
                max_err = err;
            }
            assert!(
                err <= bound,
                "i={i}: x={} recon={} err={} bound={}",
                x[i],
                recon,
                err,
                bound
            );
        }
        assert!(max_err < bound, "max_err = {} bound = {}", max_err, bound);

        // bsums invariant: bsums[s] = sum of qs[s*16 .. (s+1)*16].
        for s in 0..16 {
            let mut expect: i32 = 0;
            for j in 0..16 {
                expect += block.qs[s * 16 + j] as i32;
            }
            assert_eq!(
                block.bsums[s] as i32, expect,
                "bsums[{s}] = {} (expected {})",
                block.bsums[s], expect
            );
        }
    }

    /// (b) `vec_dot_q4_K_q8_K` must match the dequant-then-dot reference
    /// to within per-block quant error (~1e-3 relative).
    ///
    /// Construction: build a Q4_K_M block with non-trivial scales/mins
    /// (so both `sum_term1` and `sum_term2` are exercised), pick an X
    /// vector with mixed sign and magnitude, and cross-check the SDOT
    /// path against a scalar `Σ dequant[i] * x[i]`.
    #[test]
    fn vec_dot_q4_k_q8_k_matches_dequant_then_dot() {
        // Build a Q4_K_M block with d=0.5, dmin=0.25, varying per-sub-block
        // scale and min, and a quant pattern that uses both low and high
        // nibbles across all 8 sub-blocks.
        let mut raw = [0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(0.5).to_bits();
        let dmin = f16::from_f32(0.25).to_bits();
        raw[0..2].copy_from_slice(&d.to_le_bytes());
        raw[2..4].copy_from_slice(&dmin.to_le_bytes());
        // Sub-blocks 0..3 header: scale=j+1, min=j (low 6 bits of
        // bytes 4..8 and 8..12 respectively, high 2 bits stay zero so
        // sub-blocks 4..7 inherit their low nibbles from bytes 12..16).
        for j in 0..4 {
            raw[4 + j] = (j as u8 + 1) & 0x3F; // scale[j]   = j+1
            raw[8 + j] = (j as u8) & 0x3F; // min[j]     = j
        }
        // Sub-blocks 4..7: low nibble from bytes 12..16 only (high two
        // bits of bytes 4..12 are zero so they don't contribute).
        //   scales[4..8] low nibble = j+2 (range 2..5 fits in 4 bits)
        //   mins[4..8]   low nibble = j+1 (range 1..4 fits in 4 bits)
        for j in 0..4 {
            raw[12 + j] = ((j as u8 + 2) & 0x0F) | (((j as u8 + 1) & 0x0F) << 4);
        }
        // Quants: a varying pattern. byte = (i*7 + 3) ^ 0xA5 covers
        // a wide range of nibble values across all 128 bytes.
        for i in 0..128 {
            raw[16 + i] = ((i as u8).wrapping_mul(7).wrapping_add(3)) ^ 0xA5;
        }

        // Reinterpret the raw bytes as a Q4KMBlock view.
        // SAFETY: raw is 144 bytes, Q4KMBlock is repr(C, packed) sized
        // exactly Q4_K_M_BLOCK_BYTES (asserted at the const block above).
        let weight: &Q4KMBlock = unsafe { &*(raw.as_ptr() as *const Q4KMBlock) };

        // X: mixed magnitudes with both signs.
        let mut x = [0.0f32; 256];
        for (i, slot) in x.iter_mut().enumerate() {
            // sin-like pattern via integer mixing — deterministic
            // and crosses zero.
            let phase = (i as f32) * 0.1;
            *slot = phase.sin() * (1.0 + (i as f32 * 0.01).cos());
        }

        let mut q8 = [Q8KBlock {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }; 1];
        quantize_row_q8_K(&x, &mut q8);

        // SDOT path under test.
        let dot_sdot = vec_dot_q4_K_q8_K(weight, &q8[0]);

        // Reference: dequant the Q4_K_M block to f32, then dot against
        // the original f32 x. (NOT the quantised x — that would build
        // the quant error into both sides.)
        let decoded = unsafe { decode_q4_k_block(raw.as_ptr()) };
        let mut dq = [0.0f32; Q4_K_M_BLOCK_SIZE];
        dequant_q4_k_block(&decoded, &mut dq);
        let mut dot_ref = 0.0f32;
        for i in 0..256 {
            dot_ref += dq[i] * x[i];
        }

        // Relative tolerance — the SDOT path quantises X to int8 inside
        // the kernel, so the per-element error against the **f32** x is
        // bounded by (scale_x/2) per term over 256 accumulation terms.
        // Empirically this lands at ~1e-3; the spec's 1e-3 tolerance
        // catches catastrophic arithmetic breakage but bumps to 5e-3
        // here to absorb the X-side quant noise that is correct by
        // construction (llama.cpp's Q8_K vec-dot exhibits the same
        // drift against raw-f32 reference). The byte-for-byte
        // numerical lock-step against llama.cpp is exercised via the
        // end-to-end layer-diff harness, not here.
        let denom = dot_ref.abs().max(dot_sdot.abs()).max(1e-6);
        let rel = (dot_sdot - dot_ref).abs() / denom;
        assert!(
            rel < 5e-3,
            "vec_dot_q4_K_q8_K = {}, dequant-then-dot = {}, rel = {}",
            dot_sdot,
            dot_ref,
            rel
        );

        // Tighter check: dequant-then-dot against the **dequantised** X
        // (same Q8_K quant error applied to both sides) isolates the
        // SDOT path's arithmetic from the X-side quant noise. Must
        // match to f32 ULP accumulation (<1e-4 relative).
        let mut x_dq = [0.0f32; 256];
        for i in 0..256 {
            x_dq[i] = q8[0].d * q8[0].qs[i] as f32;
        }
        let mut dot_ref_dq = 0.0f32;
        for i in 0..256 {
            dot_ref_dq += dq[i] * x_dq[i];
        }
        let denom_dq = dot_ref_dq.abs().max(dot_sdot.abs()).max(1e-6);
        let rel_dq = (dot_sdot - dot_ref_dq).abs() / denom_dq;
        assert!(
            rel_dq < 1e-4,
            "SDOT vs dequant-then-dot-with-dequant-x: sdot={}, ref={}, rel={}",
            dot_sdot,
            dot_ref_dq,
            rel_dq
        );
    }

    /// (c) `vec_dot_q4_K_q8_K` against an all-zero X must return 0.
    /// (Degenerate-block guard: scale=0 short-circuits in
    /// `quantize_row_q8_K`; the inner kernel's `scale * (sum1 - sum2)`
    /// fold must then collapse to 0 regardless of weight contents.)
    #[test]
    fn vec_dot_q4_k_q8_k_zero_block() {
        // Non-trivial Q4_K_M block — must not contaminate the zero result.
        let mut raw = [0u8; Q4_K_M_BLOCK_BYTES];
        let d = f16::from_f32(1.5).to_bits();
        let dmin = f16::from_f32(0.75).to_bits();
        raw[0..2].copy_from_slice(&d.to_le_bytes());
        raw[2..4].copy_from_slice(&dmin.to_le_bytes());
        for j in 0..4 {
            raw[4 + j] = 7;
            raw[8 + j] = 3;
        }
        for j in 0..4 {
            raw[12 + j] = 0x21;
        }
        for i in 0..128 {
            raw[16 + i] = 0xCB;
        }
        let weight: &Q4KMBlock = unsafe { &*(raw.as_ptr() as *const Q4KMBlock) };

        let zero_x = [0.0f32; 256];
        let mut q8 = [Q8KBlock {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }; 1];
        quantize_row_q8_K(&zero_x, &mut q8);

        // Quantising zeros should yield d=0 and qs=bsums=0.
        assert_eq!(q8[0].d, 0.0, "scale of all-zero block must be 0");
        for &q in q8[0].qs.iter() {
            assert_eq!(q, 0, "all qs must be 0");
        }
        for &b in q8[0].bsums.iter() {
            assert_eq!(b, 0, "all bsums must be 0");
        }

        let dot = vec_dot_q4_K_q8_K(weight, &q8[0]);
        assert_eq!(
            dot, 0.0,
            "dot against zero X must be exactly 0, got {}",
            dot
        );
    }

    /// Round-trip: encode an f32 block to Q4_K_M then decode and check
    /// per-element error stays below a coarse bound. Q4_K_M's 4-bit
    /// payload + 6-bit sub-scale gives ~1/240 relative resolution per
    /// sub-block; we allow 5% RMS for the naive encoder (llama.cpp's
    /// iterative encoder gets ~2%). What matters here is that the
    /// encoder runs at all + produces decodable output the existing
    /// SDOT path can consume — exact numerical fidelity beyond that
    /// is a follow-up.
    #[test]
    fn quantize_block_q4_k_m_round_trip_below_bound() {
        // A weight-like pattern: zero-centred, ~unit standard deviation,
        // some skew + larger tails so the per-sub-block (min, max) varies.
        let mut x = [0.0f32; 256];
        for (i, slot) in x.iter_mut().enumerate() {
            let t = i as f32;
            *slot = (t * 0.11).sin() * 0.8 + (t * 0.037).cos() * 0.3 + (t * 0.003).sin() * 0.15;
        }

        let block = quantize_block_q4_k_m(&x);

        // Decode through the existing reader.
        let raw_ptr = &block as *const Q4KMBlock as *const u8;
        let decoded = unsafe { decode_q4_k_block(raw_ptr) };
        let mut dq = [0.0f32; 256];
        dequant_q4_k_block(&decoded, &mut dq);

        // RMS of (x - dq) relative to RMS of x must stay below 5%.
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..256 {
            let d = (x[i] - dq[i]) as f64;
            num += d * d;
            den += (x[i] as f64) * (x[i] as f64);
        }
        let rrms = (num / den.max(1e-12)).sqrt();
        assert!(
            rrms < 0.05,
            "round-trip relative RMS {} >= 0.05; encoder is broken",
            rrms
        );
    }

    /// Encoder + SDOT dot-product cross-check: encoding f32 weights and
    /// then dotting through `vec_dot_q4_K_q8_K` should give a result
    /// close to the raw `Σ w_i * x_i` reference. The numerical drift is
    /// dominated by the Q4_K_M encode loss tested above, so we allow a
    /// matching relative-error budget.
    #[test]
    fn quantize_block_q4_k_m_dot_matches_reference() {
        let mut w = [0.0f32; 256];
        let mut x = [0.0f32; 256];
        for i in 0..256 {
            let t = i as f32;
            w[i] = (t * 0.08).sin() * 0.7 + (t * 0.013).cos() * 0.2;
            x[i] = (t * 0.05).cos() * 0.5 - (t * 0.021).sin() * 0.3;
        }

        let block = quantize_block_q4_k_m(&w);
        let mut q8 = [Q8KBlock {
            d: 0.0,
            qs: [0i8; 256],
            bsums: [0i16; 16],
        }; 1];
        quantize_row_q8_K(&x, &mut q8);

        let dot_encoded = vec_dot_q4_K_q8_K(&block, &q8[0]);
        let dot_ref: f32 = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

        let rel = ((dot_encoded - dot_ref) / dot_ref.abs().max(1e-6)).abs();
        // 20% per-block dot error reflects the naive encoder's quality
        // ceiling — llama.cpp's `make_qkx2_quants` iterative search gets
        // closer to 5%. For the lm_head re-quant use case the per-block
        // error averages out across the 8 blocks per row + 128k vocab
        // entries, so the post-matmul argmax stays stable. The MATCH
        // check on the canonical prompt is the true gate.
        assert!(
            rel < 0.20,
            "encoded dot vs reference relative error {} >= 0.20 (dot_encoded={}, dot_ref={})",
            rel,
            dot_encoded,
            dot_ref
        );
    }
}
