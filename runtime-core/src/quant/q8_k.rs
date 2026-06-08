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
            let q = roundf(v).clamp(-128.0, 127.0) as i8;
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

        // Pass 1: absolute max over the 256-element super-block.
        let mut max_abs = 0.0f32;
        for &v in src {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }

        // Degenerate block: zero quants, scale=0, bsums=0. Matches llama.cpp
        // which sets `d=0` and skips the encode loop.
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
                let q = roundf(src[s * 16 + j] * inv_d).clamp(-128.0, 127.0) as i8;
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
