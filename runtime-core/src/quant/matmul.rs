//! Architecture-portable matmul building blocks shared by both the native
//! and WASM runtimes.
//!
//! Currently exposes:
//! - `dot_f32_simd` / `dot_f32_avx2_fma` — F32 horizontal dot product with
//!   NEON / AVX2-FMA / scalar paths; the per-core hot loop for the Q4_K_M /
//!   Q6_K Linear projections.
//! - `prepare_x_q8k_blocks` / `prepare_x_q8k_blocks_into` — F32 activation →
//!   `Vec<Q8KBlock>` for the SDOT chunk impl.
//!
//! Higher-level chunk impls (`qmatmul_chunk_impl_sdot_q4km`,
//! `qmatmul_chunk_impl`) currently take `i64` tensor handles and dereference
//! them into the native `RayzorTensor` / `RayzorQTensor` layouts; they stay
//! in `rayzor-runtime` until `Step 6` moves the tensor types into
//! `runtime-core` and the chunk impls can be retyped against shared structs.

use alloc::vec;
use alloc::vec::Vec;

use super::q8_k::quantize_row_q8_K;
use super::types::{Q8KBlock, Q4_K_M_BLOCK_SIZE};

/// Vectorized horizontal dot product `Σ a[i] * b[i]`. NEON on aarch64
/// (4×FMA-with-acc unroll); AVX2+FMA on x86_64 when feature-detected
/// (8×FMA-with-acc unroll); scalar fallback elsewhere.
///
/// This is the per-core hot loop for the Q4_K_M / Q6_K Linear projections —
/// `k = 2048` for the QKV / O projections, `k = 8192` for the FFN
/// gate/up/down. On M1 Pro the SIMD path turns the ~4 ms per dot product
/// into ~1 ms, which is the dominant CPU cost during decode.
#[inline]
pub fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is guaranteed on every aarch64 target; we index
        // strictly within `[0, n)` for both slices and bound-check the
        // unrolled tail.
        unsafe {
            use core::arch::aarch64::*;
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
        // Scalar fallback. LLVM autovectorises well-aligned slices here, so
        // this is the path on wasm and pre-AVX2 x86.
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// AVX2+FMA 8-wide horizontal dot product. Caller must have feature-detected
/// AVX2+FMA before invoking.
///
/// # Safety
/// `a` and `b` must each contain at least `n` live f32 elements. Calling on
/// a CPU without AVX2+FMA traps with SIGILL.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32_avx2_fma(a: &[f32], b: &[f32], n: usize) -> f32 {
    use core::arch::x86_64::*;
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

/// Pre-quantise a contiguous f32 X span of length `k` into `k / 256` `Q8KBlock`
/// super-blocks. Returns a freshly allocated `Vec<Q8KBlock>`; for hot-path
/// scratch reuse, see [`prepare_x_q8k_blocks_into`].
///
/// # Safety
/// `x_data` must be a valid f32 cursor with `k` contiguous elements; `k`
/// must be a multiple of `Q4_K_M_BLOCK_SIZE`.
#[inline]
#[allow(dead_code)] // allocating sibling of prepare_x_q8k_blocks_into; kept for one-shot callers
pub unsafe fn prepare_x_q8k_blocks(x_data: *const f32, k: usize) -> Vec<Q8KBlock> {
    debug_assert!(k.is_multiple_of(Q4_K_M_BLOCK_SIZE));
    let nb = k / Q4_K_M_BLOCK_SIZE;
    let x_slice = core::slice::from_raw_parts(x_data, k);
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

/// Same as [`prepare_x_q8k_blocks`] but writes into a caller-provided
/// `Vec<Q8KBlock>`, growing it in place if too small. Used by the threaded
/// matmul entry points to reuse a thread-local scratch buffer across calls
/// — for Llama-3.2-1B the per-call Vec is 8 × 292 = 2.3 KB, allocated 5+
/// times per layer × 16 layers per token. Hoisting to the per-thread
/// scratch saves ~80 allocations/token in steady-state decode and the
/// zero-init pass on the freshly allocated Vec (`quantize_row_q8_K`
/// overwrites every byte, so the `vec![]` macro's zero-init is pure waste).
///
/// # Safety
/// Same as [`prepare_x_q8k_blocks`].
pub unsafe fn prepare_x_q8k_blocks_into(x_data: *const f32, k: usize, dest: &mut Vec<Q8KBlock>) {
    debug_assert!(k.is_multiple_of(Q4_K_M_BLOCK_SIZE));
    let nb = k / Q4_K_M_BLOCK_SIZE;
    let x_slice = core::slice::from_raw_parts(x_data, k);
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
