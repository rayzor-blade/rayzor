//! Per-q_head flash-attention decode inner kernel.
//!
//! Per the 7b72ba8 parallelisation pattern, the caller (in `rayzor-runtime`)
//! wraps the q_head loop in `worker_pool.parallel_rows` and calls this once
//! per q_head with its disjoint output band — no cross-worker sync needed.
//!
//! Generic over the exponential function so callers can pick their preferred
//! implementation:
//!
//!   - On `rayzor-runtime` (native macOS / Linux) callers pass `|x| x.exp()`,
//!     which lowers to the platform `libsystem_m` / `libm` exp; on macOS that
//!     routes through the Accelerate-tuned math library.
//!   - On `rayzor-runtime-wasm` callers pass `libm::expf`, the pure-Rust
//!     polynomial implementation that the wasm32 target ships with by
//!     default (since `f32::exp()` in `no_std` is unavailable).
//!
//! The generic parameter monomorphises + inlines per call site, so neither
//! side pays an indirect-call cost on the softmax hot loop.

use crate::simd::tensor_f32::{axpy_slice, dot_slice_f32};

/// Compute the attention output row for one query head.
///
/// Three-pass kernel:
///   1. score compute: `scores[l] = (Q[h] · K[l, kv_head, :]) * scale`
///   2. softmax (numerically stabilised by `max_score` subtraction),
///      using `exp_fn` to evaluate each shifted exponent
///   3. weighted sum: `out[h, :] = Σ_l softmax[l] * V[l, kv_head, :]`
///
/// # Safety
/// `q_data`, `k_data`, `v_data`, `out_data` must reference live storage with
/// enough room for the indexed accesses (one q row, `cache_len` k/v rows of
/// `head_dim` floats each, one output row). `scores` must hold at least
/// `cache_len` entries. `q_head / group` indexes the matching KV head.
///
/// `#[inline(always)]` so the monomorphised body folds into the caller across
/// crate boundaries — without it, thin-LTO occasionally chose not to inline
/// the closure-bearing generic call and the softmax loop took a small (~3%)
/// hit on M1 Pro decode.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub unsafe fn flash_attn_decode_one_qhead<F: Fn(f32) -> f32>(
    q_head: usize,
    group: usize,
    head_dim: usize,
    cache_len: usize,
    kv_row_stride: usize,
    scale_f32: f32,
    q_data: *const f32,
    k_data: *const f32,
    v_data: *const f32,
    out_data: *mut f32,
    scores: &mut [f32],
    exp_fn: F,
) {
    let kv_head = q_head / group;
    let q_row = core::slice::from_raw_parts(q_data.add(q_head * head_dim), head_dim);

    // First pass: scores[l] = (Q[h] · K[l, kv_head, :]) * scale.
    let mut max_score = f32::NEG_INFINITY;
    for l in 0..cache_len {
        let k_row = core::slice::from_raw_parts(
            k_data.add(l * kv_row_stride + kv_head * head_dim),
            head_dim,
        );
        let s = dot_slice_f32(q_row, k_row) * scale_f32;
        scores[l] = s;
        if s > max_score {
            max_score = s;
        }
    }

    // Second pass: softmax denominator, in the standard max-shifted form
    // (matches `Tensor.softmax`'s reduction order). `exp_fn` is monomorphised
    // per call site so the inner is byte-equal to the prior inline kernel
    // when the caller passes `|x| x.exp()`.
    let mut denom = 0.0f32;
    for l in 0..cache_len {
        let e = exp_fn(scores[l] - max_score);
        scores[l] = e;
        denom += e;
    }
    let inv_denom = 1.0 / denom;

    // Third pass: context[q_head, :] = Σ_l (softmax[l] * V[l, kv_head, :]).
    let out_row = core::slice::from_raw_parts_mut(out_data.add(q_head * head_dim), head_dim);
    out_row.fill(0.0);
    for l in 0..cache_len {
        let w = scores[l] * inv_denom;
        let v_row = core::slice::from_raw_parts(
            v_data.add(l * kv_row_stride + kv_head * head_dim),
            head_dim,
        );
        axpy_slice(out_row, w, v_row);
    }
}
