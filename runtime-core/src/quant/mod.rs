//! Quantisation kernels — pure-compute portion of the Q4_K_M / Q6_K / Q8_0 /
//! INT8 stack. The threaded matmul wrappers, FFI surface, and SDOT inner
//! kernels stay in `rayzor-runtime`; what lives here is byte-layout types,
//! block encode/decode/dequant, and dequant-fused matmul reference paths
//! that share the `runtime_core::simd::tensor_f32::axpy_slice` inner.
//!
//! See `docs/design/runtime_core_extraction.md` for the migration plan;
//! this is Step 3 (types + block ops; SDOT kernels land in Step 4).

pub mod int8;
pub mod q4_k_m;
pub mod q6_k;
pub mod q8_k;
pub mod types;

pub use types::{
    Q4KBlock, Q4KMBlock, Q8Block, Q8KBlock, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_SIZE,
    Q6_K_BLOCK_BYTES, Q6_K_BLOCK_SIZE, QSCHEME_INT8, QSCHEME_Q4_K_M, QSCHEME_Q6_K,
};
