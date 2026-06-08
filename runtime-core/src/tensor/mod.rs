//! Pure-compute tensor kernels shared by `rayzor-runtime` (native) and
//! `rayzor-runtime-wasm`.
//!
//! Tensor lifetime (allocation via libc::malloc, refcount + pool integration,
//! and the FFI exports the Haxe JIT consumes) stays in `rayzor-runtime`; what
//! lives here is per-element compute that operates on raw pointers + length
//! contracts.

pub mod flash_attn;
pub mod topk;
