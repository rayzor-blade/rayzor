//! Architecture-portable compute kernels shared by `rayzor-runtime` (native)
//! and `rayzor-runtime-wasm`.
//!
//! This crate is `no_std + alloc`-compatible. Everything here is pure compute
//! with no syscalls, no threading, no global mutable state, no I/O. The
//! per-target wrapper crates handle:
//! - `rayzor-runtime/` — allocation via `libc::malloc`, threading via the spin
//!   worker pool, profiling/heap-check guards, FFI exports to the Haxe JIT.
//! - `rayzor-runtime-wasm/` — allocation via `dlmalloc`, threading via
//!   wasi-threads or browser Web Workers, FFI exports to wasm32 ABI.
//!
//! Architecture-specific SIMD lives behind `cfg(target_arch = ...)` gates
//! with scalar fallbacks. Adding a new target means adding one cfg branch per
//! kernel, not a whole new file.
//!
//! See `docs/design/runtime_core_extraction.md` for the migration plan.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod floats;
pub mod gguf;
pub mod quant;
pub mod simd;
pub mod tensor;
