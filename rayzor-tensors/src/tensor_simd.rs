//! SIMD primitives for the f32 tensor hot paths.
//!
//! The implementations now live in the architecture-portable
//! `rayzor-runtime-core` crate so the WASM runtime can share them. This module
//! is a thin re-export shim so every existing `use crate::tensor_simd::*`
//! callsite inside `rayzor-runtime` keeps working unchanged.
//!
//! New kernel work should land directly in `rayzor-runtime-core::simd::tensor_f32`.
//! See `docs/design/runtime_core_extraction.md` for the migration plan.

pub use rayzor_runtime_core::simd::tensor_f32::*;
