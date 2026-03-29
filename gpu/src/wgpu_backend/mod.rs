//! WebGPU backend for GPU compute (cross-platform via wgpu)

// Compute backend modules — core wgpu logic (portable to WASM)
pub mod buffer_ops;
pub mod compile;
pub mod device_init;
pub mod dispatch;
