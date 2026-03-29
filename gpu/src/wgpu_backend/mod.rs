//! WebGPU backend for GPU compute (cross-platform via wgpu)

// Compute backend modules — native only (depend on device_init::WgpuContext)
#[cfg(feature = "native")]
pub mod buffer_ops;
#[cfg(feature = "native")]
pub mod compile;
#[cfg(feature = "native")]
pub mod device_init;
#[cfg(feature = "native")]
pub mod dispatch;
