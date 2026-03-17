//! CUDA device initialization via the CUDA Driver API.
//!
//! Uses `cuInit`, `cuDeviceGet`, `cuCtxCreate` to set up a CUDA context.
//! All calls go through raw FFI bindings — no Rust CUDA wrapper crate needed.

use std::ffi::c_int;
use std::ptr;

// CUDA Driver API result type
pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut std::ffi::c_void;

pub const CUDA_SUCCESS: CUresult = 0;

// CUDA Driver API FFI bindings
extern "C" {
    pub fn cuInit(flags: u32) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxGetCurrent(ctx: *mut CUcontext) -> CUresult;
    pub fn cuDeviceGetName(name: *mut u8, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
}

/// CUDA device attribute: max threads per block
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
/// CUDA device attribute: max shared memory per block (bytes)
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
/// CUDA device attribute: warp size
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
/// CUDA device attribute: compute capability major
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
/// CUDA device attribute: compute capability minor
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// CUDA-specific GPU context wrapping a CUdevice + CUcontext.
pub struct CudaContext {
    pub device: CUdevice,
    pub context: CUcontext,
    pub max_threads_per_block: usize,
    pub shared_memory_per_block: usize,
    pub warp_size: usize,
}

impl CudaContext {
    /// Create a new CUDA context on device 0 (or first available).
    pub fn new() -> Option<Self> {
        unsafe {
            if cuInit(0) != CUDA_SUCCESS {
                return None;
            }

            let mut count: c_int = 0;
            if cuDeviceGetCount(&mut count) != CUDA_SUCCESS || count == 0 {
                return None;
            }

            let mut device: CUdevice = 0;
            if cuDeviceGet(&mut device, 0) != CUDA_SUCCESS {
                return None;
            }

            let mut context: CUcontext = ptr::null_mut();
            // flags=0 → default scheduling
            if cuCtxCreate_v2(&mut context, 0, device) != CUDA_SUCCESS {
                return None;
            }

            let max_threads =
                get_device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                    .unwrap_or(1024) as usize;
            let shared_mem =
                get_device_attribute(device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
                    .unwrap_or(49152) as usize;
            let warp_size =
                get_device_attribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrap_or(32) as usize;

            Some(CudaContext {
                device,
                context,
                max_threads_per_block: max_threads,
                shared_memory_per_block: shared_mem,
                warp_size,
            })
        }
    }

    /// Check if CUDA is available on this system.
    pub fn is_available() -> bool {
        unsafe {
            if cuInit(0) != CUDA_SUCCESS {
                return false;
            }
            let mut count: c_int = 0;
            cuDeviceGetCount(&mut count) == CUDA_SUCCESS && count > 0
        }
    }

    /// Get the device name as a string.
    pub fn device_name(&self) -> String {
        let mut name = [0u8; 256];
        unsafe {
            if cuDeviceGetName(name.as_mut_ptr(), 256, self.device) == CUDA_SUCCESS {
                let end = name.iter().position(|&b| b == 0).unwrap_or(256);
                String::from_utf8_lossy(&name[..end]).to_string()
            } else {
                "unknown".to_string()
            }
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if !self.context.is_null() {
            unsafe {
                cuCtxDestroy_v2(self.context);
            }
        }
    }
}

/// Helper: query a device attribute.
unsafe fn get_device_attribute(device: CUdevice, attrib: c_int) -> Option<c_int> {
    let mut value: c_int = 0;
    if cuDeviceGetAttribute(&mut value, attrib, device) == CUDA_SUCCESS {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_creation() {
        let available = CudaContext::is_available();
        println!("CUDA available: {}", available);
        if available {
            let ctx = CudaContext::new().expect("Failed to create CUDA context");
            println!(
                "CUDA device: {}, max_threads: {}, shared_mem: {}, warp_size: {}",
                ctx.device_name(),
                ctx.max_threads_per_block,
                ctx.shared_memory_per_block,
                ctx.warp_size
            );
        }
    }
}
