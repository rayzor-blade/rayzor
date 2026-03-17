//! Backend abstraction — thin enum dispatch layer over Metal and wgpu.
//!
//! `NativeBuffer`, `NativeContext`, and `NativeCompiledKernel` wrap the
//! backend-specific types. When only one feature is enabled, dead code
//! elimination removes unreachable arms (zero overhead).

//!
//! Each enum has an `Unavailable` variant so the code compiles even when
//! no backend feature is enabled.

#[cfg(feature = "metal-backend")]
use crate::metal::{buffer_ops::MetalBuffer, compile::CompiledKernel, device_init::MetalContext};

#[cfg(feature = "webgpu-backend")]
use crate::wgpu_backend::{
    buffer_ops::WgpuBuffer, compile::WgpuCompiledKernel, device_init::WgpuContext,
};

#[cfg(feature = "cuda-backend")]
use crate::cuda::{buffer_ops::CudaBuffer, compile::CudaCompiledKernel, device_init::CudaContext};

// ---------------------------------------------------------------------------
// NativeContext
// ---------------------------------------------------------------------------

pub enum NativeContext {
    #[cfg(feature = "metal-backend")]
    Metal(MetalContext),
    #[cfg(feature = "webgpu-backend")]
    Wgpu(WgpuContext),
    #[cfg(feature = "cuda-backend")]
    Cuda(CudaContext),
    /// Placeholder when no backend is enabled — never constructed at runtime.
    #[allow(dead_code)]
    Unavailable,
}

#[allow(unused_variables)]
impl NativeContext {
    /// Create a new GPU context using the best available backend.
    pub fn new() -> Option<Self> {
        #[cfg(feature = "metal-backend")]
        {
            if let Some(ctx) = MetalContext::new() {
                return Some(NativeContext::Metal(ctx));
            }
        }
        #[cfg(feature = "webgpu-backend")]
        {
            if let Some(ctx) = WgpuContext::new() {
                return Some(NativeContext::Wgpu(ctx));
            }
        }
        #[cfg(feature = "cuda-backend")]
        {
            if let Some(ctx) = CudaContext::new() {
                return Some(NativeContext::Cuda(ctx));
            }
        }
        None
    }

    /// Check if any GPU backend is available.
    pub fn is_available() -> bool {
        #[cfg(feature = "metal-backend")]
        {
            if MetalContext::is_available() {
                return true;
            }
        }
        #[cfg(feature = "webgpu-backend")]
        {
            if WgpuContext::is_available() {
                return true;
            }
        }
        #[cfg(feature = "cuda-backend")]
        {
            if CudaContext::is_available() {
                return true;
            }
        }
        false
    }

    /// Allocate an empty GPU buffer of the given byte size.
    pub fn allocate_buffer(&self, byte_size: usize) -> Option<NativeBuffer> {
        match self {
            #[cfg(feature = "metal-backend")]
            NativeContext::Metal(ctx) => {
                MetalBuffer::allocate(ctx, byte_size).map(NativeBuffer::Metal)
            }
            #[cfg(feature = "webgpu-backend")]
            NativeContext::Wgpu(ctx) => {
                WgpuBuffer::allocate(ctx, byte_size).map(NativeBuffer::Wgpu)
            }
            #[cfg(feature = "cuda-backend")]
            NativeContext::Cuda(ctx) => {
                CudaBuffer::allocate(ctx, byte_size).map(NativeBuffer::Cuda)
            }
            NativeContext::Unavailable => None,
        }
    }

    /// Create a GPU buffer by copying data from a CPU pointer.
    ///
    /// # Safety
    /// `data` must point to at least `byte_size` readable bytes.
    pub unsafe fn buffer_from_data(
        &self,
        data: *const u8,
        byte_size: usize,
    ) -> Option<NativeBuffer> {
        match self {
            #[cfg(feature = "metal-backend")]
            NativeContext::Metal(ctx) => {
                MetalBuffer::from_data(ctx, data, byte_size).map(NativeBuffer::Metal)
            }
            #[cfg(feature = "webgpu-backend")]
            NativeContext::Wgpu(ctx) => {
                WgpuBuffer::from_data(ctx, data, byte_size).map(NativeBuffer::Wgpu)
            }
            #[cfg(feature = "cuda-backend")]
            NativeContext::Cuda(ctx) => {
                CudaBuffer::from_data(ctx, data, byte_size).map(NativeBuffer::Cuda)
            }
            NativeContext::Unavailable => None,
        }
    }

    /// Create a GPU buffer from a single value (e.g., a u32 for numel).
    pub fn buffer_from_value<T: Copy>(&self, value: &T) -> Option<NativeBuffer> {
        let bytes = std::mem::size_of::<T>();
        // Safety: value is a valid reference, so the pointer and size are correct.
        unsafe { self.buffer_from_data(value as *const T as *const u8, bytes) }
    }
}

// ---------------------------------------------------------------------------
// NativeBuffer
// ---------------------------------------------------------------------------

pub enum NativeBuffer {
    #[cfg(feature = "metal-backend")]
    Metal(MetalBuffer),
    #[cfg(feature = "webgpu-backend")]
    Wgpu(WgpuBuffer),
    #[cfg(feature = "cuda-backend")]
    Cuda(CudaBuffer),
    #[allow(dead_code)]
    Unavailable,
}

#[allow(unused_variables)]
impl NativeBuffer {
    /// Read the buffer contents back to CPU memory.
    pub fn read_bytes(&self, byte_size: usize) -> Option<Vec<u8>> {
        match self {
            #[cfg(feature = "metal-backend")]
            NativeBuffer::Metal(buf) => {
                let ptr = buf.contents();
                if ptr.is_null() {
                    return None;
                }
                let mut data = vec![0u8; byte_size];
                unsafe {
                    std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), byte_size);
                }
                Some(data)
            }
            #[cfg(feature = "webgpu-backend")]
            NativeBuffer::Wgpu(buf) => buf.read_to_vec(byte_size),
            #[cfg(feature = "cuda-backend")]
            NativeBuffer::Cuda(buf) => buf.read_to_vec(byte_size),
            NativeBuffer::Unavailable => None,
        }
    }

    /// Get a raw CPU-accessible pointer to the buffer contents (Metal only).
    /// For wgpu/CUDA, this is not directly supported — use `read_bytes()` instead.
    pub fn contents_ptr(&self) -> *mut u8 {
        match self {
            #[cfg(feature = "metal-backend")]
            NativeBuffer::Metal(buf) => buf.contents(),
            #[cfg(feature = "webgpu-backend")]
            NativeBuffer::Wgpu(_) => std::ptr::null_mut(),
            #[cfg(feature = "cuda-backend")]
            NativeBuffer::Cuda(_) => std::ptr::null_mut(),
            NativeBuffer::Unavailable => std::ptr::null_mut(),
        }
    }

    /// Get the byte size of the buffer.
    pub fn byte_size(&self) -> usize {
        match self {
            #[cfg(feature = "metal-backend")]
            NativeBuffer::Metal(buf) => buf.byte_size(),
            #[cfg(feature = "webgpu-backend")]
            NativeBuffer::Wgpu(buf) => buf.byte_size(),
            #[cfg(feature = "cuda-backend")]
            NativeBuffer::Cuda(buf) => buf.byte_size(),
            NativeBuffer::Unavailable => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// NativeCompiledKernel
// ---------------------------------------------------------------------------

pub enum NativeCompiledKernel {
    #[cfg(feature = "metal-backend")]
    Metal(CompiledKernel),
    #[cfg(feature = "webgpu-backend")]
    Wgpu(WgpuCompiledKernel),
    #[cfg(feature = "cuda-backend")]
    Cuda(CudaCompiledKernel),
    #[allow(dead_code)]
    Unavailable,
}
