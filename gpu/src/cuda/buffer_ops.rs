//! CUDA buffer operations — GPU memory allocation and data transfer.
//!
//! Uses CUDA Driver API: `cuMemAlloc`, `cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemFree`.

use super::device_init::{CUresult, CudaContext, CUDA_SUCCESS};

/// CUDA device pointer (opaque GPU address).
pub type CUdeviceptr = u64;

// CUDA Driver API — memory management
extern "C" {
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const u8, bytecount: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut u8, src: CUdeviceptr, bytecount: usize) -> CUresult;
    fn cuMemsetD8_v2(dptr: CUdeviceptr, value: u8, count: usize) -> CUresult;
}

/// CUDA-specific GPU buffer wrapping a device pointer.
pub struct CudaBuffer {
    pub(crate) device_ptr: CUdeviceptr,
    pub(crate) byte_size: usize,
}

impl CudaBuffer {
    /// Create a CUDA buffer by copying data from a CPU pointer.
    /// # Safety
    /// `data` must point to at least `byte_size` readable bytes.
    pub unsafe fn from_data(_ctx: &CudaContext, data: *const u8, byte_size: usize) -> Option<Self> {
        if data.is_null() || byte_size == 0 {
            return None;
        }

        unsafe {
            let mut dptr: CUdeviceptr = 0;
            if cuMemAlloc_v2(&mut dptr, byte_size) != CUDA_SUCCESS {
                return None;
            }

            if cuMemcpyHtoD_v2(dptr, data, byte_size) != CUDA_SUCCESS {
                cuMemFree_v2(dptr);
                return None;
            }

            Some(CudaBuffer {
                device_ptr: dptr,
                byte_size,
            })
        }
    }

    /// Allocate an empty (zeroed) CUDA buffer of the given size.
    pub fn allocate(_ctx: &CudaContext, byte_size: usize) -> Option<Self> {
        if byte_size == 0 {
            return None;
        }

        unsafe {
            let mut dptr: CUdeviceptr = 0;
            if cuMemAlloc_v2(&mut dptr, byte_size) != CUDA_SUCCESS {
                return None;
            }

            // Zero-initialize
            if cuMemsetD8_v2(dptr, 0, byte_size) != CUDA_SUCCESS {
                cuMemFree_v2(dptr);
                return None;
            }

            Some(CudaBuffer {
                device_ptr: dptr,
                byte_size,
            })
        }
    }

    /// Read the buffer contents back to CPU memory.
    pub fn read_to_vec(&self, byte_size: usize) -> Option<Vec<u8>> {
        let read_size = byte_size.min(self.byte_size);
        if read_size == 0 {
            return None;
        }

        let mut data = vec![0u8; read_size];
        unsafe {
            if cuMemcpyDtoH_v2(data.as_mut_ptr(), self.device_ptr, read_size) != CUDA_SUCCESS {
                return None;
            }
        }
        Some(data)
    }

    /// Get the byte size of the buffer.
    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// Get the raw CUDA device pointer.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Create a CUDA buffer from a single value.
    pub fn from_value<T: Copy>(ctx: &CudaContext, value: &T) -> Option<Self> {
        let bytes = std::mem::size_of::<T>();
        // Safety: value is a valid reference, pointer and size are correct.
        unsafe { Self::from_data(ctx, value as *const T as *const u8, bytes) }
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.device_ptr != 0 {
            unsafe {
                cuMemFree_v2(self.device_ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_buffer_roundtrip() {
        if !CudaContext::is_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let ctx = CudaContext::new().unwrap();

        // Upload data
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let byte_size = data.len() * std::mem::size_of::<f32>();
        let buf = CudaBuffer::from_data(&ctx, data.as_ptr() as *const u8, byte_size)
            .expect("failed to create buffer");

        assert_eq!(buf.byte_size(), byte_size);

        // Read back
        let readback = buf.read_to_vec(byte_size).expect("failed to read back");
        let result: &[f32] =
            unsafe { std::slice::from_raw_parts(readback.as_ptr() as *const f32, data.len()) };

        for (i, (got, expected)) in result.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "mismatch at {i}: got {got}, expected {expected}"
            );
        }
    }
}
