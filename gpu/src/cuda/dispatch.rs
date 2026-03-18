//! CUDA compute kernel dispatch — launches kernels via cuLaunchKernel.

use super::buffer_ops::CudaBuffer;
use super::compile::CudaCompiledKernel;
use super::device_init::{CUDA_SUCCESS, CUresult, CudaContext};

type CUstream = *mut std::ffi::c_void;

// CUDA Driver API — kernel launch
extern "C" {
    fn cuLaunchKernel(
        f: super::compile::CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: CUstream,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CUresult;

    fn cuCtxSynchronize() -> CUresult;
}

/// Dispatch a 1D elementwise kernel.
///
/// For binary ops: `buffers` = [a, b, result, numel_buf]
/// For unary ops:  `buffers` = [a, result, numel_buf]
///
/// The kernel is dispatched with enough blocks to cover `numel` threads.
pub fn dispatch(
    _ctx: &CudaContext,
    kernel: &CudaCompiledKernel,
    buffers: &[&CudaBuffer],
    numel: usize,
) -> Result<(), String> {
    if numel == 0 {
        return Ok(());
    }

    let block_size = kernel.block_size as u32;
    let grid_size = (numel as u32).div_ceil(block_size);

    // Build kernel parameter array: each entry is a pointer to the device pointer value
    let mut device_ptrs: Vec<u64> = buffers.iter().map(|b| b.device_ptr()).collect();
    let numel_u32 = numel as u32;

    // Kernel params: pointers to each argument's storage
    let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(device_ptrs.len() + 1);
    for dptr in &mut device_ptrs {
        params.push(dptr as *mut u64 as *mut std::ffi::c_void);
    }
    // Add numel as the last parameter
    let numel_ptr = &numel_u32 as *const u32 as *mut std::ffi::c_void;
    params.push(numel_ptr);

    unsafe {
        let result = cuLaunchKernel(
            kernel.function,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,                    // shared memory
            std::ptr::null_mut(), // default stream
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );

        if result != CUDA_SUCCESS {
            return Err(format!("cuLaunchKernel failed: {result}"));
        }

        // Synchronize to ensure completion
        let sync = cuCtxSynchronize();
        if sync != CUDA_SUCCESS {
            return Err(format!("cuCtxSynchronize failed: {sync}"));
        }
    }

    Ok(())
}

/// Dispatch a 2D kernel (e.g., matmul with tiled shared memory).
///
/// `grid` and `block` are (x, y, z) dimensions.
/// `shared_mem_bytes` is the dynamic shared memory allocation.
pub fn dispatch_grid(
    _ctx: &CudaContext,
    kernel: &CudaCompiledKernel,
    buffers: &[&CudaBuffer],
    extra_params: &[u32],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
) -> Result<(), String> {
    // Build kernel parameter array
    let mut device_ptrs: Vec<u64> = buffers.iter().map(|b| b.device_ptr()).collect();
    let mut extra_values: Vec<u32> = extra_params.to_vec();

    let mut params: Vec<*mut std::ffi::c_void> = Vec::new();
    for dptr in &mut device_ptrs {
        params.push(dptr as *mut u64 as *mut std::ffi::c_void);
    }
    for val in &mut extra_values {
        params.push(val as *mut u32 as *mut std::ffi::c_void);
    }

    unsafe {
        let result = cuLaunchKernel(
            kernel.function,
            grid.0,
            grid.1,
            grid.2,
            block.0,
            block.1,
            block.2,
            shared_mem_bytes,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );

        if result != CUDA_SUCCESS {
            return Err(format!("cuLaunchKernel failed: {result}"));
        }

        let sync = cuCtxSynchronize();
        if sync != CUDA_SUCCESS {
            return Err(format!("cuCtxSynchronize failed: {sync}"));
        }
    }

    Ok(())
}

/// Dispatch a 1D reduction kernel with shared memory.
pub fn dispatch_reduction(
    _ctx: &CudaContext,
    kernel: &CudaCompiledKernel,
    input: &CudaBuffer,
    output: &CudaBuffer,
    numel: usize,
    workgroup_size: usize,
) -> Result<(), String> {
    let wg = workgroup_size as u32;
    let num_groups = (numel as u32).div_ceil(wg);
    let shared_bytes = wg * 4; // sizeof(float) per thread

    let mut input_ptr = input.device_ptr();
    let mut output_ptr = output.device_ptr();
    let numel_u32 = numel as u32;

    let mut params: Vec<*mut std::ffi::c_void> = vec![
        &mut input_ptr as *mut u64 as *mut std::ffi::c_void,
        &mut output_ptr as *mut u64 as *mut std::ffi::c_void,
        &numel_u32 as *const u32 as *mut std::ffi::c_void,
    ];

    unsafe {
        let result = cuLaunchKernel(
            kernel.function,
            num_groups,
            1,
            1,
            wg,
            1,
            1,
            shared_bytes,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        );

        if result != CUDA_SUCCESS {
            return Err(format!("cuLaunchKernel (reduction) failed: {result}"));
        }

        let sync = cuCtxSynchronize();
        if sync != CUDA_SUCCESS {
            return Err(format!("cuCtxSynchronize failed: {sync}"));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::compile;

    #[test]
    fn test_dispatch_add_f32() {
        if !CudaContext::is_available() {
            println!("CUDA not available, skipping");
            return;
        }

        let ctx = CudaContext::new().unwrap();

        let source = r#"
            extern "C" __global__ void test_add(
                const float* a,
                const float* b,
                float* result,
                unsigned int numel
            ) {
                unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
                if (id >= numel) return;
                result[id] = a[id] + b[id];
            }
        "#;

        let kernel = compile::compile_cuda(&ctx, source, "test_add").unwrap();

        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let byte_size = n * std::mem::size_of::<f32>();

        let a_buf = unsafe { CudaBuffer::from_data(&ctx, a_data.as_ptr() as *const u8, byte_size) }
            .unwrap();
        let b_buf = unsafe { CudaBuffer::from_data(&ctx, b_data.as_ptr() as *const u8, byte_size) }
            .unwrap();
        let result_buf = CudaBuffer::allocate(&ctx, byte_size).unwrap();

        dispatch(&ctx, &kernel, &[&a_buf, &b_buf, &result_buf], n).unwrap();

        let readback = result_buf.read_to_vec(byte_size).unwrap();
        let result: &[f32] =
            unsafe { std::slice::from_raw_parts(readback.as_ptr() as *const f32, n) };

        for (i, value) in result.iter().enumerate().take(n) {
            let expected = (i + i * 2) as f32;
            assert!(
                (*value - expected).abs() < 1e-6,
                "mismatch at {i}: expected {expected}, got {}",
                value
            );
        }
    }
}
