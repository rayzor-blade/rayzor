//! NVRTC kernel compilation — CUDA C source → PTX → CUmodule → CUfunction.
//!
//! Uses NVRTC (NVIDIA Runtime Compilation) to compile CUDA C source at runtime,
//! then loads the resulting PTX into a CUmodule via the CUDA Driver API.

use std::ffi::{c_int, CStr, CString};
use std::ptr;

use super::device_init::{CUresult, CudaContext, CUDA_SUCCESS};

// CUDA Driver API types for modules/functions
pub type CUmodule = *mut std::ffi::c_void;
pub type CUfunction = *mut std::ffi::c_void;

// NVRTC types
pub type NvrtcProgram = *mut std::ffi::c_void;
pub type NvrtcResult = c_int;

pub const NVRTC_SUCCESS: NvrtcResult = 0;

// NVRTC FFI bindings
extern "C" {
    fn nvrtcCreateProgram(
        prog: *mut NvrtcProgram,
        src: *const u8,
        name: *const u8,
        num_headers: c_int,
        headers: *const *const u8,
        include_names: *const *const u8,
    ) -> NvrtcResult;

    fn nvrtcDestroyProgram(prog: *mut NvrtcProgram) -> NvrtcResult;

    fn nvrtcCompileProgram(
        prog: NvrtcProgram,
        num_options: c_int,
        options: *const *const u8,
    ) -> NvrtcResult;

    fn nvrtcGetPTXSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;

    fn nvrtcGetPTX(prog: NvrtcProgram, ptx: *mut u8) -> NvrtcResult;

    fn nvrtcGetProgramLogSize(prog: NvrtcProgram, size: *mut usize) -> NvrtcResult;

    fn nvrtcGetProgramLog(prog: NvrtcProgram, log: *mut u8) -> NvrtcResult;
}

// CUDA Driver API — module/function loading
extern "C" {
    fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const u8,
        num_options: u32,
        options: *const c_int,
        option_values: *const *mut std::ffi::c_void,
    ) -> CUresult;

    fn cuModuleGetFunction(func: *mut CUfunction, module: CUmodule, name: *const u8) -> CUresult;

    fn cuModuleUnload(module: CUmodule) -> CUresult;
}

/// A compiled CUDA kernel ready for dispatch.
pub struct CudaCompiledKernel {
    pub module: CUmodule,
    pub function: CUfunction,
    /// Suggested block size (threads per block).
    pub block_size: usize,
}

impl Drop for CudaCompiledKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                cuModuleUnload(self.module);
            }
        }
    }
}

/// Compile CUDA C source via NVRTC → PTX → CUmodule → CUfunction.
///
/// `fn_name` must match the `extern "C" __global__` kernel function name.
pub fn compile_cuda(
    ctx: &CudaContext,
    source: &str,
    fn_name: &str,
) -> Result<CudaCompiledKernel, String> {
    let ptx = nvrtc_compile(source, fn_name)?;
    load_ptx(ctx, &ptx, fn_name)
}

/// Step 1: NVRTC compile CUDA C → PTX.
fn nvrtc_compile(source: &str, name: &str) -> Result<Vec<u8>, String> {
    let src_c = CString::new(source).map_err(|e| format!("invalid source: {e}"))?;
    let name_c = CString::new(name).map_err(|e| format!("invalid name: {e}"))?;

    unsafe {
        let mut prog: NvrtcProgram = ptr::null_mut();

        let result = nvrtcCreateProgram(
            &mut prog,
            src_c.as_ptr() as *const u8,
            name_c.as_ptr() as *const u8,
            0,
            ptr::null(),
            ptr::null(),
        );
        if result != NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram failed: {result}"));
        }

        // Compile with default options (could add --gpu-architecture here)
        let compile_result = nvrtcCompileProgram(prog, 0, ptr::null());

        if compile_result != NVRTC_SUCCESS {
            // Get compilation log
            let log = get_nvrtc_log(prog);
            nvrtcDestroyProgram(&mut prog);
            return Err(format!(
                "NVRTC compilation failed ({compile_result}):\n{log}"
            ));
        }

        // Get PTX size and contents
        let mut ptx_size: usize = 0;
        if nvrtcGetPTXSize(prog, &mut ptx_size) != NVRTC_SUCCESS {
            nvrtcDestroyProgram(&mut prog);
            return Err("nvrtcGetPTXSize failed".to_string());
        }

        let mut ptx = vec![0u8; ptx_size];
        if nvrtcGetPTX(prog, ptx.as_mut_ptr()) != NVRTC_SUCCESS {
            nvrtcDestroyProgram(&mut prog);
            return Err("nvrtcGetPTX failed".to_string());
        }

        nvrtcDestroyProgram(&mut prog);
        Ok(ptx)
    }
}

/// Load PTX into a CUmodule and extract a CUfunction.
fn load_ptx(ctx: &CudaContext, ptx: &[u8], fn_name: &str) -> Result<CudaCompiledKernel, String> {
    let fn_name_c = CString::new(fn_name).map_err(|e| format!("invalid fn_name: {e}"))?;

    unsafe {
        let mut module: CUmodule = ptr::null_mut();
        let result = cuModuleLoadDataEx(&mut module, ptx.as_ptr(), 0, ptr::null(), ptr::null());
        if result != CUDA_SUCCESS {
            return Err(format!("cuModuleLoadDataEx failed: {result}"));
        }

        let mut function: CUfunction = ptr::null_mut();
        let result = cuModuleGetFunction(&mut function, module, fn_name_c.as_ptr() as *const u8);
        if result != CUDA_SUCCESS {
            cuModuleUnload(module);
            return Err(format!(
                "cuModuleGetFunction failed for '{}': {result}",
                fn_name
            ));
        }

        // Use 256 as default block size — commonly optimal for elementwise kernels
        let block_size = 256.min(ctx.max_threads_per_block);

        Ok(CudaCompiledKernel {
            module,
            function,
            block_size,
        })
    }
}

/// Get NVRTC compilation log for error messages.
unsafe fn get_nvrtc_log(prog: NvrtcProgram) -> String {
    let mut log_size: usize = 0;
    if nvrtcGetProgramLogSize(prog, &mut log_size) != NVRTC_SUCCESS || log_size == 0 {
        return "<no log>".to_string();
    }

    let mut log = vec![0u8; log_size];
    if nvrtcGetProgramLog(prog, log.as_mut_ptr()) != NVRTC_SUCCESS {
        return "<failed to get log>".to_string();
    }

    // NVRTC log is null-terminated
    CStr::from_ptr(log.as_ptr() as *const _)
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_cuda_kernel() {
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

        let kernel = compile_cuda(&ctx, source, "test_add");
        assert!(kernel.is_ok(), "compilation failed: {:?}", kernel.err());
        let kernel = kernel.unwrap();
        assert!(kernel.block_size > 0);
        println!("block_size: {}", kernel.block_size);
    }
}
