//! Kernel cache — avoids recompiling the same kernel multiple times.
//!
//! Keyed by (KernelOp, dtype), since the same op+dtype always produces
//! identical shader source. The cache lives for the lifetime of the GpuContext.

use std::collections::HashMap;

use crate::backend::{NativeCompiledKernel, NativeContext};
use crate::kernel_ir::KernelOp;

/// Cache key: (operation, dtype tag).
type CacheKey = (KernelOp, u8);

/// Cached compiled kernel with associated metadata.
pub struct CachedKernel {
    pub compiled: NativeCompiledKernel,
}

/// Thread-local kernel cache per GPU context.
pub struct KernelCache {
    entries: HashMap<CacheKey, CachedKernel>,
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelCache {
    pub fn new() -> Self {
        KernelCache {
            entries: HashMap::new(),
        }
    }

    /// Get or compile a kernel for the given op and dtype.
    ///
    /// Returns a reference to the compiled kernel on success.
    pub fn get_or_compile(
        &mut self,
        ctx: &NativeContext,
        op: KernelOp,
        dtype: u8,
    ) -> Result<&CachedKernel, String> {
        let key = (op, dtype);

        if let std::collections::hash_map::Entry::Vacant(e) = self.entries.entry(key) {
            let compiled = compile_for_backend(ctx, op, dtype)?;
            e.insert(CachedKernel { compiled });
        }

        Ok(self.entries.get(&key).unwrap())
    }

    /// Whether the cache is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of cached kernels.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Compile a kernel for the active backend.
#[allow(unused_variables)]
fn compile_for_backend(
    ctx: &NativeContext,
    op: KernelOp,
    dtype: u8,
) -> Result<NativeCompiledKernel, String> {
    match ctx {
        #[cfg(feature = "metal-backend")]
        NativeContext::Metal(metal_ctx) => {
            use crate::codegen::msl;
            use crate::metal::compile;
            let source = msl::emit_kernel(op, dtype);
            let fn_name = msl::kernel_fn_name(op, dtype);
            let compiled = compile::compile_msl(metal_ctx, &source, &fn_name)?;
            Ok(NativeCompiledKernel::Metal(compiled))
        }
        #[cfg(feature = "webgpu-backend")]
        NativeContext::Wgpu(wgpu_ctx) => {
            use crate::codegen::wgsl;
            use crate::wgpu_backend::compile;
            let source = wgsl::emit_kernel(op, dtype);
            let fn_name = wgsl::kernel_fn_name(op, dtype);
            let num_buffers = wgsl::kernel_num_buffers(op);
            let compiled = compile::compile_wgsl(
                wgpu_ctx,
                &source,
                &fn_name,
                num_buffers,
                wgsl::WORKGROUP_SIZE,
            )?;
            Ok(NativeCompiledKernel::Wgpu(compiled))
        }
        #[cfg(feature = "cuda-backend")]
        NativeContext::Cuda(cuda_ctx) => {
            use crate::codegen::cuda;
            use crate::cuda::compile;
            let source = cuda::emit_kernel(op, dtype);
            let fn_name = cuda::kernel_fn_name(op, dtype);
            let compiled = compile::compile_cuda(cuda_ctx, &source, &fn_name)?;
            Ok(NativeCompiledKernel::Cuda(compiled))
        }
        NativeContext::Unavailable => Err("no GPU backend available".to_string()),
    }
}
