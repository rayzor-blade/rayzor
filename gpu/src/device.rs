//! GPU compute context — device initialization and lifecycle

use std::collections::HashMap;
use std::rc::Rc;

use crate::backend::{NativeCompiledKernel, NativeContext};
use crate::kernel_cache::KernelCache;

/// Opaque GPU context handle passed as i64 through the JIT ABI.
///
/// Wraps a NativeContext (Metal or wgpu) + kernel cache.
pub struct GpuContext {
    pub(crate) inner: NativeContext,
    pub(crate) kernel_cache: KernelCache,
    /// Cache for fused kernels, keyed by (structural_hash, dtype).
    pub(crate) fused_cache: HashMap<(u64, u8), Rc<NativeCompiledKernel>>,
}

// ---------------------------------------------------------------------------
// Extern C API
// ---------------------------------------------------------------------------

/// Create a new GPU compute context.
/// Returns an opaque i64 handle (pointer), or 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_gpu_compute_create() -> i64 {
    match NativeContext::new() {
        Some(ctx) => {
            let gpu_ctx = GpuContext {
                inner: ctx,
                kernel_cache: KernelCache::new(),
                fused_cache: HashMap::new(),
            };
            let boxed = Box::new(gpu_ctx);
            Box::into_raw(boxed) as i64
        }
        None => 0,
    }
}

/// Destroy a GPU compute context and free its resources.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rayzor_gpu_compute_destroy(ctx: i64) {
    unsafe {
        if ctx == 0 {
            return;
        }
        let _ = Box::from_raw(ctx as *mut GpuContext);
    }
}

/// Check if GPU compute is available on this system.
/// Returns 1 if available, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_gpu_compute_is_available() -> i8 {
    if NativeContext::is_available() { 1 } else { 0 }
}
