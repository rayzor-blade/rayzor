//! GPU buffer management — CPU↔GPU data transfer
//!
//! GpuBuffer wraps a NativeBuffer (Metal or wgpu) with metadata about
//! element count and dtype, enabling typed tensor interop.
//!
//! Buffers can be either **materialized** (backed by GPU memory) or **lazy**
//! (a pending computation DAG that gets fused and dispatched on demand).

use std::rc::Rc;

use crate::backend::{NativeBuffer, NativeCompiledKernel, NativeContext};
use crate::device::GpuContext;
use crate::lazy::LazyNode;

/// DType tags matching runtime/src/tensor.rs
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F64: u8 = 1;
pub const DTYPE_I32: u8 = 2;
pub const DTYPE_I64: u8 = 3;

/// Byte size per element for each dtype.
pub fn dtype_byte_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        DTYPE_F64 => 8,
        DTYPE_I32 => 4,
        DTYPE_I64 => 8,
        _ => 8, // default to f64
    }
}

/// The internal state of a GpuBuffer — materialized or lazy.
pub enum GpuBufferKind {
    /// Backed by actual GPU memory.
    Materialized(Rc<NativeBuffer>),
    /// Pending computation — will be fused and dispatched when materialized.
    Lazy(LazyNode),
}

/// Opaque GPU buffer handle.
pub struct GpuBuffer {
    pub(crate) kind: GpuBufferKind,
    pub numel: usize,
    pub dtype: u8,
}

impl GpuBuffer {
    /// Create a new materialized buffer.
    pub(crate) fn materialized(inner: NativeBuffer, numel: usize, dtype: u8) -> Self {
        GpuBuffer {
            kind: GpuBufferKind::Materialized(Rc::new(inner)),
            numel,
            dtype,
        }
    }

    /// Create a new lazy buffer (pending computation).
    pub(crate) fn lazy(node: LazyNode, numel: usize, dtype: u8) -> Self {
        GpuBuffer {
            kind: GpuBufferKind::Lazy(node),
            numel,
            dtype,
        }
    }

    /// Get a shared reference to the underlying NativeBuffer.
    ///
    /// Call `ensure_materialized()` first if the buffer might be lazy.
    pub(crate) fn native_buffer(&self) -> &Rc<NativeBuffer> {
        match &self.kind {
            GpuBufferKind::Materialized(buf) => buf,
            GpuBufferKind::Lazy(_) => {
                panic!("GpuBuffer not materialized — call ensure_materialized() first")
            }
        }
    }

    /// Materialize a lazy buffer by compiling and dispatching its fused kernel.
    ///
    /// No-op if already materialized.
    pub(crate) fn ensure_materialized(&mut self, gpu_ctx: &mut GpuContext) -> Result<(), String> {
        if let GpuBufferKind::Lazy(ref lazy_node) = self.kind {
            let native_buf = materialize_lazy(gpu_ctx, lazy_node)?;
            self.kind = GpuBufferKind::Materialized(Rc::new(native_buf));
        }
        Ok(())
    }
}

/// Compile and dispatch a fused kernel for a lazy node, returning the result NativeBuffer.
fn materialize_lazy(
    gpu_ctx: &mut GpuContext,
    lazy_node: &LazyNode,
) -> Result<NativeBuffer, String> {
    let op = &lazy_node.op;
    let dtype = lazy_node.dtype;
    let numel = lazy_node.numel;

    // Collect all input buffers from the lazy tree
    let (input_bufs, ptr_to_idx) = crate::lazy::collect_inputs(op);

    // Check fused kernel cache (keyed by structural hash + dtype)
    let struct_hash = crate::lazy::structural_hash(op);
    let cache_key = (struct_hash, dtype);

    let compiled = if let Some(cached) = gpu_ctx.fused_cache.get(&cache_key) {
        cached.clone()
    } else {
        let compiled =
            compile_fused_kernel(&gpu_ctx.inner, op, dtype, &ptr_to_idx, input_bufs.len())?;
        let compiled = Rc::new(compiled);
        gpu_ctx.fused_cache.insert(cache_key, compiled.clone());
        compiled
    };

    // Allocate result buffer
    let byte_size = numel * dtype_byte_size(dtype);
    let result_buf = gpu_ctx
        .inner
        .allocate_buffer(byte_size)
        .ok_or("failed to allocate result buffer for fused kernel")?;

    // Dispatch fused kernel
    dispatch_fused(&gpu_ctx.inner, &compiled, &input_bufs, &result_buf, numel)?;

    Ok(result_buf)
}

/// Compile a fused kernel for the active backend.
#[allow(unused_variables)]
fn compile_fused_kernel(
    ctx: &NativeContext,
    op: &crate::lazy::LazyOp,
    dtype: u8,
    ptr_to_idx: &std::collections::HashMap<usize, usize>,
    num_inputs: usize,
) -> Result<NativeCompiledKernel, String> {
    match ctx {
        #[cfg(feature = "metal-backend")]
        NativeContext::Metal(metal_ctx) => {
            use crate::codegen::msl_fused;
            use crate::metal::compile;
            let fused = msl_fused::emit_fused_kernel(op, dtype, ptr_to_idx, num_inputs);
            let compiled = compile::compile_msl(metal_ctx, &fused.source, &fused.fn_name)?;
            Ok(NativeCompiledKernel::Metal(compiled))
        }
        #[cfg(feature = "webgpu-backend")]
        NativeContext::Wgpu(wgpu_ctx) => {
            use crate::codegen::wgsl_fused;
            use crate::wgpu_backend::compile;
            let fused = wgsl_fused::emit_fused_kernel(op, dtype, ptr_to_idx, num_inputs);
            let num_buffers = num_inputs + 1;
            let compiled = compile::compile_wgsl(
                wgpu_ctx,
                &fused.source,
                &fused.fn_name,
                num_buffers,
                crate::codegen::wgsl::WORKGROUP_SIZE,
            )?;
            Ok(NativeCompiledKernel::Wgpu(compiled))
        }
        #[cfg(feature = "cuda-backend")]
        NativeContext::Cuda(cuda_ctx) => {
            use crate::codegen::cuda_fused;
            use crate::cuda::compile;
            let fused = cuda_fused::emit_fused_kernel(op, dtype, ptr_to_idx, num_inputs);
            let compiled = compile::compile_cuda(cuda_ctx, &fused.source, &fused.fn_name)?;
            Ok(NativeCompiledKernel::Cuda(compiled))
        }
        NativeContext::Unavailable => Err("no GPU backend available".to_string()),
    }
}

/// Dispatch a fused kernel on the active backend.
#[allow(unused_variables)]
fn dispatch_fused(
    ctx: &NativeContext,
    compiled: &NativeCompiledKernel,
    input_bufs: &[Rc<NativeBuffer>],
    result_buf: &NativeBuffer,
    numel: usize,
) -> Result<(), String> {
    match (ctx, compiled) {
        #[cfg(feature = "metal-backend")]
        (NativeContext::Metal(metal_ctx), NativeCompiledKernel::Metal(kernel)) => {
            use crate::metal::{buffer_ops::MetalBuffer, dispatch};
            let input_wrappers: Vec<MetalBuffer> = input_bufs
                .iter()
                .filter_map(|nb| match nb.as_ref() {
                    NativeBuffer::Metal(mb) => Some(MetalBuffer {
                        mtl_buffer: mb.mtl_buffer.clone(),
                        byte_size: 0,
                    }),
                    _ => None,
                })
                .collect();
            let result_metal = match result_buf {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("result buffer is not Metal".to_string()),
            };
            let mut all_bufs: Vec<&MetalBuffer> = input_wrappers.iter().collect();
            all_bufs.push(result_metal);
            dispatch::dispatch(metal_ctx, kernel, &all_bufs, numel)
        }
        #[cfg(feature = "webgpu-backend")]
        (NativeContext::Wgpu(wgpu_ctx), NativeCompiledKernel::Wgpu(kernel)) => {
            use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};
            let input_wgpu: Vec<&WgpuBuffer> = input_bufs
                .iter()
                .filter_map(|nb| match nb.as_ref() {
                    NativeBuffer::Wgpu(wb) => Some(wb),
                    _ => None,
                })
                .collect();
            let result_wgpu = match result_buf {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("result buffer is not wgpu".to_string()),
            };
            let mut all_bufs: Vec<&WgpuBuffer> = input_wgpu;
            all_bufs.push(result_wgpu);
            dispatch::dispatch(wgpu_ctx, kernel, &all_bufs, numel)
        }
        #[cfg(feature = "cuda-backend")]
        (NativeContext::Cuda(cuda_ctx), NativeCompiledKernel::Cuda(kernel)) => {
            use crate::cuda::{buffer_ops::CudaBuffer, dispatch};
            let input_cuda: Vec<&CudaBuffer> = input_bufs
                .iter()
                .filter_map(|nb| match nb.as_ref() {
                    NativeBuffer::Cuda(cb) => Some(cb),
                    _ => None,
                })
                .collect();
            let result_cuda = match result_buf {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("result buffer is not CUDA".to_string()),
            };
            let mut all_bufs: Vec<&CudaBuffer> = input_cuda;
            all_bufs.push(result_cuda);
            dispatch::dispatch(cuda_ctx, kernel, &all_bufs, numel)
        }
        _ => Err("backend mismatch between context and compiled kernel".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Extern C API
// ---------------------------------------------------------------------------

/// Create a GPU buffer from a RayzorTensor.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_create_buffer(ctx: i64, tensor_ptr: i64) -> i64 {
    if ctx == 0 || tensor_ptr == 0 {
        return 0;
    }

    let gpu_ctx = &*(ctx as *const GpuContext);
    let tensor = tensor_ptr as *const u8;
    let data_ptr = *(tensor as *const *const u8);
    let numel = *(tensor.add(32) as *const usize);
    let dtype = *tensor.add(40);
    let byte_size = numel * dtype_byte_size(dtype);

    match gpu_ctx.inner.buffer_from_data(data_ptr, byte_size) {
        Some(inner) => {
            let buf = GpuBuffer::materialized(inner, numel, dtype);
            Box::into_raw(Box::new(buf)) as i64
        }
        None => 0,
    }
}

/// Allocate an empty GPU buffer with the given element count and dtype.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_alloc_buffer(ctx: i64, numel: i64, dtype: i64) -> i64 {
    if ctx == 0 || numel <= 0 {
        return 0;
    }

    let gpu_ctx = &*(ctx as *const GpuContext);
    let numel = numel as usize;
    let dtype = dtype as u8;
    let byte_size = numel * dtype_byte_size(dtype);

    match gpu_ctx.inner.allocate_buffer(byte_size) {
        Some(inner) => {
            let buf = GpuBuffer::materialized(inner, numel, dtype);
            Box::into_raw(Box::new(buf)) as i64
        }
        None => 0,
    }
}

/// Copy GPU buffer data back to a new RayzorTensor.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_to_tensor(ctx: i64, buffer_ptr: i64) -> i64 {
    if ctx == 0 || buffer_ptr == 0 {
        return 0;
    }

    let buf = &mut *(buffer_ptr as *mut GpuBuffer);
    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    if buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }

    let byte_size = buf.numel * dtype_byte_size(buf.dtype);
    let native_buf = buf.native_buffer();

    let data_vec = match native_buf.read_bytes(byte_size) {
        Some(d) => d,
        None => return 0,
    };

    let data = libc::malloc(byte_size) as *mut u8;
    if data.is_null() {
        return 0;
    }
    std::ptr::copy_nonoverlapping(data_vec.as_ptr(), data, byte_size);

    let shape = libc::malloc(std::mem::size_of::<usize>()) as *mut usize;
    if shape.is_null() {
        libc::free(data as *mut libc::c_void);
        return 0;
    }
    *shape = buf.numel;

    let strides = libc::malloc(std::mem::size_of::<usize>()) as *mut usize;
    if strides.is_null() {
        libc::free(data as *mut libc::c_void);
        libc::free(shape as *mut libc::c_void);
        return 0;
    }
    *strides = 1;

    let tensor_size: usize = 48;
    let tensor = libc::malloc(tensor_size) as *mut u8;
    if tensor.is_null() {
        libc::free(data as *mut libc::c_void);
        libc::free(shape as *mut libc::c_void);
        libc::free(strides as *mut libc::c_void);
        return 0;
    }

    *(tensor as *mut *mut u8) = data;
    *(tensor.add(8) as *mut *mut usize) = shape;
    *(tensor.add(16) as *mut *mut usize) = strides;
    *(tensor.add(24) as *mut usize) = 1;
    *(tensor.add(32) as *mut usize) = buf.numel;
    *tensor.add(40) = buf.dtype;
    *tensor.add(41) = 1;

    tensor as i64
}

/// Free a GPU buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_free_buffer(_ctx: i64, buffer_ptr: i64) {
    if buffer_ptr == 0 {
        return;
    }
    let _ = Box::from_raw(buffer_ptr as *mut GpuBuffer);
}

/// Get the number of elements in a GPU buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_buffer_numel(buffer_ptr: i64) -> i64 {
    if buffer_ptr == 0 {
        return 0;
    }
    let buf = &*(buffer_ptr as *const GpuBuffer);
    buf.numel as i64
}

/// Get the dtype tag of a GPU buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_buffer_dtype(buffer_ptr: i64) -> i64 {
    if buffer_ptr == 0 {
        return 0;
    }
    let buf = &*(buffer_ptr as *const GpuBuffer);
    buf.dtype as i64
}

// ---------------------------------------------------------------------------
// Structured buffer API for @:gpuStruct
// ---------------------------------------------------------------------------

/// Create a GPU buffer from an array of @:gpuStruct instances.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_create_struct_buffer(
    ctx: i64,
    array_ptr: i64,
    count: i64,
    struct_size: i64,
) -> i64 {
    if ctx == 0 || array_ptr == 0 || count <= 0 || struct_size <= 0 {
        return 0;
    }

    let gpu_ctx = &*(ctx as *const GpuContext);
    let count = count as usize;
    let struct_size = struct_size as usize;
    let total_bytes = count * struct_size;

    let staging = libc::malloc(total_bytes) as *mut u8;
    if staging.is_null() {
        return 0;
    }

    let array_data = *(array_ptr as *const *const i64);
    for i in 0..count {
        let struct_ptr = *array_data.add(i) as *const u8;
        if !struct_ptr.is_null() {
            std::ptr::copy_nonoverlapping(struct_ptr, staging.add(i * struct_size), struct_size);
        } else {
            std::ptr::write_bytes(staging.add(i * struct_size), 0, struct_size);
        }
    }

    let result = match gpu_ctx.inner.buffer_from_data(staging, total_bytes) {
        Some(inner) => {
            let buf = GpuBuffer::materialized(inner, count, DTYPE_F32);
            Box::into_raw(Box::new(buf)) as i64
        }
        None => 0,
    };

    libc::free(staging as *mut libc::c_void);
    result
}

/// Allocate an empty GPU buffer for `count` structs of `struct_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_alloc_struct_buffer(
    ctx: i64,
    count: i64,
    struct_size: i64,
) -> i64 {
    if ctx == 0 || count <= 0 || struct_size <= 0 {
        return 0;
    }

    let gpu_ctx = &*(ctx as *const GpuContext);
    let total_bytes = (count as usize) * (struct_size as usize);

    match gpu_ctx.inner.allocate_buffer(total_bytes) {
        Some(inner) => {
            let buf = GpuBuffer::materialized(inner, count as usize, DTYPE_F32);
            Box::into_raw(Box::new(buf)) as i64
        }
        None => 0,
    }
}

/// Read a single f32 field from a structured GPU buffer, promote to f64.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_read_struct_float(
    _ctx: i64,
    buffer_ptr: i64,
    index: i64,
    struct_size: i64,
    field_offset: i64,
) -> f64 {
    if buffer_ptr == 0 {
        return 0.0;
    }

    let buf = &*(buffer_ptr as *const GpuBuffer);
    let native_buf = buf.native_buffer();
    let byte_offset = (index as usize) * (struct_size as usize) + (field_offset as usize);

    let ptr = native_buf.contents_ptr();
    if !ptr.is_null() {
        let val = *(ptr.add(byte_offset) as *const f32);
        return val as f64;
    }

    // Fallback for wgpu: read via staging buffer
    let total = byte_offset + 4;
    if let Some(data) = native_buf.read_bytes(total) {
        if data.len() >= total {
            let val = *(data.as_ptr().add(byte_offset) as *const f32);
            return val as f64;
        }
    }
    0.0
}

/// Read a single i32 field from a structured GPU buffer, extend to i64.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_read_struct_int(
    _ctx: i64,
    buffer_ptr: i64,
    index: i64,
    struct_size: i64,
    field_offset: i64,
) -> i64 {
    if buffer_ptr == 0 {
        return 0;
    }

    let buf = &*(buffer_ptr as *const GpuBuffer);
    let native_buf = buf.native_buffer();
    let byte_offset = (index as usize) * (struct_size as usize) + (field_offset as usize);

    let ptr = native_buf.contents_ptr();
    if !ptr.is_null() {
        let val = *(ptr.add(byte_offset) as *const i32);
        return val as i64;
    }

    let total = byte_offset + 4;
    if let Some(data) = native_buf.read_bytes(total) {
        if data.len() >= total {
            let val = *(data.as_ptr().add(byte_offset) as *const i32);
            return val as i64;
        }
    }
    0
}
