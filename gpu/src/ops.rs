//! GPU compute operations — elementwise binary and unary ops.
//!
//! Elementwise ops are **lazy** — they build a computation DAG instead of
//! dispatching immediately. When materialization is triggered (by `toTensor`,
//! a reduction, or matmul), the entire chain is fused into a single kernel.
//!
//! Non-fuseable ops (reductions, matmul) materialize their inputs first.

use std::rc::Rc;

use crate::backend::{NativeBuffer, NativeCompiledKernel, NativeContext};
use crate::buffer::{self, GpuBuffer, GpuBufferKind};
use crate::device::GpuContext;
use crate::kernel_ir::KernelOp;
use crate::lazy::{LazyNode, LazyOp};

/// Workgroup/threadgroup size for reductions.
const REDUCE_WG_SIZE: usize = 256;

// ---------------------------------------------------------------------------
// Internal helpers — lazy elementwise
// ---------------------------------------------------------------------------

/// Convert a GpuBuffer reference to a LazyOp node.
fn buf_to_lazy_op(buf: &GpuBuffer) -> Rc<LazyOp> {
    match &buf.kind {
        GpuBufferKind::Lazy(node) => node.op.clone(),
        GpuBufferKind::Materialized(native_buf) => Rc::new(LazyOp::Input(native_buf.clone())),
    }
}

/// Create a lazy binary elementwise GpuBuffer.
unsafe fn binary_lazy(a: i64, b: i64, op: KernelOp) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }

    let a_buf = &*(a as *const GpuBuffer);
    let b_buf = &*(b as *const GpuBuffer);

    if a_buf.dtype != b_buf.dtype || a_buf.numel != b_buf.numel {
        return 0;
    }

    let lhs = buf_to_lazy_op(a_buf);
    let rhs = buf_to_lazy_op(b_buf);

    let node = LazyNode {
        op: Rc::new(LazyOp::Binary { op, lhs, rhs }),
        dtype: a_buf.dtype,
        numel: a_buf.numel,
    };

    let result = GpuBuffer::lazy(node, a_buf.numel, a_buf.dtype);
    Box::into_raw(Box::new(result)) as i64
}

/// Create a lazy unary elementwise GpuBuffer.
unsafe fn unary_lazy(a: i64, op: KernelOp) -> i64 {
    if a == 0 {
        return 0;
    }

    let a_buf = &*(a as *const GpuBuffer);
    let input = buf_to_lazy_op(a_buf);

    let node = LazyNode {
        op: Rc::new(LazyOp::Unary { op, input }),
        dtype: a_buf.dtype,
        numel: a_buf.numel,
    };

    let result = GpuBuffer::lazy(node, a_buf.numel, a_buf.dtype);
    Box::into_raw(Box::new(result)) as i64
}

// ---------------------------------------------------------------------------
// Extern C API — Binary ops: (ctx, a, b) -> result (lazy)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_add(_ctx: i64, a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Add)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_sub(_ctx: i64, a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Sub)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_mul(_ctx: i64, a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Mul)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_div(_ctx: i64, a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Div)
}

// ---------------------------------------------------------------------------
// Extern C API — Unary ops: (ctx, a) -> result (lazy)
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_neg(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Neg)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_abs(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Abs)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_sqrt(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Sqrt)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_exp(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Exp)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_log(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Log)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_relu(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Relu)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_sigmoid(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Sigmoid)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_tanh(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Tanh)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_gelu(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Gelu)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_silu(_ctx: i64, a: i64) -> i64 {
    unary_lazy(a, KernelOp::Silu)
}

// ---------------------------------------------------------------------------
// Internal helpers — Reductions
// ---------------------------------------------------------------------------

fn next_power_of_2(n: usize) -> usize {
    let mut v = n.max(1);
    v -= 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

/// Perform a GPU reduction and return the scalar result as f64.
///
/// Materializes the input buffer first if it's lazy.
/// Backend dispatch for two-pass reduction: each backend handles its own
/// buffer allocation, kernel dispatch, and readback.
unsafe fn reduce_impl(ctx: i64, buf: i64, op: KernelOp) -> f64 {
    if ctx == 0 || buf == 0 {
        return 0.0;
    }

    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let a_buf = &mut *(buf as *mut GpuBuffer);

    if a_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0.0;
    }

    let dtype = a_buf.dtype;
    let numel = a_buf.numel;
    let elem_size = buffer::dtype_byte_size(dtype);

    if numel == 0 {
        return 0.0;
    }

    // Compile reduction kernel
    let cached = match gpu_ctx
        .kernel_cache
        .get_or_compile(&gpu_ctx.inner, op, dtype)
    {
        Ok(k) => k,
        Err(_) => return 0.0,
    };

    // Two-pass reduction via backend dispatch
    let tg_size = REDUCE_WG_SIZE.min(next_power_of_2(numel));
    let num_tgs = if numel <= tg_size {
        1
    } else {
        numel.div_ceil(tg_size).min(256)
    };

    reduce_dispatch(
        &gpu_ctx.inner,
        &cached.compiled,
        a_buf.native_buffer(),
        numel,
        num_tgs,
        tg_size,
        elem_size,
        dtype,
    )
    .unwrap_or(0.0)
}

/// Backend-dispatch for two-pass reduction.
#[allow(unused_variables, clippy::too_many_arguments)]
fn reduce_dispatch(
    ctx: &NativeContext,
    compiled: &NativeCompiledKernel,
    input_buf: &Rc<NativeBuffer>,
    numel: usize,
    num_tgs: usize,
    tg_size: usize,
    elem_size: usize,
    dtype: u8,
) -> Result<f64, String> {
    match (ctx, compiled) {
        #[cfg(feature = "metal-backend")]
        (NativeContext::Metal(metal_ctx), NativeCompiledKernel::Metal(kernel)) => {
            use crate::metal::{buffer_ops::MetalBuffer, dispatch};
            use objc2_metal::MTLSize;

            let input_metal = match input_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("input not Metal".into()),
            };

            let numel_u32 = numel as u32;
            let numel_buf = MetalBuffer::from_value(metal_ctx, &numel_u32)
                .ok_or("failed to alloc numel buf")?;
            let partial_buf = MetalBuffer::allocate(metal_ctx, num_tgs * elem_size)
                .ok_or("failed to alloc partial buf")?;

            let tg_count = MTLSize {
                width: num_tgs,
                height: 1,
                depth: 1,
            };
            let tg_threads = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };

            dispatch::dispatch_threadgroups(
                metal_ctx,
                kernel,
                &[input_metal, &partial_buf, &numel_buf],
                tg_count,
                tg_threads,
            )?;

            let result_buf = if num_tgs > 1 {
                let final_buf = MetalBuffer::allocate(metal_ctx, elem_size)
                    .ok_or("failed to alloc final buf")?;
                let pass2_numel = num_tgs as u32;
                let pass2_numel_buf = MetalBuffer::from_value(metal_ctx, &pass2_numel)
                    .ok_or("failed to alloc pass2 numel buf")?;
                let pass2_tg_size = next_power_of_2(num_tgs);
                dispatch::dispatch_threadgroups(
                    metal_ctx,
                    kernel,
                    &[&partial_buf, &final_buf, &pass2_numel_buf],
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: pass2_tg_size,
                        height: 1,
                        depth: 1,
                    },
                )?;
                final_buf
            } else {
                partial_buf
            };

            let ptr = result_buf.contents();
            Ok(unsafe {
                match dtype {
                    buffer::DTYPE_F32 => *(ptr as *const f32) as f64,
                    buffer::DTYPE_F64 => *(ptr as *const f64),
                    buffer::DTYPE_I32 => *(ptr as *const i32) as f64,
                    buffer::DTYPE_I64 => *(ptr as *const i64) as f64,
                    _ => 0.0,
                }
            })
        }
        #[cfg(feature = "webgpu-backend")]
        (NativeContext::Wgpu(wgpu_ctx), NativeCompiledKernel::Wgpu(kernel)) => {
            use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};

            let input_wgpu = match input_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("input not wgpu".into()),
            };

            // Create numel uniform buffer
            let numel_u32 = numel as u32;
            let numel_buf = unsafe {
                WgpuBuffer::from_data(wgpu_ctx, &numel_u32 as *const u32 as *const u8, 4)
            }
            .ok_or("failed to alloc numel buf")?;

            let partial_buf = WgpuBuffer::allocate(wgpu_ctx, num_tgs * elem_size)
                .ok_or("failed to alloc partial buf")?;

            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[input_wgpu, &partial_buf, &numel_buf],
                (num_tgs, 1, 1),
            )?;

            let result_buf = if num_tgs > 1 {
                let final_buf =
                    WgpuBuffer::allocate(wgpu_ctx, elem_size).ok_or("failed to alloc final buf")?;
                let pass2_numel = num_tgs as u32;
                let pass2_numel_buf = unsafe {
                    WgpuBuffer::from_data(wgpu_ctx, &pass2_numel as *const u32 as *const u8, 4)
                }
                .ok_or("failed to alloc pass2 numel buf")?;
                dispatch::dispatch_workgroups(
                    wgpu_ctx,
                    kernel,
                    &[&partial_buf, &final_buf, &pass2_numel_buf],
                    (1, 1, 1),
                )?;
                final_buf
            } else {
                partial_buf
            };

            let data = result_buf
                .read_to_vec(elem_size)
                .ok_or("failed to read back reduction result")?;
            Ok(match dtype {
                buffer::DTYPE_F32 => unsafe { *(data.as_ptr() as *const f32) as f64 },
                buffer::DTYPE_F64 => unsafe { *(data.as_ptr() as *const f64) },
                buffer::DTYPE_I32 => unsafe { *(data.as_ptr() as *const i32) as f64 },
                buffer::DTYPE_I64 => unsafe { *(data.as_ptr() as *const i64) as f64 },
                _ => 0.0,
            })
        }
        _ => Err("backend mismatch".into()),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — Matmul
// ---------------------------------------------------------------------------

/// Perform GPU matrix multiplication: C(M×N) = A(M×K) × B(K×N).
unsafe fn matmul_impl(ctx: i64, a: i64, b: i64, m: usize, k: usize, n: usize) -> i64 {
    if ctx == 0 || a == 0 || b == 0 || m == 0 || k == 0 || n == 0 {
        return 0;
    }

    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let a_buf = &mut *(a as *mut GpuBuffer);
    let b_buf = &mut *(b as *mut GpuBuffer);
    if a_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    if b_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }

    let dtype = a_buf.dtype;
    let cached = match gpu_ctx
        .kernel_cache
        .get_or_compile(&gpu_ctx.inner, KernelOp::Matmul, dtype)
    {
        Ok(k) => k,
        Err(_) => return 0,
    };

    let elem_size = buffer::dtype_byte_size(dtype);

    match matmul_dispatch(
        &gpu_ctx.inner,
        &cached.compiled,
        a_buf.native_buffer(),
        b_buf.native_buffer(),
        m,
        k,
        n,
        elem_size,
        dtype,
    ) {
        Ok(result_native) => {
            let result = GpuBuffer::materialized(result_native, m * n, dtype);
            Box::into_raw(Box::new(result)) as i64
        }
        Err(_) => 0,
    }
}

/// Backend-dispatch for matmul.
#[allow(unused_variables, clippy::too_many_arguments)]
fn matmul_dispatch(
    ctx: &NativeContext,
    compiled: &NativeCompiledKernel,
    a_buf: &Rc<NativeBuffer>,
    b_buf: &Rc<NativeBuffer>,
    m: usize,
    k: usize,
    n: usize,
    elem_size: usize,
    _dtype: u8,
) -> Result<NativeBuffer, String> {
    match (ctx, compiled) {
        #[cfg(feature = "metal-backend")]
        (NativeContext::Metal(metal_ctx), NativeCompiledKernel::Metal(kernel)) => {
            use crate::metal::{buffer_ops::MetalBuffer, dispatch};
            use objc2_metal::MTLSize;

            let a_metal = match a_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("a not Metal".into()),
            };
            let b_metal = match b_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("b not Metal".into()),
            };

            let result_inner = MetalBuffer::allocate(metal_ctx, m * n * elem_size)
                .ok_or("failed to alloc result")?;
            let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
            let dims_buf =
                MetalBuffer::from_value(metal_ctx, &dims).ok_or("failed to alloc dims")?;

            let threads_per_tg = 16usize;
            dispatch::dispatch_threadgroups(
                metal_ctx,
                kernel,
                &[a_metal, b_metal, &result_inner, &dims_buf],
                MTLSize {
                    width: n.div_ceil(threads_per_tg),
                    height: m.div_ceil(threads_per_tg),
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_tg,
                    height: threads_per_tg,
                    depth: 1,
                },
            )?;

            Ok(NativeBuffer::Metal(result_inner))
        }
        #[cfg(feature = "webgpu-backend")]
        (NativeContext::Wgpu(wgpu_ctx), NativeCompiledKernel::Wgpu(kernel)) => {
            use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};

            let a_wgpu = match a_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("a not wgpu".into()),
            };
            let b_wgpu = match b_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("b not wgpu".into()),
            };

            let result_inner = WgpuBuffer::allocate(wgpu_ctx, m * n * elem_size)
                .ok_or("failed to alloc result")?;
            let dims: [u32; 4] = [m as u32, k as u32, n as u32, 0];
            let dims_buf =
                unsafe { WgpuBuffer::from_data(wgpu_ctx, dims.as_ptr() as *const u8, 16) }
                    .ok_or("failed to alloc dims")?;

            let threads_per_wg = 16usize;
            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[a_wgpu, b_wgpu, &result_inner, &dims_buf],
                (n.div_ceil(threads_per_wg), m.div_ceil(threads_per_wg), 1),
            )?;

            Ok(NativeBuffer::Wgpu(result_inner))
        }
        _ => Err("backend mismatch".into()),
    }
}

// ---------------------------------------------------------------------------
// Extern C API — Reductions: (ctx, buf) -> f64
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_sum(ctx: i64, buf: i64) -> f64 {
    reduce_impl(ctx, buf, KernelOp::ReduceSum)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_mean(ctx: i64, buf: i64) -> f64 {
    if buf == 0 {
        return 0.0;
    }
    let a_buf = &*(buf as *const GpuBuffer);
    let numel = a_buf.numel;
    if numel == 0 {
        return 0.0;
    }
    reduce_impl(ctx, buf, KernelOp::ReduceSum) / numel as f64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_max(ctx: i64, buf: i64) -> f64 {
    reduce_impl(ctx, buf, KernelOp::ReduceMax)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_min(ctx: i64, buf: i64) -> f64 {
    reduce_impl(ctx, buf, KernelOp::ReduceMin)
}

// ---------------------------------------------------------------------------
// Extern C API — Dot product: (ctx, a, b) -> f64
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_dot(ctx: i64, a: i64, b: i64) -> f64 {
    let product = rayzor_gpu_compute_mul(ctx, a, b);
    if product == 0 {
        return 0.0;
    }
    let result = reduce_impl(ctx, product, KernelOp::ReduceSum);
    let _ = Box::from_raw(product as *mut GpuBuffer);
    result
}

// ---------------------------------------------------------------------------
// Extern C API — Matmul: (ctx, a, b, m, k, n) -> GpuBuffer handle
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_matmul(
    ctx: i64,
    a: i64,
    b: i64,
    m: i64,
    k: i64,
    n: i64,
) -> i64 {
    matmul_impl(ctx, a, b, m as usize, k as usize, n as usize)
}

// ---------------------------------------------------------------------------
// Extern C API — Batch Matmul: (ctx, a, b, batch, m, k, n) -> GpuBuffer handle
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_batch_matmul(
    ctx: i64,
    a: i64,
    b: i64,
    batch: i64,
    m: i64,
    k: i64,
    n: i64,
) -> i64 {
    batch_matmul_impl(ctx, a, b, batch as usize, m as usize, k as usize, n as usize)
}

unsafe fn batch_matmul_impl(
    ctx: i64,
    a: i64,
    b: i64,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> i64 {
    if ctx == 0 || a == 0 || b == 0 || batch == 0 || m == 0 || k == 0 || n == 0 {
        return 0;
    }

    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let a_buf = &mut *(a as *mut GpuBuffer);
    let b_buf = &mut *(b as *mut GpuBuffer);
    if a_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    if b_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }

    let dtype = a_buf.dtype;
    let cached = match gpu_ctx
        .kernel_cache
        .get_or_compile(&gpu_ctx.inner, KernelOp::BatchMatmul, dtype)
    {
        Ok(k) => k,
        Err(_) => return 0,
    };

    let elem_size = buffer::dtype_byte_size(dtype);

    match batch_matmul_dispatch(
        &gpu_ctx.inner,
        &cached.compiled,
        a_buf.native_buffer(),
        b_buf.native_buffer(),
        batch,
        m,
        k,
        n,
        elem_size,
    ) {
        Ok(result_native) => {
            let result = GpuBuffer::materialized(result_native, batch * m * n, dtype);
            Box::into_raw(Box::new(result)) as i64
        }
        Err(_) => 0,
    }
}

/// Backend-dispatch for batch matmul.
#[allow(unused_variables, clippy::too_many_arguments)]
fn batch_matmul_dispatch(
    ctx: &NativeContext,
    compiled: &NativeCompiledKernel,
    a_buf: &Rc<NativeBuffer>,
    b_buf: &Rc<NativeBuffer>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    elem_size: usize,
) -> Result<NativeBuffer, String> {
    match (ctx, compiled) {
        #[cfg(feature = "metal-backend")]
        (NativeContext::Metal(metal_ctx), NativeCompiledKernel::Metal(kernel)) => {
            use crate::metal::{buffer_ops::MetalBuffer, dispatch};
            use objc2_metal::MTLSize;

            let a_metal = match a_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("a not Metal".into()),
            };
            let b_metal = match b_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("b not Metal".into()),
            };

            let result_inner = MetalBuffer::allocate(metal_ctx, batch * m * n * elem_size)
                .ok_or("failed to alloc result")?;
            // dims = (M, K, N, B) — B in w component
            let dims: [u32; 4] = [m as u32, k as u32, n as u32, batch as u32];
            let dims_buf =
                MetalBuffer::from_value(metal_ctx, &dims).ok_or("failed to alloc dims")?;

            let threads_per_tg = 16usize;
            dispatch::dispatch_threadgroups(
                metal_ctx,
                kernel,
                &[a_metal, b_metal, &result_inner, &dims_buf],
                MTLSize {
                    width: n.div_ceil(threads_per_tg),
                    height: m.div_ceil(threads_per_tg),
                    depth: batch,
                },
                MTLSize {
                    width: threads_per_tg,
                    height: threads_per_tg,
                    depth: 1,
                },
            )?;

            Ok(NativeBuffer::Metal(result_inner))
        }
        #[cfg(feature = "webgpu-backend")]
        (NativeContext::Wgpu(wgpu_ctx), NativeCompiledKernel::Wgpu(kernel)) => {
            use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};

            let a_wgpu = match a_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("a not wgpu".into()),
            };
            let b_wgpu = match b_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("b not wgpu".into()),
            };

            let result_inner =
                WgpuBuffer::allocate(wgpu_ctx, batch * m * n * elem_size).ok_or("alloc failed")?;
            let dims: [u32; 4] = [m as u32, k as u32, n as u32, batch as u32];
            let dims_buf = WgpuBuffer::from_value(wgpu_ctx, &dims).ok_or("dims alloc failed")?;

            let tg = 16usize;
            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[a_wgpu, b_wgpu, &result_inner, &dims_buf],
                [n.div_ceil(tg) as u32, m.div_ceil(tg) as u32, batch as u32],
            )?;

            Ok(NativeBuffer::Wgpu(result_inner))
        }
        _ => Err("no matching backend for batch matmul".into()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_cache::KernelCache;
    use std::collections::HashMap;

    fn make_ctx() -> i64 {
        if !NativeContext::is_available() {
            return 0;
        }
        let native_ctx = NativeContext::new().unwrap();
        let gpu_ctx = GpuContext {
            inner: native_ctx,
            kernel_cache: KernelCache::new(),
            fused_cache: HashMap::new(),
        };
        Box::into_raw(Box::new(gpu_ctx)) as i64
    }

    unsafe fn create_test_buffer(ctx: i64, data: &[f32]) -> i64 {
        let gpu_ctx = &*(ctx as *const GpuContext);
        let byte_size = std::mem::size_of_val(data);
        let inner = gpu_ctx
            .inner
            .buffer_from_data(data.as_ptr() as *const u8, byte_size)
            .expect("failed to create test buffer");
        let buf = GpuBuffer::materialized(inner, data.len(), buffer::DTYPE_F32);
        Box::into_raw(Box::new(buf)) as i64
    }

    #[test]
    fn test_gpu_add_f32() {
        let ctx = make_ctx();
        if ctx == 0 {
            return;
        }

        let n = 1024;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

        let a_buf = unsafe { create_test_buffer(ctx, &a_data) };
        let b_buf = unsafe { create_test_buffer(ctx, &b_data) };

        let result = unsafe { rayzor_gpu_compute_add(ctx, a_buf, b_buf) };
        assert_ne!(result, 0, "add returned null");

        let gpu_ctx = unsafe { &mut *(ctx as *mut GpuContext) };
        let result_buf = unsafe { &mut *(result as *mut GpuBuffer) };
        assert!(
            matches!(result_buf.kind, GpuBufferKind::Lazy(_)),
            "add result should be lazy"
        );
        result_buf.ensure_materialized(gpu_ctx).unwrap();
        assert!(
            matches!(result_buf.kind, GpuBufferKind::Materialized(_)),
            "should be materialized now"
        );

        assert_eq!(result_buf.numel, n);
        assert_eq!(result_buf.dtype, buffer::DTYPE_F32);

        let byte_size = n * 4;
        let data = result_buf.native_buffer().read_bytes(byte_size).unwrap();
        let result_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        for (i, &val) in result_slice.iter().enumerate().take(n) {
            let expected = (i + i * 2) as f32;
            assert!(
                (val - expected).abs() < 1e-6,
                "add mismatch at {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        unsafe {
            let _ = Box::from_raw(result as *mut GpuBuffer);
            let _ = Box::from_raw(a_buf as *mut GpuBuffer);
            let _ = Box::from_raw(b_buf as *mut GpuBuffer);
            let _ = Box::from_raw(ctx as *mut GpuContext);
        }
    }

    #[test]
    fn test_fused_add_mul_relu() {
        let ctx = make_ctx();
        if ctx == 0 {
            return;
        }

        let n = 256;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32) - 128.0).collect();
        let b_data: Vec<f32> = vec![2.0; n];
        let c_data: Vec<f32> = vec![0.5; n];

        let a_buf = unsafe { create_test_buffer(ctx, &a_data) };
        let b_buf = unsafe { create_test_buffer(ctx, &b_data) };
        let c_buf = unsafe { create_test_buffer(ctx, &c_data) };

        let add_result = unsafe { rayzor_gpu_compute_add(ctx, a_buf, b_buf) };
        let mul_result = unsafe { rayzor_gpu_compute_mul(ctx, add_result, c_buf) };
        let relu_result = unsafe { rayzor_gpu_compute_relu(ctx, mul_result) };

        assert_ne!(relu_result, 0);

        let result_buf = unsafe { &mut *(relu_result as *mut GpuBuffer) };
        assert!(matches!(result_buf.kind, GpuBufferKind::Lazy(_)));

        let gpu_ctx = unsafe { &mut *(ctx as *mut GpuContext) };
        result_buf.ensure_materialized(gpu_ctx).unwrap();

        let byte_size = n * 4;
        let data = result_buf.native_buffer().read_bytes(byte_size).unwrap();
        let result_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        for (i, &val) in result_slice.iter().enumerate().take(n) {
            let a = (i as f32) - 128.0;
            let expected = f32::max(0.0, (a + 2.0) * 0.5);
            assert!(
                (val - expected).abs() < 1e-5,
                "fused mismatch at {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        assert!(
            !gpu_ctx.fused_cache.is_empty(),
            "fused cache should be populated"
        );

        unsafe {
            let _ = Box::from_raw(relu_result as *mut GpuBuffer);
            let _ = Box::from_raw(mul_result as *mut GpuBuffer);
            let _ = Box::from_raw(add_result as *mut GpuBuffer);
            let _ = Box::from_raw(a_buf as *mut GpuBuffer);
            let _ = Box::from_raw(b_buf as *mut GpuBuffer);
            let _ = Box::from_raw(c_buf as *mut GpuBuffer);
            let _ = Box::from_raw(ctx as *mut GpuContext);
        }
    }

    #[test]
    fn test_gpu_sum_f32() {
        let ctx = make_ctx();
        if ctx == 0 {
            return;
        }

        let n = 1024;
        let a_data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        let a_buf = unsafe { create_test_buffer(ctx, &a_data) };

        let result = unsafe { rayzor_gpu_compute_sum(ctx, a_buf) };
        let expected = (n * (n + 1) / 2) as f64;
        assert!(
            (result - expected).abs() < 1.0,
            "sum: expected {}, got {}",
            expected,
            result
        );

        unsafe {
            let _ = Box::from_raw(a_buf as *mut GpuBuffer);
            let _ = Box::from_raw(ctx as *mut GpuContext);
        }
    }

    #[test]
    fn test_lazy_sum_materializes() {
        let ctx = make_ctx();
        if ctx == 0 {
            return;
        }

        let n = 512;
        let a_data: Vec<f32> = vec![3.0; n];
        let b_data: Vec<f32> = vec![7.0; n];
        let a_buf = unsafe { create_test_buffer(ctx, &a_data) };
        let b_buf = unsafe { create_test_buffer(ctx, &b_data) };

        let add_result = unsafe { rayzor_gpu_compute_add(ctx, a_buf, b_buf) };
        assert_ne!(add_result, 0);

        let sum = unsafe { rayzor_gpu_compute_sum(ctx, add_result) };
        let expected = (3.0 + 7.0) * n as f64;
        assert!(
            (sum - expected).abs() < 1.0,
            "lazy sum: expected {}, got {}",
            expected,
            sum
        );

        unsafe {
            let _ = Box::from_raw(add_result as *mut GpuBuffer);
            let _ = Box::from_raw(a_buf as *mut GpuBuffer);
            let _ = Box::from_raw(b_buf as *mut GpuBuffer);
            let _ = Box::from_raw(ctx as *mut GpuContext);
        }
    }

    #[test]
    fn test_gpu_matmul_f32() {
        let ctx = make_ctx();
        if ctx == 0 {
            return;
        }

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let a_buf = unsafe { create_test_buffer(ctx, &a_data) };
        let b_buf = unsafe { create_test_buffer(ctx, &b_data) };

        let result = unsafe { rayzor_gpu_compute_matmul(ctx, a_buf, b_buf, 2, 2, 2) };
        assert_ne!(result, 0, "matmul returned null");

        let result_buf = unsafe { &*(result as *const GpuBuffer) };
        assert_eq!(result_buf.numel, 4);

        let data = result_buf.native_buffer().read_bytes(16).unwrap();
        let result_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, 4) };
        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (result_slice[i] - exp).abs() < 1e-3,
                "matmul[{}]: expected {}, got {}",
                i,
                exp,
                result_slice[i]
            );
        }

        unsafe {
            let _ = Box::from_raw(result as *mut GpuBuffer);
            let _ = Box::from_raw(a_buf as *mut GpuBuffer);
            let _ = Box::from_raw(b_buf as *mut GpuBuffer);
            let _ = Box::from_raw(ctx as *mut GpuContext);
        }
    }
}
