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
// GpuBuffer-method API for `@:op` overloading: `a + b` syntax.
//
// These take (self, other) — no ctx parameter, since binary_lazy doesn't
// touch the context (the result is a lazy DAG node; materialization later
// uses the GpuContext owned by GPUCompute).
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_buffer_add(a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Add)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_buffer_sub(a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Sub)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_buffer_mul(a: i64, b: i64) -> i64 {
    binary_lazy(a, b, KernelOp::Mul)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_buffer_div(a: i64, b: i64) -> i64 {
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
                    buffer::DTYPE_I32 => *(ptr as *const i32) as f64,
                    // F16/BF16/FP8 readback added in Phase 3b once half crate lands.
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
                buffer::DTYPE_I32 => unsafe { *(data.as_ptr() as *const i32) as f64 },
                _ => 0.0,
            })
        }
        #[cfg(feature = "cuda-backend")]
        (NativeContext::Cuda(cuda_ctx), NativeCompiledKernel::Cuda(kernel)) => {
            use crate::cuda::{buffer_ops::CudaBuffer, dispatch};

            let input_cuda = match input_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("input not CUDA".into()),
            };

            let partial_buf = CudaBuffer::allocate(cuda_ctx, num_tgs * elem_size)
                .ok_or("failed to alloc partial buf")?;

            dispatch::dispatch_reduction(
                cuda_ctx,
                kernel,
                input_cuda,
                &partial_buf,
                numel,
                tg_size,
            )?;

            let result_buf = if num_tgs > 1 {
                let final_buf =
                    CudaBuffer::allocate(cuda_ctx, elem_size).ok_or("failed to alloc final buf")?;
                dispatch::dispatch_reduction(
                    cuda_ctx,
                    kernel,
                    &partial_buf,
                    &final_buf,
                    num_tgs,
                    next_power_of_2(num_tgs),
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
                buffer::DTYPE_I32 => unsafe { *(data.as_ptr() as *const i32) as f64 },
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

// ---------------------------------------------------------------------------
// Q4_K matmul — weights stay quantised to the shader
// ---------------------------------------------------------------------------

/// `rayzor_gpu_compute_buffer_from_bytes(ctx, addr, byte_size) -> GpuBuffer`
///
/// Upload raw bytes with no dtype interpretation. The caller owns the layout —
/// used to hand the shader Q4_K_M blocks straight from the mmap'd GGUF, with
/// no CPU dequant and ~7x less traffic than the f32 expansion.
///
/// # Safety
/// `addr` must point to `byte_size` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_buffer_from_bytes(
    ctx: i64,
    addr: i64,
    byte_size: i64,
) -> i64 {
    gpu_thread_check("buffer_from_bytes");
    if ctx == 0 || addr == 0 || byte_size <= 0 {
        return 0;
    }
    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let native = match &gpu_ctx.inner {
        #[cfg(feature = "webgpu-backend")]
        NativeContext::Wgpu(wc) => {
            match crate::wgpu_backend::buffer_ops::WgpuBuffer::from_data(
                wc,
                addr as *const u8,
                byte_size as usize,
            ) {
                Some(b) => NativeBuffer::Wgpu(b),
                None => return 0,
            }
        }
        _ => return 0,
    };
    // u8 elements: the shader reinterprets them as u32 blocks.
    let buf = GpuBuffer::materialized(native, byte_size as usize, buffer::DTYPE_U8);
    Box::into_raw(Box::new(buf)) as i64
}

// ---------------------------------------------------------------------------
// Crash backtrace
// ---------------------------------------------------------------------------
//
// The routed prefill segfaults intermittently and NEVER under gdb, so a live
// debugger cannot catch it, and core dumps are unavailable here (core_pattern
// is piped to apport, `ulimit -c` is 0, and changing either needs root).
// `RZG_CRASH_TRACE=1` installs a SIGSEGV/SIGBUS/SIGILL handler that writes a
// native backtrace to stderr from inside the faulting process.
#[cfg(all(feature = "native", unix))]
mod crash_trace {
    extern "C" {
        fn backtrace(buf: *mut *mut libc::c_void, size: libc::c_int) -> libc::c_int;
        fn backtrace_symbols_fd(buf: *const *mut libc::c_void, size: libc::c_int, fd: libc::c_int);
    }

    unsafe extern "C" fn on_fault(sig: libc::c_int) {
        // async-signal-safe: write(2) + backtrace_symbols_fd only.
        let msg = match sig {
            libc::SIGSEGV => &b"\n[rzg] *** SIGSEGV ***\n"[..],
            libc::SIGBUS => &b"\n[rzg] *** SIGBUS ***\n"[..],
            _ => &b"\n[rzg] *** SIGILL ***\n"[..],
        };
        libc::write(2, msg.as_ptr() as *const libc::c_void, msg.len());
        let mut frames: [*mut libc::c_void; 64] = [std::ptr::null_mut(); 64];
        let n = backtrace(frames.as_mut_ptr(), 64);
        backtrace_symbols_fd(frames.as_ptr(), n, 2);
        // Restore default and re-raise so the exit status is still the signal.
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }

    pub fn install() {
        if std::env::var_os("RZG_CRASH_TRACE").is_none() {
            return;
        }
        unsafe {
            for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL] {
                let h = on_fault as unsafe extern "C" fn(libc::c_int);
                libc::signal(sig, h as usize as libc::sighandler_t);
            }
        }
        eprintln!("[rzg] crash backtrace handler installed");
    }
}

// ---------------------------------------------------------------------------
// Thread-identity probe
// ---------------------------------------------------------------------------
//
// `WgpuContext.pending` is a RefCell and `WgpuBuffer` holds raw Device/Queue/
// Context pointers, both of which assume the GPU is only ever driven from one
// thread. nue runs a SpinPool, so that assumption is load-bearing and was
// never checked. Records the first thread to touch the GPU and shouts if any
// other one does.
pub fn gpu_thread_check(site: &str) {
    // Off unless asked for: this takes a lock on every GPU entry point.
    if std::env::var_os("RZG_GPU_DEBUG").is_none() && std::env::var_os("RZG_CRASH_TRACE").is_none()
    {
        return;
    }
    use std::sync::Mutex;
    static FIRST: Mutex<Option<(std::thread::ThreadId, String)>> = Mutex::new(None);
    let me = std::thread::current().id();
    let name = std::thread::current()
        .name()
        .unwrap_or("<unnamed>")
        .to_string();
    let mut g = match FIRST.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    match g.as_ref() {
        None => {
            eprintln!("[gpu-thread] first touch at {site} on {me:?} ({name})");
            #[cfg(all(feature = "native", unix))]
            crash_trace::install();
            *g = Some((me, name));
        }
        Some((first, fname)) if *first != me => {
            eprintln!(
                "[gpu-thread] !!! CROSS-THREAD at {site}: {me:?} ({name}) != first {first:?} ({fname})"
            );
        }
        _ => {}
    }
}

/// `rayzor_gpu_compute_write_bytes(ctx, buf, addr, byte_size) -> bool`
///
/// Overwrite an existing buffer in place via the queue, so callers can REUSE
/// one allocation instead of creating and destroying per call.
///
/// This matters more than it looks. wgpu's allocator pools freed buffers and
/// does not return them to the OS for the life of the process: routing a
/// prefill's ~224 matmuls, each creating and freeing a 31 MiB weight buffer,
/// held ~3.4 GiB of committed memory (MemAvailable 5953 -> 2503 MiB, recovered
/// only at exit). On a box where the model itself needs 4 GiB that starves
/// decode, which streams every weight per token.
///
/// # Safety
/// `addr` must point to `byte_size` readable bytes, and `byte_size` must not
/// exceed the buffer's size.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_write_bytes(
    ctx: i64,
    buffer_ptr: i64,
    addr: i64,
    byte_size: i64,
) -> bool {
    gpu_thread_check("write_bytes");
    if ctx == 0 || buffer_ptr == 0 || addr == 0 || byte_size <= 0 {
        return false;
    }
    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let buf = &mut *(buffer_ptr as *mut GpuBuffer);
    if buf.ensure_materialized(gpu_ctx).is_err() {
        return false;
    }
    #[cfg(feature = "webgpu-backend")]
    {
        let (NativeContext::Wgpu(wc), NativeBuffer::Wgpu(wb)) =
            (&gpu_ctx.inner, buf.native_buffer().as_ref())
        else {
            return false;
        };
        let n = byte_size as usize;
        if n > wb.byte_size {
            return false;
        }
        // Pending compute may still reference this buffer; submit first so the
        // overwrite cannot race work that has been encoded but not run.
        wc.flush();
        let src = std::slice::from_raw_parts(addr as *const u8, n);
        wc.queue.write_buffer(&wb.buffer, 0, src);
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// `rayzor_gpu_compute_matmul_q4k(ctx, a, bq4, m, k, n) -> GpuBuffer`
///
/// `C[m,n] = A[m,k] * dequant(Bq4)[n,k]^T` with B as raw Q4_K_M blocks.
/// `k` must be a multiple of 256 (the Q4_K super-block size).
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_matmul_q4k(
    ctx: i64,
    a: i64,
    b: i64,
    m: i64,
    k: i64,
    n: i64,
) -> i64 {
    gpu_thread_check("matmul_q4k");
    if ctx == 0 || a == 0 || b == 0 || m <= 0 || k <= 0 || n <= 0 || k % 256 != 0 {
        return 0;
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let a_buf = &mut *(a as *mut GpuBuffer);
    let b_buf = &mut *(b as *mut GpuBuffer);
    if a_buf.ensure_materialized(gpu_ctx).is_err() || b_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }

    let cached = match gpu_ctx.kernel_cache.get_or_compile(
        &gpu_ctx.inner,
        KernelOp::MatmulQ4K,
        buffer::DTYPE_F32,
    ) {
        Ok(kk) => kk,
        Err(_) => return 0,
    };

    #[cfg(feature = "webgpu-backend")]
    {
        use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};
        let (NativeContext::Wgpu(wc), NativeCompiledKernel::Wgpu(kernel)) =
            (&gpu_ctx.inner, &cached.compiled)
        else {
            return 0;
        };
        let (NativeBuffer::Wgpu(aw), NativeBuffer::Wgpu(bw)) = (
            a_buf.native_buffer().as_ref(),
            b_buf.native_buffer().as_ref(),
        ) else {
            return 0;
        };
        let out = match WgpuBuffer::allocate(wc, m * n * 4) {
            Some(o) => o,
            None => return 0,
        };
        // dims.w carries blocks-per-row so the shader can index blocks.
        let dims: [u32; 4] = [m as u32, k as u32, n as u32, (k / 256) as u32];
        let dims_buf = match WgpuBuffer::from_data(wc, dims.as_ptr() as *const u8, 16) {
            Some(d) => d,
            None => return 0,
        };
        let bm = crate::codegen::wgsl_matmul::Q4K_BM as usize;
        let bn = crate::codegen::wgsl_matmul::Q4K_BN as usize;
        if dispatch::dispatch_workgroups(
            wc,
            kernel,
            &[aw, bw, &out, &dims_buf],
            (n.div_ceil(bn), m.div_ceil(bm), 1),
        )
        .is_err()
        {
            return 0;
        }
        let buf = GpuBuffer::materialized(NativeBuffer::Wgpu(out), m * n, buffer::DTYPE_F32);
        return Box::into_raw(Box::new(buf)) as i64;
    }
    #[allow(unreachable_code)]
    0
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

            // The tiled kernel covers a 64x64 output tile per workgroup (256
            // threads x 4x4 outputs each); the simple one covers 16x16. Grid
            // size must follow whichever kernel was emitted.
            let (wg_h, wg_w) = if crate::codegen::wgsl_matmul::use_tiled_matmul() {
                (
                    crate::codegen::wgsl_matmul::tiled_bm() as usize,
                    crate::codegen::wgsl_matmul::tiled_bn() as usize,
                )
            } else {
                (16usize, 16usize)
            };
            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[a_wgpu, b_wgpu, &result_inner, &dims_buf],
                (n.div_ceil(wg_w), m.div_ceil(wg_h), 1),
            )?;

            Ok(NativeBuffer::Wgpu(result_inner))
        }
        #[cfg(feature = "cuda-backend")]
        (NativeContext::Cuda(cuda_ctx), NativeCompiledKernel::Cuda(kernel)) => {
            use crate::cuda::{buffer_ops::CudaBuffer, dispatch};

            let a_cuda = match a_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("a not CUDA".into()),
            };
            let b_cuda = match b_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("b not CUDA".into()),
            };

            let result_inner = CudaBuffer::allocate(cuda_ctx, m * n * elem_size)
                .ok_or("failed to alloc result")?;

            let tile_size = 16u32;
            dispatch::dispatch_grid(
                cuda_ctx,
                kernel,
                &[a_cuda, b_cuda, &result_inner],
                &[m as u32, k as u32, n as u32],
                (
                    (n as u32).div_ceil(tile_size),
                    (m as u32).div_ceil(tile_size),
                    1,
                ),
                (tile_size, tile_size, 1),
                tile_size * tile_size * 4 * 2, // shared memory for A_tile + B_tile
            )?;

            Ok(NativeBuffer::Cuda(result_inner))
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
    batch_matmul_impl(
        ctx,
        a,
        b,
        batch as usize,
        m as usize,
        k as usize,
        n as usize,
    )
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
    let cached =
        match gpu_ctx
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
            let dims_buf =
                unsafe { WgpuBuffer::from_data(wgpu_ctx, dims.as_ptr() as *const u8, 16) }
                    .ok_or("dims alloc failed")?;

            let tg = 16usize;
            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[a_wgpu, b_wgpu, &result_inner, &dims_buf],
                (n.div_ceil(tg), m.div_ceil(tg), batch),
            )?;

            Ok(NativeBuffer::Wgpu(result_inner))
        }
        #[cfg(feature = "cuda-backend")]
        (NativeContext::Cuda(cuda_ctx), NativeCompiledKernel::Cuda(kernel)) => {
            use crate::cuda::{buffer_ops::CudaBuffer, dispatch};

            let a_cuda = match a_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("a not CUDA".into()),
            };
            let b_cuda = match b_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("b not CUDA".into()),
            };

            let result_inner = CudaBuffer::allocate(cuda_ctx, batch * m * n * elem_size)
                .ok_or("failed to alloc result")?;

            let tile_size = 16u32;
            dispatch::dispatch_grid(
                cuda_ctx,
                kernel,
                &[a_cuda, b_cuda, &result_inner],
                &[m as u32, k as u32, n as u32, batch as u32],
                (
                    (n as u32).div_ceil(tile_size),
                    (m as u32).div_ceil(tile_size),
                    batch as u32,
                ),
                (tile_size, tile_size, 1),
                tile_size * tile_size * 4 * 2,
            )?;

            Ok(NativeBuffer::Cuda(result_inner))
        }
        _ => Err("no matching backend for batch matmul".into()),
    }
}

// ---------------------------------------------------------------------------
// Transformer primitives — RmsNorm and Rope (eager dispatch).
//
// Unlike the elementwise lazy ops, these have row-shaped layouts +
// uniform parameters that don't fit cleanly into the fusion graph;
// they materialise their inputs and dispatch immediately, returning a
// fresh materialised result. The WGSL/MSL/CUDA kernel sources live in
// `codegen::{wgsl,msl,cuda}_transformer`.
// ---------------------------------------------------------------------------

/// `y[row, i] = x[row, i] / sqrt(mean(x²) + eps) * weight[i]`. `row_len`
/// is the trailing-dim length (hidden_size); the input is treated as a
/// flat `[groups, row_len]` matrix with `groups = numel / row_len`.
unsafe fn rms_norm_impl(ctx: i64, x: i64, weight: i64, row_len: usize, eps: f32) -> i64 {
    if ctx == 0 || x == 0 || weight == 0 || row_len == 0 {
        return 0;
    }
    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let x_buf = &mut *(x as *mut GpuBuffer);
    let w_buf = &mut *(weight as *mut GpuBuffer);
    if x_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    if w_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    if x_buf.numel == 0 || !x_buf.numel.is_multiple_of(row_len) {
        return 0;
    }
    let groups = x_buf.numel / row_len;
    let dtype = x_buf.dtype;
    let cached = match gpu_ctx
        .kernel_cache
        .get_or_compile(&gpu_ctx.inner, KernelOp::RmsNorm, dtype)
    {
        Ok(k) => k,
        Err(_) => return 0,
    };
    let elem_size = buffer::dtype_byte_size(dtype);

    match rms_norm_dispatch(
        &gpu_ctx.inner,
        &cached.compiled,
        x_buf.native_buffer(),
        w_buf.native_buffer(),
        x_buf.numel,
        row_len,
        groups,
        eps,
        elem_size,
    ) {
        Ok(result_native) => {
            let result = GpuBuffer::materialized(result_native, x_buf.numel, dtype);
            Box::into_raw(Box::new(result)) as i64
        }
        Err(_) => 0,
    }
}

#[allow(unused_variables, clippy::too_many_arguments)]
fn rms_norm_dispatch(
    ctx: &NativeContext,
    compiled: &NativeCompiledKernel,
    x_buf: &Rc<NativeBuffer>,
    w_buf: &Rc<NativeBuffer>,
    numel: usize,
    row_len: usize,
    groups: usize,
    eps: f32,
    elem_size: usize,
) -> Result<NativeBuffer, String> {
    // RMSNorm uniform — shared across every backend. Field order matches
    // the struct layout in the WGSL / MSL / CUDA shaders.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct RmsParams {
        row_len: u32,
        eps: f32,
    }
    let params = RmsParams {
        row_len: row_len as u32,
        eps,
    };

    match (ctx, compiled) {
        #[cfg(feature = "metal-backend")]
        (NativeContext::Metal(metal_ctx), NativeCompiledKernel::Metal(kernel)) => {
            use crate::metal::{buffer_ops::MetalBuffer, dispatch};
            use objc2_metal::MTLSize;

            let x_metal = match x_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("x not Metal".into()),
            };
            let w_metal = match w_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("weight not Metal".into()),
            };

            let result = MetalBuffer::allocate(metal_ctx, numel * elem_size)
                .ok_or("failed to alloc rmsnorm result")?;
            let params_buf = MetalBuffer::from_value(metal_ctx, &params)
                .ok_or("failed to alloc rmsnorm params")?;

            // One threadgroup per row; 256 threads per group matches
            // TG_SIZE in the MSL shader (msl_transformer::emit_rms_norm).
            dispatch::dispatch_threadgroups(
                metal_ctx,
                kernel,
                &[x_metal, w_metal, &result, &params_buf],
                MTLSize {
                    width: groups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )?;
            Ok(NativeBuffer::Metal(result))
        }
        #[cfg(feature = "webgpu-backend")]
        (NativeContext::Wgpu(wgpu_ctx), NativeCompiledKernel::Wgpu(kernel)) => {
            use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};

            let x_wgpu = match x_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("x not wgpu".into()),
            };
            let w_wgpu = match w_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("weight not wgpu".into()),
            };

            let result = WgpuBuffer::allocate(wgpu_ctx, numel * elem_size)
                .ok_or("failed to alloc rmsnorm result")?;
            let params_buf = unsafe {
                WgpuBuffer::from_data(
                    wgpu_ctx,
                    &params as *const _ as *const u8,
                    std::mem::size_of::<RmsParams>(),
                )
            }
            .ok_or("failed to alloc rmsnorm params")?;

            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[x_wgpu, w_wgpu, &result, &params_buf],
                (groups, 1, 1),
            )?;
            Ok(NativeBuffer::Wgpu(result))
        }
        #[cfg(feature = "cuda-backend")]
        (NativeContext::Cuda(cuda_ctx), NativeCompiledKernel::Cuda(kernel)) => {
            use crate::cuda::{buffer_ops::CudaBuffer, dispatch};

            let x_cuda = match x_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("x not CUDA".into()),
            };
            let w_cuda = match w_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("weight not CUDA".into()),
            };

            let result = CudaBuffer::allocate(cuda_ctx, numel * elem_size)
                .ok_or("failed to alloc rmsnorm result")?;

            // CUDA's dispatch_grid passes packed u32 params through the
            // params slice; our shader reads them as RmsParams struct.
            dispatch::dispatch_grid(
                cuda_ctx,
                kernel,
                &[x_cuda, w_cuda, &result],
                &[params.row_len, params.eps.to_bits()],
                (groups as u32, 1, 1),
                (256, 1, 1),
                /* shared bytes */ 256 * 4 + 4, // partial[LANES] + shared_inv_rms
            )?;
            Ok(NativeBuffer::Cuda(result))
        }
        _ => Err("rmsnorm: no matching backend".into()),
    }
}

/// Apply RoPE to `x [seq_len, num_heads, head_dim]` using precomputed
/// cos/sin LUTs. `position_offset` shifts the logical position used
/// for table lookup (0 for prefill, positive for decode).
#[allow(clippy::too_many_arguments)]
unsafe fn rope_impl(
    ctx: i64,
    x: i64,
    cos: i64,
    sin: i64,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: u32,
    cos_max_seq: usize,
) -> i64 {
    if ctx == 0 || x == 0 || cos == 0 || sin == 0 {
        return 0;
    }
    if seq_len == 0 || num_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
        return 0;
    }
    let gpu_ctx = &mut *(ctx as *mut GpuContext);
    let x_buf = &mut *(x as *mut GpuBuffer);
    let cos_buf = &mut *(cos as *mut GpuBuffer);
    let sin_buf = &mut *(sin as *mut GpuBuffer);
    if x_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    if cos_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    if sin_buf.ensure_materialized(gpu_ctx).is_err() {
        return 0;
    }
    let dtype = x_buf.dtype;
    let cached = match gpu_ctx
        .kernel_cache
        .get_or_compile(&gpu_ctx.inner, KernelOp::Rope, dtype)
    {
        Ok(k) => k,
        Err(_) => return 0,
    };
    let elem_size = buffer::dtype_byte_size(dtype);
    let numel = seq_len * num_heads * head_dim;

    match rope_dispatch(
        &gpu_ctx.inner,
        &cached.compiled,
        x_buf.native_buffer(),
        cos_buf.native_buffer(),
        sin_buf.native_buffer(),
        seq_len,
        num_heads,
        head_dim,
        position_offset,
        cos_max_seq,
        elem_size,
    ) {
        Ok(result_native) => {
            let result = GpuBuffer::materialized(result_native, numel, dtype);
            Box::into_raw(Box::new(result)) as i64
        }
        Err(_) => 0,
    }
}

#[allow(unused_variables, clippy::too_many_arguments)]
fn rope_dispatch(
    ctx: &NativeContext,
    compiled: &NativeCompiledKernel,
    x_buf: &Rc<NativeBuffer>,
    cos_buf: &Rc<NativeBuffer>,
    sin_buf: &Rc<NativeBuffer>,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    position_offset: u32,
    cos_max_seq: usize,
    elem_size: usize,
) -> Result<NativeBuffer, String> {
    // Shared uniform layout — matches the RopeParams struct in every
    // backend's shader (wgsl_transformer / msl_transformer / cuda_transformer).
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct RopeParams {
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        position_offset: u32,
        cos_max_seq: u32,
    }
    let params = RopeParams {
        seq_len: seq_len as u32,
        num_heads: num_heads as u32,
        head_dim: head_dim as u32,
        position_offset,
        cos_max_seq: cos_max_seq as u32,
    };
    let numel = seq_len * num_heads * head_dim;
    let half_dim = head_dim / 2;
    let lanes = seq_len * num_heads * half_dim;
    let wg_size = 256usize;

    match (ctx, compiled) {
        #[cfg(feature = "metal-backend")]
        (NativeContext::Metal(metal_ctx), NativeCompiledKernel::Metal(kernel)) => {
            use crate::metal::{buffer_ops::MetalBuffer, dispatch};
            use objc2_metal::MTLSize;

            let x_metal = match x_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("x not Metal".into()),
            };
            let cos_metal = match cos_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("cos not Metal".into()),
            };
            let sin_metal = match sin_buf.as_ref() {
                NativeBuffer::Metal(mb) => mb,
                _ => return Err("sin not Metal".into()),
            };

            let result = MetalBuffer::allocate(metal_ctx, numel * elem_size)
                .ok_or("failed to alloc rope result")?;
            let params_buf =
                MetalBuffer::from_value(metal_ctx, &params).ok_or("failed to alloc rope params")?;

            // Flat 1-D dispatch — `thread_position_in_grid` gives the
            // per-lane index inside the MSL shader.
            dispatch::dispatch_threadgroups(
                metal_ctx,
                kernel,
                &[x_metal, cos_metal, sin_metal, &result, &params_buf],
                MTLSize {
                    width: lanes.div_ceil(wg_size),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: wg_size,
                    height: 1,
                    depth: 1,
                },
            )?;
            Ok(NativeBuffer::Metal(result))
        }
        #[cfg(feature = "webgpu-backend")]
        (NativeContext::Wgpu(wgpu_ctx), NativeCompiledKernel::Wgpu(kernel)) => {
            use crate::wgpu_backend::{buffer_ops::WgpuBuffer, dispatch};

            let x_wgpu = match x_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("x not wgpu".into()),
            };
            let cos_wgpu = match cos_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("cos not wgpu".into()),
            };
            let sin_wgpu = match sin_buf.as_ref() {
                NativeBuffer::Wgpu(wb) => wb,
                _ => return Err("sin not wgpu".into()),
            };

            let result = WgpuBuffer::allocate(wgpu_ctx, numel * elem_size)
                .ok_or("failed to alloc rope result")?;
            let params_buf = unsafe {
                WgpuBuffer::from_data(
                    wgpu_ctx,
                    &params as *const _ as *const u8,
                    std::mem::size_of::<RopeParams>(),
                )
            }
            .ok_or("failed to alloc rope params")?;

            dispatch::dispatch_workgroups(
                wgpu_ctx,
                kernel,
                &[x_wgpu, cos_wgpu, sin_wgpu, &result, &params_buf],
                (lanes.div_ceil(wg_size), 1, 1),
            )?;
            Ok(NativeBuffer::Wgpu(result))
        }
        #[cfg(feature = "cuda-backend")]
        (NativeContext::Cuda(cuda_ctx), NativeCompiledKernel::Cuda(kernel)) => {
            use crate::cuda::{buffer_ops::CudaBuffer, dispatch};

            let x_cuda = match x_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("x not CUDA".into()),
            };
            let cos_cuda = match cos_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("cos not CUDA".into()),
            };
            let sin_cuda = match sin_buf.as_ref() {
                NativeBuffer::Cuda(cb) => cb,
                _ => return Err("sin not CUDA".into()),
            };

            let result = CudaBuffer::allocate(cuda_ctx, numel * elem_size)
                .ok_or("failed to alloc rope result")?;

            dispatch::dispatch_grid(
                cuda_ctx,
                kernel,
                &[x_cuda, cos_cuda, sin_cuda, &result],
                &[
                    params.seq_len,
                    params.num_heads,
                    params.head_dim,
                    params.position_offset,
                    params.cos_max_seq,
                ],
                ((lanes as u32).div_ceil(wg_size as u32), 1, 1),
                (wg_size as u32, 1, 1),
                /* no shared memory needed */ 0,
            )?;
            Ok(NativeBuffer::Cuda(result))
        }
        _ => Err("rope: no matching backend".into()),
    }
}

// ---------------------------------------------------------------------------
// Transformer primitives — FFI entry points
// ---------------------------------------------------------------------------

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_compute_rms_norm(
    ctx: i64,
    x: i64,
    weight: i64,
    row_len: i64,
    eps: f64,
) -> i64 {
    rms_norm_impl(ctx, x, weight, row_len as usize, eps as f32)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rayzor_gpu_compute_rope(
    ctx: i64,
    x: i64,
    cos: i64,
    sin: i64,
    seq_len: i64,
    num_heads: i64,
    head_dim: i64,
    position_offset: i64,
    cos_max_seq: i64,
) -> i64 {
    rope_impl(
        ctx,
        x,
        cos,
        sin,
        seq_len as usize,
        num_heads as usize,
        head_dim as usize,
        position_offset.max(0) as u32,
        cos_max_seq as usize,
    )
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
    fn test_gpu_buffer_op_overload() {
        // Verifies the rayzor_gpu_buffer_* (no-ctx) entry points used by
        // GpuBuffer @:op overloading produce the same result as the existing
        // rayzor_gpu_compute_* (with-ctx) ops.
        let ctx = make_ctx();
        if ctx == 0 {
            return;
        }

        let n = 256;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();

        let a_buf = unsafe { create_test_buffer(ctx, &a_data) };
        let b_buf = unsafe { create_test_buffer(ctx, &b_data) };

        // a + b → a*3 elementwise
        let add_result = unsafe { rayzor_gpu_buffer_add(a_buf, b_buf) };
        // a * b → i * 2*i = 2i²
        let mul_result = unsafe { rayzor_gpu_buffer_mul(a_buf, b_buf) };
        // b - a → i
        let sub_result = unsafe { rayzor_gpu_buffer_sub(b_buf, a_buf) };

        for &result in &[add_result, mul_result, sub_result] {
            assert_ne!(result, 0, "buffer op returned null");
            let result_buf = unsafe { &*(result as *const GpuBuffer) };
            assert!(
                matches!(result_buf.kind, GpuBufferKind::Lazy(_)),
                "buffer op result should be lazy"
            );
        }

        let gpu_ctx = unsafe { &mut *(ctx as *mut GpuContext) };
        let add_buf = unsafe { &mut *(add_result as *mut GpuBuffer) };
        add_buf.ensure_materialized(gpu_ctx).unwrap();
        let byte_size = n * 4;
        let data = add_buf.native_buffer().read_bytes(byte_size).unwrap();
        let add_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        for (i, &val) in add_slice.iter().enumerate() {
            let expected = (i + i * 2) as f32;
            assert!(
                (val - expected).abs() < 1e-6,
                "buffer add mismatch at {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        let mul_buf = unsafe { &mut *(mul_result as *mut GpuBuffer) };
        mul_buf.ensure_materialized(gpu_ctx).unwrap();
        let data = mul_buf.native_buffer().read_bytes(byte_size).unwrap();
        let mul_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        for (i, &val) in mul_slice.iter().enumerate() {
            let expected = (i * i * 2) as f32;
            assert!(
                (val - expected).abs() < 1e-3,
                "buffer mul mismatch at {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        let sub_buf = unsafe { &mut *(sub_result as *mut GpuBuffer) };
        sub_buf.ensure_materialized(gpu_ctx).unwrap();
        let data = sub_buf.native_buffer().read_bytes(byte_size).unwrap();
        let sub_slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        for (i, &val) in sub_slice.iter().enumerate() {
            let expected = i as f32;
            assert!(
                (val - expected).abs() < 1e-6,
                "buffer sub mismatch at {}: expected {}, got {}",
                i,
                expected,
                val
            );
        }

        unsafe {
            let _ = Box::from_raw(add_result as *mut GpuBuffer);
            let _ = Box::from_raw(mul_result as *mut GpuBuffer);
            let _ = Box::from_raw(sub_result as *mut GpuBuffer);
            let _ = Box::from_raw(a_buf as *mut GpuBuffer);
            let _ = Box::from_raw(b_buf as *mut GpuBuffer);
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
