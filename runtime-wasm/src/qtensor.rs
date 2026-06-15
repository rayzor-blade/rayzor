//! WASM-side quantised tensor lifetime + Q4_K_M × F32 matmul.
//!
//! Phase 3 of the WASM runtime parity plan
//! (docs/design/wasm_runtime_parity.md). Mirrors the first eight fields of
//! `rayzor-runtime::quant::RayzorQTensor` so that downstream readers see
//! the same offsets for `data`, `meta`, `numel`, `group_size`, `scheme`,
//! `owns_data`, `rows`, `cols`.
//!
//! The Q4_K_M matmul routes through `rayzor-runtime-core::quant::q4_k_m::
//! vec_dot_q4_K_q8_K_scalar` — the portable scalar reference that the
//! native crate uses when the AArch64 SDOT gate is off (Cortex-A53, etc).
//! On wasm32 with `+simd128 +relaxed-simd` LLVM auto-vectorises the inner
//! loop's per-byte 4-bit-mask + i32 accumulate; a dedicated
//! `i8x16.dot_i8x16_i7x16`-style inner is Phase 8 territory once the
//! relaxed-simd proposal ships in browsers.

use core::slice;
use rayzor_runtime_core::quant::matmul::prepare_x_q8k_blocks_into;
use rayzor_runtime_core::quant::q4_k_m::{
    decode_q4_k_block, dequant_q4_k_block, vec_dot_q4_K_q8_K,
};
use rayzor_runtime_core::quant::q6_k::{dequant_q6_k_block, vec_dot_q6_K_q8_K};
use rayzor_runtime_core::quant::types::{
    Q4KMBlock, Q8KBlock, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_SIZE, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_SIZE,
    QSCHEME_Q4_K_M, QSCHEME_Q6_K,
};
use std::alloc::{alloc, dealloc, Layout};

use crate::tensor::{Tensor, DTYPE_F32};

// Host-provided parallel quant matmul. The embedder (`compiler/src/codegen/
// wasm_runner.rs`) runs the output-column loop on the native worker pool over
// the shared linear memory while this single guest thread blocks in the call —
// the wasm "threading" lever. Guest passes raw wasm byte-offsets + dims; host
// prequantises `x`→Q8_K and fans `n` columns across OS threads. Returns 1 when
// handled, 0 to fall back to the sequential guest path.
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "rayzor")]
extern "C" {
    fn rayzor_host_qmatmul_par(
        x_data_off: i32,
        x_stride0: i32,
        batch: i32,
        k: i32,
        qt_data_off: i32,
        scheme: i32,
        n: i32,
        y_data_off: i32,
        threads: i32,
    ) -> i32;
}

#[repr(C)]
struct HaxeBytes {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    kind: u8,
    _pad: [u8; 7],
    owner: *mut HaxeBytes,
    refcount: i64,
    fd: i32,
    _pad2: [u8; 4],
}

/// Minimal QTensor wrapper. `#[repr(C)]` so the layout matches the native
/// runtime's first eight fields exactly. Refcount, view-parent, and the
/// extra metadata stay deferred to Phase 4.
#[repr(C)]
pub struct QTensor {
    pub data: *mut u8,
    pub meta: *mut f32, // nullable; INT8 scales OR null for Q4_K_M
    pub numel: usize,
    pub group_size: usize,
    pub scheme: u8,
    pub owns_data: bool,
    pub rows: usize,
    pub cols: usize,
}

fn q4_k_m_data_bytes(numel: usize) -> usize {
    (numel / Q4_K_M_BLOCK_SIZE) * Q4_K_M_BLOCK_BYTES
}

fn q6_k_data_bytes(numel: usize) -> usize {
    (numel / Q6_K_BLOCK_SIZE) * Q6_K_BLOCK_BYTES
}

fn qscheme_block_shape(scheme: u8) -> Option<(usize, usize)> {
    match scheme {
        QSCHEME_Q4_K_M => Some((Q4_K_M_BLOCK_SIZE, Q4_K_M_BLOCK_BYTES)),
        QSCHEME_Q6_K => Some((Q6_K_BLOCK_SIZE, Q6_K_BLOCK_BYTES)),
        _ => None,
    }
}

fn qscheme_data_bytes(scheme: u8, numel: usize) -> usize {
    match scheme {
        QSCHEME_Q4_K_M => q4_k_m_data_bytes(numel),
        QSCHEME_Q6_K => q6_k_data_bytes(numel),
        _ => 0,
    }
}

unsafe fn bytes_slice(bytes_handle: i32, needed: usize) -> Option<&'static [u8]> {
    if bytes_handle == 0 {
        return None;
    }
    let bytes = &*(bytes_handle as *const HaxeBytes);
    if bytes.ptr.is_null() || bytes.len < needed {
        return None;
    }
    Some(slice::from_raw_parts(bytes.ptr as *const u8, needed))
}

unsafe fn alloc_qtensor(scheme: u8, rows: usize, cols: usize) -> *mut QTensor {
    let Some((block_size, _)) = qscheme_block_shape(scheme) else {
        return core::ptr::null_mut();
    };
    if !cols.is_multiple_of(block_size) {
        return core::ptr::null_mut();
    }
    let numel = rows * cols;
    let data_bytes = qscheme_data_bytes(scheme, numel);

    let data_layout = Layout::from_size_align(data_bytes.max(1), 16).unwrap();
    let data = alloc(data_layout);
    if data.is_null() {
        return core::ptr::null_mut();
    }
    core::ptr::write_bytes(data, 0, data_bytes);

    let wrapper_layout = Layout::new::<QTensor>();
    let wrapper = alloc(wrapper_layout) as *mut QTensor;
    if wrapper.is_null() {
        dealloc(data, data_layout);
        return core::ptr::null_mut();
    }
    core::ptr::write(
        wrapper,
        QTensor {
            data,
            meta: core::ptr::null_mut(),
            numel,
            group_size: block_size,
            scheme,
            owns_data: true,
            rows,
            cols,
        },
    );
    wrapper
}

/// Construct a Q4_K_M `QTensor` from raw bytes. `rows × cols` weights, with
/// `cols` a multiple of `Q4_K_M_BLOCK_SIZE`. The byte buffer at `bytes_ptr`
/// holds `rows × (cols / 256) × 144` bytes of canonical GGUF Q4_K_M blocks,
/// and is COPIED into the QTensor's owned data buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_bytes_q4_k_m(
    bytes_handle: i32,
    rows: i32,
    cols: i32,
) -> i32 {
    if bytes_handle == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let numel = rows * cols;
    let needed = q4_k_m_data_bytes(numel);
    let Some(bytes) = bytes_slice(bytes_handle, needed) else {
        return 0;
    };
    let qt = alloc_qtensor(QSCHEME_Q4_K_M, rows, cols);
    if qt.is_null() {
        return 0;
    }
    let qt_ref = &*qt;
    core::ptr::copy_nonoverlapping(bytes.as_ptr(), qt_ref.data, needed);
    qt as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_bytes_q6_k(
    bytes_handle: i32,
    rows: i32,
    cols: i32,
) -> i32 {
    if bytes_handle == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let numel = rows * cols;
    if !numel.is_multiple_of(Q6_K_BLOCK_SIZE) {
        return 0;
    }
    let needed = q6_k_data_bytes(numel);
    let Some(bytes) = bytes_slice(bytes_handle, needed) else {
        return 0;
    };
    let qt = alloc_qtensor(QSCHEME_Q6_K, rows, cols);
    if qt.is_null() {
        return 0;
    }
    let qt_ref = &*qt;
    core::ptr::copy_nonoverlapping(bytes.as_ptr(), qt_ref.data, needed);
    qt as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_rows(qt: i32) -> i32 {
    if qt == 0 {
        return 0;
    }
    (*(qt as *const QTensor)).rows as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_cols(qt: i32) -> i32 {
    if qt == 0 {
        return 0;
    }
    (*(qt as *const QTensor)).cols as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_scheme(qt: i32) -> i32 {
    if qt == 0 {
        return 0;
    }
    (*(qt as *const QTensor)).scheme as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_numel(qt: i32) -> i32 {
    if qt == 0 {
        return 0;
    }
    (*(qt as *const QTensor)).numel as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_free(qt: i32) {
    if qt == 0 {
        return;
    }
    let qt_ref = &*(qt as *const QTensor);
    let data = qt_ref.data;
    let owns = qt_ref.owns_data;
    let numel = qt_ref.numel;

    if owns && !data.is_null() {
        let bytes = qscheme_data_bytes(qt_ref.scheme, numel);
        let data_layout = Layout::from_size_align(bytes.max(1), 16).unwrap();
        dealloc(data, data_layout);
    }
    let wrapper_layout = Layout::new::<QTensor>();
    dealloc(qt as *mut u8, wrapper_layout);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_dequant(qt: i32) -> i32 {
    if qt == 0 {
        return 0;
    }
    let qtr = &*(qt as *const QTensor);
    let shape = [qtr.rows, qtr.cols];
    let out = crate::tensor::alloc_tensor(&shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let out_data = (*out).data as *mut f32;
    match qtr.scheme {
        QSCHEME_Q4_K_M => {
            let blocks_per_row = qtr.cols / Q4_K_M_BLOCK_SIZE;
            let mut stage = [0.0f32; Q4_K_M_BLOCK_SIZE];
            for r in 0..qtr.rows {
                let row_ptr = qtr.data.add(r * blocks_per_row * Q4_K_M_BLOCK_BYTES);
                for b in 0..blocks_per_row {
                    let block = decode_q4_k_block(row_ptr.add(b * Q4_K_M_BLOCK_BYTES));
                    dequant_q4_k_block(&block, &mut stage);
                    core::ptr::copy_nonoverlapping(
                        stage.as_ptr(),
                        out_data.add(r * qtr.cols + b * Q4_K_M_BLOCK_SIZE),
                        Q4_K_M_BLOCK_SIZE,
                    );
                }
            }
        }
        QSCHEME_Q6_K => {
            let blocks_per_row = qtr.cols / Q6_K_BLOCK_SIZE;
            let mut stage = [0.0f32; Q6_K_BLOCK_SIZE];
            for r in 0..qtr.rows {
                let row_ptr = qtr.data.add(r * blocks_per_row * Q6_K_BLOCK_BYTES);
                for b in 0..blocks_per_row {
                    dequant_q6_k_block(row_ptr.add(b * Q6_K_BLOCK_BYTES), &mut stage);
                    core::ptr::copy_nonoverlapping(
                        stage.as_ptr(),
                        out_data.add(r * qtr.cols + b * Q6_K_BLOCK_SIZE),
                        Q6_K_BLOCK_SIZE,
                    );
                }
            }
        }
        _ => return 0,
    }
    out as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows_q6_k(
    qt: i32,
    indices_ptr: i32,
    n_indices: i32,
) -> i32 {
    if qt == 0 || indices_ptr == 0 || n_indices <= 0 {
        return 0;
    }
    let qtr = &*(qt as *const QTensor);
    if qtr.scheme != QSCHEME_Q6_K || !qtr.cols.is_multiple_of(Q6_K_BLOCK_SIZE) {
        return 0;
    }
    let k = n_indices as usize;
    let out_shape = [k, qtr.cols];
    let out = crate::tensor::alloc_tensor(&out_shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let out_data = (*out).data as *mut f32;
    let indices = indices_ptr as *const i64;
    let blocks_per_row = qtr.cols / Q6_K_BLOCK_SIZE;
    let row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
    let mut stage = [0.0f32; Q6_K_BLOCK_SIZE];
    for i in 0..k {
        let idx = *indices.add(i);
        if idx < 0 || idx as usize >= qtr.rows {
            continue;
        }
        let row_src = qtr.data.add(idx as usize * row_bytes);
        let row_dst = out_data.add(i * qtr.cols);
        for b in 0..blocks_per_row {
            dequant_q6_k_block(row_src.add(b * Q6_K_BLOCK_BYTES), &mut stage);
            core::ptr::copy_nonoverlapping(
                stage.as_ptr(),
                row_dst.add(b * Q6_K_BLOCK_SIZE),
                Q6_K_BLOCK_SIZE,
            );
        }
    }
    out as i32
}

unsafe fn qmatmul_shape(x: i32, qt: &QTensor) -> Option<(usize, usize, usize, usize, usize)> {
    let (block_size, block_bytes) = qscheme_block_shape(qt.scheme)?;
    let xr = &*(x as *const Tensor);
    if xr.ndim != 2 || xr.dtype != DTYPE_F32 {
        return None;
    }
    let shape = slice::from_raw_parts(xr.shape, 2);
    let batch = shape[0];
    let k = shape[1];
    if k != qt.cols || !k.is_multiple_of(block_size) {
        return None;
    }
    Some((batch, qt.rows, k, block_size, block_bytes))
}

unsafe fn qmatmul_chunk_impl(x: i32, qt: i32, y: i32, n_start: i32, n_end: i32) {
    let xr = &*(x as *const Tensor);
    let qtr = &*(qt as *const QTensor);
    let Some((batch, n, k, block_size, block_bytes)) = qmatmul_shape(x, qtr) else {
        return;
    };
    let x_shape = slice::from_raw_parts(xr.shape, 2);
    let x_strides = slice::from_raw_parts(xr.strides, 2);
    let y_data = (*(y as *const Tensor)).data as *mut f32;
    let x_data = xr.data as *const f32;
    let blocks_per_row = k / block_size;
    let lo = (n_start.max(0) as usize).min(n);
    let hi = (n_end.max(0) as usize).min(n);
    let mut stage = [0.0f32; 256];
    let mut x_q8k: std::vec::Vec<Q8KBlock> = std::vec::Vec::with_capacity(blocks_per_row);

    for bch in 0..batch {
        let x_base = if x_strides[1] == 1 {
            x_data.add(bch * x_strides[0])
        } else {
            x_data.add(bch * x_shape[1])
        };
        // Fast path: prequant x→Q8_K once per row, then an integer SIMD dot
        // per block. Both Q4_K_M and Q6_K use 256-elem blocks and the same
        // Q8_K activation, so Q6_K (attn_v / ffn_down) now gets the same
        // vectorized treatment instead of a per-column full f32 dequant.
        let fast_path =
            (qtr.scheme == QSCHEME_Q4_K_M || qtr.scheme == QSCHEME_Q6_K) && x_strides[1] == 1;
        if fast_path {
            prepare_x_q8k_blocks_into(x_base, k, &mut x_q8k);
        }
        for n_idx in lo..hi {
            let row_ptr = qtr.data.add(n_idx * blocks_per_row * block_bytes);
            let mut sum = 0.0f32;
            for blk in 0..blocks_per_row {
                let bp = row_ptr.add(blk * block_bytes);
                if fast_path {
                    // blk < blocks_per_row == x_q8k.len() (prepared above);
                    // skip the per-block bounds check on the hot path.
                    let xb = x_q8k.get_unchecked(blk);
                    match qtr.scheme {
                        QSCHEME_Q4_K_M => {
                            let weight = &*(bp as *const Q4KMBlock);
                            sum += vec_dot_q4_K_q8_K(weight, xb);
                        }
                        _ => {
                            sum += vec_dot_q6_K_q8_K(bp, xb);
                        }
                    }
                } else {
                    match qtr.scheme {
                        QSCHEME_Q4_K_M => {
                            let block = decode_q4_k_block(bp);
                            dequant_q4_k_block(&block, &mut stage);
                        }
                        QSCHEME_Q6_K => dequant_q6_k_block(bp, &mut stage),
                        _ => return,
                    }
                    for kk in 0..block_size {
                        let xv = if x_strides[1] == 1 {
                            *x_base.add(blk * block_size + kk)
                        } else {
                            *x_data.add(bch * x_strides[0] + (blk * block_size + kk) * x_strides[1])
                        };
                        sum += xv * stage[kk];
                    }
                }
            }
            *y_data.add(bch * n + n_idx) = sum;
        }
    }
}

/// Allocation-free per-band dot for the threaded guest matmul: dots output rows
/// `lo..hi` of a Q4_K_M / Q6_K weight matrix against PRE-PREPARED Q8_K
/// activations for all `batch` rows (`x_q8k`, laid out as
/// `[batch][blocks_per_row]`, shared read-only across workers). No heap use, so
/// it runs safely on `rayzor_thread_spawn` worker instances (no libc TLS
/// needed). Weight row is loaded once per `n_idx` and reused across the batch
/// (decode is `batch == 1`; prefill drives `batch == prompt_len`).
#[cfg(target_arch = "wasm32")]
unsafe fn qmatmul_band_dot(
    qt: i32,
    y: i32,
    batch: usize,
    n: usize,
    blocks_per_row: usize,
    block_bytes: usize,
    x_q8k: *const Q8KBlock,
    lo: usize,
    hi: usize,
) {
    let qtr = &*(qt as *const QTensor);
    let y_data = (*(y as *const Tensor)).data as *mut f32;
    for n_idx in lo..hi {
        let row_ptr = qtr.data.add(n_idx * blocks_per_row * block_bytes);
        for bch in 0..batch {
            let xq_row = x_q8k.add(bch * blocks_per_row);
            let mut sum = 0.0f32;
            for blk in 0..blocks_per_row {
                let bp = row_ptr.add(blk * block_bytes);
                let xb = &*xq_row.add(blk);
                match qtr.scheme {
                    QSCHEME_Q4_K_M => sum += vec_dot_q4_K_q8_K(&*(bp as *const Q4KMBlock), xb),
                    _ => sum += vec_dot_q6_K_q8_K(bp, xb),
                }
            }
            *y_data.add(bch * n + n_idx) = sum;
        }
    }
}

/// `RAYZOR_WASM_GUEST_THREADS` parsed once (0 = unset → use the default).
/// Reading env per matmul call (hundreds per token) would allocate + syscall;
/// cache it. WASI passes host env through `inherit_env`.
#[cfg(target_arch = "wasm32")]
pub(crate) fn guest_threads_override() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("RAYZOR_WASM_GUEST_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Per-worker job: which output-row band to dot, against the shared `x_q8k`.
#[cfg(target_arch = "wasm32")]
#[repr(C)]
struct BandWork {
    qt: i32,
    y: i32,
    batch: usize,
    n: usize,
    blocks_per_row: usize,
    block_bytes: usize,
    x_q8k: i32,
    lo: usize,
    hi: usize,
}

/// Worker entry, invoked on a `rayzor_thread_spawn` OS thread via the indirect
/// function table. Reads its `BandWork` from shared memory and dots its band.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn rayzor_qmatmul_band_worker(work_ptr: i32) -> i64 {
    unsafe {
        let w = &*(work_ptr as *const BandWork);
        qmatmul_band_dot(
            w.qt,
            w.y,
            w.batch,
            w.n,
            w.blocks_per_row,
            w.block_bytes,
            w.x_q8k as *const Q8KBlock,
            w.lo,
            w.hi,
        );
    }
    0
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "rayzor")]
extern "C" {
    fn rayzor_thread_spawn(fn_idx: i32, env_ptr: i32) -> i32;
    // Void join: blocks on the worker, discards its (always-0) result without
    // boxing. The boxing `rayzor_thread_join` is the Thread<T>.join() contract;
    // the matmul needs neither the box nor the result, just the barrier.
    fn rayzor_thread_join_void(handle: i32);
}

/// `Y = X @ Wq^T` for a Q4_K_M weight matrix `Wq [n, k]` and a contiguous
/// F32 activation `X [m, k]`. Returns a freshly allocated `[m, n]` F32
/// `Tensor`.
///
/// Inner per-row loop:
///   1. Pre-quantise the X row into `Q8KBlock`s once (reuses a single
///      `Vec<Q8KBlock>` scratch across all output rows of one matmul).
///   2. Per output column j and per K-block b, read the matching Q4_K_M
///      super-block and dispatch `vec_dot_q4_K_q8_K_scalar(weight, x_q8k)`.
///   3. Accumulate per-block partials into the output element.
///
/// Threading is sequential in Phase 3. Phase 6 (Web Workers / wasi-threads)
/// will wrap the per-output-row loop in a `parallel_rows` worker pool.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qt_t_f32(x: i32, qt: i32) -> i32 {
    if x == 0 || qt == 0 {
        return 0;
    }
    let qtr = &*(qt as *const QTensor);
    let Some((batch, n_out, _k, _block_size, _block_bytes)) = qmatmul_shape(x, qtr) else {
        return 0;
    };
    let y_shape: [usize; 2] = [batch, n_out];
    let y = crate::tensor::alloc_tensor(&y_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }
    let y = y as i32;
    qmatmul_chunk_impl(x, qt, y, 0, n_out as i32);
    y
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qt_t_f32_chunk(
    x: i32,
    qt: i32,
    y: i32,
    n_start: i32,
    n_end: i32,
) -> i32 {
    if x == 0 || qt == 0 || y == 0 {
        return 0;
    }
    qmatmul_chunk_impl(x, qt, y, n_start, n_end);
    1
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qt_t_f32_threaded(
    x: i32,
    qt: i32,
    threads: i32,
) -> i32 {
    if x == 0 || qt == 0 {
        return 0;
    }
    let qtr = &*(qt as *const QTensor);
    let Some((batch, n_out, _k, _block_size, _block_bytes)) = qmatmul_shape(x, qtr) else {
        return 0;
    };
    let y_shape: [usize; 2] = [batch, n_out];
    let y = crate::tensor::alloc_tensor(&y_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }
    let y = y as i32;

    // Host-parallel fast path: offload the embarrassingly-parallel output-column
    // loop to native worker threads over the shared linear memory. Wired for the
    // contiguous Q4_K_M / Q6_K case (the schemes `prepare_x_q8k_blocks_into`
    // covers); anything else falls through to the sequential guest kernel.
    #[cfg(target_arch = "wasm32")]
    {
        let xr = &*(x as *const Tensor);
        let x_strides = slice::from_raw_parts(xr.strides, 2);
        if (qtr.scheme == QSCHEME_Q4_K_M || qtr.scheme == QSCHEME_Q6_K) && x_strides[1] == 1 {
            let handled = rayzor_host_qmatmul_par(
                xr.data as usize as i32,
                x_strides[0] as i32,
                batch as i32,
                _k as i32,
                qtr.data as usize as i32,
                qtr.scheme as i32,
                n_out as i32,
                (*(y as *const Tensor)).data as usize as i32,
                threads,
            );
            if handled != 0 {
                return y;
            }
        }
    }

    // In-guest parallel fast path (used when the host pool is off, e.g.
    // RAYZOR_WASM_NO_HOST_MATMUL=1, or in a pure-wasm host with no native
    // pool): prequantize x→Q8_K ONCE on this thread, then split the output
    // rows across `rayzor_thread_spawn` worker threads. Each worker runs the
    // allocation-free `qmatmul_band_dot` on its own instance sharing this
    // linear memory, so there is no per-worker libc/TLS dependency.
    #[cfg(target_arch = "wasm32")]
    {
        let xr = &*(x as *const Tensor);
        let x_strides = slice::from_raw_parts(xr.strides, 2);
        let fast =
            (qtr.scheme == QSCHEME_Q4_K_M || qtr.scheme == QSCHEME_Q6_K) && x_strides[1] == 1;
        // Band count: env override (`RAYZOR_WASM_GUEST_THREADS`, read once) wins;
        // otherwise the caller's hint, or 6 for the threads<=0 auto default
        // (sized for the M1-class 6–8 core pool). 1 forces the sequential path.
        let env_n = guest_threads_override();
        let nthreads = if env_n > 0 {
            env_n.min(8)
        } else if threads <= 0 {
            6
        } else {
            (threads as usize).min(8)
        };
        // Worth fanning out once there are enough rows to amortize spawn/join.
        // Both decode (batch==1) and prefill (batch==prompt_len) parallelize:
        // prequantize ALL batch rows once, then split the output rows across
        // workers, each dotting its band against every batch row — mirroring
        // the host pool's batched path so prefill is no longer single-threaded.
        if fast && nthreads > 1 && n_out >= 256 {
            let blocks_per_row = _k / _block_size;
            let x_stride0 = x_strides[0];
            let x_data = xr.data as *const f32;
            let mut x_q8k: std::vec::Vec<Q8KBlock> =
                std::vec::Vec::with_capacity(batch * blocks_per_row);
            let mut tmp: std::vec::Vec<Q8KBlock> = std::vec::Vec::with_capacity(blocks_per_row);
            for bch in 0..batch {
                prepare_x_q8k_blocks_into(x_data.add(bch * x_stride0), _k, &mut tmp);
                x_q8k.extend_from_slice(&tmp[..blocks_per_row]);
            }
            let x_q8k_ptr = x_q8k.as_ptr() as usize as i32;
            let band = (n_out + nthreads - 1) / nthreads;
            let fn_idx = rayzor_qmatmul_band_worker as *const () as usize as i32;
            let mut handles: std::vec::Vec<i32> = std::vec::Vec::new();
            let mut works: std::vec::Vec<*mut BandWork> = std::vec::Vec::new();
            for t in 0..nthreads {
                let lo = (t * band).min(n_out);
                let hi = ((t + 1) * band).min(n_out);
                if lo >= hi {
                    break;
                }
                let work = std::boxed::Box::into_raw(std::boxed::Box::new(BandWork {
                    qt,
                    y,
                    batch,
                    n: n_out,
                    blocks_per_row,
                    block_bytes: _block_bytes,
                    x_q8k: x_q8k_ptr,
                    lo,
                    hi,
                }));
                works.push(work);
                let h = rayzor_thread_spawn(fn_idx, work as usize as i32);
                if h == 0 {
                    // Spawn failed — do this band inline so the result is complete.
                    qmatmul_band_dot(
                        qt,
                        y,
                        batch,
                        n_out,
                        blocks_per_row,
                        _block_bytes,
                        x_q8k_ptr as *const Q8KBlock,
                        lo,
                        hi,
                    );
                } else {
                    handles.push(h);
                }
            }
            // Join all workers before x_q8k (read by every worker) drops.
            for h in handles {
                rayzor_thread_join_void(h);
            }
            for w in works {
                drop(std::boxed::Box::from_raw(w));
            }
            return y;
        }
    }

    let _ = threads;
    qmatmul_chunk_impl(x, qt, y, 0, n_out as i32);
    y
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qkv_qt_t_f32_threaded(
    x: i32,
    q_w: i32,
    k_w: i32,
    v_w: i32,
    threads: i32,
    out_q: i32,
    out_k: i32,
    out_v: i32,
) -> i32 {
    if out_q == 0 || out_k == 0 || out_v == 0 {
        return 1;
    }
    let q = rayzor_tensor_matmul_qt_t_f32_threaded(x, q_w, threads);
    if q == 0 {
        return 2;
    }
    let k = rayzor_tensor_matmul_qt_t_f32_threaded(x, k_w, threads);
    if k == 0 {
        crate::tensor::rayzor_tensor_free(q);
        return 2;
    }
    let v = rayzor_tensor_matmul_qt_t_f32_threaded(x, v_w, threads);
    if v == 0 {
        crate::tensor::rayzor_tensor_free(q);
        crate::tensor::rayzor_tensor_free(k);
        return 2;
    }
    *(out_q as *mut i64) = q as i64;
    *(out_k as *mut i64) = k as i64;
    *(out_v as *mut i64) = v as i64;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayzor_runtime_core::quant::q4_k_m::quantize_block_q4_k_m;

    unsafe fn test_bytes(bytes: &mut [u8]) -> HaxeBytes {
        HaxeBytes {
            ptr: bytes.as_mut_ptr(),
            len: bytes.len(),
            cap: bytes.len(),
            kind: 0,
            _pad: [0; 7],
            owner: core::ptr::null_mut(),
            refcount: 1,
            fd: -1,
            _pad2: [0; 4],
        }
    }

    /// Build a single-block Q4_K_M weight row of 256 weights, run the wasm
    /// matmul against a scalar dequant-then-dot reference, verify they
    /// agree within the Q4_K_M precision (~0.5% relative).
    #[test]
    fn matmul_qt_t_q4_k_m_matches_dequant_reference() {
        unsafe {
            // 1 row, 1 super-block (k=256). Synthetic weights spanning [-1, 1].
            let mut w_f32 = [0.0f32; 256];
            for i in 0..256 {
                w_f32[i] = ((i as f32) / 128.0) - 1.0;
            }
            let w_block = quantize_block_q4_k_m(&w_f32);

            let w_bytes = core::slice::from_raw_parts(
                &w_block as *const Q4KMBlock as *const u8,
                Q4_K_M_BLOCK_BYTES,
            );
            let mut w_vec = w_bytes.to_vec();
            let bytes = test_bytes(&mut w_vec);
            let qt = rayzor_qtensor_from_bytes_q4_k_m(&bytes as *const HaxeBytes as i32, 1, 256);
            assert!(qt != 0, "qtensor alloc failed");

            // X is 1×256 with a smooth gradient.
            let mut x_data = [0.0f32; 256];
            for i in 0..256 {
                x_data[i] = (i as f32) * 0.001 - 0.128;
            }
            let x_shape = [1usize, 256];
            let xt = crate::tensor::rayzor_tensor_from_floats(
                x_data.as_ptr() as i32,
                256,
                x_shape.as_ptr() as i32,
                2,
            );
            assert!(xt != 0, "tensor alloc failed");

            let yt = rayzor_tensor_matmul_qt_t_f32(xt, qt);
            assert!(yt != 0, "matmul failed");
            let yr = &*(yt as *const Tensor);
            assert_eq!(yr.numel, 1);
            let got = *(yr.data as *const f32);

            // Reference: dequant the Q4_K_M block (using the same scalar
            // dequant the runtime exposes), then compute Σ w_dq[i] * x[i].
            let mut w_dq = [0.0f32; 256];
            // The dequant path here reads the Q4KMBlock through Q4KBlock —
            // build that intermediate manually since the public API is
            // raw-pointer-based.
            use rayzor_runtime_core::quant::q4_k_m::{decode_q4_k_block, dequant_q4_k_block};
            let qb = decode_q4_k_block(w_bytes.as_ptr());
            dequant_q4_k_block(&qb, &mut w_dq);
            let mut want = 0.0f32;
            for i in 0..256 {
                want += w_dq[i] * x_data[i];
            }

            // Q4_K_M dot has ~3-bit per-element precision; allow 0.5%
            // relative or 5e-4 absolute, whichever is larger.
            let err = (got - want).abs();
            let rel = if want.abs() > 1e-6 {
                err / want.abs()
            } else {
                err
            };
            assert!(
                rel < 5e-3 || err < 5e-4,
                "got {} want {} err {} rel {}",
                got,
                want,
                err,
                rel
            );

            rayzor_qtensor_free(qt);
            crate::tensor::rayzor_tensor_free(xt);
            crate::tensor::rayzor_tensor_free(yt);
        }
    }

    #[test]
    fn q6_k_bytes_dequant_and_gather_rows_zero_blocks() {
        unsafe {
            let mut raw = std::vec![0u8; 2 * Q6_K_BLOCK_BYTES];
            let bytes = test_bytes(&mut raw);
            let qt = rayzor_qtensor_from_bytes_q6_k(&bytes as *const HaxeBytes as i32, 2, 256);
            assert!(qt != 0);
            assert_eq!(rayzor_qtensor_scheme(qt), QSCHEME_Q6_K as i32);
            assert_eq!(rayzor_qtensor_numel(qt), 512);

            let dq = rayzor_qtensor_dequant(qt);
            assert!(dq != 0);
            assert_eq!(crate::tensor::rayzor_tensor_numel(dq), 512);
            assert_eq!(crate::tensor::rayzor_tensor_sum(dq), 0.0);

            let indices = [1i64, 0];
            let gathered = rayzor_tensor_gather_rows_q6_k(qt, indices.as_ptr() as i32, 2);
            assert!(gathered != 0);
            assert_eq!(crate::tensor::rayzor_tensor_numel(gathered), 512);
            assert_eq!(crate::tensor::rayzor_tensor_sum(gathered), 0.0);

            crate::tensor::rayzor_tensor_free(gathered);
            crate::tensor::rayzor_tensor_free(dq);
            rayzor_qtensor_free(qt);
        }
    }
}
