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
use rayzor_runtime_core::quant::q4_k_m::vec_dot_q4_K_q8_K_scalar;
use rayzor_runtime_core::quant::types::{
    Q4KMBlock, Q8KBlock, Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_SIZE, QSCHEME_Q4_K_M,
};
use std::alloc::{alloc, dealloc, Layout};

use crate::tensor::{Tensor, DTYPE_F32};

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

unsafe fn alloc_qtensor_q4_k_m(rows: usize, cols: usize) -> *mut QTensor {
    if !cols.is_multiple_of(Q4_K_M_BLOCK_SIZE) {
        return core::ptr::null_mut();
    }
    let numel = rows * cols;
    let data_bytes = q4_k_m_data_bytes(numel);

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
            group_size: Q4_K_M_BLOCK_SIZE,
            scheme: QSCHEME_Q4_K_M,
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
    bytes_ptr: i32,
    rows: i32,
    cols: i32,
) -> i32 {
    if bytes_ptr == 0 || rows <= 0 || cols <= 0 {
        return 0;
    }
    let rows = rows as usize;
    let cols = cols as usize;
    let qt = alloc_qtensor_q4_k_m(rows, cols);
    if qt.is_null() {
        return 0;
    }
    let qt_ref = &*qt;
    let bytes = q4_k_m_data_bytes(qt_ref.numel);
    core::ptr::copy_nonoverlapping(bytes_ptr as *const u8, qt_ref.data, bytes);
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
pub unsafe extern "C" fn rayzor_qtensor_free(qt: i32) {
    if qt == 0 {
        return;
    }
    let qt_ref = &*(qt as *const QTensor);
    let data = qt_ref.data;
    let owns = qt_ref.owns_data;
    let numel = qt_ref.numel;

    if owns && !data.is_null() {
        let bytes = q4_k_m_data_bytes(numel);
        let data_layout = Layout::from_size_align(bytes.max(1), 16).unwrap();
        dealloc(data, data_layout);
    }
    let wrapper_layout = Layout::new::<QTensor>();
    dealloc(qt as *mut u8, wrapper_layout);
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
    let xr = &*(x as *const Tensor);
    let qtr = &*(qt as *const QTensor);
    if xr.ndim != 2 || xr.dtype != DTYPE_F32 || qtr.scheme != QSCHEME_Q4_K_M {
        return 0;
    }
    let x_shape = slice::from_raw_parts(xr.shape, 2);
    let m = x_shape[0];
    let k = x_shape[1];
    if k != qtr.cols || !k.is_multiple_of(Q4_K_M_BLOCK_SIZE) {
        return 0;
    }
    let n_out = qtr.rows;
    let blocks_per_row = k / Q4_K_M_BLOCK_SIZE;

    // Allocate output Tensor via the public Tensor allocator. The wasm-side
    // tensor module owns its own dlmalloc-backed alloc path.
    let y_shape: [usize; 2] = [m, n_out];
    let y = crate::tensor::alloc_tensor(&y_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }
    let y = y as i32;
    let yr = &*(y as *const Tensor);
    let y_data = yr.data as *mut f32;

    // Pre-quant scratch reused across all output rows.
    let mut x_q8k: std::vec::Vec<Q8KBlock> = std::vec::Vec::with_capacity(blocks_per_row);

    let x_data = xr.data as *const f32;
    let qt_data = qtr.data as *const u8;

    for i in 0..m {
        let x_row_ptr = x_data.add(i * k);
        prepare_x_q8k_blocks_into(x_row_ptr, k, &mut x_q8k);

        for j in 0..n_out {
            let mut sum = 0.0f32;
            let row_ptr = qt_data.add(j * blocks_per_row * Q4_K_M_BLOCK_BYTES);
            for b in 0..blocks_per_row {
                let weight = &*(row_ptr.add(b * Q4_K_M_BLOCK_BYTES) as *const Q4KMBlock);
                sum += vec_dot_q4_K_q8_K_scalar(weight, &x_q8k[b]);
            }
            *y_data.add(i * n_out + j) = sum;
        }
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayzor_runtime_core::quant::q4_k_m::quantize_block_q4_k_m;

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
            let qt = rayzor_qtensor_from_bytes_q4_k_m(w_bytes.as_ptr() as i32, 1, 256);
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
}
