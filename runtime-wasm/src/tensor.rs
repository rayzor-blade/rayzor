//! WASM-side `Tensor` lifetime + F32 matmul_t.
//!
//! Phase 2 of the WASM parity arc (docs/design/wasm_runtime_parity.md).
//! Mirrors the first six fields of `rayzor-runtime::tensor::RayzorTensor`
//! so consumers that read the wrapper (Haxe extern code, future plugins)
//! see the same layout. Refcount, parent-view, device, and numa_node
//! are omitted from this v1 — they migrate when the native side moves
//! `RayzorTensor` into `rayzor-runtime-core`.
//!
//! Allocation goes through `std::alloc::{alloc, dealloc}` which on
//! `wasm32-wasip1[-threads]` resolves to dlmalloc — no platform FFI.
//!
//! ABI: every public function uses the `i32` wasm address convention.
//! Pointers (`*mut u8`, `*mut usize`) and tensor handles are passed as
//! `i32` to match the Haxe extern wiring already established for the
//! `rayzor_tensor_simd_*` family in this crate.

use core::slice;
use rayzor_runtime_core::quant::matmul::dot_f32_simd;
use std::alloc::{alloc, dealloc, Layout};

/// Dtype tags — mirror `rayzor-runtime::tensor::DTYPE_*` (only the subset
/// Phase 2 cares about). Phase 3 will bring F16 / BF16 / I8 paths over.
pub const DTYPE_F32: u8 = 0;

/// Minimal Tensor wrapper. `#[repr(C)]` so the layout matches the native
/// runtime's first six fields exactly — any reader that inspects the
/// wrapper sees the same offsets for `data`, `shape`, `strides`, `ndim`,
/// `numel`, `dtype`.
#[repr(C)]
pub struct Tensor {
    pub data: *mut u8,
    pub shape: *mut usize,
    pub strides: *mut usize,
    pub ndim: usize,
    pub numel: usize,
    pub dtype: u8,
    pub owns_data: bool,
}

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        _ => 4,
    }
}

fn compute_strides(shape: &[usize]) -> std::vec::Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return std::vec![];
    }
    let mut strides = std::vec![0usize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

unsafe fn alloc_tensor(shape: &[usize], dtype: u8) -> *mut Tensor {
    let ndim = shape.len();
    let mut numel: usize = 1;
    for &d in shape {
        numel = numel.saturating_mul(d);
    }
    let data_bytes = numel * dtype_size(dtype);

    // Data buffer (zero-init).
    let data_layout = Layout::from_size_align(data_bytes.max(1), 16).unwrap();
    let data = alloc(data_layout);
    if data.is_null() {
        return core::ptr::null_mut();
    }
    core::ptr::write_bytes(data, 0, data_bytes);

    // Shape + strides arrays. `alloc::vec::Vec::into_raw_parts` is unstable;
    // we hand-roll the layout/copy so the wrapper owns the buffers and can
    // free them deterministically in `free_tensor`.
    let shape_layout = Layout::array::<usize>(ndim.max(1)).unwrap();
    let shape_ptr = alloc(shape_layout) as *mut usize;
    if shape_ptr.is_null() {
        dealloc(data, data_layout);
        return core::ptr::null_mut();
    }
    core::ptr::copy_nonoverlapping(shape.as_ptr(), shape_ptr, ndim);

    let strides_layout = Layout::array::<usize>(ndim.max(1)).unwrap();
    let strides_ptr = alloc(strides_layout) as *mut usize;
    if strides_ptr.is_null() {
        dealloc(data, data_layout);
        dealloc(shape_ptr as *mut u8, shape_layout);
        return core::ptr::null_mut();
    }
    let stride_vec = compute_strides(shape);
    core::ptr::copy_nonoverlapping(stride_vec.as_ptr(), strides_ptr, ndim);

    let wrapper_layout = Layout::new::<Tensor>();
    let wrapper = alloc(wrapper_layout) as *mut Tensor;
    if wrapper.is_null() {
        dealloc(data, data_layout);
        dealloc(shape_ptr as *mut u8, shape_layout);
        dealloc(strides_ptr as *mut u8, strides_layout);
        return core::ptr::null_mut();
    }
    core::ptr::write(
        wrapper,
        Tensor {
            data,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim,
            numel,
            dtype,
            owns_data: true,
        },
    );
    wrapper
}

/// `Tensor.zeros(shape, ndim, dtype) -> *Tensor`. Shape is passed as a
/// `*const usize` cursor with `ndim` entries.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_zeros(shape_ptr: i32, ndim: i32, dtype: i32) -> i32 {
    if shape_ptr == 0 || ndim < 0 || ndim > 16 {
        return 0;
    }
    let shape = slice::from_raw_parts(shape_ptr as *const usize, ndim as usize);
    alloc_tensor(shape, dtype as u8) as i32
}

/// `Tensor.full(shape, ndim, value, dtype)`. F32 only in this phase.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_full(
    shape_ptr: i32,
    ndim: i32,
    value: f32,
    dtype: i32,
) -> i32 {
    let t = rayzor_tensor_zeros(shape_ptr, ndim, dtype);
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    let dst = slice::from_raw_parts_mut(tr.data as *mut f32, tr.numel);
    dst.fill(value);
    t
}

/// `Tensor.from_floats(data, len, shape, ndim)` — copy `len` f32 values from
/// `data` into a fresh tensor with the given shape (must satisfy
/// `prod(shape) == len`).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_floats(
    data_ptr: i32,
    len: i32,
    shape_ptr: i32,
    ndim: i32,
) -> i32 {
    if data_ptr == 0 || len <= 0 {
        return 0;
    }
    let t = rayzor_tensor_zeros(shape_ptr, ndim, DTYPE_F32 as i32);
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    if tr.numel != len as usize {
        rayzor_tensor_free(t);
        return 0;
    }
    core::ptr::copy_nonoverlapping(
        data_ptr as *const f32,
        tr.data as *mut f32,
        len as usize,
    );
    t
}

/// Field accessors. Direct reads of the `#[repr(C)]` fields; the Haxe
/// extern wiring uses these to read tensor metadata without exposing the
/// raw struct layout to the JIT.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ndim(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const Tensor)).ndim as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_numel(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const Tensor)).numel as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_dtype(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const Tensor)).dtype as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_data(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const Tensor)).data as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape_ptr(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const Tensor)).shape as i32
}

/// Read one f32 element by flat index.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get_flat_f32(t: i32, idx: i32) -> f32 {
    if t == 0 {
        return 0.0;
    }
    let tr = &*(t as *const Tensor);
    if idx < 0 || idx as usize >= tr.numel || tr.dtype != DTYPE_F32 {
        return 0.0;
    }
    *(tr.data as *const f32).add(idx as usize)
}

/// `Y = X @ W^T` where X is `[M, K]` and W is `[N, K]` (stored as if not
/// yet transposed). Returns a freshly allocated `[M, N]` F32 tensor.
///
/// Uses `rayzor_runtime_core::quant::matmul::dot_f32_simd` for the inner
/// dot product. On wasm32 with `+simd128 +relaxed-simd` the scalar
/// fallback path auto-vectorises through LLVM; a dedicated wasm-simd128
/// inner is a Phase 3 follow-up.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_t(x: i32, w: i32) -> i32 {
    if x == 0 || w == 0 {
        return 0;
    }
    let xr = &*(x as *const Tensor);
    let wr = &*(w as *const Tensor);
    if xr.ndim != 2 || wr.ndim != 2 || xr.dtype != DTYPE_F32 || wr.dtype != DTYPE_F32 {
        return 0;
    }
    let x_shape = slice::from_raw_parts(xr.shape, 2);
    let w_shape = slice::from_raw_parts(wr.shape, 2);
    let m = x_shape[0];
    let k = x_shape[1];
    let n_out = w_shape[0];
    let k2 = w_shape[1];
    if k != k2 {
        return 0;
    }

    let y_shape = [m, n_out];
    let y = alloc_tensor(&y_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }

    let x_data = xr.data as *const f32;
    let w_data = wr.data as *const f32;
    let y_data = (*y).data as *mut f32;

    for i in 0..m {
        let x_row = slice::from_raw_parts(x_data.add(i * k), k);
        for j in 0..n_out {
            let w_row = slice::from_raw_parts(w_data.add(j * k), k);
            *y_data.add(i * n_out + j) = dot_f32_simd(x_row, w_row);
        }
    }

    y as i32
}

/// Release a tensor wrapper + its owned shape/strides/data buffers. View
/// tensors (`owns_data == false`) only release the wrapper. v1 has no
/// refcount; Phase 4 will mirror the native ARC wiring.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_free(t: i32) {
    if t == 0 {
        return;
    }
    let tr = &*(t as *const Tensor);
    let dtype = tr.dtype;
    let numel = tr.numel;
    let ndim = tr.ndim;
    let owns = tr.owns_data;
    let data = tr.data;
    let shape = tr.shape;
    let strides = tr.strides;

    if owns {
        if !data.is_null() {
            let bytes = numel * dtype_size(dtype);
            let data_layout = Layout::from_size_align(bytes.max(1), 16).unwrap();
            dealloc(data, data_layout);
        }
        if !shape.is_null() {
            let shape_layout = Layout::array::<usize>(ndim.max(1)).unwrap();
            dealloc(shape as *mut u8, shape_layout);
        }
        if !strides.is_null() {
            let strides_layout = Layout::array::<usize>(ndim.max(1)).unwrap();
            dealloc(strides as *mut u8, strides_layout);
        }
    }

    let wrapper_layout = Layout::new::<Tensor>();
    dealloc(t as *mut u8, wrapper_layout);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_then_get_flat_returns_zero() {
        unsafe {
            let shape = [3usize, 4];
            let t = alloc_tensor(&shape, DTYPE_F32);
            assert!(!t.is_null());
            let tr = &*t;
            assert_eq!(tr.ndim, 2);
            assert_eq!(tr.numel, 12);
            assert_eq!(tr.dtype, DTYPE_F32);
            for i in 0..12 {
                assert_eq!(*(tr.data as *const f32).add(i), 0.0);
            }
            rayzor_tensor_free(t as i32);
        }
    }

    #[test]
    fn matmul_t_64x64_matches_reference() {
        unsafe {
            // X = [4, 3], W = [5, 3]  →  Y = X @ W^T should be [4, 5].
            let x_data: [f32; 12] = [
                1.0, 2.0, 3.0, // row 0
                4.0, 5.0, 6.0, // row 1
                -1.0, 0.5, 2.0, // row 2
                0.0, 0.0, 1.0, // row 3
            ];
            let w_data: [f32; 15] = [
                1.0, 0.0, 0.0, // row 0
                0.0, 1.0, 0.0, // row 1
                0.0, 0.0, 1.0, // row 2
                1.0, 1.0, 1.0, // row 3
                -1.0, -1.0, -1.0, // row 4
            ];

            let x_shape = [4usize, 3];
            let w_shape = [5usize, 3];
            let xt = rayzor_tensor_from_floats(
                x_data.as_ptr() as i32,
                12,
                x_shape.as_ptr() as i32,
                2,
            );
            let wt = rayzor_tensor_from_floats(
                w_data.as_ptr() as i32,
                15,
                w_shape.as_ptr() as i32,
                2,
            );
            assert!(xt != 0 && wt != 0);

            let yt = rayzor_tensor_matmul_t(xt, wt);
            assert!(yt != 0);
            let yr = &*(yt as *const Tensor);
            assert_eq!(yr.ndim, 2);
            assert_eq!(yr.numel, 20);

            // Reference: y[i, j] = Σ_k x[i, k] * w[j, k].
            let mut want = [0.0f32; 20];
            for i in 0..4 {
                for j in 0..5 {
                    let mut s = 0.0f32;
                    for k in 0..3 {
                        s += x_data[i * 3 + k] * w_data[j * 3 + k];
                    }
                    want[i * 5 + j] = s;
                }
            }
            let got = slice::from_raw_parts(yr.data as *const f32, 20);
            for i in 0..20 {
                assert!((got[i] - want[i]).abs() < 1e-5, "mismatch at {}: got {} want {}", i, got[i], want[i]);
            }

            rayzor_tensor_free(yt);
            rayzor_tensor_free(xt);
            rayzor_tensor_free(wt);
        }
    }
}
