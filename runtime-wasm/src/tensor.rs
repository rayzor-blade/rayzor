//! WASM-side `Tensor` lifetime + F32 matmul_t.
//!
//! Phase 2 of the WASM parity arc (docs/design/wasm_runtime_parity.md).
//! Mirrors the native `rayzor-runtime::tensor::RayzorTensor` header so
//! consumers that read the wrapper (Haxe extern code, future plugins) see the
//! same layout: data/shape/strides, dtype, ownership, device tags, ARC
//! refcount, and view parent backpointer.
//!
//! Allocation goes through `std::alloc::{alloc, dealloc}` which on
//! `wasm32-wasip1[-threads]` resolves to dlmalloc — no platform FFI.
//!
//! ABI: every public function uses the `i32` wasm address convention.
//! Pointers (`*mut u8`, `*mut usize`) and tensor handles are passed as
//! `i32` to match the Haxe extern wiring already established for the
//! `rayzor_tensor_simd_*` family in this crate.

use core::slice;
use half::{bf16, f16};
use rayzor_runtime_core::quant::matmul::dot_f32_simd;
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Dtype tags — mirror `rayzor-runtime::tensor::DTYPE_*`.
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
pub const DTYPE_I32: u8 = 3;
pub const DTYPE_I8: u8 = 4;
pub const DTYPE_U8: u8 = 5;
pub const DTYPE_FP8_E4M3: u8 = 6;
pub const DTYPE_FP8_E5M2: u8 = 7;

pub(crate) const DEVICE_CPU: u8 = 0;

/// Minimal Tensor wrapper. `#[repr(C)]` so the layout matches the native
/// runtime header exactly.
#[repr(C)]
pub struct Tensor {
    pub data: *mut u8,
    pub shape: *mut usize,
    pub strides: *mut usize,
    pub ndim: usize,
    pub numel: usize,
    pub dtype: u8,
    pub owns_data: bool,
    pub device: u8,
    pub numa_node: i32,
    pub refcount: AtomicUsize,
    pub parent: *mut Tensor,
}

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        DTYPE_F16 | DTYPE_BF16 => 2,
        DTYPE_I32 => 4,
        DTYPE_I8 | DTYPE_U8 | DTYPE_FP8_E4M3 | DTYPE_FP8_E5M2 => 1,
        _ => 4,
    }
}

impl Tensor {
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

    pub fn is_contiguous(&self) -> bool {
        if self.ndim == 0 {
            return true;
        }
        let shape = unsafe { slice::from_raw_parts(self.shape, self.ndim) };
        let strides = unsafe { slice::from_raw_parts(self.strides, self.ndim) };
        let mut expected = 1usize;
        for axis in (0..self.ndim).rev() {
            if strides[axis] != expected {
                return false;
            }
            expected = expected.saturating_mul(shape[axis]);
        }
        true
    }
}

pub(crate) unsafe fn load_f32_at(data: *const u8, idx: usize, dtype: u8) -> f32 {
    match dtype {
        DTYPE_F32 => *(data as *const f32).add(idx),
        DTYPE_F16 => f16::from_bits(*(data as *const u16).add(idx)).to_f32(),
        DTYPE_BF16 => bf16::from_bits(*(data as *const u16).add(idx)).to_f32(),
        DTYPE_I32 => *(data as *const i32).add(idx) as f32,
        DTYPE_I8 => *(data as *const i8).add(idx) as f32,
        DTYPE_U8 => *data.add(idx) as f32,
        DTYPE_FP8_E4M3 => fp8_e4m3_to_f32(*data.add(idx)),
        DTYPE_FP8_E5M2 => fp8_e5m2_to_f32(*data.add(idx)),
        _ => 0.0,
    }
}

pub(crate) unsafe fn store_f32_at(data: *mut u8, idx: usize, dtype: u8, value: f32) {
    match dtype {
        DTYPE_F32 => *(data as *mut f32).add(idx) = value,
        DTYPE_F16 => *(data as *mut u16).add(idx) = f16::from_f32(value).to_bits(),
        DTYPE_BF16 => *(data as *mut u16).add(idx) = bf16::from_f32(value).to_bits(),
        DTYPE_I32 => *(data as *mut i32).add(idx) = value as i32,
        DTYPE_I8 => *(data as *mut i8).add(idx) = value as i8,
        DTYPE_U8 => *data.add(idx) = value as u8,
        DTYPE_FP8_E4M3 => *data.add(idx) = fp8_e4m3_from_f32(value),
        DTYPE_FP8_E5M2 => *data.add(idx) = fp8_e5m2_from_f32(value),
        _ => {}
    }
}

fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    if exp == 0x0f && mant == 0x07 {
        return f32::NAN;
    }
    let sign_f = if sign == 1 { -1.0 } else { 1.0 };
    if exp == 0 {
        sign_f * ((mant as f32) / 8.0) * 2f32.powi(1 - 7)
    } else {
        sign_f * (1.0 + (mant as f32) / 8.0) * 2f32.powi(exp as i32 - 7)
    }
}

fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 1;
    let exp = (byte >> 2) & 0x1f;
    let mant = byte & 0x03;
    if exp == 0 && mant == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    let sign_f = if sign == 1 { -1.0 } else { 1.0 };
    if exp == 0x1f {
        return if mant == 0 {
            sign_f * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    if exp == 0 {
        sign_f * ((mant as f32) / 4.0) * 2f32.powi(1 - 15)
    } else {
        sign_f * (1.0 + (mant as f32) / 4.0) * 2f32.powi(exp as i32 - 15)
    }
}

fn fp8_e4m3_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    if value == 0.0 {
        return if value.is_sign_negative() { 0x80 } else { 0 };
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let v = value.abs().min(448.0);
    let exp_unbiased = v.log2().floor() as i32;
    let exp = (exp_unbiased + 7).clamp(0, 15);
    if exp == 0 {
        let mant = (v / 2f32.powi(1 - 7) * 8.0).round().clamp(0.0, 7.0) as u8;
        return sign | mant;
    }
    let scale = 2f32.powi(exp_unbiased);
    let mant = (((v / scale) - 1.0) * 8.0).round().clamp(0.0, 7.0) as u8;
    sign | ((exp as u8) << 3) | mant
}

fn fp8_e5m2_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    if value.is_infinite() {
        return if value.is_sign_negative() { 0xfc } else { 0x7c };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { 0x80 } else { 0 };
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let v = value.abs();
    let exp_unbiased = v.log2().floor() as i32;
    let exp = (exp_unbiased + 15).clamp(0, 31);
    if exp == 0 {
        let mant = (v / 2f32.powi(1 - 15) * 4.0).round().clamp(0.0, 3.0) as u8;
        return sign | mant;
    }
    if exp == 31 {
        return sign | 0x7c;
    }
    let scale = 2f32.powi(exp_unbiased);
    let mant = (((v / scale) - 1.0) * 4.0).round().clamp(0.0, 3.0) as u8;
    sign | ((exp as u8) << 2) | mant
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
    let stride_vec = Tensor::compute_strides(shape);
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
            device: DEVICE_CPU,
            numa_node: -1,
            refcount: AtomicUsize::new(1),
            parent: core::ptr::null_mut(),
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

/// `Tensor.full(shape, ndim, value, dtype)`.
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
    if value == 0.0 {
        core::ptr::write_bytes(tr.data, 0, tr.numel * dtype_size(tr.dtype));
    } else {
        for i in 0..tr.numel {
            store_f32_at(tr.data, i, tr.dtype, value);
        }
    }
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
    core::ptr::copy_nonoverlapping(data_ptr as *const f32, tr.data as *mut f32, len as usize);
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
    if idx < 0 || idx as usize >= tr.numel {
        return 0.0;
    }
    load_f32_at(tr.data, idx as usize, tr.dtype)
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

/// Atomic-refcount clone: bump `src`'s refcount and return the same pointer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_arc_clone(src: i32) -> i32 {
    if src == 0 {
        return 0;
    }
    let s = &*(src as *const Tensor);
    s.refcount.fetch_add(1, Ordering::Relaxed);
    src
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_clone(src: i32) -> i32 {
    rayzor_tensor_arc_clone(src)
}

/// Disjoint-storage deep clone. Materialises a fresh owning contiguous tensor.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_deep_clone(src: i32) -> i32 {
    if src == 0 {
        return 0;
    }
    let s = &*(src as *const Tensor);
    let shape = slice::from_raw_parts(s.shape, s.ndim);
    let dst = alloc_tensor(shape, s.dtype);
    if dst.is_null() {
        return 0;
    }
    let elem_size = dtype_size(s.dtype);
    let bytes = s.numel * elem_size;
    if bytes > 0 && !s.data.is_null() {
        if s.is_contiguous() {
            core::ptr::copy_nonoverlapping(s.data, (*dst).data, bytes);
        } else {
            let src_strides = slice::from_raw_parts(s.strides, s.ndim);
            let mut idx = std::vec![0usize; s.ndim];
            for linear in 0..s.numel {
                let mut src_elem_off = 0usize;
                for axis in 0..s.ndim {
                    src_elem_off += idx[axis] * src_strides[axis];
                }
                core::ptr::copy_nonoverlapping(
                    s.data.add(src_elem_off * elem_size),
                    (*dst).data.add(linear * elem_size),
                    elem_size,
                );
                for axis in (0..s.ndim).rev() {
                    idx[axis] += 1;
                    if idx[axis] < shape[axis] {
                        break;
                    }
                    idx[axis] = 0;
                }
            }
        }
    }
    dst as i32
}

/// Release a tensor wrapper + its owned shape/strides/data buffers.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_free(t: i32) {
    if t == 0 {
        return;
    }
    let tr = &*(t as *const Tensor);
    let prev = tr.refcount.fetch_sub(1, Ordering::AcqRel);
    if prev != 1 {
        return;
    }
    let dtype = tr.dtype;
    let numel = tr.numel;
    let ndim = tr.ndim;
    let owns = tr.owns_data;
    let data = tr.data;
    let shape = tr.shape;
    let strides = tr.strides;
    let parent = tr.parent;

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
    } else {
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
    if !parent.is_null() {
        rayzor_tensor_free(parent as i32);
    }
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
            assert_eq!(tr.device, DEVICE_CPU);
            assert_eq!(tr.numa_node, -1);
            assert_eq!(tr.refcount.load(Ordering::Relaxed), 1);
            for i in 0..12 {
                assert_eq!(*(tr.data as *const f32).add(i), 0.0);
            }
            rayzor_tensor_free(t as i32);
        }
    }

    #[test]
    fn f16_full_round_trips_through_get_flat() {
        unsafe {
            let shape = [1usize, 2];
            let t = rayzor_tensor_full(shape.as_ptr() as i32, 2, 1.5, DTYPE_F16 as i32);
            assert!(t != 0);
            let tr = &*(t as *const Tensor);
            assert_eq!(tr.dtype, DTYPE_F16);
            assert!((rayzor_tensor_get_flat_f32(t, 0) - 1.5).abs() < 1e-3);
            assert!((rayzor_tensor_get_flat_f32(t, 1) - 1.5).abs() < 1e-3);
            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn arc_clone_defers_physical_free_until_last_drop() {
        unsafe {
            let shape = [2usize, 2];
            let t = alloc_tensor(&shape, DTYPE_F32);
            assert!(!t.is_null());
            let h = t as i32;
            let c = rayzor_tensor_arc_clone(h);
            assert_eq!(c, h);
            assert_eq!((&*t).refcount.load(Ordering::Relaxed), 2);
            rayzor_tensor_free(c);
            assert_eq!((&*t).refcount.load(Ordering::Relaxed), 1);
            rayzor_tensor_free(h);
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
            let xt =
                rayzor_tensor_from_floats(x_data.as_ptr() as i32, 12, x_shape.as_ptr() as i32, 2);
            let wt =
                rayzor_tensor_from_floats(w_data.as_ptr() as i32, 15, w_shape.as_ptr() as i32, 2);
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
                assert!(
                    (got[i] - want[i]).abs() < 1e-5,
                    "mismatch at {}: got {} want {}",
                    i,
                    got[i],
                    want[i]
                );
            }

            rayzor_tensor_free(yt);
            rayzor_tensor_free(xt);
            rayzor_tensor_free(wt);
        }
    }
}
