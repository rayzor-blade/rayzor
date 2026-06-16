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

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        DTYPE_F16 | DTYPE_BF16 => 2,
        DTYPE_I32 => 4,
        DTYPE_I8 | DTYPE_U8 | DTYPE_FP8_E4M3 | DTYPE_FP8_E5M2 => 1,
        _ => 4,
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

#[cfg(test)]
static TEST_PHYSICAL_RELEASES: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static TEST_TRACKED_RELEASE_PTR: AtomicUsize = AtomicUsize::new(0);

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

fn read_shape(shape_ptr: i32, ndim: usize) -> std::vec::Vec<usize> {
    if shape_ptr == 0 || ndim == 0 {
        return std::vec![];
    }
    unsafe {
        let data = shape_ptr as *const i64;
        (0..ndim).map(|idx| *data.add(idx) as usize).collect()
    }
}

unsafe fn alloc_usize_array(values: &[usize]) -> *mut usize {
    let layout = Layout::array::<usize>(values.len().max(1)).unwrap();
    let ptr = alloc(layout) as *mut usize;
    if ptr.is_null() {
        return core::ptr::null_mut();
    }
    if !values.is_empty() {
        core::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
    ptr
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

pub(crate) unsafe fn alloc_tensor(shape: &[usize], dtype: u8) -> *mut Tensor {
    let ndim = shape.len();
    let mut numel: usize = 1;
    for &d in shape {
        numel = numel.saturating_mul(d);
    }
    let data_bytes = numel * dtype_size(dtype);

    #[cfg(feature = "alloc-track")]
    if data_bytes >= (4 << 20) {
        let mut v = [0i64; 9];
        let k = ndim.min(8);
        v[0] = data_bytes as i64;
        for i in 0..k {
            v[i + 1] = shape[i] as i64;
        }
        crate::rt_dbg_n("[bigtensor] bytes+dims", &v[..k + 1]);
    }

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
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8) as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ones(shape_ptr: i32, ndim: i32, dtype: i32) -> i32 {
    rayzor_tensor_full(shape_ptr, ndim, 1.0, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rand(shape_ptr: i32, ndim: i32, dtype: i32) -> i32 {
    // Deterministic placeholder until wasm RNG parity lands.
    rayzor_tensor_zeros(shape_ptr, ndim, dtype)
}

/// `Tensor.full(shape, ndim, value, dtype)`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_full(
    shape_ptr: i32,
    ndim: i32,
    value: f64,
    dtype: i32,
) -> i32 {
    let t = rayzor_tensor_zeros(shape_ptr, ndim, dtype);
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    let value = value as f32;
    if value == 0.0 {
        core::ptr::write_bytes(tr.data, 0, tr.numel * dtype_size(tr.dtype));
    } else {
        for i in 0..tr.numel {
            store_f32_at(tr.data, i, tr.dtype, value);
        }
    }
    t
}

/// `Tensor.fromArray(data_ptr, data_len, dtype)`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_array(data_ptr: i32, data_len: i32, dtype: i32) -> i32 {
    if data_ptr == 0 || data_len < 0 {
        return 0;
    }
    let shape = [data_len as usize];
    let t = alloc_tensor(&shape, dtype as u8) as i32;
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    let src = slice::from_raw_parts(data_ptr as *const f64, data_len as usize);
    for (idx, &value) in src.iter().enumerate() {
        store_f32_at(tr.data, idx, tr.dtype, value as f32);
    }
    t
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_f32(
    bytes_handle: i32,
    shape_ptr: i32,
    ndim: i32,
) -> i32 {
    if shape_ptr == 0 || ndim < 0 || ndim > 16 {
        return 0;
    }
    let shape = read_shape(shape_ptr, ndim as usize);
    let numel: usize = shape.iter().product();
    let Some(bytes) = bytes_slice(bytes_handle, numel * 4) else {
        return 0;
    };
    let t = alloc_tensor(&shape, DTYPE_F32) as i32;
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    core::ptr::copy_nonoverlapping(bytes.as_ptr(), tr.data, numel * 4);
    t
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_f16(
    bytes_handle: i32,
    shape_ptr: i32,
    ndim: i32,
) -> i32 {
    if shape_ptr == 0 || ndim < 0 || ndim > 16 {
        return 0;
    }
    let shape = read_shape(shape_ptr, ndim as usize);
    let numel: usize = shape.iter().product();
    let Some(bytes) = bytes_slice(bytes_handle, numel * 2) else {
        return 0;
    };
    let t = alloc_tensor(&shape, DTYPE_F32) as i32;
    if t == 0 {
        return 0;
    }
    let dst = (*(t as *const Tensor)).data as *mut f32;
    for i in 0..numel {
        let lo = bytes[i * 2] as u16;
        let hi = bytes[i * 2 + 1] as u16;
        *dst.add(i) = f16::from_bits(lo | (hi << 8)).to_f32();
    }
    t
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_q8_0(
    bytes_handle: i32,
    shape_ptr: i32,
    ndim: i32,
) -> i32 {
    if shape_ptr == 0 || ndim < 0 || ndim > 16 {
        return 0;
    }
    let shape = read_shape(shape_ptr, ndim as usize);
    let numel: usize = shape.iter().product();
    if !numel.is_multiple_of(32) {
        return 0;
    }
    let n_blocks = numel / 32;
    let Some(bytes) = bytes_slice(bytes_handle, n_blocks * 34) else {
        return 0;
    };
    let t = alloc_tensor(&shape, DTYPE_F32) as i32;
    if t == 0 {
        return 0;
    }
    let dst = (*(t as *const Tensor)).data as *mut f32;
    for b in 0..n_blocks {
        let base = b * 34;
        let scale = f16::from_bits(bytes[base] as u16 | ((bytes[base + 1] as u16) << 8)).to_f32();
        let q = bytes.as_ptr().add(base + 2) as *const i8;
        for j in 0..32 {
            *dst.add(b * 32 + j) = (*q.add(j) as f32) * scale;
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
    if data_ptr == 0 || len <= 0 || shape_ptr == 0 || ndim < 0 || ndim > 16 {
        return 0;
    }
    let shape = slice::from_raw_parts(shape_ptr as *const usize, ndim as usize).to_vec();
    let t = alloc_tensor(&shape, DTYPE_F32) as i32;
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
pub unsafe extern "C" fn rayzor_tensor_device(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    (*(t as *const Tensor)).device as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_numa_node(t: i32) -> i32 {
    if t == 0 {
        return -1;
    }
    (*(t as *const Tensor)).numa_node
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

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape_ndim(t: i32) -> i32 {
    rayzor_tensor_ndim(t)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    let shape = slice::from_raw_parts(tr.shape, tr.ndim);
    let header_layout = Layout::from_size_align(32, 8).unwrap();
    let header = alloc(header_layout);
    if header.is_null() {
        return 0;
    }
    core::ptr::write_bytes(header, 0, 32);
    let data_layout = Layout::from_size_align((tr.ndim * 8).max(1), 8).unwrap();
    let data = alloc(data_layout);
    if data.is_null() {
        dealloc(header, header_layout);
        return 0;
    }
    for (idx, &dim) in shape.iter().enumerate() {
        *(data as *mut i64).add(idx) = dim as i64;
    }
    let h = header as *mut u32;
    *h = data as u32;
    *h.add(2) = tr.ndim as u32;
    *h.add(4) = tr.ndim as u32;
    *h.add(6) = 8;
    header as i32
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

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get_flat(t: i32, idx: i32) -> f64 {
    rayzor_tensor_get_flat_f32(t, idx) as f64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get(t: i32, indices_ptr: i32, ndim: i32) -> f64 {
    if t == 0 || indices_ptr == 0 || ndim < 0 {
        return 0.0;
    }
    let tr = &*(t as *const Tensor);
    if ndim as usize != tr.ndim {
        return 0.0;
    }
    let indices = slice::from_raw_parts(indices_ptr as *const i64, ndim as usize);
    let shape = slice::from_raw_parts(tr.shape, tr.ndim);
    let strides = slice::from_raw_parts(tr.strides, tr.ndim);
    let mut off = 0usize;
    for axis in 0..tr.ndim {
        let idx = indices[axis];
        if idx < 0 || idx as usize >= shape[axis] {
            return 0.0;
        }
        off += idx as usize * strides[axis];
    }
    load_f32_at(tr.data, off, tr.dtype) as f64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_set(t: i32, indices_ptr: i32, ndim: i32, value: f64) {
    if t == 0 || indices_ptr == 0 || ndim < 0 {
        return;
    }
    let tr = &*(t as *const Tensor);
    if ndim as usize != tr.ndim {
        return;
    }
    let indices = slice::from_raw_parts(indices_ptr as *const i64, ndim as usize);
    let shape = slice::from_raw_parts(tr.shape, tr.ndim);
    let strides = slice::from_raw_parts(tr.strides, tr.ndim);
    let mut off = 0usize;
    for axis in 0..tr.ndim {
        let idx = indices[axis];
        if idx < 0 || idx as usize >= shape[axis] {
            return;
        }
        off += idx as usize * strides[axis];
    }
    store_f32_at(tr.data, off, tr.dtype, value as f32);
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

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 {
        return 0;
    }
    let ar = &*(a as *const Tensor);
    let br = &*(b as *const Tensor);
    if ar.ndim != 2 || br.ndim != 2 || ar.dtype != DTYPE_F32 || br.dtype != DTYPE_F32 {
        return 0;
    }
    let a_shape = slice::from_raw_parts(ar.shape, 2);
    let b_shape = slice::from_raw_parts(br.shape, 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];
    if k != k2 {
        return 0;
    }
    let y_shape = [m, n];
    let y = alloc_tensor(&y_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }
    let a_data = ar.data as *const f32;
    let b_data = br.data as *const f32;
    let y_data = (*y).data as *mut f32;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += *a_data.add(i * k + p) * *b_data.add(p * n + j);
            }
            *y_data.add(i * n + j) = sum;
        }
    }
    y as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_t_threaded(a: i32, b: i32, _threads: i32) -> i32 {
    rayzor_tensor_matmul_t(a, b)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_bmm(a: i32, b: i32) -> i32 {
    if a == 0 || b == 0 {
        return 0;
    }
    let ar = &*(a as *const Tensor);
    let br = &*(b as *const Tensor);
    // 2D inputs are a plain matmul.
    if ar.ndim == 2 && br.ndim == 2 {
        return rayzor_tensor_matmul(a, b);
    }
    // Batched matmul `a [B, M, K] @ b [B, K, N] -> [B, M, N]`.
    //
    // CRITICAL: the transformer callers feed NON-CONTIGUOUS views —
    // GQAttention's `qByHead = q.permute([1,0,2])` and
    // `kT = kAllExpanded.transposeLast2()` (and `attn.bmm(vAllExpanded)`).
    // The old wasm bmm just forwarded to `rayzor_tensor_matmul`, which
    // rejects `ndim != 2` and returned a null handle → attention scores were
    // all zeros → the attention sub-layer contributed nothing → the residual
    // passed the input straight through → degenerate "The best answer is:"
    // generation regardless of how correct the rest of the pipeline was.
    // (The native bmm fixed exactly this; mirror its strided walk here.)
    if ar.ndim != 3 || br.ndim != 3 || ar.dtype != DTYPE_F32 || br.dtype != DTYPE_F32 {
        return 0;
    }
    let a_shape = slice::from_raw_parts(ar.shape, 3);
    let b_shape = slice::from_raw_parts(br.shape, 3);
    let a_strides = slice::from_raw_parts(ar.strides, 3);
    let b_strides = slice::from_raw_parts(br.strides, 3);
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];
    if b_shape[0] != batch || b_shape[1] != k {
        return 0;
    }
    let y_shape = [batch, m, n];
    let y = alloc_tensor(&y_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }
    let a_data = ar.data as *const f32;
    let b_data = br.data as *const f32;
    let y_data = (*y).data as *mut f32;
    let (asb, asm, ask) = (a_strides[0], a_strides[1], a_strides[2]);
    let (bsb, bsk, bsn) = (b_strides[0], b_strides[1], b_strides[2]);
    for bt in 0..batch {
        let a_boff = bt * asb;
        let b_boff = bt * bsb;
        let c_boff = bt * m * n;
        for i in 0..m {
            let a_roff = a_boff + i * asm;
            let c_roff = c_boff + i * n;
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += *a_data.add(a_roff + p * ask) * *b_data.add(b_boff + p * bsk + j * bsn);
                }
                *y_data.add(c_roff + j) = sum;
            }
        }
    }
    y as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_bmm_threaded(a: i32, b: i32, _threads: i32) -> i32 {
    rayzor_tensor_bmm(a, b)
}

unsafe fn tensor_unary<F: Fn(f32) -> f32>(t: i32, f: F) -> i32 {
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    let shape = slice::from_raw_parts(tr.shape, tr.ndim);
    let y = alloc_tensor(shape, tr.dtype);
    if y.is_null() {
        return 0;
    }
    let yr = &*y;
    for idx in 0..tr.numel {
        store_f32_at(
            yr.data,
            idx,
            tr.dtype,
            f(load_f32_at(tr.data, idx, tr.dtype)),
        );
    }
    y as i32
}

unsafe fn tensor_binary<F: Fn(f32, f32) -> f32>(a: i32, b: i32, f: F) -> i32 {
    if a == 0 || b == 0 {
        return 0;
    }
    let ar = &*(a as *const Tensor);
    let br = &*(b as *const Tensor);

    // Trailing-dim broadcast: `a [..., D] op b [D]` → result shaped like a.
    // Matches the native runtime's tensor_binop_row_broadcast — the form
    // used by RMSNorm/LayerNorm gains and dense biases (e.g. RMSNorm's
    // `norm.mul(weight)` where norm is [seq, hidden] and weight is
    // [hidden]). Without this the wasm path returned null and every
    // attention input collapsed to a null activation.
    if br.ndim == 1 && ar.ndim >= 1 {
        let a_shape = slice::from_raw_parts(ar.shape, ar.ndim);
        let last = a_shape[ar.ndim - 1];
        let b_shape = slice::from_raw_parts(br.shape, 1);
        if b_shape[0] == last && last != 0 && ar.numel.is_multiple_of(last) {
            let y = alloc_tensor(a_shape, ar.dtype);
            if y.is_null() {
                return 0;
            }
            let yr = &*y;
            let groups = ar.numel / last;
            for g in 0..groups {
                let off = g * last;
                for j in 0..last {
                    let av = load_f32_at(ar.data, off + j, ar.dtype);
                    let bv = load_f32_at(br.data, j, br.dtype);
                    store_f32_at(yr.data, off + j, yr.dtype, f(av, bv));
                }
            }
            return y as i32;
        }
    }

    if ar.ndim != br.ndim || ar.numel != br.numel {
        return 0;
    }
    let a_shape = slice::from_raw_parts(ar.shape, ar.ndim);
    let b_shape = slice::from_raw_parts(br.shape, br.ndim);
    if a_shape != b_shape {
        return 0;
    }
    let y = alloc_tensor(a_shape, ar.dtype);
    if y.is_null() {
        return 0;
    }
    let yr = &*y;
    for idx in 0..ar.numel {
        let av = load_f32_at(ar.data, idx, ar.dtype);
        let bv = load_f32_at(br.data, idx, br.dtype);
        store_f32_at(yr.data, idx, yr.dtype, f(av, bv));
    }
    y as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_add(a: i32, b: i32) -> i32 {
    tensor_binary(a, b, |x, y| x + y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sub(a: i32, b: i32) -> i32 {
    tensor_binary(a, b, |x, y| x - y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mul(a: i32, b: i32) -> i32 {
    tensor_binary(a, b, |x, y| x * y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_div(a: i32, b: i32) -> i32 {
    tensor_binary(a, b, |x, y| x / y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sqrt(t: i32) -> i32 {
    tensor_unary(t, libm::sqrtf)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_exp(t: i32) -> i32 {
    tensor_unary(t, libm::expf)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_log(t: i32) -> i32 {
    tensor_unary(t, libm::logf)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_relu(t: i32) -> i32 {
    tensor_unary(t, |x| x.max(0.0))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gelu(t: i32) -> i32 {
    tensor_unary(t, |x| {
        0.5 * x * (1.0 + libm::tanhf(0.7978846 * (x + 0.044715 * x * x * x)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_silu(t: i32) -> i32 {
    tensor_unary(t, |x| x / (1.0 + libm::expf(-x)))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sum(t: i32) -> f64 {
    if t == 0 {
        return 0.0;
    }
    let tr = &*(t as *const Tensor);
    let mut sum = 0.0f64;
    for idx in 0..tr.numel {
        sum += load_f32_at(tr.data, idx, tr.dtype) as f64;
    }
    sum
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mean(t: i32) -> f64 {
    if t == 0 {
        return 0.0;
    }
    let tr = &*(t as *const Tensor);
    if tr.numel == 0 {
        0.0
    } else {
        rayzor_tensor_sum(t) / tr.numel as f64
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_max(t: i32) -> f64 {
    if t == 0 {
        return 0.0;
    }
    let tr = &*(t as *const Tensor);
    if tr.numel == 0 {
        return 0.0;
    }
    let mut best = load_f32_at(tr.data, 0, tr.dtype);
    for idx in 1..tr.numel {
        best = best.max(load_f32_at(tr.data, idx, tr.dtype));
    }
    best as f64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_min(t: i32) -> f64 {
    if t == 0 {
        return 0.0;
    }
    let tr = &*(t as *const Tensor);
    if tr.numel == 0 {
        return 0.0;
    }
    let mut best = load_f32_at(tr.data, 0, tr.dtype);
    for idx in 1..tr.numel {
        best = best.min(load_f32_at(tr.data, idx, tr.dtype));
    }
    best as f64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_dot(a: i32, b: i32) -> f64 {
    if a == 0 || b == 0 {
        return 0.0;
    }
    let ar = &*(a as *const Tensor);
    let br = &*(b as *const Tensor);
    if ar.numel != br.numel {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for idx in 0..ar.numel {
        sum +=
            load_f32_at(ar.data, idx, ar.dtype) as f64 * load_f32_at(br.data, idx, br.dtype) as f64;
    }
    sum
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_add_into(dest: i32, src: i32) {
    if dest == 0 || src == 0 {
        return;
    }
    let d = &*(dest as *const Tensor);
    let s = &*(src as *const Tensor);
    if d.ndim != s.ndim || d.numel != s.numel || d.dtype != s.dtype || !d.is_contiguous() {
        return;
    }
    let d_shape = slice::from_raw_parts(d.shape, d.ndim);
    let s_shape = slice::from_raw_parts(s.shape, s.ndim);
    if d_shape != s_shape || d.data == s.data {
        return;
    }
    if s.is_contiguous() {
        for idx in 0..d.numel {
            let v = load_f32_at(d.data, idx, d.dtype) + load_f32_at(s.data, idx, s.dtype);
            store_f32_at(d.data, idx, d.dtype, v);
        }
        return;
    }
    let s_strides = slice::from_raw_parts(s.strides, s.ndim);
    let mut idx = std::vec![0usize; d.ndim];
    for flat in 0..d.numel {
        let mut rem = flat;
        for axis in 0..d.ndim {
            let stride = d_shape[axis + 1..].iter().product::<usize>().max(1);
            idx[axis] = rem / stride;
            rem %= stride;
        }
        let mut s_off = 0usize;
        for axis in 0..s.ndim {
            s_off += idx[axis] * s_strides[axis];
        }
        let v = load_f32_at(d.data, flat, d.dtype) + load_f32_at(s.data, s_off, s.dtype);
        store_f32_at(d.data, flat, d.dtype, v);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_scale(t: i32, factor: f64) -> i32 {
    tensor_unary(t, |x| x * factor as f32)
}

// ============================================================================
// rayzor_plugin_* ABI — the composition seam for wasm plugin COMPONENTS.
//
// A plugin component (e.g. nue-plugins' Q8 KV cache + flash kernel) is built
// as wasm that IMPORTS these `rayzor_plugin_tensor_*` symbols to touch tensors
// without knowing the `Tensor` struct layout. On native the host process
// exports them (-export_dynamic); in wasm, runtime-wasm — which owns `Tensor`
// — provides them here and the wasm-linker resolves the plugin's imports
// against these exports. This is what lets the SAME plugin (Q8 cache, etc.)
// load as a portable component on both targets instead of being a native-only
// dylib. `TensorHandle` is the `Tensor` pointer as i64 (32-bit on wasm,
// zero-extended; `as usize` recovers it on either width).
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_alloc_zeros(
    shape_ptr: *const usize,
    ndim: usize,
    dtype: u8,
) -> i64 {
    if shape_ptr.is_null() && ndim != 0 {
        return 0;
    }
    let shape = slice::from_raw_parts(shape_ptr, ndim);
    let t = alloc_tensor(shape, dtype);
    t as usize as i64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_data(t: i64) -> *mut u8 {
    if t == 0 {
        return core::ptr::null_mut();
    }
    (*(t as usize as *const Tensor)).data
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_dtype(t: i64) -> u8 {
    if t == 0 {
        return 0;
    }
    (*(t as usize as *const Tensor)).dtype
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_ndim(t: i64) -> u32 {
    if t == 0 {
        return 0;
    }
    (*(t as usize as *const Tensor)).ndim as u32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_shape(t: i64) -> *const usize {
    if t == 0 {
        return core::ptr::null();
    }
    (*(t as usize as *const Tensor)).shape as *const usize
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_plugin_tensor_is_contiguous(t: i64) -> u8 {
    if t == 0 {
        return 0;
    }
    u8::from((*(t as usize as *const Tensor)).is_contiguous())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_causal_mask_(t: i32, position_offset: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    if tr.ndim < 2 {
        return 0;
    }
    let shape = slice::from_raw_parts(tr.shape, tr.ndim);
    let cols = shape[tr.ndim - 1];
    let rows = shape[tr.ndim - 2];
    let outer = shape[..tr.ndim.saturating_sub(2)]
        .iter()
        .product::<usize>()
        .max(1);
    let pos = position_offset.max(0) as usize;
    for o in 0..outer {
        let base = o * rows * cols;
        for i in 0..rows {
            let first_masked = (i + pos + 1).min(cols);
            for j in first_masked..cols {
                store_f32_at(tr.data, base + i * cols + j, tr.dtype, f32::NEG_INFINITY);
            }
        }
    }
    t
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_transpose_last2(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    if tr.ndim < 2 {
        return rayzor_tensor_arc_clone(t);
    }
    let n = tr.ndim;
    let old_shape = slice::from_raw_parts(tr.shape, n);
    let old_strides = slice::from_raw_parts(tr.strides, n);
    let mut new_shape = old_shape.to_vec();
    let mut new_strides = old_strides.to_vec();
    new_shape.swap(n - 2, n - 1);
    new_strides.swap(n - 2, n - 1);
    let shape_ptr = alloc_usize_array(&new_shape);
    let strides_ptr = alloc_usize_array(&new_strides);
    if shape_ptr.is_null() || strides_ptr.is_null() {
        return 0;
    }
    tr.refcount.fetch_add(1, Ordering::Relaxed);
    let wrapper_layout = Layout::new::<Tensor>();
    let wrapper = alloc(wrapper_layout) as *mut Tensor;
    if wrapper.is_null() {
        return 0;
    }
    core::ptr::write(
        wrapper,
        Tensor {
            data: tr.data,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim: n,
            numel: tr.numel,
            dtype: tr.dtype,
            owns_data: false,
            device: tr.device,
            numa_node: tr.numa_node,
            refcount: AtomicUsize::new(1),
            parent: t as *mut Tensor,
        },
    );
    wrapper as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_transpose(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    if tr.ndim != 2 {
        return rayzor_tensor_arc_clone(t);
    }
    rayzor_tensor_transpose_last2(t)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_permute(t: i32, axes_ptr: i32, axes_len: i32) -> i32 {
    if t == 0 || axes_ptr == 0 || axes_len <= 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    let n = axes_len as usize;
    if n != tr.ndim {
        return 0;
    }
    let axes_data = axes_ptr as *const i64;
    let mut seen = std::vec![false; n];
    let mut axes = std::vec![0usize; n];
    for i in 0..n {
        let axis = *axes_data.add(i);
        if axis < 0 || axis as usize >= n || seen[axis as usize] {
            return 0;
        }
        seen[axis as usize] = true;
        axes[i] = axis as usize;
    }
    let old_shape = slice::from_raw_parts(tr.shape, n);
    let old_strides = slice::from_raw_parts(tr.strides, n);
    let mut new_shape = std::vec![0usize; n];
    let mut new_strides = std::vec![0usize; n];
    for i in 0..n {
        new_shape[i] = old_shape[axes[i]];
        new_strides[i] = old_strides[axes[i]];
    }
    let shape_ptr = alloc_usize_array(&new_shape);
    let strides_ptr = alloc_usize_array(&new_strides);
    if shape_ptr.is_null() || strides_ptr.is_null() {
        return 0;
    }
    let wrapper = alloc(Layout::new::<Tensor>()) as *mut Tensor;
    if wrapper.is_null() {
        return 0;
    }
    tr.refcount.fetch_add(1, Ordering::Relaxed);
    core::ptr::write(
        wrapper,
        Tensor {
            data: tr.data,
            shape: shape_ptr,
            strides: strides_ptr,
            ndim: n,
            numel: tr.numel,
            dtype: tr.dtype,
            owns_data: false,
            device: tr.device,
            numa_node: tr.numa_node,
            refcount: AtomicUsize::new(1),
            parent: t as *mut Tensor,
        },
    );
    wrapper as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows(
    table_ptr: i32,
    indices_ptr: i32,
    indices_len: i32,
) -> i32 {
    if table_ptr == 0 || indices_ptr == 0 || indices_len <= 0 {
        return 0;
    }
    let table = &*(table_ptr as *const Tensor);
    if table.ndim == 0 || !table.is_contiguous() {
        return 0;
    }
    let table_shape = slice::from_raw_parts(table.shape, table.ndim);
    let n_rows = table_shape[0];
    let row_numel = table_shape[1..].iter().product::<usize>().max(1);
    let row_bytes = row_numel * dtype_size(table.dtype);
    let k = indices_len as usize;
    let mut out_shape = std::vec![k];
    out_shape.extend_from_slice(&table_shape[1..]);
    let out = alloc_tensor(&out_shape, table.dtype);
    if out.is_null() {
        return 0;
    }
    let indices = indices_ptr as *const i64;
    for i in 0..k {
        let idx = *indices.add(i);
        if idx < 0 || idx as usize >= n_rows {
            continue;
        }
        core::ptr::copy_nonoverlapping(
            table.data.add(idx as usize * row_bytes),
            (*out).data.add(i * row_bytes),
            row_bytes,
        );
    }
    out as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_broadcast_repeat_0_f32(
    dst_ptr: i32,
    src_ptr: i32,
    repeats: i32,
) -> i32 {
    if dst_ptr == 0 || src_ptr == 0 || repeats <= 0 {
        return -1;
    }
    let dst = &*(dst_ptr as *const Tensor);
    let src = &*(src_ptr as *const Tensor);
    if dst.dtype != DTYPE_F32 || src.dtype != DTYPE_F32 || dst.ndim == 0 || dst.ndim != src.ndim {
        return -1;
    }
    let dst_shape = slice::from_raw_parts(dst.shape, dst.ndim);
    let src_shape = slice::from_raw_parts(src.shape, src.ndim);
    for axis in 1..dst.ndim {
        if dst_shape[axis] != src_shape[axis] {
            return -1;
        }
    }
    let repeats = repeats as usize;
    if src_shape[0].saturating_mul(repeats) > dst_shape[0] {
        return -1;
    }
    let row_elems = src_shape[1..].iter().product::<usize>().max(1);
    let row_bytes = row_elems * dtype_size(DTYPE_F32);
    for i in 0..src_shape[0] {
        let src_row = src.data.add(i * row_bytes);
        for r in 0..repeats {
            let dst_row = dst.data.add((i * repeats + r) * row_bytes);
            core::ptr::copy_nonoverlapping(src_row, dst_row, row_bytes);
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_expand_kv_heads_axis1_f32(
    src_ptr: i32,
    repeats: i32,
) -> i32 {
    if src_ptr == 0 || repeats <= 0 {
        return 0;
    }
    let src = &*(src_ptr as *const Tensor);
    if src.dtype != DTYPE_F32 || src.ndim != 3 {
        return 0;
    }
    let shape = slice::from_raw_parts(src.shape, 3);
    let strides = slice::from_raw_parts(src.strides, 3);
    let seq_k = shape[0];
    let kv_heads = shape[1];
    let head_dim = shape[2];
    if strides[2] != 1 {
        return 0;
    }
    let repeats = repeats as usize;
    let q_heads = kv_heads * repeats;
    let out_shape = [q_heads, seq_k, head_dim];
    let out = alloc_tensor(&out_shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let dst = &*out;
    let row_bytes = head_dim * 4;
    let dst_head_stride = seq_k * head_dim;
    for qh in 0..q_heads {
        let kvh = qh / repeats;
        let src_head_off = kvh * strides[1];
        let dst_head_off = qh * dst_head_stride;
        for j in 0..seq_k {
            let src_off = src_head_off + j * strides[0];
            let dst_off = dst_head_off + j * head_dim;
            core::ptr::copy_nonoverlapping(
                src.data.add(src_off * 4),
                dst.data.add(dst_off * 4),
                row_bytes,
            );
        }
    }
    out as i32
}

/// Copy rows from `src` into `dst` along axis 0 starting at `dst_row_offset`.
///
/// Both tensors must be F32, same rank, and have identical trailing axes.
/// Returns `0` on success and `-1` on validation failure, mirroring native.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_append_along_0_f32(
    dst_ptr: i32,
    src_ptr: i32,
    dst_row_offset: i32,
) -> i32 {
    if dst_ptr == 0 || src_ptr == 0 || dst_row_offset < 0 {
        return -1;
    }
    let dst = &*(dst_ptr as *const Tensor);
    let src = &*(src_ptr as *const Tensor);
    if dst.dtype != DTYPE_F32 || src.dtype != DTYPE_F32 {
        return -1;
    }
    if dst.ndim == 0 || src.ndim == 0 || dst.ndim != src.ndim {
        return -1;
    }
    if !dst.is_contiguous() || !src.is_contiguous() {
        return -1;
    }
    let dst_shape = slice::from_raw_parts(dst.shape, dst.ndim);
    let src_shape = slice::from_raw_parts(src.shape, src.ndim);
    for axis in 1..dst.ndim {
        if dst_shape[axis] != src_shape[axis] {
            return -1;
        }
    }
    let row_stride: usize = dst_shape[1..].iter().product();
    let rows = src_shape[0];
    let offset = dst_row_offset as usize;
    if offset + rows > dst_shape[0] {
        return -1;
    }
    let byte_count = rows * row_stride * dtype_size(DTYPE_F32);
    let dst_offset = offset * row_stride * dtype_size(DTYPE_F32);
    core::ptr::copy_nonoverlapping(src.data, dst.data.add(dst_offset), byte_count);
    0
}

/// Contiguous-only reshape. Returns a zero-copy view with canonical strides.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_reshape(tensor_ptr: i32, shape_ptr: i32, ndim: i32) -> i32 {
    if tensor_ptr == 0 || ndim < 0 || ndim > 16 || (ndim > 0 && shape_ptr == 0) {
        return 0;
    }
    let t = &*(tensor_ptr as *const Tensor);
    let new_shape = read_shape(shape_ptr, ndim as usize);
    let new_numel: usize = new_shape.iter().product();
    if new_numel != t.numel {
        return 0;
    }
    // Reshape of a NON-CONTIGUOUS tensor must materialise it first. The
    // transformer feeds `context.permute([1,0,2]).reshape([seq, hidden])`
    // (a strided view) here; the old code just returned null, so attention's
    // output projection saw a null context → zero attention contribution →
    // the residual passed the input straight through → degenerate generation.
    // Copy the strided source into fresh contiguous storage in row-major
    // logical order, then reshape that (mirrors the native reshape).
    if !t.is_contiguous() {
        let out = alloc_tensor(&new_shape, t.dtype);
        if out.is_null() {
            return 0;
        }
        let t_shape = slice::from_raw_parts(t.shape, t.ndim);
        let t_strides = slice::from_raw_parts(t.strides, t.ndim);
        let out_data = (*out).data;
        for flat in 0..t.numel {
            let mut rem = flat;
            let mut off = 0usize;
            for d in (0..t.ndim).rev() {
                let dim = t_shape[d];
                let idx = rem % dim;
                rem /= dim;
                off += idx * t_strides[d];
            }
            store_f32_at(out_data, flat, t.dtype, load_f32_at(t.data, off, t.dtype));
        }
        return out as i32;
    }
    let new_strides = Tensor::compute_strides(&new_shape);
    let shape = alloc_usize_array(&new_shape);
    if shape.is_null() {
        return 0;
    }
    let strides = alloc_usize_array(&new_strides);
    if strides.is_null() {
        let layout = Layout::array::<usize>(new_shape.len().max(1)).unwrap();
        dealloc(shape as *mut u8, layout);
        return 0;
    }
    let wrapper = alloc(Layout::new::<Tensor>()) as *mut Tensor;
    if wrapper.is_null() {
        let layout = Layout::array::<usize>(new_shape.len().max(1)).unwrap();
        dealloc(shape as *mut u8, layout);
        dealloc(strides as *mut u8, layout);
        return 0;
    }
    core::ptr::write(
        wrapper,
        Tensor {
            data: t.data,
            shape,
            strides,
            ndim: new_shape.len(),
            numel: new_numel,
            dtype: t.dtype,
            owns_data: false,
            device: t.device,
            numa_node: t.numa_node,
            refcount: AtomicUsize::new(1),
            parent: tensor_ptr as *mut Tensor,
        },
    );
    t.refcount.fetch_add(1, Ordering::Relaxed);
    wrapper as i32
}

/// Slice along one axis. Returns a zero-copy view with the same strides.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_slice(
    tensor_ptr: i32,
    axis: i32,
    start: i32,
    end: i32,
) -> i32 {
    if tensor_ptr == 0 || axis < 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const Tensor);
    let axis = axis as usize;
    if axis >= t.ndim {
        return 0;
    }
    let old_shape = slice::from_raw_parts(t.shape, t.ndim);
    let old_strides = slice::from_raw_parts(t.strides, t.ndim);
    let dim = old_shape[axis];
    let s = start.max(0) as usize;
    let e = (end.max(0) as usize).min(dim);
    if s >= e {
        return 0;
    }

    let mut new_shape = old_shape.to_vec();
    new_shape[axis] = e - s;
    let new_strides = old_strides.to_vec();
    let new_numel: usize = new_shape.iter().product();

    let shape = alloc_usize_array(&new_shape);
    if shape.is_null() {
        return 0;
    }
    let strides = alloc_usize_array(&new_strides);
    if strides.is_null() {
        let layout = Layout::array::<usize>(new_shape.len().max(1)).unwrap();
        dealloc(shape as *mut u8, layout);
        return 0;
    }
    let wrapper = alloc(Layout::new::<Tensor>()) as *mut Tensor;
    if wrapper.is_null() {
        let layout = Layout::array::<usize>(new_shape.len().max(1)).unwrap();
        dealloc(shape as *mut u8, layout);
        dealloc(strides as *mut u8, layout);
        return 0;
    }
    let byte_offset = s * old_strides[axis] * dtype_size(t.dtype);
    core::ptr::write(
        wrapper,
        Tensor {
            data: t.data.add(byte_offset),
            shape,
            strides,
            ndim: t.ndim,
            numel: new_numel,
            dtype: t.dtype,
            owns_data: false,
            device: t.device,
            numa_node: t.numa_node,
            refcount: AtomicUsize::new(1),
            parent: tensor_ptr as *mut Tensor,
        },
    );
    t.refcount.fetch_add(1, Ordering::Relaxed);
    wrapper as i32
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
    #[cfg(test)]
    {
        if TEST_TRACKED_RELEASE_PTR.load(Ordering::Relaxed) == t as usize {
            TEST_PHYSICAL_RELEASES.fetch_add(1, Ordering::Relaxed);
        }
    }
    dealloc(t as *mut u8, wrapper_layout);
    if !parent.is_null() {
        rayzor_tensor_free(parent as i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of};

    fn test_bytes(bytes: &mut [u8]) -> HaxeBytes {
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
    fn tensor_refcount_slot_is_aligned() {
        assert_eq!(offset_of!(Tensor, refcount) % align_of::<AtomicUsize>(), 0);
        assert_eq!(offset_of!(Tensor, parent) % align_of::<*mut Tensor>(), 0);
    }

    #[test]
    fn f16_full_round_trips_through_get_flat() {
        unsafe {
            let shape = [1i64, 2];
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
    fn native_named_tensor_exports_cover_tensor_demo_path() {
        unsafe {
            let shape_2x3 = [2i64, 3];
            let zero_idx_2d = [0i64, 0];
            let a = rayzor_tensor_ones(shape_2x3.as_ptr() as i32, 2, DTYPE_F32 as i32);
            let b = rayzor_tensor_full(shape_2x3.as_ptr() as i32, 2, 2.0, DTYPE_F32 as i32);
            assert!(a != 0 && b != 0);
            assert_eq!(rayzor_tensor_numel(a), 6);
            assert_eq!(rayzor_tensor_get(b, zero_idx_2d.as_ptr() as i32, 2), 2.0);

            let c = rayzor_tensor_add(a, b);
            assert!(c != 0);
            assert_eq!(rayzor_tensor_get(c, zero_idx_2d.as_ptr() as i32, 2), 3.0);
            assert_eq!(rayzor_tensor_sum(a), 6.0);
            assert_eq!(rayzor_tensor_mean(b), 2.0);

            let v1_data = [1.0f64, 2.0, 3.0];
            let v2_data = [4.0f64, 5.0, 6.0];
            let v1 = rayzor_tensor_from_array(v1_data.as_ptr() as i32, 3, DTYPE_F32 as i32);
            let v2 = rayzor_tensor_from_array(v2_data.as_ptr() as i32, 3, DTYPE_F32 as i32);
            assert!(v1 != 0 && v2 != 0);
            assert_eq!(rayzor_tensor_dot(v1, v2), 32.0);

            let m_shape = [2i64, 2];
            let m1_data = [1.0f64, 2.0, 3.0, 4.0];
            let m2_data = [5.0f64, 6.0, 7.0, 8.0];
            let m1_base = rayzor_tensor_from_array(m1_data.as_ptr() as i32, 4, DTYPE_F32 as i32);
            let m2_base = rayzor_tensor_from_array(m2_data.as_ptr() as i32, 4, DTYPE_F32 as i32);
            let m1 = rayzor_tensor_reshape(m1_base, m_shape.as_ptr() as i32, 2);
            let m2 = rayzor_tensor_reshape(m2_base, m_shape.as_ptr() as i32, 2);
            let m3 = rayzor_tensor_matmul(m1, m2);
            assert!(m3 != 0);
            assert_eq!(rayzor_tensor_get(m3, [0i64, 0].as_ptr() as i32, 2), 19.0);
            assert_eq!(rayzor_tensor_get(m3, [0i64, 1].as_ptr() as i32, 2), 22.0);
            assert_eq!(rayzor_tensor_get(m3, [1i64, 0].as_ptr() as i32, 2), 43.0);
            assert_eq!(rayzor_tensor_get(m3, [1i64, 1].as_ptr() as i32, 2), 50.0);

            let sq_data = [1.0f64, 4.0, 9.0, 16.0];
            let vals = rayzor_tensor_from_array(sq_data.as_ptr() as i32, 4, DTYPE_F32 as i32);
            let sq = rayzor_tensor_sqrt(vals);
            assert!(sq != 0);
            for (idx, want) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
                assert_eq!(
                    rayzor_tensor_get(sq, [idx as i64].as_ptr() as i32, 1),
                    *want
                );
            }

            for h in [sq, vals, m3, m2, m1, m2_base, m1_base, v2, v1, c, b, a] {
                rayzor_tensor_free(h);
            }
        }
    }

    #[test]
    fn bytes_decode_exports_materialize_f32_tensors() {
        unsafe {
            let shape = [2i64, 2];
            let f32_vals = [1.0f32, 2.0, 3.0, 4.0];
            let mut f32_bytes =
                slice::from_raw_parts(f32_vals.as_ptr() as *const u8, f32_vals.len() * 4).to_vec();
            let f32_handle = test_bytes(&mut f32_bytes);
            let t = rayzor_tensor_from_bytes_f32(
                &f32_handle as *const HaxeBytes as i32,
                shape.as_ptr() as i32,
                2,
            );
            assert!(t != 0);
            assert_eq!(rayzor_tensor_sum(t), 10.0);
            rayzor_tensor_free(t);

            let halves = [f16::from_f32(1.5).to_bits(), f16::from_f32(-2.0).to_bits()];
            let mut f16_bytes = std::vec::Vec::new();
            for bits in halves {
                f16_bytes.extend_from_slice(&bits.to_le_bytes());
            }
            let f16_shape = [2i64];
            let f16_handle = test_bytes(&mut f16_bytes);
            let t = rayzor_tensor_from_bytes_f16(
                &f16_handle as *const HaxeBytes as i32,
                f16_shape.as_ptr() as i32,
                1,
            );
            assert!(t != 0);
            assert!((rayzor_tensor_get_flat_f32(t, 0) - 1.5).abs() < 1e-3);
            assert!((rayzor_tensor_get_flat_f32(t, 1) + 2.0).abs() < 1e-3);
            rayzor_tensor_free(t);

            let scale = f16::from_f32(0.5).to_bits();
            let mut q8_bytes = std::vec::Vec::new();
            q8_bytes.extend_from_slice(&scale.to_le_bytes());
            q8_bytes.extend((0..32).map(|i| i as i8 as u8));
            let q8_shape = [32i64];
            let q8_handle = test_bytes(&mut q8_bytes);
            let t = rayzor_tensor_from_bytes_q8_0(
                &q8_handle as *const HaxeBytes as i32,
                q8_shape.as_ptr() as i32,
                1,
            );
            assert!(t != 0);
            assert_eq!(rayzor_tensor_get_flat_f32(t, 2), 1.0);
            assert_eq!(rayzor_tensor_get_flat_f32(t, 31), 15.5);
            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn forward_shape_ops_cover_residual_and_attention_helpers() {
        unsafe {
            let shape = [2i64, 2];
            let a = rayzor_tensor_full(shape.as_ptr() as i32, 2, 1.0, DTYPE_F32 as i32);
            let b = rayzor_tensor_full(shape.as_ptr() as i32, 2, 2.0, DTYPE_F32 as i32);
            rayzor_tensor_add_into(a, b);
            assert_eq!(rayzor_tensor_get_flat_f32(a, 0), 3.0);
            let scaled = rayzor_tensor_scale(a, 0.5);
            assert_eq!(rayzor_tensor_get_flat_f32(scaled, 0), 1.5);

            let masked = rayzor_tensor_causal_mask_(scaled, 0);
            assert_eq!(masked, scaled);
            assert!(rayzor_tensor_get_flat_f32(scaled, 1).is_infinite());

            let trans = rayzor_tensor_transpose_last2(scaled);
            assert!(trans != 0);
            let tr = &*(trans as *const Tensor);
            assert_eq!(slice::from_raw_parts(tr.shape, tr.ndim), [2, 2]);
            let trans2 = rayzor_tensor_transpose(scaled);
            assert!(trans2 != 0);
            let axes = [1i64, 0];
            let perm = rayzor_tensor_permute(scaled, axes.as_ptr() as i32, 2);
            assert!(perm != 0);

            let src_shape = [2i64, 1];
            let dst_shape = [4i64, 1];
            let src = rayzor_tensor_full(src_shape.as_ptr() as i32, 2, 7.0, DTYPE_F32 as i32);
            let dst = rayzor_tensor_zeros(dst_shape.as_ptr() as i32, 2, DTYPE_F32 as i32);
            assert_eq!(rayzor_tensor_broadcast_repeat_0_f32(dst, src, 2), 0);
            assert_eq!(rayzor_tensor_get_flat_f32(dst, 3), 7.0);

            let kv_shape = [2i64, 1, 2];
            let kv = rayzor_tensor_full(kv_shape.as_ptr() as i32, 3, 4.0, DTYPE_F32 as i32);
            let expanded = rayzor_tensor_expand_kv_heads_axis1_f32(kv, 3);
            assert!(expanded != 0);
            let er = &*(expanded as *const Tensor);
            assert_eq!(slice::from_raw_parts(er.shape, er.ndim), [3, 2, 2]);
            assert_eq!(rayzor_tensor_get_flat_f32(expanded, 11), 4.0);

            let indices = [1i64, 0];
            let gathered = rayzor_tensor_gather_rows(a, indices.as_ptr() as i32, 2);
            assert!(gathered != 0);
            assert_eq!(rayzor_tensor_get_flat_f32(gathered, 0), 3.0);

            for h in [
                gathered, expanded, kv, dst, src, perm, trans2, trans, scaled, b, a,
            ] {
                rayzor_tensor_free(h);
            }
        }
    }

    #[test]
    fn arc_clone_defers_physical_free_until_last_drop() {
        unsafe {
            TEST_PHYSICAL_RELEASES.store(0, Ordering::Relaxed);
            let shape = [2usize, 2];
            let t = alloc_tensor(&shape, DTYPE_F32);
            assert!(!t.is_null());
            let h = t as i32;
            TEST_TRACKED_RELEASE_PTR.store(h as usize, Ordering::Relaxed);
            let c = rayzor_tensor_arc_clone(h);
            assert_eq!(c, h);
            assert_eq!((&*t).refcount.load(Ordering::Relaxed), 2);
            rayzor_tensor_free(c);
            assert_eq!((&*t).refcount.load(Ordering::Relaxed), 1);
            assert_eq!(TEST_PHYSICAL_RELEASES.load(Ordering::Relaxed), 0);
            rayzor_tensor_free(h);
            assert_eq!(TEST_PHYSICAL_RELEASES.load(Ordering::Relaxed), 1);
            TEST_TRACKED_RELEASE_PTR.store(0, Ordering::Relaxed);
        }
    }

    #[test]
    fn arc_clone_three_times_releases_on_fourth_free() {
        unsafe {
            TEST_PHYSICAL_RELEASES.store(0, Ordering::Relaxed);
            let shape = [1usize, 4];
            let t = alloc_tensor(&shape, DTYPE_F32);
            let h = t as i32;
            TEST_TRACKED_RELEASE_PTR.store(h as usize, Ordering::Relaxed);
            let c1 = rayzor_tensor_arc_clone(h);
            let c2 = rayzor_tensor_arc_clone(h);
            let c3 = rayzor_tensor_arc_clone(h);
            assert_eq!(c1, h);
            assert_eq!(c2, h);
            assert_eq!(c3, h);
            assert_eq!((&*t).refcount.load(Ordering::Relaxed), 4);

            rayzor_tensor_free(c1);
            assert_eq!(TEST_PHYSICAL_RELEASES.load(Ordering::Relaxed), 0);
            rayzor_tensor_free(c2);
            assert_eq!(TEST_PHYSICAL_RELEASES.load(Ordering::Relaxed), 0);
            rayzor_tensor_free(c3);
            assert_eq!(TEST_PHYSICAL_RELEASES.load(Ordering::Relaxed), 0);
            assert_eq!((&*t).refcount.load(Ordering::Relaxed), 1);
            rayzor_tensor_free(h);
            assert_eq!(TEST_PHYSICAL_RELEASES.load(Ordering::Relaxed), 1);
            TEST_TRACKED_RELEASE_PTR.store(0, Ordering::Relaxed);
        }
    }

    #[test]
    fn append_slice_and_reshape_back_kv_cache_rows() {
        unsafe {
            let cache_shape = [2048i64, 64];
            let row_shape = [1i64, 64];
            let cache = rayzor_tensor_zeros(cache_shape.as_ptr() as i32, 2, DTYPE_F32 as i32);
            assert!(cache != 0);

            for row in 0..10 {
                let src =
                    rayzor_tensor_full(row_shape.as_ptr() as i32, 2, row as f64, DTYPE_F32 as i32);
                assert!(src != 0);
                assert_eq!(rayzor_tensor_append_along_0_f32(cache, src, row), 0);
                rayzor_tensor_free(src);
            }

            let view = rayzor_tensor_slice(cache, 0, 0, 10);
            assert!(view != 0);
            let view_ref = &*(view as *const Tensor);
            assert!(!view_ref.owns_data);
            assert_eq!(view_ref.parent, cache as *mut Tensor);
            assert_eq!(
                (&*(cache as *const Tensor))
                    .refcount
                    .load(Ordering::Relaxed),
                2
            );

            let flat_shape = [10i64, 64];
            let reshaped = rayzor_tensor_reshape(view, flat_shape.as_ptr() as i32, 2);
            assert!(reshaped != 0);
            let reshaped_ref = &*(reshaped as *const Tensor);
            assert!(!reshaped_ref.owns_data);
            assert_eq!(reshaped_ref.numel, 640);

            let data = reshaped_ref.data as *const f32;
            for col in 0..64 {
                assert_eq!(*data.add(9 * 64 + col), 9.0);
            }

            rayzor_tensor_free(reshaped);
            rayzor_tensor_free(view);
            rayzor_tensor_free(cache);
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
