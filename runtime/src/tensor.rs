//! Tensor runtime — N-dimensional array with shape, strides, and dtype.
//!
//! A Tensor is a heap-allocated struct containing:
//! - data: *mut f32 (or other dtype, but f32 is the primary path)
//! - shape: Vec<usize> stored as (ptr, len) inline
//! - strides: Vec<usize> stored as (ptr, len) inline
//! - dtype: u8 tag
//! - numel: usize
//! - ndim: usize
//! - rc: reference count for shared views
//!
//! At MIR/Haxe level, a Tensor is an opaque i64 (pointer).
//! All extern functions take/return i64 to match the type system.

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

use half::{bf16, f16};

// DType tags matching the Haxe enum order.
// Stays in sync with `compiler/haxe-std/rayzor/ds/DType.hx` and the GPU-side
// constants in `gpu/src/buffer.rs`. Order is load-bearing: the Haxe enum
// produces ordinals 0..N matching these values, and MIR call sites pass
// the ordinal as the dtype arg.
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
pub const DTYPE_I32: u8 = 3;
pub const DTYPE_I8: u8 = 4;
pub const DTYPE_U8: u8 = 5;
pub const DTYPE_FP8_E4M3: u8 = 6;
pub const DTYPE_FP8_E5M2: u8 = 7;

// Device tags matching the Haxe Device enum.
// CPU(node) collapses node into the separate numa_node field on the struct
// — the tag itself is just "this lives on the CPU side". Vulkan and WebGPU
// both go through the `wgpu` crate at the runtime layer; the tag merely
// records which adapter family was selected.
pub(crate) const DEVICE_CPU: u8 = 0;
#[allow(dead_code)]
pub(crate) const DEVICE_METAL: u8 = 1;
#[allow(dead_code)]
pub(crate) const DEVICE_CUDA: u8 = 2;
#[allow(dead_code)]
pub(crate) const DEVICE_VULKAN: u8 = 3;
#[allow(dead_code)]
pub(crate) const DEVICE_WEBGPU: u8 = 4;

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        DTYPE_F16 => 2,
        DTYPE_BF16 => 2,
        DTYPE_I32 => 4,
        DTYPE_I8 => 1,
        DTYPE_U8 => 1,
        DTYPE_FP8_E4M3 => 1,
        DTYPE_FP8_E5M2 => 1,
        _ => 4, // default to f32
    }
}

// ============================================================================
// Per-element load / store helpers (storage ↔ f32)
//
// These convert one element at an arbitrary `*mut u8` data pointer between
// the storage representation (specified by the dtype tag) and f32 — the
// common compute representation. Phase 3b runs the kernels themselves in
// f32 and converts on load/store; Phase 3c specialises hot paths to native
// NEON f16 / x86 F16C where available.
//
// The fp8 paths (DTYPE_FP8_E4M3 / DTYPE_FP8_E5M2) implement a software
// dequant since neither the host CPU nor the `half` crate has built-in
// fp8 support. Quant-on-store is performed with naive round-to-nearest;
// kernels operating on fp8 weights are dequant-on-load only in Phase 3e.
// ============================================================================

#[inline(always)]
unsafe fn load_f32_at(data: *const u8, idx: usize, dtype: u8) -> f32 {
    match dtype {
        DTYPE_F32 => *(data as *const f32).add(idx),
        DTYPE_F16 => f16::from_bits(*(data as *const u16).add(idx)).to_f32(),
        DTYPE_BF16 => bf16::from_bits(*(data as *const u16).add(idx)).to_f32(),
        DTYPE_I32 => *(data as *const i32).add(idx) as f32,
        DTYPE_I8 => *(data as *const i8).add(idx) as f32,
        DTYPE_U8 => *data.add(idx) as f32,
        // FP8 reads go through a 256-entry LUT — ~16x faster than bit-twiddling.
        // See `init_fp8_luts` for the precomputed tables.
        DTYPE_FP8_E4M3 => FP8_E4M3_LUT[*data.add(idx) as usize],
        DTYPE_FP8_E5M2 => FP8_E5M2_LUT[*data.add(idx) as usize],
        _ => 0.0,
    }
}

#[inline(always)]
unsafe fn store_f32_at(data: *mut u8, idx: usize, dtype: u8, value: f32) {
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

/// IEEE 754 FP8 E4M3 — 1 sign + 4 exponent + 3 mantissa bits, no infinity,
/// only one NaN encoding (0x7F / 0xFF). Bias = 7. Range ≈ ±448, ULP at 1.0
/// is 1/8. Common dequant target for INT4 → FP8 → FP16 pipelines.
///
/// Note: hot-path FP8 reads now go through `FP8_E4M3_LUT` (precomputed
/// 256-entry table). This function remains as the reference implementation
/// and is exercised by the LUT-construction const path below.
#[allow(dead_code)]
fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 3) & 0xF;
    let mant = byte & 0x7;
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    let s = if sign == 0 { 1.0f32 } else { -1.0f32 };
    if exp == 0 {
        // subnormal
        s * (mant as f32) * (1.0 / 8.0) * 2.0f32.powi(-6)
    } else {
        let m = 1.0f32 + (mant as f32) / 8.0;
        s * m * 2.0f32.powi(exp as i32 - 7)
    }
}

fn fp8_e4m3_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    let abs = value.abs();
    if abs == 0.0 {
        return sign_bit;
    }
    // Clamp to E4M3 range (max = 448.0)
    let clamped = abs.min(448.0);
    let bits = clamped.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7FFFFF;

    if f32_exp < -6 {
        // Subnormal in E4M3 (or zero)
        // value = mant_e4m3 / 8 * 2^-6
        let scaled = (clamped / 2.0f32.powi(-6)) * 8.0;
        let m = scaled.round().clamp(0.0, 7.0) as u8;
        return sign_bit | m;
    }
    let exp_e4m3 = f32_exp + 7;
    if exp_e4m3 >= 0xF {
        // Saturate just below the NaN encoding
        return sign_bit | ((0xF) << 3) | 0x6;
    }
    // Round mantissa: f32 has 23 bits, E4M3 has 3.
    let m = (f32_mant >> 20) as u8;
    // Round-to-nearest-even on the discarded bits
    let lsb_mask = 1u32 << 20;
    let half_mask = 1u32 << 19;
    let round_bits = f32_mant & ((1u32 << 20) - 1);
    let rounded =
        if round_bits > half_mask || (round_bits == half_mask && (f32_mant & lsb_mask) != 0) {
            m.saturating_add(1)
        } else {
            m
        };
    if rounded > 7 {
        // mantissa overflow → bump exponent
        let new_exp = (exp_e4m3 as u8) + 1;
        if new_exp >= 0xF {
            return sign_bit | (0xF << 3) | 0x6;
        }
        sign_bit | (new_exp << 3)
    } else {
        sign_bit | ((exp_e4m3 as u8) << 3) | rounded
    }
}

/// IEEE 754 FP8 E5M2 — 1 sign + 5 exponent + 2 mantissa bits, supports inf
/// and standard NaN encodings. Bias = 15. Range ≈ ±57344, ULP at 1.0 = 1/4.
///
/// Hot-path FP8 E5M2 reads go through `FP8_E5M2_LUT`; this remains as the
/// reference for the LUT-build const path.
#[allow(dead_code)]
fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x3;
    if exp == 0 && mant == 0 {
        return if sign == 0 { 0.0 } else { -0.0 };
    }
    if exp == 0x1F {
        if mant == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return f32::NAN;
    }
    let s = if sign == 0 { 1.0f32 } else { -1.0f32 };
    if exp == 0 {
        s * (mant as f32) * (1.0 / 4.0) * 2.0f32.powi(-14)
    } else {
        let m = 1.0f32 + (mant as f32) / 4.0;
        s * m * 2.0f32.powi(exp as i32 - 15)
    }
}

fn fp8_e5m2_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7F;
    }
    let sign_bit: u8 = if value.is_sign_negative() { 0x80 } else { 0 };
    if value.is_infinite() {
        return sign_bit | (0x1F << 2);
    }
    let abs = value.abs();
    if abs == 0.0 {
        return sign_bit;
    }
    let bits = abs.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = bits & 0x7FFFFF;
    if f32_exp < -14 {
        let scaled = (abs / 2.0f32.powi(-14)) * 4.0;
        let m = scaled.round().clamp(0.0, 3.0) as u8;
        return sign_bit | m;
    }
    let exp_e5m2 = f32_exp + 15;
    if exp_e5m2 >= 0x1F {
        return sign_bit | (0x1F << 2); // saturate to ±inf
    }
    let m = (f32_mant >> 21) as u8 & 0x3;
    let half_mask = 1u32 << 20;
    let round_bits = f32_mant & ((1u32 << 21) - 1);
    let rounded =
        if round_bits > half_mask || (round_bits == half_mask && (f32_mant >> 21) & 0x1 != 0) {
            m.saturating_add(1)
        } else {
            m
        };
    if rounded > 3 {
        let new_exp = (exp_e5m2 as u8) + 1;
        if new_exp >= 0x1F {
            return sign_bit | (0x1F << 2);
        }
        sign_bit | (new_exp << 2)
    } else {
        sign_bit | ((exp_e5m2 as u8) << 2) | rounded
    }
}

/// Precomputed 256-entry lookup tables for FP8 → f32. Generated once via
/// the const-eval-friendly construction below — both fp8 formats only have
/// 256 possible byte values, so the entire dequant is a single load.
const FP8_E4M3_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e4m3_to_f32_const(i as u8);
        i += 1;
    }
    t
};

const FP8_E5M2_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e5m2_to_f32_const(i as u8);
        i += 1;
    }
    t
};

/// const-friendly versions of the fp8 decoders — same arithmetic, but
/// expressed with operations that work inside `const fn` (no .powi, no
/// transcendentals, no float arithmetic that hits the runtime softfloat).
/// Computed at compile-time as part of `FP8_*_LUT` construction.
const fn fp8_e4m3_to_f32_const(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 3) & 0xF;
    let mant = byte & 0x7;
    if exp == 0 && mant == 0 {
        // Const fn can't express -0.0 literal cleanly; LUT consumer treats
        // both as zero so this is fine.
        return 0.0;
    }
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    // value = sign * mantissa * 2^(exp - bias)
    // mantissa = (exp==0 ? mant/8 : 1 + mant/8)
    // bias = 7
    let mant_num = if exp == 0 {
        mant as u32
    } else {
        8 + mant as u32
    };
    let exp_val = if exp == 0 { -9i32 } else { exp as i32 - 10 };
    let magnitude = const_pow2(mant_num as f32, exp_val);
    if sign == 0 {
        magnitude
    } else {
        -magnitude
    }
}

const fn fp8_e5m2_to_f32_const(byte: u8) -> f32 {
    let sign = (byte >> 7) & 0x1;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x3;
    if exp == 0 && mant == 0 {
        return 0.0;
    }
    if exp == 0x1F {
        if mant == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return f32::NAN;
    }
    let mant_num = if exp == 0 {
        mant as u32
    } else {
        4 + mant as u32
    };
    let exp_val = if exp == 0 { -16i32 } else { exp as i32 - 17 };
    let magnitude = const_pow2(mant_num as f32, exp_val);
    if sign == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// `n * 2^k` in const context. Bit-manipulates the f32 representation
/// directly so we avoid runtime float ops (which const-eval forbids on
/// stable). Only valid for finite `n` and normal-range `k`.
const fn const_pow2(n: f32, k: i32) -> f32 {
    if n == 0.0 {
        return 0.0;
    }
    let bits = n.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    let new_exp = exp_bits + k;
    if new_exp <= 0 {
        // Underflow to zero — close enough for LUT semantics.
        return 0.0;
    }
    if new_exp >= 0xFF {
        return f32::INFINITY;
    }
    let mant_and_sign = bits & 0x807FFFFF;
    f32::from_bits(mant_and_sign | ((new_exp as u32) << 23))
}

/// Bulk-fill `numel` elements at `data` with the given f32 value, in the
/// storage representation specified by `dtype`. Used by ones/full/zeros.
#[inline]
unsafe fn fill_dtype(data: *mut u8, numel: usize, dtype: u8, value: f32) {
    if value == 0.0 {
        // Every supported storage format encodes +0.0 as all zero bytes.
        std::ptr::write_bytes(data, 0, numel * dtype_size(dtype));
        return;
    }
    for i in 0..numel {
        store_f32_at(data, i, dtype, value);
    }
}

/// Internal tensor representation
#[repr(C)]
struct RayzorTensor {
    data: *mut u8,
    shape: *mut usize,
    strides: *mut usize,
    ndim: usize,
    numel: usize,
    dtype: u8,
    owns_data: bool, // false for views
    // Device placement. `device` is the device tag (DEVICE_CPU/DEVICE_METAL/
    // DEVICE_CUDA/DEVICE_WEBGPU). `numa_node` is meaningful only when
    // device == DEVICE_CPU: -1 means "no affinity hint", >= 0 names a NUMA
    // node from rayzor.concurrent.NumaTopology. Phase 1a default: every
    // existing allocation tags itself CPU/-1.
    device: u8,
    numa_node: i32,
}

impl RayzorTensor {
    /// Compute row-major strides from shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        if ndim == 0 {
            return vec![];
        }
        let mut strides = vec![0usize; ndim];
        strides[ndim - 1] = 1;
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Compute flat offset from multi-dimensional indices
    fn offset(&self, indices: &[usize]) -> usize {
        let strides = unsafe { std::slice::from_raw_parts(self.strides, self.ndim) };
        let mut off = 0usize;
        for i in 0..self.ndim {
            off += indices[i] * strides[i];
        }
        off
    }
}

/// Allocate a new tensor struct on the heap, return as i64
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
unsafe fn alloc_tensor(shape: &[usize], dtype: u8, fill: Option<f32>) -> i64 {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    let elem_size = dtype_size(dtype);
    let data_bytes = numel * elem_size;

    // Allocate data
    let data = malloc(if data_bytes > 0 { data_bytes } else { 1 });
    if data.is_null() {
        return 0;
    }

    // Fill data
    if let Some(val) = fill {
        fill_dtype(data, numel, dtype, val);
    } else {
        std::ptr::write_bytes(data, 0, data_bytes);
    }

    // Allocate shape array
    let shape_ptr = malloc(ndim * std::mem::size_of::<usize>()) as *mut usize;
    if shape_ptr.is_null() {
        free(data);
        return 0;
    }
    for i in 0..ndim {
        *shape_ptr.add(i) = shape[i];
    }

    // Compute and allocate strides
    let strides = RayzorTensor::compute_strides(shape);
    let strides_ptr = malloc(ndim * std::mem::size_of::<usize>()) as *mut usize;
    if strides_ptr.is_null() {
        free(data);
        free(shape_ptr as *mut u8);
        return 0;
    }
    for i in 0..ndim {
        *strides_ptr.add(i) = strides[i];
    }

    // Allocate tensor struct
    let tensor = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if tensor.is_null() {
        free(data);
        free(shape_ptr as *mut u8);
        free(strides_ptr as *mut u8);
        return 0;
    }

    *tensor = RayzorTensor {
        data,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim,
        numel,
        dtype,
        owns_data: true,
        device: DEVICE_CPU,
        numa_node: -1,
    };

    tensor as i64
}

// ============================================================================
// Construction
// ============================================================================

/// Tensor.zeros(shape_ptr: i64, ndim: i64, dtype: i64) -> i64
///
/// shape_ptr is a pointer to an array of i64 shape values (from Haxe Array<Int>).
/// We read ndim elements, convert to usize, and create the tensor.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_zeros(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8, Some(0.0))
}

/// Tensor.ones(shape_ptr, ndim, dtype) -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ones(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8, Some(1.0))
}

/// Tensor.full(shape_ptr, ndim, value, dtype) -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_full(
    shape_ptr: i64,
    ndim: i64,
    value: f64,
    dtype: i64,
) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    alloc_tensor(&shape, dtype as u8, Some(value as f32))
}

/// Tensor.fromArray(data_ptr, data_len, dtype) -> i64
/// Creates a 1-D tensor with shape=[data_len] from a flat array of f64 values.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_array(data_ptr: i64, data_len: i64, dtype: i64) -> i64 {
    let numel = data_len as usize;
    let shape = vec![numel];
    let dtype_u8 = dtype as u8;

    let tensor_ptr = alloc_tensor(&shape, dtype_u8, None);
    if tensor_ptr == 0 {
        return 0;
    }

    let tensor = &*(tensor_ptr as *const RayzorTensor);

    // Copy f64 data from Haxe Array<Float>, converting to target dtype.
    // Goes through the generic store_f32_at helper so every supported
    // dtype (F32 / F16 / BF16 / I32 / I8 / U8 / FP8_E4M3 / FP8_E5M2) is
    // populated with the right storage format.
    let src = data_ptr as *const f64;
    for i in 0..numel {
        store_f32_at(tensor.data, i, dtype_u8, *src.add(i) as f32);
    }

    tensor_ptr
}

/// Tensor.rand(shape_ptr, ndim, dtype) -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rand(shape_ptr: i64, ndim: i64, dtype: i64) -> i64 {
    let shape = read_shape(shape_ptr, ndim as usize);
    let tensor_ptr = alloc_tensor(&shape, dtype as u8, None);
    if tensor_ptr == 0 {
        return 0;
    }

    let tensor = &*(tensor_ptr as *const RayzorTensor);

    // Simple LCG random for deterministic "random" init
    if tensor.dtype == DTYPE_F32 {
        let dst = tensor.data as *mut f32;
        let mut seed: u64 = 0xDEADBEEF_CAFEBABE;
        for i in 0..tensor.numel {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = ((seed >> 33) as u32) & 0x7FFFFF; // 23 bits mantissa
            let val = (bits as f32) / (0x800000 as f32); // [0, 1)
            *dst.add(i) = val;
        }
    }

    tensor_ptr
}

// ============================================================================
// Properties
// ============================================================================

/// tensor.ndim() -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ndim(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.ndim as i64
}

/// tensor.numel() -> i64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_numel(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.numel as i64
}

/// tensor.device() -> i64 (returns device tag: 0=CPU, 1=Metal, 2=Cuda, 3=WebGPU)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_device(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return DEVICE_CPU as i64;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.device as i64
}

/// tensor.numa_node() -> i64 (NUMA node hint when device == CPU; -1 means "any")
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_numa_node(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return -1;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.numa_node as i64
}

/// tensor.dtype() -> i64 (returns dtype tag)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_dtype(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.dtype as i64
}

/// tensor.shape() -> i64 (returns pointer to a heap-allocated HaxeArray of Int)
///
/// Allocates a HaxeArray struct + data buffer, copies shape dims as i64 values.
/// HaxeArray layout: { ptr: *mut u8, len: usize, cap: usize, elem_size: usize }
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    let ndim = t.ndim;
    let shape_slice = std::slice::from_raw_parts(t.shape, ndim);

    // Allocate HaxeArray struct (4 fields x 8 bytes = 32 bytes)
    let arr_ptr = malloc(32) as *mut usize;
    if arr_ptr.is_null() {
        return 0;
    }

    // Allocate data buffer for ndim i64 elements
    let elem_size = std::mem::size_of::<i64>();
    let cap = ndim.max(8); // match HaxeArray INITIAL_CAPACITY
    let data_ptr = malloc(cap * elem_size);
    if data_ptr.is_null() {
        free(arr_ptr as *mut u8);
        return 0;
    }

    // Copy shape values as i64
    let data_i64 = data_ptr as *mut i64;
    for (i, &val) in shape_slice[..ndim].iter().enumerate() {
        *data_i64.add(i) = val as i64;
    }

    // Fill HaxeArray fields: ptr, len, cap, elem_size
    *arr_ptr.add(0) = data_ptr as usize; // ptr
    *arr_ptr.add(1) = ndim; // len
    *arr_ptr.add(2) = cap; // cap
    *arr_ptr.add(3) = elem_size; // elem_size

    arr_ptr as i64
}

/// tensor.shape_ptr() -> i64 (returns raw pointer to shape data, for internal use)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape_ptr(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.shape as i64
}

/// tensor.shape_ndim() -> i64 (helper: returns ndim for shape access)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_shape_ndim(tensor_ptr: i64) -> i64 {
    rayzor_tensor_ndim(tensor_ptr)
}

// ============================================================================
// Element access
// ============================================================================

/// tensor.get(indices_ptr, ndim) -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get(tensor_ptr: i64, indices_ptr: i64, ndim: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    let indices = read_shape(indices_ptr, ndim as usize);
    let off = t.offset(&indices);
    load_f32_at(t.data, off, t.dtype) as f64
}

/// tensor.set(indices_ptr, ndim, value) -> void
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_set(
    tensor_ptr: i64,
    indices_ptr: i64,
    ndim: i64,
    value: f64,
) {
    if tensor_ptr == 0 {
        return;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    let indices = read_shape(indices_ptr, ndim as usize);
    let off = t.offset(&indices);
    store_f32_at(t.data, off, t.dtype, value as f32);
}

// ============================================================================
// Reshape / Transpose
// ============================================================================

/// tensor.reshape(shape_ptr, ndim) -> i64 (new tensor, shared data)
#[no_mangle]
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
pub unsafe extern "C" fn rayzor_tensor_reshape(tensor_ptr: i64, shape_ptr: i64, ndim: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    let new_shape = read_shape(shape_ptr, ndim as usize);
    let new_numel: usize = new_shape.iter().product();

    // Verify numel matches
    if new_numel != t.numel {
        return 0; // shape mismatch
    }

    let new_ndim = new_shape.len();

    // Allocate new shape
    let new_shape_ptr = malloc(new_ndim * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() {
        return 0;
    }
    for i in 0..new_ndim {
        *new_shape_ptr.add(i) = new_shape[i];
    }

    // Compute new strides
    let strides = RayzorTensor::compute_strides(&new_shape);
    let new_strides_ptr = malloc(new_ndim * std::mem::size_of::<usize>()) as *mut usize;
    if new_strides_ptr.is_null() {
        free(new_shape_ptr as *mut u8);
        return 0;
    }
    for i in 0..new_ndim {
        *new_strides_ptr.add(i) = strides[i];
    }

    // Allocate new tensor struct (view — shares data)
    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        free(new_shape_ptr as *mut u8);
        free(new_strides_ptr as *mut u8);
        return 0;
    }

    *new_t = RayzorTensor {
        data: t.data, // shared
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: new_ndim,
        numel: new_numel,
        dtype: t.dtype,
        owns_data: false, // view
        device: t.device,
        numa_node: t.numa_node,
    };

    new_t as i64
}

/// tensor.permute(axes_ptr, ndim) -> i64 (n-D permutation — reorders shape/strides, view)
#[no_mangle]
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
pub unsafe extern "C" fn rayzor_tensor_permute(
    tensor_ptr: i64,
    axes_ptr: i64,
    axes_len: i64,
) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    let n = axes_len as usize;
    if n != t.ndim {
        return 0;
    }

    let axes_data = axes_ptr as *const i64;
    let mut seen = vec![false; n];
    let mut axes = vec![0usize; n];
    for i in 0..n {
        let ax = *axes_data.add(i) as usize;
        if ax >= n || seen[ax] {
            return 0;
        }
        seen[ax] = true;
        axes[i] = ax;
    }

    let old_shape = std::slice::from_raw_parts(t.shape, n);
    let old_strides = std::slice::from_raw_parts(t.strides, n);

    let new_shape_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }
    for i in 0..n {
        *new_shape_ptr.add(i) = old_shape[axes[i]];
        *new_strides_ptr.add(i) = old_strides[axes[i]];
    }

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        return 0;
    }
    *new_t = RayzorTensor {
        data: t.data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: n,
        numel: t.numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
    };
    new_t as i64
}

/// tensor.slice(dim, start, end) -> i64 (view over [start..end) along `dim`, view)
#[no_mangle]
#[allow(clippy::manual_slice_size_calculation, clippy::needless_range_loop)]
pub unsafe extern "C" fn rayzor_tensor_slice(
    tensor_ptr: i64,
    dim: i64,
    start: i64,
    end: i64,
) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    let d = dim as usize;
    if d >= t.ndim {
        return 0;
    }

    let old_shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let old_strides = std::slice::from_raw_parts(t.strides, t.ndim);
    let dim_size = old_shape[d];

    let s = start.max(0) as usize;
    let e = (end as usize).min(dim_size);
    if s >= e {
        return 0;
    }
    let new_dim_size = e - s;

    let new_shape_ptr = malloc(t.ndim * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(t.ndim * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }
    for i in 0..t.ndim {
        *new_shape_ptr.add(i) = if i == d { new_dim_size } else { old_shape[i] };
        *new_strides_ptr.add(i) = old_strides[i];
    }

    let mut new_numel = 1usize;
    for i in 0..t.ndim {
        new_numel *= *new_shape_ptr.add(i);
    }

    // Offset data pointer by s * stride[d] elements
    let elem_size = dtype_size(t.dtype);
    let byte_offset = s * old_strides[d] * elem_size;
    let new_data = t.data.add(byte_offset);

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        return 0;
    }
    *new_t = RayzorTensor {
        data: new_data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: t.ndim,
        numel: new_numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
    };
    new_t as i64
}

/// tensor.transpose() -> i64 (2D transpose — swaps shape/strides)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_transpose(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    if t.ndim != 2 {
        return tensor_ptr;
    } // no-op for non-2D

    let old_shape = std::slice::from_raw_parts(t.shape, 2);
    let old_strides = std::slice::from_raw_parts(t.strides, 2);

    let new_shape_ptr = malloc(2 * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(2 * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }

    *new_shape_ptr.add(0) = old_shape[1];
    *new_shape_ptr.add(1) = old_shape[0];
    *new_strides_ptr.add(0) = old_strides[1];
    *new_strides_ptr.add(1) = old_strides[0];

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        return 0;
    }

    *new_t = RayzorTensor {
        data: t.data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: 2,
        numel: t.numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
    };

    new_t as i64
}

// ============================================================================
// Elementwise arithmetic
// ============================================================================

/// Set up output tensor and return (a_slice, b_slice, r_slice) for an
/// elementwise binary f32 op. Returns 0 result and `None` if the op is
/// invalid (shape mismatch, non-f32 dtype). Assumes contiguous f32 data.
#[allow(clippy::type_complexity)]
unsafe fn prepare_binop<'a>(
    a_ptr: i64,
    b_ptr: i64,
) -> Option<(&'a [f32], &'a [f32], &'a mut [f32], i64)> {
    if a_ptr == 0 || b_ptr == 0 {
        return None;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    if a.numel != b.numel || a.dtype != DTYPE_F32 || b.dtype != DTYPE_F32 {
        return None;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return None;
    }

    let r = &*(result as *const RayzorTensor);
    let n = a.numel;
    let a_slice = std::slice::from_raw_parts(a.data as *const f32, n);
    let b_slice = std::slice::from_raw_parts(b.data as *const f32, n);
    let r_slice = std::slice::from_raw_parts_mut(r.data as *mut f32, n);
    Some((a_slice, b_slice, r_slice, result))
}

/// Scalar fallback for elementwise binary ops on non-f32 dtypes.
/// Both inputs must share dtype + numel. The output tensor is allocated
/// in the same dtype, kernel runs in f32 in-register.
unsafe fn tensor_binop_scalar(a_ptr: i64, b_ptr: i64, op: fn(f32, f32) -> f32) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.numel != b.numel || a.dtype != b.dtype {
        return 0;
    }
    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    for i in 0..a.numel {
        let av = load_f32_at(a.data, i, a.dtype);
        let bv = load_f32_at(b.data, i, a.dtype);
        store_f32_at(r.data, i, a.dtype, op(av, bv));
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_add(a: i64, b: i64) -> i64 {
    match prepare_binop(a, b) {
        Some((a_s, b_s, r_s, result)) => {
            crate::tensor_simd::add_slice(r_s, a_s, b_s);
            result
        }
        None => tensor_binop_scalar(a, b, |x, y| x + y),
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sub(a: i64, b: i64) -> i64 {
    match prepare_binop(a, b) {
        Some((a_s, b_s, r_s, result)) => {
            crate::tensor_simd::sub_slice(r_s, a_s, b_s);
            result
        }
        None => tensor_binop_scalar(a, b, |x, y| x - y),
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mul(a: i64, b: i64) -> i64 {
    match prepare_binop(a, b) {
        Some((a_s, b_s, r_s, result)) => {
            crate::tensor_simd::mul_slice(r_s, a_s, b_s);
            result
        }
        None => tensor_binop_scalar(a, b, |x, y| x * y),
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_div(a: i64, b: i64) -> i64 {
    match prepare_binop(a, b) {
        Some((a_s, b_s, r_s, result)) => {
            crate::tensor_simd::div_slice(r_s, a_s, b_s);
            result
        }
        None => tensor_binop_scalar(a, b, |x, y| x / y),
    }
}

// ============================================================================
// Unary math ops
// ============================================================================

unsafe fn tensor_unary(a_ptr: i64, op: fn(f32) -> f32) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);

    if a.dtype == DTYPE_F32 {
        // Fast path: contiguous f32 → SIMD-friendly straight loop. The
        // SIMD-specialised unary kernels live in tensor_simd; non-SIMD
        // ops (e.g. transcendentals) stay scalar but in-register.
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for i in 0..a.numel {
            *r_data.add(i) = op(*a_data.add(i));
        }
    } else {
        // Generic dtype path — convert to f32, compute, convert back.
        for i in 0..a.numel {
            let v = load_f32_at(a.data, i, a.dtype);
            store_f32_at(r.data, i, a.dtype, op(v));
        }
    }

    result
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sqrt(a: i64) -> i64 {
    tensor_unary(a, |x| x.sqrt())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_exp(a: i64) -> i64 {
    tensor_unary(a, |x| x.exp())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_log(a: i64) -> i64 {
    tensor_unary(a, |x| x.ln())
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_relu(a_ptr: i64) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let n = a.numel;

    if a.dtype == DTYPE_F32 {
        let a_s = std::slice::from_raw_parts(a.data as *const f32, n);
        let r_s = std::slice::from_raw_parts_mut(r.data as *mut f32, n);
        crate::tensor_simd::relu_slice(r_s, a_s);
    } else {
        for i in 0..n {
            let v = load_f32_at(a.data, i, a.dtype);
            store_f32_at(r.data, i, a.dtype, v.max(0.0));
        }
    }
    result
}

/// GELU (approximate, tanh-based) — matches PyTorch `gelu(approximate='tanh')`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gelu(a: i64) -> i64 {
    tensor_unary(a, |x| {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let inner = c * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    })
}

/// SiLU / swish: x * sigmoid(x).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_silu(a: i64) -> i64 {
    tensor_unary(a, |x| x / (1.0 + (-x).exp()))
}

/// Softmax over the last dimension.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_softmax(a_ptr: i64) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for g in 0..groups {
            let base = g * last;
            let a_row = std::slice::from_raw_parts(a_data.add(base), last);
            let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
            let maxv = crate::tensor_simd::max_slice(a_row);
            for i in 0..last {
                r_row[i] = (a_row[i] - maxv).exp();
            }
            let sum = crate::tensor_simd::sum_slice(r_row);
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in r_row.iter_mut() {
                    *v *= inv;
                }
            }
        }
        return result;
    }

    // Generic dtype path: f32-in-register softmax with storage conversion.
    let mut row_buf = vec![0.0f32; last];
    for g in 0..groups {
        let base = g * last;
        for (i, slot) in row_buf.iter_mut().enumerate() {
            *slot = load_f32_at(a.data, base + i, a.dtype);
        }
        let mut maxv = f32::NEG_INFINITY;
        for &v in &row_buf {
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row_buf.iter_mut() {
            *v = (*v - maxv).exp();
            sum += *v;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for (i, &v) in row_buf.iter().enumerate() {
            store_f32_at(r.data, base + i, a.dtype, v * inv);
        }
    }
    result
}

/// Layer normalization over the last dimension. (x - mean) / sqrt(var + eps).
/// `eps` is passed as f64 from Haxe.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_layer_norm(a_ptr: i64, eps: f64) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps_f32 = eps as f32;
    let n = last as f32;

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for g in 0..groups {
            let base = g * last;
            let a_row = std::slice::from_raw_parts(a_data.add(base), last);
            let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
            let mean = crate::tensor_simd::sum_slice(a_row) / n;
            crate::tensor_simd::sub_const_slice(r_row, a_row, mean);
            let var = crate::tensor_simd::sum_of_squares(r_row) / n;
            let inv = 1.0 / (var + eps_f32).sqrt();
            for v in r_row.iter_mut() {
                *v *= inv;
            }
        }
        return result;
    }

    // Generic dtype path: f32-in-register stats with storage conversion.
    let mut row_buf = vec![0.0f32; last];
    for g in 0..groups {
        let base = g * last;
        let mut sum = 0.0f32;
        for (i, slot) in row_buf.iter_mut().enumerate() {
            *slot = load_f32_at(a.data, base + i, a.dtype);
            sum += *slot;
        }
        let mean = sum / n;
        let mut sumsq = 0.0f32;
        for v in row_buf.iter_mut() {
            *v -= mean;
            sumsq += *v * *v;
        }
        let inv = 1.0 / (sumsq / n + eps_f32).sqrt();
        for (i, &v) in row_buf.iter().enumerate() {
            store_f32_at(r.data, base + i, a.dtype, v * inv);
        }
    }
    result
}

/// RMS normalization over the last dimension. x / sqrt(mean(x^2) + eps).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rms_norm(a_ptr: i64, eps: f64) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, a.dtype, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps_f32 = eps as f32;
    let n = last as f32;

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let r_data = r.data as *mut f32;
        for g in 0..groups {
            let base = g * last;
            let a_row = std::slice::from_raw_parts(a_data.add(base), last);
            let r_row = std::slice::from_raw_parts_mut(r_data.add(base), last);
            let ms = crate::tensor_simd::sum_of_squares(a_row) / n;
            let inv = 1.0 / (ms + eps_f32).sqrt();
            crate::tensor_simd::mul_const_slice(r_row, a_row, inv);
        }
        return result;
    }

    for g in 0..groups {
        let base = g * last;
        let mut sumsq = 0.0f32;
        for i in 0..last {
            let v = load_f32_at(a.data, base + i, a.dtype);
            sumsq += v * v;
        }
        let inv = 1.0 / (sumsq / n + eps_f32).sqrt();
        for i in 0..last {
            let v = load_f32_at(a.data, base + i, a.dtype);
            store_f32_at(r.data, base + i, a.dtype, v * inv);
        }
    }
    result
}

/// Apply rotary position embedding (RoPE) to a 3-D tensor of shape
/// `[seq_len, num_heads, head_dim]` (or 2-D `[seq_len, head_dim]` with
/// `num_heads = 1`).
///
/// `cos` and `sin` are 2-D tables of shape `[max_seq_len, head_dim / 2]`
/// that were precomputed once for the model's max context length. Only the
/// first `seq_len` rows are consumed; the rest are ignored. `position_offset`
/// adds to the row index — used by the KV-cache decode path when generating
/// token `T` so the new query gets rotated for absolute position `T`.
///
/// The standard Llama rotation acts on adjacent pairs `(x_{2i}, x_{2i+1})`:
/// ```text
///   x_{2i}'   = x_{2i}   * cos[p, i] - x_{2i+1} * sin[p, i]
///   x_{2i+1}' = x_{2i}   * sin[p, i] + x_{2i+1} * cos[p, i]
/// ```
/// where `p` is the absolute position (`row + position_offset`).
///
/// Returns a new tensor with the same shape + dtype.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope(
    x_ptr: i64,
    cos_ptr: i64,
    sin_ptr: i64,
    position_offset: i64,
) -> i64 {
    if x_ptr == 0 || cos_ptr == 0 || sin_ptr == 0 {
        return 0;
    }
    let x = &*(x_ptr as *const RayzorTensor);
    let cos = &*(cos_ptr as *const RayzorTensor);
    let sin = &*(sin_ptr as *const RayzorTensor);

    // Expect at least 2 dims; treat last as head_dim, second-to-last as num_heads,
    // and any leading dim as seq_len (collapsed). Cos/sin must be 2-D
    // [max_seq_len, head_dim/2].
    if x.ndim < 2 || cos.ndim != 2 || sin.ndim != 2 {
        return 0;
    }
    let x_shape = std::slice::from_raw_parts(x.shape, x.ndim);
    let head_dim = x_shape[x.ndim - 1];
    if !head_dim.is_multiple_of(2) {
        return 0;
    }
    let half = head_dim / 2;
    let num_heads = if x.ndim >= 3 { x_shape[x.ndim - 2] } else { 1 };
    let seq_len: usize = x_shape[..x.ndim.saturating_sub(2)].iter().product::<usize>().max(1)
        * (if x.ndim >= 3 { 1 } else { x_shape[0] });
    let cos_shape = std::slice::from_raw_parts(cos.shape, 2);
    let sin_shape = std::slice::from_raw_parts(sin.shape, 2);
    if cos_shape[1] != half || sin_shape[1] != half {
        return 0;
    }
    let cos_max = cos_shape[0];
    let pos_off = position_offset.max(0) as usize;

    let result = alloc_tensor(x_shape, x.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);

    let elements_per_head = head_dim;
    let elements_per_row = num_heads * elements_per_head;

    for s in 0..seq_len {
        let pos = s + pos_off;
        if pos >= cos_max {
            // Position out of range — fall back to identity rotation by copying x.
            for i in 0..elements_per_row {
                let off = s * elements_per_row + i;
                let v = load_f32_at(x.data, off, x.dtype);
                store_f32_at(r.data, off, x.dtype, v);
            }
            continue;
        }
        for h in 0..num_heads {
            for i in 0..half {
                let cos_v = load_f32_at(cos.data, pos * half + i, cos.dtype);
                let sin_v = load_f32_at(sin.data, pos * half + i, sin.dtype);
                let base = s * elements_per_row + h * elements_per_head;
                let off_lo = base + 2 * i;
                let off_hi = base + 2 * i + 1;
                let xlo = load_f32_at(x.data, off_lo, x.dtype);
                let xhi = load_f32_at(x.data, off_hi, x.dtype);
                store_f32_at(r.data, off_lo, x.dtype, xlo * cos_v - xhi * sin_v);
                store_f32_at(r.data, off_hi, x.dtype, xlo * sin_v + xhi * cos_v);
            }
        }
    }
    result
}

/// Generate the RoPE cos/sin tables for a given head dimension and maximum
/// sequence length. Returns the cos table; the sin table is generated by
/// the sibling `rayzor_tensor_rope_sin_table`. Both share the same shape
/// `[max_seq_len, head_dim / 2]` and dtype F32.
///
/// The frequency schedule is the Llama / GPT-NeoX standard:
/// ```text
///   theta_i = 1.0 / (base ^ (2i / head_dim))   for i in 0..head_dim/2
///   cos[p, i] = cos(p * theta_i)
///   sin[p, i] = sin(p * theta_i)
/// ```
/// `base` defaults to 10000.0 in most Llama checkpoints (passed as f64 from
/// Haxe). Use the same base used by the model that produced your weights.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, /* sin */ false)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, /* sin */ true)
}

/// F16-stored variants of the RoPE LUTs. Same math, half the memory.
/// Useful when the host wants to keep both tables resident in a tight
/// VRAM budget (Llama 3.2 1B with `max_seq_len=2048` and `head_dim=64`
/// is 256 KB at F32, 128 KB at F16 — the savings amortise into the
/// rest of the activation budget on WebGPU buffer caps).
///
/// Precision note: the rotation step inside the kernel converts each
/// LUT element back to f32 before the multiply, so the precision loss
/// is bounded by the F16 quantisation of `cos / sin ∈ [-1, 1]` —
/// roughly 5e-4 absolute. Indistinguishable in practice for inference.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table_f16(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, /* sin */ false)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table_f16(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
) -> i64 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, /* sin */ true)
}

unsafe fn rope_table(
    head_dim: i64,
    max_seq_len: i64,
    base: f64,
    dtype: u8,
    want_sin: bool,
) -> i64 {
    if head_dim <= 0 || max_seq_len <= 0 || head_dim % 2 != 0 {
        return 0;
    }
    let head_dim = head_dim as usize;
    let max_seq_len = max_seq_len as usize;
    let half = head_dim / 2;
    let shape = [max_seq_len, half];
    let result = alloc_tensor(&shape, dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let head_dim_f = head_dim as f64;
    for p in 0..max_seq_len {
        for i in 0..half {
            let theta = 1.0_f64 / base.powf((2 * i) as f64 / head_dim_f);
            let angle = (p as f64) * theta;
            let v = if want_sin { angle.sin() } else { angle.cos() };
            store_f32_at(r.data, p * half + i, dtype, v as f32);
        }
    }
    result
}

/// Batched 3-D matmul: `a [batch, M, K]` × `b [batch, K, N]` → `[batch, M, N]`.
/// Per-batch independent matmul; reuses the existing F32 axpy fast path
/// when the inputs are contiguous F32, scalar fallback for other dtypes.
///
/// This is the core kernel that lets the Haxe layer build attention as
/// `(Q @ Kᵀ) → softmax → (· V)` without materialising a per-head loop in
/// user code.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_bmm(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.ndim != 3 || b.ndim != 3 || a.dtype != b.dtype {
        return 0;
    }
    let a_shape = std::slice::from_raw_parts(a.shape, 3);
    let b_shape = std::slice::from_raw_parts(b.shape, 3);
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];
    if b_shape[0] != batch || b_shape[1] != k {
        return 0;
    }
    let out_shape = [batch, m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let dtype = a.dtype;
    let elem = dtype_size(dtype);
    let a_step = m * k * elem;
    let b_step = k * n * elem;
    let c_step = m * n * elem;

    for batch_i in 0..batch {
        let a_data = a.data.add(batch_i * a_step);
        let b_data = b.data.add(batch_i * b_step);
        let c_data = r.data.add(batch_i * c_step);
        if dtype == DTYPE_F32 {
            let a_f = a_data as *const f32;
            let b_f = b_data as *const f32;
            let c_f = c_data as *mut f32;
            for i in 0..m {
                let a_row = a_f.add(i * k);
                let c_row = c_f.add(i * n);
                for p in 0..k {
                    let a_ik = *a_row.add(p);
                    let b_row = b_f.add(p * n);
                    let c_slice = std::slice::from_raw_parts_mut(c_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_slice(c_slice, a_ik, b_slice);
                }
            }
            continue;
        }
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let av = load_f32_at(a_data, i * k + p, dtype);
                    let bv = load_f32_at(b_data, p * n + j, dtype);
                    acc += av * bv;
                }
                store_f32_at(c_data, i * n + j, dtype, acc);
            }
        }
    }
    result
}

/// In-place causal mask. Treats the last two dimensions of `t` as
/// `[..., rows, cols]` and fills positions `(i, j)` with `-inf` whenever
/// `j > i + position_offset`. Standard pattern: a softmax row reads the
/// masked positions as zero probability.
///
/// `position_offset` shifts the diagonal — 0 for prefill (every query row
/// sees keys up to its own index), positive for decode (the single new
/// query at logical position T attends to keys 0..=T).
///
/// Returns the same tensor pointer (mutates in place) for convenience.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_causal_mask_(t_ptr: i64, position_offset: i64) -> i64 {
    if t_ptr == 0 {
        return 0;
    }
    let t = &*(t_ptr as *const RayzorTensor);
    if t.ndim < 2 {
        return 0;
    }
    let shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let cols = shape[t.ndim - 1];
    let rows = shape[t.ndim - 2];
    let outer: usize = shape[..t.ndim.saturating_sub(2)].iter().product::<usize>().max(1);
    let pos = position_offset.max(0) as usize;
    let neg_inf = f32::NEG_INFINITY;
    for o in 0..outer {
        let base = o * rows * cols;
        for i in 0..rows {
            // Mask everything strictly after the diagonal (+ position_offset).
            let first_masked = (i + pos + 1).min(cols);
            for j in first_masked..cols {
                store_f32_at(t.data, base + i * cols + j, t.dtype, neg_inf);
            }
        }
    }
    t_ptr
}

/// Scale every element by a scalar f32. Allocates a fresh tensor; no
/// in-place variant since composing with other ops works just as well
/// after the new allocation.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_scale(t_ptr: i64, factor: f64) -> i64 {
    if t_ptr == 0 {
        return 0;
    }
    let t = &*(t_ptr as *const RayzorTensor);
    let shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let result = alloc_tensor(shape, t.dtype, None);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let f = factor as f32;
    if t.dtype == DTYPE_F32 {
        let src = t.data as *const f32;
        let dst = r.data as *mut f32;
        let n = t.numel;
        let src_slice = std::slice::from_raw_parts(src, n);
        let dst_slice = std::slice::from_raw_parts_mut(dst, n);
        crate::tensor_simd::mul_const_slice(dst_slice, src_slice, f);
        return result;
    }
    for i in 0..t.numel {
        let v = load_f32_at(t.data, i, t.dtype);
        store_f32_at(r.data, i, t.dtype, v * f);
    }
    result
}

/// Transpose the last two dimensions (zero-copy view). Equivalent to
/// `tensor.permute([..., ndim-1, ndim-2])` for ndim ≥ 2. The common use is
/// turning `K [seq_k, num_kv_heads, head_dim]` into something we can
/// matmul against, but for true 3-D batched matmul on K we'd more often
/// pre-transpose at the per-head level — depends on layout choices.
/// Provided for completeness so nue doesn't need to reach for permute()
/// every time.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_transpose_last2(t_ptr: i64) -> i64 {
    if t_ptr == 0 {
        return 0;
    }
    let t = &*(t_ptr as *const RayzorTensor);
    if t.ndim < 2 {
        return t_ptr;
    }
    let n = t.ndim;
    let old_shape = std::slice::from_raw_parts(t.shape, n);
    let old_strides = std::slice::from_raw_parts(t.strides, n);

    let new_shape_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    let new_strides_ptr = malloc(n * std::mem::size_of::<usize>()) as *mut usize;
    if new_shape_ptr.is_null() || new_strides_ptr.is_null() {
        return 0;
    }
    for i in 0..n {
        *new_shape_ptr.add(i) = old_shape[i];
        *new_strides_ptr.add(i) = old_strides[i];
    }
    // Swap the last two.
    *new_shape_ptr.add(n - 1) = old_shape[n - 2];
    *new_shape_ptr.add(n - 2) = old_shape[n - 1];
    *new_strides_ptr.add(n - 1) = old_strides[n - 2];
    *new_strides_ptr.add(n - 2) = old_strides[n - 1];

    let new_t = malloc(std::mem::size_of::<RayzorTensor>()) as *mut RayzorTensor;
    if new_t.is_null() {
        free(new_shape_ptr as *mut u8);
        free(new_strides_ptr as *mut u8);
        return 0;
    }
    *new_t = RayzorTensor {
        data: t.data,
        shape: new_shape_ptr,
        strides: new_strides_ptr,
        ndim: n,
        numel: t.numel,
        dtype: t.dtype,
        owns_data: false,
        device: t.device,
        numa_node: t.numa_node,
    };
    new_t as i64
}

/// Row-gather: fetch the rows of `table` named by `indices` and stack
/// them into a new tensor.
///
/// `table` is `[N, ...rest]`; `indices` is an i64 array of length `K`.
/// The result has shape `[K, ...rest]` and shares the source dtype.
/// Used by `nue.Embedding` to turn `[seq_len]` token IDs into a
/// `[seq_len, hidden_size]` activation tensor.
///
/// Out-of-range indices return 0 — caller is responsible for validating
/// the vocabulary range. The indices array is read as i64, matching the
/// Haxe Array<Int> layout (which boxes ints to i64 in this runtime).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows(
    table_ptr: i64,
    indices_ptr: i64,
    indices_len: i64,
) -> i64 {
    if table_ptr == 0 || indices_ptr == 0 || indices_len <= 0 {
        return 0;
    }
    let table = &*(table_ptr as *const RayzorTensor);
    if table.ndim == 0 {
        return 0;
    }
    let table_shape = std::slice::from_raw_parts(table.shape, table.ndim);
    let n_rows = table_shape[0];
    let row_numel: usize = table_shape[1..].iter().product::<usize>().max(1);
    let elem = dtype_size(table.dtype);
    let row_bytes = row_numel * elem;
    let k = indices_len as usize;

    // Out shape: [K, ...table_shape[1..]]
    let mut out_shape = Vec::with_capacity(table.ndim);
    out_shape.push(k);
    for &dim in &table_shape[1..] {
        out_shape.push(dim);
    }
    let result = alloc_tensor(&out_shape, table.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const RayzorTensor);
    let indices = indices_ptr as *const i64;

    for i in 0..k {
        let idx_raw = *indices.add(i);
        if idx_raw < 0 || (idx_raw as usize) >= n_rows {
            // Out of range — leave the corresponding output row zeroed.
            continue;
        }
        let src = table.data.add((idx_raw as usize) * row_bytes);
        let dst = r.data.add(i * row_bytes);
        std::ptr::copy_nonoverlapping(src, dst, row_bytes);
    }
    result
}

// ============================================================================
// Reductions
// ============================================================================

/// tensor.sum() -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sum(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.dtype == DTYPE_F32 {
        let data = std::slice::from_raw_parts(t.data as *const f32, t.numel);
        return crate::tensor_simd::sum_slice(data) as f64;
    }
    let mut acc = 0.0f64;
    for i in 0..t.numel {
        acc += load_f32_at(t.data, i, t.dtype) as f64;
    }
    acc
}

/// tensor.max() -> f64 (returns -inf for empty tensors)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_max(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return f64::NEG_INFINITY;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.numel == 0 {
        return f64::NEG_INFINITY;
    }
    if t.dtype == DTYPE_F32 {
        let data = std::slice::from_raw_parts(t.data as *const f32, t.numel);
        return crate::tensor_simd::max_slice(data) as f64;
    }
    let mut m = f32::NEG_INFINITY;
    for i in 0..t.numel {
        let v = load_f32_at(t.data, i, t.dtype);
        if v > m {
            m = v;
        }
    }
    m as f64
}

/// tensor.min() -> f64 (returns +inf for empty tensors)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_min(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return f64::INFINITY;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.numel == 0 {
        return f64::INFINITY;
    }
    if t.dtype == DTYPE_F32 {
        let data = std::slice::from_raw_parts(t.data as *const f32, t.numel);
        return crate::tensor_simd::min_slice(data) as f64;
    }
    let mut m = f32::INFINITY;
    for i in 0..t.numel {
        let v = load_f32_at(t.data, i, t.dtype);
        if v < m {
            m = v;
        }
    }
    m as f64
}

/// tensor.mean() -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mean(tensor_ptr: i64) -> f64 {
    if tensor_ptr == 0 {
        return 0.0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    if t.numel == 0 {
        return 0.0;
    }
    rayzor_tensor_sum(tensor_ptr) / (t.numel as f64)
}

/// tensor.dot(other) -> f64
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_dot(a_ptr: i64, b_ptr: i64) -> f64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0.0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);
    if a.numel != b.numel || a.dtype != b.dtype {
        return 0.0;
    }

    if a.dtype == DTYPE_F32 {
        let a_s = std::slice::from_raw_parts(a.data as *const f32, a.numel);
        let b_s = std::slice::from_raw_parts(b.data as *const f32, b.numel);
        return crate::tensor_simd::dot_slice(a_s, b_s) as f64;
    }
    let mut acc = 0.0f64;
    for i in 0..a.numel {
        let av = load_f32_at(a.data, i, a.dtype) as f64;
        let bv = load_f32_at(b.data, i, a.dtype) as f64;
        acc += av * bv;
    }
    acc
}

// ============================================================================
// Matrix multiplication
// ============================================================================

/// tensor.matmul(other) -> i64
/// Naive O(n³) matmul for [M,K] × [K,N] -> [M,N]
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul(a_ptr: i64, b_ptr: i64) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    if a.ndim != 2 || b.ndim != 2 || a.dtype != b.dtype {
        return 0;
    }

    let a_shape = std::slice::from_raw_parts(a.shape, 2);
    let b_shape = std::slice::from_raw_parts(b.shape, 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    if k != b_shape[0] {
        return 0;
    } // dimension mismatch

    let out_shape = [m, n];
    let result = alloc_tensor(&out_shape, a.dtype, Some(0.0));
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_strides = std::slice::from_raw_parts(a.strides, 2);
    let b_strides = std::slice::from_raw_parts(b.strides, 2);

    if a.dtype == DTYPE_F32 {
        let a_data = a.data as *const f32;
        let b_data = b.data as *const f32;
        let r_data = r.data as *mut f32;

        // Fast path: both A and B row-major (innermost stride == 1). Loop
        // order is (i, k, j) so the inner `j` loop accumulates into a
        // contiguous row of R with broadcast a_ik — the textbook
        // SIMD-friendly matmul.
        if a_strides[1] == 1 && b_strides[1] == 1 {
            let r_row_size = n;
            for i in 0..m {
                let a_row = a_data.add(i * a_strides[0]);
                let r_row = r_data.add(i * r_row_size);
                for p in 0..k {
                    let a_ik = *a_row.add(p);
                    let b_row = b_data.add(p * b_strides[0]);
                    let r_slice = std::slice::from_raw_parts_mut(r_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_slice(r_slice, a_ik, b_slice);
                }
            }
            return result;
        }

        // Strided fallback (e.g. transposed views).
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_val = *a_data.add(i * a_strides[0] + p * a_strides[1]);
                    let b_val = *b_data.add(p * b_strides[0] + j * b_strides[1]);
                    sum += a_val * b_val;
                }
                *r_data.add(i * n + j) = sum;
            }
        }
        return result;
    }

    // F16 / BF16 row-major fast path: axpy specialisation that stages
    // through f32 via NEON vcvt_f32_f16 / F16C _mm_cvtph_ps in batches of 64.
    // This is the matmul hot path for LLM inference (every row update is an
    // f16 axpy against a half-precision weight row).
    if a_strides[1] == 1 && b_strides[1] == 1 {
        if a.dtype == DTYPE_F16 {
            let a_data = a.data as *const u16;
            let b_data = b.data as *const u16;
            let r_data = r.data as *mut u16;
            for i in 0..m {
                let a_row = a_data.add(i * a_strides[0]);
                let r_row = r_data.add(i * n);
                for p in 0..k {
                    let a_ik = half::f16::from_bits(*a_row.add(p)).to_f32();
                    let b_row = b_data.add(p * b_strides[0]);
                    let r_slice = std::slice::from_raw_parts_mut(r_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_f16_slice(r_slice, a_ik, b_slice);
                }
            }
            return result;
        }
        if a.dtype == DTYPE_BF16 {
            let a_data = a.data as *const u16;
            let b_data = b.data as *const u16;
            let r_data = r.data as *mut u16;
            for i in 0..m {
                let a_row = a_data.add(i * a_strides[0]);
                let r_row = r_data.add(i * n);
                for p in 0..k {
                    let a_ik = half::bf16::from_bits(*a_row.add(p)).to_f32();
                    let b_row = b_data.add(p * b_strides[0]);
                    let r_slice = std::slice::from_raw_parts_mut(r_row, n);
                    let b_slice = std::slice::from_raw_parts(b_row, n);
                    crate::tensor_simd::axpy_bf16_slice(r_slice, a_ik, b_slice);
                }
            }
            return result;
        }
    }

    // Generic dtype path: convert each element to f32 in-register, accumulate
    // in f32, store back as the source dtype. Covers strided F16/BF16 views
    // and the integer/FP8 dtypes.
    let dtype = a.dtype;
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_off = i * a_strides[0] + p * a_strides[1];
                let b_off = p * b_strides[0] + j * b_strides[1];
                let a_val = load_f32_at(a.data, a_off, dtype);
                let b_val = load_f32_at(b.data, b_off, dtype);
                sum += a_val * b_val;
            }
            store_f32_at(r.data, i * n + j, dtype, sum);
        }
    }

    result
}

// ============================================================================
// Interop
// ============================================================================

/// tensor.data() -> i64 (raw pointer to data buffer)
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_data(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);
    t.data as i64
}

/// tensor.free() -> void
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_free(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let t = &*(tensor_ptr as *const RayzorTensor);

    if t.owns_data && !t.data.is_null() {
        free(t.data);
    }
    if !t.shape.is_null() {
        free(t.shape as *mut u8);
    }
    if !t.strides.is_null() {
        free(t.strides as *mut u8);
    }
    free(tensor_ptr as *mut u8);
}

// ============================================================================
// Helpers
// ============================================================================

/// Read shape from a Haxe Array<Int> data pointer.
/// The pointer points to the raw i64 data of the array.
unsafe fn read_shape(ptr: i64, ndim: usize) -> Vec<usize> {
    if ptr == 0 || ndim == 0 {
        return vec![];
    }
    let data = ptr as *const i64;
    (0..ndim).map(|i| *data.add(i) as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        unsafe {
            let shape = [2usize, 3usize];
            let t = alloc_tensor(&shape, DTYPE_F32, Some(0.0));
            assert!(t != 0);
            let tensor = &*(t as *const RayzorTensor);
            assert_eq!(tensor.ndim, 2);
            assert_eq!(tensor.numel, 6);
            assert_eq!(tensor.dtype, DTYPE_F32);

            // All zeros
            let data = tensor.data as *const f32;
            for i in 0..6 {
                assert_eq!(*data.add(i), 0.0);
            }

            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn test_tensor_ones_and_sum() {
        unsafe {
            let shape = [2usize, 3usize];
            let t = alloc_tensor(&shape, DTYPE_F32, Some(1.0));
            assert!(t != 0);
            let sum = rayzor_tensor_sum(t);
            assert!((sum - 6.0).abs() < 1e-6);
            rayzor_tensor_free(t);
        }
    }

    #[test]
    fn test_tensor_add() {
        unsafe {
            let shape = [3usize];
            let a = alloc_tensor(&shape, DTYPE_F32, Some(2.0));
            let b = alloc_tensor(&shape, DTYPE_F32, Some(3.0));
            let c = rayzor_tensor_add(a, b);
            assert!(c != 0);
            let sum = rayzor_tensor_sum(c);
            assert!((sum - 15.0).abs() < 1e-6); // (2+3)*3 = 15
            rayzor_tensor_free(a);
            rayzor_tensor_free(b);
            rayzor_tensor_free(c);
        }
    }

    #[test]
    fn test_tensor_matmul() {
        unsafe {
            // [2,2] identity matmul [2,2] ones = [2,2] with row sums = 2
            let ident = alloc_tensor(&[2, 2], DTYPE_F32, Some(0.0));
            let ones = alloc_tensor(&[2, 2], DTYPE_F32, Some(1.0));

            // Set identity
            let id = &*(ident as *const RayzorTensor);
            let d = id.data as *mut f32;
            *d.add(0) = 1.0; // [0,0]
            *d.add(3) = 1.0; // [1,1]

            let result = rayzor_tensor_matmul(ident, ones);
            assert!(result != 0);
            let sum = rayzor_tensor_sum(result);
            assert!((sum - 4.0).abs() < 1e-6); // I * ones = ones, sum = 4

            rayzor_tensor_free(ident);
            rayzor_tensor_free(ones);
            rayzor_tensor_free(result);
        }
    }
}
