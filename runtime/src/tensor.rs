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

// DType tags matching the Haxe enum order
const DTYPE_F32: u8 = 0;
const DTYPE_F16: u8 = 1;
const DTYPE_BF16: u8 = 2;
const DTYPE_I32: u8 = 3;
const DTYPE_I8: u8 = 4;
const DTYPE_U8: u8 = 5;

fn dtype_size(dtype: u8) -> usize {
    match dtype {
        DTYPE_F32 => 4,
        DTYPE_F16 => 2,
        DTYPE_BF16 => 2,
        DTYPE_I32 => 4,
        DTYPE_I8 => 1,
        DTYPE_U8 => 1,
        _ => 4, // default to f32
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
        if dtype == DTYPE_F32 {
            let f32_ptr = data as *mut f32;
            for i in 0..numel {
                *f32_ptr.add(i) = val;
            }
        } else {
            // Zero-fill for other dtypes when fill is 0
            std::ptr::write_bytes(data, 0, data_bytes);
        }
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

    // Copy f64 data from Haxe Array<Float>, converting to target dtype
    let src = data_ptr as *const f64;
    if dtype_u8 == DTYPE_F32 {
        let dst = tensor.data as *mut f32;
        for i in 0..numel {
            *dst.add(i) = *src.add(i) as f32;
        }
    } else if dtype_u8 == DTYPE_I32 {
        let dst = tensor.data as *mut i32;
        for i in 0..numel {
            *dst.add(i) = *src.add(i) as i32;
        }
    } else {
        // Fallback: copy as f32 (most common GPU dtype)
        let dst = tensor.data as *mut f32;
        for i in 0..numel {
            *dst.add(i) = *src.add(i) as f32;
        }
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

    if t.dtype == DTYPE_F32 {
        let val = *(t.data as *const f32).add(off);
        val as f64
    } else {
        0.0
    }
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

    if t.dtype == DTYPE_F32 {
        *(t.data as *mut f32).add(off) = value as f32;
    }
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
    };

    new_t as i64
}

// ============================================================================
// Elementwise arithmetic
// ============================================================================

/// Binary elementwise op on two f32 tensors
unsafe fn tensor_binop(a_ptr: i64, b_ptr: i64, op: fn(f32, f32) -> f32) -> i64 {
    if a_ptr == 0 || b_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    let b = &*(b_ptr as *const RayzorTensor);

    // Shapes must match (no broadcasting yet)
    if a.numel != b.numel || a.dtype != DTYPE_F32 || b.dtype != DTYPE_F32 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let b_data = b.data as *const f32;
    let r_data = r.data as *mut f32;

    for i in 0..a.numel {
        *r_data.add(i) = op(*a_data.add(i), *b_data.add(i));
    }

    result
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_add(a: i64, b: i64) -> i64 {
    tensor_binop(a, b, |x, y| x + y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_sub(a: i64, b: i64) -> i64 {
    tensor_binop(a, b, |x, y| x - y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_mul(a: i64, b: i64) -> i64 {
    tensor_binop(a, b, |x, y| x * y)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_div(a: i64, b: i64) -> i64 {
    tensor_binop(a, b, |x, y| if y != 0.0 { x / y } else { 0.0 })
}

// ============================================================================
// Unary math ops
// ============================================================================

unsafe fn tensor_unary(a_ptr: i64, op: fn(f32) -> f32) -> i64 {
    if a_ptr == 0 {
        return 0;
    }
    let a = &*(a_ptr as *const RayzorTensor);
    if a.dtype != DTYPE_F32 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let r_data = r.data as *mut f32;

    for i in 0..a.numel {
        *r_data.add(i) = op(*a_data.add(i));
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
pub unsafe extern "C" fn rayzor_tensor_relu(a: i64) -> i64 {
    tensor_unary(a, |x| if x > 0.0 { x } else { 0.0 })
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
    if a.dtype != DTYPE_F32 || a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let r_data = r.data as *mut f32;
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);

    for g in 0..groups {
        let base = g * last;
        // max for numeric stability
        let mut maxv = f32::NEG_INFINITY;
        for i in 0..last {
            let v = *a_data.add(base + i);
            if v > maxv {
                maxv = v;
            }
        }
        let mut sum = 0.0f32;
        for i in 0..last {
            let e = (*a_data.add(base + i) - maxv).exp();
            *r_data.add(base + i) = e;
            sum += e;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for i in 0..last {
                *r_data.add(base + i) *= inv;
            }
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
    if a.dtype != DTYPE_F32 || a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let r_data = r.data as *mut f32;
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps = eps as f32;
    let n = last as f32;

    for g in 0..groups {
        let base = g * last;
        let mut mean = 0.0f32;
        for i in 0..last {
            mean += *a_data.add(base + i);
        }
        mean /= n;
        let mut var = 0.0f32;
        for i in 0..last {
            let d = *a_data.add(base + i) - mean;
            var += d * d;
        }
        var /= n;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..last {
            *r_data.add(base + i) = (*a_data.add(base + i) - mean) * inv;
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
    if a.dtype != DTYPE_F32 || a.ndim == 0 {
        return 0;
    }

    let shape = std::slice::from_raw_parts(a.shape, a.ndim);
    let result = alloc_tensor(shape, DTYPE_F32, None);
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let r_data = r.data as *mut f32;
    let last = shape[a.ndim - 1];
    let groups = a.numel.checked_div(last).unwrap_or(0);
    let eps = eps as f32;
    let n = last as f32;

    for g in 0..groups {
        let base = g * last;
        let mut ms = 0.0f32;
        for i in 0..last {
            let v = *a_data.add(base + i);
            ms += v * v;
        }
        ms /= n;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..last {
            *r_data.add(base + i) = *a_data.add(base + i) * inv;
        }
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
    if t.dtype != DTYPE_F32 {
        return 0.0;
    }

    let data = t.data as *const f32;
    let mut acc = 0.0f64;
    for i in 0..t.numel {
        acc += *data.add(i) as f64;
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
    if t.dtype != DTYPE_F32 || t.numel == 0 {
        return f64::NEG_INFINITY;
    }
    let data = t.data as *const f32;
    let mut m = *data;
    for i in 1..t.numel {
        let v = *data.add(i);
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
    if t.dtype != DTYPE_F32 || t.numel == 0 {
        return f64::INFINITY;
    }
    let data = t.data as *const f32;
    let mut m = *data;
    for i in 1..t.numel {
        let v = *data.add(i);
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
    if a.numel != b.numel || a.dtype != DTYPE_F32 || b.dtype != DTYPE_F32 {
        return 0.0;
    }

    let a_data = a.data as *const f32;
    let b_data = b.data as *const f32;
    let mut acc = 0.0f64;
    for i in 0..a.numel {
        acc += (*a_data.add(i) as f64) * (*b_data.add(i) as f64);
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

    if a.ndim != 2 || b.ndim != 2 || a.dtype != DTYPE_F32 || b.dtype != DTYPE_F32 {
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
    let result = alloc_tensor(&out_shape, DTYPE_F32, Some(0.0));
    if result == 0 {
        return 0;
    }

    let r = &*(result as *const RayzorTensor);
    let a_data = a.data as *const f32;
    let b_data = b.data as *const f32;
    let r_data = r.data as *mut f32;

    // Naive matmul with stride awareness
    let a_strides = std::slice::from_raw_parts(a.strides, 2);
    let b_strides = std::slice::from_raw_parts(b.strides, 2);

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
