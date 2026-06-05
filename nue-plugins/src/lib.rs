//! nue compiler plugin (cdylib).
//!
//! Domain-specific extern classes for the nue ML framework — currently
//! a Q8_0 KV cache (`rayzor.ds.KvCacheQ8`). Ships as a cdylib loaded by
//! the rayzor compiler via the `[build] native-libs = [...]` manifest
//! field, the same mechanism rayzor-gpu uses.
//!
//! The cdylib exports:
//! - `plugin_describe()` — returns the `NativeMethodDesc` table for
//!   declared extern methods. The compiler reads this at load time and
//!   auto-registers method mappings + extern declarations.
//! - The runtime symbols listed in those descriptors as
//!   `#[no_mangle] extern "C"` functions.
//!
//! All quant kernels live here (not in rayzor-runtime) to keep the
//! runtime crate domain-agnostic.

#![allow(clippy::missing_safety_doc)]

use half::f16;
use rayzor_plugin::{declare_native_methods, NativeMethodDesc};

// ============================================================================
// Method descriptor table — read by `plugin_describe()`. The compiler
// auto-registers method mappings + extern declarations from this list,
// no compiler core changes required.
// ============================================================================

declare_native_methods! {
    NUE_METHODS;
    // KvCacheQ8 lifecycle + per-step ops.
    "rayzor_ds_KvCacheQ8", "alloc",            static,   "rayzor_kv_cache_q8_alloc",
        [I64, I64, I64]                                                       => Ptr;
    "rayzor_ds_KvCacheQ8", "free",             instance, "rayzor_kv_cache_q8_free",
        [Ptr]                                                                 => Void;
    "rayzor_ds_KvCacheQ8", "append",           instance, "rayzor_kv_cache_q8_append",
        [Ptr, I64, Ptr]                                                       => I64;
    "rayzor_ds_KvCacheQ8", "dequantView",      instance, "rayzor_kv_cache_q8_dequant_view",
        [Ptr, I64]                                                            => Ptr;
    "rayzor_ds_KvCacheQ8", "flashAttnDecodeQ8", instance, "rayzor_tensor_flash_attn_decode_q8",
        [Ptr, Ptr, Ptr, I64, I64, F64]                                        => Ptr;
}

#[no_mangle]
pub unsafe extern "C" fn plugin_describe(out_count: *mut usize) -> *const NativeMethodDesc {
    if !out_count.is_null() {
        *out_count = NUE_METHODS.len();
    }
    NUE_METHODS.as_ptr()
}

// ============================================================================
// Runtime symbol table — the JIT links against this. Same shape rayzor-gpu
// uses (returns a Vec of (name_ptr, name_len, fn_ptr) entries).
// ============================================================================

#[repr(C)]
pub struct SymbolEntry {
    pub name_ptr: *const u8,
    pub name_len: usize,
    pub fn_ptr: *const std::ffi::c_void,
}

macro_rules! entry {
    ($name:expr, $fn:ident) => {
        SymbolEntry {
            name_ptr: ($name as &[u8]).as_ptr(),
            name_len: ($name as &[u8]).len(),
            fn_ptr: $fn as *const std::ffi::c_void,
        }
    };
}

#[no_mangle]
pub unsafe extern "C" fn plugin_init(out_count: *mut usize) -> *const SymbolEntry {
    let entries = Box::new([
        entry!(b"rayzor_kv_cache_q8_alloc", rayzor_kv_cache_q8_alloc),
        entry!(b"rayzor_kv_cache_q8_free", rayzor_kv_cache_q8_free),
        entry!(b"rayzor_kv_cache_q8_append", rayzor_kv_cache_q8_append),
        entry!(b"rayzor_kv_cache_q8_dequant_view", rayzor_kv_cache_q8_dequant_view),
        entry!(b"rayzor_tensor_flash_attn_decode_q8", rayzor_tensor_flash_attn_decode_q8),
        entry!(b"rayzor_kvcacheq8_clone", rayzor_kvcacheq8_clone),
        entry!(b"rayzor_kvcacheq8_arc_clone", rayzor_kvcacheq8_arc_clone),
    ]);
    let count = entries.len();
    let ptr = Box::leak(entries).as_ptr();
    if !out_count.is_null() {
        *out_count = count;
    }
    ptr
}

// ============================================================================
// Runtime kernels — Q8_0 KV cache alloc/append/dequant + fused flash attn.
//
// Storage per (token_row, kv_head): `head_dim / 32` Q8_0 blocks of 34 bytes
// each (2-byte f16 scale + 32 i8 quants). ~3.76× smaller than F32.
// ============================================================================

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_SIZE: usize = 32;

// Mirrors `runtime/src/tensor.rs::DTYPE_F32`. F32 is 0, NOT 1 — the
// initial cdylib port had this wrong and silently bailed in every
// append/flash gate.
const DTYPE_F32: u8 = 0;

#[repr(C)]
struct RayzorKvCacheQ8 {
    data: *mut u8,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    head_dim_bytes: usize,
}

// Tensor handle layout — we only read a small fixed prefix, no need to
// pull the full struct from rayzor-runtime.
#[repr(C)]
struct TensorView {
    data: *mut u8,
    shape: *mut usize,
    strides: *mut usize,
    ndim: usize,
    numel: usize,
    dtype: u8,
}

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    // From rayzor-runtime, statically linked into the host process —
    // exported via the .cargo/config.toml `-Wl,-export_dynamic` link
    // flag and resolved at dlopen time via macOS dynamic_lookup.
    // Signature matches `runtime/src/tensor.rs::rayzor_tensor_zeros`
    // exactly: shape_ptr is passed as an i64 (the function casts it
    // back to *mut usize internally).
    fn rayzor_tensor_zeros(shape_ptr: i64, ndim: i64, dtype: i64) -> i64;
}

#[inline]
unsafe fn alloc_tensor_f32(shape: &[usize]) -> i64 {
    // The runtime fn expects shape as a usize array (it reads via
    // read_shape, walking shape_ptr as `*const usize`). Pass a Vec
    // and forget it — the runtime copies into a fresh malloc'd buffer
    // before returning, so leaking the input doesn't matter beyond
    // the call.
    let mut shape_usize: Vec<usize> = shape.to_vec();
    let ptr = shape_usize.as_mut_ptr() as i64;
    let r = rayzor_tensor_zeros(ptr, shape.len() as i64, DTYPE_F32 as i64);
    drop(shape_usize);
    r
}

unsafe fn tensor_is_contiguous(t: &TensorView) -> bool {
    if t.ndim == 0 {
        return true;
    }
    let shape = std::slice::from_raw_parts(t.shape, t.ndim);
    let strides = std::slice::from_raw_parts(t.strides, t.ndim);
    let mut expected: usize = 1;
    for i in (0..t.ndim).rev() {
        if shape[i] != 1 && strides[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}

#[inline]
unsafe fn quantize_q8_0_block(src: *const f32, dst: *mut u8) {
    let mut max_abs = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE {
        let v = (*src.add(i)).abs();
        if v > max_abs {
            max_abs = v;
        }
    }
    let scale = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
    let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

    let scale_bits = f16::from_f32(scale).to_bits();
    std::ptr::write_unaligned(dst as *mut u16, scale_bits);

    let q_ptr = dst.add(2) as *mut i8;
    for i in 0..Q8_0_BLOCK_SIZE {
        let q = ((*src.add(i)) * inv_scale).round().clamp(-128.0, 127.0) as i8;
        *q_ptr.add(i) = q;
    }
}

#[inline]
unsafe fn dequant_q8_0_block(src: *const u8, dst: &mut [f32; Q8_0_BLOCK_SIZE]) {
    let scale_bits = std::ptr::read_unaligned(src as *const u16);
    let scale = f16::from_bits(scale_bits).to_f32();
    let q_ptr = src.add(2) as *const i8;
    for i in 0..Q8_0_BLOCK_SIZE {
        dst[i] = scale * (*q_ptr.add(i) as f32);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_alloc(
    max_seq_len: i64,
    num_kv_heads: i64,
    head_dim: i64,
) -> i64 {
    if max_seq_len <= 0 || num_kv_heads <= 0 || head_dim <= 0 {
        return 0;
    }
    let max_seq_len = max_seq_len as usize;
    let num_kv_heads = num_kv_heads as usize;
    let head_dim = head_dim as usize;
    if head_dim % Q8_0_BLOCK_SIZE != 0 {
        return 0;
    }
    let head_dim_bytes = (head_dim / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
    let total = match max_seq_len
        .checked_mul(num_kv_heads)
        .and_then(|v| v.checked_mul(head_dim_bytes))
    {
        Some(t) if t > 0 => t,
        _ => return 0,
    };
    let data = malloc(total);
    if data.is_null() {
        return 0;
    }
    std::ptr::write_bytes(data, 0, total);

    let handle = malloc(std::mem::size_of::<RayzorKvCacheQ8>()) as *mut RayzorKvCacheQ8;
    if handle.is_null() {
        free(data);
        return 0;
    }
    *handle = RayzorKvCacheQ8 {
        data,
        max_seq_len,
        num_kv_heads,
        head_dim,
        head_dim_bytes,
    };
    handle as i64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_free(handle: i64) {
    if handle == 0 {
        return;
    }
    let h = handle as *mut RayzorKvCacheQ8;
    if !(*h).data.is_null() {
        free((*h).data);
    }
    free(h as *mut u8);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_append(
    handle: i64,
    current_len: i64,
    src_tensor: i64,
) -> i64 {
    if handle == 0 || src_tensor == 0 {
        return -1;
    }
    let h = &*(handle as *const RayzorKvCacheQ8);
    let t = &*(src_tensor as *const TensorView);
    if t.dtype != DTYPE_F32 || !tensor_is_contiguous(t) || t.ndim != 3 {
        return -1;
    }
    let t_shape = std::slice::from_raw_parts(t.shape, 3);
    let n_new = t_shape[0];
    if t_shape[1] != h.num_kv_heads || t_shape[2] != h.head_dim {
        return -1;
    }
    let current_len = current_len.max(0) as usize;
    if current_len + n_new > h.max_seq_len {
        return -1;
    }
    let blocks_per_head = h.head_dim / Q8_0_BLOCK_SIZE;
    let row_bytes = h.num_kv_heads * h.head_dim_bytes;
    let src_data = t.data as *const f32;
    for l in 0..n_new {
        for kvh in 0..h.num_kv_heads {
            let src_row_ptr = src_data.add((l * h.num_kv_heads + kvh) * h.head_dim);
            let dst_row_ptr = h
                .data
                .add((current_len + l) * row_bytes + kvh * h.head_dim_bytes);
            for b in 0..blocks_per_head {
                quantize_q8_0_block(
                    src_row_ptr.add(b * Q8_0_BLOCK_SIZE),
                    dst_row_ptr.add(b * Q8_0_BLOCK_BYTES),
                );
            }
        }
    }
    (current_len + n_new) as i64
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_dequant_view(handle: i64, current_len: i64) -> i64 {
    if handle == 0 || current_len <= 0 {
        return 0;
    }
    let h = &*(handle as *const RayzorKvCacheQ8);
    let current_len = current_len as usize;
    if current_len > h.max_seq_len {
        return 0;
    }
    let blocks_per_head = h.head_dim / Q8_0_BLOCK_SIZE;
    let row_bytes = h.num_kv_heads * h.head_dim_bytes;

    let result = alloc_tensor_f32(&[current_len, h.num_kv_heads, h.head_dim]);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const TensorView);
    let out_data = r.data as *mut f32;
    let mut block_buf = [0.0f32; Q8_0_BLOCK_SIZE];
    for l in 0..current_len {
        for kvh in 0..h.num_kv_heads {
            let src = h.data.add(l * row_bytes + kvh * h.head_dim_bytes);
            let dst = out_data.add((l * h.num_kv_heads + kvh) * h.head_dim);
            for b in 0..blocks_per_head {
                dequant_q8_0_block(src.add(b * Q8_0_BLOCK_BYTES), &mut block_buf);
                std::ptr::copy_nonoverlapping(
                    block_buf.as_ptr(),
                    dst.add(b * Q8_0_BLOCK_SIZE),
                    Q8_0_BLOCK_SIZE,
                );
            }
        }
    }
    result
}

/// Inner dot product for the K side: returns `Σ q[i] * k_block[i]` over
/// a 32-element Q8_0 block. Plain scalar — LLVM vectorises well; we can
/// drop the cross-platform SIMD helpers in later if the bench needs it.
#[inline]
unsafe fn dot_block_f32(q: *const f32, k: &[f32; Q8_0_BLOCK_SIZE]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE {
        s += *q.add(i) * k[i];
    }
    s
}

/// out[i] += w * v[i] over a 32-element block.
#[inline]
unsafe fn axpy_block_f32(out: *mut f32, w: f32, v: &[f32; Q8_0_BLOCK_SIZE]) {
    for i in 0..Q8_0_BLOCK_SIZE {
        *out.add(i) += w * v[i];
    }
}

/// Self-first ABI: k_handle is self (the cache that produces K-side
/// scores). Order matches the Haxe instance call
/// `kCache.flashAttnDecodeQ8(q, vCache, cacheLen, numQHeads, scale)`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode_q8(
    k_handle: i64,
    q_ptr: i64,
    v_handle: i64,
    cache_len: i64,
    num_q_heads: i64,
    scale: f64,
) -> i64 {
    if q_ptr == 0 || k_handle == 0 || v_handle == 0 || cache_len < 0 {
        return 0;
    }
    let q = &*(q_ptr as *const TensorView);
    if q.dtype != DTYPE_F32 || !tensor_is_contiguous(q) || q.ndim != 3 {
        return 0;
    }
    let k_cache = &*(k_handle as *const RayzorKvCacheQ8);
    let v_cache = &*(v_handle as *const RayzorKvCacheQ8);
    let cache_len = cache_len as usize;
    if cache_len > k_cache.max_seq_len || cache_len > v_cache.max_seq_len {
        return 0;
    }
    if k_cache.num_kv_heads != v_cache.num_kv_heads || k_cache.head_dim != v_cache.head_dim {
        return 0;
    }
    let q_shape = std::slice::from_raw_parts(q.shape, 3);
    let seq_q = q_shape[0];
    let nqh = q_shape[1];
    let hd = q_shape[2];
    if seq_q != 1 || nqh != num_q_heads as usize || hd != k_cache.head_dim {
        return 0;
    }
    let num_q_heads = nqh;
    let num_kv_heads = k_cache.num_kv_heads;
    let head_dim = hd;
    if num_kv_heads == 0 || num_q_heads % num_kv_heads != 0 {
        return 0;
    }
    let group = num_q_heads / num_kv_heads;
    let blocks_per_head = head_dim / Q8_0_BLOCK_SIZE;
    let head_dim_bytes = blocks_per_head * Q8_0_BLOCK_BYTES;
    let row_bytes = num_kv_heads * head_dim_bytes;

    let result = alloc_tensor_f32(&[1, num_q_heads, head_dim]);
    if result == 0 {
        return 0;
    }
    let r = &*(result as *const TensorView);

    let q_data = q.data as *const f32;
    let out_data = r.data as *mut f32;
    let k_base = k_cache.data;
    let v_base = v_cache.data;
    let scale_f32 = scale as f32;

    let mut scores: Vec<f32> = vec![0.0; cache_len];
    let mut k_block = [0.0f32; Q8_0_BLOCK_SIZE];
    let mut v_block = [0.0f32; Q8_0_BLOCK_SIZE];

    for q_head in 0..num_q_heads {
        let kv_head = q_head / group;
        let q_row_ptr = q_data.add(q_head * head_dim);

        // Pass 1: scores
        let mut max_score = f32::NEG_INFINITY;
        for l in 0..cache_len {
            let k_row_ptr = k_base.add(l * row_bytes + kv_head * head_dim_bytes);
            let mut s = 0.0f32;
            for b in 0..blocks_per_head {
                dequant_q8_0_block(k_row_ptr.add(b * Q8_0_BLOCK_BYTES), &mut k_block);
                s += dot_block_f32(q_row_ptr.add(b * Q8_0_BLOCK_SIZE), &k_block);
            }
            let s = s * scale_f32;
            scores[l] = s;
            if s > max_score {
                max_score = s;
            }
        }

        // Pass 2: softmax denominator
        let mut denom = 0.0f32;
        for l in 0..cache_len {
            let e = (scores[l] - max_score).exp();
            scores[l] = e;
            denom += e;
        }
        let inv_denom = 1.0 / denom;

        // Pass 3: weighted V sum
        let out_row_ptr = out_data.add(q_head * head_dim);
        for d in 0..head_dim {
            *out_row_ptr.add(d) = 0.0;
        }
        for l in 0..cache_len {
            let w = scores[l] * inv_denom;
            let v_row_ptr = v_base.add(l * row_bytes + kv_head * head_dim_bytes);
            for b in 0..blocks_per_head {
                dequant_q8_0_block(v_row_ptr.add(b * Q8_0_BLOCK_BYTES), &mut v_block);
                axpy_block_f32(out_row_ptr.add(b * Q8_0_BLOCK_SIZE), w, &v_block);
            }
        }
    }

    result
}

// Convention clone shims so `KvCacheQ8` can carry `@:derive([Clone])`
// + `@:shared` (the convention picker in hir_to_mir.rs synthesises
// these symbol names). Cache doesn't currently track a refcount, so
// arc_clone is a no-op aliasing pass; deep clone duplicates bytes.
#[no_mangle]
pub unsafe extern "C" fn rayzor_kvcacheq8_arc_clone(handle: i64) -> i64 {
    handle
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kvcacheq8_clone(handle: i64) -> i64 {
    if handle == 0 {
        return 0;
    }
    let src = &*(handle as *const RayzorKvCacheQ8);
    let new_handle = rayzor_kv_cache_q8_alloc(
        src.max_seq_len as i64,
        src.num_kv_heads as i64,
        src.head_dim as i64,
    );
    if new_handle == 0 {
        return 0;
    }
    let dst = &*(new_handle as *const RayzorKvCacheQ8);
    let total = src.max_seq_len * src.num_kv_heads * src.head_dim_bytes;
    std::ptr::copy_nonoverlapping(src.data, dst.data, total);
    new_handle
}
