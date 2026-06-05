//! nue compiler plugin (cdylib).
//!
//! Domain-specific extern classes for the nue ML framework. Currently
//! a Q8_0 KV cache (`rayzor.ds.KvCacheQ8`) — others land here as nue
//! grows. Loaded by the rayzor compiler via the `[build] native-libs`
//! manifest entry; the host process exports the `rayzor_plugin_*`
//! symbols via `-Wl,-export_dynamic` and our calls resolve against
//! them at dlopen time.
//!
//! All host interaction goes through the [`rayzor_plugin`] ABI crate.
//! This file does not declare ANY `extern "C" { ... }` block of its
//! own — every reach into the host process flows through the
//! published ABI surface, so a host-side signature drift surfaces as
//! a compile error here instead of a silent SIGSEGV at dispatch.

#![allow(clippy::missing_safety_doc)]

use half::f16;
use rayzor_plugin::{declare_native_methods, dtype, NativeMethodDesc, Tensor};

// ============================================================================
// ABI handshake — host calls this at dlopen and refuses to bind any
// symbols on mismatch. Generated via the macro so plugin code can
// never forget to export it.
// ============================================================================

rayzor_plugin::export_abi_version!();

// ============================================================================
// Method descriptor table.
// ============================================================================

declare_native_methods! {
    NUE_METHODS;
    "rayzor_ds_KvCacheQ8", "alloc",             static,   "rayzor_kv_cache_q8_alloc",
        [I64, I64, I64]                                                       => Ptr;
    "rayzor_ds_KvCacheQ8", "free",              instance, "rayzor_kv_cache_q8_free",
        [Ptr]                                                                 => Void;
    "rayzor_ds_KvCacheQ8", "append",            instance, "rayzor_kv_cache_q8_append",
        [Ptr, I64, Ptr]                                                       => I64;
    "rayzor_ds_KvCacheQ8", "dequantView",       instance, "rayzor_kv_cache_q8_dequant_view",
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
// Runtime symbol table — JIT linkage entry point.
// ============================================================================

#[repr(C)]
pub struct SymbolEntry {
    pub name_ptr: *const u8,
    pub name_len: usize,
    pub fn_ptr: *const core::ffi::c_void,
}

macro_rules! entry {
    ($name:expr, $fn:ident) => {
        SymbolEntry {
            name_ptr: ($name as &[u8]).as_ptr(),
            name_len: ($name as &[u8]).len(),
            fn_ptr: $fn as *const core::ffi::c_void,
        }
    };
}

#[no_mangle]
pub unsafe extern "C" fn plugin_init(out_count: *mut usize) -> *const SymbolEntry {
    let entries = Box::new([
        entry!(b"rayzor_kv_cache_q8_alloc", rayzor_kv_cache_q8_alloc),
        entry!(b"rayzor_kv_cache_q8_free", rayzor_kv_cache_q8_free),
        entry!(b"rayzor_kv_cache_q8_append", rayzor_kv_cache_q8_append),
        entry!(
            b"rayzor_kv_cache_q8_dequant_view",
            rayzor_kv_cache_q8_dequant_view
        ),
        entry!(
            b"rayzor_tensor_flash_attn_decode_q8",
            rayzor_tensor_flash_attn_decode_q8
        ),
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
// Q8_0 KV cache — runtime kernels.
//
// Storage per (row, kv_head): `head_dim / 32` Q8_0 blocks of 34 bytes
// (2-byte f16 scale + 32 i8 quants). ~3.76× smaller than F32.
// ============================================================================

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_SIZE: usize = 32;

#[repr(C)]
struct RayzorKvCacheQ8 {
    data: *mut u8,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    head_dim_bytes: usize,
}

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
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
    core::ptr::write_unaligned(dst as *mut u16, scale_bits);

    let q_ptr = dst.add(2) as *mut i8;
    for i in 0..Q8_0_BLOCK_SIZE {
        let q = ((*src.add(i)) * inv_scale).round().clamp(-128.0, 127.0) as i8;
        *q_ptr.add(i) = q;
    }
}

#[inline]
unsafe fn dequant_q8_0_block(src: *const u8, dst: &mut [f32; Q8_0_BLOCK_SIZE]) {
    let scale_bits = core::ptr::read_unaligned(src as *const u16);
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
    core::ptr::write_bytes(data, 0, total);

    let handle = malloc(core::mem::size_of::<RayzorKvCacheQ8>()) as *mut RayzorKvCacheQ8;
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
    let t = match Tensor::from_handle(src_tensor) {
        Some(t) => t,
        None => return -1,
    };
    if t.dtype() != dtype::F32 || !t.is_contiguous() || t.ndim() != 3 {
        return -1;
    }
    let t_shape = t.shape();
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
    let src_data = t.data_ptr() as *const f32;
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

    let out = match Tensor::alloc_zeros_f32(&[current_len, h.num_kv_heads, h.head_dim]) {
        Some(t) => t,
        None => return 0,
    };
    let out_data = out.data_ptr() as *mut f32;
    let mut block_buf = [0.0f32; Q8_0_BLOCK_SIZE];
    for l in 0..current_len {
        for kvh in 0..h.num_kv_heads {
            let src = h.data.add(l * row_bytes + kvh * h.head_dim_bytes);
            let dst = out_data.add((l * h.num_kv_heads + kvh) * h.head_dim);
            for b in 0..blocks_per_head {
                dequant_q8_0_block(src.add(b * Q8_0_BLOCK_BYTES), &mut block_buf);
                core::ptr::copy_nonoverlapping(
                    block_buf.as_ptr(),
                    dst.add(b * Q8_0_BLOCK_SIZE),
                    Q8_0_BLOCK_SIZE,
                );
            }
        }
    }
    out.handle
}

#[inline]
unsafe fn dot_block_f32(q: *const f32, k: &[f32; Q8_0_BLOCK_SIZE]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE {
        s += *q.add(i) * k[i];
    }
    s
}

#[inline]
unsafe fn axpy_block_f32(out: *mut f32, w: f32, v: &[f32; Q8_0_BLOCK_SIZE]) {
    for i in 0..Q8_0_BLOCK_SIZE {
        *out.add(i) += w * v[i];
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode_q8(
    // Self-first ABI: kCache.flashAttnDecodeQ8(q, vCache, ...).
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
    let q = match Tensor::from_handle(q_ptr) {
        Some(t) => t,
        None => return 0,
    };
    if q.dtype() != dtype::F32 || !q.is_contiguous() || q.ndim() != 3 {
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
    let q_shape = q.shape();
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

    let out = match Tensor::alloc_zeros_f32(&[1, num_q_heads, head_dim]) {
        Some(t) => t,
        None => return 0,
    };

    let q_data = q.data_ptr() as *const f32;
    let out_data = out.data_ptr() as *mut f32;
    let k_base = k_cache.data;
    let v_base = v_cache.data;
    let scale_f32 = scale as f32;

    let mut scores: Vec<f32> = vec![0.0; cache_len];
    let mut k_block = [0.0f32; Q8_0_BLOCK_SIZE];
    let mut v_block = [0.0f32; Q8_0_BLOCK_SIZE];

    for q_head in 0..num_q_heads {
        let kv_head = q_head / group;
        let q_row_ptr = q_data.add(q_head * head_dim);

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

        let mut denom = 0.0f32;
        for l in 0..cache_len {
            let e = (scores[l] - max_score).exp();
            scores[l] = e;
            denom += e;
        }
        let inv_denom = 1.0 / denom;

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

    out.handle
}

// Convention clone shims so `KvCacheQ8` can carry `@:derive([Clone])`
// + `@:shared`. The compiler's convention picker synthesises these
// symbol names (`rayzor_<lower_class>_(arc_)clone`). Cache doesn't
// currently track a refcount, so arc_clone is a no-op aliasing pass;
// deep clone duplicates bytes.
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
    core::ptr::copy_nonoverlapping(src.data, dst.data, total);
    new_handle
}
