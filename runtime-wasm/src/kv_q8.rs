//! In-guest Q8_0 KV cache — the `nue.transformer.KvCacheQ8` capability,
//! ported from `nue-plugins` so the wasm target gets Q8 KV WITHOUT the
//! (blocked) dylink.0 side-module loader. Storage per (row, kv_head) is
//! `head_dim/32` Q8_0 blocks of 34 bytes (2-byte f16 scale + 32 i8 quants),
//! ~3.76x smaller than F32 — lowers the load peak (2GB-ceiling pressure) and
//! the fused dequant flash decode reads less memory.
//!
//! The guest's `KvCacheQ8` extern methods bind to the same symbol names the
//! native plugin exports; with these compiled into runtime-wasm the linker
//! resolves them in-guest (no import, no side-module).
//!
//! Kernels are scalar (correct + matches the plugin's non-NEON path); a
//! wasm-simd128 pass can follow the same template as `tensor_f32`.

use crate::tensor::{alloc_tensor, Tensor, DTYPE_F32};
use core::slice;
use half::f16;

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_SIZE: usize = 32;

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

#[repr(C)]
struct RayzorKvCacheQ8 {
    data: *mut u8,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    head_dim_bytes: usize,
}

#[inline]
unsafe fn quantize_q8_0_block(src: *const f32, dst: *mut u8) {
    let mut max_abs = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE {
        let v = libm::fabsf(*src.add(i));
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
        let r = libm::roundf((*src.add(i)) * inv_scale).clamp(-128.0, 127.0);
        *q_ptr.add(i) = r as i8;
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

/// Allocate a fresh Q8_0 KV cache. Returns 0 on bad args / malloc failure.
#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_alloc(
    max_seq_len: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> i32 {
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
    handle as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_free(handle: i32) {
    if handle == 0 {
        return;
    }
    let h = handle as *mut RayzorKvCacheQ8;
    if !(*h).data.is_null() {
        free((*h).data);
    }
    free(h as *mut u8);
}

/// Quantise + append `src_tensor` (F32 `[n, num_kv_heads, head_dim]`) at
/// `current_len`. Returns the new write cursor or -1 on mismatch/overflow.
#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_append(
    handle: i32,
    current_len: i32,
    src_tensor: i32,
) -> i32 {
    if handle == 0 || src_tensor == 0 {
        return -1;
    }
    let h = &*(handle as *const RayzorKvCacheQ8);
    let t = &*(src_tensor as *const Tensor);
    if t.dtype != DTYPE_F32 || !t.is_contiguous() || t.ndim != 3 {
        return -1;
    }
    let t_shape = slice::from_raw_parts(t.shape, 3);
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
    (current_len + n_new) as i32
}

/// Dequantise positions `[0..current_len)` into a fresh owning F32 tensor.
#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_cache_q8_dequant_view(handle: i32, current_len: i32) -> i32 {
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
    let out = alloc_tensor(&[current_len, h.num_kv_heads, h.head_dim], DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let out_data = (*out).data as *mut f32;
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
    out as i32
}

/// Fused single-pass flash attention over a Q8 KV cache (GQA-aware: dequant
/// each K/V block once, feed all `group` query heads). `k_handle` is self.
/// Returns a fresh F32 `[1, num_q_heads, head_dim]` tensor (0 on bad args).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode_q8(
    k_handle: i32,
    q_ptr: i32,
    v_handle: i32,
    cache_len: i32,
    num_q_heads: i32,
    scale: f64,
) -> i32 {
    if q_ptr == 0 || k_handle == 0 || v_handle == 0 || cache_len < 0 {
        return 0;
    }
    let q = &*(q_ptr as *const Tensor);
    if q.dtype != DTYPE_F32 || !q.is_contiguous() || q.ndim != 3 {
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
    let q_shape = slice::from_raw_parts(q.shape, 3);
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

    let out = alloc_tensor(&[1, num_q_heads, head_dim], DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let q_data = q.data as *const f32;
    let out_data = (*out).data as *mut f32;
    let k_base = k_cache.data;
    let v_base = v_cache.data;
    let scale_f32 = scale as f32;

    // GQA-aware: kv_head OUTER so each K/V block is dequantised once and feeds
    // all `group` query heads. `group_scores` holds pre-softmax scores then
    // normalised weights; reduction order is block-by-block (matches the
    // plugin / Paris MATCH).
    let mut group_scores: std::vec::Vec<f32> = std::vec![0.0; group * cache_len];
    let mut group_max: std::vec::Vec<f32> = std::vec![f32::NEG_INFINITY; group];
    let mut k_block = [0.0f32; Q8_0_BLOCK_SIZE];
    let mut v_block = [0.0f32; Q8_0_BLOCK_SIZE];

    for kv_head in 0..num_kv_heads {
        let group_start = kv_head * group;
        for s in &mut group_scores[..group * cache_len] {
            *s = 0.0;
        }
        for m in &mut group_max[..group] {
            *m = f32::NEG_INFINITY;
        }

        // K step.
        for l in 0..cache_len {
            let k_row_ptr = k_base.add(l * row_bytes + kv_head * head_dim_bytes);
            for b in 0..blocks_per_head {
                dequant_q8_0_block(k_row_ptr.add(b * Q8_0_BLOCK_BYTES), &mut k_block);
                for gi in 0..group {
                    let q_head = group_start + gi;
                    let q_row_ptr = q_data.add(q_head * head_dim);
                    let partial = dot_block_f32(q_row_ptr.add(b * Q8_0_BLOCK_SIZE), &k_block);
                    group_scores[gi * cache_len + l] += partial;
                }
            }
            for gi in 0..group {
                let s = group_scores[gi * cache_len + l] * scale_f32;
                group_scores[gi * cache_len + l] = s;
                if s > group_max[gi] {
                    group_max[gi] = s;
                }
            }
        }

        // Softmax in-place → normalised weights.
        for gi in 0..group {
            let max_g = group_max[gi];
            let row = &mut group_scores[gi * cache_len..(gi + 1) * cache_len];
            let mut denom = 0.0f32;
            for s in row.iter_mut() {
                let e = libm::expf(*s - max_g);
                *s = e;
                denom += e;
            }
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for s in row.iter_mut() {
                *s *= inv_denom;
            }
        }

        // Zero output rows for this group, then V step (axpy).
        for gi in 0..group {
            let out_row_ptr = out_data.add((group_start + gi) * head_dim);
            for d in 0..head_dim {
                *out_row_ptr.add(d) = 0.0;
            }
        }
        for l in 0..cache_len {
            let v_row_ptr = v_base.add(l * row_bytes + kv_head * head_dim_bytes);
            for b in 0..blocks_per_head {
                dequant_q8_0_block(v_row_ptr.add(b * Q8_0_BLOCK_BYTES), &mut v_block);
                for gi in 0..group {
                    let q_head = group_start + gi;
                    let w = group_scores[gi * cache_len + l];
                    let out_row_ptr = out_data.add(q_head * head_dim);
                    axpy_block_f32(out_row_ptr.add(b * Q8_0_BLOCK_SIZE), w, &v_block);
                }
            }
        }
    }

    out as i32
}

// Convention clone shims (the @:shared KvCacheQ8). Cache tracks no refcount,
// so arc_clone aliases; clone deep-copies the bytes.
#[no_mangle]
pub unsafe extern "C" fn rayzor_kvcacheq8_arc_clone(handle: i32) -> i32 {
    handle
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_kvcacheq8_clone(handle: i32) -> i32 {
    if handle == 0 {
        return 0;
    }
    let src = &*(handle as *const RayzorKvCacheQ8);
    let nh = rayzor_kv_cache_q8_alloc(
        src.max_seq_len as i32,
        src.num_kv_heads as i32,
        src.head_dim as i32,
    );
    if nh != 0 {
        let total = src.max_seq_len * src.num_kv_heads * src.head_dim_bytes;
        core::ptr::copy_nonoverlapping(src.data, (*(nh as *mut RayzorKvCacheQ8)).data, total);
    }
    nh
}
