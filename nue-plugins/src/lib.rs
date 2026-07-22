//! nue compiler plugin (cdylib).
//!
//! Domain-specific extern classes for the nue ML framework. Currently
//! a Q8_0 KV cache (`nue.transformer.KvCacheQ8`) — others land here
//! as nue grows. Loaded by the rayzor compiler via the `[build] native-libs`
//! manifest entry; the host process exports the `rayzor_plugin_*`
//! symbols via `-Wl,-export_dynamic` and our calls resolve against
//! them at dlopen time.
//!
//! All host interaction goes through the [`rayzor_plugin`] ABI crate.
//! This file does not declare ANY `extern "C" { ... }` block of its
//! own — every reach into the host process flows through the
//! published ABI surface, so a host-side signature drift surfaces as
//! a compile error here instead of a silent SIGSEGV at dispatch.

// On wasm this cdylib is a PIC side-module linked into the rayzor wasm runtime
// that resolves ONLY core+alloc (no libc/wasi) — so it must be no_std there.
// The NATIVE build (the dlopen'd `.dylib` the host loads via [build]
// native-libs) keeps std. `cfg_attr` flips no_std on for wasm32 only.
#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate alloc;
#[cfg(target_arch = "wasm32")]
use alloc::vec::Vec;

use half::f16;
use rayzor_plugin::{declare_native_methods, dtype, NativeMethodDesc, Tensor};

// In-process CoreML graph engines (macOS ANE/CPU): BERT whole-encoder + Llama
// fused prefill. Native-only (std + the CoreML ObjC shim compiled by build.rs);
// excluded from the no_std wasm side-module. These declare an `extern "C"` block
// for the in-dylib `nue_coreml_*` shim (not a host reach — the shim links into
// this same cdylib), so they live outside the "no host extern blocks" rule that
// governs the rest of this crate.
#[cfg(not(target_arch = "wasm32"))]
pub mod bert_graph;
#[cfg(not(target_arch = "wasm32"))]
pub mod prefill_graph;

// ============================================================================
// no_std wasm runtime glue: a global allocator forwarding to the host's
// malloc/free, a panic handler, and a libm-backed float shim. All cfg'd to
// wasm32 — native keeps std's allocator/panic/float methods untouched.
// ============================================================================

#[cfg(target_arch = "wasm32")]
mod wasm_rt {
    use core::alloc::{GlobalAlloc, Layout};
    use core::mem::size_of;

    // The same host malloc/free the cache storage uses (see the crate-root
    // extern block). The wasm-linker aliases `malloc`->`rayzor_malloc`, so
    // this allocator shares the merged module's single dlmalloc heap — there
    // is no separate plugin heap to collide with the runtime's.
    extern "C" {
        fn malloc(size: usize) -> *mut u8;
        fn free(ptr: *mut u8);
    }

    /// Global allocator over the host's `malloc`/`free`. dlmalloc guarantees
    /// 8-byte alignment on wasm32; larger alignments (rare — this crate only
    /// allocates `Vec<f32>` and a small boxed slice) are satisfied by
    /// over-allocating and stashing the base pointer in the word just below
    /// the returned aligned pointer.
    pub struct HostAlloc;

    const MIN_ALIGN: usize = 8;

    unsafe impl GlobalAlloc for HostAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let align = layout.align();
            if align <= MIN_ALIGN {
                malloc(layout.size())
            } else {
                let total = layout.size() + align + size_of::<usize>();
                let base = malloc(total);
                if base.is_null() {
                    return base;
                }
                let raw = base as usize;
                let aligned = (raw + size_of::<usize>() + align - 1) & !(align - 1);
                let ptr = aligned as *mut u8;
                *(ptr.sub(size_of::<usize>()) as *mut usize) = raw;
                ptr
            }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if layout.align() <= MIN_ALIGN {
                free(ptr);
            } else {
                let base = *(ptr.sub(size_of::<usize>()) as *const usize);
                free(base as *mut u8);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static GLOBAL: wasm_rt::HostAlloc = wasm_rt::HostAlloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

/// f32 transcendentals: libm on wasm (no std float methods under no_std),
/// the inherent methods on native.
mod fmath {
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn exp(x: f32) -> f32 {
        libm::expf(x)
    }
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn abs(x: f32) -> f32 {
        libm::fabsf(x)
    }
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn round(x: f32) -> f32 {
        libm::roundf(x)
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[inline(always)]
    pub fn exp(x: f32) -> f32 {
        x.exp()
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[inline(always)]
    pub fn abs(x: f32) -> f32 {
        x.abs()
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[inline(always)]
    pub fn round(x: f32) -> f32 {
        x.round()
    }
    // f32::clamp is std-only under no_std, so wasm inlines the comparison;
    // native uses the inherent method (keeps clippy's manual_clamp lint happy).
    #[cfg(target_arch = "wasm32")]
    #[inline(always)]
    pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    #[inline(always)]
    pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
        x.clamp(lo, hi)
    }
}

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
    "nue_transformer_KvCacheQ8", "alloc",             static,   "rayzor_kv_cache_q8_alloc",
        [I64, I64, I64]                                                       => Ptr;
    "nue_transformer_KvCacheQ8", "free",              instance, "rayzor_kv_cache_q8_free",
        [Ptr]                                                                 => Void;
    "nue_transformer_KvCacheQ8", "append",            instance, "rayzor_kv_cache_q8_append",
        [Ptr, I64, Ptr]                                                       => I64;
    "nue_transformer_KvCacheQ8", "dequantView",       instance, "rayzor_kv_cache_q8_dequant_view",
        [Ptr, I64]                                                            => Ptr;
    "nue_transformer_KvCacheQ8", "dataPtr",           instance, "rayzor_kv_q8_data_ptr",
        [Ptr]                                                                 => I64;
    "nue_transformer_KvCacheQ8", "rowBytes",          instance, "rayzor_kv_q8_row_bytes",
        [Ptr]                                                                 => I64;
    "nue_transformer_KvCacheQ8", "flashAttnDecodeQ8", instance, "rayzor_tensor_flash_attn_decode_q8",
        [Ptr, Ptr, Ptr, I64, I64, F64]                                        => Ptr;
    "nue_transformer_KvCacheQ8", "flashAttnDecodeQ8Host", instance, "rayzor_tensor_flash_attn_q8_host",
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
//
// Native-only: this hands the host raw fn-pointers for dlopen/JIT linkage. On
// wasm the kernels are real module exports resolved by the wasm-linker (no
// fn-ptr table), so the whole block is gated out — which also keeps `Box` and
// the fn-ptr casts out of the no_std side-module.
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
#[repr(C)]
pub struct SymbolEntry {
    pub name_ptr: *const u8,
    pub name_len: usize,
    pub fn_ptr: *const core::ffi::c_void,
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! entry {
    ($name:expr, $fn:path) => {
        SymbolEntry {
            name_ptr: ($name as &[u8]).as_ptr(),
            name_len: ($name as &[u8]).len(),
            fn_ptr: $fn as *const core::ffi::c_void,
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
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
        entry!(
            b"rayzor_tensor_flash_attn_q8_host",
            rayzor_tensor_flash_attn_q8_host
        ),
        entry!(b"rayzor_kvcacheq8_clone", rayzor_kvcacheq8_clone),
        entry!(b"rayzor_kvcacheq8_arc_clone", rayzor_kvcacheq8_arc_clone),
        // CoreML graph engines (macOS): BERT encoder + Llama fused prefill.
        entry!(
            b"nue_bert_graph_load",
            crate::bert_graph::nue_bert_graph_load
        ),
        entry!(
            b"nue_bert_graph_bucket",
            crate::bert_graph::nue_bert_graph_bucket
        ),
        entry!(
            b"nue_bert_graph_execute",
            crate::bert_graph::nue_bert_graph_execute
        ),
        entry!(
            b"nue_prefill_graph_load",
            crate::prefill_graph::nue_prefill_graph_load
        ),
        entry!(
            b"nue_prefill_graph_bucket",
            crate::prefill_graph::nue_prefill_graph_bucket
        ),
        entry!(
            b"nue_prefill_graph_execute",
            crate::prefill_graph::nue_prefill_graph_execute
        ),
        entry!(
            b"nue_prefill_graph_kv_copy",
            crate::prefill_graph::nue_prefill_graph_kv_copy
        ),
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

/// Raw base of the Q8_0 block storage (guest kernels read it directly).
#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_q8_data_ptr(handle: i64) -> i64 {
    if handle == 0 {
        return 0;
    }
    (*(handle as *const RayzorKvCacheQ8)).data as i64
}

/// Bytes per cache row (= num_kv_heads * head_dim_bytes).
#[no_mangle]
pub unsafe extern "C" fn rayzor_kv_q8_row_bytes(handle: i64) -> i64 {
    if handle == 0 {
        return 0;
    }
    let c = &*(handle as *const RayzorKvCacheQ8);
    (c.num_kv_heads * c.head_dim_bytes) as i64
}

#[inline]
unsafe fn quantize_q8_0_block(src: *const f32, dst: *mut u8) {
    let mut max_abs = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE {
        let v = fmath::abs(*src.add(i));
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
        let r = fmath::clamp(fmath::round((*src.add(i)) * inv_scale), -128.0, 127.0);
        *q_ptr.add(i) = r as i8;
    }
}

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "wasm32")))]
#[inline]
unsafe fn dequant_q8_0_block(src: *const u8, dst: &mut [f32; Q8_0_BLOCK_SIZE]) {
    let scale_bits = core::ptr::read_unaligned(src as *const u16);
    let scale = f16::from_bits(scale_bits).to_f32();
    let q_ptr = src.add(2) as *const i8;
    for i in 0..Q8_0_BLOCK_SIZE {
        dst[i] = scale * (*q_ptr.add(i) as f32);
    }
}

/// wasm SIMD128 dequant: 32 i8 → 32 f32. Mirrors the NEON path (widen
/// i8→i16→i32→f32 then multiply by the shared f16 scale) using
/// `core::arch::wasm32`; per-lane `q[i] * scale` is bit-identical to the
/// scalar loop. The side-module builds with `+simd128`, so on the in-guest
/// wasm decode path this replaces the scalar fallback that made the Q8 KV
/// attention ~18% slower than the SIMD128 F32 flash kernel.
#[cfg(target_arch = "wasm32")]
#[inline]
unsafe fn dequant_q8_0_block(src: *const u8, dst: &mut [f32; Q8_0_BLOCK_SIZE]) {
    use core::arch::wasm32::*;
    let scale_bits = core::ptr::read_unaligned(src as *const u16);
    let scale = f16::from_bits(scale_bits).to_f32();
    let scale_v = f32x4_splat(scale);
    let q_ptr = src.add(2) as *const i8;
    let dst_ptr = dst.as_mut_ptr();
    for chunk in 0..2 {
        let off = chunk * 16;
        let q16 = v128_load(q_ptr.add(off) as *const v128); // 16 i8
        let lo16 = i16x8_extend_low_i8x16(q16);
        let hi16 = i16x8_extend_high_i8x16(q16);
        let i0 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(lo16));
        let i1 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(lo16));
        let i2 = f32x4_convert_i32x4(i32x4_extend_low_i16x8(hi16));
        let i3 = f32x4_convert_i32x4(i32x4_extend_high_i16x8(hi16));
        v128_store(dst_ptr.add(off) as *mut v128, f32x4_mul(i0, scale_v));
        v128_store(dst_ptr.add(off + 4) as *mut v128, f32x4_mul(i1, scale_v));
        v128_store(dst_ptr.add(off + 8) as *mut v128, f32x4_mul(i2, scale_v));
        v128_store(dst_ptr.add(off + 12) as *mut v128, f32x4_mul(i3, scale_v));
    }
}

/// NEON dequant: 32 i8 → 32 f32, one super-block per call.
/// 16-byte i8 chunks widen i8→i16→i32→f32 then multiply by the shared
/// f16 scale. 6 vmulq + 4 vcvtq + 4 vmovl + 4 vmovl per 16-byte chunk
/// × 2 chunks = ~36 SIMD ops per block vs the 32 scalar fmuls.
/// Roughly 3-4× faster on M1 once the load issue rate stops being the
/// bottleneck.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dequant_q8_0_block(src: *const u8, dst: &mut [f32; Q8_0_BLOCK_SIZE]) {
    use std::arch::aarch64::*;
    let scale_bits = core::ptr::read_unaligned(src as *const u16);
    let scale = f16::from_bits(scale_bits).to_f32();
    let scale_v = vdupq_n_f32(scale);
    let q_ptr = src.add(2) as *const i8;
    let dst_ptr = dst.as_mut_ptr();
    for chunk in 0..2 {
        let off = chunk * 16;
        let i8x16 = vld1q_s8(q_ptr.add(off));
        let i16_lo = vmovl_s8(vget_low_s8(i8x16));
        let i16_hi = vmovl_s8(vget_high_s8(i8x16));
        let i32_0 = vmovl_s16(vget_low_s16(i16_lo));
        let i32_1 = vmovl_s16(vget_high_s16(i16_lo));
        let i32_2 = vmovl_s16(vget_low_s16(i16_hi));
        let i32_3 = vmovl_s16(vget_high_s16(i16_hi));
        let f0 = vmulq_f32(vcvtq_f32_s32(i32_0), scale_v);
        let f1 = vmulq_f32(vcvtq_f32_s32(i32_1), scale_v);
        let f2 = vmulq_f32(vcvtq_f32_s32(i32_2), scale_v);
        let f3 = vmulq_f32(vcvtq_f32_s32(i32_3), scale_v);
        vst1q_f32(dst_ptr.add(off), f0);
        vst1q_f32(dst_ptr.add(off + 4), f1);
        vst1q_f32(dst_ptr.add(off + 8), f2);
        vst1q_f32(dst_ptr.add(off + 12), f3);
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
    if !head_dim.is_multiple_of(Q8_0_BLOCK_SIZE) {
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

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "wasm32")))]
#[inline]
unsafe fn dot_block_f32(q: *const f32, k: &[f32; Q8_0_BLOCK_SIZE]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..Q8_0_BLOCK_SIZE {
        s += *q.add(i) * k[i];
    }
    s
}

/// wasm SIMD128 dot of two 32-element f32 vectors: 8 `f32x4` mul+add into
/// one accumulator, then a horizontal lane fold. Mul+add (not relaxed_madd)
/// keeps it deterministic across runtimes; the 4-lane reduction drifts a few
/// ULP from the scalar sequential sum (same trade as the NEON path), well
/// below the argmax-on-128k-vocab noise floor.
#[cfg(target_arch = "wasm32")]
#[inline]
unsafe fn dot_block_f32(q: *const f32, k: &[f32; Q8_0_BLOCK_SIZE]) -> f32 {
    use core::arch::wasm32::*;
    let k_ptr = k.as_ptr();
    let mut acc = f32x4_splat(0.0);
    for i in 0..8 {
        let off = i * 4;
        let qv = v128_load(q.add(off) as *const v128);
        let kv = v128_load(k_ptr.add(off) as *const v128);
        acc = f32x4_add(acc, f32x4_mul(qv, kv));
    }
    f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc)
}

/// NEON dot of two 32-element f32 vectors. 8 vfmaq into 4 partial
/// accumulators (one per SIMD lane) then a horizontal vaddvq fold.
/// Reduction order differs from the scalar sequential sum (the four
/// lanes accumulate in parallel) — drift bounded by a few ULP per
/// block, well below the noise floor for argmax-on-128k-vocab decode.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_block_f32(q: *const f32, k: &[f32; Q8_0_BLOCK_SIZE]) -> f32 {
    use std::arch::aarch64::*;
    let k_ptr = k.as_ptr();
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..8 {
        let off = i * 4;
        let qv = vld1q_f32(q.add(off));
        let kv = vld1q_f32(k_ptr.add(off));
        acc = vfmaq_f32(acc, qv, kv);
    }
    vaddvq_f32(acc)
}

#[cfg(all(not(target_arch = "aarch64"), not(target_arch = "wasm32")))]
#[inline]
unsafe fn axpy_block_f32(out: *mut f32, w: f32, v: &[f32; Q8_0_BLOCK_SIZE]) {
    for i in 0..Q8_0_BLOCK_SIZE {
        *out.add(i) += w * v[i];
    }
}

/// wasm SIMD128 axpy: out[i] += w * v[i] for i in 0..32. 8 `f32x4`
/// mul+add streaming load/store, `w` splat across lanes — no reduction, so
/// bit-identical to the scalar loop.
#[cfg(target_arch = "wasm32")]
#[inline]
unsafe fn axpy_block_f32(out: *mut f32, w: f32, v: &[f32; Q8_0_BLOCK_SIZE]) {
    use core::arch::wasm32::*;
    let w_v = f32x4_splat(w);
    let v_ptr = v.as_ptr();
    for i in 0..8 {
        let off = i * 4;
        let vv = v128_load(v_ptr.add(off) as *const v128);
        let ov = v128_load(out.add(off) as *const v128);
        v128_store(out.add(off) as *mut v128, f32x4_add(ov, f32x4_mul(w_v, vv)));
    }
}

/// NEON axpy: out[i] += w * v[i] for i in 0..32. 8 vfmaq with `w`
/// broadcast across all lanes. Pure streaming load/FMA/store — no
/// reduction, so result is byte-identical to the scalar version.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn axpy_block_f32(out: *mut f32, w: f32, v: &[f32; Q8_0_BLOCK_SIZE]) {
    use std::arch::aarch64::*;
    let w_v = vdupq_n_f32(w);
    let v_ptr = v.as_ptr();
    for i in 0..8 {
        let off = i * 4;
        let vv = vld1q_f32(v_ptr.add(off));
        let ov = vld1q_f32(out.add(off));
        let nv = vfmaq_f32(ov, w_v, vv);
        vst1q_f32(out.add(off), nv);
    }
}

/// Host-parallel Q8 flash attention — native binding.
///
/// On native targets there is no wasm host/guest split: the serial kernel
/// below already runs with full thread access, so returning 0 tells the Haxe
/// caller to fall through to `flashAttnDecodeQ8` and native behaviour is
/// unchanged. The wasm build instead resolves this method against the main
/// runtime module's `rayzor_tensor_flash_attn_q8_host` dispatcher, which bands
/// the attention across the embedder's otherwise-idle native worker pool.
/// Gated off wasm so the side-module never exports a shadowing copy.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_q8_host(
    _k_handle: i64,
    _q_ptr: i64,
    _v_handle: i64,
    _cache_len: i64,
    _num_q_heads: i64,
    _scale: f64,
) -> i64 {
    0
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

    // GQA-aware layout: iterate kv_head OUTER so the dequant of each
    // K/V block fires once per (kv_head, cache_pos, block) and feeds
    // ALL `group` query heads in the GQA group. The original
    // q_head-outer layout re-dequanted the same block once per
    // query head → 4× redundant dequant work for Llama-3 GQA where
    // group = num_q_heads / num_kv_heads = 4.
    //
    // Per-group scratch sized once; `group * cache_len` floats holds
    // (a) the pre-softmax scores, then (b) the normalised weights
    // after softmax. For Llama-3.2-1B (group=4, cache_len=2048)
    // that's 32 KB — well within L1.
    let mut group_scores: Vec<f32> = vec![0.0; group * cache_len];
    let mut group_max: Vec<f32> = vec![f32::NEG_INFINITY; group];
    let mut group_denom: Vec<f32> = vec![0.0; group];
    let mut k_block = [0.0f32; Q8_0_BLOCK_SIZE];
    let mut v_block = [0.0f32; Q8_0_BLOCK_SIZE];

    for kv_head in 0..num_kv_heads {
        let group_start = kv_head * group;

        // Reset per-group state.
        for s in &mut group_scores[..group * cache_len] {
            *s = 0.0;
        }
        for m in &mut group_max[..group] {
            *m = f32::NEG_INFINITY;
        }
        for d in &mut group_denom[..group] {
            *d = 0.0;
        }

        // K step. Dequant each block once; dot against every group
        // query head before moving on. Accumulation order per
        // (gi, l) stays block-by-block — same as the original
        // q_head-outer code, so the f32 reduction order (and Paris
        // MATCH) is preserved.
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
            // Apply scale + accumulate max per gi for this l.
            for gi in 0..group {
                let s = group_scores[gi * cache_len + l] * scale_f32;
                group_scores[gi * cache_len + l] = s;
                if s > group_max[gi] {
                    group_max[gi] = s;
                }
            }
        }

        // Softmax in-place: exp(x - max), accumulate denom, then
        // normalise. Two passes per gi (one to build weights and
        // denom, one to divide); the second pass turns scores into
        // pre-multiplied weights so the V step's axpy can fold the
        // 1/denom division out of the hot loop.
        for gi in 0..group {
            let max_g = group_max[gi];
            let row = &mut group_scores[gi * cache_len..(gi + 1) * cache_len];
            let mut denom = 0.0f32;
            for s in row.iter_mut() {
                let e = fmath::exp(*s - max_g);
                *s = e;
                denom += e;
            }
            group_denom[gi] = denom;
            let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            for s in row.iter_mut() {
                *s *= inv_denom;
            }
        }

        // Zero output rows for the group's q_heads before the
        // first axpy. q_heads in the same GQA group are stored at
        // contiguous offsets, so the four zeroings stream cleanly.
        for gi in 0..group {
            let q_head = group_start + gi;
            let out_row_ptr = out_data.add(q_head * head_dim);
            for d in 0..head_dim {
                *out_row_ptr.add(d) = 0.0;
            }
        }

        // V step. Dequant each block once; axpy into every group
        // query head's output row before moving on. Same dequant
        // savings as the K step (4× less work per cache block);
        // axpy work is unchanged.
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
