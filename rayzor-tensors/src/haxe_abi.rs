//! Haxe-ABI adapters: entry points that accept Haxe runtime values
//! (`Array<T>` handles) directly and unpack them here, next to the kernels.
//!
//! Previously this unpacking lived as hand-built MIR glue in
//! `compiler/src/stdlib/tensor.rs` / `qtensor.rs` — a shadow copy of the
//! tensor ABI inside the compiler. Moving it here makes rayzor-tensors the
//! single home for the tensor implementation: the `@:native` names in
//! `haxe/rayzor/ds/Tensor.hx` / `QTensor.hx` bind straight to these symbols
//! via the plugin entry table, with no compiler-side mapping layer.
//!
//! ABI coupling (deliberate, documented): a Haxe `Array<T>` handle points at
//! `{ data: *T (offset 0), len: i64 (offset 8) }`. `Array<Float>` stores f64
//! elements; `Array<Int>` stores i64. `haxe.io.Bytes` handles are consumed
//! directly by the `rayzor_tensor_from_bytes_*` kernels, so no adapter is
//! needed for the bytes argument — only for the shape arrays.

use crate::quant::{rayzor_qtensor_from_f32_int8, rayzor_tensor_gather_rows_q6_k};
use crate::tensor::{
    rayzor_tensor_from_array, rayzor_tensor_from_bytes_f16, rayzor_tensor_from_bytes_f32,
    rayzor_tensor_from_bytes_q8_0, rayzor_tensor_full, rayzor_tensor_gather_rows,
    rayzor_tensor_get, rayzor_tensor_ones, rayzor_tensor_permute, rayzor_tensor_rand,
    rayzor_tensor_reshape, rayzor_tensor_set, rayzor_tensor_topk_scan, rayzor_tensor_uninit,
    rayzor_tensor_zeros, RayzorTensor,
};

/// Raw view of a Haxe `Array<T>` handle: `{ data ptr, element count }`.
#[repr(C)]
struct HaxeArrayRef {
    data: *const u8,
    len: i64,
}

/// (data_ptr, len) from a Haxe array handle; (0, 0) for a null handle.
#[inline]
unsafe fn arr(handle: i64) -> (i64, i64) {
    if handle == 0 {
        return (0, 0);
    }
    let a = &*(handle as *const HaxeArrayRef);
    (a.data as i64, a.len)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_zeros_arr(shape_arr: i64, dtype: i64) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_zeros(p, n, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_uninit_arr(shape_arr: i64, dtype: i64) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_uninit(p, n, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_ones_arr(shape_arr: i64, dtype: i64) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_ones(p, n, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_full_arr(shape_arr: i64, value: f64, dtype: i64) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_full(p, n, value, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rand_arr(shape_arr: i64, dtype: i64) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_rand(p, n, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_array_arr(data_arr: i64, dtype: i64) -> i64 {
    let (p, n) = arr(data_arr);
    rayzor_tensor_from_array(p, n, dtype)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_f32_arr(
    bytes_handle: i64,
    shape_arr: i64,
) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_from_bytes_f32(bytes_handle, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_f16_arr(
    bytes_handle: i64,
    shape_arr: i64,
) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_from_bytes_f16(bytes_handle, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_from_bytes_q8_0_arr(
    bytes_handle: i64,
    shape_arr: i64,
) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_from_bytes_q8_0(bytes_handle, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_reshape_arr(tensor: i64, shape_arr: i64) -> i64 {
    let (p, n) = arr(shape_arr);
    rayzor_tensor_reshape(tensor, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_permute_arr(tensor: i64, axes_arr: i64) -> i64 {
    let (p, n) = arr(axes_arr);
    rayzor_tensor_permute(tensor, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_get_arr(tensor: i64, indices_arr: i64) -> f64 {
    let (p, n) = arr(indices_arr);
    rayzor_tensor_get(tensor, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_set_arr(tensor: i64, indices_arr: i64, value: f64) {
    let (p, n) = arr(indices_arr);
    rayzor_tensor_set(tensor, p, n, value)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows_arr(tensor: i64, indices_arr: i64) -> i64 {
    let (p, n) = arr(indices_arr);
    rayzor_tensor_gather_rows(tensor, p, n)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_gather_rows_q6_k_arr(qt: i64, indices_arr: i64) -> i64 {
    let (p, n) = arr(indices_arr);
    rayzor_tensor_gather_rows_q6_k(qt, p, n)
}

/// Top-K scan over logits. `out_logits`/`out_ids` are caller-allocated Haxe
/// arrays with capacity >= k; we write through their data pointers (their
/// `len` fields are untouched — same contract the MIR glue had).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_topk_scan_arr(
    logits: i64,
    out_logits_arr: i64,
    out_ids_arr: i64,
    k: i64,
    recent_ids_arr: i64,
    penalty: f64,
) -> i64 {
    let (out_l, _) = arr(out_logits_arr);
    let (out_i, _) = arr(out_ids_arr);
    let (rec, rec_n) = arr(recent_ids_arr);
    rayzor_tensor_topk_scan(logits, out_l, out_i, k, rec, rec_n, penalty)
}

/// Fused QKV matmul writing the three result handles into a caller-provided
/// `Array<Tensor>` (pre-sized >= 3, slots null). On gate-miss (non-zero
/// return) the slots are left untouched — callers null-check, same contract
/// the MIR glue had.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_matmul_qkv_fused_arr(
    x: i64,
    q_w: i64,
    k_w: i64,
    v_w: i64,
    threads: i64,
    out_arr: i64,
) -> i64 {
    let (p, n) = arr(out_arr);
    if p == 0 || n < 3 {
        return -1;
    }
    let (mut q, mut k, mut v) = (0i64, 0i64, 0i64);
    let rc = crate::quant::rayzor_tensor_matmul_qkv_qt_t_f32_threaded(
        x, q_w, k_w, v_w, threads, &mut q, &mut k, &mut v,
    );
    if rc == 0 {
        let slots = p as *mut i64;
        *slots = q;
        *slots.add(1) = k;
        *slots.add(2) = v;
    }
    rc
}

/// `QTensor.fromFloat32(src, scheme)`: reads (data, rows, cols) from the
/// source tensor here — the struct is ours — then quantises. Only INT8
/// (scheme 0) routes through this entry; Q4_K_M comes from the loader.
#[no_mangle]
pub unsafe extern "C" fn rayzor_qtensor_from_f32_int8_t(src_tensor: i64, _scheme: i64) -> i64 {
    if src_tensor == 0 {
        return 0;
    }
    let t = &*(src_tensor as *const RayzorTensor);
    if t.ndim != 2 {
        return 0;
    }
    let shape = std::slice::from_raw_parts(t.shape, t.ndim);
    rayzor_qtensor_from_f32_int8(t.data as i64, shape[0] as i64, shape[1] as i64)
}
