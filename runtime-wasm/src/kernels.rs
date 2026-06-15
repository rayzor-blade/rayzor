//! Higher-level F32 tensor kernel exports (flash attention, softmax, RoPE,
//! RMSNorm). Thin wrappers around the runtime-core kernels with the wasm-
//! preferred math primitives (`libm::expf`, `libm::sqrtf`) bound in.

use core::slice;
use rayzor_runtime_core::tensor::topk::recent_contains;
use rayzor_runtime_core::tensor::{flash_attn, rms_norm, rope};

use crate::tensor::{load_f32_at, store_f32_at, Tensor, DTYPE_F16, DTYPE_F32};

/// `out[q_head, :] = Σ_l softmax((Q[h] · K[l, kv_head, :]) * scale) * V[l, kv_head, :]`.
///
/// All buffers are F32. Layout (mirrors the native `rayzor_tensor_flash_attn_decode`):
///   - `q_data`  : `[n_q_heads * head_dim]`
///   - `k_data`  : `[cache_len * kv_row_stride]` (typically
///                 `kv_row_stride = n_kv_heads * head_dim`)
///   - `v_data`  : same shape as `k_data`
///   - `out_data`: `[n_q_heads * head_dim]`
///
/// `group = n_q_heads / n_kv_heads` (GQA grouping factor; 4 for Llama 3 1B,
/// 8 for Llama 3 70B).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode_f32(
    q_data: i32,
    k_data: i32,
    v_data: i32,
    out_data: i32,
    n_q_heads: i32,
    group: i32,
    head_dim: i32,
    cache_len: i32,
    kv_row_stride: i32,
    scale: f32,
) {
    if q_data == 0 || k_data == 0 || v_data == 0 || out_data == 0 {
        return;
    }
    if n_q_heads <= 0 || group <= 0 || head_dim <= 0 || cache_len <= 0 {
        return;
    }
    let n_q_heads = n_q_heads as usize;
    let group = group as usize;
    let head_dim = head_dim as usize;
    let cache_len = cache_len as usize;
    let kv_row_stride = kv_row_stride as usize;

    // Attention is the per-token cost that GROWS with context: each q_head
    // does O(cache_len) work (Q·Kᵀ → softmax → ·V over the KV cache). The
    // q_head outputs are disjoint (`out_data[q_head*head_dim..]`), so split the
    // heads into bands across the same persistent worker pool the matmul uses
    // (`rayzor_thread_spawn`). Each worker gets its OWN `scores` scratch (the
    // only mutable per-head state), pre-allocated here so the workers stay
    // allocation-free. Sequential below a threshold where the spawn/join tax
    // would outweigh the (small) short-context attention work.
    #[cfg(target_arch = "wasm32")]
    {
        let env_n = crate::qtensor::guest_threads_override();
        let nthreads = if env_n > 0 { env_n.min(8) } else { 6 };
        if nthreads > 1 && n_q_heads >= 2 && cache_len >= 256 {
            let nbands = nthreads.min(n_q_heads);
            let band = n_q_heads.div_ceil(nbands);
            // Per-worker scratch: `nbands` contiguous `cache_len`-wide slices.
            let mut scratch: std::vec::Vec<f32> = std::vec![0.0; nbands * cache_len];
            let scratch_base = scratch.as_mut_ptr();
            let fn_idx = rayzor_flash_band_worker as *const () as usize as i32;
            let mut handles: std::vec::Vec<i32> = std::vec::Vec::new();
            let mut works: std::vec::Vec<*mut FlashWork> = std::vec::Vec::new();
            for t in 0..nbands {
                let lo = (t * band).min(n_q_heads);
                let hi = ((t + 1) * band).min(n_q_heads);
                if lo >= hi {
                    break;
                }
                let work = std::boxed::Box::into_raw(std::boxed::Box::new(FlashWork {
                    q_data,
                    k_data,
                    v_data,
                    out_data,
                    group,
                    head_dim,
                    cache_len,
                    kv_row_stride,
                    scale,
                    scores: scratch_base.add(t * cache_len) as i32,
                    qh_lo: lo,
                    qh_hi: hi,
                }));
                works.push(work);
                let h = rayzor_thread_spawn(fn_idx, work as usize as i32);
                if h == 0 {
                    flash_band(&*work);
                } else {
                    handles.push(h);
                }
            }
            // Join before `scratch` (each worker's scores buffer) drops.
            for h in handles {
                rayzor_thread_join_void(h);
            }
            for w in works {
                drop(std::boxed::Box::from_raw(w));
            }
            return;
        }
    }

    let mut scores: std::vec::Vec<f32> = std::vec![0.0; cache_len];
    for q_head in 0..n_q_heads {
        flash_attn::flash_attn_decode_one_qhead(
            q_head,
            group,
            head_dim,
            cache_len,
            kv_row_stride,
            scale,
            q_data as *const f32,
            k_data as *const f32,
            v_data as *const f32,
            out_data as *mut f32,
            &mut scores,
            libm::expf,
        );
    }
}

/// Per-band flash job: dot q_heads `qh_lo..qh_hi` against the KV cache, using
/// this worker's private `scores` scratch (`cache_len` wide).
#[cfg(target_arch = "wasm32")]
#[repr(C)]
struct FlashWork {
    q_data: i32,
    k_data: i32,
    v_data: i32,
    out_data: i32,
    group: usize,
    head_dim: usize,
    cache_len: usize,
    kv_row_stride: usize,
    scale: f32,
    scores: i32,
    qh_lo: usize,
    qh_hi: usize,
}

#[cfg(target_arch = "wasm32")]
unsafe fn flash_band(w: &FlashWork) {
    let scores = slice::from_raw_parts_mut(w.scores as *mut f32, w.cache_len);
    for q_head in w.qh_lo..w.qh_hi {
        flash_attn::flash_attn_decode_one_qhead(
            q_head,
            w.group,
            w.head_dim,
            w.cache_len,
            w.kv_row_stride,
            w.scale,
            w.q_data as *const f32,
            w.k_data as *const f32,
            w.v_data as *const f32,
            w.out_data as *mut f32,
            scores,
            libm::expf,
        );
    }
}

/// Worker entry invoked on a pool thread via the indirect function table.
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn rayzor_flash_band_worker(work_ptr: i32) -> i64 {
    unsafe { flash_band(&*(work_ptr as *const FlashWork)) };
    0
}

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "rayzor")]
extern "C" {
    fn rayzor_thread_spawn(fn_idx: i32, env_ptr: i32) -> i32;
    fn rayzor_thread_join_void(handle: i32);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_flash_attn_decode(
    q_ptr: i32,
    k_ptr: i32,
    v_ptr: i32,
    scale: f64,
) -> i32 {
    if q_ptr == 0 || k_ptr == 0 || v_ptr == 0 {
        return 0;
    }
    let q = &*(q_ptr as *const Tensor);
    let k = &*(k_ptr as *const Tensor);
    let v = &*(v_ptr as *const Tensor);
    if q.dtype != DTYPE_F32 || k.dtype != DTYPE_F32 || v.dtype != DTYPE_F32 {
        return 0;
    }
    if q.ndim != 3 || k.ndim != 3 || v.ndim != 3 {
        return 0;
    }
    let q_shape = slice::from_raw_parts(q.shape, 3);
    let k_shape = slice::from_raw_parts(k.shape, 3);
    let v_shape = slice::from_raw_parts(v.shape, 3);
    let seq_q = q_shape[0];
    let n_q_heads = q_shape[1];
    let head_dim = q_shape[2];
    let cache_len = k_shape[0];
    let n_kv_heads = k_shape[1];
    if seq_q != 1
        || k_shape[2] != head_dim
        || v_shape[0] != cache_len
        || v_shape[1] != n_kv_heads
        || v_shape[2] != head_dim
        || n_kv_heads == 0
        || !n_q_heads.is_multiple_of(n_kv_heads)
    {
        return 0;
    }
    let k_strides = slice::from_raw_parts(k.strides, 3);
    let v_strides = slice::from_raw_parts(v.strides, 3);
    let kv_row_stride = n_kv_heads * head_dim;
    if !q.is_contiguous()
        || k_strides[0] != kv_row_stride
        || k_strides[1] != head_dim
        || k_strides[2] != 1
        || v_strides[0] != kv_row_stride
        || v_strides[1] != head_dim
        || v_strides[2] != 1
    {
        return 0;
    }
    let out_shape = [1usize, n_q_heads, head_dim];
    let out = crate::tensor::alloc_tensor(&out_shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    rayzor_tensor_flash_attn_decode_f32(
        q.data as i32,
        k.data as i32,
        v.data as i32,
        (*out).data as i32,
        n_q_heads as i32,
        (n_q_heads / n_kv_heads) as i32,
        head_dim as i32,
        cache_len as i32,
        kv_row_stride as i32,
        scale as f32,
    );
    out as i32
}

/// In-place row-wise softmax over an `[n_rows, row_len]` F32 buffer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_softmax_inplace_f32(data: i32, n_rows: i32, row_len: i32) {
    if data == 0 || n_rows <= 0 || row_len <= 0 {
        return;
    }
    let row_len = row_len as usize;
    let p = data as *mut f32;
    for r in 0..n_rows as usize {
        let row = slice::from_raw_parts_mut(p.add(r * row_len), row_len);
        // Manual numerically-stable softmax. The causal mask writes
        // f32::NEG_INFINITY into masked cells; guard non-finite exp args so
        // wasm's libm::expf (which traps on non-finite) matches native
        // `x.exp()` semantics (exp(-inf)=0).
        let mut m = f32::NEG_INFINITY;
        for &x in row.iter() {
            if x > m {
                m = x;
            }
        }
        if !m.is_finite() {
            m = 0.0;
        }
        let mut denom = 0.0f32;
        for v in row.iter_mut() {
            let d = *v - m;
            let e = if d.is_finite() { libm::expf(d) } else { 0.0 };
            *v = e;
            denom += e;
        }
        if denom > 0.0 {
            let inv = 1.0 / denom;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_softmax(t: i32) -> i32 {
    if t == 0 {
        return 0;
    }
    let tr = &*(t as *const Tensor);
    if tr.dtype != DTYPE_F32 || tr.ndim == 0 {
        return 0;
    }
    let shape = slice::from_raw_parts(tr.shape, tr.ndim);
    let out = crate::tensor::alloc_tensor(shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    core::ptr::copy_nonoverlapping(tr.data, (*out).data, tr.numel * 4);
    let row_len = *shape.last().unwrap_or(&0);
    if row_len == 0 {
        return out as i32;
    }
    rayzor_tensor_softmax_inplace_f32(
        (*out).data as i32,
        (tr.numel / row_len) as i32,
        row_len as i32,
    );
    out as i32
}

/// Row-wise RMSNorm over an `[n_rows, hidden_dim]` F32 buffer, writing to
/// `out`. `weight` is the per-channel learnable gain (`[hidden_dim]`).
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rms_norm_f32(
    out: i32,
    x: i32,
    weight: i32,
    eps: f32,
    n_rows: i32,
    hidden_dim: i32,
) {
    if out == 0 || x == 0 || weight == 0 || n_rows <= 0 || hidden_dim <= 0 {
        return;
    }
    let n = hidden_dim as usize;
    let out_p = out as *mut f32;
    let x_p = x as *const f32;
    let w_slice = slice::from_raw_parts(weight as *const f32, n);
    for r in 0..n_rows as usize {
        let x_row = slice::from_raw_parts(x_p.add(r * n), n);
        let out_row = slice::from_raw_parts_mut(out_p.add(r * n), n);
        rms_norm::rms_norm_row_f32(out_row, x_row, w_slice, eps, libm::sqrtf);
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rms_norm(x: i32, eps: f64) -> i32 {
    if x == 0 {
        return 0;
    }
    let xr = &*(x as *const Tensor);
    if xr.dtype != DTYPE_F32 || xr.ndim == 0 {
        return 0;
    }
    let shape = slice::from_raw_parts(xr.shape, xr.ndim);
    let hidden = *shape.last().unwrap_or(&0);
    if hidden == 0 {
        return 0;
    }
    let out = crate::tensor::alloc_tensor(shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let ones_shape = [hidden];
    let ones = crate::tensor::alloc_tensor(&ones_shape, DTYPE_F32);
    if ones.is_null() {
        crate::tensor::rayzor_tensor_free(out as i32);
        return 0;
    }
    let weights = slice::from_raw_parts_mut((*ones).data as *mut f32, hidden);
    for w in weights.iter_mut() {
        *w = 1.0;
    }
    rayzor_tensor_rms_norm_f32(
        (*out).data as i32,
        xr.data as i32,
        (*ones).data as i32,
        eps as f32,
        (xr.numel / hidden) as i32,
        hidden as i32,
    );
    crate::tensor::rayzor_tensor_free(ones as i32);
    out as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_layer_norm(x: i32, eps: f64) -> i32 {
    if x == 0 {
        return 0;
    }
    let xr = &*(x as *const Tensor);
    if xr.dtype != DTYPE_F32 || xr.ndim == 0 {
        return 0;
    }
    let shape = slice::from_raw_parts(xr.shape, xr.ndim);
    let hidden = *shape.last().unwrap_or(&0);
    if hidden == 0 {
        return 0;
    }
    let out = crate::tensor::alloc_tensor(shape, DTYPE_F32);
    if out.is_null() {
        return 0;
    }
    let rows = xr.numel / hidden;
    let src = xr.data as *const f32;
    let dst = (*out).data as *mut f32;
    let eps = eps as f32;
    for row in 0..rows {
        let x_row = slice::from_raw_parts(src.add(row * hidden), hidden);
        let y_row = slice::from_raw_parts_mut(dst.add(row * hidden), hidden);
        let mean = x_row.iter().copied().sum::<f32>() / hidden as f32;
        let var = x_row
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / hidden as f32;
        let inv = 1.0 / libm::sqrtf(var + eps);
        for i in 0..hidden {
            y_row[i] = (x_row[i] - mean) * inv;
        }
    }
    out as i32
}

/// Apply interleaved Llama/GGUF RoPE to an F32 tensor of shape
/// `[seq_len, num_heads, head_dim]` or `[seq_len, head_dim]`.
///
/// Returns a freshly allocated Tensor with the same shape. This export uses
/// the same public name as the native runtime so Haxe's existing
/// `Tensor.rope()` stdlib wrapper resolves on WASM too.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope(
    x: i32,
    cos: i32,
    sin: i32,
    position_offset: i32,
) -> i32 {
    if x == 0 || cos == 0 || sin == 0 {
        return 0;
    }
    let xr = &*(x as *const Tensor);
    let cr = &*(cos as *const Tensor);
    let sr = &*(sin as *const Tensor);
    if xr.dtype != DTYPE_F32 {
        return 0;
    }
    if xr.ndim < 2 || cr.ndim != 2 || sr.ndim != 2 {
        return 0;
    }

    let x_shape = slice::from_raw_parts(xr.shape, xr.ndim);
    let head_dim = x_shape[xr.ndim - 1];
    if !head_dim.is_multiple_of(2) {
        return 0;
    }
    let half = head_dim / 2;
    let num_heads = if xr.ndim >= 3 {
        x_shape[xr.ndim - 2]
    } else {
        1
    };
    let seq_len = if xr.ndim >= 3 {
        x_shape[..xr.ndim - 2].iter().product::<usize>().max(1)
    } else {
        x_shape[0]
    };

    let cos_shape = slice::from_raw_parts(cr.shape, 2);
    let sin_shape = slice::from_raw_parts(sr.shape, 2);
    if cos_shape[1] != half || sin_shape[1] != half || sin_shape[0] != cos_shape[0] {
        return 0;
    }

    let y = crate::tensor::alloc_tensor(x_shape, DTYPE_F32);
    if y.is_null() {
        return 0;
    }
    let y = y as i32;
    let yr = &*(y as *const Tensor);
    if cr.dtype == DTYPE_F32 && sr.dtype == DTYPE_F32 {
        let out = slice::from_raw_parts_mut(yr.data as *mut f32, yr.numel);
        let x_data = slice::from_raw_parts(xr.data as *const f32, xr.numel);
        let cos_data = slice::from_raw_parts(cr.data as *const f32, cr.numel);
        let sin_data = slice::from_raw_parts(sr.data as *const f32, sr.numel);
        rope::apply_interleaved_f32(
            out,
            x_data,
            cos_data,
            sin_data,
            seq_len,
            num_heads,
            head_dim,
            position_offset.max(0) as usize,
        );
    } else {
        let pos_off = position_offset.max(0) as usize;
        let elements_per_row = num_heads * head_dim;
        let x_data = xr.data as *const f32;
        let out_data = yr.data as *mut f32;
        for s in 0..seq_len {
            let pos = s + pos_off;
            let row_base = s * elements_per_row;
            if pos >= cos_shape[0] {
                core::ptr::copy_nonoverlapping(
                    x_data.add(row_base),
                    out_data.add(row_base),
                    elements_per_row,
                );
                continue;
            }
            for h in 0..num_heads {
                let head_base = row_base + h * head_dim;
                let table_base = pos * half;
                for i in 0..half {
                    let c = load_f32_at(cr.data, table_base + i, cr.dtype);
                    let sn = load_f32_at(sr.data, table_base + i, sr.dtype);
                    let lo = head_base + 2 * i;
                    let hi = lo + 1;
                    let xlo = *x_data.add(lo);
                    let xhi = *x_data.add(hi);
                    *out_data.add(lo) = xlo * c - xhi * sn;
                    *out_data.add(hi) = xlo * sn + xhi * c;
                }
            }
        }
    }
    y
}

/// Generate an F32 RoPE cosine table `[max_seq_len, head_dim / 2]`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, libm::cos)
}

/// Generate an F32 RoPE sine table `[max_seq_len, head_dim / 2]`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F32, libm::sin)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_cos_table_f16(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, libm::cos)
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_rope_sin_table_f16(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
) -> i32 {
    rope_table(head_dim, max_seq_len, base, DTYPE_F16, libm::sin)
}

unsafe fn rope_table<F: Fn(f64) -> f64>(
    head_dim: i32,
    max_seq_len: i32,
    base: f64,
    dtype: u8,
    trig_fn: F,
) -> i32 {
    if head_dim <= 0 || max_seq_len <= 0 || head_dim % 2 != 0 {
        return 0;
    }
    let head_dim = head_dim as usize;
    let max_seq_len = max_seq_len as usize;
    let shape = [max_seq_len, head_dim / 2];
    let t = crate::tensor::alloc_tensor(&shape, dtype);
    if t.is_null() {
        return 0;
    }
    let t = t as i32;
    let tr = &*(t as *const Tensor);
    if dtype == DTYPE_F32 {
        let out = slice::from_raw_parts_mut(tr.data as *mut f32, tr.numel);
        rope::fill_table_f32(out, head_dim, max_seq_len, base, trig_fn);
    } else {
        let half = head_dim / 2;
        let mut scratch = std::vec![0.0f32; max_seq_len * half];
        rope::fill_table_f32(&mut scratch, head_dim, max_seq_len, base, trig_fn);
        for (i, &v) in scratch.iter().enumerate() {
            store_f32_at(tr.data, i, dtype, v);
        }
    }
    t
}

/// Return the index of the maximum F32 value in `data[0..n]`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_argmax_f32(data: i32, n: i32) -> i32 {
    if data == 0 || n <= 0 {
        return -1;
    }
    let values = slice::from_raw_parts(data as *const f32, n as usize);
    let mut best_idx = 0usize;
    let mut best = values[0];
    for (idx, &v) in values.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            best_idx = idx;
        }
    }
    best_idx as i32
}

/// Raw F32 top-k scan used by wasm-side samplers and tests.
///
/// Writes descending logits into `out_logits` (`f32*`) and ids into `out_ids`
/// (`i32*`). `cutoff` is an additional fast-reject threshold; pass
/// `f32::NEG_INFINITY` to disable it.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_topk_scan_f32(
    data: i32,
    n: i32,
    k: i32,
    recent_ids: i32,
    recent_n: i32,
    rep_penalty: f32,
    cutoff: f32,
    out_logits: i32,
    out_ids: i32,
) {
    if data == 0 || n <= 0 || k <= 0 || out_logits == 0 || out_ids == 0 {
        return;
    }
    let n = n as usize;
    let k = (k as usize).min(n);
    let values = slice::from_raw_parts(data as *const f32, n);
    let recent = if recent_ids != 0 && recent_n > 0 && rep_penalty > 1.0 {
        Some(slice::from_raw_parts(
            recent_ids as *const i64,
            recent_n as usize,
        ))
    } else {
        None
    };
    let out_logits = out_logits as *mut f32;
    let out_ids = out_ids as *mut i32;
    let mut size = 0usize;
    for (idx, &raw) in values.iter().enumerate() {
        let mut v = raw;
        if let Some(recent) = recent {
            if recent_contains(recent, idx as i64) {
                v = if v > 0.0 {
                    v / rep_penalty
                } else {
                    v * rep_penalty
                };
            }
        }
        if v <= cutoff {
            continue;
        }
        if size == k && v <= *out_logits.add(k - 1) {
            continue;
        }
        let end = if size < k {
            let end = size;
            size += 1;
            end
        } else {
            k - 1
        };
        let mut pos = end;
        while pos > 0 && *out_logits.add(pos - 1) < v {
            *out_logits.add(pos) = *out_logits.add(pos - 1);
            *out_ids.add(pos) = *out_ids.add(pos - 1);
            pos -= 1;
        }
        *out_logits.add(pos) = v;
        *out_ids.add(pos) = idx as i32;
    }
}

/// Native-shaped Tensor top-k scan used by Haxe `Tensor.topkScan`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tensor_topk_scan(
    tensor: i32,
    out_logits_ptr: i32,
    out_ids_ptr: i32,
    k: i32,
    recent_ids_ptr: i32,
    recent_len: i32,
    repetition_penalty: f64,
) -> i32 {
    if tensor == 0 || out_logits_ptr == 0 || out_ids_ptr == 0 {
        return -1;
    }
    let t = &*(tensor as *const Tensor);
    if t.dtype != DTYPE_F32 || !t.is_contiguous() {
        return -1;
    }
    let n = t.numel;
    let k = (k.max(0) as usize).min(n);
    if k == 0 {
        return 0;
    }
    let src = slice::from_raw_parts(t.data as *const f32, n);
    let out_logits = out_logits_ptr as *mut f64;
    let out_ids = out_ids_ptr as *mut i64;
    let penalize = repetition_penalty > 1.0 && recent_ids_ptr != 0 && recent_len > 0;
    let recent = if penalize {
        Some(slice::from_raw_parts(
            recent_ids_ptr as *const i64,
            recent_len as usize,
        ))
    } else {
        None
    };
    let rp = repetition_penalty;
    let mut size = 0usize;
    for (idx, &raw) in src.iter().enumerate() {
        let mut v = raw as f64;
        if let Some(recent) = recent {
            if recent_contains(recent, idx as i64) {
                v = if v > 0.0 { v / rp } else { v * rp };
            }
        }
        if size == k && v <= *out_logits.add(k - 1) {
            continue;
        }
        let end = if size < k {
            let end = size;
            size += 1;
            end
        } else {
            k - 1
        };
        let mut pos = end;
        while pos > 0 && *out_logits.add(pos - 1) < v {
            *out_logits.add(pos) = *out_logits.add(pos - 1);
            *out_ids.add(pos) = *out_ids.add(pos - 1);
            pos -= 1;
        }
        *out_logits.add(pos) = v;
        *out_ids.add(pos) = idx as i64;
    }
    size as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_inplace_normalises_and_sums_to_one() {
        unsafe {
            let mut row = [1.0f32, 2.0, 3.0, 4.0];
            rayzor_tensor_softmax_inplace_f32(row.as_mut_ptr() as i32, 1, 4);
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
            // Monotonic — larger input ⇒ larger output.
            assert!(row[0] < row[1] && row[1] < row[2] && row[2] < row[3]);
        }
    }

    #[test]
    fn rms_norm_matches_scalar_reference() {
        unsafe {
            let x: [f32; 4] = [1.0, -2.0, 3.0, -4.0];
            let weight: [f32; 4] = [0.5, 0.25, 1.0, 2.0];
            let eps = 1e-6f32;
            let mut out = [0.0f32; 4];
            rayzor_tensor_rms_norm_f32(
                out.as_mut_ptr() as i32,
                x.as_ptr() as i32,
                weight.as_ptr() as i32,
                eps,
                1,
                4,
            );
            let ss: f32 = x.iter().map(|v| v * v).sum();
            let mean_sq = ss / 4.0;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..4 {
                let want = x[i] * weight[i] * inv;
                assert!(
                    (out[i] - want).abs() < 1e-5,
                    "i={} got={} want={}",
                    i,
                    out[i],
                    want
                );
            }
        }
    }

    #[test]
    fn flash_attn_decode_matches_reference_single_head() {
        unsafe {
            // 1 q_head, 1 kv_head, head_dim=4, cache_len=3.
            let q = [1.0f32, 2.0, -1.0, 0.5];
            let k = [
                0.5f32, -0.5, 1.0, 0.0, // K[0]
                -1.0, 1.0, 0.0, 0.5, // K[1]
                0.25, 0.25, 0.25, 0.25, // K[2]
            ];
            let v = [
                1.0f32, 0.0, 0.0, 0.0, // V[0]
                0.0, 1.0, 0.0, 0.0, // V[1]
                0.0, 0.0, 1.0, 0.0, // V[2]
            ];
            let mut out = [0.0f32; 4];
            let scale = 0.5f32;
            rayzor_tensor_flash_attn_decode_f32(
                q.as_ptr() as i32,
                k.as_ptr() as i32,
                v.as_ptr() as i32,
                out.as_mut_ptr() as i32,
                1,
                1,
                4,
                3,
                4,
                scale,
            );

            // Reference: scores, softmax, weighted V.
            let mut scores = [0.0f32; 3];
            let mut max = f32::NEG_INFINITY;
            for l in 0..3 {
                let mut dot = 0.0f32;
                for i in 0..4 {
                    dot += q[i] * k[l * 4 + i];
                }
                scores[l] = dot * scale;
                if scores[l] > max {
                    max = scores[l];
                }
            }
            let mut denom = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                denom += *s;
            }
            for s in scores.iter_mut() {
                *s /= denom;
            }
            let mut want = [0.0f32; 4];
            for l in 0..3 {
                for i in 0..4 {
                    want[i] += scores[l] * v[l * 4 + i];
                }
            }
            for i in 0..4 {
                assert!(
                    (out[i] - want[i]).abs() < 1e-5,
                    "i={} got={} want={}",
                    i,
                    out[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn rope_tables_and_apply_match_interleaved_reference() {
        unsafe {
            let cos = rayzor_tensor_rope_cos_table(4, 4, 10000.0);
            let sin = rayzor_tensor_rope_sin_table(4, 4, 10000.0);
            assert!(cos != 0);
            assert!(sin != 0);

            let x_data = [1.0f32, 2.0, 3.0, 4.0];
            let x_shape = [1usize, 1, 4];
            let x = crate::tensor::rayzor_tensor_from_floats(
                x_data.as_ptr() as i32,
                4,
                x_shape.as_ptr() as i32,
                3,
            );
            assert!(x != 0);

            let y = rayzor_tensor_rope(x, cos, sin, 1);
            assert!(y != 0);
            let yr = &*(y as *const Tensor);
            let got = slice::from_raw_parts(yr.data as *const f32, 4);

            let c0 = libm::cos(1.0) as f32;
            let s0 = libm::sin(1.0) as f32;
            let c1 = libm::cos(0.01) as f32;
            let s1 = libm::sin(0.01) as f32;
            let want = [
                x_data[0] * c0 - x_data[1] * s0,
                x_data[0] * s0 + x_data[1] * c0,
                x_data[2] * c1 - x_data[3] * s1,
                x_data[2] * s1 + x_data[3] * c1,
            ];
            for i in 0..4 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-6,
                    "i={} got={} want={}",
                    i,
                    got[i],
                    want[i]
                );
            }

            crate::tensor::rayzor_tensor_free(x);
            crate::tensor::rayzor_tensor_free(y);
            crate::tensor::rayzor_tensor_free(cos);
            crate::tensor::rayzor_tensor_free(sin);
        }
    }

    #[test]
    fn rope_f16_tables_apply_with_expected_precision() {
        unsafe {
            let cos = rayzor_tensor_rope_cos_table_f16(4, 4, 10000.0);
            let sin = rayzor_tensor_rope_sin_table_f16(4, 4, 10000.0);
            assert!(cos != 0);
            assert!(sin != 0);
            assert_eq!((*(cos as *const Tensor)).dtype, DTYPE_F16);
            assert_eq!((*(sin as *const Tensor)).dtype, DTYPE_F16);

            let x_data = [1.0f32, 2.0, 3.0, 4.0];
            let x_shape = [1usize, 1, 4];
            let x = crate::tensor::rayzor_tensor_from_floats(
                x_data.as_ptr() as i32,
                4,
                x_shape.as_ptr() as i32,
                3,
            );
            let y = rayzor_tensor_rope(x, cos, sin, 1);
            assert!(y != 0);

            let yr = &*(y as *const Tensor);
            let got = slice::from_raw_parts(yr.data as *const f32, 4);
            let c0 = libm::cos(1.0) as f32;
            let s0 = libm::sin(1.0) as f32;
            let c1 = libm::cos(0.01) as f32;
            let s1 = libm::sin(0.01) as f32;
            let want = [
                x_data[0] * c0 - x_data[1] * s0,
                x_data[0] * s0 + x_data[1] * c0,
                x_data[2] * c1 - x_data[3] * s1,
                x_data[2] * s1 + x_data[3] * c1,
            ];
            for i in 0..4 {
                assert!(
                    (got[i] - want[i]).abs() < 2e-3,
                    "i={} got={} want={}",
                    i,
                    got[i],
                    want[i]
                );
            }

            crate::tensor::rayzor_tensor_free(x);
            crate::tensor::rayzor_tensor_free(y);
            crate::tensor::rayzor_tensor_free(cos);
            crate::tensor::rayzor_tensor_free(sin);
        }
    }

    #[test]
    fn argmax_f32_returns_first_max_index() {
        let values = [1.0f32, 2.0, 3.0, 2.0, 1.0];
        unsafe {
            assert_eq!(rayzor_tensor_argmax_f32(values.as_ptr() as i32, 5), 2);
        }
    }

    #[test]
    fn raw_topk_scan_f32_returns_sorted_ids() {
        let values = [0.5f32, 0.9, 0.1, 0.7];
        let mut logits = [0.0f32; 2];
        let mut ids = [-1i32; 2];
        unsafe {
            rayzor_tensor_topk_scan_f32(
                values.as_ptr() as i32,
                4,
                2,
                0,
                0,
                1.0,
                f32::NEG_INFINITY,
                logits.as_mut_ptr() as i32,
                ids.as_mut_ptr() as i32,
            );
        }
        assert_eq!(ids, [1, 3]);
        assert_eq!(logits, [0.9, 0.7]);
    }

    #[test]
    fn tensor_topk_scan_matches_native_shape() {
        unsafe {
            let values = [0.5f32, 0.9, 0.1, 0.7];
            let shape = [4usize];
            let t = crate::tensor::rayzor_tensor_from_floats(
                values.as_ptr() as i32,
                4,
                shape.as_ptr() as i32,
                1,
            );
            assert!(t != 0);
            let mut logits = [0.0f64; 2];
            let mut ids = [-1i64; 2];
            let written = rayzor_tensor_topk_scan(
                t,
                logits.as_mut_ptr() as i32,
                ids.as_mut_ptr() as i32,
                2,
                0,
                0,
                1.0,
            );
            assert_eq!(written, 2);
            assert_eq!(ids, [1, 3]);
            assert!((logits[0] - 0.9).abs() < 1e-6);
            assert!((logits[1] - 0.7).abs() < 1e-6);
            crate::tensor::rayzor_tensor_free(t);
        }
    }
}
