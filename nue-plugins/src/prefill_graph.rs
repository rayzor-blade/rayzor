//! Fused Llama PREFILL graph engine — macOS only, CPU CoreML.
//!
//! Runs the whole block stack over a prompt bucket in ONE CoreML call and
//! hands back the final hidden states plus every layer's K (post-rope,
//! interleaved GGUF convention) and V in cache layout `[S, kv_heads,
//! head_dim]` — exactly what `KvCacheQ8.append()` consumes, so on unified
//! memory the CPU decode loop continues from the graph's KV with zero
//! copies beyond the cache quantization it already does.
//!
//! Probe-measured ~15x over the per-op Q4 pipeline (87 ms vs 1.35 s per
//! 128-token prefill on M1): the win is whole-graph fusion + pre-dequantized
//! fp16 weights. ANE rejects llama dimensions (see llama_prefill_probe.py) —
//! the CoreML compute units are pinned to CPU_ONLY.
//!
//! Artifacts: `<gguf-stem>.prefill_s{S}.mlmodelc` next to the gguf, authored
//! by `examples/llama_prefill_author.py`, fixed-shape per bucket.

#[cfg(target_os = "macos")]
mod imp {
    use std::collections::BTreeMap;
    use std::os::raw::{c_char, c_void};
    use std::sync::{Mutex, OnceLock};

    pub const BUCKETS: [usize; 2] = [128, 512];

    extern "C" {
        fn nue_coreml_load(path: *const c_char, compute_units: i32) -> *mut c_void;
        fn nue_coreml_predict_prefill(
            handle: *mut c_void,
            h: *const f32,
            out: *mut f32,
            kv: *mut f32,
            s: i64,
            hidden: i64,
            layers: i64,
            kv_heads: i64,
            head_dim: i64,
        ) -> i32;
    }

    struct Model {
        hidden: usize,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
        buckets: BTreeMap<usize, *mut c_void>,
    }
    unsafe impl Send for Model {}

    struct Registry {
        by_handle: BTreeMap<i64, Model>,
        by_key: BTreeMap<String, i64>,
        next: i64,
    }

    fn registry() -> &'static Mutex<Registry> {
        static R: OnceLock<Mutex<Registry>> = OnceLock::new();
        R.get_or_init(|| {
            Mutex::new(Registry {
                by_handle: BTreeMap::new(),
                by_key: BTreeMap::new(),
                next: 1,
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dir: &str,
        stem: &str,
        hidden: usize,
        layers: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> i64 {
        let mut reg = registry().lock().unwrap();
        let key = format!("{dir}\u{1}{stem}\u{1}prefill");
        if let Some(&h) = reg.by_key.get(&key) {
            return h;
        }
        let mut buckets = BTreeMap::new();
        for &s in BUCKETS.iter() {
            let path = format!("{dir}/{stem}.prefill_s{s}.mlmodelc");
            if !std::path::Path::new(&path).exists() {
                continue;
            }
            let Some(c) = std::ffi::CString::new(path).ok() else {
                continue;
            };
            // CPU_ONLY (0): ANE rejects llama dims; CPU is where the 15x is.
            let m = unsafe { nue_coreml_load(c.as_ptr(), 0) };
            if m.is_null() {
                eprintln!("[prefill-graph] load failed: {stem} bucket {s}");
                continue;
            }
            buckets.insert(s, m);
        }
        if buckets.is_empty() {
            return 0;
        }
        let h = reg.next;
        reg.next += 1;
        reg.by_handle.insert(
            h,
            Model {
                hidden,
                layers,
                kv_heads,
                head_dim,
                buckets,
            },
        );
        reg.by_key.insert(key, h);
        h
    }

    pub fn bucket_for(handle: i64, seq: usize) -> usize {
        let reg = registry().lock().unwrap();
        let Some(m) = reg.by_handle.get(&handle) else {
            return 0;
        };
        for &s in m.buckets.keys() {
            if seq <= s {
                return s;
            }
        }
        0
    }

    pub fn run(handle: i64, s: usize, h: *const f32, out: *mut f32, kv: *mut f32) -> i32 {
        let reg = registry().lock().unwrap();
        let Some(m) = reg.by_handle.get(&handle) else {
            return -1;
        };
        let Some(&mm) = m.buckets.get(&s) else {
            return -1;
        };
        unsafe {
            nue_coreml_predict_prefill(
                mm,
                h,
                out,
                kv,
                s as i64,
                m.hidden as i64,
                m.layers as i64,
                m.kv_heads as i64,
                m.head_dim as i64,
            )
        }
    }
}

/// Load `<stem>.prefill_s{S}.mlmodelc` buckets from `dir` for one model.
/// Strings are raw (ptr, len) UTF-8; dims come from the model's metadata.
/// Returns a handle > 0, 0 when no artifacts, -1 off-macOS.
#[no_mangle]
pub unsafe extern "C" fn nue_prefill_graph_load(
    dir_ptr: i64,
    dir_len: i64,
    stem_ptr: i64,
    stem_len: i64,
    hidden: i64,
    layers: i64,
    kv_heads: i64,
    head_dim: i64,
) -> i64 {
    #[cfg(target_os = "macos")]
    {
        if dir_ptr == 0 || dir_len <= 0 || stem_ptr == 0 || stem_len <= 0 {
            return 0;
        }
        if hidden <= 0 || layers <= 0 || kv_heads <= 0 || head_dim <= 0 {
            return 0;
        }
        let d = std::slice::from_raw_parts(dir_ptr as *const u8, dir_len as usize);
        let s = std::slice::from_raw_parts(stem_ptr as *const u8, stem_len as usize);
        let (Ok(dir), Ok(stem)) = (std::str::from_utf8(d), std::str::from_utf8(s)) else {
            return 0;
        };
        imp::load(
            dir,
            stem,
            hidden as usize,
            layers as usize,
            kv_heads as usize,
            head_dim as usize,
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (
            dir_ptr, dir_len, stem_ptr, stem_len, hidden, layers, kv_heads, head_dim,
        );
        -1
    }
}

/// Copy one KV slot out of the contiguous prefill KV block into a caller-owned
/// `[s_real, kv_heads, head_dim]` F32 buffer `dst` — the exact shape
/// `KvCacheQ8.append()` consumes. `slot` = 2*layer for K, 2*layer+1 for V;
/// the block's per-slot stride is the full BUCKET, `dst` takes the real-prompt
/// prefix rows. A dest-pointer out-param (the caller allocates the tensor via
/// the `Tensor.uninit` builtin) keeps this a plain bulk copy — no per-slot owned
/// Tensor handle to hand back across FFI. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn nue_prefill_graph_kv_copy(
    kv_ptr: i64,
    slot: i64,
    s_real: i64,
    bucket: i64,
    kv_heads: i64,
    head_dim: i64,
    dst_ptr: i64,
) -> i64 {
    #[cfg(target_os = "macos")]
    {
        if kv_ptr == 0
            || dst_ptr == 0
            || s_real <= 0
            || bucket < s_real
            || kv_heads <= 0
            || head_dim <= 0
        {
            return -1;
        }
        let row = (kv_heads * head_dim) as usize;
        let n = s_real as usize * row;
        let src = (kv_ptr as *const f32).add(slot as usize * bucket as usize * row);
        std::ptr::copy_nonoverlapping(src, dst_ptr as *mut f32, n);
        0
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (kv_ptr, slot, s_real, bucket, kv_heads, head_dim, dst_ptr);
        -1
    }
}

/// Smallest loaded bucket of `handle` that fits `seq`, or 0.
#[no_mangle]
pub extern "C" fn nue_prefill_graph_bucket(handle: i64, seq: i64) -> i64 {
    #[cfg(target_os = "macos")]
    {
        imp::bucket_for(handle, seq.max(0) as usize) as i64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (handle, seq);
        0
    }
}

/// Run bucket `s`: h[s*hidden] -> out[s*hidden] + kv laid out
/// [layers][2][s*kv_heads*head_dim] (k slot 2i, v slot 2i+1), all f32.
/// Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn nue_prefill_graph_execute(
    handle: i64,
    s: i64,
    h_ptr: i64,
    out_ptr: i64,
    kv_ptr: i64,
) -> i64 {
    #[cfg(target_os = "macos")]
    {
        if h_ptr == 0 || out_ptr == 0 || kv_ptr == 0 {
            return -2;
        }
        imp::run(
            handle,
            s as usize,
            h_ptr as *const f32,
            out_ptr as *mut f32,
            kv_ptr as *mut f32,
        ) as i64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (handle, s, h_ptr, out_ptr, kv_ptr);
        -1
    }
}
