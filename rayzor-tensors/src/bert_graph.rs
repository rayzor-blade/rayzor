//! BNNSGraph whole-encoder engine (BERT Phase 4) — macOS only.
//!
//! MODEL-GENERIC across the BERT family: artifacts are authored PER MODEL by
//! `examples/bert_graph_author.py` (dims read from the gguf's own `bert.*`
//! metadata, weights baked from its tensors) and named by the gguf stem —
//! `<stem>.encoder_s{S}.mlmodelc` — so several models can share a directory
//! and the runtime pairs them deterministically. `load` returns a HANDLE
//! keyed by (dir, stem); the hidden size is per-handle state, not a constant.
//!
//! One graph call runs the whole fused encoder stack:
//!
//!   in : h    [S, hidden] f32 (post-embedding hidden states)
//!        bias [S]         f32 (additive key mask: 0 real, -1e4 pad —
//!                              fp16-safe; the graph computes in fp16)
//!   out: out  [S, hidden] f32 (final hidden states, pre-pooling)
//!
//! Probe-verified (MiniLM, S=128): cosine 1.0 vs the CoreML reference,
//! ~2.1 ms/encode single-threaded (~475 sentences/s) vs ~170 sent/s for the
//! whole multi-threaded AMX-f16 pipeline.
//!
//! Landmines (from bert_graph_probe): argument order is the mlprogram's
//! alphabetized signature — resolve positions by NAME; Accelerate must be a
//! real framework link (build.rs does this crate-wide), not dynamic_lookup.
//!
//! Concurrency: execution is serialized by the registry mutex. Encode traffic
//! is per-sentence from the embed loop; 475/s single-thread is far above the
//! pipeline's tokenize+pool budget, so per-thread contexts are deferred.

#[cfg(target_os = "macos")]
mod imp {
    use std::collections::BTreeMap;
    use std::os::raw::{c_char, c_void};
    use std::sync::{Mutex, OnceLock};

    pub const BUCKETS: [usize; 3] = [128, 256, 512];

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct BnnsGraph {
        data: *mut c_void,
        size: usize,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct BnnsGraphContext {
        data: *mut c_void,
        size: usize,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct BnnsGraphCompileOptions {
        data: *mut c_void,
        size: usize,
    }
    /// Argument type defaults to Pointer, so the union collapses to data_ptr.
    #[repr(C)]
    struct BnnsGraphArgument {
        data_ptr: *mut c_void,
        data_ptr_size: usize,
    }

    extern "C" {
        #[link_name = "BNNSGraphCompileFromFile_v2"]
        fn compile_from_file(
            filename: *const c_char,
            function: *const c_char,
            options: BnnsGraphCompileOptions,
        ) -> BnnsGraph;
        #[link_name = "BNNSGraphContextMake"]
        fn context_make(graph: BnnsGraph) -> BnnsGraphContext;
        #[link_name = "BNNSGraphContextGetWorkspaceSize_v2"]
        fn workspace_size(context: BnnsGraphContext, function: *const c_char) -> usize;
        #[link_name = "BNNSGraphGetArgumentPosition"]
        fn argument_position(
            graph: BnnsGraph,
            function: *const c_char,
            argument: *const c_char,
        ) -> usize;
        #[link_name = "BNNSGraphContextExecute_v2"]
        fn execute(
            context: BnnsGraphContext,
            function: *const c_char,
            argument_count: usize,
            arguments: *mut BnnsGraphArgument,
            workspace_size: usize,
            workspace: *mut c_char,
        ) -> i32;
    }

    // In-process CoreML runtime (src/coreml_shim.m, compiled by build.rs) —
    // the ANE path. Same artifacts, different executor.
    extern "C" {
        fn rzt_coreml_load(path: *const c_char, compute_units: i32) -> *mut c_void;
        fn rzt_coreml_predict(
            handle: *mut c_void,
            h: *const f32,
            bias: *const f32,
            out: *mut f32,
            s: i64,
            hidden: i64,
        ) -> i32;
    }

    struct Engine {
        ctx: BnnsGraphContext,
        /// Positions of (out, h, bias) in the graph's argument order.
        pos: [usize; 3],
        ws: Vec<u8>,
        s: usize,
    }
    enum Backend {
        /// BNNSGraph — CPU-only by design.
        Bnns(Engine),
        /// CoreML runtime with computeUnits = CPU+NeuralEngine.
        CoreMl { model: *mut c_void, s: usize },
    }
    struct Model {
        hidden: usize,
        buckets: BTreeMap<usize, Backend>,
    }
    // Raw BNNS/CoreML handles are opaque; use is serialized through the mutex.
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

    fn load_bucket(dir: &str, stem: &str, s: usize) -> Option<Engine> {
        let path = format!("{dir}/{stem}.encoder_s{s}.mlmodelc");
        if !std::path::Path::new(&path).exists() {
            return None;
        }
        let cpath = std::ffi::CString::new(path).ok()?;
        let graph = unsafe {
            compile_from_file(
                cpath.as_ptr(),
                std::ptr::null(),
                BnnsGraphCompileOptions {
                    data: std::ptr::null_mut(),
                    size: 0,
                },
            )
        };
        if graph.data.is_null() {
            eprintln!("[bert-graph] compile failed: {stem} bucket {s}");
            return None;
        }
        let ctx = unsafe { context_make(graph) };
        if ctx.data.is_null() {
            eprintln!("[bert-graph] context_make failed: {stem} bucket {s}");
            return None;
        }
        let ws_len = unsafe { workspace_size(ctx, std::ptr::null()) };
        let mut pos = [0usize; 3];
        for (i, name) in ["out", "h", "bias"].iter().enumerate() {
            let c = std::ffi::CString::new(*name).ok()?;
            pos[i] = unsafe { argument_position(graph, std::ptr::null(), c.as_ptr()) };
        }
        Some(Engine {
            ctx,
            pos,
            ws: vec![0u8; ws_len.max(1)],
            s,
        })
    }

    /// Load every bucket artifact for (dir, stem) under the requested backend
    /// (`kind` 0 = BNNSGraph CPU, 1 = CoreML CPU+ANE). Returns a handle > 0
    /// when at least one bucket loaded, else 0. Idempotent per (dir, stem,
    /// kind).
    pub fn load(dir: &str, stem: &str, hidden: usize, kind: i64) -> i64 {
        let mut reg = registry().lock().unwrap();
        let key = format!("{dir}\u{1}{stem}\u{1}{kind}");
        if let Some(&h) = reg.by_key.get(&key) {
            return h;
        }
        let mut buckets = BTreeMap::new();
        for &s in BUCKETS.iter() {
            let backend = if kind == 1 {
                let path = format!("{dir}/{stem}.encoder_s{s}.mlmodelc");
                if !std::path::Path::new(&path).exists() {
                    None
                } else {
                    std::ffi::CString::new(path).ok().and_then(|c| {
                        let m = unsafe { rzt_coreml_load(c.as_ptr(), 1) };
                        if m.is_null() {
                            eprintln!("[bert-graph] coreml load failed: {stem} bucket {s}");
                            None
                        } else {
                            Some(Backend::CoreMl { model: m, s })
                        }
                    })
                }
            } else {
                load_bucket(dir, stem, s).map(Backend::Bnns)
            };
            if let Some(b) = backend {
                buckets.insert(s, b);
            }
        }
        if buckets.is_empty() {
            return 0;
        }
        let h = reg.next;
        reg.next += 1;
        reg.by_handle.insert(h, Model { hidden, buckets });
        reg.by_key.insert(key, h);
        h
    }

    /// Smallest loaded bucket of `handle` that fits `seq`, or 0.
    pub fn bucket_for(handle: i64, seq: usize) -> usize {
        let reg = registry().lock().unwrap();
        let Some(m) = reg.by_handle.get(&handle) else {
            return 0;
        };
        for (&s, _) in m.buckets.iter() {
            if seq <= s {
                return s;
            }
        }
        0
    }

    /// Execute bucket `s` of `handle`. `h`/`out` are `s*hidden` f32; `bias`
    /// is `s` f32. Returns the backend rc (0 = ok), -1 when not loaded.
    pub fn run(handle: i64, s: usize, h: *const f32, bias: *const f32, out: *mut f32) -> i32 {
        let mut reg = registry().lock().unwrap();
        let Some(model) = reg.by_handle.get_mut(&handle) else {
            return -1;
        };
        let hidden = model.hidden;
        let e = match model.buckets.get_mut(&s) {
            Some(Backend::Bnns(e)) => e,
            Some(Backend::CoreMl { model: m, s }) => {
                return unsafe { rzt_coreml_predict(*m, h, bias, out, *s as i64, hidden as i64) };
            }
            None => return -1,
        };
        let n = e.s * hidden;
        let mut args: Vec<BnnsGraphArgument> = (0..3)
            .map(|_| BnnsGraphArgument {
                data_ptr: std::ptr::null_mut(),
                data_ptr_size: 0,
            })
            .collect();
        args[e.pos[0]] = BnnsGraphArgument {
            data_ptr: out as *mut c_void,
            data_ptr_size: n * 4,
        };
        args[e.pos[1]] = BnnsGraphArgument {
            data_ptr: h as *mut c_void,
            data_ptr_size: n * 4,
        };
        args[e.pos[2]] = BnnsGraphArgument {
            data_ptr: bias as *mut c_void,
            data_ptr_size: e.s * 4,
        };
        unsafe {
            execute(
                e.ctx,
                std::ptr::null(),
                args.len(),
                args.as_mut_ptr(),
                e.ws.len(),
                e.ws.as_mut_ptr() as *mut c_char,
            )
        }
    }
}

/// Load the bucket artifacts for one MODEL: `dir`/`stem` are raw (ptr, len)
/// UTF-8 strings (the Haxe side passes `Bytes.ofString(..)` addresses),
/// `hidden` is the model's embedding width from its metadata, and `kind`
/// picks the backend (0 = BNNSGraph CPU, 1 = CoreML CPU+ANE). Returns a
/// handle > 0 on success, 0 when no artifacts were found, -1 off-macOS.
#[no_mangle]
pub unsafe extern "C" fn rayzor_bert_graph_load(
    dir_ptr: i64,
    dir_len: i64,
    stem_ptr: i64,
    stem_len: i64,
    hidden: i64,
    kind: i64,
) -> i64 {
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("RZT_DBG_GRAPH").is_some() {
            eprintln!(
                "[bert-graph] load args: dir_ptr={dir_ptr:#x} dir_len={dir_len} stem_ptr={stem_ptr:#x} stem_len={stem_len} hidden={hidden} kind={kind}"
            );
        }
        if dir_ptr == 0 || dir_len <= 0 || stem_ptr == 0 || stem_len <= 0 || hidden <= 0 {
            return 0;
        }
        let d = std::slice::from_raw_parts(dir_ptr as *const u8, dir_len as usize);
        let s = std::slice::from_raw_parts(stem_ptr as *const u8, stem_len as usize);
        let (Ok(dir), Ok(stem)) = (std::str::from_utf8(d), std::str::from_utf8(s)) else {
            return 0;
        };
        imp::load(dir, stem, hidden as usize, kind)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (dir_ptr, dir_len, stem_ptr, stem_len, hidden, kind);
        -1
    }
}

/// Smallest loaded bucket of `handle` that fits `seq`, or 0 when the graph
/// engine can't take this sequence.
#[no_mangle]
pub extern "C" fn rayzor_bert_graph_bucket(handle: i64, seq: i64) -> i64 {
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

/// Run the fused encoder for bucket `s` of `handle`: h[s*hidden] + bias[s]
/// -> out[s*hidden], all f32. Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn rayzor_bert_graph_execute(
    handle: i64,
    s: i64,
    h_ptr: i64,
    bias_ptr: i64,
    out_ptr: i64,
) -> i64 {
    #[cfg(target_os = "macos")]
    {
        if h_ptr == 0 || bias_ptr == 0 || out_ptr == 0 {
            return -2;
        }
        imp::run(
            handle,
            s as usize,
            h_ptr as *const f32,
            bias_ptr as *const f32,
            out_ptr as *mut f32,
        ) as i64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (handle, s, h_ptr, bias_ptr, out_ptr);
        -1
    }
}
