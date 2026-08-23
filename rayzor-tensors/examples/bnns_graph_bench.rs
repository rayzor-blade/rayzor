//! BNNSGraph probe: execute a one-op f16 GEMM compiled from CoreML (mlmodelc)
//! and race it against the BNNSMatMul path on identical data.
//!
//! BNNSGraph (macOS 15+) is the successor to the deprecated BNNS entry points:
//! it consumes a COMPILED COREML MODEL (no in-memory builder), so the pipeline
//! is author-.mlpackage -> `xcrun coremlc compile` -> BNNSGraphCompileFromFile.
//! This probe answers whether that engine (a) works from Rust, (b) matches
//! BNNSMatMul's rate for a single GEMM, before we consider subgraph offload.
//!
//! Run:
//!   cargo run -p rayzor-tensors --example bnns_graph_bench --release -- \
//!     <dir with gemm497.mlmodelc + x.bin/wt.bin/exp.bin>

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bnns_graph_bench is macOS-only.");
}

#[cfg(target_os = "macos")]
fn main() {
    use rayzor_tensors::apple_accel::matmul_f16f32_nt;
    use std::os::raw::{c_char, c_void};
    use std::time::Instant;

    const M: usize = 497;
    const K: usize = 2048;
    const N: usize = 8192;

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
        #[link_name = "BNNSGraphContextSetMessageLogCallback"]
        fn set_log_callback(
            context: BnnsGraphContext,
            cb: extern "C" fn(u32, *const c_char, *const c_char, *mut c_void),
            user: *mut c_void,
        ) -> i32;
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

    let dir = std::env::args()
        .nth(1)
        .expect("usage: bnns_graph_bench <dir>");
    let read = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{dir}/{name}")).unwrap_or_else(|e| panic!("read {name}: {e}"))
    };
    let as_u16 = |b: &[u8]| -> Vec<u16> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect()
    };
    let as_f32 = |b: &[u8]| -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect()
    };

    let x16 = as_u16(&read("x.bin"));
    let x32 = as_f32(&read("x32.bin")); // model I/O is FLOAT32 (fp16 internal)
    let wt16 = as_u16(&read("wt.bin"));
    let expected = as_f32(&read("exp.bin"));
    assert_eq!(x16.len(), M * K);
    assert_eq!(wt16.len(), N * K);
    assert_eq!(expected.len(), M * N);

    // --- compile + context ---
    let path = std::ffi::CString::new(format!("{dir}/gemm497.mlmodelc")).unwrap();
    let null_opts = BnnsGraphCompileOptions {
        data: std::ptr::null_mut(),
        size: 0,
    };
    let graph = unsafe { compile_from_file(path.as_ptr(), std::ptr::null(), null_opts) };
    assert!(!graph.data.is_null(), "BNNSGraphCompileFromFile failed");
    println!("graph compiled: {} bytes", graph.size);

    let ctx = unsafe { context_make(graph) };
    assert!(!ctx.data.is_null(), "BNNSGraphContextMake failed");
    let ws_size = unsafe { workspace_size(ctx, std::ptr::null()) };
    println!("workspace: {ws_size} bytes");
    let mut ws = vec![0u8; ws_size.max(1)];

    // --- execute: outputs precede inputs ---
    let mut out32 = vec![0f32; M * N];
    let run = |out32: &mut [f32], ws: &mut [u8]| -> i32 {
        let mut args = [
            BnnsGraphArgument {
                data_ptr: out32.as_mut_ptr() as *mut c_void,
                data_ptr_size: std::mem::size_of_val(out32),
            },
            BnnsGraphArgument {
                data_ptr: x32.as_ptr() as *mut c_void,
                data_ptr_size: x32.len() * 4,
            },
        ];
        unsafe {
            execute(
                ctx,
                std::ptr::null(),
                args.len(),
                args.as_mut_ptr(),
                ws.len(),
                ws.as_mut_ptr() as *mut c_char,
            )
        }
    };
    let rc = run(&mut out32, &mut ws);
    assert_eq!(rc, 0, "BNNSGraphContextExecute rc={rc}");

    // --- correctness vs f32 reference (fp16 accumulate tolerance) ---
    let mut max_rel = 0.0f32;
    let mut sum_rel = 0.0f64;
    for i in 0..M * N {
        let got = out32[i];
        let want = expected[i];
        let rel = (got - want).abs() / want.abs().max(1e-3);
        if rel > max_rel {
            max_rel = rel;
        }
        sum_rel += rel as f64;
    }
    let mean_rel = sum_rel / (M * N) as f64;
    // The graph computes fully in fp16 (compute_precision=FLOAT16), so vs an
    // f32-accumulate reference the mean relative error sits at fp16-accumulation
    // level (~1.3% over K=2048); max_rel spikes on near-zero outputs. Whether
    // fp16 accumulation is acceptable for prefill is a MODEL-level quality
    // question, not an ABI one — gate at 2e-2 here.
    println!(
        "correctness: mean_rel={mean_rel:.5} max_rel={max_rel:.4} => {}",
        if mean_rel < 2e-2 {
            "PASS (fp16-accum tolerance)"
        } else {
            "FAIL"
        }
    );

    // --- rate: BNNSGraph vs BNNSMatMul on identical data ---
    let flops = 2.0 * M as f64 * K as f64 * N as f64;
    let iters = 30;
    let t0 = Instant::now();
    for _ in 0..iters {
        let rc = run(&mut out32, &mut ws);
        assert_eq!(rc, 0);
    }
    let graph_s = t0.elapsed().as_secs_f64() / iters as f64;

    let mut out_mm = vec![0f32; M * N];
    let t0 = Instant::now();
    for _ in 0..iters {
        assert!(matmul_f16f32_nt(M, K, N, &x16, &wt16, &mut out_mm));
    }
    let mm_s = t0.elapsed().as_secs_f64() / iters as f64;

    println!(
        "BNNSGraph : {:>8.0} us  {:>7.0} GF/s",
        graph_s * 1e6,
        flops / graph_s / 1e9
    );
    println!(
        "BNNSMatMul: {:>8.0} us  {:>7.0} GF/s",
        mm_s * 1e6,
        flops / mm_s / 1e9
    );

    // --- weights-as-input variant: one 16K artifact per SHAPE, fp16 io ---
    let w32 = as_f32(&read("w32.bin")); // [K,N] f32 (model io is FLOAT32)
    assert_eq!(w32.len(), K * N);
    let path_wi = std::ffi::CString::new(format!("{dir}/gemm_wi.mlmodelc")).unwrap();
    let graph_wi = unsafe { compile_from_file(path_wi.as_ptr(), std::ptr::null(), null_opts) };
    assert!(!graph_wi.data.is_null(), "compile gemm_wi failed");
    let ctx_wi = unsafe { context_make(graph_wi) };
    assert!(!ctx_wi.data.is_null());
    extern "C" fn log_cb(level: u32, msg: *const c_char, op: *const c_char, _u: *mut c_void) {
        let m = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
        let o = if op.is_null() {
            "".into()
        } else {
            unsafe { std::ffi::CStr::from_ptr(op) }.to_string_lossy()
        };
        eprintln!("[bnns-graph lvl={level}] {m} {o}");
    }
    unsafe { set_log_callback(ctx_wi, log_cb, std::ptr::null_mut()) };
    let ws_wi_size = unsafe { workspace_size(ctx_wi, std::ptr::null()) };
    let mut ws_wi = vec![0u8; ws_wi_size.max(1)];
    let mut out_wi = vec![0f32; M * N];
    let run_wi = |out_wi: &mut [f32], ws: &mut [u8]| -> i32 {
        let mut args = [
            BnnsGraphArgument {
                data_ptr: out_wi.as_mut_ptr() as *mut c_void,
                data_ptr_size: std::mem::size_of_val(out_wi),
            },
            // positions are ALPHABETIZED: out=0, w=1, x=2 (not source order)
            BnnsGraphArgument {
                data_ptr: w32.as_ptr() as *mut c_void,
                data_ptr_size: w32.len() * 4,
            },
            BnnsGraphArgument {
                data_ptr: x32.as_ptr() as *mut c_void,
                data_ptr_size: x32.len() * 4,
            },
        ];
        unsafe {
            execute(
                ctx_wi,
                std::ptr::null(),
                args.len(),
                args.as_mut_ptr(),
                ws.len(),
                ws.as_mut_ptr() as *mut c_char,
            )
        }
    };
    for name in ["out", "x", "w"] {
        let c = std::ffi::CString::new(name).unwrap();
        let pos = unsafe { argument_position(graph_wi, std::ptr::null(), c.as_ptr()) };
        println!("arg position {name} = {pos}");
    }
    let rc = run_wi(&mut out_wi, &mut ws_wi);
    assert_eq!(rc, 0, "wi execute rc={rc}");
    let mut sum_rel = 0.0f64;
    for i in 0..M * N {
        let got = out_wi[i];
        let want = expected[i];
        sum_rel += ((got - want).abs() / want.abs().max(1e-3)) as f64;
    }
    println!(
        "wi correctness: mean_rel={:.5} (fp16 io+accum) artifact=16K weights-as-INPUT",
        sum_rel / (M * N) as f64
    );
    let t0 = Instant::now();
    for _ in 0..iters {
        let rc = run_wi(&mut out_wi, &mut ws_wi);
        assert_eq!(rc, 0);
    }
    let wi_s = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "BNNSGraph(w-input): {:>8.0} us  {:>7.0} GF/s",
        wi_s * 1e6,
        flops / wi_s / 1e9
    );
}
