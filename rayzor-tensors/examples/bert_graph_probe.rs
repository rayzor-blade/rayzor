//! BNNSGraph probe #2: execute the WHOLE authored MiniLM encoder (6 post-norm
//! BERT layers, one mlprogram from `bert_graph_author.py`) and check it against
//! the CoreML-predicted reference dumped by the authoring venv.
//!
//! This answers the Phase-4 question the single-GEMM probe couldn't: does
//! BNNSGraph run a REAL multi-op fused encoder from Rust, correctly, and at
//! what rate vs the AMX-f16 per-op path.
//!
//! Run:
//!   cargo run -p rayzor-tensors --example bert_graph_probe --release -- \
//!     <dir with bert_encoder_s128.mlmodelc + probe_{h,bias,out}.bin>

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("bert_graph_probe is macOS-only.");
}

#[cfg(target_os = "macos")]
fn main() {
    use std::os::raw::{c_char, c_void};
    use std::time::Instant;

    const S: usize = 128;
    const H: usize = 384;

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

    // Link the framework EXPLICITLY. Without it the symbols resolve via the
    // flat-namespace dynamic_lookup and Accelerate's C++ statics initialize
    // lazily mid-call → libc++abi typed-operator-new abort (macOS 26). The C
    // reference probe (-framework Accelerate) loads it at process start and
    // is fine — do the same here.
    #[link(name = "Accelerate", kind = "framework")]
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

    let dir = std::env::args()
        .nth(1)
        .expect("usage: bert_graph_probe <dir>");
    let as_f32 = |b: Vec<u8>| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let read_f32 = |name: &str| -> Vec<f32> {
        as_f32(std::fs::read(format!("{dir}/{name}")).unwrap_or_else(|e| panic!("{name}: {e}")))
    };

    let h = read_f32("probe_h.bin");
    let bias = read_f32("probe_bias.bin");
    let expected = read_f32("probe_out.bin");
    assert_eq!(h.len(), S * H);
    assert_eq!(bias.len(), S);
    assert_eq!(expected.len(), S * H);

    let model = std::ffi::CString::new(format!("{dir}/bert_encoder_s{S}.mlmodelc")).unwrap();
    let graph = unsafe {
        compile_from_file(
            model.as_ptr(),
            std::ptr::null(),
            BnnsGraphCompileOptions {
                data: std::ptr::null_mut(),
                size: 0,
            },
        )
    };
    assert!(!graph.data.is_null(), "BNNSGraphCompileFromFile_v2 failed");
    println!("graph compiled: {} bytes", graph.size);

    let ctx = unsafe { context_make(graph) };
    assert!(!ctx.data.is_null(), "context_make failed");
    let ws_len = unsafe { workspace_size(ctx, std::ptr::null()) };
    println!("workspace: {ws_len} bytes");
    let mut ws = vec![0u8; ws_len.max(1)];

    // Bind by NAME — the default argument order is the mlprogram's
    // (alphabetized) signature, so never assume it.
    let mut order: Vec<(usize, &str)> = ["out", "h", "bias"]
        .iter()
        .map(|n| {
            let c = std::ffi::CString::new(*n).unwrap();
            let pos = unsafe { argument_position(graph, std::ptr::null(), c.as_ptr()) };
            (pos, *n)
        })
        .collect();
    order.sort();
    println!("argument order: {order:?}");

    let mut out = vec![0f32; S * H];
    let mut run = |out: &mut [f32], ws: &mut [u8]| -> i32 {
        let mut slot = |name: &str| -> BnnsGraphArgument {
            match name {
                "out" => BnnsGraphArgument {
                    data_ptr: out.as_mut_ptr() as *mut c_void,
                    data_ptr_size: out.len() * 4,
                },
                "h" => BnnsGraphArgument {
                    data_ptr: h.as_ptr() as *mut c_void,
                    data_ptr_size: h.len() * 4,
                },
                "bias" => BnnsGraphArgument {
                    data_ptr: bias.as_ptr() as *mut c_void,
                    data_ptr_size: bias.len() * 4,
                },
                _ => unreachable!(),
            }
        };
        let mut args: Vec<BnnsGraphArgument> = order.iter().map(|(_, n)| slot(n)).collect();
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

    let rc = run(&mut out, &mut ws);
    assert_eq!(rc, 0, "execute rc={rc}");

    let dot: f64 = out
        .iter()
        .zip(&expected)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let na: f64 = out.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = expected
        .iter()
        .map(|a| (*a as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let cos = dot / (na * nb).max(1e-12);
    println!("cosine(BNNSGraph, CoreML-ref) = {cos:.7}");
    assert!(cos > 0.999, "encoder output diverges (cos={cos})");

    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        let rc = run(&mut out, &mut ws);
        assert_eq!(rc, 0);
    }
    let per = t0.elapsed().as_secs_f64() / iters as f64;
    println!(
        "BNNSGraph encoder S={S}: {:>7.0} us/encode  ({:.1} sentences/s single-thread)",
        per * 1e6,
        1.0 / per
    );
}
