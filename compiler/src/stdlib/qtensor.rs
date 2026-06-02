/// QTensor MIR wrappers (rayzor.ds.QTensor)
///
/// QTensor is an extern class wrapping a quantised 2-D matrix. Like Tensor
/// it's an opaque i64 pointer; the runtime owns the underlying storage.
///
/// Methods are thin shims onto the runtime extern functions in
/// `runtime/src/quant.rs`. Where the runtime takes a tensor / Ptr / Bool,
/// the Haxe-side type is automatically lowered to i64 by the existing
/// argument plumbing.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, InlineHint, IrType};

/// Build all QTensor type functions.
pub fn build_qtensor_types(builder: &mut MirBuilder) {
    declare_qtensor_externs(builder);

    build_qtensor_from_float32(builder);
    build_qtensor_wrap_q4_k_m(builder);
    build_qtensor_from_bytes_q4_k_m(builder);
    build_qtensor_from_bytes_q6_k(builder);

    build_qtensor_rows(builder);
    build_qtensor_cols(builder);
    build_qtensor_numel(builder);
    build_qtensor_scheme(builder);

    build_qtensor_dequant(builder);
    build_qtensor_matmul_f32(builder);
    build_qtensor_matmul_xtq(builder);
    build_qtensor_matmul_xtq_chunk(builder);
    build_qtensor_matmul_xtq_threaded(builder);
    build_qtensor_fused_qkv_matmul(builder);
    build_qtensor_gather_rows_q6_k(builder);
    build_qtensor_free(builder);
}

// ============================================================================
// Extern declarations
// ============================================================================

fn declare_qtensor_externs(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let void_ty = IrType::Void;

    // from_f32_int8: (src_ptr: i64, rows: i64, cols: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_qtensor_from_f32_int8")
        .param("src_ptr", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // wrap_q4_k_m: (block_data: i64, rows: i64, cols: i64, take_ownership: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_qtensor_wrap_q4_k_m")
        .param("block_data", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .param("take_ownership", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // from_bytes_q4_k_m: (bytes_handle: i64, rows: i64, cols: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_qtensor_from_bytes_q4_k_m")
        .param("bytes_handle", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // from_bytes_q6_k: (bytes_handle: i64, rows: i64, cols: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_qtensor_from_bytes_q6_k")
        .param("bytes_handle", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // Properties (each takes (qt) -> i64):
    for name in &[
        "rayzor_qtensor_rows",
        "rayzor_qtensor_cols",
        "rayzor_qtensor_numel",
        "rayzor_qtensor_scheme",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("qt", i64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // dequant: (qt) -> i64 (returns Tensor)
    let func_id = builder
        .begin_function("rayzor_qtensor_dequant")
        .param("qt", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // matmul_f32: (qt, b_tensor) -> i64
    let func_id = builder
        .begin_function("rayzor_qtensor_matmul_f32")
        .param("qt", i64_ty.clone())
        .param("b_tensor", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // matmul_qt_t_f32: (x_tensor, qt) -> i64  (y = x @ qt.T)
    let func_id = builder
        .begin_function("rayzor_tensor_matmul_qt_t_f32")
        .param("x_tensor", i64_ty.clone())
        .param("qt", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // matmul_qt_t_f32_chunk: (x_tensor, qt, y_tensor, n_start, n_end) -> i64
    // Fills rows [n_start, n_end) of y in-place. Used by NumaPool's
    // parallelRows dispatch to thread the qmatmul over output features.
    let func_id = builder
        .begin_function("rayzor_tensor_matmul_qt_t_f32_chunk")
        .param("x_tensor", i64_ty.clone())
        .param("qt", i64_ty.clone())
        .param("y_tensor", i64_ty.clone())
        .param("n_start", i64_ty.clone())
        .param("n_end", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // matmul_qt_t_f32_threaded: (x_tensor, qt, threads) -> i64
    // Per-call fork-join via std::thread::scope. `threads = 0` picks
    // a default; this is the practical threading path until
    // NumaPool import cascade is fixed.
    let func_id = builder
        .begin_function("rayzor_tensor_matmul_qt_t_f32_threaded")
        .param("x_tensor", i64_ty.clone())
        .param("qt", i64_ty.clone())
        .param("threads", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // matmul_qkv_qt_t_f32_threaded: fused Q/K/V projection. One pre-
    // quant of X to Q8_K is shared across all three Q4_K_M weights and a
    // single parallel_rows fan-out covers the concatenated [0, q_n+k_n+v_n)
    // row space — replaces 3 sequential matmulXTQThreaded fork-joins.
    //
    // Signature: (x_tensor, q_w, k_w, v_w, threads, out_q, out_k, out_v) -> i64
    // The three `out_*` slots are written ONLY on success (return 0). On
    // any non-zero return the MIR wrapper sees them at their pre-init
    // sentinel (0) and the caller falls back to three sequential calls.
    let func_id = builder
        .begin_function("rayzor_tensor_matmul_qkv_qt_t_f32_threaded")
        .param("x_tensor", i64_ty.clone())
        .param("q_w", i64_ty.clone())
        .param("k_w", i64_ty.clone())
        .param("v_w", i64_ty.clone())
        .param("threads", i64_ty.clone())
        .param("out_q_tensor", i64_ty.clone())
        .param("out_k_tensor", i64_ty.clone())
        .param("out_v_tensor", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // gather_rows_q6_k: (qt, indices_ptr, n_indices) -> i64
    // Row-wise dequant of a Q6_K weight matrix into a fresh f32 Tensor.
    // Used by nue.Embedding to dequant only the rows for the active
    // token IDs instead of materialising the whole `[vocab, hidden]`
    // matrix every forward pass.
    let func_id = builder
        .begin_function("rayzor_tensor_gather_rows_q6_k")
        .param("qt", i64_ty.clone())
        .param("indices_ptr", i64_ty.clone())
        .param("n_indices", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // free: (qt) -> void
    let func_id = builder
        .begin_function("rayzor_qtensor_free")
        .param("qt", i64_ty.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // clone: (src) -> i64
    // Deep-copy entrypoint for @:derive(Clone) on rayzor.ds.QTensor.
    // Safety-net mapping: the synthetic `.clone()` call is normally
    // intercepted in hir_to_mir.rs (lower_derived_clone) before this
    // extern is ever materialised, but it must exist for Dynamic-receiver
    // dispatch paths that route through runtime_mapping.rs.
    let func_id = builder
        .begin_function("rayzor_qtensor_clone")
        .param("src", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // arc_clone: (src) -> i64
    // Atomic-refcount entrypoint for `@:shared` classes (QTensor today).
    // Mirrors `rayzor_tensor_arc_clone`: a Relaxed fetch_add on the
    // wrapper refcount that returns the same pointer. Selected by
    // hir_to_mir.rs (lower_derived_clone) when the class is `@:shared`.
    let func_id = builder
        .begin_function("rayzor_qtensor_arc_clone")
        .param("src", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // deep_clone: (src) -> i64
    // Escape hatch for disjoint storage on QTensor — same body as the
    // legacy `rayzor_qtensor_clone`, kept addressable so `.deepClone()`
    // on the Haxe side can route through runtime_mapping.
    let func_id = builder
        .begin_function("rayzor_qtensor_deep_clone")
        .param("src", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

// ============================================================================
// Construction wrappers
// ============================================================================

/// QTensor_fromFloat32(src: Tensor, scheme: QScheme) -> QTensor
///
/// `src` is a 2-D Tensor pointer; we extract its data pointer + shape and
/// dispatch on scheme to the right runtime quantiser. INT8 path goes through
/// `rayzor_qtensor_from_f32_int8`; Q4_K_M path is reserved for the loader's
/// `wrapQ4KM` so this MIR wrapper currently supports INT8 only and returns 0
/// for other schemes (matches the runtime's behaviour).
fn build_qtensor_from_float32(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_fromFloat32")
        .param("src_tensor", i64_ty.clone())
        .param("scheme", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let src_tensor = builder.get_param(0);
    let _scheme = builder.get_param(1);

    // Pull (data_ptr, rows, cols) out of the source Tensor.
    //
    // RayzorTensor layout (see runtime/src/tensor.rs):
    //   offset 0:  data: *mut u8
    //   offset 8:  shape: *mut usize
    //   offset 16: strides: *mut usize
    //   offset 24: ndim: usize
    //   offset 32: numel: usize
    //   ...
    //
    // For 2-D, shape[0] = rows, shape[1] = cols.
    let data_ptr = builder.load(src_tensor, i64_ty.clone()); // data at offset 0

    let eight = builder.const_i64(8);
    let shape_addr = builder.bin_op(crate::ir::BinaryOp::Add, src_tensor, eight);
    let shape_ptr = builder.load(shape_addr, i64_ty.clone());

    let rows = builder.load(shape_ptr, i64_ty.clone());
    let shape_plus_8 = builder.bin_op(crate::ir::BinaryOp::Add, shape_ptr, eight);
    let cols = builder.load(shape_plus_8, i64_ty.clone());

    // Dispatch — currently only INT8 (scheme == 0) is implemented through
    // the from-f32 entrypoint. Q4_K_M weights come from the loader path.
    let extern_id = builder
        .get_function_by_name("rayzor_qtensor_from_f32_int8")
        .expect("rayzor_qtensor_from_f32_int8 not found");
    let result = builder.call(extern_id, vec![data_ptr, rows, cols]).unwrap();
    builder.ret(Some(result));
}

/// QTensor_fromBytesQ4KM(bytes: Bytes, rows: Int, cols: Int) -> QTensor
///
/// Reads the underlying byte slice out of the Haxe `Bytes` handle, copies
/// it into a freshly malloc'd buffer, and wraps it as a Q4_K_M QTensor with
/// `owns_data=true`. The copy is intentional — see the runtime function's
/// doc-comment for the lifetime reasoning.
fn build_qtensor_from_bytes_q4_k_m(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_fromBytesQ4KM")
        .param("bytes", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let rows = builder.get_param(1);
    let cols = builder.get_param(2);

    let extern_id = builder
        .get_function_by_name("rayzor_qtensor_from_bytes_q4_k_m")
        .expect("rayzor_qtensor_from_bytes_q4_k_m not found");
    let result = builder.call(extern_id, vec![bytes, rows, cols]).unwrap();
    builder.ret(Some(result));
}

/// QTensor_fromBytesQ6K(bytes: Bytes, rows: Int, cols: Int) -> QTensor
///
/// Same as `QTensor_fromBytesQ4KM` but routes to the Q6_K runtime
/// constructor (210-byte super-blocks instead of 144).
fn build_qtensor_from_bytes_q6_k(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_fromBytesQ6K")
        .param("bytes", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let rows = builder.get_param(1);
    let cols = builder.get_param(2);

    let extern_id = builder
        .get_function_by_name("rayzor_qtensor_from_bytes_q6_k")
        .expect("rayzor_qtensor_from_bytes_q6_k not found");
    let result = builder.call(extern_id, vec![bytes, rows, cols]).unwrap();
    builder.ret(Some(result));
}

/// QTensor_wrapQ4KM(blockData: Ptr<Float>, rows: Int, cols: Int, takeOwnership: Bool) -> QTensor
fn build_qtensor_wrap_q4_k_m(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_wrapQ4KM")
        .param("block_data", i64_ty.clone())
        .param("rows", i64_ty.clone())
        .param("cols", i64_ty.clone())
        .param("take_ownership", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let block_data = builder.get_param(0);
    let rows = builder.get_param(1);
    let cols = builder.get_param(2);
    let take_ownership = builder.get_param(3);

    let extern_id = builder
        .get_function_by_name("rayzor_qtensor_wrap_q4_k_m")
        .expect("rayzor_qtensor_wrap_q4_k_m not found");
    let result = builder
        .call(extern_id, vec![block_data, rows, cols, take_ownership])
        .unwrap();
    builder.ret(Some(result));
}

// ============================================================================
// Property wrappers
// ============================================================================

fn build_unary_i64_passthrough(builder: &mut MirBuilder, wrapper_name: &str, extern_name: &str) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function(wrapper_name)
        .param("qt", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let qt = builder.get_param(0);
    let extern_id = builder
        .get_function_by_name(extern_name)
        .unwrap_or_else(|| panic!("{} not found", extern_name));
    let result = builder.call(extern_id, vec![qt]).unwrap();
    builder.ret(Some(result));
}

fn build_qtensor_rows(builder: &mut MirBuilder) {
    build_unary_i64_passthrough(builder, "QTensor_rows", "rayzor_qtensor_rows");
}
fn build_qtensor_cols(builder: &mut MirBuilder) {
    build_unary_i64_passthrough(builder, "QTensor_cols", "rayzor_qtensor_cols");
}
fn build_qtensor_numel(builder: &mut MirBuilder) {
    build_unary_i64_passthrough(builder, "QTensor_numel", "rayzor_qtensor_numel");
}
fn build_qtensor_scheme(builder: &mut MirBuilder) {
    build_unary_i64_passthrough(builder, "QTensor_scheme", "rayzor_qtensor_scheme");
}

fn build_qtensor_dequant(builder: &mut MirBuilder) {
    build_unary_i64_passthrough(builder, "QTensor_dequant", "rayzor_qtensor_dequant");
}

fn build_qtensor_matmul_f32(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_matmulF32")
        .param("qt", i64_ty.clone())
        .param("b", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let qt = builder.get_param(0);
    let b = builder.get_param(1);
    let extern_id = builder
        .get_function_by_name("rayzor_qtensor_matmul_f32")
        .expect("rayzor_qtensor_matmul_f32 not found");
    let result = builder.call(extern_id, vec![qt, b]).unwrap();
    builder.ret(Some(result));
}

/// QTensor_matmulXTQ(qt, x) -> y where y = x @ qt.T.
/// The Haxe-facing receiver is `qt`; the runtime takes (x, qt) order.
fn build_qtensor_matmul_xtq(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_matmulXTQ")
        .param("qt", i64_ty.clone())
        .param("x", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let qt = builder.get_param(0);
    let x = builder.get_param(1);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_matmul_qt_t_f32")
        .expect("rayzor_tensor_matmul_qt_t_f32 not found");
    // Runtime fn order: (x_tensor, qt). Swap from Haxe's (qt, x).
    let result = builder.call(extern_id, vec![x, qt]).unwrap();
    builder.ret(Some(result));
}

/// QTensor_matmulXTQChunk(qt, x, y, n_start, n_end) -> i64
///
/// Fills rows `[n_start, n_end)` of a pre-allocated `y` tensor in
/// place. The Haxe `NumaPool.parallelRows` dispatcher splits the row
/// range across workers and calls this per chunk. Haxe receiver
/// order is `(qt, x, y, n_start, n_end)`; the runtime takes
/// `(x_tensor, qt, y_tensor, n_start, n_end)`.
fn build_qtensor_matmul_xtq_chunk(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_matmulXTQChunk")
        .param("qt", i64_ty.clone())
        .param("x", i64_ty.clone())
        .param("y", i64_ty.clone())
        .param("n_start", i64_ty.clone())
        .param("n_end", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let qt = builder.get_param(0);
    let x = builder.get_param(1);
    let y = builder.get_param(2);
    let n_start = builder.get_param(3);
    let n_end = builder.get_param(4);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_matmul_qt_t_f32_chunk")
        .expect("rayzor_tensor_matmul_qt_t_f32_chunk not found");
    // Runtime fn order: (x_tensor, qt, y_tensor, n_start, n_end).
    let result = builder
        .call(extern_id, vec![x, qt, y, n_start, n_end])
        .unwrap();
    builder.ret(Some(result));
}

/// QTensor_matmulXTQThreaded(qt, x, threads) -> y where y = x @ qt.T.
/// Per-call fork-join over output rows via std::thread::scope in the
/// runtime. `threads = 0` picks an auto default.
fn build_qtensor_matmul_xtq_threaded(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("QTensor_matmulXTQThreaded")
        .param("qt", i64_ty.clone())
        .param("x", i64_ty.clone())
        .param("threads", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let qt = builder.get_param(0);
    let x = builder.get_param(1);
    let threads = builder.get_param(2);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_matmul_qt_t_f32_threaded")
        .expect("rayzor_tensor_matmul_qt_t_f32_threaded not found");
    // Runtime order: (x_tensor, qt, threads). Swap from Haxe receiver order.
    let result = builder.call(extern_id, vec![x, qt, threads]).unwrap();
    builder.ret(Some(result));
}

/// HaxeArray struct size in bytes (ptr + len + cap + elem_size = 4 x 8).
const HAXE_ARRAY_STRUCT_SIZE: i64 = 32;

/// QTensor_fusedQkvMatmul(x, qW, kW, vW, threads) -> Array<Tensor>.
///
/// Static method: bypasses the QTensor receiver because the operation
/// takes three distinct Q4_K_M weight matrices and one F32 activation.
/// Forwards to `rayzor_tensor_matmul_qkv_qt_t_f32_threaded`, which
/// shares a single X→Q8_K pre-quant across all three projections and
/// fans one `parallel_rows` over the concatenated row space.
///
/// On success: returns a fresh heap-allocated 3-element `Array<Tensor>`
/// holding `[Q, K, V]` tensor handles. On any non-zero runtime return
/// (SDOT not available, batch != 1, not all-Q4_K_M, X non-contiguous):
/// returns the same array but with all three slots set to `null` (0),
/// so the Haxe caller can check `arr[0] == null` and fall back to three
/// sequential `matmulXTQThreaded` calls.
///
/// The wrapper builds the array with `haxe_array_push_i64` (3x), which
/// handles the zero-cap → first-grow path internally, so no separate
/// `haxe_array_new` extern is needed.
fn build_qtensor_fused_qkv_matmul(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let ptr_i64 = IrType::Ptr(Box::new(IrType::I64));

    let func_id = builder
        .begin_function("QTensor_fusedQkvMatmul")
        .param("x_tensor", i64_ty.clone())
        .param("q_w", i64_ty.clone())
        .param("k_w", i64_ty.clone())
        .param("v_w", i64_ty.clone())
        .param("threads", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let x = builder.get_param(0);
    let q_w = builder.get_param(1);
    let k_w = builder.get_param(2);
    let v_w = builder.get_param(3);
    let threads = builder.get_param(4);

    // Stack-allocate 3 i64 slots for the runtime to write tensor handles
    // into. Pre-init each to 0 so the success-then-load pattern reads a
    // sentinel null on the non-zero (gate-miss) return path. The
    // runtime promises NOT to write the slots on failure, so the zero
    // init survives all the way to the array push.
    let zero = builder.const_i64(0);
    let out_q_slot = builder.alloc(IrType::I64, None);
    builder.store(out_q_slot, zero);
    let out_k_slot = builder.alloc(IrType::I64, None);
    builder.store(out_k_slot, zero);
    let out_v_slot = builder.alloc(IrType::I64, None);
    builder.store(out_v_slot, zero);

    // Extern declares the three out-pointer params as i64 (uniform with
    // the rest of the signature), so cast each Ptr<I64> to i64 before
    // the call. The Cranelift backend treats Ptr↔I64 as a 64-bit
    // bitcast on 64-bit targets, so this is a no-op codegen-wise.
    let out_q_i64 = builder.cast(out_q_slot, ptr_i64.clone(), i64_ty.clone());
    let out_k_i64 = builder.cast(out_k_slot, ptr_i64.clone(), i64_ty.clone());
    let out_v_i64 = builder.cast(out_v_slot, ptr_i64.clone(), i64_ty.clone());

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_matmul_qkv_qt_t_f32_threaded")
        .expect("rayzor_tensor_matmul_qkv_qt_t_f32_threaded not found");
    // Runtime order: (x, q_w, k_w, v_w, threads, out_q, out_k, out_v).
    // The i64 return value (success=0, non-zero=gate-miss) is ignored
    // here: gate-miss leaves the slots at their zero pre-init, so the
    // resulting array carries (null, null, null) and the Haxe caller
    // detects that via `arr[0] == null` to trigger the fallback path.
    let _ = builder.call(
        extern_id,
        vec![x, q_w, k_w, v_w, threads, out_q_i64, out_k_i64, out_v_i64],
    );

    // Load the (possibly-written, possibly-zero) tensor handles back.
    let out_q_val = builder.load(out_q_slot, i64_ty.clone());
    let out_k_val = builder.load(out_k_slot, i64_ty.clone());
    let out_v_val = builder.load(out_v_slot, i64_ty.clone());

    // Allocate a fresh 32-byte HaxeArray struct on the heap. Zero-init
    // the four fields (ptr=null, len=0, cap=0, elem_size=8); the
    // matching `haxe_array_push_i64` calls below grow cap from 0 on
    // first push, so no separate `haxe_array_new` is needed.
    let malloc_func = builder
        .get_function_by_name("malloc")
        .expect("malloc extern not found");
    let arr_size = builder.const_i64(HAXE_ARRAY_STRUCT_SIZE);
    let arr_ptr = builder
        .call(malloc_func, vec![arr_size])
        .expect("malloc should return a pointer");

    // Zero-init ptr/len/cap (offsets 0/8/16) and set elem_size=8 at
    // offset 24. Each store uses ptr_add with byte offsets cast to
    // a Ptr<I64> target so the 8-byte store is well-typed.
    let off0 = builder.const_i64(0);
    let off8 = builder.const_i64(8);
    let off16 = builder.const_i64(16);
    let off24 = builder.const_i64(24);
    let eight = builder.const_i64(8);

    let f_ptr = builder.ptr_add(arr_ptr, off0, ptr_i64.clone());
    builder.store(f_ptr, zero);
    let f_len = builder.ptr_add(arr_ptr, off8, ptr_i64.clone());
    builder.store(f_len, zero);
    let f_cap = builder.ptr_add(arr_ptr, off16, ptr_i64.clone());
    builder.store(f_cap, zero);
    let f_es = builder.ptr_add(arr_ptr, off24, ptr_i64.clone());
    builder.store(f_es, eight);

    // Push Q, K, V handles in order. `haxe_array_push_i64` allocates
    // the underlying element buffer on the first push (cap goes 0 → 8)
    // and copies the i64 by value. All three slots end up holding the
    // raw Tensor* handle (or 0 on gate-miss).
    let push_func = builder
        .get_function_by_name("haxe_array_push_i64")
        .expect("haxe_array_push_i64 extern not found");
    // haxe_array_push_i64 takes (arr: Ptr<Void>, val: I64). Cast
    // arr_ptr (raw Ptr<U8> from malloc) to Ptr<Void> to satisfy the
    // declared signature.
    let arr_void = builder.cast(arr_ptr, IrType::Ptr(Box::new(IrType::U8)), ptr_void.clone());
    let _ = builder.call(push_func, vec![arr_void, out_q_val]);
    let _ = builder.call(push_func, vec![arr_void, out_k_val]);
    let _ = builder.call(push_func, vec![arr_void, out_v_val]);

    builder.ret(Some(arr_void));
}

/// QTensor_gatherRowsQ6K(self, indices_arr) -> i64
///
/// Unpacks the `HaxeArray<Int>` `indices` into (data_ptr, len) and forwards
/// to `rayzor_tensor_gather_rows_q6_k`. The runtime checks that `self`'s
/// scheme is Q6_K and that `cols` is a whole number of 256-element super-
/// blocks; returns 0 on mismatch.
///
/// Mirrors `Tensor_gather_rows` (tensor.rs) for the F32 case; the only
/// difference is the receiver is a Q6_K QTensor.
fn build_qtensor_gather_rows_q6_k(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let func_id = builder
        .begin_function("QTensor_gatherRowsQ6K")
        .param("self", i64_ty.clone())
        .param("indices_arr", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let indices_arr = builder.get_param(1);

    // HaxeArray layout: { ptr (offset 0), len (offset 8), ... }
    // Inline the same unpack used by tensor.rs::extract_array_ptr_len
    // (that helper is private to tensor.rs).
    let data_ptr = builder.load(indices_arr, i64_ty.clone());
    let eight = builder.const_i64(8);
    let len_addr = builder.bin_op(crate::ir::BinaryOp::Add, indices_arr, eight);
    let len = builder.load(len_addr, i64_ty.clone());

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_gather_rows_q6_k")
        .expect("rayzor_tensor_gather_rows_q6_k not found");
    let result = builder
        .call(extern_id, vec![self_val, data_ptr, len])
        .unwrap();
    builder.ret(Some(result));
}

fn build_qtensor_free(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let void_ty = IrType::Void;

    let func_id = builder
        .begin_function("QTensor_free")
        .param("qt", i64_ty.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let qt = builder.get_param(0);
    let extern_id = builder
        .get_function_by_name("rayzor_qtensor_free")
        .expect("rayzor_qtensor_free not found");
    let _ = builder.call(extern_id, vec![qt]);
    builder.ret(None);
}
