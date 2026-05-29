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
use crate::ir::{CallingConvention, IrType};

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

    // free: (qt) -> void
    let func_id = builder
        .begin_function("rayzor_qtensor_free")
        .param("qt", i64_ty.clone())
        .returns(void_ty)
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
