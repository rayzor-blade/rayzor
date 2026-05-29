/// Tensor MIR wrappers (rayzor.ds.Tensor)
///
/// Tensor is an extern class — an opaque i64 pointer to a heap-allocated
/// RayzorTensor struct. All methods delegate to runtime extern functions.
///
/// The key complexity is that Haxe `Array<Int>` parameters (for shapes/indices)
/// need to be decomposed into (data_ptr, len) pairs for the runtime.
/// HaxeArray layout: [ptr: *mut u8, len: usize, cap: usize, elem_size: usize]
/// So: data_ptr = load(array_ptr + 0), len = load(array_ptr + 8)
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{BinaryOp, CallingConvention, IrType};

/// Build all tensor type functions
pub fn build_tensor_types(builder: &mut MirBuilder) {
    // Declare all extern runtime functions
    declare_tensor_externs(builder);

    // Build MIR wrappers
    build_tensor_zeros(builder);
    build_tensor_ones(builder);
    build_tensor_full(builder);
    build_tensor_from_array(builder);
    build_tensor_from_bytes_f16(builder);
    build_tensor_from_bytes_f32(builder);
    build_tensor_from_bytes_q8_0(builder);
    build_tensor_rand(builder);

    // Properties
    build_tensor_shape(builder);
    build_tensor_ndim(builder);
    build_tensor_numel(builder);
    build_tensor_dtype(builder);
    build_tensor_device(builder);
    build_tensor_numa_node(builder);

    // Element access
    build_tensor_get(builder);
    build_tensor_set(builder);

    // Reshape / transpose / permute / slice
    build_tensor_reshape(builder);
    build_tensor_transpose(builder);
    build_tensor_permute(builder);
    build_tensor_slice(builder);

    // Arithmetic (binary)
    build_tensor_add(builder);
    build_tensor_sub(builder);
    build_tensor_mul(builder);
    build_tensor_div(builder);

    // Math (unary) / activations
    build_tensor_sqrt(builder);
    build_tensor_exp(builder);
    build_tensor_log(builder);
    build_tensor_relu(builder);
    build_tensor_gelu(builder);
    build_tensor_silu(builder);
    build_tensor_softmax(builder);

    // Normalization
    build_tensor_layer_norm(builder);
    build_tensor_rms_norm(builder);

    // Rotary position embedding (RoPE)
    build_tensor_rope(builder);
    build_tensor_rope_cos_table(builder);
    build_tensor_rope_sin_table(builder);
    build_tensor_rope_cos_table_f16(builder);
    build_tensor_rope_sin_table_f16(builder);

    // Reductions
    build_tensor_sum(builder);
    build_tensor_mean(builder);
    build_tensor_max(builder);
    build_tensor_min(builder);
    build_tensor_dot(builder);

    // Linear algebra
    build_tensor_matmul(builder);
    build_tensor_matmul_t(builder);
    build_tensor_bmm(builder);

    // Attention building blocks (composed by nue.transformer in Haxe)
    build_tensor_causal_mask(builder);
    build_tensor_scale(builder);
    build_tensor_transpose_last2(builder);
    build_tensor_gather_rows(builder);

    // Interop
    build_tensor_data(builder);
    build_tensor_free(builder);
}

// ============================================================================
// Extern declarations
// ============================================================================

fn declare_tensor_externs(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;
    let void_ty = IrType::Void;

    // Construction: (shape_ptr: i64, ndim: i64, dtype: i64) -> i64
    for name in &[
        "rayzor_tensor_zeros",
        "rayzor_tensor_ones",
        "rayzor_tensor_rand",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("shape_ptr", i64_ty.clone())
            .param("ndim", i64_ty.clone())
            .param("dtype", i64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // full: (shape_ptr, ndim, value, dtype) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_full")
        .param("shape_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .param("value", f64_ty.clone())
        .param("dtype", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // from_array: (data_ptr, data_len, dtype) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_from_array")
        .param("data_ptr", i64_ty.clone())
        .param("data_len", i64_ty.clone())
        .param("dtype", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // from_bytes_f16: (bytes_handle, shape_ptr, ndim) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_from_bytes_f16")
        .param("bytes_handle", i64_ty.clone())
        .param("shape_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // from_bytes_f32: (bytes_handle, shape_ptr, ndim) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_from_bytes_f32")
        .param("bytes_handle", i64_ty.clone())
        .param("shape_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // from_bytes_q8_0: (bytes_handle, shape_ptr, ndim) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_from_bytes_q8_0")
        .param("bytes_handle", i64_ty.clone())
        .param("shape_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // shape: (tensor: i64) -> i64 (returns HaxeArray pointer)
    let func_id = builder
        .begin_function("rayzor_tensor_shape")
        .param("tensor", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // Properties: (tensor: i64) -> i64
    for name in &[
        "rayzor_tensor_ndim",
        "rayzor_tensor_numel",
        "rayzor_tensor_dtype",
        "rayzor_tensor_device",
        "rayzor_tensor_numa_node",
        "rayzor_tensor_shape_ptr",
        "rayzor_tensor_shape_ndim",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("tensor", i64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // get: (tensor, indices_ptr, ndim) -> f64
    let func_id = builder
        .begin_function("rayzor_tensor_get")
        .param("tensor", i64_ty.clone())
        .param("indices_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // set: (tensor, indices_ptr, ndim, value) -> void
    let func_id = builder
        .begin_function("rayzor_tensor_set")
        .param("tensor", i64_ty.clone())
        .param("indices_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .param("value", f64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // reshape: (tensor, shape_ptr, ndim) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_reshape")
        .param("tensor", i64_ty.clone())
        .param("shape_ptr", i64_ty.clone())
        .param("ndim", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // transpose: (tensor) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_transpose")
        .param("tensor", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // permute: (tensor, axes_ptr, axes_len) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_permute")
        .param("tensor", i64_ty.clone())
        .param("axes_ptr", i64_ty.clone())
        .param("axes_len", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // slice: (tensor, dim, start, end) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_slice")
        .param("tensor", i64_ty.clone())
        .param("dim", i64_ty.clone())
        .param("start", i64_ty.clone())
        .param("end", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // Binary ops: (a, b) -> i64
    for name in &[
        "rayzor_tensor_add",
        "rayzor_tensor_sub",
        "rayzor_tensor_mul",
        "rayzor_tensor_div",
        "rayzor_tensor_matmul",
        "rayzor_tensor_matmul_t",
        "rayzor_tensor_bmm",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("a", i64_ty.clone())
            .param("b", i64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // causal_mask_: (tensor, position_offset) -> i64 (returns same ptr)
    let func_id = builder
        .begin_function("rayzor_tensor_causal_mask_")
        .param("tensor", i64_ty.clone())
        .param("position_offset", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // scale: (tensor, factor: f64) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_scale")
        .param("tensor", i64_ty.clone())
        .param("factor", f64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // transpose_last2: (tensor) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_transpose_last2")
        .param("tensor", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // gather_rows: (table, indices_data_ptr, indices_len) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_gather_rows")
        .param("table", i64_ty.clone())
        .param("indices_ptr", i64_ty.clone())
        .param("indices_len", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // Unary ops: (tensor) -> i64
    for name in &[
        "rayzor_tensor_sqrt",
        "rayzor_tensor_exp",
        "rayzor_tensor_log",
        "rayzor_tensor_relu",
        "rayzor_tensor_gelu",
        "rayzor_tensor_silu",
        "rayzor_tensor_softmax",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("tensor", i64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // Normalization ops: (tensor, eps: f64) -> i64
    for name in &["rayzor_tensor_layer_norm", "rayzor_tensor_rms_norm"] {
        let func_id = builder
            .begin_function(*name)
            .param("tensor", i64_ty.clone())
            .param("eps", f64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // rope: (x, cos, sin, position_offset) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_rope")
        .param("x", i64_ty.clone())
        .param("cos", i64_ty.clone())
        .param("sin", i64_ty.clone())
        .param("position_offset", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rope cos/sin tables: (head_dim, max_seq_len, base: f64) -> i64
    for name in &[
        "rayzor_tensor_rope_cos_table",
        "rayzor_tensor_rope_sin_table",
        "rayzor_tensor_rope_cos_table_f16",
        "rayzor_tensor_rope_sin_table_f16",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("head_dim", i64_ty.clone())
            .param("max_seq_len", i64_ty.clone())
            .param("base", f64_ty.clone())
            .returns(i64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // Reductions: (tensor) -> f64
    for name in &[
        "rayzor_tensor_sum",
        "rayzor_tensor_mean",
        "rayzor_tensor_max",
        "rayzor_tensor_min",
    ] {
        let func_id = builder
            .begin_function(*name)
            .param("tensor", i64_ty.clone())
            .returns(f64_ty.clone())
            .calling_convention(CallingConvention::C)
            .build();
        builder.mark_as_extern(func_id);
    }

    // dot: (a, b) -> f64
    let func_id = builder
        .begin_function("rayzor_tensor_dot")
        .param("a", i64_ty.clone())
        .param("b", i64_ty.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // data: (tensor) -> i64
    let func_id = builder
        .begin_function("rayzor_tensor_data")
        .param("tensor", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // free: (tensor) -> void
    let func_id = builder
        .begin_function("rayzor_tensor_free")
        .param("tensor", i64_ty.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

// ============================================================================
// Helper: extract (data_ptr, len) from a HaxeArray pointer
// ============================================================================

/// Given a HaxeArray pointer (i64), extract the data pointer and length.
/// HaxeArray layout: { ptr: *mut u8 (offset 0), len: usize (offset 8), ... }
fn extract_array_ptr_len(
    builder: &mut MirBuilder,
    arr: crate::ir::IrId,
) -> (crate::ir::IrId, crate::ir::IrId) {
    let i64_ty = IrType::I64;

    // data_ptr = load i64 from arr + 0
    let data_ptr = builder.load(arr, i64_ty.clone());

    // len_addr = arr + 8
    let eight = builder.const_i64(8);
    let len_addr = builder.bin_op(BinaryOp::Add, arr, eight);

    // len = load i64 from len_addr
    let len = builder.load(len_addr, i64_ty);

    (data_ptr, len)
}

// ============================================================================
// Construction wrappers
// ============================================================================

/// Tensor_zeros(shape_arr: i64, dtype: i64) -> i64
/// shape_arr is a HaxeArray pointer
fn build_tensor_zeros(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_zeros")
        .param("shape_arr", i64_ty.clone())
        .param("dtype", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let shape_arr = builder.get_param(0);
    let dtype = builder.get_param(1);
    let (data_ptr, len) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_zeros")
        .expect("rayzor_tensor_zeros not found");
    let result = builder.call(extern_id, vec![data_ptr, len, dtype]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_ones(shape_arr: i64, dtype: i64) -> i64
fn build_tensor_ones(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_ones")
        .param("shape_arr", i64_ty.clone())
        .param("dtype", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let shape_arr = builder.get_param(0);
    let dtype = builder.get_param(1);
    let (data_ptr, len) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_ones")
        .expect("rayzor_tensor_ones not found");
    let result = builder.call(extern_id, vec![data_ptr, len, dtype]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_full(shape_arr: i64, value: f64, dtype: i64) -> i64
fn build_tensor_full(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("Tensor_full")
        .param("shape_arr", i64_ty.clone())
        .param("value", f64_ty)
        .param("dtype", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let shape_arr = builder.get_param(0);
    let value = builder.get_param(1);
    let dtype = builder.get_param(2);
    let (data_ptr, len) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_full")
        .expect("rayzor_tensor_full not found");
    let result = builder
        .call(extern_id, vec![data_ptr, len, value, dtype])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_fromArray(data_arr: i64, dtype: i64) -> i64
fn build_tensor_from_array(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_fromArray")
        .param("data_arr", i64_ty.clone())
        .param("dtype", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let data_arr = builder.get_param(0);
    let dtype = builder.get_param(1);
    let (data_ptr, data_len) = extract_array_ptr_len(builder, data_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_from_array")
        .expect("rayzor_tensor_from_array not found");
    let result = builder
        .call(extern_id, vec![data_ptr, data_len, dtype])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_fromBytesF16(bytes: Bytes, shape: Array<Int>) -> Tensor
///
/// Pulls (data_ptr, len) out of the shape Array and forwards the Bytes
/// handle straight through (the runtime dereferences it as a HaxeBytes
/// struct).
fn build_tensor_from_bytes_f16(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_fromBytesF16")
        .param("bytes", i64_ty.clone())
        .param("shape_arr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let shape_arr = builder.get_param(1);
    let (shape_ptr, ndim) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_from_bytes_f16")
        .expect("rayzor_tensor_from_bytes_f16 not found");
    let result = builder
        .call(extern_id, vec![bytes, shape_ptr, ndim])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_fromBytesF32(bytes: Bytes, shape: Array<Int>) -> Tensor
///
/// Memcpy F32 bytes straight into a fresh F32 tensor. Bypasses the
/// Tensor.fromArray(Array<Float>, DType.F32) round-trip that loses
/// precision crossing the array_push i64 wrapper.
fn build_tensor_from_bytes_f32(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_fromBytesF32")
        .param("bytes", i64_ty.clone())
        .param("shape_arr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let shape_arr = builder.get_param(1);
    let (shape_ptr, ndim) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_from_bytes_f32")
        .expect("rayzor_tensor_from_bytes_f32 not found");
    let result = builder
        .call(extern_id, vec![bytes, shape_ptr, ndim])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_fromBytesQ8_0(bytes: Bytes, shape: Array<Int>) -> Tensor
fn build_tensor_from_bytes_q8_0(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_fromBytesQ8_0")
        .param("bytes", i64_ty.clone())
        .param("shape_arr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let bytes = builder.get_param(0);
    let shape_arr = builder.get_param(1);
    let (shape_ptr, ndim) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_from_bytes_q8_0")
        .expect("rayzor_tensor_from_bytes_q8_0 not found");
    let result = builder
        .call(extern_id, vec![bytes, shape_ptr, ndim])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_rand(shape_arr: i64, dtype: i64) -> i64
fn build_tensor_rand(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_rand")
        .param("shape_arr", i64_ty.clone())
        .param("dtype", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let shape_arr = builder.get_param(0);
    let dtype = builder.get_param(1);
    let (data_ptr, len) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_rand")
        .expect("rayzor_tensor_rand not found");
    let result = builder.call(extern_id, vec![data_ptr, len, dtype]).unwrap();
    builder.ret(Some(result));
}

// ============================================================================
// Property wrappers (simple pass-through: tensor_ptr -> runtime)
// ============================================================================

macro_rules! build_simple_i64_to_i64 {
    ($fn_name:ident, $mir_name:expr, $extern_name:expr) => {
        fn $fn_name(builder: &mut MirBuilder) {
            let i64_ty = IrType::I64;

            let func_id = builder
                .begin_function($mir_name)
                .param("self", i64_ty.clone())
                .returns(i64_ty)
                .calling_convention(CallingConvention::C)
                .build();

            builder.set_current_function(func_id);
            let entry = builder.create_block("entry");
            builder.set_insert_point(entry);

            let self_val = builder.get_param(0);
            let extern_id = builder
                .get_function_by_name($extern_name)
                .expect(concat!($extern_name, " not found"));
            let result = builder.call(extern_id, vec![self_val]).unwrap();
            builder.ret(Some(result));
        }
    };
}

macro_rules! build_simple_i64_to_f64 {
    ($fn_name:ident, $mir_name:expr, $extern_name:expr) => {
        fn $fn_name(builder: &mut MirBuilder) {
            let i64_ty = IrType::I64;
            let f64_ty = IrType::F64;

            let func_id = builder
                .begin_function($mir_name)
                .param("self", i64_ty)
                .returns(f64_ty)
                .calling_convention(CallingConvention::C)
                .build();

            builder.set_current_function(func_id);
            let entry = builder.create_block("entry");
            builder.set_insert_point(entry);

            let self_val = builder.get_param(0);
            let extern_id = builder
                .get_function_by_name($extern_name)
                .expect(concat!($extern_name, " not found"));
            let result = builder.call(extern_id, vec![self_val]).unwrap();
            builder.ret(Some(result));
        }
    };
}

macro_rules! build_simple_i64_to_void {
    ($fn_name:ident, $mir_name:expr, $extern_name:expr) => {
        fn $fn_name(builder: &mut MirBuilder) {
            let i64_ty = IrType::I64;
            let void_ty = IrType::Void;

            let func_id = builder
                .begin_function($mir_name)
                .param("self", i64_ty)
                .returns(void_ty)
                .calling_convention(CallingConvention::C)
                .build();

            builder.set_current_function(func_id);
            let entry = builder.create_block("entry");
            builder.set_insert_point(entry);

            let self_val = builder.get_param(0);
            let extern_id = builder
                .get_function_by_name($extern_name)
                .expect(concat!($extern_name, " not found"));
            builder.call(extern_id, vec![self_val]);
            builder.ret(None);
        }
    };
}

macro_rules! build_binop_i64 {
    ($fn_name:ident, $mir_name:expr, $extern_name:expr) => {
        fn $fn_name(builder: &mut MirBuilder) {
            let i64_ty = IrType::I64;

            let func_id = builder
                .begin_function($mir_name)
                .param("self", i64_ty.clone())
                .param("other", i64_ty.clone())
                .returns(i64_ty)
                .calling_convention(CallingConvention::C)
                .build();

            builder.set_current_function(func_id);
            let entry = builder.create_block("entry");
            builder.set_insert_point(entry);

            let self_val = builder.get_param(0);
            let other = builder.get_param(1);
            let extern_id = builder
                .get_function_by_name($extern_name)
                .expect(concat!($extern_name, " not found"));
            let result = builder.call(extern_id, vec![self_val, other]).unwrap();
            builder.ret(Some(result));
        }
    };
}

// Properties
build_simple_i64_to_i64!(build_tensor_shape, "Tensor_shape", "rayzor_tensor_shape");
build_simple_i64_to_i64!(build_tensor_ndim, "Tensor_ndim", "rayzor_tensor_ndim");
build_simple_i64_to_i64!(build_tensor_numel, "Tensor_numel", "rayzor_tensor_numel");
build_simple_i64_to_i64!(build_tensor_dtype, "Tensor_dtype", "rayzor_tensor_dtype");
build_simple_i64_to_i64!(build_tensor_device, "Tensor_device", "rayzor_tensor_device");
build_simple_i64_to_i64!(
    build_tensor_numa_node,
    "Tensor_numa_node",
    "rayzor_tensor_numa_node"
);

// Transpose (no extra params)
build_simple_i64_to_i64!(
    build_tensor_transpose,
    "Tensor_transpose",
    "rayzor_tensor_transpose"
);

// Unary math ops / activations
build_simple_i64_to_i64!(build_tensor_sqrt, "Tensor_sqrt", "rayzor_tensor_sqrt");
build_simple_i64_to_i64!(build_tensor_exp, "Tensor_exp", "rayzor_tensor_exp");
build_simple_i64_to_i64!(build_tensor_log, "Tensor_log", "rayzor_tensor_log");
build_simple_i64_to_i64!(build_tensor_relu, "Tensor_relu", "rayzor_tensor_relu");
build_simple_i64_to_i64!(build_tensor_gelu, "Tensor_gelu", "rayzor_tensor_gelu");
build_simple_i64_to_i64!(build_tensor_silu, "Tensor_silu", "rayzor_tensor_silu");
build_simple_i64_to_i64!(
    build_tensor_softmax,
    "Tensor_softmax",
    "rayzor_tensor_softmax"
);

// Reductions
build_simple_i64_to_f64!(build_tensor_sum, "Tensor_sum", "rayzor_tensor_sum");
build_simple_i64_to_f64!(build_tensor_mean, "Tensor_mean", "rayzor_tensor_mean");
build_simple_i64_to_f64!(build_tensor_max, "Tensor_max", "rayzor_tensor_max");
build_simple_i64_to_f64!(build_tensor_min, "Tensor_min", "rayzor_tensor_min");

// Interop
build_simple_i64_to_i64!(build_tensor_data, "Tensor_data", "rayzor_tensor_data");
build_simple_i64_to_void!(build_tensor_free, "Tensor_free", "rayzor_tensor_free");

// Binary ops (tensor, tensor) -> tensor
build_binop_i64!(build_tensor_add, "Tensor_add", "rayzor_tensor_add");
build_binop_i64!(build_tensor_sub, "Tensor_sub", "rayzor_tensor_sub");
build_binop_i64!(build_tensor_mul, "Tensor_mul", "rayzor_tensor_mul");
build_binop_i64!(build_tensor_div, "Tensor_div", "rayzor_tensor_div");
build_binop_i64!(build_tensor_matmul, "Tensor_matmul", "rayzor_tensor_matmul");
build_binop_i64!(
    build_tensor_matmul_t,
    "Tensor_matmulT",
    "rayzor_tensor_matmul_t"
);
build_binop_i64!(build_tensor_bmm, "Tensor_bmm", "rayzor_tensor_bmm");

/// Tensor_causal_mask_(self, position_offset) -> i64
fn build_tensor_causal_mask(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let func_id = builder
        .begin_function("Tensor_causal_mask_")
        .param("self", i64_ty.clone())
        .param("position_offset", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let s = builder.get_param(0);
    let p = builder.get_param(1);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_causal_mask_")
        .expect("rayzor_tensor_causal_mask_ not found");
    let result = builder.call(extern_id, vec![s, p]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_scale(self, factor: f64) -> i64
fn build_tensor_scale(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;
    let func_id = builder
        .begin_function("Tensor_scale")
        .param("self", i64_ty.clone())
        .param("factor", f64_ty)
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let s = builder.get_param(0);
    let f = builder.get_param(1);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_scale")
        .expect("rayzor_tensor_scale not found");
    let result = builder.call(extern_id, vec![s, f]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_transpose_last2(self) -> i64
fn build_tensor_transpose_last2(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let func_id = builder
        .begin_function("Tensor_transpose_last2")
        .param("self", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let s = builder.get_param(0);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_transpose_last2")
        .expect("rayzor_tensor_transpose_last2 not found");
    let result = builder.call(extern_id, vec![s]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_gather_rows(self, indices_arr) -> i64
/// Unpacks the HaxeArray<Int> into (ptr, len) and calls the runtime.
fn build_tensor_gather_rows(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let func_id = builder
        .begin_function("Tensor_gather_rows")
        .param("self", i64_ty.clone())
        .param("indices_arr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let self_val = builder.get_param(0);
    let indices_arr = builder.get_param(1);
    let (data_ptr, len) = extract_array_ptr_len(builder, indices_arr);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_gather_rows")
        .expect("rayzor_tensor_gather_rows not found");
    let result = builder
        .call(extern_id, vec![self_val, data_ptr, len])
        .unwrap();
    builder.ret(Some(result));
}

// ============================================================================
// Dot product: (tensor, tensor) -> f64
// ============================================================================

fn build_tensor_dot(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("Tensor_dot")
        .param("self", i64_ty.clone())
        .param("other", i64_ty)
        .returns(f64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_dot")
        .expect("rayzor_tensor_dot not found");
    let result = builder.call(extern_id, vec![self_val, other]).unwrap();
    builder.ret(Some(result));
}

// ============================================================================
// Element access with array decomposition
// ============================================================================

/// Tensor_get(tensor: i64, indices_arr: i64) -> f64
fn build_tensor_get(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("Tensor_get")
        .param("self", i64_ty.clone())
        .param("indices_arr", i64_ty)
        .returns(f64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let indices_arr = builder.get_param(1);
    let (indices_ptr, ndim) = extract_array_ptr_len(builder, indices_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_get")
        .expect("rayzor_tensor_get not found");
    let result = builder
        .call(extern_id, vec![self_val, indices_ptr, ndim])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_set(tensor: i64, indices_arr: i64, value: f64) -> void
fn build_tensor_set(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;
    let void_ty = IrType::Void;

    let func_id = builder
        .begin_function("Tensor_set")
        .param("self", i64_ty.clone())
        .param("indices_arr", i64_ty)
        .param("value", f64_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let indices_arr = builder.get_param(1);
    let value = builder.get_param(2);
    let (indices_ptr, ndim) = extract_array_ptr_len(builder, indices_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_set")
        .expect("rayzor_tensor_set not found");
    builder.call(extern_id, vec![self_val, indices_ptr, ndim, value]);
    builder.ret(None);
}

/// Tensor_permute(self: i64, axes_arr: i64) -> i64
fn build_tensor_permute(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_permute")
        .param("self", i64_ty.clone())
        .param("axes_arr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let axes_arr = builder.get_param(1);
    let (axes_ptr, axes_len) = extract_array_ptr_len(builder, axes_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_permute")
        .expect("rayzor_tensor_permute not found");
    let result = builder
        .call(extern_id, vec![self_val, axes_ptr, axes_len])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_slice(self: i64, dim: i64, start: i64, end: i64) -> i64
fn build_tensor_slice(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_slice")
        .param("self", i64_ty.clone())
        .param("dim", i64_ty.clone())
        .param("start", i64_ty.clone())
        .param("end", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let dim = builder.get_param(1);
    let start = builder.get_param(2);
    let end = builder.get_param(3);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_slice")
        .expect("rayzor_tensor_slice not found");
    let result = builder
        .call(extern_id, vec![self_val, dim, start, end])
        .unwrap();
    builder.ret(Some(result));
}

/// Tensor_layer_norm(self: i64, eps: f64) -> i64
fn build_tensor_layer_norm(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("Tensor_layer_norm")
        .param("self", i64_ty.clone())
        .param("eps", f64_ty)
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let eps = builder.get_param(1);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_layer_norm")
        .expect("rayzor_tensor_layer_norm not found");
    let result = builder.call(extern_id, vec![self_val, eps]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_rms_norm(self: i64, eps: f64) -> i64
fn build_tensor_rms_norm(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("Tensor_rms_norm")
        .param("self", i64_ty.clone())
        .param("eps", f64_ty)
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let eps = builder.get_param(1);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_rms_norm")
        .expect("rayzor_tensor_rms_norm not found");
    let result = builder.call(extern_id, vec![self_val, eps]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_rope(self: i64, cos: i64, sin: i64, positionOffset: i64) -> i64
fn build_tensor_rope(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let func_id = builder
        .begin_function("Tensor_rope")
        .param("self", i64_ty.clone())
        .param("cos", i64_ty.clone())
        .param("sin", i64_ty.clone())
        .param("position_offset", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let s = builder.get_param(0);
    let c = builder.get_param(1);
    let sn = builder.get_param(2);
    let p = builder.get_param(3);
    let extern_id = builder
        .get_function_by_name("rayzor_tensor_rope")
        .expect("rayzor_tensor_rope not found");
    let result = builder.call(extern_id, vec![s, c, sn, p]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_rope_cos_table(headDim: i64, maxSeqLen: i64, base: f64) -> i64
fn build_tensor_rope_cos_table(builder: &mut MirBuilder) {
    build_rope_table(
        builder,
        "Tensor_rope_cos_table",
        "rayzor_tensor_rope_cos_table",
    );
}

/// Tensor_rope_sin_table(headDim: i64, maxSeqLen: i64, base: f64) -> i64
fn build_tensor_rope_sin_table(builder: &mut MirBuilder) {
    build_rope_table(
        builder,
        "Tensor_rope_sin_table",
        "rayzor_tensor_rope_sin_table",
    );
}

fn build_tensor_rope_cos_table_f16(builder: &mut MirBuilder) {
    build_rope_table(
        builder,
        "Tensor_rope_cos_table_f16",
        "rayzor_tensor_rope_cos_table_f16",
    );
}

fn build_tensor_rope_sin_table_f16(builder: &mut MirBuilder) {
    build_rope_table(
        builder,
        "Tensor_rope_sin_table_f16",
        "rayzor_tensor_rope_sin_table_f16",
    );
}

fn build_rope_table(builder: &mut MirBuilder, wrapper: &str, extern_name: &str) {
    let i64_ty = IrType::I64;
    let f64_ty = IrType::F64;
    let func_id = builder
        .begin_function(wrapper)
        .param("head_dim", i64_ty.clone())
        .param("max_seq_len", i64_ty.clone())
        .param("base", f64_ty)
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let hd = builder.get_param(0);
    let msl = builder.get_param(1);
    let base = builder.get_param(2);
    let extern_id = builder
        .get_function_by_name(extern_name)
        .unwrap_or_else(|| panic!("{} not found", extern_name));
    let result = builder.call(extern_id, vec![hd, msl, base]).unwrap();
    builder.ret(Some(result));
}

/// Tensor_reshape(tensor: i64, shape_arr: i64) -> i64
fn build_tensor_reshape(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Tensor_reshape")
        .param("self", i64_ty.clone())
        .param("shape_arr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let shape_arr = builder.get_param(1);
    let (shape_ptr, ndim) = extract_array_ptr_len(builder, shape_arr);

    let extern_id = builder
        .get_function_by_name("rayzor_tensor_reshape")
        .expect("rayzor_tensor_reshape not found");
    let result = builder
        .call(extern_id, vec![self_val, shape_ptr, ndim])
        .unwrap();
    builder.ret(Some(result));
}
