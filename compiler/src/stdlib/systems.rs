/// Systems-level type MIR wrappers (Box, Ptr, Ref, Usize)
///
/// These are zero-cost abstracts over Int (i64) at MIR level.
/// Box operations delegate to runtime functions (alloc/free).
/// Ptr/Ref operations are direct load/store/arithmetic MIR instructions.
/// Usize operations are native i64 arithmetic.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, IrType, IrValue, VectorMinMaxKind, VectorUnaryOpKind,
};

/// Build all systems-level type functions
pub fn build_systems_types(builder: &mut MirBuilder) {
    // Declare extern runtime functions for Box
    declare_box_externs(builder);

    // Build Box MIR wrappers
    build_box_init(builder);
    build_box_unbox(builder);
    build_box_raw(builder);
    build_box_free(builder);

    // Build Ptr MIR wrappers (no externs needed — direct MIR ops)
    build_ptr_from_raw(builder);
    build_ptr_raw(builder);
    build_ptr_deref(builder);
    build_ptr_write(builder);
    build_ptr_offset(builder);
    build_ptr_is_null(builder);

    // Build Ref MIR wrappers (no externs needed — direct MIR ops)
    build_ref_from_raw(builder);
    build_ref_raw(builder);
    build_ref_deref(builder);

    // Build CString MIR wrappers (abstract over Int — raw/fromRaw are identity)
    build_cstring_raw(builder);
    build_cstring_from_raw(builder);

    // Build Usize MIR wrappers (no externs needed — native i64 ops)
    build_usize_from_int(builder);
    build_usize_to_int(builder);
    build_usize_add(builder);
    build_usize_sub(builder);
    build_usize_band(builder);
    build_usize_bor(builder);
    build_usize_shl(builder);
    build_usize_shr(builder);
    build_usize_align_up(builder);
    build_usize_is_zero(builder);

    // Build SIMD4f MIR wrappers (no externs needed — native vector MIR ops)
    build_simd4f_splat(builder);
    build_simd4f_make(builder);
    build_simd4f_load(builder);
    build_simd4f_store(builder);
    build_simd4f_extract(builder);
    build_simd4f_insert(builder);
    build_simd4f_sum(builder);
    build_simd4f_dot(builder);
    build_simd4f_from_array(builder);
    // Math operations
    build_simd4f_sqrt(builder);
    build_simd4f_abs(builder);
    build_simd4f_neg(builder);
    build_simd4f_min(builder);
    build_simd4f_max(builder);
    build_simd4f_ceil(builder);
    build_simd4f_floor(builder);
    build_simd4f_round(builder);
    // Compound operations
    build_simd4f_clamp(builder);
    build_simd4f_lerp(builder);
    build_simd4f_length(builder);
    build_simd4f_normalize(builder);
    build_simd4f_cross3(builder);
    build_simd4f_distance(builder);

    // Build sys.io.File MIR wrappers (default binary=true for read/write/append/update)
    declare_file_externs(builder);
    build_file_read_default(builder);
    build_file_write_default(builder);
    build_file_append_default(builder);
    build_file_update_default(builder);
}

// ============================================================================
// Box<T> — extern declarations
// ============================================================================

fn declare_box_externs(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let void_ty = IrType::Void;

    // Box is represented as i64 (opaque pointer) throughout the type system.
    // Use i64 for all params/returns to match the MIR wrappers and avoid
    // LLVM type mismatches (ptr vs i64) during module verification.

    // extern fn rayzor_box_init(value: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_box_init")
        .param("value", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_box_unbox(box_ptr: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_box_unbox")
        .param("box_ptr", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_box_raw(box_ptr: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_box_raw")
        .param("box_ptr", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_box_free(box_ptr: i64) -> void
    let func_id = builder
        .begin_function("rayzor_box_free")
        .param("box_ptr", i64_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

// ============================================================================
// Box<T> — MIR wrappers
// ============================================================================

/// Box_init(value: i64) -> i64
/// Allocates on heap, stores value, returns heap pointer as i64
fn build_box_init(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Box_init")
        .param("value", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let value = builder.get_param(0);
    let extern_id = builder
        .get_function_by_name("rayzor_box_init")
        .expect("rayzor_box_init not found");
    let result = builder.call(extern_id, vec![value]).unwrap();
    builder.ret(Some(result));
}

/// Box_unbox(box: i64) -> i64
/// Reads the value from the heap pointer
fn build_box_unbox(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Box_unbox")
        .param("box_ptr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let box_ptr = builder.get_param(0);
    let extern_id = builder
        .get_function_by_name("rayzor_box_unbox")
        .expect("rayzor_box_unbox not found");
    let result = builder.call(extern_id, vec![box_ptr]).unwrap();
    builder.ret(Some(result));
}

/// Box_raw(box: i64) -> i64
/// Identity — returns the heap address (also used for asPtr/asRef)
fn build_box_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Box_raw")
        .param("box_ptr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Identity: the box pointer IS the raw address
    let box_ptr = builder.get_param(0);
    builder.ret(Some(box_ptr));
}

/// Box_free(box: i64) -> void
/// Deallocates the heap memory
fn build_box_free(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Box_free")
        .param("box_ptr", i64_ty)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let box_ptr = builder.get_param(0);
    let extern_id = builder
        .get_function_by_name("rayzor_box_free")
        .expect("rayzor_box_free not found");
    builder.call(extern_id, vec![box_ptr]);
    builder.ret(None);
}

// ============================================================================
// Ptr<T> — MIR wrappers (direct MIR instructions, no runtime calls)
// ============================================================================

/// Ptr_fromRaw(address: i64) -> i64  — identity
fn build_ptr_from_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ptr_fromRaw")
        .param("address", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let address = builder.get_param(0);
    builder.ret(Some(address));
}

/// Ptr_raw(ptr: i64) -> i64  — identity
fn build_ptr_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ptr_raw")
        .param("ptr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ptr = builder.get_param(0);
    builder.ret(Some(ptr));
}

/// Ptr_deref(ptr: i64) -> i64  — load i64 from address
fn build_ptr_deref(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ptr_deref")
        .param("ptr", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ptr = builder.get_param(0);
    let value = builder.load(ptr, i64_ty);
    builder.ret(Some(value));
}

/// Ptr_write(ptr: i64, value: i64) -> void  — store i64 to address
fn build_ptr_write(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ptr_write")
        .param("ptr", i64_ty.clone())
        .param("value", i64_ty)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ptr = builder.get_param(0);
    let value = builder.get_param(1);
    builder.store(ptr, value);
    builder.ret(None);
}

/// Ptr_offset(ptr: i64, n: i64) -> i64  — ptr + n * 8 (element size is i64 = 8 bytes)
fn build_ptr_offset(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ptr_offset")
        .param("ptr", i64_ty.clone())
        .param("n", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ptr = builder.get_param(0);
    let n = builder.get_param(1);
    // offset = n * 8 (all values are i64 = 8 bytes)
    let eight = builder.const_i64(8);
    let byte_offset = builder.mul(n, eight, i64_ty.clone());
    let result = builder.add(ptr, byte_offset, i64_ty);
    builder.ret(Some(result));
}

/// Ptr_isNull(ptr: i64) -> bool  — ptr == 0
fn build_ptr_is_null(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ptr_isNull")
        .param("ptr", i64_ty)
        .returns(IrType::Bool)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ptr = builder.get_param(0);
    let zero = builder.const_i64(0);
    let is_null = builder.icmp(CompareOp::Eq, ptr, zero, IrType::Bool);
    builder.ret(Some(is_null));
}

// ============================================================================
// Ref<T> — MIR wrappers (same as Ptr but read-only, no write)
// ============================================================================

/// Ref_fromRaw(address: i64) -> i64  — identity
fn build_ref_from_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ref_fromRaw")
        .param("address", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let address = builder.get_param(0);
    builder.ret(Some(address));
}

/// Ref_raw(ref: i64) -> i64  — identity
fn build_ref_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ref_raw")
        .param("ref_ptr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ref_ptr = builder.get_param(0);
    builder.ret(Some(ref_ptr));
}

/// Ref_deref(ref: i64) -> i64  — load i64 from address
fn build_ref_deref(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Ref_deref")
        .param("ref_ptr", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ref_ptr = builder.get_param(0);
    let value = builder.load(ref_ptr, i64_ty);
    builder.ret(Some(value));
}

// ============================================================================
// Usize — MIR wrappers (native i64 arithmetic, all identity/inline)
// ============================================================================

/// Usize_fromInt(value: i64) -> i64  — identity
fn build_usize_from_int(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_fromInt")
        .param("value", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let value = builder.get_param(0);
    builder.ret(Some(value));
}

/// Usize_toInt(self: i64) -> i64  — identity
fn build_usize_to_int(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_toInt")
        .param("self_val", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    builder.ret(Some(self_val));
}

/// Usize_add(self: i64, other: i64) -> i64
fn build_usize_add(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_add")
        .param("self_val", i64_ty.clone())
        .param("other", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let result = builder.add(self_val, other, i64_ty);
    builder.ret(Some(result));
}

/// Usize_sub(self: i64, other: i64) -> i64
fn build_usize_sub(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_sub")
        .param("self_val", i64_ty.clone())
        .param("other", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let result = builder.sub(self_val, other, i64_ty);
    builder.ret(Some(result));
}

/// Usize_band(self: i64, other: i64) -> i64
fn build_usize_band(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_band")
        .param("self_val", i64_ty.clone())
        .param("other", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let result = builder.bin_op(BinaryOp::And, self_val, other);
    builder.ret(Some(result));
}

/// Usize_bor(self: i64, other: i64) -> i64
fn build_usize_bor(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_bor")
        .param("self_val", i64_ty.clone())
        .param("other", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let result = builder.bin_op(BinaryOp::Or, self_val, other);
    builder.ret(Some(result));
}

/// Usize_shl(self: i64, bits: i64) -> i64
fn build_usize_shl(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_shl")
        .param("self_val", i64_ty.clone())
        .param("bits", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let bits = builder.get_param(1);
    let result = builder.bin_op(BinaryOp::Shl, self_val, bits);
    builder.ret(Some(result));
}

/// Usize_shr(self: i64, bits: i64) -> i64  (unsigned/logical shift right)
fn build_usize_shr(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_shr")
        .param("self_val", i64_ty.clone())
        .param("bits", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let bits = builder.get_param(1);
    let result = builder.bin_op(BinaryOp::Shr, self_val, bits);
    builder.ret(Some(result));
}

/// Usize_alignUp(self: i64, alignment: i64) -> i64
/// Computes: (self + alignment - 1) & ~(alignment - 1)
fn build_usize_align_up(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_alignUp")
        .param("self_val", i64_ty.clone())
        .param("alignment", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let alignment = builder.get_param(1);

    // align_mask = alignment - 1
    let one = builder.const_i64(1);
    let align_mask = builder.sub(alignment, one, i64_ty.clone());

    // sum = self + align_mask
    let sum = builder.add(self_val, align_mask, i64_ty.clone());

    // neg_mask = ~align_mask  (XOR with -1)
    let neg_one = builder.const_i64(-1);
    let neg_mask = builder.bin_op(BinaryOp::Xor, align_mask, neg_one);

    // result = sum & neg_mask
    let result = builder.bin_op(BinaryOp::And, sum, neg_mask);
    builder.ret(Some(result));
}

/// Usize_isZero(self: i64) -> bool  — self == 0
fn build_usize_is_zero(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("Usize_isZero")
        .param("self_val", i64_ty)
        .returns(IrType::Bool)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let zero = builder.const_i64(0);
    let is_zero = builder.icmp(CompareOp::Eq, self_val, zero, IrType::Bool);
    builder.ret(Some(is_zero));
}

// ============================================================================
// CString — MIR wrappers (abstract over Int — raw/fromRaw are identity)
// ============================================================================

/// CString_raw(self: i64) -> i64  — identity (CString IS the raw char* address)
fn build_cstring_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("CString_raw")
        .param("self_val", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    builder.ret(Some(self_val));
}

/// CString_fromRaw(addr: i64) -> i64  — identity cast
fn build_cstring_from_raw(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("CString_fromRaw")
        .param("addr", i64_ty.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let addr = builder.get_param(0);
    builder.ret(Some(addr));
}

// ============================================================================
// SIMD4f — 128-bit vector of 4×f32 (native SIMD instructions)
// ============================================================================

/// SIMD4f_splat(scalar: f32) -> vec<f32; 4>
fn build_simd4f_splat(builder: &mut MirBuilder) {
    let f32_ty = IrType::F32;
    let vec_ty = IrType::vector(IrType::F32, 4);

    let func_id = builder
        .begin_function("SIMD4f_splat")
        .param("scalar", f32_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let scalar = builder.get_param(0);
    let result = builder.vector_splat(scalar, vec_ty);
    builder.ret(Some(result));
}

/// SIMD4f_make(x: f32, y: f32, z: f32, w: f32) -> vec<f32; 4>
fn build_simd4f_make(builder: &mut MirBuilder) {
    let f32_ty = IrType::F32;
    let vec_ty = IrType::vector(IrType::F32, 4);

    let func_id = builder
        .begin_function("SIMD4f_make")
        .param("x", f32_ty.clone())
        .param("y", f32_ty.clone())
        .param("z", f32_ty.clone())
        .param("w", f32_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let x = builder.get_param(0);
    let y = builder.get_param(1);
    let z = builder.get_param(2);
    let w = builder.get_param(3);

    // Splat x, then insert y, z, w into lanes 1, 2, 3
    let v0 = builder.vector_splat(x, vec_ty.clone());
    let v1 = builder.vector_insert(v0, y, 1, vec_ty.clone());
    let v2 = builder.vector_insert(v1, z, 2, vec_ty.clone());
    let v3 = builder.vector_insert(v2, w, 3, vec_ty);
    builder.ret(Some(v3));
}

/// SIMD4f_load(ptr: i64) -> vec<f32; 4>
fn build_simd4f_load(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let vec_ty = IrType::vector(IrType::F32, 4);

    let func_id = builder
        .begin_function("SIMD4f_load")
        .param("ptr", i64_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let ptr = builder.get_param(0);
    let result = builder.vector_load(ptr, vec_ty);
    builder.ret(Some(result));
}

/// SIMD4f_store(self: vec<f32; 4>, ptr: i64) -> void
fn build_simd4f_store(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let i64_ty = IrType::I64;

    let func_id = builder
        .begin_function("SIMD4f_store")
        .param("self_val", vec_ty.clone())
        .param("ptr", i64_ty)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let ptr = builder.get_param(1);
    builder.vector_store(ptr, self_val, vec_ty);
    builder.ret(None);
}

/// SIMD4f_extract(self: vec<f32; 4>, lane: i32) -> f32
fn build_simd4f_extract(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("SIMD4f_extract")
        .param("self_val", vec_ty)
        .param("lane", i32_ty)
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let f32_result = builder.vector_extract(self_val, 0, f32_ty.clone());
    let result = builder.cast(f32_result, f32_ty, f64_ty);
    builder.ret(Some(result));
}

/// SIMD4f_insert(self: vec<f32; 4>, lane: i32, value: f32) -> vec<f32; 4>
fn build_simd4f_insert(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("SIMD4f_insert")
        .param("self_val", vec_ty.clone())
        .param("lane", i32_ty)
        .param("value", f32_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let value = builder.get_param(2);
    // Static lane 0 for now (same limitation as extract)
    let result = builder.vector_insert(self_val, value, 0, vec_ty);
    builder.ret(Some(result));
}

/// SIMD4f_sum(self: vec<f32; 4>) -> f32  — horizontal add
fn build_simd4f_sum(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("SIMD4f_sum")
        .param("self_val", vec_ty)
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let f32_result = builder.vector_reduce(BinaryOp::Add, self_val, f32_ty.clone());
    let result = builder.cast(f32_result, f32_ty, f64_ty);
    builder.ret(Some(result));
}

/// SIMD4f_dot(self: vec<f32; 4>, other: vec<f32; 4>) -> f32
fn build_simd4f_dot(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("SIMD4f_dot")
        .param("self_val", vec_ty.clone())
        .param("other", vec_ty.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let product = builder.vector_bin_op(BinaryOp::Mul, self_val, other, vec_ty);
    let f32_result = builder.vector_reduce(BinaryOp::Add, product, f32_ty.clone());
    let result = builder.cast(f32_result, f32_ty, f64_ty);
    builder.ret(Some(result));
}

/// SIMD4f_fromArray(arr: PtrVoid) -> vec<f32; 4>  — @:from Array<Float>
fn build_simd4f_from_array(builder: &mut MirBuilder) {
    let ptr_void_ty = IrType::Ptr(Box::new(IrType::Void));
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("SIMD4f_fromArray")
        .param("arr", ptr_void_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Look up haxe_array_get_f64 (declared in array.rs, same module)
    let get_f64_id = builder
        .get_function_by_name("haxe_array_get_f64")
        .expect("haxe_array_get_f64 must be declared before SIMD4f_fromArray");

    let arr = builder.get_param(0);

    // Extract 4 elements as f64, cast to f32, insert into vector
    let zero = builder.const_value(IrValue::F32(0.0));
    let mut vec = builder.vector_splat(zero, vec_ty.clone());

    for i in 0..4u8 {
        let idx = builder.const_value(IrValue::I64(i as i64));
        let val_f64 = builder
            .call(get_f64_id, vec![arr, idx])
            .expect("haxe_array_get_f64 returns f64");
        let val_f32 = builder.cast(val_f64, f64_ty.clone(), f32_ty.clone());
        vec = builder.vector_insert(vec, val_f32, i, vec_ty.clone());
    }

    builder.ret(Some(vec));
}

// ============================================================================
// SIMD4f math operations — single IR instruction wrappers
// ============================================================================

fn build_simd4f_unary(builder: &mut MirBuilder, name: &str, op: VectorUnaryOpKind) {
    let vec_ty = IrType::vector(IrType::F32, 4);

    let func_id = builder
        .begin_function(name)
        .param("self_val", vec_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let result = builder.vector_unary_op(op, self_val, vec_ty);
    builder.ret(Some(result));
}

fn build_simd4f_sqrt(builder: &mut MirBuilder) {
    build_simd4f_unary(builder, "SIMD4f_sqrt", VectorUnaryOpKind::Sqrt);
}

fn build_simd4f_abs(builder: &mut MirBuilder) {
    build_simd4f_unary(builder, "SIMD4f_abs", VectorUnaryOpKind::Abs);
}

fn build_simd4f_neg(builder: &mut MirBuilder) {
    build_simd4f_unary(builder, "SIMD4f_neg", VectorUnaryOpKind::Neg);
}

fn build_simd4f_ceil(builder: &mut MirBuilder) {
    build_simd4f_unary(builder, "SIMD4f_ceil", VectorUnaryOpKind::Ceil);
}

fn build_simd4f_floor(builder: &mut MirBuilder) {
    build_simd4f_unary(builder, "SIMD4f_floor", VectorUnaryOpKind::Floor);
}

fn build_simd4f_round(builder: &mut MirBuilder) {
    build_simd4f_unary(builder, "SIMD4f_round", VectorUnaryOpKind::Round);
}

fn build_simd4f_minmax(builder: &mut MirBuilder, name: &str, op: VectorMinMaxKind) {
    let vec_ty = IrType::vector(IrType::F32, 4);

    let func_id = builder
        .begin_function(name)
        .param("self_val", vec_ty.clone())
        .param("other", vec_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let result = builder.vector_min_max(op, self_val, other, vec_ty);
    builder.ret(Some(result));
}

fn build_simd4f_min(builder: &mut MirBuilder) {
    build_simd4f_minmax(builder, "SIMD4f_min", VectorMinMaxKind::Min);
}

fn build_simd4f_max(builder: &mut MirBuilder) {
    build_simd4f_minmax(builder, "SIMD4f_max", VectorMinMaxKind::Max);
}

// ============================================================================
// SIMD4f compound operations — built from primitive vector ops
// ============================================================================

/// clamp(lo, hi) = max(lo, min(hi, self))
fn build_simd4f_clamp(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);

    let func_id = builder
        .begin_function("SIMD4f_clamp")
        .param("self_val", vec_ty.clone())
        .param("lo", vec_ty.clone())
        .param("hi", vec_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let lo = builder.get_param(1);
    let hi = builder.get_param(2);
    let clamped_hi = builder.vector_min_max(VectorMinMaxKind::Min, self_val, hi, vec_ty.clone());
    let result = builder.vector_min_max(VectorMinMaxKind::Max, clamped_hi, lo, vec_ty);
    builder.ret(Some(result));
}

/// lerp(other, t) = self + (other - self) * t
fn build_simd4f_lerp(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("SIMD4f_lerp")
        .param("self_val", vec_ty.clone())
        .param("other", vec_ty.clone())
        .param("t", f64_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let t_f64 = builder.get_param(2);
    let t_scalar = builder.cast(t_f64, f64_ty, f32_ty);
    let t = builder.vector_splat(t_scalar, vec_ty.clone());
    let diff = builder.vector_bin_op(BinaryOp::Sub, other, self_val, vec_ty.clone());
    let scaled = builder.vector_bin_op(BinaryOp::Mul, diff, t, vec_ty.clone());
    let result = builder.vector_bin_op(BinaryOp::Add, self_val, scaled, vec_ty);
    builder.ret(Some(result));
}

/// length() = sqrt(dot(self, self))
fn build_simd4f_length(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("SIMD4f_length")
        .param("self_val", vec_ty.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let product = builder.vector_bin_op(BinaryOp::Mul, self_val, self_val, vec_ty.clone());
    let sum_f32 = builder.vector_reduce(BinaryOp::Add, product, f32_ty.clone());
    // sqrt of the sum
    let sqrt_val = builder.vector_splat(sum_f32, vec_ty.clone());
    let sqrt_vec = builder.vector_unary_op(VectorUnaryOpKind::Sqrt, sqrt_val, vec_ty);
    let sqrt_f32 = builder.vector_extract(sqrt_vec, 0, f32_ty.clone());
    let result = builder.cast(sqrt_f32, f32_ty, f64_ty);
    builder.ret(Some(result));
}

/// normalize() = self / splat(length(self))
fn build_simd4f_normalize(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;

    let func_id = builder
        .begin_function("SIMD4f_normalize")
        .param("self_val", vec_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let product = builder.vector_bin_op(BinaryOp::Mul, self_val, self_val, vec_ty.clone());
    let sum_f32 = builder.vector_reduce(BinaryOp::Add, product, f32_ty.clone());
    let sum_vec = builder.vector_splat(sum_f32, vec_ty.clone());
    let sqrt_vec = builder.vector_unary_op(VectorUnaryOpKind::Sqrt, sum_vec, vec_ty.clone());
    let result = builder.vector_bin_op(BinaryOp::Div, self_val, sqrt_vec, vec_ty);
    builder.ret(Some(result));
}

/// cross3(other) — 3D cross product (w lane = 0)
/// cross = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx, 0)
fn build_simd4f_cross3(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;

    let func_id = builder
        .begin_function("SIMD4f_cross3")
        .param("self_val", vec_ty.clone())
        .param("other", vec_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a = builder.get_param(0);
    let b = builder.get_param(1);

    // Extract components
    let ax = builder.vector_extract(a, 0, f32_ty.clone());
    let ay = builder.vector_extract(a, 1, f32_ty.clone());
    let az = builder.vector_extract(a, 2, f32_ty.clone());
    let bx = builder.vector_extract(b, 0, f32_ty.clone());
    let by = builder.vector_extract(b, 1, f32_ty.clone());
    let bz = builder.vector_extract(b, 2, f32_ty.clone());

    // Build (ay*bz, az*bx, ax*by, 0) and (az*by, ax*bz, ay*bx, 0)
    let zero = builder.const_value(IrValue::F32(0.0));
    let mut lhs = builder.vector_splat(zero, vec_ty.clone());
    let ay_bz = builder.bin_op(BinaryOp::FMul, ay, bz);
    let az_bx = builder.bin_op(BinaryOp::FMul, az, bx);
    let ax_by = builder.bin_op(BinaryOp::FMul, ax, by);
    lhs = builder.vector_insert(lhs, ay_bz, 0, vec_ty.clone());
    lhs = builder.vector_insert(lhs, az_bx, 1, vec_ty.clone());
    lhs = builder.vector_insert(lhs, ax_by, 2, vec_ty.clone());

    let mut rhs = builder.vector_splat(zero, vec_ty.clone());
    let az_by = builder.bin_op(BinaryOp::FMul, az, by);
    let ax_bz = builder.bin_op(BinaryOp::FMul, ax, bz);
    let ay_bx = builder.bin_op(BinaryOp::FMul, ay, bx);
    rhs = builder.vector_insert(rhs, az_by, 0, vec_ty.clone());
    rhs = builder.vector_insert(rhs, ax_bz, 1, vec_ty.clone());
    rhs = builder.vector_insert(rhs, ay_bx, 2, vec_ty.clone());

    let result = builder.vector_bin_op(BinaryOp::Sub, lhs, rhs, vec_ty);
    builder.ret(Some(result));
}

/// distance(other) = length(self - other)
fn build_simd4f_distance(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::F32, 4);
    let f32_ty = IrType::F32;
    let f64_ty = IrType::F64;

    let func_id = builder
        .begin_function("SIMD4f_distance")
        .param("self_val", vec_ty.clone())
        .param("other", vec_ty.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let other = builder.get_param(1);
    let diff = builder.vector_bin_op(BinaryOp::Sub, self_val, other, vec_ty.clone());
    let product = builder.vector_bin_op(BinaryOp::Mul, diff, diff, vec_ty.clone());
    let sum_f32 = builder.vector_reduce(BinaryOp::Add, product, f32_ty.clone());
    let sum_vec = builder.vector_splat(sum_f32, vec_ty.clone());
    let sqrt_vec = builder.vector_unary_op(VectorUnaryOpKind::Sqrt, sum_vec, vec_ty);
    let sqrt_f32 = builder.vector_extract(sqrt_vec, 0, f32_ty.clone());
    let result = builder.cast(sqrt_f32, f32_ty, f64_ty);
    builder.ret(Some(result));
}

// ============================================================================
// sys.io.File — MIR wrappers for default binary=true parameter
// ============================================================================

fn declare_file_externs(builder: &mut MirBuilder) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let bool_ty = IrType::Bool;

    let fid = builder
        .begin_function("haxe_file_read")
        .param("path", ptr_void.clone())
        .param("binary", bool_ty.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(fid);

    let fid = builder
        .begin_function("haxe_file_write")
        .param("path", ptr_void.clone())
        .param("binary", bool_ty.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(fid);

    let fid = builder
        .begin_function("haxe_file_append")
        .param("path", ptr_void.clone())
        .param("binary", bool_ty.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(fid);

    let fid = builder
        .begin_function("haxe_file_update")
        .param("path", ptr_void.clone())
        .param("binary", bool_ty)
        .returns(ptr_void)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(fid);
}

/// Helper to build a File method wrapper that defaults binary=true.
/// Creates: fn {wrapper_name}(path: *void) -> *void
/// Calls:   fn {extern_name}(path: *void, binary: bool) -> *void
fn build_file_method_default_binary(
    builder: &mut MirBuilder,
    wrapper_name: &str,
    extern_name: &str,
) {
    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
    let bool_ty = IrType::Bool;

    let func_id = builder
        .begin_function(wrapper_name)
        .param("path", ptr_void.clone())
        .returns(ptr_void.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let path = builder.get_param(0);
    let binary_true = builder.const_i64(1); // true as i64

    let target_func = builder
        .get_function_by_name(extern_name)
        .unwrap_or_else(|| panic!("{} extern not found", extern_name));

    if let Some(result) = builder.call(target_func, vec![path, binary_true]) {
        builder.ret(Some(result));
    } else {
        let null = builder.const_i64(0);
        builder.ret(Some(null));
    }
}

fn build_file_read_default(builder: &mut MirBuilder) {
    build_file_method_default_binary(builder, "file_read_default", "haxe_file_read");
}

fn build_file_write_default(builder: &mut MirBuilder) {
    build_file_method_default_binary(builder, "file_write_default", "haxe_file_write");
}

fn build_file_append_default(builder: &mut MirBuilder) {
    build_file_method_default_binary(builder, "file_append_default", "haxe_file_append");
}

fn build_file_update_default(builder: &mut MirBuilder) {
    build_file_method_default_binary(builder, "file_update_default", "haxe_file_update");
}
