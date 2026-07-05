/// Systems-level type MIR wrappers (Box, Ptr, Ref, Usize)
///
/// These are zero-cost abstracts over Int (i64) at MIR level.
/// Box operations delegate to runtime functions (alloc/free).
/// Ptr/Ref operations are direct load/store/arithmetic MIR instructions.
/// Usize operations are native i64 arithmetic.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{
    AtomicRmwOp, BinaryOp, CallingConvention, CompareOp, InlineHint, IrType, IrValue,
    VectorMinMaxKind, VectorUnaryOpKind,
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

    // Size-typed Ptr variants (L3). Selected at the call site by pointee size;
    // size 8 keeps the default Ptr_offset/deref/write. Size 2 (Ptr<I16/U16>)
    // has no consumer + no IrTypeDescriptor variant, so it falls to size-8.
    build_ptr_offset_sized(builder, "Ptr_offset_1", 1);
    build_ptr_offset_sized(builder, "Ptr_offset_4", 4);
    build_ptr_deref_typed(builder, "Ptr_deref_1", IrType::U8);
    build_ptr_deref_typed(builder, "Ptr_deref_4", IrType::I32);
    build_ptr_deref_typed(builder, "Ptr_deref_4f", IrType::F32);
    build_ptr_write_typed(builder, "Ptr_write_1", IrType::U8);
    build_ptr_write_typed(builder, "Ptr_write_4", IrType::I32);
    build_ptr_write_typed(builder, "Ptr_write_4f", IrType::F32);

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
    // SIMD4i32 (integer companion) MIR wrappers
    build_simd4i32_splat(builder);
    build_simd4i32_make(builder);
    build_simd4i32_extract(builder);
    build_simd4i32_insert(builder);
    build_simd4i32_sum(builder);
    build_simd4i32_dot16(builder);
    // SIMD16i8 (i8x16 dot operands) MIR wrappers
    build_simd16i8_splat(builder);
    build_simd16i8_load(builder);
    // Bitwise + shift (Q4 nibble unpack: AND-mask low nibble, USHR high nibble)
    build_simd16i8_and(builder);
    build_simd16i8_or(builder);
    build_simd16i8_xor(builder);
    build_simd16i8_shl(builder);
    build_simd16i8_shr(builder);
    build_simd16i8_ushr(builder);
    build_simd16i8_extract(builder);
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

    // rayzor.Atomic MIR wrappers (native atomic MIR ops — no externs)
    build_atomic_of(builder);
    build_atomic_load(builder);
    build_atomic_store(builder);
    build_atomic_fetch_add(builder);
    build_atomic_cas(builder);

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

// --- Size-typed Ptr wrappers (L3: Ptr<T> size-erasure fix) -------------------
// The default Ptr_offset/deref/write above are size-erased (treat T as i64=8B),
// so Ptr<Float>/Ptr<Int32>/Ptr<U8> compute the wrong byte stride and read/write
// the wrong width. These variants bake the correct element size in; the call
// site (hir_to_mir) redirects to the matching one by the receiver's pointee
// type, falling back to the size-8 default for i64/unknown pointees (so the
// existing corpus stays byte-identical). The `ptr` param MUST stay IrType::I64
// (never Ptr<elem>): the Cranelift Store path keys `is_struct_field` on a
// Ptr(_) ptr type and would sign-extend a narrow store back to 8 bytes.

/// Ptr_offset_<sz>(ptr: i64, n: i64) -> ptr + n*<sz>   (sz = sizeof(T))
fn build_ptr_offset_sized(builder: &mut MirBuilder, name: &str, sz: i64) {
    let i64_ty = IrType::I64;
    let func_id = builder
        .begin_function(name)
        .param("ptr", i64_ty.clone())
        .param("n", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let ptr = builder.get_param(0);
    let n = builder.get_param(1);
    let k = builder.const_i64(sz);
    let byte_offset = builder.mul(n, k, i64_ty.clone());
    let result = builder.add(ptr, byte_offset, i64_ty);
    builder.ret(Some(result));
}

/// Ptr_deref_<w>(ptr: i64) -> load <elem_ty>   (elem_ty drives load width)
fn build_ptr_deref_typed(builder: &mut MirBuilder, name: &str, elem_ty: IrType) {
    let func_id = builder
        .begin_function(name)
        .param("ptr", IrType::I64) // MUST be I64, not Ptr<elem>
        .returns(elem_ty.clone())
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let ptr = builder.get_param(0);
    let value = builder.load(ptr, elem_ty);
    builder.ret(Some(value));
}

/// Ptr_write_<w>(ptr: i64, value: <elem_ty>) -> void   (value reg type drives store width)
fn build_ptr_write_typed(builder: &mut MirBuilder, name: &str, elem_ty: IrType) {
    let func_id = builder
        .begin_function(name)
        .param("ptr", IrType::I64) // MUST be I64 (Cranelift is_struct_field guard)
        .param("value", elem_ty.clone())
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .inline(InlineHint::Always)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let ptr = builder.get_param(0);
    let value = builder.get_param(1);
    builder.store(ptr, value);
    builder.ret(None);
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

// ---------------------------------------------------------------------------
// rayzor.Atomic MIR wrappers
// ---------------------------------------------------------------------------

/// Atomic_of(addr: i64) -> i64  (static identity — returns the address)
fn build_atomic_of(builder: &mut MirBuilder) {
    let func_id = builder
        .begin_function("Atomic_of")
        .param("addr", IrType::I64)
        .returns(IrType::I64)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let addr = builder.get_param(0);
    builder.ret(Some(addr)); // identity
}

/// Atomic_load(self_addr: i64) -> i32
fn build_atomic_load(builder: &mut MirBuilder) {
    let func_id = builder
        .begin_function("Atomic_load")
        .param("self_addr", IrType::I64)
        .returns(IrType::I32)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let addr = builder.get_param(0);
    let v = builder.atomic_load(addr, IrType::I32);
    builder.ret(Some(v));
}

/// Atomic_store(self_addr: i64, v: i32) -> void
fn build_atomic_store(builder: &mut MirBuilder) {
    let func_id = builder
        .begin_function("Atomic_store")
        .param("self_addr", IrType::I64)
        .param("v", IrType::I32)
        .returns(IrType::Void)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let addr = builder.get_param(0);
    let v = builder.get_param(1);
    builder.atomic_store(addr, v, IrType::I32);
    builder.ret(None);
}

/// Atomic_fetch_add(self_addr: i64, v: i32) -> i32  (returns old)
fn build_atomic_fetch_add(builder: &mut MirBuilder) {
    let func_id = builder
        .begin_function("Atomic_fetch_add")
        .param("self_addr", IrType::I64)
        .param("v", IrType::I32)
        .returns(IrType::I32)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let addr = builder.get_param(0);
    let v = builder.get_param(1);
    let old = builder.atomic_rmw(AtomicRmwOp::Add, addr, v, IrType::I32);
    builder.ret(Some(old));
}

/// Atomic_cas(self_addr: i64, expected: i32, replacement: i32) -> i32  (returns value read)
fn build_atomic_cas(builder: &mut MirBuilder) {
    let func_id = builder
        .begin_function("Atomic_cas")
        .param("self_addr", IrType::I64)
        .param("expected", IrType::I32)
        .param("replacement", IrType::I32)
        .returns(IrType::I32)
        .calling_convention(CallingConvention::C)
        .build();
    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);
    let addr = builder.get_param(0);
    let exp = builder.get_param(1);
    let rep = builder.get_param(2);
    let old = builder.atomic_cas(addr, exp, rep, IrType::I32);
    builder.ret(Some(old));
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
    let lane = builder.get_param(1);

    // VectorExtract requires a compile-time-constant lane, but `lane` here is a
    // runtime value (e.g. `v.get(i)` in a loop). Extract all four lanes with
    // constant indices and pick the matching one with a branchless select chain
    // — selecting on f32 scalars (the wasm untyped `select` is valid for f32).
    // The previous code hard-coded lane 0, so `v.get(i)` always returned lane 0.
    let e0 = builder.vector_extract(self_val, 0, f32_ty.clone());
    let e1 = builder.vector_extract(self_val, 1, f32_ty.clone());
    let e2 = builder.vector_extract(self_val, 2, f32_ty.clone());
    let e3 = builder.vector_extract(self_val, 3, f32_ty.clone());

    let c1 = builder.const_i32(1);
    let c2 = builder.const_i32(2);
    let c3 = builder.const_i32(3);
    // Default to lane 0; override when lane == 1/2/3.
    let is1 = builder.icmp(CompareOp::Eq, lane, c1, IrType::Bool);
    let r1 = builder.select(is1, e1, e0, f32_ty.clone());
    let is2 = builder.icmp(CompareOp::Eq, lane, c2, IrType::Bool);
    let r2 = builder.select(is2, e2, r1, f32_ty.clone());
    let is3 = builder.icmp(CompareOp::Eq, lane, c3, IrType::Bool);
    let r3 = builder.select(is3, e3, r2, f32_ty.clone());

    let result = builder.cast(r3, f32_ty, f64_ty);
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

// ============================================================================
// SIMD4i32 — 128-bit vector of 4×i32 (integer companion to SIMD4f)
// add/sub/mul lower through VectorBinOp (the @:op skip), so only the
// non-operator helpers need MIR wrappers.
// ============================================================================

/// SIMD4i32_splat(scalar: i32) -> vec<i32; 4>
fn build_simd4i32_splat(builder: &mut MirBuilder) {
    let i32_ty = IrType::I32;
    let vec_ty = IrType::vector(IrType::I32, 4);

    let func_id = builder
        .begin_function("SIMD4i32_splat")
        .param("scalar", i32_ty)
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

/// SIMD4i32_make(x, y, z, w: i32) -> vec<i32; 4>
fn build_simd4i32_make(builder: &mut MirBuilder) {
    let i32_ty = IrType::I32;
    let vec_ty = IrType::vector(IrType::I32, 4);

    let func_id = builder
        .begin_function("SIMD4i32_make")
        .param("x", i32_ty.clone())
        .param("y", i32_ty.clone())
        .param("z", i32_ty.clone())
        .param("w", i32_ty)
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

    let v0 = builder.vector_splat(x, vec_ty.clone());
    let v1 = builder.vector_insert(v0, y, 1, vec_ty.clone());
    let v2 = builder.vector_insert(v1, z, 2, vec_ty.clone());
    let v3 = builder.vector_insert(v2, w, 3, vec_ty);
    builder.ret(Some(v3));
}

/// SIMD4i32_extract(self: vec<i32; 4>, lane: i32) -> i32
/// Runtime lane via a branchless select chain (same approach as SIMD4f_extract).
fn build_simd4i32_extract(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::I32, 4);
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("SIMD4i32_extract")
        .param("self_val", vec_ty)
        .param("lane", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let lane = builder.get_param(1);

    let e0 = builder.vector_extract(self_val, 0, i32_ty.clone());
    let e1 = builder.vector_extract(self_val, 1, i32_ty.clone());
    let e2 = builder.vector_extract(self_val, 2, i32_ty.clone());
    let e3 = builder.vector_extract(self_val, 3, i32_ty.clone());

    let c1 = builder.const_i32(1);
    let c2 = builder.const_i32(2);
    let c3 = builder.const_i32(3);
    let is1 = builder.icmp(CompareOp::Eq, lane, c1, IrType::Bool);
    let r1 = builder.select(is1, e1, e0, i32_ty.clone());
    let is2 = builder.icmp(CompareOp::Eq, lane, c2, IrType::Bool);
    let r2 = builder.select(is2, e2, r1, i32_ty.clone());
    let is3 = builder.icmp(CompareOp::Eq, lane, c3, IrType::Bool);
    let r3 = builder.select(is3, e3, r2, i32_ty.clone());

    builder.ret(Some(r3));
}

/// SIMD4i32_insert(self: vec<i32; 4>, lane: i32, value: i32) -> vec<i32; 4>
/// Static lane 0 for now (same limitation as SIMD4f_insert).
fn build_simd4i32_insert(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::I32, 4);
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("SIMD4i32_insert")
        .param("self_val", vec_ty.clone())
        .param("lane", i32_ty.clone())
        .param("value", i32_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let value = builder.get_param(2);
    let result = builder.vector_insert(self_val, value, 0, vec_ty);
    builder.ret(Some(result));
}

/// SIMD4i32_sum(self: vec<i32; 4>) -> i32  — horizontal add
fn build_simd4i32_sum(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::I32, 4);
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("SIMD4i32_sum")
        .param("self_val", vec_ty)
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let result = builder.vector_reduce(BinaryOp::Add, self_val, i32_ty.clone());
    builder.ret(Some(result));
}

/// SIMD4i32_dot16(acc: vec<i32;4>, a: vec<i8;16>, b: vec<i8;16>) -> vec<i32;4>
/// Fused widening dot-accumulate (the quant-matmul primitive): acc + the 16
/// i8×i8 products summed in groups of 4 → 4 i32 lanes. Lowers to SDOT.
fn build_simd4i32_dot16(builder: &mut MirBuilder) {
    let vec_i32 = IrType::vector(IrType::I32, 4);
    let vec_i8 = IrType::vector(IrType::I8, 16);

    let func_id = builder
        .begin_function("SIMD4i32_dot16")
        .param("acc", vec_i32.clone())
        .param("a", vec_i8.clone())
        .param("b", vec_i8.clone())
        .returns(vec_i32)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let acc = builder.get_param(0);
    let a = builder.get_param(1);
    let b = builder.get_param(2);
    let result = builder.vector_dot(acc, a, b);
    builder.ret(Some(result));
}

// ============================================================================
// SIMD16i8 — 128-bit vector of 16×i8 (the integer dot operands)
// ============================================================================

/// SIMD16i8_splat(scalar: i32) -> vec<i8; 16>  (broadcast low byte to all lanes)
fn build_simd16i8_splat(builder: &mut MirBuilder) {
    let i32_ty = IrType::I32;
    let i8_ty = IrType::I8;
    let vec_ty = IrType::vector(IrType::I8, 16);

    let func_id = builder
        .begin_function("SIMD16i8_splat")
        .param("scalar", i32_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let scalar = builder.get_param(0);
    // Narrow i32 → i8 so the Cranelift splat gets the i8 lane type. On wasm the
    // value stays an i32 local and i8x16.splat uses its low byte either way.
    let scalar8 = builder.cast(scalar, i32_ty, i8_ty);
    let result = builder.vector_splat(scalar8, vec_ty);
    builder.ret(Some(result));
}

/// SIMD16i8_load(ptr: i64) -> vec<i8; 16>  (load 16 contiguous bytes)
fn build_simd16i8_load(builder: &mut MirBuilder) {
    let i64_ty = IrType::I64;
    let vec_ty = IrType::vector(IrType::I8, 16);

    let func_id = builder
        .begin_function("SIMD16i8_load")
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

/// Shared body for the lane-wise i8x16 bitwise wrappers
/// `SIMD16i8_and/or/xor(a, b: vec<i8;16>) -> vec<i8;16>`.
fn build_simd16i8_bitwise(builder: &mut MirBuilder, name: &str, op: BinaryOp) {
    let vec_ty = IrType::vector(IrType::I8, 16);

    let func_id = builder
        .begin_function(name)
        .param("a", vec_ty.clone())
        .param("b", vec_ty.clone())
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a = builder.get_param(0);
    let b = builder.get_param(1);
    let result = builder.vector_bin_op(op, a, b, vec_ty);
    builder.ret(Some(result));
}

fn build_simd16i8_and(builder: &mut MirBuilder) {
    build_simd16i8_bitwise(builder, "SIMD16i8_and", BinaryOp::And);
}
fn build_simd16i8_or(builder: &mut MirBuilder) {
    build_simd16i8_bitwise(builder, "SIMD16i8_or", BinaryOp::Or);
}
fn build_simd16i8_xor(builder: &mut MirBuilder) {
    build_simd16i8_bitwise(builder, "SIMD16i8_xor", BinaryOp::Xor);
}

/// Shared body for the i8x16 shift wrappers
/// `SIMD16i8_shl/shr/ushr(a: vec<i8;16>, n: i32) -> vec<i8;16>`.
/// `n` is a scalar shift amount applied to every lane (vector-by-scalar),
/// matching the wasm I8x16Shl/ShrS/ShrU and cranelift ishl/sshr/ushr shape.
fn build_simd16i8_shift(builder: &mut MirBuilder, name: &str, op: BinaryOp) {
    let vec_ty = IrType::vector(IrType::I8, 16);
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function(name)
        .param("a", vec_ty.clone())
        .param("n", i32_ty)
        .returns(vec_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let a = builder.get_param(0);
    let n = builder.get_param(1);
    let result = builder.vector_bin_op(op, a, n, vec_ty);
    builder.ret(Some(result));
}

fn build_simd16i8_shl(builder: &mut MirBuilder) {
    build_simd16i8_shift(builder, "SIMD16i8_shl", BinaryOp::Shl);
}
fn build_simd16i8_shr(builder: &mut MirBuilder) {
    build_simd16i8_shift(builder, "SIMD16i8_shr", BinaryOp::Shr);
}
fn build_simd16i8_ushr(builder: &mut MirBuilder) {
    build_simd16i8_shift(builder, "SIMD16i8_ushr", BinaryOp::Ushr);
}

/// SIMD16i8_extract(self: vec<i8;16>, lane: i32) -> i32
/// Read one i8 lane, sign-extended to i32, via a branchless select chain
/// (same approach as SIMD4i32_extract). Lets a Haxe kernel pull individual
/// bytes out of a loaded vector in-guest (the Q4 block header: d/dmin/scales),
/// avoiding the broken Ptr<Int>.deref and per-byte Bytes FFI crossings on wasm.
/// Bytes are signed i8; mask `& 0xFF` in Haxe for unsigned 0..255.
fn build_simd16i8_extract(builder: &mut MirBuilder) {
    let vec_ty = IrType::vector(IrType::I8, 16);
    let i8_ty = IrType::I8;
    let i32_ty = IrType::I32;

    let func_id = builder
        .begin_function("SIMD16i8_extract")
        .param("self_val", vec_ty)
        .param("lane", i32_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);
    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let self_val = builder.get_param(0);
    let lane = builder.get_param(1);

    // Keep this allocation-free. VectorExtract needs a constant lane, so pull
    // all lanes and select the requested one. LLVM/Cranelift then fold the
    // chain away when Haxe passes a constant lane such as `hdr.get(0)`.
    let mut selected = builder.vector_extract(self_val, 0, i8_ty.clone());
    for lane_idx in 1..16 {
        let extracted = builder.vector_extract(self_val, lane_idx, i8_ty.clone());
        let lane_const = builder.const_i32(lane_idx as i32);
        let is_lane = builder.icmp(CompareOp::Eq, lane, lane_const, IrType::Bool);
        selected = builder.select(is_lane, extracted, selected, i8_ty.clone());
    }

    let r = builder.cast(selected, i8_ty.clone(), i32_ty.clone());
    // Normalise the low byte to a sign-extended i32: the cast above zero-extends
    // on Cranelift but I8x16ExtractLaneS sign-extends on wasm — `(r << 24) >> 24`
    // (arithmetic) makes get() consistently SIGNED i8 on both targets. Header
    // reads mask `& 0xFF` for unsigned and are unaffected either way.
    let c24 = builder.const_i32(24);
    let shl = builder.bin_op(BinaryOp::Shl, r, c24);
    let signed = builder.bin_op(BinaryOp::Shr, shl, c24);
    builder.ret(Some(signed));
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
