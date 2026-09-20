/// Vec<T>: Generic vector type with monomorphized specializations
///
/// This module declares extern runtime functions for Vec operations.
/// Vec<T> is monomorphized at compile time to type-specific variants:
/// - Vec<Int> -> VecI32
/// - Vec<Float> -> VecF64
/// - Vec<Int64> -> VecI64
/// - Vec<Bool> -> VecBool
/// - Vec<T> (reference types) -> VecPtr
///
/// Unlike Thread/Channel which have MIR wrappers, Vec functions are direct
/// extern calls since they don't need closure handling or special marshalling.
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all Vec extern declarations and MIR wrappers
pub fn build_vec_externs(builder: &mut MirBuilder) {
    // Declare extern runtime functions
    declare_vec_i32_externs(builder);
    declare_vec_i64_externs(builder);
    declare_vec_f64_externs(builder);
    declare_vec_ptr_externs(builder);
    declare_vec_bool_externs(builder);

    // Build MIR wrapper functions that forward to extern functions
    build_vec_i32_wrappers(builder);
    build_vec_i64_wrappers(builder);
    build_vec_f64_wrappers(builder);
    build_vec_ptr_wrappers(builder);
    build_vec_bool_wrappers(builder);
}

/// Declare VecI32 (Vec<Int>) extern functions
fn declare_vec_i32_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // rayzor_vec_i32_new() -> *VecI32
    let func_id = builder
        .begin_function("rayzor_vec_i32_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_with_capacity(capacity: i64) -> *VecI32
    let func_id = builder
        .begin_function("rayzor_vec_i32_with_capacity")
        .param("capacity", i64_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_push(vec: *VecI32, value: i32)
    let func_id = builder
        .begin_function("rayzor_vec_i32_push")
        .param("vec", ptr_u8.clone())
        .param("value", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_pop(vec: *VecI32) -> i32
    let func_id = builder
        .begin_function("rayzor_vec_i32_pop")
        .param("vec", ptr_u8.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_get(vec: *VecI32, index: i64) -> i32
    let func_id = builder
        .begin_function("rayzor_vec_i32_get")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_set(vec: *VecI32, index: i64, value: i32)
    let func_id = builder
        .begin_function("rayzor_vec_i32_set")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .param("value", i32_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_len(vec: *VecI32) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i32_len")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_capacity(vec: *VecI32) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i32_capacity")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_is_empty(vec: *VecI32) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_i32_is_empty")
        .param("vec", ptr_u8.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_clear(vec: *VecI32)
    let func_id = builder
        .begin_function("rayzor_vec_i32_clear")
        .param("vec", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_first(vec: *VecI32) -> i32
    let func_id = builder
        .begin_function("rayzor_vec_i32_first")
        .param("vec", ptr_u8.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_last(vec: *VecI32) -> i32
    let func_id = builder
        .begin_function("rayzor_vec_i32_last")
        .param("vec", ptr_u8.clone())
        .returns(i32_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_sort(vec: *VecI32)
    let func_id = builder
        .begin_function("rayzor_vec_i32_sort")
        .param("vec", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i32_sort_by(vec: *VecI32, compare_fn: *u8, compare_env: *u8)
    let func_id = builder
        .begin_function("rayzor_vec_i32_sort_by")
        .param("vec", ptr_u8.clone())
        .param("compare_fn", ptr_u8.clone())
        .param("compare_env", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Declare VecI64 (Vec<Int64>) extern functions
fn declare_vec_i64_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // rayzor_vec_i64_new() -> *VecI64
    let func_id = builder
        .begin_function("rayzor_vec_i64_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_push(vec: *VecI64, value: i64)
    let func_id = builder
        .begin_function("rayzor_vec_i64_push")
        .param("vec", ptr_u8.clone())
        .param("value", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_pop(vec: *VecI64) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i64_pop")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_get(vec: *VecI64, index: i64) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i64_get")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_set(vec: *VecI64, index: i64, value: i64)
    let func_id = builder
        .begin_function("rayzor_vec_i64_set")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .param("value", i64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_len(vec: *VecI64) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i64_len")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_is_empty(vec: *VecI64) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_i64_is_empty")
        .param("vec", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_clear(vec: *VecI64)
    let func_id = builder
        .begin_function("rayzor_vec_i64_clear")
        .param("vec", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_first(vec: *VecI64) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i64_first")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_i64_last(vec: *VecI64) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_i64_last")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Declare VecF64 (Vec<Float>) extern functions
fn declare_vec_f64_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let f64_ty = IrType::F64;
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // rayzor_vec_f64_new() -> *VecF64
    let func_id = builder
        .begin_function("rayzor_vec_f64_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_push(vec: *VecF64, value: f64)
    let func_id = builder
        .begin_function("rayzor_vec_f64_push")
        .param("vec", ptr_u8.clone())
        .param("value", f64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_pop(vec: *VecF64) -> f64
    let func_id = builder
        .begin_function("rayzor_vec_f64_pop")
        .param("vec", ptr_u8.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_get(vec: *VecF64, index: i64) -> f64
    let func_id = builder
        .begin_function("rayzor_vec_f64_get")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_set(vec: *VecF64, index: i64, value: f64)
    let func_id = builder
        .begin_function("rayzor_vec_f64_set")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .param("value", f64_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_len(vec: *VecF64) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_f64_len")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_is_empty(vec: *VecF64) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_f64_is_empty")
        .param("vec", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_clear(vec: *VecF64)
    let func_id = builder
        .begin_function("rayzor_vec_f64_clear")
        .param("vec", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_first(vec: *VecF64) -> f64
    let func_id = builder
        .begin_function("rayzor_vec_f64_first")
        .param("vec", ptr_u8.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_last(vec: *VecF64) -> f64
    let func_id = builder
        .begin_function("rayzor_vec_f64_last")
        .param("vec", ptr_u8.clone())
        .returns(f64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_sort(vec: *VecF64)
    let func_id = builder
        .begin_function("rayzor_vec_f64_sort")
        .param("vec", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_f64_sort_by(vec: *VecF64, compare_fn: *u8, compare_env: *u8)
    let func_id = builder
        .begin_function("rayzor_vec_f64_sort_by")
        .param("vec", ptr_u8.clone())
        .param("compare_fn", ptr_u8.clone())
        .param("compare_env", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Declare VecPtr (Vec<T> for reference types) extern functions
fn declare_vec_ptr_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // rayzor_vec_ptr_new() -> *VecPtr
    let func_id = builder
        .begin_function("rayzor_vec_ptr_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_push(vec: *VecPtr, value: *u8)
    let func_id = builder
        .begin_function("rayzor_vec_ptr_push")
        .param("vec", ptr_u8.clone())
        .param("value", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_pop(vec: *VecPtr) -> *u8
    let func_id = builder
        .begin_function("rayzor_vec_ptr_pop")
        .param("vec", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_get(vec: *VecPtr, index: i64) -> *u8
    let func_id = builder
        .begin_function("rayzor_vec_ptr_get")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_set(vec: *VecPtr, index: i64, value: *u8)
    let func_id = builder
        .begin_function("rayzor_vec_ptr_set")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .param("value", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_len(vec: *VecPtr) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_ptr_len")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_is_empty(vec: *VecPtr) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_ptr_is_empty")
        .param("vec", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_clear(vec: *VecPtr)
    let func_id = builder
        .begin_function("rayzor_vec_ptr_clear")
        .param("vec", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_first(vec: *VecPtr) -> *u8
    let func_id = builder
        .begin_function("rayzor_vec_ptr_first")
        .param("vec", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_last(vec: *VecPtr) -> *u8
    let func_id = builder
        .begin_function("rayzor_vec_ptr_last")
        .param("vec", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_ptr_sort_by(vec: *VecPtr, compare_fn: *u8, compare_env: *u8)
    let func_id = builder
        .begin_function("rayzor_vec_ptr_sort_by")
        .param("vec", ptr_u8.clone())
        .param("compare_fn", ptr_u8.clone())
        .param("compare_env", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Declare VecBool (Vec<Bool>) extern functions
fn declare_vec_bool_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // rayzor_vec_bool_new() -> *VecBool
    let func_id = builder
        .begin_function("rayzor_vec_bool_new")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_push(vec: *VecBool, value: bool)
    let func_id = builder
        .begin_function("rayzor_vec_bool_push")
        .param("vec", ptr_u8.clone())
        .param("value", bool_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_pop(vec: *VecBool) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_bool_pop")
        .param("vec", ptr_u8.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_get(vec: *VecBool, index: i64) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_bool_get")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_set(vec: *VecBool, index: i64, value: bool)
    let func_id = builder
        .begin_function("rayzor_vec_bool_set")
        .param("vec", ptr_u8.clone())
        .param("index", i64_ty.clone())
        .param("value", bool_ty.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_len(vec: *VecBool) -> i64
    let func_id = builder
        .begin_function("rayzor_vec_bool_len")
        .param("vec", ptr_u8.clone())
        .returns(i64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_is_empty(vec: *VecBool) -> bool
    let func_id = builder
        .begin_function("rayzor_vec_bool_is_empty")
        .param("vec", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // rayzor_vec_bool_clear(vec: *VecBool)
    let func_id = builder
        .begin_function("rayzor_vec_bool_clear")
        .param("vec", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

// ============================================================================
// MIR Wrapper Functions
// ============================================================================
//
// These wrapper functions are called by user code. They simply forward to
// the corresponding extern runtime functions. This matches how Thread/Channel
// wrappers work in the stdlib.

/// Build VecI32 MIR wrappers
fn build_vec_i32_wrappers(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // VecI32_new() -> *VecI32
    {
        let func_id = builder
            .begin_function("VecI32_new")
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_new")
            .expect("rayzor_vec_i32_new not found");
        let result = builder.call(extern_id, vec![]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_push(vec: *VecI32, value: i32)
    {
        let func_id = builder
            .begin_function("VecI32_push")
            .param("vec", ptr_u8.clone())
            .param("value", i32_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let value = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_push")
            .expect("rayzor_vec_i32_push not found");
        let _ = builder.call(extern_id, vec![vec, value]);
        builder.ret(None);
    }

    // VecI32_pop(vec: *VecI32) -> i32
    {
        let func_id = builder
            .begin_function("VecI32_pop")
            .param("vec", ptr_u8.clone())
            .returns(i32_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_pop")
            .expect("rayzor_vec_i32_pop not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_get(vec: *VecI32, index: i64) -> i32
    {
        let func_id = builder
            .begin_function("VecI32_get")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .returns(i32_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_get")
            .expect("rayzor_vec_i32_get not found");
        let result = builder.call(extern_id, vec![vec, index]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_set(vec: *VecI32, index: i64, value: i32)
    {
        let func_id = builder
            .begin_function("VecI32_set")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .param("value", i32_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let value = builder.get_param(2);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_set")
            .expect("rayzor_vec_i32_set not found");
        let _ = builder.call(extern_id, vec![vec, index, value]);
        builder.ret(None);
    }

    // VecI32_length(vec: *VecI32) -> i64
    {
        let func_id = builder
            .begin_function("VecI32_length")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_len")
            .expect("rayzor_vec_i32_len not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_capacity(vec: *VecI32) -> i64
    {
        let func_id = builder
            .begin_function("VecI32_capacity")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_capacity")
            .expect("rayzor_vec_i32_capacity not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_isEmpty(vec: *VecI32) -> bool
    {
        let func_id = builder
            .begin_function("VecI32_isEmpty")
            .param("vec", ptr_u8.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_is_empty")
            .expect("rayzor_vec_i32_is_empty not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_clear(vec: *VecI32)
    {
        let func_id = builder
            .begin_function("VecI32_clear")
            .param("vec", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_clear")
            .expect("rayzor_vec_i32_clear not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }

    // VecI32_first(vec: *VecI32) -> i32
    {
        let func_id = builder
            .begin_function("VecI32_first")
            .param("vec", ptr_u8.clone())
            .returns(i32_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_first")
            .expect("rayzor_vec_i32_first not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_last(vec: *VecI32) -> i32
    {
        let func_id = builder
            .begin_function("VecI32_last")
            .param("vec", ptr_u8.clone())
            .returns(i32_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_last")
            .expect("rayzor_vec_i32_last not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI32_sort(vec: *VecI32)
    {
        let func_id = builder
            .begin_function("VecI32_sort")
            .param("vec", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_sort")
            .expect("rayzor_vec_i32_sort not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }

    // VecI32_sortBy(vec: *VecI32, compare: closure)
    {
        let func_id = builder
            .begin_function("VecI32_sortBy")
            .param("vec", ptr_u8.clone())
            .param("compare", ptr_u8.clone())
            .returns(void_ty)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        // The comparator is a closure record {fn_ptr, env}, as array_sort takes.
        let closure = builder.get_param(1);
        let compare_fn = builder.load(closure, ptr_u8.clone());
        let offset_8 = builder.const_i64(8);
        let env_slot = builder.ptr_add(closure, offset_8, ptr_u8.clone());
        let compare_env = builder.load(env_slot, ptr_u8.clone());
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i32_sort_by")
            .expect("rayzor_vec_i32_sort_by not found");
        let _ = builder.call(extern_id, vec![vec, compare_fn, compare_env]);
        builder.ret(None);
    }
}

/// Build VecI64 MIR wrappers
fn build_vec_i64_wrappers(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // VecI64_new() -> *VecI64
    {
        let func_id = builder
            .begin_function("VecI64_new")
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_new")
            .expect("rayzor_vec_i64_new not found");
        let result = builder.call(extern_id, vec![]).unwrap();
        builder.ret(Some(result));
    }

    // VecI64_push(vec: *VecI64, value: i64)
    {
        let func_id = builder
            .begin_function("VecI64_push")
            .param("vec", ptr_u8.clone())
            .param("value", i64_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let value = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_push")
            .expect("rayzor_vec_i64_push not found");
        let _ = builder.call(extern_id, vec![vec, value]);
        builder.ret(None);
    }

    // VecI64_pop(vec: *VecI64) -> i64
    {
        let func_id = builder
            .begin_function("VecI64_pop")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_pop")
            .expect("rayzor_vec_i64_pop not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI64_get(vec: *VecI64, index: i64) -> i64
    {
        let func_id = builder
            .begin_function("VecI64_get")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_get")
            .expect("rayzor_vec_i64_get not found");
        let result = builder.call(extern_id, vec![vec, index]).unwrap();
        builder.ret(Some(result));
    }

    // VecI64_set(vec: *VecI64, index: i64, value: i64)
    {
        let func_id = builder
            .begin_function("VecI64_set")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .param("value", i64_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let value = builder.get_param(2);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_set")
            .expect("rayzor_vec_i64_set not found");
        let _ = builder.call(extern_id, vec![vec, index, value]);
        builder.ret(None);
    }

    // VecI64_length(vec: *VecI64) -> i64
    {
        let func_id = builder
            .begin_function("VecI64_length")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_len")
            .expect("rayzor_vec_i64_len not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI64_isEmpty(vec: *VecI64) -> bool
    {
        let func_id = builder
            .begin_function("VecI64_isEmpty")
            .param("vec", ptr_u8.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_is_empty")
            .expect("rayzor_vec_i64_is_empty not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI64_clear(vec: *VecI64)
    {
        let func_id = builder
            .begin_function("VecI64_clear")
            .param("vec", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_clear")
            .expect("rayzor_vec_i64_clear not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }

    // VecI64_first(vec: *VecI64) -> i64
    {
        let func_id = builder
            .begin_function("VecI64_first")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_first")
            .expect("rayzor_vec_i64_first not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecI64_last(vec: *VecI64) -> i64
    {
        let func_id = builder
            .begin_function("VecI64_last")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_i64_last")
            .expect("rayzor_vec_i64_last not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }
}

/// Build VecF64 MIR wrappers
fn build_vec_f64_wrappers(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let f64_ty = IrType::F64;
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // VecF64_new() -> *VecF64
    {
        let func_id = builder
            .begin_function("VecF64_new")
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_new")
            .expect("rayzor_vec_f64_new not found");
        let result = builder.call(extern_id, vec![]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_push(vec: *VecF64, value: f64)
    {
        let func_id = builder
            .begin_function("VecF64_push")
            .param("vec", ptr_u8.clone())
            .param("value", f64_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let value = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_push")
            .expect("rayzor_vec_f64_push not found");
        let _ = builder.call(extern_id, vec![vec, value]);
        builder.ret(None);
    }

    // VecF64_pop(vec: *VecF64) -> f64
    {
        let func_id = builder
            .begin_function("VecF64_pop")
            .param("vec", ptr_u8.clone())
            .returns(f64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_pop")
            .expect("rayzor_vec_f64_pop not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_get(vec: *VecF64, index: i64) -> f64
    {
        let func_id = builder
            .begin_function("VecF64_get")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .returns(f64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_get")
            .expect("rayzor_vec_f64_get not found");
        let result = builder.call(extern_id, vec![vec, index]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_set(vec: *VecF64, index: i64, value: f64)
    {
        let func_id = builder
            .begin_function("VecF64_set")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .param("value", f64_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let value = builder.get_param(2);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_set")
            .expect("rayzor_vec_f64_set not found");
        let _ = builder.call(extern_id, vec![vec, index, value]);
        builder.ret(None);
    }

    // VecF64_length(vec: *VecF64) -> i64
    {
        let func_id = builder
            .begin_function("VecF64_length")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_len")
            .expect("rayzor_vec_f64_len not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_isEmpty(vec: *VecF64) -> bool
    {
        let func_id = builder
            .begin_function("VecF64_isEmpty")
            .param("vec", ptr_u8.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_is_empty")
            .expect("rayzor_vec_f64_is_empty not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_clear(vec: *VecF64)
    {
        let func_id = builder
            .begin_function("VecF64_clear")
            .param("vec", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_clear")
            .expect("rayzor_vec_f64_clear not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }

    // VecF64_first(vec: *VecF64) -> f64
    {
        let func_id = builder
            .begin_function("VecF64_first")
            .param("vec", ptr_u8.clone())
            .returns(f64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_first")
            .expect("rayzor_vec_f64_first not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_last(vec: *VecF64) -> f64
    {
        let func_id = builder
            .begin_function("VecF64_last")
            .param("vec", ptr_u8.clone())
            .returns(f64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_last")
            .expect("rayzor_vec_f64_last not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecF64_sort(vec: *VecF64)
    {
        let func_id = builder
            .begin_function("VecF64_sort")
            .param("vec", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_sort")
            .expect("rayzor_vec_f64_sort not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }

    // VecF64_sortBy(vec: *VecF64, compare: closure)
    {
        let func_id = builder
            .begin_function("VecF64_sortBy")
            .param("vec", ptr_u8.clone())
            .param("compare", ptr_u8.clone())
            .returns(void_ty)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        // The comparator is a closure record {fn_ptr, env}, as array_sort takes.
        let closure = builder.get_param(1);
        let compare_fn = builder.load(closure, ptr_u8.clone());
        let offset_8 = builder.const_i64(8);
        let env_slot = builder.ptr_add(closure, offset_8, ptr_u8.clone());
        let compare_env = builder.load(env_slot, ptr_u8.clone());
        let extern_id = builder
            .get_function_by_name("rayzor_vec_f64_sort_by")
            .expect("rayzor_vec_f64_sort_by not found");
        let _ = builder.call(extern_id, vec![vec, compare_fn, compare_env]);
        builder.ret(None);
    }
}

/// Build VecPtr MIR wrappers
fn build_vec_ptr_wrappers(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // VecPtr_new() -> *VecPtr
    {
        let func_id = builder
            .begin_function("VecPtr_new")
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_new")
            .expect("rayzor_vec_ptr_new not found");
        let result = builder.call(extern_id, vec![]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_push(vec: *VecPtr, value: *u8)
    {
        let func_id = builder
            .begin_function("VecPtr_push")
            .param("vec", ptr_u8.clone())
            .param("value", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let value = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_push")
            .expect("rayzor_vec_ptr_push not found");
        let _ = builder.call(extern_id, vec![vec, value]);
        builder.ret(None);
    }

    // VecPtr_pop(vec: *VecPtr) -> *u8
    {
        let func_id = builder
            .begin_function("VecPtr_pop")
            .param("vec", ptr_u8.clone())
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_pop")
            .expect("rayzor_vec_ptr_pop not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_get(vec: *VecPtr, index: i64) -> *u8
    {
        let func_id = builder
            .begin_function("VecPtr_get")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_get")
            .expect("rayzor_vec_ptr_get not found");
        let result = builder.call(extern_id, vec![vec, index]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_set(vec: *VecPtr, index: i64, value: *u8)
    {
        let func_id = builder
            .begin_function("VecPtr_set")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .param("value", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let value = builder.get_param(2);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_set")
            .expect("rayzor_vec_ptr_set not found");
        let _ = builder.call(extern_id, vec![vec, index, value]);
        builder.ret(None);
    }

    // VecPtr_length(vec: *VecPtr) -> i64
    {
        let func_id = builder
            .begin_function("VecPtr_length")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_len")
            .expect("rayzor_vec_ptr_len not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_isEmpty(vec: *VecPtr) -> bool
    {
        let func_id = builder
            .begin_function("VecPtr_isEmpty")
            .param("vec", ptr_u8.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_is_empty")
            .expect("rayzor_vec_ptr_is_empty not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_clear(vec: *VecPtr)
    {
        let func_id = builder
            .begin_function("VecPtr_clear")
            .param("vec", ptr_u8.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_clear")
            .expect("rayzor_vec_ptr_clear not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }

    // VecPtr_first(vec: *VecPtr) -> *u8
    {
        let func_id = builder
            .begin_function("VecPtr_first")
            .param("vec", ptr_u8.clone())
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_first")
            .expect("rayzor_vec_ptr_first not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_last(vec: *VecPtr) -> *u8
    {
        let func_id = builder
            .begin_function("VecPtr_last")
            .param("vec", ptr_u8.clone())
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_last")
            .expect("rayzor_vec_ptr_last not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecPtr_sortBy(vec: *VecPtr, compare: closure)
    {
        let func_id = builder
            .begin_function("VecPtr_sortBy")
            .param("vec", ptr_u8.clone())
            .param("compare", ptr_u8.clone())
            .returns(void_ty)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        // The comparator is a closure record {fn_ptr, env}, as array_sort takes.
        let closure = builder.get_param(1);
        let compare_fn = builder.load(closure, ptr_u8.clone());
        let offset_8 = builder.const_i64(8);
        let env_slot = builder.ptr_add(closure, offset_8, ptr_u8.clone());
        let compare_env = builder.load(env_slot, ptr_u8.clone());
        let extern_id = builder
            .get_function_by_name("rayzor_vec_ptr_sort_by")
            .expect("rayzor_vec_ptr_sort_by not found");
        let _ = builder.call(extern_id, vec![vec, compare_fn, compare_env]);
        builder.ret(None);
    }
}

/// Build VecBool MIR wrappers
fn build_vec_bool_wrappers(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i64_ty = IrType::I64;
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // VecBool_new() -> *VecBool
    {
        let func_id = builder
            .begin_function("VecBool_new")
            .returns(ptr_u8.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_new")
            .expect("rayzor_vec_bool_new not found");
        let result = builder.call(extern_id, vec![]).unwrap();
        builder.ret(Some(result));
    }

    // VecBool_push(vec: *VecBool, value: bool)
    {
        let func_id = builder
            .begin_function("VecBool_push")
            .param("vec", ptr_u8.clone())
            .param("value", bool_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let value = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_push")
            .expect("rayzor_vec_bool_push not found");
        let _ = builder.call(extern_id, vec![vec, value]);
        builder.ret(None);
    }

    // VecBool_pop(vec: *VecBool) -> bool
    {
        let func_id = builder
            .begin_function("VecBool_pop")
            .param("vec", ptr_u8.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_pop")
            .expect("rayzor_vec_bool_pop not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecBool_get(vec: *VecBool, index: i64) -> bool
    {
        let func_id = builder
            .begin_function("VecBool_get")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_get")
            .expect("rayzor_vec_bool_get not found");
        let result = builder.call(extern_id, vec![vec, index]).unwrap();
        builder.ret(Some(result));
    }

    // VecBool_set(vec: *VecBool, index: i64, value: bool)
    {
        let func_id = builder
            .begin_function("VecBool_set")
            .param("vec", ptr_u8.clone())
            .param("index", i64_ty.clone())
            .param("value", bool_ty.clone())
            .returns(void_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let index = builder.get_param(1);
        let value = builder.get_param(2);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_set")
            .expect("rayzor_vec_bool_set not found");
        let _ = builder.call(extern_id, vec![vec, index, value]);
        builder.ret(None);
    }

    // VecBool_length(vec: *VecBool) -> i64
    {
        let func_id = builder
            .begin_function("VecBool_length")
            .param("vec", ptr_u8.clone())
            .returns(i64_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_len")
            .expect("rayzor_vec_bool_len not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecBool_isEmpty(vec: *VecBool) -> bool
    {
        let func_id = builder
            .begin_function("VecBool_isEmpty")
            .param("vec", ptr_u8.clone())
            .returns(bool_ty.clone())
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_is_empty")
            .expect("rayzor_vec_bool_is_empty not found");
        let result = builder.call(extern_id, vec![vec]).unwrap();
        builder.ret(Some(result));
    }

    // VecBool_clear(vec: *VecBool)
    {
        let func_id = builder
            .begin_function("VecBool_clear")
            .param("vec", ptr_u8.clone())
            .returns(void_ty)
            .build();
        builder.set_current_function(func_id);
        let entry = builder.create_block("entry");
        builder.set_insert_point(entry);
        let vec = builder.get_param(0);
        let extern_id = builder
            .get_function_by_name("rayzor_vec_bool_clear")
            .expect("rayzor_vec_bool_clear not found");
        let _ = builder.call(extern_id, vec![vec]);
        builder.ret(None);
    }
}
