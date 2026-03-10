/// Future<T>: Lazy async futures
///
/// This module provides MIR implementations for Future operations.
/// Futures are lazy — they store a closure but don't execute until
/// `.await()` or `.then()` is called.
///
/// Memory layout:
/// ```ignore
/// struct Future<T> {
///     handle: *u8,    // Opaque FutureHandle pointer
/// }
/// ```
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all Future functions
pub fn build_future_type(builder: &mut MirBuilder) {
    declare_future_externs(builder);

    build_future_create(builder);
    build_future_await(builder);
    build_future_then(builder);
    build_future_poll(builder);
    build_future_is_ready(builder);
}

/// Declare extern runtime functions
fn declare_future_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();
    let bool_ty = builder.bool_type();
    let i64_ty = IrType::I64;

    // extern fn rayzor_future_create(fn_ptr: *u8, env_ptr: *u8) -> *u8
    let func_id = builder
        .begin_function("rayzor_future_create")
        .param("fn_ptr", ptr_u8.clone())
        .param("env_ptr", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_future_await(handle: *u8) -> *u8 (DynamicValue*)
    let func_id = builder
        .begin_function("rayzor_future_await")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_future_then(handle: *u8, cb_fn: *u8, cb_env: *u8)
    let func_id = builder
        .begin_function("rayzor_future_then")
        .param("handle", ptr_u8.clone())
        .param("cb_fn", ptr_u8.clone())
        .param("cb_env", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_future_poll(handle: *u8) -> *u8 (DynamicValue* or null)
    let func_id = builder
        .begin_function("rayzor_future_poll")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_future_is_ready(handle: *u8) -> bool
    let func_id = builder
        .begin_function("rayzor_future_is_ready")
        .param("handle", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Build: fn Future_create(closure_obj: *u8) -> *u8
/// Extracts fn_ptr and env_ptr from closure object, creates lazy future
fn build_future_create(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Future_create")
        .param("closure_obj", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let closure_obj = builder.get_param(0);

    // Extract function pointer from closure object (offset 0)
    let fn_ptr = builder.load(closure_obj, ptr_u8.clone());

    // Extract environment pointer from closure object (offset 8)
    let offset_8 = builder.const_i64(8);
    let env_ptr_addr = builder.ptr_add(closure_obj, offset_8, ptr_u8.clone());
    let env_ptr = builder.load(env_ptr_addr, ptr_u8.clone());

    // Call runtime function to create lazy future
    let create_id = builder
        .get_function_by_name("rayzor_future_create")
        .expect("rayzor_future_create not found");
    let handle = builder.call(create_id, vec![fn_ptr, env_ptr]).unwrap();

    builder.ret(Some(handle));
}

/// Build: fn Future_await(handle: *u8) -> *u8
/// Spawns future if pending, blocks until resolved, returns value
fn build_future_await(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Future_await")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    let await_id = builder
        .get_function_by_name("rayzor_future_await")
        .expect("rayzor_future_await not found");
    let result = builder.call(await_id, vec![handle]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Future_then(handle: *u8, callback_closure: *u8)
/// Extracts callback fn_ptr and env_ptr from closure, registers callback
fn build_future_then(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("Future_then")
        .param("handle", ptr_u8.clone())
        .param("callback_closure", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);
    let callback_closure = builder.get_param(1);

    // Extract callback function pointer from closure object (offset 0)
    let cb_fn_ptr = builder.load(callback_closure, ptr_u8.clone());

    // Extract callback environment pointer from closure object (offset 8)
    let offset_8 = builder.const_i64(8);
    let cb_env_addr = builder.ptr_add(callback_closure, offset_8, ptr_u8.clone());
    let cb_env_ptr = builder.load(cb_env_addr, ptr_u8.clone());

    // Call runtime function
    let then_id = builder
        .get_function_by_name("rayzor_future_then")
        .expect("rayzor_future_then not found");
    let _ = builder.call(then_id, vec![handle, cb_fn_ptr, cb_env_ptr]);

    builder.ret(None);
}

/// Build: fn Future_poll(handle: *u8) -> *u8
/// Non-blocking check; returns value if resolved, 0 if pending
fn build_future_poll(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Future_poll")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    let poll_id = builder
        .get_function_by_name("rayzor_future_poll")
        .expect("rayzor_future_poll not found");
    let result = builder.call(poll_id, vec![handle]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Future_isReady(handle: *u8) -> bool
fn build_future_is_ready(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Future_isReady")
        .param("handle", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    let is_ready_id = builder
        .get_function_by_name("rayzor_future_is_ready")
        .expect("rayzor_future_is_ready not found");
    let result = builder.call(is_ready_id, vec![handle]).unwrap();

    builder.ret(Some(result));
}
