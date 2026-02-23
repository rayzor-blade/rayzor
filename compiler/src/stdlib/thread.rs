/// Thread: Lightweight concurrent execution primitives
///
/// This module provides MIR implementations for thread operations.
/// The actual threading is delegated to extern runtime functions.
///
/// Memory layout:
/// ```ignore
/// struct Thread<T> {
///     handle: *u8,    // Opaque OS thread handle
///     result: *T,     // Pointer to result (set when joined)
///     state: u8,      // 0=running, 1=finished, 2=joined
/// }
/// ```
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrFunctionId, IrType, Linkage};

/// Build all Thread functions
pub fn build_thread_type(builder: &mut MirBuilder) {
    // Declare extern runtime functions first
    declare_thread_externs(builder);

    // Build wrapper functions
    build_thread_spawn(builder);
    build_thread_join(builder);
    build_thread_is_finished(builder);
    build_thread_yield_now(builder);
    build_thread_sleep(builder);
    build_thread_current_id(builder);

    // Lock wrappers (Lock is backed by semaphore with initial count 0)
    build_lock_init(builder);
    build_lock_wait(builder);
    build_lock_wait_timeout(builder);

    // Semaphore wrappers
    build_semaphore_try_acquire(builder);
    build_semaphore_try_acquire_timeout(builder);
}

/// Declare extern runtime functions
fn declare_thread_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let i32_ty = builder.i32_type();
    let u64_ty = builder.u64_type();
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    // extern fn rayzor_thread_spawn(closure: *u8, closure_env: *u8) -> *u8
    let func_id = builder
        .begin_function("rayzor_thread_spawn")
        .param("closure", ptr_u8.clone())
        .param("closure_env", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_thread_join(handle: *u8) -> *u8
    let func_id = builder
        .begin_function("rayzor_thread_join")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_thread_is_finished(handle: *u8) -> bool
    let func_id = builder
        .begin_function("rayzor_thread_is_finished")
        .param("handle", ptr_u8.clone())
        .returns(bool_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_thread_yield_now()
    let func_id = builder
        .begin_function("rayzor_thread_yield_now")
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_thread_sleep(millis: i32)
    let func_id = builder
        .begin_function("rayzor_thread_sleep")
        .param("millis", i32_ty)
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_thread_current_id() -> u64
    let func_id = builder
        .begin_function("rayzor_thread_current_id")
        .returns(u64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_semaphore_init(initial_value: i32) -> *u8
    let i32_ty_clone = builder.i32_type();
    let func_id = builder
        .begin_function("rayzor_semaphore_init")
        .param("initial_value", i32_ty_clone)
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_semaphore_acquire(semaphore: *u8)
    let func_id = builder
        .begin_function("rayzor_semaphore_acquire")
        .param("semaphore", ptr_u8.clone())
        .returns(void_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn rayzor_semaphore_try_acquire(semaphore: *u8, timeout: f64) -> bool
    let func_id = builder
        .begin_function("rayzor_semaphore_try_acquire")
        .param("semaphore", ptr_u8.clone())
        .param("timeout", IrType::F64)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    // extern fn sys_semaphore_try_acquire_nowait(semaphore: *u8) -> bool
    let bool_ty_clone = builder.bool_type();
    let func_id = builder
        .begin_function("sys_semaphore_try_acquire_nowait")
        .param("semaphore", ptr_u8.clone())
        .returns(bool_ty_clone)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Build: fn Thread_spawn(closure_obj: *u8) -> *Thread
/// The closure_obj is a pointer to a struct { fn_ptr: *u8, env_ptr: *u8 }
fn build_thread_spawn(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Thread_spawn")
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

    // Call runtime function with extracted pointers
    let spawn_id = builder
        .get_function_by_name("rayzor_thread_spawn")
        .expect("rayzor_thread_spawn not found");
    let handle = builder.call(spawn_id, vec![fn_ptr, env_ptr]).unwrap();

    builder.ret(Some(handle));
}

/// Build: fn Thread_join(handle: *Thread) -> *u8 (i64)
/// TODO: This should be generic Thread<T>.join() -> T
/// For now it returns i64 and relies on caller to cast to correct type
fn build_thread_join(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Thread_join")
        .param("handle", ptr_u8.clone())
        .returns(ptr_u8.clone()) // Return ptr_u8 (i64) to match rayzor_thread_join
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    // Call runtime function (returns *u8 which is i64)
    let join_id = builder
        .get_function_by_name("rayzor_thread_join")
        .expect("rayzor_thread_join not found");
    let result_ptr = builder.call(join_id, vec![handle]).unwrap();

    // TODO: The runtime returns i64, but we declared this function as returning i32.
    // Ideally we'd cast here, but the function signature already says i32, so the
    // caller will handle the truncation at the call site when the types don't match.
    // Just return the i64 value and let type checking insert the cast later.
    builder.ret(Some(result_ptr));
}

/// Build: fn Thread_isFinished(handle: *Thread) -> bool
fn build_thread_is_finished(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Thread_isFinished")
        .param("handle", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    // Call runtime function
    let is_finished_id = builder
        .get_function_by_name("rayzor_thread_is_finished")
        .expect("rayzor_thread_is_finished not found");
    let result = builder.call(is_finished_id, vec![handle]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Thread_yieldNow()
fn build_thread_yield_now(builder: &mut MirBuilder) {
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("Thread_yieldNow")
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Call runtime function
    let yield_id = builder
        .get_function_by_name("rayzor_thread_yield_now")
        .expect("rayzor_thread_yield_now not found");
    let _result = builder.call(yield_id, vec![]);

    builder.ret(None);
}

/// Build: fn Thread_sleep(millis: i32)
fn build_thread_sleep(builder: &mut MirBuilder) {
    let i32_ty = builder.i32_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("Thread_sleep")
        .param("millis", i32_ty)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let millis = builder.get_param(0);

    // Call runtime function
    let sleep_id = builder
        .get_function_by_name("rayzor_thread_sleep")
        .expect("rayzor_thread_sleep not found");
    let _result = builder.call(sleep_id, vec![millis]);

    builder.ret(None);
}

/// Build: fn Thread_currentId() -> u64
fn build_thread_current_id(builder: &mut MirBuilder) {
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("Thread_currentId")
        .returns(u64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Call runtime function
    let current_id_fn = builder
        .get_function_by_name("rayzor_thread_current_id")
        .expect("rayzor_thread_current_id not found");
    let result = builder.call(current_id_fn, vec![]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Lock_init() -> *u8
/// Lock is backed by a semaphore initialized with count 0
fn build_lock_init(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Lock_init")
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    // Call rayzor_semaphore_init(0) to create a semaphore with initial count 0
    let semaphore_init_id = builder
        .get_function_by_name("rayzor_semaphore_init")
        .expect("rayzor_semaphore_init not found");
    let zero = builder.const_i32(0);
    let handle = builder.call(semaphore_init_id, vec![zero]).unwrap();

    builder.ret(Some(handle));
}

/// Build: fn Lock_wait(handle: *u8) -> bool
/// Blocking wait with no timeout - uses semaphore acquire then returns true
fn build_lock_wait(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Lock_wait")
        .param("handle", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    // Call rayzor_semaphore_acquire (blocking wait)
    let acquire_id = builder
        .get_function_by_name("rayzor_semaphore_acquire")
        .expect("rayzor_semaphore_acquire not found");
    let _ = builder.call(acquire_id, vec![handle]);

    // Always return true (blocking wait always succeeds)
    let true_val = builder.const_bool(true);
    builder.ret(Some(true_val));
}

/// Build: fn Lock_wait_timeout(handle: *u8, timeout: f64) -> bool
/// Wait with timeout - uses semaphore try_acquire
fn build_lock_wait_timeout(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let f64_ty = IrType::F64;
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Lock_wait_timeout")
        .param("handle", ptr_u8.clone())
        .param("timeout", f64_ty)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);
    let timeout = builder.get_param(1);

    // Call rayzor_semaphore_try_acquire(handle, timeout)
    let try_acquire_id = builder
        .get_function_by_name("rayzor_semaphore_try_acquire")
        .expect("rayzor_semaphore_try_acquire not found");
    let result = builder.call(try_acquire_id, vec![handle, timeout]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Semaphore_tryAcquire(handle: *u8) -> bool
/// Non-blocking acquire attempt (no timeout)
fn build_semaphore_try_acquire(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Semaphore_tryAcquire")
        .param("handle", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);

    // Call sys_semaphore_try_acquire_nowait(handle)
    let try_acquire_id = builder
        .get_function_by_name("sys_semaphore_try_acquire_nowait")
        .expect("sys_semaphore_try_acquire_nowait not found");
    let result = builder.call(try_acquire_id, vec![handle]).unwrap();

    builder.ret(Some(result));
}

/// Build: fn Semaphore_tryAcquire_timeout(handle: *u8, timeout: f64) -> bool
/// Acquire with timeout
fn build_semaphore_try_acquire_timeout(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let f64_ty = IrType::F64;
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Semaphore_tryAcquire_timeout")
        .param("handle", ptr_u8.clone())
        .param("timeout", f64_ty)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let handle = builder.get_param(0);
    let timeout = builder.get_param(1);

    // Call rayzor_semaphore_try_acquire(handle, timeout)
    let try_acquire_id = builder
        .get_function_by_name("rayzor_semaphore_try_acquire")
        .expect("rayzor_semaphore_try_acquire not found");
    let result = builder.call(try_acquire_id, vec![handle, timeout]).unwrap();

    builder.ret(Some(result));
}
