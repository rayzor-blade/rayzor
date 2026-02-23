/// Sync: Synchronization primitives (Arc, Mutex)
///
/// This module provides MIR implementations for synchronization operations.
/// The actual implementations are delegated to extern runtime functions.
///
/// Arc<T> memory layout:
/// ```ignore
/// struct Arc<T> {
///     inner: *u8,     // Pointer to ArcInner { strong: AtomicUsize, data: T }
/// }
/// ```
///
/// Mutex<T> memory layout:
/// ```ignore
/// struct Mutex<T> {
///     inner: *u8,     // Pointer to OS mutex + data
/// }
/// ```
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

/// Build all synchronization functions
pub fn build_sync_types(builder: &mut MirBuilder) {
    // Declare extern runtime functions first
    declare_arc_externs(builder);
    declare_mutex_externs(builder);

    // Build Arc functions
    build_arc_init(builder);
    build_arc_clone(builder);
    build_arc_get(builder);
    build_arc_strong_count(builder);
    build_arc_try_unwrap(builder);
    build_arc_as_ptr(builder);

    // Build Mutex functions
    build_mutex_init(builder);
    build_mutex_lock(builder);
    build_mutex_try_lock(builder);
    build_mutex_is_locked(builder);
    build_mutex_guard_get(builder);
    build_mutex_unlock(builder);
}

/// Declare extern Arc runtime functions
fn declare_arc_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("rayzor_arc_init")
        .param("value", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_arc_clone")
        .param("arc", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_arc_get")
        .param("arc", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_arc_strong_count")
        .param("arc", ptr_u8.clone())
        .returns(u64_ty.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_arc_try_unwrap")
        .param("arc", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_arc_as_ptr")
        .param("arc", ptr_u8.clone())
        .returns(u64_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

/// Declare extern Mutex runtime functions
fn declare_mutex_externs(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("rayzor_mutex_init")
        .param("value", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_mutex_lock")
        .param("mutex", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_mutex_try_lock")
        .param("mutex", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_mutex_is_locked")
        .param("mutex", ptr_u8.clone())
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_mutex_guard_get")
        .param("guard", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);

    let func_id = builder
        .begin_function("rayzor_mutex_unlock")
        .param("guard", ptr_u8.clone())
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();
    builder.mark_as_extern(func_id);
}

// ============================================================================
// Arc Functions
// ============================================================================

/// Build: fn Arc_init(value: *u8) -> *Arc
fn build_arc_init(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Arc_init")
        .param("value", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let value = builder.get_param(0);

    let new_id = builder
        .get_function_by_name("rayzor_arc_init")
        .expect("rayzor_arc_init not found");
    let arc = builder.call(new_id, vec![value]).unwrap();

    builder.ret(Some(arc));
}

/// Build: fn Arc_clone(arc: *Arc) -> *Arc
fn build_arc_clone(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Arc_clone")
        .param("arc", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arc = builder.get_param(0);

    let clone_id = builder
        .get_function_by_name("rayzor_arc_clone")
        .expect("rayzor_arc_clone not found");
    let cloned = builder.call(clone_id, vec![arc]).unwrap();

    builder.ret(Some(cloned));
}

/// Build: fn Arc_get(arc: *Arc) -> *T
fn build_arc_get(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Arc_get")
        .param("arc", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arc = builder.get_param(0);

    let get_id = builder
        .get_function_by_name("rayzor_arc_get")
        .expect("rayzor_arc_get not found");
    let value_ptr = builder.call(get_id, vec![arc]).unwrap();

    builder.ret(Some(value_ptr));
}

/// Build: fn Arc_strongCount(arc: *Arc) -> u64
fn build_arc_strong_count(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("Arc_strongCount")
        .param("arc", ptr_u8)
        .returns(u64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arc = builder.get_param(0);

    let count_id = builder
        .get_function_by_name("rayzor_arc_strong_count")
        .expect("rayzor_arc_strong_count not found");
    let count = builder.call(count_id, vec![arc]).unwrap();

    builder.ret(Some(count));
}

/// Build: fn Arc_tryUnwrap(arc: *Arc) -> *T (null if refcount > 1)
fn build_arc_try_unwrap(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Arc_tryUnwrap")
        .param("arc", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arc = builder.get_param(0);

    let unwrap_id = builder
        .get_function_by_name("rayzor_arc_try_unwrap")
        .expect("rayzor_arc_try_unwrap not found");
    let value_ptr = builder.call(unwrap_id, vec![arc]).unwrap();

    builder.ret(Some(value_ptr));
}

/// Build: fn Arc_asPtr(arc: *Arc) -> u64
fn build_arc_as_ptr(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let u64_ty = builder.u64_type();

    let func_id = builder
        .begin_function("Arc_asPtr")
        .param("arc", ptr_u8)
        .returns(u64_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let arc = builder.get_param(0);

    let as_ptr_id = builder
        .get_function_by_name("rayzor_arc_as_ptr")
        .expect("rayzor_arc_as_ptr not found");
    let ptr_val = builder.call(as_ptr_id, vec![arc]).unwrap();

    builder.ret(Some(ptr_val));
}

// ============================================================================
// Mutex Functions
// ============================================================================

/// Build: fn Mutex_init(value: *T) -> *Mutex
fn build_mutex_init(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Mutex_init")
        .param("value", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let value = builder.get_param(0);

    let new_id = builder
        .get_function_by_name("rayzor_mutex_init")
        .expect("rayzor_mutex_init not found");
    let mutex = builder.call(new_id, vec![value]).unwrap();

    builder.ret(Some(mutex));
}

/// Build: fn Mutex_lock(mutex: *Mutex) -> *MutexGuard
fn build_mutex_lock(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Mutex_lock")
        .param("mutex", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let mutex = builder.get_param(0);

    let lock_id = builder
        .get_function_by_name("rayzor_mutex_lock")
        .expect("rayzor_mutex_lock not found");
    let guard = builder.call(lock_id, vec![mutex]).unwrap();

    builder.ret(Some(guard));
}

/// Build: fn Mutex_tryLock(mutex: *Mutex) -> *MutexGuard (null if locked)
fn build_mutex_try_lock(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("Mutex_tryLock")
        .param("mutex", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let mutex = builder.get_param(0);

    let try_lock_id = builder
        .get_function_by_name("rayzor_mutex_try_lock")
        .expect("rayzor_mutex_try_lock not found");
    let guard = builder.call(try_lock_id, vec![mutex]).unwrap();

    builder.ret(Some(guard));
}

/// Build: fn Mutex_isLocked(mutex: *Mutex) -> bool
fn build_mutex_is_locked(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let bool_ty = builder.bool_type();

    let func_id = builder
        .begin_function("Mutex_isLocked")
        .param("mutex", ptr_u8)
        .returns(bool_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let mutex = builder.get_param(0);

    let is_locked_id = builder
        .get_function_by_name("rayzor_mutex_is_locked")
        .expect("rayzor_mutex_is_locked not found");
    let locked = builder.call(is_locked_id, vec![mutex]).unwrap();

    builder.ret(Some(locked));
}

/// Build: fn MutexGuard_get(guard: *MutexGuard) -> *T
fn build_mutex_guard_get(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());

    let func_id = builder
        .begin_function("MutexGuard_get")
        .param("guard", ptr_u8.clone())
        .returns(ptr_u8.clone())
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let guard = builder.get_param(0);

    let get_id = builder
        .get_function_by_name("rayzor_mutex_guard_get")
        .expect("rayzor_mutex_guard_get not found");
    let value_ptr = builder.call(get_id, vec![guard]).unwrap();

    builder.ret(Some(value_ptr));
}

/// Build: fn MutexGuard_unlock(guard: *MutexGuard)
fn build_mutex_unlock(builder: &mut MirBuilder) {
    let ptr_u8 = builder.ptr_type(builder.u8_type());
    let void_ty = builder.void_type();

    let func_id = builder
        .begin_function("MutexGuard_unlock")
        .param("guard", ptr_u8)
        .returns(void_ty)
        .calling_convention(CallingConvention::C)
        .build();

    builder.set_current_function(func_id);

    let entry = builder.create_block("entry");
    builder.set_insert_point(entry);

    let guard = builder.get_param(0);

    let unlock_id = builder
        .get_function_by_name("rayzor_mutex_unlock")
        .expect("rayzor_mutex_unlock not found");
    let _result = builder.call(unlock_id, vec![guard]);

    builder.ret(None);
}
