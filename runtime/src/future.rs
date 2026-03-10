//! Future Runtime Implementation
//!
//! Provides lazy futures for async/await support. A Future stores a closure
//! (fn_ptr + env_ptr) but does NOT spawn a thread until `.await()` or `.then()`
//! is called. This is tokio-style lazy evaluation.
//!
//! # API
//!
//! - `rayzor_future_create(fn_ptr, env_ptr) -> handle` — create lazy future
//! - `rayzor_future_await(handle) -> i64` — spawn if pending, block until resolved
//! - `rayzor_future_then(handle, cb_fn, cb_env)` — non-blocking: register callback + spawn
//! - `rayzor_future_poll(handle) -> i64` — non-blocking check
//! - `rayzor_future_is_ready(handle) -> bool` — check if resolved

use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread;

use crate::concurrency::{arm64_jit_barrier, ACTIVE_THREAD_COUNT};

/// Future states
const STATE_PENDING: u8 = 0;
const STATE_RUNNING: u8 = 1;
const STATE_RESOLVED: u8 = 2;

/// A lazy future handle.
///
/// Stores a closure (fn_ptr + env_ptr) and doesn't execute until
/// `.await()` or `.then()` is called.
struct FutureHandle {
    /// Current state: Pending → Running → Resolved
    state: AtomicU8,
    /// Stored closure function pointer (called on worker thread)
    fn_ptr: *const u8,
    /// Stored closure environment pointer
    env_ptr: *const u8,
    /// Resolved value (valid only when state == Resolved)
    value: Mutex<i64>,
    /// Condvar for blocking .await() callers
    cvar: Condvar,
    /// .then() callback function pointer (null if none)
    then_fn: AtomicPtr<u8>,
    /// .then() callback environment pointer
    then_env: AtomicPtr<u8>,
}

// Safety: FutureHandle fields are either atomic or behind a Mutex.
// fn_ptr/env_ptr are raw pointers stored at creation time and only read
// by the worker thread — no concurrent mutation.
unsafe impl Send for FutureHandle {}
unsafe impl Sync for FutureHandle {}

/// Spawn the future's closure on a worker thread.
///
/// Transitions state from Pending → Running. The worker thread calls
/// `fn_ptr(env_ptr)`, stores the result, transitions to Resolved,
/// notifies condvar, and calls any registered .then() callback.
fn spawn_future(handle: &FutureHandle, handle_ptr: *mut FutureHandle) {
    let func_addr = handle.fn_ptr as usize;
    let env_addr = handle.env_ptr as usize;
    let handle_addr = handle_ptr as usize;

    ACTIVE_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
    arm64_jit_barrier();

    thread::spawn(move || {
        arm64_jit_barrier();

        // Call the closure — JIT closures return i32 (same convention as thread spawn)
        type ClosureFn = extern "C" fn(*const u8) -> i32;
        let func: ClosureFn = unsafe { std::mem::transmute(func_addr) };
        let env_ptr = env_addr as *const u8;
        let result = func(env_ptr) as i64;

        // Resolve the future
        let handle = unsafe { &*(handle_addr as *const FutureHandle) };
        {
            let mut val = handle.value.lock().unwrap();
            *val = result;
        }
        handle.state.store(STATE_RESOLVED, Ordering::Release);
        handle.cvar.notify_all();

        // Call .then() callback if registered
        let then_fn = handle.then_fn.load(Ordering::Acquire);
        if !then_fn.is_null() {
            let then_env = handle.then_env.load(Ordering::Acquire);
            type CallbackFn = extern "C" fn(*const u8, i64);
            let callback: CallbackFn = unsafe { std::mem::transmute(then_fn as usize) };
            callback(then_env, result);
        }

        ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
    });
}

/// Create a lazy future that stores a closure but does NOT execute it.
///
/// # Arguments
/// * `fn_ptr` - Function pointer for the closure body
/// * `env_ptr` - Environment pointer (captured variables), may be null
///
/// # Returns
/// Opaque handle to the future (must be freed when no longer needed)
#[no_mangle]
pub extern "C" fn rayzor_future_create(fn_ptr: *const u8, env_ptr: *const u8) -> *mut u8 {
    if fn_ptr.is_null() {
        return ptr::null_mut();
    }

    let handle = Box::new(FutureHandle {
        state: AtomicU8::new(STATE_PENDING),
        fn_ptr,
        env_ptr,
        value: Mutex::new(0),
        cvar: Condvar::new(),
        then_fn: AtomicPtr::new(ptr::null_mut()),
        then_env: AtomicPtr::new(ptr::null_mut()),
    });

    Box::into_raw(handle) as *mut u8
}

/// Await a future: spawn if pending, block until resolved, return value.
///
/// - If Pending: spawn worker thread, then block on condvar
/// - If Running: block on condvar
/// - If Resolved: return value immediately
///
/// Returns boxed DynamicValue* (via haxe_box_int_ptr) so trace() can unbox it.
///
/// # Safety
/// `handle` must be a valid pointer from `rayzor_future_create`
#[no_mangle]
pub extern "C" fn rayzor_future_await(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    // Try to transition Pending → Running (we do the spawn)
    if future
        .state
        .compare_exchange(STATE_PENDING, STATE_RUNNING, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        spawn_future(future, handle as *mut FutureHandle);
    }

    // Wait for resolution
    let mut val = future.value.lock().unwrap();
    while future.state.load(Ordering::Acquire) != STATE_RESOLVED {
        val = future.cvar.wait(val).unwrap();
    }
    // Box the result as DynamicValue* (same convention as thread_join)
    crate::type_system::haxe_box_int_ptr(*val)
}

/// Register a callback and spawn the future (non-blocking).
///
/// The callback is called with `(env_ptr, result_value)` when the future resolves.
/// If already resolved, the callback is called immediately on the current thread.
///
/// # Arguments
/// * `handle` - Future handle
/// * `cb_fn` - Callback function pointer: `extern "C" fn(*const u8, i64)`
/// * `cb_env` - Callback environment pointer
#[no_mangle]
pub extern "C" fn rayzor_future_then(handle: *mut u8, cb_fn: *const u8, cb_env: *const u8) {
    // Debug removed
    if handle.is_null() || cb_fn.is_null() {
        return;
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    // Store the callback
    future
        .then_fn
        .store(cb_fn as *mut u8, Ordering::Release);
    future
        .then_env
        .store(cb_env as *mut u8, Ordering::Release);

    let current_state = future.state.load(Ordering::Acquire);

    if current_state == STATE_RESOLVED {
        // Already resolved — call callback immediately
        let result = *future.value.lock().unwrap();
        type CallbackFn = extern "C" fn(*const u8, i64);
        let callback: CallbackFn = unsafe { std::mem::transmute(cb_fn as usize) };
        callback(cb_env, result);
    } else if current_state == STATE_PENDING {
        // Spawn the future
        if future
            .state
            .compare_exchange(
                STATE_PENDING,
                STATE_RUNNING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            spawn_future(future, handle as *mut FutureHandle);
        }
    }
    // If Running: worker thread will call the callback when done
}

/// Poll a future (non-blocking).
///
/// # Returns
/// - If resolved: boxed result value (DynamicValue*)
/// - If not resolved: null
#[no_mangle]
pub extern "C" fn rayzor_future_poll(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    if future.state.load(Ordering::Acquire) == STATE_RESOLVED {
        crate::type_system::haxe_box_int_ptr(*future.value.lock().unwrap())
    } else {
        ptr::null_mut()
    }
}

/// Check if a future has resolved.
///
/// # Returns
/// `true` if the future has resolved and the value is available
#[no_mangle]
pub extern "C" fn rayzor_future_is_ready(handle: *const u8) -> bool {
    if handle.is_null() {
        return false;
    }

    let future = unsafe { &*(handle as *const FutureHandle) };
    future.state.load(Ordering::Acquire) == STATE_RESOLVED
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test closures use i32 return type to match JIT convention
    extern "C" fn simple_return_42(_env: *const u8) -> i32 {
        42
    }

    extern "C" fn add_env_values(env: *const u8) -> i32 {
        unsafe {
            let a = *(env as *const i64);
            let b = *((env as *const i64).add(1));
            (a + b) as i32
        }
    }

    /// Unbox a DynamicValue* to get the raw i64 value
    fn unbox_result(boxed: *mut u8) -> i64 {
        if boxed.is_null() {
            return 0;
        }
        crate::type_system::haxe_unbox_int_ptr(boxed)
    }

    #[test]
    fn test_create_and_await() {
        let handle =
            rayzor_future_create(simple_return_42 as *const u8, ptr::null());
        assert!(!handle.is_null());

        let result = unbox_result(rayzor_future_await(handle));
        assert_eq!(result, 42);

        unsafe {
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }

    #[test]
    fn test_lazy_not_started() {
        let handle =
            rayzor_future_create(simple_return_42 as *const u8, ptr::null());
        assert!(!handle.is_null());

        // Should be pending (not started)
        assert!(!rayzor_future_is_ready(handle));

        // Now await to actually run it
        let result = unbox_result(rayzor_future_await(handle));
        assert_eq!(result, 42);
        assert!(rayzor_future_is_ready(handle));

        unsafe {
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }

    #[test]
    fn test_with_environment() {
        let env: [i64; 2] = [10, 20];
        let handle = rayzor_future_create(
            add_env_values as *const u8,
            env.as_ptr() as *const u8,
        );

        let result = unbox_result(rayzor_future_await(handle));
        assert_eq!(result, 30);

        unsafe {
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }

    #[test]
    fn test_multiple_futures() {
        let h1 =
            rayzor_future_create(simple_return_42 as *const u8, ptr::null());
        let h2 =
            rayzor_future_create(simple_return_42 as *const u8, ptr::null());

        let r1 = unbox_result(rayzor_future_await(h1));
        let r2 = unbox_result(rayzor_future_await(h2));
        assert_eq!(r1 + r2, 84);

        unsafe {
            drop(Box::from_raw(h1 as *mut FutureHandle));
            drop(Box::from_raw(h2 as *mut FutureHandle));
        }
    }

    #[test]
    fn test_null_handle() {
        assert!(rayzor_future_await(ptr::null_mut()).is_null());
        assert!(rayzor_future_poll(ptr::null_mut()).is_null());
        assert!(!rayzor_future_is_ready(ptr::null()));
    }
}
