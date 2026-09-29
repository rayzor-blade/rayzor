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
//! - `rayzor_plugin_future_pending(raw) -> handle` — native callback future
//! - `rayzor_plugin_future_resolve(handle, value)` — complete from a plugin
//! - `rayzor_plugin_future_reject(handle, utf8, len)` — reject from a plugin

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU32, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread;

use crate::concurrency::{ACTIVE_THREAD_COUNT, arm64_jit_barrier};

unsafe extern "C" {
    fn _setjmp(buf: *mut u8) -> i32;
}

/// Future states
const STATE_PENDING: u8 = 0;
const STATE_RUNNING: u8 = 1;
const STATE_RESOLVED: u8 = 2;
const STATE_SETTLING: u8 = 3;

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
    /// If true, value is a pre-boxed/raw pointer — await returns it directly
    /// without wrapping in haxe_box_int_ptr. Used by Future.all().
    raw_result: bool,
    /// True if the closure threw an exception or panicked
    has_error: AtomicBool,
    /// Exception value (pointer to Exception object) — valid when has_error is true
    error_value: Mutex<i64>,
    /// Exception type_id for typed re-throw
    error_type_id: AtomicU32,
    /// Error text supplied by a native callback. It is turned into a Haxe
    /// exception on the awaiting language thread.
    error_message: Mutex<Option<String>>,
    /// Cooperative cancellation flag — checked by worker thread
    cancelled: AtomicBool,
}

// Safety: FutureHandle fields are either atomic or behind a Mutex.
// fn_ptr/env_ptr are raw pointers stored at creation time and only read
// by the worker thread — no concurrent mutation.
unsafe impl Send for FutureHandle {}
unsafe impl Sync for FutureHandle {}

fn callback_value(future: &FutureHandle, value: i64) -> *mut u8 {
    if future.raw_result {
        value as *mut u8
    } else {
        crate::type_system::haxe_box_int_ptr(value)
    }
}

fn notify_then(future: &FutureHandle, value: i64) {
    let then_fn = future.then_fn.load(Ordering::Acquire);
    if then_fn.is_null() {
        return;
    }
    let then_env = future.then_env.load(Ordering::Acquire);
    type CallbackFn = extern "C" fn(*const u8, *mut u8);
    let callback: CallbackFn = unsafe { std::mem::transmute(then_fn as usize) };
    callback(then_env, callback_value(future, value));
}

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

        // Call the closure — wrapped in catch_unwind so panics don't leak ACTIVE_THREAD_COUNT.
        // Inside, we also install a setjmp handler to catch Haxe exceptions (which use longjmp).
        type ClosureFn = extern "C" fn(*const u8) -> i64;
        let func: ClosureFn = unsafe { std::mem::transmute(func_addr) };
        let env_ptr = env_addr as *const u8;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Install Haxe exception handler on this worker thread
            let jmp_buf = crate::exception::rayzor_exception_push_handler();
            let caught = unsafe { _setjmp(jmp_buf) };

            if caught == 0 {
                // Normal path — call the closure
                let value = func(env_ptr);
                crate::exception::rayzor_exception_pop_handler();
                Ok(value)
            } else {
                // Exception was thrown via longjmp — capture it
                let exc_value = crate::exception::rayzor_get_exception();
                let exc_type_id = crate::exception::rayzor_get_exception_type_id();
                Err((exc_value, exc_type_id))
            }
        }));

        let handle = unsafe { &*(handle_addr as *const FutureHandle) };

        match result {
            Ok(Ok(value)) => {
                // Resolve the future with the computed value
                {
                    let mut val = handle.value.lock().unwrap();
                    *val = value;
                }
                handle.state.store(STATE_RESOLVED, Ordering::Release);
                handle.cvar.notify_all();

                notify_then(handle, value);
            }
            Ok(Err((exc_value, exc_type_id))) => {
                // Haxe exception was thrown — store for re-throw on .await()
                {
                    let mut ev = handle.error_value.lock().unwrap();
                    *ev = exc_value;
                }
                handle.error_type_id.store(exc_type_id, Ordering::Release);
                handle.has_error.store(true, Ordering::Release);
                handle.state.store(STATE_RESOLVED, Ordering::Release);
                handle.cvar.notify_all();
            }
            Err(_panic) => {
                // Rust panic — resolve with error so .await() doesn't hang
                handle.has_error.store(true, Ordering::Release);
                handle.state.store(STATE_RESOLVED, Ordering::Release);
                handle.cvar.notify_all();
            }
        }

        // ALWAYS decrement, even on panic/exception
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
#[unsafe(no_mangle)]
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
        raw_result: false,
        has_error: AtomicBool::new(false),
        error_value: Mutex::new(0),
        error_type_id: AtomicU32::new(0),
        error_message: Mutex::new(None),
        cancelled: AtomicBool::new(false),
    });

    Box::into_raw(handle) as *mut u8
}

/// Create a future whose result will be supplied by a native plugin callback.
///
/// A plugin future starts in `Running`, so awaiting or attaching `then()` never
/// tries to spawn a Haxe closure. `raw_result` selects whether the resolved i64
/// is already a language object pointer or should be boxed as an integer.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_future_pending(raw_result: u8) -> *mut u8 {
    let handle = Box::new(FutureHandle {
        state: AtomicU8::new(STATE_RUNNING),
        fn_ptr: ptr::null(),
        env_ptr: ptr::null(),
        value: Mutex::new(0),
        cvar: Condvar::new(),
        then_fn: AtomicPtr::new(ptr::null_mut()),
        then_env: AtomicPtr::new(ptr::null_mut()),
        raw_result: raw_result != 0,
        has_error: AtomicBool::new(false),
        error_value: Mutex::new(0),
        error_type_id: AtomicU32::new(0),
        error_message: Mutex::new(None),
        cancelled: AtomicBool::new(false),
    });
    Box::into_raw(handle) as *mut u8
}

/// Resolve a plugin-created future. Returns 1 for the first completion.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_future_resolve(handle: *mut u8, value: i64) -> u8 {
    if handle.is_null() {
        return 0;
    }
    let future = unsafe { &*(handle as *const FutureHandle) };
    if future
        .state
        .compare_exchange(
            STATE_RUNNING,
            STATE_SETTLING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return 0;
    }
    *future.value.lock().unwrap() = value;
    future.state.store(STATE_RESOLVED, Ordering::Release);
    future.cvar.notify_all();
    notify_then(future, value);
    1
}

/// Reject a plugin-created future with a borrowed UTF-8 message.
///
/// The bytes are copied into the Future because the callback's source storage
/// does not need to outlive this call. The Haxe exception is allocated only
/// when the language thread awaits the result.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_plugin_future_reject(
    handle: *mut u8,
    message: *const u8,
    message_len: usize,
) -> u8 {
    if handle.is_null() || (message.is_null() && message_len != 0) {
        return 0;
    }
    let future = unsafe { &*(handle as *const FutureHandle) };
    if future
        .state
        .compare_exchange(
            STATE_RUNNING,
            STATE_SETTLING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return 0;
    }
    let text = if message_len == 0 {
        String::new()
    } else {
        let bytes = unsafe { std::slice::from_raw_parts(message, message_len) };
        String::from_utf8_lossy(bytes).into_owned()
    };
    *future.error_message.lock().unwrap() = Some(text);
    future.has_error.store(true, Ordering::Release);
    future.state.store(STATE_RESOLVED, Ordering::Release);
    future.cvar.notify_all();
    1
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
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_await(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    // Try to transition Pending → Running (we do the spawn)
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

    // Wait for resolution
    let mut val = future.value.lock().unwrap();
    while future.state.load(Ordering::Acquire) != STATE_RESOLVED {
        val = future.cvar.wait(val).unwrap();
    }

    // Check if the closure threw an exception — re-throw in caller's context
    if future.has_error.load(Ordering::Acquire) {
        if let Some(message) = future.error_message.lock().unwrap().clone() {
            crate::exception::throw_with_message(message);
        }
        let exc_value = *future.error_value.lock().unwrap();
        let exc_type_id = future.error_type_id.load(Ordering::Acquire);
        if exc_value != 0 {
            // Re-throw the captured Haxe exception
            crate::exception::rayzor_throw_typed(exc_value, exc_type_id);
        }
        // Rust panic with no exception info — return null
        return ptr::null_mut();
    }

    if future.raw_result {
        // Value is already a usable pointer (e.g., Array* from Future.all)
        *val as *mut u8
    } else {
        // Box the result as DynamicValue* (same convention as thread_join)
        crate::type_system::haxe_box_int_ptr(*val)
    }
}

/// Await a future with a timeout in milliseconds.
///
/// Same as `rayzor_future_await` but returns null if the timeout expires.
/// Uses `condvar.wait_timeout` for efficient waiting.
///
/// # Returns
/// - Boxed DynamicValue* on success
/// - null pointer on timeout or cancellation
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_await_timeout(handle: *mut u8, millis: i64) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    // Try to transition Pending → Running
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

    // Wait with timeout
    let timeout = std::time::Duration::from_millis(millis as u64);
    let deadline = std::time::Instant::now() + timeout;
    let mut val = future.value.lock().unwrap();

    while future.state.load(Ordering::Acquire) != STATE_RESOLVED {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return ptr::null_mut(); // Timeout
        }
        let (guard, wait_result) = future.cvar.wait_timeout(val, remaining).unwrap();
        val = guard;
        if wait_result.timed_out() {
            return ptr::null_mut(); // Timeout
        }
    }

    // Check for error
    if future.has_error.load(Ordering::Acquire) {
        if let Some(message) = future.error_message.lock().unwrap().clone() {
            crate::exception::throw_with_message(message);
        }
        let exc_value = *future.error_value.lock().unwrap();
        let exc_type_id = future.error_type_id.load(Ordering::Acquire);
        if exc_value != 0 {
            crate::exception::rayzor_throw_typed(exc_value, exc_type_id);
        }
        return ptr::null_mut();
    }

    if future.raw_result {
        *val as *mut u8
    } else {
        crate::type_system::haxe_box_int_ptr(*val)
    }
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
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_then(handle: *mut u8, cb_fn: *const u8, cb_env: *const u8) {
    // Debug removed
    if handle.is_null() || cb_fn.is_null() {
        return;
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    // Store the callback
    future.then_fn.store(cb_fn as *mut u8, Ordering::Release);
    future.then_env.store(cb_env as *mut u8, Ordering::Release);

    let current_state = future.state.load(Ordering::Acquire);

    if current_state == STATE_RESOLVED {
        // Already resolved — call callback immediately
        let result = *future.value.lock().unwrap();
        type CallbackFn = extern "C" fn(*const u8, *mut u8);
        let callback: CallbackFn = unsafe { std::mem::transmute(cb_fn as usize) };
        callback(cb_env, callback_value(future, result));
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
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_poll(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let future = unsafe { &*(handle as *const FutureHandle) };

    if future.state.load(Ordering::Acquire) != STATE_RESOLVED
        || future.has_error.load(Ordering::Acquire)
    {
        return ptr::null_mut();
    }

    callback_value(future, *future.value.lock().unwrap())
}

/// Create a lazy Future that resolves to an Array of results from multiple futures.
///
/// Like JavaScript's `Promise.all()`: returns a Future that, when awaited,
/// spawns all sub-futures in parallel, awaits each, and returns a Haxe Array.
///
/// # Arguments
/// * `arr_ptr` - Pointer to a Haxe Array of future handles (elem_size=8)
///
/// # Returns
/// A new lazy Future handle. When awaited, resolves to a Haxe Array* of boxed results.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_all(arr_ptr: *const u8) -> *mut u8 {
    if arr_ptr.is_null() {
        return ptr::null_mut();
    }

    // Read the Haxe Array struct: { ptr, len, cap, elem_size }
    let arr = unsafe { &*(arr_ptr as *const crate::haxe_array::HaxeArray) };
    let n = arr.len;
    let data_ptr = arr.ptr;

    // Copy the sub-future handle pointers (they're i64-sized elements)
    let mut sub_handles: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let handle = unsafe { *(data_ptr.add(i * 8) as *const i64) } as usize;
        sub_handles.push(handle);
    }

    // Create the outer "all" future with raw_result=true so await returns
    // the Array* directly without int-boxing it.
    // State = Running because we eagerly spawn the coordinating thread.
    let handle = Box::new(FutureHandle {
        state: AtomicU8::new(STATE_RUNNING),
        fn_ptr: ptr::null(),
        env_ptr: ptr::null(),
        value: Mutex::new(0),
        cvar: Condvar::new(),
        then_fn: AtomicPtr::new(ptr::null_mut()),
        then_env: AtomicPtr::new(ptr::null_mut()),
        raw_result: true,
        has_error: AtomicBool::new(false),
        error_value: Mutex::new(0),
        error_type_id: AtomicU32::new(0),
        error_message: Mutex::new(None),
        cancelled: AtomicBool::new(false),
    });
    let handle_ptr = Box::into_raw(handle);
    let handle_addr = handle_ptr as usize;

    // Spawn a coordinating thread that:
    // 1. Spawns all sub-futures in parallel
    // 2. Awaits each in order
    // 3. Stores full 64-bit Array* result in outer FutureHandle
    ACTIVE_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
    crate::concurrency::arm64_jit_barrier();

    thread::spawn(move || {
        crate::concurrency::arm64_jit_barrier();

        let coordinating_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Phase 1: Spawn ALL sub-futures
            let mut fh_ptrs: Vec<*mut FutureHandle> = Vec::with_capacity(n);
            for &h_addr in &sub_handles {
                let h = h_addr as *mut FutureHandle;
                fh_ptrs.push(h);
                if h.is_null() {
                    continue;
                }
                let future = unsafe { &*h };
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
                    spawn_future(future, h);
                }
            }

            // Phase 2: Await each, build Haxe Array of raw results
            let mut result_arr = crate::haxe_array::HaxeArray {
                ptr: ptr::null_mut(),
                len: 0,
                cap: 0,
                elem_size: 8,
            };
            crate::haxe_array::haxe_array_new(&mut result_arr, 8);

            for &h in &fh_ptrs {
                let raw_val: i64 = if h.is_null() {
                    0
                } else {
                    let future = unsafe { &*h };
                    let mut val = future.value.lock().unwrap();
                    while future.state.load(Ordering::Acquire) != STATE_RESOLVED {
                        val = future.cvar.wait(val).unwrap();
                    }
                    *val
                };
                crate::haxe_array::haxe_array_push(
                    &mut result_arr,
                    &raw_val as *const i64 as *const u8,
                );
            }

            // Store full 64-bit Array* in the outer FutureHandle
            let arr_box = Box::new(result_arr);
            let arr_ptr_i64 = Box::into_raw(arr_box) as i64;
            let outer = unsafe { &*(handle_addr as *const FutureHandle) };
            {
                let mut val = outer.value.lock().unwrap();
                *val = arr_ptr_i64;
            }
            outer.state.store(STATE_RESOLVED, Ordering::Release);
            outer.cvar.notify_all();

            // Call .then() callback if registered
            let then_fn = outer.then_fn.load(Ordering::Acquire);
            if !then_fn.is_null() {
                let then_env = outer.then_env.load(Ordering::Acquire);
                type CallbackFn = extern "C" fn(*const u8, *mut u8);
                let callback: CallbackFn = unsafe { std::mem::transmute(then_fn as usize) };
                callback(then_env, arr_ptr_i64 as *mut u8);
            }
        }));

        if coordinating_result.is_err() {
            // Panic in coordinating thread — resolve outer handle so .await() doesn't hang
            let outer = unsafe { &*(handle_addr as *const FutureHandle) };
            outer.state.store(STATE_RESOLVED, Ordering::Release);
            outer.cvar.notify_all();
        }

        // ALWAYS decrement
        ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
    });

    handle_ptr as *mut u8
}

/// Race multiple futures — returns a Future that resolves with the first result.
///
/// Spawns all sub-futures in parallel. The outer Future resolves as soon as
/// ANY sub-future completes.
///
/// # Arguments
/// * `arr_ptr` - Pointer to a Haxe Array of Future handles
///
/// # Returns
/// A new Future handle that resolves with the first completed result
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_race(arr_ptr: *const u8) -> *mut u8 {
    if arr_ptr.is_null() {
        return ptr::null_mut();
    }

    // Read the Haxe Array of sub-future handles (same layout as Future.all)
    let arr = unsafe { &*(arr_ptr as *const crate::haxe_array::HaxeArray) };
    let n = arr.len;

    if n == 0 {
        return ptr::null_mut();
    }

    let data_ptr = arr.ptr as *const u8;
    let mut sub_handles: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let handle = unsafe { *(data_ptr.add(i * 8) as *const i64) } as usize;
        sub_handles.push(handle);
    }

    // Create outer Future — raw_result=true so we return the value directly
    let handle = Box::new(FutureHandle {
        state: AtomicU8::new(STATE_RUNNING),
        fn_ptr: ptr::null(),
        env_ptr: ptr::null(),
        value: Mutex::new(0),
        cvar: Condvar::new(),
        then_fn: AtomicPtr::new(ptr::null_mut()),
        then_env: AtomicPtr::new(ptr::null_mut()),
        raw_result: false,
        has_error: AtomicBool::new(false),
        error_value: Mutex::new(0),
        error_type_id: AtomicU32::new(0),
        error_message: Mutex::new(None),
        cancelled: AtomicBool::new(false),
    });
    let handle_ptr = Box::into_raw(handle);
    let handle_addr = handle_ptr as usize;

    ACTIVE_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
    crate::concurrency::arm64_jit_barrier();

    thread::spawn(move || {
        crate::concurrency::arm64_jit_barrier();

        let coordinating_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Phase 1: Spawn ALL sub-futures
            let mut fh_ptrs: Vec<*mut FutureHandle> = Vec::with_capacity(n);
            for &h_addr in &sub_handles {
                let h = h_addr as *mut FutureHandle;
                fh_ptrs.push(h);
                if h.is_null() {
                    continue;
                }
                let future = unsafe { &*h };
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
                    spawn_future(future, h);
                }
            }

            // Phase 2: Poll until any resolves
            loop {
                for &h in &fh_ptrs {
                    if h.is_null() {
                        continue;
                    }
                    let future = unsafe { &*h };
                    if future.state.load(Ordering::Acquire) == STATE_RESOLVED {
                        // First to resolve — grab its value
                        let val = *future.value.lock().unwrap();
                        let outer = unsafe { &*(handle_addr as *const FutureHandle) };

                        // Propagate error if the winner had one
                        if future.has_error.load(Ordering::Acquire) {
                            let exc_val = *future.error_value.lock().unwrap();
                            let exc_tid = future.error_type_id.load(Ordering::Acquire);
                            *outer.error_value.lock().unwrap() = exc_val;
                            outer.error_type_id.store(exc_tid, Ordering::Release);
                            outer.has_error.store(true, Ordering::Release);
                        } else {
                            *outer.value.lock().unwrap() = val;
                        }

                        outer.state.store(STATE_RESOLVED, Ordering::Release);
                        outer.cvar.notify_all();
                        return;
                    }
                }
                thread::sleep(std::time::Duration::from_millis(1));
            }
        }));

        if coordinating_result.is_err() {
            let outer = unsafe { &*(handle_addr as *const FutureHandle) };
            outer.state.store(STATE_RESOLVED, Ordering::Release);
            outer.cvar.notify_all();
        }

        ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
    });

    handle_ptr as *mut u8
}

/// Check if a future has resolved.
///
/// # Returns
/// `true` if the future has resolved and the value is available
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_is_ready(handle: *const u8) -> bool {
    if handle.is_null() {
        return false;
    }

    let future = unsafe { &*(handle as *const FutureHandle) };
    future.state.load(Ordering::Acquire) == STATE_RESOLVED
}

/// Cancel a future (cooperative cancellation).
///
/// - If Pending: transitions to Resolved immediately, returns true
/// - If Running: sets cancellation flag (worker thread should check), returns true
/// - If already Resolved: returns false
///
/// After cancellation, `.await()` returns null.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_cancel(handle: *mut u8) -> bool {
    if handle.is_null() {
        return false;
    }

    let future = unsafe { &*(handle as *const FutureHandle) };
    let state = future.state.load(Ordering::Acquire);

    match state {
        STATE_PENDING => {
            // Not yet started — cancel immediately
            future.cancelled.store(true, Ordering::Release);
            future.state.store(STATE_RESOLVED, Ordering::Release);
            future.cvar.notify_all();
            true
        }
        STATE_RUNNING => {
            // Running — set flag for cooperative cancellation
            future.cancelled.store(true, Ordering::Release);
            true
        }
        _ => false, // Already resolved
    }
}

/// Check if a future has been cancelled.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_future_is_cancelled(handle: *const u8) -> bool {
    if handle.is_null() {
        return false;
    }

    let future = unsafe { &*(handle as *const FutureHandle) };
    future.cancelled.load(Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test closures use i64 return type to match JIT convention
    extern "C" fn simple_return_42(_env: *const u8) -> i64 {
        42
    }

    extern "C" fn add_env_values(env: *const u8) -> i64 {
        unsafe {
            let a = *(env as *const i64);
            let b = *((env as *const i64).add(1));
            a + b
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
        let handle = rayzor_future_create(simple_return_42 as *const u8, ptr::null());
        assert!(!handle.is_null());

        let result = unbox_result(rayzor_future_await(handle));
        assert_eq!(result, 42);

        unsafe {
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }

    #[test]
    fn test_lazy_not_started() {
        let handle = rayzor_future_create(simple_return_42 as *const u8, ptr::null());
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
        let handle = rayzor_future_create(add_env_values as *const u8, env.as_ptr() as *const u8);

        let result = unbox_result(rayzor_future_await(handle));
        assert_eq!(result, 30);

        unsafe {
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }

    #[test]
    fn test_multiple_futures() {
        let h1 = rayzor_future_create(simple_return_42 as *const u8, ptr::null());
        let h2 = rayzor_future_create(simple_return_42 as *const u8, ptr::null());

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

    #[test]
    fn plugin_future_resolves_a_raw_object_once() {
        let handle = rayzor_plugin_future_pending(1);
        let object = Box::into_raw(Box::new(42_u64)) as *mut u8;
        assert_eq!(rayzor_plugin_future_resolve(handle, object as i64), 1);
        assert_eq!(rayzor_plugin_future_resolve(handle, 0), 0);
        assert_eq!(rayzor_future_poll(handle), object);
        assert_eq!(rayzor_future_await(handle), object);
        unsafe {
            drop(Box::from_raw(object as *mut u64));
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }

    #[test]
    fn plugin_future_rejection_copies_its_message() {
        let handle = rayzor_plugin_future_pending(1);
        let mut message = b"device lost".to_vec();
        assert_eq!(
            rayzor_plugin_future_reject(handle, message.as_ptr(), message.len()),
            1
        );
        message.fill(b'x');
        let future = unsafe { &*(handle as *const FutureHandle) };
        assert_eq!(
            future.error_message.lock().unwrap().as_deref(),
            Some("device lost")
        );
        unsafe {
            drop(Box::from_raw(handle as *mut FutureHandle));
        }
    }
}
