//! Concurrency Runtime Implementation
//!
//! Provides C-ABI compatible implementations of concurrency primitives
//! for the Rayzor compiler's stdlib extern functions.
//!
//! # Architecture
//!
//! - Thread: Wraps std::thread::JoinHandle
//! - Arc: Wraps std::sync::Arc for atomic reference counting
//! - Mutex: Wraps std::sync::Mutex for mutual exclusion
//! - Channel: Wraps std::sync::mpsc for message passing
//!
//! # Thread Safety with JIT Code
//!
//! The runtime tracks all spawned threads globally to ensure that threads
//! executing JIT-compiled code don't outlive the JIT module. Call
//! `rayzor_wait_all_threads()` before dropping the JIT module.

use log::debug;
use std::collections::HashMap;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

// Native pthread support for Apple Silicon
#[cfg(target_os = "macos")]
use libc::pthread_t;

// ============================================================================
// Memory Safety Infrastructure
// ============================================================================

/// Magic numbers for validating handle types
#[allow(dead_code)]
const MAGIC_THREAD: u64 = 0xDEADBEEF_00000001;
#[allow(dead_code)]
const MAGIC_ARC: u64 = 0xDEADBEEF_00000002;
#[allow(dead_code)]
const MAGIC_MUTEX: u64 = 0xDEADBEEF_00000003;
#[allow(dead_code)]
const MAGIC_MUTEX_GUARD: u64 = 0xDEADBEEF_00000004;

/// Panic guard that ensures cleanup runs even on panic
#[allow(dead_code)]
struct PanicGuard<F: FnOnce()> {
    cleanup: Option<F>,
}

#[allow(dead_code)]
impl<F: FnOnce()> PanicGuard<F> {
    fn new(cleanup: F) -> Self {
        Self {
            cleanup: Some(cleanup),
        }
    }

    fn disarm(mut self) {
        self.cleanup = None;
    }
}

impl<F: FnOnce()> Drop for PanicGuard<F> {
    fn drop(&mut self) {
        if let Some(cleanup) = self.cleanup.take() {
            cleanup();
        }
    }
}

// ============================================================================
// Global Thread Tracking
// ============================================================================

/// Global counter for thread IDs
#[allow(dead_code)]
static THREAD_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Global count of active threads (spawned but not yet joined)
static ACTIVE_THREAD_COUNT: AtomicU64 = AtomicU64::new(0);

lazy_static::lazy_static! {
    /// Global registry of all active thread handles
    /// Maps thread ID -> JoinHandle wrapped in an Option (taken when joined)
    static ref THREAD_REGISTRY: Mutex<HashMap<u64, Option<JoinHandle<i32>>>> =
        Mutex::new(HashMap::new());
}

/// Wait for all spawned threads to complete
/// This should be called before dropping the JIT module to prevent use-after-free
#[no_mangle]
pub extern "C" fn rayzor_wait_all_threads() {
    debug!("[rayzor_wait_all_threads] Waiting for all threads to complete...");

    // Keep looping until all threads are done
    loop {
        let count = ACTIVE_THREAD_COUNT.load(Ordering::SeqCst);
        if count == 0 {
            debug!("[rayzor_wait_all_threads] All threads completed");
            break;
        }
        debug!(
            "[rayzor_wait_all_threads] {} threads still active, waiting...",
            count
        );
        thread::sleep(Duration::from_millis(10));
    }
}

/// Get the current count of active threads
#[no_mangle]
pub extern "C" fn rayzor_active_thread_count() -> u64 {
    ACTIVE_THREAD_COUNT.load(Ordering::SeqCst)
}

// ============================================================================
// Thread Implementation
// ============================================================================

/// Opaque thread handle with memory safety validation
/// Wraps JoinHandle<i32> to support returning results
#[allow(dead_code)]
struct ThreadHandle {
    /// Magic number for validation
    magic: u64,
    /// Generation counter to detect stale handles
    generation: u64,
    /// The actual thread handle (None if already joined)
    handle: Option<JoinHandle<i32>>,
    /// Unique ID for tracking
    thread_id: u64,
    /// Flag to detect double-join attempts
    joined: bool,
}

#[allow(dead_code)]
impl ThreadHandle {
    /// Validate that this is a legitimate ThreadHandle
    fn validate(&self) -> Result<(), &'static str> {
        if self.magic != MAGIC_THREAD {
            return Err("Invalid thread handle: magic number mismatch");
        }
        if self.joined {
            return Err("Thread handle already joined");
        }
        Ok(())
    }
}

/// Context passed to pthread thread function
#[cfg(target_os = "macos")]
#[repr(C)]
#[allow(dead_code)]
struct PthreadContext {
    closure: *const u8,
    env: *const u8,
    result: i32,
}

/// pthread thread entry point
#[cfg(target_os = "macos")]
#[allow(dead_code)]
extern "C" fn pthread_entry(arg: *mut libc::c_void) -> *mut libc::c_void {
    unsafe {
        let ctx = &mut *(arg as *mut PthreadContext);

        // Cast closure to function pointer
        type ClosureFn = extern "C" fn(*const u8) -> i32;
        let func: ClosureFn = std::mem::transmute(ctx.closure);

        // Call the JIT'd function
        ctx.result = func(ctx.env);

        // Return the result
        ctx.result as usize as *mut libc::c_void
    }
}

/// Native pthread handle - stores only the pthread_t
/// The context is owned by the pthread entry function
#[cfg(target_os = "macos")]
#[allow(dead_code)]
struct NativePthreadHandle {
    thread: pthread_t,
    ctx_ptr: *mut PthreadContext, // Raw pointer, not owned - thread owns it
}

/// ARM64-specific memory barriers and JIT write protection for JIT code execution
///
/// On macOS ARM64 with MAP_JIT memory:
/// - pthread_jit_write_protect_np(0): Write mode (can write, cannot execute)
/// - pthread_jit_write_protect_np(1): Execute mode (can execute, cannot write)
///
/// DSB SY: Data Synchronization Barrier - ensures all memory accesses complete
/// ISB SY: Instruction Synchronization Barrier - flushes the instruction pipeline
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[inline(always)]
fn arm64_jit_barrier() {
    unsafe {
        // Switch to execute mode - required for spawned threads to execute JIT code
        // Newly spawned threads start in write mode by default, so we must switch
        unsafe extern "C" {
            fn pthread_jit_write_protect_np(enabled: libc::c_int);
        }
        pthread_jit_write_protect_np(1);

        // DSB SY: Wait for all memory operations to complete
        // This ensures the JIT code is fully written to memory
        std::arch::asm!("dsb sy", options(nostack, preserves_flags));
        // ISB SY: Flush the instruction pipeline
        // This ensures we fetch fresh instructions from memory
        std::arch::asm!("isb sy", options(nostack, preserves_flags));
    }
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
#[inline(always)]
fn arm64_jit_barrier() {
    // No-op on other architectures
}

/// Spawn a new thread with a closure
///
/// # Safety
/// - closure must be a valid function pointer
/// - closure_env may be null if closure captures no environment
#[no_mangle]
pub unsafe extern "C" fn rayzor_thread_spawn(
    closure: *const u8,
    closure_env: *const u8,
) -> *mut u8 {
    // Simple null check
    if closure.is_null() {
        return ptr::null_mut();
    }

    // Convert pointers to usize for Send
    let env_addr = closure_env as usize;
    let func_addr = closure as usize;

    // Increment active thread count BEFORE spawning
    ACTIVE_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);

    // Execute barrier on main thread before spawning to ensure JIT code is visible
    arm64_jit_barrier();

    // Spawn thread
    let handle = thread::spawn(move || {
        // Execute barrier before calling JIT code
        arm64_jit_barrier();

        type ClosureFn = extern "C" fn(*const u8) -> i32;
        let env_ptr = env_addr as *const u8;
        let func: ClosureFn = unsafe { std::mem::transmute(func_addr) };
        func(env_ptr)
    });

    // Return simple handle
    Box::into_raw(Box::new(handle)) as *mut u8
}

/// Join a thread and wait for it to complete
///
/// # Safety
/// - handle must be a valid pointer from rayzor_thread_spawn
/// - handle is consumed and should not be used after this call
/// - returns the i32 result cast to *mut u8
#[no_mangle]
pub unsafe extern "C" fn rayzor_thread_join(handle: *mut u8) -> *mut u8 {
    if handle.is_null() {
        return ptr::null_mut();
    }

    // Simple implementation using std::thread
    let boxed_handle: Box<JoinHandle<i32>> = Box::from_raw(handle as *mut JoinHandle<i32>);
    let result = boxed_handle.join().unwrap_or(-1);
    ACTIVE_THREAD_COUNT.fetch_sub(1, Ordering::SeqCst);
    result as usize as *mut u8
}

/// Check if a thread has finished executing
#[no_mangle]
pub unsafe extern "C" fn rayzor_thread_is_finished(_handle: *const u8) -> bool {
    // Note: pthread doesn't have a simple is_finished check
    // This is a best-effort implementation
    if _handle.is_null() {
        return true;
    }
    // Return false to indicate we don't know - caller should join to be sure
    false
}

/// Yield execution to other threads
#[no_mangle]
pub extern "C" fn rayzor_thread_yield_now() {
    thread::yield_now();
}

/// Sleep for specified milliseconds
#[no_mangle]
pub extern "C" fn rayzor_thread_sleep(millis: i32) {
    if millis > 0 {
        thread::sleep(Duration::from_millis(millis as u64));
    }
}

/// Get the current thread ID as u64
#[no_mangle]
pub extern "C" fn rayzor_thread_current_id() -> u64 {
    // Convert ThreadId to u64 (simplified - uses debug format hash)
    let id = thread::current().id();
    // Use a simple hash of the debug representation
    format!("{:?}", id)
        .bytes()
        .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64))
}

// ============================================================================
// Arc Implementation
// ============================================================================

/// Initialize a new Arc with a value
///
/// # Safety
/// - value must be a valid pointer (will be owned by Arc)
#[no_mangle]
#[allow(clippy::arc_with_non_send_sync)]
pub unsafe extern "C" fn rayzor_arc_init(value: *mut u8) -> *mut u8 {
    if value.is_null() {
        return ptr::null_mut();
    }

    // Wrap the raw pointer in an Arc
    // Note: This takes ownership of the value pointer
    let arc = Arc::new(value);

    // Convert Arc to raw pointer
    Arc::into_raw(arc) as *mut u8
}

/// Clone an Arc (increment reference count)
///
/// # Safety
/// - arc must be a valid Arc pointer from rayzor_arc_init or rayzor_arc_clone
#[no_mangle]
pub unsafe extern "C" fn rayzor_arc_clone(arc: *const u8) -> *mut u8 {
    debug!("[rayzor_arc_clone] Called with arc={:?}", arc);

    if arc.is_null() {
        debug!("[SAFETY ERROR] rayzor_arc_clone: NULL arc pointer");
        return ptr::null_mut();
    }

    // Reconstruct Arc from raw pointer (without decrementing count)
    let arc_ref = Arc::from_raw(arc as *const *mut u8);
    debug!(
        "[rayzor_arc_clone] Arc reconstructed, strong_count={}",
        Arc::strong_count(&arc_ref)
    );

    // Clone it (increments ref count)
    let cloned = Arc::clone(&arc_ref);
    let cloned_ptr = Arc::into_raw(cloned) as *mut u8;
    debug!("[rayzor_arc_clone] Cloned to {:?}", cloned_ptr);

    // Forget the original to avoid decrementing ref count
    // Note: forget cannot panic, so no guard needed
    std::mem::forget(arc_ref);

    // Return new Arc as raw pointer
    cloned_ptr
}

/// Get the inner value pointer from an Arc
///
/// # Safety
/// - arc must be a valid Arc pointer
/// - returned pointer is valid as long as Arc exists
#[no_mangle]
pub unsafe extern "C" fn rayzor_arc_get(arc: *const u8) -> *const u8 {
    if arc.is_null() {
        debug!("[SAFETY ERROR] rayzor_arc_get: NULL arc pointer");
        return ptr::null();
    }

    // Reconstruct Arc temporarily
    let arc_ref = Arc::from_raw(arc as *const *mut u8);
    debug!(
        "[rayzor_arc_get] Arc reconstructed, strong_count={}",
        Arc::strong_count(&arc_ref)
    );

    // Get the inner value
    let value_ptr = *arc_ref as *const u8;

    // Forget to avoid decrementing ref count
    std::mem::forget(arc_ref);

    value_ptr
}

/// Get the strong reference count of an Arc
#[no_mangle]
pub unsafe extern "C" fn rayzor_arc_strong_count(arc: *const u8) -> u64 {
    if arc.is_null() {
        return 0;
    }

    let arc_ref = Arc::from_raw(arc as *const *mut u8);
    let count = Arc::strong_count(&arc_ref);
    std::mem::forget(arc_ref);

    count as u64
}

/// Try to unwrap an Arc (returns value if refcount == 1)
///
/// # Safety
/// - arc must be a valid Arc pointer
/// - returns null if refcount > 1
#[no_mangle]
pub unsafe extern "C" fn rayzor_arc_try_unwrap(arc: *mut u8) -> *mut u8 {
    if arc.is_null() {
        return ptr::null_mut();
    }

    let arc_obj = Arc::from_raw(arc as *const *mut u8);

    match Arc::try_unwrap(arc_obj) {
        Ok(value) => value,
        Err(arc_back) => {
            // Failed to unwrap, restore the Arc
            std::mem::forget(arc_back);
            ptr::null_mut()
        }
    }
}

/// Get the pointer address of the Arc's data
#[no_mangle]
pub unsafe extern "C" fn rayzor_arc_as_ptr(arc: *const u8) -> u64 {
    if arc.is_null() {
        return 0;
    }

    let arc_ref = Arc::from_raw(arc as *const *mut u8);
    let ptr_addr = Arc::as_ptr(&arc_ref) as u64;
    std::mem::forget(arc_ref);

    ptr_addr
}

// ============================================================================
// Mutex Implementation (using parking_lot for better FFI support)
// ============================================================================

use parking_lot::lock_api::RawMutex as RawMutexTrait;

/// Mutex handle using parking_lot's RawMutex for explicit lock/unlock
struct MutexHandle {
    /// The raw mutex for locking
    raw_mutex: parking_lot::RawMutex,
    /// The protected value
    value: *mut u8,
}

/// Mutex guard handle - stores reference back to mutex for unlocking
struct MutexGuardHandle {
    /// Pointer to the mutex handle for unlocking
    mutex: *const MutexHandle,
}

/// Initialize a new Mutex with a value
#[no_mangle]
pub unsafe extern "C" fn rayzor_mutex_init(value: *mut u8) -> *mut u8 {
    let mutex = Box::new(MutexHandle {
        raw_mutex: parking_lot::RawMutex::INIT,
        value,
    });

    Box::into_raw(mutex) as *mut u8
}

/// Lock a mutex and return a guard
///
/// # Safety
/// - mutex must be a valid Mutex pointer
/// - blocks until lock is acquired
#[no_mangle]
pub unsafe extern "C" fn rayzor_mutex_lock(mutex: *mut u8) -> *mut u8 {
    if mutex.is_null() {
        return ptr::null_mut();
    }

    let mutex_handle = &*(mutex as *const MutexHandle);

    // Lock the raw mutex (blocks until acquired)
    mutex_handle.raw_mutex.lock();

    let guard_handle = Box::new(MutexGuardHandle {
        mutex: mutex_handle as *const MutexHandle,
    });

    Box::into_raw(guard_handle) as *mut u8
}

/// Try to lock a mutex without blocking
#[no_mangle]
pub unsafe extern "C" fn rayzor_mutex_try_lock(mutex: *mut u8) -> *mut u8 {
    if mutex.is_null() {
        return ptr::null_mut();
    }

    let mutex_handle = &*(mutex as *const MutexHandle);

    if mutex_handle.raw_mutex.try_lock() {
        let guard_handle = Box::new(MutexGuardHandle {
            mutex: mutex_handle as *const MutexHandle,
        });
        Box::into_raw(guard_handle) as *mut u8
    } else {
        ptr::null_mut()
    }
}

/// Check if a mutex is currently locked
#[no_mangle]
pub unsafe extern "C" fn rayzor_mutex_is_locked(mutex: *const u8) -> bool {
    if mutex.is_null() {
        return false;
    }

    let mutex_handle = &*(mutex as *const MutexHandle);
    mutex_handle.raw_mutex.is_locked()
}

/// Get the value pointer from a mutex guard
#[no_mangle]
pub unsafe extern "C" fn rayzor_mutex_guard_get(guard: *mut u8) -> *mut u8 {
    if guard.is_null() {
        return ptr::null_mut();
    }

    let guard_handle = &*(guard as *const MutexGuardHandle);
    (*guard_handle.mutex).value
}

/// Unlock a mutex guard
#[no_mangle]
pub unsafe extern "C" fn rayzor_mutex_unlock(guard: *mut u8) {
    if !guard.is_null() {
        // Reconstruct Box and get the mutex reference
        let guard_handle = Box::from_raw(guard as *mut MutexGuardHandle);
        // Unlock the raw mutex
        (*guard_handle.mutex).raw_mutex.unlock();
        // Box will be dropped here
    }
}

// ============================================================================
// Channel Implementation
// ============================================================================

// Use crossbeam-like approach with Arc<Mutex<>> for thread-safe mpmc
use std::collections::VecDeque;
use std::sync::Condvar;

/// Channel state for multi-producer multi-consumer
struct ChannelState {
    buffer: VecDeque<*mut u8>,
    capacity: usize, // 0 = unbounded
    closed: bool,
}

/// Channel handle - thread-safe mpmc channel
struct ChannelHandle {
    state: Mutex<ChannelState>,
    not_empty: Condvar,
    not_full: Condvar,
}

/// Initialize a new channel with optional capacity
/// capacity=0 means unbounded channel
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_init(capacity: i32) -> *mut u8 {
    let channel_handle = Box::new(ChannelHandle {
        state: Mutex::new(ChannelState {
            buffer: VecDeque::new(),
            capacity: if capacity <= 0 { 0 } else { capacity as usize },
            closed: false,
        }),
        not_empty: Condvar::new(),
        not_full: Condvar::new(),
    });

    Box::into_raw(channel_handle) as *mut u8
}

/// Send a value through a channel (blocking)
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_send(channel: *mut u8, value: *mut u8) {
    debug!(
        "[rayzor_channel_send] Called with channel={:?}, value={:?}",
        channel, value
    );

    if channel.is_null() {
        debug!("[rayzor_channel_send] channel is null, returning");
        return;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    debug!("[rayzor_channel_send] Got channel_handle, locking...");
    let mut state = channel_handle.state.lock().unwrap();
    debug!("[rayzor_channel_send] Lock acquired");

    // For bounded channels, wait while full
    while state.capacity > 0 && state.buffer.len() >= state.capacity && !state.closed {
        state = channel_handle.not_full.wait(state).unwrap();
    }

    if state.closed {
        return;
    }

    state.buffer.push_back(value);
    drop(state);

    // Notify waiting receivers
    channel_handle.not_empty.notify_one();
}

/// Try to send a value through a channel (non-blocking)
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_try_send(channel: *mut u8, value: *mut u8) -> bool {
    if channel.is_null() {
        return false;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let mut state = channel_handle.state.lock().unwrap();

    if state.closed {
        return false;
    }

    // For bounded channels, check if full
    if state.capacity > 0 && state.buffer.len() >= state.capacity {
        return false;
    }

    state.buffer.push_back(value);
    drop(state);
    channel_handle.not_empty.notify_one();
    true
}

/// Receive a value from a channel (blocking)
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_receive(channel: *mut u8) -> *mut u8 {
    if channel.is_null() {
        return ptr::null_mut();
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let mut state = channel_handle.state.lock().unwrap();

    // Wait while buffer is empty and channel is not closed
    while state.buffer.is_empty() && !state.closed {
        state = channel_handle.not_empty.wait(state).unwrap();
    }

    if let Some(value) = state.buffer.pop_front() {
        drop(state);
        channel_handle.not_full.notify_one();
        value
    } else {
        ptr::null_mut()
    }
}

/// Try to receive a value from a channel (non-blocking)
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_try_receive(channel: *mut u8) -> *mut u8 {
    if channel.is_null() {
        return ptr::null_mut();
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let mut state = channel_handle.state.lock().unwrap();

    if let Some(value) = state.buffer.pop_front() {
        drop(state);
        channel_handle.not_full.notify_one();
        value
    } else {
        ptr::null_mut()
    }
}

/// Close a channel
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_close(channel: *mut u8) {
    if channel.is_null() {
        return;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let mut state = channel_handle.state.lock().unwrap();
    state.closed = true;
    drop(state);

    // Wake up all waiting threads
    channel_handle.not_empty.notify_all();
    channel_handle.not_full.notify_all();
}

/// Check if a channel is closed
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_is_closed(channel: *const u8) -> bool {
    if channel.is_null() {
        return true;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let state = channel_handle.state.lock().unwrap();
    state.closed
}

/// Get the number of messages in the channel
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_len(channel: *const u8) -> i32 {
    if channel.is_null() {
        return 0;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let state = channel_handle.state.lock().unwrap();
    state.buffer.len() as i32
}

/// Get the channel capacity
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_capacity(channel: *const u8) -> i32 {
    if channel.is_null() {
        return 0;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let state = channel_handle.state.lock().unwrap();
    if state.capacity == 0 {
        -1 // Unbounded
    } else {
        state.capacity as i32
    }
}

/// Check if channel is empty
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_is_empty(channel: *const u8) -> bool {
    if channel.is_null() {
        return true;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let state = channel_handle.state.lock().unwrap();
    state.buffer.is_empty()
}

/// Check if channel is full
#[no_mangle]
pub unsafe extern "C" fn rayzor_channel_is_full(channel: *const u8) -> bool {
    if channel.is_null() {
        return false;
    }

    let channel_handle = &*(channel as *const ChannelHandle);
    let state = channel_handle.state.lock().unwrap();
    state.capacity > 0 && state.buffer.len() >= state.capacity
}

// ============================================================================
// Semaphore Implementation
// ============================================================================

/// Semaphore state
struct SemaphoreState {
    count: i32,
}

/// Semaphore handle - counting semaphore for synchronization
struct SemaphoreHandle {
    state: Mutex<SemaphoreState>,
    not_zero: Condvar,
}

/// Initialize a new semaphore with an initial value
#[no_mangle]
pub unsafe extern "C" fn rayzor_semaphore_init(initial_value: i32) -> *mut u8 {
    let semaphore = Box::new(SemaphoreHandle {
        state: Mutex::new(SemaphoreState {
            count: initial_value.max(0),
        }),
        not_zero: Condvar::new(),
    });

    Box::into_raw(semaphore) as *mut u8
}

/// Acquire (decrement) the semaphore, blocking if count is zero
#[no_mangle]
pub unsafe extern "C" fn rayzor_semaphore_acquire(semaphore: *mut u8) {
    if semaphore.is_null() {
        return;
    }

    let sem = &*(semaphore as *const SemaphoreHandle);
    let mut state = sem.state.lock().unwrap();

    // Wait while count is zero
    while state.count == 0 {
        state = sem.not_zero.wait(state).unwrap();
    }

    // Decrement
    state.count -= 1;
}

/// Try to acquire the semaphore with optional timeout (in seconds)
/// Returns true if acquired, false if timed out or count was zero (non-blocking)
#[no_mangle]
pub unsafe extern "C" fn rayzor_semaphore_try_acquire(
    semaphore: *mut u8,
    timeout_seconds: f64,
) -> bool {
    if semaphore.is_null() {
        return false;
    }

    let sem = &*(semaphore as *const SemaphoreHandle);
    let mut state = sem.state.lock().unwrap();

    // If timeout is negative or zero, just try once without blocking
    if timeout_seconds <= 0.0 {
        if state.count > 0 {
            state.count -= 1;
            return true;
        }
        return false;
    }

    // Wait with timeout
    let timeout = Duration::from_secs_f64(timeout_seconds);
    let deadline = std::time::Instant::now() + timeout;

    while state.count == 0 {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return false; // Timeout
        }

        let result = sem.not_zero.wait_timeout(state, remaining).unwrap();
        state = result.0;
        if result.1.timed_out() && state.count == 0 {
            return false;
        }
    }

    state.count -= 1;
    true
}

// ============================================================================
// sys.thread.Lock wrapper functions
// ============================================================================

/// Wait for the lock indefinitely (blocking)
/// Returns true when acquired (always, since it blocks forever)
#[no_mangle]
pub unsafe extern "C" fn sys_lock_wait(semaphore: *mut u8) -> bool {
    if semaphore.is_null() {
        return false;
    }
    rayzor_semaphore_acquire(semaphore);
    true
}

// ============================================================================
// sys.thread.Semaphore wrapper functions
// ============================================================================

/// Try to acquire semaphore without blocking (0-arg version)
/// Returns true if acquired, false if count was zero
#[no_mangle]
pub unsafe extern "C" fn sys_semaphore_try_acquire_nowait(semaphore: *mut u8) -> bool {
    rayzor_semaphore_try_acquire(semaphore, 0.0)
}

/// Release (increment) the semaphore
#[no_mangle]
pub unsafe extern "C" fn rayzor_semaphore_release(semaphore: *mut u8) {
    if semaphore.is_null() {
        return;
    }

    let sem = &*(semaphore as *const SemaphoreHandle);
    let mut state = sem.state.lock().unwrap();
    state.count += 1;
    drop(state);

    // Wake one waiting thread
    sem.not_zero.notify_one();
}

/// Get the current count of the semaphore
#[no_mangle]
pub unsafe extern "C" fn rayzor_semaphore_count(semaphore: *const u8) -> i32 {
    if semaphore.is_null() {
        return 0;
    }

    let sem = &*(semaphore as *const SemaphoreHandle);
    let state = sem.state.lock().unwrap();
    state.count
}

// ============================================================================
// sys.thread.Thread wrapper functions
// ============================================================================

/// Create a thread using sys.thread.Thread API (wrapper around rayzor_thread_spawn)
/// This version doesn't return a value, just runs a void->void closure
#[no_mangle]
pub unsafe extern "C" fn sys_thread_create(closure: *const u8, closure_env: *const u8) -> *mut u8 {
    rayzor_thread_spawn(closure, closure_env)
}

/// Join a thread (wrapper for sys.thread.Thread compatibility)
#[no_mangle]
pub unsafe extern "C" fn sys_thread_join(handle: *mut u8) {
    let _ = rayzor_thread_join(handle);
}

/// Check if thread is finished
#[no_mangle]
pub unsafe extern "C" fn sys_thread_is_finished(handle: *const u8) -> bool {
    rayzor_thread_is_finished(handle)
}

/// Yield current thread
#[no_mangle]
pub extern "C" fn sys_thread_yield() {
    rayzor_thread_yield_now();
}

/// Sleep for specified seconds (converts to milliseconds)
#[no_mangle]
pub extern "C" fn sys_thread_sleep(seconds: f64) {
    if seconds > 0.0 {
        let millis = (seconds * 1000.0) as i32;
        rayzor_thread_sleep(millis);
    }
}

/// Get current thread (returns a thread handle representing current thread)
#[no_mangle]
pub extern "C" fn sys_thread_current() -> *mut u8 {
    // Return the current thread ID as the handle
    // Note: This is a simplified implementation
    rayzor_thread_current_id() as usize as *mut u8
}

// ============================================================================
// sys.thread.Mutex wrapper functions (simple lock without inner value)
// ============================================================================

// NOTE: Previous implementation stored a `current_guard` in a shared handle,
// which caused race conditions when multiple threads used the same mutex.
// This new implementation uses parking_lot's RawMutex directly without guards,
// making lock/unlock operations thread-safe.

/// Create a simple mutex (no inner value)
#[no_mangle]
pub unsafe extern "C" fn sys_mutex_new() -> *mut u8 {
    sys_mutex_alloc()
}

/// Acquire a mutex (blocking)
/// Uses RawMutex::lock() directly - no guard storage needed
#[no_mangle]
pub unsafe extern "C" fn sys_mutex_acquire(mutex: *mut u8) {
    if !mutex.is_null() {
        // Cast directly to MutexHandle (same struct as used by rayzor_mutex_*)
        let mutex_handle = &*(mutex as *const MutexHandle);
        // Lock the raw mutex (blocking)
        mutex_handle.raw_mutex.lock();
    }
}

/// Try to acquire a mutex (non-blocking)
/// Returns boxed Bool (Dynamic value): true if acquired, false if already locked
#[no_mangle]
pub unsafe extern "C" fn sys_mutex_try_acquire(mutex: *mut u8) -> *mut u8 {
    if mutex.is_null() {
        return crate::type_system::haxe_box_bool_ptr(false);
    }

    let mutex_handle = &*(mutex as *const MutexHandle);
    let result = mutex_handle.raw_mutex.try_lock();

    crate::type_system::haxe_box_bool_ptr(result)
}

/// Release a mutex
/// Uses RawMutex::unlock() directly - thread-safe, no guard needed
#[no_mangle]
pub unsafe extern "C" fn sys_mutex_release(mutex: *mut u8) {
    if !mutex.is_null() {
        let mutex_handle = &*(mutex as *const MutexHandle);
        // Unlock the raw mutex
        // SAFETY: Caller is responsible for only calling unlock when they hold the lock
        mutex_handle.raw_mutex.unlock();
    }
}

/// Allocate a simple mutex
/// Creates a MutexHandle with RawMutex for thread-safe acquire/release
#[no_mangle]
pub unsafe extern "C" fn sys_mutex_alloc() -> *mut u8 {
    // Use MutexHandle directly (same as rayzor_mutex_init)
    let mutex = Box::new(MutexHandle {
        raw_mutex: parking_lot::RawMutex::INIT,
        value: ptr::null_mut(),
    });
    Box::into_raw(mutex) as *mut u8
}

// ============================================================================
// sys.thread.Deque - Double-Ended Queue
// ============================================================================

/// Handle for sys.thread.Deque<T>
/// A thread-safe double-ended queue with blocking pop operation
struct DequeHandle {
    /// The actual deque storage (stores Dynamic pointers)
    deque: Arc<Mutex<VecDeque<*mut u8>>>,
    /// Condition variable for blocking on empty deque
    not_empty: Arc<Condvar>,
}

/// Create a new sys.thread.Deque
#[no_mangle]
#[allow(clippy::arc_with_non_send_sync)]
pub unsafe extern "C" fn sys_deque_alloc() -> *mut u8 {
    let handle = DequeHandle {
        deque: Arc::new(Mutex::new(VecDeque::new())),
        not_empty: Arc::new(Condvar::new()),
    };
    Box::into_raw(Box::new(handle)) as *mut u8
}

/// Add element to the end of the deque
#[no_mangle]
pub unsafe extern "C" fn sys_deque_add(deque: *mut u8, item: *mut u8) {
    if deque.is_null() {
        return;
    }

    let handle = &*(deque as *const DequeHandle);
    let mut queue = handle.deque.lock().unwrap();
    queue.push_back(item);

    // Notify one waiting thread
    handle.not_empty.notify_one();
}

/// Push element to the front of the deque
#[no_mangle]
pub unsafe extern "C" fn sys_deque_push(deque: *mut u8, item: *mut u8) {
    if deque.is_null() {
        return;
    }

    let handle = &*(deque as *const DequeHandle);
    let mut queue = handle.deque.lock().unwrap();
    queue.push_front(item);

    // Notify one waiting thread
    handle.not_empty.notify_one();
}

/// Pop element from the front of the deque
/// If block is true, blocks until an element is available
/// If block is false and deque is empty, returns null
#[no_mangle]
pub unsafe extern "C" fn sys_deque_pop(deque: *mut u8, block: bool) -> *mut u8 {
    if deque.is_null() {
        return ptr::null_mut();
    }

    let handle = &*(deque as *const DequeHandle);
    let mut queue = handle.deque.lock().unwrap();

    if block {
        // Block until an element is available
        while queue.is_empty() {
            queue = handle.not_empty.wait(queue).unwrap();
        }
        queue.pop_front().unwrap_or(ptr::null_mut())
    } else {
        // Non-blocking: return null if empty
        queue.pop_front().unwrap_or(ptr::null_mut())
    }
}

// ============================================================================
// sys.thread.Condition - Condition Variable
// ============================================================================

/// Handle for sys.thread.Condition
/// A condition variable with an internal mutex for thread synchronization
struct ConditionHandle {
    /// Internal mutex
    mutex: Arc<Mutex<()>>,
    /// The condition variable
    condvar: Arc<Condvar>,
    /// Current guard (if mutex is locked)
    guard: Option<std::sync::MutexGuard<'static, ()>>,
}

/// Create a new sys.thread.Condition
#[no_mangle]
pub unsafe extern "C" fn sys_condition_alloc() -> *mut u8 {
    let handle = ConditionHandle {
        mutex: Arc::new(Mutex::new(())),
        condvar: Arc::new(Condvar::new()),
        guard: None,
    };
    Box::into_raw(Box::new(handle)) as *mut u8
}

/// Acquire the internal mutex
#[no_mangle]
pub unsafe extern "C" fn sys_condition_acquire(condition: *mut u8) {
    if condition.is_null() {
        return;
    }

    let handle = &mut *(condition as *mut ConditionHandle);
    let guard = handle.mutex.lock().unwrap();
    // Extend lifetime to 'static - this is safe because the guard is stored in the handle
    // and will be released when sys_condition_release is called
    let guard: std::sync::MutexGuard<'static, ()> = std::mem::transmute(guard);
    handle.guard = Some(guard);
}

/// Try to acquire the internal mutex (non-blocking)
/// Returns boxed Bool (Dynamic value)
#[no_mangle]
pub unsafe extern "C" fn sys_condition_try_acquire(condition: *mut u8) -> *mut u8 {
    if condition.is_null() {
        return crate::type_system::haxe_box_bool_ptr(false);
    }

    let handle = &mut *(condition as *mut ConditionHandle);
    let result = if let Ok(guard) = handle.mutex.try_lock() {
        let guard: std::sync::MutexGuard<'static, ()> = std::mem::transmute(guard);
        handle.guard = Some(guard);
        true
    } else {
        false
    };

    crate::type_system::haxe_box_bool_ptr(result)
}

/// Release the internal mutex
#[no_mangle]
pub unsafe extern "C" fn sys_condition_release(condition: *mut u8) {
    if condition.is_null() {
        return;
    }

    let handle = &mut *(condition as *mut ConditionHandle);
    // Drop the guard to release the mutex
    handle.guard = None;
}

/// Wait on the condition variable
/// Atomically releases the mutex and blocks until signaled
#[no_mangle]
pub unsafe extern "C" fn sys_condition_wait(condition: *mut u8) {
    if condition.is_null() {
        return;
    }

    let handle = &mut *(condition as *mut ConditionHandle);

    // Take the guard out (this releases the mutex for the condvar wait)
    if let Some(guard) = handle.guard.take() {
        // Wait on the condvar - this will automatically release and reacquire the mutex
        let guard = handle.condvar.wait(guard).unwrap();
        // Store the reacquired guard
        let guard: std::sync::MutexGuard<'static, ()> = std::mem::transmute(guard);
        handle.guard = Some(guard);
    }
}

/// Signal one waiting thread
#[no_mangle]
pub unsafe extern "C" fn sys_condition_signal(condition: *mut u8) {
    if condition.is_null() {
        return;
    }

    let handle = &*(condition as *const ConditionHandle);
    handle.condvar.notify_one();
}

/// Broadcast to all waiting threads
#[no_mangle]
pub unsafe extern "C" fn sys_condition_broadcast(condition: *mut u8) {
    if condition.is_null() {
        return;
    }

    let handle = &*(condition as *const ConditionHandle);
    handle.condvar.notify_all();
}

// ============================================================================
// JIT Lifecycle Management
// ============================================================================

/// Clean up JIT-related global state
///
/// This should be called after rayzor_wait_all_threads() and before the next
/// JIT compilation to reset thread tracking state.
#[no_mangle]
pub extern "C" fn rayzor_jit_cleanup() {
    // Reset thread tracking
    ACTIVE_THREAD_COUNT.store(0, Ordering::SeqCst);

    // Clear thread registry
    if let Ok(mut registry) = THREAD_REGISTRY.lock() {
        registry.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn test_thread_fn(_env: *const u8) -> i32 {
        42
    }

    #[test]
    fn test_thread_spawn_join() {
        unsafe {
            let handle = rayzor_thread_spawn(test_thread_fn as *const u8, ptr::null());
            assert!(!handle.is_null());

            let result = rayzor_thread_join(handle);
            assert_eq!(result as *const () as usize, 42);
        }
    }

    #[test]
    fn test_arc_basic() {
        unsafe {
            let value = Box::into_raw(Box::new(42u32)) as *mut u8;
            let arc1 = rayzor_arc_init(value);
            assert!(!arc1.is_null());

            let count = rayzor_arc_strong_count(arc1);
            assert_eq!(count, 1);

            let arc2 = rayzor_arc_clone(arc1);
            assert!(!arc2.is_null());

            let count = rayzor_arc_strong_count(arc1);
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_channel_send_receive() {
        unsafe {
            let channel = rayzor_channel_init(0);
            assert!(!channel.is_null());

            let value = 42usize as *mut u8;
            rayzor_channel_send(channel, value);

            let received = rayzor_channel_receive(channel);
            assert_eq!(received as usize, 42);

            rayzor_channel_close(channel);
        }
    }
}
