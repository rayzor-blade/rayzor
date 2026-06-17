//! Real, thread-safe `Arc` and `Mutex` for the wasm runtime.
//!
//! These replace the host-side fakes that `wasm_runner.rs` used to inject
//! (identity-passthrough `Arc` with `strong_count` hard-wired to 1, and a
//! non-atomic `bool` `Mutex` living in the MAIN instance's host state that the
//! worker instances could never even see). Because every worker thread runs its
//! own wasmtime `Store`/`Instance` over the *same* shared linear memory, the
//! only place a real cross-worker primitive can live is here, in guest code:
//! the wasm linker merges this crate into the user module, so these functions
//! are present in every instance and operate on one shared `AtomicU32` cell.
//! This is the same mechanism the `Tensor` refcount (`tensor.rs`) already uses.
//!
//! ABI: every handle is an `i32` linear-memory address (the wasm32 ABI). Each
//! exported fn is `i32`-in / `i32`-out, or `()` for `void`, to MATCH the user
//! module's import signatures — `ir_type_to_wasm` lowers `PtrU8`/`U64`/`Bool`
//! all to `i32` and `Void` to no-result. A signature that differs (e.g. a
//! `u64`/`i64` return on `strong_count`, or an `i32` return on `unlock`) is
//! SILENTLY stubbed to an unreachable trap by `wasm_linker.rs`, so the
//! native-side `u64`/`bool` returns become `i32` here on purpose.

use core::sync::atomic::{fence, AtomicU32, Ordering};

// ============================================================================
// Address helpers
// ============================================================================

/// Compute the absolute linear-memory byte address of a cell field.
///
/// `handle` is a wasm32 address that may have its high bit set once the heap
/// grows past 2 GiB; it MUST be treated as unsigned (`i32 as u32 as usize`)
/// or the cast sign-extends and the access goes wild. (See the 2^31 signed
/// pointer-boundary note in the wasm OOM history.)
#[inline(always)]
fn cell_addr(handle: i32, off: i32) -> usize {
    (handle as u32 as usize) + (off as u32 as usize)
}

/// Borrow the `AtomicU32` living at `handle + off`. Caller guarantees the cell
/// was allocated 4-byte-aligned (it is: `rayzor_malloc` returns 8-aligned).
#[inline(always)]
unsafe fn atomic(handle: i32, off: i32) -> &'static AtomicU32 {
    &*(cell_addr(handle, off) as *const AtomicU32)
}

/// Plain (non-atomic) i32 load of a cell field. Used for the value pointer,
/// whose identity is stable for the lifetime of the handle.
#[inline(always)]
unsafe fn load_word(handle: i32, off: i32) -> i32 {
    *(cell_addr(handle, off) as *const i32)
}

/// Plain (non-atomic) i32 store of a cell field. Only used at `init`, before
/// the handle is published/shared.
#[inline(always)]
unsafe fn store_word(handle: i32, off: i32, v: i32) {
    *(cell_addr(handle, off) as *mut i32) = v;
}

// ============================================================================
// Arc — atomic reference count over a shared cell
//
// Layout (8 bytes, 8-aligned via rayzor_malloc):
//   +0 : strong  (AtomicU32)   live clone count
//   +4 : value   (i32)         the wrapped payload pointer
//
// Mirrors std::sync::Arc<*mut u8>: clones share one inner cell; the wrapped
// `value` pointer is the payload, not freed by the Arc inner (matching native,
// where Arc's Drop frees the inner box, not the *mut u8 pointee).
// ============================================================================

const ARC_STRONG: i32 = 0;
const ARC_VALUE: i32 = 4;

/// Allocate an Arc inner wrapping `value`, refcount 1. Returns the handle.
#[no_mangle]
pub extern "C" fn rayzor_arc_init(value: i32) -> i32 {
    let h = crate::rayzor_malloc(8);
    if h == 0 {
        return 0;
    }
    unsafe {
        // Not yet shared: a plain store of the initial count is sufficient.
        atomic(h, ARC_STRONG).store(1, Ordering::Relaxed);
        store_word(h, ARC_VALUE, value);
    }
    h
}

/// Increment the strong count and return the SAME handle (clones alias one
/// inner cell). `Relaxed` is sufficient for the increment — the new reference
/// is published through whatever synchronization hands the clone to another
/// thread (identical to `std::sync::Arc::clone`).
#[no_mangle]
pub extern "C" fn rayzor_arc_clone(arc: i32) -> i32 {
    if arc == 0 {
        return 0;
    }
    unsafe {
        atomic(arc, ARC_STRONG).fetch_add(1, Ordering::Relaxed);
    }
    arc
}

/// Return the wrapped payload pointer.
#[no_mangle]
pub extern "C" fn rayzor_arc_get(arc: i32) -> i32 {
    if arc == 0 {
        return 0;
    }
    unsafe { load_word(arc, ARC_VALUE) }
}

/// Current strong count. Native returns `u64`; on wasm that is carried as `i32`.
#[no_mangle]
pub extern "C" fn rayzor_arc_strong_count(arc: i32) -> i32 {
    if arc == 0 {
        return 0;
    }
    unsafe { atomic(arc, ARC_STRONG).load(Ordering::SeqCst) as i32 }
}

/// Identity address of the wrapped payload (for pointer comparison). Native
/// returns `u64`; on wasm that is carried as `i32`.
#[no_mangle]
pub extern "C" fn rayzor_arc_as_ptr(arc: i32) -> i32 {
    if arc == 0 {
        return 0;
    }
    unsafe { load_word(arc, ARC_VALUE) }
}

/// Consume the Arc if it is the sole owner: succeeds (returns the payload)
/// only when the strong count is exactly 1, mirroring `Arc::try_unwrap`.
/// On failure returns 0 (null) and leaves the count untouched.
#[no_mangle]
pub extern "C" fn rayzor_arc_try_unwrap(arc: i32) -> i32 {
    if arc == 0 {
        return 0;
    }
    unsafe {
        let strong = atomic(arc, ARC_STRONG);
        if strong
            .compare_exchange(1, 0, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            let v = load_word(arc, ARC_VALUE);
            // The inner cell is now dead. The runtime does not reclaim memory
            // (rayzor_free is a no-op), so we leak it — consistent with the
            // rest of the wasm runtime and harmless given its lifetime ends.
            v
        } else {
            0
        }
    }
}

/// Decrement the strong count; on the last drop (count hit 0) the inner cell is
/// dead. The `Release` on decrement + `Acquire` fence on the final drop is the
/// canonical `std::sync::Arc` discipline (on wasm both collapse to seqcst, but
/// the structure is kept for native-parity/portability).
///
/// NOTE: the compiler does not currently EMIT a drop at Arc end-of-scope (the
/// `rayzor_arc_drop` extern is not declared in `sync.rs`), so in generated code
/// the count climbs monotonically exactly as on native. This symbol exists so
/// the primitive is complete and testable, and so a future compiler change that
/// emits drops gets a correct free path here.
#[no_mangle]
pub extern "C" fn rayzor_arc_drop(arc: i32) {
    if arc == 0 {
        return;
    }
    unsafe {
        if atomic(arc, ARC_STRONG).fetch_sub(1, Ordering::Release) == 1 {
            fence(Ordering::Acquire);
            // Last reference gone; inner cell dead. Leak (no reclamation).
        }
    }
}

// ============================================================================
// Mutex — 3-state futex lock over a shared cell
//
// Layout (8 bytes, 8-aligned):
//   +0 : state  (AtomicU32)   0 = unlocked
//                             1 = locked, no waiters
//                             2 = locked, maybe waiters
//   +4 : value  (i32)         the guarded payload pointer
//
// The guard handle returned by lock()/try_lock() IS the mutex handle, so
// guard_get/unlock operate on the same address. This is the canonical
// Linux-futex / "Rust Atomics and Locks" 3-state mutex: an uncontended
// lock/unlock is a single CAS / single swap with NO park; only real contention
// pays a wait/notify.
// ============================================================================

const MUTEX_STATE: i32 = 0;
const MUTEX_VALUE: i32 = 4;

/// Park the current thread until the word at `addr` changes away from
/// `expected`. On wasm this is a real `memory.atomic.wait32` (blocks); off-wasm
/// (host tests) it degrades to a yield so the surrounding spin still makes
/// progress.
#[inline]
fn futex_wait(addr: *mut i32, expected: i32) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        // timeout -1 == wait forever; returns 0 woken / 1 not-equal / 2 timeout.
        core::arch::wasm32::memory_atomic_wait32(addr, expected, -1);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (addr, expected);
        std::thread::yield_now();
    }
}

/// Wake up to `count` threads parked on the word at `addr`.
#[inline]
fn futex_notify(addr: *mut i32, count: u32) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        core::arch::wasm32::memory_atomic_notify(addr, count);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (addr, count);
    }
}

/// Allocate a mutex wrapping `value`, initially unlocked. Returns the handle.
#[no_mangle]
pub extern "C" fn rayzor_mutex_init(value: i32) -> i32 {
    let h = crate::rayzor_malloc(8);
    if h == 0 {
        return 0;
    }
    unsafe {
        atomic(h, MUTEX_STATE).store(0, Ordering::Relaxed);
        store_word(h, MUTEX_VALUE, value);
    }
    h
}

/// Slow path: bounded spin (cheap brief holds), then park via futex until the
/// lock is free, marking the state "maybe waiters" so the unlocker knows to
/// wake us.
#[cold]
fn mutex_lock_contended(state: &AtomicU32, addr: *mut i32) {
    let mut spins = 0u32;
    // Spin only while locked-without-waiters; if a waiter is already registered
    // (state == 2) there's no point spinning, go straight to parking.
    while state.load(Ordering::Relaxed) == 1 && spins < 100 {
        spins += 1;
        core::hint::spin_loop();
    }
    // One cheap try before committing to the waiter protocol.
    if state
        .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
    {
        return;
    }
    // Contended: claim with "maybe waiters" (2). swap returns the prior state;
    // if it was 0 the lock was actually free and we now hold it. Otherwise park
    // until someone releases (state -> 0) and notifies.
    while state.swap(2, Ordering::Acquire) != 0 {
        futex_wait(addr, 2);
    }
}

/// Acquire the lock, blocking until available. Returns the handle as the guard.
#[no_mangle]
pub extern "C" fn rayzor_mutex_lock(mutex: i32) -> i32 {
    if mutex == 0 {
        return 0;
    }
    unsafe {
        let state = atomic(mutex, MUTEX_STATE);
        // Fast path: uncontended acquire is a single CAS, no syscall.
        if state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            let addr = cell_addr(mutex, MUTEX_STATE) as *mut i32;
            mutex_lock_contended(state, addr);
        }
    }
    mutex // guard == mutex handle
}

/// Try to acquire without blocking. Returns the guard handle on success, 0 if
/// already held (matching native's null-guard-on-failure).
#[no_mangle]
pub extern "C" fn rayzor_mutex_try_lock(mutex: i32) -> i32 {
    if mutex == 0 {
        return 0;
    }
    unsafe {
        if atomic(mutex, MUTEX_STATE)
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            mutex
        } else {
            0
        }
    }
}

/// Whether the lock is currently held. Native returns `bool`; wasm carries `i32`.
#[no_mangle]
pub extern "C" fn rayzor_mutex_is_locked(mutex: i32) -> i32 {
    if mutex == 0 {
        return 0;
    }
    unsafe { i32::from(atomic(mutex, MUTEX_STATE).load(Ordering::Relaxed) != 0) }
}

/// Read the guarded payload pointer (guard handle == mutex handle).
#[no_mangle]
pub extern "C" fn rayzor_mutex_guard_get(guard: i32) -> i32 {
    if guard == 0 {
        return 0;
    }
    unsafe { load_word(guard, MUTEX_VALUE) }
}

/// Release the lock. If the prior state was "maybe waiters" (2), wake exactly
/// one parked thread. Returns `void` (NO wasm result — must match the import).
#[no_mangle]
pub extern "C" fn rayzor_mutex_unlock(guard: i32) {
    if guard == 0 {
        return;
    }
    unsafe {
        let state = atomic(guard, MUTEX_STATE);
        if state.swap(0, Ordering::Release) == 2 {
            let addr = cell_addr(guard, MUTEX_STATE) as *mut i32;
            futex_notify(addr, 1);
        }
    }
}

// ============================================================================
// sys.thread.Mutex — value-less OS lock, reusing the same cell (value unused)
// ============================================================================

/// Allocate a value-less lock.
#[no_mangle]
pub extern "C" fn sys_mutex_alloc() -> i32 {
    rayzor_mutex_init(0)
}

/// Acquire (blocking).
#[no_mangle]
pub extern "C" fn sys_mutex_acquire(mutex: i32) {
    rayzor_mutex_lock(mutex);
}

/// Try-acquire; returns 1 on success, 0 if held (Bool -> i32).
#[no_mangle]
pub extern "C" fn sys_mutex_try_acquire(mutex: i32) -> i32 {
    i32::from(rayzor_mutex_try_lock(mutex) != 0)
}

/// Release.
#[no_mangle]
pub extern "C" fn sys_mutex_release(mutex: i32) {
    rayzor_mutex_unlock(mutex);
}

// ============================================================================
// Verification (this crate does not build for the host target — a pre-existing
// cfg gap, `rayzor_host_flash_attn_q8_par` — so `cargo test` here is not
// runnable):
//   * Algorithm correctness (refcount race -> count exactly 1; mutex exclusion
//     -> N*M with no lost updates) is proven on the host by a standalone mirror
//     of these exact atomic state machines.
//   * The wasm lowering (i32.atomic.rmw.* + memory.atomic.wait32/notify over
//     `shared` memory) and the linker resolution (symbols become internal, no
//     residual `rayzor_arc`/`rayzor_mutex` imports) are verified end-to-end by a
//     `rayzor run --wasm` Arc/Mutex program whose output matches the native run
//     byte-for-byte (strongCount reflects real clones, not the fake's constant 1).
// ============================================================================
