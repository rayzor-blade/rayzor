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

/// Park with a TIMEOUT (nanoseconds), then return regardless. Used by the
/// channel: parking yields the CPU so a peer worker can run (a pure spin starves
/// it), while the bounded timeout guarantees the caller re-checks its condition
/// under the lock — so even if a cross-instance `memory.atomic.notify` is missed
/// (unreliable in the current wasmtime host) it can never deadlock.
#[inline]
fn futex_wait_timed(addr: *mut i32, expected: i32, timeout_ns: i64) {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        core::arch::wasm32::memory_atomic_wait32(addr, expected, timeout_ns);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (addr, expected, timeout_ns);
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
// Channel — bounded MPMC ring buffer over shared linear memory.
//
// rayzor.concurrent.Channel was UNIMPLEMENTED on the wasmtime target (no
// runtime-wasm symbol, no host fake) so every call returned 0. This is the real
// in-guest implementation, mirroring the native std::sync::Mutex-guarded
// VecDeque: a spinlock protects the head/tail/slots, and the existing futex
// (memory.atomic.wait32/notify) blocks senders while full and receivers while
// empty. All values are i32 (the boxed/object pointer in linear memory).
//
// Layout (header 20 bytes, then `capacity` i32 slots):
//   +0  lock    (AtomicU32)  spinlock: 0 unlocked, 1 locked
//   +4  head    (AtomicU32)  monotonic consume counter
//   +8  tail    (AtomicU32)  monotonic produce counter
//   +12 closed  (AtomicU32)
//   +16 capacity(i32)
//   +20 slots   [i32; capacity]
// count = tail - head (wrapping); empty = 0; full = count >= capacity.
// ============================================================================

const CH_LOCK: i32 = 0;
const CH_HEAD: i32 = 4;
const CH_TAIL: i32 = 8;
const CH_CLOSED: i32 = 12;
const CH_CAP: i32 = 16;
const CH_SLOTS: i32 = 20;

#[inline]
unsafe fn ch_lock(ch: i32) {
    let l = atomic(ch, CH_LOCK);
    while l
        .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
}
#[inline]
unsafe fn ch_unlock(ch: i32) {
    atomic(ch, CH_LOCK).store(0, Ordering::Release);
}
#[inline]
unsafe fn ch_slot_addr(ch: i32, idx: u32) -> i32 {
    CH_SLOTS + (idx as i32).wrapping_mul(4)
}

/// Allocate a channel with the given capacity. capacity<=0 has no unbounded ring
/// on wasm, so it falls back to a fixed buffer (documented limitation).
#[no_mangle]
pub extern "C" fn rayzor_channel_init(capacity: i32) -> i32 {
    let cap: i32 = if capacity > 0 { capacity } else { 256 };
    let bytes = (CH_SLOTS + cap.wrapping_mul(4)).max(CH_SLOTS);
    let h = crate::rayzor_malloc(bytes);
    if h == 0 {
        return 0;
    }
    unsafe {
        atomic(h, CH_LOCK).store(0, Ordering::Relaxed);
        atomic(h, CH_HEAD).store(0, Ordering::Relaxed);
        atomic(h, CH_TAIL).store(0, Ordering::Relaxed);
        atomic(h, CH_CLOSED).store(0, Ordering::Relaxed);
        store_word(h, CH_CAP, cap);
    }
    h
}

/// Send (blocks while full). No-op once closed.
#[no_mangle]
pub extern "C" fn rayzor_channel_send(channel: i32, value: i32) {
    if channel == 0 {
        return;
    }
    let cap = unsafe { load_word(channel, CH_CAP) } as u32;
    loop {
        unsafe {
            ch_lock(channel);
            if atomic(channel, CH_CLOSED).load(Ordering::Relaxed) != 0 {
                ch_unlock(channel);
                return;
            }
            let h = atomic(channel, CH_HEAD).load(Ordering::Relaxed);
            let t = atomic(channel, CH_TAIL).load(Ordering::Relaxed);
            if t.wrapping_sub(h) < cap {
                store_word(channel, ch_slot_addr(channel, t % cap), value);
                atomic(channel, CH_TAIL).store(t.wrapping_add(1), Ordering::Release);
                ch_unlock(channel);
                futex_notify(cell_addr(channel, CH_TAIL) as *mut i32, 1);
                return;
            }
            // Full: park on CH_HEAD (yields CPU so the consumer can drain) with a
            // short timeout — a missed cross-instance notify can't deadlock.
            ch_unlock(channel);
            futex_wait_timed(cell_addr(channel, CH_HEAD) as *mut i32, h as i32, 1_000_000);
        }
    }
}

/// Try-send (non-blocking). Returns 1 on success, 0 if full or closed.
#[no_mangle]
pub extern "C" fn rayzor_channel_try_send(channel: i32, value: i32) -> i32 {
    if channel == 0 {
        return 0;
    }
    let cap = unsafe { load_word(channel, CH_CAP) } as u32;
    unsafe {
        ch_lock(channel);
        if atomic(channel, CH_CLOSED).load(Ordering::Relaxed) != 0 {
            ch_unlock(channel);
            return 0;
        }
        let h = atomic(channel, CH_HEAD).load(Ordering::Relaxed);
        let t = atomic(channel, CH_TAIL).load(Ordering::Relaxed);
        if t.wrapping_sub(h) < cap {
            store_word(channel, ch_slot_addr(channel, t % cap), value);
            atomic(channel, CH_TAIL).store(t.wrapping_add(1), Ordering::Release);
            ch_unlock(channel);
            futex_notify(cell_addr(channel, CH_TAIL) as *mut i32, 1);
            1
        } else {
            ch_unlock(channel);
            0
        }
    }
}

/// Receive (blocks while empty). Returns 0 (null) if the channel is closed+empty.
#[no_mangle]
pub extern "C" fn rayzor_channel_receive(channel: i32) -> i32 {
    if channel == 0 {
        return 0;
    }
    let cap = unsafe { load_word(channel, CH_CAP) } as u32;
    loop {
        unsafe {
            ch_lock(channel);
            let h = atomic(channel, CH_HEAD).load(Ordering::Relaxed);
            let t = atomic(channel, CH_TAIL).load(Ordering::Relaxed);
            if t.wrapping_sub(h) > 0 {
                let v = load_word(channel, ch_slot_addr(channel, h % cap));
                atomic(channel, CH_HEAD).store(h.wrapping_add(1), Ordering::Release);
                ch_unlock(channel);
                futex_notify(cell_addr(channel, CH_HEAD) as *mut i32, 1);
                return v;
            }
            if atomic(channel, CH_CLOSED).load(Ordering::Relaxed) != 0 {
                ch_unlock(channel);
                return 0;
            }
            // Empty: park on CH_TAIL with a short timeout (see send()).
            ch_unlock(channel);
            futex_wait_timed(cell_addr(channel, CH_TAIL) as *mut i32, t as i32, 1_000_000);
        }
    }
}

/// Try-receive (non-blocking). Returns the value, or 0 if empty.
#[no_mangle]
pub extern "C" fn rayzor_channel_try_receive(channel: i32) -> i32 {
    if channel == 0 {
        return 0;
    }
    let cap = unsafe { load_word(channel, CH_CAP) } as u32;
    unsafe {
        ch_lock(channel);
        let h = atomic(channel, CH_HEAD).load(Ordering::Relaxed);
        let t = atomic(channel, CH_TAIL).load(Ordering::Relaxed);
        if t.wrapping_sub(h) > 0 {
            let v = load_word(channel, ch_slot_addr(channel, h % cap));
            atomic(channel, CH_HEAD).store(h.wrapping_add(1), Ordering::Release);
            ch_unlock(channel);
            futex_notify(cell_addr(channel, CH_HEAD) as *mut i32, 1);
            v
        } else {
            ch_unlock(channel);
            0
        }
    }
}

/// Close: mark closed and wake every blocked sender/receiver.
#[no_mangle]
pub extern "C" fn rayzor_channel_close(channel: i32) {
    if channel == 0 {
        return;
    }
    unsafe {
        atomic(channel, CH_CLOSED).store(1, Ordering::Release);
        futex_notify(cell_addr(channel, CH_TAIL) as *mut i32, u32::MAX);
        futex_notify(cell_addr(channel, CH_HEAD) as *mut i32, u32::MAX);
    }
}

#[no_mangle]
pub extern "C" fn rayzor_channel_is_closed(channel: i32) -> i32 {
    if channel == 0 {
        return 1;
    }
    unsafe { i32::from(atomic(channel, CH_CLOSED).load(Ordering::Relaxed) != 0) }
}

#[no_mangle]
pub extern "C" fn rayzor_channel_len(channel: i32) -> i32 {
    if channel == 0 {
        return 0;
    }
    unsafe {
        let h = atomic(channel, CH_HEAD).load(Ordering::Relaxed);
        let t = atomic(channel, CH_TAIL).load(Ordering::Relaxed);
        t.wrapping_sub(h) as i32
    }
}

#[no_mangle]
pub extern "C" fn rayzor_channel_capacity(channel: i32) -> i32 {
    if channel == 0 {
        return 0;
    }
    unsafe { load_word(channel, CH_CAP) }
}

#[no_mangle]
pub extern "C" fn rayzor_channel_is_empty(channel: i32) -> i32 {
    i32::from(rayzor_channel_len(channel) == 0)
}

#[no_mangle]
pub extern "C" fn rayzor_channel_is_full(channel: i32) -> i32 {
    if channel == 0 {
        return 0;
    }
    let cap = unsafe { load_word(channel, CH_CAP) };
    i32::from(rayzor_channel_len(channel) >= cap)
}

// ============================================================================
// CPU topology — wasm has no NUMA / thread affinity, so these are in-guest
// no-ops with single-node defaults. Defining them HERE (rather than leaving them
// as host imports) is what lets a Thread.spawn WORKER call CpuTopology.bindToNode
// without trapping: the worker linker (wasm_runner.rs) stubs every host IMPORT to
// a trap, but a merged in-guest function is callable. WorkerPool.withForcedNodes'
// fanout closure calls bindToNode before the user work — without this it trapped
// on the worker and the pool silently produced nothing.
// ============================================================================

/// No NUMA on wasm.
#[no_mangle]
pub extern "C" fn rayzor_topology_multi_node() -> i32 {
    0
}

/// Single node.
#[no_mangle]
pub extern "C" fn rayzor_topology_node_count() -> i32 {
    1
}

/// The guest can't introspect host cores; wasm parallelism comes from the worker
/// pool, not topology. Report one logical CPU.
#[no_mangle]
pub extern "C" fn rayzor_topology_cpu_count() -> i32 {
    1
}

/// Every CPU maps to node 0.
#[no_mangle]
pub extern "C" fn rayzor_topology_cpu_to_node(_cpu: i32) -> i32 {
    0
}

/// Affinity bind is a no-op (success) in the wasm sandbox.
#[no_mangle]
pub extern "C" fn rayzor_topology_bind_to_node(_node: i32) -> i32 {
    0
}

/// Unbind is a no-op (success).
#[no_mangle]
pub extern "C" fn rayzor_topology_unbind() -> i32 {
    0
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
