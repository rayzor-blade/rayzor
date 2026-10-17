//! `eval.vm.Gc` finalisation for rayzor's `eval` std. A registered target
//! counts as collected once no handle besides its own is counted and the
//! handle no longer appears on the calling thread's stack. References held
//! in other heap objects are not seen, so this is approximate: it serves the
//! `eval` API, not a collector.

use std::sync::{Arc, Mutex, Weak};

use crate::anon_object::AnonObject;

struct Registration {
    /// The callback closure `{ fn_ptr, env_ptr }`, called with null.
    callback: usize,
    /// The target's handle, compared against stack words.
    handle: usize,
    /// Keeps the target alive while the registration does not know it gone.
    target: Weak<AnonObject>,
    /// The registration's own count on the target, released when it fires.
    held: Option<Arc<AnonObject>>,
}

// SAFETY: registrations hold opaque addresses; callbacks run on the thread
// that requests collection, as the eval API does.
unsafe impl Send for Registration {}

static REGISTRATIONS: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

/// `Gc.finalise(f, v)`: run `f` once `v` is no longer reachable. Only
/// anonymous objects are tracked; any other target is never reported.
///
/// # Safety
/// `callback` must point to a closure object `{ fn_ptr, env_ptr }`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rayzor_gc_finalise(callback: *const u8, target: *mut u8) {
    if callback.is_null() {
        return;
    }
    let Some(weak) = crate::anon_object::weak_of_handle(target) else {
        return;
    };
    let held = weak.upgrade();
    if let Ok(mut registrations) = REGISTRATIONS.lock() {
        registrations.push(Registration {
            callback: callback as usize,
            handle: target as usize,
            target: weak,
            held,
        });
    }
}

#[cfg(target_os = "linux")]
fn stack_base() -> Option<usize> {
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
            return None;
        }
        let mut addr: *mut libc::c_void = std::ptr::null_mut();
        let mut size: libc::size_t = 0;
        let ok = libc::pthread_attr_getstack(&attr, &mut addr, &mut size) == 0;
        libc::pthread_attr_destroy(&mut attr);
        ok.then(|| addr as usize + size)
    }
}

#[cfg(target_os = "macos")]
fn stack_base() -> Option<usize> {
    unsafe extern "C" {
        fn pthread_get_stackaddr_np(thread: libc::pthread_t) -> *mut libc::c_void;
    }
    Some(unsafe { pthread_get_stackaddr_np(libc::pthread_self()) } as usize)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn stack_base() -> Option<usize> {
    None
}

/// Whether `value` is stored in any stack word between `low` and `high`.
fn on_stack(value: usize, low: usize, high: usize) -> bool {
    let mut at = low & !7;
    while at + 8 <= high {
        if unsafe { *(at as *const usize) } == value {
            return true;
        }
        at += 8;
    }
    false
}

// `Gc.full_major()` and the other collection requests: run the callbacks of
// the targets that are gone. The entry point hands its stack pointer at
// entry to the scan, so the scan starts at the caller's frame: registers the
// caller left behind and this code's own frames are not read, where a dead
// copy of a handle would keep its target alive.
#[cfg(all(target_arch = "x86_64", not(windows), not(target_os = "macos")))]
std::arch::global_asm!(
    ".globl rayzor_gc_collect",
    "rayzor_gc_collect:",
    "lea rdi, [rsp + 8]",
    "jmp {collect}",
    collect = sym collect_from,
);
#[cfg(all(target_arch = "x86_64", target_os = "macos"))]
std::arch::global_asm!(
    ".globl _rayzor_gc_collect",
    "_rayzor_gc_collect:",
    "lea rdi, [rsp + 8]",
    "jmp {collect}",
    collect = sym collect_from,
);
#[cfg(all(target_arch = "x86_64", windows))]
std::arch::global_asm!(
    ".globl rayzor_gc_collect",
    "rayzor_gc_collect:",
    "lea rcx, [rsp + 8]",
    "jmp {collect}",
    collect = sym collect_from,
);
#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
std::arch::global_asm!(
    ".globl rayzor_gc_collect",
    "rayzor_gc_collect:",
    "mov x0, sp",
    "b {collect}",
    collect = sym collect_from,
);
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
std::arch::global_asm!(
    ".globl _rayzor_gc_collect",
    "_rayzor_gc_collect:",
    "mov x0, sp",
    "b {collect}",
    collect = sym collect_from,
);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
unsafe extern "C" {
    /// Run the finalisers that are due (see the entry point above).
    pub fn rayzor_gc_collect();
}

/// Elsewhere the scan starts at this frame.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_gc_collect() {
    let here = 0usize;
    collect_above(&here as *const usize as usize);
}

extern "C" fn collect_from(caller_sp: usize) {
    collect_above(caller_sp);
}

#[inline(never)]
fn collect_above(low: usize) {
    let Some(high) = stack_base().filter(|high| *high > low) else {
        return;
    };
    let due: Vec<Registration> = match REGISTRATIONS.lock() {
        Ok(mut registrations) => {
            let (dead, live): (Vec<_>, Vec<_>) = std::mem::take(&mut *registrations)
                .into_iter()
                .partition(|r| {
                    // Our own count is the one `held` carries.
                    let others = r
                        .target
                        .strong_count()
                        .saturating_sub(usize::from(r.held.is_some()));
                    others == 0 || (others == 1 && !on_stack(r.handle, low, high))
                });
            *registrations = live;
            dead
        }
        Err(_) => return,
    };
    // Called outside the lock: a callback may register more.
    for registration in due.into_iter().rev() {
        type ClosureFn = extern "C" fn(*const u8, i64) -> i64;
        let (func_addr, env_addr) = unsafe {
            let words = registration.callback as *const usize;
            (*words, *words.add(1))
        };
        drop(registration.held);
        let func: ClosureFn = unsafe { std::mem::transmute(func_addr) };
        func(env_addr as *const u8, 0);
    }
}
