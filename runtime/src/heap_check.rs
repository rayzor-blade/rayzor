//! Heap integrity probes for hunting the content-dependent SIGTRAP at
//! bugs_llama_chat_heap_corruption_confirmed.
//!
//! macOS's libsystem_malloc exposes `malloc_zone_check`, which walks every
//! parked freelist in the given zone (or all zones, with NULL) and returns
//! 1 if every linked-list pointer / tag validates, 0 on corruption.
//!
//! Sprinkle `check_heap("after X")` calls after each suspect kernel: when
//! the first label fires we've localised the corrupting op to that span of
//! code. The check is gated behind `RAYZOR_HEAP_CHECK=1` and is a no-op
//! otherwise (single relaxed atomic load).
//!
//! Cost when active: O(total parked chunks). On a long-running rayzor
//! process this is microseconds-to-low-milliseconds per call; fine for
//! bisection-grade tracing, not for production.

use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "macos")]
extern "C" {
    fn malloc_zone_check(zone: *mut std::ffi::c_void) -> i32;
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: std::sync::Once = std::sync::Once::new();

#[inline(always)]
fn enabled() -> bool {
    INIT.call_once(|| {
        if std::env::var("RAYZOR_HEAP_CHECK").is_ok() {
            ENABLED.store(true, Ordering::Relaxed);
            eprintln!("[heap-check] enabled (RAYZOR_HEAP_CHECK=1)");
        }
    });
    ENABLED.load(Ordering::Relaxed)
}

/// Validate every macOS malloc zone. Prints `[heap-check] CORRUPT at {label}`
/// and aborts on failure so the lldb trap + backtrace can be captured.
/// Cheap no-op when `RAYZOR_HEAP_CHECK` is unset.
#[inline]
pub fn check_heap(label: &str) {
    if !enabled() {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        let ok = unsafe { malloc_zone_check(std::ptr::null_mut()) };
        if ok == 0 {
            eprintln!("[heap-check] CORRUPT at {}", label);
            std::process::abort();
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = label;
    }
}

/// Drop-guard wrapper. `let _g = HeapCheckGuard::new("kernel_x");` at the
/// top of a function runs `check_heap` on every exit path automatically.
pub struct HeapCheckGuard {
    // Only read by the macOS zone check; the guard is inert elsewhere.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    label: &'static str,
}

impl HeapCheckGuard {
    #[inline]
    pub fn new(label: &'static str) -> Self {
        // Check on ENTRY too — if the heap was already bad when this
        // function was called, the previous kernel did the damage. The
        // "ENTER" label distinguishes from the "EXIT" check via Drop.
        if enabled() {
            #[cfg(target_os = "macos")]
            {
                let ok = unsafe { malloc_zone_check(std::ptr::null_mut()) };
                if ok == 0 {
                    eprintln!("[heap-check] CORRUPT on ENTER {}", label);
                    std::process::abort();
                }
            }
        }
        Self { label }
    }
}

impl Drop for HeapCheckGuard {
    #[inline]
    fn drop(&mut self) {
        if enabled() {
            #[cfg(target_os = "macos")]
            {
                let ok = unsafe { malloc_zone_check(std::ptr::null_mut()) };
                if ok == 0 {
                    eprintln!("[heap-check] CORRUPT on EXIT {}", self.label);
                    std::process::abort();
                }
            }
        }
    }
}
