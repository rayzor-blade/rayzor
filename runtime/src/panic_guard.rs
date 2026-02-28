//! Panic guard for native runtime functions
//!
//! Wraps extern "C" runtime functions with catch_unwind to convert Rust panics
//! into Haxe exceptions instead of causing undefined behavior (panic across FFI).
//! In debug mode, the panic message includes a source-mapped stack trace.

use std::panic::{catch_unwind, AssertUnwindSafe};

/// Extract a human-readable message from a panic payload.
fn extract_panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Wrap a closure so Rust panics become Haxe exceptions instead of UB.
/// On success, returns the closure's result. On panic, throws a Haxe exception
/// with the panic message (and source-mapped trace in debug mode) via longjmp.
///
/// In release mode (stack traces disabled), bypasses `catch_unwind` entirely
/// for zero overhead — the closure is called directly.
///
/// # Usage
/// ```rust,ignore
/// pub extern "C" fn haxe_array_get(arr: *mut HaxeArray, index: i64) -> i64 {
///     guarded_call(|| { /* original impl */ })
/// }
/// ```
pub fn guarded_call<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    if !crate::native_stack_trace::is_enabled() {
        // Release mode: zero overhead, no catch_unwind
        return f();
    }

    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let msg = extract_panic_message(&payload);

            let bt = backtrace::Backtrace::new();
            let trace = crate::native_stack_trace::resolve_backtrace_to_source(&bt);
            let full_msg = if trace.is_empty() {
                format!("Runtime panic: {}", msg)
            } else {
                format!("Runtime panic: {}\n{}", msg, trace)
            };

            crate::exception::throw_with_message(full_msg);
        }
    }
}
