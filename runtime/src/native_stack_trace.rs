//! NativeStackTrace runtime — Rust backtrace capture for haxe.NativeStackTrace
//!
//! Provides backtrace capture at throw time, accessible via the NativeStackTrace API.

use crate::exception::get_exception_stack_trace;
use crate::haxe_array::HaxeArray;
use crate::haxe_string::HaxeString;
use std::alloc::{alloc, Layout};

/// Save the current stack trace for the given exception.
/// Called automatically at throw time. The exception parameter is unused
/// in our implementation — we store the trace in thread-local ExceptionState.
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_save_stack(_exception: i64) {
    // Stack is already captured in rayzor_throw_typed, nothing extra needed
}

/// Helper: create a heap-allocated HaxeString from a Rust String.
fn make_haxe_string(s: String) -> *mut HaxeString {
    let bytes = s.into_bytes();
    let len = bytes.len();
    unsafe {
        let layout = Layout::from_size_align_unchecked(len + 1, 1);
        let ptr = alloc(layout);
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
        *ptr.add(len) = 0; // null terminator
        Box::into_raw(Box::new(HaxeString {
            ptr,
            len,
            cap: len + 1,
        }))
    }
}

/// Capture and return the current call stack as a HaxeString pointer.
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_call_stack() -> *mut u8 {
    let bt = backtrace::Backtrace::new();
    let trace_str = format!("{:?}", bt);
    make_haxe_string(trace_str) as *mut u8
}

/// Return the stored exception stack trace as a HaxeString pointer.
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_exception_stack() -> *mut u8 {
    let trace_str = get_exception_stack_trace();
    make_haxe_string(trace_str) as *mut u8
}

/// Convert a native stack trace to a Haxe Array<StackItem>.
/// V1: returns an empty HaxeArray (full StackItem conversion deferred).
#[no_mangle]
pub extern "C" fn rayzor_native_stack_trace_to_haxe(_native_trace: *mut u8, _skip: i32) -> *mut u8 {
    // Return empty array — allocate on heap
    unsafe {
        let arr = Box::into_raw(Box::new(std::mem::zeroed::<HaxeArray>()));
        crate::haxe_array::haxe_array_new(arr, 8);
        arr as *mut u8
    }
}
