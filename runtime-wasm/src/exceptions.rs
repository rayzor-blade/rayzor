//! Exception-handler bookkeeping for the setjmp-shaped try/catch MIR emits.
//!
//! wasm has no setjmp, so `_setjmp` always reports the normal path and a
//! throw traps in `rayzor_throw_typed`: a try body runs, its catch never does.

use std::cell::Cell;

thread_local! {
    static DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Stands in for a jmp_buf; `_setjmp` never writes it.
static JMP_BUF: [u64; 32] = [0; 32];

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_exception_push_handler() -> i32 {
    DEPTH.with(|d| d.set(d.get() + 1));
    JMP_BUF.as_ptr() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_exception_pop_handler() {
    DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
}

#[unsafe(no_mangle)]
pub extern "C" fn _setjmp(_buf: i32) -> i32 {
    0
}

/// No exception is ever caught, so there is none current.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_get_exception() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_get_exception_type_id() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_exception_type_matches(actual: i32, expected: i32) -> i32 {
    if actual == expected {
        return 1;
    }
    if actual < 1000 || expected < 1000 {
        return 0;
    }
    crate::type_system::haxe_class_implements(actual as i64, expected as i64)
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_null_access() {
    let msg = b"Null Object Reference";
    let s = crate::wasm_string_of(msg);
    crate::rayzor_throw_typed(s, 5);
}

/// Stack traces are not recorded on wasm; these return empty values.
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_native_stack_trace_call_stack() -> i32 {
    crate::wasm_string_of(b"")
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_native_stack_trace_exception_stack() -> i32 {
    crate::wasm_string_of(b"")
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_native_stack_trace_to_haxe(_trace: i32, _skip: i32) -> i32 {
    crate::haxe_array_new()
}
