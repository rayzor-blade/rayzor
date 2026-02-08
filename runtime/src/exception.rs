//! Exception handling runtime using setjmp/longjmp
//!
//! Thread-local handler stack enables try/catch without modifying codegen backends.
//! The compiler emits ordinary function calls + conditional branches.

use std::cell::RefCell;

/// Size of jmp_buf on macOS/aarch64 and x86_64 (both need ~200 bytes, we use 256 for safety)
const JMP_BUF_SIZE: usize = 256;

extern "C" {
    fn _setjmp(buf: *mut u8) -> i32;
    fn _longjmp(buf: *mut u8, val: i32) -> !;
}

struct ExceptionHandler {
    jmp_buf: [u8; JMP_BUF_SIZE],
}

struct ExceptionState {
    handlers: Vec<ExceptionHandler>,
    current_exception: i64,
}

thread_local! {
    static STATE: RefCell<ExceptionState> = RefCell::new(ExceptionState {
        handlers: Vec::new(),
        current_exception: 0,
    });
}

/// Push a new exception handler. Returns a pointer to the jmp_buf
/// that the compiler should pass to _setjmp.
#[no_mangle]
pub extern "C" fn rayzor_exception_push_handler() -> *mut u8 {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.handlers.push(ExceptionHandler {
            jmp_buf: [0u8; JMP_BUF_SIZE],
        });
        let handler = state.handlers.last_mut().unwrap();
        handler.jmp_buf.as_mut_ptr()
    })
}

/// Pop the current exception handler (called on normal try-block exit).
#[no_mangle]
pub extern "C" fn rayzor_exception_pop_handler() {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.handlers.pop();
    });
}

/// Throw an exception. Stores the exception value and longjmps to the
/// most recent handler. If no handler exists, aborts.
#[no_mangle]
pub extern "C" fn rayzor_throw(exception_value: i64) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.current_exception = exception_value;

        if let Some(handler) = state.handlers.last_mut() {
            let buf_ptr = handler.jmp_buf.as_mut_ptr();
            // Must drop the borrow before longjmp
            drop(state);
            unsafe {
                _longjmp(buf_ptr, 1);
            }
        } else {
            eprintln!("Uncaught exception: {}", exception_value);
            std::process::abort();
        }
    });
}

/// Get the current exception value (called after landing in catch block).
#[no_mangle]
pub extern "C" fn rayzor_get_exception() -> i64 {
    STATE.with(|state| state.borrow().current_exception)
}
