//! Exception handling runtime using setjmp/longjmp
//!
//! Thread-local handler stack enables try/catch without modifying codegen backends.
//! The compiler emits ordinary function calls + conditional branches.

use std::cell::RefCell;

/// Size of jmp_buf on macOS/aarch64 and x86_64 (both need ~200 bytes, we use 256 for safety)
const JMP_BUF_SIZE: usize = 256;

/// Dynamic type_id used when no specific type is known
const TYPE_DYNAMIC: u32 = 5;

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
    current_exception_type_id: u32,
    current_stack_trace: String,
}

thread_local! {
    static STATE: RefCell<ExceptionState> = const { RefCell::new(ExceptionState {
        handlers: Vec::new(),
        current_exception: 0,
        current_exception_type_id: TYPE_DYNAMIC,
        current_stack_trace: String::new(),
    }) };
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
/// Sets type_id to Dynamic (5) for backward compatibility.
#[no_mangle]
pub extern "C" fn rayzor_throw(exception_value: i64) {
    rayzor_throw_typed(exception_value, TYPE_DYNAMIC);
}

/// Throw a typed exception. Stores both the value and its runtime type_id,
/// then longjmps to the most recent handler.
#[no_mangle]
pub extern "C" fn rayzor_throw_typed(exception_value: i64, type_id: u32) {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        state.current_exception = exception_value;
        state.current_exception_type_id = type_id;
        // Capture shadow call stack at throw time (debug mode only).
        // Uses the shadow stack maintained by push/pop_call_frame instrumentation.
        if crate::native_stack_trace::is_enabled() {
            state.current_stack_trace = crate::native_stack_trace::capture_shadow_stack();
        } else {
            state.current_stack_trace = String::new();
        }

        if let Some(handler) = state.handlers.last_mut() {
            let buf_ptr = handler.jmp_buf.as_mut_ptr();

            // Must drop the borrow before longjmp
            drop(state);
            unsafe {
                _longjmp(buf_ptr, 1);
            }
        } else {
            let formatted =
                format_uncaught_exception(exception_value, state.current_exception_type_id);
            eprintln!("Uncaught exception: {}", formatted);
            if !state.current_stack_trace.is_empty() {
                eprintln!("{}", state.current_stack_trace);
            }
            std::process::exit(1);
        }
    });
}

/// Get the current exception value (called after landing in catch block).
#[no_mangle]
pub extern "C" fn rayzor_get_exception() -> i64 {
    STATE.with(|state| state.borrow().current_exception)
}

/// Get the runtime type_id of the current exception.
/// Used by typed catch blocks to dispatch to the correct handler.
#[no_mangle]
pub extern "C" fn rayzor_get_exception_type_id() -> u32 {
    STATE.with(|state| state.borrow().current_exception_type_id)
}

/// Get the stored exception stack trace (for NativeStackTrace module).
pub fn get_exception_stack_trace() -> String {
    STATE.with(|state| state.borrow().current_stack_trace.clone())
}

/// Throw an exception with a string message (used by panic guard).
/// Creates a HaxeString, boxes it as DynamicValue, and throws as TYPE_STRING.
pub fn throw_with_message(msg: String) -> ! {
    let haxe_str = crate::native_stack_trace::make_haxe_string(msg);
    let boxed = crate::type_system::haxe_box_reference_ptr(haxe_str as *mut u8, 5); // TYPE_STRING = TypeId(5)
    rayzor_throw_typed(boxed as i64, 5); // TYPE_STRING = 5
    unreachable!()
}

fn normalize_thrown_type_id(type_id: u32) -> u32 {
    if type_id >= crate::type_system::TYPE_USER_START {
        type_id - crate::type_system::TYPE_USER_START
    } else {
        type_id
    }
}

fn read_haxe_string(hs_ptr: *const crate::haxe_string::HaxeString) -> Option<String> {
    if hs_ptr.is_null() {
        return None;
    }

    unsafe {
        let hs = &*hs_ptr;
        if hs.ptr.is_null() {
            return Some(String::new());
        }
        let bytes = std::slice::from_raw_parts(hs.ptr, hs.len);
        Some(String::from_utf8_lossy(bytes).to_string())
    }
}

fn try_format_class_exception_message(exception_value: i64, raw_type_id: u32) -> Option<String> {
    if exception_value == 0 {
        return None;
    }

    let type_info = crate::type_system::get_type_info(crate::type_system::TypeId(raw_type_id))?;
    let class_info = type_info.class_info?;
    let message_index = class_info
        .instance_fields
        .iter()
        .position(|field| *field == "message")?;

    // Runtime RTTI may include "__type_id" in instance_fields (older registration path)
    // or omit it (newer path). Compute the correct object slot for both layouts.
    let has_header_field = class_info
        .instance_fields
        .first()
        .map(|name| *name == "__type_id")
        .unwrap_or(false);
    let message_slot = if has_header_field {
        message_index
    } else {
        message_index + 1
    };
    let message_ptr = unsafe {
        let obj_ptr = exception_value as *const i64;
        if obj_ptr.is_null() {
            return None;
        }
        *obj_ptr.add(message_slot) as *const crate::haxe_string::HaxeString
    };

    read_haxe_string(message_ptr).map(|msg| format!("Exception: \"{}\"", msg))
}

fn try_format_exception_like_message_slot(
    exception_value: i64,
    raw_type_id: u32,
) -> Option<String> {
    if exception_value == 0 {
        return None;
    }

    unsafe {
        let obj_ptr = exception_value as *const i64;
        if obj_ptr.is_null() {
            return None;
        }

        // Ensure this really looks like the thrown class object layout:
        // slot 0 is runtime class type_id.
        let header_tid = *obj_ptr as u32;
        if header_tid != raw_type_id {
            return None;
        }

        // haxe.Exception layout puts message in slot 1.
        let message_ptr = *obj_ptr.add(1) as *const crate::haxe_string::HaxeString;
        if message_ptr.is_null() || (message_ptr as usize) < 0x10000 {
            return None;
        }

        read_haxe_string(message_ptr).map(|msg| format!("Exception: \"{}\"", msg))
    }
}

fn try_format_boxed_string(exception_value: i64) -> Option<String> {
    if exception_value == 0 {
        return None;
    }

    unsafe {
        let dynamic = *(exception_value as *const crate::type_system::DynamicValue);
        if dynamic.type_id == crate::type_system::TYPE_STRING && !dynamic.value_ptr.is_null() {
            return read_haxe_string(dynamic.value_ptr as *const crate::haxe_string::HaxeString);
        }
    }

    None
}

fn format_uncaught_exception(exception_value: i64, thrown_type_id: u32) -> String {
    let raw_type_id = normalize_thrown_type_id(thrown_type_id);

    // Class throws use +1000 encoded IDs; decode and extract the `message` field if present.
    if thrown_type_id >= crate::type_system::TYPE_USER_START {
        if let Some(msg) = try_format_class_exception_message(exception_value, raw_type_id) {
            return msg;
        }
        if let Some(msg) = try_format_exception_like_message_slot(exception_value, raw_type_id) {
            return msg;
        }
    }

    if raw_type_id == crate::type_system::TYPE_STRING.0 {
        if let Some(msg) = try_format_boxed_string(exception_value) {
            return msg;
        }
        if let Some(msg) =
            read_haxe_string(exception_value as *const crate::haxe_string::HaxeString)
        {
            return msg;
        }
    }
    if raw_type_id == crate::type_system::TYPE_INT.0 {
        return exception_value.to_string();
    }
    if raw_type_id == crate::type_system::TYPE_BOOL.0 {
        return if exception_value != 0 {
            "true".to_string()
        } else {
            "false".to_string()
        };
    }
    if raw_type_id == crate::type_system::TYPE_FLOAT.0 {
        return (exception_value as f64).to_string();
    }

    exception_value.to_string()
}

/// Polymorphic type matching for catch dispatch.
/// Returns 1 if actual_type_id matches expected_type_id (including via inheritance), 0 otherwise.
/// Both IDs use the +1000 offset convention from runtime_type_id().
#[no_mangle]
pub extern "C" fn rayzor_exception_type_matches(actual_type_id: i32, expected_type_id: i32) -> i32 {
    if actual_type_id == expected_type_id {
        return 1;
    }
    // Only class types use +1000 offset
    if actual_type_id < 1000 || expected_type_id < 1000 {
        return 0;
    }
    // Convert +1000 throw/catch IDs to raw TYPE_REGISTRY keys
    let actual_raw = (actual_type_id - 1000) as i64;
    let expected_raw = (expected_type_id - 1000) as i64;
    if crate::type_system::type_id_matches_with_hierarchy(actual_raw, expected_raw) {
        1
    } else {
        0
    }
}
