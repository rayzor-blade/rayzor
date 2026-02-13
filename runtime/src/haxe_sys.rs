//! Haxe Sys runtime implementation
//!
//! System and I/O functions

use log::debug;
use std::cell::RefCell;
use std::io::{self, Write};

// Use the canonical HaxeString definition from haxe_string module
use crate::haxe_string::HaxeString;

// Thread-local trace prefix for identifying which backend owns the output
thread_local! {
    static TRACE_PREFIX: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Set the trace prefix for the current thread (e.g., "[rayzor-tiered] ")
#[no_mangle]
pub extern "C" fn rayzor_set_trace_prefix(ptr: *const u8, len: usize) {
    if ptr.is_null() || len == 0 {
        TRACE_PREFIX.with(|p| p.borrow_mut().clear());
        return;
    }
    unsafe {
        let slice = std::slice::from_raw_parts(ptr, len);
        if let Ok(s) = std::str::from_utf8(slice) {
            TRACE_PREFIX.with(|p| *p.borrow_mut() = s.to_string());
        }
    }
}

/// Set trace prefix from a Rust string (convenience for Rust callers)
pub fn set_trace_prefix(prefix: &str) {
    TRACE_PREFIX.with(|p| *p.borrow_mut() = prefix.to_string());
}

fn print_with_prefix(msg: &str) {
    TRACE_PREFIX.with(|p| {
        let prefix = p.borrow();
        if prefix.is_empty() {
            println!("{}", msg);
        } else {
            println!("{}{}", *prefix, msg);
        }
    });
}

// ============================================================================
// Console I/O
// ============================================================================

/// Print integer to stdout
#[no_mangle]
pub extern "C" fn haxe_sys_print_int(value: i64) {
    print!("{}", value);
    let _ = io::stdout().flush();
}

/// Print float to stdout
#[no_mangle]
pub extern "C" fn haxe_sys_print_float(value: f64) {
    print!("{}", value);
    let _ = io::stdout().flush();
}

/// Print boolean to stdout
#[no_mangle]
pub extern "C" fn haxe_sys_print_bool(value: bool) {
    print!("{}", value);
    let _ = io::stdout().flush();
}

/// Print newline
#[no_mangle]
pub extern "C" fn haxe_sys_println() {
    println!();
}

// ============================================================================
// Trace Functions (Runtime Logging)
// ============================================================================

/// Trace integer value
#[no_mangle]
pub extern "C" fn haxe_trace_int(value: i64) {
    print_with_prefix(&format!("{}", value));
}

/// Trace float value
#[no_mangle]
pub extern "C" fn haxe_trace_float(value: f64) {
    print_with_prefix(&format!("{}", value));
}

/// Trace boolean value
#[no_mangle]
pub extern "C" fn haxe_trace_bool(value: bool) {
    print_with_prefix(&format!("{}", value));
}

/// Trace string value (ptr + len)
#[no_mangle]
pub extern "C" fn haxe_trace_string(ptr: *const u8, len: usize) {
    if ptr.is_null() {
        print_with_prefix("null");
        return;
    }

    unsafe {
        let slice = std::slice::from_raw_parts(ptr, len);
        match std::str::from_utf8(slice) {
            Ok(s) => print_with_prefix(s),
            Err(_) => print_with_prefix("<invalid utf8>"),
        }
    }
}

/// Trace string value (takes pointer to HaxeString struct)
#[no_mangle]
pub extern "C" fn haxe_trace_string_struct(s_ptr: *const HaxeString) {
    if s_ptr.is_null() {
        println!("null");
        return;
    }
    unsafe {
        let s = &*s_ptr;
        haxe_trace_string(s.ptr, s.len);
    }
}

/// Trace any Dynamic value using Std.string() for proper type dispatch
/// The value is expected to be a pointer to a DynamicValue (boxed Dynamic)
#[no_mangle]
pub extern "C" fn haxe_trace_any(dynamic_ptr: *mut u8) {
    if dynamic_ptr.is_null() {
        print_with_prefix("null");
        return;
    }

    unsafe {
        // Call haxe_std_string_ptr to convert Dynamic to HaxeString
        let string_ptr = crate::type_system::haxe_std_string_ptr(dynamic_ptr);

        if !string_ptr.is_null() {
            let haxe_str = &*string_ptr;
            if !haxe_str.ptr.is_null() && haxe_str.len > 0 {
                let slice = std::slice::from_raw_parts(haxe_str.ptr, haxe_str.len);
                if let Ok(s) = std::str::from_utf8(slice) {
                    print_with_prefix(s);
                    return;
                }
            }
        }
        // Fallback
        print_with_prefix(&format!("<Dynamic@{:p}>", dynamic_ptr));
    }
}

/// Trace an Array value — prints elements as [e0, e1, e2, ...]
/// The value is expected to be a pointer to a HaxeArray struct.
/// Elements are printed as integers (i64) since arrays store i64 values.
#[no_mangle]
pub extern "C" fn haxe_trace_array(arr_ptr: *mut u8) {
    if arr_ptr.is_null() {
        print_with_prefix("null");
        return;
    }

    unsafe {
        let arr = &*(arr_ptr as *const crate::haxe_array::HaxeArray);
        let mut result = String::from("[");
        for i in 0..arr.len {
            if i > 0 {
                result.push_str(", ");
            }
            if arr.elem_size == 8 {
                let val = *(arr.ptr.add(i * 8) as *const i64);
                result.push_str(&val.to_string());
            } else if arr.elem_size == 4 {
                let val = *(arr.ptr.add(i * 4) as *const i32);
                result.push_str(&val.to_string());
            } else {
                result.push('?');
            }
        }
        result.push(']');
        print_with_prefix(&result);
    }
}

// ============================================================================
// Std.string() - Type-specific string conversions
// All functions return *mut HaxeString to avoid struct return ABI issues
// ============================================================================

/// Convert Int to String - returns heap-allocated HaxeString pointer
#[no_mangle]
pub extern "C" fn haxe_string_from_int(value: i64) -> *mut HaxeString {
    let s = value.to_string();
    let bytes = s.into_bytes();
    let len = bytes.len();
    let cap = bytes.capacity();
    let ptr = bytes.as_ptr() as *mut u8;
    std::mem::forget(bytes); // Transfer ownership to HaxeString

    Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
}

/// Convert Float to String - returns heap-allocated HaxeString pointer
#[no_mangle]
pub extern "C" fn haxe_string_from_float(value: f64) -> *mut HaxeString {
    let s = value.to_string();
    let bytes = s.into_bytes();
    let len = bytes.len();
    let cap = bytes.capacity();
    let ptr = bytes.as_ptr() as *mut u8;
    std::mem::forget(bytes);

    Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
}

/// Convert Bool to String - returns heap-allocated HaxeString pointer
#[no_mangle]
pub extern "C" fn haxe_string_from_bool(value: bool) -> *mut HaxeString {
    let s = if value { "true" } else { "false" };
    // For static strings, use the static pointer with cap=0 to indicate no-free
    Box::into_raw(Box::new(HaxeString {
        ptr: s.as_ptr() as *mut u8,
        len: s.len(),
        cap: 0, // cap=0 means static string, don't free
    }))
}

/// Convert String to String (identity, but normalizes representation)
#[no_mangle]
pub extern "C" fn haxe_string_from_string(ptr: *const u8, len: usize) -> *mut HaxeString {
    // Create a copy of the string data
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let vec = slice.to_vec();
    let cap = vec.capacity();
    let new_ptr = vec.as_ptr() as *mut u8;
    std::mem::forget(vec);

    Box::into_raw(Box::new(HaxeString {
        ptr: new_ptr,
        len,
        cap,
    }))
}

/// Convert null to String - returns heap-allocated HaxeString pointer
#[no_mangle]
pub extern "C" fn haxe_string_from_null() -> *mut HaxeString {
    let s = "null";
    Box::into_raw(Box::new(HaxeString {
        ptr: s.as_ptr() as *mut u8,
        len: s.len(),
        cap: 0, // static string
    }))
}

/// Create a string literal from embedded bytes
/// Returns a pointer to a heap-allocated HaxeString struct
/// The bytes are NOT copied - they must remain valid (e.g., in JIT code section)
#[no_mangle]
pub extern "C" fn haxe_string_literal(ptr: *const u8, len: usize) -> *mut HaxeString {
    Box::into_raw(Box::new(HaxeString {
        ptr: ptr as *mut u8,
        len,
        cap: 0, // cap=0 means static/borrowed, don't free the data
    }))
}

/// Convert string to uppercase (wrapper returning pointer)
/// Takes pointer to input string, returns pointer to new heap-allocated uppercase string
#[no_mangle]
pub extern "C" fn haxe_string_upper(s: *const HaxeString) -> *mut HaxeString {
    if s.is_null() {
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }
    unsafe {
        let s_ref = &*s;
        if s_ref.ptr.is_null() || s_ref.len == 0 {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }
        let slice = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);
        if let Ok(rust_str) = std::str::from_utf8(slice) {
            let upper = rust_str.to_uppercase();
            let bytes = upper.into_bytes();
            let len = bytes.len();
            let cap = bytes.capacity();
            let ptr = bytes.as_ptr() as *mut u8;
            std::mem::forget(bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        } else {
            // Invalid UTF-8, return copy of original
            let new_bytes = slice.to_vec();
            let len = new_bytes.len();
            let cap = new_bytes.capacity();
            let ptr = new_bytes.as_ptr() as *mut u8;
            std::mem::forget(new_bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        }
    }
}

/// Convert string to lowercase (wrapper returning pointer)
/// Takes pointer to input string, returns pointer to new heap-allocated lowercase string
#[no_mangle]
pub extern "C" fn haxe_string_lower(s: *const HaxeString) -> *mut HaxeString {
    if s.is_null() {
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }
    unsafe {
        let s_ref = &*s;
        if s_ref.ptr.is_null() || s_ref.len == 0 {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }
        let slice = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);
        if let Ok(rust_str) = std::str::from_utf8(slice) {
            let lower = rust_str.to_lowercase();
            let bytes = lower.into_bytes();
            let len = bytes.len();
            let cap = bytes.capacity();
            let ptr = bytes.as_ptr() as *mut u8;
            std::mem::forget(bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        } else {
            // Invalid UTF-8, return copy of original
            let new_bytes = slice.to_vec();
            let len = new_bytes.len();
            let cap = new_bytes.capacity();
            let ptr = new_bytes.as_ptr() as *mut u8;
            std::mem::forget(new_bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        }
    }
}

// ============================================================================
// String Instance Methods (working with *const HaxeString)
// ============================================================================

/// Get string length
#[no_mangle]
pub extern "C" fn haxe_string_len(s: *const HaxeString) -> i32 {
    if s.is_null() {
        return 0;
    }
    unsafe { (*s).len as i32 }
}

/// Get character at index - returns empty string if out of bounds
/// Note: charAt returns String, not Int, per Haxe specification
#[no_mangle]
pub extern "C" fn haxe_string_char_at_ptr(s: *const HaxeString, index: i64) -> *mut HaxeString {
    if s.is_null() {
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }
    unsafe {
        let s_ref = &*s;
        if index < 0 || (index as usize) >= s_ref.len || s_ref.ptr.is_null() {
            // Return empty string for out of bounds
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        // Get the byte at the index
        let byte = *s_ref.ptr.add(index as usize);
        let bytes = vec![byte];
        let len = bytes.len();
        let cap = bytes.capacity();
        let ptr = bytes.as_ptr() as *mut u8;
        std::mem::forget(bytes);
        Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
    }
}

/// Get character code at index - returns -1 (represented as null Int) if out of bounds
#[no_mangle]
pub extern "C" fn haxe_string_char_code_at_ptr(s: *const HaxeString, index: i64) -> i64 {
    if s.is_null() {
        return -1; // null
    }
    unsafe {
        let s_ref = &*s;
        if index < 0 || (index as usize) >= s_ref.len || s_ref.ptr.is_null() {
            return -1; // null
        }
        *s_ref.ptr.add(index as usize) as i64
    }
}

/// Find index of substring, starting from startIndex
/// Returns -1 if not found
#[no_mangle]
pub extern "C" fn haxe_string_index_of_ptr(
    s: *const HaxeString,
    needle: *const HaxeString,
    start_index: i32,
) -> i32 {
    if s.is_null() || needle.is_null() {
        return -1;
    }
    unsafe {
        let s_ref = &*s;
        let needle_ref = &*needle;

        if s_ref.ptr.is_null() || needle_ref.ptr.is_null() {
            return -1;
        }

        // Empty needle - return start_index (or 0 if start_index < 0)
        if needle_ref.len == 0 {
            return if start_index < 0 { 0 } else { start_index };
        }

        let start = if start_index < 0 {
            0
        } else {
            start_index as usize
        };
        if start >= s_ref.len {
            return -1;
        }

        let haystack = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);
        let needle_bytes = std::slice::from_raw_parts(needle_ref.ptr, needle_ref.len);

        // Simple substring search
        for i in start..=(s_ref.len.saturating_sub(needle_ref.len)) {
            if &haystack[i..i + needle_ref.len] == needle_bytes {
                return i as i32;
            }
        }
        -1
    }
}

/// Find last index of substring, searching backwards from startIndex
/// Returns -1 if not found
#[no_mangle]
pub extern "C" fn haxe_string_last_index_of_ptr(
    s: *const HaxeString,
    needle: *const HaxeString,
    start_index: i32,
) -> i32 {
    if s.is_null() || needle.is_null() {
        return -1;
    }
    unsafe {
        let s_ref = &*s;
        let needle_ref = &*needle;

        if s_ref.ptr.is_null() || needle_ref.ptr.is_null() {
            return -1;
        }

        // Empty needle - return end of string (or start_index if provided and smaller)
        if needle_ref.len == 0 {
            let len = s_ref.len as i32;
            return if start_index < 0 || start_index >= len {
                len
            } else {
                start_index
            };
        }

        if needle_ref.len > s_ref.len {
            return -1;
        }

        let haystack = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);
        let needle_bytes = std::slice::from_raw_parts(needle_ref.ptr, needle_ref.len);

        // Calculate the maximum starting position
        let max_start = s_ref.len - needle_ref.len;
        let search_start = if start_index < 0 {
            max_start
        } else {
            (start_index as usize).min(max_start)
        };

        // Search backwards
        for i in (0..=search_start).rev() {
            if &haystack[i..i + needle_ref.len] == needle_bytes {
                return i as i32;
            }
        }
        -1
    }
}

/// Get substring using substr semantics (pos, len)
/// If len is negative, returns empty string
/// If pos is negative, calculated from end
#[no_mangle]
pub extern "C" fn haxe_string_substr_ptr(
    s: *const HaxeString,
    pos: i32,
    len: i32,
) -> *mut HaxeString {
    if s.is_null() {
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }
    unsafe {
        let s_ref = &*s;
        if s_ref.ptr.is_null() || s_ref.len == 0 || len < 0 {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        // Handle negative pos (from end)
        let actual_pos = if pos < 0 {
            let from_end = (-pos) as usize;
            s_ref.len.saturating_sub(from_end)
        } else {
            pos as usize
        };

        if actual_pos >= s_ref.len {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        let available = s_ref.len - actual_pos;
        let actual_len = (len as usize).min(available);

        if actual_len == 0 {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        let slice = std::slice::from_raw_parts(s_ref.ptr.add(actual_pos), actual_len);
        let bytes = slice.to_vec();
        let new_len = bytes.len();
        let cap = bytes.capacity();
        let ptr = bytes.as_ptr() as *mut u8;
        std::mem::forget(bytes);
        Box::into_raw(Box::new(HaxeString {
            ptr,
            len: new_len,
            cap,
        }))
    }
}

/// Get substring using substring semantics (startIndex, endIndex)
/// Negative indices become 0
/// If startIndex > endIndex, they are swapped
#[no_mangle]
pub extern "C" fn haxe_string_substring_ptr(
    s: *const HaxeString,
    start_index: i32,
    end_index: i32,
) -> *mut HaxeString {
    if s.is_null() {
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }
    unsafe {
        let s_ref = &*s;
        if s_ref.ptr.is_null() || s_ref.len == 0 {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        // Clamp negative values to 0
        let mut start = if start_index < 0 {
            0
        } else {
            start_index as usize
        };
        let mut end = if end_index < 0 { 0 } else { end_index as usize };

        // Clamp to string length
        start = start.min(s_ref.len);
        end = end.min(s_ref.len);

        // Swap if start > end
        if start > end {
            std::mem::swap(&mut start, &mut end);
        }

        if start == end {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        let slice = std::slice::from_raw_parts(s_ref.ptr.add(start), end - start);
        let bytes = slice.to_vec();
        let new_len = bytes.len();
        let cap = bytes.capacity();
        let ptr = bytes.as_ptr() as *mut u8;
        std::mem::forget(bytes);
        Box::into_raw(Box::new(HaxeString {
            ptr,
            len: new_len,
            cap,
        }))
    }
}

/// Create string from character code (static method)
#[no_mangle]
pub extern "C" fn haxe_string_from_char_code(code: i32) -> *mut HaxeString {
    if !(0..=0x10FFFF).contains(&code) {
        // Invalid code point, return empty string
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }

    // Convert to char and encode as UTF-8
    if let Some(c) = char::from_u32(code as u32) {
        let mut buf = [0u8; 4];
        let encoded = c.encode_utf8(&mut buf);
        let bytes = encoded.as_bytes().to_vec();
        let len = bytes.len();
        let cap = bytes.capacity();
        let ptr = bytes.as_ptr() as *mut u8;
        std::mem::forget(bytes);
        Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
    } else {
        Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }))
    }
}

/// Copy string (for toString() method)
#[no_mangle]
pub extern "C" fn haxe_string_copy(s: *const HaxeString) -> *mut HaxeString {
    if s.is_null() {
        return Box::into_raw(Box::new(HaxeString {
            ptr: std::ptr::null_mut(),
            len: 0,
            cap: 0,
        }));
    }
    unsafe {
        let s_ref = &*s;
        if s_ref.ptr.is_null() || s_ref.len == 0 {
            return Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
        }

        let slice = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);
        let bytes = slice.to_vec();
        let len = bytes.len();
        let cap = bytes.capacity();
        let ptr = bytes.as_ptr() as *mut u8;
        std::mem::forget(bytes);
        Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
    }
}

/// Split string by delimiter - returns array pointer and sets length
/// Note: Caller is responsible for freeing the returned array and strings
#[no_mangle]
pub extern "C" fn haxe_string_split_ptr(
    s: *const HaxeString,
    delimiter: *const HaxeString,
    out_len: *mut i64,
) -> *mut *mut HaxeString {
    unsafe {
        if out_len.is_null() {
            return std::ptr::null_mut();
        }

        if s.is_null() {
            *out_len = 0;
            return std::ptr::null_mut();
        }

        let s_ref = &*s;

        // Handle null or empty string
        if s_ref.ptr.is_null() || s_ref.len == 0 {
            // Return array with one empty string
            let empty = Box::into_raw(Box::new(HaxeString {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
            }));
            let result = Box::into_raw(vec![empty].into_boxed_slice()) as *mut *mut HaxeString;
            *out_len = 1;
            return result;
        }

        let haystack = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);

        // Handle null delimiter - return array with original string
        if delimiter.is_null() {
            let copy = haxe_string_copy(s);
            let result = Box::into_raw(vec![copy].into_boxed_slice()) as *mut *mut HaxeString;
            *out_len = 1;
            return result;
        }

        let delim_ref = &*delimiter;

        // Empty delimiter - split into individual characters
        if delim_ref.ptr.is_null() || delim_ref.len == 0 {
            let mut parts: Vec<*mut HaxeString> = Vec::with_capacity(s_ref.len);
            for i in 0..s_ref.len {
                let byte = *s_ref.ptr.add(i);
                let bytes = vec![byte];
                let cap = bytes.capacity();
                let ptr = bytes.as_ptr() as *mut u8;
                std::mem::forget(bytes);
                parts.push(Box::into_raw(Box::new(HaxeString { ptr, len: 1, cap })));
            }
            *out_len = parts.len() as i64;
            Box::into_raw(parts.into_boxed_slice()) as *mut *mut HaxeString
        } else {
            let delim_bytes = std::slice::from_raw_parts(delim_ref.ptr, delim_ref.len);

            let mut parts: Vec<*mut HaxeString> = Vec::new();
            let mut start = 0;

            while start <= s_ref.len {
                // Find next occurrence of delimiter
                let mut found_at = None;
                for i in start..=(s_ref.len.saturating_sub(delim_ref.len)) {
                    if &haystack[i..i + delim_ref.len] == delim_bytes {
                        found_at = Some(i);
                        break;
                    }
                }

                match found_at {
                    Some(idx) => {
                        // Add substring before delimiter
                        let part_len = idx - start;
                        if part_len == 0 {
                            parts.push(Box::into_raw(Box::new(HaxeString {
                                ptr: std::ptr::null_mut(),
                                len: 0,
                                cap: 0,
                            })));
                        } else {
                            let bytes = haystack[start..idx].to_vec();
                            let len = bytes.len();
                            let cap = bytes.capacity();
                            let ptr = bytes.as_ptr() as *mut u8;
                            std::mem::forget(bytes);
                            parts.push(Box::into_raw(Box::new(HaxeString { ptr, len, cap })));
                        }
                        start = idx + delim_ref.len;
                    }
                    None => {
                        // Add remaining string
                        let part_len = s_ref.len - start;
                        if part_len == 0 {
                            parts.push(Box::into_raw(Box::new(HaxeString {
                                ptr: std::ptr::null_mut(),
                                len: 0,
                                cap: 0,
                            })));
                        } else {
                            let bytes = haystack[start..].to_vec();
                            let len = bytes.len();
                            let cap = bytes.capacity();
                            let ptr = bytes.as_ptr() as *mut u8;
                            std::mem::forget(bytes);
                            parts.push(Box::into_raw(Box::new(HaxeString { ptr, len, cap })));
                        }
                        break;
                    }
                }
            }

            *out_len = parts.len() as i64;
            Box::into_raw(parts.into_boxed_slice()) as *mut *mut HaxeString
        }
    }
}

// ============================================================================
// Program Control
// ============================================================================

/// Exit program with code
#[no_mangle]
pub extern "C" fn haxe_sys_exit(code: i32) -> ! {
    std::process::exit(code)
}

/// Get current time in milliseconds
#[no_mangle]
pub extern "C" fn haxe_sys_time() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Get command line arguments count
#[no_mangle]
pub extern "C" fn haxe_sys_args_count() -> i32 {
    std::env::args().count() as i32
}

// ============================================================================
// Environment Variables
// ============================================================================

/// Get environment variable value
/// Returns null if the variable doesn't exist
#[no_mangle]
pub extern "C" fn haxe_sys_get_env(name: *const HaxeString) -> *mut HaxeString {
    if name.is_null() {
        return std::ptr::null_mut();
    }

    unsafe {
        let name_ref = &*name;
        if name_ref.ptr.is_null() || name_ref.len == 0 {
            return std::ptr::null_mut();
        }

        let slice = std::slice::from_raw_parts(name_ref.ptr, name_ref.len);
        let var_name = match std::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        match std::env::var(var_name) {
            Ok(value) => {
                let bytes = value.into_bytes();
                let len = bytes.len();
                let cap = bytes.capacity();
                let ptr = bytes.as_ptr() as *mut u8;
                std::mem::forget(bytes);
                Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
            }
            Err(_) => std::ptr::null_mut(), // Variable not found
        }
    }
}

/// Set environment variable value
/// If value is null, removes the environment variable
#[no_mangle]
pub extern "C" fn haxe_sys_put_env(name: *const HaxeString, value: *const HaxeString) {
    if name.is_null() {
        return;
    }

    unsafe {
        let name_ref = &*name;
        if name_ref.ptr.is_null() || name_ref.len == 0 {
            return;
        }

        let slice = std::slice::from_raw_parts(name_ref.ptr, name_ref.len);
        let var_name = match std::str::from_utf8(slice) {
            Ok(s) => s,
            Err(_) => return,
        };

        if value.is_null() {
            // Remove the environment variable
            std::env::remove_var(var_name);
        } else {
            let value_ref = &*value;
            if value_ref.ptr.is_null() {
                std::env::remove_var(var_name);
            } else {
                let value_slice = std::slice::from_raw_parts(value_ref.ptr, value_ref.len);
                if let Ok(val_str) = std::str::from_utf8(value_slice) {
                    std::env::set_var(var_name, val_str);
                }
            }
        }
    }
}

// ============================================================================
// Working Directory
// ============================================================================

/// Get current working directory
#[no_mangle]
pub extern "C" fn haxe_sys_get_cwd() -> *mut HaxeString {
    match std::env::current_dir() {
        Ok(path) => {
            let path_str = path.to_string_lossy().into_owned();
            let bytes = path_str.into_bytes();
            let len = bytes.len();
            let cap = bytes.capacity();
            let ptr = bytes.as_ptr() as *mut u8;
            std::mem::forget(bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Set current working directory
#[no_mangle]
pub extern "C" fn haxe_sys_set_cwd(path: *const HaxeString) {
    if path.is_null() {
        return;
    }

    unsafe {
        let path_ref = &*path;
        if path_ref.ptr.is_null() || path_ref.len == 0 {
            return;
        }

        let slice = std::slice::from_raw_parts(path_ref.ptr, path_ref.len);
        if let Ok(path_str) = std::str::from_utf8(slice) {
            let _ = std::env::set_current_dir(path_str);
        }
    }
}

// ============================================================================
// Sleep
// ============================================================================

/// Sleep for the specified number of seconds
#[no_mangle]
pub extern "C" fn haxe_sys_sleep(seconds: f64) {
    if seconds <= 0.0 {
        return;
    }

    let duration = std::time::Duration::from_secs_f64(seconds);
    std::thread::sleep(duration);
}

// ============================================================================
// System Information
// ============================================================================

/// Get the system/OS name
/// Returns "Windows", "Linux", "Mac", or "BSD"
#[no_mangle]
pub extern "C" fn haxe_sys_system_name() -> *mut HaxeString {
    let name = if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "macos") {
        "Mac"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "freebsd")
        || cfg!(target_os = "openbsd")
        || cfg!(target_os = "netbsd")
    {
        "BSD"
    } else {
        "Unknown"
    };

    // Return a static string (cap=0 means no-free)
    Box::into_raw(Box::new(HaxeString {
        ptr: name.as_ptr() as *mut u8,
        len: name.len(),
        cap: 0,
    }))
}

/// Get CPU time for current process (in seconds)
#[no_mangle]
pub extern "C" fn haxe_sys_cpu_time() -> f64 {
    // This is a simplified implementation - full accuracy would require platform-specific code
    // On Unix, we could use getrusage() for accurate CPU time
    // On Windows, we could use GetProcessTimes()
    // For portability, we use a static start time and return elapsed time
    static START_TIME: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START_TIME.get_or_init(std::time::Instant::now);
    start.elapsed().as_secs_f64()
}

/// Get path to current executable
#[no_mangle]
pub extern "C" fn haxe_sys_program_path() -> *mut HaxeString {
    match std::env::current_exe() {
        Ok(path) => {
            let path_str = path.to_string_lossy().into_owned();
            let bytes = path_str.into_bytes();
            let len = bytes.len();
            let cap = bytes.capacity();
            let ptr = bytes.as_ptr() as *mut u8;
            std::mem::forget(bytes);
            Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Execute a shell command and return the exit code
/// Sys.command(cmd: String, args: Array<String>): Int
/// When args is null, cmd is passed directly to the shell
#[no_mangle]
pub extern "C" fn haxe_sys_command(cmd: *const HaxeString) -> i32 {
    unsafe {
        let cmd_str = match haxe_string_to_rust(cmd) {
            Some(s) => s,
            None => return -1,
        };

        // Execute command via shell
        #[cfg(target_os = "windows")]
        let output = std::process::Command::new("cmd")
            .args(["/C", &cmd_str])
            .status();

        #[cfg(not(target_os = "windows"))]
        let output = std::process::Command::new("sh")
            .args(["-c", &cmd_str])
            .status();

        match output {
            Ok(status) => status.code().unwrap_or(-1),
            Err(_) => -1,
        }
    }
}

/// Read a single character from stdin
/// Sys.getChar(echo: Bool): Int
#[no_mangle]
pub extern "C" fn haxe_sys_get_char(echo: bool) -> i32 {
    use std::io::Read;

    let mut buffer = [0u8; 1];
    match std::io::stdin().read_exact(&mut buffer) {
        Ok(_) => {
            if echo {
                print!("{}", buffer[0] as char);
            }
            buffer[0] as i32
        }
        Err(_) => -1,
    }
}

// ============================================================================
// File I/O (sys.io.File)
// ============================================================================

/// Helper to convert HaxeString pointer to Rust String
unsafe fn haxe_string_to_rust(s: *const HaxeString) -> Option<String> {
    if s.is_null() {
        return None;
    }
    let s_ref = &*s;
    if s_ref.ptr.is_null() || s_ref.len == 0 {
        return Some(String::new());
    }
    let slice = std::slice::from_raw_parts(s_ref.ptr, s_ref.len);
    std::str::from_utf8(slice).ok().map(|s| s.to_string())
}

/// Helper to create HaxeString from Rust String
fn rust_string_to_haxe(s: String) -> *mut HaxeString {
    let bytes = s.into_bytes();
    let len = bytes.len();
    let cap = bytes.capacity();
    let ptr = bytes.as_ptr() as *mut u8;
    std::mem::forget(bytes);
    Box::into_raw(Box::new(HaxeString { ptr, len, cap }))
}

/// Read entire file content as string
/// File.getContent(path: String): String
#[no_mangle]
pub extern "C" fn haxe_file_get_content(path: *const HaxeString) -> *mut HaxeString {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => match std::fs::read_to_string(&path_str) {
                Ok(content) => rust_string_to_haxe(content),
                Err(e) => {
                    debug!("File.getContent error: {} - {}", path_str, e);
                    std::ptr::null_mut()
                }
            },
            None => std::ptr::null_mut(),
        }
    }
}

/// Write string content to file
/// File.saveContent(path: String, content: String): Void
#[no_mangle]
pub extern "C" fn haxe_file_save_content(path: *const HaxeString, content: *const HaxeString) {
    unsafe {
        let path_str = match haxe_string_to_rust(path) {
            Some(s) => s,
            None => return,
        };
        let content_str = haxe_string_to_rust(content).unwrap_or_default();
        if let Err(e) = std::fs::write(&path_str, content_str) {
            debug!("File.saveContent error: {} - {}", path_str, e);
        }
    }
}

/// Copy file from src to dst
/// File.copy(srcPath: String, dstPath: String): Void
#[no_mangle]
pub extern "C" fn haxe_file_copy(src: *const HaxeString, dst: *const HaxeString) {
    unsafe {
        let src_str = match haxe_string_to_rust(src) {
            Some(s) => s,
            None => return,
        };
        let dst_str = match haxe_string_to_rust(dst) {
            Some(s) => s,
            None => return,
        };
        if let Err(e) = std::fs::copy(&src_str, &dst_str) {
            debug!("File.copy error: {} -> {} - {}", src_str, dst_str, e);
        }
    }
}

/// Read entire file content as binary bytes
/// File.getBytes(path: String): haxe.io.Bytes
#[no_mangle]
pub extern "C" fn haxe_file_get_bytes(path: *const HaxeString) -> *mut HaxeBytes {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => {
                match std::fs::read(&path_str) {
                    Ok(content) => {
                        // Create HaxeBytes from the file content
                        let len = content.len();
                        let cap = content.capacity();
                        let ptr = content.as_ptr() as *mut u8;
                        std::mem::forget(content); // Don't drop - HaxeBytes now owns the memory

                        let bytes = Box::new(HaxeBytes { ptr, len, cap });
                        Box::into_raw(bytes)
                    }
                    Err(e) => {
                        debug!("File.getBytes error: {} - {}", path_str, e);
                        std::ptr::null_mut()
                    }
                }
            }
            None => std::ptr::null_mut(),
        }
    }
}

/// Write binary bytes to file
/// File.saveBytes(path: String, bytes: haxe.io.Bytes): Void
#[no_mangle]
pub extern "C" fn haxe_file_save_bytes(path: *const HaxeString, bytes: *const HaxeBytes) {
    unsafe {
        let path_str = match haxe_string_to_rust(path) {
            Some(s) => s,
            None => return,
        };

        if bytes.is_null() {
            debug!("File.saveBytes error: bytes is null");
            return;
        }

        let b = &*bytes;
        let slice = std::slice::from_raw_parts(b.ptr, b.len);

        if let Err(e) = std::fs::write(&path_str, slice) {
            debug!("File.saveBytes error: {} - {}", path_str, e);
        }
    }
}

// ============================================================================
// FileSystem (sys.FileSystem)
// ============================================================================

/// Check if file or directory exists
/// FileSystem.exists(path: String): Bool
#[no_mangle]
pub extern "C" fn haxe_filesystem_exists(path: *const HaxeString) -> bool {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => std::path::Path::new(&path_str).exists(),
            None => false,
        }
    }
}

/// Check if path is a directory
/// FileSystem.isDirectory(path: String): Bool
#[no_mangle]
pub extern "C" fn haxe_filesystem_is_directory(path: *const HaxeString) -> bool {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => std::path::Path::new(&path_str).is_dir(),
            None => false,
        }
    }
}

/// Create directory (recursively)
/// FileSystem.createDirectory(path: String): Void
#[no_mangle]
pub extern "C" fn haxe_filesystem_create_directory(path: *const HaxeString) {
    unsafe {
        if let Some(path_str) = haxe_string_to_rust(path) {
            if let Err(e) = std::fs::create_dir_all(&path_str) {
                debug!("FileSystem.createDirectory error: {} - {}", path_str, e);
            }
        }
    }
}

/// Delete file
/// FileSystem.deleteFile(path: String): Void
#[no_mangle]
pub extern "C" fn haxe_filesystem_delete_file(path: *const HaxeString) {
    unsafe {
        if let Some(path_str) = haxe_string_to_rust(path) {
            if let Err(e) = std::fs::remove_file(&path_str) {
                debug!("FileSystem.deleteFile error: {} - {}", path_str, e);
            }
        }
    }
}

/// Delete directory (must be empty)
/// FileSystem.deleteDirectory(path: String): Void
#[no_mangle]
pub extern "C" fn haxe_filesystem_delete_directory(path: *const HaxeString) {
    unsafe {
        if let Some(path_str) = haxe_string_to_rust(path) {
            if let Err(e) = std::fs::remove_dir(&path_str) {
                debug!("FileSystem.deleteDirectory error: {} - {}", path_str, e);
            }
        }
    }
}

/// Rename/move file or directory
/// FileSystem.rename(path: String, newPath: String): Void
#[no_mangle]
pub extern "C" fn haxe_filesystem_rename(path: *const HaxeString, new_path: *const HaxeString) {
    unsafe {
        let path_str = match haxe_string_to_rust(path) {
            Some(s) => s,
            None => return,
        };
        let new_path_str = match haxe_string_to_rust(new_path) {
            Some(s) => s,
            None => return,
        };
        if let Err(e) = std::fs::rename(&path_str, &new_path_str) {
            debug!(
                "FileSystem.rename error: {} -> {} - {}",
                path_str, new_path_str, e
            );
        }
    }
}

/// Get full/absolute path
/// FileSystem.fullPath(relPath: String): String
#[no_mangle]
pub extern "C" fn haxe_filesystem_full_path(path: *const HaxeString) -> *mut HaxeString {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => match std::fs::canonicalize(&path_str) {
                Ok(full_path) => rust_string_to_haxe(full_path.to_string_lossy().into_owned()),
                Err(_) => std::ptr::null_mut(),
            },
            None => std::ptr::null_mut(),
        }
    }
}

/// Get absolute path (doesn't need to exist)
/// FileSystem.absolutePath(relPath: String): String
#[no_mangle]
pub extern "C" fn haxe_filesystem_absolute_path(path: *const HaxeString) -> *mut HaxeString {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => {
                let abs_path = if std::path::Path::new(&path_str).is_absolute() {
                    path_str
                } else {
                    match std::env::current_dir() {
                        Ok(cwd) => cwd.join(&path_str).to_string_lossy().into_owned(),
                        Err(_) => path_str,
                    }
                };
                rust_string_to_haxe(abs_path)
            }
            None => std::ptr::null_mut(),
        }
    }
}

/// FileStat struct - matches Haxe's sys.FileStat typedef
/// All fields are 8 bytes for consistent sizing/boxing
/// Date fields stored as f64 timestamps (seconds since Unix epoch)
#[repr(C)]
pub struct HaxeFileStat {
    pub gid: i64,   // group id
    pub uid: i64,   // user id
    pub atime: f64, // access time (seconds since epoch)
    pub mtime: f64, // modification time (seconds since epoch)
    pub ctime: f64, // creation/change time (seconds since epoch)
    pub size: i64,  // file size in bytes
    pub dev: i64,   // device id
    pub ino: i64,   // inode number
    pub nlink: i64, // number of hard links
    pub rdev: i64,  // device type (special files)
    pub mode: i64,  // permission bits
}

/// Get file/directory statistics
/// FileSystem.stat(path: String): FileStat
#[no_mangle]
pub extern "C" fn haxe_filesystem_stat(path: *const HaxeString) -> *mut HaxeFileStat {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => {
                match std::fs::metadata(&path_str) {
                    Ok(meta) => {
                        // Convert SystemTime to f64 (seconds since Unix epoch)
                        let to_timestamp = |time: std::io::Result<std::time::SystemTime>| -> f64 {
                            time.ok()
                                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                                .map(|d| d.as_secs_f64())
                                .unwrap_or(0.0)
                        };

                        let stat = Box::new(HaxeFileStat {
                            #[cfg(unix)]
                            gid: {
                                use std::os::unix::fs::MetadataExt;
                                meta.gid() as i64
                            },
                            #[cfg(not(unix))]
                            gid: 0,

                            #[cfg(unix)]
                            uid: {
                                use std::os::unix::fs::MetadataExt;
                                meta.uid() as i64
                            },
                            #[cfg(not(unix))]
                            uid: 0,

                            atime: to_timestamp(meta.accessed()),
                            mtime: to_timestamp(meta.modified()),
                            ctime: to_timestamp(meta.created()),
                            size: meta.len() as i64,

                            #[cfg(unix)]
                            dev: {
                                use std::os::unix::fs::MetadataExt;
                                meta.dev() as i64
                            },
                            #[cfg(not(unix))]
                            dev: 0,

                            #[cfg(unix)]
                            ino: {
                                use std::os::unix::fs::MetadataExt;
                                meta.ino() as i64
                            },
                            #[cfg(not(unix))]
                            ino: 0,

                            #[cfg(unix)]
                            nlink: {
                                use std::os::unix::fs::MetadataExt;
                                meta.nlink() as i64
                            },
                            #[cfg(not(unix))]
                            nlink: 1,

                            #[cfg(unix)]
                            rdev: {
                                use std::os::unix::fs::MetadataExt;
                                meta.rdev() as i64
                            },
                            #[cfg(not(unix))]
                            rdev: 0,

                            #[cfg(unix)]
                            mode: {
                                use std::os::unix::fs::MetadataExt;
                                meta.mode() as i64
                            },
                            #[cfg(not(unix))]
                            mode: if meta.is_dir() { 0o755 } else { 0o644 } as i64,
                        });
                        Box::into_raw(stat)
                    }
                    Err(_) => std::ptr::null_mut(),
                }
            }
            None => std::ptr::null_mut(),
        }
    }
}

/// Check if path is a file (not directory)
/// FileSystem.isFile(path: String): Bool
#[no_mangle]
pub extern "C" fn haxe_filesystem_is_file(path: *const HaxeString) -> bool {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => std::path::Path::new(&path_str).is_file(),
            None => false,
        }
    }
}

/// Read directory contents
/// FileSystem.readDirectory(path: String): Array<String>
#[no_mangle]
pub extern "C" fn haxe_filesystem_read_directory(
    path: *const HaxeString,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::{haxe_array_new, haxe_array_push, HaxeArray};

    unsafe {
        let path_str = match haxe_string_to_rust(path) {
            Some(s) => s,
            None => return std::ptr::null_mut(),
        };

        let entries = match std::fs::read_dir(&path_str) {
            Ok(entries) => entries,
            Err(_) => return std::ptr::null_mut(),
        };

        // Allocate array on heap
        let arr = Box::into_raw(Box::new(std::mem::zeroed::<HaxeArray>()));

        // Initialize array with 8-byte element size (pointer to HaxeString)
        haxe_array_new(arr, 8);

        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                // Skip . and ..
                if name == "." || name == ".." {
                    continue;
                }

                let haxe_str = rust_string_to_haxe(name.to_string());
                if !haxe_str.is_null() {
                    // Push pointer to string (pass address of the pointer)
                    let str_ptr = haxe_str as u64;
                    haxe_array_push(arr, &str_ptr as *const u64 as *const u8);
                }
            }
        }

        arr
    }
}

// ============================================================================
// FileInput (sys.io.FileInput) - File reading handle
// ============================================================================
//
// FileInput wraps a Rust File handle for reading operations.
// Extends haxe.io.Input which provides readByte() as the core method.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};

/// File input handle for reading
#[repr(C)]
pub struct HaxeFileInput {
    reader: BufReader<File>,
    eof_reached: bool,
}

/// File output handle for writing
#[repr(C)]
pub struct HaxeFileOutput {
    writer: BufWriter<File>,
}

/// FileSeek enum values (matching sys.io.FileSeek)
/// SeekBegin = 0, SeekCur = 1, SeekEnd = 2
const SEEK_BEGIN: i32 = 0;
const SEEK_CUR: i32 = 1;
const SEEK_END: i32 = 2;

/// Open file for reading
/// File.read(path: String, binary: Bool): FileInput
#[no_mangle]
pub extern "C" fn haxe_file_read(path: *const HaxeString, _binary: bool) -> *mut HaxeFileInput {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => match File::open(&path_str) {
                Ok(file) => Box::into_raw(Box::new(HaxeFileInput {
                    reader: BufReader::new(file),
                    eof_reached: false,
                })),
                Err(e) => {
                    debug!("File.read error: {} - {}", path_str, e);
                    std::ptr::null_mut()
                }
            },
            None => std::ptr::null_mut(),
        }
    }
}

/// Open file for writing (creates or truncates)
/// File.write(path: String, binary: Bool): FileOutput
#[no_mangle]
pub extern "C" fn haxe_file_write(path: *const HaxeString, _binary: bool) -> *mut HaxeFileOutput {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => match File::create(&path_str) {
                Ok(file) => Box::into_raw(Box::new(HaxeFileOutput {
                    writer: BufWriter::new(file),
                })),
                Err(e) => {
                    debug!("File.write error: {} - {}", path_str, e);
                    std::ptr::null_mut()
                }
            },
            None => std::ptr::null_mut(),
        }
    }
}

/// Open file for appending
/// File.append(path: String, binary: Bool): FileOutput
#[no_mangle]
pub extern "C" fn haxe_file_append(path: *const HaxeString, _binary: bool) -> *mut HaxeFileOutput {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => {
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path_str)
                {
                    Ok(file) => Box::into_raw(Box::new(HaxeFileOutput {
                        writer: BufWriter::new(file),
                    })),
                    Err(e) => {
                        debug!("File.append error: {} - {}", path_str, e);
                        std::ptr::null_mut()
                    }
                }
            }
            None => std::ptr::null_mut(),
        }
    }
}

/// Open file for updating (read/write, seek anywhere)
/// File.update(path: String, binary: Bool): FileOutput
#[no_mangle]
pub extern "C" fn haxe_file_update(path: *const HaxeString, _binary: bool) -> *mut HaxeFileOutput {
    unsafe {
        match haxe_string_to_rust(path) {
            Some(path_str) => {
                match std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&path_str)
                {
                    Ok(file) => Box::into_raw(Box::new(HaxeFileOutput {
                        writer: BufWriter::new(file),
                    })),
                    Err(e) => {
                        debug!("File.update error: {} - {}", path_str, e);
                        std::ptr::null_mut()
                    }
                }
            }
            None => std::ptr::null_mut(),
        }
    }
}

// ============================================================================
// FileInput methods (reading)
// ============================================================================

/// Read one byte from FileInput
/// FileInput.readByte(): Int
#[no_mangle]
pub extern "C" fn haxe_fileinput_read_byte(handle: *mut HaxeFileInput) -> i32 {
    if handle.is_null() {
        return -1;
    }
    unsafe {
        let input = &mut *handle;
        let mut buf = [0u8; 1];
        match input.reader.read(&mut buf) {
            Ok(0) => {
                input.eof_reached = true;
                -1 // EOF
            }
            Ok(_) => buf[0] as i32,
            Err(_) => {
                input.eof_reached = true;
                -1
            }
        }
    }
}

/// Read multiple bytes into buffer
/// Returns actual bytes read
#[no_mangle]
pub extern "C" fn haxe_fileinput_read_bytes(
    handle: *mut HaxeFileInput,
    buf: *mut u8,
    len: i32,
) -> i32 {
    if handle.is_null() || buf.is_null() || len <= 0 {
        return 0;
    }
    unsafe {
        let input = &mut *handle;
        let slice = std::slice::from_raw_parts_mut(buf, len as usize);
        match input.reader.read(slice) {
            Ok(0) => {
                input.eof_reached = true;
                0
            }
            Ok(n) => n as i32,
            Err(_) => {
                input.eof_reached = true;
                0
            }
        }
    }
}

/// Seek to position in FileInput
/// FileInput.seek(p: Int, pos: FileSeek): Void
#[no_mangle]
pub extern "C" fn haxe_fileinput_seek(handle: *mut HaxeFileInput, p: i32, pos: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let input = &mut *handle;
        let seek_pos = match pos {
            SEEK_BEGIN => SeekFrom::Start(p as u64),
            SEEK_CUR => SeekFrom::Current(p as i64),
            SEEK_END => SeekFrom::End(p as i64),
            _ => return,
        };
        let _ = input.reader.seek(seek_pos);
        input.eof_reached = false; // Reset EOF on seek
    }
}

/// Get current position in FileInput
/// FileInput.tell(): Int
#[no_mangle]
pub extern "C" fn haxe_fileinput_tell(handle: *mut HaxeFileInput) -> i32 {
    if handle.is_null() {
        return 0;
    }
    unsafe {
        let input = &mut *handle;
        match input.reader.stream_position() {
            Ok(pos) => pos as i32,
            Err(_) => 0,
        }
    }
}

/// Check if EOF reached
/// FileInput.eof(): Bool
#[no_mangle]
pub extern "C" fn haxe_fileinput_eof(handle: *mut HaxeFileInput) -> bool {
    if handle.is_null() {
        return true;
    }
    unsafe { (*handle).eof_reached }
}

/// Close FileInput
/// FileInput.close(): Void
#[no_mangle]
pub extern "C" fn haxe_fileinput_close(handle: *mut HaxeFileInput) {
    if handle.is_null() {
        return;
    }
    unsafe {
        // Drop the Box, which closes the file
        let _ = Box::from_raw(handle);
    }
}

// ============================================================================
// FileOutput methods (writing)
// ============================================================================

/// Write one byte to FileOutput
/// FileOutput.writeByte(c: Int): Void
#[no_mangle]
pub extern "C" fn haxe_fileoutput_write_byte(handle: *mut HaxeFileOutput, c: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let output = &mut *handle;
        let _ = output.writer.write(&[c as u8]);
    }
}

/// Write multiple bytes from buffer
/// Returns actual bytes written
#[no_mangle]
pub extern "C" fn haxe_fileoutput_write_bytes(
    handle: *mut HaxeFileOutput,
    buf: *const u8,
    len: i32,
) -> i32 {
    if handle.is_null() || buf.is_null() || len <= 0 {
        return 0;
    }
    unsafe {
        let output = &mut *handle;
        let slice = std::slice::from_raw_parts(buf, len as usize);
        match output.writer.write(slice) {
            Ok(n) => n as i32,
            Err(_) => 0,
        }
    }
}

/// Seek to position in FileOutput
/// FileOutput.seek(p: Int, pos: FileSeek): Void
#[no_mangle]
pub extern "C" fn haxe_fileoutput_seek(handle: *mut HaxeFileOutput, p: i32, pos: i32) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let output = &mut *handle;
        // Flush before seeking
        let _ = output.writer.flush();
        let seek_pos = match pos {
            SEEK_BEGIN => SeekFrom::Start(p as u64),
            SEEK_CUR => SeekFrom::Current(p as i64),
            SEEK_END => SeekFrom::End(p as i64),
            _ => return,
        };
        let _ = output.writer.seek(seek_pos);
    }
}

/// Get current position in FileOutput
/// FileOutput.tell(): Int
#[no_mangle]
pub extern "C" fn haxe_fileoutput_tell(handle: *mut HaxeFileOutput) -> i32 {
    if handle.is_null() {
        return 0;
    }
    unsafe {
        let output = &mut *handle;
        match output.writer.stream_position() {
            Ok(pos) => pos as i32,
            Err(_) => 0,
        }
    }
}

/// Flush FileOutput buffer
/// FileOutput.flush(): Void
#[no_mangle]
pub extern "C" fn haxe_fileoutput_flush(handle: *mut HaxeFileOutput) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let output = &mut *handle;
        let _ = output.writer.flush();
    }
}

/// Close FileOutput
/// FileOutput.close(): Void
#[no_mangle]
pub extern "C" fn haxe_fileoutput_close(handle: *mut HaxeFileOutput) {
    if handle.is_null() {
        return;
    }
    unsafe {
        // Flush and drop
        let mut output = Box::from_raw(handle);
        let _ = output.writer.flush();
        // Box drops here, closing the file
    }
}

// ============================================================================
// Date Class
// ============================================================================
//
// The Date class stores a timestamp in milliseconds since Unix epoch (1970-01-01).
// All getters compute values from this timestamp.

use chrono::{DateTime, Datelike, Local, NaiveDateTime, TimeZone, Timelike, Utc};

/// Haxe Date - stores milliseconds since Unix epoch
#[repr(C)]
pub struct HaxeDate {
    /// Milliseconds since 1970-01-01 00:00:00 UTC
    timestamp_ms: f64,
}

/// Create a new Date from components (local timezone)
/// Date.new(year, month, day, hour, min, sec)
#[no_mangle]
pub extern "C" fn haxe_date_new(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    min: i32,
    sec: i32,
) -> *mut HaxeDate {
    // month is 0-based in Haxe, chrono expects 1-based
    let naive = NaiveDateTime::new(
        chrono::NaiveDate::from_ymd_opt(year, (month + 1) as u32, day as u32)
            .unwrap_or_else(|| chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()),
        chrono::NaiveTime::from_hms_opt(hour as u32, min as u32, sec as u32)
            .unwrap_or_else(|| chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap()),
    );

    // Convert to local timezone then to timestamp
    let local: DateTime<Local> = Local
        .from_local_datetime(&naive)
        .single()
        .unwrap_or_else(Local::now);

    let timestamp_ms = local.timestamp_millis() as f64;

    Box::into_raw(Box::new(HaxeDate { timestamp_ms }))
}

/// Get current date/time
/// Date.now(): Date
#[no_mangle]
pub extern "C" fn haxe_date_now() -> *mut HaxeDate {
    let timestamp_ms = Local::now().timestamp_millis() as f64;
    Box::into_raw(Box::new(HaxeDate { timestamp_ms }))
}

/// Create Date from timestamp (milliseconds)
/// Date.fromTime(t: Float): Date
#[no_mangle]
pub extern "C" fn haxe_date_from_time(t: f64) -> *mut HaxeDate {
    Box::into_raw(Box::new(HaxeDate { timestamp_ms: t }))
}

/// Create Date from string
/// Date.fromString(s: String): Date
#[no_mangle]
pub extern "C" fn haxe_date_from_string(s: *const HaxeString) -> *mut HaxeDate {
    unsafe {
        let s_str = match haxe_string_to_rust(s) {
            Some(s) => s,
            None => return haxe_date_now(), // fallback to now
        };

        // Try parsing "YYYY-MM-DD hh:mm:ss"
        if let Ok(dt) = NaiveDateTime::parse_from_str(&s_str, "%Y-%m-%d %H:%M:%S") {
            let local = Local
                .from_local_datetime(&dt)
                .single()
                .unwrap_or_else(Local::now);
            return Box::into_raw(Box::new(HaxeDate {
                timestamp_ms: local.timestamp_millis() as f64,
            }));
        }

        // Try parsing "YYYY-MM-DD"
        if let Ok(d) = chrono::NaiveDate::parse_from_str(&s_str, "%Y-%m-%d") {
            let dt = d.and_hms_opt(0, 0, 0).unwrap();
            let local = Local
                .from_local_datetime(&dt)
                .single()
                .unwrap_or_else(Local::now);
            return Box::into_raw(Box::new(HaxeDate {
                timestamp_ms: local.timestamp_millis() as f64,
            }));
        }

        // Try parsing "hh:mm:ss" (relative to epoch)
        if let Ok(t) = chrono::NaiveTime::parse_from_str(&s_str, "%H:%M:%S") {
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
            let dt = epoch.and_time(t);
            return Box::into_raw(Box::new(HaxeDate {
                timestamp_ms: dt.and_utc().timestamp_millis() as f64,
            }));
        }

        haxe_date_now() // fallback
    }
}

/// Helper to get DateTime<Local> from HaxeDate
fn get_local_datetime(date: *const HaxeDate) -> Option<DateTime<Local>> {
    if date.is_null() {
        return None;
    }
    unsafe {
        let timestamp_ms = (*date).timestamp_ms as i64;
        let secs = timestamp_ms / 1000;
        let nsecs = ((timestamp_ms % 1000) * 1_000_000) as u32;
        DateTime::from_timestamp(secs, nsecs).map(|utc| utc.with_timezone(&Local))
    }
}

/// Helper to get DateTime<Utc> from HaxeDate
fn get_utc_datetime(date: *const HaxeDate) -> Option<DateTime<Utc>> {
    if date.is_null() {
        return None;
    }
    unsafe {
        let timestamp_ms = (*date).timestamp_ms as i64;
        let secs = timestamp_ms / 1000;
        let nsecs = ((timestamp_ms % 1000) * 1_000_000) as u32;
        DateTime::from_timestamp(secs, nsecs)
    }
}

/// Get timestamp in milliseconds
/// date.getTime(): Float
#[no_mangle]
pub extern "C" fn haxe_date_get_time(date: *const HaxeDate) -> f64 {
    if date.is_null() {
        return 0.0;
    }
    unsafe { (*date).timestamp_ms }
}

/// Get hours (0-23) in local timezone
/// date.getHours(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_hours(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| dt.hour() as i32)
        .unwrap_or(0)
}

/// Get minutes (0-59) in local timezone
/// date.getMinutes(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_minutes(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| dt.minute() as i32)
        .unwrap_or(0)
}

/// Get seconds (0-59) in local timezone
/// date.getSeconds(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_seconds(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| dt.second() as i32)
        .unwrap_or(0)
}

/// Get full year (4 digits) in local timezone
/// date.getFullYear(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_full_year(date: *const HaxeDate) -> i32 {
    get_local_datetime(date).map(|dt| dt.year()).unwrap_or(1970)
}

/// Get month (0-11) in local timezone
/// date.getMonth(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_month(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| (dt.month() - 1) as i32)
        .unwrap_or(0)
}

/// Get day of month (1-31) in local timezone
/// date.getDate(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_date(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| dt.day() as i32)
        .unwrap_or(1)
}

/// Get day of week (0-6, Sunday=0) in local timezone
/// date.getDay(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_day(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| dt.weekday().num_days_from_sunday() as i32)
        .unwrap_or(0)
}

/// Get hours (0-23) in UTC
/// date.getUTCHours(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_hours(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date)
        .map(|dt| dt.hour() as i32)
        .unwrap_or(0)
}

/// Get minutes (0-59) in UTC
/// date.getUTCMinutes(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_minutes(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date)
        .map(|dt| dt.minute() as i32)
        .unwrap_or(0)
}

/// Get seconds (0-59) in UTC
/// date.getUTCSeconds(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_seconds(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date)
        .map(|dt| dt.second() as i32)
        .unwrap_or(0)
}

/// Get full year (4 digits) in UTC
/// date.getUTCFullYear(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_full_year(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date).map(|dt| dt.year()).unwrap_or(1970)
}

/// Get month (0-11) in UTC
/// date.getUTCMonth(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_month(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date)
        .map(|dt| (dt.month() - 1) as i32)
        .unwrap_or(0)
}

/// Get day of month (1-31) in UTC
/// date.getUTCDate(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_date(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date)
        .map(|dt| dt.day() as i32)
        .unwrap_or(1)
}

/// Get day of week (0-6, Sunday=0) in UTC
/// date.getUTCDay(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_utc_day(date: *const HaxeDate) -> i32 {
    get_utc_datetime(date)
        .map(|dt| dt.weekday().num_days_from_sunday() as i32)
        .unwrap_or(0)
}

/// Get timezone offset in minutes (local - UTC)
/// date.getTimezoneOffset(): Int
#[no_mangle]
pub extern "C" fn haxe_date_get_timezone_offset(date: *const HaxeDate) -> i32 {
    get_local_datetime(date)
        .map(|dt| -(dt.offset().local_minus_utc() / 60))
        .unwrap_or(0)
}

/// Convert date to string "YYYY-MM-DD HH:MM:SS"
/// date.toString(): String
#[no_mangle]
pub extern "C" fn haxe_date_to_string(date: *const HaxeDate) -> *mut HaxeString {
    let s = get_local_datetime(date)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "1970-01-01 00:00:00".to_string());
    rust_string_to_haxe(s)
}

// ============================================================================
// Bytes (rayzor.Bytes / haxe.io.Bytes)
// ============================================================================
//
// Native byte buffer implementation.
// Memory layout matches vec_u8: { ptr: *u8, len: u64, cap: u64 }

/// Haxe Bytes - raw byte buffer
#[repr(C)]
pub struct HaxeBytes {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

/// Allocate a new Bytes of given size (zero-initialized)
/// Bytes.alloc(size: Int): Bytes
#[no_mangle]
pub extern "C" fn haxe_bytes_alloc(size: i32) -> *mut HaxeBytes {
    let size = size.max(0) as usize;
    let cap = size.max(16); // Minimum capacity of 16

    unsafe {
        let layout = std::alloc::Layout::from_size_align(cap, 1).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            return std::ptr::null_mut();
        }

        Box::into_raw(Box::new(HaxeBytes {
            ptr,
            len: size,
            cap,
        }))
    }
}

/// Create Bytes from String (UTF-8)
/// Bytes.ofString(s: String): Bytes
#[no_mangle]
pub extern "C" fn haxe_bytes_of_string(s: *const HaxeString) -> *mut HaxeBytes {
    unsafe {
        let s_str = match haxe_string_to_rust(s) {
            Some(s) => s,
            None => return haxe_bytes_alloc(0),
        };

        let bytes = s_str.into_bytes();
        let len = bytes.len();
        let cap = bytes.capacity();
        let ptr = bytes.as_ptr() as *mut u8;
        std::mem::forget(bytes);

        Box::into_raw(Box::new(HaxeBytes { ptr, len, cap }))
    }
}

/// Get the length of Bytes
/// bytes.length: Int
#[no_mangle]
pub extern "C" fn haxe_bytes_length(bytes: *const HaxeBytes) -> i32 {
    if bytes.is_null() {
        return 0;
    }
    unsafe { (*bytes).len as i32 }
}

/// Get a single byte
/// bytes.get(pos: Int): Int
#[no_mangle]
pub extern "C" fn haxe_bytes_get(bytes: *const HaxeBytes, pos: i32) -> i32 {
    if bytes.is_null() || pos < 0 {
        return 0;
    }
    unsafe {
        let b = &*bytes;
        if (pos as usize) >= b.len {
            return 0;
        }
        *b.ptr.add(pos as usize) as i32
    }
}

/// Set a single byte
/// bytes.set(pos: Int, value: Int): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_set(bytes: *mut HaxeBytes, pos: i32, value: i32) {
    if bytes.is_null() || pos < 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        if (pos as usize) >= b.len {
            return;
        }
        *b.ptr.add(pos as usize) = value as u8;
    }
}

/// Get a sub-range as new Bytes
/// bytes.sub(pos: Int, len: Int): Bytes
#[no_mangle]
pub extern "C" fn haxe_bytes_sub(bytes: *const HaxeBytes, pos: i32, len: i32) -> *mut HaxeBytes {
    if bytes.is_null() || pos < 0 || len < 0 {
        return haxe_bytes_alloc(0);
    }
    unsafe {
        let b = &*bytes;
        let pos = pos as usize;
        let len = len as usize;
        if pos >= b.len || pos + len > b.len {
            return haxe_bytes_alloc(0);
        }

        let new_bytes = haxe_bytes_alloc(len as i32);
        if !new_bytes.is_null() {
            std::ptr::copy_nonoverlapping(b.ptr.add(pos), (*new_bytes).ptr, len);
        }
        new_bytes
    }
}

/// Copy bytes from source to destination
/// bytes.blit(srcPos: Int, dest: Bytes, destPos: Int, len: Int): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_blit(
    src: *const HaxeBytes,
    src_pos: i32,
    dest: *mut HaxeBytes,
    dest_pos: i32,
    len: i32,
) {
    if src.is_null() || dest.is_null() || src_pos < 0 || dest_pos < 0 || len <= 0 {
        return;
    }
    unsafe {
        let s = &*src;
        let d = &mut *dest;
        let src_pos = src_pos as usize;
        let dest_pos = dest_pos as usize;
        let len = len as usize;

        if src_pos + len > s.len || dest_pos + len > d.len {
            return;
        }

        // Use memmove for potentially overlapping regions
        std::ptr::copy(s.ptr.add(src_pos), d.ptr.add(dest_pos), len);
    }
}

/// Fill a range with a byte value
/// bytes.fill(pos: Int, len: Int, value: Int): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_fill(bytes: *mut HaxeBytes, pos: i32, len: i32, value: i32) {
    if bytes.is_null() || pos < 0 || len <= 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        let pos = pos as usize;
        let len = len as usize;
        if pos + len > b.len {
            return;
        }
        std::ptr::write_bytes(b.ptr.add(pos), value as u8, len);
    }
}

/// Compare two Bytes
/// bytes.compare(other: Bytes): Int
#[no_mangle]
pub extern "C" fn haxe_bytes_compare(a: *const HaxeBytes, b: *const HaxeBytes) -> i32 {
    if a.is_null() && b.is_null() {
        return 0;
    }
    if a.is_null() {
        return -1;
    }
    if b.is_null() {
        return 1;
    }
    unsafe {
        let a = &*a;
        let b = &*b;
        let min_len = a.len.min(b.len);
        let cmp = libc::memcmp(a.ptr as *const _, b.ptr as *const _, min_len);
        if cmp != 0 {
            return cmp;
        }
        (a.len as i32) - (b.len as i32)
    }
}

/// Convert Bytes to String (UTF-8)
/// bytes.toString(): String
#[no_mangle]
pub extern "C" fn haxe_bytes_to_string(bytes: *const HaxeBytes) -> *mut HaxeString {
    if bytes.is_null() {
        return rust_string_to_haxe(String::new());
    }
    unsafe {
        let b = &*bytes;
        let slice = std::slice::from_raw_parts(b.ptr, b.len);
        let s = String::from_utf8_lossy(slice).into_owned();
        rust_string_to_haxe(s)
    }
}

/// Get 16-bit integer (little-endian)
/// bytes.getInt16(pos: Int): Int
#[no_mangle]
pub extern "C" fn haxe_bytes_get_int16(bytes: *const HaxeBytes, pos: i32) -> i32 {
    if bytes.is_null() || pos < 0 {
        return 0;
    }
    unsafe {
        let b = &*bytes;
        let pos = pos as usize;
        if pos + 2 > b.len {
            return 0;
        }
        let ptr = b.ptr.add(pos) as *const i16;
        i16::from_le(std::ptr::read_unaligned(ptr)) as i32
    }
}

/// Get 32-bit integer (little-endian)
/// bytes.getInt32(pos: Int): Int
#[no_mangle]
pub extern "C" fn haxe_bytes_get_int32(bytes: *const HaxeBytes, pos: i32) -> i32 {
    if bytes.is_null() || pos < 0 {
        return 0;
    }
    unsafe {
        let b = &*bytes;
        let pos = pos as usize;
        if pos + 4 > b.len {
            return 0;
        }
        let ptr = b.ptr.add(pos) as *const i32;
        i32::from_le(std::ptr::read_unaligned(ptr))
    }
}

/// Get 64-bit integer (little-endian)
/// bytes.getInt64(pos: Int): Int64
#[no_mangle]
pub extern "C" fn haxe_bytes_get_int64(bytes: *const HaxeBytes, pos: i32) -> i64 {
    if bytes.is_null() || pos < 0 {
        return 0;
    }
    unsafe {
        let b = &*bytes;
        let pos = pos as usize;
        if pos + 8 > b.len {
            return 0;
        }
        let ptr = b.ptr.add(pos) as *const i64;
        i64::from_le(std::ptr::read_unaligned(ptr))
    }
}

/// Get 32-bit float (little-endian)
/// bytes.getFloat(pos: Int): Float
#[no_mangle]
pub extern "C" fn haxe_bytes_get_float(bytes: *const HaxeBytes, pos: i32) -> f32 {
    if bytes.is_null() || pos < 0 {
        return 0.0;
    }
    unsafe {
        let b = &*bytes;
        let pos = pos as usize;
        if pos + 4 > b.len {
            return 0.0;
        }
        let ptr = b.ptr.add(pos) as *const u32;
        f32::from_bits(u32::from_le(std::ptr::read_unaligned(ptr)))
    }
}

/// Get 64-bit double (little-endian)
/// bytes.getDouble(pos: Int): Float
#[no_mangle]
pub extern "C" fn haxe_bytes_get_double(bytes: *const HaxeBytes, pos: i32) -> f64 {
    if bytes.is_null() || pos < 0 {
        return 0.0;
    }
    unsafe {
        let b = &*bytes;
        let pos = pos as usize;
        if pos + 8 > b.len {
            return 0.0;
        }
        let ptr = b.ptr.add(pos) as *const u64;
        f64::from_bits(u64::from_le(std::ptr::read_unaligned(ptr)))
    }
}

/// Set 16-bit integer (little-endian)
/// bytes.setInt16(pos: Int, value: Int): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_set_int16(bytes: *mut HaxeBytes, pos: i32, value: i32) {
    if bytes.is_null() || pos < 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        let pos = pos as usize;
        if pos + 2 > b.len {
            return;
        }
        let ptr = b.ptr.add(pos) as *mut i16;
        std::ptr::write_unaligned(ptr, (value as i16).to_le());
    }
}

/// Set 32-bit integer (little-endian)
/// bytes.setInt32(pos: Int, value: Int): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_set_int32(bytes: *mut HaxeBytes, pos: i32, value: i32) {
    if bytes.is_null() || pos < 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        let pos = pos as usize;
        if pos + 4 > b.len {
            return;
        }
        let ptr = b.ptr.add(pos) as *mut i32;
        std::ptr::write_unaligned(ptr, value.to_le());
    }
}

/// Set 64-bit integer (little-endian)
/// bytes.setInt64(pos: Int, value: Int64): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_set_int64(bytes: *mut HaxeBytes, pos: i32, value: i64) {
    if bytes.is_null() || pos < 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        let pos = pos as usize;
        if pos + 8 > b.len {
            return;
        }
        let ptr = b.ptr.add(pos) as *mut i64;
        std::ptr::write_unaligned(ptr, value.to_le());
    }
}

/// Set 32-bit float (little-endian)
/// bytes.setFloat(pos: Int, value: Float): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_set_float(bytes: *mut HaxeBytes, pos: i32, value: f32) {
    if bytes.is_null() || pos < 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        let pos = pos as usize;
        if pos + 4 > b.len {
            return;
        }
        let ptr = b.ptr.add(pos) as *mut u32;
        std::ptr::write_unaligned(ptr, value.to_bits().to_le());
    }
}

/// Set 64-bit double (little-endian)
/// bytes.setDouble(pos: Int, value: Float): Void
#[no_mangle]
pub extern "C" fn haxe_bytes_set_double(bytes: *mut HaxeBytes, pos: i32, value: f64) {
    if bytes.is_null() || pos < 0 {
        return;
    }
    unsafe {
        let b = &mut *bytes;
        let pos = pos as usize;
        if pos + 8 > b.len {
            return;
        }
        let ptr = b.ptr.add(pos) as *mut u64;
        std::ptr::write_unaligned(ptr, value.to_bits().to_le());
    }
}

/// Free Bytes memory
#[no_mangle]
pub extern "C" fn haxe_bytes_free(bytes: *mut HaxeBytes) {
    if bytes.is_null() {
        return;
    }
    unsafe {
        let b = Box::from_raw(bytes);
        if !b.ptr.is_null() && b.cap > 0 {
            let layout = std::alloc::Layout::from_size_align(b.cap, 1).unwrap();
            std::alloc::dealloc(b.ptr, layout);
        }
    }
}

// ============================================================================
// StringMap<T> (haxe.ds.StringMap)
// ============================================================================
//
// High-performance StringMap with inline value storage.
// Values are stored as raw 64-bit values (u64) - no boxing, no heap allocation per value.
// Type is known at compile time; the runtime stores raw bits.
//
// For primitives (Int, Float, Bool): value is stored directly as bits
// For pointers (String, objects): pointer value is stored as u64
//
// This gives us:
// - No heap allocation per value (values inline in HashMap)
// - No type tags (type known at compile time)
// - Cache-friendly layout
// - Zero-cost abstraction over HashMap<String, u64>

use std::collections::HashMap;

/// High-performance StringMap with inline 8-byte value storage
/// Values are stored as raw u64 bits - no boxing overhead
#[repr(C)]
pub struct HaxeStringMap {
    map: HashMap<String, u64>,
}

/// Create a new StringMap
#[no_mangle]
pub extern "C" fn haxe_stringmap_new() -> *mut HaxeStringMap {
    Box::into_raw(Box::new(HaxeStringMap {
        map: HashMap::new(),
    }))
}

/// Set a value in the StringMap
/// Value is passed as raw u64 bits (compiler handles type conversion)
#[no_mangle]
pub extern "C" fn haxe_stringmap_set(
    map_ptr: *mut HaxeStringMap,
    key: *const HaxeString,
    value: u64,
) {
    if map_ptr.is_null() {
        return;
    }
    unsafe {
        let map = &mut *map_ptr;
        if let Some(key_str) = haxe_string_to_rust(key) {
            map.map.insert(key_str, value);
        }
    }
}

/// Get a value from the StringMap
/// Returns raw u64 bits (compiler handles type conversion)
/// Returns 0 if key doesn't exist (caller should use exists() to distinguish)
#[no_mangle]
pub extern "C" fn haxe_stringmap_get(map_ptr: *mut HaxeStringMap, key: *const HaxeString) -> u64 {
    if map_ptr.is_null() {
        return 0;
    }
    unsafe {
        let map = &*map_ptr;
        if let Some(key_str) = haxe_string_to_rust(key) {
            map.map.get(&key_str).copied().unwrap_or(0)
        } else {
            0
        }
    }
}

/// Check if a key exists in the StringMap
#[no_mangle]
pub extern "C" fn haxe_stringmap_exists(
    map_ptr: *mut HaxeStringMap,
    key: *const HaxeString,
) -> bool {
    if map_ptr.is_null() {
        return false;
    }
    unsafe {
        let map = &*map_ptr;
        if let Some(key_str) = haxe_string_to_rust(key) {
            map.map.contains_key(&key_str)
        } else {
            false
        }
    }
}

/// Remove a key from the StringMap
/// Returns true if the key existed and was removed
#[no_mangle]
pub extern "C" fn haxe_stringmap_remove(
    map_ptr: *mut HaxeStringMap,
    key: *const HaxeString,
) -> bool {
    if map_ptr.is_null() {
        return false;
    }
    unsafe {
        let map = &mut *map_ptr;
        if let Some(key_str) = haxe_string_to_rust(key) {
            map.map.remove(&key_str).is_some()
        } else {
            false
        }
    }
}

/// Clear all entries from the StringMap
#[no_mangle]
pub extern "C" fn haxe_stringmap_clear(map_ptr: *mut HaxeStringMap) {
    if map_ptr.is_null() {
        return;
    }
    unsafe {
        let map = &mut *map_ptr;
        map.map.clear();
    }
}

/// Get the number of entries in the map
#[no_mangle]
pub extern "C" fn haxe_stringmap_count(map_ptr: *mut HaxeStringMap) -> i64 {
    if map_ptr.is_null() {
        return 0;
    }
    unsafe {
        let map = &*map_ptr;
        map.map.len() as i64
    }
}

/// Get all keys as an array
/// Returns pointer to array of HaxeString pointers, sets out_len to count
#[no_mangle]
pub extern "C" fn haxe_stringmap_keys(
    map_ptr: *mut HaxeStringMap,
    out_len: *mut i64,
) -> *mut *mut HaxeString {
    if map_ptr.is_null() || out_len.is_null() {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    }
    unsafe {
        let map = &*map_ptr;
        let keys: Vec<*mut HaxeString> = map
            .map
            .keys()
            .map(|k| rust_string_to_haxe(k.clone()))
            .collect();
        *out_len = keys.len() as i64;
        Box::into_raw(keys.into_boxed_slice()) as *mut *mut HaxeString
    }
}

/// Convert StringMap to string representation
#[no_mangle]
pub extern "C" fn haxe_stringmap_to_string(map_ptr: *mut HaxeStringMap) -> *mut HaxeString {
    if map_ptr.is_null() {
        return rust_string_to_haxe("{}".to_string());
    }
    unsafe {
        let map = &*map_ptr;
        let entries: Vec<String> = map
            .map
            .iter()
            .map(|(k, v)| format!("{} => {}", k, v))
            .collect();
        let result = format!("{{{}}}", entries.join(", "));
        rust_string_to_haxe(result)
    }
}

// ============================================================================
// IntMap<T> (haxe.ds.IntMap)
// ============================================================================
//
// High-performance IntMap with inline value storage.
// Same design as StringMap - values stored as raw u64 bits.

/// High-performance IntMap with inline 8-byte value storage
#[repr(C)]
pub struct HaxeIntMap {
    map: HashMap<i64, u64>,
}

/// Create a new IntMap
#[no_mangle]
pub extern "C" fn haxe_intmap_new() -> *mut HaxeIntMap {
    Box::into_raw(Box::new(HaxeIntMap {
        map: HashMap::new(),
    }))
}

/// Set a value in the IntMap
/// Value is passed as raw u64 bits
#[no_mangle]
pub extern "C" fn haxe_intmap_set(map_ptr: *mut HaxeIntMap, key: i64, value: u64) {
    if map_ptr.is_null() {
        return;
    }
    unsafe {
        let map = &mut *map_ptr;
        map.map.insert(key, value);
    }
}

/// Get a value from the IntMap
/// Returns raw u64 bits, 0 if key doesn't exist
#[no_mangle]
pub extern "C" fn haxe_intmap_get(map_ptr: *mut HaxeIntMap, key: i64) -> u64 {
    if map_ptr.is_null() {
        return 0;
    }
    unsafe {
        let map = &*map_ptr;
        map.map.get(&key).copied().unwrap_or(0)
    }
}

/// Check if a key exists in the IntMap
#[no_mangle]
pub extern "C" fn haxe_intmap_exists(map_ptr: *mut HaxeIntMap, key: i64) -> bool {
    if map_ptr.is_null() {
        return false;
    }
    unsafe {
        let map = &*map_ptr;
        map.map.contains_key(&key)
    }
}

/// Remove a key from the IntMap
/// Returns true if the key existed and was removed
#[no_mangle]
pub extern "C" fn haxe_intmap_remove(map_ptr: *mut HaxeIntMap, key: i64) -> bool {
    if map_ptr.is_null() {
        return false;
    }
    unsafe {
        let map = &mut *map_ptr;
        map.map.remove(&key).is_some()
    }
}

/// Clear all entries from the IntMap
#[no_mangle]
pub extern "C" fn haxe_intmap_clear(map_ptr: *mut HaxeIntMap) {
    if map_ptr.is_null() {
        return;
    }
    unsafe {
        let map = &mut *map_ptr;
        map.map.clear();
    }
}

/// Get the number of entries in the map
#[no_mangle]
pub extern "C" fn haxe_intmap_count(map_ptr: *mut HaxeIntMap) -> i64 {
    if map_ptr.is_null() {
        return 0;
    }
    unsafe {
        let map = &*map_ptr;
        map.map.len() as i64
    }
}

/// Get all keys as an array
/// Returns pointer to array of i64, sets out_len to count
#[no_mangle]
pub extern "C" fn haxe_intmap_keys(map_ptr: *mut HaxeIntMap, out_len: *mut i64) -> *mut i64 {
    if map_ptr.is_null() || out_len.is_null() {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    }
    unsafe {
        let map = &*map_ptr;
        let keys: Vec<i64> = map.map.keys().copied().collect();
        *out_len = keys.len() as i64;
        Box::into_raw(keys.into_boxed_slice()) as *mut i64
    }
}

/// Convert IntMap to string representation
#[no_mangle]
pub extern "C" fn haxe_intmap_to_string(map_ptr: *mut HaxeIntMap) -> *mut HaxeString {
    if map_ptr.is_null() {
        return rust_string_to_haxe("{}".to_string());
    }
    unsafe {
        let map = &*map_ptr;
        let entries: Vec<String> = map
            .map
            .iter()
            .map(|(k, v)| format!("{} => {}", k, v))
            .collect();
        let result = format!("{{{}}}", entries.join(", "));
        rust_string_to_haxe(result)
    }
}

/// Get StringMap keys as a HaxeArray of HaxeString pointers.
/// Returns a pointer to a heap-allocated HaxeArray with elem_size=8 (pointer-sized elements).
#[no_mangle]
pub extern "C" fn haxe_stringmap_keys_to_array(
    map_ptr: *mut HaxeStringMap,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;
    use std::alloc::{alloc, Layout};

    unsafe {
        let arr = alloc(Layout::new::<HaxeArray>()) as *mut HaxeArray;
        crate::haxe_array::haxe_array_new(arr, 8); // 8 bytes per element (pointer-sized)

        if !map_ptr.is_null() {
            let map = &*map_ptr;
            for key in map.map.keys() {
                let hs = rust_string_to_haxe(key.clone());
                let hs_as_i64 = hs as i64;
                crate::haxe_array::haxe_array_push_i64(arr, hs_as_i64);
            }
        }

        arr
    }
}

/// Get IntMap keys as a HaxeArray of i64 values.
/// Returns a pointer to a heap-allocated HaxeArray with elem_size=8.
#[no_mangle]
pub extern "C" fn haxe_intmap_keys_to_array(
    map_ptr: *mut HaxeIntMap,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;
    use std::alloc::{alloc, Layout};

    unsafe {
        let arr = alloc(Layout::new::<HaxeArray>()) as *mut HaxeArray;
        crate::haxe_array::haxe_array_new(arr, 8);

        if !map_ptr.is_null() {
            let map = &*map_ptr;
            for &key in map.map.keys() {
                crate::haxe_array::haxe_array_push_i64(arr, key);
            }
        }

        arr
    }
}

/// Get StringMap values as a HaxeArray of u64 raw values.
/// Returns a pointer to a heap-allocated HaxeArray with elem_size=8.
#[no_mangle]
pub extern "C" fn haxe_stringmap_values_to_array(
    map_ptr: *mut HaxeStringMap,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;
    use std::alloc::{alloc, Layout};

    unsafe {
        let arr = alloc(Layout::new::<HaxeArray>()) as *mut HaxeArray;
        crate::haxe_array::haxe_array_new(arr, 8);

        if !map_ptr.is_null() {
            let map = &*map_ptr;
            for &val in map.map.values() {
                crate::haxe_array::haxe_array_push_i64(arr, val as i64);
            }
        }

        arr
    }
}

/// Get IntMap values as a HaxeArray of u64 raw values.
/// Returns a pointer to a heap-allocated HaxeArray with elem_size=8.
#[no_mangle]
pub extern "C" fn haxe_intmap_values_to_array(
    map_ptr: *mut HaxeIntMap,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;
    use std::alloc::{alloc, Layout};

    unsafe {
        let arr = alloc(Layout::new::<HaxeArray>()) as *mut HaxeArray;
        crate::haxe_array::haxe_array_new(arr, 8);

        if !map_ptr.is_null() {
            let map = &*map_ptr;
            for &val in map.map.values() {
                crate::haxe_array::haxe_array_push_i64(arr, val as i64);
            }
        }

        arr
    }
}
