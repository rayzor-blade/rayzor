//! String implementation for Haxe
//!
//! Haxe strings are immutable value types. All string data is bump-allocated
//! from a thread-local arena and freed in bulk at program exit.

use crate::arena::{arena_alloc_bytes, arena_alloc_haxe_string};
use std::ptr;

// Re-export from the canonical definition in haxe_string module
pub use crate::haxe_string::HaxeString;

/// Create a new empty string (internal helper, not exported)
pub fn haxe_string_new() -> HaxeString {
    const INITIAL_CAPACITY: usize = 16;
    let ptr = arena_alloc_bytes(INITIAL_CAPACITY);
    HaxeString {
        ptr,
        len: 0,
        cap: INITIAL_CAPACITY,
    }
}

/// Create a string from a C string (null-terminated) - internal helper
#[allow(dead_code)]
fn haxe_string_from_cstr(cstr: *const u8) -> HaxeString {
    unsafe {
        if cstr.is_null() {
            return haxe_string_new();
        }

        // Find length
        let mut len = 0;
        while *cstr.add(len) != 0 {
            len += 1;
        }

        // Allocate
        let cap = if len < 16 { 16 } else { len };
        let ptr = arena_alloc_bytes(cap);

        // Copy
        ptr::copy_nonoverlapping(cstr, ptr, len);

        HaxeString { ptr, len, cap }
    }
}

/// Create a string from bytes with length - internal helper
fn haxe_string_from_bytes(bytes: *const u8, len: usize) -> HaxeString {
    if bytes.is_null() || len == 0 {
        return haxe_string_new();
    }

    let cap = if len < 16 { 16 } else { len };
    let ptr = arena_alloc_bytes(cap);

    unsafe {
        ptr::copy_nonoverlapping(bytes, ptr, len);
    }

    HaxeString { ptr, len, cap }
}

/// Get the length of the string in bytes - internal helper
#[allow(dead_code)]
fn haxe_string_len(s: *const HaxeString) -> usize {
    unsafe {
        if s.is_null() {
            return 0;
        }
        (*s).len
    }
}

/// Get a byte at index - internal helper
#[allow(dead_code)]
fn haxe_string_char_at(s: *const HaxeString, index: usize) -> u8 {
    unsafe {
        if s.is_null() {
            return 0;
        }

        let str_ref = &*s;
        if index >= str_ref.len {
            return 0;
        }

        *str_ref.ptr.add(index)
    }
}

/// Concatenate two strings and return a heap-allocated result pointer
/// This avoids struct return ABI issues
/// Note: exported as both `haxe_string_concat` (used by MIR/AOT) and `haxe_string_concat_ptr`
#[no_mangle]
pub extern "C" fn haxe_string_concat(
    a: *const HaxeString,
    b: *const HaxeString,
) -> *mut HaxeString {
    haxe_string_concat_ptr(a, b)
}

#[no_mangle]
pub extern "C" fn haxe_string_concat_ptr(
    a: *const HaxeString,
    b: *const HaxeString,
) -> *mut HaxeString {
    let result = haxe_string_concat_impl(a, b);
    arena_alloc_haxe_string(result)
}

/// Concatenate two strings (returns value - may have ABI issues with large structs)
fn haxe_string_concat_impl(a: *const HaxeString, b: *const HaxeString) -> HaxeString {
    unsafe {
        if a.is_null() && b.is_null() {
            return haxe_string_new();
        }

        let a_ref = if a.is_null() {
            &HaxeString {
                ptr: ptr::null_mut(),
                len: 0,
                cap: 0,
            }
        } else {
            &*a
        };

        let b_ref = if b.is_null() {
            &HaxeString {
                ptr: ptr::null_mut(),
                len: 0,
                cap: 0,
            }
        } else {
            &*b
        };

        // Guard against corrupted HaxeString structs
        let a_len = if a_ref.ptr.is_null() { 0 } else { a_ref.len };
        let b_len = if b_ref.ptr.is_null() { 0 } else { b_ref.len };

        let new_len = a_len + b_len;
        let new_cap = if new_len < 16 { 16 } else { new_len };

        let new_ptr = arena_alloc_bytes(new_cap);

        // Copy both strings
        if a_len > 0 && !a_ref.ptr.is_null() {
            ptr::copy_nonoverlapping(a_ref.ptr, new_ptr, a_len);
        }
        if b_len > 0 && !b_ref.ptr.is_null() {
            ptr::copy_nonoverlapping(b_ref.ptr, new_ptr.add(a_len), b_len);
        }

        HaxeString {
            ptr: new_ptr,
            len: new_len,
            cap: new_cap,
        }
    }
}

/// Get a substring - internal helper
#[allow(dead_code)]
fn haxe_string_substr(s: *const HaxeString, start: usize, len: usize) -> HaxeString {
    unsafe {
        if s.is_null() {
            return haxe_string_new();
        }

        let str_ref = &*s;

        if start >= str_ref.len {
            return haxe_string_new();
        }

        let actual_len = if start + len > str_ref.len {
            str_ref.len - start
        } else {
            len
        };

        haxe_string_from_bytes(str_ref.ptr.add(start), actual_len)
    }
}

/// Convert to lowercase (ASCII only for now) - internal helper
#[allow(dead_code)]
fn haxe_string_to_lower(s: *const HaxeString) -> HaxeString {
    unsafe {
        if s.is_null() {
            return haxe_string_new();
        }

        let str_ref = &*s;
        let result = haxe_string_from_bytes(str_ref.ptr, str_ref.len);

        for i in 0..result.len {
            let byte = *result.ptr.add(i);
            if byte.is_ascii_uppercase() {
                *result.ptr.add(i) = byte + 32; // Convert to lowercase
            }
        }

        result
    }
}

/// Convert to uppercase (ASCII only for now) - internal helper
#[allow(dead_code)]
fn haxe_string_to_upper(s: *const HaxeString) -> HaxeString {
    unsafe {
        if s.is_null() {
            return haxe_string_new();
        }

        let str_ref = &*s;
        let result = haxe_string_from_bytes(str_ref.ptr, str_ref.len);

        for i in 0..result.len {
            let byte = *result.ptr.add(i);
            if byte.is_ascii_lowercase() {
                *result.ptr.add(i) = byte - 32; // Convert to uppercase
            }
        }

        result
    }
}

/// Free a string — no-op with arena allocation.
/// Kept for API compatibility; arena frees all memory in bulk at program exit.
#[allow(dead_code)]
fn haxe_string_free(_s: *mut HaxeString) {
    // No-op: arena handles deallocation
}

/// Get pointer to the string data (for debugging/printing) - internal helper
#[allow(dead_code)]
fn haxe_string_as_ptr(s: *const HaxeString) -> *const u8 {
    unsafe {
        if s.is_null() {
            return ptr::null();
        }
        (*s).ptr
    }
}

/// Check if a string starts with another string
/// Returns 1 (true) or 0 (false)
#[no_mangle]
pub extern "C" fn haxe_string_starts_with(s: *const HaxeString, prefix: *const HaxeString) -> i8 {
    unsafe {
        // Handle null cases
        if s.is_null() || prefix.is_null() {
            return 0;
        }

        let s_ref = &*s;
        let prefix_ref = &*prefix;

        // Empty prefix always matches
        if prefix_ref.len == 0 {
            return 1;
        }

        // String shorter than prefix can't start with it
        if s_ref.len < prefix_ref.len {
            return 0;
        }

        // Compare bytes
        for i in 0..prefix_ref.len {
            if *s_ref.ptr.add(i) != *prefix_ref.ptr.add(i) {
                return 0;
            }
        }

        1
    }
}

/// Check if a string ends with another string
/// Returns 1 (true) or 0 (false)
#[no_mangle]
pub extern "C" fn haxe_string_ends_with(s: *const HaxeString, suffix: *const HaxeString) -> i8 {
    unsafe {
        // Handle null cases
        if s.is_null() || suffix.is_null() {
            return 0;
        }

        let s_ref = &*s;
        let suffix_ref = &*suffix;

        // Empty suffix always matches
        if suffix_ref.len == 0 {
            return 1;
        }

        // String shorter than suffix can't end with it
        if s_ref.len < suffix_ref.len {
            return 0;
        }

        // Compare bytes at the end
        let offset = s_ref.len - suffix_ref.len;
        for i in 0..suffix_ref.len {
            if *s_ref.ptr.add(offset + i) != *suffix_ref.ptr.add(i) {
                return 0;
            }
        }

        1
    }
}

/// Check if a string contains another string
/// Returns 1 (true) or 0 (false)
#[no_mangle]
pub extern "C" fn haxe_string_contains(s: *const HaxeString, needle: *const HaxeString) -> i8 {
    unsafe {
        // Handle null cases
        if s.is_null() || needle.is_null() {
            return 0;
        }

        let s_ref = &*s;
        let needle_ref = &*needle;

        // Empty needle always matches
        if needle_ref.len == 0 {
            return 1;
        }

        // String shorter than needle can't contain it
        if s_ref.len < needle_ref.len {
            return 0;
        }

        // Naive string search algorithm
        let search_len = s_ref.len - needle_ref.len + 1;
        'outer: for start in 0..search_len {
            for i in 0..needle_ref.len {
                if *s_ref.ptr.add(start + i) != *needle_ref.ptr.add(i) {
                    continue 'outer;
                }
            }
            // All characters matched
            return 1;
        }

        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;

    /// Safe concat wrapper for testing
    fn haxe_string_concat(a: &HaxeString, b: &HaxeString) -> HaxeString {
        haxe_string_concat_impl(a as *const HaxeString, b as *const HaxeString)
    }

    #[test]
    fn test_string_new() {
        let s = haxe_string_new();
        assert!(!s.ptr.is_null());
        assert_eq!(s.len, 0);
        assert_eq!(s.cap, 16);
    }

    #[test]
    fn test_string_from_bytes() {
        let bytes = b"Hello, World!";
        let s = haxe_string_from_bytes(bytes.as_ptr(), bytes.len());

        assert_eq!(haxe_string_len(&s), 13);

        unsafe {
            let slice = slice::from_raw_parts(s.ptr, s.len);
            assert_eq!(slice, bytes);
        }
    }

    #[test]
    fn test_string_concat() {
        let s1 = haxe_string_from_bytes(b"Hello, ".as_ptr(), 7);
        let s2 = haxe_string_from_bytes(b"World!".as_ptr(), 6);

        let result = haxe_string_concat(&s1, &s2);

        assert_eq!(haxe_string_len(&result), 13);

        unsafe {
            let slice = slice::from_raw_parts(result.ptr, result.len);
            assert_eq!(slice, b"Hello, World!");
        }
    }

    #[test]
    fn test_string_to_upper_lower() {
        let s = haxe_string_from_bytes(b"Hello".as_ptr(), 5);

        let upper = haxe_string_to_upper(&s);
        let lower = haxe_string_to_lower(&s);

        unsafe {
            let upper_slice = slice::from_raw_parts(upper.ptr, upper.len);
            let lower_slice = slice::from_raw_parts(lower.ptr, lower.len);

            assert_eq!(upper_slice, b"HELLO");
            assert_eq!(lower_slice, b"hello");
        }
    }
}
