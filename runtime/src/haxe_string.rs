//! Haxe String runtime implementation
//!
//! Memory layout: [length: usize, capacity: usize, data...]
//! All strings are UTF-8 encoded and null-terminated for C interop

use log::debug;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::slice;
use std::str;

/// Haxe String representation (pointer-based, no struct returns)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HaxeString {
    pub ptr: *mut u8, // Pointer to string data (UTF-8)
    pub len: usize,   // Length in bytes
    pub cap: usize,   // Capacity in bytes
}

const INITIAL_CAPACITY: usize = 32;

// ============================================================================
// String Creation
// ============================================================================

/// Create a new empty string
#[no_mangle]
pub extern "C" fn haxe_string_new(out: *mut HaxeString) {
    unsafe {
        let layout = Layout::from_size_align_unchecked(INITIAL_CAPACITY, 1);
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        *ptr = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = 0;
        (*out).cap = INITIAL_CAPACITY;
    }
}

/// Create a string from a C string (null-terminated)
#[no_mangle]
pub extern "C" fn haxe_string_from_cstr(out: *mut HaxeString, cstr: *const u8) {
    if cstr.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        // Find length
        let mut len = 0;
        while *cstr.add(len) != 0 {
            len += 1;
        }

        let cap = len.max(INITIAL_CAPACITY) + 1; // +1 for null terminator
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        // Copy data
        ptr::copy_nonoverlapping(cstr, ptr, len);
        *ptr.add(len) = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = len;
        (*out).cap = cap;
    }
}

/// Create a string from bytes with known length
#[no_mangle]
pub extern "C" fn haxe_string_from_bytes(out: *mut HaxeString, bytes: *const u8, len: usize) {
    if bytes.is_null() || len == 0 {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let cap = len.max(INITIAL_CAPACITY) + 1;
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        ptr::copy_nonoverlapping(bytes, ptr, len);
        *ptr.add(len) = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = len;
        (*out).cap = cap;
    }
}

// ============================================================================
// String Properties
// ============================================================================

/// Get string length
#[no_mangle]
pub extern "C" fn haxe_string_length(s: *const HaxeString) -> usize {
    if s.is_null() {
        return 0;
    }
    unsafe { (*s).len }
}

/// Get character at index
#[no_mangle]
pub extern "C" fn haxe_string_char_at(s: *const HaxeString, index: usize) -> i32 {
    if s.is_null() {
        return -1;
    }

    unsafe {
        let s_ref = &*s;
        if index >= s_ref.len {
            return -1;
        }
        *s_ref.ptr.add(index) as i32
    }
}

/// Get character code at index
#[no_mangle]
pub extern "C" fn haxe_string_char_code_at(s: *const HaxeString, index: usize) -> i32 {
    haxe_string_char_at(s, index)
}

// ============================================================================
// String Operations
// ============================================================================

/// Concatenate two strings (sret variant — use haxe_string_concat_ptr instead)
#[no_mangle]
pub extern "C" fn haxe_string_concat_sret(
    out: *mut HaxeString,
    a: *const HaxeString,
    b: *const HaxeString,
) {
    if a.is_null() && b.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let a_len = if a.is_null() { 0 } else { (*a).len };
        let b_len = if b.is_null() { 0 } else { (*b).len };
        let total_len = a_len + b_len;

        let cap = total_len.max(INITIAL_CAPACITY) + 1;
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        // Copy first string
        if a_len > 0 {
            ptr::copy_nonoverlapping((*a).ptr, ptr, a_len);
        }

        // Copy second string
        if b_len > 0 {
            ptr::copy_nonoverlapping((*b).ptr, ptr.add(a_len), b_len);
        }

        *ptr.add(total_len) = 0; // Null terminator

        (*out).ptr = ptr;
        (*out).len = total_len;
        (*out).cap = cap;
    }
}

/// Get substring
#[no_mangle]
pub extern "C" fn haxe_string_substring(
    out: *mut HaxeString,
    s: *const HaxeString,
    start: usize,
    end: usize,
) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        let actual_start = start.min(s_ref.len);
        let actual_end = end.min(s_ref.len);

        if actual_start >= actual_end {
            haxe_string_new(out);
            return;
        }

        let len = actual_end - actual_start;
        let cap = len.max(INITIAL_CAPACITY) + 1;
        let layout = Layout::from_size_align_unchecked(cap, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for String");
        }

        ptr::copy_nonoverlapping(s_ref.ptr.add(actual_start), ptr, len);
        *ptr.add(len) = 0;

        (*out).ptr = ptr;
        (*out).len = len;
        (*out).cap = cap;
    }
}

/// Substring with just start position (to end of string)
#[no_mangle]
pub extern "C" fn haxe_string_substr(
    out: *mut HaxeString,
    s: *const HaxeString,
    start: usize,
    length: usize,
) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        let actual_start = start.min(s_ref.len);
        let actual_end = (start + length).min(s_ref.len);
        haxe_string_substring(out, s, actual_start, actual_end);
    }
}

/// Convert to uppercase
#[no_mangle]
pub extern "C" fn haxe_string_to_upper_case(out: *mut HaxeString, s: *const HaxeString) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        if s_ref.len == 0 {
            haxe_string_new(out);
            return;
        }

        let slice = slice::from_raw_parts(s_ref.ptr, s_ref.len);
        if let Ok(rust_str) = str::from_utf8(slice) {
            let upper = rust_str.to_uppercase();
            haxe_string_from_bytes(out, upper.as_ptr(), upper.len());
        } else {
            // Invalid UTF-8, just copy
            haxe_string_from_bytes(out, s_ref.ptr, s_ref.len);
        }
    }
}

/// Convert to lowercase
#[no_mangle]
pub extern "C" fn haxe_string_to_lower_case(out: *mut HaxeString, s: *const HaxeString) {
    if s.is_null() {
        haxe_string_new(out);
        return;
    }

    unsafe {
        let s_ref = &*s;
        if s_ref.len == 0 {
            haxe_string_new(out);
            return;
        }

        let slice = slice::from_raw_parts(s_ref.ptr, s_ref.len);
        if let Ok(rust_str) = str::from_utf8(slice) {
            let lower = rust_str.to_lowercase();
            haxe_string_from_bytes(out, lower.as_ptr(), lower.len());
        } else {
            // Invalid UTF-8, just copy
            haxe_string_from_bytes(out, s_ref.ptr, s_ref.len);
        }
    }
}

/// Index of substring
#[no_mangle]
pub extern "C" fn haxe_string_index_of(
    s: *const HaxeString,
    needle: *const HaxeString,
    start: usize,
) -> i32 {
    if s.is_null() || needle.is_null() {
        return -1;
    }

    unsafe {
        let s_ref = &*s;
        let needle_ref = &*needle;

        if needle_ref.len == 0 || start >= s_ref.len {
            return -1;
        }

        let haystack = slice::from_raw_parts(s_ref.ptr, s_ref.len);
        let needle_bytes = slice::from_raw_parts(needle_ref.ptr, needle_ref.len);

        // Simple substring search
        for i in start..=(s_ref.len.saturating_sub(needle_ref.len)) {
            if &haystack[i..i + needle_ref.len] == needle_bytes {
                return i as i32;
            }
        }

        -1
    }
}

/// Split string by delimiter
#[no_mangle]
pub extern "C" fn haxe_string_split(
    out: *mut *mut HaxeString,
    out_len: *mut usize,
    s: *const HaxeString,
    delimiter: *const HaxeString,
) {
    debug!("[OLD haxe_string_split] Called!");
    if s.is_null() || delimiter.is_null() {
        unsafe {
            *out = ptr::null_mut();
            *out_len = 0;
        }
        return;
    }

    unsafe {
        let s_ref = &*s;
        let delim_ref = &*delimiter;

        debug!(
            "[OLD split] s.len={}, delimiter.len={}",
            s_ref.len, delim_ref.len
        );

        // Count occurrences
        let mut count = 1;
        let mut pos = 0;
        loop {
            let idx = haxe_string_index_of(s, delimiter, pos);
            if idx < 0 {
                break;
            }
            count += 1;
            pos = (idx as usize) + delim_ref.len;
        }

        // Allocate array of HaxeString
        let layout = Layout::array::<HaxeString>(count).unwrap();
        let array_ptr = alloc(layout) as *mut HaxeString;

        // Fill array
        let mut array_idx = 0;
        let mut start = 0;
        loop {
            let idx = haxe_string_index_of(s, delimiter, start);
            if idx < 0 {
                // Last part
                haxe_string_substring(array_ptr.add(array_idx), s, start, s_ref.len);
                break;
            }

            haxe_string_substring(array_ptr.add(array_idx), s, start, idx as usize);
            array_idx += 1;
            start = (idx as usize) + delim_ref.len;
        }

        *out = array_ptr;
        *out_len = count;
    }
}

/// Split string into an array of strings (returns proper HaxeArray)
/// This is the preferred version that returns Array<String> properly
#[no_mangle]
pub extern "C" fn haxe_string_split_array(
    s: *const HaxeString,
    delimiter: *const HaxeString,
) -> *mut crate::haxe_array::HaxeArray {
    use crate::haxe_array::HaxeArray;

    debug!(
        "[split] Function entry: s={:?}, delimiter={:?}",
        s, delimiter
    );

    if s.is_null() || delimiter.is_null() {
        // Return empty array
        let arr = Box::new(HaxeArray {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
            elem_size: 8, // size of pointer (i64)
        });
        return Box::into_raw(arr);
    }

    unsafe {
        let s_ref = &*s;
        let delim_ref = &*delimiter;

        debug!(
            "[split] s.len={}, delimiter.len={}",
            s_ref.len, delim_ref.len
        );

        // Count occurrences
        let mut count = 1;
        let mut pos = 0;
        loop {
            let idx = haxe_string_index_of(s, delimiter, pos);
            debug!("[split] index_of from pos={} returned idx={}", pos, idx);
            if idx < 0 {
                break;
            }
            count += 1;
            pos = (idx as usize) + delim_ref.len;
        }
        debug!("[split] Final count={}", count);

        // Create HaxeArray to hold string pointers as i64
        let elem_size = 8; // size of pointer
        let total_size = count * elem_size;
        let layout = Layout::from_size_align_unchecked(total_size, 8);
        let data_ptr = alloc(layout);

        if data_ptr.is_null() {
            panic!("Failed to allocate memory for string split array");
        }

        // Fill array with string pointers
        let mut array_idx = 0;
        let mut start = 0;
        let i64_ptr = data_ptr as *mut i64;

        loop {
            let idx = haxe_string_index_of(s, delimiter, start);
            if idx < 0 {
                // Last part - allocate and store substring
                let substring = Box::new(HaxeString {
                    ptr: ptr::null_mut(),
                    len: 0,
                    cap: 0,
                });
                let substr_ptr = Box::into_raw(substring);
                haxe_string_substring(substr_ptr, s, start, s_ref.len);
                *i64_ptr.add(array_idx) = substr_ptr as i64;
                break;
            }

            // Allocate and store substring
            let substring = Box::new(HaxeString {
                ptr: ptr::null_mut(),
                len: 0,
                cap: 0,
            });
            let substr_ptr = Box::into_raw(substring);
            haxe_string_substring(substr_ptr, s, start, idx as usize);
            *i64_ptr.add(array_idx) = substr_ptr as i64;

            array_idx += 1;
            start = (idx as usize) + delim_ref.len;
        }

        // Create and return HaxeArray
        let arr = Box::new(HaxeArray {
            ptr: data_ptr,
            len: count,
            cap: count,
            elem_size: 8,
        });
        let arr_ptr = Box::into_raw(arr);
        debug!(
            "[split] Returning HaxeArray pointer: {:?} (count={})",
            arr_ptr, count
        );
        arr_ptr
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free string memory
#[no_mangle]
pub extern "C" fn haxe_string_free(s: *mut HaxeString) {
    if s.is_null() {
        return;
    }

    unsafe {
        let s_ref = &*s;
        if !s_ref.ptr.is_null() && s_ref.cap > 0 {
            let layout = Layout::from_size_align_unchecked(s_ref.cap, 1);
            dealloc(s_ref.ptr, layout);
        }
    }
}

// ============================================================================
// I/O and Conversion
// ============================================================================

/// Print string to stdout
#[no_mangle]
pub extern "C" fn haxe_string_print(s: *const HaxeString) {
    if s.is_null() {
        return;
    }

    unsafe {
        let s_ref = &*s;
        if s_ref.len > 0 {
            let slice = slice::from_raw_parts(s_ref.ptr, s_ref.len);
            if let Ok(rust_str) = str::from_utf8(slice) {
                print!("{}", rust_str);
            }
        }
    }
}

/// Print string to stdout with newline
#[no_mangle]
pub extern "C" fn haxe_string_println(s: *const HaxeString) {
    haxe_string_print(s);
    println!();
}

/// Replace all occurrences of `needle` in `haystack` with `replacement`.
/// Returns a new HaxeString with the result.
#[no_mangle]
pub extern "C" fn haxe_string_replace(
    haystack: *const HaxeString,
    needle: *const HaxeString,
    replacement: *const HaxeString,
) -> *mut HaxeString {
    let result = Box::new(HaxeString {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    });
    let result_ptr = Box::into_raw(result);
    haxe_string_new(result_ptr);

    if haystack.is_null() || needle.is_null() || replacement.is_null() {
        return result_ptr;
    }

    unsafe {
        let h = &*haystack;
        let n = &*needle;
        let r = &*replacement;

        if h.len == 0 || n.len == 0 || h.ptr.is_null() || n.ptr.is_null() {
            // Copy haystack as-is
            if h.len > 0 && !h.ptr.is_null() {
                let h_slice = slice::from_raw_parts(h.ptr, h.len);
                haxe_string_from_bytes(result_ptr, h_slice.as_ptr(), h_slice.len());
            }
            return result_ptr;
        }

        let h_bytes = slice::from_raw_parts(h.ptr, h.len);
        let n_bytes = slice::from_raw_parts(n.ptr, n.len);
        let r_bytes = if r.len > 0 && !r.ptr.is_null() {
            slice::from_raw_parts(r.ptr, r.len)
        } else {
            &[]
        };

        // Simple search-and-replace
        let h_str = str::from_utf8_unchecked(h_bytes);
        let n_str = str::from_utf8_unchecked(n_bytes);
        let r_str = str::from_utf8_unchecked(r_bytes);
        let replaced = h_str.replace(n_str, r_str);

        haxe_string_from_bytes(result_ptr, replaced.as_ptr(), replaced.len());
        result_ptr
    }
}

/// Get C string pointer (null-terminated)
#[no_mangle]
pub extern "C" fn haxe_string_to_cstr(s: *const HaxeString) -> *const u8 {
    if s.is_null() {
        return ptr::null();
    }
    unsafe { (*s).ptr }
}
