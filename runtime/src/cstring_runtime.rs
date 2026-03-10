//! CString runtime API — null-terminated C string interop.
//!
//! Bridges Haxe's managed String (HaxeString) with C's `char*` convention.
//! Used by `rayzor.CString` extern abstract and `@:cstruct` fields.

use crate::haxe_string::HaxeString;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

/// Convert a HaxeString to a null-terminated C string (char*).
/// Allocates a new buffer, copies data, appends '\0'.
/// Returns the raw address of the buffer as i64 (0 on failure).
///
/// Signature: (source: *const HaxeString) -> i64
#[no_mangle]
pub extern "C" fn rayzor_cstring_from(source: *const HaxeString) -> i64 {
    if source.is_null() {
        return 0;
    }
    unsafe {
        let hs = &*source;
        let len = hs.len;
        let buf_size = len + 1; // +1 for null terminator
        let layout = match Layout::from_size_align(buf_size, 1) {
            Ok(l) => l,
            Err(_) => return 0,
        };
        let buf = alloc(layout);
        if buf.is_null() {
            return 0;
        }
        if len > 0 && !hs.ptr.is_null() {
            ptr::copy_nonoverlapping(hs.ptr, buf, len);
        }
        *buf.add(len) = 0; // null terminator
        buf as i64
    }
}

/// Convert a null-terminated C string (char*) back to a HaxeString.
/// Allocates a new HaxeString, copies data from the C buffer.
/// Returns a pointer to the new HaxeString.
///
/// Signature: (cstr_addr: i64) -> *mut HaxeString
#[no_mangle]
pub extern "C" fn rayzor_cstring_to_string(cstr_addr: i64) -> *mut HaxeString {
    let cstr = cstr_addr as *const u8;
    let hs = Box::new(HaxeString {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    });
    let hs_ptr = Box::into_raw(hs);

    if cstr.is_null() {
        crate::haxe_string::haxe_string_new(hs_ptr);
        return hs_ptr;
    }

    crate::haxe_string::haxe_string_from_cstr(hs_ptr, cstr);
    hs_ptr
}

/// Free a CString buffer allocated by rayzor_cstring_from.
///
/// Signature: (cstr_addr: i64) -> void
#[no_mangle]
pub extern "C" fn rayzor_cstring_free(cstr_addr: i64) {
    let ptr = cstr_addr as *mut u8;
    if ptr.is_null() {
        return;
    }
    unsafe {
        // Find length to compute layout
        let mut len = 0;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        let buf_size = len + 1;
        let layout = match Layout::from_size_align(buf_size, 1) {
            Ok(l) => l,
            Err(_) => return,
        };
        dealloc(ptr, layout);
    }
}
