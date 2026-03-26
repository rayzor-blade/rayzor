//! Rayzor WASM Runtime Library
//!
//! A `cdylib` crate targeting `wasm32-wasi`. Provides `#[no_mangle] extern "C"` functions
//! imported by the WASM backend from the "rayzor" namespace.
//!
//! All pointer parameters and return values are `i32` (WASM32 addresses).
//! No std::io, no threads, no platform FFI. Uses raw WASI fd_write for output.

#![allow(clippy::missing_safety_doc)]

use core::slice;
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr;

// ============================================================================
// WASI Imports
// ============================================================================

#[link(wasm_import_module = "wasi_snapshot_preview1")]
extern "C" {
    fn fd_write(fd: i32, iovs: i32, iovs_len: i32, nwritten: i32) -> i32;
}

/// Write bytes to a WASI file descriptor.
unsafe fn wasi_write(fd: i32, data: *const u8, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }
    // iovec: { buf: i32, buf_len: i32 } — 8 bytes on stack
    // nwritten: i32 — 4 bytes on stack
    // We place them adjacent in a small stack buffer.
    let mut iov_buf: [u32; 2] = [data as u32, len as u32];
    let mut nwritten: u32 = 0;
    fd_write(
        fd,
        iov_buf.as_mut_ptr() as i32,
        1, // one iovec
        &mut nwritten as *mut u32 as i32,
    );
}

// ============================================================================
// Section 1: Memory Allocator
// ============================================================================

/// Internal allocator wrapper — uses std::alloc.
/// The WASM module uses libc's malloc/free (from wasm32-wasip1 dlmalloc).
fn rt_alloc(size: usize) -> i32 {
    if size == 0 {
        return 0;
    }
    unsafe {
        let layout = Layout::from_size_align_unchecked(size, 8);
        let ptr = alloc(layout);
        if ptr.is_null() { 0 } else { ptr as i32 }
    }
}

/// Free — no-op in Phase 1 (bump allocator, no reclamation).
fn _rt_free(_ptr: i32) {
    // Intentional no-op. A proper GC will replace this.
}

/// Reallocate a block to `new_size` bytes.
#[no_mangle]
pub extern "C" fn realloc_block(ptr: i32, old_size: i32, new_size: i32) -> i32 {
    if new_size <= 0 {
        _rt_free(ptr);
        return 0;
    }
    if ptr == 0 {
        return rt_alloc(new_size as usize);
    }
    unsafe {
        let old_layout = Layout::from_size_align_unchecked(old_size as usize, 8);
        let new_ptr = realloc(ptr as *mut u8, old_layout, new_size as usize);
        if new_ptr.is_null() {
            0
        } else {
            new_ptr as i32
        }
    }
}

// ============================================================================
// Section 2: HaxeString — layout: { ptr: i32, len: i32, cap: i32 }
// ============================================================================
//
// In WASM32, usize == u32 == i32 in terms of width. The struct is 12 bytes.
// All string data is UTF-8 encoded and null-terminated for compatibility.

/// Internal: read HaxeString fields from a WASM pointer.
/// Returns (data_ptr, len, cap) as raw u32 values.
#[inline]
unsafe fn read_haxe_string(s: i32) -> (u32, u32, u32) {
    let base = s as *const u32;
    let data_ptr = *base;
    let len = *base.add(1);
    let cap = *base.add(2);
    (data_ptr, len, cap)
}

/// Internal: write HaxeString fields to a WASM pointer.
#[inline]
unsafe fn write_haxe_string(out: i32, data_ptr: u32, len: u32, cap: u32) {
    let base = out as *mut u32;
    *base = data_ptr;
    *base.add(1) = len;
    *base.add(2) = cap;
}

const STRING_INITIAL_CAP: u32 = 32;

/// Allocate string data buffer, returns pointer to data.
unsafe fn alloc_string_data(cap: u32) -> *mut u8 {
    let layout = Layout::from_size_align_unchecked(cap as usize, 1);
    alloc(layout)
}

/// Create a string from raw bytes with known length.
/// Writes the HaxeString struct to `out`.
#[no_mangle]
pub extern "C" fn haxe_string_from_bytes(out: i32, bytes: i32, len: i32) {
    unsafe {
        if bytes == 0 || len <= 0 {
            // Empty string
            let cap = STRING_INITIAL_CAP;
            let data = alloc_string_data(cap);
            if data.is_null() {
                return;
            }
            *data = 0; // null terminator
            write_haxe_string(out, data as u32, 0, cap);
            return;
        }

        let byte_len = len as u32;
        let cap = byte_len.max(STRING_INITIAL_CAP) + 1; // +1 for null terminator
        let data = alloc_string_data(cap);
        if data.is_null() {
            return;
        }

        ptr::copy_nonoverlapping(bytes as *const u8, data, byte_len as usize);
        *data.add(byte_len as usize) = 0; // null terminator

        write_haxe_string(out, data as u32, byte_len, cap);
    }
}

/// Get string length (in bytes).
#[no_mangle]
pub extern "C" fn haxe_string_length(s: i32) -> i32 {
    if s == 0 {
        return 0;
    }
    unsafe {
        let (_, len, _) = read_haxe_string(s);
        len as i32
    }
}

/// Get byte value at index. Returns -1 if out of bounds.
#[no_mangle]
pub extern "C" fn haxe_string_char_at(s: i32, idx: i32) -> i32 {
    if s == 0 || idx < 0 {
        return -1;
    }
    unsafe {
        let (data_ptr, len, _) = read_haxe_string(s);
        let index = idx as u32;
        if index >= len {
            return -1;
        }
        *(data_ptr as *const u8).add(index as usize) as i32
    }
}

/// Get character code at index. Same as char_at for byte-based strings.
#[no_mangle]
pub extern "C" fn haxe_string_char_code_at(s: i32, idx: i32) -> i32 {
    haxe_string_char_at(s, idx)
}

/// Concatenate two strings, write result to `out`.
#[no_mangle]
pub extern "C" fn haxe_string_concat_sret(out: i32, a: i32, b: i32) {
    unsafe {
        let (a_ptr, a_len) = if a == 0 {
            (ptr::null::<u8>(), 0u32)
        } else {
            let (p, l, _) = read_haxe_string(a);
            (p as *const u8, l)
        };

        let (b_ptr, b_len) = if b == 0 {
            (ptr::null::<u8>(), 0u32)
        } else {
            let (p, l, _) = read_haxe_string(b);
            (p as *const u8, l)
        };

        let total_len = a_len + b_len;
        let cap = total_len.max(STRING_INITIAL_CAP) + 1;
        let data = alloc_string_data(cap);
        if data.is_null() {
            return;
        }

        if a_len > 0 && !a_ptr.is_null() {
            ptr::copy_nonoverlapping(a_ptr, data, a_len as usize);
        }
        if b_len > 0 && !b_ptr.is_null() {
            ptr::copy_nonoverlapping(b_ptr, data.add(a_len as usize), b_len as usize);
        }

        *data.add(total_len as usize) = 0; // null terminator
        write_haxe_string(out, data as u32, total_len, cap);
    }
}

/// Extract a substring [start, end).
#[no_mangle]
pub extern "C" fn haxe_string_substring(out: i32, s: i32, start: i32, end: i32) {
    unsafe {
        if s == 0 {
            haxe_string_from_bytes(out, 0, 0);
            return;
        }

        let (data_ptr, len, _) = read_haxe_string(s);
        let actual_start = (start.max(0) as u32).min(len);
        let actual_end = (end.max(0) as u32).min(len);

        if actual_start >= actual_end {
            haxe_string_from_bytes(out, 0, 0);
            return;
        }

        let sub_len = actual_end - actual_start;
        let cap = sub_len.max(STRING_INITIAL_CAP) + 1;
        let data = alloc_string_data(cap);
        if data.is_null() {
            return;
        }

        ptr::copy_nonoverlapping(
            (data_ptr as *const u8).add(actual_start as usize),
            data,
            sub_len as usize,
        );
        *data.add(sub_len as usize) = 0;
        write_haxe_string(out, data as u32, sub_len, cap);
    }
}

/// Compare two strings lexicographically. Returns -1, 0, or 1.
#[no_mangle]
pub extern "C" fn haxe_string_compare(a: i32, b: i32) -> i32 {
    if a == 0 && b == 0 {
        return 0;
    }
    if a == 0 {
        return -1;
    }
    if b == 0 {
        return 1;
    }
    unsafe {
        let (a_ptr, a_len, _) = read_haxe_string(a);
        let (b_ptr, b_len, _) = read_haxe_string(b);

        let a_bytes = slice::from_raw_parts(a_ptr as *const u8, a_len as usize);
        let b_bytes = slice::from_raw_parts(b_ptr as *const u8, b_len as usize);

        match a_bytes.cmp(b_bytes) {
            core::cmp::Ordering::Less => -1,
            core::cmp::Ordering::Equal => 0,
            core::cmp::Ordering::Greater => 1,
        }
    }
}

/// Find substring `sub` in `s` starting at `start`. Returns byte index or -1.
#[no_mangle]
pub extern "C" fn haxe_string_index_of(s: i32, sub: i32, start: i32) -> i32 {
    if s == 0 || sub == 0 {
        return -1;
    }
    unsafe {
        let (s_ptr, s_len, _) = read_haxe_string(s);
        let (sub_ptr, sub_len, _) = read_haxe_string(sub);

        if sub_len == 0 || start < 0 || (start as u32) >= s_len {
            return -1;
        }

        let haystack = slice::from_raw_parts(s_ptr as *const u8, s_len as usize);
        let needle = slice::from_raw_parts(sub_ptr as *const u8, sub_len as usize);
        let start_idx = start as usize;

        if s_len < sub_len {
            return -1;
        }

        let end = (s_len - sub_len) as usize;
        for i in start_idx..=end {
            if &haystack[i..i + sub_len as usize] == needle {
                return i as i32;
            }
        }

        -1
    }
}

/// Print string to WASI stdout (fd 1). No newline.
#[no_mangle]
pub extern "C" fn haxe_string_print(s: i32) {
    if s == 0 {
        return;
    }
    unsafe {
        let (data_ptr, len, _) = read_haxe_string(s);
        if data_ptr != 0 && len > 0 {
            wasi_write(1, data_ptr as *const u8, len as usize);
        }
    }
}

/// Print string to WASI stdout (fd 1) followed by a newline.
#[no_mangle]
pub extern "C" fn haxe_string_println(s: i32) {
    haxe_string_print(s);
    unsafe {
        let newline: u8 = b'\n';
        wasi_write(1, &newline as *const u8, 1);
    }
}

/// FNV-1a hash of string bytes. Returns i32.
#[no_mangle]
pub extern "C" fn haxe_string_hash(s: i32) -> i32 {
    if s == 0 {
        return 0;
    }
    unsafe {
        let (data_ptr, len, _) = read_haxe_string(s);
        if data_ptr == 0 || len == 0 {
            return 0;
        }
        let bytes = slice::from_raw_parts(data_ptr as *const u8, len as usize);
        let mut hash: u32 = 2166136261;
        for &b in bytes {
            hash ^= b as u32;
            hash = hash.wrapping_mul(16777619);
        }
        hash as i32
    }
}

/// Free string data buffer.
#[no_mangle]
pub extern "C" fn haxe_string_free(s: i32) {
    if s == 0 {
        return;
    }
    unsafe {
        let (data_ptr, _, cap) = read_haxe_string(s);
        if data_ptr != 0 && cap > 0 {
            let layout = Layout::from_size_align_unchecked(cap as usize, 1);
            dealloc(data_ptr as *mut u8, layout);
        }
    }
}

/// Trace a HaxeString struct to stdout (with "trace: " prefix and newline).
#[no_mangle]
pub extern "C" fn haxe_trace_string_struct(s: i32) {
    unsafe {
        let prefix = b"trace: ";
        wasi_write(1, prefix.as_ptr(), prefix.len());
    }
    if s == 0 {
        unsafe {
            let null_str = b"null\n";
            wasi_write(1, null_str.as_ptr(), null_str.len());
        }
        return;
    }
    haxe_string_print(s);
    unsafe {
        let newline: u8 = b'\n';
        wasi_write(1, &newline as *const u8, 1);
    }
}

// ============================================================================
// Section 3: Math Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn haxe_math_sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_sin(x: f64) -> f64 {
    libm::sin(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_cos(x: f64) -> f64 {
    libm::cos(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_tan(x: f64) -> f64 {
    libm::tan(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_asin(x: f64) -> f64 {
    libm::asin(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_acos(x: f64) -> f64 {
    libm::acos(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_atan(x: f64) -> f64 {
    libm::atan(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

#[no_mangle]
pub extern "C" fn haxe_math_exp(x: f64) -> f64 {
    libm::exp(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_log(x: f64) -> f64 {
    libm::log(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

#[no_mangle]
pub extern "C" fn haxe_math_floor(x: f64) -> i32 {
    libm::floor(x) as i32
}

#[no_mangle]
pub extern "C" fn haxe_math_ceil(x: f64) -> i32 {
    libm::ceil(x) as i32
}

#[no_mangle]
pub extern "C" fn haxe_math_round(x: f64) -> i32 {
    libm::round(x) as i32
}

#[no_mangle]
pub extern "C" fn haxe_math_abs(x: f64) -> f64 {
    libm::fabs(x)
}

#[no_mangle]
pub extern "C" fn haxe_math_min(a: f64, b: f64) -> f64 {
    libm::fmin(a, b)
}

#[no_mangle]
pub extern "C" fn haxe_math_max(a: f64, b: f64) -> f64 {
    libm::fmax(a, b)
}

/// Simple LCG random number generator. Returns [0.0, 1.0).
#[no_mangle]
pub extern "C" fn haxe_math_random() -> f64 {
    // Static seed — no atomics needed, WASM is single-threaded.
    static mut SEED: u64 = 1;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        ((SEED / 65536) % 32768) as f64 / 32768.0
    }
}

#[no_mangle]
pub extern "C" fn haxe_math_pi() -> f64 {
    core::f64::consts::PI
}

#[no_mangle]
pub extern "C" fn haxe_math_nan() -> f64 {
    f64::NAN
}

#[no_mangle]
pub extern "C" fn haxe_math_positive_infinity() -> f64 {
    f64::INFINITY
}

#[no_mangle]
pub extern "C" fn haxe_math_is_nan(x: f64) -> i32 {
    if x.is_nan() { 1 } else { 0 }
}

#[no_mangle]
pub extern "C" fn haxe_math_is_finite(x: f64) -> i32 {
    if x.is_finite() { 1 } else { 0 }
}

// ============================================================================
// Section 4: Box/Unbox — Dynamic Value Support
// ============================================================================
//
// DynamicValue layout on WASM32: { type_id: u32, value_ptr: u32 } = 8 bytes.
// Type IDs match native runtime: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String.

const TYPE_NULL: u32 = 1;
const TYPE_BOOL: u32 = 2;
const TYPE_INT: u32 = 3;
const TYPE_FLOAT: u32 = 4;

/// Internal: allocate a DynamicValue on the heap (8 bytes).
/// Stores type_id at offset 0, value_ptr at offset 4.
unsafe fn alloc_dynamic(type_id: u32, value_ptr: u32) -> i32 {
    let layout = Layout::from_size_align_unchecked(8, 4);
    let ptr = alloc(layout);
    if ptr.is_null() {
        return 0;
    }
    *(ptr as *mut u32) = type_id;
    *(ptr as *mut u32).add(1) = value_ptr;
    ptr as i32
}

/// Internal: read DynamicValue fields. Returns (type_id, value_ptr).
#[inline]
unsafe fn read_dynamic(ptr: i32) -> (u32, u32) {
    let base = ptr as *const u32;
    (*base, *base.add(1))
}

/// Box an Int as DynamicValue. Allocates 4 bytes for the int value,
/// then 8 bytes for the DynamicValue header.
#[no_mangle]
pub extern "C" fn haxe_box_int_ptr(val: i32) -> i32 {
    unsafe {
        let layout = Layout::from_size_align_unchecked(4, 4);
        let vp = alloc(layout);
        if vp.is_null() {
            return 0;
        }
        *(vp as *mut i32) = val;
        alloc_dynamic(TYPE_INT, vp as u32)
    }
}

/// Box a Float (f64) as DynamicValue. Allocates 8 bytes for the f64 value.
#[no_mangle]
pub extern "C" fn haxe_box_float_ptr(val: f64) -> i32 {
    unsafe {
        let layout = Layout::from_size_align_unchecked(8, 8);
        let vp = alloc(layout);
        if vp.is_null() {
            return 0;
        }
        *(vp as *mut f64) = val;
        alloc_dynamic(TYPE_FLOAT, vp as u32)
    }
}

/// Box a Bool as DynamicValue.
#[no_mangle]
pub extern "C" fn haxe_box_bool_ptr(val: i32) -> i32 {
    unsafe {
        let layout = Layout::from_size_align_unchecked(4, 4);
        let vp = alloc(layout);
        if vp.is_null() {
            return 0;
        }
        *(vp as *mut i32) = val;
        alloc_dynamic(TYPE_BOOL, vp as u32)
    }
}

/// Unbox an Int from DynamicValue pointer.
#[no_mangle]
pub extern "C" fn haxe_unbox_int(ptr: i32) -> i32 {
    if ptr == 0 {
        return 0;
    }
    unsafe {
        let (type_id, value_ptr) = read_dynamic(ptr);
        if type_id == TYPE_INT {
            *(value_ptr as *const i32)
        } else if type_id == TYPE_FLOAT {
            *(value_ptr as *const f64) as i32
        } else if type_id == TYPE_BOOL {
            *(value_ptr as *const i32)
        } else {
            0
        }
    }
}

/// Unbox a Float from DynamicValue pointer.
#[no_mangle]
pub extern "C" fn haxe_unbox_float(ptr: i32) -> f64 {
    if ptr == 0 {
        return 0.0;
    }
    unsafe {
        let (type_id, value_ptr) = read_dynamic(ptr);
        if type_id == TYPE_FLOAT {
            *(value_ptr as *const f64)
        } else if type_id == TYPE_INT {
            *(value_ptr as *const i32) as f64
        } else if type_id == TYPE_BOOL {
            *(value_ptr as *const i32) as f64
        } else {
            0.0
        }
    }
}

/// Unbox a Bool from DynamicValue pointer.
#[no_mangle]
pub extern "C" fn haxe_unbox_bool(ptr: i32) -> i32 {
    if ptr == 0 {
        return 0;
    }
    unsafe {
        let (type_id, value_ptr) = read_dynamic(ptr);
        if type_id == TYPE_BOOL {
            *(value_ptr as *const i32)
        } else {
            0
        }
    }
}

/// Extract the raw pointer from a boxed DynamicValue.
/// For reference types, value_ptr is already the object pointer.
#[no_mangle]
pub extern "C" fn haxe_unbox_reference_ptr(ptr: i32) -> i32 {
    if ptr == 0 {
        return 0;
    }
    // Suspicious low pointer values are not valid heap addresses
    if (ptr as u32) < 0x100 {
        return 0;
    }
    unsafe {
        let (_, value_ptr) = read_dynamic(ptr);
        value_ptr as i32
    }
}

// ============================================================================
// Section 5: Array Basics
// ============================================================================
//
// HaxeArray layout on WASM32: { ptr: u32, len: u32, cap: u32, elem_size: u32 } = 16 bytes.
// For the basic API, elem_size is fixed at 4 (i32 elements).

const ARRAY_INITIAL_CAP: u32 = 8;
const ARRAY_ELEM_SIZE: u32 = 4; // i32 elements for basic array

/// Allocate a new empty array. Returns pointer to HaxeArray struct (i32).
#[no_mangle]
pub extern "C" fn haxe_array_new() -> i32 {
    unsafe {
        // Allocate the HaxeArray header (16 bytes)
        let header_layout = Layout::from_size_align_unchecked(16, 4);
        let header = alloc(header_layout);
        if header.is_null() {
            return 0;
        }

        // Allocate data buffer
        let data_size = (ARRAY_INITIAL_CAP * ARRAY_ELEM_SIZE) as usize;
        let data_layout = Layout::from_size_align_unchecked(data_size, 4);
        let data = alloc(data_layout);
        if data.is_null() {
            dealloc(header, header_layout);
            return 0;
        }

        // Initialize header: { ptr, len, cap, elem_size }
        let h = header as *mut u32;
        *h = data as u32;           // ptr
        *h.add(1) = 0;              // len
        *h.add(2) = ARRAY_INITIAL_CAP; // cap
        *h.add(3) = ARRAY_ELEM_SIZE;   // elem_size

        header as i32
    }
}

/// Internal: read array header. Returns (data_ptr, len, cap, elem_size).
#[inline]
unsafe fn read_array(arr: i32) -> (u32, u32, u32, u32) {
    let h = arr as *const u32;
    (*h, *h.add(1), *h.add(2), *h.add(3))
}

/// Internal: write array header fields.
#[inline]
unsafe fn write_array(arr: i32, data_ptr: u32, len: u32, cap: u32, elem_size: u32) {
    let h = arr as *mut u32;
    *h = data_ptr;
    *h.add(1) = len;
    *h.add(2) = cap;
    *h.add(3) = elem_size;
}

/// Ensure the array has capacity for at least one more element.
/// May reallocate the data buffer.
unsafe fn array_ensure_capacity(arr: i32) {
    let (data_ptr, len, cap, elem_size) = read_array(arr);
    if len < cap {
        return;
    }
    let new_cap = if cap == 0 { ARRAY_INITIAL_CAP } else { cap * 2 };
    let old_size = (cap * elem_size) as usize;
    let new_size = (new_cap * elem_size) as usize;

    let old_layout = Layout::from_size_align_unchecked(old_size, 4);
    let new_data = if data_ptr == 0 {
        alloc(Layout::from_size_align_unchecked(new_size, 4))
    } else {
        realloc(data_ptr as *mut u8, old_layout, new_size)
    };

    if !new_data.is_null() {
        let h = arr as *mut u32;
        *h = new_data as u32;    // ptr
        *h.add(2) = new_cap;    // cap
    }
}

/// Push an i32 value onto the array.
#[no_mangle]
pub extern "C" fn haxe_array_push_i64(arr: i32, val: i32) {
    if arr == 0 {
        return;
    }
    unsafe {
        array_ensure_capacity(arr);
        let (data_ptr, len, _, _) = read_array(arr);
        let slot = (data_ptr as *mut i32).add(len as usize);
        *slot = val;
        // Update len
        let h = arr as *mut u32;
        *h.add(1) = len + 1;
    }
}

/// Get an i32 element at index. Returns 0 if out of bounds.
#[no_mangle]
pub extern "C" fn haxe_array_get_i64(arr: i32, idx: i32) -> i32 {
    if arr == 0 || idx < 0 {
        return 0;
    }
    unsafe {
        let (data_ptr, len, _, _) = read_array(arr);
        let index = idx as u32;
        if index >= len {
            return 0;
        }
        *(data_ptr as *const i32).add(index as usize)
    }
}

/// Get array length.
#[no_mangle]
pub extern "C" fn haxe_array_length(arr: i32) -> i32 {
    if arr == 0 {
        return 0;
    }
    unsafe {
        let (_, len, _, _) = read_array(arr);
        len as i32
    }
}

// ============================================================================
// Section 6: Type System Stubs
// ============================================================================

/// Read the runtime type_id from an object's header (first 4 bytes at offset 0).
/// In WASM32, type_id is stored as i32 (not i64 like native).
#[no_mangle]
pub extern "C" fn haxe_object_get_type_id(ptr: i32) -> i32 {
    if ptr == 0 {
        return -1;
    }
    unsafe { *(ptr as *const i32) }
}

/// Check if an object is an instance of a target type.
/// Simplified: only checks direct type_id match (no hierarchy walk).
#[no_mangle]
pub extern "C" fn haxe_object_is_instance(ptr: i32, type_id: i32) -> i32 {
    if ptr == 0 {
        return 0;
    }
    let actual = haxe_object_get_type_id(ptr);
    if actual == type_id { 1 } else { 0 }
}

/// Allocate an anonymous object with `n_fields` slots.
/// Each slot is 4 bytes (i32). Returns pointer to the data area.
#[no_mangle]
pub extern "C" fn haxe_anon_new(n_fields: i32) -> i32 {
    if n_fields <= 0 {
        return rt_alloc(4); // minimum allocation
    }
    let size = (n_fields as u32) * 4;
    unsafe {
        let layout = Layout::from_size_align_unchecked(size as usize, 4);
        let ptr = alloc(layout);
        if ptr.is_null() {
            return 0;
        }
        // Zero-initialize
        ptr::write_bytes(ptr, 0, size as usize);
        ptr as i32
    }
}

// ============================================================================
// Section 7: WASI I/O Helpers
// ============================================================================

/// Raw trace: print bytes + newline to stdout via WASI fd_write.
#[no_mangle]
pub extern "C" fn haxe_trace_string(data: i32, len: i32) {
    unsafe {
        let prefix = b"trace: ";
        wasi_write(1, prefix.as_ptr(), prefix.len());
        if data != 0 && len > 0 {
            wasi_write(1, data as *const u8, len as usize);
        } else {
            let null_str = b"null";
            wasi_write(1, null_str.as_ptr(), null_str.len());
        }
        let newline: u8 = b'\n';
        wasi_write(1, &newline as *const u8, 1);
    }
}

// ============================================================================
// Section 8: Debug / No-op Stubs
// ============================================================================

/// No-op: call frame location tracking (used by debug stack traces).
#[no_mangle]
pub extern "C" fn rayzor_update_call_frame_location(_line: i32, _col: i32) {
    // no-op in WASM
}

/// Throw a typed exception. In WASM, this traps (unreachable).
#[no_mangle]
pub extern "C" fn rayzor_throw_typed(_ptr: i32) {
    #[cfg(target_arch = "wasm32")]
    {
        core::arch::wasm32::unreachable();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::process::abort();
    }
}

/// No-op: JIT cleanup (not applicable to AOT WASM).
#[no_mangle]
pub extern "C" fn rayzor_jit_cleanup() {
    // no-op
}

/// No-op: thread synchronization (WASM is single-threaded).
#[no_mangle]
pub extern "C" fn rayzor_wait_all_threads() {
    // no-op
}

/// Int-to-string conversion. Writes result to `out` HaxeString.
#[no_mangle]
pub extern "C" fn haxe_int_to_string(out: i32, value: i32) {
    let mut buf = [0u8; 12]; // max "-2147483648" = 11 chars + null
    let s = int_to_buf(value, &mut buf);
    haxe_string_from_bytes(out, s.as_ptr() as i32, s.len() as i32);
}

/// Internal: format an i32 into a byte buffer, return the used slice.
fn int_to_buf(mut value: i32, buf: &mut [u8; 12]) -> &[u8] {
    if value == 0 {
        buf[0] = b'0';
        return &buf[..1];
    }
    let negative = value < 0;
    if negative {
        value = -value;
    }
    let mut pos = 12;
    while value > 0 {
        pos -= 1;
        buf[pos] = b'0' + (value % 10) as u8;
        value /= 10;
    }
    if negative {
        pos -= 1;
        buf[pos] = b'-';
    }
    &buf[pos..]
}

/// Float-to-string conversion. Writes result to `out` HaxeString.
#[no_mangle]
pub extern "C" fn haxe_float_to_string(out: i32, value: f64) {
    // Simple float formatting without std::fmt
    let mut buf = [0u8; 32];
    let len = float_to_buf(value, &mut buf);
    haxe_string_from_bytes(out, buf.as_ptr() as i32, len as i32);
}

/// Internal: format an f64 into a byte buffer. Returns number of bytes written.
fn float_to_buf(value: f64, buf: &mut [u8; 32]) -> usize {
    if value.is_nan() {
        let s = b"NaN";
        buf[..3].copy_from_slice(s);
        return 3;
    }
    if value == f64::INFINITY {
        let s = b"Infinity";
        buf[..8].copy_from_slice(s);
        return 8;
    }
    if value == f64::NEG_INFINITY {
        let s = b"-Infinity";
        buf[..9].copy_from_slice(s);
        return 9;
    }

    let mut pos = 0;
    let mut v = value;

    if v < 0.0 {
        buf[pos] = b'-';
        pos += 1;
        v = -v;
    }

    let int_part = v as u64;
    let frac_part = v - int_part as f64;

    // Write integer part
    if int_part == 0 {
        buf[pos] = b'0';
        pos += 1;
    } else {
        let mut digits = [0u8; 20];
        let mut dpos = 20;
        let mut ip = int_part;
        while ip > 0 {
            dpos -= 1;
            digits[dpos] = b'0' + (ip % 10) as u8;
            ip /= 10;
        }
        let dlen = 20 - dpos;
        buf[pos..pos + dlen].copy_from_slice(&digits[dpos..]);
        pos += dlen;
    }

    // Write fractional part (up to 6 decimal places, trim trailing zeros)
    if frac_part > 0.0 {
        buf[pos] = b'.';
        pos += 1;
        let mut frac = frac_part;
        let mut last_nonzero = pos;
        for _ in 0..6 {
            frac *= 10.0;
            let digit = frac as u8;
            buf[pos] = b'0' + digit;
            if digit != 0 {
                last_nonzero = pos;
            }
            pos += 1;
            frac -= digit as f64;
        }
        pos = last_nonzero + 1;
    }

    pos
}
