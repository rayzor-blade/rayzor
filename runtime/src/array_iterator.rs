//! Array Iterator runtime support
//!
//! Implements ArrayIterator and ArrayKeyValueIterator for the Haxe iterator protocol.
//! Both are opaque pointer types: Box<Struct> cast to *mut u8.

use crate::anon_object::{rayzor_anon_new, rayzor_anon_set_field_by_index, rayzor_ensure_shape};
use crate::haxe_array::HaxeArray;
use crate::haxe_string::HaxeString;
use std::ptr;
use std::sync::Once;

// ============================================================================
// Internal types
// ============================================================================

struct HaxeArrayIterator {
    array: *mut HaxeArray,
    current: usize,
}

struct HaxeArrayKeyValueIterator {
    array: *mut HaxeArray,
    current: usize,
}

// Shape ID for {key: Int, value: Dynamic} anonymous objects
// Fields alphabetically sorted: key (index 0), value (index 1)
// Type codes: 3 = Int, 7 = Dynamic
const KV_SHAPE_ID: u32 = 1001;

static KV_SHAPE_INIT: Once = Once::new();

/// Create a heap-allocated HaxeString from a Rust &str
fn str_to_hs(s: &str) -> *mut u8 {
    let hs = Box::new(HaxeString {
        ptr: ptr::null_mut(),
        len: 0,
        cap: 0,
    });
    let hs_ptr = Box::into_raw(hs);
    crate::haxe_string::haxe_string_from_bytes(hs_ptr, s.as_ptr(), s.len());
    hs_ptr as *mut u8
}

fn ensure_kv_shape() {
    KV_SHAPE_INIT.call_once(|| {
        let descriptor = str_to_hs("key:3,value:7");
        rayzor_ensure_shape(KV_SHAPE_ID, descriptor);
    });
}

// ============================================================================
// ArrayIterator
// ============================================================================

/// Create a new ArrayIterator from an array pointer.
#[no_mangle]
pub extern "C" fn haxe_array_iterator_new(arr: *mut u8) -> *mut u8 {
    // eprintln!("[DEBUG] haxe_array_iterator_new called, arr={:?}", arr);
    let iter = Box::new(HaxeArrayIterator {
        array: arr as *mut HaxeArray,
        current: 0,
    });
    let ptr = Box::into_raw(iter) as *mut u8;
    // eprintln!("[DEBUG] haxe_array_iterator_new returning {:?}", ptr);
    ptr
}

/// Check if the iterator has more elements.
/// Returns 1 if more elements, 0 otherwise.
#[no_mangle]
pub extern "C" fn haxe_array_iterator_has_next(iter: *mut u8) -> i32 {
    // eprintln!("[DEBUG] haxe_array_iterator_has_next called, iter={:?}", iter);
    if iter.is_null() {
        return 0;
    }
    unsafe {
        let iter = &*(iter as *mut HaxeArrayIterator);
        if iter.array.is_null() {
            return 0;
        }
        let result = if iter.current < (*iter.array).len { 1 } else { 0 };
        // eprintln!("[DEBUG] has_next returning {}", result);
        result
    }
}

/// Get the next element value (raw i64).
/// Advances the iterator position.
#[no_mangle]
pub extern "C" fn haxe_array_iterator_next(iter: *mut u8) -> i64 {
    // eprintln!("[DEBUG] haxe_array_iterator_next called, iter={:?}", iter);
    if iter.is_null() {
        // eprintln!("[DEBUG] iter is null, returning 0");
        return 0;
    }
    unsafe {
        let iter = &mut *(iter as *mut HaxeArrayIterator);
        // eprintln!("[DEBUG] array={:?}, current={}, array.len={}", iter.array, iter.current, if iter.array.is_null() { 0 } else { (*iter.array).len });
        if iter.array.is_null() || iter.current >= (*iter.array).len {
            return 0;
        }
        let arr = &*iter.array;
        // eprintln!("[DEBUG] elem_size={}, ptr={:?}", arr.elem_size, arr.ptr);
        let elem_ptr = arr.ptr.add(iter.current * arr.elem_size);
        let value = *(elem_ptr as *const i64);
        iter.current += 1;
        // eprintln!("[DEBUG] returning value={}", value);
        value
    }
}

// ============================================================================
// ArrayKeyValueIterator
// ============================================================================

/// Create a new ArrayKeyValueIterator from an array pointer.
#[no_mangle]
pub extern "C" fn haxe_array_kv_iterator_new(arr: *mut u8) -> *mut u8 {
    let iter = Box::new(HaxeArrayKeyValueIterator {
        array: arr as *mut HaxeArray,
        current: 0,
    });
    Box::into_raw(iter) as *mut u8
}

/// Check if the KV iterator has more elements.
#[no_mangle]
pub extern "C" fn haxe_array_kv_iterator_has_next(iter: *mut u8) -> i32 {
    if iter.is_null() {
        return 0;
    }
    unsafe {
        let iter = &*(iter as *mut HaxeArrayKeyValueIterator);
        if iter.array.is_null() {
            return 0;
        }
        if iter.current < (*iter.array).len {
            1
        } else {
            0
        }
    }
}

/// Get the next {key: Int, value: Dynamic} anonymous object.
/// Advances the iterator position.
#[no_mangle]
pub extern "C" fn haxe_array_kv_iterator_next(iter: *mut u8) -> *mut u8 {
    ensure_kv_shape();

    if iter.is_null() {
        // Return empty {key: -1, value: 0}
        let handle = rayzor_anon_new(KV_SHAPE_ID, 2);
        rayzor_anon_set_field_by_index(handle, 0, (-1i64) as u64); // key
        rayzor_anon_set_field_by_index(handle, 1, 0u64); // value
        return handle;
    }

    unsafe {
        let iter = &mut *(iter as *mut HaxeArrayKeyValueIterator);
        let key = iter.current as i64;

        let value = if !iter.array.is_null() && iter.current < (*iter.array).len {
            let arr = &*iter.array;
            let elem_ptr = arr.ptr.add(iter.current * arr.elem_size);
            *(elem_ptr as *const i64)
        } else {
            0i64
        };

        iter.current += 1;

        // Create anon object {key: Int, value: Dynamic}
        let handle = rayzor_anon_new(KV_SHAPE_ID, 2);
        rayzor_anon_set_field_by_index(handle, 0, key as u64); // key at index 0
        rayzor_anon_set_field_by_index(handle, 1, value as u64); // value at index 1
        handle
    }
}
