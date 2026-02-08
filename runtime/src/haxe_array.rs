//! Haxe Array runtime implementation
//!
//! Generic dynamic array supporting any element type
//! Memory layout: [length: usize, capacity: usize, elements...]

use crate::haxe_string::HaxeString;
use log::debug;
use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr;

/// Haxe Array representation (generic via element size)
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HaxeArray {
    pub ptr: *mut u8,     // Pointer to array data
    pub len: usize,       // Number of elements
    pub cap: usize,       // Capacity (number of elements)
    pub elem_size: usize, // Size of each element in bytes
}

const INITIAL_CAPACITY: usize = 8;

// ============================================================================
// Array Creation
// ============================================================================

/// Create a new empty array
#[no_mangle]
pub extern "C" fn haxe_array_new(out: *mut HaxeArray, elem_size: usize) {
    unsafe {
        let total_size = INITIAL_CAPACITY * elem_size;
        let layout = Layout::from_size_align_unchecked(total_size, 8);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for Array");
        }

        (*out).ptr = ptr;
        (*out).len = 0;
        (*out).cap = INITIAL_CAPACITY;
        (*out).elem_size = elem_size;
    }
}

/// Create array from existing elements
#[no_mangle]
pub extern "C" fn haxe_array_from_elements(
    out: *mut HaxeArray,
    elements: *const u8,
    count: usize,
    elem_size: usize,
) {
    if count == 0 {
        haxe_array_new(out, elem_size);
        return;
    }

    unsafe {
        let cap = count.max(INITIAL_CAPACITY);
        let total_size = cap * elem_size;
        let layout = Layout::from_size_align_unchecked(total_size, 8);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for Array");
        }

        // Copy elements
        ptr::copy_nonoverlapping(elements, ptr, count * elem_size);

        (*out).ptr = ptr;
        (*out).len = count;
        (*out).cap = cap;
        (*out).elem_size = elem_size;
    }
}

// ============================================================================
// Array Properties
// ============================================================================

/// Get array length
#[no_mangle]
pub extern "C" fn haxe_array_length(arr: *const HaxeArray) -> usize {
    debug!("[haxe_array_length] Called with arr={:?}", arr);
    if arr.is_null() {
        debug!("[haxe_array_length] arr is null, returning 0");
        return 0;
    }
    unsafe {
        let arr_ref = &*arr;
        debug!(
            "[haxe_array_length] arr.len={}, arr.cap={}, arr.elem_size={}, arr.ptr={:?}",
            arr_ref.len, arr_ref.cap, arr_ref.elem_size, arr_ref.ptr
        );

        // If it's an array of pointers (elem_size=8), print first few elements
        if arr_ref.elem_size == 8 && arr_ref.len > 0 && !arr_ref.ptr.is_null() {
            debug!("[haxe_array_length] First few i64 values:");
            let i64_ptr = arr_ref.ptr as *const i64;
            for i in 0..arr_ref.len.min(5) {
                let val = *i64_ptr.add(i);
                debug!("  [{}] = 0x{:x} ({})", i, val, val);
            }
        }

        arr_ref.len
    }
}

// ============================================================================
// Element Access
// ============================================================================

/// Get element at index (copies to out buffer)
#[no_mangle]
pub extern "C" fn haxe_array_get(arr: *const HaxeArray, index: usize, out: *mut u8) -> bool {
    if arr.is_null() || out.is_null() {
        return false;
    }

    unsafe {
        let arr_ref = &*arr;
        if index >= arr_ref.len {
            return false;
        }

        let elem_ptr = arr_ref.ptr.add(index * arr_ref.elem_size);
        ptr::copy_nonoverlapping(elem_ptr, out, arr_ref.elem_size);
        true
    }
}

/// Set element at index (copies from data buffer)
/// Auto-expands array if index is beyond current length (Haxe semantics)
/// If data is null, stores zeros (null) at the index
#[no_mangle]
pub extern "C" fn haxe_array_set(arr: *mut HaxeArray, index: usize, data: *const u8) -> bool {
    debug!(
        "[haxe_array_set] Called with arr={:?}, index={}, data={:?}",
        arr, index, data
    );
    if arr.is_null() {
        debug!("[haxe_array_set] arr is null, returning false");
        return false;
    }

    // Allow null data - will store zeros (null) at the index
    let store_null = data.is_null();

    unsafe {
        let arr_ref = &mut *arr;
        debug!(
            "[haxe_array_set] arr.len={}, arr.elem_size={}",
            arr_ref.len, arr_ref.elem_size
        );

        // Auto-expand if index is beyond current length
        let new_len = index + 1;
        if new_len > arr_ref.len {
            // Ensure we have enough capacity
            if new_len > arr_ref.cap {
                let mut new_cap = if arr_ref.cap == 0 {
                    INITIAL_CAPACITY
                } else {
                    arr_ref.cap
                };
                while new_cap < new_len {
                    new_cap *= 2;
                }

                let new_size = new_cap * arr_ref.elem_size;
                let new_ptr = if arr_ref.ptr.is_null() || arr_ref.cap == 0 {
                    let layout = Layout::from_size_align_unchecked(new_size, 8);
                    alloc(layout)
                } else {
                    let old_size = arr_ref.cap * arr_ref.elem_size;
                    let old_layout = Layout::from_size_align_unchecked(old_size, 8);
                    realloc(arr_ref.ptr, old_layout, new_size)
                };

                if new_ptr.is_null() {
                    debug!("[haxe_array_set] Failed to allocate memory");
                    return false;
                }

                arr_ref.ptr = new_ptr;
                arr_ref.cap = new_cap;
            }

            // Zero-initialize the new elements between old len and index
            if arr_ref.len < new_len {
                let start_offset = arr_ref.len * arr_ref.elem_size;
                let zero_bytes = (new_len - arr_ref.len) * arr_ref.elem_size;
                ptr::write_bytes(arr_ref.ptr.add(start_offset), 0, zero_bytes);
            }

            arr_ref.len = new_len;
            debug!(
                "[haxe_array_set] Auto-expanded array to len={}, cap={}",
                arr_ref.len, arr_ref.cap
            );
        }

        let elem_ptr = arr_ref.ptr.add(index * arr_ref.elem_size);
        if store_null {
            // Store zeros (null) at the index
            debug!(
                "[haxe_array_set] Storing null (zeros) at {:?}, {} bytes",
                elem_ptr, arr_ref.elem_size
            );
            ptr::write_bytes(elem_ptr, 0, arr_ref.elem_size);
        } else {
            debug!(
                "[haxe_array_set] Copying {} bytes from {:?} to {:?}",
                arr_ref.elem_size, data, elem_ptr
            );
            ptr::copy_nonoverlapping(data, elem_ptr, arr_ref.elem_size);
        }
        debug!("[haxe_array_set] Successfully set element, returning true");
        true
    }
}

/// Set element at index by value (i64) - avoids boxing overhead
/// Auto-expands array if index is beyond current length (Haxe semantics)
#[no_mangle]
pub extern "C" fn haxe_array_set_i64(arr: *mut HaxeArray, index: usize, value: i64) -> bool {
    if arr.is_null() {
        return false;
    }
    unsafe {
        let arr_ref = &mut *arr;

        // Auto-expand if needed
        let new_len = index + 1;
        if new_len > arr_ref.len {
            if new_len > arr_ref.cap {
                let mut new_cap = if arr_ref.cap == 0 {
                    INITIAL_CAPACITY
                } else {
                    arr_ref.cap
                };
                while new_cap < new_len {
                    new_cap *= 2;
                }
                let new_size = new_cap * arr_ref.elem_size;
                let new_ptr = if arr_ref.ptr.is_null() || arr_ref.cap == 0 {
                    let layout = Layout::from_size_align_unchecked(new_size, 8);
                    alloc(layout)
                } else {
                    let old_size = arr_ref.cap * arr_ref.elem_size;
                    let old_layout = Layout::from_size_align_unchecked(old_size, 8);
                    realloc(arr_ref.ptr, old_layout, new_size)
                };
                if new_ptr.is_null() {
                    return false;
                }
                arr_ref.ptr = new_ptr;
                arr_ref.cap = new_cap;
            }
            // Zero-fill gap
            if arr_ref.len < new_len {
                let start = arr_ref.len * arr_ref.elem_size;
                let bytes = (new_len - arr_ref.len) * arr_ref.elem_size;
                ptr::write_bytes(arr_ref.ptr.add(start), 0, bytes);
            }
            arr_ref.len = new_len;
        }

        let elem_ptr = arr_ref.ptr.add(index * arr_ref.elem_size) as *mut i64;
        *elem_ptr = value;
        true
    }
}

/// Set element at index by value (f64) - avoids boxing overhead
/// Auto-expands array if index is beyond current length (Haxe semantics)
#[no_mangle]
pub extern "C" fn haxe_array_set_f64(arr: *mut HaxeArray, index: usize, value: f64) -> bool {
    if arr.is_null() {
        return false;
    }
    unsafe {
        let arr_ref = &mut *arr;

        // Auto-expand if needed
        let new_len = index + 1;
        if new_len > arr_ref.len {
            if new_len > arr_ref.cap {
                let mut new_cap = if arr_ref.cap == 0 {
                    INITIAL_CAPACITY
                } else {
                    arr_ref.cap
                };
                while new_cap < new_len {
                    new_cap *= 2;
                }
                let new_size = new_cap * arr_ref.elem_size;
                let new_ptr = if arr_ref.ptr.is_null() || arr_ref.cap == 0 {
                    let layout = Layout::from_size_align_unchecked(new_size, 8);
                    alloc(layout)
                } else {
                    let old_size = arr_ref.cap * arr_ref.elem_size;
                    let old_layout = Layout::from_size_align_unchecked(old_size, 8);
                    realloc(arr_ref.ptr, old_layout, new_size)
                };
                if new_ptr.is_null() {
                    return false;
                }
                arr_ref.ptr = new_ptr;
                arr_ref.cap = new_cap;
            }
            if arr_ref.len < new_len {
                let start = arr_ref.len * arr_ref.elem_size;
                let bytes = (new_len - arr_ref.len) * arr_ref.elem_size;
                ptr::write_bytes(arr_ref.ptr.add(start), 0, bytes);
            }
            arr_ref.len = new_len;
        }

        let elem_ptr = arr_ref.ptr.add(index * arr_ref.elem_size) as *mut f64;
        *elem_ptr = value;
        true
    }
}

/// Set element at index to null (store zeros)
/// Auto-expands array if index is beyond current length (Haxe semantics)
#[no_mangle]
pub extern "C" fn haxe_array_set_null(arr: *mut HaxeArray, index: usize) -> bool {
    if arr.is_null() {
        return false;
    }
    unsafe {
        let arr_ref = &mut *arr;

        // Auto-expand if needed
        let new_len = index + 1;
        if new_len > arr_ref.len {
            if new_len > arr_ref.cap {
                let mut new_cap = if arr_ref.cap == 0 {
                    INITIAL_CAPACITY
                } else {
                    arr_ref.cap
                };
                while new_cap < new_len {
                    new_cap *= 2;
                }
                let new_size = new_cap * arr_ref.elem_size;
                let new_ptr = if arr_ref.ptr.is_null() || arr_ref.cap == 0 {
                    let layout = Layout::from_size_align_unchecked(new_size, 8);
                    alloc(layout)
                } else {
                    let old_size = arr_ref.cap * arr_ref.elem_size;
                    let old_layout = Layout::from_size_align_unchecked(old_size, 8);
                    realloc(arr_ref.ptr, old_layout, new_size)
                };
                if new_ptr.is_null() {
                    return false;
                }
                arr_ref.ptr = new_ptr;
                arr_ref.cap = new_cap;
            }
            if arr_ref.len < new_len {
                let start = arr_ref.len * arr_ref.elem_size;
                let bytes = (new_len - arr_ref.len) * arr_ref.elem_size;
                ptr::write_bytes(arr_ref.ptr.add(start), 0, bytes);
            }
            arr_ref.len = new_len;
        }

        // Write zeros for null
        let elem_ptr = arr_ref.ptr.add(index * arr_ref.elem_size);
        ptr::write_bytes(elem_ptr, 0, arr_ref.elem_size);
        true
    }
}

/// Get pointer to element (for direct access)
#[no_mangle]
pub extern "C" fn haxe_array_get_ptr(arr: *const HaxeArray, index: usize) -> *mut u8 {
    debug!(
        "[haxe_array_get_ptr] Called with arr={:?}, index={}",
        arr, index
    );
    if arr.is_null() {
        debug!("[haxe_array_get_ptr] arr is null, returning null");
        return ptr::null_mut();
    }

    unsafe {
        let arr_ref = &*arr;
        debug!(
            "[haxe_array_get_ptr] arr.len={}, arr.elem_size={}",
            arr_ref.len, arr_ref.elem_size
        );
        if index >= arr_ref.len {
            debug!(
                "[haxe_array_get_ptr] index {} >= len {}, returning null",
                index, arr_ref.len
            );
            return ptr::null_mut();
        }

        let elem_ptr = arr_ref.ptr.add(index * arr_ref.elem_size);
        debug!("[haxe_array_get_ptr] Returning elem_ptr={:?}", elem_ptr);
        elem_ptr
    }
}

// ============================================================================
// Array Modification
// ============================================================================

/// Push element onto array
#[no_mangle]
pub extern "C" fn haxe_array_push(arr: *mut HaxeArray, data: *const u8) {
    debug!(
        "[haxe_array_push] Called with arr={:?}, data={:?}",
        arr, data
    );
    if arr.is_null() || data.is_null() {
        debug!("[haxe_array_push] arr or data is null, returning");
        return;
    }

    unsafe {
        let arr_ref = &mut *arr;
        debug!(
            "[haxe_array_push] Before push: len={}, cap={}, elem_size={}",
            arr_ref.len, arr_ref.cap, arr_ref.elem_size
        );

        // Check if we need to grow
        if arr_ref.len >= arr_ref.cap {
            let new_cap = if arr_ref.cap == 0 {
                INITIAL_CAPACITY
            } else {
                arr_ref.cap * 2
            };

            let new_size = new_cap * arr_ref.elem_size;

            let new_ptr = if arr_ref.ptr.is_null() || arr_ref.cap == 0 {
                // First allocation - use alloc instead of realloc
                let layout = Layout::from_size_align_unchecked(new_size, 8);
                alloc(layout)
            } else {
                // Grow existing allocation
                let old_size = arr_ref.cap * arr_ref.elem_size;
                let old_layout = Layout::from_size_align_unchecked(old_size, 8);
                realloc(arr_ref.ptr, old_layout, new_size)
            };

            if new_ptr.is_null() {
                panic!("Failed to allocate/reallocate memory for Array");
            }

            arr_ref.ptr = new_ptr;
            arr_ref.cap = new_cap;
        }

        // Add element
        let elem_ptr = arr_ref.ptr.add(arr_ref.len * arr_ref.elem_size);
        ptr::copy_nonoverlapping(data, elem_ptr, arr_ref.elem_size);
        arr_ref.len += 1;
        debug!(
            "[haxe_array_push] After push: len={}, element added successfully",
            arr_ref.len
        );
    }
}

/// Pop element from array (original version with out param)
#[no_mangle]
pub extern "C" fn haxe_array_pop(arr: *mut HaxeArray, out: *mut u8) -> bool {
    if arr.is_null() {
        return false;
    }

    unsafe {
        let arr_ref = &mut *arr;
        if arr_ref.len == 0 {
            return false;
        }

        arr_ref.len -= 1;

        if !out.is_null() {
            let elem_ptr = arr_ref.ptr.add(arr_ref.len * arr_ref.elem_size);
            ptr::copy_nonoverlapping(elem_ptr, out, arr_ref.elem_size);
        }

        true
    }
}

/// Pop element from array and return it as i64 (for Array<Int>)
/// Returns 0 if array is empty (Haxe's Null<Int> semantics)
#[no_mangle]
pub extern "C" fn haxe_array_pop_i64(arr: *mut HaxeArray) -> i64 {
    if arr.is_null() {
        return 0;
    }

    unsafe {
        let arr_ref = &mut *arr;
        if arr_ref.len == 0 {
            return 0; // Null<Int> -> 0
        }

        arr_ref.len -= 1;

        // Get pointer to the element we're popping
        let elem_ptr = arr_ref.ptr.add(arr_ref.len * arr_ref.elem_size);

        // Read as i64 (elem_size should be 8 for Int arrays)
        if arr_ref.elem_size == 8 {
            *(elem_ptr as *const i64)
        } else if arr_ref.elem_size == 4 {
            *(elem_ptr as *const i32) as i64
        } else {
            0
        }
    }
}

/// Pop element from array and return it as a boxed Dynamic value
/// Returns null if array is empty
/// The returned pointer is a DynamicValue* suitable for haxe_trace_any
#[no_mangle]
pub extern "C" fn haxe_array_pop_ptr(arr: *mut HaxeArray) -> *mut u8 {
    if arr.is_null() {
        return ptr::null_mut();
    }

    unsafe {
        let arr_ref = &mut *arr;
        if arr_ref.len == 0 {
            return ptr::null_mut();
        }

        arr_ref.len -= 1;

        // Get pointer to the element we're popping
        let elem_ptr = arr_ref.ptr.add(arr_ref.len * arr_ref.elem_size);

        // Box the value as a DynamicValue so it can be used with trace() and other dynamic operations
        if arr_ref.elem_size == 8 {
            let value = *(elem_ptr as *const i64);
            // Use haxe_box_int_ptr to create a proper DynamicValue*
            crate::type_system::haxe_box_int_ptr(value)
        } else if arr_ref.elem_size == 4 {
            let value = *(elem_ptr as *const i32);
            crate::type_system::haxe_box_int_ptr(value as i64)
        } else {
            // For other sizes (objects, etc.), the value is already a pointer
            // Return the pointer directly - caller must handle boxing if needed
            *(elem_ptr as *const *mut u8)
        }
    }
}

/// Insert element at index
#[no_mangle]
pub extern "C" fn haxe_array_insert(arr: *mut HaxeArray, index: i32, data: *const u8) {
    if arr.is_null() || data.is_null() {
        return;
    }

    unsafe {
        let arr_ref = &mut *arr;
        let insert_pos = (index.max(0) as usize).min(arr_ref.len);

        // Ensure capacity
        if arr_ref.len >= arr_ref.cap {
            let new_cap = arr_ref.cap * 2;
            let old_size = arr_ref.cap * arr_ref.elem_size;
            let new_size = new_cap * arr_ref.elem_size;

            let old_layout = Layout::from_size_align_unchecked(old_size, 8);
            let new_ptr = realloc(arr_ref.ptr, old_layout, new_size);

            if new_ptr.is_null() {
                panic!("Failed to reallocate memory for Array");
            }

            arr_ref.ptr = new_ptr;
            arr_ref.cap = new_cap;
        }

        // Shift elements to the right
        if insert_pos < arr_ref.len {
            let src = arr_ref.ptr.add(insert_pos * arr_ref.elem_size);
            let dst = src.add(arr_ref.elem_size);
            let count = (arr_ref.len - insert_pos) * arr_ref.elem_size;
            ptr::copy(src, dst, count);
        }

        // Insert new element
        let elem_ptr = arr_ref.ptr.add(insert_pos * arr_ref.elem_size);
        ptr::copy_nonoverlapping(data, elem_ptr, arr_ref.elem_size);
        arr_ref.len += 1;
    }
}

/// Remove element at index
#[no_mangle]
pub extern "C" fn haxe_array_remove(arr: *mut HaxeArray, index: usize) -> bool {
    if arr.is_null() {
        return false;
    }

    unsafe {
        let arr_ref = &mut *arr;
        if index >= arr_ref.len {
            return false;
        }

        // Shift elements to the left
        if index < arr_ref.len - 1 {
            let src = arr_ref.ptr.add((index + 1) * arr_ref.elem_size);
            let dst = arr_ref.ptr.add(index * arr_ref.elem_size);
            let count = (arr_ref.len - index - 1) * arr_ref.elem_size;
            ptr::copy(src, dst, count);
        }

        arr_ref.len -= 1;
        true
    }
}

/// Reverse array in place
#[no_mangle]
pub extern "C" fn haxe_array_reverse(arr: *mut HaxeArray) {
    if arr.is_null() {
        return;
    }

    unsafe {
        let arr_ref = &mut *arr;
        if arr_ref.len <= 1 {
            return;
        }

        let elem_size = arr_ref.elem_size;
        let mut i = 0;
        let mut j = arr_ref.len - 1;

        // Allocate temp buffer for swapping
        let temp_layout = Layout::from_size_align_unchecked(elem_size, 8);
        let temp = alloc(temp_layout);

        while i < j {
            let left = arr_ref.ptr.add(i * elem_size);
            let right = arr_ref.ptr.add(j * elem_size);

            // Swap via temp buffer
            ptr::copy_nonoverlapping(left, temp, elem_size);
            ptr::copy_nonoverlapping(right, left, elem_size);
            ptr::copy_nonoverlapping(temp, right, elem_size);

            i += 1;
            j -= 1;
        }

        dealloc(temp, temp_layout);
    }
}

/// Copy array
#[no_mangle]
pub extern "C" fn haxe_array_copy(out: *mut HaxeArray, arr: *const HaxeArray) {
    if arr.is_null() {
        return;
    }

    unsafe {
        let arr_ref = &*arr;
        haxe_array_from_elements(out, arr_ref.ptr, arr_ref.len, arr_ref.elem_size);
    }
}

/// Slice array
#[no_mangle]
pub extern "C" fn haxe_array_slice(
    out: *mut HaxeArray,
    arr: *const HaxeArray,
    start: usize,
    end: usize,
) {
    debug!(
        "[haxe_array_slice] Called with out={:?}, arr={:?}, start={}, end={}",
        out, arr, start, end
    );
    if arr.is_null() {
        debug!("[haxe_array_slice] arr is null, returning");
        return;
    }

    unsafe {
        let arr_ref = &*arr;
        debug!(
            "[haxe_array_slice] arr.len={}, arr.cap={}, arr.elem_size={}",
            arr_ref.len, arr_ref.cap, arr_ref.elem_size
        );
        let actual_start = start.min(arr_ref.len);
        let actual_end = end.min(arr_ref.len);
        debug!(
            "[haxe_array_slice] actual_start={}, actual_end={}",
            actual_start, actual_end
        );

        if actual_start >= actual_end {
            debug!("[haxe_array_slice] Creating empty array (start >= end)");
            haxe_array_new(out, arr_ref.elem_size);
            return;
        }

        let count = actual_end - actual_start;
        let start_ptr = arr_ref.ptr.add(actual_start * arr_ref.elem_size);
        debug!(
            "[haxe_array_slice] Copying {} elements from offset {}",
            count, actual_start
        );
        haxe_array_from_elements(out, start_ptr, count, arr_ref.elem_size);
        debug!("[haxe_array_slice] Done");
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free array memory
#[no_mangle]
pub extern "C" fn haxe_array_free(arr: *mut HaxeArray) {
    if arr.is_null() {
        return;
    }

    unsafe {
        let arr_ref = &*arr;
        if !arr_ref.ptr.is_null() && arr_ref.cap > 0 {
            let total_size = arr_ref.cap * arr_ref.elem_size;
            let layout = Layout::from_size_align_unchecked(total_size, 8);
            dealloc(arr_ref.ptr, layout);
        }
    }
}

// ============================================================================
// Specialized Integer Array Functions
// ============================================================================

/// Push i32 onto array
#[no_mangle]
pub extern "C" fn haxe_array_push_i32(arr: *mut HaxeArray, value: i32) {
    haxe_array_push(arr, &value as *const i32 as *const u8);
}

/// Get i32 from array
#[no_mangle]
pub extern "C" fn haxe_array_get_i32(arr: *const HaxeArray, index: usize) -> i32 {
    let mut value: i32 = 0;
    if haxe_array_get(arr, index, &mut value as *mut i32 as *mut u8) {
        value
    } else {
        0
    }
}

/// Push i64 onto array
#[no_mangle]
pub extern "C" fn haxe_array_push_i64(arr: *mut HaxeArray, value: i64) {
    haxe_array_push(arr, &value as *const i64 as *const u8);
}

/// Get i64 from array
#[no_mangle]
pub extern "C" fn haxe_array_get_i64(arr: *const HaxeArray, index: usize) -> i64 {
    let mut value: i64 = 0;
    if haxe_array_get(arr, index, &mut value as *mut i64 as *mut u8) {
        value
    } else {
        0
    }
}

/// Push f64 onto array
#[no_mangle]
pub extern "C" fn haxe_array_push_f64(arr: *mut HaxeArray, value: f64) {
    haxe_array_push(arr, &value as *const f64 as *const u8);
}

/// Get f64 from array
#[no_mangle]
pub extern "C" fn haxe_array_get_f64(arr: *const HaxeArray, index: usize) -> f64 {
    let mut value: f64 = 0.0;
    if haxe_array_get(arr, index, &mut value as *mut f64 as *mut u8) {
        value
    } else {
        0.0
    }
}

// ============================================================================
// Array Join (for string arrays)
// ============================================================================

/// Join array elements with a separator, returning a new string
/// arr: pointer to array of HaxeString pointers
/// sep: separator string
/// Returns: pointer to a new HaxeString (caller should manage memory)
#[no_mangle]
pub extern "C" fn haxe_array_join(
    arr: *const HaxeArray,
    sep: *const HaxeString,
) -> *mut HaxeString {
    unsafe {
        // Allocate result string
        let result_layout = Layout::new::<HaxeString>();
        let result_ptr = alloc(result_layout) as *mut HaxeString;
        if result_ptr.is_null() {
            panic!("Failed to allocate HaxeString for join result");
        }

        if arr.is_null() {
            // Empty array -> empty string
            crate::haxe_string::haxe_string_new(result_ptr);
            return result_ptr;
        }

        let arr_ref = &*arr;

        if arr_ref.len == 0 {
            // Empty array -> empty string
            crate::haxe_string::haxe_string_new(result_ptr);
            return result_ptr;
        }

        // Get separator string data
        let (sep_ptr, sep_len) = if sep.is_null() {
            (ptr::null(), 0usize)
        } else {
            let sep_ref = &*sep;
            (sep_ref.ptr as *const u8, sep_ref.len)
        };

        // Calculate total length needed
        let mut total_len: usize = 0;

        // The array contains pointers to HaxeString
        // elem_size should be sizeof(*HaxeString) = sizeof(usize) typically
        for i in 0..arr_ref.len {
            // Get pointer to the HaxeString pointer stored in the array
            let elem_ptr = arr_ref.ptr.add(i * arr_ref.elem_size) as *const *const HaxeString;
            let string_ptr = *elem_ptr;

            if !string_ptr.is_null() {
                total_len += (*string_ptr).len;
            }

            // Add separator length (except for last element)
            if i < arr_ref.len - 1 {
                total_len += sep_len;
            }
        }

        // Allocate result buffer
        let buf_cap = total_len + 1; // +1 for null terminator
        let buf_layout = Layout::from_size_align_unchecked(buf_cap, 1);
        let buf_ptr = alloc(buf_layout);
        if buf_ptr.is_null() {
            panic!("Failed to allocate buffer for join result");
        }

        // Copy strings with separator
        let mut offset: usize = 0;
        for i in 0..arr_ref.len {
            let elem_ptr = arr_ref.ptr.add(i * arr_ref.elem_size) as *const *const HaxeString;
            let string_ptr = *elem_ptr;

            if !string_ptr.is_null() {
                let s = &*string_ptr;
                if s.len > 0 && !s.ptr.is_null() {
                    ptr::copy_nonoverlapping(s.ptr, buf_ptr.add(offset), s.len);
                    offset += s.len;
                }
            }

            // Add separator (except for last element)
            if i < arr_ref.len - 1 && sep_len > 0 && !sep_ptr.is_null() {
                ptr::copy_nonoverlapping(sep_ptr, buf_ptr.add(offset), sep_len);
                offset += sep_len;
            }
        }

        // Null terminate
        *buf_ptr.add(offset) = 0;

        // Set up result HaxeString
        (*result_ptr).ptr = buf_ptr;
        (*result_ptr).len = total_len;
        (*result_ptr).cap = buf_cap;

        result_ptr
    }
}

// ============================================================================
// Higher-Order Array Methods
// ============================================================================

/// Map: apply callback to each element, collect results into a new array.
/// Callback signature: fn(env_ptr: *mut u8, element: i64) -> i64
#[no_mangle]
pub extern "C" fn haxe_array_map(
    out: *mut HaxeArray,
    arr: *const HaxeArray,
    fn_ptr: usize,
    env_ptr: *mut u8,
) {
    if arr.is_null() || out.is_null() || fn_ptr == 0 {
        // Initialize empty output array
        if !out.is_null() {
            haxe_array_new(out, 8);
        }
        return;
    }

    unsafe {
        let arr_ref = &*arr;
        let len = arr_ref.len;
        let elem_size = arr_ref.elem_size;

        // Initialize output array with same length capacity
        let out_cap = len.max(INITIAL_CAPACITY);
        let out_total = out_cap * 8; // result elements are always i64 (8 bytes)
        let layout = Layout::from_size_align_unchecked(out_total, 8);
        let out_ptr = alloc(layout);
        if out_ptr.is_null() {
            panic!("Failed to allocate memory for Array.map result");
        }

        // Cast fn_ptr to callable function pointer
        let callback: extern "C" fn(*mut u8, i64) -> i64 = std::mem::transmute(fn_ptr);

        for i in 0..len {
            // Read element as i64
            let elem = if elem_size == 8 {
                *(arr_ref.ptr.add(i * elem_size) as *const i64)
            } else if elem_size == 4 {
                *(arr_ref.ptr.add(i * elem_size) as *const i32) as i64
            } else {
                0i64
            };

            // Call the callback
            let result = callback(env_ptr, elem);

            // Store result
            *(out_ptr.add(i * 8) as *mut i64) = result;
        }

        (*out).ptr = out_ptr;
        (*out).len = len;
        (*out).cap = out_cap;
        (*out).elem_size = 8;
    }
}

/// Filter: keep elements where callback returns non-zero.
/// Callback signature: fn(env_ptr: *mut u8, element: i64) -> i64 (0 = false, non-zero = true)
#[no_mangle]
pub extern "C" fn haxe_array_filter(
    out: *mut HaxeArray,
    arr: *const HaxeArray,
    fn_ptr: usize,
    env_ptr: *mut u8,
) {
    if arr.is_null() || out.is_null() || fn_ptr == 0 {
        if !out.is_null() {
            haxe_array_new(out, 8);
        }
        return;
    }

    unsafe {
        let arr_ref = &*arr;
        let len = arr_ref.len;
        let elem_size = arr_ref.elem_size;

        // Allocate output with same capacity as input (worst case: all pass)
        let out_cap = len.max(INITIAL_CAPACITY);
        let out_total = out_cap * 8;
        let layout = Layout::from_size_align_unchecked(out_total, 8);
        let out_ptr = alloc(layout);
        if out_ptr.is_null() {
            panic!("Failed to allocate memory for Array.filter result");
        }

        let callback: extern "C" fn(*mut u8, i64) -> i64 = std::mem::transmute(fn_ptr);

        let mut out_len = 0usize;
        for i in 0..len {
            let elem = if elem_size == 8 {
                *(arr_ref.ptr.add(i * elem_size) as *const i64)
            } else if elem_size == 4 {
                *(arr_ref.ptr.add(i * elem_size) as *const i32) as i64
            } else {
                0i64
            };

            let keep = callback(env_ptr, elem);
            if keep != 0 {
                *(out_ptr.add(out_len * 8) as *mut i64) = elem;
                out_len += 1;
            }
        }

        (*out).ptr = out_ptr;
        (*out).len = out_len;
        (*out).cap = out_cap;
        (*out).elem_size = 8;
    }
}

/// Sort: in-place sort using comparator callback.
/// Callback signature: fn(env_ptr: *mut u8, a: i64, b: i64) -> i32
/// Returns negative if a < b, 0 if equal, positive if a > b.
#[no_mangle]
pub extern "C" fn haxe_array_sort(arr: *mut HaxeArray, fn_ptr: usize, env_ptr: *mut u8) {
    if arr.is_null() || fn_ptr == 0 {
        return;
    }

    unsafe {
        let arr_ref = &mut *arr;
        let len = arr_ref.len;
        if len <= 1 {
            return;
        }

        let callback: extern "C" fn(*mut u8, i64, i64) -> i32 = std::mem::transmute(fn_ptr);

        // Simple insertion sort for now (stable, in-place)
        // Elements are i64 (8 bytes each)
        let data = arr_ref.ptr as *mut i64;
        for i in 1..len {
            let key = *data.add(i);
            let mut j = i as isize - 1;
            while j >= 0 {
                let cmp = callback(env_ptr, *data.add(j as usize), key);
                if cmp > 0 {
                    *data.add((j + 1) as usize) = *data.add(j as usize);
                    j -= 1;
                } else {
                    break;
                }
            }
            *data.add((j + 1) as usize) = key;
        }
    }
}
