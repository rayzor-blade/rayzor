//! Vec<u8> implementation for Haxe Bytes
//!
//! This provides a dynamically-sized byte array that can be called from JIT code.

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::ptr;

/// Vec<u8> representation: { ptr: *mut u8, len: usize, cap: usize }
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HaxeVec {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

/// Create a new empty Vec with initial capacity of 16
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_new() -> HaxeVec {
    const INITIAL_CAPACITY: usize = 16;

    unsafe {
        let layout = Layout::from_size_align_unchecked(INITIAL_CAPACITY, 1);
        let ptr = alloc(layout);

        if ptr.is_null() {
            panic!("Failed to allocate memory for Vec");
        }

        HaxeVec {
            ptr,
            len: 0,
            cap: INITIAL_CAPACITY,
        }
    }
}

/// Push a byte onto the end of the vec
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_push(vec: *mut HaxeVec, value: u8) {
    unsafe {
        if vec.is_null() {
            return;
        }

        let v = &mut *vec;

        // Check if we need to grow
        if v.len >= v.cap {
            let new_cap = v.cap * 2;
            let old_layout = Layout::from_size_align_unchecked(v.cap, 1);
            let new_ptr = realloc(v.ptr, old_layout, new_cap);

            if new_ptr.is_null() {
                panic!("Failed to reallocate Vec");
            }

            v.ptr = new_ptr;
            v.cap = new_cap;
        }

        // Write the value
        *v.ptr.add(v.len) = value;
        v.len += 1;
    }
}

/// Get a byte at index (returns 0 if out of bounds)
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_get(vec: *const HaxeVec, index: usize) -> u8 {
    unsafe {
        if vec.is_null() {
            return 0;
        }

        let v = &*vec;

        if index >= v.len {
            return 0;
        }

        *v.ptr.add(index)
    }
}

/// Set a byte at index (no-op if out of bounds)
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_set(vec: *mut HaxeVec, index: usize, value: u8) {
    unsafe {
        if vec.is_null() {
            return;
        }

        let v = &mut *vec;

        if index >= v.len {
            return;
        }

        *v.ptr.add(index) = value;
    }
}

/// Get the length of the vec
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_len(vec: *const HaxeVec) -> usize {
    unsafe {
        if vec.is_null() {
            return 0;
        }

        (*vec).len
    }
}

/// Get the capacity of the vec
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_capacity(vec: *const HaxeVec) -> usize {
    unsafe {
        if vec.is_null() {
            return 0;
        }

        (*vec).cap
    }
}

/// Clear the vec (set length to 0, keep capacity)
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_clear(vec: *mut HaxeVec) {
    unsafe {
        if vec.is_null() {
            return;
        }

        (*vec).len = 0;
    }
}

/// Free the vec's memory
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_free(vec: *mut HaxeVec) {
    unsafe {
        if vec.is_null() {
            return;
        }

        let v = &*vec;

        if !v.ptr.is_null() && v.cap > 0 {
            let layout = Layout::from_size_align_unchecked(v.cap, 1);
            dealloc(v.ptr, layout);
        }

        // Zero out the struct
        (*vec).ptr = ptr::null_mut();
        (*vec).len = 0;
        (*vec).cap = 0;
    }
}

/// Reserve additional capacity
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_reserve(vec: *mut HaxeVec, additional: usize) {
    unsafe {
        if vec.is_null() {
            return;
        }

        let v = &mut *vec;
        let required_cap = v.len + additional;

        if required_cap <= v.cap {
            return; // Already have enough capacity
        }

        let old_layout = Layout::from_size_align_unchecked(v.cap, 1);
        let new_ptr = realloc(v.ptr, old_layout, required_cap);

        if new_ptr.is_null() {
            panic!("Failed to reserve Vec capacity");
        }

        v.ptr = new_ptr;
        v.cap = required_cap;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_new() {
        let vec = haxe_vec_new();
        assert!(!vec.ptr.is_null());
        assert_eq!(vec.len, 0);
        assert_eq!(vec.cap, 16);

        haxe_vec_free(&mut vec.clone());
    }

    #[test]
    fn test_vec_push_and_get() {
        let mut vec = haxe_vec_new();

        haxe_vec_push(&mut vec, 42);
        haxe_vec_push(&mut vec, 100);
        haxe_vec_push(&mut vec, 200);

        assert_eq!(haxe_vec_len(&vec), 3);
        assert_eq!(haxe_vec_get(&vec, 0), 42);
        assert_eq!(haxe_vec_get(&vec, 1), 100);
        assert_eq!(haxe_vec_get(&vec, 2), 200);

        haxe_vec_free(&mut vec);
    }

    #[test]
    fn test_vec_growth() {
        let mut vec = haxe_vec_new();

        // Push more than initial capacity
        for i in 0..20 {
            haxe_vec_push(&mut vec, i as u8);
        }

        assert_eq!(haxe_vec_len(&vec), 20);
        assert!(haxe_vec_capacity(&vec) >= 20);

        // Verify all values
        for i in 0..20 {
            assert_eq!(haxe_vec_get(&vec, i), i as u8);
        }

        haxe_vec_free(&mut vec);
    }
}
