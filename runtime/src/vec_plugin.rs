//! Vec runtime with pointer-based API (no struct returns)
//!
//! This avoids ABI issues with struct returns by using out-parameters

use std::alloc::{Layout, alloc, dealloc, realloc};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct HaxeVec {
    pub ptr: *mut u8,
    pub len: usize,
    pub cap: usize,
}

/// Create a new Vec and write it to the out pointer
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_new_ptr(out: *mut HaxeVec) {
    const INITIAL_CAPACITY: usize = 16;
    unsafe {
        let layout = Layout::from_size_align_unchecked(INITIAL_CAPACITY, 1);
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("Failed to allocate memory for Vec");
        }
        (*out).ptr = ptr;
        (*out).len = 0;
        (*out).cap = INITIAL_CAPACITY;
    }
}

/// Push a value onto the vec (may reallocate)
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_push_ptr(vec: *mut HaxeVec, value: u8) {
    if vec.is_null() {
        return;
    }

    unsafe {
        let vec_ref = &mut *vec;

        // Check if we need to grow
        if vec_ref.len >= vec_ref.cap {
            let new_cap = vec_ref.cap * 2;
            let old_layout = Layout::from_size_align_unchecked(vec_ref.cap, 1);
            let new_layout = Layout::from_size_align_unchecked(new_cap, 1);

            let new_ptr = if vec_ref.ptr.is_null() {
                alloc(new_layout)
            } else {
                realloc(vec_ref.ptr, old_layout, new_cap)
            };

            if new_ptr.is_null() {
                panic!("Failed to reallocate memory for Vec");
            }

            vec_ref.ptr = new_ptr;
            vec_ref.cap = new_cap;
        }

        // Add the element
        *vec_ref.ptr.add(vec_ref.len) = value;
        vec_ref.len += 1;
    }
}

/// Get element at index
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_get_ptr(vec: *const HaxeVec, index: usize) -> u8 {
    if vec.is_null() {
        return 0;
    }

    unsafe {
        let vec_ref = &*vec;
        if index >= vec_ref.len {
            return 0;
        }
        *vec_ref.ptr.add(index)
    }
}

/// Get length
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_len_ptr(vec: *const HaxeVec) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len }
}

/// Free the vec
#[unsafe(no_mangle)]
pub extern "C" fn haxe_vec_free_ptr(vec: *mut HaxeVec) {
    if vec.is_null() {
        return;
    }

    unsafe {
        let vec_ref = &*vec;
        if !vec_ref.ptr.is_null() && vec_ref.cap > 0 {
            let layout = Layout::from_size_align_unchecked(vec_ref.cap, 1);
            dealloc(vec_ref.ptr, layout);
        }
    }
}
