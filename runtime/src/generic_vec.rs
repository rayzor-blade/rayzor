//! Generic Vec<T> Runtime Implementation
//!
//! Provides native, high-performance vector operations for monomorphized generic types.
//! Each element type gets its own specialized functions for optimal performance.
//!
//! ## Memory Layout
//!
//! All Vec types share the same layout:
//! ```
//! struct Vec<T> {
//!     ptr: *mut T,      // Pointer to contiguous element storage
//!     len: usize,       // Number of elements currently stored
//!     cap: usize,       // Total capacity (elements, not bytes)
//! }
//! ```
//!
//! ## Performance Benefits over Array<T>
//!
//! 1. **Contiguous memory**: Better cache locality
//! 2. **No boxing**: Primitives stored directly (i32 not Object)
//! 3. **Geometric growth**: Amortized O(1) push
//! 4. **Type-specific code**: No runtime type dispatch

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::mem;
use std::ptr;

/// Initial capacity for new vectors
const INITIAL_CAPACITY: usize = 8;

/// Growth factor when resizing (2x)
const GROWTH_FACTOR: usize = 2;

// ============================================================================
// Vec<i32> - Integer vectors
// ============================================================================

/// Opaque handle for Vec<i32>
#[repr(C)]
pub struct VecI32 {
    ptr: *mut i32,
    len: usize,
    cap: usize,
}

/// Create a new empty Vec<i32>
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_new() -> *mut VecI32 {
    let layout = Layout::new::<VecI32>();
    unsafe {
        let vec_ptr = alloc(layout) as *mut VecI32;
        if vec_ptr.is_null() {
            return ptr::null_mut();
        }

        // Allocate initial element storage
        let elem_layout = Layout::array::<i32>(INITIAL_CAPACITY).unwrap();
        let data_ptr = alloc(elem_layout) as *mut i32;
        if data_ptr.is_null() {
            dealloc(vec_ptr as *mut u8, layout);
            return ptr::null_mut();
        }

        (*vec_ptr).ptr = data_ptr;
        (*vec_ptr).len = 0;
        (*vec_ptr).cap = INITIAL_CAPACITY;

        vec_ptr
    }
}

/// Create Vec<i32> with specific capacity
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_with_capacity(capacity: usize) -> *mut VecI32 {
    let cap = if capacity == 0 {
        INITIAL_CAPACITY
    } else {
        capacity
    };
    let layout = Layout::new::<VecI32>();

    unsafe {
        let vec_ptr = alloc(layout) as *mut VecI32;
        if vec_ptr.is_null() {
            return ptr::null_mut();
        }

        let elem_layout = Layout::array::<i32>(cap).unwrap();
        let data_ptr = alloc(elem_layout) as *mut i32;
        if data_ptr.is_null() {
            dealloc(vec_ptr as *mut u8, layout);
            return ptr::null_mut();
        }

        (*vec_ptr).ptr = data_ptr;
        (*vec_ptr).len = 0;
        (*vec_ptr).cap = cap;

        vec_ptr
    }
}

/// Push element to Vec<i32>
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_push(vec: *mut VecI32, value: i32) {
    if vec.is_null() {
        return;
    }

    unsafe {
        let v = &mut *vec;

        // Grow if needed
        if v.len >= v.cap {
            let new_cap = v.cap * GROWTH_FACTOR;
            let old_layout = Layout::array::<i32>(v.cap).unwrap();
            let new_ptr = realloc(
                v.ptr as *mut u8,
                old_layout,
                new_cap * mem::size_of::<i32>(),
            );
            if new_ptr.is_null() {
                panic!("Vec<i32> allocation failed");
            }
            v.ptr = new_ptr as *mut i32;
            v.cap = new_cap;
        }

        // Write element
        *v.ptr.add(v.len) = value;
        v.len += 1;
    }
}

/// Pop element from Vec<i32>, returns 0 if empty
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_pop(vec: *mut VecI32) -> i32 {
    if vec.is_null() {
        return 0;
    }

    unsafe {
        let v = &mut *vec;
        if v.len == 0 {
            return 0;
        }
        v.len -= 1;
        *v.ptr.add(v.len)
    }
}

/// Get element at index (returns 0 if out of bounds)
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_get(vec: *const VecI32, index: usize) -> i32 {
    if vec.is_null() {
        return 0;
    }

    unsafe {
        let v = &*vec;
        if index >= v.len {
            return 0;
        }
        *v.ptr.add(index)
    }
}

/// Set element at index (no-op if out of bounds)
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_set(vec: *mut VecI32, index: usize, value: i32) {
    if vec.is_null() {
        return;
    }

    unsafe {
        let v = &mut *vec;
        if index >= v.len {
            return;
        }
        *v.ptr.add(index) = value;
    }
}

/// Get length
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_len(vec: *const VecI32) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len }
}

/// Get capacity
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_capacity(vec: *const VecI32) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).cap }
}

/// Check if empty
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_is_empty(vec: *const VecI32) -> bool {
    if vec.is_null() {
        return true;
    }
    unsafe { (*vec).len == 0 }
}

/// Clear vector (keeps capacity)
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_clear(vec: *mut VecI32) {
    if vec.is_null() {
        return;
    }
    unsafe {
        (*vec).len = 0;
    }
}

/// Get first element (returns 0 if empty)
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_first(vec: *const VecI32) -> i32 {
    rayzor_vec_i32_get(vec, 0)
}

/// Get last element (returns 0 if empty)
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_last(vec: *const VecI32) -> i32 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &*vec;
        if v.len == 0 {
            return 0;
        }
        *v.ptr.add(v.len - 1)
    }
}

/// Sort Vec<i32> in ascending order
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_sort(vec: *mut VecI32) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len <= 1 {
            return;
        }
        let slice = std::slice::from_raw_parts_mut(v.ptr, v.len);
        slice.sort();
    }
}

/// Sort Vec<i32> using a comparison function (Haxe closure)
/// compare_fn: closure function pointer
/// compare_env: closure captured environment
/// The closure signature is: (env, a, b) -> Int where negative means a < b, 0 means a == b, positive means a > b
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_sort_by(
    vec: *mut VecI32,
    compare_fn: *const u8,
    compare_env: *const u8,
) {
    if vec.is_null() || compare_fn.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len <= 1 {
            return;
        }

        // The comparison function takes (env, a, b) and returns i32
        type CompareFn = extern "C" fn(*const u8, i32, i32) -> i32;
        let compare: CompareFn = std::mem::transmute(compare_fn);

        let slice = std::slice::from_raw_parts_mut(v.ptr, v.len);
        slice.sort_by(|a, b| {
            let cmp = compare(compare_env, *a, *b);
            if cmp < 0 {
                std::cmp::Ordering::Less
            } else if cmp > 0 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
}

/// Free Vec<i32>
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i32_free(vec: *mut VecI32) {
    if vec.is_null() {
        return;
    }

    unsafe {
        let v = &*vec;
        if !v.ptr.is_null() && v.cap > 0 {
            let elem_layout = Layout::array::<i32>(v.cap).unwrap();
            dealloc(v.ptr as *mut u8, elem_layout);
        }
        let vec_layout = Layout::new::<VecI32>();
        dealloc(vec as *mut u8, vec_layout);
    }
}

// ============================================================================
// Vec<i64> - Long integer vectors
// ============================================================================

#[repr(C)]
pub struct VecI64 {
    ptr: *mut i64,
    len: usize,
    cap: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_new() -> *mut VecI64 {
    let layout = Layout::new::<VecI64>();
    unsafe {
        let vec_ptr = alloc(layout) as *mut VecI64;
        if vec_ptr.is_null() {
            return ptr::null_mut();
        }

        let elem_layout = Layout::array::<i64>(INITIAL_CAPACITY).unwrap();
        let data_ptr = alloc(elem_layout) as *mut i64;
        if data_ptr.is_null() {
            dealloc(vec_ptr as *mut u8, layout);
            return ptr::null_mut();
        }

        (*vec_ptr).ptr = data_ptr;
        (*vec_ptr).len = 0;
        (*vec_ptr).cap = INITIAL_CAPACITY;
        vec_ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_push(vec: *mut VecI64, value: i64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len >= v.cap {
            let new_cap = v.cap * GROWTH_FACTOR;
            let old_layout = Layout::array::<i64>(v.cap).unwrap();
            let new_ptr = realloc(
                v.ptr as *mut u8,
                old_layout,
                new_cap * mem::size_of::<i64>(),
            );
            if new_ptr.is_null() {
                panic!("Vec<i64> allocation failed");
            }
            v.ptr = new_ptr as *mut i64;
            v.cap = new_cap;
        }
        *v.ptr.add(v.len) = value;
        v.len += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_pop(vec: *mut VecI64) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &mut *vec;
        if v.len == 0 {
            return 0;
        }
        v.len -= 1;
        *v.ptr.add(v.len)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_get(vec: *const VecI64, index: usize) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &*vec;
        if index >= v.len {
            return 0;
        }
        *v.ptr.add(index)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_set(vec: *mut VecI64, index: usize, value: i64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if index >= v.len {
            return;
        }
        *v.ptr.add(index) = value;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_len(vec: *const VecI64) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_is_empty(vec: *const VecI64) -> bool {
    if vec.is_null() {
        return true;
    }
    unsafe { (*vec).len == 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_clear(vec: *mut VecI64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        (*vec).len = 0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_first(vec: *const VecI64) -> i64 {
    rayzor_vec_i64_get(vec, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_last(vec: *const VecI64) -> i64 {
    if vec.is_null() {
        return 0;
    }
    unsafe {
        let v = &*vec;
        if v.len == 0 {
            return 0;
        }
        *v.ptr.add(v.len - 1)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_i64_free(vec: *mut VecI64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &*vec;
        if !v.ptr.is_null() && v.cap > 0 {
            let elem_layout = Layout::array::<i64>(v.cap).unwrap();
            dealloc(v.ptr as *mut u8, elem_layout);
        }
        dealloc(vec as *mut u8, Layout::new::<VecI64>());
    }
}

// ============================================================================
// Vec<f64> - Float vectors
// ============================================================================

#[repr(C)]
pub struct VecF64 {
    ptr: *mut f64,
    len: usize,
    cap: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_new() -> *mut VecF64 {
    let layout = Layout::new::<VecF64>();
    unsafe {
        let vec_ptr = alloc(layout) as *mut VecF64;
        if vec_ptr.is_null() {
            return ptr::null_mut();
        }

        let elem_layout = Layout::array::<f64>(INITIAL_CAPACITY).unwrap();
        let data_ptr = alloc(elem_layout) as *mut f64;
        if data_ptr.is_null() {
            dealloc(vec_ptr as *mut u8, layout);
            return ptr::null_mut();
        }

        (*vec_ptr).ptr = data_ptr;
        (*vec_ptr).len = 0;
        (*vec_ptr).cap = INITIAL_CAPACITY;
        vec_ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_push(vec: *mut VecF64, value: f64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len >= v.cap {
            let new_cap = v.cap * GROWTH_FACTOR;
            let old_layout = Layout::array::<f64>(v.cap).unwrap();
            let new_ptr = realloc(
                v.ptr as *mut u8,
                old_layout,
                new_cap * mem::size_of::<f64>(),
            );
            if new_ptr.is_null() {
                panic!("Vec<f64> allocation failed");
            }
            v.ptr = new_ptr as *mut f64;
            v.cap = new_cap;
        }
        *v.ptr.add(v.len) = value;
        v.len += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_pop(vec: *mut VecF64) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe {
        let v = &mut *vec;
        if v.len == 0 {
            return 0.0;
        }
        v.len -= 1;
        *v.ptr.add(v.len)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_get(vec: *const VecF64, index: usize) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe {
        let v = &*vec;
        if index >= v.len {
            return 0.0;
        }
        *v.ptr.add(index)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_set(vec: *mut VecF64, index: usize, value: f64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if index >= v.len {
            return;
        }
        *v.ptr.add(index) = value;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_len(vec: *const VecF64) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_is_empty(vec: *const VecF64) -> bool {
    if vec.is_null() {
        return true;
    }
    unsafe { (*vec).len == 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_clear(vec: *mut VecF64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        (*vec).len = 0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_first(vec: *const VecF64) -> f64 {
    rayzor_vec_f64_get(vec, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_last(vec: *const VecF64) -> f64 {
    if vec.is_null() {
        return 0.0;
    }
    unsafe {
        let v = &*vec;
        if v.len == 0 {
            return 0.0;
        }
        *v.ptr.add(v.len - 1)
    }
}

/// Sort Vec<f64> in ascending order
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_sort(vec: *mut VecF64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len <= 1 {
            return;
        }
        let slice = std::slice::from_raw_parts_mut(v.ptr, v.len);
        // Use partial_cmp for f64 since it doesn't implement Ord
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
}

/// Sort Vec<f64> using a comparison function (Haxe closure)
/// compare_fn: closure function pointer
/// compare_env: closure captured environment
/// The closure signature is: (env, a, b) -> Int where negative means a < b, 0 means a == b, positive means a > b
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_sort_by(
    vec: *mut VecF64,
    compare_fn: *const u8,
    compare_env: *const u8,
) {
    if vec.is_null() || compare_fn.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len <= 1 {
            return;
        }

        // The comparison function takes (env, a, b) and returns i32
        type CompareFn = extern "C" fn(*const u8, f64, f64) -> i32;
        let compare: CompareFn = std::mem::transmute(compare_fn);

        let slice = std::slice::from_raw_parts_mut(v.ptr, v.len);
        slice.sort_by(|a, b| {
            let cmp = compare(compare_env, *a, *b);
            if cmp < 0 {
                std::cmp::Ordering::Less
            } else if cmp > 0 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_f64_free(vec: *mut VecF64) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &*vec;
        if !v.ptr.is_null() && v.cap > 0 {
            let elem_layout = Layout::array::<f64>(v.cap).unwrap();
            dealloc(v.ptr as *mut u8, elem_layout);
        }
        dealloc(vec as *mut u8, Layout::new::<VecF64>());
    }
}

// ============================================================================
// Vec<*mut u8> - Pointer vectors (for objects/strings)
// ============================================================================

#[repr(C)]
pub struct VecPtr {
    ptr: *mut *mut u8,
    len: usize,
    cap: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_new() -> *mut VecPtr {
    let layout = Layout::new::<VecPtr>();
    unsafe {
        let vec_ptr = alloc(layout) as *mut VecPtr;
        if vec_ptr.is_null() {
            return ptr::null_mut();
        }

        let elem_layout = Layout::array::<*mut u8>(INITIAL_CAPACITY).unwrap();
        let data_ptr = alloc(elem_layout) as *mut *mut u8;
        if data_ptr.is_null() {
            dealloc(vec_ptr as *mut u8, layout);
            return ptr::null_mut();
        }

        (*vec_ptr).ptr = data_ptr;
        (*vec_ptr).len = 0;
        (*vec_ptr).cap = INITIAL_CAPACITY;
        vec_ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_push(vec: *mut VecPtr, value: *mut u8) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len >= v.cap {
            let new_cap = v.cap * GROWTH_FACTOR;
            let old_layout = Layout::array::<*mut u8>(v.cap).unwrap();
            let new_ptr = realloc(
                v.ptr as *mut u8,
                old_layout,
                new_cap * mem::size_of::<*mut u8>(),
            );
            if new_ptr.is_null() {
                panic!("Vec<ptr> allocation failed");
            }
            v.ptr = new_ptr as *mut *mut u8;
            v.cap = new_cap;
        }
        *v.ptr.add(v.len) = value;
        v.len += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_pop(vec: *mut VecPtr) -> *mut u8 {
    if vec.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let v = &mut *vec;
        if v.len == 0 {
            return ptr::null_mut();
        }
        v.len -= 1;
        *v.ptr.add(v.len)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_get(vec: *const VecPtr, index: usize) -> *mut u8 {
    if vec.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let v = &*vec;
        if index >= v.len {
            return ptr::null_mut();
        }
        *v.ptr.add(index)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_set(vec: *mut VecPtr, index: usize, value: *mut u8) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if index >= v.len {
            return;
        }
        *v.ptr.add(index) = value;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_len(vec: *const VecPtr) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_is_empty(vec: *const VecPtr) -> bool {
    if vec.is_null() {
        return true;
    }
    unsafe { (*vec).len == 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_clear(vec: *mut VecPtr) {
    if vec.is_null() {
        return;
    }
    unsafe {
        (*vec).len = 0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_first(vec: *const VecPtr) -> *mut u8 {
    rayzor_vec_ptr_get(vec, 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_last(vec: *const VecPtr) -> *mut u8 {
    if vec.is_null() {
        return ptr::null_mut();
    }
    unsafe {
        let v = &*vec;
        if v.len == 0 {
            return ptr::null_mut();
        }
        *v.ptr.add(v.len - 1)
    }
}

/// Sort Vec<ptr> using a comparison function (Haxe closure)
/// For reference types, there's no natural ordering so only sortBy is provided.
/// compare_fn: closure function pointer
/// compare_env: closure captured environment
/// The closure signature is: (env, a, b) -> Int where negative means a < b, 0 means a == b, positive means a > b
#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_sort_by(
    vec: *mut VecPtr,
    compare_fn: *const u8,
    compare_env: *const u8,
) {
    if vec.is_null() || compare_fn.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len <= 1 {
            return;
        }

        // The comparison function takes (env, a, b) and returns i32
        // where a and b are pointers to objects
        type CompareFn = extern "C" fn(*const u8, *mut u8, *mut u8) -> i32;
        let compare: CompareFn = std::mem::transmute(compare_fn);

        let slice = std::slice::from_raw_parts_mut(v.ptr, v.len);
        slice.sort_by(|a, b| {
            let cmp = compare(compare_env, *a, *b);
            if cmp < 0 {
                std::cmp::Ordering::Less
            } else if cmp > 0 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_ptr_free(vec: *mut VecPtr) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &*vec;
        if !v.ptr.is_null() && v.cap > 0 {
            let elem_layout = Layout::array::<*mut u8>(v.cap).unwrap();
            dealloc(v.ptr as *mut u8, elem_layout);
        }
        dealloc(vec as *mut u8, Layout::new::<VecPtr>());
    }
}

// ============================================================================
// Vec<bool> - Boolean vectors (packed as u8)
// ============================================================================

#[repr(C)]
pub struct VecBool {
    ptr: *mut u8, // Each bool is stored as a u8 (0 or 1)
    len: usize,
    cap: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_new() -> *mut VecBool {
    let layout = Layout::new::<VecBool>();
    unsafe {
        let vec_ptr = alloc(layout) as *mut VecBool;
        if vec_ptr.is_null() {
            return ptr::null_mut();
        }

        let elem_layout = Layout::array::<u8>(INITIAL_CAPACITY).unwrap();
        let data_ptr = alloc(elem_layout);
        if data_ptr.is_null() {
            dealloc(vec_ptr as *mut u8, layout);
            return ptr::null_mut();
        }

        (*vec_ptr).ptr = data_ptr;
        (*vec_ptr).len = 0;
        (*vec_ptr).cap = INITIAL_CAPACITY;
        vec_ptr
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_push(vec: *mut VecBool, value: bool) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if v.len >= v.cap {
            let new_cap = v.cap * GROWTH_FACTOR;
            let old_layout = Layout::array::<u8>(v.cap).unwrap();
            let new_ptr = realloc(v.ptr, old_layout, new_cap);
            if new_ptr.is_null() {
                panic!("Vec<bool> allocation failed");
            }
            v.ptr = new_ptr;
            v.cap = new_cap;
        }
        *v.ptr.add(v.len) = if value { 1 } else { 0 };
        v.len += 1;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_pop(vec: *mut VecBool) -> bool {
    if vec.is_null() {
        return false;
    }
    unsafe {
        let v = &mut *vec;
        if v.len == 0 {
            return false;
        }
        v.len -= 1;
        *v.ptr.add(v.len) != 0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_get(vec: *const VecBool, index: usize) -> bool {
    if vec.is_null() {
        return false;
    }
    unsafe {
        let v = &*vec;
        if index >= v.len {
            return false;
        }
        *v.ptr.add(index) != 0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_set(vec: *mut VecBool, index: usize, value: bool) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &mut *vec;
        if index >= v.len {
            return;
        }
        *v.ptr.add(index) = if value { 1 } else { 0 };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_len(vec: *const VecBool) -> usize {
    if vec.is_null() {
        return 0;
    }
    unsafe { (*vec).len }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_is_empty(vec: *const VecBool) -> bool {
    if vec.is_null() {
        return true;
    }
    unsafe { (*vec).len == 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_clear(vec: *mut VecBool) {
    if vec.is_null() {
        return;
    }
    unsafe {
        (*vec).len = 0;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rayzor_vec_bool_free(vec: *mut VecBool) {
    if vec.is_null() {
        return;
    }
    unsafe {
        let v = &*vec;
        if !v.ptr.is_null() && v.cap > 0 {
            let elem_layout = Layout::array::<u8>(v.cap).unwrap();
            dealloc(v.ptr, elem_layout);
        }
        dealloc(vec as *mut u8, Layout::new::<VecBool>());
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_i32_basic() {
        let vec = rayzor_vec_i32_new();
        assert!(!vec.is_null());
        assert!(rayzor_vec_i32_is_empty(vec));

        rayzor_vec_i32_push(vec, 10);
        rayzor_vec_i32_push(vec, 20);
        rayzor_vec_i32_push(vec, 30);

        assert_eq!(rayzor_vec_i32_len(vec), 3);
        assert_eq!(rayzor_vec_i32_get(vec, 0), 10);
        assert_eq!(rayzor_vec_i32_get(vec, 1), 20);
        assert_eq!(rayzor_vec_i32_get(vec, 2), 30);
        assert_eq!(rayzor_vec_i32_first(vec), 10);
        assert_eq!(rayzor_vec_i32_last(vec), 30);

        rayzor_vec_i32_set(vec, 1, 42);
        assert_eq!(rayzor_vec_i32_get(vec, 1), 42);

        let popped = rayzor_vec_i32_pop(vec);
        assert_eq!(popped, 30);
        assert_eq!(rayzor_vec_i32_len(vec), 2);

        rayzor_vec_i32_free(vec);
    }

    #[test]
    fn test_vec_i32_growth() {
        let vec = rayzor_vec_i32_new();

        // Push more than initial capacity
        for i in 0..100 {
            rayzor_vec_i32_push(vec, i);
        }

        assert_eq!(rayzor_vec_i32_len(vec), 100);
        assert!(rayzor_vec_i32_capacity(vec) >= 100);

        // Verify all values
        for i in 0..100 {
            assert_eq!(rayzor_vec_i32_get(vec, i as usize), i);
        }

        rayzor_vec_i32_free(vec);
    }

    #[test]
    fn test_vec_f64_basic() {
        let vec = rayzor_vec_f64_new();

        rayzor_vec_f64_push(vec, 1.5);
        rayzor_vec_f64_push(vec, 2.5);
        rayzor_vec_f64_push(vec, 3.5);

        assert_eq!(rayzor_vec_f64_len(vec), 3);
        assert!((rayzor_vec_f64_get(vec, 0) - 1.5).abs() < 0.001);
        assert!((rayzor_vec_f64_get(vec, 1) - 2.5).abs() < 0.001);
        assert!((rayzor_vec_f64_get(vec, 2) - 3.5).abs() < 0.001);

        rayzor_vec_f64_free(vec);
    }

    #[test]
    fn test_vec_ptr_basic() {
        let vec = rayzor_vec_ptr_new();

        let a: i32 = 10;
        let b: i32 = 20;

        rayzor_vec_ptr_push(vec, &a as *const i32 as *mut u8);
        rayzor_vec_ptr_push(vec, &b as *const i32 as *mut u8);

        assert_eq!(rayzor_vec_ptr_len(vec), 2);

        let ptr_a = rayzor_vec_ptr_get(vec, 0) as *const i32;
        let ptr_b = rayzor_vec_ptr_get(vec, 1) as *const i32;

        unsafe {
            assert_eq!(*ptr_a, 10);
            assert_eq!(*ptr_b, 20);
        }

        rayzor_vec_ptr_free(vec);
    }

    #[test]
    fn test_vec_bool_basic() {
        let vec = rayzor_vec_bool_new();

        rayzor_vec_bool_push(vec, true);
        rayzor_vec_bool_push(vec, false);
        rayzor_vec_bool_push(vec, true);

        assert_eq!(rayzor_vec_bool_len(vec), 3);
        assert!(rayzor_vec_bool_get(vec, 0));
        assert!(!rayzor_vec_bool_get(vec, 1));
        assert!(rayzor_vec_bool_get(vec, 2));

        rayzor_vec_bool_free(vec);
    }
}
