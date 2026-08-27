//! Box<T> runtime functions — single-owner heap allocation.
//!
//! Box<T> is a zero-cost abstract over Int (i64) in Haxe.
//! At runtime, a Box is just a heap pointer. These functions provide
//! the allocation and deallocation primitives.
//!
//! Uses libc malloc/free to match the Cranelift JIT backend, which maps
//! all MIR malloc/free calls to libc directly.
//!
//! All parameters and return values use i64 to match the MIR/LLVM type system
//! where Box is represented as an opaque i64 (pointer-sized integer).

// Rust 1.98 expects c_void spelling for these libc symbols; u8 has the same
// pointer ABI and is the representation used by the box runtime.
#[allow(suspicious_runtime_symbol_definitions)]
extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

/// Allocate a value on the heap (Box.init).
///
/// Takes an i64 value (which may be a pointer to a larger object or a primitive),
/// allocates 8 bytes on the heap via libc malloc, stores the value,
/// and returns the heap pointer as i64.
///
/// # Safety
/// - Caller must eventually call `rayzor_box_free` to release the memory.
#[no_mangle]
pub unsafe extern "C" fn rayzor_box_init(value: i64) -> i64 {
    let p = malloc(8);
    if p.is_null() {
        return 0;
    }
    *(p as *mut i64) = value;
    p as i64
}

/// Free a Box allocation.
///
/// # Safety
/// - `box_ptr` must be a valid i64 returned by `rayzor_box_init`.
/// - Must not be called more than once on the same pointer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_box_free(box_ptr: i64) {
    if box_ptr == 0 {
        return;
    }
    free(box_ptr as *mut u8);
}

/// Read the value from a Box (Box.unbox).
///
/// Returns the stored i64 value. Does NOT free the box.
///
/// # Safety
/// - `box_ptr` must be a valid i64 returned by `rayzor_box_init`.
#[no_mangle]
pub unsafe extern "C" fn rayzor_box_unbox(box_ptr: i64) -> i64 {
    if box_ptr == 0 {
        return 0;
    }
    *((box_ptr as usize) as *const i64)
}

/// Get the raw heap address as i64 (Box.raw / Box.asPtr / Box.asRef).
///
/// This is an identity operation — the Box IS the pointer.
#[no_mangle]
pub unsafe extern "C" fn rayzor_box_raw(box_ptr: i64) -> i64 {
    box_ptr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_init_unbox_free() {
        unsafe {
            let boxed = rayzor_box_init(42);
            assert!(boxed != 0);
            assert_eq!(rayzor_box_unbox(boxed), 42);
            assert_eq!(rayzor_box_raw(boxed), boxed);
            rayzor_box_free(boxed);
        }
    }

    #[test]
    fn test_box_null_safety() {
        unsafe {
            assert_eq!(rayzor_box_unbox(0), 0);
            rayzor_box_free(0); // should not crash
        }
    }
}
