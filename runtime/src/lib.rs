//! Rayzor Runtime Library
//!
//! Provides memory management and runtime support for compiled Haxe code.
//! Works with both JIT and AOT compilation.
//!
//! # Architecture
//!
//! - **JIT Mode**: Runtime is linked into the JIT process, functions are called directly
//! - **AOT Mode**: Runtime is statically linked or compiled alongside the output binary
//!
//! # Memory Management
//!
//! Uses Rust's global allocator (`std::alloc::Global`) which is:
//! - Fast and efficient
//! - Memory-safe
//! - Platform-independent
//! - No external C dependencies

// Runtime FFI functions take raw pointers from JIT-compiled code.
// The safety contract is between the compiler (which generates valid pointer arguments)
// and the runtime. Marking every extern "C" function as `unsafe` would be meaningless
// since callers are machine code, not Rust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::missing_safety_doc)]

use std::alloc::{alloc, dealloc, realloc, Layout};
use std::cell::RefCell;
use std::ptr;

// Export Vec module (old API - keeping for backward compat)
pub mod vec;
// String module with return-value style concat
pub mod string;

// Generic Vec<T> runtime - type-specialized vectors for monomorphization
pub mod generic_vec;

// Export Haxe core type runtime modules
pub mod anon_object; // Anonymous object runtime (Arc-based, COW)
pub mod concurrency; // Concurrency primitives (Thread, Arc, Mutex, Channel)
pub mod ereg; // EReg regular expressions (regex crate)
pub mod exception;
pub mod haxe_array; // Dynamic Array API
pub mod haxe_math; // Math functions
pub mod haxe_string; // Comprehensive String API
pub mod haxe_sys; // System/IO functions
pub mod reflect; // Reflect + Type API for anonymous objects
pub mod safety; // Safety validation and error reporting
pub mod type_system; // Runtime type information for Dynamic values
pub mod vec_plugin; // Pointer-based Vec API // Exception handling (setjmp/longjmp)

pub mod plugin_impl; // Plugin registration

// Box<T> runtime — single-owner heap allocation
pub mod box_runtime;

// CString runtime — null-terminated C string interop (rayzor.CString)
pub mod cstring_runtime;

// Tensor runtime — N-dimensional array (rayzor.ds.Tensor)
pub mod tensor;

// TinyCC runtime API (rayzor.runtime.CC)
#[cfg(feature = "tcc-runtime")]
pub mod tinycc_runtime;

// Re-export main types
pub use haxe_array::HaxeArray;
pub use haxe_string::HaxeString;
pub use vec::HaxeVec;

// Re-export generic Vec types
pub use generic_vec::{VecBool, VecF64, VecI32, VecI64, VecPtr};

// Re-export plugin
pub use plugin_impl::get_plugin;

// Feature flags for safety levels
#[cfg(feature = "runtime-safety-checks")]
pub const SAFETY_CHECKS_ENABLED: bool = true;
#[cfg(not(feature = "runtime-safety-checks"))]
pub const SAFETY_CHECKS_ENABLED: bool = false;

// Debug mode always has checks
#[cfg(debug_assertions)]
pub const DEBUG_MODE: bool = true;
#[cfg(not(debug_assertions))]
pub const DEBUG_MODE: bool = false;

/// Allocate memory on the heap
///
/// # Safety
/// The returned pointer must be freed with `rayzor_free` when no longer needed.
///
/// # Arguments
/// * `size` - Number of bytes to allocate
///
/// # Returns
/// Pointer to allocated memory, or null on failure
#[no_mangle]
pub unsafe extern "C" fn rayzor_malloc(size: u64) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }

    // Create layout for allocation
    let layout = match Layout::from_size_align(size as usize, 1) {
        Ok(layout) => layout,
        Err(_) => return ptr::null_mut(),
    };

    // Allocate memory
    let ptr = alloc(layout);

    if ptr.is_null() {
        return ptr::null_mut();
    }

    ptr
}

/// Reallocate memory to a new size
///
/// # Safety
/// - `ptr` must have been allocated by `rayzor_malloc` or `rayzor_realloc`
/// - If reallocation fails, the original pointer remains valid
///
/// # Arguments
/// * `ptr` - Pointer to existing allocation
/// * `old_size` - Original size in bytes
/// * `new_size` - New size in bytes
///
/// # Returns
/// Pointer to reallocated memory, or null on failure
#[no_mangle]
pub unsafe extern "C" fn rayzor_realloc(ptr: *mut u8, old_size: u64, new_size: u64) -> *mut u8 {
    if ptr.is_null() {
        return rayzor_malloc(new_size);
    }

    if new_size == 0 {
        rayzor_free(ptr, old_size);
        return ptr::null_mut();
    }

    // Create layouts
    let old_layout = match Layout::from_size_align(old_size as usize, 1) {
        Ok(layout) => layout,
        Err(_) => return ptr::null_mut(),
    };

    // Reallocate
    let new_ptr = realloc(ptr, old_layout, new_size as usize);

    if new_ptr.is_null() {
        return ptr::null_mut();
    }

    new_ptr
}

/// Free allocated memory
///
/// # Safety
/// - `ptr` must have been allocated by `rayzor_malloc` or `rayzor_realloc`
/// - `size` must match the size used when allocating
/// - After calling this function, `ptr` is invalid and must not be used
///
/// # Arguments
/// * `ptr` - Pointer to memory to free
/// * `size` - Size of the allocation in bytes
#[no_mangle]
pub unsafe extern "C" fn rayzor_free(ptr: *mut u8, size: u64) {
    if ptr.is_null() || size == 0 {
        return;
    }

    // Create layout
    let layout = match Layout::from_size_align(size as usize, 1) {
        Ok(layout) => layout,
        Err(_) => return, // Invalid layout, can't free
    };

    // Deallocate
    dealloc(ptr, layout);
}

// ============================================================================
// Tracked Heap Allocator
// ============================================================================
// Inline-header allocator for JIT-compiled code.
// Uses Rust's global allocator with an 8-byte size header prepended to each
// allocation. This avoids the overhead of a HashMap while still providing
// the Layout info needed by Rust's dealloc.
//
// Memory layout: [size: u64][user data (aligned_size bytes)]
//                           ^-- returned pointer
//
// Double-free protection: after dealloc we can't reliably read the header,
// but we check for obviously invalid sizes (0 or > 1TB) as a safety net.
// The MIR drop-tracking system is the primary double-free prevention mechanism.

/// Header size prepended to each tracked allocation (16 bytes for alignment).
/// Uses 16-byte header to ensure user pointer is 16-byte aligned (required for SIMD).
const TRACKED_HEADER_SIZE: usize = 16;

/// Alignment for tracked allocations (16 bytes for SIMD compatibility).
const TRACKED_ALIGNMENT: usize = 16;

/// Maximum sane allocation size (1 TB) — anything larger is treated as corrupt.
const TRACKED_MAX_SIZE: usize = 1 << 40;

/// Allocate tracked heap memory using Rust's global allocator.
///
/// Compatible with libc malloc signature: fn(size) -> *mut u8
/// Prepends a 16-byte header (size in first 8 bytes, padding in next 8).
/// Returns 16-byte aligned pointer for SIMD compatibility.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tracked_alloc(size: u64) -> *mut u8 {
    if size == 0 {
        return ptr::null_mut();
    }

    // Round up to 16-byte alignment
    let aligned_size = ((size as usize) + (TRACKED_ALIGNMENT - 1)) & !(TRACKED_ALIGNMENT - 1);
    let total = aligned_size + TRACKED_HEADER_SIZE;
    let layout = match Layout::from_size_align(total, TRACKED_ALIGNMENT) {
        Ok(layout) => layout,
        Err(_) => return ptr::null_mut(),
    };

    let base = alloc(layout);
    if base.is_null() {
        return ptr::null_mut();
    }

    // Write size header at base (first 8 bytes of the 16-byte header)
    *(base as *mut u64) = aligned_size as u64;

    base.add(TRACKED_HEADER_SIZE)
}

/// Free tracked heap memory using Rust's global allocator.
///
/// Compatible with libc free signature: fn(*mut u8)
/// Reads the size from the 16-byte header prepended by `rayzor_tracked_alloc`.
/// Rejects obviously invalid sizes as a safety net against double-free.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tracked_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }

    // Read the size header (first 8 bytes of 16-byte header before the user pointer)
    let base = ptr.sub(TRACKED_HEADER_SIZE);
    let aligned_size = *(base as *const u64) as usize;

    // Sanity check: reject zero or impossibly large sizes
    // Zero size indicates this was already freed or never a valid tracked allocation.
    // Sizes > 1TB are clearly corrupt metadata.
    if aligned_size == 0 || aligned_size > TRACKED_MAX_SIZE {
        return;
    }

    // Clear the size header to help catch double-frees
    // (subsequent free of this address will see size=0 and return early)
    *(base as *mut u64) = 0;

    let total = aligned_size + TRACKED_HEADER_SIZE;
    let layout = Layout::from_size_align_unchecked(total, TRACKED_ALIGNMENT);
    dealloc(base, layout);
}

/// Reallocate tracked heap memory using Rust's global allocator.
///
/// Compatible with libc realloc signature: fn(*mut u8, u64) -> *mut u8
/// Handles the 16-byte size header correctly by allocating a new block,
/// copying data, and freeing the old block.
#[no_mangle]
pub unsafe extern "C" fn rayzor_tracked_realloc(ptr: *mut u8, new_size: u64) -> *mut u8 {
    if ptr.is_null() {
        return rayzor_tracked_alloc(new_size);
    }
    if new_size == 0 {
        rayzor_tracked_free(ptr);
        return ptr::null_mut();
    }

    // Read old size from header
    let old_base = ptr.sub(TRACKED_HEADER_SIZE);
    let old_aligned_size = *(old_base as *const u64) as usize;

    // Sanity check old size
    if old_aligned_size == 0 || old_aligned_size > TRACKED_MAX_SIZE {
        // Corrupt or already freed — just do a fresh alloc
        return rayzor_tracked_alloc(new_size);
    }

    let new_aligned_size =
        ((new_size as usize) + (TRACKED_ALIGNMENT - 1)) & !(TRACKED_ALIGNMENT - 1);

    // If same size, nothing to do
    if new_aligned_size == old_aligned_size {
        return ptr;
    }

    // Allocate new block with header
    let new_ptr = rayzor_tracked_alloc(new_size);
    if new_ptr.is_null() {
        return ptr::null_mut();
    }

    // Copy old data (up to the smaller of old and new sizes)
    let copy_size = old_aligned_size.min(new_aligned_size);
    ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);

    // Free old block
    rayzor_tracked_free(ptr);

    new_ptr
}

/// Initialize RTTI (Runtime Type Information) for user-defined types.
///
/// Note: Primitive types (Int, Float, Bool, String, etc.) are automatically
/// registered on first access via lazy initialization. This function is only
/// needed if you want to register custom user-defined types before they're
/// accessed, or to force early initialization.
pub fn init_rtti() {
    type_system::init_type_system();
}

// ============================================================================
// Global Variable Storage
// ============================================================================
// Thread-local storage for global variables used by JIT-compiled code.
// Global IDs are small sequential u32 values, so we use a flat Vec for O(1) lookup
// instead of HashMap (avoids SipHash overhead on every access).

/// Initial capacity for the global store (grows as needed)
const GLOBAL_STORE_INITIAL_CAP: usize = 64;

thread_local! {
    static GLOBAL_STORE: RefCell<Vec<u64>> = RefCell::new(vec![0; GLOBAL_STORE_INITIAL_CAP]);
}

/// Store a value to a global variable
///
/// # Arguments
/// * `global_id` - The global variable ID (small sequential integer)
/// * `value` - The value to store (as a raw pointer cast to i64)
#[no_mangle]
pub unsafe extern "C" fn rayzor_global_store(global_id: i64, value: i64) {
    GLOBAL_STORE.with(|store| {
        let mut s = store.borrow_mut();
        let idx = global_id as usize;
        if idx >= s.len() {
            s.resize(idx + 1, 0);
        }
        s[idx] = value as u64;
    });
}

/// Load a value from a global variable
///
/// # Arguments
/// * `global_id` - The global variable ID (small sequential integer)
///
/// # Returns
/// The stored value, or 0 if not found
#[no_mangle]
pub unsafe extern "C" fn rayzor_global_load(global_id: i64) -> i64 {
    GLOBAL_STORE.with(|store| {
        let s = store.borrow();
        let idx = global_id as usize;
        if idx < s.len() {
            s[idx] as i64
        } else {
            0
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malloc_free() {
        unsafe {
            let ptr = rayzor_malloc(100);
            assert!(!ptr.is_null());

            // Write some data
            *ptr = 42;
            assert_eq!(*ptr, 42);

            rayzor_free(ptr, 100);
        }
    }

    #[test]
    fn test_realloc() {
        unsafe {
            // Allocate 10 bytes
            let ptr = rayzor_malloc(10);
            assert!(!ptr.is_null());

            // Write data
            for i in 0..10 {
                *ptr.add(i) = i as u8;
            }

            // Reallocate to 20 bytes
            let new_ptr = rayzor_realloc(ptr, 10, 20);
            assert!(!new_ptr.is_null());

            // Check that old data is preserved
            for i in 0..10 {
                assert_eq!(*new_ptr.add(i), i as u8);
            }

            rayzor_free(new_ptr, 20);
        }
    }

    #[test]
    fn test_zero_size() {
        unsafe {
            let ptr = rayzor_malloc(0);
            assert!(ptr.is_null());
        }
    }

    #[test]
    fn test_realloc_null() {
        unsafe {
            // Realloc with null ptr should act like malloc
            let ptr = rayzor_realloc(ptr::null_mut(), 0, 100);
            assert!(!ptr.is_null());
            rayzor_free(ptr, 100);
        }
    }
}
