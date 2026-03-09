//! Bump arena allocator for HaxeString structs and string data.
//!
//! Haxe strings are immutable value types — every operation (concat, substr, etc.)
//! creates a new allocation. Individual frees are impossible to get right without
//! reference counting. Instead, all string memory is bump-allocated from an arena
//! and freed in bulk when execution ends.
//!
//! Two allocation pools:
//! - **Data pool**: raw byte buffers for string content (alignment 1)
//! - **Struct pool**: `HaxeString` structs returned as `*mut HaxeString` (alignment 8)
//!
//! Performance: allocation is O(1) — just a pointer bump + overflow check.
//! No per-object free, no atomic refcount ops, no fragmentation.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::ptr;

use crate::haxe_string::HaxeString;

/// Default chunk size: 64KB
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// A chunk of contiguous memory
struct Chunk {
    ptr: *mut u8,
    size: usize,
}

impl Drop for Chunk {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.size, 8);
                dealloc(self.ptr, layout);
            }
        }
    }
}

/// Bump allocator arena
struct Arena {
    /// All allocated chunks (kept alive for bulk deallocation)
    chunks: Vec<Chunk>,
    /// Current write position in the active chunk
    cursor: *mut u8,
    /// Bytes remaining in the active chunk
    remaining: usize,
}

impl Arena {
    fn new() -> Self {
        Self {
            chunks: Vec::new(),
            cursor: ptr::null_mut(),
            remaining: 0,
        }
    }

    /// Allocate `size` bytes with the given alignment from the arena.
    /// Returns a pointer to the allocated memory. Never fails (panics on OOM).
    fn alloc(&mut self, size: usize, align: usize) -> *mut u8 {
        // Align the cursor
        let aligned = self.align_cursor(align);
        let needed = size + (aligned as usize - self.cursor as usize);

        if needed <= self.remaining {
            let result = aligned;
            self.cursor = unsafe { aligned.add(size) };
            self.remaining -= needed;
            return result;
        }

        // Need a new chunk
        let chunk_size = DEFAULT_CHUNK_SIZE.max(size + align);
        self.grow(chunk_size);

        // Re-align in the new chunk (should be already aligned since chunks are 8-aligned)
        let aligned = self.align_cursor(align);
        let result = aligned;
        let waste = aligned as usize - self.cursor as usize;
        self.cursor = unsafe { aligned.add(size) };
        self.remaining -= size + waste;
        result
    }

    /// Align cursor up to the given alignment, returning the aligned pointer
    fn align_cursor(&self, align: usize) -> *mut u8 {
        if self.cursor.is_null() {
            return ptr::null_mut();
        }
        let addr = self.cursor as usize;
        let aligned = (addr + align - 1) & !(align - 1);
        aligned as *mut u8
    }

    /// Allocate a new chunk and make it the active one
    fn grow(&mut self, size: usize) {
        unsafe {
            let layout = Layout::from_size_align_unchecked(size, 8);
            let ptr = alloc(layout);
            if ptr.is_null() {
                panic!("StringArena: failed to allocate {} bytes", size);
            }
            self.chunks.push(Chunk { ptr, size });
            self.cursor = ptr;
            self.remaining = size;
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // Chunks are dropped automatically via Vec<Chunk> drop
    }
}

thread_local! {
    static STRING_ARENA: RefCell<Arena> = RefCell::new(Arena::new());
}

/// Allocate `size` bytes for string data (alignment 1).
/// The returned pointer is valid for the lifetime of the arena (until program exit).
pub fn arena_alloc_bytes(size: usize) -> *mut u8 {
    if size == 0 {
        // Return a non-null dangling pointer for zero-size allocations
        return std::ptr::NonNull::dangling().as_ptr();
    }
    STRING_ARENA.with(|arena| arena.borrow_mut().alloc(size, 1))
}

/// Allocate a `HaxeString` struct on the arena and return a stable pointer.
/// Equivalent to `Box::into_raw(Box::new(hs))` but without individual heap allocation.
pub fn arena_alloc_haxe_string(hs: HaxeString) -> *mut HaxeString {
    STRING_ARENA.with(|arena| {
        let ptr = arena.borrow_mut().alloc(
            std::mem::size_of::<HaxeString>(),
            std::mem::align_of::<HaxeString>(),
        );
        unsafe {
            ptr::write(ptr as *mut HaxeString, hs);
        }
        ptr as *mut HaxeString
    })
}

/// Statistics about arena usage (for debugging/profiling)
#[allow(dead_code)]
pub fn arena_stats() -> (usize, usize) {
    STRING_ARENA.with(|arena| {
        let a = arena.borrow();
        let total: usize = a.chunks.iter().map(|c| c.size).sum();
        let remaining = a.remaining;
        (total, total - remaining)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc_bytes() {
        let p1 = arena_alloc_bytes(16);
        assert!(!p1.is_null());
        let p2 = arena_alloc_bytes(32);
        assert!(!p2.is_null());
        assert_ne!(p1, p2);
    }

    #[test]
    fn test_arena_alloc_haxe_string() {
        let hs = HaxeString {
            ptr: ptr::null_mut(),
            len: 0,
            cap: 0,
        };
        let p = arena_alloc_haxe_string(hs);
        assert!(!p.is_null());
        unsafe {
            assert_eq!((*p).len, 0);
            assert_eq!((*p).cap, 0);
        }
    }

    #[test]
    fn test_arena_grows() {
        // Allocate more than one chunk
        for _ in 0..10000 {
            let p = arena_alloc_bytes(64);
            assert!(!p.is_null());
        }
    }

    #[test]
    fn test_arena_alignment() {
        // HaxeString needs 8-byte alignment
        for _ in 0..100 {
            let p = arena_alloc_haxe_string(HaxeString {
                ptr: ptr::null_mut(),
                len: 42,
                cap: 0,
            });
            assert_eq!((p as usize) % 8, 0, "HaxeString not 8-byte aligned");
            unsafe {
                assert_eq!((*p).len, 42);
            }
        }
    }
}
