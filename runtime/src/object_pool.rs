//! Size-class pool for small heap objects.
//!
//! Objects were allocated one `malloc` at a time. On an allocation-heavy
//! program that dominates everything else: the tree benchmark makes ~30M of
//! them, and at ~30ns each the allocator alone accounts for most of the run.
//! Nothing is reused either, so every page is faulted in fresh and never
//! touched again.
//!
//! Allocation here is a free-list pop, or a pointer bump when the list is
//! empty. Freeing pushes the block back. Both are a handful of instructions
//! and neither enters libc.
//!
//! # Finding a block's size class
//!
//! Blocks carry no header — a per-object size word would add 8 bytes to a
//! 32-byte node, and the whole point is density. Instead chunks are carved
//! from one reserved region at a fixed alignment, so masking a pointer down to
//! its chunk gives the header, and "did this come from the pool" is a range
//! check against the region. That matters for correctness, not just speed:
//! the compiler frees pointers that came from libc as well, and reading a
//! header off one of those would be a wild load. Outside the region, the
//! pointer goes straight back to libc.
//!
//! # Threads
//!
//! Free lists are per thread, but blocks are NOT owned by the thread that
//! allocated them. A block freed on another thread joins that thread's list
//! and is reused there. Memory migrates, which is harmless -- a 32-byte block
//! is a 32-byte block. The alternative, routing a free back to an owning
//! thread, buys nothing and needs the owner recorded and consulted.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Chunk size and alignment. A pointer masked to this boundary is its header.
const CHUNK_SIZE: usize = 1 << 20;
/// Largest request served from the pool; above this, libc.
const MAX_POOLED: usize = 512;
/// 16-byte granularity keeps waste under one word for typical objects.
const GRANULE: usize = 16;
const NUM_CLASSES: usize = MAX_POOLED / GRANULE;
/// Reserved address space. Untouched pages cost nothing.
const REGION_SIZE: usize = 24 << 30;
/// Objects start past the header, on a cache line.
const CHUNK_DATA_OFFSET: usize = 64;

const MAGIC: u64 = 0x5259_5A52_4F42_4A00;

#[repr(C)]
struct ChunkHeader {
    magic: u64,
    class_index: usize,
}

static REGION_BASE: AtomicUsize = AtomicUsize::new(0);
static REGION_NEXT: AtomicUsize = AtomicUsize::new(0);
static REGION_END: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Head of each class's free list. Blocks are linked through their own
    /// first word, which is dead while the block is free.
    static FREE_LISTS: RefCell<[*mut u8; NUM_CLASSES]> =
        const { RefCell::new([std::ptr::null_mut(); NUM_CLASSES]) };
    /// Un-handed-out remainder of the current chunk per class: (next, end).
    static BUMP: RefCell<[(usize, usize); NUM_CLASSES]> =
        const { RefCell::new([(0, 0); NUM_CLASSES]) };
}

/// Reserve the region on first use. Failure is not fatal: every path falls
/// back to libc, so the pool simply stays empty.
fn ensure_region() -> bool {
    let base = REGION_BASE.load(Ordering::Acquire);
    if base != 0 {
        return base != usize::MAX;
    }
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            REGION_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        REGION_BASE.store(usize::MAX, Ordering::Release);
        return false;
    }
    // Round the first chunk up to the alignment the masking relies on.
    let addr = ptr as usize;
    let first = addr.next_multiple_of(CHUNK_SIZE);
    match REGION_BASE.compare_exchange(0, addr, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => {
            REGION_NEXT.store(first, Ordering::Release);
            REGION_END.store(addr + REGION_SIZE, Ordering::Release);
            true
        }
        Err(_) => {
            // Another thread won; give this reservation back.
            unsafe { libc::munmap(ptr, REGION_SIZE) };
            REGION_BASE.load(Ordering::Acquire) != usize::MAX
        }
    }
}

fn in_region(addr: usize) -> bool {
    let base = REGION_BASE.load(Ordering::Relaxed);
    base != 0 && base != usize::MAX && addr >= base && addr < REGION_END.load(Ordering::Relaxed)
}

/// Carve a fresh chunk for `class_index`, returning its usable span.
fn new_chunk(class_index: usize) -> Option<(usize, usize)> {
    if !ensure_region() {
        return None;
    }
    let chunk = REGION_NEXT.fetch_add(CHUNK_SIZE, Ordering::AcqRel);
    if chunk + CHUNK_SIZE > REGION_END.load(Ordering::Relaxed) {
        return None;
    }
    unsafe {
        let header = chunk as *mut ChunkHeader;
        (*header).magic = MAGIC;
        (*header).class_index = class_index;
    }
    Some((chunk + CHUNK_DATA_OFFSET, chunk + CHUNK_SIZE))
}

#[inline]
fn class_of(size: usize) -> Option<usize> {
    if size == 0 || size > MAX_POOLED {
        return None;
    }
    Some((size + GRANULE - 1) / GRANULE - 1)
}

/// Allocate `size` bytes for a heap object.
///
/// The contents are NOT zeroed. Compiler-emitted allocation writes every field
/// before use, and a reused block would otherwise be cleared twice.
#[no_mangle]
pub extern "C" fn rayzor_object_alloc(size: u64) -> *mut u8 {
    let size = size as usize;
    let Some(class_index) = class_of(size) else {
        return unsafe { libc::malloc(size.max(1)) as *mut u8 };
    };
    let block_size = (class_index + 1) * GRANULE;

    let popped = FREE_LISTS.with(|lists| {
        let mut lists = lists.borrow_mut();
        let head = lists[class_index];
        if head.is_null() {
            return std::ptr::null_mut();
        }
        lists[class_index] = unsafe { *(head as *const *mut u8) };
        head
    });
    if !popped.is_null() {
        return popped;
    }

    BUMP.with(|bump| {
        let mut bump = bump.borrow_mut();
        let (next, end) = bump[class_index];
        if next + block_size <= end {
            bump[class_index] = (next + block_size, end);
            return next as *mut u8;
        }
        match new_chunk(class_index) {
            Some((start, chunk_end)) => {
                bump[class_index] = (start + block_size, chunk_end);
                start as *mut u8
            }
            None => unsafe { libc::malloc(block_size) as *mut u8 },
        }
    })
}

/// Return a block to the pool, or to libc if it never came from one.
#[no_mangle]
pub extern "C" fn rayzor_object_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let addr = ptr as usize;
    if !in_region(addr) {
        unsafe { libc::free(ptr as *mut libc::c_void) };
        return;
    }
    let chunk = addr & !(CHUNK_SIZE - 1);
    let header = chunk as *const ChunkHeader;
    let (magic, class_index) = unsafe { ((*header).magic, (*header).class_index) };
    if magic != MAGIC || class_index >= NUM_CLASSES {
        // Inside the reservation but not a chunk we wrote: refuse rather than
        // guess. Handing it to libc would be worse.
        return;
    }
    FREE_LISTS.with(|lists| {
        let mut lists = lists.borrow_mut();
        unsafe { *(ptr as *mut *mut u8) = lists[class_index] };
        lists[class_index] = ptr;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_a_freed_block() {
        let a = rayzor_object_alloc(32);
        assert!(!a.is_null());
        rayzor_object_free(a);
        let b = rayzor_object_alloc(32);
        assert_eq!(a, b, "a freed block should come straight back");
        rayzor_object_free(b);
    }

    #[test]
    fn separates_size_classes() {
        let small = rayzor_object_alloc(16);
        let large = rayzor_object_alloc(256);
        assert_ne!(small, large);
        rayzor_object_free(small);
        rayzor_object_free(large);
        // A 16-byte request must not be served the 256-byte block.
        let again = rayzor_object_alloc(16);
        assert_eq!(again, small);
    }

    #[test]
    fn oversized_goes_to_libc_and_back() {
        let big = rayzor_object_alloc(4096);
        assert!(!big.is_null());
        unsafe { *big = 7 };
        rayzor_object_free(big);
    }

    #[test]
    fn frees_a_foreign_pointer_without_touching_the_pool() {
        let foreign = unsafe { libc::malloc(64) as *mut u8 };
        rayzor_object_free(foreign);
        let fresh = rayzor_object_alloc(64);
        assert_ne!(fresh, foreign);
        rayzor_object_free(fresh);
    }
}
