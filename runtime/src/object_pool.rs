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
/// Address space the pool carves chunks from.
///
/// This is a reservation, not memory: pages cost nothing until touched. It is
/// sized well past any real working set so chunk addresses stay maskable, but
/// not so far past that a host refuses the mapping outright.
const REGION_SIZE: usize = 8 << 30;
/// Objects start past the header, on a cache line.
const CHUNK_DATA_OFFSET: usize = 64;

const MAGIC: u64 = 0x5259_5A52_4F42_4A00;

#[repr(C)]
struct ChunkHeader {
    magic: u64,
    class_index: usize,
}

/// Blocks served from the pool, for proving compiled code reaches it.
///
/// The unit tests below prove the pool works; they say nothing about whether
/// anything USES it. A backend reverting to libc's malloc would leave every
/// one of them passing, against an allocator no program touches. This counter
/// makes that a fact a test can assert instead of something inferred from a
/// bug happening to still be visible.
static POOL_SERVED: AtomicUsize = AtomicUsize::new(0);

/// How many blocks the pool has served this process.
#[no_mangle]
pub extern "C" fn rayzor_pool_served() -> u64 {
    POOL_SERVED.load(Ordering::Relaxed) as u64
}

static REGION_BASE: AtomicUsize = AtomicUsize::new(0);
static REGION_NEXT: AtomicUsize = AtomicUsize::new(0);
static REGION_END: AtomicUsize = AtomicUsize::new(0);

/// Per class: the free-list head, and the un-handed-out remainder of the
/// current chunk as `(next, end)`.
struct ClassState {
    free: *mut u8,
    next: usize,
    end: usize,
}

thread_local! {
    /// One thread-local, not two. Allocation is a handful of instructions, so
    /// a second TLS lookup is a measurable share of it -- and the two were
    /// always consulted together anyway. Blocks on the free list are linked
    /// through their own first word, which is dead while the block is free.
    static CLASSES: RefCell<[ClassState; NUM_CLASSES]> = const {
        RefCell::new(
            [const {
                ClassState {
                    free: std::ptr::null_mut(),
                    next: 0,
                    end: 0,
                }
            }; NUM_CLASSES],
        )
    };
}

/// Reserve the region on first use. Failure is not fatal: every path falls
/// back to libc, so the pool simply stays empty.
#[cfg(not(unix))]
fn ensure_region() -> bool {
    // No mmap here, and the pool is an optimisation rather than a requirement:
    // every allocation path already falls back to libc when the region is
    // absent, which is exactly what a failed reservation does on unix. Taking
    // that path deliberately costs a pool, not correctness -- where reserving
    // address space by hand would cost a second allocator to maintain.
    REGION_BASE.store(usize::MAX, Ordering::Release);
    false
}

#[cfg(unix)]
fn ensure_region() -> bool {
    let base = REGION_BASE.load(Ordering::Acquire);
    if base != 0 {
        return base != usize::MAX;
    }
    // MAP_NORESERVE, because the region is address space rather than memory.
    // Without it a host that accounts commit up front sees a reservation far
    // larger than its RAM and refuses the mapping -- and a refused reservation
    // is the worst outcome available: every allocation still pays the check on
    // its way to the same libc call it would have made anyway.
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            REGION_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        // Say so. A pool that silently is not there looks exactly like a pool
        // that is there and does not help.
        if std::env::var_os("RZT_POOL_QUIET").is_none() {
            eprintln!(
                "[pool] could not reserve {} GiB of address space ({}); using libc",
                REGION_SIZE >> 30,
                std::io::Error::last_os_error()
            );
        }
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
    Some(size.div_ceil(GRANULE) - 1)
}

/// Allocate `size` bytes for a heap object.
///
/// The contents are NOT zeroed. Compiler-emitted allocation writes every field
/// before use, and a reused block would otherwise be cleared twice.
#[no_mangle]
pub extern "C" fn rayzor_object_alloc(size: u64) -> *mut u8 {
    let size = size as usize;
    if !pool_enabled() {
        return unsafe { libc::malloc(size.max(1)) as *mut u8 };
    }
    let Some(class_index) = class_of(size) else {
        return unsafe { libc::malloc(size.max(1)) as *mut u8 };
    };
    let block_size = (class_index + 1) * GRANULE;

    let ptr = CLASSES.with(|classes| {
        let mut classes = classes.borrow_mut();
        let state = &mut classes[class_index];

        let head = state.free;
        if !head.is_null() {
            state.free = unsafe { *(head as *const *mut u8) };
            return head;
        }
        if state.next + block_size <= state.end {
            let p = state.next as *mut u8;
            state.next += block_size;
            return p;
        }
        match new_chunk(class_index) {
            Some((start, chunk_end)) => {
                state.next = start + block_size;
                state.end = chunk_end;
                start as *mut u8
            }
            None => std::ptr::null_mut(),
        }
    });
    if ptr.is_null() {
        return unsafe { libc::malloc(block_size) as *mut u8 };
    }

    // Counting and filling are for the tests that prove compiled code reaches
    // this allocator and that a fresh block is distinguishable from a written
    // one. Both sit behind the same flag: an atomic increment per allocation
    // costs more than the allocation, and the default path must not pay for a
    // check nobody asked for -- which is the mistake that put two clock reads
    // on every virtual dispatch.
    if scribble_enabled() {
        unsafe { std::ptr::write_bytes(ptr, 0xAA, block_size) };
        POOL_SERVED.fetch_add(1, Ordering::Relaxed);
    }
    ptr
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
    if scribble_enabled() {
        // The freed-memory oracle poisons through libc's MallocScribble, which
        // cannot see pool blocks -- they come from mmap and libc never touches
        // them. Without this the oracle keeps passing on a fixture that still
        // uses libc while being blind to every object the pool now owns, which
        // is worse than having no oracle: it reports itself working.
        let block_size = (class_index + 1) * GRANULE;
        unsafe { std::ptr::write_bytes(ptr, 0x55, block_size) };
    }
    CLASSES.with(|classes| {
        let mut classes = classes.borrow_mut();
        let state = &mut classes[class_index];
        unsafe { *(ptr as *mut *mut u8) = state.free };
        state.free = ptr;
    });
}

/// Whether to serve from the pool at all. OFF unless `RZT_POOL=1`.
///
/// Default-off because the only measurement on the platform CI runs says the
/// pool LOSES there: binarytrees execute went 949ms to 1443ms on x86_64 while
/// the same change took 0.66s to 0.42s of user CPU on aarch64 macOS. glibc's
/// malloc has a per-thread cache and macOS's does not, so beating one system
/// allocator says nothing about the other. Some of that gap was overhead this
/// allocator has since shed, but until it is re-measured on x86_64 the honest
/// default is the allocator that is known not to regress.
///
/// Being a switch rather than a build flag is what makes that re-measurement
/// cheap: both arms run on one binary and can be interleaved, which is the
/// only comparison a contended machine supports.
///
/// The free path deliberately does NOT consult this. Blocks handed out while
/// the pool was on must still come back to it, or flipping the switch mid-
/// process would hand pooled memory to libc.
fn pool_enabled() -> bool {
    #[cfg(test)]
    if TEST_POOL.load(Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On, unless RZT_POOL says otherwise. It was off while a refused region
    // reservation made it look like a slowdown -- the mapping failed and every
    // allocation paid the check on its way to libc anyway. With the reservation
    // granted it serves 32-byte objects in a pop or a bump, and on a memory
    // constrained host the allocation-heavy benchmark completes with it and is
    // killed without it.
    *ON.get_or_init(|| std::env::var("RZT_POOL").as_deref() != Ok("0"))
}

/// Tests drive the pool through this, not the environment: the answer above is
/// cached for the process, so an env var would make every test depend on which
/// one ran first.
#[cfg(test)]
static TEST_POOL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether to poison blocks. Resolved once; the check is a bool read on a path
/// that runs per allocation and per free.
///
/// Tests cannot drive this through the environment: the answer is cached for
/// the process, so whichever test ran first would decide it for all of them
/// and the rest would pass or fail on ordering rather than on behaviour. They
/// set the override instead.
fn scribble_enabled() -> bool {
    #[cfg(test)]
    if TEST_SCRIBBLE.load(Ordering::Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("MallocScribble").is_some()
            || std::env::var_os("RZT_POOL_SCRIBBLE").is_some()
    })
}

#[cfg(test)]
static TEST_SCRIBBLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
mod tests {
    use super::*;

    // The pool currently reserves its backing region with mmap. Non-Unix
    // targets deliberately fall back to libc instead.
    #[cfg(unix)]
    #[test]
    fn reuses_a_freed_block() {
        let _g = pooled();
        let a = rayzor_object_alloc(32);
        assert!(!a.is_null());
        rayzor_object_free(a);
        let b = rayzor_object_alloc(32);
        assert_eq!(a, b, "a freed block should come straight back");
        rayzor_object_free(b);
    }

    #[cfg(unix)]
    #[test]
    fn separates_size_classes() {
        let _g = pooled();
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
        let _g = pooled();
        let big = rayzor_object_alloc(4096);
        assert!(!big.is_null());
        unsafe { *big = 7 };
        rayzor_object_free(big);
    }

    /// The scribble flag and the served counter are process-global, so the
    /// tests that touch them cannot run beside each other and be about
    /// anything.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Take the serial lock and turn the pool on. Every test here is about the
    /// pool, and the pool is off by default.
    fn pooled() -> std::sync::MutexGuard<'static, ()> {
        let g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        TEST_POOL.store(true, Ordering::Relaxed);
        g
    }

    #[cfg(unix)]
    #[test]
    fn poisons_a_released_block_when_asked() {
        let _g = pooled();
        TEST_SCRIBBLE.store(true, Ordering::Relaxed);
        let a = rayzor_object_alloc(64);
        unsafe { std::ptr::write_bytes(a, 0x27, 64) };
        rayzor_object_free(a);
        // The link word occupies the first 8 bytes of a free block; the rest
        // must read as poison, not as what the object used to hold.
        let tail = unsafe { std::slice::from_raw_parts(a.add(8), 56) };
        assert!(
            tail.iter().all(|b| *b == 0x55),
            "released block kept its contents; the oracle would see nothing"
        );
        TEST_SCRIBBLE.store(false, Ordering::Relaxed);
    }

    #[cfg(unix)]
    #[test]
    fn fills_a_fresh_block_when_asked() {
        let _g = pooled();
        TEST_SCRIBBLE.store(true, Ordering::Relaxed);
        let a = rayzor_object_alloc(48);
        let bytes = unsafe { std::slice::from_raw_parts(a, 48) };
        assert!(
            bytes.iter().all(|b| *b == 0xAA),
            "a fresh block read as zero; a field nothing wrote would look \
             deliberately initialised"
        );
        rayzor_object_free(a);
        TEST_SCRIBBLE.store(false, Ordering::Relaxed);
    }

    #[cfg(unix)]
    #[test]
    fn counts_only_what_it_actually_serves() {
        let _g = pooled();
        // The counter rides the same flag as the fill: an atomic increment per
        // allocation is not something the default path should pay for.
        TEST_SCRIBBLE.store(true, Ordering::Relaxed);
        let before = rayzor_pool_served();
        let a = rayzor_object_alloc(32);
        assert_eq!(rayzor_pool_served(), before + 1);
        // Above the cap is libc's, and must not be counted as the pool's.
        let big = rayzor_object_alloc((MAX_POOLED + 1) as u64);
        assert_eq!(
            rayzor_pool_served(),
            before + 1,
            "an oversized request went to libc but was counted as pooled"
        );
        rayzor_object_free(a);
        rayzor_object_free(big);
        TEST_SCRIBBLE.store(false, Ordering::Relaxed);
    }

    #[cfg(unix)]
    #[test]
    fn free_still_reclaims_pool_blocks_when_the_switch_is_off() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        // A block handed out by the pool must return to the pool even if the
        // switch is consulted again later; routing it to libc would be a free
        // of memory libc never allocated.
        TEST_POOL.store(true, Ordering::Relaxed);
        let a = rayzor_object_alloc(32);
        assert!(in_region(a as usize), "expected a pooled block");
        rayzor_object_free(a);
        let b = rayzor_object_alloc(32);
        assert_eq!(a, b, "the released block did not come back");
        rayzor_object_free(b);
    }

    #[test]
    fn frees_a_foreign_pointer_without_touching_the_pool() {
        let _g = pooled();
        let foreign = unsafe { libc::malloc(64) as *mut u8 };
        rayzor_object_free(foreign);
        let fresh = rayzor_object_alloc(64);
        assert_ne!(fresh, foreign);
        rayzor_object_free(fresh);
    }

    #[cfg(windows)]
    #[test]
    fn windows_falls_back_to_libc_when_pool_is_requested() {
        let _g = pooled();
        let before = rayzor_pool_served();
        let ptr = rayzor_object_alloc(32);

        assert!(!ptr.is_null());
        assert!(!in_region(ptr as usize));
        assert_eq!(rayzor_pool_served(), before);

        unsafe { *ptr = 0x27 };
        assert_eq!(unsafe { *ptr }, 0x27);
        rayzor_object_free(ptr);
    }
}
