//! Apple Silicon JIT Memory Manager
//!
//! Custom memory management for JIT code on ARM64/macOS (Apple Silicon).
//! This module provides proper handling of:
//!
//! - MAP_JIT memory allocation for JIT code
//! - W^X (Write XOR Execute) protection toggling
//! - Instruction cache invalidation for multi-core coherency
//!
//! ## Why Custom Memory Management?
//!
//! Apple Silicon has unique requirements for JIT compilation:
//!
//! 1. **MAP_JIT flag**: Memory must be allocated with MAP_JIT to allow
//!    runtime code generation under Hardened Runtime.
//!
//! 2. **W^X Protection**: Memory can be writable OR executable, never both.
//!    Use `pthread_jit_write_protect_np` to toggle between modes.
//!
//! 3. **Icache Coherency**: ARM64 has separate instruction and data caches.
//!    After writing code, must invalidate icache before execution.
//!
//! 4. **P/E Core Scheduling**: Code may execute on Performance or Efficiency
//!    cores with different cache hierarchies.

use std::collections::BTreeMap;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Page size on Apple Silicon (16KB for efficiency, though 4KB also works)
const PAGE_SIZE: usize = 16 * 1024;

/// Default region size (4MB - enough for most JIT compilations)
const DEFAULT_REGION_SIZE: usize = 4 * 1024 * 1024;

// FFI declarations for Apple Silicon JIT support
extern "C" {
    /// Toggle JIT write protection for the current thread.
    /// - 0: Disable write protection (memory is writable, not executable)
    /// - 1: Enable write protection (memory is executable, not writable)
    fn pthread_jit_write_protect_np(enabled: i32);

    /// Invalidate instruction cache for a memory region.
    /// Must be called after writing code and before executing.
    fn sys_icache_invalidate(start: *const std::ffi::c_void, size: usize);
}

// libc constants for mmap
const PROT_READ: i32 = 0x01;
const PROT_WRITE: i32 = 0x02;
const PROT_EXEC: i32 = 0x04;
const MAP_PRIVATE: i32 = 0x0002;
const MAP_ANONYMOUS: i32 = 0x1000;
const MAP_JIT: i32 = 0x0800;
const MAP_FAILED: *mut std::ffi::c_void = !0 as *mut std::ffi::c_void;

extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;

    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
}

/// A region of JIT-executable memory
struct JITRegion {
    /// Base address of the region
    base: *mut u8,
    /// Total size of the region
    size: usize,
    /// Current allocation offset within the region
    offset: AtomicUsize,
}

impl JITRegion {
    /// Create a new JIT memory region
    fn new(size: usize) -> Result<Self, String> {
        let aligned_size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        unsafe {
            // Allocate memory with MAP_JIT flag
            let ptr = mmap(
                ptr::null_mut(),
                aligned_size,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT,
                -1,
                0,
            );

            if ptr == MAP_FAILED {
                return Err(format!(
                    "Failed to allocate JIT memory region of size {}",
                    aligned_size
                ));
            }

            Ok(Self {
                base: ptr as *mut u8,
                size: aligned_size,
                offset: AtomicUsize::new(0),
            })
        }
    }

    /// Allocate space within this region (bump allocator)
    fn allocate(&self, size: usize, align: usize) -> Option<*mut u8> {
        loop {
            let current = self.offset.load(Ordering::Relaxed);
            let aligned_offset = (current + align - 1) & !(align - 1);
            let new_offset = aligned_offset + size;

            if new_offset > self.size {
                return None; // Region full
            }

            if self
                .offset
                .compare_exchange_weak(current, new_offset, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Some(unsafe { self.base.add(aligned_offset) });
            }
        }
    }

    /// Get remaining capacity
    #[allow(dead_code)]
    fn remaining(&self) -> usize {
        self.size - self.offset.load(Ordering::Relaxed)
    }
}

impl Drop for JITRegion {
    fn drop(&mut self) {
        unsafe {
            munmap(self.base as *mut std::ffi::c_void, self.size);
        }
    }
}

// Safety: JITRegion uses atomic operations for thread-safe allocation
unsafe impl Send for JITRegion {}
unsafe impl Sync for JITRegion {}

/// Function entry in the JIT memory manager
#[derive(Clone)]
pub struct JITFunction {
    /// Pointer to the function code
    pub ptr: *const u8,
    /// Size of the function code
    pub size: usize,
}

// Safety: Function pointers are safe to share across threads once written
unsafe impl Send for JITFunction {}
unsafe impl Sync for JITFunction {}

/// Apple Silicon JIT Memory Manager
///
/// Manages JIT code memory with proper Apple Silicon handling:
/// - Allocates memory with MAP_JIT
/// - Toggles W^X protection correctly
/// - Invalidates icache for multi-core coherency
pub struct AppleSiliconJITMemory {
    /// Memory regions for JIT code
    regions: Vec<JITRegion>,
    /// Registered functions by name
    functions: BTreeMap<String, JITFunction>,
    /// Whether we're currently in write mode
    write_mode: bool,
}

impl AppleSiliconJITMemory {
    /// Create a new JIT memory manager
    pub fn new() -> Result<Self, String> {
        let initial_region = JITRegion::new(DEFAULT_REGION_SIZE)?;

        Ok(Self {
            regions: vec![initial_region],
            functions: BTreeMap::new(),
            write_mode: false,
        })
    }

    /// Enable write mode (disables execution)
    pub fn begin_write(&mut self) {
        if !self.write_mode {
            unsafe {
                pthread_jit_write_protect_np(0); // Disable write protection = writable
            }
            self.write_mode = true;
        }
    }

    /// Enable execute mode (disables writing)
    /// Also invalidates icache for all written code
    pub fn end_write(&mut self) {
        if self.write_mode {
            unsafe {
                // Data synchronization barrier - ensure all writes complete
                std::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags));

                // Invalidate icache for all regions
                for region in &self.regions {
                    let used_size = region.offset.load(Ordering::SeqCst);
                    if used_size > 0 {
                        sys_icache_invalidate(region.base as *const std::ffi::c_void, used_size);
                    }
                }

                // Enable write protection = executable
                pthread_jit_write_protect_np(1);

                // Full memory barriers after protection change
                std::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags));
                std::arch::asm!("isb sy", options(nomem, nostack, preserves_flags));
            }
            self.write_mode = false;
        }
    }

    /// Allocate space for a function and copy the code
    ///
    /// This handles the W^X toggle automatically.
    pub fn allocate_function(&mut self, name: &str, code: &[u8]) -> Result<*const u8, String> {
        if code.is_empty() {
            return Err("Cannot allocate empty function".to_string());
        }

        // Ensure we're in write mode
        self.begin_write();

        // Try to allocate from existing regions
        let align = 16; // ARM64 functions should be 16-byte aligned
        let mut ptr: Option<*mut u8> = None;

        for region in &self.regions {
            if let Some(p) = region.allocate(code.len(), align) {
                ptr = Some(p);
                break;
            }
        }

        // If no space, allocate a new region
        let ptr = match ptr {
            Some(p) => p,
            None => {
                let new_size = std::cmp::max(DEFAULT_REGION_SIZE, code.len() * 2);
                let new_region = JITRegion::new(new_size)?;
                let p = new_region
                    .allocate(code.len(), align)
                    .ok_or("Failed to allocate from new region")?;
                self.regions.push(new_region);
                p
            }
        };

        // Copy the code
        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), ptr, code.len());
        }

        // Register the function
        let func = JITFunction {
            ptr: ptr as *const u8,
            size: code.len(),
        };
        self.functions.insert(name.to_string(), func);

        Ok(ptr as *const u8)
    }

    /// Finalize all pending writes and switch to execute mode
    ///
    /// Must be called after all functions are allocated and before execution.
    pub fn finalize(&mut self) {
        self.end_write();
    }

    /// Get a function pointer by name
    pub fn get_function(&self, name: &str) -> Option<*const u8> {
        self.functions.get(name).map(|f| f.ptr)
    }

    /// Check if a function exists
    #[allow(dead_code)]
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get total allocated memory
    #[allow(dead_code)]
    pub fn total_allocated(&self) -> usize {
        self.regions.iter().map(|r| r.size).sum()
    }

    /// Get total used memory
    #[allow(dead_code)]
    pub fn total_used(&self) -> usize {
        self.regions
            .iter()
            .map(|r| r.offset.load(Ordering::Relaxed))
            .sum()
    }

    /// Clear all functions (but keep memory allocated)
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.functions.clear();
        // Reset allocation offsets
        for region in &self.regions {
            region.offset.store(0, Ordering::SeqCst);
        }
    }
}

impl Drop for AppleSiliconJITMemory {
    fn drop(&mut self) {
        // Ensure we're not in write mode when dropping
        // (though JITRegion::drop will handle unmapping)
        if self.write_mode {
            unsafe {
                pthread_jit_write_protect_np(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_memory_basic() {
        let mut mem = AppleSiliconJITMemory::new().expect("Failed to create JIT memory");

        // Simple ARM64 function: return 42
        // mov w0, #42
        // ret
        let code: [u8; 8] = [
            0x40, 0x05, 0x80, 0x52, // mov w0, #42
            0xc0, 0x03, 0x5f, 0xd6, // ret
        ];

        let ptr = mem
            .allocate_function("return_42", &code)
            .expect("Failed to allocate function");

        mem.finalize();

        // Call the function
        unsafe {
            let func: extern "C" fn() -> i32 = std::mem::transmute(ptr);
            let result = func();
            assert_eq!(result, 42);
        }
    }

    #[test]
    fn test_multiple_functions() {
        let mut mem = AppleSiliconJITMemory::new().expect("Failed to create JIT memory");

        // return 1
        let code1: [u8; 8] = [
            0x20, 0x00, 0x80, 0x52, // mov w0, #1
            0xc0, 0x03, 0x5f, 0xd6, // ret
        ];

        // return 2
        let code2: [u8; 8] = [
            0x40, 0x00, 0x80, 0x52, // mov w0, #2
            0xc0, 0x03, 0x5f, 0xd6, // ret
        ];

        mem.allocate_function("return_1", &code1).unwrap();
        mem.allocate_function("return_2", &code2).unwrap();

        mem.finalize();

        let ptr1 = mem.get_function("return_1").unwrap();
        let ptr2 = mem.get_function("return_2").unwrap();

        unsafe {
            let func1: extern "C" fn() -> i32 = std::mem::transmute(ptr1);
            let func2: extern "C" fn() -> i32 = std::mem::transmute(ptr2);
            assert_eq!(func1(), 1);
            assert_eq!(func2(), 2);
        }
    }
}
