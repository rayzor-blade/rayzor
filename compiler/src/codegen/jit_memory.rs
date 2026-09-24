//! JIT memory placement for x86-64.
//!
//! cranelift-jit encodes a call between two functions of one module as a
//! 32-bit displacement and has no out-of-range fallback on x86-64, so every
//! allocation of a module must land within 2GB of every other. The system
//! provider maps each allocation independently and gives no such bound.
//! This provider carves allocations from one reserved region, and falls back
//! to independent mappings only once that region is exhausted.

use cranelift_jit::{
    ArenaMemoryProvider, BranchProtection, JITMemoryKind, JITMemoryProvider, SystemMemoryProvider,
};
use cranelift_module::ModuleResult;
use std::io;

/// Reserved address space per module. Windows commits the reservation
/// up front, so it gets a smaller one.
#[cfg(windows)]
const RESERVE_BYTES: usize = 64 << 20;
#[cfg(not(windows))]
const RESERVE_BYTES: usize = 1 << 30;

pub struct ReachableMemoryProvider {
    arena: Option<ArenaMemoryProvider>,
    overflow: SystemMemoryProvider,
}

impl ReachableMemoryProvider {
    pub fn new() -> Self {
        Self {
            arena: ArenaMemoryProvider::new_with_size(RESERVE_BYTES).ok(),
            overflow: SystemMemoryProvider::new(),
        }
    }
}

impl Default for ReachableMemoryProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// `JITMemoryKind` is not `Copy`; the overflow path needs its own.
fn copy_kind(kind: &JITMemoryKind) -> JITMemoryKind {
    match kind {
        JITMemoryKind::Executable => JITMemoryKind::Executable,
        JITMemoryKind::Writable => JITMemoryKind::Writable,
        JITMemoryKind::ReadOnly => JITMemoryKind::ReadOnly,
    }
}

impl JITMemoryProvider for ReachableMemoryProvider {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        if let Some(arena) = &mut self.arena
            && let Ok(ptr) = arena.allocate(size, align, copy_kind(&kind))
        {
            return Ok(ptr);
        }
        self.overflow.allocate(size, align, kind)
    }

    unsafe fn free_memory(&mut self) {
        if let Some(arena) = &mut self.arena {
            unsafe { arena.free_memory() };
        }
        unsafe { self.overflow.free_memory() };
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        if let Some(arena) = &mut self.arena {
            arena.finalize(branch_protection)?;
        }
        self.overflow.finalize(branch_protection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_stay_within_call_reach() {
        let mut memory = ReachableMemoryProvider::new();
        let first = memory
            .allocate(4096, 16, JITMemoryKind::Executable)
            .unwrap() as isize;
        for kind in [
            JITMemoryKind::ReadOnly,
            JITMemoryKind::Writable,
            JITMemoryKind::Executable,
        ] {
            let p = memory.allocate(1 << 20, 16, kind).unwrap() as isize;
            assert!(i32::try_from(p - first).is_ok());
        }
        memory.finalize(BranchProtection::None).unwrap();
    }
}
