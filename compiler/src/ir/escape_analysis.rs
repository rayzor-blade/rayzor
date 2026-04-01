//! Intra-loop escape analysis for Alloc instructions.
//!
//! Determines whether an allocation inside a loop body escapes the loop iteration,
//! enabling LICM to hoist non-escaping allocations to the loop preheader.
//! When an Alloc is hoisted and its matching Free is sunk past the loop, the same
//! memory is reused across iterations, eliminating per-iteration malloc/free overhead.

use super::blocks::IrBlockId;
use super::instructions::IrInstruction;
use super::{IrControlFlowGraph, IrId};
use std::collections::BTreeSet;

/// Result of escape analysis for a single Alloc within a loop.
#[derive(Debug)]
pub struct AllocEscapeInfo {
    /// The IrId produced by the Alloc (the pointer)
    pub alloc_dest: IrId,
    /// Block and instruction index of the Alloc
    pub alloc_location: (IrBlockId, usize),
    /// Whether the pointer escapes (true = cannot hoist)
    pub escapes: bool,
    /// Matching Free location (block, index) — None if no unique Free found
    pub free_location: Option<(IrBlockId, usize)>,
}

/// Analyze all Alloc instructions within a loop for escape behavior.
///
/// An Alloc is considered non-escaping if:
/// 1. Its pointer (and derived pointers) are never stored as values to memory
/// 2. Its pointer is never passed to function calls
/// 3. Its pointer doesn't appear in phi nodes crossing the back edge
/// 4. It has exactly one matching Free in the loop body
pub fn analyze_alloc_escapes(
    cfg: &IrControlFlowGraph,
    loop_blocks: &BTreeSet<IrBlockId>,
    loop_header: IrBlockId,
    back_edge_sources: &[IrBlockId],
) -> Vec<AllocEscapeInfo> {
    let mut results = Vec::new();

    // Find all Alloc instructions in the loop
    for &block_id in loop_blocks {
        let block = match cfg.get_block(block_id) {
            Some(b) => b,
            None => continue,
        };
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let IrInstruction::Alloc { dest, count, .. } = inst {
                // If count is dynamic and defined inside the loop, skip
                if let Some(count_id) = count {
                    if is_defined_in_loop(*count_id, cfg, loop_blocks) {
                        results.push(AllocEscapeInfo {
                            alloc_dest: *dest,
                            alloc_location: (block_id, idx),
                            escapes: true,
                            free_location: None,
                        });
                        continue;
                    }
                }

                let info = analyze_single_alloc(
                    *dest,
                    (block_id, idx),
                    cfg,
                    loop_blocks,
                    loop_header,
                    back_edge_sources,
                );
                results.push(info);
            }
        }
    }

    results
}

/// Check if a value is defined inside the loop.
fn is_defined_in_loop(
    id: IrId,
    cfg: &IrControlFlowGraph,
    loop_blocks: &BTreeSet<IrBlockId>,
) -> bool {
    for &block_id in loop_blocks {
        if let Some(block) = cfg.get_block(block_id) {
            for phi in &block.phi_nodes {
                if phi.dest == id {
                    return true;
                }
            }
            for inst in &block.instructions {
                if inst.dest() == Some(id) {
                    return true;
                }
            }
        }
    }
    false
}

/// Analyze a single Alloc for escape.
fn analyze_single_alloc(
    alloc_dest: IrId,
    alloc_location: (IrBlockId, usize),
    cfg: &IrControlFlowGraph,
    loop_blocks: &BTreeSet<IrBlockId>,
    loop_header: IrBlockId,
    back_edge_sources: &[IrBlockId],
) -> AllocEscapeInfo {
    // Phase 1: Build the set of pointers derived from this allocation.
    // Includes the alloc dest itself, plus any PtrAdd/GEP/Cast/BitCast/Copy results.
    let tracked = build_tracked_pointers(alloc_dest, cfg, loop_blocks);

    // Phase 2: Check escape conditions
    if check_escapes(&tracked, cfg, loop_blocks, loop_header, back_edge_sources) {
        return AllocEscapeInfo {
            alloc_dest,
            alloc_location,
            escapes: true,
            free_location: None,
        };
    }

    // Phase 3: Find matching Free
    let free_loc = find_matching_free(&tracked, cfg, loop_blocks);

    match free_loc {
        Some(loc) => AllocEscapeInfo {
            alloc_dest,
            alloc_location,
            escapes: false,
            free_location: Some(loc),
        },
        None => AllocEscapeInfo {
            alloc_dest,
            alloc_location,
            escapes: true,
            free_location: None,
        },
    }
}

/// Build the transitive closure of pointers derived from `alloc_dest`.
///
/// Starting from the alloc dest, track through PtrAdd, GetElementPtr,
/// Cast, BitCast, and Copy instructions to find all derived pointers.
fn build_tracked_pointers(
    alloc_dest: IrId,
    cfg: &IrControlFlowGraph,
    loop_blocks: &BTreeSet<IrBlockId>,
) -> BTreeSet<IrId> {
    let mut tracked = BTreeSet::new();
    tracked.insert(alloc_dest);

    // Iterate until fixpoint (derived pointers can chain)
    let mut changed = true;
    while changed {
        changed = false;
        for &block_id in loop_blocks {
            let block = match cfg.get_block(block_id) {
                Some(b) => b,
                None => continue,
            };
            for inst in &block.instructions {
                let derived_from_tracked = match inst {
                    IrInstruction::PtrAdd { dest, ptr, .. } if tracked.contains(ptr) => Some(*dest),
                    IrInstruction::GetElementPtr { dest, ptr, .. } if tracked.contains(ptr) => {
                        Some(*dest)
                    }
                    IrInstruction::Cast { dest, src, .. } if tracked.contains(src) => Some(*dest),
                    IrInstruction::BitCast { dest, src, .. } if tracked.contains(src) => {
                        Some(*dest)
                    }
                    IrInstruction::Copy { dest, src } if tracked.contains(src) => Some(*dest),
                    _ => None,
                };

                if let Some(derived) = derived_from_tracked {
                    if tracked.insert(derived) {
                        changed = true;
                    }
                }
            }
        }
    }

    tracked
}

/// Check if any tracked pointer escapes the loop iteration.
fn check_escapes(
    tracked: &BTreeSet<IrId>,
    cfg: &IrControlFlowGraph,
    loop_blocks: &BTreeSet<IrBlockId>,
    loop_header: IrBlockId,
    back_edge_sources: &[IrBlockId],
) -> bool {
    let back_edge_set: BTreeSet<IrBlockId> = back_edge_sources.iter().copied().collect();

    for &block_id in loop_blocks {
        let block = match cfg.get_block(block_id) {
            Some(b) => b,
            None => continue,
        };

        // Check phi nodes in the loop header for back-edge crossings
        if block_id == loop_header {
            for phi in &block.phi_nodes {
                for (incoming_block, incoming_val) in &phi.incoming {
                    // If a tracked pointer comes from a back-edge source,
                    // it lives across iterations → escapes
                    if back_edge_set.contains(incoming_block) && tracked.contains(incoming_val) {
                        return true;
                    }
                }
            }
        }

        for inst in &block.instructions {
            match inst {
                // Pointer stored as a VALUE to memory → escapes
                // (Store { ptr: tracked, value: _ } is fine — writing INTO the alloc)
                IrInstruction::Store { value, .. } if tracked.contains(value) => {
                    return true;
                }

                // Passed to a function call → escapes
                IrInstruction::CallDirect { args, .. } => {
                    if args.iter().any(|a| tracked.contains(a)) {
                        return true;
                    }
                }
                IrInstruction::CallIndirect { args, func_ptr, .. } => {
                    if tracked.contains(func_ptr) || args.iter().any(|a| tracked.contains(a)) {
                        return true;
                    }
                }

                // Returned → escapes
                IrInstruction::Return { value: Some(v) } if tracked.contains(v) => {
                    return true;
                }

                // MemCopy with tracked as source could copy the pointer bytes
                IrInstruction::MemCopy { src, .. } if tracked.contains(src) => {
                    return true;
                }

                // Stored globally → escapes
                IrInstruction::StoreGlobal { value, .. } if tracked.contains(value) => {
                    return true;
                }

                // Captured in closure → escapes
                IrInstruction::MakeClosure {
                    captured_values, ..
                } => {
                    if captured_values.iter().any(|v| tracked.contains(v)) {
                        return true;
                    }
                }

                // Thrown as exception → escapes
                IrInstruction::Throw { exception } if tracked.contains(exception) => {
                    return true;
                }

                _ => {}
            }
        }
    }

    false
}

/// Find exactly one matching Free for the tracked pointers in the loop body.
/// Returns None if zero or more than one Free is found.
fn find_matching_free(
    tracked: &BTreeSet<IrId>,
    cfg: &IrControlFlowGraph,
    loop_blocks: &BTreeSet<IrBlockId>,
) -> Option<(IrBlockId, usize)> {
    let mut found: Option<(IrBlockId, usize)> = None;

    for &block_id in loop_blocks {
        let block = match cfg.get_block(block_id) {
            Some(b) => b,
            None => continue,
        };
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let IrInstruction::Free { ptr } = inst {
                if tracked.contains(ptr) {
                    if found.is_some() {
                        // More than one Free → conservative, treat as escaping
                        return None;
                    }
                    found = Some((block_id, idx));
                }
            }
        }
    }

    found
}
