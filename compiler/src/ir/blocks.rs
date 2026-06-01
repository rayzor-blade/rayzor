//! HIR Basic Blocks
//!
//! This module defines basic blocks for the HIR, which are sequences of instructions
//! with a single entry point and single exit point. Basic blocks form the nodes
//! of the control flow graph at the HIR level.

use super::{IrId, IrInstruction, IrSourceLocation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A basic block in the HIR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrBasicBlock {
    /// Unique identifier for this block
    pub id: IrBlockId,

    /// Human-readable label (for debugging)
    pub label: Option<String>,

    /// Instructions in this block (executed sequentially)
    pub instructions: Vec<IrInstruction>,

    /// Terminator instruction (branch, return, etc.)
    pub terminator: IrTerminator,

    /// Phi nodes at the beginning of this block
    pub phi_nodes: Vec<IrPhiNode>,

    /// Source location for debugging
    pub source_location: IrSourceLocation,

    /// Predecessors in the CFG
    pub predecessors: Vec<IrBlockId>,

    /// Metadata for optimization hints
    pub metadata: BlockMetadata,

    /// `true` once a terminator has been *explicitly* set on this block
    /// (via the builder's `set_terminator`). Distinguishes deliberate
    /// `Unreachable` (e.g. after `throw`) from the default uninitialized
    /// state, which also uses `Unreachable` as its placeholder.
    /// Not part of any external invariant — serialized as `false` by
    /// default for backwards compat with pre-existing `.blade` caches.
    #[serde(default)]
    pub terminator_explicit: bool,
}

/// Unique identifier for basic blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct IrBlockId(pub u32);

impl IrBlockId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn entry() -> Self {
        Self(0)
    }

    pub fn is_entry(&self) -> bool {
        self.0 == 0
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for IrBlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// Phi node for merging values from different control flow paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrPhiNode {
    /// Destination register for the phi result
    pub dest: IrId,

    /// Incoming values from predecessor blocks
    pub incoming: Vec<(IrBlockId, IrId)>,

    /// Type of the phi node
    pub ty: super::IrType,
}

/// Terminator instructions that end a basic block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrTerminator {
    /// Unconditional branch to another block
    Branch { target: IrBlockId },

    /// Conditional branch based on a boolean value
    CondBranch {
        condition: IrId,
        true_target: IrBlockId,
        false_target: IrBlockId,
    },

    /// Switch/jump table
    Switch {
        value: IrId,
        cases: Vec<(i64, IrBlockId)>,
        default: IrBlockId,
    },

    /// Return from function
    Return { value: Option<IrId> },

    /// Unreachable code (for optimization)
    Unreachable,

    /// Call that doesn't return (e.g., throw)
    NoReturn { call: IrId },
}

/// Metadata for optimization and analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BlockMetadata {
    /// Execution frequency hint (0-100, higher = hotter)
    pub frequency_hint: Option<u8>,

    /// Whether this block is a loop header
    pub is_loop_header: bool,

    /// Whether this block is in a try/catch region
    pub in_exception_handler: bool,

    /// Custom optimization hints from semantic analysis
    pub optimization_hints: Vec<OptimizationHint>,
}

/// Optimization hints from semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationHint {
    /// This block is likely to be taken
    LikelyPath,

    /// This block is unlikely to be taken
    UnlikelyPath,

    /// Values in this block should be kept in registers
    HotPath,

    /// This block can be optimized for size
    ColdPath,

    /// Custom hint with string description
    Custom(String),
}

impl IrBasicBlock {
    /// Create a new basic block
    pub fn new(id: IrBlockId) -> Self {
        Self {
            id,
            label: None,
            instructions: Vec::new(),
            terminator: IrTerminator::Unreachable,
            phi_nodes: Vec::new(),
            source_location: IrSourceLocation::unknown(),
            predecessors: Vec::new(),
            metadata: BlockMetadata::default(),
            terminator_explicit: false,
        }
    }

    /// Add an instruction to this block
    pub fn add_instruction(&mut self, inst: IrInstruction) {
        self.instructions.push(inst);
    }

    /// Add a phi node to this block
    pub fn add_phi(&mut self, phi: IrPhiNode) {
        self.phi_nodes.push(phi);
    }

    /// Set the terminator for this block
    pub fn set_terminator(&mut self, term: IrTerminator) {
        self.terminator = term;
    }

    /// Get all successor blocks based on the terminator
    pub fn successors(&self) -> Vec<IrBlockId> {
        match &self.terminator {
            IrTerminator::Branch { target } => vec![*target],
            IrTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                vec![*true_target, *false_target]
            }
            IrTerminator::Switch { cases, default, .. } => {
                let mut succs: Vec<_> = cases.iter().map(|(_, target)| *target).collect();
                succs.push(*default);
                succs
            }
            IrTerminator::Return { .. }
            | IrTerminator::Unreachable
            | IrTerminator::NoReturn { .. } => Vec::new(),
        }
    }

    /// Check if this block is terminated properly.
    ///
    /// Returns true for any non-default terminator. The default for newly
    /// created blocks is `IrTerminator::Unreachable` (uninitialized state),
    /// which reads as "not terminated" so the standard `ensure_terminator` /
    /// branch-back-to-merge path can fill it in. Explicit `build_unreachable`
    /// (after throw/panic) uses the same variant; callers that need to
    /// distinguish explicit-noreturn from default-uninitialized must consult
    /// `terminator_explicit` on the block instead.
    pub fn is_terminated(&self) -> bool {
        !matches!(self.terminator, IrTerminator::Unreachable)
    }
}

/// Control flow graph at the HIR level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrControlFlowGraph {
    /// All basic blocks in the function
    /// Uses BTreeMap for deterministic iteration order (sorted by block ID)
    pub blocks: std::collections::BTreeMap<IrBlockId, IrBasicBlock>,

    /// Entry block ID
    pub entry_block: IrBlockId,

    /// Next available block ID (pub for MIR builder)
    pub next_block_id: u32,
}

impl IrControlFlowGraph {
    /// Create a new CFG with an entry block
    pub fn new() -> Self {
        let mut blocks = std::collections::BTreeMap::new();
        let entry_block = IrBlockId::entry();
        blocks.insert(entry_block, IrBasicBlock::new(entry_block));

        Self {
            blocks,
            entry_block,
            next_block_id: 1,
        }
    }

    /// Create a new basic block
    pub fn create_block(&mut self) -> IrBlockId {
        let id = IrBlockId::new(self.next_block_id);
        self.next_block_id += 1;
        self.blocks.insert(id, IrBasicBlock::new(id));
        id
    }

    /// Get a block by ID
    pub fn get_block(&self, id: IrBlockId) -> Option<&IrBasicBlock> {
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID
    pub fn get_block_mut(&mut self, id: IrBlockId) -> Option<&mut IrBasicBlock> {
        self.blocks.get_mut(&id)
    }

    /// Connect two blocks (update predecessors)
    pub fn connect_blocks(&mut self, from: IrBlockId, to: IrBlockId) {
        if let Some(to_block) = self.blocks.get_mut(&to) {
            if !to_block.predecessors.contains(&from) {
                to_block.predecessors.push(from);
            }
        }
    }

    /// Recompute all predecessor lists from scratch by walking every block's terminator.
    ///
    /// Some optimization passes mutate block terminators (e.g. branch redirection,
    /// dead-code elimination, block splitting) without keeping the cached
    /// IrBasicBlock::predecessors lists in sync. Loop analysis depends on
    /// accurate predecessor information to identify loop bodies and exit edges;
    /// stale predecessors cause find_loop_blocks to truncate the body and
    /// find_exit_blocks to mis-flag in-body branches as exits.
    pub fn recompute_predecessors(&mut self) {
        for block in self.blocks.values_mut() {
            block.predecessors.clear();
        }
        // Collect (from, to) edges first to avoid borrow conflicts.
        let edges: Vec<(IrBlockId, IrBlockId)> = self
            .blocks
            .iter()
            .flat_map(|(from, block)| block.successors().into_iter().map(move |to| (*from, to)))
            .collect();
        for (from, to) in edges {
            if let Some(to_block) = self.blocks.get_mut(&to) {
                if !to_block.predecessors.contains(&from) {
                    to_block.predecessors.push(from);
                }
            }
        }
    }

    /// Verify CFG integrity
    pub fn verify(&self) -> Result<(), String> {
        // Check entry block exists
        if !self.blocks.contains_key(&self.entry_block) {
            return Err("Entry block not found".to_string());
        }

        // Check all blocks are properly terminated
        for (id, block) in &self.blocks {
            if !block.is_terminated() {
                return Err(format!("Block {} is not properly terminated", id));
            }

            // Verify successor blocks exist
            for succ in block.successors() {
                if !self.blocks.contains_key(&succ) {
                    return Err(format!(
                        "Block {} references non-existent successor {}",
                        id, succ
                    ));
                }
            }

            // Verify phi node consistency
            for phi in &block.phi_nodes {
                for (pred_block, _) in &phi.incoming {
                    if !block.predecessors.contains(pred_block) {
                        return Err(format!(
                            "Phi node in block {} references non-predecessor block {}",
                            id, pred_block
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block_creation() {
        let mut block = IrBasicBlock::new(IrBlockId::new(1));
        assert_eq!(block.id.0, 1);
        assert!(block.instructions.is_empty());
        assert!(!block.is_terminated());

        block.set_terminator(IrTerminator::Return { value: None });
        assert!(block.is_terminated());
    }

    #[test]
    fn test_cfg_creation() {
        let mut cfg = IrControlFlowGraph::new();
        assert!(cfg.get_block(IrBlockId::entry()).is_some());

        let bb1 = cfg.create_block();
        let bb2 = cfg.create_block();

        cfg.connect_blocks(IrBlockId::entry(), bb1);
        cfg.connect_blocks(bb1, bb2);

        assert_eq!(
            cfg.get_block(bb1).unwrap().predecessors,
            vec![IrBlockId::entry()]
        );
        assert_eq!(cfg.get_block(bb2).unwrap().predecessors, vec![bb1]);
    }
}
