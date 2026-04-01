//! Loop Analysis for MIR Optimization
//!
//! This module provides loop analysis infrastructure for MIR optimizations including:
//! - Dominator tree computation (iterative dataflow algorithm)
//! - Natural loop detection via back-edge identification
//! - Loop nesting info and metadata

use super::{IrBlockId, IrControlFlowGraph, IrFunction};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Dominator tree for a function's control flow graph.
///
/// A block D dominates block B if every path from the entry to B goes through D.
/// The immediate dominator (idom) of B is the closest strict dominator.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Immediate dominator for each block (entry block has no idom)
    idom: BTreeMap<IrBlockId, IrBlockId>,

    /// Children in the dominator tree
    children: BTreeMap<IrBlockId, Vec<IrBlockId>>,

    /// Dominator tree depth for each block (entry = 0)
    depth: BTreeMap<IrBlockId, usize>,

    /// Entry block of the function
    entry: IrBlockId,
}

impl DominatorTree {
    /// Compute the dominator tree for a function using iterative dataflow.
    ///
    /// This uses Cooper, Harvey, and Kennedy's simple iterative algorithm
    /// which is efficient for typical CFGs and easy to understand.
    pub fn compute(function: &IrFunction) -> Self {
        let cfg = &function.cfg;
        let entry = cfg.entry_block;

        // Get blocks in reverse postorder for efficient iteration
        let rpo = Self::reverse_postorder(cfg, entry);
        let rpo_index: BTreeMap<IrBlockId, usize> =
            rpo.iter().enumerate().map(|(i, &b)| (b, i)).collect();

        // Initialize idom: entry dominates itself, others undefined
        let mut idom: BTreeMap<IrBlockId, Option<IrBlockId>> = BTreeMap::new();
        for &block in &rpo {
            idom.insert(block, None);
        }
        idom.insert(entry, Some(entry));

        // Iterative dataflow until fixed point
        let mut changed = true;
        let max_iterations = rpo.len() * 2 + 10; // Convergence guaranteed in O(n) for reducible CFGs
        let mut iteration = 0;
        while changed {
            changed = false;
            iteration += 1;
            if iteration > max_iterations {
                tracing::warn!(
                    "DominatorTree::compute: exceeded {} iterations, breaking",
                    max_iterations
                );
                break;
            }

            for &block in &rpo {
                if block == entry {
                    continue;
                }

                // Find first processed predecessor
                let predecessors: Vec<IrBlockId> = cfg
                    .get_block(block)
                    .map(|b| b.predecessors.clone())
                    .unwrap_or_default();

                let mut new_idom: Option<IrBlockId> = None;

                for &pred in &predecessors {
                    if idom.get(&pred).and_then(|x| *x).is_some() {
                        if new_idom.is_none() {
                            new_idom = Some(pred);
                        } else {
                            // Intersect: find common dominator
                            new_idom =
                                Some(Self::intersect(new_idom.unwrap(), pred, &idom, &rpo_index));
                        }
                    }
                }

                if new_idom != idom[&block] {
                    idom.insert(block, new_idom);
                    changed = true;
                }
            }
        }

        // Convert Option<IrBlockId> to IrBlockId (removing entry's self-domination)
        let mut final_idom: BTreeMap<IrBlockId, IrBlockId> = BTreeMap::new();
        for (&block, &dom) in &idom {
            if let Some(d) = dom {
                if block != entry {
                    final_idom.insert(block, d);
                }
            }
        }

        // Build children map
        // IMPORTANT: Sort children for deterministic iteration order in GVN and other passes
        let mut children: BTreeMap<IrBlockId, Vec<IrBlockId>> = BTreeMap::new();
        for (&block, &dom) in &final_idom {
            children.entry(dom).or_default().push(block);
        }
        // Sort all children lists for determinism
        for children_list in children.values_mut() {
            children_list.sort_by_key(|id| id.0);
        }

        // Compute depths via BFS from entry
        let mut depth: BTreeMap<IrBlockId, usize> = BTreeMap::new();
        depth.insert(entry, 0);
        let mut queue: VecDeque<IrBlockId> = VecDeque::new();
        queue.push_back(entry);

        while let Some(block) = queue.pop_front() {
            let d = depth[&block];
            for &child in children.get(&block).unwrap_or(&Vec::new()) {
                depth.insert(child, d + 1);
                queue.push_back(child);
            }
        }

        Self {
            idom: final_idom,
            children,
            depth,
            entry,
        }
    }

    /// Compute reverse postorder of blocks (good for dataflow iteration).
    fn reverse_postorder(cfg: &IrControlFlowGraph, entry: IrBlockId) -> Vec<IrBlockId> {
        let mut visited = BTreeSet::new();
        let mut postorder = Vec::new();

        fn dfs(
            cfg: &IrControlFlowGraph,
            block: IrBlockId,
            visited: &mut BTreeSet<IrBlockId>,
            postorder: &mut Vec<IrBlockId>,
        ) {
            if !visited.insert(block) {
                return;
            }

            if let Some(b) = cfg.get_block(block) {
                for succ in b.successors() {
                    dfs(cfg, succ, visited, postorder);
                }
            }

            postorder.push(block);
        }

        dfs(cfg, entry, &mut visited, &mut postorder);
        postorder.reverse();
        postorder
    }

    /// Find intersection of two dominators in the dominator tree.
    /// Uses the standard algorithm from Cooper, Harvey, and Kennedy.
    fn intersect(
        mut b1: IrBlockId,
        mut b2: IrBlockId,
        idom: &BTreeMap<IrBlockId, Option<IrBlockId>>,
        rpo_index: &BTreeMap<IrBlockId, usize>,
    ) -> IrBlockId {
        // Walk up the dominator tree until we find a common dominator
        let max_steps = idom.len() + 1;
        let mut steps = 0;
        while b1 != b2 {
            steps += 1;
            if steps > max_steps {
                return b1; // Safety: prevent infinite loop on malformed CFG
            }
            let mut idx1 = rpo_index.get(&b1).copied().unwrap_or(usize::MAX);
            let mut idx2 = rpo_index.get(&b2).copied().unwrap_or(usize::MAX);

            // Walk b1 up until it's at or before b2 in RPO
            while idx1 > idx2 {
                if let Some(Some(dom)) = idom.get(&b1) {
                    b1 = *dom;
                    idx1 = rpo_index.get(&b1).copied().unwrap_or(usize::MAX);
                } else {
                    // No dominator, shouldn't happen for valid CFG
                    return b1;
                }
            }

            // Walk b2 up until it's at or before b1 in RPO
            while idx2 > idx1 {
                if let Some(Some(dom)) = idom.get(&b2) {
                    b2 = *dom;
                    idx2 = rpo_index.get(&b2).copied().unwrap_or(usize::MAX);
                } else {
                    // No dominator, shouldn't happen for valid CFG
                    return b2;
                }
            }
        }
        b1
    }

    /// Get the immediate dominator of a block.
    pub fn idom(&self, block: IrBlockId) -> Option<IrBlockId> {
        self.idom.get(&block).copied()
    }

    /// Get children of a block in the dominator tree.
    pub fn children(&self, block: IrBlockId) -> &[IrBlockId] {
        self.children
            .get(&block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get the depth of a block in the dominator tree.
    pub fn depth(&self, block: IrBlockId) -> usize {
        self.depth.get(&block).copied().unwrap_or(0)
    }

    /// Check if block A dominates block B.
    pub fn dominates(&self, a: IrBlockId, b: IrBlockId) -> bool {
        if a == b {
            return true;
        }

        let mut current = b;
        while let Some(dom) = self.idom.get(&current) {
            if *dom == a {
                return true;
            }
            current = *dom;
        }

        // Entry block dominates everything
        a == self.entry
    }

    /// Check if block A strictly dominates block B (A dominates B and A != B).
    pub fn strictly_dominates(&self, a: IrBlockId, b: IrBlockId) -> bool {
        a != b && self.dominates(a, b)
    }
}

/// A natural loop in the control flow graph.
///
/// A natural loop is defined by a back edge (an edge from B to H where H dominates B).
/// The loop header is H, and the loop body contains all blocks from which H can be
/// reached without going through H.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Loop header block (entry point of the loop)
    pub header: IrBlockId,

    /// Back edge source (the block with the edge back to header)
    pub back_edge_source: IrBlockId,

    /// All blocks in the loop body (including header)
    pub blocks: BTreeSet<IrBlockId>,

    /// Exit blocks (blocks in the loop with edges outside the loop)
    pub exit_blocks: Vec<IrBlockId>,

    /// Preheader block if one exists (single predecessor of header from outside loop)
    pub preheader: Option<IrBlockId>,

    /// Estimated trip count if determinable (for unrolling decisions)
    pub trip_count: Option<TripCount>,

    /// Nesting depth (0 = outermost)
    pub nesting_depth: usize,

    /// Parent loop header if this is a nested loop
    pub parent: Option<IrBlockId>,

    /// Child loop headers (directly nested loops)
    pub children: Vec<IrBlockId>,
}

/// Trip count information for loops.
#[derive(Debug, Clone)]
pub enum TripCount {
    /// Constant trip count known at compile time
    Constant(u64),

    /// Bounded trip count (upper bound known)
    Bounded { max: u64 },

    /// Symbolic trip count based on a variable
    Symbolic { variable: super::IrId },

    /// Unknown trip count
    Unknown,
}

/// Loop nest information for a function.
#[derive(Debug, Clone)]
pub struct LoopNestInfo {
    /// All natural loops indexed by header block
    pub loops: BTreeMap<IrBlockId, NaturalLoop>,

    /// Top-level loops (not nested in any other loop)
    pub top_level_loops: Vec<IrBlockId>,

    /// Map from block to its innermost containing loop header
    pub block_to_loop: BTreeMap<IrBlockId, IrBlockId>,

    /// Maximum nesting depth in the function
    pub max_depth: usize,
}

impl LoopNestInfo {
    /// Analyze loops in a function.
    pub fn analyze(function: &IrFunction, domtree: &DominatorTree) -> Self {
        let cfg = &function.cfg;
        let mut loops = BTreeMap::new();

        // Find all back edges and create natural loops
        for (&block_id, block) in &cfg.blocks {
            for succ in block.successors() {
                // A back edge is an edge to a dominator
                if domtree.dominates(succ, block_id) {
                    // Found back edge: block_id -> succ
                    // succ is the loop header
                    let loop_blocks = Self::find_loop_blocks(cfg, succ, block_id);
                    let exit_blocks = Self::find_exit_blocks(cfg, &loop_blocks);
                    let preheader = Self::find_preheader(cfg, succ, &loop_blocks);

                    let natural_loop = NaturalLoop {
                        header: succ,
                        back_edge_source: block_id,
                        blocks: loop_blocks,
                        exit_blocks,
                        preheader,
                        trip_count: None, // Computed later if needed
                        nesting_depth: 0, // Computed after all loops found
                        parent: None,
                        children: Vec::new(),
                    };

                    // If we already have a loop with this header, merge the blocks
                    if let Some(existing) = loops.get_mut(&succ) {
                        let existing: &mut NaturalLoop = existing;
                        existing.blocks.extend(natural_loop.blocks);
                        existing.exit_blocks = Self::find_exit_blocks(cfg, &existing.blocks);
                    } else {
                        loops.insert(succ, natural_loop);
                    }
                }
            }
        }

        // Compute nesting relationships
        let loop_headers: Vec<IrBlockId> = loops.keys().copied().collect();
        for &header in &loop_headers {
            for &other_header in &loop_headers {
                if header != other_header {
                    let header_loop = &loops[&header];
                    let other_loop = &loops[&other_header];

                    // header's loop is nested in other's loop if header is in other's blocks
                    if other_loop.blocks.contains(&header)
                        && !header_loop.blocks.contains(&other_header)
                    {
                        // header is nested in other_header
                        // We'll set parent to the innermost containing loop
                        if let Some(current_parent) = loops[&header].parent {
                            let current_parent_loop = &loops[&current_parent];
                            // Use smaller loop as parent (more immediate)
                            if other_loop.blocks.len() < current_parent_loop.blocks.len() {
                                loops.get_mut(&header).unwrap().parent = Some(other_header);
                            }
                        } else {
                            loops.get_mut(&header).unwrap().parent = Some(other_header);
                        }
                    }
                }
            }
        }

        // Build children lists
        for &header in &loop_headers {
            if let Some(parent) = loops[&header].parent {
                loops.get_mut(&parent).unwrap().children.push(header);
            }
        }

        // Compute nesting depths
        let top_level_loops: Vec<IrBlockId> = loop_headers
            .iter()
            .filter(|&&h| loops[&h].parent.is_none())
            .copied()
            .collect();

        fn set_depth(loops: &mut BTreeMap<IrBlockId, NaturalLoop>, header: IrBlockId, depth: usize) {
            loops.get_mut(&header).unwrap().nesting_depth = depth;
            let children: Vec<IrBlockId> = loops[&header].children.clone();
            for child in children {
                set_depth(loops, child, depth + 1);
            }
        }

        for &top_level in &top_level_loops {
            set_depth(&mut loops, top_level, 0);
        }

        let max_depth = loops.values().map(|l| l.nesting_depth).max().unwrap_or(0);

        // Build block-to-loop mapping (map each block to its innermost loop)
        let mut block_to_loop = BTreeMap::new();
        for (&header, loop_info) in &loops {
            for &block in &loop_info.blocks {
                // Only set if not already set or if this loop is more deeply nested
                if let Some(&existing_header) = block_to_loop.get(&block) {
                    if loops[&header].nesting_depth > loops[&existing_header].nesting_depth {
                        block_to_loop.insert(block, header);
                    }
                } else {
                    block_to_loop.insert(block, header);
                }
            }
        }

        Self {
            loops,
            top_level_loops,
            block_to_loop,
            max_depth,
        }
    }

    /// Find all blocks in a natural loop given header and back edge source.
    fn find_loop_blocks(
        cfg: &IrControlFlowGraph,
        header: IrBlockId,
        back_edge_source: IrBlockId,
    ) -> BTreeSet<IrBlockId> {
        let mut loop_blocks = BTreeSet::new();
        loop_blocks.insert(header);

        if header == back_edge_source {
            return loop_blocks;
        }

        // Work backwards from back_edge_source to find all blocks that can reach header
        let mut worklist = vec![back_edge_source];
        loop_blocks.insert(back_edge_source);

        while let Some(block) = worklist.pop() {
            if let Some(b) = cfg.get_block(block) {
                for &pred in &b.predecessors {
                    if !loop_blocks.contains(&pred) {
                        loop_blocks.insert(pred);
                        worklist.push(pred);
                    }
                }
            }
        }

        loop_blocks
    }

    /// Find exit blocks (blocks in loop with successors outside loop).
    fn find_exit_blocks(
        cfg: &IrControlFlowGraph,
        loop_blocks: &BTreeSet<IrBlockId>,
    ) -> Vec<IrBlockId> {
        let mut exits = Vec::new();

        for &block in loop_blocks {
            if let Some(b) = cfg.get_block(block) {
                for succ in b.successors() {
                    if !loop_blocks.contains(&succ) {
                        exits.push(block);
                        break;
                    }
                }
            }
        }

        exits
    }

    /// Find preheader block if one exists.
    fn find_preheader(
        cfg: &IrControlFlowGraph,
        header: IrBlockId,
        loop_blocks: &BTreeSet<IrBlockId>,
    ) -> Option<IrBlockId> {
        let header_block = cfg.get_block(header)?;

        // Find predecessors outside the loop
        let outside_preds: Vec<IrBlockId> = header_block
            .predecessors
            .iter()
            .filter(|p| !loop_blocks.contains(p))
            .copied()
            .collect();

        // Preheader exists if there's exactly one predecessor outside the loop
        // and it has only one successor (the header)
        if outside_preds.len() == 1 {
            let pred = outside_preds[0];
            if let Some(pred_block) = cfg.get_block(pred) {
                if pred_block.successors().len() == 1 {
                    return Some(pred);
                }
            }
        }

        None
    }

    /// Get the loop containing a block, if any.
    pub fn get_loop(&self, block: IrBlockId) -> Option<&NaturalLoop> {
        self.block_to_loop
            .get(&block)
            .and_then(|h| self.loops.get(h))
    }

    /// Get loop depth for a block (0 if not in any loop).
    pub fn loop_depth(&self, block: IrBlockId) -> usize {
        self.get_loop(block)
            .map(|l| l.nesting_depth + 1)
            .unwrap_or(0)
    }

    /// Check if a block is a loop header.
    pub fn is_loop_header(&self, block: IrBlockId) -> bool {
        self.loops.contains_key(&block)
    }

    /// Iterate over all loops in order of nesting depth (outermost first).
    pub fn loops_by_depth(&self) -> Vec<&NaturalLoop> {
        let mut loops: Vec<&NaturalLoop> = self.loops.values().collect();
        loops.sort_by_key(|l| l.nesting_depth);
        loops
    }

    /// Iterate over all loops in reverse nesting order (innermost first).
    pub fn loops_innermost_first(&self) -> Vec<&NaturalLoop> {
        let mut loops: Vec<&NaturalLoop> = self.loops.values().collect();
        loops.sort_by_key(|l| std::cmp::Reverse(l.nesting_depth));
        loops
    }
}

/// Mark loop headers in block metadata.
pub fn annotate_loop_headers(function: &mut IrFunction, loop_info: &LoopNestInfo) {
    for header in loop_info.loops.keys() {
        if let Some(block) = function.cfg.blocks.get_mut(header) {
            block.metadata.is_loop_header = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::*;
    use crate::ir::{IrBlockId, IrType};
    use crate::tast::SymbolId;

    #[test]
    fn test_dominator_tree_simple() {
        // Create a simple function with diamond CFG:
        //      entry
        //       /\
        //      /  \
        //    bb1  bb2
        //      \  /
        //       \/
        //      bb3

        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());
        let sig = FunctionSignatureBuilder::new()
            .returns(IrType::Void)
            .build();
        builder.start_function(SymbolId::from_raw(1), "test".to_string(), sig);

        // Entry block
        let cond = builder.build_bool(true).unwrap();
        let bb1 = builder.create_block().unwrap();
        let bb2 = builder.create_block().unwrap();
        builder.build_cond_branch(cond, bb1, bb2);

        // bb1
        builder.switch_to_block(bb1);
        let bb3 = builder.create_block().unwrap();
        builder.build_branch(bb3);

        // bb2
        builder.switch_to_block(bb2);
        builder.build_branch(bb3);

        // bb3
        builder.switch_to_block(bb3);
        builder.build_return(None);

        builder.finish_function();

        let function = builder.module.functions.values().next().unwrap();
        let domtree = DominatorTree::compute(function);

        // Entry dominates everything
        assert!(domtree.dominates(IrBlockId::entry(), bb1));
        assert!(domtree.dominates(IrBlockId::entry(), bb2));
        assert!(domtree.dominates(IrBlockId::entry(), bb3));

        // bb1 and bb2 don't dominate each other
        assert!(!domtree.dominates(bb1, bb2));
        assert!(!domtree.dominates(bb2, bb1));

        // Entry is the idom of bb3 (not bb1 or bb2)
        assert_eq!(domtree.idom(bb3), Some(IrBlockId::entry()));
    }

    #[test]
    fn test_simple_loop_detection() {
        // Create a simple loop:
        //     entry
        //       |
        //       v
        //     header <----+
        //       |         |
        //       v         |
        //     body -------+
        //       |
        //       v
        //     exit

        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());
        let sig = FunctionSignatureBuilder::new()
            .returns(IrType::Void)
            .build();
        builder.start_function(SymbolId::from_raw(1), "loop_test".to_string(), sig);

        // Entry -> header
        let header = builder.create_block().unwrap();
        builder.build_branch(header);

        // Header
        builder.switch_to_block(header);
        let cond = builder.build_bool(true).unwrap();
        let body = builder.create_block().unwrap();
        let exit = builder.create_block().unwrap();
        builder.build_cond_branch(cond, body, exit);

        // Body -> header (back edge)
        builder.switch_to_block(body);
        builder.build_branch(header);

        // Exit
        builder.switch_to_block(exit);
        builder.build_return(None);

        builder.finish_function();

        let function = builder.module.functions.values().next().unwrap();
        let domtree = DominatorTree::compute(function);
        let loop_info = LoopNestInfo::analyze(function, &domtree);

        // Should find one loop with header
        assert_eq!(loop_info.loops.len(), 1);
        assert!(loop_info.is_loop_header(header));

        let the_loop = &loop_info.loops[&header];
        assert!(the_loop.blocks.contains(&header));
        assert!(the_loop.blocks.contains(&body));
        assert!(!the_loop.blocks.contains(&exit));
        assert_eq!(the_loop.back_edge_source, body);
    }
}
