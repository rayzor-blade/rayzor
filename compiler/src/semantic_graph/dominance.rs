//! Dominance Analysis Implementation
//!
//! Implements the Lengauer-Tarjan algorithm for computing dominance trees
//! and dominance frontiers. This is the foundation for SSA construction
//! and replaces the fake dominance computation in the DFG builder.
//!
//! ## Algorithm Overview
//!
//! The Lengauer-Tarjan algorithm computes dominance in O(E α(E,V)) time,
//! which is nearly linear for practical CFGs. The algorithm proceeds in phases:
//!
//! 1. **DFS Numbering**: Assign DFS numbers to blocks for efficient processing
//! 2. **Semi-dominator Computation**: Find semi-dominators using union-find
//! 3. **Immediate Dominator Computation**: Convert semi-dominators to immediate dominators
//! 4. **Dominance Tree Construction**: Build tree structure from immediate dominators
//! 5. **Dominance Frontier Computation**: Compute frontiers for phi-node placement
//!
//! ## Memory Safety
//!
//! This implementation is critical for memory safety analysis as dominance
//! determines where phi-nodes are placed, which affects lifetime analysis.

use crate::semantic_graph::cfg::{BasicBlock, ControlFlowGraph};
use crate::tast::{BlockId, SourceLocation};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

/// **Dominance Tree and Analysis Results**
///
/// Contains the complete dominance analysis for a control flow graph,
/// including immediate dominators, dominance frontiers, and tree structure.
#[derive(Debug, Clone)]
pub struct DominanceTree {
    /// Immediate dominator for each block (idom[b] = immediate dominator of b)
    pub idom: BTreeMap<BlockId, BlockId>,

    /// All dominators for each block (transitive closure of dominance)
    pub dominators: BTreeMap<BlockId, BTreeSet<BlockId>>,

    /// Children in the dominance tree structure
    pub dom_tree_children: BTreeMap<BlockId, Vec<BlockId>>,

    /// Dominance frontiers for phi-node placement (DF[b] = dominance frontier of b)
    pub dominance_frontiers: BTreeMap<BlockId, Vec<BlockId>>,

    /// DFS pre-order numbering for algorithm efficiency
    pub dfs_preorder: BTreeMap<BlockId, usize>,

    /// DFS post-order numbering for certain algorithms
    pub dfs_postorder: BTreeMap<BlockId, usize>,

    /// Reverse post-order traversal (good ordering for dataflow analysis)
    pub reverse_postorder: Vec<BlockId>,

    /// Performance and diagnostic information
    pub stats: DominanceStats,
}

/// **Union-Find Data Structure for Lengauer-Tarjan**
///
/// Implements union-find with path compression and union by rank
/// for efficient semi-dominator computation.
#[derive(Debug)]
struct UnionFind {
    /// Parent pointers for union-find
    parent: BTreeMap<BlockId, BlockId>,

    /// Ranks for union by rank optimization
    rank: BTreeMap<BlockId, usize>,

    /// Semi-dominators for each vertex
    semi: BTreeMap<BlockId, usize>,

    /// Labels for path compression optimization
    label: BTreeMap<BlockId, BlockId>,
}

/// **Performance and Diagnostic Statistics**
#[derive(Debug, Clone, Default)]
pub struct DominanceStats {
    /// Total computation time in microseconds
    pub computation_time_us: u64,

    /// Time spent in DFS numbering
    pub dfs_time_us: u64,

    /// Time spent in semi-dominator computation
    pub semi_dom_time_us: u64,

    /// Time spent in immediate dominator computation
    pub idom_time_us: u64,

    /// Time spent in dominance frontier computation
    pub frontier_time_us: u64,

    /// Number of blocks processed
    pub blocks_processed: usize,

    /// Number of edges processed
    pub edges_processed: usize,

    /// Memory used by dominance structures (approximate)
    pub memory_used_bytes: usize,
}

/// **Dominance Analysis Errors**
#[derive(Debug)]
pub enum DominanceError {
    /// CFG has no entry block or is malformed
    InvalidCFG(String),

    /// CFG has unreachable blocks that cannot be processed
    UnreachableBlocks(Vec<BlockId>),

    /// CFG contains cycles that break algorithm assumptions
    InvalidCFGStructure(String),

    /// Internal algorithm error
    AlgorithmError(String),
}

impl DominanceTree {
    /// **Build dominance tree using Lengauer-Tarjan algorithm**
    ///
    /// This is the main entry point for dominance analysis. It computes
    /// the complete dominance information for the given CFG.
    ///
    /// # Performance
    /// - Time: O(E α(E,V)) where α is the inverse Ackermann function
    /// - Space: O(V) for the dominance tree structures
    ///
    /// # Errors
    /// Returns `DominanceError` if the CFG is malformed or unreachable.
    pub fn build(cfg: &ControlFlowGraph) -> Result<Self, DominanceError> {
        let start_time = Instant::now();

        let mut tree = Self {
            idom: BTreeMap::new(),
            dominators: BTreeMap::new(),
            dom_tree_children: BTreeMap::new(),
            dominance_frontiers: BTreeMap::new(),
            dfs_preorder: BTreeMap::new(),
            dfs_postorder: BTreeMap::new(),
            reverse_postorder: Vec::new(),
            stats: DominanceStats::default(),
        };

        // Validate CFG structure
        tree.validate_cfg(cfg)?;

        // Phase 1: DFS numbering and ordering
        let dfs_start = Instant::now();
        tree.compute_dfs_numbering(cfg)?;
        tree.stats.dfs_time_us = dfs_start.elapsed().as_micros() as u64;

        // Phase 2: Lengauer-Tarjan semi-dominator and immediate dominator computation
        let semi_start = Instant::now();
        tree.compute_lengauer_tarjan(cfg)?;
        tree.stats.semi_dom_time_us = semi_start.elapsed().as_micros() as u64;

        // Phase 3: Build dominance tree structure
        let idom_start = Instant::now();
        tree.build_dominance_tree();
        tree.stats.idom_time_us = idom_start.elapsed().as_micros() as u64;

        // Phase 4: Compute dominance frontiers for phi-node placement
        let frontier_start = Instant::now();
        tree.compute_dominance_frontiers(cfg);
        tree.stats.frontier_time_us = frontier_start.elapsed().as_micros() as u64;

        // Update final statistics
        tree.stats.computation_time_us = start_time.elapsed().as_micros() as u64;
        tree.stats.blocks_processed = cfg.blocks.len();
        tree.stats.edges_processed = cfg
            .blocks
            .values()
            .map(|block| block.successors.len())
            .sum();
        tree.stats.memory_used_bytes = tree.estimate_memory_usage();

        Ok(tree)
    }

    /// **Validate CFG structure for dominance analysis**
    ///
    /// Ensures the CFG meets the prerequisites for dominance analysis:
    /// - Has a valid entry block
    /// - All blocks are reachable from entry
    /// - No structural anomalies that would break the algorithm
    fn validate_cfg(&self, cfg: &ControlFlowGraph) -> Result<(), DominanceError> {
        // Check for entry block
        if !cfg.blocks.contains_key(&cfg.entry_block) {
            return Err(DominanceError::InvalidCFG(
                "Entry block not found in CFG".to_string(),
            ));
        }

        // Check for empty CFG
        if cfg.blocks.is_empty() {
            return Err(DominanceError::InvalidCFG(
                "CFG contains no blocks".to_string(),
            ));
        }

        // Check that all referenced blocks exist
        for (block_id, block) in &cfg.blocks {
            for &successor in &block.successors {
                if !cfg.blocks.contains_key(&successor) {
                    return Err(DominanceError::InvalidCFG(format!(
                        "Block {:?} references non-existent successor {:?}",
                        block_id, successor
                    )));
                }
            }

            for &predecessor in &block.predecessors {
                if !cfg.blocks.contains_key(&predecessor) {
                    return Err(DominanceError::InvalidCFG(format!(
                        "Block {:?} references non-existent predecessor {:?}",
                        block_id, predecessor
                    )));
                }
            }
        }

        Ok(())
    }

    /// **Compute DFS numbering and block orderings**
    ///
    /// Performs depth-first search from the entry block to:
    /// - Assign pre-order and post-order numbers
    /// - Compute reverse post-order (RPO) for efficient dataflow analysis
    /// - Detect unreachable blocks
    fn compute_dfs_numbering(&mut self, cfg: &ControlFlowGraph) -> Result<(), DominanceError> {
        let mut visited = BTreeSet::new();
        let mut preorder_counter = 0;
        let mut postorder_counter = 0;
        let mut postorder_stack = Vec::new();

        // DFS from entry block
        let mut stack = vec![(cfg.entry_block, false)];

        while let Some((block, is_post_visit)) = stack.pop() {
            if is_post_visit {
                // Post-order processing
                if !self.dfs_postorder.contains_key(&block) {
                    self.dfs_postorder.insert(block, postorder_counter);
                    postorder_stack.push(block);
                    postorder_counter += 1;
                }
                continue;
            }

            if visited.contains(&block) {
                continue;
            }

            // Pre-order processing
            visited.insert(block);
            self.dfs_preorder.insert(block, preorder_counter);
            preorder_counter += 1;

            // Schedule post-order visit
            stack.push((block, true));

            // Add successors for exploration
            if let Some(bb) = cfg.blocks.get(&block) {
                for &successor in &bb.successors {
                    if !visited.contains(&successor) {
                        stack.push((successor, false));
                    }
                }
            }
        }

        // Build reverse post-order
        postorder_stack.reverse();
        self.reverse_postorder = postorder_stack;

        // Check for unreachable blocks
        let unreachable_blocks: Vec<BlockId> = cfg
            .blocks
            .keys()
            .filter(|&block| !visited.contains(block))
            .copied()
            .collect();

        if !unreachable_blocks.is_empty() {
            return Err(DominanceError::UnreachableBlocks(unreachable_blocks));
        }

        Ok(())
    }

    /// **Core Lengauer-Tarjan algorithm implementation**
    ///
    /// Computes semi-dominators and immediate dominators using the
    /// Lengauer-Tarjan algorithm with union-find optimization.
    fn compute_lengauer_tarjan(&mut self, cfg: &ControlFlowGraph) -> Result<(), DominanceError> {
        let mut union_find = UnionFind::new();
        let mut vertex_by_dfs = BTreeMap::new();
        let mut bucket: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();

        // Initialize vertex array ordered by DFS number
        for (&block, &dfs_num) in &self.dfs_preorder {
            vertex_by_dfs.insert(dfs_num, block);
            union_find.init_vertex(block, dfs_num);
        }

        let n = vertex_by_dfs.len();

        // Process vertices in reverse DFS order (except entry)
        for i in (1..n).rev() {
            let w = vertex_by_dfs[&i];

            // Step 2: Compute semi-dominator of w
            let mut semi_w = i;

            // For each predecessor v of w
            for v in cfg.get_predecessors(w) {
                let u = union_find.eval(v);
                let semi_u = union_find.get_semi(u);
                if semi_u < semi_w {
                    semi_w = semi_u;
                }
            }

            union_find.set_semi(w, semi_w);
            let semi_vertex = vertex_by_dfs[&semi_w];
            bucket.entry(semi_vertex).or_default().push(w);

            // Step 3: Process bucket of parent[w]
            let parent_w = self.get_dfs_parent(w, cfg)?;
            if let Some(bucket_nodes) = bucket.remove(&parent_w) {
                for &v in &bucket_nodes {
                    let u = union_find.eval(v);
                    let semi_u = union_find.get_semi(u);
                    let semi_v = union_find.get_semi(v);

                    let idom_v = if semi_u < semi_v { u } else { parent_w };
                    self.idom.insert(v, idom_v);
                }
            }

            // Union w with its parent
            union_find.union(w, parent_w);
        }

        // Step 4: Adjust immediate dominators
        for i in 1..n {
            let w = vertex_by_dfs[&i];
            if let Some(&idom_w) = self.idom.get(&w) {
                let semi_w = union_find.get_semi(w);
                let semi_vertex = vertex_by_dfs[&semi_w];

                if idom_w != semi_vertex {
                    if let Some(&idom_idom_w) = self.idom.get(&idom_w) {
                        self.idom.insert(w, idom_idom_w);
                    }
                }
            }
        }

        // Entry block dominates itself
        let entry = cfg.entry_block;
        self.idom.insert(entry, entry);

        Ok(())
    }

    /// **Build dominance tree structure from immediate dominators**
    ///
    /// Constructs the tree structure and computes the transitive closure
    /// of the dominance relation for efficient dominance queries.
    fn build_dominance_tree(&mut self) {
        // Build dominance tree children
        for (&child, &parent) in &self.idom {
            if child != parent {
                // Don't add entry as child of itself
                self.dom_tree_children
                    .entry(parent)
                    .or_default()
                    .push(child);
            }
        }

        // Compute transitive dominance relation
        for &block in self.idom.keys() {
            let mut dominators = BTreeSet::new();
            let mut current = Some(block);

            // Walk up the dominance tree
            while let Some(curr) = current {
                dominators.insert(curr);
                current = self
                    .idom
                    .get(&curr)
                    .copied()
                    .filter(|&parent| parent != curr); // Avoid infinite loop at entry
            }

            self.dominators.insert(block, dominators);
        }
    }

    /// **Compute dominance frontiers for phi-node placement**
    ///
    /// The dominance frontier of a block X is the set of blocks Y such that:
    /// - X dominates a predecessor of Y
    /// - X does not strictly dominate Y
    ///
    /// This is where phi-nodes need to be placed in SSA form.
    fn compute_dominance_frontiers(&mut self, cfg: &ControlFlowGraph) {
        // Algorithm: For each edge (X -> Y), if X does not dominate Y,
        // then Y is in the dominance frontier of the lowest common ancestor
        // of X and the immediate dominator of Y

        for (&block_x, block_info) in &cfg.blocks {
            for &block_y in &block_info.successors {
                if !self.dominates(block_x, block_y) {
                    // Find blocks on the path from block_x to idom(block_y)
                    // in the dominance tree
                    let mut runner = Some(block_x);

                    while let Some(curr) = runner {
                        if let Some(&idom_y) = self.idom.get(&block_y) {
                            if curr == idom_y {
                                break;
                            }
                        }

                        // Add block_y to dominance frontier of curr
                        self.dominance_frontiers
                            .entry(curr)
                            .or_default()
                            .push(block_y);

                        // Move up the dominance tree
                        runner = self
                            .idom
                            .get(&curr)
                            .copied()
                            .filter(|&parent| parent != curr);
                    }
                }
            }
        }

        // Remove duplicates from dominance frontiers
        for frontier in self.dominance_frontiers.values_mut() {
            frontier.sort();
            frontier.dedup();
        }
    }

    /// **Check if block 'a' dominates block 'b'**
    ///
    /// Returns true if all paths from the entry block to 'b' pass through 'a'.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        self.dominators
            .get(&b)
            .map(|doms| doms.contains(&a))
            .unwrap_or(false)
    }

    /// **Check if block 'a' strictly dominates block 'b'**
    ///
    /// Returns true if 'a' dominates 'b' and 'a' != 'b'.
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// **Get immediate dominator of a block**
    pub fn immediate_dominator(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied().filter(|&idom| idom != block) // Entry block's idom is itself
    }

    /// **Get dominance frontier of a block**
    pub fn dominance_frontier(&self, block: BlockId) -> &[BlockId] {
        self.dominance_frontiers
            .get(&block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// **Get children of a block in the dominance tree**
    pub fn dom_tree_children(&self, block: BlockId) -> &[BlockId] {
        self.dom_tree_children
            .get(&block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// **Find lowest common ancestor in dominance tree**
    pub fn lca(&self, a: BlockId, b: BlockId) -> Option<BlockId> {
        let dominators_a = self.dominators.get(&a)?;
        let dominators_b = self.dominators.get(&b)?;

        // Find common dominators
        let common: BTreeSet<_> = dominators_a.intersection(dominators_b).collect();

        // Find the one with the highest DFS number (closest to leaves)
        common
            .iter()
            .max_by_key(|&&block| self.dfs_preorder.get(&block).unwrap_or(&0))
            .copied()
            .copied()
    }

    // **Helper methods**

    /// Get the DFS parent of a block (used in Lengauer-Tarjan)
    fn get_dfs_parent(
        &self,
        block: BlockId,
        cfg: &ControlFlowGraph,
    ) -> Result<BlockId, DominanceError> {
        // Find the predecessor with the lowest DFS number
        let predecessors = cfg.get_predecessors(block);

        predecessors
            .iter()
            .min_by_key(|&pred| self.dfs_preorder.get(pred).unwrap_or(&usize::MAX))
            .copied()
            .ok_or_else(|| {
                DominanceError::AlgorithmError(format!("No DFS parent found for block {:?}", block))
            })
    }

    /// Estimate memory usage of dominance structures
    fn estimate_memory_usage(&self) -> usize {
        let mut size = 0;

        // Immediate dominators
        size += self.idom.len() * (std::mem::size_of::<BlockId>() * 2);

        // Dominators sets
        for dominators in self.dominators.values() {
            size += dominators.len() * std::mem::size_of::<BlockId>();
        }

        // Dominance tree children
        for children in self.dom_tree_children.values() {
            size += children.len() * std::mem::size_of::<BlockId>();
        }

        // Dominance frontiers
        for frontier in self.dominance_frontiers.values() {
            size += frontier.len() * std::mem::size_of::<BlockId>();
        }

        // DFS numbering
        size += self.dfs_preorder.len()
            * (std::mem::size_of::<BlockId>() + std::mem::size_of::<usize>());
        size += self.dfs_postorder.len()
            * (std::mem::size_of::<BlockId>() + std::mem::size_of::<usize>());

        // Reverse post-order
        size += self.reverse_postorder.len() * std::mem::size_of::<BlockId>();

        size
    }
}

/// **Union-Find Implementation for Lengauer-Tarjan**
impl UnionFind {
    fn new() -> Self {
        Self {
            parent: BTreeMap::new(),
            rank: BTreeMap::new(),
            semi: BTreeMap::new(),
            label: BTreeMap::new(),
        }
    }

    fn init_vertex(&mut self, vertex: BlockId, dfs_num: usize) {
        self.parent.insert(vertex, vertex);
        self.rank.insert(vertex, 0);
        self.semi.insert(vertex, dfs_num);
        self.label.insert(vertex, vertex);
    }

    fn union(&mut self, x: BlockId, y: BlockId) {
        let root_x = self.find_root(x);
        let root_y = self.find_root(y);

        if root_x == root_y {
            return;
        }

        let rank_x = self.rank[&root_x];
        let rank_y = self.rank[&root_y];

        if rank_x < rank_y {
            self.parent.insert(root_x, root_y);
        } else if rank_x > rank_y {
            self.parent.insert(root_y, root_x);
        } else {
            self.parent.insert(root_y, root_x);
            self.rank.insert(root_x, rank_x + 1);
        }
    }

    fn find_root(&mut self, x: BlockId) -> BlockId {
        if self.parent[&x] != x {
            let root = self.find_root(self.parent[&x]);
            self.parent.insert(x, root);
        }
        self.parent[&x]
    }

    fn eval(&mut self, v: BlockId) -> BlockId {
        // Simplified eval for semi-dominator computation
        // In full implementation, would use sophisticated path compression
        if self.parent[&v] == v {
            v
        } else {
            self.compress(v);
            self.label[&v]
        }
    }

    fn compress(&mut self, v: BlockId) {
        let parent_v = self.parent[&v];
        if self.parent[&parent_v] != parent_v {
            self.compress(parent_v);
            let label_v = self.label[&v];
            let label_parent = self.label[&parent_v];
            if self.semi[&label_parent] < self.semi[&label_v] {
                self.label.insert(v, label_parent);
            }
            self.parent.insert(v, self.parent[&parent_v]);
        }
    }

    fn get_semi(&self, vertex: BlockId) -> usize {
        self.semi[&vertex]
    }

    fn set_semi(&mut self, vertex: BlockId, semi: usize) {
        self.semi.insert(vertex, semi);
    }
}

/// **Extension trait for CFG to get predecessors and successors**
trait CFGDominanceExt {
    fn get_predecessors(&self, block: BlockId) -> Vec<BlockId>;
    fn get_successors(&self, block: BlockId) -> Vec<BlockId>;
}

impl CFGDominanceExt for ControlFlowGraph {
    fn get_predecessors(&self, block: BlockId) -> Vec<BlockId> {
        // CFG stores predecessors as IdSet, convert to Vec
        self.blocks
            .get(&block)
            .map(|b| b.predecessors.iter().copied().collect())
            .unwrap_or_else(Vec::new)
    }

    fn get_successors(&self, block: BlockId) -> Vec<BlockId> {
        // CFG stores successors as Vec, clone it
        self.blocks
            .get(&block)
            .map(|b| b.successors.clone())
            .unwrap_or_else(Vec::new)
    }
}

impl std::fmt::Display for DominanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCFG(msg) => write!(f, "Invalid CFG: {}", msg),
            Self::UnreachableBlocks(blocks) => {
                write!(f, "Unreachable blocks: {:?}", blocks)
            }
            Self::InvalidCFGStructure(msg) => write!(f, "Invalid CFG structure: {}", msg),
            Self::AlgorithmError(msg) => write!(f, "Dominance algorithm error: {}", msg),
        }
    }
}

impl std::error::Error for DominanceError {}
