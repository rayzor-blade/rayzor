//! Data Flow Graph (DFG) implementation for advanced static analysis
//!
//! The DFG represents the flow of values through the program in SSA (Static Single Assignment)
//! form, enabling precise data flow analysis, dead code elimination, and optimization.
//!
//! Key features:
//! - SSA form with Phi nodes for control flow merges
//! - Efficient def-use chains for fast analysis
//! - Value numbering for optimization opportunities
//! - Integration with CFG for complete program representation

use crate::tast::collections::{new_id_map, IdMap, IdSet};
use crate::tast::node::{BinaryOperator, UnaryOperator};
use crate::tast::{DataFlowNodeId, InternedString, ScopeId, SsaVariableId, TypeId};

use super::{BlockId, SourceLocation, SymbolId};

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

/// A data flow graph representing value flow in SSA form
#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    /// All data flow nodes indexed by ID
    pub nodes: IdMap<DataFlowNodeId, DataFlowNode>,

    /// Entry node (function parameters and constants)
    pub entry_node: DataFlowNodeId,

    /// Mapping from basic blocks to their data flow nodes
    pub block_nodes: IdMap<BlockId, Vec<DataFlowNodeId>>,

    /// Def-use chains for efficient analysis
    pub def_use_chains: DefUseChains,

    /// SSA variable information
    pub ssa_variables: IdMap<SsaVariableId, SsaVariable>,

    /// Value numbering for optimization
    pub value_numbering: ValueNumbering,

    /// DFG construction metadata
    pub metadata: DfgMetadata,
}

/// Unique identifier for SSA variables
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// pub struct SsaVariableId(u32);

// impl SsaVariableId {
//     pub fn from_raw(raw: u32) -> Self { Self(raw) }
//     pub fn raw(&self) -> u32 { self.0 }
//     pub fn is_valid(&self) -> bool { self.0 != u32::MAX }
//     pub fn invalid() -> Self { Self(u32::MAX) }
// }

/// A node in the data flow graph
#[derive(Debug, Clone)]
pub struct DataFlowNode {
    /// Unique identifier
    pub id: DataFlowNodeId,

    /// Kind of data flow node
    pub kind: DataFlowNodeKind,

    /// Type of value produced by this node
    pub value_type: TypeId,

    /// Source location for diagnostics
    pub source_location: SourceLocation,

    /// Nodes that this node depends on (inputs)
    pub operands: Vec<DataFlowNodeId>,

    /// Nodes that depend on this node (outputs)
    pub uses: IdSet<DataFlowNodeId>,

    /// SSA variable this node defines (if any)
    pub defines: Option<SsaVariableId>,

    /// Basic block containing this node
    pub basic_block: BlockId,

    /// Additional metadata
    pub metadata: NodeMetadata,
}

/// Different kinds of data flow nodes
#[derive(Debug, Clone)]
pub enum DataFlowNodeKind {
    /// Function parameter
    Parameter {
        parameter_index: usize,
        symbol_id: SymbolId,
    },

    /// Constant value
    Constant {
        value: ConstantValue,
    },

    /// Variable reference (SSA)
    Variable {
        ssa_var: SsaVariableId,
    },

    /// Binary operation
    BinaryOp {
        operator: BinaryOperator,
        left: DataFlowNodeId,
        right: DataFlowNodeId,
    },

    /// Unary operation
    UnaryOp {
        operator: UnaryOperator,
        operand: DataFlowNodeId,
    },

    /// Function call
    Call {
        function: DataFlowNodeId,
        arguments: Vec<DataFlowNodeId>,
        call_type: CallType,
    },

    Closure {
        closure_id: DataFlowNodeId,
    },

    /// Memory load operation
    Load {
        address: DataFlowNodeId,
        memory_type: MemoryType,
    },

    /// Memory store operation
    Store {
        address: DataFlowNodeId,
        value: DataFlowNodeId,
        memory_type: MemoryType,
    },

    /// Phi node for SSA form (merges values from different control flow paths)
    Phi {
        /// Incoming values from different predecessors
        incoming: Vec<PhiIncoming>,
    },

    /// Field access
    FieldAccess {
        object: DataFlowNodeId,
        field_symbol: SymbolId,
    },

    /// Static field access
    StaticFieldAccess {
        class_symbol: SymbolId,
        field_symbol: SymbolId,
    },

    /// Array access
    ArrayAccess {
        array: DataFlowNodeId,
        index: DataFlowNodeId,
    },

    /// Type cast
    Cast {
        value: DataFlowNodeId,
        target_type: TypeId,
        cast_kind: CastKind,
    },

    /// Allocation node (for escape analysis)
    Allocation {
        allocation_type: TypeId,
        size: Option<DataFlowNodeId>,
        allocation_kind: AllocationKind,
    },

    /// Return value
    Return {
        value: Option<DataFlowNodeId>,
    },

    /// Exception throw
    Throw {
        exception: DataFlowNodeId,
    },

    /// Type check (instanceof/is) - this is a value operation, not control flow
    TypeCheck {
        operand: DataFlowNodeId,
        check_type: TypeId,
    },

    /// Block expression containing multiple statements
    Block {
        statements: Vec<DataFlowNodeId>,
    },
}

/// Phi node incoming value
#[derive(Debug, Clone)]
pub struct PhiIncoming {
    /// Value from this predecessor
    pub value: DataFlowNodeId,
    /// Which predecessor block this value comes from
    pub predecessor: BlockId,
}

/// Constant values in the DFG
#[derive(Debug, Clone)]
pub enum ConstantValue {
    Void,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Null,
}

impl Eq for ConstantValue {}
impl PartialEq for ConstantValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl std::hash::Hash for ConstantValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
    }
}

impl PartialOrd for ConstantValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConstantValue {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        // Order by discriminant first, then by value within the same variant
        let disc_self = core::mem::discriminant(self);
        let disc_other = core::mem::discriminant(other);
        if disc_self != disc_other {
            // Use a consistent ordering based on variant index
            let idx = |v: &ConstantValue| -> u8 {
                match v {
                    ConstantValue::Void => 0,
                    ConstantValue::Bool(_) => 1,
                    ConstantValue::Int(_) => 2,
                    ConstantValue::Float(_) => 3,
                    ConstantValue::String(_) => 4,
                    ConstantValue::Null => 5,
                }
            };
            return idx(self).cmp(&idx(other));
        }
        match (self, other) {
            (ConstantValue::Void, ConstantValue::Void) => Ordering::Equal,
            (ConstantValue::Null, ConstantValue::Null) => Ordering::Equal,
            (ConstantValue::Bool(a), ConstantValue::Bool(b)) => a.cmp(b),
            (ConstantValue::Int(a), ConstantValue::Int(b)) => a.cmp(b),
            (ConstantValue::Float(a), ConstantValue::Float(b)) => a.total_cmp(b),
            (ConstantValue::String(a), ConstantValue::String(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

/// Types of function calls
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallType {
    /// Direct function call
    Direct,

    /// Virtual method call (dynamic dispatch)
    Virtual,
    /// Static method call
    Static,
    /// Constructor call
    Constructor,
    /// Built-in operation
    Builtin,
}

/// Memory operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// Stack memory
    Stack,
    /// Heap memory
    Heap,
    /// Global memory
    Global,
    /// Field access
    Field(SymbolId),
}

/// Type cast kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    /// Safe implicit cast
    Implicit,
    /// Explicit cast that may fail
    Explicit,
    /// Unsafe cast
    Unsafe,
    /// Runtime type check
    Checked,
}

/// Memory allocation kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationKind {
    /// Stack allocation
    Stack,
    /// Heap allocation
    Heap,
    /// Global allocation
    Global,
    /// Temporary allocation
    Temporary,
}

/// SSA variable information
#[derive(Debug, Clone)]
pub struct SsaVariable {
    /// Unique identifier
    pub id: SsaVariableId,

    /// Original symbol this SSA variable represents
    pub original_symbol: SymbolId,

    /// SSA index (for variables with multiple definitions)
    pub ssa_index: u32,

    /// Type of this variable
    pub var_type: TypeId,

    /// Definition point
    pub definition: DataFlowNodeId,

    /// All use points
    pub uses: Vec<DataFlowNodeId>,

    /// Liveness information
    pub liveness: LivenessInfo,
}

/// Variable liveness information
#[derive(Debug, Clone, Default)]
pub struct LivenessInfo {
    /// Live-in blocks (variable is live at block entry)
    pub live_in: IdSet<BlockId>,

    /// Live-out blocks (variable is live at block exit)
    pub live_out: IdSet<BlockId>,

    /// Definition blocks
    pub def_blocks: IdSet<BlockId>,

    /// Use blocks
    pub use_blocks: IdSet<BlockId>,
}

/// Def-use chains for efficient analysis
#[derive(Debug, Default, Clone)]
pub struct DefUseChains {
    /// Map from definition to all uses
    pub def_to_uses: IdMap<DataFlowNodeId, Vec<DataFlowNodeId>>,

    /// Map from use to its definition
    pub use_to_def: IdMap<DataFlowNodeId, DataFlowNodeId>,

    /// Variables defined in each block
    pub block_defs: IdMap<BlockId, Vec<SsaVariableId>>,

    /// Variables used in each block
    pub block_uses: IdMap<BlockId, Vec<SsaVariableId>>,
}

/// Value numbering for optimization
#[derive(Debug, Default, Clone)]
pub struct ValueNumbering {
    /// Map from value number to canonical expression
    pub value_to_expr: BTreeMap<ValueNumber, CanonicalExpression>,

    /// Map from expression to value number
    pub expr_to_value: BTreeMap<CanonicalExpression, ValueNumber>,

    /// Next available value number
    next_value_number: u32,
}

/// Value number for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueNumber(u32);

impl ValueNumber {
    pub fn new(n: u32) -> Self {
        Self(n)
    }
    pub fn raw(&self) -> u32 {
        self.0
    }
}

/// Canonical expression for value numbering
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CanonicalExpression {
    Constant(ConstantValue),
    BinaryOp {
        op: BinaryOperator,
        left: ValueNumber,
        right: ValueNumber,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: ValueNumber,
    },
    FieldAccess {
        object: ValueNumber,
        field: SymbolId,
    },
    ArrayAccess {
        array: ValueNumber,
        index: ValueNumber,
    },
}

/// Node metadata for analysis
#[derive(Debug, Clone, Default)]
pub struct NodeMetadata {
    /// Whether this node has side effects
    pub has_side_effects: bool,

    /// Whether this node is dead (result unused)
    pub is_dead: bool,

    /// Estimated execution frequency
    pub execution_frequency: f64,

    /// Analysis annotations
    pub annotations: BTreeMap<String, String>,
}

/// DFG construction metadata
#[derive(Debug, Clone, Default)]
pub struct DfgMetadata {
    /// Number of SSA variables created
    pub ssa_variable_count: usize,

    /// Number of Phi nodes created
    pub phi_node_count: usize,

    /// Whether DFG is in SSA form
    pub is_ssa_form: bool,

    /// Construction statistics
    pub construction_stats: DfgConstructionStats,
}

/// Statistics from DFG construction
#[derive(Debug, Clone, Default)]
pub struct DfgConstructionStats {
    /// Time taken to build DFG (microseconds)
    pub construction_time_us: u64,

    /// Number of nodes created
    pub nodes_created: usize,

    /// Number of def-use edges created
    pub def_use_edges: usize,

    /// Memory allocated for DFG (bytes)
    pub memory_bytes: usize,
}

impl DataFlowGraph {
    /// Create a new empty data flow graph
    pub fn new(entry_node: DataFlowNodeId) -> Self {
        Self {
            nodes: new_id_map(),
            entry_node,
            block_nodes: new_id_map(),
            def_use_chains: DefUseChains::default(),
            ssa_variables: new_id_map(),
            value_numbering: ValueNumbering::default(),
            metadata: DfgMetadata::default(),
        }
    }

    /// Add a node to the DFG
    pub fn add_node(&mut self, mut node: DataFlowNode) -> DataFlowNodeId {
        let node_id = node.id;

        // Update def-use chains
        for &operand in &node.operands {
            if let Some(operand_node) = self.nodes.get_mut(&operand) {
                operand_node.uses.insert(node_id);
            }
            self.def_use_chains
                .def_to_uses
                .entry(operand)
                .or_insert_with(Vec::new)
                .push(node_id);
        }

        // Track block nodes
        self.block_nodes
            .entry(node.basic_block)
            .or_insert_with(Vec::new)
            .push(node_id);

        // Add to nodes collection
        self.nodes.insert(node_id, node);

        node_id
    }

    /// Get a node by ID
    pub fn get_node(&self, id: DataFlowNodeId) -> Option<&DataFlowNode> {
        self.nodes.get(&id)
    }

    /// Get mutable node by ID
    pub fn get_node_mut(&mut self, id: DataFlowNodeId) -> Option<&mut DataFlowNode> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes in a basic block
    pub fn nodes_in_block(&self, block_id: BlockId) -> &[DataFlowNodeId] {
        self.block_nodes
            .get(&block_id)
            .map_or(&[], |v| v.as_slice())
    }

    /// Check if DFG is in valid SSA form
    pub fn is_valid_ssa(&self) -> bool {
        // Check that each SSA variable has exactly one definition
        for ssa_var in self.ssa_variables.values() {
            if self.count_definitions(ssa_var.id) != 1 {
                return false;
            }
        }

        // Check that all uses have corresponding definitions
        for node in self.nodes.values() {
            for &operand in &node.operands {
                if !self.nodes.contains_key(&operand) {
                    return false;
                }
            }
        }

        true
    }

    /// Count definitions of an SSA variable
    fn count_definitions(&self, ssa_var: SsaVariableId) -> usize {
        self.nodes
            .values()
            .filter(|node| node.defines == Some(ssa_var))
            .count()
    }

    /// Get all uses of a definition
    pub fn get_uses(&self, def: DataFlowNodeId) -> &[DataFlowNodeId] {
        self.def_use_chains
            .def_to_uses
            .get(&def)
            .map_or(&[], |v| v.as_slice())
    }

    /// Get definition of a use
    pub fn get_definition(&self, use_node: DataFlowNodeId) -> Option<DataFlowNodeId> {
        self.def_use_chains.use_to_def.get(&use_node).copied()
    }

    /// Perform dead code elimination
    pub fn eliminate_dead_code(&mut self) -> usize {
        let mut removed_count = 0;
        let mut worklist: VecDeque<DataFlowNodeId> = VecDeque::new();

        // Find all nodes without uses (potential dead code)
        for (&node_id, node) in &self.nodes {
            if node.uses.is_empty() && !self.has_side_effects(node) {
                worklist.push_back(node_id);
            }
        }

        // Iteratively remove dead nodes
        while let Some(node_id) = worklist.pop_front() {
            if let Some(node) = self.nodes.remove(&node_id) {
                removed_count += 1;

                // If this node defined an SSA variable, remove it from SSA variables map
                if let Some(ssa_var_id) = node.defines {
                    self.ssa_variables.remove(&ssa_var_id);
                }

                // Collect operands to process after removing the node
                let operands_to_check: Vec<DataFlowNodeId> = node.operands.clone();

                // Remove from operands' use lists and check if they become dead
                for &operand in &operands_to_check {
                    let should_add_to_worklist =
                        if let Some(operand_node) = self.nodes.get_mut(&operand) {
                            operand_node.uses.remove(&node_id);

                            // Check if operand becomes dead after removing this use
                            operand_node.uses.is_empty()
                        } else {
                            false
                        };

                    // Check side effects separately to avoid borrow conflicts
                    if should_add_to_worklist {
                        if let Some(operand_node) = self.nodes.get(&operand) {
                            if !self.has_side_effects(operand_node) {
                                worklist.push_back(operand);
                            }
                        }
                    }
                }

                // Remove from def-use chains
                self.def_use_chains.def_to_uses.remove(&node_id);
                for operand in &operands_to_check {
                    if let Some(uses) = self.def_use_chains.def_to_uses.get_mut(operand) {
                        uses.retain(|&use_id| use_id != node_id);
                    }
                    self.def_use_chains.use_to_def.remove(&node_id);
                }

                // Remove from block nodes
                for block_nodes in self.block_nodes.values_mut() {
                    block_nodes.retain(|&id| id != node_id);
                }
            }
        }

        removed_count
    }

    /// Check if a node has side effects
    fn has_side_effects(&self, node: &DataFlowNode) -> bool {
        if node.metadata.has_side_effects {
            return true;
        }

        match &node.kind {
            DataFlowNodeKind::Call { .. } => true,
            DataFlowNodeKind::Store { .. } => true,
            DataFlowNodeKind::Return { .. } => true,
            DataFlowNodeKind::Throw { .. } => true,
            DataFlowNodeKind::Allocation { .. } => true,
            _ => false,
        }
    }

    /// Get statistics about the DFG
    pub fn statistics(&self) -> DfgStatistics {
        let node_count = self.nodes.len();
        let edge_count = self.nodes.values().map(|n| n.operands.len()).sum();
        let phi_node_count = self
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Phi { .. }))
            .count();
        let constant_count = self
            .nodes
            .values()
            .filter(|n| matches!(n.kind, DataFlowNodeKind::Constant { .. }))
            .count();

        DfgStatistics {
            node_count,
            edge_count,
            phi_node_count,
            constant_count,
            ssa_variable_count: self.ssa_variables.len(),
            max_use_count: self.nodes.values().map(|n| n.uses.len()).max().unwrap_or(0),
        }
    }
}

/// Statistics about a DFG
#[derive(Debug, Clone)]
pub struct DfgStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub phi_node_count: usize,
    pub constant_count: usize,
    pub ssa_variable_count: usize,
    pub max_use_count: usize,
}

impl ValueNumbering {
    /// Get or create value number for an expression
    pub fn get_value_number(&mut self, expr: CanonicalExpression) -> ValueNumber {
        if let Some(&value_number) = self.expr_to_value.get(&expr) {
            value_number
        } else {
            let value_number = ValueNumber::new(self.next_value_number);
            self.next_value_number += 1;

            self.expr_to_value.insert(expr.clone(), value_number);
            self.value_to_expr.insert(value_number, expr);

            value_number
        }
    }

    /// Check if two expressions have the same value
    pub fn are_equivalent(&self, expr1: &CanonicalExpression, expr2: &CanonicalExpression) -> bool {
        self.expr_to_value.get(expr1) == self.expr_to_value.get(expr2)
    }
}

impl fmt::Display for DataFlowNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataFlowNodeKind::Parameter {
                parameter_index, ..
            } => {
                write!(f, "param_{}", parameter_index)
            }
            DataFlowNodeKind::Constant { value } => {
                write!(f, "const_{:?}", value)
            }
            DataFlowNodeKind::Variable { ssa_var } => {
                write!(f, "var_{}", ssa_var.as_raw())
            }
            DataFlowNodeKind::BinaryOp { operator, .. } => {
                write!(f, "{:?}", operator)
            }
            DataFlowNodeKind::UnaryOp { operator, .. } => {
                write!(f, "{:?}", operator)
            }
            DataFlowNodeKind::Call { call_type, .. } => {
                write!(f, "call_{:?}", call_type)
            }
            DataFlowNodeKind::Phi { incoming } => {
                write!(f, "phi_{}", incoming.len())
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tast::collections::new_id_set;

    use super::*;

    #[test]
    fn test_dfg_creation() {
        let entry_node = DataFlowNodeId::from_raw(1);
        let dfg = DataFlowGraph::new(entry_node);

        assert_eq!(dfg.entry_node, entry_node);
        assert!(dfg.nodes.is_empty());
        assert!(dfg.is_valid_ssa());
    }

    #[test]
    fn test_node_addition() {
        let mut dfg = DataFlowGraph::new(DataFlowNodeId::from_raw(1));

        let node = DataFlowNode {
            id: DataFlowNodeId::from_raw(1),
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(42),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };

        let node_id = dfg.add_node(node);

        assert_eq!(dfg.nodes.len(), 1);
        assert!(dfg.get_node(node_id).is_some());
    }

    #[test]
    fn test_def_use_chains() {
        let mut dfg = DataFlowGraph::new(DataFlowNodeId::from_raw(1));

        // Add constant node
        let const_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(1),
            kind: DataFlowNodeKind::Constant {
                value: ConstantValue::Int(42),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(const_node);

        // Add use node
        let use_node = DataFlowNode {
            id: DataFlowNodeId::from_raw(2),
            kind: DataFlowNodeKind::UnaryOp {
                operator: UnaryOperator::Neg,
                operand: DataFlowNodeId::from_raw(1),
            },
            value_type: TypeId::from_raw(1),
            source_location: SourceLocation::unknown(),
            operands: vec![DataFlowNodeId::from_raw(1)],
            uses: new_id_set(),
            defines: None,
            basic_block: BlockId::from_raw(1),
            metadata: NodeMetadata::default(),
        };
        dfg.add_node(use_node);

        // Check def-use chain
        let uses = dfg.get_uses(DataFlowNodeId::from_raw(1));
        assert_eq!(uses.len(), 1);
        assert_eq!(uses[0], DataFlowNodeId::from_raw(2));
    }

    #[test]
    fn test_value_numbering() {
        let mut vn = ValueNumbering::default();

        let expr1 = CanonicalExpression::Constant(ConstantValue::Int(42));
        let expr2 = CanonicalExpression::Constant(ConstantValue::Int(42));
        let expr3 = CanonicalExpression::Constant(ConstantValue::Int(43));

        let vn1 = vn.get_value_number(expr1.clone());
        let vn2 = vn.get_value_number(expr2.clone());
        let vn3 = vn.get_value_number(expr3.clone());

        assert_eq!(vn1, vn2); // Same constant should have same value number
        assert_ne!(vn1, vn3); // Different constants should have different value numbers

        assert!(vn.are_equivalent(&expr1, &expr2));
        assert!(!vn.are_equivalent(&expr1, &expr3));
    }

    #[test]
    fn test_dfg_statistics() {
        let mut dfg = DataFlowGraph::new(DataFlowNodeId::from_raw(1));

        // Add some nodes
        for i in 1..=5 {
            let node = DataFlowNode {
                id: DataFlowNodeId::from_raw(i),
                kind: DataFlowNodeKind::Constant {
                    value: ConstantValue::Int(i as i64),
                },
                value_type: TypeId::from_raw(1),
                source_location: SourceLocation::unknown(),
                operands: vec![],
                uses: new_id_set(),
                defines: None,
                basic_block: BlockId::from_raw(1),
                metadata: NodeMetadata::default(),
            };
            dfg.add_node(node);
        }

        let stats = dfg.statistics();
        assert_eq!(stats.node_count, 5);
        assert_eq!(stats.constant_count, 5);
        assert_eq!(stats.phi_node_count, 0);
    }
}
