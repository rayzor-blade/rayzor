//! HIR Optimization Passes
//!
//! This module implements various optimization passes for the HIR.
//! Optimizations are organized into passes that can be run independently
//! and in different orders based on optimization level.

use super::{
    BinaryOp, CompareOp, IrBasicBlock, IrBlockId, IrFunction, IrFunctionId, IrGlobalId, IrId,
    IrInstruction, IrModule, IrTerminator, IrType, IrValue,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Optimization pass trait
pub trait OptimizationPass {
    /// Get the name of this pass
    fn name(&self) -> &'static str;

    /// Run the pass on a module
    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult;

    /// Run the pass on a function (default implementation does nothing)
    fn run_on_function(&mut self, _function: &mut IrFunction) -> OptimizationResult {
        OptimizationResult::unchanged()
    }
}

/// Result of an optimization pass
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Whether the IR was modified
    pub modified: bool,

    /// Number of instructions eliminated
    pub instructions_eliminated: usize,

    /// Number of blocks eliminated
    pub blocks_eliminated: usize,

    /// Other statistics
    pub stats: HashMap<String, usize>,
}

impl OptimizationResult {
    /// Create a result indicating no changes
    pub fn unchanged() -> Self {
        Self {
            modified: false,
            instructions_eliminated: 0,
            blocks_eliminated: 0,
            stats: HashMap::new(),
        }
    }

    /// Create a result indicating changes
    pub fn changed() -> Self {
        Self {
            modified: true,
            instructions_eliminated: 0,
            blocks_eliminated: 0,
            stats: HashMap::new(),
        }
    }

    /// Combine results
    pub fn combine(mut self, other: OptimizationResult) -> Self {
        self.modified |= other.modified;
        self.instructions_eliminated += other.instructions_eliminated;
        self.blocks_eliminated += other.blocks_eliminated;

        for (key, value) in other.stats {
            *self.stats.entry(key).or_insert(0) += value;
        }

        self
    }
}

/// Optimization pass manager
pub struct PassManager {
    passes: Vec<Box<dyn OptimizationPass>>,
}

impl PassManager {
    /// Create a new pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the manager
    pub fn add_pass<P: OptimizationPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Build a default optimization pipeline
    pub fn default_pipeline() -> Self {
        let mut manager = Self::new();

        // Dead code elimination
        manager.add_pass(DeadCodeEliminationPass::new());

        // Constant folding
        manager.add_pass(ConstantFoldingPass::new());

        // Copy propagation
        manager.add_pass(CopyPropagationPass::new());

        // Unreachable block elimination
        manager.add_pass(UnreachableBlockEliminationPass::new());

        // Simplify control flow
        manager.add_pass(ControlFlowSimplificationPass::new());

        manager
    }

    /// Run all passes on a module.
    /// Only re-iterates when a non-cleanup pass modifies the module.
    pub fn run(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut total_result = OptimizationResult::unchanged();
        let max_pipeline_iterations = 5;

        for _pipeline_iter in 0..max_pipeline_iterations {
            let mut transformative_change = false;

            for pass in &mut self.passes {
                let result = pass.run_on_module(module);
                if result.modified {
                    // Only re-iterate if a transformative pass (not just cleanup) changed things
                    let is_cleanup = matches!(
                        pass.name(),
                        "dead-code-elimination"
                            | "unreachable-block-elimination"
                            | "copy-propagation"
                    );
                    if !is_cleanup {
                        transformative_change = true;
                    }
                }
                total_result = total_result.combine(result);
            }

            if !transformative_change {
                break;
            }
        }

        total_result
    }
}

/// Dead code elimination pass
pub struct DeadCodeEliminationPass {
    // Configuration options can go here
}

impl DeadCodeEliminationPass {
    pub fn new() -> Self {
        Self {}
    }

    /// Find all used registers in a function
    fn find_used_registers(&self, function: &IrFunction) -> HashSet<IrId> {
        let mut used = HashSet::new();

        for block in function.cfg.blocks.values() {
            // Mark phi node uses
            for phi in &block.phi_nodes {
                for &(_, value) in &phi.incoming {
                    used.insert(value);
                }
            }

            // Mark instruction uses
            for inst in &block.instructions {
                used.extend(inst.uses());
            }

            // Mark terminator uses
            used.extend(terminator_uses(&block.terminator));
        }

        used
    }

    /// Remove dead instructions from a function
    fn eliminate_dead_instructions(&self, function: &mut IrFunction) -> usize {
        let used = self.find_used_registers(function);
        let mut eliminated = 0;

        for block in function.cfg.blocks.values_mut() {
            // Remove dead phi nodes
            block.phi_nodes.retain(|phi| used.contains(&phi.dest));

            // Remove dead instructions
            let original_len = block.instructions.len();
            block.instructions.retain(|inst| {
                if let Some(dest) = inst.dest() {
                    used.contains(&dest) || inst.has_side_effects()
                } else {
                    true // Instructions without destinations are kept
                }
            });
            eliminated += original_len - block.instructions.len();
        }

        eliminated
    }
}

impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            let eliminated = self.eliminate_dead_instructions(function);
            if eliminated > 0 {
                result.modified = true;
                result.instructions_eliminated += eliminated;
            }
        }

        result
    }
}

/// Constant folding pass
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    pub fn new() -> Self {
        Self
    }

    /// Try to fold a binary operation
    fn fold_binary_op(&self, op: BinaryOp, left: &IrValue, right: &IrValue) -> Option<IrValue> {
        use BinaryOp::*;
        use IrValue::*;

        match (op, left, right) {
            // Integer arithmetic
            (Add, I32(a), I32(b)) => Some(I32(a.wrapping_add(*b))),
            (Sub, I32(a), I32(b)) => Some(I32(a.wrapping_sub(*b))),
            (Mul, I32(a), I32(b)) => Some(I32(a.wrapping_mul(*b))),
            (Div, I32(a), I32(b)) if *b != 0 => Some(I32(a / b)),
            (Rem, I32(a), I32(b)) if *b != 0 => Some(I32(a % b)),

            // Floating point arithmetic
            (FAdd, F64(a), F64(b)) => Some(F64(a + b)),
            (FSub, F64(a), F64(b)) => Some(F64(a - b)),
            (FMul, F64(a), F64(b)) => Some(F64(a * b)),
            (FDiv, F64(a), F64(b)) if *b != 0.0 => Some(F64(a / b)),

            // Bitwise operations
            (And, I32(a), I32(b)) => Some(I32(a & b)),
            (Or, I32(a), I32(b)) => Some(I32(a | b)),
            (Xor, I32(a), I32(b)) => Some(I32(a ^ b)),
            (Shl, I32(a), I32(b)) if *b >= 0 && *b < 32 => Some(I32(a << b)),
            (Shr, I32(a), I32(b)) if *b >= 0 && *b < 32 => Some(I32(a >> b)),

            _ => None,
        }
    }

    /// Try to fold a comparison
    fn fold_comparison(&self, op: CompareOp, left: &IrValue, right: &IrValue) -> Option<IrValue> {
        use CompareOp::*;
        use IrValue::*;

        match (op, left, right) {
            // Integer comparisons
            (Eq, I32(a), I32(b)) => Some(Bool(a == b)),
            (Ne, I32(a), I32(b)) => Some(Bool(a != b)),
            (Lt, I32(a), I32(b)) => Some(Bool(a < b)),
            (Le, I32(a), I32(b)) => Some(Bool(a <= b)),
            (Gt, I32(a), I32(b)) => Some(Bool(a > b)),
            (Ge, I32(a), I32(b)) => Some(Bool(a >= b)),

            // Floating point comparisons
            (FEq, F64(a), F64(b)) => Some(Bool((a - b).abs() < f64::EPSILON)),
            (FNe, F64(a), F64(b)) => Some(Bool((a - b).abs() >= f64::EPSILON)),
            (FLt, F64(a), F64(b)) => Some(Bool(a < b)),
            (FLe, F64(a), F64(b)) => Some(Bool(a <= b)),
            (FGt, F64(a), F64(b)) => Some(Bool(a > b)),
            (FGe, F64(a), F64(b)) => Some(Bool(a >= b)),

            // Boolean comparisons
            (Eq, Bool(a), Bool(b)) => Some(Bool(a == b)),
            (Ne, Bool(a), Bool(b)) => Some(Bool(a != b)),

            _ => None,
        }
    }
}

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &'static str {
        "constant-folding"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        // Build constant value map
        let mut constants: HashMap<IrId, IrValue> = HashMap::new();

        for function in module.functions.values_mut() {
            constants.clear();

            // First pass: collect constants
            for block in function.cfg.blocks.values() {
                for inst in &block.instructions {
                    if let IrInstruction::Const { dest, value } = inst {
                        constants.insert(*dest, value.clone());
                    }
                }
            }

            // Second pass: fold operations
            for block in function.cfg.blocks.values_mut() {
                for inst in &mut block.instructions {
                    match inst {
                        IrInstruction::BinOp {
                            dest,
                            op,
                            left,
                            right,
                        } => {
                            let dest_reg = *dest;
                            let op_val = *op;
                            let left_reg = *left;
                            let right_reg = *right;
                            if let (Some(left_val), Some(right_val)) =
                                (constants.get(&left_reg), constants.get(&right_reg))
                            {
                                if let Some(folded) =
                                    self.fold_binary_op(op_val, left_val, right_val)
                                {
                                    // Replace with constant
                                    *inst = IrInstruction::Const {
                                        dest: dest_reg,
                                        value: folded.clone(),
                                    };
                                    constants.insert(dest_reg, folded);
                                    result.modified = true;
                                }
                            }
                        }
                        IrInstruction::Cmp {
                            dest,
                            op,
                            left,
                            right,
                        } => {
                            let dest_reg = *dest;
                            let op_val = *op;
                            let left_reg = *left;
                            let right_reg = *right;
                            if let (Some(left_val), Some(right_val)) =
                                (constants.get(&left_reg), constants.get(&right_reg))
                            {
                                if let Some(folded) =
                                    self.fold_comparison(op_val, left_val, right_val)
                                {
                                    // Replace with constant
                                    *inst = IrInstruction::Const {
                                        dest: dest_reg,
                                        value: folded.clone(),
                                    };
                                    constants.insert(dest_reg, folded);
                                    result.modified = true;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        result
    }
}

/// Copy propagation pass
pub struct CopyPropagationPass;

impl CopyPropagationPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for CopyPropagationPass {
    fn name(&self) -> &'static str {
        "copy-propagation"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            // Use BTreeMap for deterministic iteration order
            let mut copies: BTreeMap<IrId, IrId> = BTreeMap::new();
            // Track registers with multiple copy definitions — these cannot be safely propagated
            let mut multi_def: BTreeSet<IrId> = BTreeSet::new();

            // Find copy instructions
            for block in function.cfg.blocks.values() {
                for inst in &block.instructions {
                    if let IrInstruction::Copy { dest, src } = inst {
                        if copies.contains_key(dest) {
                            multi_def.insert(*dest);
                        }
                        copies.insert(*dest, *src);
                    }
                }
            }

            // Remove multi-defined registers — they have different values on different paths
            for id in &multi_def {
                copies.remove(id);
            }

            if !copies.is_empty() {
                // Replace uses with original sources
                for block in function.cfg.blocks.values_mut() {
                    for inst in &mut block.instructions {
                        inst.replace_uses(&copies);
                    }

                    // Replace terminator uses
                    replace_terminator_uses(&mut block.terminator, &copies);
                }

                result.modified = true;
            }
        }

        result
    }
}

/// Unreachable block elimination pass
pub struct UnreachableBlockEliminationPass;

impl UnreachableBlockEliminationPass {
    pub fn new() -> Self {
        Self
    }

    /// Find reachable blocks from entry
    fn find_reachable(&self, function: &IrFunction) -> HashSet<IrBlockId> {
        let mut reachable = HashSet::new();
        let mut worklist = vec![function.entry_block()];

        while let Some(block_id) = worklist.pop() {
            if reachable.insert(block_id) {
                if let Some(block) = function.cfg.get_block(block_id) {
                    worklist.extend(block.successors());
                }
            }
        }

        reachable
    }
}

impl OptimizationPass for UnreachableBlockEliminationPass {
    fn name(&self) -> &'static str {
        "unreachable-block-elimination"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            let reachable = self.find_reachable(function);
            let original_count = function.cfg.blocks.len();

            // Remove unreachable blocks
            function.cfg.blocks.retain(|&id, _| reachable.contains(&id));

            let eliminated = original_count - function.cfg.blocks.len();
            if eliminated > 0 {
                result.modified = true;
                result.blocks_eliminated += eliminated;

                // Clean up phi nodes: remove incoming edges from eliminated blocks
                for block in function.cfg.blocks.values_mut() {
                    for phi in &mut block.phi_nodes {
                        phi.incoming
                            .retain(|(pred_block, _)| reachable.contains(pred_block));
                    }
                }
            }
        }

        result
    }
}

/// Control flow simplification pass
pub struct ControlFlowSimplificationPass;

impl ControlFlowSimplificationPass {
    pub fn new() -> Self {
        Self
    }

    /// Simplify branches with constant conditions
    fn simplify_conditional_branches(&self, function: &mut IrFunction) -> bool {
        let mut modified = false;

        // Collect constant values
        let mut constants: HashMap<IrId, IrValue> = HashMap::new();
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                if let IrInstruction::Const { dest, value } = inst {
                    constants.insert(*dest, value.clone());
                }
            }
        }

        // Simplify conditional branches
        for block in function.cfg.blocks.values_mut() {
            if let IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } = &block.terminator
            {
                if let Some(IrValue::Bool(cond_val)) = constants.get(condition) {
                    // Replace with unconditional branch
                    let target = if *cond_val {
                        *true_target
                    } else {
                        *false_target
                    };
                    block.terminator = IrTerminator::Branch { target };
                    modified = true;
                }
            }
        }

        modified
    }
}

impl OptimizationPass for ControlFlowSimplificationPass {
    fn name(&self) -> &'static str {
        "control-flow-simplification"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            if self.simplify_conditional_branches(function) {
                result.modified = true;
            }
        }

        result
    }
}

// Helper functions

/// Get registers used by a terminator
fn terminator_uses(term: &IrTerminator) -> Vec<IrId> {
    match term {
        IrTerminator::CondBranch { condition, .. } => vec![*condition],
        IrTerminator::Switch { value, .. } => vec![*value],
        IrTerminator::Return { value: Some(val) } => vec![*val],
        IrTerminator::NoReturn { call } => vec![*call],
        _ => Vec::new(),
    }
}

/// Replace register uses in a terminator
fn replace_terminator_uses(term: &mut IrTerminator, replacements: &BTreeMap<IrId, IrId>) {
    match term {
        IrTerminator::CondBranch { condition, .. } => {
            if let Some(&new_reg) = replacements.get(condition) {
                *condition = new_reg;
            }
        }
        IrTerminator::Switch { value, .. } => {
            if let Some(&new_reg) = replacements.get(value) {
                *value = new_reg;
            }
        }
        IrTerminator::Return { value: Some(val) } => {
            if let Some(&new_reg) = replacements.get(val) {
                *val = new_reg;
            }
        }
        IrTerminator::NoReturn { call } => {
            if let Some(&new_reg) = replacements.get(call) {
                *call = new_reg;
            }
        }
        _ => {}
    }
}

// Extension trait for instruction manipulation
pub(super) trait InstructionExt {
    fn uses(&self) -> Vec<IrId>;
    fn dest(&self) -> Option<IrId>;
    fn has_side_effects(&self) -> bool;
    fn replace_uses(&mut self, replacements: &BTreeMap<IrId, IrId>);
    fn collect_uses(&self, set: &mut HashSet<IrId>);
}

impl InstructionExt for IrInstruction {
    fn uses(&self) -> Vec<IrId> {
        // Use the inherent IrInstruction::uses() method which is complete
        IrInstruction::uses(self)
    }

    fn dest(&self) -> Option<IrId> {
        // Use the inherent IrInstruction::dest() method which is complete
        IrInstruction::dest(self)
    }

    fn has_side_effects(&self) -> bool {
        // Use the inherent IrInstruction::has_side_effects() method which is complete
        IrInstruction::has_side_effects(self)
    }

    fn replace_uses(&mut self, replacements: &BTreeMap<IrId, IrId>) {
        // Replace uses in all instruction operands
        let replace = |id: &mut IrId| {
            if let Some(&new_id) = replacements.get(id) {
                *id = new_id;
            }
        };

        match self {
            IrInstruction::Copy { src, .. } => replace(src),
            IrInstruction::Load { ptr, .. } => replace(ptr),
            IrInstruction::Store { ptr, value } => {
                replace(ptr);
                replace(value);
            }
            IrInstruction::BinOp { left, right, .. } => {
                replace(left);
                replace(right);
            }
            IrInstruction::UnOp { operand, .. } => replace(operand),
            IrInstruction::Cmp { left, right, .. } => {
                replace(left);
                replace(right);
            }
            IrInstruction::CallDirect { args, .. } => {
                for arg in args {
                    replace(arg);
                }
            }
            IrInstruction::CallIndirect { func_ptr, args, .. } => {
                replace(func_ptr);
                for arg in args {
                    replace(arg);
                }
            }
            IrInstruction::Cast { src, .. } | IrInstruction::BitCast { src, .. } => replace(src),
            IrInstruction::Select {
                condition,
                true_val,
                false_val,
                ..
            } => {
                replace(condition);
                replace(true_val);
                replace(false_val);
            }
            IrInstruction::Free { ptr } => replace(ptr),
            IrInstruction::Alloc { count, .. } => {
                if let Some(c) = count {
                    replace(c);
                }
            }
            IrInstruction::GetElementPtr { ptr, indices, .. } => {
                replace(ptr);
                for idx in indices {
                    replace(idx);
                }
            }
            IrInstruction::MemCopy { dest, src, size } => {
                replace(dest);
                replace(src);
                replace(size);
            }
            IrInstruction::MemSet { dest, value, size } => {
                replace(dest);
                replace(value);
                replace(size);
            }
            IrInstruction::PtrAdd { ptr, offset, .. } => {
                replace(ptr);
                replace(offset);
            }
            IrInstruction::MakeClosure {
                captured_values, ..
            } => {
                for val in captured_values {
                    replace(val);
                }
            }
            IrInstruction::ClosureFunc { closure, .. } => replace(closure),
            IrInstruction::ExtractValue { aggregate, .. } => replace(aggregate),
            IrInstruction::InsertValue {
                aggregate, value, ..
            } => {
                replace(aggregate);
                replace(value);
            }
            IrInstruction::Return { value } => {
                if let Some(v) = value {
                    replace(v);
                }
            }
            IrInstruction::Throw { exception } | IrInstruction::Resume { exception } => {
                replace(exception)
            }
            IrInstruction::Phi { incoming, .. } => {
                for (val, _) in incoming {
                    replace(val);
                }
            }
            IrInstruction::Branch { condition, .. } => replace(condition),
            IrInstruction::Switch { value, .. } => replace(value),
            IrInstruction::InlineAsm { inputs, .. } => {
                for (_, id) in inputs {
                    replace(id);
                }
                // outputs contains IrType, not IrId, so no replacement needed
            }
            IrInstruction::ClosureEnv { closure, .. } => replace(closure),
            // Vector / SIMD instructions
            IrInstruction::VectorLoad { ptr, .. } => replace(ptr),
            IrInstruction::VectorStore { ptr, value, .. } => {
                replace(ptr);
                replace(value);
            }
            IrInstruction::VectorBinOp { left, right, .. } => {
                replace(left);
                replace(right);
            }
            IrInstruction::VectorSplat { scalar, .. } => replace(scalar),
            IrInstruction::VectorExtract { vector, .. } => replace(vector),
            IrInstruction::VectorInsert { vector, scalar, .. } => {
                replace(vector);
                replace(scalar);
            }
            IrInstruction::VectorReduce { vector, .. } => replace(vector),
            IrInstruction::VectorUnaryOp { operand, .. } => replace(operand),
            IrInstruction::VectorMinMax { left, right, .. } => {
                replace(left);
                replace(right);
            }
            // Move/Clone instructions
            IrInstruction::Move { src, .. } => replace(src),
            IrInstruction::Clone { src, .. } => replace(src),
            // Instructions with no uses to replace
            IrInstruction::Const { .. }
            | IrInstruction::Jump { .. }
            | IrInstruction::BorrowImmutable { .. }
            | IrInstruction::BorrowMutable { .. }
            | IrInstruction::EndBorrow { .. }
            | IrInstruction::LandingPad { .. }
            | IrInstruction::Undef { .. }
            | IrInstruction::FunctionRef { .. }
            | IrInstruction::DebugLoc { .. } => {}
            // Handle any other instructions by doing nothing (they may have no register uses)
            _ => {}
        }
    }

    fn collect_uses(&self, set: &mut HashSet<IrId>) {
        for id in self.uses() {
            set.insert(id);
        }
    }
}

/// Helper to collect uses from a terminator
fn collect_terminator_uses(terminator: &IrTerminator, set: &mut HashSet<IrId>) {
    match terminator {
        IrTerminator::Return { value: Some(v) } => {
            set.insert(*v);
        }
        IrTerminator::CondBranch { condition, .. } => {
            set.insert(*condition);
        }
        IrTerminator::Switch { value, .. } => {
            set.insert(*value);
        }
        _ => {}
    }
}

/// Loop Invariant Code Motion (LICM) pass
///
/// Moves loop-invariant computations out of loops to reduce redundant work.
/// An instruction is loop-invariant if all its operands are:
/// - Defined outside the loop, OR
/// - Are themselves loop-invariant
pub struct LICMPass;

impl LICMPass {
    pub fn new() -> Self {
        Self
    }

    /// Check if an instruction is loop-invariant.
    fn is_loop_invariant(
        inst: &IrInstruction,
        loop_blocks: &HashSet<IrBlockId>,
        def_block: &HashMap<IrId, IrBlockId>,
        invariant_defs: &HashSet<IrId>,
    ) -> bool {
        // Instructions with side effects are not loop-invariant
        if inst.has_side_effects() {
            return false;
        }

        // Check if all uses are defined outside the loop or are invariant
        for use_id in inst.uses() {
            if let Some(&def_blk) = def_block.get(&use_id) {
                if loop_blocks.contains(&def_blk) && !invariant_defs.contains(&use_id) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if it's safe to hoist an instruction.
    fn is_safe_to_hoist(
        inst: &IrInstruction,
        inst_block: IrBlockId,
        loop_info: &super::loop_analysis::NaturalLoop,
        domtree: &super::loop_analysis::DominatorTree,
    ) -> bool {
        // Must dominate all exit blocks
        for &exit in &loop_info.exit_blocks {
            if !domtree.dominates(inst_block, exit) {
                return false;
            }
        }

        // Don't hoist instructions that could trap (division, etc.)
        match inst {
            IrInstruction::BinOp { op, .. } => {
                !matches!(op, BinaryOp::Div | BinaryOp::Rem | BinaryOp::FDiv)
            }
            _ => true,
        }
    }

    /// Create a preheader block for a loop if one doesn't exist.
    fn ensure_preheader(
        cfg: &mut super::IrControlFlowGraph,
        header: IrBlockId,
        loop_blocks: &HashSet<IrBlockId>,
    ) -> IrBlockId {
        let header_block = cfg.get_block(header).unwrap();

        // Find predecessors outside the loop
        let outside_preds: Vec<IrBlockId> = header_block
            .predecessors
            .iter()
            .filter(|p| !loop_blocks.contains(p))
            .copied()
            .collect();

        // If there's already a valid preheader, use it
        if outside_preds.len() == 1 {
            let pred = outside_preds[0];
            if let Some(pred_block) = cfg.get_block(pred) {
                if pred_block.successors().len() == 1 {
                    return pred;
                }
            }
        }

        // Create a new preheader
        let preheader = cfg.create_block();

        // Set up preheader terminator to branch to header
        if let Some(preheader_block) = cfg.get_block_mut(preheader) {
            preheader_block.terminator = IrTerminator::Branch { target: header };
        }

        // Update outside predecessors to branch to preheader instead of header
        for &pred in &outside_preds {
            if let Some(pred_block) = cfg.get_block_mut(pred) {
                match &mut pred_block.terminator {
                    IrTerminator::Branch { target } if *target == header => {
                        *target = preheader;
                    }
                    IrTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        if *true_target == header {
                            *true_target = preheader;
                        }
                        if *false_target == header {
                            *false_target = preheader;
                        }
                    }
                    IrTerminator::Switch { cases, default, .. } => {
                        for (_, target) in cases.iter_mut() {
                            if *target == header {
                                *target = preheader;
                            }
                        }
                        if *default == header {
                            *default = preheader;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Update header's predecessors
        if let Some(header_block) = cfg.get_block_mut(header) {
            header_block
                .predecessors
                .retain(|p| loop_blocks.contains(p));
            header_block.predecessors.push(preheader);
        }

        // Set preheader's predecessors
        if let Some(preheader_block) = cfg.get_block_mut(preheader) {
            preheader_block.predecessors = outside_preds;
        }

        preheader
    }
}

impl OptimizationPass for LICMPass {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        use super::loop_analysis::{DominatorTree, LoopNestInfo};

        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            let domtree = DominatorTree::compute(function);
            let loop_info = LoopNestInfo::analyze(function, &domtree);

            if loop_info.loops.is_empty() {
                continue;
            }

            // Build definition site map: register -> block where it's defined
            let mut def_block: HashMap<IrId, IrBlockId> = HashMap::new();
            for (&block_id, block) in &function.cfg.blocks {
                for phi in &block.phi_nodes {
                    def_block.insert(phi.dest, block_id);
                }
                for inst in &block.instructions {
                    if let Some(dest) = inst.dest() {
                        def_block.insert(dest, block_id);
                    }
                }
            }

            // Process loops from innermost to outermost
            for loop_data in loop_info.loops_innermost_first() {
                let mut invariant_defs: HashSet<IrId> = HashSet::new();
                let mut to_hoist: Vec<(IrBlockId, usize, IrInstruction)> = Vec::new();

                // Iterate until no more invariants found
                let mut changed = true;
                while changed {
                    changed = false;

                    for &block_id in &loop_data.blocks {
                        if block_id == loop_data.header {
                            continue; // Don't hoist from header
                        }

                        if let Some(block) = function.cfg.get_block(block_id) {
                            for (idx, inst) in block.instructions.iter().enumerate() {
                                if let Some(dest) = inst.dest() {
                                    if invariant_defs.contains(&dest) {
                                        continue;
                                    }

                                    if Self::is_loop_invariant(
                                        inst,
                                        &loop_data.blocks,
                                        &def_block,
                                        &invariant_defs,
                                    ) && Self::is_safe_to_hoist(
                                        inst, block_id, loop_data, &domtree,
                                    ) {
                                        invariant_defs.insert(dest);
                                        to_hoist.push((block_id, idx, inst.clone()));
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // Phase 2: Escape analysis for non-escaping Alloc hoisting
                let alloc_infos = super::escape_analysis::analyze_alloc_escapes(
                    &function.cfg,
                    &loop_data.blocks,
                    loop_data.header,
                    &[loop_data.back_edge_source],
                );

                let mut alloc_to_hoist: Vec<(IrBlockId, usize, IrInstruction)> = Vec::new();
                let mut free_to_remove: Vec<(IrBlockId, usize)> = Vec::new();
                let mut free_insts_to_sink: Vec<IrInstruction> = Vec::new();

                for info in &alloc_infos {
                    if !info.escapes {
                        if let Some(free_loc) = info.free_location {
                            // Grab the Alloc instruction
                            if let Some(block) = function.cfg.get_block(info.alloc_location.0) {
                                if info.alloc_location.1 < block.instructions.len() {
                                    alloc_to_hoist.push((
                                        info.alloc_location.0,
                                        info.alloc_location.1,
                                        block.instructions[info.alloc_location.1].clone(),
                                    ));
                                    free_to_remove.push(free_loc);
                                    free_insts_to_sink.push(IrInstruction::Free {
                                        ptr: info.alloc_dest,
                                    });
                                }
                            }
                        }
                    }
                }

                let has_alloc_hoists = !alloc_to_hoist.is_empty();

                if to_hoist.is_empty() && !has_alloc_hoists {
                    continue;
                }

                // Ensure we have a preheader
                let preheader =
                    Self::ensure_preheader(&mut function.cfg, loop_data.header, &loop_data.blocks);

                // Collect ALL indices to remove (regular hoists + alloc hoists + free removals)
                let mut indices_to_remove: HashMap<IrBlockId, Vec<usize>> = HashMap::new();

                // Sort hoisted instructions by original position to maintain order
                to_hoist.sort_by_key(|(block_id, idx, _)| (block_id.as_u32(), *idx));

                for (block_id, idx, _) in &to_hoist {
                    indices_to_remove.entry(*block_id).or_default().push(*idx);
                }
                for (block_id, idx, _) in &alloc_to_hoist {
                    indices_to_remove.entry(*block_id).or_default().push(*idx);
                }
                for (block_id, idx) in &free_to_remove {
                    indices_to_remove.entry(*block_id).or_default().push(*idx);
                }

                // Remove instructions from their original blocks (in reverse order to preserve indices)
                for (block_id, indices) in indices_to_remove {
                    if let Some(block) = function.cfg.get_block_mut(block_id) {
                        // Remove in reverse order to preserve indices
                        let mut indices_sorted = indices;
                        indices_sorted.sort_unstable();
                        indices_sorted.dedup();
                        indices_sorted.reverse();
                        for idx in indices_sorted {
                            if idx < block.instructions.len() {
                                block.instructions.remove(idx);
                            }
                        }
                    }
                }

                // Add regular hoisted instructions to preheader
                if let Some(preheader_block) = function.cfg.get_block_mut(preheader) {
                    for (_, _, inst) in to_hoist {
                        preheader_block.instructions.push(inst);
                        result.modified = true;
                        *result
                            .stats
                            .entry("instructions_hoisted".to_string())
                            .or_insert(0) += 1;
                    }

                    // Add hoisted Alloc instructions to preheader
                    for (_, _, inst) in alloc_to_hoist {
                        preheader_block.instructions.push(inst);
                        result.modified = true;
                        *result
                            .stats
                            .entry("allocs_hoisted".to_string())
                            .or_insert(0) += 1;
                    }
                }

                // Sink Free instructions to after the loop exits
                if !free_insts_to_sink.is_empty() {
                    // Find exit targets: successors of exit blocks that are outside the loop
                    let mut exit_targets: HashSet<IrBlockId> = HashSet::new();
                    for &exit_block in &loop_data.exit_blocks {
                        if let Some(block) = function.cfg.get_block(exit_block) {
                            for succ in block.successors() {
                                if !loop_data.blocks.contains(&succ) {
                                    exit_targets.insert(succ);
                                }
                            }
                        }
                    }

                    // Insert Free at the beginning of each exit target
                    for target in exit_targets {
                        if let Some(target_block) = function.cfg.get_block_mut(target) {
                            // Insert Frees at position 0 (before existing instructions)
                            for (i, free_inst) in free_insts_to_sink.iter().enumerate() {
                                target_block.instructions.insert(i, free_inst.clone());
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

/// Common Subexpression Elimination (CSE) pass
///
/// Eliminates redundant computations by reusing previously computed values.
/// Uses value numbering to identify equivalent expressions.
pub struct CSEPass;

impl CSEPass {
    pub fn new() -> Self {
        Self
    }

    /// Generate a hash key for an instruction's computation.
    fn instruction_key(inst: &IrInstruction) -> Option<String> {
        match inst {
            IrInstruction::BinOp {
                op, left, right, ..
            } => {
                // For commutative ops, normalize operand order
                let (l, r) = if Self::is_commutative(*op) && left.as_u32() > right.as_u32() {
                    (right, left)
                } else {
                    (left, right)
                };
                Some(format!("binop:{:?}:{}:{}", op, l.as_u32(), r.as_u32()))
            }
            IrInstruction::UnOp { op, operand, .. } => {
                Some(format!("unop:{:?}:{}", op, operand.as_u32()))
            }
            IrInstruction::Cmp {
                op, left, right, ..
            } => Some(format!("cmp:{:?}:{}:{}", op, left.as_u32(), right.as_u32())),
            IrInstruction::Cast { src, to_ty, .. } => {
                Some(format!("cast:{}:{:?}", src.as_u32(), to_ty))
            }
            IrInstruction::LoadGlobal { global_id, .. } => {
                Some(format!("loadglobal:{}", global_id.0))
            }
            // Loads are not CSE-safe without alias analysis
            // Calls have side effects
            _ => None,
        }
    }

    /// Check if a binary operation is commutative.
    fn is_commutative(op: BinaryOp) -> bool {
        matches!(
            op,
            BinaryOp::Add
                | BinaryOp::Mul
                | BinaryOp::FAdd
                | BinaryOp::FMul
                | BinaryOp::And
                | BinaryOp::Or
                | BinaryOp::Xor
        )
    }
}

impl OptimizationPass for CSEPass {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            // Local CSE within each block
            for block in function.cfg.blocks.values_mut() {
                let mut available: HashMap<String, IrId> = HashMap::new();
                // Use BTreeMap for deterministic iteration order
                let mut replacements: BTreeMap<IrId, IrId> = BTreeMap::new();

                for inst in &block.instructions {
                    if let Some(key) = Self::instruction_key(inst) {
                        if let Some(&existing) = available.get(&key) {
                            // Found common subexpression
                            if let Some(dest) = inst.dest() {
                                replacements.insert(dest, existing);
                                result.modified = true;
                                *result
                                    .stats
                                    .entry("cse_eliminated".to_string())
                                    .or_insert(0) += 1;
                            }
                        } else if let Some(dest) = inst.dest() {
                            available.insert(key, dest);
                        }
                    }
                }

                // Apply replacements
                if !replacements.is_empty() {
                    for inst in &mut block.instructions {
                        inst.replace_uses(&replacements);
                    }
                    replace_terminator_uses(&mut block.terminator, &replacements);
                }
            }
        }

        result
    }
}

/// Global Load Caching pass
///
/// Within each function, if the same global is loaded multiple times and never
/// stored to, replace all loads after the first with the cached value.
/// This eliminates expensive runtime HashMap lookups (rayzor_global_load) in
/// hot loops that repeatedly access static class fields.
pub struct GlobalLoadCachingPass;

impl GlobalLoadCachingPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for GlobalLoadCachingPass {
    fn name(&self) -> &'static str {
        "global_load_cache"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        use super::loop_analysis::DominatorTree;

        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            // Find all globals that are stored to in this function
            let mut stored_globals: HashSet<IrGlobalId> = HashSet::new();
            for block in function.cfg.blocks.values() {
                for inst in &block.instructions {
                    if let IrInstruction::StoreGlobal { global_id, .. } = inst {
                        stored_globals.insert(*global_id);
                    }
                }
            }

            // Collect all LoadGlobal instructions with their locations
            let domtree = DominatorTree::compute(function);

            // Map: global_id -> Vec<(block_id, dest_id)> for read-only globals
            let mut global_loads: HashMap<IrGlobalId, Vec<(IrBlockId, IrId)>> = HashMap::new();
            for (&block_id, block) in &function.cfg.blocks {
                for inst in &block.instructions {
                    if let IrInstruction::LoadGlobal {
                        dest, global_id, ..
                    } = inst
                    {
                        if !stored_globals.contains(global_id) {
                            global_loads
                                .entry(*global_id)
                                .or_default()
                                .push((block_id, *dest));
                        }
                    }
                }
            }

            // Build a map of where each IrId is used (block_id -> set of used IrIds)
            let mut uses_in_block: HashMap<IrBlockId, HashSet<IrId>> = HashMap::new();
            for (&block_id, block) in &function.cfg.blocks {
                let mut used = HashSet::new();
                for inst in &block.instructions {
                    inst.collect_uses(&mut used);
                }
                collect_terminator_uses(&block.terminator, &mut used);
                for phi in &block.phi_nodes {
                    for (_, val) in &phi.incoming {
                        used.insert(*val);
                    }
                }
                uses_in_block.insert(block_id, used);
            }

            // Use BTreeMap for deterministic iteration order
            let mut all_replacements: BTreeMap<IrId, IrId> = BTreeMap::new();

            for (_global_id, loads) in &global_loads {
                if loads.len() <= 1 {
                    continue;
                }
                // For each pair, if the first load's block dominates the second's
                // AND dominates all blocks where the second's value is used,
                // then it's safe to replace.
                for i in 0..loads.len() {
                    let (block_a, dest_a) = loads[i];
                    if all_replacements.contains_key(&dest_a) {
                        continue; // Already replaced
                    }
                    for j in (i + 1)..loads.len() {
                        let (block_b, dest_b) = loads[j];
                        if all_replacements.contains_key(&dest_b) {
                            continue;
                        }
                        // Check if block_a dominates block_b
                        if domtree.dominates(block_a, block_b) {
                            // Verify block_a dominates ALL blocks where dest_b is used
                            let safe = uses_in_block.iter().all(|(use_block, used_ids)| {
                                !used_ids.contains(&dest_b)
                                    || domtree.dominates(block_a, *use_block)
                            });
                            if safe {
                                all_replacements.insert(dest_b, dest_a);
                            }
                        } else if domtree.dominates(block_b, block_a) {
                            // Verify block_b dominates ALL blocks where dest_a is used
                            let safe = uses_in_block.iter().all(|(use_block, used_ids)| {
                                !used_ids.contains(&dest_a)
                                    || domtree.dominates(block_b, *use_block)
                            });
                            if safe {
                                all_replacements.insert(dest_a, dest_b);
                                break; // dest_a is now replaced, stop inner loop
                            }
                        }
                    }
                }
            }

            if all_replacements.is_empty() {
                continue;
            }

            // Compute transitive closure of replacement map.
            // If we have L1→L2 and L2→L3, we need L1→L3 (since L2 will be deleted).
            let mut changed = true;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 100;
            while changed && iterations < MAX_ITERATIONS {
                changed = false;
                iterations += 1;
                let keys: Vec<IrId> = all_replacements.keys().copied().collect();
                for key in keys {
                    if let Some(&value) = all_replacements.get(&key) {
                        if let Some(&transitive_value) = all_replacements.get(&value) {
                            if transitive_value != value {
                                all_replacements.insert(key, transitive_value);
                                changed = true;
                            }
                        }
                    }
                }
            }

            // Apply replacements
            let dead_dests: HashSet<IrId> = all_replacements.keys().copied().collect();

            for block in function.cfg.blocks.values_mut() {
                for inst in &mut block.instructions {
                    inst.replace_uses(&all_replacements);
                }
                replace_terminator_uses(&mut block.terminator, &all_replacements);
                for phi in &mut block.phi_nodes {
                    for (_, val) in &mut phi.incoming {
                        if let Some(&replacement) = all_replacements.get(val) {
                            *val = replacement;
                        }
                    }
                }
                block
                    .instructions
                    .retain(|inst| inst.dest().map_or(true, |d| !dead_dests.contains(&d)));
            }

            result.modified = true;
            result.instructions_eliminated += all_replacements.len();
        }

        result
    }
}

/// Global Value Numbering (GVN) pass
///
/// More powerful than local CSE, uses dominator tree to find redundancies across blocks.
pub struct GVNPass;

impl GVNPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for GVNPass {
    fn name(&self) -> &'static str {
        "gvn"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        use super::loop_analysis::DominatorTree;

        let mut result = OptimizationResult::unchanged();

        for function in module.functions.values_mut() {
            let domtree = DominatorTree::compute(function);

            // Value number table: expression -> canonical register
            let mut value_numbers: HashMap<String, IrId> = HashMap::new();
            // Registers to replace (use BTreeMap for deterministic iteration order)
            let mut replacements: BTreeMap<IrId, IrId> = BTreeMap::new();

            // Process blocks in dominator tree order (preorder DFS)
            let mut worklist = vec![function.entry_block()];
            let mut visited = HashSet::new();

            while let Some(block_id) = worklist.pop() {
                if !visited.insert(block_id) {
                    continue;
                }

                // Process this block
                if let Some(block) = function.cfg.get_block(block_id) {
                    let mut local_values = value_numbers.clone();

                    for inst in &block.instructions {
                        // First, apply known replacements to this instruction's uses
                        let key = Self::make_key_with_replacements(inst, &replacements);

                        if let Some(key) = key {
                            if let Some(&existing) = local_values.get(&key) {
                                if let Some(dest) = inst.dest() {
                                    replacements.insert(dest, existing);
                                    result.modified = true;
                                    *result
                                        .stats
                                        .entry("gvn_eliminated".to_string())
                                        .or_insert(0) += 1;
                                }
                            } else if let Some(dest) = inst.dest() {
                                local_values.insert(key, dest);
                            }
                        }
                    }

                    // Update global value numbers for dominated blocks
                    value_numbers = local_values;
                }

                // Add children in dominator tree
                for &child in domtree.children(block_id) {
                    worklist.push(child);
                }
            }

            // Apply all replacements
            if !replacements.is_empty() {
                for block in function.cfg.blocks.values_mut() {
                    for inst in &mut block.instructions {
                        inst.replace_uses(&replacements);
                    }
                    replace_terminator_uses(&mut block.terminator, &replacements);
                }
            }
        }

        result
    }
}

impl GVNPass {
    /// Create expression key with replacements applied to operands.
    fn make_key_with_replacements(
        inst: &IrInstruction,
        replacements: &BTreeMap<IrId, IrId>,
    ) -> Option<String> {
        let resolve = |id: IrId| -> IrId { *replacements.get(&id).unwrap_or(&id) };

        match inst {
            IrInstruction::BinOp {
                op, left, right, ..
            } => {
                let l = resolve(*left);
                let r = resolve(*right);
                let (l, r) = if CSEPass::is_commutative(*op) && l.as_u32() > r.as_u32() {
                    (r, l)
                } else {
                    (l, r)
                };
                Some(format!("binop:{:?}:{}:{}", op, l.as_u32(), r.as_u32()))
            }
            IrInstruction::UnOp { op, operand, .. } => {
                Some(format!("unop:{:?}:{}", op, resolve(*operand).as_u32()))
            }
            IrInstruction::Cmp {
                op, left, right, ..
            } => Some(format!(
                "cmp:{:?}:{}:{}",
                op,
                resolve(*left).as_u32(),
                resolve(*right).as_u32()
            )),
            IrInstruction::Cast { src, to_ty, .. } => {
                Some(format!("cast:{}:{:?}", resolve(*src).as_u32(), to_ty))
            }
            _ => None,
        }
    }
}

/// Tail Call Optimization pass
///
/// Identifies tail calls and marks them for optimization by the backend.
/// Also converts self-recursive tail calls to loops when possible.
pub struct TailCallOptimizationPass;

impl TailCallOptimizationPass {
    pub fn new() -> Self {
        Self
    }

    /// Check if a call is in tail position.
    fn is_tail_call(block: &IrBasicBlock, call_idx: usize) -> bool {
        // Call must be the last instruction before a return
        if call_idx + 1 != block.instructions.len() {
            return false;
        }

        // Terminator must be a return
        match &block.terminator {
            IrTerminator::Return { value } => {
                // If returning a value, it must be the call's result
                if let Some(ret_val) = value {
                    if let Some(IrInstruction::CallDirect {
                        dest: Some(dest), ..
                    })
                    | Some(IrInstruction::CallIndirect {
                        dest: Some(dest), ..
                    }) = block.instructions.get(call_idx)
                    {
                        return *ret_val == *dest;
                    }
                    false
                } else {
                    // Returning void - call must also return void
                    matches!(
                        block.instructions.get(call_idx),
                        Some(IrInstruction::CallDirect { dest: None, .. })
                            | Some(IrInstruction::CallIndirect { dest: None, .. })
                    )
                }
            }
            _ => false,
        }
    }
}

impl OptimizationPass for TailCallOptimizationPass {
    fn name(&self) -> &'static str {
        "tail-call-optimization"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        for (func_id, function) in module.functions.iter_mut() {
            let current_func_id = *func_id;

            for block in function.cfg.blocks.values_mut() {
                // First pass: identify tail calls
                let mut tail_call_indices: Vec<usize> = Vec::new();
                for (idx, inst) in block.instructions.iter().enumerate() {
                    let is_call = matches!(
                        inst,
                        IrInstruction::CallDirect { .. } | IrInstruction::CallIndirect { .. }
                    );
                    if is_call && Self::is_tail_call(block, idx) {
                        tail_call_indices.push(idx);
                    }
                }

                // Second pass: mark tail calls
                for idx in tail_call_indices {
                    if let Some(inst) = block.instructions.get_mut(idx) {
                        match inst {
                            IrInstruction::CallDirect {
                                func_id,
                                is_tail_call,
                                ..
                            } => {
                                *is_tail_call = true;
                                result.modified = true;

                                // Track self-recursive tail calls separately
                                if *func_id == current_func_id {
                                    *result
                                        .stats
                                        .entry("self_recursive_tail_calls".to_string())
                                        .or_insert(0) += 1;
                                } else {
                                    *result
                                        .stats
                                        .entry("tail_calls_marked".to_string())
                                        .or_insert(0) += 1;
                                }
                            }
                            IrInstruction::CallIndirect { is_tail_call, .. } => {
                                *is_tail_call = true;
                                result.modified = true;
                                *result
                                    .stats
                                    .entry("indirect_tail_calls_marked".to_string())
                                    .or_insert(0) += 1;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        result
    }
}

/// Optimization level for tiered compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization (fastest compilation)
    O0,
    /// Basic optimizations (fast, low overhead)
    O1,
    /// Standard optimizations (good balance)
    O2,
    /// Aggressive optimizations (best runtime, slower compile)
    O3,
}

impl PassManager {
    /// Create optimization pipeline for a specific level.
    pub fn for_level(level: OptimizationLevel) -> Self {
        let mut manager = Self::new();

        // InsertFreePass runs at ALL optimization levels — it's a correctness pass
        // that inserts Free instructions for non-escaping heap allocations.
        // The HIR-level drop analysis only handles direct `new` expressions; this
        // pass catches factory functions like createComplex() that return heap pointers.
        manager.add_pass(super::insert_free::InsertFreePass::new());

        match level {
            OptimizationLevel::O0 => {
                // At O0, inline Haxe `inline`-marked functions (InlineHint::Always) plus
                // very small functions (constructors, trivial helpers). This is needed because:
                // 1. Haxe `inline` is a language guarantee, not an optimization hint
                // 2. After inlining `inline` functions, small constructors must also be inlined
                //    to expose Alloc+GEP patterns for scalar replacement, preventing memory leaks
                let mut forced_inline_model = super::inlining::InliningCostModel::default();
                forced_inline_model.max_inline_size = 15; // Small constructors + Always-hint
                manager.add_pass(super::inlining::InliningPass::with_cost_model(
                    forced_inline_model,
                ));
                manager.add_pass(DeadCodeEliminationPass::new());
                manager.add_pass(super::scalar_replacement::ScalarReplacementPass::new());
                manager.add_pass(CopyPropagationPass::new());
                manager.add_pass(DeadCodeEliminationPass::new());
            }
            OptimizationLevel::O1 => {
                // Fast, low-overhead optimizations
                // Always inline Haxe `inline`-marked functions + cost-model inlining
                manager.add_pass(super::inlining::InliningPass::new());
                // manager.add_pass(GlobalLoadCachingPass::new()); // BUG: causes invalid IR
                manager.add_pass(DeadCodeEliminationPass::new());
                manager.add_pass(ConstantFoldingPass::new());
                manager.add_pass(CopyPropagationPass::new());
                manager.add_pass(UnreachableBlockEliminationPass::new());
            }
            OptimizationLevel::O2 => {
                // Standard optimizations
                manager.add_pass(super::inlining::InliningPass::new());
                manager.add_pass(DeadCodeEliminationPass::new());
                // SRA enabled - regular SRA doesn't modify phi nodes, phi-aware SRA remains disabled
                manager.add_pass(super::scalar_replacement::ScalarReplacementPass::new());
                manager.add_pass(ConstantFoldingPass::new());
                manager.add_pass(CopyPropagationPass::new());
                // GlobalLoadCachingPass: caches repeated global loads within functions
                // Provides ~1.67x speedup on nbody by eliminating redundant HashMap lookups
                manager.add_pass(GlobalLoadCachingPass::new());
                // BCE: eliminate redundant bounds checks in for-in loops
                manager
                    .add_pass(super::bounds_check_elimination::BoundsCheckEliminationPass::new());
                // CSE and LICM may contribute to non-determinism, keeping them for now
                manager.add_pass(CSEPass::new());
                manager.add_pass(LICMPass::new());
                manager.add_pass(ControlFlowSimplificationPass::new());
                manager.add_pass(UnreachableBlockEliminationPass::new());
                manager.add_pass(DeadCodeEliminationPass::new()); // Cleanup after other passes
            }
            OptimizationLevel::O3 => {
                // Aggressive optimizations
                // Inlining first to expose more optimization opportunities
                manager.add_pass(super::inlining::InliningPass::new());
                manager.add_pass(GlobalLoadCachingPass::new());
                manager.add_pass(DeadCodeEliminationPass::new());
                manager.add_pass(super::scalar_replacement::ScalarReplacementPass::new());
                manager.add_pass(ConstantFoldingPass::new());
                manager.add_pass(CopyPropagationPass::new());
                // BCE: eliminate redundant bounds checks in for-in loops
                manager
                    .add_pass(super::bounds_check_elimination::BoundsCheckEliminationPass::new());
                manager.add_pass(GVNPass::new());
                manager.add_pass(CSEPass::new());
                manager.add_pass(LICMPass::new());
                // Loop vectorization after LICM (LICM prepares loops for vectorization)
                manager.add_pass(super::vectorization::LoopVectorizationPass::new());
                manager.add_pass(TailCallOptimizationPass::new());
                manager.add_pass(ControlFlowSimplificationPass::new());
                manager.add_pass(UnreachableBlockEliminationPass::new());
                manager.add_pass(DeadCodeEliminationPass::new()); // Cleanup
            }
        }

        manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::*;
    use crate::tast::SymbolId;

    #[test]
    fn test_constant_folding() {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());

        let sig = FunctionSignatureBuilder::new().returns(IrType::I32).build();
        builder.start_function(SymbolId::from_raw(1), "test".to_string(), sig);

        // Build: 2 + 3
        let two = builder.build_int(2, IrType::I32).unwrap();
        let three = builder.build_int(3, IrType::I32).unwrap();
        let result = builder.build_add(two, three, false).unwrap();
        builder.build_return(Some(result));

        builder.finish_function();

        // Run constant folding
        let mut pass = ConstantFoldingPass::new();
        let opt_result = pass.run_on_module(&mut builder.module);

        assert!(opt_result.modified);
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut builder = IrBuilder::new("test".to_string(), "test.hx".to_string());

        let sig = FunctionSignatureBuilder::new().returns(IrType::I32).build();
        builder.start_function(SymbolId::from_raw(1), "test".to_string(), sig);

        // Create dead code
        let _dead = builder.build_int(42, IrType::I32).unwrap(); // Not used

        let live = builder.build_int(10, IrType::I32).unwrap();
        builder.build_return(Some(live));

        builder.finish_function();

        // Run DCE
        let mut pass = DeadCodeEliminationPass::new();
        let opt_result = pass.run_on_module(&mut builder.module);

        assert!(opt_result.modified);
        assert!(opt_result.instructions_eliminated > 0);
    }
}
