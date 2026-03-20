//! Loop Unrolling Pass
//!
//! Fully unrolls small loops with constant trip counts, and partially unrolls
//! loops with known bounds to reduce branch overhead and enable further
//! optimizations (constant folding, SRA, vectorization).
//!
//! Targets:
//! - Full unrolling: loops with constant trip count ≤ MAX_FULL_UNROLL
//! - Partial unrolling: loops with larger constant counts, unrolled PARTIAL_FACTOR×

use super::loop_analysis::{DominatorTree, LoopNestInfo, TripCount};
use super::optimization::{OptimizationPass, OptimizationResult};
use super::{
    BinaryOp, CompareOp, IrBasicBlock, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction,
    IrModule, IrPhiNode, IrTerminator, IrType, IrValue,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Maximum trip count for full unrolling
const MAX_FULL_UNROLL: u64 = 16;
/// Factor for partial unrolling
const PARTIAL_FACTOR: u64 = 4;
/// Maximum instruction count in a loop body for unrolling
const MAX_BODY_INSTRUCTIONS: usize = 32;

pub struct LoopUnrollingPass;

impl LoopUnrollingPass {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for LoopUnrollingPass {
    fn name(&self) -> &'static str {
        "loop-unrolling"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();

        let func_ids: Vec<IrFunctionId> = module.functions.keys().copied().collect();
        for func_id in func_ids {
            let function = module.functions.get_mut(&func_id).unwrap();
            let r = unroll_loops_in_function(function);
            result = result.combine(r);
        }

        result
    }
}

/// Analyze a loop to determine its induction variable and trip count.
/// Returns (induction_var_phi, init_value, step, bound, compare_op) if analyzable.
fn analyze_loop_trip_count(
    function: &IrFunction,
    header: IrBlockId,
    back_edge_source: IrBlockId,
    loop_blocks: &HashSet<IrBlockId>,
) -> Option<LoopInductionInfo> {
    let header_block = function.cfg.get_block(header)?;

    // Find the induction variable phi node in the header
    // Pattern: %iv = phi [preheader: init_val], [back_edge: next_val]
    for phi in &header_block.phi_nodes {
        // Check if one incoming is from outside loop (init) and one from inside (step)
        let mut init_val = None;
        let mut step_val = None;

        for &(pred_block, pred_val) in &phi.incoming {
            if loop_blocks.contains(&pred_block) {
                step_val = Some((pred_block, pred_val));
            } else {
                init_val = Some(pred_val);
            }
        }

        let (init_reg, (step_block, step_reg)) = match (init_val, step_val) {
            (Some(i), Some(s)) => (i, s),
            _ => continue,
        };

        // Check if init is a constant
        let init_const = find_const_value(function, init_reg)?;

        // Check if step_reg = iv + constant (the increment)
        let (step_amount, step_op) =
            find_step_instruction(function, step_reg, phi.dest, loop_blocks)?;

        // Find the exit condition: the header (or a block in the loop) should have
        // a CondBranch comparing iv against a bound
        let (bound_const, cmp_op) = find_loop_bound(function, phi.dest, loop_blocks, header)?;

        // Compute trip count from init, bound, step, and comparison
        let trip_count = compute_trip_count(init_const, bound_const, step_amount, cmp_op)?;

        return Some(LoopInductionInfo {
            iv_phi: phi.dest,
            init_val: init_reg,
            init_const,
            step_val: step_reg,
            step_amount,
            bound_const,
            trip_count,
        });
    }

    None
}

struct LoopInductionInfo {
    iv_phi: IrId,
    init_val: IrId,
    init_const: i64,
    step_val: IrId,
    step_amount: i64,
    bound_const: i64,
    trip_count: u64,
}

fn find_const_value(function: &IrFunction, reg: IrId) -> Option<i64> {
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::Const { dest, value } = inst {
                if *dest == reg {
                    return match value {
                        IrValue::I32(n) => Some(*n as i64),
                        IrValue::I64(n) => Some(*n),
                        _ => None,
                    };
                }
            }
        }
    }
    None
}

/// Find a step instruction: step_reg = iv + constant
fn find_step_instruction(
    function: &IrFunction,
    step_reg: IrId,
    iv: IrId,
    loop_blocks: &HashSet<IrBlockId>,
) -> Option<(i64, BinaryOp)> {
    for &block_id in loop_blocks {
        if let Some(block) = function.cfg.get_block(block_id) {
            for inst in &block.instructions {
                if let IrInstruction::BinOp {
                    dest,
                    op,
                    left,
                    right,
                } = inst
                {
                    if *dest == step_reg {
                        match op {
                            BinaryOp::Add => {
                                if *left == iv {
                                    if let Some(c) = find_const_value(function, *right) {
                                        return Some((c, *op));
                                    }
                                }
                                if *right == iv {
                                    if let Some(c) = find_const_value(function, *left) {
                                        return Some((c, *op));
                                    }
                                }
                            }
                            BinaryOp::Sub => {
                                if *left == iv {
                                    if let Some(c) = find_const_value(function, *right) {
                                        return Some((-c, BinaryOp::Add));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    None
}

/// Find the loop exit condition comparing the induction variable to a constant bound.
fn find_loop_bound(
    function: &IrFunction,
    iv: IrId,
    loop_blocks: &HashSet<IrBlockId>,
    header: IrBlockId,
) -> Option<(i64, CompareOp)> {
    // Check the header's terminator first, then other blocks
    let blocks_to_check: Vec<IrBlockId> = std::iter::once(header)
        .chain(loop_blocks.iter().copied().filter(|&b| b != header))
        .collect();

    for block_id in blocks_to_check {
        let block = function.cfg.get_block(block_id)?;
        if let IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = &block.terminator
        {
            // One target should be inside the loop, one outside (exit)
            let true_in = loop_blocks.contains(true_target);
            let false_in = loop_blocks.contains(false_target);

            if true_in == false_in {
                continue; // Both inside or both outside — not an exit
            }

            // Find the Cmp instruction that defines the condition
            for inst in &block.instructions {
                if let IrInstruction::Cmp {
                    dest,
                    op,
                    left,
                    right,
                } = inst
                {
                    if *dest == *condition {
                        // Check if one side is the IV and the other is a constant
                        if *left == iv {
                            if let Some(bound) = find_const_value(function, *right) {
                                // If true_target is INSIDE loop, the condition keeps the loop going
                                // so the exit condition is the NEGATION
                                let effective_op = if true_in { *op } else { negate_cmp(*op) };
                                return Some((bound, effective_op));
                            }
                        }
                        if *right == iv {
                            if let Some(bound) = find_const_value(function, *left) {
                                let swapped = swap_cmp(*op);
                                let effective_op = if true_in {
                                    swapped
                                } else {
                                    negate_cmp(swapped)
                                };
                                return Some((bound, effective_op));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn negate_cmp(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Ge,
        CompareOp::Le => CompareOp::Gt,
        CompareOp::Gt => CompareOp::Le,
        CompareOp::Ge => CompareOp::Lt,
        CompareOp::Eq => CompareOp::Ne,
        CompareOp::Ne => CompareOp::Eq,
        other => other,
    }
}

fn swap_cmp(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        other => other,
    }
}

fn compute_trip_count(init: i64, bound: i64, step: i64, cmp_op: CompareOp) -> Option<u64> {
    if step == 0 {
        return None; // Infinite loop
    }

    // For "iv < bound" starting at init, stepping by step:
    let range = match cmp_op {
        CompareOp::Lt => {
            if step > 0 && bound > init {
                (bound - init + step - 1) / step
            } else {
                return None;
            }
        }
        CompareOp::Le => {
            if step > 0 && bound >= init {
                (bound - init + step) / step
            } else {
                return None;
            }
        }
        CompareOp::Gt => {
            if step < 0 && init > bound {
                (init - bound + (-step) - 1) / (-step)
            } else {
                return None;
            }
        }
        CompareOp::Ge => {
            if step < 0 && init >= bound {
                (init - bound + (-step)) / (-step)
            } else {
                return None;
            }
        }
        CompareOp::Ne => {
            if step > 0 && bound > init && (bound - init) % step == 0 {
                (bound - init) / step
            } else if step < 0 && init > bound && (init - bound) % (-step) == 0 {
                (init - bound) / (-step)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    if range >= 0 {
        Some(range as u64)
    } else {
        None
    }
}

fn unroll_loops_in_function(function: &mut IrFunction) -> OptimizationResult {
    let domtree = DominatorTree::compute(function);
    let loop_info = LoopNestInfo::analyze(function, &domtree);

    if loop_info.loops.is_empty() {
        return OptimizationResult::unchanged();
    }

    let mut total_unrolled = 0;

    // Process loops from innermost to outermost
    let mut loops_by_depth: Vec<(IrBlockId, usize)> = loop_info
        .loops
        .iter()
        .map(|(&header, l)| (header, l.nesting_depth))
        .collect();
    loops_by_depth.sort_by(|a, b| b.1.cmp(&a.1)); // Deepest first

    for (header, _depth) in loops_by_depth {
        let natural_loop = match loop_info.loops.get(&header) {
            Some(l) => l,
            None => continue,
        };

        // Count instructions in loop body
        let body_inst_count: usize = natural_loop
            .blocks
            .iter()
            .filter_map(|&b| function.cfg.get_block(b))
            .map(|b| b.instructions.len())
            .sum();

        if body_inst_count > MAX_BODY_INSTRUCTIONS {
            continue;
        }

        // Analyze trip count
        let info = match analyze_loop_trip_count(
            function,
            header,
            natural_loop.back_edge_source,
            &natural_loop.blocks,
        ) {
            Some(info) => info,
            None => continue,
        };

        if info.trip_count == 0 {
            continue;
        }

        // Full unrolling for small trip counts
        if info.trip_count <= MAX_FULL_UNROLL {
            if fully_unroll_loop(function, &natural_loop.blocks, header, &info) {
                total_unrolled += 1;
            }
        }
    }

    if total_unrolled > 0 {
        let mut result = OptimizationResult::changed();
        result
            .stats
            .insert("loops_unrolled".to_string(), total_unrolled);
        result
    } else {
        OptimizationResult::unchanged()
    }
}

/// Fully unroll a loop by replacing it with N copies of the body,
/// substituting the induction variable with constants.
fn fully_unroll_loop(
    function: &mut IrFunction,
    loop_blocks: &HashSet<IrBlockId>,
    header: IrBlockId,
    info: &LoopInductionInfo,
) -> bool {
    // For a simple single-body-block loop, we can unroll by:
    // 1. Collecting body instructions (excluding the IV increment and branch)
    // 2. Cloning them N times with the IV replaced by constants
    // 3. Replacing the loop with the unrolled sequence

    // Only handle simple loops: header + optional single body block
    // (header contains phi + cmp + branch, body contains useful work + iv update)
    let body_blocks: Vec<IrBlockId> = loop_blocks
        .iter()
        .copied()
        .filter(|&b| b != header)
        .collect();

    if body_blocks.len() > 1 {
        return false; // Multi-block loop body too complex for now
    }

    let header_block = match function.cfg.get_block(header) {
        Some(b) => b,
        None => return false,
    };

    // Find the exit target (block outside the loop from the header's condition)
    let exit_target = match &header_block.terminator {
        IrTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            if !loop_blocks.contains(true_target) {
                *true_target
            } else if !loop_blocks.contains(false_target) {
                *false_target
            } else {
                return false;
            }
        }
        _ => return false,
    };

    // Find preheader (predecessor of header that's not in the loop)
    let preheader = header_block
        .predecessors
        .iter()
        .find(|&&pred| !loop_blocks.contains(&pred))
        .copied();

    let preheader = match preheader {
        Some(p) => p,
        None => return false,
    };

    // Collect body instructions to replicate
    // These are all instructions in body blocks (not the header)
    let mut body_instructions: Vec<IrInstruction> = Vec::new();
    for &body_id in &body_blocks {
        if let Some(body_block) = function.cfg.get_block(body_id) {
            for inst in &body_block.instructions {
                // Skip the IV increment instruction and loop control
                if let IrInstruction::BinOp { dest, .. } = inst {
                    if *dest == info.step_val {
                        continue; // Skip IV update
                    }
                }
                // Skip the branch condition check in body
                if let IrInstruction::Cmp { .. } = inst {
                    continue;
                }
                body_instructions.push(inst.clone());
            }
        }
    }

    // Also collect body instructions from the header (non-control flow)
    // Header instructions that aren't loop control (cmp/iv-increment) are unused
    // in the simple unrolling case — the body block has the useful work.
    let _ = header_block;

    // Generate unrolled instructions
    let mut unrolled_instructions: Vec<IrInstruction> = Vec::new();
    let mut reg_id = function.next_reg_id;

    for iteration in 0..info.trip_count {
        let iv_value = info.init_const + (iteration as i64) * info.step_amount;

        // Create a mapping from original registers to new ones for this iteration
        let mut reg_map: HashMap<IrId, IrId> = HashMap::new();

        // Map IV phi to a constant
        let iv_const_reg = IrId::new(reg_id);
        reg_id += 1;
        unrolled_instructions.push(IrInstruction::Const {
            dest: iv_const_reg,
            value: IrValue::I32(iv_value as i32),
        });
        function.register_types.insert(iv_const_reg, IrType::I32);
        reg_map.insert(info.iv_phi, iv_const_reg);

        // Clone body instructions with register remapping
        for inst in &body_instructions {
            let new_inst = remap_instruction(
                inst,
                &mut reg_map,
                &mut reg_id,
                &mut function.register_types,
            );
            unrolled_instructions.push(new_inst);
        }
    }

    // Now replace the loop:
    // 1. Redirect preheader to a new unrolled block
    // 2. The unrolled block branches to exit_target
    let unrolled_block_id = function.cfg.create_block();
    if let Some(unrolled_block) = function.cfg.get_block_mut(unrolled_block_id) {
        unrolled_block.instructions = unrolled_instructions;
        unrolled_block.terminator = IrTerminator::Branch {
            target: exit_target,
        };
        unrolled_block.predecessors.push(preheader);
    }

    // Update preheader to jump to unrolled block instead of header
    if let Some(pre_block) = function.cfg.get_block_mut(preheader) {
        replace_block_target(&mut pre_block.terminator, header, unrolled_block_id);
    }

    // Update exit target's predecessors: replace header with unrolled block
    if let Some(exit_block) = function.cfg.get_block_mut(exit_target) {
        exit_block
            .predecessors
            .retain(|&p| !loop_blocks.contains(&p));
        if !exit_block.predecessors.contains(&unrolled_block_id) {
            exit_block.predecessors.push(unrolled_block_id);
        }

        // Update phi nodes in exit block: replace references from loop blocks
        for phi in &mut exit_block.phi_nodes {
            let mut new_incoming = Vec::new();
            for &(pred, val) in &phi.incoming {
                if loop_blocks.contains(&pred) {
                    // Map the value through the last iteration's register map
                    new_incoming.push((unrolled_block_id, val));
                } else {
                    new_incoming.push((pred, val));
                }
            }
            phi.incoming = new_incoming;
        }
    }

    // Remove loop blocks from CFG (they're now dead)
    for &block_id in loop_blocks {
        function.cfg.blocks.remove(&block_id);
    }

    function.next_reg_id = reg_id;

    true
}

/// Clone an instruction with register remapping for unrolling.
fn remap_instruction(
    inst: &IrInstruction,
    reg_map: &mut HashMap<IrId, IrId>,
    next_reg: &mut u32,
    register_types: &mut HashMap<IrId, IrType>,
) -> IrInstruction {
    // Helper to remap a register, creating a new one if it's a definition
    let map_use =
        |r: IrId, reg_map: &HashMap<IrId, IrId>| -> IrId { reg_map.get(&r).copied().unwrap_or(r) };

    let alloc_new = |old: IrId,
                     next_reg: &mut u32,
                     reg_map: &mut HashMap<IrId, IrId>,
                     register_types: &mut HashMap<IrId, IrType>|
     -> IrId {
        let new = IrId::new(*next_reg);
        *next_reg += 1;
        reg_map.insert(old, new);
        if let Some(ty) = register_types.get(&old).cloned() {
            register_types.insert(new, ty);
        }
        new
    };

    match inst {
        IrInstruction::Const { dest, value } => {
            let new_dest = alloc_new(*dest, next_reg, reg_map, register_types);
            IrInstruction::Const {
                dest: new_dest,
                value: value.clone(),
            }
        }
        IrInstruction::BinOp {
            dest,
            op,
            left,
            right,
        } => IrInstruction::BinOp {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            op: *op,
            left: map_use(*left, reg_map),
            right: map_use(*right, reg_map),
        },
        IrInstruction::UnOp { dest, op, operand } => IrInstruction::UnOp {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            op: *op,
            operand: map_use(*operand, reg_map),
        },
        IrInstruction::Load { dest, ptr, ty } => IrInstruction::Load {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            ptr: map_use(*ptr, reg_map),
            ty: ty.clone(),
        },
        IrInstruction::Store {
            ptr,
            value,
            store_ty,
            ..
        } => IrInstruction::Store {
            ptr: map_use(*ptr, reg_map),
            value: map_use(*value, reg_map),
            store_ty: store_ty.clone(),
        },
        IrInstruction::CallDirect {
            dest,
            func_id,
            args,
            arg_ownership,
            type_args,
            is_tail_call,
        } => IrInstruction::CallDirect {
            dest: dest.map(|d| alloc_new(d, next_reg, reg_map, register_types)),
            func_id: *func_id,
            args: args.iter().map(|a| map_use(*a, reg_map)).collect(),
            arg_ownership: arg_ownership.clone(),
            type_args: type_args.clone(),
            is_tail_call: *is_tail_call,
        },
        IrInstruction::CallIndirect {
            dest,
            func_ptr,
            args,
            signature,
            arg_ownership,
            is_tail_call,
        } => IrInstruction::CallIndirect {
            dest: dest.map(|d| alloc_new(d, next_reg, reg_map, register_types)),
            func_ptr: map_use(*func_ptr, reg_map),
            args: args.iter().map(|a| map_use(*a, reg_map)).collect(),
            signature: signature.clone(),
            arg_ownership: arg_ownership.clone(),
            is_tail_call: *is_tail_call,
        },
        IrInstruction::GetElementPtr {
            dest,
            ptr,
            indices,
            ty,
            struct_context,
        } => IrInstruction::GetElementPtr {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            ptr: map_use(*ptr, reg_map),
            indices: indices.iter().map(|i| map_use(*i, reg_map)).collect(),
            ty: ty.clone(),
            struct_context: struct_context.clone(),
        },
        IrInstruction::Cast {
            dest,
            src,
            from_ty,
            to_ty,
        } => IrInstruction::Cast {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            src: map_use(*src, reg_map),
            from_ty: from_ty.clone(),
            to_ty: to_ty.clone(),
        },
        IrInstruction::BitCast { dest, src, ty } => IrInstruction::BitCast {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            src: map_use(*src, reg_map),
            ty: ty.clone(),
        },
        IrInstruction::PtrAdd {
            dest,
            ptr,
            offset,
            ty,
        } => IrInstruction::PtrAdd {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            ptr: map_use(*ptr, reg_map),
            offset: map_use(*offset, reg_map),
            ty: ty.clone(),
        },
        IrInstruction::Cmp {
            dest,
            op,
            left,
            right,
        } => IrInstruction::Cmp {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            op: *op,
            left: map_use(*left, reg_map),
            right: map_use(*right, reg_map),
        },
        IrInstruction::FunctionRef { dest, func_id } => IrInstruction::FunctionRef {
            dest: alloc_new(*dest, next_reg, reg_map, register_types),
            func_id: *func_id,
        },
        // For other instruction types, clone as-is (conservative)
        other => other.clone(),
    }
}

/// Replace a block target in a terminator.
fn replace_block_target(terminator: &mut IrTerminator, old: IrBlockId, new: IrBlockId) {
    match terminator {
        IrTerminator::Branch { target } => {
            if *target == old {
                *target = new;
            }
        }
        IrTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            if *true_target == old {
                *true_target = new;
            }
            if *false_target == old {
                *false_target = new;
            }
        }
        IrTerminator::Switch { cases, default, .. } => {
            if *default == old {
                *default = new;
            }
            for case in cases {
                if case.1 == old {
                    case.1 = new;
                }
            }
        }
        _ => {}
    }
}
