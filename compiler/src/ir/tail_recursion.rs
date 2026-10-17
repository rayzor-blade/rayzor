//! Self-recursion to loops. A function's call to itself whose result is
//! returned, or returned added to a value computed before the call, becomes a
//! jump back to the function's start with the call's arguments as the new
//! parameters. Integer addition wraps, so it is associative and commutative and
//! the sum can be carried in an accumulator: `fib(n - 1) + fib(n - 2)` keeps
//! one call instead of two. The call that becomes the jump is the last thing
//! its block does, so every call still happens in the order it did.

use super::optimization::{InstructionExt, OptimizationPass, OptimizationResult};
use super::{
    BinaryOp, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrPhiNode,
    IrTerminator, IrType, IrValue,
};
use std::collections::BTreeMap;

pub struct TailRecursionPass;

impl TailRecursionPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TailRecursionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for TailRecursionPass {
    fn name(&self) -> &'static str {
        "tail-recursion"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        for (id, function) in module.functions.iter_mut() {
            if loop_recursion(*id, function) {
                result.modified = true;
                *result
                    .stats
                    .entry("recursions_looped".to_string())
                    .or_insert(0) += 1;
            }
        }
        result
    }
}

/// A return that recurses: the block, and the value added to the call's
/// result (none for a plain tail call).
struct Site {
    block: IrBlockId,
    addend: Option<IrId>,
    args: Vec<IrId>,
}

fn loop_recursion(self_id: IrFunctionId, function: &mut IrFunction) -> bool {
    let params: Vec<IrId> = function
        .signature
        .parameters
        .iter()
        .map(|p| p.reg)
        .collect();
    let returns_int = matches!(function.signature.return_type, IrType::I32 | IrType::I64);
    let mut sites: Vec<Site> = Vec::new();
    for (id, block) in &function.cfg.blocks {
        // Shapes this pass does not rename through.
        if block.instructions.iter().any(|inst| {
            matches!(
                inst,
                IrInstruction::Phi { .. } | IrInstruction::LandingPad { .. }
            )
        }) {
            return false;
        }
        let IrTerminator::Return { value } = block.terminator else {
            continue;
        };
        let n = block.instructions.len();
        let is_self_call = |inst: &IrInstruction, want: Option<IrId>| match inst {
            IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                type_args,
                ..
            } if *func_id == self_id
                && *dest == want
                && type_args.is_empty()
                && args.len() == params.len() =>
            {
                Some(args.clone())
            }
            _ => None,
        };
        // `return f(..)`
        if let Some(args) = n
            .checked_sub(1)
            .and_then(|i| is_self_call(&block.instructions[i], value))
        {
            sites.push(Site {
                block: *id,
                addend: None,
                args,
            });
            continue;
        }
        // `return x + f(..)`, the add right after the call.
        if !returns_int || n < 2 {
            continue;
        }
        let Some(ret) = value else { continue };
        let IrInstruction::BinOp {
            dest,
            op: BinaryOp::Add,
            left,
            right,
        } = block.instructions[n - 1]
        else {
            continue;
        };
        if dest != ret {
            continue;
        }
        let IrInstruction::CallDirect {
            dest: Some(result), ..
        } = block.instructions[n - 2]
        else {
            continue;
        };
        let addend = if left == result && right != result {
            right
        } else if right == result && left != result {
            left
        } else {
            continue;
        };
        if let Some(args) = is_self_call(&block.instructions[n - 2], Some(result)) {
            sites.push(Site {
                block: *id,
                addend: Some(addend),
                args,
            });
        }
    }
    if sites.is_empty() {
        return false;
    }
    let accumulates = sites.iter().any(|s| s.addend.is_some());

    // The entry's body moves to a new header; the entry jumps to it.
    let entry = function.cfg.entry_block;
    let header = function.cfg.create_block();
    let mut body = std::mem::replace(
        function.cfg.blocks.get_mut(&entry).unwrap(),
        super::IrBasicBlock::new(entry),
    );
    body.id = header;
    function.cfg.blocks.insert(header, body);
    for block in function.cfg.blocks.values_mut() {
        for phi in &mut block.phi_nodes {
            for (from, _) in &mut phi.incoming {
                if *from == entry {
                    *from = header;
                }
            }
        }
    }

    // Each parameter becomes a phi of the header, and so does the sum so far.
    let mut renamed: BTreeMap<IrId, IrId> = BTreeMap::new();
    let mut header_phis: Vec<IrPhiNode> = Vec::new();
    for (i, param) in params.iter().enumerate() {
        let dest = function.alloc_reg();
        let ty = function.signature.parameters[i].ty.clone();
        function.register_types.insert(dest, ty.clone());
        renamed.insert(*param, dest);
        header_phis.push(IrPhiNode {
            dest,
            incoming: vec![(entry, *param)],
            ty,
        });
    }
    let acc_ty = function.signature.return_type.clone();
    let acc = accumulates.then(|| {
        let zero = function.alloc_reg();
        let acc = function.alloc_reg();
        function.register_types.insert(zero, acc_ty.clone());
        function.register_types.insert(acc, acc_ty.clone());
        let value = match acc_ty {
            IrType::I64 => IrValue::I64(0),
            _ => IrValue::I32(0),
        };
        if let Some(block) = function.cfg.get_block_mut(entry) {
            block
                .instructions
                .push(IrInstruction::Const { dest: zero, value });
        }
        header_phis.push(IrPhiNode {
            dest: acc,
            incoming: vec![(entry, zero)],
            ty: acc_ty.clone(),
        });
        acc
    });
    if let Some(block) = function.cfg.get_block_mut(entry) {
        block.terminator = IrTerminator::Branch { target: header };
    }

    // Uses of a parameter, everywhere but the entry, read its phi.
    for (id, block) in function.cfg.blocks.iter_mut() {
        if *id == entry {
            continue;
        }
        for phi in &mut block.phi_nodes {
            for (_, value) in &mut phi.incoming {
                if let Some(new) = renamed.get(value) {
                    *value = *new;
                }
            }
        }
        for inst in &mut block.instructions {
            inst.replace_uses(&renamed);
        }
        match &mut block.terminator {
            IrTerminator::CondBranch { condition, .. } => {
                if let Some(new) = renamed.get(condition) {
                    *condition = *new;
                }
            }
            IrTerminator::Switch { value, .. } => {
                if let Some(new) = renamed.get(value) {
                    *value = *new;
                }
            }
            IrTerminator::Return { value: Some(v) } => {
                if let Some(new) = renamed.get(v) {
                    *v = *new;
                }
            }
            _ => {}
        }
    }

    // Recursive returns jump back; the others return what was accumulated.
    let site_blocks: Vec<IrBlockId> = sites.iter().map(|s| s.block).collect();
    for site in &sites {
        let block_id = if site.block == entry {
            header
        } else {
            site.block
        };
        let args: Vec<IrId> = site
            .args
            .iter()
            .map(|a| *renamed.get(a).unwrap_or(a))
            .collect();
        let mut next_acc = None;
        if let Some(block) = function.cfg.get_block_mut(block_id) {
            let keep = block.instructions.len() - if site.addend.is_some() { 2 } else { 1 };
            block.instructions.truncate(keep);
            block.terminator = IrTerminator::Branch { target: header };
        }
        if let Some(acc) = acc {
            let next = match site.addend {
                Some(addend) => {
                    let addend = *renamed.get(&addend).unwrap_or(&addend);
                    let dest = function.alloc_reg();
                    function.register_types.insert(dest, acc_ty.clone());
                    if let Some(block) = function.cfg.get_block_mut(block_id) {
                        block.instructions.push(IrInstruction::BinOp {
                            dest,
                            op: BinaryOp::Add,
                            left: acc,
                            right: addend,
                        });
                    }
                    dest
                }
                None => acc,
            };
            next_acc = Some(next);
        }
        for (phi, arg) in header_phis.iter_mut().zip(args.iter()) {
            phi.incoming.push((block_id, *arg));
        }
        if let (Some(next), Some(phi)) = (next_acc, header_phis.last_mut()) {
            phi.incoming.push((block_id, next));
        }
    }
    if let Some(acc) = acc {
        let returns: Vec<(IrBlockId, IrId)> = function
            .cfg
            .blocks
            .iter()
            .filter(|(id, _)| **id != entry && !site_blocks.contains(id))
            .filter_map(|(id, b)| match b.terminator {
                IrTerminator::Return { value: Some(v) } => Some((*id, v)),
                _ => None,
            })
            .filter(|(id, _)| !(site_blocks.contains(&entry) && *id == header))
            .collect();
        for (id, value) in returns {
            let dest = function.alloc_reg();
            function.register_types.insert(dest, acc_ty.clone());
            if let Some(block) = function.cfg.get_block_mut(id) {
                block.instructions.push(IrInstruction::BinOp {
                    dest,
                    op: BinaryOp::Add,
                    left: acc,
                    right: value,
                });
                block.terminator = IrTerminator::Return { value: Some(dest) };
            }
        }
    }
    if let Some(block) = function.cfg.get_block_mut(header) {
        block.phi_nodes.extend(header_phis);
    }
    function.cfg.recompute_predecessors();
    true
}
