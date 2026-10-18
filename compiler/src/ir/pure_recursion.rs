//! Pure self-recursion: a call of a pure function with the same arguments has
//! the same result, and expanding one self-call exposes the calls its body
//! repeats. For naive fib the expanded body computes `fib(n - 2)` in one arm
//! and again after the arms join; computing it in the other arm too and
//! joining the two leaves T(n) = T(n-2) + T(n-3) calls instead of
//! T(n-1) + T(n-2). No path makes more calls than before.
//!
//! Runs before tail recursion, which would otherwise fold the second call
//! into a loop and hide the repetition.

use super::inlining::{CallSite, InliningPass};
use super::loop_analysis::DominatorTree;
use super::optimization::{OptimizationPass, OptimizationResult};
use super::{
    BinaryOp, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrPhiNode,
    IrTerminator, IrType, IrValue,
};
use std::collections::{BTreeMap, BTreeSet};

/// Larger bodies are not worth expanding.
const MAX_BODY: usize = 64;

/// Bounds the reuse rounds on one function.
const MAX_ROUNDS: usize = 8;

pub struct PureRecursionPass;

impl PureRecursionPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PureRecursionPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for PureRecursionPass {
    fn name(&self) -> &'static str {
        "pure-recursion"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        let pure = pure_functions(module);
        let candidates: Vec<IrFunctionId> = pure
            .iter()
            .copied()
            .filter(|id| {
                let f = &module.functions[id];
                self_calls(f).len() >= 2 && body_size(f) <= MAX_BODY && is_acyclic(f)
            })
            .collect();
        for id in candidates {
            let reused = expand_and_reuse(module, id);
            if reused > 0 {
                result.modified = true;
                *result
                    .stats
                    .entry("pure_self_calls_reused".to_string())
                    .or_insert(0) += reused;
            }
        }
        result
    }
}

/// Functions that only compute from their arguments: no memory, no globals,
/// no allocation, nothing that can throw, and calls only to one another.
pub(super) fn pure_functions(module: &IrModule) -> BTreeSet<IrFunctionId> {
    let frame_update = |id: &IrFunctionId| {
        module
            .extern_functions
            .get(id)
            .is_some_and(|f| f.name == "rayzor_update_call_frame_location")
    };
    let mut pure: BTreeSet<IrFunctionId> = module
        .functions
        .iter()
        .filter(|(_, f)| !f.cfg.blocks.is_empty() && !f.is_forward_stub())
        .filter(|(_, f)| {
            f.cfg.blocks.values().all(|block| {
                matches!(
                    block.terminator,
                    IrTerminator::Branch { .. }
                        | IrTerminator::CondBranch { .. }
                        | IrTerminator::Switch { .. }
                        | IrTerminator::Return { .. }
                        | IrTerminator::Unreachable
                ) && block.instructions.iter().all(|inst| match inst {
                    IrInstruction::BinOp { op, .. } => !matches!(op, BinaryOp::Div | BinaryOp::Rem),
                    IrInstruction::CallDirect { func_id, .. } => {
                        module.functions.contains_key(func_id) || frame_update(func_id)
                    }
                    IrInstruction::Const { .. }
                    | IrInstruction::Copy { .. }
                    | IrInstruction::UnOp { .. }
                    | IrInstruction::Cmp { .. }
                    | IrInstruction::Select { .. }
                    | IrInstruction::Cast { .. }
                    | IrInstruction::BitCast { .. }
                    | IrInstruction::Undef { .. }
                    | IrInstruction::DebugLoc { .. } => true,
                    _ => false,
                })
            })
        })
        .map(|(id, _)| *id)
        .collect();
    loop {
        let impure: Vec<IrFunctionId> = pure
            .iter()
            .copied()
            .filter(|id| {
                module.functions[id].cfg.blocks.values().any(|block| {
                    block.instructions.iter().any(|inst| {
                        matches!(inst, IrInstruction::CallDirect { func_id, .. }
                            if !frame_update(func_id) && !pure.contains(func_id))
                    })
                })
            })
            .collect();
        if impure.is_empty() {
            return pure;
        }
        for id in impure {
            pure.remove(&id);
        }
    }
}

fn body_size(function: &IrFunction) -> usize {
    function
        .cfg
        .blocks
        .values()
        .map(|b| b.instructions.len())
        .sum()
}

fn successors(terminator: &IrTerminator) -> Vec<IrBlockId> {
    match terminator {
        IrTerminator::Branch { target } => vec![*target],
        IrTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        IrTerminator::Switch { cases, default, .. } => {
            let mut out: Vec<IrBlockId> = cases.iter().map(|(_, b)| *b).collect();
            out.push(*default);
            out
        }
        _ => Vec::new(),
    }
}

fn is_acyclic(function: &IrFunction) -> bool {
    // Depth-first search; a block reached again while still on the stack is
    // a back edge.
    fn visit(
        function: &IrFunction,
        block: IrBlockId,
        on_stack: &mut BTreeSet<IrBlockId>,
        done: &mut BTreeSet<IrBlockId>,
    ) -> bool {
        if done.contains(&block) {
            return true;
        }
        if !on_stack.insert(block) {
            return false;
        }
        if let Some(b) = function.cfg.blocks.get(&block) {
            for next in successors(&b.terminator) {
                if !visit(function, next, on_stack, done) {
                    return false;
                }
            }
        }
        on_stack.remove(&block);
        done.insert(block);
        true
    }
    visit(
        function,
        function.cfg.entry_block,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
    )
}

/// Self-call sites in block order.
fn self_calls(function: &IrFunction) -> Vec<(IrBlockId, usize)> {
    let mut out = Vec::new();
    for (block_id, block) in &function.cfg.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if matches!(inst, IrInstruction::CallDirect { func_id, dest: Some(_), .. }
                if *func_id == function.id)
            {
                out.push((*block_id, idx));
            }
        }
    }
    out
}

/// Expands the first self-call site whose expansion lets a call be reused,
/// and keeps it only then. Returns how many calls were reused.
fn expand_and_reuse(module: &mut IrModule, id: IrFunctionId) -> usize {
    let original = module.functions[&id].clone();
    for (block, index) in self_calls(&original) {
        let IrInstruction::CallDirect { args, dest, .. } =
            &original.cfg.blocks[&block].instructions[index]
        else {
            continue;
        };
        let site = CallSite {
            caller: id,
            callee: id,
            block,
            instruction_index: index,
            loop_depth: 0,
            args: args.clone(),
            dest: *dest,
        };
        let mut next_reg = next_register(module);
        if InliningPass::inline_call_site(module, &site, &mut next_reg).is_err() {
            module.functions.insert(id, original.clone());
            continue;
        }
        let function = module.functions.get_mut(&id).unwrap();
        function.cfg.recompute_predecessors();
        function.next_reg_id = function.next_reg_id.max(next_reg);
        let mut reused = 0;
        for _ in 0..MAX_ROUNDS {
            if !reuse_one(function) {
                break;
            }
            reused += 1;
        }
        if reused > 0 {
            return reused;
        }
        module.functions.insert(id, original.clone());
    }
    0
}

fn next_register(module: &IrModule) -> u32 {
    let mut next = 0u32;
    for function in module.functions.values() {
        next = next.max(function.next_reg_id);
        for block in function.cfg.blocks.values() {
            for phi in &block.phi_nodes {
                next = next.max(phi.dest.as_u32() + 1);
            }
            for inst in &block.instructions {
                if let Some(dest) = inst.dest() {
                    next = next.max(dest.as_u32() + 1);
                }
            }
        }
    }
    next
}

fn int_bits(value: &IrValue) -> Option<i64> {
    match *value {
        IrValue::I32(v) => Some(v.into()),
        IrValue::I64(v) => Some(v),
        _ => None,
    }
}

/// An integer value as `base + offset`, wrapping in its type.
type Affine = (IrId, i64);

fn affine_forms(function: &IrFunction) -> BTreeMap<IrId, Affine> {
    let mut consts: BTreeMap<IrId, i64> = BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::Const { dest, value } = inst {
                if let Some(v) = int_bits(value) {
                    consts.insert(*dest, v);
                }
            }
        }
    }
    let int_type = |reg: &IrId| {
        function
            .register_types
            .get(reg)
            .filter(|t| matches!(t, IrType::I32 | IrType::I64))
            .cloned()
    };
    let mut forms: BTreeMap<IrId, Affine> = BTreeMap::new();
    let form =
        |forms: &BTreeMap<IrId, Affine>, reg: IrId| forms.get(&reg).copied().unwrap_or((reg, 0));
    loop {
        let mut changed = false;
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                let (dest, derived) = match inst {
                    IrInstruction::Copy { dest, src } => (*dest, Some(form(&forms, *src))),
                    IrInstruction::BinOp {
                        dest,
                        op: op @ (BinaryOp::Add | BinaryOp::Sub),
                        left,
                        right,
                    } => {
                        let Some(ty) = int_type(dest) else { continue };
                        let wrap = |v: i64| {
                            if ty == IrType::I32 {
                                v as i32 as i64
                            } else {
                                v
                            }
                        };
                        let derived = match (consts.get(left), consts.get(right), op) {
                            (_, Some(c), BinaryOp::Add) => {
                                let (b, o) = form(&forms, *left);
                                Some((b, wrap(o.wrapping_add(*c))))
                            }
                            (_, Some(c), _) => {
                                let (b, o) = form(&forms, *left);
                                Some((b, wrap(o.wrapping_sub(*c))))
                            }
                            (Some(c), None, BinaryOp::Add) => {
                                let (b, o) = form(&forms, *right);
                                Some((b, wrap(o.wrapping_add(*c))))
                            }
                            _ => None,
                        };
                        (*dest, derived)
                    }
                    _ => continue,
                };
                if let Some(derived) = derived {
                    if forms.get(&dest) != Some(&derived) {
                        forms.insert(dest, derived);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return forms;
        }
    }
}

struct Call {
    block: IrBlockId,
    index: usize,
    dest: IrId,
    key: Vec<Affine>,
}

/// Reuses one self-call: one a dominating call already made, or one that
/// follows a join where one arm already made it. False when there is none.
fn reuse_one(function: &mut IrFunction) -> bool {
    let forms = affine_forms(function);
    let form = |reg: IrId| forms.get(&reg).copied().unwrap_or((reg, 0));
    let calls: Vec<Call> = self_calls(function)
        .into_iter()
        .filter_map(|(block, index)| {
            let IrInstruction::CallDirect {
                dest: Some(dest),
                args,
                ..
            } = &function.cfg.blocks[&block].instructions[index]
            else {
                return None;
            };
            Some(Call {
                block,
                index,
                dest: *dest,
                key: args.iter().map(|a| form(*a)).collect(),
            })
        })
        .collect();
    let dom = DominatorTree::compute(function);
    let defined_in: BTreeMap<IrId, IrBlockId> = function
        .cfg
        .blocks
        .iter()
        .flat_map(|(b, block)| {
            block
                .instructions
                .iter()
                .filter_map(|i| i.dest())
                .chain(block.phi_nodes.iter().map(|p| p.dest))
                .map(move |d| (d, *b))
        })
        .collect();
    // A base is usable in `block` if it is a parameter or defined in a block
    // that dominates it.
    let available = |reg: IrId, block: IrBlockId| {
        defined_in
            .get(&reg)
            .is_none_or(|d| *d != block && dom.dominates(*d, block))
    };

    for later in &calls {
        for earlier in calls
            .iter()
            .filter(|c| c.key == later.key && c.dest != later.dest)
        {
            let before = earlier.block != later.block || earlier.index < later.index;
            if before && dom.dominates(earlier.block, later.block) {
                replace_with_copy(function, later, earlier.dest);
                return true;
            }
            let preds = function.cfg.blocks[&later.block].predecessors.clone();
            let [p0, p1] = preds[..] else { continue };
            for (missing, has) in [(p0, p1), (p1, p0)] {
                let jumps_here = |b: IrBlockId| {
                    matches!(function.cfg.blocks[&b].terminator,
                        IrTerminator::Branch { target } if target == later.block)
                };
                if !(dom.dominates(earlier.block, has)
                    && !dom.dominates(earlier.block, missing)
                    && jumps_here(missing)
                    && jumps_here(has)
                    && later.key.iter().all(|(base, _)| available(*base, missing)))
                {
                    continue;
                }
                let made = call_in(function, later, missing);
                let joined = function.alloc_reg();
                let ty = function.signature.return_type.clone();
                function.register_types.insert(joined, ty.clone());
                function
                    .cfg
                    .blocks
                    .get_mut(&later.block)
                    .unwrap()
                    .phi_nodes
                    .push(IrPhiNode {
                        dest: joined,
                        incoming: vec![(missing, made), (has, earlier.dest)],
                        ty,
                    });
                replace_with_copy(function, later, joined);
                return true;
            }
        }
    }
    false
}

fn replace_with_copy(function: &mut IrFunction, call: &Call, src: IrId) {
    function
        .cfg
        .blocks
        .get_mut(&call.block)
        .unwrap()
        .instructions[call.index] = IrInstruction::Copy {
        dest: call.dest,
        src,
    };
}

/// Appends to `block` a copy of `call` whose arguments are rebuilt from their
/// affine forms there, and returns its result.
fn call_in(function: &mut IrFunction, call: &Call, block: IrBlockId) -> IrId {
    let template = function.cfg.blocks[&call.block].instructions[call.index].clone();
    let IrInstruction::CallDirect { args, .. } = &template else {
        unreachable!("a self-call site is a CallDirect");
    };
    let mut emitted = Vec::new();
    let mut new_args = Vec::new();
    for (arg, (base, offset)) in args.iter().zip(&call.key) {
        if *offset == 0 {
            new_args.push(*base);
            continue;
        }
        let ty = function
            .register_types
            .get(arg)
            .cloned()
            .unwrap_or(IrType::I32);
        let constant = function.alloc_reg();
        let sum = function.alloc_reg();
        let value = if ty == IrType::I64 {
            IrValue::I64(*offset)
        } else {
            IrValue::I32(*offset as i32)
        };
        function.register_types.insert(constant, ty.clone());
        function.register_types.insert(sum, ty);
        emitted.push(IrInstruction::Const {
            dest: constant,
            value,
        });
        emitted.push(IrInstruction::BinOp {
            dest: sum,
            op: BinaryOp::Add,
            left: *base,
            right: constant,
        });
        new_args.push(sum);
    }
    let result = function.alloc_reg();
    let ty = function.signature.return_type.clone();
    function.register_types.insert(result, ty);
    let mut made = template.clone();
    if let IrInstruction::CallDirect { dest, args, .. } = &mut made {
        *dest = Some(result);
        *args = new_args;
    }
    emitted.push(made);
    function
        .cfg
        .blocks
        .get_mut(&block)
        .unwrap()
        .instructions
        .extend(emitted);
    result
}
