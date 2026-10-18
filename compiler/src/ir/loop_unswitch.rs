//! Loop unswitching and versioning: a loop is split into two copies chosen
//! once before it runs, so the copy that runs carries fewer branches and none
//! of the calls that only the other copy needs. A call left in a loop makes
//! the register allocator keep the loop's live values in memory, so removing
//! the calls is worth more than the branches.
//!
//! - Versioning: the outermost loop of a nest is copied with every array
//!   check decided, guarded by one precondition (each array non-null and
//!   plain, every induction variable's bound within the array it indexes).
//! - Unswitching: an innermost loop that branches on a condition it never
//!   changes is copied once per outcome.

use super::loop_analysis::{DominatorTree, LoopNestInfo, NaturalLoop};
use super::optimization::{InstructionExt, OptimizationPass, OptimizationResult};
use super::{
    ArrayField, BinaryOp, CompareOp, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction,
    IrModule, IrPhiNode, IrTerminator, IrType, IrValue, OptimizationHint,
};
use std::collections::{BTreeMap, BTreeSet};

/// Innermost loops larger than this (instructions and phis) are not unswitched.
const MAX_LOOP_SIZE: usize = 160;

/// Loop nests larger than this are not versioned.
const MAX_NEST_SIZE: usize = 600;

/// Copies made per function, which bounds the growth of a function whose
/// loops branch on several invariant conditions.
const MAX_COPIES_PER_FUNCTION: usize = 6;

pub struct LoopUnswitchPass;

impl LoopUnswitchPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LoopUnswitchPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for LoopUnswitchPass {
    fn name(&self) -> &'static str {
        "loop-unswitch"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        let preserving = super::optimization::header_preserving_functions(module);
        for function in module.functions.values_mut() {
            close_loops(function);
            let mut done = 0;
            while done < MAX_COPIES_PER_FUNCTION && split_one(function, &preserving) {
                done += 1;
            }
            if done > 0 {
                result.modified = true;
                *result.stats.entry("loops_split".to_string()).or_insert(0) += done;
            }
        }
        result
    }
}

/// Gives every value a single-exit loop defines and uses after it a phi in the
/// exit block, so the loop can be copied: after a split, either copy may be
/// the one that defined it, and only a phi in the exit can say which.
fn close_loops(function: &mut IrFunction) {
    for _ in 0..32 {
        let Some((exit, values, inside)) = unclosed_loop(function) else {
            return;
        };
        let preds = function.cfg.blocks[&exit].predecessors.clone();
        let mut replace: BTreeMap<IrId, IrId> = BTreeMap::new();
        for value in values {
            let phi = IrId::new(function.next_reg_id);
            function.next_reg_id += 1;
            let ty = function
                .register_types
                .get(&value)
                .cloned()
                .unwrap_or(IrType::I64);
            function.register_types.insert(phi, ty.clone());
            function
                .cfg
                .blocks
                .get_mut(&exit)
                .unwrap()
                .phi_nodes
                .push(IrPhiNode {
                    dest: phi,
                    incoming: preds.iter().map(|p| (*p, value)).collect(),
                    ty,
                });
            replace.insert(value, phi);
        }
        let get = |reg: &mut IrId| {
            if let Some(value) = replace.get(reg) {
                *reg = *value;
            }
        };
        let new_phis: BTreeSet<IrId> = replace.values().copied().collect();
        for (id, block) in function.cfg.blocks.iter_mut() {
            if inside.contains(id) {
                continue;
            }
            for phi in &mut block.phi_nodes {
                if new_phis.contains(&phi.dest) {
                    continue;
                }
                for (from, value) in &mut phi.incoming {
                    if !inside.contains(from) {
                        get(value);
                    }
                }
            }
            for inst in &mut block.instructions {
                inst.replace_uses(&replace);
            }
            match &mut block.terminator {
                IrTerminator::CondBranch { condition, .. } => get(condition),
                IrTerminator::Switch { value, .. } => get(value),
                IrTerminator::Return { value: Some(v) } => get(v),
                IrTerminator::NoReturn { call } => get(call),
                _ => {}
            }
        }
    }
}

/// A loop with one exit block, entered only from the loop, whose values are
/// used past it other than through that block's phis: the exit block, those
/// values and the loop's blocks.
fn unclosed_loop(function: &IrFunction) -> Option<(IrBlockId, Vec<IrId>, BTreeSet<IrBlockId>)> {
    let domtree = DominatorTree::compute(function);
    let nest = LoopNestInfo::analyze(function, &domtree);
    for natural in nest.loops_innermost_first().into_iter().rev() {
        let cfg = &function.cfg;
        let exits: BTreeSet<IrBlockId> = natural
            .blocks
            .iter()
            .filter_map(|b| cfg.get_block(*b))
            .flat_map(|b| successors(&b.terminator))
            .filter(|s| !natural.blocks.contains(s))
            .collect();
        let [exit] = exits.iter().copied().collect::<Vec<_>>()[..] else {
            continue;
        };
        let Some(exit_block) = cfg.get_block(exit) else {
            continue;
        };
        if exit_block.predecessors.is_empty()
            || !exit_block
                .predecessors
                .iter()
                .all(|p| natural.blocks.contains(p))
        {
            continue;
        }
        let mut defs: BTreeSet<IrId> = BTreeSet::new();
        for id in &natural.blocks {
            if let Some(block) = cfg.get_block(*id) {
                defs.extend(block.phi_nodes.iter().map(|p| p.dest));
                defs.extend(block.instructions.iter().filter_map(|i| i.dest()));
            }
        }
        let mut escaping: BTreeSet<IrId> = BTreeSet::new();
        let mut dominated = true;
        for (id, block) in &cfg.blocks {
            if natural.blocks.contains(id) {
                continue;
            }
            let mut note = |reg: IrId, at: IrBlockId| {
                if defs.contains(&reg) {
                    escaping.insert(reg);
                    dominated &= domtree.dominates(exit, at);
                }
            };
            for phi in &block.phi_nodes {
                for (from, value) in &phi.incoming {
                    if !natural.blocks.contains(from) {
                        note(*value, *from);
                    }
                }
            }
            for reg in block
                .instructions
                .iter()
                .flat_map(|inst| inst.uses())
                .chain(terminator_uses(&block.terminator))
            {
                note(reg, *id);
            }
        }
        if !escaping.is_empty() && dominated {
            return Some((
                exit,
                escaping.into_iter().collect(),
                natural.blocks.iter().copied().collect(),
            ));
        }
    }
    None
}

/// How one loop is split.
struct Split {
    preheader: IrBlockId,
    /// Computed in the preheader before the branch.
    prelude: Vec<IrInstruction>,
    condition: IrId,
    /// Branches decided in the copy that runs when `condition` is true, as
    /// (block, takes its true target).
    when_true: Vec<(IrBlockId, bool)>,
    /// The same for the copy that runs when it is false.
    when_false: Vec<(IrBlockId, bool)>,
    /// Mark both headers so neither is versioned again.
    mark_versioned: bool,
}

/// Split the first eligible loop of `function`; false when none is.
fn split_one(function: &mut IrFunction, preserving: &BTreeSet<IrFunctionId>) -> bool {
    let domtree = DominatorTree::compute(function);
    let nest = LoopNestInfo::analyze(function, &domtree);
    let loops: Vec<&NaturalLoop> = nest.loops_innermost_first();

    // Versioning first, outermost loops first: one copy of a whole nest beats
    // a copy per condition.
    for natural in loops.iter().rev() {
        if let Some(split) = plan_version(function, &domtree, natural, &loops, preserving) {
            log_split(function, "version", natural, &split);
            let natural = (*natural).clone();
            apply(function, &natural, split);
            return true;
        }
    }

    let headers: BTreeSet<IrBlockId> = nest.loops.keys().copied().collect();
    for natural in &loops {
        // Innermost only: an inner loop's header inside this one means a
        // copy of this loop would copy that loop too.
        if headers
            .iter()
            .any(|h| *h != natural.header && natural.blocks.contains(h))
        {
            continue;
        }
        if let Some(split) = plan_unswitch(function, natural) {
            log_split(function, "unswitch", natural, &split);
            let natural = (*natural).clone();
            apply(function, &natural, split);
            return true;
        }
    }
    false
}

/// `RAYZOR_LOOP_SPLIT_LOG=1` names each split loop. Diagnostic only.
fn log_split(function: &IrFunction, kind: &str, natural: &NaturalLoop, split: &Split) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("RAYZOR_LOOP_SPLIT_LOG").is_some()) {
        eprintln!(
            "[loop-split] {kind} {} header={:?} blocks={} decided={} prelude={}",
            function.name,
            natural.header,
            natural.blocks.len(),
            split.when_true.len(),
            split.prelude.len()
        );
    }
}

/// The loop's dedicated preheader and the registers it defines, when the loop
/// can be copied at all: no phi instruction (it names blocks a copy would not
/// remap), and no value leaving it except through an exit block's phi (after a
/// split, either copy may be the one that defined it).
fn copyable(
    function: &IrFunction,
    natural: &NaturalLoop,
    max_size: usize,
) -> Option<(IrBlockId, BTreeSet<IrId>)> {
    let cfg = &function.cfg;
    let mut size = 0;
    let mut defs: BTreeSet<IrId> = BTreeSet::new();
    for id in &natural.blocks {
        let block = cfg.get_block(*id)?;
        if block
            .instructions
            .iter()
            .any(|inst| matches!(inst, IrInstruction::Phi { .. }))
        {
            return None;
        }
        size += block.instructions.len() + block.phi_nodes.len();
        defs.extend(block.phi_nodes.iter().map(|phi| phi.dest));
        defs.extend(block.instructions.iter().filter_map(|inst| inst.dest()));
    }
    if size > max_size {
        return None;
    }

    let header = cfg.get_block(natural.header)?;
    let outside: Vec<IrBlockId> = header
        .predecessors
        .iter()
        .copied()
        .filter(|p| !natural.blocks.contains(p))
        .collect();
    let [preheader] = outside[..] else {
        return None;
    };
    if !matches!(
        cfg.get_block(preheader)?.terminator,
        IrTerminator::Branch { target } if target == natural.header
    ) {
        return None;
    }

    for (id, block) in &cfg.blocks {
        if natural.blocks.contains(id) {
            continue;
        }
        for phi in &block.phi_nodes {
            for (from, value) in &phi.incoming {
                if defs.contains(value) && !natural.blocks.contains(from) {
                    return None;
                }
            }
        }
        let used_outside = block
            .instructions
            .iter()
            .flat_map(|inst| inst.uses())
            .chain(terminator_uses(&block.terminator))
            .any(|v| defs.contains(&v));
        if used_outside {
            return None;
        }
    }
    Some((preheader, defs))
}

fn plan_unswitch(function: &IrFunction, natural: &NaturalLoop) -> Option<Split> {
    let (preheader, defs) = copyable(function, natural, MAX_LOOP_SIZE)?;
    let any_call = |inst: &IrInstruction| {
        matches!(
            inst,
            IrInstruction::CallDirect { .. } | IrInstruction::CallIndirect { .. }
        )
    };
    // Worth a copy only when one outcome leaves the loop with no call at all.
    if !reaches_call(function, natural, &BTreeMap::new(), &any_call) {
        return None;
    }
    for id in &natural.blocks {
        let block = function.cfg.get_block(*id)?;
        if let IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = block.terminator
        {
            if true_target == false_target || defs.contains(&condition) {
                continue;
            }
            let frees = [true, false].iter().any(|outcome| {
                let decided = BTreeMap::from([(*id, *outcome)]);
                !reaches_call(function, natural, &decided, &any_call)
            });
            if frees {
                return Some(Split {
                    preheader,
                    prelude: Vec::new(),
                    condition,
                    when_true: vec![(*id, true)],
                    when_false: vec![(*id, false)],
                    mark_versioned: false,
                });
            }
        }
    }
    None
}

/// Whether an iteration of `natural`, with the branches in `decided` taken as
/// given, can reach an instruction `counts` accepts.
fn reaches_call(
    function: &IrFunction,
    natural: &NaturalLoop,
    decided: &BTreeMap<IrBlockId, bool>,
    counts: &dyn Fn(&IrInstruction) -> bool,
) -> bool {
    let mut seen: BTreeSet<IrBlockId> = BTreeSet::new();
    let mut work = vec![natural.header];
    while let Some(id) = work.pop() {
        if !natural.blocks.contains(&id) || !seen.insert(id) {
            continue;
        }
        let Some(block) = function.cfg.get_block(id) else {
            continue;
        };
        if block.instructions.iter().any(counts) {
            return true;
        }
        match (&block.terminator, decided.get(&id)) {
            (
                IrTerminator::CondBranch {
                    true_target,
                    false_target,
                    ..
                },
                Some(take_true),
            ) => work.push(if *take_true {
                *true_target
            } else {
                *false_target
            }),
            (term, _) => work.extend(successors(term)),
        }
    }
    false
}

/// A loop counting `iv` up by one from `start` while `iv < bound`, `bound`
/// fixed for the whole nest. `body` is where an iteration that passed the
/// test continues, so `start <= iv < bound` in every block it dominates.
struct Induction {
    iv: IrId,
    body: IrBlockId,
    start: IrId,
    bound: IrId,
}

/// What must hold for a branch in the fast copy to be decided.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Need {
    /// An index derived from the loop at this position stays within the
    /// array's length on every iteration.
    InBounds(usize, Index, IrId),
    /// A constant index is within the array's length.
    ConstIndex(i64, IrId),
    /// An index `iv + c`, already checked against the length by a branch the
    /// access sits under, is not negative.
    NonNegative(usize, i64),
    /// The array is plain (zero flags).
    Plain(IrId),
    /// The array is not null.
    NotNull(IrId),
}

/// How an array index follows a loop's induction variable `iv`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Index {
    /// `iv + c`.
    Shift(i64),
    /// `k - iv`, `k` fixed for the nest.
    Mirror(IrId),
}

/// Registers that only rename another: a copy, or a phi whose inputs are all
/// one value or the phi itself. Lowering leaves many such phis, carrying a
/// loop-invariant value through every block of a loop.
pub(super) fn renamings(function: &IrFunction) -> BTreeMap<IrId, IrId> {
    let mut alias: BTreeMap<IrId, IrId> = BTreeMap::new();
    for inst in function
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
    {
        if let IrInstruction::Copy { dest, src } = inst {
            alias.insert(*dest, *src);
        }
    }
    let phis: Vec<(IrId, Vec<IrId>)> = function
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.phi_nodes.iter())
        .map(|p| (p.dest, p.incoming.iter().map(|(_, v)| *v).collect()))
        .collect();
    loop {
        let mut changed = false;
        for (dest, inputs) in &phis {
            if alias.contains_key(dest) {
                continue;
            }
            let values: BTreeSet<IrId> = inputs
                .iter()
                .map(|v| canonical(&alias, *v))
                .filter(|v| v != dest)
                .collect();
            if let [only] = values.iter().copied().collect::<Vec<_>>()[..] {
                alias.insert(*dest, only);
                changed = true;
            }
        }
        if !changed {
            return alias;
        }
    }
}

pub(super) fn canonical(alias: &BTreeMap<IrId, IrId>, mut reg: IrId) -> IrId {
    for _ in 0..64 {
        match alias.get(&reg) {
            Some(next) if *next != reg => reg = *next,
            _ => break,
        }
    }
    reg
}

fn plan_version(
    function: &IrFunction,
    domtree: &DominatorTree,
    nest: &NaturalLoop,
    loops: &[&NaturalLoop],
    preserving: &BTreeSet<IrFunctionId>,
) -> Option<Split> {
    if is_versioned(function, nest.header) {
        return None;
    }
    let (preheader, nest_defs) = copyable(function, nest, MAX_NEST_SIZE)?;
    let def_of: BTreeMap<IrId, &IrInstruction> = function
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|inst| inst.dest().map(|d| (d, inst)))
        .collect();
    let alias = renamings(function);
    let canon = |reg: IrId| canonical(&alias, reg);
    let constant = |reg: IrId| -> Option<i64> {
        match def_of.get(&canon(reg))? {
            IrInstruction::Const { value, .. } => match value {
                IrValue::I32(v) => Some(i64::from(*v)),
                IrValue::I64(v) => Some(*v),
                IrValue::U32(v) => Some(i64::from(*v)),
                _ => None,
            },
            _ => None,
        }
    };
    let outside = |reg: IrId| !nest_defs.contains(&canon(reg));

    // Every counted loop in the nest.
    let mut inductions: Vec<Induction> = Vec::new();
    for natural in loops {
        if !nest.blocks.contains(&natural.header) {
            continue;
        }
        let Some(header) = function.cfg.get_block(natural.header) else {
            continue;
        };
        let IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = header.terminator
        else {
            continue;
        };
        if !natural.blocks.contains(&true_target) || natural.blocks.contains(&false_target) {
            continue;
        }
        let Some(IrInstruction::Cmp {
            op: CompareOp::Lt,
            left,
            right: bound,
            ..
        }) = def_of.get(&canon(condition)).copied()
        else {
            continue;
        };
        if !outside(*bound) {
            continue;
        }
        let iv = canon(*left);
        let Some(phi) = header.phi_nodes.iter().find(|p| p.dest == iv) else {
            continue;
        };
        let mut start = None;
        let mut steps_by_one = true;
        for (from, value) in &phi.incoming {
            if natural.blocks.contains(from) {
                let ok = match def_of.get(&canon(*value)) {
                    Some(IrInstruction::BinOp {
                        op: BinaryOp::Add,
                        left,
                        right,
                        ..
                    }) => {
                        (canon(*left) == iv && constant(*right) == Some(1))
                            || (canon(*right) == iv && constant(*left) == Some(1))
                    }
                    _ => false,
                };
                steps_by_one &= ok;
            } else if start.replace(canon(*value)).is_some() {
                steps_by_one = false;
            }
        }
        let Some(start) = start else {
            continue;
        };
        let start_known = constant(start).is_some_and(|s| s >= 0) || outside(start);
        if steps_by_one && start_known {
            inductions.push(Induction {
                iv,
                body: true_target,
                start,
                bound: canon(*bound),
            });
        }
    }

    // The arrays the nest reads through a header, by the array register.
    let arrays: BTreeSet<IrId> = nest
        .blocks
        .iter()
        .filter_map(|b| function.cfg.get_block(*b))
        .flat_map(|b| b.instructions.iter())
        .filter_map(|inst| match inst {
            IrInstruction::ArrayHeader { array, .. } => Some(canon(*array)),
            _ => None,
        })
        .filter(|array| outside(*array))
        .collect();

    // Blocks entered only when `index < array.length` holds: (block, index,
    // array). Within one iteration of the fast copy the length cannot change.
    let strip = |reg: IrId| match def_of.get(&canon(reg)) {
        Some(IrInstruction::Cast { src, .. }) => canon(*src),
        _ => canon(reg),
    };
    let mut guards: Vec<(IrBlockId, IrId, IrId)> = Vec::new();
    for id in &nest.blocks {
        let Some(IrTerminator::CondBranch {
            condition,
            true_target,
            ..
        }) = function.cfg.get_block(*id).map(|b| &b.terminator)
        else {
            continue;
        };
        let single_entry = function
            .cfg
            .get_block(*true_target)
            .is_some_and(|t| t.predecessors.len() == 1);
        if let (
            true,
            Some(IrInstruction::Cmp {
                op: CompareOp::Lt,
                left,
                right,
                ..
            }),
        ) = (single_entry, def_of.get(&canon(*condition)).copied())
        {
            if let Some(IrInstruction::ArrayHeader {
                array,
                field: ArrayField::Len,
                ..
            }) = def_of.get(&strip(*right)).copied()
            {
                guards.push((*true_target, strip(*left), canon(*array)));
            }
        }
    }

    // What a branch condition needs to be decided, and the value it then has.
    struct Ctx<'a> {
        def_of: &'a BTreeMap<IrId, &'a IrInstruction>,
        guards: &'a [(IrBlockId, IrId, IrId)],
        inductions: &'a [Induction],
        arrays: &'a BTreeSet<IrId>,
        domtree: &'a DominatorTree,
        canon: &'a dyn Fn(IrId) -> IrId,
        constant: &'a dyn Fn(IrId) -> Option<i64>,
        outside: &'a dyn Fn(IrId) -> bool,
    }
    fn classify(ctx: &Ctx, block: IrBlockId, cond: IrId) -> Option<(Vec<Need>, bool)> {
        match ctx.def_of.get(&(ctx.canon)(cond))? {
            IrInstruction::BinOp {
                op: BinaryOp::And,
                left,
                right,
                ..
            } => {
                let (mut a, va) = classify(ctx, block, *left)?;
                let (b, vb) = classify(ctx, block, *right)?;
                if !(va && vb) {
                    return None;
                }
                a.extend(b);
                Some((a, true))
            }
            IrInstruction::Cmp {
                op: CompareOp::ULt,
                left,
                right,
                ..
            } => {
                let array = match ctx.def_of.get(&(ctx.canon)(*right))? {
                    IrInstruction::ArrayHeader {
                        array,
                        field: ArrayField::Len,
                        ..
                    } if (ctx.outside)(*array) => (ctx.canon)(*array),
                    _ => return None,
                };
                let index = match ctx.def_of.get(&(ctx.canon)(*left)) {
                    Some(IrInstruction::Cast { src, .. }) => (ctx.canon)(*src),
                    _ => (ctx.canon)(*left),
                };
                if let Some(c) = (ctx.constant)(index) {
                    return (c >= 0).then(|| (vec![Need::ConstIndex(c, array)], true));
                }
                let (k, form) = affine_index(ctx, block, index)?;
                let guarded = ctx.guards.iter().any(|(entered, i, a)| {
                    *i == index && *a == array && ctx.domtree.dominates(*entered, block)
                });
                if let (true, Index::Shift(c)) = (guarded, form) {
                    return match (ctx.constant)(ctx.inductions[k].start) {
                        Some(start) => (start + c >= 0).then(|| (Vec::new(), true)),
                        None => Some((vec![Need::NonNegative(k, c)], true)),
                    };
                }
                // A constant start fixes the lowest index now; a register
                // start is checked in the precondition.
                if let (Index::Shift(c), Some(start)) =
                    (form, (ctx.constant)(ctx.inductions[k].start))
                {
                    if start + c < 0 {
                        return None;
                    }
                }
                Some((vec![Need::InBounds(k, form, array)], true))
            }
            IrInstruction::Cmp {
                op: CompareOp::Eq,
                left,
                right,
                ..
            } if (ctx.constant)(*right) == Some(0) => match ctx.def_of.get(&(ctx.canon)(*left)) {
                Some(IrInstruction::ArrayHeader {
                    array,
                    field: ArrayField::Flags,
                    ..
                }) if (ctx.outside)(*array) => Some((vec![Need::Plain((ctx.canon)(*array))], true)),
                _ if ctx.arrays.contains(&(ctx.canon)(*left)) => {
                    Some((vec![Need::NotNull((ctx.canon)(*left))], false))
                }
                _ => None,
            },
            _ => None,
        }
    }
    /// `index` as a function of a loop's induction variable, for a loop whose
    /// body dominates `block`.
    fn affine_index(ctx: &Ctx, block: IrBlockId, index: IrId) -> Option<(usize, Index)> {
        let loop_of = |reg: IrId| {
            ctx.inductions.iter().position(|ind| {
                ind.iv == (ctx.canon)(reg) && ctx.domtree.dominates(ind.body, block)
            })
        };
        if let Some(k) = loop_of(index) {
            return Some((k, Index::Shift(0)));
        }
        match ctx.def_of.get(&index)? {
            IrInstruction::BinOp {
                op: BinaryOp::Add,
                left,
                right,
                ..
            } => {
                if let (Some(k), Some(c)) = (loop_of(*left), (ctx.constant)(*right)) {
                    return Some((k, Index::Shift(c)));
                }
                let (k, c) = (loop_of(*right)?, (ctx.constant)(*left)?);
                Some((k, Index::Shift(c)))
            }
            IrInstruction::BinOp {
                op: BinaryOp::Sub,
                left,
                right,
                ..
            } => {
                if let (Some(k), Some(c)) = (loop_of(*left), (ctx.constant)(*right)) {
                    return Some((k, Index::Shift(-c)));
                }
                let k = loop_of(*right)?;
                (ctx.outside)(*left).then(|| (k, Index::Mirror((ctx.canon)(*left))))
            }
            _ => None,
        }
    }

    let ctx = Ctx {
        def_of: &def_of,
        guards: &guards,
        inductions: &inductions,
        arrays: &arrays,
        domtree,
        canon: &canon,
        constant: &constant,
        outside: &outside,
    };

    let mut decided: Vec<(IrBlockId, bool)> = Vec::new();
    let mut needs: BTreeSet<Need> = BTreeSet::new();
    for id in &nest.blocks {
        let block = function.cfg.get_block(*id)?;
        if let IrTerminator::CondBranch { condition, .. } = block.terminator {
            if let Some((block_needs, value)) = classify(&ctx, *id, condition) {
                needs.extend(block_needs);
                decided.push((*id, value));
            }
        }
    }
    if decided.is_empty() {
        return None;
    }

    // The fast copy must call nothing that can resize an array.
    let taken: BTreeMap<IrBlockId, bool> = decided.iter().copied().collect();
    let resizes = |inst: &IrInstruction| match inst {
        IrInstruction::CallDirect { func_id, .. } => !preserving.contains(func_id),
        IrInstruction::CallIndirect { .. } => true,
        _ => false,
    };
    if reaches_call(function, nest, &taken, &resizes) {
        return None;
    }

    // The precondition, computed before the nest.
    let mut pre = Prelude::new(function.next_reg_id);
    let zero = pre.push_const(0);
    let mut facts: Vec<IrId> = Vec::new();
    for need in &needs {
        match *need {
            Need::NotNull(array) => facts.push(pre.push_cmp(CompareOp::Ne, array, zero)),
            Need::Plain(array) => {
                let flags = pre.header(array, ArrayField::Flags);
                facts.push(pre.push_cmp(CompareOp::Eq, flags, zero));
            }
            Need::ConstIndex(c, array) => {
                let len = pre.header(array, ArrayField::Len);
                let index = pre.push_const(c);
                facts.push(pre.push_cmp(CompareOp::Lt, index, len));
            }
            Need::NonNegative(k, c) => {
                let ind = &inductions[k];
                let start = pre.widen(ind.start, function.register_types.get(&ind.start));
                let lowest = pre.offset(start, c);
                facts.push(pre.push_cmp(CompareOp::Ge, lowest, zero));
            }
            Need::InBounds(k, form, array) => {
                // With start <= iv < bound: the highest index must be below
                // the length and the lowest must not be negative.
                let ind = &inductions[k];
                let len = pre.header(array, ArrayField::Len);
                let bound = pre.widen(ind.bound, function.register_types.get(&ind.bound));
                let start = pre.widen(ind.start, function.register_types.get(&ind.start));
                match form {
                    Index::Shift(c) => {
                        let shifted = pre.offset(bound, c);
                        facts.push(pre.push_cmp(CompareOp::Le, shifted, len));
                        if constant(ind.start).is_none() {
                            let lowest = pre.offset(start, c);
                            facts.push(pre.push_cmp(CompareOp::Ge, lowest, zero));
                        }
                    }
                    Index::Mirror(base) => {
                        let base = pre.widen(base, function.register_types.get(&base));
                        let highest = pre.push_binop(BinaryOp::Sub, base, start);
                        facts.push(pre.push_cmp(CompareOp::Lt, highest, len));
                        let above = pre.offset(base, 1);
                        facts.push(pre.push_cmp(CompareOp::Le, bound, above));
                    }
                }
            }
        }
    }
    let mut condition = *facts.first()?;
    for fact in &facts[1..] {
        condition = pre.push_and(condition, *fact);
    }
    Some(Split {
        preheader,
        prelude: pre.insts,
        condition,
        when_true: decided,
        when_false: Vec::new(),
        mark_versioned: true,
    })
}

/// Instructions for a preheader, numbering registers from `next`.
struct Prelude {
    next: u32,
    insts: Vec<IrInstruction>,
    headers: BTreeMap<(IrId, ArrayField), IrId>,
}

impl Prelude {
    fn new(next: u32) -> Self {
        Prelude {
            next,
            insts: Vec::new(),
            headers: BTreeMap::new(),
        }
    }

    fn fresh(&mut self) -> IrId {
        let reg = IrId::new(self.next);
        self.next += 1;
        reg
    }

    fn push_const(&mut self, value: i64) -> IrId {
        let dest = self.fresh();
        self.insts.push(IrInstruction::Const {
            dest,
            value: IrValue::I64(value),
        });
        dest
    }

    fn push_cmp(&mut self, op: CompareOp, left: IrId, right: IrId) -> IrId {
        let dest = self.fresh();
        self.insts.push(IrInstruction::Cmp {
            dest,
            op,
            left,
            right,
        });
        dest
    }

    fn push_binop(&mut self, op: BinaryOp, left: IrId, right: IrId) -> IrId {
        let dest = self.fresh();
        self.insts.push(IrInstruction::BinOp {
            dest,
            op,
            left,
            right,
        });
        dest
    }

    /// `reg + c`, or `reg` itself when `c` is zero.
    fn offset(&mut self, reg: IrId, c: i64) -> IrId {
        if c == 0 {
            return reg;
        }
        let amount = self.push_const(c);
        self.push_binop(BinaryOp::Add, reg, amount)
    }

    fn push_and(&mut self, left: IrId, right: IrId) -> IrId {
        let dest = self.fresh();
        self.insts.push(IrInstruction::BinOp {
            dest,
            op: BinaryOp::And,
            left,
            right,
        });
        dest
    }

    fn header(&mut self, array: IrId, field: ArrayField) -> IrId {
        if let Some(reg) = self.headers.get(&(array, field)) {
            return *reg;
        }
        let dest = self.fresh();
        self.insts.push(IrInstruction::ArrayHeader {
            dest,
            array,
            field,
            guarded: true,
        });
        self.headers.insert((array, field), dest);
        dest
    }

    /// `reg` as an I64.
    fn widen(&mut self, reg: IrId, ty: Option<&IrType>) -> IrId {
        if matches!(ty, Some(IrType::I64)) {
            return reg;
        }
        let dest = self.fresh();
        self.insts.push(IrInstruction::Cast {
            dest,
            src: reg,
            from_ty: ty.cloned().unwrap_or(IrType::I32),
            to_ty: IrType::I64,
        });
        dest
    }
}

/// The hint on both headers of a versioned nest.
const VERSIONED: &str = "loop-versioned";

fn is_versioned(function: &IrFunction, header: IrBlockId) -> bool {
    function.cfg.get_block(header).is_some_and(|block| {
        block
            .metadata
            .optimization_hints
            .iter()
            .any(|hint| matches!(hint, OptimizationHint::Custom(name) if name == VERSIONED))
    })
}

fn successors(term: &IrTerminator) -> Vec<IrBlockId> {
    match term {
        IrTerminator::Branch { target } => vec![*target],
        IrTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        IrTerminator::Switch { cases, default, .. } => cases
            .iter()
            .map(|(_, b)| *b)
            .chain(std::iter::once(*default))
            .collect(),
        _ => Vec::new(),
    }
}

fn terminator_uses(term: &IrTerminator) -> Vec<IrId> {
    match term {
        IrTerminator::CondBranch { condition, .. } => vec![*condition],
        IrTerminator::Switch { value, .. } => vec![*value],
        IrTerminator::Return { value: Some(v) } => vec![*v],
        IrTerminator::NoReturn { call } => vec![*call],
        _ => Vec::new(),
    }
}

fn map_terminator(
    term: &IrTerminator,
    blocks: &BTreeMap<IrBlockId, IrBlockId>,
    regs: &BTreeMap<IrId, IrId>,
) -> IrTerminator {
    let block = |b: &IrBlockId| *blocks.get(b).unwrap_or(b);
    let reg = |r: &IrId| *regs.get(r).unwrap_or(r);
    match term {
        IrTerminator::Branch { target } => IrTerminator::Branch {
            target: block(target),
        },
        IrTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => IrTerminator::CondBranch {
            condition: reg(condition),
            true_target: block(true_target),
            false_target: block(false_target),
        },
        IrTerminator::Switch {
            value,
            cases,
            default,
        } => IrTerminator::Switch {
            value: reg(value),
            cases: cases.iter().map(|(k, b)| (*k, block(b))).collect(),
            default: block(default),
        },
        IrTerminator::Return { value } => IrTerminator::Return {
            value: value.as_ref().map(reg),
        },
        IrTerminator::NoReturn { call } => IrTerminator::NoReturn { call: reg(call) },
        IrTerminator::Unreachable => IrTerminator::Unreachable,
    }
}

/// Replace the conditional branch ending `block` by a jump to one target.
fn decide(function: &mut IrFunction, block: IrBlockId, take_true: bool) {
    if let Some(block) = function.cfg.get_block_mut(block) {
        if let IrTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } = block.terminator
        {
            block.terminator = IrTerminator::Branch {
                target: if take_true { true_target } else { false_target },
            };
        }
    }
}

fn apply(function: &mut IrFunction, natural: &NaturalLoop, split: Split) {
    let loop_blocks: Vec<IrBlockId> = natural.blocks.iter().copied().collect();

    // The prelude's registers were numbered from `next_reg_id`; claim them.
    for inst in &split.prelude {
        if let Some(dest) = inst.dest() {
            let ty = match inst {
                IrInstruction::Cmp { .. }
                | IrInstruction::BinOp {
                    op: BinaryOp::And, ..
                } => IrType::Bool,
                IrInstruction::ArrayHeader { field, .. } => field.ir_type(),
                _ => IrType::I64,
            };
            function.register_types.insert(dest, ty);
            function.next_reg_id = function.next_reg_id.max(dest.as_u32() + 1);
        }
    }

    // Fresh registers for everything the copy defines.
    let mut regs: BTreeMap<IrId, IrId> = BTreeMap::new();
    for id in &loop_blocks {
        let block = &function.cfg.blocks[id];
        let defined: Vec<IrId> = block
            .phi_nodes
            .iter()
            .map(|phi| phi.dest)
            .chain(block.instructions.iter().filter_map(|inst| inst.dest()))
            .collect();
        for old in defined {
            let new = function.alloc_reg();
            if let Some(ty) = function.register_types.get(&old).cloned() {
                function.register_types.insert(new, ty);
            }
            regs.insert(old, new);
        }
    }
    let mut blocks: BTreeMap<IrBlockId, IrBlockId> = BTreeMap::new();
    for id in &loop_blocks {
        blocks.insert(*id, function.cfg.create_block());
    }

    // The copy runs when the condition is false.
    for id in &loop_blocks {
        let original = function.cfg.blocks[id].clone();
        let copy_id = blocks[id];
        let mut copy = original.clone();
        copy.id = copy_id;
        for phi in &mut copy.phi_nodes {
            phi.dest = regs[&phi.dest];
            for (from, value) in &mut phi.incoming {
                if let Some(mapped) = blocks.get(from) {
                    *from = *mapped;
                    *value = *regs.get(value).unwrap_or(value);
                }
            }
        }
        for inst in &mut copy.instructions {
            let dest = inst.dest();
            inst.replace_uses(&regs);
            if let Some(dest) = dest {
                inst.replace_dest(regs[&dest]);
            }
        }
        copy.terminator = map_terminator(&original.terminator, &blocks, &regs);
        function.cfg.blocks.insert(copy_id, copy);
    }
    for (id, take_true) in &split.when_true {
        decide(function, *id, *take_true);
    }
    for (id, take_true) in &split.when_false {
        decide(function, blocks[id], *take_true);
    }

    // Each copy gets a preheader of its own, so either can be split again.
    let copy_header = blocks[&natural.header];
    let pre_true = function.cfg.create_block();
    let pre_false = function.cfg.create_block();
    for (pre, header) in [(pre_true, natural.header), (pre_false, copy_header)] {
        if let Some(block) = function.cfg.get_block_mut(pre) {
            block.terminator = IrTerminator::Branch { target: header };
        }
        if let Some(block) = function.cfg.get_block_mut(header) {
            for phi in &mut block.phi_nodes {
                for (from, _) in &mut phi.incoming {
                    if *from == split.preheader {
                        *from = pre;
                    }
                }
            }
        }
    }
    if let Some(pre) = function.cfg.get_block_mut(split.preheader) {
        pre.instructions.extend(split.prelude);
        pre.terminator = IrTerminator::CondBranch {
            condition: split.condition,
            true_target: pre_true,
            false_target: pre_false,
        };
    }

    if split.mark_versioned {
        for header in [natural.header, copy_header] {
            if let Some(block) = function.cfg.get_block_mut(header) {
                block
                    .metadata
                    .optimization_hints
                    .push(OptimizationHint::Custom(VERSIONED.to_string()));
            }
        }
    }

    // Exit phis hear from the copy too.
    let copies: BTreeSet<IrBlockId> = blocks.values().copied().collect();
    for (id, block) in function.cfg.blocks.iter_mut() {
        if natural.blocks.contains(id) || copies.contains(id) {
            continue;
        }
        for phi in &mut block.phi_nodes {
            let extra: Vec<(IrBlockId, IrId)> = phi
                .incoming
                .iter()
                .filter_map(|(from, value)| {
                    blocks
                        .get(from)
                        .map(|copy| (*copy, *regs.get(value).unwrap_or(value)))
                })
                .collect();
            phi.incoming.extend(extra);
        }
    }

    // Deciding branches removed edges; phis keep only edges that still exist.
    function.cfg.recompute_predecessors();
    let preds: BTreeMap<IrBlockId, BTreeSet<IrBlockId>> = function
        .cfg
        .blocks
        .iter()
        .map(|(id, b)| (*id, b.predecessors.iter().copied().collect()))
        .collect();
    for (id, block) in function.cfg.blocks.iter_mut() {
        let live = &preds[id];
        for phi in &mut block.phi_nodes {
            phi.incoming.retain(|(from, _)| live.contains(from));
        }
    }
}
