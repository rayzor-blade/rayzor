//! Tree fusion: a call that builds a recursive structure, consumed at once by
//! a recursive fold that only reads it, becomes one recursive function that
//! never allocates.
//!
//! The producer `P(args)` returns, at every return, a fresh object: an
//! allocation, header stores, and a constructor that stores its parameters
//! into fields. A field holds a scalar computed by `P`, the result of a
//! recursive call of `P`, or null. The consumer `M(this)` reads fields of
//! `this`, tests them against null, and calls itself on the children. Then
//! `M(P(args))` is `G(args)`, where `G` runs `P`'s control flow and, in place
//! of each return, runs `M` on the fields it would have stored: a child is
//! `G` of the arguments `P` was called with, and a null test of a field whose
//! value is known is decided.
//!
//! Every function involved must be pure apart from the allocation, so calls
//! may be reordered: the children are folded where `P` would have built them.
//! Runs before inlining, while the calls are still calls, and before free
//! insertion, so no release of the folded temporary exists yet.

use super::optimization::{InstructionExt, OptimizationPass, OptimizationResult};
use super::{
    BinaryOp, CompareOp, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule,
    IrPhiNode, IrTerminator, IrType, IrValue,
};
use std::collections::{BTreeMap, BTreeSet};

pub struct TreeFusionPass;

impl TreeFusionPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TreeFusionPass {
    fn default() -> Self {
        Self::new()
    }
}

/// Calls that only maintain stack-trace positions; folding drops them.
const FRAME_UPDATES: &[&str] = &["rayzor_update_call_frame_location"];

/// Functions that return a fresh object.
const ALLOCATORS: &[&str] = &[
    "malloc",
    "haxe_type_create_empty_instance",
    "haxe_type_create_instance",
];

impl OptimizationPass for TreeFusionPass {
    fn name(&self) -> &'static str {
        "tree-fusion"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        let names = function_names(module);
        let mut fused: BTreeMap<(IrFunctionId, IrFunctionId), Option<IrFunctionId>> =
            BTreeMap::new();
        let callers: Vec<IrFunctionId> = module.functions.keys().copied().collect();
        for caller in callers {
            for site in find_sites(&module.functions[&caller], &names) {
                let key = (site.producer, site.consumer);
                let helper = match fused.get(&key) {
                    Some(h) => *h,
                    None => {
                        let h = build_fused(module, &names, site.producer, site.consumer);
                        fused.insert(key, h);
                        h
                    }
                };
                let Some(helper) = helper else { continue };
                if rewrite_site(module.functions.get_mut(&caller).unwrap(), &site, helper) {
                    result.modified = true;
                    *result.stats.entry("trees_fused".to_string()).or_insert(0) += 1;
                }
            }
        }
        result
    }
}

fn function_names(module: &IrModule) -> BTreeMap<IrFunctionId, String> {
    let mut names: BTreeMap<IrFunctionId, String> = module
        .extern_functions
        .iter()
        .map(|(id, f)| (*id, f.name.clone()))
        .collect();
    for (id, f) in &module.functions {
        names.entry(*id).or_insert_with(|| f.name.clone());
    }
    names
}

fn named(names: &BTreeMap<IrFunctionId, String>, id: &IrFunctionId, set: &[&str]) -> bool {
    names.get(id).is_some_and(|n| set.contains(&n.as_str()))
}

/// `v = M(r)` right after `r = P(args)`, `r` used nowhere else.
struct Site {
    block: IrBlockId,
    producer: IrFunctionId,
    consumer: IrFunctionId,
    built: IrId,
    folded: IrId,
    args: Vec<IrId>,
}

fn find_sites(function: &IrFunction, names: &BTreeMap<IrFunctionId, String>) -> Vec<Site> {
    let mut uses: BTreeMap<IrId, usize> = BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            for (_, v) in &phi.incoming {
                *uses.entry(*v).or_insert(0) += 1;
            }
        }
        for inst in &block.instructions {
            for v in inst.uses() {
                *uses.entry(v).or_insert(0) += 1;
            }
        }
        for v in terminator_uses(&block.terminator) {
            *uses.entry(v).or_insert(0) += 1;
        }
    }
    let mut sites = Vec::new();
    for (id, block) in &function.cfg.blocks {
        let insts = &block.instructions;
        for (i, inst) in insts.iter().enumerate() {
            let IrInstruction::CallDirect {
                dest: Some(built),
                func_id: producer,
                args,
                ..
            } = inst
            else {
                continue;
            };
            if uses.get(built) != Some(&1) {
                continue;
            }
            // The next call that is not a frame update must fold `built`.
            let next = insts[i + 1..].iter().find(|inst| match inst {
                IrInstruction::CallDirect { func_id, .. } => !named(names, func_id, FRAME_UPDATES),
                IrInstruction::CallIndirect { .. } => true,
                _ => inst.uses().contains(built),
            });
            if let Some(IrInstruction::CallDirect {
                dest: Some(folded),
                func_id: consumer,
                args: consumer_args,
                ..
            }) = next
            {
                if consumer_args.as_slice() == [*built] && consumer != producer {
                    sites.push(Site {
                        block: *id,
                        producer: *producer,
                        consumer: *consumer,
                        built: *built,
                        folded: *folded,
                        args: args.clone(),
                    });
                }
            }
        }
    }
    sites
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

fn constant(function: &IrFunction, reg: IrId) -> Option<&IrValue> {
    function
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .find_map(|inst| match inst {
            IrInstruction::Const { dest, value } if *dest == reg => Some(value),
            _ => None,
        })
}

fn const_int(function: &IrFunction, reg: IrId) -> Option<i64> {
    match constant(function, reg)? {
        IrValue::I32(v) => Some(i64::from(*v)),
        IrValue::I64(v) => Some(*v),
        IrValue::U32(v) => Some(i64::from(*v)),
        IrValue::U64(v) => i64::try_from(*v).ok(),
        _ => None,
    }
}

fn is_null_const(function: &IrFunction, reg: IrId) -> bool {
    matches!(
        constant(function, reg),
        Some(IrValue::Null) | Some(IrValue::I64(0))
    )
}

/// Arithmetic that cannot trap or touch memory.
fn pure_scalar(inst: &IrInstruction) -> bool {
    match inst {
        IrInstruction::Const { .. }
        | IrInstruction::Copy { .. }
        | IrInstruction::Cmp { .. }
        | IrInstruction::UnOp { .. }
        | IrInstruction::Cast { .. }
        | IrInstruction::BitCast { .. }
        | IrInstruction::Select { .. } => true,
        IrInstruction::BinOp { op, .. } => !matches!(
            op,
            BinaryOp::Div | BinaryOp::Rem | BinaryOp::FDiv | BinaryOp::FRem
        ),
        _ => false,
    }
}

/// The slot each constructor parameter is stored to: `this.slot = param`.
fn constructor_slots(ctor: &IrFunction) -> Option<BTreeMap<usize, i64>> {
    let params: Vec<IrId> = ctor.signature.parameters.iter().map(|p| p.reg).collect();
    let this = *params.first()?;
    let mut slot_of_gep: BTreeMap<IrId, i64> = BTreeMap::new();
    let mut stores: BTreeMap<usize, i64> = BTreeMap::new();
    for block in ctor.cfg.blocks.values() {
        if !block.phi_nodes.is_empty() {
            return None;
        }
        match block.terminator {
            IrTerminator::Return { value: None } | IrTerminator::Unreachable => {}
            _ => return None,
        }
        for inst in &block.instructions {
            match inst {
                IrInstruction::Const { .. } => {}
                IrInstruction::GetElementPtr {
                    dest, ptr, indices, ..
                } if *ptr == this && indices.len() == 1 => {
                    slot_of_gep.insert(*dest, const_int(ctor, indices[0])?);
                }
                IrInstruction::Store { ptr, value, .. } => {
                    let slot = *slot_of_gep.get(ptr)?;
                    let param = params[1..].iter().position(|p| p == value)? + 1;
                    stores.insert(param, slot);
                }
                _ => return None,
            }
        }
    }
    Some(stores)
}

/// What a field of the object a return builds holds.
#[derive(Clone, Copy)]
enum Field {
    Scalar(IrId),
    /// The result of the producer's call at this register.
    Child(IrId),
    Null,
}

/// One return of the producer: its block and the fields it stores.
struct Build {
    block: IrBlockId,
    object: IrId,
    fields: BTreeMap<i64, Field>,
}

fn producer_builds(
    module: &IrModule,
    names: &BTreeMap<IrFunctionId, String>,
    producer_id: IrFunctionId,
) -> Option<Vec<Build>> {
    let producer = module.functions.get(&producer_id)?;
    if producer.cfg.blocks.is_empty() {
        return None;
    }
    let self_calls: BTreeSet<IrId> = producer
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|inst| match inst {
            IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                type_args,
                ..
            } if *func_id == producer_id && type_args.is_empty() => Some(*d),
            _ => None,
        })
        .collect();
    let mut builds = Vec::new();
    for (id, block) in &producer.cfg.blocks {
        if block
            .phi_nodes
            .iter()
            .any(|phi| phi.incoming.iter().any(|(_, v)| self_calls.contains(v)))
        {
            return None;
        }
        let returned = match block.terminator {
            IrTerminator::Return { value: Some(v) } => Some(v),
            IrTerminator::Return { value: None } => return None,
            _ => None,
        };
        let mut object = None;
        let mut slot_of_gep: BTreeMap<IrId, i64> = BTreeMap::new();
        let mut fields: BTreeMap<i64, Field> = BTreeMap::new();
        let mut constructed = false;
        for inst in &block.instructions {
            match inst {
                inst if pure_scalar(inst) => {
                    if inst.uses().iter().any(|u| self_calls.contains(u)) {
                        return None;
                    }
                }
                IrInstruction::CallDirect { func_id, .. }
                    if named(names, func_id, FRAME_UPDATES) => {}
                IrInstruction::CallDirect {
                    dest: Some(d),
                    func_id,
                    args,
                    ..
                } if *func_id == producer_id => {
                    if args.iter().any(|a| self_calls.contains(a)) || !self_calls.contains(d) {
                        return None;
                    }
                }
                IrInstruction::CallDirect {
                    dest: Some(d),
                    func_id,
                    ..
                } if named(names, func_id, ALLOCATORS) && object.is_none() => {
                    object = Some(*d);
                }
                IrInstruction::GetElementPtr {
                    dest, ptr, indices, ..
                } if Some(*ptr) == object && indices.len() == 1 => {
                    slot_of_gep.insert(*dest, const_int(producer, indices[0])?);
                }
                IrInstruction::Store { ptr, value, .. } => {
                    let slot = *slot_of_gep.get(ptr)?;
                    let field = if self_calls.contains(value) {
                        Field::Child(*value)
                    } else if is_null_const(producer, *value) {
                        Field::Null
                    } else {
                        Field::Scalar(*value)
                    };
                    fields.insert(slot, field);
                }
                IrInstruction::CallDirect {
                    dest: None,
                    func_id,
                    args,
                    ..
                } if object.is_some() && args.first() == object.as_ref() && !constructed => {
                    let ctor = module.functions.get(func_id)?;
                    let slots = constructor_slots(ctor)?;
                    for (param, slot) in slots {
                        let value = *args.get(param)?;
                        let field = if self_calls.contains(&value) {
                            Field::Child(value)
                        } else if is_null_const(producer, value) {
                            Field::Null
                        } else {
                            Field::Scalar(value)
                        };
                        fields.insert(slot, field);
                    }
                    constructed = true;
                }
                _ => return None,
            }
        }
        if let Some(returned) = returned {
            if Some(returned) != object || !constructed {
                return None;
            }
            builds.push(Build {
                block: *id,
                object: returned,
                fields,
            });
        } else if object.is_some() {
            return None;
        }
    }
    (!builds.is_empty()).then_some(builds)
}

/// The consumer's body, when it only reads `this` as the fold needs.
fn consumer_shape(
    module: &IrModule,
    names: &BTreeMap<IrFunctionId, String>,
    consumer_id: IrFunctionId,
) -> Option<()> {
    let consumer = module.functions.get(&consumer_id)?;
    let [this] = &consumer.signature.parameters[..] else {
        return None;
    };
    let this = this.reg;
    if !matches!(consumer.signature.return_type, IrType::I32 | IrType::I64) {
        return None;
    }
    let field_geps: BTreeSet<IrId> = consumer
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|inst| match inst {
            IrInstruction::GetElementPtr { dest, ptr, .. } if *ptr == this => Some(*dest),
            _ => None,
        })
        .collect();
    for block in consumer.cfg.blocks.values() {
        for inst in &block.instructions {
            let ok = match inst {
                inst if pure_scalar(inst) => !inst.uses().contains(&this),
                IrInstruction::GetElementPtr { ptr, indices, .. } => {
                    *ptr == this && indices.len() == 1 && const_int(consumer, indices[0]).is_some()
                }
                IrInstruction::Load { ptr, .. } => field_geps.contains(ptr),
                IrInstruction::CallDirect { func_id, .. }
                    if named(names, func_id, FRAME_UPDATES) =>
                {
                    true
                }
                IrInstruction::CallDirect {
                    func_id,
                    args,
                    type_args,
                    ..
                } => *func_id == consumer_id && args.len() == 1 && type_args.is_empty(),
                _ => false,
            };
            if !ok {
                return None;
            }
        }
        if terminator_uses(&block.terminator).contains(&this) {
            return None;
        }
        for phi in &block.phi_nodes {
            if phi.incoming.iter().any(|(_, v)| *v == this) {
                return None;
            }
        }
    }
    Some(())
}

/// What a consumer register is, for one build: a known field value or an
/// ordinary register of the copy.
#[derive(Clone, Copy)]
enum Known {
    Field(Field),
    Null,
}

fn build_fused(
    module: &mut IrModule,
    names: &BTreeMap<IrFunctionId, String>,
    producer_id: IrFunctionId,
    consumer_id: IrFunctionId,
) -> Option<IrFunctionId> {
    let builds = producer_builds(module, names, producer_id)?;
    consumer_shape(module, names, consumer_id)?;
    let producer = module.functions.get(&producer_id)?.clone();
    let consumer = module.functions.get(&consumer_id)?.clone();
    let result_ty = consumer.signature.return_type.clone();
    let this = consumer.signature.parameters[0].reg;
    let helper_id = module.alloc_function_id();

    let mut helper = producer.clone();
    helper.id = helper_id;
    helper.name = format!("{}__fold_{}", producer.name, consumer.name);
    helper.qualified_name = producer
        .qualified_name
        .as_ref()
        .map(|q| format!("{}__fold_{}", q, consumer.name));
    helper.signature.return_type = result_ty.clone();

    // The producer's recursive calls fold their subtree instead.
    for block in helper.cfg.blocks.values_mut() {
        for inst in &mut block.instructions {
            if let IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                ..
            } = inst
            {
                if *func_id == producer_id {
                    *func_id = helper_id;
                    helper.register_types.insert(*d, result_ty.clone());
                }
            }
        }
    }

    for build in &builds {
        // Keep the block's scalar work and recursive calls; drop the object.
        let block = helper.cfg.blocks.get_mut(&build.block)?;
        let object = build.object;
        let mut object_regs: BTreeSet<IrId> = BTreeSet::from([object]);
        block.instructions.retain(|inst| {
            let touches = match inst {
                IrInstruction::CallDirect {
                    dest: Some(d),
                    func_id,
                    ..
                } if named(names, func_id, ALLOCATORS) && *d == object => true,
                IrInstruction::GetElementPtr { dest, ptr, .. } if object_regs.contains(ptr) => {
                    object_regs.insert(*dest);
                    true
                }
                IrInstruction::Store { ptr, .. } if object_regs.contains(ptr) => true,
                IrInstruction::CallDirect { args, .. } if args.first() == Some(&object) => true,
                IrInstruction::CallDirect { func_id, .. } => named(names, func_id, FRAME_UPDATES),
                _ => false,
            };
            !touches
        });

        // A copy of the consumer, reading the fields from what was stored.
        let entry = copy_consumer(&mut helper, &consumer, names, this, &build.fields)?;
        let block = helper.cfg.blocks.get_mut(&build.block)?;
        block.terminator = IrTerminator::Branch { target: entry };
    }
    helper.cfg.recompute_predecessors();
    module.add_function(helper);
    Some(helper_id)
}

/// Copy the consumer's blocks into `helper` for one build; returns the copy's
/// entry. Field reads become the stored values, decided null tests become
/// jumps, and a fold of a child becomes the child's folded value.
fn copy_consumer(
    helper: &mut IrFunction,
    consumer: &IrFunction,
    names: &BTreeMap<IrFunctionId, String>,
    this: IrId,
    fields: &BTreeMap<i64, Field>,
) -> Option<IrBlockId> {
    let mut blocks: BTreeMap<IrBlockId, IrBlockId> = BTreeMap::new();
    for id in consumer.cfg.blocks.keys() {
        blocks.insert(*id, helper.cfg.create_block());
    }
    let mut regs: BTreeMap<IrId, IrId> = BTreeMap::new();
    let mut known: BTreeMap<IrId, Known> = BTreeMap::new();
    let mut slot_of_gep: BTreeMap<IrId, i64> = BTreeMap::new();
    for block in consumer.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInstruction::GetElementPtr {
                    dest, ptr, indices, ..
                } if *ptr == this => {
                    slot_of_gep.insert(*dest, const_int(consumer, indices[0])?);
                }
                IrInstruction::Load { dest, ptr, .. } => {
                    let slot = *slot_of_gep.get(ptr)?;
                    known.insert(
                        *dest,
                        Known::Field(*fields.get(&slot).unwrap_or(&Field::Null)),
                    );
                }
                IrInstruction::Const { dest, .. } if is_null_const(consumer, *dest) => {
                    known.insert(*dest, Known::Null);
                }
                _ => {}
            }
            if let Some(dest) = inst.dest() {
                let new = helper.alloc_reg();
                if let Some(ty) = consumer.register_types.get(&dest) {
                    helper.register_types.insert(new, ty.clone());
                }
                regs.insert(dest, new);
            }
        }
        for phi in &block.phi_nodes {
            let new = helper.alloc_reg();
            helper.register_types.insert(new, phi.ty.clone());
            regs.insert(phi.dest, new);
        }
    }
    // A scalar field read stands for the stored value. A child may only be
    // folded or tested against null, both handled where they occur; anywhere
    // else it would be the object, which no longer exists.
    let value_of = |reg: IrId| -> Option<IrId> {
        match known.get(&reg) {
            Some(Known::Field(Field::Scalar(v))) => Some(*v),
            Some(Known::Field(Field::Child(_))) => None,
            _ => regs.get(&reg).copied(),
        }
    };
    // `a == null` / `a != null` with `a` a field and the other side null.
    let decided = |op: CompareOp, left: IrId, right: IrId| -> Option<bool> {
        let null_of = |r: IrId| match known.get(&r) {
            Some(Known::Null) | Some(Known::Field(Field::Null)) => Some(true),
            Some(Known::Field(Field::Child(_))) => Some(false),
            _ => None,
        };
        let is_null = match (null_of(left), null_of(right)) {
            (Some(a), Some(true)) => a,
            (Some(true), Some(b)) => b,
            _ => return None,
        };
        match op {
            CompareOp::Eq => Some(is_null),
            CompareOp::Ne => Some(!is_null),
            _ => None,
        }
    };

    let mut decisions: BTreeMap<IrId, bool> = BTreeMap::new();
    let mut copied: Vec<(IrBlockId, super::IrBasicBlock)> = Vec::new();
    for (id, block) in &consumer.cfg.blocks {
        let mut out = super::IrBasicBlock::new(blocks[id]);
        for phi in &block.phi_nodes {
            out.phi_nodes.push(IrPhiNode {
                dest: regs[&phi.dest],
                incoming: phi
                    .incoming
                    .iter()
                    .map(|(b, v)| Some((blocks[b], value_of(*v)?)))
                    .collect::<Option<Vec<_>>>()?,
                ty: phi.ty.clone(),
            });
        }
        for inst in &block.instructions {
            match inst {
                IrInstruction::GetElementPtr { .. } => continue,
                IrInstruction::Load { dest, .. } => {
                    if let Some(Known::Field(Field::Null)) = known.get(dest) {
                        out.instructions.push(IrInstruction::Const {
                            dest: regs[dest],
                            value: IrValue::Null,
                        });
                    }
                    continue;
                }
                IrInstruction::CallDirect { func_id, .. }
                    if named(names, func_id, FRAME_UPDATES) =>
                {
                    continue;
                }
                IrInstruction::Cmp {
                    dest,
                    op,
                    left,
                    right,
                } if decided(*op, *left, *right).is_some() => {
                    let value = decided(*op, *left, *right)?;
                    decisions.insert(*dest, value);
                    out.instructions.push(IrInstruction::Const {
                        dest: regs[dest],
                        value: IrValue::Bool(value),
                    });
                    continue;
                }
                IrInstruction::CallDirect {
                    dest: Some(d),
                    args,
                    ..
                } => {
                    // A fold of a child is its folded value; of a null field it
                    // must be unreachable, which the jump check below enforces.
                    let value = match known.get(&args[0]) {
                        Some(Known::Field(Field::Child(c))) => *c,
                        _ => {
                            out.instructions.push(IrInstruction::Undef {
                                dest: regs[d],
                                ty: helper.register_types.get(&regs[d]).cloned()?,
                            });
                            out.metadata
                                .optimization_hints
                                .push(super::OptimizationHint::Custom("fold-of-null".to_string()));
                            continue;
                        }
                    };
                    out.instructions.push(IrInstruction::Copy {
                        dest: regs[d],
                        src: value,
                    });
                    continue;
                }
                _ => {}
            }
            let mut copy = inst.clone();
            let map: BTreeMap<IrId, IrId> = inst
                .uses()
                .into_iter()
                .filter_map(|u| Some((u, value_of(u)?)))
                .collect();
            copy.replace_uses(&map);
            if let Some(dest) = inst.dest() {
                copy.replace_dest(regs[&dest]);
            }
            out.instructions.push(copy);
        }
        out.terminator = match &block.terminator {
            IrTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => match decisions.get(condition) {
                Some(true) => IrTerminator::Branch {
                    target: blocks[true_target],
                },
                Some(false) => IrTerminator::Branch {
                    target: blocks[false_target],
                },
                None => IrTerminator::CondBranch {
                    condition: value_of(*condition)?,
                    true_target: blocks[true_target],
                    false_target: blocks[false_target],
                },
            },
            IrTerminator::Branch { target } => IrTerminator::Branch {
                target: blocks[target],
            },
            IrTerminator::Return { value } => IrTerminator::Return {
                value: match value {
                    Some(v) => Some(value_of(*v)?),
                    None => None,
                },
            },
            IrTerminator::Unreachable => IrTerminator::Unreachable,
            _ => return None,
        };
        copied.push((blocks[id], out));
    }

    // A fold of a null field must not be reachable from the copy's entry.
    let entry = *blocks.get(&consumer.cfg.entry_block)?;
    let by_id: BTreeMap<IrBlockId, &super::IrBasicBlock> =
        copied.iter().map(|(id, b)| (*id, b)).collect();
    let mut seen = BTreeSet::new();
    let mut work = vec![entry];
    while let Some(id) = work.pop() {
        if !seen.insert(id) {
            continue;
        }
        let block = by_id.get(&id)?;
        if block
            .metadata
            .optimization_hints
            .iter()
            .any(|h| matches!(h, super::OptimizationHint::Custom(n) if n == "fold-of-null"))
        {
            return None;
        }
        work.extend(block.successors());
    }
    for (id, block) in copied {
        helper.cfg.blocks.insert(id, block);
    }
    Some(entry)
}

/// Replace `r = P(args); v = M(r)` by `v = G(args)`.
fn rewrite_site(function: &mut IrFunction, site: &Site, helper: IrFunctionId) -> bool {
    let Some(block) = function.cfg.blocks.get_mut(&site.block) else {
        return false;
    };
    let Some(build_at) = block.instructions.iter().position(
        |inst| matches!(inst, IrInstruction::CallDirect { dest: Some(d), .. } if *d == site.built),
    ) else {
        return false;
    };
    block.instructions.remove(build_at);
    for inst in &mut block.instructions {
        if let IrInstruction::CallDirect {
            dest: Some(d),
            func_id,
            args,
            ..
        } = inst
        {
            if *d == site.folded {
                *func_id = helper;
                *args = site.args.clone();
                return true;
            }
        }
    }
    false
}
