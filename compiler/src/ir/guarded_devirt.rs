//! Guarded devirtualization: a virtual call whose receiver's static class has
//! few implementations of the slot compares the object's class id with each
//! and calls that implementation directly, which the inliner can then expand.
//! Any other class, and a null receiver, takes the vtable lookup as before,
//! so a hierarchy this pass does not see in full only costs it a miss.
//!
//! The implementations come from the `haxe_vtable_set_slot` registrations in
//! `__vtable_init__`, the same table the lookup reads at run time.

use super::optimization::{OptimizationPass, OptimizationResult};
use super::{
    CompareOp, IrBlockId, IrFunction, IrFunctionId, IrId, IrInstruction, IrModule, IrPhiNode,
    IrTerminator, IrType, IrValue,
};
use std::collections::{BTreeMap, BTreeSet};

/// More implementations than this are compared one by one for too long.
const MAX_TARGETS: usize = 4;

/// More class ids than this make the comparisons cost what the lookup does.
const MAX_CLASSES: usize = 8;

const LOOKUP_IN: &str = "haxe_vtable_lookup_in";

pub struct GuardedDevirtualizationPass;

impl GuardedDevirtualizationPass {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GuardedDevirtualizationPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for GuardedDevirtualizationPass {
    fn name(&self) -> &'static str {
        "guarded-devirtualization"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut result = OptimizationResult::unchanged();
        let slots = registered_slots(module);
        if slots.is_empty() {
            return result;
        }
        let children = class_children(module);
        let constructed = constructed_classes(module);
        let lookups: BTreeSet<IrFunctionId> = module
            .extern_functions
            .values()
            .filter(|f| f.name == LOOKUP_IN)
            .map(|f| f.id)
            .chain(
                module
                    .functions
                    .values()
                    .filter(|f| f.cfg.blocks.is_empty() && f.name == LOOKUP_IN)
                    .map(|f| f.id),
            )
            .collect();
        if lookups.is_empty() {
            return result;
        }
        // Parameters, and whether the first is a dispatch thunk's `env`.
        let arity: BTreeMap<IrFunctionId, (usize, bool)> = module
            .functions
            .iter()
            .filter(|(_, f)| !f.cfg.blocks.is_empty())
            .map(|(id, f)| {
                let params = &f.signature.parameters;
                let env = params.first().is_some_and(|p| p.name == "env");
                (*id, (params.len(), env))
            })
            .collect();
        let mut rewritten = 0;
        for function in module.functions.values_mut() {
            while let Some(site) =
                find_site(function, &lookups, &slots, &children, &constructed, &arity)
            {
                rewrite(function, site);
                rewritten += 1;
            }
        }
        if rewritten > 0 {
            result.modified = true;
            result
                .stats
                .insert("virtual_calls_devirtualized".to_string(), rewritten);
        }
        result
    }
}

/// (class id, slot) -> the function registered for it.
fn registered_slots(module: &IrModule) -> BTreeMap<(u32, u32), IrFunctionId> {
    let set_slot: BTreeSet<IrFunctionId> = module
        .extern_functions
        .values()
        .filter(|f| f.name == "haxe_vtable_set_slot")
        .map(|f| f.id)
        .collect();
    let mut slots = BTreeMap::new();
    for function in module
        .functions
        .values()
        .filter(|f| f.name == "__vtable_init__")
    {
        let defs: BTreeMap<IrId, &IrInstruction> = function
            .cfg
            .blocks
            .values()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|i| i.dest().map(|d| (d, i)))
            .collect();
        let constant = |reg: &IrId| match defs.get(reg) {
            Some(IrInstruction::Const { value, .. }) => match *value {
                IrValue::I32(v) => Some(v as u32),
                IrValue::I64(v) => Some(v as u32),
                _ => None,
            },
            _ => None,
        };
        for inst in function.cfg.blocks.values().flat_map(|b| &b.instructions) {
            let IrInstruction::CallDirect { func_id, args, .. } = inst else {
                continue;
            };
            let [class, slot, target] = args[..] else {
                continue;
            };
            if !set_slot.contains(func_id) {
                continue;
            }
            let target = match defs.get(&target) {
                Some(IrInstruction::FunctionRef { func_id, .. }) => *func_id,
                _ => continue,
            };
            if let (Some(class), Some(slot)) = (constant(&class), constant(&slot)) {
                slots.insert((class, slot), target);
            }
        }
    }
    slots
}

/// Classes something constructs: an object header holds the class id as a
/// constant, which `__vtable_init__`'s own registrations do not count as.
fn constructed_classes(module: &IrModule) -> BTreeSet<u32> {
    let mut ids = BTreeSet::new();
    for function in module.functions.values() {
        if function.name == "__vtable_init__" {
            continue;
        }
        for inst in function.cfg.blocks.values().flat_map(|b| &b.instructions) {
            if let IrInstruction::Const { value, .. } = inst {
                match *value {
                    IrValue::I64(v) => ids.insert(v as u32),
                    IrValue::I32(v) => ids.insert(v as u32),
                    _ => false,
                };
            }
        }
    }
    ids
}

/// Each class's direct subclasses, by runtime class id.
fn class_children(module: &IrModule) -> BTreeMap<u32, Vec<u32>> {
    let rtti: BTreeMap<u32, u32> = module
        .types
        .values()
        .filter_map(|t| Some((t.type_id.0, t.runtime_type_id?)))
        .collect();
    let mut children: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for typedef in module.types.values() {
        if let Some(parent) = typedef.super_type_id {
            let own = typedef.runtime_type_id.unwrap_or(typedef.type_id.0);
            let parent = rtti.get(&parent.0).copied().unwrap_or(parent.0);
            children.entry(parent).or_default().push(own);
        }
    }
    children
}

struct Site {
    block: IrBlockId,
    lookup: usize,
    call: usize,
    receiver: IrId,
    /// (class id, implementation, takes a leading `env`), in a fixed order.
    targets: Vec<(u32, IrFunctionId, bool)>,
}

/// A lookup whose result only feeds the indirect call after it in the same
/// block, with few enough implementations to compare against.
fn find_site(
    function: &IrFunction,
    lookups: &BTreeSet<IrFunctionId>,
    slots: &BTreeMap<(u32, u32), IrFunctionId>,
    children: &BTreeMap<u32, Vec<u32>>,
    constructed: &BTreeSet<u32>,
    arity: &BTreeMap<IrFunctionId, (usize, bool)>,
) -> Option<Site> {
    let defs: BTreeMap<IrId, &IrInstruction> = function
        .cfg
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter_map(|i| i.dest().map(|d| (d, i)))
        .collect();
    let constant = |reg: &IrId| match defs.get(reg) {
        Some(IrInstruction::Const { value, .. }) => match *value {
            IrValue::I32(v) => Some(v as u32),
            IrValue::I64(v) => Some(v as u32),
            _ => None,
        },
        _ => None,
    };
    let mut uses: BTreeMap<IrId, usize> = BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            for used in inst.uses() {
                *uses.entry(used).or_default() += 1;
            }
        }
        for phi in &block.phi_nodes {
            for (_, value) in &phi.incoming {
                *uses.entry(*value).or_default() += 1;
            }
        }
    }
    for (block_id, block) in &function.cfg.blocks {
        for (lookup, inst) in block.instructions.iter().enumerate() {
            let IrInstruction::CallDirect {
                func_id,
                args,
                dest: Some(pointer),
                ..
            } = inst
            else {
                continue;
            };
            let [receiver, slot, class] = args[..] else {
                continue;
            };
            if !lookups.contains(func_id) || uses.get(pointer) != Some(&1) {
                continue;
            }
            let (Some(slot), Some(class)) = (constant(&slot), constant(&class)) else {
                continue;
            };
            let Some(call) = block.instructions[lookup + 1..]
                .iter()
                .position(|i| matches!(i, IrInstruction::CallIndirect { func_ptr, .. } if func_ptr == pointer))
                .map(|offset| lookup + 1 + offset)
            else {
                continue;
            };
            let IrInstruction::CallIndirect {
                args: call_args, ..
            } = &block.instructions[call]
            else {
                continue;
            };
            // The class and every subclass this module defines.
            let mut classes = vec![class];
            let mut seen = BTreeSet::from([class]);
            let mut i = 0;
            while i < classes.len() && classes.len() <= 64 {
                for child in children.get(&classes[i]).into_iter().flatten() {
                    if seen.insert(*child) {
                        classes.push(*child);
                    }
                }
                i += 1;
            }
            let mut callable = true;
            let targets: Vec<(u32, IrFunctionId, bool)> = classes
                .iter()
                .filter(|c| constructed.contains(c))
                .filter_map(|c| slots.get(&(*c, slot)).map(|f| (*c, *f)))
                .map(|(c, f)| {
                    let env = match arity.get(&f) {
                        Some(&(n, true)) if n == call_args.len() + 1 => true,
                        Some(&(n, _)) if n == call_args.len() => false,
                        _ => {
                            callable = false;
                            false
                        }
                    };
                    (c, f, env)
                })
                .collect();
            let IrInstruction::CallIndirect { dest, .. } = &block.instructions[call] else {
                continue;
            };
            let typed =
                dest.is_none_or(|d| result_type(function, d, &block.instructions[call]).is_some());
            let distinct: BTreeSet<IrFunctionId> = targets.iter().map(|t| t.1).collect();
            if targets.is_empty()
                || targets.len() > MAX_CLASSES
                || distinct.len() > MAX_TARGETS
                || !callable
                || !typed
            {
                continue;
            }
            return Some(Site {
                block: *block_id,
                lookup,
                call,
                receiver,
                targets,
            });
        }
    }
    None
}

fn rewrite(function: &mut IrFunction, site: Site) {
    let mut fresh = |function: &mut IrFunction, ty: IrType| {
        let reg = IrId::new(function.next_reg_id);
        function.next_reg_id += 1;
        function.register_types.insert(reg, ty);
        reg
    };
    let block = function.cfg.blocks.get_mut(&site.block).unwrap();
    let mut tail = block.instructions.split_off(site.call);
    let call = tail.remove(0);
    let lookup = block.instructions.remove(site.lookup);
    let terminator = std::mem::replace(&mut block.terminator, IrTerminator::Unreachable);
    let IrInstruction::CallIndirect {
        dest,
        args,
        arg_ownership,
        ..
    } = &call
    else {
        unreachable!("a devirtualization site ends in an indirect call");
    };
    let (dest, args, arg_ownership) = (*dest, args.clone(), arg_ownership.clone());
    let result_ty = dest.and_then(|d| result_type(function, d, &call));

    let after = function.cfg.create_block();
    let slow = function.cfg.create_block();
    let load = function.cfg.create_block();

    // Successors now leave from `after`.
    for other in function.cfg.blocks.values_mut() {
        for phi in &mut other.phi_nodes {
            for (from, _) in &mut phi.incoming {
                if *from == site.block {
                    *from = after;
                }
            }
        }
    }

    let null = fresh(function, IrType::I64);
    let is_null = fresh(function, IrType::Bool);
    let no_env = fresh(function, IrType::Ptr(Box::new(IrType::U8)));
    {
        let block = function.cfg.blocks.get_mut(&site.block).unwrap();
        block.instructions.push(IrInstruction::Const {
            dest: null,
            value: IrValue::I64(0),
        });
        block.instructions.push(IrInstruction::Const {
            dest: no_env,
            value: IrValue::Null,
        });
        block.instructions.push(IrInstruction::Cmp {
            dest: is_null,
            op: CompareOp::Eq,
            left: site.receiver,
            right: null,
        });
        block.terminator = IrTerminator::CondBranch {
            condition: is_null,
            true_target: slow,
            false_target: load,
        };
    }

    let header = fresh(function, IrType::I64);
    let low = fresh(function, IrType::I32);
    function.cfg.blocks.get_mut(&load).unwrap().instructions = vec![
        IrInstruction::Load {
            dest: header,
            ptr: site.receiver,
            ty: IrType::I64,
        },
        IrInstruction::Cast {
            dest: low,
            src: header,
            from_ty: IrType::I64,
            to_ty: IrType::I32,
        },
    ];

    // One call block per implementation; each class id tests into its own.
    let mut groups: Vec<(IrFunctionId, bool, Vec<u32>)> = Vec::new();
    for (class, target, env) in &site.targets {
        match groups.iter_mut().find(|g| g.0 == *target) {
            Some(group) => group.2.push(*class),
            None => groups.push((*target, *env, vec![*class])),
        }
    }
    let mut incoming = Vec::new();
    let mut test = load;
    for (target, env, classes) in &groups {
        let direct = function.cfg.create_block();
        for class in classes {
            let id = fresh(function, IrType::I32);
            let same = fresh(function, IrType::Bool);
            let next = function.cfg.create_block();
            let block = function.cfg.blocks.get_mut(&test).unwrap();
            block.instructions.push(IrInstruction::Const {
                dest: id,
                value: IrValue::I32(*class as i32),
            });
            block.instructions.push(IrInstruction::Cmp {
                dest: same,
                op: CompareOp::Eq,
                left: low,
                right: id,
            });
            block.terminator = IrTerminator::CondBranch {
                condition: same,
                true_target: direct,
                false_target: next,
            };
            test = next;
        }
        let value = result_ty.clone().map(|ty| fresh(function, ty));
        let block = function.cfg.blocks.get_mut(&direct).unwrap();
        let (direct_args, direct_ownership) = if *env {
            let mut a = vec![no_env];
            a.extend(args.iter().copied());
            let mut o = vec![super::OwnershipMode::Copy];
            o.extend(arg_ownership.iter().cloned());
            (a, o)
        } else {
            (args.clone(), arg_ownership.clone())
        };
        block.instructions.push(IrInstruction::CallDirect {
            dest: value,
            func_id: *target,
            args: direct_args,
            arg_ownership: direct_ownership,
            type_args: Vec::new(),
            is_tail_call: false,
        });
        block.terminator = IrTerminator::Branch { target: after };
        if let Some(value) = value {
            incoming.push((direct, value));
        }
    }
    function.cfg.blocks.get_mut(&test).unwrap().terminator = IrTerminator::Branch { target: slow };

    // The lookup left here names a class with no registrations, so this
    // pass never takes it for a site again.
    let unknown = fresh(function, IrType::I32);
    let mut lookup = lookup;
    if let IrInstruction::CallDirect { args, .. } = &mut lookup {
        args[2] = unknown;
    }
    let slow_value = result_ty.clone().map(|ty| fresh(function, ty));
    let mut slow_call = call.clone();
    if let IrInstruction::CallIndirect { dest, .. } = &mut slow_call {
        *dest = slow_value;
    }
    {
        let block = function.cfg.blocks.get_mut(&slow).unwrap();
        block.instructions = vec![
            IrInstruction::Const {
                dest: unknown,
                value: IrValue::I32(-1),
            },
            lookup,
            slow_call,
        ];
        block.terminator = IrTerminator::Branch { target: after };
    }
    if let Some(value) = slow_value {
        incoming.push((slow, value));
    }

    let block = function.cfg.blocks.get_mut(&after).unwrap();
    if let (Some(dest), Some(ty)) = (dest, result_ty) {
        block.phi_nodes.push(IrPhiNode { dest, incoming, ty });
    }
    block.instructions = tail;
    block.terminator = terminator;
    function.cfg.recompute_predecessors();
}

/// The type of an indirect call's result: its register's, else the
/// signature's return type.
fn result_type(function: &IrFunction, dest: IrId, call: &IrInstruction) -> Option<IrType> {
    function
        .register_types
        .get(&dest)
        .cloned()
        .or_else(|| match call {
            IrInstruction::CallIndirect {
                signature: IrType::Function { return_type, .. },
                ..
            } => Some((**return_type).clone()),
            _ => None,
        })
}
