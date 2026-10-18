//! Tree-shaking for .rzb bundles.
//!
//! Removes unreachable functions, extern declarations, and globals from a set
//! of MIR modules. This reduces bundle size by stripping stdlib functions that
//! the user program never calls.

use super::blocks::{IrBlockId, IrTerminator};
use super::functions::IrFunction;
use super::instructions::IrInstruction;
use super::modules::IrModule;
use super::types::IrValue;
use super::{IrFunctionId, IrGlobalId, IrId};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

fn is_keep_function(func: &crate::ir::functions::IrFunction) -> bool {
    // @:keep attribute
    if func
        .attributes
        .custom
        .get("keep")
        .map(|v| v == "true")
        .unwrap_or(false)
    {
        return true;
    }
    // @:export — WASM exports must survive DCE
    if func.wasm_export {
        return true;
    }
    // export_ prefix convention
    if func.name.starts_with("export_") {
        return true;
    }
    false
}

fn is_startup_function(func: &crate::ir::functions::IrFunction) -> bool {
    matches!(func.name.as_str(), "__vtable_init__" | "__init__")
}

/// Startup registrations made for a class, keyed by its id in the first
/// argument. They matter only for a class something can produce.
const CLASS_KEYED: &[&str] = &[
    "haxe_vtable_init",
    "haxe_vtable_set_slot",
    "haxe_iface_vtable_set_slot",
    "haxe_register_interface_impl",
    "haxe_type_register_to_string",
    "haxe_type_register_constructor",
    "haxe_register_method",
    "haxe_register_static_field",
];

/// Startup registrations keyed by the closure target in the first argument.
const TARGET_KEYED: &[&str] = &[
    "haxe_closure_register_entries",
    "haxe_closure_register_packed_entry",
    "haxe_closure_register_bound_thunk",
];

/// Calls that produce a class no constant in the program names.
const CLASS_BY_NAME: &[&str] = &["haxe_type_resolve_class", "haxe_type_get_super_class"];

#[derive(Clone, Copy, PartialEq)]
enum Key {
    Class(u32),
    Target(IrFunctionId),
}

/// A registration call in a startup hook and the operands only it uses.
struct Registration {
    module: usize,
    function: IrFunctionId,
    key: Key,
    extern_id: IrFunctionId,
    refs: Vec<IrFunctionId>,
    /// (block, instruction index) of the call and of its operands' definitions.
    sites: Vec<(IrBlockId, usize)>,
    kept: bool,
}

fn integer_bits(value: &IrValue) -> Option<u32> {
    match *value {
        IrValue::I32(v) => Some(v as u32),
        IrValue::I64(v) => Some(v as u32),
        IrValue::U32(v) => Some(v),
        IrValue::U64(v) => Some(v as u32),
        _ => None,
    }
}

/// Registrations in `function` whose operands are constants or function
/// references defined for that call alone.
fn find_registrations(
    module_idx: usize,
    module: &IrModule,
    function: &IrFunction,
) -> Vec<Registration> {
    let mut defs: BTreeMap<IrId, (IrBlockId, usize, &IrInstruction)> = BTreeMap::new();
    let mut uses: BTreeMap<IrId, usize> = BTreeMap::new();
    for (block_id, block) in &function.cfg.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let Some(dest) = inst.dest() {
                defs.insert(dest, (*block_id, idx, inst));
            }
            for used in inst.uses() {
                *uses.entry(used).or_default() += 1;
            }
        }
        for phi in &block.phi_nodes {
            for (_, value) in &phi.incoming {
                *uses.entry(*value).or_default() += 1;
            }
        }
        let terminator_use = match &block.terminator {
            IrTerminator::CondBranch { condition, .. } => Some(*condition),
            IrTerminator::Switch { value, .. } => Some(*value),
            IrTerminator::Return { value } => *value,
            IrTerminator::NoReturn { call } => Some(*call),
            _ => None,
        };
        if let Some(value) = terminator_use {
            *uses.entry(value).or_default() += 1;
        }
    }

    let mut found = Vec::new();
    for (block_id, block) in &function.cfg.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            let IrInstruction::CallDirect { func_id, args, .. } = inst else {
                continue;
            };
            let Some(name) = module
                .extern_functions
                .get(func_id)
                .map(|f| f.name.as_str())
            else {
                continue;
            };
            let class_keyed = CLASS_KEYED.contains(&name);
            if !class_keyed && !TARGET_KEYED.contains(&name) {
                continue;
            }
            let mut sites = vec![(*block_id, idx)];
            let mut refs = Vec::new();
            let mut values = Vec::new();
            let owned = args.iter().all(|arg| {
                let Some(&(def_block, def_idx, def)) = defs.get(arg) else {
                    return false;
                };
                if uses.get(arg) != Some(&1) {
                    return false;
                }
                match def {
                    IrInstruction::FunctionRef { func_id, .. }
                    | IrInstruction::Const {
                        value: IrValue::Function(func_id),
                        ..
                    } => refs.push(*func_id),
                    IrInstruction::Const { value, .. } => values.push(integer_bits(value)),
                    _ => return false,
                }
                sites.push((def_block, def_idx));
                true
            });
            if !owned {
                continue;
            }
            let key = match (class_keyed, defs.get(&args[0])) {
                (true, Some((_, _, IrInstruction::Const { value, .. }))) => {
                    integer_bits(value).map(Key::Class)
                }
                (false, Some(_)) => refs.first().copied().map(Key::Target),
                _ => None,
            };
            let Some(key) = key else {
                continue;
            };
            found.push(Registration {
                module: module_idx,
                function: function.id,
                key,
                extern_id: *func_id,
                refs,
                sites,
                kept: false,
            });
        }
    }
    found
}

/// Statistics from tree-shaking.
#[derive(Debug, Default)]
pub struct TreeShakeStats {
    pub functions_removed: usize,
    pub extern_functions_removed: usize,
    pub globals_removed: usize,
    pub modules_removed: usize,
    pub functions_kept: usize,
    pub extern_functions_kept: usize,
    /// Startup instructions removed with registrations for dead classes.
    pub registrations_removed: usize,
}

/// Tree-shake a set of modules, keeping only what's reachable from the entry point.
///
/// Walks the call graph starting from the entry function and marks all transitively
/// reachable functions, extern functions, and globals. Everything else is removed.
///
/// Returns statistics about what was removed.
pub fn tree_shake_bundle(
    modules: &mut Vec<IrModule>,
    entry_module: &str,
    entry_function: &str,
) -> TreeShakeStats {
    let mut stats = TreeShakeStats::default();

    // Phase 1: Find entry function
    let entry = find_entry(modules, entry_module, entry_function);
    let Some((entry_mod_idx, entry_func_id)) = entry else {
        // Can't find entry — don't strip anything
        return stats;
    };

    // Phase 2: Build reachable sets per module
    // Each module has its own function ID space, so we track (module_index, func_id)
    let mut reachable_functions: BTreeSet<(usize, IrFunctionId)> = BTreeSet::new();
    let mut reachable_externs: BTreeSet<(usize, IrFunctionId)> = BTreeSet::new();
    let mut reachable_globals: BTreeSet<(usize, IrGlobalId)> = BTreeSet::new();

    // Worklist: (module_index, func_id) pairs to process
    let mut worklist: Vec<(usize, IrFunctionId)> = Vec::new();

    // Function ids are globally disjoint after import renumbering, so a
    // reference whose target is not in the CURRENT module resolves to its
    // owning module here. Without this, a cross-module MakeClosure/FunctionRef
    // fell through silently and the target was shaken out of the LLVM tier —
    // the Q4 band lambda ran Cranelift's scalar lowering for the whole decode
    // while its callers were promoted (NUC: whole-kernel scalar f64 + soft-f16
    // instead of vpdpbusd).
    let mut owner_of: BTreeMap<IrFunctionId, (usize, bool)> = BTreeMap::new();
    for (mod_idx, module) in modules.iter().enumerate() {
        for func_id in module.functions.keys() {
            owner_of.entry(*func_id).or_insert((mod_idx, false));
        }
        for func_id in module.extern_functions.keys() {
            owner_of.entry(*func_id).or_insert((mod_idx, true));
        }
    }

    // Seed with entry function
    worklist.push((entry_mod_idx, entry_func_id));

    // Seed with @:keep functions and startup hooks across all modules.
    // Startup hooks are invoked externally by backends/wrappers, so they must
    // survive tree-shaking even if nothing in MIR calls them directly.
    for (mod_idx, module) in modules.iter().enumerate() {
        for (func_id, func) in &module.functions {
            if is_keep_function(func) || is_startup_function(func) {
                worklist.push((mod_idx, *func_id));
            }
        }
    }

    // Registrations in startup hooks are walked only once what they are keyed
    // on is live: a class whose id some reachable code holds, or a closure
    // target reachable some other way.
    let mut registrations: Vec<Registration> = Vec::new();
    for (mod_idx, module) in modules.iter().enumerate() {
        for func in module.functions.values() {
            if is_startup_function(func) {
                registrations.extend(find_registrations(mod_idx, module, func));
            }
        }
    }
    let deferred: BTreeSet<(usize, IrFunctionId, IrBlockId, usize)> = registrations
        .iter()
        .flat_map(|r| {
            r.sites
                .iter()
                .map(move |&(block, idx)| (r.module, r.function, block, idx))
        })
        .collect();
    let mut live_class_ids: BTreeSet<u32> = BTreeSet::new();

    // The backend links a forward stub or a bodiless declaration to the body
    // of the same qualified name, and a declaration or an extern to a stdlib
    // wrapper body of the same bare name.
    let mut by_qualified: BTreeMap<String, Vec<(usize, IrFunctionId)>> = BTreeMap::new();
    let mut by_bare: BTreeMap<String, Vec<(usize, IrFunctionId)>> = BTreeMap::new();
    let mapping = crate::stdlib::runtime_mapping::StdlibMapping::builtin();
    for (mod_idx, module) in modules.iter().enumerate() {
        for (id, func) in &module.functions {
            if func.cfg.blocks.is_empty() || func.is_forward_stub() {
                continue;
            }
            if let Some(qualified) = &func.qualified_name {
                by_qualified
                    .entry(qualified.clone())
                    .or_default()
                    .push((mod_idx, *id));
            }
            if mapping.is_mir_wrapper_function(&func.name) {
                by_bare
                    .entry(func.name.clone())
                    .or_default()
                    .push((mod_idx, *id));
            }
        }
    }
    let mut linked_externs: BTreeSet<(usize, IrFunctionId)> = BTreeSet::new();

    // A class's lookups fall back to its ancestors' registrations, so an
    // ancestor is live with it.
    let mut parent_of: BTreeMap<u32, u32> = BTreeMap::new();
    for module in modules.iter() {
        let rtti: BTreeMap<u32, u32> = module
            .types
            .values()
            .filter_map(|t| Some((t.type_id.0, t.runtime_type_id?)))
            .collect();
        for typedef in module.types.values() {
            if let Some(parent) = typedef.super_type_id {
                let own = typedef.runtime_type_id.unwrap_or(typedef.type_id.0);
                let parent = rtti.get(&parent.0).copied().unwrap_or(parent.0);
                parent_of.entry(own).or_insert(parent);
            }
        }
    }

    // Phase 3: Walk call graph
    loop {
        while let Some((mod_idx, func_id)) = worklist.pop() {
            if !reachable_functions.insert((mod_idx, func_id)) {
                continue; // Already visited
            }

            let Some(module) = modules.get(mod_idx) else {
                continue;
            };
            let Some(function) = module.functions.get(&func_id) else {
                // Might be an extern function — mark it
                if module.extern_functions.contains_key(&func_id) {
                    reachable_externs.insert((mod_idx, func_id));
                }
                continue;
            };
            if function.cfg.blocks.is_empty() || function.is_forward_stub() {
                let qualified = function
                    .qualified_name
                    .as_ref()
                    .and_then(|name| by_qualified.get(name));
                let bare = function
                    .cfg
                    .blocks
                    .is_empty()
                    .then(|| by_bare.get(&function.name))
                    .flatten();
                worklist.extend(qualified.into_iter().chain(bare).flatten().copied());
            }

            // Scan all instructions in this function
            for (block_id, block) in &function.cfg.blocks {
                for (idx, inst) in block.instructions.iter().enumerate() {
                    if deferred.contains(&(mod_idx, func_id, *block_id, idx)) {
                        continue;
                    }
                    if let IrInstruction::Const { value, .. } = inst {
                        if let Some(bits) = integer_bits(value) {
                            live_class_ids.insert(bits);
                        }
                    }
                    match inst {
                        IrInstruction::CallDirect {
                            func_id: callee, ..
                        }
                        | IrInstruction::FunctionRef {
                            func_id: callee, ..
                        }
                        | IrInstruction::MakeClosure {
                            func_id: callee, ..
                        } => {
                            if let Some(&(owner, is_extern)) = owner_of.get(callee) {
                                if is_extern {
                                    reachable_externs.insert((owner, *callee));
                                } else {
                                    worklist.push((owner, *callee));
                                }
                            }
                        }
                        IrInstruction::Const {
                            value: crate::ir::types::IrValue::Function(ref_id),
                            ..
                        } => {
                            // Function pointers stored as constants (e.g., `var fn = plusOne;`)
                            if let Some(&(owner, is_extern)) = owner_of.get(ref_id) {
                                if is_extern {
                                    reachable_externs.insert((owner, *ref_id));
                                } else {
                                    worklist.push((owner, *ref_id));
                                }
                            }
                        }
                        IrInstruction::LoadGlobal { global_id, .. }
                        | IrInstruction::StoreGlobal { global_id, .. } => {
                            reachable_globals.insert((mod_idx, *global_id));
                        }
                        _ => {}
                    }
                }
            }
        }
        for &(m, id) in &reachable_externs {
            if linked_externs.insert((m, id)) {
                if let Some(bodies) = modules[m]
                    .extern_functions
                    .get(&id)
                    .and_then(|f| by_bare.get(&f.name))
                {
                    worklist.extend(bodies.iter().copied());
                }
            }
        }
        let ancestors: Vec<u32> = live_class_ids
            .iter()
            .flat_map(|id| std::iter::successors(parent_of.get(id), |p| parent_of.get(p)).take(64))
            .copied()
            .collect();
        live_class_ids.extend(ancestors);
        let by_name = |name: &str| CLASS_BY_NAME.contains(&name);
        let every_class = reachable_externs.iter().any(|(m, id)| {
            modules[*m]
                .extern_functions
                .get(id)
                .is_some_and(|f| by_name(&f.name))
        }) || reachable_functions.iter().any(|(m, id)| {
            modules[*m]
                .functions
                .get(id)
                .is_some_and(|f| f.cfg.blocks.is_empty() && by_name(&f.name))
        });
        let mut progressed = !worklist.is_empty();
        for registration in registrations.iter_mut().filter(|r| !r.kept) {
            let live = match registration.key {
                Key::Class(id) => every_class || live_class_ids.contains(&id),
                Key::Target(target) => owner_of
                    .get(&target)
                    .is_none_or(|&(owner, _)| reachable_functions.contains(&(owner, target))),
            };
            if !live {
                continue;
            }
            registration.kept = true;
            progressed = true;
            reachable_externs.insert((registration.module, registration.extern_id));
            for target in &registration.refs {
                if let Some(&(owner, is_extern)) = owner_of.get(target) {
                    if is_extern {
                        reachable_externs.insert((owner, *target));
                    } else {
                        worklist.push((owner, *target));
                    }
                }
            }
        }
        if !progressed {
            break;
        }
    }

    // Registrations left unkept are removed with the operands only they used.
    let mut dropped: BTreeMap<(usize, IrFunctionId), BTreeSet<(IrBlockId, usize)>> =
        BTreeMap::new();
    for registration in registrations.iter().filter(|r| !r.kept) {
        dropped
            .entry((registration.module, registration.function))
            .or_default()
            .extend(registration.sites.iter().copied());
    }
    stats.registrations_removed = dropped.values().map(|sites| sites.len()).sum();
    for ((mod_idx, func_id), sites) in dropped {
        let Some(function) = modules[mod_idx].functions.get_mut(&func_id) else {
            continue;
        };
        for (block_id, block) in function.cfg.blocks.iter_mut() {
            let mut idx = 0;
            block.instructions.retain(|_| {
                idx += 1;
                !sites.contains(&(*block_id, idx - 1))
            });
        }
    }

    // Phase 4: Strip unreachable code from each module
    let original_module_count = modules.len();

    for (mod_idx, module) in modules.iter_mut().enumerate() {
        let orig_funcs = module.functions.len();
        let orig_externs = module.extern_functions.len();
        let orig_globals = module.globals.len();

        module
            .functions
            .retain(|id, _| reachable_functions.contains(&(mod_idx, *id)));
        module
            .extern_functions
            .retain(|id, _| reachable_externs.contains(&(mod_idx, *id)));
        module
            .globals
            .retain(|id, _| reachable_globals.contains(&(mod_idx, *id)));

        stats.functions_removed += orig_funcs - module.functions.len();
        stats.extern_functions_removed += orig_externs - module.extern_functions.len();
        stats.globals_removed += orig_globals - module.globals.len();
        stats.functions_kept += module.functions.len();
        stats.extern_functions_kept += module.extern_functions.len();
    }

    // Phase 5: Remove empty modules (no functions and no extern functions)
    modules.retain(|m| !m.functions.is_empty() || !m.extern_functions.is_empty());
    stats.modules_removed = original_module_count - modules.len();

    stats
}

/// Find the entry function's (module_index, function_id).
fn find_entry(
    modules: &[IrModule],
    entry_module: &str,
    entry_function: &str,
) -> Option<(usize, IrFunctionId)> {
    for (idx, module) in modules.iter().enumerate() {
        if module.name == entry_module {
            if entry_function == "main" {
                if let Some(entry) = module.entry_function() {
                    return Some((idx, entry.id));
                }
            }
            for (func_id, func) in &module.functions {
                if func.name == entry_function {
                    return Some((idx, *func_id));
                }
            }
        }
    }
    // Fallback: search all modules for function name match
    for (idx, module) in modules.iter().enumerate() {
        for (func_id, func) in &module.functions {
            if func.name == entry_function || func.name.ends_with(&format!("_{}", entry_function)) {
                return Some((idx, *func_id));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::tree_shake_bundle;
    use crate::ir::functions::{IrFunction, IrFunctionId, IrFunctionSignature};
    use crate::ir::modules::IrModule;
    use crate::ir::{CallingConvention, IrType};
    use crate::tast::SymbolId;

    fn make_function(id: u32, symbol: u32, name: &str) -> IrFunction {
        IrFunction::new(
            IrFunctionId(id),
            SymbolId::from_raw(symbol),
            name.to_string(),
            IrFunctionSignature {
                parameters: Vec::new(),
                return_type: IrType::Void,
                calling_convention: CallingConvention::Haxe,
                can_throw: false,
                type_params: Vec::new(),
                uses_sret: false,
            },
        )
    }

    #[test]
    fn entry_module_main_survives_import_renumbering() {
        let mut module = IrModule::new("Main".into(), "unit/./Main.hx".into());
        let mut imported = make_function(100_000, 1, "main");
        imported.qualified_name = Some("unit.Other.main".into());
        let mut entry = make_function(200_000, 2, "main");
        entry.qualified_name = Some("unit.Main.main".into());
        module.add_function(imported);
        module.add_function(entry);
        let mut modules = vec![module];
        tree_shake_bundle(&mut modules, "Main", "main");
        assert!(modules[0].functions.contains_key(&IrFunctionId(200_000)));
        assert!(!modules[0].functions.contains_key(&IrFunctionId(100_000)));
    }

    #[test]
    fn vtable_slots_of_a_class_nothing_names_are_dropped() {
        use crate::ir::instructions::IrInstruction;
        use crate::ir::modules::IrExternFunction;
        use crate::ir::types::IrValue;
        use crate::ir::{IrBlockId, IrId};

        let set_slot = IrFunctionId(10);
        let mut module = IrModule::new("main".into(), "main.hx".into());
        module.add_extern_function(IrExternFunction {
            id: set_slot,
            name: "haxe_vtable_set_slot".into(),
            symbol_id: SymbolId::from_raw(10),
            signature: make_function(10, 10, "x").signature,
            source: "runtime".into(),
        });
        let mut main = make_function(0, 1, "main");
        main.cfg
            .blocks
            .get_mut(&IrBlockId::entry())
            .unwrap()
            .instructions = vec![IrInstruction::Const {
            dest: IrId::new(0),
            value: IrValue::I64(0xABCD),
        }];
        let mut init = make_function(1, 2, "__vtable_init__");
        let mut body = Vec::new();
        for (class, thunk, base) in [(0xABCD, 2, 0), (0x1234, 3, 3)] {
            let reg = |i: u32| IrId::new(base + i);
            body.push(IrInstruction::Const {
                dest: reg(0),
                value: IrValue::I32(class),
            });
            body.push(IrInstruction::Const {
                dest: reg(1),
                value: IrValue::I32(0),
            });
            body.push(IrInstruction::FunctionRef {
                dest: reg(2),
                func_id: IrFunctionId(thunk),
            });
            body.push(IrInstruction::CallDirect {
                dest: None,
                func_id: set_slot,
                args: vec![reg(0), reg(1), reg(2)],
                arg_ownership: Vec::new(),
                type_args: Vec::new(),
                is_tail_call: false,
            });
        }
        init.cfg
            .blocks
            .get_mut(&IrBlockId::entry())
            .unwrap()
            .instructions = body;
        module.add_function(main);
        module.add_function(init);
        module.add_function(make_function(2, 3, "live_thunk"));
        module.add_function(make_function(3, 4, "dead_thunk"));

        let mut modules = vec![module];
        tree_shake_bundle(&mut modules, "main", "main");
        let functions = &modules[0].functions;
        assert!(functions.contains_key(&IrFunctionId(2)));
        assert!(!functions.contains_key(&IrFunctionId(3)));
        let init = &functions[&IrFunctionId(1)].cfg.blocks[&IrBlockId::entry()];
        assert_eq!(init.instructions.len(), 4);
    }

    #[test]
    fn tree_shake_preserves_keep_marked_function() {
        let mut module = IrModule::new("main".to_string(), "test.hx".to_string());

        let main_fn = make_function(0, 1, "main");
        let mut kept_fn = make_function(1, 2, "kept");
        kept_fn
            .attributes
            .custom
            .insert("keep".to_string(), "true".to_string());
        let dead_fn = make_function(2, 3, "dead");

        module.add_function(main_fn);
        module.add_function(kept_fn);
        module.add_function(dead_fn);

        let mut modules = vec![module];
        let _stats = tree_shake_bundle(&mut modules, "main", "main");

        let only_module = &modules[0];
        let has_kept = only_module.functions.values().any(|f| f.name == "kept");
        let has_dead = only_module.functions.values().any(|f| f.name == "dead");
        let has_main = only_module.functions.values().any(|f| f.name == "main");

        assert!(has_main);
        assert!(has_kept);
        assert!(!has_dead);
    }
}
