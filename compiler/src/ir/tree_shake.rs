//! Tree-shaking for .rzb bundles.
//!
//! Removes unreachable functions, extern declarations, and globals from a set
//! of MIR modules. This reduces bundle size by stripping stdlib functions that
//! the user program never calls.

use super::instructions::IrInstruction;
use super::modules::IrModule;
use super::{IrFunctionId, IrGlobalId};
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

/// Statistics from tree-shaking.
#[derive(Debug, Default)]
pub struct TreeShakeStats {
    pub functions_removed: usize,
    pub extern_functions_removed: usize,
    pub globals_removed: usize,
    pub modules_removed: usize,
    pub functions_kept: usize,
    pub extern_functions_kept: usize,
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

    // Phase 3: Walk call graph
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

        // Scan all instructions in this function
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
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
