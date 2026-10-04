//! Order in which a merged module's per-file `__init__`s run.
//!
//! Each imported file's functions and globals are renumbered into their own
//! `IMPORT_STRIDE` id range, so a user function's id names the file it came
//! from (stdlib MIR wrappers are appended past the last range). A file's
//! `__init__` runs after every file whose functions or statics it reaches
//! through direct calls, as Haxe orders class initialisation. Closures are not
//! followed: they run later. Cycles keep the original (compile) order.

use super::{FunctionKind, IrFunctionId, IrInstruction, IrModule};
use std::collections::{BTreeMap, BTreeSet};

const IMPORT_BASE: u32 = 100_000;
const IMPORT_STRIDE: u32 = 10_000;

fn file_of(id: u32) -> u32 {
    if id < IMPORT_BASE {
        u32::MAX
    } else {
        id / IMPORT_STRIDE
    }
}

/// Every function named `name` in `module`, dependencies first.
pub fn ordered_inits(module: &IrModule, name: &str) -> Vec<IrFunctionId> {
    let inits: Vec<IrFunctionId> = module
        .functions
        .values()
        .filter(|f| f.name == name)
        .map(|f| f.id)
        .collect();
    if inits.len() < 2 || name != "__init__" {
        return inits;
    }

    let init_files: BTreeSet<u32> = inits.iter().map(|id| file_of(id.0)).collect();
    let mut deps: BTreeMap<IrFunctionId, BTreeSet<u32>> = BTreeMap::new();
    for &init in &inits {
        let own = file_of(init.0);
        let mut reached = BTreeSet::new();
        let mut seen = BTreeSet::from([init]);
        let mut stack = vec![init];
        while let Some(id) = stack.pop() {
            let Some(func) = module.functions.get(&id) else {
                continue;
            };
            for block in func.cfg.blocks.values() {
                for instr in &block.instructions {
                    match instr {
                        IrInstruction::CallDirect { func_id, .. }
                            if module
                                .functions
                                .get(func_id)
                                .is_some_and(|f| f.kind == FunctionKind::UserDefined) =>
                        {
                            reached.insert(file_of(func_id.0));
                            if seen.insert(*func_id) {
                                stack.push(*func_id);
                            }
                        }
                        IrInstruction::LoadGlobal { global_id, .. }
                        | IrInstruction::StoreGlobal { global_id, .. } => {
                            reached.insert(file_of(global_id.0));
                        }
                        _ => {}
                    }
                }
            }
        }
        reached.retain(|f| *f != own && init_files.contains(f));
        deps.insert(init, reached);
    }

    // Kahn's algorithm, always taking the earliest ready init; on a cycle the
    // earliest remaining init goes next.
    let mut order = Vec::with_capacity(inits.len());
    let mut done: BTreeSet<u32> = BTreeSet::new();
    let mut remaining = inits.clone();
    while !remaining.is_empty() {
        let pick = remaining
            .iter()
            .position(|id| {
                deps[id]
                    .iter()
                    .all(|f| done.contains(f) || !remaining.iter().any(|r| file_of(r.0) == *f))
            })
            .unwrap_or(0);
        let id = remaining.remove(pick);
        if !remaining.iter().any(|r| file_of(r.0) == file_of(id.0)) {
            done.insert(file_of(id.0));
        }
        order.push(id);
    }
    order
}
