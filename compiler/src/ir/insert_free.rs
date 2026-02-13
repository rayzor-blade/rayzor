//! Insert Free Pass — adds Free instructions for non-escaping heap allocations.
//!
//! This pass runs at the MIR level to ensure heap allocations that don't escape
//! the function are properly freed. It handles both `Alloc` instructions and
//! malloc call results (e.g., from inlined constructors), as well as Arc-backed
//! anonymous objects (`rayzor_anon_new` → `rayzor_anon_drop`).
//!
//! ## Algorithm
//!
//! For each function:
//! 1. Find all allocation sources (`Alloc` + `CallDirect` to malloc/rayzor_anon_new)
//! 2. Track derived pointers (GEP, Cast, Copy of alloc result)
//! 3. Check escape conditions:
//!    - Pointer returned from function → escapes
//!    - Pointer passed as argument to a function call → escapes
//!      (except for known-safe anon object accessors like rayzor_anon_set_field_by_index)
//!    - Pointer stored as a value (not as a store target) → escapes
//!    - Pointer placed into a struct (CreateStruct) → escapes
//!    - Pointer stored to global or used in memcpy → escapes
//!    - Pointer used in phi node → escapes (conservative; SRA handles these)
//! 4. For non-escaping allocations that have no existing Free, insert Free
//!    (or rayzor_anon_drop for Arc objects) before each return instruction

use super::blocks::{IrBlockId, IrTerminator};
use super::functions::IrFunctionId;
use super::instructions::{IrInstruction, OwnershipMode};
use super::optimization::{OptimizationPass, OptimizationResult};
use super::{IrFunction, IrId, IrModule, IrType};
use std::collections::{HashMap, HashSet};

/// Collected function IDs for allocation/deallocation patterns
struct AllocFuncIds {
    malloc_ids: HashSet<IrFunctionId>,
    free_ids: HashSet<IrFunctionId>,
    anon_new_ids: HashSet<IrFunctionId>,
    anon_drop_ids: HashSet<IrFunctionId>,
    /// Functions that take an anon handle as first arg but don't capture it
    anon_safe_ids: HashSet<IrFunctionId>,
}

pub struct InsertFreePass;

impl InsertFreePass {
    pub fn new() -> Self {
        InsertFreePass
    }
}

impl OptimizationPass for InsertFreePass {
    fn name(&self) -> &'static str {
        "InsertFree"
    }

    fn run_on_module(&mut self, module: &mut IrModule) -> OptimizationResult {
        let mut total_inserted = 0;

        // Identify malloc, free, and anon object function IDs
        let mut ids = AllocFuncIds {
            malloc_ids: HashSet::new(),
            free_ids: HashSet::new(),
            anon_new_ids: HashSet::new(),
            anon_drop_ids: HashSet::new(),
            anon_safe_ids: HashSet::new(),
        };

        // Scan both local and extern functions for known names
        for (&fid, func) in &module.functions {
            classify_func(fid, &func.name, &mut ids);
        }
        for (&fid, func) in &module.extern_functions {
            classify_func(fid, &func.name, &mut ids);
        }

        // If rayzor_anon_new exists but rayzor_anon_drop doesn't, declare it as extern
        if !ids.anon_new_ids.is_empty() && ids.anon_drop_ids.is_empty() {
            let drop_id = module.alloc_function_id();
            module.extern_functions.insert(
                drop_id,
                super::modules::IrExternFunction {
                    id: drop_id,
                    name: "rayzor_anon_drop".to_string(),
                    symbol_id: crate::tast::SymbolId::from_raw(0),
                    signature: super::IrFunctionSignature {
                        parameters: vec![super::functions::IrParameter {
                            name: "ptr".to_string(),
                            ty: IrType::Ptr(Box::new(IrType::U8)),
                            reg: IrId(0),
                            by_ref: false,
                        }],
                        return_type: IrType::Void,
                        calling_convention: super::CallingConvention::C,
                        can_throw: false,
                        type_params: vec![],
                        uses_sret: false,
                    },
                    source: "runtime".to_string(),
                },
            );
            ids.anon_drop_ids.insert(drop_id);
            ids.anon_safe_ids.insert(drop_id);
        }

        let func_ids: Vec<_> = module.functions.keys().cloned().collect();
        for func_id in func_ids {
            if let Some(function) = module.functions.get_mut(&func_id) {
                total_inserted += insert_free_for_function(function, &ids);
            }
        }

        if total_inserted > 0 {
            OptimizationResult {
                modified: true,
                instructions_eliminated: 0,
                stats: {
                    let mut s = HashMap::new();
                    s.insert("free_instructions_inserted".to_string(), total_inserted);
                    s
                },
                blocks_eliminated: 0,
            }
        } else {
            OptimizationResult::unchanged()
        }
    }
}

/// Classify a function by name into the appropriate ID sets.
fn classify_func(fid: IrFunctionId, name: &str, ids: &mut AllocFuncIds) {
    match name {
        "malloc" => {
            ids.malloc_ids.insert(fid);
        }
        "free" => {
            ids.free_ids.insert(fid);
        }
        "rayzor_anon_new" => {
            ids.anon_new_ids.insert(fid);
            ids.anon_safe_ids.insert(fid);
        }
        "rayzor_anon_drop" => {
            ids.anon_drop_ids.insert(fid);
            ids.anon_safe_ids.insert(fid);
        }
        _ if name.starts_with("rayzor_anon_") || name.starts_with("haxe_reflect_") => {
            ids.anon_safe_ids.insert(fid);
        }
        _ => {}
    }
}

/// Insert Free instructions for non-escaping allocations in a single function.
/// Returns the number of Free instructions inserted.
fn insert_free_for_function(function: &mut IrFunction, ids: &AllocFuncIds) -> usize {
    if function.cfg.blocks.is_empty() {
        return 0;
    }

    // Step 1: Find all allocation sources (malloc + rayzor_anon_new calls)
    // NOTE: IrInstruction::Alloc is NOT included here because Alloc creates
    // stack slots (via Cranelift's create_sized_stack_slot), not heap memory.
    // Stack slots are automatically freed when the function returns.
    // Calling libc free() on a stack address causes SIGABRT.
    let mut alloc_ids: Vec<IrId> = Vec::new();
    let mut anon_alloc_ids: HashSet<IrId> = HashSet::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInstruction::CallDirect {
                    dest: Some(dest),
                    func_id,
                    ..
                } if ids.malloc_ids.contains(func_id) => {
                    alloc_ids.push(*dest);
                }
                IrInstruction::CallDirect {
                    dest: Some(dest),
                    func_id,
                    ..
                } if ids.anon_new_ids.contains(func_id) => {
                    alloc_ids.push(*dest);
                    anon_alloc_ids.insert(*dest);
                }
                _ => {}
            }
        }
    }

    if alloc_ids.is_empty() {
        return 0;
    }

    // Step 2: For each alloc, check escape and collect non-escaping ones
    let mut allocs_needing_free: Vec<IrId> = Vec::new();
    let dealloc_ids: HashSet<_> = ids.free_ids.union(&ids.anon_drop_ids).cloned().collect();

    for &alloc_id in &alloc_ids {
        let derived = build_derived_set(alloc_id, function);
        let is_anon = anon_alloc_ids.contains(&alloc_id);

        // Check if already has a Free (either Free instruction, free() call, or anon_drop call)
        let has_free = function.cfg.blocks.values().any(|block| {
            block.instructions.iter().any(|inst| match inst {
                IrInstruction::Free { ptr } => derived.contains(ptr) || *ptr == alloc_id,
                IrInstruction::CallDirect { func_id, args, .. }
                    if dealloc_ids.contains(func_id) =>
                {
                    args.iter().any(|a| *a == alloc_id || derived.contains(a))
                }
                _ => false,
            })
        });

        if has_free {
            continue;
        }

        // For anon allocs, use modified escape analysis that whitelists safe accessors
        let empty = HashSet::new();
        let safe_ids = if is_anon { &ids.anon_safe_ids } else { &empty };
        if !pointer_escapes(alloc_id, &derived, function, safe_ids) {
            allocs_needing_free.push(alloc_id);
        }
    }

    if allocs_needing_free.is_empty() {
        return 0;
    }

    // Step 3: Find all return blocks
    let return_blocks: Vec<IrBlockId> = function
        .cfg
        .blocks
        .iter()
        .filter(|(_, block)| matches!(block.terminator, IrTerminator::Return { .. }))
        .map(|(id, _)| *id)
        .collect();

    // Pre-compute derived sets
    let derived_sets: HashMap<IrId, HashSet<IrId>> = allocs_needing_free
        .iter()
        .map(|&id| (id, build_derived_set(id, function)))
        .collect();

    // Pick a single anon_drop function ID for emitting drop calls
    let anon_drop_id = ids.anon_drop_ids.iter().next().cloned();

    // Step 4: Insert Free/Drop before each return for each non-escaping alloc
    let mut inserted = 0;
    for block_id in &return_blocks {
        if let Some(block) = function.cfg.blocks.get_mut(block_id) {
            let return_value = if let IrTerminator::Return { value } = &block.terminator {
                *value
            } else {
                None
            };

            for &alloc_id in &allocs_needing_free {
                let derived = &derived_sets[&alloc_id];
                // Don't free if this alloc is being returned
                if let Some(ret_val) = return_value {
                    if ret_val == alloc_id || derived.contains(&ret_val) {
                        continue;
                    }
                }

                if anon_alloc_ids.contains(&alloc_id) {
                    // Arc-backed anon object: emit rayzor_anon_drop(handle) call
                    if let Some(drop_id) = anon_drop_id {
                        block.instructions.push(IrInstruction::CallDirect {
                            dest: None,
                            func_id: drop_id,
                            args: vec![alloc_id],
                            arg_ownership: vec![OwnershipMode::Move],
                            type_args: vec![],
                            is_tail_call: false,
                        });
                        inserted += 1;
                    }
                } else {
                    // Regular malloc allocation: emit Free instruction
                    block
                        .instructions
                        .push(IrInstruction::Free { ptr: alloc_id });
                    inserted += 1;
                }
            }
        }
    }

    inserted
}

/// Build the set of all IrIds derived from an allocation pointer.
/// Includes the alloc_id itself plus any GEP, Cast, BitCast, or Copy that uses it.
fn build_derived_set(alloc_id: IrId, function: &IrFunction) -> HashSet<IrId> {
    let mut derived = HashSet::new();
    derived.insert(alloc_id);

    let mut changed = true;
    while changed {
        changed = false;
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::GetElementPtr { dest, ptr, .. } => {
                        if derived.contains(ptr) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
                    IrInstruction::Cast { dest, src, .. }
                    | IrInstruction::BitCast { dest, src, .. } => {
                        if derived.contains(src) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
                    IrInstruction::Copy { dest, src } => {
                        if derived.contains(src) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    derived
}

/// Check if a pointer (or any of its derived pointers) escapes the function.
/// `safe_call_ids` are function IDs that don't capture the pointer (e.g., anon object accessors).
fn pointer_escapes(
    alloc_id: IrId,
    derived: &HashSet<IrId>,
    function: &IrFunction,
    safe_call_ids: &HashSet<IrFunctionId>,
) -> bool {
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                // Pointer passed as function argument → escapes
                // (unless the call target is known-safe, e.g. rayzor_anon_* accessors)
                IrInstruction::CallDirect { args, func_id, .. } => {
                    if !safe_call_ids.contains(func_id) {
                        for arg in args {
                            if *arg == alloc_id || derived.contains(arg) {
                                return true;
                            }
                        }
                    }
                }
                IrInstruction::CallIndirect { args, func_ptr, .. } => {
                    if *func_ptr == alloc_id || derived.contains(func_ptr) {
                        return true;
                    }
                    for arg in args {
                        if *arg == alloc_id || derived.contains(arg) {
                            return true;
                        }
                    }
                }

                // Pointer stored as a VALUE to memory → escapes
                IrInstruction::Store { value, .. } => {
                    if *value == alloc_id || derived.contains(value) {
                        return true;
                    }
                }

                // Pointer placed into a struct → escapes
                IrInstruction::CreateStruct { fields, .. } => {
                    for field in fields {
                        if *field == alloc_id || derived.contains(field) {
                            return true;
                        }
                    }
                }

                // Pointer stored to global → escapes
                IrInstruction::StoreGlobal { value, .. } => {
                    if *value == alloc_id || derived.contains(value) {
                        return true;
                    }
                }

                // Pointer used in memcpy → escapes
                IrInstruction::MemCopy { dest, src, .. } => {
                    if *dest == alloc_id
                        || derived.contains(dest)
                        || *src == alloc_id
                        || derived.contains(src)
                    {
                        return true;
                    }
                }

                _ => {}
            }
        }

        // Phi nodes — conservative: if alloc flows through phi, treat as escape.
        // SRA/phi-SRA handles these by eliminating the alloc entirely.
        for phi in &block.phi_nodes {
            for (_, val) in &phi.incoming {
                if *val == alloc_id || derived.contains(val) {
                    return true;
                }
            }
        }

        // Pointer returned → escapes
        if let IrTerminator::Return { value: Some(val) } = &block.terminator {
            if *val == alloc_id || derived.contains(val) {
                return true;
            }
        }
    }

    false
}
