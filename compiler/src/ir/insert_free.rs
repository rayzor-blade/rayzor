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
//! 1. Find all allocation sources (`Alloc` + `CallDirect` to
//!    malloc/haxe_type_create_{empty_,}instance/rayzor_anon_new)
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
use std::collections::{BTreeMap, BTreeSet};

/// Collected function IDs for allocation/deallocation patterns
struct AllocFuncIds {
    malloc_ids: BTreeSet<IrFunctionId>,
    free_ids: BTreeSet<IrFunctionId>,
    anon_new_ids: BTreeSet<IrFunctionId>,
    anon_drop_ids: BTreeSet<IrFunctionId>,
    /// Functions that take an anon handle as first arg but don't capture it
    anon_safe_ids: BTreeSet<IrFunctionId>,
    /// Anon setters: third arg is the stored value, which escapes via the
    /// receiver. Tracked separately from anon_safe_ids so we can flag the
    /// value-arg as an escape without losing the receiver-is-safe semantics.
    anon_setter_ids: BTreeSet<IrFunctionId>,
    /// Copy-only callees: they READ/COPY their pointer args into their own
    /// storage and retain no reference (e.g. `tensor_fromArray` copies a Haxe
    /// `Array<Float>` into a fresh tensor buffer). Passing a heap pointer to
    /// one is NOT an escape, so a local that flows only into such calls (and
    /// nowhere else escaping) can still be freed. Each entry MUST be verified
    /// to retain nothing.
    copy_only_ids: BTreeSet<IrFunctionId>,
    /// `haxe_array_new` — initialises a `HaxeArray` header (arg0) that owns a
    /// heap data buffer. The buffer is otherwise never freed.
    array_new_ids: BTreeSet<IrFunctionId>,
    /// `haxe_array_free` — releases the header's data buffer.
    array_free_ids: BTreeSet<IrFunctionId>,
    /// Array ops that take the header as arg0 and DO NOT retain it past the
    /// call (scalar get/set/push/pop/length/…). Deliberately EXCLUDES the
    /// retaining/aliasing ops — iterator (holds the array) and the `_ptr`
    /// getters (hand out interior pointers) — which must count as escapes.
    array_safe_ids: BTreeSet<IrFunctionId>,
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
            malloc_ids: BTreeSet::new(),
            free_ids: BTreeSet::new(),
            anon_new_ids: BTreeSet::new(),
            anon_drop_ids: BTreeSet::new(),
            anon_safe_ids: BTreeSet::new(),
            anon_setter_ids: BTreeSet::new(),
            copy_only_ids: BTreeSet::new(),
            array_new_ids: BTreeSet::new(),
            array_free_ids: BTreeSet::new(),
            array_safe_ids: BTreeSet::new(),
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
                    let mut s = BTreeMap::new();
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
        "malloc" | "haxe_type_create_empty_instance" | "haxe_type_create_instance" => {
            ids.malloc_ids.insert(fid);
        }
        "free" => {
            ids.free_ids.insert(fid);
        }
        // Copy-only runtime helpers: verified to copy their pointer arg into
        // fresh storage and retain nothing (rayzor_tensor_from_array loops
        // element-by-element into a freshly alloc'd tensor buffer). A local
        // Haxe array that flows only into one of these does not escape.
        "tensor_fromArray" | "rayzor_tensor_from_array" => {
            ids.copy_only_ids.insert(fid);
        }
        "haxe_array_new" | "array_new" => {
            ids.array_new_ids.insert(fid);
            ids.array_safe_ids.insert(fid);
        }
        "haxe_array_free" | "array_free" => {
            ids.array_free_ids.insert(fid);
        }
        // Non-retaining array ops (receiver = arg0). Conservative: only the
        // scalar get/set/push/pop/length/query ops that neither hand out an
        // interior pointer nor keep the array. Everything else (iterator,
        // *_ptr, map/filter/sort/concat/slice/…) is intentionally omitted so
        // it still counts as an escape.
        _ if is_safe_array_op(name) => {
            ids.array_safe_ids.insert(fid);
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
            // Setters store their VALUE arg inside the receiver — track
            // them so `pointer_escapes` can flag value-as-escape even
            // when the call is otherwise "safe" for the receiver.
            if name == "rayzor_anon_set_field_by_index" || name == "rayzor_anon_set_field_by_name" {
                ids.anon_setter_ids.insert(fid);
            }
        }
        _ => {}
    }
}

/// Non-retaining array ops (see `array_safe_ids`). Matches both the
/// `haxe_array_*` and bare `array_*` spellings the MIR emits. Deliberately
/// excludes iterator (retains the array), the `_ptr` getters (hand out
/// interior pointers), and closure-taking ops (map/filter/sort).
fn is_safe_array_op(name: &str) -> bool {
    let base = name.strip_prefix("haxe_").unwrap_or(name);
    matches!(
        base,
        "array_push"
            | "array_push_f64"
            | "array_push_i32"
            | "array_push_i64"
            | "array_get_f64"
            | "array_get_i32"
            | "array_get_i64"
            | "array_set_f64"
            | "array_set_i64"
            | "array_set_null"
            | "array_length"
            | "array_pop"
            | "array_pop_i64"
            | "array_contains"
            | "array_index_of"
            | "array_last_index_of"
    )
}

/// Insert Free instructions for non-escaping allocations in a single function.
/// Returns the number of Free instructions inserted.
fn insert_free_for_function(function: &mut IrFunction, ids: &AllocFuncIds) -> usize {
    if function.cfg.blocks.is_empty() {
        return 0;
    }

    // Step 1: Find all allocation sources:
    // - malloc + reflective class allocators (`haxe_type_create_*`)
    // - rayzor_anon_new (Arc-backed anonymous objects)
    // NOTE: IrInstruction::Alloc is NOT included here because Alloc creates
    // stack slots (via Cranelift's create_sized_stack_slot), not heap memory.
    // Stack slots are automatically freed when the function returns.
    // Calling libc free() on a stack address causes SIGABRT.
    let mut alloc_ids: Vec<IrId> = Vec::new();
    let mut anon_alloc_ids: BTreeSet<IrId> = BTreeSet::new();
    let mut array_alloc_ids: BTreeSet<IrId> = BTreeSet::new();
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
                // haxe_array_new(header, elem_size) initialises the HaxeArray
                // header passed as arg0; that header owns a heap data buffer
                // released by haxe_array_free. Track the header pointer.
                IrInstruction::CallDirect { func_id, args, .. }
                    if ids.array_new_ids.contains(func_id) =>
                {
                    if let Some(&hdr) = args.first() {
                        if array_alloc_ids.insert(hdr) {
                            alloc_ids.push(hdr);
                        }
                    }
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
    let mut dealloc_ids: BTreeSet<_> = ids.free_ids.union(&ids.anon_drop_ids).cloned().collect();
    dealloc_ids.extend(ids.array_free_ids.iter().cloned());

    for &alloc_id in &alloc_ids {
        let derived = build_derived_set(alloc_id, function);
        let is_anon = anon_alloc_ids.contains(&alloc_id);
        let is_array = array_alloc_ids.contains(&alloc_id);

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

        // Anon and array allocs each whitelist their own safe accessors.
        let empty = BTreeSet::new();
        let safe_ids = if is_array {
            &ids.array_safe_ids
        } else if is_anon {
            &ids.anon_safe_ids
        } else {
            &empty
        };
        if !pointer_escapes(
            alloc_id,
            &derived,
            function,
            safe_ids,
            &ids.anon_setter_ids,
            &ids.copy_only_ids,
        ) {
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
    let derived_sets: BTreeMap<IrId, BTreeSet<IrId>> = allocs_needing_free
        .iter()
        .map(|&id| (id, build_derived_set(id, function)))
        .collect();

    // Pick a single anon_drop function ID for emitting drop calls
    let anon_drop_id = ids.anon_drop_ids.iter().next().cloned();
    // ...and a single haxe_array_free ID for releasing array data buffers.
    let array_free_id = ids.array_free_ids.iter().next().cloned();

    // Step 4: Insert Free/Drop for each non-escaping alloc.
    // For allocs defined in the entry block (which dominates all returns), insert at return blocks.
    // For allocs defined in inner blocks (e.g., loop bodies from inlined constructors),
    // insert at the last-use block to avoid referencing IrIds that don't dominate the return.
    let entry_block = function.entry_block();

    // Build a map: alloc_id → defining block
    let mut alloc_def_block: BTreeMap<IrId, IrBlockId> = BTreeMap::new();
    for (&block_id, block) in &function.cfg.blocks {
        for inst in &block.instructions {
            if let IrInstruction::CallDirect {
                dest,
                func_id,
                args,
                ..
            } = inst
            {
                if let Some(dest) = dest {
                    if ids.malloc_ids.contains(func_id) || ids.anon_new_ids.contains(func_id) {
                        alloc_def_block.insert(*dest, block_id);
                    }
                }
                // Arrays are keyed by the header pointer passed to haxe_array_new.
                if ids.array_new_ids.contains(func_id) {
                    if let Some(&hdr) = args.first() {
                        alloc_def_block.insert(hdr, block_id);
                    }
                }
            }
        }
    }

    // Partition allocs into entry-block vs inner-block
    let mut entry_allocs = Vec::new();
    let mut inner_allocs = Vec::new();
    for &alloc_id in &allocs_needing_free {
        if alloc_def_block.get(&alloc_id) == Some(&entry_block) {
            entry_allocs.push(alloc_id);
        } else {
            inner_allocs.push(alloc_id);
        }
    }

    let mut inserted = 0;

    // Entry-block allocs: free at return blocks (original behavior)
    for block_id in &return_blocks {
        if let Some(block) = function.cfg.blocks.get_mut(block_id) {
            let return_value = if let IrTerminator::Return { value } = &block.terminator {
                *value
            } else {
                None
            };

            for &alloc_id in &entry_allocs {
                let derived = &derived_sets[&alloc_id];
                if let Some(ret_val) = return_value {
                    if ret_val == alloc_id || derived.contains(&ret_val) {
                        continue;
                    }
                }

                if array_alloc_ids.contains(&alloc_id) {
                    // Release the array's heap data buffer via haxe_array_free
                    // (the stack header is auto-reclaimed). NOT a libc Free —
                    // the alloc_id is the header pointer, not a malloc result.
                    if let Some(free_id) = array_free_id {
                        block.instructions.push(IrInstruction::CallDirect {
                            dest: None,
                            func_id: free_id,
                            args: vec![alloc_id],
                            arg_ownership: vec![OwnershipMode::Move],
                            type_args: vec![],
                            is_tail_call: false,
                        });
                        inserted += 1;
                    }
                } else if anon_alloc_ids.contains(&alloc_id) {
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
                    block
                        .instructions
                        .push(IrInstruction::Free { ptr: alloc_id });
                    inserted += 1;
                }
            }
        }
    }

    // Inner-block allocs: skip for now.
    // The "last-use block" approach is unsound for loop-carried allocations:
    // if an alloc is used in a loop body, freeing at the "last use" block frees it
    // after the first iteration, causing use-after-free on subsequent iterations.
    // These allocations are typically eliminated by SRA (promoted to registers).
    // Any remaining inner-block allocs will leak, which is acceptable.

    inserted
}

/// Build the set of all IrIds derived from an allocation pointer.
/// Includes the alloc_id itself plus any GEP, Cast, BitCast, or Copy that uses it.
fn build_derived_set(alloc_id: IrId, function: &IrFunction) -> BTreeSet<IrId> {
    let mut derived = BTreeSet::new();
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
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    safe_call_ids: &BTreeSet<IrFunctionId>,
    anon_setter_ids: &BTreeSet<IrFunctionId>,
    copy_only_ids: &BTreeSet<IrFunctionId>,
) -> bool {
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                // Pointer passed as function argument → escapes
                // (unless the call target is known-safe, e.g. rayzor_anon_* accessors)
                IrInstruction::CallDirect { args, func_id, .. } => {
                    if copy_only_ids.contains(func_id) {
                        // Verified copy-only callee: it copies its pointer args
                        // into fresh storage and retains nothing, so passing the
                        // alloc here is not an escape. Other uses of the alloc are
                        // still checked by the remaining match arms.
                    } else if !safe_call_ids.contains(func_id) {
                        for arg in args {
                            if *arg == alloc_id || derived.contains(arg) {
                                return true;
                            }
                        }
                    } else if anon_setter_ids.contains(func_id) {
                        // `rayzor_anon_set_field_by_index(receiver, idx, value)`
                        // is safe for the receiver (arg 0) and index (arg 1),
                        // but the VALUE arg (arg 2) gets stored inside the
                        // receiver's slot table — i.e. the value escapes via
                        // the receiver. If the receiver itself later escapes
                        // (return / store / etc.), the value transitively
                        // escapes too.
                        //
                        // For correctness we conservatively treat the value
                        // arg as an escape source. Without this, returning
                        // an outer anon that stores an inner anon
                        // (`{ inner: i }`) frees `i` at function exit and
                        // the caller reads a dangling pointer → SIGSEGV on
                        // first nested field access.
                        if args.len() >= 3 {
                            let value_arg = args[2];
                            if value_arg == alloc_id || derived.contains(&value_arg) {
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
