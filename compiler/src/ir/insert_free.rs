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
    /// `haxe_array_free` — releases the header's data buffer.
    array_free_ids: BTreeSet<IrFunctionId>,
    /// Extern producers VERIFIED to return a fresh `Box::into_raw` HaxeString
    /// with an owned (or cap=0 static-protected) buffer on EVERY path — safe
    /// to release with `haxe_string_free`. NOT `value_to_string_by_tag`: its
    /// tag-5 arm returns the INPUT pointer (not fresh).
    ext_fresh_string_ids: BTreeSet<IrFunctionId>,
    /// `haxe_string_free` — releases buffer AND Box-reclaims the header, so
    /// string allocs get ONLY this call, never an `IrInstruction::Free`.
    string_free_ids: BTreeSet<IrFunctionId>,
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
            array_free_ids: BTreeSet::new(),
            ext_fresh_string_ids: BTreeSet::new(),
            string_free_ids: BTreeSet::new(),
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

        // Likewise: nue code never calls haxe_array_free, so if we will need it
        // (any array ops are present) declare it as an extern to call.
        if !ids.array_safe_ids.is_empty() && ids.array_free_ids.is_empty() {
            let free_id = module.alloc_function_id();
            module.extern_functions.insert(
                free_id,
                super::modules::IrExternFunction {
                    id: free_id,
                    name: "haxe_array_free".to_string(),
                    symbol_id: crate::tast::SymbolId::from_raw(0),
                    signature: super::IrFunctionSignature {
                        parameters: vec![super::functions::IrParameter {
                            name: "arr".to_string(),
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
            ids.array_free_ids.insert(free_id);
        }

        // Same for haxe_string_free when fresh-string producers are present.
        if !ids.ext_fresh_string_ids.is_empty() && ids.string_free_ids.is_empty() {
            let free_id = module.alloc_function_id();
            module.extern_functions.insert(
                free_id,
                super::modules::IrExternFunction {
                    id: free_id,
                    name: "haxe_string_free".to_string(),
                    symbol_id: crate::tast::SymbolId::from_raw(0),
                    signature: super::IrFunctionSignature {
                        parameters: vec![super::functions::IrParameter {
                            name: "s".to_string(),
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
            ids.string_free_ids.insert(free_id);
        }

        // Per-parameter retention for module functions — lets a pointer flow
        // into a non-retaining wrapper (`lookup(key)` → stringmap_get) without
        // counting as an escape. Opt-in (RZT_PARAM_RETENTION=1) while the
        // remaining soundness hole is bisected: with it on, strings stored
        // into result structs get freed (tok 4/32).
        let param_retention = if std::env::var("RZT_PARAM_RETENTION").as_deref() == Ok("1") {
            let r = compute_param_retention(module, &ids);
            if std::env::var_os("RZT_DBG_RETENTION").is_some() {
                for (fid, mask) in &r {
                    if mask.iter().any(|m| !m) {
                        if let Some(f) = module.functions.get(fid) {
                            let clear: Vec<usize> = mask
                                .iter()
                                .enumerate()
                                .filter(|(_, m)| !**m)
                                .map(|(i, _)| i)
                                .collect();
                            eprintln!("[retain-clear] {} {:?}", f.name, clear);
                        }
                    }
                }
            }
            r
        } else {
            BTreeMap::new()
        };

        // Functions whose return is a fresh caller-owned array — their call
        // dests are owned allocations in the caller.
        let fresh_fns = compute_returns_fresh_arrays(module, &ids, &param_retention);
        // ...and likewise for fresh caller-owned strings (chains through
        // StringBuf.toString → encodeUtf8-style wrappers).
        let fresh_string_fns = compute_returns_fresh_strings(module, &ids, &param_retention);

        let func_ids: Vec<_> = module.functions.keys().cloned().collect();
        for func_id in func_ids {
            if let Some(function) = module.functions.get_mut(&func_id) {
                total_inserted += insert_free_for_function(
                    function,
                    &ids,
                    &fresh_fns,
                    &fresh_string_fns,
                    &param_retention,
                );
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
        "haxe_array_free" | "array_free" => {
            ids.array_free_ids.insert(fid);
        }
        "haxe_string_free" => {
            ids.string_free_ids.insert(fid);
        }
        // Verified Box-fresh string producers; their string INPUTS are
        // read-and-copied (never retained), so they are also safe consumers.
        "haxe_string_lower"
        | "haxe_string_char_at_ptr"
        | "haxe_string_substr_ptr"
        | "haxe_string_concat"
        | "haxe_string_concat_ptr"
        | "haxe_string_from_char_code"
        | "haxe_string_from_int"
        | "haxe_string_from_float"
        | "haxe_string_from_bool"
        | "haxe_string_from_string"
        | "haxe_bytes_to_string" => {
            ids.ext_fresh_string_ids.insert(fid);
            ids.copy_only_ids.insert(fid);
        }
        // Read-only string/map consumers: read every pointer arg, retain none.
        // NOT stringmap_set (stores its key) and NOT haxe_box_haxestring_ptr
        // (boxing retains).
        "haxe_string_length"
        | "haxe_string_char_code_at_ptr"
        | "haxe_string_compare"
        | "haxe_string_index_of_ptr"
        | "haxe_string_last_index_of_ptr"
        | "haxe_string_starts_with"
        | "haxe_string_print"
        | "haxe_string_println"
        | "haxe_stringmap_get"
        | "haxe_stringmap_exists" => {
            ids.copy_only_ids.insert(fid);
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
fn insert_free_for_function(
    function: &mut IrFunction,
    ids: &AllocFuncIds,
    fresh_fns: &BTreeSet<IrFunctionId>,
    fresh_string_fns: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> usize {
    if function.cfg.blocks.is_empty() {
        return 0;
    }

    // Step 1: Find all allocation sources:
    // - malloc + reflective class allocators (`haxe_type_create_*`)
    // - rayzor_anon_new (Arc-backed anonymous objects)
    // - dests of calls to returns-fresh-array functions (the caller owns
    //   the received array)
    // NOTE: IrInstruction::Alloc is NOT included here because Alloc creates
    // stack slots (via Cranelift's create_sized_stack_slot), not heap memory.
    // Stack slots are automatically freed when the function returns.
    // Calling libc free() on a stack address causes SIGABRT.
    let mut alloc_ids: Vec<IrId> = Vec::new();
    let mut anon_alloc_ids: BTreeSet<IrId> = BTreeSet::new();
    let mut array_alloc_ids: BTreeSet<IrId> = BTreeSet::new();
    let mut received_array_ids: BTreeSet<IrId> = BTreeSet::new();
    let mut string_alloc_ids: BTreeSet<IrId> = BTreeSet::new();
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
                IrInstruction::CallDirect {
                    dest: Some(dest),
                    func_id,
                    ..
                } if fresh_fns.contains(func_id) => {
                    alloc_ids.push(*dest);
                    received_array_ids.insert(*dest);
                }
                IrInstruction::CallDirect {
                    dest: Some(dest),
                    func_id,
                    ..
                } if ids.ext_fresh_string_ids.contains(func_id)
                    || fresh_string_fns.contains(func_id) =>
                {
                    alloc_ids.push(*dest);
                    string_alloc_ids.insert(*dest);
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
    dealloc_ids.extend(ids.string_free_ids.iter().cloned());

    for &alloc_id in &alloc_ids {
        let derived = build_derived_set(alloc_id, function);
        let is_anon = anon_alloc_ids.contains(&alloc_id);
        let is_string = string_alloc_ids.contains(&alloc_id);
        // A malloc'd HaxeArray header: the alloc (or a derived / loop-invariant
        // alias) is used as arg0 of a non-retaining array op — or the value was
        // RECEIVED from a returns-fresh-array callee (known array by type, even
        // if the caller never touches it). Released via haxe_array_free + Free.
        let is_array = !is_string
            && (received_array_ids.contains(&alloc_id)
                || (!is_anon && is_array_header(&derived, function, &ids.array_safe_ids)));
        if is_array {
            array_alloc_ids.insert(alloc_id);
        }

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

        // Anon and array allocs each whitelist their own safe accessors;
        // string consumers are all in copy_only_ids (checked unconditionally).
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
            param_retention,
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
    // ...and haxe_string_free for fresh strings.
    let string_free_id = ids.string_free_ids.iter().next().cloned();

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
                dest: Some(dest),
                func_id,
                ..
            } = inst
            {
                // Array headers are malloc results, so the malloc arm covers
                // them too. Received fresh arrays/strings are defined by
                // their producing call.
                if ids.malloc_ids.contains(func_id)
                    || ids.anon_new_ids.contains(func_id)
                    || fresh_fns.contains(func_id)
                    || ids.ext_fresh_string_ids.contains(func_id)
                    || fresh_string_fns.contains(func_id)
                {
                    alloc_def_block.insert(*dest, block_id);
                }
            }
        }
    }

    // Loop-carried rotations are released at the latch instead. The same object
    // must not also be released by the return-block or confined-block rules
    // below, or the second release is a double free.
    //
    // These allocs are absent from `allocs_needing_free` by construction: a
    // rotation phi merges two different allocations, which the general phi rule
    // reads as an escape. The rotation analysis re-tests them with that one phi
    // excused, so it takes the full alloc list rather than the filtered set.
    let rotations = find_rotation_releases(function, &alloc_ids, &ids, param_retention);
    let rotation_handled: BTreeSet<IrId> = rotations
        .iter()
        .flat_map(|(_, _, carried)| carried.iter().copied())
        .collect();

    // Partition allocs into entry-block vs inner-block
    let mut entry_allocs = Vec::new();
    let mut inner_allocs = Vec::new();
    for &alloc_id in &allocs_needing_free {
        if rotation_handled.contains(&alloc_id) {
            continue;
        }
        if alloc_def_block.get(&alloc_id) == Some(&entry_block) {
            entry_allocs.push(alloc_id);
        } else {
            inner_allocs.push(alloc_id);
        }
    }

    let mut inserted = 0;
    let dbg_arr = std::env::var_os("RZT_DBG_ARR").is_some();
    let fname = if dbg_arr {
        function.name.clone()
    } else {
        String::new()
    };

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

                if string_alloc_ids.contains(&alloc_id) {
                    // Fresh string: haxe_string_free releases the buffer AND
                    // Box-reclaims the header — no IrInstruction::Free.
                    if dbg_arr {
                        eprintln!("[str-ins] {fname} {alloc_id:?} entry");
                    }
                    if let Some(free_id) = string_free_id {
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
                } else if array_alloc_ids.contains(&alloc_id) {
                    // Owned HaxeArray: first release the heap DATA BUFFER via
                    // haxe_array_free (which reads the header's `ptr`), THEN free
                    // the header itself — a 32-byte malloc. Order matters: the
                    // header must still be valid when haxe_array_free reads it.
                    if dbg_arr {
                        eprintln!(
                            "[arr-ins] {fname} {alloc_id:?} entry recv={}",
                            received_array_ids.contains(&alloc_id)
                        );
                    }
                    if let Some(free_id) = array_free_id {
                        block.instructions.push(IrInstruction::CallDirect {
                            dest: None,
                            func_id: free_id,
                            args: vec![alloc_id],
                            arg_ownership: vec![OwnershipMode::Move],
                            type_args: vec![],
                            is_tail_call: false,
                        });
                        block
                            .instructions
                            .push(IrInstruction::Free { ptr: alloc_id });
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

    // Inner-block ARRAY allocs (own or received): free at the end of the
    // DEFINING block when every use is confined to that block. Such an alloc
    // is fresh on every execution of the block and dead at its terminator, so
    // an end-of-block release pairs alloc+free per iteration (the discarded
    // `embedText(s)` in a bench loop). Loop-carried values are excluded
    // structurally: anything threaded through a phi or read by another block
    // is not confined. Non-array inner allocs keep the historical skip —
    // freeing at "last use" is unsound for loop-carried allocations, and SRA
    // usually promotes them anyway.
    for &alloc_id in &inner_allocs {
        let is_string = string_alloc_ids.contains(&alloc_id);
        if !is_string && !array_alloc_ids.contains(&alloc_id) {
            continue;
        }
        let Some(&def_block) = alloc_def_block.get(&alloc_id) else {
            continue;
        };
        let derived = &derived_sets[&alloc_id];
        if !array_confined_to_block(derived, def_block, function) {
            continue;
        }
        if dbg_arr {
            eprintln!(
                "[{}-ins] {fname} {alloc_id:?} inner-confined recv={}",
                if is_string { "str" } else { "arr" },
                received_array_ids.contains(&alloc_id)
            );
        }
        if is_string {
            if let Some(free_id) = string_free_id {
                if let Some(block) = function.cfg.blocks.get_mut(&def_block) {
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
            }
        } else if let Some(free_id) = array_free_id {
            if let Some(block) = function.cfg.blocks.get_mut(&def_block) {
                block.instructions.push(IrInstruction::CallDirect {
                    dest: None,
                    func_id: free_id,
                    args: vec![alloc_id],
                    arg_ownership: vec![OwnershipMode::Move],
                    type_args: vec![],
                    is_tail_call: false,
                });
                block
                    .instructions
                    .push(IrInstruction::Free { ptr: alloc_id });
                inserted += 1;
            }
        }
    }

    // Loop-carried rotations: release the carried value at the latch.
    if std::env::var_os("RZT_DBG_ROT").is_some() && !rotations.is_empty() {
        for (latch, carried, allocs) in &rotations {
            eprintln!(
                "[rot] {} latch={latch:?} carried={carried:?} incoming={allocs:?}",
                function.name
            );
        }
    }
    for (latch, carried, _) in &rotations {
        if let Some(block) = function.cfg.blocks.get_mut(latch) {
            block
                .instructions
                .push(IrInstruction::Free { ptr: *carried });
            inserted += 1;
        }
    }

    inserted
}

/// Loop-carried rotation: a header phi whose incoming values are all fresh,
/// non-escaping allocations. The body builds a new object each iteration and
/// the phi carries the *previous* one, so by the time the latch runs the
/// carried value is dead — releasing it there pairs one free with each
/// iteration's alloc, holding the loop to a constant footprint. The value that
/// leaves the loop is selected by a later re-entry of the header, never the one
/// released here.
///
/// Conservative on every axis that could expose a released object: a single
/// backedge, a latch that falls straight through to the header, and every use
/// of the carried value confined to the loop body.
///
/// Returns `(latch, phi result, carried allocs)`. The carried allocs must be
/// excluded from the other release rules — releasing an object twice is a
/// double free.
fn find_rotation_releases(
    function: &IrFunction,
    alloc_ids: &[IrId],
    ids: &AllocFuncIds,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> Vec<(IrBlockId, IrId, Vec<IrId>)> {
    let mut out: Vec<(IrBlockId, IrId, Vec<IrId>)> = Vec::new();
    let alloc_set: BTreeSet<IrId> = alloc_ids.iter().copied().collect();
    let dbg = std::env::var_os("RZT_DBG_ROT").is_some();

    for (&header, header_block) in &function.cfg.blocks {
        if header_block.phi_nodes.is_empty() {
            continue;
        }

        // A backedge is a predecessor the header can itself reach.
        let backedges: Vec<IrBlockId> = function
            .cfg
            .blocks
            .iter()
            .filter(|(_, b)| b.successors().contains(&header))
            .map(|(&id, _)| id)
            .filter(|&p| reaches(function, header, p))
            .collect();
        if backedges.len() != 1 {
            continue;
        }
        let latch = backedges[0];

        // A conditional exit at the latch could carry the released value out.
        let Some(latch_block) = function.cfg.blocks.get(&latch) else {
            continue;
        };
        if !matches!(latch_block.terminator, IrTerminator::Branch { .. }) {
            continue;
        }

        let body = loop_body(function, header, latch);

        for phi in &header_block.phi_nodes {
            if phi.incoming.len() < 2 {
                continue;
            }
            // Every incoming must be a fresh allocation that escapes by no
            // route other than this phi.
            if !phi.incoming.iter().all(|(_, v)| alloc_set.contains(v)) {
                if dbg {
                    let which: Vec<bool> = phi
                        .incoming
                        .iter()
                        .map(|(_, v)| alloc_set.contains(v))
                        .collect();
                    eprintln!(
                        "[rot-rej] {} phi={:?} not-all-allocs incoming={:?} is_alloc={:?}",
                        function.name, phi.dest, phi.incoming, which
                    );
                }
                continue;
            }
            if !phi.incoming.iter().any(|(pred, _)| *pred == latch) {
                if dbg {
                    eprintln!(
                        "[rot-rej] {} phi={:?} no-latch-incoming",
                        function.name, phi.dest
                    );
                }
                continue;
            }
            if !uses_confined_to_body(function, phi.dest, &body) {
                if dbg {
                    eprintln!(
                        "[rot-rej] {} phi={:?} uses-escape-body",
                        function.name, phi.dest
                    );
                }
                continue;
            }
            // The pass runs more than once per module; without this the second
            // run appends a second release of the same pointer at the latch.
            let already_released = function.cfg.blocks.values().any(|b| {
                b.instructions
                    .iter()
                    .any(|i| matches!(i, IrInstruction::Free { ptr } if *ptr == phi.dest))
            });
            if already_released {
                continue;
            }
            let all_local = phi.incoming.iter().all(|(_, v)| {
                let derived = build_derived_set(*v, function);
                !pointer_escapes_ex(
                    *v,
                    &derived,
                    function,
                    &ids.array_safe_ids,
                    &ids.anon_setter_ids,
                    &ids.copy_only_ids,
                    param_retention,
                    false,
                    Some(phi.dest),
                )
            });
            if !all_local {
                if dbg {
                    eprintln!(
                        "[rot-rej] {} phi={:?} incoming-escapes",
                        function.name, phi.dest
                    );
                }
                continue;
            }
            out.push((
                latch,
                phi.dest,
                phi.incoming.iter().map(|(_, v)| *v).collect(),
            ));
        }
    }

    out
}

/// Forward reachability over the CFG.
fn reaches(function: &IrFunction, from: IrBlockId, to: IrBlockId) -> bool {
    let mut seen: BTreeSet<IrBlockId> = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(b) = stack.pop() {
        if b == to {
            return true;
        }
        if !seen.insert(b) {
            continue;
        }
        if let Some(block) = function.cfg.blocks.get(&b) {
            stack.extend(block.successors());
        }
    }
    false
}

/// Blocks the header reaches that can in turn reach the latch — the iteration's
/// span, and the only place a carried value may legally be observed.
fn loop_body(function: &IrFunction, header: IrBlockId, latch: IrBlockId) -> BTreeSet<IrBlockId> {
    let mut forward: BTreeSet<IrBlockId> = BTreeSet::new();
    let mut stack = vec![header];
    while let Some(b) = stack.pop() {
        if !forward.insert(b) {
            continue;
        }
        if let Some(block) = function.cfg.blocks.get(&b) {
            stack.extend(block.successors());
        }
    }
    forward
        .into_iter()
        .filter(|&b| b == latch || b == header || reaches(function, b, latch))
        .collect()
}

/// True iff every read of `value` sits inside the loop body — no block outside
/// it reads the carried object, and no terminator carries it out.
fn uses_confined_to_body(function: &IrFunction, value: IrId, body: &BTreeSet<IrBlockId>) -> bool {
    for (&block_id, block) in &function.cfg.blocks {
        let inside = body.contains(&block_id);
        if !inside {
            for inst in &block.instructions {
                if inst.uses().contains(&value) {
                    return false;
                }
            }
            // A phi elsewhere would carry the object out of the loop.
            for phi in &block.phi_nodes {
                if phi.incoming.iter().any(|(_, v)| *v == value) {
                    return false;
                }
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if *v == value {
                return false;
            }
        }
    }
    true
}

/// True iff every use of the alloc (and its derived set) sits in the
/// instructions of `def_block` — no other block reads it, no terminator
/// anywhere consumes it (including `def_block`'s own, so it is dead at the
/// block's end), and no phi carries it. Such a value can be released at the
/// end of its defining block as a per-execution free.
fn array_confined_to_block(
    derived: &BTreeSet<IrId>,
    def_block: IrBlockId,
    function: &IrFunction,
) -> bool {
    for (&bid, block) in &function.cfg.blocks {
        if bid != def_block {
            for inst in &block.instructions {
                for u in inst.uses() {
                    if derived.contains(&u) {
                        return false;
                    }
                }
            }
        }
        let term_uses: Vec<IrId> = match &block.terminator {
            IrTerminator::CondBranch { condition, .. } => vec![*condition],
            IrTerminator::Switch { value, .. } => vec![*value],
            IrTerminator::Return { value: Some(v) } => vec![*v],
            IrTerminator::NoReturn { call } => vec![*call],
            _ => Vec::new(),
        };
        for u in term_uses {
            if derived.contains(&u) {
                return false;
            }
        }
        for phi in &block.phi_nodes {
            for (_, v) in &phi.incoming {
                if derived.contains(v) {
                    return false;
                }
            }
        }
    }
    true
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
                    | IrInstruction::BitCast { dest, src, .. }
                    | IrInstruction::SsaBarrier { dest, src, .. } => {
                        if derived.contains(src) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
                    IrInstruction::Copy { dest, src } => {
                        if derived.contains(src) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
                    // Select over a tracked pointer yields an alias of it.
                    IrInstruction::Select {
                        dest,
                        true_val,
                        false_val,
                        ..
                    } => {
                        if (derived.contains(true_val) || derived.contains(false_val))
                            && derived.insert(*dest)
                        {
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
            // A phi that takes the alloc (or a derived value) on ANY edge is
            // pulled in optimistically. This reaches through mutually-recursive
            // loop phis (`$50 = phi[$44,$56]`, `$56 = phi[$50,$96]`) that an
            // "all incoming" rule would deadlock on. Over-inclusion is safe:
            // `pointer_escapes` separately flags any phi that ALSO merges a
            // value NOT in this set (a real foreign merge) as an escape.
            for phi in &block.phi_nodes {
                if derived.contains(&phi.dest) {
                    continue;
                }
                if phi.incoming.iter().any(|(_, v)| derived.contains(v)) && derived.insert(phi.dest)
                {
                    changed = true;
                }
            }
        }
    }

    derived
}

/// A malloc allocation is an owned `HaxeArray` header if the alloc — or any
/// pointer derived from it, including loop-invariant phi aliases threaded
/// through the fill / consume loops — is passed as arg0 (the receiver) of a
/// non-retaining array op. Borrowed arrays (params, fields) are not malloc
/// results in this function, so they never match.
fn is_array_header(
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    array_safe_ids: &BTreeSet<IrFunctionId>,
) -> bool {
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::CallDirect { func_id, args, .. } = inst {
                if array_safe_ids.contains(func_id) {
                    if let Some(a0) = args.first() {
                        if derived.contains(a0) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Check if a pointer (or any of its derived pointers) escapes the function.
/// `safe_call_ids` are function IDs that don't capture the pointer (e.g., anon object accessors).
#[allow(clippy::too_many_arguments)]
fn pointer_escapes(
    alloc_id: IrId,
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    safe_call_ids: &BTreeSet<IrFunctionId>,
    anon_setter_ids: &BTreeSet<IrFunctionId>,
    copy_only_ids: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    pointer_escapes_ex(
        alloc_id,
        derived,
        function,
        safe_call_ids,
        anon_setter_ids,
        copy_only_ids,
        param_retention,
        false,
        None,
    )
}

/// `ignore_returns` drops the "returned → escapes" arms so the returns-fresh
/// analysis can ask "does this alloc escape by any route OTHER than being
/// handed to the caller?".
///
/// `ignore_phi` names one phi whose merge is not treated as an escape, so the
/// rotation analysis can ask "does this alloc escape by any route OTHER than
/// being carried by this loop?". A rotation phi merges two DIFFERENT fresh
/// allocations (the preheader's and the body's), which the general rule below
/// must reject — the release that owns that phi is what makes it safe.
#[allow(clippy::too_many_arguments)]
fn pointer_escapes_ex(
    alloc_id: IrId,
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    safe_call_ids: &BTreeSet<IrFunctionId>,
    anon_setter_ids: &BTreeSet<IrFunctionId>,
    copy_only_ids: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
    ignore_returns: bool,
    ignore_phi: Option<IrId>,
) -> bool {
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                // Instruction-level return (rare; most returns are terminators)
                IrInstruction::Return { value: Some(v) } => {
                    if !ignore_returns && (*v == alloc_id || derived.contains(v)) {
                        return true;
                    }
                }
                // Pointer passed as function argument → escapes
                // (unless the call target is known-safe, e.g. rayzor_anon_* accessors)
                IrInstruction::CallDirect { args, func_id, .. } => {
                    if copy_only_ids.contains(func_id) {
                        // Verified copy-only callee: it copies its pointer args
                        // into fresh storage and retains nothing, so passing the
                        // alloc here is not an escape. Other uses of the alloc are
                        // still checked by the remaining match arms.
                    } else if let Some(mask) = param_retention.get(func_id) {
                        // Module function with per-parameter retention info:
                        // only a RETAINING position (or an out-of-range arg)
                        // counts as an escape.
                        for (i, arg) in args.iter().enumerate() {
                            if (*arg == alloc_id || derived.contains(arg))
                                && mask.get(i).copied().unwrap_or(true)
                            {
                                return true;
                            }
                        }
                    } else if !safe_call_ids.contains(func_id) {
                        for arg in args {
                            if *arg == alloc_id || derived.contains(arg) {
                                return true;
                            }
                        }
                    } else if safe_call_ids.contains(func_id) && !anon_setter_ids.contains(func_id)
                    {
                        // "Safe" accessors are safe for the RECEIVER (arg0)
                        // only. The pointer appearing in any VALUE position —
                        // e.g. pushed as an element into another array — is
                        // stored inside the receiver and escapes with it.
                        for arg in args.iter().skip(1) {
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
                // Captured into a closure environment, which can outlive this
                // frame — the closure owns it from here.
                IrInstruction::MakeClosure {
                    captured_values, ..
                } => {
                    for v in captured_values {
                        if *v == alloc_id || derived.contains(v) {
                            return true;
                        }
                    }
                }
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

        // Phi nodes: the alloc escapes through a phi only if the phi MERGES it
        // with a value NOT derived from the alloc. A pure loop-invariant phi
        // (every incoming already derived, or the phi's own result on a back
        // edge) is just the alloc threaded through a loop — it's in the derived
        // set and its uses are checked directly, so it is not an escape.
        for phi in &block.phi_nodes {
            if ignore_phi == Some(phi.dest) {
                continue;
            }
            let touches = phi
                .incoming
                .iter()
                .any(|(_, v)| *v == alloc_id || derived.contains(v));
            if !touches {
                continue;
            }
            let all_internal = phi
                .incoming
                .iter()
                .all(|(_, v)| *v == alloc_id || derived.contains(v) || *v == phi.dest);
            if !all_internal {
                return true;
            }
        }

        // Pointer returned → escapes
        if !ignore_returns {
            if let IrTerminator::Return { value: Some(val) } = &block.terminator {
                if *val == alloc_id || derived.contains(val) {
                    return true;
                }
            }
        }
    }

    false
}

/// Compute the set of functions that return a FRESH, caller-owned Haxe array:
/// every value-return is covered by a source — a local malloc used as an array
/// header, or the dest of a call to an already-known fresh function — whose
/// only escape route is the return itself and which is not freed locally.
/// Fixpoint so freshness propagates through forwarding wrappers
/// (`embedText` returning `embed(ids)`'s result).
fn compute_returns_fresh_arrays(
    module: &IrModule,
    ids: &AllocFuncIds,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> BTreeSet<IrFunctionId> {
    let mut fresh: BTreeSet<IrFunctionId> = BTreeSet::new();
    loop {
        let mut changed = false;
        for (&fid, function) in &module.functions {
            if fresh.contains(&fid) {
                continue;
            }
            if returns_fresh_array(function, ids, &fresh, param_retention) {
                fresh.insert(fid);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    fresh
}

/// Per-function, per-parameter: can the callee RETAIN the pointer passed at
/// that position (store it, return it, put it in a struct/global, hand it to
/// an unknown call or a retaining position)? Optimistic fixpoint: parameters
/// start non-retaining and escalate when a retaining use is found; calls into
/// other module functions consult the current map, so wrapper chains
/// (`lookup(key)` → `haxe_stringmap_get(map, key)`) resolve, and cycles
/// without sinks correctly stay non-retaining. `true` = retains.
fn compute_param_retention(
    module: &IrModule,
    ids: &AllocFuncIds,
) -> BTreeMap<IrFunctionId, Vec<bool>> {
    let mut retention: BTreeMap<IrFunctionId, Vec<bool>> = BTreeMap::new();
    // Derived sets are body-static — compute once per parameter.
    let mut param_derived: BTreeMap<IrFunctionId, Vec<BTreeSet<IrId>>> = BTreeMap::new();
    for (&fid, f) in &module.functions {
        retention.insert(fid, vec![false; f.signature.parameters.len()]);
        let sets = f
            .signature
            .parameters
            .iter()
            .map(|p| build_derived_set(p.reg, f))
            .collect();
        param_derived.insert(fid, sets);
    }
    loop {
        let mut changed = false;
        for (&fid, function) in &module.functions {
            for pi in 0..function.signature.parameters.len() {
                if retention[&fid][pi] {
                    continue;
                }
                let derived = &param_derived[&fid][pi];
                if param_retained(derived, function, ids, &retention) {
                    retention.get_mut(&fid).unwrap()[pi] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    retention
}

fn param_retained(
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let in_set = |v: &IrId| derived.contains(v);
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInstruction::Store { value, .. } if in_set(value) => return true,
                IrInstruction::StoreGlobal { value, .. } if in_set(value) => return true,
                IrInstruction::CreateStruct { fields, .. } if fields.iter().any(in_set) => {
                    return true;
                }
                IrInstruction::MemCopy { dest, src, .. } if in_set(dest) || in_set(src) => {
                    return true;
                }
                IrInstruction::Throw { exception } if in_set(exception) => return true,
                // Reading THROUGH the param extracts a field (e.g. an interior
                // buffer pointer) whose flow we don't track — conservative.
                IrInstruction::Load { ptr, .. } if in_set(ptr) => return true,
                IrInstruction::Return { value: Some(v) } if in_set(v) => return true,
                IrInstruction::CallIndirect { func_ptr, args, .. }
                    if in_set(func_ptr) || args.iter().any(in_set) =>
                {
                    return true;
                }
                IrInstruction::CallDirect { func_id, args, .. } => {
                    if ids.copy_only_ids.contains(func_id) {
                        // reads/copies every arg — safe
                    } else if let Some(mask) = retention.get(func_id) {
                        for (i, arg) in args.iter().enumerate() {
                            if in_set(arg) && mask.get(i).copied().unwrap_or(true) {
                                return true;
                            }
                        }
                    } else if ids.array_safe_ids.contains(func_id)
                        || ids.anon_safe_ids.contains(func_id)
                    {
                        // receiver-safe: arg0 fine, value positions retain
                        for arg in args.iter().skip(1) {
                            if in_set(arg) {
                                return true;
                            }
                        }
                    } else if args.iter().any(in_set) {
                        // unknown extern
                        return true;
                    }
                }
                _ => {}
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if in_set(v) {
                return true;
            }
        }
    }
    false
}

/// Fresh-STRING analogue: every value-return is covered by the dest of a call
/// to a verified Box-fresh extern producer or an already-fresh module function,
/// with no other escape and no local free. No usage-identification step —
/// string-ness is known from the producer.
fn compute_returns_fresh_strings(
    module: &IrModule,
    ids: &AllocFuncIds,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> BTreeSet<IrFunctionId> {
    let mut fresh: BTreeSet<IrFunctionId> = BTreeSet::new();
    loop {
        let mut changed = false;
        for (&fid, function) in &module.functions {
            if fresh.contains(&fid) {
                continue;
            }
            if returns_fresh_string(function, ids, &fresh, param_retention) {
                fresh.insert(fid);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    fresh
}

fn returns_fresh_string(
    function: &IrFunction,
    ids: &AllocFuncIds,
    fresh: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let mut ret_vals: Vec<IrId> = Vec::new();
    for block in function.cfg.blocks.values() {
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            ret_vals.push(*v);
        }
        for inst in &block.instructions {
            if let IrInstruction::Return { value: Some(v) } = inst {
                ret_vals.push(*v);
            }
        }
    }
    if ret_vals.is_empty() {
        return false;
    }

    let mut sources: Vec<IrId> = Vec::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                ..
            } = inst
            {
                if ids.ext_fresh_string_ids.contains(func_id) || fresh.contains(func_id) {
                    sources.push(*d);
                }
            }
        }
    }
    if sources.is_empty() {
        return false;
    }

    let mut dealloc_ids: BTreeSet<_> = ids.free_ids.union(&ids.anon_drop_ids).cloned().collect();
    dealloc_ids.extend(ids.array_free_ids.iter().cloned());
    dealloc_ids.extend(ids.string_free_ids.iter().cloned());

    let empty = BTreeSet::new();
    let mut derived_memo: BTreeMap<IrId, BTreeSet<IrId>> = BTreeMap::new();
    'ret: for v in &ret_vals {
        for &src in &sources {
            let derived = derived_memo
                .entry(src)
                .or_insert_with(|| build_derived_set(src, function));
            if !derived.contains(v) {
                continue;
            }
            let derived = &*derived;
            if pointer_escapes_ex(
                src,
                derived,
                function,
                &empty,
                &ids.anon_setter_ids,
                &ids.copy_only_ids,
                param_retention,
                true,
                None,
            ) {
                continue;
            }
            let freed_locally = function.cfg.blocks.values().any(|block| {
                block.instructions.iter().any(|inst| match inst {
                    IrInstruction::Free { ptr } => derived.contains(ptr),
                    IrInstruction::CallDirect { func_id, args, .. }
                        if dealloc_ids.contains(func_id) =>
                    {
                        args.iter().any(|a| derived.contains(a))
                    }
                    _ => false,
                })
            });
            if freed_locally {
                continue;
            }
            continue 'ret;
        }
        return false;
    }
    true
}

fn returns_fresh_array(
    function: &IrFunction,
    ids: &AllocFuncIds,
    fresh: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let mut ret_vals: Vec<IrId> = Vec::new();
    for block in function.cfg.blocks.values() {
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            ret_vals.push(*v);
        }
        for inst in &block.instructions {
            if let IrInstruction::Return { value: Some(v) } = inst {
                ret_vals.push(*v);
            }
        }
    }
    if ret_vals.is_empty() {
        return false;
    }

    // (source id, is-a-fresh-call-dest)
    let mut sources: Vec<(IrId, bool)> = Vec::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                ..
            } = inst
            {
                if ids.malloc_ids.contains(func_id) {
                    sources.push((*d, false));
                } else if fresh.contains(func_id) {
                    sources.push((*d, true));
                }
            }
        }
    }
    if sources.is_empty() {
        return false;
    }

    let mut dealloc_ids: BTreeSet<_> = ids.free_ids.union(&ids.anon_drop_ids).cloned().collect();
    dealloc_ids.extend(ids.array_free_ids.iter().cloned());

    let mut derived_memo: BTreeMap<IrId, BTreeSet<IrId>> = BTreeMap::new();
    'ret: for v in &ret_vals {
        for &(src, is_fresh_call) in &sources {
            let derived = derived_memo
                .entry(src)
                .or_insert_with(|| build_derived_set(src, function));
            if !derived.contains(v) {
                continue;
            }
            let derived = &*derived;
            if !is_fresh_call && !is_array_header(&derived, function, &ids.array_safe_ids) {
                continue;
            }
            if pointer_escapes_ex(
                src,
                &derived,
                function,
                &ids.array_safe_ids,
                &ids.anon_setter_ids,
                &ids.copy_only_ids,
                param_retention,
                true,
                None,
            ) {
                continue;
            }
            // Locally freed then returned would dangle — reject the source.
            let freed_locally = function.cfg.blocks.values().any(|block| {
                block.instructions.iter().any(|inst| match inst {
                    IrInstruction::Free { ptr } => derived.contains(ptr),
                    IrInstruction::CallDirect { func_id, args, .. }
                        if dealloc_ids.contains(func_id) =>
                    {
                        args.iter().any(|a| derived.contains(a))
                    }
                    _ => false,
                })
            });
            if freed_locally {
                continue;
            }
            continue 'ret;
        }
        return false;
    }
    true
}
