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
use super::{IrFunction, IrId, IrModule, IrType, IrValue};
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
    /// `haxe_object_free_deep` — releases an object and, through the owned-field
    /// mask its class registered, whatever it owns beneath it.
    free_deep_ids: BTreeSet<IrFunctionId>,
    /// Array ops that take the header as arg0 and DO NOT retain it past the
    /// call (scalar get/set/push/pop/length/…). Deliberately EXCLUDES the
    /// retaining/aliasing ops — iterator (holds the array) and the `_ptr`
    /// getters (hand out interior pointers) — which must count as escapes.
    array_safe_ids: BTreeSet<IrFunctionId>,
    /// Callees whose RESULT cannot alias any argument: it is a scalar, or a
    /// verified-fresh box. Retention says where an argument ENDS UP; this says
    /// whether the caller gets a second name for it back. `haxe_stringmap_get`
    /// retains nothing yet hands back a value read out of the map, so the two
    /// properties must be tracked apart. Default-deny: a callee absent from
    /// this set and from the module is assumed to return an alias.
    nonaliasing_result_ids: BTreeSet<IrFunctionId>,
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
        let _flush_free_graph = FreeGraphFlush;

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
            free_deep_ids: BTreeSet::new(),
            array_safe_ids: BTreeSet::new(),
            nonaliasing_result_ids: BTreeSet::new(),
        };

        // Only a DECLARATION is the runtime intrinsic. Haxe method names are
        // lowered bare, so a class with a `malloc`, `free` or `realloc` method
        // shares the name -- and classifying that bodied method as the
        // allocator registers its ordinary return value as a heap allocation,
        // which the pass then releases. `Pool.malloc(21)` returns 21 * 2, so
        // the release is `free((void*)42)`.
        for (&fid, func) in &module.functions {
            if func.cfg.blocks.is_empty() {
                classify_func(fid, &func.name, &mut ids);
            }
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

        // A class instance is released through the deep form, which frees what
        // the object owns before the object itself. Nothing in Haxe source
        // names it, so declare it here.
        if ids.free_deep_ids.is_empty() {
            let deep_id = module.alloc_function_id();
            module.extern_functions.insert(
                deep_id,
                super::modules::IrExternFunction {
                    id: deep_id,
                    name: "haxe_object_free_deep".to_string(),
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
            ids.free_deep_ids.insert(deep_id);
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
        // Retention KINDS feed only the ownership-transfer analysis, so unlike
        // the opt-in bool masks above they are computed unconditionally: the
        // worst they can do is transfer nothing.
        let retention_kinds = compute_retention_kinds(module, &ids);

        // Objects. A builder that hands back a freshly made instance makes its
        // call dest an owned allocation in the caller, and the owned-field
        // census says what dies with that instance. Together they reclaim a
        // structure whose interior nothing else names; apart, neither does
        // anything -- the census has nothing to hang off if the root is never
        // freed, and freeing the root alone reclaims one node of a tree.
        let fresh_object_fns = compute_returns_fresh_objects(module, &ids, &param_retention);
        let owned = compute_owned_fields(
            module,
            &ids,
            &fresh_object_fns,
            &retention_kinds.returns_alias,
        );
        register_owned_masks(module, &owned);

        let func_ids: Vec<_> = module.functions.keys().cloned().collect();
        for func_id in func_ids {
            if let Some(function) = module.functions.get_mut(&func_id) {
                total_inserted += insert_free_for_function(
                    function,
                    &ids,
                    &fresh_fns,
                    &fresh_string_fns,
                    &param_retention,
                    &retention_kinds,
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
        "haxe_object_free_deep" => {
            ids.free_deep_ids.insert(fid);
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
            ids.nonaliasing_result_ids.insert(fid);
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
            // `haxe_stringmap_get` is the one exception: it retains nothing,
            // yet hands back the stored value itself.
            if name != "haxe_stringmap_get" {
                ids.nonaliasing_result_ids.insert(fid);
            }
        }
        // Non-retaining array ops (receiver = arg0). Conservative: only the
        // scalar get/set/push/pop/length/query ops that neither hand out an
        // interior pointer nor keep the array. Everything else (iterator,
        // *_ptr, map/filter/sort/concat/slice/…) is intentionally omitted so
        // it still counts as an escape.
        _ if is_safe_array_op(name) => {
            ids.array_safe_ids.insert(fid);
            // Most of these return a scalar, but the 64-bit element getters
            // hand back a stored element -- on an array of objects that is a
            // pointer punned into the integer.
            let base = name.strip_prefix("haxe_").unwrap_or(name);
            if !matches!(base, "array_pop" | "array_pop_i64" | "array_get_i64") {
                ids.nonaliasing_result_ids.insert(fid);
            }
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
/// Writes the recorded graph when the pass finishes a module, including on the
/// early returns, so a failure later in the build still leaves a readable file.
struct FreeGraphFlush;

impl Drop for FreeGraphFlush {
    fn drop(&mut self) {
        crate::ir::free_graph::flush();
    }
}

/// Whether this value is already released anywhere in the function.
///
/// A release takes five shapes — a bare `Free` for a plain allocation, and a
/// call for a string, an array buffer, an anonymous object, or a class
/// instance released through the deep form. Recognising only the bare form
/// makes the other four invisible.
///
/// That matters within a SINGLE run of the pass, not across runs: the set of
/// values still needing a release is computed once, before any release is
/// inserted, and is not pruned as they are added. Three rules place releases —
/// on a path leaving the function, at the end of the defining block, and at a
/// loop latch — so a value matching two of them is released twice unless each
/// rule can see what the others already emitted. (Across runs the value is
/// already excluded, because the needing-release scan does recognise a call.)
///
/// Releasing a string twice frees its header twice: `haxe_string_free` poisons
/// the header so the byte buffer cannot go twice, but reclaims the header on
/// every call — so the second call frees a block that is already gone, and
/// reads it on the way.
fn already_released(
    function: &IrFunction,
    value: IrId,
    string_free_id: Option<IrFunctionId>,
    array_free_id: Option<IrFunctionId>,
    anon_drop_id: Option<IrFunctionId>,
    free_deep_id: Option<IrFunctionId>,
) -> bool {
    function.cfg.blocks.values().any(|b| {
        b.instructions.iter().any(|i| match i {
            IrInstruction::Free { ptr } => *ptr == value,
            IrInstruction::CallDirect {
                func_id,
                args,
                dest: None,
                ..
            } => {
                args.first() == Some(&value)
                    && (Some(*func_id) == string_free_id
                        || Some(*func_id) == array_free_id
                        || Some(*func_id) == anon_drop_id
                        || Some(*func_id) == free_deep_id)
            }
            _ => false,
        })
    })
}

fn insert_free_for_function(
    function: &mut IrFunction,
    ids: &AllocFuncIds,
    fresh_fns: &BTreeSet<IrFunctionId>,
    fresh_string_fns: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
    retention_kinds: &RetentionInfo,
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
    let mut escaped_candidates: Vec<IrId> = Vec::new();
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
        } else {
            escaped_candidates.push(alloc_id);
        }
    }

    // Ownership transfer and benign borrows. An allocation whose ONLY escape
    // is into another local allocation dies with that owner (the wrapper's
    // field, via its constructor). One whose every exposure is a borrowing
    // callee that publishes none of its children never really escaped at all
    // -- the conservative walk counts any call argument as an escape -- and
    // rejoins the ordinary freeing machinery under its own liveness.
    let mut adopted: BTreeMap<IrId, Vec<IrId>> = BTreeMap::new();
    if !escaped_candidates.is_empty() {
        // Two sets, and the difference between them is the whole safety
        // argument. `all_may` follows call results, because the question "does
        // anything else name this memory" must be complete before this stage
        // overrules the conservative walk. `all_must` does not, because
        // attributing a pointer to ONE owning allocation is a claim about
        // provenance, and a call result's provenance is unknown -- reading a
        // may-alias as proof of ownership adopts a value onto an owner it does
        // not belong to, and frees it there.
        let ra = &retention_kinds.returns_alias;
        let all_may: BTreeMap<IrId, BTreeSet<IrId>> = alloc_ids
            .iter()
            .map(|&a| {
                (
                    a,
                    alias_closure(&BTreeSet::from([a]), function, ids, ra, false),
                )
            })
            .collect();
        let all_must: BTreeMap<IrId, BTreeSet<IrId>> = alloc_ids
            .iter()
            .map(|&a| (a, build_derived_set(a, function)))
            .collect();
        let dbg_adopt = std::env::var_os("RZT_DBG_ADOPT").is_some();
        for x in &escaped_candidates {
            let x_derived = &all_may[x];
            match classify_escape_owner(
                *x,
                x_derived,
                function,
                ids,
                retention_kinds,
                &all_may,
                &all_must,
            ) {
                EscapeClass::Escapes => {}
                EscapeClass::Benign => {
                    if std::env::var("RZT_BENIGN_FREE").as_deref() != Ok("1") {
                        // Gated off: see the A/B in the commit message.
                    } else if child_confined_ok(*x, x_derived, function) {
                        if dbg_adopt {
                            eprintln!("[adopt] {} {x:?} benign borrow, freeing", function.name);
                        }
                        allocs_needing_free.push(*x);
                    } else if dbg_adopt {
                        eprintln!(
                            "[adopt] {} {x:?} benign but NOT confined, leaving",
                            function.name
                        );
                    }
                }
                EscapeClass::Owner(owner) => {
                    if !child_confined_ok(*x, x_derived, function) {
                        if dbg_adopt {
                            eprintln!(
                                "[adopt] {} {x:?} -> {owner:?} REFUSED (not confined)",
                                function.name
                            );
                        }
                        continue;
                    }
                    // A value read OUT of the owner may alias the child; if
                    // any such read escapes the reads-only discipline,
                    // freeing the child with the owner would dangle it.
                    if !owner_loads_confined(&all_may[&owner], function, ids, ra) {
                        if dbg_adopt {
                            eprintln!(
                                "[adopt] {} {x:?} -> {owner:?} REFUSED (owner reads escape)",
                                function.name
                            );
                        }
                        continue;
                    }
                    if dbg_adopt {
                        eprintln!("[adopt] {} {x:?} dies with {owner:?}", function.name);
                    }
                    adopted.entry(owner).or_default().push(*x);
                }
            }
        }
    }

    if allocs_needing_free.is_empty() && adopted.is_empty() {
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
    let free_deep_id = ids.free_deep_ids.iter().next().cloned();

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

    // Inner-block allocs (own or received): free at the end of the DEFINING
    // block when every use is confined to that block. Such an alloc is fresh on
    // every execution of the block and dead at its terminator, so an end-of-block
    // release pairs alloc+free per iteration (the discarded `embedText(s)` in a
    // bench loop). Loop-carried values are excluded structurally: anything
    // threaded through a phi or read by another block is not confined, which is
    // what makes releasing here safe — the old "free at last use" hazard only
    // arises for values that survive the iteration.
    //
    // Class instances are included. The comment here used to say SRA promotes
    // them anyway; measured on the mandelbrot inner loop, it does not — the
    // per-iteration temporary (`complexSquare(val)` feeding `complexAdd`) is
    // block-confined, non-escaping, and was never reclaimed.
    for &alloc_id in &inner_allocs {
        let is_string = string_alloc_ids.contains(&alloc_id);
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
        } else if array_alloc_ids.contains(&alloc_id) {
            // An array owns a separate data buffer, so the header release has
            // to go through haxe_array_free first.
            let Some(free_id) = array_free_id else {
                continue;
            };
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
        } else if let Some(block) = function.cfg.blocks.get_mut(&def_block) {
            // A class instance is a single allocation: a plain release, never
            // the array routine, which would also free a buffer it does not own.
            block
                .instructions
                .push(IrInstruction::Free { ptr: alloc_id });
            inserted += 1;
        }
    }

    // Per-iteration temporaries: release at the latch of the loop they live in.
    let temp_handled: BTreeSet<IrId> = rotations
        .iter()
        .flat_map(|(_, _, carried)| carried.iter().copied())
        .collect();
    let needing_free_set: BTreeSet<IrId> = allocs_needing_free.iter().copied().collect();
    for (latch, alloc_id) in
        find_iteration_temporaries(function, &needing_free_set, &temp_handled, &derived_sets)
    {
        if already_released(
            function,
            alloc_id,
            string_free_id,
            array_free_id,
            anon_drop_id,
            free_deep_id,
        ) {
            continue;
        }
        let Some(block) = function.cfg.blocks.get_mut(&latch) else {
            continue;
        };
        // Release it the way its allocator requires. A per-iteration temporary
        // is no different from one released at a return: a string's header is
        // reclaimed by `haxe_string_free`, an array's buffer by
        // `haxe_array_free` before its header, and only a plain allocation is a
        // bare `Free`. Freeing a string header with `Free` aborts the process —
        // the header is not a libc block, and the string's buffer leaks.
        if string_alloc_ids.contains(&alloc_id) {
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

    // Emit transferred children beside their owner's release, wherever that
    // release actually landed (return path, latch, inner-confined). Keying on
    // the emitted instructions means a child is freed exactly when its owner
    // is -- an owner whose placement failed frees nothing and leaks both,
    // which is the status quo, never a double free.
    if !adopted.is_empty() {
        // Transitive closure: freeing a releases b releases c.
        fn expand(
            owner: IrId,
            adopted: &BTreeMap<IrId, Vec<IrId>>,
            out: &mut Vec<IrId>,
            seen: &mut BTreeSet<IrId>,
        ) {
            if let Some(children) = adopted.get(&owner) {
                for &c in children {
                    if seen.insert(c) {
                        out.push(c);
                        expand(c, adopted, out, seen);
                    }
                }
            }
        }
        let owners: Vec<IrId> = adopted.keys().copied().collect();
        let mut closure: BTreeMap<IrId, Vec<IrId>> = BTreeMap::new();
        for &o in &owners {
            let mut out = Vec::new();
            let mut seen = BTreeSet::new();
            seen.insert(o);
            expand(o, &adopted, &mut out, &mut seen);
            closure.insert(o, out);
        }
        let release_of = |child: IrId| -> Vec<IrInstruction> {
            if string_alloc_ids.contains(&child) {
                string_free_id
                    .map(|fid| {
                        vec![IrInstruction::CallDirect {
                            dest: None,
                            func_id: fid,
                            args: vec![child],
                            arg_ownership: vec![OwnershipMode::Move],
                            type_args: vec![],
                            is_tail_call: false,
                        }]
                    })
                    .unwrap_or_default()
            } else if array_alloc_ids.contains(&child) {
                array_free_id
                    .map(|fid| {
                        vec![
                            IrInstruction::CallDirect {
                                dest: None,
                                func_id: fid,
                                args: vec![child],
                                arg_ownership: vec![OwnershipMode::Move],
                                type_args: vec![],
                                is_tail_call: false,
                            },
                            IrInstruction::Free { ptr: child },
                        ]
                    })
                    .unwrap_or_default()
            } else if anon_alloc_ids.contains(&child) {
                anon_drop_id
                    .map(|fid| {
                        vec![IrInstruction::CallDirect {
                            dest: None,
                            func_id: fid,
                            args: vec![child],
                            arg_ownership: vec![OwnershipMode::Move],
                            type_args: vec![],
                            is_tail_call: false,
                        }]
                    })
                    .unwrap_or_default()
            } else {
                // A class instance is not necessarily one allocation. It may
                // own the objects its fields point at, and those may own more,
                // so releasing only the object itself strands everything
                // beneath it. The deep form walks the owned-field mask the
                // class registered at startup and releases the subtree.
                //
                // Falls back to the plain release when the runtime entry is
                // unavailable, which leaks exactly as before rather than
                // leaving the object unreleased.
                free_deep_id
                    .map(|fid| {
                        vec![IrInstruction::CallDirect {
                            dest: None,
                            func_id: fid,
                            args: vec![child],
                            arg_ownership: vec![OwnershipMode::Move],
                            type_args: vec![],
                            is_tail_call: false,
                        }]
                    })
                    .unwrap_or_else(|| vec![IrInstruction::Free { ptr: child }])
            }
        };
        let free_call_ids: BTreeSet<IrFunctionId> = string_free_id
            .into_iter()
            .chain(array_free_id)
            .chain(anon_drop_id)
            .chain(free_deep_id)
            .collect();
        // Plan first over an immutable view (site discovery + dominance),
        // then apply. A child's Free must be dominated by its definition or
        // the emitted instruction reads an undefined register.
        let domtree = crate::ir::loop_analysis::DominatorTree::compute(function);
        let def_positions: BTreeMap<IrId, (IrBlockId, usize)> = {
            let mut m = BTreeMap::new();
            for (&bid, block) in &function.cfg.blocks {
                for (i, inst) in block.instructions.iter().enumerate() {
                    if let Some(d) = inst.dest() {
                        m.entry(d).or_insert((bid, i));
                    }
                }
            }
            m
        };
        let mut plan: Vec<(IrBlockId, usize, IrId)> = Vec::new();
        for (&bid, block) in &function.cfg.blocks {
            let mut sites: BTreeMap<IrId, usize> = BTreeMap::new();
            for (i, inst) in block.instructions.iter().enumerate() {
                let released = match inst {
                    IrInstruction::Free { ptr } => Some(*ptr),
                    IrInstruction::CallDirect { func_id, args, .. }
                        if free_call_ids.contains(func_id) =>
                    {
                        args.first().copied()
                    }
                    _ => None,
                };
                if let Some(r) = released {
                    if closure.contains_key(&r) {
                        sites.insert(r, i);
                    }
                }
            }
            for (o, i) in sites {
                let all_dominated = closure[&o].iter().all(|c| match def_positions.get(c) {
                    Some(&(db, di)) => {
                        if db == bid {
                            di < i
                        } else {
                            domtree.dominates(db, bid)
                        }
                    }
                    None => false,
                });
                if all_dominated {
                    plan.push((bid, i, o));
                }
            }
        }
        // Apply, highest index first within each block.
        plan.sort_by(|a, b| (b.0, b.1).cmp(&(a.0, a.1)));
        for (bid, i, o) in plan {
            let mut insts = Vec::new();
            for &c in &closure[&o] {
                insts.extend(release_of(c));
            }
            inserted += insts.len();
            if let Some(block) = function.cfg.blocks.get_mut(&bid) {
                let at = i + 1;
                for inst in insts.into_iter().rev() {
                    block.instructions.insert(at, inst);
                }
            }
        }
    }

    if crate::ir::free_graph::enabled() {
        record_free_graph(
            function,
            &alloc_ids,
            &allocs_needing_free,
            &derived_sets,
            &string_alloc_ids,
            &array_alloc_ids,
            &anon_alloc_ids,
            string_free_id,
            array_free_id,
            anon_drop_id,
        );
    }

    inserted
}

/// Describe what this function's allocations are and how each was released.
///
/// Read from the FINISHED cfg rather than from the decisions as they are made:
/// what matters when reading the graph is the code that exists, and a release
/// the pass intended but did not emit is exactly the kind of gap worth seeing.
#[allow(clippy::too_many_arguments)]
fn record_free_graph(
    function: &IrFunction,
    alloc_ids: &[IrId],
    allocs_needing_free: &[IrId],
    derived_sets: &BTreeMap<IrId, BTreeSet<IrId>>,
    string_alloc_ids: &BTreeSet<IrId>,
    array_alloc_ids: &BTreeSet<IrId>,
    anon_alloc_ids: &BTreeSet<IrId>,
    string_free_id: Option<IrFunctionId>,
    array_free_id: Option<IrFunctionId>,
    anon_drop_id: Option<IrFunctionId>,
) {
    use crate::ir::free_graph as fg;

    let domtree = crate::ir::loop_analysis::DominatorTree::compute(function);
    let loop_info = crate::ir::loop_analysis::LoopNestInfo::analyze(function, &domtree);
    let mut loops = Vec::new();
    let mut latches: BTreeSet<IrBlockId> = BTreeSet::new();
    for natural in loop_info.loops_by_depth() {
        latches.insert(natural.back_edge_source);
        loops.push(fg::LoopRecord {
            header: natural.header.0,
            latch: natural.back_edge_source.0,
            body: natural.blocks.iter().map(|b| b.0).collect(),
        });
    }

    let blocks: Vec<fg::BlockRecord> = function
        .cfg
        .blocks
        .iter()
        .map(|(id, block)| fg::BlockRecord {
            id: id.0,
            successors: block.successors().iter().map(|b| b.0).collect(),
            terminator: format!("{:?}", block.terminator)
                .split_whitespace()
                .next()
                .unwrap_or("?")
                .trim_end_matches('{')
                .to_string(),
            instructions: block.instructions.len(),
        })
        .collect();

    // Where each allocation is released, and by which of the four routes.
    let mut releases: BTreeMap<IrId, Vec<fg::ReleaseRecord>> = BTreeMap::new();
    for (bid, block) in &function.cfg.blocks {
        let site = if latches.contains(bid) {
            fg::Site::Latch
        } else if matches!(block.terminator, IrTerminator::Return { .. }) {
            fg::Site::Exit
        } else {
            fg::Site::LastUse
        };
        for inst in &block.instructions {
            let (target, how) = match inst {
                IrInstruction::Free { ptr } => (*ptr, fg::Release::PlainFree),
                IrInstruction::CallDirect {
                    func_id,
                    args,
                    dest: None,
                    ..
                } if args.len() == 1 => {
                    let how = if Some(*func_id) == string_free_id {
                        fg::Release::StringFree
                    } else if Some(*func_id) == array_free_id {
                        fg::Release::ArrayFreeThenHeader
                    } else if Some(*func_id) == anon_drop_id {
                        fg::Release::AnonDrop
                    } else {
                        continue;
                    };
                    (args[0], how)
                }
                _ => continue,
            };
            releases.entry(target).or_default().push(fg::ReleaseRecord {
                block: bid.0,
                how,
                site: site.clone(),
            });
        }
    }

    let needing: BTreeSet<IrId> = allocs_needing_free.iter().copied().collect();
    let allocs = alloc_ids
        .iter()
        .map(|id| {
            let kind = if string_alloc_ids.contains(id) {
                "string"
            } else if array_alloc_ids.contains(id) {
                "array"
            } else if anon_alloc_ids.contains(id) {
                "anon"
            } else {
                "plain"
            };
            let def = function
                .cfg
                .blocks
                .iter()
                .find_map(|(bid, b)| {
                    b.instructions
                        .iter()
                        .find(|i| i.dest() == Some(*id))
                        .map(|i| (bid.0, format!("{:?}", i)))
                })
                .unwrap_or((u32::MAX, "<no defining instruction>".to_string()));
            fg::AllocRecord {
                id: id.0,
                kind: kind.to_string(),
                def_block: (def.0 != u32::MAX).then_some(def.0),
                def_inst: def.1.chars().take(240).collect(),
                needs_free: needing.contains(id),
                derived: derived_sets
                    .get(id)
                    .map(|d| d.iter().map(|v| v.0).collect())
                    .unwrap_or_default(),
                releases: releases.get(id).cloned().unwrap_or_default(),
            }
        })
        .collect();

    fg::record(fg::FunctionRecord {
        name: function.name.clone(),
        qualified_name: function.qualified_name.clone(),
        blocks,
        loops,
        allocs,
    });
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
    // A phi incoming is usually a CAST of the allocation, not the allocation
    // itself: a class local whose slot is `*void` takes the pointer through
    // `cast *u8 -> *void`. Map every alias back to the allocation it names, so
    // the checks below run against the allocation.
    let mut alias_root: BTreeMap<IrId, IrId> = BTreeMap::new();
    for &a in alloc_ids {
        for id in build_alias_set_no_phi(a, function) {
            alias_root.entry(id).or_insert(a);
        }
    }
    let dbg = std::env::var_os("RZT_DBG_ROT").is_some();

    // Natural loops from the dominator tree. Reachability alone cannot identify
    // a back edge: in a nested loop the inner preheader is reachable from the
    // inner header via the OUTER back edge, so it reads as a second back edge
    // and every nested loop is rejected. A back edge is an edge whose target
    // DOMINATES its source.
    let domtree = crate::ir::loop_analysis::DominatorTree::compute(function);
    let loop_info = crate::ir::loop_analysis::LoopNestInfo::analyze(function, &domtree);

    for natural in loop_info.loops_by_depth() {
        let header = natural.header;
        let Some(header_block) = function.cfg.blocks.get(&header) else {
            continue;
        };
        if header_block.phi_nodes.is_empty() {
            continue;
        }

        // One latch only: a second edge back to the header would reach it
        // without running the release.
        let latches: Vec<IrBlockId> = function
            .cfg
            .blocks
            .iter()
            .filter(|(id, b)| natural.blocks.contains(id) && b.successors().contains(&header))
            .map(|(&id, _)| id)
            .collect();
        if latches.len() != 1 || latches[0] != natural.back_edge_source {
            if dbg {
                eprintln!(
                    "[rot-rej] {} header={header:?} latches={:?}",
                    function.name, latches
                );
            }
            continue;
        }
        let latch = natural.back_edge_source;

        // A conditional exit at the latch could carry the released value out.
        let Some(latch_block) = function.cfg.blocks.get(&latch) else {
            continue;
        };
        if !matches!(latch_block.terminator, IrTerminator::Branch { .. }) {
            if dbg {
                eprintln!(
                    "[rot-rej] {} header={header:?} latch={latch:?} latch-not-unconditional",
                    function.name
                );
            }
            continue;
        }

        let body = natural.blocks.clone();

        for phi in &header_block.phi_nodes {
            if phi.incoming.len() < 2 {
                continue;
            }
            // Every incoming must name a fresh allocation that escapes by no
            // route other than this phi.
            let roots: Option<Vec<IrId>> = phi
                .incoming
                .iter()
                .map(|(_, v)| alias_root.get(v).copied())
                .collect();
            let Some(roots) = roots else {
                if dbg {
                    let which: Vec<bool> = phi
                        .incoming
                        .iter()
                        .map(|(_, v)| alias_root.contains_key(v))
                        .collect();
                    eprintln!(
                        "[rot-rej] {} phi={:?} not-all-allocs incoming={:?} is_alloc={:?}",
                        function.name, phi.dest, phi.incoming, which
                    );
                }
                continue;
            };
            if !phi.incoming.iter().any(|(pred, _)| *pred == latch) {
                if dbg {
                    eprintln!(
                        "[rot-rej] {} phi={:?} no-latch-incoming",
                        function.name, phi.dest
                    );
                }
                continue;
            }
            // Not just the carried value: every name for the incoming objects
            // must die inside the loop too. `pointer_escapes` only answers
            // "does this leave the function", which was enough while releases
            // sat at return blocks — a point after every use. Releasing inside
            // the loop needs the stronger property, or an object aliased by a
            // second local and read after the loop is freed on iteration one.
            // The incoming allocations themselves, NOT their derived sets: the
            // derived walk runs through the phi and would swallow the carried
            // value along with every pointer computed from it. A second local
            // naming one of these objects reads it through the raw id.
            //
            // Reads BEFORE the loop are initialization — a constructor's field
            // stores sit in their own preheader blocks, not in the block that
            // holds the malloc — so only what the loop exits into is forbidden.
            let mut incoming: BTreeSet<IrId> = BTreeSet::new();
            incoming.insert(phi.dest);
            for v in &roots {
                incoming.extend(build_alias_set_no_phi(*v, function));
            }
            let redefines: BTreeSet<IrBlockId> = roots
                .iter()
                .filter_map(|v| def_block_of(function, *v))
                .collect();
            let mut after = blocks_after_loop(function, &body, &redefines);
            // A block that dominates the header runs before the loop, so what
            // it does to the incoming object is initialization — the header
            // and zero-fill stores of the allocation itself — not a read of a
            // released one. Around an outer loop those blocks are reachable
            // from the inner loop's exit, which is why they land here at all.
            after.retain(|b| !domtree.dominates(*b, header));
            let after = after;
            if !no_uses_in(function, &incoming, &after) {
                if dbg {
                    eprintln!(
                        "[rot-rej] {} phi={:?} incoming-read-after-loop roots={:?}",
                        function.name, phi.dest, roots
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
            let all_local = roots.iter().all(|v| {
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
            out.push((latch, phi.dest, roots));
        }
    }

    out
}

/// Blocks the loop exits into, and everything reachable from them — where a
/// released object must never be read again.
fn blocks_after_loop(
    function: &IrFunction,
    body: &BTreeSet<IrBlockId>,
    redefines: &BTreeSet<IrBlockId>,
) -> BTreeSet<IrBlockId> {
    let mut stack: Vec<IrBlockId> = Vec::new();
    for b in body {
        if let Some(block) = function.cfg.blocks.get(b) {
            stack.extend(block.successors().into_iter().filter(|s| !body.contains(s)));
        }
    }
    let mut after: BTreeSet<IrBlockId> = BTreeSet::new();
    while let Some(b) = stack.pop() {
        if body.contains(&b) || !after.insert(b) {
            continue;
        }
        // Stop where the allocation is produced again. Around an OUTER loop,
        // "after the inner loop" wraps back to before it, and those blocks
        // reuse the same SSA ids for the next iteration's fresh object — a
        // different object, not a read of the released one.
        if redefines.contains(&b) {
            continue;
        }
        if let Some(block) = function.cfg.blocks.get(&b) {
            stack.extend(block.successors());
        }
    }
    after
}

/// True iff none of `values` is read anywhere in `blocks`.
fn no_uses_in(
    function: &IrFunction,
    values: &BTreeSet<IrId>,
    blocks: &BTreeSet<IrBlockId>,
) -> bool {
    for b in blocks {
        let Some(block) = function.cfg.blocks.get(b) else {
            continue;
        };
        if block
            .instructions
            .iter()
            .any(|i| i.uses().iter().any(|u| values.contains(u)))
        {
            return false;
        }
        if block
            .phi_nodes
            .iter()
            .any(|p| p.incoming.iter().any(|(_, v)| values.contains(v)))
        {
            return false;
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if values.contains(v) {
                return false;
            }
        }
    }
    true
}

/// The block defining `value`, if any instruction there produces it.
fn def_block_of(function: &IrFunction, value: IrId) -> Option<IrBlockId> {
    function.cfg.blocks.iter().find_map(|(&bid, block)| {
        block
            .instructions
            .iter()
            .any(|i| i.dest() == Some(value))
            .then_some(bid)
    })
}

/// Per-iteration temporaries: an allocation made inside a loop whose every use
/// stays inside that loop and which no phi carries out of it. It is fresh on
/// each iteration and dead by the end of one, so releasing it at the latch
/// pairs one free with each allocation and holds the loop to a constant
/// footprint.
///
/// This is the confined-to-a-block rule widened to the iteration: a loop body
/// with a short-circuit condition spans several blocks, so a temporary that is
/// plainly dead at the end of the iteration is not confined to any single one.
///
/// The defining block must dominate the latch, or a path that skipped the
/// allocation would reach the release with an undefined register. Anything a
/// phi carries is excluded — that is a rotation, released by its own rule.
fn find_iteration_temporaries(
    function: &IrFunction,
    needing_free: &BTreeSet<IrId>,
    handled: &BTreeSet<IrId>,
    derived_sets: &BTreeMap<IrId, BTreeSet<IrId>>,
) -> Vec<(IrBlockId, IrId)> {
    let mut out: Vec<(IrBlockId, IrId)> = Vec::new();
    if needing_free.is_empty() {
        return out;
    }
    let domtree = crate::ir::loop_analysis::DominatorTree::compute(function);
    let loop_info = crate::ir::loop_analysis::LoopNestInfo::analyze(function, &domtree);

    for natural in loop_info.loops_by_depth() {
        let latch = natural.back_edge_source;
        let Some(latch_block) = function.cfg.blocks.get(&latch) else {
            continue;
        };
        // A conditional latch could leave the loop without running the release.
        if !matches!(latch_block.terminator, IrTerminator::Branch { .. }) {
            continue;
        }
        for &alloc_id in needing_free {
            if handled.contains(&alloc_id) {
                continue;
            }
            let Some(def_block) = def_block_of(function, alloc_id) else {
                continue;
            };
            if !natural.blocks.contains(&def_block) || !domtree.dominates(def_block, latch) {
                continue;
            }
            let Some(derived) = derived_sets.get(&alloc_id) else {
                continue;
            };
            // Every read must sit inside the loop, and no phi may carry it.
            let mut escapes_iteration = false;
            for (&bid, block) in &function.cfg.blocks {
                if !natural.blocks.contains(&bid)
                    && block
                        .instructions
                        .iter()
                        .any(|i| i.uses().iter().any(|u| derived.contains(u)))
                {
                    escapes_iteration = true;
                    break;
                }
                if block
                    .phi_nodes
                    .iter()
                    .any(|p| p.incoming.iter().any(|(_, v)| derived.contains(v)))
                {
                    escapes_iteration = true;
                    break;
                }
                if let IrTerminator::Return { value: Some(v) } = &block.terminator {
                    if derived.contains(v) {
                        escapes_iteration = true;
                        break;
                    }
                }
            }
            if escapes_iteration {
                continue;
            }
            out.push((latch, alloc_id));
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
fn uses_confined_to_body(
    function: &IrFunction,
    values: &BTreeSet<IrId>,
    body: &BTreeSet<IrBlockId>,
) -> bool {
    first_escaping_use(function, values, body).is_none()
}

/// The first (value, block) pair that reads one of `values` outside `body`.
fn first_escaping_use(
    function: &IrFunction,
    values: &BTreeSet<IrId>,
    body: &BTreeSet<IrBlockId>,
) -> Option<(IrId, IrBlockId)> {
    for (&block_id, block) in &function.cfg.blocks {
        let inside = body.contains(&block_id);
        if !inside {
            for inst in &block.instructions {
                if let Some(u) = inst.uses().iter().find(|u| values.contains(u)) {
                    return Some((*u, block_id));
                }
            }
            // A phi elsewhere would carry the object out of the loop.
            for phi in &block.phi_nodes {
                if let Some((_, v)) = phi.incoming.iter().find(|(_, v)| values.contains(v)) {
                    return Some((*v, block_id));
                }
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if values.contains(v) {
                return Some((*v, block_id));
            }
        }
    }
    None
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
/// Aliases of `alloc_id` reached WITHOUT crossing a phi — `var b = a` lowers to
/// a Copy with a fresh id, so a second name for the object is only visible
/// through this walk. Stopping at phis keeps the rotation's carried value out
/// of the set; that value is checked in its own right.
fn build_alias_set_no_phi(alloc_id: IrId, function: &IrFunction) -> BTreeSet<IrId> {
    let mut derived = BTreeSet::new();
    derived.insert(alloc_id);

    let mut changed = true;
    while changed {
        changed = false;
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::GetElementPtr { dest, ptr, .. }
                    | IrInstruction::PtrAdd { dest, ptr, .. } => {
                        if derived.contains(ptr) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
                    IrInstruction::Cast { dest, src, .. }
                    | IrInstruction::BitCast { dest, src, .. }
                    | IrInstruction::SsaBarrier { dest, src, .. }
                    | IrInstruction::Copy { dest, src } => {
                        if derived.contains(src) && derived.insert(*dest) {
                            changed = true;
                        }
                    }
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
        }
    }

    derived
}

fn build_derived_set(alloc_id: IrId, function: &IrFunction) -> BTreeSet<IrId> {
    let mut derived = BTreeSet::new();
    derived.insert(alloc_id);

    let mut changed = true;
    while changed {
        changed = false;
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::GetElementPtr { dest, ptr, .. }
                    | IrInstruction::PtrAdd { dest, ptr, .. } => {
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

/// Can this call hand the caller back a value aliasing one of the tracked
/// pointers it was given? Retention answers where an argument ENDS UP; this
/// answers whether a second name for it comes back out, which is a separate
/// question with a separate answer — `haxe_stringmap_get` retains nothing and
/// returns the stored value. Default-deny: a callee with no body and no
/// verified non-aliasing result may return an alias.
fn call_result_may_alias(
    func_id: &IrFunctionId,
    args: &[IrId],
    set: &BTreeSet<IrId>,
    ids: &AllocFuncIds,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let tracked: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| set.contains(a))
        .map(|(i, _)| i)
        .collect();
    if tracked.is_empty() {
        return false;
    }
    if ids.nonaliasing_result_ids.contains(func_id) {
        return false;
    }
    match returns_alias.get(func_id) {
        Some(mask) => tracked
            .iter()
            .any(|&i| mask.get(i).copied().unwrap_or(true)),
        None => true,
    }
}

/// Every value in `function` that may name the same memory as one of `seed`.
/// Beyond what `build_derived_set` follows, this follows CALL RESULTS: a
/// callee able to return an alias of a tracked argument gives the caller a
/// second name for that memory, and a caller blind to that name will free
/// underneath it.
///
/// `through_loads` additionally grows the set through reads made THROUGH it,
/// which tracks the children of the memory rather than the memory itself. The
/// two questions have different answers and different callers, so the flag
/// picks one rather than merging them.
fn alias_closure(
    seed: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
    through_loads: bool,
) -> BTreeSet<IrId> {
    let mut set = seed.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                let pulled = match inst {
                    IrInstruction::GetElementPtr { dest, ptr, .. }
                    | IrInstruction::PtrAdd { dest, ptr, .. } => set.contains(ptr).then_some(*dest),
                    IrInstruction::Cast { dest, src, .. }
                    | IrInstruction::BitCast { dest, src, .. }
                    | IrInstruction::SsaBarrier { dest, src, .. }
                    | IrInstruction::Copy { dest, src } => set.contains(src).then_some(*dest),
                    IrInstruction::Select {
                        dest,
                        true_val,
                        false_val,
                        ..
                    } => (set.contains(true_val) || set.contains(false_val)).then_some(*dest),
                    IrInstruction::Load { dest, ptr, .. } if through_loads => {
                        set.contains(ptr).then_some(*dest)
                    }
                    IrInstruction::CallDirect {
                        dest: Some(d),
                        func_id,
                        args,
                        ..
                    } => {
                        call_result_may_alias(func_id, args, &set, ids, returns_alias).then_some(*d)
                    }
                    // An indirect callee is unknown by construction, and a
                    // closure invoked through a tracked pointer can return a
                    // value out of its own environment.
                    IrInstruction::CallIndirect {
                        dest: Some(d),
                        func_ptr,
                        args,
                        ..
                    } => (set.contains(func_ptr) || args.iter().any(|a| set.contains(a)))
                        .then_some(*d),
                    _ => None,
                };
                if let Some(d) = pulled {
                    if set.insert(d) {
                        changed = true;
                    }
                }
            }
            for phi in &block.phi_nodes {
                if !set.contains(&phi.dest)
                    && phi.incoming.iter().any(|(_, v)| set.contains(v))
                    && set.insert(phi.dest)
                {
                    changed = true;
                }
            }
        }
    }
    set
}

/// The closure of values read OUT of `derived` — its children, grandchildren,
/// and every alias of them. Empty when nothing is read through it.
fn read_out_closure(
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> BTreeSet<IrId> {
    let mut seed = BTreeSet::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInstruction::Load { dest, ptr, .. } if derived.contains(ptr) => {
                    seed.insert(*dest);
                }
                // An accessor hands back a child without ever emitting a Load
                // in this function. Seeding only from loads makes a getter's
                // result invisible and the read-out question answer itself
                // vacuously.
                IrInstruction::CallDirect {
                    dest: Some(d),
                    func_id,
                    args,
                    ..
                } if call_result_may_alias(func_id, args, derived, ids, returns_alias) => {
                    seed.insert(*d);
                }
                IrInstruction::CallIndirect {
                    dest: Some(d),
                    func_ptr,
                    args,
                    ..
                } if derived.contains(func_ptr) || args.iter().any(|a| derived.contains(a)) => {
                    seed.insert(*d);
                }
                _ => {}
            }
        }
    }
    if seed.is_empty() {
        return seed;
    }
    alias_closure(&seed, function, ids, returns_alias, true)
}

/// Per class, which instance fields hold a value the object owns.
///
/// A field is owned only when EVERY store to it, anywhere in the program,
/// hands over a freshly allocated object that nothing else keeps. Anything
/// unproven stays borrowed, because the two mistakes are not symmetric:
/// declining to free an owned field leaks, while freeing a borrowed one
/// corrupts the heap.
///
/// That asymmetry is the whole safety argument, and it is not a cycle
/// question. `TreeNode.left` points at another `TreeNode`, so a rule that
/// refused self-referential types would refuse the one shape that leaks;
/// meanwhile deltablue's `Strength` is perfectly acyclic yet its instances are
/// static singletons that many constraints share, and freeing one through a
/// constraint corrupts the heap on the FIRST visit, where no cycle detection
/// would ever look. Aliasing is the property that separates them.
#[derive(Default)]
struct OwnedFields {
    /// class type id -> bitmask over INSTANCE field indices (bit 0 is the
    /// first instance field, which lives at GEP index 1: slot 0 is the header)
    masks: BTreeMap<u64, u64>,
}

/// Which class a local allocation belongs to, read from the type-id header the
/// constructor writes into slot 0.
fn alloc_type_ids(function: &IrFunction, ids: &AllocFuncIds) -> BTreeMap<IrId, u64> {
    let mut allocs: BTreeSet<IrId> = BTreeSet::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                ..
            } = inst
            {
                if ids.malloc_ids.contains(func_id) {
                    allocs.insert(*d);
                }
            }
        }
    }
    if allocs.is_empty() {
        return BTreeMap::new();
    }

    let consts = const_ints(function);
    let mut out = BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            let IrInstruction::Store { ptr, value, .. } = inst else {
                continue;
            };
            // The header write is `store gep(alloc, 0), <type id>`.
            let Some((base, 0)) = gep_base_and_index(function, *ptr, &consts) else {
                continue;
            };
            if !allocs.contains(&base) {
                continue;
            }
            if let Some(tid) = consts.get(value) {
                if *tid > 0 {
                    out.insert(base, *tid as u64);
                }
            }
        }
    }
    out
}

/// Integer constants defined in this function, by register.
fn const_ints(function: &IrFunction) -> BTreeMap<IrId, i64> {
    let mut out = BTreeMap::new();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::Const { dest, value } = inst {
                let n = match value {
                    IrValue::I8(v) => Some(*v as i64),
                    IrValue::I16(v) => Some(*v as i64),
                    IrValue::I32(v) => Some(*v as i64),
                    IrValue::I64(v) => Some(*v),
                    IrValue::U8(v) => Some(*v as i64),
                    IrValue::U16(v) => Some(*v as i64),
                    IrValue::U32(v) => Some(*v as i64),
                    IrValue::U64(v) => Some(*v as i64),
                    _ => None,
                };
                if let Some(n) = n {
                    out.insert(*dest, n);
                }
            }
        }
    }
    out
}

/// The `(base, constant index)` a GEP names, or `None` when the index is not a
/// known constant -- a computed index could reach any field, so it is not a
/// fact about one of them.
fn gep_base_and_index(
    function: &IrFunction,
    ptr: IrId,
    consts: &BTreeMap<IrId, i64>,
) -> Option<(IrId, i64)> {
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::GetElementPtr {
                dest,
                ptr: base,
                indices,
                ..
            } = inst
            {
                if *dest != ptr {
                    continue;
                }
                if indices.len() != 1 {
                    return None;
                }
                return consts.get(&indices[0]).map(|i| (*base, *i));
            }
        }
    }
    None
}

/// Which parameters a function stores into its receiver, and at which field.
///
/// `TreeNode.new(this, left, right, item)` stores `left` at GEP index 1 and
/// `right` at index 2, so the caller's argument is what the field will hold and
/// the question "does this field own its value" is really a question about
/// every call site.
///
/// A parameter that reaches more than one field, or that is used for anything
/// beyond that single store, is left out: the point is to identify a handover,
/// and a parameter doing something else as well has not handed anything over.
fn param_to_field_map(
    function: &IrFunction,
    ids: &AllocFuncIds,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> BTreeMap<usize, i64> {
    let params = &function.signature.parameters;
    if params.len() < 2 {
        return BTreeMap::new();
    }
    let consts = const_ints(function);
    let self_derived = alias_closure(
        &BTreeSet::from([params[0].reg]),
        function,
        ids,
        returns_alias,
        false,
    );

    let mut out = BTreeMap::new();
    for (pi, param) in params.iter().enumerate().skip(1) {
        let derived = alias_closure(
            &BTreeSet::from([param.reg]),
            function,
            ids,
            returns_alias,
            false,
        );
        let mut field: Option<i64> = None;
        let mut disqualified = false;
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::Store { ptr, value, .. } if derived.contains(value) => {
                        match gep_base_and_index(function, *ptr, &consts) {
                            Some((base, idx)) if self_derived.contains(&base) && idx > 0 => {
                                if field.replace(idx).is_some_and(|prev| prev != idx) {
                                    disqualified = true;
                                }
                            }
                            _ => disqualified = true,
                        }
                    }
                    // Reads and the alias-forming ops are how the value gets
                    // from the parameter to the store; anything else is a use
                    // this walk has not accounted for.
                    IrInstruction::Load { .. }
                    | IrInstruction::GetElementPtr { .. }
                    | IrInstruction::PtrAdd { .. }
                    | IrInstruction::Cast { .. }
                    | IrInstruction::BitCast { .. }
                    | IrInstruction::SsaBarrier { .. }
                    | IrInstruction::Copy { .. }
                    | IrInstruction::Select { .. }
                    | IrInstruction::Cmp { .. }
                    | IrInstruction::Store { .. } => {}
                    IrInstruction::CallDirect { func_id, args, .. }
                        if args.iter().any(|a| derived.contains(a)) =>
                    {
                        // A callee that merely reads it is fine; one that could
                        // keep it means the field is not the only holder.
                        let borrows = ids.copy_only_ids.contains(func_id)
                            || ids.nonaliasing_result_ids.contains(func_id);
                        if !borrows {
                            disqualified = true;
                        }
                    }
                    other => {
                        if other.uses().iter().any(|u| derived.contains(u)) {
                            disqualified = true;
                        }
                    }
                }
            }
        }
        if !disqualified {
            if let Some(idx) = field {
                out.insert(pi, idx);
            }
        }
    }
    out
}

/// Is this value a freshly made object that nothing else in the function keeps?
///
/// "Fresh" is the allocation itself; "single-holder" is that its only use is
/// the handover site given. Both are needed: a fresh object stored into two
/// places has two holders, and freeing it through either dangles the other.
fn fresh_single_holder(
    value: IrId,
    handover: &IrInstruction,
    function: &IrFunction,
    ids: &AllocFuncIds,
    fresh_object_fns: &BTreeSet<IrFunctionId>,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let is_fresh = function.cfg.blocks.values().any(|b| {
        b.instructions.iter().any(|i| match i {
            IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                ..
            } => {
                *d == value
                    && (ids.malloc_ids.contains(func_id) || fresh_object_fns.contains(func_id))
            }
            _ => false,
        })
    });
    if !is_fresh {
        return false;
    }

    let derived = alias_closure(
        &BTreeSet::from([value]),
        function,
        ids,
        returns_alias,
        false,
    );
    let mut uses = 0usize;
    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            if phi.incoming.iter().any(|(_, v)| derived.contains(v)) {
                return false;
            }
        }
        for inst in &block.instructions {
            if std::ptr::eq(inst, handover) {
                continue;
            }
            match inst {
                // The chain that produces and carries it.
                IrInstruction::CallDirect { dest: Some(d), .. } if *d == value => {}
                IrInstruction::GetElementPtr { .. }
                | IrInstruction::PtrAdd { .. }
                | IrInstruction::Cast { .. }
                | IrInstruction::BitCast { .. }
                | IrInstruction::SsaBarrier { .. }
                | IrInstruction::Copy { .. } => {}
                other => {
                    if other.uses().iter().any(|u| derived.contains(u)) {
                        uses += 1;
                    }
                }
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if derived.contains(v) {
                return false;
            }
        }
    }
    uses == 0
}

/// Functions whose returned value is a freshly built class instance the caller
/// owns. The object analogue of `compute_returns_fresh_arrays`: without it a
/// tree returned from a builder is not an allocation the pass can even see, so
/// nothing frees the root and the owned-field mask never gets a chance to
/// reclaim what hangs off it.
fn compute_returns_fresh_objects(
    module: &IrModule,
    ids: &AllocFuncIds,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> BTreeSet<IrFunctionId> {
    let mut fresh: BTreeSet<IrFunctionId> = BTreeSet::new();
    loop {
        let mut changed = false;
        for (fid, function) in bodied(module) {
            if fresh.contains(&fid) || returns_a_pointer(function).is_none() {
                continue;
            }
            if returns_fresh_object(function, ids, &fresh, param_retention) {
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

/// The single value this function returns, when every return agrees.
fn returns_a_pointer(function: &IrFunction) -> Option<Vec<IrId>> {
    let mut rets = Vec::new();
    for block in function.cfg.blocks.values() {
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            rets.push(*v);
        }
        for inst in &block.instructions {
            if let IrInstruction::Return { value: Some(v) } = inst {
                rets.push(*v);
            }
        }
    }
    (!rets.is_empty()).then_some(rets)
}

/// Every return of this function hands back an object it allocated and did not
/// publish anywhere else. Mirrors `returns_fresh_array`, minus the
/// array-header identification: a class instance is a single allocation.
fn returns_fresh_object(
    function: &IrFunction,
    ids: &AllocFuncIds,
    fresh: &BTreeSet<IrFunctionId>,
    param_retention: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let Some(ret_vals) = returns_a_pointer(function) else {
        return false;
    };
    let type_ids = alloc_type_ids(function, ids);

    let mut sources: Vec<IrId> = type_ids.keys().copied().collect();
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::CallDirect {
                dest: Some(d),
                func_id,
                ..
            } = inst
            {
                if fresh.contains(func_id) {
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

    'ret: for v in &ret_vals {
        for &src in &sources {
            let derived = build_derived_set(src, function);
            if !derived.contains(v) {
                continue;
            }
            if pointer_escapes_ex(
                src,
                &derived,
                function,
                &BTreeSet::new(),
                &ids.anon_setter_ids,
                &ids.copy_only_ids,
                param_retention,
                true,
                None,
            ) {
                continue;
            }
            // Freed here and then returned would hand back a dangling pointer.
            let freed = function.cfg.blocks.values().any(|block| {
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
            if freed {
                continue;
            }
            continue 'ret;
        }
        return false;
    }
    true
}

/// Whole-program owned-field census.
///
/// Evidence is gathered per (class, field) from two shapes -- a direct store
/// through a local allocation whose type is known, and an argument handed to a
/// callee that stores it into its receiver. A field is owned only if evidence
/// exists and EVERY piece of it shows a fresh, single-holder value. One
/// unproven store anywhere in the program is enough to leave the field
/// borrowed, which is why a shared singleton like deltablue's `Strength`
/// cannot be mistaken for owned: it is stored from a static, not freshly made.
fn compute_owned_fields(
    module: &IrModule,
    ids: &AllocFuncIds,
    fresh_object_fns: &BTreeSet<IrFunctionId>,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> OwnedFields {
    let mut param_fields: BTreeMap<IrFunctionId, BTreeMap<usize, i64>> = BTreeMap::new();
    for (fid, f) in bodied(module) {
        let m = param_to_field_map(f, ids, returns_alias);
        if !m.is_empty() {
            param_fields.insert(fid, m);
        }
    }

    // (type id, GEP index) -> every store seen was a proven handover
    let mut evidence: BTreeMap<(u64, i64), bool> = BTreeMap::new();
    let mut note = |key: (u64, i64), ok: bool| {
        let e = evidence.entry(key).or_insert(true);
        *e &= ok;
    };

    for (_, function) in bodied(module) {
        let type_ids = alloc_type_ids(function, ids);
        if type_ids.is_empty() {
            continue;
        }
        let consts = const_ints(function);
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInstruction::Store { ptr, value, .. } => {
                        let Some((base, idx)) = gep_base_and_index(function, *ptr, &consts) else {
                            continue;
                        };
                        if idx == 0 {
                            continue; // the type-id header
                        }
                        let Some(&tid) = type_ids.get(&base) else {
                            continue;
                        };
                        note(
                            (tid, idx),
                            fresh_single_holder(
                                *value,
                                inst,
                                function,
                                ids,
                                fresh_object_fns,
                                returns_alias,
                            ),
                        );
                    }
                    IrInstruction::CallDirect { func_id, args, .. } => {
                        let Some(map) = param_fields.get(func_id) else {
                            continue;
                        };
                        let Some(recv) = args.first() else { continue };
                        let Some(&tid) = type_ids.get(recv) else {
                            continue;
                        };
                        for (&pi, &idx) in map {
                            let Some(&arg) = args.get(pi) else { continue };
                            note(
                                (tid, idx),
                                fresh_single_holder(
                                    arg,
                                    inst,
                                    function,
                                    ids,
                                    fresh_object_fns,
                                    returns_alias,
                                ),
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut owned = OwnedFields::default();
    for ((tid, idx), ok) in evidence {
        if !ok {
            continue;
        }
        // GEP index 1 is instance field 0; slot 0 is the type-id header.
        let bit = idx - 1;
        if (0..64).contains(&bit) {
            *owned.masks.entry(tid).or_insert(0) |= 1u64 << bit;
        }
    }
    if std::env::var_os("RZT_DBG_OWNED").is_some() {
        for (tid, mask) in &owned.masks {
            eprintln!("[owned] type_id={tid} mask={mask:#x}");
        }
    }
    owned
}

/// Declare each class's owned-field mask at startup.
///
/// The masks ride `__vtable_init__` rather than the RTTI that `ClassInfo`
/// carries, because that registry is filled by a call from the compiler
/// process and an AOT-built binary therefore has none of it. `__vtable_init__`
/// is emitted code that every backend runs before main, so a mask declared
/// here is present wherever the program is.
fn register_owned_masks(module: &mut IrModule, owned: &OwnedFields) {
    if owned.masks.is_empty() {
        return;
    }
    let Some(init_id) = module
        .functions
        .iter()
        .find(|(_, f)| f.name == "__vtable_init__" && !f.cfg.blocks.is_empty())
        .map(|(id, _)| *id)
    else {
        // No startup hook in this module: the masks have nowhere to live, so
        // deep-free will fall back to releasing objects one at a time.
        return;
    };

    let register_fn = match module
        .extern_functions
        .iter()
        .find(|(_, f)| f.name == "haxe_register_owned_mask")
        .map(|(id, _)| *id)
    {
        Some(id) => id,
        None => {
            let id = module.alloc_function_id();
            module.extern_functions.insert(
                id,
                super::modules::IrExternFunction {
                    id,
                    name: "haxe_register_owned_mask".to_string(),
                    symbol_id: crate::tast::SymbolId::from_raw(0),
                    signature: super::IrFunctionSignature {
                        parameters: vec![
                            super::functions::IrParameter {
                                name: "type_id".to_string(),
                                ty: IrType::I64,
                                reg: IrId(0),
                                by_ref: false,
                            },
                            super::functions::IrParameter {
                                name: "mask".to_string(),
                                ty: IrType::I64,
                                reg: IrId(1),
                                by_ref: false,
                            },
                        ],
                        return_type: IrType::Void,
                        calling_convention: super::CallingConvention::C,
                        can_throw: false,
                        type_params: vec![],
                        uses_sret: false,
                    },
                    source: "runtime".to_string(),
                },
            );
            id
        }
    };

    let Some(function) = module.functions.get_mut(&init_id) else {
        return;
    };
    let entry = function.entry_block();
    let regs: Vec<(IrId, IrId)> = owned
        .masks
        .keys()
        .map(|_| (function.alloc_reg(), function.alloc_reg()))
        .collect();
    let Some(block) = function.cfg.blocks.get_mut(&entry) else {
        return;
    };
    for ((&tid, &mask), (tid_reg, mask_reg)) in owned.masks.iter().zip(regs) {
        block.instructions.push(IrInstruction::Const {
            dest: tid_reg,
            value: IrValue::I64(tid as i64),
        });
        block.instructions.push(IrInstruction::Const {
            dest: mask_reg,
            value: IrValue::I64(mask as i64),
        });
        block.instructions.push(IrInstruction::CallDirect {
            dest: None,
            func_id: register_fn,
            args: vec![tid_reg, mask_reg],
            arg_ownership: vec![OwnershipMode::Copy, OwnershipMode::Copy],
            type_args: vec![],
            is_tail_call: false,
        });
    }
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

                // Reads of and through the pointer, and the alias-forming ops
                // already folded into `derived`. Releasing it is what a Free
                // does, not an escape.
                IrInstruction::Load { .. }
                | IrInstruction::GetElementPtr { .. }
                | IrInstruction::PtrAdd { .. }
                | IrInstruction::Cast { .. }
                | IrInstruction::BitCast { .. }
                | IrInstruction::SsaBarrier { .. }
                | IrInstruction::Copy { .. }
                | IrInstruction::Select { .. }
                | IrInstruction::Cmp { .. }
                | IrInstruction::BinOp { .. }
                | IrInstruction::UnOp { .. }
                | IrInstruction::Free { .. } => {}

                // Default-deny. Every aggregate-forming and publishing
                // instruction that is not named above -- CreateUnion,
                // InsertValue, Throw, the atomics, InlineAsm -- hands the
                // pointer somewhere this walk cannot follow, and a silent
                // fallthrough here reads as "does not escape".
                other => {
                    if other
                        .uses()
                        .iter()
                        .any(|u| *u == alloc_id || derived.contains(u))
                    {
                        return true;
                    }
                }
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
/// Where a retained parameter ends up. `param_retained` collapses this to a
/// bool for escape analysis; ownership TRANSFER needs the distinction, because
/// "stored into the receiver" is the one retention a caller can reason about:
/// if the receiver is a caller-owned allocation that the caller frees, the
/// stored value dies with it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Retention {
    /// Never retained: the callee only reads it.
    No,
    /// Every retention is a store whose pointer derives from param 0 (the
    /// receiver), directly or through a callee with the same property.
    IntoSelf,
    /// Retained somewhere the caller cannot see: a global, a return, a
    /// different object, an unknown callee.
    Elsewhere,
}

/// Per-function, per-parameter retention analysis. Computed unconditionally --
/// unlike the bool masks this feeds ONLY the ownership-transfer and
/// benign-escape analyses, so it cannot change what the conservative escape
/// walk counts as escaping.
struct RetentionInfo {
    kinds: BTreeMap<IrFunctionId, Vec<Retention>>,
    /// Per parameter: does the callee let a value READ OUT of it escape
    /// (return it, store it, hand it to a leaking callee)? A callee with kind
    /// `No` but `leaks = true` borrows the object yet publishes one of its
    /// children -- freeing the object after the call is fine, but freeing
    /// anything the object OWNS would dangle the published child.
    leaks: BTreeMap<IrFunctionId, Vec<bool>>,
    /// Per parameter: can the RETURNED value alias it -- be it, point into it,
    /// or have been read out of it? Without this a caller sees a call it has
    /// proved harmless and never learns that its result is a second name for
    /// the argument, so every later use of that name is invisible.
    returns_alias: BTreeMap<IrFunctionId, Vec<bool>>,
}

fn compute_retention_kinds(module: &IrModule, ids: &AllocFuncIds) -> RetentionInfo {
    // Phase 1: which parameters can come back out of a return. This one must
    // rebuild its alias sets every round -- they follow call results, so they
    // GROW as the map does, and a set pinned at round zero would hand the
    // later phases an under-approximation.
    let returns_alias = compute_returns_alias(module, ids);

    // Phase 2: retention and child-leakage, over alias sets built once with
    // the settled map. They are body-static again at this point.
    let mut kinds: BTreeMap<IrFunctionId, Vec<Retention>> = BTreeMap::new();
    let mut leaks: BTreeMap<IrFunctionId, Vec<bool>> = BTreeMap::new();
    let mut param_derived: BTreeMap<IrFunctionId, Vec<BTreeSet<IrId>>> = BTreeMap::new();
    // `self_derived` decides whether a store lands in the RECEIVER, which is an
    // attribution -- it names one specific object. Only the locally-provable
    // set can answer that; a call result may name memory from anywhere, and
    // reading it as "inside the receiver" is how a value gets adopted by an
    // owner it does not belong to.
    let mut param_must: BTreeMap<IrFunctionId, BTreeSet<IrId>> = BTreeMap::new();
    for (fid, f) in bodied(module) {
        kinds.insert(fid, vec![Retention::No; f.signature.parameters.len()]);
        leaks.insert(fid, vec![false; f.signature.parameters.len()]);
        let sets = f
            .signature
            .parameters
            .iter()
            .map(|p| alias_closure(&BTreeSet::from([p.reg]), f, ids, &returns_alias, false))
            .collect();
        param_derived.insert(fid, sets);
        if let Some(p0) = f.signature.parameters.first() {
            param_must.insert(fid, build_derived_set(p0.reg, f));
        }
    }
    loop {
        let mut changed = false;
        for (fid, function) in bodied(module) {
            let self_derived = param_must.get(&fid).cloned();
            for pi in 0..function.signature.parameters.len() {
                let derived = &param_derived[&fid][pi];
                let cur = kinds[&fid][pi];
                if cur != Retention::Elsewhere {
                    let k =
                        retention_kind_of(derived, self_derived.as_ref(), function, ids, &kinds);
                    if k > cur {
                        kinds.get_mut(&fid).unwrap()[pi] = k;
                        changed = true;
                    }
                }
                if !leaks[&fid][pi]
                    && param_leaks_children(derived, function, ids, &kinds, &leaks, &returns_alias)
                {
                    leaks.get_mut(&fid).unwrap()[pi] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    RetentionInfo {
        kinds,
        leaks,
        returns_alias,
    }
}

/// Per-function, per-parameter: can the returned value alias this parameter?
///
/// Optimistic fixpoint over `false < true`: every parameter of every bodied
/// function starts at `false` and only escalates, so a recursive cycle settles
/// at the answer its base cases justify. The pessimistic default belongs in
/// the transfer function, not the initial value -- `call_result_may_alias`
/// answers `true` for a callee with no body. Reversing those two is the
/// difference between a leak and a use-after-free.
fn compute_returns_alias(
    module: &IrModule,
    ids: &AllocFuncIds,
) -> BTreeMap<IrFunctionId, Vec<bool>> {
    let mut alias: BTreeMap<IrFunctionId, Vec<bool>> = bodied(module)
        .map(|(fid, f)| (fid, vec![false; f.signature.parameters.len()]))
        .collect();
    loop {
        let mut changed = false;
        for (fid, function) in bodied(module) {
            for pi in 0..function.signature.parameters.len() {
                if alias[&fid][pi] {
                    continue;
                }
                let derived = alias_closure(
                    &BTreeSet::from([function.signature.parameters[pi].reg]),
                    function,
                    ids,
                    &alias,
                    false,
                );
                if returns_alias_of(&derived, function, ids, &alias) {
                    alias.get_mut(&fid).unwrap()[pi] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    if std::env::var_os("RZT_DBG_ALIAS").is_some() {
        for (fid, function) in bodied(module) {
            let mask = &alias[&fid];
            if mask.iter().any(|b| *b) {
                eprintln!("[alias] {} {fid:?} returns_alias={mask:?}", function.name);
            }
        }
    }
    alias
}

/// The functions that actually have a body.
///
/// `mark_as_extern` clears a function's blocks but LEAVES it in
/// `module.functions`, so the map is full of bodyless stubs. An interprocedural
/// analysis that seeds one of those at its optimistic bottom pins it there --
/// there is no body for the fixpoint to find evidence in -- and every lookup
/// then reads as PROVEN SAFE for exactly the runtime callees the pessimistic
/// default was written for. Keeping them out of the domain is what makes a
/// missing entry mean "unknown" instead of "harmless".
fn bodied(module: &IrModule) -> impl Iterator<Item = (IrFunctionId, &IrFunction)> {
    module
        .functions
        .iter()
        .filter(|(_, f)| !f.cfg.blocks.is_empty())
        .map(|(&fid, f)| (fid, f))
}

/// Does any return of `function` hand back a value aliasing this parameter or
/// something read out of it?
fn returns_alias_of(
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    // Returning a CHILD counts: the caller gets a name for memory the
    // parameter's graph owns, which is what it must know before freeing it.
    let mut exposed = derived.clone();
    exposed.extend(read_out_closure(derived, function, ids, returns_alias));
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            if let IrInstruction::Return { value: Some(v) } = inst {
                if exposed.contains(v) {
                    return true;
                }
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if exposed.contains(v) {
                return true;
            }
        }
    }
    false
}

/// Does anything READ OUT of this parameter escape the callee? The closure
/// covers loads through the parameter at any width, grown through further
/// loads, alias-forming ops and call results, so neither a pointer punned
/// through an integer register nor one handed back by a callee slips past.
fn param_leaks_children(
    derived: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    kinds: &BTreeMap<IrFunctionId, Vec<Retention>>,
    leaks: &BTreeMap<IrFunctionId, Vec<bool>>,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let read_out = read_out_closure(derived, function, ids, returns_alias);
    if read_out.is_empty() {
        return false;
    }
    let in_read = |v: &IrId| read_out.contains(v);
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInstruction::Load { .. }
                | IrInstruction::GetElementPtr { .. }
                | IrInstruction::PtrAdd { .. }
                | IrInstruction::Cast { .. }
                | IrInstruction::BitCast { .. }
                | IrInstruction::SsaBarrier { .. }
                | IrInstruction::Copy { .. }
                | IrInstruction::Select { .. }
                | IrInstruction::Cmp { .. }
                | IrInstruction::BinOp { .. }
                | IrInstruction::UnOp { .. } => {}
                IrInstruction::Store { value, .. } if in_read(value) => return true,
                IrInstruction::Store { .. } => {}
                IrInstruction::StoreGlobal { value, .. } if in_read(value) => return true,
                IrInstruction::StoreGlobal { .. } => {}
                IrInstruction::CreateStruct { fields, .. } if fields.iter().any(in_read) => {
                    return true;
                }
                IrInstruction::CreateStruct { .. } => {}
                IrInstruction::MemCopy { dest, src, .. } if in_read(dest) || in_read(src) => {
                    return true;
                }
                IrInstruction::MemCopy { .. } => {}
                IrInstruction::Throw { exception } if in_read(exception) => return true,
                IrInstruction::Return { value: Some(v) } if in_read(v) => return true,
                IrInstruction::CallIndirect { func_ptr, args, .. }
                    if in_read(func_ptr) || args.iter().any(in_read) =>
                {
                    return true;
                }
                IrInstruction::CallDirect { func_id, args, .. } => {
                    if ids.copy_only_ids.contains(func_id) {
                        continue;
                    }
                    for (i, arg) in args.iter().enumerate() {
                        if !in_read(arg) {
                            continue;
                        }
                        let kind_no = kinds
                            .get(func_id)
                            .and_then(|k| k.get(i))
                            .map(|k| *k == Retention::No)
                            .unwrap_or(false);
                        let no_leak = leaks
                            .get(func_id)
                            .and_then(|l| l.get(i))
                            .map(|l| !*l)
                            .unwrap_or(false);
                        if !(kind_no && no_leak) {
                            return true;
                        }
                    }
                }
                other => {
                    if other.uses().iter().any(|u| in_read(u)) {
                        return true;
                    }
                }
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if in_read(v) {
                return true;
            }
        }
    }
    false
}

fn retention_kind_of(
    derived: &BTreeSet<IrId>,
    self_derived: Option<&BTreeSet<IrId>>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    kinds: &BTreeMap<IrFunctionId, Vec<Retention>>,
) -> Retention {
    let in_set = |v: &IrId| derived.contains(v);
    let into_self = |ptr: &IrId| self_derived.map(|sd| sd.contains(ptr)).unwrap_or(false);
    let mut acc = Retention::No;
    let mut raise = |k: Retention, acc: &mut Retention| {
        if k > *acc {
            *acc = k;
        }
    };
    for block in function.cfg.blocks.values() {
        for inst in &block.instructions {
            match inst {
                IrInstruction::Store { ptr, value, .. } if in_set(value) => {
                    let k = if into_self(ptr) {
                        Retention::IntoSelf
                    } else {
                        Retention::Elsewhere
                    };
                    raise(k, &mut acc);
                }
                IrInstruction::StoreGlobal { value, .. } if in_set(value) => {
                    return Retention::Elsewhere;
                }
                IrInstruction::CreateStruct { fields, .. } if fields.iter().any(in_set) => {
                    return Retention::Elsewhere;
                }
                IrInstruction::MemCopy { dest, src, .. } if in_set(dest) || in_set(src) => {
                    return Retention::Elsewhere;
                }
                IrInstruction::Throw { exception } if in_set(exception) => {
                    return Retention::Elsewhere;
                }
                IrInstruction::Return { value: Some(v) } if in_set(v) => {
                    return Retention::Elsewhere;
                }
                IrInstruction::CallIndirect { func_ptr, args, .. }
                    if in_set(func_ptr) || args.iter().any(in_set) =>
                {
                    return Retention::Elsewhere;
                }
                IrInstruction::CallDirect { func_id, args, .. } => {
                    if ids.copy_only_ids.contains(func_id) {
                        // reads only
                    } else if let Some(callee_kinds) = kinds.get(func_id) {
                        for (i, arg) in args.iter().enumerate() {
                            if !in_set(arg) {
                                continue;
                            }
                            match callee_kinds.get(i).copied() {
                                Some(Retention::No) => {}
                                Some(Retention::IntoSelf) => {
                                    // Retained into the callee's receiver: that
                                    // is our receiver only if we passed it.
                                    let k = if args.first().map(|a| into_self(a)).unwrap_or(false) {
                                        Retention::IntoSelf
                                    } else {
                                        Retention::Elsewhere
                                    };
                                    raise(k, &mut acc);
                                }
                                _ => return Retention::Elsewhere,
                            }
                        }
                    } else if ids.array_safe_ids.contains(func_id)
                        || ids.anon_safe_ids.contains(func_id)
                    {
                        if args.iter().skip(1).any(in_set) {
                            return Retention::Elsewhere;
                        }
                    } else if args.iter().any(in_set) {
                        return Retention::Elsewhere;
                    }
                }
                _ => {}
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if in_set(v) {
                return Retention::Elsewhere;
            }
        }
    }
    acc
}

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

/// How a "conservatively escaping" allocation actually behaves.
enum EscapeClass {
    /// Its every escape is into this single tracked allocation.
    Owner(IrId),
    /// Every exposure is a borrow: callees read it (and leak none of its
    /// children), nothing stores, returns or captures it. The conservative
    /// walk called it an escape only because any call argument counts.
    Benign,
    /// A real escape, or one this walk cannot prove otherwise.
    Escapes,
}

/// The single tracked allocation `x` escaped into, if its every escape is one
/// the caller can prove dies with that owner. Default-deny: any use of `x`
/// this walk does not positively recognise disqualifies the transfer.
#[allow(clippy::too_many_arguments)]
fn classify_escape_owner(
    x: IrId,
    x_derived: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    info: &RetentionInfo,
    all_may: &BTreeMap<IrId, BTreeSet<IrId>>,
    all_must: &BTreeMap<IrId, BTreeSet<IrId>>,
) -> EscapeClass {
    let in_x = |v: &IrId| *v == x || x_derived.contains(v);
    // The unique tracked allocation a value belongs to, excluding x itself.
    // Attribution, so it reads the MUST set: a call result may name memory
    // reached through the argument rather than memory the argument owns, and
    // naming an owner on that basis frees x at the wrong object's death.
    let resolve = |v: &IrId| -> Option<IrId> {
        let mut found: Option<IrId> = None;
        for (&a, derived) in all_must {
            if a == x {
                continue;
            }
            if *v == a || derived.contains(v) {
                if found.is_some() {
                    return None; // ambiguous
                }
                found = Some(a);
            }
        }
        // Ambiguous if any OTHER allocation may also name it.
        if let Some(f) = found {
            for (&a, may) in all_may {
                if a != f && a != x && may.contains(v) {
                    return None;
                }
            }
        }
        found
    };

    let mut owner: Option<IrId> = None;
    let mut adopt = |cand: Option<IrId>| -> bool {
        match (cand, owner) {
            (Some(c), None) => {
                owner = Some(c);
                true
            }
            (Some(c), Some(o)) => c == o,
            (None, _) => false,
        }
    };

    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            // A phi mixing x with foreign values re-materialises x under a
            // name whose uses this walk cannot attribute; refuse.
            if phi.incoming.iter().any(|(_, v)| in_x(v)) {
                let all_ours = phi.incoming.iter().all(|(_, v)| in_x(v));
                if !all_ours || !x_derived.contains(&phi.dest) {
                    return EscapeClass::Escapes;
                }
            }
        }
        for inst in &block.instructions {
            match inst {
                // Alias-forming instructions already folded into x_derived.
                IrInstruction::GetElementPtr { .. }
                | IrInstruction::PtrAdd { .. }
                | IrInstruction::Cast { .. }
                | IrInstruction::BitCast { .. }
                | IrInstruction::SsaBarrier { .. }
                | IrInstruction::Copy { .. }
                | IrInstruction::Select { .. } => {}

                // Reads of / through x are not escapes.
                IrInstruction::Load { .. } => {}
                IrInstruction::Cmp { .. } | IrInstruction::BinOp { .. } => {}

                IrInstruction::Store { ptr, value, .. } => {
                    if in_x(value) {
                        if !adopt(resolve(ptr)) {
                            return EscapeClass::Escapes;
                        }
                    }
                    // A store INTO x mutates it; fine.
                }

                IrInstruction::CallDirect {
                    dest,
                    func_id,
                    args,
                    ..
                } => {
                    if !args.iter().any(|a| in_x(a)) {
                        continue;
                    }
                    if ids.copy_only_ids.contains(func_id) {
                        continue;
                    }
                    if let Some(callee_kinds) = info.kinds.get(func_id) {
                        let callee_leaks = info.leaks.get(func_id);
                        let leak_at =
                            |i: usize| callee_leaks.and_then(|l| l.get(i)).copied().unwrap_or(true);
                        let mut cand: Option<IrId> = None;
                        let mut ok = true;
                        for (i, arg) in args.iter().enumerate() {
                            if !in_x(arg) {
                                continue;
                            }
                            match callee_kinds.get(i).copied() {
                                // A borrowing callee that publishes one of
                                // x's children makes freeing x's OWNED graph
                                // unsound; x itself may still die, so this
                                // only blocks kinds of reasoning that follow
                                // x's fields -- but adoption frees x at its
                                // owner's death, and the child freed there is
                                // exactly what the callee published. Refuse.
                                Some(Retention::No) => {
                                    if leak_at(i) {
                                        ok = false;
                                    }
                                }
                                Some(Retention::IntoSelf) if i != 0 => {
                                    if leak_at(i) {
                                        ok = false;
                                    } else {
                                        cand = args.first().and_then(|a| resolve(a));
                                        if cand.is_none() {
                                            ok = false;
                                        }
                                    }
                                }
                                _ => ok = false,
                            }
                            if !ok {
                                break;
                            }
                        }
                        if !ok {
                            return EscapeClass::Escapes;
                        }
                        if let Some(c) = cand {
                            if !adopt(Some(c)) {
                                return EscapeClass::Escapes;
                            }
                        }
                    } else if ids.array_safe_ids.contains(func_id)
                        || ids.anon_safe_ids.contains(func_id)
                    {
                        // Receiver-safe accessors: x as arg0 is a borrow; x in
                        // a value position is stored into the receiver, whose
                        // liveness we do not model here.
                        if args.iter().skip(1).any(|a| in_x(a)) {
                            return EscapeClass::Escapes;
                        }
                    } else {
                        return EscapeClass::Escapes;
                    }
                }

                other => {
                    if other.uses().iter().any(|u| in_x(u)) {
                        return EscapeClass::Escapes;
                    }
                }
            }
        }
        match &block.terminator {
            IrTerminator::Return { value: Some(v) } if in_x(v) => return EscapeClass::Escapes,
            _ => {}
        }
    }
    if owner.is_none() && std::env::var_os("RZT_DBG_BENIGN").is_some() {
        eprintln!("[benign] {} x={x:?} exposures:", function.name);
        for block in function.cfg.blocks.values() {
            for inst in &block.instructions {
                let touches = inst.uses().iter().any(|u| in_x(u))
                    || matches!(inst, IrInstruction::CallDirect { args, .. } if args.iter().any(|a| in_x(a)));
                if touches {
                    eprintln!("[benign]   {:?}", inst);
                }
            }
            if let IrTerminator::Return { value: Some(v) } = &block.terminator {
                eprintln!("[benign]   terminator Return({v:?}) in_x={}", in_x(v));
            }
        }
    }
    match owner {
        Some(o) => EscapeClass::Owner(o),
        None => EscapeClass::Benign,
    }
}

/// A transferred child must not outlive the loop iteration its owner is
/// released in: every use must sit inside any loop containing its definition,
/// with no phi carrying it across iterations.
fn child_confined_ok(x: IrId, x_derived: &BTreeSet<IrId>, function: &IrFunction) -> bool {
    let Some(def_block) = def_block_of(function, x) else {
        return false;
    };
    let domtree = crate::ir::loop_analysis::DominatorTree::compute(function);
    let loop_info = crate::ir::loop_analysis::LoopNestInfo::analyze(function, &domtree);
    for natural in loop_info.loops_by_depth() {
        if !natural.blocks.contains(&def_block) {
            continue;
        }
        for (&bid, block) in &function.cfg.blocks {
            if !natural.blocks.contains(&bid)
                && block
                    .instructions
                    .iter()
                    .any(|i| i.uses().iter().any(|u| x_derived.contains(u)))
            {
                return false;
            }
            if block
                .phi_nodes
                .iter()
                .any(|p| p.incoming.iter().any(|(_, v)| x_derived.contains(v)))
            {
                return false;
            }
        }
    }
    true
}

/// Every value READ OUT of the owner must stay a read. A pointer loaded from
/// the owner's memory may alias a transferred child, so if any such value is
/// stored, returned, phi-carried or passed to a callee, freeing the child at
/// the owner's death could dangle it. Default-deny: only load/GEP chains,
/// comparisons and arithmetic are recognised as reads.
fn owner_loads_confined(
    owner_derived: &BTreeSet<IrId>,
    function: &IrFunction,
    ids: &AllocFuncIds,
    returns_alias: &BTreeMap<IrFunctionId, Vec<bool>>,
) -> bool {
    let read_out = read_out_closure(owner_derived, function, ids, returns_alias);
    if read_out.is_empty() {
        return true;
    }
    let in_read = |v: &IrId| read_out.contains(v);
    for block in function.cfg.blocks.values() {
        for phi in &block.phi_nodes {
            if phi.incoming.iter().any(|(_, v)| in_read(v)) {
                return false;
            }
        }
        for inst in &block.instructions {
            match inst {
                IrInstruction::Load { .. }
                | IrInstruction::GetElementPtr { .. }
                | IrInstruction::PtrAdd { .. }
                | IrInstruction::Cast { .. }
                | IrInstruction::BitCast { .. }
                | IrInstruction::SsaBarrier { .. }
                | IrInstruction::Copy { .. }
                | IrInstruction::Select { .. }
                | IrInstruction::Cmp { .. }
                | IrInstruction::BinOp { .. }
                | IrInstruction::UnOp { .. } => {}
                IrInstruction::Store { ptr, value, .. } => {
                    // Storing a read-out value anywhere re-homes a potential
                    // child alias; storing INTO a read-out pointer mutates the
                    // child, which is fine.
                    if in_read(value) {
                        return false;
                    }
                    let _ = ptr;
                }
                IrInstruction::CallDirect { func_id, args, .. } => {
                    if args.iter().any(|a| in_read(a)) && !ids.copy_only_ids.contains(func_id) {
                        return false;
                    }
                }
                other => {
                    if other.uses().iter().any(|u| in_read(u)) {
                        return false;
                    }
                }
            }
        }
        if let IrTerminator::Return { value: Some(v) } = &block.terminator {
            if in_read(v) {
                return false;
            }
        }
    }
    true
}
