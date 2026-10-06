//! Linking source modules and runtime wrappers into the entry module.

use super::*;

impl CompilationUnit {
    pub(crate) fn merge_stdlib_into_user_module(
        &mut self,
        mir_module: &mut IrModule,
        own_func_ids: &BTreeSet<IrFunctionId>,
    ) {
        let stdlib_mapping = self.compiler_plugin_registry.build_combined_mapping();
        // Merge stdlib MIR (extern functions for Thread, Channel, Mutex, Arc, etc.)
        // This ensures extern runtime functions are available.
        // Uses build_stdlib_with_plugins to include HDLL extern declarations from loaded plugins.
        // Cache the stdlib MIR module to avoid rebuilding it for each user file.
        use crate::stdlib::build_stdlib_with_plugins;
        let mut stdlib_mir = if let Some(ref cached) = self.cached_stdlib_mir {
            cached.clone()
        } else {
            let mir = build_stdlib_with_plugins(&self.compiler_plugin_registry);
            self.cached_stdlib_mir = Some(mir.clone());
            mir
        };

        // Merge on-demand imported MIR modules (e.g., BalancedTree.hx, Point2D.hx)
        // into the user module. These were already renumbered to high IDs (100000+)
        // during load_imports_efficiently, so they won't collide with user or stdlib IDs.
        // Import modules now skip the stdlib merge, so they contain both:
        // (a) source-level declarations (tracked in import_own_func_ids) — protect these
        // (b) generated MIR wrappers for stdlib calls — let stdlib merge replace these
        let mut merged_import_func_ids: std::collections::BTreeSet<IrFunctionId> =
            own_func_ids.clone();
        // Sort import modules by name for deterministic merge order.
        // Sorting ensures the merged MIR is identical regardless of resolver order.
        self.import_mir_modules.sort_by(|a, b| a.name.cmp(&b.name));
        for import_module in self.import_mir_modules.drain(..) {
            // Merge import type definitions so runtime RTTI registration includes
            // imported classes/enums (needed for uncaught exception formatting and
            // hierarchy-aware typed catches).
            for (_old_type_def_id, mut typedef) in import_module.types {
                let new_type_def_id = mir_module.alloc_typedef_id();
                typedef.id = new_type_def_id;
                mir_module.types.insert(new_type_def_id, typedef);
            }
            // Imported globals were renumbered into a disjoint id range
            // (renumber_and_push_import_mir); carry them into the merged
            // module so its globals table matches the instructions.
            // Previously they were dropped entirely while the imports'
            // LoadGlobal/StoreGlobal kept their dense-from-0 ids —
            // aliasing the main module's statics slot-for-slot.
            for (global_id, global) in import_module.globals {
                if let Some(prev) = mir_module.globals.get(&global_id) {
                    if prev.name != global.name {
                        panic!(
                            "MIR merge id collision: global id {:?} is '{}' but module '{}' \
                             provides '{}' at the same id.",
                            global_id, prev.name, import_module.name, global.name
                        );
                    }
                }
                mir_module.globals.insert(global_id, global);
            }
            for (func_id, func) in import_module.functions {
                // Only protect source-level declarations (methods, constructors).
                // MIR wrappers (not in import_own_func_ids) can be replaced by stdlib.
                //
                // `import_own_func_ids` is keyed by renumbered IrFunctionId, computed
                // independently per import module via its own local-id arithmetic
                // (`old_id + import_base`). A stdlib MIR-wrapper stub for a bodyless
                // `extern class` method (e.g. `Tensor.addInto` -> `Tensor_addInto`) can
                // land on a renumbered id that coincidentally collides with an unrelated
                // genuine "own" declaration from another import — confirmed via
                // `RAYZOR_DBG_LOAD` tracing: `Tensor_addInto`'s renumbered id showed up
                // `kind=MirWrapper` yet was still `protected=true`, and its name never
                // appeared in the `own_func_ids` filter's input set at all, so the
                // protection wasn't coming from THIS function's own membership. Guard
                // against that class of ID collision with a NAME-based veto: never
                // protect an id whose function is a known stdlib MIR-wrapper name,
                // regardless of why the id ended up in `import_own_func_ids`.
                let is_known_stdlib_wrapper = stdlib_mapping.is_mir_wrapper_function(&func.name);
                if self.import_own_func_ids.contains(&func_id) && !is_known_stdlib_wrapper {
                    merged_import_func_ids.insert(func_id);
                }
                // Import id ranges are disjoint by construction (per-module
                // base + stride); an occupied slot with a DIFFERENT name is
                // an id-space collision. Last-writer-wins here silently
                // rebinds every call site of the loser to an unrelated
                // function (observed: a SpinPool worker's stdlib-extern
                // call dispatching into rayzor_channel_receive once the
                // import count pushed a module's base into another range).
                if let Some(prev) = mir_module.functions.get(&func_id) {
                    let prev_name = prev.qualified_name.as_deref().unwrap_or(&prev.name);
                    let new_name = func.qualified_name.as_deref().unwrap_or(&func.name);
                    if prev_name != new_name {
                        panic!(
                            "MIR merge id collision: function id {:?} is '{}' but module '{}' \
                             provides '{}' at the same id. Import id ranges overlapped — this \
                             build would silently misroute calls.",
                            func_id, prev_name, import_module.name, new_name
                        );
                    }
                }
                mir_module.functions.insert(func_id, func);
            }
            for (func_id, extern_func) in import_module.extern_functions {
                if let Some(prev) = mir_module.extern_functions.get(&func_id) {
                    if prev.name != extern_func.name {
                        panic!(
                            "MIR merge id collision: extern id {:?} is '{}' but module '{}' \
                             provides '{}' at the same id.",
                            func_id, prev.name, import_module.name, extern_func.name
                        );
                    }
                }
                mir_module.extern_functions.insert(func_id, extern_func);
            }
            // Carry the import's name records into the merged module so the
            // post-merge fixup/verification passes keep their ground truth.
            for (func_id, name) in import_module.external_function_names {
                mir_module
                    .external_function_names
                    .entry(func_id)
                    .or_insert(name);
            }
        }

        // Resolve name-keyed forward-ref dispatch-thunk stubs. A file that
        // constructs an imported class as an interface (e.g.
        // `ArchRegistry.withDefaults` doing `new LlamaArch()` when
        // `LlamaArch` compiles AFTER it) emits an EMPTY
        // `__vtable_dispatch_thunk__<class>_<method>` stub as a placeholder,
        // trusting the real thunk (same name, from the class's own module)
        // to be merged in. Both now coexist by different ids; the stub's
        // `FunctionRef` in the fat-ptr slot still points at the EMPTY stub
        // (calling it traps — SIGTRAP). Redirect every ref from an empty
        // `__vtable_dispatch_thunk__*` stub to the real, non-empty
        // same-named thunk, then drop the stub. Order-independent: works
        // regardless of which file compiled first.
        {
            let mut thunk_real: std::collections::BTreeMap<String, IrFunctionId> =
                std::collections::BTreeMap::new();
            let mut thunk_stubs: Vec<(String, IrFunctionId)> = Vec::new();
            for (func_id, func) in &mir_module.functions {
                // Any bodyless function shadowing a compiled one of the same
                // identity has the same problem the thunks had: a call bound
                // to the stub traps at runtime. A declaration restored from
                // the symbol manifest produces exactly that for a method
                // whose class is also compiled here. An extern keeps its own
                // name, so nothing non-empty answers to it and it is left be.
                let identity = func
                    .qualified_name
                    .clone()
                    .unwrap_or_else(|| func.name.clone());
                if func.cfg.blocks.is_empty()
                    || func.cfg.blocks.values().all(|b| b.instructions.is_empty())
                {
                    thunk_stubs.push((identity, *func_id));
                } else {
                    // Prefer the first real (non-empty) definition per name.
                    thunk_real.entry(identity).or_insert(*func_id);
                }
            }
            let mut thunk_replacements: std::collections::BTreeMap<IrFunctionId, IrFunctionId> =
                std::collections::BTreeMap::new();
            for (name, stub_id) in &thunk_stubs {
                if let Some(&real_id) = thunk_real.get(name) {
                    if real_id != *stub_id {
                        thunk_replacements.insert(*stub_id, real_id);
                    }
                }
            }
            if !thunk_replacements.is_empty() {
                for (_, caller_func) in mir_module.functions.iter_mut() {
                    for block in caller_func.cfg.blocks.values_mut() {
                        for instr in &mut block.instructions {
                            match instr {
                                IrInstruction::CallDirect { func_id, .. }
                                | IrInstruction::FunctionRef { func_id, .. }
                                | IrInstruction::MakeClosure { func_id, .. } => {
                                    if let Some(&new_id) = thunk_replacements.get(func_id) {
                                        *func_id = new_id;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                // Drop the now-unreferenced empty stubs.
                for stub_id in thunk_replacements.keys() {
                    mir_module.functions.remove(stub_id);
                }
            }
        }

        // CRITICAL FIX: Renumber stdlib function IDs to avoid collisions with user functions
        // Each MIR module starts function IDs from 0, so when merging stdlib and user modules,
        // IDs will collide. For example:
        //   - User module: IrFunctionId(2) = "indexOf"
        //   - Stdlib module: IrFunctionId(2) = "free"
        // Without renumbering, stdlib's "free" would be skipped, causing vec_u8_free to call "indexOf"!

        // Find the maximum function ID in the user module
        let max_user_func_id = mir_module
            .functions
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0);

        let max_user_extern_id = mir_module
            .extern_functions
            .keys()
            .map(|id| id.0)
            .max()
            .unwrap_or(0);

        let offset = std::cmp::max(max_user_func_id, max_user_extern_id) + 1;

        debug!(
            "DEBUG: Renumbering stdlib functions with offset {} (max_user_func={}, max_user_extern={})",
            offset, max_user_func_id, max_user_extern_id
        );

        // Build map of function names to ALL IDs in the user module (before merging).
        // Multiple import modules can have duplicate extern declarations of the same function.
        // We need to track ALL of them to replace every copy.
        let mut user_func_name_to_ids: BTreeMap<String, Vec<IrFunctionId>> = BTreeMap::new();
        for (func_id, func) in &mir_module.functions {
            user_func_name_to_ids
                .entry(func.name.clone())
                .or_default()
                .push(*func_id);
        }

        // The full stdlib/runtime MIR module is large, and most source cold
        // runs only need the wrappers that replace stubs already emitted in
        // the user/import MIR. Keep those roots plus their transitive direct
        // function references; tree-shake still runs later as the semantic
        // safety net.
        if std::env::var_os("RAYZOR_FULL_STDLIB_MERGE").is_none() {
            let needed_stdlib_names: BTreeSet<String> = user_func_name_to_ids
                .iter()
                .filter(|(_, existing_ids)| {
                    existing_ids
                        .iter()
                        .any(|id| !merged_import_func_ids.contains(id))
                })
                .map(|(name, _)| name.clone())
                .collect();
            let dropped = retain_referenced_stdlib_functions(&mut stdlib_mir, &needed_stdlib_names);
            debug!(
                "DEBUG: Selective stdlib merge dropped {} unreferenced functions",
                dropped
            );
        }

        // Build mapping of old stdlib IDs to new renumbered IDs
        use std::collections::BTreeMap;
        let mut id_mapping: BTreeMap<IrFunctionId, IrFunctionId> = BTreeMap::new();

        // Note: extern_functions is not used - externs are in the functions map with empty CFGs
        // So we only need to renumber the functions map

        // FIRST PASS: Build complete ID mapping for all stdlib functions
        // We must do this BEFORE updating CallDirect instructions so that all IDs are available
        for (old_id, _) in &stdlib_mir.functions {
            let new_id = IrFunctionId(old_id.0 + offset);
            id_mapping.insert(*old_id, new_id);
        }

        // SECOND PASS: Renumber functions and update their internal references
        let mut renumbered_functions = BTreeMap::new();
        for (old_id, mut func) in stdlib_mir.functions {
            let new_id = *id_mapping.get(&old_id).unwrap();

            // Update the function's own ID
            func.id = new_id;

            // Update all function ID references in instructions (CallDirect, FunctionRef, MakeClosure)
            use crate::ir::IrInstruction;
            for block in func.cfg.blocks.values_mut() {
                for inst in &mut block.instructions {
                    match inst {
                        IrInstruction::CallDirect { func_id, .. } => {
                            if let Some(&new_func_id) = id_mapping.get(func_id) {
                                debug!(
                                    "DEBUG: Updated CallDirect in {} from func_id {} -> {}",
                                    func.name, func_id.0, new_func_id.0
                                );
                                *func_id = new_func_id;
                            }
                        }
                        IrInstruction::FunctionRef { func_id, .. } => {
                            if let Some(&new_func_id) = id_mapping.get(func_id) {
                                debug!(
                                    "DEBUG: Updated FunctionRef in {} from func_id {} -> {}",
                                    func.name, func_id.0, new_func_id.0
                                );
                                *func_id = new_func_id;
                            }
                        }
                        IrInstruction::MakeClosure { func_id, .. } => {
                            if let Some(&new_func_id) = id_mapping.get(func_id) {
                                debug!(
                                    "DEBUG: Updated MakeClosure in {} from func_id {} -> {}",
                                    func.name, func_id.0, new_func_id.0
                                );
                                *func_id = new_func_id;
                            }
                        }
                        _ => {}
                    }
                }
            }

            renumbered_functions.insert(new_id, func);
            debug!(
                "DEBUG: Renumbered function '{}': {} -> {}",
                renumbered_functions[&new_id].name, old_id.0, new_id.0
            );
        }

        // Merge renumbered stdlib functions - no collisions possible now!
        // (Note: extern functions are included in the functions map with empty CFGs)
        //
        // IMPORTANT: Replace user functions that have the same NAME as stdlib functions
        // The user module might have extern declarations (e.g. rayzor_channel_init) from
        // the lowering process, but these might have incorrect signatures due to type
        // inference issues. The stdlib version is the source of truth, so we REPLACE
        // the user's version with the stdlib's version.

        // Build a map of old ID -> new ID for all replacements
        let mut id_replacements: BTreeMap<IrFunctionId, IrFunctionId> = BTreeMap::new();

        // Replacement is by bare name, and a stdlib import's methods are
        // not protected: a body-less extern in the stdlib module must not
        // take the place of a function that has a body (`copy` on Int64
        // against a `copy` stub), or every caller traps at runtime.
        let bodied_existing: BTreeSet<IrFunctionId> = mir_module
            .functions
            .iter()
            .filter(|(_, f)| !f.cfg.blocks.is_empty())
            .map(|(id, _)| *id)
            .collect();
        let keeps_body = |existing_id: &IrFunctionId, stdlib_func: &crate::ir::IrFunction| {
            stdlib_func.cfg.blocks.is_empty() && bodied_existing.contains(existing_id)
        };
        // Nor may a bodied method of another type that shares the bare
        // name (`pk.Int64.make` beside `haxe.Int64.make`). Qualified names
        // compare up to qualification: `Int64.make` is `haxe.Int64.make`.
        let qualified_by_id: BTreeMap<IrFunctionId, String> = mir_module
            .functions
            .iter()
            .filter_map(|(id, f)| Some((*id, f.qualified_name.clone()?)))
            .collect();
        let other_owner = |existing_id: &IrFunctionId, stdlib_func: &crate::ir::IrFunction| {
            let (Some(mine), Some(theirs)) = (
                qualified_by_id.get(existing_id),
                stdlib_func.qualified_name.as_deref(),
            ) else {
                return false;
            };
            let suffix_of = |long: &str, short: &str| {
                long == short
                    || long
                        .strip_suffix(short)
                        .is_some_and(|head| head.ends_with('.'))
            };
            bodied_existing.contains(existing_id)
                && !suffix_of(mine, theirs)
                && !suffix_of(theirs, mine)
        };

        for (func_id, func) in &renumbered_functions {
            if let Some(existing_ids) = user_func_name_to_ids.get(&func.name) {
                for &existing_id in existing_ids {
                    // Protect source-level import functions from stdlib replacement.
                    if !merged_import_func_ids.contains(&existing_id)
                        && !keeps_body(&existing_id, func)
                        && !other_owner(&existing_id, func)
                    {
                        id_replacements.insert(existing_id, *func_id);
                    } else if func.name == "match" || func.name == "matched" {
                        eprintln!(
                            "[MERGE_PROTECT] fn{}={} protected from fn{}",
                            existing_id.0, func.name, func_id.0
                        );
                    }
                }
            }
        }

        // Now merge the stdlib functions
        for (func_id, func) in renumbered_functions {
            // If this function replaces existing ones, remove old copies
            // Remove stubs, but protect user package import functions
            if let Some(existing_ids) = user_func_name_to_ids.get(&func.name) {
                for &existing_id in existing_ids {
                    if !merged_import_func_ids.contains(&existing_id)
                        && !keeps_body(&existing_id, &func)
                    {
                        mir_module.functions.remove(&existing_id);
                    }
                }
            }

            mir_module.functions.insert(func_id, func);
            // Keep next_function_id in sync so alloc_function_id() won't collide
            mir_module.next_function_id = mir_module.next_function_id.max(func_id.0 + 1);
        }

        // Update ALL instructions that reference replaced function IDs
        // This is done AFTER all merging to avoid ID conflicts
        if !id_replacements.is_empty() {
            for (_, caller_func) in mir_module.functions.iter_mut() {
                for block in caller_func.cfg.blocks.values_mut() {
                    for instr in &mut block.instructions {
                        match instr {
                            IrInstruction::CallDirect {
                                func_id: called_func_id,
                                ..
                            } => {
                                if let Some(&new_id) = id_replacements.get(called_func_id) {
                                    *called_func_id = new_id;
                                }
                            }
                            IrInstruction::FunctionRef {
                                func_id: ref_func_id,
                                ..
                            } => {
                                if let Some(&new_id) = id_replacements.get(ref_func_id) {
                                    debug!(
                                        "DEBUG: Updated FunctionRef in {} from func_id {} -> {}",
                                        caller_func.name, ref_func_id.0, new_id.0
                                    );
                                    *ref_func_id = new_id;
                                }
                            }
                            IrInstruction::MakeClosure {
                                func_id: closure_func_id,
                                ..
                            } => {
                                if let Some(&new_id) = id_replacements.get(closure_func_id) {
                                    debug!(
                                        "DEBUG: Updated MakeClosure in {} from func_id {} -> {}",
                                        caller_func.name, closure_func_id.0, new_id.0
                                    );
                                    *closure_func_id = new_id;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Refresh the global name map after the merge.
            //
            // When the user file lowered its own MIR, every pre-merge
            // function (including MirWrapper stubs like `array_length`,
            // `Tensor_mul`, `rayzor_anon_set_field_by_index`, …) was
            // seeded into `stdlib_function_name_map` at this file's
            // pre-merge id (e.g. fn12). The stdlib merge above just
            // deleted those stub IrFunctions and replaced them with
            // renumbered real bodies (e.g. fn640115), updating every
            // CallDirect / FunctionRef / MakeClosure in the module via
            // `id_replacements`. But the name map still points at the
            // dead pre-merge ids — so the later
            // `fixup_stale_cross_module_refs` pass, when it looks up a
            // cross-module CallDirect by external-function name, gets
            // back the dead id and writes it over a valid renumbered
            // target, surfacing as a codegen "undefined function fnN"
            // error. Apply the same renumber to the name map so it
            // tracks the merged module.
            for id in self.stdlib_function_name_map.values_mut() {
                if let Some(&new_id) = id_replacements.get(id) {
                    *id = new_id;
                }
            }
        }

        // Verify MIR wrapper forward refs were replaced during merge.
        // A MirWrapper function with an empty CFG means the stdlib merge failed
        // to find the implementation — this would cause wrong values at runtime.
        if cfg!(debug_assertions) {
            for (func_id, func) in &mir_module.functions {
                if func.cfg.blocks.is_empty()
                    && !matches!(func.kind, crate::ir::FunctionKind::ExternC)
                {
                    eprintln!(
                        "warning: empty function body after stdlib merge: '{}' (ID {}) kind={:?}",
                        func.name, func_id.0, func.kind
                    );
                }
            }
        }
    }
}
