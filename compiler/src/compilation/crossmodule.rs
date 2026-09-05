//! Re-keying an imported module's ids into this unit's id space.

use super::*;

impl CompilationUnit {

    /// Post-load fixup: resolve stale cross-module function references in all import modules.
    /// During renumbering, some refs couldn't be resolved because the target module hadn't
    /// been loaded yet. Now all modules are loaded and stdlib_function_name_map is complete.
    pub(crate) fn fixup_stale_cross_module_refs(&mut self) {
        use crate::ir::{IrInstruction, IrTerminator};

        // Build a set of all valid function IDs across all import modules
        // PLUS the main user file (`mir_modules`).
        //
        // We deliberately exclude `cached_stdlib_mir` here. That cache holds
        // the pre-merge / pre-renumber stdlib MIR — its ids do not exist in
        // the final merged module the codegen will see, and folding them
        // into `all_func_ids` artificially "retains" stale CallDirect
        // targets in the Sweep 1 retain step, which then bypass the
        // name-fallback resolution and surface as missing-function errors
        // at codegen.
        let mut all_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            std::collections::BTreeSet::new();
        for m in &self.import_mir_modules {
            all_func_ids.extend(m.functions.keys().copied());
            all_func_ids.extend(m.extern_functions.keys().copied());
        }
        for m in &self.mir_modules {
            all_func_ids.extend(m.functions.keys().copied());
            all_func_ids.extend(m.extern_functions.keys().copied());
        }

        // Also build a "forward-ref stub → name" map. A stub is an
        // `IrFunction` registered by `register_stdlib_mir_forward_ref` while
        // lowering a user file whose dispatch target wasn't yet compiled
        // (e.g. `Caller.probe` calls `h.findMeta` before `Holder.hx`'s
        // retry pass produced the real findMeta MIR). The stub exists in
        // `module.functions` (so `all_func_ids` contains it) but has
        // exactly one empty entry block with an `Unreachable` terminator
        // — it's a placeholder, not a callable. Once `Holder.hx`'s real
        // findMeta is registered in `stdlib_function_name_map`, we can
        // rewrite any CallDirect to the stub so it targets the real
        // function instead. Without this, the call survives merge as a
        // dispatch into the empty stub and the runtime jumps into
        // uninitialised code (UD2 / SIGILL).
        //
        // Key: name carried by the stub IrFunction (the qualified name
        // passed to `register_stdlib_mir_forward_ref`, e.g.
        // `pkg.Holder.findMeta`). Value: the stub's renumbered func_id.
        // For every IrFunctionId in any loaded module, if the function
        // is an empty forward-ref stub, record its qualified name. The
        // earlier version of this map was keyed BY-NAME (one entry per
        // name, first-found wins) — but when the same stub name was
        // registered in multiple modules (e.g. `string_concat` in both
        // an import module via cached MIR AND a user module via a
        // fresh `register_stdlib_mir_forward_ref` call during user-file
        // lowering), only the first stub's id was retained. Any
        // CallDirect pointing at the OTHER stub's id missed the rewrite
        // and remained pointed at the eventual safety-net trap stub.
        //
        // Keying by ID (every stub id → its name) and looking up the
        // current CallDirect's func_id directly avoids the
        // first-stub-wins miss. The id space is unique per
        // post-renumber session so there's no key collision.
        let mut stub_by_id: std::collections::BTreeMap<
            crate::ir::IrFunctionId,
            (String, crate::ir::IrFunctionSignature),
        > = std::collections::BTreeMap::new();
        // Candidates carry their FULL qualified name (3rd tuple element) so the
        // stub->real match can disambiguate by qualified name. This is essential
        // for constructors: every class's constructor has the bare name "new" and
        // many share the signature `(*void)->void`, so a bare-name+signature match
        // is ambiguous and was silently giving up — leaving e.g. a real
        // `haxe.ds.BalancedTree.new` stranded behind its forward-ref trap stub.
        let mut real_funcs_by_bare_name: std::collections::BTreeMap<
            String,
            Vec<(
                crate::ir::IrFunctionId,
                crate::ir::IrFunctionSignature,
                String,
            )>,
        > = std::collections::BTreeMap::new();

        fn is_empty_forward_ref_stub(func: &crate::ir::IrFunction) -> bool {
            func.cfg.blocks.len() == 1
                && func.cfg.blocks.values().all(|b| {
                    b.instructions.is_empty() && matches!(b.terminator, IrTerminator::Unreachable)
                })
        }

        fn bare_function_name(name: &str) -> &str {
            name.rsplit('.').next().unwrap_or(name)
        }

        fn effective_name(func: &crate::ir::IrFunction) -> String {
            func.qualified_name
                .clone()
                .unwrap_or_else(|| func.name.clone())
        }

        for m in &self.import_mir_modules {
            for (id, func) in &m.functions {
                if is_empty_forward_ref_stub(func) {
                    let name = func
                        .qualified_name
                        .clone()
                        .unwrap_or_else(|| func.name.clone());
                    stub_by_id.insert(*id, (name, func.signature.clone()));
                }
                if !func.cfg.blocks.is_empty() {
                    let qname = func.qualified_name.as_deref().unwrap_or(&func.name);
                    real_funcs_by_bare_name
                        .entry(bare_function_name(qname).to_string())
                        .or_default()
                        .push((*id, func.signature.clone(), qname.to_string()));
                }
            }
        }
        for m in &self.mir_modules {
            for (id, func) in &m.functions {
                if is_empty_forward_ref_stub(func) {
                    let name = func
                        .qualified_name
                        .clone()
                        .unwrap_or_else(|| func.name.clone());
                    stub_by_id.insert(*id, (name, func.signature.clone()));
                }
                if !func.cfg.blocks.is_empty() {
                    let qname = func.qualified_name.as_deref().unwrap_or(&func.name);
                    real_funcs_by_bare_name
                        .entry(bare_function_name(qname).to_string())
                        .or_default()
                        .push((*id, func.signature.clone(), qname.to_string()));
                }
            }
        }

        // ---- Rebind runtime-intrinsic forward-ref stubs to their extern symbol ----
        //
        // `register_stdlib_mir_forward_ref` builds its stub with a *1-block*
        // `Unreachable` cfg — `IrControlFlowGraph::new()` is NOT empty; it seeds
        // one entry block whose default terminator is `Unreachable`. For a
        // stdlib MIR *wrapper* the real body merges in later and replaces it.
        // But some stubs name a C-ABI RUNTIME symbol (`haxe_bytes_sub`,
        // `haxe_bytes_get`, `haxe_string_char_code_at_ptr`, …) registered in
        // the JIT symbol table (runtime/src/plugin_impl.rs) — there is no Haxe
        // body to merge. Codegen keys `is_extern` off `cfg.blocks.is_empty()`,
        // so a 1-block stub is NOT recognised as an extern: it is skipped at
        // definition and the finalize safety net installs a `udf #0xc11f` trap.
        // A call into one (e.g. `GGUFReader.parse` → `Bytes.sub` →
        // `haxe_bytes_sub`) then SIGILLs during GGUF load. The very same symbol
        // also appears as a genuine 0-block extern in another module (that copy
        // binds fine via `declare_function`'s `Import <name>` path) — the stub
        // copy just needs the same shape. Clear the stub's cfg so `is_extern`
        // holds and it binds to the runtime symbol like any extern.
        //
        // Gate strictly: only when the name ALSO exists as a true 0-block
        // extern somewhere (proof it is a runtime-bound symbol) AND has no real
        // (non-stub, non-empty) body anywhere (a real body means "redirect to
        // it", handled by the CallDirect rewrite below — not "bind to symbol").
        let mut extern_only_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut real_body_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        {
            let mut scan = |m: &crate::ir::IrModule| {
                for func in m.functions.values() {
                    if func.cfg.blocks.is_empty() {
                        extern_only_names.insert(effective_name(func));
                    } else if !is_empty_forward_ref_stub(func) {
                        real_body_names.insert(effective_name(func));
                    }
                }
                for ext in m.extern_functions.values() {
                    extern_only_names.insert(ext.name.clone());
                }
            };
            for m in &self.import_mir_modules {
                scan(m);
            }
            for m in &self.mir_modules {
                scan(m);
            }
        }
        let mut stub_ids_to_externize: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            std::collections::BTreeSet::new();
        for (id, (name, _sig)) in stub_by_id.iter() {
            if extern_only_names.contains(name) && !real_body_names.contains(name) {
                stub_ids_to_externize.insert(*id);
            }
        }
        // Drop the externized ids from the stub map so the CallDirect rewrite
        // leaves their call sites pointing at the (now extern) function instead
        // of trying to redirect them to another stub.
        for id in &stub_ids_to_externize {
            stub_by_id.remove(id);
        }
        if std::env::var("RAYZOR_DUMP_FN_PTRS").is_ok() && !stub_ids_to_externize.is_empty() {
            eprintln!(
                "[rebind-extern] {} runtime-intrinsic stub(s) rebound to extern symbol",
                stub_ids_to_externize.len()
            );
        }

        // Apply the rewrite to BOTH import_mir_modules and mir_modules
        // (the user modules). A user-file CallDirect like `Sys.println(...
        // + counter)` lowers to a CallDirect targeting the stub
        // `string_concat` registered via `register_stdlib_mir_forward_ref`
        // — when that stub's renumbered id lands on a value the codegen
        // backend's safety net then traps (see
        // bugs_sys_call_in_generation_method's continuation), the user-
        // module CallDirect site needs name-based resolution to point at
        // the real stdlib impl. The previous version rewrote only the
        // import side, leaving user CallDirects pointing at stubs that
        // would never get a body.
        let stub_by_id = &stub_by_id;
        let real_funcs_by_bare_name = &real_funcs_by_bare_name;
        let stdlib_map = &self.stdlib_function_name_map;
        let all_func_ids = &all_func_ids;
        fn signatures_match(
            a: &crate::ir::IrFunctionSignature,
            b: &crate::ir::IrFunctionSignature,
        ) -> bool {
            a.calling_convention == b.calling_convention
                && a.return_type == b.return_type
                && a.uses_sret == b.uses_sret
                && a.parameters.len() == b.parameters.len()
                && a.parameters
                    .iter()
                    .zip(b.parameters.iter())
                    .all(|(pa, pb)| pa.ty == pb.ty && pa.by_ref == pb.by_ref)
        }
        fn unique_bare_match(
            name: &str,
            sig: &crate::ir::IrFunctionSignature,
            real_funcs_by_bare_name: &std::collections::BTreeMap<
                String,
                Vec<(
                    crate::ir::IrFunctionId,
                    crate::ir::IrFunctionSignature,
                    String,
                )>,
            >,
            skip_id: Option<crate::ir::IrFunctionId>,
        ) -> Option<crate::ir::IrFunctionId> {
            let bare_name = bare_function_name(name);
            let candidates = real_funcs_by_bare_name.get(bare_name)?;
            // Qualified-name disambiguation FIRST. Constructors all share the bare
            // name "new", so bare+sig is ambiguous across classes; prefer a real
            // candidate whose FULL qualified name equals the stub's (the real
            // `haxe.ds.BalancedTree.new` rather than some other class's `new`).
            // The qname pins the class; the signature pins the overload. Take the
            // first such match — duplicates of one qname+sig are interchangeable.
            if let Some(real_id) = candidates.iter().find_map(|(cid, csig, cqname)| {
                if Some(*cid) != skip_id && cqname == name && signatures_match(csig, sig) {
                    Some(*cid)
                } else {
                    None
                }
            }) {
                return Some(real_id);
            }
            // Fall back to a UNIQUE bare-name + signature match.
            let mut matches = candidates
                .iter()
                .filter_map(|(candidate_id, candidate_sig, _)| {
                    if Some(*candidate_id) == skip_id {
                        return None;
                    }
                    if signatures_match(candidate_sig, sig) {
                        Some(*candidate_id)
                    } else {
                        None
                    }
                });
            let real_id = matches.next()?;
            if matches.next().is_none() {
                Some(real_id)
            } else {
                None
            }
        }

        let mut rewrite_module = |module: &mut crate::ir::IrModule| {
            let ext_names = module.external_function_names.clone();
            let ext_sigs: std::collections::BTreeMap<
                crate::ir::IrFunctionId,
                crate::ir::IrFunctionSignature,
            > = module
                .extern_functions
                .iter()
                .map(|(id, ext)| (*id, ext.signature.clone()))
                .collect();
            let ext_decl_by_id: std::collections::BTreeMap<
                crate::ir::IrFunctionId,
                (String, crate::ir::IrFunctionSignature),
            > = module
                .extern_functions
                .iter()
                .map(|(id, ext)| (*id, (ext.name.clone(), ext.signature.clone())))
                .collect();
            for func in module.functions.values_mut() {
                for block in func.cfg.blocks.values_mut() {
                    for inst in &mut block.instructions {
                        match inst {
                            IrInstruction::CallDirect { func_id, .. }
                            | IrInstruction::FunctionRef { func_id, .. }
                            | IrInstruction::MakeClosure { func_id, .. } => {
                                // If the cached MIR recorded this call site
                                // as an external reference (ext_names has an
                                // entry for the func_id), ALWAYS resolve it
                                // by name. The previous "only fix when id
                                // isn't valid" rule had a silent failure
                                // mode: the cached id from session A could
                                // happen to collide with a *different*
                                // function's id in session B (both at
                                // module-index 9, say) — the call would then
                                // dispatch into an unrelated function and
                                // SIGILL at runtime instead of resolving to
                                // the named target. Name-first resolution
                                // makes the fixup robust against any
                                // import-order-dependent id assignment.
                                if let Some(name) = ext_names.get(func_id) {
                                    if let Some(&current_id) = stdlib_map.get(name) {
                                        *func_id = current_id;
                                        continue;
                                    }
                                    if let Some(sig) = ext_sigs.get(func_id) {
                                        if let Some(real_id) = unique_bare_match(
                                            name,
                                            sig,
                                            real_funcs_by_bare_name,
                                            None,
                                        ) {
                                            *func_id = real_id;
                                        }
                                    }
                                    continue;
                                }
                                if let Some((name, sig)) = ext_decl_by_id.get(func_id) {
                                    if let Some(&current_id) = stdlib_map.get(name.as_str()) {
                                        *func_id = current_id;
                                        continue;
                                    }
                                    if let Some(real_id) =
                                        unique_bare_match(name, sig, real_funcs_by_bare_name, None)
                                    {
                                        *func_id = real_id;
                                    }
                                    continue;
                                }

                                // Not an external reference: leave the id
                                // alone if it's already valid. If not valid
                                // and the cached id corresponds to a known
                                // forward-ref stub by name, redirect to the
                                // real implementation. Otherwise leave as-is
                                // (will fault at runtime, surfacing the
                                // missing-impl error rather than silently
                                // dispatching to an unrelated function).
                                if all_func_ids.contains(func_id) {
                                    if let Some((stub_name, stub_sig)) = stub_by_id.get(func_id) {
                                        if let Some(&real_id) = stdlib_map.get(stub_name.as_str()) {
                                            if real_id != *func_id {
                                                *func_id = real_id;
                                            }
                                        } else {
                                            if let Some(real_id) = unique_bare_match(
                                                stub_name,
                                                stub_sig,
                                                real_funcs_by_bare_name,
                                                Some(*func_id),
                                            ) {
                                                *func_id = real_id;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        };

        // Clear the cfg of every runtime-intrinsic stub picked above so codegen
        // sees `is_extern` (empty blocks) and binds it to the runtime symbol.
        let externize = |module: &mut crate::ir::IrModule| {
            for (id, func) in module.functions.iter_mut() {
                if stub_ids_to_externize.contains(id) {
                    func.cfg.blocks.clear();
                }
            }
        };
        for module in &mut self.import_mir_modules {
            externize(module);
            rewrite_module(module);
        }
        // `mir_modules` is `Vec<Arc<IrModule>>` (sharable across the
        // pipeline). `Arc::make_mut` clones the inner module on demand if
        // anyone else holds a handle; in the codegen-prep window only the
        // CompilationContext owns these handles, so this is in-place.
        for module in self.mir_modules.iter_mut() {
            let m = std::sync::Arc::make_mut(module);
            externize(m);
            rewrite_module(m);
        }
    }


    /// Post-load fixup: rewrite stale constructor func_ids in
    /// `import_constructor_name_map`.
    ///
    /// `restore_cached_maps` seeds this map from BLADE cache with func_ids
    /// from the *saving* session — those ids encode that session's
    /// `import_base + local_id`, which doesn't necessarily match the
    /// current session's id assignment. Plain renumbering inside
    /// `renumber_and_push_import_mir` only rewrites entries whose ids live
    /// in *that* module's id_map, so a constructor cached at e.g. fn200007
    /// (StringBuf in a session where it was at import index 10) stays at
    /// fn200007 even when the current session puts StringBuf at index 2
    /// (real id fn120007). The user file then lowers `new StringBuf()` to
    /// `call fn200007` — an id that no longer exists — and SIGILLs.
    ///
    /// `stdlib_function_name_map` does carry the current ids of every
    /// merged function keyed by qualified name (`StringBuf.new`), so we
    /// rewrite each constructor map entry by name once all modules are in.
    pub(crate) fn fixup_stale_constructor_ids(&mut self) {
        // Snapshot the qualified lookup so we don't borrow self twice.
        let map_snapshot: std::collections::BTreeMap<String, crate::ir::IrFunctionId> =
            self.stdlib_function_name_map.clone();
        for (class_name, func_id) in self.import_constructor_name_map.iter_mut() {
            let qualified = format!("{}.new", class_name);
            if let Some(&current_id) = map_snapshot.get(&qualified) {
                if *func_id != current_id {
                    if let Some(count) = self.import_constructor_param_counts.remove(func_id) {
                        self.import_constructor_param_counts
                            .entry(current_id)
                            .or_insert(count);
                    } else if let Some(params) = self.import_function_param_types.get(&current_id) {
                        self.import_constructor_param_counts
                            .entry(current_id)
                            .or_insert(params.len());
                    }
                    *func_id = current_id;
                }
            }
        }
    }


    /// Post-load fixup: rewrite stale method func_ids in
    /// `stdlib_function_map` (SymbolId → IrFunctionId) by reverse-mapping
    /// each symbol back to its qualified name and re-resolving via
    /// `stdlib_function_name_map`.
    ///
    /// Same hazard as `fixup_stale_constructor_ids` but for non-constructors:
    /// `restore_cached_maps` seeds method-map values with the *cached*
    /// (pre-renumber) func_id; `renumber_and_push_import_mir`'s id_map-keyed
    /// rewrite only fires for entries whose ids live in the module currently
    /// being renumbered. Cross-session id drift caused by changes in the
    /// import-load order (e.g. adding/removing a top-level import shifts the
    /// 10_000-block assignment for every subsequent module) leaves
    /// stdlib_function_map pointing at a slot that no longer exists.
    ///
    /// `stdlib_function_name_map` carries the *current* ids of every merged
    /// function keyed by qualified name (e.g. "LlamaModel.forwardIds"), so we
    /// re-key each entry by its symbol's qualified name once all modules are
    /// loaded. Entries whose qualified name can't be reconstructed, or whose
    /// qualified name isn't present in the name map, are left untouched —
    /// the existing `fixup_stale_cross_module_refs` MIR walk catches stale
    /// CallDirect targets that survive here.
    pub(crate) fn fixup_stale_method_ids(&mut self) {
        let map_snapshot: std::collections::BTreeMap<String, crate::ir::IrFunctionId> =
            self.stdlib_function_name_map.clone();
        // Snapshot the (sym, id) pairs so we don't borrow self.symbol_table
        // while mutating self.stdlib_function_map.
        let entries: Vec<(crate::tast::SymbolId, crate::ir::IrFunctionId)> = self
            .stdlib_function_map
            .iter()
            .map(|(s, f)| (*s, *f))
            .collect();
        for (sym_id, _) in entries {
            // Resolve the symbol's qualified name. Symbols populated by
            // `restore_cached_maps` set qualified_name to the interned form
            // "Class.method"; freshly compiled methods do the same via
            // ast_lowering. Symbols without a qualified_name are not
            // safely re-resolvable by name and are skipped.
            let qname = {
                let Some(sym) = self.symbol_table.get_symbol(sym_id) else {
                    continue;
                };
                let Some(qn_interned) = sym.qualified_name else {
                    continue;
                };
                match self.string_interner.get(qn_interned) {
                    Some(s) => s.to_string(),
                    None => continue,
                }
            };
            if let Some(&current_id) = map_snapshot.get(&qname) {
                if let Some(slot) = self.stdlib_function_map.get_mut(&sym_id) {
                    if *slot != current_id {
                        *slot = current_id;
                    }
                }
            }
        }
    }


    /// Renumber import MIR function IDs to avoid collisions and push to import_mir_modules
    pub(crate) fn renumber_and_push_import_mir(&mut self, mut import_mir: IrModule) {
        use crate::ir::{IrFunctionId, IrGlobalId, IrInstruction};

        let import_base: u32 = 100_000 + (self.import_mir_modules.len() as u32 * 10_000);

        // Build old→new ID mapping (include both functions and extern_functions)
        let mut id_map: std::collections::BTreeMap<IrFunctionId, IrFunctionId> =
            std::collections::BTreeMap::new();
        for old_id in import_mir.functions.keys() {
            id_map.insert(*old_id, IrFunctionId(old_id.0 + import_base));
        }
        for old_id in import_mir.extern_functions.keys() {
            id_map
                .entry(*old_id)
                .or_insert(IrFunctionId(old_id.0 + import_base));
        }

        // Globals get the same disjoint-range treatment as functions: every
        // module numbers its globals densely from 0, so an unrenumbered
        // import's LoadGlobal/StoreGlobal aliases the main module's slots
        // 1:1 (observed: a user static read back Math's LN2 — each module's
        // __init__ wrote the same @g0/@g1). Backends key global storage by
        // raw id value, so sparse renumbered ids are safe.
        let mut global_id_map: std::collections::BTreeMap<IrGlobalId, IrGlobalId> =
            std::collections::BTreeMap::new();
        for old_id in import_mir.globals.keys() {
            global_id_map.insert(*old_id, IrGlobalId(old_id.0 + import_base));
        }

        // Renumber functions
        let old_functions: std::collections::BTreeMap<_, _> =
            std::mem::take(&mut import_mir.functions);
        for (old_id, mut func) in old_functions {
            let new_id = *id_map.get(&old_id).unwrap();
            func.id = new_id;

            // Update internal CallDirect/FunctionRef/MakeClosure. ext_names
            // takes priority over id_map: if the cached MIR recorded this
            // site as an external reference, resolve by name regardless of
            // whether the cached id happens to alias something in this
            // module's old id space. (Same robustness argument as the
            // fixup pass: integer ids are not stable across compilation
            // sessions with different import orderings.)
            for block in func.cfg.blocks.values_mut() {
                for inst in &mut block.instructions {
                    match inst {
                        IrInstruction::CallDirect { func_id, .. }
                        | IrInstruction::FunctionRef { func_id, .. }
                        | IrInstruction::MakeClosure { func_id, .. } => {
                            if let Some(name) = import_mir.external_function_names.get(func_id) {
                                if let Some(&current_id) = self.stdlib_function_name_map.get(name) {
                                    *func_id = current_id;
                                }
                                // If name lookup fails the post-pass
                                // fixup_stale_cross_module_refs will retry
                                // once all modules are loaded.
                            } else if let Some(new_func_id) = id_map.get(func_id) {
                                *func_id = *new_func_id;
                            }
                        }
                        IrInstruction::LoadGlobal { global_id, .. }
                        | IrInstruction::StoreGlobal { global_id, .. } => {
                            if let Some(&new_gid) = global_id_map.get(global_id) {
                                *global_id = new_gid;
                            }
                        }
                        _ => {}
                    }
                }
            }

            import_mir.functions.insert(new_id, func);
        }

        // Renumber extern_functions
        let old_externs: std::collections::BTreeMap<_, _> =
            std::mem::take(&mut import_mir.extern_functions);
        for (old_id, mut efunc) in old_externs {
            let new_id = id_map
                .get(&old_id)
                .copied()
                .unwrap_or(IrFunctionId(old_id.0 + import_base));
            efunc.id = new_id;
            import_mir.extern_functions.insert(new_id, efunc);
        }

        // Renumber the globals table to match the rewritten instructions.
        let old_globals: std::collections::BTreeMap<_, _> = std::mem::take(&mut import_mir.globals);
        for (old_id, mut g) in old_globals {
            let new_id = *global_id_map.get(&old_id).unwrap();
            g.id = new_id;
            import_mir.globals.insert(new_id, g);
        }

        // Re-key the module's name records to the renumbered ids. These
        // entries are the qualified-name ground truth for every later
        // name-first repair (fixup passes, merge verification); leaving them
        // keyed by pre-renumber ids detaches them from the instructions and
        // silently degrades cross-module resolution to raw-number trust.
        let old_ext_names = std::mem::take(&mut import_mir.external_function_names);
        for (old_id, name) in old_ext_names {
            let new_id = id_map.get(&old_id).copied().unwrap_or(old_id);
            import_mir.external_function_names.insert(new_id, name);
        }

        // Update all accumulated maps to point to renumbered IDs
        for (_sym, func_id) in self.stdlib_function_map.iter_mut() {
            if let Some(&new_id) = id_map.get(func_id) {
                *func_id = new_id;
            }
        }
        for (_name, func_id) in self.stdlib_function_name_map.iter_mut() {
            if let Some(&new_id) = id_map.get(func_id) {
                *func_id = new_id;
            }
        }
        for (_name, func_id) in self.import_constructor_name_map.iter_mut() {
            if let Some(&new_id) = id_map.get(func_id) {
                *func_id = new_id;
            }
        }
        // Re-key import_function_param_iface_names from pre-renumber to
        // post-renumber func_ids. Drain into a temp so we don't iterate
        // and mutate the same map.
        let stale_iface_names = std::mem::take(&mut self.import_function_param_iface_names);
        for (old_id, names) in stale_iface_names {
            let new_id = id_map.get(&old_id).copied().unwrap_or(old_id);
            self.import_function_param_iface_names.insert(new_id, names);
        }

        // Keep MIR-lowering lookup inputs incrementally accumulated. The old
        // path rebuilt these maps by scanning every previous import module for
        // every next import/user file, which made cold source compilation scale
        // quadratically on large graphs like nue/llama-chat.
        let constructor_ids: std::collections::BTreeSet<_> =
            self.import_constructor_name_map.values().copied().collect();
        for (func_id, func) in &import_mir.functions {
            self.import_function_param_types.insert(
                *func_id,
                func.signature
                    .parameters
                    .iter()
                    .map(|p| p.ty.clone())
                    .collect(),
            );
            if constructor_ids.contains(func_id) {
                self.import_constructor_param_counts
                    .entry(*func_id)
                    .or_insert(func.signature.parameters.len());
            }
        }
        for global in import_mir.globals.values() {
            self.import_external_globals
                .entry(global.name.clone())
                .or_insert((global.id, global.ty.clone()));
        }

        // Populate `stdlib_function_name_map` with this import's functions so
        // downstream callers can resolve them by qualified name. The cache-hit
        // path in `try_load_blade_cached_full` already does this (line ~2988);
        // the fresh-compile path (this function) used to skip it, leaving
        // user-package methods unreachable by name from callers in other
        // files. Manifested as `loader.someMethod(...)` silently lowering to
        // `unreachable` even though the function existed in MIR — the
        // resolver in `hir_to_mir.rs::resolve_function_id_with_qualified_fallback`
        // falls through to this map as a last resort.
        for (func_id, func) in &import_mir.functions {
            if func.cfg.blocks.is_empty() {
                continue;
            }
            let map_name = func.qualified_name.as_deref().unwrap_or(&func.name);
            // Don't overwrite an existing entry — first writer wins to keep
            // BLADE-cache and fresh-compile entries from clobbering each other.
            self.stdlib_function_name_map
                .entry(map_name.to_string())
                .or_insert(*func_id);
        }

        self.import_mir_modules.push(import_mir);
    }
}
