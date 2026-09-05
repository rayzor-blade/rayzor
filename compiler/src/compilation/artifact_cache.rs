//! The on-disk BLADE artifact cache: where a compiled module is stored,
//! what keys it, and how its maps are rebuilt on a hit.

use super::*;

impl CompilationUnit {

    // === BLADE Caching Methods ===

    /// Get the BLADE cache path for a source file.
    ///
    /// The cache filename is the module's fully-qualified name with `.blade`
    /// appended (e.g. `nue.transformer.GQAttention.blade`). Two strategies
    /// are tried in order:
    ///
    /// 1. **Known root strip** — `haxe-std/` or `/src/` in the path. Covers
    ///    bundled stdlib and the common `src/`-rooted layout.
    ///
    /// 2. **Project class-path strip** — match the source against each
    ///    configured source path (from rayzor.toml `class-paths`) and strip
    ///    the longest matching prefix. Without this, projects whose Haxe
    ///    sources don't live under `src/` (e.g. nue's `class-paths = ["."]`
    ///    with files at `nue/nue/transformer/GQAttention.hx`) cache by
    ///    bare filename only — `GQAttention.blade` — and short-name
    ///    collisions between packages silently overwrite each other.
    ///
    /// Falls back to the filename only if nothing matches; that path is the
    /// least desirable because it loses the package and risks collisions.
    pub(crate) fn blade_cache_path(&self, source_path: &str) -> Option<PathBuf> {
        let cache_dir = if self.is_stdlib_source(source_path) {
            self.config.get_stdlib_cache_dir()
        } else {
            self.config.get_cache_dir()
        };
        let normalized = source_path.replace('\\', "/");

        let module_part: String = if let Some(pos) = normalized.rfind("haxe-std/") {
            normalized[pos + 9..].to_string()
        } else if let Some(pos) = normalized.rfind("/src/") {
            normalized[pos + 5..].to_string()
        } else {
            // Try stripping a project class-path (rayzor.toml `class-paths`).
            // Pick the longest matching prefix so nested roots resolve in
            // favour of the more-specific one.
            let stripped = {
                let abs = std::path::Path::new(&normalized)
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.to_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| normalized.clone());
                let mut best: Option<String> = None;
                for root in self.namespace_resolver.get_source_paths() {
                    let root_str = root
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| root.to_string_lossy().to_string());
                    let root_with_slash = if root_str.ends_with('/') {
                        root_str.clone()
                    } else {
                        format!("{root_str}/")
                    };
                    if abs.starts_with(&root_with_slash) {
                        let candidate = abs[root_with_slash.len()..].to_string();
                        if best
                            .as_ref()
                            .map(|b| candidate.len() < b.len())
                            .unwrap_or(true)
                        {
                            best = Some(candidate);
                        }
                    }
                }
                best
            };
            stripped.unwrap_or_else(|| {
                normalized
                    .rsplit('/')
                    .next()
                    .unwrap_or(&normalized)
                    .to_string()
            })
        };

        let module_name = module_part.replace('/', ".").replace(".hx", "");

        if module_name.is_empty() {
            return None;
        }

        Some(cache_dir.join(format!("{}.blade", module_name)))
    }


    /// This module's artifact in the standard library carried by the binary,
    /// if it holds one for the configuration being compiled.
    pub(crate) fn embedded_snapshot_entry(&self, source_path: &str) -> Option<&'static [u8]> {
        if !self.is_stdlib_source(source_path) {
            return None;
        }
        // The generator that PRODUCES the carried library must not consume it:
        // restoring a module writes no artifact, so a build whose snapshot is
        // already valid would regenerate an empty one.
        if std::env::var_os("RAYZOR_IGNORE_EMBEDDED_SNAPSHOT").is_some() {
            return None;
        }
        let file = self.blade_cache_path(source_path)?;
        let file = file.file_name()?.to_str()?;
        let key = crate::ir::snapshot::key_for(&self.config.stdlib_cache_discriminator(), file);
        crate::ir::snapshot::installed().get(key.as_str()).copied()
    }


    /// Where a module's artifact lives in the PREPARED store: a shared cache
    /// written once by `rayzor cache warm` rather than per project.
    ///
    /// The standard library lowers to the same thing for every program, so a
    /// project that has never been built can start from a prepared artifact
    /// instead of compiling the library again. Entries are keyed by the same
    /// discriminator as the project cache, and validated on load by the same
    /// build id, so an artifact from a different compiler is rejected rather
    /// than decoded.
    pub(crate) fn prepared_blade_path(&self, source_path: &str) -> Option<PathBuf> {
        let root = Self::prepared_cache_root()?;
        let file = self.blade_cache_path(source_path)?.file_name()?.to_owned();
        let discriminator = if self.is_stdlib_source(source_path) {
            self.config.stdlib_cache_discriminator()
        } else {
            self.config.cache_discriminator()
        };
        Some(root.join(discriminator).join(file))
    }


    /// Root of the shared prepared store. `RAYZOR_PREPARED_CACHE` overrides it.
    pub fn prepared_cache_root() -> Option<PathBuf> {
        if let Some(explicit) = std::env::var_os("RAYZOR_PREPARED_CACHE") {
            return Some(PathBuf::from(explicit));
        }
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/rayzor/blade"))
    }


    /// Compute the BLADE source fingerprint for this compilation context.
    ///
    /// Source text alone is not enough: `extra_defines` changes `#if`
    /// lowering, so native and wasm builds of the same file must not share
    /// the same per-module artifact.
    /// Identity hash of the USER PROGRAM being compiled (all user source
    /// files: filename + content). A `.blade` module cache stores PROGRAM-
    /// SPECIFIC state — the global function-id renumbering, class memory
    /// layout, GENERATED reflection ctor wrappers, and inherited-field tables
    /// are all assigned relative to the full module set of the program that
    /// produced it. Folding this into every module's cache key (below) makes
    /// a cached module reusable ONLY for the same program: re-running the same
    /// program hits; any edit to a user source file (or a different program
    /// entirely, e.g. the next test in the suite) misses and recompiles.
    pub(crate) fn user_program_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        "rayzor-program-identity-v1".hash(&mut h);
        for f in &self.user_files {
            f.filename.hash(&mut h);
            if let Some(src) = &f.input {
                src.hash(&mut h);
            }
        }
        h.finish()
    }


    /// Whether a source file belongs to the standard library.
    pub(crate) fn is_stdlib_source(&self, source_path: &str) -> bool {
        let path = Path::new(source_path);
        let canonical = path.canonicalize();
        let candidate = canonical.as_deref().unwrap_or(path);
        self.config
            .stdlib_paths
            .iter()
            .any(|root| match root.canonicalize() {
                // A root that does not exist cannot contain the file.
                Ok(root) => candidate.starts_with(&root),
                Err(_) => false,
            })
    }


    pub(crate) fn hash_source_for_config(&self, source_path: &str, source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        "rayzor-blade-source-v4".hash(&mut hasher);
        source.hash(&mut hasher);
        // A standard-library module counts only the defines it can observe, for
        // the same reason its cache directory does: a flag it never names
        // cannot change what it lowers to, and counting the rest means an
        // unrelated one (a plugin contributes a define per plugin) invalidates
        // the whole library.
        let mut defines: Vec<String> = if self.is_stdlib_source(source_path) {
            self.config.stdlib_relevant_defines()
        } else {
            self.config.extra_defines.clone()
        };
        defines.sort();
        defines.hash(&mut hasher);
        // A cached USER module carries state assigned relative to the whole
        // program (id renumbering, class layout, reflection ctor wrappers,
        // inherited fields). Two different programs that both import such a
        // module once shared its cache and reused that stale state — surfacing
        // as `Cannot find name 'root'` (inherited field on a fresh subclass of
        // a cached parent), the `__reflect_ctor_wrap` W0020 (Cast source
        // undefined), or a load SIGSEGV. Keying on the program confines that
        // reuse to where it is valid.
        //
        // This includes the standard library. Excluding it — on the reasoning
        // that the library is identical for every program — shares one lowered
        // copy across programs, and that is only sound if a restored class
        // carries everything a freshly lowered one does. It does not: a
        // generic library type is lowered against the instantiations the
        // program supplies, so `Channel<T>` and `Tls<T>` restored for one
        // program are wrong for the next. The symptoms are silent and varied —
        // a field that resolves as missing, a method lowered to a different
        // runtime entry point, unbounded recursion in type resolution — and
        // they depend on cache warmth, so the same test passes cold and fails
        // warm. Sharing can return when the restore path is proven equivalent.
        // The standard library is the exception: its modules lower to the same
        // thing for every program, so keying them on the program means a fresh
        // project lowers the whole library again — 222ms for a program whose
        // only statement is `Sys.println`, against 22ms when it can be reused.
        //
        // Restoring a module is only equivalent to compiling it when the
        // declarations it resolves against are present, and those come from
        // the `.bsym` manifest. Without it — `lazy_stdlib`, which skips the
        // manifest — a restored library module resolves against declarations
        // that were never registered, and the failures are silent and varied
        // (a method lowered to another class's runtime entry, unbounded
        // recursion in type resolution ending as a stack overflow). So the
        // library shares its cache exactly when the manifest is loaded.
        if !(self.stdlib_manifest_loaded && self.is_stdlib_source(source_path)) {
            self.user_program_hash().hash(&mut hasher);
        }
        // Fold in a content hash of the TRANSITIVE import set. The cache key was
        // previously the entry file's own bytes + defines only, so editing a
        // DEPENDENCY (a `.hx` imported by this entry, directly or transitively)
        // left the key unchanged: the stale MIR — e.g. an interface vtable from
        // before a method was added — validated as current and was reused,
        // segfaulting at load. `compute_import_set_hash` walks the import graph
        // and hashes each resolvable file's bytes; it is intentionally
        // permissive (imports it can't resolve simply don't contribute).
        let import_hash = crate::ir::blade::compute_import_set_hash(
            source,
            self.namespace_resolver.get_source_paths(),
        );
        import_hash.hash(&mut hasher);
        hasher.finish()
    }


    /// Try to load a cached MIR module from BLADE cache
    /// Returns Some(IrModule) if cache is valid, None otherwise
    pub(crate) fn try_load_blade_cached(&self, source_path: &str, source: &str) -> Option<IrModule> {
        if !self.config.enable_cache {
            return None;
        }

        let blade_path = self.blade_cache_path(source_path)?;
        if !blade_path.exists() {
            trace!("[BLADE] Cache miss (no file): {}", source_path);
            return None;
        }

        match load_blade(&blade_path) {
            Ok((mir, metadata, _symbols, _cached_maps)) => {
                // Validate cache by checking source hash AND compiler cache
                // ABI id — see save_to_cache / matching check at the other
                // load site for why both are required.
                let current_hash = self.hash_source_for_config(source_path, source);
                let current_build_id = env!("RAYZOR_BUILD_ID");
                if metadata.source_hash != current_hash {
                    trace!("[BLADE] Cache stale (hash mismatch): {}", source_path);
                    None
                } else if metadata.build_id != current_build_id {
                    trace!("[BLADE] Cache stale (build-id mismatch): {}", source_path);
                    None
                } else {
                    debug!(
                        "[BLADE] Cache hit: {} -> {}",
                        source_path,
                        blade_path.display()
                    );
                    Some(mir)
                }
            }
            Err(e) => {
                trace!("[BLADE] Cache read error for {}: {}", source_path, e);
                None
            }
        }
    }


    /// Save a MIR module to BLADE cache with optional type info and cross-reference maps
    pub(crate) fn save_blade_cached(
        &self,
        source_path: &str,
        source: &str,
        mir: &IrModule,
        dependencies: Vec<String>,
        symbols: Option<BladeTypeInfo>,
        cached_maps: Option<BladeCachedMaps>,
    ) {
        if !self.config.enable_cache {
            return;
        }

        let blade_path = match self.blade_cache_path(source_path) {
            Some(p) => p,
            None => return,
        };

        // Ensure cache directory exists
        if let Some(parent) = blade_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    trace!("[BLADE] Failed to create cache dir: {}", e);
                    return;
                }
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let metadata = BladeMetadata {
            name: mir.name.clone(),
            source_path: source_path.to_string(),
            source_hash: self.hash_source_for_config(source_path, source),
            source_timestamp: now, // We use hash for validation, not timestamp
            compile_timestamp: now,
            dependencies,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            build_id: env!("RAYZOR_BUILD_ID").to_string(),
        };

        // Compute per-function granular invalidation hashes (§3.2)
        let cached_maps = cached_maps.map(|mut maps| {
            for entry in &mut maps.functions {
                if let Some(func) = mir
                    .functions
                    .values()
                    .find(|f| f.name.ends_with(&entry.method_name))
                {
                    entry.signature_hash = crate::ir::blade::compute_signature_hash(func);
                    entry.body_hash = crate::ir::blade::compute_body_hash(func);
                }
            }
            maps
        });

        match save_blade_with_state(&blade_path, mir, metadata, symbols, cached_maps) {
            Ok(()) => {
                debug!(
                    "[BLADE] Cached: {} -> {}",
                    source_path,
                    blade_path.display()
                );
            }
            Err(e) => {
                trace!("[BLADE] Failed to cache {}: {}", source_path, e);
            }
        }
    }


    /// Build name-keyed cached maps from MIR lowering result for BLADE cache storage.
    /// Converts SymbolId/TypeId-keyed maps to name-keyed maps that survive across compilations.
    pub(crate) fn build_cached_maps_from_mir_result(
        &self,
        function_map: &BTreeMap<crate::tast::SymbolId, crate::ir::IrFunctionId>,
        field_index_map: &BTreeMap<crate::tast::SymbolId, (crate::tast::TypeId, u32)>,
        constructor_name_map: &BTreeMap<String, crate::ir::IrFunctionId>,
        class_alloc_sizes: &BTreeMap<crate::tast::SymbolId, u64>,
        field_class_names: &BTreeMap<crate::tast::SymbolId, String>,
        property_access_map: &BTreeMap<crate::tast::SymbolId, crate::tast::PropertyAccessInfo>,
        function_param_hir_types: &BTreeMap<crate::ir::IrFunctionId, Vec<crate::tast::TypeId>>,
        interface_vtables: &BTreeMap<
            (crate::tast::SymbolId, crate::tast::SymbolId),
            Vec<crate::tast::SymbolId>,
        >,
        interface_method_names: &BTreeMap<crate::tast::SymbolId, Vec<crate::tast::InternedString>>,
        interface_method_return_types: &BTreeMap<
            (crate::tast::SymbolId, crate::tast::InternedString),
            crate::tast::TypeId,
        >,
        interface_extends: &BTreeMap<crate::tast::SymbolId, Vec<crate::tast::SymbolId>>,
    ) -> BladeCachedMaps {
        let mut functions = Vec::new();
        let mut fields = Vec::new();
        let mut class_sizes = Vec::new();

        // A member's owning class is its scope's class, needed once per function,
        // field and property below. Answering it by searching the symbol table
        // each time is quadratic in a table that does not change here.
        let class_names_by_scope = self.class_names_by_scope();

        // Resolve a HIR TypeId to a qualified-name string for the cache.
        // Only Class/Interface types matter to Path 3 of
        // `maybe_materialize_for_call`; everything else (primitives,
        // abstracts, anonymous, …) we encode as None so the restore
        // side leaves the corresponding param-type slot unwrapped.
        let resolve_hir_type_name = |ty: crate::tast::TypeId| -> Option<String> {
            let type_table = self.type_table.borrow();
            let info = type_table.get(ty)?;
            let symbol_id = match &info.kind {
                crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                crate::tast::TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                _ => None,
            }?;
            let sym = self.symbol_table.get_symbol(symbol_id)?;
            sym.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(sym.name))
                .map(|s| s.to_string())
        };
        let param_names_for = |func_id: crate::ir::IrFunctionId| -> Vec<Option<String>> {
            function_param_hir_types
                .get(&func_id)
                .map(|tys| tys.iter().copied().map(resolve_hir_type_name).collect())
                .unwrap_or_default()
        };

        // Convert function_map: SymbolId → IrFunctionId to (class_name, method_name, func_id)
        for (symbol_id, func_id) in function_map {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                // Constructors are stored separately in constructor_name_map.
                // Skip non-function symbols here (e.g. class symbols used as ctor keys),
                // otherwise cache restore will try to resolve bogus method names like `Exception`.
                if !matches!(sym.kind, crate::tast::SymbolKind::Function) {
                    continue;
                }

                let method_name = self
                    .string_interner
                    .get(sym.name)
                    .unwrap_or("<unknown>")
                    .to_string();

                // Find the class this method belongs to by looking at its scope
                let class_name = class_names_by_scope.get(&sym.scope_id).cloned().flatten();

                functions.push(BladeFuncEntry {
                    class_name: class_name.unwrap_or_default(),
                    method_name,
                    func_id: func_id.0,
                    is_constructor: false,
                    signature_hash: 0, // computed at save time from MIR
                    body_hash: 0,
                    param_type_names: param_names_for(*func_id),
                });
            }
        }

        // Add constructors from constructor_name_map (already name-keyed)
        for (class_name, func_id) in constructor_name_map {
            functions.push(BladeFuncEntry {
                class_name: class_name.clone(),
                method_name: "new".to_string(),
                func_id: func_id.0,
                is_constructor: true,
                signature_hash: 0,
                body_hash: 0,
                param_type_names: param_names_for(*func_id),
            });
        }

        // Convert field_index_map: SymbolId → (TypeId, field_index) to (class_name, field_name, field_index)
        for (symbol_id, (_type_id, field_index)) in field_index_map {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                let field_name = self
                    .string_interner
                    .get(sym.name)
                    .unwrap_or("<unknown>")
                    .to_string();

                // Use field_class_names from MIR context (populated during register_class_metadata)
                // Fall back to accumulated import names for fields inherited from dependencies
                let class_name = field_class_names
                    .get(symbol_id)
                    .cloned()
                    .or_else(|| self.import_field_class_names.get(symbol_id).cloned())
                    .or_else(|| class_names_by_scope.get(&sym.scope_id).cloned().flatten());

                fields.push(BladeFieldEntry {
                    class_name: class_name.unwrap_or_default(),
                    field_name,
                    field_index: *field_index,
                });
            }
        }

        // Convert class_alloc_sizes: SymbolId → u64 to (class_name, size)
        for (symbol_id, size) in class_alloc_sizes {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                let name = sym
                    .qualified_name
                    .and_then(|n| self.string_interner.get(n))
                    .or_else(|| self.string_interner.get(sym.name))
                    .unwrap_or("<unknown>")
                    .to_string();
                class_sizes.push((name, *size));
            }
        }

        // Convert property_access_map: SymbolId → PropertyAccessInfo to (class_name, field_name, getter, setter)
        let mut properties = Vec::new();
        for (symbol_id, prop_info) in property_access_map {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                let field_name = self
                    .string_interner
                    .get(sym.name)
                    .unwrap_or("<unknown>")
                    .to_string();
                let class_name = field_class_names
                    .get(symbol_id)
                    .cloned()
                    .or_else(|| self.import_field_class_names.get(symbol_id).cloned())
                    .or_else(|| class_names_by_scope.get(&sym.scope_id).cloned().flatten());
                let to_blade = |acc: &crate::tast::PropertyAccessor| -> BladeAccessor {
                    match acc {
                        crate::tast::PropertyAccessor::Default => BladeAccessor::Default,
                        crate::tast::PropertyAccessor::Null => BladeAccessor::Null,
                        crate::tast::PropertyAccessor::Never => BladeAccessor::Never,
                        crate::tast::PropertyAccessor::Dynamic => BladeAccessor::Dynamic,
                        crate::tast::PropertyAccessor::Method(n) => BladeAccessor::Method(
                            self.string_interner.get(*n).unwrap_or("").to_string(),
                        ),
                    }
                };
                // Skip orphan entries with no resolvable owning class — they
                // can't be looked up correctly on restore (the load side keys
                // by class_name) and just pollute the merged map. The
                // name-based fallback in lower_field_access used to surface
                // them first and shadow the real property (e.g. an empty
                // `length` from ArrayIterator stole the StringBuf.length
                // dispatch after a cross-test cache load).
                let Some(class_name) = class_name else {
                    continue;
                };
                properties.push(BladePropertyEntry {
                    class_name,
                    field_name,
                    getter: to_blade(&prop_info.getter),
                    setter: to_blade(&prop_info.setter),
                });
            }
        }

        // Convert interface_vtables: (class_sym, iface_sym) → Vec<method_sym>
        // into qualified-name triples so the SymbolIds survive
        // re-numbering on restore.
        let mut iface_vtable_entries: Vec<crate::ir::blade::BladeIfaceVtableEntry> = Vec::new();
        let qname_of = |sid: crate::tast::SymbolId| -> Option<String> {
            let sym = self.symbol_table.get_symbol(sid)?;
            sym.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(sym.name))
                .map(|s| s.to_string())
        };
        for ((class_sym, iface_sym), methods) in interface_vtables {
            let Some(class_name) = qname_of(*class_sym) else {
                continue;
            };
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let method_qnames: Vec<String> = methods.iter().filter_map(|m| qname_of(*m)).collect();
            // Skip entries that lost their methods on resolution — they
            // can't be reconstructed deterministically on restore.
            if method_qnames.len() != methods.len() {
                continue;
            }
            iface_vtable_entries.push(crate::ir::blade::BladeIfaceVtableEntry {
                class_name,
                iface_name,
                method_qnames,
            });
        }

        // Convert interface_method_names: iface_sym → Vec<InternedString>
        // into qname-keyed entries. `qname_of` from the iface_vtables
        // block above is closed over here too.
        //
        // Slot-alignment discipline: `maybe_materialize_for_call`'s
        // vtable-slot math indexes by position into this Vec. A silently
        // empty string from a failed interner lookup would shift every
        // following slot index by one and misroute dispatch. Mirror
        // c7a170d's all-or-nothing skip — drop the whole entry if ANY
        // method name fails to intern, rather than ship a slot-misaligned
        // Vec. Same discipline as `interface_extends` below.
        let mut iface_method_names_entries: Vec<crate::ir::blade::BladeInterfaceMethodNamesEntry> =
            Vec::new();
        for (iface_sym, method_names) in interface_method_names {
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let method_names_str: Vec<String> = method_names
                .iter()
                .filter_map(|n| self.string_interner.get(*n).map(|s| s.to_string()))
                .collect();
            if method_names_str.len() != method_names.len() {
                continue;
            }
            iface_method_names_entries.push(crate::ir::blade::BladeInterfaceMethodNamesEntry {
                iface_name,
                method_names: method_names_str,
            });
        }

        // Convert interface_method_return_types: (iface_sym,
        // method_name) → TypeId into qname-keyed entries. Drop entries
        // whose return type isn't a Class/Interface (those are
        // recoverable from MIR signature on the consumer side).
        let mut iface_method_return_type_entries: Vec<
            crate::ir::blade::BladeInterfaceMethodReturnTypeEntry,
        > = Vec::new();
        for ((iface_sym, method_name), ty) in interface_method_return_types {
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let Some(return_type_name) = resolve_hir_type_name(*ty) else {
                continue;
            };
            let method_name_str = self
                .string_interner
                .get(*method_name)
                .unwrap_or("")
                .to_string();
            if method_name_str.is_empty() {
                continue;
            }
            iface_method_return_type_entries.push(
                crate::ir::blade::BladeInterfaceMethodReturnTypeEntry {
                    iface_name,
                    method_name: method_name_str,
                    return_type_name,
                },
            );
        }

        // Convert interface_extends: iface_sym → Vec<parent_sym> into
        // qname-keyed entries. Skip entries where any parent qname
        // fails to resolve so partial-restore doesn't silently lose
        // the rest of the chain.
        let mut iface_extends_entries: Vec<crate::ir::blade::BladeInterfaceExtendsEntry> =
            Vec::new();
        for (iface_sym, parents) in interface_extends {
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let parent_names: Vec<String> = parents.iter().filter_map(|p| qname_of(*p)).collect();
            if parent_names.len() != parents.len() {
                continue;
            }
            iface_extends_entries.push(crate::ir::blade::BladeInterfaceExtendsEntry {
                iface_name,
                parent_names,
            });
        }

        BladeCachedMaps {
            functions,
            fields,
            class_sizes,
            properties,
            inline_vars: Vec::new(), // populated separately in try_compile_import
            interface_vtables: iface_vtable_entries,
            interface_method_names: iface_method_names_entries,
            interface_method_return_types: iface_method_return_type_entries,
            interface_extends: iface_extends_entries,
        }
    }


    /// Find the qualified class name that owns a given scope.
    /// Used to convert scope-based symbol lookups to name-based keys for cache.
    /// Every class member scope mapped to that class's name.
    ///
    /// A scope shared by several class symbols resolves to the one declared
    /// first, which is the class a search over the symbol table would reach.
    pub(crate) fn class_names_by_scope(&self) -> std::collections::HashMap<ScopeId, Option<String>> {
        let mut by_scope = std::collections::HashMap::new();
        for i in 0..self.symbol_table.len() {
            let sym_id = crate::tast::SymbolId::from_raw(i as u32);
            let Some(sym) = self.symbol_table.get_symbol(sym_id) else {
                continue;
            };
            if !matches!(sym.kind, crate::tast::SymbolKind::Class) {
                continue;
            }
            by_scope.entry(sym.scope_id).or_insert_with(|| {
                sym.qualified_name
                    .and_then(|n| self.string_interner.get(n))
                    .or_else(|| self.string_interner.get(sym.name))
                    .map(|name| name.to_string())
            });
        }
        by_scope
    }


    /// Extract static inline var constants from a TypedFile for BLADE cache storage.
    pub(crate) fn extract_inline_vars_from_typed_file(
        typed_file: &TypedFile,
        symbol_table: &crate::tast::SymbolTable,
        string_interner: &crate::tast::StringInterner,
    ) -> Vec<crate::ir::blade::BladeInlineVarEntry> {
        use crate::ir::blade::{BladeInlineValue, BladeInlineVarEntry};
        use crate::tast::node::LiteralValue;

        let mut entries = Vec::new();

        for class in &typed_file.classes {
            let class_name = symbol_table
                .get_symbol(class.symbol_id)
                .and_then(|sym| {
                    sym.qualified_name
                        .and_then(|n| string_interner.get(n))
                        .or_else(|| string_interner.get(sym.name))
                })
                .unwrap_or("")
                .to_string();
            if class_name.is_empty() {
                continue;
            }

            for field in &class.fields {
                if !field.is_static {
                    continue;
                }
                // Only inline/final fields
                let is_inline = symbol_table
                    .get_symbol(field.symbol_id)
                    .map(|s| s.is_inline())
                    .unwrap_or(false);
                if !is_inline && field.mutability == crate::tast::symbols::Mutability::Mutable {
                    continue;
                }
                let Some(init) = &field.initializer else {
                    continue;
                };

                let field_name = string_interner
                    .get(
                        symbol_table
                            .get_symbol(field.symbol_id)
                            .map(|s| s.name)
                            .unwrap_or(unsafe { crate::tast::InternedString::from_raw(0) }),
                    )
                    .unwrap_or("")
                    .to_string();

                // Try to evaluate the initializer to a constant
                let value = match &init.kind {
                    crate::tast::node::TypedExpressionKind::Literal { value: lit } => match lit {
                        LiteralValue::Int(v) => Some(BladeInlineValue::Int(*v)),
                        LiteralValue::Float(v) => Some(BladeInlineValue::Float(*v)),
                        LiteralValue::Bool(v) => Some(BladeInlineValue::Bool(*v)),
                        LiteralValue::String(v) => Some(BladeInlineValue::String(v.clone())),
                        _ => None,
                    },
                    _ => None,
                };

                if let Some(value) = value {
                    entries.push(BladeInlineVarEntry {
                        class_name: class_name.clone(),
                        field_name,
                        value,
                    });
                }
            }
        }

        // Also handle abstract fields (enum abstract constants)
        for abs in &typed_file.abstracts {
            let abs_name = symbol_table
                .get_symbol(abs.symbol_id)
                .and_then(|sym| {
                    sym.qualified_name
                        .and_then(|n| string_interner.get(n))
                        .or_else(|| string_interner.get(sym.name))
                })
                .unwrap_or("")
                .to_string();
            if abs_name.is_empty() {
                continue;
            }

            for field in &abs.fields {
                if !field.is_static {
                    continue;
                }
                let Some(init) = &field.initializer else {
                    continue;
                };
                let field_name = string_interner
                    .get(
                        symbol_table
                            .get_symbol(field.symbol_id)
                            .map(|s| s.name)
                            .unwrap_or(unsafe { crate::tast::InternedString::from_raw(0) }),
                    )
                    .unwrap_or("")
                    .to_string();

                let value = match &init.kind {
                    crate::tast::node::TypedExpressionKind::Literal { value: lit } => match lit {
                        LiteralValue::Int(v) => Some(BladeInlineValue::Int(*v)),
                        LiteralValue::Float(v) => Some(BladeInlineValue::Float(*v)),
                        LiteralValue::Bool(v) => Some(BladeInlineValue::Bool(*v)),
                        LiteralValue::String(v) => Some(BladeInlineValue::String(v.clone())),
                        _ => None,
                    },
                    _ => None,
                };

                if let Some(value) = value {
                    entries.push(BladeInlineVarEntry {
                        class_name: abs_name.clone(),
                        field_name,
                        value,
                    });
                }
            }
        }

        entries
    }


    /// Store inline vars from BladeInlineVarEntry into the global map.
    pub(crate) fn store_inline_vars(&mut self, entries: &[crate::ir::blade::BladeInlineVarEntry]) {
        for entry in entries {
            let key = format!("{}.{}", entry.class_name, entry.field_name);
            self.global_inline_vars.insert(key, entry.value.clone());
        }
    }


    /// Load a BLADE cached file and return all components including type info and cached maps
    pub(crate) fn try_load_blade_cached_full(
        &self,
        source_path: &str,
        source: &str,
    ) -> Option<(
        IrModule,
        BladeMetadata,
        Option<BladeTypeInfo>,
        Option<BladeCachedMaps>,
    )> {
        if !self.config.enable_cache {
            return None;
        }

        // Prefer this project's own artifact. Failing that, take the standard
        // library the binary carries, so a project that has never been built
        // does not lower the library again.
        let loaded = match self.blade_cache_path(source_path) {
            Some(path) if path.exists() => load_blade(&path),
            _ => match self.embedded_snapshot_entry(source_path) {
                Some(bytes) => crate::ir::blade::load_blade_from_bytes(bytes),
                None => match self.prepared_blade_path(source_path) {
                    Some(path) if path.exists() => load_blade(&path),
                    _ => return None,
                },
            },
        };

        match loaded {
            Ok((mir, metadata, symbols, cached_maps)) => {
                let current_hash = self.hash_source_for_config(source_path, source);
                let current_build_id = env!("RAYZOR_BUILD_ID");
                // Why a cached module was rejected. The `trace!` below says
                // which check failed but not what the values were, and the
                // embedded snapshot fails BOTH when it is stale: the build id
                // changes with every compiler source change, and the source
                // hash carries the program.
                if std::env::var_os("RAYZOR_DEBUG_BLADE").is_some() {
                    eprintln!(
                        "[blade] {source_path}\n        hash  {} vs {current_hash}\n        build {} vs {current_build_id}",
                        metadata.source_hash, metadata.build_id
                    );
                }
                if metadata.source_hash != current_hash {
                    debug!("[BLADE] Cache stale (hash mismatch): {}", source_path);
                    None
                } else if metadata.build_id != current_build_id {
                    debug!("[BLADE] Cache stale (build-id mismatch): {}", source_path);
                    None
                } else {
                    Some((mir, metadata, symbols, cached_maps))
                }
            }
            Err(e) => {
                debug!("[BLADE] Cache read error for {}: {}", source_path, e);
                None
            }
        }
    }


    /// Restore MIR-level cross-reference maps from cached data using fresh symbol IDs.
    /// Rebuild the MIR-level maps a cached module refers to, and report how
    /// many references could not be resolved in this context.
    ///
    /// Each unresolved one used to be dropped where it was found. A dropped
    /// reference is not a missing optimisation — it is a call that now
    /// dispatches somewhere else, a field that reads as absent, a type that
    /// resolves to nothing — and it was invisible, so the module loaded and
    /// miscompiled. The count lets the caller decline the entry instead.
    pub(crate) fn restore_cached_maps(
        &mut self,
        cached_maps: &BladeCachedMaps,
        registered: &BTreeMap<String, (crate::tast::SymbolId, crate::tast::TypeId, ScopeId)>,
    ) -> usize {
        use crate::ir::IrFunctionId;
        let mut dropped = 0usize;

        // Restore function mappings: find method SymbolId in registered class scopes
        for entry in &cached_maps.functions {
            // Stash per-param interface-name info under the *cached* func_id;
            // renumber_and_push_import_mir later remaps the keys. Only store
            // entries that carry at least one name — empty Vecs are no-ops.
            if entry.param_type_names.iter().any(|name| name.is_some()) {
                self.import_function_param_iface_names
                    .insert(IrFunctionId(entry.func_id), entry.param_type_names.clone());
            }

            if entry.is_constructor {
                // Constructors are keyed by class name
                self.import_constructor_name_map
                    .insert(entry.class_name.clone(), IrFunctionId(entry.func_id));
                dropped += 1;
                continue;
            }

            // Look up the class, then find the method symbol in its scope
            if let Some((_class_sym, _class_type, class_scope)) = registered.get(&entry.class_name)
            {
                let method_name_interned = self.string_interner.intern(&entry.method_name);
                if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                    if let Some(method_sym) = scope.get_symbol(method_name_interned) {
                        self.stdlib_function_map
                            .insert(method_sym, IrFunctionId(entry.func_id));
                    }
                }
            }
        }

        // Restore field index mappings
        for entry in &cached_maps.fields {
            if let Some((_class_sym, class_type, class_scope)) = registered.get(&entry.class_name) {
                let field_name_interned = self.string_interner.intern(&entry.field_name);
                if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                    if let Some(field_sym) = scope.get_symbol(field_name_interned) {
                        debug!(
                            "[BLADE_FIELD] Restored {}.{} {:?} -> (TypeId({:?}), index={})",
                            entry.class_name,
                            entry.field_name,
                            field_sym,
                            class_type,
                            entry.field_index
                        );
                        self.import_field_index_map
                            .insert(field_sym, (*class_type, entry.field_index));
                    } else {
                        debug!(
                            "[BLADE_FIELD] MISS: {}.{} not found in scope {:?}",
                            entry.class_name, entry.field_name, class_scope
                        );
                    }
                }
            } else {
                debug!(
                    "[BLADE_FIELD] MISS: class '{}' not in registered map",
                    entry.class_name
                );
            }
        }

        // Restore class allocation sizes
        for (class_name, size) in &cached_maps.class_sizes {
            // Name-based (stable across contexts)
            self.import_class_alloc_sizes_by_name
                .insert(class_name.clone(), *size);
            // SymbolId-based (stable across contexts)
            if let Some((class_sym, _class_type, _)) = registered.get(class_name) {
                self.import_class_alloc_sizes.insert(*class_sym, *size);
            }
        }

        // Derive class sizes from field entries for old caches that lack class_sizes.
        // For each class with fields but no size entry, compute (max_field_index + 1) * 8.
        {
            let mut class_max_idx: BTreeMap<&str, u32> = BTreeMap::new();
            for entry in &cached_maps.fields {
                if !entry.class_name.is_empty() {
                    let cur = class_max_idx.entry(&entry.class_name).or_insert(0);
                    if entry.field_index > *cur {
                        *cur = entry.field_index;
                    }
                }
            }
            for (class_name, max_idx) in &class_max_idx {
                if !self
                    .import_class_alloc_sizes_by_name
                    .contains_key(*class_name)
                {
                    let size = ((*max_idx as u64) + 1) * 8;
                    self.import_class_alloc_sizes_by_name
                        .insert(class_name.to_string(), size);
                    if let Some((class_sym, _class_type, _)) = registered.get(*class_name) {
                        self.import_class_alloc_sizes.insert(*class_sym, size);
                    }
                }
            }
        }

        // Restore interface_vtables entries — these survive across
        // compilations because they're keyed by qualified name. For
        // each (class_qname, iface_qname, method_qnames) entry, look
        // up the SymbolIds in the current context and insert into
        // import_interface_vtables so iface-to-iface casts can
        // emit `haxe_iface_vtable_set_slot` registrations from the
        // user-module's __vtable_init__.
        for entry in &cached_maps.interface_vtables {
            let Some((class_sym, _, _)) = registered.get(&entry.class_name) else {
                dropped += 1;
                continue;
            };
            let Some((iface_sym, _, iface_scope)) = registered.get(&entry.iface_name) else {
                dropped += 1;
                continue;
            };
            let Some(iface_scope) = self.scope_tree.get_scope(*iface_scope) else {
                dropped += 1;
                continue;
            };
            // Resolve each method qname's local name (the trailing
            // segment after the last `.`) back to a SymbolId in the
            // interface's own scope.
            let mut method_syms: Vec<crate::tast::SymbolId> =
                Vec::with_capacity(entry.method_qnames.len());
            let mut all_found = true;
            for qname in &entry.method_qnames {
                let local = qname.rsplit('.').next().unwrap_or(qname.as_str());
                let interned = self.string_interner.intern(local);
                if let Some(method_sym) = iface_scope.get_symbol(interned) {
                    method_syms.push(method_sym);
                } else {
                    all_found = false;
                    break;
                }
            }
            if !all_found {
                dropped += 1;
                continue;
            }
            self.import_interface_vtables
                .insert((*class_sym, *iface_sym), method_syms);
        }

        // Restore interface_method_names: qname → ordered method
        // names. Required so downstream files that pick up a BLADE
        // cached import can still resolve `t.encode(...)` on an
        // interface-typed local — `maybe_materialize_for_call`'s
        // interface-dispatch path keys the vtable slot by method
        // index from this map.
        for entry in &cached_maps.interface_method_names {
            let Some((iface_sym, _, _)) = registered.get(&entry.iface_name) else {
                dropped += 1;
                continue;
            };
            let method_syms: Vec<crate::tast::InternedString> = entry
                .method_names
                .iter()
                .map(|n| self.string_interner.intern(n))
                .collect();
            self.import_interface_method_names
                .insert(*iface_sym, method_syms);
        }

        // Restore interface_method_return_types: (iface, method_name)
        // → return TypeId. Required for cross-context iface method
        // return-type re-resolution (W0014/W0015) on cached imports.
        for entry in &cached_maps.interface_method_return_types {
            let Some((iface_sym, _, _)) = registered.get(&entry.iface_name) else {
                dropped += 1;
                continue;
            };
            let Some((_, return_type_id, _)) = registered.get(&entry.return_type_name) else {
                dropped += 1;
                continue;
            };
            let method_name_interned = self.string_interner.intern(&entry.method_name);
            self.import_interface_method_return_types
                .insert((*iface_sym, method_name_interned), *return_type_id);
        }

        // Restore interface_extends: iface → parent ifaces. Required
        // so iface-to-iface upcasts / dispatch through a parent
        // interface resolves when both ifaces come from cached
        // imports. All-or-nothing — drop the entry if any parent
        // qname is unregistered in the consuming context.
        for entry in &cached_maps.interface_extends {
            let Some((iface_sym, _, _)) = registered.get(&entry.iface_name) else {
                dropped += 1;
                continue;
            };
            let mut parent_syms: Vec<crate::tast::SymbolId> =
                Vec::with_capacity(entry.parent_names.len());
            let mut all_found = true;
            for parent_name in &entry.parent_names {
                if let Some((parent_sym, _, _)) = registered.get(parent_name) {
                    parent_syms.push(*parent_sym);
                } else {
                    all_found = false;
                    break;
                }
            }
            if !all_found {
                dropped += 1;
                continue;
            }
            self.import_interface_extends
                .insert(*iface_sym, parent_syms);
        }

        // Restore property access mappings
        for entry in &cached_maps.properties {
            if let Some((_class_sym, _class_type, class_scope)) = registered.get(&entry.class_name)
            {
                let field_name_interned = self.string_interner.intern(&entry.field_name);
                if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                    if let Some(field_sym) = scope.get_symbol(field_name_interned) {
                        let from_blade = |acc: &BladeAccessor| -> crate::tast::PropertyAccessor {
                            match acc {
                                BladeAccessor::Default => crate::tast::PropertyAccessor::Default,
                                BladeAccessor::Null => crate::tast::PropertyAccessor::Null,
                                BladeAccessor::Never => crate::tast::PropertyAccessor::Never,
                                BladeAccessor::Dynamic => crate::tast::PropertyAccessor::Dynamic,
                                BladeAccessor::Method(n) => crate::tast::PropertyAccessor::Method(
                                    self.string_interner.intern(n),
                                ),
                            }
                        };
                        self.import_property_access_map.insert(
                            field_sym,
                            // As above: a restored entry answers dispatch, and
                            // the layout it belongs to is not rebuilt here.
                            crate::tast::PropertyAccessInfo {
                                getter: from_blade(&entry.getter),
                                setter: from_blade(&entry.setter),
                                is_var: false,
                            },
                        );
                    }
                }
            }
        }

        // Restore class_type_to_symbol and class_method_symbols mappings
        for (class_name, (class_sym, class_type, class_scope)) in registered {
            self.import_class_type_to_symbol
                .insert(*class_type, *class_sym);
            // Restore class_method_symbols by iterating symbols in the class scope
            if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                for &method_sym in &scope.symbols {
                    if let Some(sym) = self.symbol_table.get_symbol(method_sym) {
                        self.import_class_method_symbols
                            .insert((*class_sym, sym.name), method_sym);
                    }
                }
            }
        }
        dropped
    }


    /// Try to load a cached MIR module from a BLADE file
    ///
    /// Returns Some(IrModule) if cache is valid, None if cache doesn't exist or is stale
    pub fn try_load_cached(&self, source_path: &Path) -> Option<IrModule> {
        if !self.config.enable_cache {
            return None;
        }

        let cache_path = self.config.get_cache_path(source_path);
        if !cache_path.exists() {
            return None;
        }

        // Load BLADE file
        let (mir_module, metadata, _symbols, _cached_maps) = match load_blade(&cache_path) {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to load cache for {:?}: {}", source_path, e);
                return None;
            }
        };

        // Check if source file has been modified since cache was created
        if let Ok(source_meta) = std::fs::metadata(source_path) {
            if let Ok(modified) = source_meta.modified() {
                let source_timestamp = modified
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Cache is stale if source was modified after cache was created
                if source_timestamp > metadata.compile_timestamp {
                    if self.config.enable_cache {
                        debug!(
                            "Cache stale for {:?} (source: {}, cache: {})",
                            source_path, source_timestamp, metadata.compile_timestamp
                        );
                    }
                    return None;
                }
            }
        }

        if let Ok(source) = std::fs::read_to_string(source_path) {
            let current_hash = self.hash_source_for_config(&source_path.to_string_lossy(), &source);
            if metadata.source_hash != current_hash {
                if self.config.enable_cache {
                    debug!("Cache source hash mismatch for {:?}", source_path);
                }
                return None;
            }
        }

        // Check compiler version matches
        let current_version = env!("CARGO_PKG_VERSION");
        if metadata.compiler_version != current_version {
            if self.config.enable_cache {
                debug!(
                    "Cache version mismatch for {:?} (cache: {}, current: {})",
                    source_path, metadata.compiler_version, current_version
                );
            }
            return None;
        }

        // Check compiler cache ABI id matches. Parser/lowerer/MIR-shape
        // changes within the same semver bump the id (see compiler/build.rs)
        // and can silently shift function IDs or AST structure for the same
        // source — without this guard, MIR cached by an older compiler loads
        // into a newer compiler and surfaces as SIGILL at unrelated call sites.
        let current_build_id = env!("RAYZOR_BUILD_ID");
        if metadata.build_id != current_build_id {
            if self.config.enable_cache {
                debug!(
                    "Cache build-id mismatch for {:?} (cache: {}, current: {})",
                    source_path, metadata.build_id, current_build_id
                );
            }
            return None;
        }

        if self.config.enable_cache {
            debug!("Cache hit for {:?}", source_path);
        }

        Some(mir_module)
    }


    /// Save a compiled MIR module to the BLADE cache
    pub fn save_to_cache(&self, source_path: &Path, module: &IrModule) -> Result<(), String> {
        if !self.config.enable_cache {
            return Ok(());
        }

        let cache_path = self.config.get_cache_path(source_path);

        // Ensure cache directory exists
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        }

        // Get source file timestamp and compute hash
        let source_timestamp = std::fs::metadata(source_path)
            .and_then(|m| m.modified())
            .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(0);

        // Read source for hash computation
        let source_hash = std::fs::read_to_string(source_path)
            .map(|s| self.hash_source_for_config(&source_path.to_string_lossy(), &s))
            .unwrap_or(0);

        let compile_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create metadata
        let metadata = BladeMetadata {
            name: module.name.clone(),
            source_path: source_path.to_string_lossy().to_string(),
            source_hash,
            source_timestamp,
            compile_timestamp,
            dependencies: Vec::new(), // TODO: Track dependencies for proper invalidation
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            build_id: env!("RAYZOR_BUILD_ID").to_string(),
        };

        // Save to BLADE file (no type info/maps for standalone compile command)
        save_blade_with_state(&cache_path, module, metadata, None, None)
            .map_err(|e| format!("Failed to save cache: {}", e))?;

        if self.config.enable_cache {
            debug!("Cached MIR for {:?} -> {:?}", source_path, cache_path);
        }

        Ok(())
    }


    /// Clear all cached BLADE files
    pub fn clear_cache(&self) -> Result<(), String> {
        let cache_dir = self.config.get_cache_dir();
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)
                .map_err(|e| format!("Failed to clear cache: {}", e))?;
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| format!("Failed to recreate cache directory: {}", e))?;
            debug!("Cache cleared: {:?}", cache_dir);
        }
        Ok(())
    }


    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        let cache_dir = self.config.get_cache_dir();
        let mut stats = CacheStats::default();

        if !cache_dir.exists() {
            return stats;
        }

        // Count .blade files and calculate total size
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("blade") {
                        stats.cached_modules += 1;
                        stats.total_size_bytes += metadata.len();
                    }
                }
            }
        }

        stats
    }
}
