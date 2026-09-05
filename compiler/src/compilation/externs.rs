//! The surface a host program links against: externs, plugins, HDLLs.

use super::*;

impl CompilationUnit {

    /// Get extern function → JS module name mappings (from @:jsImport classes).
    pub fn get_extern_js_module_map(&self) -> &BTreeMap<String, String> {
        &self.extern_js_module_map
    }


    pub fn get_qualified_method_map(&self) -> &BTreeMap<String, String> {
        &self.qualified_method_map
    }


    /// Get class allocation sizes keyed by class name.
    /// Used by WASM bindgen to generate JS constructors that call malloc(size).
    pub fn get_class_alloc_sizes_by_name(&self) -> &BTreeMap<String, u64> {
        &self.import_class_alloc_sizes_by_name
    }


    /// Get HDLL function pointers for JIT linking.
    ///
    /// Returns symbol name and pointer pairs collected from all loaded HDLL plugins.
    /// These should be merged with runtime symbols when creating the backend.
    pub fn get_hdll_symbols(&self) -> &[(String, *const u8)] {
        &self.hdll_symbols
    }


    /// Register extern class methods from a TypedFile as plugin mappings.
    ///
    /// When an imported file contains an extern class (e.g., GPUDevice with @:native methods),
    /// this extracts the method signatures and registers them as NativePlugin entries.
    /// This makes them visible to the MIR lowerer's StdlibMapping, which otherwise only
    /// knows about methods from rpkg NativePlugins.
    /// Seed `import_property_access_map` from any property fields declared in
    /// the given typed file. Without this, extern-only stdlib files (whose MIR
    /// generation is skipped via the `is_extern_only` shortcut in
    /// `compile_file`) never populate the property accessor map, so user code
    /// like `tls.value` falls through to a "field not found" error in MIR
    /// `lower_field_access`. Each property field's `PropertyAccessInfo` is
    /// keyed by the field's SymbolId.
    pub(crate) fn seed_property_accessors_from_typed_file(&mut self, typed_file: &TypedFile) {
        for class in &typed_file.classes {
            for field in &class.fields {
                if let Some(prop_info) = field.property_access.as_ref() {
                    self.import_property_access_map
                        .entry(field.symbol_id)
                        .or_insert_with(|| prop_info.clone());
                }
            }
        }
    }


    pub(crate) fn register_extern_methods_from_typed_file(&mut self, typed_file: &TypedFile) {
        use crate::compiler_plugin::NativePlugin;
        use crate::rpkg::MethodDescEntry;

        // Snapshot everything already mapped — the builtin stdlib AND every
        // loaded native plugin — so methods those sources already bind are
        // skipped. Without the builtin part, naive @:native("spawn")
        // auto-registration overrides the Thread_spawn MIR wrapper with a
        // bare "spawn" symbol that fails at JIT time. Without the plugin
        // part, a bodyless extern method with no @:native (KvCacheQ8's
        // dequantView) gets a guessed bare-name binding that shadows the
        // dylib descriptor's real symbol the moment the class is looked up
        // by its qualified name.
        let builtin_mapping = self.compiler_plugin_registry.build_combined_mapping();

        let mut entries: Vec<MethodDescEntry> = Vec::new();

        for class in &typed_file.classes {
            // Check if this is an extern class by looking up the symbol's flags
            let is_extern = self
                .symbol_table
                .get_symbol(class.symbol_id)
                .map(|s| {
                    s.flags.contains(crate::tast::symbols::SymbolFlags::EXTERN)
                        || s.flags.is_native()
                })
                .unwrap_or(false);

            if !is_extern {
                continue;
            }

            // Get the class's native name (from @:native metadata)
            let class_native_name = self
                .symbol_table
                .get_symbol(class.symbol_id)
                .and_then(|s| {
                    s.native_name
                        .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                })
                .or_else(|| {
                    self.symbol_table.get_symbol(class.symbol_id).and_then(|s| {
                        s.qualified_name
                            .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                    })
                })
                .unwrap_or_default();

            if class_native_name.is_empty() {
                continue;
            }
            // The dot-separated name is what the mapping is keyed by; an
            // already-registered class resolves to its own key, so the skip
            // check below asks about exactly the class it means.
            let class_dot_name = class_native_name.replace("::", ".");
            let class_dot_name = builtin_mapping
                .get_class_static_str(&class_dot_name)
                .map(str::to_string)
                .unwrap_or(class_dot_name);

            // Extract method entries
            for method in &class.methods {
                // A concrete (body-bearing) method on an extern class is compiled
                // to MIR and dispatched as a direct call. Registering it as a
                // bare-name extern mapping would make the call resolve to a
                // runtime symbol that does not exist ("can't resolve symbol X").
                // Only the bodyless `@:native` methods belong in this table.
                if !method.body.is_empty() {
                    continue;
                }
                let method_name = self
                    .string_interner
                    .get(method.name)
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let symbol_name = self
                    .symbol_table
                    .get_symbol(method.symbol_id)
                    .and_then(|s| {
                        s.native_name
                            .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                    })
                    .unwrap_or_else(|| method_name.clone());

                if symbol_name.is_empty() {
                    continue;
                }

                // If anything already mapped — the builtin stdlib or a loaded
                // native plugin — carries this (class, method), keep that
                // binding and register nothing. Without this, the bare
                // @:native symbol ("spawn") or the bare method name would
                // shadow the correct one (Thread_spawn, a dylib descriptor's
                // export) and fail at JIT link time. The dotted class name is
                // enough: the mapping resolves every spelling of it.
                let builtin_match = builtin_mapping
                    .class_key(&class_dot_name)
                    .and_then(|key| builtin_mapping.find_by_name(key, &method_name));
                if let Some((_sig, call)) = builtin_match {
                    // DECLARATION vs RUNTIME arity. The lookup above is three
                    // name-only attempts with no arity awareness, so a method
                    // whose .hx signature promises more parameters than the
                    // runtime implements still binds here — and every later call
                    // inherits that binding. A call passing the extra argument is
                    // then emitted against a native symbol of a different arity:
                    // LLVM rejects the module ("Incorrect number of arguments"),
                    // the tier falls back, and the enclosing function silently
                    // emits NOTHING. No compile error, no runtime message.
                    //
                    // Known live cases: `Sys.command(cmd, ?args)` maps to
                    // `haxe_sys_command` (params: 1 — `?args` is unimplemented),
                    // and `File.read(path, binary = true)` maps to
                    // `file_read_default` (params: 1). Both are real gaps between
                    // the .hx surface and the runtime, not compiler confusion.
                    //
                    // Loud rather than fatal by default; RAYZOR_STRICT_STDLIB_ARITY=1
                    // rejects the binding instead (the intended end state).
                    let declared = method.parameters.len();
                    // OFF by default: the six known stdlib mismatches would print
                    // on every compile, which is noise until they are fixed (and
                    // buries any NEW mismatch in it). RAYZOR_STDLIB_ARITY_WARN=1
                    // lists them; RAYZOR_STRICT_STDLIB_ARITY=1 still rejects the
                    // binding regardless of the warning setting.
                    let warn_arity = std::env::var_os("RAYZOR_STDLIB_ARITY_WARN").is_some();
                    if declared != call.param_count && warn_arity {
                        eprintln!(
                            "[stdlib-arity] {}.{} declares {} parameter(s) but runtime mapping '{}' takes {} — a call supplying the extra argument(s) will be emitted against a mismatched native signature and silently produce nothing. Either implement the parameter in the runtime or narrow the .hx declaration.",
                            class_dot_name, method_name, declared, call.runtime_name, call.param_count
                        );
                    }
                    if declared != call.param_count
                        && std::env::var_os("RAYZOR_STRICT_STDLIB_ARITY").is_some()
                    {
                        continue;
                    }
                    // Record the mapping under the Haxe-qualified name so the
                    // WASM backend stub redirect still finds a canonical symbol.
                    let class_haxe_name = self
                        .symbol_table
                        .get_symbol(class.symbol_id)
                        .and_then(|s| {
                            s.qualified_name
                                .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                        })
                        .unwrap_or_default();
                    if !class_haxe_name.is_empty() {
                        let qualified = format!("{}.{}", class_haxe_name, method_name);
                        self.qualified_method_map
                            .insert(qualified, call.runtime_name.to_string());
                    }
                    if call.is_mir_wrapper {
                        continue;
                    }
                    // Non-MIR-wrapper (extern C): skip the NativePlugin entry too so
                    // that MIR lowering uses the builtin's correctly-typed mapping.
                    continue;
                }

                // Build qualified Haxe name for WASM stub resolution.
                // The MIR lowerer creates wrapper functions named e.g. "rayzor.gpu.Surface.getFormat"
                // from the Haxe package + class + method name.
                let class_haxe_name = self
                    .symbol_table
                    .get_symbol(class.symbol_id)
                    .and_then(|s| {
                        s.qualified_name
                            .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                    })
                    .unwrap_or_default();
                if !class_haxe_name.is_empty() && !symbol_name.is_empty() {
                    let qualified = format!("{}.{}", class_haxe_name, method_name);
                    self.qualified_method_map
                        .insert(qualified, symbol_name.clone());
                }

                // Real declared native-type tags for the return AND each param,
                // so a scalar (`:Int`/`:Bool`/`:Float`) doesn't decay to a boxed
                // PtrVoid across the module boundary. A C-ABI extern takes its
                // args by value; the plugin_match lowering marshals per this
                // signature, so a scalar param declared here as PtrVoid would be
                // boxed (the arg arriving as a DynamicValue pointer the kernel
                // then misreads). The leading self slot on instance methods is a
                // real pointer (tag 3).
                let return_tag = self.haxe_type_to_native_tag(method.return_type);
                let mut param_tags: Vec<u8> = Vec::with_capacity(method.parameters.len() + 1);
                if !method.is_static {
                    param_tags.push(3); // self pointer
                }
                for p in &method.parameters {
                    param_tags.push(self.haxe_type_to_native_tag(p.param_type));
                }
                entries.push(MethodDescEntry {
                    symbol_name,
                    class_name: class_native_name.clone(),
                    method_name,
                    is_static: method.is_static,
                    param_count: param_tags.len() as u8,
                    return_type: return_tag,
                    param_types: param_tags,
                });
            }

            // Store JS module mapping for @:jsImport class methods
            if let Some(class_sym) = self.symbol_table.get_symbol(class.symbol_id) {
                if let Some((mod_is, _)) = class_sym.js_import {
                    if let Some(js_module) = self.string_interner.get(mod_is) {
                        for entry in &entries {
                            self.extern_js_module_map
                                .insert(entry.symbol_name.clone(), js_module.to_string());
                        }
                    }
                }
            }
        }

        // Register under the dotted class name; the mapping's alias index
        // resolves every other spelling of it, so one key per class is enough.
        for entry in &mut entries {
            entry.class_name = entry.class_name.replace("::", ".");
        }

        if !entries.is_empty() {
            let plugin = NativePlugin::from_method_entries("extern_import", entries);
            self.compiler_plugin_registry.register(Box::new(plugin));
        }
    }


    /// Map a Haxe TypeId to an IrTypeDescriptor u8 value for MethodDescEntry.
    pub(crate) fn haxe_type_to_descriptor(&self, type_id: TypeId) -> u8 {
        use crate::tast::TypeKind;
        let tt = self.type_table.borrow();
        match tt.get(type_id).map(|t| &t.kind) {
            Some(TypeKind::Int) => 3,    // I32
            Some(TypeKind::Float) => 7,  // F64
            Some(TypeKind::Bool) => 1,   // Bool
            Some(TypeKind::String) => 8, // String
            Some(TypeKind::Void) => 0,   // Void
            _ => 9,                      // PtrVoid for class types, etc.
        }
    }


    /// Map a Haxe return/param TypeId to a `native_type` tag as decoded by
    /// `compiler_plugin::native_type_to_descriptor` (0=Void 1=I64 2=F64 3=PtrVoid
    /// 4=Bool). This is a DIFFERENT numbering than `haxe_type_to_descriptor`.
    /// Used to give `@:native` extern methods their real declared return type
    /// instead of the old blanket PtrVoid default — a PtrVoid return decays a
    /// scalar (e.g. `:Int`) to a boxed pointer (null) across the module boundary.
    pub(crate) fn haxe_type_to_native_tag(&self, type_id: TypeId) -> u8 {
        use crate::tast::TypeKind;
        let tt = self.type_table.borrow();
        match tt.get(type_id).map(|t| &t.kind) {
            Some(TypeKind::Void) => 0,
            Some(TypeKind::Int) => 1,   // I64
            Some(TypeKind::Float) => 2, // F64
            Some(TypeKind::Bool) => 4,
            _ => 3, // PtrVoid — objects, String, Tensor, Usize, etc.
        }
    }


    /// Register an external compiler plugin.
    ///
    /// This allows native packages (loaded via dlopen) to provide method mappings
    /// and extern declarations without modifying compiler source code. Must be
    /// called before `lower_to_tast()`.
    pub fn register_compiler_plugin(
        &mut self,
        plugin: Box<dyn crate::compiler_plugin::CompilerPlugin + 'static>,
    ) {
        self.compiler_plugin_registry.register(plugin);
    }


    /// Add external runtime symbols for JIT linking.
    ///
    /// These are merged with HDLL symbols and made available to the JIT backend.
    pub fn add_external_symbols(&mut self, symbols: Vec<(String, *const u8)>) {
        self.hdll_symbols.extend(symbols);
    }


    /// Scan parsed user files for `@:hlNative` metadata and load corresponding HDLL libraries.
    ///
    /// This should be called after user files have been added (so `user_files` is populated)
    /// but before MIR lowering (so the plugin registry has all HDLL mappings available).
    ///
    /// For each class with `@:hlNative("libname")`, this:
    /// 1. Extracts method names and static flags from the class declaration
    /// 2. Searches `hdll_search_paths` for `libname.hdll`
    /// 3. Loads the HDLL via `hlp_` symbol introspection
    /// 4. Registers the plugin and collects function pointers for JIT linking
    pub fn discover_and_load_hdlls(&mut self) {
        // Collect hlNative class info from user files before mutating self
        let mut hl_native_classes: Vec<(String, String, Vec<(String, bool)>)> = Vec::new();

        for file in &self.user_files {
            for decl in &file.declarations {
                if let parser::TypeDeclaration::Class(class_decl) = decl {
                    if let Some(lib_name) = Self::extract_hl_native_meta(&class_decl.meta) {
                        let methods: Vec<(String, bool)> = class_decl
                            .fields
                            .iter()
                            .filter_map(|field| {
                                if let parser::ClassFieldKind::Function(func) = &field.kind {
                                    let is_static =
                                        field.modifiers.contains(&parser::Modifier::Static);
                                    Some((func.name.clone(), is_static))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !methods.is_empty() {
                            info!(
                                "Found @:hlNative(\"{}\") on class '{}' with {} methods",
                                lib_name,
                                class_decl.name,
                                methods.len()
                            );
                            hl_native_classes.push((lib_name, class_decl.name.clone(), methods));
                        }
                    }
                }
            }
        }

        // Now load each HDLL
        for (lib_name, class_name, methods) in hl_native_classes {
            if self.loaded_hdlls.contains(&lib_name) {
                debug!("HDLL '{}' already loaded, skipping", lib_name);
                continue;
            }

            let method_refs: Vec<(&str, bool)> = methods
                .iter()
                .map(|(name, is_static)| (name.as_str(), *is_static))
                .collect();

            if let Some(hdll_path) = self.find_hdll(&lib_name) {
                match HdllPlugin::load_with_introspection(
                    &hdll_path,
                    &lib_name,
                    &class_name,
                    &method_refs,
                ) {
                    Ok(plugin) => {
                        for (name, ptr) in plugin.get_symbols() {
                            self.hdll_symbols.push((name.to_string(), ptr));
                        }
                        self.compiler_plugin_registry.register(Box::new(plugin));
                        self.loaded_hdlls.insert(lib_name);
                    }
                    Err(e) => {
                        warn!("Failed to load {}.hdll: {}", lib_name, e);
                    }
                }
            } else {
                warn!(
                    "HDLL '{}' not found in search paths: {:?}",
                    lib_name, self.config.hdll_search_paths
                );
            }
        }
    }


    /// Extract `@:hlNative("libname")` metadata from a class's metadata list.
    ///
    /// Returns `Some(lib_name)` if `@:hlNative` is found, `None` otherwise.
    pub(crate) fn extract_hl_native_meta(meta: &[parser::Metadata]) -> Option<String> {
        for m in meta {
            let name = m.name.strip_prefix(':').unwrap_or(&m.name);
            if name == "hlNative" {
                // Extract library name from first parameter
                if let Some(first_param) = m.params.first() {
                    if let parser::ExprKind::String(lib_name) = &first_param.kind {
                        return Some(lib_name.clone());
                    }
                }
                // @:hlNative with no parameters - use class name as fallback
                return None;
            }
        }
        None
    }


    /// Search for an HDLL file in the configured search paths.
    ///
    /// On macOS, HDLLs are `.dylib` files. On Linux, `.so`. On Windows, `.dll`.
    /// The Hashlink convention uses `.hdll` extension.
    pub(crate) fn find_hdll(&self, lib_name: &str) -> Option<PathBuf> {
        // Try platform-specific names and .hdll extension
        let candidates = if cfg!(target_os = "macos") {
            vec![
                format!("{}.hdll", lib_name),
                format!("lib{}.dylib", lib_name),
                format!("{}.dylib", lib_name),
            ]
        } else if cfg!(target_os = "windows") {
            vec![format!("{}.hdll", lib_name), format!("{}.dll", lib_name)]
        } else {
            vec![
                format!("{}.hdll", lib_name),
                format!("lib{}.so", lib_name),
                format!("{}.so", lib_name),
            ]
        };

        for dir in &self.config.hdll_search_paths {
            for candidate in &candidates {
                let path = dir.join(candidate);
                if path.exists() {
                    return Some(path);
                }
            }
        }

        None
    }
}
