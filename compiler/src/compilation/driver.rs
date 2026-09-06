//! The compile pipeline itself, from an AST to a lowered module.

use super::*;

impl CompilationUnit {

    /// Compile a single file using shared state (string interner, symbol table, namespace resolver, etc.)
    /// This ensures symbols from different files can see each other
    ///
    /// If `skip_pre_registration` is true, assumes types have already been pre-registered
    /// and skips the first pass in lower_file.

    pub(crate) fn compile_file_with_shared_state_ex(
        &mut self,
        filename: &str,
        source: &str,
        skip_pre_registration: bool,
        skip_stdlib_merge: bool,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        use parser::parse_haxe_file_with_diagnostics;

        // Skip if already successfully compiled - return cached TypedFile
        if let Some(cached) = self.compiled_files.get(filename) {
            return Ok(cached.clone());
        }

        // Parse the file
        let t_parse = profile_timer(self.config.profile_typecheck);
        let haxe_file = self.parse_file(filename, source).map_err(|e| {
            vec![CompilationError {
                message: format!("Parse error: {}", e),
                location: SourceLocation::unknown(),
                category: ErrorCategory::ParseError,
                suggestion: None,
                related_errors: Vec::new(),
            }]
        })?;
        add_profile_ms(&mut self.typecheck_timings.file_parse_ms, t_parse);
        // Wrap in ParseResult-like struct for compatibility
        struct ParseResultShim {
            file: parser::HaxeFile,
        }
        let parse_result = ParseResultShim { file: haxe_file };

        self.compile_ast_with_shared_state(
            filename,
            source,
            &parse_result.file,
            skip_pre_registration,
            skip_stdlib_merge,
        )
    }


    pub(crate) fn compile_ast_with_shared_state(
        &mut self,
        filename: &str,
        source: &str,
        ast_file: &parser::HaxeFile,
        skip_pre_registration: bool,
        skip_stdlib_merge: bool,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        use crate::tast::ast_lowering::AstLowering;
        if self.config.profile_typecheck {
            self.typecheck_timings.files_seen += 1;
        }
        let profile_file_detail = self.config.profile_typecheck
            && std::env::var_os("RAYZOR_PROFILE_TYPECHECK_FILES").is_some();
        let file_total = profile_timer(profile_file_detail);
        let mut file_ast_ms = 0.0;
        let mut file_hir_ms = 0.0;
        let mut file_mir_prep_ms = 0.0;
        let mut file_mir_ms = 0.0;
        let mut file_merge_ms = 0.0;

        // Allocate (or look up) a compilation-level file_id for this file.
        // Previously hardcoded to FileId(0), which caused every TypedExpression
        // span in every file (Main.hx, BPETokenizer.hx, GenerationLoop.hx,
        // …) to carry the same file_id=0 — see
        // bugs_diagnostic_span_file_id_always_zero. The counter is monotonic
        // in arrival order; the filename map dedupes when the same file is
        // re-entered (e.g. via `compiled_files` cache lookup downstream).
        let file_id_u32 = *self
            .file_id_by_filename
            .entry(filename.to_string())
            .or_insert_with(|| {
                let id = self.next_file_id;
                self.next_file_id += 1;
                id
            });
        // Capture the EXACT source bytes the span_converter sees so the
        // renderer's source_map can use the same bytes for ariadne's
        // byte_offset → line/column resolution.
        self.file_source_by_filename
            .entry(filename.to_string())
            .or_insert_with(|| source.to_string());
        let file_id = diagnostics::FileId::new(file_id_u32 as usize);

        // Extract type info from AST for BLADE cache (before macros may modify it)
        if self.config.enable_cache {
            let type_info = bsym::extract_type_info_from_ast(ast_file);
            self.last_compiled_type_info = Some(type_info);
        }

        // Stage 1.5: Macro expansion (if enabled)
        let t_macro = profile_timer(self.config.profile_typecheck);
        let macro_expansion_needed = self.config.pipeline_config.enable_macro_expansion
            && self.macro_expansion_may_apply(ast_file);
        // Typer-dependent macro calls parked by expansion; lowering re-expands
        // them at their sites. The expander must outlive `lowering` below.
        let mut deferred_macro_calls: Vec<crate::macro_system::expander::DeferredMacroCall> =
            Vec::new();
        let mut deferred_macro_expander: Option<
            std::cell::RefCell<crate::macro_system::MacroExpander>,
        > = None;
        let ast_file_owned;
        let ast_file = if macro_expansion_needed {
            let mut class_registry = crate::macro_system::ClassRegistry::new();
            class_registry.register_files(&self.stdlib_files);
            class_registry.register_files(&self.import_hx_files);
            class_registry.register_files(&self.loaded_import_haxe_files);
            class_registry.register_file(ast_file);
            // Phase 2 fix: pass user files AND macro-bearing import files as
            // "dependency" files so cross-file macros (e.g.
            // `import tink.Json` + `tink.Json.parse(...)`) are discovered.
            // Without this, only the current file's macros are in the registry
            // and cross-file calls silently fall through.
            let mut dep_files: Vec<HaxeFile> = Vec::new();
            dep_files.extend(self.user_files.iter().cloned());
            dep_files.extend(self.loaded_import_haxe_files.iter().cloned());
            let (expansion, kept_expander) =
                crate::macro_system::expander::expand_macros_with_dependencies_keep(
                    ast_file.clone(),
                    class_registry,
                    &dep_files,
                );
            deferred_macro_calls = expansion.deferred.clone();
            if !deferred_macro_calls.is_empty() {
                deferred_macro_expander = Some(std::cell::RefCell::new(kept_expander));
            }
            // Surface macro expansion diagnostics to the user, not just to
            // debug logs. A silent fallthrough is much worse than a loud
            // error — a failed macro call otherwise routes to a regular
            // method (often the stdlib namesake) with no indication the
            // macro didn't run.
            let mut macro_diagnostics: Vec<diagnostics::Diagnostic> = Vec::new();
            // The expander re-walks a dirty declaration once per iteration and
            // records the same failure each time, so the batch repeats itself.
            let mut seen_macro_diags: std::collections::BTreeSet<(String, u32, u32)> =
                std::collections::BTreeSet::new();
            for diag in &expansion.diagnostics {
                // Info is a per-macro registration trace and would spam.
                if matches!(diag.severity, crate::macro_system::MacroSeverity::Info) {
                    continue;
                }
                let severity = match diag.severity {
                    crate::macro_system::MacroSeverity::Error => {
                        diagnostics::DiagnosticSeverity::Error
                    }
                    crate::macro_system::MacroSeverity::Warning => {
                        diagnostics::DiagnosticSeverity::Warning
                    }
                    crate::macro_system::MacroSeverity::Info => {
                        diagnostics::DiagnosticSeverity::Info
                    }
                };
                let loc = &diag.location;
                let pos = diagnostics::SourcePosition::new(
                    loc.line.max(1) as usize,
                    loc.column.max(1) as usize,
                    loc.byte_offset as usize,
                );
                let end_pos = diagnostics::SourcePosition::new(
                    loc.line.max(1) as usize,
                    (loc.column.max(1) + 1) as usize,
                    (loc.byte_offset + 1) as usize,
                );
                if !seen_macro_diags.insert((diag.message.clone(), loc.line, loc.column)) {
                    continue;
                }
                let span = diagnostics::SourceSpan::new(pos, end_pos, diagnostics::FileId::new(0));
                macro_diagnostics.push(diagnostics::Diagnostic {
                    severity,
                    code: Some("MACRO".to_string()),
                    message: format!("macro expansion in {}: {}", filename, diag.message),
                    span,
                    labels: Vec::new(),
                    suggestions: Vec::new(),
                    notes: Vec::new(),
                    help: Vec::new(),
                });
                if matches!(diag.severity, crate::macro_system::MacroSeverity::Error) {
                    debug!("Macro expansion error in {}: {}", filename, diag.message);
                }
            }
            // PRINT them. Collecting alone only stores for cache replay, so the
            // reason a macro did not run was never shown: the macro's own
            // definition is stripped after expansion, and the user saw nothing
            // but `Cannot find name '<macro>'` at the call site.
            if !macro_diagnostics.is_empty() {
                self.print_mir_diagnostics(&macro_diagnostics);
            }
            if expansion.expansions_count > 0 {
                debug!(
                    "Macro expansion: {} macros expanded in {}",
                    expansion.expansions_count, filename
                );
            }
            // Store expansion origins for LSP macro hints
            self.macro_expansions.extend(expansion.expansion_origins);
            ast_file_owned = expansion.file;
            &ast_file_owned
        } else {
            if self.config.profile_typecheck && self.config.pipeline_config.enable_macro_expansion {
                self.typecheck_timings.macro_skipped_files += 1;
            }
            ast_file
        };
        add_profile_ms(&mut self.typecheck_timings.macro_ms, t_macro);

        // Lower to TAST using the SHARED state
        // NOTE: AstLowering needs an Rc<RefCell<StringInterner>> for TypedFile
        // We create a dummy one here - the actual interning happens via the &mut reference
        // TODO: Refactor CompilationUnit to store string_interner as Rc<RefCell<>> from the start
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Skip pre-registration if requested (types already registered by CompilationUnit)
        lowering.set_skip_pre_registration(skip_pre_registration);

        // CompilationUnit manages stdlib loading itself via load_stdlib() and the
        // later stdlib MIR merge. Re-loading the stdlib inside AstLowering causes
        // the uncached path to pull in a different symbol/method set than the
        // cached path, which changes bundle contents and breaks DeltaBlue parity.
        lowering.set_skip_stdlib_loading(true);

        // Declared-static-signature index: lets call sites type statics whose
        // declaring file lowers later (no untyped-placeholder decay).
        lowering.set_static_sig_index(Rc::clone(&self.static_sig_index));

        // Re-expansion of typer-dependent macro calls at their sites.
        if let Some(ref expander_cell) = deferred_macro_expander {
            if !deferred_macro_calls.is_empty() {
                lowering
                    .set_deferred_macros(expander_cell, std::mem::take(&mut deferred_macro_calls));
            }
        }

        // Seed class_fields from previously compiled files.
        // Only seed classes that have actual fields — empty entries interfere with
        // static method resolution by making the class "exist" in class_fields but
        // with no matching field, causing the static method to fall through to a
        // generic path instead of the stdlib dispatch.
        if !self.global_class_fields.is_empty() {
            let non_empty: BTreeMap<_, _> = self
                .global_class_fields
                .iter()
                .filter(|(_, fields)| !fields.is_empty())
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            if !non_empty.is_empty() {
                lowering.seed_class_fields(&non_empty);
            }
        }

        lowering.initialize_span_converter_with_filename(
            file_id.as_usize() as u32,
            source.to_string(),
            filename.to_string(),
        );

        let t_ast_lower = profile_timer(self.config.profile_typecheck);
        let typed_file = lowering
            .lower_file(ast_file)
            .map_err(|e| vec![e.to_compilation_error()])?;
        file_ast_ms = finish_profile_ms(&mut self.typecheck_timings.ast_lower_ms, t_ast_lower);

        // Export class_fields for subsequent compilations
        for (class_sym, fields) in lowering.export_class_fields() {
            self.global_class_fields
                .entry(*class_sym)
                .or_insert_with(|| fields.clone());
        }

        // Normal (non-safety) warnings: untyped empty array literals whose
        // element type stayed uncertain (never bound by a push/assign), so they
        // remain Array<Dynamic>. Always emitted — unlike ownership/safety
        // warnings these are not gated by `emit_safety_warnings`. (Last use of
        // `lowering` so its &mut borrow ends before the diagnostics push.)
        let array_warnings = lowering.take_empty_array_warnings();
        for (loc, msg) in array_warnings {
            let pos = diagnostics::SourcePosition::new(
                loc.line.max(1) as usize,
                loc.column.max(1) as usize,
                loc.byte_offset as usize,
            );
            let end_pos = diagnostics::SourcePosition::new(
                loc.line.max(1) as usize,
                (loc.column.max(1) + 1) as usize,
                (loc.byte_offset + 1) as usize,
            );
            let span = diagnostics::SourceSpan::new(pos, end_pos, diagnostics::FileId::new(0));
            self.collected_diagnostics.push(diagnostics::Diagnostic {
                severity: diagnostics::DiagnosticSeverity::Warning,
                code: Some("W0110".to_string()),
                message: msg,
                span,
                labels: Vec::new(),
                suggestions: Vec::new(),
                notes: Vec::new(),
                help: Vec::new(),
            });
        }

        // Send/Sync validation — check thread safety constraints (user files only)
        let is_stdlib = filename.contains("haxe-std/") || filename.contains("haxe-std\\");
        let t_send_sync = profile_timer(self.config.profile_typecheck);
        if !is_stdlib {
            use crate::tast::send_sync_validator::SendSyncValidator;
            let validator = SendSyncValidator::new(
                &self.type_table,
                &self.symbol_table,
                &self.string_interner,
                &typed_file.classes,
            );
            let mut send_sync_errors: Vec<CompilationError> = Vec::new();
            let collect_error =
                |error: crate::tast::send_sync_validator::SendSyncError| CompilationError {
                    message: error.message.clone(),
                    location: error.source_location,
                    category: ErrorCategory::ConcurrencyError,
                    suggestion: Some(
                        "Add @:derive([Send]) or @:derive([Send, Sync]) to the type".to_string(),
                    ),
                    related_errors: Vec::new(),
                };
            for class in &typed_file.classes {
                if let Err(error) = validator.validate_class(class) {
                    send_sync_errors.push(collect_error(error));
                }
                // Soundness: a class explicitly deriving Send/Sync must have
                // fields that fulfill the trait (extern types skip — opaque).
                for error in validator.validate_derive_soundness(class) {
                    send_sync_errors.push(collect_error(error));
                }
            }
            for function in &typed_file.functions {
                if let Err(error) = validator.validate_function(function) {
                    send_sync_errors.push(collect_error(error));
                }
            }
            if !send_sync_errors.is_empty() {
                add_profile_ms(&mut self.typecheck_timings.send_sync_ms, t_send_sync);
                return Err(send_sync_errors);
            }
        } // end if !is_stdlib
        add_profile_ms(&mut self.typecheck_timings.send_sync_ms, t_send_sync);

        // Ownership analysis: use-after-move detection (user files only).
        //
        // `@:safety` is an OPT-IN (docs/architecture/MEMORY_MANAGEMENT.md): with no
        // annotation a class is runtime-managed and gets no analysis. That default
        // is what lets libraries written for the Haxe ecosystem — which alias
        // freely and assume a GC — keep compiling untouched.
        let opted_into_safety = typed_file.classes.iter().any(|c| c.has_safety_annotation());
        let t_ownership = profile_timer(self.config.profile_typecheck);
        if !is_stdlib {
            let mut ownership_diagnostics = self.check_ownership_violations(&typed_file);
            // Advisory diagnostics belong to code that asked for them; a `@:move`
            // violation is a compile error everywhere, on every path that produces
            // an artifact.
            //
            // The polarity here was inverted, and each half was reachable by
            // accident. `@:safety` SUPPRESSED a file rather than enrolling it, so
            // the annotation `@:move`'s own documentation calls a prerequisite was
            // the way to disable the checking it enables. And `emit_safety_warnings`
            // is cleared by the bundle and AOT drivers, so the artifacts we ship
            // were the only ones never checked — a program `rayzor run` rejected
            // would bundle cleanly and then run the use-after-move.
            if !opted_into_safety || !self.config.emit_safety_warnings {
                ownership_diagnostics
                    .retain(|d| matches!(d.severity, diagnostics::DiagnosticSeverity::Error));
            }
            if !ownership_diagnostics.is_empty() {
                // Print everything (so the user sees warnings AND the error
                // labels/help text), then if any diagnostic was strict
                // (`@:move`) we fail compilation with a hard error.
                self.print_mir_diagnostics(&ownership_diagnostics);
                let strict_errors: Vec<CompilationError> = ownership_diagnostics
                    .iter()
                    .filter(|d| matches!(d.severity, diagnostics::DiagnosticSeverity::Error))
                    .map(|d| CompilationError {
                        message: d.message.clone(),
                        location: SourceLocation {
                            file_id: d.span.file_id.as_usize() as u32,
                            byte_offset: d.span.start.byte_offset as u32,
                            line: d.span.start.line as u32,
                            column: d.span.start.column as u32,
                        },
                        category: ErrorCategory::OwnershipError,
                        suggestion: d.help.first().cloned(),
                        related_errors: Vec::new(),
                    })
                    .collect();
                if !strict_errors.is_empty() {
                    add_profile_ms(&mut self.typecheck_timings.ownership_ms, t_ownership);
                    return Err(strict_errors);
                }
            }
        }
        add_profile_ms(&mut self.typecheck_timings.ownership_ms, t_ownership);

        // Lower to HIR — pass loaded stdlib typed files so cross-file
        // static inline var references can be resolved
        use crate::ir::tast_to_hir::lower_tast_to_hir_with_imports;
        let import_refs: Vec<&crate::tast::node::TypedFile> =
            self.loaded_stdlib_typed_files.iter().collect();
        // (import_refs passed for inline var seeding)
        let t_hir = profile_timer(self.config.profile_typecheck);
        let hir_module = match lower_tast_to_hir_with_imports(
            &typed_file,
            &self.symbol_table,
            &self.type_table,
            &mut self.string_interner,
            None, // No semantic graphs for now
            &import_refs,
            &self.global_inline_vars,
        ) {
            Ok(module) => module,
            Err(errors) => {
                add_profile_ms(&mut self.typecheck_timings.hir_ms, t_hir);
                return Err(errors
                    .into_iter()
                    .map(|e| CompilationError {
                        message: e.message,
                        location: e.location,
                        category: ErrorCategory::TypeError,
                        suggestion: None,
                        related_errors: Vec::new(),
                    })
                    .collect::<Vec<_>>());
            }
        };
        file_hir_ms = finish_profile_ms(&mut self.typecheck_timings.hir_ms, t_hir);

        // Set source file path on HIR module for stack trace source info
        let mut hir_module = hir_module;
        hir_module.metadata.source_file = filename.to_string();

        // Check if this file contains ONLY extern class declarations BEFORE MIR lowering.
        // Extern class files only need TAST+HIR for type system registration (symbol scopes,
        // method signatures). Their runtime code is provided by build_stdlib() from Rust
        // implementations. Generating MIR stubs here would create function entries with wrong
        // signatures (0-param stubs for methods that need a receiver), breaking codegen.
        let t_extern_check = profile_timer(self.config.profile_typecheck);
        {
            use crate::tast::symbols::SymbolFlags;
            let has_non_extern_class = typed_file.classes.iter().any(|c| {
                !self
                    .symbol_table
                    .get_symbol(c.symbol_id)
                    .map(|s| s.flags.contains(SymbolFlags::EXTERN))
                    .unwrap_or(false)
            });
            let has_non_extern_abstract = typed_file.abstracts.iter().any(|a| {
                !self
                    .symbol_table
                    .get_symbol(a.symbol_id)
                    .map(|s| s.flags.contains(SymbolFlags::EXTERN))
                    .unwrap_or(false)
            });
            let has_extern_decls =
                !typed_file.classes.is_empty() || !typed_file.abstracts.is_empty();
            // An extern class may still declare a concrete (non-`@:native`) method
            // with a body — e.g. a small helper over other extern methods. Those
            // bodies need MIR: the per-method guards in hir_to_mir already skip the
            // bodyless extern methods, so only the concrete ones get lowered.
            let has_concrete_method = typed_file
                .classes
                .iter()
                .flat_map(|c| c.methods.iter())
                .chain(typed_file.abstracts.iter().flat_map(|a| a.methods.iter()))
                .any(|m| !m.body.is_empty());
            let is_extern_only = has_extern_decls
                && !has_non_extern_class
                && !has_non_extern_abstract
                && !has_concrete_method
                && typed_file.functions.is_empty()
                && typed_file.enums.is_empty();
            if is_extern_only {
                debug!(
                    "[EXTERN_ONLY] Skipping MIR for extern-only file: {}",
                    filename
                );
                self.compiled_files
                    .insert(filename.to_string(), typed_file.clone());
                add_profile_ms(&mut self.typecheck_timings.extern_check_ms, t_extern_check);
                return Ok(typed_file);
            }
        }
        add_profile_ms(&mut self.typecheck_timings.extern_check_ms, t_extern_check);

        // Lower to MIR
        // Use lower_hir_to_mir_with_function_map to:
        // 1. Pass external function references from previously compiled stdlib files
        // 2. Collect function mappings for stdlib files so user code can call them
        use crate::ir::hir_to_mir::lower_hir_to_mir_with_function_map;

        // Check if this is a stdlib file BEFORE lowering so we can decide whether
        // to collect function mappings
        let is_stdlib_file = filename.contains("haxe-std")
            || filename.contains("/haxe-std/")
            || filename.contains("\\haxe-std\\");

        debug!(
            "[MIR_LOWER] filename='{}', is_stdlib_file={}, classes={}",
            filename,
            is_stdlib_file,
            typed_file.classes.len()
        );

        // For user files, pass the stdlib function map so they can call stdlib functions
        // For stdlib files, pass an empty map (they can call each other once we accumulate the map)
        let external_functions = if is_stdlib_file {
            // Stdlib files can call previously compiled stdlib functions
            self.stdlib_function_map.clone()
        } else {
            // User files can call all compiled stdlib functions
            self.stdlib_function_map.clone()
        };

        // Name-based external function map for cross-file lookups where SymbolIds differ
        let external_functions_by_name = self.stdlib_function_name_map.clone();

        let stdlib_mapping = self.compiler_plugin_registry.build_combined_mapping();

        let t_mir_prep = profile_timer(self.config.profile_typecheck);
        let constructor_param_counts = self.import_constructor_param_counts.clone();
        let external_function_param_types = self.import_function_param_types.clone();

        // Seed cross-file property accessors from loaded stdlib typed files.
        // Extern-only files like sys/thread/Tls.hx skip MIR generation (handled
        // by `is_extern_only` above) so their property fields never reach
        // MirContext::register_class_metadata. Without this seed, user code
        // like `tls.value` falls through to a "field not found" error.
        // Equivalent to the BLADE-cache restoration path at line ~3020 but
        // also covers fresh (uncached) stdlib loads.
        let stdlib_files: Vec<_> = self
            .loaded_stdlib_typed_files
            .iter()
            .map(|f| f as *const _)
            .collect();
        for tf_ptr in stdlib_files {
            let tf = unsafe { &*tf_ptr };
            self.seed_property_accessors_from_typed_file(tf);
        }

        // Save external constructor keys to filter them out of the result
        let external_constructor_keys: std::collections::BTreeSet<String> =
            self.import_constructor_name_map.keys().cloned().collect();

        // Globals from modules already lowered, keyed by qualified name. Their
        // ids are final: imports are renumbered into disjoint ranges before this
        // module is lowered.
        let external_globals = self.import_external_globals.clone();
        file_mir_prep_ms = finish_profile_ms(&mut self.typecheck_timings.mir_prep_ms, t_mir_prep);

        let t_mir = profile_timer(self.config.profile_typecheck);
        let mir_result = match lower_hir_to_mir_with_function_map(
            &hir_module,
            &self.string_interner,
            &self.type_table,
            &self.symbol_table,
            external_functions,
            external_functions_by_name,
            external_globals,
            &stdlib_mapping,
            self.import_field_index_map.clone(),
            self.import_property_access_map.clone(),
            self.import_constructor_name_map.clone(),
            self.import_class_alloc_sizes.clone(),
            self.import_class_method_symbols.clone(),
            self.import_class_type_to_symbol.clone(),
            constructor_param_counts,
            external_function_param_types,
            self.import_class_alloc_sizes_by_name.clone(),
            self.import_interface_method_names.clone(),
            self.import_interface_method_return_types.clone(),
            self.import_interface_extends.clone(),
            self.import_interface_vtables.clone(),
            self.import_function_param_iface_names.clone(),
            self.import_field_class_names.clone(),
            Some(Rc::clone(&self.static_sig_index)),
        ) {
            Ok(result) => result,
            Err(errors) => {
                add_profile_ms(&mut self.typecheck_timings.mir_ms, t_mir);
                add_profile_ms(&mut self.typecheck_timings.mir_lower_core_ms, t_mir);
                return Err(errors
                    .into_iter()
                    .map(|e| CompilationError {
                        message: e.message,
                        location: e.location,
                        category: ErrorCategory::TypeError,
                        suggestion: None,
                        related_errors: Vec::new(),
                    })
                    .collect::<Vec<_>>());
            }
        };
        file_mir_ms = finish_profile_ms(&mut self.typecheck_timings.mir_ms, t_mir);
        self.typecheck_timings.mir_lower_core_ms += file_mir_ms;

        // Print any diagnostics from MIR lowering (e.g., exhaustiveness warnings)
        if !mir_result.diagnostics.is_empty() {
            self.print_mir_diagnostics(&mir_result.diagnostics);
        }

        // An ownership violation found at MIR is fatal. Printing it and
        // carrying on would emit a binary the analysis just said is unsound,
        // and — because a successful compile populates the cache — the next
        // run would skip lowering and never report it again.
        let fatal: Vec<&diagnostics::Diagnostic> = mir_result
            .diagnostics
            .iter()
            .filter(|d| {
                d.severity == diagnostics::DiagnosticSeverity::Error
                    && matches!(
                        d.code.as_deref(),
                        Some("E0382") | Some("E0383") | Some("E0384") | Some("E0300")
                    )
            })
            .collect();
        if !fatal.is_empty() {
            // One summary, not one per diagnostic: each has already been
            // rendered above with its source spans, and repeating the message
            // as a bare error line only doubles it.
            let first = fatal[0];
            return Err(vec![CompilationError {
                message: format!("ownership check failed: {} error(s)", fatal.len()),
                location: SourceLocation {
                    file_id: first.span.file_id.as_usize() as u32,
                    byte_offset: first.span.start.byte_offset as u32,
                    line: first.span.start.line as u32,
                    column: first.span.start.column as u32,
                },
                category: ErrorCategory::TypeError,
                suggestion: first.help.first().cloned(),
                related_errors: Vec::new(),
            }]);
        }

        // Capture user-defined function IDs before module is consumed
        let mir_result_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            mir_result.function_map.values().copied().collect();
        let mir_result_ctor_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            mir_result.constructor_name_map.values().copied().collect();

        let mut mir_module = mir_result.module;

        // (MIR dump moved to after stdlib merge)

        // Build BladeCachedMaps for BLADE cache (name-keyed, before ID-keyed accumulation consumes the data)
        if self.config.enable_cache {
            let cached_maps = self.build_cached_maps_from_mir_result(
                &mir_result.function_map,
                &mir_result.field_index_map,
                &mir_result.constructor_name_map,
                &mir_result.class_alloc_sizes,
                &mir_result.field_class_names,
                &mir_result.property_access_map,
                &mir_result.function_param_hir_types,
                &mir_result.interface_vtables,
                &mir_result.interface_method_names,
                &mir_result.interface_method_return_types,
                &mir_result.interface_extends,
            );
            self.last_compiled_cached_maps = Some(cached_maps);
        }

        // Collect SymbolId-based function mappings from ALL files (stdlib + imports)
        // This enables cross-file method calls: user file can call import file methods
        // via the shared symbol table (SymbolIds are consistent across files)
        debug!(
            "DEBUG: Collecting {} function mappings from file: {}",
            mir_result.function_map.len(),
            filename
        );
        // (max_own_func_id computed earlier before module was moved)
        for (symbol_id, func_id) in mir_result.function_map {
            self.stdlib_function_map.insert(symbol_id, func_id);
        }

        // Collect constructor name map — only include constructors NEW to this file,
        // not external ones that were passed in via import_constructor_name_map.
        for (class_name, func_id) in mir_result.constructor_name_map {
            if !external_constructor_keys.contains(&class_name) {
                self.import_constructor_name_map.insert(class_name, func_id);
            }
        }

        // Collect class allocation sizes from ALL files
        for (type_id, size) in mir_result.class_alloc_sizes {
            self.import_class_alloc_sizes.insert(type_id, size);
        }

        // Collect name-keyed class allocation sizes (stable across compilation contexts)
        for (name, size) in mir_result.class_alloc_sizes_by_name {
            self.import_class_alloc_sizes_by_name.insert(name, size);
        }

        // Collect class method symbols from ALL files
        for (key, sym) in mir_result.class_method_symbols {
            self.import_class_method_symbols.insert(key, sym);
        }

        // Collect name-based mappings for cross-file lookups.
        // Use qualified names to avoid collisions (e.g., "current" matching
        // both ArrayIterator.current field and Thread.current method).
        // For stdlib files: all functions with non-empty bodies.
        // For user packages: only functions with qualified names (to avoid
        // polluting the namespace with bare names like "new").
        if is_stdlib_file {
            for (func_id, func) in &mir_module.functions {
                if !func.cfg.blocks.is_empty() {
                    let map_name = func.qualified_name.as_deref().unwrap_or(&func.name);
                    self.stdlib_function_name_map
                        .insert(map_name.to_string(), *func_id);
                }
            }
        } else {
            // User packages: only add functions with qualified names
            for (func_id, func) in &mir_module.functions {
                if !func.cfg.blocks.is_empty() {
                    if let Some(qn) = func.qualified_name.as_deref() {
                        self.stdlib_function_name_map
                            .insert(qn.to_string(), *func_id);
                    }
                }
            }
        }

        // Accumulate field index and property access maps from all compiled files
        // (both stdlib and imports) so user files can resolve field access on imported classes
        for (sym, val) in mir_result.field_index_map {
            self.import_field_index_map.insert(sym, val);
        }
        for (sym, name) in mir_result.field_class_names {
            self.import_field_class_names.insert(sym, name);
        }
        for (sym, val) in mir_result.property_access_map {
            self.import_property_access_map.insert(sym, val);
        }
        for (ty, sym) in mir_result.class_type_to_symbol {
            self.import_class_type_to_symbol.insert(ty, sym);
        }

        // Accumulate interface metadata so subsequent files can resolve
        // cross-file interface dispatch (variable typed as the interface,
        // method calls on it, fat-pointer wrapping at construction).
        for (sym, methods) in mir_result.interface_method_names {
            self.import_interface_method_names.insert(sym, methods);
        }
        for (key, ty) in mir_result.interface_method_return_types {
            self.import_interface_method_return_types.insert(key, ty);
        }
        for (sym, parents) in mir_result.interface_extends {
            self.import_interface_extends.insert(sym, parents);
        }
        for (key, vtable) in mir_result.interface_vtables {
            self.import_interface_vtables.insert(key, vtable);
        }

        // Harvest per-function HIR param types so the user file's
        // `maybe_materialize_for_call` Path 3 can recover class→interface
        // wrap decisions for imported constructors. MIR alone erases both
        // Class and Interface to `Ptr(Void)`; without these names the
        // wrap is silently skipped and raw class pointers land in
        // interface-typed fields (SIGBUS on first vtable dispatch).
        {
            let type_table = self.type_table.borrow();
            let string_interner = &self.string_interner;
            let symbol_table = &self.symbol_table;
            let resolve_name = |ty: crate::tast::TypeId| -> Option<String> {
                let info = type_table.get(ty)?;
                let symbol_id = match &info.kind {
                    crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    crate::tast::TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                    // Constrained type parameter `<T:Iface>`: forward the
                    // interface constraint's name. The importer's context sees
                    // this param as a TypeParameter with an EMPTY constraint
                    // list (constraints don't survive the module boundary), so
                    // it can't recover the interface to wrap the class arg —
                    // resolve it here in the declaring module where the
                    // constraint is present, and the name-based call-site wrap
                    // (external_function_param_iface_names) fires cross-module.
                    crate::tast::TypeKind::TypeParameter { constraints, .. } => constraints
                        .iter()
                        .find_map(|cid| match type_table.get(*cid) {
                            Some(ct) => {
                                if let crate::tast::TypeKind::Interface { symbol_id, .. } = &ct.kind
                                {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            }
                            None => None,
                        }),
                    _ => None,
                }?;
                let sym = symbol_table.get_symbol(symbol_id)?;
                sym.qualified_name
                    .and_then(|n| string_interner.get(n))
                    .or_else(|| string_interner.get(sym.name))
                    .map(|s| s.to_string())
            };
            for (func_id, type_ids) in &mir_result.function_param_hir_types {
                let names: Vec<Option<String>> =
                    type_ids.iter().copied().map(resolve_name).collect();
                if names.iter().any(|n| n.is_some()) {
                    self.import_function_param_iface_names
                        .insert(*func_id, names);
                }
            }
        }

        // NOTE: extern-only files are handled above (before MIR generation).

        // Capture this file's user-defined function IDs (from function_map + constructor_map).
        // Exclude MIR wrapper stubs — these have trivial bodies (ret false/null) and must be
        // replaced by the stdlib merge with real implementations that delegate to runtime.
        // Without this filter, non-extern stdlib classes like EReg have their methods (match,
        // matched, etc.) incorrectly protected from replacement.
        let own_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> = mir_result_func_ids
            .union(&mir_result_ctor_ids)
            .copied()
            .filter(|func_id| {
                mir_module
                    .functions
                    .get(func_id)
                    .map(|f| !matches!(f.kind, crate::ir::FunctionKind::MirWrapper))
                    .unwrap_or(true)
            })
            .collect();

        // Store for try_compile_import to pick up and track after renumbering
        self.last_compiled_own_func_ids = Some(own_func_ids.clone());

        // The stdlib merge (imports + stdlib MIR wrappers) should only happen for the
        // final user file, not for import files. Import files produce clean MIR with only
        // their own functions. When the main file is compiled, all imports are merged in
        // first, then a single stdlib merge resolves forward refs without corrupting
        // user package functions.
        let t_stdlib_merge = profile_timer(self.config.profile_typecheck);
        if !is_stdlib_file && !skip_stdlib_merge {
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
                    let is_known_stdlib_wrapper =
                        stdlib_mapping.is_mir_wrapper_function(&func.name);
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
                    let identity = func.qualified_name.clone().unwrap_or_else(|| func.name.clone());
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

            debug!("DEBUG: Renumbering stdlib functions with offset {} (max_user_func={}, max_user_extern={})",
                  offset, max_user_func_id, max_user_extern_id);

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
                let dropped =
                    retain_referenced_stdlib_functions(&mut stdlib_mir, &needed_stdlib_names);
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

            for (func_id, func) in &renumbered_functions {
                if let Some(existing_ids) = user_func_name_to_ids.get(&func.name) {
                    for &existing_id in existing_ids {
                        // Protect source-level import functions from stdlib replacement.
                        if !merged_import_func_ids.contains(&existing_id) {
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
                        if !merged_import_func_ids.contains(&existing_id) {
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
                                    func_id: ref mut called_func_id,
                                    ..
                                } => {
                                    if let Some(&new_id) = id_replacements.get(called_func_id) {
                                        *called_func_id = new_id;
                                    }
                                }
                                IrInstruction::FunctionRef {
                                    func_id: ref mut ref_func_id,
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
                                    func_id: ref mut closure_func_id,
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
        } // end if !is_stdlib_file (stdlib merge + renumbering)
        file_merge_ms =
            finish_profile_ms(&mut self.typecheck_timings.stdlib_merge_ms, t_stdlib_merge);

        // Dump MIR after stdlib merge so wrapper bodies are visible
        if std::env::var("RAYZOR_DUMP_MIR").is_ok() {
            eprintln!("=== MIR DUMP (post-merge) for {} ===", filename);
            eprintln!("{}", crate::ir::dump::dump_module(&mir_module));
            eprintln!("=== END MIR DUMP ===");
        }

        // Run monomorphization pass to specialize generic functions
        let t_monomorphize = profile_timer(self.config.profile_typecheck);
        let mut monomorphizer = Monomorphizer::new();
        monomorphizer.monomorphize_module(&mut mir_module);
        add_profile_ms(&mut self.typecheck_timings.monomorphize_ms, t_monomorphize);
        // let mono_stats = monomorphizer.stats();
        // if mono_stats.generic_functions_found > 0 || mono_stats.instantiations_created > 0 {
        //     debug!("DEBUG: Monomorphization stats: {} generic functions, {} instantiations, {} call sites rewritten",
        //               mono_stats.generic_functions_found,
        //               mono_stats.instantiations_created,
        //               mono_stats.call_sites_rewritten);
        // }

        // // Debug: dump alloc sizes
        // for (name, size) in &self.import_class_alloc_sizes_by_name {
        //     if name.contains("Point") || name.contains("Particle") || name.contains("Simulation") {
        //         eprintln!("[ALLOC_SIZE] {} → {} bytes", name, size);
        //     }
        // }
        // // Debug: dump constructor name map
        // for (name, fid) in &self.import_constructor_name_map {
        //     eprintln!("[CTOR_MAP] {} → fn{}", name, fid.0);
        // }
        // // Debug: dump constructor signatures in final merged module
        // for (fid, func) in &mir_module.functions {
        //     if func.name == "new" && !func.cfg.blocks.is_empty() {
        //         let params: Vec<String> = func.signature.parameters.iter()
        //             .map(|p| format!("{}:{:?}", p.name, p.ty)).collect();
        //         let qn = func.qualified_name.as_deref().unwrap_or("?");
        //         let blocks = func.cfg.blocks.len();
        //         let insts: usize = func.cfg.blocks.values().map(|b| b.instructions.len()).sum();
        //         // eprintln!("[FINAL_MIR] fn{}={} qn={} blocks={} insts={} new({})", fid.0, func.name, qn, blocks, insts, params.join(", "));
        //         // Dump instructions for Point2D-sized constructors
        //         // if params.len() == 3 && params[1].contains("F64") {
        //         //     for block in func.cfg.blocks.values() {
        //         //         for inst in &block.instructions {
        //         //             eprintln!("[FINAL_MIR]   {:?}", inst);
        //         //         }
        //         //     }
        //         // }
        //     }
        // }

        // Store the MIR module
        self.mir_modules.push(std::sync::Arc::new(mir_module));

        // Mark as successfully compiled to prevent redundant recompilation
        self.compiled_files
            .insert(filename.to_string(), typed_file.clone());

        if profile_file_detail {
            let total_ms = file_total.map(elapsed_ms).unwrap_or(0.0);
            eprintln!(
                "  typecheck-file: total={:.2}ms ast={:.2}ms hir={:.2}ms mir_prep={:.2}ms mir={:.2}ms merge={:.2}ms file={}",
                total_ms,
                file_ast_ms,
                file_hir_ms,
                file_mir_prep_ms,
                file_mir_ms,
                file_merge_ms,
                filename
            );
        }

        Ok(typed_file)
    }


    /// Compile a single file using shared state (backward-compatible wrapper).
    /// This is used for the main/final user file — stdlib merge runs.
    pub(crate) fn compile_file_with_shared_state(
        &mut self,
        filename: &str,
        source: &str,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        self.compile_file_with_shared_state_ex(filename, source, false, false)
    }


    /// Compile using an already-parsed AST (avoids redundant re-parsing).
    pub(crate) fn compile_pre_parsed_file(
        &mut self,
        ast_file: &parser::HaxeFile,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        // Skip if already compiled
        if let Some(cached) = self.compiled_files.get(&ast_file.filename) {
            return Ok(cached.clone());
        }

        let source = ast_file.input.as_deref().unwrap_or_default();
        self.compile_ast_with_shared_state(&ast_file.filename, source, ast_file, false, false)
    }


    pub(crate) fn macro_expansion_may_apply(&self, ast_file: &parser::HaxeFile) -> bool {
        haxe_file_source_has_macro_hook(ast_file)
            || self.user_files.iter().any(haxe_file_source_has_macro_hook)
            || self
                .import_hx_files
                .iter()
                .any(haxe_file_source_has_macro_hook)
            || self
                .loaded_import_haxe_files
                .iter()
                .any(haxe_file_source_has_macro_hook)
    }


    pub(crate) fn compile_user_ast_collecting_errors(
        &mut self,
        ast_file: &parser::HaxeFile,
        all_typed_files: &mut Vec<TypedFile>,
        all_errors: &mut Vec<CompilationError>,
    ) {
        match self.compile_pre_parsed_file(ast_file) {
            Ok(typed_file) => {
                all_typed_files.push(typed_file);
            }
            Err(errors) => {
                // Check if any errors are unresolved types that we can try to load on-demand
                let (loadable, other): (Vec<_>, Vec<_>) = errors.into_iter().partition(|e| {
                    e.message.contains("Unresolved type")
                        || e.message.contains("UnresolvedType")
                        || e.message.contains("Cannot find type")
                });

                // Try to load unresolved types on-demand
                let mut any_loaded = false;
                for error in loadable {
                    if let Some(type_name) = self.extract_type_name_from_error(&error.message) {
                        // Skip if we already tried to load this type and it failed
                        if self.failed_type_loads.contains(&type_name) {
                            all_errors.push(error);
                            continue;
                        }
                        if let Err(load_err) = self.load_import_file(&type_name) {
                            debug!("On-demand load failed for {}: {}", type_name, load_err);
                            self.failed_type_loads.insert(type_name.clone());
                            all_errors.push(error);
                        } else {
                            // Successfully loaded! Mark that we should retry
                            any_loaded = true;
                        }
                    } else {
                        all_errors.push(error);
                    }
                }

                // If we successfully loaded any dependencies, retry compiling this file
                if any_loaded {
                    debug!(
                        "  Retrying {} after loading dependencies...",
                        ast_file.filename
                    );
                    match self.compile_pre_parsed_file(ast_file) {
                        Ok(typed_file) => {
                            all_typed_files.push(typed_file);
                        }
                        Err(retry_errors) => {
                            // Still failed after loading dependencies
                            // Check if retry revealed NEW unresolved types that need loading
                            let (retry_loadable, retry_other): (Vec<_>, Vec<_>) =
                                retry_errors.into_iter().partition(|e| {
                                    e.message.contains("Unresolved type")
                                        || e.message.contains("UnresolvedType")
                                        || e.message.contains("Cannot find type")
                                });

                            let mut retry_loaded = false;
                            for error in retry_loadable {
                                if let Some(type_name) =
                                    self.extract_type_name_from_error(&error.message)
                                {
                                    if !self.failed_type_loads.contains(&type_name) {
                                        if let Err(load_err) = self.load_import_file(&type_name) {
                                            debug!(
                                                "On-demand load failed for {}: {}",
                                                type_name, load_err
                                            );
                                            self.failed_type_loads.insert(type_name.clone());
                                            all_errors.push(error);
                                        } else {
                                            retry_loaded = true;
                                        }
                                    } else {
                                        all_errors.push(error);
                                    }
                                } else {
                                    all_errors.push(error);
                                }
                            }

                            // If we loaded more dependencies on retry, try ONE more time
                            if retry_loaded {
                                debug!(
                                    "  Second retry of {} after loading more dependencies...",
                                    ast_file.filename
                                );
                                match self.compile_pre_parsed_file(ast_file) {
                                    Ok(typed_file) => {
                                        all_typed_files.push(typed_file);
                                    }
                                    Err(final_errors) => {
                                        all_errors.extend(final_errors);
                                    }
                                }
                            } else {
                                all_errors.extend(retry_other);
                            }
                        }
                    }
                } else {
                    // No dependencies loaded, keep original errors
                    all_errors.extend(other);
                }
            }
        }
    }


    pub fn typecheck_timings(&self) -> TypecheckStageTimings {
        self.typecheck_timings
    }


    /// Lower all files (stdlib + user) to TAST with full pipeline analysis
    ///
    /// This method delegates to HaxeCompilationPipeline for each file to leverage
    /// the complete analysis infrastructure including:
    /// - Type checking with diagnostics
    /// - Flow-sensitive analysis
    /// - Ownership and lifetime analysis
    /// - Memory safety validation
    ///
    /// Order of compilation:
    /// 1. Stdlib files (with haxe.* package)
    /// 2. Import.hx files (for global imports)
    /// 3. User files (in dependency order - dependencies first)
    ///
    /// On-demand loading: If a type is unresolved, attempts to load and compile
    /// the file that should contain it based on qualified path resolution.
    ///
    /// IMPORTANT: On error, this automatically prints formatted diagnostics to stderr

    pub fn lower_to_tast(&mut self) -> Result<Vec<TypedFile>, Vec<CompilationError>> {
        self.typecheck_timings = TypecheckStageTimings::default();

        // Step 0: Discover @:hlNative metadata in user files and load HDLL plugins
        let t_hdll = profile_timer(self.config.profile_typecheck);
        if self.config.pipeline_config.enable_semantic_analysis {
            self.discover_and_load_hdlls();
        }
        add_profile_ms(&mut self.typecheck_timings.hdll_ms, t_hdll);

        // Step 1: Analyze dependencies for user files
        // Fast path: single-file compilations don't need dependency analysis
        let t_dependency = profile_timer(self.config.profile_typecheck);
        let analysis = if self.user_files.len() <= 1 {
            DependencyAnalysis {
                compilation_order: (0..self.user_files.len()).collect(),
                circular_dependencies: Vec::new(),
            }
        } else {
            match self.analyze_dependencies() {
                Ok(a) => a,
                Err(errors) => {
                    self.print_compilation_errors(&errors);
                    return Err(errors);
                }
            }
        };
        add_profile_ms(&mut self.typecheck_timings.dependency_ms, t_dependency);

        let mut all_typed_files = Vec::new();
        let mut all_errors = Vec::new();

        // Step 2: Pre-load stdlib files for explicit imports AND using statements in user files
        // This ensures typedefs like sys.FileStat are available before compilation
        // Also handles root-level imports like "import StringTools;" and "using StringTools;"
        // Extract imports from already-parsed user file ASTs (no re-parsing needed).
        let t_import_scan = profile_timer(self.config.profile_typecheck);
        let (imports_to_load, usings_to_load): (Vec<String>, Vec<String>) =
            self.user_files.iter().fold(
                (Vec::new(), Vec::new()),
                |(mut imports, mut usings), ast| {
                    for import in &ast.imports {
                        if !import.path.is_empty() {
                            imports.push(import.path.join("."));
                        }
                    }
                    for using in &ast.using {
                        if !using.path.is_empty() {
                            usings.push(using.path.join("."));
                        }
                    }
                    let mut discovered = Vec::new();
                    collect_qualified_type_refs_from_ast(ast, &mut discovered);
                    imports.extend(discovered);
                    let user_deps: Vec<String> = Self::extract_all_dependencies(ast)
                        .into_iter()
                        .filter(|d| !d.starts_with("new:"))
                        .collect();
                    imports.extend(Self::enclosing_package_candidates(ast, &user_deps));
                    imports.extend(user_deps);
                    (imports, usings)
                },
            );
        add_profile_ms(&mut self.typecheck_timings.import_scan_ms, t_import_scan);

        // Pre-load imports using efficient topological loading (avoids retry loops)
        let mut all_imports = imports_to_load;
        all_imports.extend(usings_to_load);
        all_imports.extend(self.import_hx_type_names());
        let t_import_load = profile_timer(self.config.profile_typecheck);
        let _ = self.load_imports_efficiently(&all_imports);
        add_profile_ms(&mut self.typecheck_timings.import_load_ms, t_import_load);

        // Step 3: Compile import.hx files using SHARED state
        let import_sources: Vec<(String, String)> = self
            .import_hx_files
            .iter()
            .filter_map(|f| f.input.as_ref().map(|s| (f.filename.clone(), s.clone())))
            .collect();

        let t_import_hx = profile_timer(self.config.profile_typecheck);
        for (filename, source) in import_sources {
            match self.compile_file_with_shared_state(&filename, &source) {
                Ok(typed_file) => {
                    all_typed_files.push(typed_file);
                }
                Err(errors) => {
                    all_errors.extend(errors);
                }
            }
        }
        add_profile_ms(&mut self.typecheck_timings.import_hx_ms, t_import_hx);

        // Step 4: Compile user files in dependency order using SHARED state.
        // Use pre-parsed ASTs from self.user_files to avoid re-parsing.
        let user_file_indices: Vec<usize> = analysis.compilation_order.clone();
        let user_context_has_macro_hooks =
            self.user_files.iter().any(haxe_file_source_has_macro_hook)
                || self
                    .import_hx_files
                    .iter()
                    .any(haxe_file_source_has_macro_hook)
                || self
                    .loaded_import_haxe_files
                    .iter()
                    .any(haxe_file_source_has_macro_hook);

        let t_user_files = profile_timer(self.config.profile_typecheck);
        if user_context_has_macro_hooks {
            for idx in user_file_indices {
                let ast_file = self.user_files[idx].clone();
                self.compile_user_ast_collecting_errors(
                    &ast_file,
                    &mut all_typed_files,
                    &mut all_errors,
                );
            }
        } else {
            let user_files = std::mem::take(&mut self.user_files);
            for idx in user_file_indices {
                self.compile_user_ast_collecting_errors(
                    &user_files[idx],
                    &mut all_typed_files,
                    &mut all_errors,
                );
            }
            self.user_files = user_files;
        }
        add_profile_ms(&mut self.typecheck_timings.user_files_ms, t_user_files);

        // Step 5: Report all errors if any were found
        if !all_errors.is_empty() {
            self.print_compilation_errors(&all_errors);
            return Err(all_errors);
        }

        // Step 6: Include loaded stdlib files (typedefs, etc.) in the result
        // These were loaded on-demand during import resolution and contain type aliases
        // that need to be processed by HIR
        let t_result_stdlib = profile_timer(self.config.profile_typecheck);
        for stdlib_file in std::mem::take(&mut self.loaded_stdlib_typed_files) {
            all_typed_files.push(stdlib_file);
        }
        add_profile_ms(
            &mut self.typecheck_timings.result_stdlib_ms,
            t_result_stdlib,
        );

        self.maybe_dump_file_table();

        Ok(all_typed_files)
    }
}
