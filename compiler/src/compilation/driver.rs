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
        if let Some(cached) = self.compiled_files.get(&source_file_identity(filename)) {
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
        let identity = source_file_identity(filename);
        let retryable_import = skip_stdlib_merge
            && !self
                .user_files
                .iter()
                .any(|file| source_file_identity(&file.filename) == identity);

        // Type headers and sibling declarations supply signatures to the entry.
        let entry_name = Path::new(filename)
            .file_stem()
            .and_then(|name| name.to_str());
        let entry_index = ast_file.declarations.iter().position(|decl| {
            matches!(decl, parser::TypeDeclaration::Class(class)
            if Some(class.name.as_str()) == entry_name
                && class.fields.iter().any(|field| {
                    matches!(&field.kind, parser::ClassFieldKind::Function(function)
                        if function.name == "main")
                }))
        });
        let ordered_ast;
        let ast_file = if let Some(index) = entry_index.filter(|_| ast_file.declarations.len() > 1)
        {
            ordered_ast = {
                let mut file = ast_file.clone();
                let entry = file.declarations.remove(index);
                file.declarations.sort_by_key(|declaration| {
                    !matches!(
                        declaration,
                        parser::TypeDeclaration::Interface(_)
                            | parser::TypeDeclaration::Typedef(_)
                            | parser::TypeDeclaration::Enum(_)
                    )
                });
                file.declarations.push(entry);
                file
            };
            &ordered_ast
        } else {
            ast_file
        };

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
            // Macros read user and import files as macro code sees them,
            // `#if macro` members included; the file being compiled keeps
            // its own parse.
            let view = |f: &HaxeFile| {
                self.macro_context_view(f, None)
                    .unwrap_or_else(|| f.clone())
            };
            let current_view = self.macro_context_view(ast_file, None);
            // A module named only in `@:build(pkg.Mod.f())` is never imported,
            // so its macros would be unknown when the build runs.
            let build_modules: Vec<HaxeFile> = build_macro_module_paths(ast_file)
                .into_iter()
                .filter_map(|path| {
                    let file = self
                        .namespace_resolver
                        .resolve_qualified_path_to_file_force(&path)?;
                    let name = file.to_string_lossy().to_string();
                    let source = std::fs::read_to_string(&file).ok()?;
                    self.parse_file(&name, &source).ok()
                })
                .map(|f| view(&f))
                .collect();
            let mut class_registry = crate::macro_system::ClassRegistry::new();
            class_registry.use_statics(self.macro_statics.clone());
            class_registry.register_files(&build_modules);
            class_registry.register_files(&self.stdlib_files);
            class_registry
                .register_files(&self.import_hx_files.iter().map(view).collect::<Vec<_>>());
            class_registry.register_files(
                &self
                    .loaded_import_haxe_files
                    .iter()
                    .map(view)
                    .collect::<Vec<_>>(),
            );
            class_registry.register_file(current_view.as_ref().unwrap_or(ast_file));
            // Phase 2 fix: pass user files AND macro-bearing import files as
            // "dependency" files so cross-file macros (e.g.
            // `import tink.Json` + `tink.Json.parse(...)`) are discovered.
            // Without this, only the current file's macros are in the registry
            // and cross-file calls silently fall through.
            let mut dep_files: Vec<HaxeFile> = build_modules;
            dep_files.extend(self.user_files.iter().map(view));
            dep_files.extend(self.loaded_import_haxe_files.iter().map(view));
            if let Some(v) = current_view {
                dep_files.push(v);
            }
            // One file reached by two spellings (entry and import) is one key.
            let file_key = std::fs::canonicalize(filename)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| filename.to_string());
            let replay = self.macro_state_by_file.get(&file_key).cloned();
            let lock = |s: &crate::macro_system::class_registry::MacroStatics| {
                s.lock().map(|m| m.clone()).unwrap_or_default()
            };
            let before = match &replay {
                Some((before, _)) => {
                    if let Ok(mut statics) = self.macro_statics.lock() {
                        *statics = before.clone();
                    }
                    before.clone()
                }
                None => lock(&self.macro_statics),
            };
            let (mut expansion, kept_expander) =
                crate::macro_system::expander::expand_macros_with_dependencies_keep(
                    ast_file.clone(),
                    class_registry,
                    &dep_files,
                );
            match replay {
                Some((_, after)) => {
                    if let Ok(mut statics) = self.macro_statics.lock() {
                        *statics = after;
                    }
                    expansion.hooks.clear();
                }
                None => {
                    let after = lock(&self.macro_statics);
                    self.macro_state_by_file.insert(file_key, (before, after));
                }
            }
            deferred_macro_calls = expansion.deferred.clone();
            if kept_expander.registry().macro_count() != 0 {
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
                // Macro positions carry no file; this file expanded them.
                let mut loc = diag.location;
                loc.file_id = file_id_u32;
                let loc = &loc;
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
                let span = diagnostics::SourceSpan::new(pos, end_pos, file_id);
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
            let macro_errors: Vec<CompilationError> = expansion
                .diagnostics
                .iter()
                .filter(|diag| matches!(diag.severity, crate::macro_system::MacroSeverity::Error))
                .map(|diag| CompilationError {
                    message: format!("[E0700] {}", diag.message),
                    location: SourceLocation {
                        file_id: file_id_u32,
                        ..diag.location
                    },
                    category: ErrorCategory::MacroExpansionError,
                    suggestion: None,
                    related_errors: Vec::new(),
                })
                .collect();
            if !macro_errors.is_empty() {
                // Definitions are stripped by expansion. Lowering the failed
                // call afterwards would replace its cause with a name error.
                return Err(macro_errors);
            }
            if expansion.expansions_count > 0 {
                debug!(
                    "Macro expansion: {} macros expanded in {}",
                    expansion.expansions_count, filename
                );
            }
            // Store expansion origins for LSP macro hints
            self.macro_expansions.extend(expansion.expansion_origins);
            self.macro_hooks.extend(expansion.hooks);
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
        lowering.retrying_failed_attempt = self.failed_attempts.contains(filename);
        lowering.incomplete_classes = self.incomplete_classes.clone();
        // Only the import loop passes skip_stdlib_merge, and it retries.
        lowering.retryable_import = retryable_import;

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
            lowering.set_deferred_macros(expander_cell, std::mem::take(&mut deferred_macro_calls));
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
        let lowered = lowering.lower_file(ast_file);
        file_ast_ms = finish_profile_ms(&mut self.typecheck_timings.ast_lower_ms, t_ast_lower);

        // Export class_fields for subsequent compilations. A failed attempt
        // exports too: its members are declared, and a file compiled before
        // its retry must read `Class.staticField` as the static it is.
        // A successful retry replaces what its failed attempt exported.
        let replaces = lowered.is_ok() && self.failed_attempts.contains(filename);
        for (class_sym, fields) in lowering.export_class_fields() {
            if lowered.is_ok() {
                self.incomplete_classes.remove(class_sym);
            } else {
                self.incomplete_classes.insert(*class_sym);
            }
            if replaces {
                self.global_class_fields.insert(*class_sym, fields.clone());
            } else {
                self.global_class_fields
                    .entry(*class_sym)
                    .or_insert_with(|| fields.clone());
            }
        }
        let typed_file = lowered.map_err(|e| vec![e.to_compilation_error()])?;

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
            let span = diagnostics::SourceSpan::new(pos, end_pos, file_id);
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
                    .insert(source_file_identity(filename), typed_file.clone());
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
            self.import_abstract_cast_rules.clone(),
            Some(Rc::clone(&self.static_sig_index)),
            &self.import_param_defaults,
            // Only the import loop passes skip_stdlib_merge, and it retries.
            retryable_import,
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
        self.import_param_defaults.extend(mir_result.param_defaults);
        self.import_abstract_cast_rules
            .from
            .extend(mir_result.abstract_cast_rules.from);
        self.import_abstract_cast_rules
            .to
            .extend(mir_result.abstract_cast_rules.to);
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
            for (func_id, type_ids) in &mir_result.function_param_hir_types {
                let names: Vec<Option<String>> = type_ids
                    .iter()
                    .map(|ty| self.call_param_type_name(*ty))
                    .collect();
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
            self.merge_stdlib_into_user_module(&mut mir_module, &own_func_ids);
        } // end if !is_stdlib_file (stdlib merge + renumbering)
        file_merge_ms =
            finish_profile_ms(&mut self.typecheck_timings.stdlib_merge_ms, t_stdlib_merge);

        super::runtime_metadata::attach(&mut mir_module, ast_file).map_err(|message| {
            vec![CompilationError {
                message,
                location: SourceLocation::unknown(),
                category: ErrorCategory::TypeError,
                suggestion: None,
                related_errors: Vec::new(),
            }]
        })?;

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
        // RAYZOR_DUMP_MIR_MONO=1 dumps the module after specialization.
        if std::env::var("RAYZOR_DUMP_MIR_MONO").is_ok() {
            eprintln!("=== MIR DUMP (post-mono) for {} ===", filename);
            eprintln!("{}", crate::ir::dump::dump_module(&mir_module));
            eprintln!("=== END MIR DUMP ===");
        }
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
        self.cache_inferred_primitive_returns(&typed_file);

        // Mark as successfully compiled to prevent redundant recompilation
        self.compiled_files
            .insert(source_file_identity(filename), typed_file.clone());

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
        let identity = source_file_identity(&ast_file.filename);
        if let Some(cached) = self.compiled_files.get(&identity).cloned() {
            if let Some(index) = self
                .import_mir_modules
                .iter()
                .position(|module| source_file_identity(&module.source_file) == identity)
            {
                // Keep the first typing and its inferred member types. Its MIR
                // has already been renumbered for references from other imports.
                let mut module = self.import_mir_modules.remove(index);
                module.source_file = ast_file.filename.clone();
                let own_func_ids = module
                    .functions
                    .keys()
                    .filter(|id| self.import_own_func_ids.contains(id))
                    .copied()
                    .collect();
                let t_merge = profile_timer(self.config.profile_typecheck);
                self.merge_stdlib_into_user_module(&mut module, &own_func_ids);
                add_profile_ms(&mut self.typecheck_timings.stdlib_merge_ms, t_merge);
                let t_mono = profile_timer(self.config.profile_typecheck);
                Monomorphizer::new().monomorphize_module(&mut module);
                add_profile_ms(&mut self.typecheck_timings.monomorphize_ms, t_mono);
                if std::env::var_os("RAYZOR_DUMP_MIR").is_some() {
                    eprintln!("=== MIR DUMP (post-merge) for {} ===", ast_file.filename);
                    eprintln!("{}", crate::ir::dump::dump_module(&module));
                    eprintln!("=== END MIR DUMP ===");
                }
                self.mir_modules.push(std::sync::Arc::new(module));
                self.loaded_stdlib_typed_files
                    .retain(|file| source_file_identity(&file.metadata.file_path) != identity);
            }
            return Ok(cached);
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
                        imports.extend(Self::import_dependencies(import));
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

        self.run_macro_hooks()?;
        self.append_entry_point_run();

        Ok(all_typed_files)
    }

    /// Haxe ends the program's `main` with `haxe.EntryPoint.run()` when
    /// EntryPoint is part of the program: the main event loop runs out before
    /// the process ends. The call goes before every return of the entry `main`.
    fn append_entry_point_run(&mut self) {
        use crate::ir::{FunctionKind, IrInstruction, IrTerminator};
        let Some(module) = self.mir_modules.last_mut() else {
            return;
        };
        let find = |m: &crate::ir::IrModule, test: &dyn Fn(&crate::ir::IrFunction) -> bool| {
            m.functions
                .values()
                .find(|f| !f.cfg.blocks.is_empty() && test(f))
                .map(|f| f.id)
        };
        let Some(run) = find(module, &|f| {
            f.qualified_name.as_deref() == Some("haxe.EntryPoint.run")
        }) else {
            return;
        };
        let Some(main) = module
            .entry_function()
            .filter(|f| f.kind == FunctionKind::UserDefined)
            .map(|f| f.id)
        else {
            return;
        };
        let module = std::sync::Arc::make_mut(module);
        let Some(main) = module.functions.get_mut(&main) else {
            return;
        };
        for block in main.cfg.blocks.values_mut() {
            if matches!(block.terminator, IrTerminator::Return { .. }) {
                block.instructions.push(IrInstruction::CallDirect {
                    dest: None,
                    func_id: run,
                    args: Vec::new(),
                    arg_ownership: Vec::new(),
                    type_args: Vec::new(),
                    is_tail_call: false,
                });
            }
        }
    }

    /// Runs the macros' `onAfterTyping`/`onGenerate` callbacks once every
    /// module is typed, reporting what they say as macro diagnostics.
    fn run_macro_hooks(&mut self) -> Result<(), Vec<CompilationError>> {
        if self.macro_hooks.is_empty() {
            return Ok(());
        }
        let hooks = std::mem::take(&mut self.macro_hooks);
        let mut files = self.user_files.clone();
        files.extend(self.loaded_import_haxe_files.iter().cloned());
        let to_error = |message: String, location| CompilationError {
            message: format!("[E0700] {}", message),
            location,
            category: ErrorCategory::MacroExpansionError,
            suggestion: None,
            related_errors: Vec::new(),
        };
        let diagnostics = crate::macro_system::expander::run_generation_hooks(
            &hooks,
            &files,
            self.macro_statics.clone(),
        )
        .map_err(|e| vec![to_error(format!("macro hook failed: {}", e), e.location())])?;
        let mut errors = Vec::new();
        for diag in diagnostics {
            match diag.severity {
                crate::macro_system::MacroSeverity::Error => {
                    errors.push(to_error(diag.message, diag.location));
                }
                crate::macro_system::MacroSeverity::Warning => {
                    eprintln!("Warning: [MACRO] {}", diag.message);
                }
                _ => {}
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            self.print_compilation_errors(&errors);
            Err(errors)
        }
    }
}

/// The modules `@:build` / `@:autoBuild` calls in `file` name: for
/// `pkg.Mod.f()`, `pkg.Mod` and, for a sub-type, `pkg.Mod.Sub`'s module.
fn build_macro_module_paths(file: &HaxeFile) -> Vec<String> {
    fn dotted(e: &parser::Expr) -> Option<Vec<String>> {
        match &e.kind {
            parser::ExprKind::Ident(name) => Some(vec![name.clone()]),
            parser::ExprKind::Field { expr, field, .. } => {
                let mut path = dotted(expr)?;
                path.push(field.clone());
                Some(path)
            }
            _ => None,
        }
    }
    let metas = file.declarations.iter().flat_map(|decl| match decl {
        parser::TypeDeclaration::Class(c) => c.meta.iter().collect::<Vec<_>>(),
        parser::TypeDeclaration::Interface(i) => i.meta.iter().collect(),
        parser::TypeDeclaration::Abstract(a) => a.meta.iter().collect(),
        parser::TypeDeclaration::Enum(e) => e.meta.iter().collect(),
        parser::TypeDeclaration::Typedef(t) => t.meta.iter().collect(),
        _ => Vec::new(),
    });
    let mut paths = Vec::new();
    for meta in metas {
        if !matches!(meta.name.trim_start_matches(':'), "build" | "autoBuild") {
            continue;
        }
        let Some(parser::ExprKind::Call { expr, .. }) = meta.params.first().map(|p| &p.kind) else {
            continue;
        };
        let Some(mut path) = dotted(expr) else {
            continue;
        };
        path.pop();
        while path.len() > 1 {
            paths.push(path.join("."));
            path.pop();
        }
    }
    paths.sort();
    paths.dedup();
    paths
}
