//! Rendering what went wrong: compile errors, MIR diagnostics, source maps.

use super::*;

impl CompilationUnit {

    /// When `RAYZOR_DUMP_JIT_MAP=1`, write `/tmp/rayzor_file_table.csv`
    /// containing `file_id,path` rows for every file seen by the compiler.
    /// Pairs with `/tmp/rayzor_jit_symbols.csv` (whose `file_id` column
    /// references rows in this table) so off-line resolvers can map a
    /// PC → `IrFunction` → `Haxe file:line`.
    pub fn maybe_dump_file_table(&self) {
        if std::env::var_os("RAYZOR_DUMP_JIT_MAP").as_deref() != Some(std::ffi::OsStr::new("1")) {
            return;
        }
        let path = "/tmp/rayzor_file_table.csv";
        let mut out = String::from("file_id,path\n");
        let mut rows: Vec<(&String, u32)> = self
            .file_id_by_filename
            .iter()
            .map(|(name, id)| (name, *id))
            .collect();
        rows.sort_by_key(|r| r.1);
        for (name, id) in rows {
            let safe = name.replace('"', "''");
            out.push_str(&format!("{},\"{}\"\n", id, safe));
        }
        let n_files = self.file_id_by_filename.len();
        match std::fs::write(path, out) {
            Ok(_) => eprintln!("[jit-map] wrote {} file paths to {}", n_files, path),
            Err(e) => eprintln!("[jit-map] failed to write {}: {}", path, e),
        }
    }


    /// Extract the type name from an unresolved type error message
    pub(crate) fn extract_type_name_from_error(&self, message: &str) -> Option<String> {
        // Try to extract type name from error message formats:
        // "UnresolvedType { type_name: \"haxe.iterators.ArrayIterator\", ... }"
        // "Unresolved type: haxe.iterators.ArrayIterator"
        let type_name = if let Some(start) = message.find("type_name: \"") {
            let start = start + "type_name: \"".len();
            if let Some(end) = message[start..].find('"') {
                Some(message[start..start + end].to_string())
            } else {
                None
            }
        } else if let Some(start) = message.find("Unresolved type: ") {
            let start = start + "Unresolved type: ".len();
            let end = message[start..]
                .find(|c: char| !c.is_alphanumeric() && c != '.')
                .unwrap_or(message.len() - start);
            Some(message[start..start + end].to_string())
        } else if let Some(start) = message.find("Cannot find type '") {
            let start = start + "Cannot find type '".len();
            if let Some(end) = message[start..].find('\'') {
                Some(message[start..start + end].to_string())
            } else {
                None
            }
        } else {
            None
        };

        // Filter out generic type parameters and built-in typedefs:
        // - Single uppercase letters (T, K, V, E, R, etc.)
        // - Short names like "TKey", "TValue", etc.
        // - Built-in typedefs from StdTypes.hx (Iterator, KeyValueIterator, etc.)
        // These should NOT be treated as importable types
        if let Some(ref name) = type_name {
            // Skip single uppercase letter type parameters
            if name.len() == 1
                && name
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
            {
                return None;
            }
            // Skip common generic type parameter patterns
            if name == "Key" || name == "Value" || name == "Item" || name == "Element" {
                return None;
            }
            // Skip built-in top-level types from StdTypes.hx (these are already loaded).
            if Self::is_stdtypes_prelude_type_name(name) {
                debug!("  Filtering out StdTypes prelude type: {}", name);
                return None;
            }
        }

        type_name
    }


    /// Print compilation errors with formatted diagnostics to stderr.
    /// Uses the diagnostics crate's ErrorFormatter for consistent formatting.
    pub fn print_compilation_errors(&self, errors: &[CompilationError]) {
        use diagnostics::ErrorFormatter;

        // Build the same full source_map the MIR-diagnostic path uses,
        // so hard-error spans on cross-file types resolve to the right
        // file. See `build_full_source_map` for the inclusion order.
        let source_map = self.build_full_source_map();

        let formatter = ErrorFormatter::with_colors();

        for error in errors {
            let diagnostic = error.to_diagnostic(&source_map);
            let formatted = formatter.format_diagnostic(&diagnostic, &source_map);
            eprint!("{}", formatted);
        }
    }


    /// Print diagnostics from MIR lowering using the diagnostics formatter.
    /// The source map is built with the user file at FileId 0 to match the
    /// compiler's SourceLocation.file_id convention (user file = 0).
    pub(crate) fn print_mir_diagnostics(&mut self, mir_diagnostics: &[diagnostics::Diagnostic]) {
        use diagnostics::ErrorFormatter;

        // Store for cache replay
        self.collected_diagnostics
            .extend_from_slice(mir_diagnostics);

        // Build the full source_map covering user files + stdlib + imports +
        // on-demand-compiled package files. Cross-file diagnostics (E0382 on
        // a moved Tensor binding inside nue.sampling.LocalTempSampler, W0014
        // on a cross-context iface return inside nue.transformer.*, etc.)
        // carry the file_id of the ACTUAL source file. Without the imported
        // files registered here, the formatter falls back to whichever
        // file happens to occupy that file_id slot in the under-populated
        // map — typically the entry Main.hx — and uses the diagnostic's
        // line number as an offset INTO Main.hx, so the warning ends up
        // cited against an unrelated comment or statement.
        let source_map = self.build_full_source_map();

        let formatter = ErrorFormatter::with_colors();
        for diagnostic in mir_diagnostics {
            if diagnostic.code.as_deref() == Some("W0014")
                && std::env::var_os("RAYZOR_SHOW_W0014").is_none()
                && std::env::var_os("RAYZOR_SHOW_ADVICE").is_none()
            {
                continue;
            }
            let formatted = formatter.format_diagnostic(diagnostic, &source_map);
            eprint!("{}", formatted);
        }
    }


    /// Construct a SourceMap that registers every Haxe file the
    /// current compilation knows about, **at the same file_id** the
    /// lowering pipeline assigned to it via `file_id_by_filename`.
    ///
    /// SourceMap::add_file auto-assigns FileIds sequentially from 0,
    /// so the order we insert here determines the file_ids in the
    /// rendered output. We sort the (filename, file_id) pairs by
    /// file_id and insert in that order — if there are no gaps, the
    /// resulting source_map's FileIds align with the diagnostic
    /// spans' file_ids. Where the lowering pipeline emitted a file_id
    /// without populating file_id_by_filename (parser cache, generated
    /// modules, etc.), we still see file_id=0 fallthrough for those
    /// specific spans, but every cross-file user import resolves
    /// correctly.
    pub(crate) fn build_full_source_map(&self) -> diagnostics::SourceMap {
        let mut source_map = diagnostics::SourceMap::new();

        // Collect every file source we know about, keyed by filename.
        // file_source_by_filename is AUTHORITATIVE — it holds the exact
        // bytes the span_converter saw at lowering time, so byte_offset
        // → line/column resolution at the renderer matches what was
        // stored on diagnostic spans. The other lists are FALLBACKS
        // for files that bypassed `compile_ast_with_shared_state`
        // (e.g. parser-cache hits, stdlib pre-loads); their input
        // fields are typically the same bytes anyway.
        let mut sources: BTreeMap<String, String> = BTreeMap::new();
        for user_file in &self.user_files {
            if let Some(ref source) = user_file.input {
                sources.insert(user_file.filename.clone(), source.clone());
            }
        }
        for stdlib_file in &self.stdlib_files {
            if let Some(ref source) = stdlib_file.input {
                sources.insert(stdlib_file.filename.clone(), source.clone());
            }
        }
        for import_file in &self.import_hx_files {
            if let Some(ref source) = import_file.input {
                sources.insert(import_file.filename.clone(), source.clone());
            }
        }
        for import_haxe_file in &self.loaded_import_haxe_files {
            if let Some(ref source) = import_haxe_file.input {
                sources.insert(import_haxe_file.filename.clone(), source.clone());
            }
        }
        // Authoritative source: bytes captured at lowering entry. These
        // OVERWRITE any input-field copy so the renderer ALWAYS sees
        // the exact bytes the byte_offset was computed against.
        for (filename, source) in &self.file_source_by_filename {
            sources.insert(filename.clone(), source.clone());
        }
        for (filename, _) in &self.compiled_files {
            if !sources.contains_key(filename) {
                if let Ok(source) = std::fs::read_to_string(filename) {
                    sources.insert(filename.clone(), source);
                }
            }
        }

        // Sort file_id_by_filename by file_id ascending so the
        // sequential add_file calls land each file at the expected
        // FileId. Fall back to reading from disk if we don't have an
        // in-memory copy.
        let mut id_pairs: Vec<(&String, u32)> = self
            .file_id_by_filename
            .iter()
            .map(|(name, id)| (name, *id))
            .collect();
        id_pairs.sort_by_key(|(_, id)| *id);

        let mut next_expected: u32 = 0;
        for (filename, file_id) in id_pairs {
            // Fill any gaps with empty placeholder entries so subsequent
            // files land at their correct FileId. Gaps can occur when a
            // file_id was reserved but the file failed to compile.
            while next_expected < file_id {
                source_map.add_file(format!("<unknown:{}>", next_expected), String::new());
                next_expected += 1;
            }
            let content = sources
                .get(filename)
                .cloned()
                .or_else(|| std::fs::read_to_string(filename).ok())
                .unwrap_or_default();
            source_map.add_file(filename.clone(), content);
            next_expected += 1;
        }

        source_map
    }
}
