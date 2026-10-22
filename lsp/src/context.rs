//! Compilation context for the LSP server.
//!
//! Maintains open file state and provides compilation for diagnostics,
//! hover, goto-definition, and completions.

use crate::analysis::{self, CompletionEntry, FileSymbolIndex};
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::tast::SymbolId;
use std::collections::HashMap;
use std::path::PathBuf;

/// Persistent compilation context shared across LSP requests.
pub struct LspContext {
    /// Root workspace directory.
    pub root: PathBuf,
    /// Open file contents (URI → source text).
    pub open_files: HashMap<String, String>,
    /// Where imports are searched: the project's class paths, its
    /// libraries', then the workspace root.
    pub class_paths: Vec<PathBuf>,
    /// The project's defines, seen by its own files only.
    pub defines: Vec<String>,
    pub initialization_macros: Vec<String>,
    /// The build file the configuration came from, if any.
    pub hxml: Option<PathBuf>,
    /// What could not be applied from the configuration.
    pub config_problems: Vec<String>,
    /// How long the macros of one compile may run in all.
    pub macro_budget: std::time::Duration,
    /// The last analysis of each open document that produced an index.
    analyses: HashMap<String, Analysis>,
    diagnostics: HashMap<String, Vec<LspDiagnostic>>,
    workspace_index: crate::workspace_index::WorkspaceIndex,
}

/// A compiled document: the unit its symbols live in and its position index.
pub struct Analysis {
    pub unit: CompilationUnit,
    pub index: FileSymbolIndex,
}

impl Analysis {
    /// Whether `symbol` is called `name`: an index of earlier text can hold
    /// another name at a position that has since been edited.
    fn names(&self, symbol: SymbolId, name: &str) -> bool {
        self.unit
            .symbol_table
            .get_symbol(symbol)
            .and_then(|s| self.unit.string_interner.get(s.name))
            == Some(name)
    }
}

impl LspContext {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            open_files: HashMap::new(),
            class_paths: Vec::new(),
            defines: Vec::new(),
            initialization_macros: Vec::new(),
            hxml: None,
            config_problems: Vec::new(),
            macro_budget: std::time::Duration::from_secs(5),
            analyses: HashMap::new(),
            diagnostics: HashMap::new(),
            workspace_index: Default::default(),
        }
    }

    /// Explicit build settings take precedence over discovered library metadata.
    pub fn load_workspace_config(&mut self, hxml: Option<&str>) {
        self.class_paths.clear();
        self.defines.clear();
        self.initialization_macros.clear();
        self.config_problems.clear();
        self.hxml = hxml
            .map(|h| self.root.join(h))
            .or_else(|| crate::project::find_hxml(&self.root, &mut self.config_problems));
        let mut requested = match self.hxml.clone() {
            Some(path) => self.apply_hxml(&path),
            None => {
                self.class_paths = self.manifest_class_paths();
                let extra = self.root.join("extraParams.hxml");
                if self.root.join("haxelib.json").is_file() && extra.is_file() {
                    self.apply_hxml(&extra)
                } else {
                    Vec::new()
                }
            }
        };
        let mut metadata = crate::project::library_metadata(
            &self.root,
            &self.class_paths,
            &mut self.config_problems,
        );
        // A direct -lib or path dependency requests hooks even if a class path
        // also names its sources. Other inferred dependencies use sources as-is.
        metadata.provided.retain(|name| {
            !requested.iter().any(|lib| &lib.name == name)
                && !metadata
                    .libraries
                    .iter()
                    .any(|lib| &lib.name == name && lib.is_local())
        });
        self.class_paths.extend(metadata.class_paths);
        // A manifest can fill omissions, but must not replace an explicit
        // version selected in the build file.
        for dependency in metadata.libraries {
            if let Some(existing) = requested.iter_mut().find(|lib| lib.name == dependency.name) {
                if existing.is_unpinned() && dependency.is_local() {
                    *existing = dependency;
                }
            } else {
                requested.push(dependency);
            }
        }
        let (libraries, problems) =
            compiler::haxelib::resolve_dependencies(&requested, &self.root, &metadata.provided);
        self.config_problems.extend(problems);
        for lib in libraries {
            self.class_paths.extend(lib.class_paths);
            self.initialization_macros.extend(lib.macros);
            self.defines
                .extend(lib.defines.into_iter().map(|(name, value)| {
                    value.map_or_else(|| name.clone(), |value| format!("{name}={value}"))
                }));
        }
        if !self.class_paths.contains(&self.root) {
            self.class_paths.push(self.root.clone());
        }
        let mut seen = std::collections::HashSet::new();
        self.class_paths
            .retain(|p| seen.insert(p.canonicalize().unwrap_or_else(|_| p.clone())));
        self.workspace_index = crate::workspace_index::WorkspaceIndex::start(
            self.class_paths.clone(),
            self.defines.clone(),
        );
    }

    /// Use one build; merging `--next` builds would mix incompatible targets.
    fn apply_hxml(&mut self, path: &std::path::Path) -> Vec<compiler::haxelib::Dependency> {
        use compiler::hxml::HxmlConfig;
        let build = match HxmlConfig::builds_from_file(path) {
            Ok(builds) => builds.into_iter().next().unwrap_or_default(),
            Err(e) => {
                self.config_problems.push(e);
                return Vec::new();
            }
        };
        for (file, line, message) in &build.warnings {
            self.config_problems
                .push(format!("{}:{}: {}", file.display(), line, message));
        }
        // Haxe gives the last declared class path precedence.
        self.class_paths
            .extend(build.class_paths.iter().rev().cloned());
        self.initialization_macros = build.macros.clone();
        self.defines = build
            .defines
            .iter()
            .map(|(name, value)| {
                value
                    .as_ref()
                    .map_or_else(|| name.clone(), |value| format!("{name}={value}"))
            })
            .chain(build.target_defines().into_iter().map(str::to_string))
            .collect();
        build
            .libraries
            .iter()
            .map(|spec| {
                compiler::haxelib::Dependency::parse(spec, path.parent().unwrap_or(&self.root))
            })
            .collect()
    }

    fn manifest_class_paths(&self) -> Vec<PathBuf> {
        let Ok(content) = std::fs::read_to_string(self.root.join("rayzor.toml")) else {
            return Vec::new();
        };
        let Ok(manifest) = compiler::workspace::manifest::parse_manifest(&content) else {
            return Vec::new();
        };
        match manifest {
            compiler::workspace::manifest::RayzorManifest::SingleProject(project) => project
                .build
                .map(|build| {
                    build
                        .class_paths
                        .iter()
                        .map(|cp| self.root.join(cp))
                        .collect()
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Compile a file and return diagnostics. Also builds the symbol index.
    pub fn compile_file(&mut self, uri: &str, source: &str) -> Vec<LspDiagnostic> {
        let filename = uri_to_path(uri);
        self.workspace_index
            .update(std::path::Path::new(&filename), Some(source));

        let mut config = CompilationConfig {
            load_stdlib: true,
            emit_safety_warnings: false,
            record_references: true,
            user_defines: self.defines.clone(),
            continue_after_macro_errors: true,
            typecheck_only: true,
            ..Default::default()
        };
        config.stdlib_paths = match crate::stdlib::discover() {
            Ok(paths) => paths,
            Err(message) => return vec![LspDiagnostic::error(&filename, 0, 0, message)],
        };
        // Cached Rayzor library symbols do not describe a system Haxe library.
        config.enable_cache = false;
        config.pipeline_config = config.pipeline_config.skip_analysis();

        // A syntax error is reported as such, and the last analysis is kept:
        // compiling text the parser rejects yields a partial program and no
        // error, because a fallback parser recovers by dropping what it cannot
        // read.
        let syntax = self.syntax_errors(&filename, source);
        if !syntax.is_empty() {
            return syntax;
        }

        let mut unit = CompilationUnit::new(config);

        for cp in &self.class_paths {
            unit.add_source_path(cp.clone());
        }

        if let Err(e) = unit.load_stdlib() {
            return vec![LspDiagnostic::error(
                &filename,
                0,
                0,
                format!("Stdlib: {}", e),
            )];
        }

        let initialization_errors = unit.run_initialization_macros(&self.initialization_macros);
        if let Err(e) = unit.add_file(source, &filename) {
            return vec![LspDiagnostic::error(
                &filename,
                0,
                0,
                format!("Parse: {}", e),
            )];
        }

        let mut result = match unit.lower_to_tast() {
            Ok(_typed_files) => Vec::new(),
            Err(errors) => errors
                .iter()
                .filter(|e| e.category != compiler::pipeline::ErrorCategory::MacroExpansionError)
                .map(|e| LspDiagnostic {
                    file: unit
                        .file_path_of(e.location.file_id)
                        .unwrap_or(&filename)
                        .to_string(),
                    line: e.location.line,
                    column: e.location.column,
                    end_line: e.location.line,
                    end_column: e.location.column + 1,
                    severity: match e.category {
                        compiler::pipeline::ErrorCategory::ParseError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::TypeError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::ConcurrencyError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::MacroExpansionError => {
                            DiagSeverity::Error
                        }
                        _ => DiagSeverity::Warning,
                    },
                    message: e.message.clone(),
                    suggestion: e.suggestion.clone(),
                    related: e.related_errors.clone(),
                    related_information: Vec::new(),
                })
                .collect(),
        };
        result.extend(unit.macro_diagnostics.iter().filter_map(|diagnostic| {
            use compiler::macro_system::MacroSeverity;
            let severity = match diagnostic.severity {
                MacroSeverity::Error => DiagSeverity::Error,
                MacroSeverity::Warning => DiagSeverity::Warning,
                MacroSeverity::Info => return None,
            };
            let file = unit
                .file_path_of(diagnostic.location.file_id)
                .unwrap_or(&filename);
            let mut item = LspDiagnostic::error(
                file,
                diagnostic.location.line,
                diagnostic.location.column,
                diagnostic.message.clone(),
            );
            item.severity = severity;
            if let Some(call) = diagnostic.call_site
                && let Some(file) = unit.file_path_of(call.file_id)
                && let Ok(uri) = file_uri(file).parse()
            {
                let at = lsp_types::Position::new(
                    call.line.saturating_sub(1),
                    call.column.saturating_sub(1),
                );
                item.related_information
                    .push(lsp_types::DiagnosticRelatedInformation {
                        location: lsp_types::Location {
                            uri,
                            range: lsp_types::Range::new(at, at),
                        },
                        message: "Macro invoked here".into(),
                    });
            }
            Some(item)
        }));
        result.extend(
            initialization_errors
                .into_iter()
                .map(|message| LspDiagnostic::error(&filename, 0, 0, message)),
        );

        let file_id = unit.file_id_of(&filename).unwrap_or(u32::MAX);
        let mut preprocessor = parser::preprocessor::PreprocessorConfig::default();
        preprocessor.defines.extend(
            self.defines
                .iter()
                .map(|define| define.split('=').next().unwrap().to_string()),
        );
        preprocessor.defines.insert("eval".into());
        let text = parser::preprocessor::preprocess(source, &preprocessor);
        if let Ok(file) = parser::rd::rd_parse(&text, &filename, false, false) {
            FileSymbolIndex::record_source_declarations(&file, &text, file_id, &mut unit);
            crate::type_references::record(&file, &text, file_id, &mut unit);
        }
        crate::documentation::record(&mut unit);
        for file in unit.stdlib_files.iter().chain(&unit.user_files) {
            self.workspace_index.ensure_parsed_file(file);
        }
        let index = FileSymbolIndex::build(
            &unit.symbol_table,
            &unit.string_interner,
            &unit.symbol_references,
            file_id,
        );
        // A document mid-edit often fails to type; until it does again, the
        // last analysis still holds for the text before the edit.
        if !index.names.is_empty() || !self.analyses.contains_key(uri) {
            self.analyses
                .insert(uri.to_string(), Analysis { unit, index });
        }

        result
    }

    /// Publish errors on their own files, merging contributions from open documents.
    pub fn update_diagnostics(
        &mut self,
        uri: &str,
        diagnostics: Vec<LspDiagnostic>,
    ) -> Vec<(String, Vec<LspDiagnostic>)> {
        let mut files = std::collections::BTreeSet::from([uri_to_path(uri)]);
        files.extend(diagnostics.iter().map(|d| d.file.clone()));
        if let Some(previous) = self.diagnostics.insert(uri.to_string(), diagnostics) {
            files.extend(previous.into_iter().map(|d| d.file));
        }
        files
            .into_iter()
            .map(|file| {
                let mut found = Vec::new();
                for diagnostic in self
                    .diagnostics
                    .values()
                    .flatten()
                    .filter(|d| d.file == file)
                {
                    if !found.iter().any(|d: &LspDiagnostic| {
                        d.line == diagnostic.line
                            && d.column == diagnostic.column
                            && d.message == diagnostic.message
                    }) {
                        found.push(diagnostic.clone());
                    }
                }
                (file, found)
            })
            .collect()
    }

    /// Forget a closed document and its analysis.
    pub fn close(&mut self, uri: &str) {
        self.open_files.remove(uri);
        self.analyses.remove(uri);
        self.diagnostics.remove(uri);
        self.index_disk_file(uri);
    }

    pub fn index_disk_file(&self, uri: &str) {
        let path = PathBuf::from(uri_to_path(uri));
        let source = self
            .open_files
            .get(uri)
            .cloned()
            .or_else(|| std::fs::read_to_string(&path).ok());
        self.workspace_index.update(&path, source.as_deref());
    }

    pub fn workspace_symbols(&self, query: &str) -> Vec<serde_json::Value> {
        self.workspace_index.search(query)
    }

    fn syntax_errors(&self, filename: &str, source: &str) -> Vec<LspDiagnostic> {
        let mut config = parser::preprocessor::PreprocessorConfig::default();
        config.defines.insert("eval".to_string());
        config.defines.extend(
            self.defines
                .iter()
                .map(|define| define.split('=').next().unwrap().to_string()),
        );
        let text = parser::preprocessor::preprocess(source, &config);
        let import_file = filename.ends_with("import.hx");
        let Err(errors) = parser::rd::rd_parse(&text, filename, import_file, false) else {
            return Vec::new();
        };
        errors
            .iter()
            .map(|e| {
                let (line, column) = line_column(&text, e.span.start);
                let (end_line, end_column) = line_column(&text, e.span.end.max(e.span.start + 1));
                LspDiagnostic {
                    file: filename.to_string(),
                    line,
                    column,
                    end_line,
                    end_column,
                    severity: DiagSeverity::Error,
                    message: e.message.clone(),
                    suggestion: None,
                    related: Vec::new(),
                    related_information: Vec::new(),
                }
            })
            .collect()
    }

    fn symbol_at(&self, uri: &str, line: u32, col: u32) -> Option<(&Analysis, SymbolId)> {
        let a = self.analyses.get(uri)?;
        let current = self.open_files.get(uri)?;
        let original = a.unit.source_of(a.index.file_id)?;
        if current != original {
            let (&(row, start), &(len, _)) = a.index.names.range(..=(line, col)).next_back()?;
            let old = original.lines().nth(row.checked_sub(1)? as usize)?;
            let new = current.lines().nth(row.checked_sub(1)? as usize)?;
            let begins = start.saturating_sub(1) as usize;
            let is_name = |unit: Option<u16>| {
                unit.and_then(|c| char::from_u32(c as u32))
                    .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '$')
            };
            for boundary in [begins.checked_sub(1), Some(begins + len as usize)]
                .into_iter()
                .flatten()
            {
                if is_name(new.encode_utf16().nth(boundary))
                    && !is_name(old.encode_utf16().nth(boundary))
                {
                    return None;
                }
            }
            if !old
                .encode_utf16()
                .skip(start.saturating_sub(1) as usize)
                .take(len as usize)
                .eq(new
                    .encode_utf16()
                    .skip(start.saturating_sub(1) as usize)
                    .take(len as usize))
            {
                return None;
            }
        }
        Some((a, a.index.find_symbol_at(line, col)?))
    }

    /// Look up type/doc info for a symbol at a given position.
    pub fn hover_info(&self, uri: &str, line: u32, col: u32) -> Option<String> {
        let (a, sym_id) = self.symbol_at(uri, line, col)?;
        let unit = &a.unit;
        let mut hover = analysis::format_hover(
            sym_id,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
        )?;
        let symbol = unit.symbol_table.get_symbol(sym_id)?;
        let qualified = symbol.qualified_name.unwrap_or_else(|| {
            unit.symbol_table
                .build_qualified_name(sym_id, &unit.scope_tree, &unit.string_interner)
        });
        if let Some(declaration) = unit
            .string_interner
            .get(qualified)
            .and_then(|name| self.workspace_index.declaration(name))
            .or_else(|| {
                let text = self.open_files.get(uri)?;
                let (rline, rcol, _) = analysis::receiver_before(text, line, col)?;
                let receiver = a.index.find_symbol_at(rline, rcol)?;
                let receiver_type = unit.symbol_table.get_symbol(receiver)?.type_id;
                let owner = analysis::class_of_type(
                    receiver_type,
                    &unit.symbol_table,
                    &unit.type_table.borrow(),
                    &unit.string_interner,
                )?;
                let path = unit
                    .symbol_table
                    .display_type_path(owner, &unit.string_interner)?;
                if let Some(source) = unit
                    .namespace_resolver
                    .resolve_qualified_path_to_file_force(&path)
                {
                    self.workspace_index.ensure_disk_file(&source);
                }
                let member = unit.string_interner.get(symbol.name)?;
                self.workspace_index
                    .declaration(&format!("{path}.{member}"))
            })
        {
            if let Some(old) = symbol
                .documentation
                .and_then(|doc| unit.string_interner.get(doc))
                && hover.ends_with(old)
            {
                hover.truncate(hover.len() - old.len());
                hover = hover.trim_end().to_string();
            }
            if let Some(doc) = declaration.documentation {
                hover.push_str("\n\n");
                hover.push_str(&doc);
            }
            if symbol.kind == compiler::tast::symbols::SymbolKind::Function
                && hover.contains("*Signature not yet resolved.*")
                && declaration.signature.contains("function ")
                && let Some(end) = hover.find("\n```")
            {
                hover.replace_range("```haxe\n".len()..end, &declaration.signature);
                hover = hover.replace("\n\n*Signature not yet resolved.*", "");
            }
        }
        Some(hover)
    }

    /// Find definition location for a symbol at a given position.
    pub fn goto_definition(&self, uri: &str, line: u32, col: u32) -> Option<(String, u32, u32)> {
        let (a, sym_id) = self.symbol_at(uri, line, col)?;
        let loc = a.unit.symbol_table.get_symbol(sym_id)?.definition_location;
        if !loc.is_valid() {
            return None;
        }
        let file = a.unit.file_path_of(loc.file_id)?.to_string();
        Some((file, loc.line, loc.column))
    }

    /// Every place the symbol at a position is named, across the compiled
    /// files: (path, line, column, length), 1-based.
    pub fn references(
        &self,
        uri: &str,
        line: u32,
        col: u32,
        include_declaration: bool,
    ) -> Vec<(String, u32, u32, u32)> {
        let Some((a, sym_id)) = self.symbol_at(uri, line, col) else {
            return Vec::new();
        };
        let unit = &a.unit;
        let mut out = Vec::new();
        let place = |at: compiler::tast::SourceLocation, len: u32| {
            unit.file_path_of(at.file_id)
                .map(|path| (path.to_string(), at.line, at.column, len))
        };
        if include_declaration
            && let Some(sym) = unit.symbol_table.get_symbol(sym_id)
            && sym.definition_location.is_valid()
        {
            let len = unit
                .string_interner
                .get(sym.name)
                .map_or(1, |n| n.len() as u32);
            out.extend(place(sym.definition_location, len));
        }
        for r in unit.symbol_references.iter().filter(|r| r.symbol == sym_id) {
            out.extend(place(r.at, r.len));
        }
        out
    }

    /// Completions at (line, col), 1-based: the members of the receiver
    /// when the cursor follows `receiver.`, otherwise every visible name.
    pub fn completions(&self, uri: &str, line: u32, col: u32) -> Completions {
        let Some(a) = self.analyses.get(uri) else {
            return Completions::Names(Vec::new());
        };
        let unit = &a.unit;
        if let Some(text) = self.open_files.get(uri)
            && let Some((rline, rcol, name)) = analysis::receiver_before(text, line, col)
        {
            let members = a
                .index
                .find_symbol_at(rline, rcol)
                .filter(|&s| a.names(s, &name))
                .or_else(|| {
                    let (st, interner) = (&unit.symbol_table, &unit.string_interner);
                    analysis::resolve_name(&name, a.index.file_id, rline, rcol, st, interner)
                })
                .map(|receiver| {
                    let members = analysis::member_completions(
                        receiver,
                        &unit.symbol_table,
                        &unit.type_table,
                        &unit.string_interner,
                        |class| {
                            unit.class_fields(class)
                                .iter()
                                .map(|f| (f.1, f.2))
                                .collect()
                        },
                    );
                    if members.is_empty() {
                        analysis::stdlib_member_completions(receiver, unit)
                    } else {
                        members
                    }
                })
                .unwrap_or_default();
            return Completions::Members(members);
        }
        Completions::Names(analysis::collect_completions(
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            a.index.file_id,
        ))
    }

    /// Build semantic tokens for syntax highlighting.
    pub fn semantic_tokens(&self, uri: &str) -> Option<Vec<analysis::SemanticToken>> {
        let a = self.analyses.get(uri)?;
        Some(analysis::build_semantic_tokens(
            &a.unit.symbol_table,
            &a.index,
        ))
    }

    /// Build document symbols for the outline panel.
    pub fn document_symbols(&self, uri: &str) -> Option<Vec<analysis::DocumentSymbolEntry>> {
        let a = self.analyses.get(uri)?;
        Some(analysis::build_document_symbols(
            &a.unit.symbol_table,
            &a.unit.type_table,
            &a.unit.string_interner,
            a.index.file_id,
        ))
    }

    /// The signature of the call whose argument list holds (line, col),
    /// 1-based, with the argument the cursor is in.
    pub fn signature_help(
        &self,
        uri: &str,
        line: u32,
        col: u32,
    ) -> Option<analysis::SignatureInfo> {
        let text = self.open_files.get(uri)?;
        let call = analysis::enclosing_call(text, line, col)?;
        let a = self.analyses.get(uri)?;
        let (st, interner) = (&a.unit.symbol_table, &a.unit.string_interner);
        let callee = a
            .index
            .find_symbol_at(call.callee_line, call.callee_col)
            .filter(|&s| a.names(s, &call.callee))
            .or_else(|| {
                let (l, c) = (call.callee_line, call.callee_col);
                analysis::resolve_name(&call.callee, a.index.file_id, l, c, st, interner)
            })?;
        analysis::build_signature_help(
            callee,
            &a.unit.symbol_table,
            &a.unit.type_table,
            &a.unit.string_interner,
            call.argument,
        )
    }

    pub fn expand_macro(
        &self,
        uri: &str,
        position: lsp_types::Position,
    ) -> Option<serde_json::Value> {
        let analysis = self.analyses.get(uri)?;
        let source = self.open_files.get(uri)?;
        if !analysis.unit.source_matches(analysis.index.file_id, source) {
            return None;
        }
        let source = analysis.unit.source_of(analysis.index.file_id)?;
        let line_start = source
            .split_inclusive('\n')
            .take(position.line as usize)
            .map(str::len)
            .sum::<usize>();
        let line = source.get(line_start..)?.split('\n').next()?;
        let mut utf16 = 0;
        let column = line
            .char_indices()
            .find_map(|(offset, c)| {
                if utf16 >= position.character {
                    Some(offset)
                } else {
                    utf16 += c.len_utf16() as u32;
                    None
                }
            })
            .unwrap_or(line.len());
        let offset = line_start + column;
        let origin = analysis
            .unit
            .macro_expansions
            .iter()
            .filter(|o| {
                o.call_site.file_id == analysis.index.file_id
                    && o.call_span.start <= offset
                    && offset < o.call_span.end
            })
            .min_by_key(|o| o.call_span.end.saturating_sub(o.call_span.start))?;
        let pos = |at| {
            let (line, character) = line_column(source, at);
            serde_json::json!({"line":line-1,"character":character-1})
        };
        Some(serde_json::json!({
            "name": origin.macro_name,
            "expansion": origin.expanded_text,
            "range": {"start":pos(origin.call_span.start),"end":pos(origin.call_span.end)},
            "definition": origin.definition_file.as_ref().zip(origin.definition_site).and_then(|(file, at)| {
                let (line, character) = if at.line > 0 && at.column > 0 { (at.line, at.column) } else {
                    let text = std::fs::read_to_string(file).ok()?;
                    line_column(&text, at.byte_offset as usize)
                };
                Some(serde_json::json!({"uri":file_uri(file),"range":{"start":{"line":line-1,"character":character-1},"end":{"line":line-1,"character":character}}}))
            })
        }))
    }

    /// The macro expansions in the document, for inlay hints.
    pub fn macro_expansion_hints(&self, uri: &str) -> Vec<MacroExpansionHint> {
        let Some(a) = self.analyses.get(uri) else {
            return Vec::new();
        };
        if !self
            .open_files
            .get(uri)
            .is_some_and(|source| a.unit.source_matches(a.index.file_id, source))
        {
            return Vec::new();
        }
        a.unit
            .macro_expansions
            .iter()
            .filter_map(|origin| {
                let text = origin.expanded_text.as_ref()?;
                let loc = &origin.call_site;
                if !loc.is_valid() || loc.file_id != a.index.file_id {
                    return None;
                }
                let source = a.unit.source_of(a.index.file_id)?;
                let (line, column) = line_column(source, origin.call_span.end);
                Some(MacroExpansionHint {
                    line,
                    column,
                    macro_name: origin.macro_name.clone(),
                    expanded_text: text.clone(),
                })
            })
            .collect()
    }
}

/// What completion offers at a position.
pub enum Completions {
    /// After `receiver.`: only what the receiver has.
    Members(Vec<CompletionEntry>),
    /// Anywhere else: every name in reach, with keywords.
    Names(Vec<CompletionEntry>),
}

/// A macro expansion hint for display as an inlay hint.
pub struct MacroExpansionHint {
    pub line: u32,
    pub column: u32,
    pub macro_name: String,
    pub expanded_text: String,
}

/// A diagnostic ready for LSP conversion.
#[allow(dead_code)]
#[derive(Clone)]
pub struct LspDiagnostic {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub severity: DiagSeverity,
    pub message: String,
    pub suggestion: Option<String>,
    pub related: Vec<String>,
    pub related_information: Vec<lsp_types::DiagnosticRelatedInformation>,
}

impl LspDiagnostic {
    pub fn error(file: &str, line: u32, col: u32, message: String) -> Self {
        Self {
            file: file.to_string(),
            line,
            column: col,
            end_line: line,
            end_column: col + 1,
            severity: DiagSeverity::Error,
            message,
            suggestion: None,
            related: Vec::new(),
            related_information: Vec::new(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone)]
pub enum DiagSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

/// The 1-based (line, column) of byte `offset` in `text`, clamped to its end.
fn line_column(text: &str, offset: usize) -> (u32, u32) {
    let before = &text.as_bytes()[..offset.min(text.len())];
    let line = before.iter().filter(|&&b| b == b'\n').count() as u32 + 1;
    let start = before
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |p| p + 1);
    let column = String::from_utf8_lossy(&before[start..])
        .encode_utf16()
        .count() as u32
        + 1;
    (line, column)
}

pub(crate) fn uri_to_path(uri: &str) -> String {
    if let Some(path) = uri.strip_prefix("file://") {
        path.replace("%20", " ")
    } else {
        uri.to_string()
    }
}

pub(crate) fn file_uri(path: &str) -> String {
    format!(
        "file://{}",
        path.replace('%', "%25")
            .replace(' ', "%20")
            .replace('#', "%23")
            .replace('?', "%3F")
    )
}
