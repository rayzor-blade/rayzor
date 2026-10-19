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
    /// The build file the configuration came from, if any.
    pub hxml: Option<PathBuf>,
    /// What could not be applied from the configuration.
    pub config_problems: Vec<String>,
    /// How long the macros of one compile may run in all.
    pub macro_budget: std::time::Duration,
    /// The last analysis of each open document that produced an index.
    analyses: HashMap<String, Analysis>,
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
            hxml: None,
            config_problems: Vec::new(),
            macro_budget: std::time::Duration::from_secs(5),
            analyses: HashMap::new(),
        }
    }

    /// Configure the project from its build file: `hxml` when the client
    /// names one, else `build.hxml`, else the first `.hxml` in the root, else
    /// `rayzor.toml`. The workspace root comes last, as haxe always searches
    /// the working directory.
    pub fn load_workspace_config(&mut self, hxml: Option<&str>) {
        self.class_paths.clear();
        self.defines.clear();
        self.config_problems.clear();
        self.hxml = hxml.map(|h| self.root.join(h)).or_else(|| self.find_hxml());
        match self.hxml.clone() {
            Some(path) => self.apply_hxml(&path),
            None => self.class_paths = self.manifest_class_paths(),
        }
        if !self.class_paths.contains(&self.root) {
            self.class_paths.push(self.root.clone());
        }
    }

    fn find_hxml(&self) -> Option<PathBuf> {
        let build = self.root.join("build.hxml");
        if build.is_file() {
            return Some(build);
        }
        let mut found: Vec<PathBuf> = std::fs::read_dir(&self.root)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "hxml"))
            .collect();
        found.sort();
        found.into_iter().next()
    }

    /// The first build the file describes; a `--next` build only differs in
    /// output or target, and one view of the sources is what an editor shows.
    fn apply_hxml(&mut self, path: &std::path::Path) {
        use compiler::hxml::HxmlConfig;
        let build = match HxmlConfig::builds_from_file(path) {
            Ok(builds) => builds.into_iter().next().unwrap_or_default(),
            Err(e) => {
                self.config_problems.push(e);
                return;
            }
        };
        for (file, line, message) in &build.warnings {
            self.config_problems
                .push(format!("{}:{}: {}", file.display(), line, message));
        }
        self.class_paths = build.class_paths.clone();
        self.defines = build
            .defines
            .iter()
            .map(|(name, _)| name.clone())
            .chain(build.target_defines().into_iter().map(str::to_string))
            .collect();
        let (libraries, problems) = compiler::haxelib::resolve(&build.libraries, &self.root);
        self.config_problems.extend(problems);
        for lib in libraries {
            self.class_paths.extend(lib.class_paths);
            self.defines
                .extend(lib.defines.into_iter().map(|(name, _)| name));
        }
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

        let mut config = CompilationConfig {
            load_stdlib: true,
            emit_safety_warnings: false,
            record_references: true,
            user_defines: self.defines.clone(),
            continue_after_macro_errors: true,
            typecheck_only: true,
            ..Default::default()
        };
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

        if let Err(e) = unit.add_file(source, &filename) {
            return vec![LspDiagnostic::error(
                &filename,
                0,
                0,
                format!("Parse: {}", e),
            )];
        }

        let result = match unit.lower_to_tast() {
            Ok(_typed_files) => Vec::new(),
            Err(errors) => errors
                .iter()
                .map(|e| LspDiagnostic {
                    file: filename.clone(),
                    line: e.location.line,
                    column: e.location.column,
                    end_line: e.location.line,
                    end_column: e.location.column + 1,
                    severity: match e.category {
                        compiler::pipeline::ErrorCategory::ParseError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::TypeError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::ConcurrencyError => DiagSeverity::Error,
                        _ => DiagSeverity::Warning,
                    },
                    message: e.message.clone(),
                    suggestion: e.suggestion.clone(),
                    related: e.related_errors.clone(),
                })
                .collect(),
        };

        let file_id = unit.file_id_of(&filename).unwrap_or(u32::MAX);
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

    /// Forget a closed document and its analysis.
    pub fn close(&mut self, uri: &str) {
        self.open_files.remove(uri);
        self.analyses.remove(uri);
    }

    fn syntax_errors(&self, filename: &str, source: &str) -> Vec<LspDiagnostic> {
        let mut config = parser::preprocessor::PreprocessorConfig::default();
        config.defines.insert("eval".to_string());
        config.defines.extend(self.defines.iter().cloned());
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
                }
            })
            .collect()
    }

    fn symbol_at(&self, uri: &str, line: u32, col: u32) -> Option<(&Analysis, SymbolId)> {
        let a = self.analyses.get(uri)?;
        Some((a, a.index.find_symbol_at(line, col)?))
    }

    /// Look up type/doc info for a symbol at a given position.
    pub fn hover_info(&self, uri: &str, line: u32, col: u32) -> Option<String> {
        let (a, sym_id) = self.symbol_at(uri, line, col)?;
        let unit = &a.unit;
        analysis::format_hover(
            sym_id,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
        )
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
                    analysis::member_completions(
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
                    )
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

    /// The macro expansions in the document, for inlay hints.
    pub fn macro_expansion_hints(&self, uri: &str) -> Vec<MacroExpansionHint> {
        let Some(a) = self.analyses.get(uri) else {
            return Vec::new();
        };
        a.unit
            .macro_expansions
            .iter()
            .filter_map(|origin| {
                let text = origin.expanded_text.as_ref()?;
                let loc = &origin.call_site;
                if !loc.is_valid() || loc.file_id != a.index.file_id {
                    return None;
                }
                Some(MacroExpansionHint {
                    line: loc.line,
                    column: loc.column,
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
        }
    }
}

#[allow(dead_code)]
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
    (line, (before.len() - start) as u32 + 1)
}

fn uri_to_path(uri: &str) -> String {
    if let Some(path) = uri.strip_prefix("file://") {
        path.replace("%20", " ")
    } else {
        uri.to_string()
    }
}
