//! Compilation context for the LSP server.
//!
//! Maintains open file state and provides compilation for diagnostics,
//! hover, goto-definition, and completions.

use crate::analysis::{self, CompletionEntry, FileSymbolIndex};
use compiler::compilation::{CompilationConfig, CompilationUnit};
use std::collections::HashMap;
use std::path::PathBuf;

/// Persistent compilation context shared across LSP requests.
pub struct LspContext {
    /// Root workspace directory.
    pub root: PathBuf,
    /// Open file contents (URI → source text).
    pub open_files: HashMap<String, String>,
    /// Class paths from rayzor.toml.
    pub class_paths: Vec<PathBuf>,
    /// Per-file symbol indices (URI → index).
    file_indices: HashMap<String, FileSymbolIndex>,
    /// Last compilation unit (reused for symbol lookups).
    last_unit: Option<CompilationUnit>,
}

impl LspContext {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            open_files: HashMap::new(),
            class_paths: Vec::new(),
            file_indices: HashMap::new(),
            last_unit: None,
        }
    }

    /// Load class paths from rayzor.toml if present. The workspace root comes
    /// last, as haxe always searches the working directory.
    pub fn load_workspace_config(&mut self) {
        self.class_paths = self.manifest_class_paths();
        if !self.class_paths.contains(&self.root) {
            self.class_paths.push(self.root.clone());
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
            ..Default::default()
        };
        config.pipeline_config = config.pipeline_config.skip_analysis();

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
        self.file_indices.insert(uri.to_string(), index);
        self.last_unit = Some(unit);

        result
    }

    /// Look up type/doc info for a symbol at a given position.
    pub fn hover_info(&self, uri: &str, line: u32, col: u32) -> Option<String> {
        let index = self.file_indices.get(uri)?;
        let unit = self.last_unit.as_ref()?;

        let sym_id = index.find_symbol_at(line, col)?;

        analysis::format_hover(
            sym_id,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
        )
    }

    /// Find definition location for a symbol at a given position.
    pub fn goto_definition(&self, uri: &str, line: u32, col: u32) -> Option<(String, u32, u32)> {
        let index = self.file_indices.get(uri)?;
        let unit = self.last_unit.as_ref()?;

        let sym_id = index.find_symbol_at(line, col)?;
        let sym = unit.symbol_table.get_symbol(sym_id)?;
        let loc = sym.definition_location;

        if !loc.is_valid() {
            return None;
        }

        let file = unit.file_path_of(loc.file_id)?.to_string();
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
        let (Some(index), Some(unit)) = (self.file_indices.get(uri), self.last_unit.as_ref())
        else {
            return Vec::new();
        };
        let Some(sym_id) = index.find_symbol_at(line, col) else {
            return Vec::new();
        };
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

    /// Get completion entries for a position.
    pub fn completions(&self, uri: &str) -> Vec<CompletionEntry> {
        let (Some(unit), Some(index)) = (self.last_unit.as_ref(), self.file_indices.get(uri))
        else {
            return Vec::new();
        };

        analysis::collect_completions(
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            index.file_id,
        )
    }

    /// Build semantic tokens for syntax highlighting.
    pub fn semantic_tokens(&self, uri: &str) -> Option<Vec<analysis::SemanticToken>> {
        let unit = self.last_unit.as_ref()?;
        let index = self.file_indices.get(uri)?;
        Some(analysis::build_semantic_tokens(&unit.symbol_table, index))
    }

    /// Build document symbols for the outline panel.
    pub fn document_symbols(&self, uri: &str) -> Option<Vec<analysis::DocumentSymbolEntry>> {
        let unit = self.last_unit.as_ref()?;
        let index = self.file_indices.get(uri)?;
        Some(analysis::build_document_symbols(
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            index.file_id,
        ))
    }

    /// Build signature help for a function call.
    pub fn signature_help(
        &self,
        uri: &str,
        line: u32,
        col: u32,
    ) -> Option<analysis::SignatureInfo> {
        let index = self.file_indices.get(uri)?;
        let unit = self.last_unit.as_ref()?;

        // Find the function symbol at or near the cursor
        let sym_id = index.find_symbol_at(line, col)?;

        // TODO: determine active parameter from cursor position (count commas before cursor)
        analysis::build_signature_help(
            sym_id,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            0,
        )
    }

    /// Get macro expansion hints for inlay hints display.
    pub fn macro_expansion_hints(&self) -> Vec<MacroExpansionHint> {
        let unit = match self.last_unit.as_ref() {
            Some(u) => u,
            None => return Vec::new(),
        };

        unit.macro_expansions
            .iter()
            .filter_map(|origin| {
                let text = origin.expanded_text.as_ref()?;
                let loc = &origin.call_site;
                if !loc.is_valid() {
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

fn uri_to_path(uri: &str) -> String {
    if let Some(path) = uri.strip_prefix("file://") {
        path.replace("%20", " ")
    } else {
        uri.to_string()
    }
}
