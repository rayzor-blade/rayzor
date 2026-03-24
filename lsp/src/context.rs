//! Compilation context for the LSP server.
//!
//! Maintains open file state and provides compilation for diagnostics,
//! hover, goto-definition, and completions.

use crate::analysis::{self, CompletionEntry, FileSymbolIndex};
use compiler::compilation::{CompilationConfig, CompilationUnit};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

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

    /// Load workspace configuration from rayzor.toml if present.
    pub fn load_workspace_config(&mut self) {
        let manifest_path = self.root.join("rayzor.toml");
        if let Ok(content) = std::fs::read_to_string(&manifest_path) {
            if let Ok(manifest) = compiler::workspace::manifest::parse_manifest(&content) {
                if let compiler::workspace::manifest::RayzorManifest::SingleProject(project) =
                    manifest
                {
                    if let Some(ref build) = project.build {
                        self.class_paths = build
                            .class_paths
                            .iter()
                            .map(|cp| self.root.join(cp))
                            .collect();
                    }
                }
            }
        }
    }

    /// Compile a file and return diagnostics. Also builds the symbol index.
    pub fn compile_file(&mut self, uri: &str, source: &str) -> Vec<LspDiagnostic> {
        let filename = uri_to_path(uri);

        let mut config = CompilationConfig {
            load_stdlib: true,
            emit_safety_warnings: false,
            ..Default::default()
        };
        config.pipeline_config = config.pipeline_config.skip_analysis();

        let mut unit = CompilationUnit::new(config);

        for cp in &self.class_paths {
            unit.add_source_path(cp.clone());
        }

        if let Err(e) = unit.load_stdlib() {
            return vec![LspDiagnostic::error(&filename, 0, 0, format!("Stdlib: {}", e))];
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
                    related: e
                        .related_errors
                        .iter()
                        .map(|r| r.clone())
                        .collect(),
                })
                .collect(),
        };

        // Build symbol index for this file (file_id 0 for the main user file)
        let index = FileSymbolIndex::build(&unit.symbol_table, 0);
        self.file_indices.insert(uri.to_string(), index);
        self.last_unit = Some(unit);

        result
    }

    /// Look up type/doc info for a symbol at a given position.
    pub fn hover_info(&self, uri: &str, line: u32, col: u32) -> Option<String> {
        let index = self.file_indices.get(uri)?;
        let unit = self.last_unit.as_ref()?;

        let sym_id =
            index.find_symbol_at(line, col, &unit.symbol_table, &unit.string_interner)?;

        analysis::format_hover(
            sym_id,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
        )
    }

    /// Find definition location for a symbol at a given position.
    pub fn goto_definition(
        &self,
        uri: &str,
        line: u32,
        col: u32,
    ) -> Option<(String, u32, u32)> {
        let index = self.file_indices.get(uri)?;
        let unit = self.last_unit.as_ref()?;

        let sym_id =
            index.find_symbol_at(line, col, &unit.symbol_table, &unit.string_interner)?;
        let sym = unit.symbol_table.get_symbol(sym_id)?;
        let loc = sym.definition_location;

        if !loc.is_valid() {
            return None;
        }

        // For now, file_id 0 = the current file. Cross-file navigation
        // needs a file_id → path map (TODO: populate from compilation).
        let file = if loc.file_id == 0 {
            uri_to_path(uri)
        } else {
            // Try to find the file path from the URI
            uri_to_path(uri)
        };

        Some((file, loc.line, loc.column))
    }

    /// Get completion entries for a position.
    pub fn completions(&self, uri: &str) -> Vec<CompletionEntry> {
        let unit = match self.last_unit.as_ref() {
            Some(u) => u,
            None => return Vec::new(),
        };

        analysis::collect_completions(&unit.symbol_table, &unit.type_table, &unit.string_interner, 0)
    }

    /// Build semantic tokens for syntax highlighting.
    pub fn semantic_tokens(&self, uri: &str) -> Option<Vec<analysis::SemanticToken>> {
        let unit = self.last_unit.as_ref()?;
        let index = self.file_indices.get(uri)?;
        Some(analysis::build_semantic_tokens(
            &unit.symbol_table,
            &unit.string_interner,
            index.file_id,
        ))
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
    pub fn signature_help(&self, uri: &str, line: u32, col: u32) -> Option<analysis::SignatureInfo> {
        let index = self.file_indices.get(uri)?;
        let unit = self.last_unit.as_ref()?;

        // Find the function symbol at or near the cursor
        let sym_id = index.find_symbol_at(line, col, &unit.symbol_table, &unit.string_interner)?;

        // TODO: determine active parameter from cursor position (count commas before cursor)
        analysis::build_signature_help(sym_id, &unit.symbol_table, &unit.type_table, &unit.string_interner, 0)
    }
}

/// A diagnostic ready for LSP conversion.
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
