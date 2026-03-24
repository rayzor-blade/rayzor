//! Compilation context for the LSP server.
//!
//! Maintains open file state and provides compilation for diagnostics.

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
}

impl LspContext {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            open_files: HashMap::new(),
            class_paths: Vec::new(),
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

    /// Compile a file and return diagnostics.
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
            return vec![LspDiagnostic::error(&filename, 0, 0, format!("Parse: {}", e))];
        }

        match unit.lower_to_tast() {
            Ok(_typed_files) => Vec::new(),
            Err(errors) => errors
                .iter()
                .map(|e| LspDiagnostic {
                    file: filename.clone(),
                    line: e.location.line,
                    column: e.location.column,
                    end_line: e.location.line,
                    end_column: e.location.column + 1,
                    severity: DiagSeverity::Error,
                    message: e.message.clone(),
                })
                .collect(),
        }
    }

    /// Look up type/doc info for a symbol at a given position.
    pub fn hover_info(&self, _uri: &str, _line: u32, _col: u32) -> Option<String> {
        // TODO: symbol lookup at position using symbol table
        None
    }

    /// Find definition location for a symbol at a given position.
    pub fn goto_definition(
        &self,
        _uri: &str,
        _line: u32,
        _col: u32,
    ) -> Option<(String, u32, u32)> {
        // TODO: resolve symbol and return definition_location
        None
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
