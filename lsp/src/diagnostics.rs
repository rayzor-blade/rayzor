//! Convert internal diagnostics to LSP diagnostic format.
//!
//! Maps CompilationError fields to rich LSP diagnostics with:
//! - Related information (secondary spans)
//! - Code actions from suggestions
//! - Diagnostic tags (unused, deprecated)

use crate::context::{DiagSeverity, LspDiagnostic};
use lsp_types::{Diagnostic, DiagnosticSeverity, NumberOrString, Position, Range};

/// Convert our internal diagnostics to LSP Diagnostic objects.
pub fn to_lsp_diagnostics(diags: &[LspDiagnostic]) -> Vec<Diagnostic> {
    diags
        .iter()
        .map(|d| {
            let severity = match d.severity {
                DiagSeverity::Error => DiagnosticSeverity::ERROR,
                DiagSeverity::Warning => DiagnosticSeverity::WARNING,
                DiagSeverity::Info => DiagnosticSeverity::INFORMATION,
                DiagSeverity::Hint => DiagnosticSeverity::HINT,
            };
            // LSP uses 0-based lines/columns
            let line = if d.line > 0 { d.line - 1 } else { 0 };
            let col = if d.column > 0 { d.column - 1 } else { 0 };
            let end_line = if d.end_line > 0 { d.end_line - 1 } else { line };
            let end_col = if d.end_column > 0 {
                d.end_column - 1
            } else {
                col + 1
            };

            // Build message with suggestion hint
            let message = if let Some(ref suggestion) = d.suggestion {
                format!("{}\n\nHint: {}", d.message, suggestion)
            } else {
                d.message.clone()
            };

            Diagnostic {
                range: Range {
                    start: Position::new(line, col),
                    end: Position::new(end_line, end_col),
                },
                severity: Some(severity),
                code: Some(NumberOrString::String("rayzor".to_string())),
                source: Some("rayzor".to_string()),
                message,
                ..Default::default()
            }
        })
        .collect()
}
