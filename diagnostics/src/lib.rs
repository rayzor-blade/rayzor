//! Diagnostics library for rich error reporting
//!
//! This library provides Rust-style diagnostics with:
//! - Multiple severity levels (Error, Warning, Info, Hint)
//! - Source code snippets with highlighting
//! - Suggestions with applicability levels
//! - Multi-file source map support
//! - Colored terminal output

use std::fmt;

// Re-export source mapping types from the source_map crate
pub use source_map::{FileId, SourceFile, SourceMap, SourcePosition, SourceSpan};

/// Severity level for diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

impl fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticSeverity::Error => write!(f, "error"),
            DiagnosticSeverity::Warning => write!(f, "warning"),
            DiagnosticSeverity::Info => write!(f, "info"),
            DiagnosticSeverity::Hint => write!(f, "hint"),
        }
    }
}

/// Style for diagnostic labels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelStyle {
    Primary,
    Secondary,
}

/// A label that points to a span of code
#[derive(Debug, Clone)]
pub struct Label {
    pub span: SourceSpan,
    pub message: String,
    pub style: LabelStyle,
}

impl Label {
    pub fn primary(span: SourceSpan, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            style: LabelStyle::Primary,
        }
    }

    pub fn secondary(span: SourceSpan, message: impl Into<String>) -> Self {
        Self {
            span,
            message: message.into(),
            style: LabelStyle::Secondary,
        }
    }
}

/// Applicability level for suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    MachineApplicable,
    HasPlaceholders,
    MaybeIncorrect,
    Unspecified,
}

/// A suggestion for fixing an issue
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub message: String,
    pub span: SourceSpan,
    pub replacement: String,
    pub applicability: Applicability,
}

/// A diagnostic message with severity, labels, and suggestions
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub code: Option<String>,
    pub message: String,
    pub span: SourceSpan,
    pub labels: Vec<Label>,
    pub suggestions: Vec<Suggestion>,
    pub notes: Vec<String>,
    pub help: Vec<String>,
}

/// Collection of diagnostics
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub diagnostics: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn extend(&mut self, other: Diagnostics) {
        self.diagnostics.extend(other.diagnostics);
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error)
    }

    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
    }

    pub fn warnings(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Warning)
    }

    pub fn infos(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Info)
    }

    pub fn hints(&self) -> impl Iterator<Item = &Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Hint)
    }
}

/// Builder for creating diagnostics
pub struct DiagnosticBuilder {
    severity: DiagnosticSeverity,
    code: Option<String>,
    message: String,
    span: SourceSpan,
    labels: Vec<Label>,
    suggestions: Vec<Suggestion>,
    notes: Vec<String>,
    help: Vec<String>,
}

impl DiagnosticBuilder {
    pub fn error(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            severity: DiagnosticSeverity::Error,
            code: None,
            message: message.into(),
            span,
            labels: vec![],
            suggestions: vec![],
            notes: vec![],
            help: vec![],
        }
    }

    pub fn warning(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            severity: DiagnosticSeverity::Warning,
            code: None,
            message: message.into(),
            span,
            labels: vec![],
            suggestions: vec![],
            notes: vec![],
            help: vec![],
        }
    }

    pub fn info(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            severity: DiagnosticSeverity::Info,
            code: None,
            message: message.into(),
            span,
            labels: vec![],
            suggestions: vec![],
            notes: vec![],
            help: vec![],
        }
    }

    pub fn hint(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            severity: DiagnosticSeverity::Hint,
            code: None,
            message: message.into(),
            span,
            labels: vec![],
            suggestions: vec![],
            notes: vec![],
            help: vec![],
        }
    }

    pub fn code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn label(mut self, span: SourceSpan, message: impl Into<String>) -> Self {
        self.labels.push(Label::primary(span, message));
        self
    }

    pub fn secondary_label(mut self, span: SourceSpan, message: impl Into<String>) -> Self {
        self.labels.push(Label::secondary(span, message));
        self
    }

    pub fn suggestion(
        mut self,
        message: impl Into<String>,
        span: SourceSpan,
        replacement: impl Into<String>,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: message.into(),
            span,
            replacement: replacement.into(),
            applicability: Applicability::MachineApplicable,
        });
        self
    }

    pub fn suggestion_with_applicability(
        mut self,
        message: impl Into<String>,
        span: SourceSpan,
        replacement: impl Into<String>,
        applicability: Applicability,
    ) -> Self {
        self.suggestions.push(Suggestion {
            message: message.into(),
            span,
            replacement: replacement.into(),
            applicability,
        });
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn help(mut self, help_msg: impl Into<String>) -> Self {
        self.help.push(help_msg.into());
        self
    }

    pub fn build(self) -> Diagnostic {
        Diagnostic {
            severity: self.severity,
            code: self.code,
            message: self.message,
            span: self.span,
            labels: self.labels,
            suggestions: self.suggestions,
            notes: self.notes,
            help: self.help,
        }
    }
}

/// Formatter for displaying diagnostics
pub struct ErrorFormatter {
    use_colors: bool,
}

impl ErrorFormatter {
    pub fn new() -> Self {
        Self { use_colors: false }
    }

    pub fn with_colors() -> Self {
        Self { use_colors: true }
    }

    pub fn format_diagnostics(&self, diagnostics: &Diagnostics, source_map: &SourceMap) -> String {
        let mut output = String::new();

        for (i, diagnostic) in diagnostics.diagnostics.iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            output.push_str(&self.format_diagnostic(diagnostic, source_map));
        }

        output
    }

    pub fn format_diagnostic(&self, diagnostic: &Diagnostic, source_map: &SourceMap) -> String {
        use ariadne::{Color, Config, Label as ALabel, Report, ReportKind, Source};

        // Map severity to ariadne ReportKind
        let kind = match diagnostic.severity {
            DiagnosticSeverity::Error => ReportKind::Error,
            DiagnosticSeverity::Warning => ReportKind::Warning,
            DiagnosticSeverity::Info => ReportKind::Advice,
            DiagnosticSeverity::Hint => ReportKind::Advice,
        };

        // Get source file info
        let file_name = source_map
            .get_file(diagnostic.span.file_id)
            .map(|f| f.name.as_str())
            .unwrap_or("<unknown>");
        let source_content = source_map
            .get_file(diagnostic.span.file_id)
            .map(|f| f.content.as_str())
            .unwrap_or("");

        // Build the ariadne report
        let offset = diagnostic.span.start.byte_offset;
        let mut builder = Report::<(&str, std::ops::Range<usize>)>::build(kind, file_name, offset)
            .with_config(Config::default().with_color(self.use_colors));

        // Add diagnostic code to message
        let msg = if let Some(code) = &diagnostic.code {
            format!("[{}] {}", code, diagnostic.message)
        } else {
            diagnostic.message.clone()
        };
        builder = builder.with_message(&msg);

        // Add labels
        for label in &diagnostic.labels {
            let start = label.span.start.byte_offset;
            let end = label.span.end.byte_offset.max(start + 1);
            let color = match label.style {
                LabelStyle::Primary => Color::Red,
                LabelStyle::Secondary => Color::Blue,
            };
            builder = builder.with_label(
                ALabel::new((file_name, start..end))
                    .with_message(&label.message)
                    .with_color(color),
            );
        }

        // Add help messages
        for help_msg in &diagnostic.help {
            builder = builder.with_help(help_msg);
        }

        // Add notes
        for note in &diagnostic.notes {
            builder = builder.with_note(note);
        }

        // Add suggestions as notes
        for suggestion in &diagnostic.suggestions {
            builder = builder.with_note(format!("suggestion: {}", suggestion.message));
        }

        // Render to string
        let report = builder.finish();
        let mut buf = Vec::new();
        report
            .write((file_name, Source::from(source_content)), &mut buf)
            .unwrap_or_default();
        String::from_utf8(buf).unwrap_or_default()
    }
}

impl Default for ErrorFormatter {
    fn default() -> Self {
        Self::new()
    }
}

/// Result type that includes diagnostics
pub type DiagnosticResult<T> = Result<T, Diagnostics>;

// Haxe-specific diagnostics
pub mod haxe;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_map() {
        let mut source_map = SourceMap::new();
        let file_id =
            source_map.add_file("test.hx".to_string(), "line 1\nline 2\nline 3".to_string());

        assert_eq!(source_map.get_line(file_id, 1), Some("line 1"));
        assert_eq!(source_map.get_line(file_id, 2), Some("line 2"));
        assert_eq!(source_map.get_line(file_id, 3), Some("line 3"));
        assert_eq!(source_map.get_line(file_id, 4), None);
    }

    #[test]
    fn test_diagnostic_builder() {
        let span = SourceSpan::new(
            SourcePosition::new(1, 5, 4),
            SourcePosition::new(1, 6, 5),
            FileId::new(0),
        );

        let diagnostic = DiagnosticBuilder::error("test error", span.clone())
            .code("E0001")
            .label(span, "here")
            .help("try this")
            .note("additional info")
            .build();

        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.code, Some("E0001".to_string()));
        assert_eq!(diagnostic.message, "test error");
        assert_eq!(diagnostic.labels.len(), 1);
        assert_eq!(diagnostic.help.len(), 1);
        assert_eq!(diagnostic.notes.len(), 1);
    }
}
