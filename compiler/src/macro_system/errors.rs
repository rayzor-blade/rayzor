use crate::pipeline::{CompilationError, CompilationWarning, ErrorCategory, WarningCategory};
use crate::tast::SourceLocation;
use parser::Span;
use std::fmt;

/// Severity level for macro diagnostics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroSeverity {
    Error,
    Warning,
    Info,
}

/// A diagnostic message emitted during macro processing
#[derive(Debug, Clone)]
pub struct MacroDiagnostic {
    pub call_site: Option<SourceLocation>,
    pub severity: MacroSeverity,
    pub message: String,
    pub location: SourceLocation,
    /// Optional suggestion for fixing the issue
    pub suggestion: Option<String>,
}

/// Errors that can occur during macro processing
#[derive(Debug, Clone)]
pub enum MacroError {
    /// Macro function not found
    UndefinedMacro {
        name: String,
        location: SourceLocation,
    },

    /// Wrong number of arguments passed to macro
    ArgumentCountMismatch {
        macro_name: String,
        expected: usize,
        found: usize,
        location: SourceLocation,
    },

    /// Type error during macro evaluation
    TypeError {
        message: String,
        location: SourceLocation,
    },

    /// Runtime error during macro interpretation
    RuntimeError {
        message: String,
        location: SourceLocation,
    },

    /// Macro expansion exceeded recursion depth limit
    RecursionLimitExceeded {
        macro_name: String,
        depth: usize,
        max_depth: usize,
        location: SourceLocation,
    },

    /// The macro body asked a question only the live typer can answer
    /// (`Context.typeof` before typing exists). Not a failure and not
    /// catchable by the macro's own try/catch — it is the signal that the
    /// whole call must be re-run during lowering, and letting a catch arm
    /// swallow it would expand the macro to a wrong constant instead.
    NeedsTyper { location: SourceLocation },

    /// Macro evaluation ran past the time budget set for it
    /// (`budget::limit`). Not catchable by the macro's own try/catch.
    TimeBudgetExceeded { location: SourceLocation },

    /// Circular macro dependency detected
    CircularDependency {
        chain: Vec<String>,
        location: SourceLocation,
    },

    /// Reification error (e.g., invalid dollar-ident usage)
    ReificationError {
        message: String,
        location: SourceLocation,
    },

    /// Invalid macro definition
    InvalidDefinition {
        message: String,
        location: SourceLocation,
    },

    /// Context API error (e.g., getType failed)
    ContextError {
        method: String,
        message: String,
        location: SourceLocation,
    },

    /// Variable not found in macro environment
    UndefinedVariable {
        name: String,
        location: SourceLocation,
    },

    /// Unsupported operation in macro context
    UnsupportedOperation {
        operation: String,
        location: SourceLocation,
    },

    /// Division by zero during macro evaluation
    DivisionByZero { location: SourceLocation },

    /// A value thrown by the macro's own `throw`
    Thrown {
        value: Box<super::value::MacroValue>,
        location: SourceLocation,
    },

    /// Return from macro function
    Return {
        value: Option<Box<super::value::MacroValue>>,
    },

    /// Break from loop
    Break,

    /// Continue in loop
    Continue,
}

impl MacroError {
    /// Get the source location for this error
    pub fn location(&self) -> SourceLocation {
        match self {
            MacroError::UndefinedMacro { location, .. } => *location,
            MacroError::ArgumentCountMismatch { location, .. } => *location,
            MacroError::TypeError { location, .. } => *location,
            MacroError::RuntimeError { location, .. } => *location,
            MacroError::RecursionLimitExceeded { location, .. } => *location,
            MacroError::CircularDependency { location, .. } => *location,
            MacroError::ReificationError { location, .. } => *location,
            MacroError::InvalidDefinition { location, .. } => *location,
            MacroError::ContextError { location, .. } => *location,
            MacroError::NeedsTyper { location } => *location,
            MacroError::TimeBudgetExceeded { location } => *location,
            MacroError::UndefinedVariable { location, .. } => *location,
            MacroError::UnsupportedOperation { location, .. } => *location,
            MacroError::DivisionByZero { location } => *location,
            MacroError::Thrown { location, .. } => *location,
            MacroError::Return { .. } => SourceLocation::unknown(),
            MacroError::Break => SourceLocation::unknown(),
            MacroError::Continue => SourceLocation::unknown(),
        }
    }

    /// Whether this is a control flow signal (not a real error)
    pub fn is_control_flow(&self) -> bool {
        matches!(
            self,
            MacroError::Return { .. } | MacroError::Break | MacroError::Continue
        )
    }

    /// Fine-grained error code for this specific macro error kind.
    ///
    /// Error codes E0700-E0799 are reserved for macro expansion errors:
    /// - E0700: General macro expansion error
    /// - E0701: Undefined macro
    /// - E0702: Argument count mismatch
    /// - E0703: Macro type error
    /// - E0704: Macro runtime error
    /// - E0705: Recursion limit exceeded
    /// - E0706: Circular dependency
    /// - E0707: Reification error
    /// - E0708: Invalid macro definition
    /// - E0709: Context API error
    /// - E0710: Undefined variable in macro
    /// - E0711: Unsupported operation
    /// - E0712: Division by zero
    pub fn error_code(&self) -> &'static str {
        match self {
            MacroError::UndefinedMacro { .. } => "E0701",
            MacroError::ArgumentCountMismatch { .. } => "E0702",
            MacroError::TypeError { .. } => "E0703",
            MacroError::RuntimeError { .. } => "E0704",
            MacroError::RecursionLimitExceeded { .. } => "E0705",
            MacroError::CircularDependency { .. } => "E0706",
            MacroError::ReificationError { .. } => "E0707",
            MacroError::InvalidDefinition { .. } => "E0708",
            MacroError::ContextError { .. } => "E0709",
            MacroError::NeedsTyper { .. } => "E0709",
            MacroError::TimeBudgetExceeded { .. } => "E0713",
            MacroError::UndefinedVariable { .. } => "E0710",
            MacroError::UnsupportedOperation { .. } => "E0711",
            MacroError::DivisionByZero { .. } => "E0712",
            MacroError::Thrown { .. } => "E0704",
            MacroError::Return { .. } | MacroError::Break | MacroError::Continue => "E0700",
        }
    }

    /// Generate a suggestion string for this error
    fn suggestion(&self) -> Option<String> {
        match self {
            MacroError::UndefinedMacro { name, .. } => Some(format!(
                "Check that macro '{}' is defined and imported correctly",
                name
            )),
            MacroError::ArgumentCountMismatch {
                macro_name,
                expected,
                ..
            } => Some(format!(
                "Macro '{}' requires {} argument(s)",
                macro_name, expected
            )),
            MacroError::RecursionLimitExceeded { macro_name, .. } => Some(format!(
                "Check for infinite recursion in macro '{}', or increase the recursion limit",
                macro_name
            )),
            MacroError::CircularDependency { chain, .. } => Some(format!(
                "Break the circular dependency chain: {}",
                chain.join(" -> ")
            )),
            MacroError::UndefinedVariable { name, .. } => Some(format!(
                "Ensure '{}' is defined before use in the macro body",
                name
            )),
            MacroError::DivisionByZero { .. } => {
                Some("Check divisor value before division".to_string())
            }
            _ => None,
        }
    }

    /// Convert this macro error into a CompilationError for the pipeline
    pub fn to_compilation_error(&self) -> CompilationError {
        CompilationError {
            message: format!("[{}] {}", self.error_code(), self),
            location: self.location(),
            category: ErrorCategory::MacroExpansionError,
            suggestion: self.suggestion(),
            related_errors: Vec::new(),
        }
    }
}

impl fmt::Display for MacroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacroError::UndefinedMacro { name, .. } => {
                write!(f, "undefined macro: '{}'", name)
            }
            MacroError::NeedsTyper { .. } => {
                write!(f, "needs the live typer; deferred to lowering")
            }
            MacroError::TimeBudgetExceeded { .. } => {
                write!(f, "macro ran past its time budget and was stopped")
            }
            MacroError::Thrown { value, .. } => {
                write!(f, "uncaught exception: {}", value.to_display_string())
            }
            MacroError::ArgumentCountMismatch {
                macro_name,
                expected,
                found,
                ..
            } => {
                write!(
                    f,
                    "macro '{}' expects {} argument(s), found {}",
                    macro_name, expected, found
                )
            }
            MacroError::TypeError { message, .. } => {
                write!(f, "macro type error: {}", message)
            }
            MacroError::RuntimeError { message, .. } => {
                write!(f, "macro runtime error: {}", message)
            }
            MacroError::RecursionLimitExceeded {
                macro_name,
                depth,
                max_depth,
                ..
            } => {
                write!(
                    f,
                    "macro '{}' exceeded recursion limit: depth {} > max {}",
                    macro_name, depth, max_depth
                )
            }
            MacroError::CircularDependency { chain, .. } => {
                write!(f, "circular macro dependency: {}", chain.join(" -> "))
            }
            MacroError::ReificationError { message, .. } => {
                write!(f, "reification error: {}", message)
            }
            MacroError::InvalidDefinition { message, .. } => {
                write!(f, "invalid macro definition: {}", message)
            }
            MacroError::ContextError {
                method, message, ..
            } => {
                write!(f, "Context.{}(): {}", method, message)
            }
            MacroError::UndefinedVariable { name, .. } => {
                write!(f, "undefined variable in macro: '{}'", name)
            }
            MacroError::UnsupportedOperation { operation, .. } => {
                write!(f, "unsupported operation in macro context: {}", operation)
            }
            MacroError::DivisionByZero { .. } => {
                write!(f, "division by zero in macro evaluation")
            }
            MacroError::Return { .. } => write!(f, "return"),
            MacroError::Break => write!(f, "break"),
            MacroError::Continue => write!(f, "continue"),
        }
    }
}

impl std::error::Error for MacroError {}

/// Convert MacroError directly into a CompilationError
impl From<MacroError> for CompilationError {
    fn from(err: MacroError) -> Self {
        err.to_compilation_error()
    }
}

// --- MacroDiagnostic ---

impl MacroDiagnostic {
    pub fn error(message: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            call_site: None,
            severity: MacroSeverity::Error,
            message: message.into(),
            location,
            suggestion: None,
        }
    }

    pub fn warning(message: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            call_site: None,
            severity: MacroSeverity::Warning,
            message: message.into(),
            location,
            suggestion: None,
        }
    }

    pub fn info(message: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            call_site: None,
            severity: MacroSeverity::Info,
            message: message.into(),
            location,
            suggestion: None,
        }
    }

    pub fn with_call_site(mut self, location: SourceLocation) -> Self {
        self.call_site = Some(location);
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Convert to a CompilationError (for Error severity diagnostics)
    pub fn to_compilation_error(&self) -> CompilationError {
        CompilationError {
            message: self.message.clone(),
            location: self.location,
            category: ErrorCategory::MacroExpansionError,
            suggestion: self.suggestion.clone(),
            related_errors: Vec::new(),
        }
    }

    /// Convert to a CompilationWarning (for Warning/Info severity diagnostics)
    pub fn to_compilation_warning(&self) -> CompilationWarning {
        CompilationWarning {
            message: self.message.clone(),
            location: self.location,
            category: WarningCategory::Correctness,
            suppressible: true,
        }
    }

    /// Convert to the appropriate pipeline type based on severity
    pub fn into_pipeline_diagnostic(self) -> PipelineDiagnostic {
        match self.severity {
            MacroSeverity::Error => PipelineDiagnostic::Error(self.to_compilation_error()),
            MacroSeverity::Warning | MacroSeverity::Info => {
                PipelineDiagnostic::Warning(self.to_compilation_warning())
            }
        }
    }
}

/// Result of converting a MacroDiagnostic to a pipeline-level diagnostic
pub enum PipelineDiagnostic {
    Error(CompilationError),
    Warning(CompilationWarning),
}

// --- Helper: convert parser Span to SourceLocation ---

/// Convert a parser Span (byte offsets) to a SourceLocation.
///
/// When file_id is known, pass it explicitly. For macro-generated code
/// where the file context is available from the SpanConverter, use
/// `SpanConverter::convert_span()` instead for accurate line/column info.
pub fn span_to_source_location(span: Span, file_id: u32) -> SourceLocation {
    SourceLocation::new(file_id, 0, 0, span.start as u32)
}

/// Spans at or above this lie in a file named by a macro position
/// (`Context.makePosition({file: "name", ..})`): bits 32.. index the
/// file in `context_api`'s named-file table, the low 32 bits are the offset.
pub const NAMED_FILE_SPAN: usize = 1 << 48;

/// `SourceLocation::file_id` of a named-file position is this plus its index.
/// Such a location is a range: `byte_offset` is its min, `column` its max.
pub const NAMED_FILE_ID: u32 = 1 << 30;

/// Convert a parser Span to a SourceLocation with unknown file context
pub fn span_to_location(span: Span) -> SourceLocation {
    if span.start >= NAMED_FILE_SPAN {
        let file = ((span.start - NAMED_FILE_SPAN) >> 32) as u32;
        let offset = |at: usize| (at & 0xffff_ffff) as u32;
        return SourceLocation::new(
            NAMED_FILE_ID + file,
            0,
            offset(span.end),
            offset(span.start),
        );
    }
    SourceLocation::new(0, 0, 0, span.start as u32)
}

/// Positions in the exact source view that produced AST spans.
pub(crate) struct SourceLocations<'a> {
    source: Option<&'a str>,
    line_starts: Vec<usize>,
}

impl<'a> SourceLocations<'a> {
    pub(crate) fn new(source: Option<&'a str>) -> Self {
        let mut line_starts = vec![0];
        if let Some(source) = source {
            line_starts.extend(
                source
                    .bytes()
                    .enumerate()
                    .filter_map(|(at, byte)| (byte == b'\n').then_some(at + 1)),
            );
        }
        Self {
            source,
            line_starts,
        }
    }

    pub(crate) fn location(&self, span: Span) -> SourceLocation {
        let mut location = span_to_location(span);
        if location.file_id != 0 {
            return location;
        }
        if let Some(before) = self.source.and_then(|text| text.get(..span.start)) {
            let line = self.line_starts.partition_point(|&at| at <= span.start) - 1;
            location.line = line as u32 + 1;
            location.column = before[self.line_starts[line]..].encode_utf16().count() as u32 + 1;
        }
        location
    }
}

#[cfg(test)]
mod source_location_tests {
    use super::*;

    #[test]
    fn indexed_locations_preserve_utf16_columns_and_unknown_spans() {
        let source = "first\r\n😀é last\n";
        let locations = SourceLocations::new(Some(source));
        for at in 0..=source.len() {
            let span = Span::new(at, at);
            let location = locations.location(span);
            if let Some(before) = source.get(..at) {
                assert_eq!(
                    location.line,
                    before.bytes().filter(|&b| b == b'\n').count() as u32 + 1
                );
                assert_eq!(
                    location.column,
                    before.rsplit('\n').next().unwrap().encode_utf16().count() as u32 + 1
                );
            } else {
                assert_eq!(location, span_to_location(span));
            }
        }
        let named = Span::new(NAMED_FILE_SPAN + 3, NAMED_FILE_SPAN + 8);
        assert_eq!(locations.location(named), span_to_location(named));
        let beyond = Span::new(source.len() + 1, source.len() + 1);
        assert_eq!(locations.location(beyond), span_to_location(beyond));
        assert_eq!(
            SourceLocations::new(None).location(Span::new(3, 4)),
            span_to_location(Span::new(3, 4))
        );
    }
}
