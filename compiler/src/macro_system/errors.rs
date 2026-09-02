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
            MacroError::UndefinedVariable { location, .. } => *location,
            MacroError::UnsupportedOperation { location, .. } => *location,
            MacroError::DivisionByZero { location } => *location,
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
            MacroError::UndefinedVariable { .. } => "E0710",
            MacroError::UnsupportedOperation { .. } => "E0711",
            MacroError::DivisionByZero { .. } => "E0712",
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
            severity: MacroSeverity::Error,
            message: message.into(),
            location,
            suggestion: None,
        }
    }

    pub fn warning(message: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            severity: MacroSeverity::Warning,
            message: message.into(),
            location,
            suggestion: None,
        }
    }

    pub fn info(message: impl Into<String>, location: SourceLocation) -> Self {
        Self {
            severity: MacroSeverity::Info,
            message: message.into(),
            location,
            suggestion: None,
        }
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

/// Convert a parser Span to a SourceLocation with unknown file context
pub fn span_to_location(span: Span) -> SourceLocation {
    SourceLocation::new(0, 0, 0, span.start as u32)
}
