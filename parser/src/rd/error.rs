//! Parse error types for the recursive descent parser.

use crate::haxe_ast::Span;

/// A parse error with span and message.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl ParseError {
    pub fn new(message: &str, span: Span) -> Self {
        Self {
            message: message.to_string(),
            span,
        }
    }

    pub fn eof(message: &str) -> Self {
        Self {
            message: message.to_string(),
            span: Span::new(0, 0),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at {}..{}: {}",
            self.span.start, self.span.end, self.message
        )
    }
}

impl std::error::Error for ParseError {}

impl From<ParseError> for Vec<ParseError> {
    fn from(e: ParseError) -> Self {
        vec![e]
    }
}

// Convert from token_stream::ParseError
impl From<crate::token_stream::ParseError> for ParseError {
    fn from(e: crate::token_stream::ParseError) -> Self {
        Self {
            message: e.message,
            span: e.span,
        }
    }
}
