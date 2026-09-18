//! Standard error message syntax for parser context strings
//!
//! Context strings follow a structured format that can be parsed during diagnostic generation:
//! `[CODE] message | help: help_text | label: label_text`
//!
//! Examples:
//! - `[E0040] expected '(' after 'if' in case guard | help: case guard expressions must be enclosed in parentheses`
//! - `[E0002] expected ';' after statement | help: add a semicolon at the end of the statement`
//! - `[E0020] expected '(' after 'switch' | help: switch expression must be in parentheses: switch(expr)`

use diagnostics::{Diagnostic, DiagnosticBuilder, SourceSpan};

/// Parse a structured error context string into diagnostic components
pub struct ParsedError {
    pub code: Option<String>,
    pub message: String,
    pub help: Option<String>,
    pub label: Option<String>,
    pub note: Option<String>,
}

impl ParsedError {
    /// Parse a structured context string
    /// Format: `[CODE] message | help: text | label: text | note: text`
    pub fn parse(context: &str) -> Self {
        let mut code = None;
        let mut message = context.to_string();
        let mut help = None;
        let mut label = None;
        let mut note = None;

        // Check if context starts with error code [EXXX]
        if context.starts_with('[')
            && let Some(end) = context.find(']')
        {
            code = Some(context[1..end].to_string());
            message = context[end + 1..].trim().to_string();
        }

        // Split by pipe to find additional fields
        let message_for_split = message.clone();
        let parts: Vec<&str> = message_for_split.splitn(2, " | ").collect();
        if parts.len() == 2 {
            message = parts[0].to_string();

            // Parse additional fields
            for field in parts[1].split(" | ") {
                let field = field.trim();
                if let Some(help_text) = field.strip_prefix("help: ") {
                    help = Some(help_text.to_string());
                } else if let Some(label_text) = field.strip_prefix("label: ") {
                    label = Some(label_text.to_string());
                } else if let Some(note_text) = field.strip_prefix("note: ") {
                    note = Some(note_text.to_string());
                }
            }
        }

        Self {
            code,
            message,
            help,
            label,
            note,
        }
    }

    /// Convert to a diagnostic using the parsed components
    pub fn to_diagnostic(&self, span: SourceSpan) -> Diagnostic {
        let mut builder = DiagnosticBuilder::error(self.message.clone(), span.clone());

        if let Some(code) = &self.code {
            builder = builder.code(code);
        }

        // Use label if provided, otherwise use message
        let label_text = self.label.as_ref().unwrap_or(&self.message);
        builder = builder.label(span, label_text);

        if let Some(help) = &self.help {
            builder = builder.help(help);
        }

        if let Some(note) = &self.note {
            builder = builder.note(note);
        }

        builder.build()
    }
}

/// Helper macro to create structured error context strings
#[macro_export]
macro_rules! error_context {
    // With code and help
    ($code:expr, $msg:expr, help: $help:expr) => {
        concat!("[", $code, "] ", $msg, " | help: ", $help)
    };

    // With code, help, and label
    ($code:expr, $msg:expr, help: $help:expr, label: $label:expr) => {
        concat!(
            "[",
            $code,
            "] ",
            $msg,
            " | help: ",
            $help,
            " | label: ",
            $label
        )
    };

    // With code only
    ($code:expr, $msg:expr) => {
        concat!("[", $code, "] ", $msg)
    };

    // Message only (no code)
    ($msg:expr) => {
        $msg
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_format() {
        let input =
            "[E0040] expected '(' after 'if' | help: use parentheses | label: missing parenthesis";
        let parsed = ParsedError::parse(input);

        assert_eq!(parsed.code, Some("E0040".to_string()));
        assert_eq!(parsed.message, "expected '(' after 'if'");
        assert_eq!(parsed.help, Some("use parentheses".to_string()));
        assert_eq!(parsed.label, Some("missing parenthesis".to_string()));
    }

    #[test]
    fn test_parse_code_and_message() {
        let input = "[E0002] expected semicolon";
        let parsed = ParsedError::parse(input);

        assert_eq!(parsed.code, Some("E0002".to_string()));
        assert_eq!(parsed.message, "expected semicolon");
        assert_eq!(parsed.help, None);
    }

    #[test]
    fn test_parse_message_only() {
        let input = "expected identifier";
        let parsed = ParsedError::parse(input);

        assert_eq!(parsed.code, None);
        assert_eq!(parsed.message, "expected identifier");
        assert_eq!(parsed.help, None);
    }
}
