//! Enhanced context system for better error reporting with suggestions
//!
//! This module provides a sophisticated error reporting system that can:
//! - Generate multiple diagnostics from a single parsing failure
//! - Provide specific suggestions for common syntax errors
//! - Add helpful notes and hints for complex scenarios
//! - Integrate seamlessly with nom parsing

use diagnostics::{Diagnostic, DiagnosticBuilder, FileId, SourcePosition, SourceSpan};
use nom::{Parser, error::ParseError as NomParseError};

/// Helper utilities for error suggestions
pub struct ErrorHelpers;

impl ErrorHelpers {
    /// Suggest corrections for common typos in keywords
    pub fn suggest_keyword(input: &str) -> Option<String> {
        let suggestions = [
            ("fucntion", "function"),
            ("classe", "class"),
            ("publik", "public"),
            ("privat", "private"),
            ("statik", "static"),
            ("overrid", "override"),
            ("abstrak", "abstract"),
            ("interfac", "interface"),
            ("extend", "extends"),
            ("implemen", "implements"),
            ("packg", "package"),
            ("imprt", "import"),
            ("varr", "var"),
            ("finall", "final"),
            ("retur", "return"),
            ("swittch", "switch"),
            ("whil", "while"),
        ];

        // Direct match
        for (typo, correct) in &suggestions {
            if input == *typo {
                return Some(correct.to_string());
            }
        }

        // Levenshtein distance for close matches
        let keywords = [
            "abstract",
            "break",
            "case",
            "cast",
            "catch",
            "class",
            "continue",
            "default",
            "do",
            "dynamic",
            "else",
            "enum",
            "extends",
            "extern",
            "false",
            "final",
            "for",
            "function",
            "if",
            "implements",
            "import",
            "in",
            "inline",
            "interface",
            "macro",
            "new",
            "null",
            "override",
            "package",
            "private",
            "public",
            "return",
            "static",
            "super",
            "switch",
            "this",
            "throw",
            "true",
            "try",
            "typedef",
            "untyped",
            "using",
            "var",
            "while",
        ];

        let input_lower = input.to_lowercase();
        let mut best_match = None;
        let mut best_distance = usize::MAX;

        for keyword in &keywords {
            let distance = levenshtein_distance(&input_lower, keyword);
            if distance < best_distance && distance <= 2 {
                best_distance = distance;
                best_match = Some(keyword.to_string());
            }
        }

        best_match
    }
}

/// Calculate Levenshtein distance between two strings
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row = vec![0; b_len + 1];

    for i in 1..=a_len {
        curr_row[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr_row[j] = (prev_row[j] + 1) // deletion
                .min(curr_row[j - 1] + 1) // insertion
                .min(prev_row[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

/// Enhanced context that can generate rich diagnostics
pub trait EnhancedContext<I, O> {
    fn enhanced_context<F>(
        self,
        message: &'static str,
        diagnostic_builder: F,
    ) -> impl Parser<I, Output = O, Error = nom::error::Error<I>>
    where
        F: Fn(&I, &nom::error::Error<I>) -> Diagnostic;
}

impl<I, O, E, P> EnhancedContext<I, O> for P
where
    I: Clone,
    P: Parser<I, Output = O, Error = E>,
    E: NomParseError<I>,
{
    fn enhanced_context<F>(
        mut self,
        _message: &'static str,
        _diagnostic_builder: F,
    ) -> impl Parser<I, Output = O, Error = nom::error::Error<I>>
    where
        F: Fn(&I, &nom::error::Error<I>) -> Diagnostic,
    {
        move |input: I| match self.parse(input.clone()) {
            Ok(result) => Ok(result),
            Err(nom::Err::Error(_e)) | Err(nom::Err::Failure(_e)) => {
                let enhanced_error =
                    nom::error::Error::new(input.clone(), nom::error::ErrorKind::Fail);
                Err(nom::Err::Error(enhanced_error))
            }
            Err(nom::Err::Incomplete(needed)) => Err(nom::Err::Incomplete(needed)),
        }
    }
}

/// Common diagnostic builders for typical Haxe parsing scenarios
pub struct HaxeDiagnostics;

impl HaxeDiagnostics {
    /// Create a missing semicolon diagnostic with suggestion
    pub fn missing_semicolon(span: SourceSpan, after_what: &str) -> Diagnostic {
        DiagnosticBuilder::error(format!("expected `;` after {}", after_what), span.clone())
            .code("E0002")
            .label(span.clone(), "expected `;` here".to_string())
            .suggestion("add `;`", span.clone(), ";".to_string())
            .help(format!("add a semicolon after the {}", after_what))
            .build()
    }

    /// Create a missing closing delimiter diagnostic
    pub fn missing_closing_delimiter(
        opening_span: SourceSpan,
        expected_close_span: SourceSpan,
        delimiter: char,
    ) -> Diagnostic {
        let closing_delimiter = match delimiter {
            '{' => '}',
            '(' => ')',
            '[' => ']',
            '<' => '>',
            _ => delimiter,
        };

        let delimiter_name = match delimiter {
            '{' => "brace",
            '(' => "parenthesis",
            '[' => "bracket",
            '<' => "angle bracket",
            _ => "delimiter",
        };

        DiagnosticBuilder::error(format!("unclosed {}", delimiter_name), opening_span.clone())
            .code("E0003")
            .label(opening_span.clone(), format!("unclosed `{}`", delimiter))
            .secondary_label(
                expected_close_span.clone(),
                "help: closing delimiter expected here",
            )
            .suggestion(
                format!("add closing `{}`", closing_delimiter),
                expected_close_span.clone(),
                closing_delimiter.to_string(),
            )
            .help(format!(
                "add a closing `{}` to match the opening `{}`",
                closing_delimiter, delimiter
            ))
            .build()
    }

    /// Create an unexpected token diagnostic with suggestions
    pub fn unexpected_token(span: SourceSpan, found: &str, expected: &[String]) -> Diagnostic {
        let mut builder = if expected.len() == 1 {
            DiagnosticBuilder::error(
                format!("expected `{}`, found `{}`", expected[0], found),
                span.clone(),
            )
        } else {
            DiagnosticBuilder::error(format!("unexpected token `{}`", found), span.clone())
        };

        builder = builder
            .code("E0001")
            .label(span.clone(), "unexpected token");

        if !expected.is_empty() {
            let expected_str = Self::format_expected_list(expected);
            builder = builder.help(format!("expected {}", expected_str));
        }

        // Try to suggest keyword corrections
        if let Some(suggestion) = ErrorHelpers::suggest_keyword(found) {
            builder = builder.suggestion(
                format!("replace with `{}`", suggestion),
                span.clone(),
                suggestion,
            );
        }

        builder.build()
    }

    /// Create a diagnostic for invalid identifier with suggestion
    pub fn invalid_identifier(span: SourceSpan, name: &str, reason: &str) -> Diagnostic {
        let mut builder =
            DiagnosticBuilder::warning(format!("invalid identifier `{}`", name), span.clone())
                .code("W0001")
                .label(span.clone(), reason.to_string());

        // Try to suggest keyword corrections
        if let Some(suggestion) = ErrorHelpers::suggest_keyword(name) {
            builder = builder
                .suggestion(
                    format!("replace with `{}`", suggestion),
                    span.clone(),
                    suggestion.clone(),
                )
                .help(format!("did you mean `{}`?", suggestion));
        }

        builder.build()
    }

    /// Create a diagnostic for missing import with suggestions
    pub fn missing_import_suggestion(span: SourceSpan, type_name: &str) -> Diagnostic {
        let common_imports = [
            ("String", "import String"),
            ("Array", "import haxe.ds.Array"),
            ("Map", "import haxe.ds.Map"),
            ("StringMap", "import haxe.ds.StringMap"),
            ("IntMap", "import haxe.ds.IntMap"),
            ("Vector", "import haxe.ds.Vector"),
            ("Date", "import Date"),
            ("Math", "import Math"),
            ("Reflect", "import Reflect"),
            ("Type", "import Type"),
            ("Json", "import haxe.Json"),
            ("Http", "import haxe.Http"),
            ("Timer", "import haxe.Timer"),
        ];

        let mut builder = DiagnosticBuilder::warning(
            format!("type `{}` might need to be imported", type_name),
            span.clone(),
        )
        .code("W0003")
        .label(span.clone(), "unknown type");

        // Look for common import suggestions
        for (name, import_stmt) in &common_imports {
            if type_name == *name {
                builder = builder
                    .note(format!("add `{}` to the top of your file", import_stmt))
                    .help("this type is available in the standard library".to_string());
                break;
            }
        }

        builder.build()
    }

    /// Create a diagnostic for incomplete switch expression
    pub fn incomplete_switch_expression(span: SourceSpan, missing_cases: &[String]) -> Diagnostic {
        let mut builder =
            DiagnosticBuilder::warning("incomplete switch expression".to_string(), span.clone())
                .code("W0002")
                .label(span.clone(), "missing cases");

        for case in missing_cases {
            builder = builder.note(format!("add case for `{}`", case));
        }

        builder = builder.help("consider adding a default case with `case _:`");

        builder.build()
    }

    /// Create multiple diagnostics for common parsing scenarios
    pub fn analyze_function_declaration_error(
        input: &str,
        function_start: SourceSpan,
        file_id: FileId,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        // Check for missing parentheses
        if !input.contains('(') {
            let span = SourceSpan::new(
                function_start.end,
                SourcePosition::new(
                    function_start.end.line,
                    function_start.end.column + 1,
                    function_start.end.byte_offset + 1,
                ),
                file_id,
            );

            diagnostics.push(
                DiagnosticBuilder::error("missing parameter list".to_string(), span.clone())
                    .code("E0004")
                    .label(span.clone(), "expected `()` here")
                    .suggestion("add empty parameter list", span, "()".to_string())
                    .help("function declarations must have parentheses for parameters")
                    .build(),
            );
        }

        // Check for missing return type hint
        if !input.contains(':') && input.contains('{') {
            diagnostics.push(
                DiagnosticBuilder::hint(
                    "consider adding a return type".to_string(),
                    function_start.clone(),
                )
                .code("H0001")
                .help("explicit return types improve code readability")
                .note("use `:Void` if the function doesn't return a value")
                .build(),
            );
        }

        diagnostics
    }

    /// Helper to format expected token list
    fn format_expected_list(expected: &[String]) -> String {
        match expected.len() {
            0 => "nothing".to_string(),
            1 => format!("`{}`", expected[0]),
            2 => format!("`{}` or `{}`", expected[0], expected[1]),
            _ => {
                let (last, rest) = expected.split_last().unwrap();
                format!(
                    "{}, or `{}`",
                    rest.iter()
                        .map(|s| format!("`{}`", s))
                        .collect::<Vec<_>>()
                        .join(", "),
                    last
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::{DiagnosticSeverity, FileId, SourcePosition};

    #[test]
    fn test_missing_semicolon_diagnostic() {
        let span = SourceSpan::new(
            SourcePosition::new(1, 10, 9),
            SourcePosition::new(1, 11, 10),
            FileId::new(0),
        );

        let diagnostic = HaxeDiagnostics::missing_semicolon(span, "variable declaration");

        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.code, Some("E0002".to_string()));
        assert_eq!(
            diagnostic.message,
            "expected `;` after variable declaration"
        );
        assert_eq!(diagnostic.suggestions.len(), 1);
        assert_eq!(diagnostic.suggestions[0].replacement, ";");
    }

    #[test]
    fn test_unexpected_token_diagnostic() {
        let span = SourceSpan::new(
            SourcePosition::new(1, 5, 4),
            SourcePosition::new(1, 9, 8),
            FileId::new(0),
        );

        let diagnostic =
            HaxeDiagnostics::unexpected_token(span, "fucntion", &["function".to_string()]);

        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error);
        assert_eq!(diagnostic.code, Some("E0001".to_string()));
        assert!(!diagnostic.suggestions.is_empty());
        assert_eq!(diagnostic.suggestions[0].replacement, "function");
    }

    #[test]
    fn test_invalid_identifier_diagnostic() {
        let span = SourceSpan::new(
            SourcePosition::new(2, 1, 20),
            SourcePosition::new(2, 6, 25),
            FileId::new(0),
        );

        let diagnostic = HaxeDiagnostics::invalid_identifier(
            span,
            "classe",
            "unknown keyword, did you mean something else?",
        );

        assert_eq!(diagnostic.severity, DiagnosticSeverity::Warning);
        assert_eq!(diagnostic.code, Some("W0001".to_string()));
        assert!(!diagnostic.suggestions.is_empty());
        assert_eq!(diagnostic.suggestions[0].replacement, "class");
    }
}
