//! Enhanced incremental parser with integrated diagnostic collection
//!
//! This module provides an improved version of the incremental parser that
//! collects rich diagnostics instead of simple error strings.

use diagnostics::{Diagnostics, FileId, SourceMap, SourcePosition, SourceSpan};

use crate::enhanced_context::HaxeDiagnostics;
use crate::haxe_ast::HaxeFile;
use nom::IResult;
use nom::character::complete::multispace0;

/// Result of enhanced incremental parsing
#[derive(Debug)]
pub struct EnhancedParseResult {
    pub file: Option<HaxeFile>,
    pub diagnostics: Diagnostics,
    pub source_map: SourceMap,
}

impl EnhancedParseResult {
    pub fn new(file_name: String, input: &str) -> Self {
        let mut source_map = SourceMap::new();
        let _file_id = source_map.add_file(file_name, input.to_string());

        Self {
            file: None,
            diagnostics: Diagnostics::new(),
            source_map,
        }
    }

    pub fn with_file(mut self, file: HaxeFile) -> Self {
        self.file = Some(file);
        self
    }

    pub fn add_diagnostic(&mut self, diagnostic: diagnostics::Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }
}

/// Enhanced incremental parser that collects rich diagnostics
pub fn parse_incrementally_enhanced(file_name: &str, input: &str) -> EnhancedParseResult {
    let mut result = EnhancedParseResult::new(file_name.to_string(), input);
    let file_id = FileId::new(0); // First file in source map
    // let mut collector = ContextErrorCollector::new(file_id);

    // Always run style validation for enhanced diagnostics
    validate_source_style(input, &mut result, file_id);

    // Try to parse the full file first
    if let Ok(file) = crate::haxe_parser::parse_haxe_file(file_name, input, false) {
        return result.with_file(file);
    }

    // If full parsing fails, try incremental parsing with enhanced diagnostics
    let mut current_input = input;
    let mut byte_offset = 0;
    let mut line_number = 1;
    let mut column_number = 1;

    while !current_input.trim().is_empty() {
        // Try to identify the next declaration type
        let declaration_result = identify_next_declaration(current_input);

        match declaration_result {
            Ok((remaining, decl_type)) => {
                let consumed = current_input.len() - remaining.len();

                // Try to parse this specific declaration
                match parse_single_declaration(current_input, &decl_type) {
                    Ok((new_remaining, _)) => {
                        // Successfully parsed declaration
                        let new_consumed = current_input.len() - new_remaining.len();
                        update_position(
                            &mut byte_offset,
                            &mut line_number,
                            &mut column_number,
                            &current_input[..new_consumed],
                        );
                        current_input = new_remaining;
                    }
                    Err(_) => {
                        // Failed to parse declaration - create enhanced diagnostic
                        let span = create_span_for_error(
                            line_number,
                            column_number,
                            byte_offset,
                            consumed,
                            file_id,
                        );
                        let diagnostic = create_enhanced_diagnostic_for_declaration(
                            &decl_type,
                            span,
                            current_input,
                        );
                        result.add_diagnostic(diagnostic);

                        // Skip to next potential declaration
                        let (new_input, new_consumed) = skip_to_next_declaration(current_input);
                        update_position(
                            &mut byte_offset,
                            &mut line_number,
                            &mut column_number,
                            &current_input[..new_consumed],
                        );
                        current_input = new_input;
                    }
                }
            }
            Err(_) => {
                // Couldn't identify declaration type - create generic error
                let span =
                    create_span_for_error(line_number, column_number, byte_offset, 1, file_id);
                let diagnostic = HaxeDiagnostics::unexpected_token(
                    span,
                    current_input
                        .chars()
                        .next()
                        .unwrap_or(' ')
                        .to_string()
                        .as_str(),
                    &[
                        "class".to_string(),
                        "function".to_string(),
                        "interface".to_string(),
                    ],
                );
                result.add_diagnostic(diagnostic);

                // Skip one character and continue
                if let Some(ch) = current_input.chars().next() {
                    let char_len = ch.len_utf8();
                    update_position(
                        &mut byte_offset,
                        &mut line_number,
                        &mut column_number,
                        &current_input[..char_len],
                    );
                    current_input = &current_input[char_len..];
                } else {
                    break;
                }
            }
        }
    }

    // Add any diagnostics collected during context parsing
    // for diagnostic in collector.into_diagnostics() {
    //     result.add_diagnostic(diagnostic);
    // }

    result
}

/// Strip access modifiers and other keywords from the beginning of a line
fn strip_modifiers(s: &str) -> &str {
    let modifiers = [
        "public ",
        "private ",
        "static ",
        "override ",
        "inline ",
        "extern ",
        "final ",
    ];
    let mut result = s;
    loop {
        let mut changed = false;
        for m in &modifiers {
            if result.starts_with(m) {
                result = &result[m.len()..];
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    result
}

/// Post-parse validation that detects common style issues even when the full parse succeeds.
/// The enhanced parser provides richer diagnostics — missing semicolons, braces, etc.
fn validate_source_style(input: &str, result: &mut EnhancedParseResult, file_id: FileId) {
    let lines: Vec<&str> = input.lines().collect();
    let mut brace_depth: i32 = 0;
    let mut line_byte_offset: usize = 0;

    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let line_number = line_idx + 1;

        // Count braces on this line to track depth
        let mut in_string = false;
        for ch in trimmed.chars() {
            match ch {
                '"' | '\'' => in_string = !in_string,
                '{' if !in_string => brace_depth += 1,
                '}' if !in_string => brace_depth -= 1,
                _ => {}
            }
        }

        let stripped = strip_modifiers(trimmed);

        // Check for var declarations without trailing semicolon (inside class/function bodies)
        if brace_depth >= 1
            && stripped.starts_with("var ")
            && !trimmed.ends_with(';')
            && !trimmed.ends_with('{')
            && !trimmed.is_empty()
        {
            let end_of_content = line.trim_end().len();
            let byte_pos = line_byte_offset + end_of_content;
            let span = create_span_for_error(line_number, end_of_content + 1, byte_pos, 1, file_id);
            result.add_diagnostic(HaxeDiagnostics::missing_semicolon(
                span,
                "variable declaration",
            ));
        }

        // Check for function declarations without opening brace
        if brace_depth >= 1 && stripped.starts_with("function ") {
            let has_brace = trimmed.contains('{');
            if !has_brace {
                // Check if next non-empty line starts with '{'
                let next_has_brace = lines
                    .get(line_idx + 1)
                    .map(|l| l.trim().starts_with('{'))
                    .unwrap_or(false);
                if !next_has_brace {
                    let end_of_content = line.trim_end().len();
                    let byte_pos = line_byte_offset + end_of_content;
                    let span = create_span_for_error(
                        line_number,
                        end_of_content + 1,
                        byte_pos,
                        1,
                        file_id,
                    );
                    result.add_diagnostic(HaxeDiagnostics::missing_closing_delimiter(
                        span.clone(),
                        span,
                        '{',
                    ));
                }
            }
        }

        line_byte_offset += line.len() + 1; // +1 for newline character
    }
}

/// Identify the next declaration type in the input
fn identify_next_declaration(input: &str) -> IResult<&str, String> {
    let (input, _) = multispace0(input)?;
    let trimmed = input.trim_start();

    let keywords = [
        "package",
        "import",
        "using",
        "class",
        "interface",
        "enum",
        "typedef",
        "abstract",
        "function",
    ];

    for keyword in &keywords {
        if trimmed.starts_with(keyword) {
            return Ok((input, keyword.to_string()));
        }
    }

    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Tag,
    )))
}

/// Parse a single declaration based on its type
fn parse_single_declaration<'a>(
    input: &'a str,
    _decl_type: &str,
    // _collector: &mut ContextErrorCollector,
) -> IResult<&'a str, ()> {
    // For now, just consume until semicolon or brace
    // TODO: Integrate with actual declaration parsers
    consume_until_delimiter(input)
}

/// Consume input until we find a delimiter that indicates end of declaration
fn consume_until_delimiter(input: &str) -> IResult<&str, ()> {
    let _remaining = input;
    let mut brace_depth = 0;
    let mut in_string = false;
    let mut chars = input.chars();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_string => in_string = true,
            '"' if in_string => in_string = false,
            '{' if !in_string => brace_depth += 1,
            '}' if !in_string => {
                brace_depth -= 1;
                if brace_depth == 0 {
                    // Found end of declaration
                    let consumed = input.len() - chars.as_str().len();
                    return Ok((&input[consumed..], ()));
                }
            }
            ';' if !in_string && brace_depth == 0 => {
                // Found end of declaration
                let consumed = input.len() - chars.as_str().len();
                return Ok((&input[consumed..], ()));
            }
            _ => {}
        }
    }

    // Consumed everything
    Ok(("", ()))
}

/// Create an enhanced diagnostic for a failed declaration
fn create_enhanced_diagnostic_for_declaration(
    decl_type: &str,
    span: SourceSpan,
    input: &str,
) -> diagnostics::Diagnostic {
    match decl_type {
        "class" => {
            if !input.contains('{') {
                HaxeDiagnostics::missing_closing_delimiter(span.clone(), span, '{')
            } else if input.matches('{').count() != input.matches('}').count() {
                HaxeDiagnostics::missing_closing_delimiter(span.clone(), span, '}')
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    "class",
                    &["valid class declaration".to_string()],
                )
            }
        }
        "function" => {
            if !input.contains('(') {
                HaxeDiagnostics::missing_closing_delimiter(span.clone(), span, '(')
            } else if !input.contains(')') {
                HaxeDiagnostics::missing_closing_delimiter(span.clone(), span, ')')
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    "function",
                    &["valid function declaration".to_string()],
                )
            }
        }
        "import" => {
            if !input.contains(';') {
                HaxeDiagnostics::missing_semicolon(span, "import statement")
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    "import",
                    &["valid import path".to_string()],
                )
            }
        }
        _ => HaxeDiagnostics::unexpected_token(span, decl_type, &["valid declaration".to_string()]),
    }
}

/// Create a source span for an error at the given position
fn create_span_for_error(
    line: usize,
    column: usize,
    byte_offset: usize,
    length: usize,
    file_id: FileId,
) -> SourceSpan {
    let start = SourcePosition::new(line, column, byte_offset);
    let end = SourcePosition::new(line, column + length, byte_offset + length);
    SourceSpan::new(start, end, file_id)
}

/// Update position counters based on consumed input
fn update_position(
    byte_offset: &mut usize,
    line_number: &mut usize,
    column_number: &mut usize,
    consumed: &str,
) {
    for ch in consumed.chars() {
        *byte_offset += ch.len_utf8();
        if ch == '\n' {
            *line_number += 1;
            *column_number = 1;
        } else {
            *column_number += 1;
        }
    }
}

/// Skip to the next potential declaration
fn skip_to_next_declaration(input: &str) -> (&str, usize) {
    // Look for common declaration keywords
    let keywords = [
        "class",
        "interface",
        "enum",
        "typedef",
        "abstract",
        "function",
        "package",
        "import",
        "using",
    ];

    for (i, line) in input.lines().enumerate() {
        let trimmed = line.trim();
        if keywords.iter().any(|&keyword| trimmed.starts_with(keyword)) {
            // Found a potential declaration
            let lines_to_skip: usize = input.lines().take(i).map(|l| l.len() + 1).sum();
            return (&input[lines_to_skip..], lines_to_skip);
        }
    }

    // No declaration found, consume everything
    ("", input.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_parse_with_missing_semicolon() {
        let input = r#"
class Test {
    var x = 1
    function test() {
        return x;
    }
}
"#;

        let result = parse_incrementally_enhanced("test.hx", input);

        // Should have diagnostic for missing semicolon
        assert!(!result.diagnostics.is_empty());
        let has_semicolon_error = result
            .diagnostics
            .diagnostics
            .iter()
            .any(|d| d.code == Some("E0002".to_string()));
        assert!(has_semicolon_error);
    }

    #[test]
    fn test_enhanced_parse_with_valid_code() {
        let input = r#"
class Test {
    var x = 1;
    function test() {
        return x;
    }
}
"#;

        let result = parse_incrementally_enhanced("test.hx", input);

        // Should parse successfully with no errors
        assert!(!result.has_errors());
        assert!(result.file.is_some());
    }

    #[test]
    fn test_enhanced_parse_with_missing_braces() {
        let input = r#"
class Test {
    var x = 1;
    function test()
        return x;
    }
"#;

        let result = parse_incrementally_enhanced("test.hx", input);

        // Should have diagnostic for missing opening brace
        assert!(!result.diagnostics.is_empty());
        let has_brace_error = result
            .diagnostics
            .diagnostics
            .iter()
            .any(|d| d.code == Some("E0003".to_string()));
        assert!(has_brace_error);
    }
}
