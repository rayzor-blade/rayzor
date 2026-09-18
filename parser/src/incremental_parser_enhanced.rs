//! Enhanced incremental parsing module for Haxe files using diagnostics crate
//!
//! This module provides utilities for parsing Haxe files incrementally,
//! with rich error reporting using the diagnostics crate.

use crate::custom_error::{clear_full_input, set_full_input};
use crate::haxe_ast::*;
use crate::haxe_parser::{
    import_decl, module_field, package_decl, type_declaration, using_decl, ws,
};

use diagnostics::haxe::HaxeDiagnostics;
use diagnostics::{
    Diagnostic, DiagnosticBuilder, Diagnostics, ErrorFormatter, FileId, SourceMap, SourcePosition,
    SourceSpan,
};

/// Result of incremental parsing
#[derive(Debug)]
pub struct IncrementalParseResult {
    /// Successfully parsed elements
    pub parsed_elements: Vec<ParsedElement>,
    /// Diagnostics collected during parsing
    pub diagnostics: Diagnostics,
    /// Source map for the parsed file
    pub source_map: SourceMap,
    /// Whether the entire file was successfully parsed
    pub complete: bool,
    /// Whether we're in error recovery mode
    in_error_recovery: bool,
}

impl IncrementalParseResult {
    pub fn new(file_name: String, input: &str) -> Self {
        let mut source_map = SourceMap::new();
        source_map.add_file(file_name, input.to_string());

        Self {
            parsed_elements: Vec::new(),
            diagnostics: Diagnostics::new(),
            source_map,
            complete: false,
            in_error_recovery: false,
        }
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    pub fn format_diagnostics(&self, use_colors: bool) -> String {
        let formatter = if use_colors {
            ErrorFormatter::with_colors()
        } else {
            ErrorFormatter::new()
        };
        formatter.format_diagnostics(&self.diagnostics, &self.source_map)
    }
}

/// A successfully parsed element
#[derive(Debug)]
pub enum ParsedElement {
    Package(Package),
    Import(Import),
    Using(Using),
    ModuleField(ModuleField),
    TypeDeclaration(TypeDeclaration),
    ConditionalBlock(String), // Simplified for now
}

/// Calculate position in source from remaining input
fn calculate_position(full_input: &str, remaining: &str) -> SourcePosition {
    let consumed = full_input.len() - remaining.len();
    let consumed_str = &full_input[..consumed];
    let line = consumed_str.lines().count().max(1);
    let column = consumed_str
        .lines()
        .last()
        .map(|l| l.len() + 1)
        .unwrap_or(1);
    SourcePosition::new(line, column, consumed)
}

/// Extract context diagnostic from nom ContextualError
fn extract_context_diagnostic(
    error: &crate::custom_error::ContextualError<&str>,
    full_input: &str,
    _remaining: &str,
    file_id: FileId,
) -> Option<Diagnostic> {
    use crate::custom_error::get_deepest_error;

    // Check if we have a deepest error stored and use its offset
    let byte_offset = if let Some((_, offset)) = get_deepest_error() {
        // println!("Using deepest error offset {} for span calculation", offset);
        Some(offset)
    } else {
        error.byte_offset
    };

    // Use the preserved byte offset if available, otherwise calculate from error.input
    let (start, end) = if let Some(byte_offset) = byte_offset {
        // println!("Using preserved byte offset: {}", byte_offset);
        // Calculate position from the preserved byte offset
        let error_pos = &full_input[byte_offset.min(full_input.len())..];
        let start = calculate_position(full_input, error_pos);
        let end = calculate_position(full_input, &error_pos[1.min(error_pos.len())..]);
        (start, end)
    } else {
        // println!("No byte offset, using error.input - first 50 chars: {:?}", &error.input[..50.min(error.input.len())]);
        // Fall back to using error.input
        let start = calculate_position(full_input, error.input);
        let end = calculate_position(full_input, &error.input[1.min(error.input.len())..]);
        (start, end)
    };

    let span = SourceSpan::new(start, end, file_id);

    // Convert the ContextualError to a proper diagnostic
    Some(error.to_diagnostic(span))
}

/// Create a span for an error at the current position
#[allow(dead_code)]
fn create_error_span(
    full_input: &str,
    remaining: &str,
    length: usize,
    file_id: FileId,
) -> SourceSpan {
    let start = calculate_position(full_input, remaining);
    let end = calculate_position(full_input, &remaining[length.min(remaining.len())..]);
    SourceSpan::new(start, end, file_id)
}

/// Analyze syntax error and create appropriate diagnostic
#[allow(dead_code)]
fn create_syntax_diagnostic(
    full_input: &str,
    remaining: &str,
    keyword: &str,
    file_id: FileId,
    in_recovery: bool,
) -> Diagnostic {
    let trimmed = remaining.trim();
    let span = create_error_span(full_input, remaining, keyword.len(), file_id);

    // If we're in error recovery mode, add a note about it
    let mut diagnostic = match keyword {
        "import" => {
            if !trimmed.contains('.') {
                DiagnosticBuilder::error("missing package path in import statement", span.clone())
                    .code("E0030")
                    .label(span, "import statement requires a package path")
                    .help("use format: import package.name.ClassName;")
                    .build()
            } else if !find_statement_end(trimmed).1 {
                let end_span = create_error_span(
                    full_input,
                    &remaining[trimmed.len().saturating_sub(1)..],
                    1,
                    file_id,
                );
                HaxeDiagnostics::missing_semicolon(end_span, "import statement")
            } else if trimmed.contains("..") {
                DiagnosticBuilder::error("invalid double dots in import path", span.clone())
                    .code("E0031")
                    .label(span, "'..' is not valid in import paths")
                    .build()
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    keyword,
                    &["valid import statement".to_string()],
                )
            }
        }
        "using" => {
            if !trimmed.contains('.')
                && trimmed
                    .split_whitespace()
                    .nth(1)
                    .is_none_or(|s| !s.chars().next().unwrap_or(' ').is_uppercase())
            {
                DiagnosticBuilder::error("missing package path in using statement", span.clone())
                    .code("E0032")
                    .label(
                        span,
                        "using statement requires a package path or class name",
                    )
                    .help("use format: using StringTools; or using haxe.macro.Tools;")
                    .build()
            } else if !find_statement_end(trimmed).1 {
                let end_span = create_error_span(
                    full_input,
                    &remaining[trimmed.len().saturating_sub(1)..],
                    1,
                    file_id,
                );
                HaxeDiagnostics::missing_semicolon(end_span, "using statement")
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    keyword,
                    &["valid using statement".to_string()],
                )
            }
        }
        "class" => create_class_diagnostic(full_input, remaining, file_id),
        "interface" => {
            if !trimmed.contains('{') {
                let name_end = trimmed.find(' ').unwrap_or(keyword.len());
                let brace_span = create_error_span(full_input, &remaining[name_end..], 1, file_id);
                HaxeDiagnostics::missing_closing_delimiter(brace_span.clone(), span, '{')
            } else if trimmed.matches('{').count() != trimmed.matches('}').count() {
                let last_char_pos = trimmed.len().saturating_sub(1);
                let closing_span =
                    create_error_span(full_input, &remaining[last_char_pos..], 1, file_id);
                HaxeDiagnostics::missing_closing_delimiter(closing_span, span, '}')
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    keyword,
                    &["valid interface declaration".to_string()],
                )
            }
        }
        "enum" => {
            if !trimmed.contains('{') {
                let name_end = trimmed.find(' ').unwrap_or(keyword.len());
                let brace_span = create_error_span(full_input, &remaining[name_end..], 1, file_id);
                HaxeDiagnostics::missing_closing_delimiter(brace_span.clone(), span, '{')
            } else if trimmed.matches('{').count() != trimmed.matches('}').count() {
                let last_char_pos = trimmed.len().saturating_sub(1);
                let closing_span =
                    create_error_span(full_input, &remaining[last_char_pos..], 1, file_id);
                HaxeDiagnostics::missing_closing_delimiter(closing_span, span, '}')
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    keyword,
                    &["valid enum declaration".to_string()],
                )
            }
        }
        "typedef" => {
            if !trimmed.contains('=') {
                DiagnosticBuilder::error("missing equals sign in typedef", span.clone())
                    .code("E0033")
                    .label(span, "typedef requires '=' after the name")
                    .help("use format: typedef MyType = String;")
                    .build()
            } else if !find_statement_end(trimmed).1 {
                let end_span = create_error_span(
                    full_input,
                    &remaining[trimmed.len().saturating_sub(1)..],
                    1,
                    file_id,
                );
                HaxeDiagnostics::missing_semicolon(end_span, "typedef statement")
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    keyword,
                    &["valid typedef declaration".to_string()],
                )
            }
        }
        "abstract" => {
            if !trimmed.contains('(') || !trimmed.contains(')') {
                DiagnosticBuilder::error("missing parentheses for abstract type", span.clone())
                    .code("E0034")
                    .label(span, "abstract requires underlying type in parentheses")
                    .help("use format: abstract MyInt(Int) { ... }")
                    .build()
            } else if !trimmed.contains('{') {
                let paren_end = trimmed.find(')').unwrap_or(keyword.len()) + 1;
                let brace_span = create_error_span(full_input, &remaining[paren_end..], 1, file_id);
                HaxeDiagnostics::missing_closing_delimiter(brace_span.clone(), span, '{')
            } else {
                HaxeDiagnostics::unexpected_token(
                    span,
                    keyword,
                    &["valid abstract declaration".to_string()],
                )
            }
        }
        _ => HaxeDiagnostics::unexpected_token(span, keyword, &["valid declaration".to_string()]),
    };

    if in_recovery {
        // Add a note if we're in error recovery mode
        if let Some(note) = diagnostic.notes.first() {
            if !note.contains("error recovery") {
                diagnostic.notes.push(
                    "this error may be a consequence of the previous syntax error".to_string(),
                );
            }
        } else {
            diagnostic
                .notes
                .push("this error may be a consequence of the previous syntax error".to_string());
        }
    }

    diagnostic
}

/// Create diagnostic for class-specific syntax errors
#[allow(dead_code)]
fn create_class_diagnostic(full_input: &str, remaining: &str, file_id: FileId) -> Diagnostic {
    let trimmed = remaining.trim();
    let span = create_error_span(full_input, remaining, 5, file_id); // "class" length

    if !trimmed.contains('{') {
        let name_end = trimmed.find(' ').unwrap_or(5);
        let brace_span = create_error_span(full_input, &remaining[name_end..], 1, file_id);
        return HaxeDiagnostics::missing_closing_delimiter(brace_span.clone(), span, '{');
    }

    // Look for common issues within class body
    if let Some(body_start) = trimmed.find('{') {
        let body = &trimmed[body_start..];

        // Check for switch expression without semicolon
        if body.contains("switch")
            && body.contains("case")
            && let Some(switch_start) = body.find("switch")
        {
            let after_switch = &body[switch_start..];
            if let Some(brace_start) = after_switch.find('{')
                && let Some(brace_end) = find_matching_brace(&after_switch[brace_start..])
            {
                let after_switch_block = &after_switch[brace_start + brace_end + 1..].trim_start();

                if !after_switch_block.starts_with(';') && !after_switch_block.is_empty() {
                    let error_pos = body_start + switch_start + brace_start + brace_end + 1;
                    let error_span =
                        create_error_span(full_input, &remaining[error_pos..], 1, file_id);
                    return HaxeDiagnostics::missing_semicolon(error_span, "switch expression");
                }
            }
        }

        // Check for variable declarations missing semicolons
        let lines: Vec<&str> = body.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed_line = line.trim();
            if trimmed_line.starts_with("var ")
                && !trimmed_line.ends_with(';')
                && !trimmed_line.ends_with('{')
            {
                // Calculate the position of this line
                let line_start = body.lines().take(i).map(|l| l.len() + 1).sum::<usize>();
                let var_end = line_start + trimmed_line.len();
                let error_pos = body_start + var_end;
                let error_span = create_error_span(full_input, &remaining[error_pos..], 1, file_id);
                return HaxeDiagnostics::missing_semicolon(error_span, "variable declaration");
            }
        }
    }

    // Check for mismatched braces
    if trimmed.matches('{').count() != trimmed.matches('}').count() {
        let last_char_pos = trimmed.len().saturating_sub(1);
        let closing_span = create_error_span(full_input, &remaining[last_char_pos..], 1, file_id);
        HaxeDiagnostics::missing_closing_delimiter(closing_span, span, '}')
    } else {
        // Check class name convention
        if let Some(name) = trimmed.split_whitespace().nth(1)
            && let Some(first_char) = name.chars().next()
            && !first_char.is_uppercase()
        {
            let name_start = trimmed.find(name).unwrap_or(5);
            let name_span =
                create_error_span(full_input, &remaining[name_start..], name.len(), file_id);
            return HaxeDiagnostics::naming_convention(name_span, "class", name, "PascalCase");
        }

        DiagnosticBuilder::error("syntax error in class body", span.clone())
            .code("E0035")
            .label(span, "invalid syntax")
            .help("check for missing semicolons, braces, or invalid syntax")
            .build()
    }
}

/// Find if a statement ends with semicolon
#[allow(dead_code)]
fn find_statement_end(input: &str) -> (usize, bool) {
    let mut in_string = false;
    let mut escape = false;

    for (i, ch) in input.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            ';' if !in_string => return (i, true),
            '\n' if !in_string => return (i, false),
            _ => {}
        }
    }

    (input.len(), false)
}

/// Find the matching closing brace for an opening brace
#[allow(dead_code)]
fn find_matching_brace(input: &str) -> Option<usize> {
    let mut in_string = false;
    let mut escape = false;
    let mut chars = input.char_indices();

    // Skip the opening brace
    if !matches!(chars.next(), Some((_, '{'))) {
        return None;
    }
    let mut depth = 1;

    for (i, ch) in chars {
        if escape {
            escape = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse a Haxe file incrementally with diagnostics
pub fn parse_incrementally_enhanced(file_name: &str, input: &str) -> IncrementalParseResult {
    // Set the full input for automatic byte offset calculation in error contexts
    set_full_input(input);

    let mut result = IncrementalParseResult::new(file_name.to_string(), input);
    let file_id = FileId::new(0); // First file in source map
    let mut current_input = input;
    let full_input = input;

    // Skip leading whitespace
    if let Ok((input, _)) = ws(current_input) {
        current_input = input;
    }

    // Try to parse package declaration
    match package_decl(full_input, current_input) {
        Ok((remaining, pkg)) => {
            result.parsed_elements.push(ParsedElement::Package(pkg));
            current_input = remaining;
        }
        Err(_) => {
            // Package is optional, continue
        }
    }

    // Parse remaining elements
    let mut _loop_iteration = 0;
    while !current_input.trim().is_empty() {
        _loop_iteration += 1;
        let _input_before = current_input;

        // Skip whitespace
        if let Ok((input, _)) = ws(current_input) {
            current_input = input;
            if current_input.trim().is_empty() {
                break;
            }
        }

        // If we're in error recovery mode and this looks like it's inside a class/function,
        // skip to the next potential top-level declaration
        if result.in_error_recovery {
            let trimmed = current_input.trim_start();
            // Check if this line looks like it's inside a block (starts with case, return, etc.)
            if trimmed.starts_with("case ")
                || trimmed.starts_with("return ")
                || trimmed.starts_with("};")
                || trimmed.starts_with("}")
                || trimmed.starts_with("\"")
                || trimmed.starts_with("default")
            {
                // Skip this line - it's part of the errored structure
                if let Some(newline_pos) = current_input.find('\n') {
                    current_input = &current_input[newline_pos + 1..];
                    continue;
                } else {
                    break;
                }
            }

            // Look for the next class/interface/enum/typedef/abstract keyword
            let keywords = [
                "class ",
                "interface ",
                "enum ",
                "typedef ",
                "abstract ",
                "import ",
                "using ",
                "package ",
            ];
            let mut found_keyword_pos = None;
            for keyword in &keywords {
                if let Some(pos) = current_input.find(keyword) {
                    // Make sure it's at the start of a line
                    if pos == 0 || current_input[..pos].chars().all(|c| c.is_whitespace()) {
                        found_keyword_pos = Some(pos);
                        break;
                    }
                }
            }

            if let Some(pos) = found_keyword_pos {
                current_input = &current_input[pos..];
                result.in_error_recovery = false; // Reset recovery flag
            } else {
                // No more declarations found
                break;
            }
        }

        // Try imports
        if let Ok((remaining, import)) = import_decl(full_input, current_input) {
            result.parsed_elements.push(ParsedElement::Import(import));
            current_input = remaining;
            continue;
        }

        // Try using
        if let Ok((remaining, using)) = using_decl(full_input, current_input) {
            result.parsed_elements.push(ParsedElement::Using(using));
            current_input = remaining;
            continue;
        }

        // Try module fields
        if let Ok((remaining, module_field)) = module_field(full_input, current_input) {
            result
                .parsed_elements
                .push(ParsedElement::ModuleField(module_field));
            current_input = remaining;
            continue;
        }

        // Try type declarations
        match type_declaration(full_input, current_input) {
            Ok((remaining, type_decl)) => {
                result
                    .parsed_elements
                    .push(ParsedElement::TypeDeclaration(type_decl));
                current_input = remaining;
                // Reset error recovery flag on successful parse
                result.in_error_recovery = false;
                continue;
            }
            Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => {
                // Only report the error if we're not already in error recovery
                if !result.in_error_recovery {
                    // Extract context from nom VerboseError and convert to diagnostic
                    if let Some(diagnostic) =
                        extract_context_diagnostic(&e, full_input, current_input, file_id)
                    {
                        result.diagnostics.push(diagnostic);
                        result.in_error_recovery = true;
                    }
                }

                // Skip to next line for error recovery
                if let Some(newline_pos) = current_input.find('\n') {
                    current_input = &current_input[newline_pos + 1..];
                } else {
                    break;
                }
            }
            Err(nom::Err::Incomplete(_)) => {
                // Not enough input, break
                break;
            }
        }

        // // If we get here, we have a parse error
        // let trimmed = current_input.trim_start();

        // // Check for specific keywords
        // let keywords = ["import", "using", "class", "interface", "enum", "typedef", "abstract"];
        // let mut found_keyword = None;

        // for keyword in &keywords {
        //     if trimmed.starts_with(keyword) {
        //         found_keyword = Some(*keyword);
        //         break;
        //     }
        // }

        // if let Some(keyword) = found_keyword {
        //     // Create diagnostic for this specific error
        //     let diagnostic = create_syntax_diagnostic(full_input, current_input, keyword, file_id, result.in_error_recovery);
        //     result.diagnostics.push(diagnostic);

        //     // Set error recovery flag after first error
        //     if !result.in_error_recovery {
        //         result.in_error_recovery = true;
        //     }

        //     // Try to recover by finding the next declaration
        //     let mut next_pos = None;
        //     for search_keyword in &keywords {
        //         if let Some(pos) = current_input[1..].find(search_keyword) {
        //             if next_pos.is_none() || pos < next_pos.unwrap() {
        //                 next_pos = Some(pos + 1);
        //             }
        //         }
        //     }

        //     if let Some(pos) = next_pos {
        //         current_input = &current_input[pos..];
        //     } else {
        //         // No more keywords found, skip to next line
        //         if let Some(newline_pos) = current_input.find('\n') {
        //             current_input = &current_input[newline_pos + 1..];
        //         } else {
        //             break;
        //         }
        //     }
        // } else {
        //     // Handle metadata-prefixed declarations
        //     if trimmed.starts_with("@") {
        //         // This is a metadata-prefixed declaration - try to parse it normally
        //         // Instead of treating it as an error, attempt to parse the whole line with metadata
        //         match type_declaration(full_input, current_input) {
        //             Ok((remaining, declaration)) => {
        //                 result.parsed_elements.push(ParsedElement::TypeDeclaration(declaration));
        //                 current_input = remaining;
        //                 continue;
        //             }
        //             Err(_) => {
        //                 // If parsing fails, this might be an actual metadata syntax error
        //                 let span = create_error_span(full_input, current_input, 1, file_id);

        //                 // Check if there's a recognizable declaration keyword after the metadata
        //                 let mut found_decl_keyword = None;
        //                 let decl_keywords = ["abstract", "class", "interface", "enum", "typedef"];

        //                 for keyword in &decl_keywords {
        //                     if trimmed.contains(keyword) {
        //                         found_decl_keyword = Some(*keyword);
        //                         break;
        //                     }
        //                 }

        //                 let diagnostic = if let Some(keyword) = found_decl_keyword {
        //                     DiagnosticBuilder::error(
        //                         format!("malformed metadata before {} declaration", keyword),
        //                         span.clone()
        //                     )
        //                     .code("E0039")
        //                     .label(span, "check metadata syntax")
        //                     .help("metadata should be in format @:name or @name")
        //                     .build()
        //                 } else {
        //                     DiagnosticBuilder::error(
        //                         "malformed metadata or missing declaration",
        //                         span.clone()
        //                     )
        //                     .code("E0040")
        //                     .label(span, "metadata should be followed by a declaration")
        //                     .help("use format: @:metadata abstract MyType {}")
        //                     .build()
        //                 };

        //                 result.diagnostics.push(diagnostic);

        //                 // Try to find the next valid declaration for recovery
        //                 let mut next_pos = None;
        //                 let all_keywords = ["import", "using", "class", "interface", "enum", "typedef", "abstract"];
        //                 for search_keyword in &all_keywords {
        //                     if let Some(pos) = current_input[1..].find(search_keyword) {
        //                         if next_pos.is_none() || pos < next_pos.unwrap() {
        //                             next_pos = Some(pos + 1);
        //                         }
        //                     }
        //                 }

        //                 if let Some(pos) = next_pos {
        //                     current_input = &current_input[pos..];
        //                 } else {
        //                     // Skip to next line if no keywords found
        //                     if let Some(newline_pos) = current_input.find('\n') {
        //                         current_input = &current_input[newline_pos + 1..];
        //                     } else {
        //                         break;
        //                     }
        //                 }
        //             }
        //         }

        //         result.in_error_recovery = true;
        //         continue;
        //     }

        //     // Handle conditional compilation blocks
        //     if trimmed.starts_with("#if") {
        //         if let Some(end_pos) = current_input.find("#end") {
        //             let block = &current_input[..end_pos + 4];
        //             result.parsed_elements.push(ParsedElement::ConditionalBlock(block.to_string()));
        //             current_input = &current_input[end_pos + 4..];
        //             continue;
        //         }
        //     }

        //     // Unknown syntax error
        //     let span = create_error_span(full_input, current_input, 1, file_id);
        //     let first_token = trimmed.split_whitespace().next().unwrap_or("");

        //     let mut diagnostic = if first_token.is_empty() {
        //         DiagnosticBuilder::error("unexpected end of input", span)
        //             .code("E0036")
        //             .build()
        //     } else if first_token == "{" {
        //         DiagnosticBuilder::error("unexpected opening brace", span.clone())
        //             .code("E0037")
        //             .label(span, "unexpected '{' at this position")
        //             .help("check for syntax errors in the preceding code")
        //             .build()
        //     } else if first_token == "}" {
        //         HaxeDiagnostics::unexpected_token(span, "}", &["declaration or statement".to_string()])
        //     } else if first_token.chars().next().unwrap_or(' ').is_alphabetic() {
        //         // Check for common typos
        //         if let Some(suggestion) = match first_token {
        //             "fucntion" => Some("function"),
        //             "calss" => Some("class"),
        //             "improt" => Some("import"),
        //             _ => None
        //         } {
        //             HaxeDiagnostics::invalid_identifier(span, first_token,
        //                 &format!("did you mean '{}'?", suggestion))
        //         } else {
        //             HaxeDiagnostics::unexpected_token(span, first_token,
        //                 &["type declaration".to_string(), "import".to_string(), "using".to_string()])
        //         }
        //     } else {
        //         DiagnosticBuilder::error(
        //             format!("unexpected character '{}'", first_token.chars().next().unwrap_or('?')),
        //             span
        //         )
        //         .code("E0038")
        //         .build()
        //     };

        //     // Add note if in error recovery
        //     if result.in_error_recovery {
        //         diagnostic.notes.push("this error may be a consequence of the previous syntax error".to_string());
        //     } else {
        //         result.in_error_recovery = true;
        //     }

        //     result.diagnostics.push(diagnostic);

        //     // Skip to next line for recovery
        //     if let Some(newline_pos) = current_input.find('\n') {
        //         current_input = &current_input[newline_pos + 1..];
        //     } else {
        //         break;
        //     }
        // }

        // // Safety check to prevent infinite loops
        // if current_input == input_before {
        //     if current_input.len() > 1 {
        //         current_input = &current_input[1..];
        //     } else {
        //         break;
        //     }
        // }
    }

    // Check if we consumed all input
    result.complete = current_input.trim().is_empty();

    // Clear the full input after parsing
    clear_full_input();

    result
}
