//! Incremental parsing module for Haxe files
//!
//! This module provides utilities for parsing Haxe files incrementally,
//! allowing for better error recovery and partial parsing results.

use crate::haxe_ast::*;
use crate::haxe_parser::{
    import_decl, module_field, package_decl, type_declaration, using_decl, ws,
};

/// Result of incremental parsing
#[derive(Debug)]
pub struct IncrementalParseResult {
    /// Successfully parsed elements
    pub parsed_elements: Vec<ParsedElement>,
    /// Errors encountered during parsing
    pub errors: Vec<ParseError>,
    /// Whether the entire file was successfully parsed
    pub complete: bool,
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

/// Parse error with location information
#[derive(Debug)]
pub struct ParseError {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub remaining_input: String,
}

/// Analyze syntax error to provide helpful error messages
fn analyze_syntax_error(input: &str, keyword: &str) -> String {
    let trimmed = input.trim();

    match keyword {
        "import" => {
            if !trimmed.contains('.') {
                "Missing package path (e.g., 'import haxe.ds.StringMap;')".to_string()
            } else if !trimmed.ends_with(';') {
                "Missing semicolon at end of import statement".to_string()
            } else if trimmed.contains("..") {
                "Invalid double dots in import path".to_string()
            } else {
                "Invalid import syntax".to_string()
            }
        }
        "using" => {
            if !trimmed.contains('.') {
                "Missing package path (e.g., 'using StringTools;')".to_string()
            } else if !trimmed.ends_with(';') {
                "Missing semicolon at end of using statement".to_string()
            } else {
                "Invalid using syntax".to_string()
            }
        }
        "class" => analyze_class_error(trimmed),
        "interface" => {
            if !trimmed.contains('{') {
                "Missing opening brace for interface body".to_string()
            } else if trimmed.matches('{').count() != trimmed.matches('}').count() {
                "Mismatched braces in interface declaration".to_string()
            } else {
                "Invalid interface syntax".to_string()
            }
        }
        "enum" => {
            if !trimmed.contains('{') {
                "Missing opening brace for enum body".to_string()
            } else if trimmed.matches('{').count() != trimmed.matches('}').count() {
                "Mismatched braces in enum declaration".to_string()
            } else {
                "Invalid enum syntax".to_string()
            }
        }
        "typedef" => {
            if !trimmed.contains('=') {
                "Missing equals sign in typedef (e.g., 'typedef MyType = String;')".to_string()
            } else if !trimmed.ends_with(';') {
                "Missing semicolon at end of typedef".to_string()
            } else {
                "Invalid typedef syntax".to_string()
            }
        }
        "abstract" => {
            if !trimmed.contains('(') || !trimmed.contains(')') {
                "Missing parentheses for abstract type (e.g., 'abstract MyInt(Int)')".to_string()
            } else if !trimmed.contains('{') {
                "Missing opening brace for abstract body".to_string()
            } else {
                "Invalid abstract syntax".to_string()
            }
        }
        _ => analyze_general_syntax_error(trimmed),
    }
}

/// Analyze class-specific syntax errors with more detail
fn analyze_class_error(input: &str) -> String {
    if !input.contains('{') {
        return "Missing opening brace for class body".to_string();
    }

    // Look for common issues within class body
    if let Some(body_start) = input.find('{') {
        let body = &input[body_start..];

        // Check for switch expression without semicolon
        if body.contains("switch") && body.contains("case") {
            // Look for switch expressions that might be missing semicolons
            if let Some(switch_start) = body.find("switch") {
                let after_switch = &body[switch_start..];
                if let Some(brace_start) = after_switch.find('{')
                    && let Some(brace_end) = find_matching_brace(&after_switch[brace_start..])
                {
                    let _switch_block = &after_switch[..brace_start + brace_end + 1];
                    let after_switch_block =
                        &after_switch[brace_start + brace_end + 1..].trim_start();

                    // If the next character after switch is not a semicolon and we have more content
                    if !after_switch_block.starts_with(';') && !after_switch_block.is_empty() {
                        return "Missing semicolon after switch expression in variable assignment"
                            .to_string();
                    }
                }
            }
        }

        // Check for other variable declarations missing semicolons
        if body.contains("var ") && !body.ends_with(';') && !body.ends_with('}') {
            // Look for var declarations that might be missing semicolons
            let lines: Vec<&str> = body.lines().collect();
            for line in lines {
                let trimmed_line = line.trim();
                if trimmed_line.starts_with("var ")
                    && !trimmed_line.ends_with(';')
                    && !trimmed_line.ends_with('{')
                {
                    return "Missing semicolon after variable declaration".to_string();
                }
            }
        }

        // Check for function declarations with issues
        if body.contains("function ") && !body.contains("()") && !body.contains("(") {
            return "Missing parentheses in function declaration".to_string();
        }
    }

    // Fallback to general class errors
    if input.matches('{').count() != input.matches('}').count() {
        "Mismatched braces in class declaration".to_string()
    } else if input
        .split_whitespace()
        .nth(1)
        .is_none_or(|s| !s.chars().next().unwrap_or(' ').is_uppercase())
    {
        "Class name should start with uppercase letter".to_string()
    } else {
        "Syntax error in class body - check for missing semicolons, braces, or invalid syntax"
            .to_string()
    }
}

/// Find the matching closing brace for an opening brace
fn find_matching_brace(input: &str) -> Option<usize> {
    let mut chars = input.char_indices();

    // Skip the opening brace
    if !matches!(chars.next(), Some((_, '{'))) {
        return None;
    }
    let mut depth = 1;

    for (i, ch) in chars {
        match ch {
            '{' => depth += 1,
            '}' => {
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

/// Analyze general syntax errors when no specific keyword is detected
fn analyze_general_syntax_error(input: &str) -> String {
    // Look for common patterns that suggest specific errors
    if input.contains("switch") && input.contains("case") && !input.contains(';') {
        "Missing semicolon after switch expression".to_string()
    } else if input.contains("var ") && !input.ends_with(';') {
        "Missing semicolon after variable declaration".to_string()
    } else if input.contains("function ") && !input.contains('(') {
        "Missing parentheses in function declaration".to_string()
    } else if input.starts_with('}') {
        "Unexpected closing brace - possible missing opening brace or semicolon".to_string()
    } else {
        "Syntax error".to_string()
    }
}

/// Parse a Haxe file incrementally
pub fn parse_incrementally(_file_name: &str, input: &str) -> IncrementalParseResult {
    let mut parsed_elements = Vec::new();
    let mut errors = Vec::new();
    let mut current_input = input;
    let full_input = input;

    // Helper to calculate line/column
    let calculate_position = |remaining: &str| -> (usize, usize) {
        let consumed = full_input.len() - remaining.len();
        let consumed_str = &full_input[..consumed];
        let line = consumed_str.lines().count();
        let column = consumed_str
            .lines()
            .last()
            .map(|l| l.len() + 1)
            .unwrap_or(1);
        (line, column)
    };

    // Skip leading whitespace
    if let Ok((input, _)) = ws(current_input) {
        current_input = input;
    }

    // Try to parse package declaration
    match package_decl(full_input, current_input) {
        Ok((remaining, pkg)) => {
            parsed_elements.push(ParsedElement::Package(pkg));
            current_input = remaining;
        }
        Err(_) => {
            // Package is optional, continue
        }
    }

    // Parse remaining elements using simple loop with proper error recovery
    while !current_input.trim().is_empty() {
        let input_before = current_input;

        // Skip whitespace
        if let Ok((input, _)) = ws(current_input) {
            current_input = input;
            if current_input.trim().is_empty() {
                break;
            }
        }

        // Try imports, using, and module fields first
        if let Ok((remaining, import)) = import_decl(full_input, current_input) {
            parsed_elements.push(ParsedElement::Import(import));
            current_input = remaining;
            continue;
        }

        if let Ok((remaining, using)) = using_decl(full_input, current_input) {
            parsed_elements.push(ParsedElement::Using(using));
            current_input = remaining;
            continue;
        }

        if let Ok((remaining, module_field)) = module_field(full_input, current_input) {
            parsed_elements.push(ParsedElement::ModuleField(module_field));
            current_input = remaining;
            continue;
        }

        // Try type declarations
        if let Ok((remaining, type_decl)) = type_declaration(full_input, current_input) {
            parsed_elements.push(ParsedElement::TypeDeclaration(type_decl));
            current_input = remaining;
            continue;
        }

        // If we get here, check if this looks like it should have been one of the above
        if current_input.trim_start().starts_with("import") {
            let (line, column) = calculate_position(current_input);
            let detailed_error = analyze_syntax_error(current_input, "import");
            errors.push(ParseError {
                line,
                column,
                message: format!("Invalid import statement: {}", detailed_error),
                remaining_input: current_input[..current_input.len().min(150)].to_string(),
            });
        } else if current_input.trim_start().starts_with("using") {
            let (line, column) = calculate_position(current_input);
            let detailed_error = analyze_syntax_error(current_input, "using");
            errors.push(ParseError {
                line,
                column,
                message: format!("Invalid using statement: {}", detailed_error),
                remaining_input: current_input[..current_input.len().min(150)].to_string(),
            });
        }

        // Handle conditional compilation blocks
        if current_input.starts_with("#if")
            && let Some(end_pos) = current_input.find("#end")
        {
            let block = &current_input[..end_pos + 4];
            parsed_elements.push(ParsedElement::ConditionalBlock(block.to_string()));
            current_input = &current_input[end_pos + 4..];
            continue;
        }

        // Enhanced error analysis and recovery
        let keywords = ["class", "interface", "enum", "typedef", "abstract"];
        let mut found_recovery = false;

        for keyword in &keywords {
            if current_input.trim_start().starts_with(keyword) {
                // Perform detailed syntax analysis for this type declaration
                let detailed_error = analyze_syntax_error(current_input, keyword);
                let (line, column) = calculate_position(current_input);

                let error_message = format!(
                    "Syntax error in {} declaration: {}",
                    keyword, detailed_error
                );

                errors.push(ParseError {
                    line,
                    column,
                    message: error_message,
                    remaining_input: current_input[..current_input.len().min(200)].to_string(),
                });

                // Try to find the next occurrence of any keyword
                let mut next_pos = None;
                let mut _closest_keyword = None;

                for search_keyword in &keywords {
                    if let Some(pos) = current_input[1..].find(search_keyword)
                        && (next_pos.is_none() || pos < next_pos.unwrap())
                    {
                        next_pos = Some(pos + 1); // +1 because we searched from index 1
                        _closest_keyword = Some(search_keyword);
                    }
                }

                if let Some(pos) = next_pos {
                    current_input = &current_input[pos..];
                    found_recovery = true;
                    break;
                } else {
                    // No more keywords found, skip one character and try again
                    if current_input.len() > 1 {
                        current_input = &current_input[1..];
                        found_recovery = true;
                        break;
                    }
                }
            }
        }

        if !found_recovery {
            // No progress made, perform comprehensive error analysis
            let (line, column) = calculate_position(current_input);

            // Analyze what we're looking at to provide better error messages
            let detailed_analysis = analyze_syntax_error(current_input, "unknown");

            let error_message = if !detailed_analysis.is_empty()
                && detailed_analysis != "Syntax error"
            {
                format!("Syntax error: {}", detailed_analysis)
            } else {
                // Fall back to basic analysis
                let trimmed = current_input.trim();
                if trimmed.is_empty() {
                    "Unexpected end of input".to_string()
                } else if trimmed.starts_with('{') {
                    "Unexpected opening brace - missing type declaration".to_string()
                } else if trimmed.starts_with('}') {
                    "Unexpected closing brace - possible missing opening brace".to_string()
                } else if trimmed.starts_with('(') {
                    "Unexpected opening parenthesis - possible incomplete expression".to_string()
                } else if trimmed.starts_with(')') {
                    "Unexpected closing parenthesis - possible missing opening parenthesis"
                        .to_string()
                } else if trimmed.chars().next().unwrap().is_alphabetic() {
                    format!(
                        "Unexpected identifier '{}' - possible missing keyword or operator",
                        trimmed.split_whitespace().next().unwrap_or("")
                    )
                } else {
                    format!(
                        "Unexpected character '{}' at this position",
                        trimmed.chars().next().unwrap_or('?')
                    )
                }
            };

            errors.push(ParseError {
                line,
                column,
                message: error_message,
                remaining_input: current_input[..current_input.len().min(200)].to_string(),
            });
            break;
        }

        // Safety check to prevent infinite loops
        if current_input == input_before {
            break;
        }
    }

    // Check if we consumed all input
    let complete = current_input.trim().is_empty();

    IncrementalParseResult {
        parsed_elements,
        errors,
        complete,
    }
}

/// Parse a specific section of a Haxe file
pub fn parse_section(section: &str, full_context: &str) -> Result<ParsedElement, String> {
    let trimmed = section.trim();

    if trimmed.starts_with("package") {
        package_decl(full_context, trimmed)
            .map(|(_, pkg)| ParsedElement::Package(pkg))
            .map_err(|e| format!("Failed to parse package: {:?}", e))
    } else if trimmed.starts_with("import") {
        import_decl(full_context, trimmed)
            .map(|(_, imp)| ParsedElement::Import(imp))
            .map_err(|e| format!("Failed to parse import: {:?}", e))
    } else if trimmed.starts_with("using") {
        using_decl(full_context, trimmed)
            .map(|(_, use_)| ParsedElement::Using(use_))
            .map_err(|e| format!("Failed to parse using: {:?}", e))
    } else if trimmed.starts_with("class")
        || trimmed.starts_with("interface")
        || trimmed.starts_with("enum")
        || trimmed.starts_with("typedef")
        || trimmed.starts_with("abstract")
        || trimmed.starts_with('@')
    {
        type_declaration(full_context, trimmed)
            .map(|(_, td)| ParsedElement::TypeDeclaration(td))
            .map_err(|e| format!("Failed to parse type declaration: {:?}", e))
    } else {
        Err("Unknown element type".to_string())
    }
}
