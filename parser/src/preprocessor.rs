// ! Haxe Conditional Compilation Preprocessor
//!
//! This module handles Haxe's conditional compilation directives (#if, #else, #elseif, #end)
//! by preprocessing the source code before parsing.
//!
//! Haxe supports conditional compilation based on compiler defines like:
//! - Platform targets: js, jvm, cpp, cs, python, lua, etc.
//! - Features: debug, release, etc.
//! - Custom defines
//!
//! Since Rayzor is a new target, we need to:
//! 1. Define which platform defines are active (rayzor, and maybe sys for system access)
//! 2. Strip out platform-specific code for other targets
//! 3. Keep only code that applies to Rayzor

use std::collections::HashSet;

/// Configuration for conditional compilation
#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    /// Active compiler defines (e.g., "rayzor", "sys", "debug")
    pub defines: HashSet<String>,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        let mut defines = HashSet::new();

        // Rayzor is our target
        defines.insert("rayzor".to_string());

        // We support system access
        defines.insert("sys".to_string());

        // Add debug in debug builds
        #[cfg(debug_assertions)]
        defines.insert("debug".to_string());

        Self { defines }
    }
}

/// Preprocess Haxe source code to handle conditional compilation
///
/// This strips out platform-specific code that doesn't apply to Rayzor.
///
/// # Example
/// ```ignore
/// let source = r#"
/// #if jvm
/// @:runtimeValue
/// #end
/// @:coreType abstract Void {}
/// "#;
///
/// let config = PreprocessorConfig::default();
/// let preprocessed = preprocess(source, &config);
/// // Result: "@:coreType abstract Void {}"
/// // (jvm-specific metadata removed)
/// ```
pub fn preprocess(source: &str, config: &PreprocessorConfig) -> String {
    let mut result = String::with_capacity(source.len());
    let lines: Vec<&str> = source.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim_start();

        // Check if this line contains inline conditionals (e.g., return #if flash ... #else ... #end;)
        if line.contains("#if ") && line.contains("#end") {
            // Process inline conditional
            let processed = process_inline_conditionals(line, config);
            result.push_str(&processed);
            result.push('\n');
            i += 1;
        } else if trimmed.starts_with("#if ") {
            // Parse conditional block
            let condition = trimmed.strip_prefix("#if ").unwrap().trim();
            let (block_lines, end_idx) = extract_conditional_block(&lines, i);

            // Process the conditional block and extract the appropriate branch
            let selected_lines = process_conditional_block(&block_lines, condition, config);

            // Recursively preprocess the selected branch so nested
            // `#if/#end` blocks inside the chosen branch are also resolved.
            // Without this a nested `#if flash` inside an active `#if !eval`
            // would leak through as raw tokens and break downstream parsing.
            let joined: String = selected_lines.join("\n");
            let nested = preprocess(&joined, config);
            result.push_str(&nested);
            if !nested.is_empty() {
                result.push('\n');
            }

            i = end_idx + 1;
        } else if trimmed.starts_with("#else")
            || trimmed.starts_with("#elseif ")
            || trimmed.starts_with("#end")
        {
            // These should be handled by extract_conditional_block
            // If we encounter them here, just skip them
            i += 1;
        } else if trimmed.starts_with("#error") {
            // Skip #error directives - they're compile-time messages
            i += 1;
        } else {
            // Regular line, keep it
            result.push_str(line);
            result.push('\n');
            i += 1;
        }
    }

    // Preserve original trailing newline behavior: if source didn't end
    // with newline, don't add one; if it did, keep it.
    if !source.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }

    result
}

/// Process inline conditional compilation expressions
///
/// Handles patterns like: `return #if flash __global__["isFinite"](i); #else false; #end`
///
/// This extracts the condition, if-branch, and else-branch, evaluates the condition,
/// and returns the appropriate branch.
fn process_inline_conditionals(line: &str, config: &PreprocessorConfig) -> String {
    let mut result = String::new();
    let mut pos = 0;
    let bytes = line.as_bytes();

    while pos < bytes.len() {
        // Look for #if
        if let Some(if_start) = line[pos..].find("#if ") {
            let absolute_if_start = pos + if_start;

            // Add everything before #if
            result.push_str(&line[pos..absolute_if_start]);

            // Find the condition (from "#if " until we find whitespace)
            let cond_start = absolute_if_start + 4; // Skip "#if "

            // Condition can include: identifiers, !, ||, &&, ()
            // Find where the condition ends (first whitespace that ends the condition)
            let mut cond_end = cond_start;
            for (i, ch) in line[cond_start..].char_indices() {
                // Condition can contain: alphanumeric, _, !, |, &, (, ), and spaces within operators
                if ch.is_whitespace() {
                    // Check if this whitespace is followed by more condition tokens
                    let remaining = &line[cond_start + i..];
                    let trimmed = remaining.trim_start();
                    if !trimmed.starts_with("||")
                        && !trimmed.starts_with("&&")
                        && !trimmed.starts_with("!")
                    {
                        cond_end = cond_start + i;
                        break;
                    }
                }
            }
            if cond_end == cond_start {
                cond_end = line.len();
            }

            // Find #else or #end
            let else_pos = line[cond_end..].find("#else").map(|idx| cond_end + idx);
            let end_pos = line[cond_end..].find("#end").map(|idx| cond_end + idx);

            if let Some(end_idx) = end_pos {
                let condition = line[cond_start..cond_end].trim();

                if let Some(else_idx) = else_pos {
                    if else_idx < end_idx {
                        // We have both if and else branches
                        let if_branch = &line[cond_end..else_idx].trim();
                        let else_start = else_idx + 5; // Skip "#else"
                        let else_branch = &line[else_start..end_idx].trim();

                        // Check if branches have trailing semicolons before trimming
                        let if_has_semicolon = if_branch.trim_end().ends_with(';');
                        let else_has_semicolon = else_branch.trim_end().ends_with(';');

                        // Remove leading/trailing semicolons and whitespace
                        let if_content =
                            if_branch.trim_matches(|c: char| c == ';' || c.is_whitespace());
                        let else_content =
                            else_branch.trim_matches(|c: char| c == ';' || c.is_whitespace());

                        // Evaluate condition and add semicolon back if needed
                        if evaluate_condition(condition, config) {
                            result.push_str(if_content);
                            if if_has_semicolon {
                                result.push(';');
                            }
                        } else {
                            result.push_str(else_content);
                            if else_has_semicolon {
                                result.push(';');
                            }
                        }

                        // Move past #end
                        pos = end_idx + 4; // Skip "#end"
                        continue;
                    }
                }

                // No else branch, just if
                let if_branch = &line[cond_end..end_idx].trim();
                let if_has_semicolon = if_branch.trim_end().ends_with(';');
                let if_content = if_branch.trim_matches(|c: char| c == ';' || c.is_whitespace());

                if evaluate_condition(condition, config) {
                    result.push_str(if_content);
                    if if_has_semicolon {
                        result.push(';');
                    }
                }

                pos = end_idx + 4; // Skip "#end"
            } else {
                // No matching #end, just add the rest of the line
                result.push_str(&line[pos..]);
                break;
            }
        } else {
            // No more #if directives
            result.push_str(&line[pos..]);
            break;
        }
    }

    result
}

/// Process a conditional block and return the lines from the active branch
fn process_conditional_block<'a>(
    block_lines: &[&'a str],
    initial_condition: &str,
    config: &PreprocessorConfig,
) -> Vec<&'a str> {
    let mut result = Vec::new();
    let mut i = 1; // Skip the #if line
    let mut condition_met = evaluate_condition(initial_condition, config);
    let mut in_active_branch = condition_met;
    let mut depth = 0; // Track nested #if blocks

    while i < block_lines.len() - 1 {
        // Skip the #end line
        let line = block_lines[i];
        let trimmed = line.trim_start();

        // Handle nested #if blocks
        if trimmed.starts_with("#if ") {
            depth += 1;
            if in_active_branch {
                result.push(line);
            }
            i += 1;
            continue;
        }

        if depth > 0 {
            // We're inside a nested block, just pass through if we're in the active branch
            if trimmed.starts_with("#end") {
                depth -= 1;
            }
            if in_active_branch {
                result.push(line);
            }
            i += 1;
            continue;
        }

        // At depth 0, handle our block's directives
        if trimmed.starts_with("#elseif ") {
            in_active_branch = false;
            if !condition_met {
                let cond = trimmed.strip_prefix("#elseif ").unwrap().trim();
                if evaluate_condition(cond, config) {
                    condition_met = true;
                    in_active_branch = true;
                }
            }
            i += 1;
            continue;
        }

        if trimmed.starts_with("#else") {
            in_active_branch = !condition_met;
            i += 1;
            continue;
        }

        // Regular line - add if we're in the active branch and it's not an #error
        if in_active_branch && !trimmed.starts_with("#error") {
            result.push(line);
        }

        i += 1;
    }

    result
}

/// Extract a conditional block from #if to #end
/// Returns (lines in block, ending line index)
fn extract_conditional_block<'a>(lines: &[&'a str], start_idx: usize) -> (Vec<&'a str>, usize) {
    let mut block = vec![lines[start_idx]];
    let mut depth = 1;
    let mut i = start_idx + 1;

    while i < lines.len() && depth > 0 {
        let line = lines[i];
        let trimmed = line.trim_start();

        // Inline `#if ... #end` on a single line balances out; count
        // BOTH directives by occurrence so a body line like
        // `#if lua return foo; #else return bar; #end` contributes net 0
        // to the nesting depth (one for #if, one for #end). Without
        // this an inline pair stuck depth at 2 and the outer block's
        // matching `#end` never zeroed it, so the recursive
        // preprocessor consumed the rest of the file.
        if trimmed.starts_with("#if ") {
            // Block-form #if at the line head. Count this one normally
            // and ALSO any inline #if/#end pairs after it.
            depth += 1;
            depth +=
                count_inline_directives(trimmed.strip_prefix("#if ").unwrap_or(trimmed), "#if ");
            depth -= count_inline_directives(line, "#end");
        } else if trimmed.starts_with("#end") {
            depth -= 1;
        } else {
            // Plain content line — may still contain inline #if/#end pairs.
            depth += count_inline_directives(line, "#if ");
            depth -= count_inline_directives(line, "#end");
        }

        block.push(line);

        if depth == 0 {
            return (block, i);
        }

        i += 1;
    }

    // Unclosed block
    (block, i - 1)
}

fn count_inline_directives(s: &str, needle: &str) -> i32 {
    let mut count = 0i32;
    let mut rest = s;
    while let Some(idx) = rest.find(needle) {
        count += 1;
        rest = &rest[idx + needle.len()..];
    }
    count
}

/// Evaluate a conditional compilation condition
///
/// Supports:
/// - Simple identifiers: `jvm`, `sys`, `debug`
/// - Boolean OR: `java || cs`
/// - Boolean AND: `sys && debug`
/// - Parentheses: `(java || cs) && sys`
/// - Negation: `!jvm`
fn evaluate_condition(condition: &str, config: &PreprocessorConfig) -> bool {
    // For MVP, we'll implement a simple recursive descent evaluator
    let tokens = tokenize_condition(condition);
    evaluate_tokens(&tokens, config)
}

#[derive(Debug, Clone, PartialEq)]
enum CondToken {
    Ident(String),
    Or,
    And,
    Not,
    LParen,
    RParen,
}

fn tokenize_condition(condition: &str) -> Vec<CondToken> {
    let mut tokens = Vec::new();
    let mut chars = condition.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            ' ' | '\t' => continue,
            '(' => tokens.push(CondToken::LParen),
            ')' => tokens.push(CondToken::RParen),
            '!' => tokens.push(CondToken::Not),
            '|' if chars.peek() == Some(&'|') => {
                chars.next();
                tokens.push(CondToken::Or);
            }
            '&' if chars.peek() == Some(&'&') => {
                chars.next();
                tokens.push(CondToken::And);
            }
            _ if ch.is_alphabetic() || ch == '_' => {
                let mut ident = String::new();
                ident.push(ch);
                while let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        ident.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                tokens.push(CondToken::Ident(ident));
            }
            _ => {}
        }
    }

    tokens
}

fn evaluate_tokens(tokens: &[CondToken], config: &PreprocessorConfig) -> bool {
    // Recursive descent with real precedence: or -> and -> unary -> primary.
    //
    // The previous version tested for `||`/`&&` BEFORE stripping parentheses and
    // split on them without tracking nesting, so `(a && b)` split into
    // `[LParen, a]` and `[b, RParen]`; the first fragment began with `(`, matched
    // no arm, and evaluated false. Every parenthesized condition satisfied by its
    // FIRST term was wrong — `#if (cpp || java)` was false on cpp. It only ever
    // appeared to work when the satisfying term happened to be last.
    let mut pos = 0usize;
    let value = parse_or(tokens, &mut pos, config);
    // Trailing junk means we failed to consume the condition; treat as false
    // rather than silently accepting a partial parse.
    if pos != tokens.len() {
        return false;
    }
    value
}

fn parse_or(tokens: &[CondToken], pos: &mut usize, config: &PreprocessorConfig) -> bool {
    let mut acc = parse_and(tokens, pos, config);
    while matches!(tokens.get(*pos), Some(CondToken::Or)) {
        *pos += 1;
        let rhs = parse_and(tokens, pos, config);
        acc = acc || rhs;
    }
    acc
}

fn parse_and(tokens: &[CondToken], pos: &mut usize, config: &PreprocessorConfig) -> bool {
    let mut acc = parse_unary(tokens, pos, config);
    while matches!(tokens.get(*pos), Some(CondToken::And)) {
        *pos += 1;
        let rhs = parse_unary(tokens, pos, config);
        acc = acc && rhs;
    }
    acc
}

fn parse_unary(tokens: &[CondToken], pos: &mut usize, config: &PreprocessorConfig) -> bool {
    if matches!(tokens.get(*pos), Some(CondToken::Not)) {
        *pos += 1;
        return !parse_unary(tokens, pos, config);
    }
    parse_primary(tokens, pos, config)
}

fn parse_primary(tokens: &[CondToken], pos: &mut usize, config: &PreprocessorConfig) -> bool {
    match tokens.get(*pos) {
        Some(CondToken::Ident(name)) => {
            *pos += 1;
            config.defines.contains(name)
        }
        Some(CondToken::LParen) => {
            *pos += 1;
            let inner = parse_or(tokens, pos, config);
            if matches!(tokens.get(*pos), Some(CondToken::RParen)) {
                *pos += 1;
            }
            inner
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(defines: &[&str]) -> PreprocessorConfig {
        let mut c = PreprocessorConfig::default();
        for d in defines {
            c.defines.insert((*d).to_string());
        }
        c
    }

    /// Parenthesized conditions used to be split on `||`/`&&` BEFORE the parens
    /// were stripped, so the first fragment began with `(` and evaluated false.
    /// Anything satisfied by its FIRST term was wrong; it only looked correct
    /// when the satisfying term happened to be last.
    #[test]
    fn paren_condition_satisfied_by_first_term() {
        let c = cfg(&["cpp"]);
        assert!(evaluate_condition("(cpp || java)", &c));
        assert!(evaluate_condition("(java || cpp)", &c));
        assert!(evaluate_condition("(java || cs || hl || cpp)", &c));
        assert!(!evaluate_condition("(java || cs || hl)", &c));
    }

    #[test]
    fn paren_and_and_negation() {
        let c = cfg(&["rayzor", "llvm"]);
        assert!(evaluate_condition("(rayzor && llvm)", &c));
        assert!(evaluate_condition("(rayzor && !nosuch)", &c));
        assert!(!evaluate_condition("(rayzor && nosuch)", &c));
        assert!(evaluate_condition("(rayzor || nosuch)", &c));
    }

    /// `&&` binds tighter than `||`.
    #[test]
    fn operator_precedence() {
        let c = cfg(&["a", "c"]);
        // a || (b && d) == true, and must not parse as (a || b) && d == false
        assert!(evaluate_condition("a || b && d", &c));
        assert!(evaluate_condition("(a || b) && c", &c));
        assert!(!evaluate_condition("(a && b) || d", &c));
    }

    #[test]
    fn nested_parens() {
        let c = cfg(&["sys", "cpp"]);
        assert!(evaluate_condition("((cpp || java) && sys)", &c));
        assert!(!evaluate_condition("((java || cs) && sys)", &c));
    }

    #[test]
    fn test_simple_condition_true() {
        let mut config = PreprocessorConfig::default();
        config.defines.insert("test".to_string());

        assert!(evaluate_condition("test", &config));
    }

    #[test]
    fn test_simple_condition_false() {
        let config = PreprocessorConfig::default();
        assert!(!evaluate_condition("jvm", &config));
    }

    #[test]
    fn test_or_condition() {
        let mut config = PreprocessorConfig::default();
        config.defines.insert("rayzor".to_string());

        assert!(evaluate_condition("jvm || rayzor", &config));
        assert!(evaluate_condition("rayzor || jvm", &config));
        assert!(!evaluate_condition("jvm || cpp", &config));
    }

    #[test]
    fn test_and_condition() {
        let mut config = PreprocessorConfig::default();
        config.defines.insert("rayzor".to_string());
        config.defines.insert("sys".to_string());

        assert!(evaluate_condition("rayzor && sys", &config));
        assert!(!evaluate_condition("rayzor && jvm", &config));
    }

    #[test]
    fn test_preprocess_simple() {
        let source = r#"
#if jvm
@:runtimeValue
#end
@:coreType abstract Void {}
"#;

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        // jvm block should be removed
        assert!(!result.contains("@:runtimeValue"));
        assert!(result.contains("@:coreType abstract Void"));
    }

    #[test]
    fn test_preprocess_keeps_rayzor_code() {
        let source = r#"
#if rayzor
var x = 42;
#end
"#;

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        // rayzor block should be kept
        assert!(result.contains("var x = 42"));
    }

    #[test]
    fn test_inline_conditional_else_branch() {
        let source = r#"return #if flash __global__["isFinite"](i); #else false; #end"#;

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        // flash is not defined, so else branch should be kept
        assert!(result.contains("return false"));
        assert!(!result.contains("__global__"));
        assert!(!result.contains("#if"));
        assert!(!result.contains("#else"));
        assert!(!result.contains("#end"));
    }

    #[test]
    fn test_inline_conditional_if_branch() {
        let mut config = PreprocessorConfig::default();
        config.defines.insert("flash".to_string());

        let source = r#"return #if flash __global__["isFinite"](i); #else false; #end"#;
        let result = preprocess(source, &config);

        println!("Result: {}", result);

        // flash is defined, so if branch should be kept
        assert!(
            result.contains(r#"__global__["isFinite"](i)"#),
            "Result was: {}",
            result
        );
        assert!(!result.contains("false"), "Result was: {}", result);
        assert!(!result.contains("#if"), "Result was: {}", result);
    }

    #[test]
    fn test_inline_conditional_in_function() {
        let source = r#"
Math.isFinite = function(i) {
    return #if flash __global__["isFinite"](i); #else false; #end
};
"#;

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        // Should have the else branch
        assert!(result.contains("return false"));
        assert!(!result.contains("__global__"));
    }

    #[test]
    fn test_block_conditional_inside_untyped() {
        let source = r#"
untyped {
    #if flash
    NaN = __global__["Number"].NaN;
    #else
    Math.NaN = Number["NaN"];
    #end
}
"#;

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        println!("Result: {}", result);

        // flash is not defined, so else branch should be kept
        assert!(result.contains("Math.NaN"));
        assert!(!result.contains("__global__"));
        assert!(!result.contains("#if"));
    }

    #[test]
    fn test_inline_conditional_in_modifier_position() {
        // Pattern from UInt.hx: private static #if !js inline #end function gt(...)
        // Note: UInt.hx has a TAB before "private"
        let source = "\tprivate static #if !js inline #end function gt(a:UInt, b:UInt):Bool {";

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        println!("Result: '{}'", result);

        // !js is true (js is not defined), so inline should be kept
        assert!(result.contains("inline"), "Result was: '{}'", result);
        assert!(
            result.contains("inline function"),
            "Result was: '{}'",
            result
        );
        assert!(!result.contains("#if"), "Result was: '{}'", result);
    }

    #[test]
    fn test_inline_conditional_in_modifier_position_false() {
        // Same pattern but with js defined
        let source = r#"private static #if !js inline #end function gt(a:UInt, b:UInt):Bool {"#;

        let mut config = PreprocessorConfig::default();
        config.defines.insert("js".to_string());
        let result = preprocess(source, &config);

        println!("Result: {}", result);

        // !js is false (js IS defined), so inline should be removed
        assert!(!result.contains("inline"), "Result was: {}", result);
        assert!(
            result.contains("private static  function"),
            "Result was: {}",
            result
        );
        assert!(!result.contains("#if"), "Result was: {}", result);
    }

    #[test]
    fn test_block_conditional_with_comment() {
        // Block conditional with doc comment inside
        let source = r#"
#if flash
/**
    Flash implementation
**/
abstract UInt {}
#else
/**
    Generic implementation
**/
abstract UInt(Int) {}
#end
"#;

        let config = PreprocessorConfig::default();
        let result = preprocess(source, &config);

        println!("Result: {}", result);

        // flash is not defined, so else branch should be kept
        assert!(
            result.contains("Generic implementation"),
            "Result was: {}",
            result
        );
        assert!(
            !result.contains("Flash implementation"),
            "Result was: {}",
            result
        );
        assert!(
            result.contains("abstract UInt(Int)"),
            "Result was: {}",
            result
        );
    }

    #[test]
    fn test_stringtools_complex_condition() {
        // Pattern from StringTools.hx: complex OR condition with parentheses
        let source = r#"public static #if (cs || java || python || (js && js_es >= 6)) inline #end function startsWith(s:String, start:String):Bool {"#;

        let config = PreprocessorConfig::default(); // rayzor defined, none of cs/java/python/js
        let result = preprocess(source, &config);

        println!("Result: '{}'", result);

        // None of cs, java, python, js are defined, so inline should be REMOVED
        assert!(
            !result.contains("inline"),
            "inline should be removed. Result was: '{}'",
            result
        );
        assert!(
            result.contains("public static  function startsWith"),
            "Result was: '{}'",
            result
        );
        assert!(result.contains("s:String"), "Result was: '{}'", result);
        assert!(!result.contains("#if"), "Result was: '{}'", result);
    }
}
