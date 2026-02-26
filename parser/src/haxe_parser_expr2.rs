//! Expression parsing for Haxe (continuation)
//!
//! This module contains the remaining expression parsers

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::char,
    combinator::{map, opt},
    error::context,
    multi::{many0, many1, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, tuple},
    IResult, Parser,
};

use crate::custom_error::ContextualError;
use crate::haxe_ast::*;
use crate::haxe_parser::{
    compiler_specific_identifier, identifier, keyword, position, symbol, ws, PResult,
};
use crate::haxe_parser_expr::{expression, postfix_expr};
use crate::haxe_parser_types::type_expr;

// Simple expressions

pub fn identifier_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    // Skip leading whitespace before recording start so the identifier span
    // begins at the actual identifier character, not at a preceding newline.
    let (input, _) = ws(input)?;
    let start = position(full, input);
    let (input, id) = identifier(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Ident(id),
            span: Span::new(start, end),
        },
    ))
}

/// Parse keyword as identifier (for contexts where keywords can be used as identifiers)
#[allow(dead_code)]
fn keyword_as_identifier(input: &str) -> PResult<String> {
    let (input, _) = ws(input)?;
    alt((
        map(keyword("macro"), |_| "macro".to_string()),
        // Add other keywords as needed
    ))
    .parse(input)
}

/// Parse macro expression: `macro expr`
pub fn macro_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("macro").parse(input)?;
    // Parse at unary expression level to get proper precedence
    let (input, expr) = crate::haxe_parser_expr::unary_expr(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Macro(Box::new(expr)),
            span: Span::new(start, end),
        },
    ))
}

/// Parse dollar expression: either reification `$expr` or dollar identifier `$type`, `$v{...}`, etc.
pub fn reify_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = ws(input)?;
    let (input, _) = char('$')(input)?;

    // Try to parse as dollar identifier first
    if let Ok((rest, dollar_ident)) = dollar_identifier(full, input) {
        return Ok((rest, dollar_ident));
    }

    // Otherwise, parse as macro reification
    let (input, expr) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Reify(Box::new(expr)),
            span: Span::new(start, end),
        },
    ))
}

/// Parse dollar identifier: `$type`, `$v{...}`, `$i{...}`, etc.
fn dollar_identifier<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);

    // Try `$type` first — a standalone keyword that doesn't require braces.
    // Must check that 'type' is a complete word (not prefix of 'typeCheck' etc.).
    if let Ok((rest, _)) = tag::<&str, &str, ContextualError<&str>>("type")(input) {
        if rest.is_empty() || !rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            let end = position(full, rest);
            return Ok((
                rest,
                Expr {
                    kind: ExprKind::DollarIdent {
                        name: "type".to_string(),
                        arg: None,
                    },
                    span: Span::new(start, end),
                },
            ));
        }
    }

    // Macro reification identifiers: $v{...}, $i{...}, $a{...}, $b{...}, $p{...}, $e{...}
    // These MUST be followed by `{` to distinguish from regular $identifier references.
    let (input, name) = alt((
        map(tag("v"), |_| "v".to_string()),
        map(tag("i"), |_| "i".to_string()),
        map(tag("a"), |_| "a".to_string()),
        map(tag("b"), |_| "b".to_string()),
        map(tag("p"), |_| "p".to_string()),
        map(tag("e"), |_| "e".to_string()),
    ))
    .parse(input)?;

    // Reification identifiers require braces: $e{expr}, $v{value}, etc.
    let (input, _) = symbol("{").parse(input)?;
    let (input, expr) = expression(full, input)?;
    let (input, _) = symbol("}")(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::DollarIdent {
                name,
                arg: Some(Box::new(expr)),
            },
            span: Span::new(start, end),
        },
    ))
}

pub fn this_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("this").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::This,
            span: Span::new(start, end),
        },
    ))
}

pub fn super_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("super").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Super,
            span: Span::new(start, end),
        },
    ))
}

pub fn null_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("null").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Null,
            span: Span::new(start, end),
        },
    ))
}

/// Parse compiler-specific code block: `__c__("code {0} {1}", arg0, arg1)`
pub fn compiler_specific_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, target) = compiler_specific_identifier(input)?;

    // Parse: (code_expr, optional_args...)
    let (input, _) = symbol("(").parse(input)?;
    let (input, code) = expression(full, input)?;

    // Parse optional comma-separated additional arguments
    let mut args = Vec::new();
    let mut input = input;
    loop {
        // Try to parse a comma followed by an expression
        let comma_result: PResult<'_, &str> = symbol(",").parse(input);
        match comma_result {
            Ok((rest, _)) => match expression(full, rest) {
                Ok((rest2, arg_expr)) => {
                    args.push(arg_expr);
                    input = rest2;
                }
                Err(_) => break,
            },
            Err(_) => break,
        }
    }

    let (input, _) = symbol(")").parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::CompilerSpecific {
                target,
                code: Box::new(code),
                args,
            },
            span: Span::new(start, end),
        },
    ))
}

// Constructor and cast expressions

pub fn new_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("new").parse(input)?;

    // Parse type path
    let (input, path_parts) = separated_list1(symbol("."), identifier).parse(input)?;

    // Split into package and name
    let (package, name) = if path_parts.len() == 1 {
        (vec![], path_parts[0].clone())
    } else {
        let mut parts = path_parts;
        let name = parts.pop().unwrap();
        (parts, name)
    };

    let type_path = TypePath {
        package,
        name,
        sub: None,
    };

    // Type parameters
    let (input, params) = opt(delimited(
        symbol("<"),
        separated_list1(symbol(","), |i| type_expr(full, i)),
        symbol(">"),
    ))
    .parse(input)?;

    // Arguments
    let (input, _) = symbol("(").parse(input)?;
    let (input, args) = separated_list0(symbol(","), |i| expression(full, i)).parse(input)?;
    let (input, _) = symbol(")").parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::New {
                type_path,
                params: params.unwrap_or_default(),
                args,
            },
            span: Span::new(start, end),
        },
    ))
}

pub fn cast_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("cast").parse(input)?;

    alt((
        // cast(expr : Type) - type annotation style in parentheses
        map(
            delimited(
                symbol("("),
                pair(
                    |i| expression(full, i),
                    opt(preceded(symbol(":"), |i| type_expr(full, i))),
                ),
                symbol(")"),
            ),
            move |(expr, type_hint)| {
                let end = position(full, input);
                Expr {
                    kind: ExprKind::Cast {
                        expr: Box::new(expr),
                        type_hint,
                    },
                    span: Span::new(start, end),
                }
            },
        ),
        // cast(expr, Type) - comma style
        map(
            delimited(
                symbol("("),
                pair(
                    |i| expression(full, i),
                    opt(preceded(symbol(","), |i| type_expr(full, i))),
                ),
                symbol(")"),
            ),
            move |(expr, type_hint)| {
                let end = position(full, input);
                Expr {
                    kind: ExprKind::Cast {
                        expr: Box::new(expr),
                        type_hint,
                    },
                    span: Span::new(start, end),
                }
            },
        ),
        // cast expr - unsafe cast without type annotation (no parentheses)
        // Note: type-annotated casts require parentheses: cast(expr, Type) or cast(expr : Type)
        // The `: Type` form is NOT supported here because it's ambiguous with the ternary operator's `:`
        map(
            |i| postfix_expr(full, i), // Use postfix_expr to avoid infinite recursion with unary
            move |expr| {
                let end = position(full, input);
                Expr {
                    kind: ExprKind::Cast {
                        expr: Box::new(expr),
                        type_hint: None,
                    },
                    span: Span::new(start, end),
                }
            },
        ),
    ))
    .parse(input)
}

/// Parse untyped expression: `untyped expr`
pub fn untyped_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("untyped").parse(input)?;
    let (input, expr) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Untyped(Box::new(expr)),
            span: Span::new(start, end),
        },
    ))
}

/// Parse inline expression: `inline expr` - forces inlining at call site
pub fn inline_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("inline").parse(input)?;
    // Parse at unary expression level to get proper precedence
    let (input, expr) = crate::haxe_parser_expr::unary_expr(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Inline(Box::new(expr)),
            span: Span::new(start, end),
        },
    ))
}

/// Parse inline preprocessor conditional: `#if cond expr1 #else expr2 #end`
/// For Rayzor, we skip the #if branch and return the #else branch expression
pub fn inline_preprocessor_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    use nom::bytes::complete::{tag, take_while};

    // Skip any leading whitespace (but not newlines)
    let (input, _) = take_while(|c: char| c.is_whitespace() && c != '\n')(input)?;

    // Must start with #if
    let (input, _) = tag("#if")(input)?;

    // Skip whitespace
    let (mut input, _) = take_while(|c: char| c.is_whitespace() && c != '\n')(input)?;

    // Skip condition - can be: identifier, !identifier, (expr)
    if input.starts_with('(') {
        // Parenthesized expression - find matching closing paren
        let mut depth = 1;
        let mut pos = 1;
        for (i, c) in input[1..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        pos = i + 2;
                        break;
                    }
                }
                _ => {}
            }
        }
        input = &input[pos..];
    } else {
        // Simple identifier or !identifier
        if input.starts_with('!') {
            input = &input[1..];
        }
        let (rest, _) = take_while(|c: char| c.is_alphanumeric() || c == '_')(input)?;
        input = rest;
    }

    // Skip whitespace
    let (input, _) = take_while(|c: char| c.is_whitespace())(input)?;

    // Now we need to find #else, handling nested #if...#end
    let mut current = input;
    let mut depth = 1;

    while depth > 0 && !current.is_empty() {
        if current.starts_with("#if")
            && (current.len() == 3
                || !current
                    .chars()
                    .nth(3)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            depth += 1;
            current = &current[3..];
        } else if current.starts_with("#else")
            && depth == 1
            && (current.len() == 5
                || !current
                    .chars()
                    .nth(5)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            // Found #else at our level
            current = &current[5..];
            // Skip whitespace
            while !current.is_empty() && current.starts_with(char::is_whitespace) {
                current = &current[1..];
            }
            break;
        } else if current.starts_with("#end")
            && (current.len() == 4
                || !current
                    .chars()
                    .nth(4)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            depth -= 1;
            if depth == 0 {
                // No #else found, just skip to end
                current = &current[4..];
                break;
            }
            current = &current[4..];
        } else {
            let mut chars = current.chars();
            chars.next();
            current = chars.as_str();
        }
    }

    // The #else branch content is a block - parse everything up to #end
    // This block can contain multiple statements with semicolons
    // Example: return #if flash x; #else y; #end
    // The #else branch is "y;" which includes the semicolon

    let _start_pos = position(full, current);
    let else_start = current;

    // Find the matching #end for this conditional
    let mut current_scan = current;
    let mut depth = 1;
    let mut _end_offset = 0;

    while depth > 0 && !current_scan.is_empty() {
        if current_scan.starts_with("#if") {
            depth += 1;
            current_scan = &current_scan[3..];
            _end_offset += 3;
        } else if current_scan.starts_with("#end")
            && (current_scan.len() == 4
                || !current_scan
                    .chars()
                    .nth(4)
                    .map(|c| c.is_alphanumeric())
                    .unwrap_or(false))
        {
            depth -= 1;
            if depth == 0 {
                // Found the matching #end
                break;
            }
            current_scan = &current_scan[4..];
            _end_offset += 4;
        } else {
            let ch = current_scan.chars().next().unwrap();
            current_scan = &current_scan[ch.len_utf8()..];
            _end_offset += ch.len_utf8();
        }
    }

    // Parse the #else branch content as an expression
    let (after_expr, expr) = expression(full, else_start)?;

    // Skip whitespace and check for semicolon
    let (mut remaining, _) = take_while(|c: char| c.is_whitespace() && c != '\n')(after_expr)?;
    let has_semicolon = remaining.starts_with(';');
    if has_semicolon {
        remaining = &remaining[1..];
    }
    let (remaining, _) = take_while(|c: char| c.is_whitespace())(remaining)?;

    // Now we should be at #end
    if !remaining.starts_with("#end") {
        return Err(nom::Err::Error(ContextualError::new(
            remaining,
            nom::error::ErrorKind::Tag,
        )));
    }
    // Skip "#end"
    let mut input = &remaining[4..];

    // HACK: If the #else branch had a semicolon, we need to provide one to the caller
    // because both #if and #else branches should have the same syntax.
    // We do this by creating a synthetic input string with a semicolon prepended.
    if has_semicolon {
        // Create a string "; " + input and use Box::leak to get a 'static lifetime
        let synthetic = Box::leak(format!(";{}", input).into_boxed_str());
        input = synthetic;
    }

    Ok((input, expr))
}

// Collection literals

pub fn array_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = symbol("[").parse(input)?;

    // Check for comprehensions
    if let Ok((_, _)) = keyword("for").parse(input) {
        return array_comprehension(full, start, input);
    }

    // Check for map literal (has =>)
    let is_map = {
        let mut check_input = input;
        let mut depth = 0;
        let mut found_arrow = false;

        while !check_input.is_empty() && !found_arrow {
            if check_input.starts_with("=>") {
                found_arrow = true;
            } else if check_input.starts_with('[') {
                depth += 1;
            } else if check_input.starts_with(']') {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            check_input = &check_input[1..];
        }

        found_arrow
    };

    if is_map {
        map_literal(full, start, input)
    } else {
        array_literal(full, start, input)
    }
}

fn array_literal<'a>(full: &'a str, start: usize, input: &'a str) -> PResult<'a, Expr> {
    let (input, elements) = separated_list0(symbol(","), |i| expression(full, i)).parse(input)?;
    let (input, _) = opt(symbol(",")).parse(input)?; // Trailing comma
    let (input, _) = symbol("]").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Array(elements),
            span: Span::new(start, end),
        },
    ))
}

fn map_literal<'a>(full: &'a str, start: usize, input: &'a str) -> PResult<'a, Expr> {
    let (input, pairs) = separated_list0(
        symbol(","),
        map(
            tuple((
                |i| expression(full, i),
                symbol("=>"),
                |i| expression(full, i),
            )),
            |(k, _, v)| (k, v),
        ),
    )
    .parse(input)?;

    let (input, _) = opt(symbol(",")).parse(input)?;
    let (input, _) = symbol("]").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Map(pairs),
            span: Span::new(start, end),
        },
    ))
}

fn array_comprehension<'a>(full: &'a str, start: usize, input: &'a str) -> PResult<'a, Expr> {
    let (input, for_parts) = many1(|i| comprehension_for(full, i)).parse(input)?;

    // Check for map comprehension (has =>)
    let check_result: IResult<_, _, ContextualError<&str>> = tuple((
        |i| expression(full, i),
        symbol("=>"),
        |i| expression(full, i),
    ))
    .parse(input);

    let (input, kind) = if let Ok((rest, (key, _, value))) = check_result {
        (
            rest,
            ExprKind::MapComprehension {
                for_parts,
                key: Box::new(key),
                value: Box::new(value),
            },
        )
    } else {
        let (rest, expr) = expression(full, input)?;
        (
            rest,
            ExprKind::ArrayComprehension {
                for_parts,
                expr: Box::new(expr),
            },
        )
    };

    let (input, _) = symbol("]").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind,
            span: Span::new(start, end),
        },
    ))
}

fn comprehension_for<'a>(full: &'a str, input: &'a str) -> PResult<'a, ComprehensionFor> {
    let start = position(full, input);
    let (input, _) = keyword("for").parse(input)?;
    let (input, _) = symbol("(").parse(input)?;

    // Try to parse key => value pattern first
    let (input, var, key_var) = match parse_key_value_pattern(input) {
        Ok((rest, (key, value))) => (rest, value, Some(key)),
        Err(_) => {
            // Fall back to simple variable
            let (rest, var) = identifier(input)?;
            (rest, var, None)
        }
    };

    let (input, _) = keyword("in").parse(input)?;
    let (input, iter) = expression(full, input)?;
    let (input, _) = symbol(")").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        ComprehensionFor {
            var,
            key_var,
            iter,
            span: Span::new(start, end),
        },
    ))
}

pub fn object_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = symbol("{").parse(input)?;
    let (input, fields) = separated_list0(symbol(","), |i| object_field(full, i)).parse(input)?;
    let (input, _) = opt(symbol(",")).parse(input)?;
    let (input, _) = symbol("}").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Object(fields),
            span: Span::new(start, end),
        },
    ))
}

fn object_field<'a>(full: &'a str, input: &'a str) -> PResult<'a, ObjectField> {
    let start = position(full, input);
    let (input, name) = identifier(input)?;
    let (input, _) = symbol(":").parse(input)?;
    let (input, expr) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        ObjectField {
            name,
            expr,
            span: Span::new(start, end),
        },
    ))
}

// Control flow expressions

pub fn block_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = context("[E0003] expected '{' to start block", symbol("{")).parse(input)?;

    let (input, elements) = context("[E0004] expected block contents | help: provide statements or expressions inside the block", many0(|i|block_element(full, i))).parse(input)?;

    let (input, _) = context(
        "[E0005] expected '}' to close block | help: blocks must be properly closed with '}'",
        symbol("}"),
    )
    .parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Block(elements),
            span: Span::new(start, end),
        },
    ))
}

fn block_element<'a>(full: &'a str, input: &'a str) -> PResult<'a, BlockElement> {
    alt((
        // Conditional compilation inside block
        |i| {
            let peek_result: Result<_, nom::Err<nom::error::Error<_>>> =
                nom::combinator::peek(tag("#if")).parse(i);
            if peek_result.is_ok() {
                map(
                    |i| crate::haxe_parser::conditional_compilation(full, i, block_element),
                    BlockElement::Conditional,
                )
                .parse(i)
            } else {
                Err(nom::Err::Error(ContextualError::new(
                    i,
                    nom::error::ErrorKind::Tag,
                )))
            }
        },
        // Import/using inside block
        map(
            |i| crate::haxe_parser::import_decl(full, i),
            BlockElement::Import,
        ),
        map(
            |i| crate::haxe_parser::using_decl(full, i),
            BlockElement::Using,
        ),
        // Expression - semicolon is optional for control flow statements
        map(
            |i| {
                let (input, expr) = expression(full, i)?;
                // Check if this is a control flow statement that doesn't need semicolon
                // Helper to check if an expression is a brace-terminated control flow
                fn is_brace_terminated(kind: &ExprKind) -> bool {
                    matches!(
                        kind,
                        ExprKind::If { .. }
                            | ExprKind::Switch { .. }
                            | ExprKind::For { .. }
                            | ExprKind::While { .. }
                            | ExprKind::DoWhile { .. }
                            | ExprKind::Try { .. }
                            | ExprKind::Block(_)
                            | ExprKind::Function(_)
                    )
                }

                let needs_semicolon = match &expr.kind {
                    // Direct control flow statements don't need semicolon
                    k if is_brace_terminated(k) => false,
                    // Wrapper expressions with brace-terminated inner don't need semicolon
                    // e.g., return switch { }, throw if (cond) { } else { },
                    // untyped { ... }, macro { ... }
                    ExprKind::Return(Some(inner)) if is_brace_terminated(&inner.kind) => false,
                    ExprKind::Throw(inner) if is_brace_terminated(&inner.kind) => false,
                    ExprKind::Untyped(inner) if is_brace_terminated(&inner.kind) => false,
                    ExprKind::Macro(inner) if is_brace_terminated(&inner.kind) => false,
                    _ => true,
                };

                if needs_semicolon {
                    // Provide specific context based on the expression type
                    let error_msg = match &expr.kind {
                        ExprKind::Var { .. } => "expected ';' after variable declaration",
                        ExprKind::Final { .. } => "expected ';' after final variable declaration",
                        ExprKind::Assign { .. } => "expected ';' after assignment",
                        ExprKind::Call { .. } => "expected ';' after function call",
                        ExprKind::Return { .. } => "expected ';' after return statement",
                        ExprKind::Break => "expected ';' after break statement",
                        ExprKind::Continue => "expected ';' after continue statement",
                        ExprKind::Throw { .. } => "expected ';' after throw statement",
                        _ => "expected ';' after statement",
                    };
                    let (input, _) = context(error_msg, symbol(";")).parse(input)?;
                    Ok((input, expr))
                } else {
                    // Optional semicolon for control flow statements
                    let (input, _) = opt(symbol(";")).parse(input)?;
                    Ok((input, expr))
                }
            },
            BlockElement::Expr,
        ),
    ))
    .parse(input)
}

pub fn if_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("if").parse(input)?;
    let (input, _) = symbol("(").parse(input)?;
    let (input, cond) = expression(full, input)?;
    let (input, _) = symbol(")").parse(input)?;
    let (input, then_branch) = expression(full, input)?;

    // In Haxe, if-expressions can have an optional semicolon after the then-branch
    // before the else clause: if (cond) expr; else expr2
    // We consume the optional semicolon here before checking for else
    let (input, _) = opt(symbol(";")).parse(input)?;

    let (input, else_branch) =
        opt(preceded(keyword("else"), |i| expression(full, i))).parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: else_branch.map(Box::new),
            },
            span: Span::new(start, end),
        },
    ))
}

pub fn switch_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("switch").parse(input)?;
    let (input, _) = context("[E0020] expected '(' after 'switch' | help: switch expression must be enclosed in parentheses: switch(expr)", symbol("(")).parse(input)?;
    let (input, expr) = context(
        "[E0021] expected expression inside switch parentheses",
        |i| expression(full, i),
    )
    .parse(input)?;

    // Handle type check syntax: switch (expr : Type) { ... }
    let (input, expr) = if let Ok((rest, _)) = symbol(":").parse(input) {
        let (rest, type_hint) = type_expr(full, rest)?;
        let tc_end = position(full, rest);
        (
            rest,
            Expr {
                kind: ExprKind::TypeCheck {
                    expr: Box::new(expr),
                    type_hint,
                },
                span: Span::new(start, tc_end),
            },
        )
    } else {
        (input, expr)
    };

    let (input, _) =
        context("[E0021] expected ')' after switch expression", symbol(")")).parse(input)?;
    let (input, _) = context(
        "[E0022] expected '{' to start switch body | help: switch body must be enclosed in braces",
        symbol("{"),
    )
    .parse(input)?;

    // Don't wrap in context here - let inner errors bubble up with their contexts
    let (input, cases) = many0(|i| case(full, i)).parse(input)?;

    let (input, default) = opt(preceded(
        keyword("default"),
        preceded(
            context("[E0027] expected ':' after 'default'", symbol(":")),
            context("[E0028] expected default case body", |i| {
                parse_case_body(full, i)
            }),
        ),
    ))
    .parse(input)?;

    let (input, _) =
        context("[E0023] expected '}' to close switch body", symbol("}")).parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Switch {
                expr: Box::new(expr),
                cases,
                default: default.map(Box::new),
            },
            span: Span::new(start, end),
        },
    ))
}

fn case<'a>(full: &'a str, input: &'a str) -> PResult<'a, Case> {
    let start = position(full, input);
    let (input, _) = keyword("case").parse(input)?;
    // Haxe allows patterns to be separated by either '|' or ','
    // e.g., case "a" | "b": or case "a", "b":
    let (input, patterns) = context(
        "[E0025] expected pattern(s) after 'case' | help: provide a pattern to match against",
        separated_list1(alt((symbol("|"), symbol(","))), |i| pattern(full, i)),
    )
    .parse(input)?;

    // Try to parse guard - if we see 'if', parse the guard expression.
    // Supports both `case n if (expr):` and `case n if expr:` forms.
    let (input, guard) = {
        // First, check if there's an 'if' keyword
        if let Ok((after_if, _)) = keyword("if").parse(input) {
            // We found 'if', so we MUST have a guard expression.
            // Try parenthesized form first: if (expr)
            if let Ok((input, guard_expr)) =
                delimited(symbol("("), |i| expression(full, i), symbol(")")).parse(after_if)
            {
                (input, Some(guard_expr))
            } else {
                // No parentheses — parse a bare expression up to the ':'
                let (input, guard_expr) = context(
                    "[E0042] expected guard expression | help: provide a boolean expression for the guard condition",
                    |i| expression(full, i)
                ).parse(after_if)?;
                (input, Some(guard_expr))
            }
        } else {
            // No 'if' keyword found, no guard
            (input, None)
        }
    };

    let (input, _) = context(
        "[E0024] expected ':' after case pattern | help: case patterns must be followed by a colon",
        symbol(":"),
    )
    .parse(input)?;

    // Parse case body as a sequence of statements until next case/default/}
    let (input, body) = context(
        "[E0026] expected case body | help: provide an expression or block for the case",
        |i| parse_case_body(full, i),
    )
    .parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Case {
            patterns,
            guard,
            body,
            span: Span::new(start, end),
        },
    ))
}

fn parse_case_body<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let mut statements = Vec::new();
    let mut current_input = input;

    // Parse statements until we hit a case, default, or closing brace
    loop {
        // Skip whitespace
        let (input, _) = ws(current_input)?;
        current_input = input;

        // Check for terminating conditions
        if current_input.is_empty() || current_input.starts_with("}") {
            break;
        }

        // Check for case/default keywords
        let trimmed = current_input.trim_start();
        if trimmed.starts_with("case ")
            || trimmed.starts_with("default:")
            || trimmed.starts_with("default ")
        {
            break;
        }

        // Parse next statement
        let (input, expr) = expression(full, current_input)?;
        statements.push(BlockElement::Expr(expr));

        // Check if this is a control flow statement that doesn't need semicolon
        let needs_semicolon = match &statements.last().unwrap() {
            BlockElement::Expr(expr) => !matches!(
                &expr.kind,
                ExprKind::If { .. }
                    | ExprKind::Switch { .. }
                    | ExprKind::For { .. }
                    | ExprKind::While { .. }
                    | ExprKind::DoWhile { .. }
                    | ExprKind::Try { .. }
                    | ExprKind::Block(_)
            ),
            _ => true,
        };

        if needs_semicolon {
            let (input, _) = symbol(";")(input)?;
            current_input = input;
        } else {
            // Optional semicolon for control flow statements
            let (input, _) = opt(symbol(";")).parse(input)?;
            current_input = input;
        }
    }

    let end = position(full, current_input);

    // If we have no statements, return an empty block
    if statements.is_empty() {
        Ok((
            current_input,
            Expr {
                kind: ExprKind::Block(vec![]),
                span: Span::new(start, end),
            },
        ))
    } else if statements.len() == 1 {
        // Single statement - return it directly
        let first_stmt = statements.into_iter().next().unwrap();
        if let BlockElement::Expr(expr) = first_stmt {
            Ok((current_input, expr))
        } else {
            Ok((
                current_input,
                Expr {
                    kind: ExprKind::Block(vec![first_stmt]),
                    span: Span::new(start, end),
                },
            ))
        }
    } else {
        // Multiple statements - wrap in a block
        Ok((
            current_input,
            Expr {
                kind: ExprKind::Block(statements),
                span: Span::new(start, end),
            },
        ))
    }
}

fn pattern<'a>(full: &'a str, input: &'a str) -> PResult<'a, Pattern> {
    alt((
        // Null pattern
        map(keyword("null"), |_| Pattern::Null),
        // Type pattern: (var:Type)
        |input| {
            if let Ok((input, _)) = symbol("(").parse(input) {
                if let Ok((rest, var)) = identifier(input) {
                    if let Ok((rest, _)) = symbol(":").parse(rest) {
                        if let Ok((rest, type_hint)) = type_expr(full, rest) {
                            if let Ok((rest, _)) = symbol(")").parse(rest) {
                                return Ok((rest, Pattern::Type { var, type_hint }));
                            }
                        }
                    }
                }
            }
            Err(nom::Err::Error(ContextualError::new(
                input,
                nom::error::ErrorKind::Tag,
            )))
        },
        // Object pattern: {x: 0, y: 0}
        |input| {
            if let Ok((input, _)) = symbol("{").parse(input) {
                let (input, fields) = separated_list0(symbol(","), |i| {
                    let (i, field_name) = identifier(i)?;
                    let (i, _) = symbol(":").parse(i)?;
                    let (i, field_pattern) = pattern(full, i)?;
                    Ok((i, (field_name, field_pattern)))
                })
                .parse(input)?;
                let (input, _) = symbol("}").parse(input)?;
                Ok((input, Pattern::Object { fields }))
            } else {
                Err(nom::Err::Error(ContextualError::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        },
        // Array pattern with possible rest
        |input| {
            if let Ok((input, _)) = symbol("[").parse(input) {
                let mut elements = Vec::new();
                let mut rest = None;
                let mut current_input = input;

                loop {
                    // Skip whitespace
                    let (input, _) = ws(current_input)?;
                    current_input = input;

                    // Check for closing bracket
                    if let Ok((input, _)) = symbol("]").parse(current_input) {
                        current_input = input;
                        break;
                    }

                    // Check for rest pattern
                    if let Ok((input, _)) = symbol("...").parse(current_input) {
                        let (input, rest_var) = identifier(input)?;
                        rest = Some(rest_var);
                        current_input = input;

                        // Skip optional comma
                        if let Ok((input, _)) = symbol(",").parse(current_input) {
                            current_input = input;
                        }

                        // Must be at end
                        let (input, _) = symbol("]").parse(current_input)?;
                        current_input = input;
                        break;
                    }

                    // Parse regular pattern
                    let (input, pat) = pattern(full, current_input)?;
                    elements.push(pat);
                    current_input = input;

                    // Check for comma
                    if let Ok((input, _)) = symbol(",").parse(current_input) {
                        current_input = input;
                    } else {
                        // No comma, must be at end
                        let (input, _) = symbol("]").parse(current_input)?;
                        current_input = input;
                        break;
                    }
                }

                if rest.is_some() {
                    Ok((current_input, Pattern::ArrayRest { elements, rest }))
                } else {
                    Ok((current_input, Pattern::Array(elements)))
                }
            } else {
                Err(nom::Err::Error(ContextualError::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        },
        // Try extractor pattern first since it's more specific (has =>)
        |input| {
            // Try to parse a postfix expression (stops before binary operators to avoid consuming ':')
            match crate::haxe_parser_expr::postfix_expr(full, input) {
                Ok((rest, expr)) => {
                    // Check if followed by =>
                    match symbol("=>").parse(rest) {
                        Ok((rest, _)) => {
                            // Parse the value expression (also postfix level)
                            match crate::haxe_parser_expr::postfix_expr(full, rest) {
                                Ok((rest, value)) => Ok((
                                    rest,
                                    Pattern::Extractor {
                                        expr: Box::new(expr),
                                        value: Box::new(value),
                                    },
                                )),
                                Err(_) => {
                                    // Not a valid extractor
                                    Err(nom::Err::Error(ContextualError::new(
                                        input,
                                        nom::error::ErrorKind::Tag,
                                    )))
                                }
                            }
                        }
                        Err(_) => {
                            // Not an extractor pattern
                            Err(nom::Err::Error(ContextualError::new(
                                input,
                                nom::error::ErrorKind::Tag,
                            )))
                        }
                    }
                }
                Err(_) => Err(nom::Err::Error(ContextualError::new(
                    input,
                    nom::error::ErrorKind::Tag,
                ))),
            }
        },
        // Constructor pattern
        |input| {
            let (input, path_parts) = separated_list1(symbol("."), identifier).parse(input)?;

            // Check if followed by parentheses (constructor with params)
            if let Ok((rest, _)) = symbol("(").parse(input) {
                let (rest, params) =
                    separated_list0(symbol(","), |i| pattern(full, i)).parse(rest)?;
                let (rest, _) = symbol(")")(rest)?;

                let (package, name) = if path_parts.len() == 1 {
                    (vec![], path_parts[0].clone())
                } else {
                    let mut parts = path_parts;
                    let name = parts.pop().unwrap();
                    (parts, name)
                };

                Ok((
                    rest,
                    Pattern::Constructor {
                        path: TypePath {
                            package,
                            name,
                            sub: None,
                        },
                        params,
                    },
                ))
            } else {
                // Just a variable
                Ok((input, Pattern::Var(path_parts.join("."))))
            }
        },
        // Try literal expressions as constant patterns
        // Use postfix_expr to allow .code on character literals (e.g., '&'.code)
        |input| {
            if let Ok((rest, expr)) = crate::haxe_parser_expr::postfix_expr(full, input) {
                // Only accept expressions that start with a literal (string, int, float, etc.)
                // This prevents consuming complex expressions like function calls
                match &expr.kind {
                    crate::haxe_ast::ExprKind::String(_)
                    | crate::haxe_ast::ExprKind::Int(_)
                    | crate::haxe_ast::ExprKind::Float(_)
                    | crate::haxe_ast::ExprKind::Bool(_)
                    | crate::haxe_ast::ExprKind::Field { .. } => Ok((rest, Pattern::Const(expr))),
                    _ => Err(nom::Err::Error(ContextualError::new(
                        input,
                        nom::error::ErrorKind::Tag,
                    ))),
                }
            } else {
                Err(nom::Err::Error(ContextualError::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        },
        // Underscore pattern - must be after extractor pattern to avoid consuming "_."
        // Only match standalone underscore, not _.something
        |input| {
            let (input, _) = symbol("_").parse(input)?;

            // Check if followed by a dot - if so, this should be an expression
            if let Ok((_, _)) = symbol(".").parse(input) {
                // This is _.something, should be parsed as an expression
                Err(nom::Err::Error(ContextualError::new(
                    input,
                    nom::error::ErrorKind::Tag,
                )))
            } else {
                Ok((input, Pattern::Underscore))
            }
        },
    ))
    .parse(input)
}

// Loop expressions

pub fn for_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("for").parse(input)?;
    let (input, _) = symbol("(").parse(input)?;

    // Try to parse key => value pattern first
    let (input, var, key_var) = match parse_key_value_pattern(input) {
        Ok((rest, (key, value))) => (rest, value, Some(key)),
        Err(_) => {
            // Fall back to simple variable
            let (rest, var) = identifier(input)?;
            (rest, var, None)
        }
    };

    let (input, _) = keyword("in").parse(input)?;
    let (input, iter) = expression(full, input)?;
    let (input, _) = symbol(")").parse(input)?;
    let (input, body) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::For {
                var,
                key_var,
                iter: Box::new(iter),
                body: Box::new(body),
            },
            span: Span::new(start, end),
        },
    ))
}

/// Parse key => value pattern for for loops
fn parse_key_value_pattern(input: &str) -> PResult<(String, String)> {
    let (input, key) = identifier(input)?;
    let (input, _) = symbol("=>").parse(input)?;
    let (input, value) = identifier(input)?;
    Ok((input, (key, value)))
}

pub fn while_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("while").parse(input)?;
    let (input, _) = symbol("(").parse(input)?;
    let (input, cond) = expression(full, input)?;
    let (input, _) = symbol(")").parse(input)?;
    let (input, body) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::While {
                cond: Box::new(cond),
                body: Box::new(body),
            },
            span: Span::new(start, end),
        },
    ))
}

pub fn do_while_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("do").parse(input)?;
    let (input, body) = expression(full, input)?;
    let (input, _) = keyword("while").parse(input)?;
    let (input, _) = symbol("(").parse(input)?;
    let (input, cond) = expression(full, input)?;
    let (input, _) = symbol(")").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::DoWhile {
                body: Box::new(body),
                cond: Box::new(cond),
            },
            span: Span::new(start, end),
        },
    ))
}

// Continue in next part...
