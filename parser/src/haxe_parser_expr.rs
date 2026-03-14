//! Expression parsing for Haxe
//!
//! This module handles all expression parsing with proper precedence and associativity

use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{char, digit1, hex_digit1, none_of, oct_digit1, one_of},
    combinator::{map, opt, recognize, value},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded},
    Parser,
};

use crate::haxe_ast::*;
use crate::haxe_parser::{identifier, keyword, position, symbol, ws, PResult};
use crate::haxe_parser_expr2::{
    array_expr, block_expr, cast_expr, compiler_specific_expr, do_while_expr, for_expr,
    identifier_expr, if_expr, inline_expr, inline_preprocessor_expr, macro_expr, new_expr,
    null_expr, object_expr, reify_expr, super_expr, switch_expr, this_expr, untyped_expr,
    while_expr,
};
use crate::haxe_parser_expr3::{
    arrow_params, break_expr, continue_expr, function_expr, metadata_expr, paren_expr, return_expr,
    throw_expr, try_expr, var_expr,
};

/// Parse any expression
pub fn expression<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    assignment_expr(full, input)
}

/// Parse ternary expression: `cond ? then : else`
/// Ternary has higher precedence than assignment but lower than null coalescing.
fn ternary_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    use nom::error::context;
    let start = position(full, input);
    let (input, cond) = null_coalescing_expr(full, input)?;

    // Check for ternary
    if let Ok((input, _)) = symbol("?")(input) {
        // Then/else branches allow assignment expressions (e.g. `a ? b = c : d`)
        let (input, then_expr) = context("[E0050] expected expression after '?' in ternary operator | help: provide the expression to return when condition is true", |i| assignment_expr(full, i)).parse(input)?;
        let (input, _) = context("[E0051] expected ':' after then expression in ternary operator | help: ternary operator requires ':' to separate then and else branches", symbol(":")).parse(input)?;
        let (input, else_expr) = context("[E0052] expected expression after ':' in ternary operator | help: provide the expression to return when condition is false", |i| assignment_expr(full, i)).parse(input)?;
        let end = position(full, input);

        Ok((
            input,
            Expr {
                kind: ExprKind::Ternary {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                },
                span: Span::new(start, end),
            },
        ))
    } else {
        Ok((input, cond))
    }
}

/// Parse assignment expression: `a = b`, `a += b`, etc.
pub fn assignment_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);

    // Try arrow function first (higher precedence than assignment)
    if let Ok((rest, params)) = arrow_params(full, input) {
        let (rest, _) = ws(rest)?; // Skip whitespace before arrow
        if let Ok((rest, _)) = symbol("->")(rest) {
            let (rest, _) = ws(rest)?; // Skip whitespace after arrow

            // Try block expression first, then fall back to assignment expression
            let (rest, body) =
                alt((|i| block_expr(full, i), |i| assignment_expr(full, i))).parse(rest)?;

            let end = position(full, rest);

            return Ok((
                rest,
                Expr {
                    kind: ExprKind::Arrow {
                        params,
                        expr: Box::new(body),
                    },
                    span: Span::new(start, end),
                },
            ));
        }
    }

    let (input, left) = ternary_expr(full, input)?;

    // Check for assignment operators
    let assign_ops = [
        ("=", AssignOp::Assign),
        // Longer operators must come first to avoid partial matches
        (">>>=", AssignOp::UshrAssign), // Must come before >>=
        (">>=", AssignOp::ShrAssign),
        ("<<=", AssignOp::ShlAssign),
        ("+=", AssignOp::AddAssign),
        ("-=", AssignOp::SubAssign),
        ("*=", AssignOp::MulAssign),
        ("/=", AssignOp::DivAssign),
        ("%=", AssignOp::ModAssign),
        ("&=", AssignOp::AndAssign),
        ("|=", AssignOp::OrAssign),
        ("^=", AssignOp::XorAssign),
    ];

    // Try to parse assignment operator
    for (op_str, assign_op) in &assign_ops {
        if let Ok((rest, _)) = symbol(op_str).parse(input) {
            // Special case: don't consume = if it's part of =>
            if *op_str == "=" && rest.trim_start().starts_with('>') {
                continue;
            }
            let (rest, right) = assignment_expr(full, rest)?; // Right-associative
            let end = position(full, rest);

            return Ok((
                rest,
                Expr {
                    kind: ExprKind::Assign {
                        left: Box::new(left),
                        op: *assign_op,
                        right: Box::new(right),
                    },
                    span: Span::new(start, end),
                },
            ));
        }
    }

    // No assignment, return the null coalescing expression
    Ok((input, left))
}

/// Parse null coalescing expression: `a ?? b`
fn null_coalescing_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, logical_or_expr, &[("??", BinaryOp::NullCoal)])
}

/// Parse logical OR expression: `a || b`
fn logical_or_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, logical_and_expr, &[("||", BinaryOp::Or)])
}

/// Parse logical AND expression: `a && b`
fn logical_and_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, bitwise_or_expr, &[("&&", BinaryOp::And)])
}

/// Parse bitwise OR expression: `a | b`
fn bitwise_or_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, bitwise_xor_expr, &[("|", BinaryOp::BitOr)])
}

/// Parse bitwise XOR expression: `a ^ b`
fn bitwise_xor_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, bitwise_and_expr, &[("^", BinaryOp::BitXor)])
}

/// Parse bitwise AND expression: `a & b`
fn bitwise_and_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, equality_expr, &[("&", BinaryOp::BitAnd)])
}

/// Parse equality expression: `a == b`, `a != b`
fn equality_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(
        full,
        input,
        relational_expr,
        &[("==", BinaryOp::Eq), ("!=", BinaryOp::NotEq)],
    )
}

/// Parse relational expression: `a < b`, `a <= b`, etc.
fn relational_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    // First parse the left side
    let (input, mut left) = shift_expr(full, input)?;
    let mut current_input = input;

    loop {
        // Skip whitespace
        let (input, _) = ws(current_input)?;
        current_input = input;

        // Check for `is` operator
        if let Ok((rest, _)) = keyword("is")(current_input) {
            let (rest, right) = shift_expr(full, rest)?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinaryOp::Is,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
            current_input = rest;
            continue;
        }

        // Check for comparison operators
        // Note: Must check longer operators first and avoid matching when part of >>>, >>=, >>>=
        let op = if let Ok((rest, _)) = symbol("<=").parse(current_input) {
            // <= is safe, not part of any longer operator
            current_input = rest;
            Some(BinaryOp::Le)
        } else if let Ok((rest, _)) = symbol(">=").parse(current_input) {
            // >= is safe, not part of any longer operator
            current_input = rest;
            Some(BinaryOp::Ge)
        } else if let Ok((rest, _)) = symbol("<").parse(current_input) {
            // < - check it's not <<= or <<
            if !rest.starts_with('<') && !rest.starts_with('=') {
                current_input = rest;
                Some(BinaryOp::Lt)
            } else {
                None
            }
        } else if let Ok((rest, _)) = symbol(">").parse(current_input) {
            // > - check it's not >>, >>>, >>=, >>>=
            if !rest.starts_with('>') && !rest.starts_with('=') {
                current_input = rest;
                Some(BinaryOp::Gt)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(op) = op {
            let (rest, right) = shift_expr(full, current_input)?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
            current_input = rest;
        } else {
            break;
        }
    }

    Ok((current_input, left))
}

/// Parse shift expression: `a << b`, `a >> b`, `a >>> b`
fn shift_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(
        full,
        input,
        range_expr,
        &[
            (">>>", BinaryOp::Ushr),
            ("<<", BinaryOp::Shl),
            (">>", BinaryOp::Shr),
        ],
    )
}

/// Parse range expression: `a...b`
fn range_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(full, input, additive_expr, &[("...", BinaryOp::Range)])
}

/// Parse additive expression: `a + b`, `a - b`
fn additive_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(
        full,
        input,
        multiplicative_expr,
        &[("+", BinaryOp::Add), ("-", BinaryOp::Sub)],
    )
}

/// Parse multiplicative expression: `a * b`, `a / b`, `a % b`
fn multiplicative_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    binary_expr(
        full,
        input,
        unary_expr,
        &[
            ("*", BinaryOp::Mul),
            ("/", BinaryOp::Div),
            ("%", BinaryOp::Mod),
        ],
    )
}

/// Generic binary expression parser
fn binary_expr<'a>(
    full: &'a str,
    input: &'a str,
    mut sub_expr: impl FnMut(&'a str, &'a str) -> PResult<'a, Expr>,
    ops: &'a [(&str, BinaryOp)],
) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, mut left) = sub_expr(full, input)?;

    let (input, rest) =
        many0(pair(ws_before_one_of_ops(ops), |i| sub_expr(full, i))).parse(input)?;

    // Build left-associative tree
    for (op_str, right) in rest {
        let op = ops.iter().find(|(s, _)| *s == op_str).unwrap().1;
        let end = position(full, input);

        left = Expr {
            kind: ExprKind::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            span: Span::new(start, end),
        };
    }

    Ok((input, left))
}

/// Parse operator from a list
fn ws_before_one_of_ops<'a>(
    ops: &'a [(&'a str, BinaryOp)],
) -> impl FnMut(&'a str) -> PResult<'a, &'a str> + 'a {
    move |input| {
        let (input, _) = ws.parse(input)?;
        for (op_str, _) in ops {
            if let Ok((rest, _)) = tag::<_, _, nom::error::Error<_>>(*op_str).parse(input) {
                // Don't match if the operator is followed by '=' (compound assignment)
                // e.g., don't match >>> if the input is >>>=
                if rest.starts_with('=') {
                    continue;
                }
                return Ok((rest, *op_str));
            }
        }
        Err(nom::Err::Error(crate::custom_error::ContextualError::new(
            input,
            nom::error::ErrorKind::Tag,
        )))
    }
}

/// Parse unary expression
pub fn unary_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    alt((|i| prefix_unary_expr(full, i), |i| postfix_expr(full, i))).parse(input)
}

/// Parse prefix unary expression
fn prefix_unary_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);

    let (input, op) = alt((
        value(UnaryOp::Not, symbol("!")),
        value(UnaryOp::Neg, symbol("-")),
        value(UnaryOp::BitNot, symbol("~")),
        value(UnaryOp::PreIncr, symbol("++")),
        value(UnaryOp::PreDecr, symbol("--")),
    ))
    .parse(input)?;

    let (input, expr) = unary_expr(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Unary {
                op,
                expr: Box::new(expr),
            },
            span: Span::new(start, end),
        },
    ))
}

/// Parse postfix expression
pub fn postfix_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    // Skip leading whitespace before recording start so expression spans begin at the
    // actual first character of the expression, not at a preceding newline or indent.
    let (input, _) = ws(input)?;
    let start = position(full, input);
    let (input, mut expr) = primary_expr(full, input)?;

    // Parse postfix operations
    let mut input = input;
    let mut loop_counter = 0;
    const MAX_POSTFIX_ITERATIONS: usize = 100; // Safety limit

    loop {
        loop_counter += 1;
        if loop_counter > MAX_POSTFIX_ITERATIONS {
            // Emergency brake - break the loop if we're stuck
            break;
        }

        let input_before = input;
        // Try each postfix operation
        // Check for optional chaining `?.` before regular field access
        if let Ok((rest, _)) = symbol("?.").parse(input) {
            // Optional chaining field access: obj?.field
            let (rest, field) = identifier(rest)?;
            let end = position(full, rest);
            expr = Expr {
                kind: ExprKind::Field {
                    expr: Box::new(expr),
                    field,
                    is_optional: true,
                },
                span: Span::new(start, end),
            };
            input = rest;
        } else if let Ok((rest, _)) = symbol(".").parse(input) {
            // Make sure this isn't part of a ... operator
            if rest.starts_with("..") {
                // This is part of ..., don't consume it
                break;
            }
            // Field access
            let (rest, field) = identifier(rest)?;
            let end = position(full, rest);
            expr = Expr {
                kind: ExprKind::Field {
                    expr: Box::new(expr),
                    field,
                    is_optional: false,
                },
                span: Span::new(start, end),
            };
            input = rest;
        } else if let Ok((rest, _)) = symbol("[").parse(input) {
            // Array/map access
            let (rest, index) = expression(full, rest)?;
            let (rest, _) = symbol("]")(rest)?;
            let end = position(full, rest);
            expr = Expr {
                kind: ExprKind::Index {
                    expr: Box::new(expr),
                    index: Box::new(index),
                },
                span: Span::new(start, end),
            };
            input = rest;
        } else if let Ok((rest, _)) = symbol("(").parse(input) {
            // Function call
            let (rest, args) = separated_list0(symbol(","), |i| expression(full, i)).parse(rest)?;
            let (rest, _) = opt(symbol(",")).parse(rest)?; // Trailing comma
            let (rest, _) = symbol(")")(rest)?;
            let end = position(full, rest);
            expr = Expr {
                kind: ExprKind::Call {
                    expr: Box::new(expr),
                    args,
                },
                span: Span::new(start, end),
            };
            input = rest;
        } else if let Ok((rest, op)) = alt((
            value(UnaryOp::PostIncr, symbol("++")),
            value(UnaryOp::PostDecr, symbol("--")),
        ))
        .parse(input)
        {
            // Postfix increment/decrement
            let end = position(full, rest);
            expr = Expr {
                kind: ExprKind::Unary {
                    op,
                    expr: Box::new(expr),
                },
                span: Span::new(start, end),
            };
            input = rest;
        } else {
            // No more postfix operations
            break;
        }

        // Safety check: ensure we consumed some input to prevent infinite loops
        if input == input_before {
            // If we didn't advance the input, break to prevent infinite loop
            break;
        }
    }

    Ok((input, expr))
}

/// Parse primary expression
fn primary_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    alt((
        |i| {
            alt((
                |i| literal_expr(full, i),
                |i| macro_expr(full, i),
                |i| reify_expr(full, i),
                |i| compiler_specific_expr(full, i),
                |i| identifier_expr(full, i),
                |i| this_expr(full, i),
                |i| super_expr(full, i),
                |i| null_expr(full, i),
                |i| new_expr(full, i),
                |i| cast_expr(full, i),
                |i| untyped_expr(full, i),
                |i| inline_expr(full, i),
                |i| inline_preprocessor_expr(full, i),
                |i| array_expr(full, i),
            ))
            .parse(i)
        },
        |i| {
            alt((
                |i| object_expr(full, i),
                |i| block_expr(full, i),
                |i| if_expr(full, i),
                |i| switch_expr(full, i),
                |i| for_expr(full, i),
                |i| while_expr(full, i),
                |i| do_while_expr(full, i),
                |i| try_expr(full, i),
            ))
            .parse(i)
        },
        |i| {
            alt((
                |i| metadata_expr(full, i),
                |i| function_expr(full, i),
                |i| return_expr(full, i),
                |i| break_expr(full, i),
                |i| continue_expr(full, i),
                |i| throw_expr(full, i),
                |i| var_expr(full, i),
                |i| paren_expr(full, i),
            ))
            .parse(i)
        },
    ))
    .parse(input)
}

// Literal expressions

pub fn literal_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    alt((
        |i| float_literal(full, i),
        |i| int_literal(full, i),
        |i| string_literal(full, i),
        |i| bool_literal(full, i),
        |i| regex_literal(full, i),
    ))
    .parse(input)
}

fn int_literal<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = ws(input)?;

    let (input, value) = alt((
        // Hex literal
        map(preceded(tag("0x"), hex_digit1), |s: &str| {
            i64::from_str_radix(s, 16).unwrap_or(0)
        }),
        // Octal literal
        map(preceded(tag("0"), oct_digit1), |s: &str| {
            i64::from_str_radix(s, 8).unwrap_or(0)
        }),
        // Decimal literal
        map(digit1, |s: &str| s.parse().unwrap_or(0)),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Int(value),
            span: Span::new(start, end),
        },
    ))
}

fn float_literal<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = ws(input)?;

    let (input, value) = map(
        recognize((
            digit1,
            char('.'),
            digit1,
            opt((one_of("eE"), opt(one_of("+-")), digit1)),
        )),
        |s: &str| {
            s.parse::<f64>().unwrap_or_else(|_| {
                eprintln!("Failed to parse float: '{}'", s);
                0.0
            })
        },
    )
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Float(value),
            span: Span::new(start, end),
        },
    ))
}

fn string_literal<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let _start = position(full, input);
    let (input, _) = ws(input)?;

    alt((
        // Single-quoted string with interpolation
        |i| interpolated_string(full, i, '\''),
        // Double-quoted string (no interpolation)
        |i| simple_string(full, i, '"'),
        // Single-quoted string (no interpolation if no $)
        |i| {
            let (i, s) = simple_string(full, i, '\'')?;
            // Check if it contains $ for interpolation
            if let Expr {
                kind: ExprKind::String(ref str_val),
                ..
            } = s
            {
                if str_val.contains('$') {
                    // Re-parse as interpolated
                    interpolated_string(full, input, '\'')
                } else {
                    Ok((i, s))
                }
            } else {
                Ok((i, s))
            }
        },
    ))
    .parse(input)
}

fn simple_string<'a>(full: &'a str, input: &'a str, quote: char) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = char(quote).parse(input)?;

    // Check if this is an empty string (next char is the closing quote)
    if input.starts_with(quote) {
        let (input, _) = char(quote).parse(input)?;
        let end = position(full, input);
        return Ok((
            input,
            Expr {
                kind: ExprKind::String(String::new()),
                span: Span::new(start, end),
            },
        ));
    }

    // Non-empty string
    let (input, content) = escaped(
        none_of(&format!("\\{}", quote)[..]),
        '\\',
        one_of(&format!("\\{}nrtbfv", quote)[..]),
    )
    .parse(input)?;
    let (input, _) = char(quote).parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::String(unescape_string(content)),
            span: Span::new(start, end),
        },
    ))
}

/// Convert escape sequences in a parsed string literal to their actual characters.
fn unescape_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some('b') => result.push('\x08'), // backspace
                Some('f') => result.push('\x0C'), // form feed
                Some('v') => result.push('\x0B'), // vertical tab
                Some('0') => result.push('\0'),   // null
                Some('u') => {
                    // Unicode escape: \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(code) {
                            result.push(ch);
                        }
                    }
                }
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn interpolated_string<'a>(full: &'a str, input: &'a str, quote: char) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = char(quote).parse(input)?;

    let mut parts = Vec::new();
    let mut current = String::new();
    let mut remaining = input;

    while !remaining.starts_with(quote) {
        if remaining.starts_with("$$") {
            // Escaped dollar
            current.push('$');
            remaining = &remaining[2..];
        } else if remaining.starts_with("${") {
            // Expression interpolation
            if !current.is_empty() {
                parts.push(StringPart::Literal(current.clone()));
                current.clear();
            }

            let (rest, expr) =
                delimited(tag("${"), |i| expression(full, i), char('}')).parse(remaining)?;

            parts.push(StringPart::Interpolation(expr));
            remaining = rest;
        } else if remaining.starts_with('$') && remaining.len() > 1 {
            // Simple identifier interpolation
            if !current.is_empty() {
                parts.push(StringPart::Literal(current.clone()));
                current.clear();
            }

            let (rest, id) = preceded(char('$'), identifier).parse(remaining)?;
            parts.push(StringPart::Interpolation(Expr {
                kind: ExprKind::Ident(id),
                span: Span::default(), // Will be fixed
            }));
            remaining = rest;
        } else if remaining.starts_with('\\') && remaining.len() > 1 {
            // Escape sequence - convert to actual character
            if let Some(ch) = remaining.chars().nth(1) {
                let actual = match ch {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '\\' => '\\',
                    '\'' => '\'',
                    '"' => '"',
                    'b' => '\x08',
                    'f' => '\x0C',
                    'v' => '\x0B',
                    '0' => '\0',
                    other => other, // Unknown escape: keep the char after backslash
                };
                current.push(actual);
                let mut char_indices = remaining.char_indices();
                char_indices.next(); // Position 0: backslash
                char_indices.next(); // Position 1: escaped char
                if let Some((next_idx, _)) = char_indices.next() {
                    remaining = &remaining[next_idx..];
                } else {
                    remaining = "";
                }
            } else {
                remaining = "";
            }
        } else {
            // Regular character
            if let Some(ch) = remaining.chars().next() {
                current.push(ch);
                let mut char_indices = remaining.char_indices();
                char_indices.next(); // Skip current char
                if let Some((idx, _)) = char_indices.next() {
                    remaining = &remaining[idx..];
                } else {
                    remaining = "";
                }
            } else {
                remaining = "";
            }
        }
    }

    if !current.is_empty() {
        parts.push(StringPart::Literal(current));
    }

    let (input, _) = char(quote)(remaining)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: if parts.len() == 1 && matches!(parts[0], StringPart::Literal(_)) {
                // Simple string
                if let StringPart::Literal(s) = &parts[0] {
                    ExprKind::String(s.clone())
                } else {
                    unreachable!()
                }
            } else {
                // Interpolated string
                ExprKind::StringInterpolation(parts)
            },
            span: Span::new(start, end),
        },
    ))
}

fn bool_literal<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);

    let (input, value) =
        alt((value(true, keyword("true")), value(false, keyword("false")))).parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Bool(value),
            span: Span::new(start, end),
        },
    ))
}

fn regex_literal<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = ws(input)?;

    // Parse ~/pattern/flags
    let (input, _) = tag("~/")(input)?;

    // Parse the pattern - anything until we hit an unescaped /
    let mut pattern = String::new();
    let chars = input.char_indices();
    let mut escaped = false;
    let mut end_idx = input.len(); // Default to end if no closing / found

    for (idx, ch) in chars {
        if escaped {
            pattern.push('\\');
            pattern.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '/' {
            end_idx = idx;
            break;
        } else {
            pattern.push(ch);
        }
    }

    // Get the substring after the pattern
    let pattern_end_input = &input[end_idx..];

    // Expect the closing /
    let (after_slash, _) = char('/')(pattern_end_input)?;

    // Parse optional flags (igmsux)
    let (remaining, flags) = recognize(many0(one_of("igmsux"))).parse(after_slash)?;

    let end = position(full, remaining);

    Ok((
        remaining,
        Expr {
            kind: ExprKind::Regex {
                pattern,
                flags: flags.to_string(),
            },
            span: Span::new(start, end),
        },
    ))
}

// Continue with other expression parsers...
