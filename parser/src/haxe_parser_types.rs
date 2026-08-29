//! Type parsing for Haxe
//!
//! This module handles parsing of type expressions, type parameters, etc.

use nom::{
    branch::alt,
    combinator::{map, opt, peek},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, preceded},
    IResult, Parser,
};

use crate::custom_error::ContextualError;
use crate::haxe_ast::*;
use crate::haxe_parser::{dot_path, identifier, keyword, position, symbol, ws, PResult};

/// Parse type parameters: `<T, U>`
pub fn type_params<'a>(full: &'a str, input: &'a str) -> PResult<'a, Vec<TypeParam>> {
    opt(delimited(
        symbol("<"),
        separated_list1(symbol(","), |i| type_param(full, i)),
        symbol(">"),
    ))
    .parse(input)
    .map(|(i, params)| (i, params.unwrap_or_default()))
}

/// Parse a single type parameter with constraints
fn type_param<'a>(full: &'a str, input: &'a str) -> PResult<'a, TypeParam> {
    let start = position(full, input);

    // Parse optional variance annotation
    let (input, variance) = opt(alt((
        map(keyword("in"), |_| Variance::Contravariant),
        map(keyword("out"), |_| Variance::Covariant),
        map(symbol("+"), |_| Variance::Covariant),
        map(symbol("-"), |_| Variance::Contravariant),
    )))
    .parse(input)?;

    let (input, name) = identifier(input)?;

    // Optional constraints: `:Type` or `:Type & Type` or `:(Type1, Type2)` or `Type & Type`
    let (input, constraints) = alt((
        // Standard colon-prefixed constraints: `:Type` or `:Type & Type` or `:(Type1, Type2)`
        preceded(
            symbol(":"),
            alt((
                // Multiple constraints separated by &
                separated_list1(symbol("&"), |i| type_expr(full, i)),
                // Multiple constraints in parens
                delimited(
                    symbol("("),
                    separated_list1(symbol(","), |i| type_expr(full, i)),
                    symbol(")"),
                ),
            )),
        ),
        // Direct intersection constraint: `U & Comparable<U>` (without colon)
        preceded(symbol("&"), |i| {
            let (i, first_constraint) = type_expr(full, i)?;
            // Keep parsing more & constraints if they exist
            let (i, rest) = opt(separated_list1(symbol("&"), |j| type_expr(full, j))).parse(i)?;
            let mut constraints = vec![first_constraint];
            if let Some(mut additional) = rest {
                constraints.append(&mut additional);
            }
            Ok((i, constraints))
        }),
        // No constraints
        |i| Ok((i, vec![])),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        TypeParam {
            name,
            constraints,
            variance: variance.unwrap_or(Variance::Invariant),
            // Empty because this path does not recover them yet, not because
            // that is correct. The legacy parser is a fallback on its way out,
            // kept for the constructs the RD parser cannot read; anything it
            // drops is a gap, and a file that reaches here loses this
            // information silently.
            meta: Vec::new(),
            default_type: None,
            span: Span::new(start, end),
        },
    ))
}

/// Parse a type expression
pub fn type_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    intersection_type(full, input)
}

/// Parse intersection type: `Type & Type`
fn intersection_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);

    // Parse the first type
    let (mut input, mut left) = union_base_type(full, input)?;

    // Keep parsing & Type while we find them (left-associative)
    loop {
        // Check for & operator
        let result = preceded(symbol("&"), |i| union_base_type(full, i)).parse(input);

        match result {
            Ok((new_input, right)) => {
                let end = position(full, new_input);
                left = Type::Intersection {
                    left: Box::new(left),
                    right: Box::new(right),
                    span: Span::new(start, end),
                };
                input = new_input;
            }
            Err(_) => break,
        }
    }

    Ok((input, left))
}

/// Base types for union/intersection (everything except union/intersection itself)
fn union_base_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    alt((
        |i| optional_type(full, i),
        |i| function_type(full, i),
        |i| basic_type(full, i),
    ))
    .parse(input)
}

/// Parse optional type: `?Type`
fn optional_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);
    let (input, _) = symbol("?").parse(input)?;

    // Parse the inner type - could be a function type or basic type
    let (input, inner) = alt((|i| function_type(full, i), |i| basic_type(full, i))).parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Type::Optional {
            inner: Box::new(inner),
            span: Span::new(start, end),
        },
    ))
}

/// Parse function type: `Int -> String -> Void` or `(Int, String) -> Float`
fn function_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);

    // Try to parse function type with parenthesized parameters first
    if let Ok((input, _)) = symbol("(").parse(input) {
        let (input, params) = separated_list0(symbol(","), |i| type_expr(full, i)).parse(input)?;
        let (input, _) = symbol(")").parse(input)?;
        let (input, _) = symbol("->").parse(input)?;
        let (input, ret) = type_expr(full, input)?;
        let end = position(full, input);

        return Ok((
            input,
            Type::Function {
                params,
                ret: Box::new(ret),
                span: Span::new(start, end),
            },
        ));
    }

    // Try to parse as right-associative function type: `Int -> String -> Void`
    let result: IResult<_, _, ContextualError<&str>> = (
        |i| basic_type(full, i),
        symbol("->"),
        separated_list1(symbol("->"), |i| basic_type(full, i)),
    )
        .parse(input);

    match result {
        Ok((input, (first_param, _, mut rest))) => {
            let ret = rest.pop().unwrap();
            let mut params = vec![first_param];
            params.extend(rest);

            let end = position(full, input);

            Ok((
                input,
                Type::Function {
                    params,
                    ret: Box::new(ret),
                    span: Span::new(start, end),
                },
            ))
        }
        Err(_) => basic_type(full, input),
    }
}

/// Parse wildcard type: `?` (standalone, not optional type)
fn wildcard_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);

    // Try to parse "?" followed by something that indicates it's not an optional type
    let (input, _) = symbol("?")(input)?;

    // Peek ahead to ensure this is a standalone wildcard, not an optional type
    // Wildcard should be followed by >, comma, or ) (end of type parameter list)
    let peek_result = peek(alt((symbol(">"), symbol(","), symbol(")")))).parse(input);

    // If peek fails, this means ? is followed by a type (optional type)
    if peek_result.is_err() {
        return Err(nom::Err::Error(ContextualError::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let end = position(full, input);

    Ok((
        input,
        Type::Wildcard {
            span: Span::new(start, end),
        },
    ))
}

/// Parse basic type (non-function)
fn basic_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    alt((
        |i| wildcard_type(full, i),
        |i| anonymous_type(full, i),
        |i| parenthesized_type(full, i),
        |i| path_type(full, i),
    ))
    .parse(input)
}

/// Parse path type with optional parameters: `Map<String, Int>`
fn path_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);

    // Parse the path
    let (input, path_parts) = dot_path(input)?;

    // Split into package, name, and sub
    let (package, name, sub) = if path_parts.len() == 1 {
        (vec![], path_parts[0].clone(), None)
    } else {
        let mut parts = path_parts;
        let name = parts.pop().unwrap();

        // Check if the last part before name is capitalized (likely a type)
        if let Some(last) = parts.last() {
            if last.chars().next().unwrap().is_uppercase() {
                let sub = Some(name);
                let name = parts.pop().unwrap();
                (parts, name, sub)
            } else {
                (parts, name, None)
            }
        } else {
            (parts, name, None)
        }
    };

    let path = TypePath { package, name, sub };

    // Optional type parameters
    let (input, params) = opt(delimited(
        symbol("<"),
        separated_list1(symbol(","), |i| type_expr(full, i)),
        symbol(">"),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        Type::Path {
            path,
            params: params.unwrap_or_default(),
            span: Span::new(start, end),
        },
    ))
}

/// Parse anonymous type: `{ var x:Int; var y:String; }`
fn anonymous_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);
    let (input, _) = symbol("{").parse(input)?;

    // Parse fields - each field ends with a semicolon
    let mut fields = Vec::new();
    let mut current_input = input;

    loop {
        // Skip whitespace before checking for closing brace
        let (input, _) = ws(current_input)?;
        current_input = input;

        // Try to parse closing brace first
        if let Ok((rest, _)) = symbol("}").parse(current_input) {
            current_input = rest;
            break;
        }

        // Check for type extension syntax: > TypeName
        if let Ok((rest, _)) = symbol(">")(current_input) {
            // Parse extended type - for now just skip it
            // TODO: Properly handle type extensions
            match type_expr(full, rest) {
                Ok((rest, _extended_type)) => {
                    // Comma or semicolon after extension
                    if let Ok((rest, _)) = symbol(",")(rest) {
                        current_input = rest;
                    } else if let Ok((rest, _)) = symbol(";")(rest) {
                        current_input = rest;
                    } else {
                        current_input = rest;
                    }
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Parse field
        match anon_field(full, current_input) {
            Ok((rest, field)) => {
                fields.push(field);
                current_input = rest;

                // Skip whitespace after field
                let (rest, _) = ws(current_input)?;
                current_input = rest;

                // Semicolon or comma after each field (optional for last field)
                if let Ok((rest, _)) = symbol(";")(current_input) {
                    current_input = rest;
                } else if let Ok((rest, _)) = symbol(",")(current_input) {
                    current_input = rest;
                }
                // If no separator found, that's ok - might be the last field
            }
            Err(_) => {
                return Err(nom::Err::Error(ContextualError::new(
                    current_input,
                    nom::error::ErrorKind::Tag,
                )))
            }
        }
    }

    let end = position(full, current_input);

    Ok((
        current_input,
        Type::Anonymous {
            fields,
            span: Span::new(start, end),
        },
    ))
}

/// Parse anonymous type field
fn anon_field<'a>(full: &'a str, input: &'a str) -> PResult<'a, AnonField> {
    let start = position(full, input);

    // Parse metadata first
    let (input, meta) = crate::haxe_parser::metadata_list(full, input)?;

    // Check if this has @:optional metadata
    let has_optional_meta = meta.iter().any(|m| m.name == "optional");

    // Check for function field
    if let Ok((input, _)) = keyword("function")(input) {
        // Parse function name
        let (input, name) = identifier(input)?;

        // Skip the rest of the function signature for now
        // Look for balanced parentheses and then : Type
        let (input, _) = symbol("(")(input)?;
        let mut paren_count = 1;
        let mut current_input = input;

        // Skip until we find matching closing parenthesis
        while paren_count > 0 && !current_input.is_empty() {
            if current_input.starts_with('(') {
                paren_count += 1;
                current_input = &current_input[1..];
            } else if current_input.starts_with(')') {
                paren_count -= 1;
                current_input = &current_input[1..];
            } else {
                // Skip character
                let mut char_indices = current_input.char_indices();
                char_indices.next();
                current_input = match char_indices.next() {
                    Some((idx, _)) => &current_input[idx..],
                    None => &current_input[current_input.len()..],
                };
            }
        }

        // Parse return type
        let (input, _) = symbol(":")(current_input)?;
        let (input, type_hint) = type_expr(full, input)?;

        let end = position(full, input);

        return Ok((
            input,
            AnonField {
                name,
                optional: has_optional_meta,
                type_hint,
                span: Span::new(start, end),
            },
        ));
    }

    // Optional var keyword (not required in typedef anonymous types)
    let (input, _) = opt(keyword("var")).parse(input)?;

    // Optional ? for optional field (can appear before or after var)
    let (input, optional) = opt(symbol("?")).parse(input)?;
    let (input, name) = identifier(input)?;
    let (input, _) = symbol(":").parse(input)?;
    let (input, type_hint) = type_expr(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        AnonField {
            name,
            optional: optional.is_some() || has_optional_meta,
            type_hint,
            span: Span::new(start, end),
        },
    ))
}

/// Parse parenthesized type: `(Type)`
fn parenthesized_type<'a>(full: &'a str, input: &'a str) -> PResult<'a, Type> {
    let start = position(full, input);
    let (input, _) = symbol("(")(input)?;
    let (input, inner) = type_expr(full, input)?;
    let (input, _) = symbol(")")(input)?;
    let end = position(full, input);

    Ok((
        input,
        Type::Parenthesis {
            inner: Box::new(inner),
            span: Span::new(start, end),
        },
    ))
}
