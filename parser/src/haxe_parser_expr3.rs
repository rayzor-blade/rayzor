//! Expression parsing for Haxe (final part)
//!
//! This module contains try/catch, function, and other remaining expression parsers

use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char},
    combinator::{map, opt, recognize, value},
    multi::{many0, many1, separated_list0},
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};

use crate::custom_error::ContextualError;
use crate::haxe_ast::*;
use crate::haxe_parser::{identifier, keyword, position, symbol, ws, PResult};
use crate::haxe_parser_decls::function_param;
use crate::haxe_parser_expr::expression;
use crate::haxe_parser_expr2::block_expr;
use crate::haxe_parser_types::{type_expr, type_params};

/// Parse function body - handles both block and single expression bodies
fn function_body<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let (input, _) = ws(input)?;

    // Check if this is a block body (starts with '{')
    if input.starts_with('{') {
        block_expr(full, input)
    } else {
        // Single expression body
        expression(full, input)
    }
}

// Exception handling

pub fn try_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    use nom::error::context;
    let start = position(full, input);
    let (input, _) = context("[E0060] expected 'try' keyword", keyword("try")).parse(input)?;
    let (input, expr) = context("[E0061] expected expression after 'try' | help: provide the expression that might throw an exception", |i| expression(full, i)).parse(input)?;
    let (input, catches) = context("[E0062] expected at least one 'catch' clause after try expression | help: try blocks must have at least one catch clause", many1(|i| catch_clause(full, i))).parse(input)?;
    let (input, finally_block) =
        opt(preceded(keyword("finally"), |i| expression(full, i))).parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Try {
                expr: Box::new(expr),
                catches,
                finally_block: finally_block.map(Box::new),
            },
            span: Span::new(start, end),
        },
    ))
}

fn catch_clause<'a>(full: &'a str, input: &'a str) -> PResult<'a, Catch> {
    use nom::error::context;
    let start = position(full, input);
    let (input, _) = context("[E0063] expected 'catch' keyword", keyword("catch")).parse(input)?;
    let (input, _) = context("[E0064] expected '(' after 'catch' | help: catch clause requires parentheses around the exception variable", symbol("(")).parse(input)?;
    let (input, var) = context("[E0065] expected variable name in catch clause | help: provide a name for the caught exception", identifier).parse(input)?;
    let (input, type_hint) = opt(preceded(
        context("[E0066] expected ':' before exception type", symbol(":")),
        |i| type_expr(full, i),
    ))
    .parse(input)?;
    let (input, _) = context(
        "[E0067] expected ')' to close catch parameter list",
        symbol(")"),
    )
    .parse(input)?;
    let (input, filter) = opt(preceded(
        context(
            "[E0068] expected 'if' keyword for catch filter",
            keyword("if"),
        ),
        |i| expression(full, i),
    ))
    .parse(input)?;
    let (input, body) = context("[E0069] expected catch body expression | help: provide an expression or block to handle the caught exception", |i| expression(full, i)).parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Catch {
            var,
            type_hint,
            filter,
            body,
            span: Span::new(start, end),
        },
    ))
}

// Function expressions

pub fn function_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);

    // Try arrow function first
    if let Ok((rest, params)) = arrow_params(full, input) {
        let (rest, _) = ws(rest)?; // Skip whitespace before arrow
        if let Ok((rest, _)) = symbol("->")(rest) {
            let (rest, _) = ws(rest)?; // Skip whitespace after arrow
            let (rest, body) = expression(full, rest)?;
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

    // Regular function expression
    let (input, _) = keyword("function").parse(input)?;
    let (input, name) = opt(identifier).parse(input)?;
    let (input, type_params) = type_params(full, input)?;

    let (input, _) = symbol("(").parse(input)?;
    let (input, params) = separated_list0(symbol(","), |i| function_param(full, i)).parse(input)?;
    let (input, _) = symbol(")").parse(input)?;

    let (input, return_type) = opt(preceded(symbol(":"), |i| type_expr(full, i))).parse(input)?;
    let (input, body) = opt(|i| function_body(full, i)).parse(input)?;

    let end = position(full, input);
    let span = Span::new(start, end);

    Ok((
        input,
        Expr {
            kind: ExprKind::Function(Function {
                name: name.unwrap_or_default(),
                type_params,
                params,
                return_type,
                body: body.map(Box::new),
                span,
            }),
            span,
        },
    ))
}

pub fn arrow_params<'a>(full: &'a str, input: &'a str) -> PResult<'a, Vec<ArrowParam>> {
    let (input, _) = ws(input)?; // Skip leading whitespace
    alt((
        // Single parameter without parentheses (no type annotation allowed)
        map(
            |i| {
                let (i, _) = ws(i)?;
                identifier(i)
            },
            |id| {
                vec![ArrowParam {
                    name: id,
                    type_hint: None,
                }]
            },
        ),
        // Multiple parameters in parentheses, with optional type annotations
        // Supports: (x), (x:Int), (x:Int, y:String), etc.
        delimited(
            symbol("("),
            separated_list0(symbol(","), |i| {
                let (i, _) = ws(i)?;
                let (i, name) = identifier(i)?;
                let (i, type_hint) = opt(preceded(symbol(":"), |i| type_expr(full, i))).parse(i)?;
                Ok((i, ArrowParam { name, type_hint }))
            }),
            symbol(")"),
        ),
    ))
    .parse(input)
}

// Control flow expressions

pub fn return_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("return").parse(input)?;
    let (input, value) = opt(|i| expression(full, i)).parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Return(value.map(Box::new)),
            span: Span::new(start, end),
        },
    ))
}

pub fn break_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("break").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Break,
            span: Span::new(start, end),
        },
    ))
}

pub fn continue_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("continue").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Continue,
            span: Span::new(start, end),
        },
    ))
}

pub fn throw_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = keyword("throw").parse(input)?;
    let (input, expr) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Throw(Box::new(expr)),
            span: Span::new(start, end),
        },
    ))
}

// Variable declarations

pub fn var_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    use nom::error::context;
    let start = position(full, input);

    let (input, is_final) = alt((
        value(
            true,
            context("[E0070] expected 'final' keyword", keyword("final")),
        ),
        value(
            false,
            context("[E0071] expected 'var' keyword", keyword("var")),
        ),
    ))
    .parse(input)?;

    let (input, name) = context(
        "[E0072] expected variable name | help: provide a name for the variable declaration",
        identifier,
    )
    .parse(input)?;
    let (input, type_hint) = opt(preceded(
        context("[E0073] expected ':' before type annotation", symbol(":")),
        |i| type_expr(full, i),
    ))
    .parse(input)?;
    let (input, expr) = opt(preceded(
        context("[E0074] expected '=' before initializer", symbol("=")),
        |i| expression(full, i),
    ))
    .parse(input)?;

    let end = position(full, input);

    let kind = if is_final {
        ExprKind::Final {
            name,
            type_hint,
            expr: expr.map(Box::new),
        }
    } else {
        ExprKind::Var {
            name,
            type_hint,
            expr: expr.map(Box::new),
        }
    };

    Ok((
        input,
        Expr {
            kind,
            span: Span::new(start, end),
        },
    ))
}

// Parentheses and metadata

pub fn paren_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, _) = symbol("(").parse(input)?;

    // Check for type check syntax: (expr : Type)
    let mut check_typecheck = |input| -> IResult<_, _, ContextualError<&str>> {
        let (input, expr) = expression(full, input)?;
        let (input, _) = symbol(":").parse(input)?;
        let (input, type_hint) = type_expr(full, input)?;
        Ok((input, (expr, type_hint)))
    };

    if let Ok((rest, (expr, type_hint))) = check_typecheck.parse(input) {
        let (rest, _) = symbol(")")(rest)?;
        let end = position(full, rest);

        return Ok((
            rest,
            Expr {
                kind: ExprKind::TypeCheck {
                    expr: Box::new(expr),
                    type_hint,
                },
                span: Span::new(start, end),
            },
        ));
    }

    // Regular parenthesized expression or tuple literal
    let (input, first_expr) = expression(full, input)?;

    // Check for tuple: (expr, expr, ...)
    if let Ok((input, _)) = symbol(",").parse(input) {
        let mut elements = vec![first_expr];
        let (input, rest) = separated_list0(symbol(","), |i| expression(full, i)).parse(input)?;
        elements.extend(rest);
        let (input, _) = opt(symbol(",")).parse(input)?; // trailing comma
        let (input, _) = symbol(")")(input)?;
        let end = position(full, input);
        return Ok((
            input,
            Expr {
                kind: ExprKind::Tuple(elements),
                span: Span::new(start, end),
            },
        ));
    }

    let (input, _) = symbol(")").parse(input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Paren(Box::new(first_expr)),
            span: Span::new(start, end),
        },
    ))
}

pub fn metadata_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let start = position(full, input);
    let (input, meta) = single_metadata_for_expr(full, input)?;
    let (input, expr) = expression(full, input)?;
    let end = position(full, input);

    Ok((
        input,
        Expr {
            kind: ExprKind::Meta {
                meta,
                expr: Box::new(expr),
            },
            span: Span::new(start, end),
        },
    ))
}

// Helper to parse a single metadata attribute for expressions
fn single_metadata_for_expr<'a>(full: &'a str, input: &'a str) -> PResult<'a, Metadata> {
    let start = position(full, input);
    let (input, _) = ws(input)?;
    let (input, _) = char('@').parse(input)?;
    let (input, has_colon) = opt(char(':')).parse(input)?;
    let (input, name) = if has_colon.is_some() {
        // @:metadata format - allow keywords in metadata context
        let (input, _) = ws(input)?;
        let (input, id) = recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))
        .parse(input)?;
        (input, id.to_string())
    } else {
        // @metadata format - also allow keywords in metadata context
        let (input, _) = ws(input)?;
        let (input, id) = recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        ))
        .parse(input)?;
        (input, id.to_string())
    };

    // Only parse parameters if there's an immediate opening parenthesis (no whitespace)
    let (input, params) = if input.starts_with('(') {
        let (input, params) = opt(delimited(
            char('('),
            separated_list0(symbol(","), |i| expression(full, i)),
            char(')'),
        ))
        .parse(input)?;
        (input, params)
    } else {
        (input, None)
    };

    let end = position(full, input);

    Ok((
        input,
        Metadata {
            name,
            params: params.unwrap_or_default(),
            span: Span::new(start, end),
        },
    ))
}

// Assignment handling (for binary_expr to use)

pub fn handle_assignment<'a>(
    full: &'a str,
    start: usize,
    left: Expr,
    op: BinaryOp,
    right: Expr,
    end_input: &'a str,
) -> Expr {
    let end = position(full, end_input);

    // Convert binary op to assignment op
    let assign_op = match op {
        BinaryOp::Add => Some(AssignOp::AddAssign),
        BinaryOp::Sub => Some(AssignOp::SubAssign),
        BinaryOp::Mul => Some(AssignOp::MulAssign),
        BinaryOp::Div => Some(AssignOp::DivAssign),
        BinaryOp::Mod => Some(AssignOp::ModAssign),
        BinaryOp::BitAnd => Some(AssignOp::AndAssign),
        BinaryOp::BitOr => Some(AssignOp::OrAssign),
        BinaryOp::BitXor => Some(AssignOp::XorAssign),
        BinaryOp::Shl => Some(AssignOp::ShlAssign),
        BinaryOp::Shr => Some(AssignOp::ShrAssign),
        BinaryOp::Ushr => Some(AssignOp::UshrAssign),
        _ => None,
    };

    if let Some(assign_op) = assign_op {
        Expr {
            kind: ExprKind::Assign {
                left: Box::new(left),
                op: assign_op,
                right: Box::new(right),
            },
            span: Span::new(start, end),
        }
    } else {
        // Regular binary expression
        Expr {
            kind: ExprKind::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            span: Span::new(start, end),
        }
    }
}
