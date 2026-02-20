//! Type declaration parsing for Haxe
//!
//! This module handles parsing of classes, interfaces, enums, typedefs, and abstracts

use crate::haxe_ast::*;
use crate::haxe_parser::{
    access, function_name, identifier, keyword, metadata_list, modifiers, position, symbol, ws,
    PResult,
};
use crate::haxe_parser_expr::expression;
use crate::haxe_parser_expr2::block_expr;
use crate::haxe_parser_types::{type_expr, type_params};
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt, peek, value},
    error::context,
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, preceded, tuple},
    Parser,
};

/// Parse function body - handles both block and single expression bodies
fn function_body<'a>(full: &'a str, input: &'a str) -> PResult<'a, Expr> {
    let (input, _) = ws(input)?;

    // Check if this is a block body (starts with '{')
    if input.starts_with('{') {
        let result = block_expr(full, input);

        result
    } else {
        // Single expression body
        expression(full, input)
    }
}

/// Parse class declaration
pub fn class_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, ClassDecl> {
    context("class declaration", |input: &'a str| {

    let start = position(full, input);

    // Metadata
    let (input, meta) = metadata_list(full, input)?;

    // Access and modifiers
    let (input, access_mod) = opt(access).parse(input)?;
    let (input, modifiers) = modifiers(input)?;

    // class keyword and name
    let (input, _) = context("[E0110] expected 'class' keyword", keyword("class")).parse(input)?;
    let (input, name) = context("[E0111] expected class name | help: provide a valid class name starting with uppercase", identifier).parse(input)?;

    // Type parameters
    let (input, type_params) = type_params(full, input)?;

    // Extends clause
    let (input, extends) = opt(preceded(
        context("[E0112] expected 'extends' keyword", keyword("extends")),
        context("[E0113] expected parent class type | help: provide the class to extend from", |i| type_expr(full, i))
    )).parse(input)?;

    // Implements clause: supports both `implements A implements B` (Haxe standard)
    // and `implements A, B` (comma-separated) syntax
    let (input, implements) = {
        let mut ifaces = Vec::new();
        let mut rest = input;
        while let Ok((next, _)) = keyword("implements").parse(rest) {
            let (next, iface_list) = context(
                "[E0115] expected interface type",
                separated_list1(symbol(","), |i| type_expr(full, i)),
            ).parse(next)?;
            ifaces.extend(iface_list);
            rest = next;
        }
        (rest, ifaces)
    };

    // Class body
    let (input, _) = context("[E0116] expected '{' to start class body | help: class body must be enclosed in braces", symbol("{")).parse(input)?;

    let (input, fields) = context("[E0117] expected class members | help: provide fields, methods, or properties inside the class body", |i| class_fields(full, i)).parse(input)?;

    let (input, _) = context("[E0118] expected '}' to close class body", symbol("}")).parse(input)?;

    let end = position(full, input);

    Ok((input, ClassDecl {
        meta,
        access: access_mod,
        modifiers,
        name,
        type_params,
        extends,
        implements,
        fields,
        span: Span::new(start, end),
    }))
    }).parse(input)
}

/// Parse interface declaration
pub fn interface_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, InterfaceDecl> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, access) = opt(access).parse(input)?;
    let (input, modifiers) = modifiers(input)?;

    let (input, _) = keyword("interface").parse(input)?;
    let (input, name) = identifier(input)?;
    let (input, type_params) = type_params(full, input)?;

    // Multiple extends allowed for interfaces
    let (input, extends) = opt(preceded(
        keyword("extends"),
        separated_list1(symbol(","), |i| type_expr(full, i)),
    ))
    .parse(input)?;

    let (input, _) = symbol("{").parse(input)?;
    let (input, fields) = class_fields(full, input)?;
    let (input, _) = symbol("}").parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        InterfaceDecl {
            meta,
            access,
            modifiers,
            name,
            type_params,
            extends: extends.unwrap_or_default(),
            fields,
            span: Span::new(start, end),
        },
    ))
}

/// Parse enum declaration
pub fn enum_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, EnumDecl> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, access) = opt(access).parse(input)?;

    let (input, _) = keyword("enum").parse(input)?;
    let (input, name) = identifier(input)?;
    let (input, type_params) = type_params(full, input)?;

    let (input, _) = symbol("{").parse(input)?;
    let (input, constructors) =
        separated_list0(symbol(";"), |i| enum_constructor(full, i)).parse(input)?;
    let (input, _) = opt(symbol(";")).parse(input)?; // Trailing semicolon
    let (input, _) = symbol("}").parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        EnumDecl {
            meta,
            access,
            name,
            type_params,
            constructors,
            span: Span::new(start, end),
        },
    ))
}

/// Parse enum constructor
fn enum_constructor<'a>(full: &'a str, input: &'a str) -> PResult<'a, EnumConstructor> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, name) = identifier(input)?;

    // Optional parameters
    let (input, params) = opt(delimited(
        symbol("("),
        separated_list1(symbol(","), |i| function_param(full, i)),
        symbol(")"),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        EnumConstructor {
            meta,
            name,
            params: params.unwrap_or_default(),
            span: Span::new(start, end),
        },
    ))
}

/// Parse typedef declaration
pub fn typedef_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, TypedefDecl> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, access) = opt(access).parse(input)?;

    let (input, _) = keyword("typedef").parse(input)?;
    let (input, name) = identifier(input)?;
    let (input, type_params) = type_params(full, input)?;

    let (input, _) = symbol("=").parse(input)?;
    let (input, type_def) = type_expr(full, input)?;

    // Semicolon is optional for anonymous structures, required for other types
    let input = match &type_def {
        Type::Anonymous { .. } => {
            // Anonymous structures don't require semicolons
            let (input, _) = opt(symbol(";")).parse(input)?;
            input
        }
        _ => {
            // Other types (function types, paths) require semicolons
            let (input, _) = symbol(";").parse(input)?;
            input
        }
    };

    let end = position(full, input);

    Ok((
        input,
        TypedefDecl {
            meta,
            access,
            name,
            type_params,
            type_def,
            span: Span::new(start, end),
        },
    ))
}

/// Parse abstract declaration
pub fn abstract_decl<'a>(full: &'a str, input: &'a str) -> PResult<'a, AbstractDecl> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, access) = opt(access).parse(input)?;
    let (input, modifiers) = modifiers(input)?;

    // Check for optional "enum" keyword before "abstract" (enum abstract syntax)
    let (input, is_enum_abstract) = opt(keyword("enum")).parse(input)?;

    let (input, _) = keyword("abstract").parse(input)?;
    let (input, name) = identifier(input)?;
    let (input, type_params) = type_params(full, input)?;

    // Underlying type parsing depends on abstract type:
    // - @:coreType abstracts: no underlying type
    // - enum abstract: underlying type in parentheses (required)
    // - regular abstract: underlying type in parentheses (required)
    let has_core_type = meta.iter().any(|m| m.name == "coreType");
    let (input, underlying) = if has_core_type {
        // @:coreType abstracts don't need an underlying type
        (input, None)
    } else if is_enum_abstract.is_some() {
        // enum abstract requires underlying type in parentheses
        // Example: enum abstract XmlType(Int)
        let (input, _) = symbol("(").parse(input)?;
        let (input, underlying) = type_expr(full, input)?;
        let (input, _) = symbol(")").parse(input)?;
        (input, Some(underlying))
    } else {
        // Regular abstracts require an underlying type in parentheses
        let (input, _) = symbol("(").parse(input)?;
        let (input, underlying) = type_expr(full, input)?;
        let (input, _) = symbol(")").parse(input)?;
        (input, Some(underlying))
    };

    // From/to clauses
    let (input, from) = many0(preceded(keyword("from"), |i| type_expr(full, i))).parse(input)?;

    let (input, to) = many0(preceded(keyword("to"), |i| type_expr(full, i))).parse(input)?;

    // Body (optional)
    let (input, fields) = alt((
        // With body
        delimited(symbol("{"), |i| class_fields(full, i), symbol("}")),
        // Without body
        value(vec![], symbol(";")),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        AbstractDecl {
            meta,
            access,
            modifiers,
            name,
            type_params,
            underlying,
            from,
            to,
            fields,
            is_enum_abstract: is_enum_abstract.is_some(),
            span: Span::new(start, end),
        },
    ))
}

/// Parse class fields with conditional compilation support
fn class_fields<'a>(full: &'a str, input: &'a str) -> PResult<'a, Vec<ClassField>> {
    let mut result = Vec::new();
    let mut current_input = input;

    loop {
        // Skip whitespace
        let (input, _) = ws(current_input)?;
        current_input = input;

        // Check for end of fields
        if current_input.is_empty() || current_input.starts_with('}') {
            break;
        }

        // Check for conditional compilation
        let peek_result: Result<_, nom::Err<nom::error::Error<_>>> =
            peek(tag("#if")).parse(current_input);
        if peek_result.is_ok() {
            // Parse conditional compilation
            let (input, conditional) =
                crate::haxe_parser::conditional_compilation(full, current_input, class_field)?;
            // Flatten the conditional fields
            fn flatten_conditional_fields(
                conditional: crate::haxe_ast::ConditionalCompilation<ClassField>,
            ) -> Vec<ClassField> {
                let mut fields = Vec::new();
                fields.extend(conditional.if_branch.content);
                for branch in conditional.elseif_branches {
                    fields.extend(branch.content);
                }
                if let Some(else_content) = conditional.else_branch {
                    fields.extend(else_content);
                }
                fields
            }
            result.extend(flatten_conditional_fields(conditional));
            current_input = input;
        } else {
            // Parse regular field

            match class_field(full, current_input) {
                Ok((input, field)) => {
                    result.push(field);
                    current_input = input;
                }
                Err(_) => {
                    break;
                }
            }
        }
    }

    Ok((current_input, result))
}

/// Parse access specifiers and modifiers in any order
pub fn parse_access_and_modifiers(input: &str) -> PResult<(Option<Access>, Vec<Modifier>)> {
    let mut access_spec = None;
    let mut modifiers_list = Vec::new();
    let mut current_input = input;

    loop {
        let (input, _) = ws(current_input)?;
        current_input = input;

        // Try to parse access specifier
        if access_spec.is_none() {
            if let Ok((rest, access_val)) = access(current_input) {
                access_spec = Some(access_val);
                current_input = rest;
                continue;
            }
        }

        // Try to parse modifiers
        if let Ok((rest, modifier)) = alt((
            value(Modifier::Static, keyword("static")),
            value(Modifier::Inline, keyword("inline")),
            value(Modifier::Macro, keyword("macro")),
            value(Modifier::Dynamic, keyword("dynamic")),
            value(Modifier::Override, keyword("override")),
            value(Modifier::Final, keyword("final")),
            value(Modifier::Extern, keyword("extern")),
        ))
        .parse(current_input)
        {
            modifiers_list.push(modifier);
            current_input = rest;
            continue;
        }

        // No more access specifiers or modifiers found
        break;
    }

    Ok((current_input, (access_spec, modifiers_list)))
}

/// Parse class field
fn class_field<'a>(full: &'a str, input: &'a str) -> PResult<'a, ClassField> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;
    let (input, (access, modifiers)) = parse_access_and_modifiers(input)?;

    // Check if final was parsed as a modifier
    let has_final_modifier = modifiers.iter().any(|m| matches!(m, Modifier::Final));

    // Field kind
    // NOTE: field_property MUST come before field_var_or_final because both parse
    // "var identifier" but field_property expects "(accessor, accessor)" after,
    // while field_var_or_final expects ":" or "=" or ";" after.
    // If field_var_or_final is tried first, it will consume "var identifier" and
    // then fail on "(", and field_property won't get a chance to try.
    let (input, kind) = alt((
        |i| field_function(full, i),
        |i| field_property(full, i),
        |i| field_var_or_final(full, i, has_final_modifier),
    ))
    .parse(input)?;

    let end = position(full, input);

    Ok((
        input,
        ClassField {
            meta,
            access,
            modifiers,
            kind,
            span: Span::new(start, end),
        },
    ))
}

/// Parse interface field (similar to class field but more restricted)
#[allow(dead_code)]
fn interface_field<'a>(full: &'a str, input: &'a str) -> PResult<'a, ClassField> {
    class_field(full, input)
}

/// Parse function field
fn field_function<'a>(full: &'a str, input: &'a str) -> PResult<'a, ClassFieldKind> {
    let (input, _) =
        context("[E0120] expected 'function' keyword", keyword("function")).parse(input)?;
    let (input, name) = context(
        "[E0121] expected function name | help: provide a valid function name",
        function_name,
    )
    .parse(input)?;

    let (input, type_params) = type_params(full, input)?;

    let (input, _) = context("[E0122] expected '(' to start parameter list | help: function parameters must be enclosed in parentheses", symbol("(")).parse(input)?;
    let (input, params) = context(
        "[E0123] expected function parameters | help: provide parameter list or leave empty",
        separated_list0(symbol(","), |i| function_param(full, i)),
    )
    .parse(input)?;
    let (input, _) = opt(symbol(",")).parse(input)?; // Trailing comma
    let (input, _) =
        context("[E0124] expected ')' to close parameter list", symbol(")")).parse(input)?;

    let (input, return_type) = opt(preceded(
        context("[E0125] expected ':' before return type", symbol(":")),
        |i| type_expr(full, i),
    ))
    .parse(input)?;

    // Body is optional (for interfaces and extern functions)
    let (input, body) = opt(|i| function_body(full, i)).parse(input)?;

    // Handle semicolon consumption based on body type
    let input = match &body {
        Some(expr) => {
            // Check if expression is brace-terminated (block, if, switch, untyped, etc.)
            fn is_brace_terminated(kind: &ExprKind) -> bool {
                matches!(
                    kind,
                    ExprKind::Block(_)
                        | ExprKind::If { .. }
                        | ExprKind::Switch { .. }
                        | ExprKind::For { .. }
                        | ExprKind::While { .. }
                        | ExprKind::DoWhile { .. }
                        | ExprKind::Try { .. }
                        | ExprKind::Untyped(_)
                )
            }

            let is_brace_term = is_brace_terminated(&expr.kind);

            // If body is a brace-terminated expression, no semicolon needed
            if is_brace_term {
                input
            } else {
                // Single expression body needs semicolon
                let (input, _) = context("[E0126] expected ';' after function expression | help: function expressions must end with semicolon", symbol(";")).parse(input)?;
                input
            }
        }
        None => {
            // No body, semicolon required (interface/extern functions)
            let (input, _) = context("[E0127] expected ';' after function declaration | help: function declarations without body must end with semicolon", symbol(";")).parse(input)?;
            input
        }
    };

    let span = Span::default(); // Will be set by parent

    Ok((
        input,
        ClassFieldKind::Function(Function {
            name: name.clone(),
            type_params,
            params,
            return_type,
            body: body.map(Box::new),
            span,
        }),
    ))
}

/// Parse var field or final field, with final modifier awareness
fn field_var_or_final<'a>(
    full: &'a str,
    input: &'a str,
    has_final_modifier: bool,
) -> PResult<'a, ClassFieldKind> {
    if has_final_modifier {
        // If final was already parsed as a modifier, expect an identifier (no keyword)
        map(
            tuple((
                context("[E0128] expected field name | help: provide a valid field name", identifier),
                opt(preceded(context("[E0129] expected ':' before type annotation", symbol(":")), |i| type_expr(full, i))),
                opt(preceded(context("[E0130] expected '=' before initializer", symbol("=")), |i| expression(full, i))),
                context("[E0131] expected ';' after final field declaration | help: field declarations must end with semicolon", symbol(";"))
            )),
            |(name, type_hint, expr, _)| ClassFieldKind::Final {
                name,
                type_hint,
                expr,
            }
        ).parse(input)
    } else {
        alt((
            // var field
            map(
                tuple((
                    context("[E0132] expected 'var' keyword", keyword("var")),
                    context("[E0133] expected field name | help: provide a valid field name", identifier),
                    opt(preceded(context("[E0134] expected ':' before type annotation", symbol(":")), |i| type_expr(full, i))),
                    opt(preceded(context("[E0135] expected '=' before initializer", symbol("=")), |i| expression(full, i))),
                    context("[E0136] expected ';' after variable field declaration | help: field declarations must end with semicolon", symbol(";"))
                )),
                |(_, name, type_hint, expr, _)| ClassFieldKind::Var {
                    name,
                    type_hint,
                    expr,
                }
            ),
            // final field (standalone)
            map(
                tuple((
                    context("[E0137] expected 'final' keyword", keyword("final")),
                    context("[E0138] expected field name | help: provide a valid field name", identifier),
                    opt(preceded(context("[E0139] expected ':' before type annotation", symbol(":")), |i| type_expr(full, i))),
                    opt(preceded(context("[E0140] expected '=' before initializer", symbol("=")), |i| expression(full, i))),
                    context("[E0141] expected ';' after final field declaration | help: field declarations must end with semicolon", symbol(";"))
                )),
                |(_, name, type_hint, expr, _)| ClassFieldKind::Final {
                    name,
                    type_hint,
                    expr,
                }
            ),
        )).parse(input)
    }
}

/// Parse property field
fn field_property<'a>(full: &'a str, input: &'a str) -> PResult<'a, ClassFieldKind> {
    let (input, _) = keyword("var").parse(input)?;
    let (input, name) = identifier(input)?;

    // Property accessors in parentheses
    let (input, _) = symbol("(").parse(input)?;
    let (input, getter) = property_access(input)?;
    let (input, _) = symbol(",").parse(input)?;
    let (input, setter) = property_access(input)?;
    let (input, _) = symbol(")").parse(input)?;

    let (input, type_hint) = opt(preceded(symbol(":"), |i| type_expr(full, i))).parse(input)?;
    let (input, _default_value) =
        opt(preceded(symbol("="), |i| expression(full, i))).parse(input)?;
    let (input, _) = symbol(";").parse(input)?;

    Ok((
        input,
        ClassFieldKind::Property {
            name,
            type_hint,
            getter,
            setter,
        },
    ))
}

/// Parse property access mode
fn property_access(input: &str) -> PResult<PropertyAccess> {
    alt((
        value(PropertyAccess::Default, keyword("default")),
        value(PropertyAccess::Null, keyword("null")),
        value(PropertyAccess::Never, keyword("never")),
        value(PropertyAccess::Dynamic, keyword("dynamic")),
        map(|i| identifier(i), PropertyAccess::Custom),
    ))
    .parse(input)
}

/// Parse function parameter
pub fn function_param<'a>(full: &'a str, input: &'a str) -> PResult<'a, FunctionParam> {
    let start = position(full, input);

    let (input, meta) = metadata_list(full, input)?;

    // Check for rest parameter: ...name
    let (input, rest) = opt(symbol("...")).parse(input)?;
    let is_rest = rest.is_some();

    let (input, optional) = if !is_rest {
        opt(symbol("?")).parse(input)?
    } else {
        (input, None) // Rest parameters can't be optional
    };

    let (input, name) = identifier(input)?;
    let (input, type_hint) = opt(preceded(symbol(":"), |i| type_expr(full, i))).parse(input)?;

    let (input, default_value) = if !is_rest {
        opt(preceded(symbol("="), |i| expression(full, i))).parse(input)?
    } else {
        (input, None) // Rest parameters can't have defaults
    };

    let end = position(full, input);

    Ok((
        input,
        FunctionParam {
            meta,
            name,
            type_hint,
            optional: optional.is_some(),
            rest: is_rest,
            default_value: default_value.map(Box::new),
            span: Span::new(start, end),
        },
    ))
}
