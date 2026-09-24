//! `haxe.macro.Expr` as macro values, and back.
//!
//! An `ExprDef` is `MacroValue::Enum("ExprDef", variant, args)`; a nested
//! `Expr` inside one stays a `MacroValue::Expr` until a macro reads its
//! `.expr`. Types are `ComplexType` enums over `TypePath` objects, as
//! haxe.macro declares them. `expr_of` accepts any of these forms (and an
//! `{expr, pos}` object) and rebuilds the parser AST.

use super::ast_bridge::span_to_location;
use super::value::MacroValue;
use parser::{
    AssignOp, BinaryOp, BlockElement, Case, Catch, Expr, ExprKind, Function, FunctionParam,
    Metadata, ObjectField, Pattern, Span, StringPart, Type, TypePath, UnaryOp,
};
use std::sync::Arc;

fn en(e: &str, v: &str, args: Vec<MacroValue>) -> MacroValue {
    MacroValue::Enum(Arc::from(e), Arc::from(v), Arc::new(args))
}

fn s(v: &str) -> MacroValue {
    MacroValue::String(Arc::from(v))
}

fn arr(items: Vec<MacroValue>) -> MacroValue {
    MacroValue::Array(Arc::new(items))
}

fn obj(fields: Vec<(&str, MacroValue)>) -> MacroValue {
    MacroValue::Object(Arc::new(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    ))
}

fn sub(e: &Expr) -> MacroValue {
    MacroValue::Expr(Arc::new(e.clone()))
}

fn opt_sub(e: Option<&Expr>) -> MacroValue {
    e.map(sub).unwrap_or(MacroValue::Null)
}

fn constant(v: &str, args: Vec<MacroValue>) -> MacroValue {
    en("ExprDef", "EConst", vec![en("Constant", v, args)])
}

fn binop_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "OpAdd",
        BinaryOp::Sub => "OpSub",
        BinaryOp::Mul => "OpMult",
        BinaryOp::Div => "OpDiv",
        BinaryOp::Mod => "OpMod",
        BinaryOp::Eq => "OpEq",
        BinaryOp::NotEq => "OpNotEq",
        BinaryOp::Lt => "OpLt",
        BinaryOp::Le => "OpLte",
        BinaryOp::Gt => "OpGt",
        BinaryOp::Ge => "OpGte",
        BinaryOp::And => "OpBoolAnd",
        BinaryOp::Or => "OpBoolOr",
        BinaryOp::BitAnd => "OpAnd",
        BinaryOp::BitOr => "OpOr",
        BinaryOp::BitXor => "OpXor",
        BinaryOp::Shl => "OpShl",
        BinaryOp::Shr => "OpShr",
        BinaryOp::Ushr => "OpUShr",
        BinaryOp::Range => "OpInterval",
        BinaryOp::Arrow => "OpArrow",
        BinaryOp::NullCoal => "OpNullCoal",
        BinaryOp::In => "OpIn",
        BinaryOp::Is => "OpEq",
    }
}

fn binop_of(name: &str) -> Option<BinaryOp> {
    Some(match name {
        "OpAdd" => BinaryOp::Add,
        "OpSub" => BinaryOp::Sub,
        "OpMult" => BinaryOp::Mul,
        "OpDiv" => BinaryOp::Div,
        "OpMod" => BinaryOp::Mod,
        "OpEq" => BinaryOp::Eq,
        "OpNotEq" => BinaryOp::NotEq,
        "OpLt" => BinaryOp::Lt,
        "OpLte" => BinaryOp::Le,
        "OpGt" => BinaryOp::Gt,
        "OpGte" => BinaryOp::Ge,
        "OpBoolAnd" => BinaryOp::And,
        "OpBoolOr" => BinaryOp::Or,
        "OpAnd" => BinaryOp::BitAnd,
        "OpOr" => BinaryOp::BitOr,
        "OpXor" => BinaryOp::BitXor,
        "OpShl" => BinaryOp::Shl,
        "OpShr" => BinaryOp::Shr,
        "OpUShr" => BinaryOp::Ushr,
        "OpInterval" => BinaryOp::Range,
        "OpArrow" => BinaryOp::Arrow,
        "OpNullCoal" => BinaryOp::NullCoal,
        "OpIn" => BinaryOp::In,
        _ => return None,
    })
}

fn assign_base(op: AssignOp) -> Option<BinaryOp> {
    Some(match op {
        AssignOp::Assign => return None,
        AssignOp::AddAssign => BinaryOp::Add,
        AssignOp::SubAssign => BinaryOp::Sub,
        AssignOp::MulAssign => BinaryOp::Mul,
        AssignOp::DivAssign => BinaryOp::Div,
        AssignOp::ModAssign => BinaryOp::Mod,
        AssignOp::AndAssign => BinaryOp::BitAnd,
        AssignOp::OrAssign => BinaryOp::BitOr,
        AssignOp::XorAssign => BinaryOp::BitXor,
        AssignOp::ShlAssign => BinaryOp::Shl,
        AssignOp::ShrAssign => BinaryOp::Shr,
        AssignOp::UshrAssign => BinaryOp::Ushr,
    })
}

fn assign_of(op: BinaryOp) -> Option<AssignOp> {
    Some(match op {
        BinaryOp::Add => AssignOp::AddAssign,
        BinaryOp::Sub => AssignOp::SubAssign,
        BinaryOp::Mul => AssignOp::MulAssign,
        BinaryOp::Div => AssignOp::DivAssign,
        BinaryOp::Mod => AssignOp::ModAssign,
        BinaryOp::BitAnd => AssignOp::AndAssign,
        BinaryOp::BitOr => AssignOp::OrAssign,
        BinaryOp::BitXor => AssignOp::XorAssign,
        BinaryOp::Shl => AssignOp::ShlAssign,
        BinaryOp::Shr => AssignOp::ShrAssign,
        BinaryOp::Ushr => AssignOp::UshrAssign,
        _ => return None,
    })
}

fn unop_name(op: UnaryOp) -> (&'static str, bool) {
    match op {
        UnaryOp::Not => ("OpNot", false),
        UnaryOp::Neg => ("OpNeg", false),
        UnaryOp::BitNot => ("OpNegBits", false),
        UnaryOp::PreIncr => ("OpIncrement", false),
        UnaryOp::PreDecr => ("OpDecrement", false),
        UnaryOp::PostIncr => ("OpIncrement", true),
        UnaryOp::PostDecr => ("OpDecrement", true),
    }
}

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------

fn type_path_value(path: &TypePath, params: &[Type]) -> MacroValue {
    obj(vec![
        ("pack", arr(path.package.iter().map(|p| s(p)).collect())),
        ("name", s(&path.name)),
        (
            "params",
            arr(params
                .iter()
                .map(|p| match p {
                    Type::Const { value, .. } => en("TypeParam", "TPExpr", vec![sub(value)]),
                    other => en("TypeParam", "TPType", vec![complex_type_of(other)]),
                })
                .collect()),
        ),
        (
            "sub",
            path.sub.as_deref().map(s).unwrap_or(MacroValue::Null),
        ),
    ])
}

/// A parser type as a `ComplexType` value.
pub fn complex_type_of(t: &Type) -> MacroValue {
    match t {
        Type::Path { path, params, .. } => {
            en("ComplexType", "TPath", vec![type_path_value(path, params)])
        }
        Type::Function { params, ret, .. } => en(
            "ComplexType",
            "TFunction",
            vec![
                arr(params.iter().map(complex_type_of).collect()),
                complex_type_of(ret),
            ],
        ),
        Type::Anonymous { fields, .. } => en(
            "ComplexType",
            "TAnonymous",
            vec![arr(fields
                .iter()
                .map(|f| {
                    let mut meta = Vec::new();
                    if f.optional {
                        meta.push(obj(vec![
                            ("name", s(":optional")),
                            ("params", arr(Vec::new())),
                            ("pos", MacroValue::Position(span_to_location(f.span))),
                        ]));
                    }
                    obj(vec![
                        ("name", s(&f.name)),
                        (
                            "kind",
                            en(
                                "FieldType",
                                "FVar",
                                vec![complex_type_of(&f.type_hint), MacroValue::Null],
                            ),
                        ),
                        ("access", arr(Vec::new())),
                        ("meta", arr(meta)),
                        ("pos", MacroValue::Position(span_to_location(f.span))),
                    ])
                })
                .collect())],
        ),
        Type::Optional { inner, .. } => {
            en("ComplexType", "TOptional", vec![complex_type_of(inner)])
        }
        Type::Parenthesis { inner, .. } => {
            en("ComplexType", "TParent", vec![complex_type_of(inner)])
        }
        Type::Intersection { left, right, .. } => en(
            "ComplexType",
            "TIntersection",
            vec![arr(vec![complex_type_of(left), complex_type_of(right)])],
        ),
        Type::Wildcard { .. } => en(
            "ComplexType",
            "TPath",
            vec![obj(vec![
                ("pack", arr(Vec::new())),
                ("name", s("_")),
                ("params", arr(Vec::new())),
                ("sub", MacroValue::Null),
            ])],
        ),
        Type::Const { value, .. } => en(
            "ComplexType",
            "TPath",
            vec![obj(vec![
                ("pack", arr(Vec::new())),
                ("name", s(&format!("{:?}", value.kind))),
                ("params", arr(Vec::new())),
                ("sub", MacroValue::Null),
            ])],
        ),
    }
}

fn field<'v>(v: &'v MacroValue, name: &str) -> Option<&'v MacroValue> {
    match v {
        MacroValue::Object(m) => m.get(name),
        _ => None,
    }
}

fn string_of(v: Option<&MacroValue>) -> Option<String> {
    match v? {
        MacroValue::String(s) => Some(s.to_string()),
        _ => None,
    }
}

fn list_of(v: Option<&MacroValue>) -> Vec<MacroValue> {
    match v {
        Some(MacroValue::Array(a)) => a.as_ref().clone(),
        _ => Vec::new(),
    }
}

fn type_path_of(v: &MacroValue, span: Span) -> Option<(TypePath, Vec<Type>)> {
    let path = TypePath {
        package: list_of(field(v, "pack"))
            .iter()
            .filter_map(|p| string_of(Some(p)))
            .collect(),
        name: string_of(field(v, "name"))?,
        sub: string_of(field(v, "sub")),
    };
    let params = list_of(field(v, "params"))
        .iter()
        .filter_map(|p| match p {
            MacroValue::Enum(_, variant, args) if &**variant == "TPType" => {
                args.first().and_then(|t| type_of_value(t, span))
            }
            MacroValue::Enum(_, variant, args) if &**variant == "TPExpr" => {
                args.first().map(|e| Type::Const {
                    value: Box::new(expr_of(e, span)),
                    span,
                })
            }
            other => type_of_value(other, span),
        })
        .collect();
    Some((path, params))
}

/// A `ComplexType` value (or the older `{kind: "TPath", ...}` object) as a
/// parser type.
pub fn type_of_value(v: &MacroValue, span: Span) -> Option<Type> {
    match v {
        MacroValue::Enum(e, variant, args) if &**e == "ComplexType" => {
            match (&**variant, args.as_slice()) {
                ("TPath", [p]) => {
                    let (path, params) = type_path_of(p, span)?;
                    Some(Type::Path { path, params, span })
                }
                ("TFunction", [ps, ret]) => Some(Type::Function {
                    params: list_of(Some(ps))
                        .iter()
                        .filter_map(|p| type_of_value(p, span))
                        .collect(),
                    ret: Box::new(type_of_value(ret, span)?),
                    span,
                }),
                ("TOptional", [inner]) => Some(Type::Optional {
                    inner: Box::new(type_of_value(inner, span)?),
                    span,
                }),
                ("TParent", [inner]) => Some(Type::Parenthesis {
                    inner: Box::new(type_of_value(inner, span)?),
                    span,
                }),
                ("TAnonymous", [fields]) => Some(Type::Anonymous {
                    fields: list_of(Some(fields))
                        .iter()
                        .filter_map(|f| {
                            let name = string_of(field(f, "name"))?;
                            let kind = field(f, "kind")?;
                            let hint = match kind {
                                MacroValue::Enum(_, v, a) if &**v == "FVar" => {
                                    a.first().and_then(|t| type_of_value(t, span))
                                }
                                _ => None,
                            }?;
                            let optional = list_of(field(f, "meta")).iter().any(|m| {
                                string_of(field(m, "name")).as_deref() == Some(":optional")
                            });
                            Some(parser::AnonField {
                                name,
                                optional,
                                type_hint: hint,
                                span,
                            })
                        })
                        .collect(),
                    span,
                }),
                _ => None,
            }
        }
        // The object form a constructor call builds: `{kind, <payload fields>}`
        // for one object argument, `{kind, args}` otherwise.
        MacroValue::Object(m) => {
            let Some(MacroValue::String(kind)) = m.get("kind") else {
                return None;
            };
            if m.contains_key("name") {
                let (path, params) = type_path_of(v, span)?;
                let path_type = Type::Path { path, params, span };
                return Some(match &**kind {
                    "TOptional" => Type::Optional {
                        inner: Box::new(path_type),
                        span,
                    },
                    "TParent" => Type::Parenthesis {
                        inner: Box::new(path_type),
                        span,
                    },
                    _ => path_type,
                });
            }
            let args = list_of(m.get("args"));
            type_of_value(
                &MacroValue::Enum(Arc::from("ComplexType"), Arc::from(&**kind), Arc::new(args)),
                span,
            )
        }
        _ => None,
    }
}

// ----------------------------------------------------------------------
// Patterns <-> expressions (switch case values)
// ----------------------------------------------------------------------

/// A case pattern written as the expression haxe keeps in `Case.values`.
pub fn pattern_to_expr(p: &Pattern, span: Span) -> Expr {
    let mk = |kind| Expr { kind, span };
    match p {
        Pattern::Const(e) => e.clone(),
        Pattern::Var(n) => mk(ExprKind::Ident(n.clone())),
        Pattern::Underscore => mk(ExprKind::Ident("_".to_string())),
        Pattern::Null => mk(ExprKind::Null),
        Pattern::Constructor { path, params } => {
            let mut callee = None;
            for part in path.package.iter().chain(std::iter::once(&path.name)) {
                callee = Some(match callee {
                    None => mk(ExprKind::Ident(part.clone())),
                    Some(base) => mk(ExprKind::Field {
                        expr: Box::new(base),
                        field: part.clone(),
                        is_optional: false,
                    }),
                });
            }
            mk(ExprKind::Call {
                expr: Box::new(callee.expect("a constructor has a name")),
                args: params.iter().map(|p| pattern_to_expr(p, span)).collect(),
            })
        }
        Pattern::Array(items) => mk(ExprKind::Array(
            items.iter().map(|p| pattern_to_expr(p, span)).collect(),
        )),
        Pattern::ArrayRest { elements, rest } => {
            let mut items: Vec<Expr> = elements.iter().map(|p| pattern_to_expr(p, span)).collect();
            if let Some(r) = rest {
                items.push(mk(ExprKind::Spread(Box::new(mk(ExprKind::Ident(
                    r.clone(),
                ))))));
            }
            mk(ExprKind::Array(items))
        }
        Pattern::Object { fields } => mk(ExprKind::Object(
            fields
                .iter()
                .map(|(n, p)| ObjectField {
                    name: n.clone(),
                    expr: pattern_to_expr(p, span),
                    span,
                })
                .collect(),
        )),
        Pattern::Type { var, type_hint } => mk(ExprKind::TypeCheck {
            expr: Box::new(mk(ExprKind::Ident(var.clone()))),
            type_hint: type_hint.clone(),
        }),
        Pattern::Or(alts) => alts
            .iter()
            .map(|p| pattern_to_expr(p, span))
            .reduce(|a, b| {
                mk(ExprKind::Binary {
                    left: Box::new(a),
                    op: BinaryOp::BitOr,
                    right: Box::new(b),
                })
            })
            .unwrap_or_else(|| mk(ExprKind::Ident("_".to_string()))),
        Pattern::Extractor { expr, value } => mk(ExprKind::Binary {
            left: expr.clone(),
            op: BinaryOp::Arrow,
            right: Box::new(pattern_to_expr(value, span)),
        }),
        Pattern::Bind { name, pattern } => mk(ExprKind::Assign {
            left: Box::new(mk(ExprKind::Ident(name.clone()))),
            op: AssignOp::Assign,
            right: Box::new(pattern_to_expr(pattern, span)),
        }),
    }
}

/// An expression in pattern position read as that pattern.
pub fn expr_to_pattern(e: &Expr) -> Pattern {
    match &e.kind {
        ExprKind::Ident(n) if n == "_" => Pattern::Underscore,
        ExprKind::Ident(n) => Pattern::Var(n.clone()),
        ExprKind::Null => Pattern::Null,
        ExprKind::Paren(inner) => expr_to_pattern(inner),
        ExprKind::Array(items) => Pattern::Array(items.iter().map(expr_to_pattern).collect()),
        ExprKind::Binary {
            left,
            op: BinaryOp::BitOr,
            right,
        } => {
            let mut alts = Vec::new();
            for side in [left, right] {
                match expr_to_pattern(side) {
                    Pattern::Or(inner) => alts.extend(inner),
                    other => alts.push(other),
                }
            }
            Pattern::Or(alts)
        }
        ExprKind::Binary {
            left,
            op: BinaryOp::Arrow,
            right,
        } => Pattern::Extractor {
            expr: left.clone(),
            value: Box::new(expr_to_pattern(right)),
        },
        ExprKind::Assign {
            left,
            op: AssignOp::Assign,
            right,
        } if matches!(left.kind, ExprKind::Ident(_)) => {
            let ExprKind::Ident(name) = &left.kind else {
                unreachable!()
            };
            Pattern::Bind {
                name: name.clone(),
                pattern: Box::new(expr_to_pattern(right)),
            }
        }
        ExprKind::Call { expr: callee, args } => {
            let mut parts = Vec::new();
            let mut cursor = callee.as_ref();
            loop {
                match &cursor.kind {
                    ExprKind::Ident(n) => {
                        parts.push(n.clone());
                        break;
                    }
                    ExprKind::Field { expr, field, .. } => {
                        parts.push(field.clone());
                        cursor = expr;
                    }
                    _ => return Pattern::Const(e.clone()),
                }
            }
            parts.reverse();
            let name = parts.pop().unwrap_or_default();
            Pattern::Constructor {
                path: TypePath {
                    package: parts,
                    name,
                    sub: None,
                },
                params: args.iter().map(expr_to_pattern).collect(),
            }
        }
        _ => Pattern::Const(e.clone()),
    }
}

// ----------------------------------------------------------------------
// Expressions → ExprDef
// ----------------------------------------------------------------------

fn function_value(func: &Function, kind: MacroValue) -> MacroValue {
    let args = func.params.iter().map(arg_value).collect();
    let f = obj(vec![
        ("args", arr(args)),
        (
            "ret",
            func.return_type
                .as_ref()
                .map(complex_type_of)
                .unwrap_or(MacroValue::Null),
        ),
        ("expr", opt_sub(func.body.as_deref())),
        (
            "params",
            arr(func
                .type_params
                .iter()
                .map(|tp| obj(vec![("name", s(&tp.name))]))
                .collect()),
        ),
    ]);
    en("ExprDef", "EFunction", vec![kind, f])
}

fn arg_value(p: &FunctionParam) -> MacroValue {
    obj(vec![
        ("name", s(&p.name)),
        ("opt", MacroValue::Bool(p.optional)),
        (
            "type",
            p.type_hint
                .as_ref()
                .map(complex_type_of)
                .unwrap_or(MacroValue::Null),
        ),
        ("value", opt_sub(p.default_value.as_deref())),
        ("meta", arr(Vec::new())),
    ])
}

fn var_value(name: &str, hint: Option<&Type>, init: Option<&Expr>, is_final: bool) -> MacroValue {
    obj(vec![
        ("name", s(name)),
        (
            "type",
            hint.map(complex_type_of).unwrap_or(MacroValue::Null),
        ),
        ("expr", opt_sub(init)),
        ("isFinal", MacroValue::Bool(is_final)),
    ])
}

fn metadata_value(m: &Metadata) -> MacroValue {
    let name = if m.compile_time {
        format!(":{}", m.name.trim_start_matches(':'))
    } else {
        m.name.clone()
    };
    obj(vec![
        ("name", s(&name)),
        ("params", arr(m.params.iter().map(sub).collect())),
        ("pos", MacroValue::Position(span_to_location(m.span))),
    ])
}

/// An expression's `ExprDef`; `None` for a form haxe.macro has no
/// constructor for (the caller keeps it opaque).
pub fn def_of(kind: &ExprKind, span: Span) -> Option<MacroValue> {
    let mk = |kind| Expr { kind, span };
    Some(match kind {
        ExprKind::Int(i) => constant("CInt", vec![s(&i.to_string()), MacroValue::Null]),
        ExprKind::Float(f) => {
            let text = if f.fract() == 0.0 && f.is_finite() {
                format!("{:.1}", f)
            } else {
                f.to_string()
            };
            constant("CFloat", vec![s(&text), MacroValue::Null])
        }
        ExprKind::String(v) => constant(
            "CString",
            vec![s(v), en("StringLiteralKind", "DoubleQuotes", Vec::new())],
        ),
        // A single-quoted string with interpolations, kept as written.
        ExprKind::StringInterpolation(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    StringPart::Literal(l) => text.push_str(l),
                    StringPart::Interpolation(e) => match &e.kind {
                        ExprKind::Ident(n) => {
                            text.push('$');
                            text.push_str(n);
                        }
                        _ => text.push_str("${...}"),
                    },
                }
            }
            constant(
                "CString",
                vec![
                    s(&text),
                    en("StringLiteralKind", "SingleQuotes", Vec::new()),
                ],
            )
        }
        ExprKind::Bool(b) => constant("CIdent", vec![s(if *b { "true" } else { "false" })]),
        ExprKind::Null => constant("CIdent", vec![s("null")]),
        ExprKind::This => constant("CIdent", vec![s("this")]),
        ExprKind::Super => constant("CIdent", vec![s("super")]),
        ExprKind::Ident(n) => constant("CIdent", vec![s(n)]),
        ExprKind::Regex { pattern, flags } => constant("CRegexp", vec![s(pattern), s(flags)]),
        ExprKind::Field {
            expr,
            field,
            is_optional,
        } => en(
            "ExprDef",
            "EField",
            vec![
                sub(expr),
                s(field),
                en(
                    "EFieldKind",
                    if *is_optional { "Safe" } else { "Normal" },
                    Vec::new(),
                ),
            ],
        ),
        ExprKind::Index { expr, index } => en("ExprDef", "EArray", vec![sub(expr), sub(index)]),
        ExprKind::Call { expr, args } => en(
            "ExprDef",
            "ECall",
            vec![sub(expr), arr(args.iter().map(sub).collect())],
        ),
        ExprKind::New {
            type_path,
            params,
            args,
        } => en(
            "ExprDef",
            "ENew",
            vec![
                type_path_value(type_path, params),
                arr(args.iter().map(sub).collect()),
            ],
        ),
        ExprKind::Unary { op, expr } => {
            let (name, postfix) = unop_name(*op);
            en(
                "ExprDef",
                "EUnop",
                vec![
                    en("Unop", name, Vec::new()),
                    MacroValue::Bool(postfix),
                    sub(expr),
                ],
            )
        }
        ExprKind::Binary {
            left,
            op: BinaryOp::Is,
            right,
        } => {
            let hint = type_of_is_operand(right)?;
            en("ExprDef", "EIs", vec![sub(left), complex_type_of(&hint)])
        }
        ExprKind::Binary { left, op, right } => en(
            "ExprDef",
            "EBinop",
            vec![
                en("Binop", binop_name(*op), Vec::new()),
                sub(left),
                sub(right),
            ],
        ),
        ExprKind::Assign { left, op, right } => {
            let binop = match assign_base(*op) {
                None => en("Binop", "OpAssign", Vec::new()),
                Some(base) => en(
                    "Binop",
                    "OpAssignOp",
                    vec![en("Binop", binop_name(base), Vec::new())],
                ),
            };
            en("ExprDef", "EBinop", vec![binop, sub(left), sub(right)])
        }
        ExprKind::Ternary {
            cond,
            then_expr,
            else_expr,
        } => en(
            "ExprDef",
            "ETernary",
            vec![sub(cond), sub(then_expr), sub(else_expr)],
        ),
        ExprKind::Array(items) => en(
            "ExprDef",
            "EArrayDecl",
            vec![arr(items.iter().map(sub).collect())],
        ),
        ExprKind::Map(pairs) => en(
            "ExprDef",
            "EArrayDecl",
            vec![arr(pairs
                .iter()
                .map(|(k, v)| {
                    sub(&mk(ExprKind::Binary {
                        left: Box::new(k.clone()),
                        op: BinaryOp::Arrow,
                        right: Box::new(v.clone()),
                    }))
                })
                .collect())],
        ),
        ExprKind::Object(fields) => en(
            "ExprDef",
            "EObjectDecl",
            vec![arr(fields
                .iter()
                .map(|f| {
                    obj(vec![
                        ("field", s(&f.name)),
                        ("expr", sub(&f.expr)),
                        ("quotes", en("QuoteStatus", "Unquoted", Vec::new())),
                    ])
                })
                .collect())],
        ),
        ExprKind::Block(elements) => en(
            "ExprDef",
            "EBlock",
            vec![arr(elements
                .iter()
                .filter_map(|el| match el {
                    BlockElement::Expr(e) => Some(sub(e)),
                    _ => None,
                })
                .collect())],
        ),
        ExprKind::Var {
            name,
            type_hint,
            expr,
        } => en(
            "ExprDef",
            "EVars",
            vec![arr(vec![var_value(
                name,
                type_hint.as_ref(),
                expr.as_deref(),
                false,
            )])],
        ),
        ExprKind::Final {
            name,
            type_hint,
            expr,
        } => en(
            "ExprDef",
            "EVars",
            vec![arr(vec![var_value(
                name,
                type_hint.as_ref(),
                expr.as_deref(),
                true,
            )])],
        ),
        ExprKind::Function(func) => {
            let kind = if func.name.is_empty() {
                en("FunctionKind", "FAnonymous", Vec::new())
            } else {
                en(
                    "FunctionKind",
                    "FNamed",
                    vec![s(&func.name), MacroValue::Bool(false)],
                )
            };
            function_value(func, kind)
        }
        // `x -> e`: haxe keeps the body as `@:implicitReturn return e`.
        ExprKind::Arrow { params, expr } => {
            let body = mk(ExprKind::Meta {
                meta: Metadata {
                    name: "implicitReturn".to_string(),
                    params: Vec::new(),
                    span,
                    compile_time: true,
                },
                expr: Box::new(mk(ExprKind::Return(Some(expr.clone())))),
            });
            let func = Function {
                name: String::new(),
                type_params: Vec::new(),
                params: params
                    .iter()
                    .map(|p| FunctionParam {
                        meta: Vec::new(),
                        name: p.name.clone(),
                        type_hint: p.type_hint.clone(),
                        optional: false,
                        rest: false,
                        default_value: None,
                        span,
                    })
                    .collect(),
                return_type: None,
                body: Some(Box::new(body)),
                span,
            };
            function_value(&func, en("FunctionKind", "FArrow", Vec::new()))
        }
        ExprKind::Return(value) => en("ExprDef", "EReturn", vec![opt_sub(value.as_deref())]),
        ExprKind::Break => en("ExprDef", "EBreak", Vec::new()),
        ExprKind::Continue => en("ExprDef", "EContinue", Vec::new()),
        ExprKind::Throw(e) => en("ExprDef", "EThrow", vec![sub(e)]),
        ExprKind::Untyped(e) => en("ExprDef", "EUntyped", vec![sub(e)]),
        ExprKind::Paren(e) => en("ExprDef", "EParenthesis", vec![sub(e)]),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => en(
            "ExprDef",
            "EIf",
            vec![sub(cond), sub(then_branch), opt_sub(else_branch.as_deref())],
        ),
        ExprKind::While { cond, body } => en(
            "ExprDef",
            "EWhile",
            vec![sub(cond), sub(body), MacroValue::Bool(true)],
        ),
        ExprKind::DoWhile { body, cond } => en(
            "ExprDef",
            "EWhile",
            vec![sub(cond), sub(body), MacroValue::Bool(false)],
        ),
        ExprKind::For {
            var,
            key_var,
            iter,
            body,
        } => en(
            "ExprDef",
            "EFor",
            vec![
                sub(&for_head(var, key_var.as_deref(), iter, span)),
                sub(body),
            ],
        ),
        ExprKind::ArrayComprehension { for_parts, expr } => {
            let mut inner = (**expr).clone();
            for part in for_parts.iter().rev() {
                inner = mk(ExprKind::For {
                    var: part.var.clone(),
                    key_var: part.key_var.clone(),
                    iter: Box::new(part.iter.clone()),
                    body: Box::new(inner),
                });
            }
            en("ExprDef", "EArrayDecl", vec![arr(vec![sub(&inner)])])
        }
        ExprKind::Switch {
            expr,
            cases,
            default,
        } => en(
            "ExprDef",
            "ESwitch",
            vec![
                sub(expr),
                arr(cases
                    .iter()
                    .map(|c| {
                        obj(vec![
                            (
                                "values",
                                arr(c
                                    .patterns
                                    .iter()
                                    .map(|p| sub(&pattern_to_expr(p, c.span)))
                                    .collect()),
                            ),
                            ("guard", opt_sub(c.guard.as_ref())),
                            ("expr", sub(&c.body)),
                        ])
                    })
                    .collect()),
                opt_sub(default.as_deref()),
            ],
        ),
        ExprKind::Try { expr, catches, .. } => en(
            "ExprDef",
            "ETry",
            vec![
                sub(expr),
                arr(catches
                    .iter()
                    .map(|c| {
                        obj(vec![
                            ("name", s(&c.var)),
                            (
                                "type",
                                c.type_hint
                                    .as_ref()
                                    .map(complex_type_of)
                                    .unwrap_or(MacroValue::Null),
                            ),
                            ("expr", sub(&c.body)),
                        ])
                    })
                    .collect()),
            ],
        ),
        ExprKind::Cast { expr, type_hint } => en(
            "ExprDef",
            "ECast",
            vec![
                sub(expr),
                type_hint
                    .as_ref()
                    .map(complex_type_of)
                    .unwrap_or(MacroValue::Null),
            ],
        ),
        ExprKind::TypeCheck { expr, type_hint } => en(
            "ExprDef",
            "ECheckType",
            vec![sub(expr), complex_type_of(type_hint)],
        ),
        ExprKind::Meta { meta, expr } => {
            en("ExprDef", "EMeta", vec![metadata_value(meta), sub(expr)])
        }
        _ => return None,
    })
}

/// `k => v in iter` / `v in iter`, the head haxe keeps in `EFor`.
fn for_head(var: &str, key_var: Option<&str>, iter: &Expr, span: Span) -> Expr {
    let mk = |kind| Expr { kind, span };
    let binder = match key_var {
        Some(k) => mk(ExprKind::Binary {
            left: Box::new(mk(ExprKind::Ident(k.to_string()))),
            op: BinaryOp::Arrow,
            right: Box::new(mk(ExprKind::Ident(var.to_string()))),
        }),
        None => mk(ExprKind::Ident(var.to_string())),
    };
    mk(ExprKind::Binary {
        left: Box::new(binder),
        op: BinaryOp::In,
        right: Box::new(iter.clone()),
    })
}

/// The type named on the right of `e is T`.
fn type_of_is_operand(e: &Expr) -> Option<Type> {
    let mut parts = Vec::new();
    let mut cursor = e;
    loop {
        match &cursor.kind {
            ExprKind::Ident(n) => {
                parts.push(n.clone());
                break;
            }
            ExprKind::Field { expr, field, .. } => {
                parts.push(field.clone());
                cursor = expr;
            }
            _ => return None,
        }
    }
    parts.reverse();
    let name = parts.pop()?;
    Some(Type::Path {
        path: TypePath {
            package: parts,
            name,
            sub: None,
        },
        params: Vec::new(),
        span: e.span,
    })
}

// ----------------------------------------------------------------------
// ExprDef → expressions
// ----------------------------------------------------------------------

fn exprs_of(v: Option<&MacroValue>, span: Span) -> Vec<Expr> {
    list_of(v).iter().map(|e| expr_of(e, span)).collect()
}

fn opt_expr_of(v: Option<&MacroValue>, span: Span) -> Option<Box<Expr>> {
    match v {
        None | Some(MacroValue::Null) => None,
        Some(e) => Some(Box::new(expr_of(e, span))),
    }
}

fn function_of(f: &MacroValue, name: String, span: Span) -> Function {
    Function {
        name,
        type_params: Vec::new(),
        params: list_of(field(f, "args"))
            .iter()
            .map(|a| FunctionParam {
                meta: Vec::new(),
                name: string_of(field(a, "name")).unwrap_or_default(),
                type_hint: field(a, "type").and_then(|t| type_of_value(t, span)),
                optional: matches!(field(a, "opt"), Some(MacroValue::Bool(true))),
                rest: false,
                default_value: opt_expr_of(field(a, "value"), span),
                span,
            })
            .collect(),
        return_type: field(f, "ret").and_then(|t| type_of_value(t, span)),
        body: opt_expr_of(field(f, "expr"), span),
        span,
    }
}

/// An `Expr` value, an `{expr, pos}` object, an `ExprDef` or a `Constant`
/// as the expression it stands for; `None` when it is none of those.
pub fn try_expr_of(v: &MacroValue, span: Span) -> Option<Expr> {
    let mk = |kind| Expr { kind, span };
    match v {
        MacroValue::Expr(e) => Some((**e).clone()),
        MacroValue::Object(m) if matches!(m.get("expr"), Some(MacroValue::Enum(e, _, _)) if &**e == "ExprDef") => {
            try_expr_of(m.get("expr")?, span)
        }
        MacroValue::Enum(e, variant, args) if &**e == "Constant" => {
            let text = match args.first() {
                Some(MacroValue::String(t)) => t.to_string(),
                Some(MacroValue::Int(i)) => i.to_string(),
                Some(MacroValue::Float(f)) => f.to_string(),
                _ => String::new(),
            };
            Some(mk(match &**variant {
                "CInt" => ExprKind::Int(text.parse().unwrap_or(0)),
                "CFloat" => ExprKind::Float(text.parse().unwrap_or(0.0)),
                "CString" => ExprKind::String(text),
                "CIdent" => match text.as_str() {
                    "true" => ExprKind::Bool(true),
                    "false" => ExprKind::Bool(false),
                    "null" => ExprKind::Null,
                    "this" => ExprKind::This,
                    "super" => ExprKind::Super,
                    _ => ExprKind::Ident(text),
                },
                "CRegexp" => ExprKind::Regex {
                    pattern: text,
                    flags: string_of(args.get(1)).unwrap_or_default(),
                },
                _ => return None,
            }))
        }
        MacroValue::Enum(e, variant, args) if &**e == "ExprDef" => {
            let a = |i: usize| args.get(i);
            let ex = |i: usize| expr_of(args.get(i).unwrap_or(&MacroValue::Null), span);
            let bx = |i: usize| Box::new(ex(i));
            Some(mk(match &**variant {
                "EConst" => return try_expr_of(a(0)?, span),
                "EArray" => ExprKind::Index {
                    expr: bx(0),
                    index: bx(1),
                },
                "EBinop" => {
                    let (op_name, op_args) = match a(0)? {
                        MacroValue::Enum(_, n, oa) => (n.to_string(), oa.clone()),
                        MacroValue::String(n) => (format!("Op{}", n), Arc::new(Vec::new())),
                        _ => return None,
                    };
                    match op_name.as_str() {
                        "OpAssign" => ExprKind::Assign {
                            left: bx(1),
                            op: AssignOp::Assign,
                            right: bx(2),
                        },
                        "OpAssignOp" => {
                            let inner = match op_args.first() {
                                Some(MacroValue::Enum(_, n, _)) => binop_of(n)?,
                                _ => return None,
                            };
                            ExprKind::Assign {
                                left: bx(1),
                                op: assign_of(inner)?,
                                right: bx(2),
                            }
                        }
                        other => ExprKind::Binary {
                            left: bx(1),
                            op: binop_of(other)?,
                            right: bx(2),
                        },
                    }
                }
                "EField" => ExprKind::Field {
                    expr: bx(0),
                    field: string_of(a(1)).unwrap_or_default(),
                    is_optional: matches!(a(2), Some(MacroValue::Enum(_, k, _)) if &**k == "Safe"),
                },
                "EParenthesis" => ExprKind::Paren(bx(0)),
                "EObjectDecl" => ExprKind::Object(
                    list_of(a(0))
                        .iter()
                        .map(|f| ObjectField {
                            name: string_of(field(f, "field")).unwrap_or_default(),
                            expr: expr_of(field(f, "expr").unwrap_or(&MacroValue::Null), span),
                            span,
                        })
                        .collect(),
                ),
                "EArrayDecl" => ExprKind::Array(exprs_of(a(0), span)),
                "ECall" => ExprKind::Call {
                    expr: bx(0),
                    args: exprs_of(a(1), span),
                },
                "ENew" => {
                    let (type_path, params) = type_path_of(a(0)?, span)?;
                    ExprKind::New {
                        type_path,
                        params,
                        args: exprs_of(a(1), span),
                    }
                }
                "EUnop" => {
                    let name = match a(0)? {
                        MacroValue::Enum(_, n, _) => n.to_string(),
                        _ => return None,
                    };
                    let postfix = matches!(a(1), Some(MacroValue::Bool(true)));
                    let op = match (name.as_str(), postfix) {
                        ("OpNot", _) => UnaryOp::Not,
                        ("OpNeg", _) => UnaryOp::Neg,
                        ("OpNegBits", _) => UnaryOp::BitNot,
                        ("OpIncrement", false) => UnaryOp::PreIncr,
                        ("OpIncrement", true) => UnaryOp::PostIncr,
                        ("OpDecrement", false) => UnaryOp::PreDecr,
                        ("OpDecrement", true) => UnaryOp::PostDecr,
                        _ => return None,
                    };
                    ExprKind::Unary { op, expr: bx(2) }
                }
                "EVars" => {
                    let vars = list_of(a(0));
                    let decls: Vec<Expr> = vars
                        .iter()
                        .map(|v| {
                            let name = string_of(field(v, "name")).unwrap_or_default();
                            let type_hint = field(v, "type").and_then(|t| type_of_value(t, span));
                            let expr = opt_expr_of(field(v, "expr"), span);
                            mk(
                                if matches!(field(v, "isFinal"), Some(MacroValue::Bool(true))) {
                                    ExprKind::Final {
                                        name,
                                        type_hint,
                                        expr,
                                    }
                                } else {
                                    ExprKind::Var {
                                        name,
                                        type_hint,
                                        expr,
                                    }
                                },
                            )
                        })
                        .collect();
                    if decls.len() == 1 {
                        return decls.into_iter().next();
                    }
                    ExprKind::Block(decls.into_iter().map(BlockElement::Expr).collect())
                }
                "EFunction" => {
                    let name = match a(0) {
                        Some(MacroValue::Enum(_, k, ka)) if &**k == "FNamed" => {
                            string_of(ka.first()).unwrap_or_default()
                        }
                        _ => String::new(),
                    };
                    ExprKind::Function(function_of(a(1)?, name, span))
                }
                "EBlock" => ExprKind::Block(
                    exprs_of(a(0), span)
                        .into_iter()
                        .map(BlockElement::Expr)
                        .collect(),
                ),
                "EFor" => {
                    let head = ex(0);
                    let ExprKind::Binary {
                        left,
                        op: BinaryOp::In,
                        right,
                    } = head.kind
                    else {
                        return None;
                    };
                    let (var, key_var) = match left.kind {
                        ExprKind::Ident(v) => (v, None),
                        ExprKind::Binary {
                            left: k,
                            op: BinaryOp::Arrow,
                            right: v,
                        } => match (k.kind, v.kind) {
                            (ExprKind::Ident(k), ExprKind::Ident(v)) => (v, Some(k)),
                            _ => return None,
                        },
                        _ => return None,
                    };
                    ExprKind::For {
                        var,
                        key_var,
                        iter: right,
                        body: bx(1),
                    }
                }
                "EIf" => ExprKind::If {
                    cond: bx(0),
                    then_branch: bx(1),
                    else_branch: opt_expr_of(a(2), span),
                },
                "EWhile" => {
                    if matches!(a(2), Some(MacroValue::Bool(false))) {
                        ExprKind::DoWhile {
                            body: bx(1),
                            cond: bx(0),
                        }
                    } else {
                        ExprKind::While {
                            cond: bx(0),
                            body: bx(1),
                        }
                    }
                }
                "ESwitch" => ExprKind::Switch {
                    expr: bx(0),
                    cases: list_of(a(1))
                        .iter()
                        .map(|c| Case {
                            patterns: exprs_of(field(c, "values"), span)
                                .iter()
                                .map(expr_to_pattern)
                                .collect(),
                            guard: opt_expr_of(field(c, "guard"), span).map(|g| *g),
                            body: opt_expr_of(field(c, "expr"), span)
                                .map(|b| *b)
                                .unwrap_or_else(|| mk(ExprKind::Block(Vec::new()))),
                            span,
                        })
                        .collect(),
                    default: opt_expr_of(a(2), span),
                },
                "ETry" => ExprKind::Try {
                    expr: bx(0),
                    catches: list_of(a(1))
                        .iter()
                        .map(|c| Catch {
                            var: string_of(field(c, "name")).unwrap_or_default(),
                            type_hint: field(c, "type").and_then(|t| type_of_value(t, span)),
                            filter: None,
                            body: expr_of(field(c, "expr").unwrap_or(&MacroValue::Null), span),
                            span,
                        })
                        .collect(),
                    finally_block: None,
                },
                "EReturn" => ExprKind::Return(opt_expr_of(a(0), span)),
                "EBreak" => ExprKind::Break,
                "EContinue" => ExprKind::Continue,
                "EUntyped" => ExprKind::Untyped(bx(0)),
                "EThrow" => ExprKind::Throw(bx(0)),
                "ECast" => ExprKind::Cast {
                    expr: bx(0),
                    type_hint: a(1).and_then(|t| type_of_value(t, span)),
                },
                "ETernary" => ExprKind::Ternary {
                    cond: bx(0),
                    then_expr: bx(1),
                    else_expr: bx(2),
                },
                "ECheckType" => ExprKind::TypeCheck {
                    expr: bx(0),
                    type_hint: type_of_value(a(1)?, span)?,
                },
                "EIs" => ExprKind::Binary {
                    left: bx(0),
                    op: BinaryOp::Is,
                    right: Box::new(mk(match type_of_value(a(1)?, span)? {
                        Type::Path { path, .. } => ExprKind::Ident(path.name),
                        _ => return None,
                    })),
                },
                "EMeta" => {
                    let entry = a(0)?;
                    let raw = string_of(field(entry, "name")).unwrap_or_default();
                    ExprKind::Meta {
                        meta: Metadata {
                            name: raw.trim_start_matches(':').to_string(),
                            params: exprs_of(field(entry, "params"), span),
                            span,
                            compile_time: raw.starts_with(':'),
                        },
                        expr: bx(1),
                    }
                }
                "EDisplay" => return try_expr_of(a(0)?, span),
                _ => return None,
            }))
        }
        _ => None,
    }
}

/// As `try_expr_of`, with `null` for anything that is not an expression.
pub fn expr_of(v: &MacroValue, span: Span) -> Expr {
    try_expr_of(v, span).unwrap_or(Expr {
        kind: ExprKind::Null,
        span,
    })
}

/// Whether a macro parameter of this type receives the argument expression:
/// `Expr`, `ExprOf<T>`, or a rest / array of those.
pub fn is_expr_type(t: &Type) -> bool {
    match t {
        Type::Path { path, params, .. } => match path.name.as_str() {
            "Expr" | "ExprOf" => true,
            "Rest" | "Array" => params.first().is_some_and(is_expr_type),
            _ => false,
        },
        Type::Optional { inner, .. } | Type::Parenthesis { inner, .. } => is_expr_type(inner),
        _ => false,
    }
}

/// The constant a macro argument stands for, as a parameter of a
/// non-`Expr` type receives it; `None` when it is not a constant. A
/// single-quoted string is its text as written.
pub fn constant_of(e: &Expr) -> Option<MacroValue> {
    Some(match &e.kind {
        ExprKind::Int(i) => MacroValue::Int(*i),
        ExprKind::Float(f) => MacroValue::Float(*f),
        ExprKind::String(v) => s(v),
        ExprKind::Bool(b) => MacroValue::Bool(*b),
        ExprKind::Null => MacroValue::Null,
        ExprKind::StringInterpolation(_) => match def_of(&e.kind, e.span)? {
            MacroValue::Enum(_, _, args) => match args.first()? {
                MacroValue::Enum(_, _, c) => c.first()?.clone(),
                _ => return None,
            },
            _ => return None,
        },
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr,
        } => match constant_of(expr)? {
            MacroValue::Int(i) => MacroValue::Int(-i),
            MacroValue::Float(f) => MacroValue::Float(-f),
            _ => return None,
        },
        ExprKind::Paren(inner) => constant_of(inner)?,
        ExprKind::Array(items) => arr(items.iter().map(constant_of).collect::<Option<_>>()?),
        ExprKind::Object(fields) => MacroValue::Object(Arc::new(
            fields
                .iter()
                .map(|f| Some((f.name.clone(), constant_of(&f.expr)?)))
                .collect::<Option<_>>()?,
        )),
        _ => return None,
    })
}
