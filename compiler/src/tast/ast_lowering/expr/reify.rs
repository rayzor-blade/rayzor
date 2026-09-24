//! `macro e` evaluated at run time: the code that builds the `haxe.macro.Expr`
//! value `e` denotes, with `$`-escapes read from the surrounding scope.

use super::*;
use parser::{AssignOp, BinaryOp, Expr, ExprKind, ObjectField, Span, UnaryOp};

impl<'a> AstLowering<'a> {
    /// The construction expression for `macro e`, or `None` for a form this
    /// does not build.
    pub(crate) fn runtime_reification(e: &Expr) -> Option<Expr> {
        Reifier { span: e.span }.expr(e)
    }
}

struct Reifier {
    span: Span,
}

impl Reifier {
    fn mk(&self, kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: self.span,
        }
    }

    fn string(&self, s: &str) -> Expr {
        self.mk(ExprKind::String(s.to_string()))
    }

    fn ident(&self, name: &str) -> Expr {
        self.mk(ExprKind::Ident(name.to_string()))
    }

    /// `haxe.macro.Expr.<enum_name>.<ctor>(args)`, or the bare constructor
    /// when there are no arguments.
    fn ctor(&self, enum_name: &str, ctor: &str, args: Vec<Expr>) -> Expr {
        let mut path = self.ident("haxe");
        for part in ["macro", "Expr", enum_name, ctor] {
            path = self.mk(ExprKind::Field {
                expr: Box::new(path),
                field: part.to_string(),
                is_optional: false,
            });
        }
        if args.is_empty() {
            path
        } else {
            self.mk(ExprKind::Call {
                expr: Box::new(path),
                args,
            })
        }
    }

    fn object(&self, fields: Vec<(&str, Expr)>) -> Expr {
        self.mk(ExprKind::Object(
            fields
                .into_iter()
                .map(|(name, expr)| ObjectField {
                    name: name.to_string(),
                    expr,
                    span: self.span,
                })
                .collect(),
        ))
    }

    /// `({pos: null, expr: def} : haxe.macro.Expr)`, laid out as the typedef.
    fn wrap(&self, def: Expr) -> Expr {
        let object = self.object(vec![("pos", self.mk(ExprKind::Null)), ("expr", def)]);
        self.mk(ExprKind::TypeCheck {
            expr: Box::new(object),
            type_hint: parser::Type::Path {
                path: parser::TypePath {
                    package: vec!["haxe".to_string(), "macro".to_string()],
                    name: "Expr".to_string(),
                    sub: None,
                },
                params: Vec::new(),
                span: self.span,
            },
        })
    }

    fn constant(&self, ctor: &str, text: &str) -> Expr {
        self.wrap(self.ctor(
            "ExprDef",
            "EConst",
            vec![self.ctor("Constant", ctor, vec![self.string(text)])],
        ))
    }

    fn list(&self, items: &[Expr]) -> Option<Expr> {
        Some(self.mk(ExprKind::Array(
            items.iter().map(|e| self.expr(e)).collect::<Option<_>>()?,
        )))
    }

    fn expr(&self, e: &Expr) -> Option<Expr> {
        let def = |ctor: &str, args: Vec<Expr>| self.wrap(self.ctor("ExprDef", ctor, args));
        Some(match &e.kind {
            ExprKind::Int(i) => self.constant("CInt", &i.to_string()),
            ExprKind::Float(f) => self.constant("CFloat", &f.to_string()),
            ExprKind::String(s) => self.constant("CString", s),
            ExprKind::Bool(b) => self.constant("CIdent", if *b { "true" } else { "false" }),
            ExprKind::Null => self.constant("CIdent", "null"),
            ExprKind::This => self.constant("CIdent", "this"),
            ExprKind::Ident(name) => self.constant("CIdent", name),
            // `$i{name}`: an identifier spelled by a runtime string.
            ExprKind::DollarIdent {
                name,
                arg: Some(arg),
            } if name == "i" => def(
                "EConst",
                vec![self.ctor("Constant", "CIdent", vec![(**arg).clone()])],
            ),
            // `$v{literal}`: the literal as an expression. A computed value
            // would need its runtime type, which this does not build.
            ExprKind::DollarIdent {
                name,
                arg: Some(arg),
            } if name == "v" => match &arg.kind {
                ExprKind::Int(i) => self.constant("CInt", &i.to_string()),
                ExprKind::Float(f) => self.constant("CFloat", &f.to_string()),
                ExprKind::String(s) => self.constant("CString", s),
                ExprKind::Bool(b) => self.constant("CIdent", if *b { "true" } else { "false" }),
                ExprKind::Null => self.constant("CIdent", "null"),
                _ => return None,
            },
            // `$e` / `${e}`: an expression value spliced in as it is.
            ExprKind::DollarIdent { name, arg: None } => self.ident(name),
            ExprKind::DollarIdent {
                name,
                arg: Some(arg),
            } if name == "e" || name.is_empty() => (**arg).clone(),
            ExprKind::Reify(inner) => (**inner).clone(),
            ExprKind::Field {
                expr: obj, field, ..
            } => {
                // `$e.$name`: the field name is a runtime string.
                let name = match field.strip_prefix('$') {
                    Some(var) => self.ident(var),
                    None => self.string(field),
                };
                def("EField", vec![self.expr(obj)?, name])
            }
            ExprKind::Call { expr: callee, args } => {
                def("ECall", vec![self.expr(callee)?, self.list(args)?])
            }
            ExprKind::Index { expr: obj, index } => {
                def("EArray", vec![self.expr(obj)?, self.expr(index)?])
            }
            ExprKind::Paren(inner) => def("EParenthesis", vec![self.expr(inner)?]),
            ExprKind::Array(items) => def("EArrayDecl", vec![self.list(items)?]),
            ExprKind::Block(elements) => {
                let items = elements
                    .iter()
                    .map(|el| match el {
                        parser::BlockElement::Expr(e) => self.expr(e),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                def("EBlock", vec![self.mk(ExprKind::Array(items))])
            }
            ExprKind::Object(fields) => {
                let items = fields
                    .iter()
                    .map(|f| {
                        Some(self.object(vec![
                            ("field", self.string(&f.name)),
                            ("expr", self.expr(&f.expr)?),
                        ]))
                    })
                    .collect::<Option<Vec<_>>>()?;
                def("EObjectDecl", vec![self.mk(ExprKind::Array(items))])
            }
            ExprKind::Binary { left, op, right } => def(
                "EBinop",
                vec![self.binop(*op)?, self.expr(left)?, self.expr(right)?],
            ),
            ExprKind::Assign { left, op, right } => {
                let binop = match op {
                    AssignOp::Assign => self.ctor("Binop", "OpAssign", vec![]),
                    other => self.ctor(
                        "Binop",
                        "OpAssignOp",
                        vec![self.binop(Self::assign_base(*other)?)?],
                    ),
                };
                def("EBinop", vec![binop, self.expr(left)?, self.expr(right)?])
            }
            ExprKind::Unary { op, expr: inner } => {
                let (name, postfix) = match op {
                    UnaryOp::Not => ("OpNot", false),
                    UnaryOp::Neg => ("OpNeg", false),
                    UnaryOp::BitNot => ("OpNegBits", false),
                    UnaryOp::PreIncr => ("OpIncrement", false),
                    UnaryOp::PreDecr => ("OpDecrement", false),
                    UnaryOp::PostIncr => ("OpIncrement", true),
                    UnaryOp::PostDecr => ("OpDecrement", true),
                };
                def(
                    "EUnop",
                    vec![
                        self.ctor("Unop", name, vec![]),
                        self.mk(ExprKind::Bool(postfix)),
                        self.expr(inner)?,
                    ],
                )
            }
            ExprKind::Var {
                name,
                type_hint: None,
                expr: init,
            } => {
                let init = match init {
                    Some(v) => self.expr(v)?,
                    None => self.mk(ExprKind::Null),
                };
                let var = self.object(vec![
                    ("name", self.string(name)),
                    ("type", self.mk(ExprKind::Null)),
                    ("expr", init),
                ]);
                def("EVars", vec![self.mk(ExprKind::Array(vec![var]))])
            }
            ExprKind::Return(value) => {
                let value = match value {
                    Some(v) => self.expr(v)?,
                    None => self.mk(ExprKind::Null),
                };
                def("EReturn", vec![value])
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                let otherwise = match else_branch {
                    Some(e) => self.expr(e)?,
                    None => self.mk(ExprKind::Null),
                };
                def(
                    "EIf",
                    vec![self.expr(cond)?, self.expr(then_branch)?, otherwise],
                )
            }
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => def(
                "ETernary",
                vec![
                    self.expr(cond)?,
                    self.expr(then_expr)?,
                    self.expr(else_expr)?,
                ],
            ),
            ExprKind::Throw(inner) => def("EThrow", vec![self.expr(inner)?]),
            ExprKind::Break => def("EBreak", vec![]),
            ExprKind::Continue => def("EContinue", vec![]),
            _ => return None,
        })
    }

    fn binop(&self, op: BinaryOp) -> Option<Expr> {
        let name = match op {
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
            BinaryOp::Is => return None,
        };
        Some(self.ctor("Binop", name, vec![]))
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
}
