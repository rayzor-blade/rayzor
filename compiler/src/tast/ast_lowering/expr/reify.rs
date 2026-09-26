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
            // `e is T` is its own node, with T as a type.
            ExprKind::Binary {
                left,
                op: BinaryOp::Is,
                right,
            } => {
                let path = Self::expr_type_path(right)?;
                let ty = parser::Type::Path {
                    path,
                    params: Vec::new(),
                    span: right.span,
                };
                def("EIs", vec![self.expr(left)?, self.complex_type(&ty)?])
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
            ExprKind::TypeDecl(decl) => self.type_definition(decl)?,
            ExprKind::Break => def("EBreak", vec![]),
            ExprKind::Continue => def("EContinue", vec![]),
            _ => return None,
        })
    }

    /// An object literal laid out as the haxe.macro.Expr typedef `name`.
    fn typed(&self, name: &str, fields: Vec<(&str, Expr)>) -> Expr {
        self.mk(ExprKind::TypeCheck {
            expr: Box::new(self.object(fields)),
            type_hint: parser::Type::Path {
                path: parser::TypePath {
                    package: vec!["haxe".to_string(), "macro".to_string(), "Expr".to_string()],
                    name: name.to_string(),
                    sub: None,
                },
                params: Vec::new(),
                span: self.span,
            },
        })
    }

    fn null(&self) -> Expr {
        self.mk(ExprKind::Null)
    }

    fn array(&self, items: Vec<Expr>) -> Expr {
        self.mk(ExprKind::Array(items))
    }

    fn opt<T>(&self, value: Option<T>, f: impl FnOnce(T) -> Option<Expr>) -> Option<Expr> {
        match value {
            Some(v) => f(v),
            None => Some(self.null()),
        }
    }

    fn type_path(&self, path: &parser::TypePath, params: &[parser::Type]) -> Option<Expr> {
        let params = params
            .iter()
            .map(|p| match p {
                parser::Type::Const { value, .. } => {
                    Some(self.ctor("TypeParam", "TPExpr", vec![self.expr(value)?]))
                }
                other => Some(self.ctor("TypeParam", "TPType", vec![self.complex_type(other)?])),
            })
            .collect::<Option<Vec<_>>>()?;
        Some(self.typed(
            "TypePath",
            vec![
                (
                    "pack",
                    self.array(path.package.iter().map(|p| self.string(p)).collect()),
                ),
                ("name", self.string(&path.name)),
                ("params", self.array(params)),
                (
                    "sub",
                    match &path.sub {
                        Some(sub) => self.string(sub),
                        None => self.null(),
                    },
                ),
            ],
        ))
    }

    /// `pack.Name` written as an expression, as a type path.
    fn expr_type_path(e: &Expr) -> Option<parser::TypePath> {
        let mut parts = Vec::new();
        let mut cur = e;
        loop {
            match &cur.kind {
                ExprKind::Ident(name) => {
                    parts.push(name.clone());
                    break;
                }
                ExprKind::Field { expr, field, .. } => {
                    parts.push(field.clone());
                    cur = expr;
                }
                _ => return None,
            }
        }
        parts.reverse();
        let name = parts.pop()?;
        Some(parser::TypePath {
            package: parts,
            name,
            sub: None,
        })
    }

    fn complex_type(&self, t: &parser::Type) -> Option<Expr> {
        Some(match t {
            parser::Type::Path { path, params, .. } => {
                self.ctor("ComplexType", "TPath", vec![self.type_path(path, params)?])
            }
            parser::Type::Function { params, ret, .. } => self.ctor(
                "ComplexType",
                "TFunction",
                vec![
                    self.array(
                        params
                            .iter()
                            .map(|p| self.complex_type(p))
                            .collect::<Option<_>>()?,
                    ),
                    self.complex_type(ret)?,
                ],
            ),
            parser::Type::Anonymous { fields, .. } => {
                let fields = fields
                    .iter()
                    .map(|f| {
                        let meta = if f.optional {
                            vec![self.meta_entry(":optional", &[])?]
                        } else {
                            Vec::new()
                        };
                        Some(self.typed(
                            "Field",
                            vec![
                                ("name", self.string(&f.name)),
                                ("doc", self.null()),
                                ("access", self.array(Vec::new())),
                                (
                                    "kind",
                                    self.ctor(
                                        "FieldType",
                                        "FVar",
                                        vec![self.complex_type(&f.type_hint)?, self.null()],
                                    ),
                                ),
                                ("pos", self.null()),
                                ("meta", self.array(meta)),
                            ],
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?;
                self.ctor("ComplexType", "TAnonymous", vec![self.array(fields)])
            }
            parser::Type::Optional { inner, .. } => {
                self.ctor("ComplexType", "TOptional", vec![self.complex_type(inner)?])
            }
            parser::Type::Parenthesis { inner, .. } => {
                self.ctor("ComplexType", "TParent", vec![self.complex_type(inner)?])
            }
            parser::Type::Intersection { left, right, .. } => self.ctor(
                "ComplexType",
                "TIntersection",
                vec![self.array(vec![self.complex_type(left)?, self.complex_type(right)?])],
            ),
            parser::Type::Wildcard { .. } | parser::Type::Const { .. } => return None,
        })
    }

    fn meta_entry(&self, name: &str, params: &[Expr]) -> Option<Expr> {
        Some(self.typed(
            "MetadataEntry",
            vec![
                ("name", self.string(name)),
                ("params", self.list(params)?),
                ("pos", self.null()),
            ],
        ))
    }

    fn metadata(&self, meta: &[parser::Metadata]) -> Option<Expr> {
        Some(
            self.array(
                meta.iter()
                    .map(|m| {
                        let name = if m.compile_time {
                            format!(":{}", m.name)
                        } else {
                            m.name.clone()
                        };
                        self.meta_entry(&name, &m.params)
                    })
                    .collect::<Option<_>>()?,
            ),
        )
    }

    fn type_params(&self, params: &[parser::TypeParam]) -> Option<Expr> {
        Some(
            self.array(
                params
                    .iter()
                    .map(|p| {
                        Some(self.typed(
                            "TypeParamDecl",
                            vec![
                            ("name", self.string(&p.name)),
                            (
                                "constraints",
                                self.array(
                                    p.constraints
                                        .iter()
                                        .map(|c| self.complex_type(c))
                                        .collect::<Option<_>>()?,
                                ),
                            ),
                            ("params", self.array(Vec::new())),
                            ("meta", self.metadata(&p.meta)?),
                            (
                                "defaultType",
                                self.opt(p.default_type.as_ref(), |t| self.complex_type(t))?,
                            ),
                        ],
                        ))
                    })
                    .collect::<Option<_>>()?,
            ),
        )
    }

    fn function(&self, f: &parser::Function) -> Option<Expr> {
        let args = f
            .params
            .iter()
            .map(|a| {
                Some(self.typed(
                    "FunctionArg",
                    vec![
                        ("name", self.string(&a.name)),
                        ("opt", self.mk(ExprKind::Bool(a.optional))),
                        (
                            "type",
                            self.opt(a.type_hint.as_ref(), |t| self.complex_type(t))?,
                        ),
                        (
                            "value",
                            self.opt(a.default_value.as_deref(), |e| self.expr(e))?,
                        ),
                        ("meta", self.metadata(&a.meta)?),
                    ],
                ))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(self.typed(
            "Function",
            vec![
                ("args", self.array(args)),
                (
                    "ret",
                    self.opt(f.return_type.as_ref(), |t| self.complex_type(t))?,
                ),
                ("expr", self.opt(f.body.as_deref(), |b| self.expr(b))?),
                ("params", self.type_params(&f.type_params)?),
            ],
        ))
    }

    fn field(&self, f: &parser::ClassField) -> Option<Expr> {
        use parser::{ClassFieldKind, Modifier};
        let mut access = Vec::new();
        match f.access {
            Some(parser::Access::Public) => access.push("APublic"),
            Some(parser::Access::Private) => access.push("APrivate"),
            None => {}
        }
        for m in &f.modifiers {
            access.push(match m {
                Modifier::Static => "AStatic",
                Modifier::Inline => "AInline",
                Modifier::Macro => "AMacro",
                Modifier::Dynamic => "ADynamic",
                Modifier::Override => "AOverride",
                Modifier::Final => "AFinal",
                Modifier::Extern => "AExtern",
            });
        }
        let (name, kind) = match &f.kind {
            ClassFieldKind::Var {
                name,
                type_hint,
                expr,
            }
            | ClassFieldKind::Final {
                name,
                type_hint,
                expr,
            } => {
                if matches!(f.kind, ClassFieldKind::Final { .. }) && !access.contains(&"AFinal") {
                    access.push("AFinal");
                }
                (
                    name,
                    self.ctor(
                        "FieldType",
                        "FVar",
                        vec![
                            self.opt(type_hint.as_ref(), |t| self.complex_type(t))?,
                            self.opt(expr.as_ref(), |e| self.expr(e))?,
                        ],
                    ),
                )
            }
            ClassFieldKind::Property {
                name,
                type_hint,
                getter,
                setter,
            } => {
                let spell = |a: &parser::PropertyAccess| match a {
                    parser::PropertyAccess::Default => "default".to_string(),
                    parser::PropertyAccess::Null => "null".to_string(),
                    parser::PropertyAccess::Never => "never".to_string(),
                    parser::PropertyAccess::Dynamic => "dynamic".to_string(),
                    parser::PropertyAccess::Custom(name) => name.clone(),
                };
                (
                    name,
                    self.ctor(
                        "FieldType",
                        "FProp",
                        vec![
                            self.string(&spell(getter)),
                            self.string(&spell(setter)),
                            self.opt(type_hint.as_ref(), |t| self.complex_type(t))?,
                            self.null(),
                        ],
                    ),
                )
            }
            ClassFieldKind::Function(func) => (
                &func.name,
                self.ctor("FieldType", "FFun", vec![self.function(func)?]),
            ),
        };
        Some(self.typed(
            "Field",
            vec![
                ("name", self.string(name)),
                ("doc", self.null()),
                (
                    "access",
                    self.array(
                        access
                            .iter()
                            .map(|a| self.ctor("Access", a, Vec::new()))
                            .collect(),
                    ),
                ),
                ("kind", kind),
                ("pos", self.null()),
                ("meta", self.metadata(&f.meta)?),
            ],
        ))
    }

    /// `macro class ...` as the TypeDefinition value it builds.
    fn type_definition(&self, decl: &parser::TypeDeclaration) -> Option<Expr> {
        use parser::TypeDeclaration;
        let bool_expr = |b: bool| self.mk(ExprKind::Bool(b));
        let (name, meta, params, kind, fields): (&str, _, _, Expr, &[parser::ClassField]) =
            match decl {
                TypeDeclaration::Class(c) => {
                    let super_class = match &c.extends {
                        Some(parser::Type::Path { path, params, .. }) => {
                            self.type_path(path, params)?
                        }
                        _ => self.null(),
                    };
                    let interfaces = c
                        .implements
                        .iter()
                        .map(|t| match t {
                            parser::Type::Path { path, params, .. } => self.type_path(path, params),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    let kind = self.ctor(
                        "TypeDefKind",
                        "TDClass",
                        vec![
                            super_class,
                            self.array(interfaces),
                            bool_expr(false),
                            bool_expr(c.modifiers.contains(&parser::Modifier::Final)),
                            bool_expr(false),
                        ],
                    );
                    (&c.name, &c.meta, &c.type_params, kind, &c.fields)
                }
                TypeDeclaration::Interface(i) => {
                    let interfaces = i
                        .extends
                        .iter()
                        .map(|t| match t {
                            parser::Type::Path { path, params, .. } => self.type_path(path, params),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()?;
                    let kind = self.ctor(
                        "TypeDefKind",
                        "TDClass",
                        vec![
                            self.null(),
                            self.array(interfaces),
                            bool_expr(true),
                            bool_expr(false),
                            bool_expr(false),
                        ],
                    );
                    (&i.name, &i.meta, &i.type_params, kind, &i.fields)
                }
                TypeDeclaration::Typedef(t) => {
                    let kind = self.ctor(
                        "TypeDefKind",
                        "TDAlias",
                        vec![self.complex_type(&t.type_def)?],
                    );
                    (&t.name, &t.meta, &t.type_params, kind, &[])
                }
                _ => return None,
            };
        let fields = fields
            .iter()
            .map(|f| self.field(f))
            .collect::<Option<Vec<_>>>()?;
        Some(self.typed(
            "TypeDefinition",
            vec![
                ("pack", self.array(Vec::new())),
                ("name", self.string(name)),
                ("doc", self.null()),
                ("pos", self.null()),
                ("meta", self.metadata(meta)?),
                ("params", self.type_params(params)?),
                ("isExtern", bool_expr(false)),
                ("kind", kind),
                ("fields", self.array(fields)),
            ],
        ))
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
            BinaryOp::In => "OpIn",
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
