//! `haxe.macro.Printer` at macro time: an expression value printed exactly
//! as `std/haxe/macro/Printer.hx` prints it, case for case.

use super::ast_bridge::expr_kind_to_value;
use super::value::MacroValue;

const TAB: &str = "\t";

pub struct Printer {
    tabs: String,
}

fn text(v: Option<&MacroValue>) -> String {
    match v {
        Some(MacroValue::String(s)) => s.to_string(),
        Some(MacroValue::Int(i)) => i.to_string(),
        Some(MacroValue::Float(f)) => f.to_string(),
        _ => String::new(),
    }
}

fn is_null(v: Option<&MacroValue>) -> bool {
    matches!(v, None | Some(MacroValue::Null))
}

fn field<'v>(v: &'v MacroValue, name: &str) -> Option<&'v MacroValue> {
    match v {
        MacroValue::Object(m) => m.get(name),
        _ => None,
    }
}

fn list(v: Option<&MacroValue>) -> Vec<MacroValue> {
    match v {
        Some(MacroValue::Array(a)) => a.as_ref().clone(),
        _ => Vec::new(),
    }
}

fn variant(v: &MacroValue) -> Option<(&str, &[MacroValue])> {
    match v {
        MacroValue::Enum(_, name, args) => Some((name, args.as_slice())),
        _ => None,
    }
}

/// The ExprDef an expression value stands for.
fn def_of(v: &MacroValue) -> Option<MacroValue> {
    match v {
        MacroValue::Expr(e) => Some(expr_kind_to_value(&e.kind, e.span)),
        MacroValue::Object(m) => m.get("expr").cloned(),
        MacroValue::Enum(e, _, _) if &**e == "ExprDef" => Some(v.clone()),
        _ => None,
    }
}

fn escape(s: &str, delim: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\'', "\\'")
        .replace('"', "\\\"")
        .replace('\0', "\\x00");
    format!("{delim}{escaped}{delim}")
}

impl Printer {
    pub fn new() -> Self {
        Printer {
            tabs: String::new(),
        }
    }

    fn opt(
        &mut self,
        v: Option<&MacroValue>,
        prefix: &str,
        f: fn(&mut Self, &MacroValue) -> String,
    ) -> String {
        match v {
            None | Some(MacroValue::Null) => String::new(),
            Some(v) => format!("{}{}", prefix, f(self, v)),
        }
    }

    pub fn print_unop(op: &str) -> &'static str {
        match op {
            "OpIncrement" => "++",
            "OpDecrement" => "--",
            "OpNot" => "!",
            "OpNeg" => "-",
            "OpNegBits" => "~",
            "OpSpread" => "...",
            _ => "",
        }
    }

    pub fn print_binop(op: &MacroValue) -> String {
        let Some((name, args)) = variant(op) else {
            return String::new();
        };
        match name {
            "OpAdd" => "+",
            "OpMult" => "*",
            "OpDiv" => "/",
            "OpSub" => "-",
            "OpAssign" => "=",
            "OpEq" => "==",
            "OpNotEq" => "!=",
            "OpGt" => ">",
            "OpGte" => ">=",
            "OpLt" => "<",
            "OpLte" => "<=",
            "OpAnd" => "&",
            "OpOr" => "|",
            "OpXor" => "^",
            "OpBoolAnd" => "&&",
            "OpBoolOr" => "||",
            "OpShl" => "<<",
            "OpShr" => ">>",
            "OpUShr" => ">>>",
            "OpMod" => "%",
            "OpInterval" => "...",
            "OpArrow" => "=>",
            "OpIn" => "in",
            "OpNullCoal" => "??",
            "OpAssignOp" => {
                return format!(
                    "{}=",
                    args.first().map(Self::print_binop).unwrap_or_default()
                );
            }
            _ => "",
        }
        .to_string()
    }

    pub fn print_constant(c: &MacroValue) -> String {
        let Some((name, args)) = variant(c) else {
            return String::new();
        };
        let s = text(args.first());
        match name {
            "CString" => match args.get(1).and_then(variant) {
                Some(("SingleQuotes", _)) => escape(&s, "'"),
                _ => escape(&s, "\""),
            },
            "CIdent" => s,
            "CInt" | "CFloat" => {
                if is_null(args.get(1)) {
                    s
                } else {
                    format!("{}{}", s, text(args.get(1)))
                }
            }
            "CRegexp" => format!("~/{}/{}", s, text(args.get(1))),
            _ => s,
        }
    }

    fn print_type_param(&mut self, p: &MacroValue) -> String {
        match variant(p) {
            Some(("TPType", [ct])) => self.print_complex_type(ct),
            Some(("TPExpr", [e])) => self.print_expr(e),
            _ => String::new(),
        }
    }

    fn print_type_path(&mut self, tp: &MacroValue) -> String {
        let pack: Vec<String> = list(field(tp, "pack"))
            .iter()
            .map(|p| text(Some(p)))
            .collect();
        let mut s = if pack.is_empty() {
            String::new()
        } else {
            format!("{}.", pack.join("."))
        };
        s.push_str(&text(field(tp, "name")));
        if !is_null(field(tp, "sub")) {
            s.push('.');
            s.push_str(&text(field(tp, "sub")));
        }
        let params = list(field(tp, "params"));
        if !params.is_empty() {
            let rendered: Vec<String> = params.iter().map(|p| self.print_type_param(p)).collect();
            s.push_str(&format!("<{}>", rendered.join(", ")));
        }
        s
    }

    pub fn print_complex_type(&mut self, ct: &MacroValue) -> String {
        match variant(ct) {
            Some(("TPath", [tp])) => self.print_type_path(tp),
            Some(("TFunction", [args, ret])) => {
                let args = list(Some(args));
                let wrap = !matches!(
                    args.as_slice(),
                    [single] if matches!(variant(single), Some(("TParent", _)) | Some(("TPath", _)))
                        || matches!(variant(single), Some(("TOptional", [inner])) if matches!(variant(inner), Some(("TPath", _))))
                );
                let arg_str: Vec<String> =
                    args.iter().map(|a| self.print_complex_type(a)).collect();
                let arg_str = arg_str.join(", ");
                let ret_str = if matches!(variant(ret), Some(("TFunction", _))) {
                    format!("({})", self.print_complex_type(ret))
                } else {
                    self.print_complex_type(ret)
                };
                if wrap {
                    format!("({arg_str}) -> {ret_str}")
                } else {
                    format!("{arg_str} -> {ret_str}")
                }
            }
            Some(("TAnonymous", [fields])) => {
                let parts: Vec<String> = list(Some(fields))
                    .iter()
                    .map(|f| format!("{}; ", self.print_field(f)))
                    .collect();
                format!("{{ {}}}", parts.join(""))
            }
            Some(("TParent", [inner])) => format!("({})", self.print_complex_type(inner)),
            Some(("TOptional", [inner])) => format!("?{}", self.print_complex_type(inner)),
            Some(("TNamed", [n, inner])) => {
                format!("{}:{}", text(Some(n)), self.print_complex_type(inner))
            }
            Some(("TIntersection", [types])) => list(Some(types))
                .iter()
                .map(|t| self.print_complex_type(t))
                .collect::<Vec<_>>()
                .join(" & "),
            _ => String::new(),
        }
    }

    pub fn print_metadata(&mut self, meta: &MacroValue) -> String {
        let params = list(field(meta, "params"));
        let args = if params.is_empty() {
            String::new()
        } else {
            format!("({})", self.print_exprs(&params, ", "))
        };
        format!("@{}{}", text(field(meta, "name")), args)
    }

    fn print_field(&mut self, f: &MacroValue) -> String {
        let name = text(field(f, "name"));
        match field(f, "kind").and_then(variant) {
            Some(("FVar", args)) => format!(
                "var {}{}{}",
                name,
                self.opt(args.first(), " : ", Self::print_complex_type),
                self.opt(args.get(1), " = ", Self::print_expr)
            ),
            Some(("FProp", args)) => format!(
                "var {}({}, {}){}{}",
                name,
                text(args.first()),
                text(args.get(1)),
                self.opt(args.get(2), " : ", Self::print_complex_type),
                self.opt(args.get(3), " = ", Self::print_expr)
            ),
            Some(("FFun", [func])) => {
                format!("function {}{}", name, self.print_function(func, None))
            }
            _ => name,
        }
    }

    fn print_type_param_decl(&mut self, tpd: &MacroValue) -> String {
        let meta = list(field(tpd, "meta"));
        let mut s = if meta.is_empty() {
            String::new()
        } else {
            let m: Vec<String> = meta.iter().map(|m| self.print_metadata(m)).collect();
            format!("{} ", m.join(" "))
        };
        s.push_str(&text(field(tpd, "name")));
        let params = list(field(tpd, "params"));
        if !params.is_empty() {
            let p: Vec<String> = params
                .iter()
                .map(|p| self.print_type_param_decl(p))
                .collect();
            s.push_str(&format!("<{}>", p.join(", ")));
        }
        let constraints = list(field(tpd, "constraints"));
        if !constraints.is_empty() {
            let c: Vec<String> = constraints
                .iter()
                .map(|c| self.print_complex_type(c))
                .collect();
            s.push_str(&format!(":({})", c.join(", ")));
        }
        if !is_null(field(tpd, "defaultType")) {
            let d = self.print_complex_type(field(tpd, "defaultType").unwrap());
            s.push_str(&format!("={d}"));
        }
        s
    }

    fn print_function_arg(&mut self, arg: &MacroValue) -> String {
        format!(
            "{}{}{}{}",
            if matches!(field(arg, "opt"), Some(MacroValue::Bool(true))) {
                "?"
            } else {
                ""
            },
            text(field(arg, "name")),
            self.opt(field(arg, "type"), ":", Self::print_complex_type),
            self.opt(field(arg, "value"), " = ", Self::print_expr)
        )
    }

    fn print_function(&mut self, func: &MacroValue, kind: Option<&MacroValue>) -> String {
        let arrow = matches!(kind.and_then(variant), Some(("FArrow", _)));
        let args = list(field(func, "args"));
        let skip_parens = arrow && args.len() == 1 && is_null(field(&args[0], "type"));
        let params = list(field(func, "params"));
        let mut s = String::new();
        if !params.is_empty() {
            let p: Vec<String> = params
                .iter()
                .map(|p| self.print_type_param_decl(p))
                .collect();
            s.push_str(&format!("<{}>", p.join(", ")));
        }
        if !skip_parens {
            s.push('(');
        }
        let a: Vec<String> = args.iter().map(|a| self.print_function_arg(a)).collect();
        s.push_str(&a.join(", "));
        if !skip_parens {
            s.push(')');
        }
        if arrow {
            s.push_str(" ->");
        }
        s.push_str(&self.opt(field(func, "ret"), ":", Self::print_complex_type));
        s.push_str(&self.opt(field(func, "expr"), " ", Self::print_expr));
        s
    }

    fn print_var(&mut self, v: &MacroValue) -> String {
        let s = format!(
            "{}{}{}",
            text(field(v, "name")),
            self.opt(field(v, "type"), ":", Self::print_complex_type),
            self.opt(field(v, "expr"), " = ", Self::print_expr)
        );
        let meta = list(field(v, "meta"));
        if meta.is_empty() {
            s
        } else {
            let m: Vec<String> = meta.iter().map(|m| self.print_metadata(m)).collect();
            format!("{} {}", m.join(" "), s)
        }
    }

    pub fn print_exprs(&mut self, el: &[MacroValue], sep: &str) -> String {
        el.iter()
            .map(|e| self.print_expr(e))
            .collect::<Vec<_>>()
            .join(sep)
    }

    pub fn print_expr(&mut self, e: &MacroValue) -> String {
        if matches!(e, MacroValue::Null) {
            return "#NULL".to_string();
        }
        let Some(def) = def_of(e) else {
            return "#NULL".to_string();
        };
        let Some((name, args)) = variant(&def) else {
            return String::new();
        };
        let p = |me: &mut Self, i: usize| me.print_expr(args.get(i).unwrap_or(&MacroValue::Null));
        match name {
            "EConst" => args.first().map(Self::print_constant).unwrap_or_default(),
            "EArray" => format!("{}[{}]", p(self, 0), p(self, 1)),
            "EBinop" => format!(
                "{} {} {}",
                p(self, 1),
                args.first().map(Self::print_binop).unwrap_or_default(),
                p(self, 2)
            ),
            "EField" => {
                let safe = matches!(args.get(2).and_then(variant), Some(("Safe", _)));
                format!(
                    "{}{}{}",
                    p(self, 0),
                    if safe { "?." } else { "." },
                    text(args.get(1))
                )
            }
            "EParenthesis" => format!("({})", p(self, 0)),
            "EObjectDecl" => {
                let fields: Vec<String> = list(args.first())
                    .iter()
                    .map(|f| {
                        let quoted =
                            matches!(field(f, "quotes").and_then(variant), Some(("Quoted", _)));
                        let key = text(field(f, "field"));
                        let key = if quoted { format!("\"{key}\"") } else { key };
                        format!(
                            "{} : {}",
                            key,
                            self.print_expr(field(f, "expr").unwrap_or(&MacroValue::Null))
                        )
                    })
                    .collect();
                format!("{{ {} }}", fields.join(", "))
            }
            "EArrayDecl" => format!("[{}]", self.print_exprs(&list(args.first()), ", ")),
            "ECall" => format!(
                "{}({})",
                p(self, 0),
                self.print_exprs(&list(args.get(1)), ", ")
            ),
            "ENew" => {
                let tp = self.print_type_path(args.first().unwrap_or(&MacroValue::Null));
                format!("new {}({})", tp, self.print_exprs(&list(args.get(1)), ", "))
            }
            "EUnop" => {
                let op = args.first().and_then(variant).map(|(n, _)| n).unwrap_or("");
                let op = Self::print_unop(op);
                if matches!(args.get(1), Some(MacroValue::Bool(true))) {
                    format!("{}{}", p(self, 2), op)
                } else {
                    format!("{}{}", op, p(self, 2))
                }
            }
            "EFunction" => {
                let kind = args.first();
                let func = args.get(1).unwrap_or(&MacroValue::Null);
                match kind.and_then(variant) {
                    Some(("FNamed", [no, inlined])) => format!(
                        "{}function {}{}",
                        if matches!(inlined, MacroValue::Bool(true)) {
                            "inline "
                        } else {
                            ""
                        },
                        text(Some(no)),
                        self.print_function(func, None)
                    ),
                    Some((k, _)) => format!(
                        "{}{}",
                        if k != "FArrow" { "function" } else { "" },
                        self.print_function(func, kind)
                    ),
                    None => format!("function{}", self.print_function(func, None)),
                }
            }
            "EVars" => {
                let vars = list(args.first());
                if vars.is_empty() {
                    return "var ".to_string();
                }
                let first = &vars[0];
                let head = format!(
                    "{}{}",
                    if matches!(field(first, "isStatic"), Some(MacroValue::Bool(true))) {
                        "static "
                    } else {
                        ""
                    },
                    if matches!(field(first, "isFinal"), Some(MacroValue::Bool(true))) {
                        "final "
                    } else {
                        "var "
                    }
                );
                let v: Vec<String> = vars.iter().map(|v| self.print_var(v)).collect();
                format!("{}{}", head, v.join(", "))
            }
            "EBlock" => {
                let el = list(args.first());
                if el.is_empty() {
                    return "{ }".to_string();
                }
                let old = self.tabs.clone();
                self.tabs.push_str(TAB);
                let sep = format!(";\n{}", self.tabs);
                let body = self.print_exprs(&el, &sep);
                let s = format!("{{\n{}{}", self.tabs, body);
                self.tabs = old;
                format!("{};\n{}}}", s, self.tabs)
            }
            "EFor" => format!("for ({}) {}", p(self, 0), p(self, 1)),
            "EIf" => {
                if is_null(args.get(2)) {
                    format!("if ({}) {}", p(self, 0), p(self, 1))
                } else {
                    format!("if ({}) {} else {}", p(self, 0), p(self, 1), p(self, 2))
                }
            }
            "EWhile" => {
                if matches!(args.get(2), Some(MacroValue::Bool(false))) {
                    format!("do {} while ({})", p(self, 1), p(self, 0))
                } else {
                    format!("while ({}) {}", p(self, 0), p(self, 1))
                }
            }
            "ESwitch" => {
                let old = self.tabs.clone();
                self.tabs.push_str(TAB);
                let subject = p(self, 0);
                let cases: Vec<String> = list(args.get(1))
                    .iter()
                    .map(|c| {
                        let values = self.print_exprs(&list(field(c, "values")), ", ");
                        let guard = match field(c, "guard") {
                            None | Some(MacroValue::Null) => ":".to_string(),
                            Some(g) => format!(" if ({}):", self.print_expr(g)),
                        };
                        let body = match field(c, "expr") {
                            None | Some(MacroValue::Null) => String::new(),
                            Some(b) => format!("{};", self.print_expr(b)),
                        };
                        format!("case {values}{guard}{body}")
                    })
                    .collect();
                let mut s = format!(
                    "switch {} {{\n{}{}",
                    subject,
                    self.tabs,
                    cases.join(&format!("\n{}", self.tabs))
                );
                if !is_null(args.get(2)) {
                    let edef = args.get(2).unwrap();
                    let body = if def_of(edef).is_none() {
                        String::new()
                    } else {
                        format!("{};", self.print_expr(edef))
                    };
                    s.push_str(&format!("\n{}default:{}", self.tabs, body));
                }
                self.tabs = old;
                format!("{}\n{}}}", s, self.tabs)
            }
            "ETry" => {
                let mut s = format!("try {}", p(self, 0));
                for c in list(args.get(1)) {
                    let ty = match field(&c, "type") {
                        None | Some(MacroValue::Null) => String::new(),
                        Some(t) => format!(":{}", self.print_complex_type(t)),
                    };
                    s.push_str(&format!(
                        " catch({}{}) {}",
                        text(field(&c, "name")),
                        ty,
                        self.print_expr(field(&c, "expr").unwrap_or(&MacroValue::Null))
                    ));
                }
                s
            }
            "EReturn" => format!("return{}", self.opt(args.first(), " ", Self::print_expr)),
            "EBreak" => "break".to_string(),
            "EContinue" => "continue".to_string(),
            "EUntyped" => format!("untyped {}", p(self, 0)),
            "EThrow" => format!("throw {}", p(self, 0)),
            "ECast" => {
                if is_null(args.get(1)) {
                    format!("cast {}", p(self, 0))
                } else {
                    let ct = self.print_complex_type(args.get(1).unwrap());
                    format!("cast({}, {})", p(self, 0), ct)
                }
            }
            "EIs" => {
                let ct = self.print_complex_type(args.get(1).unwrap_or(&MacroValue::Null));
                format!("{} is {}", p(self, 0), ct)
            }
            "EDisplay" => format!("#DISPLAY({})", p(self, 0)),
            "ETernary" => format!("{} ? {} : {}", p(self, 0), p(self, 1), p(self, 2)),
            "ECheckType" => {
                let ct = self.print_complex_type(args.get(1).unwrap_or(&MacroValue::Null));
                format!("({} : {})", p(self, 0), ct)
            }
            "EMeta" => {
                let meta = args.first().unwrap_or(&MacroValue::Null);
                let inner = args.get(1).unwrap_or(&MacroValue::Null);
                if text(field(meta, "name")) == ":implicitReturn" {
                    if let Some(("EReturn", [ret])) = def_of(inner).as_ref().and_then(variant) {
                        return self.print_expr(ret);
                    }
                }
                format!("{} {}", self.print_metadata(meta), self.print_expr(inner))
            }
            _ => String::new(),
        }
    }
}

impl Default for Printer {
    fn default() -> Self {
        Self::new()
    }
}
