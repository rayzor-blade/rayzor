//! Haxe 5 coroutines, run on fibers (`rayzor.coro.Fiber`).
//!
//! A `@:coroutine function f(a):R` takes its continuation first and returns
//! `haxe.coro.SuspensionResult<R>` (`haxe.Unit` for Void), as Haxe expands
//! it. Its body runs on a fiber started by `Fiber.start`; a coroutine call
//! inside that body suspends the fiber rather than the state machine Haxe
//! generates. The first parameter is typed `rayzor.coro.CoroCont<T>`, which
//! is how lowering recognises a coroutine callee by its type alone.
//! `Coroutine<F>` is that expanded function type. Calls are rewritten during
//! lowering (`AstLowering::coroutine_call`), which needs types.

use parser::haxe_ast::*;
use std::collections::BTreeSet;
use std::sync::Mutex;

/// Marks the closure holding an expanded coroutine's body.
pub(crate) const BODY_META: &str = "rayzor.coroBody";

/// The continuation parameter an expansion adds.
pub(crate) const CONTINUATION: &str = "_hx_continuation";

/// Names a coroutine can be called by: functions, and parameters or
/// variables of a `Coroutine<F>` type. Lowering checks a callee's type only
/// when its name is here.
static NAMES: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Files that declare a coroutine or name a `Coroutine<F>` type: only these
/// can call one with an explicit continuation outside a coroutine body.
static FILES: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

pub(crate) fn file_uses_coroutines(filename: &str) -> bool {
    FILES.lock().is_ok_and(|files| files.contains(filename))
}

pub(crate) fn may_name_coroutine(name: &str) -> bool {
    NAMES.lock().is_ok_and(|names| names.contains(name))
}

fn record(name: &str) {
    if let Ok(mut names) = NAMES.lock() {
        names.insert(name.to_string());
    }
}

/// Expands `file`'s coroutines and `Coroutine<F>` types.
pub fn desugar(file: &mut HaxeFile) {
    let imports_coroutine = file.imports.iter().any(|import| {
        let path = import.path.join(".");
        path == "haxe.coro.Coroutine" || path == "haxe.coro"
    });
    let mut pass = Pass {
        imports_coroutine,
        found: false,
    };
    for declaration in &mut file.declarations {
        pass.declaration(declaration);
    }
    for field in &mut file.module_fields {
        let coroutine = coroutine_options(&field.meta);
        match &mut field.kind {
            ModuleFieldKind::Function(function) => {
                if let Some(options) = coroutine {
                    field.meta.retain(|m| !is_coroutine_meta(m));
                    record(&function.name);
                    pass.expand(function, options, field.span);
                }
                pass.function(function);
            }
            ModuleFieldKind::Var {
                name,
                type_hint,
                expr,
            }
            | ModuleFieldKind::Final {
                name,
                type_hint,
                expr,
            }
            | ModuleFieldKind::Property {
                name,
                type_hint,
                expr,
                ..
            } => {
                pass.typed_name(name, type_hint);
                if let Some(expr) = expr {
                    pass.expr(expr);
                }
            }
        }
    }
    if pass.found
        && let Ok(mut files) = FILES.lock()
    {
        files.insert(file.filename.clone());
    }
}

#[derive(Clone, Copy)]
struct Options {
    transformed: bool,
    no_throw: bool,
}

fn is_coroutine_meta(meta: &Metadata) -> bool {
    meta.name.trim_start_matches(':') == "coroutine"
}

/// `@:coroutine`, `@:coroutine(transformed)`, `@:coroutine(outcome = {noThrow: true})`.
fn coroutine_options(meta: &[Metadata]) -> Option<Options> {
    let meta = meta.iter().find(|m| is_coroutine_meta(m))?;
    let mut options = Options {
        transformed: false,
        no_throw: false,
    };
    for param in &meta.params {
        match &param.kind {
            ExprKind::Ident(name) if name == "transformed" => options.transformed = true,
            ExprKind::Assign { left, right, .. } if matches!(&left.kind, ExprKind::Ident(name) if name == "outcome") => {
                if let ExprKind::Object(fields) = &right.kind {
                    options.no_throw = fields.iter().any(|f| {
                        f.name == "noThrow" && matches!(f.expr.kind, ExprKind::Bool(true))
                    });
                }
            }
            _ => {}
        }
    }
    Some(options)
}

fn path(package: &[&str], name: &str, params: Vec<Type>, span: Span) -> Type {
    Type::Path {
        path: TypePath {
            package: package.iter().map(|p| p.to_string()).collect(),
            name: name.to_string(),
            sub: None,
        },
        params,
        span,
    }
}

fn is_void(ty: &Type) -> bool {
    matches!(ty, Type::Path { path, .. } if path.package.is_empty() && path.name == "Void")
}

fn unit(span: Span) -> Type {
    path(&["haxe"], "Unit", Vec::new(), span)
}

fn continuation_type(arg: Type, span: Span) -> Type {
    path(&["rayzor", "coro"], "CoroCont", vec![arg], span)
}

fn suspension_result(result: Type, span: Span) -> Type {
    path(&["haxe", "coro"], "SuspensionResult", vec![result], span)
}

fn at(kind: ExprKind, span: Span) -> Expr {
    Expr { kind, span }
}

fn field_chain(parts: &[&str], span: Span) -> Expr {
    let mut expr = at(ExprKind::Ident(parts[0].to_string()), span);
    for part in &parts[1..] {
        expr = at(
            ExprKind::Field {
                expr: Box::new(expr),
                field: part.to_string(),
                is_optional: false,
            },
            span,
        );
    }
    expr
}

/// Whether `expr` returns a value, outside any nested function.
fn returns_value(expr: &Expr) -> bool {
    let mut found = false;
    visit_own_returns(expr, &mut |r| found |= r.is_some());
    found
}

fn visit_own_returns(expr: &Expr, f: &mut impl FnMut(Option<&Expr>)) {
    match &expr.kind {
        ExprKind::Return(value) => {
            f(value.as_deref());
            if let Some(value) = value {
                visit_own_returns(value, f);
            }
        }
        ExprKind::Function(_) | ExprKind::Arrow { .. } => {}
        _ => for_each_child(expr, &mut |child| visit_own_returns(child, f)),
    }
}

/// `return;` becomes `return null;`, outside any nested function.
fn bare_returns_to_null(expr: &mut Expr) {
    match &mut expr.kind {
        ExprKind::Return(value @ None) => {
            *value = Some(Box::new(at(ExprKind::Null, expr.span)));
        }
        ExprKind::Function(_) | ExprKind::Arrow { .. } => {}
        _ => for_each_child_mut(expr, &mut bare_returns_to_null),
    }
}

struct Pass {
    imports_coroutine: bool,
    /// The file declares a coroutine or names a `Coroutine<F>` type.
    found: bool,
}

impl Pass {
    fn declaration(&mut self, declaration: &mut TypeDeclaration) {
        match declaration {
            TypeDeclaration::Class(class) => self.fields(&mut class.fields, false),
            TypeDeclaration::Interface(interface) => self.fields(&mut interface.fields, false),
            TypeDeclaration::Abstract(abstract_decl) => {
                self.fields(&mut abstract_decl.fields, true)
            }
            TypeDeclaration::Typedef(typedef) => {
                self.ty(&mut typedef.type_def);
            }
            TypeDeclaration::Enum(enum_decl) => {
                for constructor in &mut enum_decl.constructors {
                    for param in &mut constructor.params {
                        if let Some(ty) = &mut param.type_hint {
                            self.ty(ty);
                        }
                    }
                }
            }
            TypeDeclaration::Conditional(conditional) => {
                each_branch(conditional, &mut |d| self.declaration(d));
            }
        }
    }

    fn fields(&mut self, fields: &mut [ClassField], in_abstract: bool) {
        for field in fields {
            let coroutine = coroutine_options(&field.meta);
            match &mut field.kind {
                ClassFieldKind::Function(function) => {
                    // The metadata is consumed, so a second pass changes nothing.
                    if let Some(options) = coroutine {
                        field.meta.retain(|m| !is_coroutine_meta(m));
                        // Abstract constructors are lowered as plain
                        // constructors: one that never suspends behaves the
                        // same either way.
                        if !(in_abstract && function.name == "new") {
                            record(&function.name);
                            self.expand(function, options, field.span);
                        }
                    }
                    self.function(function);
                }
                ClassFieldKind::Var {
                    name,
                    type_hint,
                    expr,
                }
                | ClassFieldKind::Final {
                    name,
                    type_hint,
                    expr,
                }
                | ClassFieldKind::Property {
                    name,
                    type_hint,
                    expr,
                    ..
                } => {
                    self.typed_name(name, type_hint);
                    if let Some(expr) = expr {
                        self.expr(expr);
                    }
                }
            }
        }
    }

    /// A name declared with a `Coroutine<F>` type can be called as one.
    fn typed_name(&mut self, name: &str, type_hint: &mut Option<Type>) {
        if let Some(ty) = type_hint
            && self.ty(ty)
        {
            record(name);
        }
    }

    /// Rewrites `f` into its expanded form.
    fn expand(&mut self, f: &mut Function, options: Options, span: Span) {
        self.found = true;
        if options.transformed {
            // Already written expanded; its continuation parameter takes the
            // marker type.
            if let Some(first) = f.params.first_mut() {
                let arg = match &first.type_hint {
                    Some(Type::Path { params, .. }) if params.len() == 1 => params[0].clone(),
                    _ => path(&[], "Any", Vec::new(), span),
                };
                first.type_hint = Some(continuation_type(arg, span));
            }
            return;
        }
        let declared = f.return_type.take();
        let (result, is_unit) = match declared {
            Some(ty) if is_void(&ty) => (unit(span), true),
            Some(ty) => (ty, false),
            None if f.body.as_deref().is_some_and(returns_value) => {
                (path(&[], "Dynamic", Vec::new(), span), false)
            }
            None => (unit(span), true),
        };
        f.params.insert(
            0,
            FunctionParam {
                meta: Vec::new(),
                name: CONTINUATION.to_string(),
                type_hint: Some(continuation_type(result.clone(), span)),
                optional: false,
                rest: false,
                default_value: None,
                span,
            },
        );
        f.return_type = Some(suspension_result(result.clone(), span));
        let Some(body) = f.body.take() else {
            return;
        };
        let mut body = *body;
        let mut statements = match body.kind {
            ExprKind::Block(elements) => elements,
            _ => vec![BlockElement::Expr(body)],
        };
        // A Void body yields `null` for its `haxe.Unit`.
        if is_unit {
            let mut block = at(ExprKind::Block(statements), span);
            bare_returns_to_null(&mut block);
            let ExprKind::Block(elements) = block.kind else {
                unreachable!()
            };
            statements = elements;
            statements.push(BlockElement::Expr(at(
                ExprKind::Return(Some(Box::new(at(ExprKind::Null, span)))),
                span,
            )));
        }
        let closure = at(
            ExprKind::Meta {
                meta: Metadata {
                    name: BODY_META.to_string(),
                    params: Vec::new(),
                    span,
                    compile_time: true,
                },
                expr: Box::new(at(
                    ExprKind::Function(Function {
                        name: String::new(),
                        type_params: Vec::new(),
                        params: Vec::new(),
                        return_type: Some(result),
                        body: Some(Box::new(at(ExprKind::Block(statements), span))),
                        span,
                    }),
                    span,
                )),
            },
            span,
        );
        let start = at(
            ExprKind::Call {
                expr: Box::new(field_chain(&["rayzor", "coro", "Fiber", "start"], span)),
                args: vec![
                    at(ExprKind::Ident(CONTINUATION.to_string()), span),
                    closure,
                    at(ExprKind::Bool(options.no_throw), span),
                ],
            },
            span,
        );
        f.body = Some(Box::new(at(
            ExprKind::Block(vec![BlockElement::Expr(at(
                ExprKind::Return(Some(Box::new(start))),
                span,
            ))]),
            span,
        )));
    }

    fn function(&mut self, f: &mut Function) {
        for param in &mut f.params {
            self.typed_name(&param.name, &mut param.type_hint);
            if let Some(default) = &mut param.default_value {
                self.expr(default);
            }
        }
        if let Some(ty) = &mut f.return_type {
            self.ty(ty);
        }
        if let Some(body) = &mut f.body {
            self.expr(body);
        }
    }

    /// Expands `Coroutine<F>` within `ty`; true when `ty` itself was one.
    fn ty(&mut self, ty: &mut Type) -> bool {
        let coroutine = match ty {
            Type::Path { path, params, .. } if path.name == "Coroutine" && params.len() == 1 => {
                let qualified = path.package == ["haxe", "coro"];
                qualified || (path.package.is_empty() && self.imports_coroutine)
            }
            _ => false,
        };
        match ty {
            Type::Path { params, .. } => {
                for param in params.iter_mut() {
                    self.ty(param);
                }
            }
            Type::Function { params, ret, .. } => {
                for param in params.iter_mut() {
                    self.ty(param);
                }
                self.ty(ret);
            }
            Type::Anonymous { fields, .. } => {
                for field in fields {
                    if self.ty(&mut field.type_hint) {
                        record(&field.name);
                    }
                }
            }
            Type::Optional { inner, .. }
            | Type::Parenthesis { inner, .. }
            | Type::Named { inner, .. } => {
                self.ty(inner);
            }
            Type::Intersection { left, right, .. } => {
                self.ty(left);
                self.ty(right);
            }
            Type::Wildcard { .. } | Type::Const { .. } => {}
        }
        if !coroutine {
            return false;
        }
        self.found = true;
        let Type::Path { params, span, .. } = ty else {
            return false;
        };
        let span = *span;
        let mut function = params[0].clone();
        while let Type::Parenthesis { inner, .. } = function {
            function = *inner;
        }
        let Type::Function {
            params: fn_params,
            ret,
            ..
        } = function
        else {
            return false;
        };
        // `Void -> R` takes no arguments.
        let mut fn_params: Vec<Type> = fn_params.into_iter().filter(|p| !is_void(p)).collect();
        let result = if is_void(&ret) { unit(span) } else { *ret };
        fn_params.insert(0, continuation_type(result.clone(), span));
        *ty = Type::Function {
            params: fn_params,
            ret: Box::new(suspension_result(result, span)),
            span,
        };
        true
    }

    fn expr(&mut self, expr: &mut Expr) {
        match &mut expr.kind {
            ExprKind::Meta { meta, expr: inner } if is_coroutine_meta(meta) => {
                let options = coroutine_options(std::slice::from_ref(meta));
                if let (Some(options), ExprKind::Function(function)) = (options, &mut inner.kind) {
                    if !function.name.is_empty() {
                        record(&function.name);
                    }
                    self.expand(function, options, inner.span);
                    self.function(function);
                    // The metadata has done its work; the function stays.
                    let function = std::mem::replace(&mut inner.kind, ExprKind::Null);
                    expr.kind = function;
                    return;
                }
                self.expr(inner);
            }
            ExprKind::Function(function) => self.function(function),
            ExprKind::Arrow { params, expr: body } => {
                for param in params {
                    if let Some(ty) = &mut param.type_hint
                        && self.ty(ty)
                    {
                        record(&param.name);
                    }
                }
                self.expr(body);
            }
            ExprKind::Var {
                name,
                type_hint,
                expr: init,
                ..
            }
            | ExprKind::Final {
                name,
                type_hint,
                expr: init,
                ..
            } => {
                self.typed_name(name, type_hint);
                if let Some(init) = init {
                    self.expr(init);
                }
            }
            ExprKind::New { params, args, .. } => {
                for param in params {
                    self.ty(param);
                }
                for arg in args {
                    self.expr(arg);
                }
            }
            ExprKind::Cast {
                expr: inner,
                type_hint,
            } => {
                if let Some(ty) = type_hint {
                    self.ty(ty);
                }
                self.expr(inner);
            }
            ExprKind::TypeCheck {
                expr: inner,
                type_hint,
            } => {
                self.ty(type_hint);
                self.expr(inner);
            }
            ExprKind::Try {
                expr: body,
                catches,
                finally_block,
            } => {
                self.expr(body);
                for catch in catches {
                    if let Some(ty) = &mut catch.type_hint {
                        self.ty(ty);
                    }
                    if let Some(filter) = &mut catch.filter {
                        self.expr(filter);
                    }
                    self.expr(&mut catch.body);
                }
                if let Some(finally_block) = finally_block {
                    self.expr(finally_block);
                }
            }
            ExprKind::MacroType(ty) => {
                self.ty(ty);
            }
            _ => for_each_child_mut(expr, &mut |child| self.expr(child)),
        }
    }
}

/// Calls `f` on each direct subexpression of `expr`.
fn for_each_child(expr: &Expr, f: &mut impl FnMut(&Expr)) {
    // The mutable walk is the single source of truth for the shape; a
    // clone keeps this read-only view honest without duplicating it.
    let mut copy = expr.clone();
    for_each_child_mut(&mut copy, &mut |child| f(child));
}

/// Calls `f` on each direct subexpression of `expr`.
fn for_each_child_mut(expr: &mut Expr, f: &mut impl FnMut(&mut Expr)) {
    match &mut expr.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::String(_)
        | ExprKind::Bool(_)
        | ExprKind::Null
        | ExprKind::This
        | ExprKind::Super
        | ExprKind::Regex { .. }
        | ExprKind::Ident(_)
        | ExprKind::Break
        | ExprKind::Continue
        | ExprKind::MacroType(_)
        | ExprKind::TypeDecl(_)
        | ExprKind::Return(None)
        | ExprKind::DollarIdent { arg: None, .. } => {}
        ExprKind::Field { expr, .. }
        | ExprKind::Unary { expr, .. }
        | ExprKind::Throw(expr)
        | ExprKind::Spread(expr)
        | ExprKind::Untyped(expr)
        | ExprKind::Paren(expr)
        | ExprKind::Inline(expr)
        | ExprKind::Macro(expr)
        | ExprKind::Reify(expr)
        | ExprKind::Meta { expr, .. }
        | ExprKind::Cast { expr, .. }
        | ExprKind::TypeCheck { expr, .. }
        | ExprKind::Return(Some(expr))
        | ExprKind::DollarIdent {
            arg: Some(expr), ..
        } => f(expr),
        ExprKind::Index { expr, index } => {
            f(expr);
            f(index);
        }
        ExprKind::Call { expr, args } => {
            f(expr);
            args.iter_mut().for_each(f);
        }
        ExprKind::New { args, .. } | ExprKind::Array(args) | ExprKind::Tuple(args) => {
            args.iter_mut().for_each(f)
        }
        ExprKind::CompilerSpecific { code, args, .. } => {
            f(code);
            args.iter_mut().for_each(f);
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Assign { left, right, .. } => {
            f(left);
            f(right);
        }
        ExprKind::Ternary {
            cond,
            then_expr,
            else_expr,
        } => {
            f(cond);
            f(then_expr);
            f(else_expr);
        }
        ExprKind::Map(items) => items.iter_mut().for_each(|(key, value)| {
            f(key);
            f(value);
        }),
        ExprKind::Object(fields) => fields.iter_mut().for_each(|field| f(&mut field.expr)),
        ExprKind::StringInterpolation(parts) => {
            for part in parts {
                if let StringPart::Interpolation(expr) = part {
                    f(expr);
                }
            }
        }
        ExprKind::Block(elements) => {
            for element in elements {
                block_element(element, f);
            }
        }
        ExprKind::Var { expr, .. } | ExprKind::Final { expr, .. } => {
            if let Some(expr) = expr {
                f(expr);
            }
        }
        ExprKind::Function(function) => {
            for param in &mut function.params {
                if let Some(default) = &mut param.default_value {
                    f(default);
                }
            }
            if let Some(body) = &mut function.body {
                f(body);
            }
        }
        ExprKind::Arrow { expr, .. } => f(expr),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            f(cond);
            f(then_branch);
            if let Some(else_branch) = else_branch {
                f(else_branch);
            }
        }
        ExprKind::Switch {
            expr,
            cases,
            default,
        } => {
            f(expr);
            for case in cases {
                if let Some(guard) = &mut case.guard {
                    f(guard);
                }
                f(&mut case.body);
            }
            if let Some(default) = default {
                f(default);
            }
        }
        ExprKind::For { iter, body, .. } => {
            f(iter);
            f(body);
        }
        ExprKind::While { cond, body } | ExprKind::DoWhile { body, cond } => {
            f(cond);
            f(body);
        }
        ExprKind::Try {
            expr,
            catches,
            finally_block,
        } => {
            f(expr);
            for catch in catches {
                if let Some(filter) = &mut catch.filter {
                    f(filter);
                }
                f(&mut catch.body);
            }
            if let Some(finally_block) = finally_block {
                f(finally_block);
            }
        }
        ExprKind::ArrayComprehension { for_parts, expr } => {
            for part in for_parts {
                comprehension(part, f);
            }
            f(expr);
        }
        ExprKind::MapComprehension {
            for_parts,
            key,
            value,
        } => {
            for part in for_parts {
                comprehension(part, f);
            }
            f(key);
            f(value);
        }
    }
}

fn block_element(element: &mut BlockElement, f: &mut impl FnMut(&mut Expr)) {
    match element {
        BlockElement::Expr(expr) => f(expr),
        BlockElement::Conditional(conditional) => {
            each_branch(conditional, &mut |e| block_element(e, f));
        }
        BlockElement::Import(_) | BlockElement::Using(_) => {}
    }
}

fn comprehension(part: &mut ComprehensionFor, f: &mut impl FnMut(&mut Expr)) {
    f(&mut part.iter);
}

/// Every branch of a conditional-compilation block, whichever is active.
fn each_branch<T>(conditional: &mut ConditionalCompilation<T>, f: &mut impl FnMut(&mut T)) {
    conditional.if_branch.content.iter_mut().for_each(&mut *f);
    for branch in &mut conditional.elseif_branches {
        branch.content.iter_mut().for_each(&mut *f);
    }
    if let Some(branch) = &mut conditional.else_branch {
        branch.iter_mut().for_each(&mut *f);
    }
}

use super::{AstLowering, LoweringResult};
use crate::tast::TypeId;
use crate::tast::core::TypeKind;
use crate::tast::node::TypedExpression;

impl AstLowering<'_> {
    fn in_coroutine_body(&self) -> bool {
        self.coroutine_scopes.last() == Some(&true)
    }

    /// The parameters and result of a coroutine's expanded function type.
    fn coroutine_signature(&self, ty: TypeId) -> Option<(Vec<TypeId>, TypeId)> {
        let table = self.context.type_table.borrow();
        let ty = Self::resolve_alias_chain(&table, ty);
        let Some(TypeKind::Function {
            params,
            return_type,
            ..
        }) = table.get(ty).map(|t| &t.kind)
        else {
            return None;
        };
        let first = Self::resolve_alias_chain(&table, *params.first()?);
        let symbol = match table.get(first).map(|t| &t.kind) {
            Some(TypeKind::Abstract { symbol_id, .. }) => *symbol_id,
            Some(TypeKind::GenericInstance { base_type, .. }) => {
                match table.get(*base_type).map(|t| &t.kind) {
                    Some(TypeKind::Abstract { symbol_id, .. }) => *symbol_id,
                    _ => return None,
                }
            }
            _ => return None,
        };
        let name = self.context.symbol_table.get_symbol(symbol)?.name;
        (self.context.string_interner.get(name) == Some("CoroCont"))
            .then(|| (params.clone(), *return_type))
    }

    /// Whether `expr`'s type may have the member `name`. Probing a member a
    /// class lacks lowers a failed access, so a class is asked first.
    fn may_have_member(&mut self, expr: &Expr, name: &str) -> bool {
        let Ok(typed) = self.lower_expression(expr) else {
            return false;
        };
        let class = {
            let table = self.context.type_table.borrow();
            let ty = Self::resolve_alias_chain(&table, typed.expr_type);
            match table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Class { symbol_id, .. }) => Some(*symbol_id),
                Some(TypeKind::GenericInstance { base_type, .. }) => {
                    match table.get(*base_type).map(|t| &t.kind) {
                        Some(TypeKind::Class { symbol_id, .. }) => Some(*symbol_id),
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        let Some(class) = class else {
            return true;
        };
        let Some(owner) = self.context.symbol_table.get_symbol(class).and_then(|s| {
            self.context
                .string_interner
                .get(s.qualified_name.unwrap_or(s.name))
                .map(str::to_string)
        }) else {
            return true;
        };
        let Some(index) = self.static_sig_index.clone() else {
            return true;
        };
        let declared = index
            .borrow_mut()
            .instance_method_sig(&owner, name)
            .is_some();
        declared
    }

    /// The declared type of `expr` when it is an async iterator whose type
    /// is not yet known: `x.iterator()` with `x`'s `iterator` declared to
    /// return `AsyncIterator<T>`.
    fn declared_async_iterator(&mut self, expr: &Expr) -> Option<Type> {
        let ExprKind::Call { expr: callee, args } = &expr.kind else {
            return None;
        };
        let ExprKind::Field {
            expr: receiver,
            field,
            ..
        } = &callee.kind
        else {
            return None;
        };
        if field != "iterator" || !args.is_empty() {
            return None;
        }
        let Ok(receiver) = self.lower_expression(receiver) else {
            return None;
        };
        let owner = {
            let table = self.context.type_table.borrow();
            let ty = Self::resolve_alias_chain(&table, receiver.expr_type);
            match table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Class { symbol_id, .. } | TypeKind::Abstract { symbol_id, .. }) => {
                    *symbol_id
                }
                _ => return None,
            }
        };
        let Some(name) = self.context.symbol_table.get_symbol(owner).and_then(|s| {
            self.context
                .string_interner
                .get(s.qualified_name.unwrap_or(s.name))
                .map(str::to_string)
        }) else {
            return None;
        };
        let Some(index) = self.static_sig_index.clone() else {
            return None;
        };
        let sig = index.borrow_mut().instance_method_sig(&name, "iterator");
        sig.and_then(|sig| sig.return_type)
            .filter(|ty| matches!(ty, Type::Path { path, .. } if path.name == "AsyncIterator"))
    }

    /// `for (v in x)` in a coroutine body over an async iterator (`x`, or
    /// `x.iterator()`, has a coroutine `hasNext`), which wins over any other
    /// way to iterate `x`:
    /// `{ final it = x; while (it.hasNext()) { var v = it.next(); body } }`.
    pub(crate) fn async_for(
        &mut self,
        span: Span,
        var: &str,
        iter: &Expr,
        body: &Expr,
    ) -> LoweringResult<Option<TypedExpression>> {
        if !self.in_coroutine_body() {
            return Ok(None);
        }
        let field = |expr: Expr, name: &str| {
            at(
                ExprKind::Field {
                    expr: Box::new(expr),
                    field: name.to_string(),
                    is_optional: false,
                },
                span,
            )
        };
        let call = |callee: Expr| {
            at(
                ExprKind::Call {
                    expr: Box::new(callee),
                    args: Vec::new(),
                },
                span,
            )
        };
        let has_coroutine_has_next = |this: &mut Self, iterator: &Expr| {
            let probe = this.lower_expression(&field(iterator.clone(), "hasNext"));
            probe
                .ok()
                .is_some_and(|probe| this.coroutine_signature(probe.expr_type).is_some())
        };
        // The declared type, when lowering cannot yet type the iterator.
        let (iterator, declared) =
            if self.may_have_member(iter, "hasNext") && has_coroutine_has_next(self, iter) {
                (iter.clone(), None)
            } else if let Some(declared) = self.declared_async_iterator(iter) {
                (iter.clone(), Some(declared))
            } else {
                let iterator = call(field(iter.clone(), "iterator"));
                if has_coroutine_has_next(self, &iterator) {
                    (iterator, None)
                } else if let Some(declared) = self.declared_async_iterator(&iterator) {
                    (iterator, Some(declared))
                } else {
                    return Ok(None);
                }
            };
        let it = format!("_hx_async_iterator{}", span.start);
        let ident = || at(ExprKind::Ident(it.clone()), span);
        // A declared iterator whose type lowering cannot see is called in
        // its expanded form directly: `(Fiber.await(it.hasNext(cont)) : Bool)`.
        let has_next = if declared.is_some() {
            let suspending = at(
                ExprKind::Call {
                    expr: Box::new(field(ident(), "hasNext")),
                    args: vec![call(field_chain(
                        &["rayzor", "coro", "Fiber", "current"],
                        span,
                    ))],
                },
                span,
            );
            at(
                ExprKind::TypeCheck {
                    expr: Box::new(at(
                        ExprKind::Call {
                            expr: Box::new(field_chain(
                                &["rayzor", "coro", "Fiber", "await"],
                                span,
                            )),
                            args: vec![suspending],
                        },
                        span,
                    )),
                    type_hint: path(&[], "Bool", Vec::new(), span),
                },
                span,
            )
        } else {
            call(field(ident(), "hasNext"))
        };
        let declare = |name: &str, type_hint: Option<Type>, init: Expr, is_final: bool| {
            let name = name.to_string();
            let init = Some(Box::new(init));
            at(
                if is_final {
                    ExprKind::Final {
                        name,
                        type_hint,
                        expr: init,
                        attrs: Default::default(),
                    }
                } else {
                    ExprKind::Var {
                        name,
                        type_hint,
                        expr: init,
                        attrs: Default::default(),
                    }
                },
                span,
            )
        };
        let rewritten = at(
            ExprKind::Block(vec![
                BlockElement::Expr(declare(&it, declared, iterator, true)),
                BlockElement::Expr(at(
                    ExprKind::While {
                        cond: Box::new(has_next),
                        body: Box::new(at(
                            ExprKind::Block(vec![
                                BlockElement::Expr(declare(
                                    var,
                                    None,
                                    call(field(ident(), "next")),
                                    false,
                                )),
                                BlockElement::Expr(body.clone()),
                            ]),
                            span,
                        )),
                    },
                    span,
                )),
            ]),
            span,
        );
        self.lower_expression(&rewritten).map(Some)
    }

    /// A call to a coroutine. Inside a coroutine body, `g(x)` becomes
    /// `Fiber.await(g(Fiber.current(), x))`, typed as `g`'s result. Outside
    /// one, `f(cont, x)` becomes `f(cont, x).resolveTo(cont)` so a
    /// coroutine that completes without suspending resumes `cont`.
    pub(crate) fn coroutine_call(
        &mut self,
        call: &Expr,
        callee: &Expr,
        args: &[Expr],
    ) -> LoweringResult<Option<TypedExpression>> {
        let key = (call.span.start, call.span.end);
        if self.coroutine_rewritten.contains(&key) {
            return Ok(None);
        }
        if !self.in_coroutine_body() && !self.file_uses_coroutines {
            return Ok(None);
        }
        let named = match &callee.kind {
            ExprKind::Ident(name) | ExprKind::Field { field: name, .. } => may_name_coroutine(name),
            ExprKind::Function(_) | ExprKind::Paren(_) | ExprKind::Meta { .. } => true,
            _ => false,
        };
        if !named {
            return Ok(None);
        }
        // Typing the callee can lower a body that makes this same call (a
        // coroutine calling itself); that inner probe is left as written.
        let probe = (callee.span.start, callee.span.end);
        if !self.coroutine_probing.insert(probe) {
            return Ok(None);
        }
        let callee_typed = self.lower_expression(callee);
        self.coroutine_probing.remove(&probe);
        let Ok(callee_typed) = callee_typed else {
            return Ok(None);
        };
        let Some((params, _)) = self.coroutine_signature(callee_typed.expr_type) else {
            return Ok(None);
        };
        let span = call.span;
        let in_body = self.in_coroutine_body();
        let rewritten = if in_body && args.len() + 1 == params.len() {
            let mut call_args = vec![at(
                ExprKind::Call {
                    expr: Box::new(field_chain(&["rayzor", "coro", "Fiber", "current"], span)),
                    args: Vec::new(),
                },
                span,
            )];
            call_args.extend(args.iter().cloned());
            let suspending = at(
                ExprKind::Call {
                    expr: Box::new(callee.clone()),
                    args: call_args,
                },
                span,
            );
            at(
                ExprKind::Call {
                    expr: Box::new(field_chain(&["rayzor", "coro", "Fiber", "await"], span)),
                    args: vec![suspending],
                },
                span,
            )
        } else if !in_body && !args.is_empty() && args.len() == params.len() {
            let cont = format!("_hx_cont{}", span.start);
            let value = format!("_hx_result{}", span.start);
            let ident = |name: &str| at(ExprKind::Ident(name.to_string()), span);
            let mut call_args = vec![ident(&cont)];
            call_args.extend(args[1..].iter().cloned());
            let declare = |name: &str, type_hint: Option<Type>, expr: Expr| {
                BlockElement::Expr(at(
                    ExprKind::Final {
                        name: name.to_string(),
                        type_hint,
                        expr: Some(Box::new(expr)),
                        attrs: Default::default(),
                    },
                    span,
                ))
            };
            at(
                ExprKind::Block(vec![
                    declare(&cont, None, args[0].clone()),
                    declare(
                        &value,
                        None,
                        at(
                            ExprKind::Call {
                                expr: Box::new(callee.clone()),
                                args: call_args,
                            },
                            span,
                        ),
                    ),
                    BlockElement::Expr(at(
                        ExprKind::Call {
                            expr: Box::new(at(
                                ExprKind::Field {
                                    expr: Box::new(ident(&value)),
                                    field: "resolveTo".to_string(),
                                    is_optional: false,
                                },
                                span,
                            )),
                            args: vec![ident(&cont)],
                        },
                        span,
                    )),
                    BlockElement::Expr(ident(&value)),
                ]),
                span,
            )
        } else {
            return Ok(None);
        };
        self.coroutine_rewritten.insert(key);
        let lowered = self.lower_expression(&rewritten);
        self.coroutine_rewritten.remove(&key);
        lowered.map(Some)
    }
}
