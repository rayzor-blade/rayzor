//! Class specializations whose type parameters bind constant expressions.

use super::{AstLowering, LoweringError, LoweringResult};
use crate::tast::SymbolId;
use parser::haxe_ast::*;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub(super) struct Template {
    class: ClassDecl,
    pack: Vec<String>,
}

fn is_const(param: &TypeParam) -> bool {
    param
        .meta
        .iter()
        .any(|meta| meta.name.trim_start_matches(':') == "const")
}

pub(super) fn is_template(class: &ClassDecl) -> bool {
    class
        .meta
        .iter()
        .any(|meta| meta.name.trim_start_matches(':') == "generic")
        && class.type_params.iter().any(is_const)
}

impl AstLowering<'_> {
    pub(crate) fn seed_const_generics<'f>(&mut self, files: impl Iterator<Item = &'f HaxeFile>) {
        for file in files {
            let pack = file
                .package
                .as_ref()
                .map(|p| p.path.clone())
                .unwrap_or_default();
            for declaration in &file.declarations {
                if let TypeDeclaration::Class(class) = declaration
                    && is_template(class)
                {
                    let name = pack
                        .iter()
                        .chain(std::iter::once(&class.name))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(".");
                    self.const_generic_templates.insert(
                        name,
                        Template {
                            class: class.clone(),
                            pack: pack.clone(),
                        },
                    );
                }
            }
        }
    }

    pub(crate) fn const_generic_type(
        &mut self,
        symbol: SymbolId,
        params: &[Type],
        constructing: bool,
        span: Span,
    ) -> LoweringResult<Option<Type>> {
        let Some(name) = self
            .context
            .symbol_table
            .get_symbol(symbol)
            .and_then(|s| {
                self.context
                    .string_interner
                    .get(s.qualified_name.unwrap_or(s.name))
            })
            .map(str::to_string)
        else {
            return Ok(None);
        };
        let Some(template) = self.const_generic_templates.get(&name).cloned() else {
            return Ok(None);
        };
        if params.is_empty() {
            if constructing
                && let Some(hint) = self.context.expected_new_type_hint
                && let Some(owner) = self.resolve_type_to_class_symbol(hint)
                && let Some(qualified) = self.context.symbol_table.get_symbol(owner).and_then(|s| {
                    self.context
                        .string_interner
                        .get(s.qualified_name.unwrap_or(s.name))
                })
                && self.const_generic_origins.get(qualified) == Some(&name)
            {
                let mut parts = qualified.split('.').map(str::to_string).collect::<Vec<_>>();
                let class = parts.pop().unwrap();
                return Ok(Some(Type::Path {
                    path: TypePath {
                        package: parts,
                        name: class,
                        sub: None,
                    },
                    params: Vec::new(),
                    span,
                }));
            }
            if constructing {
                return Err(LoweringError::SemanticError {
                    message: format!("constant type arguments required for '{name}'"),
                    location: self.context.create_location_from_span(span),
                });
            }
            return Ok(None);
        }
        if params.len() != template.class.type_params.len() {
            return Err(LoweringError::SemanticError {
                message: format!(
                    "constant generic '{name}' expects {} type arguments",
                    template.class.type_params.len()
                ),
                location: self.context.create_location_from_span(span),
            });
        }
        let mut bindings = BTreeMap::new();
        let mut constants = Vec::new();
        let mut regular = Vec::new();
        for (param, argument) in template.class.type_params.iter().zip(params) {
            if is_const(param) {
                let Type::Const { value, .. } = argument else {
                    return Err(LoweringError::SemanticError {
                        message: format!(
                            "constant expression required for '{}.{}'",
                            name, param.name
                        ),
                        location: self.context.create_location_from_span(span),
                    });
                };
                let value =
                    normalized_constant(value).ok_or_else(|| LoweringError::SemanticError {
                        message: format!(
                            "constant expression required for '{}.{}'",
                            name, param.name
                        ),
                        location: self.context.create_location_from_span(span),
                    })?;
                constants.push(format!("{:?}", value.kind));
                bindings.insert(param.name.clone(), value);
            } else {
                regular.push(argument.clone());
            }
        }
        let key = (name.clone(), constants);
        if let Some(Type::Path { path, .. }) = self.const_generic_instances.get(&key) {
            return Ok(Some(Type::Path {
                path: path.clone(),
                params: regular,
                span,
            }));
        }
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hash);
        let mut class = template.class;
        class.name = format!("{}__const_{:016x}", class.name, hash.finish());
        class.type_params.retain(|param| !is_const(param));
        if class.type_params.is_empty() {
            class
                .meta
                .retain(|meta| meta.name.trim_start_matches(':') != "generic");
        }
        let mut substitution = Substitution {
            values: bindings,
            shadowed: BTreeSet::new(),
        };
        substitution.class(&mut class);
        let qualified = template
            .pack
            .iter()
            .chain(std::iter::once(&class.name))
            .cloned()
            .collect::<Vec<_>>()
            .join(".");
        let result = Type::Path {
            path: TypePath {
                package: template.pack.clone(),
                name: class.name.clone(),
                sub: None,
            },
            params: regular,
            span,
        };
        // Cache before typing so recursive references reuse the same declaration.
        self.const_generic_instances
            .insert(key.clone(), result.clone());
        self.const_generic_origins.insert(qualified.clone(), name);
        let lowered =
            self.lower_generated_declaration(&TypeDeclaration::Class(class), &template.pack);
        if lowered.is_err() {
            self.const_generic_instances.remove(&key);
            self.const_generic_origins.remove(&qualified);
        }
        lowered?;
        Ok(Some(result))
    }
}

fn normalized_constant(expr: &Expr) -> Option<Expr> {
    let kind = match &expr.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::String(_)
        | ExprKind::Bool(_)
        | ExprKind::Null
        | ExprKind::Regex { .. } => expr.kind.clone(),
        ExprKind::Paren(inner) => return normalized_constant(inner),
        ExprKind::Unary {
            op: UnaryOp::Neg,
            expr: inner,
        } => match normalized_constant(inner)?.kind {
            ExprKind::Int(value) => ExprKind::Int(value.checked_neg()?),
            ExprKind::Float(value) => ExprKind::Float(-value),
            _ => return None,
        },
        _ => return None,
    };
    Some(Expr {
        kind,
        span: expr.span,
    })
}

struct Substitution {
    values: BTreeMap<String, Expr>,
    shadowed: BTreeSet<String>,
}

impl Substitution {
    fn ty(&mut self, ty: &mut Type) {
        if let Type::Path { path, params, span } = ty
            && path.package.is_empty()
            && path.sub.is_none()
            && params.is_empty()
            && let Some(value) = self
                .values
                .get(&path.name)
                .filter(|_| !self.shadowed.contains(&path.name))
        {
            *ty = Type::Const {
                value: Box::new(value.clone()),
                span: *span,
            };
            return;
        }
        match ty {
            Type::Path { params, .. } => params.iter_mut().for_each(|ty| self.ty(ty)),
            Type::Function { params, ret, .. } => {
                params.iter_mut().for_each(|ty| self.ty(ty));
                self.ty(ret);
            }
            Type::Anonymous { fields, .. } => fields
                .iter_mut()
                .for_each(|field| self.ty(&mut field.type_hint)),
            Type::Optional { inner, .. } | Type::Parenthesis { inner, .. } => self.ty(inner),
            Type::Intersection { left, right, .. } => {
                self.ty(left);
                self.ty(right);
            }
            Type::Const { value, .. } => self.expr(value),
            Type::Wildcard { .. } => {}
        }
    }

    fn class(&mut self, class: &mut ClassDecl) {
        if let Some(ty) = &mut class.extends {
            self.ty(ty);
        }
        class.implements.iter_mut().for_each(|ty| self.ty(ty));
        for field in &mut class.fields {
            for meta in &mut field.meta {
                meta.params.iter_mut().for_each(|expr| self.expr(expr));
            }
            match &mut field.kind {
                ClassFieldKind::Function(function) => self.function(function),
                ClassFieldKind::Var {
                    type_hint, expr, ..
                }
                | ClassFieldKind::Final {
                    type_hint, expr, ..
                }
                | ClassFieldKind::Property {
                    type_hint, expr, ..
                } => {
                    if let Some(ty) = type_hint {
                        self.ty(ty);
                    }
                    if let Some(expr) = expr {
                        self.expr(expr);
                    }
                }
            }
        }
    }

    fn function(&mut self, function: &mut Function) {
        let shadowed = self.shadowed.clone();
        self.shadowed
            .extend(function.type_params.iter().map(|p| p.name.clone()));
        for param in &mut function.params {
            if let Some(ty) = &mut param.type_hint {
                self.ty(ty);
            }
            if let Some(expr) = &mut param.default_value {
                self.expr(expr);
            }
            self.shadowed.insert(param.name.clone());
        }
        if let Some(ty) = &mut function.return_type {
            self.ty(ty);
        }
        if let Some(body) = &mut function.body {
            self.expr(body);
        }
        self.shadowed = shadowed;
    }

    fn pattern(&mut self, pattern: &mut Pattern) {
        match pattern {
            Pattern::Const(expr) => self.expr(expr),
            Pattern::Var(name) | Pattern::Type { var: name, .. } => {
                self.shadowed.insert(name.clone());
            }
            Pattern::Constructor { params, .. } | Pattern::Array(params) | Pattern::Or(params) => {
                params.iter_mut().for_each(|p| self.pattern(p))
            }
            Pattern::ArrayRest { elements, rest } => {
                elements.iter_mut().for_each(|p| self.pattern(p));
                self.shadowed.extend(rest.iter().cloned());
            }
            Pattern::Object { fields } => fields.iter_mut().for_each(|(_, p)| self.pattern(p)),
            Pattern::Extractor { expr, value } => {
                self.expr(expr);
                self.pattern(value);
            }
            Pattern::Bind { name, pattern } => {
                self.shadowed.insert(name.clone());
                self.pattern(pattern);
            }
            Pattern::Null | Pattern::Underscore => {}
        }
    }

    fn expr(&mut self, expr: &mut Expr) {
        if let ExprKind::Ident(name) = &expr.kind
            && !self.shadowed.contains(name)
            && let Some(value) = self.values.get(name)
        {
            expr.kind = value.kind.clone();
            return;
        }
        let shadowed = self.shadowed.clone();
        match &mut expr.kind {
            ExprKind::Block(elements) => {
                for element in elements {
                    if let BlockElement::Expr(expr) = element {
                        self.expr(expr);
                        if let ExprKind::Var { name, .. } | ExprKind::Final { name, .. } =
                            &expr.kind
                        {
                            self.shadowed.insert(name.clone());
                        }
                    }
                }
            }
            ExprKind::Var {
                type_hint, expr, ..
            }
            | ExprKind::Final {
                type_hint, expr, ..
            } => {
                if let Some(ty) = type_hint {
                    self.ty(ty);
                }
                if let Some(expr) = expr {
                    self.expr(expr);
                }
            }
            ExprKind::Function(function) => self.function(function),
            ExprKind::Arrow { params, expr } => {
                for param in params {
                    if let Some(ty) = &mut param.type_hint {
                        self.ty(ty);
                    }
                    self.shadowed.insert(param.name.clone());
                }
                self.expr(expr);
            }
            ExprKind::New { params, args, .. } => {
                params.iter_mut().for_each(|ty| self.ty(ty));
                args.iter_mut().for_each(|expr| self.expr(expr));
            }
            ExprKind::Call { expr, args } => {
                self.expr(expr);
                args.iter_mut().for_each(|expr| self.expr(expr));
            }
            ExprKind::Field { expr, .. }
            | ExprKind::Unary { expr, .. }
            | ExprKind::Throw(expr)
            | ExprKind::Spread(expr)
            | ExprKind::Untyped(expr)
            | ExprKind::Paren(expr)
            | ExprKind::Inline(expr)
            | ExprKind::Macro(expr)
            | ExprKind::Reify(expr) => self.expr(expr),
            ExprKind::Return(Some(expr))
            | ExprKind::DollarIdent {
                arg: Some(expr), ..
            } => self.expr(expr),
            ExprKind::Binary { left, right, .. } | ExprKind::Assign { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            ExprKind::Index { expr, index } => {
                self.expr(expr);
                self.expr(index);
            }
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                self.expr(cond);
                self.expr(then_expr);
                self.expr(else_expr);
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.expr(then_branch);
                if let Some(expr) = else_branch {
                    self.expr(expr);
                }
            }
            ExprKind::Array(items) | ExprKind::Tuple(items) => {
                items.iter_mut().for_each(|expr| self.expr(expr))
            }
            ExprKind::Map(items) => items.iter_mut().for_each(|(key, value)| {
                self.expr(key);
                self.expr(value);
            }),
            ExprKind::Object(fields) => fields
                .iter_mut()
                .for_each(|field| self.expr(&mut field.expr)),
            ExprKind::StringInterpolation(parts) => {
                for part in parts {
                    if let StringPart::Interpolation(expr) = part {
                        self.expr(expr);
                    }
                }
            }
            ExprKind::Cast { expr, type_hint } => {
                self.expr(expr);
                if let Some(ty) = type_hint {
                    self.ty(ty);
                }
            }
            ExprKind::TypeCheck { expr, type_hint } => {
                self.expr(expr);
                self.ty(type_hint);
            }
            ExprKind::MacroType(ty) => self.ty(ty),
            ExprKind::For {
                var,
                key_var,
                iter,
                body,
            } => {
                self.expr(iter);
                self.shadowed.insert(var.clone());
                self.shadowed.extend(key_var.iter().cloned());
                self.expr(body);
            }
            ExprKind::While { cond, body } | ExprKind::DoWhile { cond, body } => {
                self.expr(cond);
                self.expr(body);
            }
            ExprKind::Switch {
                expr,
                cases,
                default,
            } => {
                self.expr(expr);
                for case in cases {
                    case.patterns.iter_mut().for_each(|p| self.pattern(p));
                    if let Some(guard) = &mut case.guard {
                        self.expr(guard);
                    }
                    self.expr(&mut case.body);
                    self.shadowed = shadowed.clone();
                }
                if let Some(expr) = default {
                    self.expr(expr);
                }
            }
            ExprKind::Try {
                expr,
                catches,
                finally_block,
            } => {
                self.expr(expr);
                for catch in catches {
                    if let Some(ty) = &mut catch.type_hint {
                        self.ty(ty);
                    }
                    self.shadowed.insert(catch.var.clone());
                    if let Some(filter) = &mut catch.filter {
                        self.expr(filter);
                    }
                    self.expr(&mut catch.body);
                    self.shadowed = shadowed.clone();
                }
                if let Some(expr) = finally_block {
                    self.expr(expr);
                }
            }
            ExprKind::ArrayComprehension { for_parts, expr } => {
                self.comprehension(for_parts);
                self.expr(expr);
            }
            ExprKind::MapComprehension {
                for_parts,
                key,
                value,
            } => {
                self.comprehension(for_parts);
                self.expr(key);
                self.expr(value);
            }
            ExprKind::Meta { meta, expr } => {
                meta.params.iter_mut().for_each(|expr| self.expr(expr));
                self.expr(expr);
            }
            ExprKind::CompilerSpecific { code, args, .. } => {
                self.expr(code);
                args.iter_mut().for_each(|expr| self.expr(expr));
            }
            _ => {}
        }
        self.shadowed = shadowed;
    }

    fn comprehension(&mut self, parts: &mut [ComprehensionFor]) {
        for part in parts {
            self.expr(&mut part.iter);
            self.shadowed.insert(part.var.clone());
            self.shadowed.extend(part.key_var.iter().cloned());
        }
    }
}
