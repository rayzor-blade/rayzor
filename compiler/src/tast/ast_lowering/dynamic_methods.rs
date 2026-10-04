//! `dynamic` methods, desugared before typing.
//!
//! `dynamic function m(a) body` becomes three members:
//! - `var __dyn_m:F`, the rebinding slot, null until assigned;
//! - `function __orig_m(a) body`, the declared implementation;
//! - `function m(a)`, forwarding to the slot when set, else to `__orig_m`.
//! Every call path (direct, virtual, interface) reaches `m` unchanged. Writing
//! `x.m` writes the slot; reading it takes the slot, else `x.__orig_m`, so a
//! read is a snapshot. An interface's dynamic method gets the slot declared too.

use super::AstLowering;
use crate::tast::SymbolId;
use crate::tast::symbols::{SymbolFlags, SymbolKind};
use parser::haxe_ast::*;

pub(crate) const SLOT_PREFIX: &str = "__dyn_";
pub(crate) const ORIG_PREFIX: &str = "__orig_";

/// The file with its dynamic methods desugared, or None when it has none,
/// and the slots whose method declares no return type and returns no
/// annotated parameter: their type says Dynamic for want of anything better.
pub(crate) fn desugar(file: &HaxeFile) -> Option<(HaxeFile, Vec<String>)> {
    if !file.declarations.iter().any(declares_dynamic) {
        return None;
    }
    let mut file = file.clone();
    let mut unknown = Vec::new();
    for decl in &mut file.declarations {
        desugar_declaration(decl, &mut unknown);
    }
    Some((file, unknown))
}

fn is_dynamic_method(field: &ClassField) -> bool {
    field.modifiers.contains(&Modifier::Dynamic)
        && matches!(&field.kind, ClassFieldKind::Function(f) if f.name != "new"
            && !f.params.iter().any(|p| p.rest))
}

fn declares_dynamic(decl: &TypeDeclaration) -> bool {
    match decl {
        TypeDeclaration::Class(c) => c.fields.iter().any(is_dynamic_method),
        TypeDeclaration::Interface(i) => i.fields.iter().any(is_dynamic_method),
        TypeDeclaration::Conditional(c) => {
            conditional_branches(c).any(|b| b.iter().any(declares_dynamic))
        }
        _ => false,
    }
}

fn conditional_branches<T>(c: &ConditionalCompilation<T>) -> impl Iterator<Item = &Vec<T>> {
    std::iter::once(&c.if_branch.content)
        .chain(c.elseif_branches.iter().map(|b| &b.content))
        .chain(c.else_branch.iter())
}

fn desugar_declaration(decl: &mut TypeDeclaration, unknown: &mut Vec<String>) {
    match decl {
        TypeDeclaration::Class(c) => {
            let class_name = c.name.clone();
            c.fields = desugar_fields(std::mem::take(&mut c.fields), Some(&class_name), unknown);
        }
        TypeDeclaration::Interface(i) => {
            i.fields = desugar_fields(std::mem::take(&mut i.fields), None, unknown);
        }
        TypeDeclaration::Conditional(c) => {
            for branch in std::iter::once(&mut c.if_branch.content)
                .chain(c.elseif_branches.iter_mut().map(|b| &mut b.content))
                .chain(c.else_branch.iter_mut())
            {
                for d in branch {
                    desugar_declaration(d, unknown);
                }
            }
        }
        _ => {}
    }
}

/// `class_name` is None for an interface: only the slot is declared there.
fn desugar_fields(
    fields: Vec<ClassField>,
    class_name: Option<&str>,
    unknown: &mut Vec<String>,
) -> Vec<ClassField> {
    let mut out = Vec::with_capacity(fields.len());
    for field in fields {
        if !is_dynamic_method(&field) {
            out.push(field);
            continue;
        }
        let ClassFieldKind::Function(func) = &field.kind else {
            unreachable!()
        };
        let is_static = field.modifiers.contains(&Modifier::Static);
        let span = field.span;
        if func.return_type.is_none()
            && func.body.as_deref().is_some_and(has_value_return)
            && returned_param_hint(func).is_none()
        {
            unknown.push(format!("{SLOT_PREFIX}{}", func.name));
        }
        out.push(ClassField {
            meta: vec![],
            access: field.access.clone(),
            modifiers: if is_static {
                vec![Modifier::Static]
            } else {
                vec![]
            },
            kind: ClassFieldKind::Var {
                name: format!("{SLOT_PREFIX}{}", func.name),
                type_hint: Some(slot_type(func, span)),
                expr: None,
            },
            span,
        });
        let (Some(class_name), Some(body)) = (class_name, &func.body) else {
            out.push(field);
            continue;
        };
        let mut orig = field.clone();
        orig.modifiers.retain(|m| *m != Modifier::Dynamic);
        if let ClassFieldKind::Function(f) = &mut orig.kind {
            f.name = format!("{ORIG_PREFIX}{}", func.name);
        }
        let returns_value = func
            .return_type
            .as_ref()
            .map_or_else(|| has_value_return(body), |t| !is_void(t));
        let mut forward = field.clone();
        if let ClassFieldKind::Function(f) = &mut forward.kind {
            f.body = Some(Box::new(forwarding_body(
                func,
                is_static,
                returns_value,
                span,
            )));
        }
        out.push(orig);
        out.push(forward);
    }
    out
}

fn is_void(t: &Type) -> bool {
    matches!(t, Type::Path { path, params, .. }
        if path.name == "Void" && path.package.is_empty() && params.is_empty())
}

fn has_value_return(body: &Expr) -> bool {
    let mut found = false;
    super::walk_expr_pruned(body, &mut |e| {
        found |= matches!(e.kind, ExprKind::Return(Some(_)));
        !matches!(e.kind, ExprKind::Function(_) | ExprKind::Arrow { .. })
    });
    found
}

fn path_type(name: &str, span: Span) -> Type {
    Type::Path {
        path: TypePath {
            package: vec![],
            name: name.to_string(),
            sub: None,
        },
        params: vec![],
        span,
    }
}

/// The function type of the method, Dynamic where it is not written down.
fn slot_type(func: &Function, span: Span) -> Type {
    if !func.type_params.is_empty() {
        return path_type("Dynamic", span);
    }
    let params = func
        .params
        .iter()
        .map(|p| {
            p.type_hint
                .clone()
                .unwrap_or_else(|| path_type("Dynamic", span))
        })
        .collect();
    let ret = match &func.return_type {
        Some(t) => t.clone(),
        None if func.body.as_deref().is_some_and(has_value_return) => {
            returned_param_hint(func).unwrap_or_else(|| path_type("Dynamic", span))
        }
        None => path_type("Void", span),
    };
    Type::Function {
        params,
        ret: Box::new(ret),
        span,
    }
}

/// The annotation of the parameter a body returns outright (`return x;`).
fn returned_param_hint(func: &Function) -> Option<Type> {
    let mut body = func.body.as_deref()?;
    if let ExprKind::Block(elements) = &body.kind {
        let [BlockElement::Expr(only)] = elements.as_slice() else {
            return None;
        };
        body = only;
    }
    let ExprKind::Return(Some(value)) = &body.kind else {
        return None;
    };
    let ExprKind::Ident(name) = &value.kind else {
        return None;
    };
    func.params
        .iter()
        .find(|p| &p.name == name)
        .and_then(|p| p.type_hint.clone())
}

fn expr(kind: ExprKind, span: Span) -> Expr {
    Expr { kind, span }
}

/// `owner.member`: `this` for an instance method, the class for a static one.
/// `this.member`, or the bare `member` for a static one.
fn member(is_static: bool, name: String, span: Span) -> Expr {
    if is_static {
        return expr(ExprKind::Ident(name), span);
    }
    let receiver = ExprKind::This;
    expr(
        ExprKind::Field {
            expr: Box::new(expr(receiver, span)),
            field: name,
            is_optional: false,
        },
        span,
    )
}

/// `if (slot != null) return slot(args); return __orig_m(args);`. Without a
/// declared return type the original goes first and the slot's result is a
/// `cast`, so the method keeps the type its body infers rather than Dynamic.
fn forwarding_body(func: &Function, is_static: bool, returns_value: bool, span: Span) -> Expr {
    let args: Vec<Expr> = func
        .params
        .iter()
        .map(|p| expr(ExprKind::Ident(p.name.clone()), span))
        .collect();
    let call = |callee: Expr| {
        expr(
            ExprKind::Call {
                expr: Box::new(callee),
                args: args.clone(),
            },
            span,
        )
    };
    let slot = || member(is_static, format!("{SLOT_PREFIX}{}", func.name), span);
    let exit = |value: Expr| -> Vec<BlockElement> {
        if returns_value {
            vec![BlockElement::Expr(expr(
                ExprKind::Return(Some(Box::new(value))),
                span,
            ))]
        } else {
            vec![
                BlockElement::Expr(value),
                BlockElement::Expr(expr(ExprKind::Return(None), span)),
            ]
        }
    };
    let original = || {
        call(member(
            is_static,
            format!("{ORIG_PREFIX}{}", func.name),
            span,
        ))
    };
    let infer = returns_value && func.return_type.is_none();
    let test = expr(
        ExprKind::Binary {
            left: Box::new(slot()),
            op: if infer { BinaryOp::Eq } else { BinaryOp::NotEq },
            right: Box::new(expr(ExprKind::Null, span)),
        },
        span,
    );
    let (first, last) = if infer {
        let cast = expr(
            ExprKind::Cast {
                expr: Box::new(call(slot())),
                type_hint: None,
            },
            span,
        );
        (original(), cast)
    } else {
        (call(slot()), original())
    };
    let mut block = vec![BlockElement::Expr(expr(
        ExprKind::If {
            cond: Box::new(test),
            then_branch: Box::new(expr(ExprKind::Block(exit(first)), span)),
            else_branch: None,
        },
        span,
    ))];
    block.extend(exit(last));
    expr(ExprKind::Block(block), span)
}

impl<'a> AstLowering<'a> {
    /// Whether some declaration has a slot for a dynamic method named `name`.
    fn dynamic_slot_declared(&self, name: &str) -> bool {
        self.context
            .string_interner
            .get_id(&format!("{SLOT_PREFIX}{name}"))
            .is_some()
    }

    pub(crate) fn is_method_symbol(&self, symbol: SymbolId) -> bool {
        self.context
            .symbol_table
            .get_symbol(symbol)
            .is_some_and(|s| s.kind == SymbolKind::Function)
    }

    /// Whether the slot of the current class's dynamic method `name` has a
    /// known return type: a call may then read the binding before its
    /// arguments, as Haxe evaluates a callee first.
    pub(crate) fn slot_return_known(&self, name: &str) -> bool {
        !self
            .unknown_return_slots
            .contains(&format!("{SLOT_PREFIX}{name}"))
    }

    /// Whether `symbol`, named `name`, is a dynamic method of the class being
    /// lowered. Its own slot says so before its body, and its flags, lower.
    pub(crate) fn is_own_dynamic_method(&self, symbol: SymbolId, name: &str) -> bool {
        self.is_method_symbol(symbol)
            && self
                .context
                .class_context_stack
                .last()
                .is_some_and(|class| self.declares_slot(*class, name))
    }

    /// Whether `class` or an ancestor declares the slot of dynamic method `field`.
    fn declares_slot(&self, class: SymbolId, field: &str) -> bool {
        let Some(slot) = self
            .context
            .string_interner
            .get_id(&format!("{SLOT_PREFIX}{field}"))
        else {
            return false;
        };
        let mut class = class;
        for _ in 0..16 {
            if self.lookup_data_field(class, slot).is_some() {
                return true;
            }
            let Some(parent) = self
                .context
                .symbol_table
                .get_class_hierarchy(class)
                .and_then(|h| h.superclass)
                .and_then(|t| self.resolve_type_to_class_symbol(t))
            else {
                return false;
            };
            class = parent;
        }
        false
    }

    /// Whether `class` or an ancestor keeps the declared body of dynamic `field`.
    pub(crate) fn has_original_body(&self, class: SymbolId, field: &str) -> bool {
        let Some(name) = self
            .context
            .string_interner
            .get_id(&format!("{ORIG_PREFIX}{field}"))
        else {
            return false;
        };
        let mut class = class;
        for _ in 0..16 {
            if self.resolve_class_method_symbol(class, name).is_some() {
                return true;
            }
            let Some(parent) = self
                .context
                .symbol_table
                .get_class_hierarchy(class)
                .and_then(|h| h.superclass)
                .and_then(|t| self.resolve_type_to_class_symbol(t))
            else {
                return false;
            };
            class = parent;
        }
        false
    }

    /// The class a `recv.field` names `field` on: the class itself for a type
    /// name, else the class of the receiver's type.
    fn member_owner(&mut self, recv: &Expr) -> Option<(SymbolId, bool)> {
        if let ExprKind::Ident(n) = &recv.kind {
            let id = self.context.intern_string(n);
            let is_value = self
                .resolve_symbol_in_scope_hierarchy(id)
                .and_then(|s| self.context.symbol_table.get_symbol(s))
                .is_some_and(|s| {
                    matches!(
                        s.kind,
                        SymbolKind::Variable | SymbolKind::Parameter | SymbolKind::Field
                    )
                });
            if !is_value {
                if let Some(class) = self.resolve_class_like_symbol_by_name(id) {
                    return Some((class, true));
                }
            }
        }
        let ty = self.lower_expression(recv).ok()?.expr_type;
        Some((self.resolve_type_to_class_symbol(ty)?, false))
    }

    /// The slot an assignment to a dynamic method writes: `x.m = f` is
    /// `x.__dyn_m = f`, and a bare `m = f` inside the class is too.
    pub(crate) fn dynamic_method_slot(&mut self, target: &Expr) -> Option<Expr> {
        let span = target.span;
        let (owner, name) = match &target.kind {
            ExprKind::Field { expr, field, .. } => {
                if !self.dynamic_slot_declared(field) {
                    return None;
                }
                let (class, _) = self.member_owner(expr)?;
                if !self.declares_slot(class, field) {
                    return None;
                }
                (Some((**expr).clone()), field.clone())
            }
            ExprKind::Ident(name) => {
                if !self.dynamic_slot_declared(name) {
                    return None;
                }
                let id = self.context.intern_string(name);
                let symbol = self.resolve_symbol_in_scope_hierarchy(id)?;
                if !self.is_own_dynamic_method(symbol, name) {
                    return None;
                }
                (self.implicit_owner(symbol, span)?, name.clone())
            }
            _ => return None,
        };
        let slot = format!("{SLOT_PREFIX}{name}");
        Some(match owner {
            Some(owner) => Expr {
                kind: ExprKind::Field {
                    expr: Box::new(owner),
                    field: slot,
                    is_optional: false,
                },
                span,
            },
            None => Expr {
                kind: ExprKind::Ident(slot),
                span,
            },
        })
    }

    /// `this`, or the current class for a static member.
    /// The receiver a bare member name implies: `this`, or none (`Some(None)`)
    /// for a static, which is named bare.
    pub(crate) fn implicit_owner(
        &self,
        member: SymbolId,
        span: parser::Span,
    ) -> Option<Option<Expr>> {
        let class = *self.context.class_context_stack.last()?;
        let is_static = self
            .class_methods
            .get(&class)
            .and_then(|methods| methods.iter().find(|(_, sym, _)| *sym == member))
            .map(|(_, _, is_static)| *is_static)
            .unwrap_or_else(|| {
                self.context
                    .symbol_table
                    .get_symbol(member)
                    .is_some_and(|s| s.flags.contains(SymbolFlags::STATIC))
            });
        Some((!is_static).then_some(Expr {
            kind: ExprKind::This,
            span,
        }))
    }

    /// Reading a dynamic method as a value takes its current binding:
    /// `owner.__dyn_m != null ? owner.__dyn_m : owner.__orig_m`.
    pub(crate) fn dynamic_method_read(
        &self,
        owner: Option<&Expr>,
        name: &str,
        span: parser::Span,
    ) -> Expr {
        let member = |field: String| match owner {
            Some(owner) => Expr {
                kind: ExprKind::Field {
                    expr: Box::new(owner.clone()),
                    field,
                    is_optional: false,
                },
                span,
            },
            None => Expr {
                kind: ExprKind::Ident(field),
                span,
            },
        };
        let slot = format!("{SLOT_PREFIX}{name}");
        Expr {
            kind: ExprKind::Ternary {
                cond: Box::new(Expr {
                    kind: ExprKind::Binary {
                        left: Box::new(member(slot.clone())),
                        op: parser::BinaryOp::NotEq,
                        right: Box::new(Expr {
                            kind: ExprKind::Null,
                            span,
                        }),
                    },
                    span,
                }),
                then_expr: Box::new(member(slot)),
                else_expr: Box::new(member(format!("{ORIG_PREFIX}{name}"))),
            },
            span,
        }
    }
}
