//! Module-level fields, as Haxe models them: public statics of a class
//! standing for the module, named after it, or `<Module>_Fields_` when a type
//! already has that name. The class is declared first, so its fields are
//! typed before the module's other declarations use them, as module fields
//! were. They keep their bare names in the module
//! (`AstLowering::qualify_module_static`). Macro functions stay module
//! fields: the macro registry takes them from there.

use super::AstLowering;
use crate::tast::{InternedString, ScopeId, ScopeKind};
use parser::haxe_ast::*;
use std::collections::BTreeMap;

/// Marks a static that was a module-level field.
pub(crate) const MODULE_FIELD_META: &str = "rayzor.moduleField";

/// Moves `file`'s module-level fields, all but macros, into the class
/// standing for its module.
pub fn desugar(file: &mut HaxeFile) {
    if !file.module_fields.iter().any(|f| !is_macro(f)) {
        return;
    }
    let (macros, fields): (Vec<_>, Vec<_>) = std::mem::take(&mut file.module_fields)
        .into_iter()
        .partition(is_macro);
    file.module_fields = macros;
    let span = match (fields.first(), fields.last()) {
        (Some(first), Some(last)) => first.span.merge(last.span),
        _ => Span::default(),
    };
    let module = std::path::Path::new(&file.filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let name = if super::decls::module_type_names(&file.declarations).contains(&module) {
        format!("{module}_Fields_")
    } else {
        module
    };
    let class = ClassDecl {
        meta: Vec::new(),
        access: None,
        modifiers: Vec::new(),
        name,
        type_params: Vec::new(),
        extends: None,
        implements: Vec::new(),
        fields: fields.into_iter().map(static_field).collect(),
        span,
        is_abstract: false,
    };
    file.declarations.insert(0, TypeDeclaration::Class(class));
}

/// The module's fields by bare name: the class holding them, and whether the
/// field is a function.
pub(crate) fn fields_by_name(file: &HaxeFile) -> BTreeMap<String, (String, bool)> {
    let mut fields = BTreeMap::new();
    for decl in &file.declarations {
        let TypeDeclaration::Class(class) = decl else {
            continue;
        };
        for field in &class.fields {
            if field.meta.iter().any(|m| m.name == MODULE_FIELD_META) {
                let function = matches!(field.kind, ClassFieldKind::Function(_));
                fields.insert(
                    field_name(field).to_string(),
                    (class.name.clone(), function),
                );
            }
        }
    }
    fields
}

fn field_name(field: &ClassField) -> &str {
    match &field.kind {
        ClassFieldKind::Var { name, .. }
        | ClassFieldKind::Final { name, .. }
        | ClassFieldKind::Property { name, .. } => name,
        ClassFieldKind::Function(f) => &f.name,
    }
}

fn is_macro(field: &ModuleField) -> bool {
    field.modifiers.contains(&Modifier::Macro)
}

/// Public whatever its access: a private module field is the module's own,
/// and only the module names it bare.
fn static_field(field: ModuleField) -> ClassField {
    let mut modifiers = field.modifiers;
    if !modifiers.contains(&Modifier::Static) {
        modifiers.push(Modifier::Static);
    }
    let kind = match field.kind {
        ModuleFieldKind::Var {
            name,
            type_hint,
            expr,
        } => ClassFieldKind::Var {
            name,
            type_hint,
            expr,
        },
        ModuleFieldKind::Final {
            name,
            type_hint,
            expr,
        } => ClassFieldKind::Final {
            name,
            type_hint,
            expr,
        },
        ModuleFieldKind::Property {
            name,
            type_hint,
            getter,
            setter,
            expr,
        } => ClassFieldKind::Property {
            name,
            type_hint,
            getter,
            setter,
            expr,
        },
        ModuleFieldKind::Function(function) => ClassFieldKind::Function(function),
    };
    let mut meta = field.meta;
    meta.push(Metadata {
        name: MODULE_FIELD_META.to_string(),
        params: Vec::new(),
        span: field.span,
        compile_time: true,
    });
    ClassField {
        meta,
        access: Some(Access::Public),
        modifiers,
        kind,
        span: field.span,
    }
}

impl<'a> AstLowering<'a> {
    /// `expression` with a module field it names bare, as the head of a
    /// call, an assignment, an operand or a field chain, spelled through the
    /// class holding it, so it lowers as that class's static.
    pub(crate) fn qualify_module_static(&mut self, expression: &Expr) -> Option<Expr> {
        if self.module_statics.is_empty() {
            return None;
        }
        let kind = match &expression.kind {
            ExprKind::Ident(name) => {
                return self.module_static_access(name, false, expression.span);
            }
            ExprKind::Field { .. } => return self.qualified_head(expression),
            ExprKind::Call { expr, args } => ExprKind::Call {
                expr: Box::new(self.qualified_target(expr)?),
                args: args.clone(),
            },
            ExprKind::Assign { left, op, right } => ExprKind::Assign {
                left: Box::new(self.qualified_target(left)?),
                op: *op,
                right: right.clone(),
            },
            ExprKind::Unary { op, expr } => ExprKind::Unary {
                op: *op,
                expr: Box::new(self.qualified_target(expr)?),
            },
            _ => return None,
        };
        Some(Expr {
            kind,
            span: expression.span,
        })
    }

    fn qualified_target(&mut self, expr: &Expr) -> Option<Expr> {
        match &expr.kind {
            ExprKind::Ident(name) => self.module_static_access(name, false, expr.span),
            ExprKind::Field { .. } => self.qualified_head(expr),
            _ => None,
        }
    }

    /// A field chain whose head is a module variable. A function head is
    /// left alone: the chain is then more likely a package path.
    fn qualified_head(&mut self, expr: &Expr) -> Option<Expr> {
        match &expr.kind {
            ExprKind::Ident(name) => self.module_static_access(name, true, expr.span),
            ExprKind::Field {
                expr: inner,
                field,
                is_optional,
            } => Some(Expr {
                kind: ExprKind::Field {
                    expr: Box::new(self.qualified_head(inner)?),
                    field: field.clone(),
                    is_optional: *is_optional,
                },
                span: expr.span,
            }),
            _ => None,
        }
    }

    /// `Owner.name` for a bare `name` naming one of this module's fields that
    /// nothing nearer declares.
    fn module_static_access(&mut self, name: &str, head: bool, span: Span) -> Option<Expr> {
        let (owner, function) = self.module_statics.get(name)?.clone();
        if (head && function) || owner == name {
            return None;
        }
        let interned = self.context.intern_string(name);
        if self.declared_nearer(interned) {
            return None;
        }
        Some(Expr {
            kind: ExprKind::Field {
                expr: Box::new(Expr {
                    kind: ExprKind::Ident(owner),
                    span,
                }),
                field: name.to_string(),
                is_optional: false,
            },
            span,
        })
    }

    /// A local, a parameter, or a member of the enclosing class or one of
    /// its ancestors: each shadows a module field.
    fn declared_nearer(&self, name: InternedString) -> bool {
        let mut scope = self.context.current_scope;
        while let Some(current) = self.context.scope_tree.get_scope(scope) {
            if scope == ScopeId::first()
                || matches!(
                    current.kind,
                    ScopeKind::Global | ScopeKind::Package | ScopeKind::Import
                )
            {
                break;
            }
            if self
                .context
                .symbol_table
                .lookup_symbol(scope, name)
                .is_some()
            {
                return true;
            }
            match current.parent_id {
                Some(parent) => scope = parent,
                None => break,
            }
        }
        let Some(&class) = self.context.class_context_stack.last() else {
            return false;
        };
        let mut current = class;
        let mut seen = std::collections::BTreeSet::new();
        while seen.insert(current) {
            let declared = [&self.class_fields, &self.class_methods]
                .iter()
                .any(|members| {
                    members
                        .get(&current)
                        .is_some_and(|m| m.iter().any(|(n, _, _)| *n == name))
                });
            if declared {
                return true;
            }
            let body = self
                .context
                .symbol_table
                .get_symbol(current)
                .map(|s| s.scope_id)
                .filter(|s| *s != ScopeId::first());
            if let Some(body) = body
                && self
                    .context
                    .symbol_table
                    .lookup_symbol(body, name)
                    .is_some()
            {
                return true;
            }
            match self.class_parents.get(&current) {
                Some(&parent) => current = parent,
                None => break,
            }
        }
        false
    }
}
