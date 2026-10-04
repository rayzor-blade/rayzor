//! A function literal stored where a function type with `Dynamic` positions
//! is expected keeps its own representation: its `Int` parameter receives a
//! raw Int, not the box a caller of the wider type passes. Such a literal is
//! wrapped in an adapter of the wider type, whose call into the literal
//! converts each argument and its result the way any call does.

use super::AstLowering;
use crate::tast::TypeId;
use crate::tast::core::TypeKind;
use parser::haxe_ast::*;

const SCALARS: &[&str] = &["Int", "Float", "Bool", "Single", "UInt"];

fn is_scalar_hint(t: &Type) -> bool {
    matches!(t, Type::Path { path, params, .. }
        if path.package.is_empty() && params.is_empty() && SCALARS.contains(&path.name.as_str()))
}

impl<'a> AstLowering<'a> {
    /// The adapter for function literal `value` stored as `target`, when a
    /// `Dynamic` position of `target` meets a scalar one of the literal.
    pub(crate) fn widen_function_literal(&self, value: &Expr, target: TypeId) -> Option<Expr> {
        let mut literal = value;
        while let ExprKind::Paren(inner) = &literal.kind {
            literal = inner;
        }
        let (hints, ret_hint): (Vec<Option<&Type>>, Option<&Type>) = match &literal.kind {
            ExprKind::Function(f) => (
                f.params.iter().map(|p| p.type_hint.as_ref()).collect(),
                f.return_type.as_ref(),
            ),
            ExprKind::Arrow { params, .. } => {
                (params.iter().map(|p| p.type_hint.as_ref()).collect(), None)
            }
            _ => return None,
        };
        let tt = self.context.type_table.borrow();
        let dynamic = tt.dynamic_type();
        let void = tt.void_type();
        let Some(TypeKind::Function {
            params,
            return_type,
            ..
        }) = tt.get(target).map(|t| &t.kind)
        else {
            return None;
        };
        if params.len() != hints.len() {
            return None;
        }
        let widened_param = |i: usize| params[i] == dynamic && hints[i].is_some_and(is_scalar_hint);
        let widened_ret = *return_type == dynamic && ret_hint.is_some_and(is_scalar_hint);
        if !(0..params.len()).any(widened_param) && !widened_ret {
            return None;
        }
        let span = value.span;
        let at = |kind| Expr { kind, span };
        let dynamic_hint = || Type::Path {
            path: TypePath {
                package: vec![],
                name: "Dynamic".to_string(),
                sub: None,
            },
            params: vec![],
            span,
        };
        let names: Vec<String> = (0..params.len()).map(|i| format!("__widen{i}")).collect();
        // A widened argument is first bound at the literal's own type, which
        // converts it as a typed declaration does.
        let mut block = Vec::new();
        let mut args = Vec::new();
        for (i, name) in names.iter().enumerate() {
            if widened_param(i) {
                let local = format!("{name}_typed");
                block.push(BlockElement::Expr(at(ExprKind::Var {
                    name: local.clone(),
                    type_hint: hints[i].cloned(),
                    expr: Some(Box::new(at(ExprKind::Ident(name.clone())))),
                })));
                args.push(at(ExprKind::Ident(local)));
            } else {
                args.push(at(ExprKind::Ident(name.clone())));
            }
        }
        // Called through a local, so the call carries the literal's own type.
        block.insert(
            0,
            BlockElement::Expr(at(ExprKind::Var {
                name: "__widen_fn".to_string(),
                type_hint: None,
                expr: Some(Box::new(literal.clone())),
            })),
        );
        let call = at(ExprKind::Call {
            expr: Box::new(at(ExprKind::Ident("__widen_fn".to_string()))),
            args,
        });
        let returns = *return_type != void;
        block.push(BlockElement::Expr(if returns {
            at(ExprKind::Return(Some(Box::new(call))))
        } else {
            call
        }));
        let body = at(ExprKind::Block(block));
        Some(at(ExprKind::Function(Function {
            name: String::new(),
            type_params: vec![],
            params: names
                .iter()
                .enumerate()
                .map(|(i, name)| FunctionParam {
                    meta: vec![],
                    name: name.clone(),
                    type_hint: if params[i] == dynamic {
                        Some(dynamic_hint())
                    } else {
                        hints[i].cloned()
                    },
                    optional: false,
                    rest: false,
                    default_value: None,
                    span,
                })
                .collect(),
            return_type: (*return_type == dynamic).then(dynamic_hint),
            body: Some(Box::new(body)),
            span,
        })))
    }
}
