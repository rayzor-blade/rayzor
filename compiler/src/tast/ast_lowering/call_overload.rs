//! Picking the member of an `overload` group a call means, from the types of
//! its arguments (see `overloads.rs` for how groups are formed).

use super::overloads::{MODULE_OWNER, OverloadCandidate};
use super::*;
use crate::tast::core::TypeKind;
use parser::{Expr, ExprKind};

impl<'a> AstLowering<'a> {
    /// The callee rewritten to the overload member the arguments select, or
    /// None when the callee names no overload group.
    pub(crate) fn overload_callee(
        &mut self,
        callee: &Expr,
        args: &[Expr],
    ) -> LoweringResult<Option<Expr>> {
        if self.overload_groups.is_empty() {
            return Ok(None);
        }
        let rename = |name: String| -> Expr {
            let kind = match &callee.kind {
                ExprKind::Field {
                    expr, is_optional, ..
                } => ExprKind::Field {
                    expr: expr.clone(),
                    field: name,
                    is_optional: *is_optional,
                },
                _ => ExprKind::Ident(name),
            };
            Expr {
                kind,
                span: callee.span,
            }
        };
        match &callee.kind {
            ExprKind::Ident(name) => {
                let owners = [self.current_class_name(), Some(MODULE_OWNER.to_string())];
                for owner in owners.into_iter().flatten() {
                    if let Some(group) = self.overload_group(&owner, name) {
                        let types = self.overload_arg_types(None, args)?;
                        return Ok(self.pick_overload(&group, &types).map(rename));
                    }
                }
                Ok(None)
            }
            ExprKind::Field {
                expr: receiver,
                field,
                ..
            } => {
                // `Owner.f(..)` on a class that declares the group.
                if let ExprKind::Ident(owner) = &receiver.kind
                    && let Some(group) = self.overload_group(owner, field)
                {
                    let types = self.overload_arg_types(None, args)?;
                    return Ok(self.pick_overload(&group, &types).map(rename));
                }
                if !self.overload_groups.keys().any(|(_, name)| name == field) {
                    return Ok(None);
                }
                let receiver_ty = self.lower_expression(receiver)?.expr_type;
                // `value.f(..)` on an instance of the declaring class.
                if let Some(owner) = self.class_name_of_type(receiver_ty)
                    && let Some(group) = self.overload_group(&owner, field)
                {
                    let types = self.overload_arg_types(None, args)?;
                    return Ok(self.pick_overload(&group, &types).map(rename));
                }
                // `value.f(..)` through `using`: the receiver is the first argument.
                let groups: Vec<Vec<OverloadCandidate>> = self
                    .overload_groups
                    .iter()
                    .filter(|((_, name), members)| {
                        name == field && members.iter().all(|m| m.is_static)
                    })
                    .map(|(_, members)| members.clone())
                    .collect();
                for group in groups {
                    let types = self.overload_arg_types(Some(receiver_ty), args)?;
                    if let Some(name) = self.pick_overload(&group, &types) {
                        return Ok(Some(rename(name)));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn overload_group(&self, owner: &str, name: &str) -> Option<Vec<OverloadCandidate>> {
        self.overload_groups
            .get(&(owner.to_string(), name.to_string()))
            .cloned()
    }

    fn current_class_name(&self) -> Option<String> {
        let class = *self.context.class_context_stack.last()?;
        let symbol = self.context.symbol_table.get_symbol(class)?;
        self.context
            .string_interner
            .get(symbol.name)
            .map(str::to_string)
    }

    fn class_name_of_type(&self, ty: TypeId) -> Option<String> {
        let class = self.resolve_type_to_class_symbol(ty)?;
        let symbol = self.context.symbol_table.get_symbol(class)?;
        self.context
            .string_interner
            .get(symbol.name)
            .map(str::to_string)
    }

    fn overload_arg_types(
        &mut self,
        receiver: Option<TypeId>,
        args: &[Expr],
    ) -> LoweringResult<Vec<TypeId>> {
        let mut types: Vec<TypeId> = receiver.into_iter().collect();
        for arg in args {
            types.push(self.lower_expression(arg)?.expr_type);
        }
        Ok(types)
    }

    /// The member whose parameters accept the arguments best; the first
    /// declared wins a tie. An optional parameter an argument does not fit
    /// is skipped, and a rest parameter takes every remaining argument.
    fn pick_overload(&mut self, group: &[OverloadCandidate], args: &[TypeId]) -> Option<String> {
        let mut best: Option<(u32, String)> = None;
        for candidate in group {
            let Some(score) = self.overload_score(candidate, args) else {
                continue;
            };
            if best.as_ref().is_none_or(|(s, _)| score > *s) {
                best = Some((score, candidate.name.clone()));
            }
        }
        best.map(|(_, name)| name)
    }

    fn overload_score(&mut self, candidate: &OverloadCandidate, args: &[TypeId]) -> Option<u32> {
        let mut score = 0;
        let mut next = 0;
        for param in &candidate.function.params {
            // A rest parameter is written `haxe.Rest<T>`; each argument is a T.
            let hint = match &param.type_hint {
                Some(parser::Type::Path { params, .. }) if param.rest => params.first(),
                hint => hint.as_ref(),
            };
            let param_ty = hint.and_then(|hint| self.lower_type(hint).ok());
            if param.rest {
                for arg in &args[next..] {
                    score += self.overload_fit(*arg, param_ty)?;
                }
                next = args.len();
                break;
            }
            let fit = args
                .get(next)
                .and_then(|arg| self.overload_fit(*arg, param_ty));
            match fit {
                Some(fit) => {
                    score += fit;
                    next += 1;
                }
                None if param.optional || param.default_value.is_some() => {}
                None => return None,
            }
        }
        (next == args.len()).then_some(score)
    }

    /// How well an argument fits a parameter: 3 exact, 2 by a widening
    /// (Int to Float, subclass), 1 through Dynamic or a type parameter.
    fn overload_fit(&self, arg: TypeId, param: Option<TypeId>) -> Option<u32> {
        let Some(param) = param else {
            return Some(1);
        };
        let tt = self.context.type_table.borrow();
        let unwrap = |mut ty: TypeId| {
            for _ in 0..4 {
                match tt.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::Optional { inner_type }) => ty = *inner_type,
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            ty
        };
        let (arg, param) = (unwrap(arg), unwrap(param));
        if arg == param {
            return Some(3);
        }
        let (a, p) = (tt.get(arg).map(|t| &t.kind), tt.get(param).map(|t| &t.kind));
        match (a, p) {
            (_, None | Some(TypeKind::Dynamic | TypeKind::TypeParameter { .. }))
            | (None | Some(TypeKind::Dynamic | TypeKind::Unknown), _) => Some(1),
            (Some(TypeKind::Int), Some(TypeKind::Float)) => Some(2),
            (Some(a), Some(p)) if std::mem::discriminant(a) == std::mem::discriminant(p) => {
                match (a, p) {
                    (TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String, _) => {
                        Some(3)
                    }
                    _ => Some(2),
                }
            }
            _ => None,
        }
    }
}
