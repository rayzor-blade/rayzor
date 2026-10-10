//! Picking the member of an `overload` group a call means, from the types of
//! its arguments (see `overloads.rs` for how groups are formed). The rules
//! follow Haxe's `Overloads.Resolution`: the compatible members with the
//! fewest optional parameters are rated argument by argument, and a member
//! must rate at least as well as every other on each argument to win.

use super::overloads::{MODULE_OWNER, OverloadCandidate, mentions};
use super::*;
use crate::tast::SymbolId;
use crate::tast::core::TypeKind;
use parser::{Expr, ExprKind};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// The rate of an argument passed to a Dynamic parameter: worse than any
/// conversion.
const WORST: u32 = u32::MAX;

/// What overload rating needs to know of a type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Dynamic,
    /// An anonymous structure: rated as Dynamic, but only a structure or
    /// Dynamic accepts it.
    Anon,
    /// `Null<T>` of a basic type or an enum; nullable types are themselves.
    Null(TypeId),
    Int,
    Float,
    Bool,
    Str,
    Class(SymbolId),
    Interface(SymbolId),
    Enum(SymbolId),
    Abstract(SymbolId),
    Function(usize),
    Array,
    Map,
    /// A `Class<T>`: a class used as a value.
    ClassValue,
}

enum OverloadPick {
    One(String),
    Ambiguous,
    NoMatch,
}

struct OverloadFit {
    optionals: usize,
    /// None when an argument cannot be rated against its parameter.
    rates: Option<Vec<(u32, u32)>>,
    is_rest: bool,
}

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
        // A pick that keeps the callee's own name needs no rewrite; returning
        // the same call would lower it again forever.
        let current = match &callee.kind {
            ExprKind::Field { field, .. } => Some(field.clone()),
            ExprKind::Ident(name) => Some(name.clone()),
            _ => None,
        };
        let rename = |name: String| -> Option<Expr> {
            if current.as_deref() == Some(name.as_str()) {
                return None;
            }
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
            Some(Expr {
                kind,
                span: callee.span,
            })
        };
        match &callee.kind {
            ExprKind::Ident(name) => {
                let owners = [self.current_class_name(), Some(MODULE_OWNER.to_string())];
                for owner in owners.into_iter().flatten() {
                    if let Some(group) = self.overload_group(&owner, name) {
                        let types = self.overload_arg_types(None, args)?;
                        let picked = self.pick_overload(&group, &types, None);
                        return self.picked_member(picked, &group, name, callee).map(rename);
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
                    let picked = self.pick_overload(&group, &types, None);
                    return self
                        .picked_member(picked, &group, field, callee)
                        .map(rename);
                }
                if !self
                    .overload_groups
                    .groups
                    .keys()
                    .any(|(_, name)| name == field)
                {
                    return Ok(None);
                }
                let receiver_ty = self.lower_expression(receiver)?.expr_type;
                // `value.f(..)` on an instance of a type that has the group.
                if let Some(owner) = self.class_name_of_type(receiver_ty)
                    && let Some(group) = self.overload_group(&owner, field)
                {
                    let types = self.overload_arg_types(None, args)?;
                    let picked = self.pick_overload(&group, &types, Some(receiver_ty));
                    return self
                        .picked_member(picked, &group, field, callee)
                        .map(rename);
                }
                // `value.f(..)` through `using`: the receiver is the first argument.
                let groups: Vec<Vec<OverloadCandidate>> = self
                    .overload_groups
                    .groups
                    .iter()
                    .filter(|((_, name), members)| {
                        name == field && members.iter().all(|m| m.is_static)
                    })
                    .map(|(_, members)| members.clone())
                    .collect();
                for group in groups {
                    let types = self.overload_arg_types(Some(receiver_ty), args)?;
                    match self.pick_overload(&group, &types, None) {
                        OverloadPick::NoMatch => {}
                        picked => {
                            return self
                                .picked_member(picked, &group, field, callee)
                                .map(rename);
                        }
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// `new C(args)` for a class whose constructors are overloaded: the
    /// instance from its constructor without parameters, then the selected
    /// constructor body run on it.
    pub(crate) fn overloaded_construction(
        &mut self,
        type_path: &parser::TypePath,
        params: &[parser::Type],
        args: &[Expr],
        span: parser::Span,
    ) -> LoweringResult<Option<Expr>> {
        if args.is_empty() || self.overload_groups.is_empty() {
            return Ok(None);
        }
        let Some(group) = self.overload_group(&type_path.name, "new") else {
            return Ok(None);
        };
        let at = |kind: ExprKind| Expr { kind, span };
        let types = self.overload_arg_types(None, args)?;
        let picked = self.pick_overload(&group, &types, None);
        let member = self.picked_member(
            picked,
            &group,
            "new",
            &at(ExprKind::Ident(type_path.name.clone())),
        )?;
        let instance = format!("__ovl_new{}", span.start);
        let block = vec![
            parser::BlockElement::Expr(at(ExprKind::Var {
                name: instance.clone(),
                type_hint: None,
                expr: Some(Box::new(at(ExprKind::New {
                    type_path: type_path.clone(),
                    params: params.to_vec(),
                    args: Vec::new(),
                }))),
            })),
            parser::BlockElement::Expr(at(ExprKind::Call {
                expr: Box::new(at(ExprKind::Field {
                    expr: Box::new(at(ExprKind::Ident(instance.clone()))),
                    field: member,
                    is_optional: false,
                })),
                args: args.to_vec(),
            })),
            parser::BlockElement::Expr(at(ExprKind::Ident(instance))),
        ];
        Ok(Some(at(ExprKind::Block(block))))
    }

    /// The selected member's name; Haxe rejects a call no member accepts and
    /// one two members accept equally well.
    fn picked_member(
        &self,
        picked: OverloadPick,
        group: &[OverloadCandidate],
        name: &str,
        callee: &Expr,
    ) -> LoweringResult<String> {
        let message = match picked {
            OverloadPick::One(member) => return Ok(member),
            // Argument types here can be less precise than Haxe's (a lost
            // type argument), so only a macro type probe treats a failed
            // pick as the error Haxe reports; elsewhere the first member
            // stands, as it did before overloads were rated.
            _ if self.macro_probe_depth == 0 && !group.is_empty() => {
                return Ok(group[0].name.clone());
            }
            OverloadPick::Ambiguous => format!("Ambiguous overload for {name}"),
            OverloadPick::NoMatch => format!("No overload of {name} accepts these arguments"),
        };
        Err(LoweringError::SemanticError {
            message,
            location: self.context.span_to_location(&callee.span),
        })
    }

    fn overload_group(&self, owner: &str, name: &str) -> Option<Vec<OverloadCandidate>> {
        self.overload_groups
            .groups
            .get(&(owner.to_string(), name.to_string()))
            .cloned()
    }

    fn current_class_name(&self) -> Option<String> {
        let class = *self.context.class_context_stack.last()?;
        self.symbol_name(class)
    }

    fn class_name_of_type(&self, ty: TypeId) -> Option<String> {
        let class = self.resolve_type_to_class_symbol(ty)?;
        self.symbol_name(class)
    }

    fn symbol_name(&self, symbol: SymbolId) -> Option<String> {
        let symbol = self.context.symbol_table.get_symbol(symbol)?;
        self.context
            .string_interner
            .get(symbol.name)
            .map(str::to_string)
    }

    /// Argument types, None for a `null` literal (it takes the type of
    /// whichever parameter it is passed to).
    fn overload_arg_types(
        &mut self,
        receiver: Option<TypeId>,
        args: &[Expr],
    ) -> LoweringResult<Vec<Option<TypeId>>> {
        let mut types: Vec<Option<TypeId>> = receiver.into_iter().map(Some).collect();
        for arg in args {
            if matches!(arg.kind, ExprKind::Null) {
                types.push(None);
            } else {
                let typed = self.lower_expression(arg)?;
                // A class used as a value is a `Class<C>`, not a C.
                let ty = match self.class_value_owner(&typed) {
                    Some(_) => self.class_value_type().unwrap_or(typed.expr_type),
                    None => typed.expr_type,
                };
                types.push(Some(ty));
            }
        }
        Ok(types)
    }

    /// The type of the `Class` abstract, which rates class values.
    fn class_value_type(&mut self) -> Option<TypeId> {
        let name = self.context.intern_string("Class");
        let symbol = self.resolve_symbol_in_scope_hierarchy(name)?;
        let ty = self.context.symbol_table.get_symbol(symbol)?.type_id;
        (self.overload_shape(ty) == Shape::ClassValue).then_some(ty)
    }

    /// The member the arguments select: among the compatible members with
    /// the fewest optional parameters, the one rated best on every argument,
    /// a member without a rest parameter winning a tie.
    fn pick_overload(
        &mut self,
        group: &[OverloadCandidate],
        args: &[Option<TypeId>],
        receiver: Option<TypeId>,
    ) -> OverloadPick {
        let mut fits: Vec<(String, OverloadFit)> = Vec::new();
        for candidate in group {
            if let Some(fit) = self.candidate_fit(candidate, args, receiver) {
                fits.push((candidate.name.clone(), fit));
            }
        }
        let Some(fewest) = fits.iter().map(|(_, fit)| fit.optionals).min() else {
            return OverloadPick::NoMatch;
        };
        fits.retain(|(_, fit)| fit.optionals == fewest);
        if let [(name, _)] = fits.as_slice() {
            return OverloadPick::One(name.clone());
        }
        let mut best: Option<(&[(u32, u32)], bool, Vec<&str>)> = None;
        for (name, fit) in &fits {
            let Some(rates) = fit.rates.as_deref() else {
                continue;
            };
            best = Some(match best.take() {
                None => (rates, fit.is_rest, vec![name.as_str()]),
                Some((best_rates, best_rest, mut names)) => {
                    if is_best(rates, best_rates) {
                        (rates, fit.is_rest, vec![name.as_str()])
                    } else if is_best(best_rates, rates) || (fit.is_rest && !best_rest) {
                        (best_rates, best_rest, names)
                    } else if !fit.is_rest && best_rest {
                        (rates, fit.is_rest, vec![name.as_str()])
                    } else {
                        names.push(name.as_str());
                        (best_rates, best_rest, names)
                    }
                }
            });
        }
        match best {
            Some((_, _, names)) if names.len() == 1 => OverloadPick::One(names[0].to_string()),
            Some(_) => OverloadPick::Ambiguous,
            None => OverloadPick::NoMatch,
        }
    }

    /// Whether `candidate` accepts the arguments, and how each rates. An
    /// optional parameter an argument does not fit is skipped, and is not
    /// rated; a rest parameter takes every remaining argument.
    fn candidate_fit(
        &mut self,
        candidate: &OverloadCandidate,
        args: &[Option<TypeId>],
        receiver: Option<TypeId>,
    ) -> Option<OverloadFit> {
        let bindings = receiver
            .map(|ty| self.receiver_bindings(ty, &candidate.owner_params))
            .unwrap_or_default();
        let generic: Vec<&str> = candidate
            .owner_params
            .iter()
            .map(String::as_str)
            .chain(
                candidate
                    .function
                    .type_params
                    .iter()
                    .map(|p| p.name.as_str()),
            )
            .collect();
        let mut fit = OverloadFit {
            optionals: 0,
            rates: Some(Vec::new()),
            is_rest: false,
        };
        let mut next = 0;
        for param in &candidate.function.params {
            // A rest parameter is written `haxe.Rest<T>`; each argument is a T.
            let hint = match &param.type_hint {
                Some(parser::Type::Path { params, .. }) if param.rest => params.first(),
                hint => hint.as_ref(),
            };
            let param_ty =
                hint.and_then(|hint| self.candidate_param_type(hint, &bindings, &generic));
            if param.rest {
                fit.is_rest = true;
                for arg in &args[next.min(args.len())..] {
                    if !self.overload_accepts(*arg, param_ty) {
                        return None;
                    }
                    add_rate(&mut fit.rates, self.overload_rate(param_ty, *arg));
                }
                next = args.len();
                break;
            }
            let optional = param.optional || param.default_value.is_some();
            if optional {
                fit.optionals += 1;
            }
            match args.get(next) {
                Some(arg) if self.overload_accepts(*arg, param_ty) => {
                    if !optional {
                        add_rate(&mut fit.rates, self.overload_rate(param_ty, *arg));
                    }
                    next += 1;
                }
                _ if optional => {}
                _ => return None,
            }
        }
        (next == args.len()).then_some(fit)
    }

    /// A parameter's type at this call: the owner's type parameters bound
    /// from the receiver. None when it names a type parameter left open.
    fn candidate_param_type(
        &mut self,
        hint: &parser::Type,
        bindings: &BTreeMap<String, TypeId>,
        generic: &[&str],
    ) -> Option<TypeId> {
        if let parser::Type::Path { path, params, .. } = hint
            && path.package.is_empty()
            && path.sub.is_none()
            && params.is_empty()
            && let Some(bound) = bindings.get(&path.name)
        {
            return Some(*bound);
        }
        if mentions(hint, generic) {
            return None;
        }
        self.lower_type(hint).ok()
    }

    /// The receiver's type arguments, keyed by the owner's type parameters.
    fn receiver_bindings(&self, receiver: TypeId, params: &[String]) -> BTreeMap<String, TypeId> {
        let tt = self.context.type_table.borrow();
        let mut ty = receiver;
        let mut args = Vec::new();
        for _ in 0..8 {
            match tt.get(ty).map(|t| &t.kind) {
                Some(
                    TypeKind::Class { type_args, .. }
                    | TypeKind::Interface { type_args, .. }
                    | TypeKind::GenericInstance { type_args, .. },
                ) => {
                    args = type_args.clone();
                    break;
                }
                Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                Some(TypeKind::Optional { inner_type }) => ty = *inner_type,
                _ => break,
            }
        }
        params
            .iter()
            .cloned()
            .zip(args)
            .filter(|(_, arg)| {
                !matches!(
                    tt.get(*arg).map(|t| &t.kind),
                    None | Some(TypeKind::Unknown | TypeKind::TypeParameter { .. })
                )
            })
            .collect()
    }

    fn overload_shape(&self, ty: TypeId) -> Shape {
        let tt = self.context.type_table.borrow();
        let mut ty = ty;
        for _ in 0..16 {
            let Some(info) = tt.get(ty) else {
                return Shape::Dynamic;
            };
            let named_class = |symbol: SymbolId| {
                self.context
                    .symbol_table
                    .get_symbol(symbol)
                    .and_then(|s| self.context.string_interner.get(s.name))
                    == Some("Class")
            };
            match &info.kind {
                TypeKind::Abstract { symbol_id, .. } | TypeKind::Class { symbol_id, .. }
                    if named_class(*symbol_id) =>
                {
                    return Shape::ClassValue;
                }
                TypeKind::Placeholder { name }
                    if self.context.string_interner.get(*name) == Some("Class") =>
                {
                    return Shape::ClassValue;
                }
                TypeKind::TypeAlias { target_type, .. } => ty = *target_type,
                TypeKind::GenericInstance { base_type, .. } => ty = *base_type,
                TypeKind::Optional { inner_type } => {
                    let inner = *inner_type;
                    return match self.overload_shape(inner) {
                        Shape::Int | Shape::Float | Shape::Bool | Shape::Enum(_) => {
                            Shape::Null(inner)
                        }
                        shape => shape,
                    };
                }
                TypeKind::Int => return Shape::Int,
                TypeKind::Float => return Shape::Float,
                TypeKind::Bool => return Shape::Bool,
                TypeKind::String => return Shape::Str,
                // `{}` as an argument lowers as an empty block.
                TypeKind::Void | TypeKind::Anonymous { .. } => return Shape::Anon,
                TypeKind::Class { symbol_id, .. } => return Shape::Class(*symbol_id),
                TypeKind::Interface { symbol_id, .. } => return Shape::Interface(*symbol_id),
                TypeKind::Enum { symbol_id, .. } => return Shape::Enum(*symbol_id),
                TypeKind::Abstract { symbol_id, .. } => return Shape::Abstract(*symbol_id),
                TypeKind::Function { params, .. } => return Shape::Function(params.len()),
                TypeKind::Array { .. } => return Shape::Array,
                TypeKind::Map { .. } => return Shape::Map,
                TypeKind::Placeholder { .. } => {
                    return match self.resolve_type_to_class_symbol(ty) {
                        Some(symbol) if self.context.symbol_table.is_interface(symbol) => {
                            Shape::Interface(symbol)
                        }
                        Some(symbol) => Shape::Class(symbol),
                        None => Shape::Dynamic,
                    };
                }
                _ => return Shape::Dynamic,
            }
        }
        Shape::Dynamic
    }

    /// Whether an argument unifies with a parameter. A `null` argument and
    /// an open parameter accept anything.
    fn overload_accepts(&self, arg: Option<TypeId>, param: Option<TypeId>) -> bool {
        let (Some(arg), Some(param)) = (arg, param) else {
            return true;
        };
        self.shape_accepts(self.overload_shape(arg), self.overload_shape(param))
    }

    fn shape_accepts(&self, arg: Shape, param: Shape) -> bool {
        match (arg, param) {
            (Shape::Dynamic, _) | (_, Shape::Dynamic) => true,
            (Shape::Null(inner), _) => self.shape_accepts(self.overload_shape(inner), param),
            (_, Shape::Null(inner)) => self.shape_accepts(arg, self.overload_shape(inner)),
            (Shape::Anon, param) => param == Shape::Anon,
            (Shape::Int, Shape::Int | Shape::Float)
            | (Shape::Float, Shape::Float)
            | (Shape::Bool, Shape::Bool)
            | (Shape::Str, Shape::Str)
            | (Shape::Array, Shape::Array)
            | (Shape::Map, Shape::Map)
            | (Shape::ClassValue, Shape::ClassValue) => true,
            (
                Shape::Class(from) | Shape::Interface(from),
                Shape::Class(to) | Shape::Interface(to),
            ) => self.type_distance(from, to).is_some(),
            (Shape::Enum(a), Shape::Enum(b)) => a == b,
            // Abstracts convert through casts this check does not model.
            (Shape::Abstract(_), _) | (_, Shape::Abstract(_)) => true,
            (Shape::Function(a), Shape::Function(b)) => a == b,
            _ => false,
        }
    }

    /// How far an argument is from a parameter; lower is better.
    fn overload_rate(&self, param: Option<TypeId>, arg: Option<TypeId>) -> Option<(u32, u32)> {
        match (param, arg) {
            (None, None) => Some((0, 0)),
            (None, Some(arg)) => match self.overload_shape(arg) {
                Shape::Dynamic | Shape::Anon => Some((0, 0)),
                _ => Some((WORST, 0)),
            },
            // `null` is typed `Null<T>` of the parameter.
            (Some(param), None) => match self.overload_shape(param) {
                Shape::Int | Shape::Float | Shape::Bool | Shape::Enum(_) => Some((1, 0)),
                _ => Some((0, 0)),
            },
            (Some(param), Some(arg)) => {
                self.rate_conv(0, self.overload_shape(param), self.overload_shape(arg))
            }
        }
    }

    /// Haxe's `rate_conv`: conversions needed to pass `arg` as `param`.
    fn rate_conv(&self, acc: u32, param: Shape, arg: Shape) -> Option<(u32, u32)> {
        let dynamic = |shape: Shape| matches!(shape, Shape::Dynamic | Shape::Anon);
        match (param, arg) {
            (Shape::Interface(to), Shape::Class(from) | Shape::Interface(from))
            | (Shape::Class(to), Shape::Class(from)) => self
                .type_distance(from, to)
                .map(|d| (acc.saturating_add(d), 0)),
            (Shape::Enum(to), Shape::Enum(from)) => (to == from).then_some((acc, 0)),
            (p, a) if dynamic(p) && dynamic(a) => Some((acc, 0)),
            (p, _) if dynamic(p) => Some((WORST, 0)),
            // A Dynamic unboxed to a basic type pays more than one boxed.
            (Shape::Int | Shape::Float | Shape::Bool, a) if dynamic(a) => {
                Some((acc.saturating_add(2), 0))
            }
            (_, a) if dynamic(a) => Some((acc.saturating_add(1), 0)),
            (Shape::Null(p), Shape::Null(a)) => {
                self.rate_conv(acc, self.overload_shape(p), self.overload_shape(a))
            }
            (Shape::Null(p), a) => self.rate_conv(acc.saturating_add(1), self.overload_shape(p), a),
            (p, Shape::Null(a)) => self.rate_conv(acc.saturating_add(1), p, self.overload_shape(a)),
            (Shape::Float, Shape::Int) => Some((acc.saturating_add(1), 0)),
            (Shape::Int, Shape::Int)
            | (Shape::Float, Shape::Float)
            | (Shape::Bool, Shape::Bool)
            | (Shape::Str, Shape::Str)
            | (Shape::Array, Shape::Array)
            | (Shape::Map, Shape::Map)
            | (Shape::ClassValue, Shape::ClassValue)
            | (Shape::Function(_), Shape::Function(_)) => Some((acc, 0)),
            (Shape::Abstract(p), Shape::Abstract(a)) if p == a => Some((acc, 0)),
            (Shape::Abstract(_), _) | (_, Shape::Abstract(_)) => Some((acc.saturating_add(1), 0)),
            _ => None,
        }
    }

    /// Steps from a class or interface up to `to` through the types it
    /// extends or implements, or None when it is not one of them.
    fn type_distance(&self, from: SymbolId, to: SymbolId) -> Option<u32> {
        if from == to {
            return Some(0);
        }
        let target = self.symbol_name(to)?;
        let mut queue = VecDeque::from([(Some(from), self.symbol_name(from)?, 0)]);
        let mut seen = BTreeSet::new();
        while let Some((symbol, name, depth)) = queue.pop_front() {
            if name == target {
                return Some(depth);
            }
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(parents) = self.overload_groups.parents.get(&name) {
                for parent in parents {
                    queue.push_back((None, parent.clone(), depth + 1));
                }
            } else if let Some(symbol) = symbol
                && let Some(parent) = self.context.symbol_table.get_class_super_type(symbol)
                && let Some(parent) = self.resolve_type_to_class_symbol(parent)
                && let Some(parent_name) = self.symbol_name(parent)
            {
                queue.push_back((Some(parent), parent_name, depth + 1));
            }
        }
        None
    }
}

/// Haxe's `is_best`: no worse on any argument and better on one.
fn is_best(a: &[(u32, u32)], b: &[(u32, u32)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| x <= y)
        && a.iter().zip(b).any(|(x, y)| x < y)
}

fn add_rate(rates: &mut Option<Vec<(u32, u32)>>, rate: Option<(u32, u32)>) {
    match rate {
        Some(rate) => {
            if let Some(rates) = rates {
                rates.push(rate);
            }
        }
        None => *rates = None,
    }
}
