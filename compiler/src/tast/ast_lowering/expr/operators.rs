//! Binary and unary operator lowering.

use super::*;
use crate::tast::node::HasSourceLocation;
use crate::tast::{core::*, node::MemoryEffects, node::*, type_resolution, *};
use parser::{
    AbstractDecl, BinaryOp, BlockElement, ClassDecl, ClassField, ClassFieldKind, EnumConstructor,
    EnumDecl, Expr, ExprKind, Function, FunctionParam, HaxeFile, Import, InterfaceDecl, Metadata,
    Modifier, ModuleField, Package, Type, TypeDeclaration, TypeParam, TypedefDecl, UnaryOp, Using,
};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use tracing::warn;

impl<'a> AstLowering<'a> {
    /// Lower a binary operator
    pub(crate) fn lower_binary_operator(
        &mut self,
        operator: &BinaryOp,
    ) -> LoweringResult<BinaryOperator> {
        match operator {
            BinaryOp::Add => Ok(BinaryOperator::Add),
            BinaryOp::Sub => Ok(BinaryOperator::Sub),
            BinaryOp::Mul => Ok(BinaryOperator::Mul),
            BinaryOp::Div => Ok(BinaryOperator::Div),
            BinaryOp::Mod => Ok(BinaryOperator::Mod),
            BinaryOp::Eq => Ok(BinaryOperator::Eq),
            BinaryOp::NotEq => Ok(BinaryOperator::Ne),
            BinaryOp::Lt => Ok(BinaryOperator::Lt),
            BinaryOp::Le => Ok(BinaryOperator::Le),
            BinaryOp::Gt => Ok(BinaryOperator::Gt),
            BinaryOp::Ge => Ok(BinaryOperator::Ge),
            BinaryOp::And => Ok(BinaryOperator::And),
            BinaryOp::Or => Ok(BinaryOperator::Or),
            BinaryOp::BitAnd => Ok(BinaryOperator::BitAnd),
            BinaryOp::BitOr => Ok(BinaryOperator::BitOr),
            BinaryOp::BitXor => Ok(BinaryOperator::BitXor),
            BinaryOp::Shl => Ok(BinaryOperator::Shl),
            BinaryOp::Shr => Ok(BinaryOperator::Shr),
            BinaryOp::Ushr => Ok(BinaryOperator::Ushr),
            BinaryOp::Range => Ok(BinaryOperator::Range),
            BinaryOp::Arrow => Ok(BinaryOperator::Arrow),
            BinaryOp::Is => {
                // 'is' operator needs runtime type checking support
                // For now, lower as a comparison (downstream passes handle it)
                Ok(BinaryOperator::Eq)
            }
            BinaryOp::NullCoal => Ok(BinaryOperator::NullCoal),
            BinaryOp::In => Ok(BinaryOperator::In),
        }
    }

    /// Lower a unary operator
    pub(crate) fn lower_unary_operator(
        &mut self,
        operator: &UnaryOp,
    ) -> LoweringResult<UnaryOperator> {
        match operator {
            UnaryOp::Neg => Ok(UnaryOperator::Neg),
            UnaryOp::Not => Ok(UnaryOperator::Not),
            UnaryOp::BitNot => Ok(UnaryOperator::BitNot),
            UnaryOp::PreIncr => Ok(UnaryOperator::PreInc),
            UnaryOp::PostIncr => Ok(UnaryOperator::PostInc),
            UnaryOp::PreDecr => Ok(UnaryOperator::PreDec),
            UnaryOp::PostDecr => Ok(UnaryOperator::PostDec),
            // Lowered as a call by `lower_postfix_not`; no built-in operator.
            UnaryOp::PostNot => Err(LoweringError::SemanticError {
                message: "postfix `!` is only defined by an abstract's @:op(A!)".to_string(),
                location: self.context.create_location(),
            }),
        }
    }

    /// `a!`: the operand abstract's `@:op(A!)` method, called as written
    /// source (`a.method()`, or `A.method(a)` for a static) so it lowers like
    /// any member call. Postfix `!` has no meaning on any other type.
    pub(crate) fn lower_postfix_not(
        &mut self,
        expression: &Expr,
        operand: &Expr,
    ) -> LoweringResult<TypedExpression> {
        let operand_type = self.lower_expression(operand)?.expr_type;
        let owner = {
            let table = self.context.type_table.borrow();
            let mut ty = operand_type;
            for _ in 0..16 {
                match table.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            match table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Abstract { symbol_id, .. }) => Some(*symbol_id),
                _ => None,
            }
        };
        let Some((owner, (method, is_static))) =
            owner.and_then(|owner| Some((owner, self.postfix_not_method(owner)?)))
        else {
            return Err(LoweringError::SemanticError {
                message: format!(
                    "{} has no postfix `!` operator",
                    super::super::macro_defer::render_type(
                        operand_type,
                        self.context.type_table,
                        &*self.context.symbol_table,
                        &*self.context.string_interner,
                        0,
                    )
                ),
                location: self.context.create_location_from_span(expression.span),
            });
        };
        let span = expression.span;
        let mk = |kind: ExprKind| Expr { kind, span };
        let (receiver, args) = if is_static {
            let owner_name = self
                .context
                .symbol_table
                .get_symbol(owner)
                .and_then(|s| self.context.string_interner.get(s.name))
                .unwrap_or_default()
                .to_string();
            (mk(ExprKind::Ident(owner_name)), vec![operand.clone()])
        } else {
            (operand.clone(), Vec::new())
        };
        let call = mk(ExprKind::Call {
            expr: Box::new(mk(ExprKind::Field {
                expr: Box::new(receiver),
                field: method,
                is_optional: false,
            })),
            args,
        });
        self.lower_expression(&call)
    }

    /// Haxe promotes an expected abstract declaring exactly one `@:op` for
    /// `operator` to the operands: `var f:EnumFlags<E> = A | B` reads as
    /// `(A : EnumFlags<E>) | (B : EnumFlags<E>)`, converting through `@:from`.
    /// Operands with an operator of their own (a primitive, an abstract
    /// declaring it) keep it.
    pub(crate) fn promote_operands_to_expected_abstract(
        &mut self,
        operator: BinaryOperator,
        left: TypedExpression,
        right: TypedExpression,
    ) -> (TypedExpression, TypedExpression) {
        let Some(expected) = self.expected_arg_type_stack.last().copied().flatten() else {
            return (left, right);
        };
        let Some(target) = self.abstract_symbol_of(expected) else {
            return (left, right);
        };
        if self.binary_operator_method_count(target, operator) != 1 {
            return (left, right);
        }
        let has_own_operator = |this: &mut Self, ty: TypeId| {
            let builtin = {
                let tt = this.context.type_table.borrow();
                let ty = Self::resolve_alias_chain(&tt, ty);
                !matches!(
                    tt.get(ty).map(|t| &t.kind),
                    Some(
                        TypeKind::Enum { .. }
                            | TypeKind::Class { .. }
                            | TypeKind::Anonymous { .. }
                            | TypeKind::Abstract { .. }
                            | TypeKind::GenericInstance { .. }
                    )
                )
            };
            builtin
                || this
                    .abstract_symbol_of(ty)
                    .is_some_and(|owner| this.binary_operator_method_count(owner, operator) > 0)
        };
        if has_own_operator(self, left.expr_type) || has_own_operator(self, right.expr_type) {
            return (left, right);
        }
        let cast = |operand: TypedExpression| TypedExpression {
            expr_type: expected,
            source_location: operand.source_location,
            usage: operand.usage.clone(),
            lifetime_id: operand.lifetime_id,
            metadata: operand.metadata.clone(),
            kind: TypedExpressionKind::Cast {
                expression: Box::new(operand),
                target_type: expected,
                cast_kind: CastKind::Checked,
            },
        };
        (cast(left), cast(right))
    }

    /// How many `@:op` methods `owner` declares for a binary operator.
    fn binary_operator_method_count(&mut self, owner: SymbolId, operator: BinaryOperator) -> usize {
        let Some(index) = self.static_sig_index.as_ref().cloned() else {
            return 0;
        };
        let Some(symbol) = self.context.symbol_table.get_symbol(owner) else {
            return 0;
        };
        let Some(name) = self
            .context
            .string_interner
            .get(symbol.qualified_name.unwrap_or(symbol.name))
            .map(str::to_string)
        else {
            return 0;
        };
        let methods = index.borrow_mut().binary_operator_methods(&name, operator);
        methods.len()
    }

    /// The non-enum abstract `ty` names, its index name, and the bare name
    /// of its underlying type when that is a basic type.
    fn operand_abstract(&self, ty: TypeId) -> Option<(String, Option<&'static str>)> {
        let (symbol, underlying) = {
            let tt = self.context.type_table.borrow();
            let mut ty = Self::resolve_alias_chain(&tt, ty);
            if let Some(TypeKind::GenericInstance { base_type, .. }) = tt.get(ty).map(|t| &t.kind) {
                ty = *base_type;
            }
            let Some(TypeKind::Abstract {
                symbol_id,
                underlying,
                ..
            }) = tt.get(ty).map(|t| &t.kind)
            else {
                return None;
            };
            let underlying = underlying.and_then(|u| {
                match tt.get(Self::resolve_alias_chain(&tt, u)).map(|t| &t.kind) {
                    Some(TypeKind::Int) => Some("Int"),
                    Some(TypeKind::Float) => Some("Float"),
                    Some(TypeKind::Bool) => Some("Bool"),
                    Some(TypeKind::String) => Some("String"),
                    _ => None,
                }
            });
            (*symbol_id, underlying)
        };
        if self.context.symbol_table.is_enum_abstract(symbol) {
            return None;
        }
        let symbol = self.context.symbol_table.get_symbol(symbol)?;
        let name = self
            .context
            .string_interner
            .get(symbol.qualified_name.unwrap_or(symbol.name))?
            .to_string();
        Some((name, underlying))
    }

    fn operator_signatures(
        &self,
        owner: &str,
        key: &str,
    ) -> Vec<crate::tast::sig_index::StaticMethodSig> {
        self.static_sig_index
            .as_ref()
            .map(|index| index.borrow_mut().operator_signatures(owner, key))
            .unwrap_or_default()
    }

    /// Operators Haxe rejects on abstracts, checked only under a macro probe
    /// (`typeError`): an arithmetic operator neither abstract declares over
    /// a String, and a bodyless `@:op` whose declared return type differs
    /// from the underlying operation's.
    pub(crate) fn probe_abstract_binary_error(
        &mut self,
        operator: BinaryOperator,
        left: &TypedExpression,
        right: &TypedExpression,
    ) -> Option<String> {
        let key = match operator {
            BinaryOperator::Add => "Add",
            BinaryOperator::Sub => "Sub",
            BinaryOperator::Mul => "Mul",
            BinaryOperator::Div => "Div",
            BinaryOperator::Mod => "Mod",
            _ => return None,
        };
        let operands = [left.expr_type, right.expr_type].map(|ty| self.operand_abstract(ty));
        if operands.iter().all(Option::is_none) {
            return None;
        }
        let basic = |ty: TypeId, abstract_info: &Option<(String, Option<&'static str>)>| {
            if let Some((_, underlying)) = abstract_info {
                return *underlying;
            }
            let tt = self.context.type_table.borrow();
            match tt.get(Self::resolve_alias_chain(&tt, ty)).map(|t| &t.kind) {
                Some(TypeKind::Int) => Some("Int"),
                Some(TypeKind::Float) => Some("Float"),
                Some(TypeKind::Bool) => Some("Bool"),
                Some(TypeKind::String) => Some("String"),
                _ => None,
            }
        };
        let kinds = [
            basic(left.expr_type, &operands[0]),
            basic(right.expr_type, &operands[1]),
        ];
        let names: Vec<Option<String>> = operands
            .iter()
            .zip(kinds)
            .map(|(info, kind)| match info {
                Some((name, _)) => Some(name.rsplit('.').next().unwrap_or(name).to_string()),
                None => kind.map(str::to_string),
            })
            .collect();
        let signatures: Vec<_> = operands
            .iter()
            .flatten()
            .flat_map(|(owner, _)| self.operator_signatures(owner, key))
            .collect();
        if signatures.is_empty() {
            let string = kinds.contains(&Some("String"));
            return (key != "Add" && string).then(|| format!("Cannot apply {key} to String"));
        }
        let result = if kinds.contains(&Some("String")) {
            "String"
        } else if kinds.contains(&Some("Float")) {
            "Float"
        } else {
            "Int"
        };
        let bare = |ty: &Option<parser::Type>| match ty {
            Some(parser::Type::Path { path, .. }) if path.package.is_empty() => {
                Some(path.name.clone())
            }
            _ => None,
        };
        let chosen = signatures.iter().find(|sig| {
            sig.param_kinds.len() == 2
                && sig.param_kinds[0].is_some()
                && sig.param_kinds[0] == names[0]
                && sig.param_kinds[1].is_some()
                && sig.param_kinds[1] == names[1]
        })?;
        let declared = bare(&chosen.return_type)?;
        let mismatched = !chosen.has_body
            && matches!(declared.as_str(), "Int" | "Float" | "Bool" | "String")
            && declared != result
            && !(declared == "Float" && result == "Int");
        mismatched.then(|| format!("{result} should be {declared}"))
    }

    /// A unary operator on an abstract that declares none for it: `!` over
    /// a non-Bool, or `++`/`--` in the fixity the abstract does not declare.
    /// Checked only under a macro probe.
    pub(crate) fn probe_abstract_unary_error(
        &mut self,
        op: &UnaryOp,
        operand: &TypedExpression,
    ) -> Option<String> {
        let (owner, underlying) = self.operand_abstract(operand.expr_type)?;
        let declared = |this: &Self, op: &UnaryOp| {
            !this
                .operator_signatures(&owner, &format!("Unary{op:?}"))
                .is_empty()
        };
        if declared(self, op) {
            return None;
        }
        let other_fixity = match op {
            UnaryOp::PostIncr => Some(UnaryOp::PreIncr),
            UnaryOp::PreIncr => Some(UnaryOp::PostIncr),
            UnaryOp::PostDecr => Some(UnaryOp::PreDecr),
            UnaryOp::PreDecr => Some(UnaryOp::PostDecr),
            _ => None,
        };
        let rejected = match op {
            UnaryOp::Not => underlying.is_some_and(|u| u != "Bool"),
            _ => other_fixity.is_some_and(|other| declared(self, &other)),
        };
        rejected.then(|| format!("{owner} has no {op:?} operator"))
    }

    /// The `@:op(A!)` method of an abstract: (name, is_static).
    fn postfix_not_method(&self, owner: SymbolId) -> Option<(String, bool)> {
        if let Some(found) = self.abstract_postfix_not.get(&owner) {
            return Some(found.clone());
        }
        // Declared in another module: known from its indexed declaration.
        let index = self.static_sig_index.as_ref()?.clone();
        let symbol = self.context.symbol_table.get_symbol(owner)?;
        let name = self
            .context
            .string_interner
            .get(symbol.qualified_name.unwrap_or(symbol.name))?
            .to_string();
        let (method, is_static, _) = index
            .borrow_mut()
            .operator_methods(&name, "UnaryPostNot")
            .into_iter()
            .next()?;
        Some((method, is_static))
    }

    // Property access handling removed for simplicity
}
