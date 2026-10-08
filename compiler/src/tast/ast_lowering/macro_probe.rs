//! Access rules for expressions typed by a live macro probe.

use super::{AstLowering, LoweringError, LoweringResult};
use crate::tast::{
    SymbolId, SymbolKind, TypeId, Visibility,
    core::TypeKind,
    node::{BinaryOperator, TypedExpression, TypedExpressionKind},
};
use parser::{Expr, ExprKind};

impl AstLowering<'_> {
    pub(crate) fn validate_macro_probe_expression(
        &self,
        expression: &TypedExpression,
    ) -> LoweringResult<()> {
        let error = |message| LoweringError::SemanticError {
            message,
            location: expression.source_location,
        };
        match &expression.kind {
            TypedExpressionKind::VarDeclarationExpr { var_type, .. }
            | TypedExpressionKind::FinalDeclarationExpr { var_type, .. }
                if *var_type == self.context.type_table.borrow().void_type() =>
            {
                return Err(error("Variables of type Void are not allowed".to_string()));
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                if self.macro_probe_is_any(object.expr_type) {
                    let field = self
                        .context
                        .symbol_table
                        .get_symbol(*field_symbol)
                        .and_then(|symbol| self.context.string_interner.get(symbol.name))
                        .unwrap_or("");
                    return Err(error(format!("Any has no field {field}")));
                }
                self.macro_probe_member_access(*field_symbol, expression)?;
            }
            TypedExpressionKind::StaticFieldAccess { field_symbol, .. } => {
                self.macro_probe_member_access(*field_symbol, expression)?;
            }
            TypedExpressionKind::Variable { symbol_id } => {
                self.macro_probe_member_access(*symbol_id, expression)?;
            }
            TypedExpressionKind::MethodCall { method_symbol, .. }
            | TypedExpressionKind::MethodReference { method_symbol, .. }
            | TypedExpressionKind::StaticMethodCall { method_symbol, .. } => {
                self.macro_probe_member_access(*method_symbol, expression)?;
            }
            TypedExpressionKind::ArrayAccess { array, .. }
                if self.macro_probe_is_any(array.expr_type) =>
            {
                return Err(error("Array access is not allowed on Any".to_string()));
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
            } if matches!(
                operator,
                BinaryOperator::Lt | BinaryOperator::Le | BinaryOperator::Gt | BinaryOperator::Ge
            ) && (self.macro_probe_is_any(left.expr_type)
                || self.macro_probe_is_any(right.expr_type)) =>
            {
                let left = super::macro_defer::render_type(
                    left.expr_type,
                    self.context.type_table,
                    self.context.symbol_table,
                    self.context.string_interner,
                    0,
                );
                let right = super::macro_defer::render_type(
                    right.expr_type,
                    self.context.type_table,
                    self.context.symbol_table,
                    self.context.string_interner,
                    0,
                );
                return Err(error(format!("Cannot compare {left} and {right}")));
            }
            _ => {}
        }
        Ok(())
    }

    fn macro_probe_is_any(&self, id: TypeId) -> bool {
        let table = self.context.type_table.borrow();
        let id = Self::resolve_alias_chain(&table, id);
        let symbol = match table.get(id).map(|ty| &ty.kind) {
            Some(TypeKind::Abstract { symbol_id, .. } | TypeKind::Class { symbol_id, .. }) => {
                *symbol_id
            }
            _ => return false,
        };
        self.context
            .symbol_table
            .get_symbol(symbol)
            .is_some_and(|symbol| {
                self.context
                    .string_interner
                    .get(symbol.qualified_name.unwrap_or(symbol.name))
                    == Some("Any")
            })
    }

    fn macro_probe_member_access(
        &self,
        member: SymbolId,
        expression: &TypedExpression,
    ) -> LoweringResult<()> {
        let Some(symbol) = self.context.symbol_table.get_symbol(member) else {
            return Ok(());
        };
        if self.macro_private_access
            || !matches!(
                symbol.visibility,
                Visibility::Private | Visibility::Internal
            )
            || symbol.kind.is_type()
        {
            return Ok(());
        }
        let owner = self
            .class_fields
            .iter()
            .find_map(|(owner, fields)| {
                fields
                    .iter()
                    .any(|(_, field, _)| *field == member)
                    .then_some(*owner)
            })
            .or_else(|| {
                self.context
                    .symbol_table
                    .all_symbols()
                    .find(|owner| {
                        matches!(
                            owner.kind,
                            SymbolKind::Class | SymbolKind::Interface | SymbolKind::Abstract
                        ) && owner.scope_id == symbol.scope_id
                            && owner.scope_id != crate::tast::ScopeId::first()
                            && owner.scope_id.is_valid()
                    })
                    .map(|owner| owner.id)
            });
        let Some(owner) = owner else {
            return Ok(());
        };
        let mut current = self.context.class_context_stack.last().copied();
        for _ in 0..32 {
            let Some(class) = current else {
                break;
            };
            if class == owner {
                return Ok(());
            }
            current = self
                .context
                .symbol_table
                .get_class_super_type(class)
                .and_then(|ty| self.resolve_type_to_class_symbol(ty));
        }
        let name = self.context.string_interner.get(symbol.name).unwrap_or("");
        Err(LoweringError::SemanticError {
            message: format!("Cannot access private field {name}"),
            location: expression.source_location,
        })
    }

    pub(crate) fn macro_probe_argument_error(
        &mut self,
        expression: &Expr,
        error: LoweringError,
    ) -> LoweringError {
        let ExprKind::Call { expr: callee, args } = &expression.kind else {
            return error;
        };
        let LoweringError::SemanticError { message, location } = &error else {
            return error;
        };
        let Some(index) = args.iter().position(|arg| {
            arg.span.start <= location.byte_offset as usize
                && location.byte_offset as usize <= arg.span.end
        }) else {
            return error;
        };
        let ExprKind::Field {
            expr: receiver,
            field,
            ..
        } = &callee.kind
        else {
            return error;
        };
        let ExprKind::Ident(name) = &receiver.kind else {
            return error;
        };
        let name = self.context.intern_string(name);
        let Some(owner) = self.resolve_symbol_in_scope_hierarchy(name) else {
            return error;
        };
        let Some(owner) = self
            .context
            .symbol_table
            .get_symbol(owner)
            .and_then(|symbol| {
                self.context
                    .string_interner
                    .get(symbol.qualified_name.unwrap_or(symbol.name))
            })
        else {
            return error;
        };
        let parameter = self.static_sig_index.as_ref().and_then(|indexer| {
            indexer
                .borrow()
                .parameter_name(owner, field, index)
                .map(str::to_string)
        });
        if let Some(parameter) = parameter {
            return LoweringError::SemanticError {
                message: format!("{message}\n... For function argument '{parameter}'"),
                location: *location,
            };
        }
        error
    }
}
