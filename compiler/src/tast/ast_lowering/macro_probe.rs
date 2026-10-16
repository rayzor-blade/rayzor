//! Access rules for expressions typed by a live macro probe.

use super::{AstLowering, LoweringError, LoweringResult};
use crate::tast::{
    SymbolId, SymbolKind, TypeId, Visibility,
    core::TypeKind,
    node::{BinaryOperator, CastKind, TypedExpression, TypedExpressionKind},
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
        // An exit (`final a = return;`) types as a monomorph in Haxe.
        let exits = |mut init: &TypedExpression| {
            while let TypedExpressionKind::Cast {
                expression,
                cast_kind: CastKind::Implicit,
                ..
            } = &init.kind
            {
                init = &**expression;
            }
            matches!(
                init.kind,
                TypedExpressionKind::Return { .. }
                    | TypedExpressionKind::Throw { .. }
                    | TypedExpressionKind::Break
                    | TypedExpressionKind::Continue
            )
        };
        match &expression.kind {
            TypedExpressionKind::VarDeclarationExpr {
                var_type,
                initializer,
                ..
            }
            | TypedExpressionKind::FinalDeclarationExpr {
                var_type,
                initializer,
                ..
            } if *var_type == self.context.type_table.borrow().void_type()
                && !exits(&**initializer) =>
            {
                return Err(error("Variables of type Void are not allowed".to_string()));
            }
            TypedExpressionKind::VarDeclarationExpr {
                var_type,
                initializer,
                ..
            }
            | TypedExpressionKind::FinalDeclarationExpr {
                var_type,
                initializer,
                ..
            } => {
                if let Some(message) = self.macro_probe_structure_mismatch(initializer, *var_type) {
                    return Err(error(message));
                }
                if let Some(message) = self.macro_probe_abstract_to_mismatch(initializer, *var_type)
                {
                    return Err(error(message));
                }
                let mut init: &TypedExpression = initializer;
                while let TypedExpressionKind::Cast {
                    expression,
                    cast_kind: CastKind::Implicit,
                    ..
                } = &init.kind
                {
                    init = &**expression;
                }
                if !self.macro_probe_abstract_assignable(init.expr_type, *var_type) {
                    let render = |id| {
                        super::macro_defer::render_type(
                            id,
                            self.context.type_table,
                            self.context.symbol_table,
                            self.context.string_interner,
                            0,
                        )
                    };
                    return Err(error(format!(
                        "{} should be {}",
                        render(init.expr_type),
                        render(*var_type)
                    )));
                }
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
                if let Some(message) = self.macro_probe_missing_field(object, *field_symbol) {
                    return Err(error(message));
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
            TypedExpressionKind::Cast {
                expression: literal,
                target_type,
                cast_kind: CastKind::Checked,
            } => {
                if let Some(message) = self.macro_probe_structure_mismatch(literal, *target_type) {
                    return Err(error(message));
                }
            }
            TypedExpressionKind::MapLiteral { entries } => {
                if let Some(message) = self.macro_probe_map_literal(entries) {
                    return Err(error(message));
                }
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                if let Some(message) =
                    self.macro_probe_call_arguments(function.expr_type, arguments)
                {
                    return Err(error(message));
                }
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator: BinaryOperator::Assign,
                right,
            } => {
                if let Some(message) = self.macro_probe_structure_mismatch(right, left.expr_type) {
                    return Err(error(message));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// A read of a field a structure does not declare. A local's macro view
    /// stands in for its runtime type; an empty structure is left alone.
    fn macro_probe_missing_field(
        &self,
        object: &TypedExpression,
        field: SymbolId,
    ) -> Option<String> {
        let mut ty = match &object.kind {
            TypedExpressionKind::Variable { symbol_id } => self
                .macro_source_types
                .get(symbol_id)
                .copied()
                .unwrap_or(object.expr_type),
            _ => object.expr_type,
        };
        let name = self.context.symbol_table.get_symbol(field)?.name;
        let table = self.context.type_table.borrow();
        for _ in 0..8 {
            ty = Self::resolve_alias_chain(&table, ty);
            match table.get(ty).map(|t| &t.kind)? {
                TypeKind::Optional { inner_type } => ty = *inner_type,
                TypeKind::Anonymous { fields } => {
                    if fields.is_empty() || fields.iter().any(|f| f.name == name) {
                        return None;
                    }
                    let shown = super::macro_defer::render_type(
                        ty,
                        self.context.type_table,
                        self.context.symbol_table,
                        self.context.string_interner,
                        0,
                    );
                    let field = self.context.string_interner.get(name)?;
                    return Some(format!("{shown} has no field {field}"));
                }
                _ => return None,
            }
        }
        None
    }

    /// `({..} : {..})`: the literal must supply every required field of the
    /// structure and no other. Haxe lists fields in descending name order and
    /// names the last extra field the literal declares.
    fn macro_probe_structure_mismatch(
        &self,
        mut literal: &TypedExpression,
        target: TypeId,
    ) -> Option<String> {
        while let TypedExpressionKind::Cast {
            expression,
            cast_kind: CastKind::Implicit,
            ..
        } = &literal.kind
        {
            literal = &**expression;
        }
        let TypedExpressionKind::ObjectLiteral { fields } = &literal.kind else {
            return None;
        };
        let expected: Vec<_> = {
            let table = self.context.type_table.borrow();
            match table
                .get(Self::resolve_alias_chain(&table, target))
                .map(|ty| &ty.kind)
            {
                Some(TypeKind::Anonymous { fields }) => fields
                    .iter()
                    .map(|f| (f.name, f.optional, f.type_id))
                    .collect(),
                _ => return None,
            }
        };
        let name = |id| self.context.string_interner.get(id).unwrap_or("");
        for (i, field) in fields.iter().enumerate() {
            if fields[..i].iter().any(|f| f.name == field.name) {
                return Some(format!(
                    "Duplicate field in object declaration : {}",
                    name(field.name)
                ));
            }
        }
        let mut missing: Vec<&str> = expected
            .iter()
            .filter(|(field, optional, _)| !optional && !fields.iter().any(|f| f.name == *field))
            .map(|(field, _, _)| name(*field))
            .collect();
        missing.sort_unstable_by(|a, b| b.cmp(a));
        match missing.as_slice() {
            [] => {}
            [one] => return Some(format!("Object requires field {one}")),
            many => return Some(format!("Object requires fields: {}", many.join(", "))),
        }
        for field in fields {
            let Some((_, _, want)) = expected.iter().find(|(n, _, _)| *n == field.name) else {
                continue;
            };
            if !self.macro_probe_value_fits(field.value.expr_type, *want) {
                let render = |id| {
                    super::macro_defer::render_type(
                        id,
                        self.context.type_table,
                        self.context.symbol_table,
                        self.context.string_interner,
                        0,
                    )
                };
                return Some(format!(
                    "{} should be {}",
                    render(field.value.expr_type),
                    render(*want)
                ));
            }
        }
        let extra = fields
            .iter()
            .rev()
            .find(|f| !expected.iter().any(|(field, _, _)| *field == f.name))?;
        let mut shown: Vec<(&str, String)> = fields
            .iter()
            .map(|f| {
                let ty = super::macro_defer::render_type(
                    f.value.expr_type,
                    self.context.type_table,
                    self.context.symbol_table,
                    self.context.string_interner,
                    0,
                );
                (name(f.name), ty)
            })
            .collect();
        shown.sort_by(|a, b| b.0.cmp(&a.0));
        let shown: Vec<String> = shown.iter().map(|(n, t)| format!("{n} : {t}")).collect();
        Some(format!(
            "{{ {} }} has extra field {}",
            shown.join(", "),
            name(extra.name)
        ))
    }

    /// Arguments matched to a function's parameters as haxe matches them: an
    /// optional parameter an argument does not fit is skipped.
    fn macro_probe_call_arguments(
        &self,
        function: TypeId,
        arguments: &[TypedExpression],
    ) -> Option<String> {
        let params = {
            let table = self.context.type_table.borrow();
            match table
                .get(Self::resolve_alias_chain(&table, function))
                .map(|t| &t.kind)
            {
                Some(TypeKind::Function { params, .. }) => params.clone(),
                _ => return None,
            }
        };
        let optional = |ty: TypeId| {
            let table = self.context.type_table.borrow();
            match table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => Some(*inner_type),
                _ => None,
            }
        };
        let rest = params.iter().any(|&p| {
            let table = self.context.type_table.borrow();
            match table.get(p).map(|t| &t.kind) {
                Some(TypeKind::Class { symbol_id, .. } | TypeKind::Abstract { symbol_id, .. }) => {
                    self.context
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .is_some_and(|s| self.context.string_interner.get(s.name) == Some("Rest"))
                }
                _ => false,
            }
        });
        if rest {
            return None;
        }
        let mut next = params.iter().copied();
        for argument in arguments {
            loop {
                let Some(param) = next.next() else {
                    return Some("Too many arguments".to_string());
                };
                let want = optional(param).unwrap_or(param);
                if self.macro_probe_argument_fits(argument.expr_type, want) {
                    break;
                }
                // A required parameter the argument does not fit is the
                // ordinary type error, reported where it is checked.
                if optional(param).is_none() {
                    return None;
                }
            }
        }
        None
    }

    /// Whether an argument of type `have` can fill a parameter of type `want`,
    /// telling apart only basic types and enums.
    fn macro_probe_argument_fits(&self, have: TypeId, want: TypeId) -> bool {
        let table = self.context.type_table.borrow();
        let kind = |id: TypeId| {
            table
                .get(Self::resolve_alias_chain(&table, id))
                .map(|t| t.kind.clone())
        };
        match (kind(have), kind(want)) {
            (
                Some(TypeKind::Enum { symbol_id: a, .. }),
                Some(TypeKind::Enum { symbol_id: b, .. }),
            ) => a == b,
            (
                Some(TypeKind::Enum { .. }),
                Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String),
            )
            | (
                Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String),
                Some(TypeKind::Enum { .. }),
            ) => false,
            _ => {
                drop(table);
                self.macro_probe_value_fits(have, want)
            }
        }
    }

    /// An abstract value assigned to a basic type that none of its `@:to`
    /// casts reaches for its type arguments. Decided only when a static
    /// `@:to` limits the instantiations it takes.
    fn macro_probe_abstract_to_mismatch(
        &self,
        mut value: &TypedExpression,
        target: TypeId,
    ) -> Option<String> {
        while let TypedExpressionKind::Cast { expression, .. } = &value.kind {
            value = &**expression;
        }
        let (abstract_symbol, args) = {
            let table = self.context.type_table.borrow();
            match table
                .get(Self::resolve_alias_chain(&table, value.expr_type))
                .map(|t| &t.kind)
            {
                Some(TypeKind::Abstract {
                    symbol_id,
                    type_args,
                    ..
                }) if !type_args.is_empty() => (*symbol_id, type_args.clone()),
                _ => return None,
            }
        };
        let kind = |id: TypeId| {
            let table = self.context.type_table.borrow();
            table
                .get(Self::resolve_alias_chain(&table, id))
                .map(|t| t.kind.clone())
        };
        let target_kind = kind(target);
        if !matches!(
            target_kind,
            Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String)
        ) {
            return None;
        }
        let owner = self.context.symbol_table.get_symbol(abstract_symbol)?;
        let prefix = format!(
            "{}.",
            self.context
                .string_interner
                .get(owner.qualified_name.unwrap_or(owner.name))?
        );
        let casts: Vec<(Vec<TypeId>, TypeId, bool)> = self
            .context
            .symbol_table
            .find_symbols(|s| {
                (s.flags
                    .contains(crate::tast::symbols::SymbolFlags::TO_CONVERSION)
                    || self.context.symbol_table.is_static_to_conversion(s.id))
                    && s.qualified_name
                        .and_then(|q| self.context.string_interner.get(q))
                        .is_some_and(|q| q.starts_with(&prefix))
            })
            .into_iter()
            .filter_map(|s| match kind(s.type_id) {
                Some(TypeKind::Function {
                    params,
                    return_type,
                    ..
                }) => Some((params, return_type, s.is_static())),
                _ => None,
            })
            .collect();
        let constrained = casts
            .iter()
            .any(|(params, _, is_static)| *is_static && params.len() == 1);
        if !constrained {
            return None;
        }
        let applies = |params: &[TypeId], is_static: bool| {
            if !is_static {
                return true;
            }
            let Some(Some(TypeKind::Class { type_args, .. })) = params.first().map(|p| kind(*p))
            else {
                return true;
            };
            type_args.len() != args.len()
                || type_args
                    .iter()
                    .zip(&args)
                    .all(|(&w, &h)| kind(w) == kind(h))
        };
        if casts.iter().any(|(params, ret, is_static)| {
            applies(params, *is_static) && kind(*ret) == target_kind
        }) {
            return None;
        }
        let render = |id| {
            super::macro_defer::render_type(
                id,
                self.context.type_table,
                self.context.symbol_table,
                self.context.string_interner,
                0,
            )
        };
        Some(format!(
            "{} should be {}",
            render(value.expr_type),
            render(target)
        ))
    }

    /// A map literal's keys are distinct constants, and its keys and its
    /// values each share a type.
    fn macro_probe_map_literal(
        &self,
        entries: &[crate::tast::node::TypedMapEntry],
    ) -> Option<String> {
        use crate::tast::node::LiteralValue;
        let constant = |e: &TypedExpression| match &e.kind {
            TypedExpressionKind::Literal { value } => match value {
                LiteralValue::Int(i) => Some(format!("{i}")),
                LiteralValue::String(s) => Some(format!("\"{s}\"")),
                LiteralValue::Bool(b) => Some(format!("{b}")),
                _ => None,
            },
            _ => None,
        };
        let mut seen: Vec<String> = Vec::new();
        for entry in entries {
            if let Some(key) = constant(&entry.key) {
                if seen.contains(&key) {
                    return Some(format!("Duplicate key {key}"));
                }
                seen.push(key);
            }
        }
        let (first, rest) = entries.split_first()?;
        let render = |id| {
            super::macro_defer::render_type(
                id,
                self.context.type_table,
                self.context.symbol_table,
                self.context.string_interner,
                0,
            )
        };
        for entry in rest {
            for (have, want) in [
                (entry.key.expr_type, first.key.expr_type),
                (entry.value.expr_type, first.value.expr_type),
            ] {
                if !self.macro_probe_value_fits(have, want)
                    && !self.macro_probe_value_fits(want, have)
                {
                    return Some(format!("{} should be {}", render(have), render(want)));
                }
            }
        }
        None
    }

    /// Whether a structure field typed `want` takes a value of type `have`.
    /// Only basic types are told apart; anything else is taken.
    fn macro_probe_value_fits(&self, have: TypeId, want: TypeId) -> bool {
        let table = self.context.type_table.borrow();
        let kind = |id: TypeId| {
            table
                .get(Self::resolve_alias_chain(&table, id))
                .map(|t| t.kind.clone())
        };
        let basic = |k: &Option<TypeKind>| {
            matches!(
                k,
                Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String)
            )
        };
        let (have, want) = (kind(have), kind(want));
        if !basic(&have) || !basic(&want) {
            return true;
        }
        have == want || matches!((have, want), (Some(TypeKind::Int), Some(TypeKind::Float)))
    }

    /// One abstract reaches another only through a cast either declares
    /// (`to B` on the source, `from A` on the target); casts do not chain,
    /// so `Kilometer to Float` and `Meter from Float` do not connect.
    fn macro_probe_abstract_assignable(&self, from: TypeId, to: TypeId) -> bool {
        let table = self.context.type_table.borrow();
        let abstract_of = |id: TypeId| match table
            .get(Self::resolve_alias_chain(&table, id))
            .map(|ty| &ty.kind)
        {
            Some(TypeKind::Abstract { symbol_id, .. }) => Some(*symbol_id),
            _ => None,
        };
        let (Some(source), Some(target)) = (abstract_of(from), abstract_of(to)) else {
            return true;
        };
        if source == target {
            return true;
        }
        let (Some((_, source_to)), Some((target_from, _))) = (
            self.abstract_casts.get(&source),
            self.abstract_casts.get(&target),
        ) else {
            return true;
        };
        // A cast over a type parameter or Dynamic admits anything.
        let admits = |cast: TypeId, other: SymbolId| {
            let cast = Self::resolve_alias_chain(&table, cast);
            table.is_type_parameter(cast)
                || matches!(table.get(cast).map(|ty| &ty.kind), Some(TypeKind::Dynamic))
                || abstract_of(cast) == Some(other)
        };
        source_to.iter().any(|&t| admits(t, target))
            || target_from.iter().any(|&t| admits(t, source))
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
