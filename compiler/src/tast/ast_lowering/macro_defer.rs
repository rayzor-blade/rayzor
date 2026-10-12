//! Deferred macro re-expansion: the lowering side.
//!
//! Macro expansion runs on the raw AST, before any typing exists. A macro
//! body that asks the typer a question (`Context.typeof`, `typeExpr`,
//! `TypeTools.toString` on a typed Type) cannot be answered there, so the
//! expander parks the call site untouched and records it as deferred. When
//! lowering reaches that call, everything the question needs is live: locals
//! upstream of the call are typed, the enclosing class is on the context
//! stack, imports are resolved. This module is the bridge — it re-runs the
//! parked macro with `AstLowering` itself answering as the `MacroTyper`.

use super::AstLowering;
use crate::macro_system::context_api::MacroTyper;
use crate::tast::TypeId;

/// `AstLowering` wearing its `MacroTyper` hat for the span of one deferred
/// call. Separate struct rather than an impl on `AstLowering` so the borrow
/// handed to the expander is visibly scoped.
pub(crate) struct DeferredMacroTyper<'l, 'a> {
    pub lowering: &'l mut AstLowering<'a>,
    pub receiver: Option<(&'l parser::Expr, TypeId)>,
    /// The type the call's position expects (`Context.getExpectedType`).
    pub expected: Option<TypeId>,
    /// Bindings imposed on the expected generic type by `Context.unify`.
    pub inferred: Option<&'l mut std::collections::BTreeMap<crate::tast::SymbolId, TypeId>>,
}

impl AstLowering<'_> {
    /// Record what a declaration expects of its initializer — the annotation,
    /// else a monomorph — for a deferred macro call that is the initializer.
    /// Returns the previous record for the caller to restore.
    pub(crate) fn enter_var_init(
        &mut self,
        init: &parser::Expr,
        declared: Option<TypeId>,
        hint: Option<&parser::Type>,
    ) -> Option<((usize, usize), TypeId)> {
        let declared = hint.and_then(|h| self.dynamic_of_view(h)).or(declared);
        let expected = declared.unwrap_or_else(|| self.context.type_table.borrow().unknown_type());
        self.var_init_expected
            .replace(((init.span.start, init.span.end), expected))
    }

    /// `Dynamic<T>` as macro typing sees it: a Dynamic carrying its
    /// parameter. The annotation itself lowers to plain Dynamic.
    pub(crate) fn dynamic_of_view(&mut self, hint: &parser::Type) -> Option<TypeId> {
        let parser::Type::Path { path, params, .. } = hint else {
            return None;
        };
        let std_types = path.package.is_empty() || path.package == ["StdTypes"];
        if path.name != "Dynamic" || !std_types || path.sub.is_some() || params.len() != 1 {
            return None;
        }
        let errors = (self.collected_errors.len(), self.context.errors.len());
        let value = self.lower_type(&params[0]);
        self.collected_errors.truncate(errors.0);
        self.context.errors.truncate(errors.1);
        let value = value.ok()?;
        let mut table = self.context.type_table.borrow_mut();
        let dynamic = table.dynamic_type();
        Some(table.create_generic_instance(dynamic, vec![value]))
    }

    /// What macro typing sees for a declared local beyond its runtime type:
    /// a closure handing back an untyped `var x = null` local, that local
    /// once an annotated declaration binds it through such a closure's
    /// result, or a call result typed by its callee's constraints.
    pub(crate) fn record_macro_local_views(
        &mut self,
        local: crate::tast::SymbolId,
        init: &parser::Expr,
        typed: &crate::tast::TypedExpression,
        declared: Option<TypeId>,
    ) {
        use crate::tast::core::TypeKind;
        let Some(declared) = declared else {
            if let Some(name) = closure_result_name(init) {
                let name = self.context.intern_string(name);
                if let Some(result) = self.resolve_symbol_in_scope_hierarchy(name)
                    && self.null_inferred.contains(&result)
                {
                    self.null_returning_closures.insert(local, result);
                }
            } else if let Some(view) = self.constrained_return_view(typed) {
                self.macro_source_types.insert(local, view);
            }
            return;
        };
        // `var s:T = f(..)` binds the local `f` returns to `Null<T>`.
        let parser::ExprKind::Call { expr: callee, .. } = &init.kind else {
            return;
        };
        let parser::ExprKind::Ident(name) = &callee.kind else {
            return;
        };
        let name = self.context.intern_string(name);
        let Some(result) = self
            .resolve_symbol_in_scope_hierarchy(name)
            .and_then(|closure| self.null_returning_closures.get(&closure).copied())
        else {
            return;
        };
        if !self.null_inferred.contains(&result) || self.macro_source_types.contains_key(&result) {
            return;
        }
        let kind = self
            .context
            .type_table
            .borrow()
            .get(declared)
            .map(|t| t.kind.clone());
        let view = match kind {
            Some(TypeKind::Optional { .. }) => declared,
            None
            | Some(TypeKind::Dynamic | TypeKind::Unknown | TypeKind::Error | TypeKind::Void) => {
                return;
            }
            Some(_) => self
                .context
                .type_table
                .borrow_mut()
                .create_optional_type(declared),
        };
        self.macro_source_types.insert(result, view);
    }

    /// A call whose result is one of its callee's own type parameters that
    /// the arguments leave unbound: that parameter's structure constraints,
    /// with the parameters the arguments do bind substituted, merged into
    /// one structure.
    fn constrained_return_view(&mut self, call: &crate::tast::TypedExpression) -> Option<TypeId> {
        use crate::tast::TypedExpressionKind as K;
        use crate::tast::core::TypeKind;
        if !self
            .context
            .type_table
            .borrow()
            .is_type_parameter(call.expr_type)
        {
            return None;
        }
        let (owner, method, arguments) = match &call.kind {
            K::StaticMethodCall {
                class_symbol,
                method_symbol,
                arguments,
                ..
            } => (*class_symbol, *method_symbol, arguments),
            K::FunctionCall {
                function,
                arguments,
                ..
            } => match &function.kind {
                K::StaticFieldAccess {
                    class_symbol,
                    field_symbol,
                } => (*class_symbol, *field_symbol, arguments),
                K::Variable {
                    symbol_id: field_symbol,
                }
                | K::FieldAccess { field_symbol, .. } => (
                    *self.context.class_context_stack.last()?,
                    *field_symbol,
                    arguments,
                ),
                _ => return None,
            },
            _ => return None,
        };
        let method = self.context.symbol_table.get_symbol(method)?;
        if matches!(
            method.kind,
            crate::tast::SymbolKind::Variable | crate::tast::SymbolKind::Parameter
        ) {
            return None;
        }
        let name = method.name;
        let sig = self.resolve_declared_method_sig(owner, name, true)?;
        let bare = |t: &parser::Type| match t {
            parser::Type::Path { path, params, .. }
                if path.package.is_empty() && path.sub.is_none() && params.is_empty() =>
            {
                Some(path.name.clone())
            }
            _ => None,
        };
        let returned = bare(sig.return_type.as_ref()?)?;
        let constraints = &sig
            .type_params
            .iter()
            .find(|tp| tp.name == returned)?
            .constraints;
        if constraints.is_empty() {
            return None;
        }
        // Each of the callee's parameters is what the argument declared as
        // it is; the rest stay unknown rather than resolve to outer names.
        let unknown = self.context.type_table.borrow().unknown_type();
        let mut scope = std::collections::BTreeMap::new();
        for tp in &sig.type_params {
            let bound = sig
                .params
                .iter()
                .zip(arguments.iter())
                .find_map(|(hint, mut arg)| {
                    if hint.as_ref().and_then(bare).as_deref() != Some(tp.name.as_str()) {
                        return None;
                    }
                    while let K::Cast {
                        expression,
                        cast_kind: crate::tast::node::CastKind::Implicit,
                        ..
                    } = &arg.kind
                    {
                        arg = &**expression;
                    }
                    Some(arg.expr_type)
                });
            scope.insert(
                self.context.intern_string(&tp.name),
                bound.unwrap_or(unknown),
            );
        }
        let errors = (self.collected_errors.len(), self.context.errors.len());
        self.context.push_type_parameters(scope);
        let lowered: Vec<_> = constraints.iter().map(|c| self.lower_type(c)).collect();
        self.context.pop_type_parameters();
        self.collected_errors.truncate(errors.0);
        self.context.errors.truncate(errors.1);
        let mut fields: Vec<crate::tast::core::AnonymousField> = Vec::new();
        for constraint in lowered {
            let constraint = constraint.ok()?;
            let more = {
                let table = self.context.type_table.borrow();
                match table
                    .get(Self::resolve_alias_chain(&table, constraint))
                    .map(|t| &t.kind)
                {
                    Some(TypeKind::Anonymous { fields }) => fields.clone(),
                    _ => return None,
                }
            };
            for field in more {
                match fields.iter_mut().find(|f| f.name == field.name) {
                    Some(existing) => *existing = field,
                    None => fields.push(field),
                }
            }
        }
        Some(
            self.context
                .type_table
                .borrow_mut()
                .create_type(TypeKind::Anonymous { fields }),
        )
    }

    /// The types `{>A, >B, ..}` or `A & B` is built from, in source order: an
    /// extension's own fields are not among them, an intersection's operands
    /// all are. None for any other type.
    pub(crate) fn extended_types(&mut self, t: &parser::Type) -> Option<Vec<TypeId>> {
        fn parts<'t>(t: &'t parser::Type, out: &mut Vec<&'t parser::Type>) {
            match t {
                // `{>A, ..}` parses as an intersection spanning from the brace.
                parser::Type::Intersection { left, right, span }
                    if span.start < left.span().start =>
                {
                    out.push(left);
                    if !matches!(**right, parser::Type::Anonymous { .. }) {
                        parts(right, out);
                    }
                }
                parser::Type::Intersection { left, right, .. } => {
                    parts(left, out);
                    parts(right, out);
                }
                other => out.push(other),
            }
        }
        if !matches!(t, parser::Type::Intersection { .. }) {
            return None;
        }
        let mut leaves = Vec::new();
        parts(t, &mut leaves);
        leaves
            .into_iter()
            .map(|t| self.lower_type(t).ok())
            .collect()
    }

    /// A `?.` chain is nullable as a whole: `a?.b.c` is `Null<typeof c>`.
    fn safe_chain_type(&self, e: &crate::tast::TypedExpression) -> TypeId {
        use crate::tast::TypedExpressionKind as K;
        fn safe(e: &crate::tast::TypedExpression) -> bool {
            match &e.kind {
                K::FieldAccess {
                    object,
                    is_optional,
                    ..
                } => *is_optional || safe(object),
                K::MethodCall {
                    receiver,
                    is_optional,
                    ..
                } => *is_optional || safe(receiver),
                K::MethodReference { receiver, .. } => safe(receiver),
                K::FunctionCall { function, .. } => safe(function),
                K::ArrayAccess { array, .. } => safe(array),
                _ => false,
            }
        }
        let nullable = matches!(
            self.context
                .type_table
                .borrow()
                .get(e.expr_type)
                .map(|t| &t.kind),
            Some(crate::tast::core::TypeKind::Optional { .. })
        );
        if nullable || !safe(e) {
            return e.expr_type;
        }
        self.context
            .type_table
            .borrow_mut()
            .create_optional_type(e.expr_type)
    }

    pub(crate) fn generic_build_type(
        &mut self,
        symbol: crate::tast::SymbolId,
        params: &[parser::Type],
        args: Option<&[parser::Expr]>,
        span: parser::Span,
    ) -> super::LoweringResult<Option<Option<parser::Type>>> {
        if let Some(ty) = self.const_generic_type(symbol, params, args.is_some(), span)? {
            return Ok(Some(Some(ty)));
        }
        let Some(engine) = self.generic_build_engine.clone() else {
            return Ok(None);
        };
        let Some(sym) = self
            .context
            .symbol_table
            .get_symbol(symbol)
            .filter(|s| s.kind == crate::tast::SymbolKind::Class)
        else {
            return Ok(None);
        };
        let name = self
            .context
            .string_interner
            .get(sym.qualified_name.unwrap_or(sym.name))
            .unwrap_or("")
            .to_string();
        let Some(info) = engine.definition(&name) else {
            return Ok(None);
        };
        if self.generic_build_resolving.contains(&symbol) {
            return Err(super::LoweringError::SemanticError {
                message: format!("recursive generic build result for '{name}'"),
                location: self.context.create_location_from_span(span),
            });
        }
        let type_args = params
            .iter()
            .map(|p| self.lower_type(p))
            .collect::<super::LoweringResult<Vec<_>>>()?;
        let key = (symbol, type_args.clone());
        if args.is_none() || !params.is_empty() {
            if let Some(result) = self.generic_build_results.get(&key) {
                return Ok(Some(Some(result.clone())));
            }
        }
        if !self.generic_build_active.insert(key.clone()) {
            return Err(super::LoweringError::SemanticError {
                message: format!("recursive generic build for '{name}'"),
                location: self.context.create_location_from_span(span),
            });
        }
        let local_type = self
            .context
            .type_table
            .borrow_mut()
            .create_class_type(symbol, type_args);
        let result = {
            let mut typer = DeferredMacroTyper {
                lowering: self,
                receiver: None,
                expected: None,
                inferred: None,
            };
            engine.evaluate(&info, local_type, args, span, &mut typer)
        };
        self.generic_build_active.remove(&key);
        let result = result.map_err(|e| super::LoweringError::SemanticError {
            message: format!("generic build for '{name}' failed: {e}"),
            location: self.context.create_location_from_span(span),
        })?;
        for defined in result.defined_types {
            self.lower_generated_type(&defined)?;
        }
        if (args.is_none() || !params.is_empty())
            && let Some(ty) = &result.type_hint
        {
            self.generic_build_results.insert(key, ty.clone());
        }
        Ok(Some(result.type_hint))
    }

    fn lower_generated_type(
        &mut self,
        defined: &crate::macro_system::context_api::DefinedType,
    ) -> super::LoweringResult<()> {
        let name = defined
            .pack
            .iter()
            .cloned()
            .chain(std::iter::once(defined.name.clone()))
            .collect::<Vec<_>>()
            .join(".");
        if !self.generated_type_names.insert(name.clone()) {
            return Err(super::LoweringError::SemanticError {
                message: format!("type '{name}' is already defined"),
                location: self
                    .context
                    .create_location_from_span(parser::Span::default()),
            });
        }
        let declaration = crate::macro_system::expander::defined_declaration(defined);
        self.lower_generated_declaration(&declaration, &defined.pack)
    }

    pub(crate) fn lower_generated_declaration(
        &mut self,
        declaration: &parser::TypeDeclaration,
        pack: &[String],
    ) -> super::LoweringResult<()> {
        let result = self.in_generated_context(pack, |this| {
            this.pre_register_declaration(declaration)?;
            if let parser::TypeDeclaration::Class(class) = declaration {
                this.pre_register_class_fields(class)?;
            }
            this.lower_declaration(declaration)
        });
        self.generated_declarations.push(result?);
        Ok(())
    }

    /// Run `lower` as a declaration of `pack`, outside the calling function.
    pub(crate) fn in_generated_context<R>(
        &mut self,
        pack: &[String],
        lower: impl FnOnce(&mut Self) -> super::LoweringResult<R>,
    ) -> super::LoweringResult<R> {
        let scope = self.context.current_scope;
        let package = self.context.current_package;
        let classes = std::mem::take(&mut self.context.class_context_stack);
        let type_params = std::mem::take(&mut self.context.type_parameter_stack);
        let case_bindings = std::mem::take(&mut self.context.case_type_bindings);
        let return_type = self.context.expected_return_type.take();
        let new_hint = self.context.expected_new_type_hint.take();
        let switch_type = self.context.switch_discriminant_type.take();
        let static_method = self.in_static_method;
        let callee = self.lowering_callee;
        let closure_depth = self.closure_depth;
        let lambda_params = std::mem::take(&mut self.expected_lambda_params_stack);
        let arg_types = std::mem::take(&mut self.expected_arg_type_stack);
        let map_uses = std::mem::take(&mut self.map_first_uses);
        self.context.current_scope = crate::tast::ScopeId::first();
        self.context.current_package = None;
        self.set_package_from_parts(pack);
        self.in_static_method = false;
        self.lowering_callee = false;
        self.closure_depth = 0;
        let result = lower(self);
        self.context.current_scope = scope;
        self.context.current_package = package;
        self.context.class_context_stack = classes;
        self.context.type_parameter_stack = type_params;
        self.context.case_type_bindings = case_bindings;
        self.context.expected_return_type = return_type;
        self.context.expected_new_type_hint = new_hint;
        self.context.switch_discriminant_type = switch_type;
        self.in_static_method = static_method;
        self.lowering_callee = callee;
        self.closure_depth = closure_depth;
        self.expected_lambda_params_stack = lambda_params;
        self.expected_arg_type_stack = arg_types;
        self.map_first_uses = map_uses;
        result
    }

    /// Static macros inside another macro's quoted argument reach typing unexpanded.
    pub(crate) fn static_macro_name(&mut self, callee: &parser::Expr) -> Option<String> {
        use crate::tast::SymbolKind;
        use parser::ExprKind;

        let registry = self.deferred_macro_registry.as_ref()?;
        if let ExprKind::Ident(name) = &callee.kind {
            let name = self.context.intern_string(name);
            let symbol = self.resolve_symbol_in_scope_hierarchy(name)?;
            let symbol = self.context.symbol_table.get_symbol(symbol)?;
            let qualified = self.context.string_interner.get(symbol.qualified_name?)?;
            return registry
                .get_macro(qualified)
                .filter(|definition| definition.is_static)
                .map(|definition| definition.qualified_name.clone());
        }
        let ExprKind::Field { expr, field, .. } = &callee.kind else {
            return None;
        };
        if !registry
            .all_macros()
            .any(|definition| definition.name == *field)
        {
            return None;
        }
        fn path(expression: &parser::Expr) -> Option<String> {
            match &expression.kind {
                ExprKind::Ident(name) => Some(name.clone()),
                ExprKind::Field { expr, field, .. } => Some(format!("{}.{field}", path(expr)?)),
                _ => None,
            }
        }
        let path = path(expr)?;
        let first = self.context.intern_string(path.split('.').next()?);
        if let Some(symbol) = self.resolve_symbol_in_scope_hierarchy(first)
            && self
                .context
                .symbol_table
                .get_symbol(symbol)
                .is_some_and(|symbol| {
                    !matches!(
                        symbol.kind,
                        SymbolKind::Class
                            | SymbolKind::Abstract
                            | SymbolKind::Enum
                            | SymbolKind::TypeAlias
                    )
                })
        {
            return None;
        }
        let name = self.context.intern_string(&path);
        let owner = self.resolve_class_like_symbol_by_name(name)?;
        let owner = self.context.symbol_table.get_symbol(owner)?;
        let owner = self
            .context
            .string_interner
            .get(owner.qualified_name.unwrap_or(owner.name))?;
        registry
            .get_macro(&format!("{owner}.{field}"))
            .filter(|definition| definition.is_static)
            .map(|definition| definition.qualified_name.clone())
    }

    /// Resolve compile-time members against the typed receiver before ordinary arguments lower.
    pub(crate) fn lower_receiver_macro_call(
        &mut self,
        call: &parser::Expr,
        callee: &parser::Expr,
        args: &[parser::Expr],
    ) -> Result<Option<crate::tast::TypedExpression>, super::LoweringError> {
        use crate::tast::{SymbolKind, TypedExpressionKind};
        use parser::ExprKind;

        let ExprKind::Field {
            expr: receiver_ast,
            field,
            ..
        } = &callee.kind
        else {
            return Ok(None);
        };
        let Some(registry) = self.deferred_macro_registry.as_ref() else {
            return Ok(None);
        };
        if !registry.all_macros().any(|def| def.name == *field) {
            return Ok(None);
        }
        fn type_path(expr: &parser::Expr) -> Option<String> {
            match &expr.kind {
                ExprKind::Ident(name) => Some(name.clone()),
                ExprKind::Field { expr, field, .. } => {
                    Some(format!("{}.{field}", type_path(expr)?))
                }
                _ => None,
            }
        }
        if let Some(path) = type_path(receiver_ast) {
            let first = self.context.intern_string(path.split('.').next().unwrap());
            let bound = self.resolve_symbol_in_scope_hierarchy(first);
            if bound
                .and_then(|s| self.context.symbol_table.get_symbol(s))
                .is_some_and(|s| {
                    matches!(
                        s.kind,
                        SymbolKind::Class
                            | SymbolKind::Abstract
                            | SymbolKind::Enum
                            | SymbolKind::TypeAlias
                    )
                })
            {
                return Ok(None);
            }
            if bound.is_none() && path.contains('.') {
                let path = self.context.intern_string(&path);
                if self.resolve_class_like_symbol_by_name(path).is_some() {
                    return Ok(None);
                }
            }
        }
        let receiver = self.lower_expression(receiver_ast)?;
        if matches!(receiver.kind, TypedExpressionKind::Variable { symbol_id }
            if self.context.symbol_table.get_symbol(symbol_id).is_some_and(|s|
                matches!(s.kind, SymbolKind::Class | SymbolKind::Abstract | SymbolKind::Enum)))
        {
            return Ok(None);
        }
        let method = self.context.intern_string(field);
        let builtin = {
            let tt = self.context.type_table.borrow();
            let mut ty = receiver.expr_type;
            for _ in 0..8 {
                match tt.get(ty).map(|t| &t.kind) {
                    Some(crate::tast::TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    Some(crate::tast::TypeKind::Optional { inner_type }) => ty = *inner_type,
                    _ => break,
                }
            }
            match tt.get(ty).map(|t| &t.kind) {
                Some(crate::tast::TypeKind::Array { .. }) => Some("Array"),
                Some(crate::tast::TypeKind::String) => Some("String"),
                _ => None,
            }
        };
        if let (Some(class), Some(index)) = (builtin, &self.static_sig_index) {
            let resolve_file = |q: &str| {
                self.context
                    .namespace_resolver
                    .resolve_qualified_path_to_file_force(q)
            };
            if index
                .borrow_mut()
                .resolve(class, field, false, &resolve_file)
                .is_some()
            {
                return Ok(None);
            }
        }
        let registry = self.deferred_macro_registry.as_ref().unwrap();
        let find = |owner| {
            let sym = self.context.symbol_table.get_symbol(owner)?;
            let owner_name = self
                .context
                .string_interner
                .get(sym.qualified_name.unwrap_or(sym.name))?;
            registry.get_macro(&format!("{owner_name}.{field}"))
        };

        let mut class = self.resolve_type_to_class_symbol(receiver.expr_type);
        let mut seen = std::collections::BTreeSet::new();
        let mut name = None;
        let mut type_usings = Vec::new();
        while let Some(owner) = class {
            if !seen.insert(owner) {
                break;
            }
            if let Some(def) = find(owner).filter(|def| !def.is_static) {
                name = Some(def.qualified_name.clone());
                break;
            }
            let has_field = self.class_fields.get(&owner).is_some_and(|fields| {
                fields.iter().any(|(n, s, _)| {
                    *n == method
                        && self
                            .context
                            .symbol_table
                            .get_symbol(*s)
                            .is_some_and(|s| s.kind != SymbolKind::Function)
                })
            });
            if has_field || self.resolve_class_method_symbol(owner, method).is_some() {
                return Ok(None);
            }
            if let Some(sym) = self.context.symbol_table.get_symbol(owner) {
                if let Some(usings) = self.type_usings.get(&sym.name) {
                    type_usings.extend(usings.iter().copied());
                }
            }
            class = self.parent_class_symbol(owner);
        }
        if name.is_none() {
            let late = self
                .unresolved_usings
                .iter()
                .rev()
                .chain(type_usings.iter())
                .filter_map(|n| self.resolve_class_like_symbol_by_name(*n));
            for owner in self.using_modules.iter().rev().map(|(_, s)| *s).chain(late) {
                if let Some(def) = find(owner).filter(|def| def.is_static) {
                    name = Some(def.qualified_name.clone());
                    break;
                }
                if self.resolve_class_method_symbol(owner, method).is_some() {
                    return Ok(None);
                }
            }
        }
        let Some(name) = name else { return Ok(None) };
        let mut macro_args = Vec::with_capacity(args.len() + 1);
        macro_args.push((**receiver_ast).clone());
        macro_args.extend_from_slice(args);
        let macro_call = parser::Expr {
            kind: ExprKind::Call {
                expr: Box::new(callee.clone()),
                args: macro_args,
            },
            span: call.span,
        };
        let expanded = {
            let mut typer = DeferredMacroTyper {
                lowering: self,
                receiver: Some((receiver_ast, receiver.expr_type)),
                expected: None,
                inferred: None,
            };
            typer.expand_deferred_call(&name, &macro_call)
        }
        .map_err(|e| super::LoweringError::SemanticError {
            message: format!("macro '{}' failed during typing: {}", name, e),
            location: self.context.create_location_from_span(call.span),
        })?;
        self.lower_expression(&expanded).map(Some)
    }

    /// `x.name` read through a macro `@:resolve`/`@:op(a.b)` of `x`'s
    /// abstract: the expansion of `resolver(x, "name")`.
    pub(crate) fn lower_resolve_macro_field(
        &mut self,
        field_expr: &parser::Expr,
        receiver_ast: &parser::Expr,
        field: &str,
        receiver_type: TypeId,
    ) -> Result<Option<crate::tast::TypedExpression>, super::LoweringError> {
        use crate::tast::TypeKind;
        use parser::ExprKind;

        let Some(registry) = self.deferred_macro_registry.as_ref() else {
            return Ok(None);
        };
        let owner = {
            let tt = self.context.type_table.borrow();
            let mut ty = receiver_type;
            for _ in 0..8 {
                match tt.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            match tt.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Abstract { symbol_id, .. }) => *symbol_id,
                _ => return Ok(None),
            }
        };
        let Some(sym) = self.context.symbol_table.get_symbol(owner) else {
            return Ok(None);
        };
        let interner = &self.context.string_interner;
        let bare = interner.get(sym.name).unwrap_or_default();
        let qualified = sym
            .qualified_name
            .and_then(|q| interner.get(q))
            .unwrap_or(bare);
        let owner_of = |def: &crate::macro_system::registry::MacroDefinition| {
            def.qualified_name
                .rsplit_once('.')
                .map_or("", |(owner, _)| owner)
                .to_string()
        };
        let resolvers: Vec<_> = registry
            .all_macros()
            .filter(|def| def.is_resolver && def.params.len() == 2)
            .collect();
        // The abstract's qualified name first, else its bare name.
        let Some(def) = resolvers
            .iter()
            .find(|def| owner_of(**def) == qualified)
            .or_else(|| {
                resolvers.iter().find(|def| {
                    let o = owner_of(**def);
                    o == bare || o.ends_with(&format!(".{bare}"))
                })
            })
        else {
            return Ok(None);
        };
        let (name, method) = (def.qualified_name.clone(), def.name.clone());
        let span = field_expr.span;
        let macro_call = parser::Expr {
            kind: ExprKind::Call {
                expr: Box::new(parser::Expr {
                    kind: ExprKind::Field {
                        expr: Box::new(receiver_ast.clone()),
                        field: method,
                        is_optional: false,
                    },
                    span,
                }),
                args: vec![
                    receiver_ast.clone(),
                    parser::Expr {
                        kind: ExprKind::String(field.to_string()),
                        span,
                    },
                ],
            },
            span,
        };
        let expanded = {
            let mut typer = DeferredMacroTyper {
                lowering: self,
                receiver: Some((receiver_ast, receiver_type)),
                expected: None,
                inferred: None,
            };
            typer.expand_deferred_call(&name, &macro_call)
        }
        .map_err(|e| super::LoweringError::SemanticError {
            message: format!("macro '{}' failed during typing: {}", name, e),
            location: self.context.create_location_from_span(span),
        })?;
        self.lower_expression(&expanded).map(Some)
    }

    /// `var v:A = e` where `e` does not unify with the abstract `A`, which
    /// declares a macro `@:from`: the expansion of that macro over `e`.
    pub(crate) fn lower_macro_from(
        &mut self,
        init: &parser::Expr,
        typed: &crate::tast::TypedExpression,
        declared: Option<TypeId>,
    ) -> Result<Option<crate::tast::TypedExpression>, super::LoweringError> {
        use crate::tast::TypeKind;
        use parser::ExprKind;

        let Some(declared) = declared else {
            return Ok(None);
        };
        let Some(registry) = self.deferred_macro_registry.as_ref() else {
            return Ok(None);
        };
        let owner = {
            let tt = self.context.type_table.borrow();
            let mut ty = declared;
            for _ in 0..8 {
                match tt.get(ty).map(|t| &t.kind) {
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            match tt.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Abstract { symbol_id, .. }) => *symbol_id,
                _ => return Ok(None),
            }
        };
        let Some(sym) = self.context.symbol_table.get_symbol(owner) else {
            return Ok(None);
        };
        let interner = &self.context.string_interner;
        let bare = interner.get(sym.name).unwrap_or_default().to_string();
        let qualified = sym
            .qualified_name
            .and_then(|q| interner.get(q))
            .unwrap_or(bare.as_str())
            .to_string();
        let owner_of = |def: &crate::macro_system::registry::MacroDefinition| {
            def.qualified_name
                .rsplit_once('.')
                .map_or("", |(owner, _)| owner)
                .to_string()
        };
        let converters: Vec<_> = registry
            .all_macros()
            .filter(|def| def.is_from && def.is_static && def.params.len() == 1)
            .collect();
        let Some(def) = converters
            .iter()
            .find(|def| owner_of(**def) == qualified)
            .or_else(|| {
                converters.iter().find(|def| {
                    let o = owner_of(**def);
                    o == bare || o.ends_with(&format!(".{bare}"))
                })
            })
        else {
            return Ok(None);
        };
        let (name, method) = (def.qualified_name.clone(), def.name.clone());
        // A value the abstract already accepts, or one not yet typed, converts
        // as it would without the macro.
        let undecided = matches!(
            self.context
                .type_table
                .borrow()
                .get(typed.expr_type)
                .map(|t| &t.kind),
            None | Some(
                TypeKind::Unknown
                    | TypeKind::Error
                    | TypeKind::Placeholder { .. }
                    | TypeKind::TypeParameter { .. }
            )
        );
        if undecided
            || unify(
                typed.expr_type,
                declared,
                self.context.type_table,
                &self.abstract_casts,
                &std::collections::BTreeSet::new(),
                &mut std::collections::BTreeMap::new(),
                0,
            )
        {
            return Ok(None);
        }
        let span = init.span;
        let macro_call = parser::Expr {
            kind: ExprKind::Call {
                expr: Box::new(parser::Expr {
                    kind: ExprKind::Field {
                        expr: Box::new(parser::Expr {
                            kind: ExprKind::Ident(bare),
                            span,
                        }),
                        field: method,
                        is_optional: false,
                    },
                    span,
                }),
                args: vec![init.clone()],
            },
            span,
        };
        let mut inferred = std::collections::BTreeMap::new();
        let expanded = {
            let mut typer = DeferredMacroTyper {
                lowering: self,
                receiver: None,
                expected: Some(declared),
                inferred: Some(&mut inferred),
            };
            typer.expand_deferred_call(&name, &macro_call)
        }
        .map_err(|e| super::LoweringError::SemanticError {
            message: format!("macro '{}' failed during typing: {}", name, e),
            location: self.context.create_location_from_span(span),
        })?;
        let bindings: Vec<_> = inferred.into_iter().collect();
        let declared = self.substitute_type_bindings(declared, &bindings, false);
        self.expected_arg_type_stack.push(Some(declared));
        let previous_hint = self.context.expected_new_type_hint.replace(declared);
        let lowered = self.lower_value_expression(&expanded);
        self.context.expected_new_type_hint = previous_hint;
        self.expected_arg_type_stack.pop();
        lowered.map(Some)
    }
}

impl DeferredMacroTyper<'_, '_> {
    pub(crate) fn expand_deferred_call(
        &mut self,
        name: &str,
        call: &parser::Expr,
    ) -> Result<parser::Expr, crate::macro_system::errors::MacroError> {
        let cell = self
            .lowering
            .deferred_macro_expander
            .expect("deferred call recorded without its expander");
        match cell.try_borrow_mut() {
            Ok(mut engine) => engine.expand_deferred_call(name, call, self),
            Err(_) => self
                .lowering
                .deferred_macro_template
                .clone()
                .expect("deferred call recorded without its template")
                .fork_for_typing()
                .expand_deferred_call(name, call, self),
        }
    }

    /// A forward-declared type the typer left as a named placeholder, as the
    /// type that name resolves to at the call site.
    fn resolve_placeholder(&mut self, ty: TypeId) -> TypeId {
        use crate::tast::core::TypeKind;
        let name = match self
            .lowering
            .context
            .type_table
            .borrow()
            .get(ty)
            .map(|t| &t.kind)
        {
            Some(TypeKind::Placeholder { name }) => *name,
            _ => return ty,
        };
        let Some(name) = self
            .lowering
            .context
            .string_interner
            .get(name)
            .map(str::to_string)
        else {
            return ty;
        };
        match self.resolve_type_by_name(&name) {
            Ok(resolved)
                if matches!(
                    self.lowering
                        .context
                        .type_table
                        .borrow()
                        .get(resolved)
                        .map(|t| &t.kind),
                    Some(
                        TypeKind::Class { .. }
                            | TypeKind::Interface { .. }
                            | TypeKind::Enum { .. }
                            | TypeKind::Abstract { .. }
                    )
                ) =>
            {
                resolved
            }
            _ => ty,
        }
    }

    /// An `@:overload` signature as a `ClassField`. Its type is a `TFun` over
    /// `{name, opt, t}` args, as haxe builds it; a function TypeId has no
    /// argument names.
    fn overload_field_view(
        &mut self,
        name: &str,
        func: &parser::Function,
    ) -> crate::macro_system::value::MacroValue {
        use crate::macro_system::value::MacroValue as V;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let errors_before = self.lowering.collected_errors.len();
        let ctx_errors_before = self.lowering.context.errors.len();
        let (dynamic, void) = {
            let table = self.lowering.context.type_table.borrow();
            (table.dynamic_type(), table.void_type())
        };
        let mut args = Vec::with_capacity(func.params.len());
        for param in &func.params {
            let t = param.type_hint.as_ref().map_or(dynamic, |hint| {
                self.lowering.lower_type(hint).unwrap_or(dynamic)
            });
            args.push(V::Object(Arc::new(BTreeMap::from([
                (
                    "name".to_string(),
                    V::String(Arc::from(param.name.as_str())),
                ),
                ("opt".to_string(), V::Bool(param.optional)),
                ("t".to_string(), V::Type(t)),
            ]))));
        }
        let ret = func.return_type.as_ref().map_or(void, |hint| {
            self.lowering.lower_type(hint).unwrap_or(dynamic)
        });
        self.lowering.collected_errors.truncate(errors_before);
        self.lowering.context.errors.truncate(ctx_errors_before);
        let fun = V::Enum(
            Arc::from("Type"),
            Arc::from("TFun"),
            Arc::new(vec![V::Array(Arc::new(args)), V::Type(ret)]),
        );
        V::Object(Arc::new(BTreeMap::from([
            ("name".to_string(), V::String(Arc::from(name))),
            ("type".to_string(), fun),
            (
                "meta".to_string(),
                V::Object(Arc::new(BTreeMap::from([(
                    "__meta__".to_string(),
                    V::Array(Arc::new(Vec::new())),
                )]))),
            ),
        ])))
    }

    fn definition_target(&self, symbol: crate::tast::SymbolId) -> Option<TypeId> {
        let table = self.lowering.context.type_table.borrow();
        let sym = self.lowering.context.symbol_table.get_symbol(symbol)?;
        match table.get(sym.type_id).map(|t| &t.kind) {
            Some(crate::tast::TypeKind::Abstract { underlying, .. }) => *underlying,
            Some(crate::tast::TypeKind::TypeAlias { target_type, .. }) => Some(*target_type),
            _ => match sym.kind {
                crate::tast::SymbolKind::Abstract => table.resolve_abstract_underlying(symbol),
                crate::tast::SymbolKind::TypeAlias => Some(table.resolve_type_alias(symbol)),
                _ => None,
            },
        }
    }

    /// An abstract's type with a fresh unknown for each of its parameters.
    fn with_unknown_arguments(&self, abs: TypeId) -> TypeId {
        let kind = self
            .lowering
            .context
            .type_table
            .borrow()
            .get(abs)
            .map(|t| t.kind.clone());
        let Some(crate::tast::TypeKind::Abstract {
            symbol_id,
            underlying,
            type_args,
        }) = kind
        else {
            return abs;
        };
        let params = self.definition_parameters(symbol_id).len();
        if params == 0 || !type_args.is_empty() {
            return abs;
        }
        let mut table = self.lowering.context.type_table.borrow_mut();
        let unknown = table.unknown_type();
        table.create_abstract_type(symbol_id, underlying, vec![unknown; params])
    }

    /// A call result's type with each type parameter not in scope at the
    /// call replaced by a monomorph, as Haxe instantiates an unbound one.
    fn unbind_foreign_parameters(&mut self, id: TypeId, depth: usize) -> TypeId {
        use crate::tast::core::TypeKind;
        if depth > 16 {
            return id;
        }
        let param = match self
            .lowering
            .context
            .type_table
            .borrow()
            .get(id)
            .map(|t| &t.kind)
        {
            Some(TypeKind::TypeParameter { symbol_id, .. }) => Some(*symbol_id),
            _ => None,
        };
        if let Some(symbol) = param {
            let in_scope = self
                .lowering
                .context
                .symbol_table
                .get_symbol(symbol)
                .and_then(|s| self.lowering.context.resolve_type_parameter(s.name))
                .is_some();
            return if in_scope {
                id
            } else {
                self.lowering.context.type_table.borrow().unknown_type()
            };
        }
        let children = self.type_children(id);
        let unbound: Vec<TypeId> = children
            .iter()
            .map(|&child| self.unbind_foreign_parameters(child, depth + 1))
            .collect();
        self.rebuild_type(id, &unbound)
    }

    fn definition_parameters(&self, symbol: crate::tast::SymbolId) -> Vec<TypeId> {
        if let Some(params) = self
            .lowering
            .context
            .symbol_table
            .get_class_type_params(symbol)
        {
            return params.clone();
        }
        let table = self.lowering.context.type_table.borrow();
        self.lowering
            .context
            .symbol_table
            .get_symbol(symbol)
            .and_then(|s| table.get(s.type_id))
            .and_then(|t| match &t.kind {
                crate::tast::TypeKind::TypeAlias { type_args, .. } => Some(type_args.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }
}

impl MacroTyper for DeferredMacroTyper<'_, '_> {
    fn type_expr_in_scope(&mut self, expr: &parser::Expr) -> Result<TypeId, String> {
        if let Some((receiver, ty)) = self.receiver {
            if expr == receiver {
                return Ok(ty);
            }
        }
        // A typing PROBE: lower the expression for its type and discard the
        // result. Errors raised by the probe belong to the macro (typeError
        // catches them); they must not surface as user diagnostics, so both
        // error sinks are wound back to where they were.
        let outer_before = self.lowering.collected_errors.len();
        let ctx_before = self.lowering.context.errors.len();

        self.lowering.macro_probe_depth += 1;
        let result = self.lowering.lower_expression(expr);
        self.lowering.macro_probe_depth -= 1;

        let mut probe_errors: Vec<String> = self
            .lowering
            .collected_errors
            .drain(outer_before..)
            .map(|e| e.to_compilation_error().message)
            .collect();
        probe_errors.extend(
            self.lowering
                .context
                .errors
                .drain(ctx_before..)
                .map(|e| e.to_compilation_error().message),
        );

        match result {
            Ok(typed) if probe_errors.is_empty() => {
                if let crate::tast::TypedExpressionKind::Variable { symbol_id } = &typed.kind
                    && let Some(&source) = self.lowering.macro_source_types.get(symbol_id)
                {
                    return Ok(source);
                }
                // An unhinted enum abstract value has its abstract's type.
                if let Some(abs) = self.lowering.enum_abstract_value_type(&typed) {
                    let dynamic = self.lowering.context.type_table.borrow().dynamic_type();
                    if typed.expr_type == dynamic {
                        return Ok(self.with_unknown_arguments(abs));
                    }
                }
                let ty = self.lowering.safe_chain_type(&typed);
                let ty = self.resolve_placeholder(ty);
                let ty = match typed.kind {
                    crate::tast::TypedExpressionKind::FunctionCall { .. }
                    | crate::tast::TypedExpressionKind::MethodCall { .. }
                    | crate::tast::TypedExpressionKind::StaticMethodCall { .. } => {
                        self.unbind_foreign_parameters(ty, 0)
                    }
                    _ => ty,
                };
                // `a ?? b` where `b` never completes is `a` without its Null.
                let mut inner = expr;
                while let parser::ExprKind::Paren(e) = &inner.kind {
                    inner = &**e;
                }
                if let parser::ExprKind::Binary {
                    op: parser::BinaryOp::NullCoal,
                    right,
                    ..
                } = &inner.kind
                    && dead_end(right) != Flow::Completes
                    && let Some(crate::tast::TypeKind::Optional { inner_type }) = self
                        .lowering
                        .context
                        .type_table
                        .borrow()
                        .get(ty)
                        .map(|t| t.kind.clone())
                {
                    return Ok(inner_type);
                }
                // `a ??= b` (parsed as `a = a ?? b`) is `a`'s type, out of
                // `Null` when `b` is a basic type.
                if let parser::ExprKind::Assign {
                    left,
                    op: parser::AssignOp::Assign,
                    right,
                } = &inner.kind
                    && let parser::ExprKind::Binary {
                        left: read,
                        op: parser::BinaryOp::NullCoal,
                        ..
                    } = &right.kind
                    && read == left
                    && let crate::tast::TypedExpressionKind::BinaryOp { right: value, .. } =
                        &typed.kind
                    && let crate::tast::TypedExpressionKind::BinaryOp { right: value, .. } =
                        &value.kind
                {
                    let table = self.lowering.context.type_table.borrow();
                    let basic = matches!(
                        table.get(value.expr_type).map(|t| &t.kind),
                        Some(
                            crate::tast::TypeKind::Int
                                | crate::tast::TypeKind::Float
                                | crate::tast::TypeKind::Bool
                        )
                    );
                    if basic
                        && let Some(crate::tast::TypeKind::Optional { inner_type }) =
                            table.get(ty).map(|t| &t.kind)
                    {
                        return Ok(*inner_type);
                    }
                }
                Ok(ty)
            }
            Ok(_) => Err(probe_errors.join("\n")),
            Err(e) => {
                let mut msg = e.to_compilation_error().message;
                for extra in probe_errors {
                    msg.push('\n');
                    msg.push_str(&extra);
                }
                Err(msg)
            }
        }
    }

    fn enum_constructor_access(
        &mut self,
        expression: &parser::Expr,
    ) -> Option<(TypeId, crate::macro_system::value::MacroValue)> {
        use crate::macro_system::value::MacroValue as V;
        use crate::tast::{SymbolKind, TypedExpressionKind};
        use std::{collections::BTreeMap, sync::Arc};

        fn is_path(expression: &parser::Expr) -> bool {
            match &expression.kind {
                parser::ExprKind::Ident(_) => true,
                parser::ExprKind::Field { expr, .. } | parser::ExprKind::Paren(expr) => {
                    is_path(expr)
                }
                _ => false,
            }
        }
        if !is_path(expression) {
            return None;
        }
        let typed = self.lowering.lower_expression(expression).ok()?;
        let symbol = match typed.kind {
            TypedExpressionKind::Variable { symbol_id } => symbol_id,
            TypedExpressionKind::StaticFieldAccess { field_symbol, .. } => field_symbol,
            _ => return None,
        };
        let table = &self.lowering.context.symbol_table;
        let field = table.get_symbol(symbol)?;
        if field.kind != SymbolKind::EnumVariant {
            return None;
        }
        let parent = table.find_parent_enum_for_constructor(symbol)?;
        let enum_type = table.get_symbol(parent)?.type_id;
        let index = table
            .get_enum_variants(parent)?
            .iter()
            .position(|&id| id == symbol)?;
        let name = self.lowering.context.string_interner.get(field.name)?;
        Some((
            enum_type,
            V::Object(Arc::new(BTreeMap::from([
                ("name".to_string(), V::String(Arc::from(name))),
                ("index".to_string(), V::Int(index as i64)),
                ("type".to_string(), V::Type(typed.expr_type)),
                ("params".to_string(), V::Array(Arc::new(Vec::new()))),
            ]))),
        ))
    }

    fn field_access_kind(&mut self, receiver: TypeId, name: &str) -> Option<(bool, bool)> {
        use crate::tast::core::TypeKind;
        let mut ty = receiver;
        for _ in 0..8 {
            let kind = self
                .lowering
                .context
                .type_table
                .borrow()
                .get(ty)
                .map(|t| t.kind.clone())?;
            match kind {
                TypeKind::String => return Some((true, name != "length")),
                TypeKind::Array { .. } => return Some((true, name != "length")),
                TypeKind::Anonymous { fields } => {
                    let field = fields.iter().find(|f| {
                        self.lowering.context.string_interner.get(f.name) == Some(name)
                    })?;
                    let is_method = matches!(
                        self.lowering
                            .context
                            .type_table
                            .borrow()
                            .get(field.type_id)
                            .map(|t| &t.kind),
                        Some(TypeKind::Function { .. })
                    );
                    return Some((false, is_method));
                }
                TypeKind::TypeAlias { target_type, .. } => ty = target_type,
                TypeKind::Optional { inner_type } => ty = inner_type,
                TypeKind::Class { symbol_id, .. } => {
                    let resolved = self
                        .lowering
                        .context
                        .type_table
                        .borrow()
                        .resolve_type_alias(symbol_id);
                    let dynamic = self.lowering.context.type_table.borrow().dynamic_type();
                    if resolved != dynamic && resolved != ty {
                        ty = resolved;
                        continue;
                    }
                    let interned = self.lowering.context.intern_string(name);
                    if self
                        .lowering
                        .resolve_class_method_symbol(symbol_id, interned)
                        .is_some()
                    {
                        return Some((true, true));
                    }
                    return Some((true, false));
                }
                _ => return None,
            }
        }
        None
    }

    fn forwarded_receiver(&mut self, receiver: TypeId, name: &str) -> Option<TypeId> {
        let (symbol_id, underlying) = match self
            .lowering
            .context
            .type_table
            .borrow()
            .get(receiver)
            .map(|t| &t.kind)
        {
            Some(crate::tast::TypeKind::Abstract {
                symbol_id,
                underlying,
                ..
            }) => (*symbol_id, *underlying),
            _ => return None,
        };
        let forwards = self
            .lowering
            .context
            .symbol_table
            .get_symbol(symbol_id)
            .is_some_and(|s| s.flags.contains(crate::tast::SymbolFlags::FORWARD));
        if !forwards {
            return None;
        }
        let interned = self.lowering.context.intern_string(name);
        let declared = self
            .lowering
            .class_fields
            .get(&symbol_id)
            .is_some_and(|fields| fields.iter().any(|(n, _, _)| *n == interned))
            || self
                .lowering
                .resolve_class_method_symbol(symbol_id, interned)
                .is_some();
        if declared {
            return None;
        }
        underlying.or_else(|| {
            self.lowering
                .context
                .type_table
                .borrow()
                .resolve_abstract_underlying(symbol_id)
        })
    }

    fn expected_type(&mut self) -> Option<TypeId> {
        self.expected
    }

    // The imports land on a scope of their own, so they end with it; usings
    // are a list on the lowering, cut back to its length at entry.
    fn enter_imports(&mut self, imports: &[String], usings: &[String]) -> [usize; 2] {
        let token = [
            self.lowering.using_modules.len(),
            self.lowering.unresolved_usings.len(),
        ];
        self.lowering
            .context
            .enter_scope(crate::tast::ScopeKind::Block);
        let span = parser::Span::default();
        for source in imports {
            let _ = self
                .lowering
                .lower_import(&import_from_source(source, span));
        }
        for source in usings {
            let path = source.trim().split('.').map(str::to_string).collect();
            let _ = self.lowering.lower_using(&parser::Using { path, span });
        }
        token
    }

    fn exit_imports(&mut self, token: [usize; 2]) {
        self.lowering.using_modules.truncate(token[0]);
        self.lowering.unresolved_usings.truncate(token[1]);
        self.lowering.context.exit_scope();
    }

    fn type_display(&mut self, id: TypeId) -> String {
        render_type(
            id,
            self.lowering.context.type_table,
            self.lowering.context.symbol_table,
            self.lowering.context.string_interner,
            0,
        )
    }

    fn type_std_string(&mut self, id: TypeId) -> String {
        // `Std.string` on a macro Type value: the constructor form, `TInst(String,[])`.
        render_type_constructor(
            id,
            self.lowering.context.type_table,
            self.lowering.context.symbol_table,
            self.lowering.context.string_interner,
            0,
        )
    }

    fn resolve_type_by_name(&mut self, name: &str) -> Result<TypeId, String> {
        // The canonical resolver is `lower_type` on a parsed annotation: it
        // sees the module's private types, its imports, and the type
        // parameters in scope — everything a hand-rolled name lookup would
        // have to re-implement.
        let src = format!(
            "class __GetType__ {{ static function __g__() {{ var __x:{} = null; }} }}",
            name
        );
        let file = parser::parse_haxe_file("__gettype__", &src, false)
            .map_err(|e| format!("getType: '{}' does not parse as a type: {:?}", name, e))?;
        let annotation = (|| {
            for decl in &file.declarations {
                if let parser::TypeDeclaration::Class(class) = decl {
                    for field in &class.fields {
                        if let parser::ClassFieldKind::Function(func) = &field.kind {
                            let body = func.body.as_deref()?;
                            if let parser::ExprKind::Block(elements) = &body.kind {
                                for element in elements {
                                    if let parser::BlockElement::Expr(e) = element {
                                        if let parser::ExprKind::Var { type_hint, .. } = &e.kind {
                                            return type_hint.clone();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            None
        })()
        .ok_or_else(|| format!("getType: '{}' does not parse as a type", name))?;

        let before = self.lowering.collected_errors.len();
        let ctx_before = self.lowering.context.errors.len();
        let resolved = self.lowering.lower_type(&annotation);
        self.lowering.collected_errors.truncate(before);
        self.lowering.context.errors.truncate(ctx_before);
        let id = resolved.map_err(|e| e.to_compilation_error().message)?;
        Ok(with_unknown_parameters(
            self.lowering.context.type_table,
            id,
        ))
    }

    fn fresh_monomorph(&mut self) -> TypeId {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        // Placeholders intern by kind, so each monomorph carries a unique name
        // to guarantee a distinct TypeId — bindings are keyed by id.
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = self
            .lowering
            .context
            .string_interner
            .intern(&format!("?mono{}", n));
        self.lowering
            .context
            .type_table
            .borrow_mut()
            .create_type_with_location(
                crate::tast::core::TypeKind::Placeholder { name },
                crate::tast::SourceLocation::unknown(),
            )
    }

    fn unify_types(
        &mut self,
        a: TypeId,
        b: TypeId,
        monomorphs: &std::collections::BTreeSet<TypeId>,
        bindings: &mut std::collections::BTreeMap<TypeId, TypeId>,
    ) -> bool {
        let mut variables = monomorphs.clone();
        let mut parameters = Vec::new();
        if self.inferred.is_some() {
            let mut pending: Vec<_> = self.expected.into_iter().collect();
            let mut visited = std::collections::BTreeSet::new();
            while let Some(ty) = pending.pop() {
                if !visited.insert(ty) {
                    continue;
                }
                if matches!(
                    self.lowering
                        .context
                        .type_table
                        .borrow()
                        .get(ty)
                        .map(|ty| &ty.kind),
                    Some(crate::tast::TypeKind::TypeParameter { .. })
                ) {
                    variables.insert(ty);
                    parameters.push(ty);
                }
                pending.extend(self.type_children(ty));
            }
        }
        let compatible = unify(
            a,
            b,
            self.lowering.context.type_table,
            &self.lowering.abstract_casts,
            &variables,
            bindings,
            0,
        );
        if compatible && let Some(inferred) = &mut self.inferred {
            let table = self.lowering.context.type_table.borrow();
            for parameter in parameters {
                let bound = resolve_bound(bindings, parameter);
                if bound != parameter
                    && let Some(crate::tast::TypeKind::TypeParameter { symbol_id, .. }) =
                        table.get(parameter).map(|ty| &ty.kind)
                {
                    inferred.insert(*symbol_id, bound);
                }
            }
        }
        compatible
    }

    fn type_adt_view(
        &mut self,
        id: TypeId,
    ) -> Option<(String, Vec<crate::macro_system::value::MacroValue>)> {
        use crate::macro_system::value::MacroValue as V;
        use crate::tast::core::TypeKind;
        use std::sync::Arc;
        let kind = self
            .lowering
            .context
            .type_table
            .borrow()
            .get(id)
            .map(|t| t.kind.clone())?;
        // The def payload slot carries the type's OWN id as the handle; a
        // reconstruction reads its symbol back out. Primitives present as the
        // abstracts haxe models them as.
        let args_of = |args: &[TypeId]| {
            V::Array(std::sync::Arc::new(
                args.iter().map(|&a| V::Type(a)).collect::<Vec<_>>(),
            ))
        };
        Some(match kind {
            TypeKind::TypeAlias { type_args, .. } => {
                ("TType".to_string(), vec![V::Type(id), args_of(&type_args)])
            }
            TypeKind::Class {
                symbol_id,
                type_args,
            }
            | TypeKind::Interface {
                symbol_id,
                type_args,
            } => {
                // A forward reference to a typedef (or abstract) carries a
                // provisional Class kind until its declaration lowers; the
                // symbol table already knows what it really is, so IT names
                // the constructor.
                let ctor = match self
                    .lowering
                    .context
                    .symbol_table
                    .get_symbol(symbol_id)
                    .map(|sym| sym.kind)
                {
                    Some(crate::tast::symbols::SymbolKind::TypeAlias) => "TType",
                    Some(crate::tast::symbols::SymbolKind::Abstract) => "TAbstract",
                    Some(crate::tast::symbols::SymbolKind::Enum) => "TEnum",
                    _ => "TInst",
                };
                (ctor.to_string(), vec![V::Type(id), args_of(&type_args)])
            }
            TypeKind::Enum { type_args, .. } => {
                ("TEnum".to_string(), vec![V::Type(id), args_of(&type_args)])
            }
            TypeKind::Abstract { type_args, .. } => (
                "TAbstract".to_string(),
                vec![V::Type(id), args_of(&type_args)],
            ),
            TypeKind::Optional { inner_type } => (
                "TAbstract".to_string(),
                vec![V::Type(id), args_of(&[inner_type])],
            ),
            TypeKind::String => ("TInst".to_string(), vec![V::Type(id), args_of(&[])]),
            TypeKind::Array { element_type } => (
                "TInst".to_string(),
                vec![V::Type(id), args_of(&[element_type])],
            ),
            TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::Void | TypeKind::Char => {
                ("TAbstract".to_string(), vec![V::Type(id), args_of(&[])])
            }
            TypeKind::Function {
                params,
                return_type,
                labels,
                ..
            } => (
                "TFun".to_string(),
                vec![
                    V::Array(Arc::new(
                        params
                            .iter()
                            .enumerate()
                            .map(|(index, &ty)| {
                                let label = labels.get(index).copied().unwrap_or_default();
                                let name = label
                                    .name
                                    .and_then(|name| {
                                        self.lowering.context.string_interner.get(name)
                                    })
                                    .unwrap_or("");
                                V::Object(Arc::new(std::collections::BTreeMap::from([
                                    ("name".to_string(), V::String(Arc::from(name))),
                                    ("opt".to_string(), V::Bool(label.optional)),
                                    ("t".to_string(), V::Type(ty)),
                                ])))
                            })
                            .collect(),
                    )),
                    V::Type(return_type),
                ],
            ),
            TypeKind::Dynamic => ("TDynamic".to_string(), vec![V::Null]),
            TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } if type_args.len() == 1
                && matches!(
                    self.lowering
                        .context
                        .type_table
                        .borrow()
                        .get(base_type)
                        .map(|t| &t.kind),
                    Some(TypeKind::Dynamic)
                ) =>
            {
                ("TDynamic".to_string(), vec![V::Type(type_args[0])])
            }
            TypeKind::Anonymous { .. } => ("TAnonymous".to_string(), vec![V::Type(id)]),
            TypeKind::Placeholder { .. } | TypeKind::Unknown => {
                ("TMono".to_string(), vec![V::Null])
            }
            _ => return None,
        })
    }

    fn type_ref_view(&mut self, id: TypeId) -> Option<crate::macro_system::value::MacroValue> {
        use crate::macro_system::value::MacroValue as V;
        use crate::tast::core::TypeKind;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let kind = self
            .lowering
            .context
            .type_table
            .borrow()
            .get(id)?
            .kind
            .clone();
        if matches!(kind, TypeKind::Placeholder { .. } | TypeKind::Unknown) {
            return Some(V::Null);
        }
        let (symbol, underlying, builtin) = match &kind {
            TypeKind::Class { symbol_id, .. }
            | TypeKind::Interface { symbol_id, .. }
            | TypeKind::Enum { symbol_id, .. } => (Some(*symbol_id), None, None),
            TypeKind::Abstract {
                symbol_id,
                underlying,
                ..
            } => (Some(*symbol_id), *underlying, None),
            TypeKind::TypeAlias {
                symbol_id,
                target_type,
                ..
            } => (Some(*symbol_id), Some(*target_type), None),
            TypeKind::Optional { inner_type } => (None, Some(*inner_type), Some("Null")),
            TypeKind::Int => (None, Some(id), Some("Int")),
            TypeKind::Float => (None, Some(id), Some("Float")),
            TypeKind::Bool => (None, Some(id), Some("Bool")),
            TypeKind::Void => (None, Some(id), Some("Void")),
            TypeKind::String => (None, None, Some("String")),
            TypeKind::Array { .. } => (None, None, Some("Array")),
            TypeKind::Anonymous { .. } => (None, None, Some("")),
            _ => return None,
        };
        let string = |s: &str| V::String(Arc::from(s));
        let array = |values: Vec<V>| V::Array(Arc::new(values));
        let object = |fields: BTreeMap<String, V>| V::Object(Arc::new(fields));
        let reference = |value: V| object(BTreeMap::from([("__ref__".to_string(), value)]));
        let qualified = symbol
            .and_then(|s| self.lowering.context.symbol_table.get_symbol(s))
            .and_then(|s| {
                self.lowering
                    .context
                    .string_interner
                    .get(s.qualified_name.unwrap_or(s.name))
            })
            .unwrap_or(builtin.unwrap_or(""));
        let mut parts: Vec<&str> = qualified.split('.').collect();
        let name = parts.pop().unwrap_or("");
        let mut view = BTreeMap::from([
            ("name".to_string(), string(name)),
            (
                "pack".to_string(),
                array(parts.iter().map(|p| string(p)).collect()),
            ),
            ("module".to_string(), string(qualified)),
            (
                "isInterface".to_string(),
                V::Bool(matches!(kind, TypeKind::Interface { .. })),
            ),
        ]);
        let mut metadata = Vec::new();
        if matches!(
            kind,
            TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::Void
        ) {
            for name in [":coreType", ":notNull"] {
                metadata.push(object(BTreeMap::from([("name".to_string(), string(name))])));
            }
        } else if symbol
            .and_then(|s| self.lowering.context.symbol_table.get_symbol(s))
            .is_some_and(|s| {
                s.flags
                    .contains(crate::tast::symbols::SymbolFlags::NOT_NULL)
            })
        {
            metadata.push(object(BTreeMap::from([(
                "name".to_string(),
                string(":notNull"),
            )])));
        }
        view.insert(
            "meta".to_string(),
            object(BTreeMap::from([("__meta__".to_string(), array(metadata))])),
        );
        let params = symbol
            .map(|s| self.definition_parameters(s))
            .unwrap_or_default();
        view.insert(
            "params".to_string(),
            array(
                params
                    .iter()
                    .map(|&t| {
                        object(BTreeMap::from([
                            ("name".to_string(), string(&self.type_display(t))),
                            ("t".to_string(), V::Type(t)),
                        ]))
                    })
                    .collect(),
            ),
        );
        let underlying = symbol
            .and_then(|s| self.definition_target(s))
            .or(underlying);
        view.insert(
            "type".to_string(),
            underlying.map(V::Type).unwrap_or(V::Null),
        );
        let mut fields = Vec::new();
        let mut statics = Vec::new();
        if let Some(symbol) = symbol {
            let members = self
                .lowering
                .class_fields
                .get(&symbol)
                .cloned()
                .unwrap_or_default();
            for (name, member, is_static) in members {
                let Some(sym) = self.lowering.context.symbol_table.get_symbol(member) else {
                    continue;
                };
                let type_id = sym.type_id;
                let field_name = self
                    .lowering
                    .context
                    .string_interner
                    .get(name)
                    .unwrap_or("")
                    .to_string();
                let overloads = self
                    .lowering
                    .method_overloads
                    .get(&member)
                    .cloned()
                    .unwrap_or_default();
                let overloads = overloads
                    .iter()
                    .map(|func| self.overload_field_view(&field_name, func))
                    .collect();
                let entry = object(BTreeMap::from([
                    ("name".to_string(), string(&field_name)),
                    ("type".to_string(), V::Type(type_id)),
                    (
                        "meta".to_string(),
                        object(BTreeMap::from([(
                            "__meta__".to_string(),
                            array(Vec::new()),
                        )])),
                    ),
                    ("overloads".to_string(), reference(array(overloads))),
                ]));
                if is_static {
                    statics.push(entry);
                } else {
                    fields.push(entry);
                }
            }
        } else if let TypeKind::Anonymous { fields: members } = &kind {
            fields.extend(members.iter().map(|f| {
                object(BTreeMap::from([
                    (
                        "name".to_string(),
                        string(
                            self.lowering
                                .context
                                .string_interner
                                .get(f.name)
                                .unwrap_or(""),
                        ),
                    ),
                    ("type".to_string(), V::Type(f.type_id)),
                ]))
            }));
        }
        let fields = array(fields);
        view.insert(
            "fields".to_string(),
            if matches!(kind, TypeKind::Anonymous { .. }) {
                fields
            } else {
                reference(fields)
            },
        );
        view.insert("statics".to_string(), reference(array(statics)));
        if let (TypeKind::Abstract { .. }, Some(symbol)) = (&kind, symbol) {
            let unops = self
                .lowering
                .abstract_unops
                .get(&symbol)
                .cloned()
                .unwrap_or_default();
            let unops = unops
                .iter()
                .map(|(op, method)| {
                    let (op, postfix) = crate::macro_system::expr_adt::unop_name(*op);
                    object(BTreeMap::from([
                        (
                            "op".to_string(),
                            V::Enum(Arc::from("Unop"), Arc::from(op), Arc::new(Vec::new())),
                        ),
                        ("postFix".to_string(), V::Bool(postfix)),
                        (
                            "field".to_string(),
                            object(BTreeMap::from([(
                                "name".to_string(),
                                string(method.as_str()),
                            )])),
                        ),
                    ]))
                })
                .collect();
            view.insert("unops".to_string(), array(unops));
        }
        Some(object(view))
    }

    fn follow_type(&mut self, id: TypeId, once: bool, abstracts: bool) -> TypeId {
        use crate::tast::core::TypeKind;
        let mut current = id;
        let mut seen = std::collections::BTreeSet::new();
        while seen.insert(current) {
            let kind = self
                .lowering
                .context
                .type_table
                .borrow()
                .get(current)
                .map(|t| t.kind.clone());
            let (next, parameterization) = match kind {
                Some(TypeKind::TypeAlias {
                    target_type,
                    symbol_id,
                    type_args,
                }) => (Some(target_type), Some((symbol_id, type_args))),
                Some(TypeKind::Optional { inner_type }) => (Some(inner_type), None),
                Some(TypeKind::Class {
                    symbol_id,
                    type_args,
                }) => {
                    let next = self
                        .lowering
                        .context
                        .symbol_table
                        .get_symbol(symbol_id)
                        .and_then(|s| match s.kind {
                            crate::tast::symbols::SymbolKind::TypeAlias => {
                                self.definition_target(symbol_id)
                            }
                            crate::tast::symbols::SymbolKind::Abstract if abstracts => {
                                self.definition_target(symbol_id)
                            }
                            _ => None,
                        });
                    (next, Some((symbol_id, type_args)))
                }
                Some(TypeKind::Abstract {
                    underlying,
                    symbol_id,
                    type_args,
                }) if abstracts => (
                    self.definition_target(symbol_id).or(underlying),
                    Some((symbol_id, type_args)),
                ),
                _ => (None, None),
            };
            let Some(mut next) = next else { break };
            if let Some((symbol, args)) = parameterization {
                let params = self.definition_parameters(symbol);
                next = self.apply_type_parameters(next, &params, &args);
            }
            current = next;
            if once {
                break;
            }
        }
        current
    }

    fn extension_types(&mut self, id: TypeId) -> Option<Vec<TypeId>> {
        use crate::tast::core::TypeKind;
        let mut current = id;
        let mut seen = std::collections::BTreeSet::new();
        while seen.insert(current) {
            let kind = self
                .lowering
                .context
                .type_table
                .borrow()
                .get(current)
                .map(|t| t.kind.clone());
            let symbol = match kind {
                Some(TypeKind::TypeAlias { symbol_id, .. } | TypeKind::Class { symbol_id, .. }) => {
                    symbol_id
                }
                _ => return None,
            };
            if let Some(extended) = self.lowering.alias_extensions.get(&symbol) {
                return Some(extended.clone());
            }
            current = self.follow_type(current, true, false);
        }
        None
    }

    fn apply_type_parameters(&mut self, id: TypeId, params: &[TypeId], args: &[TypeId]) -> TypeId {
        let bindings: Vec<_> = params
            .iter()
            .zip(args)
            .filter_map(|(param, &arg)| {
                let table = self.lowering.context.type_table.borrow();
                match table.get(*param).map(|t| &t.kind) {
                    Some(crate::tast::core::TypeKind::TypeParameter { symbol_id, .. }) => {
                        Some((*symbol_id, arg))
                    }
                    _ => None,
                }
            })
            .collect();
        self.lowering.substitute_type_bindings(id, &bindings, false)
    }

    fn type_children(&mut self, id: TypeId) -> Vec<TypeId> {
        use crate::tast::TypeKind;
        let table = self.lowering.context.type_table.borrow();
        match table.get(id).map(|t| &t.kind) {
            Some(
                TypeKind::Class { type_args, .. }
                | TypeKind::Interface { type_args, .. }
                | TypeKind::Enum { type_args, .. }
                | TypeKind::Abstract { type_args, .. }
                | TypeKind::TypeAlias { type_args, .. }
                | TypeKind::GenericInstance { type_args, .. },
            ) => type_args.clone(),
            Some(TypeKind::Array { element_type }) => vec![*element_type],
            Some(TypeKind::Optional { inner_type }) => vec![*inner_type],
            Some(TypeKind::Map {
                key_type,
                value_type,
            }) => vec![*key_type, *value_type],
            Some(TypeKind::Function {
                params,
                return_type,
                ..
            }) => params
                .iter()
                .copied()
                .chain(std::iter::once(*return_type))
                .collect(),
            Some(TypeKind::Anonymous { fields }) => fields.iter().map(|f| f.type_id).collect(),
            _ => Vec::new(),
        }
    }

    fn rebuild_type(&mut self, id: TypeId, children: &[TypeId]) -> TypeId {
        use crate::tast::TypeKind;
        if self.type_children(id) == children {
            return id;
        }
        let Some(mut kind) = self
            .lowering
            .context
            .type_table
            .borrow()
            .get(id)
            .map(|t| t.kind.clone())
        else {
            return id;
        };
        let mut children = children.iter().copied();
        match &mut kind {
            TypeKind::Class { type_args, .. }
            | TypeKind::Interface { type_args, .. }
            | TypeKind::Enum { type_args, .. }
            | TypeKind::Abstract { type_args, .. }
            | TypeKind::TypeAlias { type_args, .. }
            | TypeKind::GenericInstance { type_args, .. } => *type_args = children.collect(),
            TypeKind::Array { element_type } => {
                *element_type = children.next().unwrap_or(*element_type)
            }
            TypeKind::Optional { inner_type } => {
                *inner_type = children.next().unwrap_or(*inner_type)
            }
            TypeKind::Map {
                key_type,
                value_type,
            } => {
                *key_type = children.next().unwrap_or(*key_type);
                *value_type = children.next().unwrap_or(*value_type);
            }
            TypeKind::Function {
                params,
                return_type,
                ..
            } => {
                for param in params {
                    *param = children.next().unwrap_or(*param);
                }
                *return_type = children.next().unwrap_or(*return_type);
            }
            TypeKind::Anonymous { fields } => {
                for field in fields {
                    field.type_id = children.next().unwrap_or(field.type_id);
                }
            }
            _ => return id,
        }
        self.lowering
            .context
            .type_table
            .borrow_mut()
            .create_type(kind)
    }

    fn function_type(
        &mut self,
        arguments: Vec<(String, bool, TypeId)>,
        return_type: TypeId,
    ) -> Option<TypeId> {
        let labels = arguments
            .iter()
            .map(|(name, optional, _)| crate::tast::core::FunctionArgLabel {
                name: (!name.is_empty()).then(|| self.lowering.context.intern_string(name)),
                optional: *optional,
            })
            .collect();
        Some(
            self.lowering
                .context
                .type_table
                .borrow_mut()
                .create_labelled_function_type(
                    arguments.into_iter().map(|(_, _, ty)| ty).collect(),
                    return_type,
                    Default::default(),
                    labels,
                ),
        )
    }

    fn instantiate_alias(&mut self, def: TypeId, args: Vec<TypeId>) -> Option<TypeId> {
        use crate::tast::core::TypeKind;
        // Same named type, fresh arguments — whatever kind the def currently
        // carries. A typedef still wearing its provisional Class kind (its
        // declaration has not lowered yet) reconstructs as that same shape,
        // which is what its other mentions in the file lowered to as well, so
        // unification's same-symbol rule still lines the two up.
        let kind = self
            .lowering
            .context
            .type_table
            .borrow()
            .get(def)
            .map(|t| t.kind.clone())?;
        let rebuilt = match kind {
            TypeKind::TypeAlias {
                symbol_id,
                target_type,
                ..
            } => TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args: args,
            },
            TypeKind::Class { symbol_id, .. } => TypeKind::Class {
                symbol_id,
                type_args: args,
            },
            TypeKind::Interface { symbol_id, .. } => TypeKind::Interface {
                symbol_id,
                type_args: args,
            },
            TypeKind::Enum { symbol_id, .. } => TypeKind::Enum {
                symbol_id,
                type_args: args,
            },
            TypeKind::Abstract {
                symbol_id,
                underlying,
                ..
            } => TypeKind::Abstract {
                symbol_id,
                underlying,
                type_args: args,
            },
            _ => return None,
        };
        Some(
            self.lowering
                .context
                .type_table
                .borrow_mut()
                .create_type_with_location(rebuilt, crate::tast::SourceLocation::unknown()),
        )
    }
}

/// An import as written after `import`, read the way the parser reads one.
fn import_from_source(source: &str, span: parser::Span) -> parser::Import {
    use parser::ImportMode;
    let mut words = source.split_whitespace();
    let mut path: Vec<String> = words
        .next()
        .unwrap_or("")
        .split('.')
        .map(str::to_string)
        .collect();
    let mode = match (words.next(), words.next()) {
        (Some("as" | "in"), Some(alias)) => ImportMode::Alias(alias.to_string()),
        _ if path.last().is_some_and(|s| s == "*") => {
            path.pop();
            ImportMode::Wildcard
        }
        _ if path.len() >= 2
            && path
                .last()
                .is_some_and(|s| s.starts_with(|c: char| c.is_ascii_lowercase())) =>
        {
            ImportMode::Field(path.pop().unwrap_or_default())
        }
        _ => ImportMode::Normal,
    };
    parser::Import { path, mode, span }
}

/// The local a function literal's body is nothing but a read of:
/// `function(..) return x`, `(..) -> x`.
fn closure_result_name(init: &parser::Expr) -> Option<&str> {
    let mut body = match &init.kind {
        parser::ExprKind::Function(f) => f.body.as_deref()?,
        parser::ExprKind::Arrow { expr, .. } => &**expr,
        _ => return None,
    };
    loop {
        body = match &body.kind {
            parser::ExprKind::Block(elements) => match elements.as_slice() {
                [parser::BlockElement::Expr(e)] => e,
                _ => return None,
            },
            parser::ExprKind::Return(Some(e)) | parser::ExprKind::Paren(e) => &**e,
            parser::ExprKind::Ident(name) => return Some(name),
            _ => return None,
        };
    }
}

/// Follow monomorph bindings to the representative.
fn resolve_bound(bindings: &std::collections::BTreeMap<TypeId, TypeId>, mut id: TypeId) -> TypeId {
    let mut hops = 0;
    while let Some(&next) = bindings.get(&id) {
        id = next;
        hops += 1;
        if hops > 32 {
            break;
        }
    }
    id
}

/// A named type's (symbol, args) view, through GenericInstance indirection.
fn named_shape(
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    id: TypeId,
) -> Option<(crate::tast::SymbolId, Vec<TypeId>)> {
    use crate::tast::core::TypeKind;
    let kind = type_table.borrow().get(id).map(|t| t.kind.clone())?;
    match kind {
        TypeKind::Class {
            symbol_id,
            type_args,
        }
        | TypeKind::Interface {
            symbol_id,
            type_args,
        }
        | TypeKind::Enum {
            symbol_id,
            type_args,
        }
        | TypeKind::Abstract {
            symbol_id,
            type_args,
            ..
        }
        | TypeKind::TypeAlias {
            symbol_id,
            type_args,
            ..
        } => Some((symbol_id, type_args)),
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } => named_shape(type_table, base_type).map(|(sym, _)| (sym, type_args)),
        _ => None,
    }
}

/// Structural unification with monomorph binding, the shape `Context.unify`
/// needs: same-symbol named types unify their arguments (BEFORE any alias
/// following, so `B<mono>` against `B<String>` binds rather than both
/// collapsing to the alias target), abstracts reach across their declared
/// casts, Dynamic unifies with everything, and a monomorph binds to whatever
/// faces it.
#[allow(clippy::too_many_arguments)]
fn unify(
    a: TypeId,
    b: TypeId,
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    abstract_casts: &std::collections::BTreeMap<crate::tast::SymbolId, (Vec<TypeId>, Vec<TypeId>)>,
    monomorphs: &std::collections::BTreeSet<TypeId>,
    bindings: &mut std::collections::BTreeMap<TypeId, TypeId>,
    depth: usize,
) -> bool {
    use crate::tast::core::TypeKind;
    if depth > 32 {
        return false;
    }
    let a = resolve_bound(bindings, a);
    let b = resolve_bound(bindings, b);
    if a == b {
        return true;
    }
    if monomorphs.contains(&a) {
        bindings.insert(a, b);
        return true;
    }
    if monomorphs.contains(&b) {
        bindings.insert(b, a);
        return true;
    }

    let kind_of = |id: TypeId| type_table.borrow().get(id).map(|t| t.kind.clone());
    let (Some(ka), Some(kb)) = (kind_of(a), kind_of(b)) else {
        return false;
    };

    // Dynamic, `Dynamic<T>` included, unifies with anything, in either direction.
    let dynamic = |kind: &TypeKind| match kind {
        TypeKind::Dynamic => true,
        TypeKind::GenericInstance { base_type, .. } => {
            matches!(kind_of(*base_type), Some(TypeKind::Dynamic))
        }
        _ => false,
    };
    if dynamic(&ka) || dynamic(&kb) {
        return true;
    }

    // Same named symbol: unify the arguments. This must come before alias
    // following so the alias's own parameters can bind.
    if let (Some((sa, args_a)), Some((sb, args_b))) =
        (named_shape(type_table, a), named_shape(type_table, b))
    {
        if sa == sb && args_a.len() == args_b.len() {
            return args_a.iter().zip(args_b.iter()).all(|(&x, &y)| {
                unify(
                    x,
                    y,
                    type_table,
                    abstract_casts,
                    monomorphs,
                    bindings,
                    depth + 1,
                )
            });
        }
    }

    match (&ka, &kb) {
        (TypeKind::Optional { inner_type: x }, TypeKind::Optional { inner_type: y }) => unify(
            *x,
            *y,
            type_table,
            abstract_casts,
            monomorphs,
            bindings,
            depth + 1,
        ),
        (TypeKind::Array { element_type: x }, TypeKind::Array { element_type: y }) => unify(
            *x,
            *y,
            type_table,
            abstract_casts,
            monomorphs,
            bindings,
            depth + 1,
        ),
        (
            TypeKind::Map {
                key_type: ka_,
                value_type: va,
            },
            TypeKind::Map {
                key_type: kb_,
                value_type: vb,
            },
        ) => {
            unify(
                *ka_,
                *kb_,
                type_table,
                abstract_casts,
                monomorphs,
                bindings,
                depth + 1,
            ) && unify(
                *va,
                *vb,
                type_table,
                abstract_casts,
                monomorphs,
                bindings,
                depth + 1,
            )
        }
        (
            TypeKind::Function {
                params: pa,
                return_type: ra,
                ..
            },
            TypeKind::Function {
                params: pb,
                return_type: rb,
                ..
            },
        ) => {
            pa.len() == pb.len()
                && pa.iter().zip(pb.iter()).all(|(&x, &y)| {
                    unify(
                        x,
                        y,
                        type_table,
                        abstract_casts,
                        monomorphs,
                        bindings,
                        depth + 1,
                    )
                })
                && unify(
                    *ra,
                    *rb,
                    type_table,
                    abstract_casts,
                    monomorphs,
                    bindings,
                    depth + 1,
                )
        }
        _ => {
            // An abstract reaches across its declared casts: `a` unifies with
            // `b` through any of a's `to` types, or any of b's `from` types.
            if let TypeKind::Abstract { symbol_id, .. } = ka {
                if let Some((_, to_types)) = abstract_casts.get(&symbol_id) {
                    if to_types.iter().any(|&t| {
                        unify(
                            t,
                            b,
                            type_table,
                            abstract_casts,
                            monomorphs,
                            bindings,
                            depth + 1,
                        )
                    }) {
                        return true;
                    }
                }
            }
            if let TypeKind::Abstract { symbol_id, .. } = kb {
                if let Some((from_types, _)) = abstract_casts.get(&symbol_id) {
                    if from_types.iter().any(|&t| {
                        unify(
                            a,
                            t,
                            type_table,
                            abstract_casts,
                            monomorphs,
                            bindings,
                            depth + 1,
                        )
                    }) {
                        return true;
                    }
                }
            }
            // A typedef stands for its target once symbol-level matching has
            // had its chance.
            if let TypeKind::TypeAlias { target_type, .. } = ka {
                return unify(
                    target_type,
                    b,
                    type_table,
                    abstract_casts,
                    monomorphs,
                    bindings,
                    depth + 1,
                );
            }
            if let TypeKind::TypeAlias { target_type, .. } = kb {
                return unify(
                    a,
                    target_type,
                    type_table,
                    abstract_casts,
                    monomorphs,
                    bindings,
                    depth + 1,
                );
            }
            false
        }
    }
}

/// A type as `Std.string` prints a macro Type: `TAbstract(Int,[])`,
/// `TInst(Array,[TInst(String,[])])`; forms without one fall back to the
/// display form.
pub(super) fn render_type_constructor(
    id: TypeId,
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    symbol_table: &crate::tast::SymbolTable,
    interner: &crate::tast::StringInterner,
    depth: usize,
) -> String {
    use crate::tast::core::TypeKind;
    if depth > 24 {
        return "...".to_string();
    }
    let kind = match type_table.borrow().get(id) {
        Some(t) => t.kind.clone(),
        None => return "<invalid-type>".to_string(),
    };
    let name_of = |symbol_id| {
        symbol_table
            .display_type_path(symbol_id, interner)
            .unwrap_or_else(|| "<unnamed>".to_string())
    };
    let form = |ctor: &str, path: String, args: &[TypeId]| {
        let args: Vec<String> = args
            .iter()
            .map(|&a| render_type_constructor(a, type_table, symbol_table, interner, depth + 1))
            .collect();
        format!("{ctor}({path},[{}])", args.join(","))
    };
    match kind {
        TypeKind::Void => form("TAbstract", "Void".into(), &[]),
        TypeKind::Int => form("TAbstract", "Int".into(), &[]),
        TypeKind::Float => form("TAbstract", "Float".into(), &[]),
        TypeKind::Bool => form("TAbstract", "Bool".into(), &[]),
        TypeKind::String => form("TInst", "String".into(), &[]),
        TypeKind::Class {
            symbol_id,
            type_args,
        }
        | TypeKind::Interface {
            symbol_id,
            type_args,
        } => form("TInst", name_of(symbol_id), &type_args),
        TypeKind::Enum {
            symbol_id,
            type_args,
        } => form("TEnum", name_of(symbol_id), &type_args),
        TypeKind::Abstract {
            symbol_id,
            type_args,
            ..
        } => form("TAbstract", name_of(symbol_id), &type_args),
        TypeKind::TypeAlias {
            symbol_id,
            type_args,
            ..
        } => form("TType", name_of(symbol_id), &type_args),
        TypeKind::Array { element_type } => form("TInst", "Array".into(), &[element_type]),
        TypeKind::Map {
            key_type,
            value_type,
        } => form("TAbstract", "Map".into(), &[key_type, value_type]),
        TypeKind::Optional { inner_type } => form("TAbstract", "Null".into(), &[inner_type]),
        TypeKind::Dynamic => "TDynamic(null)".to_string(),
        TypeKind::Unknown => "TMono(null)".to_string(),
        TypeKind::TypeParameter { symbol_id, .. } => form("TInst", name_of(symbol_id), &[]),
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } if type_args.len() == 1
            && matches!(
                type_table.borrow().get(base_type).map(|t| &t.kind),
                Some(TypeKind::Dynamic)
            ) =>
        {
            format!(
                "TDynamic({})",
                render_type_constructor(
                    type_args[0],
                    type_table,
                    symbol_table,
                    interner,
                    depth + 1
                )
            )
        }
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } => {
            let base =
                render_type_constructor(base_type, type_table, symbol_table, interner, depth + 1);
            match base.find("(") {
                Some(open) if base.ends_with(",[])") => {
                    let ctor = &base[..open];
                    let path = base[open + 1..base.len() - 4].to_string();
                    form(ctor, path, &type_args)
                }
                _ => base,
            }
        }
        // Without argument labels: declared functions carry none yet, and
        // `typedAs` compares a method's string with an annotation's.
        _ => render_type_as(id, type_table, symbol_table, interner, depth, false),
    }
}

/// The source-level spelling of a type, the way `haxe.macro.TypeTools.toString`
/// prints it: named types by name, `Null<T>` written out, functions arrow-form.
/// The general error formatter is not reused here because it spells `Null<T>`
/// as `T?` and Debug-dumps named types — both visible to tests that compare
/// the string against a literal.
pub(super) fn render_type(
    id: TypeId,
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    symbol_table: &crate::tast::SymbolTable,
    interner: &crate::tast::StringInterner,
    depth: usize,
) -> String {
    render_type_as(id, type_table, symbol_table, interner, depth, true)
}

/// `render_type`, with function-type argument names and `?` flags printed
/// when `labels` is set.
fn render_type_as(
    id: TypeId,
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    symbol_table: &crate::tast::SymbolTable,
    interner: &crate::tast::StringInterner,
    depth: usize,
    labels: bool,
) -> String {
    use crate::tast::core::TypeKind;
    if depth > 24 {
        return "...".to_string();
    }
    let kind = match type_table.borrow().get(id) {
        Some(t) => t.kind.clone(),
        None => return "<invalid-type>".to_string(),
    };
    // A type in a package prints with its path, as TypeTools.toString does.
    let name_of = |symbol_id| {
        symbol_table
            .display_type_path(symbol_id, interner)
            .unwrap_or_else(|| "<unnamed>".to_string())
    };
    let with_args = |base: String, args: &[TypeId]| {
        if args.is_empty() {
            base
        } else {
            let rendered: Vec<String> = args
                .iter()
                .map(|&a| render_type_as(a, type_table, symbol_table, interner, depth + 1, labels))
                .collect();
            format!("{}<{}>", base, rendered.join(", "))
        }
    };
    match kind {
        TypeKind::Void => "Void".to_string(),
        TypeKind::Bool => "Bool".to_string(),
        TypeKind::Int => "Int".to_string(),
        TypeKind::Float => "Float".to_string(),
        TypeKind::String => "String".to_string(),
        TypeKind::Char => "Char".to_string(),
        TypeKind::Dynamic => "Dynamic".to_string(),
        TypeKind::Unknown => "Unknown<0>".to_string(),
        TypeKind::Error => "<error>".to_string(),
        TypeKind::ConstArgument { value } => interner.get(value).unwrap_or("?").to_string(),
        TypeKind::Class {
            symbol_id,
            type_args,
        }
        | TypeKind::Interface {
            symbol_id,
            type_args,
        }
        | TypeKind::Enum {
            symbol_id,
            type_args,
        } => with_args(name_of(symbol_id), &type_args),
        TypeKind::Abstract {
            symbol_id,
            type_args,
            ..
        } => with_args(name_of(symbol_id), &type_args),
        TypeKind::TypeAlias {
            symbol_id,
            type_args,
            ..
        } => with_args(name_of(symbol_id), &type_args),
        TypeKind::Function {
            params,
            return_type,
            labels: arg_labels,
            ..
        } => {
            // Haxe's `s_type`: a lone unnamed argument goes without
            // parentheses; a function-typed part, and `Void` as an argument,
            // inside them.
            let part = |t: TypeId, argument: bool| {
                let text = render_type_as(t, type_table, symbol_table, interner, depth + 1, labels);
                let wrap = match type_table.borrow().get(t).map(|t| &t.kind) {
                    Some(TypeKind::Function { .. }) => true,
                    Some(TypeKind::Void) => argument,
                    _ => false,
                };
                if wrap { format!("({text})") } else { text }
            };
            let mut named = false;
            let args: Vec<String> = params
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let label = arg_labels
                        .get(i)
                        .copied()
                        .filter(|_| labels)
                        .unwrap_or_default();
                    // The `Null<T>` a `?` wrapped the argument in prints as the `?`.
                    let p = match type_table.borrow().get(p).map(|t| &t.kind) {
                        Some(TypeKind::Optional { inner_type }) if label.optional => *inner_type,
                        _ => p,
                    };
                    let opt = if label.optional { "?" } else { "" };
                    match label.name.and_then(|n| interner.get(n)) {
                        Some(name) => {
                            named = true;
                            format!("{opt}{name} : {}", part(p, true))
                        }
                        None => format!("{opt}{}", part(p, true)),
                    }
                })
                .collect();
            let ret = part(return_type, false);
            match args.as_slice() {
                [] => format!("() -> {ret}"),
                [only] if !named => format!("{only} -> {ret}"),
                _ => format!("({}) -> {ret}", args.join(", ")),
            }
        }
        TypeKind::Array { element_type } => format!(
            "Array<{}>",
            render_type_as(
                element_type,
                type_table,
                symbol_table,
                interner,
                depth + 1,
                labels
            )
        ),
        TypeKind::Map {
            key_type,
            value_type,
        } => format!(
            "Map<{}, {}>",
            render_type_as(
                key_type,
                type_table,
                symbol_table,
                interner,
                depth + 1,
                labels
            ),
            render_type_as(
                value_type,
                type_table,
                symbol_table,
                interner,
                depth + 1,
                labels
            )
        ),
        TypeKind::Optional { inner_type } => format!(
            "Null<{}>",
            render_type_as(
                inner_type,
                type_table,
                symbol_table,
                interner,
                depth + 1,
                labels
            )
        ),
        TypeKind::Placeholder { name } => interner
            .get(name)
            .map(|n| n.to_string())
            .unwrap_or_else(|| "<placeholder>".to_string()),
        TypeKind::TypeParameter { symbol_id, .. } => name_of(symbol_id),
        TypeKind::Anonymous { fields } => {
            // haxe spells a structure `{ a : Int, b : String }`, fields in
            // declaration order.
            let rendered: Vec<String> = fields
                .iter()
                .map(|f| {
                    let name = interner
                        .get(f.name)
                        .map(|n| n.to_string())
                        .unwrap_or_else(|| "<field>".to_string());
                    format!(
                        "{} : {}",
                        name,
                        render_type_as(
                            f.type_id,
                            type_table,
                            symbol_table,
                            interner,
                            depth + 1,
                            labels
                        )
                    )
                })
                .collect();
            if rendered.is_empty() {
                "{ }".to_string()
            } else {
                format!("{{ {} }}", rendered.join(", "))
            }
        }
        TypeKind::GenericInstance {
            base_type,
            type_args,
            ..
        } => with_args(
            render_type_as(
                base_type,
                type_table,
                symbol_table,
                interner,
                depth + 1,
                labels,
            ),
            &type_args,
        ),
        other => format!("{:?}", other),
    }
}

/// A generic type named without arguments has a fresh unknown for each
/// of its own parameters, as `getType("haxe.macro.ExprOf")` gives in Haxe.
fn with_unknown_parameters(
    type_table: &std::cell::RefCell<crate::tast::TypeTable>,
    id: TypeId,
) -> TypeId {
    use crate::tast::TypeKind;
    let mut tt = type_table.borrow_mut();
    let Some(kind) = tt.get(id).map(|t| t.kind.clone()) else {
        return id;
    };
    let is_param = |tt: &crate::tast::TypeTable, a: &TypeId| {
        matches!(
            tt.get(*a).map(|t| &t.kind),
            Some(TypeKind::TypeParameter { .. })
        )
    };
    let unknown = tt.unknown_type();
    let fresh = |args: &[TypeId]| args.iter().map(|_| unknown).collect::<Vec<_>>();
    let kind = match kind {
        TypeKind::Class {
            symbol_id,
            type_args,
        } if !type_args.is_empty() && type_args.iter().all(|a| is_param(&tt, a)) => {
            TypeKind::Class {
                symbol_id,
                type_args: fresh(&type_args),
            }
        }
        TypeKind::TypeAlias {
            symbol_id,
            target_type,
            type_args,
        } if !type_args.is_empty() && type_args.iter().all(|a| is_param(&tt, a)) => {
            TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args: fresh(&type_args),
            }
        }
        _ => return id,
    };
    tt.create_type(kind)
}

/// How control leaves an expression, as Haxe's `DeadEnd` judges it: it
/// completes, it exits (`return`/`throw`) on every path, or a `break` or
/// `continue` is reached first.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Completes,
    Exits,
    Escapes,
}

/// Sub-expressions are judged in evaluation order and the first that does
/// not complete decides; a function literal is never run; a branch, case or
/// catch is dead only when every alternative is.
fn dead_end(e: &parser::Expr) -> Flow {
    use parser::{BinaryOp, ExprKind as K};
    fn is_bool(mut e: &parser::Expr, value: bool) -> bool {
        while let K::Paren(inner) = &e.kind {
            e = &**inner;
        }
        matches!(e.kind, K::Bool(b) if b == value)
    }
    // Every alternative must be dead; the first that is not decides.
    fn all<'e>(alternatives: impl IntoIterator<Item = &'e parser::Expr>) -> Flow {
        for alternative in alternatives {
            match dead_end(alternative) {
                Flow::Exits => {}
                other => return other,
            }
        }
        Flow::Exits
    }
    // A `break` or `continue` in a loop body leaves the loop, not the code.
    let body = |b: &parser::Expr| match dead_end(b) {
        Flow::Escapes => Flow::Completes,
        flow => flow,
    };
    let first = |parts: &[&parser::Expr]| {
        parts
            .iter()
            .map(|&p| dead_end(p))
            .find(|f| *f != Flow::Completes)
            .unwrap_or(Flow::Completes)
    };
    match &e.kind {
        K::Break | K::Continue => Flow::Escapes,
        K::Return(_) | K::Throw(_) => Flow::Exits,
        K::Function(_) | K::Arrow { .. } => Flow::Completes,
        K::If {
            cond,
            then_branch,
            else_branch: Some(else_branch),
        }
        | K::Ternary {
            cond,
            then_expr: then_branch,
            else_expr: else_branch,
        } => match dead_end(cond) {
            Flow::Completes => all([&**then_branch, &**else_branch]),
            flow => flow,
        },
        K::If { cond, .. } | K::For { iter: cond, .. } => dead_end(cond),
        K::Binary {
            op: op @ (BinaryOp::And | BinaryOp::Or),
            left,
            right,
        } => match dead_end(left) {
            Flow::Completes if is_bool(left, matches!(op, BinaryOp::And)) => dead_end(right),
            flow => flow,
        },
        K::Binary {
            op: BinaryOp::NullCoal,
            left,
            ..
        } => dead_end(left),
        K::While { cond, body: b } => match dead_end(cond) {
            Flow::Completes if is_bool(cond, true) => body(b),
            flow => flow,
        },
        K::DoWhile { body: b, cond } => match body(b) {
            Flow::Completes => dead_end(cond),
            flow => flow,
        },
        K::Switch {
            expr: subject,
            cases,
            default,
        } => {
            let subject = dead_end(subject);
            let exhaustive = default.is_some()
                || cases.iter().any(|c| {
                    c.guard.is_none()
                        && c.patterns
                            .iter()
                            .any(|p| matches!(p, parser::Pattern::Underscore))
                });
            match subject {
                Flow::Completes if exhaustive => {
                    all(cases.iter().map(|c| &c.body).chain(default.as_deref()))
                }
                flow => flow,
            }
        }
        K::Try {
            expr: tried,
            catches,
            ..
        } => match dead_end(tried) {
            Flow::Exits => all(catches.iter().map(|c| &c.body)),
            flow => flow,
        },
        _ => {
            let mut children = Vec::new();
            let mut root = true;
            super::walk_expr_pruned(e, &mut |child| {
                if std::mem::take(&mut root) {
                    return true;
                }
                children.push(child);
                false
            });
            first(&children)
        }
    }
}
