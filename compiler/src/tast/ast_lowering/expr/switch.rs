//! `switch` cases and their extra values.

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
    /// The wildcard-plus-guard case an extractor becomes.
    fn build_extractor_case(
        &mut self,
        case: &parser::Case,
        extractor_guard: TypedExpression,
        bindings: Vec<(String, parser::Expr)>,
        as_expression: bool,
    ) -> Result<TypedSwitchCase, LoweringError> {
        let case_scope = self
            .context
            .scope_tree
            .create_scope(Some(self.context.current_scope));
        let prev_scope = self.context.current_scope;
        self.context.current_scope = case_scope;

        // Names the value side binds are declared in the case's own scope,
        // ahead of the body that reads them.
        let mut prelude = Vec::with_capacity(bindings.len());
        for (name, accessor) in &bindings {
            let value = self.lower_expression(accessor)?;
            let interned = self.context.intern_string(name);
            let symbol_id = self
                .context
                .symbol_table
                .create_variable_in_scope(interned, case_scope);
            if let Some(scope) = self.context.scope_tree.get_scope_mut(case_scope) {
                scope.add_symbol(symbol_id, interned);
            }
            self.context
                .symbol_table
                .update_symbol_type(symbol_id, value.expr_type);
            prelude.push(TypedStatement::VarDeclaration {
                symbol_id,
                var_type: value.expr_type,
                initializer: Some(value),
                mutability: crate::tast::symbols::Mutability::Immutable,
                source_location: self.context.span_to_location(&case.span),
            });
        }

        // A `case` guard of its own still applies, on top of the extraction.
        let guard = match case.guard.as_ref() {
            Some(own) => {
                let own = self.lower_expression(own)?;
                let ty = self.context.type_table.borrow().bool_type();
                Some(TypedExpression {
                    kind: TypedExpressionKind::BinaryOp {
                        left: Box::new(extractor_guard),
                        operator: BinaryOperator::And,
                        right: Box::new(own),
                    },
                    expr_type: ty,
                    usage: VariableUsage::Borrow,
                    lifetime_id: LifetimeId::from_raw(1),
                    source_location: self.context.span_to_location(&case.span),
                    metadata: ExpressionMetadata::default(),
                })
            }
            None => Some(extractor_guard),
        };

        let body_expr = self.lower_expression(&case.body)?;
        self.context.current_scope = prev_scope;
        // The bindings and the body travel as one EXPRESSION: a switch used
        // as an expression requires every case body to be one.
        let body_expr = if prelude.is_empty() {
            body_expr
        } else {
            let body_type = body_expr.expr_type;
            let mut statements = prelude;
            statements.push(TypedStatement::Expression {
                expression: body_expr,
                source_location: self.context.span_to_location(&case.span),
            });
            TypedExpression {
                kind: TypedExpressionKind::Block {
                    statements,
                    scope_id: case_scope,
                },
                expr_type: body_type,
                usage: VariableUsage::Borrow,
                lifetime_id: LifetimeId::from_raw(1),
                source_location: self.context.span_to_location(&case.span),
                metadata: ExpressionMetadata::default(),
            }
        };
        let body = TypedStatement::Expression {
            expression: body_expr,
            source_location: self.context.span_to_location(&case.span),
        };
        let _ = as_expression;
        Ok(TypedSwitchCase {
            case_value: TypedExpression {
                kind: TypedExpressionKind::PatternPlaceholder {
                    pattern: parser::Pattern::Underscore,
                    source_location: self.context.span_to_location(&case.span),
                    variable_bindings: Vec::new(),
                },
                expr_type: self.context.type_table.borrow().dynamic_type(),
                usage: VariableUsage::Borrow,
                lifetime_id: LifetimeId::from_raw(1),
                source_location: self.context.span_to_location(&case.span),
                metadata: ExpressionMetadata::default(),
            },
            extra_case_values: Vec::new(),
            guard,
            body,
            source_location: self.context.span_to_location(&case.span),
        })
    }

    /// `_` inside an extractor stands for the value being switched on.
    fn substitute_extractor_placeholder(
        expr: &parser::Expr,
        subject: &parser::Expr,
    ) -> parser::Expr {
        use parser::ExprKind as K;
        let sub = |e: &parser::Expr| Box::new(Self::substitute_extractor_placeholder(e, subject));
        let sub_opt = |e: &Option<Box<parser::Expr>>| {
            e.as_ref()
                .map(|e| Box::new(Self::substitute_extractor_placeholder(e, subject)))
        };
        let replaced = match &expr.kind {
            K::Ident(name) if name == "_" => return subject.clone(),
            K::Binary { left, op, right } => K::Binary {
                left: sub(left),
                op: *op,
                right: sub(right),
            },
            K::Unary { op, expr: inner } => K::Unary {
                op: *op,
                expr: sub(inner),
            },
            K::Paren(inner) => K::Paren(sub(inner)),
            K::Field {
                expr: obj,
                field,
                is_optional,
            } => K::Field {
                expr: sub(obj),
                field: field.clone(),
                is_optional: *is_optional,
            },
            K::Index { expr: obj, index } => K::Index {
                expr: sub(obj),
                index: sub(index),
            },
            K::Call { expr: callee, args } => K::Call {
                expr: sub(callee),
                args: args
                    .iter()
                    .map(|a| Self::substitute_extractor_placeholder(a, subject))
                    .collect(),
            },
            K::Assign { left, op, right } => K::Assign {
                left: sub(left),
                op: *op,
                right: sub(right),
            },
            K::Ternary {
                cond,
                then_expr,
                else_expr,
            } => K::Ternary {
                cond: sub(cond),
                then_expr: sub(then_expr),
                else_expr: sub(else_expr),
            },
            K::Array(elements) => K::Array(
                elements
                    .iter()
                    .map(|e| Self::substitute_extractor_placeholder(e, subject))
                    .collect(),
            ),
            K::Block(elements) => K::Block(
                elements
                    .iter()
                    .map(|element| match element {
                        parser::BlockElement::Expr(e) => parser::BlockElement::Expr(
                            Self::substitute_extractor_placeholder(e, subject),
                        ),
                        other => other.clone(),
                    })
                    .collect(),
            ),
            K::Var {
                name,
                type_hint,
                expr: init,
            } => K::Var {
                name: name.clone(),
                type_hint: type_hint.clone(),
                expr: sub_opt(init),
            },
            K::Return(value) => K::Return(sub_opt(value)),
            K::Throw(value) => K::Throw(sub(value)),
            K::If {
                cond,
                then_branch,
                else_branch,
            } => K::If {
                cond: sub(cond),
                then_branch: sub(then_branch),
                else_branch: sub_opt(else_branch),
            },
            K::While { cond, body } => K::While {
                cond: sub(cond),
                body: sub(body),
            },
            K::DoWhile { body, cond } => K::DoWhile {
                body: sub(body),
                cond: sub(cond),
            },
            K::Try {
                expr: body,
                catches,
                finally_block,
            } => K::Try {
                expr: sub(body),
                catches: catches
                    .iter()
                    .map(|c| parser::Catch {
                        body: Self::substitute_extractor_placeholder(&c.body, subject),
                        ..c.clone()
                    })
                    .collect(),
                finally_block: sub_opt(finally_block),
            },
            K::Cast {
                expr: inner,
                type_hint,
            } => K::Cast {
                expr: sub(inner),
                type_hint: type_hint.clone(),
            },
            K::TypeCheck {
                expr: inner,
                type_hint,
            } => K::TypeCheck {
                expr: sub(inner),
                type_hint: type_hint.clone(),
            },
            _ => return expr.clone(),
        };
        parser::Expr {
            kind: replaced,
            span: expr.span,
        }
    }

    /// Whether a pattern needs the guard desugaring: it binds with `name =`
    /// or runs an extractor somewhere inside.
    fn pattern_needs_guard(pattern: &parser::Pattern) -> bool {
        use parser::Pattern as P;
        match pattern {
            P::Extractor { .. } | P::Bind { .. } => true,
            P::Array(items) | P::Or(items) => items.iter().any(Self::pattern_needs_guard),
            P::ArrayRest { elements, .. } => elements.iter().any(Self::pattern_needs_guard),
            P::Object { fields } => fields.iter().any(|(_, p)| Self::pattern_needs_guard(p)),
            // The ordinary matcher compares an object argument as a literal
            // and takes an array argument for a wildcard.
            P::Constructor { params, .. } => params.iter().any(|p| {
                Self::pattern_needs_guard(p)
                    || matches!(p, P::Object { .. } | P::Array(_))
                    || matches!(p, P::Const(v) if matches!(v.kind, parser::ExprKind::Object(_)))
            }),
            _ => false,
        }
    }

    /// A pattern as a test on `subject` and the names it binds to parts of
    /// `subject`. The test is `None` when the pattern always matches. `None`
    /// overall means the pattern needs the real matcher (an enum
    /// constructor, a type check, a rest element).
    fn pattern_guard_parts(
        &mut self,
        pattern: &parser::Pattern,
        subject: &parser::Expr,
    ) -> Option<(Option<parser::Expr>, Vec<(String, parser::Expr)>)> {
        use parser::Pattern as P;
        let span = subject.span;
        let expr = |kind| parser::Expr { kind, span };
        let eq = |left: &parser::Expr, right: parser::Expr| {
            expr(parser::ExprKind::Binary {
                left: Box::new(left.clone()),
                op: parser::BinaryOp::Eq,
                right: Box::new(right),
            })
        };
        let field = |name: &str| {
            expr(parser::ExprKind::Field {
                expr: Box::new(subject.clone()),
                field: name.to_string(),
                is_optional: false,
            })
        };
        match pattern {
            P::Underscore => Some((None, Vec::new())),
            P::Null => Some((Some(eq(subject, expr(parser::ExprKind::Null))), Vec::new())),
            // `E.A` in a pattern names a value, never a capture.
            P::Var(name) if name.contains('.') => {
                let mut parts: Vec<String> = name.split('.').map(str::to_string).collect();
                let last = parts.pop()?;
                let constructor = P::Constructor {
                    path: parser::TypePath {
                        package: parts,
                        name: last,
                        sub: None,
                    },
                    params: Vec::new(),
                };
                self.pattern_guard_parts(&constructor, subject)
            }
            P::Var(name) => {
                if self.names_enum_variant(name) {
                    let constructor = P::Constructor {
                        path: parser::TypePath {
                            package: Vec::new(),
                            name: name.clone(),
                            sub: None,
                        },
                        params: Vec::new(),
                    };
                    self.pattern_guard_parts(&constructor, subject)
                } else {
                    Some((None, vec![(name.clone(), subject.clone())]))
                }
            }
            P::Const(value) => match &value.kind {
                // An object read as a literal: a name in a field captures,
                // anything else compares. A null never matches it.
                parser::ExprKind::Object(fields) => {
                    let mut tests = vec![Self::not_null(subject)];
                    let mut bindings = Vec::new();
                    for f in fields {
                        let inner = Self::pattern_from_expr(&f.expr)
                            .unwrap_or_else(|| P::Const(f.expr.clone()));
                        let (test, bound) = self.pattern_guard_parts(&inner, &field(&f.name))?;
                        tests.extend(test);
                        bindings.extend(bound);
                    }
                    Some((Self::conjoin(tests, span), bindings))
                }
                _ => Some((Some(eq(subject, value.clone())), Vec::new())),
            },
            P::Object { fields } => {
                let mut tests = vec![Self::not_null(subject)];
                let mut bindings = Vec::new();
                for (name, inner) in fields {
                    let (test, bound) = self.pattern_guard_parts(inner, &field(name))?;
                    tests.extend(test);
                    bindings.extend(bound);
                }
                Some((Self::conjoin(tests, span), bindings))
            }
            // `switch [a, b]` matches each element against its own
            // expression, whatever their types; nothing is indexed.
            P::Array(elements) if matches!(&subject.kind, parser::ExprKind::Array(items) if items.len() == elements.len()) =>
            {
                let parser::ExprKind::Array(items) = &subject.kind else {
                    return None;
                };
                let mut tests = Vec::new();
                let mut bindings = Vec::new();
                for (inner, item) in elements.iter().zip(items) {
                    let (test, bound) = self.pattern_guard_parts(inner, item)?;
                    tests.extend(test);
                    bindings.extend(bound);
                }
                Some((Self::conjoin(tests, span), bindings))
            }
            P::Array(elements) => {
                let length = eq(
                    &field("length"),
                    expr(parser::ExprKind::Int(elements.len() as i64)),
                );
                let mut tests = vec![Self::not_null(subject), length];
                let mut bindings = Vec::new();
                for (index, inner) in elements.iter().enumerate() {
                    let at = expr(parser::ExprKind::Index {
                        expr: Box::new(subject.clone()),
                        index: Box::new(expr(parser::ExprKind::Int(index as i64))),
                    });
                    let (test, bound) = self.pattern_guard_parts(inner, &at)?;
                    tests.extend(test);
                    bindings.extend(bound);
                }
                Some((Self::conjoin(tests, span), bindings))
            }
            // Alternatives that bind would each need their own bindings.
            P::Or(alternatives) => {
                let mut tests = Vec::new();
                for alt in alternatives {
                    let (test, bound) = self.pattern_guard_parts(alt, subject)?;
                    if !bound.is_empty() {
                        return None;
                    }
                    tests.push(test?);
                }
                let any = tests.into_iter().reduce(|a, b| {
                    expr(parser::ExprKind::Binary {
                        left: Box::new(a),
                        op: parser::BinaryOp::Or,
                        right: Box::new(b),
                    })
                });
                Some((any, Vec::new()))
            }
            P::Bind { name, pattern } => {
                let (test, mut bindings) = self.pattern_guard_parts(pattern, subject)?;
                bindings.insert(0, (name.clone(), subject.clone()));
                Some((test, bindings))
            }
            P::Extractor {
                expr: extractor,
                value,
            } => {
                let extracted = expr(parser::ExprKind::Paren(Box::new(
                    Self::substitute_extractor_placeholder(extractor, subject),
                )));
                self.pattern_guard_parts(value, &extracted)
            }
            // The enum test stays with the real matcher: an inner `switch`
            // binds each argument to a fresh name, and the argument patterns
            // are tested (and their names bound) through those.
            P::Constructor { path, params } => {
                static NEXT: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let base = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let fresh: Vec<String> = (0..params.len())
                    .map(|i| format!("__pattern{base}_{i}"))
                    .collect();
                let mut tests = Vec::new();
                let mut inner_bindings = Vec::new();
                for (param, name) in params.iter().zip(&fresh) {
                    let arg = expr(parser::ExprKind::Ident(name.clone()));
                    let (test, bound) = self.pattern_guard_parts(param, &arg)?;
                    tests.extend(test);
                    inner_bindings.extend(bound);
                }
                // `case N1:` is written, and matched, as a bare name, and
                // `case E.A:` as the qualified constant.
                let shape = if params.is_empty() && path.package.is_empty() {
                    P::Var(path.name.clone())
                } else if params.is_empty() {
                    let mut chain = expr(parser::ExprKind::Ident(path.package[0].clone()));
                    for part in path.package[1..].iter().chain(std::iter::once(&path.name)) {
                        chain = expr(parser::ExprKind::Field {
                            expr: Box::new(chain),
                            field: part.clone(),
                            is_optional: false,
                        });
                    }
                    P::Const(chain)
                } else {
                    P::Constructor {
                        path: path.clone(),
                        params: fresh.iter().cloned().map(P::Var).collect(),
                    }
                };
                let within = |body: parser::Expr, otherwise: parser::Expr| {
                    expr(parser::ExprKind::Switch {
                        expr: Box::new(subject.clone()),
                        cases: vec![parser::Case {
                            patterns: vec![shape.clone()],
                            guard: None,
                            body,
                            span,
                        }],
                        default: Some(Box::new(otherwise)),
                    })
                };
                let matched = Self::conjoin(tests, span)
                    .unwrap_or_else(|| expr(parser::ExprKind::Bool(true)));
                let test = within(matched, expr(parser::ExprKind::Bool(false)));
                let bindings = inner_bindings
                    .into_iter()
                    .map(|(name, accessor)| (name, within(accessor, expr(parser::ExprKind::Null))))
                    .collect();
                Some((Some(test), bindings))
            }
            P::ArrayRest { .. } | P::Type { .. } => None,
        }
    }

    fn not_null(subject: &parser::Expr) -> parser::Expr {
        parser::Expr {
            kind: parser::ExprKind::Binary {
                left: Box::new(subject.clone()),
                op: parser::BinaryOp::NotEq,
                right: Box::new(parser::Expr {
                    kind: parser::ExprKind::Null,
                    span: subject.span,
                }),
            },
            span: subject.span,
        }
    }

    fn conjoin(tests: Vec<parser::Expr>, span: parser::Span) -> Option<parser::Expr> {
        tests.into_iter().reduce(|a, b| parser::Expr {
            kind: parser::ExprKind::Binary {
                left: Box::new(a),
                op: parser::BinaryOp::And,
                right: Box::new(b),
            },
            span,
        })
    }

    fn names_enum_variant(&mut self, name: &str) -> bool {
        let interned = self.context.intern_string(name);
        if self
            .resolve_enum_constructor_from_discriminant(interned)
            .is_some()
        {
            return true;
        }
        self.resolve_symbol_in_scope_hierarchy(interned)
            .and_then(|sym| self.context.symbol_table.get_symbol(sym))
            .is_some_and(|sy| sy.kind == crate::tast::symbols::SymbolKind::EnumVariant)
    }

    /// An array or object pattern, or any pattern that binds with `name =`
    /// or runs an extractor, becomes a wildcard case guarded by its test,
    /// with its bindings declared ahead of the body.
    fn extractor_case_guard(
        &mut self,
        pattern: &parser::Pattern,
        subject: &parser::Expr,
    ) -> Option<Result<(TypedExpression, Vec<(String, parser::Expr)>), LoweringError>> {
        let destructures = match pattern {
            parser::Pattern::Array(_) | parser::Pattern::Object { .. } => true,
            parser::Pattern::Const(value) => {
                matches!(&value.kind, parser::ExprKind::Object(fields) if !fields.is_empty())
            }
            _ => false,
        };
        if !destructures && !Self::pattern_needs_guard(pattern) {
            return None;
        }
        let (test, bindings) = self.pattern_guard_parts(pattern, subject)?;
        let test = test.unwrap_or(parser::Expr {
            kind: parser::ExprKind::Bool(true),
            span: subject.span,
        });
        Some(self.lower_expression(&test).map(|g| (g, bindings)))
    }

    /// Lower a switch case
    /// Lower a switch case for expression context (where case body is an expression)
    pub(crate) fn lower_switch_case_expression(
        &mut self,
        case: &parser::Case,
        subject: &parser::Expr,
    ) -> Result<TypedSwitchCase, LoweringError> {
        if let Some(guard) = case
            .patterns
            .first()
            .and_then(|p| self.extractor_case_guard(p, subject))
        {
            let (guard, bindings) = guard?;
            return self.build_extractor_case(case, guard, bindings, true);
        }
        // For switch expressions, the case body should be an expression
        let case_value = if let Some(first_pattern) = case.patterns.first() {
            // Check if this is a complex pattern that requires variable binding
            if self.pattern_has_variables(first_pattern) {
                // Create a new scope for this case to bind pattern variables
                let case_scope = self
                    .context
                    .scope_tree
                    .create_scope(Some(self.context.current_scope));
                let prev_scope = self.context.current_scope;
                self.context.current_scope = case_scope;

                // Bind pattern variables in the new scope. For an or-pattern
                // (`case U8(x)|I8(x)|..`), bind from an alternative whose
                // payloads are all concrete rather than the first: a
                // cross-module-imported enum can leave a leading variant's
                // payload Dynamic while later ones are concrete, and binding
                // `x` from the Dynamic one poisons its type → the
                // switch-as-expression result temp becomes `*void` and the
                // raw payload is spuriously `haxe_unbox_int_ptr`'d at return
                // → SIGSEGV. The case VALUE still matches `first_pattern`.
                let bind_pattern = self.pick_concrete_binding_pattern(case, first_pattern);
                let var_bindings = self.bind_pattern_variables(bind_pattern)?;

                // For constructor patterns, create the constructor expression
                let case_expr =
                    self.create_constructor_expression_with_bindings(first_pattern, var_bindings)?;

                // Lower guard in the new scope (before restoring) so pattern vars are visible
                let guard = case
                    .guard
                    .as_ref()
                    .map(|g| self.lower_expression(g))
                    .transpose()?;

                // Lower case body as expression in the new scope with bound variables
                let body_expr = self.lower_expression(&case.body)?;

                // Restore previous scope
                self.context.current_scope = prev_scope;

                let body = TypedStatement::Expression {
                    expression: body_expr,
                    source_location: self.context.span_to_location(&case.span),
                };

                return Ok(TypedSwitchCase {
                    case_value: case_expr,
                    extra_case_values: Vec::new(),
                    guard,
                    body,
                    source_location: self.context.span_to_location(&case.span),
                });
            } else {
                self.lower_pattern_to_expression(first_pattern)?
            }
        } else {
            return Err(LoweringError::IncompleteImplementation {
                feature: "Empty switch case patterns".to_string(),
                location: self.context.span_to_location(&case.span),
            });
        };

        // Multi-value clause: `case A, B:` — patterns[1..] are
        // alternates that should match the same body. The HIR layer
        // already expresses this via `HirMatchCase.patterns: Vec`;
        // carry them through TAST so we don't drop them.
        let extra_case_values = self.lower_extra_case_values(&case.patterns)?;

        // Lower case body as expression
        let body_expr = self.lower_expression(&case.body)?;

        // Convert to statement for compatibility
        let body = TypedStatement::Expression {
            expression: body_expr,
            source_location: self.context.span_to_location(&case.span),
        };

        Ok(TypedSwitchCase {
            case_value,
            extra_case_values,
            guard: case
                .guard
                .as_ref()
                .map(|g| self.lower_expression(g))
                .transpose()?,
            body,
            source_location: self.context.span_to_location(&case.span),
        })
    }

    pub(crate) fn lower_switch_case(
        &mut self,
        case: &parser::Case,
        subject: &parser::Expr,
    ) -> Result<TypedSwitchCase, LoweringError> {
        if let Some(guard) = case
            .patterns
            .first()
            .and_then(|p| self.extractor_case_guard(p, subject))
        {
            let (guard, bindings) = guard?;
            return self.build_extractor_case(case, guard, bindings, false);
        }
        // For now, use the first pattern as the case value
        // TODO: Handle multiple patterns and guards properly
        let case_value = if let Some(first_pattern) = case.patterns.first() {
            // Check if this is a complex pattern that requires variable binding
            if self.pattern_has_variables(first_pattern) {
                // Create a new scope for this case to bind pattern variables
                let case_scope = self
                    .context
                    .scope_tree
                    .create_scope(Some(self.context.current_scope));
                let prev_scope = self.context.current_scope;
                self.context.current_scope = case_scope;

                // Bind pattern variables in the new scope. For an or-pattern
                // (`case U8(x)|I8(x)|..`), bind from an alternative whose
                // payloads are all concrete rather than the first: a
                // cross-module-imported enum can leave a leading variant's
                // payload Dynamic while later ones are concrete, and binding
                // `x` from the Dynamic one poisons its type → the
                // switch-as-expression result temp becomes `*void` and the
                // raw payload is spuriously `haxe_unbox_int_ptr`'d at return
                // → SIGSEGV. The case VALUE still matches `first_pattern`.
                let bind_pattern = self.pick_concrete_binding_pattern(case, first_pattern);
                let var_bindings = self.bind_pattern_variables(bind_pattern)?;

                // For constructor patterns, create the constructor expression
                let case_expr =
                    self.create_constructor_expression_with_bindings(first_pattern, var_bindings)?;

                // Lower guard in the new scope (before restoring) so pattern vars are visible
                let guard = case
                    .guard
                    .as_ref()
                    .map(|g| self.lower_expression(g))
                    .transpose()?;

                // Lower case body in the new scope with bound variables
                let body = self.lower_expression_to_statement(&case.body)?;

                // Restore previous scope
                self.context.current_scope = prev_scope;

                return Ok(TypedSwitchCase {
                    case_value: case_expr,
                    extra_case_values: Vec::new(),
                    guard,
                    body,
                    source_location: self.context.span_to_location(&case.span),
                });
            } else {
                // Simple patterns can be converted to expressions directly
                self.lower_pattern_to_expression(first_pattern)?
            }
        } else {
            return Err(LoweringError::IncompleteImplementation {
                feature: "Empty switch case patterns".to_string(),
                location: self.context.span_to_location(&case.span),
            });
        };

        // Multi-value clause: see `lower_switch_case_expression`.
        let extra_case_values = self.lower_extra_case_values(&case.patterns)?;

        // Lower case body as statement
        let body = self.lower_expression_to_statement(&case.body)?;

        Ok(TypedSwitchCase {
            case_value,
            extra_case_values,
            guard: case
                .guard
                .as_ref()
                .map(|g| self.lower_expression(g))
                .transpose()?,
            body,
            source_location: self.context.span_to_location(&case.span),
        })
    }

    /// Lower the trailing patterns of a `case A, B, C:` multi-value
    /// clause (patterns 1..). The first pattern is handled separately
    /// because it may involve var binding / constructor expression
    /// synthesis; the extras for now are restricted to *simple*
    /// patterns (literals, named consts, idents that aren't binders).
    /// Complex extras silently fall through to a single-pattern
    /// match — flagged via a TODO so the regression test catches it
    /// if it bites in real code.
    #[allow(clippy::wrong_self_convention)]
    fn lower_extra_case_values(
        &mut self,
        patterns: &[parser::Pattern],
    ) -> Result<Vec<TypedExpression>, LoweringError> {
        // Two shapes produce multi-value alternates:
        //
        // 1. `Case.patterns` has length > 1 — older parser path that
        //    kept `case A, B:` as a Vec.
        // 2. `Case.patterns` has length 1 but the single pattern is
        //    `Pattern::Or(alternatives)` — the rd parser collapses
        //    `case A, B:` into one `Pattern::Or` (see
        //    `parser/src/rd/expr.rs` around line 798).
        //
        // Both shapes feed into the same "match any of these" semantic;
        // the TAST layer flattens them into a single
        // `extra_case_values` list so HIR/MIR don't need to know
        // which parser produced the input. The *first* alternate
        // becomes `case_value` (handled by the caller); we return
        // the remaining alternates here.
        if patterns.is_empty() {
            return Ok(Vec::new());
        }
        let mut alternates: Vec<&parser::Pattern> = Vec::new();
        // Skip the first pattern — it's `case_value` already.
        // BUT if that first pattern is a `Pattern::Or`, its first
        // alt was already used as `case_value`; the rest go into
        // alternates. Then any further `patterns[1..]` (older parser
        // shape) get appended too.
        if let parser::Pattern::Or(alts) = &patterns[0] {
            for alt in alts.iter().skip(1) {
                alternates.push(alt);
            }
        }
        for p in &patterns[1..] {
            if let parser::Pattern::Or(alts) = p {
                for alt in alts {
                    alternates.push(alt);
                }
            } else {
                alternates.push(p);
            }
        }
        let mut extras = Vec::with_capacity(alternates.len());
        for p in alternates {
            // Skip extras that would need variable bindings —
            // multi-value-with-bindings is a rare combination and
            // mixing bindings across alternates would require the
            // pattern compiler to merge their scopes.
            if self.pattern_has_variables(p) {
                continue;
            }
            match self.lower_pattern_to_expression(p) {
                Ok(expr) => extras.push(expr),
                Err(_) => continue,
            }
        }
        Ok(extras)
    }
}
