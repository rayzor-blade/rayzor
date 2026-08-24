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
    /// Lower a switch case
    /// Lower a switch case for expression context (where case body is an expression)
    pub(crate) fn lower_switch_case_expression(
        &mut self,
        case: &parser::Case,
    ) -> Result<TypedSwitchCase, LoweringError> {
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
    ) -> Result<TypedSwitchCase, LoweringError> {
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
