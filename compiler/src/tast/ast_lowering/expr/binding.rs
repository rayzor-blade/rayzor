//! `var` binding, and the inference that fills an untyped one.

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
    /// Monomorph rewrite: if `arr_ast` names a var that was declared as an
    /// untyped empty array literal (still `Array<Dynamic>`), bind its element
    /// type from `elem_ast`'s peeked type. Called BEFORE lowering the
    /// `arr.push(e)` / `arr[i] = e`, so the receiver and every later reference
    /// resolve to the concrete `Array<T>` and route through the typed (e.g. f64)
    /// push/get path instead of the generic one that truncates floats.
    pub(crate) fn try_bind_inferred_array(&mut self, arr_ast: &Expr, elem_ast: &Expr) {
        let name = match &arr_ast.kind {
            ExprKind::Ident(n) => n.clone(),
            _ => return,
        };
        let interned = self.context.intern_string(&name);
        let sym = match self.resolve_symbol_in_scope_hierarchy(interned) {
            Some(s) => s,
            None => return,
        };
        if !self.empty_array_inferred.contains_key(&sym) {
            return;
        }
        // A use happened on an empty-inferred array. Resolve the element type:
        // cheap syntactic peek first, then lower Call/Field/Index args (e.g.
        // `a.push(x.getFlat(i))`) for their type — the lowered result is
        // discarded (the push re-lowers it). Lambdas etc. are left to a later
        // peekable push (avoids double-lowering a closure).
        let elem_ty = self.peek_ast_expr_type(elem_ast).or_else(|| {
            if matches!(
                &elem_ast.kind,
                ExprKind::Call { .. }
                    | ExprKind::Field { .. }
                    | ExprKind::Index { .. }
                    | ExprKind::New { .. }
            ) {
                self.lower_expression(elem_ast).ok().map(|te| te.expr_type)
            } else {
                None
            }
        });
        // Only bind to a CONCRETE element type; Dynamic/Unknown/None means the
        // type is uncertain at this use — mark it (a later peekable push can
        // still bind & clear it; if not, it warns at end of file).
        let concrete = elem_ty.filter(|t| {
            let tt = self.context.type_table.borrow();
            !matches!(
                tt.get(*t).map(|ty| &ty.kind),
                None | Some(TypeKind::Dynamic) | Some(TypeKind::Unknown) | Some(TypeKind::Error)
            )
        });
        match concrete {
            Some(t) => {
                let array_ty = self.context.type_table.borrow_mut().create_array_type(t);
                self.context.symbol_table.update_symbol_type(sym, array_ty);
                self.empty_array_inferred.remove(&sym);
                self.empty_array_used_uncertain.remove(&sym);
            }
            None => {
                self.empty_array_used_uncertain.insert(sym);
            }
        }
    }

    /// Drain the "untyped empty array, element type uncertain" warnings: symbols
    /// that were used but never bound to a concrete element type, so they stay
    /// `Array<Dynamic>`. Returns (declaration location, message) pairs for the
    /// pipeline to surface as `Correctness` warnings. Call after `lower_file`.
    pub fn take_empty_array_warnings(&mut self) -> Vec<(SourceLocation, String)> {
        let mut out = Vec::new();
        for sym in std::mem::take(&mut self.empty_array_used_uncertain) {
            if let Some(loc) = self.empty_array_inferred.get(&sym) {
                out.push((
                    *loc,
                    "type of array literal can not be inferred at assignment, \
                     this would fall back to Array<Dynamic>; annotate \
                     `var a:Array<T> = []` for deterministic behavior at runtime"
                        .to_string(),
                ));
            }
        }
        out
    }

    /// Infer generic type arguments from constructor argument types.
    /// When `new Container(42)` is written without explicit `<Int>`, this matches
    /// constructor param types (TypeParameter) against argument types to infer type args.
    /// Resolve `@:multiType` for the `haxe.ds.Map` abstract. When the
    /// caller writes `new Map<K, V>(...)`, replace the construction with
    /// the right concrete container based on K:
    ///
    ///   - K = String      → haxe.ds.StringMap<V>
    ///   - K = Int         → haxe.ds.IntMap<V>
    ///   - K = EnumValue   → haxe.ds.EnumValueMap<K, V>
    ///   - otherwise       → haxe.ds.ObjectMap<K, V>
    ///
    /// Returns `(class_type, type_args, class_name_override)`. When the
    /// receiver isn't `Map`, returns the inputs unchanged. This is a
    /// targeted shim — proper `@:multiType` parsing + general resolution
    /// is the long-term fix (would also unlock `EnumMap`, custom user
    /// abstracts that use the same dispatch trick, etc.). For now `Map`
    /// is the only abstract that needs this and using a non-Map abstract
    /// here would be a no-op pass-through.
    /// Reverse of `resolve_multitype_map_to_concrete` for the cases where
    /// the concrete arity dropped K. Returns the K TypeId given the
    /// concrete container's class symbol — e.g. `StringMap` ↦ `String`,
    /// `IntMap` ↦ `Int`. Returns `None` for `ObjectMap`/`EnumValueMap`
    /// (their K is already carried, so the caller doesn't need this) or
    /// unrecognized class names. Used when `var m:Map<String,V> = new Map()`
    /// has been annotation-resolved to `StringMap<V>` and we need the
    /// abstract's `<K, V>` type args back to re-run multiType cleanly at
    /// the New site.
    pub(crate) fn recover_map_key_type_from_concrete(
        &self,
        concrete_sym: SymbolId,
    ) -> Option<TypeId> {
        let name = self.context.symbol_table.get_symbol(concrete_sym)?.name;
        let name_str = self.context.string_interner.get(name)?.to_string();
        let tt = self.context.type_table.borrow();
        match name_str.as_str() {
            "StringMap" => Some(tt.string_type()),
            "IntMap" => Some(tt.int_type()),
            _ => None,
        }
    }

    /// Desugar f.bind(a, _, c) → function(b) { return f(a, b, c); }
    /// Handles partial application where `_` marks unbound parameters.
    pub(crate) fn lower_bind_expression(
        &mut self,
        expression: &Expr,
        receiver: TypedExpression,
        bind_args: &[Expr],
    ) -> LoweringResult<TypedExpression> {
        use crate::tast::core::TypeKind;

        // Get function type info
        let (func_params, func_return_type) = {
            let tt = self.context.type_table.borrow();
            if let Some(type_info) = tt.get(receiver.expr_type) {
                if let TypeKind::Function {
                    params,
                    return_type,
                    ..
                } = &type_info.kind
                {
                    (params.clone(), *return_type)
                } else {
                    return Err(LoweringError::IncompleteImplementation {
                        feature: "bind on non-function type".to_string(),
                        location: self.context.create_location(),
                    });
                }
            } else {
                return Err(LoweringError::IncompleteImplementation {
                    feature: "bind: unknown function type".to_string(),
                    location: self.context.create_location(),
                });
            }
        };

        let location = self
            .context
            .create_location_from_span(expression.span.clone());

        // Enter new function scope for the generated lambda
        let _function_scope = self.context.enter_scope(ScopeKind::Function);

        let mut lambda_params = Vec::new();
        let mut call_args: Vec<TypedExpression> = Vec::new();

        // Process bind args — `_` becomes a lambda parameter, others are bound values
        for (i, bind_arg) in bind_args.iter().enumerate() {
            let is_placeholder = matches!(&bind_arg.kind, ExprKind::Ident(name) if name == "_");

            if is_placeholder {
                // Create a lambda parameter for this placeholder
                let param_type = if i < func_params.len() {
                    func_params[i]
                } else {
                    self.context.type_table.borrow().dynamic_type()
                };
                let param_name = format!("_bind_{}", i);
                let param_interned = self.context.string_interner.intern(&param_name);
                let param_symbol = self.context.symbol_table.create_variable_with_type(
                    param_interned,
                    self.context.current_scope,
                    param_type,
                );

                lambda_params.push(TypedParameter {
                    symbol_id: param_symbol,
                    name: param_interned,
                    param_type,
                    is_optional: false,
                    default_value: None,
                    mutability: crate::tast::symbols::Mutability::Immutable,
                    ownership: Default::default(),
                    source_location: location,
                });

                // Reference to this parameter in the call
                call_args.push(TypedExpression {
                    kind: TypedExpressionKind::Variable {
                        symbol_id: param_symbol,
                    },
                    expr_type: param_type,
                    usage: VariableUsage::Copy,
                    lifetime_id: crate::tast::LifetimeId::default(),
                    source_location: location,
                    metadata: ExpressionMetadata::default(),
                });
            } else {
                // Bound value — lower normally
                let lowered = self.lower_expression(bind_arg)?;
                call_args.push(lowered);
            }
        }

        // Any remaining function params not covered by bind args become lambda params
        for i in bind_args.len()..func_params.len() {
            let param_type = func_params[i];
            let param_name = format!("_bind_{}", i);
            let param_interned = self.context.string_interner.intern(&param_name);
            let param_symbol = self.context.symbol_table.create_variable_with_type(
                param_interned,
                self.context.current_scope,
                param_type,
            );

            lambda_params.push(TypedParameter {
                symbol_id: param_symbol,
                name: param_interned,
                param_type,
                is_optional: false,
                default_value: None,
                mutability: crate::tast::symbols::Mutability::Immutable,
                ownership: Default::default(),
                source_location: location,
            });

            call_args.push(TypedExpression {
                kind: TypedExpressionKind::Variable {
                    symbol_id: param_symbol,
                },
                expr_type: param_type,
                usage: VariableUsage::Copy,
                lifetime_id: crate::tast::LifetimeId::default(),
                source_location: location,
                metadata: ExpressionMetadata::default(),
            });
        }

        // Body: return f(bound_args..., unbound_args...)
        let call_expr = TypedExpression {
            kind: TypedExpressionKind::FunctionCall {
                function: Box::new(receiver),
                arguments: call_args,
                type_arguments: Vec::new(),
            },
            expr_type: func_return_type,
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::default(),
            source_location: location,
            metadata: ExpressionMetadata::default(),
        };

        let body = vec![TypedStatement::Return {
            value: Some(call_expr),
            source_location: location,
        }];

        // Exit function scope
        self.context.exit_scope();

        // Compute Send-ness from the lambda's free variables. A lambda whose
        // captures are all `Send` produces a Function value that's itself
        // `Send` — safe to pass to `Thread.spawn` / `Future.create` / any
        // other Send sink. A lambda that captures non-Send state (an object
        // without `@:derive([Send])`, etc.) yields a Function value with
        // `is_send = false`, propagating the constraint through every
        // subsequent API the value passes through.
        //
        // Without this, the trait checker's `TypeKind::Function` rule has
        // to either reject ALL function values (over-restrictive: rejects
        // legitimate `Thread.spawn(named_fn)` calls and breaks `WorkerPool`,
        // `Future.all`, etc.) or accept ALL function values (unsound:
        // closures with non-Send captures escape to other threads). The
        // per-construction Send computation lets us be precise.
        let captures_are_send = {
            use crate::tast::capture_analyzer::CaptureAnalyzer;
            use crate::tast::trait_checker::TraitChecker;
            let analyzer = CaptureAnalyzer::new(crate::tast::ScopeId::invalid());
            let analysis = analyzer.analyze_function_literal(&lambda_params, &body);
            let trait_checker = TraitChecker::new(
                self.context.type_table,
                self.context.symbol_table,
                self.context.string_interner,
                &[],
            );
            analysis
                .captures
                .iter()
                .all(|c| !c.type_id.is_valid() || trait_checker.is_send(c.type_id))
        };

        // Result type: function from unbound params → return type
        let lambda_param_types: Vec<TypeId> = lambda_params.iter().map(|p| p.param_type).collect();
        let result_type = if captures_are_send {
            // Common case — route through the cached factory.
            self.context
                .type_table
                .borrow_mut()
                .create_function_type(lambda_param_types, func_return_type)
        } else {
            // Non-Send capture present — bypass the shape-keyed cache so
            // this lambda's TypeId carries the `is_send = false` bit
            // independently from any other lambda of the same signature.
            let mut effects = crate::tast::core::FunctionEffects::default();
            effects.is_send = false;
            self.context
                .type_table
                .borrow_mut()
                .create_function_type_with_effects(lambda_param_types, func_return_type, effects)
        };

        Ok(TypedExpression {
            kind: TypedExpressionKind::FunctionLiteral {
                parameters: lambda_params,
                body,
                return_type: func_return_type,
            },
            expr_type: result_type,
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::default(),
            source_location: location,
            metadata: ExpressionMetadata::default(),
        })
    }
}
