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
}

impl MacroTyper for DeferredMacroTyper<'_, '_> {
    fn type_expr_in_scope(&mut self, expr: &parser::Expr) -> Result<TypeId, String> {
        // A typing PROBE: lower the expression for its type and discard the
        // result. Errors raised by the probe belong to the macro (typeError
        // catches them); they must not surface as user diagnostics, so both
        // error sinks are wound back to where they were.
        let outer_before = self.lowering.collected_errors.len();
        let ctx_before = self.lowering.context.errors.len();

        let result = self.lowering.lower_expression(expr);

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
            Ok(typed) if probe_errors.is_empty() => Ok(typed.expr_type),
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
        // `Std.string` on a macro Type value. Haxe prints the constructor
        // form (`TInst(String,[])`); what the tests actually compare is two
        // of these against each other, so the load-bearing property is that
        // equal types render equal and distinct types render distinct — which
        // the display form already provides.
        self.type_display(id)
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
        resolved.map_err(|e| e.to_compilation_error().message)
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
        unify(
            a,
            b,
            self.lowering.context.type_table,
            &self.lowering.abstract_casts,
            monomorphs,
            bindings,
            0,
        )
    }

    fn type_adt_view(
        &mut self,
        id: TypeId,
    ) -> Option<(String, Vec<crate::macro_system::value::MacroValue>)> {
        use crate::macro_system::value::MacroValue as V;
        use crate::tast::core::TypeKind;
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
            TypeKind::Int
            | TypeKind::Float
            | TypeKind::Bool
            | TypeKind::String
            | TypeKind::Void
            | TypeKind::Char => ("TAbstract".to_string(), vec![V::Type(id), args_of(&[])]),
            TypeKind::Function {
                params,
                return_type,
                ..
            } => (
                "TFun".to_string(),
                vec![args_of(&params), V::Type(return_type)],
            ),
            TypeKind::Dynamic => ("TDynamic".to_string(), vec![V::Null]),
            TypeKind::Anonymous { .. } => ("TAnonymous".to_string(), vec![V::Type(id)]),
            TypeKind::Placeholder { .. } | TypeKind::Unknown => {
                ("TMono".to_string(), vec![V::Null])
            }
            _ => return None,
        })
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

    // Dynamic unifies with anything, in either direction.
    if matches!(ka, TypeKind::Dynamic) || matches!(kb, TypeKind::Dynamic) {
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

/// The source-level spelling of a type, the way `haxe.macro.TypeTools.toString`
/// prints it: named types by name, `Null<T>` written out, functions arrow-form.
/// The general error formatter is not reused here because it spells `Null<T>`
/// as `T?` and Debug-dumps named types — both visible to tests that compare
/// the string against a literal.
fn render_type(
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
            .get_symbol(symbol_id)
            .and_then(|sym| interner.get(sym.name))
            .map(|n| n.to_string())
            .unwrap_or_else(|| "<unnamed>".to_string())
    };
    let with_args = |base: String, args: &[TypeId]| {
        if args.is_empty() {
            base
        } else {
            let rendered: Vec<String> = args
                .iter()
                .map(|&a| render_type(a, type_table, symbol_table, interner, depth + 1))
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
            ..
        } => {
            let ret = render_type(return_type, type_table, symbol_table, interner, depth + 1);
            if params.is_empty() {
                format!("() -> {}", ret)
            } else {
                let ps: Vec<String> = params
                    .iter()
                    .map(|&p| render_type(p, type_table, symbol_table, interner, depth + 1))
                    .collect();
                format!("({}) -> {}", ps.join(", "), ret)
            }
        }
        TypeKind::Array { element_type } => format!(
            "Array<{}>",
            render_type(element_type, type_table, symbol_table, interner, depth + 1)
        ),
        TypeKind::Map {
            key_type,
            value_type,
        } => format!(
            "Map<{}, {}>",
            render_type(key_type, type_table, symbol_table, interner, depth + 1),
            render_type(value_type, type_table, symbol_table, interner, depth + 1)
        ),
        TypeKind::Optional { inner_type } => format!(
            "Null<{}>",
            render_type(inner_type, type_table, symbol_table, interner, depth + 1)
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
                        render_type(f.type_id, type_table, symbol_table, interner, depth + 1)
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
            render_type(base_type, type_table, symbol_table, interner, depth + 1),
            &type_args,
        ),
        other => format!("{:?}", other),
    }
}
