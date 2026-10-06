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

impl DeferredMacroTyper<'_, '_> {
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
                let entry = object(BTreeMap::from([
                    (
                        "name".to_string(),
                        string(
                            self.lowering
                                .context
                                .string_interner
                                .get(name)
                                .unwrap_or(""),
                        ),
                    ),
                    ("type".to_string(), V::Type(sym.type_id)),
                    (
                        "meta".to_string(),
                        object(BTreeMap::from([(
                            "__meta__".to_string(),
                            array(Vec::new()),
                        )])),
                    ),
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
    // A type in a package prints with its path, as TypeTools.toString does.
    let name_of = |symbol_id| {
        symbol_table
            .get_symbol(symbol_id)
            .and_then(|sym| interner.get(sym.qualified_name.unwrap_or(sym.name)))
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
