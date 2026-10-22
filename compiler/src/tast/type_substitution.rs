//! Shared substitution for lowered expressions and subsequent validation.

use super::{SymbolId, SymbolTable, TypeId, TypeTable, core::TypeKind};
use std::{cell::RefCell, collections::BTreeMap};

pub(crate) struct TypeSubstitutions<'a> {
    pub type_table: &'a RefCell<TypeTable>,
    pub symbol_table: &'a SymbolTable,
    pub class_type_params: Option<&'a BTreeMap<SymbolId, Vec<TypeId>>>,
}

impl TypeSubstitutions<'_> {
    pub(crate) fn substitute_receiver_type(&self, member_type: TypeId, receiver: TypeId) -> TypeId {
        let mut bindings = Vec::new();
        let mut current = receiver;
        let mut visited = std::collections::BTreeSet::new();
        for _ in 0..32 {
            if !visited.insert(current) {
                break;
            }
            let (current_bindings, superclass) = {
                let tt = self.type_table.borrow();
                if let Some(TypeKind::TypeAlias { target_type, .. }) =
                    tt.get(current).map(|t| &t.kind)
                {
                    current = *target_type;
                    continue;
                }
                let (class_symbol, params, args) = match tt.get(current).map(|t| &t.kind) {
                    Some(TypeKind::Class {
                        symbol_id,
                        type_args,
                    })
                    | Some(TypeKind::Interface {
                        symbol_id,
                        type_args,
                    })
                    | Some(TypeKind::Abstract {
                        symbol_id,
                        type_args,
                        ..
                    }) => {
                        let params =
                            self.symbol_table
                                .get_class_type_params(*symbol_id)
                                .or_else(|| {
                                    self.class_type_params
                                        .and_then(|params| params.get(symbol_id))
                                });
                        (
                            Some(*symbol_id),
                            params.map(Vec::as_slice).unwrap_or(&[]),
                            type_args.as_slice(),
                        )
                    }
                    Some(TypeKind::Enum {
                        symbol_id,
                        type_args,
                    }) => {
                        let declared = self.symbol_table.get_symbol(*symbol_id);
                        let Some(TypeKind::Enum {
                            type_args: params, ..
                        }) = declared.and_then(|s| tt.get(s.type_id)).map(|t| &t.kind)
                        else {
                            break;
                        };
                        (None, params.as_slice(), type_args.as_slice())
                    }
                    Some(TypeKind::GenericInstance {
                        base_type,
                        type_args,
                        ..
                    }) => match tt.get(*base_type).map(|t| &t.kind) {
                        Some(TypeKind::Class {
                            symbol_id,
                            type_args: params,
                        })
                        | Some(TypeKind::Interface {
                            symbol_id,
                            type_args: params,
                        }) => (
                            Some(*symbol_id),
                            self.symbol_table
                                .get_class_type_params(*symbol_id)
                                .map(Vec::as_slice)
                                .unwrap_or(params.as_slice()),
                            type_args.as_slice(),
                        ),
                        Some(TypeKind::Enum {
                            type_args: params, ..
                        }) => (None, params.as_slice(), type_args.as_slice()),
                        _ => break,
                    },
                    _ => break,
                };
                let current_bindings = params
                    .iter()
                    .zip(args)
                    .filter_map(|(param, arg)| match tt.get(*param).map(|t| &t.kind) {
                        Some(TypeKind::TypeParameter { symbol_id, .. }) if param != arg => {
                            Some((*symbol_id, *arg))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let superclass =
                    class_symbol.and_then(|symbol| self.symbol_table.get_class_super_type(symbol));
                (current_bindings, superclass)
            };
            bindings.extend(current_bindings);
            let Some(superclass) = superclass else { break };
            // Instantiate each parent before binding its own parameters.
            current = self.substitute_type_bindings(superclass, &bindings, false);
        }
        // Imported declarations can retain a different symbol for the same
        // class parameter. Match its declared name within this receiver.
        let mut mentioned = std::collections::BTreeSet::new();
        self.collect_type_param_symbols(member_type, 0, &mut mentioned);
        for symbol in mentioned {
            if bindings.iter().any(|(param, _)| *param == symbol) {
                continue;
            }
            let Some(name) = self.symbol_table.get_symbol(symbol).map(|s| s.name) else {
                continue;
            };
            if let Some((_, arg)) = bindings.iter().find(|(param, _)| {
                self.symbol_table
                    .get_symbol(*param)
                    .is_some_and(|s| s.name == name)
            }) {
                bindings.push((symbol, *arg));
            }
        }
        self.substitute_type_bindings(member_type, &bindings, false)
    }

    pub(crate) fn substitute_type_bindings(
        &self,
        ty: TypeId,
        bindings: &[(SymbolId, TypeId)],
        expand_bindings: bool,
    ) -> TypeId {
        use crate::tast::core::{AnonymousField, TypeKind};
        if bindings.is_empty() {
            return ty;
        }
        let kind = self.type_table.borrow().get(ty).map(|i| i.kind.clone());
        let sub = |ids: &[TypeId]| -> Vec<TypeId> {
            ids.iter()
                .map(|t| self.substitute_type_bindings(*t, bindings, expand_bindings))
                .collect()
        };
        let rebuilt = match kind {
            Some(TypeKind::TypeParameter { symbol_id, .. }) => {
                return match bindings.iter().find(|(p, _)| *p == symbol_id) {
                    Some((_, a)) if expand_bindings && *a != ty => {
                        // A binding expands once along each substitution path.
                        let remaining: Vec<_> = bindings
                            .iter()
                            .copied()
                            .filter(|(parameter, _)| *parameter != symbol_id)
                            .collect();
                        self.substitute_type_bindings(*a, &remaining, expand_bindings)
                    }
                    Some((_, a)) => *a,
                    None => ty,
                };
            }
            Some(TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                TypeKind::TypeAlias {
                    symbol_id,
                    target_type,
                    type_args: args,
                }
            }
            Some(TypeKind::GenericInstance {
                base_type,
                type_args,
                instantiation_cache_id,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                TypeKind::GenericInstance {
                    base_type,
                    type_args: args,
                    instantiation_cache_id,
                }
            }
            Some(TypeKind::Abstract {
                symbol_id,
                underlying,
                type_args,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                TypeKind::Abstract {
                    symbol_id,
                    underlying,
                    type_args: args,
                }
            }
            Some(TypeKind::Array { element_type }) => {
                let elem = self.substitute_type_bindings(element_type, bindings, expand_bindings);
                if elem == element_type {
                    return ty;
                }
                TypeKind::Array { element_type: elem }
            }
            Some(TypeKind::Function {
                params,
                return_type,
                effects,
                labels,
            }) => {
                let p = sub(&params);
                let r = self.substitute_type_bindings(return_type, bindings, expand_bindings);
                if p == params && r == return_type {
                    return ty;
                }
                TypeKind::Function {
                    params: p,
                    return_type: r,
                    effects,
                    labels,
                }
            }
            Some(TypeKind::Anonymous { fields }) => {
                let rebuilt: Vec<AnonymousField> = fields
                    .iter()
                    .map(|f| AnonymousField {
                        name: f.name,
                        type_id: self.substitute_type_bindings(
                            f.type_id,
                            bindings,
                            expand_bindings,
                        ),
                        is_public: f.is_public,
                        optional: f.optional,
                    })
                    .collect();
                if rebuilt
                    .iter()
                    .zip(&fields)
                    .all(|(a, b)| a.type_id == b.type_id)
                {
                    return ty;
                }
                TypeKind::Anonymous { fields: rebuilt }
            }
            // A typedef pre-registered as a class instantiates as a Class
            // node over the alias's parameters.
            Some(TypeKind::Class {
                symbol_id,
                type_args,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                return self
                    .type_table
                    .borrow_mut()
                    .create_class_type(symbol_id, args);
            }
            Some(TypeKind::Enum {
                symbol_id,
                type_args,
            }) => {
                let args = sub(&type_args);
                if args == type_args {
                    return ty;
                }
                return self
                    .type_table
                    .borrow_mut()
                    .create_enum_type(symbol_id, args);
            }
            Some(TypeKind::Optional { inner_type }) => {
                let inner = self.substitute_type_bindings(inner_type, bindings, expand_bindings);
                if inner == inner_type {
                    return ty;
                }
                return self.type_table.borrow_mut().create_optional_type(inner);
            }
            _ => return ty,
        };
        self.type_table.borrow_mut().create_type(rebuilt)
    }

    pub(crate) fn collect_type_param_symbols(
        &self,
        ty: TypeId,
        depth: u32,
        out: &mut std::collections::BTreeSet<SymbolId>,
    ) {
        use crate::tast::core::TypeKind;
        if depth > 8 {
            return;
        }
        let kind = self.type_table.borrow().get(ty).map(|t| t.kind.clone());
        let children: Vec<TypeId> = match kind {
            Some(TypeKind::TypeParameter { symbol_id, .. }) => {
                out.insert(symbol_id);
                Vec::new()
            }
            Some(TypeKind::Array { element_type }) => vec![element_type],
            Some(TypeKind::Optional { inner_type }) => vec![inner_type],
            Some(TypeKind::Function {
                params,
                return_type,
                ..
            }) => params
                .into_iter()
                .chain(std::iter::once(return_type))
                .collect(),
            Some(TypeKind::Anonymous { fields }) => fields.iter().map(|f| f.type_id).collect(),
            Some(TypeKind::Class { type_args, .. })
            | Some(TypeKind::Enum { type_args, .. })
            | Some(TypeKind::Abstract { type_args, .. })
            | Some(TypeKind::TypeAlias { type_args, .. })
            | Some(TypeKind::GenericInstance { type_args, .. }) => type_args,
            _ => Vec::new(),
        };
        for child in children {
            self.collect_type_param_symbols(child, depth + 1, out);
        }
    }
}
