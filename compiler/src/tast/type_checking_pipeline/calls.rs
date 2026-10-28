//! Call signatures instantiated with the same substitutions as lowering.

use super::*;

impl TypeCheckingPhase<'_> {
    fn bind_result_parameters(
        &self,
        formal: TypeId,
        actual: TypeId,
        bindings: &mut Vec<(SymbolId, TypeId)>,
        depth: usize,
    ) {
        if depth > 32 || formal == actual {
            return;
        }
        let (formal, actual) = {
            let substitutions = crate::tast::type_substitution::TypeSubstitutions {
                type_table: self.type_table,
                symbol_table: self.symbol_table,
                class_type_params: None,
            };
            (
                substitutions.resolve_alias_type(formal),
                substitutions.resolve_alias_type(actual),
            )
        };
        let table = self.type_table.borrow();
        let (Some(formal_kind), Some(actual_kind)) = (
            table.get(formal).map(|ty| ty.kind.clone()),
            table.get(actual).map(|ty| ty.kind.clone()),
        ) else {
            return;
        };
        if matches!(
            actual_kind,
            TypeKind::TypeParameter { .. } | TypeKind::Unknown | TypeKind::Error
        ) {
            return;
        }
        if let TypeKind::TypeParameter { symbol_id, .. } = formal_kind {
            if !bindings.iter().any(|(symbol, _)| *symbol == symbol_id) {
                bindings.push((symbol_id, actual));
            }
            return;
        }
        let named = |kind: &TypeKind| match kind {
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
            } => Some((*symbol_id, type_args.clone())),
            TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => table
                .get_type_symbol(*base_type)
                .map(|symbol| (symbol, type_args.clone())),
            _ => None,
        };
        let children = match (&formal_kind, &actual_kind) {
            (TypeKind::Array { element_type: a }, TypeKind::Array { element_type: b }) => {
                vec![(*a, *b)]
            }
            (TypeKind::Optional { inner_type: a }, TypeKind::Optional { inner_type: b }) => {
                vec![(*a, *b)]
            }
            (
                TypeKind::Function {
                    params: a,
                    return_type: ar,
                    ..
                },
                TypeKind::Function {
                    params: b,
                    return_type: br,
                    ..
                },
            ) if a.len() == b.len() => a
                .iter()
                .copied()
                .zip(b.iter().copied())
                .chain(std::iter::once((*ar, *br)))
                .collect(),
            (TypeKind::Anonymous { fields: a }, TypeKind::Anonymous { fields: b }) => a
                .iter()
                .filter_map(|a| {
                    b.iter()
                        .find(|b| a.name == b.name)
                        .map(|b| (a.type_id, b.type_id))
                })
                .collect(),
            _ => match (named(&formal_kind), named(&actual_kind)) {
                (Some((a, aa)), Some((b, ba))) if a == b && aa.len() == ba.len() => {
                    aa.into_iter().zip(ba).collect()
                }
                _ => Vec::new(),
            },
        };
        drop(table);
        for (formal, actual) in children {
            self.bind_result_parameters(formal, actual, bindings, depth + 1);
        }
    }

    pub(super) fn argument_parameter_types(
        &mut self,
        signature: TypeId,
        values: &[TypeId],
    ) -> Option<Vec<TypeId>> {
        let (params, labels) = {
            let table = self.type_table.borrow();
            let TypeKind::Function { params, labels, .. } = &table.get(signature)?.kind else {
                return None;
            };
            (params.clone(), labels.clone())
        };
        let optional = |index: usize| labels.get(index).is_some_and(|label| label.optional);
        let mut next = 0;
        let mut matched = Vec::with_capacity(values.len());
        for &value in values {
            loop {
                let &parameter = params.get(next)?;
                let index = next;
                next += 1;
                if self.type_checker.check_compatibility(value, parameter)
                    != TypeCompatibility::Incompatible
                {
                    matched.push(parameter);
                    break;
                }
                if !optional(index) {
                    return None;
                }
            }
        }
        (next..params.len()).all(optional).then_some(matched)
    }

    pub(super) fn call_signature(
        &mut self,
        declared: TypeId,
        receiver: Option<TypeId>,
        arguments: &[TypeId],
        values: &[TypeId],
        result: TypeId,
        location: SourceLocation,
    ) -> TypeId {
        let substitutions = crate::tast::type_substitution::TypeSubstitutions {
            type_table: self.type_table,
            symbol_table: self.symbol_table,
            class_type_params: None,
        };
        let declared = substitutions.callable_type(declared);
        let mut parameters = std::collections::BTreeSet::new();
        substitutions.collect_type_param_symbols(declared, 0, &mut parameters);
        // Receiver bindings are fixed: arguments cannot rebind Map<K, Bool>'s
        // value parameter to Int merely because an Int was passed to set.
        let instantiated = receiver.map_or(declared, |receiver| {
            substitutions.substitute_receiver_type(declared, receiver)
        });
        let mut remaining = std::collections::BTreeSet::new();
        substitutions.collect_type_param_symbols(instantiated, 0, &mut remaining);
        let order = if parameters.len() == arguments.len() {
            &parameters
        } else {
            &remaining
        };
        let instantiated = if !arguments.is_empty() && order.len() == arguments.len() {
            let bindings: Vec<_> = order
                .iter()
                .copied()
                .zip(arguments.iter().copied())
                .filter(|(symbol, _)| remaining.contains(symbol))
                .collect();
            substitutions.substitute_type_bindings(instantiated, &bindings, false)
        } else {
            instantiated
        };
        // Erased calls deliberately omit specialization arguments for direct
        // parameters. Validation still needs their inferred value types.
        let instantiated = if arguments.is_empty() {
            let params = match self.type_table.borrow().get(instantiated).map(|t| &t.kind) {
                Some(TypeKind::Function { params, .. }) => params.clone(),
                _ => Vec::new(),
            };
            let mut bindings = Vec::new();
            for (parameter, value) in params.iter().zip(values) {
                let types = self.type_table.borrow();
                if let Some(TypeKind::TypeParameter { symbol_id, .. }) =
                    types.get(*parameter).map(|t| &t.kind)
                {
                    if let Some((_, previous)) = bindings.iter_mut().find(|(id, _)| id == symbol_id)
                    {
                        if *previous == types.int_type() && *value == types.float_type() {
                            *previous = *value;
                        }
                    } else {
                        bindings.push((*symbol_id, *value));
                    }
                }
            }
            for (parameter, value) in params.iter().zip(values) {
                self.bind_result_parameters(*parameter, *value, &mut bindings, 0);
            }
            substitutions.substitute_type_bindings(instantiated, &bindings, false)
        } else {
            instantiated
        };
        let returned = {
            let table = self.type_table.borrow();
            match table.get(instantiated).map(|ty| &ty.kind) {
                Some(TypeKind::Function { return_type, .. }) => Some(*return_type),
                _ => None,
            }
        };
        let mut bindings = Vec::new();
        if let Some(returned) = returned {
            self.bind_result_parameters(returned, result, &mut bindings, 0);
        }
        let instantiated = substitutions.substitute_type_bindings(instantiated, &bindings, false);
        let instantiated = substitutions.resolve_alias_type(instantiated);
        if parameters.is_empty() {
            return instantiated;
        }
        let mut bindings = Vec::new();
        self.bind_result_parameters(declared, instantiated, &mut bindings, 0);
        let declared_params = {
            let table = self.type_table.borrow();
            match table.get(declared).map(|ty| &ty.kind) {
                Some(TypeKind::Function { params, .. }) => params.clone(),
                _ => Vec::new(),
            }
        };
        for (parameter, value) in declared_params.iter().zip(values) {
            self.bind_result_parameters(*parameter, *value, &mut bindings, 0);
        }
        let instantiated = substitutions.substitute_type_bindings(instantiated, &bindings, false);
        let constraints = {
            let table = self.type_table.borrow();
            parameters
                .iter()
                .flat_map(|symbol| {
                    let constraints = table
                        .types_for_symbol(*symbol)
                        .into_iter()
                        .flatten()
                        .rev()
                        .find_map(|id| match &table.get(*id)?.kind {
                            TypeKind::TypeParameter { constraints, .. }
                                if !constraints.is_empty() =>
                            {
                                Some(constraints.clone())
                            }
                            _ => None,
                        })
                        .unwrap_or_default();
                    bindings
                        .iter()
                        .find(|(bound, _)| bound == symbol)
                        .into_iter()
                        .flat_map(|(_, actual)| {
                            constraints
                                .iter()
                                .map(|constraint| (*actual, *constraint))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let constraints: Vec<_> = constraints
            .into_iter()
            .map(|(actual, constraint)| {
                let mut constraint_bindings = bindings.clone();
                let mut mentioned = std::collections::BTreeSet::new();
                substitutions.collect_type_param_symbols(constraint, 0, &mut mentioned);
                for symbol in mentioned {
                    if constraint_bindings
                        .iter()
                        .any(|(bound, _)| *bound == symbol)
                    {
                        continue;
                    }
                    let name = self
                        .symbol_table
                        .get_symbol(symbol)
                        .map(|symbol| symbol.name);
                    if let Some(argument) = bindings.iter().find_map(|(parameter, argument)| {
                        (self
                            .symbol_table
                            .get_symbol(*parameter)
                            .map(|symbol| symbol.name)
                            == name)
                            .then_some(*argument)
                    }) {
                        constraint_bindings.push((symbol, argument));
                    }
                }
                (
                    actual,
                    substitutions.substitute_type_bindings(constraint, &constraint_bindings, false),
                )
            })
            .collect();
        for (actual, constraint) in constraints {
            if self.constructible_constraint_holds(actual, constraint) == Some(false) {
                self.emit_enhanced_type_error(
                    actual,
                    constraint,
                    location,
                    "Constructor does not satisfy the type parameter constraint",
                    &TypeErrorContext::GenericConstraint,
                );
            }
        }
        instantiated
    }

    fn constructible_constraint_holds(
        &mut self,
        actual: TypeId,
        constraint: TypeId,
    ) -> Option<bool> {
        let (actual_symbol, signature) = {
            let table = self.type_table.borrow();
            let kind = &table.get(table.resolve_alias_type(constraint))?.kind;
            let (symbol, args) = match kind {
                TypeKind::Abstract {
                    symbol_id,
                    type_args,
                    ..
                } => (*symbol_id, type_args),
                TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => (table.get_type_symbol(*base_type)?, type_args),
                _ => return None,
            };
            let declaration = self.symbol_table.get_symbol(symbol)?;
            if self.string_interner.get(declaration.name)? != "Constructible" || args.len() != 1 {
                return None;
            }
            let actual = table.resolve_alias_type(actual);
            let actual_symbol = match table.get(actual).map(|ty| &ty.kind) {
                Some(TypeKind::Class { symbol_id, .. }) => *symbol_id,
                Some(TypeKind::TypeParameter { .. } | TypeKind::Unknown | TypeKind::Dynamic) => {
                    return None;
                }
                _ => return Some(false),
            };
            (actual_symbol, args[0])
        };
        let mut owner = Some(actual_symbol);
        let mut seen = std::collections::BTreeSet::new();
        while let Some(symbol) = owner {
            if !seen.insert(symbol) {
                break;
            }
            if let Some(constructor) = self
                .symbol_table
                .get_class_constructor(symbol)
                .and_then(|symbol| self.symbol_table.get_symbol(symbol))
            {
                if constructor.visibility != crate::tast::Visibility::Public {
                    return Some(false);
                }
                let substitutions = crate::tast::type_substitution::TypeSubstitutions {
                    type_table: self.type_table,
                    symbol_table: self.symbol_table,
                    class_type_params: None,
                };
                let constructor =
                    substitutions.substitute_receiver_type(constructor.type_id, actual);
                return Some(
                    self.type_checker
                        .check_compatibility(constructor, signature)
                        != TypeCompatibility::Incompatible,
                );
            }
            owner = self
                .symbol_table
                .get_class_super_type(symbol)
                .and_then(|parent| self.type_table.borrow().get_type_symbol(parent));
        }
        None
    }

    pub(super) fn accepts_argument_count(&self, signature: TypeId, count: usize) -> bool {
        let types = self.type_table.borrow();
        let Some(TypeKind::Function { params, labels, .. }) = types.get(signature).map(|t| &t.kind)
        else {
            return false;
        };
        count <= params.len()
            && (count..params.len()).all(|i| labels.get(i).is_some_and(|label| label.optional))
    }
}
