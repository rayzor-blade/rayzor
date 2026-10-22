//! Call signatures instantiated with the same substitutions as lowering.

use super::*;

impl TypeCheckingPhase<'_> {
    pub(super) fn call_signature(
        &self,
        declared: TypeId,
        receiver: Option<TypeId>,
        arguments: &[TypeId],
        values: &[TypeId],
    ) -> TypeId {
        let substitutions = crate::tast::type_substitution::TypeSubstitutions {
            type_table: self.type_table,
            symbol_table: self.symbol_table,
            class_type_params: None,
        };
        let declared = self.type_table.borrow().resolve_alias_type(declared);
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
            substitutions.substitute_type_bindings(instantiated, &bindings, false)
        } else {
            instantiated
        };
        self.type_table.borrow().resolve_alias_type(instantiated)
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
