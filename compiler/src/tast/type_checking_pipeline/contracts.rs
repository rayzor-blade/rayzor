//! What a type must satisfy to be thrown, caught, iterated, or used with
//! an operator.

use super::*;

impl<'a> TypeCheckingPhase<'a> {

    /// Validate that a type can be used as an exception type in catch clauses
    pub(crate) fn validate_exception_type(
        &mut self,
        exception_type: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(exception_type) {
            match &type_info.kind {
                // Any type can be thrown in Haxe, but we can warn about unusual types
                super::TypeKind::Dynamic => Ok(()),
                super::TypeKind::String => Ok(()),
                super::TypeKind::Class { .. } => Ok(()),
                super::TypeKind::Interface { .. } => Ok(()),
                super::TypeKind::Int | super::TypeKind::Float | super::TypeKind::Bool => {
                    // Primitive types are unusual as exceptions but technically allowed
                    Ok(())
                }
                _ => Ok(()), // Allow any type to be throwable for flexibility
            }
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<unknown_exception_type>"),
                },
                location,
                context: "Exception type is not defined".to_string(),
                suggestion: Some("Use a defined class or interface as exception type".to_string()),
            });
            Err("Undefined exception type".to_string())
        }
    }


    /// Validate that a type can be thrown
    pub(crate) fn validate_throwable_type(
        &mut self,
        throwable_type: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        // In Haxe, any type can be thrown, but we provide helpful warnings
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(throwable_type) {
            match &type_info.kind {
                super::TypeKind::Dynamic => Ok(()),
                super::TypeKind::String => Ok(()),
                super::TypeKind::Class { .. } => Ok(()),
                super::TypeKind::Interface { .. } => Ok(()),
                _ => Ok(()), // Allow throwing any type
            }
        } else {
            self.emit_error(TypeCheckError {
                kind: TypeErrorKind::UndefinedType {
                    name: self.string_interner.intern("<unknown_throwable_type>"),
                },
                location,
                context: "Thrown type is not defined".to_string(),
                suggestion: Some("Ensure the thrown expression has a valid type".to_string()),
            });
            Err("Undefined throwable type".to_string())
        }
    }


    /// Validate that a type is iterable (for for-in loops)
    pub(crate) fn validate_iterable_type(
        &mut self,
        iterable_type: TypeId,
        location: SourceLocation,
    ) -> Result<(), String> {
        let type_table = self.type_checker.type_table.borrow();

        if let Some(type_info) = type_table.get(iterable_type) {
            let is_iterable = match &type_info.kind {
                super::TypeKind::Array { .. } => true,
                super::TypeKind::String => true, // Strings are iterable (char by char)
                super::TypeKind::Dynamic => true, // Dynamic allows anything
                super::TypeKind::Class { .. } => {
                    // TODO: Check if class implements Iterable interface
                    // For now, assume classes with "iterator" or "keyValueIterator" methods are iterable
                    true // Be permissive for now
                }
                super::TypeKind::Interface { .. } => {
                    // TODO: Check if it's an Iterable interface
                    true // Be permissive for now
                }
                _ => false,
            };

            if !is_iterable {
                self.emit_enhanced_type_error(
                    iterable_type,
                    self.type_checker.type_table.borrow().dynamic_type(), // Use Dynamic as "any iterable"
                    location,
                    "Type is not iterable",
                    &TypeErrorContext::ForInLoop,
                );
                return Err("Type is not iterable".to_string());
            }

            Ok(())
        } else {
            // Type not found in type table — this can happen for generic array types,
            // map types, or other parameterized types whose TypeIds aren't fully
            // registered during the validation pass. Be permissive here; actual type
            // resolution is handled correctly in later pipeline stages (HIR/MIR lowering).
            Ok(())
        }
    }


    /// Find a method with matching @:op metadata for the given operator
    /// Returns (method_symbol, abstract_symbol) if found
    pub(crate) fn find_operator_method(
        &self,
        operand_type: TypeId,
        operator: &BinaryOperator,
    ) -> Option<(SymbolId, SymbolId)> {
        let type_table = self.type_checker.type_table.borrow();

        // Check if this type is an abstract type
        let type_info = type_table.get(operand_type)?;
        let abstract_symbol = match &type_info.kind {
            super::TypeKind::Abstract { symbol_id, .. } => *symbol_id,
            _ => return None, // Not an abstract type
        };

        drop(type_table);

        // Get the abstract definition from the current file being checked
        let typed_file_ptr = self.current_typed_file?;
        let typed_file = unsafe { &*typed_file_ptr };

        // Search all abstracts for the one matching our symbol
        for abstract_def in &typed_file.abstracts {
            if abstract_def.symbol_id != abstract_symbol {
                continue;
            }

            // Found the abstract, now search for a method with matching @:op metadata
            for method in &abstract_def.methods {
                for (op_str, _params) in &method.metadata.operator_metadata {
                    if let Some(parsed_op) = Self::parse_operator_from_metadata(op_str) {
                        if std::mem::discriminant(&parsed_op) == std::mem::discriminant(operator) {
                            // Found a matching operator method!
                            return Some((method.symbol_id, abstract_symbol));
                        }
                    }
                }
            }
        }

        None
    }


    /// Parse operator metadata string to extract the operator type
    /// e.g. "A Add B" → Some(BinaryOperator::Add)
    pub(crate) fn parse_operator_from_metadata(op_str: &str) -> Option<BinaryOperator> {
        if op_str.contains("Add") {
            Some(BinaryOperator::Add)
        } else if op_str.contains("Sub") {
            Some(BinaryOperator::Sub)
        } else if op_str.contains("Mul") {
            Some(BinaryOperator::Mul)
        } else if op_str.contains("Div") {
            Some(BinaryOperator::Div)
        } else if op_str.contains("Mod") {
            Some(BinaryOperator::Mod)
        } else if op_str.contains("Eq") && !op_str.contains("Ne") {
            Some(BinaryOperator::Eq)
        } else if op_str.contains("Ne") {
            Some(BinaryOperator::Ne)
        } else if op_str.contains("Lt") {
            Some(BinaryOperator::Lt)
        } else if op_str.contains("Le") {
            Some(BinaryOperator::Le)
        } else if op_str.contains("Gt") {
            Some(BinaryOperator::Gt)
        } else if op_str.contains("Ge") {
            Some(BinaryOperator::Ge)
        } else {
            None
        }
    }
}
