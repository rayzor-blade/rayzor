//! `for`, comprehensions, and the iteration protocol.

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
    /// Lower a for-in loop expression (ExprKind::For).
    /// Extracted from lower_expression to reduce stack frame size.
    #[inline(never)]
    pub(crate) fn lower_for_expression(
        &mut self,
        expression: &Expr,
        var: &str,
        key_var: Option<&str>,
        iter: &Expr,
        body: &Expr,
    ) -> LoweringResult<TypedExpression> {
        // Check if the iterator is a range expression (0...len)
        // If so, desugar it to a while loop instead of trying to lower as iterable
        if let ExprKind::Binary {
            op: BinaryOp::Range,
            left,
            right,
        } = &iter.kind
        {
            // Desugar: for (i in start...end) { body }
            // Into: var i = start; while (i < end) { body; i++; }

            let start_expr = self.lower_expression(left)?;
            let end_expr = self.lower_expression(right)?;

            // Create the loop body scope
            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());

            // Create the loop variable
            let var_name = self.context.intern_string(var);
            let int_type = self.context.type_table.borrow().int_type();
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                int_type,
            );

            // Enter the loop body scope
            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;

            let body_stmt = self.convert_expression_to_statement(body)?;

            // Restore the previous scope
            self.context.current_scope = old_scope;

            // Create: var i = start
            let init_stmt = TypedStatement::VarDeclaration {
                symbol_id: var_symbol,
                var_type: int_type,
                initializer: Some(start_expr),
                source_location: SourceLocation::unknown(),
                mutability: crate::tast::Mutability::Mutable,
            };

            // Create: i < end
            let var_ref = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: var_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            let condition = TypedExpression {
                expr_type: self.context.type_table.borrow().bool_type(),
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(var_ref.clone()),
                    operator: BinaryOperator::Lt,
                    right: Box::new(end_expr),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            // Create: i++
            let one_literal = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(1),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            let increment = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(var_ref),
                    operator: BinaryOperator::AddAssign,
                    right: Box::new(one_literal),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            };

            // Create: for (var i = start; i < end; i++) { body }
            // Using TypedStatement::For separates the update from the body,
            // so `continue` properly executes i++ before jumping to condition.
            let for_stmt = TypedStatement::For {
                init: Some(Box::new(init_stmt)),
                condition: Some(condition),
                update: Some(increment),
                body: Box::new(body_stmt),
                source_location: SourceLocation::unknown(),
            };

            // Return block: { for (...) { body } }
            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![for_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location: SourceLocation::unknown(),
                metadata: ExpressionMetadata::default(),
            });
        }

        // Not a range - check if it's an Array (we can inline the iterator)

        // Lower the iterable expression first
        let iterable_expr = self.lower_expression(iter)?;

        // Check if the iterable is an Array type - if so, we inline the iterator pattern
        // to avoid needing to compile ArrayIterator with its generic type parameters
        let is_array = {
            let type_table = self.context.type_table.borrow();
            if let Some(actual_type) = type_table.get(iterable_expr.expr_type) {
                matches!(&actual_type.kind, TypeKind::Array { .. })
            } else {
                false
            }
        };

        if is_array && key_var.is_none() {
            // INLINE ARRAY ITERATOR PATTERN:
            // for (x in arr) becomes:
            // {
            //     var _i = 0;
            //     var _len = arr.length;
            //     while (_i < _len) {
            //         var x = arr[_i];
            //         body;
            //         _i++;
            //     }
            // }

            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());
            let source_location = self.context.create_location_from_span(expression.span);
            let int_type = self.context.type_table.borrow().int_type();
            let bool_type = self.context.type_table.borrow().bool_type();
            let element_type = self.infer_element_type_from_iterable(&iterable_expr);

            // Create loop variable
            let var_name = self.context.intern_string(var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                element_type,
            );

            // Create internal _i counter
            let counter_name = self.context.intern_string("_i");
            let counter_symbol = self.context.symbol_table.create_variable_with_type(
                counter_name,
                loop_body_scope_id,
                int_type,
            );

            // Create internal _len variable
            let len_name = self.context.intern_string("_len");
            let len_symbol = self.context.symbol_table.create_variable_with_type(
                len_name,
                loop_body_scope_id,
                int_type,
            );

            // var _i = 0
            let zero_literal = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(0),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let counter_init = TypedStatement::VarDeclaration {
                symbol_id: counter_symbol,
                var_type: int_type,
                initializer: Some(zero_literal),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };

            // var _len = arr.length (field access)
            // Create a symbol for the field access. The name "length" will be used during
            // HIR->MIR lowering to look up the stdlib runtime function (haxe_array_length)
            let length_name = self.context.intern_string("length");
            let length_symbol = self.context.symbol_table.create_variable(length_name);

            let length_access = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::FieldAccess {
                    object: Box::new(iterable_expr.clone()),
                    field_symbol: length_symbol,
                    is_optional: false,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let len_init = TypedStatement::VarDeclaration {
                symbol_id: len_symbol,
                var_type: int_type,
                initializer: Some(length_access),
                source_location,
                mutability: crate::tast::Mutability::Immutable,
            };

            // _i < _len
            let counter_ref = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: counter_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let len_ref = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: len_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let condition = TypedExpression {
                expr_type: bool_type,
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(counter_ref.clone()),
                    operator: BinaryOperator::Lt,
                    right: Box::new(len_ref),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            // var x = arr[_i]
            let array_access = TypedExpression {
                expr_type: element_type,
                kind: TypedExpressionKind::ArrayAccess {
                    array: Box::new(iterable_expr),
                    index: Box::new(counter_ref.clone()),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let var_decl = TypedStatement::VarDeclaration {
                symbol_id: var_symbol,
                var_type: element_type,
                initializer: Some(array_access),
                source_location,
                mutability: crate::tast::Mutability::Immutable,
            };

            // Convert body
            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;
            let body_stmt = self.convert_expression_to_statement(body)?;
            self.context.current_scope = old_scope;

            // _i++
            let one_literal = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::Literal {
                    value: LiteralValue::Int(1),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            let increment = TypedExpression {
                expr_type: int_type,
                kind: TypedExpressionKind::BinaryOp {
                    left: Box::new(counter_ref),
                    operator: BinaryOperator::AddAssign,
                    right: Box::new(one_literal),
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };
            // for-body: { var x = arr[_i]; body }  — the _i++ is the loop UPDATE,
            // not a body statement. Using TypedStatement::For (like the range
            // desugar above) separates the update from the body so `continue`
            // executes _i++ before jumping to the condition; a `while` with _i++
            // at the end of the body would skip the increment on `continue` and
            // spin forever on the element at the `continue`.
            let for_body = TypedStatement::Block {
                statements: vec![var_decl, body_stmt],
                scope_id: loop_body_scope_id,
                source_location,
            };

            // for (; _i < _len; _i++) { var x = arr[_i]; body }
            let for_stmt = TypedStatement::For {
                init: None,
                condition: Some(condition),
                update: Some(increment),
                body: Box::new(for_body),
                source_location,
            };

            // Return block: { var _i = 0; var _len = arr.length; for (...) }
            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![counter_init, len_init, for_stmt],
                    scope_id: ScopeId::from_raw(self.context.next_scope_id()),
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            });
        }

        // For Map types (including IntMap/StringMap extern classes), emit a ForIn
        // statement that passes through to MIR level where keys_to_array + array
        // iteration handles it
        let map_key_value_types: Option<(TypeId, TypeId)> = {
            let type_table = self.context.type_table.borrow();
            if let Some(actual_type) = type_table.get(iterable_expr.expr_type) {
                match &actual_type.kind {
                    TypeKind::Map {
                        key_type,
                        value_type,
                    } => Some((*key_type, *value_type)),
                    TypeKind::Class {
                        symbol_id,
                        type_args,
                    } => {
                        // Check if this is IntMap<T> or StringMap<T> extern class
                        let class_name = self
                            .context
                            .symbol_table
                            .get_symbol(*symbol_id)
                            .and_then(|sym| self.context.string_interner.get(sym.name));
                        match class_name {
                            Some("IntMap") => {
                                let value_type = type_args
                                    .first()
                                    .copied()
                                    .unwrap_or_else(|| type_table.dynamic_type());
                                Some((type_table.int_type(), value_type))
                            }
                            Some("StringMap") => {
                                let value_type = type_args
                                    .first()
                                    .copied()
                                    .unwrap_or_else(|| type_table.dynamic_type());
                                Some((type_table.string_type(), value_type))
                            }
                            Some("ObjectMap") => {
                                // ObjectMap<K, V> has two type args
                                let key_type = type_args
                                    .first()
                                    .copied()
                                    .unwrap_or_else(|| type_table.dynamic_type());
                                let value_type = type_args
                                    .get(1)
                                    .copied()
                                    .unwrap_or_else(|| type_table.dynamic_type());
                                Some((key_type, value_type))
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        };

        if let Some((key_type_id, value_type_id)) = map_key_value_types {
            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());
            let source_location = self.context.create_location_from_span(expression.span);

            // The loop variable always binds the VALUE: `for (v in map)` iterates map
            // VALUES in Haxe (keys come from `for (k in map.keys())`), and
            // `for (k => v in map)` binds v to the value and k to the key. Previously
            // single-variable iteration bound the variable to the KEY type, so e.g.
            // `for (v in stringIntMap) sum += v` typed v as String → `sum += v`
            // compiled as string concat and the Int value was cast to a string
            // pointer → SIGSEGV.
            let var_name = self.context.intern_string(var);
            let var_type = value_type_id;
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                var_type,
            );

            // Create key variable if key=>value syntax is used
            let key_sym = if let Some(ref key_name) = key_var {
                let key_interned = self.context.intern_string(key_name);
                Some(self.context.symbol_table.create_variable_with_type(
                    key_interned,
                    loop_body_scope_id,
                    key_type_id,
                ))
            } else {
                None
            };

            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;
            let body_stmt = self.convert_expression_to_statement(body)?;
            self.context.current_scope = old_scope;

            let for_in_stmt = TypedStatement::ForIn {
                value_var: var_symbol,
                key_var: key_sym,
                iterable: iterable_expr,
                body: Box::new(body_stmt),
                source_location,
            };

            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![for_in_stmt],
                    scope_id: loop_body_scope_id,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            });
        }

        // For non-Array, non-Map types (classes with iterator protocol, interfaces, etc.),
        // emit a ForIn statement and let MIR handle the iteration dispatch.
        {
            let loop_body_scope_id = ScopeId::from_raw(self.context.next_scope_id());
            let source_location = self.context.create_location_from_span(expression.span);

            let var_name = self.context.intern_string(var);
            let element_type = self.infer_element_type_from_iterable(&iterable_expr);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                loop_body_scope_id,
                element_type,
            );

            let key_sym = if let Some(ref key_name) = key_var {
                let key_interned = self.context.intern_string(key_name);
                let int_type = self.context.type_table.borrow().int_type();
                Some(self.context.symbol_table.create_variable_with_type(
                    key_interned,
                    loop_body_scope_id,
                    int_type,
                ))
            } else {
                None
            };

            let old_scope = self.context.current_scope;
            self.context.current_scope = loop_body_scope_id;
            let body_stmt = self.convert_expression_to_statement(body)?;
            self.context.current_scope = old_scope;

            let for_in_stmt = TypedStatement::ForIn {
                value_var: var_symbol,
                key_var: key_sym,
                iterable: iterable_expr,
                body: Box::new(body_stmt),
                source_location,
            };

            return Ok(TypedExpression {
                expr_type: self.context.type_table.borrow().void_type(),
                kind: TypedExpressionKind::Block {
                    statements: vec![for_in_stmt],
                    scope_id: loop_body_scope_id,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            });
        }
    }

    /// Lower array comprehension: [for (i in 0...10) i * 2]
    pub(crate) fn lower_array_comprehension(
        &mut self,
        for_parts: &[parser::ComprehensionFor],
        expr: &Expr,
        location: &SourceLocation,
    ) -> LoweringResult<TypedExpression> {
        // Create a new scope for the comprehension
        let comprehension_scope = self
            .context
            .scope_tree
            .create_scope(Some(self.context.current_scope));

        let previous_scope = self.context.current_scope;
        self.context.current_scope = comprehension_scope;

        // Lower all for parts
        let mut typed_for_parts = Vec::new();

        for for_part in for_parts {
            // Lower the iterator expression first
            let typed_iterator = self.lower_expression(&for_part.iter)?;

            // Determine the element type from the iterator
            let (element_type, key_type) = self.infer_iterator_types(&typed_iterator)?;

            // Create symbol for the loop variable
            let var_name = self.context.intern_string(&for_part.var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                comprehension_scope,
                element_type,
            );

            // Handle optional key variable for key-value iteration
            let key_var_symbol = if let Some(key_var) = &for_part.key_var {
                let key_type =
                    key_type.unwrap_or_else(|| self.context.type_table.borrow().int_type());
                let key_name = self.context.intern_string(key_var);
                let symbol = self.context.symbol_table.create_variable_with_type(
                    key_name,
                    comprehension_scope,
                    key_type,
                );
                Some(symbol)
            } else {
                None
            };

            typed_for_parts.push(TypedComprehensionFor {
                var_symbol,
                key_var_symbol,
                iterator: typed_iterator,
                var_type: element_type,
                key_type,
                scope_id: comprehension_scope,
                source_location: location.clone(),
            });
        }

        // Lower the expression in the comprehension scope
        let typed_expr = self.lower_expression(expr)?;
        let element_type = typed_expr.expr_type;

        // Restore the previous scope
        self.context.current_scope = previous_scope;

        // Create the array type
        let array_type = self
            .context
            .type_table
            .borrow_mut()
            .create_array_type(element_type);

        Ok(TypedExpression {
            expr_type: array_type,
            kind: TypedExpressionKind::ArrayComprehension {
                for_parts: typed_for_parts,
                expression: Box::new(typed_expr),
                element_type,
            },
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::first(),
            source_location: location.clone(),
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Lower map comprehension: [for (i in 0...10) i => i * 2]
    pub(crate) fn lower_map_comprehension(
        &mut self,
        for_parts: &[parser::ComprehensionFor],
        key: &Expr,
        value: &Expr,
        location: &SourceLocation,
    ) -> LoweringResult<TypedExpression> {
        // Create a new scope for the comprehension
        let comprehension_scope = self
            .context
            .scope_tree
            .create_scope(Some(self.context.current_scope));

        let previous_scope = self.context.current_scope;
        self.context.current_scope = comprehension_scope;

        // Lower all for parts
        let mut typed_for_parts = Vec::new();

        for for_part in for_parts {
            // Lower the iterator expression first
            let typed_iterator = self.lower_expression(&for_part.iter)?;

            // Determine the element type from the iterator
            let (element_type, key_type) = self.infer_iterator_types(&typed_iterator)?;

            // Create symbol for the loop variable
            let var_name = self.context.intern_string(&for_part.var);
            let var_symbol = self.context.symbol_table.create_variable_with_type(
                var_name,
                comprehension_scope,
                element_type,
            );

            // Handle optional key variable for key-value iteration
            let key_var_symbol = if let Some(key_var) = &for_part.key_var {
                let key_type =
                    key_type.unwrap_or_else(|| self.context.type_table.borrow().int_type());
                let key_name = self.context.intern_string(key_var);
                let symbol = self.context.symbol_table.create_variable_with_type(
                    key_name,
                    comprehension_scope,
                    key_type,
                );
                Some(symbol)
            } else {
                None
            };

            typed_for_parts.push(TypedComprehensionFor {
                var_symbol,
                key_var_symbol,
                iterator: typed_iterator,
                var_type: element_type,
                key_type,
                scope_id: comprehension_scope,
                source_location: location.clone(),
            });
        }

        // Lower the key and value expressions in the comprehension scope
        let typed_key = self.lower_expression(key)?;
        let typed_value = self.lower_expression(value)?;

        let key_type = typed_key.expr_type;
        let value_type = typed_value.expr_type;

        // Restore the previous scope
        self.context.current_scope = previous_scope;

        // Create the map type
        let map_type = self
            .context
            .type_table
            .borrow_mut()
            .create_map_type(key_type, value_type);

        Ok(TypedExpression {
            expr_type: map_type,
            kind: TypedExpressionKind::MapComprehension {
                for_parts: typed_for_parts,
                key_expr: Box::new(typed_key),
                value_expr: Box::new(typed_value),
                key_type,
                value_type,
            },
            usage: VariableUsage::Copy,
            lifetime_id: crate::tast::LifetimeId::first(),
            source_location: location.clone(),
            metadata: ExpressionMetadata::default(),
        })
    }

    /// Convert for-in loop to iterator-based while loop for TAST compatibility
    ///
    /// Haxe for-in loops use the iterator pattern:
    /// ```haxe
    /// for (x in iterable) { body }
    /// // becomes:
    /// var __iter = iterable.iterator();
    /// while (__iter.hasNext()) {
    ///     var x = __iter.next();
    ///     body;
    /// }
    /// ```
    fn convert_for_in_to_c_style_for(
        &mut self,
        variable: SymbolId,
        key_variable: Option<SymbolId>,
        iterable: TypedExpression,
        body: TypedStatement,
        loop_body_scope_id: ScopeId,
        source_location: SourceLocation,
    ) -> LoweringResult<TypedStatement> {
        // Infer element type from the iterable
        let element_type = self.infer_element_type_from_iterable(&iterable);
        let bool_type = self.context.type_table.borrow().bool_type();
        let int_type = self.context.type_table.borrow().int_type();

        // Create iterator type (Dynamic for now, should be Iterator<T> or KeyValueIterator<K,V>)
        let iter_type = self.context.type_table.borrow().dynamic_type();

        // Create: var __iter = iterable.iterator() or iterable.keyValueIterator()
        let iter_str = self.context.intern_string("__iter");
        let iterator_symbol = self.context.symbol_table.create_variable(iter_str);

        // Choose iterator method based on whether we have key-value iteration
        let iterator_method_name = if key_variable.is_some() {
            self.context.intern_string("keyValueIterator")
        } else {
            self.context.intern_string("iterator")
        };
        let iterator_method_symbol = self
            .context
            .symbol_table
            .create_variable(iterator_method_name);

        let iterator_call = TypedExpression {
            expr_type: iter_type,
            kind: TypedExpressionKind::MethodCall {
                receiver: Box::new(iterable),
                method_symbol: iterator_method_symbol,
                arguments: vec![],
                type_arguments: vec![],
                is_optional: false,
            },
            usage: VariableUsage::Move,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location,
            metadata: ExpressionMetadata::default(),
        };

        let init_stmt = TypedStatement::VarDeclaration {
            symbol_id: iterator_symbol,
            var_type: iter_type,
            initializer: Some(iterator_call),
            source_location,
            mutability: crate::tast::Mutability::Mutable,
        };

        // Create iterator variable reference
        let iterator_var = TypedExpression {
            expr_type: iter_type,
            kind: TypedExpressionKind::Variable {
                symbol_id: iterator_symbol,
            },
            usage: VariableUsage::Copy,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location,
            metadata: ExpressionMetadata::default(),
        };

        // Create condition: __iter.hasNext()
        let has_next_name = self.context.intern_string("hasNext");
        let has_next_symbol = self.context.symbol_table.create_variable(has_next_name);
        let condition = TypedExpression {
            expr_type: bool_type,
            kind: TypedExpressionKind::MethodCall {
                receiver: Box::new(iterator_var.clone()),
                method_symbol: has_next_symbol,
                arguments: vec![],
                type_arguments: vec![],
                is_optional: false,
            },
            usage: VariableUsage::Copy,
            lifetime_id: LifetimeId::static_lifetime(),
            source_location,
            metadata: ExpressionMetadata::default(),
        };

        // Create loop body statements
        let mut loop_statements = Vec::new();

        if let Some(key_sym) = key_variable {
            // Key-value iteration: for (key => value in iterable)
            // var __pair = __iter.next();
            // var key = __pair.key;
            // var value = __pair.value;

            let pair_type = self.context.type_table.borrow().dynamic_type();
            let pair_str = self.context.intern_string("__pair");
            let pair_symbol = self.context.symbol_table.create_variable(pair_str);

            let next_name = self.context.intern_string("next");
            let next_symbol = self.context.symbol_table.create_variable(next_name);
            let next_call = TypedExpression {
                expr_type: pair_type,
                kind: TypedExpressionKind::MethodCall {
                    receiver: Box::new(iterator_var),
                    method_symbol: next_symbol,
                    arguments: vec![],
                    type_arguments: vec![],
                    is_optional: false,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let pair_decl = TypedStatement::VarDeclaration {
                symbol_id: pair_symbol,
                var_type: pair_type,
                initializer: Some(next_call),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(pair_decl);

            // Create pair variable reference
            let pair_var = TypedExpression {
                expr_type: pair_type,
                kind: TypedExpressionKind::Variable {
                    symbol_id: pair_symbol,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            // var key = __pair.key
            let key_field_name = self.context.intern_string("key");
            let key_field_symbol = self.context.symbol_table.create_variable(key_field_name);
            let key_access = TypedExpression {
                expr_type: int_type, // Keys are typically Int for arrays, could be other types for maps
                kind: TypedExpressionKind::FieldAccess {
                    object: Box::new(pair_var.clone()),
                    field_symbol: key_field_symbol,
                    is_optional: false,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let key_decl = TypedStatement::VarDeclaration {
                symbol_id: key_sym,
                var_type: int_type,
                initializer: Some(key_access),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(key_decl);

            // var value = __pair.value
            let value_field_name = self.context.intern_string("value");
            let value_field_symbol = self.context.symbol_table.create_variable(value_field_name);
            let value_access = TypedExpression {
                expr_type: element_type,
                kind: TypedExpressionKind::FieldAccess {
                    object: Box::new(pair_var),
                    field_symbol: value_field_symbol,
                    is_optional: false,
                },
                usage: VariableUsage::Copy,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let value_decl = TypedStatement::VarDeclaration {
                symbol_id: variable,
                var_type: element_type,
                initializer: Some(value_access),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(value_decl);
        } else {
            // Simple iteration: for (value in iterable)
            // var value = __iter.next()
            let next_name = self.context.intern_string("next");
            let next_symbol = self.context.symbol_table.create_variable(next_name);
            let next_call = TypedExpression {
                expr_type: element_type,
                kind: TypedExpressionKind::MethodCall {
                    receiver: Box::new(iterator_var),
                    method_symbol: next_symbol,
                    arguments: vec![],
                    type_arguments: vec![],
                    is_optional: false,
                },
                usage: VariableUsage::Move,
                lifetime_id: LifetimeId::static_lifetime(),
                source_location,
                metadata: ExpressionMetadata::default(),
            };

            let value_decl = TypedStatement::VarDeclaration {
                symbol_id: variable,
                var_type: element_type,
                initializer: Some(next_call),
                source_location,
                mutability: crate::tast::Mutability::Mutable,
            };
            loop_statements.push(value_decl);
        }

        // Add the loop body
        loop_statements.push(body);

        // Create while body block
        let while_body = TypedStatement::Block {
            statements: loop_statements,
            scope_id: loop_body_scope_id,
            source_location,
        };

        // Create while loop: while (__iter.hasNext()) { ... }
        let while_stmt = TypedStatement::While {
            condition,
            body: Box::new(while_body),
            source_location,
        };

        // Return block: { var __iter = iterable.iterator(); while (__iter.hasNext()) { ... } }
        Ok(TypedStatement::Block {
            statements: vec![init_stmt, while_stmt],
            scope_id: ScopeId::from_raw(self.context.next_scope_id()),
            source_location,
        })
    }

    /// Infer the element type from an iterable expression (array, map, etc.)
    fn infer_element_type_from_iterable(&self, iterable: &TypedExpression) -> TypeId {
        let (kind, dynamic) = {
            let type_table = self.context.type_table.borrow();
            (
                type_table.get(iterable.expr_type).map(|t| t.kind.clone()),
                type_table.dynamic_type(),
            )
        };
        match kind {
            Some(TypeKind::Array { element_type }) => element_type,
            Some(TypeKind::Map { value_type, .. }) => value_type,
            // A structure declaring `next()` is an iterator and one declaring
            // `iterator()` yields one; either way the element is what `next()`
            // answers. A class reaches neither and stays Dynamic, as before.
            // A bare parameter is kept Dynamic, as it was: a local typed by it
            // makes the enclosing template a stub until instantiated.
            Some(TypeKind::TypeAlias { .. })
            | Some(TypeKind::Class { .. })
            | Some(TypeKind::Anonymous { .. }) => self
                .structural_element_type(iterable.expr_type)
                .filter(|t| !self.is_bare_type_param(*t))
                .unwrap_or(dynamic),
            _ => dynamic,
        }
    }

    fn is_bare_type_param(&self, ty: TypeId) -> bool {
        matches!(
            self.context.type_table.borrow().get(ty).map(|t| &t.kind),
            Some(TypeKind::TypeParameter { .. }) | Some(TypeKind::Placeholder { .. })
        )
    }

    fn structural_element_type(&self, ty: TypeId) -> Option<TypeId> {
        let next = self.context.string_interner.intern("next");
        if let Some(elem) = self.structural_method_return_type(ty, next) {
            return Some(elem);
        }
        let iterator = self.context.string_interner.intern("iterator");
        let it = self.structural_method_return_type(ty, iterator)?;
        self.structural_method_return_type(it, next)
    }

    /// Determine variable usage based on expression kind (simplified for TAST)
    pub(crate) fn determine_variable_usage(&self, kind: &TypedExpressionKind) -> VariableUsage {
        match kind {
            TypedExpressionKind::Literal { .. } => {
                // Literals are always copyable
                VariableUsage::Copy
            }
            _ => {
                // Default to copy for TAST - ownership analysis happens in semantic graph
                VariableUsage::Copy
            }
        }
    }
}
