//! Type inference for expressions and calls.

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
    /// Infer the element and optional key types from an iterator expression
    pub(crate) fn infer_iterator_types(
        &mut self,
        iterator: &TypedExpression,
    ) -> LoweringResult<(TypeId, Option<TypeId>)> {
        let type_table = self.context.type_table.borrow();

        match type_table.get(iterator.expr_type) {
            Some(iter_type) => match &iter_type.kind {
                // Array<T> -> element type T, no key type
                crate::tast::core::TypeKind::Array { element_type } => Ok((*element_type, None)),
                // Map<K, V> -> value type V, key type K
                crate::tast::core::TypeKind::Map {
                    key_type,
                    value_type,
                } => Ok((*value_type, Some(*key_type))),
                // String -> Char elements, Int keys
                _ if iter_type.kind == crate::tast::core::TypeKind::String => {
                    // In Haxe, iterating over strings yields characters
                    let int_type = type_table.int_type();
                    drop(type_table);
                    let char_type = self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_type(crate::tast::core::TypeKind::Char);
                    Ok((char_type, Some(int_type)))
                }
                // IntIterator (from range expressions like 0...10)
                _ if self.is_int_iterator_type(iterator.expr_type) => {
                    Ok((type_table.int_type(), None))
                }
                // Dynamic or unknown - default to dynamic element type
                _ => Ok((type_table.dynamic_type(), None)),
            },
            None => Ok((type_table.dynamic_type(), None)),
        }
    }

    /// Infer the iterator type name to load based on the iterable expression type
    /// Returns the qualified name of the iterator class to load (e.g., "haxe.iterators.ArrayIterator")
    fn infer_iterator_type_name(&self, type_id: &TypeId) -> Option<String> {
        let type_table = self.context.type_table.borrow();

        if let Some(actual_type) = type_table.get(*type_id) {
            match &actual_type.kind {
                TypeKind::Array { .. } => {
                    // Arrays use ArrayIterator from haxe.iterators
                    Some("haxe.iterators.ArrayIterator".to_string())
                }
                TypeKind::Map { .. } => {
                    // Maps might use a MapIterator (implementation dependent)
                    Some("haxe.iterators.MapIterator".to_string())
                }
                TypeKind::Class { symbol_id, .. } => {
                    // Check the class name to determine iterator type
                    if let Some(class_symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                        if let Some(class_name) =
                            self.context.string_interner.get(class_symbol.name)
                        {
                            match class_name {
                                "Array" => Some("haxe.iterators.ArrayIterator".to_string()),
                                "String" => Some("haxe.iterators.StringIterator".to_string()),
                                "IntIterator" => Some("haxe.iterators.IntIterator".to_string()),
                                _ => {
                                    // For other classes, try to infer from package
                                    // Could also look at implemented interfaces
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    /// Infer the type of an expression based on its kind
    pub(crate) fn infer_expression_type(
        &mut self,
        kind: &TypedExpressionKind,
    ) -> LoweringResult<TypeId> {
        match kind {
            TypedExpressionKind::Literal { value } => {
                let type_table = self.context.type_table.borrow();
                match value {
                    LiteralValue::Bool(_) => Ok(type_table.bool_type()),
                    LiteralValue::Int(_) => Ok(type_table.int_type()),
                    LiteralValue::Float(_) => Ok(type_table.float_type()),
                    LiteralValue::String(_) => Ok(type_table.string_type()),
                    LiteralValue::Char(_) => Ok(type_table.string_type()), // Haxe treats char as string
                    LiteralValue::Regex(_) | LiteralValue::RegexWithFlags { .. } => {
                        // EReg type in Haxe — resolve as proper class type
                        drop(type_table);
                        let ereg_name = self.context.string_interner.intern("EReg");
                        if let Some(symbol_id) = self.resolve_symbol_in_scope_hierarchy(ereg_name) {
                            if let Some(symbol) = self.context.symbol_table.get_symbol(symbol_id) {
                                let tid = symbol.type_id;
                                if tid != TypeId::invalid() {
                                    return Ok(tid);
                                }
                                // EReg placeholder exists but has no TypeId yet —
                                // create a proper Class type and link it to the symbol.
                                let ereg_type = self.context.type_table.borrow_mut().create_type(
                                    crate::tast::core::TypeKind::Class {
                                        symbol_id,
                                        type_args: Vec::new(),
                                    },
                                );
                                self.context
                                    .symbol_table
                                    .update_symbol_type(symbol_id, ereg_type);
                                return Ok(ereg_type);
                            }
                        }
                        // Fallback to structural type
                        Ok(type_resolution::get_regex_type(
                            &self.context.type_table,
                            self.context.string_interner,
                        ))
                    }
                }
            }
            TypedExpressionKind::Variable { symbol_id } => {
                // Look up the symbol's type
                if let Some(symbol) = self.context.symbol_table.get_symbol(*symbol_id) {
                    Ok(symbol.type_id)
                } else {
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
            } => {
                let type_table = self.context.type_table.borrow();

                // OPERATOR OVERLOADING: when the LHS is a user-defined class or abstract
                // type and the operator is one of the arithmetic/comparison ones, the
                // result type is the LHS type (per @:op semantics on Tensor, SIMD4f, …).
                // Without this we'd default to Float below and silently miscompile —
                // `var t = a + b` would type `t` as Float, then `t.sum()` dispatches on
                // the wrong type. This handles the symmetric-typed ops; asymmetric ops
                // are a future extension if needed.
                // String concatenation wins over the @:op rules below: `"x" + v`
                // is a String no matter what `v` is. Without this the abstract
                // arms claim it — `"s=" + aSingle` typed as Single, so the MIR
                // called the concat (returning a String pointer) and then tried
                // to cast that pointer float→float.
                if matches!(operator, BinaryOperator::Add) {
                    let string_type = type_table.string_type();
                    if left.expr_type == string_type || right.expr_type == string_type {
                        return Ok(string_type);
                    }
                }

                if matches!(
                    operator,
                    BinaryOperator::Add
                        | BinaryOperator::Sub
                        | BinaryOperator::Mul
                        | BinaryOperator::Div
                        | BinaryOperator::Mod
                ) {
                    let lhs_is_user_type = type_table
                        .get(left.expr_type)
                        .map(|t| {
                            matches!(
                                t.kind,
                                crate::tast::core::TypeKind::Class { .. }
                                    | crate::tast::core::TypeKind::Abstract { .. }
                            )
                        })
                        .unwrap_or(false);
                    if lhs_is_user_type {
                        return Ok(left.expr_type);
                    }
                    // Symmetric case: `<primitive> + <abstract>`. A stdlib-typed
                    // method whose return is an abstract can read back as its
                    // primitive (e.g. `Bytes.address()` → an integer, not `Usize`),
                    // so the abstract lands on the RHS instead. The @:op result is
                    // still the abstract type; without this the arithmetic default
                    // below hits `else => Float` and a `Usize` address silently
                    // decays to Float — every use then coerces the i64 pointer
                    // i64→f64→i64 (sitofp/bitcast), destroying it → SIGSEGV.
                    let rhs_is_user_type = type_table
                        .get(right.expr_type)
                        .map(|t| {
                            matches!(
                                t.kind,
                                crate::tast::core::TypeKind::Class { .. }
                                    | crate::tast::core::TypeKind::Abstract { .. }
                            )
                        })
                        .unwrap_or(false);
                    if rhs_is_user_type {
                        return Ok(right.expr_type);
                    }
                }

                match operator {
                    BinaryOperator::Add => {
                        // Add can be either string concatenation or numeric addition
                        let left_type = left.expr_type;
                        let right_type = right.expr_type;
                        let dynamic_type = type_table.dynamic_type();
                        let string_type = type_table.string_type();
                        let int_type = type_table.int_type();
                        let float_type = type_table.float_type();

                        // If either operand is Dynamic, result is Dynamic
                        if left_type == dynamic_type || right_type == dynamic_type {
                            Ok(dynamic_type)
                        }
                        // If either operand is string, result is string (concatenation)
                        else if left_type == string_type || right_type == string_type {
                            Ok(string_type)
                        }
                        // If either operand is Float, result is Float
                        else if left_type == float_type || right_type == float_type {
                            Ok(float_type)
                        }
                        // If both are Int, result is Int
                        else if left_type == int_type && right_type == int_type {
                            Ok(int_type)
                        }
                        // Default to Float for safety
                        else {
                            Ok(float_type)
                        }
                    }
                    BinaryOperator::Sub
                    | BinaryOperator::Mul
                    | BinaryOperator::Div
                    | BinaryOperator::Mod => {
                        // Purely numeric operations
                        let left_type = left.expr_type;
                        let right_type = right.expr_type;
                        let dynamic_type = type_table.dynamic_type();

                        // If either operand is Dynamic, result is Dynamic
                        if left_type == dynamic_type || right_type == dynamic_type {
                            Ok(dynamic_type)
                        }
                        // If either operand is Float, result is Float
                        else if left_type == type_table.float_type()
                            || right_type == type_table.float_type()
                        {
                            Ok(type_table.float_type())
                        }
                        // If both are Int, result is Int (except for division which returns Float)
                        else if left_type == type_table.int_type()
                            && right_type == type_table.int_type()
                        {
                            match operator {
                                BinaryOperator::Div => Ok(type_table.float_type()), // Division always returns Float in Haxe
                                _ => Ok(type_table.int_type()),
                            }
                        }
                        // Default to Float for safety
                        else {
                            Ok(type_table.float_type())
                        }
                    }
                    // Bitwise and shift operators are Int -> Int in Haxe.
                    // Without this arm they fell to the `_ => Dynamic`
                    // catch-all below, so `var r = x >> 13` typed `r` as
                    // Dynamic; a later `if (c) r = 0` then BOXED the constant
                    // (haxe_box_int_ptr) and the merge returned the box
                    // POINTER truncated to i32 — wrong values plus a 48-byte
                    // leak per assignment.
                    BinaryOperator::BitAnd
                    | BinaryOperator::BitOr
                    | BinaryOperator::BitXor
                    | BinaryOperator::Shl
                    | BinaryOperator::Shr
                    | BinaryOperator::Ushr => {
                        let left_type = left.expr_type;
                        let right_type = right.expr_type;
                        let dynamic_type = type_table.dynamic_type();
                        // A user Class/Abstract operand carries an @:op
                        // overload whose result is that type (e.g. masking a
                        // Usize address), so it wins over the Int default.
                        let is_user_type = |t: TypeId| {
                            type_table
                                .get(t)
                                .map(|x| {
                                    matches!(
                                        x.kind,
                                        crate::tast::core::TypeKind::Class { .. }
                                            | crate::tast::core::TypeKind::Abstract { .. }
                                    )
                                })
                                .unwrap_or(false)
                        };
                        if is_user_type(left_type) {
                            Ok(left_type)
                        } else if is_user_type(right_type) {
                            Ok(right_type)
                        } else if left_type == dynamic_type || right_type == dynamic_type {
                            // Same convention as the arithmetic arm above.
                            Ok(dynamic_type)
                        } else {
                            Ok(type_table.int_type())
                        }
                    }
                    BinaryOperator::Eq
                    | BinaryOperator::Ne
                    | BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge => Ok(type_table.bool_type()),
                    BinaryOperator::And | BinaryOperator::Or => Ok(type_table.bool_type()),
                    BinaryOperator::Assign
                    | BinaryOperator::AddAssign
                    | BinaryOperator::SubAssign
                    | BinaryOperator::MulAssign
                    | BinaryOperator::DivAssign
                    | BinaryOperator::ModAssign => {
                        // Assignment returns the type of the left operand
                        Ok(left.expr_type)
                    }
                    BinaryOperator::NullCoal => {
                        // Null coalescing: result type is the LHS type (non-null version)
                        Ok(left.expr_type)
                    }
                    _ => Ok(type_table.dynamic_type()),
                }
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                // Extract return type from function signature
                let type_table = self.context.type_table.borrow();
                match type_table.get(function.expr_type) {
                    Some(function_type) => match &function_type.kind {
                        crate::tast::core::TypeKind::Function {
                            params,
                            return_type,
                            ..
                        } => {
                            let ret = *return_type;
                            // If return type is a TypeParameter, infer concrete type from arguments
                            if type_table.is_type_parameter(ret) {
                                for (i, param_ty) in params.iter().enumerate() {
                                    if *param_ty == ret && i < arguments.len() {
                                        return Ok(arguments[i].expr_type);
                                    }
                                }
                            }
                            Ok(ret)
                        }
                        _ => Ok(type_table.dynamic_type()),
                    },
                    None => Ok(type_table.dynamic_type()),
                }
            }
            TypedExpressionKind::New { class_type, .. } => Ok(*class_type),
            TypedExpressionKind::ArrayLiteral { elements } => {
                if let Some(first_element) = elements.first() {
                    let element_type = first_element.expr_type;
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(element_type))
                } else {
                    let dyn_type = self.context.type_table.borrow().dynamic_type();
                    Ok(self
                        .context
                        .type_table
                        .borrow_mut()
                        .create_array_type(dyn_type))
                }
            }
            TypedExpressionKind::Null => {
                Ok(type_resolution::get_null_type(&self.context.type_table))
            }
            TypedExpressionKind::This { this_type } => Ok(*this_type),
            TypedExpressionKind::Super { super_type } => Ok(*super_type),
            TypedExpressionKind::ObjectLiteral { fields } => {
                // For anonymous objects, infer type from fields
                let field_types: Vec<(InternedString, TypeId)> =
                    fields.iter().map(|f| (f.name, f.value.expr_type)).collect();
                Ok(type_resolution::create_anonymous_object_type(
                    &self.context.type_table,
                    field_types,
                ))
            }
            TypedExpressionKind::StringInterpolation { .. } => {
                Ok(self.context.type_table.borrow().string_type())
            }
            TypedExpressionKind::Cast { target_type, .. } => Ok(*target_type),
            TypedExpressionKind::Is { .. } => Ok(self.context.type_table.borrow().bool_type()),
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                // PRIORITY: when the receiver is an anonymous type, read the
                // field type directly from the receiver's shape. The
                // symbol-table lookup below uses the field's NAME symbol
                // (shared across all anonymous types that use this name),
                // and its type_id is whichever anon happened to last set
                // it — typically a narrow/empty shape from another
                // literal. That stale type then propagates to bindings
                // (`var x = q.child`), and downstream `x.field` reads
                // miss the slot and return null. Reading the shape
                // directly off the receiver dodges the collision.
                let want_name = self
                    .context
                    .symbol_table
                    .get_symbol(*field_symbol)
                    .map(|s| s.name);
                if let Some(name) = want_name {
                    // Walk through TypeAlias chains so typedef-wrapped
                    // anons (`typedef Inner = {x:Int}`) participate in
                    // this lookup too.
                    let mut current = object.expr_type;
                    let mut hops = 0;
                    while hops < 8 {
                        let kind = {
                            let type_table = self.context.type_table.borrow();
                            type_table.get(current).map(|t| t.kind.clone())
                        };
                        let Some(kind) = kind else { break };
                        match &kind {
                            crate::tast::core::TypeKind::Anonymous { fields } => {
                                if let Some(field) = fields.iter().find(|f| f.name == name) {
                                    return Ok(field.type_id);
                                }
                                break;
                            }
                            crate::tast::core::TypeKind::TypeAlias { target_type, .. } => {
                                current = *target_type;
                                hops += 1;
                            }
                            crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                                // Cross-module decay: this Class may actually be a
                                // structural typedef whose real Anonymous target
                                // lives elsewhere in the shared type table (see
                                // find_typedef_anonymous_target_by_qname). Recover
                                // it by qualified name before giving up.
                                let qname = self
                                    .context
                                    .symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|s| {
                                        s.qualified_name
                                            .and_then(|n| self.context.string_interner.get(n))
                                    })
                                    .map(|s| s.to_string());
                                if let Some(qname) = qname {
                                    if let Some(anon_ty) =
                                        self.find_typedef_anonymous_target_by_qname(&qname)
                                    {
                                        current = anon_ty;
                                        hops += 1;
                                        continue;
                                    }
                                }
                                break;
                            }
                            _ => break,
                        }
                    }
                }
                // Look up field type from symbol table
                if let Some(symbol) = self.context.symbol_table.get_symbol(*field_symbol) {
                    // Check if this is a valid typed symbol
                    if symbol.type_id.is_valid() {
                        Ok(symbol.type_id)
                    } else {
                        // Handle built-in method access for Array, String, etc.
                        self.infer_builtin_method_type(object.expr_type, *field_symbol)
                    }
                } else {
                    // Handle built-in method access for Array, String, etc.
                    self.infer_builtin_method_type(object.expr_type, *field_symbol)
                }
            }
            TypedExpressionKind::ArrayAccess { array, .. } => {
                // Extract element type from array type. For a class/abstract
                // receiver (e.g. an @:arrayAccess extern abstract like
                // SIMD16i8), resolve the element type from its `get` accessor
                // instead of collapsing to Dynamic — a Dynamic-typed read
                // boxes the extracted scalar (`cast i8 -> *void`) and poisons
                // all downstream arithmetic typing.
                let accessor_sym = {
                    let type_table = self.context.type_table.borrow();
                    match type_table.get(array.expr_type).map(|t| &t.kind) {
                        Some(crate::tast::core::TypeKind::Array { element_type }) => {
                            return Ok(*element_type);
                        }
                        Some(crate::tast::core::TypeKind::Map { value_type, .. }) => {
                            // `m[k]` on a Map literal resolves to the value type.
                            return Ok(*value_type);
                        }
                        Some(crate::tast::core::TypeKind::Class {
                            symbol_id,
                            type_args,
                            ..
                        }) => {
                            // haxe.ds.IntMap/StringMap/ObjectMap are extern classes
                            // whose `get` yields the value type argument. Match on the
                            // qualified name so a user class named `IntMap` cannot collide.
                            let class_name = self
                                .context
                                .symbol_table
                                .get_symbol(*symbol_id)
                                .and_then(|s| s.qualified_name)
                                .and_then(|n| self.context.string_interner.get(n));
                            match class_name {
                                Some("haxe.ds.IntMap") | Some("haxe.ds.StringMap") => {
                                    return Ok(type_args
                                        .first()
                                        .copied()
                                        .unwrap_or_else(|| type_table.dynamic_type()));
                                }
                                Some("haxe.ds.ObjectMap") => {
                                    return Ok(type_args
                                        .get(1)
                                        .copied()
                                        .unwrap_or_else(|| type_table.dynamic_type()));
                                }
                                _ => Some(*symbol_id),
                            }
                        }
                        Some(crate::tast::core::TypeKind::Abstract { symbol_id, .. }) => {
                            Some(*symbol_id)
                        }
                        _ => None,
                    }
                };
                if let Some(class_sym) = accessor_sym {
                    if let Some(get_sym) = self.find_wrapper_get_method(class_sym) {
                        if let Ok(ret) =
                            self.infer_method_call_return_type(get_sym, array.expr_type)
                        {
                            if ret.is_valid() {
                                return Ok(ret);
                            }
                        }
                    }
                }
                Ok(self.context.type_table.borrow().dynamic_type())
            }
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                ..
            } => {
                // Extract return type from method signature and substitute type parameters
                self.infer_method_call_return_type(*method_symbol, receiver.expr_type)
            }
            TypedExpressionKind::StaticMethodCall {
                method_symbol,
                arguments,
                ..
            } => {
                // Extract return type from static method signature
                if let Some(symbol) = self.context.symbol_table.get_symbol(*method_symbol) {
                    let type_table = self.context.type_table.borrow();
                    if let Some(method_type) = type_table.get(symbol.type_id) {
                        match &method_type.kind {
                            crate::tast::core::TypeKind::Function {
                                params,
                                return_type,
                                ..
                            } => {
                                let ret = *return_type;
                                // If return type is a TypeParameter, infer from arguments
                                if type_table.is_type_parameter(ret) {
                                    for (i, param_ty) in params.iter().enumerate() {
                                        if *param_ty == ret && i < arguments.len() {
                                            return Ok(arguments[i].expr_type);
                                        }
                                    }
                                }
                                Ok(ret)
                            }
                            _ => Ok(type_table.dynamic_type()),
                        }
                    } else {
                        Ok(type_table.dynamic_type())
                    }
                } else {
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
            TypedExpressionKind::UnaryOp { operator, operand } => {
                let type_table = self.context.type_table.borrow();
                match operator {
                    UnaryOperator::Not => Ok(type_table.bool_type()),
                    UnaryOperator::Neg | UnaryOperator::BitNot => Ok(operand.expr_type),
                    UnaryOperator::PreInc
                    | UnaryOperator::PostInc
                    | UnaryOperator::PreDec
                    | UnaryOperator::PostDec => Ok(operand.expr_type),
                }
            }
            TypedExpressionKind::Conditional {
                then_expr,
                else_expr,
                ..
            } => {
                // Type unification handled by type checker
                Ok(then_expr.expr_type)
            }
            TypedExpressionKind::While { .. }
            | TypedExpressionKind::For { .. }
            | TypedExpressionKind::ForIn { .. } => Ok(self.context.type_table.borrow().void_type()),
            TypedExpressionKind::FunctionLiteral {
                parameters,
                return_type,
                ..
            } => {
                // Create a function type: (param1_type, param2_type, ...) -> return_type
                let param_types: Vec<TypeId> = parameters.iter().map(|p| p.param_type).collect();
                Ok(self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_function_type(param_types, *return_type))
            }
            TypedExpressionKind::Return { .. } => Ok(self.context.type_table.borrow().void_type()),
            TypedExpressionKind::Throw { .. } => Ok(self.context.type_table.borrow().void_type()),
            TypedExpressionKind::Break | TypedExpressionKind::Continue => {
                Ok(self.context.type_table.borrow().void_type())
            }
            TypedExpressionKind::Block { statements, .. } => {
                // Block type is the type of the last expression
                let type_table = self.context.type_table.borrow();
                if let Some(last_stmt) = statements.last() {
                    // Extract type from last statement
                    match last_stmt {
                        TypedStatement::Expression { expression, .. } => Ok(expression.expr_type),
                        _ => {
                            // Non-expression statements result in void type
                            Ok(type_table.void_type())
                        }
                    }
                } else {
                    // Empty block has void type
                    Ok(type_table.void_type())
                }
            }
            TypedExpressionKind::Meta { expression, .. } => Ok(expression.expr_type),
            TypedExpressionKind::DollarIdent { .. } => {
                Ok(self.context.type_table.borrow().dynamic_type()) // Macro-related
            }
            TypedExpressionKind::CompilerSpecific { .. } => {
                Ok(self.context.type_table.borrow().dynamic_type())
            }
            TypedExpressionKind::Switch {
                cases,
                default_case,
                ..
            } => {
                // Switch expression type is inferred from the branches
                // All branches should have compatible types
                let mut branch_types = Vec::new();

                // Collect types from case branches
                for case in cases {
                    // Extract expression type from the case body statement
                    match &case.body {
                        TypedStatement::Expression { expression, .. } => {
                            branch_types.push(expression.expr_type);
                            // eprintln!(
                            //     "DEBUG: Switch case expression type: {:?}",
                            //     expression.expr_type
                            // );

                            // For switch expressions (not statements), check for void type
                            // But only if this is truly a switch expression context
                            // TODO: Properly distinguish between switch expressions and statements
                        }
                        _ => {
                            // Non-expression statements in switch expression context
                            // This shouldn't happen for valid switch expressions
                            self.context.errors.push(LoweringError::InternalError {
                                message: "Switch expression case must be an expression".to_string(),
                                location: case.source_location,
                            });
                        }
                    }
                }

                // Add default case type if present
                if let Some(default) = default_case {
                    branch_types.push(default.expr_type);
                    // eprintln!(
                    //     "DEBUG: Switch default expression type: {:?}",
                    //     default.expr_type
                    // );

                    // For switch expressions (not statements), check for void type in default
                    // But only if this is truly a switch expression context
                    // TODO: Properly distinguish between switch expressions and statements
                }

                // If no branches have expressions, it's a void switch
                if branch_types.is_empty() {
                    return Ok(self.context.type_table.borrow().void_type());
                }

                // Filter out void types and use the first non-void type for expression result
                let void_type = self.context.type_table.borrow().void_type();
                let non_void_types: Vec<TypeId> = branch_types
                    .iter()
                    .filter(|&&t| t != void_type)
                    .copied()
                    .collect();

                if non_void_types.is_empty() {
                    // All branches are void, return void (but errors should have been generated above)
                    return Ok(void_type);
                }

                // For now, use the first non-void branch type
                // Type unification deferred to type checker
                // eprintln!(
                //     "DEBUG: Switch expression inferred type: {:?}",
                //     non_void_types[0]
                // );
                Ok(non_void_types[0])
            }
            TypedExpressionKind::Try { try_expr, .. } => {
                // Try expression type is the type of the try block
                Ok(try_expr.expr_type)
            }
            TypedExpressionKind::VarDeclarationExpr { var_type, .. } => Ok(*var_type),
            TypedExpressionKind::FinalDeclarationExpr { var_type, .. } => Ok(*var_type),
            TypedExpressionKind::MapLiteral { entries } => {
                // Infer key and value types from initial values
                let (key_type, value_type) = if entries.is_empty() {
                    let dyn_type = self.context.type_table.borrow().dynamic_type();
                    (dyn_type, dyn_type)
                } else {
                    // Use first entry to infer types
                    let first = &entries[0];
                    (first.key.expr_type, first.value.expr_type)
                };
                Ok(type_resolution::create_map_type(
                    &self.context.type_table,
                    key_type,
                    value_type,
                ))
            }
            TypedExpressionKind::MacroExpression { .. } => {
                Ok(self.context.type_table.borrow().dynamic_type()) // Macro result type
            }
            TypedExpressionKind::ArrayComprehension { element_type, .. } => {
                // Array comprehension creates an Array<T> type
                Ok(self
                    .context
                    .type_table
                    .borrow_mut()
                    .create_array_type(*element_type))
            }
            TypedExpressionKind::MapComprehension {
                key_type,
                value_type,
                ..
            } => {
                // Map comprehension creates a Map<K, V> type
                Ok(type_resolution::create_map_type(
                    &self.context.type_table,
                    *key_type,
                    *value_type,
                ))
            }
            TypedExpressionKind::FunctionCall { function, .. } => {
                // Extract return type from the function's type
                // For enum constructors, the function has a Function type where return_type is the enum type
                let type_table = self.context.type_table.borrow();
                if let Some(func_type) = type_table.get(function.expr_type) {
                    match &func_type.kind {
                        crate::tast::core::TypeKind::Function { return_type, .. } => {
                            Ok(*return_type)
                        }
                        // If it's already an enum type (for simple enum variants), use it directly
                        crate::tast::core::TypeKind::Enum { .. } => Ok(function.expr_type),
                        _ => Ok(type_table.dynamic_type()),
                    }
                } else {
                    Ok(type_table.dynamic_type())
                }
            }
            TypedExpressionKind::StaticFieldAccess { field_symbol, .. } => {
                // Look up the field symbol's type from the symbol table
                if let Some(sym) = self.context.symbol_table.get_symbol(*field_symbol) {
                    Ok(sym.type_id)
                } else {
                    Ok(self.context.type_table.borrow().dynamic_type())
                }
            }
            _ => {
                // Log unhandled case as warning but continue with dynamic type
                self.context
                    .add_error(LoweringError::IncompleteImplementation {
                        feature: format!("Type inference for expression kind: {:?}", kind),
                        location: self.context.create_location(),
                    });
                Ok(self.context.type_table.borrow().dynamic_type())
            }
        }
    }

    /// Infer the type of built-in methods like Array.push, String.charAt, etc.
    pub(crate) fn infer_builtin_method_type(
        &mut self,
        receiver_type: TypeId,
        field_symbol: SymbolId,
    ) -> LoweringResult<TypeId> {
        // Get the field name from the symbol
        let field_name = if let Some(symbol) = self.context.symbol_table.get_symbol(field_symbol) {
            self.context
                .string_interner
                .get(symbol.name)
                .unwrap_or("<unknown>")
                .to_string()
        } else {
            return Ok(self.context.type_table.borrow().dynamic_type());
        };

        // Check the object type to see if it's a built-in type with known methods
        let type_table = self.context.type_table.borrow();
        if let Some(object_type_info) = type_table.get(receiver_type) {
            match &object_type_info.kind {
                crate::tast::core::TypeKind::Array { element_type } => {
                    match field_name.as_str() {
                        "push" => {
                            // push(item: T): Void
                            let void_type = type_table.void_type();
                            let element_type_copy = *element_type;
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![element_type_copy], void_type))
                        }
                        "pop" => {
                            // pop(): T
                            Ok(*element_type)
                        }
                        "length" => {
                            // length: Int
                            Ok(type_table.int_type())
                        }
                        "map" => {
                            // map(f: (T) -> S): Array<S>
                            // For now, return Array<T> (same element type as input)
                            // The actual return element type depends on the callback,
                            // but we preserve the array type so trace/dispatch works.
                            let elem = *element_type;
                            drop(type_table);
                            let arr_type =
                                self.context.type_table.borrow_mut().create_array_type(elem);
                            let func_type = {
                                let tt = self.context.type_table.borrow();
                                let callback_type = tt.dynamic_type();
                                drop(tt);
                                self.context
                                    .type_table
                                    .borrow_mut()
                                    .create_function_type(vec![callback_type], arr_type)
                            };
                            Ok(func_type)
                        }
                        "filter" => {
                            // filter(f: (T) -> Bool): Array<T>
                            let elem = *element_type;
                            drop(type_table);
                            let arr_type =
                                self.context.type_table.borrow_mut().create_array_type(elem);
                            let func_type = {
                                let tt = self.context.type_table.borrow();
                                let callback_type = tt.dynamic_type();
                                drop(tt);
                                self.context
                                    .type_table
                                    .borrow_mut()
                                    .create_function_type(vec![callback_type], arr_type)
                            };
                            Ok(func_type)
                        }
                        "sort" => {
                            // sort(f: (T, T) -> Int): Void
                            let void_type = type_table.void_type();
                            let callback_type = type_table.dynamic_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![callback_type], void_type))
                        }
                        "indexOf" | "lastIndexOf" => {
                            // indexOf(x: T, ?fromIndex: Int): Int
                            Ok(type_table.int_type())
                        }
                        "contains" => {
                            // contains(x: T): Bool
                            Ok(type_table.bool_type())
                        }
                        "join" => {
                            // join(sep: String): String
                            Ok(type_table.string_type())
                        }
                        "slice" | "splice" | "concat" | "copy" | "reverse" => {
                            // Returns Array<T>
                            let elem = *element_type;
                            drop(type_table);
                            Ok(self.context.type_table.borrow_mut().create_array_type(elem))
                        }
                        "remove" => {
                            // remove(x: T): Bool
                            Ok(type_table.bool_type())
                        }
                        "insert" | "unshift" => {
                            // insert(pos: Int, x: T): Void
                            Ok(type_table.void_type())
                        }
                        "toString" => Ok(type_table.string_type()),
                        "iterator" | "keyValueIterator" => Ok(type_table.dynamic_type()),
                        _ => Ok(type_table.dynamic_type()),
                    }
                }
                crate::tast::core::TypeKind::String => {
                    match field_name.as_str() {
                        "length" => Ok(type_table.int_type()),
                        "charAt" => {
                            // charAt(index: Int): String
                            let string_type = type_table.string_type();
                            let int_type = type_table.int_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![int_type], string_type))
                        }
                        "toUpperCase" | "toLowerCase" | "toString" | "trim" => {
                            // toUpperCase(): String, toLowerCase(): String, toString(): String, trim(): String
                            let string_type = type_table.string_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![], string_type))
                        }
                        "substring" | "substr" => {
                            // substring(startIndex: Int, ?endIndex: Int): String
                            let string_type = type_table.string_type();
                            let int_type = type_table.int_type();
                            drop(type_table);
                            // For simplicity, we'll create a function that takes two Int parameters
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![int_type, int_type], string_type))
                        }
                        "indexOf" | "lastIndexOf" => {
                            // indexOf(str: String, ?startIndex: Int): Int
                            let string_type = type_table.string_type();
                            let int_type = type_table.int_type();
                            drop(type_table);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![string_type], int_type))
                        }
                        "split" => {
                            // split(delimiter: String): Array<String>
                            let string_type = type_table.string_type();
                            drop(type_table);
                            let array_of_strings = self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_array_type(string_type);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![string_type], array_of_strings))
                        }
                        _ => Ok(type_table.dynamic_type()),
                    }
                }
                crate::tast::core::TypeKind::Abstract {
                    symbol_id,
                    underlying,
                    ..
                } => {
                    // For abstracts (including @:forward), resolve through underlying type
                    let resolved_underlying =
                        underlying.or_else(|| type_table.resolve_abstract_underlying(*symbol_id));
                    if let Some(underlying_type) = resolved_underlying {
                        drop(type_table);
                        self.infer_builtin_method_type(underlying_type, field_symbol)
                    } else {
                        Ok(type_table.dynamic_type())
                    }
                }
                crate::tast::core::TypeKind::GenericInstance { base_type, .. } => {
                    // For generic instances, resolve through base type
                    let base = *base_type;
                    drop(type_table);
                    self.infer_builtin_method_type(base, field_symbol)
                }
                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                    // For extern classes with known methods, provide proper return types
                    let class_name = self
                        .context
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|s| self.context.string_interner.get(s.name))
                        .unwrap_or("");
                    match (class_name, field_name.as_str()) {
                        ("EReg", "match" | "matchSub") => Ok(type_table.bool_type()),
                        ("EReg", "matched" | "matchedLeft" | "matchedRight" | "replace") => {
                            Ok(type_table.string_type())
                        }
                        ("EReg", "split") => {
                            let string_type = type_table.string_type();
                            drop(type_table);
                            let array_of_strings = self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_array_type(string_type);
                            Ok(self
                                .context
                                .type_table
                                .borrow_mut()
                                .create_function_type(vec![string_type], array_of_strings))
                        }
                        // `rayzor.Bytes` is an `extern class` (compiler/haxe-std/rayzor/Bytes.hx)
                        // reached via the `haxe.io.Bytes` typedef. Its method signatures'
                        // return types are NOT established as symbol type_ids on import, so a
                        // bound `sub`/`toString` symbol is typeless and decays here. Without
                        // these arms the `_` fallback returns Dynamic, and a chained
                        // `.toString()` then mis-dispatches to an unrelated class's toString.
                        // Infer the declared return types: `sub`/`slice` yield the same Bytes
                        // type as the receiver; `toString`/`getString` yield String.
                        ("Bytes" | "rayzor_Bytes" | "haxe_io_Bytes", "sub" | "slice") => {
                            Ok(receiver_type)
                        }
                        ("Bytes" | "rayzor_Bytes" | "haxe_io_Bytes", "toString" | "getString") => {
                            Ok(type_table.string_type())
                        }
                        // `free` returns Void (the native `haxe_bytes_free` binding
                        // takes no return value). Without this arm the `_` fallback
                        // types the call as Dynamic, so the caller allocates a
                        // result register the callee never fills.
                        ("Bytes" | "rayzor_Bytes" | "haxe_io_Bytes", "free") => {
                            Ok(type_table.void_type())
                        }
                        _ => Ok(type_table.dynamic_type()),
                    }
                }
                _ => Ok(type_table.dynamic_type()),
            }
        } else {
            Ok(type_table.dynamic_type())
        }
    }

    pub(crate) fn infer_return_type_from_body(&self, body: &[TypedStatement]) -> TypeId {
        // Look for return statements in the body
        for stmt in body {
            if let Some(return_type) = self.find_return_type_in_statement(stmt) {
                return return_type;
            }
        }
        // No return statements found, assume void
        self.context.type_table.borrow().void_type()
    }
}
