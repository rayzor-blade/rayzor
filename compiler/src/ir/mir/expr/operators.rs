//! Unary and binary operators.

use super::*;
use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::{
    BinaryOp, CallingConvention, CompareOp, EnvironmentLayout, FunctionKind,
    FunctionSignatureBuilder, IrBasicBlock, IrBlockId, IrBuilder, IrEnumVariant, IrField,
    IrFunction, IrFunctionId, IrFunctionSignature, IrGlobal, IrGlobalId, IrId, IrInstruction,
    IrLocal, IrModule, IrParameter, IrPhiNode, IrSourceLocation, IrTerminator, IrType, IrTypeDef,
    IrTypeDefId, IrTypeDefinition, IrValue, Linkage, UnaryOp,
};
use crate::stdlib::{IrTypeDescriptor, MethodSignature, StdlibMapping};
use crate::tast::symbols::SymbolFlags;
use crate::tast::{
    InternedString, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId, TypeKind,
    TypeTable,
};
use log::{debug, trace, warn};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

impl<'a> HirToMirContext<'a> {
    pub(crate) fn lower_unary(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Unary { op, operand } = &expr.kind else {
            unreachable!("lower_unary on a non-Unary expression")
        };
        match op {
            HirUnaryOp::PostIncr
            | HirUnaryOp::PreIncr
            | HirUnaryOp::PostDecr
            | HirUnaryOp::PreDecr => {
                let old_value = self.lower_expression(operand)?;
                let one = self.builder.build_const(IrValue::I32(1))?;

                let is_increment = matches!(op, HirUnaryOp::PostIncr | HirUnaryOp::PreIncr);
                let new_value = if is_increment {
                    self.builder.build_binop(BinaryOp::Add, old_value, one)?
                } else {
                    self.builder.build_binop(BinaryOp::Sub, old_value, one)?
                };

                let result_type = self.convert_type(expr.ty);
                let src_loc = self.convert_source_location(&expr.source_location);
                if let Some(func) = self.builder.current_function_mut() {
                    func.locals.insert(
                        new_value,
                        crate::ir::IrLocal {
                            name: format!("_incr{}", new_value.0),
                            ty: result_type.clone(),
                            mutable: false,
                            source_location: src_loc.clone(),
                            allocation: crate::ir::AllocationHint::Stack,
                        },
                    );
                }

                match &operand.kind {
                    HirExprKind::Variable { symbol, .. } => {
                        // A bare static field lives in GLOBAL storage, not an
                        // SSA local, so the write must go through the global.
                        if let Some(&global_id) = self.global_symbol_map.get(symbol) {
                            self.builder.build_store_global(global_id, new_value);
                        } else if let Some(&cell) = self.capture_cells.get(symbol) {
                            self.builder.build_store(cell, new_value);
                            self.symbol_map.insert(*symbol, new_value);
                        } else {
                            // If we're inside a lambda with captured variables, also store back to environment
                            if let Some(ref env_layout) = self.current_env_layout {
                                if env_layout.find_field(*symbol).is_some() {
                                    let env_ptr = IrId::new(0); // First parameter in lambda is environment pointer
                                    env_layout.store_field(
                                        &mut self.builder,
                                        env_ptr,
                                        *symbol,
                                        new_value,
                                    );
                                }
                            }

                            self.symbol_map.insert(*symbol, new_value);
                        }
                    }
                    HirExprKind::Field { object, field } => {
                        if let Some(obj_reg) = self.lower_expression(object) {
                            let field_idx = self
                                .field_index_map
                                .get(field)
                                .map(|&(_, idx)| idx)
                                .or_else(|| {
                                    let field_name =
                                        self.symbol_table.get_symbol(*field).map(|s| s.name)?;
                                    let receiver_ty = object.ty;
                                    self.resolve_field_index_by_name(field_name, receiver_ty)
                                        .map(|(_, idx)| idx)
                                });
                            if let Some(idx) = field_idx {
                                let idx_const = self.builder.build_const(IrValue::I32(idx as i32));
                                if let Some(idx_reg) = idx_const {
                                    let field_ty = result_type.clone();
                                    let field_ptr = self.builder.build_gep(
                                        obj_reg,
                                        vec![idx_reg],
                                        field_ty.clone(),
                                    );
                                    if let Some(ptr) = field_ptr {
                                        self.builder.build_store(ptr, new_value);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }

                let result_reg = match op {
                    HirUnaryOp::PostIncr | HirUnaryOp::PostDecr => old_value,
                    HirUnaryOp::PreIncr | HirUnaryOp::PreDecr => new_value,
                    _ => unreachable!(),
                };

                Some(result_reg)
            }
            _ => {
                let operand_reg = self.lower_expression(operand)?;
                let result_reg = self
                    .builder
                    .build_unop(self.convert_unary_op(*op), operand_reg)?;

                // Register the result with its type so Cranelift can find it
                let result_type = self.convert_type(expr.ty);
                let src_loc = self.convert_source_location(&expr.source_location);
                if let Some(func) = self.builder.current_function_mut() {
                    func.locals.insert(
                        result_reg,
                        crate::ir::IrLocal {
                            name: format!("_temp{}", result_reg.0),
                            ty: result_type,
                            mutable: false,
                            source_location: src_loc,
                            allocation: crate::ir::AllocationHint::Stack,
                        },
                    );
                }

                Some(result_reg)
            }
        }
    }

    pub(crate) fn lower_binary(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Binary { op, lhs, rhs } = &expr.kind else {
            unreachable!("lower_binary on a non-Binary expression")
        };
        match op {
            HirBinaryOp::And => return self.lower_logical_and(lhs, rhs),
            HirBinaryOp::Or => return self.lower_logical_or(lhs, rhs),
            HirBinaryOp::NullCoalesce => return self.lower_null_coalesce(lhs, rhs),
            _ => {}
        }

        // Special handling for string concatenation with +
        if matches!(op, HirBinaryOp::Add) {
            let lhs_type_raw = self.convert_type(lhs.ty);
            let rhs_type_raw = self.convert_type(rhs.ty);

            // Override types with resolved IR types for pattern-bound variables
            let lhs_type = self.resolve_expr_ir_type(lhs, lhs_type_raw);
            let rhs_type = self.resolve_expr_ir_type(rhs, rhs_type_raw);

            let lhs_is_string = matches!(&lhs_type, IrType::String)
                || matches!(&lhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));
            let rhs_is_string = matches!(&rhs_type, IrType::String)
                || matches!(&rhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));

            // String concat chain detection: `prev + x` where `prev` is itself a
            // Binary Add producing a string. The HIR type may erase to Dynamic, so we
            // recursively inspect the LHS to detect string concat chains.
            fn is_string_concat_chain(expr: &HirExpr) -> bool {
                if let HirExprKind::Binary {
                    op: HirBinaryOp::Add,
                    lhs,
                    rhs,
                } = &expr.kind
                {
                    if matches!(&lhs.kind, HirExprKind::Literal(HirLiteral::String(_)))
                        || matches!(&rhs.kind, HirExprKind::Literal(HirLiteral::String(_)))
                    {
                        return true;
                    }
                    return is_string_concat_chain(lhs) || is_string_concat_chain(rhs);
                }
                false
            }
            let lhs_is_string = lhs_is_string || is_string_concat_chain(lhs);
            let rhs_is_string = rhs_is_string || is_string_concat_chain(rhs);

            if lhs_is_string || rhs_is_string {
                let lhs_reg = self.lower_expression(lhs)?;
                let rhs_reg = self.lower_expression(rhs)?;

                // Use MIR register types (from runtime mapping) instead of HIR types,
                // which may be unresolved generics (e.g. Ptr(Void) for Vec<Int>.length())
                let lhs_mir_type = self
                    .builder
                    .get_register_type(lhs_reg)
                    .unwrap_or(lhs_type.clone());
                let rhs_mir_type = self
                    .builder
                    .get_register_type(rhs_reg)
                    .unwrap_or(rhs_type.clone());

                let lhs_is_string_mir = matches!(&lhs_mir_type, IrType::String)
                    || matches!(&lhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String))
                    // Defensive: if HIR type says String but MIR register is Ptr(Void)
                    // (e.g. extern/stdlib string returns), trust the HIR type
                    || (lhs_is_string && matches!(&lhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void)));
                let rhs_is_string_mir = matches!(&rhs_mir_type, IrType::String)
                    || matches!(&rhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String))
                    || (rhs_is_string
                        && matches!(&rhs_mir_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void)));

                let lhs_str_val = if !lhs_is_string_mir {
                    if self.expr_is_value_type_expr(lhs) {
                        self.convert_value_type_to_string(lhs_reg)?
                    } else if let Some(reg) =
                        self.try_call_tostring(lhs_reg, self.resolve_expr_type_id(lhs))?
                    {
                        reg
                    } else {
                        self.convert_to_string_with_hint(lhs_reg, &lhs_mir_type, Some(lhs.ty))?
                    }
                } else {
                    lhs_reg
                };

                let rhs_str_val = if !rhs_is_string_mir {
                    if self.expr_is_value_type_expr(rhs) {
                        self.convert_value_type_to_string(rhs_reg)?
                    } else if let Some(reg) =
                        self.try_call_tostring(rhs_reg, self.resolve_expr_type_id(rhs))?
                    {
                        reg
                    } else {
                        self.convert_to_string_with_hint(rhs_reg, &rhs_mir_type, Some(rhs.ty))?
                    }
                } else {
                    rhs_reg
                };

                // String values are already *HaxeString (both literals and the
                // conversion helpers return pointers), which is what
                // string_concat expects, so pass them through unwrapped.
                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                let concat_func_id = self.register_stdlib_mir_forward_ref(
                    "string_concat",
                    vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                    string_ptr_ty.clone(),
                );

                return self.builder.build_call_direct(
                    concat_func_id,
                    vec![lhs_str_val, rhs_str_val],
                    string_ptr_ty,
                );
            }
        }

        // Comparison against a type parameter. A generic body is lowered ONCE
        // with its parameters erased to i64, so `a == b` on `T` compiles to a
        // raw integer compare: correct for Int, wrong for every value whose
        // identity is not its bits. Two equal Strings arrive as two different
        // HaxeString pointers and compare unequal -- and since a generic
        // `eq<T>(a, b)` is what test helpers are made of, that reads as the
        // program computing the wrong answer.
        //
        // `haxe_reflect_compare_typed` already decides by type tag, and the
        // tag is a placeholder that monomorphisation replaces with the
        // concrete one, the same way Reflect.compare inside a generic gets it.
        if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne) {
            let type_param_of = |ty: crate::tast::TypeId| -> Option<String> {
                let ti = self.type_table.get(ty)?;
                match &ti.kind {
                    crate::tast::core::TypeKind::TypeParameter { symbol_id, .. } => self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|sym| self.string_interner.get(sym.name))
                        .map(|n| n.to_string()),
                    _ => None,
                }
            };
            if let Some(tp_name) = type_param_of(lhs.ty).or_else(|| type_param_of(rhs.ty)) {
                let lhs_reg = self.lower_expression(lhs)?;
                let rhs_reg = self.lower_expression(rhs)?;
                let as_i64 = |this: &mut Self, reg| {
                    let ty = this.builder.get_register_type(reg).unwrap_or(IrType::I64);
                    if ty == IrType::I64 {
                        Some(reg)
                    } else {
                        this.builder.build_cast(reg, ty, IrType::I64).or(Some(reg))
                    }
                };
                let lhs_i64 = as_i64(self, lhs_reg)?;
                let rhs_i64 = as_i64(self, rhs_reg)?;

                let tag = self.builder.build_const(IrValue::I32(0))?;
                if let Some(func) = self.builder.current_function_mut() {
                    func.type_param_tag_fixups.push((tag, tp_name));
                }

                let cmp_func = self.get_or_register_extern_function(
                    "haxe_reflect_compare_typed",
                    vec![IrType::I64, IrType::I64, IrType::I32],
                    IrType::I64,
                );
                let ordering = self.builder.build_call_direct(
                    cmp_func,
                    vec![lhs_i64, rhs_i64, tag],
                    IrType::I64,
                )?;
                let zero = self.builder.build_const(IrValue::I64(0))?;
                let cmp_op = if matches!(op, HirBinaryOp::Eq) {
                    CompareOp::Eq
                } else {
                    CompareOp::Ne
                };
                return self.builder.build_cmp(cmp_op, ordering, zero);
            }
        }

        // String comparison: Eq/Ne/Lt/Le/Gt/Ge on strings need content comparison
        if matches!(
            op,
            HirBinaryOp::Eq
                | HirBinaryOp::Ne
                | HirBinaryOp::Lt
                | HirBinaryOp::Le
                | HirBinaryOp::Gt
                | HirBinaryOp::Ge
        ) {
            let lhs_type_raw = self.convert_type(lhs.ty);
            let rhs_type_raw = self.convert_type(rhs.ty);
            let lhs_type = self.resolve_expr_ir_type(lhs, lhs_type_raw);
            let rhs_type = self.resolve_expr_ir_type(rhs, rhs_type_raw);

            let lhs_is_string = matches!(&lhs_type, IrType::String)
                || matches!(&lhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));
            let rhs_is_string = matches!(&rhs_type, IrType::String)
                || matches!(&rhs_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String));

            if lhs_is_string && rhs_is_string {
                let lhs_reg = self.lower_expression(lhs)?;
                let rhs_reg = self.lower_expression(rhs)?;
                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                let cmp_func = self.get_or_register_extern_function(
                    "haxe_string_compare",
                    vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                    IrType::I32,
                );
                let cmp_result = self.builder.build_call_direct(
                    cmp_func,
                    vec![lhs_reg, rhs_reg],
                    IrType::I32,
                )?;
                let zero = self.builder.build_const(IrValue::I32(0))?;
                let cmp_op = match op {
                    HirBinaryOp::Eq => CompareOp::Eq,
                    HirBinaryOp::Ne => CompareOp::Ne,
                    HirBinaryOp::Lt => CompareOp::Lt,
                    HirBinaryOp::Le => CompareOp::Le,
                    HirBinaryOp::Gt => CompareOp::Gt,
                    HirBinaryOp::Ge => CompareOp::Ge,
                    _ => unreachable!(),
                };
                return self.builder.build_cmp(cmp_op, cmp_result, zero);
            }
        }

        // @:derive(PartialEq) field-by-field equality for class instances
        if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne) {
            let class_sym = {
                let type_table = self.type_table;
                let lhs_sym = type_table.get(lhs.ty).and_then(|t| {
                    if let TypeKind::Class { symbol_id, .. } = &t.kind {
                        Some(*symbol_id)
                    } else {
                        None
                    }
                });
                let rhs_sym = type_table.get(rhs.ty).and_then(|t| {
                    if let TypeKind::Class { symbol_id, .. } = &t.kind {
                        Some(*symbol_id)
                    } else {
                        None
                    }
                });
                match (lhs_sym, rhs_sym) {
                    (Some(l), Some(r)) if l == r && self.derive_partial_eq_classes.contains(&l) => {
                        Some(l)
                    }
                    _ => None,
                }
            };
            if let Some(sym) = class_sym {
                return self.lower_derived_equality(op, lhs, rhs, sym);
            }

            // Null<prim> equality routes through null-guarded unboxing compares:
            // the nullable is a box, so a raw cmp would compare box pointers.
            // `x == null` keeps the pointer compare.
            if !matches!(&lhs.kind, HirExprKind::Null) && !matches!(&rhs.kind, HirExprKind::Null) {
                if let Some(res) = self.lower_nullable_prim_eq(op, lhs, rhs) {
                    return Some(res);
                }
            }
        }

        // @:derive(PartialOrd) lexicographic ordering for class instances
        if matches!(
            op,
            HirBinaryOp::Lt | HirBinaryOp::Le | HirBinaryOp::Gt | HirBinaryOp::Ge
        ) {
            let class_sym = {
                let type_table = self.type_table;
                let lhs_sym = type_table.get(lhs.ty).and_then(|t| {
                    if let TypeKind::Class { symbol_id, .. } = &t.kind {
                        Some(*symbol_id)
                    } else {
                        None
                    }
                });
                let rhs_sym = type_table.get(rhs.ty).and_then(|t| {
                    if let TypeKind::Class { symbol_id, .. } = &t.kind {
                        Some(*symbol_id)
                    } else {
                        None
                    }
                });
                match (lhs_sym, rhs_sym) {
                    (Some(l), Some(r))
                        if l == r && self.derive_partial_ord_classes.contains(&l) =>
                    {
                        Some(l)
                    }
                    _ => None,
                }
            };
            if let Some(sym) = class_sym {
                return self.lower_derived_ordering(op, lhs, rhs, sym);
            }
        }

        // Dynamic arithmetic: a Dynamic HIR type may still lower to a raw concrete
        // value, so the MIR register type after lowering decides whether an operand
        // is a boxed DynamicValue. Dynamic-ness itself comes from the HIR TypeKind,
        // not from MIR Ptr(Void) — class types lower to Ptr(Void) but are not boxes.
        {
            let (lhs_is_dyn, rhs_is_dyn) = {
                let type_table = self.type_table;
                let lhs_dyn = type_table
                    .get(lhs.ty)
                    .map(|t| matches!(t.kind, TypeKind::Dynamic))
                    .unwrap_or(false);
                let rhs_dyn = type_table
                    .get(rhs.ty)
                    .map(|t| matches!(t.kind, TypeKind::Dynamic))
                    .unwrap_or(false);
                (lhs_dyn, rhs_dyn)
            };

            if lhs_is_dyn && rhs_is_dyn {
                let is_supported = matches!(
                    op,
                    HirBinaryOp::Add
                        | HirBinaryOp::Sub
                        | HirBinaryOp::Mul
                        | HirBinaryOp::Div
                        | HirBinaryOp::Mod
                        | HirBinaryOp::Eq
                        | HirBinaryOp::Lt
                        | HirBinaryOp::Gt
                        | HirBinaryOp::Le
                        | HirBinaryOp::Ge
                        | HirBinaryOp::Ne
                );
                if is_supported {
                    let lhs_reg = self.lower_expression(lhs)?;
                    let rhs_reg = self.lower_expression(rhs)?;

                    // SIMD vector short-circuit: @:coreType abstracts like SIMD4f
                    // have HIR type Dynamic but lower to Vector register type.
                    // Emit VectorBinOp directly for vector+vector arithmetic.
                    {
                        let lhs_rty = self.builder.get_register_type(lhs_reg);
                        let rhs_rty = self.builder.get_register_type(rhs_reg);
                        let lhs_is_vec = matches!(&lhs_rty, Some(IrType::Vector { .. }));
                        let rhs_is_vec = matches!(&rhs_rty, Some(IrType::Vector { .. }));
                        if lhs_is_vec || rhs_is_vec {
                            let vec_ty = if lhs_is_vec {
                                lhs_rty.unwrap()
                            } else {
                                rhs_rty.unwrap()
                            };
                            let bin_op = match op {
                                HirBinaryOp::Add => BinaryOp::Add,
                                HirBinaryOp::Sub => BinaryOp::Sub,
                                HirBinaryOp::Mul => BinaryOp::Mul,
                                HirBinaryOp::Div => BinaryOp::Div,
                                _ => {
                                    return self.builder.build_vector_binop(
                                        BinaryOp::Add,
                                        lhs_reg,
                                        rhs_reg,
                                        vec_ty,
                                    );
                                }
                            };
                            return self
                                .builder
                                .build_vector_binop(bin_op, lhs_reg, rhs_reg, vec_ty);
                        }
                    }

                    // Whether an operand is a boxed DynamicValue: variables are
                    // tracked in boxed_dynamic_symbols (a lambda param can be
                    // Dynamic yet hold a raw i64); anything else is judged by its
                    // MIR register type, since a concrete type is never a box.
                    let is_concrete_ir_type = |ty: &IrType| -> bool {
                        matches!(
                            ty,
                            IrType::I8
                                | IrType::I16
                                | IrType::I32
                                | IrType::I64
                                | IrType::U8
                                | IrType::U16
                                | IrType::U32
                                | IrType::U64
                                | IrType::F32
                                | IrType::F64
                                | IrType::Bool
                        )
                    };

                    let is_concrete = |reg| {
                        self.builder
                            .get_register_type(reg)
                            .as_ref()
                            .map(|t| is_concrete_ir_type(t))
                            .unwrap_or(false)
                    };
                    let lhs_boxed = match &lhs.kind {
                        HirExprKind::Variable { symbol, .. } => {
                            self.boxed_dynamic_symbols.contains(symbol)
                        }
                        _ => {
                            let ty = self.builder.get_register_type(lhs_reg);
                            !ty.as_ref().map(|t| is_concrete_ir_type(t)).unwrap_or(false)
                        }
                    };
                    let rhs_boxed = match &rhs.kind {
                        HirExprKind::Variable { symbol, .. } => {
                            self.boxed_dynamic_symbols.contains(symbol)
                        }
                        _ => {
                            let ty = self.builder.get_register_type(rhs_reg);
                            !ty.as_ref().map(|t| is_concrete_ir_type(t)).unwrap_or(false)
                        }
                    };
                    // Eq/Ne on two pointer-shaped Dynamic operands cannot be decided
                    // here: a box and a lambda param holding a raw i64 are both
                    // Ptr(Void) and neither is tracked. The runtime validates each
                    // side before dereferencing and falls back to the raw value.
                    if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne)
                        && !is_concrete(lhs_reg)
                        && !is_concrete(rhs_reg)
                    {
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        let eq_func = self.get_or_register_extern_function(
                            "haxe_dynamic_equals",
                            vec![ptr_void.clone(), ptr_void],
                            IrType::Bool,
                        );
                        let eq = self.builder.build_call_direct(
                            eq_func,
                            vec![lhs_reg, rhs_reg],
                            IrType::Bool,
                        )?;
                        if matches!(op, HirBinaryOp::Eq) {
                            return Some(eq);
                        }
                        let f = self.builder.build_const(IrValue::Bool(false))?;
                        return self.builder.build_cmp(CompareOp::Eq, eq, f);
                    }

                    if lhs_boxed && rhs_boxed {
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        let unbox_func = self.get_or_register_extern_function(
                            "haxe_unbox_float_ptr",
                            vec![ptr_void.clone()],
                            IrType::F64,
                        );
                        let lhs_f64 = self.builder.build_call_direct(
                            unbox_func,
                            vec![lhs_reg],
                            IrType::F64,
                        )?;
                        let rhs_f64 = self.builder.build_call_direct(
                            unbox_func,
                            vec![rhs_reg],
                            IrType::F64,
                        )?;

                        let is_comparison = matches!(
                            op,
                            HirBinaryOp::Eq
                                | HirBinaryOp::Ne
                                | HirBinaryOp::Lt
                                | HirBinaryOp::Le
                                | HirBinaryOp::Gt
                                | HirBinaryOp::Ge
                        );

                        if is_comparison {
                            // Eq/Ne go through the runtime, which dispatches on the
                            // box tag; unboxing to f64 would equate 1 and "1" and
                            // read a string as a double. Ordering stays numeric.
                            if matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne) {
                                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                                let eq_func = self.get_or_register_extern_function(
                                    "haxe_dynamic_equals",
                                    vec![ptr_void.clone(), ptr_void],
                                    IrType::Bool,
                                );
                                let eq = self.builder.build_call_direct(
                                    eq_func,
                                    vec![lhs_reg, rhs_reg],
                                    IrType::Bool,
                                )?;
                                if matches!(op, HirBinaryOp::Eq) {
                                    return Some(eq);
                                }
                                let f = self.builder.build_const(IrValue::Bool(false))?;
                                return self.builder.build_cmp(CompareOp::Eq, eq, f);
                            }
                            let cmp_op = match op {
                                HirBinaryOp::Lt => CompareOp::Lt,
                                HirBinaryOp::Le => CompareOp::Le,
                                HirBinaryOp::Gt => CompareOp::Gt,
                                HirBinaryOp::Ge => CompareOp::Ge,
                                _ => unreachable!(),
                            };
                            return self.builder.build_cmp(cmp_op, lhs_f64, rhs_f64);
                        } else {
                            let bin_op = match op {
                                HirBinaryOp::Add => BinaryOp::FAdd,
                                HirBinaryOp::Sub => BinaryOp::FSub,
                                HirBinaryOp::Mul => BinaryOp::FMul,
                                HirBinaryOp::Div => BinaryOp::FDiv,
                                HirBinaryOp::Mod => BinaryOp::FRem,
                                _ => unreachable!(),
                            };
                            let result_f64 = self.builder.build_binop(bin_op, lhs_f64, rhs_f64)?;

                            let box_func = self.get_or_register_extern_function(
                                "haxe_box_float_ptr",
                                vec![IrType::F64],
                                IrType::Ptr(Box::new(IrType::U8)),
                            );
                            return self.builder.build_call_direct(
                                box_func,
                                vec![result_f64],
                                IrType::Ptr(Box::new(IrType::U8)),
                            );
                        }
                    }

                    // At least one operand is a raw concrete value carrying a
                    // Dynamic HIR type; unbox the other side if it is a box.
                    let mut effective_lhs = lhs_reg;
                    let mut effective_rhs = rhs_reg;

                    if lhs_boxed && !rhs_boxed {
                        let rhs_ty = self
                            .builder
                            .get_register_type(rhs_reg)
                            .unwrap_or(IrType::I64);
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        if matches!(rhs_ty, IrType::F32 | IrType::F64) {
                            let unbox = self.get_or_register_extern_function(
                                "haxe_unbox_float_ptr",
                                vec![ptr_void],
                                IrType::F64,
                            );
                            effective_lhs = self.builder.build_call_direct(
                                unbox,
                                vec![lhs_reg],
                                IrType::F64,
                            )?;
                        } else {
                            let unbox = self.get_or_register_extern_function(
                                "haxe_unbox_int_ptr",
                                vec![ptr_void],
                                IrType::I64,
                            );
                            effective_lhs = self.builder.build_call_direct(
                                unbox,
                                vec![lhs_reg],
                                IrType::I64,
                            )?;
                        }
                    } else if !lhs_boxed && rhs_boxed {
                        let lhs_ty = self
                            .builder
                            .get_register_type(lhs_reg)
                            .unwrap_or(IrType::I64);
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        if matches!(lhs_ty, IrType::F32 | IrType::F64) {
                            let unbox = self.get_or_register_extern_function(
                                "haxe_unbox_float_ptr",
                                vec![ptr_void],
                                IrType::F64,
                            );
                            effective_rhs = self.builder.build_call_direct(
                                unbox,
                                vec![rhs_reg],
                                IrType::F64,
                            )?;
                        } else {
                            let unbox = self.get_or_register_extern_function(
                                "haxe_unbox_int_ptr",
                                vec![ptr_void],
                                IrType::I64,
                            );
                            effective_rhs = self.builder.build_call_direct(
                                unbox,
                                vec![rhs_reg],
                                IrType::I64,
                            )?;
                        }
                    }

                    let eff_lhs_ty = self
                        .builder
                        .get_register_type(effective_lhs)
                        .unwrap_or(IrType::I64);
                    let eff_rhs_ty = self
                        .builder
                        .get_register_type(effective_rhs)
                        .unwrap_or(IrType::I64);

                    let l_is_int = matches!(
                        eff_lhs_ty,
                        IrType::I8
                            | IrType::I16
                            | IrType::I32
                            | IrType::I64
                            | IrType::U8
                            | IrType::U16
                            | IrType::U32
                            | IrType::U64
                    );
                    let r_is_int = matches!(
                        eff_rhs_ty,
                        IrType::I8
                            | IrType::I16
                            | IrType::I32
                            | IrType::I64
                            | IrType::U8
                            | IrType::U16
                            | IrType::U32
                            | IrType::U64
                    );
                    let l_is_float = matches!(eff_lhs_ty, IrType::F32 | IrType::F64);
                    let r_is_float = matches!(eff_rhs_ty, IrType::F32 | IrType::F64);

                    if l_is_int && r_is_float {
                        effective_lhs = self.builder.build_cast(
                            effective_lhs,
                            eff_lhs_ty.clone(),
                            IrType::F64,
                        )?;
                    }
                    if r_is_int && l_is_float {
                        effective_rhs = self.builder.build_cast(
                            effective_rhs,
                            eff_rhs_ty.clone(),
                            IrType::F64,
                        )?;
                    }
                    if matches!(op, HirBinaryOp::Div) && l_is_int && r_is_int {
                        effective_lhs = self.builder.build_cast(
                            effective_lhs,
                            eff_lhs_ty.clone(),
                            IrType::F64,
                        )?;
                        effective_rhs = self.builder.build_cast(
                            effective_rhs,
                            eff_rhs_ty.clone(),
                            IrType::F64,
                        )?;
                    }

                    let result_reg = match self.convert_binary_op_to_mir(*op) {
                        MirBinaryOp::Binary(bin_op) => {
                            self.builder
                                .build_binop(bin_op, effective_lhs, effective_rhs)?
                        }
                        MirBinaryOp::Compare(cmp_op) => {
                            self.builder
                                .build_cmp(cmp_op, effective_lhs, effective_rhs)?
                        }
                    };
                    return Some(result_reg);
                }
            }

            // Mixed Dynamic + concrete arithmetic. Boxed DynamicValue* (Ptr(U8))
            // and type-erased raw values (Ptr(Void)) differ: only Ptr(U8) may be
            // unboxed, Ptr(Void) is cast to integer.
            // `x == null` is a pointer comparison, not a Dynamic one. `null`
            // carries no type, so it lands on the Dynamic side and the mixed
            // path below coerces it through `haxe_coerce_dynamic_to_int` -- a
            // runtime call, a cast and a compare, to ask whether a pointer is
            // zero. A tree walk is mostly null checks, so that call is per
            // node in the hottest code a recursive structure has.
            //
            // Only a literal null qualifies. A Dynamic VARIABLE compared
            // against a reference still needs the coercion, because a boxed
            // DynamicValue's address is not its contents.
            let comparing_to_null = matches!(op, HirBinaryOp::Eq | HirBinaryOp::Ne)
                && (matches!(lhs.kind, HirExprKind::Null) || matches!(rhs.kind, HirExprKind::Null));

            if !comparing_to_null && (lhs_is_dyn || rhs_is_dyn) && !(lhs_is_dyn && rhs_is_dyn) {
                let is_arith = matches!(
                    op,
                    HirBinaryOp::Add
                        | HirBinaryOp::Sub
                        | HirBinaryOp::Mul
                        | HirBinaryOp::Div
                        | HirBinaryOp::Mod
                        | HirBinaryOp::Eq
                        | HirBinaryOp::Lt
                        | HirBinaryOp::Gt
                        | HirBinaryOp::Le
                        | HirBinaryOp::Ge
                        | HirBinaryOp::Ne
                );
                if is_arith {
                    let mut lhs_reg = self.lower_expression(lhs)?;
                    let mut rhs_reg = self.lower_expression(rhs)?;

                    let concrete_ty = if lhs_is_dyn {
                        self.builder
                            .get_register_type(rhs_reg)
                            .unwrap_or(IrType::I32)
                    } else {
                        self.builder
                            .get_register_type(lhs_reg)
                            .unwrap_or(IrType::I32)
                    };

                    let is_float = matches!(concrete_ty, IrType::F32 | IrType::F64);

                    // Coerce the Dynamic-side register to the concrete type through
                    // haxe_coerce_dynamic_to_int/float, which accepts both boxed
                    // DynamicValue* and type-erased raw integers.
                    let coerce_dyn = |s: &mut Self, reg: IrId| -> Option<IrId> {
                        let reg_ty = s.builder.get_register_type(reg);
                        if reg_ty
                            .as_ref()
                            .map(|t| {
                                matches!(
                                    t,
                                    IrType::I32
                                        | IrType::I64
                                        | IrType::F32
                                        | IrType::F64
                                        | IrType::Bool
                                )
                            })
                            .unwrap_or(false)
                        {
                            return Some(reg);
                        }
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        if is_float {
                            let f = s.get_or_register_extern_function(
                                "haxe_coerce_dynamic_to_float",
                                vec![ptr_void],
                                IrType::F64,
                            );
                            s.builder.build_call_direct(f, vec![reg], IrType::F64)
                        } else {
                            let f = s.get_or_register_extern_function(
                                "haxe_coerce_dynamic_to_int",
                                vec![ptr_void],
                                IrType::I64,
                            );
                            let v = s.builder.build_call_direct(f, vec![reg], IrType::I64)?;
                            Some(
                                s.builder
                                    .build_cast(v, IrType::I64, concrete_ty.clone())
                                    .unwrap_or(v),
                            )
                        }
                    };

                    if lhs_is_dyn {
                        lhs_reg = coerce_dyn(self, lhs_reg)?;
                    } else {
                        rhs_reg = coerce_dyn(self, rhs_reg)?;
                    }

                    let mir_op = self.convert_binary_op_to_mir(*op);
                    let result_reg = match mir_op {
                        MirBinaryOp::Binary(arith_op) => {
                            self.builder.build_binop(arith_op, lhs_reg, rhs_reg)?
                        }
                        MirBinaryOp::Compare(cmp_op) => {
                            self.builder.build_cmp(cmp_op, lhs_reg, rhs_reg)?
                        }
                    };
                    return Some(result_reg);
                }
            }
        }

        let mut lhs_reg = self.lower_expression(lhs)?;
        let mut rhs_reg = self.lower_expression(rhs)?;

        // Primitive arithmetic must not run on raw DynamicValue* pointers, so
        // unbox Null<T> operands using each operand's OWN inner primitive type —
        // expr.ty may still be Optional after typechecking unified to Null<Int>.
        if matches!(
            op,
            HirBinaryOp::Add
                | HirBinaryOp::Sub
                | HirBinaryOp::Mul
                | HirBinaryOp::Div
                | HirBinaryOp::Mod
        ) {
            if self.is_optional_primitive(lhs.ty) {
                if let Some(inner) = self.optional_inner_type(lhs.ty) {
                    if let Some(unboxed) = self.maybe_unbox_optional(lhs_reg, lhs.ty, inner) {
                        lhs_reg = unboxed;
                    }
                }
            }
            if self.is_optional_primitive(rhs.ty) {
                if let Some(inner) = self.optional_inner_type(rhs.ty) {
                    if let Some(unboxed) = self.maybe_unbox_optional(rhs_reg, rhs.ty, inner) {
                        rhs_reg = unboxed;
                    }
                }
            }
        }

        let lhs_type = self.convert_type(lhs.ty);
        let rhs_type = self.convert_type(rhs.ty);

        let lhs_is_int = matches!(
            lhs_type,
            IrType::I8
                | IrType::I16
                | IrType::I32
                | IrType::I64
                | IrType::U8
                | IrType::U16
                | IrType::U32
                | IrType::U64
        );
        let rhs_is_int = matches!(
            rhs_type,
            IrType::I8
                | IrType::I16
                | IrType::I32
                | IrType::I64
                | IrType::U8
                | IrType::U16
                | IrType::U32
                | IrType::U64
        );
        let lhs_is_float = matches!(lhs_type, IrType::F32 | IrType::F64);
        let rhs_is_float = matches!(rhs_type, IrType::F32 | IrType::F64);

        // Cast int to float when mixing types (promotes to F64)
        if lhs_is_int && rhs_is_float {
            lhs_reg = self
                .builder
                .build_cast(lhs_reg, lhs_type.clone(), IrType::F64)?;
        }
        if rhs_is_int && lhs_is_float {
            rhs_reg = self
                .builder
                .build_cast(rhs_reg, rhs_type.clone(), IrType::F64)?;
        }

        // Haxe division always yields Float, so int operands are promoted first.
        if matches!(op, HirBinaryOp::Div) && lhs_is_int && rhs_is_int {
            lhs_reg = self
                .builder
                .build_cast(lhs_reg, lhs_type.clone(), IrType::F64)?;
            rhs_reg = self
                .builder
                .build_cast(rhs_reg, rhs_type.clone(), IrType::F64)?;
        }

        // Vector-ness is decided from the operand register types rather than
        // convert_type(expr.ty): @:coreType abstracts may not resolve through it.
        let lhs_actual_type = self
            .builder
            .get_register_type(lhs_reg)
            .unwrap_or(IrType::I64);
        let rhs_actual_type = self
            .builder
            .get_register_type(rhs_reg)
            .unwrap_or(IrType::I64);
        let result_type = if lhs_actual_type.is_vector() {
            lhs_actual_type.clone()
        } else if rhs_actual_type.is_vector() {
            rhs_actual_type.clone()
        } else {
            self.convert_type(expr.ty)
        };
        let result_reg = if result_type.is_vector() {
            let bin_op = match op {
                HirBinaryOp::Add => BinaryOp::Add,
                HirBinaryOp::Sub => BinaryOp::Sub,
                HirBinaryOp::Mul => BinaryOp::Mul,
                HirBinaryOp::Div => BinaryOp::Div,
                _ => {
                    debug!("Unsupported vector binary op: {:?}", op);
                    return None;
                }
            };
            self.builder
                .build_vector_binop(bin_op, lhs_reg, rhs_reg, result_type.clone())?
        } else {
            match self.convert_binary_op_to_mir(*op) {
                MirBinaryOp::Binary(bin_op) => {
                    self.builder.build_binop(bin_op, lhs_reg, rhs_reg)?
                }
                MirBinaryOp::Compare(cmp_op) => self.builder.build_cmp(cmp_op, lhs_reg, rhs_reg)?,
            }
        };
        let src_loc = self.convert_source_location(&expr.source_location);
        if let Some(func) = self.builder.current_function_mut() {
            func.locals.insert(
                result_reg,
                crate::ir::IrLocal {
                    name: format!("_temp{}", result_reg.0),
                    ty: result_type,
                    mutable: false,
                    source_location: src_loc,
                    allocation: crate::ir::AllocationHint::Stack,
                },
            );
        }

        Some(result_reg)
    }
}
