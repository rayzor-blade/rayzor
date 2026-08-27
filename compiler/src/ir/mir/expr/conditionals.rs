//! Operators that lower to branches: short-circuit `&&`/`||`, `??`, and `?:`.

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
    pub(crate) fn lower_logical_and(&mut self, lhs: &HirExpr, rhs: &HirExpr) -> Option<IrId> {
        // Short-circuit AND: if lhs is false, don't evaluate rhs.
        let eval_rhs = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        let lhs_val = self.lower_expression(lhs)?;

        // Build false_val before branching so it lives in this block's scope.
        let false_val = self.builder.build_bool(false)?;

        // Capture the block LHS was evaluated in, before branching.
        let lhs_block = self.builder.current_block()?;

        self.builder.build_cond_branch(lhs_val, eval_rhs, merge)?;

        self.builder.switch_to_block(eval_rhs);
        let rhs_val = self.lower_expression(rhs)?;
        let rhs_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(merge);
        let result = self.builder.build_phi(merge, IrType::Bool)?;
        // lhs_block is where we came from if LHS was false (short-circuit path)
        self.builder
            .add_phi_incoming(merge, result, lhs_block, false_val)?;
        self.builder
            .add_phi_incoming(merge, result, rhs_block, rhs_val)?;

        Some(result)
    }

    pub(crate) fn lower_logical_or(&mut self, lhs: &HirExpr, rhs: &HirExpr) -> Option<IrId> {
        // Short-circuit OR: if lhs is true, don't evaluate rhs.
        let eval_rhs = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        let lhs_val = self.lower_expression(lhs)?;

        // Build true_val before branching so it lives in this block's scope.
        let true_val = self.builder.build_bool(true)?;

        // Capture the block LHS was evaluated in, before branching.
        let lhs_block = self.builder.current_block()?;

        self.builder.build_cond_branch(lhs_val, merge, eval_rhs)?;

        self.builder.switch_to_block(eval_rhs);
        let rhs_val = self.lower_expression(rhs)?;
        let rhs_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(merge);
        let result = self.builder.build_phi(merge, IrType::Bool)?;
        // lhs_block is where we came from if LHS was true (short-circuit path)
        self.builder
            .add_phi_incoming(merge, result, lhs_block, true_val)?;
        self.builder
            .add_phi_incoming(merge, result, rhs_block, rhs_val)?;

        Some(result)
    }

    /// Lower `Null<prim>` equality. The nullable is a DynamicValue box, so the
    /// generic compare would test the box pointer against the value. Routes
    /// Optional<Int|Float|Bool> vs bare prim (either order) and Optional vs
    /// Optional through null-guarded tag-aware runtime compares. Returns None
    /// when the shape doesn't apply (caller falls through to the generic
    /// compare); `x == null` never reaches here (HirExprKind::Null operands are
    /// excluded at the call site).
    pub(crate) fn lower_nullable_prim_eq(
        &mut self,
        op: &HirBinaryOp,
        lhs: &HirExpr,
        rhs: &HirExpr,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;
        #[derive(Clone, Copy, PartialEq)]
        enum Prim {
            Int,
            Float,
            Bool,
        }
        let classify = |me: &Self, ty: crate::tast::TypeId| -> (Option<Prim>, Option<Prim>) {
            // (optional_inner_prim, bare_prim)
            let tt = me.type_table;
            match tt.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => {
                    let inner = match tt.get(*inner_type).map(|t| &t.kind) {
                        Some(TypeKind::Int) => Some(Prim::Int),
                        Some(TypeKind::Float) => Some(Prim::Float),
                        Some(TypeKind::Bool) => Some(Prim::Bool),
                        _ => None,
                    };
                    (inner, None)
                }
                Some(TypeKind::Int) => (None, Some(Prim::Int)),
                Some(TypeKind::Float) => (None, Some(Prim::Float)),
                Some(TypeKind::Bool) => (None, Some(Prim::Bool)),
                _ => (None, None),
            }
        };
        let (lhs_opt, lhs_bare) = classify(self, lhs.ty);
        let (rhs_opt, rhs_bare) = classify(self, rhs.ty);

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        // Widen a bare prim register for the helper ABI.
        let widen_int = |me: &mut Self, reg: IrId, from: Prim| -> Option<IrId> {
            match from {
                Prim::Int => me.builder.build_cast(reg, IrType::I32, IrType::I64),
                Prim::Bool => me.builder.build_cast(reg, IrType::Bool, IrType::I64),
                Prim::Float => None, // routed to the float helper instead
            }
        };
        let widen_float = |me: &mut Self, reg: IrId, from: Prim| -> Option<IrId> {
            match from {
                Prim::Float => Some(reg), // Haxe Float is already f64
                Prim::Int => me.builder.build_cast(reg, IrType::I32, IrType::F64),
                Prim::Bool => me.builder.build_cast(reg, IrType::Bool, IrType::F64),
            }
        };
        // The null-eq helpers take the Optional operand as a *DynamicValue box
        // pointer. Most Optional<prim> values already are one (a variable, a
        // user-fn return), but some stdlib calls return a raw primitive despite
        // a Null<T> type — Std.parseInt returns i64 (i64::MIN = null) — and the
        // helper would dereference that raw value as a pointer. Detect "raw" by
        // the register not being a pointer and box it, as `var x:Null<T> = ...`
        // would.
        let box_if_raw = |me: &mut Self, reg: IrId, inner: Prim| -> Option<IrId> {
            if matches!(me.builder.get_register_type(reg), Some(IrType::Ptr(_))) {
                return Some(reg);
            }
            let boxed_ptr = IrType::Ptr(Box::new(IrType::U8));
            let (name, arg_ty) = match inner {
                Prim::Int => ("haxe_box_int_ptr", IrType::I64),
                Prim::Bool => ("haxe_box_bool_ptr", IrType::Bool),
                Prim::Float => ("haxe_box_float_ptr", IrType::F64),
            };
            let cur = me
                .builder
                .get_register_type(reg)
                .unwrap_or_else(|| arg_ty.clone());
            let v = if cur != arg_ty {
                me.builder.build_cast(reg, cur, arg_ty.clone())?
            } else {
                reg
            };
            let f = me.get_or_register_extern_function(name, vec![arg_ty], boxed_ptr.clone());
            me.builder.build_call_direct(f, vec![v], boxed_ptr)
        };

        // Dynamic vs Optional<prim>: coerce the Dynamic side to the Optional's
        // inner primitive and compare tag-aware against the box. `classify`
        // returns (None, None) for Dynamic, so this shape is otherwise
        // invisible to the match below and would fall through to a raw
        // pointer compare.
        let lhs_is_dyn = matches!(
            self.type_table.get(lhs.ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic)
        );
        let rhs_is_dyn = matches!(
            self.type_table.get(rhs.ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic)
        );
        if lhs_is_dyn != rhs_is_dyn {
            if let Some(inner) = if rhs_is_dyn { lhs_opt } else { rhs_opt } {
                let opt_first = rhs_is_dyn; // Optional is on the left iff rhs is Dynamic
                let opt_expr = if opt_first { lhs } else { rhs };
                let dyn_expr = if opt_first { rhs } else { lhs };
                let opt_reg = self.lower_expression(opt_expr)?;
                let opt_reg = box_if_raw(self, opt_reg, inner)?;
                let dyn_reg = self.lower_expression(dyn_expr)?;
                let use_float = inner == Prim::Float;
                let (coerce_name, coerce_ret) = if use_float {
                    ("haxe_coerce_dynamic_to_float", IrType::F64)
                } else {
                    ("haxe_coerce_dynamic_to_int", IrType::I64)
                };
                let cf = self.get_or_register_extern_function(
                    coerce_name,
                    vec![ptr_u8.clone()],
                    coerce_ret.clone(),
                );
                let coerced =
                    self.builder
                        .build_call_direct(cf, vec![dyn_reg], coerce_ret.clone())?;
                let eq_name = if use_float {
                    "haxe_null_float_eq"
                } else {
                    "haxe_null_int_eq"
                };
                let f = self.get_or_register_extern_function(
                    eq_name,
                    vec![ptr_u8, coerce_ret],
                    IrType::Bool,
                );
                let eq = self
                    .builder
                    .build_call_direct(f, vec![opt_reg, coerced], IrType::Bool)?;
                if matches!(op, HirBinaryOp::Ne) {
                    let ffalse = self.builder.build_bool(false)?;
                    return self.builder.build_cmp(CompareOp::Eq, eq, ffalse);
                }
                return Some(eq);
            }
        }

        let eq_result = match (lhs_opt, lhs_bare, rhs_opt, rhs_bare) {
            // Optional<prim> vs bare prim (either order)
            (Some(inner), _, None, Some(bare)) | (None, Some(bare), Some(inner), _) => {
                let opt_first = lhs_opt.is_some();
                let (opt_expr, bare_expr) = if opt_first { (lhs, rhs) } else { (rhs, lhs) };
                let opt_reg = self.lower_expression(opt_expr)?;
                let opt_reg = box_if_raw(self, opt_reg, inner)?;
                let bare_reg = self.lower_expression(bare_expr)?;
                let use_float = inner == Prim::Float || bare == Prim::Float;
                if use_float {
                    let v = widen_float(self, bare_reg, bare)?;
                    let f = self.get_or_register_extern_function(
                        "haxe_null_float_eq",
                        vec![ptr_u8, IrType::F64],
                        IrType::Bool,
                    );
                    self.builder
                        .build_call_direct(f, vec![opt_reg, v], IrType::Bool)?
                } else {
                    let v = widen_int(self, bare_reg, bare)?;
                    let f = self.get_or_register_extern_function(
                        "haxe_null_int_eq",
                        vec![ptr_u8, IrType::I64],
                        IrType::Bool,
                    );
                    self.builder
                        .build_call_direct(f, vec![opt_reg, v], IrType::Bool)?
                }
            }
            // Optional<prim> vs Optional<prim>
            (Some(li), _, Some(ri), _) => {
                let lreg = self.lower_expression(lhs)?;
                let lreg = box_if_raw(self, lreg, li)?;
                let rreg = self.lower_expression(rhs)?;
                let rreg = box_if_raw(self, rreg, ri)?;
                let name = if li == Prim::Float || ri == Prim::Float {
                    "haxe_null_null_eq_float"
                } else {
                    "haxe_null_null_eq_int"
                };
                let f = self.get_or_register_extern_function(
                    name,
                    vec![ptr_u8.clone(), ptr_u8],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(f, vec![lreg, rreg], IrType::Bool)?
            }
            _ => return None,
        };

        if matches!(op, HirBinaryOp::Ne) {
            let f = self.builder.build_bool(false)?;
            self.builder.build_cmp(CompareOp::Eq, eq_result, f)
        } else {
            Some(eq_result)
        }
    }

    pub(crate) fn lower_null_coalesce(&mut self, lhs: &HirExpr, rhs: &HirExpr) -> Option<IrId> {
        // Intermediary blocks (as for ternary) avoid Cranelift phi issues when
        // br_if targets a merge block directly.
        let lhs_pass = self.builder.create_block()?;
        let eval_rhs = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        // Optional{primitive} needs an unbox on the pass-through path.
        let opt_prim = self.is_optional_primitive(lhs.ty);

        let lhs_val = self.lower_expression(lhs)?;

        // Null check: lhs != 0 (null pointers and null values are 0). The zero
        // takes the LHS type to avoid a type mismatch in the comparison.
        let lhs_ir_type = self.builder.get_register_type(lhs_val);
        let zero = match lhs_ir_type {
            Some(IrType::I32) => self.builder.build_const(IrValue::I32(0))?,
            Some(IrType::F32) => self.builder.build_const(IrValue::F32(0.0))?,
            Some(IrType::F64) => self.builder.build_const(IrValue::F64(0.0))?,
            _ => self.builder.build_const(IrValue::I64(0))?,
        };
        let is_not_null = self.builder.build_cmp(CompareOp::Ne, lhs_val, zero)?;

        self.builder
            .build_cond_branch(is_not_null, lhs_pass, eval_rhs)?;

        self.builder.switch_to_block(lhs_pass);
        let lhs_final = if opt_prim {
            // Unbox the boxed primitive: Ptr(U8) → inner type
            let inner_type = {
                let type_table = self.type_table;
                match type_table.get(lhs.ty).map(|t| &t.kind) {
                    Some(crate::tast::TypeKind::Optional { inner_type }) => Some(*inner_type),
                    _ => None,
                }
            };
            if let Some(inner_ty) = inner_type {
                let inner_ir = self.convert_type(inner_ty);
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                match &inner_ir {
                    IrType::I32 => {
                        let unbox_func = self.get_or_register_extern_function(
                            "haxe_unbox_int_ptr",
                            vec![ptr_u8],
                            IrType::I64,
                        );
                        let unboxed = self.builder.build_call_direct(
                            unbox_func,
                            vec![lhs_val],
                            IrType::I64,
                        )?;
                        self.builder.build_cast(unboxed, IrType::I64, IrType::I32)?
                    }
                    IrType::F64 => {
                        let unbox_func = self.get_or_register_extern_function(
                            "haxe_unbox_float_ptr",
                            vec![ptr_u8],
                            IrType::F64,
                        );
                        self.builder
                            .build_call_direct(unbox_func, vec![lhs_val], IrType::F64)?
                    }
                    IrType::Bool => {
                        let unbox_func = self.get_or_register_extern_function(
                            "haxe_unbox_bool_ptr",
                            vec![ptr_u8],
                            IrType::I64,
                        );
                        let unboxed = self.builder.build_call_direct(
                            unbox_func,
                            vec![lhs_val],
                            IrType::I64,
                        )?;
                        self.builder
                            .build_cast(unboxed, IrType::I64, IrType::Bool)?
                    }
                    _ => lhs_val,
                }
            } else {
                lhs_val
            }
        } else {
            lhs_val
        };
        let lhs_pass_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        self.builder.switch_to_block(eval_rhs);
        let rhs_val = self.lower_expression(rhs)?;
        let rhs_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        // When LHS was Optional{primitive}, the result type is the unboxed
        // primitive type (the RHS type).
        self.builder.switch_to_block(merge);
        let result_type = if opt_prim {
            self.convert_type(rhs.ty)
        } else {
            self.convert_type(lhs.ty)
        };
        let result = self.builder.build_phi(merge, result_type)?;
        self.builder
            .add_phi_incoming(merge, result, lhs_pass_block, lhs_final)?;
        self.builder
            .add_phi_incoming(merge, result, rhs_block, rhs_val)?;

        Some(result)
    }

    pub(crate) fn lower_conditional(
        &mut self,
        cond: &HirExpr,
        then_expr: &HirExpr,
        else_expr: &HirExpr,
    ) -> Option<IrId> {
        self.lower_conditional_typed(cond, then_expr, else_expr, None)
    }

    pub(crate) fn lower_conditional_typed(
        &mut self,
        cond: &HirExpr,
        then_expr: &HirExpr,
        else_expr: &HirExpr,
        result_ty: Option<TypeId>,
    ) -> Option<IrId> {
        let then_block = self.builder.create_block()?;
        let else_block = self.builder.create_block()?;
        let merge_block = self.builder.create_block()?;

        // Snapshot symbol_map before branches so each branch can be lowered
        // against the same starting bindings.
        let symbol_map_before = self.symbol_map.clone();

        let cond_val = self.lower_expression(cond)?;

        // Branch-phi for effectful Call results.
        //
        // Cranelift's egraph elaboration panics when an effectful instruction's
        // dest is referenced by another instruction in a different block (i.e.,
        // a cross-block direct SSA reference). Classic trigger:
        //
        //     v = call @effectful_extern    (in cond_eval block)
        //     brif cond, then, else
        //     then: <use v>                 <-- cross-block use of v
        //
        // Route v through a phi/block-arg at branch entry to sidestep this.
        let cond_eval_block = self.builder.current_block()?;
        let effectful_call_result_regs: BTreeSet<IrId> = {
            let mut regs = BTreeSet::new();
            if let Some(func) = self.builder.current_function() {
                if let Some(block) = func.cfg.blocks.get(&cond_eval_block) {
                    for inst in &block.instructions {
                        match inst {
                            IrInstruction::CallDirect { dest: Some(d), .. }
                            | IrInstruction::CallIndirect { dest: Some(d), .. } => {
                                regs.insert(*d);
                            }
                            _ => {}
                        }
                    }
                }
            }
            regs
        };

        let mut branch_phi_rebind: Vec<(SymbolId, IrId, IrId)> = Vec::new();
        if !effectful_call_result_regs.is_empty() {
            let mut candidates: Vec<(SymbolId, IrId)> = symbol_map_before
                .iter()
                .filter(|(_, r)| effectful_call_result_regs.contains(r))
                .map(|(s, r)| (*s, *r))
                .collect();
            candidates.sort_by_key(|(s, _)| *s);

            for (sym, reg) in candidates {
                let var_ty = match self.builder.get_register_type(reg) {
                    Some(t) => t,
                    None => continue,
                };
                if !matches!(
                    var_ty,
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
                        | IrType::Ptr(_)
                ) {
                    continue;
                }
                let then_phi = match self.builder.build_phi(then_block, var_ty.clone()) {
                    Some(p) => p,
                    None => continue,
                };
                self.builder
                    .add_phi_incoming(then_block, then_phi, cond_eval_block, reg);
                let else_phi = match self.builder.build_phi(else_block, var_ty.clone()) {
                    Some(p) => p,
                    None => continue,
                };
                self.builder
                    .add_phi_incoming(else_block, else_phi, cond_eval_block, reg);
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(local) = func.locals.get(&reg).cloned() {
                        func.locals.insert(
                            then_phi,
                            crate::ir::IrLocal {
                                name: format!("{}_then_branchphi", local.name),
                                ty: var_ty.clone(),
                                mutable: local.mutable,
                                source_location: local.source_location,
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                        func.locals.insert(
                            else_phi,
                            crate::ir::IrLocal {
                                name: format!("{}_else_branchphi", local.name),
                                ty: var_ty,
                                mutable: local.mutable,
                                source_location: local.source_location,
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }
                branch_phi_rebind.push((sym, then_phi, else_phi));
            }
        }

        self.builder
            .build_cond_branch(cond_val, then_block, else_block)?;

        self.builder.switch_to_block(then_block);
        // Rebind effectful-call-bound symbols to the then-branch phi.
        for (sym, then_phi, _) in &branch_phi_rebind {
            self.symbol_map.insert(*sym, *then_phi);
        }
        let mut then_val = self.lower_expression(then_expr);
        let then_terminated = self.is_terminated();
        // Box primitive values for Optional<primitive> result types
        if !then_terminated {
            if let (Some(val), Some(rty)) = (then_val, result_ty) {
                if let Some(boxed) = self.maybe_box_for_optional(val, then_expr.ty, rty) {
                    then_val = Some(boxed);
                }
            }
            // Branch building is deferred until after type harmonization.
        }
        let then_end_block = self.builder.current_block()?;
        let symbol_map_after_then = self.symbol_map.clone();

        // Reset to before-branch state.
        self.symbol_map = symbol_map_before.clone();
        // Rebind effectful-call-bound symbols to the else-branch phi.
        for (sym, _, else_phi) in &branch_phi_rebind {
            self.symbol_map.insert(*sym, *else_phi);
        }
        self.builder.switch_to_block(else_block);
        let mut else_val = self.lower_expression(else_expr);
        let else_terminated = self.is_terminated();
        // Box primitive values for Optional<primitive> result types
        if !else_terminated {
            if let (Some(val), Some(rty)) = (else_val, result_ty) {
                if let Some(boxed) = self.maybe_box_for_optional(val, else_expr.ty, rty) {
                    else_val = Some(boxed);
                }
            }
            // Branch building is deferred until after type harmonization.
        }
        let else_end_block = self.builder.current_block()?;
        let symbol_map_after_else = self.symbol_map.clone();

        // Type harmonization: if one branch produces Ptr (e.g. null) and other produces
        // a primitive (e.g. i64 from field access), box the primitive to make the phi valid.
        // This handles Null<T> return types where if(x==null) null else x.field merges
        // incompatible types.
        if let (Some(tv), Some(ev)) = (then_val, else_val) {
            if !then_terminated && !else_terminated {
                let then_rty = self.builder.get_register_type(tv);
                let else_rty = self.builder.get_register_type(ev);
                let then_is_ptr = matches!(then_rty, Some(IrType::Ptr(_)));
                let else_is_ptr = matches!(else_rty, Some(IrType::Ptr(_)));
                let then_is_prim = matches!(
                    then_rty,
                    Some(IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool)
                );
                let else_is_prim = matches!(
                    else_rty,
                    Some(IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool)
                );

                // When the conditional's own result type is a concrete scalar,
                // harmonise down to that scalar instead of boxing the live branch
                // up to a pointer: the pointer side is a synthesised null for a
                // missing else (tast_to_hir::make_null_literal) and is never the
                // value the expression yields, while boxing it leaves a dead phi
                // fed by haxe_box_float_ptr that allocates on every execution
                // (haxe_box_* is invisible to insert_free.rs). Skipping the
                // harmonisation is not an option — the phi would merge f64 with
                // Ptr. Null<T> cases such as `if (x == null) null else x.field`
                // have a Dynamic/Null result type and keep the boxing path.
                let result_is_scalar = result_ty
                    .and_then(|t| self.type_table.get(t))
                    .map(|t| matches!(t.kind, TypeKind::Float | TypeKind::Int | TypeKind::Bool))
                    .unwrap_or(false);

                // ...and only when the pointer side is literally `null`. A pointer
                // branch carrying a real value (e.g. `x` in `(x == null) ? 0 : x`
                // where x is Null<Int>) must keep the box/unbox path; zeroing it
                // there would silently yield the wrong value.
                let then_is_null_lit = matches!(then_expr.kind, HirExprKind::Null);
                let else_is_null_lit = matches!(else_expr.kind, HirExprKind::Null);

                if result_is_scalar && then_is_ptr && else_is_prim && then_is_null_lit {
                    let cur = self.builder.current_block();
                    self.builder.switch_to_block(then_end_block);
                    if let Some(z) = self.zero_of_ir_type(&else_rty.clone().unwrap()) {
                        then_val = Some(z);
                    }
                    if let Some(c) = cur {
                        self.builder.switch_to_block(c);
                    }
                } else if result_is_scalar && else_is_ptr && then_is_prim && else_is_null_lit {
                    let cur = self.builder.current_block();
                    self.builder.switch_to_block(else_end_block);
                    if let Some(z) = self.zero_of_ir_type(&then_rty.clone().unwrap()) {
                        else_val = Some(z);
                    }
                    if let Some(c) = cur {
                        self.builder.switch_to_block(c);
                    }
                } else if then_is_ptr && else_is_prim {
                    // Box the else value (primitive) to match the then pointer
                    self.builder.switch_to_block(else_end_block);
                    let boxed = self.box_primitive_to_dynamic(ev, else_rty.unwrap());
                    if let Some(b) = boxed {
                        else_val = Some(b);
                    }
                } else if else_is_ptr && then_is_prim {
                    // Box the then value (primitive) to match the else pointer
                    self.builder.switch_to_block(then_end_block);
                    let boxed = self.box_primitive_to_dynamic(tv, then_rty.unwrap());
                    if let Some(b) = boxed {
                        then_val = Some(b);
                    }
                }
            }
        }

        // Per-branch interface wrap for divergent-class conditionals. When the
        // two branches are different concrete classes upcast to a shared
        // interface, phi-ing the raw objects and wrapping once downstream binds
        // every vtable slot to the then-branch's class (find_common_supertype
        // returns the first type), so the else branch's object dispatches through
        // the wrong method table. Wrapping inside each branch block makes the phi
        // merge two already-correct fat pointers.
        let mut branch_iface_wrapped = false;
        if !then_terminated && !else_terminated {
            if let (Some(tv), Some(ev)) = (then_val, else_val) {
                if let (Some(tc), Some(ec)) = (
                    self.get_class_symbol(then_expr.ty),
                    self.get_class_symbol(else_expr.ty),
                ) {
                    if tc != ec {
                        if let Some(iface) = self.shared_interface_for(tc, ec) {
                            self.builder.switch_to_block(then_end_block);
                            if let Some(w) = self.wrap_in_interface_fat_ptr(tv, tc, iface) {
                                self.interface_wrapped_args.insert(w);
                                then_val = Some(w);
                                branch_iface_wrapped = true;
                            }
                            self.builder.switch_to_block(else_end_block);
                            if let Some(w) = self.wrap_in_interface_fat_ptr(ev, ec, iface) {
                                self.interface_wrapped_args.insert(w);
                                else_val = Some(w);
                            }
                        }
                    }
                }
            }
        }

        // Now build branches for non-terminated blocks
        if !then_terminated {
            self.builder.switch_to_block(then_end_block);
            self.builder.build_branch(merge_block)?;
        }
        if !else_terminated {
            self.builder.switch_to_block(else_end_block);
            self.builder.build_branch(merge_block)?;
        }
        // Both branches returned/broke/continued: there is no value, and the
        // merge block would be unreachable.
        if then_terminated && else_terminated {
            return None;
        }

        self.builder.switch_to_block(merge_block);

        // Find variables that were modified in either branch
        let mut modified_symbols = std::collections::BTreeSet::new();

        for (sym, reg_after_then) in &symbol_map_after_then {
            if symbol_map_before.get(sym) != Some(reg_after_then) {
                modified_symbols.insert(*sym);
            }
        }
        for (sym, reg_after_else) in &symbol_map_after_else {
            if symbol_map_before.get(sym) != Some(reg_after_else) {
                modified_symbols.insert(*sym);
            }
        }

        for symbol_id in &modified_symbols {
            let before_reg = symbol_map_before.get(symbol_id).copied();
            let then_reg = symbol_map_after_then.get(symbol_id).copied();
            let else_reg = symbol_map_after_else.get(symbol_id).copied();

            // Take the type from the "before" register (the variable declaration):
            // registers created by assignments have no locals entry.
            let type_lookup_reg = before_reg.or(then_reg).or(else_reg);
            let var_type = match type_lookup_reg.and_then(|r| {
                self.builder
                    .current_function()
                    .and_then(|f| f.locals.get(&r))
                    .map(|local| local.ty.clone())
            }) {
                Some(t) => t,
                None => {
                    continue;
                }
            };

            // Only phi variables that have a value from every non-terminated
            // branch; branch-local variables would produce an invalid phi.
            let has_then_value = !then_terminated && (then_reg.is_some() || before_reg.is_some());
            let has_else_value = !else_terminated && (else_reg.is_some() || before_reg.is_some());

            if (!then_terminated && !has_then_value) || (!else_terminated && !has_else_value) {
                continue;
            }

            let sample_reg = then_reg.or(else_reg).or(before_reg).unwrap();

            let phi_reg = match self.builder.build_phi(merge_block, var_type.clone()) {
                Some(r) => r,
                None => {
                    continue;
                }
            };

            // Add incoming edges for non-terminated branches.
            if !then_terminated {
                // then_reg if it exists, otherwise before_reg; else_reg here would
                // violate SSA dominance.
                if let Some(val) = then_reg.or(before_reg) {
                    self.builder
                        .add_phi_incoming(merge_block, phi_reg, then_end_block, val);
                }
            }
            if !else_terminated {
                // else_reg if it exists, otherwise before_reg; then_reg here would
                // violate SSA dominance.
                if let Some(val) = else_reg.or(before_reg) {
                    self.builder
                        .add_phi_incoming(merge_block, phi_reg, else_end_block, val);
                }
            }

            if let Some(func) = self.builder.current_function_mut() {
                if let Some(local) = func.locals.get(&sample_reg).cloned() {
                    func.locals.insert(
                        phi_reg,
                        crate::ir::IrLocal {
                            name: format!("{}_phi", local.name),
                            ty: var_type.clone(),
                            mutable: true,
                            source_location: local.source_location,
                            allocation: crate::ir::AllocationHint::Register,
                        },
                    );
                }
            }

            self.symbol_map.insert(*symbol_id, phi_reg);
        }

        // Create phi for expression result if both branches returned values
        let mut result_phi = None;

        // Asymmetric case: one branch terminated (throw / return / unreachable
        // infinite loop) while the other produced a value. The if-as-expression
        // still has that value at the merge point, unconditionally, and the merge
        // block has a single predecessor — reuse the live value rather than
        // building a phi. Returning None would lower
        // `var x:Int = if (cond) 42 else throw "..."` to `Return(None)`, which
        // the cranelift backend rejects for a typed-return function.
        if !then_terminated && else_terminated && then_val.is_some() {
            return then_val;
        }
        if then_terminated && !else_terminated && else_val.is_some() {
            return else_val;
        }

        // Expression-style ifs only get a result phi when both branches yield a
        // value; only one yielding a value is a type error.
        if then_val.is_some() && else_val.is_some() {
            // Take the result type from the register types after harmonization,
            // not the original HIR type (which may be pre-boxing, e.g. I32 before
            // harmonization boxed it to Ptr(U8) to match a null branch).
            let result_type = if let Some(tv) = then_val {
                self.builder
                    .get_register_type(tv)
                    .unwrap_or_else(|| self.convert_type(then_expr.ty))
            } else {
                self.convert_type(then_expr.ty)
            };
            let result = match self.builder.build_phi(merge_block, result_type.clone()) {
                Some(r) => r,
                None => {
                    return None;
                }
            };
            // The branches were wrapped to a shared interface above, so the phi
            // is itself an interface fat pointer — mark it so the enclosing sink
            // (Let/return/call-arg) doesn't class-wrap it a second time.
            if branch_iface_wrapped {
                self.interface_wrapped_args.insert(result);
            }

            if !then_terminated {
                let val = then_val.unwrap(); // Safe because we checked is_some() above
                self.builder
                    .add_phi_incoming(merge_block, result, then_end_block, val);
            }
            if !else_terminated {
                let val = else_val.unwrap(); // Safe because we checked is_some() above
                self.builder
                    .add_phi_incoming(merge_block, result, else_end_block, val);
            }
            // Cross-context interface-return propagation through the phi. When
            // both arms are interface method calls whose concrete return type
            // was recovered (tracked in `interface_call_result_types`) and they
            // agree, the merged result is that same concrete type. Without this
            // a `cond ? iface.a() : iface.b()` result stays erased to Dynamic,
            // and a downstream `var x:T = …` unboxes a raw pointer. Require
            // agreement so a mixed merge is never mislabelled.
            if let (Some(tv), Some(ev)) = (then_val, else_val) {
                if let (Some(&t_ty), Some(&e_ty)) = (
                    self.interface_call_result_types.get(&tv),
                    self.interface_call_result_types.get(&ev),
                ) {
                    if t_ty == e_ty {
                        self.interface_call_result_types.insert(result, t_ty);
                    }
                }
            }
            result_phi = Some(result);
        }

        result_phi
    }
}
