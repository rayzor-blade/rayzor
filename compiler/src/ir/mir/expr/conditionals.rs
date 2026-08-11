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
        // Short-circuit AND: if lhs is false, don't evaluate rhs
        // Create blocks: eval_rhs, merge
        let eval_rhs = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        // Evaluate LHS
        let lhs_val = self.lower_expression(lhs)?;

        // Create false_val BEFORE branching so it's in this block's scope
        let false_val = self.builder.build_bool(false)?;

        // Capture the current block BEFORE branching - this is where LHS was evaluated
        let lhs_block = self.builder.current_block()?;

        // Branch on LHS: if true, evaluate RHS; if false, skip to merge with false
        self.builder.build_cond_branch(lhs_val, eval_rhs, merge)?;

        // Block for evaluating RHS
        self.builder.switch_to_block(eval_rhs);
        let rhs_val = self.lower_expression(rhs)?;
        let rhs_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        // Merge block with phi node
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
        // Short-circuit OR: if lhs is true, don't evaluate rhs
        // Create blocks: eval_rhs, merge
        let eval_rhs = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        // Evaluate LHS
        let lhs_val = self.lower_expression(lhs)?;

        // Create true_val BEFORE branching so it's in this block's scope
        let true_val = self.builder.build_bool(true)?;

        // Capture the current block BEFORE branching - this is where LHS was evaluated
        let lhs_block = self.builder.current_block()?;

        // Branch on LHS: if false, evaluate RHS; if true, skip to merge with true
        self.builder.build_cond_branch(lhs_val, merge, eval_rhs)?;

        // Block for evaluating RHS
        self.builder.switch_to_block(eval_rhs);
        let rhs_val = self.lower_expression(rhs)?;
        let rhs_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        // Merge block with phi node
        self.builder.switch_to_block(merge);
        let result = self.builder.build_phi(merge, IrType::Bool)?;
        // lhs_block is where we came from if LHS was true (short-circuit path)
        self.builder
            .add_phi_incoming(merge, result, lhs_block, true_val)?;
        self.builder
            .add_phi_incoming(merge, result, rhs_block, rhs_val)?;

        Some(result)
    }

    /// Lower `Null<prim>` equality. The nullable is a DynamicValue box, so
    /// the generic path compared the BOX POINTER against the value —
    /// `var x:Null<Int> = 9; x == 9` was always false. Routes
    /// Optional<Int|Float|Bool> vs bare prim (either order) and
    /// Optional vs Optional through null-guarded tag-aware runtime
    /// compares. Returns None when the shape doesn't apply (caller falls
    /// through to the generic compare); `x == null` never reaches here
    /// (HirExprKind::Null operands are excluded at the call site).
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
        // user-fn return), but some stdlib calls return a RAW primitive despite
        // a Null<T> type — e.g. Std.parseInt returns i64 (i64::MIN = null). A raw
        // value reinterpreted as a pointer and dereferenced by the helper
        // SIGSEGVs, so box it first (this mirrors what a `var x:Null<T> = ...`
        // assignment does). Detect "raw" by the register not being a pointer.
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
        // Null coalescing: lhs ?? rhs
        // If lhs is non-null, return lhs; otherwise evaluate and return rhs
        //
        // Uses intermediary blocks (like ternary) to avoid Cranelift phi issues
        // when br_if targets a merge block directly.
        let lhs_pass = self.builder.create_block()?;
        let eval_rhs = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        // Check if LHS is Optional{primitive} — needs unbox in pass-through
        let opt_prim = self.is_optional_primitive(lhs.ty);

        // Evaluate LHS
        let lhs_val = self.lower_expression(lhs)?;

        // Null check: lhs != 0 (null pointers and null values are 0)
        // Use same type as LHS to avoid type mismatch in comparison
        let lhs_ir_type = self.builder.get_register_type(lhs_val);
        let zero = match lhs_ir_type {
            Some(IrType::I32) => self.builder.build_const(IrValue::I32(0))?,
            Some(IrType::F32) => self.builder.build_const(IrValue::F32(0.0))?,
            Some(IrType::F64) => self.builder.build_const(IrValue::F64(0.0))?,
            _ => self.builder.build_const(IrValue::I64(0))?,
        };
        let is_not_null = self.builder.build_cmp(CompareOp::Ne, lhs_val, zero)?;

        // If not null -> lhs_pass block, else -> evaluate rhs
        self.builder
            .build_cond_branch(is_not_null, lhs_pass, eval_rhs)?;

        // LHS pass-through block — unbox if Optional{primitive}
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

        // Block for evaluating RHS (only when lhs is null)
        self.builder.switch_to_block(eval_rhs);
        let rhs_val = self.lower_expression(rhs)?;
        let rhs_block = self.builder.current_block()?;
        self.builder.build_branch(merge)?;

        // Merge block with phi node
        // When LHS was Optional{primitive}, result type is the unboxed primitive type (RHS type)
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
        // Conditional expression: cond ? then : else
        //
        // Becomes:
        //   %cond_val = <evaluate cond>
        //   br %cond_val, then_block, else_block
        // then_block:
        //   %then_val = <evaluate then>
        //   br merge_block
        // else_block:
        //   %else_val = <evaluate else>
        //   br merge_block
        // merge_block:
        //   %result = phi [%then_val, then_block], [%else_val, else_block]
        //   (plus phi nodes for any variables modified in branches)

        let then_block = self.builder.create_block()?;
        let else_block = self.builder.create_block()?;
        let merge_block = self.builder.create_block()?;

        // Snapshot symbol_map before branches
        let symbol_map_before = self.symbol_map.clone();
        // eprintln!(
        //     "DEBUG lower_conditional: symbol_map has {} entries before condition",
        //     symbol_map_before.len()
        // );

        // Evaluate condition
        let cond_val = self.lower_expression(cond)?;
        // eprintln!(
        //     "DEBUG lower_conditional: After evaluating condition, in block {:?}",
        //     self.builder.current_block()
        // );

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

        // Branch based on condition
        self.builder
            .build_cond_branch(cond_val, then_block, else_block)?;

        // Then block
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
            // NOTE: branch building deferred until after type harmonization
        }
        let then_end_block = self.builder.current_block()?;
        let symbol_map_after_then = self.symbol_map.clone();

        // Else block
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
            // NOTE: branch building deferred until after type harmonization
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
                // harmonise DOWN to that scalar instead of boxing the live
                // branch up to a pointer. The pointer side is a synthesised
                // null for a missing else (tast_to_hir::make_null_literal) and
                // is never the value the expression yields; replacing it with a
                // zero of the scalar type keeps the phi well-typed at no cost.
                //
                // Boxing instead produced a SECOND, dead phi fed by
                // haxe_box_float_ptr, allocating on every execution:
                // Int8Matmul.quantizeActRow does this three times per row, per
                // matmul, per token -- 1,876,152 boxes in one generation, none
                // freed (haxe_box_* is invisible to insert_free.rs).
                //
                // Simply SKIPPING the harmonisation is not an option: the phi
                // then merges f64 with Ptr and segfaults.
                //
                // Null<T> cases such as `if (x == null) null else x.field` have
                // a Dynamic/Null result type, so they keep the boxing path.
                let result_is_scalar = result_ty
                    .and_then(|t| self.type_table.get(t))
                    .map(|t| matches!(t.kind, TypeKind::Float | TypeKind::Int | TypeKind::Bool))
                    .unwrap_or(false);

                // ...and ONLY when the pointer side is literally `null`. A
                // pointer branch carrying a REAL value (e.g. `x` in
                // `(x == null) ? 0 : x` where x is Null<Int>) must still be
                // boxed/unboxed by the existing path: zeroing it there turned a
                // crash into `nc(9) == 0`, a silent wrong answer, which is worse
                // than the crash it replaced.
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

        // Per-branch interface wrap for divergent-class conditionals.
        // When the two branches are DIFFERENT concrete classes upcast to a
        // shared interface, phi-ing the raw objects and wrapping ONCE downstream
        // binds every vtable slot to a single branch's class (the then-branch's,
        // since find_common_supertype returns the first type) — the other
        // branch's object then dispatches through the wrong method table. Wrap
        // each branch to the shared interface INSIDE its own block so the phi
        // merges two already-correct fat pointers.
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
        // eprintln!(
        //     "DEBUG lower_conditional: else_end_block = {:?}, symbol_map has {} entries",
        //     else_end_block,
        //     symbol_map_after_else.len()
        // );

        // If both branches terminated, no merge block needed
        if then_terminated && else_terminated {
            // Both branches returned/broke/continued
            // No value to return, and we shouldn't create unreachable merge block
            return None;
        }

        // Merge block with phi nodes
        self.builder.switch_to_block(merge_block);

        // Find variables that were modified in either branch
        let mut modified_symbols = std::collections::BTreeSet::new();
        // debug!("Checking for modified symbols");
        // eprintln!("  symbol_map_before: {} entries", symbol_map_before.len());
        // eprintln!(
        //     "  symbol_map_after_then: {} entries",
        //     symbol_map_after_then.len()
        // );
        // eprintln!(
        //     "  symbol_map_after_else: {} entries",
        //     symbol_map_after_else.len()
        // );

        for (sym, reg_after_then) in &symbol_map_after_then {
            if symbol_map_before.get(sym) != Some(reg_after_then) {
                // eprintln!(
                //     "  Modified in then branch: {:?} (before: {:?}, after: {:?})",
                //     sym,
                //     symbol_map_before.get(sym),
                //     reg_after_then
                // );
                modified_symbols.insert(*sym);
            }
        }
        for (sym, reg_after_else) in &symbol_map_after_else {
            if symbol_map_before.get(sym) != Some(reg_after_else) {
                // eprintln!(
                //     "  Modified in else branch: {:?} (before: {:?}, after: {:?})",
                //     sym,
                //     symbol_map_before.get(sym),
                //     reg_after_else
                // );
                modified_symbols.insert(*sym);
            }
        }
        // debug!("Found {} modified symbols", modified_symbols.len());

        for symbol_id in &modified_symbols {
            // eprintln!("  Processing symbol {:?}", symbol_id);
            let before_reg = symbol_map_before.get(symbol_id).copied();
            let then_reg = symbol_map_after_then.get(symbol_id).copied();
            let else_reg = symbol_map_after_else.get(symbol_id).copied();

            // Get type from locals table using the "before" register (from variable declaration)
            // because new registers from assignments don't have local entries
            let type_lookup_reg = before_reg.or(then_reg).or(else_reg);
            let var_type = match type_lookup_reg.and_then(|r| {
                self.builder
                    .current_function()
                    .and_then(|f| f.locals.get(&r))
                    .map(|local| local.ty.clone())
            }) {
                Some(t) => {
                    // eprintln!("  Found type {:?} for symbol {:?}", t, symbol_id);
                    t
                }
                None => {
                    // eprintln!(
                    //     "  No type found for symbol {:?} (tried {:?}), skipping",
                    //     symbol_id, type_lookup_reg
                    // );
                    continue;
                }
            };

            // Only create phi nodes for variables that have values from all non-terminated branches
            // This prevents creating invalid phi nodes for branch-local variables
            let has_then_value = !then_terminated && (then_reg.is_some() || before_reg.is_some());
            let has_else_value = !else_terminated && (else_reg.is_some() || before_reg.is_some());

            // Skip if we can't provide values from all active branches
            if (!then_terminated && !has_then_value) || (!else_terminated && !has_else_value) {
                // eprintln!("  Skipping phi for {:?} - not in all branches", symbol_id);
                continue;
            }

            let sample_reg = then_reg.or(else_reg).or(before_reg).unwrap();

            // Create phi node
            // eprintln!(
            //     "  Creating phi for {:?} with type {:?}",
            //     symbol_id, var_type
            // );
            let phi_reg = match self.builder.build_phi(merge_block, var_type.clone()) {
                Some(r) => r,
                None => {
                    // eprintln!("  Failed to create phi node");
                    continue;
                }
            };
            // eprintln!("  Created phi node {:?}", phi_reg);

            // Add incoming edges for non-terminated branches
            // IMPORTANT: Only add phi incoming if the variable exists in that branch
            // Don't use variables from other branches (causes domination errors)
            // eprintln!(
            //     "  Adding phi incoming: then_terminated={}, else_terminated={}",
            //     then_terminated, else_terminated
            // );
            if !then_terminated {
                // Use then_reg if it exists, otherwise before_reg
                // Do NOT use else_reg here - it would violate SSA dominance
                if let Some(val) = then_reg.or(before_reg) {
                    // eprintln!(
                    //     "  Calling add_phi_incoming(merge={:?}, phi={:?}, from={:?}, val={:?})",
                    //     merge_block, phi_reg, then_end_block, val
                    // );
                    self.builder
                        .add_phi_incoming(merge_block, phi_reg, then_end_block, val);
                    // {
                    //     Some(()) => eprintln!("  Successfully added phi incoming from then"),
                    //     None => eprintln!(
                    //         "  WARNING: Failed to add phi incoming from then block {:?}",
                    //         then_end_block
                    //     ),
                    // }
                }
            }
            if !else_terminated {
                // Use else_reg if it exists, otherwise before_reg
                // Do NOT use then_reg here - it would violate SSA dominance
                if let Some(val) = else_reg.or(before_reg) {
                    // eprintln!(
                    //     "  Calling add_phi_incoming(merge={:?}, phi={:?}, from={:?}, val={:?})",
                    //     merge_block, phi_reg, else_end_block, val
                    // );
                    self.builder
                        .add_phi_incoming(merge_block, phi_reg, else_end_block, val);
                    // {
                    //     Some(()) => eprintln!("  Successfully added phi incoming from else"),
                    //     None => eprintln!(
                    //         "  WARNING: Failed to add phi incoming from else block {:?}",
                    //         else_end_block
                    //     ),
                    // }
                }
            }

            // Register phi as local
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

            // Update symbol map to use phi
            self.symbol_map.insert(*symbol_id, phi_reg);
        }

        // Create phi for expression result if both branches returned values
        let mut result_phi = None;

        // Asymmetric case: ONE branch terminated (throw / return /
        // unreachable infinite loop) while the OTHER produced a
        // value. The if-as-expression still has a value at the merge
        // point — it's just the non-terminated branch's value
        // unconditionally, because the terminated branch never
        // reaches merge. Without this, e.g.
        //
        //   var x:Int = if (cond) 42 else throw "...";
        //
        // returns `None` from `lower_conditional_typed`, and the
        // surrounding `return x` becomes `Return(None)` in MIR —
        // which the cranelift backend rejects for a typed-return
        // function ("Return with no value"). The merge block has a
        // single predecessor (the value branch), so no phi is
        // needed; reuse the live value directly. Same applies for
        // `return switch { … case _: throw … };` where the throw is
        // the if-chain's terminal else.
        if !then_terminated && else_terminated && then_val.is_some() {
            return then_val;
        }
        if then_terminated && !else_terminated && else_val.is_some() {
            return else_val;
        }

        // Only create result phi if BOTH branches return values (for expression-style ifs)
        // If only one returns a value, that's a type error - skip result phi
        if then_val.is_some() && else_val.is_some() {
            // Determine result type from the ACTUAL register types after harmonization,
            // not the original HIR type (which may be pre-boxing, e.g. I32 before the
            // type harmonization boxed it to Ptr(U8) to match a null branch).
            let result_type = if let Some(tv) = then_val {
                self.builder
                    .get_register_type(tv)
                    .unwrap_or_else(|| self.convert_type(then_expr.ty))
            } else {
                self.convert_type(then_expr.ty)
            };
            let result = match self.builder.build_phi(merge_block, result_type.clone()) {
                Some(r) => {
                    // debug!("Created result phi {:?}", r);
                    r
                }
                None => {
                    // debug!("Failed to create result phi");
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
                // {
                //     Some(()) => debug!("  Success"),
                //     None => debug!("  FAILED!"),
                // }
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
            // and a downstream `var x:T = …` unboxes a raw pointer (→ SIGSEGV
            // for object returns). Require agreement so a mixed merge is never
            // mislabelled.
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
