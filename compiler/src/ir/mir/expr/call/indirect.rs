//! The terminal case: a call through a function pointer.

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
    pub(crate) fn lower_indirect_call(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("lower_indirect_call on a non-Call expression")
        };
        self.builder.call_label = Some("INDIRECT_CALL".to_string());

        debug!(
            "Taking indirect function call path - callee kind={:?}, args.len()={}",
            std::mem::discriminant(&callee.kind),
            args.len()
        );

        // Formal parameter types from the callee's function type. A `Void`
        // entry is Haxe's spelling for "takes nothing", not a slot.
        let formal_tys: Option<Vec<TypeId>> = {
            let type_table = self.type_table;
            type_table.get(callee.ty).and_then(|t| match &t.kind {
                crate::tast::TypeKind::Function { params, .. } => Some(
                    params
                        .iter()
                        .copied()
                        .filter(|p| {
                            !matches!(type_table.get(*p).map(|t| &t.kind), Some(TypeKind::Void))
                        })
                        .collect(),
                ),
                _ => None,
            })
        };

        // Arguments are lowered before the callee, so lambdas passed as
        // arguments are still generated when callee lowering fails.
        debug!("About to lower {} indirect call arguments", args.len());
        // Every argument must lower. Dropping the ones that fail would keep the
        // call but shift the survivors into the wrong parameter slots, so a
        // miscompiled argument becomes a silently miscompiled call.
        let mut arg_regs: Vec<IrId> = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            debug!("  arg[{}] kind={:?}", i, std::mem::discriminant(&a.kind));
            let Some(reg) = self.lower_expression(a) else {
                warn!(
                    "indirect call: argument {} of {} failed to lower ({:?}); abandoning the call",
                    i,
                    args.len(),
                    std::mem::discriminant(&a.kind)
                );
                return None;
            };
            // A scalar or String handed to a `Dynamic` formal must box, as the
            // direct-call path does: the callee unboxes those. It does NOT unbox
            // a function, anonymous, class or enum value received as `Dynamic` —
            // it uses the raw pointer, so boxing those here breaks the callee.
            let boxable = {
                let type_table = self.type_table;
                matches!(
                    type_table.get(a.ty).map(|t| &t.kind),
                    Some(TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::String)
                )
            };
            let reg = match formal_tys.as_ref().and_then(|f| f.get(i).copied()) {
                Some(formal) if boxable => self.maybe_box_value(reg, a.ty, formal).unwrap_or(reg),
                Some(formal) => self
                    .unbox_optional_for_erased_formal(reg, a.ty, formal)
                    .unwrap_or(reg),
                None => reg,
            };
            arg_regs.push(reg);
        }
        debug!(
            "Lowered {} indirect call arguments successfully",
            arg_regs.len()
        );

        let func_ptr = self.lower_expression(callee)?;
        // A function held in a Dynamic is a box; the call goes to the
        // closure inside it, not to the box's tag.
        let func_ptr = self.unbox_dynamic_function(func_ptr, callee);

        // A callee typed Dynamic has no signature to call by: every argument
        // travels as a box to the closure's box-shaped entry, and the result
        // comes back as one.
        let callee_is_dynamic = matches!(
            self.type_table.get(callee.ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic)
        );
        if callee_is_dynamic {
            return self.lower_dynamic_closure_call(func_ptr, args, &arg_regs);
        }

        // Signature from the callee's function type, else from the arguments.
        let param_types: Vec<IrType> = {
            let type_table = self.type_table;
            let callee_type = type_table.get(callee.ty);
            if let Some(type_ref) = callee_type {
                if let crate::tast::TypeKind::Function { params, .. } = &type_ref.kind {
                    // `Void -> T` is Haxe's spelling for "takes nothing", so a
                    // Void entry is notation, not a slot. Keeping it yields a
                    // parameter no call site can fill: LLVM rejects a Void
                    // parameter, and Cranelift asserts on the argument count.
                    params
                        .iter()
                        .map(|p| self.convert_type(*p))
                        .filter(|t| !matches!(t, IrType::Void))
                        .collect()
                } else {
                    // Fallback: infer from actual argument types
                    args.iter().map(|a| self.convert_type(a.ty)).collect()
                }
            } else {
                args.iter().map(|a| self.convert_type(a.ty)).collect()
            }
        };
        let return_type = Box::new(self.convert_type(expr.ty));

        // A function value carries no defaults: `fill_default_args` works from
        // the callee's IrFunctionId, which a call through a pointer does not
        // have. Emitting the call anyway hands the backend fewer arguments than
        // the signature declares, which Cranelift reports as a failed assertion
        // inside its ABI code rather than as anything the author can act on.
        if arg_regs.len() < param_types.len() {
            self.errors.push(LoweringError {
                message: format!(
                    "function value called with {} of {} arguments; a parameter's \
                     default value is only applied when the function is called by name",
                    arg_regs.len(),
                    param_types.len()
                ),
                location: expr.source_location.clone(),
            });
            return None;
        }

        let func_signature = IrType::Function {
            params: param_types,
            return_type,
            varargs: false,
        };

        self.builder
            .build_call_indirect(func_ptr, arg_regs, func_signature)
    }

    /// `f(args)` with `f: Dynamic`: box the arguments, take the closure's
    /// box-shaped entry view and call it as `(box..) -> box`.
    fn lower_dynamic_closure_call(
        &mut self,
        closure: IrId,
        args: &[HirExpr],
        arg_regs: &[IrId],
    ) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let dynamic_ty = self.type_table.dynamic_type();
        let mut boxed_args = Vec::with_capacity(arg_regs.len());
        for (a, reg) in args.iter().zip(arg_regs) {
            let boxed = self.maybe_box_value(*reg, a.ty, dynamic_ty).unwrap_or(*reg);
            let boxed = match self.builder.get_register_type(boxed) {
                Some(IrType::Ptr(_)) | None => boxed,
                Some(other) => self
                    .builder
                    .build_cast(boxed, other, ptr_u8.clone())
                    .unwrap_or(boxed),
            };
            boxed_args.push(boxed);
        }
        let closure = match self.builder.get_register_type(closure) {
            Some(IrType::Ptr(_)) => closure,
            Some(other) => self
                .builder
                .build_cast(closure, other, ptr_u8.clone())
                .unwrap_or(closure),
            None => closure,
        };
        let view_fn = self.get_or_register_extern_function(
            "haxe_closure_dynamic_view",
            vec![ptr_u8.clone()],
            ptr_u8.clone(),
        );
        let view = self
            .builder
            .build_call_direct(view_fn, vec![closure], ptr_u8.clone())?;
        let signature = IrType::Function {
            params: vec![ptr_u8.clone(); boxed_args.len()],
            return_type: Box::new(ptr_u8),
            varargs: false,
        };
        let result = self
            .builder
            .build_call_indirect(view, boxed_args, signature)?;
        self.boxed_value_regs.insert(result);
        Some(result)
    }
}
