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
            arg_regs.push(reg);
        }
        debug!(
            "Lowered {} indirect call arguments successfully",
            arg_regs.len()
        );

        let func_ptr = self.lower_expression(callee)?;

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
}
