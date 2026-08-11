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
        // Indirect function call (function pointer)
        // TODO: Get the full function signature from the callee's type
        // For now, we'll infer it from the arguments and return type
        // This is a temporary workaround until we pass type_table to HIR→MIR
        self.builder.call_label = Some("INDIRECT_CALL".to_string());

        debug!(
            "Taking indirect function call path - callee kind={:?}, args.len()={}",
            std::mem::discriminant(&callee.kind),
            args.len()
        );

        // Lower arguments FIRST, before trying to lower the callee
        // This ensures lambdas in arguments get generated even if callee lowering fails
        debug!("About to lower {} indirect call arguments", args.len());
        for (i, a) in args.iter().enumerate() {
            debug!("  arg[{}] kind={:?}", i, std::mem::discriminant(&a.kind));
        }
        let arg_regs: Vec<_> = args
            .iter()
            .filter_map(|a| {
                debug!(
                    "NOW lowering arg with kind={:?}",
                    std::mem::discriminant(&a.kind)
                );
                self.lower_expression(a)
            })
            .collect();
        debug!(
            "Lowered {} indirect call arguments successfully",
            arg_regs.len()
        );

        // Now try to lower the callee - if this fails, the call won't be generated
        // but the lambda functions in arguments will have been created
        let func_ptr = self.lower_expression(callee)?;

        // Build function signature from callee type or argument types
        let param_types: Vec<IrType> = {
            // Try to get param types from callee's function type
            let type_table = self.type_table;
            let callee_type = type_table.get(callee.ty);
            if let Some(type_ref) = callee_type {
                if let crate::tast::TypeKind::Function { params, .. } = &type_ref.kind {
                    // `Void -> T` is Haxe's spelling for "takes nothing",
                    // so a Void entry is notation rather than a slot. Left
                    // in, the signature has a parameter no call site can
                    // fill: LLVM rejects a Void parameter outright, and
                    // Cranelift asserts on the argument count once the
                    // closure is called with its environment.
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

        let func_signature = IrType::Function {
            params: param_types,
            return_type,
            varargs: false,
        };

        self.builder
            .build_call_indirect(func_ptr, arg_regs, func_signature)
    }
}
