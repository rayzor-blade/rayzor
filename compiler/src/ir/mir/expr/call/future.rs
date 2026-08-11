//! `@:async` handles: `.await()`, `.poll()`, `.isReady()` on a Future.

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
    pub(crate) fn try_future_method_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_future_method_call on a non-Call expression")
        };
        if *is_method {
            // Get method name from callee symbol
            let async_method_sym = match &callee.kind {
                HirExprKind::Variable { symbol, .. } => Some(*symbol),
                HirExprKind::Field { field, .. } => Some(*field),
                _ => None,
            };
            // Get receiver symbol from first arg (MethodCall puts receiver as args[0])
            let receiver_sym_from_args = args.first().and_then(|a| {
                if let HirExprKind::Variable { symbol, .. } = &a.kind {
                    Some(*symbol)
                } else {
                    None
                }
            });
            if let (Some(method_sym), Some(recv_sym)) = (async_method_sym, receiver_sym_from_args) {
                let receiver_reg = self.symbol_map.get(&recv_sym).copied();
                if let Some(recv_reg) = receiver_reg {
                    if self.async_result_registers.contains(&recv_reg) {
                        let method_name = self
                            .symbol_table
                            .get_symbol(method_sym)
                            .and_then(|s| self.string_interner.get(s.name));
                        if let Some(method) = method_name {
                            let ext_name = match method {
                                "await" => Some("rayzor_future_await"),
                                "poll" => Some("rayzor_future_poll"),
                                "isReady" => Some("rayzor_future_is_ready"),
                                _ => None,
                            };
                            if let Some(extern_name) = ext_name {
                                // Look up the extern directly (declared by ensure_future_externs)
                                let func_id = self
                                    .builder
                                    .module
                                    .extern_functions
                                    .iter()
                                    .find(|(_, f)| f.name == extern_name)
                                    .map(|(id, _)| *id);
                                if let Some(func_id) = func_id {
                                    let ret_ty = if method == "isReady" {
                                        IrType::Bool
                                    } else {
                                        IrType::Ptr(Box::new(IrType::U8))
                                    };
                                    return self.builder.build_call_direct(
                                        func_id,
                                        vec![recv_reg],
                                        ret_ty,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        *fell_through = true;
        None
    }
}
