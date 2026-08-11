//! Calls on an interface-typed receiver: dispatch through the fat pointer's vtable.

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
    pub(crate) fn try_interface_dispatch(
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
            unreachable!("try_interface_dispatch on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if *is_method && !args.is_empty() {
            let receiver = &args[0];
            let receiver_type = receiver.ty;

            if let Some(iface_sym) = self.get_interface_symbol(receiver_type) {
                let method_name_interned = self.symbol_table.get_symbol(*symbol).map(|s| s.name);

                if let Some(method_name_i) = method_name_interned {
                    let method_index = self
                        .resolve_interface_method_names(iface_sym)
                        .and_then(|names| names.iter().position(|n| *n == method_name_i));

                    if let Some(idx) = method_index {
                        // Lower the receiver (fat pointer)
                        let fat_ptr_raw = self.lower_expression(receiver)?;

                        // The fat pointer may be stored as I64 - bitcast to Ptr if needed
                        let fat_ptr_ty = self
                            .builder
                            .get_register_type(fat_ptr_raw)
                            .unwrap_or(IrType::I64);
                        let fat_ptr = if !matches!(fat_ptr_ty, IrType::Ptr(_)) {
                            self.builder
                                .build_bitcast(fat_ptr_raw, IrType::Ptr(Box::new(IrType::I64)))?
                        } else {
                            fat_ptr_raw
                        };

                        // Lower the actual arguments (skip args[0] which is receiver)
                        let arg_regs: Vec<_> = args[1..]
                            .iter()
                            .filter_map(|a| self.lower_expression(a))
                            .collect();

                        // Load object pointer from fat_ptr[0]
                        let obj_ptr = self.builder.build_load(fat_ptr, IrType::I64)?;

                        // Load function pointer from fat_ptr[(idx+1)*8]
                        let fn_offset = self
                            .builder
                            .build_const(IrValue::I64(((idx + 1) * 8) as i64))?;
                        let fn_slot = self.builder.build_ptr_add(
                            fat_ptr,
                            fn_offset,
                            IrType::Ptr(Box::new(IrType::U8)),
                        )?;
                        let fn_ptr = self.builder.build_load(fn_slot, IrType::I64)?;

                        // Build call args: self (obj_ptr) + user args
                        let mut call_args = vec![obj_ptr];
                        call_args.extend(arg_regs);

                        // Build signature: (self: Ptr, args...) -> return_type
                        let param_types = {
                            let mut types = vec![IrType::Ptr(Box::new(IrType::Void))]; // self
                            for arg in args[1..].iter() {
                                types.push(self.convert_type(arg.ty));
                            }
                            types
                        };
                        // Resolve return type from the method's symbol type,
                        // not expr.ty (which may be the interface type instead
                        // of the method's return type in some TAST configurations)
                        let (return_ir_type, resolved_ret_type_id) =
                            self.resolve_interface_method_return_type_full(*symbol, expr.ty);
                        self.emit_iface_return_diagnostic(
                            *symbol,
                            expr.ty,
                            resolved_ret_type_id,
                            expr.source_location,
                        );
                        let return_type = Box::new(return_ir_type);
                        let func_signature = IrType::Function {
                            params: param_types,
                            return_type,
                            varargs: false,
                        };

                        let call_result =
                            self.builder
                                .build_call_indirect(fn_ptr, call_args, func_signature)?;
                        if let Some(real_ty) = resolved_ret_type_id {
                            self.interface_call_result_types
                                .insert(call_result, real_ty);
                        }
                        return Some(call_result);
                    }
                }
            }
        }
        *fell_through = true;
        None
    }
}
