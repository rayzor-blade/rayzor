//! Virtual calls: dispatch through the receiver's vtable slot.

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
    pub(crate) fn try_virtual_dispatch(
        &mut self,
        expr: &HirExpr,
        receiver_is_super: bool,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_virtual_dispatch on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if *is_method && !args.is_empty() && !receiver_is_super {
            let vtable_slot = self.virtual_dispatch_info.get(symbol).copied().or_else(|| {
                let method_name = self.symbol_table.get_symbol(*symbol).map(|s| s.name)?;
                let receiver_type = self.resolve_through_aliases(args[0].ty);
                let type_table = self.type_table;
                let class_sym = match &type_table.get(receiver_type)?.kind {
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    _ => None,
                }?;
                let mut current = Some(class_sym);
                while let Some(cls) = current {
                    if let Some(&method_sym) = self.class_method_by_name.get(&(cls, method_name)) {
                        if let Some(info) = self.virtual_dispatch_info.get(&method_sym) {
                            return Some(*info);
                        }
                    }
                    current = self.class_parent_map.get(&cls).copied();
                }
                None
            });

            if let Some((slot_index, _defining_class)) = vtable_slot {
                let obj_reg = self.lower_expression(&args[0])?;
                let mut call_args = vec![obj_reg];
                for arg in args.iter().skip(1) {
                    if let Some(reg) = self.lower_expression(arg) {
                        call_args.push(reg);
                    }
                }
                let lookup_fn = self.get_or_register_extern_function(
                    "haxe_vtable_lookup",
                    vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                    IrType::I64,
                );
                let slot_reg = self.builder.build_const(IrValue::I32(slot_index as i32))?;
                let fn_ptr = self.builder.build_call_direct(
                    lookup_fn,
                    vec![obj_reg, slot_reg],
                    IrType::I64,
                )?;
                let mut param_types = vec![IrType::Ptr(Box::new(IrType::Void))];
                for arg in args.iter().skip(1) {
                    param_types.push(self.convert_type(arg.ty));
                }
                let return_type = Box::new(self.convert_type(expr.ty));
                let func_signature = IrType::Function {
                    params: param_types,
                    return_type,
                    varargs: false,
                };
                return self
                    .builder
                    .build_call_indirect(fn_ptr, call_args, func_signature);
            }
        }
        *fell_through = true;
        None
    }
}
