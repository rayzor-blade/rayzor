//! `super.method(...)`: dispatches to the parent class implementation.

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
    pub(crate) fn try_super_call(
        &mut self,
        expr: &HirExpr,
        receiver_is_super: bool,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("try_super_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if receiver_is_super {
            let method_name = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
            if let Some(method_name) = method_name {
                // The owning class of the current function gives the parent to
                // dispatch into.
                let current_class = self.builder.current_function().and_then(|f| {
                    self.class_method_by_name
                        .iter()
                        .find(|(_, &method_sym)| self.function_map.get(&method_sym) == Some(&f.id))
                        .map(|((class_sym, _), _)| *class_sym)
                });
                let parent_class =
                    current_class.and_then(|cls| self.class_parent_map.get(&cls).copied());
                let super_func_id = parent_class
                    .and_then(|pc| {
                        self.class_method_by_name
                            .get(&(pc, method_name))
                            .and_then(|&sym| self.resolve_function_id_with_qualified_fallback(sym))
                    })
                    .or_else(|| {
                        // No parent entry — resolve the called symbol directly.
                        self.resolve_function_id_with_qualified_fallback(*symbol)
                    });
                if let Some(func_id) = super_func_id {
                    let obj_reg = self.lower_expression(&args[0])?;
                    let mut call_args = vec![obj_reg];
                    for arg in args.iter().skip(1) {
                        if let Some(reg) = self.lower_expression(arg) {
                            call_args.push(reg);
                        }
                    }
                    let ret_type = self.convert_type(expr.ty);
                    return self.builder.build_call_direct(func_id, call_args, ret_type);
                }
            }
        }
        *fell_through = true;
        None
    }
}
