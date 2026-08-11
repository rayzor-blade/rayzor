//! Calls whose target is not yet defined: resolved by name, or registered as
//! a forward reference for a later module.

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
    pub(crate) fn try_forward_declared_call(
        &mut self,
        expr: &HirExpr,
        result_type: IrType,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_forward_declared_call on a non-Call expression")
        };
        if let HirExprKind::Variable { symbol, .. } = &callee.kind {
            let unresolved_call_args = if !*is_method {
                self.effective_static_call_args(args)
            } else {
                args
            };
            if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                if let Some(qual_name) = sym_info.qualified_name {
                    if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                        let _method_name = self
                            .string_interner
                            .get(sym_info.name)
                            .unwrap_or("<unknown>");
                        debug!("[PRE-CHECK] Qualified name: '{}'", qual_name_str);

                        // FIRST: Check if this function is already compiled and in the name map
                        if let Some(&existing_func_id) =
                            self.external_function_name_map.get(qual_name_str)
                        {
                            self.builder.call_label =
                                Some(format!("NAME_MAP_HIT:{}", qual_name_str));
                            debug!(
                                "[NAME MAP HIT] Found '{}' in external_function_name_map -> {:?}",
                                qual_name_str, existing_func_id
                            );

                            // Lower arguments
                            let arg_regs: Vec<_> = unresolved_call_args
                                .iter()
                                .filter_map(|a| self.lower_expression(a))
                                .collect();

                            // Generate the call to the external function
                            return self.builder.build_call_direct(
                                existing_func_id,
                                arg_regs,
                                result_type,
                            );
                        }

                        self.builder.call_label = Some(format!("FORWARD_REF:{}", qual_name_str));
                        debug!(
                            "[FORWARD REF] Registering forward reference for unresolved call to '{}'",
                            qual_name_str
                        );

                        // Lower arguments and collect their types
                        let mut arg_regs = Vec::new();
                        let mut param_types = Vec::new();
                        for arg in unresolved_call_args {
                            if let Some(reg) = self.lower_expression(arg) {
                                arg_regs.push(reg);
                                param_types.push(self.convert_type(arg.ty));
                            }
                        }

                        // Register as a forward reference using qualified name
                        // This will be resolved later during module linking
                        let forward_func_id = self.register_stdlib_mir_forward_ref(
                            qual_name_str,
                            param_types,
                            result_type.clone(),
                        );

                        debug!(
                            "[FORWARD REF] Registered forward ref to '{}' with ID {:?}",
                            qual_name_str, forward_func_id
                        );

                        // Generate the call to the forward reference
                        return self.builder.build_call_direct(
                            forward_func_id,
                            arg_regs,
                            result_type,
                        );
                    }
                }
            }
        }
        *fell_through = true;
        None
    }
}
