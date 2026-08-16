//! Calls whose target resolves directly from the callee symbol.

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
    pub(crate) fn try_resolved_function_call(
        &mut self,
        expr: &HirExpr,
        result_type: IrType,
        converted_hir_type_args: Vec<IrType>,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_resolved_function_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if let Some(func_id) = self.resolve_function_id_with_qualified_fallback(*symbol) {
            let is_user_defined = self
                .builder
                .module
                .functions
                .get(&func_id)
                .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                .unwrap_or(false);

            // Check if receiver needs special dispatch (Dynamic unbox or Interface fat pointer)
            let receiver_needs_special_dispatch = if *is_method && !args.is_empty() {
                let receiver_type = self.resolve_through_aliases(args[0].ty);
                let type_table = self.type_table;
                type_table
                    .get(receiver_type)
                    .map(|t| {
                        matches!(
                            t.kind,
                            TypeKind::Dynamic
                                | TypeKind::Interface { .. }
                                | TypeKind::TypeParameter { .. }
                                | TypeKind::Placeholder { .. }
                                | TypeKind::Unknown
                        )
                    })
                    .unwrap_or(false)
            } else {
                false
            };

            // An imported generic instance method resolves to a cross-module
            // forward-ref stub whose FunctionKind is not UserDefined, and the
            // fallback path below does not attach the receiver's type_args, so
            // the monomorphizer could never specialize it. Generic instance
            // calls therefore take the type_args-aware block regardless of
            // kind, as long as the callee is not a genuine extern/intrinsic.
            let (callee_has_type_params, callee_is_externish) = self
                .builder
                .module
                .functions
                .get(&func_id)
                .map(|f| {
                    (
                        !f.signature.type_params.is_empty(),
                        matches!(
                            f.kind,
                            crate::ir::functions::FunctionKind::ExternC
                                | crate::ir::functions::FunctionKind::Intrinsic
                        ),
                    )
                })
                .unwrap_or((false, false));
            // A callee registered as an extern must not take the
            // receiver-type_args route: type_args would ask the monomorphizer
            // to specialize a body-less extern, leaving an unresolvable
            // `Import` symbol. Real cross-module methods are not in
            // extern_functions, so they still route correctly.
            let callee_is_externish =
                callee_is_externish || self.builder.module.extern_functions.contains_key(&func_id);
            // A cross-module callee may not be merged into this module yet, so
            // callee_has_type_params is unreliable; genericity is detected from
            // the receiver carrying non-empty type_args instead.
            let receiver_is_generic_instance = *is_method && !args.is_empty() && {
                let rt = self.resolve_through_aliases(args[0].ty);
                self.type_table
                    .get(rt)
                    .map(|t| match &t.kind {
                        TypeKind::GenericInstance { type_args, .. }
                        | TypeKind::Class { type_args, .. } => !type_args.is_empty(),
                        _ => false,
                    })
                    .unwrap_or(false)
            };
            let route_as_generic_method = *is_method
                && !callee_is_externish
                && (callee_has_type_params || receiver_is_generic_instance);

            if (is_user_defined || route_as_generic_method) && !receiver_needs_special_dispatch {
                // Instance calls get call-boundary materialization (class→iface
                // fat-ptr wrap, anon coercion) so an interface-typed param
                // never receives a raw class instance. HIR params exclude
                // `this`, so user arg `i` is HIR param `i - 1`. Static calls
                // are left untouched: they take the static-call path earlier in
                // this handler, and materializing them clashes with the stdlib
                // MIR wrappers.
                let arg_regs: Vec<_> = if *is_method {
                    args.iter()
                        .enumerate()
                        .filter_map(|(i, a)| {
                            let reg = self.lower_expression(a)?;
                            if i == 0 {
                                Some(reg)
                            } else {
                                Some(self.maybe_materialize_for_call(a, reg, Some(func_id), i - 1))
                            }
                        })
                        .collect()
                } else {
                    args.iter()
                        .filter_map(|a| self.lower_expression(a))
                        .collect()
                };

                let actual_return_type =
                    if let Some(func) = self.builder.module.functions.get(&func_id) {
                        func.signature.return_type.clone()
                    } else {
                        result_type.clone()
                    };

                // Generic class method calls carry the receiver's concrete type
                // args (Container<String>.get() → type_args=[String]) so the
                // monomorphizer can specialize. A not-yet-merged cross-module
                // callee has invisible type_params, hence the receiver signal.
                let has_type_params = self
                    .builder
                    .module
                    .functions
                    .get(&func_id)
                    .map(|f| !f.signature.type_params.is_empty())
                    .unwrap_or(false)
                    || (receiver_is_generic_instance && !callee_is_externish);

                // Gather type_args: first from HIR call-site type_args, then from receiver's
                // generic instance type_args, then from the converted HIR type_args computed
                // earlier in the Call handler.
                let call_type_args = if has_type_params {
                    if !converted_hir_type_args.is_empty() {
                        converted_hir_type_args.clone()
                    } else if *is_method && !args.is_empty() {
                        // Extract from receiver's GenericInstance / Class type_args
                        let receiver_type = self.resolve_through_aliases(args[0].ty);
                        let type_table = self.type_table;
                        type_table
                            .get(receiver_type)
                            .and_then(|t| match &t.kind {
                                TypeKind::GenericInstance { type_args, .. }
                                | TypeKind::Class { type_args, .. } => {
                                    if type_args.is_empty() {
                                        None
                                    } else {
                                        Some(
                                            type_args
                                                .iter()
                                                .map(|&ta| self.convert_type(ta))
                                                .collect::<Vec<_>>(),
                                        )
                                    }
                                }
                                _ => None,
                            })
                            .unwrap_or_default()
                    } else {
                        vec![]
                    }
                } else {
                    vec![]
                };

                if !call_type_args.is_empty() {
                    let result = self.builder.build_call_direct_with_type_args(
                        func_id,
                        arg_regs,
                        actual_return_type,
                        call_type_args,
                    );
                    // Generic bodies return type-erased i64: the caller's
                    // register has the concrete type but still holds the i64 bit
                    // pattern, so float results need a bitcast to land in the
                    // float register file.
                    if let Some(reg) = result {
                        let reg_type = self.builder.get_register_type(reg).unwrap_or(IrType::I64);
                        if matches!(reg_type, IrType::F64 | IrType::F32) {
                            return self.builder.build_bitcast(reg, reg_type);
                        }
                    }
                    return result;
                }
                return self
                    .builder
                    .build_call_direct(func_id, arg_regs, actual_return_type);
            }
        }
        *fell_through = true;
        None
    }

    pub(crate) fn try_user_class_method_call(
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
            unreachable!("try_user_class_method_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if *is_method && !args.is_empty() {
            let method_name_i = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
            // Check if receiver class has runtime mappings (skip early resolution if so)
            let receiver_has_runtime_mapping = {
                let receiver_type = self.resolve_through_aliases(args[0].ty);
                let type_table = self.type_table;
                type_table
                    .get(receiver_type)
                    .and_then(|ti| {
                        if let crate::tast::core::TypeKind::Class { symbol_id, .. } = &ti.kind {
                            self.symbol_table
                                .get_symbol(*symbol_id)
                                .map(|sym| self.is_stdlib_class_by_symbol(sym))
                        } else {
                            None
                        }
                    })
                    .unwrap_or(false)
            };
            if let Some(mn) = method_name_i {
                let resolved = if receiver_has_runtime_mapping {
                    None // Let runtime mapping handle it
                } else {
                    self.resolve_method_function_id(args[0].ty, mn)
                };
                if let Some(func_id) = resolved {
                    if func_id.0 >= 100_000 {
                        // Resolved to an import function — use it directly
                        let mut arg_regs = Vec::new();
                        for arg in args.iter() {
                            if let Some(reg) = self.lower_expression(arg) {
                                arg_regs.push(reg);
                            }
                        }
                        self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
                        return self
                            .builder
                            .build_call_direct(func_id, arg_regs, result_type);
                    }
                }
            }
        }
        *fell_through = true;
        None
    }
}
