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

            // A generic instance method imported from another module
            // resolves here to a cross-module forward-ref stub whose
            // FunctionKind is not UserDefined (the real impl arrives via
            // merge + fixup later). The is_user_defined gate would route
            // it to the fallback path below, which does NOT attach the
            // receiver's concrete type_args — so the monomorphizer can
            // never specialize the imported generic method, and its whole
            // call chain (e.g. an imported haxe.ds.BalancedTree.set ->
            // setLoop -> balance -> compare) reaches codegen as generic
            // trap stubs and SIGILLs. Route generic instance-method calls
            // through the type_args-aware block regardless of kind, as
            // long as the callee is not a genuine extern/intrinsic.
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
            // A callee registered as an extern (e.g. the iterator-protocol
            // methods List.iterator / .keys, which lower to extern Imports
            // resolved at link time) must NOT take the receiver-type_args
            // route: attaching type_args would ask the monomorphizer to
            // specialize an extern that has no body, producing an
            // unresolvable `Import` symbol that makes finalize panic on the
            // whole module. Real cross-module methods (BalancedTree.set) are
            // not in extern_functions, so they still route correctly.
            let callee_is_externish =
                callee_is_externish || self.builder.module.extern_functions.contains_key(&func_id);
            // The callee may be a cross-module function not yet present
            // in this module (resolved to its eventual id but merged
            // later), so callee_has_type_params is unreliable here. Detect
            // genericity from the RECEIVER instead: a concrete generic
            // class instance (Class/GenericInstance carrying type_args).
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
                // Lower args and, for instance method
                // calls, apply call-boundary materialization
                // (class→iface fat-ptr wrap, anon coercion).
                // HIR params don't include `this`, so the
                // user arg at index `i` corresponds to HIR
                // param index `i - 1` when `is_method=true`.
                // Without this wrap, passing a raw class
                // instance to a method whose param is
                // interface-typed stores a non-fat-ptr in
                // any iface field the callee assigns to →
                // later virtual dispatch on that field
                // SIGSEGVs (e.g. `reg.register("llama",
                // new LlamaArch())` in nue.arch).
                // Static calls are left untouched: they
                // already go through the static-call path
                // earlier in this handler when receiver is
                // missing, and routing them through
                // `maybe_materialize_for_call` here has
                // triggered Cranelift symbol-clash on
                // stdlib MIR wrappers like `array_length`.
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

                // For generic class method calls, extract concrete type args
                // from the receiver's type (e.g., Container<String>.get() → type_args=[String]).
                // This enables the monomorphizer to specialize the function.
                //
                // The callee may be a cross-module function not yet merged
                // into this module, so its type_params are invisible here
                // (has_type_params is false). Fall back to the receiver
                // signal: a concrete generic-class instance receiver means
                // we still want to attach the receiver's type_args so the
                // monomorphizer can specialize the imported generic method.
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
                    // The function body uses type-erased I64 for all type param values.
                    // When the resolved concrete type differs (F64, I32, String, etc.),
                    // the caller's register has the right TYPE but the value is still
                    // the I64 bit pattern. After inlining + SRA, this becomes visible.
                    // Insert a bitcast for float types where i64→f64 reinterpretation
                    // is needed at the calling convention level.
                    if let Some(reg) = result {
                        let reg_type = self.builder.get_register_type(reg).unwrap_or(IrType::I64);
                        if matches!(reg_type, IrType::F64 | IrType::F32) {
                            // Bitcast from i64 to f64 to ensure calling convention
                            // uses the float register file
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
                                .and_then(|sym| self.string_interner.get(sym.name))
                                .map(|name| self.stdlib_mapping.class_has_any_method(name))
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
