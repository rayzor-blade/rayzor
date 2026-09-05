//! Calls written `Class.method(...)`, where the receiver names a class rather
//! than a value.

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
    pub(crate) fn try_static_receiver_call(
        &mut self,
        expr: &HirExpr,
        stdlib_info: Option<(
            &'static str,
            &'static str,
            crate::stdlib::RuntimeFunctionCall,
        )>,
        is_static_class_call: bool,
        result_type: IrType,
        func_id: IrFunctionId,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("try_static_receiver_call on a non-Call expression")
        };
        let HirExprKind::Field { object, field } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let stdlib_info = stdlib_info.as_ref().map(|(c, m, r)| (*c, *m, r));
        if is_static_class_call {
            // Static calls (Reflect.hasField, Type.typeof, Std.string) arrive as
            // Field { object: ClassVar, field: method }, and the stdlib_info
            // lookup above can miss because the class variable's TypeId isn't in
            // type_table. Re-look up the mapping by name and route through the
            // extern function.
            let static_class_name = self.find_receiver_class_name(object);
            let static_method_name = self
                .symbol_table
                .get_symbol(*field)
                .and_then(|s| self.string_interner.get(s.name))
                .map(|s| s.to_string());

            // Also try to get class name from the object symbol directly.
            // Prefer native_name or qualified_name (fully qualified) over
            // bare name so that extern classes like sys.net.Host resolve to
            // their registered key instead of just "Host".
            let static_class_name = static_class_name.or_else(|| {
                if let HirExprKind::Variable {
                    symbol: obj_sym, ..
                } = &object.kind
                {
                    let sym = self.symbol_table.get_symbol(*obj_sym)?;
                    // 1) native_name  (e.g. "sys::net::Host" → "sys.net.Host")
                    if let Some(native) = sym.native_name {
                        if let Some(ns) = self.string_interner.get(native) {
                            return Some(self.canonical_class_spelling(ns));
                        }
                    }
                    // 2) qualified_name  (e.g. "sys.net.Host")
                    if let Some(qn) = sym.qualified_name {
                        if let Some(qs) = self.string_interner.get(qn) {
                            return Some(self.canonical_class_spelling(qs));
                        }
                    }
                    // 3) bare name fallback
                    self.string_interner.get(sym.name).map(|s| s.to_string())
                } else {
                    None
                }
            });

            if let (Some(ref cls), Some(ref mn)) = (&static_class_name, &static_method_name) {
                let static_stdlib_info = self
                    .stdlib_mapping
                    .class_key(cls)
                    .and_then(|key| {
                        self.stdlib_mapping
                            .find_by_name_and_params(key, mn, args.len())
                            .or_else(|| self.stdlib_mapping.find_by_name(key, mn))
                    })
                    .map(|(sig, mapping)| (sig.class, sig.method, mapping));

                if let Some((sc_class_name, sc_method_name, runtime_call)) = static_stdlib_info {
                    let runtime_func = runtime_call.runtime_name;
                    let is_mir_wrapper = runtime_call.is_mir_wrapper;
                    let returns_raw_value = runtime_call.returns_raw_value;
                    let has_return = runtime_call.has_return;
                    let raw_value_params = runtime_call.raw_value_params;
                    let extend_to_i64_params = runtime_call.extend_to_i64_params;
                    let explicit_return_type = runtime_call.return_type.map(|rt| rt.to_ir_type());

                    // First, try special runtime calls that need custom MIR lowering
                    // (e.g., Reflect.callMethod, Reflect.makeVarArgs, Type.typeof)
                    if let Some(special_result) = self.try_lower_special_runtime_call(
                        runtime_func,
                        args,
                        result_type.clone(),
                        expr.source_location,
                    ) {
                        return special_result;
                    }

                    let (expected_param_types, actual_return_type) = self
                        .get_stdlib_mir_wrapper_signature(runtime_func)
                        .unwrap_or_else(|| {
                            let mut params = Vec::new();
                            for (i, arg) in args.iter().enumerate() {
                                let param_bit = 1u32 << i;
                                if raw_value_params & param_bit != 0 {
                                    params.push(IrType::U64);
                                } else if extend_to_i64_params & param_bit != 0 {
                                    params.push(IrType::I64);
                                } else {
                                    params.push(self.convert_type(arg.ty));
                                }
                            }
                            let ret_type = if let Some(ref ert) = explicit_return_type {
                                ert.clone()
                            } else if returns_raw_value {
                                IrType::U64
                            } else if has_return {
                                result_type.clone()
                            } else {
                                IrType::Void
                            };
                            (params, ret_type)
                        });

                    // Lower arguments (no 'this' for static methods)
                    let mut arg_regs = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let arg_reg = self.lower_expression(arg)?;
                        let actual_ty = self.convert_type(arg.ty);
                        let expected_ty = expected_param_types
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| actual_ty.clone());
                        let final_reg =
                            self.maybe_box_for_extern_call(arg_reg, &actual_ty, &expected_ty)?;
                        arg_regs.push(final_reg);
                    }

                    if is_mir_wrapper {
                        let param_types: Vec<_> = arg_regs
                            .iter()
                            .map(|r| self.builder.get_register_type(*r).unwrap_or(IrType::I64))
                            .collect();
                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                            runtime_func,
                            param_types,
                            actual_return_type.clone(),
                        );
                        return self.builder.build_call_direct(
                            mir_func_id,
                            arg_regs,
                            actual_return_type,
                        );
                    }

                    // Inject hidden enum type_id arg for runtime enum helpers
                    // (enumEq, enumConstructor, enumParameters, getEnum)
                    self.inject_hidden_enum_type_id_arg(runtime_func, args, &mut arg_regs);

                    let param_types: Vec<_> = arg_regs
                        .iter()
                        .map(|r| self.builder.get_register_type(*r).unwrap_or(IrType::I64))
                        .collect();
                    let extern_func_id = self.get_or_register_extern_function(
                        runtime_func,
                        param_types,
                        actual_return_type.clone(),
                    );

                    let call_result = self.builder.build_call_direct(
                        extern_func_id,
                        arg_regs,
                        actual_return_type.clone(),
                    );

                    // Tag a static factory result with its class so a
                    // cross-module `var b = Bytes.ofString(s)` keeps a
                    // class handle when the local's own type stays
                    // unresolved. Applies when the factory returns the
                    // same reference class (a `PtrVoid`/`Ptr` handle).
                    if let Some(reg) = call_result {
                        if matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_))) {
                            if let Some(hint) =
                                self.static_factory_return_class(sc_class_name, sc_method_name)
                            {
                                self.register_class_hints.insert(reg, hint);
                            }
                        }
                    }

                    if returns_raw_value {
                        if let Some(raw_reg) = call_result {
                            return match &result_type {
                                IrType::I32 => {
                                    self.builder.build_cast(raw_reg, IrType::U64, IrType::I32)
                                }
                                IrType::Bool => {
                                    self.builder.build_cast(raw_reg, IrType::U64, IrType::Bool)
                                }
                                // Map<K,Float>.get stores f64 bits as u64; the
                                // bitcast reverses the set-side one so reads
                                // return the original f64.
                                IrType::F64 => self.builder.build_bitcast(raw_reg, IrType::F64),
                                IrType::F32 => {
                                    let f64v = self.builder.build_bitcast(raw_reg, IrType::F64)?;
                                    self.builder.build_cast(f64v, IrType::F64, IrType::F32)
                                }
                                _ => Some(raw_reg),
                            };
                        }
                    }

                    return call_result;
                }
            }

            // Static call: do NOT include the class reference as 'this'
            let callee_is_user_defined = self
                .builder
                .module
                .functions
                .get(&func_id)
                .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                .unwrap_or(false);

            let mut arg_regs = Vec::new();
            for (param_idx, arg) in args.iter().enumerate() {
                if let Some(reg) = self.lower_expression(arg) {
                    // Materialize anon-backed variables at call boundary
                    let reg = self.maybe_materialize_for_call(arg, reg, Some(func_id), param_idx);
                    // @:derive(Copy): copy variable args at call boundary
                    let reg = if let HirExprKind::Variable { .. } = &arg.kind {
                        if let Some(class_sym) = self.get_copy_class_symbol(arg.ty) {
                            self.emit_shallow_copy(reg, class_sym).unwrap_or(reg)
                        } else {
                            reg
                        }
                    } else {
                        reg
                    };
                    if callee_is_user_defined {
                        let is_heap_intermediate = matches!(
                            &arg.kind,
                            HirExprKind::New { .. } | HirExprKind::Call { .. }
                        ) && self.get_drop_behavior(arg.ty)
                            == DropBehavior::AutoDrop
                            && !self.interface_wrapped_args.contains(&reg);
                        if is_heap_intermediate {
                            self.temp_heap_values.push(reg);
                        }
                    }
                    arg_regs.push(reg);
                }
            }

            let call_arg_types: Vec<TypeId> = args.iter().map(|a| a.ty).collect();
            self.bind_skipped_optional_args(func_id, &mut arg_regs, &call_arg_types, false);
            self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
            let hir_types: Vec<Option<TypeId>> = call_arg_types.iter().map(|t| Some(*t)).collect();
            self.unbox_optional_args_for_erased_formals(func_id, &mut arg_regs, &hir_types, false);
            self.fill_default_args(func_id, &mut arg_regs, false);

            let actual_return_type = if let Some(func) = self.builder.module.functions.get(&func_id)
            {
                func.signature.return_type.clone()
            } else {
                result_type.clone()
            };

            let result = self
                .builder
                .build_call_direct(func_id, arg_regs, actual_return_type);
            // Set class hint on result for cross-module method dispatch
            if let Some(reg) = result {
                self.set_class_hint_for_return(reg, expr.ty);
            }
            return result;
        }
        *fell_through = true;
        None
    }
}
