//! Static methods on stdlib classes.

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
    pub(crate) fn try_stdlib_static_call(
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
            unreachable!("try_stdlib_static_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if !*is_method || self.effective_static_call_args(args).len() != args.len() {
            if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                if let Some(method_name) = self.string_interner.get(sym_info.name) {
                    let static_args = self.effective_static_call_args(args);
                    debug!(
                        "[STATIC-PATH] method_name='{}', symbol={:?}, has_qualified_name={}",
                        method_name,
                        symbol,
                        sym_info.qualified_name.is_some()
                    );

                    if let Some(qual_name) = sym_info.qualified_name {
                        if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                            debug!("[PRE-CHECK] Qualified name: '{}'", qual_name_str);

                            // Thread/Channel/Mutex/Arc methods are stdlib MIR
                            // wrappers rather than runtime_mapping entries:
                            // "rayzor.concurrent.Thread.spawn" -> "Thread_spawn".
                            let parts: Vec<&str> = qual_name_str.split('.').collect();
                            if parts.len() >= 2 {
                                let class_name = parts[parts.len() - 2];

                                // Only rayzor.concurrent.*; sys.thread.Thread uses
                                // the runtime mapping directly.
                                let is_rayzor_concurrent =
                                    qual_name_str.starts_with("rayzor.concurrent.");
                                if is_rayzor_concurrent
                                    && self.stdlib_mapping.class_key(class_name).is_some_and(|k| {
                                        self.stdlib_mapping.is_mir_wrapper_class(k)
                                    })
                                {
                                    // Use capitalized class names for rayzor.concurrent (Thread, Channel, etc.)
                                    let mir_func_name = format!("{}_{}", class_name, method_name);
                                    debug!(
                                        "[STDLIB MIR] Detected stdlib MIR function: {}, args.len()={}",
                                        mir_func_name,
                                        static_args.len()
                                    );
                                    for (idx, arg) in static_args.iter().enumerate() {
                                        debug!(
                                            "[STDLIB MIR PRE] arg[{}] kind={:?}, ty={:?}",
                                            idx,
                                            std::mem::discriminant(&arg.kind),
                                            arg.ty
                                        );
                                    }

                                    // A static call may carry a synthetic class
                                    // receiver argument; the mapping signature's
                                    // arity decides whether to trim it.
                                    let mut actual_args = static_args;
                                    if let Some((expected_params, _)) =
                                        self.get_stdlib_mir_wrapper_signature(&mir_func_name)
                                    {
                                        if actual_args.len() != expected_params.len()
                                            && static_args.len() == expected_params.len() + 1
                                            && !static_args.is_empty()
                                        {
                                            debug!(
                                                "[STDLIB MIR FIX] Arity-based static receiver trim for {}: {} -> {} args",
                                                mir_func_name,
                                                static_args.len(),
                                                expected_params.len()
                                            );
                                            actual_args = &static_args[1..];
                                        }
                                    }

                                    let mut arg_regs = Vec::new();
                                    let mut param_types = Vec::new();
                                    for (idx, arg) in actual_args.iter().enumerate() {
                                        debug!("[STDLIB MIR] arg[{}] ty={:?}", idx, arg.ty);
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                            param_types.push(self.convert_type(arg.ty));
                                        }
                                    }
                                    // The body comes from the merged stdlib module,
                                    // so the signature is inferred from the call site.
                                    let mir_func_id = self.register_stdlib_mir_forward_ref(
                                        &mir_func_name,
                                        param_types,
                                        result_type.clone(),
                                    );

                                    debug!(
                                        "[STDLIB MIR] Registered forward ref to {} with ID {:?}",
                                        mir_func_name, mir_func_id
                                    );

                                    let result = self.builder.build_call_direct(
                                        mir_func_id,
                                        arg_regs,
                                        result_type,
                                    );
                                    debug!("[STDLIB MIR] Generated call, result: {:?}", result);
                                    return result;
                                }
                            }

                            // Stdlib class methods are keyed by qualified name, e.g.
                            // "rayzor.concurrent.Thread.spawn" or "test.Thread.spawn".
                            let lookup_result = self.get_static_stdlib_runtime_func_with_params(
                                qual_name_str,
                                method_name,
                                static_args.len(),
                            );
                            debug!(
                                "[PRE-CHECK] get_static_stdlib_runtime_func returned: {:?}",
                                lookup_result
                            );

                            if let Some(runtime_func) = lookup_result {
                                debug!(
                                    "[STATIC METHOD] Found stdlib runtime func: {}.{} -> {}, args.len()={}",
                                    qual_name_str,
                                    method_name,
                                    runtime_func,
                                    static_args.len()
                                );

                                if let Some(result) = self.try_lower_special_runtime_call(
                                    &runtime_func,
                                    static_args,
                                    result_type.clone(),
                                    expr.source_location.clone(),
                                ) {
                                    return result;
                                }

                                // The registered extern signature is authoritative
                                // for the types (e.g. I64 for Std.random).
                                let (expected_param_types, expected_return_type) = self
                                    .get_extern_function_signature(&runtime_func)
                                    .unwrap_or_else(|| {
                                        // Fall back to inferred types from TAST
                                        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                                        let param_types: Vec<IrType> = static_args
                                            .iter()
                                            .map(|a| {
                                                let arg_ty = self.convert_type(a.ty);
                                                if arg_ty == IrType::String {
                                                    string_ptr_ty.clone()
                                                } else {
                                                    arg_ty
                                                }
                                            })
                                            .collect();
                                        (param_types, result_type.clone())
                                    });

                                let runtime_call_args = if static_args.len()
                                    == expected_param_types.len() + 1
                                    && !static_args.is_empty()
                                {
                                    &static_args[1..]
                                } else {
                                    static_args
                                };

                                let arg_regs: Vec<_> = runtime_call_args
                                    .iter()
                                    .filter_map(|a| self.lower_expression(a))
                                    .collect();

                                debug!(
                                    "[STATIC METHOD] Lowered {} args: {:?}",
                                    arg_regs.len(),
                                    arg_regs
                                );

                                let final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                    .map(|(i, &reg)| {
                                        if let (Some(expected_ty), Some(actual_ty)) = (
                                            expected_param_types.get(i),
                                            self.builder.get_register_type(reg)
                                        ) {
                                            if *expected_ty != actual_ty {
                                                // Ptr(U8) means the param is Dynamic, so the value is boxed rather than cast.
                                                let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                if is_ptr_u8 && i < runtime_call_args.len() {
                                                    debug!("[STATIC METHOD BOX] Attempting auto-box for arg {} with type {:?}", i, runtime_call_args[i].ty);
                                                    if let Some(boxed) = self.box_value_for_dynamic(reg, runtime_call_args[i].ty) {
                                                        debug!("[STATIC METHOD BOX] Auto-boxed arg {} for Dynamic param", i);
                                                        return boxed;
                                                    }
                                                    debug!("[STATIC METHOD BOX] box_value_for_dynamic returned None for arg {}", i);
                                                }
                                                debug!("[STATIC METHOD] Casting arg {} from {:?} to {:?}", i, actual_ty, expected_ty);
                                                if let Some(casted) = self.builder.build_cast(reg, actual_ty.clone(), expected_ty.clone()) {
                                                    return casted;
                                                }
                                            }
                                        }
                                        reg
                                    })
                                    .collect();

                                // Inject hidden enum type_id for enum helper runtime calls.
                                let mut final_arg_regs = final_arg_regs;
                                self.inject_hidden_enum_type_id_arg(
                                    &runtime_func,
                                    runtime_call_args,
                                    &mut final_arg_regs,
                                );

                                let runtime_func_id = self.get_or_register_extern_function(
                                    &runtime_func,
                                    expected_param_types,
                                    expected_return_type.clone(),
                                );

                                debug!(
                                    "[STATIC METHOD] Registered runtime func {} with ID {:?}",
                                    runtime_func, runtime_func_id
                                );

                                let result = self.builder.build_call_direct(
                                    runtime_func_id,
                                    final_arg_regs,
                                    expected_return_type,
                                );
                                debug!("[STATIC METHOD] Generated call, result: {:?}", result);
                                return result;
                            }
                        }
                    }

                    // With no qualified_name (e.g. Reflect.compare from import
                    // files), find the method by scanning known classes; static
                    // methods only, to avoid false positives. A qualified name is
                    // tried class-first, before the global static-name scan.
                    let mut static_fallback = None;
                    if let Some(qual_name_str) = sym_info
                        .qualified_name
                        .and_then(|q| self.string_interner.get(q))
                    {
                        let parts: Vec<&str> = qual_name_str.split('.').collect();
                        if parts.len() >= 2 {
                            let mut class_candidates: Vec<String> = Vec::new();
                            // Fully-qualified class form used in runtime mapping
                            class_candidates.push(parts[..parts.len() - 1].join("."));
                            // Simple class name fallback
                            class_candidates.push(parts[parts.len() - 2].to_string());

                            for class_name in class_candidates {
                                let Some(class_key) = self.stdlib_mapping.class_key(&class_name)
                                else {
                                    continue;
                                };
                                if let Some(found) = self.stdlib_mapping.find_by_name_and_params(
                                    class_key,
                                    method_name,
                                    static_args.len(),
                                ) {
                                    static_fallback = Some(found);
                                    break;
                                }
                            }
                        }
                    }

                    if static_fallback.is_none() {
                        debug!(
                            "[STATIC-FALLBACK] Trying global find_static_method_by_name_and_params('{}', {})...",
                            method_name,
                            static_args.len()
                        );
                        static_fallback = self
                            .stdlib_mapping
                            .find_static_method_by_name_and_params(method_name, static_args.len());
                    }

                    if let Some((_sig, mapping)) = static_fallback {
                        let runtime_func_name = mapping.runtime_name.to_string();
                        debug!(
                            "[STATIC FALLBACK] Found static {}.{} -> {} via name scan",
                            _sig.class, method_name, runtime_func_name
                        );

                        if let Some(result) = self.try_lower_special_runtime_call(
                            &runtime_func_name,
                            static_args,
                            result_type.clone(),
                            expr.source_location.clone(),
                        ) {
                            return result;
                        }

                        let mut arg_regs: Vec<IrId> = Vec::new();
                        let mut arg_types: Vec<IrType> = Vec::new();
                        for arg in static_args.iter() {
                            if let Some(reg) = self.lower_expression(arg) {
                                arg_regs.push(reg);
                                arg_types.push(self.convert_type(arg.ty));
                            }
                        }

                        // Reflect.compare → haxe_reflect_compare_typed: detect the
                        // argument type and append a type_tag param to avoid boxing.
                        if runtime_func_name == "haxe_reflect_compare" {
                            let mut known_type_tag: Option<i32> = None;
                            let mut type_param_name: Option<String> = None;
                            let mut use_typed = false;

                            if let Some(first_arg) = static_args.first() {
                                let type_table = self.type_table;
                                if let Some(ti) = type_table.get(first_arg.ty) {
                                    use crate::tast::core::TypeKind;
                                    match &ti.kind {
                                        TypeKind::TypeParameter { symbol_id, .. } => {
                                            use_typed = true;
                                            if let Some(sym) =
                                                self.symbol_table.get_symbol(*symbol_id)
                                            {
                                                if let Some(name_str) =
                                                    self.string_interner.get(sym.name)
                                                {
                                                    type_param_name = Some(name_str.to_string());
                                                }
                                            }
                                        }
                                        TypeKind::Int => {
                                            use_typed = true;
                                            known_type_tag = Some(1);
                                        }
                                        TypeKind::Float => {
                                            use_typed = true;
                                            known_type_tag = Some(4);
                                        }
                                        TypeKind::Bool => {
                                            use_typed = true;
                                            known_type_tag = Some(2);
                                        }
                                        TypeKind::String => {
                                            use_typed = true;
                                            known_type_tag = Some(5);
                                        }
                                        _ => {}
                                    }
                                }
                            }

                            if use_typed {
                                // Cast value args to I64 — haxe_reflect_compare_typed
                                // takes type-erased i64 values, not typed structs
                                for i in 0..arg_regs.len().min(2) {
                                    let reg_ty = self
                                        .builder
                                        .get_register_type(arg_regs[i])
                                        .unwrap_or(IrType::I64);
                                    if reg_ty != IrType::I64 {
                                        if let Some(cast) = self.builder.build_cast(
                                            arg_regs[i],
                                            reg_ty,
                                            IrType::I64,
                                        ) {
                                            arg_regs[i] = cast;
                                        }
                                    }
                                    arg_types[i] = IrType::I64;
                                }

                                let tag_reg = if let Some(tp_name) = type_param_name {
                                    let tag = self.builder.build_const(IrValue::I32(0))?;
                                    if let Some(func) = self.builder.current_function_mut() {
                                        func.type_param_tag_fixups.push((tag, tp_name));
                                    }
                                    tag
                                } else {
                                    self.builder
                                        .build_const(IrValue::I32(known_type_tag.unwrap_or(1)))?
                                };
                                arg_regs.push(tag_reg);
                                arg_types.push(IrType::I32);

                                let extern_func_id = self.get_or_register_extern_function(
                                    "haxe_reflect_compare_typed",
                                    arg_types,
                                    result_type.clone(),
                                );
                                return self.builder.build_call_direct(
                                    extern_func_id,
                                    arg_regs,
                                    result_type,
                                );
                            }
                        }

                        // General case: call the runtime function directly
                        let (expected_param_types, expected_return_type) = self
                            .get_extern_function_signature(&runtime_func_name)
                            .unwrap_or_else(|| (arg_types, result_type.clone()));

                        let final_arg_regs: Vec<_> = arg_regs
                            .iter()
                            .enumerate()
                            .map(|(i, &reg)| {
                                if let (Some(expected_ty), Some(actual_ty)) = (
                                    expected_param_types.get(i),
                                    self.builder.get_register_type(reg),
                                ) {
                                    if *expected_ty != actual_ty {
                                        if let Some(casted) = self.builder.build_cast(
                                            reg,
                                            actual_ty.clone(),
                                            expected_ty.clone(),
                                        ) {
                                            return casted;
                                        }
                                    }
                                }
                                reg
                            })
                            .collect();

                        // Inject hidden enum type_id for enum helper runtime calls.
                        let mut final_arg_regs = final_arg_regs;
                        self.inject_hidden_enum_type_id_arg(
                            &runtime_func_name,
                            args,
                            &mut final_arg_regs,
                        );

                        let runtime_func_id = self.get_or_register_extern_function(
                            &runtime_func_name,
                            expected_param_types,
                            expected_return_type.clone(),
                        );

                        return self.builder.build_call_direct(
                            runtime_func_id,
                            final_arg_regs,
                            expected_return_type,
                        );
                    }
                }
            }
        }
        *fell_through = true;
        None
    }
}
