//! Emitting a method call whose target function is already resolved:
//! runtime mapping, dispatch choice, argument coercion.

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
    pub(crate) fn try_resolved_method_call(
        &mut self,
        expr: &HirExpr,
        maybe_func_id: Option<IrFunctionId>,
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
            unreachable!("try_resolved_method_call on a non-Call expression")
        };
        let HirExprKind::Field { object, field } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let method_name_interned = self.symbol_table.get_symbol(*field).map(|s| s.name);
        let method_name = method_name_interned.and_then(|name| self.string_interner.get(name));
        if let Some(func_id) = maybe_func_id {
            // Route through the runtime mapping for extern class methods.
            // get_stdlib_runtime_info's internal guard returns None for
            // user-defined class receivers, preventing name collisions.
            let stdlib_info = {
                let method_name_str = self
                    .symbol_table
                    .get_symbol(*field)
                    .and_then(|s| self.string_interner.get(s.name));

                if let Some(mn) = method_name_str {
                    if mn == "indexOf" || mn == "lastIndexOf" || mn == "substr" {
                        // Overloaded String methods register each arity as a separate
                        // mapping, so a name-only lookup matches the wrong one.
                        let arg_count = args.len();
                        debug!(
                            "[String overload lookup] method={}, arg_count={}",
                            mn, arg_count
                        );
                        self.stdlib_mapping
                            .find_by_name_and_params("String", mn, arg_count)
                            .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                    } else if mn == "wait" {
                        // Lock.wait() has overloads: 0 params (blocking) vs 1 param (with timeout)
                        let arg_count = args.len();
                        debug!("[wait lookup] method={}, arg_count={}", mn, arg_count);
                        self.stdlib_mapping
                            .find_by_name_and_params("sys_thread_Lock", mn, arg_count)
                            .or_else(|| {
                                self.stdlib_mapping.find_by_name_and_params(
                                    "sys_thread_Condition",
                                    mn,
                                    arg_count,
                                )
                            })
                            .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                            .or_else(|| {
                                self.get_stdlib_runtime_info(
                                    *field,
                                    object.ty,
                                    Some(arg_count),
                                    None,
                                )
                            })
                    } else if mn == "tryAcquire" {
                        // Semaphore.tryAcquire() has overloads: 0 params vs 1 param (with timeout)
                        let arg_count = args.len();
                        debug!("[tryAcquire lookup] method={}, arg_count={}", mn, arg_count);
                        self.stdlib_mapping
                            .find_by_name_and_params("sys_thread_Semaphore", mn, arg_count)
                            .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                            .or_else(|| {
                                self.get_stdlib_runtime_info(
                                    *field,
                                    object.ty,
                                    Some(arg_count),
                                    None,
                                )
                            })
                    } else {
                        let receiver_hint: Option<String> = self.find_receiver_class_name(object);
                        let hint_ref = receiver_hint.as_deref();
                        self.get_stdlib_runtime_info(*field, object.ty, Some(args.len()), hint_ref)
                    }
                } else {
                    self.get_stdlib_runtime_info(*field, object.ty, Some(args.len()), None)
                }
            }
            .map(|(c, m, r)| (c, m, r.clone()));

            probe!(self.try_stdlib_mapped_method_call(
                expr,
                stdlib_info.clone(),
                result_type.clone()
            ));

            // Extern classes not in type_table (e.g. rayzor.Bytes): recover the
            // class name from the MIR function's qualified_name.
            debug!(
                "[FALLBACK check] func_id={:?}, in module={}",
                func_id,
                self.builder.module.functions.contains_key(&func_id)
            );
            if let Some(func) = self.builder.module.functions.get(&func_id) {
                debug!(
                    "[FALLBACK] MIR function '{}' has qualified_name: {:?}",
                    func.name, func.qualified_name
                );
                if let Some(ref qn) = func.qualified_name {
                    // Pattern: "rayzor.Bytes.set" -> class="rayzor_Bytes", method="set"
                    let parts: Vec<&str> = qn.split('.').collect();
                    if parts.len() >= 2 {
                        let mir_method_name = *parts.last().unwrap();
                        let class_parts = &parts[..parts.len() - 1];
                        let qualified_class = class_parts.join("_");

                        if let Some((_sig, mapping)) = self
                            .stdlib_mapping
                            .find_by_name(&qualified_class, mir_method_name)
                        {
                            let runtime_func = mapping.runtime_name;
                            debug!(
                                "[Extern method redirect via qualified_name] {}.{} -> {}",
                                qualified_class, mir_method_name, runtime_func
                            );

                            let (expected_param_types, actual_return_type) = self
                                .get_stdlib_mir_wrapper_signature(runtime_func)
                                .map(|(params, ret)| (params, ret))
                                .unwrap_or_else(|| {
                                    let mut params = vec![IrType::Ptr(Box::new(IrType::U8))];
                                    for arg in args {
                                        params.push(self.convert_type(arg.ty));
                                    }
                                    let ret_type = if let Some(ref rt) = mapping.return_type {
                                        rt.to_ir_type()
                                    } else if mapping.has_return {
                                        result_type.clone()
                                    } else {
                                        IrType::Void
                                    };
                                    (params, ret_type)
                                });

                            let obj_reg = self.lower_expression(object)?;

                            let mut arg_regs = vec![obj_reg];
                            for (i, arg) in args.iter().enumerate() {
                                let arg_reg = self.lower_expression(arg)?;
                                let actual_ty = self.convert_type(arg.ty);
                                let expected_ty = expected_param_types
                                    .get(i + 1)
                                    .cloned()
                                    .unwrap_or_else(|| actual_ty.clone());
                                let final_reg = self.maybe_box_for_extern_call(
                                    arg_reg,
                                    &actual_ty,
                                    &expected_ty,
                                )?;
                                arg_regs.push(final_reg);
                            }

                            let param_types = if expected_param_types.len() == arg_regs.len() {
                                expected_param_types.clone()
                            } else {
                                let mut params = vec![IrType::Ptr(Box::new(IrType::U8))];
                                for arg in args {
                                    params.push(self.convert_type(arg.ty));
                                }
                                params
                            };

                            let extern_func_id = self.get_or_register_extern_function(
                                runtime_func,
                                param_types,
                                actual_return_type.clone(),
                            );

                            let call_result = self.builder.build_call_direct(
                                extern_func_id,
                                arg_regs,
                                actual_return_type.clone(),
                            )?;

                            // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                            return self.maybe_unbox_for_extern_return(
                                call_result,
                                &actual_return_type,
                                &result_type,
                            );
                        }
                    }
                }
            }

            // Methods overridden in a class hierarchy dispatch through the vtable;
            // super.method() must bypass it and call the parent directly.
            let object_is_super = matches!(object.kind, HirExprKind::Super);
            let vtable_lookup = if object_is_super {
                None
            } else {
                self.virtual_dispatch_info.get(field).copied().or_else(|| {
                    let method_name = self.symbol_table.get_symbol(*field).map(|s| s.name);
                    if let Some(method_name) = method_name {
                        let receiver_class_sym = {
                            let type_table = self.type_table;
                            type_table.get(object.ty).and_then(|t| match &t.kind {
                                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                _ => None,
                            })
                        };
                        if let Some(class_sym) = receiver_class_sym {
                            if let Some(&method_sym) =
                                self.class_method_by_name.get(&(class_sym, method_name))
                            {
                                return self.virtual_dispatch_info.get(&method_sym).copied();
                            }
                            let mut current = class_sym;
                            while let Some(&parent) = self.class_parent_map.get(&current) {
                                if let Some(&method_sym) =
                                    self.class_method_by_name.get(&(parent, method_name))
                                {
                                    return self.virtual_dispatch_info.get(&method_sym).copied();
                                }
                                current = parent;
                            }
                        }
                    }
                    None
                })
            };
            if let Some((slot_index, _)) = vtable_lookup {
                let obj_reg = self.lower_expression(object)?;

                // If Dynamic-typed, unbox to get raw object pointer
                let obj_reg = {
                    let is_dynamic = {
                        let type_table = self.type_table;
                        type_table
                            .get(object.ty)
                            .map(|t| matches!(t.kind, TypeKind::Dynamic))
                            .unwrap_or(false)
                    };
                    if is_dynamic {
                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                        let unbox_func_id = self.get_or_register_extern_function(
                            "haxe_unbox_reference_ptr",
                            vec![ptr_u8.clone()],
                            ptr_u8.clone(),
                        );
                        self.builder
                            .build_call_direct(unbox_func_id, vec![obj_reg], ptr_u8)?
                    } else {
                        obj_reg
                    }
                };

                let mut call_args = vec![obj_reg];
                for arg in args.iter() {
                    if let Some(reg) = self.lower_expression(arg) {
                        call_args.push(reg);
                    }
                }

                // haxe_vtable_lookup(obj, slot) -> closure_ptr (i64)
                let lookup_fn = self.get_or_register_extern_function(
                    "haxe_vtable_lookup",
                    vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                    IrType::I64,
                );
                let slot_reg = self.builder.build_const(IrValue::I32(slot_index as i32))?;
                let closure_ptr = self.builder.build_call_direct(
                    lookup_fn,
                    vec![obj_reg, slot_reg],
                    IrType::I64,
                )?;

                let mut param_types = vec![IrType::Ptr(Box::new(IrType::Void))]; // self
                for arg in args {
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
                    .build_call_indirect(closure_ptr, call_args, func_signature);
            }

            // super.method() — resolve to parent class method directly
            if object_is_super {
                if let Some(method_name_i) = method_name_interned {
                    let current_class = self.builder.current_function().and_then(|f| {
                        self.class_method_by_name
                            .iter()
                            .find(|(_, &method_sym)| {
                                self.function_map.get(&method_sym) == Some(&f.id)
                            })
                            .map(|((class_sym, _), _)| *class_sym)
                    });
                    let parent_class =
                        current_class.and_then(|cls| self.class_parent_map.get(&cls).copied());
                    let parent_method_func_id = parent_class
                        .and_then(|pc| {
                            self.class_method_by_name
                                .get(&(pc, method_name_i))
                                .and_then(|&sym| {
                                    self.resolve_function_id_with_qualified_fallback(sym)
                                })
                        })
                        .or_else(|| self.resolve_function_id_with_qualified_fallback(*field));
                    if let Some(func_id) = parent_method_func_id {
                        let obj_reg = self.lower_expression(object)?;
                        let mut call_args = vec![obj_reg];
                        for arg in args {
                            if let Some(reg) = self.lower_expression(arg) {
                                call_args.push(reg);
                            }
                        }
                        let ret = self.convert_type(expr.ty);
                        return self.builder.build_call_direct(func_id, call_args, ret);
                    }
                }
            }

            // A class/abstract symbol as receiver means a static call: the object
            // must not be passed as 'this'.
            let is_static_class_call = if let HirExprKind::Variable {
                symbol: obj_sym, ..
            } = &object.kind
            {
                let kind = self.symbol_table.get_symbol(*obj_sym).map(|s| s.kind);
                kind.map(|k| {
                    matches!(
                        k,
                        crate::tast::symbols::SymbolKind::Class
                            | crate::tast::symbols::SymbolKind::Abstract
                            | crate::tast::symbols::SymbolKind::TypeAlias
                    )
                })
                .unwrap_or(false)
            } else {
                false
            };

            // @:derive(Default) synthetic static createDefault() — zero-initialized instance
            if is_static_class_call && method_name == Some("createDefault") && args.is_empty() {
                if let HirExprKind::Variable {
                    symbol: obj_sym, ..
                } = &object.kind
                {
                    // For static calls obj_sym is the class symbol itself.
                    let class_sym = self
                        .symbol_table
                        .get_symbol(*obj_sym)
                        .and_then(|s| {
                            let type_table = self.type_table;
                            type_table.get(s.type_id).and_then(|t| {
                                if let TypeKind::Class { symbol_id, .. } = &t.kind {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            })
                        })
                        .or(Some(*obj_sym))
                        .filter(|sym| self.derive_default_classes.contains(sym));
                    if let Some(sym) = class_sym {
                        return self.lower_derived_default(sym);
                    }
                }
            }

            probe!(self.try_static_receiver_call(
                expr,
                stdlib_info.clone(),
                is_static_class_call,
                result_type.clone(),
                func_id
            ));

            let obj_reg = self.lower_expression(object)?;

            // Dynamic variables hold a boxed DynamicValue*, but the method expects
            // a raw class pointer as 'this'.
            let obj_reg = {
                let is_dynamic = {
                    let type_table = self.type_table;
                    type_table
                        .get(object.ty)
                        .map(|t| matches!(t.kind, TypeKind::Dynamic))
                        .unwrap_or(false)
                };
                if is_dynamic {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let unbox_func_id = self.get_or_register_extern_function(
                        "haxe_unbox_reference_ptr",
                        vec![ptr_u8.clone()],
                        ptr_u8.clone(),
                    );
                    self.builder
                        .build_call_direct(unbox_func_id, vec![obj_reg], ptr_u8)?
                } else {
                    obj_reg
                }
            };

            // Track NEW expressions as temps (not Call results — they may be references)
            let is_new_temp = matches!(&object.kind, HirExprKind::New { .. })
                && self.get_drop_behavior(object.ty) == DropBehavior::AutoDrop;
            if is_new_temp {
                self.temp_heap_values.push(obj_reg);
            }

            // Lower the arguments — track heap intermediates only for user-defined callees
            let callee_is_user_defined = self
                .builder
                .module
                .functions
                .get(&func_id)
                .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                .unwrap_or(false);

            let mut method_arg_regs = vec![obj_reg]; // 'this' as first arg
            for arg in args.iter() {
                if let Some(reg) = self.lower_expression(arg) {
                    if callee_is_user_defined {
                        let is_heap_intermediate = matches!(
                            &arg.kind,
                            HirExprKind::New { .. } | HirExprKind::Call { .. }
                        ) && self.get_drop_behavior(arg.ty)
                            == DropBehavior::AutoDrop;
                        if is_heap_intermediate {
                            self.temp_heap_values.push(reg);
                        }
                    }
                    method_arg_regs.push(reg);
                }
            }
            let arg_regs = method_arg_regs;

            // Use the function's actual return type: expr.ty can be an unresolved
            // TypeParameter.
            let actual_return_type = if let Some(func) = self.builder.module.functions.get(&func_id)
            {
                debug!(
                    "[Field method] Using actual return type {:?} for function {:?}",
                    func.signature.return_type, func.name
                );
                func.signature.return_type.clone()
            } else {
                debug!(
                    "[Field method] Function not found in module, using expr return type {:?}",
                    result_type
                );
                result_type.clone()
            };

            let call_result =
                self.builder
                    .build_call_direct(func_id, arg_regs, actual_return_type.clone())?;

            // Generic stdlib methods return Ptr(U8) while the caller expects the
            // resolved T — resolve it from the receiver's type arguments and unbox.
            let actual_is_ptr = matches!(&actual_return_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8 | IrType::Void));
            if actual_is_ptr && actual_return_type != IrType::Void {
                let resolved_type = {
                    let type_table = self.type_table;
                    type_table.get(object.ty).and_then(|ti| {
                        match &ti.kind {
                            crate::tast::TypeKind::Class { type_args, .. }
                            | crate::tast::TypeKind::GenericInstance { type_args, .. } => {
                                if !type_args.is_empty() {
                                    let t = self.convert_type(type_args[0]);
                                    // Only unbox if resolved to a concrete primitive
                                    if matches!(
                                        t,
                                        IrType::I32
                                            | IrType::I64
                                            | IrType::F32
                                            | IrType::F64
                                            | IrType::Bool
                                    ) {
                                        Some(t)
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    })
                };
                if let Some(ref resolved) = resolved_type {
                    return self.maybe_unbox_for_extern_return(
                        call_result,
                        &actual_return_type,
                        resolved,
                    );
                }
            }
            return Some(call_result);
        } else {
            // Method not found by direct symbol lookup. Try rpkg/plugin extern
            // dispatch via a direct mapping lookup: get_stdlib_runtime_info's
            // guard rejects rpkg classes.
            {
                let method_name_str = self
                    .symbol_table
                    .get_symbol(*field)
                    .and_then(|s| self.string_interner.get(s.name))
                    .map(|s| s.to_string());
                let receiver_class = self.find_receiver_class_name(object);

                if let (Some(ref cls), Some(ref mn)) = (&receiver_class, &method_name_str) {
                    let plugin_match = self
                        .stdlib_mapping
                        .find_by_name_and_params(cls, mn, args.len())
                        .or_else(|| self.stdlib_mapping.find_by_name(cls, mn));

                    if let Some((sig, runtime_call)) = plugin_match {
                        let runtime_func = runtime_call.runtime_name;
                        let is_mir_wrapper = runtime_call.is_mir_wrapper;
                        let explicit_return_type =
                            runtime_call.return_type.map(|rt| rt.to_ir_type());
                        let is_static_call = sig.is_static;

                        // A plugin mapping declares its exact ABI (param_types
                        // includes self for instance methods) and is authoritative;
                        // the name-keyed wrapper registry has no entry for a pure
                        // plugin symbol and would fall back to Ptr.
                        let (expected_param_types, actual_return_type) =
                            if let Some(descs) = runtime_call.param_types {
                                let params: Vec<IrType> =
                                    descs.iter().map(|d| d.to_ir_type()).collect();
                                let ret = runtime_call
                                    .return_type
                                    .map(|d| d.to_ir_type())
                                    .or_else(|| explicit_return_type.clone())
                                    .unwrap_or_else(|| self.convert_type(expr.ty));
                                (params, ret)
                            } else {
                                self.get_stdlib_mir_wrapper_signature(runtime_func)
                                    .unwrap_or_else(|| {
                                        let mut params = if is_static_call {
                                            Vec::new()
                                        } else {
                                            vec![IrType::Ptr(Box::new(IrType::U8))]
                                        };
                                        for arg in args {
                                            params.push(self.convert_type(arg.ty));
                                        }
                                        let ret = explicit_return_type
                                            .clone()
                                            .unwrap_or_else(|| self.convert_type(expr.ty));
                                        (params, ret)
                                    })
                            };

                        let mut arg_regs = if is_static_call {
                            Vec::new()
                        } else {
                            let obj_reg = self.lower_expression(object)?;
                            vec![obj_reg]
                        };
                        // Coerce each user arg to the wrapper's declared param type:
                        // SIMD4f splat/make/load declare F32 lanes while Haxe `Float`
                        // args are F64, so without the demote the f64 bit pattern lands
                        // in the lanes as garbage. param_offset skips the leading self
                        // slot on instance wrappers.
                        let param_offset = if is_static_call { 0 } else { 1 };
                        for (i, arg) in args.iter().enumerate() {
                            if let Some(reg) = self.lower_expression(arg) {
                                let actual_ty = self.convert_type(arg.ty);
                                let expected_ty = expected_param_types
                                    .get(i + param_offset)
                                    .cloned()
                                    .unwrap_or_else(|| actual_ty.clone());
                                let final_reg =
                                    self.maybe_box_for_extern_call(reg, &actual_ty, &expected_ty)?;
                                arg_regs.push(final_reg);
                            }
                        }

                        let call_result = if is_mir_wrapper {
                            let fid = self.register_stdlib_mir_forward_ref(
                                runtime_func,
                                expected_param_types,
                                actual_return_type.clone(),
                            );
                            self.builder.build_call_direct(
                                fid,
                                arg_regs,
                                actual_return_type.clone(),
                            )?
                        } else {
                            let fid = self.get_or_register_extern_function(
                                runtime_func,
                                expected_param_types,
                                actual_return_type.clone(),
                            );
                            self.builder.build_call_direct(
                                fid,
                                arg_regs,
                                actual_return_type.clone(),
                            )?
                        };
                        // Reconcile the descriptor's native return type with the
                        // Haxe-declared type at this callsite.
                        let declared_ir = self.convert_type(expr.ty);
                        return Some(self.reconcile_extern_return(
                            call_result,
                            &actual_return_type,
                            &declared_ir,
                        ));
                    }
                }
            }

            // Fallback: Dynamic method call or stdlib method
            let object_type = object.ty;

            // Stdlib classes (including extern abstracts like Ptr/Ref/Box/Usize)
            // resolve via stdlib_mapping without any Dynamic unboxing.
            debug!(
                "[FIELDACCESS] Entering stdlib class check for object_type={:?}",
                object_type
            );
            {
                let type_table = self.type_table;
                let class_symbol_id = if let Some(type_info) = type_table.get(object_type) {
                    match &type_info.kind {
                        TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                        TypeKind::GenericInstance { base_type, .. } => {
                            // ObjectMap<Point, Int> and friends resolve to the base symbol.
                            if let Some(base_info) = type_table.get(*base_type) {
                                match &base_info.kind {
                                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                    TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                // Also try HIR type declarations for extern classes not in type_table
                let class_symbol_id = class_symbol_id.or_else(|| {
                    if let Some(type_decl) = self.current_hir_types.get(&object_type) {
                        if let HirTypeDecl::Class(class) = type_decl {
                            return Some(class.symbol_id);
                        }
                    }
                    None
                });

                // For static class calls where object.ty is invalid (TypeId::MAX),
                // extract the class symbol directly from the object Variable expression
                let class_symbol_id = class_symbol_id.or_else(|| {
                    if let HirExprKind::Variable {
                        symbol: obj_sym, ..
                    } = &object.kind
                    {
                        let sym = self.symbol_table.get_symbol(*obj_sym)?;
                        if matches!(
                            sym.kind,
                            crate::tast::SymbolKind::Class
                                | crate::tast::SymbolKind::Abstract
                                | crate::tast::SymbolKind::TypeAlias
                        ) {
                            return Some(*obj_sym);
                        }
                    }
                    None
                });

                if let Some(sym_id) = class_symbol_id {
                    let class_name_from_obj = self.symbol_table.get_symbol(sym_id).and_then(|s| {
                        // Prefer native_name (from @:native annotation)
                        s.native_name
                            .and_then(|nn| self.string_interner.get(nn))
                            .map(|ns| ns.replace("::", "_"))
                            .or_else(|| {
                                s.qualified_name
                                    .and_then(|qn| self.string_interner.get(qn))
                                    .map(|s| s.replace(".", "_"))
                            })
                    });

                    let method_name_opt = self
                        .symbol_table
                        .get_symbol(*field)
                        .and_then(|s| self.string_interner.get(s.name));

                    // Derive the class from the field's qualified name: the field symbol
                    // knows its package, so sys.thread.Thread.sleep does not resolve via
                    // rayzor.concurrent.Thread when the class symbol is shared.
                    let class_name_from_field = self
                        .symbol_table
                        .get_symbol(*field)
                        .and_then(|s| s.qualified_name)
                        .and_then(|qn| self.string_interner.get(qn))
                        .and_then(|qn_str| {
                            let parts: Vec<&str> = qn_str.split('.').collect();
                            if parts.len() >= 2 {
                                Some(parts[..parts.len() - 1].join("_"))
                            } else {
                                None
                            }
                        });

                    let class_name_opt = class_name_from_field.or(class_name_from_obj);

                    let class_name_opt = class_name_opt.or_else(|| {
                        self.symbol_table
                            .get_symbol(sym_id)
                            .and_then(|s| self.string_interner.get(s.name))
                            .map(|s| s.to_string())
                    });

                    if let (Some(class_name), Some(method_name)) = (class_name_opt, method_name_opt)
                    {
                        // Look up in stdlib_mapping (try abstract's own name first, then @:forward underlying)
                        let stdlib_result = self
                            .stdlib_mapping
                            .find_by_name(&class_name, method_name)
                            .map(|(sig, m)| (sig.clone(), m.clone()))
                            .or_else(|| {
                                let (underlying_type, forward_list) =
                                    self.abstract_forward_rules.get(&sym_id)?;
                                // An empty forward list forwards every method.
                                let method_interned =
                                    self.symbol_table.get_symbol(*field).map(|s| s.name);
                                let is_forwarded = forward_list.is_empty()
                                    || method_interned.map_or(false, |n| forward_list.contains(&n));
                                if !is_forwarded {
                                    return None;
                                }
                                let underlying_class = self
                                    .resolve_type_class_name_with(&type_table, *underlying_type)?;
                                self.stdlib_mapping
                                    .find_by_name(&underlying_class, method_name)
                                    .map(|(sig, m)| (sig.clone(), m.clone()))
                            });

                        if let Some((sig, mapping)) = stdlib_result {
                            // Extract data before dropping borrows
                            let is_mir_wrapper = mapping.is_mir_wrapper;
                            let runtime_name = mapping.runtime_name.to_string();
                            let has_return = mapping.has_return;
                            let returns_raw_value = mapping.returns_raw_value;
                            let raw_value_params = mapping.raw_value_params;
                            let extend_to_i64_params = mapping.extend_to_i64_params;
                            let explicit_return_type =
                                mapping.return_type.map(|rt| rt.to_ir_type());
                            let mapping_is_static = sig.is_static;

                            // First, try special runtime calls that need custom MIR lowering
                            // (e.g., Reflect.callMethod, Reflect.makeVarArgs, Type.typeof)
                            if let Some(special_result) = self.try_lower_special_runtime_call(
                                &runtime_name,
                                args,
                                result_type.clone(),
                                expr.source_location,
                            ) {
                                return special_result;
                            }

                            // Std.string on ValueType enum values needs special routing
                            // (same check that trace/interpolation paths do)
                            if runtime_name == "haxe_std_string_ptr" && args.len() == 1 {
                                if self.expr_is_value_type_expr(&args[0]) {
                                    let arg_reg = self.lower_expression(&args[0])?;
                                    return self.convert_value_type_to_string(arg_reg);
                                }
                            }

                            // Reflect.compare routes to haxe_reflect_compare_typed with
                            // types read off the arg expressions, which avoids boxing.
                            // Must run before the arg boxing loop below.
                            if runtime_name == "haxe_reflect_compare" && args.len() >= 2 {
                                let type_info = self.infer_reflect_compare_type_info(args);
                                if let Some(info) = type_info {
                                    let mut arg_regs = Vec::new();
                                    for arg in args.iter() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            let reg_ty = self
                                                .builder
                                                .get_register_type(reg)
                                                .unwrap_or(IrType::I64);
                                            let final_reg = if reg_ty != IrType::I64 {
                                                self.builder
                                                    .build_cast(reg, reg_ty, IrType::I64)
                                                    .unwrap_or(reg)
                                            } else {
                                                reg
                                            };
                                            arg_regs.push(final_reg);
                                        }
                                    }
                                    let tag_reg = match info {
                                        Ok(tag_value) => {
                                            self.builder.build_const(IrValue::I32(tag_value))?
                                        }
                                        Err(type_param_name) => {
                                            // Generic: placeholder tag with fixup
                                            let tag = self.builder.build_const(IrValue::I32(0))?;
                                            if let Some(func) = self.builder.current_function_mut()
                                            {
                                                func.type_param_tag_fixups
                                                    .push((tag, type_param_name));
                                            }
                                            tag
                                        }
                                    };
                                    arg_regs.push(tag_reg);
                                    let extern_func_id = self.get_or_register_extern_function(
                                        "haxe_reflect_compare_typed",
                                        vec![IrType::I64, IrType::I64, IrType::I32],
                                        IrType::I64,
                                    );
                                    let call_result = self.builder.build_call_direct(
                                        extern_func_id,
                                        arg_regs,
                                        IrType::I64,
                                    )?;
                                    if result_type == IrType::I32 {
                                        return self.builder.build_cast(
                                            call_result,
                                            IrType::I64,
                                            IrType::I32,
                                        );
                                    }
                                    return Some(call_result);
                                }
                            }

                            // Lower args, auto-boxing primitives when the MIR wrapper
                            // expects Ptr(U8). Instance methods prepend the receiver as
                            // param 0; static methods skip the object (a bare class ref).
                            let mir_wrapper_sig =
                                self.get_stdlib_mir_wrapper_signature(&runtime_name);
                            let is_static_call = mapping_is_static || !*is_method;
                            let mut arg_regs = Vec::new();
                            if !is_static_call {
                                let obj_reg = self.lower_expression(object)?;
                                arg_regs.push(obj_reg);
                            }
                            for (i, arg) in args.iter().enumerate() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    let actual_ty = self.convert_type(arg.ty);
                                    let param_idx = if is_static_call { i } else { i + 1 };
                                    let expected_ty = mir_wrapper_sig
                                        .as_ref()
                                        .and_then(|(params, _)| params.get(param_idx).cloned())
                                        .unwrap_or_else(|| actual_ty.clone());
                                    let final_reg = self.maybe_box_for_extern_call(
                                        reg,
                                        &actual_ty,
                                        &expected_ty,
                                    )?;
                                    arg_regs.push(final_reg);
                                }
                            }

                            if is_mir_wrapper {
                                let param_types: Vec<_> = mir_wrapper_sig
                                    .as_ref()
                                    .map(|(params, _)| params.clone())
                                    .unwrap_or_else(|| {
                                        arg_regs
                                            .iter()
                                            .map(|r| {
                                                self.builder
                                                    .get_register_type(*r)
                                                    .unwrap_or(IrType::I64)
                                            })
                                            .collect()
                                    });

                                let mir_return_type = mir_wrapper_sig
                                    .as_ref()
                                    .map(|(_, ret)| ret.clone())
                                    .unwrap_or_else(|| result_type.clone());

                                let mir_func_id = self.register_stdlib_mir_forward_ref(
                                    &runtime_name,
                                    param_types,
                                    mir_return_type.clone(),
                                );

                                let call_result = self.builder.build_call_direct(
                                    mir_func_id,
                                    arg_regs,
                                    mir_return_type.clone(),
                                )?;

                                // result_type can still be Ptr(Void) when the generic T is
                                // unresolved, so resolve T from the receiver's type args.
                                let resolved_result = {
                                    let needs_resolve = result_type == IrType::Any
                                        || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                        || result_type == IrType::I64;
                                    if needs_resolve {
                                        let type_table = self.type_table;
                                        type_table.get(object.ty).and_then(|ti| {
                                            if let crate::tast::TypeKind::Class { type_args, .. } = &ti.kind {
                                                if !type_args.is_empty() {
                                                    Some(self.convert_type(type_args[0]))
                                                } else {
                                                    None
                                                }
                                            } else if let crate::tast::TypeKind::GenericInstance { type_args, .. } = &ti.kind {
                                                if !type_args.is_empty() {
                                                    Some(self.convert_type(type_args[0]))
                                                } else {
                                                    None
                                                }
                                            } else {
                                                None
                                            }
                                        }).unwrap_or_else(|| result_type.clone())
                                    } else {
                                        result_type.clone()
                                    }
                                };

                                // MIR wrappers return their declared type directly — they
                                // never return a boxed DynamicValue*, so no unboxing.
                                return Some(call_result);
                            } else {
                                // Inject hidden enum type_id arg for runtime enum helpers
                                let pre = arg_regs.len();
                                self.inject_hidden_enum_type_id_arg(
                                    &runtime_name,
                                    args,
                                    &mut arg_regs,
                                );

                                // Use explicit types from the types: descriptor when available
                                let (param_types, return_type) = self
                                    .get_stdlib_mir_wrapper_signature(&runtime_name)
                                    .unwrap_or_else(|| {
                                        let params: Vec<_> = arg_regs
                                            .iter()
                                            .enumerate()
                                            .map(|(i, r)| {
                                                let param_bit = 1u32 << i;
                                                if raw_value_params & param_bit != 0 {
                                                    IrType::U64
                                                } else if extend_to_i64_params & param_bit != 0 {
                                                    IrType::I64
                                                } else {
                                                    self.builder
                                                        .get_register_type(*r)
                                                        .unwrap_or(IrType::I64)
                                                }
                                            })
                                            .collect();
                                        let ret = if let Some(ref ert) = explicit_return_type {
                                            ert.clone()
                                        } else if returns_raw_value {
                                            IrType::U64
                                        } else if has_return {
                                            result_type.clone()
                                        } else {
                                            IrType::Void
                                        };
                                        (params, ret)
                                    });

                                let extern_func_id = self.get_or_register_extern_function(
                                    &runtime_name,
                                    param_types,
                                    return_type.clone(),
                                );

                                let call_result = self.builder.build_call_direct(
                                    extern_func_id,
                                    arg_regs,
                                    return_type.clone(),
                                );

                                // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                                if let Some(call_reg) = call_result {
                                    return self.maybe_unbox_for_extern_return(
                                        call_reg,
                                        &return_type,
                                        &result_type,
                                    );
                                }
                                return call_result;
                            }
                        }
                    }
                }
            }

            // First check if the object is Dynamic - handle auto-unbox for method calls
            let type_table = self.type_table;
            if let Some(type_info) = type_table.get(object_type) {
                if matches!(type_info.kind, TypeKind::Dynamic) {
                    // Resolve the method by name, excluding the
                    // currently-compiling function: a same-named
                    // method would otherwise resolve back to the
                    // enclosing function and recurse forever.
                    let method_name = self.symbol_table.get_symbol(*field).map(|s| s.name);
                    let caller_func_id = self.builder.current_function;
                    if let Some(name) = method_name {
                        // Match on arity too, so same-named methods
                        // on unrelated classes are not picked up.
                        let target_argc = args.len() + 1; // +1 for receiver
                        let mut found_func = None;
                        for (sym, &func_id) in &self.function_map {
                            if Some(func_id) == caller_func_id {
                                continue;
                            }
                            if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                if sym_info.name != name {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                            if let Some(func) = self.builder.module.functions.get(&func_id) {
                                if func.signature.parameters.len() != target_argc {
                                    continue;
                                }
                            }
                            found_func = Some(func_id);
                            break;
                        }

                        if let Some(func_id) = found_func {
                            let obj_reg = self.lower_expression(object)?;

                            // Unbox the Dynamic to get the actual object pointer
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let unbox_func_id = self.get_or_register_extern_function(
                                "haxe_unbox_reference_ptr",
                                vec![ptr_u8.clone()],
                                ptr_u8.clone(),
                            );
                            let unboxed_obj = self.builder.build_call_direct(
                                unbox_func_id,
                                vec![obj_reg],
                                ptr_u8,
                            )?;

                            let arg_regs: Vec<_> =
                                std::iter::once(unboxed_obj) // unboxed 'this' as first arg
                                    .chain(
                                        args.iter()
                                            .filter_map(|a| self.lower_expression(a)),
                                    )
                                    .collect();

                            let actual_return_type =
                                if let Some(func) = self.builder.module.functions.get(&func_id) {
                                    func.signature.return_type.clone()
                                } else {
                                    result_type.clone()
                                };

                            return self.builder.build_call_direct(
                                func_id,
                                arg_regs,
                                actual_return_type,
                            );
                        }
                    }
                }
            }

            {
                let type_table = self.type_table;
                if let Some(type_info) = type_table.get(object_type) {
                    debug!(
                        "[CHECK STRING] object_type={:?}, kind={:?}",
                        object_type, type_info.kind
                    );
                    if matches!(type_info.kind, TypeKind::String) {
                        let method_name = self
                            .symbol_table
                            .get_symbol(*field)
                            .and_then(|s| self.string_interner.get(s.name));

                        if let Some(method_name) = method_name {
                            // String methods with optional params (indexOf, lastIndexOf)
                            // register one mapping per arity.
                            let arg_count = args.len();
                            let mapping_opt = self
                                .stdlib_mapping
                                .find_by_name_and_params("String", method_name, arg_count)
                                .or_else(|| {
                                    self.stdlib_mapping.find_by_name("String", method_name)
                                });

                            if let Some((_sig, mapping)) = mapping_opt {
                                let runtime_func = mapping.runtime_name;

                                debug!(
                                    "[STRING METHOD] Found String.{} with {} args -> {}",
                                    method_name, arg_count, runtime_func
                                );

                                let obj_reg = self.lower_expression(object)?;

                                let method_arg_regs: Vec<_> = args
                                    .iter()
                                    .filter_map(|a| self.lower_expression(a))
                                    .collect();

                                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                                let mut param_types = vec![string_ptr_ty.clone()];
                                for arg in &method_arg_regs {
                                    // Haxe Int is i32, default to I32 for integer args
                                    let arg_ty =
                                        self.builder.get_register_type(*arg).unwrap_or(IrType::I32);
                                    param_types.push(arg_ty);
                                }

                                // String-returning methods hand back a HaxeString pointer.
                                let return_type = if result_type == IrType::String {
                                    string_ptr_ty.clone()
                                } else {
                                    result_type.clone()
                                };

                                let runtime_func_id = self.get_or_register_extern_function(
                                    runtime_func,
                                    param_types,
                                    return_type.clone(),
                                );

                                let mut call_args = vec![obj_reg];
                                call_args.extend(method_arg_regs);

                                return self.builder.build_call_direct(
                                    runtime_func_id,
                                    call_args,
                                    return_type,
                                );
                            }
                        }
                    }
                }
            }

            // Check if the object type is a rayzor stdlib class (or GenericInstance like Deque<Int>)
            let type_table = self.type_table;
            let mut class_symbol_id = if let Some(type_info) = type_table.get(object_type) {
                match &type_info.kind {
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    TypeKind::Abstract { symbol_id, .. } => Some(*symbol_id),
                    TypeKind::GenericInstance { base_type, .. } => {
                        // Deque<Int> and friends resolve to the base symbol.
                        if let Some(base_info) = type_table.get(*base_type) {
                            match &base_info.kind {
                                TypeKind::Class { symbol_id, .. } => {
                                    debug!(
                                        "[STDLIB FALLBACK] GenericInstance base class symbol_id={:?}",
                                        symbol_id
                                    );
                                    Some(*symbol_id)
                                }
                                TypeKind::Abstract { symbol_id, .. } => {
                                    debug!(
                                        "[STDLIB FALLBACK] GenericInstance base abstract symbol_id={:?}",
                                        symbol_id
                                    );
                                    Some(*symbol_id)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };

            // Fallback for static class references where object_type is not concrete
            // (e.g., extern class identifiers like Std/Math).
            if class_symbol_id.is_none() {
                if let HirExprKind::Variable {
                    symbol: object_symbol,
                    ..
                } = &object.kind
                {
                    if let Some(sym) = self.symbol_table.get_symbol(*object_symbol) {
                        if matches!(
                            sym.kind,
                            crate::tast::symbols::SymbolKind::Class
                                | crate::tast::symbols::SymbolKind::Abstract
                                | crate::tast::symbols::SymbolKind::TypeAlias
                        ) {
                            class_symbol_id = Some(sym.id);
                        }
                    }
                }
            }

            if let Some(symbol_id) = class_symbol_id {
                if let Some(class_symbol) = self.symbol_table.get_symbol(symbol_id) {
                    if let Some(class_name) = self.string_interner.get(class_symbol.name) {
                        debug!(
                            "[STDLIB FALLBACK] Found class '{}', checking for stdlib method",
                            class_name
                        );

                        let qualified_name_opt = class_symbol
                            .native_name
                            .and_then(|nn| self.string_interner.get(nn))
                            .map(|n| n.replace("::", "_"))
                            .or_else(|| {
                                class_symbol
                                    .qualified_name
                                    .and_then(|qn| self.string_interner.get(qn))
                                    .map(|s| s.to_string())
                            });

                        let method_name =
                            if let Some(field_sym) = self.symbol_table.get_symbol(*field) {
                                self.string_interner.get(field_sym.name)
                            } else {
                                None
                            };

                        if let Some(method_name) = method_name {
                            let static_args = self.effective_static_call_args(args);
                            let object_qualified_name_opt = if let HirExprKind::Variable {
                                symbol: object_symbol,
                                ..
                            } = &object.kind
                            {
                                self.symbol_table
                                    .get_symbol(*object_symbol)
                                    .and_then(|s| s.qualified_name)
                                    .and_then(|qn| self.string_interner.get(qn))
                                    .map(|s| s.to_string())
                            } else {
                                None
                            };

                            // Prefer class-qualified lookup, but keep the global static
                            // fallback: some extern classes (e.g. Math) carry no
                            // qualified or native name on the symbol.
                            let runtime_func_opt = qualified_name_opt
                                .as_deref()
                                .and_then(|class_qualified_name| {
                                    let lookup =
                                        format!("{}.{}", class_qualified_name, method_name);
                                    self.get_static_stdlib_runtime_func_with_params(
                                        &lookup,
                                        method_name,
                                        static_args.len(),
                                    )
                                })
                                .or_else(|| {
                                    object_qualified_name_opt.as_deref().and_then(
                                        |class_qualified_name| {
                                            let lookup =
                                                format!("{}.{}", class_qualified_name, method_name);
                                            self.get_static_stdlib_runtime_func_with_params(
                                                &lookup,
                                                method_name,
                                                static_args.len(),
                                            )
                                        },
                                    )
                                })
                                .or_else(|| {
                                    let lookup = format!("{}.{}", class_name, method_name);
                                    self.get_static_stdlib_runtime_func_with_params(
                                        &lookup,
                                        method_name,
                                        static_args.len(),
                                    )
                                })
                                .or_else(|| {
                                    self.stdlib_mapping
                                        .find_static_method_by_name_and_params(
                                            method_name,
                                            static_args.len(),
                                        )
                                        .map(|(_, mapping)| mapping.runtime_name)
                                });

                            if let Some(runtime_func) = runtime_func_opt {
                                // Static methods take no receiver, so the object is dropped.
                                let arg_regs: Vec<_> = static_args
                                    .iter()
                                    .filter_map(|a| self.lower_expression(a))
                                    .collect();
                                debug!(
                                    "[FIELD-PATH STATIC] Dispatching {}.{} -> {}, arg_count={}",
                                    class_name,
                                    method_name,
                                    runtime_func,
                                    arg_regs.len()
                                );

                                // Use the function signature from the mapping (hlp_* introspection)
                                // if available; this is the authoritative source of type info.
                                let (expected_param_types, expected_return_type) = self
                                    .get_extern_function_signature(&runtime_func)
                                    .unwrap_or_else(|| {
                                        let param_types: Vec<IrType> = arg_regs
                                            .iter()
                                            .map(|reg| {
                                                self.builder
                                                    .get_register_type(*reg)
                                                    .unwrap_or(IrType::Any)
                                            })
                                            .collect();
                                        (param_types, result_type.clone())
                                    });

                                let final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                        .map(|(i, &reg)| {
                                            if let (Some(expected_ty), Some(actual_ty)) = (
                                                expected_param_types.get(i),
                                                self.builder.get_register_type(reg)
                                            ) {
                                                if *expected_ty != actual_ty {
                                                    let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                    if is_ptr_u8 && i < args.len() {
                                                        if let Some(boxed) = self.box_value_for_dynamic(reg, args[i].ty) {
                                                            return boxed;
                                                        }
                                                    }
                                                    if let Some(casted) = self.builder.build_cast(reg, actual_ty.clone(), expected_ty.clone()) {
                                                        return casted;
                                                    }
                                                }
                                            }
                                            reg
                                        })
                                        .collect();

                                let runtime_func_id = self.get_or_register_extern_function(
                                    &runtime_func,
                                    expected_param_types,
                                    expected_return_type.clone(),
                                );

                                let call_result = self.builder.build_call_direct(
                                    runtime_func_id,
                                    final_arg_regs,
                                    expected_return_type.clone(),
                                );
                                return call_result;
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
