//! Instance methods on stdlib and Dynamic receivers.

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
    pub(crate) fn try_stdlib_instance_call(
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
            unreachable!("try_stdlib_instance_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if *is_method && !args.is_empty() {
            // args[0] is the receiver; resolve it through TypeAlias. Prefer the
            // MIR-time override for a receiver bound by an iface call whose return
            // type was re-resolved (Dynamic → concrete): `args[0].ty` still reads
            // Dynamic and would route the call to the Dynamic-receiver path.
            let receiver_type = self
                .effective_receiver_type(&args[0])
                .map(|tid| self.resolve_through_aliases(tid))
                .unwrap_or_else(|| self.resolve_through_aliases(args[0].ty));

            {
                let type_table = self.type_table;
                if let Some(type_info) = type_table.get(receiver_type) {
                    debug!(
                        "[METHOD CALL] receiver_type={:?}, kind={:?}",
                        receiver_type, type_info.kind
                    );
                } else {
                    let method_name = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .map(|s| self.string_interner.get(s.name));
                    debug!(
                        "[METHOD CALL] receiver_type={:?} NOT IN TYPE TABLE, method={:?}",
                        receiver_type, method_name
                    );
                }
            }

            // Dynamic/TypeParameter receivers resolve the method by name. TypeParameter
            // arises from chained calls on generics, e.g. Arc<T>.get().lock() where
            // get() returns T.
            {
                let type_table = self.type_table;
                if let Some(type_info) = type_table.get(receiver_type) {
                    if matches!(
                        type_info.kind,
                        TypeKind::Dynamic
                            | TypeKind::TypeParameter { .. }
                            | TypeKind::Placeholder { .. }
                            | TypeKind::Unknown
                    ) {
                        let method_name_str = self
                            .symbol_table
                            .get_symbol(*symbol)
                            .and_then(|s| self.string_interner.get(s.name));

                        // For Dynamic receivers check user-defined methods first: stdlib
                        // has common names ("get", "set", "sum") that collide with user
                        // methods on Dynamic-typed objects.
                        let receiver_is_dynamic = {
                            let type_table = self.type_table;
                            type_table
                                .get(receiver_type)
                                .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                .unwrap_or(false)
                        };
                        let user_func_for_dynamic = if receiver_is_dynamic {
                            let method_name_is =
                                self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                            if let Some(name) = method_name_is {
                                let mut found = None;
                                for (sym, &fid) in &self.function_map {
                                    if let Some(si) = self.symbol_table.get_symbol(*sym) {
                                        if si.name == name {
                                            found = Some(fid);
                                            break;
                                        }
                                    }
                                }
                                found
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        if let Some(func_id) = user_func_for_dynamic {
                            let receiver_reg = self.lower_expression(&args[0])?;

                            // Dynamic receivers are always boxed by haxe_box_reference_ptr,
                            // even when the register type reads Ptr(Void) after a cast, so
                            // unbox unless a class hint marks it a raw stdlib container.
                            let has_class_hint =
                                self.register_class_hints.contains_key(&receiver_reg);
                            let should_unbox = !has_class_hint;
                            let actual_receiver = if should_unbox {
                                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                let unbox_func_id = self.get_or_register_extern_function(
                                    "haxe_unbox_reference_ptr",
                                    vec![ptr_u8.clone()],
                                    ptr_u8.clone(),
                                );
                                self.builder.build_call_direct(
                                    unbox_func_id,
                                    vec![receiver_reg],
                                    ptr_u8,
                                )?
                            } else {
                                receiver_reg
                            };

                            let arg_regs: Vec<_> = std::iter::once(actual_receiver)
                                .chain(args[1..].iter().filter_map(|a| self.lower_expression(a)))
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

                        let is_stdlib_method = method_name_str
                            .map(|m| self.stdlib_mapping.any_class_has_method(m))
                            .unwrap_or(false);
                        if is_stdlib_method {
                            let method_name = method_name_str.unwrap();
                            // The receiver does not count towards the params.
                            let actual_param_count = args.len().saturating_sub(1);
                            debug!(
                                "[DYNAMIC METHOD] Found stdlib method '{}' in mapping, param_count={}",
                                method_name, actual_param_count
                            );

                            let matching_classes =
                                self.stdlib_mapping.find_classes_with_method(method_name);
                            debug!(
                                "[DYNAMIC STDLIB] {} classes have method '{}' (before param count filter)",
                                matching_classes.len(),
                                method_name
                            );

                            // Filter by param count to disambiguate overloaded methods
                            // e.g., Array.join(sep) with 1 param vs Thread.join() with 0 params
                            let mut filtered_classes: Vec<_> = matching_classes
                                .into_iter()
                                .filter(|(_, _, call)| call.param_count == actual_param_count)
                                .collect();
                            debug!(
                                "[DYNAMIC STDLIB] {} classes after param count filter",
                                filtered_classes.len()
                            );

                            // Disambiguate using class hints when multiple classes match
                            // (e.g., Arc.get vs MutexGuard.get — both have 0 params)
                            if filtered_classes.len() > 1 {
                                let receiver_hint = if let HirExprKind::Variable {
                                    symbol: recv_sym,
                                    ..
                                } = &args[0].kind
                                {
                                    self.monomorphized_var_types.get(&recv_sym).cloned()
                                } else {
                                    None
                                };

                                if let Some(hint) = &receiver_hint {
                                    let hinted: Vec<_> = filtered_classes
                                        .iter()
                                        .filter(|(class, _, _)| {
                                            self.stdlib_mapping.same_class(class, hint)
                                        })
                                        .copied()
                                        .collect();
                                    if !hinted.is_empty() {
                                        debug!(
                                            "[DYNAMIC STDLIB] Disambiguated by class hint '{}': {} -> {} matches",
                                            hint,
                                            filtered_classes.len(),
                                            hinted.len()
                                        );
                                        filtered_classes = hinted;
                                    }
                                }
                            }

                            // No priority guessing: more than one distinct runtime target
                            // (same-target aliases like rayzor_Bytes.get / haxe_io_Bytes.get
                            // are not ambiguous) means the receiver type is unresolved, and
                            // any pick would call an unrelated class's method.
                            {
                                let mut distinct: Vec<&str> = filtered_classes
                                    .iter()
                                    .map(|(_, _, call)| call.runtime_name)
                                    .collect();
                                distinct.sort_unstable();
                                distinct.dedup();
                                if distinct.len() > 1 {
                                    let candidates = filtered_classes
                                        .iter()
                                        .map(|(class, _, _)| *class)
                                        .collect::<Vec<_>>()
                                        .join(", ");
                                    self.add_error(
                                        &format!(
                                            "E0801: ambiguous dynamic method dispatch: `{}` with {} argument(s) matches multiple stdlib classes ({}) and the receiver's type is unresolved. Annotate the receiver's type so the call resolves to one class",
                                            method_name, actual_param_count, candidates
                                        ),
                                        expr.source_location,
                                    );
                                    return None;
                                }
                            }

                            // A unique runtime target remains — dispatch to it.
                            if let Some(&(class_name, _sig, runtime_call)) =
                                filtered_classes.first()
                            {
                                debug!(
                                    "[DYNAMIC STDLIB] Using {}.{} -> {}",
                                    class_name, method_name, runtime_call.runtime_name
                                );
                                let runtime_func = runtime_call.runtime_name;

                                if self.stdlib_mapping.is_mir_wrapper_class(class_name) {
                                    // Use runtime_name directly as the MIR wrapper function name
                                    // (e.g., "Arc_init" not "rayzor_concurrent_Arc_init")
                                    let mir_func_name = runtime_func.to_string();
                                    debug!(
                                        "[DYNAMIC STDLIB MIR] Using MIR wrapper: {}",
                                        mir_func_name
                                    );

                                    // If the receiver (args[0]) can't be lowered, skip this
                                    // handler rather than emit a 0-arg call for an instance
                                    // method that expects self.
                                    let mir_wrapper_sig =
                                        self.get_stdlib_mir_wrapper_signature(&mir_func_name);
                                    let mut arg_regs = Vec::new();
                                    let mut param_types = Vec::new();
                                    let mut receiver_failed = false;
                                    for (i, arg) in args.iter().enumerate() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            let actual_ty = self.convert_type(arg.ty);
                                            let expected_ty = mir_wrapper_sig
                                                .as_ref()
                                                .and_then(|(params, _)| params.get(i).cloned())
                                                .unwrap_or_else(|| actual_ty.clone());

                                            // Erased TypeParameter/Dynamic/Placeholder args are
                                            // pointers carried in an I64 and are cast to Ptr(U8);
                                            // concrete primitives (I32/F64/Bool, e.g. from
                                            // Channel<Int>) must be boxed instead.
                                            let is_erased_type_param = {
                                                let type_table = self.type_table;
                                                type_table
                                                    .get(arg.ty)
                                                    .map(|ti| {
                                                        matches!(
                                                            ti.kind,
                                                            TypeKind::TypeParameter { .. }
                                                                | TypeKind::Dynamic
                                                                | TypeKind::Placeholder { .. }
                                                        )
                                                    })
                                                    .unwrap_or(false)
                                            };
                                            let reg_ir_type = self.builder.get_register_type(reg);
                                            let is_concrete_primitive = matches!(
                                                reg_ir_type,
                                                Some(IrType::I32)
                                                    | Some(IrType::F32)
                                                    | Some(IrType::F64)
                                                    | Some(IrType::Bool)
                                            );
                                            let final_reg = if (mir_func_name == "Channel_send"
                                                || mir_func_name == "Channel_trySend")
                                                && i >= 1
                                            {
                                                // Uniformly box Channel payloads (refs too)
                                                // so the erased receive can tag-dispatch.
                                                // i==0 is the channel handle — never box it.
                                                self.box_channel_payload(
                                                    reg,
                                                    arg.ty,
                                                    &actual_ty,
                                                    &expected_ty,
                                                )?
                                            } else if is_erased_type_param
                                                && matches!(actual_ty, IrType::I64)
                                                && matches!(&expected_ty, IrType::Ptr(_))
                                                && !is_concrete_primitive
                                            {
                                                self.builder
                                                    .build_cast(
                                                        reg,
                                                        IrType::I64,
                                                        expected_ty.clone(),
                                                    )
                                                    .unwrap_or(reg)
                                            } else if is_concrete_primitive
                                                && matches!(&expected_ty, IrType::Ptr(_))
                                            {
                                                let box_ty = reg_ir_type.unwrap();
                                                self.maybe_box_for_extern_call(
                                                    reg,
                                                    &box_ty,
                                                    &expected_ty,
                                                )?
                                            } else {
                                                self.maybe_box_for_extern_call(
                                                    reg,
                                                    &actual_ty,
                                                    &expected_ty,
                                                )?
                                            };
                                            arg_regs.push(final_reg);
                                            param_types.push(expected_ty);
                                        } else if i == 0 {
                                            // Receiver failed to lower — can't call instance method
                                            receiver_failed = true;
                                            break;
                                        }
                                    }

                                    if receiver_failed {
                                        // Don't generate a broken call; fall through
                                    } else {
                                        let mir_return_type = mir_wrapper_sig
                                            .as_ref()
                                            .map(|(_, ret)| ret.clone())
                                            .unwrap_or_else(|| result_type.clone());

                                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                                            &mir_func_name,
                                            param_types,
                                            mir_return_type.clone(),
                                        );

                                        let call_result = self.builder.build_call_direct(
                                            mir_func_id,
                                            arg_regs,
                                            mir_return_type.clone(),
                                        )?;

                                        // Resolve generic T from the receiver's type args to
                                        // unbox: Channel<Int>.tryReceive() returns Ptr(U8)
                                        // where the caller expects I32.
                                        let resolved_expected = {
                                            let type_table = self.type_table;
                                            let from_receiver = if !args.is_empty() {
                                                type_table.get(args[0].ty).and_then(|ti| match &ti
                                                    .kind
                                                {
                                                    crate::tast::TypeKind::Class {
                                                        type_args,
                                                        ..
                                                    }
                                                    | crate::tast::TypeKind::GenericInstance {
                                                        type_args,
                                                        ..
                                                    } => {
                                                        if !type_args.is_empty() {
                                                            let t = self.convert_type(type_args[0]);
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
                                                })
                                            } else {
                                                None
                                            };
                                            // A Null<T> return resolves to its inner primitive.
                                            let from_optional =
                                                type_table.get(expr.ty).and_then(|ti| {
                                                    if let crate::tast::TypeKind::Optional {
                                                        inner_type,
                                                    } = &ti.kind
                                                    {
                                                        let t = self.convert_type(*inner_type);
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
                                                });
                                            from_receiver
                                                .or(from_optional)
                                                .unwrap_or_else(|| result_type.clone())
                                        };
                                        let final_result = if mir_func_name == "Channel_receive"
                                            || mir_func_name == "Channel_tryReceive"
                                        {
                                            self.unbox_channel_return(
                                                call_result,
                                                &resolved_expected,
                                                mir_func_name == "Channel_tryReceive",
                                            )
                                        } else {
                                            self.maybe_unbox_for_extern_return(
                                                call_result,
                                                &mir_return_type,
                                                &resolved_expected,
                                            )
                                        };

                                        // Tag the result register with its class so later calls
                                        // on the value disambiguate: Mutex.lock() yields a
                                        // MutexGuard, which then resolves .get()/.unlock().
                                        if let Some(result_reg) = final_result {
                                            let return_class =
                                                self.get_return_class_hint(class_name, method_name);
                                            self.register_class_hints
                                                .insert(result_reg, return_class.to_string());
                                        }

                                        return final_result;
                                    } // end else !receiver_failed
                                } else {
                                    // Direct extern call
                                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

                                    // Extract runtime_call data before borrowing self mutably
                                    let has_return = runtime_call.has_return;

                                    // Lower all arguments using a for loop (not a closure)
                                    // to avoid borrow conflict with stdlib_mapping
                                    let mut arg_regs = Vec::new();
                                    for arg in args {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                        }
                                    }

                                    let param_types: Vec<_> =
                                        arg_regs.iter().map(|_| ptr_u8.clone()).collect();

                                    let return_type = if has_return {
                                        ptr_u8.clone()
                                    } else {
                                        IrType::Void
                                    };

                                    let extern_func_id = self.get_or_register_extern_function(
                                        runtime_func,
                                        param_types,
                                        return_type.clone(),
                                    );

                                    return self.builder.build_call_direct(
                                        extern_func_id,
                                        arg_regs,
                                        return_type,
                                    );
                                }
                            }
                            // If no mapping found, fall through to regular dispatch
                        } else {
                            // Look up method by name in function_map (generic Dynamic dispatch)
                            let method_name = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                            if let Some(name) = method_name {
                                let mut found_func = None;
                                for (sym, &func_id) in &self.function_map {
                                    if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                        if sym_info.name == name {
                                            found_func = Some(func_id);
                                            break;
                                        }
                                    }
                                }

                                if let Some(func_id) = found_func {
                                    let receiver_reg = self.lower_expression(&args[0])?;

                                    // A class hint means the receiver is a raw class pointer
                                    // from a stdlib MIR wrapper (e.g. MutexGuard_get), not a
                                    // boxed DynamicValue. Every other Dynamic receiver is
                                    // boxed by haxe_box_reference_ptr even when the register
                                    // type reads Ptr(Void) after a cast.
                                    let has_class_hint =
                                        self.register_class_hints.contains_key(&receiver_reg);
                                    let receiver_mir_type =
                                        self.builder.get_register_type(receiver_reg);
                                    let should_unbox = !has_class_hint;

                                    let actual_receiver = if should_unbox {
                                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                        let unbox_func_id = self.get_or_register_extern_function(
                                            "haxe_unbox_reference_ptr",
                                            vec![ptr_u8.clone()],
                                            ptr_u8.clone(),
                                        );
                                        self.builder.build_call_direct(
                                            unbox_func_id,
                                            vec![receiver_reg],
                                            ptr_u8,
                                        )?
                                    } else {
                                        debug!(
                                            "[DYNAMIC METHOD] Skipping unbox - stdlib container method"
                                        );
                                        receiver_reg
                                    };

                                    let arg_regs: Vec<_> = std::iter::once(actual_receiver)
                                        .chain(
                                            args[1..]
                                                .iter()
                                                .filter_map(|a| self.lower_expression(a)),
                                        )
                                        .collect();

                                    let actual_return_type = if let Some(func) =
                                        self.builder.module.functions.get(&func_id)
                                    {
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
                }
            }

            // Extern generic classes like Vec<T> can carry a receiver type of
            // TypeId::MAX; fall back to the monomorphized class tracked at the
            // variable's assignment.
            if receiver_type == TypeId::from_raw(u32::MAX) {
                debug!(
                    "[MONO VAR CHECK] receiver_type is MAX, checking monomorphized_var_types ({} entries)",
                    self.monomorphized_var_types.len()
                );

                let receiver_symbol = match &args[0].kind {
                    HirExprKind::Variable { symbol, .. } => Some(*symbol),
                    HirExprKind::Field { field, .. } => Some(*field),
                    _ => None,
                };
                debug!(
                    "[MONO VAR CHECK] Receiver expression symbol: {:?}",
                    receiver_symbol
                );

                if let Some(var_symbol) = receiver_symbol {
                    if let Some(mono_class) = self.monomorphized_var_types.get(&var_symbol).cloned()
                    {
                        if let Some(method_sym) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(method_sym.name) {
                                debug!(
                                    "[MONO VAR DISPATCH] Found tracked class '{}' for variable {:?}, method '{}'",
                                    mono_class, var_symbol, method_name
                                );

                                // Resolve the MIR wrapper (VecI32_push, VecF64_get, …)
                                // through the mapping rather than by spelling the
                                // wrapper's name out of the class key, so the key's
                                // spelling and the runtime symbol stay independent.
                                let wrapper = self
                                    .stdlib_mapping
                                    .find_by_name_and_params(
                                        &mono_class,
                                        method_name,
                                        args.len().saturating_sub(1),
                                    )
                                    .filter(|(_, call)| call.is_mir_wrapper)
                                    .map(|(_, call)| call.runtime_name);

                                if let Some((mir_func_name, (mir_param_types, mir_return_type))) =
                                    wrapper.and_then(|name| {
                                        Some((name, self.get_stdlib_mir_wrapper_signature(name)?))
                                    })
                                {
                                    debug!(
                                        "[MONO VAR DISPATCH] Using MIR wrapper: {}",
                                        mir_func_name
                                    );

                                    let mut arg_regs = Vec::new();
                                    for arg in args {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                        }
                                    }

                                    let mir_func_id = self.register_stdlib_mir_forward_ref(
                                        mir_func_name,
                                        mir_param_types.clone(),
                                        mir_return_type.clone(),
                                    );

                                    debug!(
                                        "[MONO VAR DISPATCH] Registered forward ref to {} with ID {:?}",
                                        mir_func_name, mir_func_id
                                    );

                                    let result = self.builder.build_call_direct(
                                        mir_func_id,
                                        arg_regs,
                                        mir_return_type,
                                    );
                                    debug!(
                                        "[MONO VAR DISPATCH] Generated call, result: {:?}",
                                        result
                                    );
                                    return result;
                                }
                            }
                        }
                    }
                }
            }

            // Skip the instance-method path when the receiver is a class type: static
            // calls can arrive with is_method=true, e.g. Thread.spawn(closure) seen as
            // Thread(receiver).spawn(closure).
            let receiver_is_class_type = {
                let type_table = self.type_table;
                type_table.get(receiver_type)
                    .map(|ti| {
                        if let crate::tast::core::TypeKind::Class { symbol_id, .. } = &ti.kind {
                            self.symbol_table.get_symbol(*symbol_id)
                                .and_then(|s| self.canonical_stdlib_class_name(s))
                                .map(|key| {
                                    let is_mir_wrapper = self.stdlib_mapping.is_mir_wrapper_class(&key);
                                    if is_mir_wrapper {
                                        debug!("[GUARD] Receiver type is {} class (MIR wrapper), skipping instance method path", key);
                                    }
                                    is_mir_wrapper
                                })
                                .unwrap_or(false)
                        } else {
                            false
                        }
                    })
                    .unwrap_or(false)
            };

            // Try the receiver type path first (for true instance methods)
            // Skip if receiver is a MIR wrapper class type (those are static methods)
            {
                let sym_name = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name))
                    .unwrap_or("?");
                if matches!(
                    sym_name,
                    "balance" | "setLoop" | "compare" | "merge" | "minBinding" | "removeMinBinding"
                ) {
                    debug!(
                        "[DISPATCH_TRACE] '{}' receiver_is_class_type={}, receiver_type={:?}",
                        sym_name, receiver_is_class_type, receiver_type
                    );
                }
            }
            probe!(self.try_stdlib_runtime_dispatch(
                expr,
                receiver_type,
                receiver_is_class_type,
                result_type.clone()
            ));
        }
        *fell_through = true;
        None
    }
}
