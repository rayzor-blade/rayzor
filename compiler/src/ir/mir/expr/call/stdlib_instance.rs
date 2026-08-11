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
            // The first arg is the receiver for instance method calls
            // Resolve TypeAlias to get the actual receiver type.
            //
            // Cross-context override: when the receiver
            // expression is a Variable whose binding was
            // populated by an iface call whose return type
            // we re-resolved at MIR time (Dynamic →
            // concrete), use that override instead of the
            // poisoned `args[0].ty`. Without this, the
            // dispatch falls into the Dynamic-receiver path
            // and the downstream MIR-wrapper boxing
            // produces a malformed call (SIGSEGVs on
            // e.g. `Array.push`).
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
                    // Print method name for calls with invalid receiver type
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

            // SPECIAL CASE: Handle Dynamic and TypeParameter method calls
            // When receiver is Dynamic or TypeParameter (unresolved generic), resolve method by name
            // TypeParameter arises from chained calls on generic types like Arc<T>.get().lock()
            // where the return type of get() is TypeParameter T
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
                        // First, check if this might be a stdlib method call
                        // by checking if the receiver expression comes from a stdlib function
                        // (i.e., its result type would be Ptr(Void) for MIR wrappers)
                        let method_name_str = self
                            .symbol_table
                            .get_symbol(*symbol)
                            .and_then(|s| self.string_interner.get(s.name));

                        // Check if any stdlib class has this method - use the mapping dynamically
                        // instead of hardcoding method names. This handles cases like:
                        // - MutexGuard.get() vs Arc.get() - both have "get" but are different
                        // - Mutex.lock() returning Dynamic typed as MutexGuard
                        // For Dynamic receivers, check user-defined methods FIRST.
                        // Stdlib has common names like "sum", "get", "set" that
                        // collide with user methods on Dynamic-typed objects.
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
                            // User-defined method found for Dynamic receiver — use it with unboxing
                            let receiver_reg = self.lower_expression(&args[0])?;

                            // Dynamic receivers are always boxed (from haxe_box_reference_ptr),
                            // even if the MIR register type shows Ptr(Void) due to cast.
                            // Always unbox unless receiver has a class hint (stdlib container).
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

                            // Lower remaining args
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
                            // Calculate actual param count (exclude receiver for instance methods)
                            let actual_param_count = args.len().saturating_sub(1);
                            debug!(
                                "[DYNAMIC METHOD] Found stdlib method '{}' in mapping, param_count={}",
                                method_name, actual_param_count
                            );

                            // Query the stdlib mapping for all classes that have this method.
                            // Results are sorted by priority (MutexGuard before Arc, etc.)
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
                                // Check if the receiver variable has a class hint
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
                                            *class == hint.as_str()
                                                || class.ends_with(&format!("_{}", hint))
                                                || hint.ends_with(&format!("_{}", class))
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

                            // No priority guessing: if the candidates still name more
                            // than one distinct runtime function (same-target aliases
                            // like rayzor_Bytes.get / haxe_io_Bytes.get are NOT
                            // ambiguous), the receiver's type is unresolved and any
                            // pick would silently call an unrelated class's method.
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

                                // Check if this is a MIR wrapper class
                                if self.stdlib_mapping.is_mir_wrapper_class(class_name) {
                                    // Use runtime_name directly as the MIR wrapper function name
                                    // (e.g., "Arc_init" not "rayzor_concurrent_Arc_init")
                                    let mir_func_name = runtime_func.to_string();
                                    debug!(
                                        "[DYNAMIC STDLIB MIR] Using MIR wrapper: {}",
                                        mir_func_name
                                    );

                                    // Lower all arguments with auto-boxing
                                    // CRITICAL: If the receiver (args[0]) can't be lowered, skip this handler
                                    // to prevent generating 0-arg calls for instance methods that expect self.
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

                                            // TypeParameter/Dynamic/Placeholder args erased to I64
                                            // should be CAST to Ptr(U8), not BOXED — but ONLY
                                            // when the actual register value is a pointer (I64).
                                            // For concrete primitives (I32, F64, Bool from Channel<Int>),
                                            // the value must be BOXED, not cast.
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
                                            // Check if register holds a concrete primitive
                                            // (e.g., Channel<Int>.send(42) → reg is I32, not a pointer)
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
                                                // Cast I64 → Ptr(U8) — the I64 is actually a pointer
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
                                                // Box concrete primitive for generic param
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

                                    // If receiver failed to lower, skip this handler
                                    // and let the general fallback chain handle it
                                    if receiver_failed {
                                        // Don't generate a broken call; fall through
                                    } else {
                                        // Get MIR wrapper return type
                                        let mir_return_type = mir_wrapper_sig
                                            .as_ref()
                                            .map(|(_, ret)| ret.clone())
                                            .unwrap_or_else(|| result_type.clone());

                                        // Register forward reference
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

                                        // Auto-unbox: resolve generic T from receiver type args
                                        // e.g., Channel<Int>.tryReceive() returns Ptr(U8) but should produce I32
                                        let resolved_expected = {
                                            let type_table = self.type_table;
                                            // The receiver is args[0] - check its type for generic args
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
                                            // Also check if return type is Optional{primitive} (Null<T>)
                                            // and resolve to the inner primitive for unboxing
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

                                        // Store class hint for the result register to enable
                                        // disambiguation of subsequent method calls on this value.
                                        // E.g., Mutex.lock() returns MutexGuard, so the result
                                        // should be tagged as MutexGuard for .get()/.unlock() dispatch.
                                        if let Some(result_reg) = final_result {
                                            let return_class = Self::get_return_class_hint(
                                                class_name,
                                                method_name,
                                            );
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

                                    // Build param types
                                    let param_types: Vec<_> =
                                        arg_regs.iter().map(|_| ptr_u8.clone()).collect();

                                    // Determine return type: Void if function doesn't return, otherwise ptr
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
                                    // Lower the receiver
                                    let receiver_reg = self.lower_expression(&args[0])?;

                                    // Check if the receiver was boxed by examining its MIR register type.
                                    // Boxing creates a Ptr(U8) value. If the receiver has a different
                                    // pointer type (like Ptr(Void) from a stdlib function return),
                                    // it wasn't boxed and shouldn't be unboxed.
                                    //
                                    // IMPORTANT: If the receiver has a class hint (set by stdlib MIR
                                    // wrapper dispatch), it's a raw class pointer from a method like
                                    // MutexGuard_get — NOT a boxed DynamicValue. Don't unbox it.
                                    let has_class_hint =
                                        self.register_class_hints.contains_key(&receiver_reg);
                                    let receiver_mir_type =
                                        self.builder.get_register_type(receiver_reg);
                                    // Dynamic receivers are always boxed (from haxe_box_reference_ptr),
                                    // even if MIR register type shows Ptr(Void) due to cast.
                                    // Always unbox for Dynamic unless it has a class hint.
                                    let should_unbox = !has_class_hint;

                                    let actual_receiver = if should_unbox {
                                        // Unbox the Dynamic to get the actual object pointer
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

                                    // Lower the rest of arguments (skip receiver at index 0)
                                    let arg_regs: Vec<_> = std::iter::once(actual_receiver)
                                        .chain(
                                            args[1..]
                                                .iter()
                                                .filter_map(|a| self.lower_expression(a)),
                                        )
                                        .collect();

                                    // Get the function's actual return type
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

            // NOTE: MutexGuard method calls are handled through the general stdlib mechanism:
            // 1. Dynamic dispatch uses find_classes_with_method() with dynamic priority
            // 2. MutexGuard is prioritized (return-only type with no constructor)
            // 3. MutexGuard_get MIR wrapper is called via stdlib_mapping

            // NOTE: String method calls are handled through the general stdlib mechanism:
            // 1. get_stdlib_runtime_info() maps TypeKind::String to class name "String"
            // 2. stdlib_mapping lookup finds the correct runtime function
            // 3. The general path handles param types and return types

            // PRIORITY CHECK: For extern generic classes like Vec<T>, the receiver type
            // may be TypeId::MAX (invalid). In this case, try to use the tracked
            // monomorphized class from variable assignment.
            if receiver_type == TypeId::from_raw(u32::MAX) {
                debug!(
                    "[MONO VAR CHECK] receiver_type is MAX, checking monomorphized_var_types ({} entries)",
                    self.monomorphized_var_types.len()
                );

                // Try to extract the SymbolId from the receiver expression
                // The receiver (args[0]) should be a variable reference like HirExprKind::Variable
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
                    // Check if this variable has a tracked monomorphized class
                    if let Some(mono_class) = self.monomorphized_var_types.get(&var_symbol).cloned()
                    {
                        // Get the method name
                        if let Some(method_sym) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(method_sym.name) {
                                debug!(
                                    "[MONO VAR DISPATCH] Found tracked class '{}' for variable {:?}, method '{}'",
                                    mono_class, var_symbol, method_name
                                );

                                // Build the MIR wrapper function name: VecI32_push, VecF64_get, etc.
                                let mir_func_name = format!("{}_{}", mono_class, method_name);

                                // Get the signature from get_stdlib_mir_wrapper_signature
                                if let Some((mir_param_types, mir_return_type)) =
                                    self.get_stdlib_mir_wrapper_signature(&mir_func_name)
                                {
                                    debug!(
                                        "[MONO VAR DISPATCH] Using MIR wrapper: {}",
                                        mir_func_name
                                    );

                                    // Lower all arguments (including receiver)
                                    let mut arg_regs = Vec::new();
                                    for arg in args {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                        }
                                    }

                                    // Register forward reference
                                    let mir_func_id = self.register_stdlib_mir_forward_ref(
                                        &mir_func_name,
                                        mir_param_types.clone(),
                                        mir_return_type.clone(),
                                    );

                                    debug!(
                                        "[MONO VAR DISPATCH] Registered forward ref to {} with ID {:?}",
                                        mir_func_name, mir_func_id
                                    );

                                    // Generate the call
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

            // GUARD: Skip instance method handling if receiver is a Class type itself
            // This can happen when static method calls come through with is_method=true
            // e.g., Thread.spawn(closure) might be seen as Thread(receiver).spawn(closure)
            let receiver_is_class_type = {
                let type_table = self.type_table;
                type_table.get(receiver_type)
                    .map(|ti| {
                        // Check if the type is a class AND matches one of our MIR wrapper classes
                        if let crate::tast::core::TypeKind::Class { symbol_id, .. } = &ti.kind {
                            self.symbol_table.get_symbol(*symbol_id)
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|name| {
                                    // Use dynamic check via stdlib_mapping instead of hardcoded list
                                    let is_mir_wrapper = self.stdlib_mapping.is_mir_wrapper_class(name);
                                    if is_mir_wrapper {
                                        debug!("[GUARD] Receiver type is {} class (MIR wrapper), skipping instance method path", name);
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
            if !receiver_is_class_type {
                // Calculate param_count for overload disambiguation: args[0] is receiver, rest are params
                let method_param_count = if args.len() > 1 { args.len() - 1 } else { 0 };
                {
                    if let Some((class_name, method_name, runtime_call)) = self
                        .get_stdlib_runtime_info(
                            *symbol,
                            receiver_type,
                            Some(method_param_count),
                            None,
                        )
                    {
                        let runtime_func = runtime_call.runtime_name;
                        let ptr_conversion_mask = runtime_call.params_need_ptr_conversion;
                        let raw_value_mask = runtime_call.raw_value_params;
                        let returns_raw_value = runtime_call.returns_raw_value;
                        let extend_i64_mask = runtime_call.extend_to_i64_params;
                        let needs_out_param = runtime_call.needs_out_param;
                        let has_return = runtime_call.has_return; // Copy for use in fallback closure

                        // SPECIAL CASE: Instance methods that need out parameter (like Array.slice, String.split)
                        // These have void return but write result to first out parameter
                        // Generate inline wrapper: allocate + call runtime + return pointer
                        if needs_out_param {
                            debug!(
                                "[OUT PARAM] Instance method {}.{} needs out param inline wrapper",
                                class_name, method_name
                            );

                            // Lower all arguments (receiver + method args)
                            let mut call_arg_regs = Vec::new();
                            for arg in args {
                                if let Some(reg) = self.lower_expression(arg) {
                                    call_arg_regs.push(reg);
                                }
                            }

                            // Allocate space for the result object
                            // For arrays/strings, allocate an opaque pointer-sized value
                            let out_ptr_ty = IrType::Ptr(Box::new(IrType::Void));
                            let out_ptr = self.builder.build_alloc(out_ptr_ty.clone(), None)?;

                            // Register the extern runtime function
                            // Signature: void runtime_func(out: *Ptr(Void), receiver: Ptr(Void), ...params)
                            let mut extern_param_types = vec![out_ptr_ty.clone()]; // out parameter
                            for arg in args {
                                extern_param_types.push(self.convert_type(arg.ty));
                            }

                            let extern_func_id = self.get_or_register_extern_function(
                                runtime_func,
                                extern_param_types,
                                IrType::Void,
                            );

                            // Call runtime function: runtime_func(out_ptr, receiver, ...args)
                            let mut runtime_args = vec![out_ptr];
                            runtime_args.extend(call_arg_regs);

                            self.builder.build_call_direct(
                                extern_func_id,
                                runtime_args,
                                IrType::Void,
                            );

                            // Load the result pointer from the out parameter
                            let result_ptr = self.builder.build_load(out_ptr, out_ptr_ty)?;

                            debug!(
                                "[OUT PARAM] Generated inline wrapper for {}, result_ptr: {:?}",
                                runtime_func, result_ptr
                            );

                            return Some(result_ptr);
                        }

                        // SPECIAL CASE: Check if this is a stdlib MIR wrapper function
                        // MIR wrappers are functions that forward to extern runtime functions.
                        // The wrappers handle calling convention differences and provide default arguments.
                        // NOTE: We check runtime_call.is_mir_wrapper, not just is_mir_wrapper_class(),
                        // because some methods on MIR wrapper classes (e.g., String.split) are
                        // direct extern calls without wrappers.
                        if runtime_call.is_mir_wrapper {
                            // Use the runtime function name from the mapping to handle overloaded methods
                            // For example, String.indexOf can map to String_indexOf (1-arg) or String_indexOf_2 (2-arg)
                            let mir_func_name = runtime_func.to_string();
                            debug!(
                                "[STDLIB MIR] Detected stdlib MIR wrapper function (instance): {}",
                                mir_func_name
                            );

                            // Lower all arguments and collect their types
                            // Auto-box primitive args when MIR wrapper expects Ptr(U8)
                            // (e.g., Channel<Int>.send(42) needs to box the Int)
                            let mir_wrapper_params = self
                                .get_stdlib_mir_wrapper_signature(&mir_func_name)
                                .map(|(params, _)| params);
                            let mut arg_regs = Vec::new();
                            let mut param_types = Vec::new();
                            for (i, arg) in args.iter().enumerate() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    let actual_ty = self.convert_type(arg.ty);
                                    // Check if MIR wrapper expects a different type (e.g., Ptr(U8) for boxed value)
                                    let expected_ty = mir_wrapper_params
                                        .as_ref()
                                        .and_then(|params| params.get(i).cloned())
                                        .unwrap_or_else(|| actual_ty.clone());
                                    // Channel payloads are uniformly boxed (refs too) so
                                    // the erased receive arm can tag-dispatch. i==0 is the
                                    // channel handle — never box it.
                                    let final_reg = if (mir_func_name == "Channel_send"
                                        || mir_func_name == "Channel_trySend")
                                        && i >= 1
                                    {
                                        self.box_channel_payload(
                                            reg,
                                            arg.ty,
                                            &actual_ty,
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
                                }
                            }

                            // SPECIAL: For generic methods that return T (like Thread<T>.join() -> T,
                            // Channel<T>.tryReceive() -> Null<T>), we need to resolve the type parameter
                            // from the receiver's generic arguments.
                            // Also resolve when result_type is Ptr(Void) which comes from Dynamic/unresolved generics.
                            let needs_generic_resolve = result_type == IrType::Any
                                || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                || result_type == IrType::I64;
                            let resolved_result_type = if needs_generic_resolve {
                                // Check if the receiver is a generic class with type parameters
                                let type_table = self.type_table;
                                if let Some(receiver_info) = type_table.get(receiver_type) {
                                    if let crate::tast::TypeKind::Class { type_args, .. } =
                                        &receiver_info.kind
                                    {
                                        // For Thread<T>.join(), type_args[0] is T
                                        if !type_args.is_empty() {
                                            let concrete_type = self.convert_type(type_args[0]);
                                            debug!(
                                            "[GENERIC RESOLVE] Resolved return type from {:?} to {:?}",
                                            result_type, concrete_type
                                        );
                                            concrete_type
                                        } else {
                                            result_type.clone()
                                        }
                                    } else {
                                        result_type.clone()
                                    }
                                } else {
                                    result_type.clone()
                                }
                            } else {
                                result_type.clone()
                            };

                            // Register forward reference - will be provided by merged stdlib module
                            let mir_func_id = self.register_stdlib_mir_forward_ref(
                                &mir_func_name,
                                param_types,
                                resolved_result_type.clone(),
                            );

                            // IMPORTANT: For Void-returning functions, use the function's ACTUAL return type.
                            // For non-void functions, trust resolved_result_type (which handles generics correctly).
                            // This fixes the bug where void functions like Channel.send incorrectly get dest registers.
                            let final_return_type = if let Some(func) =
                                self.builder.module.functions.get(&mir_func_id)
                            {
                                if func.signature.return_type == IrType::Void {
                                    debug!(
                                    "[STDLIB MIR] Function {} returns Void, using actual signature",
                                    mir_func_name
                                );
                                    IrType::Void
                                } else if resolved_result_type == IrType::Any
                                    || matches!(resolved_result_type, IrType::Ptr(ref inner) if **inner == IrType::Void)
                                {
                                    debug!(
                                    "[STDLIB MIR] resolved_result_type is Any/Ptr(Void), using function signature {:?}",
                                    func.signature.return_type
                                );
                                    func.signature.return_type.clone()
                                } else {
                                    debug!(
                                    "[STDLIB MIR] Using resolved_result_type {:?} (handles generics)",
                                    resolved_result_type
                                );
                                    resolved_result_type.clone()
                                }
                            } else {
                                resolved_result_type.clone()
                            };

                            debug!(
                            "[STDLIB MIR] Registered forward ref (instance) to {} with ID {:?}, final return type: {:?}",
                            mir_func_name, mir_func_id, final_return_type
                        );

                            // Generate the call with the MIR wrapper's actual return type
                            // (which may be Ptr(U8) for generic methods returning T)
                            let mir_actual_return = self
                                .get_stdlib_mir_wrapper_signature(&mir_func_name)
                                .map(|(_, ret)| ret)
                                .unwrap_or_else(|| final_return_type.clone());
                            let call_result = self.builder.build_call_direct(
                                mir_func_id,
                                arg_regs,
                                mir_actual_return.clone(),
                            )?;

                            // Auto-unbox if MIR wrapper returns Ptr(U8) but caller expects primitive
                            // (e.g., Channel<Int>.tryReceive() returns boxed int that needs unboxing)
                            let final_result = if mir_func_name == "Channel_receive"
                                || mir_func_name == "Channel_tryReceive"
                            {
                                self.unbox_channel_return(
                                    call_result,
                                    &resolved_result_type,
                                    mir_func_name == "Channel_tryReceive",
                                )
                            } else {
                                self.maybe_unbox_for_extern_return(
                                    call_result,
                                    &mir_actual_return,
                                    &resolved_result_type,
                                )
                            };

                            // Set class hint on the FINAL result register (after potential unboxing)
                            // to enable disambiguation of subsequent method calls.
                            // E.g., Array.iterator() returns ArrayIterator, so subsequent
                            // .hasNext()/.next() calls dispatch to ArrayIterator methods.
                            if let Some(result_reg) = final_result {
                                let return_class =
                                    Self::get_return_class_hint(class_name, method_name);
                                self.register_class_hints
                                    .insert(result_reg, return_class.to_string());
                            }

                            return final_result;
                        }

                        // println!(
                        //     "✅ Generating runtime call to {} (receiver type path)",
                        //     runtime_func
                        // );

                        // Lower all arguments
                        let arg_regs: Vec<_> = args
                            .iter()
                            .filter_map(|a| self.lower_expression(a))
                            .collect();

                        // Apply raw value conversion for high-performance inline storage (StringMap, IntMap)
                        // Values are cast to u64 raw bits - no boxing, no heap allocation
                        let mut final_arg_regs = arg_regs.clone();
                        if raw_value_mask != 0 {
                            for i in 0..arg_regs.len() {
                                if (raw_value_mask & (1 << i)) != 0 {
                                    let arg_reg = arg_regs[i];
                                    let arg_type = self
                                        .builder
                                        .get_register_type(arg_reg)
                                        .unwrap_or(IrType::I64);

                                    // Cast value to U64 raw bits - zero-cost for same-size types
                                    let raw_reg = match &arg_type {
                                        IrType::I32 => {
                                            // Zero-extend i32 to u64
                                            self.builder.build_cast(
                                                arg_reg,
                                                IrType::I32,
                                                IrType::U64,
                                            )
                                        }
                                        IrType::I64 => {
                                            // Reinterpret i64 as u64 (same bits) - use cast
                                            self.builder.build_cast(
                                                arg_reg,
                                                IrType::I64,
                                                IrType::U64,
                                            )
                                        }
                                        IrType::F64 => {
                                            // Reinterpret f64 bits as u64 - use BitCast instruction
                                            self.builder.build_bitcast(arg_reg, IrType::U64)
                                        }
                                        IrType::F32 => {
                                            // Extend f32 to f64, then reinterpret as u64
                                            let f64_reg = self
                                                .builder
                                                .build_cast(arg_reg, IrType::F32, IrType::F64)
                                                .unwrap_or(arg_reg);
                                            self.builder.build_bitcast(f64_reg, IrType::U64)
                                        }
                                        IrType::Bool => {
                                            // Zero-extend bool to u64
                                            self.builder.build_cast(
                                                arg_reg,
                                                IrType::Bool,
                                                IrType::U64,
                                            )
                                        }
                                        IrType::Ptr(_) => {
                                            // Pointer to u64 (address as integer)
                                            self.builder.build_cast(
                                                arg_reg,
                                                arg_type.clone(),
                                                IrType::U64,
                                            )
                                        }
                                        _ => {
                                            // For other types, try direct cast to U64
                                            self.builder.build_cast(
                                                arg_reg,
                                                arg_type.clone(),
                                                IrType::U64,
                                            )
                                        }
                                    };

                                    if let Some(raw) = raw_reg {
                                        final_arg_regs[i] = raw;
                                    }
                                }
                            }
                        }
                        // Apply pointer conversion for parameters that need it (DEPRECATED - use raw_value_params)
                        // This creates boxed Dynamic values for legacy runtime functions.
                        else if ptr_conversion_mask != 0 {
                            for i in 0..arg_regs.len() {
                                // Check if bit i is set in the mask
                                if (ptr_conversion_mask & (1 << i)) != 0 {
                                    let arg_reg = arg_regs[i];
                                    let arg_type = self
                                        .builder
                                        .get_register_type(arg_reg)
                                        .unwrap_or(IrType::I64);

                                    // Use proper Dynamic boxing based on the argument type
                                    // This creates a tagged Dynamic value that can be unboxed later
                                    // Use the haxe_box_*_ptr wrapper functions which handle type conversion internally
                                    let boxed_reg = match &arg_type {
                                        IrType::I32 => {
                                            // Box int using haxe_box_int_ptr wrapper (which handles i32->i64 cast)
                                            let box_func = self.get_or_register_extern_function(
                                                "haxe_box_int_ptr",
                                                vec![IrType::I32],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            );
                                            self.builder.build_call_direct(
                                                box_func,
                                                vec![arg_reg],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            )
                                        }
                                        IrType::I64 => {
                                            // Box int64 - truncate to i32 and use haxe_box_int_ptr wrapper
                                            let truncated = self
                                                .builder
                                                .build_cast(arg_reg, IrType::I64, IrType::I32)
                                                .unwrap_or(arg_reg);
                                            let box_func = self.get_or_register_extern_function(
                                                "haxe_box_int_ptr",
                                                vec![IrType::I32],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            );
                                            self.builder.build_call_direct(
                                                box_func,
                                                vec![truncated],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            )
                                        }
                                        IrType::F32 | IrType::F64 => {
                                            // Box float using haxe_box_float_ptr wrapper
                                            let float_val = if arg_type == IrType::F32 {
                                                self.builder
                                                    .build_cast(arg_reg, IrType::F32, IrType::F64)
                                                    .unwrap_or(arg_reg)
                                            } else {
                                                arg_reg
                                            };
                                            let box_func = self.get_or_register_extern_function(
                                                "haxe_box_float_ptr",
                                                vec![IrType::F64],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            );
                                            self.builder.build_call_direct(
                                                box_func,
                                                vec![float_val],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            )
                                        }
                                        IrType::Bool => {
                                            // Box bool using haxe_box_bool_ptr wrapper
                                            let box_func = self.get_or_register_extern_function(
                                                "haxe_box_bool_ptr",
                                                vec![IrType::Bool],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            );
                                            self.builder.build_call_direct(
                                                box_func,
                                                vec![arg_reg],
                                                IrType::Ptr(Box::new(IrType::U8)),
                                            )
                                        }
                                        IrType::Ptr(_) | IrType::Struct { .. } => {
                                            // Pointer/reference types still need stack allocation for ptr_params
                                            // because the runtime function expects a pointer TO the value,
                                            // and the value itself is a pointer we need to pass BY REFERENCE.
                                            // Example: haxe_array_push(arr, data) where data = &value
                                            // For Array<Thread>, value is a pointer, so data = &pointer
                                            if let Some(stack_slot) =
                                                self.builder.build_alloc(arg_type.clone(), None)
                                            {
                                                self.builder.build_store(stack_slot, arg_reg);
                                                Some(stack_slot)
                                            } else {
                                                Some(arg_reg)
                                            }
                                        }
                                        _ => {
                                            // For other types, fallback to stack allocation
                                            // (This preserves the old behavior for edge cases)
                                            if let Some(stack_slot) =
                                                self.builder.build_alloc(arg_type.clone(), None)
                                            {
                                                self.builder.build_store(stack_slot, arg_reg);
                                                Some(stack_slot)
                                            } else {
                                                Some(arg_reg)
                                            }
                                        }
                                    };

                                    if let Some(boxed) = boxed_reg {
                                        final_arg_regs[i] = boxed;
                                    }
                                }
                            }
                        }

                        // Apply i32 -> i64 extension for IntMap key parameters
                        // This is needed because Haxe Int is 32-bit but the runtime uses 64-bit keys
                        if extend_i64_mask != 0 {
                            for i in 0..final_arg_regs.len() {
                                if (extend_i64_mask & (1 << i)) != 0 {
                                    let arg_reg = final_arg_regs[i];
                                    let arg_type = self
                                        .builder
                                        .get_register_type(arg_reg)
                                        .unwrap_or(IrType::I32);

                                    // Only extend i32 to i64, skip if already i64
                                    if arg_type == IrType::I32 {
                                        if let Some(extended) = self.builder.build_cast(
                                            arg_reg,
                                            IrType::I32,
                                            IrType::I64,
                                        ) {
                                            final_arg_regs[i] = extended;
                                        }
                                    }
                                }
                            }
                        }

                        // Get or register the extern runtime function
                        // Use actual argument types from TAST, applying type conversion where needed
                        let param_types: Vec<IrType> = args
                            .iter()
                            .enumerate()
                            .map(|(i, arg)| {
                                // Raw value params are passed as U64 (high-performance inline storage)
                                if raw_value_mask != 0 && (raw_value_mask & (1 << i)) != 0 {
                                    IrType::U64
                                }
                                // Extended i64 params need i64 type in signature
                                else if extend_i64_mask != 0 && (extend_i64_mask & (1 << i)) != 0
                                {
                                    IrType::I64
                                }
                                // Legacy ptr_conversion params are passed as Ptr (boxed Dynamic)
                                else if ptr_conversion_mask != 0
                                    && (ptr_conversion_mask & (1 << i)) != 0
                                {
                                    IrType::Ptr(Box::new(IrType::U8))
                                } else {
                                    self.convert_type(arg.ty)
                                }
                            })
                            .collect();

                        // For functions that return raw values (u64), we need to:
                        // 1. Resolve the actual type parameter T from the receiver's generic args
                        // 2. Call with U64 return type
                        // 3. Cast the result to the resolved type
                        //
                        // `resolved_from_type_args` records whether the resolution
                        // came from a real receiver type_arg substitution (true) or
                        // fell through to `result_type` because there were no type
                        // args. The U64 → Ptr post-cast uses this to decide whether
                        // bitcasting a Ptr-typed result is safe — bitcasting a Ptr
                        // when the source u64 actually holds an Int (because T was
                        // unresolved) would silently produce a garbage address.
                        let (resolved_return_type, resolved_from_type_args) = if returns_raw_value {
                            // Resolve value T from receiver's type args. The
                            // LAST type arg is V — covers both single-param
                            // (StringMap<T>, IntMap<T>) and two-param
                            // (ObjectMap<K, V>) container shapes.
                            let type_table = self.type_table;
                            let resolved = if let Some(receiver_info) =
                                type_table.get(receiver_type)
                            {
                                match &receiver_info.kind {
                                    crate::tast::TypeKind::Class { type_args, .. }
                                    | crate::tast::TypeKind::GenericInstance {
                                        type_args, ..
                                    } => type_args.last().map(|ta| self.convert_type(*ta)),
                                    _ => None,
                                }
                            } else {
                                None
                            };
                            match resolved {
                                Some(t) => (t, true),
                                None => (result_type.clone(), false),
                            }
                        } else {
                            // IMPORTANT: For MIR wrappers, use their actual return type instead of HIR type
                            // HIR type may be Dynamic/Ptr(Void) but the wrapper returns a concrete type (e.g., Bool)
                            let ret = self
                                .get_stdlib_mir_wrapper_signature(&runtime_func)
                                .map(|(_, ret_ty)| ret_ty)
                                .unwrap_or_else(|| {
                                    if has_return {
                                        result_type.clone()
                                    } else {
                                        IrType::Void
                                    }
                                });
                            (ret, false)
                        };
                        debug!(
                        "[RESOLVED RETURN TYPE] runtime_func={}, result_type={:?}, resolved={:?}",
                        runtime_func, result_type, resolved_return_type
                    );

                        let call_return_type = if returns_raw_value {
                            IrType::U64
                        } else {
                            resolved_return_type.clone()
                        };

                        let runtime_func_id = self.get_or_register_extern_function(
                            &runtime_func,
                            param_types,
                            call_return_type.clone(),
                        );

                        // Generate the call to the runtime function
                        let call_result = self.builder.build_call_direct(
                            runtime_func_id,
                            final_arg_regs,
                            call_return_type,
                        );

                        // If this returns raw value, cast U64 back to the resolved type parameter
                        if returns_raw_value {
                            if let Some(raw_reg) = call_result {
                                // Cast U64 to the resolved type parameter
                                let final_result = match &resolved_return_type {
                                    IrType::I32 => {
                                        self.builder.build_cast(raw_reg, IrType::U64, IrType::I32)
                                    }
                                    IrType::I64 => {
                                        self.builder.build_cast(raw_reg, IrType::U64, IrType::I64)
                                    }
                                    IrType::F64 => self.builder.build_bitcast(raw_reg, IrType::F64),
                                    IrType::F32 => {
                                        // Bitcast to F64, then convert to F32
                                        if let Some(f64_reg) =
                                            self.builder.build_bitcast(raw_reg, IrType::F64)
                                        {
                                            self.builder.build_cast(
                                                f64_reg,
                                                IrType::F64,
                                                IrType::F32,
                                            )
                                        } else {
                                            None
                                        }
                                    }
                                    IrType::Bool => {
                                        self.builder.build_cast(raw_reg, IrType::U64, IrType::Bool)
                                    }
                                    IrType::Ptr(_) => {
                                        // Pointer type — bit-reinterpret u64 → ptr,
                                        // BUT only when we actually resolved T from
                                        // the receiver's type_args. When T was
                                        // unresolved (e.g. `new StringMap()` with
                                        // no type args), the user is storing
                                        // primitives as raw bits — bitcasting to
                                        // Ptr would silently mangle them. The
                                        // legacy U64→I64 cast keeps the bits as
                                        // an integer for downstream unboxing.
                                        if resolved_from_type_args {
                                            self.builder.build_bitcast(
                                                raw_reg,
                                                resolved_return_type.clone(),
                                            )
                                        } else {
                                            self.builder.build_cast(
                                                raw_reg,
                                                IrType::U64,
                                                IrType::I64,
                                            )
                                        }
                                    }
                                    _ => {
                                        // Truly unresolved T (Dynamic, type parameter)
                                        // — keep as I64 so the raw value isn't
                                        // misinterpreted as anything else.
                                        self.builder.build_cast(raw_reg, IrType::U64, IrType::I64)
                                    }
                                };
                                return final_result;
                            }
                        }

                        return call_result;
                    }

                    // GUARD: Check if receiver is a user-defined class (not stdlib)
                    // If so, skip all stdlib fallbacks - they would incorrectly match stdlib methods
                    let receiver_is_user_class = {
                        let type_table = self.type_table;
                        type_table
                            .get(receiver_type)
                            .map(|ti| {
                                match &ti.kind {
                                    crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                                        // Check if this is a stdlib class
                                        self.symbol_table
                                            .get_symbol(*symbol_id)
                                            .map(|s| !self.is_stdlib_class_by_symbol(s))
                                            .unwrap_or(false)
                                    }
                                    // TypeParameter receivers always come from user-defined generics.
                                    // Method calls on T should resolve through function_map, not stdlib.
                                    // (Constrained T:Interface is handled earlier by interface dispatch.)
                                    crate::tast::core::TypeKind::TypeParameter { .. } => true,
                                    // GenericInstance: check if the base type is a user class
                                    crate::tast::core::TypeKind::GenericInstance {
                                        base_type,
                                        ..
                                    } => type_table
                                        .get(*base_type)
                                        .map(|bt| {
                                            if let crate::tast::core::TypeKind::Class {
                                                symbol_id,
                                                ..
                                            } = &bt.kind
                                            {
                                                self.symbol_table
                                                    .get_symbol(*symbol_id)
                                                    .map(|s| !self.is_stdlib_class_by_symbol(s))
                                                    .unwrap_or(false)
                                            } else {
                                                false
                                            }
                                        })
                                        .unwrap_or(false),
                                    // Abstract types with user-defined methods
                                    crate::tast::core::TypeKind::Abstract { symbol_id, .. } => self
                                        .symbol_table
                                        .get_symbol(*symbol_id)
                                        .map(|s| !self.is_stdlib_class_by_symbol(s))
                                        .unwrap_or(false),
                                    _ => false,
                                }
                            })
                            .unwrap_or(false)
                    };

                    // Skip stdlib fallbacks for user-defined classes
                    if receiver_is_user_class {
                        // For user-defined classes, the method should be in function_map
                        // Don't try to match stdlib methods
                    } else {
                        // Fallback: Use stdlib mapping to try all possible class/method combinations
                        // This is necessary when qualified names aren't set properly
                        if let Some(method_sym) = self.symbol_table.get_symbol(*symbol) {
                            if let Some(method_name) = self.string_interner.get(method_sym.name) {
                                let static_args = self.effective_static_call_args(args);
                                // First try to use the qualified name if available
                                if let Some(qual_name) = method_sym
                                    .qualified_name
                                    .and_then(|qn| self.string_interner.get(qn))
                                {
                                    if let Some(runtime_func) = self
                                        .get_static_stdlib_runtime_func_with_params(
                                            qual_name,
                                            method_name,
                                            static_args.len(),
                                        )
                                    {
                                        // CHECK: Is this a MIR wrapper function or a true extern?
                                        // The mapping's `is_mir_wrapper` flag decides — having
                                        // explicit type info does NOT (typed extern intrinsics
                                        // like `haxe_bytes_get` carry signatures too; routing
                                        // them here creates a body-less forward-ref stub that
                                        // traps at runtime).
                                        if let Some((_mir_param_types, _mir_return_type)) = self
                                            .get_stdlib_mir_wrapper_signature(runtime_func)
                                            .filter(|_| {
                                                self.stdlib_mapping
                                                    .is_mir_wrapper_function(runtime_func)
                                            })
                                        {
                                            debug!(
                                                "[QUALIFIED NAME PATH] Detected MIR wrapper: {}",
                                                runtime_func
                                            );

                                            // Lower all arguments and collect their types
                                            let mut arg_regs = Vec::new();
                                            let mut param_types = Vec::new();
                                            for arg in static_args {
                                                if let Some(reg) = self.lower_expression(arg) {
                                                    arg_regs.push(reg);
                                                    param_types.push(self.convert_type(arg.ty));
                                                }
                                            }

                                            // Register forward reference - will be provided by merged stdlib module
                                            let mir_func_id = self.register_stdlib_mir_forward_ref(
                                                runtime_func,
                                                param_types,
                                                result_type.clone(),
                                            );

                                            debug!(
                                            "[QUALIFIED NAME PATH] Registered forward ref to {} with ID {:?}",
                                            runtime_func, mir_func_id
                                        );

                                            // Generate the call
                                            let result = self.builder.build_call_direct(
                                                mir_func_id,
                                                arg_regs,
                                                result_type,
                                            );
                                            debug!(
                                            "[QUALIFIED NAME PATH] Generated call, result: {:?}",
                                            result
                                        );
                                            return result;
                                        }

                                        // Lower all arguments
                                        let arg_regs: Vec<_> = static_args
                                            .iter()
                                            .filter_map(|a| self.lower_expression(a))
                                            .collect();

                                        // Get expected types FIRST so we can auto-box before ptr_conversion
                                        let (expected_param_types_qn, expected_return_type_qn) =
                                            self.get_extern_function_signature(&runtime_func)
                                                .unwrap_or_else(|| {
                                                    let param_types: Vec<IrType> = static_args
                                                        .iter()
                                                        .map(|arg| self.convert_type(arg.ty))
                                                        .collect();
                                                    (param_types, result_type.clone())
                                                });

                                        // Auto-box arguments when expected type is Ptr(U8) (Dynamic)
                                        let mut final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                        .map(|(i, &reg)| {
                                            if let (Some(expected_ty), Some(actual_ty)) = (
                                                expected_param_types_qn.get(i),
                                                self.builder.get_register_type(reg)
                                            ) {
                                                if *expected_ty != actual_ty {
                                                    let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                    if is_ptr_u8 && i < static_args.len() {
                                                        if let Some(boxed) = self.box_value_for_dynamic(reg, static_args[i].ty) {
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

                                        let runtime_func_id_qn = self
                                            .get_or_register_extern_function(
                                                &runtime_func,
                                                expected_param_types_qn,
                                                expected_return_type_qn.clone(),
                                            );

                                        // Generate the call to the runtime function
                                        let call_result_qn = self.builder.build_call_direct(
                                            runtime_func_id_qn,
                                            final_arg_regs,
                                            expected_return_type_qn.clone(),
                                        )?;
                                        return Some(self.reconcile_extern_return(
                                            call_result_qn,
                                            &expected_return_type_qn,
                                            &result_type,
                                        ));

                                        // DEAD CODE below (kept for reference): old ptr_conversion path
                                        #[allow(unreachable_code)]
                                        let _unused_final_arg_regs = arg_regs.clone();
                                        #[allow(unreachable_code)]
                                        let mut final_arg_regs = arg_regs.clone();
                                        let ptr_conversion_mask = self
                                            .stdlib_mapping
                                            .find_by_runtime_name(&runtime_func)
                                            .map(|m| m.params_need_ptr_conversion)
                                            .unwrap_or(0);
                                        if ptr_conversion_mask != 0 {
                                            for i in 0..arg_regs.len() {
                                                // Check if bit i is set in the mask
                                                if (ptr_conversion_mask & (1 << i)) != 0 {
                                                    let arg_reg = arg_regs[i];
                                                    // Default to I64 (pointer-sized) if type is unknown.
                                                    // This is safer than I32 since pointers and most values are 64-bit.
                                                    let arg_type = self
                                                        .builder
                                                        .get_register_type(arg_reg)
                                                        .unwrap_or(IrType::I64);

                                                    // For array operations, always allocate 8 bytes (elem_size is always 8)
                                                    // and extend smaller values to 64-bit
                                                    let (alloc_type, value_to_store) =
                                                        match arg_type {
                                                            IrType::I32 => {
                                                                let ext_val =
                                                                    self.builder.build_cast(
                                                                        arg_reg,
                                                                        IrType::I32,
                                                                        IrType::I64,
                                                                    );
                                                                (
                                                                    IrType::I64,
                                                                    ext_val.unwrap_or(arg_reg),
                                                                )
                                                            }
                                                            IrType::F32 => {
                                                                let ext_val =
                                                                    self.builder.build_cast(
                                                                        arg_reg,
                                                                        IrType::F32,
                                                                        IrType::F64,
                                                                    );
                                                                (
                                                                    IrType::F64,
                                                                    ext_val.unwrap_or(arg_reg),
                                                                )
                                                            }
                                                            _ => (arg_type.clone(), arg_reg),
                                                        };

                                                    // Allocate stack space and pass a pointer to the value.
                                                    if let Some(stack_slot) = self
                                                        .builder
                                                        .build_alloc(alloc_type.clone(), None)
                                                    {
                                                        // Store the value into the stack slot
                                                        self.builder.build_store(
                                                            stack_slot,
                                                            value_to_store,
                                                        );
                                                        // Use the pointer for the call
                                                        final_arg_regs[i] = stack_slot;
                                                    }
                                                }
                                            }
                                        }

                                        // Use the function signature from the mapping (hlp_* introspection)
                                        // if available; this is the authoritative source of type info.
                                        let (expected_param_types, expected_return_type) = self
                                            .get_extern_function_signature(&runtime_func)
                                            .unwrap_or_else(|| {
                                                let param_types: Vec<IrType> = args
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(i, arg)| {
                                                        if ptr_conversion_mask != 0
                                                            && (ptr_conversion_mask & (1 << i)) != 0
                                                        {
                                                            IrType::Ptr(Box::new(IrType::U8))
                                                        } else {
                                                            self.convert_type(arg.ty)
                                                        }
                                                    })
                                                    .collect();
                                                (param_types, result_type.clone())
                                            });
                                        let runtime_func_id = self.get_or_register_extern_function(
                                            &runtime_func,
                                            expected_param_types,
                                            expected_return_type.clone(),
                                        );

                                        // Generate the call to the runtime function
                                        return self.builder.build_call_direct(
                                            runtime_func_id,
                                            final_arg_regs,
                                            expected_return_type,
                                        );
                                    }
                                }

                                // Fallback: try each possible stdlib class (only if qualified name didn't work)
                                // For static methods like Arc.init, Mutex.init, etc, try to infer the class from the return type
                                // debug!("Qualified name not available, trying to infer class from return type={:?}", expr.ty);

                                let inferred_class = {
                                    let type_table = self.type_table;
                                    debug!(
                                        "[INFER CLASS] Checking return type expr.ty={:?}",
                                        expr.ty
                                    );
                                    if let Some(type_info) = type_table.get(expr.ty) {
                                        debug!(
                                            "[INFER CLASS] Return type kind={:?}",
                                            type_info.kind
                                        );
                                        if let TypeKind::Class { symbol_id, .. } = &type_info.kind {
                                            if let Some(class_sym) =
                                                self.symbol_table.get_symbol(*symbol_id)
                                            {
                                                let class_name =
                                                    self.string_interner.get(class_sym.name);
                                                debug!(
                                                "[INFER CLASS] Inferred class from return type: {:?}",
                                                class_name
                                            );
                                                class_name
                                            } else {
                                                debug!("[INFER CLASS] Class symbol not found");
                                                None
                                            }
                                        } else {
                                            debug!("[INFER CLASS] Return type is not a class");
                                            None
                                        }
                                    } else {
                                        debug!(
                                            "[INFER CLASS] Type info not found for expr.ty={:?}",
                                            expr.ty
                                        );
                                        None
                                    }
                                };

                                if let Some(class_name) = inferred_class {
                                    // SPECIAL CASE: Check if this is a stdlib MIR function
                                    if self.stdlib_mapping.is_mir_wrapper_class(class_name) {
                                        // The mapping is the source of truth for the
                                        // wrapper's name — synthesizing it by the
                                        // `{class.lowercase()}_{method}` convention
                                        // produces a body-less stub whenever the real
                                        // entry differs (`QTensor_requantQ6KToQ4KM` vs
                                        // `qtensor_requantQ6KToQ4KM` → trap at call).
                                        // The class here was inferred from the RETURN
                                        // type, which differs from the declaring class
                                        // for non-factory methods (QTensor.gatherRowsQ6K
                                        // returns Tensor) — a globally unique method
                                        // name still identifies the entry.
                                        let mir_func_name = self
                                            .stdlib_mapping
                                            .find_by_name(class_name, method_name)
                                            .or_else(|| {
                                                self.stdlib_mapping
                                                    .find_unique_by_method(method_name)
                                            })
                                            .map(|(_, call)| call.runtime_name.to_string())
                                            .unwrap_or_else(|| {
                                                format!(
                                                    "{}_{}",
                                                    class_name.to_lowercase(),
                                                    method_name
                                                )
                                            });
                                        debug!(
                                            "[STDLIB MIR] Detected stdlib MIR function: {}",
                                            mir_func_name
                                        );

                                        // Lower all arguments and collect their types
                                        let mut arg_regs = Vec::new();
                                        let mut param_types = Vec::new();
                                        for arg in static_args {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                arg_regs.push(reg);
                                                param_types.push(self.convert_type(arg.ty));
                                            }
                                        }

                                        // Register forward reference - will be provided by merged stdlib module
                                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                                            &mir_func_name,
                                            param_types,
                                            result_type.clone(),
                                        );

                                        debug!(
                                        "[STDLIB MIR] Registered forward ref to {} with ID {:?}",
                                        mir_func_name, mir_func_id
                                    );

                                        // Generate the call
                                        let result = self.builder.build_call_direct(
                                            mir_func_id,
                                            arg_regs,
                                            result_type,
                                        );
                                        debug!("[STDLIB MIR] Generated call, result: {:?}", result);
                                        return result;
                                    }

                                    // Try the inferred class first
                                    let fake_qual_name =
                                        format!("rayzor.concurrent.{}.{}", class_name, method_name);
                                    if let Some(runtime_func) = self
                                        .get_static_stdlib_runtime_func_with_params(
                                            &fake_qual_name,
                                            method_name,
                                            static_args.len(),
                                        )
                                    {
                                        debug!(
                                        "[INFERRED CLASS PATH] Got runtime_func='{}' for class={}, method={}",
                                        runtime_func, class_name, method_name
                                    );
                                        // println!("✅ Generating runtime call to {} for {}.{} (inferred from return type)", runtime_func, class_name, method_name);

                                        // Lower all arguments
                                        let arg_regs: Vec<_> = static_args
                                            .iter()
                                            .filter_map(|a| self.lower_expression(a))
                                            .collect();

                                        // Apply pointer conversion for parameters that need it (metadata-driven)
                                        // Look up the RuntimeFunctionCall metadata by runtime function name
                                        // This means the runtime function expects a POINTER TO the value, not the value directly.
                                        let mut final_arg_regs = arg_regs.clone();
                                        let ptr_conversion_mask = self
                                            .stdlib_mapping
                                            .find_by_runtime_name(&runtime_func)
                                            .map(|m| m.params_need_ptr_conversion)
                                            .unwrap_or(0);
                                        if ptr_conversion_mask != 0 {
                                            for i in 0..arg_regs.len() {
                                                // Check if bit i is set in the mask
                                                if (ptr_conversion_mask & (1 << i)) != 0 {
                                                    let arg_reg = arg_regs[i];
                                                    // Default to I64 (pointer-sized) if type is unknown.
                                                    // This is safer than I32 since pointers and most values are 64-bit.
                                                    let arg_type = self
                                                        .builder
                                                        .get_register_type(arg_reg)
                                                        .unwrap_or(IrType::I64);

                                                    // For array operations, always allocate 8 bytes (elem_size is always 8)
                                                    // and extend smaller values to 64-bit
                                                    let (alloc_type, value_to_store) =
                                                        match arg_type {
                                                            IrType::I32 => {
                                                                let ext_val =
                                                                    self.builder.build_cast(
                                                                        arg_reg,
                                                                        IrType::I32,
                                                                        IrType::I64,
                                                                    );
                                                                (
                                                                    IrType::I64,
                                                                    ext_val.unwrap_or(arg_reg),
                                                                )
                                                            }
                                                            IrType::F32 => {
                                                                let ext_val =
                                                                    self.builder.build_cast(
                                                                        arg_reg,
                                                                        IrType::F32,
                                                                        IrType::F64,
                                                                    );
                                                                (
                                                                    IrType::F64,
                                                                    ext_val.unwrap_or(arg_reg),
                                                                )
                                                            }
                                                            _ => (arg_type.clone(), arg_reg),
                                                        };

                                                    // Allocate stack space and pass a pointer to the value.
                                                    if let Some(stack_slot) = self
                                                        .builder
                                                        .build_alloc(alloc_type.clone(), None)
                                                    {
                                                        // Store the value into the stack slot
                                                        self.builder.build_store(
                                                            stack_slot,
                                                            value_to_store,
                                                        );
                                                        // Use the pointer for the call
                                                        final_arg_regs[i] = stack_slot;
                                                    }
                                                }
                                            }
                                        }

                                        // Use the function signature from the mapping (hlp_* introspection)
                                        // if available; this is the authoritative source of type info.
                                        let (expected_param_types, expected_return_type) = self
                                            .get_extern_function_signature(&runtime_func)
                                            .unwrap_or_else(|| {
                                                let param_types: Vec<IrType> = args
                                                    .iter()
                                                    .enumerate()
                                                    .map(|(i, arg)| {
                                                        if ptr_conversion_mask != 0
                                                            && (ptr_conversion_mask & (1 << i)) != 0
                                                        {
                                                            IrType::Ptr(Box::new(IrType::U8))
                                                        } else {
                                                            self.convert_type(arg.ty)
                                                        }
                                                    })
                                                    .collect();
                                                (param_types, result_type.clone())
                                            });
                                        let runtime_func_id = self.get_or_register_extern_function(
                                            &runtime_func,
                                            expected_param_types,
                                            expected_return_type.clone(),
                                        );

                                        // Generate the call to the runtime function
                                        return self.builder.build_call_direct(
                                            runtime_func_id,
                                            final_arg_regs,
                                            expected_return_type,
                                        );
                                    }
                                }

                                // Last resort: try all stdlib classes with param count matching
                                // NOTE: We must match by param count to disambiguate overloaded methods
                                // (e.g., Array.join(sep) with 1 param vs Thread.join() with 0 params)
                                let actual_arg_count = args.len().saturating_sub(1); // Subtract 1 for receiver (self)
                                debug!(
                                "[LAST RESORT] Could not infer class for method '{}' with {} args, trying all stdlib classes",
                                method_name, actual_arg_count
                            );
                                // Get all stdlib classes dynamically from the mapping
                                // NOTE: We do NOT add stdlib MIR detection here because we don't know which class
                                // to use - the fallback tries all classes and would match the wrong one
                                let stdlib_classes = self.stdlib_mapping.get_all_classes();
                                for class_name in &stdlib_classes {
                                    // Use find_by_name_and_params to ensure param count matches
                                    // This prevents Array.join(1 param) from matching Thread.join(0 params)
                                    if let Some((sig, mapping)) =
                                        self.stdlib_mapping.find_by_name_and_params(
                                            class_name,
                                            method_name,
                                            actual_arg_count,
                                        )
                                    {
                                        let runtime_func = mapping.runtime_name;

                                        // CHECK: Is this a MIR wrapper or an extern?
                                        // Gate on the mapping's `is_mir_wrapper` flag —
                                        // typed extern intrinsics carry signatures too,
                                        // and a forward-ref stub for one never gets a
                                        // body (traps at runtime).
                                        if let Some((mir_param_types, mir_return_type)) = self
                                            .get_stdlib_mir_wrapper_signature(&runtime_func)
                                            .filter(|_| mapping.is_mir_wrapper)
                                        {
                                            debug!(
                                                "[FALLBACK PATH] Detected MIR wrapper: {}",
                                                runtime_func
                                            );

                                            // Lower all arguments
                                            let mut arg_regs = Vec::new();
                                            for arg in args {
                                                if let Some(reg) = self.lower_expression(arg) {
                                                    arg_regs.push(reg);
                                                }
                                            }

                                            // Register forward reference - signature comes from get_stdlib_mir_wrapper_signature
                                            let mir_func_id = self.register_stdlib_mir_forward_ref(
                                                &runtime_func,
                                                mir_param_types,
                                                mir_return_type,
                                            );

                                            debug!(
                                            "[FALLBACK PATH] Registered forward ref to {} with ID {:?}",
                                            runtime_func, mir_func_id
                                        );

                                            // Generate the call
                                            let result = self.builder.build_call_direct(
                                                mir_func_id,
                                                arg_regs,
                                                result_type,
                                            );
                                            debug!(
                                                "[FALLBACK PATH] Generated call, result: {:?}",
                                                result
                                            );
                                            return result;
                                        }

                                        // Lower all arguments
                                        let arg_regs: Vec<_> = args
                                            .iter()
                                            .filter_map(|a| self.lower_expression(a))
                                            .collect();

                                        // Apply pointer conversion for parameters that need it (metadata-driven)
                                        // Look up the RuntimeFunctionCall metadata by runtime function name
                                        // This means the runtime function expects a POINTER TO the value, not the value directly.
                                        let mut final_arg_regs = arg_regs.clone();
                                        let ptr_conversion_mask = self
                                            .stdlib_mapping
                                            .find_by_runtime_name(&runtime_func)
                                            .map(|m| m.params_need_ptr_conversion)
                                            .unwrap_or(0);
                                        if ptr_conversion_mask != 0 {
                                            for i in 0..arg_regs.len() {
                                                // Check if bit i is set in the mask
                                                if (ptr_conversion_mask & (1 << i)) != 0 {
                                                    let arg_reg = arg_regs[i];
                                                    // Default to I64 (pointer-sized) if type is unknown.
                                                    // This is safer than I32 since pointers and most values are 64-bit.
                                                    let arg_type = self
                                                        .builder
                                                        .get_register_type(arg_reg)
                                                        .unwrap_or(IrType::I64);

                                                    // For array operations, always allocate 8 bytes (elem_size is always 8)
                                                    // and extend smaller values to 64-bit
                                                    let (alloc_type, value_to_store) =
                                                        match arg_type {
                                                            IrType::I32 => {
                                                                let ext_val =
                                                                    self.builder.build_cast(
                                                                        arg_reg,
                                                                        IrType::I32,
                                                                        IrType::I64,
                                                                    );
                                                                (
                                                                    IrType::I64,
                                                                    ext_val.unwrap_or(arg_reg),
                                                                )
                                                            }
                                                            IrType::F32 => {
                                                                let ext_val =
                                                                    self.builder.build_cast(
                                                                        arg_reg,
                                                                        IrType::F32,
                                                                        IrType::F64,
                                                                    );
                                                                (
                                                                    IrType::F64,
                                                                    ext_val.unwrap_or(arg_reg),
                                                                )
                                                            }
                                                            _ => (arg_type.clone(), arg_reg),
                                                        };

                                                    // Allocate stack space and pass a pointer to the value.
                                                    if let Some(stack_slot) = self
                                                        .builder
                                                        .build_alloc(alloc_type.clone(), None)
                                                    {
                                                        // Store the value into the stack slot
                                                        self.builder.build_store(
                                                            stack_slot,
                                                            value_to_store,
                                                        );
                                                        // Use the pointer for the call
                                                        final_arg_regs[i] = stack_slot;
                                                    }
                                                }
                                            }
                                        }

                                        // Get or register the extern runtime function
                                        // Use actual argument types from TAST, applying ptr conversion where needed
                                        let param_types: Vec<IrType> = args
                                            .iter()
                                            .enumerate()
                                            .map(|(i, arg)| {
                                                // If this param was converted to a pointer, the type is Ptr
                                                if ptr_conversion_mask != 0
                                                    && (ptr_conversion_mask & (1 << i)) != 0
                                                {
                                                    IrType::Ptr(Box::new(IrType::U8))
                                                } else {
                                                    self.convert_type(arg.ty)
                                                }
                                            })
                                            .collect();
                                        let runtime_func_id = self.get_or_register_extern_function(
                                            &runtime_func,
                                            param_types,
                                            result_type.clone(),
                                        );

                                        // Generate the call to the runtime function
                                        return self.builder.build_call_direct(
                                            runtime_func_id,
                                            final_arg_regs,
                                            result_type,
                                        );
                                    }
                                }
                            }
                        }
                    } // end of else block for receiver_is_user_class
                }
            } else {
                // receiver_is_class_type == true
                // This is an instance method call on a MIR wrapper class (Thread, Channel, etc.)
                // Route to the MIR wrapper function (Thread_join, Channel_send, etc.)
                let receiver_is_synthetic_class = args
                    .first()
                    .map(|arg| self.is_class_symbol_expr(arg))
                    .unwrap_or(false);
                if !receiver_is_synthetic_class {
                    if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                        if let Some(method_name) = self.string_interner.get(sym_info.name) {
                            // Get the class name from the receiver type
                            let class_name = {
                                let type_table = self.type_table;
                                type_table.get(receiver_type).and_then(|ti| {
                                    if let crate::tast::core::TypeKind::Class {
                                        symbol_id, ..
                                    } = &ti.kind
                                    {
                                        self.symbol_table
                                            .get_symbol(*symbol_id)
                                            .and_then(|s| self.string_interner.get(s.name))
                                            .map(|s| s.to_string())
                                    } else {
                                        None
                                    }
                                })
                            };

                            if let Some(class_name) = class_name {
                                // Build MIR wrapper function name: Thread_join, Channel_send, etc.
                                let mir_func_name = format!("{}_{}", class_name, method_name);
                                debug!(
                                    "[MIR WRAPPER INSTANCE] Routing {}.{} to {}",
                                    class_name, method_name, mir_func_name
                                );

                                // Get the registered signature for this MIR wrapper
                                if let Some((mir_param_types, mir_return_type)) =
                                    self.get_stdlib_mir_wrapper_signature(&mir_func_name)
                                {
                                    let call_args = if args.len() == mir_param_types.len() + 1
                                        && !args.is_empty()
                                    {
                                        &args[1..]
                                    } else {
                                        &args[..]
                                    };
                                    // Lower all arguments (first arg is receiver/self)
                                    // Auto-box primitive args when MIR wrapper expects Ptr(U8)
                                    // (e.g., Channel<Int>.send(42) needs to box the Int)
                                    let mut arg_regs = Vec::new();
                                    for (i, arg) in call_args.iter().enumerate() {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            let actual_ty = self.convert_type(arg.ty);
                                            let expected_ty = mir_param_types
                                                .get(i)
                                                .cloned()
                                                .unwrap_or_else(|| actual_ty.clone());

                                            // Auto-box if MIR wrapper expects Ptr(U8) but arg is primitive.
                                            // Channel payloads box uniformly (refs too); i==0 is the
                                            // channel handle/self (a reference) — never box it.
                                            let final_reg = if (mir_func_name == "Channel_send"
                                                || mir_func_name == "Channel_trySend")
                                                && i >= 1
                                            {
                                                self.box_channel_payload(
                                                    reg,
                                                    arg.ty,
                                                    &actual_ty,
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
                                        }
                                    }

                                    // Register forward reference to MIR wrapper
                                    let mir_func_id = self.register_stdlib_mir_forward_ref(
                                        &mir_func_name,
                                        mir_param_types,
                                        mir_return_type.clone(),
                                    );

                                    debug!(
                                    "[MIR WRAPPER INSTANCE] Registered forward ref to {} with ID {:?}",
                                    mir_func_name, mir_func_id
                                );

                                    // Generate the call with the MIR wrapper's return type
                                    let call_result = self.builder.build_call_direct(
                                        mir_func_id,
                                        arg_regs,
                                        mir_return_type.clone(),
                                    )?;

                                    // Auto-unbox if MIR wrapper returns Ptr(U8) but HIR expects primitive
                                    // (e.g., Channel<Int>.tryReceive() returns boxed int)
                                    debug!(
                                    "[MIR WRAPPER INSTANCE] call_result={:?}, mir_return_type={:?}, result_type={:?}",
                                    call_result, mir_return_type, result_type
                                );
                                    if mir_func_name == "Channel_receive"
                                        || mir_func_name == "Channel_tryReceive"
                                    {
                                        return self.unbox_channel_return(
                                            call_result,
                                            &result_type,
                                            mir_func_name == "Channel_tryReceive",
                                        );
                                    }
                                    return self.maybe_unbox_for_extern_return(
                                        call_result,
                                        &mir_return_type,
                                        &result_type,
                                    );
                                } else {
                                    debug!(
                                    "[MIR WRAPPER INSTANCE] No signature found for {}, falling through",
                                    mir_func_name
                                );
                                }
                            }
                        }
                    }
                } else {
                    debug!(
                        "[MIR WRAPPER INSTANCE] Skipping instance-wrapper dispatch for synthetic class receiver"
                    );
                }
            } // end if receiver_is_class_type else block
        }
        *fell_through = true;
        None
    }
}
