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
