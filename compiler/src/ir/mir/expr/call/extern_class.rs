//! Methods on `extern` classes, redirected to their runtime entry points.

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
    pub(crate) fn try_extern_class_method_call(
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
            unreachable!("try_extern_class_method_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if *is_method && !args.is_empty() {
            let receiver = &args[0];
            // Resolve TypeAlias to get the actual receiver type
            // (e.g., List<Int> may be wrapped in TypeAlias)
            let receiver_type = self.resolve_through_aliases(receiver.ty);

            // Extract receiver class hint for disambiguation (e.g., "rayzor_ds_Tensor")
            let receiver_class_hint_owned: Option<String> = if let HirExprKind::Variable {
                symbol: recv_sym,
                ..
            } = &receiver.kind
            {
                self.monomorphized_var_types
                    .get(recv_sym)
                    .map(|s| s.to_string())
            } else {
                None
            };
            // Fallback: use find_receiver_class_name if monomorphized_var_types didn't have it
            let receiver_class_hint_owned = receiver_class_hint_owned
                .or_else(|| self.find_receiver_class_name(receiver))
                .or_else(|| {
                    // Fallback: check register_class_hints for the receiver's MIR register.
                    // This resolves class names for variables assigned from extern class
                    // constructors (e.g., `var map = new ObjectMap<K,V>()`), where the
                    // New handler stored a class hint on the result register.
                    if let HirExprKind::Variable {
                        symbol: recv_sym, ..
                    } = &receiver.kind
                    {
                        self.symbol_map
                            .get(recv_sym)
                            .and_then(|reg| self.register_class_hints.get(reg).cloned())
                    } else {
                        None
                    }
                });

            // The declared (TAST) receiver type is the sole authority for
            // SIMD-vector classification: SSA register-id reuse lets the
            // name/register hints computed above carry a stale class either way
            // (a SIMD receiver hinted as Usize, or a Usize receiver hinted as
            // SIMD4f, which would build a VectorBinOp over i64 operands).
            let ir_ty = self.convert_type(receiver_type);
            let receiver_class_hint_owned = if ir_ty.is_vector() {
                // SIMD4i32 (i32x4) and SIMD4f (f32x4) are both vectors but their
                // instance methods (sum/get/set) map to different wrappers.
                Some(simd_vector_class(&ir_ty).to_string())
            } else {
                // Not a vector, so a SIMD class named by the hint is stale and
                // must be rejected before reaching the arith interception below.
                // A non-SIMD hint (a genuine Bytes/Usize) is left intact.
                let non_simd_hint = receiver_class_hint_owned.filter(|h| {
                    !self.stdlib_mapping.same_class(h, "rayzor.SIMD4f")
                        && !self.stdlib_mapping.same_class(h, "rayzor.SIMD4i32")
                });
                if non_simd_hint.is_some() {
                    non_simd_hint
                } else if self.type_is_native_named(receiver_type, "rayzor::Atomic") {
                    // Atomic's type-map returns Ptr<I32> (not a vector), so is_vector()
                    // never fires; resolve it by the abstract's @:native name instead.
                    Some("rayzor.Atomic".to_string())
                } else if let crate::ir::hir::HirExprKind::Variable {
                    symbol: recv_sym, ..
                } = &receiver.kind
                {
                    // Chained-call recovery: receiver's HIR type is Dynamic but its
                    // register was typed VecF32x4 by a previous SIMD4f call (e.g.
                    // b.sum() where b = a.sqrt()). Only trust the register type when
                    // it is genuinely a vector — a plain address register is I64.
                    self.symbol_map
                        .get(recv_sym)
                        .and_then(|reg| self.builder.get_register_type(*reg))
                        .filter(|ty| ty.is_vector())
                        .map(|ty| simd_vector_class(&ty).to_string())
                } else {
                    None
                }
            };
            let receiver_class_hint = receiver_class_hint_owned.as_deref();

            // SIMD4f arithmetic methods (`a.add(b)`) must compile to the same
            // single vector instruction as the operators (`a + b`); the default
            // method-call path routes them to a MIR wrapper that mishandles the
            // vector ABI. Restricted to rayzor_SIMD4f (f32x4) — integer
            // VectorBinOp miscompiles on the wasm backend.
            if receiver_class_hint == Some("rayzor.SIMD4f") && args.len() == 2 {
                let mname = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name));
                let vbop = match mname {
                    Some("add") => Some(BinaryOp::Add),
                    Some("sub") => Some(BinaryOp::Sub),
                    Some("mul") => Some(BinaryOp::Mul),
                    Some("div") => Some(BinaryOp::Div),
                    _ => None,
                };
                // The hint alone is not trusted: the operand's own declared type
                // must classify as f32x4, so a VectorBinOp is never built over a
                // non-vector operand.
                let operands_are_simd4f = {
                    let t = self.convert_type(args[0].ty);
                    t.is_vector() && simd_vector_class(&t) == "rayzor.SIMD4f"
                };
                if let Some(bin_op) = vbop.filter(|_| operands_are_simd4f) {
                    let lhs_reg = self.lower_expression(&args[0])?;
                    let rhs_reg = self.lower_expression(&args[1])?;
                    // vec_ty must always be a vector: fall through both operand
                    // register types to the f32x4 default rather than emitting
                    // VectorBinOp{I64}.
                    let vec_ty = self
                        .builder
                        .get_register_type(lhs_reg)
                        .filter(|t| matches!(t, IrType::Vector { .. }))
                        .or_else(|| {
                            self.builder
                                .get_register_type(rhs_reg)
                                .filter(|t| matches!(t, IrType::Vector { .. }))
                        })
                        .unwrap_or(IrType::Vector {
                            element: Box::new(IrType::F32),
                            count: 4,
                        });
                    return self
                        .builder
                        .build_vector_binop(bin_op, lhs_reg, rhs_reg, vec_ty);
                }
            }

            // Calculate actual param count (excluding the receiver) for overload disambiguation
            // e.g., s.indexOf("World", 0) has args=[s, "World", 0], param_count=2
            let param_count = args.len().saturating_sub(1);

            // SIMD4f direct lookup: When receiver is known to be SIMD4f, bypass
            // get_stdlib_runtime_info (whose FALLBACK2 excludes SIMD matches).
            let runtime_info = if matches!(
                receiver_class_hint,
                Some("rayzor.SIMD4f") | Some("rayzor.SIMD4i32")
            ) {
                let simd_cls = receiver_class_hint.unwrap();
                let method_name_str = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name));
                let simd_key = self.stdlib_mapping.class_key(simd_cls);
                if let (Some(mn), Some(simd_key)) = (method_name_str, simd_key) {
                    self.stdlib_mapping
                        .find_by_name_and_params(simd_key, mn, param_count)
                        .or_else(|| self.stdlib_mapping.find_by_name(simd_key, mn))
                        .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                } else {
                    None
                }
            } else if receiver_class_hint == Some("rayzor.Atomic") {
                // Atomic direct lookup: bypass FALLBACK2 (mirror of SIMD4f).
                let method_name_str = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name));
                method_name_str.and_then(|mn| {
                    self.stdlib_mapping
                        .find_by_name_and_params(
                            self.stdlib_mapping.key("rayzor.Atomic"),
                            mn,
                            param_count,
                        )
                        .or_else(|| {
                            self.stdlib_mapping
                                .find_by_name(self.stdlib_mapping.key("rayzor.Atomic"), mn)
                        })
                        .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                })
            } else {
                // Try to find stdlib runtime mapping for this method
                self.get_stdlib_runtime_info(
                    *symbol,
                    receiver_type,
                    Some(param_count),
                    receiver_class_hint,
                )
            };
            if let Some((class_name, method_name, runtime_call)) = runtime_info {
                // Skip methods that need ptr_conversion - let them fall through to
                // the existing handler which properly handles params_need_ptr_conversion
                if runtime_call.params_need_ptr_conversion != 0 {
                    debug!(
                        "[EXTERN METHOD VAR] Skipping {} - has ptr_conversion, using fallback path",
                        runtime_call.runtime_name
                    );
                } else {
                    let runtime_func = runtime_call.runtime_name;
                    let is_instance_method = runtime_call.has_self_param;
                    let is_mir_wrapper = runtime_call.is_mir_wrapper;
                    let returns_raw_value = runtime_call.returns_raw_value;
                    let raw_value_params = runtime_call.raw_value_params;
                    let extend_to_i64_params = runtime_call.extend_to_i64_params;
                    let has_return = runtime_call.has_return;
                    let explicit_return_type = runtime_call.return_type.map(|t| t.to_ir_type());
                    if std::env::var_os("RAYZOR_TRACE_STDLIB_DISPATCH").is_some() {
                        eprintln!(
                            "[EXTERN METHOD VAR] Redirecting {}.{} -> {} (instance={}, mir_wrapper={})",
                            class_name, method_name, runtime_func, is_instance_method, is_mir_wrapper
                        );
                    }
                    debug!(
                        "[EXTERN METHOD VAR] Redirecting {}.{} -> {} (instance={}, mir_wrapper={})",
                        class_name, method_name, runtime_func, is_instance_method, is_mir_wrapper
                    );

                    // MIR wrapper path: use register_stdlib_mir_forward_ref
                    // MIR wrappers (SIMD4f, Thread, Channel, etc.) are compiled by
                    // Cranelift alongside user code. They must NOT be registered as
                    // extern C functions.
                    if is_mir_wrapper {
                        self.builder.call_label = Some(format!("MIR_WRAPPER:{}", runtime_func));

                        // Get the MIR wrapper's expected signature for auto-boxing/unboxing
                        let mir_wrapper_sig = self.get_stdlib_mir_wrapper_signature(runtime_func);

                        // Lower receiver + args with auto-boxing
                        // When MIR wrapper expects Ptr(U8) but arg is a concrete primitive
                        // (I32, F64, Bool from Channel<Int>.send(42)), box the value.
                        // But for type-erased pointers (I64 from TypeParameter), just cast.
                        let mut arg_regs = Vec::new();
                        let mut param_types = Vec::new();
                        for (i, arg) in args.iter().enumerate() {
                            if i == 0 && !is_instance_method {
                                continue; // Skip class receiver for static methods
                            }
                            if let Some(reg) = self.lower_expression(arg) {
                                let actual_ty =
                                    self.builder.get_register_type(reg).unwrap_or(IrType::I64);
                                let param_idx = if is_instance_method { i } else { i - 1 };
                                let expected_ty = mir_wrapper_sig
                                    .as_ref()
                                    .and_then(|(params, _)| params.get(param_idx).cloned())
                                    .unwrap_or_else(|| actual_ty.clone());

                                // Check if this arg is a type-erased pointer (I64 from
                                // TypeParameter/class/GenericInstance/Array) vs a concrete
                                // primitive (I32/F64/Bool). Type-erased pointers should
                                // be CAST to Ptr(U8), not BOXED as integers.
                                let is_type_erased_ptr = matches!(actual_ty, IrType::I64) && {
                                    let type_table = self.type_table;
                                    type_table
                                        .get(arg.ty)
                                        .map(|ti| {
                                            matches!(
                                                ti.kind,
                                                crate::tast::TypeKind::TypeParameter { .. }
                                                    | crate::tast::TypeKind::Class { .. }
                                                    | crate::tast::TypeKind::GenericInstance { .. }
                                                    | crate::tast::TypeKind::Interface { .. }
                                                    | crate::tast::TypeKind::Dynamic
                                                    | crate::tast::TypeKind::Placeholder { .. }
                                                    | crate::tast::TypeKind::Array { .. }
                                                    | crate::tast::TypeKind::Abstract { .. }
                                                    | crate::tast::TypeKind::Function { .. }
                                            )
                                        })
                                        .unwrap_or(false)
                                };

                                let final_reg = if (runtime_func == "Channel_send"
                                    || runtime_func == "Channel_trySend")
                                    && i >= 1
                                {
                                    // Uniformly box Channel payloads (refs too) so the
                                    // erased receive can tag-dispatch. i==0 is the channel
                                    // handle — never box it.
                                    self.box_channel_payload(reg, arg.ty, &actual_ty, &expected_ty)?
                                } else if is_type_erased_ptr
                                    && matches!(&expected_ty, IrType::Ptr(_))
                                {
                                    // Cast I64 → Ptr(U8) for type-erased pointers
                                    self.builder
                                        .build_cast(reg, IrType::I64, expected_ty.clone())
                                        .unwrap_or(reg)
                                } else {
                                    self.maybe_box_for_extern_call(reg, &actual_ty, &expected_ty)?
                                };
                                arg_regs.push(final_reg);
                                param_types.push(expected_ty);
                            }
                        }

                        // Use the MIR wrapper's actual return type instead of the
                        // erased HIR type. For Dynamic/TypeParameter returns, the HIR
                        // type erases to Ptr(Void) or I64, but the MIR wrapper returns
                        // a concrete type (e.g., Ptr(U8)). Using the concrete type
                        // prevents spurious unboxing in downstream field access.
                        let mir_return_type = mir_wrapper_sig
                            .map(|(_, ret)| ret)
                            .unwrap_or_else(|| result_type.clone());

                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                            runtime_func,
                            param_types,
                            mir_return_type.clone(),
                        );

                        let call_result = self.builder.build_call_direct(
                            mir_func_id,
                            arg_regs,
                            mir_return_type.clone(),
                        )?;

                        // Store class hint for result to enable disambiguation
                        // of subsequent method calls on TypeParameter receivers
                        {
                            let return_class = self.get_return_class_hint(class_name, method_name);
                            self.register_class_hints
                                .insert(call_result, return_class.to_string());
                        }

                        // Auto-unbox if MIR wrapper returns Ptr(U8) but HIR expects primitive
                        // (e.g., Thread<Int>.join() returns boxed int, Channel<Int>.tryReceive()
                        // returns boxed int). Resolve T from receiver's generic type_args.
                        let resolved_expected = {
                            let needs_resolve = result_type == IrType::Any
                                || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                                || result_type == IrType::I64;
                            if needs_resolve {
                                let type_table = self.type_table;
                                type_table
                                    .get(receiver_type)
                                    .and_then(|ti| match &ti.kind {
                                        crate::tast::TypeKind::Class { type_args, .. }
                                        | crate::tast::TypeKind::GenericInstance {
                                            type_args,
                                            ..
                                        } => {
                                            if !type_args.is_empty() {
                                                Some(self.convert_type(type_args[0]))
                                            } else {
                                                None
                                            }
                                        }
                                        _ => None,
                                    })
                                    .unwrap_or_else(|| result_type.clone())
                            } else {
                                result_type.clone()
                            }
                        };

                        // Thread<T>.join() boxes its result, so for a concrete heap
                        // T the boxed i64 payload is the object pointer. maybe_unbox's
                        // raw passthrough arm is shared with methods returning an
                        // un-boxed handle (Arc.get) and would hand back the box
                        // address, so unbox inline keyed on the wrapper name.
                        // Ptr(primitive) (Null<Int>) is handled above.
                        let resolved_is_heap_ptr =
                            matches!(&resolved_expected, IrType::Ptr(inner) if !matches!(
                                inner.as_ref(),
                                IrType::I32
                                    | IrType::I64
                                    | IrType::F32
                                    | IrType::F64
                                    | IrType::Bool
                            )) || matches!(resolved_expected, IrType::String);
                        let mir_ret_is_ptr_u8 = matches!(&mir_return_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8 | IrType::Void));
                        if runtime_func == "Thread_join"
                            && resolved_is_heap_ptr
                            && mir_ret_is_ptr_u8
                        {
                            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                            let unbox_id = self.get_or_register_extern_function(
                                "haxe_unbox_int_ptr",
                                vec![ptr_u8.clone()],
                                IrType::I64,
                            );
                            let i64v = self.builder.build_call_direct(
                                unbox_id,
                                vec![call_result],
                                IrType::I64,
                            )?;
                            return self.builder.build_cast(i64v, IrType::I64, ptr_u8);
                        }

                        // Channel payloads are uniformly boxed DynamicValues, so the
                        // return goes through the tag-aware unbox, not the raw path.
                        if runtime_func == "Channel_receive" || runtime_func == "Channel_tryReceive"
                        {
                            let is_try = runtime_func == "Channel_tryReceive";
                            // Inferred channels erase T to I64, whose unbox
                            // int-truncates a Float payload; the enclosing
                            // declared type is the ground truth. Scoped to floats
                            // and non-try, so tryReceive keeps the tag-driven
                            // unbox and nullables stay correct.
                            let refined = if !is_try && matches!(resolved_expected, IrType::I64) {
                                self.let_target_type_hint
                                    .map(|t| self.convert_type(t))
                                    .filter(|t| matches!(t, IrType::F64 | IrType::F32))
                            } else {
                                None
                            };
                            return self.unbox_channel_return(
                                call_result,
                                refined.as_ref().unwrap_or(&resolved_expected),
                                is_try,
                            );
                        }

                        return self.maybe_unbox_for_extern_return(
                            call_result,
                            &mir_return_type,
                            &resolved_expected,
                        );
                    }

                    // Extern C path: register as extern function
                    // Get expected parameter types from the extern function signature
                    let (expected_param_types, actual_return_type) = self
                        .get_stdlib_mir_wrapper_signature(runtime_func)
                        .map(|(params, ret)| (params, ret))
                        .unwrap_or_else(|| {
                            // When is_method=true, args[0] is always receiver/class - skip it
                            // For instance methods, add self param first
                            let mut params = if is_instance_method {
                                vec![IrType::Ptr(Box::new(IrType::U8))]
                            } else {
                                vec![]
                            };
                            // Always skip args[0] since is_method=true
                            // Use stdlib mapping hints for param types
                            for (i, arg) in args.iter().skip(1).enumerate() {
                                // raw_value_params: bit 0 = self, bit 1 = first user param, etc.
                                let user_bit = 1u32 << (i + 1);
                                if raw_value_params & user_bit != 0 {
                                    params.push(IrType::U64);
                                } else if extend_to_i64_params & user_bit != 0 {
                                    params.push(IrType::I64);
                                } else {
                                    params.push(self.convert_type(arg.ty));
                                }
                            }
                            // Use explicit return type from mapping if available,
                            // otherwise fall back to HIR-inferred result_type
                            let ret_type = if returns_raw_value {
                                IrType::U64
                            } else if let Some(ref ert) = explicit_return_type {
                                ert.clone()
                            } else if has_return {
                                result_type.clone()
                            } else {
                                IrType::Void
                            };
                            (params, ret_type)
                        });

                    self.builder.call_label = Some(format!("EXTERN_C:{}", runtime_func));
                    // Build argument list based on whether this is instance or static method
                    let mut arg_regs = Vec::new();
                    let args_to_process: &[HirExpr] = if is_instance_method {
                        let receiver_reg = self.lower_expression(receiver)?;
                        arg_regs.push(receiver_reg);
                        &args[1..]
                    } else {
                        &args[1..]
                    };

                    // Lower the arguments and auto-box if needed
                    let param_offset = if is_instance_method { 1 } else { 0 };
                    for (i, arg) in args_to_process.iter().enumerate() {
                        let arg_reg = self.lower_expression(arg)?;
                        let actual_ty = self.convert_type(arg.ty);
                        let expected_ty = expected_param_types
                            .get(i + param_offset)
                            .cloned()
                            .unwrap_or_else(|| actual_ty.clone());
                        let final_reg =
                            self.maybe_box_for_extern_call(arg_reg, &actual_ty, &expected_ty)?;
                        arg_regs.push(final_reg);
                    }

                    // Use expected parameter types for registration
                    let param_types = if expected_param_types.len() == arg_regs.len() {
                        expected_param_types.clone()
                    } else {
                        let mut params = if is_instance_method {
                            vec![IrType::Ptr(Box::new(IrType::U8))]
                        } else {
                            vec![]
                        };
                        for arg in args.iter().skip(1) {
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

                    // Handle returns_raw_value: cast raw U64 to the appropriate type
                    if returns_raw_value {
                        // Compute the actual element type T from the receiver's
                        // generic args. When T resolves (e.g. `Map<String, Tensor>`
                        // → T = Tensor), the u64 returned by the runtime is a
                        // pointer that needs bit-reinterpret to the right Ptr
                        // type. When T is unresolved (`new StringMap()` with no
                        // type args), the value is a primitive stored as raw
                        // bits — keep on the U64→I64 cast path for downstream
                        // unboxing.
                        let resolved_value_ty: Option<IrType> = {
                            let type_table = self.type_table;
                            type_table.get(receiver_type).and_then(|ti| {
                                match &ti.kind {
                                    crate::tast::TypeKind::Class { type_args, .. }
                                    | crate::tast::TypeKind::GenericInstance {
                                        type_args, ..
                                    } => {
                                        // Value T is the LAST type arg: StringMap<T> /
                                        // IntMap<T> have type_args=[T]; ObjectMap<K, V>
                                        // has type_args=[K, V] — the value lives at the
                                        // tail in both cases.
                                        type_args.last().map(|ta| self.convert_type(*ta))
                                    }
                                    _ => None,
                                }
                            })
                        };
                        // When the HIR result_type is opaque (Any/I64/Ptr<U8>) because
                        // the Haxe surface declares `Null<V>` for Map.get, the *real*
                        // value type is `resolved_value_ty` (V from receiver's type
                        // args). For F64/F32 V, bitcast the raw u64 bits directly to
                        // float rather than try to unbox a DynamicValue (the runtime
                        // stores raw bits, not a heap-boxed value, so the unbox path
                        // would mis-interpret the bytes).
                        let result_is_opaque = match &result_type {
                            IrType::Any | IrType::I64 => true,
                            IrType::Ptr(inner) => {
                                matches!(**inner, IrType::U8 | IrType::Void)
                            }
                            _ => false,
                        };
                        let effective_ty = match resolved_value_ty.as_ref() {
                            Some(rty @ (IrType::F64 | IrType::F32)) if result_is_opaque => {
                                rty.clone()
                            }
                            _ => result_type.clone(),
                        };
                        let final_result = match &effective_ty {
                            IrType::I32 => {
                                self.builder
                                    .build_cast(call_result, IrType::U64, IrType::I32)
                            }
                            IrType::I64 => {
                                self.builder
                                    .build_cast(call_result, IrType::U64, IrType::I64)
                            }
                            IrType::F64 => self.builder.build_bitcast(call_result, IrType::F64),
                            IrType::F32 => {
                                if let Some(f64_reg) =
                                    self.builder.build_bitcast(call_result, IrType::F64)
                                {
                                    self.builder.build_cast(f64_reg, IrType::F64, IrType::F32)
                                } else {
                                    None
                                }
                            }
                            IrType::Bool => {
                                self.builder
                                    .build_cast(call_result, IrType::U64, IrType::Bool)
                            }
                            IrType::Ptr(ref inner) if matches!(inner.as_ref(), IrType::String) => {
                                // Concrete pointer type (e.g., Ptr(String))
                                self.builder.build_cast(
                                    call_result,
                                    IrType::U64,
                                    result_type.clone(),
                                )
                            }
                            IrType::Ptr(_)
                                if matches!(resolved_value_ty.as_ref(), Some(IrType::Ptr(_))) =>
                            {
                                // Receiver parameterised with a concrete
                                // pointer-typed T (extern class, user class,
                                // array). Bit-reinterpret the runtime u64 as
                                // the resolved pointer type.
                                self.builder
                                    .build_bitcast(call_result, resolved_value_ty.unwrap())
                            }
                            _ => {
                                // Unresolved T or Dynamic, or T resolved to a
                                // primitive that arrived here boxed (result_type
                                // = Ptr<U8> from `Null<Int>`): keep as I64 so
                                // the downstream unbox path can extract the
                                // primitive value. Bitcasting U64 → I32 here
                                // would skip that boxing and produce values
                                // that don't match the consumer's expected
                                // register type.
                                self.builder
                                    .build_cast(call_result, IrType::U64, IrType::I64)
                            }
                        };
                        return final_result;
                    }

                    // Auto-unbox if runtime returns Ptr(U8) but HIR expects primitive
                    let unboxed = self.maybe_unbox_for_extern_return(
                        call_result,
                        &actual_return_type,
                        &result_type,
                    );
                    return unboxed;
                } // end else (no ptr_conversion needed)
            }
        }
        *fell_through = true;
        None
    }
}
