//! Calls that map onto a stdlib runtime function.

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
    pub(crate) fn try_stdlib_runtime_dispatch(
        &mut self,
        expr: &HirExpr,
        receiver_type: TypeId,
        receiver_is_class_type: bool,
        result_type: IrType,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("try_stdlib_runtime_dispatch on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        if !receiver_is_class_type {
            // args[0] is the receiver, the rest are params.
            let method_param_count = if args.len() > 1 { args.len() - 1 } else { 0 };
            {
                if let Some((class_name, method_name, runtime_call)) = self.get_stdlib_runtime_info(
                    *symbol,
                    receiver_type,
                    Some(method_param_count),
                    None,
                ) {
                    let runtime_func = runtime_call.runtime_name;
                    let ptr_conversion_mask = runtime_call.params_need_ptr_conversion;
                    let raw_value_mask = runtime_call.raw_value_params;
                    let returns_raw_value = runtime_call.returns_raw_value;
                    let extend_i64_mask = runtime_call.extend_to_i64_params;
                    let needs_out_param = runtime_call.needs_out_param;
                    let has_return = runtime_call.has_return;

                    // These entries return void and write the result to the first out
                    // param, so wrap inline: allocate + call + load.
                    if needs_out_param {
                        debug!(
                            "[OUT PARAM] Instance method {}.{} needs out param inline wrapper",
                            class_name, method_name
                        );

                        let mut call_arg_regs = Vec::new();
                        for arg in args {
                            if let Some(reg) = self.lower_expression(arg) {
                                call_arg_regs.push(reg);
                            }
                        }

                        // Opaque pointer-sized slot for the result object.
                        let out_ptr_ty = IrType::Ptr(Box::new(IrType::Void));
                        let out_ptr = self.builder.build_alloc(out_ptr_ty.clone(), None)?;

                        // void runtime_func(out: *Ptr(Void), receiver: Ptr(Void), ...params)
                        let mut extern_param_types = vec![out_ptr_ty.clone()];
                        for arg in args {
                            extern_param_types.push(self.convert_type(arg.ty));
                        }

                        let extern_func_id = self.get_or_register_extern_function(
                            runtime_func,
                            extern_param_types,
                            IrType::Void,
                        );

                        let mut runtime_args = vec![out_ptr];
                        runtime_args.extend(call_arg_regs);

                        self.builder
                            .build_call_direct(extern_func_id, runtime_args, IrType::Void);

                        let result_ptr = self.builder.build_load(out_ptr, out_ptr_ty)?;

                        debug!(
                            "[OUT PARAM] Generated inline wrapper for {}, result_ptr: {:?}",
                            runtime_func, result_ptr
                        );

                        return Some(result_ptr);
                    }

                    // MIR wrappers forward to extern runtime functions, absorbing
                    // calling-convention differences and default arguments. Gate on the
                    // mapping's is_mir_wrapper, not the class: some methods on wrapper
                    // classes (String.split) are direct externs.
                    if runtime_call.is_mir_wrapper {
                        // The mapping's runtime name disambiguates overloads
                        // (String_indexOf vs String_indexOf_2).
                        let mir_func_name = runtime_func.to_string();
                        debug!(
                            "[STDLIB MIR] Detected stdlib MIR wrapper function (instance): {}",
                            mir_func_name
                        );

                        // Auto-box primitive args when the wrapper expects Ptr(U8).
                        let mir_wrapper_params = self
                            .get_stdlib_mir_wrapper_signature(&mir_func_name)
                            .map(|(params, _)| params);
                        let mut arg_regs = Vec::new();
                        let mut param_types = Vec::new();
                        for (i, arg) in args.iter().enumerate() {
                            if let Some(reg) = self.lower_expression(arg) {
                                let actual_ty = self.convert_type(arg.ty);
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
                                    self.box_channel_payload(reg, arg.ty, &actual_ty, &expected_ty)?
                                } else {
                                    self.maybe_box_for_extern_call(reg, &actual_ty, &expected_ty)?
                                };
                                arg_regs.push(final_reg);
                                param_types.push(expected_ty);
                            }
                        }

                        // Methods returning T (Thread<T>.join, Channel<T>.tryReceive) take
                        // their return type from the receiver's type args. Any, Ptr(Void)
                        // and I64 all mean "unresolved" here.
                        let needs_generic_resolve = result_type == IrType::Any
                            || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                            || result_type == IrType::I64;
                        let resolved_result_type = if needs_generic_resolve {
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

                        // Void-returning wrappers must use the signature's return type, or
                        // they get a dest register; otherwise resolved_result_type wins,
                        // since it handles generics.
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

                        // Call with the wrapper's own return type, which may be Ptr(U8) for
                        // generic methods returning T.
                        let mir_actual_return = self
                            .get_stdlib_mir_wrapper_signature(&mir_func_name)
                            .map(|(_, ret)| ret)
                            .unwrap_or_else(|| final_return_type.clone());
                        let call_result = self.builder.build_call_direct(
                            mir_func_id,
                            arg_regs,
                            mir_actual_return.clone(),
                        )?;

                        // Unbox when the wrapper returns Ptr(U8) but the caller expects a
                        // primitive.
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

                        // Hint the post-unbox register so later calls dispatch on the
                        // returned class (Array.iterator() -> ArrayIterator, so
                        // .hasNext()/.next() reach ArrayIterator's methods).
                        if let Some(result_reg) = final_result {
                            let return_class = self.get_return_class_hint(class_name, method_name);
                            self.register_class_hints
                                .insert(result_reg, return_class.to_string());
                            if let Some(h) = self.wrap_stdlib_iter_result(result_reg, expr.ty) {
                                return Some(h);
                            }
                        }

                        return final_result;
                    }

                    let arg_regs: Vec<_> = args
                        .iter()
                        .filter_map(|a| self.lower_expression(a))
                        .collect();

                    // Raw-value params (StringMap, IntMap) are stored inline as u64 bits:
                    // no boxing, no heap allocation.
                    let mut final_arg_regs = arg_regs.clone();
                    if raw_value_mask != 0 {
                        for i in 0..arg_regs.len() {
                            if (raw_value_mask & (1 << i)) != 0 {
                                let arg_reg = arg_regs[i];
                                let arg_type = self
                                    .builder
                                    .get_register_type(arg_reg)
                                    .unwrap_or(IrType::I64);

                                let raw_reg = match &arg_type {
                                    IrType::I32 => {
                                        self.builder.build_cast(arg_reg, IrType::I32, IrType::U64)
                                    }
                                    IrType::I64 => {
                                        self.builder.build_cast(arg_reg, IrType::I64, IrType::U64)
                                    }
                                    IrType::F64 => {
                                        // Bit pattern, not a value conversion.
                                        self.builder.build_bitcast(arg_reg, IrType::U64)
                                    }
                                    IrType::F32 => {
                                        // Widen to f64 first so the bit pattern is canonical.
                                        let f64_reg = self
                                            .builder
                                            .build_cast(arg_reg, IrType::F32, IrType::F64)
                                            .unwrap_or(arg_reg);
                                        self.builder.build_bitcast(f64_reg, IrType::U64)
                                    }
                                    IrType::Bool => {
                                        self.builder.build_cast(arg_reg, IrType::Bool, IrType::U64)
                                    }
                                    IrType::Ptr(_) => {
                                        // Address as an integer.
                                        self.builder.build_cast(
                                            arg_reg,
                                            arg_type.clone(),
                                            IrType::U64,
                                        )
                                    }
                                    _ => {
                                        // Anything else: direct cast.
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
                    // Legacy path, superseded by raw_value_params: boxes args as Dynamic.
                    else if ptr_conversion_mask != 0 {
                        for i in 0..arg_regs.len() {
                            if (ptr_conversion_mask & (1 << i)) != 0 {
                                let arg_reg = arg_regs[i];
                                let arg_type = self
                                    .builder
                                    .get_register_type(arg_reg)
                                    .unwrap_or(IrType::I64);

                                // haxe_box_*_ptr produces a tagged Dynamic that can be
                                // unboxed later.
                                let boxed_reg = match &arg_type {
                                    IrType::I32 => {
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
                                        // The runtime expects a pointer TO the value, so a
                                        // pointer argument is itself passed by reference:
                                        // haxe_array_push(arr, &value).
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

                    // Haxe Int is 32-bit but IntMap keys are 64-bit in the runtime.
                    if extend_i64_mask != 0 {
                        for i in 0..final_arg_regs.len() {
                            if (extend_i64_mask & (1 << i)) != 0 {
                                let arg_reg = final_arg_regs[i];
                                let arg_type = self
                                    .builder
                                    .get_register_type(arg_reg)
                                    .unwrap_or(IrType::I32);

                                if arg_type == IrType::I32 {
                                    if let Some(extended) =
                                        self.builder.build_cast(arg_reg, IrType::I32, IrType::I64)
                                    {
                                        final_arg_regs[i] = extended;
                                    }
                                }
                            }
                        }
                    }

                    // Param types come from TAST, adjusted for the conversions applied above.
                    let param_types: Vec<IrType> = args
                        .iter()
                        .enumerate()
                        .map(|(i, arg)| {
                            // Raw value params are passed as U64 (inline storage)
                            if raw_value_mask != 0 && (raw_value_mask & (1 << i)) != 0 {
                                IrType::U64
                            }
                            // Extended i64 params need i64 type in signature
                            else if extend_i64_mask != 0 && (extend_i64_mask & (1 << i)) != 0 {
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
                        let resolved = if let Some(receiver_info) = type_table.get(receiver_type) {
                            match &receiver_info.kind {
                                crate::tast::TypeKind::Class { type_args, .. }
                                | crate::tast::TypeKind::GenericInstance { type_args, .. } => {
                                    type_args.last().map(|ta| self.convert_type(*ta))
                                }
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
                        // A MIR wrapper's own return type wins: the HIR type may be
                        // Dynamic/Ptr(Void) while the wrapper returns something concrete.
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

                    let probe_param_types = param_types.clone();
                    let probe_args = final_arg_regs.clone();
                    let runtime_func_id = self.get_or_register_extern_function(
                        &runtime_func,
                        param_types,
                        call_return_type.clone(),
                    );

                    let call_result = self.builder.build_call_direct(
                        runtime_func_id,
                        final_arg_regs,
                        call_return_type,
                    );

                    if returns_raw_value {
                        if let Some(raw_reg) = call_result {
                            let final_result = match &resolved_return_type {
                                IrType::I32 => {
                                    self.builder.build_cast(raw_reg, IrType::U64, IrType::I32)
                                }
                                IrType::I64 => {
                                    self.builder.build_cast(raw_reg, IrType::U64, IrType::I64)
                                }
                                IrType::F64 => self.builder.build_bitcast(raw_reg, IrType::F64),
                                IrType::F32 => {
                                    if let Some(f64_reg) =
                                        self.builder.build_bitcast(raw_reg, IrType::F64)
                                    {
                                        self.builder.build_cast(f64_reg, IrType::F64, IrType::F32)
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
                                        self.builder
                                            .build_bitcast(raw_reg, resolved_return_type.clone())
                                    } else {
                                        self.builder.build_cast(raw_reg, IrType::U64, IrType::I64)
                                    }
                                }
                                _ => {
                                    // Truly unresolved T (Dynamic, type parameter)
                                    // — keep as I64 so the raw value isn't
                                    // misinterpreted as anything else.
                                    self.builder.build_cast(raw_reg, IrType::U64, IrType::I64)
                                }
                            };
                            // Declared `Null<scalar>`: consumers expect the box, and
                            // the raw bits cannot say "absent". Probe with the
                            // container's `exists`, then box.
                            if let Some(raw) = final_result {
                                let probe = self.raw_optional_probe(
                                    class_name,
                                    method_param_count,
                                    probe_param_types,
                                    probe_args,
                                );
                                if let Some(boxed) =
                                    self.box_raw_optional_result(raw, expr.ty, probe)
                                {
                                    return Some(boxed);
                                }
                            }
                            return final_result;
                        }
                    }

                    return call_result;
                }

                // A user-defined receiver must skip the stdlib fallbacks below, which
                // would otherwise match a same-named stdlib method.
                let receiver_is_user_class = {
                    let type_table = self.type_table;
                    type_table
                        .get(receiver_type)
                        .map(|ti| {
                            match &ti.kind {
                                crate::tast::core::TypeKind::Class { symbol_id, .. } => {
                                    // User-defined is anything the stdlib doesn't claim.
                                    self.symbol_table
                                        .get_symbol(*symbol_id)
                                        .map(|s| !self.is_stdlib_class_by_symbol(s))
                                        .unwrap_or(false)
                                }
                                // TypeParameter receivers always come from user-defined generics.
                                // Method calls on T should resolve through function_map, not stdlib.
                                // (Constrained T:Interface is handled earlier by interface dispatch.)
                                crate::tast::core::TypeKind::TypeParameter { .. } => true,
                                crate::tast::core::TypeKind::GenericInstance {
                                    base_type, ..
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

                if receiver_is_user_class {
                    // The method lives in function_map; stdlib matching would be wrong.
                } else {
                    // Fallback for when qualified names aren't set: try the stdlib
                    // mapping's class/method combinations.
                    if let Some(method_sym) = self.symbol_table.get_symbol(*symbol) {
                        if let Some(method_name) = self.string_interner.get(method_sym.name) {
                            let static_args = self.effective_static_call_args(args);
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
                                    // The mapping's `is_mir_wrapper` flag decides wrapper vs
                                    // true extern — having explicit type info does not, since
                                    // typed extern intrinsics like `haxe_bytes_get` carry
                                    // signatures too and routing them here creates a
                                    // body-less forward-ref stub that traps at runtime.
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

                                        let mut arg_regs = Vec::new();
                                        let mut param_types = Vec::new();
                                        for arg in static_args {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                arg_regs.push(reg);
                                                param_types.push(self.convert_type(arg.ty));
                                            }
                                        }

                                        // Body comes from the merged stdlib module.
                                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                                            runtime_func,
                                            param_types,
                                            result_type.clone(),
                                        );

                                        debug!(
                                        "[QUALIFIED NAME PATH] Registered forward ref to {} with ID {:?}",
                                        runtime_func, mir_func_id
                                    );

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

                                    let arg_regs: Vec<_> = static_args
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    // Expected types are needed before boxing.
                                    let (expected_param_types_qn, expected_return_type_qn) = self
                                        .get_extern_function_signature(&runtime_func)
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

                                    let runtime_func_id_qn = self.get_or_register_extern_function(
                                        &runtime_func,
                                        expected_param_types_qn,
                                        expected_return_type_qn.clone(),
                                    );

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

                                    // Unreachable: superseded by the return above.
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
                                            if (ptr_conversion_mask & (1 << i)) != 0 {
                                                let arg_reg = arg_regs[i];
                                                // Unknown type defaults to I64: pointers and
                                                // most values are 64-bit.
                                                let arg_type = self
                                                    .builder
                                                    .get_register_type(arg_reg)
                                                    .unwrap_or(IrType::I64);

                                                // Array element slots are always 8 bytes, so
                                                // widen smaller values.
                                                let (alloc_type, value_to_store) = match arg_type {
                                                    IrType::I32 => {
                                                        let ext_val = self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::I32,
                                                            IrType::I64,
                                                        );
                                                        (IrType::I64, ext_val.unwrap_or(arg_reg))
                                                    }
                                                    IrType::F32 => {
                                                        let ext_val = self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::F32,
                                                            IrType::F64,
                                                        );
                                                        (IrType::F64, ext_val.unwrap_or(arg_reg))
                                                    }
                                                    _ => (arg_type.clone(), arg_reg),
                                                };

                                                if let Some(stack_slot) = self
                                                    .builder
                                                    .build_alloc(alloc_type.clone(), None)
                                                {
                                                    self.builder
                                                        .build_store(stack_slot, value_to_store);
                                                    final_arg_regs[i] = stack_slot;
                                                }
                                            }
                                        }
                                    }

                                    // The mapping's signature (hlp_* introspection) is the
                                    // authoritative source of type info when present.
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

                                    return self.builder.build_call_direct(
                                        runtime_func_id,
                                        final_arg_regs,
                                        expected_return_type,
                                    );
                                }
                            }

                            // No usable qualified name: infer the class from the return type
                            // (Arc.init, Mutex.init, ...).
                            let inferred_class = {
                                let type_table = self.type_table;
                                debug!("[INFER CLASS] Checking return type expr.ty={:?}", expr.ty);
                                if let Some(type_info) = type_table.get(expr.ty) {
                                    debug!("[INFER CLASS] Return type kind={:?}", type_info.kind);
                                    if let TypeKind::Class { symbol_id, .. } = &type_info.kind {
                                        if let Some(class_sym) =
                                            self.symbol_table.get_symbol(*symbol_id)
                                        {
                                            // The returned class's registered key, so the
                                            // wrapper check below asks about that class.
                                            let class_name =
                                                self.canonical_stdlib_class_name(class_sym);
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
                                if self.stdlib_mapping.is_mir_wrapper_class(class_name) {
                                    // The mapping is the source of truth for the wrapper's
                                    // name — synthesizing it by the
                                    // `{class.lowercase()}_{method}` convention produces a
                                    // body-less stub whenever the real entry differs, which
                                    // traps at the call. The class here came from the RETURN
                                    // type, which differs from the declaring class for
                                    // non-factory methods (QTensor.gatherRowsQ6K returns
                                    // Tensor), so a globally unique method name is used to
                                    // identify the entry instead.
                                    let mir_func_name = self
                                        .stdlib_mapping
                                        .find_by_name(class_name, method_name)
                                        .or_else(|| {
                                            self.stdlib_mapping.find_unique_by_method(method_name)
                                        })
                                        .map(|(_, call)| call.runtime_name.to_string())
                                        .unwrap_or_else(|| {
                                            format!(
                                                "{}_{}",
                                                class_name.as_str().to_lowercase(),
                                                method_name
                                            )
                                        });
                                    debug!(
                                        "[STDLIB MIR] Detected stdlib MIR function: {}",
                                        mir_func_name
                                    );

                                    let mut arg_regs = Vec::new();
                                    let mut param_types = Vec::new();
                                    for arg in static_args {
                                        if let Some(reg) = self.lower_expression(arg) {
                                            arg_regs.push(reg);
                                            param_types.push(self.convert_type(arg.ty));
                                        }
                                    }

                                    // Body comes from the merged stdlib module.
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

                                    let arg_regs: Vec<_> = static_args
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    // Params flagged in the mapping are passed as a POINTER
                                    // TO the value, not the value itself.
                                    let mut final_arg_regs = arg_regs.clone();
                                    let ptr_conversion_mask = self
                                        .stdlib_mapping
                                        .find_by_runtime_name(&runtime_func)
                                        .map(|m| m.params_need_ptr_conversion)
                                        .unwrap_or(0);
                                    if ptr_conversion_mask != 0 {
                                        for i in 0..arg_regs.len() {
                                            if (ptr_conversion_mask & (1 << i)) != 0 {
                                                let arg_reg = arg_regs[i];
                                                // Unknown type defaults to I64: pointers and
                                                // most values are 64-bit.
                                                let arg_type = self
                                                    .builder
                                                    .get_register_type(arg_reg)
                                                    .unwrap_or(IrType::I64);

                                                // Array element slots are always 8 bytes, so
                                                // widen smaller values.
                                                let (alloc_type, value_to_store) = match arg_type {
                                                    IrType::I32 => {
                                                        let ext_val = self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::I32,
                                                            IrType::I64,
                                                        );
                                                        (IrType::I64, ext_val.unwrap_or(arg_reg))
                                                    }
                                                    IrType::F32 => {
                                                        let ext_val = self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::F32,
                                                            IrType::F64,
                                                        );
                                                        (IrType::F64, ext_val.unwrap_or(arg_reg))
                                                    }
                                                    _ => (arg_type.clone(), arg_reg),
                                                };

                                                if let Some(stack_slot) = self
                                                    .builder
                                                    .build_alloc(alloc_type.clone(), None)
                                                {
                                                    self.builder
                                                        .build_store(stack_slot, value_to_store);
                                                    final_arg_regs[i] = stack_slot;
                                                }
                                            }
                                        }
                                    }

                                    // The mapping's signature (hlp_* introspection) is the
                                    // authoritative source of type info when present.
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

                                    return self.builder.build_call_direct(
                                        runtime_func_id,
                                        final_arg_regs,
                                        expected_return_type,
                                    );
                                }
                            }

                            // Last resort: try all stdlib classes, matching param count to
                            // disambiguate overloads (Array.join(sep) vs Thread.join()).
                            let actual_arg_count = args.len().saturating_sub(1); // receiver
                            debug!(
                            "[LAST RESORT] Could not infer class for method '{}' with {} args, trying all stdlib classes",
                            method_name, actual_arg_count
                        );
                            // No MIR-wrapper-class detection here: this loop tries every
                            // class and would match the wrong one.
                            let stdlib_classes = self.stdlib_mapping.all_class_keys();
                            for class_name in &stdlib_classes {
                                if let Some((sig, mapping)) =
                                    self.stdlib_mapping.find_by_name_and_params(
                                        *class_name,
                                        method_name,
                                        actual_arg_count,
                                    )
                                {
                                    let runtime_func = mapping.runtime_name;

                                    // Wrapper vs extern is decided by the mapping's
                                    // `is_mir_wrapper` flag: typed extern intrinsics carry
                                    // signatures too, and a forward-ref stub for one never
                                    // gets a body, so it traps at runtime.
                                    if let Some((mir_param_types, mir_return_type)) = self
                                        .get_stdlib_mir_wrapper_signature(&runtime_func)
                                        .filter(|_| mapping.is_mir_wrapper)
                                    {
                                        debug!(
                                            "[FALLBACK PATH] Detected MIR wrapper: {}",
                                            runtime_func
                                        );

                                        let mut arg_regs = Vec::new();
                                        for arg in args {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                arg_regs.push(reg);
                                            }
                                        }

                                        // Body comes from the merged stdlib module.
                                        let mir_func_id = self.register_stdlib_mir_forward_ref(
                                            &runtime_func,
                                            mir_param_types,
                                            mir_return_type,
                                        );

                                        debug!(
                                        "[FALLBACK PATH] Registered forward ref to {} with ID {:?}",
                                        runtime_func, mir_func_id
                                    );

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

                                    let arg_regs: Vec<_> = args
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    // Params flagged in the mapping are passed as a POINTER
                                    // TO the value, not the value itself.
                                    let mut final_arg_regs = arg_regs.clone();
                                    let ptr_conversion_mask = self
                                        .stdlib_mapping
                                        .find_by_runtime_name(&runtime_func)
                                        .map(|m| m.params_need_ptr_conversion)
                                        .unwrap_or(0);
                                    if ptr_conversion_mask != 0 {
                                        for i in 0..arg_regs.len() {
                                            if (ptr_conversion_mask & (1 << i)) != 0 {
                                                let arg_reg = arg_regs[i];
                                                // Unknown type defaults to I64: pointers and
                                                // most values are 64-bit.
                                                let arg_type = self
                                                    .builder
                                                    .get_register_type(arg_reg)
                                                    .unwrap_or(IrType::I64);

                                                // Array element slots are always 8 bytes, so
                                                // widen smaller values.
                                                let (alloc_type, value_to_store) = match arg_type {
                                                    IrType::I32 => {
                                                        let ext_val = self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::I32,
                                                            IrType::I64,
                                                        );
                                                        (IrType::I64, ext_val.unwrap_or(arg_reg))
                                                    }
                                                    IrType::F32 => {
                                                        let ext_val = self.builder.build_cast(
                                                            arg_reg,
                                                            IrType::F32,
                                                            IrType::F64,
                                                        );
                                                        (IrType::F64, ext_val.unwrap_or(arg_reg))
                                                    }
                                                    _ => (arg_type.clone(), arg_reg),
                                                };

                                                if let Some(stack_slot) = self
                                                    .builder
                                                    .build_alloc(alloc_type.clone(), None)
                                                {
                                                    self.builder
                                                        .build_store(stack_slot, value_to_store);
                                                    final_arg_regs[i] = stack_slot;
                                                }
                                            }
                                        }
                                    }

                                    // Param types come from TAST, with ptr conversion applied.
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
                                    let runtime_func_id = self.get_or_register_extern_function(
                                        &runtime_func,
                                        param_types,
                                        result_type.clone(),
                                    );

                                    return self.builder.build_call_direct(
                                        runtime_func_id,
                                        final_arg_regs,
                                        result_type,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Instance method on a MIR wrapper class: route to the wrapper function
            // (Thread_join, Channel_send, ...).
            let receiver_is_synthetic_class = args
                .first()
                .map(|arg| self.is_class_symbol_expr(arg))
                .unwrap_or(false);
            if !receiver_is_synthetic_class {
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(method_name) = self.string_interner.get(sym_info.name) {
                        let class_name = {
                            let type_table = self.type_table;
                            type_table.get(receiver_type).and_then(|ti| {
                                if let crate::tast::core::TypeKind::Class { symbol_id, .. } =
                                    &ti.kind
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
                            let mir_func_name = format!("{}_{}", class_name, method_name);
                            debug!(
                                "[MIR WRAPPER INSTANCE] Routing {}.{} to {}",
                                class_name, method_name, mir_func_name
                            );

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
                                // Auto-box primitive args when the wrapper expects Ptr(U8).
                                let mut arg_regs = Vec::new();
                                for (i, arg) in call_args.iter().enumerate() {
                                    if let Some(reg) = self.lower_expression(arg) {
                                        let actual_ty = self.convert_type(arg.ty);
                                        let expected_ty = mir_param_types
                                            .get(i)
                                            .cloned()
                                            .unwrap_or_else(|| actual_ty.clone());

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

                                let mir_func_id = self.register_stdlib_mir_forward_ref(
                                    &mir_func_name,
                                    mir_param_types,
                                    mir_return_type.clone(),
                                );

                                debug!(
                                "[MIR WRAPPER INSTANCE] Registered forward ref to {} with ID {:?}",
                                mir_func_name, mir_func_id
                            );

                                let call_result = self.builder.build_call_direct(
                                    mir_func_id,
                                    arg_regs,
                                    mir_return_type.clone(),
                                )?;

                                // Unbox when the wrapper returns Ptr(U8) but HIR expects a
                                // primitive.
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
        }
        *fell_through = true;
        None
    }

    pub(crate) fn try_stdlib_mapped_method_call(
        &mut self,
        expr: &HirExpr,
        stdlib_info: Option<(
            &'static str,
            &'static str,
            crate::stdlib::RuntimeFunctionCall,
        )>,
        result_type: IrType,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("try_stdlib_mapped_method_call on a non-Call expression")
        };
        let HirExprKind::Field { object, field } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let stdlib_info = stdlib_info.as_ref().map(|(c, m, r)| (*c, *m, r));
        if let Some((class_name, method_name, runtime_call)) = stdlib_info {
            let mut runtime_func_owned = runtime_call.runtime_name.to_string();
            let is_mir_wrapper = runtime_call.is_mir_wrapper;
            let raw_value_params = runtime_call.raw_value_params;
            let extend_to_i64_params = runtime_call.extend_to_i64_params;
            let returns_raw_value = runtime_call.returns_raw_value;
            let has_return = runtime_call.has_return;
            let explicit_return_type = runtime_call.return_type.map(|rt| rt.to_ir_type());
            let has_self_param = runtime_call.has_self_param;

            // Size-correct Ptr<T> wrappers: the default Ptr_offset/deref/write are
            // size-erased (T treated as 8 bytes). A narrow pointee redirects to the
            // sized variant registered in systems.rs; unknown, generic or >=8-byte
            // pointees keep the default name.
            if matches!(
                runtime_func_owned.as_str(),
                "Ptr_offset" | "Ptr_deref" | "Ptr_write"
            ) {
                let pointee = {
                    let type_table = self.type_table;
                    type_table.get(object.ty).and_then(|ti| match &ti.kind {
                        crate::tast::TypeKind::Class { type_args, .. }
                        | crate::tast::TypeKind::GenericInstance { type_args, .. } => {
                            if !type_args.is_empty() {
                                Some(self.convert_type(type_args[0]))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    })
                };
                if let Some(pointee) = pointee {
                    let suffix = match &pointee {
                        IrType::F32 => "_4f",
                        IrType::I32 | IrType::U32 => "_4",
                        IrType::U8 | IrType::I8 | IrType::Bool => "_1",
                        _ => "",
                    };
                    if !suffix.is_empty() {
                        runtime_func_owned.push_str(suffix);
                    }
                }
            }
            let runtime_func: &str = &runtime_func_owned;

            // Some runtime calls need custom lowering (Type.typeof returns a boxed
            // ValueType enum).
            if let Some(special_result) = self.try_lower_special_runtime_call(
                runtime_func,
                args,
                result_type.clone(),
                expr.source_location,
            ) {
                return special_result;
            }

            // Reflect.compare redirects to haxe_reflect_compare_typed with a type tag.
            // Must run before the generic arg boxing loop below.
            if runtime_func == "haxe_reflect_compare" && args.len() >= 2 {
                let type_info = self.infer_reflect_compare_type_info(args);
                if let Some(info) = type_info {
                    let mut typed_args = Vec::new();
                    for arg in args.iter() {
                        if let Some(reg) = self.lower_expression(arg) {
                            typed_args.push(self.erase_reflect_compare_arg(reg));
                        }
                    }
                    let tag_reg = match info {
                        Ok(tag_value) => self.builder.build_const(IrValue::I32(tag_value))?,
                        Err(type_param_name) => {
                            let tag = self.builder.build_const(IrValue::I32(0))?;
                            if let Some(func) = self.builder.current_function_mut() {
                                func.type_param_tag_fixups.push((tag, type_param_name));
                            }
                            tag
                        }
                    };
                    typed_args.push(tag_reg);
                    let extern_func_id = self.get_or_register_extern_function(
                        "haxe_reflect_compare_typed",
                        vec![IrType::I64, IrType::I64, IrType::I32],
                        IrType::I64,
                    );
                    let call_result =
                        self.builder
                            .build_call_direct(extern_func_id, typed_args, IrType::I64)?;
                    if result_type == IrType::I32 {
                        return self
                            .builder
                            .build_cast(call_result, IrType::I64, IrType::I32);
                    }
                    return Some(call_result);
                }
            }

            // Generic classes like Deque<T> take boxed pointers even when the HIR
            // types are primitives, so the signature decides, not HIR.
            let (expected_param_types, actual_return_type) = self
                .get_stdlib_mir_wrapper_signature(runtime_func)
                .map(|(params, ret)| (params, ret))
                .unwrap_or_else(|| {
                    // Fallback: derive from arguments, using stdlib mapping hints
                    let mut params = vec![IrType::Ptr(Box::new(IrType::U8))];
                    for (i, arg) in args.iter().enumerate() {
                        // In these bitmasks bit 0 is self, bit i+1 is user arg i.
                        let param_bit = 1u32 << (i + 1);
                        if raw_value_params & param_bit != 0 {
                            params.push(IrType::U64);
                        } else if extend_to_i64_params & param_bit != 0 {
                            params.push(IrType::I64);
                        } else {
                            params.push(self.convert_type(arg.ty));
                        }
                    }
                    // An explicit descriptor return type wins over inference.
                    let ret_type = if let Some(ref rt) = explicit_return_type {
                        rt.clone()
                    } else if returns_raw_value {
                        IrType::U64
                    } else if has_return {
                        result_type.clone()
                    } else {
                        IrType::Void
                    };
                    (params, ret_type)
                });
            debug!(
                "[Extern method redirect] expected params: {:?}, return type: {:?}",
                expected_param_types, actual_return_type
            );

            // For static stdlib methods (StringTools.startsWith via `using`) the object
            // is a class reference, not a receiver, and `using` desugaring already put
            // the real receiver in args — so don't prepend it as 'this'.
            let is_static_stdlib = !has_self_param;

            let mut arg_regs = if is_static_stdlib {
                Vec::new()
            } else {
                let obj_reg = self.lower_expression(object)?;
                vec![obj_reg] // 'this' as first arg
            };
            for (i, arg) in args.iter().enumerate() {
                let arg_reg = self.lower_expression(arg)?;
                let actual_ty = self.convert_type(arg.ty);

                // Instance methods offset by 1 for 'this'.
                let param_idx = if is_static_stdlib { i } else { i + 1 };
                let expected_ty = expected_param_types
                    .get(param_idx)
                    .cloned()
                    .unwrap_or_else(|| actual_ty.clone());

                // Auto-box if needed (Int -> Ptr(U8) for Deque<Int>.add()).
                let final_reg =
                    self.maybe_box_for_extern_call(arg_reg, &actual_ty, &expected_ty)?;
                arg_regs.push(final_reg);
            }

            // Runtime enum helpers (enumEq, enumConstructor, enumParameters, getEnum)
            // take a hidden type_id argument.
            self.inject_hidden_enum_type_id_arg(runtime_func, args, &mut arg_regs);

            let param_types = if expected_param_types.len() == arg_regs.len() {
                expected_param_types.clone()
            } else {
                let mut params = if is_static_stdlib {
                    Vec::new()
                } else {
                    vec![IrType::Ptr(Box::new(IrType::U8))]
                };
                for arg in args {
                    params.push(self.convert_type(arg.ty));
                }
                params
            };

            let call_result = if is_mir_wrapper {
                let mir_func_id = self.register_stdlib_mir_forward_ref(
                    runtime_func,
                    param_types,
                    actual_return_type.clone(),
                );
                self.builder
                    .build_call_direct(mir_func_id, arg_regs, actual_return_type.clone())?
            } else {
                let extern_func_id = self.get_or_register_extern_function(
                    runtime_func,
                    param_types,
                    actual_return_type.clone(),
                );
                self.builder.build_call_direct(
                    extern_func_id,
                    arg_regs,
                    actual_return_type.clone(),
                )?
            };

            // Unbox when the runtime returns Ptr(U8) but HIR expects a primitive; for
            // generic classes like Channel<Int>, resolve T from the receiver's type args.
            let resolved_expected = {
                let needs_resolve = result_type == IrType::Any
                    || matches!(&result_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void))
                    || result_type == IrType::I64;
                if needs_resolve {
                    let type_table = self.type_table;
                    type_table
                        .get(object.ty)
                        .and_then(|ti| match &ti.kind {
                            crate::tast::TypeKind::Class { type_args, .. }
                            | crate::tast::TypeKind::GenericInstance { type_args, .. } => {
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
            // Hint the result register so later calls dispatch on the returned class
            // (Array.iterator() -> ArrayIterator for it.hasNext()/it.next()).
            //
            // MIR wrappers return their declared type directly, not a boxed
            // DynamicValue*, so skip unboxing for them or Host.localhost()'s raw
            // string pointer gets dereferenced.
            let final_result = if is_mir_wrapper {
                // array_pop returns raw I64 — cast to Ptr(Void) for class types.
                let expects_class_ptr = matches!(&resolved_expected, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::Void));
                if actual_return_type == IrType::I64 && expects_class_ptr {
                    self.builder.build_cast(
                        call_result,
                        IrType::I64,
                        IrType::Ptr(Box::new(IrType::Void)),
                    )
                } else if actual_return_type == IrType::I64 && result_type == IrType::I32 {
                    // stdlib returns usize/i64 but Haxe Int is i32.
                    self.builder
                        .build_cast(call_result, IrType::I64, IrType::I32)
                } else {
                    Some(call_result)
                }
            } else {
                self.maybe_unbox_for_extern_return(
                    call_result,
                    &actual_return_type,
                    &resolved_expected,
                )
            };
            if let Some(result_reg) = final_result {
                let return_class = self.get_return_class_hint(class_name, method_name);
                self.register_class_hints
                    .insert(result_reg, return_class.to_string());
                if let Some(h) = self.wrap_stdlib_iter_result(result_reg, expr.ty) {
                    return Some(h);
                }
            }
            return final_result;
        }
        *fell_through = true;
        None
    }
}
