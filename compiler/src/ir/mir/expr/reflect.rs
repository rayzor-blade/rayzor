//! `Reflect.*` and `Type.typeof` call sites.

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
    /// Lower a synthetic Reflect.makeVarArgs bridge.
    /// Current parity path preserves the original function value so it can
    /// still be invoked via Reflect.callMethod(args-array form).
    pub(crate) fn lower_reflect_make_var_args(
        &mut self,
        args: &[HirExpr],
        _result_type: IrType,
        location: SourceLocation,
    ) -> Option<IrId> {
        let Some(func_expr) = args.first() else {
            self.add_error("Reflect.makeVarArgs requires one argument", location);
            return None;
        };

        let func_val = self.lower_expression(func_expr)?;
        // Reflect.makeVarArgs returns Dynamic in stdlib API.
        // Box function/closure values as TYPE_FUNCTION for Reflect.isFunction parity.
        self.box_value_for_dynamic(func_val, func_expr.ty)
            .or(Some(func_val))
    }

    /// Lower Reflect.callMethod(o, func, argsArray) without a runtime trampoline by
    /// generating an indirect call from compile-time function type information.
    pub(crate) fn lower_reflect_call_method(
        &mut self,
        args: &[HirExpr],
        result_type: IrType,
        location: SourceLocation,
    ) -> Option<IrId> {
        if args.len() < 3 {
            self.add_error(
                "Reflect.callMethod requires (object, function, argsArray)",
                location,
            );
            return None;
        }

        let func_expr = &args[1];
        let args_array_expr = &args[2];
        let func_ptr = if let HirExprKind::Variable { symbol, .. } = &func_expr.kind {
            let resolved_func = self.get_function_id(symbol).or_else(|| {
                self.symbol_table
                    .get_symbol(*symbol)
                    .and_then(|sym| self.string_interner.get(sym.name))
                    .and_then(|name| {
                        self.find_function_by_name(name)
                            .or_else(|| self.find_external_function_by_name(name))
                    })
            });

            if let Some(func_id) = resolved_func {
                // CallIndirect expects a closure object {fn_ptr, env_ptr}, not a raw fn address.
                self.builder
                    .build_function_ref(func_id)
                    .or_else(|| self.lower_expression(func_expr))?
            } else {
                self.lower_expression(func_expr)?
            }
        } else {
            self.lower_expression(func_expr)?
        };

        // Fallback path for makeVarArgs-style functions where the function type is
        // erased/dynamic: call with a single Array argument.
        let fallback_single_array_call = |this: &mut Self| -> Option<IrId> {
            let arr_arg = this.lower_expression(args_array_expr)?;
            let sig = IrType::Function {
                params: vec![this.convert_type(args_array_expr.ty)],
                return_type: Box::new(result_type.clone()),
                varargs: false,
            };
            this.builder
                .build_call_indirect(func_ptr, vec![arr_arg], sig)
        };

        let lower_with_ir_signature = |this: &mut Self,
                                       param_ir_types: Vec<IrType>,
                                       return_ir: IrType|
         -> Option<IrId> {
            let args_array_reg = this.lower_expression(args_array_expr)?;
            let args_array_ptr =
                if let Some(actual_ty) = this.builder.get_register_type(args_array_reg) {
                    if actual_ty == IrType::Ptr(Box::new(IrType::Void)) {
                        args_array_reg
                    } else {
                        this.builder
                            .build_cast(
                                args_array_reg,
                                actual_ty,
                                IrType::Ptr(Box::new(IrType::Void)),
                            )
                            .unwrap_or(args_array_reg)
                    }
                } else {
                    args_array_reg
                };

            let call_args: Vec<IrId> = if param_ir_types.len() == 1 {
                // makeVarArgs-style callback: pass args array directly.
                let target_ty = param_ir_types[0].clone();
                let arr_arg =
                    if let Some(actual_ty) = this.builder.get_register_type(args_array_ptr) {
                        if actual_ty == target_ty {
                            args_array_ptr
                        } else {
                            this.builder
                                .build_cast(args_array_ptr, actual_ty, target_ty.clone())
                                .unwrap_or(args_array_ptr)
                        }
                    } else {
                        args_array_ptr
                    };
                vec![arr_arg]
            } else {
                let array_get_i64_fn = this.get_or_register_extern_function(
                    "haxe_array_get_i64",
                    vec![IrType::Ptr(Box::new(IrType::Void)), IrType::I64],
                    IrType::I64,
                );

                let mut unpacked = Vec::with_capacity(param_ir_types.len());
                for (idx, target_ty) in param_ir_types.iter().enumerate() {
                    let index_reg = this.builder.build_const(IrValue::I64(idx as i64))?;
                    let raw_val = this.builder.build_call_direct(
                        array_get_i64_fn,
                        vec![args_array_ptr, index_reg],
                        IrType::I64,
                    )?;
                    let coerced = match target_ty {
                        IrType::I64 => Some(raw_val),
                        IrType::F64 => this.builder.build_bitcast(raw_val, IrType::F64),
                        IrType::F32 => {
                            let as_i32 =
                                this.builder.build_cast(raw_val, IrType::I64, IrType::I32)?;
                            this.builder.build_bitcast(as_i32, IrType::F32)
                        }
                        _ => this
                            .builder
                            .build_cast(raw_val, IrType::I64, target_ty.clone()),
                    }
                    .unwrap_or(raw_val);
                    unpacked.push(coerced);
                }
                unpacked
            };

            let call_sig = IrType::Function {
                params: param_ir_types,
                return_type: Box::new(return_ir.clone()),
                varargs: false,
            };
            let call_result = this
                .builder
                .build_call_indirect(func_ptr, call_args, call_sig)?;

            if return_ir == result_type {
                return Some(call_result);
            }

            if matches!(result_type, IrType::Ptr(ref inner) if matches!(inner.as_ref(), IrType::Void))
            {
                if let Some(boxed) = this.box_primitive_to_dynamic(call_result, return_ir.clone()) {
                    return Some(boxed);
                }
            }

            if let Some(casted) =
                this.builder
                    .build_cast(call_result, return_ir.clone(), result_type.clone())
            {
                return Some(casted);
            }

            Some(call_result)
        };

        if let Some(IrType::Function {
            params,
            return_type,
            ..
        }) = self.builder.get_register_type(func_ptr)
        {
            return lower_with_ir_signature(self, params, *return_type);
        }

        let Some((param_type_ids, return_type_id)) =
            self.resolve_function_type_signature(func_expr.ty)
        else {
            return fallback_single_array_call(self);
        };

        // Special-case Array<Dynamic> callback shape (used by makeVarArgs): pass args array directly.
        if param_type_ids.len() == 1 {
            let type_table = self.type_table;
            let takes_array = type_table
                .get(param_type_ids[0])
                .map(|ti| matches!(ti.kind, crate::tast::TypeKind::Array { .. }))
                .unwrap_or(false);
            if takes_array {
                let arr_arg = self.lower_expression(args_array_expr)?;
                let call_sig = IrType::Function {
                    params: vec![self.convert_type(param_type_ids[0])],
                    return_type: Box::new(self.convert_type(return_type_id)),
                    varargs: false,
                };
                let result = self
                    .builder
                    .build_call_indirect(func_ptr, vec![arr_arg], call_sig)?;
                if self.convert_type(return_type_id) == result_type {
                    return Some(result);
                }
                if matches!(result_type, IrType::Ptr(ref inner) if matches!(inner.as_ref(), IrType::Void))
                {
                    if let Some(boxed) = self.box_value_for_dynamic(result, return_type_id) {
                        return Some(boxed);
                    }
                }
                if let Some(casted) = self.builder.build_cast(
                    result,
                    self.convert_type(return_type_id),
                    result_type.clone(),
                ) {
                    return Some(casted);
                }
                return Some(result);
            }
        }

        // General typed path: unpack argsArray[i] as raw i64 slots then coerce into parameter types.
        let mut call_args: Vec<IrId> = Vec::new();
        let args_array_reg = self.lower_expression(args_array_expr)?;
        let args_array_ptr = if let Some(actual_ty) = self.builder.get_register_type(args_array_reg)
        {
            if actual_ty == IrType::Ptr(Box::new(IrType::Void)) {
                args_array_reg
            } else {
                self.builder
                    .build_cast(
                        args_array_reg,
                        actual_ty,
                        IrType::Ptr(Box::new(IrType::Void)),
                    )
                    .unwrap_or(args_array_reg)
            }
        } else {
            args_array_reg
        };

        let array_get_i64_fn = self.get_or_register_extern_function(
            "haxe_array_get_i64",
            vec![IrType::Ptr(Box::new(IrType::Void)), IrType::I64],
            IrType::I64,
        );

        for (idx, param_ty_id) in param_type_ids.iter().enumerate() {
            let index_reg = self.builder.build_const(IrValue::I64(idx as i64))?;
            let raw_val = self.builder.build_call_direct(
                array_get_i64_fn,
                vec![args_array_ptr, index_reg],
                IrType::I64,
            )?;
            let coerced = self
                .coerce_from_i64(raw_val, *param_ty_id)
                .unwrap_or(raw_val);
            call_args.push(coerced);
        }

        let func_signature = IrType::Function {
            params: param_type_ids
                .iter()
                .map(|ty| self.convert_type(*ty))
                .collect(),
            return_type: Box::new(self.convert_type(return_type_id)),
            varargs: false,
        };

        let call_result = self
            .builder
            .build_call_indirect(func_ptr, call_args, func_signature)?;

        let actual_return_ir = self.convert_type(return_type_id);
        if actual_return_ir == result_type {
            return Some(call_result);
        }

        // callMethod returns Dynamic in stdlib API; ensure primitive returns are boxed.
        if matches!(result_type, IrType::Ptr(ref inner) if matches!(inner.as_ref(), IrType::Void)) {
            if let Some(boxed) = self.box_value_for_dynamic(call_result, return_type_id) {
                return Some(boxed);
            }
        }

        if let Some(casted) =
            self.builder
                .build_cast(call_result, actual_return_ir, result_type.clone())
        {
            return Some(casted);
        }

        Some(call_result)
    }

    /// Lower `Type.typeof(v)` with ValueType parity.
    ///
    /// Runtime `haxe_type_typeof` returns an ordinal (`i32`). When the call
    /// result is used as `ValueType`, we build the actual enum value through
    /// `haxe_type_typeof_value`.
    pub(crate) fn lower_type_typeof_call(
        &mut self,
        args: &[HirExpr],
        _result_type: IrType,
    ) -> Option<IrId> {
        if args.len() != 1 {
            return None;
        }
        let value_reg = self.lower_expression(&args[0])?;
        let actual_ty = self
            .builder
            .get_register_type(value_reg)
            .unwrap_or_else(|| self.convert_type(args[0].ty));
        let dynamic_ptr_ty = IrType::Ptr(Box::new(IrType::U8));
        // A statically-`Dynamic` argument is ALREADY a Dynamic pointer, so use it
        // directly. `box_value_for_dynamic` returns `None` for a Dynamic value
        // (its "no boxing needed" signal); routing that through the `?` below
        // would mis-read it as failure and drop the whole `typeof` result,
        // emitting a "Return with no value" trap-stub (W0020) for any
        // `Type.typeof(v)` where `v : Dynamic`. Concrete values still get boxed.
        let arg_is_dynamic = matches!(
            self.type_table.get(args[0].ty).map(|t| &t.kind),
            Some(crate::tast::TypeKind::Dynamic)
        );
        let value_dyn = if arg_is_dynamic || actual_ty == dynamic_ptr_ty {
            value_reg
        } else {
            self.box_value_for_dynamic(value_reg, args[0].ty)?
        };

        // Always use the boxed ValueType path (haxe_type_typeof_value) so that
        // parameterized variants like TClass(c) and TEnum(e) carry their type_id
        // payload. The I32 ordinal path loses this information, causing pattern
        // matching like `case TEnum(_):` to fail.
        let value_fn = self.get_or_register_extern_function(
            "haxe_type_typeof_value",
            vec![dynamic_ptr_ty.clone()],
            IrType::I64,
        );
        let result = self
            .builder
            .build_call_direct(value_fn, vec![value_dyn], IrType::I64)?;
        // The I64 return is actually a pointer to a heap-allocated boxed enum.
        // Cast to Ptr(U8) so pattern matching treats it as a boxed enum pointer
        // rather than a plain integer (which would skip the tag-load path).
        self.builder.build_cast(result, IrType::I64, dynamic_ptr_ty)
    }
}
