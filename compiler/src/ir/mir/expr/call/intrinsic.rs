//! Calls the compiler lowers itself rather than dispatching: `trace` and `Std.string`.

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
    pub(crate) fn try_trace_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call { callee, args, .. } = &expr.kind else {
            unreachable!("try_trace_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let symbol_name = self
            .symbol_table
            .get_symbol(*symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("<unknown>");
        if symbol_name == "trace" && args.len() == 1 {
            let arg = &args[0];

            // Route trace(Type.typeof(x)) to enum tracing directly.
            // This preserves parity even when the call-site type was widened.
            if let Some(typeof_arg) = self.trace_typeof_inner_arg(arg) {
                let value_reg =
                    self.lower_type_typeof_call(std::slice::from_ref(typeof_arg), IrType::I64)?;
                let trace_typeof_id = self.get_or_register_extern_function(
                    "haxe_trace_value_type",
                    vec![IrType::I64],
                    IrType::Void,
                );
                return self.builder.build_call_direct(
                    trace_typeof_id,
                    vec![value_reg],
                    IrType::Void,
                );
            }

            // Handle ValueType values that were previously produced/stored.
            if self.expr_is_value_type_expr(arg) {
                let arg_reg = self.lower_expression(arg)?;
                let trace_typeof_id = self.get_or_register_extern_function(
                    "haxe_trace_value_type",
                    vec![IrType::I64],
                    IrType::Void,
                );
                return self.builder.build_call_direct(
                    trace_typeof_id,
                    vec![arg_reg],
                    IrType::Void,
                );
            }

            // Check if arg is a class or enum type
            // For classes: try to call toString() method
            // For enums: for now, fall through to traceAny (enum toString not yet implemented)
            let type_table = self.type_table;
            let type_kind = type_table.get(arg.ty).map(|ti| ti.kind.clone());

            debug!(
                "[TRACE ARG TYPE] arg.ty={:?}, type_kind={:?}",
                arg.ty, type_kind
            );

            let class_info =
                if let Some(crate::tast::core::TypeKind::Class { symbol_id, .. }) = &type_kind {
                    // Skip extern abstracts (CString, Usize, Ptr, etc.)
                    // — they appear as Class in the type table but don't have toString()
                    // Get class name for stdlib lookup
                    let class_name_str = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|s| self.string_interner.get(s.name))
                        .unwrap_or("");

                    let is_extern = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .map(|s| s.flags.contains(crate::tast::symbols::SymbolFlags::EXTERN))
                        .unwrap_or(false);

                    // Skip extern classes UNLESS they have a toString in stdlib_mapping
                    // (e.g., StringMap, IntMap, Date have stdlib toString methods)
                    let has_stdlib_tostring = self
                        .stdlib_mapping
                        .find_by_name(class_name_str, "toString")
                        .is_some();

                    if is_extern && !has_stdlib_tostring {
                        None
                    } else {
                        Some(class_name_str.to_string())
                    }
                } else {
                    None
                };

            // Check if the trace argument is an enum variant expression (e.g., Color.Red)
            // If so, we can print the variant name directly
            if let HirExprKind::Field { object, field } = &arg.kind {
                if let HirExprKind::Variable {
                    symbol: enum_symbol,
                    ..
                } = &object.kind
                {
                    if let Some(enum_sym) = self.symbol_table.get_symbol(*enum_symbol) {
                        use crate::tast::SymbolKind;
                        if enum_sym.kind == SymbolKind::Enum {
                            // Get the variant name
                            let field_sym = self.symbol_table.get_symbol(*field);
                            if let Some(variant_name) =
                                field_sym.and_then(|s| self.string_interner.get(s.name))
                            {
                                // Create a string constant with the variant name
                                // IrValue::String will be converted by Cranelift to call haxe_string_literal
                                // which returns a *mut HaxeString pointer
                                let variant_name_str = variant_name.to_string();
                                let string_ptr = self
                                    .builder
                                    .build_const(IrValue::String(variant_name_str))?;

                                // Get or create the string trace function
                                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                                let string_trace_id = self.get_or_register_extern_function(
                                    "haxe_trace_string_struct",
                                    vec![string_ptr_ty],
                                    IrType::Void,
                                );

                                // Trace the string
                                return self.builder.build_call_direct(
                                    string_trace_id,
                                    vec![string_ptr],
                                    IrType::Void,
                                );
                            }
                        }
                    }
                }
            }

            // Check if it's an enum variable - print discriminant for now
            // Full variant name lookup for variables would require runtime RTTI
            // Direct enum variant expressions (Color.Red) are handled above

            // If this is a class type, try to call toString() on it
            if class_info.is_some() {
                let obj_reg = self.lower_expression(arg)?;
                if let Some(string_reg) = self.try_call_tostring(obj_reg, arg.ty)? {
                    let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                    let string_trace_id = self.get_or_register_extern_function(
                        "haxe_trace_string_struct",
                        vec![string_ptr_ty],
                        IrType::Void,
                    );
                    return self.builder.build_call_direct(
                        string_trace_id,
                        vec![string_reg],
                        IrType::Void,
                    );
                }
            }

            // Lower the argument first to get the actual MIR register
            // Check if this is a field access
            let is_field = matches!(&arg.kind, HirExprKind::Field { .. });
            if is_field {
                if let HirExprKind::Field { object, field } = &arg.kind {
                    let field_sym = self.symbol_table.get_symbol(*field);
                    let field_name = field_sym
                        .and_then(|s| self.string_interner.get(s.name))
                        .unwrap_or("<unknown>");
                    debug!("[TRACE] Argument is Field access: field={}", field_name);

                    // Check what the object is
                    if let HirExprKind::Variable { symbol, .. } = &object.kind {
                        let var_sym = self.symbol_table.get_symbol(*symbol);
                        let var_name = var_sym
                            .and_then(|s| self.string_interner.get(s.name))
                            .unwrap_or("<unknown>");
                        debug!("[TRACE] Field object is Variable: {}", var_name);
                    }
                }
            }
            let arg_reg = self.lower_expression(arg)?;
            debug!(
                "[TRACE] After lowering, arg_reg={}, checking type...",
                arg_reg
            );
            if let Some(ty) = self.builder.get_register_type(arg_reg) {
                debug!("[TRACE] arg_reg type from builder: {:?}", ty);
            }

            // Check if the HIR type is an enum
            // Also check if the arg is a variable and look up its declared type
            // (trace() takes Dynamic, so arg.ty might be Dynamic even if the variable is an enum)
            let type_table = self.type_table;
            let mut hir_type_kind = type_table.get(arg.ty).map(|ti| ti.kind.clone());

            // If arg.ty is Dynamic but the argument is a variable, look up the variable's declared type
            // This is needed because trace() accepts Dynamic, so the expression type might be Dynamic
            // even when the underlying variable has a more specific type (like an enum)
            if matches!(
                &hir_type_kind,
                Some(crate::tast::core::TypeKind::Dynamic) | None
            ) {
                if let HirExprKind::Variable { symbol, .. } = &arg.kind {
                    if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                        let var_type_kind = type_table.get(sym.type_id).map(|ti| ti.kind.clone());
                        if var_type_kind.is_some() {
                            hir_type_kind = var_type_kind;
                        }
                    }
                }
            }

            // Handle enum variables - use RTTI-based trace with compile-time type_id
            // Direct enum variant expressions (Color.Red) are handled above and print variant names
            if let Some(crate::tast::core::TypeKind::Enum {
                symbol_id,
                ref type_args,
            }) = hir_type_kind
            {
                if self.symbol_table.get_symbol(symbol_id).is_some() {
                    let enum_type_id = self.enum_runtime_id(symbol_id);

                    // Build type_id constant (u32)
                    let type_id_const = self
                        .builder
                        .build_const(IrValue::I32(enum_type_id as i32))?;

                    // Check if enum is boxed (has parameterized variants)
                    // Boxed enums store a pointer to heap-allocated struct
                    // Unboxed enums store just the discriminant as i64
                    if self.enum_is_boxed(symbol_id) {
                        // Resolve concrete param types from type_args (type inference)
                        // type_args maps type parameters to concrete types
                        let concrete_type_args: Vec<u8> = {
                            let type_table = self.type_table;
                            type_args
                                .iter()
                                .map(|&ta| match type_table.get(ta).map(|ti| &ti.kind) {
                                    Some(crate::tast::core::TypeKind::Int) => 0u8,
                                    Some(crate::tast::core::TypeKind::Float) => 1u8,
                                    Some(crate::tast::core::TypeKind::Bool) => 2u8,
                                    Some(crate::tast::core::TypeKind::String) => 3u8,
                                    Some(crate::tast::core::TypeKind::TypeParameter { .. }) => 5u8,
                                    Some(crate::tast::core::TypeKind::Dynamic) => 5u8,
                                    _ => 4u8,
                                })
                                .collect()
                        };

                        // If we have concrete type args, use the typed trace
                        if !concrete_type_args.is_empty()
                            && concrete_type_args.iter().any(|&t| t != 5)
                        {
                            let trace_typed_id = self.get_or_register_extern_function(
                                "haxe_trace_enum_boxed_typed",
                                vec![
                                    IrType::I32,
                                    IrType::Ptr(Box::new(IrType::I8)),
                                    IrType::Ptr(Box::new(IrType::I8)),
                                    IrType::I64,
                                ],
                                IrType::Void,
                            );

                            let ptr_reg = self
                                .builder
                                .build_bitcast(arg_reg, IrType::Ptr(Box::new(IrType::I8)))?;

                            // Build param types data via heap alloc + stores
                            let alloc_size = self
                                .builder
                                .build_const(IrValue::I64(concrete_type_args.len() as i64))?;
                            let alloc_func = self.get_or_register_extern_function(
                                "malloc",
                                vec![IrType::I64],
                                IrType::Ptr(Box::new(IrType::I8)),
                            );
                            let param_types_data = self.builder.build_call_direct(
                                alloc_func,
                                vec![alloc_size],
                                IrType::Ptr(Box::new(IrType::I8)),
                            )?;
                            for (i, &ptype) in concrete_type_args.iter().enumerate() {
                                let offset = self.builder.build_const(IrValue::I64(i as i64))?;
                                let elem_ptr = self.builder.build_gep(
                                    param_types_data,
                                    vec![offset],
                                    IrType::Ptr(Box::new(IrType::I8)),
                                )?;
                                let val = self.builder.build_const(IrValue::I8(ptype as i8))?;
                                self.builder.build_store(elem_ptr, val);
                            }
                            let param_count = self
                                .builder
                                .build_const(IrValue::I64(concrete_type_args.len() as i64))?;

                            return self.builder.build_call_direct(
                                trace_typed_id,
                                vec![type_id_const, ptr_reg, param_types_data, param_count],
                                IrType::Void,
                            );
                        }

                        // Fallback: use untyped boxed trace
                        let trace_enum_boxed_id = self.get_or_register_extern_function(
                            "haxe_trace_enum_boxed",
                            vec![IrType::I32, IrType::Ptr(Box::new(IrType::I8))],
                            IrType::Void,
                        );

                        let ptr_reg = self
                            .builder
                            .build_bitcast(arg_reg, IrType::Ptr(Box::new(IrType::I8)))?;

                        return self.builder.build_call_direct(
                            trace_enum_boxed_id,
                            vec![type_id_const, ptr_reg],
                            IrType::Void,
                        );
                    } else {
                        // Unboxed enum: arg_reg holds the discriminant (i64)
                        // Call haxe_trace_enum(type_id: u32, discriminant: i64)
                        let trace_enum_id = self.get_or_register_extern_function(
                            "haxe_trace_enum",
                            vec![IrType::I32, IrType::I64],
                            IrType::Void,
                        );

                        return self.builder.build_call_direct(
                            trace_enum_id,
                            vec![type_id_const, arg_reg],
                            IrType::Void,
                        );
                    }
                }
            }

            // Get the actual MIR type from the register (not the HIR type)
            // This is important because HIR types may be vague (Ptr(Void)) but
            // MIR registers have the actual type (String, etc.)
            let actual_reg_type = self
                .builder
                .get_register_type(arg_reg)
                .unwrap_or_else(|| self.convert_type(arg.ty));

            let mut arg_type = actual_reg_type.clone();
            // If the MIR type is Ptr(Void) but we have better type info from the symbol,
            // use the symbol's type instead. This handles cases like trace(t) where t is
            // a float from Sys.time() but the trace() signature says Dynamic.
            // BUT: don't override Ptr(U8) — that means a boxed DynamicValue* (e.g., from
            // Array.pop() returning Null<T>), which traceAny can properly unbox.
            let is_boxed_dynamic =
                matches!(&arg_type, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
            if matches!(arg_type, IrType::Ptr(_)) && !is_boxed_dynamic {
                if let Some(ref type_kind) = hir_type_kind {
                    let better_type = match type_kind {
                        crate::tast::core::TypeKind::Float => Some(IrType::F64),
                        crate::tast::core::TypeKind::Int => Some(IrType::I64),
                        crate::tast::core::TypeKind::Bool => Some(IrType::Bool),
                        crate::tast::core::TypeKind::String => Some(IrType::String),
                        _ => None,
                    };
                    if let Some(better) = better_type {
                        arg_type = better;
                    }
                }
            }

            // Check if this is an Array type from HIR type info
            let is_array_type = matches!(
                &hir_type_kind,
                Some(crate::tast::core::TypeKind::Array { .. })
            );

            // For Array types, call haxe_trace_array directly
            if is_array_type {
                let trace_array_id = self.get_or_register_extern_function(
                    "haxe_trace_array",
                    vec![IrType::Ptr(Box::new(IrType::Void))],
                    IrType::Void,
                );
                return self
                    .builder
                    .build_call_direct(trace_array_id, vec![arg_reg], IrType::Void);
            }

            // Handle TypeParameter types that are still type-erased (I64).
            // Only activate when the register type didn't reveal the concrete type.
            // Inside generic functions, emit a fixup for the monomorphize pass.
            // Outside generic functions (shouldn't normally happen with proper type
            // resolution), fall through to traceInt.
            if matches!(arg_type, IrType::I64 | IrType::I32)
                && matches!(
                    &hir_type_kind,
                    Some(crate::tast::core::TypeKind::TypeParameter { .. })
                )
            {
                if let Some(crate::tast::core::TypeKind::TypeParameter { symbol_id, .. }) =
                    &hir_type_kind
                {
                    let type_param_name = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|sym| self.string_interner.get(sym.name))
                        .map(|s| s.to_string());

                    if let Some(ref tp_name) = type_param_name {
                        // Only emit a tag fixup if the current function actually
                        // has this type parameter (i.e., we're inside a generic function).
                        // If not, the fixup would never be resolved, so fall through
                        // to normal trace dispatch instead.
                        let current_func_has_param = self
                            .builder
                            .current_function()
                            .map(|f| f.signature.type_params.iter().any(|tp| tp.name == *tp_name))
                            .unwrap_or(false);

                        if current_func_has_param {
                            let tag_reg = self.builder.build_const(IrValue::I32(0))?;
                            if let Some(func) = self.builder.current_function_mut() {
                                func.type_param_tag_fixups.push((tag_reg, tp_name.clone()));
                            }

                            let trace_typed_id = self.get_or_register_extern_function(
                                "haxe_trace_typed",
                                vec![IrType::I64, IrType::I32],
                                IrType::Void,
                            );

                            return self.builder.build_call_direct(
                                trace_typed_id,
                                vec![arg_reg, tag_reg],
                                IrType::Void,
                            );
                        }
                        // If not in a generic function, fall through to normal dispatch.
                    }
                }
            }

            // Special case: Optional<primitive> returned from MIR wrappers (e.g. array pop/shift)
            // MIR wrappers cast DynamicValue* to IrType::Any (I64), but the value is still a boxed pointer.
            // Detect via hir_type_kind and route to traceAny for proper unboxing.
            // BUT: extern functions with returns_raw_value (e.g., StringMap.get) return the
            // actual value bits as I64, NOT a boxed pointer. Nested Call expressions produce
            // these raw values, so skip is_optional_boxed for Call args.
            let is_optional_boxed = matches!(&arg_type, IrType::I64 | IrType::I32)
                && matches!(
                    &hir_type_kind,
                    Some(crate::tast::core::TypeKind::Optional { .. })
                )
                && !matches!(&arg.kind, HirExprKind::Call { .. });

            let trace_method = {
                match &arg_type {
                    IrType::I32 | IrType::I64 | IrType::U64 if is_optional_boxed => "traceAny",
                    IrType::I32 | IrType::I64 | IrType::U64 => "traceInt",
                    IrType::F32 | IrType::F64 => "traceFloat",
                    IrType::Bool => "traceBool",
                    IrType::String => "traceString", // String is ptr+len struct
                    // Also handle Ptr(String) - returned by String methods like toUpperCase()
                    IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String) => "traceString",
                    // Ptr(U8) when HIR type is String — from MIR wrappers returning raw string pointers
                    IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8) => {
                        let is_hir_string =
                            matches!(&hir_type_kind, Some(crate::tast::core::TypeKind::String));
                        if is_hir_string {
                            "traceString"
                        } else {
                            "traceAny"
                        }
                    }
                    IrType::TypeVar(_) => "traceTypedGeneric", // tag-based dispatch
                    _ => "traceAny", // Fallback for Dynamic or unknown types
                }
            };

            // Debug: Print trace method selection
            debug!(
                "[DEBUG trace] arg_reg={}, arg_type={:?}, trace_method={}",
                arg_reg, arg_type, trace_method
            );

            // Build the qualified name for the trace function
            let trace_func_name = format!("rayzor.Trace.{}", trace_method);

            // Look up the runtime function name
            // For now, manually map to the runtime function
            let runtime_func = match trace_method {
                "traceInt" => "haxe_trace_int",
                "traceFloat" => "haxe_trace_float",
                "traceBool" => "haxe_trace_bool",
                "traceString" => "haxe_trace_string",
                "traceAny" => "haxe_trace_any",
                _ => "haxe_trace_any",
            };

            // Special handling for String: use haxe_trace_string_struct that takes a pointer
            if trace_method == "traceString" {
                // String is represented as a pointer to HaxeString struct
                let param_types = vec![IrType::Ptr(Box::new(IrType::String))];
                let string_trace_id = self.get_or_register_extern_function(
                    "haxe_trace_string_struct",
                    param_types,
                    IrType::Void,
                );
                return self.builder.build_call_direct(
                    string_trace_id,
                    vec![arg_reg],
                    IrType::Void,
                );
            }

            // TypeVar trace: use haxe_trace_typed with tag fixup.
            // Bitcast value to I64 (TypeVar is pointer-sized) to avoid
            // Cranelift type mismatch when inlining resolves to F64.
            if trace_method == "traceTypedGeneric" {
                let tag_reg = self.builder.build_const(IrValue::I32(0))?;
                if let IrType::TypeVar(ref name) = arg_type {
                    if let Some(func) = self.builder.current_function_mut() {
                        func.type_param_tag_fixups.push((tag_reg, name.clone()));
                    }
                }
                let val_as_i64 = self
                    .builder
                    .build_bitcast(arg_reg, IrType::I64)
                    .unwrap_or(arg_reg);
                let trace_typed_id = self.get_or_register_extern_function(
                    "haxe_trace_typed",
                    vec![IrType::I64, IrType::I32],
                    IrType::Void,
                );
                return self.builder.build_call_direct(
                    trace_typed_id,
                    vec![val_as_i64, tag_reg],
                    IrType::Void,
                );
            }

            // Get or register the extern runtime function
            // Note: Runtime trace functions expect specific types:
            // - haxe_trace_int expects i64
            // - haxe_trace_float expects f64
            // We need to cast arguments to match
            // Note: We don't need to cast arguments here - the Cranelift backend
            // handles signature-aware type conversion automatically (see cranelift_backend.rs:1487-1491)
            // It will insert sextend for i32->i64, fcvt for f32->f64, etc.
            let param_types = match trace_method {
                "traceInt" => vec![IrType::I64],
                "traceFloat" => vec![IrType::F64],
                "traceBool" => vec![IrType::Bool],
                _ => vec![arg_type.clone()],
            };

            // If Optional boxed value routed to traceAny, cast I64 back to pointer
            let final_arg_reg = if is_optional_boxed && trace_method == "traceAny" {
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                self.builder.build_cast(arg_reg, IrType::I64, ptr_u8)?
            } else {
                arg_reg
            };

            let final_param_types = if is_optional_boxed && trace_method == "traceAny" {
                vec![IrType::Ptr(Box::new(IrType::U8))]
            } else {
                param_types
            };

            let runtime_func_id =
                self.get_or_register_extern_function(runtime_func, final_param_types, IrType::Void);

            // Generate the call
            return self.builder.build_call_direct(
                runtime_func_id,
                vec![final_arg_reg],
                IrType::Void,
            );
        }
        *fell_through = true;
        None
    }

    pub(crate) fn try_std_string_call(
        &mut self,
        expr: &HirExpr,
        fell_through: &mut bool,
    ) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            ..
        } = &expr.kind
        else {
            unreachable!("try_std_string_call on a non-Call expression")
        };
        let HirExprKind::Variable { symbol, .. } = &callee.kind else {
            *fell_through = true;
            return None;
        };
        let symbol_name = self
            .symbol_table
            .get_symbol(*symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("<unknown>");
        if symbol_name == "string" && (args.len() == 1 || (args.len() == 2 && *is_method)) {
            debug!(
                "[STD.STRING CHECK] Found 'string' call, is_method={}, args.len()={}",
                is_method,
                args.len()
            );

            // For static method calls, the actual argument is the second one (skip Std class)
            let arg = if *is_method && args.len() == 2 {
                &args[1]
            } else {
                &args[0]
            };
            let arg_is_value_type = self.expr_is_value_type_expr(arg);

            // ValueType pretty-print parity path
            if arg_is_value_type {
                let arg_reg = self.lower_expression(arg)?;
                return self.convert_value_type_to_string(arg_reg);
            }

            let arg_type = self.convert_type(arg.ty);

            // Check HIR type for Array (TypeKind::Array maps to Ptr in MIR)
            let hir_type_kind = {
                let tt = self.type_table;
                tt.get(arg.ty).map(|ti| ti.kind.clone())
            };
            let is_array = matches!(hir_type_kind.as_ref(), Some(TypeKind::Array { .. }));

            if is_array {
                let arg_reg = self.lower_expression(arg)?;
                let conv_fn = self.get_or_register_extern_function(
                    "haxe_array_to_string",
                    vec![IrType::Ptr(Box::new(IrType::Void))],
                    IrType::Ptr(Box::new(IrType::String)),
                );
                return self.builder.build_call_direct(
                    conv_fn,
                    vec![arg_reg],
                    IrType::Ptr(Box::new(IrType::String)),
                );
            }

            // Determine which MIR wrapper function to call based on type
            // These wrappers call the extern runtime functions
            let mir_wrapper = match arg_type {
                IrType::I32 | IrType::I64 => "int_to_string",
                IrType::F32 | IrType::F64 => "float_to_string",
                IrType::Bool => "bool_to_string",
                IrType::String => "string_to_string",
                _ => "int_to_string",
            };

            debug!(
                "[STD.STRING] Routing Std.string() call to {} for type {:?}",
                mir_wrapper, arg_type
            );

            // Lower the argument
            let arg_reg = self.lower_expression(arg)?;

            // Get or register the MIR wrapper function
            // These return String (a struct with ptr + len)
            let param_types = vec![arg_type.clone()];
            let return_type = IrType::String; // String is represented as ptr+len
            let mir_wrapper_id =
                self.get_or_register_extern_function(mir_wrapper, param_types, return_type.clone());

            // Generate the call to MIR wrapper
            return self
                .builder
                .build_call_direct(mir_wrapper_id, vec![arg_reg], return_type);
        }
        *fell_through = true;
        None
    }
}
