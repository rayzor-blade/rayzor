//! Call lowering.
//!
//! The callee's shape decides the dispatch, so lowering is a chain of probes
//! ending in an indirect call through a function pointer. A probe reports "not
//! my shape" through `fell_through` rather than by returning `None`: `None`
//! alone means the shape matched but lowering failed, and no later probe may
//! then claim the call.

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

mod array;
mod derived;
mod enum_ctor;
mod forward_ref;
mod future;
mod indirect;
mod method;
mod native_struct;
mod shader;

/// Runs one call-shape probe, returning its result unless the probe reports
/// that the callee was not its shape.
macro_rules! probe {
    ($self:ident.$m:ident($($a:expr),* $(,)?)) => {{
        let mut fell_through = false;
        let lowered = $self.$m($($a,)* &mut fell_through);
        if !fell_through {
            return lowered;
        }
    }};
}

impl<'a> HirToMirContext<'a> {
    pub(crate) fn lower_call(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            type_args: hir_type_args,
            // Carried from TAST; the shape probes below still derive the
            // target themselves until they are replaced by a match on this.
            target: _resolved_target,
        } = &expr.kind
        else {
            unreachable!("lower_call on a non-Call expression")
        };
        // RAYZOR_PROBE_CALLTARGET=1 tabulates (target, callee shape) so
        // the carried target can be checked against what the shape
        // probes below discriminate on, before anything dispatches on it.
        if std::env::var_os("RAYZOR_PROBE_CALLTARGET").is_some() {
            let t = match _resolved_target {
                crate::ir::hir::CallTarget::Function => "Function",
                crate::ir::hir::CallTarget::Method { .. } => "Method",
                crate::ir::hir::CallTarget::Static { .. } => "Static",
            };
            let shape = match &callee.kind {
                HirExprKind::Field { object, .. } => match &object.kind {
                    HirExprKind::Variable { .. } => "Field(Variable)",
                    _ => "Field(other)",
                },
                HirExprKind::Variable { .. } => "Variable",
                HirExprKind::Super => "Super",
                _ => "other",
            };
            eprintln!("[calltarget] {} {} is_method={}", t, shape, is_method);
        }
        // @:shader wgsl() — intercept at Call entry point
        probe!(self.try_shader_call(expr));

        // Reset call_label for tracing which path generates the call
        self.builder.call_label = Some("CALL_START".to_string());
        let result_type = self.convert_type(expr.ty);

        // Update the caller's shadow-stack frame to this call-site line/col so
        // the trace shows WHERE the call was made, not the function definition line.
        let call_loc = expr.source_location;
        if call_loc.is_valid() && call_loc.line > 0 {
            let update_loc_fn = self.get_or_register_extern_function(
                "rayzor_update_call_frame_location",
                vec![IrType::I32, IrType::I32],
                IrType::Void,
            );
            if let (Some(line_c), Some(col_c)) = (
                self.builder.build_const(IrValue::I32(call_loc.line as i32)),
                self.builder
                    .build_const(IrValue::I32(call_loc.column as i32)),
            ) {
                self.builder
                    .build_call_direct(update_loc_fn, vec![line_c, col_c], IrType::Void);
            }
        }

        // Convert HIR type_args to IrType for use in CallDirect
        let converted_hir_type_args: Vec<IrType> = hir_type_args
            .iter()
            .map(|&ty_id| self.convert_type(ty_id))
            .collect();

        debug!(
            "[CALL] expr.ty={:?}, result_type={:?}, is_method={}",
            expr.ty, result_type, is_method
        );

        // @:async method dispatch: .await(), .poll(), .isReady()
        // on registers known to hold Future handles from async function calls.
        // MethodCall pattern: callee = Variable(method_symbol), args[0] = receiver
        probe!(self.try_future_method_call(expr));

        // Static synthetic calls resolved as Variable — find parent class
        probe!(self.try_native_struct_static_call(expr));

        if let HirExprKind::Variable { symbol, .. } = &callee.kind {
            let vname = self
                .symbol_table
                .get_symbol(*symbol)
                .and_then(|s| self.string_interner.get(s.name))
                .unwrap_or("?");
            debug!(
                "[CALL-VAR] callee='{}', is_method={}, args.len()={}",
                vname,
                is_method,
                args.len()
            );

            probe!(self.try_derived_instance_call(expr));

            probe!(self.try_array_runtime_call(expr));
        }
        {
            // DEBUG: check callee kind for localhost
        }
        probe!(self.try_method_call(expr, result_type.clone()));

        // Enum constructors can arrive as field callees for imported
        // modules, e.g. `ForeignMetaish.U32(2048)`. Lower those here
        // before the callee expression itself turns `Enum.Variant`
        // into a tag-only value and drops the payload arguments.
        probe!(self.try_enum_constructor_via_field(expr));

        // Check if callee is an enum constructor (EnumVariant symbol kind)
        // Handle enum constructors with parameters like MyResult.Ok(42)
        probe!(self.try_enum_constructor(expr));

        // Check if callee is a direct function reference
        if let HirExprKind::Variable { symbol, .. } = &callee.kind {
            // Virtual dispatch for instance method calls (is_method=true):
            // Skip vtable dispatch for super.method() calls — these must call
            // the parent's implementation directly, not the overridden version.
            let receiver_is_super = !args.is_empty() && matches!(args[0].kind, HirExprKind::Super);
            // super.method() — bypass vtable AND resolve to parent's implementation.
            if receiver_is_super {
                let method_name = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                if let Some(method_name) = method_name {
                    // Find parent class: determine which class the current function
                    // belongs to, then look up its parent via class_parent_map.
                    let current_class = self.builder.current_function().and_then(|f| {
                        // Find the class this function is a method of
                        self.class_method_by_name
                            .iter()
                            .find(|(_, &method_sym)| {
                                self.function_map.get(&method_sym) == Some(&f.id)
                            })
                            .map(|((class_sym, _), _)| *class_sym)
                    });
                    let parent_class =
                        current_class.and_then(|cls| self.class_parent_map.get(&cls).copied());
                    // Resolve parent's method by name
                    let super_func_id = parent_class
                        .and_then(|pc| {
                            self.class_method_by_name
                                .get(&(pc, method_name))
                                .and_then(|&sym| {
                                    self.resolve_function_id_with_qualified_fallback(sym)
                                })
                        })
                        .or_else(|| {
                            // Fallback: direct symbol resolution
                            self.resolve_function_id_with_qualified_fallback(*symbol)
                        });
                    if let Some(func_id) = super_func_id {
                        let obj_reg = self.lower_expression(&args[0])?;
                        let mut call_args = vec![obj_reg];
                        for arg in args.iter().skip(1) {
                            if let Some(reg) = self.lower_expression(arg) {
                                call_args.push(reg);
                            }
                        }
                        let ret_type = self.convert_type(expr.ty);
                        return self.builder.build_call_direct(func_id, call_args, ret_type);
                    }
                }
            }
            if *is_method && !args.is_empty() && !receiver_is_super {
                let vtable_slot = self.virtual_dispatch_info.get(symbol).copied().or_else(|| {
                    let method_name = self.symbol_table.get_symbol(*symbol).map(|s| s.name)?;
                    let receiver_type = self.resolve_through_aliases(args[0].ty);
                    let type_table = self.type_table;
                    let class_sym = match &type_table.get(receiver_type)?.kind {
                        TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    }?;
                    let mut current = Some(class_sym);
                    while let Some(cls) = current {
                        if let Some(&method_sym) =
                            self.class_method_by_name.get(&(cls, method_name))
                        {
                            if let Some(info) = self.virtual_dispatch_info.get(&method_sym) {
                                return Some(*info);
                            }
                        }
                        current = self.class_parent_map.get(&cls).copied();
                    }
                    None
                });

                if let Some((slot_index, _defining_class)) = vtable_slot {
                    let obj_reg = self.lower_expression(&args[0])?;
                    let mut call_args = vec![obj_reg];
                    for arg in args.iter().skip(1) {
                        if let Some(reg) = self.lower_expression(arg) {
                            call_args.push(reg);
                        }
                    }
                    let lookup_fn = self.get_or_register_extern_function(
                        "haxe_vtable_lookup",
                        vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                        IrType::I64,
                    );
                    let slot_reg = self.builder.build_const(IrValue::I32(slot_index as i32))?;
                    let fn_ptr = self.builder.build_call_direct(
                        lookup_fn,
                        vec![obj_reg, slot_reg],
                        IrType::I64,
                    )?;
                    let mut param_types = vec![IrType::Ptr(Box::new(IrType::Void))];
                    for arg in args.iter().skip(1) {
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
                        .build_call_indirect(fn_ptr, call_args, func_signature);
                }
            }

            let symbol_name = self
                .symbol_table
                .get_symbol(*symbol)
                .and_then(|s| self.string_interner.get(s.name))
                .unwrap_or("<unknown>");
            debug!(
                "DEBUG: Callee is Variable, symbol={:?} ({}), is_method={}, args.len()={}",
                symbol,
                symbol_name,
                is_method,
                args.len()
            );

            // DIRECT SYMBOL RESOLUTION:
            // For static extension methods (using IntTools; → x.add(3)) and
            // other user-defined method calls, try resolving the function by symbol ID first.
            // This avoids bare-name collisions (e.g., user "add" vs "rayzor_ssl_cert_add").
            // Only intercept for user-defined functions — extern/stdlib methods need the
            // more specific handlers below (auto-boxing, runtime mapping, etc.).
            //
            // IMPORTANT: Skip this fast path when the receiver is Dynamic or Interface-typed,
            // because those need special dispatch (unboxing / fat pointer extraction) handled below.
            if let Some(func_id) = self.resolve_function_id_with_qualified_fallback(*symbol) {
                let is_user_defined = self
                    .builder
                    .module
                    .functions
                    .get(&func_id)
                    .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                    .unwrap_or(false);

                // Check if receiver needs special dispatch (Dynamic unbox or Interface fat pointer)
                let receiver_needs_special_dispatch = if *is_method && !args.is_empty() {
                    let receiver_type = self.resolve_through_aliases(args[0].ty);
                    let type_table = self.type_table;
                    type_table
                        .get(receiver_type)
                        .map(|t| {
                            matches!(
                                t.kind,
                                TypeKind::Dynamic
                                    | TypeKind::Interface { .. }
                                    | TypeKind::TypeParameter { .. }
                                    | TypeKind::Placeholder { .. }
                                    | TypeKind::Unknown
                            )
                        })
                        .unwrap_or(false)
                } else {
                    false
                };

                // A generic instance method imported from another module
                // resolves here to a cross-module forward-ref stub whose
                // FunctionKind is not UserDefined (the real impl arrives via
                // merge + fixup later). The is_user_defined gate would route
                // it to the fallback path below, which does NOT attach the
                // receiver's concrete type_args — so the monomorphizer can
                // never specialize the imported generic method, and its whole
                // call chain (e.g. an imported haxe.ds.BalancedTree.set ->
                // setLoop -> balance -> compare) reaches codegen as generic
                // trap stubs and SIGILLs. Route generic instance-method calls
                // through the type_args-aware block regardless of kind, as
                // long as the callee is not a genuine extern/intrinsic.
                let (callee_has_type_params, callee_is_externish) = self
                    .builder
                    .module
                    .functions
                    .get(&func_id)
                    .map(|f| {
                        (
                            !f.signature.type_params.is_empty(),
                            matches!(
                                f.kind,
                                crate::ir::functions::FunctionKind::ExternC
                                    | crate::ir::functions::FunctionKind::Intrinsic
                            ),
                        )
                    })
                    .unwrap_or((false, false));
                // A callee registered as an extern (e.g. the iterator-protocol
                // methods List.iterator / .keys, which lower to extern Imports
                // resolved at link time) must NOT take the receiver-type_args
                // route: attaching type_args would ask the monomorphizer to
                // specialize an extern that has no body, producing an
                // unresolvable `Import` symbol that makes finalize panic on the
                // whole module. Real cross-module methods (BalancedTree.set) are
                // not in extern_functions, so they still route correctly.
                let callee_is_externish = callee_is_externish
                    || self.builder.module.extern_functions.contains_key(&func_id);
                // The callee may be a cross-module function not yet present
                // in this module (resolved to its eventual id but merged
                // later), so callee_has_type_params is unreliable here. Detect
                // genericity from the RECEIVER instead: a concrete generic
                // class instance (Class/GenericInstance carrying type_args).
                let receiver_is_generic_instance = *is_method && !args.is_empty() && {
                    let rt = self.resolve_through_aliases(args[0].ty);
                    self.type_table
                        .get(rt)
                        .map(|t| match &t.kind {
                            TypeKind::GenericInstance { type_args, .. }
                            | TypeKind::Class { type_args, .. } => !type_args.is_empty(),
                            _ => false,
                        })
                        .unwrap_or(false)
                };
                let route_as_generic_method = *is_method
                    && !callee_is_externish
                    && (callee_has_type_params || receiver_is_generic_instance);

                if (is_user_defined || route_as_generic_method) && !receiver_needs_special_dispatch
                {
                    // Lower args and, for instance method
                    // calls, apply call-boundary materialization
                    // (class→iface fat-ptr wrap, anon coercion).
                    // HIR params don't include `this`, so the
                    // user arg at index `i` corresponds to HIR
                    // param index `i - 1` when `is_method=true`.
                    // Without this wrap, passing a raw class
                    // instance to a method whose param is
                    // interface-typed stores a non-fat-ptr in
                    // any iface field the callee assigns to →
                    // later virtual dispatch on that field
                    // SIGSEGVs (e.g. `reg.register("llama",
                    // new LlamaArch())` in nue.arch).
                    // Static calls are left untouched: they
                    // already go through the static-call path
                    // earlier in this handler when receiver is
                    // missing, and routing them through
                    // `maybe_materialize_for_call` here has
                    // triggered Cranelift symbol-clash on
                    // stdlib MIR wrappers like `array_length`.
                    let arg_regs: Vec<_> = if *is_method {
                        args.iter()
                            .enumerate()
                            .filter_map(|(i, a)| {
                                let reg = self.lower_expression(a)?;
                                if i == 0 {
                                    Some(reg)
                                } else {
                                    Some(self.maybe_materialize_for_call(
                                        a,
                                        reg,
                                        Some(func_id),
                                        i - 1,
                                    ))
                                }
                            })
                            .collect()
                    } else {
                        args.iter()
                            .filter_map(|a| self.lower_expression(a))
                            .collect()
                    };

                    let actual_return_type =
                        if let Some(func) = self.builder.module.functions.get(&func_id) {
                            func.signature.return_type.clone()
                        } else {
                            result_type.clone()
                        };

                    // For generic class method calls, extract concrete type args
                    // from the receiver's type (e.g., Container<String>.get() → type_args=[String]).
                    // This enables the monomorphizer to specialize the function.
                    //
                    // The callee may be a cross-module function not yet merged
                    // into this module, so its type_params are invisible here
                    // (has_type_params is false). Fall back to the receiver
                    // signal: a concrete generic-class instance receiver means
                    // we still want to attach the receiver's type_args so the
                    // monomorphizer can specialize the imported generic method.
                    let has_type_params = self
                        .builder
                        .module
                        .functions
                        .get(&func_id)
                        .map(|f| !f.signature.type_params.is_empty())
                        .unwrap_or(false)
                        || (receiver_is_generic_instance && !callee_is_externish);

                    // Gather type_args: first from HIR call-site type_args, then from receiver's
                    // generic instance type_args, then from the converted HIR type_args computed
                    // earlier in the Call handler.
                    let call_type_args = if has_type_params {
                        if !converted_hir_type_args.is_empty() {
                            converted_hir_type_args.clone()
                        } else if *is_method && !args.is_empty() {
                            // Extract from receiver's GenericInstance / Class type_args
                            let receiver_type = self.resolve_through_aliases(args[0].ty);
                            let type_table = self.type_table;
                            type_table
                                .get(receiver_type)
                                .and_then(|t| match &t.kind {
                                    TypeKind::GenericInstance { type_args, .. }
                                    | TypeKind::Class { type_args, .. } => {
                                        if type_args.is_empty() {
                                            None
                                        } else {
                                            Some(
                                                type_args
                                                    .iter()
                                                    .map(|&ta| self.convert_type(ta))
                                                    .collect::<Vec<_>>(),
                                            )
                                        }
                                    }
                                    _ => None,
                                })
                                .unwrap_or_default()
                        } else {
                            vec![]
                        }
                    } else {
                        vec![]
                    };

                    if !call_type_args.is_empty() {
                        let result = self.builder.build_call_direct_with_type_args(
                            func_id,
                            arg_regs,
                            actual_return_type,
                            call_type_args,
                        );
                        // The function body uses type-erased I64 for all type param values.
                        // When the resolved concrete type differs (F64, I32, String, etc.),
                        // the caller's register has the right TYPE but the value is still
                        // the I64 bit pattern. After inlining + SRA, this becomes visible.
                        // Insert a bitcast for float types where i64→f64 reinterpretation
                        // is needed at the calling convention level.
                        if let Some(reg) = result {
                            let reg_type =
                                self.builder.get_register_type(reg).unwrap_or(IrType::I64);
                            if matches!(reg_type, IrType::F64 | IrType::F32) {
                                // Bitcast from i64 to f64 to ensure calling convention
                                // uses the float register file
                                return self.builder.build_bitcast(reg, reg_type);
                            }
                        }
                        return result;
                    }
                    return self
                        .builder
                        .build_call_direct(func_id, arg_regs, actual_return_type);
                }
            }

            // INTERFACE DISPATCH:
            // When is_method=true, args[0] is the receiver. If the receiver has
            // an interface type, dispatch through the fat pointer vtable.
            if *is_method && !args.is_empty() {
                let receiver = &args[0];
                let receiver_type = receiver.ty;

                if let Some(iface_sym) = self.get_interface_symbol(receiver_type) {
                    let method_name_interned =
                        self.symbol_table.get_symbol(*symbol).map(|s| s.name);

                    if let Some(method_name_i) = method_name_interned {
                        let method_index = self
                            .resolve_interface_method_names(iface_sym)
                            .and_then(|names| names.iter().position(|n| *n == method_name_i));

                        if let Some(idx) = method_index {
                            // Lower the receiver (fat pointer)
                            let fat_ptr_raw = self.lower_expression(receiver)?;

                            // The fat pointer may be stored as I64 - bitcast to Ptr if needed
                            let fat_ptr_ty = self
                                .builder
                                .get_register_type(fat_ptr_raw)
                                .unwrap_or(IrType::I64);
                            let fat_ptr = if !matches!(fat_ptr_ty, IrType::Ptr(_)) {
                                self.builder.build_bitcast(
                                    fat_ptr_raw,
                                    IrType::Ptr(Box::new(IrType::I64)),
                                )?
                            } else {
                                fat_ptr_raw
                            };

                            // Lower the actual arguments (skip args[0] which is receiver)
                            let arg_regs: Vec<_> = args[1..]
                                .iter()
                                .filter_map(|a| self.lower_expression(a))
                                .collect();

                            // Load object pointer from fat_ptr[0]
                            let obj_ptr = self.builder.build_load(fat_ptr, IrType::I64)?;

                            // Load function pointer from fat_ptr[(idx+1)*8]
                            let fn_offset = self
                                .builder
                                .build_const(IrValue::I64(((idx + 1) * 8) as i64))?;
                            let fn_slot = self.builder.build_ptr_add(
                                fat_ptr,
                                fn_offset,
                                IrType::Ptr(Box::new(IrType::U8)),
                            )?;
                            let fn_ptr = self.builder.build_load(fn_slot, IrType::I64)?;

                            // Build call args: self (obj_ptr) + user args
                            let mut call_args = vec![obj_ptr];
                            call_args.extend(arg_regs);

                            // Build signature: (self: Ptr, args...) -> return_type
                            let param_types = {
                                let mut types = vec![IrType::Ptr(Box::new(IrType::Void))]; // self
                                for arg in args[1..].iter() {
                                    types.push(self.convert_type(arg.ty));
                                }
                                types
                            };
                            // Resolve return type from the method's symbol type,
                            // not expr.ty (which may be the interface type instead
                            // of the method's return type in some TAST configurations)
                            let (return_ir_type, resolved_ret_type_id) =
                                self.resolve_interface_method_return_type_full(*symbol, expr.ty);
                            self.emit_iface_return_diagnostic(
                                *symbol,
                                expr.ty,
                                resolved_ret_type_id,
                                expr.source_location,
                            );
                            let return_type = Box::new(return_ir_type);
                            let func_signature = IrType::Function {
                                params: param_types,
                                return_type,
                                varargs: false,
                            };

                            let call_result = self.builder.build_call_indirect(
                                fn_ptr,
                                call_args,
                                func_signature,
                            )?;
                            if let Some(real_ty) = resolved_ret_type_id {
                                self.interface_call_result_types
                                    .insert(call_result, real_ty);
                            }
                            return Some(call_result);
                        }
                    }
                }
            }

            // ENUM INSTANCE METHOD DISPATCH:
            // Delegates to runtime functions registered in runtime_mapping.rs.
            // Injects compile-time constants (type_id, is_boxed) as extra params.
            if *is_method && !args.is_empty() {
                if let Some(Some(result)) = self.try_dispatch_enum_method(*symbol, args) {
                    return Some(result);
                }
            }

            // EARLY RESOLUTION: For typed instance method calls on USER classes,
            // resolve to the import function BEFORE the extern class method dispatch.
            // This prevents user methods like Point2D.add from being incorrectly
            // matched to stdlib methods (sys_deque_add).
            // Skip for classes that have runtime mappings (e.g., EReg) — those
            // must go through get_stdlib_runtime_info for proper dispatch.
            if *is_method && !args.is_empty() {
                let method_name_i = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
                // Check if receiver class has runtime mappings (skip early resolution if so)
                let receiver_has_runtime_mapping = {
                    let receiver_type = self.resolve_through_aliases(args[0].ty);
                    let type_table = self.type_table;
                    type_table
                        .get(receiver_type)
                        .and_then(|ti| {
                            if let crate::tast::core::TypeKind::Class { symbol_id, .. } = &ti.kind {
                                self.symbol_table
                                    .get_symbol(*symbol_id)
                                    .and_then(|sym| self.string_interner.get(sym.name))
                                    .map(|name| self.stdlib_mapping.class_has_any_method(name))
                            } else {
                                None
                            }
                        })
                        .unwrap_or(false)
                };
                if let Some(mn) = method_name_i {
                    let resolved = if receiver_has_runtime_mapping {
                        None // Let runtime mapping handle it
                    } else {
                        self.resolve_method_function_id(args[0].ty, mn)
                    };
                    if let Some(func_id) = resolved {
                        if func_id.0 >= 100_000 {
                            // Resolved to an import function — use it directly
                            let mut arg_regs = Vec::new();
                            for arg in args.iter() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    arg_regs.push(reg);
                                }
                            }
                            self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
                            return self
                                .builder
                                .build_call_direct(func_id, arg_regs, result_type);
                        }
                    }
                }
            }

            // EXTERN CLASS METHOD HANDLING:
            // When MethodCall is desugared to Call with Variable callee,
            // is_method=true and args[0] is the receiver (for instance methods).
            // For static methods, there is no receiver - all args are actual arguments.
            // We need to check if this is an extern class method and redirect to runtime.
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

                // SIMD4f detection: the DECLARED (TAST) receiver type is the SOLE
                // authority for SIMD-vector classification. Two independent hazards,
                // both caused by SSA register-id REUSE corrupting the name/register
                // class hint computed above, are corrected here:
                //
                //  1. A loop-carried phi accumulator (`vacc = vacc.add(..)`) inherits
                //     a stale hint ("rayzor_Usize") from a register a prior Usize
                //     value held — masking the real SIMD4f type and routing `.add()`
                //     to Usize_add (scalar integer add on a vector — garbage).
                //  2. Conversely, a Usize/Bytes receiver (plain address arithmetic)
                //     inherits a stale SIMD hint ("rayzor_SIMD4f") from a register a
                //     prior SIMD value held — mis-routing it into the SIMD4f arith
                //     interception below, which builds a VectorBinOp fed i64 operands
                //     that the LLVM tier rejects (panic / Cranelift-only fallback).
                //
                // Resolution: convert_type(receiver_type) decides. If it IS a vector,
                // that class wins unconditionally (hazard 1). If it is NOT, any SIMD
                // class named by the hint is stale and is discarded (hazard 2); a
                // genuine chained SIMD receiver whose HIR type is opaque (Dynamic) is
                // still recovered from the receiver register's type.
                let ir_ty = self.convert_type(receiver_type);
                let receiver_class_hint_owned = if ir_ty.is_vector() {
                    // Distinguish the integer companion SIMD4i32 (i32x4)
                    // from SIMD4f (f32x4) — both are vectors, but their
                    // instance methods (sum/get/set) map to different
                    // wrappers. Without this, SIMD4i32.sum() dispatched to
                    // SIMD4f_sum (f32 reduce) — masked on native (Cranelift
                    // reduces by SSA value type) but wrong on wasm.
                    Some(simd_vector_class(&ir_ty).to_string())
                } else {
                    // receiver_type is NOT a vector: a SIMD class named by the
                    // name/register hint is stale (hazard 2) and MUST be rejected
                    // before it reaches the arith interception. A non-SIMD hint
                    // (e.g. a genuine Bytes/Usize) is left intact.
                    let non_simd_hint = receiver_class_hint_owned
                        .filter(|h| h != "rayzor_SIMD4f" && h != "rayzor_SIMD4i32");
                    if non_simd_hint.is_some() {
                        non_simd_hint
                    } else if self.type_is_native_named(receiver_type, "rayzor::Atomic") {
                        // Atomic's type-map returns Ptr<I32> (not a vector), so is_vector()
                        // never fires; resolve it by the abstract's @:native name instead.
                        Some("rayzor_Atomic".to_string())
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

                // SIMD4f arithmetic METHODS (`a.add(b)` etc.) must compile
                // to the same single vector instruction as the OPERATORS
                // (`a + b`, lowered to VectorBinOp at ~19541). The default
                // method-call path routes them to a MIR wrapper that
                // mishandles the vector ABI and returns garbage (a SIMD4f
                // value carried as I64/Ptr(Void)). Emit VectorBinOp
                // directly. Restricted to rayzor_SIMD4f (f32x4); the i32x4
                // companion is excluded because integer VectorBinOp
                // miscompiles on the wasm backend.
                if receiver_class_hint == Some("rayzor_SIMD4f") && args.len() == 2 {
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
                    // Defense-in-depth: the receiver operand's own DECLARED type
                    // must itself classify as f32x4. The hint is no longer trusted
                    // in isolation — this refuses to build a VectorBinOp over a
                    // non-vector operand (the failure mode a stale SIMD hint on a
                    // reused register would otherwise cause: VectorBinOp fed i64).
                    let operands_are_simd4f = {
                        let t = self.convert_type(args[0].ty);
                        t.is_vector() && simd_vector_class(&t) == "rayzor_SIMD4f"
                    };
                    if let Some(bin_op) = vbop.filter(|_| operands_are_simd4f) {
                        let lhs_reg = self.lower_expression(&args[0])?;
                        let rhs_reg = self.lower_expression(&args[1])?;
                        // vec_ty must ALWAYS be a vector: fall through both operand
                        // register types (a scalar-typed operand register is a bug)
                        // to the f32x4 default rather than emitting VectorBinOp{I64}.
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
                    Some("rayzor_SIMD4f") | Some("rayzor_SIMD4i32")
                ) {
                    let simd_cls = receiver_class_hint.unwrap();
                    let method_name_str = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|s| self.string_interner.get(s.name));
                    if let Some(mn) = method_name_str {
                        self.stdlib_mapping
                            .find_by_name_and_params(simd_cls, mn, param_count)
                            .or_else(|| self.stdlib_mapping.find_by_name(simd_cls, mn))
                            .map(|(sig, mapping)| (sig.class, sig.method, mapping))
                    } else {
                        None
                    }
                } else if receiver_class_hint == Some("rayzor_Atomic") {
                    // Atomic direct lookup: bypass FALLBACK2 (mirror of SIMD4f).
                    let method_name_str = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|s| self.string_interner.get(s.name));
                    method_name_str.and_then(|mn| {
                        self.stdlib_mapping
                            .find_by_name_and_params("rayzor_Atomic", mn, param_count)
                            .or_else(|| self.stdlib_mapping.find_by_name("rayzor_Atomic", mn))
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
                            class_name,
                            method_name,
                            runtime_func,
                            is_instance_method,
                            is_mir_wrapper
                        );

                        // MIR wrapper path: use register_stdlib_mir_forward_ref
                        // MIR wrappers (SIMD4f, Thread, Channel, etc.) are compiled by
                        // Cranelift alongside user code. They must NOT be registered as
                        // extern C functions.
                        if is_mir_wrapper {
                            self.builder.call_label = Some(format!("MIR_WRAPPER:{}", runtime_func));

                            // Get the MIR wrapper's expected signature for auto-boxing/unboxing
                            let mir_wrapper_sig =
                                self.get_stdlib_mir_wrapper_signature(runtime_func);

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
                                        self.box_channel_payload(
                                            reg,
                                            arg.ty,
                                            &actual_ty,
                                            &expected_ty,
                                        )?
                                    } else if is_type_erased_ptr
                                        && matches!(&expected_ty, IrType::Ptr(_))
                                    {
                                        // Cast I64 → Ptr(U8) for type-erased pointers
                                        self.builder
                                            .build_cast(reg, IrType::I64, expected_ty.clone())
                                            .unwrap_or(reg)
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
                                let return_class =
                                    Self::get_return_class_hint(class_name, method_name);
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

                            // Thread<T>.join() boxes its result via haxe_box_int_ptr
                            // (rayzor_thread_join), so for a concrete HEAP T the boxed
                            // i64 payload IS the object pointer. maybe_unbox's raw
                            // passthrough arm — shared with methods that return an
                            // UN-boxed handle (e.g. Arc.get) — would skip the unbox and
                            // hand back the box address (garbage). Unbox inline, keyed on
                            // the wrapper so only the boxing method changes behavior.
                            // Excludes Ptr(primitive) (Null<Int>), handled above.
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

                            // Channel payloads are uniformly boxed DynamicValues; route the
                            // return through the tag-aware unbox (fixes erased prim receive,
                            // recovers boxed refs) instead of the raw shared path.
                            if runtime_func == "Channel_receive"
                                || runtime_func == "Channel_tryReceive"
                            {
                                let is_try = runtime_func == "Channel_tryReceive";
                                // Inferred channels erase T to I64, whose unbox
                                // int-truncates a Float payload (4.75 -> 4). The
                                // enclosing `var x:Float = ...` declared type is the
                                // ground truth — refine to the float arm. Scoped to
                                // floats + non-try (tryReceive keeps the tag-driven
                                // unbox so nullables stay correct).
                                let refined = if !is_try && matches!(resolved_expected, IrType::I64)
                                {
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
                                            type_args,
                                            ..
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
                                IrType::Ptr(ref inner)
                                    if matches!(inner.as_ref(), IrType::String) =>
                                {
                                    // Concrete pointer type (e.g., Ptr(String))
                                    self.builder.build_cast(
                                        call_result,
                                        IrType::U64,
                                        result_type.clone(),
                                    )
                                }
                                IrType::Ptr(_)
                                    if matches!(
                                        resolved_value_ty.as_ref(),
                                        Some(IrType::Ptr(_))
                                    ) =>
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
            // SPECIAL CASE: Handle global trace() function
            // Route to type-specific trace functions based on argument type
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

                let class_info = if let Some(crate::tast::core::TypeKind::Class {
                    symbol_id, ..
                }) = &type_kind
                {
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
                            let var_type_kind =
                                type_table.get(sym.type_id).map(|ti| ti.kind.clone());
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
                                        Some(crate::tast::core::TypeKind::TypeParameter {
                                            ..
                                        }) => 5u8,
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
                                    let offset =
                                        self.builder.build_const(IrValue::I64(i as i64))?;
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
                    return self.builder.build_call_direct(
                        trace_array_id,
                        vec![arg_reg],
                        IrType::Void,
                    );
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
                                .map(|f| {
                                    f.signature.type_params.iter().any(|tp| tp.name == *tp_name)
                                })
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
                        IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::String) => {
                            "traceString"
                        }
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

                let runtime_func_id = self.get_or_register_extern_function(
                    runtime_func,
                    final_param_types,
                    IrType::Void,
                );

                // Generate the call
                return self.builder.build_call_direct(
                    runtime_func_id,
                    vec![final_arg_reg],
                    IrType::Void,
                );
            }

            // SPECIAL CASE: Handle Std.string() function
            // Route to type-specific string conversion functions based on argument type
            // Note: Std.string() comes as a static method call with 2 args (Std class + actual arg)
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
                let mir_wrapper_id = self.get_or_register_extern_function(
                    mir_wrapper,
                    param_types,
                    return_type.clone(),
                );

                // Generate the call to MIR wrapper
                return self
                    .builder
                    .build_call_direct(mir_wrapper_id, vec![arg_reg], return_type);
            }

            // For instance method calls, check if this is a stdlib method or Dynamic method
            // Note: Static methods like Thread.spawn() can also come through here with is_method=true
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
                                    .chain(
                                        args[1..].iter().filter_map(|a| self.lower_expression(a)),
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
                                                let reg_ir_type =
                                                    self.builder.get_register_type(reg);
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
                                                    type_table.get(args[0].ty).and_then(|ti| {
                                                    match &ti.kind {
                                                        crate::tast::TypeKind::Class { type_args, .. }
                                                        | crate::tast::TypeKind::GenericInstance { type_args, .. } => {
                                                            if !type_args.is_empty() {
                                                                let t = self.convert_type(type_args[0]);
                                                                if matches!(t, IrType::I32 | IrType::I64 | IrType::F32 | IrType::F64 | IrType::Bool) {
                                                                    Some(t)
                                                                } else {
                                                                    None
                                                                }
                                                            } else { None }
                                                        }
                                                        _ => None,
                                                    }
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
                                let method_name =
                                    self.symbol_table.get_symbol(*symbol).map(|s| s.name);
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
                                            let unbox_func_id = self
                                                .get_or_register_extern_function(
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
                        if let Some(mono_class) =
                            self.monomorphized_var_types.get(&var_symbol).cloned()
                        {
                            // Get the method name
                            if let Some(method_sym) = self.symbol_table.get_symbol(*symbol) {
                                if let Some(method_name) = self.string_interner.get(method_sym.name)
                                {
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
                        "balance"
                            | "setLoop"
                            | "compare"
                            | "merge"
                            | "minBinding"
                            | "removeMinBinding"
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
                                                let box_func = self
                                                    .get_or_register_extern_function(
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
                                                let box_func = self
                                                    .get_or_register_extern_function(
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
                                                        .build_cast(
                                                            arg_reg,
                                                            IrType::F32,
                                                            IrType::F64,
                                                        )
                                                        .unwrap_or(arg_reg)
                                                } else {
                                                    arg_reg
                                                };
                                                let box_func = self
                                                    .get_or_register_extern_function(
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
                                                let box_func = self
                                                    .get_or_register_extern_function(
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
                                    else if extend_i64_mask != 0
                                        && (extend_i64_mask & (1 << i)) != 0
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
                            let (resolved_return_type, resolved_from_type_args) =
                                if returns_raw_value {
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
                                                type_args,
                                                ..
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
                                        IrType::I32 => self.builder.build_cast(
                                            raw_reg,
                                            IrType::U64,
                                            IrType::I32,
                                        ),
                                        IrType::I64 => self.builder.build_cast(
                                            raw_reg,
                                            IrType::U64,
                                            IrType::I64,
                                        ),
                                        IrType::F64 => {
                                            self.builder.build_bitcast(raw_reg, IrType::F64)
                                        }
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
                                        IrType::Bool => self.builder.build_cast(
                                            raw_reg,
                                            IrType::U64,
                                            IrType::Bool,
                                        ),
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
                                            self.builder.build_cast(
                                                raw_reg,
                                                IrType::U64,
                                                IrType::I64,
                                            )
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
                                        crate::tast::core::TypeKind::Class {
                                            symbol_id, ..
                                        } => {
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
                                        crate::tast::core::TypeKind::Abstract {
                                            symbol_id, ..
                                        } => self
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
                                if let Some(method_name) = self.string_interner.get(method_sym.name)
                                {
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
                                                let mir_func_id = self
                                                    .register_stdlib_mir_forward_ref(
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
                                                                && (ptr_conversion_mask & (1 << i))
                                                                    != 0
                                                            {
                                                                IrType::Ptr(Box::new(IrType::U8))
                                                            } else {
                                                                self.convert_type(arg.ty)
                                                            }
                                                        })
                                                        .collect();
                                                    (param_types, result_type.clone())
                                                });
                                            let runtime_func_id = self
                                                .get_or_register_extern_function(
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
                                            if let TypeKind::Class { symbol_id, .. } =
                                                &type_info.kind
                                            {
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
                                            debug!(
                                                "[STDLIB MIR] Generated call, result: {:?}",
                                                result
                                            );
                                            return result;
                                        }

                                        // Try the inferred class first
                                        let fake_qual_name = format!(
                                            "rayzor.concurrent.{}.{}",
                                            class_name, method_name
                                        );
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
                                                                && (ptr_conversion_mask & (1 << i))
                                                                    != 0
                                                            {
                                                                IrType::Ptr(Box::new(IrType::U8))
                                                            } else {
                                                                self.convert_type(arg.ty)
                                                            }
                                                        })
                                                        .collect();
                                                    (param_types, result_type.clone())
                                                });
                                            let runtime_func_id = self
                                                .get_or_register_extern_function(
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
                                                let mir_func_id = self
                                                    .register_stdlib_mir_forward_ref(
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
                                            let runtime_func_id = self
                                                .get_or_register_extern_function(
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
                                            symbol_id,
                                            ..
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
            // For static methods, check if it's a stdlib static method
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

                        // Try to get the qualified name to determine the class
                        if let Some(qual_name) = sym_info.qualified_name {
                            if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                                debug!("[PRE-CHECK] Qualified name: '{}'", qual_name_str);

                                // SPECIAL CASE: Thread/Channel/Mutex/Arc methods are MIR wrappers, not runtime_mapping
                                // These are implemented in stdlib MIR (thread.rs, channel.rs, etc.)
                                // Pattern: "rayzor.concurrent.Thread.spawn" -> "Thread_spawn"
                                // NOTE: This only applies to rayzor.concurrent.*, NOT sys.thread.*
                                let parts: Vec<&str> = qual_name_str.split('.').collect();
                                if parts.len() >= 2 {
                                    let class_name = parts[parts.len() - 2];

                                    // Check if this is a rayzor.concurrent.* class (NOT sys.thread.*)
                                    // sys.thread.Thread uses runtime mapping directly, not MIR wrappers
                                    // Use dynamic check via stdlib_mapping instead of hardcoded list
                                    let is_rayzor_concurrent =
                                        qual_name_str.starts_with("rayzor.concurrent.");
                                    if is_rayzor_concurrent
                                        && self.stdlib_mapping.is_mir_wrapper_class(class_name)
                                    {
                                        // Use capitalized class names for rayzor.concurrent (Thread, Channel, etc.)
                                        let mir_func_name =
                                            format!("{}_{}", class_name, method_name);
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

                                        // WORKAROUND: static calls may carry a synthetic
                                        // class receiver argument. Prefer the mapping
                                        // signature arity to trim that argument.
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

                                        // Lower all arguments and collect their types
                                        let mut arg_regs = Vec::new();
                                        let mut param_types = Vec::new();
                                        for (idx, arg) in actual_args.iter().enumerate() {
                                            debug!("[STDLIB MIR] arg[{}] ty={:?}", idx, arg.ty);
                                            if let Some(reg) = self.lower_expression(arg) {
                                                arg_regs.push(reg);
                                                param_types.push(self.convert_type(arg.ty));
                                            }
                                        }
                                        // Register forward reference - will be provided by merged stdlib module
                                        // We infer the signature from the call site arguments
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
                                }

                                // Check if this is a stdlib class method by looking at qualified name
                                // e.g., "rayzor.concurrent.Thread.spawn" or "test.Thread.spawn"
                                let lookup_result = self
                                    .get_static_stdlib_runtime_func_with_params(
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

                                    // Get the expected signature from our registered extern functions
                                    // This ensures we use the correct types (e.g., I64 for Std.random)
                                    let (expected_param_types, expected_return_type) = self
                                        .get_extern_function_signature(&runtime_func)
                                        .unwrap_or_else(|| {
                                            // Fall back to inferred types from TAST
                                            let string_ptr_ty =
                                                IrType::Ptr(Box::new(IrType::String));
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

                                    // Lower all arguments
                                    let arg_regs: Vec<_> = runtime_call_args
                                        .iter()
                                        .filter_map(|a| self.lower_expression(a))
                                        .collect();

                                    debug!(
                                        "[STATIC METHOD] Lowered {} args: {:?}",
                                        arg_regs.len(),
                                        arg_regs
                                    );

                                    // Cast/box arguments to expected types if needed
                                    let final_arg_regs: Vec<_> = arg_regs.iter().enumerate()
                                        .map(|(i, &reg)| {
                                            if let (Some(expected_ty), Some(actual_ty)) = (
                                                expected_param_types.get(i),
                                                self.builder.get_register_type(reg)
                                            ) {
                                                // If types differ, insert a cast or box
                                                if *expected_ty != actual_ty {
                                                    // When expected is Ptr(U8) (Dynamic/boxed), auto-box the value
                                                    let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                                    if is_ptr_u8 && i < runtime_call_args.len() {
                                                        debug!("[STATIC METHOD BOX] Attempting auto-box for arg {} with type {:?}", i, runtime_call_args[i].ty);
                                                        // Use box_value_for_dynamic to properly box based on HIR type
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

                                    // Generate the call to the runtime function
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

                        // Fallback: still inside method_name scope.
                        // If qualified_name is not set (e.g., Reflect.compare from import files),
                        // try to find a matching static stdlib method by scanning all known classes.
                        // Only match static methods to avoid false positives.
                        // If qualified_name is available, prefer class-qualified lookup
                        // before doing a global static-name fallback.
                        let mut static_fallback = None;
                        if let Some(qual_name_str) = sym_info
                            .qualified_name
                            .and_then(|q| self.string_interner.get(q))
                        {
                            let parts: Vec<&str> = qual_name_str.split('.').collect();
                            if parts.len() >= 2 {
                                let mut class_candidates: Vec<String> = Vec::new();
                                // Fully-qualified class form used in runtime mapping
                                class_candidates.push(parts[..parts.len() - 1].join("_"));
                                // Simple class name fallback
                                class_candidates.push(parts[parts.len() - 2].to_string());

                                for class_name in class_candidates {
                                    if let Some(found) =
                                        self.stdlib_mapping.find_by_name_and_params(
                                            &class_name,
                                            method_name,
                                            static_args.len(),
                                        )
                                    {
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
                            static_fallback =
                                self.stdlib_mapping.find_static_method_by_name_and_params(
                                    method_name,
                                    static_args.len(),
                                );
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

                            // Lower all arguments first
                            let mut arg_regs: Vec<IrId> = Vec::new();
                            let mut arg_types: Vec<IrType> = Vec::new();
                            for arg in static_args.iter() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    arg_regs.push(reg);
                                    arg_types.push(self.convert_type(arg.ty));
                                }
                            }

                            // Special case: Reflect.compare → haxe_reflect_compare_typed
                            // Same logic as the qualified-name path: detect argument type and
                            // append a type_tag parameter to avoid boxing.
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
                                                        type_param_name =
                                                            Some(name_str.to_string());
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
                                        self.builder.build_const(IrValue::I32(
                                            known_type_tag.unwrap_or(1),
                                        ))?
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
                    } // end of if let Some(method_name)
                }
            }

            // Check if this symbol is a function (local or external)
            // First try direct symbol ID lookup
            let method_name_interned = self.symbol_table.get_symbol(*symbol).map(|s| s.name);
            let mut func_id_opt = self.get_function_id(symbol);

            // Intercept @:shader wgsl() calls — the stub function has
            // an empty body; redirect to the transpiler output.
            if func_id_opt.is_some() {
                let callee_is_wgsl = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name))
                    .map(|n| n == "wgsl")
                    .unwrap_or(false);
                if callee_is_wgsl {
                    // Find the @:shader class in current_hir_types
                    for (_tid, decl) in self.current_hir_types.iter() {
                        if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                            let is_shader = self
                                .symbol_table
                                .get_symbol(c.symbol_id)
                                .map(|s| s.flags.is_shader())
                                .unwrap_or(false);
                            if is_shader {
                                let type_table = self.type_table;
                                match crate::codegen::wgsl_transpiler::transpile_shader_from_hir(
                                    c,
                                    self.symbol_table,
                                    type_table,
                                    self.string_interner,
                                    self.current_hir_types,
                                ) {
                                    Ok(wgsl) => {
                                        return self.builder.build_const(IrValue::String(wgsl))
                                    }
                                    Err(e) => {
                                        return self.builder.build_const(IrValue::String(format!(
                                            "/* WGSL error: {} */",
                                            e
                                        )))
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let has_synthetic_static_receiver =
                *is_method && self.effective_static_call_args(args).len() != args.len();

            if func_id_opt.is_none()
                && *is_method
                && !has_synthetic_static_receiver
                && !args.is_empty()
            {
                if let Some(method_name) = method_name_interned {
                    func_id_opt = self.resolve_method_function_id(args[0].ty, method_name);
                }
            }

            // If not found by symbol ID, try lookup by qualified name
            // This handles cross-module calls where symbol IDs differ between modules,
            // and also intra-module static method calls where the call site symbol
            // differs from the method definition symbol (e.g., Body.Sun() in nbody)
            if func_id_opt.is_none() {
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(qual_name) = sym_info.qualified_name {
                        if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                            // Search local function_map by qualified name first
                            for (local_sym, &local_func_id) in &self.function_map {
                                if let Some(local_sym_info) =
                                    self.symbol_table.get_symbol(*local_sym)
                                {
                                    if let Some(local_qual) = local_sym_info.qualified_name {
                                        if let Some(local_qual_str) =
                                            self.string_interner.get(local_qual)
                                        {
                                            if local_qual_str == qual_name_str {
                                                debug!(
                                                    "[QUAL-NAME LOCAL] Found function by qualified name '{}': symbol {:?} -> func_id={:?}",
                                                    qual_name_str, local_sym, local_func_id
                                                );
                                                func_id_opt = Some(local_func_id);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            // If not found locally, search external_function_map
                            if func_id_opt.is_none() {
                                for (ext_sym, &ext_func_id) in &self.external_function_map {
                                    if let Some(ext_sym_info) =
                                        self.symbol_table.get_symbol(*ext_sym)
                                    {
                                        if let Some(ext_qual) = ext_sym_info.qualified_name {
                                            if let Some(ext_qual_str) =
                                                self.string_interner.get(ext_qual)
                                            {
                                                if ext_qual_str == qual_name_str {
                                                    debug!(
                                                        "[CROSS-MODULE] Found function by qualified name '{}': symbol {:?} -> func_id={:?}",
                                                        qual_name_str, ext_sym, ext_func_id
                                                    );
                                                    func_id_opt = Some(ext_func_id);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Call-site symbol has no qualified name (common for cross-class
                        // static method calls in multi-class files, e.g., Body.Jupiter()).
                        // Fall back to searching function_map by bare name.
                        // Never the currently-compiling function (a lone
                        // same-named local — e.g. `.forward` inside a
                        // forward — would self-bind into infinite
                        // recursion), and for method calls never a
                        // candidate whose class positively differs from
                        // the receiver's.
                        let recv_class_bare: Option<String> = if *is_method && !args.is_empty() {
                            let type_table = self.type_table;
                            type_table
                                .get(args[0].ty)
                                .and_then(|ti| match &ti.kind {
                                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                    _ => None,
                                })
                                .and_then(|sid| self.symbol_table.get_symbol(sid))
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|s| s.to_string())
                        } else {
                            None
                        };
                        if let Some(func_name) = self.string_interner.get(sym_info.name) {
                            for (func_sym, &func_id) in &self.function_map {
                                if *func_sym == *symbol {
                                    continue;
                                }
                                if Some(func_id) == self.builder.current_function {
                                    continue;
                                }
                                if let Some(func_sym_info) = self.symbol_table.get_symbol(*func_sym)
                                {
                                    if let (Some(rb), Some(qn)) = (
                                        recv_class_bare.as_deref(),
                                        func_sym_info
                                            .qualified_name
                                            .and_then(|q| self.string_interner.get(q)),
                                    ) {
                                        let parts: Vec<&str> = qn.split('.').collect();
                                        if parts.len() >= 2 && parts[parts.len() - 2] != rb {
                                            continue;
                                        }
                                    }
                                    if let Some(fm_name) =
                                        self.string_interner.get(func_sym_info.name)
                                    {
                                        if fm_name == func_name {
                                            debug!(
                                                "[BARE-NAME LOCAL] Found function by bare name '{}': sym {:?} -> func_id={:?}",
                                                func_name, func_sym, func_id
                                            );
                                            func_id_opt = Some(func_id);
                                            break;
                                        }
                                    }
                                }
                            }
                            // NOTE: Removed bare-name search in external_function_map.
                            // Bare-name matching across modules causes false positives
                            // (e.g., ListNode.create() -> rayzor_tcc_create).
                            // Cross-module calls must use qualified name matching.
                        }
                    }
                }
            }

            // If still not found, try lookup by function name in function_map
            // This handles cases where method calls use different symbol IDs than the definition
            // (e.g., chained method calls like z.mul(z).add(c) where add has a different symbol)
            //
            // IMPORTANT: When matching by bare name, also verify the function belongs to
            // the receiver's class (via qualified name). Without this, common names like
            // "get", "set", "toString" could match wrong stdlib functions.
            if func_id_opt.is_none() && *is_method && !has_synthetic_static_receiver {
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(method_name) = self.string_interner.get(sym_info.name) {
                        debug!(
                            "[NAME-FALLBACK] Searching for method '{}' sym={:?}",
                            method_name, symbol
                        );
                        // Get receiver's class name for disambiguation
                        let receiver_class_name = if !args.is_empty() {
                            let type_table = self.type_table;
                            let class_sym =
                                type_table.get(args[0].ty).and_then(|ti| match &ti.kind {
                                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                    TypeKind::GenericInstance { base_type, .. } => {
                                        type_table.get(*base_type).and_then(|bt| match &bt.kind {
                                            TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                                            _ => None,
                                        })
                                    }
                                    _ => None,
                                });
                            let name = class_sym.and_then(|sid| {
                                self.symbol_table
                                    .get_symbol(sid)
                                    .and_then(|s| self.string_interner.get(s.name))
                                    .map(|s| s.to_string())
                            });
                            name
                        } else {
                            None
                        };

                        debug!(
                            "[NAME-FALLBACK] receiver_class_name={:?}",
                            receiver_class_name
                        );
                        // Search function_map by name, preferring qualified name match
                        // Pass 1: strict class name matching
                        for (func_sym, &func_id) in &self.function_map {
                            if let Some(func_sym_info) = self.symbol_table.get_symbol(*func_sym) {
                                if let Some(func_name) =
                                    self.string_interner.get(func_sym_info.name)
                                {
                                    if func_name == method_name {
                                        // If we know the receiver class, verify via qualified name
                                        if let Some(ref class_name) = receiver_class_name {
                                            let qual_match = func_sym_info
                                                .qualified_name
                                                .and_then(|qn| self.string_interner.get(qn))
                                                .map(|qn| qn.contains(class_name.as_str()))
                                                .unwrap_or(false);
                                            if !qual_match {
                                                continue; // Skip — wrong class
                                            }
                                        }
                                        debug!(
                                            "[NAME FALLBACK] Found method '{}' by name: {:?} -> {:?}",
                                            method_name, func_sym, func_id
                                        );
                                        func_id_opt = Some(func_id);
                                        break;
                                    }
                                }
                            }
                        }

                        // Pass 2: Search external_function_name_map by qualified name
                        // Try both "ClassName.method" and "pkg.ClassName.method" patterns
                        if func_id_opt.is_none() {
                            if let Some(ref class_name) = receiver_class_name {
                                let suffix = format!("{}.{}", class_name, method_name);
                                // Direct match first
                                if let Some(&fid) = self.external_function_name_map.get(&suffix) {
                                    func_id_opt = Some(fid);
                                } else {
                                    // Suffix match: find "pkg.ClassName.method"
                                    for (name, &fid) in &self.external_function_name_map {
                                        if name.ends_with(&suffix) {
                                            func_id_opt = Some(fid);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Guard against short-name symbol collisions for static calls:
            // if the resolved function signature arity does not match this call site,
            // drop it so static runtime fallback can re-resolve by (name, arity).
            if func_id_opt.is_some() && (!*is_method || has_synthetic_static_receiver) {
                let static_arg_count = self.effective_static_call_args(args).len();
                if let Some(func_id) = func_id_opt {
                    let expected_params_opt = self
                        .builder
                        .module
                        .functions
                        .get(&func_id)
                        .map(|func| func.signature.parameters.len())
                        .or_else(|| {
                            self.symbol_table.get_symbol(*symbol).and_then(|sym| {
                                let type_table = self.type_table;
                                type_table.get(sym.type_id).and_then(|ti| {
                                    if let TypeKind::Function { params, .. } = &ti.kind {
                                        Some(params.len())
                                    } else {
                                        None
                                    }
                                })
                            })
                        });

                    if let Some(expected_params) = expected_params_opt {
                        if expected_params != static_arg_count {
                            debug!(
                                "[STATIC ARITY MISMATCH] symbol={:?} resolved func_id={:?} expected_params={} call_args={} -> fallback",
                                symbol, func_id, expected_params, static_arg_count
                            );
                            func_id_opt = None;
                        }
                    }
                }
            }

            // Fallback for extern class static methods (e.g. NativeStackTrace.exceptionStack()).
            // StaticFieldAccess becomes Variable in HIR, bypassing the Field-callee stdlib
            // dispatch. Prefer symbol qualified_name, then class-less name fallback.
            if func_id_opt.is_none() {
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(method_name) = self.string_interner.get(sym_info.name) {
                        let static_args = self.effective_static_call_args(args);
                        let mut runtime_func_name: Option<String> = None;

                        // Prefer class-qualified dispatch if present on the symbol.
                        if let Some(qual_name_str) = sym_info
                            .qualified_name
                            .and_then(|qn| self.string_interner.get(qn))
                        {
                            if let Some(found) = self.get_static_stdlib_runtime_func_with_params(
                                qual_name_str,
                                method_name,
                                static_args.len(),
                            ) {
                                runtime_func_name = Some(found.to_string());
                            }
                        }

                        // True last resort: class-less static method lookup.
                        if runtime_func_name.is_none() {
                            if let Some((_sig, mapping)) =
                                self.stdlib_mapping.find_static_method_by_name_and_params(
                                    method_name,
                                    static_args.len(),
                                )
                            {
                                runtime_func_name = Some(mapping.runtime_name.to_string());
                            }
                        }

                        if let Some(runtime_func_name) = runtime_func_name {
                            let mut arg_regs: Vec<IrId> = Vec::new();
                            let mut arg_types: Vec<IrType> = Vec::new();
                            for arg in static_args.iter() {
                                if let Some(reg) = self.lower_expression(arg) {
                                    arg_regs.push(reg);
                                    arg_types.push(self.convert_type(arg.ty));
                                }
                            }
                            let (expected_param_types, expected_return_type) = self
                                .get_extern_function_signature(&runtime_func_name)
                                .unwrap_or_else(|| (arg_types, result_type.clone()));
                            let runtime_func_id = self.get_or_register_extern_function(
                                &runtime_func_name,
                                expected_param_types,
                                expected_return_type.clone(),
                            );
                            return self.builder.build_call_direct(
                                runtime_func_id,
                                arg_regs,
                                expected_return_type,
                            );
                        }
                    }
                }
            }

            if let Some(func_id) = func_id_opt {
                let sym_name = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| self.string_interner.get(s.name))
                    .unwrap_or("<unknown>");
                self.builder.call_label = Some(format!("FUNC_MAP:{}", sym_name));
                let qual_name = self
                    .symbol_table
                    .get_symbol(*symbol)
                    .and_then(|s| s.qualified_name)
                    .and_then(|qn| self.string_interner.get(qn))
                    .unwrap_or("<none>");
                let is_external = self.external_function_map.contains_key(symbol);

                debug!(
                    "[FUNCTION_MAP LOOKUP] Found symbol {:?} '{}' (qual: '{}') -> func_id={:?}, is_method={}, external={}",
                    symbol, sym_name, qual_name, func_id, is_method, is_external
                );

                // IMPORTANT: Use the function's actual return type, not expr.ty
                // Check both functions (local) and extern_functions (forward refs to stdlib)
                let actual_return_type = if let Some(func) =
                    self.builder.module.functions.get(&func_id)
                {
                    debug!(
                        "[FUNCTION_MAP] Using actual return type {:?} for function {:?}",
                        func.signature.return_type, func.name
                    );
                    func.signature.return_type.clone()
                } else if let Some(func) = self.builder.module.extern_functions.get(&func_id) {
                    debug!(
                        "[FUNCTION_MAP] Using extern_functions return type {:?} for {:?}",
                        func.signature.return_type, func_id
                    );
                    func.signature.return_type.clone()
                } else {
                    // Function not in module yet (probably forward ref to stdlib MIR wrapper)
                    // Try to look up the correct signature by function name
                    debug!(
                        "[FUNCTION_MAP] Function {:?} not found in module, checking stdlib signatures",
                        func_id
                    );
                    if let Some((_params, ret_ty)) =
                        self.get_stdlib_mir_wrapper_signature(&sym_name)
                    {
                        debug!(
                            "[FUNCTION_MAP] Found stdlib signature for '{}': returns {:?}",
                            sym_name, ret_ty
                        );
                        ret_ty
                    } else {
                        debug!(
                            "[FUNCTION_MAP] No stdlib signature found, using expr return type {:?}",
                            result_type
                        );
                        result_type.clone()
                    }
                };
                let function_param_count = self
                    .builder
                    .module
                    .functions
                    .get(&func_id)
                    .map(|f| f.signature.parameters.len())
                    .or_else(|| {
                        self.builder
                            .module
                            .extern_functions
                            .get(&func_id)
                            .map(|f| f.signature.parameters.len())
                    });
                let has_arity_static_receiver = *is_method
                    && function_param_count
                        .map(|param_count| args.len() == param_count + 1)
                        .unwrap_or(false);
                let treat_as_static_call =
                    has_synthetic_static_receiver || has_arity_static_receiver;
                if has_arity_static_receiver {
                    debug!(
                        "[STATIC-RECEIVER ARITY] Treating method call as static: symbol={:?}, func_id={:?}, args={}, params={:?}",
                        symbol,
                        func_id,
                        args.len(),
                        function_param_count
                    );
                }

                // Handle method calls where the object is passed as first argument
                if *is_method && !treat_as_static_call {
                    // For method calls, args already includes the object as first arg.
                    // Track non-receiver args as temps ONLY if the callee is user-defined.
                    // Stdlib/runtime methods (e.g., Array.push) may store arguments.
                    let callee_is_user_defined = self
                        .builder
                        .module
                        .functions
                        .get(&func_id)
                        .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                        .unwrap_or(false);

                    let mut arg_regs = Vec::new();

                    // Check if receiver (args[0]) is Dynamic-typed — needs unboxing
                    let receiver_is_dynamic = if !args.is_empty() {
                        let type_table = self.type_table;
                        type_table
                            .get(args[0].ty)
                            .map(|t| matches!(t.kind, TypeKind::Dynamic))
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    for (i, arg) in args.iter().enumerate() {
                        if let Some(reg) = self.lower_expression(arg) {
                            // Materialize anon-backed variables at call boundary (skip receiver)
                            // For method calls, args[0] is receiver, args[1..] are params
                            // HIR param_types don't include `this`, so param_index = i - 1
                            let reg = if i > 0 {
                                self.maybe_materialize_for_call(arg, reg, Some(func_id), i - 1)
                            } else {
                                reg
                            };
                            if i == 0 && receiver_is_dynamic && callee_is_user_defined {
                                // Dynamic receiver: unbox DynamicValue* to get raw object pointer
                                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                let unbox_func_id = self.get_or_register_extern_function(
                                    "haxe_unbox_reference_ptr",
                                    vec![ptr_u8.clone()],
                                    ptr_u8.clone(),
                                );
                                if let Some(unboxed) =
                                    self.builder
                                        .build_call_direct(unbox_func_id, vec![reg], ptr_u8)
                                {
                                    arg_regs.push(unboxed);
                                } else {
                                    arg_regs.push(reg);
                                }
                            } else {
                                // @:derive(Copy): copy variable args at call boundary
                                let reg = if i > 0 {
                                    if let HirExprKind::Variable { .. } = &arg.kind {
                                        if let Some(class_sym) = self.get_copy_class_symbol(arg.ty)
                                        {
                                            self.emit_shallow_copy(reg, class_sym).unwrap_or(reg)
                                        } else {
                                            reg
                                        }
                                    } else {
                                        reg
                                    }
                                } else {
                                    reg
                                };
                                if i > 0 && callee_is_user_defined {
                                    let is_heap_intermediate = matches!(
                                        &arg.kind,
                                        HirExprKind::New { .. } | HirExprKind::Call { .. }
                                    ) && self.get_drop_behavior(arg.ty)
                                        == DropBehavior::AutoDrop;
                                    if is_heap_intermediate {
                                        self.temp_heap_values.push(reg);
                                    }
                                }
                                arg_regs.push(reg);
                            }
                        }
                    }

                    // Coerce Int→Float at cross-module call boundaries
                    self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, true);
                    // Fill in default values for any missing optional parameters
                    self.fill_default_args(func_id, &mut arg_regs, true);

                    // Extract type_args for generic method calls.
                    // Priority: 1) HIR type_args, 2) class type_args, 3) infer from args
                    let ir_type_args = if !converted_hir_type_args.is_empty() {
                        // Method-level type args from HIR (e.g., explicitly specified)
                        converted_hir_type_args.clone()
                    } else if !args.is_empty() {
                        let receiver_type = args[0].ty;
                        let class_type_args = {
                            let type_table = self.type_table;
                            if let Some(receiver_info) = type_table.get(receiver_type) {
                                if let crate::tast::TypeKind::Class { type_args, .. } =
                                    &receiver_info.kind
                                {
                                    type_args.clone()
                                } else {
                                    Vec::new()
                                }
                            } else {
                                Vec::new()
                            }
                        };
                        if !class_type_args.is_empty() {
                            class_type_args
                                .iter()
                                .map(|&ty_id| self.convert_type(ty_id))
                                .collect::<Vec<_>>()
                        } else {
                            // Infer method's own type params from argument types
                            // (e.g., add<T>(x:T) called with String → T=String)
                            if let Some(func) = self.builder.module.functions.get(&func_id) {
                                if !func.signature.type_params.is_empty() {
                                    let mut inferred: Vec<IrType> = Vec::new();
                                    for type_param in &func.signature.type_params {
                                        let mut found = false;
                                        for (i, sig_param) in
                                            func.signature.parameters.iter().enumerate()
                                        {
                                            if let IrType::TypeVar(ref name) = sig_param.ty {
                                                if name == &type_param.name && i < args.len() {
                                                    let arg_type = self.convert_type(args[i].ty);
                                                    inferred.push(arg_type);
                                                    found = true;
                                                    break;
                                                }
                                            }
                                        }
                                        if !found
                                            && func.signature.type_params.len() == 1
                                            && args.len() > 1
                                        {
                                            // Single type param, infer from first non-this arg
                                            let arg_type = self.convert_type(args[1].ty);
                                            inferred.push(arg_type);
                                        }
                                    }
                                    inferred
                                } else {
                                    Vec::new()
                                }
                            } else {
                                Vec::new()
                            }
                        }
                    } else {
                        Vec::new()
                    };

                    debug!(
                        "[FUNCTION_MAP] Method call lowered {} args: {:?}, type_args: {:?}",
                        arg_regs.len(),
                        arg_regs,
                        ir_type_args
                    );

                    // Virtual dispatch: if this method is in a class hierarchy
                    // with overrides, dispatch through the vtable.
                    if let Some(&(slot_index, _)) = self.virtual_dispatch_info.get(symbol) {
                        if !arg_regs.is_empty() {
                            let receiver_reg = arg_regs[0];
                            let lookup_fn = self.get_or_register_extern_function(
                                "haxe_vtable_lookup",
                                vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                                IrType::I64,
                            );
                            let slot_reg =
                                self.builder.build_const(IrValue::I32(slot_index as i32));
                            if let Some(slot_r) = slot_reg {
                                if let Some(closure_ptr) = self.builder.build_call_direct(
                                    lookup_fn,
                                    vec![receiver_reg, slot_r],
                                    IrType::I64,
                                ) {
                                    let mut param_types = vec![IrType::Ptr(Box::new(IrType::Void))];
                                    for arg in args.iter().skip(1) {
                                        param_types.push(self.convert_type(arg.ty));
                                    }
                                    let return_type = Box::new(actual_return_type.clone());
                                    let func_signature = IrType::Function {
                                        params: param_types,
                                        return_type,
                                        varargs: false,
                                    };
                                    return self.builder.build_call_indirect(
                                        closure_ptr,
                                        arg_regs,
                                        func_signature,
                                    );
                                }
                            }
                        }
                    }

                    let result = if ir_type_args.is_empty() {
                        self.builder.build_call_direct(
                            func_id,
                            arg_regs,
                            actual_return_type.clone(),
                        )
                    } else {
                        self.builder.build_call_direct_with_type_args(
                            func_id,
                            arg_regs,
                            actual_return_type.clone(),
                            ir_type_args,
                        )
                    };
                    // Set class hint on result for cross-module method dispatch
                    if let Some(reg) = result {
                        self.set_class_hint_for_return(reg, expr.ty);
                    }
                    debug!("[FUNCTION_MAP] Result: {:?}", result);

                    // Type erasure coercion for generic method returns:
                    // The function returns I64 (type-erased), but the concrete
                    // return type may differ. Only apply to methods on generic classes —
                    // non-generic classes (Thread, Bytes, etc.) must NOT be coerced.
                    let receiver_is_generic = if !args.is_empty() {
                        let type_table = self.type_table;
                        type_table
                            .get(args[0].ty)
                            .map(|ti| match &ti.kind {
                                TypeKind::Class { type_args, .. } => !type_args.is_empty(),
                                TypeKind::GenericInstance { .. } => true,
                                TypeKind::TypeParameter { .. } => true,
                                _ => false,
                            })
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    if receiver_is_generic {
                        if let Some(call_result) = result {
                            if actual_return_type == IrType::I64 {
                                let expected_ir_type = self.convert_type(expr.ty);
                                if expected_ir_type != IrType::I64 {
                                    // Path 1: AST resolved type (e.g., Box<Int> → Ptr)
                                    return self.coerce_from_i64(call_result, expr.ty);
                                }
                                // Path 2: expr.ty is still TypeParameter → resolve via receiver's type_args
                                if !args.is_empty() {
                                    if let Some(concrete_ty_id) =
                                        self.resolve_type_param_from_receiver(expr.ty, args[0].ty)
                                    {
                                        let concrete_ir_type = self.convert_type(concrete_ty_id);
                                        if concrete_ir_type != IrType::I64 {
                                            return self
                                                .coerce_from_i64(call_result, concrete_ty_id);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    return result;
                } else {
                    // Direct function call (static method or free function)
                    // Track heap-allocated intermediates passed as arguments,
                    // but ONLY if the callee is a user-defined function.
                    // Stdlib/runtime functions (MirWrapper, ExternC) may store
                    // arguments (e.g., Array.push), so freeing would cause
                    // dangling pointers.
                    let callee_is_user_defined = self
                        .builder
                        .module
                        .functions
                        .get(&func_id)
                        .map(|f| f.kind == crate::ir::functions::FunctionKind::UserDefined)
                        .unwrap_or(false);

                    let mut call_args = self.effective_static_call_args(args);
                    if call_args.len() == args.len() {
                        if let Some(param_count) = function_param_count {
                            if args.len() == param_count + 1 && !args.is_empty() {
                                call_args = &args[1..];
                            }
                        }
                    }
                    let mut arg_regs = Vec::new();
                    for (param_idx, arg) in call_args.iter().enumerate() {
                        if let Some(reg) = self.lower_expression(arg) {
                            // Materialize anon-backed variables at call boundary
                            let reg =
                                self.maybe_materialize_for_call(arg, reg, Some(func_id), param_idx);
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

                    self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
                    // Fill in default values for any missing optional parameters
                    self.fill_default_args(func_id, &mut arg_regs, false);

                    // Last-chance parity guard for static-call symbol collisions:
                    // if the resolved function arity still does not match the call site,
                    // re-resolve by (method_name, arity) through stdlib static mapping.
                    if let Some(expected_params) = self
                        .builder
                        .module
                        .functions
                        .get(&func_id)
                        .map(|f| f.signature.parameters.len())
                    {
                        if expected_params != arg_regs.len() {
                            if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                                if let Some(method_name) = self.string_interner.get(sym_info.name) {
                                    if let Some((_sig, mapping)) =
                                        self.stdlib_mapping.find_static_method_by_name_and_params(
                                            method_name,
                                            call_args.len(),
                                        )
                                    {
                                        let runtime_func_name = mapping.runtime_name.to_string();
                                        let mut fallback_arg_regs: Vec<IrId> = Vec::new();
                                        let mut fallback_arg_types: Vec<IrType> = Vec::new();
                                        for arg in call_args.iter() {
                                            if let Some(reg) = self.lower_expression(arg) {
                                                fallback_arg_regs.push(reg);
                                                fallback_arg_types.push(self.convert_type(arg.ty));
                                            }
                                        }
                                        let (expected_param_types, expected_return_type) = self
                                            .get_extern_function_signature(&runtime_func_name)
                                            .unwrap_or_else(|| {
                                                (fallback_arg_types, actual_return_type.clone())
                                            });
                                        let runtime_func_id = self.get_or_register_extern_function(
                                            &runtime_func_name,
                                            expected_param_types,
                                            expected_return_type.clone(),
                                        );
                                        return self.builder.build_call_direct(
                                            runtime_func_id,
                                            fallback_arg_regs,
                                            expected_return_type,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    // Auto-box arguments when expected type is Ptr(U8) but actual is primitive
                    // This handles cases like Type.enumIndex(Color.Red) where the enum discriminant
                    // (raw i64) needs to be boxed as DynamicValue* for the runtime function.
                    if let Some(func) = self.builder.module.functions.get(&func_id) {
                        let expected_types: Vec<IrType> = func
                            .signature
                            .parameters
                            .iter()
                            .map(|p| p.ty.clone())
                            .collect();
                        for (i, expected_ty) in expected_types.iter().enumerate() {
                            if i < arg_regs.len() {
                                let is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8));
                                if is_ptr_u8 {
                                    if let Some(actual_ty) =
                                        self.builder.get_register_type(arg_regs[i])
                                    {
                                        if !matches!(actual_ty, IrType::Ptr(_))
                                            && i < call_args.len()
                                        {
                                            if let Some(boxed) = self
                                                .box_value_for_dynamic(arg_regs[i], call_args[i].ty)
                                            {
                                                arg_regs[i] = boxed;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Inject hidden enum type_id for enum helper runtime calls.
                    {
                        let func_name = self
                            .builder
                            .module
                            .functions
                            .get(&func_id)
                            .map(|f| f.name.clone())
                            .or_else(|| {
                                self.builder
                                    .module
                                    .extern_functions
                                    .get(&func_id)
                                    .map(|f| f.name.clone())
                            })
                            .unwrap_or_default();
                        if let Some(result) = self.try_lower_special_runtime_call(
                            &func_name,
                            call_args,
                            result_type.clone(),
                            expr.source_location.clone(),
                        ) {
                            return result;
                        }
                        self.inject_hidden_enum_type_id_arg(&func_name, call_args, &mut arg_regs);
                    }

                    // Infer type_args for static generic calls if not already provided
                    let final_type_args = if converted_hir_type_args.is_empty() {
                        // Check if the function has type parameters
                        if let Some(func) = self.builder.module.functions.get(&func_id) {
                            if !func.signature.type_params.is_empty() && !call_args.is_empty() {
                                // Try to infer type_args from argument types
                                debug!(
                                    "[TYPE INFERENCE] Function {} has type_params: {:?}",
                                    func.name, func.signature.type_params
                                );
                                debug!(
                                    "[TYPE INFERENCE] Function params: {:?}",
                                    func.signature
                                        .parameters
                                        .iter()
                                        .map(|p| (&p.name, &p.ty))
                                        .collect::<Vec<_>>()
                                );

                                let mut inferred: Vec<IrType> = Vec::new();
                                for (_param_idx, type_param) in
                                    func.signature.type_params.iter().enumerate()
                                {
                                    // Look for a parameter using this type variable
                                    let mut found = false;
                                    for (i, sig_param) in
                                        func.signature.parameters.iter().enumerate()
                                    {
                                        debug!(
                                            "[TYPE INFERENCE] Checking param {} type {:?} against type_param {}",
                                            sig_param.name, sig_param.ty, type_param.name
                                        );
                                        if let IrType::TypeVar(ref name) = sig_param.ty {
                                            if name == &type_param.name && i < call_args.len() {
                                                // Use the concrete type of the corresponding argument
                                                let arg_type = self.convert_type(call_args[i].ty);
                                                debug!(
                                                    "[TYPE INFERENCE] Inferred {}={:?} from arg {}",
                                                    type_param.name, arg_type, i
                                                );
                                                inferred.push(arg_type);
                                                found = true;
                                                break;
                                            }
                                        }
                                    }
                                    if !found {
                                        // Couldn't infer this type param from signature params
                                        // Try using the first argument's type as a fallback for single type param
                                        if func.signature.type_params.len() == 1
                                            && !call_args.is_empty()
                                        {
                                            let arg_type = self.convert_type(call_args[0].ty);
                                            debug!(
                                                "[TYPE INFERENCE] Fallback: Inferred {}={:?} from first arg",
                                                type_param.name, arg_type
                                            );
                                            inferred.push(arg_type);
                                        } else {
                                            debug!(
                                                "[TYPE INFERENCE] Could not infer {}, using Any",
                                                type_param.name
                                            );
                                            inferred.push(IrType::Any);
                                        }
                                    }
                                }
                                inferred
                            } else {
                                Vec::new()
                            }
                        } else {
                            Vec::new()
                        }
                    } else {
                        converted_hir_type_args.clone()
                    };

                    // Wrap arguments for constrained type parameters (T:Interface)
                    // If a parameter expects a fat pointer (constrained TypeParam),
                    // wrap the class argument in a fat pointer with the interface's vtable.
                    if let Some(constrained) =
                        self.constrained_param_interfaces.get(&func_id).cloned()
                    {
                        for (param_idx, iface_sym) in &constrained {
                            if *param_idx < arg_regs.len() && *param_idx < call_args.len() {
                                let arg_type = call_args[*param_idx].ty;
                                if let Some(class_sym) = self.get_class_symbol(arg_type) {
                                    if self
                                        .interface_vtables
                                        .contains_key(&(class_sym, *iface_sym))
                                    {
                                        if let Some(wrapped) = self.wrap_in_interface_fat_ptr(
                                            arg_regs[*param_idx],
                                            class_sym,
                                            *iface_sym,
                                        ) {
                                            arg_regs[*param_idx] = wrapped;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Use HIR type_args or inferred type_args for static generic calls
                    debug!(
                        "[FUNCTION_MAP] Direct call lowered {} args: {:?}, final_type_args: {:?}",
                        arg_regs.len(),
                        arg_regs,
                        final_type_args
                    );
                    let result = if final_type_args.is_empty() {
                        self.builder
                            .build_call_direct(func_id, arg_regs, actual_return_type)
                    } else {
                        self.builder.build_call_direct_with_type_args(
                            func_id,
                            arg_regs,
                            actual_return_type,
                            final_type_args,
                        )
                    };
                    debug!("[FUNCTION_MAP] Result: {:?}", result);
                    return result;
                }
            } else {
                // Function not in function_map - might be an extern/stdlib function
                // Check if it's a stdlib static method (like Math.sin, Sys.println)
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(method_name) = self.string_interner.get(sym_info.name) {
                        let static_args = self.effective_static_call_args(args);
                        // Check if method name matches known Math/Sys methods
                        // Try to find this method in ANY stdlib class with static methods
                        // This replaces the hardcoded is_math_method and is_sys_method checks
                        let method_static: &'static str =
                            Box::leak(method_name.to_string().into_boxed_str());

                        // Try all stdlib classes that have static methods
                        let mut found_mapping = None;
                        for class_name in self.stdlib_mapping.get_all_classes() {
                            if self.stdlib_mapping.class_has_static_methods(class_name) {
                                let sig = crate::stdlib::MethodSignature {
                                    class: class_name,
                                    method: method_static,
                                    is_static: true,
                                    is_constructor: false, // Normal static method, not constructor
                                    param_count: static_args.len(),
                                };
                                if let Some(mapping) = self.stdlib_mapping.get(&sig) {
                                    found_mapping = Some((class_name, mapping));
                                    break;
                                }
                            }
                        }

                        if let Some((class_name, mapping)) = found_mapping {
                            self.builder.call_label = Some(format!("STATIC_SEARCH:{}", class_name));
                            let runtime_name = mapping.runtime_name;
                            // eprintln!(
                            //     "INFO: {} static method detected: {} (runtime: {})",
                            //     class_name, method_name, runtime_name
                            // );

                            // Lower arguments and get their types
                            let mut arg_regs = Vec::new();
                            let mut arg_types = Vec::new();
                            for arg in static_args {
                                if let Some(reg) = self.lower_expression(arg) {
                                    arg_regs.push(reg);
                                    arg_types.push(self.convert_type(arg.ty));
                                }
                            }

                            // Reflect.compare: use haxe_reflect_compare_typed which accepts
                            // raw type-erased i64 values + a type tag, avoiding boxing.
                            // For generic code, the type tag is a placeholder resolved at
                            // monomorphization time.
                            if runtime_name == "haxe_reflect_compare" {
                                let mut type_param_name: Option<String> = None;
                                let mut known_type_tag: Option<i32> = None;
                                let mut use_typed_compare = false;

                                if let Some(first_arg) = static_args.first() {
                                    let type_table = self.type_table;
                                    if let Some(ti) = type_table.get(first_arg.ty) {
                                        match &ti.kind {
                                            TypeKind::TypeParameter { symbol_id, .. } => {
                                                use_typed_compare = true;
                                                // Get type param name from symbol table
                                                if let Some(sym) =
                                                    self.symbol_table.get_symbol(*symbol_id)
                                                {
                                                    if let Some(name_str) =
                                                        self.string_interner.get(sym.name)
                                                    {
                                                        type_param_name =
                                                            Some(name_str.to_string());
                                                    }
                                                }
                                            }
                                            TypeKind::Int => {
                                                use_typed_compare = true;
                                                known_type_tag = Some(1);
                                            }
                                            TypeKind::Float => {
                                                use_typed_compare = true;
                                                known_type_tag = Some(4);
                                            }
                                            TypeKind::Bool => {
                                                use_typed_compare = true;
                                                known_type_tag = Some(2);
                                            }
                                            TypeKind::String => {
                                                use_typed_compare = true;
                                                known_type_tag = Some(5);
                                            }
                                            _ => {} // Dynamic/other: fall through to boxing path
                                        }
                                    }
                                }

                                if use_typed_compare {
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

                                    // Emit type tag constant (placeholder 0 for generics, real value for concrete)
                                    let tag_reg = if let Some(tp_name) = type_param_name {
                                        let tag = self.builder.build_const(IrValue::I32(0))?;
                                        // Record fixup for the monomorphize pass to resolve
                                        if let Some(func) = self.builder.current_function_mut() {
                                            func.type_param_tag_fixups.push((tag, tp_name));
                                        }
                                        tag
                                    } else {
                                        self.builder.build_const(IrValue::I32(
                                            known_type_tag.unwrap_or(1),
                                        ))?
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
                                } else {
                                    // Dynamic case: box arguments for haxe_reflect_compare
                                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                    for (i, arg) in static_args.iter().enumerate() {
                                        if i >= arg_regs.len() {
                                            break;
                                        }
                                        if let Some(boxed) =
                                            self.box_value_for_dynamic(arg_regs[i], arg.ty)
                                        {
                                            arg_regs[i] = boxed;
                                            arg_types[i] = ptr_u8.clone();
                                        }
                                    }
                                }
                            }

                            // Register the external runtime function
                            let extern_func_id = self.get_or_register_extern_function(
                                runtime_name,
                                arg_types,
                                result_type.clone(),
                            );

                            // Generate call to external function
                            return self.builder.build_call_direct(
                                extern_func_id,
                                arg_regs,
                                result_type,
                            );
                        }
                    }
                }
            }
        }

        // Before falling through to indirect call, try to look up by name or register a forward reference
        // for unresolved static method calls (cross-module dependencies during stdlib compilation)
        probe!(self.try_forward_declared_call(expr, result_type.clone()));

        self.lower_indirect_call(expr)
    }
}
