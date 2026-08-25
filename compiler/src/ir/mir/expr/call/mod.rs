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

mod array;
mod derived;
mod direct;
mod enum_ctor;
mod extern_class;
mod forward_ref;
mod future;
mod indirect;
mod interface;
mod intrinsic;
mod method;
mod native_struct;
mod resolved_method;
mod shader;
mod static_receiver;
mod stdlib_instance;
mod stdlib_runtime;
mod stdlib_static;
mod super_call;
mod virtual_dispatch;

impl<'a> HirToMirContext<'a> {
    pub(crate) fn lower_call(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Call {
            callee,
            args,
            is_method,
            type_args: hir_type_args,
            // Carried from TAST; the shape probes below still derive the
            // target themselves.
            target: _resolved_target,
        } = &expr.kind
        else {
            unreachable!("lower_call on a non-Call expression")
        };
        // RAYZOR_PROBE_CALLTARGET=1 tabulates (target, callee shape).
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

        // call_label traces which path generated the call.
        self.builder.call_label = Some("CALL_START".to_string());
        let result_type = self.convert_type(expr.ty);

        // Update the caller's shadow-stack frame to this call-site line/col so the
        // trace shows where the call was made, not the function definition line.
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

        let converted_hir_type_args: Vec<IrType> = hir_type_args
            .iter()
            .map(|&ty_id| self.convert_type(ty_id))
            .collect();

        debug!(
            "[CALL] expr.ty={:?}, result_type={:?}, is_method={}",
            expr.ty, result_type, is_method
        );

        // @:async dispatch (.await/.poll/.isReady) on registers holding Future
        // handles. Method shape: callee = Variable(method_symbol), args[0] = receiver.
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
        probe!(self.try_method_call(expr, result_type.clone()));

        // Enum constructors can arrive as field callees for imported
        // modules, e.g. `ForeignMetaish.U32(2048)`. Lower those here
        // before the callee expression itself turns `Enum.Variant`
        // into a tag-only value and drops the payload arguments.
        probe!(self.try_enum_constructor_via_field(expr));

        // Enum constructors with payload arguments, e.g. MyResult.Ok(42).
        probe!(self.try_enum_constructor(expr));

        if let HirExprKind::Variable { symbol, .. } = &callee.kind {
            // super.method() must reach the parent's implementation directly, so it
            // bypasses the vtable that would otherwise select the override.
            let receiver_is_super = !args.is_empty() && matches!(args[0].kind, HirExprKind::Super);
            probe!(self.try_super_call(expr, receiver_is_super));
            probe!(self.try_virtual_dispatch(expr, receiver_is_super));

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

            // Resolve user-defined calls by symbol id first, which avoids bare-name
            // collisions (user "add" vs "rayzor_ssl_cert_add"). Extern/stdlib methods
            // and Dynamic/Interface receivers fall to the handlers below, which do the
            // auto-boxing, runtime mapping and fat-pointer extraction.
            probe!(self.try_resolved_function_call(
                expr,
                result_type.clone(),
                converted_hir_type_args.clone()
            ));

            // With is_method, args[0] is the receiver; an interface-typed receiver
            // dispatches through the fat pointer's vtable.
            probe!(self.try_interface_dispatch(expr));

            // Enum instance methods delegate to runtime functions registered in
            // runtime_mapping.rs, with (type_id, is_boxed) injected as extra params.
            if *is_method && !args.is_empty() {
                if let Some(Some(result)) = self.try_dispatch_enum_method(*symbol, args) {
                    return Some(result);
                }
            }

            // User-class methods must resolve before extern-class dispatch, or a user
            // Point2D.add matches a stdlib method (sys_deque_add). Classes with runtime
            // mappings (EReg) still go through get_stdlib_runtime_info.
            probe!(self.try_user_class_method_call(expr, result_type.clone()));

            // Extern class methods redirect to the runtime. A desugared MethodCall has
            // args[0] as the receiver; static methods have no receiver, so all args are
            // actual arguments.
            probe!(self.try_extern_class_method_call(expr, result_type.clone()));
            // Route global trace() to a type-specific trace function by argument type.
            probe!(self.try_trace_call(expr));

            // Std.string() arrives as a static call with 2 args (Std class + value);
            // route to a type-specific conversion by argument type.
            probe!(self.try_std_string_call(expr));

            // Static methods such as Thread.spawn() also arrive with is_method=true.
            probe!(self.try_stdlib_instance_call(expr, result_type.clone()));
            probe!(self.try_stdlib_static_call(expr, result_type.clone()));

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
                    for (_tid, decl) in self.current_hir_types.iter() {
                        if let crate::ir::hir::HirTypeDecl::Class(c) = decl {
                            let is_shader = self
                                .symbol_table
                                .get_symbol(c.symbol_id)
                                .map(|s| s.flags.is_shader())
                                .unwrap_or(false);
                            if is_shader {
                                let type_table = self.type_table;
                                match crate::ir::wgsl_transpiler::transpile_shader_from_hir(
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

            // Fall back to qualified-name lookup: symbol ids differ between modules, and
            // an intra-module static call site can carry a different symbol than the
            // method definition.
            if func_id_opt.is_none() {
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(qual_name) = sym_info.qualified_name {
                        if let Some(qual_name_str) = self.string_interner.get(qual_name) {
                            // Local function_map first, then external.
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
                        // No qualified name on the call-site symbol (cross-class static
                        // calls in multi-class files): search function_map by bare name.
                        // Never the currently-compiling function — a same-named local
                        // would self-bind into infinite recursion — and for method calls
                        // never a candidate whose class differs from the receiver's.
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
                            // No bare-name search in external_function_map: across modules
                            // it false-positives (ListNode.create() -> rayzor_tcc_create).
                            // Cross-module calls must match on qualified name.
                        }
                    }
                }
            }

            // Name lookup in function_map: a chained method call (z.mul(z).add(c)) carries
            // a different symbol id than the definition. A bare-name match must be confirmed
            // against the receiver's class, or "get"/"set"/"toString" hit stdlib functions.
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
                        // Pass 1: strict class-name matching.
                        for (func_sym, &func_id) in &self.function_map {
                            if let Some(func_sym_info) = self.symbol_table.get_symbol(*func_sym) {
                                if let Some(func_name) =
                                    self.string_interner.get(func_sym_info.name)
                                {
                                    if func_name == method_name {
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

                // Use the function's actual return type, not expr.ty. Check both functions
                // (local) and extern_functions (forward refs to stdlib).
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
                    // Not in the module yet: a forward ref to a stdlib MIR wrapper, whose
                    // signature is looked up by function name.
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

                // Method call: args already includes the receiver as args[0].
                if *is_method && !treat_as_static_call {
                    // Track non-receiver args as temps only for user-defined callees;
                    // stdlib/runtime methods (Array.push) may store their arguments.
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
                            // Materialize anon-backed variables at the call boundary (the
                            // receiver is skipped). HIR param_types exclude `this`, so the
                            // param index is i - 1.
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
                    let call_arg_types: Vec<TypeId> = args.iter().map(|a| a.ty).collect();
                    self.bind_skipped_optional_args(func_id, &mut arg_regs, &call_arg_types, true);
                    self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, true);
                    self.fill_default_args(func_id, &mut arg_regs, true);

                    // Extract type_args for generic method calls.
                    // Priority: 1) HIR type_args, 2) class type_args, 3) infer from args
                    let ir_type_args = if !converted_hir_type_args.is_empty() {
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

                    let call_arg_types: Vec<TypeId> = args.iter().map(|a| a.ty).collect();
                    self.bind_skipped_optional_args(func_id, &mut arg_regs, &call_arg_types, false);
                    self.coerce_args_for_cross_module_call(func_id, &mut arg_regs, false);
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
                        if let Some(func) = self.builder.module.functions.get(&func_id) {
                            if !func.signature.type_params.is_empty() && !call_args.is_empty() {
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
                // Not in function_map: may be a stdlib static method (Math.sin, Sys.println),
                // searched for across every stdlib class that has static methods.
                if let Some(sym_info) = self.symbol_table.get_symbol(*symbol) {
                    if let Some(method_name) = self.string_interner.get(sym_info.name) {
                        let static_args = self.effective_static_call_args(args);
                        let method_static: &'static str =
                            Box::leak(method_name.to_string().into_boxed_str());

                        // The method symbol's qualified name names its owner
                        // ("Math.sin" names Math), so the owner answers. Only
                        // when the symbol carries no owner is the method name
                        // consulted alone, and then only if it identifies one
                        // binding — never a pick among classes that share it.
                        let arity = static_args.len();
                        let found_mapping = self
                            .class_key_from_method_qname(
                                sym_info
                                    .qualified_name
                                    .and_then(|qn| self.string_interner.get(qn)),
                            )
                            .and_then(|key| {
                                self.stdlib_mapping
                                    .get(&crate::stdlib::MethodSignature {
                                        class: key.as_str(),
                                        method: method_static,
                                        is_static: true,
                                        is_constructor: false,
                                        param_count: arity,
                                    })
                                    .map(|mapping| (key, mapping))
                            })
                            .or_else(|| {
                                self.stdlib_mapping
                                    .find_unique_static_by_name_and_params(method_static, arity)
                                    .map(|(sig, mapping)| {
                                        (self.stdlib_mapping.key(sig.class), mapping)
                                    })
                            });

                        if let Some((class_name, mapping)) = found_mapping {
                            self.builder.call_label = Some(format!("STATIC_SEARCH:{}", class_name));
                            let runtime_name = mapping.runtime_name;

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

                            let extern_func_id = self.get_or_register_extern_function(
                                runtime_name,
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
                }
            }
        }

        // Last before the indirect call: look up by name, or register a forward reference
        // for static calls left unresolved by cross-module stdlib compilation.
        probe!(self.try_forward_declared_call(expr, result_type.clone()));

        self.lower_indirect_call(expr)
    }
}
