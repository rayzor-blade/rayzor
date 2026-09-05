//! Dispatch thunks and the generated vtable-init function.

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
    /// A mapped static method taken as a value: a function shaped like the
    /// reference's own type that boxes its arguments for the runtime's
    /// `Dynamic` formals and calls the mapping, so `a.sort(Reflect.compare)`
    /// gets the comparator a user static would be. `None` when the symbol
    /// names no such mapping.
    pub(crate) fn mapped_static_function_ref(
        &mut self,
        symbol: SymbolId,
        ref_ty: TypeId,
    ) -> Option<IrFunctionId> {
        let (runtime_name, param_count) = {
            let sym = self.symbol_table.get_symbol(symbol)?;
            if sym.kind != crate::tast::SymbolKind::Function {
                return None;
            }
            let qname = sym
                .qualified_name
                .and_then(|n| self.string_interner.get(n))?
                .to_string();
            let (class_name, method) = qname.rsplit_once('.')?;
            let key = self.stdlib_mapping.class_key(class_name)?;
            let (sig, call) = self.stdlib_mapping.find_by_name(key, method)?;
            if !sig.is_static || sig.is_constructor || call.is_mir_wrapper {
                return None;
            }
            (call.runtime_name, call.param_count)
        };
        let (param_tys, ret_ty) = self.resolve_function_type_signature(ref_ty)?;
        if param_tys.len() != param_count {
            return None;
        }
        let thunk_name = format!("__mapped_static_ref__{}__{}", runtime_name, ref_ty.as_raw());
        for (func_id, func) in &self.builder.module.functions {
            if func.name == thunk_name {
                return Some(*func_id);
            }
        }
        let ret_ir = self.convert_type(ret_ty);
        let mut sig_builder = FunctionSignatureBuilder::new()
            .returns(ret_ir.clone())
            .calling_convention(CallingConvention::Haxe);
        for (i, t) in param_tys.iter().enumerate() {
            sig_builder = sig_builder.param(format!("a{i}"), self.convert_type(*t));
        }
        let thunk_sig = sig_builder.build();
        let thunk_symbol = SymbolId::from_raw(u32::MAX - 3000 - self.next_wrapper_id);
        self.next_wrapper_id += 1;

        let saved_current_function = self.builder.current_function;
        let saved_current_block = self.builder.current_block;
        let saved_symbol_map = self.symbol_map.clone();
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();
        let thunk_id = self
            .builder
            .start_function(thunk_symbol, thunk_name, thunk_sig);
        let restore = |ctx: &mut Self| {
            ctx.check_move_flow();
            ctx.builder.finish_function();
            ctx.builder.current_function = saved_current_function;
            ctx.builder.current_block = saved_current_block;
            ctx.symbol_map = saved_symbol_map.clone();
            ctx.strict_move_locals = saved_strict_move_locals.clone();
        };
        let mut params = Vec::with_capacity(param_count);
        for i in 0..param_count {
            match self.builder.current_function().and_then(|f| f.get_param_reg(i)) {
                Some(reg) => params.push(reg),
                None => {
                    restore(self);
                    return None;
                }
            }
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let dynamic = self.type_table.dynamic_type();
        let mut args = Vec::with_capacity(param_count);
        for (reg, ty) in params.into_iter().zip(param_tys.iter()) {
            let boxed = self.maybe_box_value(reg, *ty, dynamic).unwrap_or(reg);
            // A parameter still erased here carries raw bits; the callee's
            // `Dynamic` formal reads a box, so box them as an int.
            let boxed = if boxed == reg
                && !matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_)))
            {
                let v64 = match self.builder.get_register_type(reg) {
                    Some(IrType::I32) => self.builder.build_cast(reg, IrType::I32, IrType::I64)?,
                    _ => reg,
                };
                let box_fn = self.get_or_register_extern_function(
                    "haxe_box_int_ptr",
                    vec![IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_fn, vec![v64], ptr_u8.clone())?
            } else {
                boxed
            };
            args.push(boxed);
        }
        let extern_id = self.get_or_register_extern_function(
            runtime_name,
            vec![ptr_u8; param_count],
            IrType::I64,
        );
        let native_ret = self
            .builder
            .module
            .functions
            .get(&extern_id)
            .map(|f| f.signature.return_type.clone())
            .unwrap_or(IrType::I64);
        if matches!(ret_ir, IrType::Void) {
            self.builder.build_call_direct(extern_id, args, IrType::Void);
            self.builder.build_return(None);
        } else {
            let result = self
                .builder
                .build_call_direct(extern_id, args, native_ret.clone())
                .map(|r| self.reconcile_extern_return(r, &native_ret, &ret_ir));
            self.builder.build_return(result);
        }
        restore(self);
        Some(thunk_id)
    }

    /// Generate (or return cached) a virtual/interface dispatch thunk.
    ///
    /// Thunk ABI is `(env, this, ...args)`: the indirect-call convention used by
    /// vtable slots prepends a closure env that class methods don't declare, so
    /// the thunk drops `env` and forwards to the real method.
    pub(crate) fn ensure_vtable_dispatch_thunk(
        &mut self,
        method_func_id: IrFunctionId,
    ) -> Option<IrFunctionId> {
        if let Some(cached) = self.vtable_dispatch_thunks.get(&method_func_id) {
            return Some(*cached);
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let (method_sig, method_qname) = {
            let func = self.builder.module.functions.get(&method_func_id)?;
            let qname = func
                .qualified_name
                .clone()
                .unwrap_or_else(|| func.name.clone());
            (func.signature.clone(), qname)
        };
        if method_sig.parameters.is_empty() {
            return None;
        }

        let mut sig_builder = FunctionSignatureBuilder::new()
            .param("env".to_string(), ptr_u8.clone())
            .returns(method_sig.return_type.clone())
            .calling_convention(CallingConvention::Haxe);
        for param in &method_sig.parameters {
            sig_builder = sig_builder.param(param.name.clone(), param.ty.clone());
        }
        let thunk_sig = sig_builder.build();

        let thunk_symbol = SymbolId::from_raw(u32::MAX - 3000 - self.next_wrapper_id);
        self.next_wrapper_id += 1;
        // Name the thunk by the target method's QUALIFIED name, not its raw
        // module-local func id: local ids collide across modules post-merge and
        // the LLVM declare pass dedupes by name, so a stale FunctionRef would
        // bind to another module's thunk. A qualified-name thunk is
        // behaviorally identical whichever module's copy it lands on.
        let sanitized_qname: String = method_qname
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let thunk_name = format!("__vtable_dispatch_thunk__{}", sanitized_qname);

        let saved_current_function = self.builder.current_function;
        let saved_current_block = self.builder.current_block;
        let saved_symbol_map = self.symbol_map.clone();
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();

        let thunk_id = self
            .builder
            .start_function(thunk_symbol, thunk_name, thunk_sig);

        let call_args = {
            let Some(func) = self.builder.current_function() else {
                self.check_move_flow();
                self.builder.finish_function();
                self.builder.current_function = saved_current_function;
                self.builder.current_block = saved_current_block;
                self.symbol_map = saved_symbol_map;
                self.strict_move_locals = saved_strict_move_locals;
                return None;
            };
            let mut args = Vec::with_capacity(method_sig.parameters.len());
            for i in 0..method_sig.parameters.len() {
                if let Some(reg) = func.get_param_reg(i + 1) {
                    args.push(reg);
                } else {
                    self.check_move_flow();
                    self.builder.finish_function();
                    self.builder.current_function = saved_current_function;
                    self.builder.current_block = saved_current_block;
                    self.symbol_map = saved_symbol_map;
                    self.strict_move_locals = saved_strict_move_locals;
                    return None;
                }
            }
            args
        };

        let ret_ty = method_sig.return_type.clone();
        if matches!(ret_ty, IrType::Void) {
            self.builder
                .build_call_direct(method_func_id, call_args, IrType::Void);
            self.builder.build_return(None);
        } else {
            let result = self
                .builder
                .build_call_direct(method_func_id, call_args, ret_ty.clone());
            self.builder.build_return(result);
        }

        self.check_move_flow();
        self.builder.finish_function();
        self.builder.current_function = saved_current_function;
        self.builder.current_block = saved_current_block;
        self.symbol_map = saved_symbol_map;
        self.strict_move_locals = saved_strict_move_locals;

        self.vtable_dispatch_thunks.insert(method_func_id, thunk_id);
        Some(thunk_id)
    }

    /// Cross-module variant of `ensure_vtable_dispatch_thunk`: the method's
    /// IrFunction lives in another module's IrModule, so its MIR signature
    /// isn't reachable through `self.builder.module.functions`. Derive the
    /// signature from the method SYMBOL's function type instead and emit a
    /// local thunk that CallDirects the cross-module id (resolved by the
    /// same fixups every other cross-module direct call uses). Reuses any
    /// already-imported thunk by qualified name first.
    pub(crate) fn ensure_cross_module_dispatch_thunk(
        &mut self,
        method_sym: SymbolId,
        method_func_id: IrFunctionId,
    ) -> Option<IrFunctionId> {
        if let Some(cached) = self.vtable_dispatch_thunks.get(&method_func_id) {
            return Some(*cached);
        }

        let symbol = self.symbol_table.get_symbol(method_sym)?;
        let qname = symbol
            .qualified_name
            .or(Some(symbol.name))
            .and_then(|n| self.string_interner.get(n))
            .map(|s| s.to_string())?;
        let sanitized_qname: String = qname
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let thunk_name = format!("__vtable_dispatch_thunk__{}", sanitized_qname);

        // Reuse the thunk another module already emitted for this method.
        if let Some(&existing) = self.external_function_name_map.get(&thunk_name) {
            self.vtable_dispatch_thunks.insert(method_func_id, existing);
            return Some(existing);
        }

        let (param_type_ids, ret_type_id) = self.resolve_function_type_signature(symbol.type_id)?;
        let param_ir_types: Vec<IrType> = param_type_ids
            .iter()
            .map(|t| self.convert_type(*t))
            .collect();
        let ret_ir_type = self.convert_type(ret_type_id);

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
        let mut sig_builder = FunctionSignatureBuilder::new()
            .param("env".to_string(), ptr_u8)
            .param("this".to_string(), ptr_void)
            .returns(ret_ir_type.clone())
            .calling_convention(CallingConvention::Haxe);
        for (i, ty) in param_ir_types.iter().enumerate() {
            sig_builder = sig_builder.param(format!("p{}", i), ty.clone());
        }
        let thunk_sig = sig_builder.build();

        let thunk_symbol = SymbolId::from_raw(u32::MAX - 3000 - self.next_wrapper_id);
        self.next_wrapper_id += 1;

        let saved_current_function = self.builder.current_function;
        let saved_current_block = self.builder.current_block;
        let saved_symbol_map = self.symbol_map.clone();
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();

        let restore = |s: &mut Self| {
            s.builder.finish_function();
            s.builder.current_function = saved_current_function;
            s.builder.current_block = saved_current_block;
        };

        let thunk_id = self
            .builder
            .start_function(thunk_symbol, thunk_name, thunk_sig);

        let call_args = {
            let Some(func) = self.builder.current_function() else {
                restore(self);
                self.symbol_map = saved_symbol_map;
                self.strict_move_locals = saved_strict_move_locals;
                return None;
            };
            // Skip env (param 0); forward this + user params.
            let mut args = Vec::with_capacity(1 + param_ir_types.len());
            for i in 0..(1 + param_ir_types.len()) {
                if let Some(reg) = func.get_param_reg(i + 1) {
                    args.push(reg);
                } else {
                    restore(self);
                    self.symbol_map = saved_symbol_map;
                    self.strict_move_locals = saved_strict_move_locals;
                    return None;
                }
            }
            args
        };

        if matches!(ret_ir_type, IrType::Void) {
            self.builder
                .build_call_direct(method_func_id, call_args, IrType::Void);
            self.builder.build_return(None);
        } else {
            let result =
                self.builder
                    .build_call_direct(method_func_id, call_args, ret_ir_type.clone());
            self.builder.build_return(result);
        }

        self.check_move_flow();
        self.builder.finish_function();
        self.builder.current_function = saved_current_function;
        self.builder.current_block = saved_current_block;
        self.symbol_map = saved_symbol_map;
        self.strict_move_locals = saved_strict_move_locals;

        self.vtable_dispatch_thunks.insert(method_func_id, thunk_id);
        Some(thunk_id)
    }

    /// Generate (or return cached) a thunk for `obj.method` lvalue
    /// support. The thunk bridges the closure-call ABI
    /// `fn_ptr(env_ptr, ...args)` to the underlying method's
    /// `(this, ...args)` signature: it loads `this` from `env[0]`
    /// (where `MakeClosure` puts the first captured value) and
    /// forwards to the method.
    ///
    /// Returns `None` if the method's signature isn't available in
    /// this module (e.g. cross-module method from a `.blade` cache).
    pub(crate) fn ensure_method_ref_thunk(
        &mut self,
        method_func_id: IrFunctionId,
    ) -> Option<IrFunctionId> {
        if let Some(cached) = self.method_ref_thunks.get(&method_func_id) {
            return Some(*cached);
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let method_sig = self
            .builder
            .module
            .functions
            .get(&method_func_id)?
            .signature
            .clone();
        if method_sig.parameters.is_empty() {
            return None;
        }

        // Thunk signature: `(env: *u8, ...method_params_without_this) -> method_return`.
        let mut sig_builder = FunctionSignatureBuilder::new()
            .param("env".to_string(), ptr_u8.clone())
            .returns(method_sig.return_type.clone())
            .calling_convention(CallingConvention::Haxe);
        for param in method_sig.parameters.iter().skip(1) {
            sig_builder = sig_builder.param(param.name.clone(), param.ty.clone());
        }
        let thunk_sig = sig_builder.build();

        let thunk_symbol = SymbolId::from_raw(u32::MAX - 2000 - self.next_wrapper_id);
        self.next_wrapper_id += 1;
        let thunk_name = format!("__method_ref_thunk_{}", method_func_id.0);

        // Save outer function/block state — thunk generation may run *during*
        // lowering of another function (e.g. `var f = obj.m;` inside main).
        // `start_function` re-targets the builder, so without this the outer
        // function's `build_*` calls leak into the thunk's body.
        let saved_current_function = self.builder.current_function;
        let saved_current_block = self.builder.current_block;
        let saved_symbol_map = self.symbol_map.clone();
        // Per-function isolation: thunk has its own SSA register namespace;
        // snapshot and clear strict_move_locals so MarkMoved/CheckLive emitted
        // inside the thunk body do not collide with the outer function's IrIds.
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();

        let thunk_id = self
            .builder
            .start_function(thunk_symbol, thunk_name, thunk_sig);

        // Pull out env + the forwarded method args from the thunk's
        // parameter registers.
        let (env_reg, forward_regs) = {
            let Some(func) = self.builder.current_function() else {
                self.check_move_flow();
                self.builder.finish_function();
                self.builder.current_function = saved_current_function;
                self.builder.current_block = saved_current_block;
                self.symbol_map = saved_symbol_map;
                self.strict_move_locals = saved_strict_move_locals;
                return None;
            };
            let Some(env) = func.get_param_reg(0) else {
                self.check_move_flow();
                self.builder.finish_function();
                self.builder.current_function = saved_current_function;
                self.builder.current_block = saved_current_block;
                self.symbol_map = saved_symbol_map;
                self.strict_move_locals = saved_strict_move_locals;
                return None;
            };
            let mut args: Vec<IrId> = Vec::new();
            for i in 1..method_sig.parameters.len() {
                if let Some(reg) = func.get_param_reg(i) {
                    args.push(reg);
                }
            }
            (env, args)
        };

        // Load `this` from env[0]. The closure-ABI env points at a
        // heap struct whose first slot holds the captured receiver.
        let this_ty = method_sig.parameters[0].ty.clone();
        let this_loaded = self.builder.build_load(env_reg, ptr_u8.clone());
        let this_arg = this_loaded.map(|raw_ptr| {
            if this_ty == ptr_u8 {
                raw_ptr
            } else {
                self.builder
                    .build_cast(raw_ptr, ptr_u8.clone(), this_ty.clone())
                    .unwrap_or(raw_ptr)
            }
        });

        let Some(this_arg) = this_arg else {
            self.check_move_flow();
            self.builder.finish_function();
            self.symbol_map = saved_symbol_map;
            self.strict_move_locals = saved_strict_move_locals;
            return None;
        };

        let mut call_args: Vec<IrId> = Vec::with_capacity(1 + forward_regs.len());
        call_args.push(this_arg);
        call_args.extend(forward_regs);

        let ret_ty = method_sig.return_type.clone();
        if matches!(ret_ty, IrType::Void) {
            self.builder
                .build_call_direct(method_func_id, call_args, IrType::Void);
            self.builder.build_return(None);
        } else {
            let result = self
                .builder
                .build_call_direct(method_func_id, call_args, ret_ty.clone());
            self.builder.build_return(result);
        }

        self.check_move_flow();
        self.builder.finish_function();

        // Restore outer function/block context.
        self.builder.current_function = saved_current_function;
        self.builder.current_block = saved_current_block;
        self.symbol_map = saved_symbol_map;
        self.strict_move_locals = saved_strict_move_locals;

        self.method_ref_thunks.insert(method_func_id, thunk_id);
        Some(thunk_id)
    }

    pub(crate) fn generate_vtable_init_function(&mut self) {
        // __vtable_init__ registers class vtables at startup; the backend calls
        // it before main(), same as __init__.
        let sig = FunctionSignatureBuilder::new()
            .returns(IrType::Void)
            .calling_convention(CallingConvention::Haxe)
            .build();

        let vtable_init_symbol = SymbolId::from_raw(u32::MAX - 2);
        let _func_id =
            self.builder
                .start_function(vtable_init_symbol, "__vtable_init__".to_string(), sig);

        let saved_symbol_map = self.symbol_map.clone();
        // Per-function isolation: __vtable_init__ has its own SSA namespace;
        // snapshot and clear strict_move_locals in parallel with symbol_map.
        let saved_strict_move_locals = self.strict_move_locals.clone();
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();

        let vtable_init_fn = self.get_or_register_extern_function(
            "haxe_vtable_init",
            vec![IrType::I32, IrType::I32],
            IrType::Void,
        );
        let vtable_set_fn = self.get_or_register_extern_function(
            "haxe_vtable_set_slot",
            vec![IrType::I32, IrType::I32, IrType::I64],
            IrType::Void,
        );
        let register_ctor_fn = self.get_or_register_extern_function(
            "haxe_type_register_constructor",
            vec![IrType::I64, IrType::I64],
            IrType::Void,
        );
        let register_iface_impl_fn = self.get_or_register_extern_function(
            "haxe_register_interface_impl",
            vec![IrType::I64, IrType::I64],
            IrType::Void,
        );
        let iface_vtable_set_slot_fn = self.get_or_register_extern_function(
            "haxe_iface_vtable_set_slot",
            vec![IrType::I32, IrType::I32, IrType::I32, IrType::I64],
            IrType::Void,
        );

        let class_vtables = self.class_vtables.clone();
        for (class_sym, vtable) in &class_vtables {
            // Vtable keys must match the value stored in object headers
            // (deterministic name-hash), since virtual dispatch reads the
            // header and looks up the vtable by that id. Fall back to the
            // legacy SymbolId encoding if name resolution fails.
            let type_id = self
                .deterministic_class_type_id(*class_sym)
                .map(|h| h as i32)
                .unwrap_or(class_sym.as_raw() as i32);
            let slot_count = vtable.len() as i32;

            let type_id_reg = self.builder.build_const(IrValue::I32(type_id));
            let slot_count_reg = self.builder.build_const(IrValue::I32(slot_count));
            if let (Some(tid), Some(sc)) = (type_id_reg, slot_count_reg) {
                self.builder
                    .build_call_direct(vtable_init_fn, vec![tid, sc], IrType::Void);
            }

            for (slot_idx, method_sym) in vtable.iter().enumerate() {
                if let Some(&func_id) = self.function_map.get(method_sym) {
                    let dispatch_func_id = self
                        .ensure_vtable_dispatch_thunk(func_id)
                        .unwrap_or(func_id);
                    let closure_ptr = self.builder.build_function_ref(dispatch_func_id);
                    let slot_reg = self.builder.build_const(IrValue::I32(slot_idx as i32));
                    let type_id_reg2 = self.builder.build_const(IrValue::I32(type_id));
                    if let (Some(cp), Some(sr), Some(tid2)) = (closure_ptr, slot_reg, type_id_reg2)
                    {
                        self.builder.build_call_direct(
                            vtable_set_fn,
                            vec![tid2, sr, cp],
                            IrType::Void,
                        );
                    }
                }
            }
        }

        // Register class -> interface implementation pairs for Std.is(..., Interface)
        // and interface-method dispatch. Use the deterministic name-hash for
        // both sides so the runtime's class_implements lookup matches what the
        // emit sites pass for type-id comparisons.
        //
        // Also register per-(class, iface, slot) closure pointers so that
        // iface-to-iface casts (e.g. `cast(model:Module, CausalLanguageModel)`)
        // can call `haxe_iface_fat_ptr_build` and rebuild a fat pointer with the
        // target interface's full method set. A pass-through cast would keep the
        // source-shaped fat ptr, and the wider interface's slot reads would run
        // off the end of the source vtable.
        let mut registered_iface_pairs: BTreeSet<(i64, i64)> = BTreeSet::new();
        let interface_vtables = self.interface_vtables.clone();
        for ((class_sym, iface_sym), methods) in &interface_vtables {
            // These ids must match the ones the cast emit sites and the runtime
            // registrations use; a context-local id would not.
            let class_tid = self.deterministic_class_type_id(*class_sym);
            let iface_tid = self.deterministic_iface_or_enum_type_id(*iface_sym, "iface");
            if std::env::var_os("RAYZOR_IFACE_DEBUG").is_some() {
                let names: Vec<&str> = methods
                    .iter()
                    .map(|m| {
                        self.symbol_table
                            .get_symbol(*m)
                            .and_then(|s| self.string_interner.get(s.name))
                            .unwrap_or("?")
                    })
                    .collect();
                eprintln!(
                    "[iface_vtable_emit] class_tid={:?} iface_tid={:?} methods={:?}",
                    class_tid, iface_tid, names
                );
            }
            if let (Some(class_tid), Some(iface_tid)) = (class_tid, iface_tid) {
                for (slot_idx, method_sym) in methods.iter().enumerate() {
                    // Only locally-compiled methods: imported ones get their slot
                    // from their own file's `__vtable_init__`. An id out of
                    // `external_function_map` is renumbered and unresolvable by
                    // `build_function_ref`, which trap-stubs the whole init.
                    let func_id = self.function_map.get(method_sym).copied();
                    if let Some(func_id) = func_id {
                        let dispatch_func_id = self
                            .ensure_vtable_dispatch_thunk(func_id)
                            .unwrap_or(func_id);
                        let closure_ptr = self.builder.build_function_ref(dispatch_func_id);
                        let class_tid_reg =
                            self.builder.build_const(IrValue::I32(class_tid as i32));
                        let iface_tid_reg =
                            self.builder.build_const(IrValue::I32(iface_tid as i32));
                        let slot_idx_reg = self.builder.build_const(IrValue::I32(slot_idx as i32));
                        if let (Some(ctid), Some(itid), Some(sidx), Some(cp)) =
                            (class_tid_reg, iface_tid_reg, slot_idx_reg, closure_ptr)
                        {
                            self.builder.build_call_direct(
                                iface_vtable_set_slot_fn,
                                vec![ctid, itid, sidx, cp],
                                IrType::Void,
                            );
                        }
                    }
                }
            }
        }
        for ((class_sym, iface_sym), _methods) in interface_vtables {
            // Register exactly the ids the runtime will look up: the object
            // header's class id and the id baked at the `is`/cast site, both
            // produced by the deterministic name hash. Registering extra
            // aliases (the symbol-id form, or that form + 1000) can only make a
            // check that should FAIL succeed — one class's shifted alias can
            // equal another class's real id, handing it interfaces it does not
            // implement. The lookup side matches exactly.
            let mut class_ids: BTreeSet<i64> = BTreeSet::new();
            match self.deterministic_class_type_id(class_sym) {
                Some(h) => {
                    class_ids.insert(h as i64);
                }
                // Unnamed class: mirror `runtime_type_id`'s fallback, which is
                // what the object header will carry.
                None => {
                    if let Some(sym) = self.symbol_table.get_symbol(class_sym) {
                        class_ids.insert(
                            self.resolve_runtime_class_type_id(sym.type_id, class_sym)
                                .as_raw() as i64
                                + 1000,
                        );
                    }
                }
            }

            let mut iface_ids: BTreeSet<i64> = BTreeSet::new();
            match self.deterministic_iface_or_enum_type_id(iface_sym, "iface") {
                Some(h) => {
                    iface_ids.insert(h as i64);
                }
                None => {
                    if let Some(sym) = self.symbol_table.get_symbol(iface_sym) {
                        iface_ids.insert(sym.type_id.as_raw() as i64);
                    }
                }
            }

            for class_type_id in &class_ids {
                for iface_type_id in &iface_ids {
                    if !registered_iface_pairs.insert((*class_type_id, *iface_type_id)) {
                        continue;
                    }
                    let class_tid = self.builder.build_const(IrValue::I64(*class_type_id));
                    let iface_tid = self.builder.build_const(IrValue::I64(*iface_type_id));
                    if let (Some(cid), Some(iid)) = (class_tid, iface_tid) {
                        self.builder.build_call_direct(
                            register_iface_impl_fn,
                            vec![cid, iid],
                            IrType::Void,
                        );
                    }
                }
            }
        }

        // Register constructor closure pointers for Type.createInstance.
        let ctor_wrappers = self.constructor_reflect_wrappers.clone();
        for (class_type_id, wrapper_func_id) in ctor_wrappers {
            // Key the ctor by the SAME deterministic id the class metadata is
            // registered under: `register_class_rtti_from_module` fills
            // TYPE_REGISTRY with `runtime_type_id` (FNV-1a of the qualified
            // name, stable across contexts), and `haxe_type_create_instance`
            // passes ONE id to both registries. A raw context-local TypeId here
            // would put CONSTRUCTOR_REGISTRY in a different key space.
            let stable_id = self
                .class_type_to_symbol
                .get(&class_type_id)
                .and_then(|sym| self.deterministic_class_type_id(*sym))
                .map(|v| v as i64)
                .unwrap_or(class_type_id.as_raw() as i64);
            let type_id_reg = self.builder.build_const(IrValue::I64(stable_id));
            let closure_ptr = self.builder.build_function_ref(wrapper_func_id);
            if let (Some(tid), Some(cptr)) = (type_id_reg, closure_ptr) {
                self.builder
                    .build_call_direct(register_ctor_fn, vec![tid, cptr], IrType::Void);
            }
        }

        self.builder.build_return(None);
        self.check_move_flow();
        self.builder.finish_function();
        self.symbol_map = saved_symbol_map;
        self.strict_move_locals = saved_strict_move_locals;
    }
}
