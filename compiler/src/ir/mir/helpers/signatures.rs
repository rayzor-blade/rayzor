//! Signature construction, default arguments, and call-site argument fixups.

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
    pub(crate) fn build_function_signature(
        &self,
        func: &HirFunction,
    ) -> crate::ir::IrFunctionSignature {
        let mut builder = FunctionSignatureBuilder::new();

        for type_param in &func.type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            builder = builder.type_param(param_name);
        }

        for param in &func.params {
            let param_type = self.convert_type(param.ty);
            let param_name = self
                .string_interner
                .get(param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("arg{}", param.name.as_raw()));
            builder = builder.param(param_name, param_type);
        }

        let return_type = self.convert_type(func.return_type);
        builder = builder.returns(return_type);

        if func.is_extern {
            builder = builder.calling_convention(CallingConvention::C);
        }

        builder.build()
    }

    /// Build function signature with class type parameters (for generic class methods)
    pub(crate) fn build_function_signature_with_class_type_params(
        &self,
        func: &HirFunction,
        class_type_params: &[HirTypeParam],
    ) -> crate::ir::IrFunctionSignature {
        let mut builder = FunctionSignatureBuilder::new();

        // Collect all type param names (class + method) for TypeVar resolution
        let mut type_param_names: Vec<String> = Vec::new();

        // Class type parameters precede the method's own.
        for type_param in class_type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            type_param_names.push(param_name.clone());
            builder = builder.type_param(param_name);
        }

        for type_param in &func.type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            type_param_names.push(param_name.clone());
            builder = builder.type_param(param_name);
        }

        for param in &func.params {
            let param_type = self.convert_type(param.ty);
            let param_name = self
                .string_interner
                .get(param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("arg{}", param.name.as_raw()));
            builder = builder.param(param_name, param_type);
        }

        // Use TypeVar only for return type so the monomorphizer and call-site
        // can resolve the concrete return type. Parameters stay I64 (type-erased)
        // to keep the function body compilable as a generic template.
        let return_type = self.convert_type_or_type_var(func.return_type, &type_param_names);
        builder = builder.returns(return_type);

        if func.is_extern {
            builder = builder.calling_convention(CallingConvention::C);
        }

        builder.build()
    }

    /// Build function signature for an instance method with implicit 'this' parameter
    pub(crate) fn build_instance_method_signature(
        &self,
        func: &HirFunction,
        class_type_id: TypeId,
    ) -> crate::ir::IrFunctionSignature {
        let mut builder = FunctionSignatureBuilder::new();

        for type_param in &func.type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            builder = builder.type_param(param_name);
        }

        // Implicit 'this' comes first and is always a pointer to the instance.
        let this_type = match self.convert_type(class_type_id) {
            IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
            // Also when convert_type couldn't resolve it (uninstantiated generic class).
            _ => IrType::Ptr(Box::new(IrType::Void)),
        };

        builder = builder.param("this".to_string(), this_type);

        for param in &func.params {
            let param_name = self
                .string_interner
                .get(param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("param_{}", param.symbol_id.as_raw()));
            let param_type = self.convert_type(param.ty);
            builder = builder.param(param_name, param_type);
        }

        let return_type = self.convert_type(func.return_type);
        builder = builder.returns(return_type);

        if func.is_extern {
            builder = builder.calling_convention(CallingConvention::C);
        }

        builder.build()
    }

    /// Place supplied arguments in the parameters they can actually occupy,
    /// filling any LEADING optional the caller skipped.
    ///
    /// `f(?a:Int, b:String)` called as `f("x")` binds `b`, not `a` -- Haxe lets
    /// a caller omit an optional and supply what follows. Binding positionally
    /// put the String in the Int slot, which reads back as a raw pointer and,
    /// when the parameter is a Bool the body branches on, reaches the backend as
    /// a pointer where a condition belongs.
    ///
    /// Only a mismatch that could not possibly be a conversion moves anything:
    /// a reference offered to a scalar parameter, or a scalar to a reference.
    /// Anything the existing coercions handle is left alone, so calls that
    /// already bind correctly are untouched.
    pub(crate) fn bind_skipped_optional_args(
        &mut self,
        func_id: IrFunctionId,
        arg_regs: &mut Vec<IrId>,
        arg_types: &[TypeId],
        has_implicit_this: bool,
    ) {
        let Some(optional) = self.function_param_optional.get(&func_id).cloned() else {
            return;
        };
        let Some(param_types) = self.function_param_hir_types.get(&func_id).cloned() else {
            return;
        };
        // An optional parameter need not carry a default: `?a:Int` is Null<Int>
        // and is filled with nothing rather than with a value.
        let defaults = self
            .function_param_defaults
            .get(&func_id)
            .cloned()
            .unwrap_or_else(|| vec![None; param_types.len()]);

        let offset = usize::from(has_implicit_this);
        let supplied = arg_regs.len().saturating_sub(offset);
        // Only a caller that left something out can have skipped anything.
        if supplied == 0 || supplied >= param_types.len() {
            return;
        }

        let bindable = |a: &IrType, p: &IrType| -> bool {
            let a_ref = matches!(a, IrType::Ptr(_) | IrType::String | IrType::Any);
            let p_ref = matches!(p, IrType::Ptr(_) | IrType::String | IrType::Any);
            a_ref == p_ref
        };

        let mut plan: Vec<Option<usize>> = Vec::with_capacity(param_types.len());
        let mut next_arg = 0usize;
        for (p_idx, p_ty) in param_types.iter().enumerate() {
            if next_arg >= supplied {
                plan.push(None);
                continue;
            }
            let arg_ir = self.convert_type(arg_types[next_arg]);
            let param_ir = self.convert_type(*p_ty);
            let can_skip = optional.get(p_idx).copied().unwrap_or(false);
            if !bindable(&arg_ir, &param_ir) && can_skip {
                plan.push(None);
            } else {
                plan.push(Some(next_arg));
                next_arg += 1;
            }
        }

        // Nothing was skipped, or an argument found no home -- leave the call as
        // it was rather than guessing at a shape this cannot describe.
        if next_arg != supplied || plan.iter().all(|slot| slot.is_some()) {
            return;
        }

        let supplied_regs: Vec<IrId> = arg_regs[offset..].to_vec();
        let mut rebound: Vec<IrId> = arg_regs[..offset].to_vec();
        for (p_idx, slot) in plan.iter().enumerate() {
            match slot {
                Some(a_idx) => rebound.push(supplied_regs[*a_idx]),
                None => {
                    let reg = match defaults.get(p_idx) {
                        Some(Some(default_expr)) => self.lower_expression(&default_expr.clone()),
                        // Skipped and undefaulted: the parameter is Null<T>, so
                        // it reads as absent rather than as a value.
                        _ => {
                            let empty = match self.convert_type(param_types[p_idx]) {
                                IrType::I32 | IrType::U32 => IrValue::I32(0),
                                IrType::F32 => IrValue::F32(0.0),
                                IrType::F64 => IrValue::F64(0.0),
                                IrType::Bool => IrValue::Bool(false),
                                _ => IrValue::I64(0),
                            };
                            self.builder.build_const(empty)
                        }
                    };
                    let Some(reg) = reg else {
                        return;
                    };
                    rebound.push(reg);
                }
            }
        }
        *arg_regs = rebound;
    }

    pub(crate) fn fill_default_args(
        &mut self,
        func_id: IrFunctionId,
        arg_regs: &mut Vec<IrId>,
        has_implicit_this: bool,
    ) {
        let user_arg_count = if has_implicit_this {
            arg_regs.len().saturating_sub(1)
        } else {
            arg_regs.len()
        };

        if std::env::var_os("RAYZOR_DEFAULTS_DEBUG").is_some() {
            let name = self
                .builder
                .module
                .functions
                .get(&func_id)
                .map(|f| f.name.clone())
                .unwrap_or_else(|| "<unknown>".to_string());
            eprintln!(
                "[defaults] {} id={:?} args={} implicit_this={} has_defaults={} has_optional={}",
                name,
                func_id,
                arg_regs.len(),
                has_implicit_this,
                self.function_param_defaults.contains_key(&func_id),
                self.function_param_optional.contains_key(&func_id),
            );
        }

        // Try HIR-level defaults first (available for freshly compiled functions)
        if let Some(defaults) = self.function_param_defaults.get(&func_id).cloned() {
            if user_arg_count >= defaults.len() {
                return; // All args provided
            }
            for i in user_arg_count..defaults.len() {
                if let Some(ref default_expr) = defaults[i] {
                    if let Some(reg) = self.lower_expression(default_expr) {
                        arg_regs.push(reg);
                    }
                }
            }
            return;
        }

        // BLADE-cached and cross-module functions have no HIR defaults: fill the missing
        // params with zero of the signature's type, looking in both local functions and
        // extern declarations.
        let sig_params: Vec<IrType> = self
            .builder
            .module
            .functions
            .get(&func_id)
            .map(|f| {
                f.signature
                    .parameters
                    .iter()
                    .map(|p| p.ty.clone())
                    .collect()
            })
            .or_else(|| {
                self.builder.module.extern_functions.get(&func_id).map(|f| {
                    f.signature
                        .parameters
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect()
                })
            })
            .or_else(|| self.external_function_param_types.get(&func_id).cloned())
            .unwrap_or_default();

        let sig_param_count = if sig_params.is_empty() {
            self.external_constructor_param_counts
                .get(&func_id)
                .copied()
                .unwrap_or(0)
        } else {
            sig_params.len()
        };

        let total_provided = arg_regs.len();
        if total_provided < sig_param_count {
            for i in total_provided..sig_param_count {
                let default_val = if i < sig_params.len() {
                    match &sig_params[i] {
                        IrType::I32 | IrType::U32 => IrValue::I32(0),
                        IrType::F32 => IrValue::F32(0.0),
                        IrType::F64 => IrValue::F64(0.0),
                        IrType::Bool => IrValue::Bool(false),
                        _ => IrValue::I64(0), // pointers, strings, etc.
                    }
                } else {
                    IrValue::I64(0)
                };
                if let Some(reg) = self.builder.build_const(default_val) {
                    arg_regs.push(reg);
                }
            }
        }
    }

    pub(crate) fn record_constrained_params(
        &mut self,
        func_id: IrFunctionId,
        hir_func: &HirFunction,
        has_this: bool,
    ) {
        let mut constrained = Vec::new();
        for (i, param) in hir_func.params.iter().enumerate() {
            let type_table = self.type_table;
            if let Some(type_info) = type_table.get(param.ty) {
                if let TypeKind::TypeParameter { constraints, .. } = &type_info.kind {
                    // Find the first interface constraint
                    for constraint_id in constraints {
                        if let Some(constraint_type) = type_table.get(*constraint_id) {
                            if let TypeKind::Interface { symbol_id, .. } = &constraint_type.kind {
                                // param_index in MIR includes 'this' offset
                                let mir_index = if has_this { i + 1 } else { i };
                                constrained.push((mir_index, *symbol_id));
                                break;
                            }
                        }
                    }
                }
            }
        }
        if !constrained.is_empty() {
            self.constrained_param_interfaces
                .insert(func_id, constrained);
        }
    }

    /// Strip a synthetic class "receiver" from static-call args when present.
    pub(crate) fn effective_static_call_args<'b>(&self, args: &'b [HirExpr]) -> &'b [HirExpr] {
        if args.is_empty() {
            return args;
        }
        if self.is_class_symbol_expr(&args[0]) {
            return &args[1..];
        }
        args
    }

    /// @:derive(Copy) — copy any Copy-typed variable arguments at call boundaries.
    /// Returns new arg_regs with copies substituted for Copy-type variables.
    pub(crate) fn maybe_copy_call_args(
        &mut self,
        args: &[HirExpr],
        arg_regs: Vec<IrId>,
    ) -> Vec<IrId> {
        if self.derive_copy_classes.is_empty() {
            return arg_regs;
        }
        args.iter()
            .zip(arg_regs)
            .map(|(arg_expr, reg)| {
                if let HirExprKind::Variable { .. } = &arg_expr.kind {
                    if let Some(class_sym) = self.get_copy_class_symbol(arg_expr.ty) {
                        if let Some(copy_ptr) = self.emit_shallow_copy(reg, class_sym) {
                            return copy_ptr;
                        }
                    }
                }
                reg
            })
            .collect()
    }

    /// Check if an expression produces a value backed by an anon view, and if so,
    /// materialize it into a real AnonObject handle. Used at escape points (call args).
    /// Also handles direct class→anon or wider-anon→anon conversion at call boundaries
    /// when the callee expects an anonymous-typed parameter.
    pub(crate) fn maybe_materialize_for_call(
        &mut self,
        arg_expr: &HirExpr,
        arg_reg: IrId,
        callee_func_id: Option<IrFunctionId>,
        param_index: usize,
    ) -> IrId {
        // Path 1: Variable with existing anon_views entry → materialize from backing
        if let HirExprKind::Variable { symbol, .. } = &arg_expr.kind {
            if self.anon_views.contains_key(symbol) {
                if let Some(materialized) =
                    self.materialize_anon_view(arg_reg, *symbol, arg_expr.ty)
                {
                    return materialized;
                }
            }
        }

        // Path 2: direct class→anon or wider-anon→anon conversion at the call boundary.
        if let Some(func_id) = callee_func_id {
            if let Some(param_types) = self.function_param_hir_types.get(&func_id).cloned() {
                if let Some(&param_type_id) = param_types.get(param_index) {
                    let resolved_param = self.resolve_through_aliases(param_type_id);
                    let resolved_arg = self.resolve_through_aliases(arg_expr.ty);

                    // A parameter typed `Iterable<T>`/`Iterator<T>` takes an
                    // iteration handle, built here while the argument's concrete
                    // type is still known. This precedes the anon handling below
                    // because both protocols are structural, and materializing one
                    // as a plain anonymous structure would drop the entry points.
                    if let Some(handle) =
                        self.maybe_wrap_for_iter_protocol(arg_reg, arg_expr.ty, param_type_id)
                    {
                        return handle;
                    }

                    // Concrete → Dynamic at the call boundary must box, as Let/Assign and
                    // the Cast→Dynamic path do: a raw primitive would be read back as a
                    // bogus Dynamic pointer.
                    let param_is_dynamic = {
                        let type_table = self.type_table;
                        // `Null<scalar>` is also a boxed DynamicValue* (see convert_type),
                        // so gating on Dynamic alone would let a raw scalar travel into a
                        // Ptr(U8) slot the callee then unboxes.
                        let param_is_optional_scalar =
                            match type_table.get(resolved_param).map(|t| &t.kind) {
                                Some(TypeKind::Optional { inner_type }) => type_table
                                    .get(*inner_type)
                                    .map(|t| {
                                        matches!(
                                            t.kind,
                                            TypeKind::Int | TypeKind::Float | TypeKind::Bool
                                        )
                                    })
                                    .unwrap_or(false),
                                _ => false,
                            };
                        matches!(
                            type_table.get(resolved_param).map(|t| &t.kind),
                            Some(TypeKind::Dynamic)
                        ) || param_is_optional_scalar
                    };
                    if param_is_dynamic {
                        if let Some(boxed) =
                            self.maybe_box_value(arg_reg, arg_expr.ty, param_type_id)
                        {
                            if boxed != arg_reg {
                                return boxed;
                            }
                        }
                    }

                    let param_is_anon = {
                        let type_table = self.type_table;
                        type_table
                            .get(resolved_param)
                            .map(|t| matches!(t.kind, TypeKind::Anonymous { .. }))
                            .unwrap_or(false)
                    };

                    if param_is_anon {
                        let arg_kind = {
                            let type_table = self.type_table;
                            type_table.get(resolved_arg).map(|t| t.kind.clone())
                        };

                        match arg_kind {
                            Some(TypeKind::Class { .. }) => {
                                // Class→anon: build temporary AnonBacking::Class and materialize
                                if let Some(handle) = self.materialize_class_to_anon(
                                    arg_reg,
                                    resolved_arg,
                                    resolved_param,
                                ) {
                                    return handle;
                                }
                            }
                            Some(TypeKind::Anonymous { .. }) => {
                                // Wider-anon→narrower-anon: materialize with index remapping
                                if resolved_arg != resolved_param {
                                    if let Some(handle) = self.materialize_wider_anon_to_anon(
                                        arg_reg,
                                        resolved_arg,
                                        resolved_param,
                                    ) {
                                        return handle;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // Path 3: Class → Interface at the call boundary. An interface-typed
                    // parameter needs a fat pointer, or dispatch on it reads garbage off
                    // the raw class pointer. `wrap_in_interface_fat_ptr` handles vtable
                    // presence itself, with a lazy-build fallback for (class, interface)
                    // pairs that eager registration missed.
                    if let Some(iface_sym) = self.get_interface_symbol(resolved_param) {
                        // Don't double-wrap an already-fat interface value.
                        if self.interface_wrapped_args.contains(&arg_reg) {
                            return arg_reg;
                        }
                        // Already a fat pointer of this same interface (the receiver of
                        // `iface.method()`, a forwarded interface-typed local): wrapping
                        // again would resolve a class through the drift-prone
                        // type→SymbolId path and overwrite the real vtable with an empty
                        // one, so pass it through.
                        if self.get_interface_symbol(resolved_arg) == Some(iface_sym) {
                            return arg_reg;
                        }
                        // A syntactic `new ClassName()` (optionally behind a Cast) carries
                        // an authoritative class name, so it resolves by FQN ONLY and never
                        // falls through to `get_class_symbol`, whose type→SymbolId lookup
                        // drifts across compile contexts. Resolvable here → normal vtable
                        // wrap; not materialized in this context → by-name wrap with
                        // forward-ref thunks that dedupe at merge.
                        if let Some(class_fqn) = self.new_arg_class_fqn(arg_expr) {
                            if let Some(class_sym) = self.lookup_class_symbol_by_name(&class_fqn) {
                                if let Some(wrapped) =
                                    self.wrap_in_interface_fat_ptr(arg_reg, class_sym, iface_sym)
                                {
                                    self.interface_wrapped_args.insert(wrapped);
                                    return wrapped;
                                }
                            }
                            if let Some(wrapped) = self
                                .wrap_new_class_as_interface_by_name(arg_reg, &class_fqn, iface_sym)
                            {
                                self.interface_wrapped_args.insert(wrapped);
                                return wrapped;
                            }
                        }
                        // Not a `new` expr: the static class of the variable/field/call is
                        // the trustworthy handle, recovered from the register's class hint
                        // when the typechecker promoted the arg to the interface.
                        let class_sym = self
                            .get_class_symbol(resolved_arg)
                            .or_else(|| self.recover_arg_concrete_class(arg_expr, arg_reg));
                        if let Some(class_sym) = class_sym {
                            if let Some(wrapped) =
                                self.wrap_in_interface_fat_ptr(arg_reg, class_sym, iface_sym)
                            {
                                // The fat pointer may escape through the callee (pushed
                                // into a long-lived Array<I>), so marking it escaped keeps
                                // callers from pushing it to `temp_heap_values` and freeing
                                // it on return.
                                self.interface_wrapped_args.insert(wrapped);
                                return wrapped;
                            }
                        }
                    }
                }
            }
        }

        // Path 3 fallback for imported callees, whose HIR never went through this
        // context's lowering so their param types are absent from
        // `function_param_hir_types`: resolve the interface symbol from the per-param
        // qualified names in `external_function_param_iface_names` and run the same wrap.
        if let Some(func_id) = callee_func_id {
            if !self.function_param_hir_types.contains_key(&func_id) {
                if let Some(names) = self
                    .external_function_param_iface_names
                    .get(&func_id)
                    .cloned()
                {
                    if let Some(Some(param_name)) = names.get(param_index) {
                        // The name is all this context holds of an imported
                        // callee's signature, and it can only mean one thing:
                        // an interface's name, or a protocol's. Resolving the
                        // interface first settles which, so a program that
                        // declares its own `Iterator`/`Iterable` interface keeps
                        // the fat pointer its callee dispatches through.
                        let iface_sym = self.lookup_interface_symbol_by_qualified_name(param_name);
                        // A parameter the callee declares as `Iterable<T>`/
                        // `Iterator<T>` takes an iteration handle, built here
                        // while the argument's concrete type is still known.
                        // Both protocols are structural, so a value that reaches
                        // the callee as a plain pointer arrives with no entry
                        // points at all.
                        if iface_sym.is_none() {
                            if let Some(handle) = self.maybe_wrap_for_named_iter_protocol(
                                arg_reg,
                                arg_expr.ty,
                                param_name,
                            ) {
                                return handle;
                            }
                        }
                        if let Some(iface_sym) = iface_sym {
                            // Don't double-wrap a value that is already a fat pointer: the
                            // nested wrapper would become the callee's `this`.
                            if self.interface_wrapped_args.contains(&arg_reg) {
                                return arg_reg;
                            }
                            // Already an interface value of this same interface (see the
                            // primary path): re-wrapping would build a bogus-class fat ptr.
                            if self.get_interface_symbol(arg_expr.ty) == Some(iface_sym) {
                                return arg_reg;
                            }
                            // As in the primary path, a syntactic `new ClassName()`
                            // resolves by FQN only: resolvable here → normal vtable wrap;
                            // otherwise by-name wrap with forward-ref thunks.
                            if let Some(class_fqn) = self.new_arg_class_fqn(arg_expr) {
                                if let Some(class_sym) =
                                    self.lookup_class_symbol_by_name(&class_fqn)
                                {
                                    self.interface_vtables.remove(&(class_sym, iface_sym));
                                    if let Some(wrapped) = self
                                        .wrap_in_interface_fat_ptr(arg_reg, class_sym, iface_sym)
                                    {
                                        self.interface_wrapped_args.insert(wrapped);
                                        return wrapped;
                                    }
                                }
                                if let Some(wrapped) = self.wrap_new_class_as_interface_by_name(
                                    arg_reg, &class_fqn, iface_sym,
                                ) {
                                    self.interface_wrapped_args.insert(wrapped);
                                    return wrapped;
                                }
                            }
                            // Not a `new` expr: the typechecker often promotes the arg to
                            // the interface, erasing the class from `arg_expr.ty` while the
                            // register still holds a raw class object, recoverable from the
                            // class hint. Any pre-cached vtable is dropped first — its
                            // method SymbolIds may come from the class's originating
                            // context, so the lazy name-based rebuild must resolve against
                            // the current symbol table.
                            let class_sym = self
                                .get_class_symbol(arg_expr.ty)
                                .or_else(|| self.recover_arg_concrete_class(arg_expr, arg_reg));
                            if let Some(class_sym) = class_sym {
                                self.interface_vtables.remove(&(class_sym, iface_sym));
                                if let Some(wrapped) =
                                    self.wrap_in_interface_fat_ptr(arg_reg, class_sym, iface_sym)
                                {
                                    self.interface_wrapped_args.insert(wrapped);
                                    return wrapped;
                                }
                            }
                        }
                    }
                }
            }
        }

        arg_reg
    }
}
