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

        // Add type parameters from the HIR function (for generic functions)
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

        // Add class type parameters first (T, U, etc from the generic class)
        for type_param in class_type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            type_param_names.push(param_name.clone());
            builder = builder.type_param(param_name);
        }

        // Add method's own type parameters (if any)
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

        // Add type parameters from the HIR function (for generic methods)
        for type_param in &func.type_params {
            let param_name = self
                .string_interner
                .get(type_param.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("T{}", type_param.name.as_raw()));
            builder = builder.type_param(param_name);
        }

        // Add implicit 'this' parameter as first parameter
        // 'this' is always a pointer to the class instance, regardless of generic parameters
        let this_type = match self.convert_type(class_type_id) {
            IrType::Ptr(_) => IrType::Ptr(Box::new(IrType::Void)),
            // If convert_type failed to resolve (e.g., generic class without instantiation),
            // default to pointer since 'this' is always a pointer to instance
            _ => IrType::Ptr(Box::new(IrType::Void)),
        };

        builder = builder.param("this".to_string(), this_type);

        // Add regular parameters
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

        // Fallback for BLADE-cached or cross-module functions: HIR defaults aren't available.
        // Fill missing params with null/zero using the correct type from the signature.
        // Check both local functions and extern function declarations (cross-module calls).
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
                // Use the correct type from the function signature
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

        // Path 2: Direct class→anon or wider-anon→anon conversion at call boundary
        // Check if callee parameter expects anonymous type but arg provides class/wider-anon
        if let Some(func_id) = callee_func_id {
            if let Some(param_types) = self.function_param_hir_types.get(&func_id).cloned() {
                if let Some(&param_type_id) = param_types.get(param_index) {
                    let resolved_param = self.resolve_through_aliases(param_type_id);
                    let resolved_arg = self.resolve_through_aliases(arg_expr.ty);

                    // Concrete → Dynamic at the call boundary: box the value so the
                    // callee receives a real DynamicValue. Without this a primitive
                    // arg is passed as raw bits and read back as a bogus Dynamic
                    // pointer (Std.string garbles / crashes). Mirrors Let/Assign
                    // boxing and the Cast→Dynamic path.
                    let param_is_dynamic = {
                        let type_table = self.type_table;
                        // `Null<scalar>` is represented as a boxed DynamicValue*
                        // (see convert_type), so an argument for such a
                        // parameter must be boxed here too. Gating only on
                        // Dynamic let a raw scalar travel into a Ptr(U8) slot:
                        // the callee then unboxed it and dereferenced the value
                        // itself (`nc(9)` dereferenced address 9), and at -O the
                        // register typed Ptr(U8) actually held a float, which
                        // crashed codegen comparing it against null.
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

                    // Path 3: Class → Interface at call boundary. When the callee
                    // expects an interface-typed parameter and the arg is a class
                    // implementing that interface, wrap the class pointer in a fat
                    // pointer so the callee can do virtual dispatch via the vtable.
                    // Without this, methods called on the parameter would resolve
                    // through a raw class pointer and read garbage.
                    //
                    // We let `wrap_in_interface_fat_ptr` handle vtable presence
                    // itself — it has a lazy-build fallback against
                    // `interface_method_names` + `class_method_symbols` for
                    // cases where eager registration in `register_class_metadata`
                    // missed the (class, interface) pair (e.g. the interface
                    // was declared in a later-loaded file).
                    if let Some(iface_sym) = self.get_interface_symbol(resolved_param) {
                        // Don't double-wrap an already-fat interface value.
                        if self.interface_wrapped_args.contains(&arg_reg) {
                            return arg_reg;
                        }
                        // The arg is ALREADY an interface value (fat pointer) of
                        // this same interface — e.g. materializing the receiver
                        // of `iface.method()`, or forwarding an interface-typed
                        // local. A class→interface wrap here would resolve a
                        // concrete class via the drift-prone type→SymbolId path
                        // (which can land on an unrelated class) and overwrite
                        // the real vtable with an empty one — the next dispatch
                        // faults. Pass it through.
                        if self.get_interface_symbol(resolved_arg) == Some(iface_sym) {
                            return arg_reg;
                        }
                        // A syntactic `new ClassName()` (optionally behind a
                        // Cast) carries an AUTHORITATIVE class name. The type→
                        // SymbolId resolution (`get_class_symbol`) drifts across
                        // compile contexts and can land on an unrelated class, so
                        // for a `new` arg we resolve ONLY by FQN and never fall
                        // through to it:
                        //  - resolvable here → normal vtable wrap;
                        //  - not materialized in THIS context → by-NAME wrap
                        //    with forward-ref dispatch thunks that dedupe with
                        //    the real ones at merge (order-independent).
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
                        // Not a `new` expr: the arg is a variable/field/call
                        // whose static class is the trustworthy handle. Recover
                        // the concrete class from the value register's class
                        // hint when the typechecker promoted it to the interface.
                        let class_sym = self
                            .get_class_symbol(resolved_arg)
                            .or_else(|| self.recover_arg_concrete_class(arg_expr, arg_reg));
                        if let Some(class_sym) = class_sym {
                            if let Some(wrapped) =
                                self.wrap_in_interface_fat_ptr(arg_reg, class_sym, iface_sym)
                            {
                                // The wrapped fat pointer may escape via the
                                // callee (e.g. push into a long-lived Array<I>),
                                // so we cannot auto-free it on return. Mark it
                                // as escaped so callers skip the
                                // `temp_heap_values` push that would emit Free.
                                self.interface_wrapped_args.insert(wrapped);
                                return wrapped;
                            }
                        }
                    }
                }
            }
        }

        // Path 3 fallback for imported callees: when the callee's HIR
        // param types aren't in `function_param_hir_types` (cache-loaded
        // or fresh-imported class — its HIR never went through *this*
        // context's lowering), fall back to per-param qualified names
        // captured in `external_function_param_iface_names`. Find the
        // current-context Interface symbol by name and run the same
        // wrap. Without this, every cross-file constructor that takes
        // an interface-typed param stores the raw class pointer and
        // SIGBUSes on first vtable dispatch.
        if let Some(func_id) = callee_func_id {
            if !self.function_param_hir_types.contains_key(&func_id) {
                if let Some(names) = self
                    .external_function_param_iface_names
                    .get(&func_id)
                    .cloned()
                {
                    if let Some(Some(param_name)) = names.get(param_index) {
                        if let Some(iface_sym) =
                            self.lookup_interface_symbol_by_qualified_name(param_name)
                        {
                            // Already a fat pointer (e.g. produced by a
                            // Class→Interface cast upstream)? Don't double-wrap
                            // — that would nest fat pointers and make `this` the
                            // inner wrapper.
                            if self.interface_wrapped_args.contains(&arg_reg) {
                                return arg_reg;
                            }
                            // Already an interface value of this same interface
                            // (see the primary path): pass it through rather than
                            // re-wrapping it into a bogus-class fat pointer.
                            if self.get_interface_symbol(arg_expr.ty) == Some(iface_sym) {
                                return arg_reg;
                            }
                            // A syntactic `new ClassName()` carries an
                            // AUTHORITATIVE class name; the type→SymbolId
                            // resolution drifts across compile contexts and can
                            // land on an unrelated class, so for a `new` arg we
                            // resolve ONLY by FQN and never fall through to it:
                            //  - resolvable here → normal vtable wrap;
                            //  - not materialized in THIS context → by-NAME wrap
                            //    with forward-ref dispatch thunks that dedupe at
                            //    merge (order-independent).
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
                            // Not a `new` expr: the typechecker often PROMOTES
                            // the arg to the interface, erasing the class from
                            // `arg_expr.ty`; the value register still holds a RAW
                            // class object, recoverable from the class hint. Drop
                            // any pre-cached vtable first — its method SymbolIds
                            // may be from the class's originating context and
                            // resolve to unrelated methods here; forcing the
                            // lazy name-based rebuild resolves against the
                            // current symbol table.
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
