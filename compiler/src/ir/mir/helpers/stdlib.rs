//! Stdlib class detection, and dispatch of stdlib properties and calls.

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
    /// Check if a symbol refers to a known stdlib class by trying native name then simple name
    pub(crate) fn is_stdlib_class_by_symbol(&self, symbol: &crate::tast::symbols::Symbol) -> bool {
        self.canonical_stdlib_class_name(symbol).is_some()
    }

    /// Resolve a stdlib class symbol to its registered runtime-mapping key
    /// (e.g., `Tensor` → the key rayzor-tensors registered it under, `Arc` →
    /// `rayzor.concurrent.Arc`).
    ///
    /// Returns `None` if the symbol isn't a registered stdlib class. The
    /// `@:native("a::b::C")` annotation is the strongest name and is tried
    /// first; the bare simple name covers extern classes without one. Either
    /// spelling resolves through the mapping's alias index, so what comes
    /// back is always the key the class is actually registered under.
    pub(crate) fn canonical_stdlib_class_name(
        &self,
        symbol: &crate::tast::symbols::Symbol,
    ) -> Option<crate::stdlib::ClassKey> {
        if let Some(native) = symbol.native_name {
            if let Some(native_str) = self.string_interner.get(native) {
                let dotted = native_str.replace("::", ".");
                if let Some(key) = self.stdlib_mapping.class_key(&dotted) {
                    return Some(key);
                }
            }
        }
        let class_name = self.string_interner.get(symbol.name)?;
        self.stdlib_mapping.class_key(class_name)
    }

    /// The stdlib class a call returns, read from the callee's declaration.
    ///
    /// `Arc.init` is declared `init<T>(v:T):Arc<T>`: the method symbol's
    /// function type names the returned class even when the generic argument
    /// is erased, so the answer comes from the signature the typechecker
    /// already resolved — never from a list of method names, and never from
    /// searching the registered classes for one that has a method of that
    /// name. `None` means the declaration does not name a stdlib class,
    /// which is not license to guess one.
    pub(crate) fn detect_stdlib_class_from_call(
        &self,
        callee: &HirExpr,
        _args: &[HirExpr],
    ) -> Option<String> {
        let method_symbol = match &callee.kind {
            // `Arc.init(...)` / `arc.clone()` — the field is the resolved
            // method symbol.
            HirExprKind::Field { field, .. } => *field,
            // `init(data)` / `clone(shared)` — the callee resolved straight
            // to a method symbol.
            HirExprKind::Variable { symbol, .. } => *symbol,
            _ => return None,
        };
        self.stdlib_class_of_declared_return(method_symbol)
    }

    /// The registered stdlib class named by a method symbol's declared
    /// return type, resolved through typedefs (`haxe.io.Bytes` →
    /// `rayzor.Bytes`).
    fn stdlib_class_of_declared_return(&self, method_symbol: SymbolId) -> Option<String> {
        let sym = self.symbol_table.get_symbol(method_symbol)?;
        let returned_class_symbol = {
            let type_table = self.type_table;
            let TypeKind::Function { return_type, .. } = &type_table.get(sym.type_id)?.kind else {
                return None;
            };
            let returned = type_table.get(self.resolve_through_aliases(*return_type))?;
            let TypeKind::Class { symbol_id, .. } = &returned.kind else {
                return None;
            };
            *symbol_id
        };
        let class_sym = self.symbol_table.get_symbol(returned_class_symbol)?;
        let key = self.canonical_stdlib_class_name(class_sym)?;
        debug!(
            "[STDLIB CLASS DETECT] declared return of {:?} is {}",
            self.string_interner.get(sym.name),
            key
        );
        Some(key.as_str().to_string())
    }

    pub(crate) fn try_stdlib_from_cast(
        &mut self,
        expr: &HirExpr,
        target: TypeId,
        abs_name: &InternedString,
    ) -> Option<IrId> {
        // Resolve abstract's stdlib class name (e.g., "rayzor.SIMD4f")
        let class_name = {
            let type_table = self.type_table;
            let ti = type_table.get(target)?;
            if let TypeKind::Abstract { symbol_id, .. } = &ti.kind {
                let sym = self.symbol_table.get_symbol(*symbol_id)?;
                sym.native_name
                    .and_then(|nn| self.string_interner.get(nn))
                    .map(|n| n.replace("::", "."))
                    .or_else(|| {
                        self.string_interner
                            .get(sym.name)
                            .map(|n| format!("rayzor.{}", n))
                    })
            } else {
                None
            }
        }?;

        // Search stdlib mapping for a static @:from method (e.g., "fromArray")
        let class_key = self.stdlib_mapping.class_key(&class_name)?;
        let mapping_info = self
            .stdlib_mapping
            .find_by_name(class_key, "fromArray")
            .map(|(_, m)| (m.runtime_name.to_string(), m.is_mir_wrapper));

        let (runtime_func, is_mir_wrapper) = mapping_info?;

        let value_reg = self.lower_expression(expr)?;
        let result_type = self.convert_type(target);
        let param_types = vec![IrType::Ptr(Box::new(IrType::Void))];

        let func_id = if is_mir_wrapper {
            self.register_stdlib_mir_forward_ref(&runtime_func, param_types, result_type.clone())
        } else {
            self.get_or_register_extern_function(&runtime_func, param_types, result_type.clone())
        };

        self.builder
            .build_call_direct(func_id, vec![value_reg], result_type)
    }

    /// Resolve a property/field on a receiver whose type stayed unresolved
    /// cross-module, using a surviving class hint (register hint from a factory
    /// result, or the object variable's tracked stdlib class). Maps the hint to
    /// the stdlib class and emits the mapped runtime getter directly. Returns
    /// None when no hint is available or the class exposes no such property.
    pub(crate) fn try_stdlib_property_by_hint(
        &mut self,
        obj_reg: IrId,
        object_kind: &HirExprKind,
        field: SymbolId,
        field_ty: TypeId,
    ) -> Option<IrId> {
        // Receiver class hint: register-level first, then the object variable.
        let hint = self
            .register_class_hints
            .get(&obj_reg)
            .cloned()
            .or_else(|| {
                if let HirExprKind::Variable { symbol, .. } = object_kind {
                    self.monomorphized_var_types.get(symbol).cloned()
                } else {
                    None
                }
            })?;
        let field_name = self
            .symbol_table
            .get_symbol(field)
            .and_then(|s| self.string_interner.get(s.name))?
            .to_string();

        // The hint names the receiver's class; the mapping resolves whatever
        // spelling it arrived in.
        let hint_key = self.stdlib_mapping.class_key(&hint)?;
        let runtime_name = self
            .stdlib_mapping
            .find_by_name(hint_key, &field_name)
            .map(|(_, call)| call.runtime_name)?;

        let field_kind = self.type_table.get(field_ty).map(|t| t.kind.clone());
        let result_type = match field_kind {
            Some(crate::tast::TypeKind::Int) => IrType::I32,
            Some(crate::tast::TypeKind::Float) => IrType::F64,
            Some(crate::tast::TypeKind::Bool) => IrType::Bool,
            _ => IrType::Ptr(Box::new(IrType::Void)),
        };
        let func_id = self.get_or_register_extern_function(
            runtime_name,
            vec![IrType::Ptr(Box::new(IrType::Void))],
            result_type.clone(),
        );
        self.builder
            .build_call_direct(func_id, vec![obj_reg], result_type)
    }

    /// Dispatch `tls.value` / `tls.value = v` on extern classes whose property
    /// accessor (`get_value` / `set_value`) is declared with `@:native(...)` and
    /// so lives in the stdlib mapping, not the user's `function_map`.
    ///
    /// Looks up `(receiver_class_name, accessor_method_name)` in `stdlib_mapping`
    /// and emits a direct call to the runtime function. Returns `None` if no
    /// mapping is found, letting the caller fall through to other paths.
    pub(crate) fn try_property_call_via_stdlib(
        &mut self,
        receiver_ty: TypeId,
        accessor_name: InternedString,
        args: Vec<IrId>,
        return_type: IrType,
    ) -> Option<IrId> {
        let class_symbol = self.resolve_receiver_class_symbol(receiver_ty)?;
        let class_name = self.symbol_table.get_symbol(class_symbol).and_then(|s| {
            s.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(s.name))
        })?;
        let method_name = self.string_interner.get(accessor_name)?;

        // The symbol stores the dotted qualified name ("sys.thread.Tls"),
        // which is the mapping's key form.
        let dot_form = class_name.to_string();

        // expected_param_count = args.len() - 1 because args includes `this`
        let expected_extra = args.len().saturating_sub(1);

        let cn = self.stdlib_mapping.class_key(&dot_form)?;
        let (runtime_name, ret_ty, param_types, is_mir_wrapper) = self
            .stdlib_mapping
            .find_by_name_and_params(cn, method_name, expected_extra)
            .or_else(|| self.stdlib_mapping.find_by_name(cn, method_name))
            .map(|(_sig, mapping)| {
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                (
                    mapping.runtime_name.to_string(),
                    return_type.clone(),
                    vec![ptr_u8; args.len()],
                    mapping.is_mir_wrapper,
                )
            })?;
        // MIR-wrapper names (e.g. `array_length`) must go through forward-ref to share one
        // FuncId with the wrapper's body; registering them as an extern declares an Import
        // the backend can then never define.
        let func_id = if is_mir_wrapper {
            self.register_stdlib_mir_forward_ref(&runtime_name, param_types, ret_ty.clone())
        } else {
            self.get_or_register_extern_function(&runtime_name, param_types, ret_ty.clone())
        };
        let r = self
            .builder
            .build_call_direct(func_id, args, ret_ty.clone());
        // A void call has no result register, but callers only test `is_some()` to see
        // whether dispatch happened, so a successful void call returns a sentinel IrId(0).
        if r.is_some() {
            r
        } else if matches!(ret_ty, IrType::Void) {
            Some(crate::ir::IrId(0))
        } else {
            None
        }
    }

    /// If `type_id` resolves to a Class with a `toString()` in the current module,
    /// call `obj.toString()` and return the resulting `*HaxeString` register.
    /// Returns `Some(string_reg)` on success, `None` if not a class or toString not found.
    /// Std.string's tag for a value type: 1=Int 2=Bool 4=Float 5=String
    /// 6=Ref, 0 when nothing is known. Same table `Array.join` uses, so a map
    /// and an array render a value the same way.
    fn std_string_tag(&self, ty: TypeId) -> i32 {
        let resolved = self.resolve_through_aliases(ty);
        match self.type_table.get(resolved).map(|t| &t.kind) {
            Some(TypeKind::Int) => 1,
            Some(TypeKind::Bool) => 2,
            Some(TypeKind::Float) => 4,
            Some(TypeKind::String) => 5,
            Some(TypeKind::Optional { inner_type }) => {
                let t = *inner_type;
                self.std_string_tag(t)
            }
            Some(
                TypeKind::Class { .. }
                | TypeKind::Interface { .. }
                | TypeKind::Anonymous { .. }
                | TypeKind::Array { .. }
                | TypeKind::Enum { .. }
                | TypeKind::Map { .. },
            ) => 6,
            _ => 0,
        }
    }

    /// A map's `toString` through the renderer that knows what its keys and
    /// values are. `None` when the receiver is not a map container, which
    /// leaves every other class on its usual path.
    pub(crate) fn try_lower_map_to_string(
        &mut self,
        obj_reg: IrId,
        receiver_ty: TypeId,
    ) -> Option<IrId> {
        let resolved = self.resolve_through_aliases(receiver_ty);
        let (class_sym, type_args) = match self.type_table.get(resolved).map(|t| &t.kind) {
            Some(TypeKind::Class {
                symbol_id,
                type_args,
            })
            | Some(TypeKind::Interface {
                symbol_id,
                type_args,
            }) => (*symbol_id, type_args.clone()),
            _ => return None,
        };
        let name = self
            .symbol_table
            .get_symbol(class_sym)
            .and_then(|s| s.qualified_name.or(Some(s.name)))
            .and_then(|n| self.string_interner.get(n))?
            .to_string();
        let bare = name.rsplit('.').next().unwrap_or(&name);
        let value_tag = self.std_string_tag(*type_args.last()?);
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
        let string_ptr = IrType::Ptr(Box::new(IrType::String));
        let (runtime, mut params, mut call_args) = match bare {
            "StringMap" => (
                "haxe_stringmap_to_string_typed",
                vec![ptr_void.clone(), IrType::I32],
                vec![obj_reg],
            ),
            "IntMap" => (
                "haxe_intmap_to_string_typed",
                vec![ptr_void.clone(), IrType::I32],
                vec![obj_reg],
            ),
            "ObjectMap" | "EnumValueMap" | "WeakMap" => (
                "haxe_objectmap_to_string_typed",
                vec![ptr_void.clone(), IrType::I32, IrType::I32],
                vec![obj_reg],
            ),
            _ => return None,
        };
        if params.len() == 3 {
            let key_tag = self.std_string_tag(*type_args.first()?);
            call_args.push(self.builder.build_const(IrValue::I32(key_tag))?);
        }
        call_args.push(self.builder.build_const(IrValue::I32(value_tag))?);
        let ret = params.pop();
        let _ = ret;
        let func = self.get_or_register_extern_function(
            runtime,
            match bare {
                "StringMap" | "IntMap" => vec![ptr_void.clone(), IrType::I32],
                _ => vec![ptr_void.clone(), IrType::I32, IrType::I32],
            },
            string_ptr.clone(),
        );
        self.builder.build_call_direct(func, call_args, string_ptr)
    }

    /// The `toString` a class or abstract declares itself.
    ///
    /// An abstract keeps its methods on its own HIR declaration, so a scan
    /// that only walks classes misses `Foo.toString` and leaves the value to
    /// be printed by its underlying representation.
    pub(crate) fn user_tostring_symbol(&self, type_id: TypeId) -> Option<SymbolId> {
        let owner = self.type_table.get(type_id).and_then(|ti| match &ti.kind {
            TypeKind::Class { symbol_id, .. } | TypeKind::Abstract { symbol_id, .. } => {
                Some(*symbol_id)
            }
            _ => None,
        })?;
        for (_tid, decl) in self.current_hir_types.iter() {
            let (sym, methods) = match decl {
                HirTypeDecl::Class(c) => (c.symbol_id, &c.methods),
                HirTypeDecl::Abstract(a) => (a.symbol_id, &a.methods),
                _ => continue,
            };
            if sym != owner {
                continue;
            }
            return methods
                .iter()
                .find(|m| {
                    !m.is_static && self.string_interner.get(m.function.name) == Some("toString")
                })
                .map(|m| m.function.symbol_id);
        }
        None
    }

    /// An abstract's own `toString`, called with the value as its `this`.
    pub(crate) fn try_call_abstract_tostring(
        &mut self,
        value: IrId,
        type_id: TypeId,
    ) -> Option<IrId> {
        let abstract_symbol = match self
            .type_table
            .get(self.resolve_through_aliases(type_id))
            .map(|t| &t.kind)
        {
            Some(TypeKind::Abstract { symbol_id, .. }) => *symbol_id,
            _ => return None,
        };
        let method = self
            .current_hir_types
            .values()
            .find_map(|decl| match decl {
                HirTypeDecl::Abstract(a) if a.symbol_id == abstract_symbol => a
                    .methods
                    .iter()
                    .find(|m| {
                        !m.is_static
                            && self.string_interner.get(m.function.name) == Some("toString")
                    })
                    .map(|m| m.function.symbol_id),
                _ => None,
            })?;
        let function = self.function_map.get(&method).copied()?;
        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
        self.builder
            .build_call_direct(function, vec![value], string_ptr_ty)
    }

    pub(crate) fn try_call_tostring(
        &mut self,
        obj_reg: IrId,
        type_id: TypeId,
    ) -> Option<Option<IrId>> {
        let class_symbol = {
            let type_table = self.type_table;
            type_table.get(type_id).and_then(|ti| {
                if let TypeKind::Class { symbol_id, .. } = &ti.kind {
                    Some(*symbol_id)
                } else {
                    None
                }
            })
        };

        let class_symbol = match class_symbol {
            Some(s) => s,
            None => return Some(self.try_call_abstract_tostring(obj_reg, type_id)),
        };

        // toString must be the one declared on THIS class, so scan the HIR declarations.
        let mut tostring_symbol = None;
        for (_tid, type_decl) in self.current_hir_types.iter() {
            if let HirTypeDecl::Class(class) = type_decl {
                if class.symbol_id == class_symbol {
                    for method in &class.methods {
                        let method_name =
                            self.string_interner.get(method.function.name).unwrap_or("");
                        if method_name == "toString" && !method.is_static {
                            tostring_symbol = Some(method.function.symbol_id);
                            break;
                        }
                    }
                    break;
                }
            }
        }

        if let Some(tostring_symbol) = tostring_symbol {
            // User class with toString() in HIR — look up compiled function
            if let Some(tostring_id) = self.function_map.get(&tostring_symbol).copied() {
                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                let string_reg =
                    self.builder
                        .build_call_direct(tostring_id, vec![obj_reg], string_ptr_ty)?;
                return Some(Some(string_reg));
            }
        }

        // A map renders its VALUES, and only the call site knows what they are:
        // the container stores every value in a u64 slot.
        if let Some(rendered) = self.try_lower_map_to_string(obj_reg, type_id) {
            return Some(Some(rendered));
        }
        // Fallback for stdlib classes (StringMap, IntMap, Date, Bytes): a toString()
        // registered in stdlib_mapping, looked up under the class's key.
        let class_name = self
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| self.canonical_stdlib_class_name(s));

        if let Some((_sig, mapping)) =
            class_name.and_then(|k| self.stdlib_mapping.find_by_name(k, "toString"))
        {
            let runtime_name = mapping.runtime_name;
            let is_mir_wrapper = mapping.is_mir_wrapper;
            let ptr_void = IrType::Ptr(Box::new(IrType::Void));

            let func_id = if is_mir_wrapper {
                self.register_stdlib_mir_forward_ref(runtime_name, vec![ptr_void.clone()], ptr_void)
            } else {
                self.get_or_register_extern_function(runtime_name, vec![ptr_void.clone()], ptr_void)
            };

            let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
            if let Some(str_ptr) = self.builder.build_call_direct(
                func_id,
                vec![obj_reg],
                IrType::Ptr(Box::new(IrType::Void)),
            ) {
                let string_reg = self
                    .builder
                    .build_cast(str_ptr, IrType::Ptr(Box::new(IrType::Void)), string_ptr_ty)
                    .unwrap_or(str_ptr);
                return Some(Some(string_reg));
            }
        }

        Some(None) // No toString() found
    }

    /// Try to apply an abstract @:from conversion in a Cast expression.
    /// Returns Some(result_reg) if a matching @:from rule was found, None otherwise.
    pub(crate) fn try_abstract_from_cast(
        &mut self,
        expr: &HirExpr,
        target: TypeId,
    ) -> Option<IrId> {
        let source_type = expr.ty;
        let abs_name = self.resolve_abstract_name(target)?;

        let matching_rule = self.abstract_from_rule(abs_name, source_type);

        if let Some(rule) = matching_rule {
            if let Some(cast_func_sym) = rule.cast_function {
                let value_reg = self.lower_expression(expr)?;
                let value_reg = self.from_rule_argument(value_reg, source_type, &rule);
                let func_id = self.resolve_abstract_conversion_function(cast_func_sym, target)?;
                let result_type = self.convert_type(target);
                self.builder
                    .build_call_direct(func_id, vec![value_reg], result_type)
            } else {
                // No conversion function — keyword from (identity conversion)
                let value_reg = self.lower_expression(expr)?;
                let from_type = self.convert_type(source_type);
                let to_type = self.convert_type(target);
                if from_type == to_type {
                    Some(value_reg)
                } else {
                    self.builder.build_cast(value_reg, from_type, to_type)
                }
            }
        } else if let Some(mid) = self.chained_from_target(target, source_type) {
            let value = self.lower_expression(expr)?;
            self.maybe_abstract_from_convert(value, source_type, mid)
        } else {
            // Fallback: for extern/imported abstracts (e.g., SIMD4f) whose @:from rules
            // weren't populated (not in file.abstracts), try stdlib mapping directly.
            self.try_stdlib_from_cast(expr, target, &abs_name)
        }
    }

    /// An abstract's symbol and type arguments, through aliases and instances.
    fn abstract_and_args(&self, ty: TypeId) -> Option<(SymbolId, Vec<TypeId>)> {
        let ty = self.resolve_through_aliases(ty);
        match &self.type_table.get(ty)?.kind {
            TypeKind::Abstract {
                symbol_id,
                type_args,
                ..
            } => Some((*symbol_id, type_args.clone())),
            TypeKind::GenericInstance {
                base_type,
                type_args,
                ..
            } => match &self
                .type_table
                .get(self.resolve_through_aliases(*base_type))?
                .kind
            {
                TypeKind::Abstract { symbol_id, .. } => Some((*symbol_id, type_args.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    /// `M<A>` with `from Null<T>` takes a value A converts from: the
    /// instantiated `T` when it is another abstract with an `@:from` for
    /// `source` (one step, as Haxe allows).
    fn chained_from_target(&self, target: TypeId, source: TypeId) -> Option<TypeId> {
        let abs_name = self.abstract_name_of(target)?;
        let (symbol, args) = self.abstract_and_args(target)?;
        self.abstract_from_rules
            .get(&abs_name)?
            .iter()
            .filter(|r| r.cast_function.is_none())
            .find_map(|r| {
                let inner = match self.type_table.get(r.from_type).map(|t| &t.kind) {
                    Some(TypeKind::Optional { inner_type }) => *inner_type,
                    _ => r.from_type,
                };
                let mid = self.substitute_abstract_type_arg(symbol, inner, &args)?;
                let mid_name = self.abstract_name_of(mid)?;
                (mid_name != abs_name && self.abstract_from_rule(mid_name, source).is_some())
                    .then_some(mid)
            })
    }

    /// `M2<A2>` with `to T` converts to what A2 converts to: the instantiated
    /// `T` when it has an `@:to` for `target` (one step).
    fn chained_to_source(&self, source: TypeId, target: TypeId) -> Option<TypeId> {
        let abs_name = self.abstract_name_of(source)?;
        let (symbol, args) = self.abstract_and_args(source)?;
        self.abstract_to_rules
            .get(&abs_name)?
            .iter()
            .filter(|r| r.cast_function.is_none())
            .find_map(|r| {
                let mid = self.substitute_abstract_type_arg(symbol, r.to_type, &args)?;
                (self.abstract_name_of(mid) != Some(abs_name)
                    && self.has_direct_abstract_to_function(mid, target))
                .then_some(mid)
            })
    }

    /// The abstract's `@:from` rule for `source_type`: one declared for that
    /// type, else a constrained generic conversion (`fromT<T:C>(t:T)`) whose
    /// constraints the source meets. A value already of the abstract is not
    /// converted again.
    fn abstract_from_rule(
        &self,
        abs_name: InternedString,
        source_type: TypeId,
    ) -> Option<HirCastRule> {
        let rules = self.abstract_from_rules.get(&abs_name)?;
        if let Some(rule) = rules.iter().find(|r| r.from_type == source_type) {
            return Some(rule.clone());
        }
        if self.abstract_name_of(source_type) == Some(abs_name) {
            return None;
        }
        // `from(i:Iterable<T>)` takes anything that supplies `iterator()`.
        if self.iter_protocol_of(source_type).is_none()
            && matches!(
                self.iter_source_of(source_type),
                Some(
                    super::iter_handle::IterSource::Array
                        | super::iter_handle::IterSource::MapValues { .. }
                        | super::iter_handle::IterSource::ClassIterable { .. }
                        | super::iter_handle::IterSource::AnonIterable { .. }
                )
            )
            && let Some(rule) = rules.iter().find(|r| {
                r.cast_function.is_some()
                    && self.iter_protocol_of(r.from_type)
                        == Some(super::iter_handle::IterProtocol::Iterable)
            })
        {
            return Some(rule.clone());
        }
        rules
            .iter()
            .find(|r| {
                r.cast_function.is_some()
                    && match self.type_table.get(r.from_type).map(|t| &t.kind) {
                        Some(TypeKind::TypeParameter { constraints, .. }) => {
                            !constraints.is_empty()
                                && constraints
                                    .iter()
                                    .all(|c| self.meets_from_constraint(source_type, *c))
                        }
                        _ => false,
                    }
            })
            .cloned()
    }

    /// A generic conversion reads a structural constraint's fields by name,
    /// which needs a String in its box: the raw string has no header.
    fn from_rule_argument(&mut self, value: IrId, source_type: TypeId, rule: &HirCastRule) -> IrId {
        if let Some(handle) = self.maybe_wrap_for_iter_protocol(value, source_type, rule.from_type)
        {
            return handle;
        }
        let generic = matches!(
            self.type_table.get(rule.from_type).map(|t| &t.kind),
            Some(TypeKind::TypeParameter { .. })
        );
        let is_string = matches!(
            self.type_table
                .get(self.resolve_through_aliases(source_type))
                .map(|t| &t.kind),
            Some(TypeKind::String)
        );
        if generic && is_string {
            self.box_value_for_dynamic(value, source_type)
                .unwrap_or(value)
        } else {
            value
        }
    }

    /// The abstract a type names, through aliases and instantiations.
    fn abstract_name_of(&self, ty: TypeId) -> Option<InternedString> {
        let mut ty = self.resolve_through_aliases(ty);
        for _ in 0..8 {
            match self.type_table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::GenericInstance { base_type, .. }) => {
                    ty = self.resolve_through_aliases(*base_type)
                }
                _ => break,
            }
        }
        self.resolve_abstract_name(ty)
    }

    /// Whether `source` satisfies a generic `@:from` parameter's constraint:
    /// `EnumValue` takes enums, a structure takes any object.
    fn meets_from_constraint(&self, source: TypeId, constraint: TypeId) -> bool {
        let source_kind = self
            .type_table
            .get(self.resolve_through_aliases(source))
            .map(|t| t.kind.clone());
        let constraint = self.resolve_through_aliases(constraint);
        let is_enum_value = self
            .abstract_name_of(constraint)
            .or_else(|| match self.type_table.get(constraint).map(|t| &t.kind) {
                Some(TypeKind::Class { symbol_id, .. }) => {
                    self.symbol_table.get_symbol(*symbol_id).map(|s| s.name)
                }
                _ => None,
            })
            .and_then(|name| self.string_interner.get(name))
            .is_some_and(|name| name == "EnumValue" || name.ends_with(".EnumValue"));
        if is_enum_value {
            return matches!(source_kind, Some(TypeKind::Enum { .. }));
        }
        match self.type_table.get(constraint).map(|t| &t.kind) {
            Some(TypeKind::Anonymous { .. }) => matches!(
                source_kind,
                Some(
                    TypeKind::String
                        | TypeKind::Class { .. }
                        | TypeKind::Anonymous { .. }
                        | TypeKind::Interface { .. }
                )
            ),
            _ => false,
        }
    }

    /// Try to apply an abstract @:from conversion in Let/Assign context.
    /// Returns Some(converted_reg) if a matching @:from rule was found, None otherwise.
    pub(crate) fn maybe_abstract_from_convert(
        &mut self,
        value: IrId,
        source_type: TypeId,
        target_type: TypeId,
    ) -> Option<IrId> {
        // Int64 is the native i64, so `@:from ofInt` is the widening itself;
        // a target spelled as the class under the abstract reaches here too.
        if self.is_int64_type(target_type) {
            let src = self
                .builder
                .get_register_type(value)
                .unwrap_or_else(|| self.convert_type(source_type));
            if matches!(src, IrType::I32 | IrType::I16 | IrType::I8 | IrType::Bool) {
                return self.builder.build_cast(value, src, IrType::I64);
            }
            return None;
        }
        let abs_name = self.resolve_abstract_name(target_type)?;

        let Some(rule) = self.abstract_from_rule(abs_name, source_type) else {
            let mid = self.chained_from_target(target_type, source_type)?;
            return self.maybe_abstract_from_convert(value, source_type, mid);
        };

        if let Some(cast_func_sym) = rule.cast_function {
            let value = self.from_rule_argument(value, source_type, &rule);
            let func_id = self.resolve_abstract_conversion_function(cast_func_sym, target_type)?;
            let result_type = self.convert_type(target_type);
            self.builder
                .build_call_direct(func_id, vec![value], result_type)
        } else {
            // Keyword `from` with no conversion function. The value passes
            // through ONLY when both sides share a representation — `from Float`
            // on `Single` is cast compatibility, not identity, and letting f64
            // bits ride into an f32 slot is a wrong answer, not a slow one.
            let src = self.convert_type(source_type);
            let dst = self.convert_type(target_type);
            let numeric = |t: &IrType| t.is_float() || t.is_integer();
            if src != dst && numeric(&src) && numeric(&dst) {
                self.builder.build_cast(value, src, dst)
            } else {
                None
            }
        }
    }

    /// The source abstract's `@:to` conversion to `target_type`, when it
    /// declares one (`var f:Float = u` with `u:UInt` runs `toFloat`).
    /// Whether `source_type` has an `@:to` method for the target type.
    pub(crate) fn has_abstract_to_function(
        &self,
        source_type: TypeId,
        target_type: TypeId,
    ) -> bool {
        self.has_direct_abstract_to_function(source_type, target_type)
            || self.chained_to_source(source_type, target_type).is_some()
    }

    fn has_direct_abstract_to_function(&self, source_type: TypeId, target_type: TypeId) -> bool {
        let Some(abs_name) = self.resolve_abstract_name(source_type) else {
            return false;
        };
        let Some(target_kind) = self
            .type_table
            .get(self.resolve_through_aliases(target_type))
            .map(|t| t.kind.clone())
        else {
            return false;
        };
        self.abstract_to_rules.get(&abs_name).is_some_and(|rules| {
            rules.iter().any(|r| {
                r.cast_function.is_some()
                    && self
                        .type_table
                        .get(self.resolve_through_aliases(r.to_type))
                        .is_some_and(|t| t.kind == target_kind)
            })
        })
    }

    pub(crate) fn maybe_abstract_to_convert(
        &mut self,
        value: IrId,
        source_type: TypeId,
        target_type: TypeId,
    ) -> Option<IrId> {
        let abs_name = self.resolve_abstract_name(source_type)?;
        let target_kind = self
            .type_table
            .get(self.resolve_through_aliases(target_type))
            .map(|t| t.kind.clone())?;
        let rule = self.abstract_to_rules.get(&abs_name).and_then(|rules| {
            rules
                .iter()
                .find(|r| {
                    r.cast_function.is_some()
                        && self
                            .type_table
                            .get(self.resolve_through_aliases(r.to_type))
                            .is_some_and(|t| t.kind == target_kind)
                })
                .cloned()
        });
        let Some(rule) = rule else {
            let mid = self.chained_to_source(source_type, target_type)?;
            return self.maybe_abstract_to_convert(value, mid, target_type);
        };
        let func_id =
            self.resolve_abstract_conversion_function(rule.cast_function?, source_type)?;
        let result_type = self.convert_type(target_type);
        self.builder
            .build_call_direct(func_id, vec![value], result_type)
    }

    pub(crate) fn try_lower_special_runtime_call(
        &mut self,
        runtime_func: &str,
        args: &[HirExpr],
        result_type: IrType,
        location: SourceLocation,
    ) -> Option<Option<IrId>> {
        match runtime_func {
            "haxe_std_downcast" if args.len() == 2 => {
                // A typed Array is already the raw value the cast should return.
                let source_is_array = matches!(
                    self.type_table
                        .get(self.resolve_through_aliases(args[0].ty))
                        .map(|ty| &ty.kind),
                    Some(TypeKind::Array { .. })
                );
                let target_is_array = matches!(
                    &args[1].kind,
                    HirExprKind::Variable { symbol, .. }
                        if self.symbol_table.get_symbol(*symbol).is_some_and(|sym| {
                            sym.kind == crate::tast::symbols::SymbolKind::Class
                                && self.class_is_named(sym.id, "Array")
                        })
                );
                if source_is_array && target_is_array {
                    return Some(self.lower_expression(&args[0]));
                }
                let source_interface = self.get_interface_symbol(args[0].ty);
                if source_interface.is_some() || self.get_class_symbol(args[0].ty).is_some() {
                    let target_class = match &args[1].kind {
                        HirExprKind::Variable { symbol, .. } => self
                            .symbol_table
                            .get_symbol(*symbol)
                            .filter(|s| s.kind == crate::tast::symbols::SymbolKind::Class)
                            .filter(|s| {
                                !self.class_is_named(s.id, "Array")
                                    && !self.class_is_named(s.id, "String")
                            }),
                        _ => None,
                    };
                    if target_class.is_some() {
                        let mut value = self.lower_expression(&args[0])?;
                        let target = self.lower_expression(&args[1])?;
                        let ptr = IrType::Ptr(Box::new(IrType::U8));
                        // Interface wrappers carry the raw object in word zero.
                        if source_interface.is_some() {
                            let identity = self.get_or_register_extern_function(
                                "haxe_iface_identity",
                                vec![ptr.clone()],
                                ptr.clone(),
                            );
                            value = self.builder.build_call_direct(
                                identity,
                                vec![value],
                                ptr.clone(),
                            )?;
                        }
                        let downcast = self.get_or_register_extern_function(
                            "haxe_safe_downcast_class",
                            vec![ptr.clone(), IrType::I64],
                            ptr.clone(),
                        );
                        let result =
                            self.builder
                                .build_call_direct(downcast, vec![value, target], ptr)?;
                        return Some(self.coerce_reg_to(result, &result_type));
                    }
                }
                None
            }
            "haxe_reflect_call_method" | "Reflect.callMethod" => {
                Some(self.lower_reflect_call_method(args, result_type, location))
            }
            "haxe_reflect_make_var_args" | "Reflect.makeVarArgs" => {
                Some(self.lower_reflect_make_var_args(args, result_type, location))
            }
            "haxe_reflect_set_field" | "haxe_reflect_set_property" if args.len() == 3 => {
                let object = self.lower_expression(&args[0])?;
                let name = self.lower_expression(&args[1])?;
                let value = self.lower_expression(&args[2])?;
                // Reflect setters consume a Dynamic box, including reference values.
                let value_ty = self.resolve_storage_type(args[2].ty);
                let value = self
                    .maybe_box_value(value, value_ty, self.type_table.dynamic_type())
                    .unwrap_or(value);
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let setter = self.get_or_register_extern_function(
                    runtime_func,
                    vec![ptr_u8.clone(), ptr_u8.clone(), ptr_u8],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(setter, vec![object, name, value], IrType::Void);
                Some(None)
            }
            "haxe_type_typeof" | "Type.typeof" => {
                Some(self.lower_type_typeof_call(args, result_type))
            }
            "haxe_type_get_class" if args.len() == 1 => {
                let mut ty = self.resolve_storage_type(args[0].ty);
                if let Some(TypeKind::Optional { inner_type }) =
                    self.type_table.get(ty).map(|ty| &ty.kind)
                {
                    ty = self.resolve_storage_type(*inner_type);
                }
                if !matches!(
                    self.type_table.get(ty).map(|ty| &ty.kind),
                    Some(TypeKind::String | TypeKind::Array { .. })
                ) {
                    return None;
                }
                let value = self.lower_expression(&args[0])?;
                let boxed = self.maybe_box_value(value, ty, self.type_table.dynamic_type())?;
                let ptr = IrType::Ptr(Box::new(IrType::U8));
                let function =
                    self.get_or_register_extern_function(runtime_func, vec![ptr], IrType::I64);
                Some(
                    self.builder
                        .build_call_direct(function, vec![boxed], IrType::I64),
                )
            }
            // `Std.string(x)` converts using the best known type, which is what
            // `convert_to_string_with_hint` does: String is the identity, a type parameter
            // dispatches on its fixed-up tag, and only a genuinely Dynamic value reaches
            // haxe_std_string_ptr. The mapping can't decide this alone — its PtrVoid param
            // descriptor is shared with raw-HaxeString callees like haxe_string_println.
            "haxe_std_string_ptr" => {
                // ValueType enum values keep their own routing at the call site.
                if args.len() != 1 || self.expr_is_value_type_expr(&args[0]) {
                    return None;
                }
                let reg = self.lower_expression(&args[0])?;
                let reg_ty = self
                    .builder
                    .get_register_type(reg)
                    .unwrap_or(IrType::Ptr(Box::new(IrType::Void)));
                let arg_ty = self.resolve_expr_type_id(&args[0]);
                Some(self.convert_to_string_with_hint(reg, &reg_ty, Some(arg_ty)))
            }
            // `Sys.print(v)` / `Sys.println(v)` take a Dynamic and print
            // `Std.string(v)`; the runtime functions take a String.
            "haxe_string_print" | "haxe_string_println" => {
                if args.len() != 1 {
                    return None;
                }
                let arg_ty = self.resolve_expr_type_id(&args[0]);
                if matches!(
                    self.type_table.get(arg_ty).map(|t| &t.kind),
                    Some(TypeKind::String)
                ) {
                    return None;
                }
                let reg = self.lower_expression(&args[0])?;
                let text = if self.expr_is_value_type_expr(&args[0]) {
                    self.convert_value_type_to_string(reg)?
                } else {
                    let reg_ty = self
                        .builder
                        .get_register_type(reg)
                        .unwrap_or(IrType::Ptr(Box::new(IrType::Void)));
                    self.convert_to_string_with_hint(reg, &reg_ty, Some(arg_ty))?
                };
                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                let text = self.builder.build_bitcast(text, ptr_void.clone())?;
                let func_id = self.get_or_register_extern_function(
                    runtime_func,
                    vec![ptr_void],
                    IrType::Void,
                );
                self.builder
                    .build_call_direct(func_id, vec![text], IrType::Void);
                Some(None)
            }
            // These inspect the DynamicValue type_id tag, which raw function/class
            // pointers lack, so the argument is boxed. Register type Ptr(U8) is not a
            // usable shortcut — raw closure and class pointers share it without being
            // boxed DynamicValues; `box_value_for_dynamic` returning None is the
            // pass-through signal for an already-Dynamic HIR type.
            "haxe_reflect_is_function" | "haxe_reflect_is_object" => {
                if args.len() != 1 {
                    return None;
                }
                let value_reg = self.lower_expression(&args[0])?;
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let value_dyn = self
                    .box_value_for_dynamic(value_reg, args[0].ty)
                    .unwrap_or(value_reg);
                let func_id = self.get_or_register_extern_function(
                    runtime_func,
                    vec![ptr_u8.clone()],
                    IrType::Bool,
                );
                Some(
                    self.builder
                        .build_call_direct(func_id, vec![value_dyn], IrType::Bool),
                )
            }
            // A function reflects through its box, which says it has no fields;
            // the raw closure would be read as an object.
            "haxe_reflect_has_field" | "haxe_reflect_field"
                if args.len() == 2
                    && matches!(
                        self.type_table
                            .get(self.resolve_through_aliases(args[0].ty))
                            .map(|t| &t.kind),
                        Some(TypeKind::Function { .. })
                    ) =>
            {
                let value_reg = self.lower_expression(&args[0])?;
                let value_dyn = self
                    .box_value_for_dynamic(value_reg, args[0].ty)
                    .unwrap_or(value_reg);
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let field = self.lower_expression(&args[1])?;
                let field = self.builder.build_bitcast(field, ptr_u8.clone())?;
                let ret = if runtime_func == "haxe_reflect_has_field" {
                    IrType::Bool
                } else {
                    ptr_u8.clone()
                };
                let func_id = self.get_or_register_extern_function(
                    runtime_func,
                    vec![ptr_u8.clone(), ptr_u8],
                    ret.clone(),
                );
                Some(
                    self.builder
                        .build_call_direct(func_id, vec![value_dyn, field], ret),
                )
            }
            // Reflect.compare: use haxe_reflect_compare_typed with a type tag
            // to avoid boxing-based comparison (which loses type info for generics)
            "haxe_reflect_compare" => {
                if args.len() >= 2 {
                    let type_info = self.infer_reflect_compare_type_info(args);
                    if let Some(info) = type_info {
                        let mut arg_regs = Vec::new();
                        for arg in args.iter() {
                            if let Some(reg) = self.lower_expression(arg) {
                                arg_regs.push(self.erase_reflect_compare_arg(reg));
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
                        arg_regs.push(tag_reg);
                        let extern_func_id = self.get_or_register_extern_function(
                            "haxe_reflect_compare_typed",
                            vec![IrType::I64, IrType::I64, IrType::I32],
                            IrType::I64,
                        );
                        let call_result = self.builder.build_call_direct(
                            extern_func_id,
                            arg_regs,
                            IrType::I64,
                        )?;
                        if result_type == IrType::I64 {
                            return Some(Some(call_result));
                        }
                        return Some(self.builder.build_cast(
                            call_result,
                            IrType::I64,
                            result_type,
                        ));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// True if `ty` resolves to an extern abstract whose @:native string == `name`.
    pub(crate) fn type_is_native_named(&self, ty: TypeId, name: &str) -> bool {
        if let Some(ti) = self.type_table.get(ty) {
            if let TypeKind::Abstract { symbol_id, .. } = &ti.kind {
                if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                    return sym
                        .native_name
                        .and_then(|nn| self.string_interner.get(nn))
                        .map(|s| s == name)
                        .unwrap_or(false);
                }
            }
        }
        false
    }
}
