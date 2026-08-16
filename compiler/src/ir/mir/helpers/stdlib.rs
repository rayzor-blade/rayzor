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
    ) -> Option<String> {
        if let Some(native) = symbol.native_name {
            if let Some(native_str) = self.string_interner.get(native) {
                let dotted = native_str.replace("::", ".");
                if let Some(key) = self.stdlib_mapping.get_class_static_str(&dotted) {
                    return Some(key.to_string());
                }
            }
        }
        let class_name = self.string_interner.get(symbol.name)?;
        self.stdlib_mapping
            .get_class_static_str(class_name)
            .map(str::to_string)
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
        Some(key)
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
        let mapping_info = self
            .stdlib_mapping
            .find_by_name(&class_name, "fromArray")
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
        let runtime_name = self
            .stdlib_mapping
            .find_by_name(&hint, &field_name)
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

        let candidate_classes = [dot_form.as_str()];
        let mut found: Option<(String, IrType, Vec<IrType>, bool)> = None;
        for cn in &candidate_classes {
            if let Some((_sig, mapping)) = self
                .stdlib_mapping
                .find_by_name_and_params(cn, method_name, expected_extra)
                .or_else(|| self.stdlib_mapping.find_by_name(cn, method_name))
            {
                let runtime_name = mapping.runtime_name.to_string();
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let param_types = vec![ptr_u8.clone(); args.len()];
                found = Some((
                    runtime_name,
                    return_type.clone(),
                    param_types,
                    mapping.is_mir_wrapper,
                ));
                break;
            }
        }
        let (runtime_name, ret_ty, param_types, is_mir_wrapper) = found?;
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
            None => return Some(None), // Not a class type
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

        // Fallback for stdlib classes (StringMap, IntMap, Date, Bytes): a toString()
        // registered in stdlib_mapping.
        let class_name = self
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("")
            .to_string();

        if let Some((_sig, mapping)) = self.stdlib_mapping.find_by_name(&class_name, "toString") {
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

        let matching_rule = self
            .abstract_from_rules
            .get(&abs_name)
            .and_then(|rules| rules.iter().find(|r| r.from_type == source_type))
            .cloned();

        if let Some(rule) = matching_rule {
            if let Some(cast_func_sym) = rule.cast_function {
                let value_reg = self.lower_expression(expr)?;
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
        } else {
            // Fallback: for extern/imported abstracts (e.g., SIMD4f) whose @:from rules
            // weren't populated (not in file.abstracts), try stdlib mapping directly.
            self.try_stdlib_from_cast(expr, target, &abs_name)
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
        let abs_name = self.resolve_abstract_name(target_type)?;

        let matching_rule = self
            .abstract_from_rules
            .get(&abs_name)
            .and_then(|rules| rules.iter().find(|r| r.from_type == source_type))
            .cloned();

        let rule = matching_rule?;

        if let Some(cast_func_sym) = rule.cast_function {
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

    pub(crate) fn try_lower_special_runtime_call(
        &mut self,
        runtime_func: &str,
        args: &[HirExpr],
        result_type: IrType,
        location: SourceLocation,
    ) -> Option<Option<IrId>> {
        match runtime_func {
            "haxe_reflect_call_method" | "Reflect.callMethod" => {
                Some(self.lower_reflect_call_method(args, result_type, location))
            }
            "haxe_reflect_make_var_args" | "Reflect.makeVarArgs" => {
                Some(self.lower_reflect_make_var_args(args, result_type, location))
            }
            "haxe_type_typeof" | "Type.typeof" => {
                Some(self.lower_type_typeof_call(args, result_type))
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
            // Reflect.compare: use haxe_reflect_compare_typed with a type tag
            // to avoid boxing-based comparison (which loses type info for generics)
            "haxe_reflect_compare" => {
                if args.len() >= 2 {
                    let type_info = self.infer_reflect_compare_type_info(args);
                    if let Some(info) = type_info {
                        let mut arg_regs = Vec::new();
                        for arg in args.iter() {
                            if let Some(reg) = self.lower_expression(arg) {
                                let reg_ty =
                                    self.builder.get_register_type(reg).unwrap_or(IrType::I64);
                                let final_reg = if reg_ty != IrType::I64 {
                                    self.builder
                                        .build_cast(reg, reg_ty, IrType::I64)
                                        .unwrap_or(reg)
                                } else {
                                    reg
                                };
                                arg_regs.push(final_reg);
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
                            result_type.clone(),
                        );
                        return Some(self.builder.build_call_direct(
                            extern_func_id,
                            arg_regs,
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
