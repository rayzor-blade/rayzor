//! `cast` and `is`: conversions between types, and runtime type tests.

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
    pub(crate) fn lower_cast(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::Cast {
            expr,
            target,
            is_safe,
        } = &expr.kind
        else {
            unreachable!("lower_cast on a non-Cast expression")
        };
        // Collapse double-cast pattern from abstract methods:
        // Cast(safe, target=Int) { Cast(unsafe, target=Dynamic) { This/Variable } }
        // This pattern comes from `(cast this : Int)` syntax in abstract methods.
        // The inner `cast this` (→Dynamic) is a no-op for abstract types, and the
        // outer `(... : Int)` extracts the underlying value. Collapse to just the
        // innermost expression.
        if let HirExprKind::Cast {
            expr: inner_expr,
            target: inner_target,
            is_safe: false,
        } = &expr.kind
        {
            let inner_target_is_dynamic = {
                let type_table = self.type_table;
                type_table
                    .get(*inner_target)
                    .map(|t| matches!(t.kind, TypeKind::Dynamic))
                    .unwrap_or(false)
            };
            let inner_source_is_abstract = {
                let type_table = self.type_table;
                type_table
                    .get(inner_expr.ty)
                    .map(|t| matches!(t.kind, TypeKind::Abstract { .. }))
                    .unwrap_or(false)
            };
            if inner_target_is_dynamic && inner_source_is_abstract {
                // The whole double-cast is an identity: abstract → dynamic → underlying
                // Just lower the innermost expression directly
                return self.lower_expression(inner_expr);
            }
        }

        // Check if target is an abstract type with @:from conversion rules
        // This generalizes the SIMD4f path to work with any abstract
        if let Some(converted) = self.try_abstract_from_cast(expr, *target) {
            return Some(converted);
        }

        let from_type = self.convert_type(expr.ty);
        let to_type = self.convert_type(*target);

        // For abstract types, the cast between abstract and underlying is a no-op
        // (they share the same MIR type).
        // BUT: for safe casts between class types, we must NOT skip — even though
        // both are Ptr(Void), the runtime needs to verify the class hierarchy.
        if from_type == to_type && !*is_safe {
            return self.lower_expression(expr);
        }
        if from_type == to_type && *is_safe {
            // Same MIR type — but for class/interface types, fall through to
            // the type-kind handlers for runtime verification and fat pointer wrapping.
            let type_table = self.type_table;
            let src_kind = type_table.get(expr.ty).map(|t| &t.kind).cloned();
            let tgt_kind = type_table.get(*target).map(|t| &t.kind).cloned();
            let needs_runtime_check = matches!(
                (&src_kind, &tgt_kind),
                (Some(TypeKind::Class { .. }), Some(TypeKind::Class { .. }))
                    | (
                        Some(TypeKind::Class { .. }),
                        Some(TypeKind::Interface { .. })
                    )
                    | (
                        Some(TypeKind::Interface { .. }),
                        Some(TypeKind::Class { .. })
                    )
                    | (
                        Some(TypeKind::Interface { .. }),
                        Some(TypeKind::Interface { .. })
                    )
            );
            if !needs_runtime_check {
                return self.lower_expression(expr);
            }
            // Fall through to Class→Class safe cast handler
        }

        // For `cast this` in abstract methods: the source is Abstract (resolves to
        // underlying e.g. I32) but target is Dynamic (Ptr(U8)). This is NOT a real
        // Dynamic boxing — it's just the Haxe syntax for extracting the underlying
        // value. Skip the cast to avoid reinterpreting integers as pointers.
        // Also handle the case where `cast this` target is Dynamic and the source
        // is a primitive type (the underlying type of the abstract).
        {
            let type_table = self.type_table;
            let source_kind = type_table.get(expr.ty).map(|t| t.kind.clone());
            let target_kind = type_table.get(*target).map(|t| t.kind.clone());

            let source_is_abstract = matches!(&source_kind, Some(TypeKind::Abstract { .. }));
            let target_is_dynamic = matches!(&target_kind, Some(TypeKind::Dynamic));

            if source_is_abstract && target_is_dynamic {
                return self.lower_expression(expr);
            }

            // Also handle: source is concrete primitive, target is Dynamic,
            // inside an abstract method body where `this` was registered as
            // the underlying type. `cast this` produces
            // Cast(This(underlying_type) → Dynamic), which is a no-op.
            if target_is_dynamic
                && !*is_safe
                && matches!(
                    &source_kind,
                    Some(TypeKind::Int)
                        | Some(TypeKind::Float)
                        | Some(TypeKind::Bool)
                        | Some(TypeKind::String)
                )
                && matches!(&expr.kind, HirExprKind::This)
            {
                return self.lower_expression(expr);
            }
        }

        // For unsafe casts, emit a direct cast instruction (no runtime check).
        // For safe casts, we need to fall through to the type-specific
        // handling even when MIR types match (e.g., Class→Class are both
        // Ptr(U8) but need runtime hierarchy verification).
        if !*is_safe {
            let value_reg = self.lower_expression(expr)?;
            return self.builder.build_cast(value_reg, from_type, to_type);
        }

        // Safe cast: resolve at compile time based on source/target type kinds
        let source_kind = {
            let type_table = self.type_table;
            type_table.get(expr.ty).map(|ti| ti.kind.clone())
        };
        let target_kind = {
            let type_table = self.type_table;
            type_table.get(*target).map(|ti| ti.kind.clone())
        };

        match (&source_kind, &target_kind) {
            // Primitive-to-primitive: always succeeds, emit static conversion
            (Some(TypeKind::Int), Some(TypeKind::Float))
            | (Some(TypeKind::Float), Some(TypeKind::Int))
            | (Some(TypeKind::Int), Some(TypeKind::Bool))
            | (Some(TypeKind::Bool), Some(TypeKind::Int))
            | (Some(TypeKind::Float), Some(TypeKind::Bool))
            | (Some(TypeKind::Bool), Some(TypeKind::Float)) => {
                let value_reg = self.lower_expression(expr)?;
                self.builder.build_cast(value_reg, from_type, to_type)
            }

            // Dynamic → primitive type: unbox the DynamicValue
            // But skip unboxing if the source register is already a raw
            // primitive (e.g., `cast this` in enum abstract methods where
            // `this` is the underlying Int value, not a boxed DynamicValue*).
            (Some(TypeKind::Dynamic), Some(TypeKind::Int)) => {
                let value_reg = self.lower_expression(expr)?;
                let reg_ty = self.builder.get_register_type(value_reg);
                if matches!(
                    reg_ty,
                    Some(IrType::I32) | Some(IrType::I64) | Some(IrType::U32)
                ) {
                    // Already a raw integer — cast to I32 if needed
                    if matches!(reg_ty, Some(IrType::I32)) {
                        Some(value_reg)
                    } else {
                        self.builder
                            .build_cast(value_reg, reg_ty.unwrap(), IrType::I32)
                    }
                } else {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let unbox_id = self.get_or_register_extern_function(
                        "haxe_unbox_int_ptr",
                        vec![ptr_u8],
                        IrType::I32,
                    );
                    self.builder
                        .build_call_direct(unbox_id, vec![value_reg], IrType::I32)
                }
            }
            (Some(TypeKind::Dynamic), Some(TypeKind::Float)) => {
                let value_reg = self.lower_expression(expr)?;
                let reg_ty = self.builder.get_register_type(value_reg);
                if matches!(reg_ty, Some(IrType::F32) | Some(IrType::F64)) {
                    if matches!(reg_ty, Some(IrType::F64)) {
                        Some(value_reg)
                    } else {
                        self.builder.build_cast(value_reg, IrType::F32, IrType::F64)
                    }
                } else {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let unbox_id = self.get_or_register_extern_function(
                        "haxe_unbox_float_ptr",
                        vec![ptr_u8],
                        IrType::F64,
                    );
                    self.builder
                        .build_call_direct(unbox_id, vec![value_reg], IrType::F64)
                }
            }
            (Some(TypeKind::Dynamic), Some(TypeKind::Bool)) => {
                let value_reg = self.lower_expression(expr)?;
                let reg_ty = self.builder.get_register_type(value_reg);
                if matches!(reg_ty, Some(IrType::Bool)) {
                    Some(value_reg)
                } else {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let unbox_id = self.get_or_register_extern_function(
                        "haxe_unbox_bool_ptr",
                        vec![ptr_u8],
                        IrType::Bool,
                    );
                    self.builder
                        .build_call_direct(unbox_id, vec![value_reg], IrType::Bool)
                }
            }
            // Dynamic → non-primitive type: runtime downcast (null on failure)
            (Some(TypeKind::Dynamic), _) if !matches!(&target_kind, Some(TypeKind::Dynamic)) => {
                let value_reg = self.lower_expression(expr)?;
                // DynamicValue boxing uses TAST TypeId (value_ty.as_raw()),
                // so downcast comparison must also use TAST TypeId for consistency.
                let type_id_const = self
                    .builder
                    .build_const(IrValue::I64(target.as_raw() as i64))?;
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let downcast_id = self.get_or_register_extern_function(
                    "haxe_std_downcast",
                    vec![ptr_u8.clone(), IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(downcast_id, vec![value_reg, type_id_const], ptr_u8)
            }

            // Concrete → Dynamic: box the value
            (_, Some(TypeKind::Dynamic)) => {
                let value_reg = self.lower_expression(expr)?;
                self.maybe_box_value(value_reg, expr.ty, *target)
                    .or(Some(value_reg))
            }

            // Cross-module `(class : Interface)` where the SOURCE type
            // didn't resolve to a Class in this MIR-lowering context
            // (arrives Unknown/Placeholder — the class metadata wasn't
            // imported here). The typechecker still promoted this to a
            // Class→Interface cast (Bug #2's TAST promotion), and the
            // value register IS a raw class object we can wrap: recover
            // the source class SymbolId by NAME from the inner
            // expression. Without this, the cast is a silent no-op and
            // the raw pointer flows into an interface slot → later
            // vtable dispatch reads garbage (the nue `new LlamaArch()` →
            // ArchBuilder registration path).
            (
                src,
                Some(TypeKind::Interface {
                    symbol_id: tgt_sym, ..
                }),
            ) if !matches!(
                src,
                Some(TypeKind::Class { .. }) | Some(TypeKind::Interface { .. })
            ) && (self.recover_cast_src_class_by_name(expr).is_some()
                || self.new_arg_class_fqn(expr).is_some()) =>
            {
                let tgt_sym = *tgt_sym;
                let src_sym = self.recover_cast_src_class_by_name(expr);
                let value_reg = self.lower_expression(expr)?;
                if self.interface_wrapped_args.contains(&value_reg) {
                    Some(value_reg)
                } else if let Some(src_sym) = src_sym.filter(|s| {
                    self.interface_vtables.contains_key(&(*s, tgt_sym))
                        || self.interface_method_names.contains_key(&tgt_sym)
                }) {
                    // In-context (or forwardable) class symbol available.
                    match self.wrap_in_interface_fat_ptr(value_reg, src_sym, tgt_sym) {
                        Some(fat_ptr) => {
                            self.interface_wrapped_args.insert(fat_ptr);
                            Some(fat_ptr)
                        }
                        None => Some(value_reg),
                    }
                } else if let Some(class_fqn) = self.new_arg_class_fqn(expr) {
                    // Fully-imported class not present here: wrap by NAME
                    // via forward-ref dispatch thunks (order-independent).
                    match self.wrap_new_class_as_interface_by_name(value_reg, &class_fqn, tgt_sym) {
                        Some(fat_ptr) => {
                            self.interface_wrapped_args.insert(fat_ptr);
                            Some(fat_ptr)
                        }
                        None => Some(value_reg),
                    }
                } else {
                    Some(value_reg)
                }
            }

            // Class → Class safe cast: runtime downcast via object header
            (
                Some(TypeKind::Class {
                    symbol_id: src_sym, ..
                }),
                Some(TypeKind::Class {
                    symbol_id: tgt_sym, ..
                }),
            ) => {
                let src_sym = *src_sym;
                let tgt_sym = *tgt_sym;
                let value_reg = self.lower_expression(expr)?;

                if src_sym == tgt_sym || self.is_subclass_of(expr.ty, *target) {
                    // Same class or upcast: always succeeds
                    self.builder.build_cast(value_reg, from_type, to_type)
                } else {
                    // Downcast or unrelated: runtime check via object header.
                    // haxe_safe_downcast_class reads the type id from offset 0,
                    // walks the hierarchy, and returns obj_ptr or null.
                    // Use `runtime_type_id` directly so the value here
                    // matches the value the New handler stored in the
                    // header (both sides are now the deterministic
                    // name-hash, no transform on either end).
                    let _ = tgt_sym;
                    let target_type_id = self.runtime_type_id(*target) as i64;
                    let type_id_const = self.builder.build_const(IrValue::I64(target_type_id))?;
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let downcast_func = self.get_or_register_extern_function(
                        "haxe_safe_downcast_class",
                        vec![ptr_u8.clone(), IrType::I64],
                        ptr_u8.clone(),
                    );
                    self.builder.build_call_direct(
                        downcast_func,
                        vec![value_reg, type_id_const],
                        ptr_u8,
                    )
                }
            }

            // Class → Interface safe cast: wrap in fat pointer if implements, null if not
            (
                Some(TypeKind::Class {
                    symbol_id: src_sym, ..
                }),
                Some(TypeKind::Interface {
                    symbol_id: tgt_sym, ..
                }),
            ) => {
                let src_sym = *src_sym;
                let tgt_sym = *tgt_sym;

                // Wrap in fat pointer if the class implements the iface.
                // Cannot defer to the Let handler because SafeCast's
                // result type is the interface type, so Let would see
                // Interface→Interface instead of Class→Interface.
                //
                // Presence in `interface_vtables` is the fast check, but
                // cross-module the eager (class, iface) pair is often
                // missing here (built lazily). `wrap_in_interface_fat_ptr`
                // has a name-based lazy-build fallback, so ATTEMPT the
                // wrap whenever the interface is known to have methods
                // in this context; only fall back to null when we can't
                // build a vtable at all (class genuinely doesn't
                // implement it). Returning null on a valid cross-module
                // implementer was the SIGSEGV root for a plain
                // `param:Interface` arg — the raw class pointer flowed
                // through and dispatch read past the object.
                let known_implements = self.interface_vtables.contains_key(&(src_sym, tgt_sym))
                    || self.interface_method_names.contains_key(&tgt_sym);
                if known_implements {
                    let value_reg = self.lower_expression(expr)?;
                    match self.wrap_in_interface_fat_ptr(value_reg, src_sym, tgt_sym) {
                        Some(fat_ptr) => {
                            self.interface_wrapped_args.insert(fat_ptr);
                            Some(fat_ptr)
                        }
                        None => Some(value_reg),
                    }
                } else {
                    // Not known to implement at compile time — return null
                    self.builder.build_const(IrValue::Null)
                }
            }

            // Interface → Class safe cast: extract raw obj + downcast
            (
                Some(TypeKind::Interface { .. }),
                Some(TypeKind::Class {
                    symbol_id: tgt_sym, ..
                }),
            ) => {
                let tgt_sym = *tgt_sym;
                let value_reg = self.lower_expression(expr)?;

                // Load raw object pointer from fat pointer offset 0
                let raw_obj = self
                    .builder
                    .build_load(value_reg, IrType::Ptr(Box::new(IrType::U8)))?;

                // Downcast via object header type_id check.
                // Match the New handler: object header stores the
                // raw `runtime_type_id` (deterministic name-hash).
                let _ = tgt_sym;
                let target_type_id = self.runtime_type_id(*target) as i64;
                let type_id_const = self.builder.build_const(IrValue::I64(target_type_id))?;
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let downcast_func = self.get_or_register_extern_function(
                    "haxe_safe_downcast_class",
                    vec![ptr_u8.clone(), IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(downcast_func, vec![raw_obj, type_id_const], ptr_u8)
            }

            // Interface → Interface cast.
            //
            // Three sub-cases:
            //
            // - Same iface or source has ≥ target's methods: the
            //   existing fat pointer's vtable already covers the
            //   target's slots (interface-method order is stable
            //   per declared interface), so a pass-through cast is
            //   safe.
            //
            // - Source has fewer methods than target (e.g.
            //   `Module` → `CausalLanguageModel`): the source fat
            //   pointer was built with only the source iface's
            //   slots. Reading the target's extra slots from the
            //   source fat pointer runs past the allocated method
            //   slots into garbage → call-indirect to a bogus
            //   address → SIGSEGV.
            //
            //   Rebuild via `haxe_iface_fat_ptr_build`: extract
            //   the obj_ptr from the source fat pointer, read the
            //   class type_id from the object header, look up the
            //   per-(class, target_iface) vtable populated in
            //   `__vtable_init__`, and allocate a fresh fat
            //   pointer with the correct method slots.
            (
                Some(TypeKind::Interface {
                    symbol_id: src_iface_sym,
                    ..
                }),
                Some(TypeKind::Interface {
                    symbol_id: tgt_iface_sym,
                    ..
                }),
            ) => {
                let src_iface_sym = *src_iface_sym;
                let tgt_iface_sym = *tgt_iface_sym;
                // Pass-through is only correct when the target's method
                // slots are a PREFIX of the source's (same names, same
                // order) — i.e. the source vtable already holds each
                // target method at the same index. A method-COUNT test
                // is wrong for a sub-interface with its OWN methods:
                // `Module` (forward, parameters) → `CausalLanguageModel`
                // (forwardIds, resetCache) has equal counts but disjoint
                // slots, so pass-through would dispatch `resetCache`
                // through `Module`'s `parameters` slot. Rebuild unless
                // the names line up as a prefix.
                let target_is_prefix = match (
                    self.interface_method_names.get(&src_iface_sym),
                    self.interface_method_names.get(&tgt_iface_sym),
                ) {
                    (Some(src_names), Some(tgt_names)) => {
                        tgt_names.len() <= src_names.len()
                            && tgt_names.iter().zip(src_names.iter()).all(|(t, s)| t == s)
                    }
                    // Same interface (src == tgt) is trivially a prefix.
                    _ => src_iface_sym == tgt_iface_sym,
                };
                if src_iface_sym == tgt_iface_sym || target_is_prefix {
                    // Source vtable covers the target's slots at the same
                    // indices — existing pass-through behaviour is fine.
                    let value_reg = self.lower_expression(expr)?;
                    self.builder.build_cast(value_reg, from_type, to_type)
                } else {
                    // Target has methods source doesn't —
                    // rebuild the fat pointer with the target's
                    // method slots via the runtime registry.
                    let value_reg = self.lower_expression(expr)?;
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let src_ptr = {
                        let src_ty = self
                            .builder
                            .get_register_type(value_reg)
                            .unwrap_or(IrType::I64);
                        if matches!(src_ty, IrType::Ptr(_)) {
                            value_reg
                        } else {
                            self.builder.build_bitcast(value_reg, ptr_u8.clone())?
                        }
                    };
                    let obj_ptr = self.builder.build_load(src_ptr, ptr_u8.clone())?;
                    let target_iface_type_id =
                        self.deterministic_iface_or_enum_type_id(tgt_iface_sym, "iface")
                            .unwrap_or(tgt_iface_sym.as_raw()) as i32;
                    let iface_tid_reg = self
                        .builder
                        .build_const(IrValue::I32(target_iface_type_id))?;
                    let rebuild_fn = self.get_or_register_extern_function(
                        "haxe_iface_fat_ptr_build",
                        vec![ptr_u8.clone(), IrType::I32],
                        ptr_u8.clone(),
                    );
                    self.builder
                        .build_call_direct(rebuild_fn, vec![obj_ptr, iface_tid_reg], ptr_u8)
                }
            }

            // Fallback: emit raw cast (same as unsafe)
            _ => {
                let value_reg = self.lower_expression(expr)?;
                self.builder.build_cast(value_reg, from_type, to_type)
            }
        }
    }

    pub(crate) fn lower_type_check(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::TypeCheck { expr, expected } = &expr.kind else {
            unreachable!("lower_type_check on a non-TypeCheck expression")
        };
        // (expr is Type) — compile-time type check for statically-typed code
        let source_kind = {
            let type_table = self.type_table;
            type_table.get(expr.ty).map(|ti| ti.kind.clone())
        };
        let target_kind = {
            let type_table = self.type_table;
            type_table.get(*expected).map(|ti| ti.kind.clone())
        };

        // For statically-typed code, resolve at compile time
        let result = match (&source_kind, &target_kind) {
            // Dynamic source: runtime type check via haxe_std_is
            (Some(TypeKind::Dynamic), _) => {
                let value_reg = self.lower_expression(expr)?;
                let rt_type_id = self.runtime_type_id(*expected);
                let type_id_const = self.builder.build_const(IrValue::I64(rt_type_id as i64))?;
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let is_func_id = self.get_or_register_extern_function(
                    "haxe_std_is",
                    vec![ptr_u8, IrType::I64],
                    IrType::Bool,
                );
                self.builder.build_call_direct(
                    is_func_id,
                    vec![value_reg, type_id_const],
                    IrType::Bool,
                )
            }
            // Same type kind → always true (but null check needed for refs)
            _ if expr.ty == *expected => {
                // Exact same type → true
                self.builder.build_const(IrValue::Bool(true))
            }
            // Primitive type checks
            (Some(TypeKind::Int), Some(TypeKind::Int))
            | (Some(TypeKind::Float), Some(TypeKind::Float))
            | (Some(TypeKind::Bool), Some(TypeKind::Bool))
            | (Some(TypeKind::String), Some(TypeKind::String)) => {
                self.builder.build_const(IrValue::Bool(true))
            }
            // Cross-primitive: always false
            (Some(TypeKind::Int), Some(TypeKind::Float))
            | (Some(TypeKind::Float), Some(TypeKind::Int))
            | (Some(TypeKind::Int), Some(TypeKind::String))
            | (Some(TypeKind::String), Some(TypeKind::Int))
            | (Some(TypeKind::Float), Some(TypeKind::String))
            | (Some(TypeKind::String), Some(TypeKind::Float))
            | (Some(TypeKind::Bool), Some(TypeKind::Int))
            | (Some(TypeKind::Int), Some(TypeKind::Bool))
            | (Some(TypeKind::Bool), Some(TypeKind::Float))
            | (Some(TypeKind::Float), Some(TypeKind::Bool)) => {
                self.builder.build_const(IrValue::Bool(false))
            }
            // Class-to-class: check class hierarchy at compile time
            (
                Some(TypeKind::Class {
                    symbol_id: src_sym, ..
                }),
                Some(TypeKind::Class {
                    symbol_id: tgt_sym, ..
                }),
            ) => {
                let src_sym = *src_sym;
                let tgt_sym = *tgt_sym;
                if src_sym == tgt_sym {
                    // Same class → true
                    let _value = self.lower_expression(expr);
                    self.builder.build_const(IrValue::Bool(true))
                } else if self.is_subclass_of(expr.ty, *expected) {
                    // Source is subclass of target (upcast) → always true
                    let _value = self.lower_expression(expr);
                    self.builder.build_const(IrValue::Bool(true))
                } else if self.is_subclass_of(*expected, expr.ty) {
                    // Target is subclass of source (downcast) →
                    // runtime check via object header. Both the
                    // header (written by the New handler) and this
                    // comparison use `runtime_type_id` directly,
                    // which is the deterministic name-hash. No
                    // ±1000 transform on either side.
                    let value_reg = self.lower_expression(expr)?;
                    let target_type_id = self.runtime_type_id(*expected) as i64;
                    let type_id_const = self.builder.build_const(IrValue::I64(target_type_id))?;
                    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                    let is_func = self.get_or_register_extern_function(
                        "haxe_object_is_instance",
                        vec![ptr_void, IrType::I64],
                        IrType::I64,
                    );
                    let result_i64 = self.builder.build_call_direct(
                        is_func,
                        vec![value_reg, type_id_const],
                        IrType::I64,
                    )?;
                    // Convert i64 (0/1) to Bool
                    let zero = self.builder.build_const(IrValue::I64(0))?;
                    self.builder.build_cmp(CompareOp::Ne, result_i64, zero)
                } else {
                    // Unrelated classes → false
                    let _value = self.lower_expression(expr);
                    self.builder.build_const(IrValue::Bool(false))
                }
            }
            // Class -> Interface: runtime check against registered interface impl map.
            (Some(TypeKind::Class { .. }), Some(TypeKind::Interface { .. })) => {
                let value_reg = self.lower_expression(expr)?;
                let target_type_id = self.runtime_type_id(*expected) as i64;
                let type_id_const = self.builder.build_const(IrValue::I64(target_type_id))?;
                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                let is_func = self.get_or_register_extern_function(
                    "haxe_object_is_instance",
                    vec![ptr_void, IrType::I64],
                    IrType::I64,
                );
                let result_i64 = self.builder.build_call_direct(
                    is_func,
                    vec![value_reg, type_id_const],
                    IrType::I64,
                )?;
                let zero = self.builder.build_const(IrValue::I64(0))?;
                self.builder.build_cmp(CompareOp::Ne, result_i64, zero)
            }
            // Class vs primitive or other unrelated → false
            (Some(TypeKind::Class { .. }), Some(TypeKind::Int))
            | (Some(TypeKind::Class { .. }), Some(TypeKind::Float))
            | (Some(TypeKind::Class { .. }), Some(TypeKind::Bool))
            | (Some(TypeKind::Class { .. }), Some(TypeKind::String))
            | (Some(TypeKind::Int), Some(TypeKind::Class { .. }))
            | (Some(TypeKind::Float), Some(TypeKind::Class { .. }))
            | (Some(TypeKind::Bool), Some(TypeKind::Class { .. }))
            | (Some(TypeKind::String), Some(TypeKind::Class { .. })) => {
                let _value = self.lower_expression(expr);
                self.builder.build_const(IrValue::Bool(false))
            }
            // Fallback: lower the expression (for side effects) and return true
            // (trust static types — proper runtime checks need object headers)
            _ => {
                let _value = self.lower_expression(expr);
                self.builder.build_const(IrValue::Bool(true))
            }
        };
        result
    }
}
