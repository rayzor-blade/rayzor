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
        // `(cast this : Int)` in an abstract method lowers to
        // Cast(safe, Int) { Cast(unsafe, Dynamic) { This/Variable } }. The inner
        // cast is a no-op for abstract types and the outer one extracts the
        // underlying value, so the pair collapses to the innermost expression.
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
                // abstract → dynamic → underlying is an identity.
                return self.lower_expression(inner_expr);
            }
        }

        // Abstract targets may define @:from conversion rules.
        if let Some(converted) = self.try_abstract_from_cast(expr, *target) {
            return Some(converted);
        }

        let from_type = self.convert_type(expr.ty);
        let to_type = self.convert_type(*target);

        // An abstract and its underlying type share a MIR type, so the cast is a
        // no-op. Safe casts between class types must NOT take this shortcut:
        // both are Ptr(Void), but the runtime still verifies the hierarchy.
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

        // `cast this` in an abstract method has an Abstract source (underlying
        // e.g. I32) and a Dynamic target (Ptr(U8)). That is not Dynamic boxing but
        // the Haxe syntax for extracting the underlying value, so the cast is
        // skipped to avoid reinterpreting an integer as a pointer.
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

        // Unsafe casts are a direct cast instruction with no runtime check; safe
        // casts fall through to the type-specific handlers even when the MIR
        // types match, since Class→Class needs hierarchy verification.
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
                // so downcast comparison must also use TAST TypeId for consistency;
                // an anonymous target carries the runtime's own tag instead.
                let expected_id = match &target_kind {
                    Some(TypeKind::Anonymous { .. }) => self.runtime_type_id(*target) as i64,
                    _ => target.as_raw() as i64,
                };
                let type_id_const = self.builder.build_const(IrValue::I64(expected_id))?;
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

            // Cross-module `(class : Interface)` where the SOURCE type didn't
            // resolve to a Class in this lowering context (it arrives
            // Unknown/Placeholder because the class metadata wasn't imported).
            // The value register is still a raw class object, so recover the
            // source class SymbolId by NAME from the inner expression; otherwise
            // the cast is a silent no-op and a raw pointer lands in an interface
            // slot, where vtable dispatch reads garbage.
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
                    // Downcast or unrelated: runtime check via the object header.
                    // haxe_safe_downcast_class reads the type id at offset 0,
                    // walks the hierarchy, and returns obj_ptr or null. The id
                    // must be the raw `runtime_type_id` the New handler stored.
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

                // Wrap in a fat pointer when the class implements the iface.
                // The Let handler cannot do this: SafeCast's result type is the
                // interface type, so Let would see Interface→Interface.
                //
                // The question is whether THIS CLASS implements the interface.
                // Asking whether the interface is known here instead is true of
                // every interface in the module, so it wraps any class as any
                // interface — a cast that must fail then yields a fat pointer
                // whose slots are filled with unreachable stubs.
                //
                // `interface_vtables` answers directly, but only for classes it
                // has entries for; cross-module the pairs may never have been
                // built. So it is authoritative only once we know SOMETHING
                // about this class, and a class we know nothing about keeps the
                // permissive path that builds a vtable lazily by name.
                let implements = self.interface_vtables.contains_key(&(src_sym, tgt_sym));
                let class_known_here = self
                    .interface_vtables
                    .keys()
                    .any(|(class_sym, _)| *class_sym == src_sym);
                let known_implements = implements
                    || (!class_known_here && self.interface_method_names.contains_key(&tgt_sym));
                if known_implements {
                    let value_reg = self.lower_expression(expr)?;
                    match self.wrap_in_interface_fat_ptr(value_reg, src_sym, tgt_sym) {
                        Some(fat_ptr) => {
                            self.interface_wrapped_args.insert(fat_ptr);
                            Some(fat_ptr)
                        }
                        // No vtable could be built, so there is no interface
                        // value to hand back. The raw class pointer is not one:
                        // call sites would read its fields as vtable slots.
                        None => self.builder.build_const(IrValue::Null),
                    }
                } else {
                    // Does not implement it — a failed cast is null
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
            // When the source fat pointer's vtable already covers the target's
            // slots, the cast passes through. When it does not (e.g. `Module` →
            // `CausalLanguageModel`), reading the target's extra slots would run
            // past the source's allocated method slots, so the fat pointer is
            // rebuilt via `haxe_iface_fat_ptr_build`: take obj_ptr from the
            // source fat pointer, read the class type_id from the object header,
            // look up the per-(class, target_iface) vtable populated in
            // `__vtable_init__`, and allocate a fresh fat pointer.
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
                // Pass-through is only correct when the target's method slots are
                // a PREFIX of the source's (same names, same order), so the source
                // vtable holds each target method at the same index. A method-COUNT
                // test is wrong for a sub-interface with its OWN methods:
                // `Module` (forward, parameters) → `CausalLanguageModel`
                // (forwardIds, resetCache) has equal counts but disjoint slots, so
                // pass-through would dispatch `resetCache` through `parameters`.
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
                    // Source vtable covers the target's slots at the same indices.
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
        // `expr is Type` resolves at compile time for statically-typed code.
        let source_kind = {
            let type_table = self.type_table;
            type_table.get(expr.ty).map(|ti| ti.kind.clone())
        };
        let target_kind = {
            let type_table = self.type_table;
            type_table.get(*expected).map(|ti| ti.kind.clone())
        };

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
            _ if expr.ty == *expected => self.builder.build_const(IrValue::Bool(true)),
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
                    // Downcast: runtime check via the object header. The header
                    // (written by the New handler) and this comparison both use
                    // `runtime_type_id` directly.
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
            // Enum-to-enum: only the same enum declaration matches. SymbolIds
            // are context-local, so two spellings of one declaration can carry
            // different ones; the FQN-derived runtime id is what crosses
            // contexts, and it is the same key the object header and the
            // Dynamic path compare. Type arguments play no part — `is` tests
            // the declaration, not the instantiation.
            (
                Some(TypeKind::Enum {
                    symbol_id: src_sym, ..
                }),
                Some(TypeKind::Enum {
                    symbol_id: tgt_sym, ..
                }),
            ) => {
                let src_sym = *src_sym;
                let tgt_sym = *tgt_sym;
                let same_enum = src_sym == tgt_sym
                    || match (
                        self.deterministic_iface_or_enum_type_id(src_sym, "enum"),
                        self.deterministic_iface_or_enum_type_id(tgt_sym, "enum"),
                    ) {
                        (Some(src_id), Some(tgt_id)) => src_id == tgt_id,
                        // A symbol with no resolvable name leaves the question
                        // undecidable, so keep the permissive answer rather
                        // than report a mismatch that may not be one.
                        _ => true,
                    };
                let _value = self.lower_expression(expr);
                self.builder.build_const(IrValue::Bool(same_enum))
            }
            // Enum against a kind an enum value can never inhabit → false.
            // Enums are their own runtime shape: they are not primitives, and
            // Haxe gives them no place in the class or interface hierarchy.
            (Some(TypeKind::Enum { .. }), Some(TypeKind::Int))
            | (Some(TypeKind::Enum { .. }), Some(TypeKind::Float))
            | (Some(TypeKind::Enum { .. }), Some(TypeKind::Bool))
            | (Some(TypeKind::Enum { .. }), Some(TypeKind::String))
            | (Some(TypeKind::Enum { .. }), Some(TypeKind::Class { .. }))
            | (Some(TypeKind::Enum { .. }), Some(TypeKind::Interface { .. }))
            | (Some(TypeKind::Int), Some(TypeKind::Enum { .. }))
            | (Some(TypeKind::Float), Some(TypeKind::Enum { .. }))
            | (Some(TypeKind::Bool), Some(TypeKind::Enum { .. }))
            | (Some(TypeKind::String), Some(TypeKind::Enum { .. }))
            | (Some(TypeKind::Class { .. }), Some(TypeKind::Enum { .. }))
            | (Some(TypeKind::Interface { .. }), Some(TypeKind::Enum { .. })) => {
                let _value = self.lower_expression(expr);
                self.builder.build_const(IrValue::Bool(false))
            }
            // Fallback: lower the expression (for side effects) and return true.
            // Abstracts, typedefs, arrays, anonymous structures, type parameters
            // and interface-to-class narrowing all land here, and each of them
            // has shapes where the answer really is true — a structurally typed
            // local does hold the class it was assigned, an erased type
            // parameter does hold the argument it was instantiated with. A
            // constant `false` would turn those into silent wrong answers, so
            // the permissive answer stays until each kind gets a real test.
            _ => {
                let _value = self.lower_expression(expr);
                self.builder.build_const(IrValue::Bool(true))
            }
        };
        result
    }
}
