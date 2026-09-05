//! Boxing across the Dynamic, extern and optional boundaries.

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
    pub(crate) fn maybe_box_value(
        &mut self,
        value: IrId,
        value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        let out = self.maybe_box_value_inner(value, value_ty, target_ty);
        if let Some(reg) = out {
            if reg != value {
                // A box was emitted: remember the resulting register so
                // Ptr(Void)-typed consumers can distinguish this genuine
                // Dynamic box from raw payload temps of the same MIR type.
                self.boxed_value_regs.insert(reg);
            }
        }
        out
    }

    pub(crate) fn maybe_box_value_inner(
        &mut self,
        value: IrId,
        value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;

        // Clone TypeKind to avoid borrow checker issues
        let (target_is_dynamic, value_kind_cloned) = {
            let type_table = self.type_table;
            let value_kind = type_table.get(value_ty).map(|t| t.kind.clone());
            let target_is_optional_scalar = match type_table.get(target_ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => type_table
                    .get(*inner_type)
                    .map(|t| matches!(t.kind, TypeKind::Int | TypeKind::Float | TypeKind::Bool))
                    .unwrap_or(false),
                _ => false,
            };
            let value_is_scalar = matches!(
                value_kind,
                Some(TypeKind::Int) | Some(TypeKind::Float) | Some(TypeKind::Bool)
            );
            let target_is_dyn = matches!(
                type_table.get(target_ty).map(|t| &t.kind),
                Some(TypeKind::Dynamic)
            ) || (target_is_optional_scalar && value_is_scalar);
            (target_is_dyn, value_kind)
        };

        let value_is_dynamic = matches!(&value_kind_cloned, Some(TypeKind::Dynamic));
        let needs_boxing = target_is_dynamic && !value_is_dynamic;

        if !needs_boxing {
            return Some(value);
        }

        match &value_kind_cloned {
            // Value types (need malloc + copy)
            Some(TypeKind::Int) => {
                debug!("[BOXING] Auto-boxing Int to Dynamic using haxe_box_int_ptr");
                let value_mir_type = self
                    .builder
                    .get_register_type(value)
                    .unwrap_or_else(|| self.convert_type(value_ty));
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_int_ptr",
                    vec![value_mir_type],
                    ptr_u8,
                );
                self.builder.build_call_direct(
                    box_func_id,
                    vec![value],
                    IrType::Ptr(Box::new(IrType::U8)),
                )
            }
            Some(TypeKind::Float) => {
                debug!("[BOXING] Auto-boxing Float to Dynamic using haxe_box_float_ptr");
                let value_mir_type = self
                    .builder
                    .get_register_type(value)
                    .unwrap_or_else(|| self.convert_type(value_ty));
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_float_ptr",
                    vec![value_mir_type],
                    ptr_u8,
                );
                self.builder.build_call_direct(
                    box_func_id,
                    vec![value],
                    IrType::Ptr(Box::new(IrType::U8)),
                )
            }
            Some(TypeKind::Bool) => {
                debug!("[BOXING] Auto-boxing Bool to Dynamic using haxe_box_bool_ptr");
                let value_mir_type = self
                    .builder
                    .get_register_type(value)
                    .unwrap_or_else(|| self.convert_type(value_ty));
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_bool_ptr",
                    vec![value_mir_type],
                    ptr_u8,
                );
                self.builder.build_call_direct(
                    box_func_id,
                    vec![value],
                    IrType::Ptr(Box::new(IrType::U8)),
                )
            }

            // Reference types (already pointers, just wrap with type_id)
            Some(TypeKind::Class { .. })
            | Some(TypeKind::Enum { .. })
            | Some(TypeKind::Interface { .. })
            | Some(TypeKind::Anonymous { .. })
            | Some(TypeKind::Array { .. }) => {
                debug!(
                    "[BOXING] Auto-boxing reference type {:?} to Dynamic using box_reference",
                    value_kind_cloned
                );

                // Tag with the stable runtime id where one exists (class /
                // interface / enum), so the box agrees with RTTI registration
                // and with the ids baked at `is` and reflection sites. The
                // context-local TypeId only stands in for kinds that have no
                // stable id (anonymous, array).
                let type_id_u32 = match self.runtime_type_id(value_ty) {
                    0 => value_ty.as_raw(),
                    stable => stable,
                };

                let type_id_const = self.builder.build_const(IrValue::U32(type_id_u32))?;

                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_reference_ptr",
                    vec![ptr_u8.clone(), IrType::U32],
                    ptr_u8.clone(),
                );

                self.builder
                    .build_call_direct(box_func_id, vec![value, type_id_const], ptr_u8)
            }

            Some(TypeKind::Function { .. }) => {
                debug!("[BOXING] Auto-boxing Function to Dynamic using haxe_box_function_ptr");
                self.box_value_for_dynamic(value, value_ty)
            }

            Some(TypeKind::String) => {
                debug!("[BOXING] Auto-boxing String to Dynamic via haxe_box_haxestring_ptr");
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let value_as_ptr = self
                    .builder
                    .build_bitcast(value, ptr_u8.clone())
                    .unwrap_or(value);
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_haxestring_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![value_as_ptr], ptr_u8)
            }

            // Abstract, TypeParam, etc. — skip boxing for unsupported types
            _ => {
                debug!(
                    "[BOXING] Unsupported type for boxing: {:?}",
                    value_kind_cloned
                );
                Some(value)
            }
        }
    }

    /// Box a value for extern function calls when the expected type is a pointer but the actual type is a primitive.
    /// This is needed for generic extern classes like Deque<T> where the runtime expects boxed pointers
    /// but the compiler generates primitive values for concrete type parameters like Int.
    ///
    /// Returns the (potentially boxed) value register.
    pub(crate) fn maybe_box_for_extern_call(
        &mut self,
        value: IrId,
        actual_ty: &IrType,
        expected_ty: &IrType,
    ) -> Option<IrId> {
        // Primitive demotions/promotions for typed extern calls (e.g., SIMD4f_make
        // expects f32 lanes but Haxe Float is f64). These are non-boxing casts.
        if actual_ty == expected_ty {
            return Some(value);
        }
        match (actual_ty, expected_ty) {
            (IrType::F64, IrType::F32) => {
                return self.builder.build_cast(value, IrType::F64, IrType::F32);
            }
            (IrType::F32, IrType::F64) => {
                return self.builder.build_cast(value, IrType::F32, IrType::F64);
            }
            _ => {}
        }

        // raw_value_params expect U64 — if the actual is a pointer (class
        // instance, array, etc.), reinterpret its bits as U64 so the runtime
        // sees the address as an inline 64-bit value rather than a
        // register-tagged Ptr where the ABI expects a u64 integer. Bitcast,
        // not numeric cast: pointers and u64 share representation.
        if matches!(expected_ty, IrType::U64) {
            match actual_ty {
                IrType::Ptr(_) => {
                    return self.builder.build_bitcast(value, IrType::U64);
                }
                IrType::F64 => {
                    // f64 bits → u64 raw bits. The Map runtime stores raw u64;
                    // reads back via the corresponding F64 typed `get` reverse
                    // the bitcast.
                    return self.builder.build_bitcast(value, IrType::U64);
                }
                IrType::F32 => {
                    // Promote f32 → f64 (so we have 8 bytes) then bitcast.
                    let promoted = self.builder.build_cast(value, IrType::F32, IrType::F64)?;
                    return self.builder.build_bitcast(promoted, IrType::U64);
                }
                IrType::I32 => {
                    // Zero-extend i32 → i64 → u64 bits.
                    let extended = self.builder.build_cast(value, IrType::I32, IrType::I64)?;
                    return self.builder.build_bitcast(extended, IrType::U64);
                }
                IrType::I64 => {
                    return self.builder.build_bitcast(value, IrType::U64);
                }
                _ => {}
            }
        }

        let expected_is_ptr_u8 = matches!(expected_ty, IrType::Ptr(inner) if matches!(**inner, IrType::U8 | IrType::Void));

        if !expected_is_ptr_u8 {
            return Some(value);
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        match actual_ty {
            IrType::I32 => {
                debug!("[EXTERN BOXING] Boxing I32 to Ptr(U8) for extern call");
                let extended = self.builder.build_cast(value, IrType::I32, IrType::I64)?;
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_int_ptr",
                    vec![IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![extended], ptr_u8)
            }
            IrType::I64 => {
                // I64 values are already pointer-sized — cast directly to Ptr
                // without boxing. This handles Usize/pointer values passed to
                // extern functions expecting Ptr params.
                self.builder
                    .build_cast(value, IrType::I64, expected_ty.clone())
            }
            IrType::Bool => {
                debug!("[EXTERN BOXING] Boxing Bool to Ptr(U8) for extern call");
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_bool_ptr",
                    vec![IrType::Bool],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![value], ptr_u8)
            }
            IrType::F32 => {
                debug!("[EXTERN BOXING] Boxing F32 to Ptr(U8) for extern call");
                let promoted = self.builder.build_cast(value, IrType::F32, IrType::F64)?;
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_float_ptr",
                    vec![IrType::F64],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![promoted], ptr_u8)
            }
            IrType::F64 => {
                debug!("[EXTERN BOXING] Boxing F64 to Ptr(U8) for extern call");
                let box_func_id = self.get_or_register_extern_function(
                    "haxe_box_float_ptr",
                    vec![IrType::F64],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(box_func_id, vec![value], ptr_u8)
            }
            // Already a pointer type (including function pointers) - no boxing needed.
            // Function types also pass through: most callees (Thread.spawn, array.map)
            // expect raw closure pointers, not boxed DynamicValue. Reflection APIs that
            // need boxed functions (Reflect.isFunction) handle boxing at their call sites.
            //
            // String stays raw: PtrVoid is declared both by DynamicValue
            // consumers (haxe_std_string_ptr) and by raw-HaxeString ones
            // (haxe_string_println), so the expected IrType cannot decide this.
            // Dynamic consumers that need a boxed String box at their own site.
            IrType::Function { .. } | IrType::Ptr(_) | IrType::String | IrType::Any => Some(value),
            _ => {
                debug!("[EXTERN BOXING] Skipping boxing for type {:?}", actual_ty);
                Some(value)
            }
        }
    }

    /// A `Null<scalar>` answered as raw bits becomes the box every other
    /// producer hands out — null when `probe` (the container's `exists` over
    /// the same arguments) says the key is absent. `None` leaves the raw value.
    pub(crate) fn box_raw_optional_result(
        &mut self,
        raw: IrId,
        declared_ty: TypeId,
        probe: Option<(crate::ir::IrFunctionId, Vec<IrId>)>,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;
        let inner = match self.type_table.get(declared_ty).map(|t| &t.kind) {
            Some(TypeKind::Optional { inner_type }) => *inner_type,
            _ => return None,
        };
        if !self.optional_inner_is_boxable_primitive(inner) {
            return None;
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let (box_name, arg_ty, arg) = match self.builder.get_register_type(raw)? {
            IrType::I32 => (
                "haxe_box_int_ptr",
                IrType::I64,
                self.builder.build_cast(raw, IrType::I32, IrType::I64)?,
            ),
            IrType::I64 => ("haxe_box_int_ptr", IrType::I64, raw),
            IrType::F64 => ("haxe_box_float_ptr", IrType::F64, raw),
            IrType::F32 => (
                "haxe_box_float_ptr",
                IrType::F64,
                self.builder.build_cast(raw, IrType::F32, IrType::F64)?,
            ),
            IrType::Bool => ("haxe_box_bool_ptr", IrType::Bool, raw),
            _ => return None,
        };
        let box_fn = self.get_or_register_extern_function(box_name, vec![arg_ty], ptr_u8.clone());
        let Some((exists_fn, probe_args)) = probe else {
            return self.builder.build_call_direct(box_fn, vec![arg], ptr_u8);
        };
        let present = self
            .builder
            .build_call_direct(exists_fn, probe_args, IrType::Bool)?;
        let box_block = self.builder.create_block()?;
        let null_block = self.builder.create_block()?;
        let join_block = self.builder.create_block()?;
        self.builder
            .build_cond_branch(present, box_block, null_block)?;
        self.builder.switch_to_block(box_block);
        let boxed = self
            .builder
            .build_call_direct(box_fn, vec![arg], ptr_u8.clone())?;
        self.builder.build_branch(join_block)?;
        self.builder.switch_to_block(null_block);
        let zero = self.builder.build_const(IrValue::I64(0))?;
        let null = self.builder.build_bitcast(zero, ptr_u8.clone())?;
        self.builder.build_branch(join_block)?;
        self.builder.switch_to_block(join_block);
        let result = self.builder.build_phi(join_block, ptr_u8)?;
        self.builder
            .add_phi_incoming(join_block, result, box_block, boxed);
        self.builder
            .add_phi_incoming(join_block, result, null_block, null);
        Some(result)
    }

    /// The container's `exists` over the call's own arguments, when the class
    /// maps one of the same arity: the presence test for a raw `Null<scalar>`.
    pub(crate) fn raw_optional_probe(
        &mut self,
        class_name: &str,
        argc: usize,
        param_types: Vec<IrType>,
        args: Vec<IrId>,
    ) -> Option<(crate::ir::IrFunctionId, Vec<IrId>)> {
        let name = self
            .stdlib_mapping
            .class_key(class_name)
            .and_then(|k| self.stdlib_mapping.find_by_name(k, "exists"))
            .filter(|(_, ex)| ex.param_count == argc)
            .map(|(_, ex)| ex.runtime_name)?;
        let f = self.get_or_register_extern_function(name, param_types, IrType::Bool);
        Some((f, args))
    }

    pub(crate) fn maybe_box_for_optional(
        &mut self,
        value: IrId,
        value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;

        let inner_type = {
            let type_table = self.type_table;
            match type_table.get(target_ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => Some(*inner_type),
                _ => return None,
            }
        };
        let inner_type = inner_type?;

        let is_type_param = {
            let type_table = self.type_table;
            match type_table.get(inner_type).map(|t| &t.kind) {
                Some(TypeKind::TypeParameter { symbol_id, .. }) => {
                    let tp_name = self
                        .symbol_table
                        .get_symbol(*symbol_id)
                        .and_then(|sym| self.string_interner.get(sym.name))
                        .map(|s| s.to_string());
                    tp_name
                }
                _ => None,
            }
        };

        if let Some(tp_name) = is_type_param {
            // TypeParameter inner type: use haxe_box_typed_ptr with a fixup
            // The monomorphizer will replace the type_tag with the correct value
            let val_ir = self.builder.get_register_type(value);
            match val_ir {
                Some(IrType::Ptr(_)) => return None, // Already a pointer
                _ => {}
            }

            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

            // Ensure value is I64
            let v64 = match val_ir {
                Some(IrType::I32) => self.builder.build_cast(value, IrType::I32, IrType::I64)?,
                Some(IrType::Bool) => self.builder.build_cast(value, IrType::Bool, IrType::I64)?,
                _ => value,
            };

            // Create type_tag constant with fixup (placeholder 0, monomorphizer fills in)
            let tag_reg = self.builder.build_const(IrValue::I32(0))?;
            if let Some(func) = self.builder.current_function_mut() {
                func.type_param_tag_fixups.push((tag_reg, tp_name));
            }

            let box_func = self.get_or_register_extern_function(
                "haxe_box_typed_ptr",
                vec![IrType::I64, IrType::I32],
                ptr_u8.clone(),
            );
            return self
                .builder
                .build_call_direct(box_func, vec![v64, tag_reg], ptr_u8);
        }

        // Only box genuine primitive *value* types. An IR-type check alone is
        // not enough: an Enum and a pointer-sized abstract (Usize/Ptr/Ref/Box)
        // both lower to `I64`, yet they are reference/handle values that are
        // already nullable — int-boxing them corrupts the value. Decide from
        // the TAST type kind (resolving abstracts through their underlying
        // primitive); only Int/Float/Bool/Char are boxable.
        if !self.optional_inner_is_boxable_primitive(inner_type) {
            return None;
        }

        let inner_ir = self.convert_type(inner_type);
        // Only box primitives — reference types in Optional are already nullable
        if !matches!(
            inner_ir,
            IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool
        ) {
            return None;
        }

        let val_ir = self.builder.get_register_type(value);
        match val_ir {
            Some(IrType::Ptr(_)) => return None, // Already a pointer, no boxing needed
            _ => {}
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        // Null literals aren't directly detectable: a Dynamic/Unknown source
        // sitting in an I64 register stands in for one.
        let source_is_null_literal = {
            let type_table = self.type_table;
            matches!(
                type_table.get(value_ty).map(|t| &t.kind),
                Some(TypeKind::Dynamic) | Some(TypeKind::Unknown) | None
            )
        };
        if source_is_null_literal && matches!(val_ir, Some(IrType::I64)) {
            return self.builder.build_cast(value, IrType::I64, ptr_u8);
        }

        match &inner_ir {
            IrType::I32 | IrType::I64 => {
                // Ensure value is I64 for haxe_box_int_ptr
                let v64 = if matches!(val_ir, Some(IrType::I32)) {
                    self.builder.build_cast(value, IrType::I32, IrType::I64)?
                } else {
                    value
                };
                let box_func = self.get_or_register_extern_function(
                    "haxe_box_int_ptr",
                    vec![IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_func, vec![v64], ptr_u8)
            }
            IrType::F64 | IrType::F32 => {
                let v64 = if matches!(val_ir, Some(IrType::F32)) {
                    self.builder.build_cast(value, IrType::F32, IrType::F64)?
                } else {
                    value
                };
                let box_func = self.get_or_register_extern_function(
                    "haxe_box_float_ptr",
                    vec![IrType::F64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_func, vec![v64], ptr_u8)
            }
            IrType::Bool => {
                let v64 = self.builder.build_cast(value, IrType::Bool, IrType::I64)?;
                let box_func = self.get_or_register_extern_function(
                    "haxe_box_bool_ptr",
                    vec![IrType::I64],
                    ptr_u8.clone(),
                );
                self.builder.build_call_direct(box_func, vec![v64], ptr_u8)
            }
            _ => None,
        }
    }

    /// Insert automatic unboxing if needed when reading from Dynamic
    /// Returns the (potentially unboxed) value
    pub(crate) fn maybe_unbox_value(
        &mut self,
        value: IrId,
        value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;

        // Clone TypeKind to avoid borrow checker issues
        let (value_is_dynamic, target_kind_cloned) = {
            let type_table = self.type_table;
            let value_kind = type_table.get(value_ty).map(|t| &t.kind);
            let target_kind = type_table.get(target_ty).map(|t| t.kind.clone());

            let value_is_optional_scalar = match value_kind {
                Some(TypeKind::Optional { inner_type }) => type_table
                    .get(*inner_type)
                    .map(|t| matches!(t.kind, TypeKind::Int | TypeKind::Float | TypeKind::Bool))
                    .unwrap_or(false),
                _ => false,
            };
            let target_is_scalar = matches!(
                target_kind,
                Some(TypeKind::Int) | Some(TypeKind::Float) | Some(TypeKind::Bool)
            );

            let value_is_dyn = matches!(value_kind, Some(TypeKind::Dynamic))
                || (value_is_optional_scalar && target_is_scalar);
            (value_is_dyn, target_kind)
        };

        let target_is_dynamic = matches!(&target_kind_cloned, Some(TypeKind::Dynamic));
        let needs_unboxing = value_is_dynamic && !target_is_dynamic;

        if !needs_unboxing {
            return Some(value);
        }

        // A value whose MIR register is already a primitive is not a boxed
        // Dynamic (e.g. Usize ops return I64 but are typed Dynamic in HIR).
        // Same for a reference target whose register is already a pointer:
        // unboxing would read the array/class header's first 8 bytes as a
        // `DynamicValue.value_ptr`.
        let actual_mir_type = self.builder.get_register_type(value);
        if let Some(ref mir_ty) = actual_mir_type {
            let is_already_unboxed = matches!(
                mir_ty,
                IrType::I64 | IrType::I32 | IrType::F64 | IrType::Bool
            );
            if is_already_unboxed {
                debug!(
                    "[UNBOXING] Skipping unbox — value MIR type {:?} is already a primitive",
                    mir_ty
                );
                return Some(value);
            }
            if matches!(mir_ty, IrType::Ptr(_))
                && matches!(
                    &target_kind_cloned,
                    Some(TypeKind::Array { .. })
                        | Some(TypeKind::Interface { .. })
                        | Some(TypeKind::Anonymous { .. })
                )
            {
                debug!(
                    "[UNBOXING] Skipping unbox — value MIR type {:?} is already a pointer for reference target {:?}",
                    mir_ty, target_kind_cloned
                );
                return Some(value);
            }
        }

        match &target_kind_cloned {
            // Value types (need to extract from heap)
            Some(TypeKind::Int) => {
                debug!("[UNBOXING] Auto-unboxing Dynamic to Int using haxe_unbox_int_ptr");
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8],
                    IrType::I32,
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], IrType::I32)
            }
            Some(TypeKind::Float) => {
                debug!("[UNBOXING] Auto-unboxing Dynamic to Float using haxe_unbox_float_ptr");
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8],
                    IrType::F64,
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], IrType::F64)
            }
            Some(TypeKind::Bool) => {
                debug!("[UNBOXING] Auto-unboxing Dynamic to Bool using haxe_unbox_bool_ptr");
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_bool_ptr",
                    vec![ptr_u8],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], IrType::Bool)
            }

            // Enums: a variant access may be typed Dynamic in HIR while the MIR
            // value is a raw i64 discriminant rather than a boxed pointer.
            Some(TypeKind::Enum { .. }) => {
                let actual_type = self.builder.get_register_type(value);
                if matches!(actual_type, Some(IrType::I64) | Some(IrType::I32)) {
                    debug!(
                        "[UNBOXING] Skipping unbox for enum - value is already i64 discriminant"
                    );
                    return Some(value);
                }
                debug!("[UNBOXING] Auto-unboxing Dynamic to Enum using unbox_reference");

                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], ptr_u8)
            }

            // Abstract types (Ptr<T>, Ref<T>, Box<T>, Usize) are zero-cost over
            // Int: plain i64 values, never heap-boxed.
            Some(TypeKind::Abstract { .. }) => {
                debug!("[UNBOXING] Skipping unbox for Abstract type (zero-cost over Int)");
                Some(value)
            }

            // Reference types (just extract the pointer)
            Some(TypeKind::Class { symbol_id, .. }) => {
                // MIR wrapper classes (extern abstracts like Ptr, Ref, Box,
                // Usize) are zero-cost over Int, never heap-boxed.
                if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                    // Qualified name first ("rayzor.Ptr"), then native name.
                    let class_name = sym
                        .qualified_name
                        .and_then(|qn| self.string_interner.get(qn))
                        .map(|qn| qn.to_string())
                        .or_else(|| {
                            sym.native_name
                                .and_then(|nn| self.string_interner.get(nn))
                                .map(|nn| nn.replace("::", "."))
                        });
                    if let Some(cn) = class_name
                        .as_deref()
                        .and_then(|n| self.stdlib_mapping.class_key(n))
                    {
                        if self.stdlib_mapping.is_mir_wrapper_class(cn) {
                            debug!(
                                "[UNBOXING] Skipping unbox for MIR wrapper class '{}' (extern abstract)",
                                cn
                            );
                            return Some(value);
                        }
                    }
                }

                debug!(
                    "[UNBOXING] Auto-unboxing Dynamic to reference type {:?} using unbox_reference",
                    target_kind_cloned
                );

                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );

                self.builder
                    .build_call_direct(unbox_func_id, vec![value], ptr_u8)
            }
            Some(TypeKind::Interface { .. })
            | Some(TypeKind::Anonymous { .. })
            | Some(TypeKind::Array { .. }) => {
                debug!(
                    "[UNBOXING] Auto-unboxing Dynamic to reference type {:?} using unbox_reference",
                    target_kind_cloned
                );

                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );

                self.builder
                    .build_call_direct(unbox_func_id, vec![value], ptr_u8)
            }

            // TODO: String (special struct handling), Abstract (depends on underlying type)
            _ => {
                debug!(
                    "[UNBOXING] Unsupported type for unboxing: {:?}",
                    target_kind_cloned
                );
                Some(value)
            }
        }
    }

    /// Auto-unbox return values from extern calls when runtime returns Ptr(U8) but HIR expects a primitive
    ///
    /// This is the inverse of `maybe_box_for_extern_call`. Generic extern classes like Deque<Int>
    /// store elements as boxed pointers (Ptr(U8)), so when calling pop() the runtime returns
    /// a boxed pointer that needs to be unboxed to the actual type T.
    ///
    /// Also handles nullable types: Null<Int> (Ptr(I32)) - we need to unbox the inner type.
    pub(crate) fn maybe_unbox_for_extern_return(
        &mut self,
        value: IrId,
        actual_ty: &IrType,
        expected_ty: &IrType,
    ) -> Option<IrId> {
        let actual_is_ptr_u8 =
            matches!(actual_ty, IrType::Ptr(inner) if matches!(**inner, IrType::U8 | IrType::Void));

        if !actual_is_ptr_u8 {
            // Truncate I64 → I32 when extern returns i64 but Haxe type is Int (I32).
            if matches!(actual_ty, IrType::I64) && matches!(expected_ty, IrType::I32) {
                return self.builder.build_cast(value, IrType::I64, IrType::I32);
            }
            // Map<K,Float>.get / .iterator and other raw-u64 returners hand back
            // an 8-byte bit pattern; a Float caller reinterprets those bits.
            if matches!(actual_ty, IrType::U64) && matches!(expected_ty, IrType::F64) {
                return self.builder.build_bitcast(value, IrType::F64);
            }
            if matches!(actual_ty, IrType::U64) && matches!(expected_ty, IrType::F32) {
                let f64v = self.builder.build_bitcast(value, IrType::F64)?;
                return self.builder.build_cast(f64v, IrType::F64, IrType::F32);
            }
            return Some(value);
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        // A nullable primitive expectation is a Ptr(primitive) — Null<Int> is
        // Ptr(I32) — so unbox to the inner type.
        let target_type = match expected_ty {
            IrType::Ptr(inner) => match inner.as_ref() {
                IrType::I32 | IrType::I64 | IrType::Bool | IrType::F32 | IrType::F64 => {
                    debug!(
                        "[EXTERN UNBOXING] Nullable primitive detected: Ptr({:?}) -> {:?}",
                        inner, inner
                    );
                    inner.as_ref()
                }
                _ => expected_ty,
            },
            _ => expected_ty,
        };

        match target_type {
            IrType::I32 => {
                debug!("[EXTERN UNBOXING] Unboxing Ptr(U8) return to I32");
                // haxe_unbox_int_ptr returns i64, need to truncate to i32
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8],
                    IrType::I64,
                );
                let i64_val =
                    self.builder
                        .build_call_direct(unbox_func_id, vec![value], IrType::I64)?;
                self.builder.build_cast(i64_val, IrType::I64, IrType::I32)
            }
            IrType::I64 => {
                // I64 expected type comes from TypeParameter via type erasure.
                // The Ptr(U8) IS the raw i64 value (not a boxed DynamicValue*).
                // Just cast ptr→i64 instead of trying to dereference as DynamicValue.
                debug!("[EXTERN UNBOXING] Type-erased Ptr(U8) to I64 — casting (not unboxing)");
                self.builder
                    .build_cast(value, IrType::Ptr(Box::new(IrType::U8)), IrType::I64)
            }
            IrType::Bool => {
                debug!("[EXTERN UNBOXING] Unboxing Ptr(U8) return to Bool");
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_bool_ptr",
                    vec![ptr_u8],
                    IrType::Bool,
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], IrType::Bool)
            }
            IrType::F32 => {
                debug!("[EXTERN UNBOXING] Unboxing Ptr(U8) return to F32");
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8],
                    IrType::F64,
                );
                let f64_val =
                    self.builder
                        .build_call_direct(unbox_func_id, vec![value], IrType::F64)?;
                self.builder.build_cast(f64_val, IrType::F64, IrType::F32)
            }
            IrType::F64 => {
                debug!("[EXTERN UNBOXING] Unboxing Ptr(U8) return to F64");
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8],
                    IrType::F64,
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], IrType::F64)
            }
            IrType::String => {
                debug!("[EXTERN UNBOXING] String type - using reference unboxing");
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8],
                    IrType::String,
                );
                self.builder
                    .build_call_direct(unbox_func_id, vec![value], IrType::String)
            }
            // Already a pointer type or other complex type - no unboxing needed
            _ => {
                debug!(
                    "[EXTERN UNBOXING] No unboxing needed for expected type {:?}",
                    expected_ty
                );
                Some(value)
            }
        }
    }

    /// Auto-unbox a Null<T> (Optional{primitive}) value to its inner primitive type.
    /// Returns Some(unboxed_val) if unboxing was performed, None if not needed.
    pub(crate) fn maybe_unbox_optional(
        &mut self,
        value: IrId,
        source_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;

        let inner_type = {
            let type_table = self.type_table;
            match type_table.get(source_ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => Some(*inner_type),
                _ => return None,
            }
        };
        let inner_type = inner_type?;

        let inner_ir = self.convert_type(inner_type);
        let target_ir = self.convert_type(target_ty);

        // Only unbox when target is the matching primitive type
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        match (&inner_ir, &target_ir) {
            (IrType::I32, IrType::I32) => {
                let unbox_func = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8],
                    IrType::I64,
                );
                let result =
                    self.builder
                        .build_call_direct(unbox_func, vec![value], IrType::I64)?;
                self.builder.build_cast(result, IrType::I64, IrType::I32)
            }
            // Generic-instantiated Null<T> where T is a TypeParameter lowers
            // to (I64, I64): unbox via haxe_unbox_int_ptr and stay at I64.
            (IrType::I64, IrType::I64) => {
                let unbox_func = self.get_or_register_extern_function(
                    "haxe_unbox_int_ptr",
                    vec![ptr_u8],
                    IrType::I64,
                );
                self.builder
                    .build_call_direct(unbox_func, vec![value], IrType::I64)
            }
            (IrType::F64, IrType::F64) => {
                let unbox_func = self.get_or_register_extern_function(
                    "haxe_unbox_float_ptr",
                    vec![ptr_u8],
                    IrType::F64,
                );
                self.builder
                    .build_call_direct(unbox_func, vec![value], IrType::F64)
            }
            (IrType::Bool, IrType::Bool) => {
                let unbox_func = self.get_or_register_extern_function(
                    "haxe_unbox_bool_ptr",
                    vec![ptr_u8],
                    IrType::I64,
                );
                let result =
                    self.builder
                        .build_call_direct(unbox_func, vec![value], IrType::I64)?;
                self.builder.build_cast(result, IrType::I64, IrType::Bool)
            }
            _ => None,
        }
    }

    /// Inverse of `maybe_box_for_optional`: unbox a `Null<T>` value to its
    /// primitive `T` when the target type is the bare primitive. Returns
    /// the unboxed register, or `None` when the conversion doesn't apply.
    /// Used at function-return / variable-assignment sites where the declared
    /// target is a primitive but the expression produces a boxed `Null<T>`.
    pub(crate) fn maybe_unbox_optional_for_target(
        &mut self,
        value: IrId,
        _value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        // Driven by the MIR register type, not the HIR type: a ternary like
        // `(v == null) ? -1 : v` unifies its branches to bare `Int` in TAST
        // while the register still holds a boxed `DynamicValue*` from the
        // `StringMap<Int>.get(k)` branch.
        let target_ir = self.convert_type(target_ty);

        // Only apply to primitive targets — pointer/object targets are
        // already in the right shape and any required null-handling
        // belongs elsewhere.
        if !matches!(
            target_ir,
            IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool
        ) {
            return None;
        }

        let val_ir = self.builder.get_register_type(value);
        // Same shape already — no-op.
        if matches!(&val_ir, Some(t) if *t == target_ir) {
            return None;
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        // A non-`*u8` pointer register (e.g. `*void`) is NOT a boxed
        // `DynamicValue*` — genuine boxes are always `*u8`. It is a
        // switch-as-expression temp holding a RAW primitive payload read out of
        // an enum's `{tag, payload}` struct, surfaced as untyped `*void`
        // because the variant's type was lost. Unboxing it would dereference
        // the payload as a pointer, so recover the raw bits and cast instead.
        let mut is_known_box = self.boxed_value_regs.contains(&value);
        if !is_known_box {
            // Casts re-wrap a box without changing what it is (e.g. the Let
            // binding casts the *u8 box to the Dynamic slot's *void). Follow
            // one level of Cast/BitCast provenance before trusting the
            // raw-payload heuristic below.
            if let Some(func) = self.builder.current_function() {
                'prov: for block in func.cfg.blocks.values() {
                    for inst in &block.instructions {
                        match inst {
                            crate::ir::IrInstruction::Cast { dest, src, .. }
                            | crate::ir::IrInstruction::BitCast { dest, src, .. }
                                if *dest == value =>
                            {
                                is_known_box = self.boxed_value_regs.contains(src);
                                break 'prov;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        if !is_known_box
            && matches!(&val_ir, Some(IrType::Ptr(inner)) if !matches!(inner.as_ref(), IrType::U8))
        {
            let raw = self.builder.build_bitcast(value, IrType::I64)?;
            return match target_ir {
                IrType::I32 => self.builder.build_cast(raw, IrType::I64, IrType::I32),
                IrType::I64 => Some(raw),
                IrType::Bool => self.builder.build_cast(raw, IrType::I64, IrType::Bool),
                // Float payloads are stored as their raw f64 bit pattern, so
                // reinterpret (bitcast) rather than integer-convert.
                IrType::F64 => self.builder.build_bitcast(value, IrType::F64),
                IrType::F32 => self
                    .builder
                    .build_bitcast(value, IrType::F64)
                    .and_then(|f| self.builder.build_cast(f, IrType::F64, IrType::F32)),
                _ => None,
            };
        }

        let is_ptr = matches!(val_ir, Some(IrType::Ptr(_)));
        if is_ptr {
            return match target_ir {
                IrType::I32 => {
                    let unbox_fn = self.get_or_register_extern_function(
                        "haxe_unbox_int_ptr",
                        vec![ptr_u8.clone()],
                        IrType::I32,
                    );
                    self.builder
                        .build_call_direct(unbox_fn, vec![value], IrType::I32)
                }
                IrType::I64 => {
                    let unbox_fn = self.get_or_register_extern_function(
                        "haxe_unbox_int_ptr",
                        vec![ptr_u8.clone()],
                        IrType::I32,
                    );
                    let i32_reg =
                        self.builder
                            .build_call_direct(unbox_fn, vec![value], IrType::I32)?;
                    self.builder.build_cast(i32_reg, IrType::I32, IrType::I64)
                }
                IrType::F64 => {
                    let unbox_fn = self.get_or_register_extern_function(
                        "haxe_unbox_float_ptr",
                        vec![ptr_u8.clone()],
                        IrType::F64,
                    );
                    self.builder
                        .build_call_direct(unbox_fn, vec![value], IrType::F64)
                }
                IrType::F32 => {
                    let unbox_fn = self.get_or_register_extern_function(
                        "haxe_unbox_float_ptr",
                        vec![ptr_u8.clone()],
                        IrType::F64,
                    );
                    let f64_reg =
                        self.builder
                            .build_call_direct(unbox_fn, vec![value], IrType::F64)?;
                    self.builder.build_cast(f64_reg, IrType::F64, IrType::F32)
                }
                IrType::Bool => {
                    let unbox_fn = self.get_or_register_extern_function(
                        "haxe_unbox_bool_ptr",
                        vec![ptr_u8.clone()],
                        IrType::Bool,
                    );
                    self.builder
                        .build_call_direct(unbox_fn, vec![value], IrType::Bool)
                }
                _ => None,
            };
        }

        None
    }

    pub(crate) fn zero_of_ir_type(&mut self, ty: &IrType) -> Option<IrId> {
        let v = match ty {
            IrType::F64 => IrValue::F64(0.0),
            IrType::F32 => IrValue::F32(0.0),
            IrType::I64 => IrValue::I64(0),
            IrType::I32 => IrValue::I32(0),
            IrType::Bool => IrValue::Bool(false),
            _ => return None,
        };
        self.builder.build_const(v)
    }

    /// Return the inner TypeId of an Optional, or None if not Optional.
    pub(crate) fn optional_inner_type(&self, type_id: TypeId) -> Option<TypeId> {
        use crate::tast::TypeKind;
        let type_table = self.type_table;
        match type_table.get(type_id).map(|t| &t.kind) {
            Some(TypeKind::Optional { inner_type }) => Some(*inner_type),
            _ => None,
        }
    }

    /// Whether an `Optional<T>` inner type `T` is a genuine primitive *value*
    /// type that needs int/float boxing to distinguish `null` from `0`/`false`
    /// (`Int`, `Float`, `Bool`, `Char`, or an abstract resolving to one of
    /// those). Returns `false` for reference/handle types — `Enum`, `Class`,
    /// `Interface`, `Array`, `Map`, `String`, `Function`, pointer-sized
    /// abstracts (Usize/Ptr/Ref/Box), etc. — which are already nullable and
    /// must NOT be int-boxed (boxing them corrupts the value). `TypeParameter`
    /// is handled separately upstream, so it is not boxable here.
    pub(crate) fn optional_inner_is_boxable_primitive(&self, inner_type: TypeId) -> bool {
        use crate::tast::TypeKind;
        let mut ty = inner_type;
        // Resolve through abstracts / aliases (bounded to avoid cycles).
        for _ in 0..8 {
            let kind = match self.type_table.get(ty).map(|t| t.kind.clone()) {
                Some(k) => k,
                None => return false,
            };
            match kind {
                TypeKind::Int | TypeKind::Float | TypeKind::Bool | TypeKind::Char => return true,
                TypeKind::Abstract {
                    underlying,
                    symbol_id,
                    ..
                } => {
                    // Pointer-sized abstracts (Usize/Ptr/Ref/Box) and SIMD
                    // vectors carry machine addresses / wide registers; never
                    // int-box them even if a stray underlying is present.
                    let name = self
                        .symbol_table
                        .get_symbol(symbol_id)
                        .and_then(|sym| self.string_interner.get(sym.name))
                        .unwrap_or("");
                    if matches!(
                        name,
                        "Usize"
                            | "Ptr"
                            | "Ref"
                            | "Box"
                            | "SIMD4f"
                            | "SIMD4i32"
                            | "SIMD16i8"
                            | "SIMD8i32"
                            | "SIMD32i8"
                            | "Atomic"
                    ) {
                        return false;
                    }
                    match underlying {
                        Some(u) => ty = u,
                        None => return false,
                    }
                }
                TypeKind::TypeAlias { target_type, .. } => {
                    ty = target_type;
                }
                _ => return false,
            }
        }
        false
    }

    /// Check if a TypeId is Optional{primitive} (Null<Int>, Null<Float>, Null<Bool>).
    pub(crate) fn is_optional_primitive(&self, type_id: TypeId) -> bool {
        use crate::tast::TypeKind;
        let type_table = self.type_table;
        if let Some(TypeKind::Optional { inner_type }) = type_table.get(type_id).map(|t| &t.kind) {
            let inner_ir = self.convert_type(*inner_type);
            matches!(
                inner_ir,
                IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool
            )
        } else {
            false
        }
    }
}
