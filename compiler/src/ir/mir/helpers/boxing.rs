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
    /// Array slot kinds: zero is boxed, builtins use their runtime tag, and
    /// references carry their type ID above kind 6. Kind 7 nests an array layout;
    /// kind 8 is an unboxed `Null<Int>`.
    pub(crate) fn erased_array_slot_layout(&self, ty: TypeId, depth: usize) -> u64 {
        if depth >= 8 {
            return 0;
        }
        if self.is_null_int_type(ty) {
            return 8;
        }
        if self.is_int64_type(ty) {
            return 9;
        }
        let ty = self.resolve_storage_type(ty);
        match self.type_table.get(ty).map(|t| &t.kind) {
            Some(TypeKind::Int) => 3,
            Some(TypeKind::Float) => 4,
            Some(TypeKind::Bool) => 2,
            Some(TypeKind::String) => 5,
            Some(TypeKind::Array { element_type }) => {
                let inner = self.erased_array_slot_layout(*element_type, depth + 1);
                if inner >> 56 != 0 {
                    0
                } else {
                    7 | (inner << 8)
                }
            }
            Some(
                TypeKind::Class { .. }
                | TypeKind::Enum { .. }
                | TypeKind::Interface { .. }
                | TypeKind::Anonymous { .. }
                | TypeKind::Function { .. },
            ) => 6 | (u64::from(self.runtime_type_id(ty)) << 8),
            _ => 0,
        }
    }

    pub(crate) fn maybe_box_value(
        &mut self,
        value: IrId,
        value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        let mut value_ty = self.resolve_through_aliases(value_ty);
        let mut target_ty = self.resolve_through_aliases(target_ty);
        let value_storage = self.resolve_storage_type(value_ty);
        let target_storage = self.resolve_storage_type(target_ty);
        let stores_dynamic = |ty| {
            matches!(
                self.type_table.get(ty).map(|ty| &ty.kind),
                Some(TypeKind::Dynamic)
            )
        };
        // Dynamic-backed abstracts use boxes at assignments, arguments and returns.
        if stores_dynamic(value_storage) {
            value_ty = value_storage;
        }
        // An abstract over an Array boxes as that Array.
        if value_storage != value_ty
            && matches!(
                self.type_table.get(value_storage).map(|ty| &ty.kind),
                Some(TypeKind::Array { .. })
            )
        {
            value_ty = value_storage;
        }
        if stores_dynamic(target_storage) && target_storage != target_ty {
            target_ty = target_storage;
            if !self.is_int64_type(value_ty) {
                value_ty = value_storage;
            }
        }
        let out = self.maybe_box_value_inner(value, value_ty, target_ty);
        if let Some(reg) = out {
            let scalar = matches!(
                self.builder.get_register_type(reg),
                Some(
                    IrType::NullInt
                        | IrType::I32
                        | IrType::I64
                        | IrType::F64
                        | IrType::F32
                        | IrType::Bool
                )
            );
            if reg != value && !scalar {
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

        // `Null<Int>` on either side converts by the target's IR type.
        if self.is_null_int_type(target_ty) {
            return self.to_null_int_typed(value, value_ty);
        }
        if self.builder.get_register_type(value) == Some(IrType::NullInt) {
            let target_ir = self.convert_type(target_ty);
            return self.null_int_to_ir(value, &target_ir);
        }

        // Clone TypeKind to avoid borrow checker issues
        let (target_is_dynamic, value_kind_cloned) = {
            let type_table = self.type_table;
            let value_kind = type_table.get(value_ty).map(|t| t.kind.clone());
            let target_is_optional_scalar = match type_table.get(target_ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => {
                    self.optional_inner_is_boxable_primitive(*inner_type)
                        || self.optional_inner_is_type_param(*inner_type)
                }
                _ => false,
            };
            let value_is_scalar = matches!(
                value_kind,
                Some(TypeKind::Int) | Some(TypeKind::Float) | Some(TypeKind::Bool)
            ) || self.is_int64_type(value_ty);
            let target_is_dyn = matches!(
                type_table.get(target_ty).map(|t| &t.kind),
                Some(TypeKind::Dynamic)
            ) || matches!(
                type_table.get(target_ty).map(|t| &t.kind),
                Some(TypeKind::Optional { inner_type })
                    if matches!(
                        type_table.get(self.resolve_storage_type(*inner_type)).map(|t| &t.kind),
                        Some(TypeKind::Dynamic)
                    )
            ) || (target_is_optional_scalar && value_is_scalar);
            (target_is_dyn, value_kind)
        };

        let value_is_dynamic = matches!(&value_kind_cloned, Some(TypeKind::Dynamic));
        let needs_boxing = target_is_dynamic && !value_is_dynamic;

        if !needs_boxing {
            // A `Null<scalar>` slot is a box whatever the typer called the
            // value: `len = b.length - pos` types as Null<Int> yet the
            // register holds the difference itself.
            let target_is_optional_scalar = match self.type_table.get(target_ty).map(|t| &t.kind) {
                Some(TypeKind::Optional { inner_type }) => {
                    self.optional_inner_is_boxable_primitive(*inner_type)
                        || self.optional_inner_is_type_param(*inner_type)
                }
                _ => false,
            };
            if target_is_optional_scalar {
                if let Some(boxed) = self.box_scalar_register(value) {
                    return Some(boxed);
                }
            }
            return Some(value);
        }

        // Int64 keeps its backing class identity across Dynamic.
        if self.is_int64_type(value_ty) {
            let value_mir_type = self.builder.get_register_type(value).unwrap_or(IrType::I64);
            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
            let box_func_id = self.get_or_register_extern_function(
                "haxe_box_int64_ptr",
                vec![value_mir_type],
                ptr_u8.clone(),
            );
            return self
                .builder
                .build_call_direct(box_func_id, vec![value], ptr_u8);
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

            Some(TypeKind::Array { element_type }) => {
                let layout = self.erased_array_slot_layout(*element_type, 0);
                let layout = self.builder.build_const(IrValue::U64(layout))?;
                let ptr = IrType::Ptr(Box::new(IrType::U8));
                let function = self.get_or_register_extern_function(
                    "haxe_box_array_ptr",
                    vec![ptr.clone(), IrType::U64],
                    ptr.clone(),
                );
                self.builder
                    .build_call_direct(function, vec![value, layout], ptr)
            }

            // Reference types (already pointers, just wrap with type_id)
            Some(TypeKind::Class { .. })
            | Some(TypeKind::Enum { .. })
            | Some(TypeKind::Interface { .. })
            | Some(TypeKind::Anonymous { .. }) => {
                // Type tokens lower to integer IDs, not interface instances.
                if self.get_interface_symbol(value_ty).is_some()
                    && matches!(self.builder.get_register_type(value), Some(IrType::Ptr(_)))
                {
                    let ptr = IrType::Ptr(Box::new(IrType::U8));
                    let identity = self.get_or_register_extern_function(
                        "haxe_iface_identity",
                        vec![ptr.clone()],
                        ptr.clone(),
                    );
                    let object =
                        self.builder
                            .build_call_direct(identity, vec![value], ptr.clone())?;
                    let object = self.builder.build_bitcast(object, IrType::I64)?;
                    let tag = self.builder.build_const(IrValue::I32(
                        rayzor_runtime::type_system::ValueTag::Reference as i32,
                    ))?;
                    let boxed = self.get_or_register_extern_function(
                        "haxe_box_typed_ptr",
                        vec![IrType::I64, IrType::I32],
                        ptr.clone(),
                    );
                    return self
                        .builder
                        .build_call_direct(boxed, vec![object, tag], ptr);
                }
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

                let boxed = self.builder.build_call_direct(
                    box_func_id,
                    vec![value, type_id_const],
                    ptr_u8.clone(),
                )?;
                // A tag-only enum's `NULL_INT` is a null Dynamic.
                let tag_only_enum = matches!(
                    &value_kind_cloned,
                    Some(TypeKind::Enum { symbol_id, .. }) if !self.enum_is_boxed(*symbol_id)
                ) && self.builder.get_register_type(value) == Some(IrType::I64);
                if !tag_only_enum {
                    return Some(boxed);
                }
                let null = self.builder.build_const(IrValue::null_int())?;
                let absent = self.builder.build_cmp(CompareOp::Eq, value, null)?;
                let bits = self.builder.build_bitcast(boxed, IrType::I64)?;
                let zero = self.builder.build_const(IrValue::I64(0))?;
                let out = self.builder.build_select(absent, zero, bits)?;
                self.builder.build_bitcast(out, ptr_u8)
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

            // Null<T> over a scalar is already a box. `Null<String>` is the
            // nullable string pointer, boxed as a String (a null stays null).
            // Other references stay raw: an enum or class read back from a
            // Dynamic slot is not unboxed on the way out yet.
            Some(TypeKind::Optional { inner_type }) => {
                let inner_type = *inner_type;
                let inner_is_string = matches!(
                    self.type_table.get(inner_type).map(|t| &t.kind),
                    Some(TypeKind::String)
                );
                if inner_is_string {
                    return self.maybe_box_value_inner(value, inner_type, target_ty);
                }
                Some(value)
            }

            Some(TypeKind::TypeParameter { symbol_id, .. }) => {
                let symbol_id = *symbol_id;
                self.box_type_param_for_dynamic(value, symbol_id)
                    .or(Some(value))
            }

            // Abstract etc. — skip boxing for unsupported types
            _ => {
                debug!(
                    "[BOXING] Unsupported type for boxing: {:?}",
                    value_kind_cloned
                );
                Some(value)
            }
        }
    }

    /// A `T` bound for a Dynamic slot: its raw bits and a placeholder tag the
    /// monomorphizer replaces with the instance's binding. None for a `T` the
    /// current function does not declare -- nothing would fill the tag in.
    pub(crate) fn box_type_param_for_dynamic(
        &mut self,
        value: IrId,
        symbol_id: SymbolId,
    ) -> Option<IrId> {
        let tp_name = self
            .symbol_table
            .get_symbol(symbol_id)
            .and_then(|sym| self.string_interner.get(sym.name))
            .map(|s| s.to_string())?;
        let bound = self
            .builder
            .current_function()
            .is_some_and(|f| f.signature.type_params.iter().any(|tp| tp.name == tp_name));
        if !bound {
            return None;
        }
        if self.boxed_value_regs.contains(&value) {
            return Some(value);
        }
        let v64 = match self.builder.get_register_type(value) {
            Some(IrType::I32) => self.builder.build_cast(value, IrType::I32, IrType::I64)?,
            Some(IrType::Bool) => self.builder.build_cast(value, IrType::Bool, IrType::I64)?,
            Some(IrType::F64) | Some(IrType::Ptr(_)) | Some(IrType::String) => {
                self.builder.build_bitcast(value, IrType::I64)?
            }
            _ => value,
        };
        // An object-constrained `T` that no instance binds still holds a
        // reference, not the Int an unresolved tag stands for.
        let default_tag = if self.type_param_is_object(symbol_id) {
            rayzor_runtime::type_system::ValueTag::Reference as i32
        } else {
            0
        };
        let (tag_reg, metadata) = self.type_param_value_descriptor(tp_name, default_tag)?;
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let box_func = self.get_or_register_extern_function(
            "haxe_box_generic_ptr",
            vec![IrType::I64, IrType::I32, IrType::U64],
            ptr_u8.clone(),
        );
        self.builder
            .build_call_direct(box_func, vec![v64, tag_reg, metadata], ptr_u8)
    }

    pub(crate) fn type_param_value_descriptor(
        &mut self,
        name: String,
        default_tag: i32,
    ) -> Option<(IrId, IrId)> {
        let tag = self.builder.build_const(IrValue::I32(default_tag))?;
        let metadata = self.builder.build_const(IrValue::U64(0))?;
        if let Some(function) = self.builder.current_function_mut() {
            function.type_param_tag_fixups.push((tag, name.clone()));
            function.type_param_metadata_fixups.push((metadata, name));
        }
        Some((tag, metadata))
    }

    /// A type parameter constrained to an object type (`K:{}`, `O:{} & T`,
    /// `T:SomeClass`), whose values are references.
    fn type_param_is_object(&self, symbol_id: SymbolId) -> bool {
        let Some(types) = self.type_table.types_for_symbol(symbol_id) else {
            return false;
        };
        types
            .iter()
            .any(|ty| match self.type_table.get(*ty).map(|t| &t.kind) {
                Some(TypeKind::TypeParameter { constraints, .. }) => {
                    constraints.iter().any(|c| self.is_object_constraint(*c))
                }
                _ => false,
            })
    }

    fn is_object_constraint(&self, ty: TypeId) -> bool {
        match self
            .type_table
            .get(self.resolve_through_aliases(ty))
            .map(|t| &t.kind)
        {
            Some(TypeKind::Anonymous { .. } | TypeKind::Class { .. }) => true,
            Some(TypeKind::Intersection { types }) => {
                types.iter().any(|t| self.is_object_constraint(*t))
            }
            _ => false,
        }
    }

    /// Box a value for extern function calls when the expected type is a pointer but the actual type is a primitive.
    /// This is needed for generic extern classes like Deque<T> where the runtime expects boxed pointers
    /// but the compiler generates primitive values for concrete type parameters like Int.
    ///
    /// Returns the (potentially boxed) value register.
    /// A structural value handed to a runtime `Dynamic` slot travels as its
    /// box: the runtime reads the tag, and a raw anonymous-object handle has
    /// none, so `Std.string({..})` printed null. Only the anonymous kind is
    /// boxed here; the other reference kinds already are, or are read raw.
    pub(crate) fn box_anon_for_dynamic_slot(
        &mut self,
        value: IrId,
        hir_ty: TypeId,
        expected_ty: &IrType,
    ) -> Option<IrId> {
        if !matches!(expected_ty, IrType::Ptr(inner) if matches!(**inner, IrType::U8)) {
            return None;
        }
        let is_anon = matches!(
            self.type_table.get(hir_ty).map(|t| &t.kind),
            Some(crate::tast::TypeKind::Anonymous { .. })
        );
        if !is_anon || self.boxed_value_regs.contains(&value) {
            return None;
        }
        let dynamic_ty = self.type_table.dynamic_type();
        self.maybe_box_value(value, hir_ty, dynamic_ty)
            .filter(|boxed| *boxed != value)
    }

    /// A runtime parameter that reads every argument as a `Dynamic` box: the
    /// JSON printer tells a function from an object or a string by its tag,
    /// and a raw closure or string carries none.
    pub(crate) fn box_for_runtime_dynamic_param(
        &mut self,
        runtime_func: &str,
        param_idx: usize,
        value: IrId,
        argument: &HirExpr,
    ) -> Option<IrId> {
        if !matches!(
            (runtime_func, param_idx),
            ("haxe_json_stringify" | "haxe_reflect_fields", 0)
        ) || self.boxed_value_regs.contains(&value)
            || self.is_class_symbol_expr(argument)
            || self.is_type_token_call(argument)
        {
            return None;
        }
        let dynamic_ty = self.type_table.dynamic_type();
        self.maybe_box_value(value, argument.ty, dynamic_ty)
            .filter(|boxed| *boxed != value)
    }

    /// A `Null<scalar>` (or `Null<T>`) register opened for arithmetic: the
    /// scalar out of the box, as an I32/F64/Bool register. None when the
    /// type is not a nullable scalar or the register is not a pointer.
    pub(crate) fn open_nullable_scalar(&mut self, reg: IrId, ty: TypeId) -> Option<IrId> {
        use crate::tast::TypeKind;
        // A `NullInt` truncates: null reads as 0.
        if self.builder.get_register_type(reg) == Some(IrType::NullInt) {
            return self.builder.build_cast(reg, IrType::NullInt, IrType::I32);
        }
        let inner = match self.type_table.get(ty).map(|t| &t.kind) {
            Some(TypeKind::Optional { inner_type }) => *inner_type,
            _ => return None,
        };
        if !matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_))) {
            return None;
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let (unbox_fn, out) = match self.type_table.get(inner).map(|t| &t.kind) {
            Some(TypeKind::Int) | Some(TypeKind::TypeParameter { .. }) => {
                ("haxe_unbox_int_ptr", IrType::I64)
            }
            Some(TypeKind::Float) => ("haxe_unbox_float_ptr", IrType::F64),
            Some(TypeKind::Bool) => ("haxe_unbox_bool_ptr", IrType::Bool),
            _ => return None,
        };
        let f = self.get_or_register_extern_function(unbox_fn, vec![ptr_u8], out.clone());
        let raw = self.builder.build_call_direct(f, vec![reg], out.clone())?;
        if matches!(out, IrType::I64) {
            self.builder.build_cast(raw, IrType::I64, IrType::I32)
        } else {
            Some(raw)
        }
    }

    /// `Null<T>` over a type parameter: the slot is a box (`convert_type`
    /// gives it the box pointer type), so a scalar written there is boxed.
    pub(crate) fn optional_inner_is_type_param(&self, inner: TypeId) -> bool {
        matches!(
            self.type_table.get(inner).map(|t| &t.kind),
            Some(crate::tast::TypeKind::TypeParameter { .. })
        )
    }

    /// A box for a register that holds a scalar, whatever the typer called
    /// the value. None for anything else -- an I64 may be an erased pointer.
    pub(crate) fn box_scalar_register(&mut self, value: IrId) -> Option<IrId> {
        let (box_fn, param) = match self.builder.get_register_type(value)? {
            IrType::I32 => ("haxe_box_int_ptr", IrType::I64),
            IrType::F64 => ("haxe_box_float_ptr", IrType::F64),
            IrType::Bool => ("haxe_box_bool_ptr", IrType::Bool),
            _ => return None,
        };
        let arg = if param == IrType::I64 {
            self.builder
                .build_cast(value, IrType::I32, IrType::I64)
                .unwrap_or(value)
        } else {
            value
        };
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let func = self.get_or_register_extern_function(box_fn, vec![param], ptr_u8.clone());
        let boxed = self.builder.build_call_direct(func, vec![arg], ptr_u8)?;
        self.boxed_value_regs.insert(boxed);
        Some(boxed)
    }

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
        // `Null<Int>` converts by the parameter's type. A raw u64 slot gets
        // the Int sign-extended, as an Int argument does.
        if *expected_ty == IrType::NullInt {
            return self.to_null_int(value);
        }
        if *actual_ty == IrType::NullInt {
            let target = if *expected_ty == IrType::U64 {
                IrType::I64
            } else {
                expected_ty.clone()
            };
            let converted = self.null_int_to_ir(value, &target)?;
            if *expected_ty == IrType::U64 {
                return self.builder.build_bitcast(converted, IrType::U64);
            }
            return Some(converted);
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

        // A box handed to a scalar parameter (`Int64.ofInt(d)` with `d` a
        // `Null<Int>` local) is unboxed by the parameter's type. Only a
        // register this lowering boxed itself: an erased generic value is
        // raw bits in a pointer-typed register.
        if !expected_is_ptr_u8 {
            if self.boxed_value_regs.contains(&value) {
                let unbox = match expected_ty {
                    IrType::I32 => Some(("haxe_unbox_int_ptr", IrType::I64)),
                    IrType::F64 => Some(("haxe_unbox_float_ptr", IrType::F64)),
                    IrType::Bool => Some(("haxe_unbox_bool_ptr", IrType::Bool)),
                    _ => None,
                };
                if let Some((name, ret)) = unbox {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let func =
                        self.get_or_register_extern_function(name, vec![ptr_u8], ret.clone());
                    let raw = self
                        .builder
                        .build_call_direct(func, vec![value], ret.clone())?;
                    if *expected_ty == IrType::I32 {
                        return self.builder.build_cast(raw, IrType::I64, IrType::I32);
                    }
                    return Some(raw);
                }
            }
            // An Int index into an i64 slot is widened; left as i32 the upper
            // half of the register is whatever the caller last held there.
            if actual_ty.is_integer() && expected_ty.is_integer() {
                return self
                    .builder
                    .build_cast(value, actual_ty.clone(), expected_ty.clone());
            }
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
        // `Null<Int>`: the raw Int, or `NULL_INT` when the probe says absent.
        if self.optional_inner_is_null_int(inner) {
            let value = self.to_null_int(raw)?;
            let Some((exists_fn, probe_args)) = probe else {
                return Some(value);
            };
            let present = self
                .builder
                .build_call_direct(exists_fn, probe_args, IrType::Bool)?;
            let null = self.builder.build_null_int()?;
            return self.builder.build_select(present, value, null);
        }
        if !self.optional_inner_is_boxable_primitive(inner) {
            return None;
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        // The box's kind comes from the DECLARED inner type: a container erases
        // its value into a u64 slot and hands a Bool back as 0/1 in an i64, so
        // the register alone would say Int and the box would print `1`.
        let declared_bool = matches!(self.convert_type(inner), IrType::Bool);
        let (box_name, arg_ty, arg) = if declared_bool {
            let b = match self.builder.get_register_type(raw)? {
                IrType::Bool => raw,
                t @ (IrType::I32 | IrType::I64) => self.builder.build_cast(raw, t, IrType::Bool)?,
                _ => return None,
            };
            ("haxe_box_bool_ptr", IrType::Bool, b)
        } else {
            match self.builder.get_register_type(raw)? {
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
            }
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

    /// A `Null<scalar>` box bound for an erased `T` formal hands over the
    /// scalar's bits, as a plain Int passed there does. `None` when this is
    /// not that case.
    pub(crate) fn unbox_optional_for_erased_formal(
        &mut self,
        reg: IrId,
        arg_ty: TypeId,
        formal_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;
        let formal_is_param = matches!(
            self.type_table.get(formal_ty).map(|t| &t.kind),
            Some(TypeKind::TypeParameter { .. })
        );
        // A `NullInt` hands over its Int sign-extended; null becomes 0.
        if formal_is_param && self.builder.get_register_type(reg) == Some(IrType::NullInt) {
            return self.null_int_to_ir(reg, &IrType::I64);
        }
        if !formal_is_param || !matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_)))
        {
            return None;
        }
        let inner = match self.type_table.get(arg_ty).map(|t| &t.kind) {
            Some(TypeKind::Optional { inner_type }) => *inner_type,
            _ => return None,
        };
        if !self.optional_inner_is_boxable_primitive(inner) {
            return None;
        }
        let scalar = self.maybe_unbox_optional(reg, arg_ty, inner)?;
        self.coerce_to_i64(scalar, inner)
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

        // `Null<Int>`: the value as a `NullInt` (a null literal is `NULL_INT`).
        if self.optional_inner_is_null_int(inner_type) {
            if self.builder.get_register_type(value) == Some(IrType::NullInt) {
                return None;
            }
            return self.to_null_int_typed(value, value_ty);
        }

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
                // A `Null<Int>` keeps its null: it becomes the box a `Null<T>` holds.
                Some(IrType::NullInt) => return self.null_int_to_dynamic(value),
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
            let (tag_reg, metadata) = self.type_param_value_descriptor(tp_name, 0)?;

            let box_func = self.get_or_register_extern_function(
                "haxe_box_generic_ptr",
                vec![IrType::I64, IrType::I32, IrType::U64],
                ptr_u8.clone(),
            );
            return self
                .builder
                .build_call_direct(box_func, vec![v64, tag_reg, metadata], ptr_u8);
        }

        // A `Null<E>` over a tag-only enum carries the discriminant raw, null
        // as 0; the enum's own null is `NULL_INT`.
        if self.is_tag_only_enum_type(inner_type)
            && self.builder.get_register_type(value) == Some(IrType::I64)
        {
            let null = self.builder.build_const(IrValue::null_int())?;
            let absent = self.builder.build_cmp(CompareOp::Eq, value, null)?;
            let zero = self.builder.build_const(IrValue::I64(0))?;
            return self.builder.build_select(absent, zero, value);
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
            Some(IrType::NullInt) => return self.null_int_to_dynamic(value),
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
        mut value: IrId,
        value_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;

        // `Null<Int>` on either side converts by the target's IR type.
        if self.is_null_int_type(target_ty) {
            return self.to_null_int_typed(value, value_ty);
        }
        if self.builder.get_register_type(value) == Some(IrType::NullInt) {
            let target_ir = self.convert_type(target_ty);
            return self.null_int_to_ir(value, &target_ir);
        }

        let storage = self.resolve_storage_type(value_ty);
        let value_ty = if storage != value_ty
            && matches!(
                self.type_table.get(storage).map(|ty| &ty.kind),
                Some(TypeKind::Dynamic)
            ) {
            // Erased array slots carry a Dynamic-backed abstract's box as bits.
            if matches!(self.builder.get_register_type(value), Some(IrType::I64)) {
                value = self
                    .builder
                    .build_bitcast(value, IrType::Ptr(Box::new(IrType::U8)))?;
            }
            storage
        } else {
            value_ty
        };
        let storage = self.resolve_storage_type(target_ty);
        let type_handle = matches!(
            self.type_table.get(target_ty).map(|ty| &ty.kind),
            Some(TypeKind::Abstract { symbol_id, .. })
                if self.symbol_table.get_symbol(*symbol_id).is_some_and(|symbol| {
                    crate::tast::core::is_haxe_std_type(
                        symbol, self.string_interner, &["Class", "Enum"],
                    )
                })
        );
        // Abstract references use their underlying object at typed boundaries.
        let target_ty = if self.is_int64_type(target_ty) || type_handle {
            target_ty
        } else {
            storage
        };

        // Clone TypeKind to avoid borrow checker issues
        let (value_is_dynamic, target_kind_cloned, value_is_optional_scalar) = {
            let type_table = self.type_table;
            let value_kind = type_table.get(value_ty).map(|t| &t.kind);
            let value_is_optional_scalar = match value_kind {
                Some(TypeKind::Optional { inner_type }) => {
                    self.optional_inner_is_boxable_primitive(*inner_type)
                }
                _ => false,
            };
            // A typedef of a structure unboxes as the structure; a
            // `Null<abstract over a scalar>` unboxes as that scalar.
            let mut target_ty = target_ty;
            for _ in 0..4 {
                match type_table.get(target_ty).map(|t| &t.kind) {
                    Some(TypeKind::TypeAlias { target_type, .. }) => target_ty = *target_type,
                    Some(TypeKind::Abstract {
                        underlying: Some(u),
                        ..
                    }) if value_is_optional_scalar => target_ty = *u,
                    _ => break,
                }
            }
            let target_kind = type_table.get(target_ty).map(|t| t.kind.clone());
            let target_is_scalar = matches!(
                target_kind,
                Some(TypeKind::Int) | Some(TypeKind::Float) | Some(TypeKind::Bool)
            ) || self.is_int64_type(target_ty);

            // `Null<Dynamic>` is Dynamic.
            let value_is_null_dynamic = match value_kind {
                Some(TypeKind::Optional { inner_type }) => matches!(
                    type_table.get(*inner_type).map(|t| &t.kind),
                    Some(TypeKind::Dynamic)
                ),
                _ => false,
            };
            let value_is_dyn = matches!(value_kind, Some(TypeKind::Dynamic))
                || value_is_null_dynamic
                || (value_is_optional_scalar && target_is_scalar);
            (value_is_dyn, target_kind, value_is_optional_scalar)
        };

        let target_is_dynamic = matches!(&target_kind_cloned, Some(TypeKind::Dynamic));
        let needs_unboxing = value_is_dynamic && !target_is_dynamic;

        if !needs_unboxing {
            return Some(value);
        }

        if matches!(&target_kind_cloned, Some(TypeKind::Function { .. })) {
            let ptr = IrType::Ptr(Box::new(IrType::U8));
            let unbox = self.get_or_register_extern_function(
                "haxe_unbox_if_tag",
                vec![ptr.clone(), IrType::U32],
                ptr.clone(),
            );
            let tag = self.builder.build_const(IrValue::U32(u32::MAX - 1))?;
            let closure = self
                .builder
                .build_call_direct(unbox, vec![value, tag], ptr.clone())?;
            let adapter = self.ensure_dynamic_typed_adapter(target_ty)?;
            let adapter = self.builder.build_function_ref(adapter)?;
            let view = self.get_or_register_extern_function(
                "haxe_closure_typed_view",
                vec![ptr.clone(), ptr.clone()],
                ptr.clone(),
            );
            let result = self
                .builder
                .build_call_direct(view, vec![closure, adapter], ptr)?;
            return self
                .builder
                .build_bitcast(result, self.convert_type(target_ty));
        }

        // `Iterator<T>`/`Iterable<T>` are structural, and a value crossing into
        // one is a raw collection or iterator, never a boxed DynamicValue.
        // Unboxing would read its first word as the box's payload, and the fresh
        // register it hands on carries no class hint, so the boundary that turns
        // the value into an iteration handle can no longer classify it.
        if self.iter_protocol_of(target_ty).is_some() {
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
                && matches!(&target_kind_cloned, Some(TypeKind::Interface { .. }))
            {
                debug!(
                    "[UNBOXING] Skipping unbox — value MIR type {:?} is already a pointer for reference target {:?}",
                    mir_ty, target_kind_cloned
                );
                return Some(value);
            }
            // An array, string or anonymous object begins with a data pointer,
            // never with its own tag, so the box is told from the raw object
            // by the tag alone and a raw one passes through. Producers still
            // hand some of these back raw under a Dynamic type.
            let tagged = match &target_kind_cloned {
                Some(TypeKind::Array { .. }) => Some(7u32),
                Some(TypeKind::String) => Some(5u32),
                Some(TypeKind::Anonymous { .. }) => Some(6u32),
                _ => None,
            };
            if let (Some(tag), true) = (tagged, matches!(mir_ty, IrType::Ptr(_) | IrType::String)) {
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox = self.get_or_register_extern_function(
                    "haxe_unbox_if_tag",
                    vec![ptr_u8.clone(), IrType::U32],
                    ptr_u8.clone(),
                );
                let tag_reg = self.builder.build_const(IrValue::U32(tag))?;
                let as_ptr = if matches!(mir_ty, IrType::String) {
                    self.builder.build_bitcast(value, ptr_u8.clone())?
                } else {
                    value
                };
                let raw = self
                    .builder
                    .build_call_direct(unbox, vec![as_ptr, tag_reg], ptr_u8)?;
                let to = self.convert_type(target_ty);
                return self.builder.build_bitcast(raw, to);
            }
        }

        // Primitive MIR registers already passed through above. A pointer
        // crossing from Dynamic or Null<Int64> must yield the full native i64.
        if self.is_int64_type(target_ty) {
            let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
            let unbox_func_id = self.get_or_register_extern_function(
                "haxe_unbox_int_ptr",
                vec![ptr_u8],
                IrType::I64,
            );
            return self
                .builder
                .build_call_direct(unbox_func_id, vec![value], IrType::I64);
        }

        match &target_kind_cloned {
            Some(TypeKind::TypeParameter { .. }) => self.unbox_generic_value(value, target_ty),
            // Value types (need to extract from heap)
            Some(TypeKind::Int) => {
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                // A Dynamic-typed field can already carry an erased integer slot.
                let unbox = if value_is_optional_scalar {
                    "haxe_unbox_int_ptr"
                } else {
                    "haxe_coerce_dynamic_to_int"
                };
                let unbox_func_id =
                    self.get_or_register_extern_function(unbox, vec![ptr_u8], IrType::I64);
                let unboxed =
                    self.builder
                        .build_call_direct(unbox_func_id, vec![value], IrType::I64)?;
                self.builder.build_cast(unboxed, IrType::I64, IrType::I32)
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
            Some(TypeKind::Enum { symbol_id, .. }) => {
                let actual_type = self.builder.get_register_type(value);
                if matches!(actual_type, Some(IrType::I64) | Some(IrType::I32)) {
                    debug!(
                        "[UNBOXING] Skipping unbox for enum - value is already i64 discriminant"
                    );
                    return Some(value);
                }
                // A tag-only enum reads a null Dynamic as `NULL_INT`.
                let tag_only = !self.enum_is_boxed(*symbol_id);
                if tag_only && self.builder.is_null_const(value) {
                    return self.builder.build_const(IrValue::null_int());
                }
                debug!("[UNBOXING] Auto-unboxing Dynamic to Enum using unbox_reference");

                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox_func_id = self.get_or_register_extern_function(
                    "haxe_unbox_reference_ptr",
                    vec![ptr_u8.clone()],
                    ptr_u8.clone(),
                );
                let unboxed = self
                    .builder
                    .build_call_direct(unbox_func_id, vec![value], ptr_u8)?;
                if !tag_only {
                    return Some(unboxed);
                }
                let bits = self.builder.build_bitcast(unboxed, IrType::I64)?;
                let source = self.builder.build_bitcast(value, IrType::I64)?;
                let zero = self.builder.build_const(IrValue::I64(0))?;
                let absent = self.builder.build_cmp(CompareOp::Eq, source, zero)?;
                let null = self.builder.build_const(IrValue::null_int())?;
                self.builder.build_select(absent, null, bits)
            }

            // A class or enum used as a value is its type id; under Dynamic it
            // may travel bare, in an Int box or in a box of its own.
            Some(TypeKind::Abstract { symbol_id, .. })
                if matches!(actual_mir_type, Some(IrType::Ptr(_)))
                    && self.symbol_table.get_symbol(*symbol_id).is_some_and(|s| {
                        crate::tast::core::is_haxe_std_type(
                            s,
                            self.string_interner,
                            &["Class", "Enum"],
                        )
                    }) =>
            {
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let unbox = self.get_or_register_extern_function(
                    "haxe_unbox_type_handle",
                    vec![ptr_u8],
                    IrType::I64,
                );
                let id = self
                    .builder
                    .build_call_direct(unbox, vec![value], IrType::I64)?;
                match self.convert_type(target_ty) {
                    IrType::I64 => Some(id),
                    to => self.builder.build_cast(id, IrType::I64, to),
                }
            }

            // Abstract types (Ptr<T>, Ref<T>, Box<T>, Usize) are zero-cost over
            // Int: plain i64 values, never heap-boxed.
            Some(TypeKind::Abstract { .. }) => {
                debug!("[UNBOXING] Skipping unbox for Abstract type (zero-cost over Int)");
                Some(value)
            }

            // Reference types (just extract the pointer)
            Some(TypeKind::Class { .. }) => {
                // Only integer-backed wrappers are already their erased payload.
                if self.convert_type(target_ty).is_integer() {
                    return Some(value);
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
            | Some(TypeKind::GenericInstance { .. })
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
    /// Whether a callee's DECLARED return type is (or wraps) a type parameter --
    /// the only returns that arrive erased. `isEmpty(): Bool` on the same
    /// generic receiver returns a concrete Bool and must be left alone.
    pub(crate) fn callee_returns_type_param(&self, callee: SymbolId) -> bool {
        use crate::tast::TypeKind;
        let Some(fn_ty) = self.symbol_table.get_symbol(callee).map(|s| s.type_id) else {
            return false;
        };
        let tt = self.type_table;
        let Some(TypeKind::Function { return_type, .. }) = tt.get(fn_ty).map(|i| &i.kind) else {
            return false;
        };
        let mut cur = *return_type;
        for _ in 0..4 {
            match tt.get(cur).map(|i| &i.kind) {
                Some(TypeKind::TypeParameter { .. }) => return true,
                Some(TypeKind::Optional { inner_type }) => cur = *inner_type,
                Some(TypeKind::TypeAlias { target_type, .. }) => cur = *target_type,
                _ => return false,
            }
        }
        false
    }

    /// Convert a condition to Bool, opening nullable boxes and abstract casts.
    pub(crate) fn truth_of(&mut self, reg: IrId, hir_ty: TypeId) -> Option<IrId> {
        use crate::tast::TypeKind;
        let mut ty = hir_ty;
        for _ in 0..4 {
            match self.type_table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                _ => break,
            }
        }
        if let Some(converted) =
            self.maybe_abstract_to_convert(reg, ty, self.type_table.bool_type())
        {
            return Some(converted);
        }
        if self.builder.get_register_type(reg) == Some(IrType::NullInt) {
            return self.null_int_to_ir(reg, &IrType::Bool);
        }
        let unbox_fn = match self.type_table.get(ty).map(|t| &t.kind) {
            Some(TypeKind::Optional { inner_type })
                if matches!(
                    self.type_table.get(*inner_type).map(|t| &t.kind),
                    Some(TypeKind::Bool)
                ) =>
            {
                "haxe_unbox_bool_ptr"
            }
            // A Dynamic condition is a box (a raw value passes through).
            Some(TypeKind::Dynamic) => "haxe_dynamic_truthy",
            _ => return Some(reg),
        };
        if !matches!(self.builder.get_register_type(reg), Some(IrType::Ptr(_))) {
            return Some(reg);
        }
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let unbox = self.get_or_register_extern_function(unbox_fn, vec![ptr_u8], IrType::Bool);
        self.builder
            .build_call_direct(unbox, vec![reg], IrType::Bool)
    }

    /// Unbox the return of an erased generic method, resolving T from the
    /// receiver's type argument.
    ///
    /// A generic class is compiled once, so `first(): Null<T>` cannot know T:
    /// some bodies box the stored bits with haxe_box_int_ptr, others hand the
    /// bits back raw, and the caller cannot tell which from the type. So the
    /// RUNTIME decides: `haxe_unbox_erased_return` unwraps either box shape
    /// to its bits and passes anything else through unchanged. The caller's
    /// knowledge of T is used only to shape the result register.
    ///
    /// With one type parameter the receiver's argument is T. With several,
    /// only the call's own type (`Null<V>` bound at the call site) says which
    /// one the return is, and a call type still naming a parameter leaves the
    /// bits alone.
    pub(crate) fn unbox_erased_generic_return(
        &mut self,
        call_result: IrId,
        actual_return_type: Option<&IrType>,
        receiver_ty: TypeId,
        call_ty: TypeId,
    ) -> Option<IrId> {
        use crate::tast::TypeKind;
        // An import's signature is not recorded, so its return type arrives as
        // None: treat it as erased and let the runtime check decide.
        let erased = actual_return_type.map_or(true, |t| {
            matches!(t, IrType::Ptr(inner) if matches!(inner.as_ref(), IrType::U8 | IrType::Void))
        });
        if !erased {
            return Some(call_result);
        }
        // A `Null<Int>` result: the runtime tells a box, a raw Int and null
        // apart, so null stays null.
        if self.is_null_int_type(call_ty) {
            // The register may carry the caller's NullInt type while the erased
            // callee returned a box, a raw Int or a null pointer.
            let as_ptr = match self.builder.get_register_type(call_result) {
                Some(IrType::I64 | IrType::U64 | IrType::NullInt) => self
                    .builder
                    .build_bitcast(call_result, IrType::Ptr(Box::new(IrType::U8)))?,
                _ => call_result,
            };
            return self.to_null_int(as_ptr);
        }
        let concrete_call = {
            let mut ty = call_ty;
            for _ in 0..8 {
                match self.type_table.get(ty).map(|ty| &ty.kind) {
                    Some(TypeKind::Optional { inner_type }) => ty = *inner_type,
                    Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                    _ => break,
                }
            }
            match self.type_table.get(ty).map(|ty| &ty.kind) {
                None
                | Some(
                    TypeKind::TypeParameter { .. }
                    | TypeKind::Dynamic
                    | TypeKind::Unknown
                    | TypeKind::Placeholder { .. },
                ) => None,
                Some(_) => Some(ty),
            }
        };
        let resolved = concrete_call.or_else(|| {
            let type_table = self.type_table;
            match type_table.get(receiver_ty).map(|ti| &ti.kind) {
                Some(TypeKind::Class { type_args, .. })
                | Some(TypeKind::GenericInstance { type_args, .. })
                | Some(TypeKind::Abstract { type_args, .. })
                    if type_args.len() == 1 =>
                {
                    Some(type_args[0])
                }
                Some(TypeKind::Class { type_args, .. })
                | Some(TypeKind::GenericInstance { type_args, .. })
                | Some(TypeKind::Abstract { type_args, .. })
                    if type_args.len() > 1 =>
                {
                    let mut ty = call_ty;
                    for _ in 0..4 {
                        match type_table.get(ty).map(|ti| &ti.kind) {
                            Some(TypeKind::Optional { inner_type }) => ty = *inner_type,
                            Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                            _ => break,
                        }
                    }
                    match type_table.get(ty).map(|ti| &ti.kind) {
                        None
                        | Some(
                            TypeKind::TypeParameter { .. }
                            | TypeKind::Unknown
                            | TypeKind::Placeholder { .. },
                        ) => None,
                        Some(_) => Some(ty),
                    }
                }
                _ => None,
            }
        });
        let Some(resolved) = resolved else {
            return Some(call_result);
        };
        // A Dynamic caller consumes the box itself; concrete callers consume
        // its payload with the caller's type, including the enum marker.
        if matches!(
            self.type_table
                .get(self.resolve_storage_type(resolved))
                .map(|ty| &ty.kind),
            Some(TypeKind::Dynamic)
        ) {
            return Some(call_result);
        }
        let target = if self.resolve_enum_symbol(resolved).is_some() {
            self.convert_type_arg(resolved)
        } else {
            self.convert_type(resolved)
        };
        let shape_as_scalar = matches!(
            target,
            IrType::I32 | IrType::I64 | IrType::F32 | IrType::F64 | IrType::Bool
        );
        let shape_as_pointer = matches!(target, IrType::String | IrType::Ptr(_));
        if !shape_as_scalar && !shape_as_pointer {
            return Some(call_result);
        }
        let bits = self.unbox_generic_value(call_result, resolved)?;
        match target {
            IrType::I64 => Some(bits),
            IrType::I32 | IrType::Bool => self.builder.build_cast(bits, IrType::I64, target),
            IrType::F64 => self.builder.build_bitcast(bits, IrType::F64),
            IrType::F32 => {
                let f = self.builder.build_bitcast(bits, IrType::F64)?;
                self.builder.build_cast(f, IrType::F64, IrType::F32)
            }
            // A String register is not a plain pointer to the builder; raw
            // bits become one through a cast, as the vtable thunks do.
            IrType::String => self.builder.build_cast(bits, IrType::I64, IrType::String),
            other => self.builder.build_bitcast(bits, other),
        }
    }

    /// Resolve the payload representation when an enclosing generic binds T.
    pub(crate) fn unbox_generic_value(&mut self, value: IrId, ty: TypeId) -> Option<IrId> {
        let parameter = self.resolve_type_param_name(self.resolve_through_aliases(ty));
        let name = parameter
            .clone()
            .filter(|name| self.current_type_param_names().contains(name));
        let default_tag = if parameter.is_some() {
            rayzor_runtime::type_system::ValueTag::Dynamic as i32
        } else {
            self.convert_type_arg(ty).value_tag() as i32
        };
        let tag = self.builder.build_const(IrValue::I32(default_tag))?;
        if let Some(name) = name
            && let Some(function) = self.builder.current_function_mut()
        {
            function.type_param_tag_fixups.push((tag, name));
        }
        let ptr = IrType::Ptr(Box::new(IrType::U8));
        let value = self.coerce_reg_to(value, &ptr)?;
        let unbox = self.get_or_register_extern_function(
            "haxe_unbox_generic_return",
            vec![ptr, IrType::I32],
            IrType::I64,
        );
        self.builder
            .build_call_direct(unbox, vec![value, tag], IrType::I64)
    }

    pub(crate) fn maybe_unbox_for_extern_return(
        &mut self,
        value: IrId,
        actual_ty: &IrType,
        expected_ty: &IrType,
    ) -> Option<IrId> {
        // `Null<Int>`: an 8-byte slot holds a box, null or a raw Int, read
        // through the runtime; a 32-bit Int sign-extends. A `NullInt` result
        // read as a plain type converts by that type.
        if *expected_ty == IrType::NullInt {
            let value = match self.builder.get_register_type(value) {
                Some(IrType::I64 | IrType::U64) => self
                    .builder
                    .build_bitcast(value, IrType::Ptr(Box::new(IrType::U8)))?,
                _ => value,
            };
            return self.to_null_int(value);
        }
        if *actual_ty == IrType::NullInt {
            return self.null_int_to_ir(value, expected_ty);
        }

        let actual_is_ptr_u8 =
            matches!(actual_ty, IrType::Ptr(inner) if matches!(**inner, IrType::U8 | IrType::Void));

        if !actual_is_ptr_u8 {
            // Runtime integer predicates become Haxe booleans at the call boundary.
            if actual_ty.is_integer() && *expected_ty == IrType::Bool {
                let zero = self.builder.build_const(actual_ty.default_value())?;
                return self.builder.build_cmp(CompareOp::Ne, value, zero);
            }
            // Truncate I64 → I32 when extern returns i64 but Haxe type is Int (I32).
            if matches!(actual_ty, IrType::I64) && matches!(expected_ty, IrType::I32) {
                return self.builder.build_cast(value, IrType::I64, IrType::I32);
            }
            // A raw 8-byte slot (Array.pop/shift) holding a Float's bits or a
            // String.
            if matches!(actual_ty, IrType::I64) {
                match expected_ty {
                    IrType::F64 => return self.builder.build_bitcast(value, IrType::F64),
                    IrType::String => {
                        return self.builder.build_cast(value, IrType::I64, IrType::String);
                    }
                    _ => {}
                }
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

        // `Null<Int>` reads its Int; null is 0.
        if self.optional_inner_is_null_int(inner_type)
            || self.builder.get_register_type(value) == Some(IrType::NullInt)
        {
            if !matches!(
                target_ir,
                IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32
            ) {
                return None;
            }
            let null_int = self.to_null_int(value)?;
            return self.null_int_to_ir(null_int, &target_ir);
        }

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
        value_ty: TypeId,
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
        if val_ir == Some(IrType::NullInt) {
            return self.null_int_to_ir(value, &target_ir);
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        // A non-`*u8` pointer register (e.g. `*void`) is NOT a boxed
        // `DynamicValue*` — genuine boxes are always `*u8`. It is a
        // switch-as-expression temp holding a RAW primitive payload read out of
        // an enum's `{tag, payload}` struct, surfaced as untyped `*void`
        // because the variant's type was lost. Unboxing it would dereference
        // the payload as a pointer, so recover the raw bits and cast instead.
        let storage = self.resolve_storage_type(value_ty);
        let mut is_known_box = self.boxed_value_regs.contains(&value)
            || (storage != value_ty
                && matches!(
                    self.type_table.get(storage).map(|ty| &ty.kind),
                    Some(TypeKind::Dynamic)
                ));
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
            IrType::NullInt => return self.builder.build_null_int(),
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
        // Int64 is the native i64, and null is not an i64 value.
        if self.is_int64_type(inner_type) {
            return true;
        }
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
                    let handle = self.symbol_table.get_symbol(symbol_id).is_some_and(|sym| {
                        crate::tast::core::is_handle_abstract(sym, self.string_interner)
                    });
                    if handle {
                        return false;
                    }
                    // An instantiation (`EnumFlags<E>`) carries no underlying.
                    match underlying
                        .or_else(|| self.type_table.resolve_abstract_underlying(symbol_id))
                    {
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

    /// Check if a TypeId is a boxed Optional{primitive} (Null<Float>, Null<Bool>;
    /// not the unboxed `Null<Int>`).
    pub(crate) fn is_optional_primitive(&self, type_id: TypeId) -> bool {
        use crate::tast::TypeKind;
        let type_table = self.type_table;
        if let Some(TypeKind::Optional { inner_type }) = type_table.get(type_id).map(|t| &t.kind) {
            if self.optional_inner_is_null_int(*inner_type) {
                return false;
            }
            let inner_ir = self.convert_type(*inner_type);
            matches!(
                inner_ir,
                IrType::I32 | IrType::I64 | IrType::F64 | IrType::F32 | IrType::Bool
            )
        } else {
            false
        }
    }
    /// `Null<T>` whose `T` resolves to `Int` (through aliases and non-handle
    /// abstracts: `UInt`, enum abstracts) is held unboxed as `IrType::NullInt`.
    /// Int64, Char, Float, Bool and type parameters stay boxed.
    pub(crate) fn optional_inner_is_null_int(&self, inner_type: TypeId) -> bool {
        use crate::tast::TypeKind;
        if crate::debug_flags::null_int_boxed() || self.is_int64_type(inner_type) {
            return false;
        }
        let mut ty = inner_type;
        for _ in 0..8 {
            match self.type_table.get(ty).map(|t| &t.kind) {
                Some(TypeKind::Int) => return true,
                Some(TypeKind::TypeAlias { target_type, .. }) => ty = *target_type,
                Some(TypeKind::Abstract {
                    underlying,
                    symbol_id,
                    ..
                }) => {
                    let (underlying, symbol_id) = (*underlying, *symbol_id);
                    let handle = self.symbol_table.get_symbol(symbol_id).is_some_and(|sym| {
                        crate::tast::core::is_handle_abstract(sym, self.string_interner)
                    });
                    if handle {
                        return false;
                    }
                    match underlying
                        .or_else(|| self.type_table.resolve_abstract_underlying(symbol_id))
                    {
                        Some(u) => ty = u,
                        None => return false,
                    }
                }
                _ => return false,
            }
        }
        false
    }

    /// Whether `reg` holds a scalar (a `NullInt` included), never a box.
    pub(crate) fn is_scalar_register(&self, reg: IrId) -> bool {
        matches!(
            self.builder.get_register_type(reg),
            Some(
                IrType::NullInt
                    | IrType::I32
                    | IrType::I64
                    | IrType::F64
                    | IrType::F32
                    | IrType::Bool
            )
        )
    }

    /// Whether an array type's elements are `Null<Int>`. Its slots hold the
    /// `NullInt` itself, so a non-null element shares `Array<Int>`'s layout.
    pub(crate) fn array_of_null_int(&self, array_ty: TypeId) -> bool {
        match self
            .type_table
            .get(self.resolve_through_aliases(array_ty))
            .map(|t| &t.kind)
        {
            Some(TypeKind::Array { element_type }) => self.is_null_int_type(*element_type),
            _ => false,
        }
    }

    /// A value of HIR type `ty` as the bits an `Array<Null<Int>>` slot holds.
    pub(crate) fn null_int_slot_bits(&mut self, reg: IrId, ty: TypeId) -> Option<IrId> {
        let value = self.to_null_int_typed(reg, ty)?;
        self.builder.build_bitcast(value, IrType::I64)
    }

    /// Whether a type is held as `IrType::NullInt`. Only an Optional, an
    /// alias or an abstract can be.
    pub(crate) fn is_null_int_type(&self, ty: TypeId) -> bool {
        matches!(
            self.type_table.get(ty).map(|t| &t.kind),
            Some(
                TypeKind::Optional { .. } | TypeKind::TypeAlias { .. } | TypeKind::Abstract { .. }
            )
        ) && self.convert_type(ty) == IrType::NullInt
    }

    /// `reg` as a `NullInt`, decided by its register: an Int sign-extends, a
    /// pointer is a Dynamic (a box, a raw scalar, or null), and an erased i64
    /// slot keeps its low 32 bits as an Int bound for `T` does.
    pub(crate) fn to_null_int(&mut self, reg: IrId) -> Option<IrId> {
        let ty = self.builder.get_register_type(reg);
        match ty {
            None | Some(IrType::NullInt) => Some(reg),
            Some(
                t @ (IrType::I8
                | IrType::I16
                | IrType::I32
                | IrType::U8
                | IrType::U16
                | IrType::U32
                | IrType::Bool),
            ) => self.builder.build_cast(reg, t, IrType::NullInt),
            Some(t @ (IrType::I64 | IrType::U64)) => {
                let int = self.builder.build_cast(reg, t, IrType::I32)?;
                self.builder.build_cast(int, IrType::I32, IrType::NullInt)
            }
            Some(t @ (IrType::F64 | IrType::F32)) => {
                let int = self.builder.build_cast(reg, t, IrType::I32)?;
                self.builder.build_cast(int, IrType::I32, IrType::NullInt)
            }
            Some(IrType::Ptr(_)) if self.builder.is_null_const(reg) => {
                self.builder.build_null_int()
            }
            Some(t @ (IrType::Ptr(_) | IrType::Any | IrType::Dynamic | IrType::String)) => {
                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                let arg = if matches!(t, IrType::Ptr(_)) {
                    reg
                } else {
                    self.builder.build_bitcast(reg, ptr_u8.clone())?
                };
                let from = self.get_or_register_extern_function(
                    "haxe_null_int_from_dynamic",
                    vec![ptr_u8],
                    IrType::NullInt,
                );
                self.builder
                    .build_call_direct(from, vec![arg], IrType::NullInt)
            }
            Some(_) => Some(reg),
        }
    }

    /// `to_null_int` for a value of HIR type `value_ty`: an i64 register typed
    /// Dynamic carries a box (or null) as bits, not an erased Int.
    pub(crate) fn to_null_int_typed(&mut self, reg: IrId, value_ty: TypeId) -> Option<IrId> {
        use crate::tast::TypeKind;
        let dynamic = matches!(
            self.type_table.get(value_ty).map(|t| &t.kind),
            Some(TypeKind::Dynamic | TypeKind::Unknown) | None
        );
        if dynamic
            && matches!(
                self.builder.get_register_type(reg),
                Some(IrType::I64 | IrType::U64)
            )
        {
            let ptr = self
                .builder
                .build_bitcast(reg, IrType::Ptr(Box::new(IrType::U8)))?;
            return self.to_null_int(ptr);
        }
        self.to_null_int(reg)
    }

    /// A `NullInt` as a Dynamic: null for null, else an Int box.
    pub(crate) fn null_int_to_dynamic(&mut self, reg: IrId) -> Option<IrId> {
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let to = self.get_or_register_extern_function(
            "haxe_null_int_to_dynamic",
            vec![IrType::NullInt],
            ptr_u8.clone(),
        );
        let boxed = self.builder.build_call_direct(to, vec![reg], ptr_u8)?;
        self.boxed_value_regs.insert(boxed);
        Some(boxed)
    }

    /// `reg == null` for a `NullInt`.
    pub(crate) fn null_int_is_null(&mut self, reg: IrId) -> Option<IrId> {
        let null = self.builder.build_null_int()?;
        self.builder.build_cmp(CompareOp::Eq, reg, null)
    }

    /// A `NullInt` handed to a slot of IR type `target`: an Int or float
    /// reads null as 0, an erased i64 slot gets the Int sign-extended (null
    /// becomes 0 there, as before), and a pointer slot gets a Dynamic.
    /// Other targets take the register unchanged.
    pub(crate) fn null_int_to_ir(&mut self, reg: IrId, target: &IrType) -> Option<IrId> {
        let reg = self.to_null_int(reg)?;
        match target {
            IrType::NullInt => Some(reg),
            IrType::I8 | IrType::I16 | IrType::I32 | IrType::U8 | IrType::U16 | IrType::U32 => self
                .builder
                .build_cast(reg, IrType::NullInt, target.clone()),
            IrType::I64 | IrType::U64 | IrType::TypeVar(_) => {
                let int = self.builder.build_cast(reg, IrType::NullInt, IrType::I32)?;
                let wide = if *target == IrType::U64 {
                    IrType::U64
                } else {
                    IrType::I64
                };
                self.builder.build_cast(int, IrType::I32, wide)
            }
            IrType::F64 | IrType::F32 => {
                let int = self.builder.build_cast(reg, IrType::NullInt, IrType::I32)?;
                self.builder.build_cast(int, IrType::I32, target.clone())
            }
            IrType::Bool => {
                let int = self.builder.build_cast(reg, IrType::NullInt, IrType::I32)?;
                let zero = self.builder.build_const(IrValue::I32(0))?;
                self.builder.build_cmp(CompareOp::Ne, int, zero)
            }
            IrType::Ptr(_) | IrType::Any | IrType::Dynamic => {
                let boxed = self.null_int_to_dynamic(reg)?;
                if matches!(target, IrType::Ptr(_)) {
                    Some(boxed)
                } else {
                    self.builder.build_bitcast(boxed, target.clone())
                }
            }
            _ => Some(reg),
        }
    }

    /// Reconcile `reg` with a slot of IR type `target` when either side is a
    /// `NullInt`; None when neither is.
    pub(crate) fn coerce_null_int(&mut self, reg: IrId, target: &IrType) -> Option<IrId> {
        let reg_ty = self.builder.get_register_type(reg);
        if *target == IrType::NullInt {
            return self.to_null_int(reg);
        }
        if reg_ty == Some(IrType::NullInt) {
            return self.null_int_to_ir(reg, target);
        }
        None
    }
}
