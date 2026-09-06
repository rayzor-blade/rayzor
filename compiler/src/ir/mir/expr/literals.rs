//! Literals and aggregate construction — arrays, maps, objects, interpolation.

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
    pub(crate) fn lower_literal(&mut self, lit: &HirLiteral, type_id: TypeId) -> Option<IrId> {
        match lit {
            HirLiteral::Int(i) => {
                // Use the actual type from type checking instead of always using I64
                let mut ir_type = self.convert_type(type_id);
                // A literal too wide for the slot the typer picked is widened
                // rather than cut: `3000000000000` typed `Int` was emitted as
                // its low 32 bits and printed 2112827392, and
                // `9223372036854775807i64` printed -1.
                if i32::try_from(*i).is_err()
                    && matches!(ir_type, IrType::I8 | IrType::I16 | IrType::I32)
                {
                    ir_type = IrType::I64;
                }
                let result = self.builder.build_int(*i, ir_type);
                if result.is_none() {
                    // Fallback to I32 if the type is unrecognized (e.g., Ptr(Void) from inlined static fields)
                    return self.builder.build_int(*i, IrType::I32);
                }
                result
            }
            HirLiteral::Float(f) => {
                let ir_type = self.convert_type(type_id);
                match ir_type {
                    IrType::F32 => self.builder.build_const(IrValue::F32(*f as f32)),
                    IrType::F64 => self.builder.build_const(IrValue::F64(*f)),
                    _ => self.builder.build_const(IrValue::F64(*f)),
                }
            }
            HirLiteral::String(s) => {
                let string_content = self
                    .string_interner
                    .get(*s)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| String::new());
                self.builder.build_string(string_content)
            }
            HirLiteral::Bool(b) => self.builder.build_bool(*b),
            HirLiteral::Regex { pattern, flags } => {
                // ~/pattern/flags => haxe_ereg_new("pattern", "flags")
                let pattern_str = self
                    .string_interner
                    .get(*pattern)
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let flags_str = self
                    .string_interner
                    .get(*flags)
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let pattern_val = self.builder.build_string(pattern_str)?;
                let flags_val = self.builder.build_string(flags_str)?;
                let ereg_new = self.get_or_register_extern_function(
                    "haxe_ereg_new",
                    vec![IrType::String, IrType::String],
                    IrType::Ptr(Box::new(IrType::U8)),
                );
                self.builder.build_call_direct(
                    ereg_new,
                    vec![pattern_val, flags_val],
                    IrType::Ptr(Box::new(IrType::U8)),
                )
            }
        }
    }

    pub(crate) fn lower_array_literal(
        &mut self,
        elements: &[HirExpr],
        array_type: TypeId,
    ) -> Option<IrId> {
        // Elements of an interface-typed array must be stored as fat pointers,
        // so resolve the element type's interface symbol once up front.
        //
        // Limitation: if the target `Array<Interface>` type doesn't propagate
        // into the literal's own `expr.ty`, this yields None and elements are
        // stored as raw class pointers. Fixing it needs typechecker-side
        // Class→Interface coercion, or the target type plumbed down here.
        let elem_iface_sym = {
            let element_type_id = {
                let type_table = self.type_table;
                type_table.get(array_type).and_then(|t| match &t.kind {
                    TypeKind::Array { element_type } => Some(*element_type),
                    _ => None,
                })
            };
            element_type_id.and_then(|et| self.get_interface_symbol(et))
        };
        // The HaxeArray struct must live on the heap: a stack allocation is a
        // use-after-free once the array is stored in a global and read back
        // after the function returns.

        let element_count = elements.len();

        // HaxeArray is 4 x i64 = 32 bytes: { len, cap, elem_size, ptr }
        let malloc_func_id = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::U64],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let size_32 = self.builder.build_const(IrValue::U64(32))?;
        let array_ptr = self.builder.build_call_direct(
            malloc_func_id,
            vec![size_32],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        // Zero-initialize the HaxeArray struct fields
        if let Some(zero_i64) = self.builder.build_const(IrValue::I64(0)) {
            // Zero out ptr field (offset 0)
            if let Some(index_0) = self.builder.build_const(IrValue::I32(0)) {
                if let Some(ptr_field) =
                    self.builder
                        .build_gep(array_ptr, vec![index_0], IrType::I64)
                {
                    self.builder.build_store(ptr_field, zero_i64);
                }
            }
            // Zero out len field (offset 8)
            if let Some(index_1) = self.builder.build_const(IrValue::I32(1)) {
                if let Some(len_field) =
                    self.builder
                        .build_gep(array_ptr, vec![index_1], IrType::I64)
                {
                    self.builder.build_store(len_field, zero_i64);
                }
            }
            // Zero out cap field (offset 16)
            if let Some(index_2) = self.builder.build_const(IrValue::I32(2)) {
                if let Some(cap_field) =
                    self.builder
                        .build_gep(array_ptr, vec![index_2], IrType::I64)
                {
                    self.builder.build_store(cap_field, zero_i64);
                }
            }
            // Set elem_size field to 8 bytes (offset 24) - assume pointer size
            if let Some(elem_size_val) = self.builder.build_const(IrValue::I64(8)) {
                if let Some(index_3) = self.builder.build_const(IrValue::I32(3)) {
                    if let Some(elem_size_field) =
                        self.builder
                            .build_gep(array_ptr, vec![index_3], IrType::I64)
                    {
                        self.builder.build_store(elem_size_field, elem_size_val);
                    }
                }
            }
        }

        // For non-empty arrays, push each element using the appropriate
        // typed runtime function. This avoids cross-type bitcasts that lose
        // precision on WASM32 (where IrType::I64 lowers to WASM i32).
        if element_count > 0 {
            // Register haxe_array_push_i64: fn(arr: *HaxeArray, val: i64) -> void
            let push_i64_func_id = self.get_or_register_extern_function(
                "haxe_array_push_i64",
                vec![
                    IrType::Ptr(Box::new(IrType::I64)), // arr pointer
                    IrType::I64,                        // value (i64 for pointer-sized values)
                ],
                IrType::Void,
            );
            // Register haxe_array_push_f64: fn(arr: *HaxeArray, val: f64) -> void
            let push_f64_func_id = self.get_or_register_extern_function(
                "haxe_array_push_f64",
                vec![IrType::Ptr(Box::new(IrType::I64)), IrType::F64],
                IrType::Void,
            );

            // Pre-pass to learn each element's MIR type. Elements with mixed
            // register types must each be boxed as a `DynamicValue*`, otherwise
            // different-typed bytes share one `i64` slot and read back as
            // nonsense; homogeneous arrays stay on the fast path. The HIR
            // `TypeKind` is kept so boxing can pick the right `haxe_box_*`.
            let lowered: Vec<Option<(IrId, Option<IrType>, Option<crate::tast::TypeKind>)>> =
                elements
                    .iter()
                    .map(|elem| {
                        let v = self.lower_expression(elem)?;
                        let v = if let Some(iface_sym) = elem_iface_sym {
                            let class_sym = self.get_class_symbol(elem.ty);
                            if let Some(class_sym) = class_sym {
                                // Not gated on `interface_vtables.contains_key`:
                                // wrap_in_interface_fat_ptr builds the
                                // (class, iface) vtable lazily when it wasn't
                                // built eagerly.
                                if let Some(wrapped) =
                                    self.wrap_in_interface_fat_ptr(v, class_sym, iface_sym)
                                {
                                    self.interface_wrapped_args.insert(wrapped);
                                    wrapped
                                } else {
                                    v
                                }
                            } else {
                                v
                            }
                        } else {
                            v
                        };
                        let t = self.builder.get_register_type(v);
                        let hir_kind = {
                            let type_table = self.type_table;
                            type_table.get(elem.ty).map(|tr| tr.kind.clone())
                        };
                        Some((v, t, hir_kind))
                    })
                    .collect();
            let heterogeneous = {
                let normalize = |t: &Option<IrType>| match t {
                    Some(IrType::I32) | Some(IrType::I64) => 1u8,
                    Some(IrType::F32) | Some(IrType::F64) => 2u8,
                    Some(IrType::Bool) => 3u8,
                    Some(IrType::Ptr(_)) => 4u8,
                    _ => 5u8,
                };
                let first = lowered
                    .iter()
                    .find_map(|p| p.as_ref())
                    .map(|(_, t, _)| normalize(t));
                first.map_or(false, |first_kind| {
                    lowered
                        .iter()
                        .filter_map(|p| p.as_ref())
                        .any(|(_, t, _)| normalize(t) != first_kind)
                })
            };

            for entry in lowered {
                let (elem_val, elem_type, hir_kind) = match entry {
                    Some(triple) => triple,
                    None => continue,
                };
                // Heterogeneous-array path: box every primitive element so the
                // runtime walking the array sees a uniform sequence of
                // `DynamicValue*`s and can dispatch type-aware formatting.
                if heterogeneous {
                    let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                    let boxed = match &elem_type {
                        Some(IrType::Bool) => {
                            let f = self.get_or_register_extern_function(
                                "haxe_box_bool_ptr",
                                vec![IrType::Bool],
                                ptr_u8.clone(),
                            );
                            self.builder
                                .build_call_direct(f, vec![elem_val], ptr_u8.clone())
                        }
                        Some(IrType::I32) => {
                            let ext =
                                self.builder
                                    .build_cast(elem_val, IrType::I32, IrType::I64)?;
                            let f = self.get_or_register_extern_function(
                                "haxe_box_int_ptr",
                                vec![IrType::I64],
                                ptr_u8.clone(),
                            );
                            self.builder.build_call_direct(f, vec![ext], ptr_u8.clone())
                        }
                        Some(IrType::I64) => {
                            let f = self.get_or_register_extern_function(
                                "haxe_box_int_ptr",
                                vec![IrType::I64],
                                ptr_u8.clone(),
                            );
                            self.builder
                                .build_call_direct(f, vec![elem_val], ptr_u8.clone())
                        }
                        Some(IrType::F32) => {
                            let promoted =
                                self.builder
                                    .build_cast(elem_val, IrType::F32, IrType::F64)?;
                            let f = self.get_or_register_extern_function(
                                "haxe_box_float_ptr",
                                vec![IrType::F64],
                                ptr_u8.clone(),
                            );
                            self.builder
                                .build_call_direct(f, vec![promoted], ptr_u8.clone())
                        }
                        Some(IrType::F64) => {
                            let f = self.get_or_register_extern_function(
                                "haxe_box_float_ptr",
                                vec![IrType::F64],
                                ptr_u8.clone(),
                            );
                            self.builder
                                .build_call_direct(f, vec![elem_val], ptr_u8.clone())
                        }
                        Some(IrType::String) => {
                            // `IrType::String` is a `HaxeString*`, not a generic
                            // `Ptr`; wrap it so readers see a `TYPE_STRING`
                            // DynamicValue header at offset 0.
                            let f = self.get_or_register_extern_function(
                                "haxe_box_haxestring_ptr",
                                vec![ptr_u8.clone()],
                                ptr_u8.clone(),
                            );
                            let cast = self
                                .builder
                                .build_bitcast(elem_val, ptr_u8.clone())
                                .unwrap_or(elem_val);
                            self.builder
                                .build_call_direct(f, vec![cast], ptr_u8.clone())
                        }
                        Some(IrType::Ptr(_)) => {
                            // `null`, class instances and pre-boxed
                            // `DynamicValue*`s carry the right shape
                            // already (or are null) — pass through.
                            Some(elem_val)
                        }
                        _ => Some(elem_val),
                    };
                    let push_val = boxed
                        .and_then(|b| self.builder.build_bitcast(b, IrType::I64))
                        .unwrap_or(elem_val);
                    self.builder.build_call_direct(
                        push_i64_func_id,
                        vec![array_ptr, push_val],
                        IrType::Void,
                    );
                    continue;
                }

                // For F64 elements, use the dedicated f64 push function to
                // preserve the full 64 bits on WASM32.
                if matches!(elem_type, Some(IrType::F64)) {
                    self.builder.build_call_direct(
                        push_f64_func_id,
                        vec![array_ptr, elem_val],
                        IrType::Void,
                    );
                    continue;
                }

                // Convert element to i64 to match haxe_array_push_i64's signature.
                let push_val = match &elem_type {
                    Some(IrType::Ptr(_)) => {
                        // Pointer → i64: bitcast (reinterpret pointer as integer)
                        self.builder
                            .build_bitcast(elem_val, IrType::I64)
                            .unwrap_or(elem_val)
                    }
                    Some(IrType::I32) => {
                        // I32 → I64: sign-extend (not bitcast)
                        self.builder
                            .build_cast(elem_val, IrType::I32, IrType::I64)
                            .unwrap_or(elem_val)
                    }
                    Some(IrType::F32) => {
                        // F32 → F64 → push_f64 (use the float path)
                        let as_f64 = self
                            .builder
                            .build_cast(elem_val, IrType::F32, IrType::F64)
                            .unwrap_or(elem_val);
                        self.builder.build_call_direct(
                            push_f64_func_id,
                            vec![array_ptr, as_f64],
                            IrType::Void,
                        );
                        continue;
                    }
                    _ => elem_val,
                };

                self.builder.build_call_direct(
                    push_i64_func_id,
                    vec![array_ptr, push_val],
                    IrType::Void,
                );
            }
        }

        Some(array_ptr)
    }

    pub(crate) fn lower_map_literal(&mut self, entries: &[(HirExpr, HirExpr)]) -> Option<IrId> {
        // Map literal: [key1 => val1, key2 => val2, ...]
        //
        // Determine key type from first entry and use appropriate runtime:
        // - Int keys → IntMap (haxe_intmap_new/set)
        // - String keys → StringMap (haxe_stringmap_new/set)
        // - Object keys → ObjectMap (haxe_objectmap_new/set, pointer identity)

        if entries.is_empty() {
            // Default to StringMap for empty map literals
            let new_fn = self.get_or_register_extern_function(
                "haxe_stringmap_new",
                vec![],
                IrType::Ptr(Box::new(IrType::Void)),
            );
            return self.builder.build_call_direct(
                new_fn,
                vec![],
                IrType::Ptr(Box::new(IrType::Void)),
            );
        }

        let key_type_kind = {
            let type_table = self.type_table;
            type_table.get(entries[0].0.ty).map(|t| t.kind.clone())
        };
        let is_int_key = matches!(
            key_type_kind,
            Some(crate::tast::TypeKind::Int) | Some(crate::tast::TypeKind::Bool)
        );
        let is_string_key = matches!(key_type_kind, Some(crate::tast::TypeKind::String));

        let map_ptr_type = IrType::Ptr(Box::new(IrType::Void));

        if is_int_key {
            // IntMap
            let new_fn = self.get_or_register_extern_function(
                "haxe_intmap_new",
                vec![],
                map_ptr_type.clone(),
            );
            let map_ptr = self
                .builder
                .build_call_direct(new_fn, vec![], map_ptr_type.clone())?;

            let set_fn = self.get_or_register_extern_function(
                "haxe_intmap_set",
                vec![map_ptr_type.clone(), IrType::I64, IrType::U64],
                IrType::Void,
            );

            for (key, value) in entries.iter() {
                let key_val = self.lower_expression(key)?;
                let value_val = self.lower_expression(value)?;

                // Cast key to i64
                let key_type = self.convert_type(key.ty);
                let key_i64 = if key_type != IrType::I64 {
                    self.builder
                        .build_cast(key_val, key_type, IrType::I64)
                        .unwrap_or(key_val)
                } else {
                    key_val
                };

                // Cast value to u64 for raw storage
                let val_type = self.convert_type(value.ty);
                let val_u64 = match &val_type {
                    IrType::F64 => self
                        .builder
                        .build_bitcast(value_val, IrType::U64)
                        .unwrap_or(value_val),
                    IrType::U64 => value_val,
                    _ => self
                        .builder
                        .build_cast(value_val, val_type, IrType::U64)
                        .unwrap_or(value_val),
                };

                self.builder.build_call_direct(
                    set_fn,
                    vec![map_ptr, key_i64, val_u64],
                    IrType::Void,
                );
            }

            Some(map_ptr)
        } else if is_string_key {
            // StringMap
            let new_fn = self.get_or_register_extern_function(
                "haxe_stringmap_new",
                vec![],
                map_ptr_type.clone(),
            );
            let map_ptr = self
                .builder
                .build_call_direct(new_fn, vec![], map_ptr_type.clone())?;

            let set_fn = self.get_or_register_extern_function(
                "haxe_stringmap_set",
                vec![
                    map_ptr_type.clone(),
                    IrType::Ptr(Box::new(IrType::U8)),
                    IrType::U64,
                ],
                IrType::Void,
            );

            for (key, value) in entries.iter() {
                let key_val = self.lower_expression(key)?;
                let value_val = self.lower_expression(value)?;

                // Cast value to u64 for raw storage
                let val_type = self.convert_type(value.ty);
                let val_u64 = match &val_type {
                    IrType::F64 => self
                        .builder
                        .build_bitcast(value_val, IrType::U64)
                        .unwrap_or(value_val),
                    IrType::U64 => value_val,
                    _ => self
                        .builder
                        .build_cast(value_val, val_type, IrType::U64)
                        .unwrap_or(value_val),
                };

                self.builder.build_call_direct(
                    set_fn,
                    vec![map_ptr, key_val, val_u64],
                    IrType::Void,
                );
            }

            Some(map_ptr)
        } else {
            // ObjectMap (object/pointer keys, identity-based)
            let new_fn = self.get_or_register_extern_function(
                "haxe_objectmap_new",
                vec![],
                map_ptr_type.clone(),
            );
            let map_ptr = self
                .builder
                .build_call_direct(new_fn, vec![], map_ptr_type.clone())?;

            let set_fn = self.get_or_register_extern_function(
                "haxe_objectmap_set",
                vec![map_ptr_type.clone(), IrType::U64, IrType::U64],
                IrType::Void,
            );

            for (key, value) in entries.iter() {
                let key_val = self.lower_expression(key)?;
                let value_val = self.lower_expression(value)?;

                // Cast key pointer to u64 for identity-based lookup
                let key_type = self.convert_type(key.ty);
                let key_u64 = match &key_type {
                    IrType::U64 | IrType::I64 => key_val,
                    _ => self
                        .builder
                        .build_cast(key_val, key_type, IrType::U64)
                        .unwrap_or(key_val),
                };

                // Cast value to u64 for raw storage
                let val_type = self.convert_type(value.ty);
                let val_u64 = match &val_type {
                    IrType::F64 => self
                        .builder
                        .build_bitcast(value_val, IrType::U64)
                        .unwrap_or(value_val),
                    IrType::U64 => value_val,
                    _ => self
                        .builder
                        .build_cast(value_val, val_type, IrType::U64)
                        .unwrap_or(value_val),
                };

                self.builder.build_call_direct(
                    set_fn,
                    vec![map_ptr, key_u64, val_u64],
                    IrType::Void,
                );
            }

            Some(map_ptr)
        }
    }

    fn lower_struct_init_class_literal(
        &mut self,
        fields: &[(InternedString, HirExpr)],
        class_type: TypeId,
        class_symbol: SymbolId,
    ) -> Option<IrId> {
        let storage_fields = self.class_instance_fields.get(&class_symbol)?.clone();
        let class_name = self
            .symbol_table
            .get_symbol(class_symbol)
            .and_then(|symbol| {
                symbol
                    .qualified_name
                    .and_then(|name| self.string_interner.get(name))
                    .or_else(|| self.string_interner.get(symbol.name))
            });
        let layout_size = storage_fields
            .iter()
            .map(|(_, _, index)| (*index as u64 + 1) * 8)
            .max()
            .unwrap_or(16)
            .max(16);
        let object_size = self
            .class_alloc_sizes
            .get(&class_symbol)
            .copied()
            .or_else(|| {
                class_name.and_then(|name| self.class_alloc_sizes_by_name.get(name).copied())
            })
            .unwrap_or(layout_size)
            .max(layout_size);

        let raw_type_id = self.runtime_type_id(class_type);
        let runtime_type_id = if raw_type_id != 0 {
            raw_type_id
        } else {
            self.deterministic_class_type_id(class_symbol)
                .unwrap_or(raw_type_id)
        } as i64;
        let header = self.builder.build_const(IrValue::I64(runtime_type_id));
        let object = self.build_heap_alloc_with_header(object_size, header)?;

        // malloc leaves every field after the header uninitialized. Initialize
        // the complete storage shape before applying the literal so omitted
        // optional/default fields have deterministic Haxe defaults.
        for &(_, field_type, index) in &storage_fields {
            let value = self.build_type_default(field_type)?;
            let index = self.builder.build_const(IrValue::I64(index as i64))?;
            let field_ir_type = self.convert_type(field_type);
            let field_ptr = self.builder.build_gep(object, vec![index], field_ir_type)?;
            self.builder.build_store(field_ptr, value);
        }

        for (literal_name, expression) in fields {
            let literal_name = self
                .string_interner
                .get(*literal_name)
                .unwrap_or("<unknown>");
            let (field_type, index) =
                storage_fields
                    .iter()
                    .find_map(|(field_symbol, field_type, index)| {
                        let field_name = self
                            .symbol_table
                            .get_symbol(*field_symbol)
                            .and_then(|symbol| self.string_interner.get(symbol.name))?;
                        (field_name == literal_name).then_some((*field_type, *index))
                    })?;
            let value = self.lower_expression(expression)?;
            let (value, _) = self.maybe_wrap_for_interface(value, expression.ty, field_type);
            let index = self.builder.build_const(IrValue::I64(index as i64))?;
            let field_ir_type = self.convert_type(field_type);
            let field_ptr = self.builder.build_gep(object, vec![index], field_ir_type)?;
            self.builder.build_store(field_ptr, value);
        }

        Some(object)
    }

    pub(crate) fn lower_object_literal(
        &mut self,
        fields: &[(InternedString, HirExpr)],
        expr_type: TypeId,
    ) -> Option<IrId> {
        // Lowered to an Arc-based anonymous object: fields are sorted
        // alphabetically into a shape, then set by slot index.

        let mut literal_field_names: Vec<(String, usize)> = Vec::with_capacity(fields.len());
        for (i, (interned_name, _expr)) in fields.iter().enumerate() {
            let name = self
                .string_interner
                .get(*interned_name)
                .unwrap_or("<unknown>")
                .to_string();
            literal_field_names.push((name, i));
        }

        // Check if the expression type (resolved through aliases) has additional
        // optional fields not present in the literal. The literal's own
        // `expr_type` is its narrow inferred shape (just its own fields); if a
        // wider typedef is expected at the use site (return, let, assign, call
        // arg), prefer that — otherwise optional fields are silently dropped
        // from the slot layout and reader/writer slot indices diverge.
        let effective_ty = self.object_literal_target_ty.unwrap_or(expr_type);
        let resolved_ty = self.resolve_through_aliases(effective_ty);
        let class_symbol = self
            .type_table
            .get(resolved_ty)
            .and_then(|ty| match &ty.kind {
                TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                TypeKind::GenericInstance { base_type, .. } => self
                    .type_table
                    .get(*base_type)
                    .and_then(|base| match &base.kind {
                        TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                        _ => None,
                    }),
                _ => None,
            });
        if let Some(class_symbol) = class_symbol {
            return self.lower_struct_init_class_literal(fields, resolved_ty, class_symbol);
        }
        let mut optional_defaults: Vec<String> = Vec::new();
        {
            let type_table = self.type_table;
            if let Some(ty_info) = type_table.get(resolved_ty) {
                if let TypeKind::Anonymous {
                    fields: anon_fields,
                } = &ty_info.kind
                {
                    let literal_names: std::collections::BTreeSet<&str> = literal_field_names
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect();
                    for af in anon_fields {
                        if af.optional {
                            if let Some(fname) = self.string_interner.get(af.name) {
                                if !literal_names.contains(fname) {
                                    optional_defaults.push(fname.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Build sorted (name, source) list — source is either an index into literal fields
        // or None for optional defaults
        let mut named_fields: Vec<(String, Option<usize>)> = literal_field_names
            .into_iter()
            .map(|(name, idx)| (name, Some(idx)))
            .collect();
        for opt_name in &optional_defaults {
            named_fields.push((opt_name.clone(), None));
        }
        // Sort by field name for canonical ordering (matches runtime shape table)
        named_fields.sort_by(|a, b| a.0.cmp(&b.0));

        let total_field_count = named_fields.len();

        // Build shape key as comma-joined sorted field names
        let shape_key: String = named_fields
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(",");

        let shape_id = if let Some(&existing_id) = self.anonymous_shapes.get(&shape_key) {
            existing_id
        } else {
            let id = self.next_anon_shape_id;
            self.next_anon_shape_id += 1;
            self.anonymous_shapes.insert(shape_key, id);
            id
        };

        // Build shape descriptor string: "name1:type_id1,name2:type_id2,..."
        // Type IDs: 0=Void, 1=Null, 2=Bool, 3=Int, 4=Float, 5=String
        let descriptor = {
            let mut parts = Vec::with_capacity(named_fields.len());
            for (name, source) in &named_fields {
                let runtime_type_id = match source {
                    Some(orig_idx) => {
                        let field_type = fields[*orig_idx].1.ty;
                        self.runtime_type_id(field_type)
                    }
                    None => 1, // null for optional defaults
                };
                parts.push(format!("{}:{}", name, runtime_type_id));
            }
            parts.join(",")
        };

        // Register rayzor_ensure_shape extern: (shape_id: u32, descriptor: HaxeString*) -> void
        let ensure_shape_id = self.get_or_register_extern_function(
            "rayzor_ensure_shape",
            vec![IrType::I32, IrType::String],
            IrType::Void,
        );

        // Emit: rayzor_ensure_shape(shape_id, descriptor) — idempotent, registers once
        let shape_id_const = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let desc_str = self.builder.build_const(IrValue::String(descriptor))?;
        self.builder.build_call_direct(
            ensure_shape_id,
            vec![shape_id_const, desc_str],
            IrType::Void,
        );

        // Register rayzor_anon_new extern: (shape_id: u32, field_count: u32) -> *mut u8
        let anon_new_id = self.get_or_register_extern_function(
            "rayzor_anon_new",
            vec![IrType::I32, IrType::I32],
            IrType::Ptr(Box::new(IrType::U8)),
        );

        // Register rayzor_anon_set_field_by_index extern: (handle: *mut u8, index: u32, value: u64) -> void
        let anon_set_id = self.get_or_register_extern_function(
            "rayzor_anon_set_field_by_index",
            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32, IrType::I64],
            IrType::Void,
        );

        // Map field name → declared target type (from the anon typedef), so a
        // class value assigned to an interface-typed field is wrapped in the
        // interface fat pointer (invariant: interface-typed anon fields hold a
        // fat pointer, never a raw class object).
        let field_target_types: BTreeMap<String, TypeId> = {
            let type_table = self.type_table;
            match type_table.get(resolved_ty).map(|t| &t.kind) {
                Some(TypeKind::Anonymous {
                    fields: anon_fields,
                }) => anon_fields
                    .iter()
                    .filter_map(|af| {
                        self.string_interner
                            .get(af.name)
                            .map(|n| (n.to_string(), af.type_id))
                    })
                    .collect(),
                _ => BTreeMap::new(),
            }
        };

        // Emit: handle = rayzor_anon_new(shape_id, total_field_count)
        let shape_id_val = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let field_count_val = self
            .builder
            .build_const(IrValue::I32(total_field_count as i32))?;
        let handle = self.builder.build_call_direct(
            anon_new_id,
            vec![shape_id_val, field_count_val],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        // Lower each field value and store at its sorted index
        for (sorted_idx, (field_name, source)) in named_fields.iter().enumerate() {
            let idx_val = self.builder.build_const(IrValue::I32(sorted_idx as i32))?;

            match source {
                Some(orig_idx) => {
                    // Literal field — lower the expression
                    let field_expr = &fields[*orig_idx].1;
                    let field_val = self.lower_expression(field_expr)?;
                    // Wrap a class value into an interface fat pointer when the
                    // field's declared type is that interface.
                    let (field_val, field_val_ty) =
                        if let Some(&target_ty) = field_target_types.get(field_name) {
                            let (wrapped, did_wrap) =
                                self.maybe_wrap_for_interface(field_val, field_expr.ty, target_ty);
                            if did_wrap {
                                (wrapped, target_ty)
                            } else {
                                (field_val, field_expr.ty)
                            }
                        } else {
                            (field_val, field_expr.ty)
                        };
                    // A field typed by a bare type parameter holds raw bits, as an
                    // erased formal does, so a `Null<T>` box gives up its payload.
                    let erased_target = field_target_types
                        .get(field_name)
                        .and_then(|t| self.type_table.get(*t))
                        .map(|t| match &t.kind {
                            TypeKind::TypeParameter { .. } => true,
                            TypeKind::Optional { inner_type, .. } => self
                                .type_table
                                .get(*inner_type)
                                .map(|i| matches!(i.kind, TypeKind::TypeParameter { .. }))
                                .unwrap_or(false),
                            _ => false,
                        })
                        .unwrap_or(false);
                    let boxed_source = self
                        .type_table
                        .get(field_val_ty)
                        .map(|t| matches!(t.kind, TypeKind::Optional { .. }))
                        .unwrap_or(false);
                    let val_as_i64 = if erased_target && boxed_source {
                        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                        let unbox_id = self.get_or_register_extern_function(
                            "haxe_unbox_scalar_or_addr",
                            vec![ptr_u8],
                            IrType::I64,
                        );
                        self.builder
                            .build_call_direct(unbox_id, vec![field_val], IrType::I64)?
                    } else {
                        self.coerce_to_i64(field_val, field_val_ty)?
                    };
                    self.builder.build_call_direct(
                        anon_set_id,
                        vec![handle, idx_val, val_as_i64],
                        IrType::Void,
                    );
                }
                None => {
                    // Optional field with default value (0/null)
                    let zero = self.builder.build_const(IrValue::I64(0))?;
                    self.builder.build_call_direct(
                        anon_set_id,
                        vec![handle, idx_val, zero],
                        IrType::Void,
                    );
                }
            }
        }

        Some(handle)
    }

    pub(crate) fn lower_string_interpolation(&mut self, parts: &[HirStringPart]) -> Option<IrId> {
        if parts.is_empty() {
            return self.builder.build_string(String::new());
        }

        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));

        // Pre-register extern functions for type conversion and concatenation
        let concat_fn = self.get_or_register_extern_function(
            "haxe_string_concat",
            vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
            string_ptr_ty.clone(),
        );

        let mut result: Option<IrId> = None;

        for part in parts {
            let part_str = match part {
                HirStringPart::Literal(s) => {
                    let text = self.interned_str(*s).to_string();
                    self.builder.build_string(text)?
                }
                HirStringPart::Interpolation(expr) => {
                    let expr_val = self.lower_expression(expr)?;

                    if self.expr_is_value_type_expr(expr) {
                        self.convert_value_type_to_string(expr_val)?
                    } else {
                        let expr_type_kind = {
                            let type_table = self.type_table;
                            type_table.get(expr.ty).map(|ti| ti.kind.clone())
                        };

                        match expr_type_kind.as_ref() {
                            Some(TypeKind::String) => expr_val, // already a string
                            Some(TypeKind::Int) => {
                                let conv_fn = self.get_or_register_extern_function(
                                    "haxe_string_from_int",
                                    vec![IrType::I64],
                                    string_ptr_ty.clone(),
                                );
                                self.builder.build_call_direct(
                                    conv_fn,
                                    vec![expr_val],
                                    string_ptr_ty.clone(),
                                )?
                            }
                            Some(TypeKind::Float) => {
                                let conv_fn = self.get_or_register_extern_function(
                                    "haxe_string_from_float",
                                    vec![IrType::F64],
                                    string_ptr_ty.clone(),
                                );
                                self.builder.build_call_direct(
                                    conv_fn,
                                    vec![expr_val],
                                    string_ptr_ty.clone(),
                                )?
                            }
                            Some(TypeKind::Bool) => {
                                let conv_fn = self.get_or_register_extern_function(
                                    "haxe_string_from_bool",
                                    vec![IrType::I32],
                                    string_ptr_ty.clone(),
                                );
                                self.builder.build_call_direct(
                                    conv_fn,
                                    vec![expr_val],
                                    string_ptr_ty.clone(),
                                )?
                            }
                            Some(TypeKind::Array { .. }) => {
                                // Array<T> → haxe_array_to_string
                                let conv_fn = self.get_or_register_extern_function(
                                    "haxe_array_to_string",
                                    vec![IrType::Ptr(Box::new(IrType::Void))],
                                    string_ptr_ty.clone(),
                                );
                                self.builder.build_call_direct(
                                    conv_fn,
                                    vec![expr_val],
                                    string_ptr_ty.clone(),
                                )?
                            }
                            _ => {
                                // Fallback: treat as int (prints raw i64 value)
                                let conv_fn = self.get_or_register_extern_function(
                                    "haxe_string_from_int",
                                    vec![IrType::I64],
                                    string_ptr_ty.clone(),
                                );
                                self.builder.build_call_direct(
                                    conv_fn,
                                    vec![expr_val],
                                    string_ptr_ty.clone(),
                                )?
                            }
                        }
                    }
                }
            };

            result = match result {
                None => Some(part_str),
                Some(acc) => self.builder.build_call_direct(
                    concat_fn,
                    vec![acc, part_str],
                    string_ptr_ty.clone(),
                ),
            };
        }

        result
    }
}
