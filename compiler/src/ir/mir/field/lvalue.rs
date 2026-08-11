//! Assignment targets: the read side of compound assignment, and the write.

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
    pub(crate) fn lower_lvalue_read(&mut self, lvalue: &HirLValue) -> Option<IrId> {
        match lvalue {
            HirLValue::Variable(symbol) => self.symbol_map.get(symbol).copied(),
            HirLValue::Field { object, field } => {
                if let Some(obj_reg) = self.lower_expression(object) {
                    let receiver_ty = object.ty;
                    // TODO: look up the field type from the symbol table; the runtime
                    // call path does not need it.
                    let field_ty = TypeId(u32::MAX);
                    self.lower_field_access(obj_reg, *field, receiver_ty, field_ty)
                } else {
                    None
                }
            }
            HirLValue::Index { object, index } => {
                if let Some(obj_reg) = self.lower_expression(object) {
                    if let Some(idx_reg) = self.lower_expression(index) {
                        let elem_ty = object.ty;
                        self.lower_index_access(obj_reg, idx_reg, elem_ty)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    pub(crate) fn lower_lvalue_write(&mut self, lvalue: &HirLValue, value: IrId) {
        match lvalue {
            HirLValue::Variable(symbol) => {
                let global_id = self.global_symbol_map.get(symbol).copied().or_else(|| {
                    // Name-based fallback: SymbolIds may differ between contexts
                    let sym_name = self
                        .symbol_table
                        .get_symbol(*symbol)
                        .and_then(|s| self.string_interner.get(s.name))?;
                    for (&gsym, &gid) in &self.global_symbol_map {
                        if let Some(gsym_info) = self.symbol_table.get_symbol(gsym) {
                            if let Some(gname) = self.string_interner.get(gsym_info.name) {
                                if gname == sym_name {
                                    return Some(gid);
                                }
                            }
                        }
                    }
                    // Also search module globals by name suffix
                    for global in self.builder.module.globals.values() {
                        if global.name.ends_with(&format!(".{}", sym_name))
                            || global.name == sym_name
                        {
                            return Some(global.id);
                        }
                    }
                    None
                });
                if let Some(global_id) = global_id {
                    self.builder.build_store_global(global_id, value);
                    // Untrack from drop system — value escapes to global storage
                    self.owned_heap_values.remove(symbol);
                    return;
                }

                // Kept for the type of the previous binding.
                let old_reg = self.symbol_map.get(symbol).copied();

                self.symbol_map.insert(*symbol, value);

                // The new value register needs a local entry for phi node tracking.
                if let Some(func) = self.builder.current_function_mut() {
                    if !func.locals.contains_key(&value) {
                        let var_type = old_reg
                            .and_then(|r| func.locals.get(&r))
                            .map(|local| local.ty.clone())
                            .unwrap_or(IrType::Ptr(Box::new(IrType::Void)));

                        let var_name = self
                            .symbol_table
                            .get_symbol(*symbol)
                            .and_then(|s| self.string_interner.get(s.name))
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("var_{}", symbol.as_raw()));

                        func.locals.insert(
                            value,
                            crate::ir::IrLocal {
                                name: format!("{}_v{}", var_name, value.0),
                                ty: var_type,
                                mutable: true,
                                source_location: crate::ir::IrSourceLocation::unknown(),
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }
            }
            HirLValue::Field { object, field } => {
                if let Some(obj_reg) = self.lower_expression(object) {
                    // Check if this is a property with a custom setter.
                    // Clone the info so the immutable borrow of `self` is released
                    // before any of the per-arm fallbacks that need `&mut self`.
                    let property_info_owned = self.property_access_map.get(field).cloned();
                    if let Some(property_info) = property_info_owned.as_ref() {
                        match &property_info.setter {
                            crate::tast::PropertyAccessor::Method(setter_method_name) => {
                                let setter_func_id = self
                                    .function_map
                                    .iter()
                                    .find(|(sym_id, _)| {
                                        if let Some(symbol) = self.symbol_table.get_symbol(**sym_id)
                                        {
                                            symbol.name == *setter_method_name
                                        } else {
                                            false
                                        }
                                    })
                                    .map(|(_, func_id)| *func_id);

                                if let Some(func_id) = setter_func_id {
                                    // Setters take (this, value) and return the value set.
                                    let return_type =
                                        if let Some(func) = self.builder.current_function() {
                                            func.locals
                                                .get(&value)
                                                .map(|local| local.ty.clone())
                                                .unwrap_or(IrType::I32)
                                        } else {
                                            IrType::I32
                                        };

                                    self.builder.build_call_direct(
                                        func_id,
                                        vec![obj_reg, value],
                                        return_type,
                                    );
                                    return;
                                }

                                // Fallback: extern-class accessor — try the stdlib mapping
                                // (e.g. sys.thread.Tls.set_value → sys_tls_set_value).
                                let setter_name = *setter_method_name;
                                let receiver_ty = object.ty;
                                if self
                                    .try_property_call_via_stdlib(
                                        receiver_ty,
                                        setter_name,
                                        vec![obj_reg, value],
                                        IrType::Void,
                                    )
                                    .is_some()
                                {
                                    return;
                                }

                                let method_name_str = self
                                    .string_interner
                                    .get(*setter_method_name)
                                    .unwrap_or("<unknown>");
                                self.add_error(
                                    &format!(
                                        "Property setter method '{}' not found",
                                        method_name_str
                                    ),
                                    SourceLocation::unknown(),
                                );
                                return;
                            }
                            crate::tast::PropertyAccessor::Null => {
                                // `null` setter = writable from inside the class only;
                                // access control is enforced in the type checker, so
                                // lowering falls through to direct field access.
                            }
                            crate::tast::PropertyAccessor::Never => {
                                self.add_error(
                                    "Cannot write to read-only property (never setter)",
                                    SourceLocation::unknown(),
                                );
                                return;
                            }
                            crate::tast::PropertyAccessor::Default
                            | crate::tast::PropertyAccessor::Dynamic => {
                                // Fall through to direct field access
                            }
                        }
                    }

                    // Structural subtyping: if object is a variable with an anon view,
                    // redirect field store to the backing representation
                    if let HirExprKind::Variable {
                        symbol: obj_sym, ..
                    } = &object.kind
                    {
                        if let Some(backing) = self.anon_views.get(obj_sym).cloned() {
                            let field_name = self
                                .symbol_table
                                .get_symbol(*field)
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|s| s.to_string());

                            if let Some(field_name) = field_name {
                                match &backing {
                                    AnonBacking::Class { field_map, .. } => {
                                        if let Some((_, gep_idx, field_type_id)) =
                                            field_map.iter().find(|(n, ..)| *n == field_name)
                                        {
                                            let field_ir_ty = self.convert_type(*field_type_id);
                                            let idx_const = self
                                                .builder
                                                .build_const(IrValue::I64(*gep_idx as i64));
                                            if let Some(idx_c) = idx_const {
                                                let field_ptr = self.builder.build_gep(
                                                    obj_reg,
                                                    vec![idx_c],
                                                    field_ir_ty.clone(),
                                                );
                                                if let Some(fp) = field_ptr {
                                                    self.builder.build_store(fp, value);
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                    AnonBacking::WiderAnon { field_map, .. } => {
                                        if let Some((_, src_idx, field_type_id)) =
                                            field_map.iter().find(|(n, ..)| *n == field_name)
                                        {
                                            let anon_set_id = self.get_or_register_extern_function(
                                                "rayzor_anon_set_field_by_index",
                                                vec![
                                                    IrType::Ptr(Box::new(IrType::U8)),
                                                    IrType::I32,
                                                    IrType::I64,
                                                ],
                                                IrType::Void,
                                            );
                                            let coerced = self.coerce_to_i64(value, *field_type_id);
                                            let idx_val = self
                                                .builder
                                                .build_const(IrValue::I32(*src_idx as i32));
                                            if let (Some(cv), Some(iv)) = (coerced, idx_val) {
                                                self.builder.build_call_direct(
                                                    anon_set_id,
                                                    vec![obj_reg, iv, cv],
                                                    IrType::Void,
                                                );
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Anonymous types (or typedef aliases to one) store by sorted
                    // field index through rayzor_anon_set_field_by_index.
                    {
                        let resolved_obj_ty = self.resolve_through_aliases(object.ty);
                        let is_anon = {
                            let type_table = self.type_table;
                            if let Some(ty_info) = type_table.get(resolved_obj_ty) {
                                matches!(ty_info.kind, TypeKind::Anonymous { .. })
                            } else {
                                false
                            }
                        };

                        if is_anon {
                            let field_name = self
                                .symbol_table
                                .get_symbol(*field)
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|s| s.to_string());

                            if let Some(field_name) = field_name {
                                let sorted_index = {
                                    let type_table = self.type_table;
                                    if let Some(ty_info) = type_table.get(resolved_obj_ty) {
                                        if let TypeKind::Anonymous {
                                            fields: anon_fields,
                                        } = &ty_info.kind
                                        {
                                            let mut field_names: Vec<String> = anon_fields
                                                .iter()
                                                .filter_map(|f| {
                                                    self.string_interner
                                                        .get(f.name)
                                                        .map(|s| s.to_string())
                                                })
                                                .collect();
                                            field_names.sort();
                                            field_names.iter().position(|n| *n == field_name)
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                };

                                if let Some(sorted_idx) = sorted_index {
                                    let anon_set_id = self.get_or_register_extern_function(
                                        "rayzor_anon_set_field_by_index",
                                        vec![
                                            IrType::Ptr(Box::new(IrType::U8)),
                                            IrType::I32,
                                            IrType::I64,
                                        ],
                                        IrType::Void,
                                    );

                                    // Anon storage is i64; coerce via the field's
                                    // declared type when the symbol has one.
                                    let field_type_id = self
                                        .symbol_table
                                        .get_symbol(*field)
                                        .map(|s| s.type_id)
                                        .unwrap_or(TypeId(u32::MAX));
                                    let val_i64 =
                                        self.coerce_to_i64(value, field_type_id).unwrap_or(value);

                                    let idx_val =
                                        self.builder.build_const(IrValue::I32(sorted_idx as i32));
                                    if let Some(idx_val) = idx_val {
                                        self.builder.build_call_direct(
                                            anon_set_id,
                                            vec![obj_reg, idx_val, val_i64],
                                            IrType::Void,
                                        );
                                    }
                                    return;
                                }
                            }
                        }
                    }

                    // Look up the field index (with fallback to name lookup)
                    let field_index_opt = self
                        .field_index_map
                        .get(field)
                        .map(|&(_, idx)| idx)
                        .or_else(|| {
                            // Fallback: disambiguate by receiver type when multiple classes
                            // have the same field name (e.g., StringBuf.length vs List.length)
                            let field_name =
                                self.symbol_table.get_symbol(*field).map(|s| s.name)?;
                            let receiver_ty = object.ty;
                            self.resolve_field_index_by_name(field_name, receiver_ty)
                                .map(|(_, idx)| idx)
                        });

                    if let Some(field_index) = field_index_opt {
                        // @:cstruct: use byte-offset PtrAdd instead of GEP
                        let obj_type_id = object.ty;
                        if self.is_cstruct_class(obj_type_id) {
                            if let Some(layout) = self.get_or_compute_cstruct_layout(obj_type_id) {
                                let field_layout = layout
                                    .fields
                                    .iter()
                                    .find(|f| f.symbol_id == *field)
                                    .or_else(|| {
                                        let fname = self
                                            .symbol_table
                                            .get_symbol(*field)
                                            .and_then(|s| self.string_interner.get(s.name));
                                        fname.and_then(|n| {
                                            layout.fields.iter().find(|f| f.name == n)
                                        })
                                    });
                                if let Some(fl) = field_layout {
                                    let offset_const = self
                                        .builder
                                        .build_const(IrValue::I64(fl.byte_offset as i64));
                                    if let Some(offset_const) = offset_const {
                                        let field_ptr = self.builder.build_ptr_add(
                                            obj_reg,
                                            offset_const,
                                            IrType::Ptr(Box::new(IrType::U8)),
                                        );
                                        if let Some(field_ptr) = field_ptr {
                                            self.builder.build_store(field_ptr, value);
                                        }
                                    }
                                    return;
                                }
                            }
                        }

                        // @:gpuStruct: byte-offset PtrAdd with f64→f32 truncation on write
                        if self.is_gpu_struct_class(obj_type_id) {
                            if let Some(layout) = self.get_or_compute_gpu_struct_layout(obj_type_id)
                            {
                                let field_layout = layout
                                    .fields
                                    .iter()
                                    .find(|f| f.symbol_id == *field)
                                    .or_else(|| {
                                        let fname = self
                                            .symbol_table
                                            .get_symbol(*field)
                                            .and_then(|s| self.string_interner.get(s.name));
                                        fname.and_then(|n| {
                                            layout.fields.iter().find(|f| f.name == n)
                                        })
                                    })
                                    .cloned();
                                if let Some(fl) = field_layout {
                                    let offset_const = self
                                        .builder
                                        .build_const(IrValue::I64(fl.byte_offset as i64));
                                    if let Some(offset_const) = offset_const {
                                        let field_ptr = self.builder.build_ptr_add(
                                            obj_reg,
                                            offset_const,
                                            IrType::Ptr(Box::new(IrType::U8)),
                                        );
                                        if let Some(field_ptr) = field_ptr {
                                            // GPU structs use f32 for Float — truncate f64→f32 on write
                                            let store_val = if fl.ir_type == IrType::F32 {
                                                self.builder
                                                    .build_cast(value, IrType::F64, IrType::F32)
                                                    .unwrap_or(value)
                                            } else if fl.ir_type == IrType::I32 {
                                                self.builder
                                                    .build_cast(value, IrType::I64, IrType::I32)
                                                    .unwrap_or(value)
                                            } else {
                                                value
                                            };
                                            self.builder.build_store(field_ptr, store_val);
                                        }
                                    }
                                    return;
                                }
                            }
                        }

                        if let Some(index_const) =
                            self.builder.build_const(IrValue::I32(field_index as i32))
                        {
                            // The field's declared type in the symbol table wins; the
                            // name scan below only runs when it converts to void*.
                            let field_ty = self.symbol_table.get_symbol(*field)
                                .and_then(|s| {
                                    let converted = self.convert_type(s.type_id);
                                    if matches!(&converted, IrType::Ptr(inner) if matches!(**inner, IrType::Void)) {
                                        None
                                    } else {
                                        Some(converted)
                                    }
                                })
                                .unwrap_or_else(|| {
                                    let field_name = self.symbol_table.get_symbol(*field).map(|s| s.name);
                                    for (sym, _) in &self.field_index_map {
                                        if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                            if field_name == Some(sym_info.name) {
                                                return self.convert_type(sym_info.type_id);
                                            }
                                        }
                                    }
                                    IrType::I32
                                });

                            // Build struct context for typed field access
                            let store_struct_ctx = self.field_class_names.get(field).map(|cn| {
                                crate::ir::instructions::StructFieldRef {
                                    struct_name: cn.clone(),
                                    field_name: self
                                        .symbol_table
                                        .get_symbol(*field)
                                        .and_then(|s| self.string_interner.get(s.name))
                                        .unwrap_or("<unknown>")
                                        .to_string(),
                                    field_index: field_index as u32,
                                }
                            });
                            if let Some(field_ptr) = self.builder.build_gep_with_context(
                                obj_reg,
                                vec![index_const],
                                field_ty.clone(),
                                store_struct_ctx,
                            ) {
                                // Type erasure coercion: if field is I64 (erased type param)
                                // but value is a concrete type, coerce before storing
                                let store_value = if field_ty == IrType::I64 {
                                    if let Some(val_ty) = self.builder.get_register_type(value) {
                                        if val_ty != IrType::I64 {
                                            match val_ty {
                                                IrType::F64 | IrType::F32 => self
                                                    .builder
                                                    .build_bitcast(value, IrType::I64)
                                                    .unwrap_or(value),
                                                // A `*void` source is either a boxed scalar
                                                // or a reference: the same lowered store
                                                // serves every instantiation of an erased
                                                // type parameter. Unwrap only a real scalar
                                                // box and let a reference keep its bits;
                                                // casting the pointer would store the box
                                                // address as the value.
                                                IrType::Ptr(ref inner)
                                                    if matches!(**inner, IrType::Void) =>
                                                {
                                                    let coerce = self
                                                        .get_or_register_extern_function(
                                                            "haxe_unbox_scalar_or_addr",
                                                            vec![IrType::Ptr(Box::new(
                                                                IrType::Void,
                                                            ))],
                                                            IrType::I64,
                                                        );
                                                    self.builder
                                                        .build_call_direct(
                                                            coerce,
                                                            vec![value],
                                                            IrType::I64,
                                                        )
                                                        .unwrap_or(value)
                                                }
                                                _ => self
                                                    .builder
                                                    .build_cast(value, val_ty, IrType::I64)
                                                    .unwrap_or(value),
                                            }
                                        } else {
                                            value
                                        }
                                    } else {
                                        value
                                    }
                                } else {
                                    value
                                };
                                self.builder.build_store(field_ptr, store_value);
                            }
                        }
                    } else {
                        // Dynamic field WRITE fallback via Reflect API.
                        // If the object is Dynamic (anonymous object boxed as Dynamic),
                        // unbox and use haxe_reflect_set_field.
                        let is_dynamic = {
                            let type_table = self.type_table;
                            type_table
                                .get(object.ty)
                                .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                .unwrap_or(false)
                        };
                        if is_dynamic {
                            let field_name_str = self
                                .symbol_table
                                .get_symbol(*field)
                                .and_then(|s| self.string_interner.get(s.name))
                                .map(|s| s.to_string());
                            if let Some(fname) = field_name_str {
                                let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
                                // Unbox Dynamic to get the anonymous object handle.
                                let unbox_ref_id = self.get_or_register_extern_function(
                                    "haxe_unbox_reference_ptr",
                                    vec![ptr_u8.clone()],
                                    ptr_u8.clone(),
                                );
                                if let Some(handle) = self.builder.build_call_direct(
                                    unbox_ref_id,
                                    vec![obj_reg],
                                    ptr_u8.clone(),
                                ) {
                                    // Box the value based on its IR type
                                    let value_ir_type = self.builder.get_register_type(value);
                                    let boxed_value = match &value_ir_type {
                                        Some(IrType::F64) | Some(IrType::F32) => {
                                            let box_id = self.get_or_register_extern_function(
                                                "haxe_box_float_ptr",
                                                vec![IrType::F64],
                                                ptr_u8.clone(),
                                            );
                                            self.builder.build_call_direct(
                                                box_id,
                                                vec![value],
                                                ptr_u8.clone(),
                                            )
                                        }
                                        Some(IrType::Bool) => {
                                            let box_id = self.get_or_register_extern_function(
                                                "haxe_box_bool_ptr",
                                                vec![IrType::Bool],
                                                ptr_u8.clone(),
                                            );
                                            self.builder.build_call_direct(
                                                box_id,
                                                vec![value],
                                                ptr_u8.clone(),
                                            )
                                        }
                                        Some(IrType::Ptr(_)) => {
                                            // Already a pointer — pass through as-is
                                            Some(value)
                                        }
                                        _ => {
                                            // Int or other integer types
                                            let box_id = self.get_or_register_extern_function(
                                                "haxe_box_int_ptr",
                                                vec![IrType::I64],
                                                ptr_u8.clone(),
                                            );
                                            self.builder.build_call_direct(
                                                box_id,
                                                vec![value],
                                                ptr_u8.clone(),
                                            )
                                        }
                                    };
                                    if let Some(boxed) = boxed_value {
                                        let field_name_reg =
                                            self.builder.build_const(IrValue::String(fname));
                                        if let Some(field_name_reg) = field_name_reg {
                                            let set_field_id = self
                                                .get_or_register_extern_function(
                                                    "haxe_reflect_set_field",
                                                    vec![ptr_u8.clone(), ptr_u8.clone(), ptr_u8],
                                                    IrType::Void,
                                                );
                                            self.builder.build_call_direct(
                                                set_field_id,
                                                vec![handle, field_name_reg, boxed],
                                                IrType::Void,
                                            );
                                        }
                                    }
                                }
                                return;
                            }
                        }
                        // Check if this is a static field that should be written as a global
                        let field_name_str = self
                            .symbol_table
                            .get_symbol(*field)
                            .and_then(|s| self.string_interner.get(s.name))
                            .unwrap_or("<unknown>");
                        let global_lookup =
                            self.global_symbol_map.get(field).copied().or_else(|| {
                                self.builder
                                    .module
                                    .globals
                                    .values()
                                    .find(|g| g.name.ends_with(&format!(".{}", field_name_str)))
                                    .map(|g| g.id)
                            });
                        if let Some(global_id) = global_lookup {
                            self.builder.build_store_global(global_id, value);
                            return;
                        }
                        let field_name = self
                            .symbol_table
                            .get_symbol(*field)
                            .map(|s| format!("{}", s.name))
                            .unwrap_or_else(|| format!("{:?}", field));
                        self.add_error(
                            &format!("Field '{}' ({:?}) index not found for write - class may not be registered", field_name, field),
                            SourceLocation::unknown()
                        );
                    }
                }
            }
            HirLValue::Index { object, index } => {
                // Typed array setters pass the value directly: boxing through
                // haxe_box_*_ptr would store the DynamicValue tag, not the value.
                if let Some(obj_reg) = self.lower_expression(object) {
                    if let Some(idx_reg) = self.lower_expression(index) {
                        let idx_i64 = {
                            let idx_ty = self.builder.get_register_type(idx_reg);
                            match idx_ty {
                                Some(IrType::I64) => idx_reg,
                                Some(IrType::I32) => self
                                    .builder
                                    .build_cast(idx_reg, IrType::I32, IrType::I64)
                                    .unwrap_or(idx_reg),
                                Some(IrType::Bool) => self
                                    .builder
                                    .build_cast(idx_reg, IrType::Bool, IrType::I64)
                                    .unwrap_or(idx_reg),
                                Some(IrType::Ptr(_)) => self
                                    .builder
                                    .build_bitcast(idx_reg, IrType::I64)
                                    .unwrap_or(idx_reg),
                                Some(other) => self
                                    .builder
                                    .build_cast(idx_reg, other, IrType::I64)
                                    .unwrap_or(idx_reg),
                                None => idx_reg,
                            }
                        };

                        let value_ir_type = self.builder.get_register_type(value);
                        match &value_ir_type {
                            Some(IrType::F32) | Some(IrType::F64) => {
                                let func_id = self.get_or_register_extern_function(
                                    "haxe_array_set_f64",
                                    vec![
                                        IrType::Ptr(Box::new(IrType::Void)),
                                        IrType::I64,
                                        IrType::F64,
                                    ],
                                    IrType::Bool,
                                );
                                let value_f64 = match value_ir_type.clone() {
                                    Some(IrType::F64) => value,
                                    Some(IrType::F32) => self
                                        .builder
                                        .build_cast(value, IrType::F32, IrType::F64)
                                        .unwrap_or(value),
                                    Some(other) => self
                                        .builder
                                        .build_cast(value, other, IrType::F64)
                                        .unwrap_or(value),
                                    None => value,
                                };
                                self.builder.build_call_direct(
                                    func_id,
                                    vec![obj_reg, idx_i64, value_f64],
                                    IrType::Bool,
                                );
                            }
                            _ => {
                                // For Int, Bool, null, pointers: use haxe_array_set_i64
                                // null=0, bool=0/1, pointers=address - all fit in i64
                                let func_id = self.get_or_register_extern_function(
                                    "haxe_array_set_i64",
                                    vec![
                                        IrType::Ptr(Box::new(IrType::Void)),
                                        IrType::I64,
                                        IrType::I64,
                                    ],
                                    IrType::Bool,
                                );
                                let value_i64 = match value_ir_type.clone() {
                                    Some(IrType::I64) => value,
                                    Some(IrType::I32) => self
                                        .builder
                                        .build_cast(value, IrType::I32, IrType::I64)
                                        .unwrap_or(value),
                                    Some(IrType::Bool) => self
                                        .builder
                                        .build_cast(value, IrType::Bool, IrType::I64)
                                        .unwrap_or(value),
                                    Some(IrType::Ptr(_)) => self
                                        .builder
                                        .build_bitcast(value, IrType::I64)
                                        .unwrap_or(value),
                                    Some(IrType::F64) => self
                                        .builder
                                        .build_bitcast(value, IrType::I64)
                                        .unwrap_or(value),
                                    Some(IrType::F32) => {
                                        let as_i32 = self
                                            .builder
                                            .build_bitcast(value, IrType::I32)
                                            .unwrap_or(value);
                                        self.builder
                                            .build_cast(as_i32, IrType::I32, IrType::I64)
                                            .unwrap_or(as_i32)
                                    }
                                    Some(other) => self
                                        .builder
                                        .build_cast(value, other, IrType::I64)
                                        .unwrap_or(value),
                                    None => value,
                                };
                                self.builder.build_call_direct(
                                    func_id,
                                    vec![obj_reg, idx_i64, value_i64],
                                    IrType::Bool,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
