//! Anonymous-structure views: class-to-anon and anon-to-wider-anon adapters.

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
    /// Materialize an anon-backed variable into a real AnonObject handle.
    /// Called at escape points (function call args, return) where the callee
    /// expects a real `rayzor_anon_*`-compatible handle.
    pub(crate) fn materialize_anon_view(
        &mut self,
        value: IrId,
        symbol: SymbolId,
        target_type: TypeId,
    ) -> Option<IrId> {
        let backing = self.anon_views.get(&symbol)?.clone();

        // Get target anonymous fields sorted alphabetically
        let resolved_target = self.resolve_through_aliases(target_type);
        let target_fields = {
            let type_table = self.type_table;
            if let Some(ty_info) = type_table.get(resolved_target) {
                if let TypeKind::Anonymous { fields } = &ty_info.kind {
                    let mut named: Vec<(String, TypeId)> = fields
                        .iter()
                        .filter_map(|f| {
                            self.string_interner
                                .get(f.name)
                                .map(|s| (s.to_string(), f.type_id))
                        })
                        .collect();
                    named.sort_by(|a, b| a.0.cmp(&b.0));
                    named
                } else {
                    return None;
                }
            } else {
                return None;
            }
        };

        let total_field_count = target_fields.len();

        // Build shape key and get/create shape ID
        let shape_key: String = target_fields
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

        // Build descriptor string
        let descriptor = {
            let mut parts = Vec::with_capacity(target_fields.len());
            for (name, type_id) in &target_fields {
                let runtime_tid = self.runtime_type_id(*type_id);
                parts.push(format!("{}:{}", name, runtime_tid));
            }
            parts.join(",")
        };

        // Emit rayzor_ensure_shape + rayzor_anon_new
        let ensure_shape_id = self.get_or_register_extern_function(
            "rayzor_ensure_shape",
            vec![IrType::I32, IrType::String],
            IrType::Void,
        );
        let anon_new_id = self.get_or_register_extern_function(
            "rayzor_anon_new",
            vec![IrType::I32, IrType::I32],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let anon_set_id = self.get_or_register_extern_function(
            "rayzor_anon_set_field_by_index",
            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32, IrType::I64],
            IrType::Void,
        );

        let shape_id_const = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let desc_str = self.builder.build_const(IrValue::String(descriptor))?;
        self.builder.build_call_direct(
            ensure_shape_id,
            vec![shape_id_const, desc_str],
            IrType::Void,
        );

        let shape_id_val = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let field_count_val = self
            .builder
            .build_const(IrValue::I32(total_field_count as i32))?;
        let handle = self.builder.build_call_direct(
            anon_new_id,
            vec![shape_id_val, field_count_val],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        // Copy fields from backing source into the new AnonObject
        match &backing {
            AnonBacking::Class { field_map, .. } => {
                for (sorted_idx, (name, _)) in target_fields.iter().enumerate() {
                    if let Some((_, gep_idx, field_type_id)) =
                        field_map.iter().find(|(n, ..)| n == name)
                    {
                        let field_ir_ty = self.convert_type(*field_type_id);
                        let idx_const = self.builder.build_const(IrValue::I64(*gep_idx as i64))?;
                        let field_ptr =
                            self.builder
                                .build_gep(value, vec![idx_const], field_ir_ty.clone())?;
                        let field_val = self.builder.build_load(field_ptr, field_ir_ty)?;
                        let val_as_i64 = self.coerce_to_i64(field_val, *field_type_id)?;
                        let idx_val = self.builder.build_const(IrValue::I32(sorted_idx as i32))?;
                        self.builder.build_call_direct(
                            anon_set_id,
                            vec![handle, idx_val, val_as_i64],
                            IrType::Void,
                        );
                    }
                }
            }
            AnonBacking::WiderAnon { field_map, .. } => {
                let anon_get_id = self.get_or_register_extern_function(
                    "rayzor_anon_get_field_by_index",
                    vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
                    IrType::I64,
                );
                for (sorted_idx, (name, _)) in target_fields.iter().enumerate() {
                    if let Some((_, src_idx, _field_type_id)) =
                        field_map.iter().find(|(n, ..)| n == name)
                    {
                        let src_idx_val =
                            self.builder.build_const(IrValue::I32(*src_idx as i32))?;
                        let raw_val = self.builder.build_call_direct(
                            anon_get_id,
                            vec![value, src_idx_val],
                            IrType::I64,
                        )?;
                        let idx_val = self.builder.build_const(IrValue::I32(sorted_idx as i32))?;
                        self.builder.build_call_direct(
                            anon_set_id,
                            vec![handle, idx_val, raw_val],
                            IrType::Void,
                        );
                    }
                }
            }
        }

        Some(handle)
    }

    /// Materialize a class-typed value into an AnonObject for a callee that expects anonymous type.
    pub(crate) fn materialize_class_to_anon(
        &mut self,
        class_ptr: IrId,
        class_type: TypeId,
        target_anon_type: TypeId,
    ) -> Option<IrId> {
        // Get class symbol to look up fields
        let class_symbol = {
            let type_table = self.type_table;
            if let Some(ty_info) = type_table.get(class_type) {
                match &ty_info.kind {
                    TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    TypeKind::GenericInstance { base_type, .. } => {
                        if let Some(base_info) = type_table.get(*base_type) {
                            if let TypeKind::Class { symbol_id, .. } = &base_info.kind {
                                Some(*symbol_id)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        }?;

        // Get target anonymous fields
        let target_fields = {
            let type_table = self.type_table;
            if let Some(ty_info) = type_table.get(target_anon_type) {
                if let TypeKind::Anonymous { fields } = &ty_info.kind {
                    let mut named: Vec<(String, TypeId)> = fields
                        .iter()
                        .filter_map(|f| {
                            self.string_interner
                                .get(f.name)
                                .map(|s| (s.to_string(), f.type_id))
                        })
                        .collect();
                    named.sort_by(|a, b| a.0.cmp(&b.0));
                    named
                } else {
                    return None;
                }
            } else {
                return None;
            }
        };

        // Build field_map from field_index_map (same approach as try_register_anon_view).
        // Both id tests are context-local, so confirm with the field's declaring
        // class NAME where recorded: entries are keyed by field name here, so a
        // foreign class's same-named field would overwrite the correct index.
        let mut class_field_by_name: BTreeMap<String, (u32, TypeId)> = BTreeMap::new();
        let expected_owner = self.class_qualified_name(class_symbol);
        for (field_sym, (fclass_ty, idx)) in &self.field_index_map {
            let matches = *fclass_ty == class_type || {
                self.class_type_to_symbol
                    .get(fclass_ty)
                    .map_or(false, |s| *s == class_symbol)
            };
            if matches {
                if let (Some(owner), Some(expected)) = (
                    self.field_class_names.get(field_sym),
                    expected_owner.as_deref(),
                ) {
                    if !Self::class_names_match(owner, expected) {
                        continue;
                    }
                }
                if let Some(sym) = self.symbol_table.get_symbol(*field_sym) {
                    if let Some(name) = self.string_interner.get(sym.name) {
                        class_field_by_name.insert(name.to_string(), (*idx, sym.type_id));
                    }
                }
            }
        }

        let mut field_map: Vec<(String, u32, TypeId)> = Vec::new();
        for (name, target_field_ty) in &target_fields {
            if let Some((gep_idx, _field_ty)) = class_field_by_name.get(name) {
                field_map.push((name.clone(), *gep_idx, *target_field_ty));
            } else {
                return None; // Target field not found in class
            }
        }

        // Now build a temporary AnonBacking::Class and call materialize_anon_view logic
        // We'll inline the materialization here to avoid needing a temporary symbol
        let total_field_count = target_fields.len();

        let shape_key: String = target_fields
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

        let descriptor = {
            let mut parts = Vec::with_capacity(target_fields.len());
            for (name, type_id) in &target_fields {
                let runtime_tid = self.runtime_type_id(*type_id);
                parts.push(format!("{}:{}", name, runtime_tid));
            }
            parts.join(",")
        };

        let ensure_shape_id = self.get_or_register_extern_function(
            "rayzor_ensure_shape",
            vec![IrType::I32, IrType::String],
            IrType::Void,
        );
        let anon_new_id = self.get_or_register_extern_function(
            "rayzor_anon_new",
            vec![IrType::I32, IrType::I32],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let anon_set_id = self.get_or_register_extern_function(
            "rayzor_anon_set_field_by_index",
            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32, IrType::I64],
            IrType::Void,
        );

        let shape_id_const = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let desc_str = self.builder.build_const(IrValue::String(descriptor))?;
        self.builder.build_call_direct(
            ensure_shape_id,
            vec![shape_id_const, desc_str],
            IrType::Void,
        );

        let shape_id_val = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let field_count_val = self
            .builder
            .build_const(IrValue::I32(total_field_count as i32))?;
        let handle = self.builder.build_call_direct(
            anon_new_id,
            vec![shape_id_val, field_count_val],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        // Copy fields from class via GEP+Load
        for (sorted_idx, (name, _)) in target_fields.iter().enumerate() {
            if let Some((_, gep_idx, field_type_id)) = field_map.iter().find(|(n, ..)| n == name) {
                let field_ir_ty = self.convert_type(*field_type_id);
                let idx_const = self.builder.build_const(IrValue::I64(*gep_idx as i64))?;
                let field_ptr =
                    self.builder
                        .build_gep(class_ptr, vec![idx_const], field_ir_ty.clone())?;
                let field_val = self.builder.build_load(field_ptr, field_ir_ty)?;
                let val_as_i64 = self.coerce_to_i64(field_val, *field_type_id)?;
                let idx_val = self.builder.build_const(IrValue::I32(sorted_idx as i32))?;
                self.builder.build_call_direct(
                    anon_set_id,
                    vec![handle, idx_val, val_as_i64],
                    IrType::Void,
                );
            }
        }

        Some(handle)
    }

    /// Materialize a wider anonymous object into a narrower anonymous object.
    pub(crate) fn materialize_wider_anon_to_anon(
        &mut self,
        source_handle: IrId,
        source_type: TypeId,
        target_anon_type: TypeId,
    ) -> Option<IrId> {
        // Get source fields sorted alphabetically
        let source_fields = {
            let type_table = self.type_table;
            if let Some(ty_info) = type_table.get(source_type) {
                if let TypeKind::Anonymous { fields } = &ty_info.kind {
                    let mut named: Vec<(String, TypeId)> = fields
                        .iter()
                        .filter_map(|f| {
                            self.string_interner
                                .get(f.name)
                                .map(|s| (s.to_string(), f.type_id))
                        })
                        .collect();
                    named.sort_by(|a, b| a.0.cmp(&b.0));
                    named
                } else {
                    return None;
                }
            } else {
                return None;
            }
        };

        // Get target fields sorted alphabetically
        let target_fields = {
            let type_table = self.type_table;
            if let Some(ty_info) = type_table.get(target_anon_type) {
                if let TypeKind::Anonymous { fields } = &ty_info.kind {
                    let mut named: Vec<(String, TypeId)> = fields
                        .iter()
                        .filter_map(|f| {
                            self.string_interner
                                .get(f.name)
                                .map(|s| (s.to_string(), f.type_id))
                        })
                        .collect();
                    named.sort_by(|a, b| a.0.cmp(&b.0));
                    named
                } else {
                    return None;
                }
            } else {
                return None;
            }
        };

        // Build source index map: field_name → sorted index in source
        let source_index_map: std::collections::BTreeMap<&str, usize> = source_fields
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.as_str(), i))
            .collect();

        // Verify all target fields exist in source
        for (name, _) in &target_fields {
            if !source_index_map.contains_key(name.as_str()) {
                return None;
            }
        }

        let total_field_count = target_fields.len();

        let shape_key: String = target_fields
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

        let descriptor = {
            let mut parts = Vec::with_capacity(target_fields.len());
            for (name, type_id) in &target_fields {
                let runtime_tid = self.runtime_type_id(*type_id);
                parts.push(format!("{}:{}", name, runtime_tid));
            }
            parts.join(",")
        };

        let ensure_shape_id = self.get_or_register_extern_function(
            "rayzor_ensure_shape",
            vec![IrType::I32, IrType::String],
            IrType::Void,
        );
        let anon_new_id = self.get_or_register_extern_function(
            "rayzor_anon_new",
            vec![IrType::I32, IrType::I32],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let anon_set_id = self.get_or_register_extern_function(
            "rayzor_anon_set_field_by_index",
            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32, IrType::I64],
            IrType::Void,
        );
        let anon_get_id = self.get_or_register_extern_function(
            "rayzor_anon_get_field_by_index",
            vec![IrType::Ptr(Box::new(IrType::U8)), IrType::I32],
            IrType::I64,
        );

        let shape_id_const = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let desc_str = self.builder.build_const(IrValue::String(descriptor))?;
        self.builder.build_call_direct(
            ensure_shape_id,
            vec![shape_id_const, desc_str],
            IrType::Void,
        );

        let shape_id_val = self.builder.build_const(IrValue::I32(shape_id as i32))?;
        let field_count_val = self
            .builder
            .build_const(IrValue::I32(total_field_count as i32))?;
        let handle = self.builder.build_call_direct(
            anon_new_id,
            vec![shape_id_val, field_count_val],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        // Copy fields from source anon with remapped indices
        for (sorted_idx, (name, _)) in target_fields.iter().enumerate() {
            if let Some(&src_idx) = source_index_map.get(name.as_str()) {
                let src_idx_val = self.builder.build_const(IrValue::I32(src_idx as i32))?;
                let raw_val = self.builder.build_call_direct(
                    anon_get_id,
                    vec![source_handle, src_idx_val],
                    IrType::I64,
                )?;
                let idx_val = self.builder.build_const(IrValue::I32(sorted_idx as i32))?;
                self.builder.build_call_direct(
                    anon_set_id,
                    vec![handle, idx_val, raw_val],
                    IrType::Void,
                );
            }
        }

        Some(handle)
    }
}
