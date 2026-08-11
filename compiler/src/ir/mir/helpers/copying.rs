//! Value-semantics copies: shallow copy and anonymous-structure cloning.

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
    /// @:derive(Copy) — emit a shallow (bitwise) copy of a class instance.
    /// All fields must be primitives (enforced at validation time).
    pub(crate) fn emit_shallow_copy(&mut self, src_reg: IrId, class_sym: SymbolId) -> Option<IrId> {
        let fields = self.class_instance_fields.get(&class_sym)?.clone();

        // Allocate: (num_fields + 1) * 8  (slot 0 = type_id header)
        let alloc_size = ((fields.len() as u64 + 1) * 8).max(16);
        let new_ptr = self.build_heap_alloc(alloc_size)?;

        // Copy type_id header (GEP index 0)
        let zero = self.builder.build_const(IrValue::I32(0))?;
        let header_ptr = self.builder.build_gep(src_reg, vec![zero], IrType::I64)?;
        let header_val = self.builder.build_load(header_ptr, IrType::I64)?;
        let zero2 = self.builder.build_const(IrValue::I32(0))?;
        let new_header_ptr = self.builder.build_gep(new_ptr, vec![zero2], IrType::I64)?;
        self.builder.build_store(new_header_ptr, header_val)?;

        // Copy each field (all primitives — bitwise copy)
        for &(_field_sym, field_type_id, gep_idx) in &fields {
            let field_ir_type = self.convert_type(field_type_id);
            let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let src_field_ptr =
                self.builder
                    .build_gep(src_reg, vec![idx_const], field_ir_type.clone())?;
            let field_val = self
                .builder
                .build_load(src_field_ptr, field_ir_type.clone())?;
            let dst_idx = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let dst_field_ptr = self
                .builder
                .build_gep(new_ptr, vec![dst_idx], field_ir_type)?;
            self.builder.build_store(dst_field_ptr, field_val)?;
        }

        // Set class hint for field access on copied value
        if let Some(sym) = self.symbol_table.get_symbol(class_sym) {
            let name = self.string_interner.get(sym.name).unwrap_or("").to_string();
            if !name.is_empty() {
                self.register_class_hints.insert(new_ptr, name);
            }
        }

        Some(new_ptr)
    }

    /// Clone an anonymous object handle if the value type is Anonymous.
    /// This ensures each variable gets its own Box<Arc<AnonObject>> for proper COW semantics.
    /// Returns the cloned handle, or the original value if not anonymous.
    pub(crate) fn maybe_clone_anonymous(&mut self, value: IrId, value_ty: TypeId) -> IrId {
        use crate::tast::TypeKind;

        let is_anon = {
            let type_table = self.type_table;
            matches!(
                type_table.get(value_ty).map(|t| &t.kind),
                Some(TypeKind::Anonymous { .. })
            )
        };

        if !is_anon {
            return value;
        }

        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));
        let clone_func_id =
            self.get_or_register_extern_function("rayzor_anon_clone", vec![ptr_u8.clone()], ptr_u8);

        self.builder
            .build_call_direct(
                clone_func_id,
                vec![value],
                IrType::Ptr(Box::new(IrType::U8)),
            )
            .unwrap_or(value)
    }

    /// Try to register a structural subtyping view for an anonymous-typed variable.
    /// If source is a class or wider anon and target is a narrower anonymous type,
    /// records an AnonBacking so field access can be redirected at compile time.
    /// Returns true if a view was registered (caller should skip clone).
    pub(crate) fn try_register_anon_view(
        &mut self,
        symbol: SymbolId,
        _value: IrId,
        source_type: TypeId,
        target_type: TypeId,
    ) -> bool {
        let resolved_source = self.resolve_through_aliases(source_type);
        let resolved_target = self.resolve_through_aliases(target_type);

        // Target must be Anonymous
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
                    return false;
                }
            } else {
                return false;
            }
        };

        if target_fields.is_empty() {
            return false;
        }

        let source_kind = {
            let type_table = self.type_table;
            type_table.get(resolved_source).map(|t| t.kind.clone())
        };

        // Extract class symbol if source is a class or generic instance of a class
        let class_symbol_opt = match &source_kind {
            Some(TypeKind::Class { symbol_id, .. }) => Some(*symbol_id),
            Some(TypeKind::GenericInstance { base_type, .. }) => {
                let type_table = self.type_table;
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
        };

        if let Some(class_symbol) = class_symbol_opt {
            // Source is a class — build name → (gep_index, type) map from field_index_map
            let mut class_field_by_name: BTreeMap<String, (u32, TypeId)> = BTreeMap::new();
            // Fast path: lookup by TypeId via reverse index
            let mut fields = self.fields_for_type(resolved_source);
            // Fallback: match by class symbol_id. The map is keyed by context-local
            // TypeIds, so confirm against the field's declaring class name where
            // recorded — otherwise another class's fields enter this map and, being
            // keyed by field name, overwrite the right indices.
            if fields.is_empty() {
                let expected = self.class_qualified_name(class_symbol);
                for (field_sym, (class_ty, idx)) in &self.field_index_map {
                    if self
                        .class_type_to_symbol
                        .get(class_ty)
                        .map_or(false, |s| *s == class_symbol)
                    {
                        if let (Some(owner), Some(expected)) =
                            (self.field_class_names.get(field_sym), expected.as_deref())
                        {
                            if !Self::class_names_match(owner, expected) {
                                continue;
                            }
                        }
                        fields.push((*field_sym, *idx));
                    }
                }
            }
            for (field_sym, idx) in &fields {
                if let Some(sym) = self.symbol_table.get_symbol(*field_sym) {
                    if let Some(name) = self.string_interner.get(sym.name) {
                        class_field_by_name.insert(name.to_string(), (*idx, sym.type_id));
                    }
                }
            }

            // Check all target fields exist in the class
            let mut field_map = Vec::new();
            for (tgt_name, tgt_ty) in &target_fields {
                if let Some((gep_idx, _field_ty)) = class_field_by_name.get(tgt_name) {
                    field_map.push((tgt_name.clone(), *gep_idx, *tgt_ty));
                } else {
                    return false; // Target field not found in class
                }
            }

            self.anon_views.insert(
                symbol,
                AnonBacking::Class {
                    class_symbol,
                    class_type: resolved_source,
                    field_map,
                },
            );
            return true;
        }

        // Source is anonymous — check for wider-to-narrower
        if let Some(TypeKind::Anonymous { fields: src_fields }) = source_kind {
            let mut src_named: Vec<(String, TypeId)> = src_fields
                .iter()
                .filter_map(|f| {
                    self.string_interner
                        .get(f.name)
                        .map(|s| (s.to_string(), f.type_id))
                })
                .collect();
            src_named.sort_by(|a, b| a.0.cmp(&b.0));

            // If same field names, no remapping needed
            let src_names: Vec<&str> = src_named.iter().map(|(n, _)| n.as_str()).collect();
            let tgt_names: Vec<&str> = target_fields.iter().map(|(n, _)| n.as_str()).collect();
            if src_names == tgt_names {
                return false;
            }

            // Check all target fields exist in source, build remap
            let mut field_map = Vec::new();
            for (tgt_name, tgt_ty) in &target_fields {
                if let Some(src_idx) = src_named.iter().position(|(n, _)| n == tgt_name) {
                    field_map.push((tgt_name.clone(), src_idx, *tgt_ty));
                } else {
                    return false; // Target field not found in source
                }
            }

            self.anon_views.insert(
                symbol,
                AnonBacking::WiderAnon {
                    source_type: resolved_source,
                    field_map,
                },
            );
            return true;
        }

        false
    }
}
