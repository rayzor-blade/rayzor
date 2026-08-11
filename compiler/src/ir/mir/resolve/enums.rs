//! Resolves enum symbols, variant discriminants, and payload types.

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
    /// Resolve the discriminant value for an enum variant by name.
    /// Looks up the enum type → symbol → variants, then finds the matching variant index.
    pub(crate) fn resolve_enum_variant_discriminant(
        &self,
        enum_type: TypeId,
        variant_name: InternedString,
    ) -> Option<i64> {
        let enum_symbol = self
            .symbol_table
            .get_symbol_from_type(enum_type)
            .or_else(|| {
                let type_table = self.type_table;
                if let Some(type_info) = type_table.get(enum_type) {
                    match &type_info.kind {
                        crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                            return self.symbol_table.get_symbol_from_type(*base_type);
                        }
                        crate::tast::TypeKind::Enum { symbol_id, .. } => {
                            return Some(*symbol_id);
                        }
                        _ => {}
                    }
                }
                None
            });

        if let Some(enum_sym) = enum_symbol {
            if let Some(variants) = self.symbol_table.get_enum_variants(enum_sym) {
                for (idx, &variant_id) in variants.iter().enumerate() {
                    if let Some(variant_sym) = self.symbol_table.get_symbol(variant_id) {
                        if variant_sym.name == variant_name {
                            return Some(idx as i64);
                        }
                    }
                }
            }
        }

        // Fallback for an enum_type that maps to no symbol: scan every enum
        // variant in the symbol table for this name.
        let variant_name_str = self.string_interner.get(variant_name)?;
        for sym in self.symbol_table.all_symbols() {
            if sym.kind == crate::tast::symbols::SymbolKind::EnumVariant {
                let sym_name = self.string_interner.get(sym.name).unwrap_or("");
                if sym_name == variant_name_str {
                    if let Some(parent_id) =
                        self.symbol_table.find_parent_enum_for_constructor(sym.id)
                    {
                        if let Some(variants) = self.symbol_table.get_enum_variants(parent_id) {
                            for (idx, &vid) in variants.iter().enumerate() {
                                if vid == sym.id {
                                    return Some(idx as i64);
                                }
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Resolve an enum TypeId to its SymbolId
    pub(crate) fn resolve_enum_symbol(&self, enum_type: TypeId) -> Option<SymbolId> {
        self.symbol_table
            .get_symbol_from_type(enum_type)
            .or_else(|| {
                let type_table = self.type_table;
                if let Some(type_info) = type_table.get(enum_type) {
                    match &type_info.kind {
                        crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                            return self.symbol_table.get_symbol_from_type(*base_type);
                        }
                        crate::tast::TypeKind::Enum { symbol_id, .. } => {
                            return Some(*symbol_id);
                        }
                        _ => {}
                    }
                }
                None
            })
    }

    /// Resolve the concrete IrTypes and TypeIds for an enum variant's fields.
    /// Handles generic type parameter substitution (e.g., Result<Int,String>.Ok → [(I32, IntTypeId)]).
    pub(crate) fn get_enum_variant_field_types(
        &self,
        enum_type: TypeId,
        variant_name: InternedString,
    ) -> Vec<(IrType, TypeId)> {
        // Step 1: Resolve enum_type to base enum SymbolId and optional generic type_args
        let (enum_symbol, generic_type_args) = {
            let type_table = self.type_table;
            if let Some(type_info) = type_table.get(enum_type) {
                match &type_info.kind {
                    crate::tast::TypeKind::GenericInstance {
                        base_type,
                        type_args,
                        ..
                    } => {
                        // Generic enum: Result<Int, String>
                        let base_sym =
                            self.symbol_table
                                .get_symbol_from_type(*base_type)
                                .or_else(|| {
                                    if let Some(base_info) = type_table.get(*base_type) {
                                        if let crate::tast::TypeKind::Enum { symbol_id, .. } =
                                            &base_info.kind
                                        {
                                            return Some(*symbol_id);
                                        }
                                    }
                                    None
                                });
                        (base_sym, Some((*base_type, type_args.clone())))
                    }
                    crate::tast::TypeKind::Enum {
                        symbol_id,
                        type_args,
                        ..
                    } => {
                        // Non-TypeParameter type_args mean this is already the
                        // concrete instance, so the base enum must be located.
                        let has_concrete_args = !type_args.is_empty()
                            && type_args.iter().any(|ta| {
                                type_table
                                    .get(*ta)
                                    .map(|ti| {
                                        !matches!(
                                            ti.kind,
                                            crate::tast::TypeKind::TypeParameter { .. }
                                        )
                                    })
                                    .unwrap_or(false)
                            });
                        if has_concrete_args {
                            // Find the base enum type (with TypeParameter type_args) via symbol
                            let base_type = self
                                .symbol_table
                                .get_symbol(*symbol_id)
                                .map(|sym| sym.type_id)
                                .unwrap_or(enum_type);
                            (Some(*symbol_id), Some((base_type, type_args.clone())))
                        } else {
                            (Some(*symbol_id), None)
                        }
                    }
                    _ => {
                        let sym = self.symbol_table.get_symbol_from_type(enum_type);
                        (sym, None)
                    }
                }
            } else {
                (self.symbol_table.get_symbol_from_type(enum_type), None)
            }
        };

        let enum_sym = match enum_symbol {
            Some(s) => s,
            None => {
                return vec![];
            }
        };

        // Step 2: Find the variant by name and get its field TypeIds from HIR
        let variant_name_str = self.string_interner.get(variant_name).unwrap_or("");
        for (_type_id, type_decl) in self.current_hir_types.iter() {
            if let HirTypeDecl::Enum(enum_decl) = type_decl {
                if enum_decl.symbol_id == enum_sym {
                    for variant in &enum_decl.variants {
                        let vname = self.string_interner.get(variant.name).unwrap_or("");
                        if vname == variant_name_str {
                            return variant
                                .fields
                                .iter()
                                .map(|field| {
                                    let ir_type =
                                        self.resolve_field_type(field.ty, &generic_type_args);
                                    let type_id =
                                        self.resolve_field_type_id(field.ty, &generic_type_args);
                                    (ir_type, type_id)
                                })
                                .collect();
                        }
                    }
                }
            }
        }

        // Step 3: Fallback — symbol table lookup for imported enums
        if let Some(variants) = self.symbol_table.get_enum_variants(enum_sym) {
            for &variant_id in variants.iter() {
                if let Some(variant_sym) = self.symbol_table.get_symbol(variant_id) {
                    if variant_sym.name == variant_name {
                        let type_table = self.type_table;
                        if let Some(type_info) = type_table.get(variant_sym.type_id) {
                            if let crate::tast::core::TypeKind::Function { params, .. } =
                                &type_info.kind
                            {
                                return params
                                    .iter()
                                    .map(|p| {
                                        let ir_type =
                                            self.resolve_field_type(*p, &generic_type_args);
                                        let type_id =
                                            self.resolve_field_type_id(*p, &generic_type_args);
                                        (ir_type, type_id)
                                    })
                                    .collect();
                            }
                        }
                    }
                }
            }
        }

        vec![]
    }

    pub(crate) fn get_enum_variant_field_count(
        &self,
        enum_symbol: SymbolId,
        variant_idx: usize,
    ) -> usize {
        // First check current module's HIR types
        for (_type_id, type_decl) in self.current_hir_types.iter() {
            if let HirTypeDecl::Enum(enum_decl) = type_decl {
                if enum_decl.symbol_id == enum_symbol {
                    if let Some(variant) = enum_decl.variants.get(variant_idx) {
                        return variant.fields.len();
                    }
                }
            }
        }
        // Fallback: check symbol table for enums from other modules (e.g. StdTypes.hx)
        if let Some(variants) = self.symbol_table.get_enum_variants(enum_symbol) {
            if let Some(&variant_id) = variants.get(variant_idx) {
                if let Some(variant_sym) = self.symbol_table.get_symbol(variant_id) {
                    let type_table = self.type_table;
                    if let Some(type_info) = type_table.get(variant_sym.type_id) {
                        if let crate::tast::core::TypeKind::Function { params, .. } =
                            &type_info.kind
                        {
                            return params.len();
                        }
                    }
                }
            }
        }
        0 // Default: no fields
    }
}
