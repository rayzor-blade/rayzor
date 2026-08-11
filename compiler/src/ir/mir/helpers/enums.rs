//! Enum representation: boxed payloads, runtime ids, hidden type-id argument.

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
    /// Check if an enum has any parameterized variants (requires boxed representation)
    pub(crate) fn enum_is_boxed(&self, enum_symbol: SymbolId) -> bool {
        // First check current module's HIR types
        for (_type_id, type_decl) in self.current_hir_types.iter() {
            if let HirTypeDecl::Enum(enum_decl) = type_decl {
                if enum_decl.symbol_id == enum_symbol {
                    return enum_decl.variants.iter().any(|v| !v.fields.is_empty());
                }
            }
        }
        // Also check by name — generic instantiation may create different SymbolIds
        if let Some(sym) = self.symbol_table.get_symbol(enum_symbol) {
            let enum_name = self.string_interner.get(sym.name).unwrap_or("");
            for (_type_id, type_decl) in self.current_hir_types.iter() {
                if let HirTypeDecl::Enum(enum_decl) = type_decl {
                    let decl_name = self.string_interner.get(enum_decl.name).unwrap_or("");
                    if decl_name == enum_name {
                        return enum_decl.variants.iter().any(|v| !v.fields.is_empty());
                    }
                }
            }
        }
        // Fallback: check symbol table for enums from other modules (e.g. StdTypes.hx)
        if let Some(variants) = self.symbol_table.get_enum_variants(enum_symbol) {
            let type_table = self.type_table;
            for &variant_id in variants {
                if let Some(variant_sym) = self.symbol_table.get_symbol(variant_id) {
                    // A variant with parameters has a Function type
                    if let Some(type_info) = type_table.get(variant_sym.type_id) {
                        if matches!(type_info.kind, crate::tast::core::TypeKind::Function { .. }) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Extract the runtime type_id (as u32) from an expression that represents an enum value.
    /// Stable runtime id for an enum, from its declaration symbol. Mirrors the
    /// `TypeKind::Enum` arm of `runtime_type_id` so a value boxed by type and a
    /// lookup keyed by symbol land on the SAME key. Enum RTTI registration uses
    /// this id too; keeping one key space is what makes reflection on a
    /// Dynamic-typed enum work.
    pub(crate) fn enum_runtime_id(&self, enum_symbol: SymbolId) -> u32 {
        self.deterministic_iface_or_enum_type_id(enum_symbol, "enum")
            .unwrap_or_else(|| {
                self.symbol_table
                    .get_symbol(enum_symbol)
                    .map(|s| s.type_id.0 + 1000)
                    .unwrap_or(0)
            })
    }

    /// Allocate a boxed enum struct with only a tag (no fields).
    /// Used for parameterless variants of enums that have other parameterized variants.
    /// Layout: [tag:i32][pad:i32] = 8 bytes, returned as ptr bitcast to i64.
    pub(crate) fn build_boxed_enum_tag_only(&mut self, tag_idx: i32) -> Option<IrId> {
        let size_const = self.builder.build_const(IrValue::I64(8))?;
        let alloc_func = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::I64],
            IrType::Ptr(Box::new(IrType::I8)),
        );
        let ptr = self.builder.build_call_direct(
            alloc_func,
            vec![size_const],
            IrType::Ptr(Box::new(IrType::I8)),
        )?;
        let zero_offset = self.builder.build_const(IrValue::I64(0))?;
        let tag_gep =
            self.builder
                .build_gep(ptr, vec![zero_offset], IrType::Ptr(Box::new(IrType::I8)))?;
        let tag_ptr = self
            .builder
            .build_bitcast(tag_gep, IrType::Ptr(Box::new(IrType::I32)))?;
        let tag_val = self.builder.build_const(IrValue::I32(tag_idx))?;
        self.builder.build_store(tag_ptr, tag_val)?;
        self.builder.build_bitcast(ptr, IrType::I64)
    }

    /// Allocate a boxed enum struct with payload fields.
    /// Layout: [tag:i32][pad:i32][field0:i64][field1:i64]...
    pub(crate) fn build_boxed_enum_with_fields(
        &mut self,
        tag_idx: i32,
        field_count: usize,
        constructor_args: &[HirExpr],
    ) -> Option<IrId> {
        let struct_size = 8 + 8 * field_count;
        let size_const = self.builder.build_const(IrValue::I64(struct_size as i64))?;
        let alloc_func = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::I64],
            IrType::Ptr(Box::new(IrType::I8)),
        );
        let ptr = self.builder.build_call_direct(
            alloc_func,
            vec![size_const],
            IrType::Ptr(Box::new(IrType::I8)),
        )?;

        let zero_offset = self.builder.build_const(IrValue::I64(0))?;
        let tag_ptr =
            self.builder
                .build_gep(ptr, vec![zero_offset], IrType::Ptr(Box::new(IrType::I8)))?;
        let tag_ptr_i32 = self
            .builder
            .build_bitcast(tag_ptr, IrType::Ptr(Box::new(IrType::I32)))?;
        let tag_val = self.builder.build_const(IrValue::I32(tag_idx))?;
        self.builder.build_store(tag_ptr_i32, tag_val)?;

        for (i, arg) in constructor_args.iter().take(field_count).enumerate() {
            let arg_reg = self.lower_expression(arg)?;
            let field_offset = self.builder.build_const(IrValue::I64((8 + i * 8) as i64))?;
            let field_ptr = self.builder.build_gep(
                ptr,
                vec![field_offset],
                IrType::Ptr(Box::new(IrType::I8)),
            )?;
            let field_ptr_i64 = self
                .builder
                .build_bitcast(field_ptr, IrType::Ptr(Box::new(IrType::I64)))?;
            self.builder.build_store(field_ptr_i64, arg_reg)?;
        }

        self.builder.build_bitcast(ptr, IrType::I64)
    }

    /// Try to resolve enum type id from call arguments for runtime enum helpers.
    /// For `enumEq(a, b)`, prefer the first arg then second arg.
    pub(crate) fn extract_enum_type_id_from_args(
        &self,
        runtime_func: &str,
        args: &[HirExpr],
    ) -> Option<u32> {
        if args.is_empty() {
            return None;
        }

        let preferred_indices: &[usize] = if runtime_func == "haxe_type_enum_eq" {
            &[0, 1]
        } else {
            &[0]
        };

        for &idx in preferred_indices {
            if let Some(arg) = args.get(idx) {
                if let Some(type_id) = self.extract_enum_type_id_from_expr(arg) {
                    return Some(type_id);
                }
            }
        }

        for arg in args {
            if let Some(type_id) = self.extract_enum_type_id_from_expr(arg) {
                return Some(type_id);
            }
        }

        None
    }

    /// Inspects the expression to find the enum variant symbol, then looks up the parent enum.
    pub(crate) fn extract_enum_type_id_from_expr(&self, expr: &HirExpr) -> Option<u32> {
        // First, try to resolve from the expression's type
        let enum_sym = self.resolve_enum_symbol(expr.ty);
        if let Some(sym_id) = enum_sym {
            if self.symbol_table.get_symbol(sym_id).is_some() {
                return Some(self.enum_runtime_id(sym_id));
            }
        }

        // Try to find enum type from the expression kind
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => {
                if let Some(sym) = self.symbol_table.get_symbol(*symbol) {
                    if sym.kind == crate::tast::SymbolKind::EnumVariant {
                        // Find parent enum
                        if let Some(parent) =
                            self.symbol_table.find_parent_enum_for_constructor(*symbol)
                        {
                            if self.symbol_table.get_symbol(parent).is_some() {
                                return Some(self.enum_runtime_id(parent));
                            }
                        }
                    }
                }
            }
            HirExprKind::Field { field, .. } => {
                if let Some(sym) = self.symbol_table.get_symbol(*field) {
                    if sym.kind == crate::tast::SymbolKind::EnumVariant {
                        if let Some(parent) =
                            self.symbol_table.find_parent_enum_for_constructor(*field)
                        {
                            if self.symbol_table.get_symbol(parent).is_some() {
                                return Some(self.enum_runtime_id(parent));
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        None
    }

    /// Append hidden enum type id to runtime enum helper calls.
    /// Falls back to `0` when type inference fails to preserve ABI parity (no arg-count mismatch).
    pub(crate) fn inject_hidden_enum_type_id_arg(
        &mut self,
        runtime_func: &str,
        args: &[HirExpr],
        arg_regs: &mut Vec<IrId>,
    ) {
        if !Self::runtime_func_needs_enum_type_id(runtime_func) {
            return;
        }

        let type_id = self.extract_enum_type_id_from_args(runtime_func, args);
        let tid_reg = match type_id {
            Some(type_id) => self.builder.build_const(IrValue::I32(type_id as i32)),
            None => {
                debug!(
                    "[ENUM TYPE_ID] Could not resolve hidden enum type_id for {}, using 0 fallback",
                    runtime_func
                );
                self.builder.build_const(IrValue::I32(0))
            }
        };

        if let Some(tid_reg) = tid_reg {
            arg_regs.push(tid_reg);
        }
    }

    /// Record an enum for RTTI registration
    /// This collects enum metadata during lowering so it can be registered at module init
    pub(crate) fn record_enum_for_registration(
        &mut self,
        enum_symbol_id: SymbolId,
        type_id: TypeId,
    ) {
        // Skip if already recorded
        if self.enums_for_registration.contains_key(&enum_symbol_id) {
            return;
        }

        // Use the deterministic name-hash runtime_type_id so the RTTI
        // entry's key matches every other site (object headers, vtable
        // lookups, cast/is checks) for this enum, stable across sessions.
        let runtime_type_id = self.runtime_type_id(type_id);

        // Get enum name from symbol table
        let enum_name = self
            .symbol_table
            .get_symbol(enum_symbol_id)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Enum_{}", enum_symbol_id.as_raw()));

        // Get variant names from the symbol table's enum_variants map
        let variant_names: Vec<String> =
            if let Some(variant_ids) = self.symbol_table.get_enum_variants(enum_symbol_id) {
                variant_ids
                    .iter()
                    .filter_map(|&var_id| {
                        self.symbol_table
                            .get_symbol(var_id)
                            .and_then(|sym| self.string_interner.get(sym.name))
                            .map(|s| s.to_string())
                    })
                    .collect()
            } else {
                Vec::new()
            };

        debug!(
            "[RTTI] Recording enum '{}' (type_id={}) with variants: {:?}",
            enum_name, runtime_type_id, variant_names
        );

        self.enums_for_registration
            .insert(enum_symbol_id, (runtime_type_id, enum_name, variant_names));
    }

    pub(crate) fn is_enum_symbol_expr(&self, expr: &HirExpr) -> bool {
        match &expr.kind {
            HirExprKind::Variable { symbol, .. } => self
                .symbol_table
                .get_symbol(*symbol)
                .map(|sym| sym.kind == crate::tast::symbols::SymbolKind::Enum)
                .unwrap_or(false),
            HirExprKind::Cast { expr: inner, .. } => self.is_enum_symbol_expr(inner),
            _ => false,
        }
    }

    /// Try to dispatch an enum built-in method (getIndex, getName, getParameters).
    /// Uses runtime mapping registered in runtime_mapping.rs.
    /// Injects compile-time constants (type_id, is_boxed) as extra params.
    pub(crate) fn try_dispatch_enum_method(
        &mut self,
        method_symbol: SymbolId,
        args: &[HirExpr],
    ) -> Option<Option<IrId>> {
        let receiver = &args[0];
        let enum_sym_id = self.resolve_enum_symbol(receiver.ty)?;

        // Look up the method in stdlib mapping under "Enum" class
        let method_name = self
            .symbol_table
            .get_symbol(method_symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())?;

        let (_, runtime_func) = self
            .stdlib_mapping
            .find_by_name("Enum", &method_name)
            .map(|(sig, call)| (sig.method, call.runtime_name))?;

        // Gather compile-time enum metadata
        let is_boxed = self.enum_is_boxed(enum_sym_id);
        let enum_type_id = self.enum_runtime_id(enum_sym_id) as i32;

        // Lower the receiver (the enum value)
        let receiver_reg = self.lower_expression(receiver)?;
        let is_boxed_const =
            self.builder
                .build_const(IrValue::I32(if is_boxed { 1 } else { 0 }))?;

        // Build args and return type from the runtime mapping's param_types/return_type.
        // getIndex takes (value, is_boxed), getName/getParameters take (type_id, value, is_boxed).
        let (call_args, param_types, return_type) = if runtime_func == "haxe_enum_get_index" {
            (
                vec![receiver_reg, is_boxed_const],
                vec![IrType::I64, IrType::I32],
                IrType::I64,
            )
        } else {
            let type_id_const = self.builder.build_const(IrValue::I32(enum_type_id))?;
            let ret_ty = if runtime_func == "haxe_enum_get_name" {
                IrType::Ptr(Box::new(IrType::String))
            } else {
                // getParameters returns opaque array pointer
                IrType::Ptr(Box::new(IrType::Void))
            };
            (
                vec![type_id_const, receiver_reg, is_boxed_const],
                vec![IrType::I32, IrType::I64, IrType::I32],
                ret_ty,
            )
        };

        let func_id =
            self.get_or_register_extern_function(runtime_func, param_types, return_type.clone());
        let result = self
            .builder
            .build_call_direct(func_id, call_args, return_type)?;
        Some(Some(result))
    }

    /// Return true for runtime functions that expect a hidden enum `type_id: i32`.
    pub(crate) fn runtime_func_needs_enum_type_id(runtime_func: &str) -> bool {
        matches!(
            runtime_func,
            "haxe_type_enum_constructor"
                | "haxe_type_enum_parameters"
                | "haxe_type_enum_index"
                | "haxe_type_get_enum"
                | "haxe_type_enum_eq"
        )
    }
}
