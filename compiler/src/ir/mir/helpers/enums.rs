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

    /// Stable runtime id for an enum, from its declaration symbol. Mirrors the
    /// `TypeKind::Enum` arm of `runtime_type_id` so a value boxed by type and a
    /// lookup keyed by symbol land on the same key. Enum RTTI registration uses
    /// this id too; the single key space is what makes reflection on a
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

    /// The boxed form of a parameterless variant of an enum that has other
    /// parameterized variants: the runtime's one `[tag:i32][pad:i32]` cell for
    /// that enum and tag, so two evaluations of `None` are the same value.
    /// Returned as ptr bitcast to i64.
    pub(crate) fn build_boxed_enum_tag_only(
        &mut self,
        enum_symbol: SymbolId,
        tag_idx: i32,
    ) -> Option<IrId> {
        let cell_func = self.get_or_register_extern_function(
            "haxe_enum_nullary_cell",
            vec![IrType::I32, IrType::I32],
            IrType::Ptr(Box::new(IrType::I8)),
        );
        let type_id = self.enum_runtime_id(enum_symbol);
        let type_val = self.builder.build_const(IrValue::I32(type_id as i32))?;
        let tag_val = self.builder.build_const(IrValue::I32(tag_idx))?;
        let ptr = self.builder.build_call_direct(
            cell_func,
            vec![type_val, tag_val],
            IrType::Ptr(Box::new(IrType::I8)),
        )?;
        self.builder.build_bitcast(ptr, IrType::I64)
    }

    /// Allocate a boxed enum struct with payload fields.
    /// Layout: [tag:i32][type_id:i32][field0:i64][field1:i64]...
    /// The enum's runtime id in the pad word lets code that holds the value
    /// as a bare `EnumValue` recover its enum.
    pub(crate) fn build_boxed_enum_with_fields(
        &mut self,
        enum_symbol: SymbolId,
        tag_idx: i32,
        field_count: usize,
        constructor_args: &[HirExpr],
        field_types: &[(IrType, TypeId)],
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
        let type_offset = self.builder.build_const(IrValue::I64(4))?;
        let type_ptr =
            self.builder
                .build_gep(ptr, vec![type_offset], IrType::Ptr(Box::new(IrType::I8)))?;
        let type_ptr_i32 = self
            .builder
            .build_bitcast(type_ptr, IrType::Ptr(Box::new(IrType::I32)))?;
        let type_id = self.enum_runtime_id(enum_symbol) as i32;
        let type_val = self.builder.build_const(IrValue::I32(type_id))?;
        self.builder.build_store(type_ptr_i32, type_val)?;

        for (i, arg) in constructor_args.iter().take(field_count).enumerate() {
            let prev_target = self.let_target_type_hint.take();
            if matches!(&arg.kind, HirExprKind::Call { .. }) {
                self.let_target_type_hint = field_types.get(i).map(|(_, ty)| *ty);
            }
            let arg_result = self.lower_expression(arg);
            self.let_target_type_hint = prev_target;
            let mut arg_reg = arg_result?;
            if let Some((target_ir, target_ty)) = field_types.get(i) {
                // Concrete fields unbox Dynamic inputs; erased fields keep raw slots.
                arg_reg = self
                    .maybe_box_for_optional(arg_reg, arg.ty, *target_ty)
                    .unwrap_or(arg_reg);
                arg_reg = self.maybe_unbox_value(arg_reg, arg.ty, *target_ty)?;
                let source_ir = self.builder.get_register_type(arg_reg)?;
                if *target_ir == IrType::F64 && source_ir.is_integer() {
                    arg_reg = self.builder.build_cast(arg_reg, source_ir, IrType::F64)?;
                }
            }
            arg_reg = self.coerce_reg_to(arg_reg, &IrType::I64)?;
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
    /// Type receivers do not identify the enum of the value arguments.
    pub(crate) fn extract_enum_type_id_from_args(
        &self,
        runtime_func: &str,
        args: &[HirExpr],
    ) -> Option<u32> {
        if args.is_empty() {
            return None;
        }
        if crate::debug_flags::enum_lookup_debug() {
            for arg in args {
                eprintln!(
                    "[enum lookup] {runtime_func}: kind={:?}, type={:?}, ValueType={}",
                    std::mem::discriminant(&arg.kind),
                    self.type_table.get(arg.ty).map(|ty| &ty.kind),
                    self.expr_is_value_type_expr(arg)
                );
                if let HirExprKind::Call { callee, .. } = &arg.kind {
                    eprintln!("[enum lookup] callee={:?}", callee.kind);
                }
            }
        }
        args.iter()
            .filter(|arg| !self.is_class_symbol_expr(arg))
            .find_map(|arg| self.extract_enum_type_id_from_expr(arg))
    }

    /// Inspects the expression to find the enum variant symbol, then looks up the parent enum.
    pub(crate) fn extract_enum_type_id_from_expr(&self, expr: &HirExpr) -> Option<u32> {
        if self.expr_is_value_type_expr(expr) {
            if let Some(symbol) = self.symbol_table.all_symbols().find(|symbol| {
                symbol.kind == crate::tast::SymbolKind::Enum
                    && self.string_interner.get(symbol.name) == Some("ValueType")
                    && matches!(
                        symbol
                            .qualified_name
                            .and_then(|name| self.string_interner.get(name)),
                        Some("ValueType" | "Type.ValueType")
                    )
            }) {
                return Some(self.enum_runtime_id(symbol.id));
            }
        }
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
        if runtime_func == "haxe_type_enum_eq" {
            let mask = self.enum_string_param_mask(args);
            if let Some(mask_reg) = self.builder.build_const(IrValue::I64(mask)) {
                arg_regs.push(mask_reg);
            }
        }
    }

    /// The String parameter slots of the arguments' enum instantiation:
    /// bit `n` is the `n`th parameter counted through the constructors in
    /// order. An erased `T` payload is a raw reference, so the runtime can
    /// only compare it by value when told it is a String.
    fn enum_string_param_mask(&self, args: &[HirExpr]) -> i64 {
        let mut mask = 0i64;
        for arg in args {
            let ty = self.resolve_through_aliases(arg.ty);
            let Some(enum_sym) = self.resolve_enum_symbol(ty) else {
                continue;
            };
            let Some(variants) = self.symbol_table.get_enum_variants(enum_sym) else {
                continue;
            };
            let mut slot = 0u32;
            for &variant in variants {
                let Some(name) = self.symbol_table.get_symbol(variant).map(|s| s.name) else {
                    break;
                };
                for (_, field_ty) in self.get_enum_variant_field_types(ty, name) {
                    let field_ty = self.resolve_through_aliases(field_ty);
                    if slot < 64
                        && matches!(
                            self.type_table.get(field_ty).map(|t| &t.kind),
                            Some(TypeKind::String)
                        )
                    {
                        mask |= 1i64 << slot;
                    }
                    slot += 1;
                }
            }
        }
        mask
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

        let enum_name = self
            .symbol_table
            .get_symbol(enum_symbol_id)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Enum_{}", enum_symbol_id.as_raw()));

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
        let Some(enum_sym_id) = self.resolve_enum_symbol(receiver.ty) else {
            return self.try_dispatch_enum_value_method(method_symbol, receiver);
        };

        // Look up the method in stdlib mapping under "Enum" class
        let method_name = self
            .symbol_table
            .get_symbol(method_symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .map(|s| s.to_string())?;

        let (_, runtime_func) = self
            .stdlib_mapping
            .find_by_name(self.stdlib_mapping.key("Enum"), &method_name)
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

    /// An `EnumValue` method on a value whose enum is not known statically:
    /// the Type API entry points recover the enum from the value itself.
    fn try_dispatch_enum_value_method(
        &mut self,
        method_symbol: SymbolId,
        receiver: &HirExpr,
    ) -> Option<Option<IrId>> {
        let qualified = self
            .symbol_table
            .get_symbol(method_symbol)?
            .qualified_name
            .and_then(|name| self.string_interner.get(name))?;
        let (runtime_func, return_type) = match qualified {
            "EnumValue.getIndex" => ("haxe_type_enum_index", IrType::I64),
            "EnumValue.getName" => (
                "haxe_type_enum_constructor",
                IrType::Ptr(Box::new(IrType::String)),
            ),
            "EnumValue.getParameters" => (
                "haxe_type_enum_parameters",
                IrType::Ptr(Box::new(IrType::Void)),
            ),
            _ => return None,
        };
        let value = self.lower_expression(receiver)?;
        let value = self.coerce_reg_to(value, &IrType::I64)?;
        let unknown_enum = self.builder.build_const(IrValue::I32(0))?;
        let func_id = self.get_or_register_extern_function(
            runtime_func,
            vec![IrType::I64, IrType::I32],
            return_type.clone(),
        );
        let result =
            self.builder
                .build_call_direct(func_id, vec![value, unknown_enum], return_type)?;
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
