//! Field-wise bodies emitted for formatting, equality, hash and compare.

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
    /// Emit a custom @:debugFormat string. Parses "{fieldName}" placeholders in the format
    /// and replaces them with field value conversions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_custom_debug_format(
        &mut self,
        obj_reg: IrId,
        fields: &[(SymbolId, TypeId, u32)],
        fmt: &str,
        concat_fn: IrFunctionId,
        std_string_fn: IrFunctionId,
        box_int_fn: IrFunctionId,
        box_float_fn: IrFunctionId,
        string_ptr_ty: &IrType,
    ) -> Option<IrId> {
        // Build a name→(type_id, gep_idx) map for field lookup
        let field_map: std::collections::BTreeMap<String, (TypeId, u32)> = fields
            .iter()
            .filter_map(|&(sym, ty, idx)| {
                let name = self
                    .symbol_table
                    .get_symbol(sym)
                    .and_then(|s| self.string_interner.get(s.name))
                    .map(|s| s.to_string())?;
                Some((name, (ty, idx)))
            })
            .collect();

        // Parse format string into segments: literal text and {fieldName} references
        let mut result = self.builder.build_string(String::new())?;
        let mut chars = fmt.chars().peekable();
        let mut literal = String::new();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                // Flush accumulated literal
                if !literal.is_empty() {
                    let lit_str = self.builder.build_string(literal.clone())?;
                    result = self.builder.build_call_direct(
                        concat_fn,
                        vec![result, lit_str],
                        string_ptr_ty.clone(),
                    )?;
                    literal.clear();
                }
                // Read field name until '}'
                let mut field_name = String::new();
                for ch2 in chars.by_ref() {
                    if ch2 == '}' {
                        break;
                    }
                    field_name.push(ch2);
                }
                // Look up field and emit value
                if let Some(&(field_type_id, gep_idx)) = field_map.get(&field_name) {
                    let field_ir_type = self.convert_type(field_type_id);
                    let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
                    let field_ptr =
                        self.builder
                            .build_gep(obj_reg, vec![idx_const], field_ir_type.clone())?;
                    let field_val = self.builder.build_load(field_ptr, field_ir_type)?;

                    let type_kind = {
                        let type_table = self.type_table;
                        type_table.get(field_type_id).map(|t| t.kind.clone())
                    };
                    let field_str = match type_kind.as_ref() {
                        Some(TypeKind::String) => field_val,
                        Some(TypeKind::Bool) => {
                            let t = self.builder.build_string("true".to_string())?;
                            let f = self.builder.build_string("false".to_string())?;
                            self.builder.build_select(field_val, t, f)?
                        }
                        Some(TypeKind::Int) => {
                            let as_i64 =
                                self.builder
                                    .build_cast(field_val, IrType::I32, IrType::I64)?;
                            let boxed = self.builder.build_call_direct(
                                box_int_fn,
                                vec![as_i64],
                                IrType::Ptr(Box::new(IrType::U8)),
                            )?;
                            self.builder.build_call_direct(
                                std_string_fn,
                                vec![boxed],
                                string_ptr_ty.clone(),
                            )?
                        }
                        Some(TypeKind::Float) => {
                            let boxed = self.builder.build_call_direct(
                                box_float_fn,
                                vec![field_val],
                                IrType::Ptr(Box::new(IrType::U8)),
                            )?;
                            self.builder.build_call_direct(
                                std_string_fn,
                                vec![boxed],
                                string_ptr_ty.clone(),
                            )?
                        }
                        _ => self.builder.build_string("<object>".to_string())?,
                    };
                    result = self.builder.build_call_direct(
                        concat_fn,
                        vec![result, field_str],
                        string_ptr_ty.clone(),
                    )?;
                } else {
                    // Unknown field — emit literal "{fieldName}"
                    let unknown = self.builder.build_string(format!("{{{}}}", field_name))?;
                    result = self.builder.build_call_direct(
                        concat_fn,
                        vec![result, unknown],
                        string_ptr_ty.clone(),
                    )?;
                }
            } else {
                literal.push(ch);
            }
        }
        // Flush remaining literal
        if !literal.is_empty() {
            let lit_str = self.builder.build_string(literal)?;
            result = self.builder.build_call_direct(
                concat_fn,
                vec![result, lit_str],
                string_ptr_ty.clone(),
            )?;
        }
        Some(result)
    }

    /// Emit Debug toString for a pointer to a @:derive(Debug) class, using pre-resolved extern fns.
    /// Used for recursive nested object formatting.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_debug_to_string_for_ptr(
        &mut self,
        obj_reg: IrId,
        class_sym: SymbolId,
        fields: &[(SymbolId, TypeId, u32)],
        concat_fn: IrFunctionId,
        std_string_fn: IrFunctionId,
        box_int_fn: IrFunctionId,
        box_float_fn: IrFunctionId,
        string_ptr_ty: &IrType,
    ) -> Option<IrId> {
        self.emit_debug_to_string_for_ptr_depth(
            obj_reg,
            class_sym,
            fields,
            concat_fn,
            std_string_fn,
            box_int_fn,
            box_float_fn,
            string_ptr_ty,
            0,
        )
    }

    pub(crate) fn emit_debug_to_string_for_ptr_depth(
        &mut self,
        obj_reg: IrId,
        class_sym: SymbolId,
        fields: &[(SymbolId, TypeId, u32)],
        concat_fn: IrFunctionId,
        std_string_fn: IrFunctionId,
        box_int_fn: IrFunctionId,
        box_float_fn: IrFunctionId,
        string_ptr_ty: &IrType,
        depth: u32,
    ) -> Option<IrId> {
        // Depth limit to prevent infinite recursion for self-referential types
        const MAX_DEBUG_DEPTH: u32 = 4;
        if depth >= MAX_DEBUG_DEPTH {
            return self.builder.build_string("<...>".to_string());
        }

        let class_name = self
            .symbol_table
            .get_symbol(class_sym)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("Unknown");

        let mut result = self.builder.build_string(format!("{} {{ ", class_name))?;

        for (i, &(field_sym, field_type_id, gep_idx)) in fields.iter().enumerate() {
            let field_name = self
                .symbol_table
                .get_symbol(field_sym)
                .and_then(|s| self.string_interner.get(s.name))
                .unwrap_or("?");

            let prefix = if i > 0 {
                format!(", {}: ", field_name)
            } else {
                format!("{}: ", field_name)
            };
            let prefix_str = self.builder.build_string(prefix)?;
            result = self.builder.build_call_direct(
                concat_fn,
                vec![result, prefix_str],
                string_ptr_ty.clone(),
            )?;

            let field_ir_type = self.convert_type(field_type_id);
            let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let field_ptr =
                self.builder
                    .build_gep(obj_reg, vec![idx_const], field_ir_type.clone())?;
            let field_val = self.builder.build_load(field_ptr, field_ir_type.clone())?;

            let type_kind = {
                let type_table = self.type_table;
                type_table.get(field_type_id).map(|t| t.kind.clone())
            };
            let field_str = match type_kind.as_ref() {
                Some(TypeKind::String) => field_val,
                Some(TypeKind::Bool) => {
                    let true_str = self.builder.build_string("true".to_string())?;
                    let false_str = self.builder.build_string("false".to_string())?;
                    self.builder.build_select(field_val, true_str, false_str)?
                }
                Some(TypeKind::Int) => {
                    let as_i64 = self
                        .builder
                        .build_cast(field_val, IrType::I32, IrType::I64)?;
                    let boxed = self.builder.build_call_direct(
                        box_int_fn,
                        vec![as_i64],
                        IrType::Ptr(Box::new(IrType::U8)),
                    )?;
                    self.builder.build_call_direct(
                        std_string_fn,
                        vec![boxed],
                        string_ptr_ty.clone(),
                    )?
                }
                Some(TypeKind::Float) => {
                    let boxed = self.builder.build_call_direct(
                        box_float_fn,
                        vec![field_val],
                        IrType::Ptr(Box::new(IrType::U8)),
                    )?;
                    self.builder.build_call_direct(
                        std_string_fn,
                        vec![boxed],
                        string_ptr_ty.clone(),
                    )?
                }
                _ => self.builder.build_string("<object>".to_string())?,
            };

            result = self.builder.build_call_direct(
                concat_fn,
                vec![result, field_str],
                string_ptr_ty.clone(),
            )?;
        }

        let suffix = self.builder.build_string(" }".to_string())?;
        self.builder
            .build_call_direct(concat_fn, vec![result, suffix], string_ptr_ty.clone())
    }

    /// Emit equality comparison for a single field based on its type.
    /// Returns a Bool register: true if equal.
    pub(crate) fn emit_field_equality(
        &mut self,
        type_id: TypeId,
        lhs: IrId,
        rhs: IrId,
    ) -> Option<IrId> {
        let is_string = {
            let type_table = self.type_table;
            type_table
                .get(type_id)
                .map(|t| matches!(t.kind, TypeKind::String))
                .unwrap_or(false)
        };

        if is_string {
            let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
            let cmp_func = self.get_or_register_extern_function(
                "haxe_string_compare",
                vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                IrType::I32,
            );
            let cmp_result =
                self.builder
                    .build_call_direct(cmp_func, vec![lhs, rhs], IrType::I32)?;
            let zero = self.builder.build_const(IrValue::I32(0))?;
            self.builder.build_cmp(CompareOp::Eq, cmp_result, zero)
        } else {
            // Direct comparison: Int, Float, Bool, pointers (class/enum)
            self.builder.build_cmp(CompareOp::Eq, lhs, rhs)
        }
    }

    /// Compute a hash value (i64) for a single field based on its type.
    pub(crate) fn emit_field_hash(
        &mut self,
        type_id: TypeId,
        ir_type: &IrType,
        val: IrId,
    ) -> Option<IrId> {
        let type_kind = {
            let type_table = self.type_table;
            type_table.get(type_id).map(|t| t.kind.clone())
        };

        match type_kind.as_ref() {
            Some(TypeKind::String) => {
                // Call haxe_string_hash → i32, then widen to i64
                let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                let hash_func = self.get_or_register_extern_function(
                    "haxe_string_hash",
                    vec![string_ptr_ty],
                    IrType::I32,
                );
                let hash_i32 = self
                    .builder
                    .build_call_direct(hash_func, vec![val], IrType::I32)?;
                self.builder.build_cast(hash_i32, IrType::I32, IrType::I64)
            }
            Some(TypeKind::Int) => {
                // Widen i32 to i64
                self.builder.build_cast(val, IrType::I32, IrType::I64)
            }
            Some(TypeKind::Bool) => {
                // false → 0, true → 1
                self.builder.build_cast(val, IrType::Bool, IrType::I64)
            }
            Some(TypeKind::Float) => {
                // Bitcast f64 to i64 for hash
                self.builder.build_bitcast(val, IrType::I64)
            }
            _ => {
                // Pointer/enum/other — use as i64 directly
                match ir_type {
                    IrType::I32 => self.builder.build_cast(val, IrType::I32, IrType::I64),
                    IrType::I64 => Some(val),
                    _ => self.builder.build_cast(val, ir_type.clone(), IrType::I64),
                }
            }
        }
    }

    /// Compute a three-way comparison value (i32: -1, 0, 1) for a single field.
    pub(crate) fn emit_field_three_way_cmp(
        &mut self,
        type_id: TypeId,
        lhs: IrId,
        rhs: IrId,
    ) -> Option<IrId> {
        let is_string = {
            let type_table = self.type_table;
            type_table
                .get(type_id)
                .map(|t| matches!(t.kind, TypeKind::String))
                .unwrap_or(false)
        };

        if is_string {
            // haxe_string_compare already returns a three-way i32
            let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
            let cmp_func = self.get_or_register_extern_function(
                "haxe_string_compare",
                vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                IrType::I32,
            );
            self.builder
                .build_call_direct(cmp_func, vec![lhs, rhs], IrType::I32)
        } else {
            // (a < b) ? -1 : ((a > b) ? 1 : 0)
            let lt_block = self.builder.create_block()?;
            let gt_check = self.builder.create_block()?;
            let gt_block = self.builder.create_block()?;
            let eq_block = self.builder.create_block()?;
            let cmp_merge = self.builder.create_block()?;

            let is_lt = self.builder.build_cmp(CompareOp::Lt, lhs, rhs)?;
            self.builder.build_cond_branch(is_lt, lt_block, gt_check)?;

            self.builder.switch_to_block(lt_block);
            let neg_one = self.builder.build_const(IrValue::I32(-1))?;
            self.builder.build_branch(cmp_merge)?;

            self.builder.switch_to_block(gt_check);
            let is_gt = self.builder.build_cmp(CompareOp::Gt, lhs, rhs)?;
            self.builder.build_cond_branch(is_gt, gt_block, eq_block)?;

            self.builder.switch_to_block(gt_block);
            let pos_one = self.builder.build_const(IrValue::I32(1))?;
            self.builder.build_branch(cmp_merge)?;

            self.builder.switch_to_block(eq_block);
            let zero = self.builder.build_const(IrValue::I32(0))?;
            self.builder.build_branch(cmp_merge)?;

            self.builder.switch_to_block(cmp_merge);
            let result = self.builder.build_phi(cmp_merge, IrType::I32)?;
            self.builder
                .add_phi_incoming(cmp_merge, result, lt_block, neg_one)?;
            self.builder
                .add_phi_incoming(cmp_merge, result, gt_block, pos_one)?;
            self.builder
                .add_phi_incoming(cmp_merge, result, eq_block, zero)?;

            Some(result)
        }
    }
}
