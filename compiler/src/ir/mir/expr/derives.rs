//! Bodies synthesised for `@:derive` — equality, ordering, hash, clone, toString, default.

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
    /// Lower field-by-field equality for classes that @:derive([PartialEq]).
    /// Generates: ptr_eq fast path → null guards → field-by-field short-circuit chain.
    pub(crate) fn lower_derived_equality(
        &mut self,
        op: &HirBinaryOp,
        lhs: &HirExpr,
        rhs: &HirExpr,
        class_sym: SymbolId,
    ) -> Option<IrId> {
        let is_eq = matches!(op, HirBinaryOp::Eq);
        let fields = self.class_instance_fields.get(&class_sym)?.clone();

        let lhs_reg = self.lower_expression(lhs)?;
        let rhs_reg = self.lower_expression(rhs)?;

        let null_check = self.builder.create_block()?;
        let null_check2 = self.builder.create_block()?;
        let fail = self.builder.create_block()?;
        let pass = self.builder.create_block()?;
        let merge = self.builder.create_block()?;

        // Fast path: pointer equality (handles same-object and both-null)
        let ptr_eq = self.builder.build_cmp(CompareOp::Eq, lhs_reg, rhs_reg)?;
        self.builder.build_cond_branch(ptr_eq, pass, null_check)?;

        // Null check LHS (ptr_eq was false, so if lhs is null → rhs is not → not equal)
        self.builder.switch_to_block(null_check);
        let null_val = self.builder.build_const(IrValue::I64(0))?;
        let lhs_is_null = self.builder.build_cmp(CompareOp::Eq, lhs_reg, null_val)?;
        self.builder
            .build_cond_branch(lhs_is_null, fail, null_check2)?;

        // Null check RHS
        self.builder.switch_to_block(null_check2);
        let null_val2 = self.builder.build_const(IrValue::I64(0))?;
        let rhs_is_null = self.builder.build_cmp(CompareOp::Eq, rhs_reg, null_val2)?;

        if fields.is_empty() {
            // No instance fields — both non-null objects of same type are equal
            self.builder.build_cond_branch(rhs_is_null, fail, pass)?;
        } else {
            let first_field = self.builder.create_block()?;
            self.builder
                .build_cond_branch(rhs_is_null, fail, first_field)?;

            // Field-by-field comparison with short-circuit
            let mut current_block = first_field;
            for (i, &(_field_sym, field_type_id, gep_idx)) in fields.iter().enumerate() {
                self.builder.switch_to_block(current_block);

                let field_ir_type = self.convert_type(field_type_id);
                let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;

                // Load field from LHS
                let lhs_ptr =
                    self.builder
                        .build_gep(lhs_reg, vec![idx_const], field_ir_type.clone())?;
                let lhs_val = self.builder.build_load(lhs_ptr, field_ir_type.clone())?;

                // Load field from RHS (reuse same index constant — IrId is Copy)
                let rhs_ptr =
                    self.builder
                        .build_gep(rhs_reg, vec![idx_const], field_ir_type.clone())?;
                let rhs_val = self.builder.build_load(rhs_ptr, field_ir_type.clone())?;

                let field_eq = self.emit_field_equality(field_type_id, lhs_val, rhs_val)?;

                if i == fields.len() - 1 {
                    // Last field — branch to pass or fail
                    self.builder.build_cond_branch(field_eq, pass, fail)?;
                } else {
                    let next = self.builder.create_block()?;
                    self.builder.build_cond_branch(field_eq, next, fail)?;
                    current_block = next;
                }
            }
        }

        // Pass block
        self.builder.switch_to_block(pass);
        let pass_val = self.builder.build_bool(is_eq)?;
        self.builder.build_branch(merge)?;

        // Fail block
        self.builder.switch_to_block(fail);
        let fail_val = self.builder.build_bool(!is_eq)?;
        self.builder.build_branch(merge)?;

        // Merge
        self.builder.switch_to_block(merge);
        let result = self.builder.build_phi(merge, IrType::Bool)?;
        self.builder
            .add_phi_incoming(merge, result, pass, pass_val)?;
        self.builder
            .add_phi_incoming(merge, result, fail, fail_val)?;

        Some(result)
    }

    /// Lower lexicographic ordering for classes that @:derive([PartialOrd]).
    /// Computes a three-way comparison per field, short-circuiting on first difference.
    pub(crate) fn lower_derived_ordering(
        &mut self,
        op: &HirBinaryOp,
        lhs: &HirExpr,
        rhs: &HirExpr,
        class_sym: SymbolId,
    ) -> Option<IrId> {
        let fields = self.class_instance_fields.get(&class_sym)?.clone();

        let lhs_reg = self.lower_expression(lhs)?;
        let rhs_reg = self.lower_expression(rhs)?;

        // All paths converge at final_cmp with a three-way i32 value
        let final_cmp = self.builder.create_block()?;
        let mut phi_inputs: Vec<(IrBlockId, IrId)> = Vec::new();

        // Fast path: pointer equality → cmp_val = 0 (equal)
        let null_check = self.builder.create_block()?;
        let same_ptr = self.builder.create_block()?;
        let ptr_eq = self.builder.build_cmp(CompareOp::Eq, lhs_reg, rhs_reg)?;
        self.builder
            .build_cond_branch(ptr_eq, same_ptr, null_check)?;

        self.builder.switch_to_block(same_ptr);
        let zero_same = self.builder.build_const(IrValue::I32(0))?;
        phi_inputs.push((same_ptr, zero_same));
        self.builder.build_branch(final_cmp)?;

        // Null guards: null is treated as "less than" any non-null value
        self.builder.switch_to_block(null_check);
        let null_val = self.builder.build_const(IrValue::I64(0))?;
        let lhs_is_null = self.builder.build_cmp(CompareOp::Eq, lhs_reg, null_val)?;
        let lhs_null_block = self.builder.create_block()?;
        let null_check2 = self.builder.create_block()?;
        self.builder
            .build_cond_branch(lhs_is_null, lhs_null_block, null_check2)?;

        // LHS is null, RHS is not → LHS < RHS → cmp_val = -1
        self.builder.switch_to_block(lhs_null_block);
        let neg_one_null = self.builder.build_const(IrValue::I32(-1))?;
        phi_inputs.push((lhs_null_block, neg_one_null));
        self.builder.build_branch(final_cmp)?;

        self.builder.switch_to_block(null_check2);
        let null_val2 = self.builder.build_const(IrValue::I64(0))?;
        let rhs_is_null = self.builder.build_cmp(CompareOp::Eq, rhs_reg, null_val2)?;
        let rhs_null_block = self.builder.create_block()?;
        let field_start = self.builder.create_block()?;
        self.builder
            .build_cond_branch(rhs_is_null, rhs_null_block, field_start)?;

        // RHS is null, LHS is not → LHS > RHS → cmp_val = 1
        self.builder.switch_to_block(rhs_null_block);
        let pos_one_null = self.builder.build_const(IrValue::I32(1))?;
        phi_inputs.push((rhs_null_block, pos_one_null));
        self.builder.build_branch(final_cmp)?;

        // Field-by-field three-way comparison
        self.builder.switch_to_block(field_start);
        if fields.is_empty() {
            // No fields → equal → cmp_val = 0
            let zero_empty = self.builder.build_const(IrValue::I32(0))?;
            phi_inputs.push((field_start, zero_empty));
            self.builder.build_branch(final_cmp)?;
        } else {
            for (i, &(_field_sym, field_type_id, gep_idx)) in fields.iter().enumerate() {
                let field_ir_type = self.convert_type(field_type_id);
                let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;

                let lhs_ptr =
                    self.builder
                        .build_gep(lhs_reg, vec![idx_const], field_ir_type.clone())?;
                let lhs_val = self.builder.build_load(lhs_ptr, field_ir_type.clone())?;

                let rhs_ptr =
                    self.builder
                        .build_gep(rhs_reg, vec![idx_const], field_ir_type.clone())?;
                let rhs_val = self.builder.build_load(rhs_ptr, field_ir_type.clone())?;

                let cmp_val = self.emit_field_three_way_cmp(field_type_id, lhs_val, rhs_val)?;

                if i == fields.len() - 1 {
                    // Last field — always go to final_cmp
                    let cur = self.builder.current_block()?;
                    phi_inputs.push((cur, cmp_val));
                    self.builder.build_branch(final_cmp)?;
                } else {
                    // Non-zero → decided, go to final_cmp; zero → next field
                    let zero = self.builder.build_const(IrValue::I32(0))?;
                    let is_ne = self.builder.build_cmp(CompareOp::Ne, cmp_val, zero)?;
                    let cur = self.builder.current_block()?;
                    phi_inputs.push((cur, cmp_val));
                    let next = self.builder.create_block()?;
                    self.builder.build_cond_branch(is_ne, final_cmp, next)?;
                    self.builder.switch_to_block(next);
                }
            }
        }

        // Final comparison: cmp_val <op> 0
        self.builder.switch_to_block(final_cmp);
        let cmp_val = self.builder.build_phi(final_cmp, IrType::I32)?;
        for &(block, val) in &phi_inputs {
            self.builder
                .add_phi_incoming(final_cmp, cmp_val, block, val)?;
        }
        let zero_final = self.builder.build_const(IrValue::I32(0))?;
        let cmp_op = match op {
            HirBinaryOp::Lt => CompareOp::Lt,
            HirBinaryOp::Le => CompareOp::Le,
            HirBinaryOp::Gt => CompareOp::Gt,
            HirBinaryOp::Ge => CompareOp::Ge,
            _ => unreachable!(),
        };
        self.builder.build_cmp(cmp_op, cmp_val, zero_final)
    }

    /// Lower inline hash computation for classes that @:derive([Hash]).
    /// Computes: hash = seed; for each field: hash = hash * 31 + field_hash; return hash as I32.
    pub(crate) fn lower_derived_hash_code(
        &mut self,
        object: &HirExpr,
        class_sym: SymbolId,
    ) -> Option<IrId> {
        let fields = self.class_instance_fields.get(&class_sym)?.clone();
        let obj_reg = self.lower_expression(object)?;

        // Start with seed based on number of fields (deterministic per-class)
        let mut hash_reg = self.builder.build_const(IrValue::I64(2166136261i64))?; // FNV offset basis

        let thirty_one = self.builder.build_const(IrValue::I64(31))?;

        for &(_field_sym, field_type_id, gep_idx) in &fields {
            let field_ir_type = self.convert_type(field_type_id);
            let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let field_ptr =
                self.builder
                    .build_gep(obj_reg, vec![idx_const], field_ir_type.clone())?;
            let field_val = self.builder.build_load(field_ptr, field_ir_type.clone())?;

            // Compute field hash based on type
            let field_hash = self.emit_field_hash(field_type_id, &field_ir_type, field_val)?;

            // hash = hash * 31 + field_hash
            let mul = self
                .builder
                .build_binop(BinaryOp::Mul, hash_reg, thirty_one)?;
            hash_reg = self.builder.build_binop(BinaryOp::Add, mul, field_hash)?;
        }

        // Truncate to I32 for Haxe Int return type
        self.builder.build_cast(hash_reg, IrType::I64, IrType::I32)
    }

    /// @:derive(Clone) — deep copy an instance by allocating a new object and copying each field.
    /// String and Array fields are deep-copied; class fields with Clone are recursively cloned;
    /// primitives and other pointers are shallow-copied.
    pub(crate) fn lower_derived_clone(
        &mut self,
        object: &HirExpr,
        class_sym: SymbolId,
    ) -> Option<IrId> {
        // Tier B: extern runtime-backed clone (e.g. Tensor, QTensor).
        // Delegate to a named extern (rayzor_tensor_clone / rayzor_qtensor_clone)
        // with the receiver as a single i64 arg; field-by-field synthesis is skipped.
        if let Some(&extern_name) = self.derive_clone_extern_fns.get(&class_sym) {
            let obj_reg = self.lower_expression(object)?;
            let clone_fn =
                self.get_or_register_extern_function(extern_name, vec![IrType::I64], IrType::I64);
            return self
                .builder
                .build_call_direct(clone_fn, vec![obj_reg], IrType::I64);
        }

        let fields = self.class_instance_fields.get(&class_sym)?.clone();
        let obj_reg = self.lower_expression(object)?;

        // Compute allocation size: (num_fields + 1) * 8  (slot 0 = type_id header)
        let alloc_size = (fields.len() as u64 + 1) * 8;
        let alloc_size = alloc_size.max(16); // minimum 16 bytes
        let new_ptr = self.build_heap_alloc(alloc_size)?;

        // Copy type_id header from source (GEP index 0)
        let zero = self.builder.build_const(IrValue::I32(0))?;
        let header_ptr = self.builder.build_gep(obj_reg, vec![zero], IrType::I64)?;
        let header_val = self.builder.build_load(header_ptr, IrType::I64)?;
        let zero2 = self.builder.build_const(IrValue::I32(0))?;
        let new_header_ptr = self.builder.build_gep(new_ptr, vec![zero2], IrType::I64)?;
        self.builder.build_store(new_header_ptr, header_val)?;

        // Copy each field
        for &(_field_sym, field_type_id, gep_idx) in &fields {
            let field_ir_type = self.convert_type(field_type_id);
            let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let src_field_ptr =
                self.builder
                    .build_gep(obj_reg, vec![idx_const], field_ir_type.clone())?;
            let field_val = self
                .builder
                .build_load(src_field_ptr, field_ir_type.clone())?;

            // Determine copy strategy based on type
            let cloned_val = {
                let type_kind = {
                    let type_table = self.type_table;
                    type_table.get(field_type_id).map(|t| t.kind.clone())
                };
                match type_kind.as_ref() {
                    Some(TypeKind::String) => {
                        // Deep copy string via haxe_string_copy
                        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
                        let copy_func = self.get_or_register_extern_function(
                            "haxe_string_copy",
                            vec![string_ptr_ty.clone()],
                            string_ptr_ty,
                        );
                        self.builder.build_call_direct(
                            copy_func,
                            vec![field_val],
                            IrType::Ptr(Box::new(IrType::String)),
                        )?
                    }
                    Some(TypeKind::Class { symbol_id, .. })
                        if self.derive_clone_classes.contains(symbol_id) =>
                    {
                        // Recursively clone nested Clone class
                        // For now, shallow copy (recursive clone would need the HirExpr)
                        field_val
                    }
                    _ => {
                        // Primitive (Int, Float, Bool) or other pointer — bitwise copy
                        field_val
                    }
                }
            };

            let dst_idx = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let dst_field_ptr = self
                .builder
                .build_gep(new_ptr, vec![dst_idx], field_ir_type)?;
            self.builder.build_store(dst_field_ptr, cloned_val)?;
        }

        // Set class hint so field access on cloned value uses GEP, not reflection
        if let Some(sym) = self.symbol_table.get_symbol(class_sym) {
            let name = self.string_interner.get(sym.name).unwrap_or("").to_string();
            if !name.is_empty() {
                self.register_class_hints.insert(new_ptr, name);
            }
        }

        Some(new_ptr)
    }

    /// @:derive(Debug) — generate toString() returning "ClassName { field1: val1, field2: val2 }"
    pub(crate) fn lower_derived_to_string(
        &mut self,
        object: &HirExpr,
        class_sym: SymbolId,
    ) -> Option<IrId> {
        let fields = self.class_instance_fields.get(&class_sym)?.clone();
        let obj_reg = self.lower_expression(object)?;

        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
        let concat_fn = self.get_or_register_extern_function(
            "haxe_string_concat",
            vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
            string_ptr_ty.clone(),
        );
        let std_string_fn = self.get_or_register_extern_function(
            "haxe_std_string_ptr",
            vec![IrType::Ptr(Box::new(IrType::U8))],
            string_ptr_ty.clone(),
        );
        let box_int_fn = self.get_or_register_extern_function(
            "haxe_box_int_ptr",
            vec![IrType::I64],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let box_float_fn = self.get_or_register_extern_function(
            "haxe_box_float_ptr",
            vec![IrType::F64],
            IrType::Ptr(Box::new(IrType::U8)),
        );

        // Check for @:debugFormat custom format string
        if let Some(fmt) = self.debug_format_strings.get(&class_sym).cloned() {
            return self.emit_custom_debug_format(
                obj_reg,
                &fields,
                &fmt,
                concat_fn,
                std_string_fn,
                box_int_fn,
                box_float_fn,
                &string_ptr_ty,
            );
        }

        // Get class name
        let class_name = self
            .symbol_table
            .get_symbol(class_sym)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("Unknown");

        // Start with "ClassName { "
        let mut result = self.builder.build_string(format!("{} {{ ", class_name))?;

        for (i, &(field_sym, field_type_id, gep_idx)) in fields.iter().enumerate() {
            // Get field name
            let field_name = self
                .symbol_table
                .get_symbol(field_sym)
                .and_then(|s| self.string_interner.get(s.name))
                .unwrap_or("?");

            // Append "fieldName: "
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

            // Load field value
            let field_ir_type = self.convert_type(field_type_id);
            let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let field_ptr =
                self.builder
                    .build_gep(obj_reg, vec![idx_const], field_ir_type.clone())?;
            let field_val = self.builder.build_load(field_ptr, field_ir_type.clone())?;

            // Convert field to string based on type
            let type_kind = {
                let type_table = self.type_table;
                type_table.get(field_type_id).map(|t| t.kind.clone())
            };
            let field_str = match type_kind.as_ref() {
                Some(TypeKind::String) => {
                    // String field — use directly
                    field_val
                }
                Some(TypeKind::Bool) => {
                    // Bool → "true" / "false" via branch
                    let true_str = self.builder.build_string("true".to_string())?;
                    let false_str = self.builder.build_string("false".to_string())?;
                    self.builder.build_select(field_val, true_str, false_str)?
                }
                Some(TypeKind::Int) => {
                    // Int → box → haxe_std_string_ptr
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
                    // Float → box → haxe_std_string_ptr
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
                _ => {
                    // Check if field type is a @:derive(Debug) class — recursive toString
                    let nested_debug_sym = {
                        let type_table = self.type_table;
                        match type_table.get(field_type_id).map(|t| &t.kind) {
                            Some(TypeKind::Class { symbol_id, .. }) => {
                                if self.derive_debug_classes.contains(symbol_id) {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    };
                    if let Some(nested_sym) = nested_debug_sym {
                        // Recursively generate toString for the nested @:derive(Debug) object
                        if let Some(nested_fields) =
                            self.class_instance_fields.get(&nested_sym).cloned()
                        {
                            self.emit_debug_to_string_for_ptr_depth(
                                field_val,
                                nested_sym,
                                &nested_fields,
                                concat_fn,
                                std_string_fn,
                                box_int_fn,
                                box_float_fn,
                                &string_ptr_ty,
                                1,
                            )?
                        } else {
                            self.builder.build_string("<object>".to_string())?
                        }
                    } else {
                        // Non-debug class — "<object>"
                        self.builder.build_string("<object>".to_string())?
                    }
                }
            };

            result = self.builder.build_call_direct(
                concat_fn,
                vec![result, field_str],
                string_ptr_ty.clone(),
            )?;
        }

        // Append " }"
        let suffix = self.builder.build_string(" }".to_string())?;
        self.builder
            .build_call_direct(concat_fn, vec![result, suffix], string_ptr_ty)
    }

    /// @:derive(Default) — allocate a zero-initialized instance with default values for all fields.
    pub(crate) fn lower_derived_default(&mut self, class_sym: SymbolId) -> Option<IrId> {
        let fields = self.class_instance_fields.get(&class_sym)?.clone();

        // Compute allocation size
        let alloc_size = (fields.len() as u64 + 1) * 8;
        let alloc_size = alloc_size.max(16);
        let new_ptr = self.build_heap_alloc(alloc_size)?;

        // Store type_id header at GEP index 0
        let type_id_val = self
            .builder
            .build_const(IrValue::I64(class_sym.as_raw() as i64))?;
        let zero = self.builder.build_const(IrValue::I32(0))?;
        let header_ptr = self.builder.build_gep(new_ptr, vec![zero], IrType::I64)?;
        self.builder.build_store(header_ptr, type_id_val)?;

        // Initialize each field to its default value
        // Priority: @:default(value) metadata > field initializer > type-based default
        let field_defaults: Vec<_> = fields
            .iter()
            .map(|&(field_sym, _, _)| self.field_default_exprs.get(&field_sym).cloned())
            .collect();

        for (i, &(_field_sym, field_type_id, gep_idx)) in fields.iter().enumerate() {
            let field_ir_type = self.convert_type(field_type_id);

            // Try custom default from @:default(value) or field initializer first
            let default_val = if let Some(ref default_expr) = field_defaults[i] {
                match self.lower_expression(default_expr) {
                    Some(val) => val,
                    None => self.build_type_default(field_type_id)?,
                }
            } else {
                self.build_type_default(field_type_id)?
            };

            let idx_const = self.builder.build_const(IrValue::I32(gep_idx as i32))?;
            let field_ptr = self
                .builder
                .build_gep(new_ptr, vec![idx_const], field_ir_type)?;
            self.builder.build_store(field_ptr, default_val)?;
        }

        // Set class hint so field access on default instance uses GEP, not reflection
        if let Some(sym) = self.symbol_table.get_symbol(class_sym) {
            let name = self.string_interner.get(sym.name).unwrap_or("").to_string();
            if !name.is_empty() {
                self.register_class_hints.insert(new_ptr, name);
            }
        }

        Some(new_ptr)
    }
}
