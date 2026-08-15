//! `switch` lowering and the pattern tests it emits.

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
    pub(crate) fn lower_switch_statement(&mut self, scrutinee: &HirExpr, cases: &[HirMatchCase]) {
        // Check exhaustiveness (analysis only, no codegen effect)
        self.check_switch_exhaustiveness(scrutinee, cases);

        // Switch/match statement lowering — see lower_if_statement for the
        // analogous if/else pattern. Variables modified inside any case body
        // get phi nodes at the continuation block so cross-case writes are
        // visible (and SSA-valid) to code following the switch.

        // Evaluate scrutinee once
        let scrut_val = match self.lower_expression(scrutinee) {
            Some(v) => v,
            None => return,
        };

        let continuation = match self.builder.create_block() {
            Some(b) => b,
            None => return,
        };

        let mut case_test_blocks = Vec::new();
        let mut case_body_blocks = Vec::new();

        for _ in cases {
            if let (Some(test), Some(body)) =
                (self.builder.create_block(), self.builder.create_block())
            {
                case_test_blocks.push(test);
                case_body_blocks.push(body);
            }
        }

        // Default block (for non-exhaustive matches)
        let default_block = match self.builder.create_block() {
            Some(b) => b,
            None => return,
        };

        // Find variables modified in any case body. Capture their pre-switch
        // register so we can phi-merge per-case writes at the continuation.
        let mut modified_vars: BTreeSet<SymbolId> = BTreeSet::new();
        for case in cases {
            for stmt in &case.body.statements {
                self.find_modified_variables_in_statement(stmt, &mut modified_vars);
            }
        }
        let mut var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                if let Some(func) = self.builder.current_function() {
                    if let Some(local) = func.locals.get(&reg) {
                        var_initial_values.insert(*symbol_id, (reg, local.ty.clone()));
                    }
                }
            }
        }

        let entry_block = self.builder.current_block();

        if let Some(&first_test) = case_test_blocks.first() {
            self.builder.build_branch(first_test);
        } else {
            self.builder.build_branch(default_block);
            self.builder.switch_to_block(continuation);
            return;
        }

        // Per-case end block + post-body symbol_map snapshots, used to wire
        // phi node incoming edges at the continuation.
        let mut case_incoming: Vec<(IrBlockId, BTreeMap<SymbolId, IrId>)> = Vec::new();

        for (i, case) in cases.iter().enumerate() {
            let test_block = case_test_blocks[i];
            let body_block = case_body_blocks[i];
            let next_test = case_test_blocks
                .get(i + 1)
                .copied()
                .unwrap_or(default_block);

            self.builder.switch_to_block(test_block);

            // A case whose pattern list holds only wildcards (or is empty) is
            // unconditional — emit a plain branch rather than
            // `br_if true, body, next_test`. Otherwise next_test and the
            // downstream default/continuation blocks linger in the CFG with the
            // wrong terminator type (e.g. `ret void` inside a non-void function),
            // which Cranelift turns into invalid machine code even though it is
            // runtime-unreachable. The guard arm below keeps its conditional
            // branch: a guard can still fail, so next_test stays reachable.
            // Only the constructor-pattern switch path reaches here; other
            // patterns are desugared to an if/else chain in tast_to_hir, which
            // has its own wildcard short-circuit.
            let all_wildcards = !case.patterns.is_empty()
                && case
                    .patterns
                    .iter()
                    .all(|p| matches!(p, HirPattern::Wildcard));

            if all_wildcards && case.guard.is_none() {
                self.builder.build_branch(body_block);
            } else {
                let pattern_matches = if case.patterns.is_empty() {
                    // No pattern means default case
                    self.builder.build_bool(true)
                } else if case.patterns.len() == 1 {
                    self.lower_pattern_test_with_scrutinee_type(
                        scrut_val,
                        &case.patterns[0],
                        Some(scrutinee.ty),
                    )
                } else {
                    // Multiple patterns per case: OR them all together
                    let mut result = self.lower_pattern_test_with_scrutinee_type(
                        scrut_val,
                        &case.patterns[0],
                        Some(scrutinee.ty),
                    );
                    for pat in &case.patterns[1..] {
                        if let Some(prev) = result {
                            if let Some(pat_match) = self.lower_pattern_test_with_scrutinee_type(
                                scrut_val,
                                pat,
                                Some(scrutinee.ty),
                            ) {
                                result = self.builder.build_binop(BinaryOp::Or, prev, pat_match);
                            }
                        }
                    }
                    result
                };

                let pattern_matches = match pattern_matches {
                    Some(v) => v,
                    None => {
                        self.builder.build_branch(next_test);
                        continue;
                    }
                };

                if let Some(ref guard) = case.guard {
                    let guard_block = match self.builder.create_block() {
                        Some(b) => b,
                        None => return,
                    };

                    self.builder
                        .build_cond_branch(pattern_matches, guard_block, next_test);

                    self.builder.switch_to_block(guard_block);
                    let guard_val = match self.lower_expression(guard) {
                        Some(v) => v,
                        None => {
                            self.builder.build_branch(next_test);
                            continue;
                        }
                    };

                    self.builder
                        .build_cond_branch(guard_val, body_block, next_test);
                } else {
                    self.builder
                        .build_cond_branch(pattern_matches, body_block, next_test);
                }
            }

            // Snapshot the symbol_map entries for tracked vars BEFORE the body,
            // so each case starts from the pre-switch state instead of the
            // previous case's writes.
            for symbol_id in var_initial_values.keys() {
                if let Some(&(reg, _)) = var_initial_values.get(symbol_id) {
                    self.symbol_map.insert(*symbol_id, reg);
                }
            }

            self.builder.switch_to_block(body_block);
            // Bind pattern variables (extract enum fields into variable symbols)
            if !case.patterns.is_empty() {
                self.bind_pattern_with_scrutinee_type(
                    &case.patterns[0],
                    scrut_val,
                    Some(scrutinee.ty),
                );
            }
            self.lower_block(&case.body);
            if !self.is_terminated() {
                let end = self.builder.current_block();
                let mut snapshot: BTreeMap<SymbolId, IrId> = BTreeMap::new();
                for symbol_id in var_initial_values.keys() {
                    if let Some(&reg) = self.symbol_map.get(symbol_id) {
                        snapshot.insert(*symbol_id, reg);
                    }
                }
                self.builder.build_branch(continuation);
                if let Some(end_block) = end {
                    case_incoming.push((end_block, snapshot));
                }
            }
        }

        // Restore tracked vars to their initial values for the default path.
        for symbol_id in var_initial_values.keys() {
            if let Some(&(reg, _)) = var_initial_values.get(symbol_id) {
                self.symbol_map.insert(*symbol_id, reg);
            }
        }
        self.builder.switch_to_block(default_block);
        let default_end = self.builder.current_block();
        let mut default_snapshot: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, _)) in &var_initial_values {
            default_snapshot.insert(*symbol_id, *initial_reg);
        }
        self.builder.build_branch(continuation);
        if let Some(end_block) = default_end {
            case_incoming.push((end_block, default_snapshot));
        }

        self.builder.switch_to_block(continuation);

        // Build phi nodes for variables modified in any case so the
        // post-switch code sees the matched case's value (and SSA stays valid).
        for (symbol_id, (initial_reg, var_type)) in &var_initial_values {
            // Skip if no case actually changed the variable.
            let any_changed = case_incoming.iter().any(|(_, snap)| {
                snap.get(symbol_id).copied().unwrap_or(*initial_reg) != *initial_reg
            });
            if !any_changed {
                continue;
            }
            if let Some(phi_reg) = self.builder.build_phi(continuation, var_type.clone()) {
                for (end_block, snapshot) in &case_incoming {
                    let val = snapshot.get(symbol_id).copied().unwrap_or(*initial_reg);
                    self.builder
                        .add_phi_incoming(continuation, phi_reg, *end_block, val);
                }
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(local) = func.locals.get(initial_reg).cloned() {
                        func.locals.insert(
                            phi_reg,
                            crate::ir::IrLocal {
                                name: format!("{}_phi", local.name),
                                ty: var_type.clone(),
                                mutable: true,
                                source_location: local.source_location,
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }
                self.symbol_map.insert(*symbol_id, phi_reg);
            }
        }
        let _ = entry_block;
    }

    pub(crate) fn lower_pattern_test(
        &mut self,
        scrutinee: IrId,
        pattern: &HirPattern,
    ) -> Option<IrId> {
        self.lower_pattern_test_with_scrutinee_type(scrutinee, pattern, None)
    }

    pub(crate) fn lower_pattern_test_with_scrutinee_type(
        &mut self,
        scrutinee: IrId,
        pattern: &HirPattern,
        scrutinee_type: Option<TypeId>,
    ) -> Option<IrId> {
        // Returns a boolean IrId indicating match success.
        match pattern {
            HirPattern::Variable { name, symbol } => {
                // Variable pattern always matches. Don't bind here — binding happens
                // in bind_pattern_with_scrutinee_type() in the body block, which ensures
                // the register is defined in the correct SSA block.
                self.builder.build_bool(true)
            }

            HirPattern::Wildcard => {
                // Wildcard always matches
                self.builder.build_bool(true)
            }

            HirPattern::Literal(lit) => {
                // TODO: take the type from the pattern context instead of
                // guessing it from the literal kind.
                let default_type = match lit {
                    HirLiteral::Int(_) => TypeId::from_raw(1), // Assume Int type (ID 1)
                    HirLiteral::Float(_) => TypeId::from_raw(2), // Assume Float type
                    HirLiteral::Bool(_) => TypeId::from_raw(3), // Assume Bool type
                    HirLiteral::String(_) => TypeId::from_raw(4), // Assume String type
                    _ => TypeId::from_raw(1),                  // Default to Int
                };
                let lit_val = self.lower_literal(lit, default_type)?;
                // TODO: Use proper comparison based on type
                self.builder.build_cmp(CompareOp::Eq, scrutinee, lit_val)
            }

            HirPattern::Constructor {
                enum_type,
                variant,
                fields,
            } => {
                // Constructor pattern: check enum tag and optionally extract fields.
                // Resolve whether this enum uses boxed or unboxed representation.
                let effective_enum_type = scrutinee_type
                    .filter(|t| *t != TypeId::invalid())
                    .filter(|t| self.resolve_enum_symbol(*t).is_some())
                    .unwrap_or(*enum_type);
                let enum_symbol = self.resolve_enum_symbol(effective_enum_type);
                let mut is_boxed = enum_symbol.map_or(false, |s| self.enum_is_boxed(s));

                // The scrutinee's register type decides the representation: an I32
                // scalar is a plain discriminant, a Ptr is a heap-allocated box
                // (e.g. Type.typeof() returning ValueType).
                let scrut_type = self.builder.get_register_type(scrutinee);
                if matches!(scrut_type, Some(IrType::I32)) {
                    is_boxed = false;
                } else if matches!(scrut_type, Some(IrType::Ptr(_))) {
                    is_boxed = true;
                }

                let variant_discriminant = self
                    .resolve_enum_variant_discriminant(effective_enum_type, *variant)
                    .or_else(|| self.resolve_enum_variant_discriminant(*enum_type, *variant))
                    .unwrap_or(0);

                if !is_boxed {
                    // Unboxed enum: scrutinee IS the discriminant (i32 or i64)
                    let scrut_ir_type = scrut_type.unwrap_or(IrType::I64);
                    let expected = self
                        .builder
                        .build_int(variant_discriminant, scrut_ir_type)?;
                    return self.builder.build_cmp(CompareOp::Eq, scrutinee, expected);
                }

                // Boxed enum: scrutinee is ptr-as-i64, bitcast to pointer
                let enum_ptr = self
                    .builder
                    .build_bitcast(scrutinee, IrType::Ptr(Box::new(IrType::I8)))?;

                // Load tag at offset 0
                let zero_offset = self.builder.build_int(0, IrType::I64)?;
                let tag_gep = self.builder.build_gep(
                    enum_ptr,
                    vec![zero_offset],
                    IrType::Ptr(Box::new(IrType::I8)),
                )?;
                let tag_ptr = self
                    .builder
                    .build_bitcast(tag_gep, IrType::Ptr(Box::new(IrType::I32)))?;
                let tag_val = self.builder.build_load(tag_ptr, IrType::I32)?;

                let expected_tag = self.builder.build_int(variant_discriminant, IrType::I32)?;
                let tag_matches = self
                    .builder
                    .build_cmp(CompareOp::Eq, tag_val, expected_tag)?;

                if fields.is_empty() || fields.iter().all(|f| matches!(f, HirPattern::Wildcard)) {
                    return Some(tag_matches);
                }

                // Field extraction must short-circuit behind the tag check: other
                // variants may have smaller allocations (None is 8 bytes of tag),
                // so loading at offset 8+ is out of bounds when the tag differs.
                let false_val = self.builder.build_const(IrValue::Bool(false))?;
                let tag_check_block = self.builder.current_block()?;
                let fields_block = self.builder.create_block()?;
                let merge_block = self.builder.create_block()?;
                self.builder
                    .build_cond_branch(tag_matches, fields_block, merge_block);

                // fields_block: tag matched, extract and test fields.
                self.builder.switch_to_block(fields_block);
                let mut all_fields_match = None;

                for (i, field_pattern) in fields.iter().enumerate() {
                    // Field at byte offset 8 + i*8
                    let field_offset = self.builder.build_int((8 + i * 8) as i64, IrType::I64)?;
                    let field_gep = self.builder.build_gep(
                        enum_ptr,
                        vec![field_offset],
                        IrType::Ptr(Box::new(IrType::I8)),
                    )?;
                    let field_ptr = self
                        .builder
                        .build_bitcast(field_gep, IrType::Ptr(Box::new(IrType::I64)))?;
                    let field_val = self.builder.build_load(field_ptr, IrType::I64)?;

                    let field_match = self.lower_pattern_test(field_val, field_pattern)?;
                    all_fields_match = Some(match all_fields_match {
                        Some(prev) => self.builder.build_binop(BinaryOp::And, prev, field_match)?,
                        None => field_match,
                    });
                }

                let fields_result = all_fields_match.unwrap_or(tag_matches);
                self.builder.build_branch(merge_block);
                let fields_exit_block = self.builder.current_block()?;

                // merge_block: phi(fields_result | false).
                self.builder.switch_to_block(merge_block);
                let result = self.builder.build_phi(merge_block, IrType::Bool)?;
                self.builder.add_phi_incoming(
                    merge_block,
                    result,
                    fields_exit_block,
                    fields_result,
                );
                self.builder
                    .add_phi_incoming(merge_block, result, tag_check_block, false_val);

                Some(result)
            }

            HirPattern::Tuple(patterns) => {
                // Layout: struct Tuple { elem0, elem1, ... }; every element is
                // tested against its pattern and the results ANDed.
                if patterns.is_empty() {
                    // Empty tuple always matches
                    return self.builder.build_bool(true);
                }

                let mut all_match = self.builder.build_bool(true)?;

                for (i, elem_pattern) in patterns.iter().enumerate() {
                    let Some(elem_idx) = self.builder.build_int(i as i64, IrType::I64) else {
                        return None;
                    };

                    let Some(elem_ptr) = self.builder.build_gep(
                        scrutinee,
                        vec![elem_idx],
                        IrType::Ptr(Box::new(IrType::Any)),
                    ) else {
                        return None;
                    };

                    let Some(elem_val) = self.builder.build_load(elem_ptr, IrType::Any) else {
                        return None;
                    };

                    let Some(elem_match) = self.lower_pattern_test(elem_val, elem_pattern) else {
                        return None;
                    };

                    all_match = self
                        .builder
                        .build_binop(BinaryOp::And, all_match, elem_match)?;
                }

                Some(all_match)
            }

            HirPattern::Array { elements, rest } => {
                // Layout: struct Array { length: i64, data: [elements...] }, so the
                // length lives at index 0.
                let Some(zero_idx) = self.builder.build_int(0, IrType::I64) else {
                    return None;
                };

                let Some(length_ptr) = self.builder.build_gep(
                    scrutinee,
                    vec![zero_idx],
                    IrType::Ptr(Box::new(IrType::I64)),
                ) else {
                    return None;
                };

                let Some(array_length) = self.builder.build_load(length_ptr, IrType::I64) else {
                    return None;
                };

                let mut all_match = self.builder.build_bool(true)?;

                // If no rest pattern, check exact length
                if rest.is_none() {
                    let Some(expected_len) =
                        self.builder.build_int(elements.len() as i64, IrType::I64)
                    else {
                        return None;
                    };

                    let Some(length_matches) =
                        self.builder
                            .build_cmp(CompareOp::Eq, array_length, expected_len)
                    else {
                        return None;
                    };

                    all_match =
                        self.builder
                            .build_binop(BinaryOp::And, all_match, length_matches)?;
                } else {
                    // With rest pattern, check minimum length
                    let Some(min_len) = self.builder.build_int(elements.len() as i64, IrType::I64)
                    else {
                        return None;
                    };

                    let Some(length_sufficient) =
                        self.builder.build_cmp(CompareOp::Ge, array_length, min_len)
                    else {
                        return None;
                    };

                    all_match =
                        self.builder
                            .build_binop(BinaryOp::And, all_match, length_sufficient)?;
                }

                for (i, elem_pattern) in elements.iter().enumerate() {
                    // Array elements start at index 1 (after length header)
                    let Some(elem_idx) = self.builder.build_int((i + 1) as i64, IrType::I64) else {
                        return None;
                    };

                    let Some(elem_ptr) = self.builder.build_gep(
                        scrutinee,
                        vec![elem_idx],
                        IrType::Ptr(Box::new(IrType::Any)),
                    ) else {
                        return None;
                    };

                    let Some(elem_val) = self.builder.build_load(elem_ptr, IrType::Any) else {
                        return None;
                    };

                    let Some(elem_match) = self.lower_pattern_test(elem_val, elem_pattern) else {
                        return None;
                    };

                    all_match = self
                        .builder
                        .build_binop(BinaryOp::And, all_match, elem_match)?;
                }

                // TODO: bind the rest pattern to a slice of the remaining elements.

                Some(all_match)
            }

            HirPattern::Object { fields, rest } => {
                // Each named field is tested against its pattern and the results
                // ANDed; `rest` says whether extra fields are allowed.
                if fields.is_empty() {
                    // Empty object pattern always matches (or matches any object if rest=true)
                    return self.builder.build_bool(true);
                }

                let mut all_match = self.builder.build_bool(true)?;

                for (field_name, field_pattern) in fields {
                    // TODO: real field lookup by name; this offset is a
                    // placeholder derived from the name's length.
                    let field_offset = self.interned_str(*field_name).len() as i64;

                    let Some(field_idx) = self.builder.build_int(field_offset, IrType::I64) else {
                        return None;
                    };

                    let Some(field_ptr) = self.builder.build_gep(
                        scrutinee,
                        vec![field_idx],
                        IrType::Ptr(Box::new(IrType::Any)),
                    ) else {
                        return None;
                    };

                    let Some(field_val) = self.builder.build_load(field_ptr, IrType::Any) else {
                        return None;
                    };

                    let Some(field_match) = self.lower_pattern_test(field_val, field_pattern)
                    else {
                        return None;
                    };

                    all_match = self
                        .builder
                        .build_binop(BinaryOp::And, all_match, field_match)?;
                }

                // TODO: when rest=false, verify no additional fields exist.

                Some(all_match)
            }

            HirPattern::Typed { pattern, ty } => {
                // TODO: check the type; only the inner pattern is tested today.
                self.lower_pattern_test_with_scrutinee_type(scrutinee, pattern, scrutinee_type)
            }

            HirPattern::Or(patterns) => {
                if patterns.is_empty() {
                    return self.builder.build_bool(false);
                }
                let mut result = self.lower_pattern_test_with_scrutinee_type(
                    scrutinee,
                    &patterns[0],
                    scrutinee_type,
                )?;
                for pat in &patterns[1..] {
                    let pat_match = self.lower_pattern_test_with_scrutinee_type(
                        scrutinee,
                        pat,
                        scrutinee_type,
                    )?;
                    result = self.builder.build_binop(BinaryOp::Or, result, pat_match)?;
                }
                Some(result)
            }

            HirPattern::Guard { pattern, condition } => {
                let pattern_match = self.lower_pattern_test_with_scrutinee_type(
                    scrutinee,
                    pattern,
                    scrutinee_type,
                )?;
                let guard_val = self.lower_expression(condition)?;
                self.builder
                    .build_binop(BinaryOp::And, pattern_match, guard_val)
            }
        }
    }
}
