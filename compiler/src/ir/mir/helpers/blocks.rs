//! Basic-block bookkeeping: termination, dominance, and saved lowering state.

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
    pub(crate) fn is_terminated(&self) -> bool {
        let block_id = match self.builder.current_block() {
            Some(id) => id,
            None => return false,
        };

        // A block is terminated if either:
        // 1. Its terminator is non-default (Branch/CondBranch/Return/Switch/NoReturn), or
        // 2. The terminator was explicitly set (covers deliberate `Unreachable`
        //    after `throw`/`panic`, which shares the variant with the default).
        self.builder
            .current_function()
            .and_then(|func| func.cfg.get_block(block_id))
            .map(|block| block.is_terminated() || block.terminator_explicit)
            .unwrap_or(false)
    }

    /// `def_bb` dominates `use_bb` on the CFG built so far: no entry→use
    /// path avoids def. (BFS from entry skipping def; dominated ⟺ use
    /// unreachable.)
    pub(crate) fn block_dominates(&self, def_bb: IrBlockId, use_bb: IrBlockId) -> bool {
        if def_bb == use_bb {
            return true;
        }
        let Some(func_id) = self.builder.current_function else {
            return false;
        };
        let Some(func) = self.builder.module.functions.get(&func_id) else {
            return false;
        };
        let entry = func.cfg.entry_block;
        if def_bb == entry {
            return true;
        }
        let mut seen = BTreeSet::new();
        let mut work = vec![entry];
        while let Some(b) = work.pop() {
            if b == def_bb || !seen.insert(b) {
                continue;
            }
            if b == use_bb {
                return false;
            }
            if let Some(blk) = func.cfg.blocks.get(&b) {
                match &blk.terminator {
                    crate::ir::IrTerminator::Branch { target } => work.push(*target),
                    crate::ir::IrTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        work.push(*true_target);
                        work.push(*false_target);
                    }
                    crate::ir::IrTerminator::Switch { cases, default, .. } => {
                        work.extend(cases.iter().map(|(_, t)| *t));
                        work.push(*default);
                    }
                    _ => {}
                }
            }
        }
        true
    }

    /// @:derive(Drop) — emit a call to the user's drop() method on the given object.
    /// Block that defines `ir` in the currently-built function (params →
    /// entry block). None when no emitted instruction produces it.
    pub(crate) fn def_block_of(&self, ir: IrId) -> Option<IrBlockId> {
        let func_id = self.builder.current_function?;
        let func = self.builder.module.functions.get(&func_id)?;
        if func.signature.parameters.iter().any(|p| p.reg == ir) {
            return Some(func.cfg.entry_block);
        }
        for (bid, block) in &func.cfg.blocks {
            for inst in &block.instructions {
                if inst.dest() == Some(ir) {
                    return Some(*bid);
                }
            }
        }
        None
    }

    /// Type a loop-carried phi should use for `reg`. Normally the variable's
    /// declared local type — but when that diverges from the value's actual
    /// register type on float-vs-integer (e.g. a `Usize` address whose local was
    /// mis-inferred as `Float`), the phi MUST follow the VALUE. Otherwise the
    /// backend lowers the mismatched phi with a corrupting `sitofp` at the entry
    /// and a `bitcast` at the use, destroying the carried pointer → SIGSEGV.
    pub(crate) fn loop_phi_type(&self, reg: IrId, local_ty: IrType) -> IrType {
        match self.builder.get_register_type(reg) {
            Some(rt)
                if matches!(rt, IrType::F32 | IrType::F64)
                    != matches!(local_ty, IrType::F32 | IrType::F64) =>
            {
                rt
            }
            _ => local_ty,
        }
    }

    // ========================================================================
    // Two-Pass Lambda Generation (New Architecture) - Helper Methods
    // ========================================================================

    pub(crate) fn save_state(&self) -> SavedLoweringState {
        SavedLoweringState {
            current_function: self.builder.current_function,
            current_block: self.builder.current_block,
            symbol_map: self.symbol_map.clone(),
            current_env_layout: self.current_env_layout.clone(),
            // Save drop tracking state so lambda bodies don't inherit parent's owned values
            owned_heap_values: self.owned_heap_values.clone(),
            drop_scope_stack: self.drop_scope_stack.clone(),
            loop_carried_symbols: self.loop_carried_symbols.clone(),
            temp_heap_values: self.temp_heap_values.clone(),
            reassigned_in_scope: self.reassigned_in_scope.clone(),
            current_drop_points: self.current_drop_points.clone(),
            current_stmt_index: self.current_stmt_index,
            boxed_dynamic_symbols: self.boxed_dynamic_symbols.clone(),
            value_type_symbols: self.value_type_symbols.clone(),
            anon_views: self.anon_views.clone(),
            raw_anon_symbols: self.raw_anon_symbols.clone(),
            strict_move_locals: self.strict_move_locals.clone(),
        }
    }

    pub(crate) fn restore_state(&mut self, state: SavedLoweringState) {
        self.builder.current_function = state.current_function;
        self.builder.current_block = state.current_block;
        self.symbol_map = state.symbol_map;
        self.current_env_layout = state.current_env_layout;
        // Restore drop tracking state
        self.owned_heap_values = state.owned_heap_values;
        self.drop_scope_stack = state.drop_scope_stack;
        self.loop_carried_symbols = state.loop_carried_symbols;
        self.temp_heap_values = state.temp_heap_values;
        self.reassigned_in_scope = state.reassigned_in_scope;
        self.current_drop_points = state.current_drop_points;
        self.current_stmt_index = state.current_stmt_index;
        self.boxed_dynamic_symbols = state.boxed_dynamic_symbols;
        self.value_type_symbols = state.value_type_symbols;
        self.anon_views = state.anon_views;
        self.raw_anon_symbols = state.raw_anon_symbols;
        self.strict_move_locals = state.strict_move_locals;
    }

    /// Emit a while loop that calls hasNext/next on an iterator object.
    /// Used by lower_for_in_iterator_protocol for both typed and Dynamic iterators.
    pub(crate) fn emit_iterator_while_loop(
        &mut self,
        pattern: &HirPattern,
        obj_reg: IrId,
        has_next_fn: IrFunctionId,
        next_fn: IrFunctionId,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) {
        // Store obj pointer to a stack slot so it survives across blocks
        let obj_ty = self
            .builder
            .get_register_type(obj_reg)
            .unwrap_or(IrType::Ptr(Box::new(IrType::Void)));
        let Some(obj_slot) = self.builder.build_alloc(obj_ty.clone(), None) else {
            return;
        };
        self.builder.build_store(obj_slot, obj_reg);

        // Also store mutable variables from outer scope in stack slots
        // so assignments in the loop body are visible after the loop
        let modified_vars = {
            let mut modified = std::collections::BTreeSet::new();
            for stmt in &body.statements {
                self.find_modified_variables_in_statement(stmt, &mut modified);
            }
            modified
        };

        // Create stack slots for each modified variable
        let mut var_slots: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for sym in &modified_vars {
            if let Some(&current_reg) = self.symbol_map.get(sym) {
                let ty = self
                    .builder
                    .get_register_type(current_reg)
                    .unwrap_or(IrType::I64);
                if let Some(slot) = self.builder.build_alloc(ty.clone(), None) {
                    self.builder.build_store(slot, current_reg);
                    var_slots.insert(*sym, (slot, ty));
                }
            }
        }

        // Create loop blocks
        let Some(loop_cond_block) = self.builder.create_block() else {
            return;
        };
        let Some(loop_body_block) = self.builder.create_block() else {
            return;
        };
        let Some(loop_exit_block) = self.builder.create_block() else {
            return;
        };

        self.builder.build_branch(loop_cond_block);

        // Push loop context
        self.loop_stack.push(LoopContext {
            continue_block: loop_cond_block,
            break_block: loop_exit_block,
            label: label.cloned(),
            exit_phi_nodes: BTreeMap::new(),
            continue_phi_nodes: BTreeMap::new(),
        });

        // Condition block: call hasNext()
        self.builder.switch_to_block(loop_cond_block);
        let Some(obj_for_cond) = self.builder.build_load(obj_slot, obj_ty.clone()) else {
            self.loop_stack.pop();
            return;
        };
        let Some(has_next_result) =
            self.builder
                .build_call_direct(has_next_fn, vec![obj_for_cond], IrType::Bool)
        else {
            self.loop_stack.pop();
            return;
        };
        self.builder
            .build_cond_branch(has_next_result, loop_body_block, loop_exit_block);

        // Body block
        self.builder.switch_to_block(loop_body_block);

        // Load modified vars from slots at body entry
        for (sym, (slot, ty)) in &var_slots {
            if let Some(loaded) = self.builder.build_load(*slot, ty.clone()) {
                self.symbol_map.insert(*sym, loaded);
            }
        }

        // Call next() and bind the pattern
        let Some(obj_for_body) = self.builder.build_load(obj_slot, obj_ty) else {
            self.loop_stack.pop();
            return;
        };
        let Some(next_value) =
            self.builder
                .build_call_direct(next_fn, vec![obj_for_body], IrType::I64)
        else {
            self.loop_stack.pop();
            return;
        };

        match pattern {
            HirPattern::Variable { symbol, .. } => {
                self.symbol_map.insert(*symbol, next_value);
            }
            HirPattern::Tuple(sub_patterns) if sub_patterns.len() == 2 => {
                // Destructure {key, value} from next() return (anon object handle)
                // Fields are alphabetically sorted: key at index 0, value at index 1
                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                let get_field_fn = self.get_or_register_extern_function(
                    "rayzor_anon_get_field_by_index",
                    vec![ptr_void.clone(), IrType::I32],
                    IrType::I64,
                );
                // key = rayzor_anon_get_field_by_index(next_value, 0)
                if let Some(idx_0) = self.builder.build_const(crate::ir::IrValue::I32(0)) {
                    if let Some(key_val) = self.builder.build_call_direct(
                        get_field_fn,
                        vec![next_value, idx_0],
                        IrType::I64,
                    ) {
                        if let HirPattern::Variable { symbol, .. } = &sub_patterns[0] {
                            self.symbol_map.insert(*symbol, key_val);
                        }
                    }
                }
                // value = rayzor_anon_get_field_by_index(next_value, 1)
                if let Some(idx_1) = self.builder.build_const(crate::ir::IrValue::I32(1)) {
                    if let Some(value_val) = self.builder.build_call_direct(
                        get_field_fn,
                        vec![next_value, idx_1],
                        IrType::I64,
                    ) {
                        if let HirPattern::Variable { symbol, .. } = &sub_patterns[1] {
                            self.symbol_map.insert(*symbol, value_val);
                        }
                    }
                }
            }
            _ => {}
        }

        // Track loop-carried symbols (stored back to slots below) so the body's
        // exit_drop_scope does not free values that escape via the slot and are
        // read after the loop (see lower_for_in_over_array).
        self.loop_carried_symbols
            .push(var_slots.keys().copied().collect());
        // Lower the loop body
        self.enter_drop_scope();
        self.lower_block(body);

        // Store modified vars back to slots before branching back
        if !self.is_terminated() {
            for (sym, (slot, _ty)) in &var_slots {
                if let Some(&current_reg) = self.symbol_map.get(sym) {
                    self.builder.build_store(*slot, current_reg);
                }
            }
            self.exit_drop_scope();
            self.builder.build_branch(loop_cond_block);
        } else {
            self.exit_drop_scope();
        }
        self.loop_carried_symbols.pop();

        self.loop_stack.pop();

        // Exit block: load final values from slots
        self.builder.switch_to_block(loop_exit_block);
        for (sym, (slot, ty)) in &var_slots {
            if let Some(loaded) = self.builder.build_load(*slot, ty.clone()) {
                self.symbol_map.insert(*sym, loaded);
            }
        }
    }

    /// Infer the return type from generated MIR instructions
    pub(crate) fn infer_lambda_return_type(
        &self,
        function: &IrFunction,
        entry_block: IrBlockId,
        body_result: Option<IrId>,
    ) -> IrType {
        use crate::ir::IrInstruction;

        // Strategy 1: Search ALL blocks for Return terminators (not just entry block)
        // Lambdas with complex control flow (loops, conditionals) may have return in other blocks
        for (block_id, block) in &function.cfg.blocks {
            if let IrTerminator::Return {
                value: Some(ret_reg),
            } = &block.terminator
            {
                debug!(
                    "Found Return terminator in block {:?} with register {:?}",
                    block_id, ret_reg
                );

                // First try locals table (for parameters and explicitly registered values)
                if let Some(local) = function.locals.get(ret_reg) {
                    debug!("Found type in locals table: {:?}", local.ty);
                    return local.ty.clone();
                }

                // If not in locals table, scan ALL blocks for the instruction that defines this register
                debug!("Register not in locals, scanning all blocks for defining instruction...");
                for (search_block_id, search_block) in &function.cfg.blocks {
                    for inst in &search_block.instructions {
                        match inst {
                            IrInstruction::Load { dest, ty, .. } if *dest == *ret_reg => {
                                debug!(
                                    "Inferred type from Load instruction in block {:?}: {:?}",
                                    search_block_id, ty
                                );
                                return ty.clone();
                            }
                            IrInstruction::Cast { dest, to_ty, .. } if *dest == *ret_reg => {
                                debug!("Inferred type from Cast instruction: {:?}", to_ty);
                                return to_ty.clone();
                            }
                            IrInstruction::Const { dest, value, .. } if *dest == *ret_reg => {
                                // Infer from constant value
                                let ty = match value {
                                    IrValue::I32(_) => IrType::I32,
                                    IrValue::I64(_) => IrType::I64,
                                    IrValue::F64(_) => IrType::F64,
                                    IrValue::Bool(_) => IrType::Bool,
                                    IrValue::Null => IrType::Ptr(Box::new(IrType::Void)),
                                    _ => IrType::I64,
                                };
                                debug!("Inferred type from Const instruction: {:?}", ty);
                                return ty;
                            }
                            IrInstruction::Cmp { dest, .. } if *dest == *ret_reg => {
                                debug!("Inferred type from Cmp instruction: Bool");
                                return IrType::Bool;
                            }
                            IrInstruction::BinOp { dest, left, .. } if *dest == *ret_reg => {
                                // Infer from left operand type (BinOp preserves operand type)
                                if let Some(local) = function.locals.get(left) {
                                    debug!(
                                        "Inferred type from BinOp left operand local: {:?}",
                                        local.ty
                                    );
                                    return local.ty.clone();
                                }
                                if let Some(ty) = function.register_types.get(left) {
                                    debug!("Inferred type from BinOp left operand register_types: {:?}", ty);
                                    return ty.clone();
                                }
                                // BinOp on params is likely I64 (Haxe Int)
                                debug!("BinOp operand type unknown, defaulting to I64");
                                return IrType::I64;
                            }
                            IrInstruction::UnOp { dest, operand, .. } if *dest == *ret_reg => {
                                if let Some(local) = function.locals.get(operand) {
                                    debug!("Inferred type from UnOp operand local: {:?}", local.ty);
                                    return local.ty.clone();
                                }
                                if let Some(ty) = function.register_types.get(operand) {
                                    debug!(
                                        "Inferred type from UnOp operand register_types: {:?}",
                                        ty
                                    );
                                    return ty.clone();
                                }
                                return IrType::I64;
                            }
                            _ => {}
                        }
                    }
                }

                // Instruction scan didn't match — try register_types for the return register
                if let Some(ty) = function.register_types.get(ret_reg) {
                    debug!(
                        "Inferred return type from register_types for ret_reg: {:?}",
                        ty
                    );
                    return ty.clone();
                }
            }
        }

        // Strategy 2: Use body result register type from locals
        if let Some(result_reg) = body_result {
            if let Some(local) = function.locals.get(&result_reg) {
                debug!(
                    "Inferred return type from body result locals: {:?}",
                    local.ty
                );
                return local.ty.clone();
            }
        }

        // Strategy 3: Check register_types map (populated by build_binop, build_call_direct, etc.)
        if let Some(result_reg) = body_result {
            if let Some(ty) = function.register_types.get(&result_reg) {
                debug!("Inferred return type from register_types: {:?}", ty);
                return ty.clone();
            }
        }

        // Fallback: Void return
        debug!("No return type found, using Void");
        IrType::Void
    }

    /// Add terminator to lambda if not already present (static version)
    pub(crate) fn finalize_lambda_terminator_static(
        function: &mut IrFunction,
        entry_block: IrBlockId,
        body_result: Option<IrId>,
        return_type: &IrType,
    ) -> Option<()> {
        // Check if terminator already exists
        {
            let block = function.cfg.get_block_mut(entry_block)?;
            if !matches!(block.terminator, IrTerminator::Unreachable) {
                return Some(()); // Already has terminator
            }
        }

        // Create appropriate terminator
        let terminator = match return_type {
            IrType::Void => IrTerminator::Return { value: None },
            _ => {
                if let Some(result_reg) = body_result {
                    IrTerminator::Return {
                        value: Some(result_reg),
                    }
                } else {
                    // Create default value - allocate register first
                    let default_reg = function.alloc_reg();
                    let default_value = match return_type {
                        IrType::I32 => IrValue::I32(0),
                        IrType::I64 => IrValue::I64(0),
                        IrType::Bool => IrValue::Bool(false),
                        _ => IrValue::I64(0),
                    };

                    // Now get block and add instruction
                    let block = function.cfg.get_block_mut(entry_block)?;
                    block.add_instruction(IrInstruction::Const {
                        dest: default_reg,
                        value: default_value,
                    });

                    IrTerminator::Return {
                        value: Some(default_reg),
                    }
                }
            }
        };

        let block = function.cfg.get_block_mut(entry_block)?;
        block.set_terminator(terminator);
        Some(())
    }
}
