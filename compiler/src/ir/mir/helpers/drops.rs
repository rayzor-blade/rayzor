//! Drop scopes: registration, scope exit, and the emitted drop and free calls.

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
    /// Enter a new scope for drop tracking
    pub(crate) fn enter_drop_scope(&mut self) {
        self.drop_scope_stack.push(Vec::new());
    }

    /// Exit a scope, emitting Free instructions for all owned heap values in that scope
    pub(crate) fn exit_drop_scope(&mut self) {
        if let Some(scope) = self.drop_scope_stack.pop() {
            for (symbol, scope_ir_id) in scope {
                // Get the CURRENT value from owned_heap_values, not the stale scope entry.
                // The scope entry might have an old IrId if the variable was reassigned.
                let current_ir_id = match self.owned_heap_values.get(&symbol).copied() {
                    Some(id) => id,
                    None => {
                        // Variable was already freed or transferred (e.g., to closure)
                        continue;
                    }
                };

                // Skip lambda captures - they're owned by the closure
                if let Some(drop_points) = &self.current_drop_points {
                    if drop_points.lambda_captures.contains(&symbol) {
                        continue;
                    }
                }

                // Loop-carried escape: if this symbol was declared in an enclosing
                // scope and merely assigned inside the loop body, its value flows
                // out through the loop-carried / exit phi and is read AFTER the
                // loop. Freeing it here (at loop-body exit) is a use-after-free.
                // Defer the free: keep it owned and re-register the ownership in
                // the enclosing scope so it is dropped when THAT scope exits
                // (the loop-exit code repoints owned_heap_values to the exit phi,
                // which dominates the post-loop reads).
                if self
                    .loop_carried_symbols
                    .last()
                    .map_or(false, |s| s.contains(&symbol))
                {
                    if std::env::var("RAYZOR_DBG_LOOPCARRY").is_ok() {
                        eprintln!(
                            "[RAYZOR_LOOPCARRY_SKIP_FREE] deferring free of loop-carried {:?}",
                            symbol
                        );
                    }
                    if let Some(enclosing) = self.drop_scope_stack.last_mut() {
                        enclosing.push((symbol, scope_ir_id));
                    }
                    continue;
                }

                self.emit_tracked_free(current_ir_id, true);
                self.owned_heap_values.remove(&symbol);
            }
        }
    }

    /// Cleanup all scopes - used for early return from functions
    /// Frees all heap values in all active scopes (innermost to outermost)
    pub(crate) fn cleanup_all_scopes(&mut self) {
        self.cleanup_all_scopes_except_symbol(None);
    }

    pub(crate) fn cleanup_all_scopes_except_symbol(&mut self, skip_symbol: Option<SymbolId>) {
        // Collect IrIds to free first (to avoid borrow conflict with maybe_emit_drop_call)
        let mut to_free = Vec::new();
        for scope in self.drop_scope_stack.iter().rev() {
            for (symbol, ir_id) in scope {
                // Skip the returned variable — its value escapes the function
                if skip_symbol == Some(*symbol) {
                    trace!(
                        "Drop: Skipping {:?} ({:?}) in cleanup (returned value)",
                        symbol,
                        ir_id
                    );
                    continue;
                }
                // Skip lambda captures
                if let Some(drop_points) = &self.current_drop_points {
                    if drop_points.lambda_captures.contains(symbol) {
                        trace!("Drop: Skipping {:?} in cleanup (lambda capture)", symbol);
                        continue;
                    }
                }

                if self.reassigned_in_scope.contains(symbol) {
                    // For reassigned variables, free the CURRENT value from owned_heap_values
                    // (the old value was already freed at reassignment time)
                    if let Some(&current_ir) = self.owned_heap_values.get(symbol) {
                        to_free.push((*symbol, current_ir));
                    }
                } else {
                    to_free.push((*symbol, *ir_id));
                }
            }
        }

        // Only @:derive(Drop) classes are freed here. Non-Drop heap allocations
        // belong to InsertFreePass, which has the escape analysis needed to see
        // a local passed on through a constructor argument; freeing them here
        // would be a use-after-free on the escaped value.
        for (symbol, ir_id) in to_free {
            if !self.is_terminated() && self.get_drop_class_for_ir(ir_id).is_some() {
                self.emit_tracked_free(ir_id, true);
                trace!(
                    "Drop: Freed {:?} ({:?}) in cleanup_all_scopes (Drop class)",
                    symbol,
                    ir_id
                );
            }
        }
        // Scopes are left intact: the function is about to return and they are
        // cleared with the function context.
    }

    /// Cleanup only @:derive(Drop) class values at implicit function end.
    /// Unlike cleanup_all_scopes(), this only touches Drop-class values to avoid
    /// double-freeing non-Drop values that InsertFreePass handles.
    pub(crate) fn cleanup_drop_classes_only(&mut self) {
        let mut to_free = Vec::new();
        for scope in self.drop_scope_stack.iter().rev() {
            for (symbol, ir_id) in scope {
                // Skip lambda captures
                if let Some(drop_points) = &self.current_drop_points {
                    if drop_points.lambda_captures.contains(symbol) {
                        continue;
                    }
                }

                // Get the current IrId (may differ from scope entry if reassigned)
                let current_ir = if self.reassigned_in_scope.contains(symbol) {
                    match self.owned_heap_values.get(symbol).copied() {
                        Some(ir) => ir,
                        None => continue,
                    }
                } else {
                    *ir_id
                };

                // Only process Drop-class values
                if self.get_drop_class_for_ir(current_ir).is_some() {
                    to_free.push(current_ir);
                }
            }
        }

        for ir_id in to_free {
            self.emit_tracked_free(ir_id, true);
        }
    }

    /// Free all temporary values (called after expression completes)
    pub(crate) fn drop_temps(&mut self) {
        for ir_id in std::mem::take(&mut self.temp_heap_values) {
            self.builder.build_free(ir_id);
        }
    }

    /// Check if a type needs drop (convenience wrapper for get_drop_behavior)
    pub(crate) fn type_needs_drop(&self, type_id: TypeId) -> bool {
        matches!(
            self.get_drop_behavior(type_id),
            DropBehavior::AutoDrop | DropBehavior::AutoDropWithDtor
        )
    }

    /// Called before Free at scope exit, reassignment, and early return.
    pub(crate) fn emit_drop_call(&mut self, obj_reg: IrId, class_sym: SymbolId) {
        let drop_name = self.string_interner.intern("drop");
        let method_sym = match self.resolve_class_method_symbol(class_sym, drop_name) {
            Some(sym) => sym,
            None => {
                trace!(
                    "Drop: Could not find drop() method for class {:?}",
                    class_sym
                );
                return;
            }
        };
        let func_id = match self.get_function_id(&method_sym) {
            Some(id) => id,
            None => {
                trace!(
                    "Drop: Could not find IrFunctionId for drop() method {:?}",
                    method_sym
                );
                return;
            }
        };
        // Call drop(this) — the object pointer is passed as the receiver
        self.builder
            .build_call_direct(func_id, vec![obj_reg], IrType::Void);
    }

    /// @:derive(Drop) — conditionally emit drop() call before Free for an IrId.
    /// Returns true if a drop call was emitted.
    pub(crate) fn maybe_emit_drop_call(&mut self, ir_id: IrId) -> bool {
        if let Some(class_sym) = self.get_drop_class_for_ir(ir_id) {
            self.emit_drop_call(ir_id, class_sym);
            true
        } else {
            false
        }
    }

    /// Drop+free a tracked value only when its definition dominates this point.
    /// Tracker state (`owned_heap_values` / drop scopes) is linear over lowering
    /// order, so a value registered in one arm of an if/else is visible to the
    /// sibling arm; a free emitted there would name a register that never
    /// materialized and alias an unrelated live object. Skipping the free trades
    /// a bounded leak for correctness.
    pub(crate) fn emit_tracked_free(&mut self, ir: IrId, with_drop_call: bool) {
        if self.is_terminated() {
            return;
        }
        let Some(cur) = self.builder.current_block else {
            return;
        };
        match self.def_block_of(ir) {
            Some(def_bb) if self.block_dominates(def_bb, cur) => {
                if with_drop_call {
                    self.maybe_emit_drop_call(ir);
                }
                self.builder.build_free(ir);
            }
            _ => {}
        }
    }

    /// Check drop points and emit Free for variables at their last use
    /// Called after each statement during lowering
    pub(crate) fn check_drop_points_after_statement(&mut self) {
        use crate::ir::drop_analysis::should_drop_at_statement;

        let drop_points = match &self.current_drop_points {
            Some(dp) => dp,
            None => return,
        };

        let current_idx = self.current_stmt_index;
        let mut to_drop = Vec::new();

        for (&symbol, &ir_id) in &self.owned_heap_values {
            if should_drop_at_statement(drop_points, symbol, current_idx) {
                to_drop.push((symbol, ir_id));
            }
        }

        for (symbol, ir_id) in to_drop {
            self.emit_tracked_free(ir_id, false);
            self.owned_heap_values.remove(&symbol);
        }
    }
}
