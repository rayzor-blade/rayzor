//! `while` and `do`/`while`.

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
    /// Lower while loop
    pub(crate) fn lower_while_loop(
        &mut self,
        condition: &HirExpr,
        body: &HirBlock,
        label: Option<&SymbolId>,
        continue_update: Option<&HirBlock>,
    ) {
        debug!("[lower_while_loop] ENTERED");
        let Some(cond_block) = self.builder.create_block() else {
            debug!("[lower_while_loop] FAILED to create cond_block");
            return;
        };
        let Some(body_block) = self.builder.create_block() else {
            debug!("[lower_while_loop] FAILED to create body_block");
            return;
        };
        let Some(exit_block) = self.builder.create_block() else {
            debug!("[lower_while_loop] FAILED to create exit_block");
            return;
        };
        // Create update block if there's a continue_update (for range/for loops)
        let update_block = if continue_update.is_some() {
            self.builder.create_block()
        } else {
            None
        };
        debug!(
            "[lower_while_loop] Created blocks: cond={:?}, body={:?}, exit={:?}",
            cond_block, body_block, exit_block
        );

        // Save the entry block (current block before loop)
        let entry_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            return;
        };

        // Find all variables that are referenced in the loop
        // For now, use a simple heuristic: any variable in the symbol_map
        // that is referenced in the condition or body is a potential loop variable
        //    // debug!("Loop body has {} statements", body.statements.len());
        //     for (i, stmt) in body.statements.iter().enumerate() {
        //        // debug!("Statement {}: {:?}", i, std::mem::discriminant(stmt));
        //     }

        // Collect all variables referenced in condition
        let mut referenced_vars = std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_expr(condition, &mut referenced_vars);

        // Collect all variables referenced in body
        self.collect_referenced_variables_in_block(body, &mut referenced_vars);

        // Also collect variables referenced in continue_update if present
        if let Some(upd) = continue_update {
            self.collect_referenced_variables_in_block(upd, &mut referenced_vars);
        }

        // Only include variables that were declared before the loop
        // (i.e., they're already in the symbol_map)
        // Exclude function parameters since they're immutable
        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| {
                let in_map = self.symbol_map.contains_key(sym);
                // Exclude only ACTUAL parameter symbols (by kind, not by
                // register — see is_parameter_symbol; a `var i = lo` local
                // aliases the param's register and must still get a phi).
                let is_param = self.is_parameter_symbol(sym);
                in_map && !is_param
            })
            .collect();

        // Save initial values of loop variables before jumping to condition
        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let local_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg))
                    .map(|local| local.ty.clone());
                if let Some(local_ty) = local_ty {
                    loop_var_initial_values
                        .insert(*symbol_id, (reg, self.loop_phi_type(reg, local_ty)));
                }
            }
        }

        // Jump to condition block
        self.builder.build_branch(cond_block);

        // Condition block - create phi nodes for loop variables
        self.builder.switch_to_block(cond_block);

        // Create phi nodes for all loop variables
        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            // debug!("Creating phi for symbol {:?}, initial reg {:?}", symbol_id, initial_reg);
            if let Some(phi_reg) = self.builder.build_phi(cond_block, var_type.clone()) {
                // debug!("Created phi node with dest {:?}", phi_reg);
                // Add incoming value from entry block
                self.builder
                    .add_phi_incoming(cond_block, phi_reg, entry_block, *initial_reg);
                // debug!("Added incoming edge from entry block {:?}", entry_block);

                // Register the phi node as a local so Cranelift can find its type
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

                // Update symbol map to use phi node
                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);

                // Also update owned_heap_values so drop tracking uses the phi result
                // This ensures that when a loop variable is reassigned, we free the
                // current iteration's value (from phi), not the initial value
                if self.owned_heap_values.contains_key(symbol_id) {
                    self.owned_heap_values.insert(*symbol_id, phi_reg);
                }
            }
        }
        // debug!("Created {} phi nodes", phi_nodes.len());

        // Create phi nodes in update_block for all loop variables (if update block exists).
        // These are needed because the update block can be reached from both:
        //   1. Normal body end (variables may have been modified in body)
        //   2. Continue statements (variables at their pre-body-modification values)
        // Without these phis, we'd use the wrong value on one of the paths.
        let mut continue_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        if let Some(upd_block) = update_block {
            // Temporarily switch to the update block to create phi nodes
            // (We'll switch back and lower the body first)
            let saved_block = self.builder.current_block();
            self.builder.switch_to_block(upd_block);
            for (symbol_id, (_, var_type)) in &loop_var_initial_values {
                if let Some(upd_phi_reg) = self.builder.build_phi(upd_block, var_type.clone()) {
                    // Register as a local
                    if let Some(func) = self.builder.current_function_mut() {
                        func.locals.insert(
                            upd_phi_reg,
                            crate::ir::IrLocal {
                                name: format!("update_phi_{}", symbol_id.as_raw()),
                                ty: var_type.clone(),
                                mutable: true,
                                source_location: crate::ir::IrSourceLocation::unknown(),
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                    continue_phi_nodes.insert(*symbol_id, upd_phi_reg);
                }
            }
            // Switch back to entry block
            if let Some(saved) = saved_block {
                self.builder.switch_to_block(saved);
            }
        }

        // Push loop context (exit phi nodes will be added after condition evaluation)
        // We need to evaluate the condition FIRST to know which block we're actually in
        // (short-circuit operators like && create additional blocks)
        // When there's a continue_update block, continue should jump there (not cond_block)
        // so the loop counter increment always executes.
        //
        // For plain while loops (no update block), continue jumps to cond_block.
        // The cond_block has phi nodes for loop variables — continue must add incoming
        // edges to these phis, otherwise cranelift sees missing block arguments → SIGILL.
        let loop_continue_phi_nodes = if update_block.is_some() {
            continue_phi_nodes.clone()
        } else {
            // Plain while: continue targets cond_block, so use cond_block's phi nodes
            phi_nodes.clone()
        };
        self.loop_stack.push(LoopContext {
            continue_block: update_block.unwrap_or(cond_block),
            break_block: exit_block,
            label: label.cloned(),
            exit_phi_nodes: BTreeMap::new(), // Will be populated after condition eval
            continue_phi_nodes: loop_continue_phi_nodes,
        });

        // Evaluate condition - this may create additional blocks for short-circuit operators!
        // After this, we may be in a different block than cond_block
        debug!(
            "[lower_while_loop] Lowering condition expression, kind={:?}",
            std::mem::discriminant(&condition.kind)
        );
        let cond_result = self.lower_expression(condition);
        debug!("[lower_while_loop] Condition result: {:?}", cond_result);
        if cond_result.is_none() {
            debug!(
                "[lower_while_loop] DETAILED: condition.kind = {:?}",
                condition.kind
            );
        }

        // Capture the block we're actually in AFTER condition evaluation
        // This is where the conditional branch to body/exit will happen
        let cond_end_block = self.builder.current_block().unwrap_or(cond_block);

        // Now create exit block phi nodes with the correct predecessor block
        let mut exit_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, loop_phi_reg) in &phi_nodes {
            if let Some((_, var_type)) = loop_var_initial_values.get(symbol_id) {
                // Allocate a new register for the exit block parameter
                let exit_param_reg = self.builder.alloc_reg().unwrap();

                // Create the phi node in the exit block with incoming edge from the ACTUAL
                // block that branches to exit (cond_end_block, not necessarily cond_block)
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(exit_block_data) = func.cfg.get_block_mut(exit_block) {
                        let exit_phi = crate::ir::IrPhiNode {
                            dest: exit_param_reg,
                            incoming: vec![(cond_end_block, *loop_phi_reg)],
                            ty: var_type.clone(),
                        };
                        exit_block_data.add_phi(exit_phi);

                        // Register as a local
                        func.locals.insert(
                            exit_param_reg,
                            crate::ir::IrLocal {
                                name: format!("loop_exit_{}", symbol_id.as_raw()),
                                ty: var_type.clone(),
                                mutable: false,
                                source_location: crate::ir::IrSourceLocation::unknown(),
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }

                exit_phi_nodes.insert(*symbol_id, exit_param_reg);
            }
        }

        // Update loop context with the exit phi nodes
        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.exit_phi_nodes = exit_phi_nodes.clone();
        }

        // Build conditional branch from the block we're actually in
        if let Some(cond_reg) = cond_result {
            debug!(
                "[lower_while_loop] Building cond_branch with cond_reg={:?}",
                cond_reg
            );
            self.builder
                .build_cond_branch(cond_reg, body_block, exit_block);
        } else {
            warn!("[lower_while_loop] CONDITION FAILED TO LOWER - no branch built!");
        }

        // Body block
        debug!(
            "[lower_while_loop] Switching to body_block and lowering body ({} statements)",
            body.statements.len()
        );
        self.builder.switch_to_block(body_block);
        // Track loop-carried symbols so the body's exit_drop_scope does not free
        // values that escape via the exit phi (see lower_for_in_over_array).
        self.loop_carried_symbols
            .push(phi_nodes.keys().copied().collect());
        self.enter_drop_scope(); // Enter scope for loop body allocations
        self.lower_block(body);
        debug!("[lower_while_loop] Body lowered");

        // Get the end block of the loop body (might be different if there are nested blocks)
        let body_end_block = if let Some(block_id) = self.builder.current_block() {
            debug!("[lower_while_loop] body_end_block={:?}", block_id);
            block_id
        } else {
            warn!("[lower_while_loop] NO CURRENT BLOCK after body lowering - early return!");
            self.loop_carried_symbols.pop();
            return;
        };

        if update_block.is_some() {
            // When there's an update block, add incoming edges to the UPDATE block's
            // phis from the body end. The update block will then provide the back-edge
            // to cond_block.
            for (symbol_id, upd_phi_reg) in &continue_phi_nodes {
                let body_end_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                    updated_reg
                } else if let Some(&cond_phi) = phi_nodes.get(symbol_id) {
                    cond_phi
                } else {
                    continue;
                };
                self.builder.add_phi_incoming(
                    update_block.unwrap(),
                    *upd_phi_reg,
                    body_end_block,
                    body_end_value,
                );
            }
        } else {
            // Plain while loop: add back-edge directly from body to cond_block.
            for (symbol_id, phi_reg) in &phi_nodes {
                let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                    updated_reg
                } else {
                    *phi_reg
                };
                self.builder.add_phi_incoming(
                    cond_block,
                    *phi_reg,
                    body_end_block,
                    back_edge_value,
                );
            }
        }

        // Branch to update block (or directly to cond_block) if body didn't terminate
        if !self.is_terminated() {
            self.exit_drop_scope(); // Free loop body allocations before next iteration
            if let Some(upd_block) = update_block {
                self.builder.build_branch(upd_block);
            } else {
                self.builder.build_branch(cond_block);
            }
        }
        self.loop_carried_symbols.pop();

        // Lower the update block (e.g., i++ for range loops)
        if let (Some(upd_block), Some(upd_body)) = (update_block, continue_update) {
            self.builder.switch_to_block(upd_block);

            // Point symbol_map to the update block's phi nodes so the update code
            // uses the merged values (from both body-end and continue paths).
            for (symbol_id, upd_phi_reg) in &continue_phi_nodes {
                self.symbol_map.insert(*symbol_id, *upd_phi_reg);
            }

            self.lower_block(upd_body);

            // Get the block we're in after lowering the update
            let update_end_block = self.builder.current_block().unwrap_or(upd_block);

            // Add phi incoming edges from the update block end to cond_block
            for (symbol_id, phi_reg) in &phi_nodes {
                let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                    updated_reg
                } else {
                    *phi_reg
                };
                self.builder.add_phi_incoming(
                    cond_block,
                    *phi_reg,
                    update_end_block,
                    back_edge_value,
                );
            }

            if !self.is_terminated() {
                self.builder.build_branch(cond_block);
            }
        }

        // Pop loop context
        self.loop_stack.pop();

        // Continue in exit block
        // The exit block will receive loop-carried values as block parameters when
        // the conditional branch from the loop header takes the false path
        self.builder.switch_to_block(exit_block);

        // Update symbol map to use exit phi nodes (already created before loop body)
        for (symbol_id, exit_param_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_param_reg);
        }

        // CRITICAL: Also update owned_heap_values to use exit phi values.
        // Without this, owned_heap_values retains IrIds from the loop body block
        // that don't dominate post-loop blocks. When an outer scope's exit_drop_scope
        // later emits Free instructions using these stale IrIds, it creates an SSA
        // dominance violation — at runtime, this causes double-free crashes because
        // the stale register may hold an already-freed pointer.
        for (symbol_id, exit_param_reg) in &exit_phi_nodes {
            if self.owned_heap_values.contains_key(symbol_id) {
                self.owned_heap_values.insert(*symbol_id, *exit_param_reg);
            }
        }
    }

    pub(crate) fn lower_do_while_loop(
        &mut self,
        body: &HirBlock,
        condition: &HirExpr,
        label: Option<&SymbolId>,
    ) {
        // Do-while loop structure:
        // do {
        //     body;
        // } while (condition);
        //
        // MIR structure with phi nodes:
        // entry_block:
        //     goto body_block(initial_values)
        // body_block(phi_nodes):
        //     <body statements>
        //     goto cond_block
        // cond_block:
        //     %cond = <evaluate condition>
        //     br %cond, body_block(updated_values), exit_block(final_values)
        // exit_block(exit_phi_nodes):
        //     <continue>

        // Create blocks
        let Some(body_block) = self.builder.create_block() else {
            return;
        };
        let Some(cond_block) = self.builder.create_block() else {
            return;
        };
        let Some(exit_block) = self.builder.create_block() else {
            return;
        };

        // Save the entry block (current block before loop)
        let entry_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            return;
        };

        // Find all variables that are referenced in the loop body and condition
        let mut referenced_vars = std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_block(body, &mut referenced_vars);
        self.collect_referenced_variables_in_expr(condition, &mut referenced_vars);

        // Only include variables that were declared before the loop
        // (i.e., they're already in the symbol_map)
        // Exclude function parameters since they're immutable
        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| {
                let in_map = self.symbol_map.contains_key(sym);
                // Check if this is a function parameter by seeing if it's in the current function's parameters
                // Exclude only ACTUAL parameter symbols, by kind not register
                // (see is_parameter_symbol).
                let is_param = self.is_parameter_symbol(sym);
                in_map && !is_param
            })
            .collect();

        // Save initial values of loop variables before jumping to body
        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let local_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg))
                    .map(|local| local.ty.clone());
                if let Some(local_ty) = local_ty {
                    loop_var_initial_values
                        .insert(*symbol_id, (reg, self.loop_phi_type(reg, local_ty)));
                }
            }
        }

        // Jump to body first (do-while always executes once)
        self.builder.build_branch(body_block);

        // Body block - create phi nodes for loop variables
        self.builder.switch_to_block(body_block);

        // Create phi nodes for all loop variables at the start of body block
        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            if let Some(phi_reg) = self.builder.build_phi(body_block, var_type.clone()) {
                // Add incoming value from entry block (first iteration)
                self.builder
                    .add_phi_incoming(body_block, phi_reg, entry_block, *initial_reg);

                // Register the phi node as a local so Cranelift can find its type
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

                // Update symbol map to use phi node
                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);

                // Also update owned_heap_values so drop tracking uses the phi result
                // This ensures that when a loop variable is reassigned, we free the
                // current iteration's value (from phi), not the initial value
                if self.owned_heap_values.contains_key(symbol_id) {
                    self.owned_heap_values.insert(*symbol_id, phi_reg);
                }
            }
        }

        // Push loop context with empty exit_phi_nodes (will be populated later)
        self.loop_stack.push(LoopContext {
            continue_block: cond_block,
            break_block: exit_block,
            label: label.cloned(),
            exit_phi_nodes: BTreeMap::new(),
            continue_phi_nodes: BTreeMap::new(),
        });

        // Lower the body statements. Track loop-carried symbols so the body's
        // exit_drop_scope does not free values that escape via the exit phi
        // (see lower_for_in_over_array for the use-after-free rationale).
        self.loop_carried_symbols
            .push(phi_nodes.keys().copied().collect());
        self.enter_drop_scope(); // Enter scope for loop body allocations
        self.lower_block(body);

        // Get the block we're in after the body (might be different if there are nested blocks)
        let body_end_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            self.loop_carried_symbols.pop();
            self.loop_stack.pop();
            return;
        };

        // Branch to condition block if not already terminated
        if !self.is_terminated() {
            self.exit_drop_scope(); // Free loop body allocations before condition check
            self.builder.build_branch(cond_block);
        }
        self.loop_carried_symbols.pop();

        // Build condition block
        self.builder.switch_to_block(cond_block);
        let cond_result = self.lower_expression(condition);

        // Capture the block we're actually in AFTER condition evaluation
        // This is where the conditional branch to body/exit will happen
        let cond_end_block = self.builder.current_block().unwrap_or(cond_block);

        // Now create exit block phi nodes with the correct predecessor block
        let mut exit_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, _phi_reg) in &phi_nodes {
            if let Some((_, var_type)) = loop_var_initial_values.get(symbol_id) {
                // Get the current value of the variable after the loop body
                let current_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                    updated_reg
                } else {
                    continue;
                };

                // Allocate a new register for the exit block parameter
                let exit_param_reg = self.builder.alloc_reg().unwrap();

                // Create the phi node in the exit block with incoming edge from cond_end_block
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(exit_block_data) = func.cfg.get_block_mut(exit_block) {
                        let exit_phi = crate::ir::IrPhiNode {
                            dest: exit_param_reg,
                            incoming: vec![(cond_end_block, current_value)],
                            ty: var_type.clone(),
                        };
                        exit_block_data.add_phi(exit_phi);

                        // Register as a local
                        func.locals.insert(
                            exit_param_reg,
                            crate::ir::IrLocal {
                                name: format!("loop_exit_{}", symbol_id.as_raw()),
                                ty: var_type.clone(),
                                mutable: false,
                                source_location: crate::ir::IrSourceLocation::unknown(),
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }

                exit_phi_nodes.insert(*symbol_id, exit_param_reg);
            }
        }

        // Update loop context with the exit phi nodes
        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.exit_phi_nodes = exit_phi_nodes.clone();
        }

        // Add back-edge phi incoming values from body end to body block
        // These represent the updated values for the next iteration
        for (symbol_id, phi_reg) in &phi_nodes {
            // Get the current value of the variable after the loop body
            let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                updated_reg
            } else {
                *phi_reg
            };

            // Add incoming value from cond block (back edge for next iteration)
            // The back edge comes from cond_end_block (after condition is evaluated)
            self.builder
                .add_phi_incoming(body_block, *phi_reg, cond_end_block, back_edge_value);
        }

        // Build conditional branch from the block we're actually in
        if let Some(cond_reg) = cond_result {
            self.builder
                .build_cond_branch(cond_reg, body_block, exit_block);
        }

        // Pop loop context
        self.loop_stack.pop();

        // Continue at exit block
        self.builder.switch_to_block(exit_block);

        // Update symbol map to use exit phi nodes after the loop
        for (symbol_id, exit_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_reg);
        }

        // CRITICAL: Also update owned_heap_values to use exit phi values.
        // Same fix as in lower_while_loop — prevents SSA dominance violations
        // and double-free crashes when outer scopes free loop-carried heap values.
        for (symbol_id, exit_reg) in &exit_phi_nodes {
            if self.owned_heap_values.contains_key(symbol_id) {
                self.owned_heap_values.insert(*symbol_id, *exit_reg);
            }
        }
    }
}
