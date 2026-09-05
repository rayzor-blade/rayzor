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

        // Heuristic for loop variables: anything referenced in the condition,
        // the body or the continue update.
        let mut referenced_vars = std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_expr(condition, &mut referenced_vars);

        self.collect_referenced_variables_in_block(body, &mut referenced_vars);

        if let Some(upd) = continue_update {
            self.collect_referenced_variables_in_block(upd, &mut referenced_vars);
        }

        // Which of the referenced variables the body actually ASSIGNS. Used
        // below to decide whether a symbol with no `locals` entry — which is
        // what a function parameter looks like here — is loop-carried.
        let mut assigned_in_body = self.find_modified_variables_in_block(body);
        if let Some(update) = continue_update {
            assigned_in_body.extend(self.find_modified_variables_in_block(update));
        }

        // Anything already in symbol_map is a candidate. There is deliberately
        // no parameter exclusion: Haxe parameters are assignable, and the
        // `SymbolKind::Parameter` this used to test for is never constructed
        // (`create_variable_with_type` gives every binding `SymbolKind::Variable`),
        // so the old filter term was always false and the comment claiming
        // "function parameters are immutable" described neither Haxe nor this
        // code.
        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| self.symbol_map.contains_key(sym))
            .collect();

        // Save initial values of loop variables before jumping to condition
        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let local_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg))
                    .map(|local| local.ty.clone())
                    // A register that has a symbol_map entry but no `locals`
                    // entry is a function parameter: parameters are seeded into
                    // `register_types` and bound into `symbol_map`, and never
                    // registered as locals. One the body ASSIGNS is loop-carried
                    // like any local and needs the phi triple; without it the
                    // post-loop repoint never happens and a later read sees the
                    // register the body left behind — defined in a block that
                    // does not dominate the read, which is not a value the
                    // backend can find (`return head` after `head = n` became a
                    // trap stub). One that stays read-only keeps its signature
                    // register, which already dominates every block.
                    .or_else(|| {
                        if assigned_in_body.contains(symbol_id) {
                            self.builder.get_register_type(reg)
                        } else {
                            None
                        }
                    });
                if let Some(local_ty) = local_ty {
                    loop_var_initial_values
                        .insert(*symbol_id, (reg, self.loop_phi_type(reg, local_ty)));
                }
            }
        }

        self.builder.build_branch(cond_block);

        self.builder.switch_to_block(cond_block);

        // Create phi nodes for all loop variables
        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            if let Some(phi_reg) = self.builder.build_phi(cond_block, var_type.clone()) {
                self.builder
                    .add_phi_incoming(cond_block, phi_reg, entry_block, *initial_reg);

                // Register the phi node as a local so Cranelift can find its type.
                // Unconditional: the initial register has no `locals` entry when
                // it is a parameter, and the update and exit phis register theirs
                // unconditionally already — leaving only the header phi
                // unregistered gives the backend a phi triple it can type in two
                // places out of three.
                if let Some(func) = self.builder.current_function_mut() {
                    let prior = func.locals.get(initial_reg).cloned();
                    let (name, source_location) = match &prior {
                        Some(local) => (format!("{}_phi", local.name), local.source_location),
                        None => (
                            format!("loop_phi_{}", symbol_id.as_raw()),
                            crate::ir::IrSourceLocation::unknown(),
                        ),
                    };
                    func.locals.insert(
                        phi_reg,
                        crate::ir::IrLocal {
                            name,
                            ty: var_type.clone(),
                            mutable: true,
                            source_location,
                            allocation: crate::ir::AllocationHint::Register,
                        },
                    );
                }

                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);

                // Move drop tracking onto the phi so a reassigned loop variable
                // frees the current iteration's value, not the initial one.
                if self.owned_heap_values.contains_key(symbol_id) {
                    self.owned_heap_values.insert(*symbol_id, phi_reg);
                }
            }
        }

        // Create phi nodes in update_block for all loop variables (if update block exists).
        // These are needed because the update block can be reached from both:
        //   1. Normal body end (variables may have been modified in body)
        //   2. Continue statements (variables at their pre-body-modification values)
        // Without these phis, we'd use the wrong value on one of the paths.
        let mut continue_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        if let Some(upd_block) = update_block {
            // Switch to the update block only to create the phis; the body is
            // lowered first.
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
            if let Some(saved) = saved_block {
                self.builder.switch_to_block(saved);
            }
        }

        // Exit phi nodes are added after condition evaluation: the condition must
        // be lowered first to know which block we end up in (short-circuit
        // operators like && create additional blocks). With a continue_update
        // block, continue jumps there rather than to cond_block so the loop
        // counter increment always executes. For plain while loops continue
        // targets cond_block, and must add incoming edges to its phi nodes or
        // cranelift sees missing block arguments.
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
            carried_slots: BTreeMap::new(),
        });

        // Short-circuit operators create extra blocks, so this may leave us in a
        // different block than cond_block.
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

        // The block we are in after condition evaluation is the one that branches
        // to body/exit.
        let cond_end_block = self.builder.current_block().unwrap_or(cond_block);

        // Now create exit block phi nodes with the correct predecessor block
        let mut exit_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, loop_phi_reg) in &phi_nodes {
            if let Some((_, var_type)) = loop_var_initial_values.get(symbol_id) {
                let exit_param_reg = self.builder.alloc_reg().unwrap();

                // The incoming edge comes from the block that actually branches to
                // exit (cond_end_block, not necessarily cond_block).
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

        self.loop_stack.pop();

        // The exit block receives loop-carried values as block parameters when the
        // conditional branch from the loop header takes the false path.
        self.builder.switch_to_block(exit_block);

        for (symbol_id, exit_param_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_param_reg);
        }

        // owned_heap_values must move onto the exit phis as well: IrIds from the
        // loop body block do not dominate post-loop blocks, so Free instructions
        // emitted by an outer exit_drop_scope would violate SSA dominance and can
        // free an already-freed pointer.
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

        // See lower_while_loop: no parameter exclusion, and the body's own
        // assignments decide what is loop-carried.
        let assigned_in_body = self.find_modified_variables_in_block(body);

        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| self.symbol_map.contains_key(sym))
            .collect();

        // Save initial values of loop variables before jumping to body
        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let local_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg))
                    .map(|local| local.ty.clone())
                    // A register that has a symbol_map entry but no `locals`
                    // entry is a function parameter: parameters are seeded into
                    // `register_types` and bound into `symbol_map`, and never
                    // registered as locals. One the body ASSIGNS is loop-carried
                    // like any local and needs the phi triple; without it the
                    // post-loop repoint never happens and a later read sees the
                    // register the body left behind — defined in a block that
                    // does not dominate the read, which is not a value the
                    // backend can find (`return head` after `head = n` became a
                    // trap stub). One that stays read-only keeps its signature
                    // register, which already dominates every block.
                    .or_else(|| {
                        if assigned_in_body.contains(symbol_id) {
                            self.builder.get_register_type(reg)
                        } else {
                            None
                        }
                    });
                if let Some(local_ty) = local_ty {
                    loop_var_initial_values
                        .insert(*symbol_id, (reg, self.loop_phi_type(reg, local_ty)));
                }
            }
        }

        // Jump to body first (do-while always executes once)
        self.builder.build_branch(body_block);

        self.builder.switch_to_block(body_block);

        // Create phi nodes for all loop variables at the start of body block
        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            if let Some(phi_reg) = self.builder.build_phi(body_block, var_type.clone()) {
                // Add incoming value from entry block (first iteration)
                self.builder
                    .add_phi_incoming(body_block, phi_reg, entry_block, *initial_reg);

                // Register the phi node as a local so Cranelift can find its type.
                // Unconditional: the initial register has no `locals` entry when
                // it is a parameter, and the update and exit phis register theirs
                // unconditionally already — leaving only the header phi
                // unregistered gives the backend a phi triple it can type in two
                // places out of three.
                if let Some(func) = self.builder.current_function_mut() {
                    let prior = func.locals.get(initial_reg).cloned();
                    let (name, source_location) = match &prior {
                        Some(local) => (format!("{}_phi", local.name), local.source_location),
                        None => (
                            format!("loop_phi_{}", symbol_id.as_raw()),
                            crate::ir::IrSourceLocation::unknown(),
                        ),
                    };
                    func.locals.insert(
                        phi_reg,
                        crate::ir::IrLocal {
                            name,
                            ty: var_type.clone(),
                            mutable: true,
                            source_location,
                            allocation: crate::ir::AllocationHint::Register,
                        },
                    );
                }

                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);

                // Move drop tracking onto the phi so a reassigned loop variable
                // frees the current iteration's value, not the initial one.
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
            carried_slots: BTreeMap::new(),
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

        self.builder.switch_to_block(cond_block);
        let cond_result = self.lower_expression(condition);

        // The block we are in after condition evaluation is the one that branches
        // to body/exit.
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

                let exit_param_reg = self.builder.alloc_reg().unwrap();

                // The incoming edge comes from cond_end_block.
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

        if let Some(loop_ctx) = self.loop_stack.last_mut() {
            loop_ctx.exit_phi_nodes = exit_phi_nodes.clone();
        }

        // Back-edge phi incoming values: the updated values for the next iteration,
        // reaching body_block from cond_end_block.
        for (symbol_id, phi_reg) in &phi_nodes {
            let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                updated_reg
            } else {
                *phi_reg
            };

            self.builder
                .add_phi_incoming(body_block, *phi_reg, cond_end_block, back_edge_value);
        }

        // Build conditional branch from the block we're actually in
        if let Some(cond_reg) = cond_result {
            self.builder
                .build_cond_branch(cond_reg, body_block, exit_block);
        }

        self.loop_stack.pop();

        self.builder.switch_to_block(exit_block);

        for (symbol_id, exit_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_reg);
        }

        // owned_heap_values must move onto the exit phis as well (see
        // lower_while_loop): body-block IrIds do not dominate post-loop blocks.
        for (symbol_id, exit_reg) in &exit_phi_nodes {
            if self.owned_heap_values.contains_key(symbol_id) {
                self.owned_heap_values.insert(*symbol_id, *exit_reg);
            }
        }
    }
}
