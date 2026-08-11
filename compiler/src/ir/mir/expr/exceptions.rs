//! `try`/`catch`: landing pads and binding the caught value.

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
    pub(crate) fn lower_try_catch(
        &mut self,
        try_block: &HirBlock,
        catches: &[HirCatchClause],
        finally: Option<&HirBlock>,
    ) {
        // Exception handling via setjmp/longjmp:
        //
        //   try_entry (current block):
        //     %jmp_buf = call rayzor_exception_push_handler()  // returns *mut u8
        //     %result = call _setjmp(%jmp_buf)                 // returns 0 normally, 1 on longjmp
        //     br %result == 0, normal_path, landing_pad
        //   normal_path:
        //     <try block>
        //     call rayzor_exception_pop_handler()
        //     br finally / continuation
        //   landing_pad:
        //     call rayzor_exception_pop_handler()
        //     %exc = call rayzor_get_exception()
        //     br catch_0
        //   catch_0:
        //     <catch block>
        //     br finally / continuation
        //   finally_block:
        //     <finally code>
        //     br continuation
        //   continuation:
        //     <rest of code>

        // Snapshot the pre-try value of every variable the try or catch bodies
        // reference and that already exists (not declared inside a branch, not a
        // parameter), then build phis at the convergence point as
        // `lower_if_statement` does. Without the merge the continuation keeps
        // whichever branch was lowered last — the catch.
        let mut tc_tracked: std::collections::BTreeSet<SymbolId> =
            std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_block(try_block, &mut tc_tracked);
        for c in catches {
            self.collect_referenced_variables_in_block(&c.body, &mut tc_tracked);
        }
        let mut tc_pre: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for s in &tc_tracked {
            if self.is_parameter_symbol(s) {
                continue;
            }
            if let Some(&reg) = self.symbol_map.get(s) {
                let ty = self
                    .builder
                    .current_function()
                    .and_then(|f| f.locals.get(&reg).map(|l| l.ty.clone()));
                if let Some(ty) = ty {
                    tc_pre.insert(*s, (reg, ty));
                }
            }
        }
        // Captured (exit_block, {var -> value}) for each non-terminated path.
        let mut tc_try_exit: Option<(IrBlockId, BTreeMap<SymbolId, IrId>)> = None;
        let mut tc_catch_exits: Vec<(IrBlockId, BTreeMap<SymbolId, IrId>)> = Vec::new();
        let mut tc_fallthrough: Option<IrBlockId> = None;

        let normal_path_block = match self.builder.create_block() {
            Some(b) => b,
            None => return,
        };

        let landing_pad_block = match self.builder.create_block() {
            Some(b) => b,
            None => return,
        };

        let finally_block = if finally.is_some() {
            self.builder.create_block()
        } else {
            None
        };

        let continuation_block = match self.builder.create_block() {
            Some(b) => b,
            None => return,
        };

        // --- try_entry block (current block) ---
        let push_fn = self.get_or_register_extern_function(
            "rayzor_exception_push_handler",
            vec![],
            IrType::Ptr(Box::new(IrType::Void)),
        );
        let jmp_buf =
            self.builder
                .build_call_direct(push_fn, vec![], IrType::Ptr(Box::new(IrType::Void)));

        let setjmp_fn = self.get_or_register_extern_function(
            "_setjmp",
            vec![IrType::Ptr(Box::new(IrType::Void))],
            IrType::I32,
        );
        let jmp_buf_reg = jmp_buf.unwrap_or_else(|| {
            self.builder
                .build_const(IrValue::I64(0))
                .expect("failed to create const")
        });
        let setjmp_result =
            self.builder
                .build_call_direct(setjmp_fn, vec![jmp_buf_reg], IrType::I32);

        let zero = self
            .builder
            .build_const(IrValue::I32(0))
            .expect("failed to create const");
        let setjmp_reg = setjmp_result.unwrap_or(zero);
        let cmp = self
            .builder
            .build_cmp(CompareOp::Eq, setjmp_reg, zero)
            .expect("failed to build cmp");
        self.builder
            .build_cond_branch(cmp, normal_path_block, landing_pad_block);

        // --- normal_path block: execute try body ---
        self.builder.switch_to_block(normal_path_block);
        self.lower_block(try_block);

        let pop_fn = self.get_or_register_extern_function(
            "rayzor_exception_pop_handler",
            vec![],
            IrType::Void,
        );
        self.builder.build_call_direct(pop_fn, vec![], IrType::Void);

        // Capture the try-path's exit values before it leaves for the merge.
        if !self.is_terminated() {
            if let Some(blk) = self.builder.current_block() {
                let mut vals: BTreeMap<SymbolId, IrId> = BTreeMap::new();
                for s in tc_pre.keys() {
                    if let Some(&reg) = self.symbol_map.get(s) {
                        vals.insert(*s, reg);
                    }
                }
                tc_try_exit = Some((blk, vals));
            }
        }

        if let Some(fb) = finally_block {
            self.builder.build_branch(fb);
        } else {
            self.builder.build_branch(continuation_block);
        }

        // The catch bodies run INSTEAD of the try (on exception), so they must
        // see the PRE-try values, not the try's modifications. Reset now.
        for (s, (reg, _)) in &tc_pre {
            self.symbol_map.insert(*s, *reg);
        }

        // --- landing_pad block: exception was thrown ---
        self.builder.switch_to_block(landing_pad_block);

        // The longjmp already fired; the handler still has to leave the stack.
        let pop_fn2 = self.get_or_register_extern_function(
            "rayzor_exception_pop_handler",
            vec![],
            IrType::Void,
        );
        self.builder
            .build_call_direct(pop_fn2, vec![], IrType::Void);

        let get_exc_fn =
            self.get_or_register_extern_function("rayzor_get_exception", vec![], IrType::I64);
        let exception_id = self
            .builder
            .build_call_direct(get_exc_fn, vec![], IrType::I64)
            .unwrap_or_else(|| {
                self.builder
                    .build_const(IrValue::I64(0))
                    .expect("failed to create const")
            });

        let get_exc_type_fn = self.get_or_register_extern_function(
            "rayzor_get_exception_type_id",
            vec![],
            IrType::I32,
        );
        let exc_type_id = self
            .builder
            .build_call_direct(get_exc_type_fn, vec![], IrType::I32)
            .unwrap_or_else(|| {
                self.builder
                    .build_const(IrValue::I32(0))
                    .expect("failed to create const")
            });

        // Type-based dispatch across catch clauses
        if !catches.is_empty() {
            let after_catch_target = if let Some(fb) = finally_block {
                fb
            } else {
                continuation_block
            };

            // Build chain: for each catch, test type match → body or next catch
            let mut next_test_block: Option<IrBlockId> = None;

            for (i, catch_clause) in catches.iter().enumerate() {
                let catch_type_kind = {
                    let type_table = self.type_table;
                    type_table
                        .get(catch_clause.exception_type)
                        .map(|t| t.kind.clone())
                };
                let is_dynamic = matches!(catch_type_kind, Some(crate::tast::TypeKind::Dynamic));

                let catch_body_block = match self.builder.create_block() {
                    Some(b) => b,
                    None => return,
                };

                // If this is not the first catch, the previous test's false branch leads here
                if let Some(test_block) = next_test_block {
                    self.builder.switch_to_block(test_block);
                }
                // else: we're still in the landing_pad block from above

                if is_dynamic || i == catches.len() - 1 {
                    // Dynamic catches everything; last catch is also unconditional fallback
                    self.builder.build_branch(catch_body_block);
                    next_test_block = None;
                } else {
                    let expected_type_id = self.runtime_type_id(catch_clause.exception_type);
                    let expected_const = self
                        .builder
                        .build_const(IrValue::I32(expected_type_id as i32))
                        .expect("failed to create type_id const");

                    // Class types match polymorphically (walks inheritance);
                    // primitives (Int, String, …) match exactly.
                    let is_class_type =
                        matches!(catch_type_kind, Some(crate::tast::TypeKind::Class { .. }));

                    let type_match = if is_class_type {
                        let match_fn = self.get_or_register_extern_function(
                            "rayzor_exception_type_matches",
                            vec![IrType::I32, IrType::I32],
                            IrType::I32,
                        );
                        let result = self
                            .builder
                            .build_call_direct(
                                match_fn,
                                vec![exc_type_id, expected_const],
                                IrType::I32,
                            )
                            .unwrap_or_else(|| {
                                self.builder
                                    .build_const(IrValue::I32(0))
                                    .expect("failed to create const")
                            });
                        let zero_val = self
                            .builder
                            .build_const(IrValue::I32(0))
                            .expect("failed to create const");
                        self.builder
                            .build_cmp(CompareOp::Ne, result, zero_val)
                            .expect("failed to build type cmp")
                    } else {
                        self.builder
                            .build_cmp(CompareOp::Eq, exc_type_id, expected_const)
                            .expect("failed to build type cmp")
                    };

                    let next_block = self.builder.create_block().expect("failed to create block");
                    self.builder
                        .build_cond_branch(type_match, catch_body_block, next_block);
                    next_test_block = Some(next_block);
                }

                // --- catch body ---
                self.builder.switch_to_block(catch_body_block);
                // Catches are mutually exclusive; reset tracked vars to pre-try
                // so catch N does not inherit catch N-1's writes.
                for (s, (reg, _)) in &tc_pre {
                    self.symbol_map.insert(*s, *reg);
                }
                self.symbol_map
                    .insert(catch_clause.exception_var, exception_id);
                self.lower_block(&catch_clause.body);
                if !self.is_terminated() {
                    if let Some(blk) = self.builder.current_block() {
                        let mut vals: BTreeMap<SymbolId, IrId> = BTreeMap::new();
                        for s in tc_pre.keys() {
                            if let Some(&reg) = self.symbol_map.get(s) {
                                vals.insert(*s, reg);
                            }
                        }
                        tc_catch_exits.push((blk, vals));
                    }
                }
                self.builder.build_branch(after_catch_target);
            }

            // If all typed catches failed and no Dynamic/final catch consumed it,
            // branch to finally/continuation (exception goes unhandled at this level)
            if let Some(fallthrough_block) = next_test_block {
                self.builder.switch_to_block(fallthrough_block);
                tc_fallthrough = Some(fallthrough_block);
                self.builder.build_branch(after_catch_target);
            }
        } else {
            // No catch clauses — go straight to finally or continuation.
            if let Some(fb) = finally_block {
                self.builder.build_branch(fb);
            } else {
                self.builder.build_branch(continuation_block);
            }
        }

        // --- Merge: build phis at the convergence point (the finally entry if
        // there is a finally, else the continuation) so each tracked variable
        // gets the value from the path actually taken. All of try-exit /
        // catch-exits / fallthrough branch to exactly this block.
        let merge_target = finally_block.unwrap_or(continuation_block);
        self.builder.switch_to_block(merge_target);
        for (s, (pre_reg, ty)) in &tc_pre {
            let mut incomings: Vec<(IrBlockId, IrId)> = Vec::new();
            if let Some((blk, vals)) = &tc_try_exit {
                incomings.push((*blk, vals.get(s).copied().unwrap_or(*pre_reg)));
            }
            for (blk, vals) in &tc_catch_exits {
                incomings.push((*blk, vals.get(s).copied().unwrap_or(*pre_reg)));
            }
            if let Some(blk) = tc_fallthrough {
                incomings.push((blk, *pre_reg));
            }
            if incomings.is_empty() {
                continue;
            }
            let first = incomings[0].1;
            if incomings.iter().all(|(_, v)| *v == first) {
                // Every path carries the same value — no phi needed.
                self.symbol_map.insert(*s, first);
                continue;
            }
            if let Some(phi_reg) = self.builder.build_phi(merge_target, ty.clone()) {
                for (blk, val) in &incomings {
                    self.builder
                        .add_phi_incoming(merge_target, phi_reg, *blk, *val);
                }
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(local) = func.locals.get(pre_reg).cloned() {
                        func.locals.insert(
                            phi_reg,
                            crate::ir::IrLocal {
                                name: format!("{}_tcphi", local.name),
                                ty: ty.clone(),
                                mutable: true,
                                source_location: local.source_location,
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }
                self.symbol_map.insert(*s, phi_reg);
            }
        }

        // --- finally block (if present) ---
        if let Some(finally_body) = finally {
            if let Some(fb) = finally_block {
                self.builder.switch_to_block(fb);
                self.lower_block(finally_body);
                self.builder.build_branch(continuation_block);
            }
        }

        self.builder.switch_to_block(continuation_block);
    }

    pub(crate) fn lower_try_catch_expr(&mut self, expr: &HirExpr) -> Option<IrId> {
        let HirExprKind::TryCatch {
            try_expr,
            catch_handlers,
            finally_expr,
        } = &expr.kind
        else {
            unreachable!("lower_try_catch_expr on a non-TryCatch expression")
        };
        // Exception handling via setjmp/longjmp (expression form).
        // try { body } catch (e) { handler } finally { cleanup }

        // Snapshot the pre-try value of every variable the try/catch bodies
        // touch, to build merge phis at the continuation. The catch runs instead
        // of the try, so without a merge the continuation keeps whichever branch
        // lowered last and a var the catch modified leaks out. Mirrors
        // lower_if_statement's then/else merge.
        let mut tc_tracked: std::collections::BTreeSet<SymbolId> =
            std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_expr(try_expr, &mut tc_tracked);
        for h in catch_handlers {
            self.collect_referenced_variables_in_expr(&h.body, &mut tc_tracked);
        }
        let mut tc_pre: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for s in &tc_tracked {
            if self.is_parameter_symbol(s) {
                continue;
            }
            if let Some(&reg) = self.symbol_map.get(s) {
                let ty = self
                    .builder
                    .current_function()
                    .and_then(|f| f.locals.get(&reg).map(|l| l.ty.clone()));
                if let Some(ty) = ty {
                    tc_pre.insert(*s, (reg, ty));
                }
            }
        }
        // (exit_block, {var -> value}) captured at every path that reaches
        // the continuation, used to build the merge phis below.
        let mut tc_exits: Vec<(IrBlockId, BTreeMap<SymbolId, IrId>)> = Vec::new();

        let normal_path_block = self.builder.create_block()?;
        let landing_pad_block = self.builder.create_block()?;
        let continuation_block = self.builder.create_block()?;

        // --- Setup: push handler and call _setjmp ---
        let push_fn = self.get_or_register_extern_function(
            "rayzor_exception_push_handler",
            vec![],
            IrType::Ptr(Box::new(IrType::Void)),
        );
        let jmp_buf =
            self.builder
                .build_call_direct(push_fn, vec![], IrType::Ptr(Box::new(IrType::Void)));

        let setjmp_fn = self.get_or_register_extern_function(
            "_setjmp",
            vec![IrType::Ptr(Box::new(IrType::Void))],
            IrType::I32,
        );
        let jmp_buf_reg =
            jmp_buf.unwrap_or_else(|| self.builder.build_const(IrValue::I64(0)).expect("const"));
        let setjmp_result =
            self.builder
                .build_call_direct(setjmp_fn, vec![jmp_buf_reg], IrType::I32);

        let zero = self.builder.build_const(IrValue::I32(0)).expect("const");
        let setjmp_reg = setjmp_result.unwrap_or(zero);
        let cmp = self.builder.build_cmp(CompareOp::Eq, setjmp_reg, zero)?;
        self.builder
            .build_cond_branch(cmp, normal_path_block, landing_pad_block);

        // --- normal_path: execute try body ---
        self.builder.switch_to_block(normal_path_block);
        self.lower_expression(try_expr);

        let pop_fn = self.get_or_register_extern_function(
            "rayzor_exception_pop_handler",
            vec![],
            IrType::Void,
        );
        self.builder.build_call_direct(pop_fn, vec![], IrType::Void);

        if let Some(finally_body) = &finally_expr {
            self.lower_expression(finally_body);
        }

        // Capture the try-path's values before they leave for the merge.
        if !self.is_terminated() {
            if let Some(blk) = self.builder.current_block() {
                tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
            }
        }
        self.builder.build_branch(continuation_block);

        // Catch bodies run INSTEAD of the try, so reset every tracked var
        // to its pre-try value before lowering them.
        for (s, (reg, _)) in &tc_pre {
            self.symbol_map.insert(*s, *reg);
        }

        // --- landing_pad: exception was thrown ---
        self.builder.switch_to_block(landing_pad_block);

        let pop_fn2 = self.get_or_register_extern_function(
            "rayzor_exception_pop_handler",
            vec![],
            IrType::Void,
        );
        self.builder
            .build_call_direct(pop_fn2, vec![], IrType::Void);

        let get_exc_fn =
            self.get_or_register_extern_function("rayzor_get_exception", vec![], IrType::I64);
        let exception_id = self
            .builder
            .build_call_direct(get_exc_fn, vec![], IrType::I64)
            .unwrap_or_else(|| self.builder.build_const(IrValue::I64(0)).expect("const"));

        let get_exc_type_fn = self.get_or_register_extern_function(
            "rayzor_get_exception_type_id",
            vec![],
            IrType::I32,
        );
        let exc_type_id = self
            .builder
            .build_call_direct(get_exc_type_fn, vec![], IrType::I32)
            .unwrap_or_else(|| self.builder.build_const(IrValue::I32(0)).expect("const"));

        // Type-based dispatch across catch handlers
        if !catch_handlers.is_empty() {
            let mut next_test_block: Option<IrBlockId> = None;

            for (i, handler) in catch_handlers.iter().enumerate() {
                let catch_type_kind = {
                    let type_table = self.type_table;
                    type_table
                        .get(handler.exception_type)
                        .map(|t| t.kind.clone())
                };
                let is_dynamic = matches!(catch_type_kind, Some(crate::tast::TypeKind::Dynamic));

                let catch_body_block = match self.builder.create_block() {
                    Some(b) => b,
                    None => return None,
                };

                if let Some(test_block) = next_test_block {
                    self.builder.switch_to_block(test_block);
                }

                if is_dynamic || i == catch_handlers.len() - 1 {
                    self.builder.build_branch(catch_body_block);
                    next_test_block = None;
                } else {
                    let expected_type_id = self.runtime_type_id(handler.exception_type);
                    let expected_const = self
                        .builder
                        .build_const(IrValue::I32(expected_type_id as i32))
                        .expect("const");

                    // Class types match polymorphically (walks inheritance);
                    // primitives match exactly.
                    let is_class_type =
                        matches!(catch_type_kind, Some(crate::tast::TypeKind::Class { .. }));

                    let type_match = if is_class_type {
                        let match_fn = self.get_or_register_extern_function(
                            "rayzor_exception_type_matches",
                            vec![IrType::I32, IrType::I32],
                            IrType::I32,
                        );
                        let result = self
                            .builder
                            .build_call_direct(
                                match_fn,
                                vec![exc_type_id, expected_const],
                                IrType::I32,
                            )
                            .unwrap_or_else(|| {
                                self.builder.build_const(IrValue::I32(0)).expect("const")
                            });
                        let zero_val = self.builder.build_const(IrValue::I32(0)).expect("const");
                        self.builder
                            .build_cmp(CompareOp::Ne, result, zero_val)
                            .expect("cmp")
                    } else {
                        self.builder
                            .build_cmp(CompareOp::Eq, exc_type_id, expected_const)
                            .expect("cmp")
                    };

                    let next_block = self.builder.create_block().expect("create block");
                    self.builder
                        .build_cond_branch(type_match, catch_body_block, next_block);
                    next_test_block = Some(next_block);
                }

                // --- catch body ---
                self.builder.switch_to_block(catch_body_block);
                // Each catch is mutually exclusive and runs in place of
                // the try — reset tracked vars to pre-try values first.
                for (s, (reg, _)) in &tc_pre {
                    self.symbol_map.insert(*s, *reg);
                }
                self.symbol_map.insert(handler.exception_var, exception_id);
                self.lower_expression(&handler.body);

                if let Some(finally_body) = &finally_expr {
                    self.lower_expression(finally_body);
                }
                if !self.is_terminated() {
                    if let Some(blk) = self.builder.current_block() {
                        tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                    }
                }
                self.builder.build_branch(continuation_block);
            }

            // Fallthrough if no catch matched (exception unhandled here):
            // tracked vars keep their pre-try values.
            if let Some(fallthrough_block) = next_test_block {
                self.builder.switch_to_block(fallthrough_block);
                for (s, (reg, _)) in &tc_pre {
                    self.symbol_map.insert(*s, *reg);
                }
                if let Some(finally_body) = &finally_expr {
                    self.lower_expression(finally_body);
                }
                if !self.is_terminated() {
                    if let Some(blk) = self.builder.current_block() {
                        tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                    }
                }
                self.builder.build_branch(continuation_block);
            }
        } else {
            // No catch clauses: the landing pad keeps pre-try values.
            for (s, (reg, _)) in &tc_pre {
                self.symbol_map.insert(*s, *reg);
            }
            if let Some(finally_body) = &finally_expr {
                self.lower_expression(finally_body);
            }
            if !self.is_terminated() {
                if let Some(blk) = self.builder.current_block() {
                    tc_exits.push((blk, self.capture_tracked_values(&tc_pre)));
                }
            }
            self.builder.build_branch(continuation_block);
        }

        // --- continuation: merge the tracked vars across all paths ---
        self.builder.switch_to_block(continuation_block);
        for (s, (pre_reg, ty)) in &tc_pre {
            let mut incomings: Vec<(IrBlockId, IrId)> = Vec::new();
            for (blk, vals) in &tc_exits {
                incomings.push((*blk, vals.get(s).copied().unwrap_or(*pre_reg)));
            }
            if incomings.is_empty() {
                continue;
            }
            let first = incomings[0].1;
            if incomings.iter().all(|(_, v)| *v == first) {
                // Every path carries the same value — no phi needed.
                self.symbol_map.insert(*s, first);
                continue;
            }
            if let Some(phi_reg) = self.builder.build_phi(continuation_block, ty.clone()) {
                for (blk, val) in &incomings {
                    self.builder
                        .add_phi_incoming(continuation_block, phi_reg, *blk, *val);
                }
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(local) = func.locals.get(pre_reg).cloned() {
                        func.locals.insert(
                            phi_reg,
                            crate::ir::IrLocal {
                                name: format!("{}_tcphi", local.name),
                                ty: ty.clone(),
                                mutable: true,
                                source_location: local.source_location,
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }
                self.symbol_map.insert(*s, phi_reg);
            }
        }
        None // try/catch as statement has no return value
    }
}
