//! Closures and method references; captures become an environment struct.

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
    pub(crate) fn lower_lambda(
        &mut self,
        params: &[HirParam],
        body: &HirExpr,
        captures: &[HirCapture],
        lambda_type: TypeId,
    ) -> Option<IrId> {
        // Closure/Lambda lowering using MakeClosure instruction:
        //
        // For: |x, y| { x + y + captured_z }
        //
        // Strategy:
        // 1. Generate a lambda function that takes (env*, params...) where
        //    env* is a struct containing all captured variables
        // 2. Collect the values to be captured (from current scope)
        // 3. Use MakeClosure instruction to create closure at runtime
        //
        // The MakeClosure instruction will:
        // - Allocate an environment struct
        // - Copy captured values into it
        // - Create a closure struct { func_ptr, env_ptr }
        // - Return the closure

        // Step 1: Collect captured values from current scope FIRST
        // (before generate_lambda_function which saves/restores state)
        let mut captured_values = Vec::new();
        // Filter captures to only include actual variables, not global functions
        // Global functions (like `trace`) don't need to be captured - they're resolved by name
        let filtered_captures: Vec<_> = captures
            .iter()
            .filter(|c| {
                if let Some(sym) = self.symbol_table.get_symbol(c.symbol) {
                    // Skip Function symbols - they don't need capturing
                    if sym.kind == crate::tast::SymbolKind::Function {
                        debug!(
                            "Skipping function capture {:?} (name={:?})",
                            c.symbol,
                            self.string_interner.get(sym.name)
                        );
                        return false;
                    }
                }
                true
            })
            .collect();

        debug!(
            "Collecting {} captured values (filtered from {})",
            filtered_captures.len(),
            captures.len()
        );
        debug!("symbol_map has {} entries", self.symbol_map.len());
        for capture in &filtered_captures {
            debug!("Looking for captured symbol {:?}", capture.symbol);
            if let Some(&captured_val) = self.symbol_map.get(&capture.symbol) {
                debug!("  Found! Register: {:?}", captured_val);
                captured_values.push(captured_val);
            } else {
                // Captured variable not found in current scope
                debug!("  NOT FOUND! Available symbols:");
                for (sym, reg) in &self.symbol_map {
                    debug!("    {:?} -> {:?}", sym, reg);
                }
                self.errors.push(LoweringError {
                    message: format!("Captured variable {:?} not found in scope", capture.symbol),
                    location: body.source_location.clone(),
                });
                return None;
            }
        }

        // Step 2: Generate the lambda function
        // Pass filtered captures so the lambda doesn't try to load global functions from env
        let filtered_captures_slice: Vec<HirCapture> =
            filtered_captures.iter().map(|c| (*c).clone()).collect();
        let lambda_func_id =
            self.generate_lambda_function(params, body, &filtered_captures_slice, lambda_type)?;

        // Step 3: Use MakeClosure instruction to create closure
        let result = self
            .builder
            .build_make_closure(lambda_func_id, captured_values);

        // Step 4: Transfer ownership of captured variables to the closure
        // When a variable is captured by a closure, ownership is MOVED into the closure
        // environment. The enclosing scope should NOT free captured variables.
        // This prevents double-free when both the enclosing scope and the closure
        // try to free the same memory.
        for capture in &filtered_captures {
            if self.owned_heap_values.remove(&capture.symbol).is_some() {
                debug!(
                    "Transferred ownership of {:?} to closure (removed from owned_heap_values)",
                    capture.symbol
                );
            }
        }

        result
    }

    /// PASS 2: Lower lambda body and infer signature
    #[allow(dead_code)] // Will be used once we switch to two-pass
    pub(crate) fn lower_lambda_body(
        &mut self,
        context: LambdaContext,
        params: &[HirParam],
        body: &HirExpr,
    ) -> Option<IrFunctionId> {
        let LambdaContext {
            func_id,
            entry_block,
            param_offset,
            env_layout,
        } = context;

        // Save state
        let saved_state = self.save_state();

        // Switch to lambda context
        self.builder.current_function = Some(func_id);
        self.builder.current_block = Some(entry_block);
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        // Per-function isolation: lambda body has its own SSA register namespace;
        // saved_state already snapshotted strict_move_locals for restore on exit.
        self.strict_move_locals.clear();
        // Lambda body lowers its own loops with a fresh loop-carried stack.
        self.loop_carried_symbols.clear();
        self.current_env_layout = env_layout.clone();

        // Clear drop tracking state for the lambda body.
        // Lambda bodies are separate functions with their own register namespace.
        // Without this, cleanup_all_scopes (called on return) would try to free
        // the parent function's owned heap values using IrIds that refer to
        // different registers in the lambda's context, causing heap corruption.
        self.owned_heap_values.clear();
        self.drop_scope_stack.clear();
        self.temp_heap_values.clear();
        self.reassigned_in_scope.clear();
        self.current_drop_points = None;
        self.current_stmt_index = 0;

        // Map lambda parameters to registers AND register as locals
        // (matching regular function param setup at line ~2274-2288)
        for (i, param) in params.iter().enumerate() {
            let param_reg = IrId::new(param_offset + i as u32);
            self.symbol_map.insert(param.symbol_id, param_reg);

            // Also register parameter as a local so type inference can find it
            let param_type = self.convert_type(param.ty);
            let param_name = self
                .string_interner
                .get(param.name)
                .unwrap_or("<param>")
                .to_string();
            if let Some(func) = self.builder.current_function_mut() {
                func.locals.insert(
                    param_reg,
                    crate::ir::IrLocal {
                        name: param_name,
                        ty: param_type,
                        mutable: false,
                        source_location: crate::ir::IrSourceLocation::unknown(),
                        allocation: crate::ir::AllocationHint::Register,
                    },
                );
            }
        }

        // Setup captured variables using environment layout
        if let Some(layout) = &env_layout {
            debug!(
                "Lambda has {} captured variables in environment",
                layout.fields.len()
            );
            for field in &layout.fields {
                debug!("Captured symbol: {:?}", field.symbol);
            }

            let env_ptr = IrId::new(0); // First parameter

            for field in &layout.fields {
                // Use layout to load field (handles casting automatically)
                let value_reg = layout.load_field(&mut self.builder, env_ptr, field.symbol)?;
                self.symbol_map.insert(field.symbol, value_reg);
            }
        }

        // Lower the body expression.
        // For arrow functions, the body is typically a Block wrapping a single expression
        // statement. lower_expression(Block) returns None because blocks don't have return
        // values in general. To get the lambda's body result, we unwrap the block and
        // lower the inner expression directly.
        let body_result = match &body.kind {
            crate::ir::hir::HirExprKind::Block(block) => {
                if block.statements.len() == 1 && block.expr.is_none() {
                    // Single-expression block (common in arrow functions):
                    // Extract the expression from the Expr statement and lower directly
                    if let crate::ir::hir::HirStatement::Expr(expr) = &block.statements[0] {
                        self.lower_expression(expr)
                    } else {
                        // Single non-expression statement (e.g., Let) — lower normally
                        self.lower_statement(&block.statements[0]);
                        None
                    }
                } else {
                    // Multi-statement block: lower all statements, use trailing expr if any
                    for stmt in &block.statements {
                        self.lower_statement(stmt);
                    }
                    if let Some(trailing_expr) = &block.expr {
                        self.lower_expression(trailing_expr)
                    } else {
                        None
                    }
                }
            }
            _ => self.lower_expression(body),
        };

        // Infer return type from actual generated code (borrows function immutably)
        let return_type = {
            let lambda_func = self.builder.module.functions.get(&func_id)?;
            self.infer_lambda_return_type(lambda_func, entry_block, body_result)
        };

        // The block the builder ENDED on after lowering the body — for a body
        // whose last construct is a loop / conditional this is the loop-exit /
        // merge block, NOT the entry block. The implicit return must terminate
        // THAT block (mirroring how `ensure_terminator` finalizes a regular
        // function on its current block). Finalizing only the entry block left
        // a fall-through Void lambda's loop-exit block as `Unreachable`, which
        // executes as a trap (`udf` / SIGILL) once the loop exits — e.g. a
        // `(lo,hi,n)->{ var i=lo; while(i<hi){...;i++;} }` worker closure.
        let term_block = self.builder.current_block().unwrap_or(entry_block);

        // Update signature and add terminator (borrows function mutably)
        {
            let lambda_func = self.builder.module.functions.get_mut(&func_id)?;
            debug!(
                "Updating lambda signature from {:?} to {:?}",
                lambda_func.signature.return_type, return_type
            );
            debug!(
                "Lambda has {} parameters: {:?}",
                lambda_func.signature.parameters.len(),
                lambda_func
                    .signature
                    .parameters
                    .iter()
                    .map(|p| &p.name)
                    .collect::<Vec<_>>()
            );
            lambda_func.signature.return_type = return_type.clone();
            Self::finalize_lambda_terminator_static(
                lambda_func,
                term_block,
                body_result,
                &return_type,
            )?;
        }

        // Restore state
        self.current_env_layout = None;
        self.restore_state(saved_state);

        Some(func_id)
    }

    /// Lower `HirExprKind::MethodReference { receiver, method_symbol }`
    /// to a closure value: `{ fn_ptr: thunk, env_ptr: env_struct }`
    /// where the env struct holds the receiver at offset 0. Calling
    /// the closure dispatches through the thunk, which loads the
    /// receiver back from env and invokes the underlying method.
    pub(crate) fn lower_method_reference(
        &mut self,
        receiver: &HirExpr,
        method_symbol: SymbolId,
    ) -> Option<IrId> {
        let receiver_reg = self.lower_expression(receiver)?;
        let method_func_id = *self.function_map.get(&method_symbol)?;
        let thunk_id = self.ensure_method_ref_thunk(method_func_id)?;
        self.builder
            .build_make_closure(thunk_id, vec![receiver_reg])
    }
}
