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
    pub(crate) fn refresh_boxed_capture_symbols(&mut self) {
        let captures = self
            .current_drop_points
            .as_ref()
            .map(|points| points.lambda_captures.clone())
            .unwrap_or_default();
        self.boxed_capture_symbols = captures
            .into_iter()
            .filter(|symbol| {
                self.symbol_table.get_symbol(*symbol).map_or(false, |sym| {
                    matches!(
                        sym.kind,
                        crate::tast::SymbolKind::Variable | crate::tast::SymbolKind::Parameter
                    ) && sym.mutability != crate::tast::Mutability::Immutable
                })
            })
            .collect();
    }

    /// Put a captured mutable binding in a shared cell at its declaration.
    /// Allocating here (rather than at the lambda expression) makes the cell
    /// dominate closures created conditionally and lets multiple closures
    /// reuse exactly the same binding.
    pub(crate) fn box_capture_binding(&mut self, symbol: SymbolId, value: IrId) -> Option<IrId> {
        if !self.boxed_capture_symbols.contains(&symbol) {
            return Some(value);
        }
        if let Some(&cell) = self.capture_cells.get(&symbol) {
            self.builder.build_store(cell, value)?;
            return Some(cell);
        }
        let cell = self.build_heap_alloc(8)?;
        self.builder.build_store(cell, value)?;
        self.capture_cells.insert(symbol, cell);
        Some(cell)
    }

    pub(crate) fn lower_lambda(
        &mut self,
        params: &[HirParam],
        body: &HirExpr,
        captures: &[HirCapture],
        lambda_type: TypeId,
    ) -> Option<IrId> {
        // A lambda lowers to a function taking (env*, params...) plus a MakeClosure
        // that allocates the environment struct, copies the captured values into it
        // and yields a { func_ptr, env_ptr } closure.

        // Collect captured values before generate_lambda_function, which
        // saves/restores lowering state.
        let mut captured_values = Vec::new();
        // Global functions (like `trace`) are resolved by name, never captured.
        let filtered_captures: Vec<_> = captures
            .iter()
            .filter(|c| {
                if let Some(sym) = self.symbol_table.get_symbol(c.symbol) {
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
            // An implicit `this` gets a fresh symbol at each site AST lowering
            // synthesises one, so a closure reading an instance field captures a
            // symbol that was never bound to a register. The receiver is always
            // SymbolId(0); reading a variable already redirects a `this`-named
            // symbol there, and a capture has to do the same or every closure
            // touching a field of its enclosing object fails to compile.
            let capture_symbol = self
                .symbol_map
                .get(&capture.symbol)
                .map(|_| capture.symbol)
                .or_else(|| {
                    let named_this = self
                        .symbol_table
                        .get_symbol(capture.symbol)
                        .and_then(|sym| self.string_interner.get(sym.name))
                        .map(|name| name == "this")
                        .unwrap_or(false);
                    named_this.then(|| SymbolId::from_raw(0))
                })
                .unwrap_or(capture.symbol);
            let captured_val = if capture.mode == HirCaptureMode::ByMutableRef {
                self.capture_cells
                    .get(&capture_symbol)
                    .copied()
                    .or_else(|| {
                        let value = self.symbol_map.get(&capture_symbol).copied()?;
                        self.boxed_capture_symbols.insert(capture_symbol);
                        self.box_capture_binding(capture_symbol, value)
                    })
            } else {
                self.symbol_map.get(&capture_symbol).copied()
            };
            if let Some(captured_val) = captured_val {
                debug!("  Found! Register: {:?}", captured_val);
                captured_values.push(captured_val);
            } else {
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

        // Pass the filtered captures so the lambda doesn't try to load global
        // functions out of the environment.
        let filtered_captures_slice: Vec<HirCapture> =
            filtered_captures.iter().map(|c| (*c).clone()).collect();
        let lambda_func_id =
            self.generate_lambda_function(params, body, &filtered_captures_slice, lambda_type)?;

        let result = self
            .builder
            .build_make_closure(lambda_func_id, captured_values);

        // Ownership of a captured variable moves into the closure environment, so
        // the enclosing scope must not free it as well.
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

    /// Lower a lambda body and infer its signature.
    #[allow(dead_code)] // Used once lowering switches to two passes
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

        let saved_state = self.save_state();

        // Switch to lambda context
        self.builder.current_function = Some(func_id);
        self.builder.current_block = Some(entry_block);
        self.symbol_map.clear();
        self.capture_cells.clear();
        self.boxed_capture_symbols.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        // Per-function isolation: lambda body has its own SSA register namespace;
        // saved_state already snapshotted strict_move_locals for restore on exit.
        self.strict_move_locals.clear();
        self.reset_move_recorder();
        // Lambda body lowers its own loops with a fresh loop-carried stack.
        self.loop_carried_symbols.clear();
        self.current_env_layout = env_layout.clone();

        // Drop tracking must be cleared: a lambda body is a separate function with
        // its own register namespace, so cleanup_all_scopes would otherwise free
        // the parent's owned heap values through IrIds naming different registers.
        self.owned_heap_values.clear();
        self.drop_scope_stack.clear();
        self.temp_heap_values.clear();
        self.reassigned_in_scope.clear();
        self.current_drop_points = match &body.kind {
            HirExprKind::Block(block) => {
                let mut analyzer = DropPointAnalyzer::new();
                Some(analyzer.analyze_function(block))
            }
            _ => None,
        };
        self.refresh_boxed_capture_symbols();
        self.current_stmt_index = 0;

        // Map lambda parameters to registers and register them as locals, the same
        // way regular function parameters are set up.
        for (i, param) in params.iter().enumerate() {
            let param_reg = IrId::new(param_offset + i as u32);
            self.symbol_map.insert(param.symbol_id, param_reg);
            let _ = self.box_capture_binding(param.symbol_id, param_reg);

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
                if field.mode == HirCaptureMode::ByMutableRef {
                    let cell = layout.load_cell_ptr(&mut self.builder, env_ptr, field.symbol)?;
                    self.capture_cells.insert(field.symbol, cell);
                }
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
                    // Single-expression block, the common arrow-function shape.
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

        // The block the builder ended on after lowering the body: for a body whose
        // last construct is a loop or conditional that is the loop-exit / merge
        // block, not the entry block. The implicit return must terminate that
        // block (as `ensure_terminator` does for a regular function), otherwise it
        // stays `Unreachable` and traps at runtime.
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
