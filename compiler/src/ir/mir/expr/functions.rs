//! Function bodies: free functions, instance methods, constructors, async.

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
use std::sync::OnceLock;
use std::time::Instant;

fn profile_mir_functions_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RAYZOR_PROFILE_MIR_FUNCTIONS").is_some())
}

fn mir_function_profile_record(
    module: Option<&str>,
    func: &IrFunction,
    start: Instant,
    statements: usize,
    has_expr: bool,
    kind: &str,
) {
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let blocks = func.cfg.blocks.len();
    let instructions: usize = func.cfg.blocks.values().map(|b| b.instructions.len()).sum();
    let name = func.qualified_name.as_deref().unwrap_or(&func.name);
    eprintln!(
        "  mir-function: total={:.2}ms blocks={} insts={} stmts={} expr={} kind={} module={} fn={}",
        elapsed_ms,
        blocks,
        instructions,
        statements,
        has_expr,
        kind,
        module.unwrap_or("?"),
        name
    );
}

impl<'a> HirToMirContext<'a> {
    /// Lower a HIR function to MIR (Legacy - combines Pass 1 and Pass 2)
    pub(crate) fn lower_function(&mut self, symbol_id: SymbolId, hir_func: &HirFunction) {
        let body_stmt_count = hir_func
            .body
            .as_ref()
            .map(|b| b.statements.len())
            .unwrap_or(0);

        self.record_param_ownership(symbol_id, hir_func);
        self.record_consume_method(symbol_id, hir_func);
        let signature = self.build_function_signature(hir_func);

        let func_name = self
            .string_interner
            .get(hir_func.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("func_{}", symbol_id.as_raw()));
        let func_id = self.builder.start_function(symbol_id, func_name, signature);

        self.function_map.insert(symbol_id, func_id);

        if hir_func.is_keep {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes
                    .custom
                    .insert("keep".to_string(), "true".to_string());
            }
        }

        // Record constrained type parameter info for call-site fat pointer wrapping
        self.record_constrained_params(func_id, hir_func, false);

        // Optimization hints derived from DFG/SSA analysis.
        if self.ssa_hints.inline_candidates.contains(&symbol_id) {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes.inline = crate::ir::InlineHint::Always;
            }
        }

        // Respect Haxe `inline` keyword — always inline these functions
        if hir_func.is_inline {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes.inline = crate::ir::InlineHint::Always;
            }
        }

        if self.ssa_hints.straight_line_functions.contains(&symbol_id) {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes.pure = true; // Assume pure for straight-line code
            }
        }

        if self
            .ssa_hints
            .complex_control_flow_functions
            .contains(&symbol_id)
        {
            // Complex phi nodes benefit from full optimization passes.
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.attributes.optimize_size = false;
            }
        }

        // qualified_name and source_location are set in lower_function_body (Pass 2).

        for (i, param) in hir_func.params.iter().enumerate() {
            if let Some(func) = self.builder.current_function() {
                if let Some(sig_param) = func.signature.parameters.get(i) {
                    let param_reg = sig_param.reg;
                    self.symbol_map.insert(param.symbol_id, param_reg);

                    // Register the parameter as a local so Cranelift can find its type
                    let param_type = self.convert_type(param.ty);
                    let src_loc = IrSourceLocation::unknown();
                    let param_name = self.interned_str(param.name).to_string();
                    if let Some(func_mut) = self.builder.current_function_mut() {
                        func_mut.locals.insert(
                            param_reg,
                            crate::ir::IrLocal {
                                name: param_name,
                                ty: param_type,
                                mutable: false, // Parameters are immutable by default
                                source_location: src_loc,
                                allocation: crate::ir::AllocationHint::Register,
                            },
                        );
                    }
                }
            }
        }

        // A by-value parameter of a `@:move` class owns its value, so it is a
        // binding the move-flow analysis tracks from function entry.
        self.enroll_move_params(hir_func);
        self.enroll_borrowed_params(hir_func);

        if let Some(body) = &hir_func.body {
            let stmt_count = body.statements.len();
            let has_expr = body.expr.is_some();

            self.lower_block(body);

            // Add implicit return if needed
            self.ensure_terminator();
        }

        self.check_move_flow();
        self.builder.finish_function();

        // Clear per-function state
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();
        self.block_map.clear();
    }

    /// Lower a function body after signature is registered (Pass 2)
    /// Reuses the existing function created in Pass 1
    pub(crate) fn lower_function_body(
        &mut self,
        symbol_id: SymbolId,
        hir_func: &HirFunction,
        this_type: Option<TypeId>,
        class_symbol: Option<SymbolId>,
    ) {
        let profile_fn = profile_mir_functions_enabled();
        let profile_start = if profile_fn {
            Some(Instant::now())
        } else {
            None
        };
        let profile_statements = hir_func
            .body
            .as_ref()
            .map(|body| body.statements.len())
            .unwrap_or(0);
        let profile_has_expr = hir_func
            .body
            .as_ref()
            .map(|body| body.expr.is_some())
            .unwrap_or(false);

        // The function already exists from Pass 1, we just need to fill in the body
        let func_id = self
            .function_map
            .get(&symbol_id)
            .copied()
            .expect("Function should have been registered in Pass 1");

        // Re-open the function for body lowering
        let func = self
            .builder
            .module
            .functions
            .get(&func_id)
            .expect("Function should exist")
            .clone();

        self.builder.current_function = Some(func_id);
        self.builder.current_block = Some(func.entry_block());
        self.current_function_symbol = Some(symbol_id);

        // Qualified name for stack traces; already carries the class prefix
        // (e.g. "Main.thrower") from TAST lowering.
        if let Some(qname_id) = hir_func.qualified_name {
            if let Some(qname_str) = self.string_interner.get(qname_id) {
                let qname = qname_str.to_string();
                if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
                    func_mut.qualified_name = Some(qname);
                }
            }
        } else if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
            // Fallback: build Class.method from this_type or current_module
            if let Some(class_type) = this_type {
                if let Some(class_name) = self
                    .class_type_to_symbol
                    .get(&class_type)
                    .and_then(|&sym| self.symbol_table.get_symbol(sym))
                    .and_then(|s| {
                        s.qualified_name
                            .and_then(|n| self.string_interner.get(n))
                            .or_else(|| self.string_interner.get(s.name))
                    })
                {
                    func_mut.qualified_name = Some(format!("{}.{}", class_name, func_mut.name));
                }
            } else if let Some(module_name) =
                self.current_module.as_deref().filter(|m| !m.is_empty())
            {
                let bare = func_mut.name.clone();
                func_mut.qualified_name = Some(format!("{}.{}", module_name, bare));
            }
        }

        // @:export on the function or its owning class marks it for WASM export.
        {
            let mut should_export = false;
            if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                if sym.flags.is_wasm_export() {
                    should_export = true;
                }
            }
            // Check owning class for @:export (instance methods via this_type)
            if let Some(class_type) = this_type {
                if let Some(&class_sym_id) = self.class_type_to_symbol.get(&class_type) {
                    if let Some(class_sym) = self.symbol_table.get_symbol(class_sym_id) {
                        if class_sym.flags.is_wasm_export() {
                            should_export = true;
                        }
                    }
                }
            }
            // Check owning class for @:export (static methods via class_symbol)
            if let Some(class_sym_id) = class_symbol {
                if let Some(class_sym) = self.symbol_table.get_symbol(class_sym_id) {
                    if class_sym.flags.is_wasm_export() {
                        should_export = true;
                    }
                }
            }
            if should_export {
                if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
                    func_mut.wasm_export = true;
                }
            }
        }

        // Propagate @:jsImport from symbol or owning class to MIR function.
        // Method-level @:jsImport("mod", "func") takes precedence.
        // Class-level @:jsImport("mod") + method @:native("func-name") also works.
        {
            let mut js_mod: Option<String> = None;
            let mut js_func: Option<String> = None;

            // Check method's own @:jsImport
            if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                if let Some((module_is, func_is)) = sym.js_import {
                    js_mod = Some(
                        self.string_interner
                            .get(module_is)
                            .unwrap_or("")
                            .to_string(),
                    );
                    js_func = Some(self.string_interner.get(func_is).unwrap_or("").to_string());
                }
            }

            // Check owning class for @:jsImport (via this_type or class_symbol)
            if js_mod.is_none() {
                let class_sym_id = class_symbol
                    .or_else(|| this_type.and_then(|t| self.class_type_to_symbol.get(&t).copied()));
                if let Some(csid) = class_sym_id {
                    if let Some(class_sym) = self.symbol_table.get_symbol(csid) {
                        if let Some((mod_is, _)) = class_sym.js_import {
                            js_mod =
                                Some(self.string_interner.get(mod_is).unwrap_or("").to_string());
                            // Use the function's native name or bare name as import name
                            if let Some(func_ref) = self.builder.module.functions.get(&func_id) {
                                js_func = Some(func_ref.name.clone());
                            }
                        }
                    }
                }
            }

            if let (Some(module), Some(func_name)) = (js_mod, js_func) {
                if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
                    func_mut.js_import = Some((module, func_name));
                }
            }
        }

        // Set source location from HIR function (propagated from TAST TypedFunction.source_location).
        {
            let loc = hir_func.source_location;
            if loc.is_valid() && loc.line > 0 {
                if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
                    func_mut.source_location = IrSourceLocation {
                        file_id: loc.file_id,
                        line: loc.line,
                        column: loc.column,
                    };
                }
            }
        }

        // Clear per-function drop tracking state
        self.owned_heap_values.clear();
        self.drop_scope_stack.clear();
        self.temp_heap_values.clear();
        self.reassigned_in_scope.clear();
        self.capture_cells.clear();
        self.boxed_capture_symbols.clear();

        if let Some(body) = &hir_func.body {
            let mut analyzer = DropPointAnalyzer::new();
            self.current_drop_points = Some(analyzer.analyze_function(body));
            self.current_stmt_index = 0;
            trace!(
                "Drop analysis: Found {} heap variables, {} escaping",
                self.current_drop_points
                    .as_ref()
                    .map(|dp| dp.heap_allocated.len())
                    .unwrap_or(0),
                self.current_drop_points
                    .as_ref()
                    .map(|dp| dp.escaping.len())
                    .unwrap_or(0)
            );
        } else {
            self.current_drop_points = None;
            self.current_stmt_index = 0;
        }
        self.refresh_boxed_capture_symbols();

        // Set current_this_type for implicit field access resolution
        self.current_this_type = this_type;

        // Track return type for Null<T> boxing in Return handler
        self.current_function_return_type = Some(hir_func.return_type);

        // Map 'this' parameter for instance methods
        if this_type.is_some() {
            // 'this' is parameter 0
            if let Some(this_param) = func.signature.parameters.get(0) {
                // Map 'this' to a special symbol ID (SymbolId(0))
                // This is what HirExprKind::This looks up
                self.symbol_map
                    .insert(SymbolId::from_raw(0), this_param.reg);
            }
        }

        // Map parameters from the function signature (which already has register IDs assigned)
        let param_offset = if this_type.is_some() { 1 } else { 0 };
        for (i, param) in hir_func.params.iter().enumerate() {
            if let Some(sig_param) = func.signature.parameters.get(i + param_offset) {
                self.symbol_map.insert(param.symbol_id, sig_param.reg);
                let _ = self.box_capture_binding(param.symbol_id, sig_param.reg);
            }
        }

        // Lower body
        let func_name = self
            .string_interner
            .get(hir_func.name)
            .unwrap_or("?")
            .to_string();

        // @:async function transform: wrap body in a closure and return Future handle
        if hir_func.is_async {
            if let Some(body) = &hir_func.body {
                self.lower_async_function_body(func_id, hir_func, body, &func_name);
            } else {
                self.ensure_terminator();
            }
        } else if let Some(body) = &hir_func.body {
            debug!(
                "[lower_function_body]: {} has body with {} statements, expr: {}",
                func_name,
                body.statements.len(),
                body.expr.is_some()
            );
            if log::log_enabled!(log::Level::Debug) {
                for (i, stmt) in body.statements.iter().enumerate() {
                    debug!(
                        "[lower_function_body]: {} - stmt[{}] = {:?}",
                        func_name,
                        i,
                        std::mem::discriminant(stmt)
                    );
                }
            }
            // Parameter ownership is per-body, and this is a body: Pass 1 only
            // registered the signature, so without this a function lowered here
            // has no enrolled parameters and every check on them is vacuous.
            self.enroll_move_params(hir_func);
            self.enroll_borrowed_params(hir_func);

            // Function-level scope so its variables are freed on function exit.
            self.enter_drop_scope();
            self.lower_block(body);
            // Implicit-return path: emit drop() only for @:derive(Drop) values.
            // Non-Drop allocations are freed by InsertFreePass, so a full
            // cleanup_all_scopes() here would double-free. Explicit returns are
            // handled in the Return statement handler.
            if !self.is_terminated() && !self.derive_drop_classes.is_empty() {
                self.cleanup_drop_classes_only();
            }
            self.ensure_terminator();
        } else {
            debug!("[lower_function_body]: {} has NO body", func_name);
            // Functions without body (extern, abstract) still need a terminator
            self.ensure_terminator();
        }

        self.check_move_flow();
        self.builder.finish_function();

        if let Some(start) = profile_start {
            if let Some(func) = self.builder.module.functions.get(&func_id) {
                mir_function_profile_record(
                    self.current_module.as_deref(),
                    func,
                    start,
                    profile_statements,
                    profile_has_expr,
                    "function",
                );
            }
        }

        self.symbol_map.clear();
        self.capture_cells.clear();
        self.boxed_capture_symbols.clear();

        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        // strict_move_locals holds IrIds in THIS function's SSA namespace;
        // stale entries would mis-emit MarkMoved/CheckLive in the next one.
        self.strict_move_locals.clear();
        self.reset_move_recorder();
        self.block_map.clear();
        self.current_this_type = None;
    }

    /// Lower an @:async function body.
    ///
    /// Transforms the function so that:
    /// 1. The original body becomes an inner closure function (env_ptr -> i32)
    /// 2. The outer function allocates an environment, stores params, and calls
    ///    rayzor_future_create(fn_ptr, env_ptr) to return a lazy Future handle.
    pub(crate) fn lower_async_function_body(
        &mut self,
        func_id: IrFunctionId,
        hir_func: &HirFunction,
        body: &HirBlock,
        func_name: &str,
    ) {
        use crate::ir::hir::{HirCapture, HirCaptureMode, HirExprKind};

        debug!(
            "[lower_async_function_body]: transforming {} to return Future",
            func_name
        );

        // 1. Create captures from function params (the inner closure captures all params)
        let captures: Vec<HirCapture> = hir_func
            .params
            .iter()
            .map(|p| HirCapture {
                symbol: p.symbol_id,
                mode: HirCaptureMode::ByValue,
                ty: p.ty,
            })
            .collect();

        // 2. Generate the inner closure function (async body)
        //    Signature: (env_ptr: *u8) -> i32 (JIT closure convention)
        //    The inner function has NO explicit params — all values are loaded from env.
        let inner_context = self.generate_lambda_skeleton(&[], &captures);
        let inner_func_id = inner_context.func_id;

        // 3. Lower the original body into the inner function
        {
            let saved_state = self.save_state();
            // Per-function isolation: the inner body lowers its own loops with a
            // fresh loop-carried stack; don't consult the parent's.
            self.loop_carried_symbols.clear();

            self.builder.current_function = Some(inner_context.func_id);
            self.builder.current_block = Some(inner_context.entry_block);
            self.symbol_map.clear();
            // Register-keyed: IrIds restart per function, so stale entries
            // from the previous body would collide with unrelated registers.
            self.interface_call_result_types.clear();
            self.boxed_value_regs.clear();
            self.boxed_value_regs.clear();
            // Per-function isolation: inner async closure has its own SSA namespace;
            // saved_state already snapshotted strict_move_locals for restore.
            self.strict_move_locals.clear();
            self.reset_move_recorder();
            self.current_env_layout = inner_context.env_layout.clone();

            // Clear drop tracking for inner function
            self.owned_heap_values.clear();
            self.drop_scope_stack.clear();
            self.temp_heap_values.clear();
            self.reassigned_in_scope.clear();
            self.current_drop_points = None;
            self.current_stmt_index = 0;

            // Load captured params from environment
            if let Some(layout) = &inner_context.env_layout {
                let env_ptr = IrId::new(0); // First parameter
                for field in &layout.fields {
                    if let Some(value_reg) =
                        layout.load_field(&mut self.builder, env_ptr, field.symbol)
                    {
                        self.symbol_map.insert(field.symbol, value_reg);
                        // Register as local for type inference
                        let param_name = self
                            .string_interner
                            .get(
                                self.symbol_table
                                    .get_symbol(field.symbol)
                                    .map(|s| s.name)
                                    .unwrap_or_default(),
                            )
                            .unwrap_or("<capture>")
                            .to_string();
                        let field_type = field.storage_ty.clone();
                        if let Some(func) = self.builder.current_function_mut() {
                            func.locals.insert(
                                value_reg,
                                crate::ir::IrLocal {
                                    name: param_name,
                                    ty: field_type,
                                    mutable: false,
                                    source_location: crate::ir::IrSourceLocation::unknown(),
                                    allocation: crate::ir::AllocationHint::Register,
                                },
                            );
                        }
                    }
                }
            }

            // Lower the function body
            self.enter_drop_scope();
            self.lower_block(body);
            self.ensure_terminator();

            // Fix up return type: inner closure returns I64 (JIT convention)
            if let Some(inner_func) = self
                .builder
                .module
                .functions
                .get_mut(&inner_context.func_id)
            {
                inner_func.signature.return_type = IrType::I64;
            }

            self.current_env_layout = None;
            self.restore_state(saved_state);
        }

        // 4. Back in the outer function: create closure and call rayzor_future_create
        // Collect captured values (param registers from the outer function's symbol_map)
        let captured_values: Vec<IrId> = hir_func
            .params
            .iter()
            .filter_map(|p| self.symbol_map.get(&p.symbol_id).copied())
            .collect();

        // Create closure object (MakeClosure instruction)
        // MakeClosure allocates struct { fn_ptr: i64, env_ptr: i64 } on heap
        let closure_reg = match self
            .builder
            .build_make_closure(inner_func_id, captured_values)
        {
            Some(r) => r,
            None => {
                self.ensure_terminator();
                return;
            }
        };

        // Extract fn_ptr (offset 0) and env_ptr (offset 8) from closure struct
        let ptr_u8_ty = IrType::Ptr(Box::new(IrType::U8));
        let fn_ptr = match self.builder.build_load(closure_reg, ptr_u8_ty.clone()) {
            Some(r) => r,
            None => {
                self.ensure_terminator();
                return;
            }
        };
        // PtrAdd closure_reg + 8 bytes to get env_ptr field address
        let eight = match self.builder.build_const(IrValue::I64(8)) {
            Some(r) => r,
            None => {
                self.ensure_terminator();
                return;
            }
        };
        let env_field_ptr = match self
            .builder
            .build_ptr_add(closure_reg, eight, ptr_u8_ty.clone())
        {
            Some(r) => r,
            None => {
                self.ensure_terminator();
                return;
            }
        };
        let env_ptr = match self.builder.build_load(env_field_ptr, ptr_u8_ty.clone()) {
            Some(r) => r,
            None => {
                self.ensure_terminator();
                return;
            }
        };

        // Ensure all Future runtime externs are declared in this module
        let (create_id, _await_id, _poll_id, _is_ready_id) = self.ensure_future_externs();

        let ptr_u8_ty = IrType::Ptr(Box::new(IrType::U8));
        if let Some(future_handle) =
            self.builder
                .build_call_direct(create_id, vec![fn_ptr, env_ptr], ptr_u8_ty)
        {
            // Return the future handle
            self.builder.build_return(Some(future_handle));

            // Update the outer function's return type to Ptr(U8) (future handle)
            if let Some(outer_func) = self.builder.module.functions.get_mut(&func_id) {
                outer_func.signature.return_type = IrType::Ptr(Box::new(IrType::U8));
            }
        } else {
            self.ensure_terminator();
        }
    }

    /// Lower an instance method (non-static class method) to MIR
    /// Instance methods receive an implicit 'this' parameter as their first argument
    pub(crate) fn lower_instance_method(
        &mut self,
        symbol_id: SymbolId,
        hir_func: &HirFunction,
        class_type_id: TypeId,
    ) {
        // Build MIR function signature with implicit 'this' parameter
        let signature = self.build_instance_method_signature(hir_func, class_type_id);

        // Start building the function
        let func_name = self
            .string_interner
            .get(hir_func.name)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("func_{}", symbol_id.as_raw()));
        let func_id = self.builder.start_function(symbol_id, func_name, signature);

        // Store function mapping for call resolution
        self.function_map.insert(symbol_id, func_id);

        // Set qualified name for debugging, profiling, and stack traces
        if let Some(qualified_name) = hir_func.qualified_name {
            if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                func.qualified_name = self
                    .string_interner
                    .get(qualified_name)
                    .map(|s| s.to_string());
            }
        } else if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
            // Build qualified name from class type
            let class_name = self
                .class_type_to_symbol
                .get(&class_type_id)
                .and_then(|&sym| self.symbol_table.get_symbol(sym))
                .and_then(|s| {
                    s.qualified_name
                        .and_then(|n| self.string_interner.get(n))
                        .or_else(|| self.string_interner.get(s.name))
                });
            if let Some(cn) = class_name {
                func.qualified_name = Some(format!("{}.{}", cn, func.name));
            }
        }

        // Set source location from symbol table for stack traces
        if let Some(sym) = self.symbol_table.get_symbol(hir_func.symbol_id) {
            let loc = sym.definition_location;
            if loc.line > 0 {
                if let Some(func) = self.builder.module.functions.get_mut(&func_id) {
                    func.source_location = IrSourceLocation {
                        file_id: loc.file_id,
                        line: loc.line,
                        column: loc.column,
                    };
                }
            }
        }

        // Parameter 0 is the implicit 'this'; user parameters start at index 1.
        if let Some(func) = self.builder.current_function() {
            // TODO: 'this' should come from TAST/HIR instead of a synthetic SymbolId.
            if let Some(this_param) = func.signature.parameters.get(0) {
                let this_symbol = SymbolId::from_raw(0); // Synthetic 'this' symbol
                self.symbol_map.insert(this_symbol, this_param.reg);
            }

            for (i, param) in hir_func.params.iter().enumerate() {
                if let Some(sig_param) = func.signature.parameters.get(i + 1) {
                    let param_reg = sig_param.reg;
                    self.symbol_map.insert(param.symbol_id, param_reg);
                }
            }
        }

        // A by-value parameter of a `@:move` class owns its value, so it is a
        // binding the move-flow analysis tracks from function entry.
        self.enroll_move_params(hir_func);
        self.enroll_borrowed_params(hir_func);

        // Lower the function body
        if let Some(body) = &hir_func.body {
            debug!(
                "lower_instance_method: {} has body with {} statements",
                self.string_interner.get(hir_func.name).unwrap_or("?"),
                body.statements.len()
            );

            // Clear per-function drop tracking state
            self.owned_heap_values.clear();
            self.drop_scope_stack.clear();
            self.temp_heap_values.clear();
            self.reassigned_in_scope.clear();

            // Enter function-level scope for variables declared at function level
            self.enter_drop_scope();
            self.lower_block(body);
            debug!(
                "lower_instance_method: {} - after lower_block",
                self.string_interner.get(hir_func.name).unwrap_or("?")
            );

            // Add implicit return if needed
            self.ensure_terminator();
            debug!(
                "lower_instance_method: {} - after ensure_terminator",
                self.string_interner.get(hir_func.name).unwrap_or("?")
            );
        } else {
            debug!(
                "lower_instance_method: {} has NO body!",
                self.string_interner.get(hir_func.name).unwrap_or("?")
            );
        }

        if let Some(func) = self.builder.current_function() {
            debug!(
                "lower_instance_method: {} - CFG has {} blocks",
                func.name,
                func.cfg.blocks.len()
            );
            if log::log_enabled!(log::Level::Debug) {
                for (block_id, block) in &func.cfg.blocks {
                    debug!(
                        "  Block {:?}: {} instructions, terminator: {:?}",
                        block_id,
                        block.instructions.len(),
                        block.terminator
                    );
                }
            }
        }

        self.check_move_flow();
        self.builder.finish_function();

        // Clear per-function state
        self.symbol_map.clear();
        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();
        self.block_map.clear();
    }

    /// Lower constructor body (Pass 2)
    pub(crate) fn lower_constructor_body(
        &mut self,
        class_symbol: SymbolId,
        constructor: &HirConstructor,
        type_id: TypeId,
        parent_type: Option<TypeId>,
    ) {
        let profile_fn = profile_mir_functions_enabled();
        let profile_start = if profile_fn {
            Some(Instant::now())
        } else {
            None
        };
        let profile_statements = constructor.pre_super_stmts.len()
            + constructor.field_inits.len()
            + constructor.body.statements.len();
        let profile_has_expr = constructor.body.expr.is_some();

        let func_id = self
            .constructor_map
            .get(&type_id)
            .copied()
            .expect("Constructor should have been registered in Pass 1");

        let func = self
            .builder
            .module
            .functions
            .get(&func_id)
            .expect("Constructor function should exist")
            .clone();

        self.builder.current_function = Some(func_id);
        self.builder.current_block = Some(func.entry_block());

        // Must be cleared before the DropPointAnalyzer runs below, so the
        // constructor body is analysed against a clean slate rather than
        // inheriting the previously-lowered function's state.
        self.owned_heap_values.clear();
        self.drop_scope_stack.clear();
        self.temp_heap_values.clear();
        self.reassigned_in_scope.clear();

        // Drop-point analysis on the constructor body, as `lower_function_body`
        // does. Without it the body lowers under the previously-lowered
        // function's `current_drop_points` / `current_stmt_index`, misrouting
        // `build_free` / `build_drop` for heap values from extern callees.
        let mut analyzer = DropPointAnalyzer::new();
        self.current_drop_points = Some(analyzer.analyze_function(&constructor.body));
        self.current_stmt_index = 0;

        // Implicit field accesses in the body resolve against the class type.
        self.current_this_type = Some(type_id);

        // HIR constructors have no return type expression; at MIR level they
        // return the allocated instance pointer, so use the class type_id.
        self.current_function_return_type = Some(type_id);

        // Take 'this' and the parameters from the signature's own `reg` fields
        // (assigned in Pass 1). Fabricating IrIds positionally would collide
        // with registers earlier passes already allocated.
        let this_reg = func
            .signature
            .parameters
            .first()
            .map(|p| p.reg)
            .unwrap_or_else(|| IrId::new(0));
        self.symbol_map.insert(SymbolId::from_raw(0), this_reg);
        for (i, param) in constructor.params.iter().enumerate() {
            if let Some(sig_param) = func.signature.parameters.get(i + 1) {
                self.symbol_map.insert(param.symbol_id, sig_param.reg);
            }
        }

        // Must come after the drop-points analysis, so the analyzer sees the
        // cleared heap-tracking state and doesn't double-count escapes.
        self.enter_drop_scope();

        // Execute pre-super statements (e.g., field assignments that come before
        // super() in the source code and are needed by virtual methods called from super)
        for stmt in &constructor.pre_super_stmts {
            self.lower_statement(stmt);
        }

        // Handle super() call if present
        if let Some(super_call) = &constructor.super_call {
            if let Some(parent_type_id) = parent_type {
                // Look up parent constructor by TypeId first, then by name as fallback
                let parent_ctor_id =
                    self.constructor_map
                        .get(&parent_type_id)
                        .copied()
                        .or_else(|| {
                            // TypeId mismatch: class.extends uses TAST TypeIds, constructor_map uses MIR TypeIds.
                            // Fall back to looking up by class name via constructor_name_map.
                            // Resolve parent class symbol from the type_table.
                            let type_table = self.type_table;
                            let parent_symbol = type_table.get(parent_type_id).and_then(|ti| {
                                if let TypeKind::Class { symbol_id, .. } = &ti.kind {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            });

                            if let Some(parent_sym) = parent_symbol {
                                if let Some(sym_info) = self.symbol_table.get_symbol(parent_sym) {
                                    // Try qualified name first
                                    if let Some(qual_name) = sym_info
                                        .qualified_name
                                        .and_then(|q| self.string_interner.get(q))
                                    {
                                        if let Some(&fid) = self.constructor_name_map.get(qual_name)
                                        {
                                            return Some(fid);
                                        }
                                    }
                                    // Try simple name
                                    if let Some(name) = self.string_interner.get(sym_info.name) {
                                        if let Some(&fid) = self.constructor_name_map.get(name) {
                                            return Some(fid);
                                        }
                                    }
                                }
                            }

                            // The parent's TypeId can be renumbered when type_table
                            // entries are rebuilt across compilation contexts;
                            // `class_parent_map` is keyed by stable SymbolIds, so
                            // resolve through it to the name, then constructor_name_map.
                            if let Some(&parent_sym) = self.class_parent_map.get(&class_symbol) {
                                if let Some(sym_info) = self.symbol_table.get_symbol(parent_sym) {
                                    if let Some(qual_name) = sym_info
                                        .qualified_name
                                        .and_then(|q| self.string_interner.get(q))
                                    {
                                        if let Some(&fid) = self.constructor_name_map.get(qual_name)
                                        {
                                            return Some(fid);
                                        }
                                    }
                                    if let Some(name) = self.string_interner.get(sym_info.name) {
                                        if let Some(&fid) = self.constructor_name_map.get(name) {
                                            return Some(fid);
                                        }
                                    }
                                }
                            }
                            None
                        });

                // The parent's module may lower later, so its constructor is
                // not in `constructor_name_map` yet. Name the call and let the
                // cross-module fixup bind it, the way `new` does.
                let parent_ctor_key: Option<String> = if parent_ctor_id.is_some() {
                    None
                } else {
                    // A cross-module parent often arrives as a placeholder
                    // rather than a Class, and the placeholder carries the name.
                    self.super_constructor_fqn_key()
                };

                if let Some(parent_ctor_id) = parent_ctor_id {
                    // Lower super call arguments
                    let mut arg_regs = vec![this_reg]; // 'this' is first argument
                    for arg in &super_call.args {
                        if let Some(arg_reg) = self.lower_expression(arg) {
                            arg_regs.push(arg_reg);
                        }
                    }

                    // Fill in default values for any missing optional parameters
                    self.fill_default_args(parent_ctor_id, &mut arg_regs, true);

                    // Call parent constructor (returns void)
                    self.builder.build_call_direct(
                        parent_ctor_id,
                        arg_regs,
                        crate::ir::IrType::Void,
                    );
                } else if let Some(ctor_key) = parent_ctor_key {
                    let mut arg_regs = vec![this_reg];
                    for arg in &super_call.args {
                        if let Some(arg_reg) = self.lower_expression(arg) {
                            arg_regs.push(arg_reg);
                        }
                    }
                    let param_types: Vec<crate::ir::IrType> = arg_regs
                        .iter()
                        .map(|r| {
                            self.builder
                                .get_register_type(*r)
                                .unwrap_or(crate::ir::IrType::Ptr(Box::new(
                                    crate::ir::IrType::Void,
                                )))
                        })
                        .collect();
                    let stub_id = self.register_stdlib_mir_forward_ref(
                        &ctor_key,
                        param_types,
                        crate::ir::IrType::Void,
                    );
                    self.builder
                        .build_call_direct(stub_id, arg_regs, crate::ir::IrType::Void);
                } else {
                    self.add_error(
                        &format!(
                            "Parent constructor not found for TypeId {:?}",
                            parent_type_id
                        ),
                        crate::tast::SourceLocation::unknown(),
                    );
                }
            }
        }

        // Apply field initializers declared at field-declaration sites
        // (e.g. `public var num:Int = 42;`). These run after super() but before
        // the user-written body so explicit assignments in the body can override.
        for field_init in &constructor.field_inits {
            let Some(value_reg) = self.lower_expression(&field_init.value) else {
                continue;
            };
            let Some(&(_class_type, field_index)) = self.field_index_map.get(&field_init.field)
            else {
                continue;
            };
            let Some(index_const) = self.builder.build_const(IrValue::I32(field_index as i32))
            else {
                continue;
            };
            let field_ty = self
                .symbol_table
                .get_symbol(field_init.field)
                .map(|s| self.convert_type(s.type_id))
                .unwrap_or(IrType::I32);
            if let Some(field_ptr) = self
                .builder
                .build_gep(this_reg, vec![index_const], field_ty)
            {
                self.builder.build_store(field_ptr, value_reg);
            }
        }

        // Lower constructor body statements
        for stmt in &constructor.body.statements {
            self.lower_statement(stmt);
        }

        // Constructor implicitly returns void
        self.builder.build_return(None);

        // Set qualified name and @:export flag for constructor
        if let Some(func_mut) = self.builder.module.functions.get_mut(&func_id) {
            if let Some(class_sym) = self.symbol_table.get_symbol(class_symbol) {
                let class_name = class_sym
                    .qualified_name
                    .and_then(|n| self.string_interner.get(n))
                    .or_else(|| self.string_interner.get(class_sym.name))
                    .unwrap_or("Unknown");
                func_mut.qualified_name = Some(format!("{}.new", class_name));
                if class_sym.flags.is_wasm_export() {
                    func_mut.wasm_export = true;
                }
            }
        }

        self.check_move_flow();
        self.builder.finish_function();

        if let Some(start) = profile_start {
            if let Some(func) = self.builder.module.functions.get(&func_id) {
                mir_function_profile_record(
                    self.current_module.as_deref(),
                    func,
                    start,
                    profile_statements,
                    profile_has_expr,
                    "constructor",
                );
            }
        }

        self.symbol_map.clear();

        // Register-keyed: IrIds restart per function, so stale entries
        // from the previous body would collide with unrelated registers.
        self.interface_call_result_types.clear();
        self.boxed_value_regs.clear();
        self.strict_move_locals.clear();
        self.reset_move_recorder();
        self.block_map.clear();
    }
}
