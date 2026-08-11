//! Expression entry point: source spans, drop bookkeeping, then per-kind lowering.

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
    /// Lower a HIR expression to MIR value
    /// Public entry: lower a HIR expression to MIR, then post-process to
    /// wrap a Class value as an interface fat pointer when the expression's
    /// static type was promoted to an Interface by the typechecker (e.g.
    /// when used as a function arg, return value, or array element where
    /// the slot expects an interface). Without this, stdlib calls like
    /// `array_push` and any non-user-function dispatch would store a raw
    /// class pointer, breaking subsequent interface dispatch on reads.
    pub(crate) fn lower_expression(&mut self, expr: &HirExpr) -> Option<IrId> {
        let result = self.lower_expression_inner(expr)?;
        // Promotion wrap: only fires when expr.ty IS an interface AND the
        // expression kind reveals an underlying class. Variable expressions
        // whose symbol is interface-typed correctly skip this (their
        // underlying type is also interface, get_class_symbol returns None).
        // Inner Let/Cast handlers also produce wrapped fat pointers when
        // their RHS is a class; that runs first, and by then expr.ty for
        // the outer expression is the interface, leaving the underlying
        // unchanged — but our extractor only fires when it can extract a
        // class from `kind`, which a Let-result expression cannot, so no
        // double-wrap.
        if let Some(iface_sym) = self.get_interface_symbol(expr.ty) {
            let underlying = self.extract_underlying_class_symbol(expr);
            if let Some(class_sym) = underlying {
                if self.interface_vtables.contains_key(&(class_sym, iface_sym)) {
                    if let Some(wrapped) =
                        self.wrap_in_interface_fat_ptr(result, class_sym, iface_sym)
                    {
                        // Track for the same lifetime caveat as Path 3.
                        self.interface_wrapped_args.insert(wrapped);
                        return Some(wrapped);
                    }
                }
            }
        }
        Some(result)
    }

    /// Lower if statement/expression
    pub(crate) fn lower_if_statement(
        &mut self,
        condition: &HirExpr,
        then_branch: &HirBlock,
        else_branch: Option<&HirBlock>,
    ) {
        debug!(
            "lower_if_statement called, has_else={}",
            else_branch.is_some()
        );
        let Some(then_block) = self.builder.create_block() else {
            return;
        };
        let Some(merge_block) = self.builder.create_block() else {
            return;
        };

        let else_block = if else_branch.is_some() {
            self.builder.create_block().unwrap_or(merge_block)
        } else {
            merge_block
        };

        // Get the current block before branching
        let entry_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            return;
        };

        // Find all variables that are modified in either branch
        let mut modified_vars = std::collections::BTreeSet::new();
        for stmt in &then_branch.statements {
            self.find_modified_variables_in_statement(stmt, &mut modified_vars);
        }
        if let Some(else_br) = else_branch {
            for stmt in &else_br.statements {
                self.find_modified_variables_in_statement(stmt, &mut modified_vars);
            }
        }

        debug!("modified_vars.len() = {}", modified_vars.len());

        // Save initial values of variables that will be modified
        let mut var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                // Get the type from the locals table
                if let Some(func) = self.builder.current_function() {
                    if let Some(local) = func.locals.get(&reg) {
                        debug!("var {:?} has initial value {:?}", symbol_id, reg);
                        var_initial_values.insert(*symbol_id, (reg, local.ty.clone()));
                    }
                }
            } else {
                debug!("var {:?} NOT in symbol_map (new in branch)", symbol_id);
            }
        }

        debug!("var_initial_values.len() = {}", var_initial_values.len());

        // Evaluate condition
        if let Some(cond_reg) = self.lower_expression(condition) {
            // Branch-phi for effectful Call results — same fix as in
            // lower_conditional_typed. See that fn for the rationale.
            let cond_eval_block = match self.builder.current_block() {
                Some(b) => b,
                None => return,
            };
            let effectful_call_result_regs: BTreeSet<IrId> = {
                let mut regs = BTreeSet::new();
                if let Some(func) = self.builder.current_function() {
                    if let Some(block) = func.cfg.blocks.get(&cond_eval_block) {
                        for inst in &block.instructions {
                            match inst {
                                IrInstruction::CallDirect { dest: Some(d), .. }
                                | IrInstruction::CallIndirect { dest: Some(d), .. } => {
                                    regs.insert(*d);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                regs
            };

            // (sym, then_phi, else_phi). For if-stmt, else_block may equal
            // merge_block (no else branch case); in that case the value
            // flows through unchanged so we don't emit an else_phi.
            let mut branch_phi_rebind: Vec<(SymbolId, IrId, Option<IrId>)> = Vec::new();
            if !effectful_call_result_regs.is_empty() {
                let mut candidates: Vec<(SymbolId, IrId)> = self
                    .symbol_map
                    .iter()
                    .filter(|(_, r)| effectful_call_result_regs.contains(r))
                    .map(|(s, r)| (*s, *r))
                    .collect();
                candidates.sort_by_key(|(s, _)| *s);

                for (sym, reg) in candidates {
                    let var_ty = match self.builder.get_register_type(reg) {
                        Some(t) => t,
                        None => continue,
                    };
                    if !matches!(
                        var_ty,
                        IrType::I8
                            | IrType::I16
                            | IrType::I32
                            | IrType::I64
                            | IrType::U8
                            | IrType::U16
                            | IrType::U32
                            | IrType::U64
                            | IrType::F32
                            | IrType::F64
                            | IrType::Bool
                            | IrType::Ptr(_)
                    ) {
                        continue;
                    }
                    let then_phi = match self.builder.build_phi(then_block, var_ty.clone()) {
                        Some(p) => p,
                        None => continue,
                    };
                    self.builder
                        .add_phi_incoming(then_block, then_phi, cond_eval_block, reg);
                    let else_phi_opt = if else_block != merge_block {
                        let p = self.builder.build_phi(else_block, var_ty.clone());
                        if let Some(p) = p {
                            self.builder
                                .add_phi_incoming(else_block, p, cond_eval_block, reg);
                            Some(p)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(func) = self.builder.current_function_mut() {
                        if let Some(local) = func.locals.get(&reg).cloned() {
                            func.locals.insert(
                                then_phi,
                                crate::ir::IrLocal {
                                    name: format!("{}_then_branchphi", local.name),
                                    ty: var_ty.clone(),
                                    mutable: local.mutable,
                                    source_location: local.source_location,
                                    allocation: crate::ir::AllocationHint::Register,
                                },
                            );
                            if let Some(else_phi) = else_phi_opt {
                                func.locals.insert(
                                    else_phi,
                                    crate::ir::IrLocal {
                                        name: format!("{}_else_branchphi", local.name),
                                        ty: var_ty.clone(),
                                        mutable: local.mutable,
                                        source_location: local.source_location,
                                        allocation: crate::ir::AllocationHint::Register,
                                    },
                                );
                            }
                        }
                    }
                    // Register the symbol in var_initial_values so the
                    // existing merge-phi machinery wraps it back to a
                    // dominating SSA value post-merge.
                    var_initial_values
                        .entry(sym)
                        .or_insert_with(|| (reg, var_ty));
                    branch_phi_rebind.push((sym, then_phi, else_phi_opt));
                }
            }

            self.builder
                .build_cond_branch(cond_reg, then_block, else_block);

            // Lower then branch
            self.builder.switch_to_block(then_block);
            for (sym, then_phi, _) in &branch_phi_rebind {
                self.symbol_map.insert(*sym, *then_phi);
            }
            self.lower_block(then_branch);
            let then_end_block = if !self.is_terminated() {
                let current = self.builder.current_block();
                self.builder.build_branch(merge_block);
                current
            } else {
                None
            };

            // Save values after then branch
            let mut then_values: BTreeMap<SymbolId, IrId> = BTreeMap::new();
            if then_end_block.is_some() {
                // Only collect values for variables that existed BEFORE the if/else
                // Variables defined within the then branch should not be added
                for symbol_id in var_initial_values.keys() {
                    if let Some(&reg) = self.symbol_map.get(symbol_id) {
                        then_values.insert(*symbol_id, reg);
                    }
                }
            }

            // Lower else branch if present
            let mut else_values: BTreeMap<SymbolId, IrId> = BTreeMap::new();
            let else_end_block = if let Some(else_branch) = else_branch {
                self.builder.switch_to_block(else_block);
                // Rebind effectful-call-bound symbols to the else-branch phi
                // before lowering, so the else view sees the right SSA value.
                for (sym, _, else_phi_opt) in &branch_phi_rebind {
                    if let Some(else_phi) = else_phi_opt {
                        self.symbol_map.insert(*sym, *else_phi);
                    } else if let Some(&(orig_reg, _)) = var_initial_values.get(sym) {
                        self.symbol_map.insert(*sym, orig_reg);
                    }
                }
                self.lower_block(else_branch);
                if !self.is_terminated() {
                    let current = self.builder.current_block();
                    self.builder.build_branch(merge_block);

                    // Save values after else branch
                    // Only collect values for variables that existed BEFORE the if/else
                    // Variables defined within one branch should not leak into the other
                    for symbol_id in var_initial_values.keys() {
                        if let Some(&reg) = self.symbol_map.get(symbol_id) {
                            else_values.insert(*symbol_id, reg);
                        }
                    }
                    current
                } else {
                    None
                }
            } else {
                // If no else branch, the else path just falls through to merge
                // with the original values
                for (symbol_id, (initial_reg, _)) in &var_initial_values {
                    else_values.insert(*symbol_id, *initial_reg);
                }
                Some(entry_block)
            };

            // Continue in merge block and create phi nodes
            self.builder.switch_to_block(merge_block);

            // Create phi nodes for modified variables
            debug!("var_initial_values.len() = {}", var_initial_values.len());
            for (symbol_id, (initial_reg, var_type)) in &var_initial_values {
                debug!("Checking var {:?} for phi node", symbol_id);
                // Only create phi if at least one branch modified the variable
                let then_val = then_values.get(symbol_id).copied().unwrap_or(*initial_reg);
                let else_val = else_values.get(symbol_id).copied().unwrap_or(*initial_reg);

                debug!("  then_val {:?}, else_val {:?}", then_val, else_val);

                // If both branches lead to the same value, no phi needed
                if then_val == else_val {
                    debug!("  Skipping phi - same value");
                    continue;
                }

                // Only create phi if we have valid incoming blocks
                if then_end_block.is_none() && else_end_block.is_none() {
                    debug!("  Skipping phi - no valid incoming blocks");
                    continue;
                }

                debug!(
                    "Creating phi for symbol {:?}, then_val {:?}, else_val {:?}",
                    symbol_id, then_val, else_val
                );
                debug!(
                    "  then_end_block {:?}, else_end_block {:?}",
                    then_end_block, else_end_block
                );

                // Create phi node
                if let Some(phi_reg) = self.builder.build_phi(merge_block, var_type.clone()) {
                    debug!(
                        "  Created phi node {:?} in merge block {:?}",
                        phi_reg, merge_block
                    );

                    // Add incoming values from both branches
                    if let Some(then_blk) = then_end_block {
                        debug!(
                            "  Adding phi incoming from then block {:?}, value {:?}",
                            then_blk, then_val
                        );
                        self.builder
                            .add_phi_incoming(merge_block, phi_reg, then_blk, then_val);
                    }
                    if let Some(else_blk) = else_end_block {
                        debug!(
                            "  Adding phi incoming from else block {:?}, value {:?}",
                            else_blk, else_val
                        );
                        self.builder
                            .add_phi_incoming(merge_block, phi_reg, else_blk, else_val);
                    }

                    // Register the phi node as a local
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
                    self.symbol_map.insert(*symbol_id, phi_reg);
                }
            }
        }
    }

    pub(crate) fn lower_global(&mut self, symbol: SymbolId, global: &HirGlobal) {
        // Allocate a global ID
        let global_id = self.builder.module.alloc_global_id();

        // Convert initialization expression to IrValue if present
        // For now, we only support constant expressions
        let initializer = if let Some(init_expr) = &global.init {
            // Try to evaluate as constant expression
            match &init_expr.kind {
                HirExprKind::Literal(lit) => {
                    match lit {
                        HirLiteral::Bool(b) => Some(IrValue::Bool(*b)),
                        HirLiteral::Int(i) => Some(IrValue::I64(*i)),
                        HirLiteral::Float(f) => Some(IrValue::F64(*f)),
                        HirLiteral::String(s) => {
                            // String literals are added to string pool
                            // and referenced by their pool ID
                            let string_id = self.builder.module.string_pool.add(s.to_string());
                            // Store the string pool ID as an integer value
                            // The runtime will look up the actual string from the pool
                            Some(IrValue::I32(string_id as i32))
                        }
                        HirLiteral::Regex { .. } => {
                            // Regex needs special handling
                            None
                        }
                    }
                }
                _ => {
                    // Try constant folding for expressions like 1 << 16
                    if let Some(folded) = self.try_evaluate_constant_init(init_expr) {
                        Some(folded)
                    } else {
                        // Non-constant initialization - needs runtime evaluation
                        self.dynamic_globals.push((symbol, init_expr.clone()));
                        Some(IrValue::Undef)
                    }
                }
            }
        } else {
            // No initializer - use Undef
            Some(IrValue::Undef)
        };

        // Create the global variable
        // Note: Using placeholder name based on symbol ID since HirGlobal doesn't store name
        let global_ty = self.refine_global_type_from_initializer(IrType::Any, initializer.as_ref());

        let ir_global = IrGlobal {
            id: global_id,
            name: format!("global_{}", symbol.as_raw()),
            symbol_id: symbol,
            ty: global_ty, // TODO: Convert TypeId to IrType properly
            initializer,
            mutable: !global.is_const,
            linkage: Linkage::Internal, // TODO: Determine linkage from visibility
            alignment: None,
            source_location: IrSourceLocation::unknown(),
        };

        // Add to module
        self.builder.module.add_global(ir_global);

        // Store the mapping so we can look up globals by symbol ID later
        self.global_symbol_map.insert(symbol, global_id);
        debug!("[GLOBAL] Registered global {:?} -> {:?}", symbol, global_id);

        // TODO: For non-constant initializers, create an __init__ function
        // that runs at module load time to initialize the global
    }

    /// Lower __c__("C code {0} {1}", arg0, arg1) to TCC JIT compilation + execution
    pub(crate) fn lower_inline_code(
        &mut self,
        target: &str,
        code_expr: &HirExpr,
        args: &[HirExpr],
    ) -> Option<IrId> {
        if target != "__c__" {
            self.errors.push(LoweringError {
                message: format!(
                    "Unsupported inline code target: '{}' (only '__c__' is supported)",
                    target
                ),
                location: code_expr.source_location,
            });
            return None;
        }

        let tcc = self.ensure_tcc_runtime();
        let create_id = tcc.create;
        let compile_id = tcc.compile;
        let add_value_symbol_id = tcc.add_value_symbol;
        let free_value_id = tcc.free_value;
        let relocate_id = tcc.relocate;
        let get_symbol_id = tcc.get_symbol;
        let delete_id = tcc.delete;
        let call0_id = tcc.call0;
        let replace_id = tcc.string_replace;

        // Lower the code expression (string literal or concat like cdef() + "...")
        let code_reg = self.lower_expression(code_expr)?;

        // Lower all argument expressions
        let mut arg_regs = Vec::new();
        for arg in args {
            if let Some(reg) = self.lower_expression(arg) {
                arg_regs.push(reg);
            }
        }

        // Auto-inject module-local @:cstruct typedefs into TCC context
        // Use already-cached cstruct_layouts (populated by field accesses, constructors, cdef() calls)
        let mut cstruct_prefix = String::new();
        {
            let mut seen_typedefs = std::collections::BTreeSet::new();
            let layouts: Vec<_> = self.cstruct_layouts.values().cloned().collect();
            for layout in &layouts {
                for dep in &layout.dep_cdefs {
                    if seen_typedefs.insert(dep.clone()) {
                        cstruct_prefix.push_str(dep);
                    }
                }
                if seen_typedefs.insert(layout.own_cdef.clone()) {
                    cstruct_prefix.push_str(&layout.own_cdef);
                }
            }
        }

        // Build prefix: cstruct typedefs + extern declarations, do {N} -> __argN substitution
        let string_ptr_ty = IrType::Ptr(Box::new(IrType::String));
        let has_prefix = !cstruct_prefix.is_empty() || !arg_regs.is_empty();
        let final_code_reg = if !has_prefix {
            code_reg
        } else {
            // Replace {0} -> __arg0, {1} -> __arg1, etc. in the code string
            let mut current_code = code_reg;
            for i in 0..arg_regs.len() {
                let needle = self
                    .builder
                    .build_const(IrValue::String(format!("{{{}}}", i)))?;
                let replacement = self
                    .builder
                    .build_const(IrValue::String(format!("__arg{}", i)))?;
                current_code = self.builder.build_call_direct(
                    replace_id,
                    vec![current_code, needle, replacement],
                    string_ptr_ty.clone(),
                )?;
            }

            // Build compile-time prefix: cstruct typedefs + extern long declarations
            let mut prefix = cstruct_prefix;
            for i in 0..arg_regs.len() {
                prefix.push_str(&format!("extern long __arg{};\n", i));
            }
            let prefix_reg = self.builder.build_const(IrValue::String(prefix))?;

            // Concat: prefix + substituted_code via string_concat runtime
            let concat_func_id = self.register_stdlib_mir_forward_ref(
                "string_concat",
                vec![string_ptr_ty.clone(), string_ptr_ty.clone()],
                string_ptr_ty.clone(),
            );
            self.builder.build_call_direct(
                concat_func_id,
                vec![prefix_reg, current_code],
                string_ptr_ty.clone(),
            )?
        };

        // Collect @:frameworks, @:cInclude, @:cSource from module-local types and current function
        let add_framework_id = tcc.add_framework;
        let add_include_path_id = tcc.add_include_path;
        let add_file_id = tcc.add_file;
        let add_clib_id = tcc.add_clib;
        let mut framework_names: Vec<String> = Vec::new();
        let mut include_paths: Vec<String> = Vec::new();
        let mut source_files: Vec<String> = Vec::new();
        let mut clib_names: Vec<String> = Vec::new();
        {
            let mut seen_fw = std::collections::BTreeSet::new();
            let mut seen_inc = std::collections::BTreeSet::new();
            let mut seen_src = std::collections::BTreeSet::new();
            let mut seen_lib = std::collections::BTreeSet::new();

            // Collect metadata from all relevant symbols
            let mut sym_ids: Vec<SymbolId> = Vec::new();
            for (_type_id, type_decl) in self.current_hir_types {
                if let crate::ir::hir::HirTypeDecl::Class(c) = type_decl {
                    sym_ids.push(c.symbol_id);
                }
            }
            if let Some(func_sym_id) = self.current_function_symbol {
                sym_ids.push(func_sym_id);
            }

            for sid in &sym_ids {
                if let Some(sym) = self.symbol_table.get_symbol(*sid) {
                    if let Some(ref fws) = sym.frameworks {
                        for f in fws {
                            if let Some(n) = self.string_interner.get(*f) {
                                if seen_fw.insert(n.to_string()) {
                                    framework_names.push(n.to_string());
                                }
                            }
                        }
                    }
                    if let Some(ref incs) = sym.c_includes {
                        for i in incs {
                            if let Some(n) = self.string_interner.get(*i) {
                                if seen_inc.insert(n.to_string()) {
                                    include_paths.push(n.to_string());
                                }
                            }
                        }
                    }
                    if let Some(ref srcs) = sym.c_sources {
                        for s in srcs {
                            if let Some(n) = self.string_interner.get(*s) {
                                if seen_src.insert(n.to_string()) {
                                    source_files.push(n.to_string());
                                }
                            }
                        }
                    }
                    if let Some(ref libs) = sym.c_libs {
                        for l in libs {
                            if let Some(n) = self.string_interner.get(*l) {
                                if seen_lib.insert(n.to_string()) {
                                    clib_names.push(n.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }

        // 1. state = rayzor_tcc_create()
        let state = self
            .builder
            .build_call_direct(create_id, vec![], IrType::I64)?;

        // 1b. Auto-load @:cInclude paths
        for path in &include_paths {
            let path_reg = self.builder.build_const(IrValue::String(path.clone()))?;
            self.builder
                .build_call_direct(add_include_path_id, vec![state, path_reg], IrType::I32);
        }

        // 1c. Auto-load @:frameworks into TCC context
        for fw_name in &framework_names {
            let name_reg = self.builder.build_const(IrValue::String(fw_name.clone()))?;
            self.builder
                .build_call_direct(add_framework_id, vec![state, name_reg], IrType::I32);
        }

        // 1d. Auto-add @:cSource files (compiled before main code so symbols are available)
        for src_path in &source_files {
            let path_reg = self
                .builder
                .build_const(IrValue::String(src_path.clone()))?;
            self.builder
                .build_call_direct(add_file_id, vec![state, path_reg], IrType::I32);
        }

        // 1e. Auto-load @:clib libraries via pkg-config discovery
        for lib_name in &clib_names {
            let name_reg = self
                .builder
                .build_const(IrValue::String(lib_name.clone()))?;
            self.builder
                .build_call_direct(add_clib_id, vec![state, name_reg], IrType::I32);
        }

        // 2. rayzor_tcc_compile(state, code)
        self.builder
            .build_call_direct(compile_id, vec![state, final_code_reg], IrType::I32);

        // 3. Register arg values via add_value_symbol (heap-allocates each value so TCC can read it)
        //    Must be after compile, before relocate (resolves extern symbols at link time)
        let mut value_addrs = Vec::new();
        for (i, arg_reg) in arg_regs.iter().enumerate() {
            let name_reg = self
                .builder
                .build_const(IrValue::String(format!("__arg{}", i)))?;
            let addr = self.builder.build_call_direct(
                add_value_symbol_id,
                vec![state, name_reg, *arg_reg],
                IrType::I64,
            )?;
            value_addrs.push(addr);
        }

        // 4. rayzor_tcc_relocate(state)
        self.builder
            .build_call_direct(relocate_id, vec![state], IrType::I32);

        // 5. entry = rayzor_tcc_get_symbol(state, "__entry__")
        let entry_name = self
            .builder
            .build_const(IrValue::String("__entry__".to_string()))?;
        let entry_addr =
            self.builder
                .build_call_direct(get_symbol_id, vec![state, entry_name], IrType::I64)?;

        // 6. result = rayzor_tcc_call0(entry_addr)
        let result = self
            .builder
            .build_call_direct(call0_id, vec![entry_addr], IrType::I64);

        // 7. rayzor_tcc_delete(state)
        self.builder
            .build_call_direct(delete_id, vec![state], IrType::Void);

        // 8. Free heap-allocated value symbols
        for addr in &value_addrs {
            self.builder
                .build_call_direct(free_value_id, vec![*addr], IrType::Void);
        }

        result
    }

    /// Direct field access for typedef'd anonymous structs (like FileStat)
    /// All fields are 8 bytes for consistent boxing/sizing
    pub(crate) fn lower_typedef_field_access(
        &mut self,
        obj: IrId,
        _typedef_ty: TypeId,
        field_index: u32,
        field_ty: TypeId,
    ) -> Option<IrId> {
        // Create constant for field index
        let index_const = self.builder.build_const(IrValue::I32(field_index as i32))?;

        // Get the type of the field
        let field_ir_ty = self.convert_type(field_ty);

        // Use GetElementPtr to get pointer to the field
        // All typedef anonymous struct fields are 8 bytes
        let field_ptr = self
            .builder
            .build_gep(obj, vec![index_const], field_ir_ty.clone())?;

        // Load the value from the field pointer
        let field_value = self.builder.build_load(field_ptr, field_ir_ty.clone())?;

        // Register the type of the loaded value
        self.builder.set_register_type(field_value, field_ir_ty);

        Some(field_value)
    }

    pub(crate) fn lower_expression_inner(&mut self, expr: &HirExpr) -> Option<IrId> {
        // DEBUG: Check if this is Field expression being lowered
        if matches!(&expr.kind, HirExprKind::Field { .. }) {
            debug!("[lower_expression] START - Field expression");
        }

        // Set source location for debugging
        self.builder
            .set_source_location(self.convert_source_location(&expr.source_location));

        let result = match &expr.kind {
            HirExprKind::Literal(lit) => self.lower_literal(lit, expr.ty),

            HirExprKind::Variable { .. } => self.lower_variable_expr(expr),
            HirExprKind::Field { .. } => self.lower_field_expr(expr),
            HirExprKind::Index { .. } => self.lower_index_expr(expr),
            HirExprKind::Call { .. } => self.lower_call(expr),
            HirExprKind::New { .. } => self.lower_new(expr),
            HirExprKind::Unary { .. } => self.lower_unary(expr),
            HirExprKind::Binary { .. } => self.lower_binary(expr),
            HirExprKind::Cast { .. } => self.lower_cast(expr),
            HirExprKind::TypeCheck { .. } => self.lower_type_check(expr),
            HirExprKind::If {
                condition,
                then_expr,
                else_expr,
            } => self.lower_conditional_typed(condition, then_expr, else_expr, Some(expr.ty)),

            HirExprKind::Block(block) => self.lower_block_expr(block),

            HirExprKind::Lambda {
                params,
                body,
                captures,
            } => {
                debug!(
                    "Lowering lambda with {} params, {} captures",
                    params.len(),
                    captures.len()
                );
                self.lower_lambda(params, body, captures, expr.ty)
            }

            HirExprKind::MethodReference {
                receiver,
                method_symbol,
            } => self.lower_method_reference(receiver, *method_symbol),

            HirExprKind::Array { elements } => self.lower_array_literal(elements, expr.ty),

            HirExprKind::Map { entries } => self.lower_map_literal(entries),

            HirExprKind::ObjectLiteral { fields } => self.lower_object_literal(fields, expr.ty),

            HirExprKind::ArrayComprehension { .. } => {
                // Array comprehensions are desugared to loops
                self.add_error(
                    "Array comprehensions not yet implemented in MIR",
                    expr.source_location,
                );
                None
            }

            HirExprKind::StringInterpolation { parts } => self.lower_string_interpolation(parts),

            HirExprKind::This => {
                // 'this' is typically passed as first parameter
                self.symbol_map.get(&SymbolId::from_raw(0)).copied()
            }

            HirExprKind::Super => {
                // 'super' should only appear in constructor super calls, which are handled
                // specially in lower_constructor_body. If we reach here, it's likely being
                // used incorrectly (e.g., super.method() which isn't supported yet)
                // eprintln!("WARNING: HirExprKind::Super encountered in expression lowering");
                // eprintln!("  This might be super.field or super.method() which isn't implemented yet");
                // For now, treat it like 'this' (same object, but calling parent methods)
                self.symbol_map.get(&SymbolId::from_raw(0)).copied()
            }

            HirExprKind::Null => self.builder.build_null(),

            HirExprKind::Untyped(inner) => {
                // Untyped expressions bypass type checking
                self.lower_expression(inner)
            }

            HirExprKind::InlineCode { target, code, args } => {
                // Platform-specific inline code (__c__, __js__, etc.)
                self.lower_inline_code(target, code, args)
            }

            HirExprKind::TryCatch { .. } => self.lower_try_catch_expr(expr),
            _ => {
                self.add_error("Unsupported expression type in MIR", expr.source_location);
                None
            }
        };

        // debug!("lower_expression result: {:?}", result);
        result
    }
}
