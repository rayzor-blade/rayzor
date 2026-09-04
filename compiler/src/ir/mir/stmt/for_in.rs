//! `for`-in, one form per iterable: integer range, array, map key/value,
//! and the general iterator protocol.

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
    pub(crate) fn lower_for_in_loop(
        &mut self,
        pattern: &HirPattern,
        iter_expr: &HirExpr,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) {
        debug!("[for-in]: ENTERED lower_for_in_loop!");
        debug!("[for-in]: pattern={:?}", pattern);
        debug!("[for-in]: iter_expr.ty={:?}", iter_expr.ty);

        // Check for range expressions (0...5) — desugar to counter loop
        if let HirExprKind::Binary {
            op: HirBinaryOp::Range,
            lhs,
            rhs,
        } = &iter_expr.kind
        {
            self.lower_for_in_range(pattern, lhs, rhs, body, label);
            return;
        }

        // Check the iterable expression's type to determine iteration strategy
        let iter_type_kind = {
            let type_table = self.type_table;
            type_table.get(iter_expr.ty).map(|t| t.kind.clone())
        };

        // For Map types (including IntMap/StringMap extern classes), convert keys
        // to a HaxeArray and iterate over that
        let map_kv_types: Option<(TypeId, TypeId)> = match &iter_type_kind {
            Some(crate::tast::TypeKind::Map {
                key_type,
                value_type,
            }) => Some((*key_type, *value_type)),
            Some(crate::tast::TypeKind::Class {
                symbol_id,
                type_args,
            }) => {
                let class_name = self
                    .symbol_table
                    .get_symbol(*symbol_id)
                    .and_then(|sym| self.string_interner.get(sym.name));
                let type_table = self.type_table;
                match class_name {
                    Some("IntMap") => {
                        let value_type = type_args
                            .first()
                            .copied()
                            .unwrap_or_else(|| type_table.dynamic_type());
                        Some((type_table.int_type(), value_type))
                    }
                    Some("StringMap") => {
                        let value_type = type_args
                            .first()
                            .copied()
                            .unwrap_or_else(|| type_table.dynamic_type());
                        Some((type_table.string_type(), value_type))
                    }
                    Some("ObjectMap") => {
                        // ObjectMap<K, V> has two type args
                        let key_type = type_args
                            .first()
                            .copied()
                            .unwrap_or_else(|| type_table.dynamic_type());
                        let value_type = type_args
                            .get(1)
                            .copied()
                            .unwrap_or_else(|| type_table.dynamic_type());
                        Some((key_type, value_type))
                    }
                    _ => None,
                }
            }
            _ => None,
        };
        if let Some((key_type, value_type)) = map_kv_types {
            let key_type_kind = {
                let type_table = self.type_table;
                type_table.get(key_type).map(|t| t.kind.clone())
            };
            let is_int_key = matches!(
                key_type_kind,
                Some(crate::tast::TypeKind::Int) | Some(crate::tast::TypeKind::Bool)
            );
            let is_string_key = matches!(key_type_kind, Some(crate::tast::TypeKind::String));

            let Some(map_ptr) = self.lower_expression(iter_expr) else {
                return;
            };

            let ptr_void = IrType::Ptr(Box::new(IrType::Void));

            // key=>value iteration (Tuple pattern): iterate the keys array and look
            // up each value via map.get(key).
            if let HirPattern::Tuple(sub_patterns) = pattern {
                if sub_patterns.len() == 2 {
                    let keys_fn_name = if is_int_key {
                        "haxe_intmap_keys_to_array"
                    } else if is_string_key {
                        "haxe_stringmap_keys_to_array"
                    } else {
                        "haxe_objectmap_keys_to_array"
                    };
                    let keys_fn = self.get_or_register_extern_function(
                        keys_fn_name,
                        vec![ptr_void.clone()],
                        ptr_void.clone(),
                    );
                    let Some(keys_array) =
                        self.builder
                            .build_call_direct(keys_fn, vec![map_ptr], ptr_void.clone())
                    else {
                        return;
                    };
                    self.lower_for_in_map_kv(
                        &sub_patterns[0],
                        &sub_patterns[1],
                        map_ptr,
                        keys_array,
                        key_type,
                        value_type,
                        is_int_key,
                        is_string_key,
                        body,
                        label,
                    );
                    return;
                }
            }

            let values_fn_name = if is_int_key {
                "haxe_intmap_values_to_array"
            } else if is_string_key {
                "haxe_stringmap_values_to_array"
            } else {
                "haxe_objectmap_values_to_array"
            };
            let values_fn = self.get_or_register_extern_function(
                values_fn_name,
                vec![ptr_void.clone()],
                ptr_void.clone(),
            );
            let Some(values_array) =
                self.builder
                    .build_call_direct(values_fn, vec![map_ptr], ptr_void)
            else {
                return;
            };
            self.lower_for_in_over_array(pattern, values_array, value_type, body, label);
            return;
        }

        // A receiver typed `Iterable<T>`/`Iterator<T>` is structural and names no
        // class to call, so it carries an iteration handle built where it crossed
        // into that type. This precedes the class path below because such a
        // receiver resolves to a symbol whose methods are declarations only.
        if self.try_lower_for_in_iter_handle(pattern, iter_expr, body, label) {
            return;
        }

        // For class/interface types with hasNext()/next() iterator protocol,
        // desugar to a while loop calling those methods directly.
        // Dynamic is included because arr.iterator() returns Dynamic-typed iterators
        // that have hasNext/next via stdlib mappings.
        if let Some(ref kind) = iter_type_kind {
            let is_iterator_class = matches!(
                kind,
                crate::tast::TypeKind::Class { .. }
                    | crate::tast::TypeKind::Interface { .. }
                    | crate::tast::TypeKind::TypeAlias { .. }
                    | crate::tast::TypeKind::Placeholder { .. }
                    | crate::tast::TypeKind::Dynamic
            );
            if is_iterator_class {
                self.lower_for_in_iterator_protocol(pattern, iter_expr, body, label);
                return;
            }
        }

        // Arrays desugar to index-based iteration:
        // `var _i = 0; while (_i < len) { var x = arr[_i]; body; _i++; }`
        debug!("[for-in]: lowering collection expression...");
        let Some(collection) = self.lower_expression(iter_expr) else {
            debug!("[for-in]: FAILED to lower collection expression!");
            return;
        };

        let collection_type = self.builder.get_register_type(collection);
        debug!(
            "[for-in]: collection reg={:?}, type={:?}",
            collection, collection_type
        );

        let elem_type_id = self
            .get_array_element_type(iter_expr.ty)
            .unwrap_or(iter_expr.ty);
        debug!(
            "[for-in]: array_type={:?}, elem_type={:?}",
            iter_expr.ty, elem_type_id
        );

        self.lower_for_in_over_array(pattern, collection, elem_type_id, body, label);
    }

    /// Lower a range-based for-in loop: `for (i in start...end) { body }`
    /// Desugars to: `var i = start; while (i < end) { body; i++; }`
    pub(crate) fn lower_for_in_range(
        &mut self,
        pattern: &HirPattern,
        start_expr: &HirExpr,
        end_expr: &HirExpr,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) {
        debug!("[for-in-range]: lowering range-based for-in loop");

        let Some(start_val) = self.lower_expression(start_expr) else {
            return;
        };
        let Some(end_val) = self.lower_expression(end_expr) else {
            return;
        };
        let start_i64 = {
            let start_ty = self.builder.get_register_type(start_val);
            match start_ty {
                Some(IrType::I64) => start_val,
                Some(IrType::I32) => self
                    .builder
                    .build_cast(start_val, IrType::I32, IrType::I64)
                    .unwrap_or(start_val),
                Some(IrType::Bool) => self
                    .builder
                    .build_cast(start_val, IrType::Bool, IrType::I64)
                    .unwrap_or(start_val),
                Some(other) => self
                    .builder
                    .build_cast(start_val, other, IrType::I64)
                    .unwrap_or(start_val),
                None => start_val,
            }
        };
        let end_i64 = {
            let end_ty = self.builder.get_register_type(end_val);
            match end_ty {
                Some(IrType::I64) => end_val,
                Some(IrType::I32) => self
                    .builder
                    .build_cast(end_val, IrType::I32, IrType::I64)
                    .unwrap_or(end_val),
                Some(IrType::Bool) => self
                    .builder
                    .build_cast(end_val, IrType::Bool, IrType::I64)
                    .unwrap_or(end_val),
                Some(other) => self
                    .builder
                    .build_cast(end_val, other, IrType::I64)
                    .unwrap_or(end_val),
                None => end_val,
            }
        };

        // Save the entry block for loop-carried phi incoming edges.
        let entry_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            return;
        };

        // Collect variables referenced in the body that need loop-carried SSA values.
        let mut referenced_vars = std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_block(body, &mut referenced_vars);

        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| {
                let in_map = self.symbol_map.contains_key(sym);
                // Parameters are excluded by symbol kind, not by register.
                let is_param = self.is_parameter_symbol(sym);
                in_map && !is_param
            })
            .collect();

        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let reg_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg).map(|l| l.ty.clone()))
                    .or_else(|| self.builder.get_register_type(reg))
                    .unwrap_or(IrType::I64);
                loop_var_initial_values.insert(*symbol_id, (reg, reg_ty));
            }
        }

        let Some(counter_ptr) = self.builder.build_alloc(IrType::I64, None) else {
            return;
        };
        self.builder.build_store(counter_ptr, start_i64);

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

        // Condition block: create loop-carried phis first.
        self.builder.switch_to_block(loop_cond_block);
        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            if let Some(phi_reg) = self.builder.build_phi(loop_cond_block, var_type.clone()) {
                self.builder
                    .add_phi_incoming(loop_cond_block, phi_reg, entry_block, *initial_reg);

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

                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);

                if self.owned_heap_values.contains_key(symbol_id) {
                    self.owned_heap_values.insert(*symbol_id, phi_reg);
                }
            }
        }

        // Build exit phi nodes up-front so break statements can target them.
        let mut exit_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, loop_phi_reg) in &phi_nodes {
            if let Some((_, var_type)) = loop_var_initial_values.get(symbol_id) {
                let exit_param_reg = self.builder.alloc_reg().unwrap();

                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(exit_block_data) = func.cfg.get_block_mut(loop_exit_block) {
                        let exit_phi = crate::ir::IrPhiNode {
                            dest: exit_param_reg,
                            incoming: vec![(loop_cond_block, *loop_phi_reg)],
                            ty: var_type.clone(),
                        };
                        exit_block_data.add_phi(exit_phi);

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

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            continue_block: loop_cond_block,
            break_block: loop_exit_block,
            label: label.cloned(),
            exit_phi_nodes: exit_phi_nodes.clone(),
            continue_phi_nodes: BTreeMap::new(),
        });

        // Condition block: counter < end
        let Some(current_val) = self.builder.build_load(counter_ptr, IrType::I64) else {
            self.loop_stack.pop();
            return;
        };
        let Some(cmp_result) =
            self.builder
                .build_cmp(crate::ir::instructions::CompareOp::Lt, current_val, end_i64)
        else {
            self.loop_stack.pop();
            return;
        };
        self.builder
            .build_cond_branch(cmp_result, loop_body_block, loop_exit_block);

        // Body block: reload the counter and bind it to the pattern variable.
        self.builder.switch_to_block(loop_body_block);

        let Some(loop_val) = self.builder.build_load(counter_ptr, IrType::I64) else {
            self.loop_stack.pop();
            return;
        };
        match pattern {
            HirPattern::Variable { symbol, .. } => {
                self.symbol_map.insert(*symbol, loop_val);
            }
            _ => {}
        }

        // Track loop-carried symbols so the body's exit_drop_scope does not free
        // values that escape via the exit phi.
        self.loop_carried_symbols
            .push(phi_nodes.keys().copied().collect());
        self.enter_drop_scope();
        self.lower_block(body);

        // Capture the block we ended up in after lowering the body.
        let body_end_block = self.builder.current_block().unwrap_or(loop_body_block);

        // Add loop back-edge values for loop-carried variables.
        for (symbol_id, phi_reg) in &phi_nodes {
            let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                updated_reg
            } else {
                *phi_reg
            };
            self.builder.add_phi_incoming(
                loop_cond_block,
                *phi_reg,
                body_end_block,
                back_edge_value,
            );
        }

        if !self.is_terminated() {
            self.exit_drop_scope();
            self.loop_carried_symbols.pop();
            let Some(idx_to_inc) = self.builder.build_load(counter_ptr, IrType::I64) else {
                self.loop_stack.pop();
                return;
            };
            let Some(one) = self.builder.build_const(IrValue::I64(1)) else {
                self.loop_stack.pop();
                return;
            };
            let Some(next_val) =
                self.builder
                    .build_binop(crate::ir::instructions::BinaryOp::Add, idx_to_inc, one)
            else {
                self.loop_stack.pop();
                return;
            };
            self.builder.build_store(counter_ptr, next_val);
            self.builder.build_branch(loop_cond_block);
        } else {
            self.loop_carried_symbols.pop();
        }

        self.loop_stack.pop();
        self.builder.switch_to_block(loop_exit_block);

        // Update symbols to exit phi values after the loop.
        for (symbol_id, exit_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_reg);
            if self.owned_heap_values.contains_key(symbol_id) {
                self.owned_heap_values.insert(*symbol_id, *exit_reg);
            }
        }
    }

    /// Lower for-in iteration using the hasNext()/next() iterator protocol.
    /// Synthesizes HIR for `while (obj.hasNext()) { var x = obj.next(); body; }`
    /// and delegates to lower_while_loop which handles phi nodes for mutable variables.
    pub(crate) fn lower_for_in_iterator_protocol(
        &mut self,
        pattern: &HirPattern,
        iter_expr: &HirExpr,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) {
        let Some(obj_reg) = self.lower_expression(iter_expr) else {
            return;
        };

        // Resolve the class symbol from the iterable's type
        let class_sym = {
            let type_table = self.type_table;
            let mut tid = iter_expr.ty;
            let mut result: Option<SymbolId> = None;
            let mut visited = BTreeSet::new();
            loop {
                if !visited.insert(tid) {
                    break;
                }
                if let Some(ty) = type_table.get(tid) {
                    match &ty.kind {
                        crate::tast::TypeKind::Class { symbol_id, .. } => {
                            result = Some(*symbol_id);
                            break;
                        }
                        crate::tast::TypeKind::TypeAlias {
                            symbol_id,
                            target_type,
                            ..
                        } => {
                            result = Some(*symbol_id);
                            tid = *target_type;
                        }
                        crate::tast::TypeKind::GenericInstance { base_type, .. } => {
                            tid = *base_type;
                        }
                        crate::tast::TypeKind::Placeholder { .. } => break,
                        _ => break,
                    }
                } else {
                    break;
                }
            }
            result
        };

        // For Dynamic-typed iterators (e.g., from arr.iterator()), class_sym is None.
        // Use register class hints to find the iterator class and resolve via stdlib mappings.
        if class_sym.is_none() {
            if let Some(class_hint) = self
                .register_class_hints
                .get(&obj_reg)
                .and_then(|hint| self.stdlib_mapping.class_key(hint))
            {
                let hn_mapping = self.stdlib_mapping.find_by_name(class_hint, "hasNext");
                let n_mapping = self.stdlib_mapping.find_by_name(class_hint, "next");
                if let (Some((_hn_sig, hn_rt)), Some((_n_sig, n_rt))) = (hn_mapping, n_mapping) {
                    let hn_name = hn_rt.runtime_name.to_string();
                    let n_name = n_rt.runtime_name.to_string();
                    let hn_is_mir = hn_rt.is_mir_wrapper;
                    let n_is_mir = n_rt.is_mir_wrapper;
                    let ptr_void = IrType::Ptr(Box::new(IrType::Void));

                    let has_next_fn = if hn_is_mir {
                        self.register_stdlib_mir_forward_ref(
                            &hn_name,
                            vec![ptr_void.clone()],
                            IrType::I32,
                        )
                    } else {
                        self.get_or_register_extern_function(
                            &hn_name,
                            vec![ptr_void.clone()],
                            IrType::I32,
                        )
                    };
                    let next_fn = if n_is_mir {
                        self.register_stdlib_mir_forward_ref(
                            &n_name,
                            vec![ptr_void.clone()],
                            IrType::I64,
                        )
                    } else {
                        self.get_or_register_extern_function(
                            &n_name,
                            vec![ptr_void.clone()],
                            IrType::I64,
                        )
                    };

                    self.emit_iterator_while_loop(
                        pattern,
                        obj_reg,
                        has_next_fn,
                        next_fn,
                        body,
                        label,
                    );
                    return;
                }
            }
            return;
        }
        let class_sym = class_sym.unwrap();

        let has_next_name = self.string_interner.intern("hasNext");
        let next_name = self.string_interner.intern("next");

        let has_next_sym = self
            .class_method_symbols
            .get(&(class_sym, has_next_name))
            .copied();
        let next_sym = self
            .class_method_symbols
            .get(&(class_sym, next_name))
            .copied();

        // If the class doesn't have hasNext/next directly, try several fallback paths:
        // 1. Check stdlib runtime mappings (for runtime-backed iterators like ArrayIterator)
        // 2. Check for iterator() method on the class (Haxe iterator protocol)
        // 3. Check for keyValueIterator() method on the class
        let (obj_reg, has_next_fn, next_fn) = if let (Some(hn_sym), Some(n_sym)) =
            (has_next_sym, next_sym)
        {
            match (self.get_function_id(&hn_sym), self.get_function_id(&n_sym)) {
                (Some(hn), Some(n)) => (obj_reg, hn, n),
                _ => return,
            }
        } else {
            // Fallback 1: Check stdlib runtime mappings by class name
            // This handles runtime-backed iterator classes (ArrayIterator, ArrayKeyValueIterator)
            // that don't have compiled hasNext/next in class_method_symbols.
            let class_name_str = self
                .symbol_table
                .get_symbol(class_sym)
                .and_then(|sym| self.string_interner.get(sym.name))
                .map(|s| s.to_string());

            // Extract runtime names first to avoid borrow conflict with get_or_register_extern_function
            let stdlib_names = class_name_str
                .as_deref()
                .and_then(|name| self.stdlib_mapping.class_key(name))
                .and_then(|class_name| {
                    let hn_mapping = self.stdlib_mapping.find_by_name(class_name, "hasNext");
                    let n_mapping = self.stdlib_mapping.find_by_name(class_name, "next");
                    if let (Some((_hn_sig, hn_rt)), Some((_n_sig, n_rt))) = (hn_mapping, n_mapping)
                    {
                        Some((
                            hn_rt.runtime_name.to_string(),
                            n_rt.runtime_name.to_string(),
                        ))
                    } else {
                        None
                    }
                });
            let stdlib_result = stdlib_names.map(|(hn_name, n_name)| {
                let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                let hn_fn = self.get_or_register_extern_function(
                    &hn_name,
                    vec![ptr_void.clone()],
                    IrType::I32,
                );
                let n_fn = self.get_or_register_extern_function(
                    &n_name,
                    vec![ptr_void.clone()],
                    IrType::I64,
                );
                (obj_reg, hn_fn, n_fn)
            });

            if let Some(result) = stdlib_result {
                result
            } else {
                // Fallback 2: Look for iterator() method on the class via class_method_symbols
                let iterator_name = self.string_interner.intern("iterator");
                let kv_iterator_name = self.string_interner.intern("keyValueIterator");
                let iterator_sym = self
                    .class_method_symbols
                    .get(&(class_sym, iterator_name))
                    .copied();

                // Also check for iterator/keyValueIterator via stdlib mapping (e.g., Array.iterator)
                // Extract names first to avoid borrow conflicts
                let stdlib_iter_names = if iterator_sym.is_none() {
                    class_name_str
                        .as_deref()
                        .and_then(|name| self.stdlib_mapping.class_key(name))
                        .and_then(|class_name| {
                            let iter_mapping = self
                                .stdlib_mapping
                                .find_by_name(class_name, "iterator")
                                .or_else(|| {
                                    self.stdlib_mapping
                                        .find_by_name(class_name, "keyValueIterator")
                                });
                            if let Some((_sig, rt)) = iter_mapping {
                                let iter_rt_name = rt.runtime_name.to_string();
                                let is_kv = iter_rt_name.contains("kv_iterator");
                                let iter_class_name = if is_kv {
                                    "haxe.iterators.ArrayKeyValueIterator"
                                } else {
                                    "haxe.iterators.ArrayIterator"
                                };
                                let iter_key = self.stdlib_mapping.key(iter_class_name);
                                let hn_mapping =
                                    self.stdlib_mapping.find_by_name(iter_key, "hasNext");
                                let n_mapping = self.stdlib_mapping.find_by_name(iter_key, "next");
                                if let (Some((_hn_sig, hn_rt)), Some((_n_sig, n_rt))) =
                                    (hn_mapping, n_mapping)
                                {
                                    Some((
                                        iter_rt_name,
                                        hn_rt.runtime_name.to_string(),
                                        n_rt.runtime_name.to_string(),
                                    ))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                } else {
                    None
                };
                let stdlib_iter_result =
                    stdlib_iter_names.and_then(|(iter_name, hn_name, n_name)| {
                        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                        let iter_fn = self.get_or_register_extern_function(
                            &iter_name,
                            vec![ptr_void.clone()],
                            ptr_void.clone(),
                        );
                        let iter_obj = self.builder.build_call_direct(
                            iter_fn,
                            vec![obj_reg],
                            ptr_void.clone(),
                        )?;
                        let hn_fn = self.get_or_register_extern_function(
                            &hn_name,
                            vec![ptr_void.clone()],
                            IrType::I32,
                        );
                        let n_fn = self.get_or_register_extern_function(
                            &n_name,
                            vec![ptr_void],
                            IrType::I64,
                        );
                        Some((iter_obj, hn_fn, n_fn))
                    });

                if let Some(result) = stdlib_iter_result {
                    result
                } else if let Some(iter_sym) = iterator_sym {
                    // Compiled iterator() method path.
                    let Some(iter_fn) = self.get_function_id(&iter_sym) else {
                        return;
                    };

                    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                    let Some(iter_obj) =
                        self.builder
                            .build_call_direct(iter_fn, vec![obj_reg], ptr_void)
                    else {
                        return;
                    };

                    // Find the iterator class from the return type
                    let iter_class_sym = {
                        let sym = self.symbol_table.get_symbol(iter_sym);
                        sym.and_then(|s| {
                            let tt = self.type_table;
                            let ret_ty = tt.get(s.type_id)?;
                            if let crate::tast::TypeKind::Function { return_type, .. } =
                                &ret_ty.kind
                            {
                                let ret = tt.get(*return_type)?;
                                if let crate::tast::TypeKind::Class { symbol_id, .. } = &ret.kind {
                                    return Some(*symbol_id);
                                }
                            }
                            None
                        })
                    };

                    // Fallback: search class_method_symbols for any class with hasNext+next
                    let iter_class_sym = iter_class_sym.or_else(|| {
                        for ((cls, method_name), _) in self.class_method_symbols.iter() {
                            if *method_name == has_next_name && *cls != class_sym {
                                if self.class_method_symbols.contains_key(&(*cls, next_name)) {
                                    return Some(*cls);
                                }
                            }
                        }
                        None
                    });

                    let Some(iter_cls) = iter_class_sym else {
                        return;
                    };
                    let hn_sym = self
                        .class_method_symbols
                        .get(&(iter_cls, has_next_name))
                        .copied();
                    let n_sym = self
                        .class_method_symbols
                        .get(&(iter_cls, next_name))
                        .copied();
                    match (hn_sym, n_sym) {
                        (Some(hn), Some(n)) => {
                            match (self.get_function_id(&hn), self.get_function_id(&n)) {
                                (Some(hn_fn), Some(n_fn)) => (iter_obj, hn_fn, n_fn),
                                _ => return,
                            }
                        }
                        _ => return,
                    }
                } else {
                    // Fallback 3: Check for keyValueIterator() method (compiled user class)
                    let kv_iter_sym = self
                        .class_method_symbols
                        .get(&(class_sym, kv_iterator_name))
                        .copied();
                    let Some(kv_sym) = kv_iter_sym else {
                        return;
                    };
                    let Some(kv_fn) = self.get_function_id(&kv_sym) else {
                        return;
                    };

                    let ptr_void = IrType::Ptr(Box::new(IrType::Void));
                    let Some(kv_obj) =
                        self.builder
                            .build_call_direct(kv_fn, vec![obj_reg], ptr_void)
                    else {
                        return;
                    };

                    // Find hasNext/next on the returned KV iterator class
                    let kv_class_sym = {
                        let sym = self.symbol_table.get_symbol(kv_sym);
                        sym.and_then(|s| {
                            let tt = self.type_table;
                            let ret_ty = tt.get(s.type_id)?;
                            if let crate::tast::TypeKind::Function { return_type, .. } =
                                &ret_ty.kind
                            {
                                let ret = tt.get(*return_type)?;
                                if let crate::tast::TypeKind::Class { symbol_id, .. } = &ret.kind {
                                    return Some(*symbol_id);
                                }
                            }
                            None
                        })
                    };

                    let kv_class_sym = kv_class_sym.or_else(|| {
                        for ((cls, method_name), _) in self.class_method_symbols.iter() {
                            if *method_name == has_next_name && *cls != class_sym {
                                if self.class_method_symbols.contains_key(&(*cls, next_name)) {
                                    return Some(*cls);
                                }
                            }
                        }
                        None
                    });

                    let Some(kv_cls) = kv_class_sym else {
                        return;
                    };
                    let hn_sym = self
                        .class_method_symbols
                        .get(&(kv_cls, has_next_name))
                        .copied();
                    let n_sym = self.class_method_symbols.get(&(kv_cls, next_name)).copied();
                    match (hn_sym, n_sym) {
                        (Some(hn), Some(n)) => {
                            match (self.get_function_id(&hn), self.get_function_id(&n)) {
                                (Some(hn_fn), Some(n_fn)) => (kv_obj, hn_fn, n_fn),
                                _ => return,
                            }
                        }
                        _ => return,
                    }
                }
            }
        };

        self.emit_iterator_while_loop(pattern, obj_reg, has_next_fn, next_fn, body, label);
    }

    /// Helper: iterate over map keys and look up values for `for (key => value in map)`.
    /// Iterates the keys array, calls map.get(key) for each, and binds both key and value.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lower_for_in_map_kv(
        &mut self,
        key_pattern: &HirPattern,
        value_pattern: &HirPattern,
        map_ptr: IrId,
        keys_array: IrId,
        key_type_id: TypeId,
        value_type_id: TypeId,
        is_int_key: bool,
        is_string_key: bool,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) {
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));

        // Read keys array length
        let Some(offset_8) = self.builder.build_const(IrValue::I64(8)) else {
            return;
        };
        let Some(len_ptr) =
            self.builder
                .build_binop(crate::ir::instructions::BinaryOp::Add, keys_array, offset_8)
        else {
            return;
        };
        let Some(array_len) = self.builder.build_load(len_ptr, IrType::I64) else {
            return;
        };

        let Some(zero) = self.builder.build_const(IrValue::I64(0)) else {
            return;
        };
        let Some(index_ptr) = self.builder.build_alloc(IrType::I64, None) else {
            return;
        };
        self.builder.build_store(index_ptr, zero);

        // Save entry block for phi node incoming edges
        let entry_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            return;
        };

        // Collect referenced variables in body for phi node creation.
        let mut referenced_vars = std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_block(body, &mut referenced_vars);
        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| {
                let in_map = self.symbol_map.contains_key(sym);
                // Parameters are excluded by symbol kind, not by register.
                let is_param = self.is_parameter_symbol(sym);
                in_map && !is_param
            })
            .collect();

        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let reg_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg).map(|l| l.ty.clone()))
                    .or_else(|| self.builder.get_register_type(reg))
                    .unwrap_or(IrType::I64);
                loop_var_initial_values.insert(*symbol_id, (reg, reg_ty));
            }
        }

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

        // Condition block: create phi nodes for loop-carried variables
        self.builder.switch_to_block(loop_cond_block);

        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            if let Some(phi_reg) = self.builder.build_phi(loop_cond_block, var_type.clone()) {
                self.builder
                    .add_phi_incoming(loop_cond_block, phi_reg, entry_block, *initial_reg);
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
                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);
            }
        }

        // Create exit phi nodes so post-loop code sees final values
        let mut exit_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, loop_phi_reg) in &phi_nodes {
            if let Some((_, var_type)) = loop_var_initial_values.get(symbol_id) {
                let Some(exit_param_reg) = self.builder.alloc_reg() else {
                    continue;
                };
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(exit_block_data) = func.cfg.get_block_mut(loop_exit_block) {
                        let exit_phi = crate::ir::IrPhiNode {
                            dest: exit_param_reg,
                            incoming: vec![(loop_cond_block, *loop_phi_reg)],
                            ty: var_type.clone(),
                        };
                        exit_block_data.add_phi(exit_phi);
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

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            continue_block: loop_cond_block,
            break_block: loop_exit_block,
            label: label.cloned(),
            exit_phi_nodes: exit_phi_nodes.clone(),
            continue_phi_nodes: BTreeMap::new(),
        });

        // Condition: index < length
        let Some(current_index) = self.builder.build_load(index_ptr, IrType::I64) else {
            self.loop_stack.pop();
            return;
        };
        let Some(cmp_result) = self.builder.build_cmp(
            crate::ir::instructions::CompareOp::Lt,
            current_index,
            array_len,
        ) else {
            self.loop_stack.pop();
            return;
        };
        self.builder
            .build_cond_branch(cmp_result, loop_body_block, loop_exit_block);

        // Body block: get key, look up value, bind both, execute body, increment
        self.builder.switch_to_block(loop_body_block);
        let Some(idx_for_access) = self.builder.build_load(index_ptr, IrType::I64) else {
            self.loop_stack.pop();
            return;
        };

        let Some(key_value) = self.lower_index_access(keys_array, idx_for_access, key_type_id)
        else {
            self.loop_stack.pop();
            return;
        };

        if let HirPattern::Variable { symbol, .. } = key_pattern {
            self.symbol_map.insert(*symbol, key_value);
        }

        // Look up value: map.get(key)
        let get_fn_name = if is_int_key {
            "haxe_intmap_get"
        } else if is_string_key {
            "haxe_stringmap_get"
        } else {
            "haxe_objectmap_get"
        };
        let key_ir_type = if is_int_key {
            IrType::I64
        } else {
            // Both StringMap and ObjectMap keys are pointers
            IrType::Ptr(Box::new(IrType::U8))
        };
        let get_fn = self.get_or_register_extern_function(
            get_fn_name,
            vec![ptr_void.clone(), key_ir_type],
            IrType::I64,
        );
        let Some(raw_value) =
            self.builder
                .build_call_direct(get_fn, vec![map_ptr, key_value], IrType::I64)
        else {
            self.loop_stack.pop();
            return;
        };

        // Convert raw u64 to the correct value type
        let value_ir_type = self.convert_type(value_type_id);
        let map_value = match &value_ir_type {
            IrType::Ptr(_) => {
                if let Some(cast) = self.builder.build_bitcast(raw_value, value_ir_type) {
                    cast
                } else {
                    raw_value
                }
            }
            IrType::F64 => {
                if let Some(cast) = self.builder.build_bitcast(raw_value, IrType::F64) {
                    cast
                } else {
                    raw_value
                }
            }
            _ => {
                // Int, Bool, etc.: the raw i64 is already the value.
                raw_value
            }
        };

        if let HirPattern::Variable { symbol, .. } = value_pattern {
            self.symbol_map.insert(*symbol, map_value);
        }

        // Track loop-carried symbols so the body's exit_drop_scope does not free
        // values that escape via the exit phi.
        self.loop_carried_symbols
            .push(phi_nodes.keys().copied().collect());
        self.enter_drop_scope();
        self.lower_block(body);

        // Add back-edge phi incoming values from body end to cond block
        let body_end_block = self.builder.current_block().unwrap_or(loop_body_block);
        for (symbol_id, phi_reg) in &phi_nodes {
            let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                updated_reg
            } else {
                *phi_reg
            };
            self.builder.add_phi_incoming(
                loop_cond_block,
                *phi_reg,
                body_end_block,
                back_edge_value,
            );
        }

        if !self.is_terminated() {
            self.exit_drop_scope();
            self.loop_carried_symbols.pop();
            let Some(idx_to_inc) = self.builder.build_load(index_ptr, IrType::I64) else {
                self.loop_stack.pop();
                return;
            };
            let Some(one) = self.builder.build_const(IrValue::I64(1)) else {
                self.loop_stack.pop();
                return;
            };
            let Some(next_val) =
                self.builder
                    .build_binop(crate::ir::instructions::BinaryOp::Add, idx_to_inc, one)
            else {
                self.loop_stack.pop();
                return;
            };
            self.builder.build_store(index_ptr, next_val);
            self.builder.build_branch(loop_cond_block);
        } else {
            self.loop_carried_symbols.pop();
        }

        self.loop_stack.pop();
        self.builder.switch_to_block(loop_exit_block);

        // Post-loop code must see the exit phi values.
        for (symbol_id, exit_param_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_param_reg);
        }
        for (symbol_id, exit_param_reg) in &exit_phi_nodes {
            if self.owned_heap_values.contains_key(symbol_id) {
                self.owned_heap_values.insert(*symbol_id, *exit_param_reg);
            }
        }
    }

    /// Helper: iterate over a HaxeArray by index. Used by both Array for-in and Map for-in
    /// (where keys are first converted to a HaxeArray).
    pub(crate) fn lower_for_in_over_array(
        &mut self,
        pattern: &HirPattern,
        collection: IrId,
        elem_type_id: TypeId,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) {
        // Read array length from HaxeArray struct (offset 8 = len field)
        let Some(offset_8) = self.builder.build_const(IrValue::I64(8)) else {
            return;
        };
        let Some(len_ptr) =
            self.builder
                .build_binop(crate::ir::instructions::BinaryOp::Add, collection, offset_8)
        else {
            return;
        };
        let Some(array_len) = self.builder.build_load(len_ptr, IrType::I64) else {
            return;
        };

        let Some(zero) = self.builder.build_const(IrValue::I64(0)) else {
            return;
        };
        let Some(index_ptr) = self.builder.build_alloc(IrType::I64, None) else {
            return;
        };
        self.builder.build_store(index_ptr, zero);

        // Save entry block for loop-carried phi incoming edges.
        let entry_block = if let Some(block_id) = self.builder.current_block() {
            block_id
        } else {
            return;
        };

        // Collect referenced variables in loop body for loop-carried SSA updates.
        let mut referenced_vars = std::collections::BTreeSet::new();
        self.collect_referenced_variables_in_block(body, &mut referenced_vars);

        let modified_vars: std::collections::BTreeSet<SymbolId> = referenced_vars
            .into_iter()
            .filter(|sym| {
                let in_map = self.symbol_map.contains_key(sym);
                // Parameters are excluded by symbol kind, not by register.
                let is_param = self.is_parameter_symbol(sym);
                in_map && !is_param
            })
            .collect();

        let mut loop_var_initial_values: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for symbol_id in &modified_vars {
            if let Some(&reg) = self.symbol_map.get(symbol_id) {
                let reg_ty = self
                    .builder
                    .current_function()
                    .and_then(|func| func.locals.get(&reg).map(|l| l.ty.clone()))
                    .or_else(|| self.builder.get_register_type(reg))
                    .unwrap_or(IrType::I64);
                loop_var_initial_values.insert(*symbol_id, (reg, reg_ty));
            }
        }

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

        // Condition block: create loop-carried phis first.
        self.builder.switch_to_block(loop_cond_block);
        let mut phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, (initial_reg, var_type)) in &loop_var_initial_values {
            if let Some(phi_reg) = self.builder.build_phi(loop_cond_block, var_type.clone()) {
                self.builder
                    .add_phi_incoming(loop_cond_block, phi_reg, entry_block, *initial_reg);
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
                phi_nodes.insert(*symbol_id, phi_reg);
                self.symbol_map.insert(*symbol_id, phi_reg);
                if self.owned_heap_values.contains_key(symbol_id) {
                    self.owned_heap_values.insert(*symbol_id, phi_reg);
                }
            }
        }

        // Build exit phis now so break statements can target the correct values.
        let mut exit_phi_nodes: BTreeMap<SymbolId, IrId> = BTreeMap::new();
        for (symbol_id, loop_phi_reg) in &phi_nodes {
            if let Some((_, var_type)) = loop_var_initial_values.get(symbol_id) {
                let Some(exit_param_reg) = self.builder.alloc_reg() else {
                    continue;
                };
                if let Some(func) = self.builder.current_function_mut() {
                    if let Some(exit_block_data) = func.cfg.get_block_mut(loop_exit_block) {
                        let exit_phi = crate::ir::IrPhiNode {
                            dest: exit_param_reg,
                            incoming: vec![(loop_cond_block, *loop_phi_reg)],
                            ty: var_type.clone(),
                        };
                        exit_block_data.add_phi(exit_phi);
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

        // Push loop context for break/continue
        self.loop_stack.push(LoopContext {
            continue_block: loop_cond_block,
            break_block: loop_exit_block,
            label: label.cloned(),
            exit_phi_nodes: exit_phi_nodes.clone(),
            continue_phi_nodes: BTreeMap::new(),
        });

        // Condition block: index < length
        let Some(current_index) = self.builder.build_load(index_ptr, IrType::I64) else {
            self.loop_stack.pop();
            return;
        };
        let Some(cmp_result) = self.builder.build_cmp(
            crate::ir::instructions::CompareOp::Lt,
            current_index,
            array_len,
        ) else {
            self.loop_stack.pop();
            return;
        };
        self.builder
            .build_cond_branch(cmp_result, loop_body_block, loop_exit_block);

        // Body block: get element, bind, execute body, increment
        self.builder.switch_to_block(loop_body_block);
        let Some(idx_for_access) = self.builder.build_load(index_ptr, IrType::I64) else {
            self.loop_stack.pop();
            return;
        };
        let Some(element_value) = self.lower_index_access(collection, idx_for_access, elem_type_id)
        else {
            self.loop_stack.pop();
            return;
        };

        match pattern {
            HirPattern::Variable { symbol, .. } => {
                self.symbol_map.insert(*symbol, element_value);
            }
            // `for (k => v in array)`: the key is the index, the value the element.
            HirPattern::Tuple(subs) if subs.len() == 2 => {
                if let HirPattern::Variable { symbol: key, .. } = &subs[0] {
                    let key_i32 = self
                        .builder
                        .build_cast(idx_for_access, IrType::I64, IrType::I32)
                        .unwrap_or(idx_for_access);
                    self.symbol_map.insert(*key, key_i32);
                }
                if let HirPattern::Variable { symbol: value, .. } = &subs[1] {
                    self.symbol_map.insert(*value, element_value);
                }
            }
            _ => {}
        }

        // Lower the loop body. Track loop-carried symbols so the body's
        // exit_drop_scope does not free values that escape via the exit phi.
        self.loop_carried_symbols
            .push(phi_nodes.keys().copied().collect());
        self.enter_drop_scope();
        self.lower_block(body);

        // Add loop-carried phi incoming values from the body-end block.
        let body_end_block = self.builder.current_block().unwrap_or(loop_body_block);
        for (symbol_id, phi_reg) in &phi_nodes {
            let back_edge_value = if let Some(&updated_reg) = self.symbol_map.get(symbol_id) {
                updated_reg
            } else {
                *phi_reg
            };
            self.builder.add_phi_incoming(
                loop_cond_block,
                *phi_reg,
                body_end_block,
                back_edge_value,
            );
        }

        if !self.is_terminated() {
            self.exit_drop_scope();
            self.loop_carried_symbols.pop();
            let Some(idx_to_inc) = self.builder.build_load(index_ptr, IrType::I64) else {
                self.loop_stack.pop();
                return;
            };
            let Some(one) = self.builder.build_const(IrValue::I64(1)) else {
                self.loop_stack.pop();
                return;
            };
            let Some(next_index) =
                self.builder
                    .build_binop(crate::ir::instructions::BinaryOp::Add, idx_to_inc, one)
            else {
                self.loop_stack.pop();
                return;
            };
            self.builder.build_store(index_ptr, next_index);
            self.builder.build_branch(loop_cond_block);
        } else {
            // Terminated body (break/return/continue) skipped exit_drop_scope;
            // keep the loop-carried stack balanced.
            self.loop_carried_symbols.pop();
        }

        self.loop_stack.pop();
        self.builder.switch_to_block(loop_exit_block);

        // Update symbol map and owned heap tracking to loop-exit values.
        for (symbol_id, exit_param_reg) in &exit_phi_nodes {
            self.symbol_map.insert(*symbol_id, *exit_param_reg);
            if self.owned_heap_values.contains_key(symbol_id) {
                self.owned_heap_values.insert(*symbol_id, *exit_param_reg);
            }
        }
    }
}
