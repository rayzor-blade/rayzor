//! Statement dispatch and block lowering.

use super::*;
use crate::ir::drop_analysis::{DropBehavior, DropPointAnalyzer, DropPoints};
use crate::ir::hir::*;
use crate::ir::mir::moveflow::MoveEventKind;
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
    /// Lower a HIR statement to MIR instructions
    pub(crate) fn lower_statement(&mut self, stmt: &HirStatement) {
        match stmt {
            HirStatement::Let {
                pattern,
                type_hint,
                init,
                is_mutable,
            } => {
                debug!(
                    "[LOWER STMT] Processing Let statement, has_init={}",
                    init.is_some()
                );
                if let Some(init_expr) = init {
                    debug!(
                        "[LOWER STMT] init_expr.kind = {:?}",
                        std::mem::discriminant(&init_expr.kind)
                    );
                    let init_is_value_type = self.expr_is_value_type_expr(init_expr);

                    // Stdlib class methods return their own class type (Arc.init() -> Arc).
                    let monomorphized_class = if let HirExprKind::Call {
                        callee,
                        args: call_args,
                        is_method,
                        ..
                    } = &init_expr.kind
                    {
                        self.detect_stdlib_class_from_call(callee, call_args)
                            .or_else(|| {
                                // A static call with a Variable callee resolves its class through
                                // the stdlib mapping (extern factories like GPUCompute.create()).
                                if !is_method {
                                    if let HirExprKind::Variable { symbol, .. } = &callee.kind {
                                        self.get_stdlib_runtime_info(
                                            *symbol,
                                            TypeId::invalid(),
                                            Some(call_args.len()),
                                            None,
                                        )
                                        .map(|(class_name, _, _)| class_name.to_string())
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

                    // Track ownership for explicit allocations:
                    // - `new` class expressions (except array literals)
                    // - reflective Type.createInstance/createEmptyInstance calls
                    let mut is_heap_alloc = match &init_expr.kind {
                        HirExprKind::New { .. } => {
                            let type_table = self.type_table;
                            let is_array = if let Some(type_ref) = type_table.get(init_expr.ty) {
                                matches!(type_ref.kind, crate::tast::TypeKind::Array { .. })
                            } else {
                                false
                            };
                            !is_array
                        }
                        HirExprKind::Call { callee, args, .. } => {
                            self.type_needs_drop(init_expr.ty)
                                && self.is_reflective_type_alloc_call(callee, args.len())
                        }
                        _ => false,
                    };

                    // @:derive(Copy) — check if RHS is a variable with Copy type
                    let copy_class_sym = if let HirExprKind::Variable { .. } = &init_expr.kind {
                        self.get_copy_class_symbol(init_expr.ty)
                    } else {
                        None
                    };

                    // An array literal's `expr.ty` is inferred from its FIRST element, so lower
                    // it against the declared `Array<Interface>` slot type instead to get the
                    // per-element fat-pointer wrap. Erased-generic returns need the declared
                    // type as well, to pick the right unbox.
                    let prev_let_hint = self.let_target_type_hint.take();
                    if matches!(&init_expr.kind, HirExprKind::Call { .. }) {
                        self.let_target_type_hint = *type_hint;
                    }
                    let value = if let (HirExprKind::Array { elements }, Some(target_ty)) =
                        (&init_expr.kind, type_hint)
                    {
                        let target_is_iface_array = {
                            let type_table = self.type_table;
                            type_table.get(*target_ty).and_then(|t| match &t.kind {
                                TypeKind::Array { element_type } => {
                                    self.get_interface_symbol(*element_type)
                                }
                                _ => None,
                            })
                        };
                        if target_is_iface_array.is_some() {
                            self.lower_array_literal(elements, *target_ty)
                        } else {
                            self.lower_expression(init_expr)
                        }
                    } else {
                        // Anon-literal target hint: a typed Let on a wider typedef carries
                        // optional fields the literal omits; pass it down so the writer's slot
                        // layout matches what readers compute from the typedef.
                        let prev_target = self.object_literal_target_ty.take();
                        if matches!(&init_expr.kind, HirExprKind::ObjectLiteral { .. }) {
                            self.object_literal_target_ty = *type_hint;
                        }
                        let v = self.lower_expression(init_expr);
                        self.object_literal_target_ty = prev_target;
                        v
                    };
                    self.let_target_type_hint = prev_let_hint;

                    // If Copy type, emit shallow copy and mark as heap alloc for drop tracking
                    let value = if let (Some(class_sym), Some(val)) = (copy_class_sym, value) {
                        if let Some(copy_ptr) = self.emit_shallow_copy(val, class_sym) {
                            is_heap_alloc = true;
                            Some(copy_ptr)
                        } else {
                            Some(val)
                        }
                    } else {
                        value
                    };

                    if value.is_none() {
                        warn!(
                            "[LET STMT] INIT EXPRESSION FAILED TO LOWER - variable won't be added to symbol_map! pattern={:?}",
                            pattern
                        );
                    }
                    // Track @:async function call results
                    let is_async_call = if let HirExprKind::Call { callee, .. } = &init_expr.kind {
                        match &callee.kind {
                            HirExprKind::Variable { symbol, .. } => self
                                .symbol_table
                                .get_symbol(*symbol)
                                .map(|s| s.flags.contains(SymbolFlags::ASYNC))
                                .unwrap_or(false),
                            // StaticMethodCall produces Field { object: Variable(class), field: method }
                            HirExprKind::Field { field, .. } => self
                                .symbol_table
                                .get_symbol(*field)
                                .map(|s| s.flags.contains(SymbolFlags::ASYNC))
                                .unwrap_or(false),
                            _ => false,
                        }
                    } else {
                        false
                    };

                    if let Some(value_reg) = value {
                        // Mark variable as async result for Future method dispatch
                        if is_async_call {
                            self.async_result_registers.insert(value_reg);
                            if let HirPattern::Variable { symbol, .. } = pattern {
                                self.async_result_registers.insert(value_reg);
                            }
                        }

                        // Track monomorphized class name for this variable (by SymbolId)
                        if let Some(mono_class) = monomorphized_class {
                            if let HirPattern::Variable { symbol, .. } = pattern {
                                self.monomorphized_var_types.insert(*symbol, mono_class);
                            }
                        } else if let HirPattern::Variable { symbol, .. } = pattern {
                            // Fallback: carry the register's class hint (set by
                            // Dynamic/TypeParameter dispatch) onto the variable so chained
                            // calls disambiguate: `var g = mutex.lock(); g.get(); g.unlock();`
                            if let Some(class_hint) = self.register_class_hints.get(&value_reg) {
                                self.monomorphized_var_types
                                    .insert(*symbol, class_hint.clone());
                            }
                        }

                        // An interface method call's return is Dynamic at TAST but
                        // re-resolved concretely at MIR: override the variable's effective
                        // type so receiver dispatch sees the real type instead of the
                        // Dynamic fallback. Only when the declared hint is itself erased —
                        // a real annotation is handled by the assign-side coercion.
                        if let HirPattern::Variable { symbol, .. } = pattern {
                            if let Some(&real_ty) = self.interface_call_result_types.get(&value_reg)
                            {
                                let hint_is_concrete = type_hint
                                    .and_then(|tid| self.type_table.get(tid))
                                    .map(|t| {
                                        !matches!(
                                            t.kind,
                                            TypeKind::Dynamic
                                                | TypeKind::Placeholder { .. }
                                                | TypeKind::Unknown
                                        )
                                    })
                                    .unwrap_or(false);
                                // The override drives `effective_receiver_type`; it is
                                // meaningless for a scalar return and disturbs direct uses
                                // of the value, so restrict it to reference types.
                                let is_scalar = matches!(
                                    self.convert_type(real_ty),
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
                                );
                                if !hint_is_concrete && !is_scalar {
                                    self.var_concrete_overrides.insert(*symbol, real_ty);
                                }
                            }
                        }

                        // The RHS's real source type: an interface method call returns a raw
                        // concrete value even though TAST erased its type to Dynamic. The
                        // box/unbox coercions below key off it, so the erased type would
                        // insert a spurious unbox of a raw pointer (or box a scalar).
                        let recovered_init_ty =
                            self.interface_call_result_types.get(&value_reg).copied();
                        let init_ty = recovered_init_ty.unwrap_or(init_expr.ty);

                        // An unannotated binding takes the recovered concrete type. An erased
                        // HINT stays the binding type on purpose: for scalars the box is the
                        // representation every Dynamic consumer expects, and objects pass
                        // through raw; consumers needing the value unbox at the use site.
                        let var_type = type_hint.or(Some(init_ty));

                        // Auto-box if assigning concrete value to Dynamic variable
                        // Auto-unbox if assigning Dynamic value to concrete variable
                        let final_value = if let Some(target_ty) = var_type {
                            let after_box = self
                                .maybe_box_value(value_reg, init_ty, target_ty)
                                .unwrap_or(value_reg);

                            if after_box != value_reg {
                                if let HirPattern::Variable { symbol, .. } = pattern {
                                    self.boxed_dynamic_symbols.insert(*symbol);
                                }
                            } else if let HirPattern::Variable { symbol, .. } = pattern {
                                // A Ptr(U8) from a Dynamic-typed expression is a raw anon
                                // handle (e.g. haxe_ereg_matched_pos_anon); track it so field
                                // access skips haxe_unbox_reference_ptr.
                                let is_dynamic_init = {
                                    let tt = self.type_table;
                                    tt.get(init_expr.ty)
                                        .map(|t| matches!(t.kind, TypeKind::Dynamic))
                                        .unwrap_or(false)
                                };
                                let is_ptr_u8 = matches!(
                                    self.builder.get_register_type(value_reg),
                                    Some(IrType::Ptr(ref inner)) if matches!(**inner, IrType::U8)
                                );
                                if is_dynamic_init && is_ptr_u8 {
                                    self.raw_anon_symbols.insert(*symbol);
                                }
                            }

                            self.maybe_unbox_value(after_box, init_ty, target_ty)
                                .unwrap_or(after_box)
                        } else {
                            value_reg
                        };

                        // Auto-box primitive for Null<T> (Optional) assignment
                        let final_value = if let Some(target_ty) = var_type {
                            self.maybe_box_for_optional(final_value, init_ty, target_ty)
                                .unwrap_or(final_value)
                        } else {
                            final_value
                        };

                        let final_value = if let Some(target_ty) = var_type {
                            self.maybe_abstract_from_convert(final_value, init_ty, target_ty)
                                .unwrap_or(final_value)
                        } else {
                            final_value
                        };

                        // Wrap a class instance in an interface fat pointer. SafeCast results
                        // are skipped: that handler already built the fat pointer, or null on
                        // a failed cast, which must not be cloned.
                        let init_is_safe_cast =
                            matches!(&init_expr.kind, HirExprKind::Cast { is_safe: true, .. });
                        let (final_value, wrapped_for_interface) = if let Some(target_ty) = var_type
                        {
                            if init_is_safe_cast {
                                (final_value, false)
                            } else {
                                self.maybe_wrap_for_interface(final_value, init_expr.ty, target_ty)
                            }
                        } else {
                            (final_value, false)
                        };

                        // Structural subtyping: register anon view for class→anon or wider-anon→narrower
                        let is_anon_view =
                            if let (HirPattern::Variable { symbol, .. }, Some(target_ty)) =
                                (pattern, var_type)
                            {
                                self.try_register_anon_view(
                                    *symbol,
                                    final_value,
                                    init_expr.ty,
                                    target_ty,
                                )
                            } else {
                                false
                            };

                        // Propagate existing anon view when copying from a backed variable
                        if !is_anon_view {
                            if let HirExprKind::Variable {
                                symbol: src_symbol, ..
                            } = &init_expr.kind
                            {
                                if let Some(backing) = self.anon_views.get(src_symbol).cloned() {
                                    if let HirPattern::Variable {
                                        symbol: dst_symbol, ..
                                    } = pattern
                                    {
                                        self.anon_views.insert(*dst_symbol, backing);
                                    }
                                }
                            }
                        }

                        // Clone anonymous object handles for COW semantics.
                        // Skip if: object literal (fresh), anon view (backed by class/wider anon),
                        // or copying from a backed variable (shares backing reference).
                        let src_has_view = if let HirExprKind::Variable {
                            symbol: src_sym, ..
                        } = &init_expr.kind
                        {
                            self.anon_views.contains_key(src_sym)
                        } else {
                            false
                        };
                        let final_value = if !is_anon_view
                            && !src_has_view
                            && !matches!(&init_expr.kind, HirExprKind::ObjectLiteral { .. })
                        {
                            self.maybe_clone_anonymous(final_value, init_expr.ty)
                        } else {
                            final_value
                        };

                        self.bind_pattern_with_type(pattern, final_value, var_type, *is_mutable);

                        // @:move tracking: mark the destination register strict-move, and if
                        // the initializer consumed another strict-move local, emit
                        // `MarkMoved` for the source so later reads trip CheckLive.
                        {
                            let is_move_class = var_type
                                .map(|t| self.type_is_move_class(t))
                                .unwrap_or(false)
                                || self.type_is_move_class(init_expr.ty);
                            if is_move_class {
                                self.strict_move_locals.insert(final_value);
                            }
                            // The move-flow recorder works on bindings. A move
                            // out of the initializer is recorded before the new
                            // binding, which is the order `var d = b` happens in.
                            if let HirExprKind::Variable {
                                symbol: src_symbol, ..
                            } = &init_expr.kind
                            {
                                self.record_move_event(
                                    MoveEventKind::Move,
                                    *src_symbol,
                                    init_expr.source_location,
                                );
                            }
                            if let (
                                HirPattern::Variable { symbol: dst, .. },
                                HirExprKind::Variable { symbol: src, .. },
                            ) = (pattern, &init_expr.kind)
                            {
                                self.propagate_borrow(*dst, *src);
                            }
                            if is_move_class {
                                if let HirPattern::Variable { symbol, .. } = pattern {
                                    self.enroll_move_symbol(*symbol);
                                    self.record_move_event(
                                        MoveEventKind::Bind,
                                        *symbol,
                                        init_expr.source_location,
                                    );
                                }
                            }
                            // Variable-on-RHS is a move-by-consume site.
                            if let HirExprKind::Variable {
                                symbol: src_symbol, ..
                            } = &init_expr.kind
                            {
                                if let Some(&src_reg) = self.symbol_map.get(src_symbol) {
                                    // Only when the value actually changed register.
                                    // `var d = b` needs no cast, so `d` IS `b`'s
                                    // register; marking the source moved would poison
                                    // the binding just created from it, and the
                                    // Cranelift liveness check — the one backend that
                                    // enforces this — then traps on correct code.
                                    // A move that relocates (`$9 = cast $2`) still
                                    // marks `$2`, whose reads all belong to the old
                                    // binding.
                                    if src_reg != final_value
                                        && self.strict_move_locals.contains(&src_reg)
                                    {
                                        let _ = self.builder.build_mark_moved(src_reg);
                                    }
                                }
                            }
                        }

                        // Only AutoDrop types (user-defined classes) are drop-tracked, not
                        // RuntimeManaged externs (Thread, Channel, Arc, Mutex) or NoDrop.
                        if let HirPattern::Variable { symbol, .. } = pattern {
                            if init_is_value_type {
                                self.value_type_symbols.insert(*symbol);
                            } else {
                                self.value_type_symbols.remove(symbol);
                            }
                            if wrapped_for_interface {
                                // Interface assignments allocate a fat pointer wrapper that
                                // must be freed on reassignment/scope-exit.
                                self.register_owned_value(*symbol, final_value);
                                if self.get_class_symbol(init_expr.ty).is_some() {
                                    if let HirExprKind::Variable {
                                        symbol: src_symbol, ..
                                    } = &init_expr.kind
                                    {
                                        // Class -> interface wrapping aliases the class object,
                                        // so the source stops being the sole owner while the
                                        // wrapper still references it.
                                        if self.owned_heap_values.remove(src_symbol).is_some() {
                                            self.reassigned_in_scope.insert(*src_symbol);
                                        }
                                    }
                                }
                            } else if is_heap_alloc {
                                let needs_drop = self.type_needs_drop(init_expr.ty);
                                if needs_drop {
                                    self.register_owned_value(*symbol, final_value);
                                }
                            }
                        }

                        // Ownership transfer: `var current = n1` aliases the heap value, so
                        // ownership moves to the destination and the source is no longer freed
                        // while the alias lives. Interface wrapping already did this transfer.
                        //
                        // Skipped for @:derive(Copy): `emit_shallow_copy` already gave the
                        // destination an independent allocation it owns, so transferring would
                        // free that copy and re-point the destination at the source's storage.
                        if !wrapped_for_interface && copy_class_sym.is_none() {
                            if let HirPattern::Variable {
                                symbol: dst_symbol, ..
                            } = pattern
                            {
                                let src_symbol = match &init_expr.kind {
                                    HirExprKind::Variable { symbol, .. } => Some(*symbol),
                                    HirExprKind::Cast {
                                        expr: cast_expr,
                                        is_safe: true,
                                        ..
                                    } => {
                                        if let HirExprKind::Variable { symbol, .. } =
                                            &cast_expr.kind
                                        {
                                            Some(*symbol)
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                };
                                if let Some(src_sym) = src_symbol {
                                    if let Some(src_ir_id) = self.owned_heap_values.remove(&src_sym)
                                    {
                                        self.reassigned_in_scope.insert(src_sym);
                                        self.register_owned_value(*dst_symbol, src_ir_id);
                                    }
                                }

                                // Ownership-taking constructors: `new Arc(d)`, `new Mutex(d)`,
                                // `new Box(d)`. These transfer ownership of the argument to the
                                // wrapper. Remove the argument variable from drop tracking so
                                // it isn't freed when the scope exits (the wrapper now owns it).
                                if let HirExprKind::New {
                                    args, class_name, ..
                                } = &init_expr.kind
                                {
                                    let is_ownership_wrapper = class_name
                                        .and_then(|n| self.string_interner.get(n))
                                        .map_or(false, |name| {
                                            matches!(name, "Arc" | "Mutex" | "Box")
                                        });
                                    if is_ownership_wrapper {
                                        for arg in args {
                                            if let HirExprKind::Variable {
                                                symbol: arg_sym, ..
                                            } = &arg.kind
                                            {
                                                if self.owned_heap_values.remove(arg_sym).is_some()
                                                {
                                                    self.reassigned_in_scope.insert(*arg_sym);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            HirStatement::Expr(expr) => {
                self.temp_heap_values.clear();

                // Shadow-stack location updates are emitted only at call expressions and
                // throw statements, to avoid N extra extern calls per function.

                let result = self.lower_expression(expr);

                // The top-level result is not tracked as a temp: functions like
                // Array.push() may store their argument, so freeing the result would
                // dangle. Only chained receivers are freed, at the field-access site.
                let _ = result;

                self.drop_temps();
            }

            HirStatement::Assign { lhs, rhs, op } => {
                // type_needs_drop catches any heap-allocated RHS, not just `new`.
                let rhs_type_needs_drop = self.type_needs_drop(rhs.ty);

                // Resolved before RHS evaluation, so a RHS that reuses the variable can't
                // cause a double-free of the old value.
                let lhs_symbol = match lhs {
                    HirLValue::Variable(symbol) => Some(*symbol),
                    _ => None,
                };

                self.temp_heap_values.clear();
                let rhs_is_value_type = self.expr_is_value_type_expr(rhs) && op.is_none();

                // @:derive(Copy) — check if RHS is a variable with Copy type
                let assign_copy_class = if op.is_none() {
                    if let HirExprKind::Variable { .. } = &rhs.kind {
                        self.get_copy_class_symbol(rhs.ty)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Anon-literal target hint: `typedVar = { ... }` passes the lhs type down so
                // the writer's slot layout includes optional fields the literal omits.
                let prev_anon_target = self.object_literal_target_ty.take();
                if matches!(&rhs.kind, HirExprKind::ObjectLiteral { .. }) {
                    if let Some(lhs_sym) = lhs_symbol {
                        if let Some(sym_info) = self.symbol_table.get_symbol(lhs_sym) {
                            if sym_info.type_id != TypeId::invalid() {
                                self.object_literal_target_ty = Some(sym_info.type_id);
                            }
                        }
                    }
                }
                let rhs_value = self.lower_expression(rhs);
                self.object_literal_target_ty = prev_anon_target;

                // Storing a borrow in a field keeps it alive past the call that
                // lent it. An assignment to a LOCAL is not an escape - it is
                // just another name - so only field and index targets count.
                if !matches!(lhs, HirLValue::Variable(_)) {
                    if let HirExprKind::Variable { symbol, .. } = &rhs.kind {
                        self.check_borrow_escape(*symbol, "is stored", rhs.source_location);
                    }
                }
                if let (HirLValue::Variable(dst), HirExprKind::Variable { symbol: src, .. }) =
                    (lhs, &rhs.kind)
                {
                    self.propagate_borrow(*dst, *src);
                }

                // Move-flow: the RHS consumes a binding, the LHS starts a new
                // one. A plain reassignment is therefore what revives a moved
                // binding — the kill has to be recorded, or every later read
                // of a reassigned variable reports against the stale move.
                if op.is_none() {
                    if let HirExprKind::Variable {
                        symbol: rhs_symbol, ..
                    } = &rhs.kind
                    {
                        self.record_move_event(
                            MoveEventKind::Move,
                            *rhs_symbol,
                            rhs.source_location,
                        );
                    }
                    if let Some(lhs_sym) = lhs_symbol {
                        let lhs_is_move = self
                            .symbol_table
                            .get_symbol(lhs_sym)
                            .map(|s| s.type_id)
                            .filter(|t| *t != TypeId::invalid())
                            .map(|t| self.type_is_move_class(t))
                            .unwrap_or(false);
                        if lhs_is_move {
                            self.enroll_move_symbol(lhs_sym);
                            self.record_move_event(
                                MoveEventKind::Bind,
                                lhs_sym,
                                rhs.source_location,
                            );
                        }
                    }
                }

                let (rhs_value, rhs_was_copied) =
                    if let (Some(class_sym), Some(val)) = (assign_copy_class, rhs_value) {
                        // Guard against self-assignment (a = a)
                        let is_self_assign = if let (
                            Some(lhs_sym),
                            HirExprKind::Variable {
                                symbol: rhs_sym, ..
                            },
                        ) = (lhs_symbol, &rhs.kind)
                        {
                            lhs_sym == *rhs_sym
                        } else {
                            false
                        };
                        if !is_self_assign {
                            if let Some(copy_ptr) = self.emit_shallow_copy(val, class_sym) {
                                (Some(copy_ptr), true)
                            } else {
                                (Some(val), false)
                            }
                        } else {
                            (Some(val), false)
                        }
                    } else {
                        (rhs_value, false)
                    };

                if let Some(rhs_reg) = rhs_value {
                    let final_value = if let Some(bin_op) = op {
                        let lhs_value = self.lower_lvalue_read(lhs);
                        lhs_value.and_then(|lhs_reg| {
                            let result_reg = self.builder.build_binop(
                                self.convert_binary_op(*bin_op),
                                lhs_reg,
                                rhs_reg,
                            )?;

                            // Register the result type for Cranelift
                            let result_type = self.convert_type(rhs.ty);
                            if let Some(func) = self.builder.current_function_mut() {
                                func.locals.insert(
                                    result_reg,
                                    crate::ir::IrLocal {
                                        name: format!("_binop{}", result_reg.0),
                                        ty: result_type,
                                        mutable: false,
                                        source_location: crate::ir::IrSourceLocation {
                                            file_id: 0,
                                            line: 0,
                                            column: 0,
                                        },
                                        allocation: crate::ir::AllocationHint::Register,
                                    },
                                );
                            }

                            Some(result_reg)
                        })
                    } else {
                        Some(rhs_reg)
                    };

                    if let Some(value) = final_value {
                        // Dynamic↔typed coercion for variable targets (`d = 60`) and field
                        // targets (`obj.v = 60` where v:Dynamic) alike: a primitive written
                        // raw into a Dynamic field reads back as a bogus Dynamic pointer.
                        let lhs_target_ty: Option<TypeId> = match lhs {
                            HirLValue::Variable(sym) => self
                                .symbol_table
                                .get_symbol(*sym)
                                .map(|s| s.type_id)
                                .filter(|t| *t != TypeId::invalid()),
                            HirLValue::Field { field, .. } => self
                                .symbol_table
                                .get_symbol(*field)
                                .map(|s| s.type_id)
                                .filter(|t| *t != TypeId::invalid()),
                            _ => None,
                        };
                        // Recover the concrete type of an interface-call RHS, which TAST
                        // erased to `Dynamic`; otherwise the coercion below unboxes a raw
                        // concrete pointer as though it were a box.
                        let recovered_rhs_ty =
                            self.interface_call_result_types.get(&value).copied();
                        let rhs_ty = recovered_rhs_ty.unwrap_or(rhs.ty);
                        // A variable target whose declared type is itself erased takes the
                        // recovered type, so a reassigned local keeps the raw concrete
                        // representation its binding established. Field targets keep their
                        // declared type: a Dynamic field may be deliberate and its readers
                        // use the box protocol.
                        let lhs_target_ty = match (lhs, recovered_rhs_ty, lhs_target_ty) {
                            (HirLValue::Variable(_), Some(recovered), Some(declared)) => {
                                let declared_is_erased = self
                                    .type_table
                                    .get(declared)
                                    .map(|t| {
                                        matches!(
                                            t.kind,
                                            TypeKind::Dynamic
                                                | TypeKind::Placeholder { .. }
                                                | TypeKind::Unknown
                                        )
                                    })
                                    .unwrap_or(false);
                                Some(if declared_is_erased {
                                    recovered
                                } else {
                                    declared
                                })
                            }
                            (_, _, t) => t,
                        };
                        let value = if let Some(target_ty) = lhs_target_ty {
                            // RAYZOR_ASSIGN_DEBUG=1: names the variable whose declared type
                            // drives the box/unbox coercion.
                            if std::env::var("RAYZOR_ASSIGN_DEBUG").is_ok() {
                                if let HirLValue::Variable(sym) = lhs {
                                    let name = self
                                        .symbol_table
                                        .get_symbol(*sym)
                                        .and_then(|s| self.string_interner.get(s.name))
                                        .unwrap_or("?")
                                        .to_string();
                                    let kind = |t: TypeId| {
                                        self.type_table
                                            .get(t)
                                            .map(|x| format!("{:?}", x.kind))
                                            .unwrap_or_else(|| "?".to_string())
                                    };
                                    eprintln!(
                                        "[ASSIGN] {} target={} rhs={}",
                                        name,
                                        kind(target_ty),
                                        kind(rhs_ty)
                                    );
                                }
                            }
                            let after_box = self
                                .maybe_box_value(value, rhs_ty, target_ty)
                                .unwrap_or(value);
                            // Track boxing for Dynamic arithmetic safety (variable targets only).
                            if after_box != value {
                                if let HirLValue::Variable(sym) = lhs {
                                    self.boxed_dynamic_symbols.insert(*sym);
                                }
                            }
                            self.maybe_unbox_value(after_box, rhs_ty, target_ty)
                                .unwrap_or(after_box)
                        } else {
                            value
                        };

                        let value = if let HirLValue::Variable(sym) = lhs {
                            if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                let target_ty = sym_info.type_id;
                                if target_ty != TypeId::invalid() {
                                    self.maybe_abstract_from_convert(value, rhs_ty, target_ty)
                                        .unwrap_or(value)
                                } else {
                                    value
                                }
                            } else {
                                value
                            }
                        } else {
                            value
                        };

                        // Wrap class instance in interface fat pointer if assigning to interface var
                        let (value, wrapped_for_interface) = if let HirLValue::Variable(sym) = lhs {
                            if let Some(sym_info) = self.symbol_table.get_symbol(*sym) {
                                if sym_info.type_id != TypeId::invalid() {
                                    self.maybe_wrap_for_interface(value, rhs.ty, sym_info.type_id)
                                } else {
                                    (value, false)
                                }
                            } else {
                                (value, false)
                            }
                        } else {
                            (value, false)
                        };

                        // Numeric promotion: `var f:Float = i` must sitofp the int bits, not
                        // reinterpret them as f64. Synthetic temps from desugars have no
                        // usable symbol-table TypeId (`gen_temp_var` allocates a SymbolId
                        // without registering it), so fall back to the lhs register's
                        // tracked IR type, which the Let-init already set.
                        let value = {
                            let tgt_ir_opt: Option<IrType> = match lhs {
                                HirLValue::Variable(sym) => {
                                    let from_symbol = self
                                        .symbol_table
                                        .get_symbol(*sym)
                                        .map(|s| s.type_id)
                                        .filter(|t| *t != TypeId::invalid())
                                        .map(|t| self.convert_type(t));
                                    from_symbol.or_else(|| {
                                        self.symbol_map
                                            .get(sym)
                                            .and_then(|reg| self.builder.get_register_type(*reg))
                                    })
                                }
                                _ => None,
                            };
                            if let Some(tgt_ir) = tgt_ir_opt {
                                let val_ir =
                                    self.builder.get_register_type(value).unwrap_or(IrType::I64);
                                let needs_coerce = matches!(
                                    (&val_ir, &tgt_ir),
                                    (
                                        IrType::I8
                                            | IrType::I16
                                            | IrType::I32
                                            | IrType::I64
                                            | IrType::U8
                                            | IrType::U16
                                            | IrType::U32
                                            | IrType::U64,
                                        IrType::F32 | IrType::F64,
                                    ) | (
                                        IrType::F32 | IrType::F64,
                                        IrType::I8
                                            | IrType::I16
                                            | IrType::I32
                                            | IrType::I64
                                            | IrType::U8
                                            | IrType::U16
                                            | IrType::U32
                                            | IrType::U64,
                                    )
                                );
                                if needs_coerce {
                                    self.builder
                                        .build_cast(value, val_ir, tgt_ir)
                                        .unwrap_or(value)
                                } else {
                                    value
                                }
                            } else {
                                value
                            }
                        };

                        // Clone anonymous object handles for COW semantics on reassignment.
                        // Skip for object literals (fresh handles) and compound assignments.
                        let value = if op.is_none()
                            && !matches!(&rhs.kind, HirExprKind::ObjectLiteral { .. })
                        {
                            self.maybe_clone_anonymous(value, rhs.ty)
                        } else {
                            value
                        };

                        self.lower_lvalue_write(lhs, value);

                        // If the LHS is a global variable, the value escapes to global storage
                        // and must NOT be tracked for drop/free. Skip all drop tracking.
                        let lhs_is_global = match lhs {
                            HirLValue::Variable(sym) => self.global_symbol_map.contains_key(sym),
                            _ => false,
                        };

                        // Only `new` counts as an owned allocation. Field access and variable
                        // reads are borrowed references into existing objects, and a call
                        // result is ambiguous — a method may return an existing object.
                        let rhs_is_owned_allocation =
                            rhs_was_copied || matches!(&rhs.kind, HirExprKind::New { .. });

                        // Assigning into a Field/Index lvalue escapes the value into another
                        // object, so a tracked RHS variable stops being tracked — its pointer
                        // now lives inside that object and must survive scope exit.
                        let lhs_is_field =
                            matches!(lhs, HirLValue::Field { .. } | HirLValue::Index { .. });
                        if lhs_is_field {
                            if let HirExprKind::Variable {
                                symbol: rhs_sym, ..
                            } = &rhs.kind
                            {
                                if self.owned_heap_values.remove(rhs_sym).is_some() {
                                    self.reassigned_in_scope.insert(*rhs_sym);
                                }
                            }
                        }

                        let lhs_was_tracked =
                            lhs_symbol.map_or(false, |s| self.owned_heap_values.contains_key(&s));
                        let rhs_needs_tracking = (rhs_is_owned_allocation && rhs_type_needs_drop)
                            || wrapped_for_interface;

                        if wrapped_for_interface && self.get_class_symbol(rhs.ty).is_some() {
                            if let HirExprKind::Variable {
                                symbol: src_symbol, ..
                            } = &rhs.kind
                            {
                                // Class -> interface wrapping aliases the class object, so the
                                // source stops being the sole owner.
                                if self.owned_heap_values.remove(src_symbol).is_some() {
                                    self.reassigned_in_scope.insert(*src_symbol);
                                }
                            }
                        }

                        if !lhs_is_global {
                            if lhs_was_tracked {
                                if let Some(symbol) = lhs_symbol {
                                    if rhs_needs_tracking {
                                        // RHS creates a new allocation → free old, track new
                                        self.register_owned_value(symbol, value);
                                    } else {
                                        // Borrowed RHS: free the old owned value, stop tracking
                                        if let Some(old_ir_id) =
                                            self.owned_heap_values.remove(&symbol)
                                        {
                                            self.emit_tracked_free(old_ir_id, false);
                                        }
                                        self.reassigned_in_scope.insert(symbol);
                                    }
                                }
                            } else if rhs_needs_tracking {
                                // New allocation assigned to previously-untracked variable
                                if let Some(symbol) = lhs_symbol {
                                    self.register_owned_value(symbol, value);
                                }
                            }
                        }

                        // Remove the assigned value from temps (it's now owned by the variable)
                        self.temp_heap_values.retain(|&id| id != value);
                    }

                    if let Some(symbol) = lhs_symbol {
                        if rhs_is_value_type {
                            self.value_type_symbols.insert(symbol);
                        } else {
                            self.value_type_symbols.remove(&symbol);
                        }
                    }

                    // Free intermediates from RHS evaluation, e.g. `z.mul(z)` in
                    // `z = z.mul(z).add(c)`.
                    self.drop_temps();
                }
            }

            HirStatement::Return(value) => {
                debug!("[Return]: has_value: {}", value.is_some());
                // Identify the returned symbol (if returning a variable) so we can
                // skip freeing its allocation in cleanup
                let returned_symbol = value.as_ref().and_then(|e| match &e.kind {
                    HirExprKind::Variable { symbol, .. } => Some(*symbol),
                    _ => None,
                });
                // A borrow may be read and passed on, but returning it hands the
                // caller a reference that outlives the call it was lent for.
                if let Some(sym) = returned_symbol {
                    let loc = value
                        .as_ref()
                        .map(|e| e.source_location)
                        .unwrap_or_else(SourceLocation::unknown);
                    self.check_borrow_escape(sym, "is returned", loc);
                }
                let ret_value = value.as_ref().and_then(|e| {
                    debug!(
                        "[Return]: Lowering return expression, expr kind: {:?}",
                        std::mem::discriminant(&e.kind)
                    );
                    // Anon-literal target hint: a returned ObjectLiteral needs the declared
                    // return typedef so its optional fields land in the slot table.
                    let prev_target = self.object_literal_target_ty.take();
                    if matches!(&e.kind, HirExprKind::ObjectLiteral { .. }) {
                        self.object_literal_target_ty = self.current_function_return_type;
                    }
                    let result = self.lower_expression(e);
                    self.object_literal_target_ty = prev_target;
                    debug!("[Return]: Return expression lowered to: {:?}", result);
                    if result.is_none() {
                        warn!("ERROR [Return]: Failed to lower return expression!");
                        debug!("ERROR [Return]: Expression was: {:#?}", e);
                    }
                    // Box for Null<T> return type: if function returns Optional{primitive}
                    // but the expression produces a raw primitive, box it.
                    if let (Some(val), Some(fn_ret_ty)) =
                        (result, self.current_function_return_type)
                    {
                        if let Some(boxed) = self.maybe_box_for_optional(val, e.ty, fn_ret_ty) {
                            return Some(boxed);
                        }
                        // Inverse: a `Null<T>` expression (boxed `DynamicValue*`) returned
                        // from a `:T` function must be unboxed, or the caller reads the box
                        // pointer's address as the value.
                        if let Some(unboxed) =
                            self.maybe_unbox_optional_for_target(val, e.ty, fn_ret_ty)
                        {
                            return Some(unboxed);
                        }
                        // Interface return wrapping: the auto-wrap in `lower_expression` only
                        // fires when `expr.ty` is the interface, but `return classInstance`
                        // keeps `expr.ty` as the class, so wrap the raw class pointer here
                        // for the caller's vtable dispatch.
                        let (wrapped, did_wrap) =
                            self.maybe_wrap_for_interface(val, e.ty, fn_ret_ty);
                        if did_wrap {
                            return Some(wrapped);
                        }
                        // Cross-module the returned type can arrive Unknown/Placeholder, so
                        // `maybe_wrap_for_interface` cannot wrap it; recover the class by
                        // name and wrap via the name-based fat-ptr build, as the call-arg
                        // path does.
                        if let Some(iface_sym) = self.get_interface_symbol(fn_ret_ty) {
                            if !self.interface_wrapped_args.contains(&val) {
                                if let Some(class_fqn) = self.new_arg_class_fqn(e) {
                                    if let Some(w) = self.wrap_new_class_as_interface_by_name(
                                        val, &class_fqn, iface_sym,
                                    ) {
                                        self.interface_wrapped_args.insert(w);
                                        return Some(w);
                                    }
                                }
                            }
                        }

                        // Numeric promotion: `function foo():Float { return x; }` where x is
                        // an Int must cast, or the raw int bits are reinterpreted as a float
                        // across the callee/caller boundary. Both directions, for parity
                        // with the Let/Assign handlers.
                        let val_ir = self.builder.get_register_type(val).unwrap_or(IrType::I64);
                        let ret_ir = self.convert_type(fn_ret_ty);
                        let needs_int_to_float = matches!(
                            (&val_ir, &ret_ir),
                            (
                                IrType::I8
                                    | IrType::I16
                                    | IrType::I32
                                    | IrType::I64
                                    | IrType::U8
                                    | IrType::U16
                                    | IrType::U32
                                    | IrType::U64,
                                IrType::F32 | IrType::F64,
                            )
                        );
                        let needs_float_to_int = matches!(
                            (&val_ir, &ret_ir),
                            (
                                IrType::F32 | IrType::F64,
                                IrType::I8
                                    | IrType::I16
                                    | IrType::I32
                                    | IrType::I64
                                    | IrType::U8
                                    | IrType::U16
                                    | IrType::U32
                                    | IrType::U64,
                            )
                        );
                        if needs_int_to_float || needs_float_to_int {
                            if let Some(cast_val) = self.builder.build_cast(val, val_ir, ret_ir) {
                                return Some(cast_val);
                            }
                        }
                    }
                    result
                });
                // Cleanup all scopes before returning - free all owned heap values
                // BUT skip the returned value (it escapes the function)
                self.cleanup_all_scopes_except_symbol(returned_symbol);
                debug!(
                    "[Return]: Building return instruction with value: {:?}",
                    ret_value
                );
                self.builder.build_return(ret_value);
            }

            HirStatement::Break(label) => {
                if let Some(loop_ctx) = self.find_loop_context(label.as_ref()) {
                    let break_block = loop_ctx.break_block;
                    let exit_phi_nodes = loop_ctx.exit_phi_nodes.clone();

                    let current_block = self.builder.current_block().unwrap();

                    // Feed the exit-block phis the symbols' values on this edge.
                    for (symbol_id, exit_phi_reg) in &exit_phi_nodes {
                        let current_value = if let Some(&reg) = self.symbol_map.get(symbol_id) {
                            reg
                        } else {
                            // If symbol not in map, use the phi register itself (shouldn't happen)
                            *exit_phi_reg
                        };

                        self.builder.add_phi_incoming(
                            break_block,
                            *exit_phi_reg,
                            current_block,
                            current_value,
                        );
                    }

                    // The break path emits Frees and mutates
                    // drop_scope_stack/owned_heap_values; those mutations must not persist
                    // into the non-break path, or the back-edge's exit_drop_scope pops the
                    // wrong scope.
                    let saved_scope_stack = self.drop_scope_stack.clone();
                    let saved_owned = self.owned_heap_values.clone();

                    // Free loop body allocations before breaking out
                    self.exit_drop_scope();
                    self.builder.build_branch(break_block);

                    // Restore drop state for non-break path code generation
                    self.drop_scope_stack = saved_scope_stack;
                    self.owned_heap_values = saved_owned;
                } else {
                    self.add_error("Break outside of loop", SourceLocation::unknown());
                }
            }

            HirStatement::Continue(label) => {
                if let Some(loop_ctx) = self.find_loop_context(label.as_ref()) {
                    // Copy fields before mutable borrow
                    let continue_block = loop_ctx.continue_block;
                    let continue_phi_nodes = loop_ctx.continue_phi_nodes.clone();

                    // Feed the update-block phis, as break does for the exit phis.
                    if !continue_phi_nodes.is_empty() {
                        if let Some(current_block) = self.builder.current_block() {
                            for (symbol_id, upd_phi_reg) in &continue_phi_nodes {
                                let current_value = self
                                    .symbol_map
                                    .get(symbol_id)
                                    .copied()
                                    .unwrap_or(*upd_phi_reg);
                                self.builder.add_phi_incoming(
                                    continue_block,
                                    *upd_phi_reg,
                                    current_block,
                                    current_value,
                                );
                            }
                        }
                    }

                    // Save drop state around the cleanup, same reason as break above.
                    let saved_scope_stack = self.drop_scope_stack.clone();
                    let saved_owned = self.owned_heap_values.clone();

                    // Free loop body allocations before continuing to next iteration
                    self.exit_drop_scope();
                    self.builder.build_branch(continue_block);

                    // Restore drop state for non-continue path code generation
                    self.drop_scope_stack = saved_scope_stack;
                    self.owned_heap_values = saved_owned;
                } else {
                    self.add_error("Continue outside of loop", SourceLocation::unknown());
                }
            }

            HirStatement::Throw(expr) => {
                let thrown_type = expr.ty;
                let throw_loc = expr.source_location;
                if let Some(exception_reg) = self.lower_expression(expr) {
                    // Update the top shadow-stack frame to the exact throw line/col so
                    // the trace snippet points at the throw statement, not the function def.
                    if throw_loc.is_valid() && throw_loc.line > 0 {
                        let update_loc_fn = self.get_or_register_extern_function(
                            "rayzor_update_call_frame_location",
                            vec![IrType::I32, IrType::I32],
                            IrType::Void,
                        );
                        let line_const = self
                            .builder
                            .build_const(IrValue::I32(throw_loc.line as i32))
                            .expect("failed to create throw line const");
                        let col_const = self
                            .builder
                            .build_const(IrValue::I32(throw_loc.column as i32))
                            .expect("failed to create throw col const");
                        self.builder.build_call_direct(
                            update_loc_fn,
                            vec![line_const, col_const],
                            IrType::Void,
                        );
                    }

                    // Populate the exception's `stack` field with the current call stack trace.
                    // This must happen AFTER rayzor_update_call_frame_location (so the
                    // throw location is in the trace) but BEFORE rayzor_throw_typed.
                    // Only for class types (Exception subclasses), not primitive throws.
                    if self.get_class_symbol(thrown_type).is_some() {
                        let stack_name = self.string_interner.intern("stack");
                        let stack_field = self.resolve_field_index_by_name(stack_name, thrown_type);
                        if let Some((_class_ty, field_idx)) = stack_field {
                            let call_stack_fn = self.get_or_register_extern_function(
                                "rayzor_native_stack_trace_call_stack",
                                vec![],
                                IrType::Ptr(Box::new(IrType::U8)),
                            );
                            let stack_str = self
                                .builder
                                .build_call_direct(
                                    call_stack_fn,
                                    vec![],
                                    IrType::Ptr(Box::new(IrType::U8)),
                                )
                                .expect("failed to call rayzor_native_stack_trace_call_stack");

                            let idx_const = self
                                .builder
                                .build_const(IrValue::I32(field_idx as i32))
                                .expect("failed to create stack field index const");
                            let field_ptr = self
                                .builder
                                .build_gep(
                                    exception_reg,
                                    vec![idx_const],
                                    IrType::Ptr(Box::new(IrType::Void)),
                                )
                                .expect("failed to build stack field GEP");
                            self.builder.build_store(field_ptr, stack_str);
                        }
                    }

                    // Cast exception to i64 for uniform storage
                    let reg_type = self
                        .builder
                        .get_register_type(exception_reg)
                        .unwrap_or(IrType::I64);
                    let exc_as_i64 = if reg_type != IrType::I64 {
                        self.builder
                            .build_cast(exception_reg, reg_type, IrType::I64)
                            .unwrap_or(exception_reg)
                    } else {
                        exception_reg
                    };

                    // For class throws the type id comes from the object header[0]: it
                    // already holds the id `runtime_type_id` produces for typed catches, so
                    // thrown and expected ids share one encoding.
                    let thrown_type_reg = if self.get_class_symbol(thrown_type).is_some() {
                        let obj_ptr_ty = IrType::Ptr(Box::new(IrType::U8));
                        let obj_ptr = self
                            .builder
                            .build_cast(exc_as_i64, IrType::I64, obj_ptr_ty.clone())
                            .unwrap_or(exception_reg);
                        let idx0 = self
                            .builder
                            .build_const(IrValue::I32(0))
                            .expect("failed to create throw header index const");
                        let header_ptr = self
                            .builder
                            .build_gep(obj_ptr, vec![idx0], IrType::I64)
                            .expect("failed to build throw header gep");
                        let header_raw = self
                            .builder
                            .build_load(header_ptr, IrType::I64)
                            .expect("failed to load throw header type id");
                        self.builder
                            .build_cast(header_raw, IrType::I64, IrType::I32)
                            .expect("failed to cast throw class type id")
                    } else {
                        let thrown_type_id = self.runtime_type_id(expr.ty);
                        self.builder
                            .build_const(IrValue::I32(thrown_type_id as i32))
                            .expect("failed to create throw type_id const")
                    };
                    let throw_fn = self.get_or_register_extern_function(
                        "rayzor_throw_typed",
                        vec![IrType::I64, IrType::I32],
                        IrType::Void,
                    );
                    self.builder.build_call_direct(
                        throw_fn,
                        vec![exc_as_i64, thrown_type_reg],
                        IrType::Void,
                    );
                    self.builder.build_unreachable();
                }
            }

            HirStatement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.lower_if_statement(condition, then_branch, else_branch.as_ref());
            }

            HirStatement::Switch { scrutinee, cases } => {
                self.lower_switch_statement(scrutinee, cases);
            }

            HirStatement::While {
                condition,
                body,
                label,
                continue_update,
            } => {
                self.lower_while_loop(condition, body, label.as_ref(), continue_update.as_ref());
            }

            HirStatement::DoWhile {
                body,
                condition,
                label,
            } => {
                self.lower_do_while_loop(body, condition, label.as_ref());
            }

            HirStatement::ForIn {
                pattern,
                iterator,
                body,
                label,
            } => {
                self.lower_for_in_loop(pattern, iterator, body, label.as_ref());
            }

            HirStatement::TryCatch {
                try_block,
                catches,
                finally_block,
            } => {
                self.lower_try_catch(try_block, catches, finally_block.as_ref());
            }

            HirStatement::Label { symbol, block } => {
                // Labels in MIR become block labels
                let label_block = self
                    .builder
                    .create_block_with_label(format!("label_{}", symbol.as_raw()));
                if let Some(block_id) = label_block {
                    self.builder.build_branch(block_id);
                    self.builder.switch_to_block(block_id);
                    self.lower_block(block);
                }
            }
        }
    }

    /// Lower a HIR block to MIR
    pub(crate) fn lower_block(&mut self, block: &HirBlock) {
        for stmt in block.statements.iter() {
            self.lower_statement(stmt);

            // Free variables whose last use is this statement (lifetime-based drop).
            self.check_drop_points_after_statement();

            self.current_stmt_index += 1;
        }

        if let Some(expr) = &block.expr {
            let _result = self.lower_expression(expr);
        }
    }

    /// Lower a HIR block expression to MIR, returning the trailing expression's value
    pub(crate) fn lower_block_expr(&mut self, block: &HirBlock) -> Option<IrId> {
        for stmt in block.statements.iter() {
            self.lower_statement(stmt);
            self.check_drop_points_after_statement();
            self.current_stmt_index += 1;
        }

        if let Some(expr) = &block.expr {
            self.lower_expression(expr)
        } else {
            None
        }
    }
}
