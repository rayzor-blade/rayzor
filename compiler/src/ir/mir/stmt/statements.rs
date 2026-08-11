//! Statement dispatch and block lowering.

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
                // Lower initialization expression if present
                if let Some(init_expr) = init {
                    debug!(
                        "[LOWER STMT] init_expr.kind = {:?}",
                        std::mem::discriminant(&init_expr.kind)
                    );
                    let init_is_value_type = self.expr_is_value_type_expr(init_expr);

                    // Check for stdlib class method calls like Arc.init() or arc.clone()
                    // These methods return the same stdlib class type (e.g., Arc.init() -> Arc, Arc.clone() -> Arc)
                    let monomorphized_class = if let HirExprKind::Call {
                        callee,
                        args: call_args,
                        is_method,
                        ..
                    } = &init_expr.kind
                    {
                        self.detect_stdlib_class_from_call(callee, call_args)
                            .or_else(|| {
                                // For static calls (is_method=false) with Variable callee,
                                // resolve the method via stdlib mapping to identify the class.
                                // This handles extern class factory methods like GPUCompute.create().
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

                    // Special case: if init is an array literal AND the var
                    // has an explicit `Array<Interface>` type hint, the literal
                    // expression's own `expr.ty` was inferred from the FIRST
                    // element (typechecker doesn't propagate slot context), so
                    // `lower_array_literal` reading `array_type=expr.ty` would
                    // see `Array<ConcreteClass>` and skip the element wrap.
                    // Bypass that and lower the literal with the declared
                    // `Array<Interface>` slot type so each Class element gets
                    // wrapped in a fat pointer at construction.
                    // Erased-generic returns (inferred Channel<T>.receive())
                    // need the declared type to pick the right unbox; carry
                    // the Let's type_hint across the init lowering (same
                    // save/restore pattern as object_literal_target_ty).
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
                        // Anon-literal target hint: a typed Let on a wider
                        // typedef carries optional fields the literal omits;
                        // pass it down so the writer's slot layout matches
                        // what readers compute from the typedef. Same fix
                        // shape as the Return handler above.
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

                    // Bind to pattern and register as local
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
                                // Also mark any register the symbol maps to later
                                self.async_result_registers.insert(value_reg);
                            }
                        }

                        // Track monomorphized class name for this variable (by SymbolId)
                        if let Some(mono_class) = monomorphized_class {
                            if let HirPattern::Variable { symbol, .. } = pattern {
                                self.monomorphized_var_types.insert(*symbol, mono_class);
                            }
                        } else if let HirPattern::Variable { symbol, .. } = pattern {
                            // Fallback: propagate class hint from register (set by
                            // Dynamic/TypeParameter method dispatch) to variable.
                            // This enables disambiguation of chained calls like:
                            //   var guard = mutex.lock(); guard.get(); guard.unlock();
                            if let Some(class_hint) = self.register_class_hints.get(&value_reg) {
                                self.monomorphized_var_types
                                    .insert(*symbol, class_hint.clone());
                            }
                        }

                        // Cross-context interface return-type propagation:
                        // when the RHS register came from an iface
                        // method call whose return type was Dynamic at
                        // TAST but we re-resolved a concrete TypeId at
                        // MIR, override the variable's effective HIR
                        // type so subsequent receiver dispatch sees the
                        // real type (e.g. `Array<Int>`) instead of
                        // routing into the Dynamic-fallback path.
                        //
                        // Propagate when the variable's "declared" type
                        // (`type_hint`) is itself Dynamic/Placeholder/
                        // Unknown — that's the inference fallback TAST
                        // emits when it couldn't pin a real type
                        // cross-context. A real user annotation
                        // (`var x:Array<Int> = …`) pins a concrete
                        // hint, which the assign-side coercion already
                        // handles, so we skip in that case.
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
                                // The override drives object method dispatch
                                // (`effective_receiver_type`). Setting it for a
                                // scalar (Int/Float/Bool) return is meaningless
                                // and disturbs direct uses of the value, so
                                // restrict it to reference-typed returns.
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

                        // The RHS's real source type. When the RHS is an
                        // interface method call, TAST erased its return to
                        // `Dynamic` but MIR recovered the concrete type into
                        // `interface_call_result_types` (the call already
                        // returned a raw concrete value, not a box). The
                        // box/unbox coercions below key off the source type, so
                        // they must see that concrete type — using the erased
                        // `Dynamic` inserts a spurious unbox that reads a raw
                        // pointer as a box header (→ SIGSEGV for object returns
                        // like Tensor), or, for an inferred binding, spuriously
                        // boxes a scalar return (its `var_type` falls back to
                        // the erased Dynamic and disagrees with the recovered
                        // override).
                        let recovered_init_ty =
                            self.interface_call_result_types.get(&value_reg).copied();
                        let init_ty = recovered_init_ty.unwrap_or(init_expr.ty);

                        // Determine the type for the binding. An unannotated
                        // binding infers the recovered concrete type, not the
                        // erased Dynamic. An erased HINT is kept as the
                        // binding type on purpose: for scalars the box is the
                        // representation every Dynamic consumer (string
                        // concat, Std.string) expects, and `maybe_box_value`
                        // passes objects through raw regardless. The
                        // consumers that need the concrete value back
                        // (return-site, comparisons) unbox it there.
                        let var_type = type_hint.or(Some(init_ty));

                        // Auto-box if assigning concrete value to Dynamic variable
                        // Auto-unbox if assigning Dynamic value to concrete variable
                        let final_value = if let Some(target_ty) = var_type {
                            // Try boxing first (concrete → Dynamic)
                            let after_box = self
                                .maybe_box_value(value_reg, init_ty, target_ty)
                                .unwrap_or(value_reg);

                            // Track if boxing actually happened (value was boxed into Dynamic)
                            if after_box != value_reg {
                                if let HirPattern::Variable { symbol, .. } = pattern {
                                    self.boxed_dynamic_symbols.insert(*symbol);
                                }
                            } else if let HirPattern::Variable { symbol, .. } = pattern {
                                // No boxing happened. If the register holds a Ptr(U8) value
                                // from a Dynamic-typed expression, it's a raw anon handle
                                // (e.g., from haxe_ereg_matched_pos_anon). Track it so
                                // field access skips haxe_unbox_reference_ptr.
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

                            // Then try unboxing (Dynamic → concrete)
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

                        // Apply abstract @:from implicit conversion if needed
                        let final_value = if let Some(target_ty) = var_type {
                            self.maybe_abstract_from_convert(final_value, init_ty, target_ty)
                                .unwrap_or(final_value)
                        } else {
                            final_value
                        };

                        // Wrap class instance in interface fat pointer if needed.
                        // Skip for SafeCast results — the SafeCast handler already creates
                        // the fat pointer (or null for failed casts). Cloning a null pointer
                        // from a failed SafeCast would SIGSEGV.
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

                        // @:move strict-move tracking: if the bound class is
                        // annotated `@:move`, mark the destination register as
                        // strict-move-tracked, and if the initializer consumed
                        // another strict-move local, emit a `MarkMoved` for
                        // the source so later reads through it trip CheckLive.
                        {
                            let bound_class_sym = var_type
                                .and_then(|t| self.get_class_symbol(t))
                                .or_else(|| self.get_class_symbol(init_expr.ty));
                            let is_move_class = bound_class_sym
                                .map(|s| self.derive_move_classes.contains(&s))
                                .unwrap_or(false);
                            if is_move_class {
                                self.strict_move_locals.insert(final_value);
                            }
                            // Variable-on-RHS is a move-by-consume site.
                            if let HirExprKind::Variable {
                                symbol: src_symbol, ..
                            } = &init_expr.kind
                            {
                                if let Some(&src_reg) = self.symbol_map.get(src_symbol) {
                                    if self.strict_move_locals.contains(&src_reg) {
                                        let _ = self.builder.build_mark_moved(src_reg);
                                    }
                                }
                            }
                        }

                        // Register heap-allocated value for drop tracking
                        // Only register AutoDrop types (user-defined classes), not RuntimeManaged
                        // extern types (Thread, Channel, Arc, Mutex) or NoDrop types
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
                                        // Class -> interface wrapping aliases the class object.
                                        // Stop tracking source as sole owner to avoid premature
                                        // class frees while interface wrappers still reference it.
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

                        // Ownership transfer: when `var current = n1`, the heap-allocated
                        // value is aliased. Transfer ownership from source to destination
                        // so the drop analysis doesn't free the source while the alias lives.
                        // Skip if interface wrapping already handled the transfer.
                        //
                        // CRITICAL: Skip entirely for @:derive(Copy) bindings. For a Copy
                        // class, `var b = a` does NOT alias `a` — `emit_shallow_copy` already
                        // produced an INDEPENDENT allocation and registered `b` to own it.
                        // Running the transfer here would (1) `register_owned_value(b, a's id)`
                        // while `b` already owns the fresh copy, which `register_owned_value`
                        // treats as a reassignment and FREES the just-created copy mid-function
                        // (use-after-free on the next `b.x = ...`), and (2) re-point `b` at
                        // `a`'s storage, destroying Copy independence and un-tracking `a`.
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
                // Clear temps before expression
                self.temp_heap_values.clear();

                // Shadow-stack location updates are emitted only at call expressions
                // (in the Call handler) and throw statements — not at every statement.
                // This avoids the overhead of N extra extern calls per function.

                let result = self.lower_expression(expr);

                // Free any temporaries created during expression evaluation
                // The result itself (if heap-allocated) is NOT a temporary if it's
                // being used as a return value, but for bare expression statements
                // like method calls for side effects, we should free the result too
                //
                // IMPORTANT: Don't track method call results for drop because:
                // Track heap-allocated results for cleanup:
                // 1. Direct `new` expressions: `new Complex(...)`
                // 2. Method calls that return class instances: `z.mul(z)` returns new Complex
                //
                // The type_needs_drop check ensures we only track Class instances,
                // not primitives, strings, or Dynamic values from runtime functions.
                // NOTE: We do NOT track the top-level expression result as a temp.
                // Functions like Array.push() may store their argument, and freeing
                // the result would create dangling pointers. Only chained method call
                // receivers are safe to free (tracked at the field-access lowering site).
                let _ = result;

                // Free all temporaries
                self.drop_temps();
            }

            HirStatement::Assign { lhs, rhs, op } => {
                // Check if RHS produces a heap-allocated value
                // This includes:
                // 1. Direct `new` expressions (e.g., `new Point(1, 2)`)
                // 2. Method calls returning class instances (e.g., `z.mul(z).add(c)`)
                // We use type_needs_drop to detect any heap-allocated value, not just `new`
                let rhs_type_needs_drop = self.type_needs_drop(rhs.ty);

                // For simple variable assignment, check if we need to free the old value
                // We do this BEFORE evaluating RHS to avoid double-free if RHS reuses the variable
                let lhs_symbol = match lhs {
                    HirLValue::Variable(symbol) => Some(*symbol),
                    _ => None,
                };

                // If assigning a heap-allocated value to a variable that already owns one,
                // free the old value first (handled by register_owned_value)
                // Note: The actual free happens in register_owned_value after we evaluate RHS

                // Clear temps before RHS evaluation
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

                // Anon-literal target hint: assigning a typed-var = { ... }
                // must pass the lhs type down so the writer's slot layout
                // includes optional fields the literal omits.
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

                // If Copy type, emit shallow copy
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
                    // Handle compound assignment if present
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

                    // Store to lvalue
                    if let Some(value) = final_value {
                        // Dynamic↔typed coercion (box concrete→Dynamic, unbox Dynamic→concrete).
                        // Applies to BOTH variable targets (`d = 60`) and field targets
                        // (`obj.v = 60` where v:Dynamic) — the field's declared type comes
                        // from its symbol. Without boxing on the field path, a primitive
                        // written to a Dynamic field is stored raw and read back as a bogus
                        // Dynamic pointer (Std.string garbles / crashes).
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
                        // Recover the RHS's concrete type when it came from an
                        // interface method call: TAST erased the return to
                        // `Dynamic`, but MIR recovered the concrete type into
                        // `interface_call_result_types`. Same rationale as the
                        // Let-binding path — without this the coercion below
                        // unboxes a raw concrete pointer (reading it as a box
                        // header → SIGSEGV for object returns like Tensor).
                        let recovered_rhs_ty =
                            self.interface_call_result_types.get(&value).copied();
                        let rhs_ty = recovered_rhs_ty.unwrap_or(rhs.ty);
                        // For VARIABLE targets whose declared type is itself
                        // inference-erased, the recovered type also overrides
                        // the coercion target — mirroring the Let-binding
                        // policy so a reassigned local keeps the raw concrete
                        // representation its binding established. Field
                        // targets keep their declared type: a Dynamic field
                        // may be deliberate, and its readers use the box
                        // protocol.
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
                            // RAYZOR_ASSIGN_DEBUG=1: names the variable whose
                            // declared type drives the box/unbox coercion. A
                            // primitive local showing target=Dynamic here is
                            // the shape that boxes a constant into a phi.
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

                        // Apply abstract @:from implicit conversion if needed
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

                        // Numeric promotion at assignment: `var f:Float = i;`
                        // where i is an Int must sitofp-cast the int bits,
                        // not reinterpret them as f64. Surfaces as
                        // `case IntV(x): x` in a Float-returning switch
                        // returning ~5e-322 — the switch-as-expression
                        // desugar inits the temp to 0.0 (F64 register), but
                        // the case-body assigns the i64 bit pattern of an
                        // Int-bound pattern variable straight into an
                        // f64-typed phi with no int→float instruction.
                        //
                        // Symbol-table lookup of the lhs's TypeId is
                        // unreliable for synthetic temps from desugars —
                        // `gen_temp_var()` allocates a SymbolId without
                        // registering it. Fall back to the lhs register's
                        // tracked IR type, which the Let-init already set
                        // (e.g. 0.0 → F64).
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

                        // Register heap-allocated value for drop tracking.
                        // Only track when the RHS actually creates a NEW allocation (New or Call).
                        // Field access and variable reads produce borrowed references that must
                        // NOT be freed — they point into existing objects.
                        // Only `new Foo(...)` creates guaranteed owned allocations.
                        // Call results (method returns) are ambiguous — they may return
                        // references to existing objects (e.g., satisfy() returns an existing
                        // constraint). Treating Call as owned causes use-after-free in while
                        // loops where exit_drop_scope frees the value before the back edge.
                        let rhs_is_owned_allocation =
                            rhs_was_copied || matches!(&rhs.kind, HirExprKind::New { .. });

                        // When assigning to a Field/ArrayIndex lvalue, the RHS value escapes
                        // to another object. If the RHS is a variable that we're tracking as
                        // an owned heap allocation, we must STOP tracking it — the pointer now
                        // lives inside another object and must NOT be freed at scope exit.
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
                                // Class -> interface wrapping aliases the class object.
                                // Don't keep source as sole owner.
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
                                        // RHS is a borrowed reference (field access, variable, etc.)
                                        // → free old owned value, STOP tracking
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

                    // Free any intermediate temporaries created during RHS evaluation
                    // These are values like the result of z.mul(z) in z = z.mul(z).add(c)
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
                let ret_value = value.as_ref().and_then(|e| {
                    debug!(
                        "[Return]: Lowering return expression, expr kind: {:?}",
                        std::mem::discriminant(&e.kind)
                    );
                    // Anon-literal target hint: if returning an ObjectLiteral
                    // and the function's declared return is a wider typedef,
                    // pass it down so optional fields land in the slot table.
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
                        // Inverse: function returns primitive `T` but expression
                        // produces `Null<T>` (boxed `DynamicValue*`). Unbox via
                        // the runtime helper. Without this the boxed pointer
                        // gets returned as if it were the primitive — the
                        // caller reads the pointer's address as the value and
                        // sees garbage. Common case: returning a value pulled
                        // from `StringMap<Int>.get(k)` (typed `Null<Int>`)
                        // from a function declared `:Int` after a manual
                        // null-coalesce ternary.
                        if let Some(unboxed) =
                            self.maybe_unbox_optional_for_target(val, e.ty, fn_ret_ty)
                        {
                            return Some(unboxed);
                        }
                        // Interface return wrapping: if the function's return
                        // type is an interface and the expression has the
                        // implementing class type, wrap the raw class pointer
                        // in a fat pointer so the caller can vtable-dispatch.
                        // The auto-wrap in `lower_expression` only fires when
                        // `expr.ty` is interface — for `return classInstance`
                        // when the declared return type is the interface,
                        // `expr.ty` stays as the class, so we wrap here.
                        let (wrapped, did_wrap) =
                            self.maybe_wrap_for_interface(val, e.ty, fn_ret_ty);
                        if did_wrap {
                            return Some(wrapped);
                        }
                        // Cross-module: the returned value's type may not resolve
                        // to a Class here (arrives Unknown/Placeholder), so
                        // `maybe_wrap_for_interface` can't wrap `return newC()`
                        // for an interface return. Recover the class by name and
                        // wrap via the name-based fat-ptr build (mirrors the
                        // call-arg path). Without this, a raw class object is
                        // returned where the caller expects an interface fat
                        // pointer (`LlamaArch.build():Module` → `new LlamaModel()`).
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

                        // Numeric promotion: `function foo():Float { return x; }`
                        // where x is an Int (e.g. bound from an enum variant
                        // field, the case-arm result type-coerces to the
                        // function's return type). The Let/Assign handlers do
                        // this via `build_cast`, but Return doesn't — so the
                        // raw int bits got reinterpreted as a float at the
                        // callee/caller boundary. Surfaces as garbage like
                        // 5e-322 from `case IntV(x): x;` when the function
                        // returns Float. Cover Int→Float and Float→Int both
                        // ways for parity with assignment behaviour.
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
                // debug!("Return instruction built");
            }

            HirStatement::Break(label) => {
                if let Some(loop_ctx) = self.find_loop_context(label.as_ref()) {
                    let break_block = loop_ctx.break_block;
                    let exit_phi_nodes = loop_ctx.exit_phi_nodes.clone();

                    // Get the current block before branching
                    let current_block = self.builder.current_block().unwrap();

                    // Add incoming edges to exit block phi nodes with current symbol values
                    for (symbol_id, exit_phi_reg) in &exit_phi_nodes {
                        // Get the current value of this symbol
                        let current_value = if let Some(&reg) = self.symbol_map.get(symbol_id) {
                            reg
                        } else {
                            // If symbol not in map, use the phi register itself (shouldn't happen)
                            *exit_phi_reg
                        };

                        // Add incoming edge from current block to exit phi
                        self.builder.add_phi_incoming(
                            break_block,
                            *exit_phi_reg,
                            current_block,
                            current_value,
                        );
                    }

                    // CRITICAL: Save drop state before cleanup. The break path emits Free
                    // instructions and modifies drop_scope_stack/owned_heap_values, but these
                    // mutations must NOT persist for the non-break code path. Without this
                    // save/restore, the scope stack is corrupted: the normal loop exit code
                    // (exit_drop_scope at the back-edge) pops the WRONG scope, causing
                    // variables from outer scopes to be freed prematurely → double-free.
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

                    // Add incoming edges for update block phi nodes (if any).
                    // This is analogous to how break adds edges for exit phi nodes.
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

                    // CRITICAL: Save drop state before cleanup (same reason as break above).
                    // Continue emits Free instructions for the current scope, but these
                    // mutations must not persist for the non-continue code path.
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

                    // Call rayzor_throw_typed(exception_value, type_id).
                    // For class throws, read the RUNTIME type id straight from the
                    // object header[0] — it already holds the deterministic class
                    // id that `runtime_type_id` produces for typed catches, so the
                    // thrown id and the catch's expected id are the SAME encoding.
                    // (A stale `+1000` here double-encoded it, so a derived class
                    // thrown as `throw new MyException(...)` reported id N+1000
                    // while `catch (e:MyException)` expected N → the specific catch
                    // never matched and a later `catch (e:Exception)` caught it.)
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
                // debug!("About to call lower_if_statement, has_else={}", else_branch.is_some());
                self.lower_if_statement(condition, then_branch, else_branch.as_ref());
                // debug!("Returned from lower_if_statement");
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
        // Process all statements
        for stmt in block.statements.iter() {
            self.lower_statement(stmt);

            // Check if any variables have their last use at this statement
            // and emit Free for them (lifetime-based drop)
            self.check_drop_points_after_statement();

            // Increment statement index for drop point tracking
            self.current_stmt_index += 1;
        }

        // Process trailing expression if present
        if let Some(expr) = &block.expr {
            let _result = self.lower_expression(expr);
            // The result could be used for implicit returns
        }
    }

    /// Lower a HIR block expression to MIR, returning the trailing expression's value
    pub(crate) fn lower_block_expr(&mut self, block: &HirBlock) -> Option<IrId> {
        // Process all statements
        for stmt in block.statements.iter() {
            self.lower_statement(stmt);
            self.check_drop_points_after_statement();
            self.current_stmt_index += 1;
        }

        // Return trailing expression value if present
        if let Some(expr) = &block.expr {
            self.lower_expression(expr)
        } else {
            None
        }
    }
}
