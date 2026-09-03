//! Iteration handles: how a value reaches a slot typed `Iterable<T>`/`Iterator<T>`.
//!
//! `Iterator<T>` and `Iterable<T>` are structural — `StdTypes.hx` declares them as
//! anonymous structures of methods, not classes — so a value typed as either
//! carries nothing a call can dispatch through. An `Array` in particular is a
//! bare `HaxeArray`, whose first word is its data pointer rather than a type id,
//! so nothing about it can be recovered at run time.
//!
//! What the concrete type IS, though, is known where the value crosses into the
//! protocol-typed slot. A handle is built there, carrying the collection
//! alongside the three entry points a loop needs:
//!
//! ```text
//! [0]  tag       ITER_HANDLE_TAG
//! [8]  obj       the collection, or the iterator itself
//! [16] iterator  returns obj's iterator; null when obj already IS the iterator
//! [24] hasNext   on the iterator that slot 16 returns
//! [32] next      on that same iterator
//! ```
//!
//! Slots 24 and 32 describe the iterator slot 16 produces, which is why the
//! iterator is fetched at loop entry rather than at the boundary: the same
//! `Iterable` value then iterates from the start every time it is looped over.
//!
//! The tag is what makes partial coverage safe. A receiver that reaches a loop
//! without passing a boundary has no handle, and the loop checks the tag before
//! reading any slot, so it iterates zero times instead of calling through a
//! collection's length field.

use super::*;

/// Marks an allocation as an iteration handle. Loops read it before trusting any
/// other slot, so it must not collide with a plausible first word of anything
/// else that can arrive in a protocol-typed slot.
pub(crate) const ITER_HANDLE_TAG: i64 = 0x52_5A_49_54_45_52_00_01;

/// Total bytes: tag, object, and three entry points.
const ITER_HANDLE_SIZE: u64 = 40;

/// Which protocol a type names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IterProtocol {
    /// Declares `iterator()`.
    Iterable,
    /// Declares `hasNext()` and `next()`.
    Iterator,
}

/// How a concrete value supplies the protocol.
#[derive(Debug, Clone)]
pub(crate) enum IterSource {
    /// A `HaxeArray`, iterated through the array iterator wrappers.
    Array,
    /// A class declaring `iterator()`; the iterator's own class is named so
    /// `hasNext`/`next` resolve against it rather than against the collection.
    ClassIterable {
        class_sym: SymbolId,
        iterator_class: Option<SymbolId>,
    },
    /// A class that is itself an iterator.
    ClassIterator { class_sym: SymbolId },
}

impl<'a> HirToMirContext<'a> {
    /// The protocol `type_id` names, if it names one.
    ///
    /// Matched by name: `Iterator` and `Iterable` are the language's own two
    /// protocol shapes, so their slots are fixed rather than derived from a
    /// canonical ordering over arbitrary structures.
    pub(crate) fn iter_protocol_of(&self, type_id: TypeId) -> Option<IterProtocol> {
        let type_table = self.type_table;
        let mut tid = type_id;
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(tid) {
                return None;
            }
            let ty = type_table.get(tid)?;
            let sym = match &ty.kind {
                TypeKind::Class { symbol_id, .. }
                | TypeKind::Interface { symbol_id, .. }
                | TypeKind::TypeAlias { symbol_id, .. } => Some(*symbol_id),
                TypeKind::GenericInstance { base_type, .. } => {
                    tid = *base_type;
                    continue;
                }
                _ => None,
            };
            let name = sym
                .and_then(|s| self.symbol_table.get_symbol(s))
                .and_then(|s| self.string_interner.get(s.name));
            match name {
                Some("Iterator") => return Some(IterProtocol::Iterator),
                Some("Iterable") => return Some(IterProtocol::Iterable),
                _ => {}
            }
            // A typedef names the protocol at its own symbol; keep walking only
            // through the alias chain, since anything else is a different type.
            match &ty.kind {
                TypeKind::TypeAlias { target_type, .. } => tid = *target_type,
                _ => return None,
            }
        }
    }

    /// Classify the concrete type a value has where it crosses into a
    /// protocol-typed slot. `None` means no handle can be built, and the value
    /// is left exactly as it is today.
    pub(crate) fn iter_source_of(&self, type_id: TypeId) -> Option<IterSource> {
        let type_table = self.type_table;
        let mut tid = type_id;
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(tid) {
                return None;
            }
            let ty = type_table.get(tid)?;
            match &ty.kind {
                TypeKind::Array { .. } => return Some(IterSource::Array),
                TypeKind::Class { symbol_id, .. } => {
                    let class_sym = *symbol_id;
                    let name = self
                        .symbol_table
                        .get_symbol(class_sym)
                        .and_then(|s| self.string_interner.get(s.name));
                    if name == Some("Array") {
                        return Some(IterSource::Array);
                    }
                    let has = |m: &str| {
                        let interned = self.string_interner.intern(m);
                        self.class_method_symbols
                            .contains_key(&(class_sym, interned))
                            || self
                                .class_method_by_name
                                .contains_key(&(class_sym, interned))
                    };
                    if has("hasNext") && has("next") {
                        return Some(IterSource::ClassIterator { class_sym });
                    }
                    if has("iterator") {
                        return Some(IterSource::ClassIterable {
                            class_sym,
                            iterator_class: self.iterator_class_of(class_sym),
                        });
                    }
                    return None;
                }
                TypeKind::TypeAlias { target_type, .. } => tid = *target_type,
                TypeKind::GenericInstance { base_type, .. } => tid = *base_type,
                _ => return None,
            }
        }
    }

    /// The class a collection's `iterator()` returns, read off the method's own
    /// declared return type.
    fn iterator_class_of(&self, class_sym: SymbolId) -> Option<SymbolId> {
        let iterator_name = self.string_interner.intern("iterator");
        let method_sym = self
            .class_method_symbols
            .get(&(class_sym, iterator_name))
            .copied()
            .or_else(|| {
                self.class_method_by_name
                    .get(&(class_sym, iterator_name))
                    .copied()
            })?;
        let ty = self
            .type_table
            .get(self.symbol_table.get_symbol(method_sym)?.type_id)?;
        let TypeKind::Function { return_type, .. } = &ty.kind else {
            return None;
        };
        match &self.type_table.get(*return_type)?.kind {
            TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
            _ => None,
        }
    }

    /// The entry point for one protocol method on one class, as a thunk that can
    /// sit in a handle slot.
    ///
    /// Slots hold thunks rather than raw methods because indirect calls use the
    /// closure ABI on every backend, so a bare `(this, args)` method placed in a
    /// slot would read its receiver as the environment.
    fn iter_thunk_for_class(&mut self, class_sym: SymbolId, method: &str) -> Option<IrFunctionId> {
        let interned = self.string_interner.intern(method);
        let method_sym = self
            .class_method_symbols
            .get(&(class_sym, interned))
            .copied()
            .or_else(|| {
                self.class_method_by_name
                    .get(&(class_sym, interned))
                    .copied()
            })?;
        let func_id = self
            .function_map
            .get(&method_sym)
            .copied()
            .or_else(|| self.external_function_map.get(&method_sym).copied())?;
        self.ensure_vtable_dispatch_thunk(func_id).or(Some(func_id))
    }

    /// The entry point for one of the array iterator wrappers.
    fn iter_thunk_for_wrapper(
        &mut self,
        name: &str,
        params: Vec<IrType>,
        ret: IrType,
    ) -> Option<IrFunctionId> {
        let func_id = self.register_stdlib_mir_forward_ref(name, params, ret);
        self.ensure_vtable_dispatch_thunk(func_id).or(Some(func_id))
    }

    /// Build a handle over `obj_reg`, whose concrete type is `source`.
    ///
    /// `None` leaves the caller's value untouched, so a source whose entry points
    /// cannot all be named produces no handle rather than a half-filled one.
    pub(crate) fn build_iter_handle(&mut self, obj_reg: IrId, source: &IterSource) -> Option<IrId> {
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
        let (iterator_fn, has_next_fn, next_fn) = match source {
            IterSource::Array => (
                Some(self.iter_thunk_for_wrapper(
                    "array_iterator",
                    vec![ptr_void.clone()],
                    ptr_void.clone(),
                )?),
                self.iter_thunk_for_wrapper(
                    "ArrayIterator_hasNext",
                    vec![ptr_void.clone()],
                    IrType::I32,
                )?,
                self.iter_thunk_for_wrapper(
                    "ArrayIterator_next",
                    vec![ptr_void.clone()],
                    IrType::I64,
                )?,
            ),
            IterSource::ClassIterable {
                class_sym,
                iterator_class,
            } => {
                let it_class = (*iterator_class)?;
                (
                    Some(self.iter_thunk_for_class(*class_sym, "iterator")?),
                    self.iter_thunk_for_class(it_class, "hasNext")?,
                    self.iter_thunk_for_class(it_class, "next")?,
                )
            }
            IterSource::ClassIterator { class_sym } => (
                None,
                self.iter_thunk_for_class(*class_sym, "hasNext")?,
                self.iter_thunk_for_class(*class_sym, "next")?,
            ),
        };

        let malloc_fn = self.get_or_register_extern_function(
            "malloc",
            vec![IrType::U64],
            IrType::Ptr(Box::new(IrType::U8)),
        );
        let size_reg = self.builder.build_const(IrValue::U64(ITER_HANDLE_SIZE))?;
        let handle = self.builder.build_call_direct(
            malloc_fn,
            vec![size_reg],
            IrType::Ptr(Box::new(IrType::U8)),
        )?;

        let tag = self.builder.build_const(IrValue::I64(ITER_HANDLE_TAG))?;
        self.builder.build_store(handle, tag);

        let obj_as_i64 = {
            let obj_ty = self
                .builder
                .get_register_type(obj_reg)
                .unwrap_or(IrType::I64);
            if matches!(obj_ty, IrType::Ptr(_)) {
                self.builder.build_bitcast(obj_reg, IrType::I64)?
            } else {
                obj_reg
            }
        };
        self.store_handle_slot_value(handle, 8, obj_as_i64)?;

        // A source that is already an iterator has no `iterator()` to call, and
        // the loop reads a null slot as "obj is the iterator".
        let iterator_val = match iterator_fn {
            Some(f) => self.builder.build_function_ref(f)?,
            None => self.builder.build_const(IrValue::I64(0))?,
        };
        self.store_handle_slot_value(handle, 16, iterator_val)?;
        let hn = self.builder.build_function_ref(has_next_fn)?;
        self.store_handle_slot_value(handle, 24, hn)?;
        let nx = self.builder.build_function_ref(next_fn)?;
        self.store_handle_slot_value(handle, 32, nx)?;

        Some(handle)
    }

    fn store_handle_slot_value(&mut self, handle: IrId, offset: i64, value: IrId) -> Option<()> {
        let off = self.builder.build_const(IrValue::I64(offset))?;
        let slot = self
            .builder
            .build_ptr_add(handle, off, IrType::Ptr(Box::new(IrType::U8)))?;
        self.builder.build_store(slot, value);
        Some(())
    }

    /// Whether a protocol type's arguments are still type parameters.
    fn protocol_mentions_type_parameter(&self, type_id: TypeId) -> bool {
        let type_table = self.type_table;
        let mut tid = type_id;
        let mut seen = BTreeSet::new();
        loop {
            if !seen.insert(tid) {
                return false;
            }
            let Some(ty) = type_table.get(tid) else {
                return false;
            };
            let args = match &ty.kind {
                TypeKind::Class { type_args, .. } => type_args.clone(),
                TypeKind::GenericInstance {
                    base_type,
                    type_args,
                    ..
                } => {
                    if type_args.iter().any(|a| self.is_type_parameter(*a)) {
                        return true;
                    }
                    tid = *base_type;
                    continue;
                }
                TypeKind::TypeAlias { target_type, .. } => {
                    tid = *target_type;
                    continue;
                }
                _ => return false,
            };
            return args.iter().any(|a| self.is_type_parameter(*a));
        }
    }

    fn is_type_parameter(&self, type_id: TypeId) -> bool {
        matches!(
            self.type_table.get(type_id).map(|t| &t.kind),
            Some(TypeKind::TypeParameter { .. }) | Some(TypeKind::Placeholder { .. })
        )
    }

    /// Wrap `value_reg` for a slot typed `Iterable<T>`/`Iterator<T>`.
    ///
    /// Returns `None` when the target names no protocol, when the source is
    /// already a handle, or when the source cannot be classified — in each case
    /// the caller keeps the value it had.
    pub(crate) fn maybe_wrap_for_iter_protocol(
        &mut self,
        value_reg: IrId,
        source_ty: TypeId,
        target_ty: TypeId,
    ) -> Option<IrId> {
        self.iter_protocol_of(target_ty)?;
        // A protocol still carrying an unbound type parameter belongs to a generic
        // template, whose specialisation is chosen from the argument's own type.
        // Handing the callee a handle there hides the concrete type that choice
        // reads, so such a call keeps the shape it has today.
        if self.protocol_mentions_type_parameter(target_ty) {
            return None;
        }
        // A value already typed as the protocol carries a handle if it has one;
        // rewrapping would bury it a level deeper.
        if self.iter_protocol_of(source_ty).is_some() {
            return None;
        }
        let source = self.iter_source_of(source_ty)?;
        self.build_iter_handle(value_reg, &source)
    }
}

impl<'a> HirToMirContext<'a> {
    /// Lower `for (x in it)` where `it` is typed `Iterable<T>`/`Iterator<T>`.
    ///
    /// Returns false when the receiver names no protocol, leaving the caller to
    /// lower the loop as it does today.
    ///
    /// The receiver is only assumed to be a handle after its tag matches. A value
    /// that reached here without crossing a boundary is some other object, whose
    /// first word is its own data rather than the tag, so the loop is skipped
    /// instead of calling through whatever that word holds.
    pub(crate) fn try_lower_for_in_iter_handle(
        &mut self,
        pattern: &HirPattern,
        iter_expr: &HirExpr,
        body: &HirBlock,
        label: Option<&SymbolId>,
    ) -> bool {
        if self.iter_protocol_of(iter_expr.ty).is_none() {
            return false;
        }
        // Inside a generic template the element type is still a parameter, and the
        // body is lowered once for a receiver no call site has supplied yet. The
        // loop keeps the shape it has there and the specialisations carry the
        // handle, so a template is never lowered against a receiver it cannot have.
        if self.protocol_mentions_type_parameter(iter_expr.ty) {
            return false;
        }
        let Some(handle_raw) = self.lower_expression(iter_expr) else {
            return true;
        };
        let ptr_void = IrType::Ptr(Box::new(IrType::Void));
        let ptr_u8 = IrType::Ptr(Box::new(IrType::U8));

        let handle = {
            let ty = self
                .builder
                .get_register_type(handle_raw)
                .unwrap_or(IrType::I64);
            if matches!(ty, IrType::Ptr(_)) {
                handle_raw
            } else {
                match self.builder.build_bitcast(handle_raw, ptr_u8.clone()) {
                    Some(p) => p,
                    None => return true,
                }
            }
        };

        // Variables the body assigns cross blocks through stack slots, the way
        // every other loop here carries them.
        let mut modified_vars = {
            let mut modified = BTreeSet::new();
            for stmt in &body.statements {
                self.find_modified_variables_in_statement(stmt, &mut modified);
            }
            modified
        };
        // `n++` writes `n`, and the shared walker records only assignments, so a
        // counter incremented in the body would otherwise leave the loop through a
        // register the exit block cannot see.
        for stmt in &body.statements {
            collect_incremented_symbols(stmt, &mut modified_vars);
        }
        let mut var_slots: BTreeMap<SymbolId, (IrId, IrType)> = BTreeMap::new();
        for sym in &modified_vars {
            if let Some(&current_reg) = self.symbol_map.get(sym) {
                let ty = self
                    .builder
                    .get_register_type(current_reg)
                    .unwrap_or(IrType::I64);
                if let Some(slot) = self.builder.build_alloc(ty.clone(), None) {
                    self.builder.build_store(slot, current_reg);
                    var_slots.insert(*sym, (slot, ty));
                }
            }
        }

        let (Some(obj_slot), Some(hn_slot), Some(nx_slot)) = (
            self.builder.build_alloc(IrType::I64, None),
            self.builder.build_alloc(IrType::I64, None),
            self.builder.build_alloc(IrType::I64, None),
        ) else {
            return true;
        };

        let (
            Some(setup_block),
            Some(call_iter_block),
            Some(cond_block),
            Some(body_block),
            Some(exit_block),
        ) = (
            self.builder.create_block(),
            self.builder.create_block(),
            self.builder.create_block(),
            self.builder.create_block(),
            self.builder.create_block(),
        )
        else {
            return true;
        };

        // Tag check. A null receiver never reaches the load.
        let Some(handle_i) = self.builder.build_bitcast(handle, IrType::I64) else {
            return true;
        };
        let (Some(zero), Some(tag_const)) = (
            self.builder.build_const(IrValue::I64(0)),
            self.builder.build_const(IrValue::I64(ITER_HANDLE_TAG)),
        ) else {
            return true;
        };
        let Some(not_null) = self.builder.build_cmp(CompareOp::Ne, handle_i, zero) else {
            return true;
        };
        let Some(read_tag_block) = self.builder.create_block() else {
            return true;
        };
        self.builder
            .build_cond_branch(not_null, read_tag_block, exit_block);

        self.builder.switch_to_block(read_tag_block);
        let Some(tag) = self.builder.build_load(handle, IrType::I64) else {
            return true;
        };
        let Some(is_handle) = self.builder.build_cmp(CompareOp::Eq, tag, tag_const) else {
            return true;
        };
        self.builder
            .build_cond_branch(is_handle, setup_block, exit_block);

        // Setup: read the object and the three entry points out of the handle.
        self.builder.switch_to_block(setup_block);
        let mut slot_of = |ctx: &mut Self, off: i64| -> Option<IrId> {
            let o = ctx.builder.build_const(IrValue::I64(off))?;
            let p = ctx.builder.build_ptr_add(handle, o, ptr_u8.clone())?;
            ctx.builder.build_load(p, IrType::I64)
        };
        let (Some(obj0), Some(it_fn), Some(hn_fn), Some(nx_fn)) = (
            slot_of(self, 8),
            slot_of(self, 16),
            slot_of(self, 24),
            slot_of(self, 32),
        ) else {
            return true;
        };
        self.builder.build_store(obj_slot, obj0);
        self.builder.build_store(hn_slot, hn_fn);
        self.builder.build_store(nx_slot, nx_fn);
        let Some(zero2) = self.builder.build_const(IrValue::I64(0)) else {
            return true;
        };
        let Some(needs_iter) = self.builder.build_cmp(CompareOp::Ne, it_fn, zero2) else {
            return true;
        };
        self.builder
            .build_cond_branch(needs_iter, call_iter_block, cond_block);

        // A collection hands over its iterator here, at loop entry, so looping the
        // same value twice starts from the beginning both times.
        self.builder.switch_to_block(call_iter_block);
        let iter_sig = IrType::Function {
            params: vec![ptr_void.clone()],
            return_type: Box::new(ptr_void.clone()),
            varargs: false,
        };
        if let Some(it_obj) = self
            .builder
            .build_call_indirect(it_fn, vec![obj0], iter_sig)
        {
            let stored = self
                .builder
                .build_bitcast(it_obj, IrType::I64)
                .unwrap_or(it_obj);
            self.builder.build_store(obj_slot, stored);
        }
        self.builder.build_branch(cond_block);

        self.loop_stack.push(LoopContext {
            continue_block: cond_block,
            break_block: exit_block,
            label: label.cloned(),
            exit_phi_nodes: BTreeMap::new(),
            continue_phi_nodes: BTreeMap::new(),
        });

        self.builder.switch_to_block(cond_block);
        let bool_sig = IrType::Function {
            params: vec![ptr_void.clone()],
            return_type: Box::new(IrType::Bool),
            varargs: false,
        };
        let (Some(obj_c), Some(hn_c)) = (
            self.builder.build_load(obj_slot, IrType::I64),
            self.builder.build_load(hn_slot, IrType::I64),
        ) else {
            self.loop_stack.pop();
            return true;
        };
        let Some(more) = self
            .builder
            .build_call_indirect(hn_c, vec![obj_c], bool_sig)
        else {
            self.loop_stack.pop();
            return true;
        };
        self.builder.build_cond_branch(more, body_block, exit_block);

        self.builder.switch_to_block(body_block);
        for (sym, (slot, ty)) in &var_slots {
            if let Some(loaded) = self.builder.build_load(*slot, ty.clone()) {
                self.symbol_map.insert(*sym, loaded);
            }
        }
        let next_sig = IrType::Function {
            params: vec![ptr_void],
            return_type: Box::new(IrType::I64),
            varargs: false,
        };
        let (Some(obj_b), Some(nx_b)) = (
            self.builder.build_load(obj_slot, IrType::I64),
            self.builder.build_load(nx_slot, IrType::I64),
        ) else {
            self.loop_stack.pop();
            return true;
        };
        if let Some(value) = self
            .builder
            .build_call_indirect(nx_b, vec![obj_b], next_sig)
        {
            if let HirPattern::Variable { symbol, .. } = pattern {
                self.symbol_map.insert(*symbol, value);
            }
        }

        self.loop_carried_symbols
            .push(var_slots.keys().copied().collect());
        self.enter_drop_scope();
        self.lower_block(body);
        if !self.is_terminated() {
            for (sym, (slot, _ty)) in &var_slots {
                if let Some(&current_reg) = self.symbol_map.get(sym) {
                    self.builder.build_store(*slot, current_reg);
                }
            }
            self.exit_drop_scope();
            self.builder.build_branch(cond_block);
        } else {
            self.exit_drop_scope();
        }
        self.loop_carried_symbols.pop();
        self.loop_stack.pop();

        self.builder.switch_to_block(exit_block);
        for (sym, (slot, ty)) in &var_slots {
            if let Some(loaded) = self.builder.build_load(*slot, ty.clone()) {
                self.symbol_map.insert(*sym, loaded);
            }
        }
        true
    }
}

/// Symbols an expression statement increments or decrements.
///
/// Written here rather than added to the shared walker because every loop form
/// depends on that one, and widening what it reports changes how they all carry
/// their variables.
fn collect_incremented_symbols(stmt: &HirStatement, out: &mut BTreeSet<SymbolId>) {
    fn walk_expr(e: &HirExpr, out: &mut BTreeSet<SymbolId>) {
        match &e.kind {
            HirExprKind::Unary { op, operand } => {
                if matches!(
                    op,
                    HirUnaryOp::PostIncr
                        | HirUnaryOp::PreIncr
                        | HirUnaryOp::PostDecr
                        | HirUnaryOp::PreDecr
                ) {
                    if let HirExprKind::Variable { symbol, .. } = &operand.kind {
                        out.insert(*symbol);
                    }
                }
                walk_expr(operand, out);
            }
            HirExprKind::Binary { lhs, rhs, .. } => {
                walk_expr(lhs, out);
                walk_expr(rhs, out);
            }
            HirExprKind::Call { args, .. } => {
                for a in args {
                    walk_expr(a, out);
                }
            }
            HirExprKind::If {
                condition,
                then_expr,
                else_expr,
            } => {
                walk_expr(condition, out);
                walk_expr(then_expr, out);
                walk_expr(else_expr, out);
            }
            HirExprKind::Block(block) => {
                for s in &block.statements {
                    collect_incremented_symbols(s, out);
                }
            }
            _ => {}
        }
    }
    match stmt {
        HirStatement::Expr(e) => walk_expr(e, out),
        HirStatement::Let { init: Some(e), .. } => walk_expr(e, out),
        HirStatement::Assign { rhs, .. } => walk_expr(rhs, out),
        HirStatement::If {
            condition,
            then_branch,
            else_branch,
        } => {
            walk_expr(condition, out);
            for s in &then_branch.statements {
                collect_incremented_symbols(s, out);
            }
            if let Some(b) = else_branch {
                for s in &b.statements {
                    collect_incremented_symbols(s, out);
                }
            }
        }
        HirStatement::While { body, .. } | HirStatement::DoWhile { body, .. } => {
            for s in &body.statements {
                collect_incremented_symbols(s, out);
            }
        }
        _ => {}
    }
}
