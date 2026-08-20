//! Use-after-move analysis over the MIR control-flow graph.
//!
//! The property being checked is about *bindings*, so the analysis is keyed on
//! `SymbolId`. Registers cannot stand in for bindings: `var d = b` needs no
//! cast, so `d` and `b` are the same register, and a register-keyed check
//! cannot tell a read of `d` from a read of the binding that was moved into it.
//!
//! Events are recorded during lowering, where the symbol is still in hand, and
//! replayed here against the finished CFG. Reading the instruction stream
//! instead would not work: `MarkMoved` and `CheckLive` both report their
//! operand as a use, so a move would count as a use of itself at its own
//! program point and cancel out — which is exactly why a same-location check
//! sees nothing.
//!
//! Enrolment happens at record time. A binding is recorded only when its class
//! carries `@:move` (and not `@:shared`), so ordinary Haxe produces no events
//! at all and reaches this file with nothing to analyse.

use crate::ir::{IrBlockId, IrControlFlowGraph};
use crate::tast::{SourceLocation, SymbolId};
use std::collections::{BTreeMap, BTreeSet};

use super::HirToMirContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoveEventKind {
    /// The binding starts (or restarts) owning a value. Kills any prior move.
    Bind,
    /// The value is transferred away. Checked against the incoming state
    /// before it is applied, so moving an already-moved value is itself a
    /// violation — that is what makes `f(a); g(a)` fire.
    Move,
    /// The binding is observed without being transferred.
    Read,
}

#[derive(Debug, Clone)]
pub(crate) struct MoveEvent {
    pub(crate) block: IrBlockId,
    /// Program order across the whole function. Orders events inside a block
    /// and picks the earliest move when two paths join.
    pub(crate) order: u32,
    pub(crate) symbol: SymbolId,
    pub(crate) kind: MoveEventKind,
    pub(crate) location: SourceLocation,
}

/// Per-binding lattice value. Absence from the state map is the bottom
/// element: not moved, no witness.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MovedState {
    moved: bool,
    witness: Option<(u32, SourceLocation)>,
}

type State = BTreeMap<SymbolId, MovedState>;

/// Join `src` into `dst`, returning whether `dst` grew.
///
/// `moved` joins by OR — a value moved on any incoming path may be moved here.
/// The witness joins by earliest `order`, which is what makes it monotone: a
/// "first seen" witness that could move later in program order would let the
/// fixpoint oscillate instead of settling.
fn join_into(dst: &mut State, src: &State) -> bool {
    let mut changed = false;
    for (sym, s) in src {
        if !s.moved && s.witness.is_none() {
            continue;
        }
        match dst.get_mut(sym) {
            None => {
                dst.insert(*sym, s.clone());
                changed = true;
            }
            Some(d) => {
                if s.moved && !d.moved {
                    d.moved = true;
                    changed = true;
                }
                let take = match (&d.witness, &s.witness) {
                    (None, Some(_)) => true,
                    (Some(dw), Some(sw)) => sw.0 < dw.0,
                    _ => false,
                };
                if take {
                    d.witness = s.witness.clone();
                    changed = true;
                }
            }
        }
    }
    changed
}

/// One reported violation: the binding, where it was moved, where it was used.
struct Violation {
    symbol: SymbolId,
    moved_at: SourceLocation,
    used_at: SourceLocation,
}

/// Run the events of one block against `state`, collecting violations when
/// `report` is set. The transfer is identical in both passes so that what is
/// reported is exactly what the fixpoint converged on.
fn transfer(
    events: &[MoveEvent],
    indices: &[usize],
    state: &mut State,
    report: Option<&mut Vec<Violation>>,
) {
    let mut sink = report;
    for &i in indices {
        let ev = &events[i];
        match ev.kind {
            MoveEventKind::Bind => {
                state.remove(&ev.symbol);
            }
            MoveEventKind::Move | MoveEventKind::Read => {
                let already = state.get(&ev.symbol).filter(|s| s.moved).cloned();
                if let Some(prior) = &already {
                    if let Some(out) = sink.as_deref_mut() {
                        if let Some((_, moved_at)) = &prior.witness {
                            out.push(Violation {
                                symbol: ev.symbol,
                                moved_at: *moved_at,
                                used_at: ev.location,
                            });
                        }
                    }
                }
                if ev.kind == MoveEventKind::Move {
                    // Keep the earliest witness, so a repeat move is reported
                    // against the move that first consumed the value. Taking
                    // the minimum rather than "keep whatever arrived" is what
                    // keeps the transfer monotone: an incoming witness only
                    // ever moves earlier, so the outgoing one must too, or the
                    // fixpoint below can oscillate instead of settling.
                    let witness = match already.as_ref().and_then(|s| s.witness) {
                        Some(prev) if prev.0 <= ev.order => Some(prev),
                        _ => Some((ev.order, ev.location)),
                    };
                    state.insert(
                        ev.symbol,
                        MovedState {
                            moved: true,
                            witness,
                        },
                    );
                }
            }
        }
    }
}

impl HirToMirContext<'_> {
    /// Record that `symbol` holds a `@:move` value from here on. Only enrolled
    /// symbols produce move or read events.
    pub(crate) fn enroll_move_symbol(&mut self, symbol: SymbolId) {
        self.move_symbols.insert(symbol);
    }

    /// True when `symbol` is an enrolled `@:move` binding in this function.
    pub(crate) fn is_move_symbol(&self, symbol: SymbolId) -> bool {
        self.move_symbols.contains(&symbol)
    }

    /// Drop the recording state before lowering a nested body. Blocks and
    /// symbols belong to one CFG, so events must never cross a function
    /// boundary — replayed against the wrong graph they would report on
    /// control flow that does not exist.
    pub(crate) fn reset_move_recorder(&mut self) {
        self.move_symbols.clear();
        self.move_events.clear();
        self.move_event_order = 0;
        self.move_events_func = None;
    }

    /// Whether `ty` is a `@:move` class.
    ///
    /// Checks this context's own set first, then the name-keyed registry: a
    /// class reached through an import was annotated in a different lowering
    /// context, and its SymbolId there means nothing here.
    pub(crate) fn type_is_move_class(&self, ty: crate::tast::TypeId) -> bool {
        let Some(sym) = self.get_class_symbol(ty) else {
            return false;
        };
        if self.derive_shared_classes.contains(&sym) {
            return false;
        }
        if self.derive_move_classes.contains(&sym) {
            return true;
        }
        let Some(info) = self.symbol_table.get_symbol(sym) else {
            return false;
        };
        let name = info
            .qualified_name
            .and_then(|q| self.string_interner.get(q))
            .or_else(|| self.string_interner.get(info.name));
        let found = name.map(super::lookup_move_class).unwrap_or(false);
        if found && std::env::var_os("RAYZOR_DEBUG_MOVECLASS").is_some() {
            eprintln!("[moveclass] {:?} resolved as @:move by name", name);
        }
        found
    }

    /// Enrol every by-value `@:move` parameter and bind it at function entry.
    pub(crate) fn enroll_move_params(&mut self, hir_func: &crate::ir::hir::HirFunction) {
        let params: Vec<(SymbolId, crate::tast::TypeId)> = hir_func
            .params
            .iter()
            .map(|p| (p.symbol_id, p.ty))
            .collect();
        for (symbol, ty) in params {
            if self.type_is_move_class(ty) {
                self.enroll_move_symbol(symbol);
                self.record_move_event(MoveEventKind::Bind, symbol, hir_func.source_location);
            }
        }
    }

    /// Publish a function's parameter ownership, by SymbolId for this context
    /// and by qualified name for consumers that lower in another one.
    ///
    /// Recorded when the signature is registered, which happens for every
    /// function before any body lowers, so a call site always finds a
    /// complete entry for a callee in the same module.
    pub(crate) fn record_param_ownership(
        &mut self,
        symbol_id: SymbolId,
        hir_func: &crate::ir::hir::HirFunction,
    ) {
        if hir_func.params.is_empty() {
            return;
        }
        let owns: Vec<crate::tast::ParamOwnership> =
            hir_func.params.iter().map(|p| p.ownership).collect();
        if let Some(qualified) = self
            .symbol_table
            .get_symbol(symbol_id)
            .and_then(|s| s.qualified_name)
            .and_then(|q| self.string_interner.get(q))
        {
            super::record_param_ownership_by_name(qualified, &owns);
        }
        self.function_param_ownership.insert(symbol_id, owns);
    }

    /// Publish `@:consume` on a method, by SymbolId here and by qualified name
    /// for callers in another lowering context.
    pub(crate) fn record_consume_method(
        &mut self,
        symbol_id: SymbolId,
        hir_func: &crate::ir::hir::HirFunction,
    ) {
        if !hir_func.is_consume {
            return;
        }
        if let Some(qualified) = self
            .symbol_table
            .get_symbol(symbol_id)
            .and_then(|s| s.qualified_name)
            .and_then(|q| self.string_interner.get(q))
        {
            super::record_consume_method_by_name(qualified);
        }
        self.consume_methods.insert(symbol_id);
    }

    /// Whether calling `callee` ends the caller's binding on the receiver.
    pub(crate) fn callee_consumes_receiver(&self, callee: SymbolId) -> bool {
        if self.consume_methods.contains(&callee) {
            return true;
        }
        self.symbol_table
            .get_symbol(callee)
            .and_then(|s| s.qualified_name)
            .and_then(|q| self.string_interner.get(q))
            .map(super::lookup_consume_method_by_name)
            .unwrap_or(false)
    }

    /// Ownership of the callee's parameters, or None when the callee is not
    /// known — an indirect call through a function value, say. Unknown means
    /// the positional default applies, which is what the analysis did before
    /// signatures could say anything.
    pub(crate) fn callee_param_ownership(
        &self,
        callee: SymbolId,
    ) -> Option<Vec<crate::tast::ParamOwnership>> {
        if let Some(owns) = self.function_param_ownership.get(&callee) {
            return Some(owns.clone());
        }
        let qualified = self
            .symbol_table
            .get_symbol(callee)
            .and_then(|s| s.qualified_name)
            .and_then(|q| self.string_interner.get(q))?;
        super::lookup_param_ownership_by_name(qualified)
    }

    pub(crate) fn record_move_event(
        &mut self,
        kind: MoveEventKind,
        symbol: SymbolId,
        location: SourceLocation,
    ) {
        if !self.move_symbols.contains(&symbol) {
            return;
        }
        let Some(block) = self.builder.current_block() else {
            return;
        };
        self.move_events_func = self.builder.current_function;
        let order = self.move_event_order;
        self.move_event_order += 1;
        self.move_events.push(MoveEvent {
            block,
            order,
            symbol,
            kind,
            location,
        });
    }

    /// Solve the recorded events against the current function's CFG and emit a
    /// diagnostic for every binding read or moved after a move reaches it.
    /// Clears the per-function recording state; safe to call on functions that
    /// recorded nothing.
    pub(crate) fn check_move_flow(&mut self) {
        // Only when the recorded events belong to the function now finishing.
        // A forward-declaration stub finishes a function mid-way through
        // another one's body; consuming the events there would both analyse
        // them against the wrong CFG and silently drop them.
        if self.move_events.is_empty() || self.move_events_func != self.builder.current_function {
            return;
        }
        let events = std::mem::take(&mut self.move_events);
        self.move_symbols.clear();
        self.move_event_order = 0;
        self.move_events_func = None;
        let Some(func) = self.builder.current_function() else {
            return;
        };
        let violations = solve(&func.cfg, &events);
        for v in violations {
            self.report_use_after_move(v.symbol, v.moved_at, v.used_at);
        }
    }

    fn report_use_after_move(
        &mut self,
        symbol: SymbolId,
        moved_at: SourceLocation,
        used_at: SourceLocation,
    ) {
        // One report per (binding, use site): a use recorded both as a read of
        // the argument expression and as the move that consumes it is one
        // violation, not two.
        if !self
            .move_diag_seen
            .insert((symbol, used_at.file_id, used_at.line, used_at.column))
        {
            return;
        }
        let name = self
            .symbol_table
            .get_symbol(symbol)
            .and_then(|s| self.string_interner.get(s.name))
            .unwrap_or("<unknown>")
            .to_string();
        let use_span = Self::source_location_to_span(&used_at);
        let move_span = Self::source_location_to_span(&moved_at);
        let diag = diagnostics::DiagnosticBuilder::error(
            format!("use of moved value: `{}`", name),
            use_span.clone(),
        )
        .code("E0382")
        .label(use_span, "value used here after move")
        .secondary_label(move_span, "value moved here")
        .note(format!(
            "`{}` holds a `@:move` value, so the binding ends where the value is transferred away.",
            name
        ))
        .help(format!(
            "Clone the value explicitly (`var copy = {}.clone();`) or restructure so the original binding is no longer reachable.",
            name
        ))
        .build();
        self.diagnostics.push(diag);
    }
}

/// Forward may-analysis: a binding is "moved" at a point when some path
/// reaching that point moved it.
fn solve(cfg: &IrControlFlowGraph, events: &[MoveEvent]) -> Vec<Violation> {
    let mut by_block: BTreeMap<IrBlockId, Vec<usize>> = BTreeMap::new();
    for (i, ev) in events.iter().enumerate() {
        by_block.entry(ev.block).or_default().push(i);
    }
    for indices in by_block.values_mut() {
        indices.sort_by_key(|&i| events[i].order);
    }

    let mut preds: BTreeMap<IrBlockId, Vec<IrBlockId>> = BTreeMap::new();
    for (&id, block) in &cfg.blocks {
        for succ in block.successors() {
            preds.entry(succ).or_default().push(id);
        }
    }

    // Bottom everywhere. Blocks no path reaches keep it and report nothing,
    // which is what makes the branch that returns before a later read silent.
    let mut in_state: BTreeMap<IrBlockId, State> = BTreeMap::new();
    let mut out_state: BTreeMap<IrBlockId, State> = BTreeMap::new();
    for &id in cfg.blocks.keys() {
        in_state.insert(id, State::new());
        out_state.insert(id, State::new());
    }

    let mut worklist: Vec<IrBlockId> = vec![cfg.entry_block];
    let mut queued: BTreeSet<IrBlockId> = worklist.iter().copied().collect();
    let mut reachable: BTreeSet<IrBlockId> = queued.clone();

    while let Some(id) = worklist.pop() {
        queued.remove(&id);
        let mut entry = State::new();
        if let Some(ps) = preds.get(&id) {
            for p in ps {
                if !reachable.contains(p) {
                    continue;
                }
                if let Some(s) = out_state.get(p) {
                    join_into(&mut entry, s);
                }
            }
        }
        in_state.insert(id, entry.clone());

        let mut exit = entry;
        if let Some(indices) = by_block.get(&id) {
            transfer(events, indices, &mut exit, None);
        }

        // The transfer is monotone and `entry` only ever grows, so `exit` only
        // grows too; a plain inequality is enough to detect progress.
        let grew = out_state.get(&id) != Some(&exit);
        if grew {
            out_state.insert(id, exit);
        }

        let succs = cfg
            .blocks
            .get(&id)
            .map(|b| b.successors())
            .unwrap_or_default();
        for s in succs {
            let first_visit = reachable.insert(s);
            if (grew || first_visit) && queued.insert(s) {
                worklist.push(s);
            }
        }
    }

    // Second pass reports, so a loop body is diagnosed once rather than once
    // per fixpoint iteration.
    let mut violations = Vec::new();
    for (&id, indices) in &by_block {
        if !reachable.contains(&id) {
            continue;
        }
        let mut state = in_state.get(&id).cloned().unwrap_or_default();
        transfer(events, indices, &mut state, Some(&mut violations));
    }
    violations.sort_by_key(|v| (v.used_at.line, v.used_at.column));
    violations
}
