# Modularising HIR→MIR Lowering

Status: proposed (2026-08-11)
Subject: `compiler/src/ir/hir_to_mir.rs` — 42,727 lines, one `impl`, 312 methods

## Measurements first

| | |
|---|---|
| file | 42,727 lines |
| top-level items | header 1..777, one `impl<'a> HirToMirContext<'a>` 778..42,424, tail 42,425..42,728 |
| methods in that impl | 312 |
| `lower_*` | 25,636 lines across 54 fns |
| `lower_expression_inner` | **13,421 lines** — one method |
| ...of which `HirExprKind::Call` | **9,555 lines** — one match arm |

`lower_expression_inner` is a single `match &expr.kind` with 25 arms. It is
already the right shape for most of them: `Literal`, `If`, `This`, `Super`,
`Lambda`, `ArrayComprehension` and friends are 5–150 lines and delegate to a
named method. Four arms were written inline instead — `Call` 9,555, `New` 993,
`Binary` 891, `Cast` 533 — and they are 87% of the method.

## "Passes" is the wrong word here

A pass is a module-level transformation with a defined before/after over the
whole IR — what `optimization.rs` runs. Expression lowering is not that. It is a
recursive descent over a tree, where each node kind produces MIR. Modelling it
as passes would mean either walking the tree once per node kind, or inventing a
pass that only handles one variant — neither is real. The correct axis of
decomposition is **the node kind and, inside calls, the callee form**.

## Pattern 1 — dispatcher + per-variant lowering (for the match)

Make uniform what the file already does for 21 of 25 arms:

```rust
fn lower_expression_inner(&mut self, expr: &HirExpr) -> Option<IrId> {
    self.builder.set_source_location(...);
    let result = match &expr.kind {
        HirExprKind::Literal(l)   => self.lower_literal(l, expr.ty),
        HirExprKind::Call { .. }  => self.lower_call_expr(expr),
        HirExprKind::New { .. }   => self.lower_new_expr(expr),
        HirExprKind::Binary { .. }=> self.lower_binary_expr(expr),
        ...
    };
    ...
}
```

Each `lower_*_expr` keeps the `&mut self` receiver, so no state plumbing
changes — the methods simply live in different files. This is the whole reason
to prefer split `impl` blocks over free functions: `HirToMirContext` carries
~40 fields of lowering state, and threading that through free functions would
be a far larger and riskier change than moving method bodies.

## Pattern 2 — ordered chain of call-form handlers (for `Call`)

The 9,555-line arm is not a monolith. It is already a chain of eight sequential
probes on the callee's shape, each handling its form or falling through:

    +7     if let Field    { .. }   @:shader wgsl intercept
    +143   if let Variable { .. }
    +363   if let Variable { .. }
    +545   if let Field    { .. }   ~2.5k lines
    +3081  if let Field    { .. }
    +3141  if let Variable { .. }
    +3257  if let Variable { .. }   ~6.1k lines
    +9399  if let Variable { .. }
    +9498  fallthrough: generic indirect call through a function pointer

Give that structure a name instead of leaving it implicit:

```rust
fn lower_call_expr(&mut self, expr: &HirExpr) -> Option<IrId> {
    let cx = CallSite::new(expr)?;          // callee, args, type_args, result type
    // ORDER IS SEMANTIC: an earlier handler shadows every later one.
    if let Some(v) = self.try_lower_shader_intrinsic(&cx) { return Some(v); }
    if let Some(v) = self.try_lower_stdlib_static(&cx)    { return Some(v); }
    if let Some(v) = self.try_lower_stdlib_instance(&cx)  { return Some(v); }
    ...
    self.lower_indirect_call(&cx)           // the one documented fallback
}
```

Each handler becomes an `Option`-returning unit that can be read and tested on
its own, and the ordering — which currently exists only as the sequence of
`if let`s in a 9.5k-line block — becomes an explicit, reviewable list. It also
gives the codebase's "no silent fallthrough" rule a single place to hold: one
named fallback at the end rather than an implicit tail.

`CallSite` is a small borrow-only struct (callee, args, type args, converted
result type, source location) so handlers do not each re-derive it — that
derivation is currently repeated at the top of the arm.

## Proposed layout

    compiler/src/ir/mir/
      mod.rs           struct HirToMirContext, helper types, statics, free fns,
                       entry points (lower_hir_to_mir*), `mod` decls
      expr/
        mod.rs         lower_expression_inner — the dispatcher, one line per arm
        literal.rs     new.rs      binary.rs   unary.rs
        cast.rs        field.rs    index.rs    variable.rs
        control.rs     (If, TryCatch, TypeCheck)
        call/
          mod.rs       CallSite + the ordered handler chain
          shader.rs    stdlib_static.rs   stdlib_instance.rs
          method.rs    closure.rs         indirect.rs
      stmt.rs          lower_statement, loops
      pattern.rs       match/pattern lowering
      types.rs         convert_type and friends
      class.rs         register_class_metadata, vtables
      globals.rs       closures.rs   helpers.rs

`ir/mod.rs` keeps `pub mod mir;` plus `pub use mir as hir_to_mir;` so the three
external references (`pipeline.rs`, `compilation.rs`, `ast_lowering.rs` doc)
keep working unchanged.

Each submodule opens with `use super::*;` — child modules can see a parent's
private items, so the shared type and import surface needs no `pub` churn.

## Sequencing

The value is in the split, and the risk is in doing it blind. In order:

1. `mir/mod.rs` with the header, the tail and an empty `mod` list; move the
   impl wholesale into `mir/impl_all.rs`. Two files, no method moved, build
   green. This proves the module plumbing before anything is cut.
2. Move whole methods out of `impl_all.rs` into topic files, largest groups
   first, building after each move. Methods move unedited.
3. Only then decompose `lower_expression_inner`: extract each inline arm into
   `lower_*_expr`, one arm per commit.
4. Only then decompose `Call` into the handler chain, one handler per commit,
   keeping the order identical and asserting it in the dispatcher's comment.

Steps 1–2 are pure moves and cannot change behaviour. Steps 3–4 do change
control flow shape, so each wants the full test suite: the 5 MIR regression
tests, `test_rayzor_stdlib_e2e` (15 cases), and a nue bundle + inference run,
since expression lowering is what nue exercises hardest.

## Repetition that should have been one call

Measured over the same file, not impressions:

| occurrences | what | should be |
|---|---|---|
| 19 | `vec![IrType::Ptr(Box::new(IrType::Void))]` as a self/receiver parameter list | `Self::self_param_types()` |
| 19 | the integer-type match list `IrType::I8 \| I16 \| .. \| U64` | `fn is_integer_ty(&IrType) -> bool` |
| 17 | `IrType::Function { params, return_type, varargs: false }` built by hand | `Self::fn_sig(params, ret)` |
| 8 | phi-local registration — `IrLocal { name: format!("{}_phi", local.name), ty, mutable: true, source_location, allocation: Register }` | `fn register_phi_local(&mut self, phi_reg, local, ty)` |
| 7 | `global_symbol_map.get(..)` followed by an ad-hoc name scan (lines 9260, 9465, 20528, 27813, 28407, 29622, 42365) | one `fn resolve_global(&self, sym) -> Option<IrGlobalId>` |
| 8 | the callee-shape probes inside the `Call` arm | the handler chain above |

The global-symbol one is the most valuable: four of those seven sites carry
their own slightly different fallback scan, which is exactly how a resolution
rule drifts between call sites. One function, FQN-first, is also what
`feedback_fqn_first_resolution` asks for.

None of these are style points — each is a place where a fix has to be applied
N times and will eventually be applied N-1 times.

## Comment budget

    total 42,728   code 32,672   comment 7,227 (16%)   blank 2,829

16% is not itself wrong; the composition is. Narrative markers counted in
comment text: **BUG 90, FIX 50, NOTE 29, "used to" 24, "previously" 17,
TRAP 13, TODO 12, "earlier" 6**. That is ~230 comments whose subject is the
history of the code rather than the code. The longest blocks run 20–29 lines
and several open by explaining what the previous implementation did wrong.

Rule for the split: a comment states what the code does and why it is shaped
that way. It does not narrate which commit broke it, which probe found it, or
what it used to be — that belongs in the commit message and in the bug record.
Applying that to the ~230 narrative comments should remove on the order of a
thousand lines while making the survivors worth reading.

Do this DURING the mechanical move (step 2), not as a separate pass: the diff
is already being reviewed line by line at that point, and a comment-only pass
over 42k lines afterwards is a second full review for no additional safety.

## Can HIR fold into MIR?

In principle yes — HIR is a re-representation, not a desugaring layer. It keeps
`ArrayComprehension`, `MapComprehension`, `StringInterpolation`,
`MacroExpansion` and `Reification` as first-class nodes, so it does not lower
the high-level constructs; `hir_to_mir` does. Layer sizes:

    hir.rs          800 lines   (node definitions)
    tast_to_hir.rs  6,186
    hir_to_mir.rs   42,727

But folding is the wrong FIRST move, because the pain attributed to "two
lowerings" is really one specific information loss.

### HIR erases the call resolution TAST already computed

    TAST                                        HIR
    FunctionCall     { function, .. }
    MethodCall       { receiver,                Call { callee, args,
                       method_symbol: SymbolId,        is_method: bool,
                       .. }                            type_args }
    StaticMethodCall { class_symbol: SymbolId,
                       method_symbol: SymbolId, .. }

TAST distinguishes three call forms and carries the RESOLVED `method_symbol`
and `class_symbol`. HIR collapses all three into one node and replaces the
symbols with a bool. `hir_to_mir` then spends its 9,555-line `Call` arm
recovering that information by pattern-matching on the *shape* of the callee —
`if let Field { object, field }` for something method-like, `if let Variable
{ symbol }` for something function-like — eight probes deep, re-resolving
through the symbol table and the stdlib mapping each time.

That is the single highest-value change available in this area, and it is far
smaller than folding two IRs: **carry the resolution through**. Either keep the
three TAST forms in HIR, or give `HirExprKind::Call` a resolved target:

```rust
enum CallTarget {
    Function(Box<HirExpr>),
    Method { receiver: Box<HirExpr>, method: SymbolId },
    Static { class: SymbolId, method: SymbolId },
}
```

The eight shape probes then become a match on a known enum, and most of the
9,555 lines becomes unnecessary rather than merely relocated. Doing the
modularisation first would move that code into files; doing this first means
there is much less of it to move.

### If the fold is still wanted afterwards

The 42,727 lines live in `hir_to_mir` and do not shrink because TAST feeds it
instead of HIR — folding deletes `tast_to_hir.rs` (6,186) and `hir.rs` (800)
and removes one IR to keep in sync, which is real but smaller than it looks.
The cost is the HIR-level consumers, which would each need a new home:
`drop_analysis.rs` (506), `optimizable.rs` (588), and `wgsl_transpiler.rs`.
Order of operations: restore call resolution, then modularise, then decide
whether the remaining HIR still earns its 7k lines.

## Housekeeping found on the way

Four stale editor backups are sitting in the source tree, and two of them are
TRACKED IN GIT:

    compiler/src/ir/hir_to_mir.rs.bak       235 KB   tracked
    compiler/src/ir/hir_to_mir.rs.bak2      235 KB   tracked
    compiler/src/tast/ast_lowering.rs.backup
    compiler/src/tast/type_flow_guard.rs.backup

They date from Nov 2025. They should be deleted, and `*.bak*`/`*.backup` added
to `.gitignore` so a copy-before-edit habit stops landing in the repo.

## What NOT to do

- Do not convert methods to free functions taking `&mut HirToMirContext`. The
  receiver is the only thing keeping this tractable.
- Do not reorder call handlers while moving them. The current order is load
  bearing and undocumented; preserve it exactly, then document it.
- Do not merge the split with behaviour fixes. A 42k-line move is only safe to
  review if the diff is provably a move.
