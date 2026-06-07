# MIR SsaBarrier Instruction — Design Sketch

Status: **PARTIALLY LANDED** (2026-06-07) — Cranelift in-place opacity
shipping at extern CallDirect; MIR-level SsaBarrier instruction kept as a
passthrough primitive for future passes. Does NOT fully close
`bugs_sys_call_in_generation_method` — Sys.time at top of generate still
hangs at a downstream point that isn't an extern call.
Owners: TBD
Tracking: bugs_sys_getenv_in_ctor_residual, bugs_sys_call_in_generation_method

## What landed

- `IrInstruction::SsaBarrier { dest, src, ty }` variant with C / interpreter
  / Cranelift lowering (passthrough — primitive kept for future MIR passes).
- `Self::ssa_barrier_value(builder, value)` helper in `cranelift_backend.rs`:
  emits `fadd(v, +0.0)` for floats, stack store/load for ints. Picked by
  ACTUAL Cranelift machine type (NOT MIR's declared return_ty).
- In-place application at extern CallDirect's result, AFTER `value_map.insert`.
  No new IrId. No MIR-level barrier_dest. Avoids the
  register-class-mismatch trap that the original "alloc barrier_dest in
  MIR" plan hit.

## Why MIR-level barrier_dest didn't work

The original plan (Phase 3 in the sketch below) was to allocate a new
barrier_dest IrId in MIR via `alloc_reg_typed(return_ty)`, emit
SsaBarrier{dest:barrier_dest, src:call_d}, return barrier_dest from `call()`.

Bisect found: for some extern calls, MIR's declared `return_ty` is `F64`
while the actual Cranelift call result has machine type `I32`. Downstream
consumers of `barrier_dest` query the MIR type, get F64, emit
float-typed instructions; the actual value is I32. aarch64's `gen_move`
asserts `to_reg.class() == from_reg.class()`. Crash.

This is a pre-existing MIR/Cranelift type-mismatch bug in rayzor's lowering
that the new barrier IrId exposed but didn't cause. The in-place
approach sidesteps it: we reuse the SAME IrId, so downstream type queries
are unchanged.

## Problem

A class of MIR/Cranelift bugs surfaces whenever an effectful CallDirect's
result is used across a block boundary in certain shapes. Documented
symptoms:

- **Cranelift egraph elaboration panic** ("elaborating effectful
  instructions, should have remained in the skeleton") at
  `cranelift/codegen/src/egraph/elaborate.rs:722` — original
  `bugs_sys_getenv_in_ctor_residual` shape, fixed for if/else with
  field-store branches at 290e475 via branch-phi + per-block
  reaching-defs.
- **Block cascade to `Unreachable`** — workflow `w80ppz2wj` (2026-06-07)
  traced this in `GenerationLoop.generate`: forward-ref stub bodies
  inlined → downstream block predecessor severed →
  UnreachableBlockEliminationPass deletes 50+ blocks → Cranelift
  lowers truncated body to `trap user100` → SIGTRAP. Fixed at
  `d5f2b48` by refusing to inline stubs.
- **Sys.* at function entry hang** — still residual after `d5f2b48`.
  Same family but a different downstream path (not stub-inlining).
  Tracked as `bugs_sys_call_in_generation_method`.

Common thread: passes that analyze MIR locally treat the effectful
Call result as if it can be folded, forwarded, eliminated, or
treated as divergent. Each instance has been a separate cause but
the same defensive primitive — a value-flow boundary that no MIR
pass and no Cranelift egraph step can look through — would close
the entire class.

## Why existing tools don't work

Four bridge attempts documented in `bugs_sys_getenv_in_ctor_residual`:

| # | Bridge | Why it failed |
|---|---|---|
| 1 | `IrInstruction::Copy { dest, src: call_result }` | Cranelift's MIR Copy lowering at `cranelift_backend.rs:2426-2433` aliases `dest` to `src` in `value_map`. No CLIF instruction emitted. Egraph sees through. |
| 2 | `BinOp::Add(src, const 0)` | Cranelift's egraph folds `x + 0 → x` during e-class union. Bridge gone before elaboration. |
| 3 | Stack-slot store + load | Cranelift's egraph alias-analyzes through memory; forwards `store slot, x` → `load slot` as `x` during elaboration. |
| 4 | Branch-phi at start of each branch block | Works for if/else patterns (became part of 290e475 fix). But it presumes there IS an if/else at the use site — not a general primitive. |

The pattern: any IR construct that "looks like an identity, like Copy
or Add 0 or Load-after-Store-this-value, gets folded by an
optimization. The egraph in particular forwards through pure data
movement.

## SsaBarrier design

A new MIR instruction whose contract is exactly "the value flows
through unchanged, but no pass MAY look through it." Backends emit a
real instruction that the optimizer can't recognize as identity.

### MIR-level shape

```rust
// In compiler/src/ir/instructions.rs
pub enum IrInstruction {
    // ... existing variants ...

    /// SSA barrier — value-flow identity that prevents any MIR pass
    /// or Cranelift egraph step from looking through it. Lowers to
    /// a real backend instruction (an inline-asm nop in Cranelift,
    /// `llvm.assume(true)` chained or an opaque intrinsic in LLVM,
    /// a value copy in the interpreter).
    ///
    /// Used to wrap effectful Call results before cross-block use, to
    /// dodge the egraph-elaborates-effectful-call class of bugs (see
    /// bugs_sys_getenv_in_ctor_residual and bugs_sys_call_in_generation_method).
    ///
    /// Semantics: `dest = src` (value-wise). NO side effects of its
    /// own; the barrier exists only as an opacity hint to optimizers.
    SsaBarrier {
        dest: IrId,
        src: IrId,
        ty: IrType,
    },
}
```

### Backend lowering

**Cranelift** (`compiler/src/codegen/cranelift_backend.rs`):

Emit a single `iconst_imm` followed by a `select` that the egraph
can't fold. Concretely, the simplest form that survives egraph
elaboration is `select_spectre_guard`:

```rust
// In emit_instruction:
IrInstruction::SsaBarrier { dest, src, ty } => {
    let src_v = *value_map.get(src).ok_or("SsaBarrier src")?;
    // select_spectre_guard is marked side-effect-bearing in
    // Cranelift's instruction info (see cranelift-codegen/meta/
    // src/shared/instructions.rs). The egraph doesn't elaborate
    // through it. Cost is one branch-free conditional move at
    // runtime; codegen is one CSEL on aarch64, one CMOV on x86.
    let true_v = builder.ins().iconst(types::I8, 1);
    let opaque = builder.ins().select_spectre_guard(true_v, src_v, src_v);
    value_map.insert(*dest, opaque);
}
```

Alternative if `select_spectre_guard` proves foldable in future
Cranelift releases: emit an inline-asm `nop` with `src` as
input/output operand. Cranelift supports inline asm via the
`InstBuilder::asm` extension; the egraph cannot fold across asm
boundaries.

**LLVM tier** (when llvm-backend feature is on):

```rust
// In compiler/src/codegen/llvm_backend.rs
IrInstruction::SsaBarrier { dest, src, ty } => {
    let src_val = self.value_map[src];
    // llvm.assume blocks LLVM's value-forwarding analyses across
    // the call; combined with a no-op true predicate it's a hard
    // SSA barrier.
    let true_const = self.context.bool_type().const_int(1, false);
    self.builder.build_call(
        self.intrinsics.assume,
        &[true_const.into()],
        "ssa_barrier_guard",
    );
    self.value_map.insert(dest, src_val);
}
```

Note: LLVM `assume` may not be aggressive enough on its own; combine
with `inline_assembly` to "use" the value so LLVM can't drop it.

**Interpreter** (`compiler/src/interpreter/`):

```rust
IrInstruction::SsaBarrier { dest, src, ty: _ } => {
    let value = self.get(src);
    self.set(*dest, value);
}
```

### Insertion strategy

Two phases of insertion considered. **Phase A first; Phase B as a
followup if A is insufficient.**

**Phase A — at hir_to_mir for effectful Call results crossing block
boundaries.**

In `lower_hir_call` (or wherever CallDirect MIR is emitted), check
if the callee is marked effectful and if the result is used outside
the current block. If yes:

1. Emit the CallDirect into a fresh temp `t_raw`.
2. Emit `SsaBarrier { dest: t_barrier, src: t_raw, ty: <call return ty> }`.
3. Use `t_barrier` everywhere downstream (including the original
   destination register the caller wanted bound).

"Used outside the current block" can be computed cheaply with a
single liveness sweep over the current function (we already do
similar work in `DropPointAnalyzer`).

"Effectful" comes from `RuntimeFunctionCall.source` (`FunctionSource::Builtin`,
`FunctionSource::ExternC`) and from any user-defined call without a
`@:pure` attribute. Defaulting to "barrier-wrap if unsure" is safe:
the barrier is just a value-flow identity, never wrong, only
sometimes redundant. Cost is one extra MIR instruction per Call.

**Phase B — as a defensive MIR pass.**

A new pass `EffectfulCallBarrierPass` after `hir_to_mir` and before
the existing optimization passes. It walks the CFG, finds CallDirect
nodes whose result has at least one cross-block use, and inserts
SsaBarriers between the call and the cross-block use sites.

Phase B is more conservative (catches anything Phase A missed) but
duplicates work. Recommend Phase A as the primary insertion, Phase B
as a verification pass that asserts "no effectful CallDirect's result
crosses a block boundary without an intervening SsaBarrier" — fires
in debug builds only.

### Validation

The branch-phi attempt at 290e475 worked standalone but failed when
the function was inlined (SROA cascade). The same risk exists for
SsaBarrier: an inliner that copies the barrier in but renumbers
SSA values incorrectly could break it. Three guard rails:

1. **`InliningPass` already syncs `caller.next_reg_id` after `inline_call_site`**
   (655d7ac). SsaBarrier becomes just another MIR instruction; the
   sync covers it.

2. **`apply_sra` already does per-block reaching-defs with phi
   insertion at merges** (290e475 fix). SsaBarrier's dest is
   tracked the same way as any other SSA def; no special handling
   needed.

3. **New: SsaBarrier identity check in `CopyPropagationPass`**.
   Currently CopyProp substitutes Copy uses with src. It MUST
   NOT do the same for SsaBarrier — that would defeat the point.
   Explicit gate: `if matches!(inst, IrInstruction::SsaBarrier { .. }) { continue; }`.

A debug-mode post-optimization validator (per
`project_compiler_ssa_hygiene_followups` #1) would catch any pass
that elided a barrier illegally.

### Risks

| Risk | Mitigation |
|---|---|
| Cranelift's `select_spectre_guard` is foldable in some future release | Fall back to inline-asm-nop pattern; either way the barrier should be small. Pin Cranelift version via Cargo.lock. |
| Barrier overhead at runtime (one CSEL per effectful call) | CSEL is single-cycle on M1; ~112 effectful calls per layer × 16 layers ≈ 1.8k CSELs per token, ~600 ns at 3.2 GHz — <0.002% of token budget. Negligible. |
| Phase A insertion misses some call shape | Phase B defensive pass catches the gap (debug-only assertion). |
| Inliner duplicates the barrier when inlining a caller | Acceptable — duplicate barriers compose, they don't break. Worst case: one extra CSEL per inlined call site. |
| LLVM tier ignores `assume` | Pair with `inline_assembly` that USES the value — `asm("" :: "r"(src))` style. LLVM is conservative around inline asm. |

### Implementation phases

| Phase | What | Effort |
|---|---|---|
| 1 | Add `IrInstruction::SsaBarrier` variant + interpreter lowering | half-day |
| 2 | Cranelift lowering via `select_spectre_guard`; test that egraph doesn't fold | half-day |
| 3 | Phase-A insertion in `lower_hir_call` for effectful Call results | 1-2 days |
| 4 | Verify on `bugs_sys_call_in_generation_method` repro (Sys.time at top of generate must not hang) | hours |
| 5 | LLVM tier lowering | half-day |
| 6 | Phase-B defensive pass (debug-only assertion) | half-day |
| 7 | Document + memory update + remove the LlamaArch.hx workaround | hours |

Total: ~4-5 days of focused compiler work. Cleanly scoped, mostly
additive. Highest risk: Phase 2 — the Cranelift lowering needs to
survive egraph elaboration. Mitigation: have Phase 2's smoke test
be specifically "Sys.time + downstream block produces CLIF that
emits actual blocks past the call, not Unreachable."

### Verification

- **Primary**: `bugs_sys_call_in_generation_method.md`'s repro —
  insert `Sys.time();` at the top of `GenerationLoop.generate`,
  build, run llama-chat with canonical Paris prompt. Expected:
  Paris MATCH passes (no hang, no SIGTRAP).
- **Original**: `bugs_sys_getenv_in_ctor_residual.md`'s repro —
  /tmp/min2.hx with Sys.getEnv in constructor + if/else field
  store. Expected: exit=0, prints "fallback" with FOO unset.
- **128/128 haxe regression** must remain green.
- **LlamaArch.hx workaround removal**: the comment at LlamaArch.hx:124
  reads env in a method and threads a bool through. With SsaBarrier
  landed, the workaround can be deleted; env read can move into
  RoPE.hx ctor or similar.
- **Perf**: full Voronoi 600-tok A/B. Expected: ±0.1% (sub-noise).
  The CSEL added to ~1.8k call sites per token is below the
  measurement floor; a regression would indicate a Cranelift
  lowering pessimization that we'd have to investigate.

### Out of scope

- WASM backend lowering of `SsaBarrier`. The WASM target has a
  separate runtime story (`runtime-wasm`) and a separate set of
  egraph constraints. Add it when needed.
- A more elaborate "effect system" for marking which calls need
  barriers vs which are pure. Default-barrier is fine.
- A general `Volatile` IR instruction. SsaBarrier is the narrow
  primitive we need; a full volatile model is a bigger redesign.

## Decision points before implementing

1. **Cranelift primitive choice**: `select_spectre_guard` (my pick)
   vs inline-asm nop. The former is cheaper and more portable; the
   latter is hard-guaranteed against any future Cranelift
   optimization. If the rayzor team plans to update Cranelift
   regularly, lean toward inline-asm. If pinning, `select_spectre_guard`
   is fine.

2. **Phase A scope**: barrier EVERY effectful CallDirect, or only
   those whose result is used cross-block? The lazy answer (barrier
   every Call) wastes one CSEL per call but eliminates the
   cross-block-detection complexity. The smart answer requires a
   one-pass liveness scan but reduces overhead by ~70% (most calls
   are used in the same block). I'd start with the lazy version
   and add the optimization if perf measurement shows it matters.

3. **Phase B (debug-only assertion pass)**: yes/no. If the team's
   compiler-internals discipline keeps the rules straight, Phase B
   is overkill. If we keep finding "another corner case," Phase B
   pays back its half-day cost by surfacing the next bug at
   compile time instead of as a SIGTRAP under llama-chat.

## Appendix — what the SsaBarrier shape would look like in IR

Before (problematic):

```text
; In some function:
bb0:
  v1 = call_direct effectful_fn
  jump bb1

bb1:                  ; egraph elaborator looks at v1
  store field, v1     ; cross-block use of effectful Call result
  jump bb2

bb2:
  ...                 ; gets deleted by cascade or panic
```

After:

```text
bb0:
  v1 = call_direct effectful_fn
  v1b = ssa_barrier v1    ; <-- new instruction
  jump bb1

bb1:
  store field, v1b        ; egraph sees an opaque value, can't fold
  jump bb2

bb2:
  ...                     ; stays intact
```

Cranelift lowering of `v1b = ssa_barrier v1`:

```text
v_true = iconst.i8 1
v1b = select_spectre_guard v_true, v1, v1
```

LLVM IR lowering:

```text
call void @llvm.assume(i1 true)
%v1b = bitcast %v1 to <same type>     ; or just SSA-rename, plus an
                                       ; inline asm "" "r,~{memory}"(%v1)
                                       ; right after, to force LLVM
                                       ; to keep %v1 live
```

Interpreter: `set(v1b, get(v1))`.
