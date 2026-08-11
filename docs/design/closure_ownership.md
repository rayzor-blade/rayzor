# Closure Ownership — Giving Closures a Drop Point

Status: proposed (2026-08-11)
Related: `compiler/src/ir/insert_free.rs`, `compiler/src/codegen/*_backend.rs`,
`compiler/haxe-std/rayzor/concurrent/Arc.hx`

## Premise

Rayzor has no GC. The model is: every heap value has exactly one owner, and the
compiler frees it at a drop point it computes. Closures do not participate in
that model at all — they are the one construct that allocates without an owner.

`MakeClosure` heap-allocates **two** blocks, and both are created in the
backends rather than in MIR:

    // llvm_jit_backend.rs, and the equivalent in cranelift/c/wasm
    env_ptr     = malloc(captured_values.len() * 8)   // one i64 slot per capture
    closure_ptr = malloc(16)                          // { fn_ptr, env_ptr }

Because neither allocation exists as a MIR instruction, `insert_free` cannot see
them: it recognises allocations only as `CallDirect` to a known allocator
(`malloc_ids`, `anon_new_ids`, fresh-array and fresh-string sets). Nothing in
the compiler frees a closure, its environment, or anything captured into it.

## What this cost, measured

`insert_free`'s escape walk had arms for `Return`, calls, `Store`,
`CreateStruct`, `StoreGlobal` and `MemCopy`, and none for closure creation. A
captured allocation therefore looked dead at the end of the frame that made it
and was freed there, while the closure still pointed at it:

```haxe
static function makeCounter():Int->Int {
    var buf = [0];
    return function(k:Int) { buf[0] = buf[0] + k; return buf[0]; };
}
```

SIGSEGV on both tiers, at every optimisation level. Fixed in `4161b47e` by
treating a capture as escaping — the frame stops claiming ownership. That
removes the crash and leaves a leak, because no one else takes ownership:

| 500k closures, each capturing an 8-element array | peak footprint |
|---|---|
| raw array captured | 97 MB |
| `Arc.init` + `.clone()` captured | 42 MB |
| same allocations, no closure (freed normally) | ~22 MB |

At 2M iterations the raw case reaches 254 MB against 41 MB for the same
allocations without a closure — about 107 bytes per closure, which is the
24 bytes of env + struct plus the ~96-byte array.

**Arc does not rescue this, and the reason matters.** `Arc<T>` is the sanctioned
shared-ownership mechanism (`clone()` bumps the refcount; `@:rc`/`@:arc` classes
are required to derive `Clone`), and a closure capturing an Arc computes the
right answer. But the Arc handle lives *inside* the environment, and nothing
ever frees the environment, so that handle never drops, the refcount never
reaches zero, and the payload never releases. Arc shrinks the per-item cost
(42 MB vs 97 MB) without changing the outcome. The thing with no owner is the
closure, not the capture.

## The core problem

A closure that **escapes** its defining function has its drop point in a
*different* function. At that point the capture ids are not in scope, so the
compiler cannot emit per-capture frees there:

```haxe
static function make():Int->Int {
    var buf = [0];                       // capture allocated HERE
    return function(k:Int) { ... };      // escapes
}
static function main() {
    var f = make();                      // drop point is HERE
}                                        // ...where `buf` has no name
```

Per-capture frees at the drop point only work for a closure that never leaves
its defining function. Any general fix therefore needs the *environment itself*
to know how to release its contents.

## Proposal: drop glue in the closure

For each lambda the compiler already knows the capture types at `MakeClosure`.
Synthesise a per-lambda **drop function** that releases the owned slots, and
carry a pointer to it in the closure:

    MakeClosure { dest, func_id, captured_values, drop_fn }

    closure struct: { fn_ptr, env_ptr, drop_ptr }        // 16 -> 24 bytes

    rayzor_closure_free(c):
        if c.drop_ptr != null: c.drop_ptr(c.env_ptr)     // release owned captures
        free(c.env_ptr)
        free(c)

`insert_free` then treats a `MakeClosure` dest as an allocation like any other
and emits `rayzor_closure_free` at the computed drop point. An Arc capture
releases by refcount exactly as `Arc.hx` documents, because the generated drop
function decrements it.

Work: MIR instruction shape, four backends (LLVM, Cranelift, C, wasm), one
runtime helper, and the allocation/escape arms in `insert_free`.

## Alternative considered: clone on capture

Emit `Clone` for heap captures so the closure owns a copy and the defining frame
keeps its own. Simpler, no struct-layout change, and it would finally give
`IrInstruction::Clone` a producer — the opcode is implemented in every backend
and consumed by `dump`/BCE/optimization but is never constructed in lowering.

Rejected as the primary design because it changes semantics: two closures over
the same buffer would stop observing each other's writes, which the counter
idiom above depends on. It remains a reasonable fallback for capture types where
a copy is provably equivalent.

## Decisions needed before implementing

1. **Struct layout.** Growing to 24 bytes touches every backend and anything
   that assumes the 16-byte shape. Alternative: park `drop_ptr` in env slot 0
   and keep the struct at 16 bytes, at the cost of an extra indirection.
2. **A capture used after the closure.** `var buf = [0]; var f = ...buf...;
   use(buf);` — the frame and the closure both want it. Options: refuse
   (require `Arc`), transfer and reject later use, or clone.
3. **Escape through a call.** Passing a closure as an argument is currently an
   escape, so a non-escaping-in-practice case like `pool.parallelRows(rows,
   band)` would never be freed. Needs the same per-parameter retention story as
   `compute_param_retention`, which is itself opt-in and unsound today.
4. **Interaction with `@:derive(Drop)`.** A captured class deriving `Drop` must
   have its `drop()` run by the generated drop function, not just be freed.
5. **Calling a closure must not count as an escape.** `CallIndirect`'s
   `func_ptr` position needs to be exempt, or every called closure is
   permanently un-freeable and the feature does nothing.

## Risk

The bug this replaces was a use-after-free, and a drop point computed slightly
too eagerly reintroduces exactly that. Sequencing that keeps the failure mode on
the safe side: land the allocation tracking with frees **disabled** and assert
the drop points look right, then enable them for non-escaping closures only,
then extend to escaping ones once the retention story is settled. The
regression test `compiler/tests/haxe/test_closure_captured_heap.hx` covers the
crash; a leak check belongs beside it, since a leak is what a too-conservative
version produces and it currently passes silently.
