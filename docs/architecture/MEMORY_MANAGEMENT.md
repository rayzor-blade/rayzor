# Rayzor Memory Management Strategy

## Overview

Rayzor uses **ownership-based memory management** inspired by Rust's model. The compiler performs compile-time analysis to determine when objects should be allocated and freed, eliminating the need for a garbage collector in the common case. GC is reserved exclusively for `Dynamic` types and objects whose sizes cannot be determined at compile time.

This document covers the full memory safety pipeline: ownership analysis, lifetime analysis, borrow checking, drop analysis, and escape analysis.

## Design Philosophy

| Approach | When Used |
| --- | --- |
| Ownership + Automatic Drop | Default for all heap-allocated classes when drop conditions are met |
| Compile-time move checking (`@:move`) | Opt-in per class; enforced by MoveFlow, no runtime cost |
| Reference Counting (`@:shared`) | Shared ownership; `.clone()` becomes an atomic increment |
| Runtime Managed | Thread, Channel, Arc, Mutex (runtime handles cleanup) |
| No Drop | Primitives (Int, Float, Bool), Dynamic |
| GC | `Dynamic` types or objects with unknown compile-time size only |

`@:rc` and `@:arc` appear in older notes as the reference-counting opt-in. They parse and
do nothing; `@:shared` is the annotation with a consumer.

The key insight: most Haxe programs use concrete types with known sizes. For these, the compiler can statically determine ownership, insertion of `Free` instructions at the correct points, and verify safety -- all without runtime overhead.

## Memory Annotations

Rayzor extends Haxe with opt-in memory annotations. **Opt-in is structural, not a
filter**: a class that carries no annotation records no ownership events at all, so
existing Haxe from the wider ecosystem compiles unchanged and silently. There is no
mode in which unannotated code is analysed.

The table below separates what the compiler **enforces** from what it merely parses.
An annotation in the second group is accepted by the parser and reaches the TAST, then
forwards to nothing — writing one changes no behaviour.

### Enforced

| Annotation | Position | Meaning |
| --- | --- | --- |
| `@:move` | class | Bindings of this type are linear: a value is owned by one binding at a time, and using a binding after its value has been transferred is an error. |
| `@:shared` | class | The type is reference-counted at runtime; `.clone()` lowers to an atomic increment and move-tracking is suppressed. Mutually exclusive with `@:move` (W0030). |
| `@:borrow` | parameter | The callee only observes the argument. The caller keeps its binding, and the value may not outlive the call. |
| `@:owned` | parameter | The callee takes the argument. This is the default; the annotation states it in the signature so a reader does not have to infer it. |
| `@:consume` | method | Calling the method ends the caller's binding on the receiver. |
| `@:manualDrop` | class | The compiler inserts no automatic `Free`; the program is responsible. |
| `@:safety` | class | Historical opt-in marker. Enrolment now follows `@:move`, so this annotation does not itself enable or disable analysis. |

### Parsed, with no consumer

`@:unique`, `@:linear`, `@:affine`, `@:box`, `@:arc`, `@:atomic`, `@:rc`, `@:managed`.

These reach `MemoryAnnotation` and stop at the exhaustive match in
`compiler/src/ir/tast_to_hir.rs`, where each maps to `None`. The match has no `_` arm
on purpose: a new annotation is a compile error until someone decides what it means.
`rayzor.Box`, `rayzor.concurrent.Arc` and `rayzor.Atomic` exist as real extern types —
it is the *annotations* of those names that do nothing.

### Where an annotation can go

Haxe allows metadata on a type, on a field or method, and on a **function parameter**.
It does not allow metadata inside a function body, so a *local binding* can never carry
one:

```haxe
var r = session.cache();      // nothing can be written here
```

This is why the parameter and method positions carry the ownership vocabulary, and why
any future statement about a local has to be carried by its **type** rather than by an
annotation.

### Examples

```haxe
@:move
class Session {
    public var id:Int;
    public function new(id:Int) { this.id = id; }

    // Calling this ends the caller's binding: `s` is unusable afterwards.
    @:consume
    public function close():Void { … }

    public function decode():Int { return id; }
}

class Pipeline {
    // `keep` is observed; the caller still owns it after the call.
    // `give` is taken; the caller's binding ends here.
    static function run(@:borrow keep:Session, give:Session):Int { … }
}
```

## Pipeline Overview

The memory safety pipeline runs as part of semantic analysis, before code generation:

```text
Source Code (with memory annotations)
        |
   Type Checking (TAST)
        |
   Semantic Graph Construction (CFG, DFG/SSA, Call Graph)
        |
   Ownership Graph Analysis
   - Track ownership kinds per variable
   - Record borrow edges and move edges
   - Detect aliasing violations and use-after-move
        |
   Lifetime Analysis
   - Create lifetime regions from CFG scopes
   - Assign lifetimes to SSA variables
   - Generate constraints from code structure
        |
   Constraint Solver
   - Union-Find for equality constraints
   - Outlives graph for ordering constraints
   - Tarjan's SCC for cycle detection
   - Kahn's topological sort for ordering
        |
   Global Lifetime Constraints
   - Inter-procedural analysis across call graph
   - Call site constraint generation
   - Recursive function group handling (SCCs)
   - Virtual method lifetime polymorphism
        |
   Escape Analysis
   - Detect allocation sites in DFG
   - Trace escape via def-use chains
   - Identify stack allocation opportunities
        |
   Send/Sync Validation
   - Thread::spawn closure capture validation
   - Channel<T>: T must be Send
   - Arc<T>: T must be Send + Sync
        |
   Drop Analysis (during HIR/MIR lowering)
   - Last-use analysis per variable
   - Insert Free instructions at drop points
   - Handle lambda captures as escaping
        |
   Code Generation (with memory instructions)
```

---

## Move Checking (MoveFlow)

`compiler/src/ir/mir/moveflow.rs`. This is the pass that actually enforces `@:move`,
`@:borrow`, `@:owned` and `@:consume`. It is a forward may-analysis over the MIR
control-flow graph: a binding is *moved* at a point when **some** path reaching that
point moved it.

### Keyed on bindings, never on registers

The property is about bindings, so the analysis is keyed on `SymbolId`. A register
cannot stand in for a binding: `var d = b` needs no cast, so `d` and `b` are the same
register, and a register-keyed check cannot tell a read of `d` from a read of the
binding that was moved into it. An earlier register-keyed liveness check rejected
correct code for exactly this reason.

### Events, recorded while lowering

Events are recorded where the symbol is still in hand, then replayed against the
finished CFG:

| Event | Recorded at |
| --- | --- |
| `Bind` — the binding starts (or restarts) owning a value | `var x = …`, an assignment, a by-value `@:move` parameter at function entry |
| `Move` — the value is transferred away | a variable on the right of a binding, a call or constructor argument, the receiver of an `@:consume` call |
| `Read` — the binding is observed | every variable read, and every closure capture |

The instruction stream is deliberately **not** the input. `MarkMoved` and `CheckLive`
both report their operand as a use, so a use/def scan over MIR would count a move as a
use of itself at its own program point, and the two would cancel.

### The rules

- **A move is checked, then set.** A move first tests the incoming state — moving an
  already-moved value is itself a violation — and only then records the move. This is
  what makes `f(a); g(a)` fire; nothing anywhere compares two source locations.
- **A bind kills.** Reassignment revives a binding. There is no other revival: a moved
  binding does not come back at scope exit.
- **Fixpoint first, report second.** A loop body is diagnosed once rather than once per
  iteration.
- **Unreachable blocks stay at bottom** and report nothing. This is what keeps
  `if (flag) return take(r); return r.v;` silent — the read is only reachable on the
  path where no move happened.
- **The witness is the earliest move**, joined by minimum program order. The transfer
  takes the minimum too; "keep whichever arrived" is not monotone and the fixpoint can
  oscillate instead of settling.

### Arguments and receivers

A method call is desugared to `method(receiver, args…)`, so the receiver occupies
`args[0]`. It is **borrowed by default** — a receiver has no parameter to annotate, and
counting it as a move would end the binding at the first `a.f()`. `@:consume` is what
changes that; when it applies, the receiver holds slot 0 and the declared parameters
shift by one.

Argument moves are recorded **after** the call lowers, never before: an argument's own
read is recorded while it lowers, and a move ordered ahead of it would make the first
call report against itself.

A call returning `Void` lowers to no destination register. Recording therefore happens
before that early return, or every `f(a);` in statement position — which is how a
consuming call is usually written — would go unchecked while `var x = f(a);` was checked.

### Borrows may not outlive the call (E0383)

`@:borrow` on its own would silence the check that works and promise nothing back. A
borrowed parameter is therefore rejected when it reaches somewhere that outlives the
call: returned, stored through a field or index, or captured by a closure. Copying it
into a local first does not launder it — the local inherits the borrow's root, and the
diagnostic names the alias it travelled through.

This is checked at the escape site rather than by dataflow: an escape on any path is an
escape, so there is nothing for a fixpoint to decide.

Reading a *field* of a borrow is not an escape. `return r.v` must stay legal, or the
annotation would be safe by being useless.

### Crossing module boundaries

Every fact the analysis needs at a call site is published under the callee's **qualified
name** as well as its `SymbolId`, because a callee lowered in another compilation context
has a different `SymbolId` there. `@:move` classes, parameter ownership and `@:consume`
methods each have a name-keyed registry in `compiler/src/ir/mir/mod.rs`.

The two directions are not symmetric, and the asymmetry decides how much the registry
matters. Missing a `@:move` or a `@:consume` across a boundary **under-reports** — the
safe direction. Missing a `@:borrow` makes the compiler report correct code as a
use-after-move: a false positive, which is worse than the error it claims to be.

### Cost

The analysis emits no instructions. A compiled artifact is byte-identical with and
without it; the check is a compile-time reading of events, and the only runtime
ownership machinery (`MarkMoved` / `CheckLive`) predates it and is now redundant.

### What it does not see

- **Places.** The analysis is keyed on `SymbolId`, and a field is not a binding. Rather
  than track one silently and wrongly, taking a `@:move` value **out of** a field is
  refused (E0384). Borrowing a field, reading through one, and calling a method on one
  are all unaffected — what is refused is carrying the value away.
- **Constructor parameters.** Every constructor argument is a move; `@:borrow` on a
  constructor parameter is ignored.
- **Receiver escape.** E0383 covers parameters; a receiver escaping its method is silent.
- **Lambda bodies.** A capture is checked at the point of capture, in the enclosing
  function. The body lowers into its own graph with its own recorder, so an acquire or
  a move *inside* a closure is not seen by the frame that owns the value.
- **Values reached through a deref.** `@:autoDeref` rewrites `h.close()` to
  `h.get().close()`, and a receiver that is a call result names no binding — so a
  `@:consume` reached through a wrapper consumes nothing the analysis can record. This
  is the place-domain limitation above, seen from the other side.

### Oracle

`compiler/tests/move/` holds the cases, with `check.sh` and a README. Each declares
`ERROR` or `SILENT`, and every case runs cold because a warm cache skips MIR lowering
and the analysis with it.

Two habits that the cases exist to enforce:

- **A `SILENT` case must be silent for the right reason.** A checker that detects nothing
  passes every silent case. Where an annotation is what makes a case silent, there is a
  control that removes the annotation and must flip it to `ERROR`.
- **A helper's return type is part of the case.** All eighteen cases once used an
  `Int`-returning helper and reported 18/18 while void calls were entirely unchecked.

## Planned: `RefMut<T>`, a scoped exclusive borrow

**Status: designed, not built. The design below has been measured against the
compiler as it stands, and the prerequisites are larger than the feature.** Nothing
in this section is implemented; it is written down so the cost is visible before
anyone starts.

### What it is for

Everything above answers *who owns a value after a call*. None of it answers
*may anyone else touch it while I hold it* — exclusivity. `@:move` gives a unique
**binding**, which is adjacent but not the same: once a value is inside a structure
there is no way to take a temporary exclusive view and give it back.

`RefMut<T>` is that view. While one is live, the owner may not be read, written or
borrowed again; when it goes out of scope, the owner comes back.

### Why it must be a class

`RefMut<T>` cannot join `Ptr` / `Ref` / `Box` in the pointer-abstract family.
`resolve_receiver_class_symbol` (`compiler/src/ir/mir/resolve/classes.rs`) matches
`Class`, `TypeAlias` and `GenericInstance`, and breaks on everything else — there is
no `Abstract` arm, so `@:move` on an abstract is **inert** and a handle written as an
extern abstract would carry no ownership checking whatsoever.

It also cannot be metadata. Haxe permits no annotation inside a function body, so a
statement about a **local** binding — which is exactly what a borrow handle is — can
only be carried by its type.

### The mechanism, and what already works

Acquiring is a move: `new RefMut(c)` transfers `c` into the handle, because every
constructor argument is a move. That much needs no compiler change, and it already
gives real exclusivity for a local owner:

| shape | today |
| --- | --- |
| two acquires of one owner | error |
| reading the owner while borrowed | error |
| mutating the owner while borrowed, including through a method | error |
| duplicating the handle (`var h2 = h1`) | error, when the *handle* is also `@:move` |

The annotation contract is precise and not obvious: **`@:move` on the owner class is
what arms the acquire check**, and `@:move` on the handle is separately needed or
handles duplicate by plain assignment. `@:safety` is not the gate — it is inert.

What is missing is the **release**. MoveFlow revives a binding only on reassignment,
so today an acquired owner never comes back: block exit, loop iteration and an
explicit `release()` all leave it dead. Every realistic usage shape is currently a
hard error, which makes release the critical path rather than a refinement.

### Six prerequisites, each verified

**1. Release has no hook.** `exit_drop_scope` is the wrong place: `if`/`else` arms,
bare nested blocks, switch arms and function bodies push no drop scope at all, and
where one does fire (a loop-body tail) it lands in the merge block after the handle's
real scope has ended. Release needs a new scope notion in lowering, and it must emit
one `Bind` per live handle, ordered outer-before-inner, and survive `break`,
`continue`, early `return` and `throw`.

**2. The accessor is the exclusivity hole.** `@:autoDeref` — the thing that would make
a class handle ergonomic — requires a `get():T` returning the owner *by value*. That
accessor hands back a second unrestricted owner, and the new binding is a fresh
tracked symbol that the analysis reads as legitimately owned. Making `get()` private
does not close it, because privacy is not enforced. Either `@:autoDeref` lowers
`h.field` to a direct projection on the handle's stored pointer so the owner is never
materialised as a bindable value, or the design gives up ergonomics.

**3. Acquiring from a field — CLOSED.** `new RefMut(o.cell)` used to record nothing, so
two such handles coexisted happily. Taking a `@:move` value out of a field is now
refused outright (E0384), which makes the field acquire a diagnostic rather than a
silent hole. Copying the field to a local first is refused by the same rule.

**4. `@:shared` owners make it a no-op.** Move tracking is suppressed for `@:shared`
classes, so a handle over one checks nothing. That family includes the reference-counted
tensor types — the aliasing-prone values a scoped exclusive borrow is most wanted for.
Silence there is worse than absence, so an acquire over a `@:shared` owner must be
refused at the acquire site.

**5. The handle costs an allocation, and leaks the owner.** The handle is a real
16-byte heap allocation per acquire. Scalar replacement erases it — but declines the
moment the handle pointer is an argument to a call the inliner refused, which is
exactly the lending idiom. Worse, the handle's constructor stores the owner pointer
into a field, and `insert_free` reads that store as an escape, so it stops freeing the
**owner**. Measured over 4,000,000 acquires handed to a non-inlinable callee: 66 MB of
owners never freed, growth linear in acquire count, no diagnostic.

The mitigation is measured and must become the documented rule: **lend the deref'd
value, never the handle.** `kernel(h.get(), i)` allocates and frees exactly as the
handle-free control does; `kernel(h, i)` leaks.

**6. Disjoint borrows are impossible.** Borrowing two fields of one object
independently is the headline reason to want a borrow checker, and MoveFlow has no
place domain — `o.k` and `o.v` are indistinguishable from `o`. The design's two
outcomes for that case are a hard error or an unchecked local copy. Neither is the
feature a user came for.

### Lowering

Compile-time only, if the prerequisites are met. The handle holds the owner's pointer
and nothing else; acquire and release emit no instructions, exactly as the rest of the
ownership analysis does. Under prerequisite 5 that is true only where scalar
replacement fires, so "zero cost" is a property of the *usage rule*, not of the design.

The runtime ownership opcodes (`MarkMoved` / `CheckLive`) are not part of this. They
are emitted at one site, discarded by the LLVM backend and enforced only by Cranelift,
and a release that binds in MoveFlow would not clear the runtime slot. If a dynamic
check is ever wanted — the `RefCell` shape, for the cases a static rule cannot reach —
it should be built deliberately on that slot rather than inherited by accident.

### Recommendation

Prerequisites 1, 2 and 5 are each comparable in size to the rest of the ownership
model, and 6 is the one users would actually ask for.

The first step has been taken, and it was narrower than the feature: the acquire
behaviour that already works is pinned by `c32` and `c33`, and the field case that was
silently wrong is now a refusal (prerequisite 3). What remains before any of this is a
feature is release (1), the accessor (2), the allocation and owner leak (5), and a
decision about `@:shared` owners (4) — for which the refusal has to happen at an acquire
site that does not exist yet, since there is no `RefMut` type to hang it on.

## The `semantic_graph` analyses

The sections below describe the analyses in `compiler/src/semantic_graph/`. They predate
MoveFlow and are a separate body of code; read them as a description of that
infrastructure rather than as a second account of the rules above.

What is wired today:

- **`OwnershipGraph::check_use_after_move`** runs, from
  `CompilationUnit::check_ownership_violations`. It works on expression shape rather than
  on control flow, and it knows nothing of `@:borrow`, `@:owned` or `@:consume`. It
  therefore **defers on `@:move` types**, which MoveFlow owns, and emits only the
  non-fatal E0382 *warning* for ordinary Haxe.
- The lifetime analysis, constraint solver and escape analysis exist in that directory.
  Before relying on any claim in the sections below, check whether the pass has a caller —
  several describe intended behaviour rather than behaviour that runs.

Nothing in `semantic_graph` enforces the annotations documented above.

## 1. Ownership Analysis

**File:** `compiler/src/semantic_graph/ownership_graph.rs`

The ownership graph tracks every variable's ownership state and all borrowing/moving relationships.

### Ownership Kinds

```text
Owned       - Full ownership. Can move, mutate, and drop.
Borrowed    - Immutable borrow. Read-only access, cannot move or mutate.
BorrowedMut - Mutable borrow. Exclusive modification access.
Shared      - Reference-counted shared ownership (for Haxe interop / @:rc / @:arc).
Moved       - Ownership transferred. Variable is no longer accessible.
Unknown     - Analysis could not determine ownership (conservative).
```

### Core Data Structures

**OwnershipNode** -- per-variable tracking:
- `ownership_kind`: Current ownership state
- `lifetime`: Assigned lifetime ID
- `borrowed_by`: List of borrow edges pointing to this variable
- `borrows_from`: List of borrow edges this variable holds
- `is_moved`: Whether ownership has been transferred away
- `allocation_site`: Where the variable was allocated (from DFG)

**BorrowEdge** -- borrowing relationship:
- `borrower` / `borrowed`: The two ends of the borrow
- `borrow_type`: Immutable, Mutable, or Weak
- `borrow_scope`: The scope in which the borrow is active
- `borrow_lifetime`: How long the borrow persists

**MoveEdge** -- ownership transfer:
- `source` / `destination`: The two ends of the move
- `move_type`: Explicit, Implicit, Call (argument passing), or Destruction
- `invalidates_source`: Whether the source becomes unusable

### Violation Detection

The ownership graph detects four categories of violations:

1. **Use After Move** -- accessing a variable after its ownership was transferred
2. **Aliasing Violation** -- holding both mutable and immutable borrows simultaneously
3. **Dangling Pointer** -- using a reference after the referent's lifetime has ended
4. **Double Free** -- multiple deallocation of the same resource

---

## 2. Lifetime Analysis

**File:** `compiler/src/semantic_graph/analysis/lifetime_analyzer.rs`

Lifetime analysis determines how long each variable and reference must remain valid.

### Analysis Phases

The analyzer runs 5 phases per function:

1. **Create Lifetime Regions** -- build a hierarchy of regions from CFG scopes (global, function-level, block-scoped, with parent-child relationships)
2. **Assign Initial Lifetimes** -- map SSA variables to lifetime IDs based on their defining scope, refine from uses (flow-sensitive)
3. **Generate Constraints** -- walk the MIR and produce constraints from field access, array access, memory load/store, function calls, return statements, and phi nodes
4. **Solve Constraint System** -- invoke the constraint solver
5. **Check Violations** -- detect use-after-free, dangling references, and return-of-local-reference

### Constraint Types

```text
Outlives { longer, shorter }          -- 'a must outlive 'b
Equal { left, right }                 -- 'a and 'b are the same lifetime
CallConstraint { callee, args, ret }  -- function call flows
BorrowConstraint { var, lifetime }    -- borrow must not outlive referent
ReturnConstraint { func, ret, params} -- return lifetime bounds
FieldConstraint { object, field }     -- field access lifetime
TypeConstraint { variable, type }     -- type-based lifetime bounds
```

### Violation Types

```text
UseAfterFree          -- Variable used after its lifetime ended
DanglingReference     -- Reference outlives its referent
ReturnLocalReference  -- Returning a reference to a local variable
ConflictingConstraints-- Unsatisfiable constraint system
```

### Inter-Procedural Analysis

**File:** `compiler/src/semantic_graph/analysis/global_lifetime_constraints.rs`

For cross-function analysis, the global constraint system tracks:

- **Function Lifetime Signatures** -- parameter lifetimes, return lifetime, generic lifetime params, lifetime bounds
- **Call Site Constraints** -- how arguments flow to parameters and returns flow to callers
- **Cross-Function Flows** -- lifetime relationships that span function boundaries
- **Recursive Constraint Groups** -- handled via SCC detection in the call graph
- **Virtual Method Constraints** -- lifetime polymorphism for overridden methods

---

## 3. Constraint Solver

**File:** `compiler/src/semantic_graph/analysis/lifetime_solver.rs`

The solver resolves the constraint system produced by lifetime analysis.

### Algorithm (7 phases)

1. **Hash + Cache Check** -- LRU cache lookup for previously solved constraint sets (85-95% hit rate in incremental scenarios)
2. **Union-Find for Equality** -- `Equal` constraints are resolved in O(alpha(n)) using union-find with path compression and union by rank
3. **Build Outlives Graph** -- `Outlives` constraints form a directed graph of lifetime ordering
4. **Cycle Detection** -- Tarjan's algorithm identifies strongly connected components in the outlives graph (O(V+E))
5. **Topological Sort** -- Kahn's algorithm produces a longest-lived to shortest-lived ordering (O(V+E))
6. **Generate Assignments** -- map variables to canonical lifetime representatives
7. **Cache Solution** -- store result in LRU cache for future queries

### Conflict Detection

When constraints are unsatisfiable, the solver reports:

- **OutlivesCycle** -- cyclic outlives relationships (A: B, B: A)
- **EqualityOutlivesConflict** -- equal lifetimes with conflicting outlives
- **ImpossibleConstraints** -- fundamentally unsatisfiable system

### Performance

- Constraint solving: <1ms for typical systems
- Memory: ~20 bytes/constraint + ~40 bytes/variable
- Cache hit ratio: 85-95% for incremental compilation
- Max constraint system: 50,000 constraints (configurable)

---

## 4. Drop Analysis

**File:** `compiler/src/ir/drop_analysis.rs`

Drop analysis determines when and how each variable should be deallocated.

### Drop Behaviors

```text
AutoDrop        -- Compiler inserts a Free instruction at the drop point.
                   Used for heap-allocated classes when drop conditions are met
                   (heap-allocated, non-escaping, at last use). Works regardless
                   of @:safety annotation -- the compiler automatically determines
                   whether a Free is needed based on analysis.

RuntimeManaged  -- The runtime handles cleanup.
                   Used for Thread, Channel, Arc, Mutex.
                   These types have custom Drop implementations in the
                   rayzor-runtime library.

NoDrop          -- No cleanup needed.
                   Used for primitives (Int, Float, Bool), arrays,
                   and Dynamic types. Primitives are value types;
                   Dynamic uses runtime management.
```

### Last-Use Analysis

The drop point analyzer traverses each function body to identify:

1. **All variable uses** -- every statement and expression that references a variable
2. **Last use** -- the final statement index where a variable is referenced
3. **Heap allocations** -- variables created via `new` or allocation calls
4. **Reassignments** -- variables assigned multiple times (drop at reassignment, not last use)
5. **Escaping variables** -- variables returned, passed to functions, or captured by lambdas
6. **Lambda captures** -- variables captured by closures are marked as truly escaping (the closure owns them)

### Drop Point Rules

A `Free` instruction is inserted for a variable when ALL of these conditions hold:
- The variable is heap-allocated
- The variable is NOT escaping (not returned, not captured by lambda, not stored globally)
- The current statement is the variable's last use
- The variable's type has `AutoDrop` behavior

Variables in loops receive special handling -- if the last use is inside a loop, the drop must account for multiple iterations.

---

## 5. Escape Analysis

**File:** `compiler/src/semantic_graph/analysis/escape_analyzer.rs`

Escape analysis determines whether heap-allocated objects can be optimized to stack allocation.

### Escape Classifications

```text
NoEscape            -- Object does not escape its defining scope.
                       Candidate for stack allocation.

EscapesViaReturn    -- Object is returned from the function.
                       Must remain on the heap.

EscapesViaCall      -- Object is passed as an argument to another function.
                       May need heap allocation depending on callee.

EscapesViaGlobal    -- Object is stored in a global variable.
                       Must remain on the heap.

EscapesViaContainer -- Object is stored in another object that itself escapes.
                       Transitively requires heap allocation.

Unknown             -- Conservative assumption when analysis is incomplete.
                       Treated as escaping (heap allocation).
```

### Analysis Algorithm

1. **Find Allocation Sites** -- scan the DFG for `Allocation` nodes, constructor calls, and implicit allocations (string concatenation, array operations)
2. **Trace Def-Use Chains** -- for each allocation, follow all uses through the DFG
3. **Classify Escapes** -- each use is checked: return -> EscapesViaReturn, call argument -> EscapesViaCall, store -> EscapesViaGlobal, field/array access -> NoEscape
4. **Generate Optimization Hints** -- NoEscape allocations suggest stack allocation; small non-escaping functions suggest inlining; dead allocations suggest removal

### Optimization Hints

The escape analyzer generates actionable optimization hints:

- **StackAllocation** -- replace `malloc` with stack-based `alloca` for non-escaping objects
- **InlineFunction** -- inline small functions (<10 DFG nodes, single basic block) to expose more escape analysis opportunities
- **RemoveAllocation** -- eliminate allocations whose results are never used
- **CombineAllocations** -- merge multiple small allocations into a single larger one

---

## 6. Send/Sync Validation

**File:** `compiler/src/tast/send_sync_validator.rs`

Rayzor validates thread-safety properties at compile time for concurrent code.

### Validation Rules

**Thread::spawn(closure)**:
- All variables captured by the closure must implement `Send`
- The closure body is analyzed by `CaptureAnalyzer` to identify captures
- Each captured variable is validated against `Send` requirements

**Channel\<T\>**:
- The type `T` must implement `Send`
- Checked at Channel construction time

**Arc\<T\>**:
- The type `T` must implement both `Send` and `Sync`
- Enforced at Arc instantiation

### Deriving Send/Sync

Classes can derive Send and Sync traits:

```haxe
@:safety
@:derive([Send, Sync])
class SharedState {
    var counter: Int;    // Int is Send + Sync
    var name: String;    // String is Send + Sync
}
```

The validator checks that all fields of a `Send`/`Sync` class are themselves `Send`/`Sync`. If a field fails the check, a compile error is emitted.

---

## Runtime Memory Primitives

The `rayzor-runtime` crate provides the low-level memory operations that generated code calls:

```text
rayzor_malloc(size: u64) -> *mut u8       -- Allocate memory
rayzor_realloc(ptr, old_size, new_size)   -- Resize allocation
rayzor_free(ptr: *mut u8, size: u64)      -- Deallocate memory
```

These are pure Rust functions using `std::alloc`, with no C dependencies. They work for both JIT (linked into process) and AOT (compiled into binary) modes.

### Size Tracking

`rayzor_free` requires a size parameter because Rust's `dealloc` needs the `Layout` (size + alignment). This is more efficient than storing the size in a header because:
- `Vec` already tracks its capacity
- `String` already tracks its length
- User objects have compile-time known sizes

### Monomorphized Collections

`Vec<T>` is specialized at compile time to avoid runtime type dispatch:

```text
Vec<Int>   -> VecI32  -> rayzor_vec_i32_push, rayzor_vec_i32_get
Vec<Float> -> VecF64  -> rayzor_vec_f64_push, rayzor_vec_f64_get
Vec<Bool>  -> VecBool -> rayzor_vec_bool_push, rayzor_vec_bool_get
Vec<T*>    -> VecPtr  -> rayzor_vec_ptr_push, rayzor_vec_ptr_get
```

---

## Dynamic Types and GC

For `Dynamic` types -- values whose concrete type is not known at compile time -- Rayzor uses runtime-managed memory. This is the **only** case where garbage collection semantics apply:

- **Dynamic variables**: Type is resolved at runtime; the compiler cannot insert deterministic `Free` instructions
- **Unknown-size objects**: Objects whose size depends on runtime values and cannot be tracked statically
- **Unannotated classes in non-strict mode**: Auto-wrapped in `Rc` (reference counting), not traditional GC

In all other cases (the vast majority of typed Haxe code), ownership-based memory management eliminates the need for GC entirely.

---

## Summary

| Analysis | Purpose | Key Algorithm | Complexity |
| --- | --- | --- | --- |
| **MoveFlow** | **Enforce `@:move` / `@:borrow` / `@:owned` / `@:consume`** | **Forward may-analysis over the MIR CFG, keyed on SymbolId** | **O(B*S) to fixpoint** |
| Ownership Graph | Track ownership state, borrows, moves | Graph traversal | O(V+E) |
| Lifetime Analysis | Determine variable validity periods | Constraint generation | O(V+E) per function |
| Constraint Solver | Resolve lifetime ordering | Union-Find + Tarjan's SCC + Kahn's sort | O(V+E) |
| Global Lifetimes | Cross-function lifetime flows | Call graph SCC analysis | O(F*C) |
| Escape Analysis | Stack vs heap allocation | Def-use chain tracing | O(V+E) |
| Send/Sync | Thread safety validation | Recursive type checking | O(T*F) |
| Drop Analysis | Determine deallocation points | Last-use analysis | O(V) per function |

Where V = variables, E = edges/constraints, F = functions, C = call sites, T = types,
B = basic blocks, S = tracked bindings.

MoveFlow is the row that enforces the annotations; the others are the `semantic_graph`
infrastructure described above.
