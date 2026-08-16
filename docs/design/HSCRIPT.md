# hscript, backed by the compiler

Rayzor runs Haxe as a script through its own front end, under the package and
API name existing code already imports.

## Why not port the interpreter

`hscript` is a second language. It is dynamically typed, has no generics, and
its operators and `null` handling differ from Haxe's in ways that only show up
in the programs people actually write. An embedder that ships it maintains two
Haxes that disagree at the edges, and every disagreement is a bug report.

Rayzor already has the expensive half of a script engine: a Haxe front end, and
an interpreter that executes the same MIR the JIT compiles. A script run
through it has the semantics of compiled code because it *is* compiled code,
lowered through the same pipeline. It also tiers up — a script function called
enough times is promoted to Cranelift and then to LLVM. A tree-walking
interpreter cannot do that at any level of effort.

So: keep the API, replace the engine.

## The surface

The package is `hscript`, matching upstream, so existing imports resolve:

```haxe
var parser = new hscript.Parser();
var ast    = parser.parseString("player.hp > 0");
var interp = new hscript.Interp();
interp.variables.set("player", player);
var alive  = interp.execute(ast);
```

`hscript.Expr` keeps its shape, so code that builds or matches expression trees
by hand keeps working. `Parser` is Rayzor's parser with a mapping to `Expr`,
which means better diagnostics than upstream. `Interp.execute` lowers to MIR
and runs it.

Parts of the upstream API that describe a tree-walking interpreter and have no
counterpart here — dialect toggles, the checker, the macro bridge — exist and
do nothing. Preserving the signature keeps code compiling; pretending to
implement semantics we do not have would be worse than a no-op.

## Typing

Leniency here does not mean untyped. A script is only as dynamic as its data.
Type information is recovered from four sources, in order of strength:

**Exported values.** A value handed in from the host arrives carrying its
runtime type tag, so the script knows it is a `Player` rather than an object of
unknown shape. Field access resolves against the real class, with real layout
and real dispatch. This is the source upstream cannot have, and it is the
strongest.

**Imports.** A script that names `haxe.io.Bytes` has named a declaration the
compiler already carries fully restored. Everything reachable through that name
is statically typed at no cost. This rides on the same manifest and restore
path the compiler uses for every other module, and inherits its guarantees.

**Usage.** Local inference over the script body: a variable assigned from a
typed expression takes its type, arithmetic constrains to numeric, a call site
constrains its arguments. This covers what the first two did not reach.

**Dynamic.** What is left is genuinely heterogeneous, and is handled by the
same anonymous-object path the rest of the compiler uses, copy-on-write. No
script-specific mechanism.

Nothing is dynamic merely because it appears in a script.

## The host boundary

Objects cross by pointer, copy-on-write. Reads are free and aliased; a write
forks. The host gets a script that cannot corrupt its state, and pays no
marshalling copy for the common case of a script that only reads. A typed
export never enters the boxed path at all, which keeps it clear of the
boxing behaviour that dynamic values are subject to.

Failure is a value. A script's type error is returned for the host to handle,
not printed and exited.

## Overriding Interp

Real deployments subclass `Interp` to intercept `get`, `set` and `call` — it is
how variable resolution gets hooked. Those hooks stay.

They cost something, though: an overridden accessor is a virtual call the type
analysis cannot see through, so every variable access becomes opaque. Rather
than degrade every script for a facility few use, **an overridden `Interp` opts
that script into fully dynamic mode.** Subclass and you get upstream's
flexibility; leave it alone and you get typing and tier-up. The choice is made
by whether the hook exists, so nothing silently changes behaviour underneath a
caller.

## Order of work

1. `Expr`, `Parser`, `Interp.execute` lowering to MIR; exported values carrying
   their type; import-driven typing; dynamic fallback; errors as values.
2. Usage analysis, to shrink what stays dynamic. Persistent bindings, and a
   REPL over the same engine.
3. Capability control — a script reaching `Sys.command` is a decision the host
   makes. Enforced at name resolution, which is cheap there and near-impossible
   to retrofit.
4. Tier-up for hot script functions. Mostly already built.

## What this depends on

The typing model rests on a declaration restored from cache being identical to
one lowered from source — an exported `Player` is only usefully typed if the
script's `Player` is the host's. That invariant is the compiler's, not this
feature's, and it is not fully held today: unresolved references are dropped
during restore, which `RAYZOR_STRICT_BLADE` counts. Scripts inherit whatever
that invariant is worth when they ship.
