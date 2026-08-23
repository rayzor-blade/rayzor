# Haxe conformance

Runs the official HaxeFoundation/haxe `tests/unit` regression corpus through
rayzor and reports what fraction of standard Haxe actually runs.

## Corpus

`tests/unit/src/unit/issues/` — 1165 files, one per upstream GitHub issue, each
a small self-contained regression test. 451 of them extend `unit.Test` and are
runnable; the rest are macro-only or type-declaration-only.

The corpus is **not vendored**. Haxe's LICENSE says a file carrying no license
header outside `std/` and `libs/` is GPL-2.0-or-later, and none of the 1165
files carries one — so copying them into this Apache-2.0 repository would mix
incompatible licences. `fetch.sh` clones them instead, blobless and sparse, at
the revision in `corpus.pin`, into a gitignored `corpus/`.

## Running

```
./fetch.sh                 # once, and again when corpus.pin moves
./run.sh                   # whole corpus
LIMIT=25 ./run.sh          # pilot
SRC=<path> ./run.sh        # score a clone you already have
```

The official suite depends on `utest`, which is reflection-heavy. `unit/Test.hx`
here is a minimal stand-in providing only what the corpus uses — `eq` (2125
call sites), `t` (533), `f` (265), `noAssert` (139), `feq` (97), `aeq` (35),
`assert` (14), `exc` (10). Each test class gets a `main()` injected as its last
member, so its own `test*()` methods are legitimately reachable.

## What is out of scope

A test that **imports** a target-language package (`php.Syntax`, `cpp.Star`,
`flash.Vector`, `jvm.NativeArray`) asserts about that target's semantics, not
about Haxe, and rayzor will never satisfy it. Those are skipped and the reason
is recorded.

Conditional compilation is deliberately *not* a filter. `#if cpp … #else … #end`
still has a branch that is ours, so judging a file by its `#if` directives would
drop tests that legitimately apply. Conditionals are consulted for one narrower
purpose: deciding which `test*()` methods are live, so the injected `main()`
does not call a method that only exists under another target.

## Scoring

A test passes only if it exits 0 **and** prints `CONFORMANCE_OK` **and** emits no
`FAILCHECK`. Silence is never a pass — rayzor has several failure modes that
produce no output at all, including SIGTRAP at exit 133, and the existing
`run_haxe_tests.sh` would score those as passes.

Outcomes are `PASS`, `COMPILE_FAIL`, `CRASH` (132/133/134/139), `WRONG_ANSWER`,
`NO_OUTPUT`, `SKIP`, written as TSV so runs diff against each other.

A crash whose cause is an uncompiled function records that function's name
(`uncompiled unit.issues.Issue2725.new`) rather than an exit code, which turns
the failing set into a ranked list of what the compiler cannot yet build.

## Layout

```
run.sh              the harness
fetch.sh            pinned, blobless, sparse clone of the upstream corpus
corpus.pin          upstream revision the numbers describe
shims/unit/         minimal Test.hx / ConfCheck.hx, standing in for utest
known-failing/      repros distilled from failures
corpus/             fetched, gitignored
```

## known-failing/

Minimal repros distilled from conformance failures, kept out of the main Haxe
suite so they do not turn it red. Each documents one blocker.

## Verified baseline (2026-08-21)

Oracle is `haxe 4.3.6 --interp` on the same source. rayzor `--llvm --release
--no-cache`.

| probe | rayzor | haxe 4.3.6 |
|---|---|---|
| expression-bodied fn `static function sh() return 8;` | *empty* | `shorthand=8` |
| loop-allocated objects retained in an Array | `0 1 2 3 4` | `0 1 2 3 4` |
| `case Bx(i) if (i > 10)` on `Bx(99)` | `small99` | `big99` |
| `Std.string({x:1})` | `null` | `{x: 1}` |

Three of four are WRONG and every one of them **exits 0**.

On the official issue corpus, 40 tests: 0 pass, and 29 of them fail on a single
defect -- an unqualified call to a method inherited from a superclass in
another module does not resolve.

## Why the existing suite could not see any of this

`run_haxe_tests.sh` scores PASS when the exit code is 0 and no line begins with
`FAIL`. Only **45 of 196** files in `compiler/tests/haxe/` contain a self-check
that can print one; the other 151 pass by exiting 0. So "197/197" is a crash and
regression count, not a correctness count. Report conformance separately rather
than folding it into that number, or progress here will read as regression
there.
