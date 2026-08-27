# Haxe conformance

Runs the official HaxeFoundation/haxe `tests/unit` regression corpus through
rayzor and reports what fraction of standard Haxe actually runs.

## Corpus

`tests/unit/src/unit/issues/` — 1165 files, one per upstream GitHub issue, each
a small regression test extending `unit.Test` (qualified or unqualified).
The harness scores 1030 on the current pin; 59 target-specific tests and 76
files with no test method live for this target are reported as skips.

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
JOBS=1 ./run.sh            # force serial execution (default: up to 8 workers)
PRESET=application ./run.sh # use the production conformance preset (default)
SRC=<path> ./run.sh        # score a clone you already have
OUT=<path> ./run.sh        # keep the TSV report at a specific path
```

The harness builds each issue in its own work directory and schedules bounded
parallel workers. Corpus sibling fixtures and the assertion shims live in one
shared, read-only class-path tree, so they are copied once per run instead of
once per issue. Set `JOBS` to tune CPU and memory use; final TSV rows remain in
source order regardless of which worker finishes first.

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

## Verified baseline (2026-08-27)

Rayzor `--release --no-cache --preset application`, corpus revision from
`corpus.pin`, eight workers:

| outcome | count |
|---|---:|
| PASS | 370 |
| WRONG_ANSWER | 189 |
| COMPILE_FAIL | 279 |
| CRASH | 185 |
| TIMEOUT | 7 |
| NO_OUTPUT | 0 |
| SKIP | 135 |

That is 370/1030 scored tests, or 35.9%. The durable report has 1165 rows plus
its header; the completion marker is written only after every worker in a full
corpus run has been reaped. The largest actionable failure families are parser
fallback losing the injected `main`, unresolved class/member metadata, and
named functions left uncompiled. Wrong answers retain their first `FAILVALUES`
line so distinct semantic failures are no longer collapsed into a generic
`FAILCHECK eq`.

## Why the existing suite could not see any of this

`run_haxe_tests.sh` scores PASS when the exit code is 0 and no line begins with
`FAIL`. Only **45 of 196** files in `compiler/tests/haxe/` contain a self-check
that can print one; the other 151 pass by exiting 0. So "197/197" is a crash and
regression count, not a correctness count. Report conformance separately rather
than folding it into that number, or progress here will read as regression
there.
