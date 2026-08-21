# Haxe conformance

Runs the official HaxeFoundation/haxe `tests/unit` regression corpus through
rayzor and reports what fraction of standard Haxe actually runs.

## Corpus

`tests/unit/src/unit/issues/` — 1165 files, one per upstream GitHub issue, each
a small self-contained regression test. 451 of them extend `unit.Test` and are
runnable; the rest are macro-only, type-declaration-only, or target-specific.

Fetch it without cloning the whole compiler:

```
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/HaxeFoundation/haxe.git haxe-tests
cd haxe-tests && git sparse-checkout set tests/unit tests/misc std
```

## Running

```
SRC=<path>/haxe-tests/tests/unit/src/unit/issues ./scripts/haxe_conformance.sh
LIMIT=25 ./scripts/haxe_conformance.sh      # pilot
```

The official suite depends on `utest`, which is reflection-heavy. `unit/Test.hx`
here is a minimal stand-in providing only what the corpus uses — `eq` (2125
call sites), `t` (533), `f` (265), `noAssert` (139), `feq` (97), `aeq` (35),
`assert` (14), `exc` (10). Each test class gets a `main()` injected as its last
member, so its own `test*()` methods are legitimately reachable.

## Scoring

A test passes only if it exits 0 **and** prints `CONFORMANCE_OK` **and** emits no
`FAILCHECK`. Silence is never a pass — rayzor has several failure modes that
produce no output at all, including SIGTRAP at exit 133, and the existing
`run_haxe_tests.sh` would score those as passes.

Outcomes are `PASS`, `COMPILE_FAIL`, `CRASH` (132/133/134/139), `WRONG_ANSWER`,
`NO_OUTPUT`, `SKIP`, written as TSV so runs diff against each other.

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
