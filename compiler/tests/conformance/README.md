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
