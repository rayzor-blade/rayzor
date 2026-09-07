# Contributing

Build prerequisites and the LLVM link modes are in the
[README](../README.md#contributing). This page is the diagnostic toolkit: how to
see what the compiler produced and how to corner a bug in it.

User-facing commands are in [CLI.md](CLI.md).

---

## Build and test

```bash
cargo build
cargo test
./run_haxe_tests.sh              # Haxe end-to-end suite
bash haxe-conformance/run.sh     # official Haxe conformance corpus
```

<a id="llvm"></a>Needs LLVM 21 via `LLVM_SYS_211_PREFIX` — see
[README](../README.md#contributing) for both link shapes and why CI builds each.

Two conventions worth knowing before a first patch: symbol ids are
per-compilation-context, so only fully-qualified names may cross module
boundaries; and MIR collections are ordered deliberately — do not swap a
`BTreeMap` for a `HashMap`.

---

## See what a stage produced

```bash
rayzor compile main.hx --stage ast|tast|hir|mir|native
rayzor compile main.hx --stage mir --show-ir
```

## Read the MIR

```bash
rayzor dump main.hx                    # optimized (O2)
rayzor dump main.hx -O0                # before the optimizer
rayzor dump main.hx --diff             # before vs after — what a pass did
rayzor dump main.hx --function advance # one function
rayzor dump main.hx --cfg-only         # blocks and edges, no instructions
rayzor dump main.hx --format dot | dot -Tpng -o cfg.png
rayzor dump main.hx -i                 # interactive viewer
```

`rayzor dump` shows MIR **before** codegen and renumbers block labels. When the
backend disagrees with it, print what codegen actually received instead — see
[DEBUGGING_MIR.md](architecture/DEBUGGING_MIR.md).

## Investigate a crash or a regression

```bash
rayzor debug run main.hx         # crash handlers pre-armed
rayzor debug resolve 0x104f2a1c  # hex PC → Haxe function and line
rayzor debug lldb main.hx        # launch under lldb
rayzor debug bench main.hx       # N runs, per-run and aggregate stats
rayzor debug compare HEAD~1 main.hx   # A/B vs a git ref, median delta
rayzor debug server              # live metrics over HTTP
```

`compare` restores the working tree on exit, including on failure.

## Bisect a miscompile

```bash
RAYZOR_PASS_DEBUG=1 rayzor run main.hx        # run passes one at a time
RAYZOR_DISABLE_PASSES=loop-unrolling rayzor run main.hx
RAYZOR_NO_SRA=1 rayzor run main.hx            # no scalar replacement
RAYZOR_NO_PHI_SRA=1 rayzor run main.hx        # only its phi-aware part
RAYZOR_RAW_MIR=1 rayzor dump main.hx          # dump with no passes at all
```

`RAYZOR_DISABLE_PASSES` takes comma-separated pass names: `constant-folding`,
`control-flow-simplification`, `copy-propagation`, `cse`,
`dead-code-elimination`, `devirtualization`, `gvn`,
`hir-dead-code-elimination`, `inlining`, `licm`, `loop-unrolling`,
`strip-stack-trace-updates`, `tail-call-optimization`,
`unreachable-block-elimination`.

## Look at backend output

```bash
RAYZOR_DUMP_CLIF=1 rayzor run main.hx         # Cranelift IR
RAYZOR_DUMP_LLVM_IR=1 rayzor run main.hx      # LLVM IR around optimization
RAYZOR_DUMP_FN_PTRS=1 rayzor run main.hx      # resolved function pointers,
                                              # and which functions got stubs
RAYZOR_LLVM_OPT=0 rayzor run main.hx          # override the LLVM opt level
```

A trap stub means a function failed to compile and something called it.
`RAYZOR_DUMP_FN_PTRS=1` names it and says why.

## Other knobs

```bash
RAYZOR_STD_PATH=/path/to/haxe-std rayzor run main.hx
RAYZOR_STRICT_MOVE_CHECK=1 rayzor run main.hx  # strict move violations
rayzor preblade                                # extract stdlib symbols to .bsym
```

---

Start with the [architecture doc](architecture/ARCHITECTURE.md) for the pipeline,
the IR levels, pass ordering, the tier ladder and the runtime ABI.
