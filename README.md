
<p align="center">
<img style="display: block;" src="website/logo.svg" alt="Rayzor Blade Logo" width="250"/>
</p>

# Rayzor

> A Haxe compiler with tiered JIT compilation and native code generation

[![Tests](https://github.com/darmie/rayzor/actions/workflows/tests.yml/badge.svg)](https://github.com/darmie/rayzor/actions/workflows/tests.yml)
[![Examples](https://github.com/darmie/rayzor/actions/workflows/examples.yml/badge.svg)](https://github.com/darmie/rayzor/actions/workflows/examples.yml)
[![Benchmarks](https://github.com/darmie/rayzor/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/darmie/rayzor/actions/workflows/benchmarks.yml)
[![Benchmark Results](https://img.shields.io/badge/Benchmark-Results-blueviolet)](https://darmie.github.io/rayzor/benchmarks/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

---

## Overview

Rayzor is a Haxe compiler written in Rust that generates native code. It runs
your program immediately in a MIR interpreter and promotes hot functions through
Cranelift to LLVM, so you get startup without a compile wait and steady-state
speed without giving it up.

- **Native code generation** — Cranelift and LLVM backends, plus WebAssembly and
  a C99 route that needs no LLVM
- **Tiered execution** — five tiers, promoted per function on profile data
- **Ownership-based memory** — compile-time drop insertion, no garbage collector
- **Incremental compilation** — a per-module cache and a single-file `.rzb`
  bundle that skips compilation entirely
- **One optimization pipeline** — every backend consumes the same optimized SSA

**Not a goal:** language transpilation. The official Haxe compiler already
excels at emitting JavaScript, Python and PHP; Rayzor has no such target and
none is planned.

---

## Quick start

```bash
git clone https://github.com/darmie/rayzor.git
cd rayzor
cargo build --release

rayzor run hello.hx            # execute with tiered JIT
rayzor check hello.hx          # type-check only
rayzor aot hello.hx -o hello   # native binary via LLVM
rayzor bundle hello.hx -o app.rzb
```

Full command and flag reference: **[docs/CLI.md](docs/CLI.md)**.

---

## Architecture

Haxe source is parsed to an AST, macro-expanded, lowered to a typed AST, then to
HIR, then to MIR — an SSA form over basic blocks. MIR is optimized once and
consumed by every backend.

See **[docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md)**
for the pipeline, the IR levels and why each exists, the optimization pass
ordering, the tier ladder, the memory model, and the runtime ABI.

---

## Memory

There is no garbage collector. The compiler decides at compile time when a value
is freed, from last-use and escape analysis, with a MIR-level pass as the
backstop for anything the HIR analysis cannot see.

Ownership is opt-in through annotations:

```haxe
@:move class UniqueResource { ... }      // move semantics, no aliasing
@:arc class SharedState { ... }          // atomic reference counting
@:derive([Send, Sync]) class Data { ... } // thread-safety marker traits
```

`@:safety` on the Main class selects a program-wide mode: strict requires every
class to be annotated, non-strict wraps the rest in `Rc`. Use-after-move is a
hard error for `@:move` types and a warning otherwise.

Details: [MEMORY_MANAGEMENT.md](docs/architecture/MEMORY_MANAGEMENT.md).

---

## Artifacts

| Format | What it is |
|---|---|
| `.blade` | per-module MIR cache for incremental builds — [spec](docs/architecture/BLADE_FORMAT_SPEC.md) |
| `.rzb` | single-file bundle of all modules; `rayzor run app.rzb` skips compilation — [spec](docs/architecture/RZB_FORMAT_SPEC.md) |
| `.rpkg` | package of Haxe sources and optional native libraries |

Caching is on by default; `--no-cache` disables it. The cache is invalidated by
source content, compiler version, and a content-derived compiler cache ABI id.
That ABI id changes when compiler/parser/stdlib inputs change, but not for a
redundant relink of identical sources.

---

## Status

Working today:

- Parser, type checker, macro system (interpreter, reification, `@:build`)
- Full lowering to SSA MIR, monomorphization, four optimization levels
- MIR interpreter, Cranelift JIT, LLVM JIT and AOT, C99 backend
- WebAssembly: core modules, WASI, and P2 components
- Five-tier promotion with a safepoint-style promotion barrier
- Ownership analysis, drop insertion, `Send`/`Sync` validation
- Concurrency runtime: Thread, Channel, Mutex, Arc
- BLADE cache, `.rzb` bundles, `.rpkg` packages
- LSP server, and a `rayzor debug` toolkit (forensic run, bench, git A/B compare,
  crash-PC resolution, live metrics dashboard)

In progress: standard library coverage and optimization tuning.

---

## Design principles

1. **Correctness first.** Optimizations come after correctness is demonstrated,
   and the passes that exist for correctness — drop insertion, guaranteed
   inlining of `inline` — run at every optimization level including `-O0`.
2. **Analysis is infrastructure.** Dominance, loop structure and escape
   information are computed once and queried by many passes.
3. **Ownership over GC.** Memory is freed by compile-time analysis.
4. **Determinism.** Every collection in MIR is ordered, so codegen is
   reproducible.
5. **Incremental everywhere.** Parsing, type checking, module caching and
   bundling all avoid redoing unchanged work.

---

## Comparison with the official Haxe compiler

| | Haxe (official) | Rayzor |
|---|---|---|
| Language support | Full Haxe 4.x | Haxe 4.x, in progress |
| JS / Python / PHP | Excellent | Not a goal |
| Native output | Via C++ | Direct, via Cranelift/LLVM |
| JIT runtime | No | Five-tier, profile-driven |
| Memory model | Garbage collected | Ownership, compile-time |
| Optimizations | Backend-specific | SSA-based, shared by all backends |
| Incremental builds | Limited | Per-module cache |

---

## Documentation

- **[Architecture](docs/architecture/ARCHITECTURE.md)** — pipeline, IRs, passes, tiers, runtime
- **[CLI reference](docs/CLI.md)** — commands, compilation modes, manifest, environment variables
- **[Memory management](docs/architecture/MEMORY_MANAGEMENT.md)** — ownership, lifetimes, drops
- **[BLADE format](docs/architecture/BLADE_FORMAT_SPEC.md)** · **[RZB format](docs/architecture/RZB_FORMAT_SPEC.md)**
- **[Backlog](docs/architecture/BACKLOG.md)** — feature tracking

---

## Contributing

```bash
cargo build
cargo test
./run_haxe_tests.sh      # Haxe end-to-end suite
```

Start with the [architecture doc](docs/architecture/ARCHITECTURE.md). Two
conventions worth knowing before your first patch: symbol ids are per-compilation-context, so only fully-qualified names may
cross module boundaries; and MIR collections are ordered deliberately — do not
swap a `BTreeMap` for a `HashMap`.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

Haxe Foundation, the Cranelift project, LLVM, and the Rust community.

## Contact

[Issues](https://github.com/darmie/rayzor/issues) · [Discussions](https://github.com/darmie/rayzor/discussions)
