
<p align="center">
<img style="display: block;" src="website/logo.svg" alt="Rayzor Blade Logo" width="250"/>
</p>

# Rayzor

> A high-performance, next-generation Haxe compiler with tiered JIT compilation and native code generation

[![Tests](https://github.com/darmie/rayzor/actions/workflows/tests.yml/badge.svg)](https://github.com/darmie/rayzor/actions/workflows/tests.yml)
[![Examples](https://github.com/darmie/rayzor/actions/workflows/examples.yml/badge.svg)](https://github.com/darmie/rayzor/actions/workflows/examples.yml)
[![Benchmarks](https://github.com/darmie/rayzor/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/darmie/rayzor/actions/workflows/benchmarks.yml)
[![Benchmark Results](https://img.shields.io/badge/Benchmark-Results-blueviolet)](https://darmie.github.io/rayzor/benchmarks/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)

---

## Overview

**Rayzor** is a complete reimplementation of a Haxe compiler in Rust, designed for:

- **High Performance**: Native compilation via Cranelift and LLVM backends
- **Fast Compilation**: 50-200ms JIT compilation vs 2-5s C++ compilation
- **Tiered JIT**: 5-tier adaptive optimization (Interpreter, Baseline, Standard, Optimized, LLVM)
- **Ownership-Based Memory**: Rust-inspired ownership, lifetime analysis, and automatic drop insertion instead of garbage collection
- **Incremental Compilation**: BLADE module cache and RayzorBundle (.rzb) single-file executables
- **Modern Architecture**: SSA-based IR with optimization passes, monomorphization, and SIMD vectorization infrastructure

### What Makes Rayzor Different?

Unlike the official Haxe compiler which excels at language transpilation (JavaScript, Python, PHP), **Rayzor focuses exclusively on native code generation**:

- **Cranelift** for fast JIT compilation (Tiers 0-2)
- **LLVM** for maximum optimization (Tier 3 + AOT object files)
- **MIR Interpreter** with NaN-boxing for instant startup (Tier 0)
- **Ownership-based memory management** with compile-time drop analysis, not garbage collection

**Not a goal:** Language transpilation - the official Haxe compiler already excels at this.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/darmie/rayzor.git
cd rayzor

# Build the compiler
cargo build --release

# Run tests
cargo test

# Run a Haxe file with tiered JIT
rayzor run hello.hx --preset application

# Compile to native (shows MIR pipeline)
rayzor compile hello.hx --stage native

# Check syntax and types
rayzor check hello.hx
```

---

## Architecture

Rayzor implements a **multi-stage compilation pipeline** with sophisticated analysis and optimization:

```
                    ┌─────────────┐
                    │ Haxe Source │
                    └──────┬──────┘
                           │
                           v
┌─────────────────────────────────────────────────────────┐
│                  Parser (parser/ crate)                  │
│  Nom-based combinators, incremental parsing,            │
│  error recovery, precise source location tracking       │
└────────────────────────┬────────────────────────────────┘
                         │
                         v
                   ┌──────────┐
                   │   AST    │
                   └─────┬────┘
                         │
                         v
┌─────────────────────────────────────────────────────────┐
│       Macro Expansion (compiler/src/macro_system/)       │
│  Tree-walking interpreter, reification ($v,$i,$e,$a,     │
│  $p,$b), @:build/@:autoBuild, Context API, registry     │
└────────────────────────┬────────────────────────────────┘
                         │
                         v
                   ┌──────────┐
                   │ Expanded │
                   │   AST    │
                   └─────┬────┘
                         │
                         v
┌─────────────────────────────────────────────────────────┐
│            Type Checker (compiler/src/tast/)             │
│  Symbol resolution, type inference, constraint solving,  │
│  generics, nullables, abstract types, send/sync traits  │
└────────────────────────┬────────────────────────────────┘
                         │
                         v
                   ┌──────────┐
                   │   TAST   │ (Typed AST)
                   └─────┬────┘
                         │
                         v
┌─────────────────────────────────────────────────────────┐
│      Semantic Analysis (compiler/src/semantic_graph/)    │
│  CFG, DFG (SSA), Call Graph, Ownership & Lifetime       │
│  tracking, Escape Analysis, TypeFlowGuard               │
└────────────────────────┬────────────────────────────────┘
                         │
                         v
                   ┌──────────┐
                   │   HIR    │ (High-level IR)
                   └─────┬────┘
                         │  TAST -> HIR lowering (3,875 LOC)
                         │  Preserve semantics + optimization hints
                         │
                         v
                   ┌──────────┐
                   │   MIR    │ (Mid-level IR - SSA form)
                   └─────┬────┘
                         │  HIR -> MIR lowering (12,994 LOC)
                         │  Platform-independent SSA with phi nodes
                         │  Optimization passes (DCE, const fold,
                         │  copy prop, inlining, loop analysis)
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       v                 v                 v
┌────────────┐    ┌────────────┐    ┌────────────┐
│ MIR Interp │    │ Cranelift  │    │    LLVM    │
│  (Tier 0)  │    │(Tier 1-2)  │    │  (Tier 3)  │
│  NaN-boxed │    │  JIT Fast  │    │  -O3 / AOT │
└────────────┘    └────────────┘    └────────────┘
       │                 │                 │
       └────────┬────────┘                 │
                │                          │
                v                          v
         ┌────────────┐           ┌────────────┐
         │   Native   │           │   Object   │
         │    Code    │           │   Files    │
         └────────────┘           └────────────┘
                │
                v
         ┌────────────┐
         │   BLADE    │ (.blade module cache)
         │   RZB      │ (.rzb single-file bundle)
         └────────────┘
```

### Key Components

#### 1. Parser (`parser/` crate)

- **Technology**: Nom parser combinators for composability
- **Features**: Incremental parsing, error recovery, precise source location tracking, enhanced diagnostics

#### 2. Macro System (`compiler/src/macro_system/`)

- **Tree-walking interpreter** for compile-time macro function evaluation
- **Reification engine**: `$v{}`, `$i{}`, `$e{}`, `$a{}`, `$p{}`, `$b{}` dollar-identifier splicing
- **Build macros**: `@:build` and `@:autoBuild` metadata-driven class field generation
- **Context API**: `haxe.macro.Context` methods (error, getType, getBuildFields, defineType, parse)
- **Pipeline integration**: Expansion stage between parsing and type checking
- **Error recovery**: Per-field error handling preserves original AST on failure
- **Expansion origin tracking**: Traces errors through macro expansions to call sites

#### 3. Type Checker (`compiler/src/tast/`)

- Bidirectional type checking with constraint-based type inference
- Rich type system: Generics, nullables, abstract types, function types
- Send/Sync trait validation for concurrency safety
- Memory annotations: `@:move`, `@:unique`, `@:borrow`, `@:owned`, `@:arc`, `@:rc`

#### 4. Semantic Analysis (`compiler/src/semantic_graph/`)

Production-ready analysis infrastructure built in SSA form:

- **Control Flow Graph (CFG)**: Basic blocks, dominance, loop detection
- **Data Flow Graph (DFG)**: SSA form, def-use chains, value numbering
- **Call Graph**: Inter-procedural analysis, recursion detection
- **Ownership Graph**: Move semantics, borrow checking, aliasing violation detection
- **Lifetime Analysis**: Constraint-based solver with region inference
- **Escape Analysis**: Stack vs heap allocation optimization

#### 5. HIR (High-level IR)

Preserves high-level language features (closures, pattern matching, try-catch) with resolved symbols, optimization hints, and lambda capture modes (ByValue/ByRef/ByMutableRef).

#### 6. MIR (Mid-level IR) - ~31,000 LOC

Platform-independent optimization target in full SSA form:

- **Instructions**: Value ops, memory ops (Alloc/Free/BorrowImmutable/BorrowMutable), closure ops (MakeClosure/CallIndirect), SIMD vector ops
- **Optimization Passes**: Dead code elimination, constant folding, copy propagation, unreachable block elimination, control flow simplification
- **Advanced Infrastructure**: Function inlining with cost model, loop analysis with trip count estimation, SIMD vectorization (V4F32, V2F64, V4I32)
- **Validation**: Ownership state tracking, SSA invariants, borrow overlap detection

#### 7. Code Generation Backends

| Backend | Tier | Compilation | Speed | Status |
|---------|------|-------------|-------|--------|
| **MIR Interpreter** | 0 | Instant | ~5-10x native | Complete |
| **Cranelift (none)** | 1 | ~3ms/function | ~15x native | Complete |
| **Cranelift (speed)** | 2 | ~10ms/function | ~20x native | Complete |
| **Cranelift (speed_and_size)** | 2 | ~30ms/function | ~25x native | Complete |
| **LLVM (-O3)** | 3 | ~500ms/function | ~50x native | Complete |

#### 8. Tiered JIT System

HotSpot JVM-inspired adaptive compilation with safe tier promotion:

- **PromotionBarrier**: Atomic function pointer replacement with execution draining
- **Background Worker**: Async optimization on separate thread with Rayon parallelism
- **Presets**: Script, Application, Server, Benchmark, Development, Embedded
- **BailoutStrategy**: Configurable interpreter-to-JIT transition thresholds (10 to 10,000 block executions)

#### 9. Incremental Compilation (BLADE)

- **BLADE Cache** (`.blade`): Per-module binary cache with source hash validation and dependency tracking. ~30x faster incremental builds.
- **RayzorBundle** (`.rzb`): Single-file executable format containing all compiled modules, symbol manifest for O(1) startup, and build metadata.

#### 10. Memory Management

Rayzor uses **ownership-based memory management** with compile-time analysis, not garbage collection:

- **Drop Analysis**: Automatic `Free` insertion at last-use points with escape tracking
- **Three Drop Behaviors**: AutoDrop (heap-allocated classes meeting drop conditions), RuntimeManaged (Thread/Arc/Channel), NoDrop (primitives)
- **Ownership Tracking**: Move semantics, borrow checking, use-after-move detection
- **Lifetime Analysis**: Constraint-based solver, region inference, inter-procedural analysis

See [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) for the complete strategy.

#### 11. Diagnostics System

Rich error messages with source locations, suggestions, and error codes.

---

## Compilation Modes

Rayzor supports **three compilation strategies** optimized for different use cases:

### 1. Development Mode (JIT)

```bash
rayzor run main.hx --preset development
```

- **Fast iteration**: 50-200ms compilation via Cranelift JIT
- **MIR Interpreter**: Instant startup for cold paths
- **Good performance**: 15-25x interpreter speed
- **Use case**: Rapid prototyping, development

### 2. JIT Runtime Mode (Tiered)

```bash
rayzor run main.hx --preset application
```

- **Adaptive optimization**: MIR interpreter -> Cranelift -> LLVM
- **Profile-guided**: Auto-detect hot functions via execution counters
- **Background compilation**: LLVM optimizes hot code while running
- **Safe promotion**: PromotionBarrier ensures atomic function pointer swap
- **Use case**: Long-running applications, servers

**Execution Flow:**
```
Function First Call:
  1. Interpret with NaN-boxed MIR interpreter (instant)
  2. Profile execution via atomic counters
  3. At threshold → compile with Cranelift (~3-30ms)
  4. If stays hot → compile with LLVM in background (~500ms)
  5. PromotionBarrier drains execution, swaps pointer atomically

Result: Instant start → 15-25x (Cranelift) → 45-50x (LLVM)
```

### 3. AOT Production Mode

```bash
rayzor compile main.hx --stage native
```

- **LLVM -O3**: Maximum optimization for all code
- **Object file generation**: `.o` files for system linker integration
- **BLADE caching**: `--cache` flag for incremental builds
- **Use case**: Production deployments, CI/CD pipelines

---

## CLI Reference

### Project Management

```bash
# Initialize a new project
rayzor init --name my-app

# Initialize a workspace (multi-project)
rayzor init --name my-workspace --workspace

# Build from rayzor.toml (auto-detected)
cd my-app && rayzor build

# Build from .hxml (backwards compatible)
rayzor build build.hxml

# Run using entry point from rayzor.toml
cd my-app && rayzor run
```

### `rayzor init`

Scaffolds a new project or workspace.

```bash
rayzor init [--name <NAME>] [--workspace]
```

- **Without `--workspace`**: Creates `rayzor.toml`, `src/Main.hx`, `.rayzor/cache/`, `.gitignore`
- **With `--workspace`**: Creates a workspace `rayzor.toml` with empty `members` list

### `rayzor run`

Runs a Haxe source file with tiered JIT compilation.

```bash
rayzor run [FILE] [--preset <PRESET>] [--cache] [--cache-dir <DIR>]
```

If `FILE` is omitted, reads the entry point from `rayzor.toml` in the current directory.

### `rayzor build`

Compiles a project from `.hxml` or `rayzor.toml`.

```bash
rayzor build [FILE] [--cache] [--cache-dir <DIR>]
```

Auto-detection order:

1. If `FILE` ends with `.hxml` → parse as HXML build file
2. If `FILE` is omitted and `rayzor.toml` exists → build from manifest
3. If manifest has `hxml = "build.hxml"` → delegate to HXML parser

### `rayzor bundle`

Creates a RayzorBundle (`.rzb`) single-file executable.

```bash
rayzor bundle <FILES...> [--output <PATH>] [--opt-level <0|1|2|3>] [--strip] [--no-compress] [--cache] [--cache-dir <DIR>] [--verbose]
```

- `--strip`: Tree-shake unreachable code
- `--no-compress`: Disable zstd compression
- `--cache`: Enable BLADE incremental cache

### `rayzor aot`

Ahead-of-time compilation via LLVM (requires `llvm-backend` feature).

```bash
rayzor aot <FILES...> [--output <PATH>] [--target <TRIPLE>] [--emit <FORMAT>] [--opt-level <0|1|2|3>] [--strip] [--strip-symbols] [--verbose]
```

- `--emit`: `exe` (default), `obj`, `llvm-ir`, `llvm-bc`, `asm`
- `--target`: Target triple for cross-compilation (default: host)
- `--strip-symbols`: Strip debug symbols from binary

### `rayzor preblade`

Extracts stdlib symbols and generates precompiled BLADE caches.

```bash
rayzor preblade [FILES...] [--out <DIR>] [--list] [--cache-dir <DIR>] [--verbose]
```

### Other Commands

```bash
rayzor check <FILE>                  # Type-check without compiling
rayzor compile <FILE> --stage native # Compile to native code
rayzor jit <FILE>                    # Run with Cranelift JIT
rayzor cache stats                   # View BLADE cache statistics
rayzor cache clear                   # Clear BLADE cache
rayzor info                          # Show compiler info
```

### Project Manifest (`rayzor.toml`)

#### Single Project

```toml
[project]
name = "my-app"
version = "0.1.0"
entry = "src/Main.hx"

[build]
class-paths = ["src"]
opt-level = 2
preset = "application"
output = "build/my-app"

[cache]
enabled = true
```

#### Workspace

```toml
[workspace]
members = ["game", "engine", "tools/level-editor"]

[workspace.cache]
dir = ".rayzor/cache"
```

#### HXML Delegation

```toml
[project]
name = "legacy-app"
hxml = "build.hxml"
```

---

## Performance Targets

### Compilation Speed

| Mode | Target | Goal | vs. Haxe/C++ |
|------|--------|------|--------------|
| **JIT** | MIR Interpreter | Instant | N/A |
| **JIT** | Cranelift (Tier 1) | ~3ms/function | ~100x faster |
| **JIT** | Cranelift (Tier 2) | ~30ms/function | ~20x faster |
| **JIT** | LLVM (Tier 3) | ~500ms/function | Similar |
| **AOT** | LLVM | 10-30s | Similar |

### Runtime Performance

| Target | Goal | Use Case |
|--------|------|----------|
| MIR Interpreter | ~5-10x native | Instant startup |
| Cranelift (Tiers 1-2) | 15-25x interpreter | Fast compilation |
| LLVM (Tier 3 + AOT) | 45-50x interpreter | Maximum speed |

---

## Project Status

### Complete

| Component | Coverage | LOC |
|-----------|----------|-----|
| Parser | ~95% | Incremental, error recovery, enhanced diagnostics |
| Type Checker | ~85% | Inference, generics, abstract types, Send/Sync |
| Semantic Analysis | ~90% | CFG, DFG/SSA, call graph, ownership, lifetime, escape |
| HIR | ~95% | All Haxe features, lambda captures, optimization hints |
| MIR | ~95% | Full SSA, phi nodes, 31k LOC across IR modules |
| Optimization Passes | ~70% | 5 core passes + inlining, loop analysis, vectorization |
| Cranelift Backend | ~90% | JIT compilation, 3 optimization levels, ARM64 support |
| LLVM Backend | ~85% | -O3 optimization, object file generation |
| MIR Interpreter | ~90% | NaN-boxing, all MIR instructions |
| Tiered JIT | ~90% | 5 tiers, safe promotion, background worker, presets |
| BLADE/RZB | ~80% | Module cache, bundle format, source hash validation |
| Drop Analysis | ~85% | Last-use analysis, escape tracking, 3 drop behaviors |
| Monomorphization | ~85% | Lazy instantiation, caching, recursive generics |
| Macro System | ~85% | Interpreter, reification, @:build/@:autoBuild, Context API |
| Runtime | ~80% | Thread, Channel, Mutex, Arc, Vec, String, Math, File I/O |

### In Progress

- **AOT Binary Output**: Object file generation works via LLVM; full binary linking pipeline pending
- **Standard Library Expansion**: Generic collections (Vec\<T\>, Option\<T\>, Result\<T,E\>)
- **Optimization Tuning**: Tier promotion thresholds and bailout strategy profiling

### Next Steps

1. **AOT Codegen Pipeline**: Integrate LLVM object file output with system linker for standalone binaries
2. **WebAssembly Backend**: Direct WASM compilation target
3. **IDE Support**: LSP server for editor integration
4. **Full Haxe Standard Library**: Complete API coverage

---

## Incremental Compilation

### BLADE Module Cache

The **BLADE** (Blazing Language Artifact Deployment Environment) system provides per-module binary caching:

```bash
# Enable caching for incremental builds
rayzor run main.hx --cache

# Specify cache directory
rayzor compile main.hx --cache --cache-dir ./build-cache

# View cache statistics
rayzor cache stats

# Clear cache
rayzor cache clear
```

**How it works:**
- Each module is serialized to a `.blade` file using postcard binary format
- Source file hash validates cache freshness (more reliable than timestamps)
- Dependency tracking enables transitive cache invalidation
- ~30x faster incremental builds for unchanged modules

### RayzorBundle (.rzb)

Single-file executable format for deployment:

```
┌─────────────────────────────────┐
│ Header: RZBF magic, version     │
│ Metadata: entry module/function │
│ Module Table: name → offset     │
│ Modules: serialized IrModules   │
│ Symbol Manifest (optional)      │
│ Build Info: compiler, platform  │
└─────────────────────────────────┘
```

- **O(1) startup**: Entry module/function index stored in header
- **Symbol Manifest**: Pre-resolved symbols eliminate re-parsing
- **Reproducible**: Build info captures compiler version and platform

See [BLADE_FORMAT_SPEC.md](BLADE_FORMAT_SPEC.md) and [RZB_FORMAT_SPEC.md](RZB_FORMAT_SPEC.md) for format details.

---

## Memory Management

Rayzor uses **ownership-based memory management** instead of garbage collection. The compiler performs static analysis to determine when objects should be freed, inserting `Free` instructions at compile time.

### Key Principles

1. **Ownership Tracking**: Each heap allocation has a single owner. Ownership transfers via move semantics.
2. **Drop Analysis**: The compiler identifies the last use of each variable and inserts `Free` after it.
3. **Escape Analysis**: Objects that escape their scope (returned, captured by lambdas) are not freed prematurely.
4. **Three Drop Behaviors**:
   - `AutoDrop` - Heap-allocated classes: compiler inserts `Free` when drop conditions are met
   - `RuntimeManaged` - Concurrency types (Thread, Arc, Channel): runtime handles cleanup
   - `NoDrop` - Primitives and Dynamic values: no cleanup needed

### Opt-In Annotations

```haxe
@:move class UniqueResource { ... }     // Move semantics, no aliasing
@:arc class SharedState { ... }          // Atomic reference counting
@:derive([Send, Sync]) class Data { ... } // Thread-safe marker traits
```

GC is reserved only for `Dynamic` typed values or types whose size cannot be determined at compile time.

See [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md) for the complete strategy.

---

## Documentation

- **[MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md)** - Memory management strategy (ownership, lifetimes, drops)
- **[ARCHITECTURE.md](compiler/ARCHITECTURE.md)** - Complete system architecture
- **[SSA_ARCHITECTURE.md](compiler/SSA_ARCHITECTURE.md)** - SSA integration details
- **[RAYZOR_ARCHITECTURE.md](compiler/RAYZOR_ARCHITECTURE.md)** - Vision, roadmap, tiered JIT
- **[BLADE_FORMAT_SPEC.md](BLADE_FORMAT_SPEC.md)** - BLADE module cache format
- **[RZB_FORMAT_SPEC.md](RZB_FORMAT_SPEC.md)** - RayzorBundle executable format
- **[RUNTIME_ARCHITECTURE.md](RUNTIME_ARCHITECTURE.md)** - Runtime library and extern functions
- **[BACKLOG.md](BACKLOG.md)** - Feature backlog and progress tracking

---

## Design Philosophy

### 1. Correctness First

```
Correctness -> Safety -> Clarity -> Performance
```

The compiler prioritizes generating correct code. Performance optimizations come after correctness is proven.

### 2. Layered Architecture

Each layer has a single, well-defined responsibility. Information flows forward through explicit interfaces.

### 3. Analysis as Infrastructure

Complex analyses (SSA, CFG, DFG, ownership, lifetime, escape) are built once and queried by multiple passes. This enables sophisticated optimizations without code duplication.

### 4. Ownership Over GC

Memory is managed through compile-time ownership analysis and automatic drop insertion. Garbage collection is not used for statically-typed code. Only `Dynamic` values or types with unknown compile-time sizes fall back to runtime-managed memory.

### 5. Incremental Everything

Support incremental operations at every level:
- Incremental parsing (re-parse only changed regions)
- Incremental type checking (re-check only affected code)
- Incremental analysis (re-analyze only dependencies)
- Incremental codegen (re-generate only changed functions)
- BLADE cache (skip unchanged modules entirely)

### 6. Developer Experience

- **Fast feedback**: Instant MIR interpretation, 3-30ms Cranelift JIT
- **Rich diagnostics**: Helpful error messages with suggestions
- **Tiered presets**: Script, Application, Server, Development, Embedded
- **Modern tooling**: Profiling, caching, bundle deployment

---

## Comparison with Official Haxe Compiler

| Feature | Haxe (official) | Rayzor |
|---------|-----------------|--------|
| **Language Support** | Full Haxe 4.x | Haxe 4.x (in progress) |
| **JS/Python/PHP** | Excellent | Not a goal |
| **C++ Target** | Slow compile, fast runtime | Fast compile, fast runtime |
| **Native Perf** | ~1.0x baseline | 45-50x interpreter (LLVM) |
| **Compile Speed** | 2-5s (C++) | ~3ms (Cranelift Tier 1) |
| **JIT Runtime** | No | 5-tier (Interp->Cranelift->LLVM) |
| **Hot Path Optimization** | No | Profile-guided tier-up |
| **Memory Model** | Garbage collected | Ownership-based (compile-time) |
| **Incremental Builds** | Limited | BLADE cache (~30x speedup) |
| **Type Checking** | Production | Production |
| **Macro System** | Full (eval-based) | Interpreter, reification, @:build |
| **Optimizations** | Backend-specific | SSA-based (universal) |

---

## Contributing

Rayzor is under active development. Contributions are welcome!

### Getting Started

1. **Clone and build**:
   ```bash
   git clone https://github.com/darmie/rayzor.git
   cd rayzor
   cargo build
   ```

2. **Run tests**:
   ```bash
   cargo test
   ```

3. **Read the architecture docs**:
   - Start with [ARCHITECTURE.md](compiler/ARCHITECTURE.md)
   - Understand SSA integration in [SSA_ARCHITECTURE.md](compiler/SSA_ARCHITECTURE.md)
   - Review memory management in [MEMORY_MANAGEMENT.md](MEMORY_MANAGEMENT.md)

### Development Workflow

See [ARCHITECTURE.md](compiler/ARCHITECTURE.md#contributing) for:
- Code organization principles
- Adding new features
- Testing strategy
- Pull request guidelines

---

## Roadmap

### Complete

- Full MIR lowering pipeline (TAST -> HIR -> MIR) with SSA, phi nodes, validation
- 5 optimization passes (DCE, constant folding, copy propagation, unreachable block elimination, control flow simplification)
- Advanced optimization infrastructure (function inlining, loop analysis, SIMD vectorization)
- Cranelift JIT backend with 3 optimization levels
- LLVM backend with -O3 optimization and object file generation
- MIR interpreter with NaN-boxing optimization
- 5-tier JIT system with safe promotion barrier and background optimization
- BLADE module cache and RayzorBundle (.rzb) format
- Drop analysis with last-use tracking and escape analysis
- Monomorphization with lazy instantiation and caching
- Concurrency runtime (Thread, Channel, Mutex, Arc) with Send/Sync validation
- Pure Rust runtime (~250 extern symbols: String, Array, Math, File I/O, Vec, Collections)
- Compile-time macro system: tree-walking interpreter, reification engine ($v/$i/$e/$a/$p/$b), @:build/@:autoBuild, Context API, pipeline integration

### Near-term

- AOT binary output pipeline (LLVM object files -> system linker -> standalone executable)
- Generic stdlib types (Vec\<T\>, Option\<T\>, Result\<T,E\>)
- Optimization pass tuning and benchmarking

### Medium-term

- WebAssembly compilation target (browser + WASI)
- Full Haxe standard library coverage
- IDE support (LSP server)

### Long-term

- Production-ready for real projects
- Performance parity with Haxe/C++
- Community adoption

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details

---

## Acknowledgments

- **Haxe Foundation** - For the excellent Haxe programming language
- **Cranelift Project** - For the fast JIT compiler framework
- **LLVM Project** - For the industry-leading optimization infrastructure
- **Rust Community** - For the amazing language and ecosystem

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/darmie/rayzor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/darmie/rayzor/discussions)

---

**Rayzor**: Fast compilation, native performance, ownership-based memory. The future of Haxe compilation.
