# Rayzor Compiler

> A modern, safe, and performant Haxe compiler implementation in Rust

## Quick Links

- **[Architecture](../docs/architecture/ARCHITECTURE.md)** - Detailed compiler architecture
- **[ARCHITECTURE.md](../docs/architecture/ARCHITECTURE.md)** - General architecture overview
- **[Architecture](../docs/architecture/ARCHITECTURE.md)** - SSA integration strategy (advanced)
- **[IMPLEMENTATION_ROADMAP.md](../docs/architecture/IMPLEMENTATION_ROADMAP.md)** - Development roadmap
- **PRODUCTION_READINESS.md** - Production checklist
- **[../MEMORY_MANAGEMENT.md](../docs/architecture/MEMORY_MANAGEMENT.md)** - Memory management strategy

## What is Rayzor?

Rayzor is a complete reimplementation of a Haxe compiler in Rust, designed for:

- **High Performance**: Native compilation via Cranelift JIT and LLVM AOT backends
- **Memory Safety**: Ownership-based memory management with compile-time safety analysis
- **Developer Experience**: 5-tier JIT with fast startup, hot-reload support, excellent error messages
- **Production Ready**: LLVM -O3 optimization with AOT object file generation
- **Incremental Builds**: BLADE cache (.blade) and RayzorBundle (.rzb) for fast rebuilds

## Features

### Implemented

- **Parser**: Incremental nom-based parser with error recovery
- **Type System**: Sophisticated type inference and checking
- **Semantic Analysis**: CFG, DFG (SSA), Call Graph, Ownership tracking
- **Flow-Sensitive Checking**: TypeFlowGuard with precise safety analysis
- **Multi-tier IR**: HIR (high-level) and MIR (SSA-form, optimizable)
- **Optimization Framework**: DCE, constant folding, copy propagation, function inlining, loop analysis, SIMD vectorization infrastructure
- **MIR Interpreter**: NaN-boxing interpreter for development and Tier 0 execution
- **Cranelift Backend**: Full JIT compilation with 3 optimization levels (speed/default/best)
- **LLVM Backend**: Full compilation with -O3 optimization and AOT object file generation
- **Tiered JIT**: 5-tier system (Interpreted -> Baseline -> Standard -> Optimized -> Maximum)
- **BLADE Cache**: Per-module binary cache with source hash validation (~30x incremental speedup)
- **RayzorBundle (.rzb)**: Single-file distributable format with O(1) startup
- **Memory Safety Analysis**: Ownership, lifetime, borrow checking, drop analysis, escape analysis
- **Concurrency**: Thread, Channel, Arc, Mutex with Send/Sync validation
- **Standard Library**: String, Array, Math, File I/O, Map, IntMap, Vec (monomorphized), Bytes

### In Progress

- **AOT Binary Linking**: LLVM generates .o files; full linking pipeline (object files -> system linker -> standalone executable) pending
- **WASM Backend**: WebAssembly target
- **Standard Library**: Expanding Haxe API compatibility
- **Tooling**: LSP server, debugger integration

## Architecture Overview

```
Source Code (.hx)
    |
Parser (nom-based, incremental)
    |
AST (Abstract Syntax Tree)
    |
Type Checker + Type Inference
    |
TAST (Typed AST with memory annotations)
    |
Semantic Analysis (CFG, DFG/SSA, Ownership Graph)
    |
TypeFlowGuard (Flow-sensitive checking)
    |
HIR (High-level IR with language semantics)
    |
MIR (Mid-level IR, SSA form with phi nodes)
    |
Optimization Passes (DCE, folding, inlining, etc.)
    |
+-- BLADE Cache (.blade) -- incremental builds
+-- RayzorBundle (.rzb)  -- distributable format
    |
Code Generation
+-- MIR Interpreter  (Tier 0: instant startup, ~10x slower)
+-- Cranelift JIT    (Tier 1-3: fast compile, near-native speed)
+-- LLVM             (Tier 4: max optimization, AOT object files)
    |
Target Output (JIT execution, native .o files)
```

See [Architecture](../docs/architecture/ARCHITECTURE.md) for detailed pipeline diagrams.

## Key Innovations

### 1. SSA as Analysis Infrastructure

SSA (Static Single Assignment) is built **once** in the Data Flow Graph and **queried** by all subsequent passes. This eliminates duplication while enabling precise analysis.

```
DFG (SSA form) -> TypeFlowGuard -> HIR hints -> MIR attributes -> Optimizations
     ^
Single Source of Truth
```

See [Architecture](../docs/architecture/ARCHITECTURE.md) for the complete strategy.

### 2. Layered IR Design

- **HIR**: Preserves language semantics for hot-reload and debugging
- **MIR**: Platform-independent SSA optimization target with phi nodes
- Both use SSA insights without requiring re-computation

### 3. Ownership-Based Memory Management

Compile-time memory safety without garbage collection:

```haxe
@:safety
@:move
class Resource {
    var data: Array<Int>;

    public function borrow(): &Array<Int> {
        return &data;  // Compile-time borrow checking
    }
}
```

The compiler performs ownership analysis, lifetime analysis (constraint-based solver), borrow checking, drop analysis (automatic Free insertion), and escape analysis (stack allocation optimization). GC is only used for `Dynamic` types or objects with unknown sizes at compile time.

See [../MEMORY_MANAGEMENT.md](../docs/architecture/MEMORY_MANAGEMENT.md) for the full strategy.

### 4. 5-Tier JIT Compilation

Functions start interpreted and promote to faster tiers based on execution frequency:

| Tier | Backend | Compile Speed | Run Speed | Use Case |
|------|---------|---------------|-----------|----------|
| 0 | MIR Interpreter | Instant | ~10x slower | Startup, rarely-called |
| 1 | Cranelift (speed) | ~1ms | ~1.5x slower | Warming up |
| 2 | Cranelift (default) | ~5ms | ~1.2x slower | Most code |
| 3 | Cranelift (best) | ~20ms | Near-native | Hot loops |
| 4 | LLVM (-O3) | ~100ms | Native+ | Hottest code |

### 5. Incremental Compilation (BLADE)

`.blade` files cache per-module MIR with source hash validation and dependency tracking. On incremental rebuilds, only changed modules are recompiled (~30x faster). The `.rzb` RayzorBundle format packages all modules into a single distributable file with O(1) startup via a module table and optional symbol manifest.

## Getting Started

### Build

```bash
# Build the compiler
cargo build --release

# Run tests
cargo test

# Build with all features
cargo build --release --all-features
```

### Example Usage

```bash
# Compile and run a Haxe file
./target/release/rayzor compile example.hx

# With optimization tier
./target/release/rayzor compile --tier optimized example.hx

# Development mode with hot-reload
./target/release/rayzor dev --watch --hot-reload example.hx

# Generate AOT object file via LLVM
./target/release/rayzor compile --backend llvm --aot example.hx
```

## Project Structure

```
rayzor/
+-- parser/              # Parsing infrastructure
|   +-- src/
|   |   +-- haxe_parser.rs
|   |   +-- haxe_ast.rs
|   |   +-- incremental_parser_enhanced.rs
|   +-- Cargo.toml
|
+-- compiler/            # Main compiler crate
|   +-- src/
|   |   +-- tast/               # Type-checked AST
|   |   +-- semantic_graph/     # Analysis (CFG, DFG/SSA, etc.)
|   |   +-- ir/                 # HIR and MIR
|   |   +-- codegen/            # Code generation backends
|   |   |   +-- cranelift_backend.rs
|   |   |   +-- llvm_backend.rs
|   |   |   +-- llvm_jit_backend.rs
|   |   |   +-- mir_interpreter.rs
|   |   +-- stdlib/             # Runtime function mappings
|   |   +-- pipeline.rs         # Compilation pipeline
|   |   +-- tiered_jit.rs       # 5-tier JIT manager
|   |   +-- blade_cache.rs      # BLADE incremental cache
|   |   +-- rayzor_bundle.rs    # .rzb bundle format
|   |
|   +-- examples/               # Test programs
|   +-- Cargo.toml
|
+-- runtime/             # Native runtime library (rayzor-runtime)
+-- diagnostics/         # Error reporting
+-- source_map/          # Source location tracking
+-- cranelift-fork/      # Customized Cranelift with W^X and mmap fixes
+-- Cargo.toml
```

## Documentation Guide

### For New Contributors

Start here:
1. **[Architecture](../docs/architecture/ARCHITECTURE.md)** - Understand the overall design
2. **[IMPLEMENTATION_ROADMAP.md](../docs/architecture/IMPLEMENTATION_ROADMAP.md)** - See what's being built
3. Look at `examples/` for working code

### For Compiler Developers

Deep dives:
1. **[Architecture](../docs/architecture/ARCHITECTURE.md)** - SSA integration pattern
2. **[src/ir/README.md](src/ir/README.md)** - IR design details
3. **[../MEMORY_MANAGEMENT.md](../docs/architecture/MEMORY_MANAGEMENT.md)** - Memory management strategy
4. **[Architecture](../docs/architecture/ARCHITECTURE.md)** - Runtime library and extern functions

### For Users

- **[../resource/strategy.md](../resource/strategy.md)** - Development workflow
- **../resource/plan.md** - Project goals

## Development

### Code Organization

Each crate follows this structure:
```
src/
+-- lib.rs              # Public API
+-- component1/         # Major component
|   +-- mod.rs
|   +-- submodule.rs
|   +-- tests.rs        # Co-located tests
+-- component2/
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with logging
RUST_LOG=debug cargo test

# Run examples
cargo run --example test_hir_pipeline
```

### Adding Features

1. Parse new syntax (parser crate)
2. Type check (compiler/src/tast/)
3. Analyze (compiler/src/semantic_graph/)
4. Lower to HIR (compiler/src/ir/tast_to_hir.rs)
5. Lower to MIR (compiler/src/ir/hir_to_mir.rs)
6. Generate code (compiler/src/codegen/)

See [Architecture](../docs/architecture/ARCHITECTURE.md) for details.

## Current Status

### Completeness

| Component | Status | Coverage |
|-----------|--------|----------|
| Parser | Complete | ~95% |
| Type Checker | Complete | ~80% |
| Semantic Analysis | Complete | ~85% |
| HIR | Complete | ~90% |
| MIR | Complete | ~95% |
| Optimization Passes | Implemented | ~70% |
| Cranelift Backend | Complete | ~90% |
| LLVM Backend | Complete | ~85% |
| MIR Interpreter | Complete | ~90% |
| Tiered JIT | Complete | ~90% |
| BLADE Cache | Complete | ~95% |
| RayzorBundle (.rzb) | Complete | ~90% |
| AOT Binary Linking | Not Started | 0% |

### Known Limitations

- No macro system yet
- Limited standard library coverage
- WASM backend not started
- AOT full binary linking pipeline pending (LLVM generates .o files but no linker integration yet)
- No package manager integration

## Performance

### Compilation Speed

- **Parsing**: ~50us per KB
- **Type Checking**: ~200us per function
- **Analysis**: ~500us per function
- **Optimization**: ~1ms per function
- **Cranelift JIT**: ~1-20ms per function (depends on tier)
- **LLVM**: ~100ms per function (-O3)

### Memory Usage

- **AST**: ~500 bytes per node
- **TAST**: ~800 bytes per node
- **Semantic Graphs**: ~2KB per function
- **MIR**: ~3KB per function

## Contributing

We welcome contributions! Please:

1. Read [Architecture](../docs/architecture/ARCHITECTURE.md) to understand the design
2. Check [IMPLEMENTATION_ROADMAP.md](../docs/architecture/IMPLEMENTATION_ROADMAP.md) for planned work
3. Look at existing code for style guidelines
4. Add tests for new features
5. Update documentation

### Coding Standards

- **Rust 2021 Edition**
- **Format**: `cargo fmt` before committing
- **Lint**: `cargo clippy` should pass
- **Tests**: All tests must pass
- **Documentation**: Public APIs must be documented

## License

MIT License - see LICENSE file for details
