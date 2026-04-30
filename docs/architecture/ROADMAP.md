# Rayzor Roadmap

A focused view of what's left. The full feature inventory and historical
progress live in [BACKLOG.md](BACKLOG.md); this file extracts only the
**unfinished** items, ranked by priority, with cross-references back into
the backlog.

**Generated:** 2026-04-29 — synthesised from [BACKLOG.md](BACKLOG.md)
unchecked items + 🟡/🔴/⏸️ status markers.

---

## At a Glance

| Track | Status | Owner section |
|---|---|---|
| Tensor (CPU) | 🔴 Not started | [§14.2](BACKLOG.md#142-rayzordstensor-cpu) |
| GPU Compute Phase 4–7 | 🟡 Phases 1–3 done; reductions/matmul/fusion/cross-platform pending | [§14.3](BACKLOG.md#143-rayzor-gpu-plugin) |
| Networking | ✅ Host/Socket/UdpSocket/SSL all shipped (was misclassified — no work pending) | [§6.6](BACKLOG.md#66-not-implemented---low-priority) |
| Documentation | 🟡 Core docs exist; user guides missing | [§10](BACKLOG.md#10-documentation) |
| Testing infrastructure | 🟡 Unit + e2e green; fuzzing/perf-suite missing | [§9](BACKLOG.md#9-testing-infrastructure) |
| Diagnostics — LSP / warnings | 🟡 Errors solid; IDE integration absent | [§7](BACKLOG.md#7-error-recovery--diagnostics) |
| Cross-compilation | 🟡 Flags wired; runtime libs + CI not | [§15.4](BACKLOG.md#154-cross-compilation) |
| Known runtime gaps | 🟡 Deref coercion, @:native on extern abstracts | [Known Issues](BACKLOG.md#known-issues) |
| Technical debt | 🟡 Cleanup + refactor backlog | [Technical Debt](BACKLOG.md#technical-debt) |

---

## P0 — Required for Real-World Programs

These block legitimate Haxe code from running. Do these first.

### 1. ~~Networking~~ ✅ DONE (was misclassified in backlog)

`sys.net.Host`, `sys.net.Socket`, `sys.net.UdpSocket`, `sys.net.Address`, `sys.ssl.Socket` (rustls-backed) and friends are all shipped with runtime backing and stdlib mappings. Verified by `socket_host_basic` and `host_localhost` e2e tests.

### 2. Deref Coercion for Wrapper Types ([Known Issues](BACKLOG.md#known-issues))

- [x] **Field access** on `Arc<T>` / `MutexGuard<T>` auto-inserts `.get()` (commit `19acb3a`, 2026-04-30) — `arc.value` desugars to `arc.get().value`.
- [x] **Method calls** on inner type (commit `d145433`, 2026-04-30) — `arc.double()` auto-inserts `.get()`. The fix lifted `class_type_params` + `class_constructor_symbols` from per-`AstLoweringContext` state onto `SymbolTable` so generic-class metadata propagates across files; `infer_type_args_from_constructor` now succeeds for stdlib `Arc<T>`/`MutexGuard<T>`. Mirror hook added in `lower_call_expression`.
- [ ] **Nested wrappers** (`Arc<Mutex<T>>`) — separate bug in `compute_type_substitution`. `arc.get().lock()` returns a `MutexGuard` with empty type_args because the substitution doesn't recurse into nested generic receiver types when computing `MutexGuard<T>` → `MutexGuard<State>`. So `guard.x` deref produces `Dynamic` instead of `State` and SIGSEGV's at runtime. Single-level wrappers work fine; nested still requires explicit `.get()` chain.
- [ ] Optional `@:autoDeref` metadata on user classes (currently the wrapper list is hardcoded to `rayzor.concurrent.Arc` and `rayzor.concurrent.MutexGuard` by qualified name).

Affects ergonomics of every concurrency program — single-level cases now ergonomic.

### 3. ~~`@:native` Metadata Ignored on Extern Abstract Methods~~ ✅ DONE (2026-04-30)

`BladeMethodInfo.native_name` now persists across cache, `register_method_from_blade` restores it on load, and codegen paths that consult `symbol.native_name` work for cached methods. See commit `e8d2b1d`.

---

## P1 — High-Impact Quality of Life

### 4. Tensor CPU Stdlib ([§14.2](BACKLOG.md#142-rayzordstensor-cpu))

ML / numerical workloads need a Tensor type even before GPU.

- [ ] Tensor type with shape / strides / dtype (extern class, Rust runtime)
- [ ] DType enum (F32, F16, BF16, I32, I8, U8)
- [ ] Construction: `zeros`, `ones`, `full`, `fromArray`, `rand`
- [ ] View ops: `reshape`, `transpose`, `permute`, `slice` (no-copy via strides)
- [ ] Elementwise: `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`
- [ ] Reductions: `sum`, `mean`, `max`, `min`
- [ ] Linear algebra: `matmul`, `dot`
- [ ] Activations: `relu`, `gelu`, `silu`, `softmax`
- [ ] Normalization: `layerNorm`, `rmsNorm`
- [ ] SIMD4f vectorised CPU paths for f32 ops

### 5. GPU Compute — Phases 4 to 7 ([§14.3](BACKLOG.md#143-rayzor-gpu-plugin))

Phases 1–3 (Metal device + buffers + MSL elementwise kernels, 15 tests) are ✅ shipped. Remaining:

#### Phase 4 — Reductions + Matmul
- [ ] Tree-reduction kernels (sum, mean, max, min) with threadgroup shared memory
- [ ] Tiled 16×16 shared-memory matmul
- [ ] Dot product

#### Phase 5 — Compute Data Structures (`@:gpuStruct`)
- [ ] `@:gpuStruct` annotation (GPU-aligned flat structs, 4-byte floats)
- [ ] Structured buffer create / alloc / read
- [ ] MSL/CUDA typedef generation via `gpuDef()`

#### Phase 6 — Kernel Fusion
- [ ] Lazy evaluation DAG for elementwise op chains
- [ ] Fused kernel codegen (`a.add(b).mul(c).relu()` → single kernel)

#### Phase 7 — Additional Backends
- [ ] CUDA backend (NVRTC) — NVIDIA GPUs
- [ ] WebGPU backend (wgpu) — cross-platform
- [ ] Vulkan backend (SPIR-V) — Windows / Linux / Android
- [ ] OpenCL backend — cross-platform legacy

### 6. Operator Overloading for GPU / Tensor ([§14.5](BACKLOG.md#145-operator-overloading-for-gputensor-types))

- [ ] Exercise existing `@:op` annotations on Tensor (E2E tests using `a + b` syntax)
- [ ] Add `@:op` overloading to GpuBuffer (requires ctx back-pointer in buffer struct)
- [ ] Verify abstract-type `@:op` support works end-to-end (currently only extern class is tested)

### 7. Generic Metadata Pipeline Integration ([Phase 1](BACKLOG.md#phase-1-foundation-mostly-complete))

- [ ] Last 🔴 item under "Phase 1: Foundation" — generic metadata still needs end-to-end pipeline integration. Specifics in §1 of the backlog.

---

## P2 — Standard Library / Runtime Polish

### 8. `sys.thread.Tls<T>` ([§6.6](BACKLOG.md#66-not-implemented---low-priority))

- [x] Extern class shipped in `compiler/haxe-std/sys/thread/Tls.hx`
- [ ] Runtime backing (`sys_tls_*` functions)
- [ ] Stdlib mapping
- [ ] Basic test

The other `sys.thread.*` primitives (Lock, Mutex, Semaphore, Condition, Deque) are already implemented — see [§3.2](BACKLOG.md#32-channel-system-message-passing) / [§3.3](BACKLOG.md#33-synchronization-primitives).

### 9. Channel `select` Macro ([§3.2](BACKLOG.md#32-channel-system-message-passing))

- [ ] Multi-channel `Select` class / macro for non-deterministic receive across multiple channels (Go-style `select { case <-ch1: ...; case <-ch2: ... }`).

### 10. Inline C / TinyCC Polish ([§13.7](BACKLOG.md#137-remaining--future-enhancements))

- [ ] Source caching: hash C source to avoid recompiling identical `__c__()` blocks
- [ ] `@:unsafe` metadata warning when using `__c__` (currently allowed without annotation)
- [ ] `CC.addClib()` explicit API method (currently `@:clib` metadata only)
- [ ] Windows: test MSYS2/MinGW pkg-config integration end-to-end

### 11. Interpreter SIMD Correctness ([§14.4](BACKLOG.md#144-interpreter-simd-correctness))

- [ ] Integrate the `wide` crate for real SIMD in the interpreter (currently returns void), **or**
- [ ] Force-promote SIMD functions to skip Tier 0
- [ ] Close TCC Linker SIMD gap on Linux (final tier lacks SIMD)

### 12. AOT — Static Linking + Cross-Compilation Gaps

- [ ] Fully static linking with musl ([§15.3](BACKLOG.md#153-static-linking))
- [ ] Runtime library for target arch — build-on-demand or user-provided ([§15.4](BACKLOG.md#154-cross-compilation))
- [ ] CI testing for cross-compilation (x86_64 → aarch64, etc.) ([§15.4](BACKLOG.md#154-cross-compilation))

### 13. Full RTTI for Type / Reflect Classes ([Remaining Work](BACKLOG.md#remaining-work))

Type.getClass() and the `__type_id` header are in place ([memory entry](../../docs/RAYZOR_ARCHITECTURE.md)). Full Type / Reflect runtime introspection (field iteration, method invocation, etc.) is the remaining ask.

### 14. Compile-Time Type Generation — `MacroType<[expr]>` ([§6.6](BACKLOG.md#66-not-implemented---low-priority))

- [ ] Compiler/parser support for the `MacroType<[expr]>` substitution syntax. The extern class is shipped; the macro system itself is in place. This is implementable, just unbuilt.

### 15. Interface Compatibility Runtime Checks ([§16.2](BACKLOG.md#162-interface-dispatch-vtables))

- [ ] Add interface compatibility checks at runtime (`obj is SomeInterface` style validation).

---

## P3 — Diagnostics, Tooling, Tests, Docs

### 16. Error Recovery & Diagnostics ([§7](BACKLOG.md#7-error-recovery--diagnostics))

- [ ] IDE integration (LSP) — full diagnostics, hover types, go-to-def, completion
- [ ] Warning levels and configuration

### 17. Testing Infrastructure ([§9.2](BACKLOG.md#92-in-progress--needed))

- [ ] Comprehensive generics test suite
- [ ] Async/await integration tests
- [ ] Memory safety violation tests (edge cases)
- [ ] Performance benchmarks (formal suite)
- [ ] Fuzzing infrastructure
- [ ] Property-access mode tests ([§11 Phase 5](BACKLOG.md#11-haxe-property-access-support)): default / get / set / custom / null / never / inheritance / error messages

### 18. JIT Documentation Polish ([§12](BACKLOG.md#12-jit-execution-cranelift-backend))

- [ ] Document runtime API for concurrency primitives
- [ ] Add execution examples to README
- [ ] Performance benchmarks (JIT vs interpretation)

### 19. Documentation ([§10](BACKLOG.md#10-documentation))

- [ ] Complete API documentation
- [ ] Generics user guide
- [ ] Async/await tutorial
- [ ] Concurrency guide
- [ ] Memory safety best practices
- [ ] Performance tuning guide
- [ ] Migration guide (from Haxe)
- [ ] Contributing guide

---

## P4 — Technical Debt ([Technical Debt](BACKLOG.md#technical-debt))

Internal hygiene; doesn't block users but slows down development.

- [ ] Remove DEBUG log statements cleanly (without breaking code)
- [ ] Consolidate error handling (CompilationError vs custom errors)
- [ ] Reduce warnings in codebase
- [ ] Improve type inference completeness
- [ ] Refactor HIR / MIR distinction (clarify naming)
- [ ] Performance profiling and bottleneck identification

---

## Phase 6 Polish (umbrella) ([§Phase 6](BACKLOG.md#phase-6-polish))

- [ ] Performance optimization (catch-all)
- [ ] Comprehensive testing (covered by §17 above)
- [ ] Complete documentation (covered by §19 above)

---

## Suggested Sequencing

The dependencies are mostly independent, so the rough sequencing is by **user impact** rather than technical prerequisite:

1. **P0 first** — Networking unblocks real-world programs; deref coercion + `@:native` polish make existing programs ergonomic.
2. **Tensor CPU** before **GPU Phase 4–7** — most numerical code wants Tensor with sane CPU paths first; GPU is acceleration on top.
3. **GPU additional backends** (CUDA / WebGPU / Vulkan) only after Phase 4–6 lands on Metal — proves the Kernel-IR → text-per-backend strategy before fanning out.
4. **AOT static-linking + cross-compile CI** independently — the prerequisites are wired, just needs target-runtime work + CI infrastructure.
5. **LSP + tests + docs** in parallel — these don't block features and benefit from being kept in lockstep with the codebase rather than batched.

---

## Tracking

When an item lands, update the corresponding `[ ]` in [BACKLOG.md](BACKLOG.md) to `[x]` and remove it here. The backlog remains the source of truth; this roadmap is a periodic snapshot.
