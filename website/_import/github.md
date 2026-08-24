repo: rayzor-blade/rayzor
branch: main

## Last sync
date: 2026-08-15T22:02:30Z

### Updated in this project
- Added a full CLI reference page built from docs/CLI.md (modes, presets, per-command flags, manifest, env vars)
- Added an Architecture page built from docs/architecture/ARCHITECTURE.md, with the repo's pipeline mermaid diagram rendered live
- Added first-class SIMD coverage — the full rayzor.SIMD* family (SIMD4f, SIMD4i32, SIMD16i8, SIMD8i32, SIMD32i8) read from compiler/src/ir/mir/types.rs, plus construction, ops and tier policy from the SIMD e2e tests
- Homepage gained a SIMD section; docs sidebar now links Architecture and SIMD
- Benchmarks card rebuilt around the real targets and methodology from rayzor.tech/benchmarks/

## Screen map
| Screen | Built from |
|---|---|
| Rayzor Home.dc.html | README.md, docs/CLI.md, website/logo.svg |
| Rayzor Docs.dc.html | docs/CLI.md, README.md |
| Rayzor CLI.dc.html | docs/CLI.md |
| Rayzor Concurrency.dc.html | local rayzor/concurrent/*.hx (Thread, Channel, Select, Mutex, Arc, Future, WorkerPool, SpinPool, Parker, CpuTopology) |
| Rayzor Architecture.dc.html | docs/architecture/ARCHITECTURE.md, docs/architecture/ROADMAP.md, docs/architecture/BACKLOG.md, compiler/examples/test_simd_e2e.rs, compiler/src/ir/mir/types.rs |

## Sync history
- 2026-08-15T19:45:00Z — initial import: homepage + docs page from README.md and docs/CLI.md, logo.svg and favicon.svg copied in
