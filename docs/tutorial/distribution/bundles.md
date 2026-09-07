# Distributing Applications with `.rzb` Bundles

Rayzor Bundles (`.rzb`) are pre-compiled, portable executables that package all
your application's modules into a single file. They contain optimized MIR
(Mid-level IR) ready for instant JIT execution -- no source code or
recompilation needed.

## When to Use Bundles

- Distributing compiled applications to end users
- Shipping pre-compiled benchmarks or demos
- Deploying server-side Haxe applications
- Reducing startup time by skipping the parse/typecheck phases

## Creating a Bundle

### Basic Usage

```bash
rayzor bundle src/Main.hx -o app.rzb
```

This compiles `Main.hx` (and all its imports) through the full pipeline, applies
O2 optimizations, compresses with zstd, and writes the result.

### Multiple Source Files

```bash
rayzor bundle src/Main.hx src/Utils.hx src/Config.hx -o app.rzb
```

All files are compiled together as a single compilation unit. Imports between
them resolve normally.

### CLI Options

```
rayzor bundle [OPTIONS] <FILES...> --output <PATH>

Arguments:
  <FILES...>              Source .hx files to compile

Options:
  -o, --output <PATH>     Output .rzb path (required)
  -O, --opt-level <0-3>   Optimization level (default: 2)
      --strip             Tree-shake unreachable code
      --no-compress       Disable zstd compression
      --cache             Enable BLADE incremental cache
      --cache-dir <DIR>   Custom cache directory
  -v, --verbose           Show compilation steps
```

### Optimization Levels

| Level | What you get | Use case |
| ----- | ------------ | -------- |
| O0 | Least optimization, quickest to compile | Debug builds |
| O1 | Light optimization | Balanced |
| O2 | Full optimization | Production (default) |
| O3 | O2, tuned harder. Slower to compile | Maximum performance |

Which passes run at each level is in
[ARCHITECTURE](../../architecture/ARCHITECTURE.md).

### Tree-Shaking

The `--strip` flag removes unreachable functions, extern declarations, and
globals from the bundle. This can significantly reduce bundle size for
applications that import large libraries but only use a fraction of them.

```bash
rayzor bundle src/Main.hx -o app.rzb --strip -v
```

Verbose output shows what was removed:

```
  shake    -142 fn, -38 ext, -12 glob, -3 mod | kept 27 fn, 5 ext
```

### Incremental Compilation

For large projects, enable the BLADE cache to avoid recompiling unchanged
modules on subsequent bundle builds:

```bash
rayzor bundle src/Main.hx -o app.rzb --cache
```

Cached `.blade` artifacts are stored in `target/debug/cache/` (or
`target/release/cache/` with `--release`). On the next build, only modified
modules are recompiled.

## Bundle Internals

A `.rzb` file contains:

| Component | Description |
| --------- | ----------- |
| MIR Modules | All compiled modules (stdlib + user code) |
| Entry Point | Pre-resolved module name and function ID for O(1) startup |
| Module Table | Fast lookup table for module access by name |
| Build Info | Compiler version, timestamp, target platform, source list |
| Flags | Compression, debug info, source map toggles |

The binary format uses `postcard` serialization with optional zstd compression
(level 3). Magic bytes: `RZBF`, format version: 1.

### Typical Sizes

Bundles are compact. Representative examples:

- Mandelbrot benchmark: ~3.7 KB (compressed)
- N-body simulation: ~7.3 KB (compressed)

The `--no-compress` flag skips zstd and produces a larger but faster-to-load
file (useful if you need minimal load latency over file size).

## Example Workflow

A typical production workflow:

```bash
# Development: run from source
rayzor run src/Main.hx

# Release: create optimized bundle
rayzor bundle src/Main.hx -o dist/app.rzb -O2 --strip -v

# Verify
ls -lh dist/app.rzb
```

Output:

```
Creating Rayzor Bundle: dist/app.rzb
  stdlib   loading
  check    src/Main.hx
  check    passed
  entry    Main::main
  shake    -142 fn, -38 ext, -12 glob, -3 mod | kept 27 fn, 5 ext
  opt      O2 (4 modules)
  bundle   4 modules in 89ms
  Bundle size: 12.34 KB

Bundle created: dist/app.rzb
  Modules: 4
```
