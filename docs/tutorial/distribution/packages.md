# Distributing Libraries with `.rpkg` Packages

Rayzor Packages (`.rpkg`) are the standard format for distributing reusable
libraries. A package can contain pure Haxe source libraries, native code with
FFI bindings, or both. Consumers load packages with `rayzor run --rpkg` and
import them like any other module.

## When to Use Packages

- Publishing a reusable Haxe library for other Rayzor projects
- Distributing native extensions (GPU compute, database drivers, crypto, etc.)
- Shipping a high-level Haxe API backed by a native implementation
- Bundling extern declarations alongside the library classes that wrap them

## Package Types

### Pure Haxe Package

A library written entirely in Haxe. No native code, no FFI -- just `.hx` source
files bundled into a single distributable file.

```
my-math-lib.rpkg
  math/Vec2.hx
  math/Vec3.hx
  math/Matrix4.hx
  math/Quaternion.hx
```

### Native Package

A package that includes a platform-specific dynamic library and FFI bindings.
Contains extern class declarations (`.hx` stubs mapping to native functions), a
method table (serialized function signatures), and the compiled dylib.

```
rayzor-gpu.rpkg
  rayzor/gpu/GPUCompute.hx      (extern class)
  rayzor/gpu/GpuBuffer.hx       (extern class)
  macos-aarch64/librayzor_gpu.dylib
  macos-x86_64/librayzor_gpu.dylib
  linux-x86_64/librayzor_gpu.so
  windows-x86_64/rayzor_gpu.dll
  method_table                   (serialized FFI descriptors)
```

### Mixed Package

The most common pattern for native packages: extern classes for the low-level
FFI, plus pure Haxe classes that provide a high-level API on top.

```
rayzor-gpu.rpkg
  rayzor/gpu/GPUCompute.hx      (extern class — low-level FFI)
  rayzor/gpu/GpuBuffer.hx       (extern class — low-level FFI)
  rayzor/gpu/Tensor.hx          (pure Haxe — high-level API)
  rayzor/gpu/nn/Linear.hx       (pure Haxe — neural net layer)
  macos-aarch64/librayzor_gpu.dylib
  linux-x86_64/librayzor_gpu.so
  method_table                   (serialized FFI descriptors)
```

The consumer only needs to import the high-level classes:

```haxe
import rayzor.gpu.Tensor;
import rayzor.gpu.nn.Linear;
```

The extern classes and native library are resolved automatically.

## Creating a Package

### Pure Haxe Package

```bash
rayzor rpkg pack --haxe-dir src/math/ -o math-lib.rpkg
```

This recursively collects all `.hx` files under `src/math/` and bundles them.
The directory structure is preserved, so `src/math/Vec2.hx` becomes the module
path `Vec2.hx` inside the package.

### Native Package

```bash
rayzor rpkg pack \
  --dylib target/release/librayzor_gpu.dylib \
  --haxe-dir src/rayzor/gpu/ \
  -o gpu.rpkg
```

This:

1. Reads the dylib and embeds it for the current platform (e.g. macos-aarch64)
2. Calls the dylib's `rayzor_rpkg_entry()` export to extract the method table
3. Collects all `.hx` files under the haxe directory
4. Writes everything into a single `.rpkg` archive

### Custom Package Name

By default the package name is derived from the output filename. Override it
with `--name`:

```bash
rayzor rpkg pack --haxe-dir src/ -o my-lib.rpkg --name "my-awesome-lib"
```

### CLI Reference

```
rayzor rpkg pack [OPTIONS] --haxe-dir <DIR> --output <PATH>

Options:
      --dylib <FILE>       Native library to embed (repeatable for multi-platform)
      --os <OS>            OS for preceding --dylib (macos, linux, windows)
      --arch <ARCH>        Architecture for preceding --dylib (aarch64, x86_64)
      --haxe-dir <DIR>     Directory of .hx files to bundle (required)
  -o, --output <PATH>      Output .rpkg path (required)
      --name <NAME>        Package name (defaults to output filename)

rayzor rpkg strip <INPUT> [OPTIONS] --output <PATH>

Options:
      --os <OS>            Target OS (defaults to current)
      --arch <ARCH>        Target architecture (defaults to current)
  -o, --output <PATH>      Output stripped .rpkg path (required)
```

## Inspecting a Package

Use `rayzor rpkg inspect` to view the contents of an `.rpkg` file:

```bash
rayzor rpkg inspect gpu.rpkg
```

Output:

```
RPKG: gpu.rpkg
  package: gpu

  Method Table (plugin: gpu)
    static rayzor_gpu_GPUCompute.create  ->  rayzor_gpu_compute_create (params: 0, ret: 3)
    static rayzor_gpu_GPUCompute.isAvailable  ->  rayzor_gpu_compute_is_available (params: 0, ret: 1)
    instance rayzor_gpu_GPUCompute.destroy  ->  rayzor_gpu_compute_destroy (params: 2, ret: 0)
    instance rayzor_gpu_GPUCompute.createBuffer  ->  rayzor_gpu_compute_create_buffer (params: 3, ret: 3)
    ...

  Haxe Sources (4):
    GPUCompute.hx
    GpuBuffer.hx
    Tensor.hx
    nn/Linear.hx

  Native Library: present for current platform (macos-aarch64)
```

## Using a Package

### Loading at Runtime

Pass `--rpkg` to `rayzor run`:

```bash
rayzor run --rpkg gpu.rpkg src/Main.hx
```

Multiple packages can be loaded:

```bash
rayzor run --rpkg gpu.rpkg --rpkg math-lib.rpkg src/Main.hx
```

### What Happens on Load

When Rayzor loads an `.rpkg`, it:

1. Parses the archive and reads the table of contents
2. Extracts bundled `.hx` files to a temp directory
3. Adds that directory to the compiler's source paths (so `import` resolves)
4. If a native library is present:
   - Extracts the dylib matching the current OS/architecture to a temp file
   - Loads it via `dlopen`
   - Reads runtime symbols for JIT linking
5. If a method table is present:
   - Deserializes the FFI descriptors
   - Registers them as a compiler plugin (extern declarations + method mappings)
6. Compilation proceeds normally -- bundled `.hx` files are compiled on demand
   when imported by user code

### Importing from a Package

Once loaded, package modules are imported by their path relative to the package
root. If the package was built from `src/rayzor/gpu/` containing
`GPUCompute.hx` and `Tensor.hx`:

```haxe
import rayzor.gpu.GPUCompute;
import rayzor.gpu.Tensor;
```

The directory structure inside the `.rpkg` directly maps to Haxe package paths.

## Structuring a Package for Distribution

### Directory Layout

A typical native package project:

```
rayzor-gpu/
  Cargo.toml                     # Rust crate for the native library
  src/
    lib.rs                       # Native implementation
  haxe/
    rayzor/gpu/
      GPUCompute.hx              # @:native extern class
      GpuBuffer.hx               # @:native extern class
      Tensor.hx                  # Pure Haxe high-level API
      nn/
        Linear.hx                # Pure Haxe
        Activation.hx            # Pure Haxe
```

Build and pack:

```bash
# Build the native library
cargo build --release

# Pack into .rpkg
rayzor rpkg pack \
  --dylib target/release/librayzor_gpu.dylib \
  --haxe-dir haxe/rayzor/gpu/ \
  -o rayzor-gpu.rpkg \
  --name rayzor-gpu
```

### Writing Extern Classes

Extern classes declare the FFI boundary. They map Haxe method signatures to
native C functions via `@:native` metadata:

```haxe
// rayzor/gpu/GPUCompute.hx
@:native("rayzor_gpu_GPUCompute")
extern class GPUCompute {
    @:native("rayzor_gpu_compute_create")
    static function create():GPUCompute;

    @:native("rayzor_gpu_compute_destroy")
    function destroy():Void;

    @:native("rayzor_gpu_compute_add")
    function add(a:GpuBuffer, b:GpuBuffer):GpuBuffer;
}
```

### Writing Library Classes

Library classes are regular Haxe code that uses the extern classes internally:

```haxe
// rayzor/gpu/Tensor.hx
class Tensor {
    var buffer:GpuBuffer;
    var ctx:GPUCompute;

    public function new(ctx:GPUCompute, data:Array<Float>) {
        this.ctx = ctx;
        this.buffer = ctx.createBuffer(data, data.length, 2);
    }

    public function add(other:Tensor):Tensor {
        var result = new Tensor(ctx, []);
        result.buffer = ctx.add(this.buffer, other.buffer);
        return result;
    }
}
```

### Multi-Platform Packages

A single `.rpkg` file embeds native libraries for **all target platforms**.
At load time, Rayzor picks the dylib matching the current OS and architecture.

We recommend [cross](https://github.com/cross-rs/cross) for cross-compiling
native rpkg libraries. `cross` uses pre-built Docker images with the correct
toolchains — no manual sysroot setup needed:

```bash
cargo install cross --git https://github.com/cross-rs/cross
```

Build for each platform, then pack into one rpkg:

```bash
# macOS targets (native — cross doesn't support macOS as a target)
cargo build -p rayzor-gpu --features webgpu-backend --release --target aarch64-apple-darwin
cargo build -p rayzor-gpu --features webgpu-backend --release --target x86_64-apple-darwin

# Linux and Windows targets via cross
cross build -p rayzor-gpu --features webgpu-backend --release --target x86_64-unknown-linux-gnu
cross build -p rayzor-gpu --features webgpu-backend --release --target x86_64-pc-windows-gnu

# Pack all platforms into one rpkg
rayzor rpkg pack \
  --dylib target/aarch64-apple-darwin/release/librayzor_gpu.dylib --os macos --arch aarch64 \
  --dylib target/x86_64-apple-darwin/release/librayzor_gpu.dylib --os macos --arch x86_64 \
  --dylib target/x86_64-unknown-linux-gnu/release/librayzor_gpu.so --os linux --arch x86_64 \
  --dylib target/x86_64-pc-windows-gnu/release/rayzor_gpu.dll --os windows --arch x86_64 \
  --haxe-dir haxe/ \
  -o rayzor-gpu.rpkg
```

The built-in pack scripts support this out of the box:

```bash
cd gpu && ./pack-rpkg.sh --cross    # builds all platforms + packs
cd gpu && ./pack-rpkg.sh            # current platform only (fast)
```

### Stripping for Distribution

A universal rpkg containing all platform dylibs can be large. Use `--strip` to
produce a platform-specific rpkg by removing unused native libraries:

```bash
# Strip to current platform only
rayzor rpkg strip rayzor-gpu.rpkg -o rayzor-gpu-slim.rpkg

# Strip to a specific platform
rayzor rpkg strip rayzor-gpu.rpkg --os linux --arch x86_64 -o rayzor-gpu-linux.rpkg
```

This is useful for deployment — ship the universal rpkg to CI, then strip per
target before bundling into your application.

## Package Format Reference

### Binary Layout

```
[entry data][entry data]...[TOC (postcard)][toc_size: u32][version: u32][magic: "RPKG"]
```

The footer (last 12 bytes) is read first. The TOC is a postcard-serialized table
of contents listing all entries with their byte offsets and metadata.

### Entry Types

| Type | Contents | Metadata |
| ---- | -------- | -------- |
| NativeLib | Platform dylib bytes | os, arch (e.g. "macos", "aarch64") |
| HaxeSource | UTF-8 `.hx` source text | module path (e.g. "Tensor.hx") |
| MethodTable | Serialized FFI descriptors | plugin name |

### `.rzb` vs `.rpkg`

| | `.rzb` Bundle | `.rpkg` Package |
| --- | --- | --- |
| Purpose | Distribute compiled applications | Distribute reusable libraries |
| Contains | Compiled MIR modules | Haxe sources + optional native lib |
| Execution | Direct JIT execution | Compiled on import by consumer |
| Entry point | Pre-resolved main function | None (library, not application) |
| Compression | zstd (optional) | None (sources are small) |
| Dependencies | Self-contained | Consumer compiles against their stdlib |
