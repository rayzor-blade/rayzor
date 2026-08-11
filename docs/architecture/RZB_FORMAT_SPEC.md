# RayzorBundle (.rzb) Format Specification
## Single-File Executable Bundle for Instant Startup

### Purpose

The `.rzb` (RayzorBundle) format packages an entire compiled application into a single binary file, enabling:

1. **Instant Startup** - Skip compilation entirely, load pre-compiled MIR directly
2. **Single-File Distribution** - Ship one `.rzb` file instead of source code
3. **13x Faster Loading** - ~530µs load time vs ~6.9ms full compilation
4. **Portable Executables** - Platform-independent bytecode format

### Comparison with BLADE

| Feature | BLADE (.blade) | RayzorBundle (.rzb) |
|---------|---------------|---------------------|
| **Purpose** | Cache individual modules | Package entire application |
| **Contents** | Single MIR module | All modules + entry point |
| **Use Case** | Incremental compilation | Distribution / deployment |
| **Typical Size** | 1-50 KB per module | 10-500 KB total |
| **Load Time** | ~50µs per module | ~500µs total |

### Technical Design

**Serialization**: `postcard` crate for compact binary format
**File Extension**: `.rzb`
**Magic Number**: `"RZBF"` (4 bytes)
**Version**: u32 version number for format evolution

### File Structure

```
┌──────────────────────────┐
│ Magic (4 bytes)          │  "RZBF"
├──────────────────────────┤
│ Version (4 bytes)        │  Format version (currently 1)
├──────────────────────────┤
│ Flags (4 bytes)          │  Bundle flags (compression, debug, etc.)
├──────────────────────────┤
│ Entry Module Name        │  String: module containing main()
├──────────────────────────┤
│ Entry Function Name      │  String: entry point function name
├──────────────────────────┤
│ Module Table             │  Index of all modules
│   - Module count (u32)   │
│   - For each module:     │
│     - Name (String)      │
│     - Offset (u64)       │
│     - Size (u64)         │
├──────────────────────────┤
│ Modules                  │  Serialized IrModule array
│   - Module 1             │
│   - Module 2             │
│   - ...                  │
├──────────────────────────┤
│ Symbol Manifest          │  Optional: extern function signatures
│   (if present)           │
├──────────────────────────┤
│ Build Info               │  Compiler version, timestamp, target
└──────────────────────────┘
```

### Data Structures

```rust
/// Magic number identifying RayzorBundle files
pub const BUNDLE_MAGIC: &[u8; 4] = b"RZBF";

/// Current bundle format version
pub const BUNDLE_VERSION: u32 = 1;

/// RayzorBundle - Single-file executable bundle
#[derive(Debug, Serialize, Deserialize)]
pub struct RayzorBundle {
    /// Magic number ("RZBF")
    pub magic: [u8; 4],

    /// Format version
    pub version: u32,

    /// Bundle flags
    pub flags: BundleFlags,

    /// Name of the entry module (e.g., "main")
    pub entry_module: String,

    /// Name of the entry function (e.g., "main")
    pub entry_function: String,

    /// Module table for quick lookup
    pub module_table: Vec<ModuleTableEntry>,

    /// All compiled modules
    pub modules: Vec<IrModule>,

    /// Optional symbol manifest for FFI
    pub symbols: Option<BladeSymbolManifest>,

    /// Build information
    pub build_info: BundleBuildInfo,
}

/// Bundle configuration flags
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BundleFlags {
    /// Bundle contains debug information
    pub debug_info: bool,

    /// Modules are compressed (future)
    pub compressed: bool,

    /// Bundle is signed (future)
    pub signed: bool,
}

/// Module table entry for quick lookup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleTableEntry {
    /// Module name
    pub name: String,

    /// Number of functions in module
    pub function_count: u32,

    /// Number of extern functions
    pub extern_count: u32,
}

/// Build information embedded in bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleBuildInfo {
    /// Compiler version used to create bundle
    pub compiler_version: String,

    /// Unix timestamp of bundle creation
    pub build_timestamp: u64,

    /// Target architecture (e.g., "aarch64", "x86_64")
    pub target_arch: String,

    /// Whether bundle includes stdlib
    pub includes_stdlib: bool,
}
```

### API

```rust
/// Save a bundle to disk
pub fn save_bundle(path: impl AsRef<Path>, bundle: &RayzorBundle) -> Result<(), BladeError>;

/// Load a bundle from disk
pub fn load_bundle(path: impl AsRef<Path>) -> Result<RayzorBundle, BladeError>;

/// Load a bundle from bytes (for embedded bundles)
pub fn load_bundle_from_bytes(bytes: &[u8]) -> Result<RayzorBundle, BladeError>;

impl RayzorBundle {
    /// Create a new bundle from modules
    pub fn new(
        modules: Vec<IrModule>,
        entry_module: &str,
        entry_function: &str,
    ) -> Self;

    /// Get the entry module
    pub fn entry_module(&self) -> Option<&IrModule>;

    /// Get the entry function name
    pub fn entry_function(&self) -> &str;

    /// Get module count
    pub fn module_count(&self) -> usize;

    /// Find a module by name
    pub fn get_module(&self, name: &str) -> Option<&IrModule>;
}
```

### Usage Examples

#### Creating a Bundle

```bash
# Compile source to bundle
rayzor bundle Main.hx -o app.rzb

# Output:
#   Compiling Main.hx...
#   Creating bundle: app.rzb
#     Modules: 1
#     Size: 22.94 KB
#     Time: 7ms
```

#### Loading and Executing a Bundle

```rust
use compiler::ir::blade::{load_bundle, RayzorBundle};
use compiler::codegen::tiered_backend::TieredBackend;

// Load the bundle (< 1ms)
let bundle = load_bundle("app.rzb")?;

// Get entry module and function
let entry_module = bundle.entry_module()
    .ok_or("No entry module")?;
let entry_func = bundle.entry_function();

// Create interpreter backend
let mut backend = TieredBackend::with_symbols(config, &symbols)?;

// Load module into backend
backend.compile_module(entry_module.clone())?;

// Find and execute main
let main_id = find_main_function(entry_module)?;
backend.execute_function(main_id, vec![])?;
```

#### From Haxe Source to Bundle

```haxe
// Main.hx
class Main {
    static function main() {
        trace("Hello from bundle!");
        var sum = 0;
        for (i in 0...10) {
            sum += i;
        }
        trace(sum);  // 45
    }
}
```

```bash
rayzor bundle Main.hx -o app.rzb
rayzor run app.rzb
```

### Performance

| Metric | Time | Notes |
|--------|------|-------|
| **Bundle Load** | ~530µs | Deserialize from disk |
| **Interpreter Exec** | ~840µs | Execute via MIR interpreter |
| **Total from Bundle** | ~1.4ms | Load + execute |
| **Full Compilation** | ~6.9ms | Parse + type check + lower |
| **Speedup** | **13.1x** | Bundle vs full compile |

### Bundle Size

| Content | Typical Size |
|---------|--------------|
| Simple "Hello World" | ~5 KB |
| Medium app (10 classes) | ~20-50 KB |
| Large app with stdlib | ~100-500 KB |

The postcard format is extremely compact - typically 10-20x smaller than equivalent JSON.

### Validation

Bundles are validated on load:

1. **Magic Number** - Must be "RZBF"
2. **Version** - Must match current BUNDLE_VERSION
3. **Entry Module** - Must exist in module table
4. **Module Integrity** - All modules must deserialize correctly

### Error Handling

```rust
#[derive(Debug)]
pub enum BladeError {
    /// Magic number doesn't match "RZBF"
    InvalidMagic,

    /// Version mismatch
    VersionMismatch { expected: u32, found: u32 },

    /// Serialization/deserialization failed
    SerializationError(String),

    /// IO error (file not found, permissions, etc.)
    IoError(std::io::Error),

    /// Entry module not found in bundle
    EntryModuleNotFound(String),
}
```

### Future Enhancements

1. **Signing** - Cryptographic signatures for integrity
2. **Lazy Loading** - Load modules on demand
3. **Streaming** - Stream large bundles without full load
4. **Incremental Updates** - Patch bundles instead of full rebuild
5. **AOT Hints** - Embed profiling data for better JIT decisions

Bundles are zstd-compressed by default; the `compressed` flag records it and the
loader detects it from the zstd magic bytes. `rayzor bundle --no-compress`
turns it off.

### Related Formats

| Format | Runtime | Purpose |
|--------|---------|---------|
| `.rzb` (RayzorBundle) | Rayzor | Single-file executable |
| `.blade` (BLADE) | Rayzor | Module cache |
| `.hl` | HashLink | Haxe bytecode |
| `.class` | JVM | Java bytecode |
| `.pyc` | Python | Compiled bytecode |
| `.wasm` | WebAssembly | Portable binary |

RayzorBundle serves the same purpose as HashLink's `.hl` format - a portable, pre-compiled executable that can be distributed without source code.
