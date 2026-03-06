//! BLADE Format - Blazing Language Artifact Deployment Environment
//!
//! This module provides serialization and deserialization of MIR (Mid-level IR)
//! to the `.blade` binary format using the `postcard` crate for efficient,
//! compact binary serialization.
//!
//! # BLADE Format Benefits
//!
//! - **Incremental Compilation**: Avoid recompiling unchanged modules (30x faster)
//! - **Module Caching**: Fast startup by loading pre-compiled modules
//! - **Build Artifacts**: Distribute pre-compiled libraries
//! - **Compact**: Uses `postcard` for minimal size
//! - **Fast**: Efficient binary deserialization
//!
//! # File Extension
//!
//! - **`.blade`** - Compiled Rayzor module (binary format)
//!
//! # Usage
//!
//! ```rust,ignore
//! use compiler::ir::blade::{save_blade, load_blade, BladeMetadata};
//!
//! // Serialize MIR to .blade file
//! let metadata = BladeMetadata {
//!     name: "MyModule".to_string(),
//!     source_path: "src/Main.hx".to_string(),
//!     source_timestamp: 1234567890,
//!     compile_timestamp: 1234567900,
//!     dependencies: vec![],
//!     compiler_version: env!("CARGO_PKG_VERSION").to_string(),
//! };
//! save_blade("output.blade", &mir_module, metadata)?;
//!
//! // Deserialize .blade file to MIR
//! let (mir_module, metadata) = load_blade("output.blade")?;
//! ```

use crate::ir::IrModule;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// BLADE file magic number (first 4 bytes)
const BLADE_MAGIC: &[u8; 4] = b"BLAD";

/// Current BLADE format version
const BLADE_VERSION: u32 = 2;

/// Metadata about the compiled module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeMetadata {
    /// Module name
    pub name: String,

    /// Source file path
    pub source_path: String,

    /// Hash of the source content for cache validation
    /// More reliable than timestamps for detecting changes
    pub source_hash: u64,

    /// Source file modification timestamp (Unix epoch seconds)
    pub source_timestamp: u64,

    /// Compilation timestamp (Unix epoch seconds)
    pub compile_timestamp: u64,

    /// List of module dependencies
    pub dependencies: Vec<String>,

    /// Compiler version that created this BLADE file
    pub compiler_version: String,
}

/// MIR-level cross-reference maps for cache restoration.
/// Keyed by name (not SymbolId/TypeId) so they survive across compilations
/// where IDs are re-assigned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeCachedMaps {
    /// Function entries: (class_name, method_name, ir_func_id, is_constructor)
    pub functions: Vec<BladeFuncEntry>,
    /// Field entries: (class_name, field_name, field_index)
    pub fields: Vec<BladeFieldEntry>,
    /// Class allocation sizes: (class_qualified_name, alloc_bytes)
    pub class_sizes: Vec<(String, u64)>,
    /// Property access entries: (class_name, field_name, getter, setter)
    #[serde(default)]
    pub properties: Vec<BladePropertyEntry>,
}

/// A function entry in the cached maps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeFuncEntry {
    /// Qualified class name (e.g., "haxe.ds.BalancedTree")
    pub class_name: String,
    /// Method name (e.g., "set", "new")
    pub method_name: String,
    /// Original MIR function ID (pre-renumber)
    pub func_id: u32,
    /// Whether this is a constructor
    pub is_constructor: bool,
}

/// A field entry in the cached maps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeFieldEntry {
    /// Qualified class name
    pub class_name: String,
    /// Field name
    pub field_name: String,
    /// Field index in the class struct (0 = header, 1+ = user fields)
    pub field_index: u32,
}

/// A property access entry in the cached maps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladePropertyEntry {
    /// Qualified class name
    pub class_name: String,
    /// Field name (the property field, e.g., "length")
    pub field_name: String,
    /// Getter accessor kind
    pub getter: BladeAccessor,
    /// Setter accessor kind
    pub setter: BladeAccessor,
}

/// Serializable property accessor kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BladeAccessor {
    Default,
    Null,
    Never,
    Dynamic,
    Method(String),
}

/// A complete BLADE module ready for serialization
#[derive(Debug, Serialize, Deserialize)]
struct BladeModule {
    /// Magic number for validation
    magic: [u8; 4],

    /// Format version
    version: u32,

    /// Module metadata
    metadata: BladeMetadata,

    /// The actual MIR module (directly serialized)
    mir: IrModule,

    /// Type info for symbol registration on cache load (added in v2)
    symbols: Option<BladeTypeInfo>,

    /// MIR-level cross-reference maps, keyed by name (added in v2)
    cached_maps: Option<BladeCachedMaps>,
}

/// Errors that can occur during BLADE operations
#[derive(Debug)]
pub enum BladeError {
    /// I/O error
    Io(std::io::Error),

    /// Serialization error
    Serialization(postcard::Error),

    /// Invalid magic number
    InvalidMagic,

    /// Unsupported version
    UnsupportedVersion(u32),

    /// Compression/decompression error
    Compression(String),
}

impl std::fmt::Display for BladeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BladeError::Io(e) => write!(f, "I/O error: {}", e),
            BladeError::Serialization(e) => write!(f, "Serialization error: {}", e),
            BladeError::Compression(e) => write!(f, "Compression error: {}", e),
            BladeError::InvalidMagic => write!(f, "Invalid BLADE magic number"),
            BladeError::UnsupportedVersion(v) => write!(f, "Unsupported BLADE version: {}", v),
        }
    }
}

impl std::error::Error for BladeError {}

impl From<std::io::Error> for BladeError {
    fn from(e: std::io::Error) -> Self {
        BladeError::Io(e)
    }
}

impl From<postcard::Error> for BladeError {
    fn from(e: postcard::Error) -> Self {
        BladeError::Serialization(e)
    }
}

/// Save a MIR module to a .blade file
///
/// # Arguments
///
/// * `path` - Path to the .blade file to create
/// * `module` - The MIR module to serialize
/// * `metadata` - Metadata about the module
///
/// # Example
///
/// ```rust,ignore
/// let metadata = BladeMetadata {
///     name: "Main".to_string(),
///     source_path: "Main.hx".to_string(),
///     source_timestamp: 1234567890,
///     compile_timestamp: 1234567900,
///     dependencies: vec![],
///     compiler_version: env!("CARGO_PKG_VERSION").to_string(),
/// };
/// save_blade("Main.blade", &mir_module, metadata)?;
/// ```
pub fn save_blade(
    path: impl AsRef<Path>,
    module: &IrModule,
    metadata: BladeMetadata,
) -> Result<(), BladeError> {
    save_blade_with_state(path, module, metadata, None, None)
}

/// Save a MIR module with optional type info and cached maps
pub fn save_blade_with_state(
    path: impl AsRef<Path>,
    module: &IrModule,
    metadata: BladeMetadata,
    symbols: Option<BladeTypeInfo>,
    cached_maps: Option<BladeCachedMaps>,
) -> Result<(), BladeError> {
    let blade = BladeModule {
        magic: *BLADE_MAGIC,
        version: BLADE_VERSION,
        metadata,
        mir: module.clone(),
        symbols,
        cached_maps,
    };

    // Serialize using postcard
    let bytes = postcard::to_allocvec(&blade)?;

    // Write to file
    fs::write(path, bytes)?;

    Ok(())
}

/// Load a MIR module from a .blade file
///
/// # Arguments
///
/// * `path` - Path to the .blade file to load
///
/// # Returns
///
/// A tuple of (IrModule, BladeMetadata)
///
/// # Example
///
/// ```rust,ignore
/// let (mir_module, metadata) = load_blade("Main.blade")?;
/// println!("Loaded module: {}", metadata.name);
/// ```
pub fn load_blade(
    path: impl AsRef<Path>,
) -> Result<
    (
        IrModule,
        BladeMetadata,
        Option<BladeTypeInfo>,
        Option<BladeCachedMaps>,
    ),
    BladeError,
> {
    // Read file
    let bytes = fs::read(path)?;

    // Deserialize using postcard
    let blade: BladeModule = postcard::from_bytes(&bytes)?;

    // Validate magic number
    if &blade.magic != BLADE_MAGIC {
        return Err(BladeError::InvalidMagic);
    }

    // Check version
    if blade.version != BLADE_VERSION {
        return Err(BladeError::UnsupportedVersion(blade.version));
    }

    Ok((blade.mir, blade.metadata, blade.symbols, blade.cached_maps))
}

// ============================================================================
// BLADE Symbol Format - Pre-resolved symbol storage for fast startup
// ============================================================================

/// Magic number for symbol manifest files
const SYMBOL_MAGIC: &[u8; 4] = b"BSYM";

/// Current symbol format version
const SYMBOL_VERSION: u32 = 1;

/// Complete symbol information for a field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeFieldInfo {
    /// Field name
    pub name: String,
    /// Field type as string (e.g., "Int", "Array<String>")
    pub field_type: String,
    /// Is this field public?
    pub is_public: bool,
    /// Is this a static field?
    pub is_static: bool,
    /// Is this field final/immutable?
    pub is_final: bool,
    /// Does this field have a default value?
    pub has_default: bool,
}

/// Parameter information for methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeParamInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type as string
    pub param_type: String,
    /// Does this parameter have a default value?
    pub has_default: bool,
    /// Is this parameter optional (nullable)?
    pub is_optional: bool,
}

/// Complete symbol information for a method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeMethodInfo {
    /// Method name
    pub name: String,
    /// Method parameters
    pub params: Vec<BladeParamInfo>,
    /// Return type as string
    pub return_type: String,
    /// Is this method public?
    pub is_public: bool,
    /// Is this a static method?
    pub is_static: bool,
    /// Is this an inline method?
    pub is_inline: bool,
    /// Type parameters for generic methods
    pub type_params: Vec<String>,
}

/// Complete symbol information for a class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeClassInfo {
    /// Class name (short name, e.g., "Bytes")
    pub name: String,
    /// Package path (e.g., ["haxe", "io"])
    pub package: Vec<String>,
    /// Superclass qualified name (if any)
    pub extends: Option<String>,
    /// Implemented interfaces
    pub implements: Vec<String>,
    /// Type parameters (e.g., ["T", "U"])
    pub type_params: Vec<String>,
    /// Is this an extern class?
    pub is_extern: bool,
    /// Is this an abstract class?
    pub is_abstract: bool,
    /// Is this a final class?
    pub is_final: bool,
    /// Instance fields
    pub fields: Vec<BladeFieldInfo>,
    /// Instance methods
    pub methods: Vec<BladeMethodInfo>,
    /// Static fields
    pub static_fields: Vec<BladeFieldInfo>,
    /// Static methods
    pub static_methods: Vec<BladeMethodInfo>,
    /// Constructor info (if any)
    pub constructor: Option<BladeMethodInfo>,
    /// Native name from @:native metadata (e.g., "rayzor::concurrent::Arc")
    /// Lowered form replaces :: with _ for symbol resolution
    pub native_name: Option<String>,
}

/// Enum variant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeEnumVariantInfo {
    /// Variant name
    pub name: String,
    /// Constructor parameters (for variants with data)
    pub params: Vec<BladeParamInfo>,
    /// Ordinal index
    pub index: usize,
}

/// Complete symbol information for an enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeEnumInfo {
    /// Enum name
    pub name: String,
    /// Package path
    pub package: Vec<String>,
    /// Type parameters
    pub type_params: Vec<String>,
    /// Enum variants
    pub variants: Vec<BladeEnumVariantInfo>,
    /// Is this an extern enum?
    pub is_extern: bool,
}

/// Type alias information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeTypeAliasInfo {
    /// Alias name
    pub name: String,
    /// Package path
    pub package: Vec<String>,
    /// Type parameters
    pub type_params: Vec<String>,
    /// Target type as string
    pub target_type: String,
}

/// Abstract type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeAbstractInfo {
    /// Abstract name
    pub name: String,
    /// Package path
    pub package: Vec<String>,
    /// Type parameters
    pub type_params: Vec<String>,
    /// Underlying type
    pub underlying_type: String,
    /// Forward fields (from @:forward)
    pub forward_fields: Vec<String>,
    /// From types (implicit conversions from)
    pub from_types: Vec<String>,
    /// To types (implicit conversions to)
    pub to_types: Vec<String>,
    /// Methods defined on the abstract
    pub methods: Vec<BladeMethodInfo>,
    /// Static methods defined on the abstract
    pub static_methods: Vec<BladeMethodInfo>,
    /// Native name from @:native metadata (e.g., "rayzor::Ptr")
    pub native_name: Option<String>,
}

/// All type information for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeTypeInfo {
    /// Classes defined in this module
    pub classes: Vec<BladeClassInfo>,
    /// Enums defined in this module
    pub enums: Vec<BladeEnumInfo>,
    /// Type aliases defined in this module
    pub type_aliases: Vec<BladeTypeAliasInfo>,
    /// Abstract types defined in this module
    pub abstracts: Vec<BladeAbstractInfo>,
}

impl Default for BladeTypeInfo {
    fn default() -> Self {
        Self {
            classes: Vec::new(),
            enums: Vec::new(),
            type_aliases: Vec::new(),
            abstracts: Vec::new(),
        }
    }
}

/// Symbol manifest for all pre-compiled stdlib symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeSymbolManifest {
    /// Magic number for validation
    magic: [u8; 4],
    /// Format version
    version: u32,
    /// Compiler version
    pub compiler_version: String,
    /// Build timestamp
    pub build_timestamp: u64,
    /// All modules in this manifest
    pub modules: Vec<BladeModuleSymbols>,
}

/// Symbols for a single module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BladeModuleSymbols {
    /// Module name (e.g., "haxe.io.Bytes")
    pub name: String,
    /// Source file path
    pub source_path: String,
    /// Source hash for cache validation
    pub source_hash: u64,
    /// Type information
    pub types: BladeTypeInfo,
    /// Dependencies (other modules this depends on)
    pub dependencies: Vec<String>,
}

/// Save a symbol manifest to file
pub fn save_symbol_manifest(
    path: impl AsRef<Path>,
    modules: Vec<BladeModuleSymbols>,
) -> Result<(), BladeError> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let manifest = BladeSymbolManifest {
        magic: *SYMBOL_MAGIC,
        version: SYMBOL_VERSION,
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        build_timestamp: now,
        modules,
    };

    let bytes = postcard::to_allocvec(&manifest)?;
    fs::write(path, bytes)?;

    Ok(())
}

/// Load a symbol manifest from file
pub fn load_symbol_manifest(path: impl AsRef<Path>) -> Result<BladeSymbolManifest, BladeError> {
    let bytes = fs::read(path)?;
    let manifest: BladeSymbolManifest = postcard::from_bytes(&bytes)?;

    if &manifest.magic != SYMBOL_MAGIC {
        return Err(BladeError::InvalidMagic);
    }

    if manifest.version != SYMBOL_VERSION {
        return Err(BladeError::UnsupportedVersion(manifest.version));
    }

    Ok(manifest)
}

// ============================================================================
// Rayzor Bundle Format (.rzb) - Single-file executable bundle
// ============================================================================
//
// The .rzb format bundles all compiled modules into a single file for
// instant startup, similar to HashLink's .hl format.
//
// Structure:
//   [Header]
//     - Magic: "RZBF" (4 bytes)
//     - Version: u32
//     - Flags: u32 (compression, debug info, etc.)
//     - Entry module index: u32
//     - Entry function name length: u32
//     - Entry function name: [u8]
//     - Module count: u32
//     - Symbol manifest offset: u64
//     - Symbol manifest size: u64
//   [Module Table] (for fast lookup)
//     - For each module:
//       - Name length: u32
//       - Name: [u8]
//       - Offset: u64
//       - Size: u64
//   [Module Data]
//     - Serialized IrModules (postcard format)
//   [Symbol Manifest]
//     - Embedded BladeSymbolManifest
//

/// Magic number for Rayzor Bundle files
const BUNDLE_MAGIC: &[u8; 4] = b"RZBF";

/// Current bundle format version
const BUNDLE_VERSION: u32 = 1;

/// Bundle flags
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BundleFlags {
    /// Is the bundle compressed (zstd)?
    pub compressed: bool,
    /// Include debug info?
    pub debug_info: bool,
    /// Include source maps?
    pub source_maps: bool,
}

impl Default for BundleFlags {
    fn default() -> Self {
        Self {
            compressed: false,
            debug_info: false,
            source_maps: false,
        }
    }
}

/// Entry in the module table
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleTableEntry {
    /// Module name (e.g., "Main", "haxe.io.Bytes")
    name: String,
    /// Offset in the data section
    offset: u64,
    /// Size of the serialized module
    size: u64,
}

/// Rayzor Bundle - all modules in a single file
#[derive(Debug, Serialize, Deserialize)]
pub struct RayzorBundle {
    /// Magic number
    magic: [u8; 4],
    /// Format version
    version: u32,
    /// Bundle flags
    pub flags: BundleFlags,
    /// Entry point module name
    entry_module: String,
    /// Entry point function name (usually "main" or "Main_main")
    entry_function: String,
    /// Entry point module index (for O(1) lookup)
    entry_module_index: Option<usize>,
    /// Entry point function ID (for O(1) lookup, avoids iterating at runtime)
    entry_function_id: Option<crate::ir::IrFunctionId>,
    /// Module table (name -> offset/size)
    module_table: Vec<ModuleTableEntry>,
    /// All modules serialized
    modules: Vec<IrModule>,
    /// Optional embedded symbol manifest
    symbols: Option<BladeSymbolManifest>,
    /// Build metadata
    build_info: BundleBuildInfo,
}

/// Build information for the bundle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleBuildInfo {
    /// Compiler version
    pub compiler_version: String,
    /// Build timestamp
    pub build_timestamp: u64,
    /// Target platform (e.g., "aarch64-apple-darwin")
    pub target_platform: String,
    /// Original source files (for debugging)
    pub source_files: Vec<String>,
}

impl RayzorBundle {
    /// Create a new bundle from modules
    pub fn new(
        modules: Vec<IrModule>,
        entry_module: &str,
        entry_function: &str,
        symbols: Option<BladeSymbolManifest>,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // Build module table
        let module_table: Vec<ModuleTableEntry> = modules
            .iter()
            .enumerate()
            .map(|(i, m)| ModuleTableEntry {
                name: m.name.clone(),
                offset: i as u64, // Will be calculated during serialization
                size: 0,
            })
            .collect();

        let source_files: Vec<String> = modules.iter().map(|m| m.source_file.clone()).collect();

        // Pre-compute entry module index and function ID at bundle creation time
        // This eliminates runtime iteration when loading the bundle
        let entry_module_index = modules.iter().position(|m| m.name == entry_module);

        let entry_function_id = entry_module_index.and_then(|idx| {
            modules[idx]
                .functions
                .iter()
                .find(|(_, f)| f.name == entry_function)
                .map(|(id, _)| *id)
        });

        Self {
            magic: *BUNDLE_MAGIC,
            version: BUNDLE_VERSION,
            flags: BundleFlags::default(),
            entry_module: entry_module.to_string(),
            entry_function: entry_function.to_string(),
            entry_module_index,
            entry_function_id,
            module_table,
            modules,
            symbols,
            build_info: BundleBuildInfo {
                compiler_version: env!("CARGO_PKG_VERSION").to_string(),
                build_timestamp: now,
                target_platform: std::env::consts::ARCH.to_string(),
                source_files,
            },
        }
    }

    /// Get the entry module (O(1) using pre-computed index)
    pub fn entry_module(&self) -> Option<&IrModule> {
        self.entry_module_index
            .and_then(|idx| self.modules.get(idx))
    }

    /// Get the entry function ID (O(1), pre-computed at bundle creation)
    /// This eliminates the need to iterate through functions at runtime
    pub fn entry_function_id(&self) -> Option<crate::ir::IrFunctionId> {
        self.entry_function_id
    }

    /// Get a module by name
    pub fn get_module(&self, name: &str) -> Option<&IrModule> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// Get all modules
    pub fn modules(&self) -> &[IrModule] {
        &self.modules
    }

    /// Get the entry function name
    pub fn entry_function(&self) -> &str {
        &self.entry_function
    }

    /// Get embedded symbols (if any)
    pub fn symbols(&self) -> Option<&BladeSymbolManifest> {
        self.symbols.as_ref()
    }

    /// Get build info
    pub fn build_info(&self) -> &BundleBuildInfo {
        &self.build_info
    }

    /// Get module count
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

/// Save a Rayzor Bundle to file
///
/// # Arguments
///
/// * `path` - Path to the .rzb file to create
/// * `bundle` - The bundle to save
///
/// # Example
///
/// ```rust,ignore
/// let bundle = RayzorBundle::new(modules, "Main", "main", None);
/// save_bundle("app.rzb", &bundle)?;
/// ```
pub fn save_bundle(path: impl AsRef<Path>, bundle: &RayzorBundle) -> Result<(), BladeError> {
    let bytes = postcard::to_allocvec(bundle)?;
    let output = if bundle.flags.compressed {
        zstd::encode_all(bytes.as_slice(), 3)
            .map_err(|e| BladeError::Compression(format!("zstd compress: {}", e)))?
    } else {
        bytes
    };
    fs::write(path, output)?;
    Ok(())
}

/// Load a Rayzor Bundle from file
///
/// # Arguments
///
/// * `path` - Path to the .rzb file to load
///
/// # Returns
///
/// The loaded RayzorBundle
///
/// # Example
///
/// ```rust,ignore
/// let bundle = load_bundle("app.rzb")?;
/// println!("Loaded {} modules", bundle.module_count());
/// let entry = bundle.entry_module().unwrap();
/// ```
pub fn load_bundle(path: impl AsRef<Path>) -> Result<RayzorBundle, BladeError> {
    let raw = fs::read(path)?;
    // Detect zstd compression via magic bytes (0x28 0xB5 0x2F 0xFD)
    let bytes =
        if raw.len() >= 4 && raw[0] == 0x28 && raw[1] == 0xB5 && raw[2] == 0x2F && raw[3] == 0xFD {
            zstd::decode_all(raw.as_slice())
                .map_err(|e| BladeError::Compression(format!("zstd decompress: {}", e)))?
        } else {
            raw
        };
    let bundle: RayzorBundle = postcard::from_bytes(&bytes)?;

    // Validate magic
    if &bundle.magic != BUNDLE_MAGIC {
        return Err(BladeError::InvalidMagic);
    }

    // Check version
    if bundle.version != BUNDLE_VERSION {
        return Err(BladeError::UnsupportedVersion(bundle.version));
    }

    Ok(bundle)
}

/// Load a Rayzor Bundle from bytes (for embedded bundles)
pub fn load_bundle_from_bytes(bytes: &[u8]) -> Result<RayzorBundle, BladeError> {
    // Detect zstd compression via magic bytes
    let decompressed;
    let data = if bytes.len() >= 4
        && bytes[0] == 0x28
        && bytes[1] == 0xB5
        && bytes[2] == 0x2F
        && bytes[3] == 0xFD
    {
        decompressed = zstd::decode_all(bytes)
            .map_err(|e| BladeError::Compression(format!("zstd decompress: {}", e)))?;
        &decompressed[..]
    } else {
        bytes
    };
    let bundle: RayzorBundle = postcard::from_bytes(data)?;

    if &bundle.magic != BUNDLE_MAGIC {
        return Err(BladeError::InvalidMagic);
    }

    if bundle.version != BUNDLE_VERSION {
        return Err(BladeError::UnsupportedVersion(bundle.version));
    }

    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::modules::IrModule;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn test_blade_roundtrip() {
        // Create a simple IR module
        let module = IrModule::new("test_module".to_string(), "test.hx".to_string());

        // Create metadata
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = BladeMetadata {
            name: "test_module".to_string(),
            source_path: "test.hx".to_string(),
            source_hash: 0x12345678, // Test hash value
            source_timestamp: now,
            compile_timestamp: now,
            dependencies: vec![],
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Serialize to bytes
        let blade = BladeModule {
            magic: *BLADE_MAGIC,
            version: BLADE_VERSION,
            metadata: metadata.clone(),
            mir: module.clone(),
            symbols: None,
            cached_maps: None,
        };

        let bytes = postcard::to_allocvec(&blade).unwrap();

        // Deserialize
        let decoded: BladeModule = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(&decoded.magic, BLADE_MAGIC);
        assert_eq!(decoded.version, BLADE_VERSION);
        assert_eq!(decoded.metadata.name, "test_module");
        assert_eq!(decoded.mir.name, "test_module");
    }
}
