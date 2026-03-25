//! RPKG Format — Rayzor Package distribution format
//!
//! `.rpkg` is the standard distribution format for Rayzor packages. A package
//! can contain Haxe source libraries, platform-specific native libraries, or
//! both. Pure-Haxe packages bundle `.hx` source files that are compiled on
//! import. Native packages additionally include a platform dylib and a
//! serialized method table for FFI binding.
//!
//! # Binary Layout
//!
//! ```text
//! [entry 1 data][entry 2 data]...[entry N data][TOC (postcard)][toc_size: u32][version: u32][magic: "RPKG"]
//! ```
//!
//! The footer (last 12 bytes) is read first: 4-byte magic `b"RPKG"`, 4-byte
//! format version, 4-byte TOC size.  The TOC is `postcard`-deserialized from
//! the `toc_size` bytes immediately before the footer.

pub mod install;
pub mod pack;
pub mod registry;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RPKG_MAGIC: &[u8; 4] = b"RPKG";
const RPKG_VERSION: u32 = 1;
const FOOTER_SIZE: usize = 12; // magic(4) + version(4) + toc_size(4)

// ---------------------------------------------------------------------------
// TOC types (serialized with postcard)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpkgToc {
    /// Human-readable package name (e.g. "rayzor-gpu")
    pub package_name: String,
    /// Entries in the archive
    pub entries: Vec<RpkgEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpkgEntry {
    pub kind: EntryKind,
    /// Byte offset from the start of the file
    pub offset: u64,
    /// Byte length of this entry's data
    pub size: u64,
    /// Kind-specific metadata (see below)
    pub meta: EntryMeta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryKind {
    NativeLib,
    HaxeSource,
    MethodTable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryMeta {
    /// For `NativeLib`: target triple components
    NativeLib { os: String, arch: String },
    /// For `HaxeSource`: module path relative to package root
    HaxeSource { module_path: String },
    /// For `MethodTable`: plugin name
    MethodTable { plugin_name: String },
}

// ---------------------------------------------------------------------------
// Serializable method descriptor (mirrors NativeMethodInfo)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodDescEntry {
    pub symbol_name: String,
    pub class_name: String,
    pub method_name: String,
    pub is_static: bool,
    pub param_count: u8,
    pub return_type: u8,
    pub param_types: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Loaded RPKG (result of reading an .rpkg file)
// ---------------------------------------------------------------------------

/// A parsed rpkg archive ready for use.
pub struct LoadedRpkg {
    /// Package name from TOC
    pub package_name: String,
    /// Method descriptors (empty if no MethodTable entry)
    pub methods: Vec<MethodDescEntry>,
    /// Haxe source files: module_path → source text
    pub haxe_sources: HashMap<String, String>,
    /// Raw native lib bytes for the current platform (if present)
    pub native_lib_bytes: Option<Vec<u8>>,
    /// Plugin name from method table entry
    pub plugin_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RpkgError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    DeserializationFailed(postcard::Error),
    TocTooLarge(u64),
    NoNativeLibForPlatform,
}

impl std::fmt::Display for RpkgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RpkgError::Io(e) => write!(f, "I/O error: {}", e),
            RpkgError::InvalidMagic => write!(f, "not a valid .rpkg file (bad magic)"),
            RpkgError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "unsupported rpkg version {} (expected {})",
                    v, RPKG_VERSION
                )
            }
            RpkgError::DeserializationFailed(e) => write!(f, "failed to deserialize TOC: {}", e),
            RpkgError::TocTooLarge(s) => write!(f, "TOC size {} exceeds file size", s),
            RpkgError::NoNativeLibForPlatform => {
                write!(f, "no native library for current platform")
            }
        }
    }
}

impl From<std::io::Error> for RpkgError {
    fn from(e: std::io::Error) -> Self {
        RpkgError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Current platform identifiers for matching NativeLib entries.
fn current_os() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    }
}

fn current_arch() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        "unknown"
    }
}

/// Load and parse an `.rpkg` file, extracting method table, haxe sources,
/// and the native library matching the current platform.
pub fn load_rpkg(path: &Path) -> Result<LoadedRpkg, RpkgError> {
    use std::io::Read;

    let data = std::fs::read(path)?;

    if data.len() < FOOTER_SIZE {
        return Err(RpkgError::InvalidMagic);
    }

    // Read footer (last 12 bytes): [toc_size: u32][version: u32][magic: 4 bytes]
    let footer_start = data.len() - FOOTER_SIZE;
    let toc_size = u32::from_le_bytes(data[footer_start..footer_start + 4].try_into().unwrap());
    let version = u32::from_le_bytes(data[footer_start + 4..footer_start + 8].try_into().unwrap());
    let magic = &data[footer_start + 8..footer_start + 12];

    if magic != RPKG_MAGIC {
        return Err(RpkgError::InvalidMagic);
    }
    if version != RPKG_VERSION {
        return Err(RpkgError::UnsupportedVersion(version));
    }

    let toc_size = toc_size as usize;
    if toc_size > footer_start {
        return Err(RpkgError::TocTooLarge(toc_size as u64));
    }

    // Deserialize TOC
    let toc_start = footer_start - toc_size;
    let toc: RpkgToc = postcard::from_bytes(&data[toc_start..footer_start])
        .map_err(RpkgError::DeserializationFailed)?;

    let os = current_os();
    let arch = current_arch();

    let mut methods = Vec::new();
    let mut haxe_sources = HashMap::new();
    let mut native_lib_bytes = None;
    let mut plugin_name = None;

    for entry in &toc.entries {
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        if end > data.len() {
            return Err(RpkgError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "entry data out of bounds: {}..{} in {} byte file",
                    start,
                    end,
                    data.len()
                ),
            )));
        }
        let entry_data = &data[start..end];

        match (&entry.kind, &entry.meta) {
            (
                EntryKind::NativeLib,
                EntryMeta::NativeLib {
                    os: lib_os,
                    arch: lib_arch,
                },
            ) => {
                if lib_os == os && lib_arch == arch {
                    native_lib_bytes = Some(entry_data.to_vec());
                }
            }
            (EntryKind::HaxeSource, EntryMeta::HaxeSource { module_path }) => {
                if let Ok(source) = std::str::from_utf8(entry_data) {
                    haxe_sources.insert(module_path.clone(), source.to_string());
                }
            }
            (EntryKind::MethodTable, EntryMeta::MethodTable { plugin_name: name }) => {
                plugin_name = Some(name.clone());
                let table: Vec<MethodDescEntry> =
                    postcard::from_bytes(entry_data).map_err(RpkgError::DeserializationFailed)?;
                methods = table;
            }
            _ => {} // mismatched kind/meta — skip
        }
    }

    Ok(LoadedRpkg {
        package_name: toc.package_name,
        methods,
        haxe_sources,
        native_lib_bytes,
        plugin_name,
    })
}

/// Strip an rpkg to keep only the native lib matching a specific platform.
///
/// Copies all non-NativeLib entries unchanged and only includes the NativeLib
/// entry matching (target_os, target_arch). Produces a smaller platform-specific rpkg.
pub fn strip_rpkg(
    input: &Path,
    target_os: &str,
    target_arch: &str,
    output: &Path,
) -> Result<(), RpkgError> {
    use std::io::Write;

    let data = std::fs::read(input)?;

    if data.len() < FOOTER_SIZE {
        return Err(RpkgError::InvalidMagic);
    }

    let footer_start = data.len() - FOOTER_SIZE;
    let toc_size =
        u32::from_le_bytes(data[footer_start..footer_start + 4].try_into().unwrap()) as usize;
    let version = u32::from_le_bytes(data[footer_start + 4..footer_start + 8].try_into().unwrap());
    let magic = &data[footer_start + 8..footer_start + 12];

    if magic != RPKG_MAGIC {
        return Err(RpkgError::InvalidMagic);
    }
    if version != RPKG_VERSION {
        return Err(RpkgError::UnsupportedVersion(version));
    }
    if toc_size > footer_start {
        return Err(RpkgError::TocTooLarge(toc_size as u64));
    }

    let toc_start = footer_start - toc_size;
    let toc: RpkgToc = postcard::from_bytes(&data[toc_start..footer_start])
        .map_err(RpkgError::DeserializationFailed)?;

    // Rebuild with only matching NativeLib entries
    let mut file = std::fs::File::create(output)?;
    let mut new_entries = Vec::new();
    let mut offset: u64 = 0;

    for entry in &toc.entries {
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        let entry_data = &data[start..end];

        match &entry.meta {
            EntryMeta::NativeLib {
                os: lib_os,
                arch: lib_arch,
            } => {
                // Only keep the matching platform
                if lib_os == target_os && lib_arch == target_arch {
                    file.write_all(entry_data)?;
                    new_entries.push(RpkgEntry {
                        kind: entry.kind,
                        offset,
                        size: entry.size,
                        meta: entry.meta.clone(),
                    });
                    offset += entry.size;
                }
            }
            _ => {
                // Keep all non-NativeLib entries
                file.write_all(entry_data)?;
                new_entries.push(RpkgEntry {
                    kind: entry.kind,
                    offset,
                    size: entry.size,
                    meta: entry.meta.clone(),
                });
                offset += entry.size;
            }
        }
    }

    let new_toc = RpkgToc {
        package_name: toc.package_name,
        entries: new_entries,
    };
    let toc_bytes = postcard::to_allocvec(&new_toc).map_err(RpkgError::DeserializationFailed)?;
    let toc_size = toc_bytes.len() as u32;
    file.write_all(&toc_bytes)?;
    file.write_all(&toc_size.to_le_bytes())?;
    file.write_all(&RPKG_VERSION.to_le_bytes())?;
    file.write_all(RPKG_MAGIC)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpkg::pack::RpkgBuilder;
    use std::io::Write;

    #[test]
    fn round_trip_rpkg() {
        let methods = vec![MethodDescEntry {
            symbol_name: "my_func".to_string(),
            class_name: "MyClass".to_string(),
            method_name: "doStuff".to_string(),
            is_static: true,
            param_count: 2,
            return_type: 1,          // I64
            param_types: vec![1, 2], // I64, F64
        }];

        let haxe_src =
            "extern class MyClass {\n  static function doStuff(a:Int, b:Float):Int;\n}\n";

        let mut builder = RpkgBuilder::new("test-pkg");
        builder.add_native_lib(b"fake dylib bytes", "macos", "aarch64");
        builder.add_haxe_source("test/MyClass.hx", haxe_src);
        builder.add_method_table("test_plugin", &methods);

        let tmp = std::env::temp_dir().join("test_round_trip.rpkg");
        builder.write(&tmp).expect("write failed");

        let loaded = load_rpkg(&tmp).expect("load failed");
        assert_eq!(loaded.package_name, "test-pkg");
        assert_eq!(loaded.methods.len(), 1);
        assert_eq!(loaded.methods[0].symbol_name, "my_func");
        assert_eq!(loaded.methods[0].param_types, vec![1, 2]);
        assert_eq!(loaded.haxe_sources.len(), 1);
        assert!(loaded.haxe_sources.contains_key("test/MyClass.hx"));
        assert_eq!(loaded.plugin_name, Some("test_plugin".to_string()));

        // native lib matches current platform
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            assert!(loaded.native_lib_bytes.is_some());
            assert_eq!(loaded.native_lib_bytes.unwrap(), b"fake dylib bytes");
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn invalid_magic_rejected() {
        let tmp = std::env::temp_dir().join("test_bad_magic.rpkg");
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(b"NOT_AN_RPKG_FILE").unwrap();
        drop(f);

        let result = load_rpkg(&tmp);
        assert!(result.is_err());

        std::fs::remove_file(&tmp).ok();
    }
}
