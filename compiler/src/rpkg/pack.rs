//! RPKG Builder — constructs `.rpkg` archives from components.
//!
//! Packages can be pure Haxe (library classes only), native (extern classes +
//! dylib), or mixed (extern classes, library classes that wrap them, and a
//! dylib). The builder accepts any combination of entries.

use super::{EntryKind, EntryMeta, MethodDescEntry, RpkgEntry, RpkgToc, RPKG_MAGIC, RPKG_VERSION};
use std::path::Path;

/// Accumulates entries and writes the final `.rpkg` archive.
pub struct RpkgBuilder {
    package_name: String,
    /// (kind, meta, raw bytes)
    entries: Vec<(EntryKind, EntryMeta, Vec<u8>)>,
}

impl RpkgBuilder {
    pub fn new(package_name: &str) -> Self {
        RpkgBuilder {
            package_name: package_name.to_string(),
            entries: Vec::new(),
        }
    }

    /// Add a native library for a specific platform.
    pub fn add_native_lib(&mut self, data: &[u8], os: &str, arch: &str) {
        self.entries.push((
            EntryKind::NativeLib,
            EntryMeta::NativeLib {
                os: os.to_string(),
                arch: arch.to_string(),
            },
            data.to_vec(),
        ));
    }

    /// Add a native library from a file path.
    pub fn add_native_lib_from_file(
        &mut self,
        path: &Path,
        os: &str,
        arch: &str,
    ) -> Result<(), std::io::Error> {
        let data = std::fs::read(path)?;
        self.add_native_lib(&data, os, arch);
        Ok(())
    }

    /// Add a Haxe source file for extern class declarations.
    pub fn add_haxe_source(&mut self, module_path: &str, source: &str) {
        self.entries.push((
            EntryKind::HaxeSource,
            EntryMeta::HaxeSource {
                module_path: module_path.to_string(),
            },
            source.as_bytes().to_vec(),
        ));
    }

    /// Add a WASM component (universal fallback for platforms without native lib).
    pub fn add_wasm_component(&mut self, name: &str, data: &[u8]) {
        self.entries.push((
            EntryKind::WasmComponent,
            EntryMeta::WasmComponent {
                name: name.to_string(),
            },
            data.to_vec(),
        ));
    }

    /// Add a JavaScript host module for WASM @:jsImport functions.
    /// `module_name` is the @:jsImport module (e.g., "rayzor-gpu").
    /// `js_source` is the JavaScript source code providing the host implementations.
    pub fn add_js_host(&mut self, module_name: &str, js_source: &str) {
        self.entries.push((
            EntryKind::JsHost,
            EntryMeta::JsHost {
                module_name: module_name.to_string(),
            },
            js_source.as_bytes().to_vec(),
        ));
    }

    /// Add a JS host module with its companion _bg.wasm binary.
    pub fn add_js_host_with_wasm(&mut self, module_name: &str, js_source: &str, wasm_bytes: &[u8]) {
        self.add_js_host(module_name, js_source);
        self.entries.push((
            EntryKind::JsHostWasm,
            EntryMeta::JsHostWasm {
                module_name: module_name.to_string(),
            },
            wasm_bytes.to_vec(),
        ));
    }

    /// Add a serialized method table.
    pub fn add_method_table(&mut self, plugin_name: &str, methods: &[MethodDescEntry]) {
        let data = postcard::to_allocvec(methods).expect("method table serialization failed");
        self.entries.push((
            EntryKind::MethodTable,
            EntryMeta::MethodTable {
                plugin_name: plugin_name.to_string(),
            },
            data,
        ));
    }

    /// Write the complete `.rpkg` archive to disk.
    ///
    /// Layout: [entry data...][TOC (postcard)][toc_size: u32][version: u32][magic: 4]
    pub fn write(&self, output: &Path) -> Result<(), super::RpkgError> {
        use std::io::Write;

        let mut file = std::fs::File::create(output)?;
        let mut toc_entries = Vec::with_capacity(self.entries.len());
        let mut offset: u64 = 0;

        // Write entry data and build TOC
        for (kind, meta, data) in &self.entries {
            file.write_all(data)?;
            toc_entries.push(RpkgEntry {
                kind: *kind,
                offset,
                size: data.len() as u64,
                meta: meta.clone(),
            });
            offset += data.len() as u64;
        }

        // Serialize and write TOC
        let toc = RpkgToc {
            package_name: self.package_name.clone(),
            entries: toc_entries,
        };
        let toc_bytes =
            postcard::to_allocvec(&toc).map_err(super::RpkgError::DeserializationFailed)?;
        let toc_size = toc_bytes.len() as u32;
        file.write_all(&toc_bytes)?;

        // Write footer: toc_size, version, magic
        file.write_all(&toc_size.to_le_bytes())?;
        file.write_all(&RPKG_VERSION.to_le_bytes())?;
        file.write_all(RPKG_MAGIC)?;

        Ok(())
    }
}

/// Build an `.rpkg` from a compiled native dylib and a directory of `.hx` files.
///
/// Single-dylib convenience wrapper — tags the dylib with the current platform.
pub fn build_from_dylib(
    package_name: &str,
    dylib_path: &Path,
    haxe_dir: &Path,
    output: &Path,
) -> Result<(), String> {
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        return Err("unsupported OS".to_string());
    };
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        return Err("unsupported architecture".to_string());
    };

    build_from_dylibs(package_name, &[(dylib_path, os, arch)], haxe_dir, output)
}

/// Build an `.rpkg` from multiple platform dylibs and a directory of `.hx` files.
///
/// Each entry in `dylibs` is `(path, os, arch)`. Method descriptors are extracted
/// from the first dylib that has a `rayzor_rpkg_entry` or `plugin_describe` export.
pub fn build_from_dylibs(
    package_name: &str,
    dylibs: &[(&Path, &str, &str)],
    haxe_dir: &Path,
    output: &Path,
) -> Result<(), String> {
    let mut builder = RpkgBuilder::new(package_name);

    // 1. Add each native lib tagged with its platform
    let mut method_table_extracted = false;
    for &(dylib_path, os, arch) in dylibs {
        builder
            .add_native_lib_from_file(dylib_path, os, arch)
            .map_err(|e| format!("failed to read dylib {}: {}", dylib_path.display(), e))?;

        // 2. Extract method descriptors from the first dylib that has them
        if !method_table_extracted {
            let methods = extract_method_table_from_dylib(dylib_path)?;
            if !methods.is_empty() {
                builder.add_method_table(package_name, &methods);
                method_table_extracted = true;
            }
        }
    }

    // 3. Collect .hx files
    if haxe_dir.is_dir() {
        collect_haxe_sources(&mut builder, haxe_dir, haxe_dir)?;
    }

    // 4. Write
    builder
        .write(output)
        .map_err(|e| format!("failed to write rpkg: {}", e))?;

    Ok(())
}

/// Build a pure-Haxe `.rpkg` from a directory of `.hx` files (no native lib).
pub fn build_from_haxe_dir(
    package_name: &str,
    haxe_dir: &Path,
    output: &Path,
) -> Result<(), String> {
    let mut builder = RpkgBuilder::new(package_name);

    if haxe_dir.is_dir() {
        collect_haxe_sources(&mut builder, haxe_dir, haxe_dir)?;
    } else {
        return Err(format!("{} is not a directory", haxe_dir.display()));
    }

    builder
        .write(output)
        .map_err(|e| format!("failed to write rpkg: {}", e))?;

    Ok(())
}

/// Walk a directory tree and add all `.hx` files as HaxeSource entries.
pub fn collect_haxe_sources(
    builder: &mut RpkgBuilder,
    base_dir: &Path,
    current_dir: &Path,
) -> Result<(), String> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(current_dir)
        .map_err(|e| format!("failed to read dir {}: {}", current_dir.display(), e))?
        .map(|entry| {
            entry
                .map(|e| e.path())
                .map_err(|e| format!("dir entry error: {}", e))
        })
        .collect::<Result<_, _>>()?;
    paths.sort(); // Deterministic ordering

    for path in paths {
        if path.is_dir() {
            collect_haxe_sources(builder, base_dir, &path)?;
        } else if path.extension().map(|e| e == "hx").unwrap_or(false) {
            let rel_path = path
                .strip_prefix(base_dir)
                .map_err(|e| format!("strip_prefix failed: {}", e))?;
            let module_path = rel_path.to_string_lossy().to_string();
            let source = std::fs::read_to_string(&path)
                .map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
            builder.add_haxe_source(&module_path, &source);
        }
    }

    Ok(())
}

/// Load the dylib and extract its method descriptor table.
///
/// First tries the universal `rayzor_rpkg_entry` export (preferred).
/// Falls back to legacy `plugin_describe` names for backward compatibility.
fn extract_method_table_from_dylib(dylib_path: &Path) -> Result<Vec<MethodDescEntry>, String> {
    let lib = unsafe { libloading::Library::new(dylib_path) }
        .map_err(|e| format!("failed to load dylib {}: {}", dylib_path.display(), e))?;

    // Try universal entry point first
    if let Some(methods) = extract_methods_via_rpkg_entry(&lib) {
        return Ok(methods);
    }

    // Legacy fallback: try old-style plugin_describe exports
    type DescribeFn = unsafe extern "C" fn(*mut usize) -> *const rayzor_plugin::NativeMethodDesc;

    let describe_names: &[&[u8]] = &[b"plugin_describe", b"rayzor_plugin_describe"];

    for name in describe_names {
        if let Ok(describe_fn) = unsafe { lib.get::<DescribeFn>(name) } {
            let methods = read_method_descs_from_ptr(&describe_fn);
            if !methods.is_empty() {
                return Ok(methods);
            }
        }
    }

    // No describe function found — that's OK, might be a plain dylib
    Ok(Vec::new())
}

/// Extract method descriptors via the universal `rayzor_rpkg_entry` export.
fn extract_methods_via_rpkg_entry(lib: &libloading::Library) -> Option<Vec<MethodDescEntry>> {
    type EntryFn = unsafe extern "C" fn() -> rayzor_plugin::RpkgPluginInfo;

    let entry_fn = unsafe { lib.get::<EntryFn>(b"rayzor_rpkg_entry") }.ok()?;
    let info = unsafe { entry_fn() };

    if info.methods_count == 0 || info.methods_ptr.is_null() {
        return Some(Vec::new());
    }

    let slice = unsafe { std::slice::from_raw_parts(info.methods_ptr, info.methods_count) };

    Some(read_native_method_descs(slice))
}

/// Read method descriptors from a `plugin_describe` function pointer.
fn read_method_descs_from_ptr(
    describe_fn: &unsafe extern "C" fn(*mut usize) -> *const rayzor_plugin::NativeMethodDesc,
) -> Vec<MethodDescEntry> {
    let mut count: usize = 0;
    let descs = unsafe { describe_fn(&mut count) };
    if descs.is_null() || count == 0 {
        return Vec::new();
    }

    let slice = unsafe { std::slice::from_raw_parts(descs, count) };
    read_native_method_descs(slice)
}

/// Convert a slice of NativeMethodDesc into MethodDescEntry values.
fn read_native_method_descs(slice: &[rayzor_plugin::NativeMethodDesc]) -> Vec<MethodDescEntry> {
    let mut methods = Vec::with_capacity(slice.len());

    for desc in slice {
        let symbol_name = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                desc.symbol_name,
                desc.symbol_name_len,
            ))
            .to_string()
        };
        let class_name = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                desc.class_name,
                desc.class_name_len,
            ))
            .to_string()
        };
        let method_name = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                desc.method_name,
                desc.method_name_len,
            ))
            .to_string()
        };
        let param_types = desc.param_types[..desc.param_count as usize].to_vec();

        methods.push(MethodDescEntry {
            symbol_name,
            class_name,
            method_name,
            is_static: desc.is_static != 0,
            param_count: desc.param_count,
            return_type: desc.return_type,
            param_types,
        });
    }

    methods
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_empty_package() {
        let builder = RpkgBuilder::new("empty");
        let tmp = std::env::temp_dir().join("test_empty.rpkg");
        builder.write(&tmp).expect("write failed");

        let loaded = super::super::load_rpkg(&tmp).expect("load failed");
        assert_eq!(loaded.package_name, "empty");
        assert!(loaded.methods.is_empty());
        assert!(loaded.haxe_sources.is_empty());
        assert!(loaded.native_lib_bytes.is_none());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn builder_multiple_platforms() {
        let mut builder = RpkgBuilder::new("multi-platform");
        builder.add_native_lib(b"macos-arm", "macos", "aarch64");
        builder.add_native_lib(b"linux-x64", "linux", "x86_64");

        let tmp = std::env::temp_dir().join("test_multi_platform.rpkg");
        builder.write(&tmp).expect("write failed");

        let loaded = super::super::load_rpkg(&tmp).expect("load failed");

        // Should pick the matching platform
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            assert_eq!(
                loaded.native_lib_bytes.as_deref(),
                Some(b"macos-arm" as &[u8])
            );
        } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
            assert_eq!(
                loaded.native_lib_bytes.as_deref(),
                Some(b"linux-x64" as &[u8])
            );
        }

        std::fs::remove_file(&tmp).ok();
    }
}
