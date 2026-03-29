//! RPKG Package Installation — loads an rpkg for use at compile/run time.
//!
//! For pure-Haxe packages: extracts `.hx` sources so they can be imported.
//! For native packages: also extracts the platform dylib to a temp file,
//! dlopens it, reads runtime symbols via `plugin_init()`, and creates a
//! `NativePlugin` from the embedded method table.

use super::{LoadedRpkg, MethodDescEntry, RpkgError};
use crate::compiler_plugin::NativePlugin;
use std::path::Path;

/// A loaded rpkg package ready to register with the compiler.
///
/// Holds the dlopen'd library, runtime symbols, and compiler plugin.
/// The temp file for the native lib is cleaned up on drop.
pub struct RpkgPlugin {
    /// Keep the library alive for the lifetime of the plugin
    _lib: Option<libloading::Library>,
    /// Runtime symbols for JIT linking (name, fn_ptr)
    pub runtime_symbols: Vec<(String, *const u8)>,
    /// Compiler plugin (method mappings + extern declarations)
    pub compiler_plugin: Option<NativePlugin>,
    /// Haxe source files from the package (module_path → source)
    pub haxe_sources: std::collections::HashMap<String, String>,
    /// Package name
    pub package_name: String,
    /// Temp file for extracted native lib (cleaned up on drop)
    temp_lib_path: Option<std::path::PathBuf>,
}

impl Drop for RpkgPlugin {
    fn drop(&mut self) {
        // Drop the library first (before removing the file)
        self._lib.take();
        if let Some(path) = self.temp_lib_path.take() {
            let _ = std::fs::remove_file(&path);
        }
    }
}

impl RpkgPlugin {
    /// Load an rpkg file and prepare it for registration.
    ///
    /// 1. Parse the rpkg archive
    /// 2. Extract native lib to a temp file and dlopen it
    /// 3. Load runtime symbols via `plugin_init()` export
    /// 4. Create `NativePlugin` from the embedded method table
    pub fn load(rpkg_path: &Path) -> Result<Self, String> {
        let loaded = super::load_rpkg(rpkg_path)
            .map_err(|e| format!("failed to load rpkg {}: {}", rpkg_path.display(), e))?;

        Self::from_loaded(loaded)
    }

    /// Create from an already-parsed LoadedRpkg.
    pub fn from_loaded(loaded: LoadedRpkg) -> Result<Self, String> {
        let mut runtime_symbols = Vec::new();
        let mut lib = None;
        let mut temp_lib_path = None;

        // Extract and load native library if present.
        // If no native lib matches the current platform, fall back to WASM component.
        if let Some(lib_bytes) = &loaded.native_lib_bytes {
            let ext = if cfg!(target_os = "macos") {
                "dylib"
            } else if cfg!(target_os = "windows") {
                "dll"
            } else {
                "so"
            };

            let temp_path = std::env::temp_dir().join(format!(
                "rpkg_{}_{}_{}.{}",
                loaded.package_name,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis(),
                ext
            ));

            std::fs::write(&temp_path, lib_bytes)
                .map_err(|e| format!("failed to extract native lib: {}", e))?;

            // dlopen the extracted library
            let library = unsafe { libloading::Library::new(&temp_path) }
                .map_err(|e| format!("failed to load native lib: {}", e))?;

            // Load runtime symbols via plugin_init()
            runtime_symbols = load_runtime_symbols(&library, &loaded.package_name);

            temp_lib_path = Some(temp_path);
            lib = Some(library);
        }

        // WASM Component fallback notice: if no native lib matched but WASM is available
        if lib.is_none() && loaded.wasm_component_bytes.is_some() {
            eprintln!(
                "  rpkg '{}': no native lib for this platform — WASM component available for `rayzor build --target wasm`",
                loaded.package_name,
            );
        }

        // Create compiler plugin from method table
        let compiler_plugin = if !loaded.methods.is_empty() {
            let name = loaded
                .plugin_name
                .as_deref()
                .unwrap_or(&loaded.package_name);
            Some(NativePlugin::from_method_entries(name, loaded.methods))
        } else {
            None
        };

        Ok(RpkgPlugin {
            _lib: lib,
            runtime_symbols,
            compiler_plugin,
            haxe_sources: loaded.haxe_sources,
            package_name: loaded.package_name,
            temp_lib_path,
        })
    }
}

/// Load runtime symbols from a dlopen'd library.
///
/// First tries the universal `rayzor_rpkg_entry` export (preferred).
/// Falls back to legacy `plugin_init` names for backward compatibility.
fn load_runtime_symbols(
    lib: &libloading::Library,
    _package_name: &str,
) -> Vec<(String, *const u8)> {
    // Try universal entry point first
    if let Some(symbols) = load_symbols_via_rpkg_entry(lib) {
        return symbols;
    }

    // Legacy fallback: try old-style plugin_init exports
    type InitFn = unsafe extern "C" fn(*mut usize) -> *const u8;

    let init_names: &[&[u8]] = &[
        b"rayzor_gpu_plugin_init",
        b"rayzor_window_plugin_init",
        b"plugin_init",
        b"rayzor_plugin_init",
    ];

    for name in init_names {
        if let Ok(init_fn) = unsafe { lib.get::<InitFn>(name) } {
            let mut count: usize = 0;
            let entries_ptr = unsafe { init_fn(&mut count) };
            if entries_ptr.is_null() || count == 0 {
                continue;
            }

            let entries = unsafe {
                std::slice::from_raw_parts(entries_ptr as *const (usize, usize, usize), count)
            };

            let mut symbols = Vec::with_capacity(count);
            for &(name_ptr, name_len, fn_ptr) in entries {
                let name = unsafe {
                    std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                        name_ptr as *const u8,
                        name_len,
                    ))
                };
                symbols.push((name.to_string(), fn_ptr as *const u8));
            }

            return symbols;
        }
    }

    Vec::new()
}

/// Load runtime symbols via the universal `rayzor_rpkg_entry` export.
///
/// Returns `Some(symbols)` if the export exists and has valid data.
fn load_symbols_via_rpkg_entry(lib: &libloading::Library) -> Option<Vec<(String, *const u8)>> {
    type EntryFn = unsafe extern "C" fn() -> rayzor_plugin::RpkgPluginInfo;

    let entry_fn = unsafe { lib.get::<EntryFn>(b"rayzor_rpkg_entry") }.ok()?;
    let info = unsafe { entry_fn() };

    if info.symbols_count == 0 || info.symbols_ptr.is_null() {
        return Some(Vec::new());
    }

    let entries = unsafe {
        std::slice::from_raw_parts(
            info.symbols_ptr as *const (usize, usize, usize),
            info.symbols_count,
        )
    };

    let mut symbols = Vec::with_capacity(info.symbols_count);
    for &(name_ptr, name_len, fn_ptr) in entries {
        let name = unsafe {
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                name_ptr as *const u8,
                name_len,
            ))
        };
        symbols.push((name.to_string(), fn_ptr as *const u8));
    }

    Some(symbols)
}
