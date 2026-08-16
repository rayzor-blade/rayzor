//! HDLL Plugin - Load Hashlink Dynamic Libraries
//!
//! This module provides support for loading and using Hashlink's HDLL format,
//! enabling rayzor to call functions from the Hashlink ecosystem.
//!
//! HDLL files are standard C shared libraries (`.so`, `.dylib`, `.dll`) that
//! export functions following Hashlink's calling conventions.
//!
//! # Usage
//!
//! ```rust,ignore
//! use compiler::stdlib::hdll_plugin::HdllPlugin;
//! use std::path::Path;
//!
//! // Load from manifest file
//! let plugin = HdllPlugin::load_from_manifest(Path::new("ssl.hdll.json"))?;
//!
//! // Get function pointers for JIT linking
//! let symbols = plugin.get_symbols();
//! ```

use libloading::Library;
use serde::Deserialize;
use std::ffi::CStr;
use std::path::Path;

use super::hl_types::{self, HlTypeKind};
use super::{FunctionSource, IrTypeDescriptor, MethodSignature, RuntimeFunctionCall};
use crate::compiler_plugin::CompilerPlugin;
use crate::ir::mir_builder::MirBuilder;
use crate::ir::{CallingConvention, IrType};

// ============================================================================
// Manifest Structures (JSON format)
// ============================================================================

/// Manifest format for HDLL libraries.
///
/// This JSON format describes the functions exported by an HDLL,
/// allowing rayzor to generate correct call signatures without
/// parsing the binary library format.
///
/// # Example
///
/// ```json
/// {
///     "name": "ssl",
///     "version": "1.0.0",
///     "library": "ssl.hdll",
///     "functions": [
///         {
///             "name": "ssl_new",
///             "haxe_name": "Ssl.create",  // "<fully.qualified.Class>.<method>"
///             "params": [{"name": "conf", "type": "dyn"}],
///             "returns": "dyn"
///         }
///     ]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct HdllManifest {
    /// Plugin/class name (e.g., "ssl", "sqlite")
    pub name: String,

    /// Optional version string
    #[serde(default)]
    pub version: Option<String>,

    /// Library filename (relative to manifest location)
    pub library: String,

    /// Function definitions
    pub functions: Vec<HdllFunctionDef>,
}

/// Function definition in the manifest
#[derive(Debug, Clone, Deserialize)]
pub struct HdllFunctionDef {
    /// C symbol name in the library (e.g., "ssl_new")
    pub name: String,

    /// Haxe-style name in "Class.method" format (e.g., "Ssl.create")
    pub haxe_name: String,

    /// Parameter definitions
    #[serde(default)]
    pub params: Vec<HdllParam>,

    /// Return type as string
    pub returns: String,

    /// Whether this is a static function (default: true)
    #[serde(default = "default_true")]
    pub is_static: bool,
}

fn default_true() -> bool {
    true
}

/// Parameter definition in the manifest
#[derive(Debug, Clone, Deserialize)]
pub struct HdllParam {
    /// Parameter name (for documentation)
    pub name: String,

    /// Type as string (e.g., "i32", "f64", "dyn")
    #[serde(rename = "type")]
    pub param_type: String,
}

// ============================================================================
// Loaded Function Information
// ============================================================================

/// Loaded HDLL function with resolved type information
pub struct HdllFunction {
    /// C symbol name
    pub symbol_name: String,

    /// Haxe class name
    pub class_name: String,

    /// Haxe method name
    pub method_name: String,

    /// Whether this is a static method
    pub is_static: bool,

    /// Parameter types (converted to IrTypeDescriptor)
    pub param_types: Vec<IrTypeDescriptor>,

    /// Return type
    pub return_type: IrTypeDescriptor,

    /// Function pointer from dynamic library
    pub fn_ptr: *const (),
}

// ============================================================================
// HDLL Plugin
// ============================================================================

/// Plugin for loading Hashlink HDLL dynamic libraries.
///
/// This plugin implements `CompilerPlugin` to integrate HDLL functions
/// into the rayzor compilation pipeline.
pub struct HdllPlugin {
    /// Plugin name (from manifest)
    name: String,

    /// Loaded dynamic library (kept alive for function pointers)
    #[allow(dead_code)]
    library: Library,

    /// Resolved function information
    functions: Vec<HdllFunction>,
}

// Safety: Library is loaded once and kept alive. Function pointers remain valid
// for the lifetime of the plugin. Plugin is typically long-lived.
unsafe impl Send for HdllPlugin {}
unsafe impl Sync for HdllPlugin {}

impl HdllPlugin {
    /// Load an HDLL plugin from a manifest file.
    ///
    /// The manifest file should be a JSON file describing the library
    /// and its exported functions. The library file is expected to be
    /// in the same directory as the manifest.
    ///
    /// # Errors
    ///
    /// Returns `HdllError` if:
    /// - The manifest file cannot be read
    /// - The manifest JSON is malformed
    /// - The library cannot be loaded
    /// - A function symbol cannot be found
    pub fn load_from_manifest(manifest_path: &Path) -> Result<Self, HdllError> {
        let manifest_content = std::fs::read_to_string(manifest_path).map_err(|e| {
            HdllError::IoError(format!("Failed to read {}: {}", manifest_path.display(), e))
        })?;

        let manifest: HdllManifest = serde_json::from_str(&manifest_content).map_err(|e| {
            HdllError::ManifestError(format!(
                "Invalid JSON in {}: {}",
                manifest_path.display(),
                e
            ))
        })?;

        let lib_path = manifest_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(&manifest.library);

        Self::load_with_manifest(&lib_path, manifest)
    }

    /// Load an HDLL with an explicit manifest.
    ///
    /// This allows loading a library from any path with a programmatically
    /// constructed manifest.
    pub fn load_with_manifest(lib_path: &Path, manifest: HdllManifest) -> Result<Self, HdllError> {
        // Load the dynamic library
        let library = unsafe {
            Library::new(lib_path).map_err(|e| {
                HdllError::LoadError(format!("Failed to load {}: {}", lib_path.display(), e))
            })?
        };

        let mut functions = Vec::new();

        for func_def in &manifest.functions {
            // Parse "<fully.qualified.Class>.method": the LAST dot separates
            // the method, so a packaged class ("sys.ssl.Ssl.connect") keeps
            // its package instead of parsing as class "sys".
            let (class_name, method_name) =
                func_def.haxe_name.rsplit_once('.').ok_or_else(|| {
                    HdllError::ManifestError(format!(
                        "Invalid haxe_name format '{}': expected 'Class.method' \
                         with the class fully qualified",
                        func_def.haxe_name
                    ))
                })?;

            // Look up function pointer in the library
            let fn_ptr: *const () = unsafe {
                let symbol: libloading::Symbol<*const ()> =
                    library.get(func_def.name.as_bytes()).map_err(|e| {
                        HdllError::SymbolError(
                            func_def.name.clone(),
                            format!("Symbol not found: {}", e),
                        )
                    })?;
                *symbol
            };

            // Convert parameter types
            let param_types: Vec<IrTypeDescriptor> = func_def
                .params
                .iter()
                .map(|p| {
                    HlTypeKind::from_manifest_str(&p.param_type)
                        .map(|t| t.to_ir_type_descriptor())
                        .unwrap_or_else(|| {
                            log::warn!(
                                "Unknown type '{}' in HDLL manifest, defaulting to PtrVoid",
                                p.param_type
                            );
                            IrTypeDescriptor::PtrVoid
                        })
                })
                .collect();

            // Convert return type
            let return_type = HlTypeKind::from_manifest_str(&func_def.returns)
                .map(|t| t.to_ir_type_descriptor())
                .unwrap_or_else(|| {
                    if func_def.returns.to_lowercase() != "void" {
                        log::warn!(
                            "Unknown return type '{}' in HDLL manifest, defaulting to Void",
                            func_def.returns
                        );
                    }
                    IrTypeDescriptor::Void
                });

            functions.push(HdllFunction {
                symbol_name: func_def.name.clone(),
                class_name: class_name.to_string(),
                method_name: method_name.to_string(),
                is_static: func_def.is_static,
                param_types,
                return_type,
                fn_ptr,
            });
        }

        log::info!(
            "Loaded HDLL plugin '{}' with {} functions",
            manifest.name,
            functions.len()
        );

        Ok(HdllPlugin {
            name: manifest.name,
            library,
            functions,
        })
    }

    /// Load an HDLL plugin by introspecting `hlp_` symbols.
    ///
    /// This is the primary loading method for HDLL libraries. Hashlink HDLL
    /// libraries use the `DEFINE_PRIM` macro to generate `hlp_<name>` symbols
    /// that are self-describing: calling each `hlp_` function returns the
    /// actual function pointer and sets a type signature string.
    ///
    /// # Arguments
    ///
    /// * `lib_path` - Path to the `.hdll` (`.dylib`/`.so`/`.dll`) file
    /// * `lib_name` - Library name from `@:hlNative` (e.g., "simplex")
    /// * `class_name` - Haxe class name (e.g., "SimplexGenerator")
    /// * `methods` - List of `(method_name, is_static)` from the Haxe class
    ///
    /// # How `hlp_` symbols work
    ///
    /// Each `DEFINE_PRIM(_RET, name, _ARGS)` in the C source generates:
    /// ```c
    /// void* hlp_name(const char** sign) {
    ///     *sign = "arg_codes_return_code";  // e.g., "ii_i"
    ///     return (void*)libname_name;       // actual function pointer
    /// }
    /// ```
    pub fn load_with_introspection(
        lib_path: &Path,
        lib_name: &str,
        class_name: &str,
        methods: &[(&str, bool)],
    ) -> Result<Self, HdllError> {
        let library = unsafe {
            Library::new(lib_path).map_err(|e| {
                HdllError::LoadError(format!("Failed to load {}: {}", lib_path.display(), e))
            })?
        };

        let mut functions = Vec::new();

        for (method_name, is_static) in methods {
            let hlp_symbol = format!("hlp_{}", method_name);

            // The hlp_ function takes a pointer-to-pointer for the signature string
            // and returns the actual function pointer.
            type HlpFn = unsafe extern "C" fn(*mut *const std::os::raw::c_char) -> *const ();

            let fn_ptr: *const ();
            let signature: String;

            unsafe {
                let hlp_fn: libloading::Symbol<HlpFn> =
                    library.get(hlp_symbol.as_bytes()).map_err(|e| {
                        HdllError::SymbolError(
                            hlp_symbol.clone(),
                            format!("hlp_ symbol not found (is DEFINE_PRIM used?): {}", e),
                        )
                    })?;

                let mut sign_ptr: *const std::os::raw::c_char = std::ptr::null();
                fn_ptr = hlp_fn(&mut sign_ptr);

                if sign_ptr.is_null() {
                    return Err(HdllError::SymbolError(
                        hlp_symbol,
                        "hlp_ function returned null signature".to_string(),
                    ));
                }

                signature = CStr::from_ptr(sign_ptr)
                    .to_str()
                    .map_err(|e| {
                        HdllError::SymbolError(
                            hlp_symbol.clone(),
                            format!("Invalid UTF-8 in signature: {}", e),
                        )
                    })?
                    .to_string();
            }

            if fn_ptr.is_null() {
                return Err(HdllError::SymbolError(
                    hlp_symbol,
                    "hlp_ function returned null function pointer".to_string(),
                ));
            }

            // Parse the HL signature string (e.g., "ii_i" -> params=[I32,I32], ret=I32)
            let (param_kinds, return_kind) =
                hl_types::parse_hl_signature(&signature).ok_or_else(|| {
                    HdllError::SymbolError(
                        hlp_symbol.clone(),
                        format!("Failed to parse HL signature '{}'", signature),
                    )
                })?;

            let param_types: Vec<IrTypeDescriptor> = param_kinds
                .iter()
                .map(|k| k.to_ir_type_descriptor())
                .collect();
            let return_type = return_kind.to_ir_type_descriptor();

            // The actual C symbol is lib_name + "_" + method_name
            let symbol_name = format!("{}_{}", lib_name, method_name);

            log::debug!(
                "HDLL introspection: {} -> sig='{}', params={:?}, ret={:?}",
                symbol_name,
                signature,
                param_types,
                return_type
            );

            functions.push(HdllFunction {
                symbol_name,
                class_name: class_name.to_string(),
                method_name: method_name.to_string(),
                is_static: *is_static,
                param_types,
                return_type,
                fn_ptr,
            });
        }

        log::info!(
            "Loaded HDLL plugin '{}' via introspection with {} functions from {}",
            lib_name,
            functions.len(),
            lib_path.display()
        );

        Ok(HdllPlugin {
            name: lib_name.to_string(),
            library,
            functions,
        })
    }

    /// Get function pointers for JIT linking.
    ///
    /// Returns a list of (symbol_name, function_pointer) pairs that can
    /// be registered with the JIT compiler for runtime linking.
    pub fn get_symbols(&self) -> Vec<(&str, *const u8)> {
        self.functions
            .iter()
            .map(|f| (f.symbol_name.as_str(), f.fn_ptr as *const u8))
            .collect()
    }

    /// Get the number of functions in this plugin
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Check if a class is provided by this plugin
    pub fn has_class(&self, class_name: &str) -> bool {
        self.functions.iter().any(|f| f.class_name == class_name)
    }
}

impl CompilerPlugin for HdllPlugin {
    fn name(&self) -> &str {
        &self.name
    }

    fn method_mappings(&self) -> Vec<(MethodSignature, RuntimeFunctionCall)> {
        self.functions
            .iter()
            .map(|f| {
                // Create method signature
                // Note: We leak these strings because MethodSignature requires 'static lifetime.
                // This is acceptable for long-lived plugins.
                let sig = MethodSignature {
                    class: Box::leak(f.class_name.clone().into_boxed_str()),
                    method: Box::leak(f.method_name.clone().into_boxed_str()),
                    is_static: f.is_static,
                    is_constructor: f.method_name == "new",
                    param_count: f.param_types.len(),
                };

                // Create runtime function call descriptor
                let call = RuntimeFunctionCall {
                    runtime_name: Box::leak(f.symbol_name.clone().into_boxed_str()),
                    needs_out_param: false,
                    has_self_param: !f.is_static,
                    param_count: f.param_types.len(),
                    has_return: f.return_type != IrTypeDescriptor::Void,
                    params_need_ptr_conversion: 0,
                    raw_value_params: 0,
                    returns_raw_value: false,
                    extend_to_i64_params: 0,
                    param_types: Some(Box::leak(f.param_types.clone().into_boxed_slice())),
                    return_type: Some(f.return_type),
                    is_mir_wrapper: false,
                    source: FunctionSource::Hdll,
                };

                (sig, call)
            })
            .collect()
    }

    fn declare_externs(&self, builder: &mut MirBuilder) {
        for func in &self.functions {
            // Convert IrTypeDescriptor to IrType for MIR builder
            let return_type = func.return_type.to_ir_type();

            // Start building the function
            let mut func_builder = builder.begin_function(&func.symbol_name);

            // Add parameters
            for (i, param_type) in func.param_types.iter().enumerate() {
                let ir_type = param_type.to_ir_type();
                func_builder = func_builder.param(&format!("p{}", i), ir_type);
            }

            // Finish function declaration
            let func_id = func_builder
                .returns(return_type)
                .calling_convention(CallingConvention::C)
                .build();

            builder.mark_as_extern(func_id);
        }
    }

    fn build_mir_wrappers(&self, _builder: &mut MirBuilder) {
        // HDLL functions use C calling convention directly.
        // No MIR wrappers needed - calls go directly to the extern functions.
    }

    fn priority(&self) -> i32 {
        // Higher than builtin (0), allows HDLL to override built-in implementations
        10
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur when loading an HDLL plugin
#[derive(Debug)]
pub enum HdllError {
    /// I/O error (file not found, permission denied, etc.)
    IoError(String),

    /// Library loading error (invalid format, missing dependencies)
    LoadError(String),

    /// Manifest parsing error (invalid JSON, missing fields)
    ManifestError(String),

    /// Symbol resolution error (function not found in library)
    SymbolError(String, String),
}

impl std::fmt::Display for HdllError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HdllError::IoError(e) => write!(f, "HDLL I/O error: {}", e),
            HdllError::LoadError(e) => write!(f, "HDLL load error: {}", e),
            HdllError::ManifestError(e) => write!(f, "HDLL manifest error: {}", e),
            HdllError::SymbolError(sym, e) => write!(f, "HDLL symbol '{}' error: {}", sym, e),
        }
    }
}

impl std::error::Error for HdllError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_parsing() {
        let json = r#"{
            "name": "TestLib",
            "version": "1.0.0",
            "library": "libtest.so",
            "functions": [
                {
                    "name": "test_add",
                    "haxe_name": "TestLib.add",
                    "params": [
                        {"name": "a", "type": "i32"},
                        {"name": "b", "type": "i32"}
                    ],
                    "returns": "i32"
                },
                {
                    "name": "test_greet",
                    "haxe_name": "TestLib.greet",
                    "params": [],
                    "returns": "void"
                }
            ]
        }"#;

        let manifest: HdllManifest = serde_json::from_str(json).expect("Failed to parse manifest");

        assert_eq!(manifest.name, "TestLib");
        assert_eq!(manifest.version, Some("1.0.0".to_string()));
        assert_eq!(manifest.library, "libtest.so");
        assert_eq!(manifest.functions.len(), 2);

        let add_fn = &manifest.functions[0];
        assert_eq!(add_fn.name, "test_add");
        assert_eq!(add_fn.haxe_name, "TestLib.add");
        assert_eq!(add_fn.params.len(), 2);
        assert_eq!(add_fn.returns, "i32");

        let greet_fn = &manifest.functions[1];
        assert_eq!(greet_fn.name, "test_greet");
        assert_eq!(greet_fn.params.len(), 0);
        assert_eq!(greet_fn.returns, "void");
    }

    #[test]
    fn test_manifest_minimal() {
        let json = r#"{
            "name": "Minimal",
            "library": "libminimal.dylib",
            "functions": []
        }"#;

        let manifest: HdllManifest =
            serde_json::from_str(json).expect("Failed to parse minimal manifest");

        assert_eq!(manifest.name, "Minimal");
        assert_eq!(manifest.version, None);
        assert_eq!(manifest.library, "libminimal.dylib");
        assert!(manifest.functions.is_empty());
    }

    #[test]
    fn test_haxe_name_parsing() {
        // The method is everything after the LAST dot, so a packaged class
        // keeps its package.
        assert_eq!("Ssl.connect".rsplit_once('.'), Some(("Ssl", "connect")));
        assert_eq!(
            "sys.ssl.Ssl.connect".rsplit_once('.'),
            Some(("sys.ssl.Ssl", "connect"))
        );

        // Invalid format (no dot)
        assert!("invalid".rsplit_once('.').is_none());
    }

    #[test]
    fn test_param_type_conversion() {
        let params = vec![
            HdllParam {
                name: "a".to_string(),
                param_type: "i32".to_string(),
            },
            HdllParam {
                name: "b".to_string(),
                param_type: "f64".to_string(),
            },
            HdllParam {
                name: "c".to_string(),
                param_type: "dyn".to_string(),
            },
        ];

        let types: Vec<IrTypeDescriptor> = params
            .iter()
            .map(|p| {
                HlTypeKind::from_manifest_str(&p.param_type)
                    .map(|t| t.to_ir_type_descriptor())
                    .unwrap_or(IrTypeDescriptor::PtrVoid)
            })
            .collect();

        assert_eq!(types[0], IrTypeDescriptor::I32);
        assert_eq!(types[1], IrTypeDescriptor::F64);
        assert_eq!(types[2], IrTypeDescriptor::PtrVoid); // Dynamic maps to PtrVoid
    }
}
