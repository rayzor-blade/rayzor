//! Plugin system for runtime function registration
//!
//! This module provides a trait-based plugin architecture that allows to register runtime
//! functions with the compiler backend.
//!
//! # Native Plugin Registration
//!
//! External packages (like rayzor-gpu) can declare their methods using the
//! [`declare_native_methods!`] macro, which generates a C-compatible descriptor
//! table that the compiler reads at load time — **no compiler core changes needed**.
//!
//! ```rust,ignore
//! use rayzor_plugin::declare_native_methods;
//!
//! declare_native_methods! {
//!     GPU_METHODS;
//!     // class,               method,          kind,     symbol,                                params      => return
//!     "rayzor_gpu_GPUCompute", "create",        static,  "rayzor_gpu_compute_create",            []          => Ptr;
//!     "rayzor_gpu_GPUCompute", "isAvailable",   static,  "rayzor_gpu_compute_is_available",      []          => I64;
//!     "rayzor_gpu_GPUCompute", "destroy",       instance,"rayzor_gpu_compute_destroy",           [Ptr]       => Void;
//!     "rayzor_gpu_GPUCompute", "createBuffer",  instance,"rayzor_gpu_compute_create_buffer",     [Ptr, Ptr]  => Ptr;
//! }
//! ```

/// Trait for runtime plugins
///
/// Implement this trait to provide
/// runtime functions that can be called from compiled code.
pub trait RuntimePlugin: Send + Sync {
    /// Returns the name of this plugin (e.g., "haxe", "stdlib")
    fn name(&self) -> &str;

    /// Returns the runtime symbols this plugin provides
    ///
    /// Each symbol is a tuple of (symbol_name, function_pointer).
    /// The function pointer must point to a valid function with C calling convention.
    fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)>;

    /// Called when the plugin is loaded (optional)
    fn on_load(&self) -> Result<(), String> {
        Ok(())
    }

    /// Called when the plugin is unloaded (optional)
    fn on_unload(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Registry for managing runtime plugins
pub struct PluginRegistry {
    plugins: Vec<Box<dyn RuntimePlugin>>,
}

impl PluginRegistry {
    /// Create a new empty plugin registry
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Register a runtime plugin
    pub fn register(&mut self, plugin: Box<dyn RuntimePlugin>) -> Result<(), String> {
        let name = plugin.name();

        // Check for duplicate registrations
        if self.plugins.iter().any(|p| p.name() == name) {
            return Err(format!("Plugin '{}' is already registered", name));
        }

        // Call the plugin's load hook
        plugin.on_load()?;

        self.plugins.push(plugin);
        Ok(())
    }

    /// Get all runtime symbols from all registered plugins
    pub fn collect_symbols(&self) -> Vec<(&'static str, *const u8)> {
        let mut symbols = Vec::new();
        for plugin in &self.plugins {
            symbols.extend(plugin.runtime_symbols());
        }
        symbols
    }

    /// List all registered plugin names
    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// Get a specific plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<&dyn RuntimePlugin> {
        self.plugins
            .iter()
            .find(|p| p.name() == name)
            .map(|p| &**p as &dyn RuntimePlugin)
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Native Plugin Method Descriptor (crosses dlopen boundary)
// ============================================================================

/// Type tags for native plugin method descriptors.
pub mod native_type {
    pub const VOID: u8 = 0;
    pub const I64: u8 = 1;
    pub const F64: u8 = 2;
    pub const PTR: u8 = 3;
    pub const BOOL: u8 = 4;
}

/// Describes a single method exported by a native plugin.
///
/// This struct is `#[repr(C)]` so it can safely cross dlopen boundaries.
/// Strings are represented as `(*const u8, usize)` pairs pointing to
/// static string data in the plugin's binary.
#[repr(C)]
pub struct NativeMethodDesc {
    pub symbol_name: *const u8,
    pub symbol_name_len: usize,
    pub class_name: *const u8,
    pub class_name_len: usize,
    pub method_name: *const u8,
    pub method_name_len: usize,
    /// 1 = static method, 0 = instance method (self is first param)
    pub is_static: u8,
    /// Total parameter count INCLUDING self for instance methods
    pub param_count: u8,
    /// Return type tag (native_type::*)
    pub return_type: u8,
    /// Parameter type tags (native_type::*), first `param_count` entries valid
    pub param_types: [u8; 8],
}

// SAFETY: NativeMethodDesc contains only raw pointers to static data and
// plain integer fields. Safe to share across threads.
unsafe impl Send for NativeMethodDesc {}
unsafe impl Sync for NativeMethodDesc {}

/// Declare a static table of native method descriptors for plugin registration.
///
/// Generates a `static` array of [`NativeMethodDesc`] that the compiler reads
/// at plugin load time to auto-register method mappings and extern declarations.
///
/// # Syntax
///
/// ```rust,ignore
/// declare_native_methods! {
///     TABLE_NAME;
///     // class,    method,   static|instance, symbol,  [ParamTypes...] => ReturnType;
///     "ClassName", "method", static,  "c_symbol_name", []          => Ptr;
///     "ClassName", "method", instance,"c_symbol_name", [Ptr, I64]  => Void;
/// }
/// ```
///
/// **Type tokens**: `Void`, `I64`, `F64`, `Ptr`, `Bool`
///
/// For instance methods, the param list includes `self` (always `Ptr`).
/// For static methods, the param list is only explicit arguments.
#[macro_export]
macro_rules! declare_native_methods {
    (
        $name:ident;
        $($class:literal, $method:literal, $kind:ident, $symbol:literal,
          [$($ptype:ident),*] => $rtype:ident;)*
    ) => {
        static $name: &[$crate::NativeMethodDesc] = &[
            $(
                $crate::NativeMethodDesc {
                    symbol_name: $symbol.as_ptr(),
                    symbol_name_len: $symbol.len(),
                    class_name: $class.as_ptr(),
                    class_name_len: $class.len(),
                    method_name: $method.as_ptr(),
                    method_name_len: $method.len(),
                    is_static: $crate::_is_static!($kind),
                    param_count: $crate::_count_params!($($ptype),*),
                    return_type: $crate::_nt!($rtype),
                    param_types: $crate::_param_array!($($ptype),*),
                },
            )*
        ];
    };
}

// ---------------------------------------------------------------------------
// Internal helper macros (exported for cross-crate macro use)
// ---------------------------------------------------------------------------

#[doc(hidden)]
#[macro_export]
macro_rules! _is_static {
    (static) => {
        1u8
    };
    (instance) => {
        0u8
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _nt {
    (Void) => {
        0u8
    };
    (I64) => {
        1u8
    };
    (F64) => {
        2u8
    };
    (Ptr) => {
        3u8
    };
    (Bool) => {
        4u8
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _count_params {
    () => {
        0u8
    };
    ($a:ident) => {
        1u8
    };
    ($a:ident, $b:ident) => {
        2u8
    };
    ($a:ident, $b:ident, $c:ident) => {
        3u8
    };
    ($a:ident, $b:ident, $c:ident, $d:ident) => {
        4u8
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        5u8
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        6u8
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        7u8
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident) => {
        8u8
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! _param_array {
    () => {
        [0u8; 8]
    };
    ($a:ident) => {
        [$crate::_nt!($a), 0, 0, 0, 0, 0, 0, 0]
    };
    ($a:ident, $b:ident) => {
        [$crate::_nt!($a), $crate::_nt!($b), 0, 0, 0, 0, 0, 0]
    };
    ($a:ident, $b:ident, $c:ident) => {
        [
            $crate::_nt!($a),
            $crate::_nt!($b),
            $crate::_nt!($c),
            0,
            0,
            0,
            0,
            0,
        ]
    };
    ($a:ident, $b:ident, $c:ident, $d:ident) => {
        [
            $crate::_nt!($a),
            $crate::_nt!($b),
            $crate::_nt!($c),
            $crate::_nt!($d),
            0,
            0,
            0,
            0,
        ]
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        [
            $crate::_nt!($a),
            $crate::_nt!($b),
            $crate::_nt!($c),
            $crate::_nt!($d),
            $crate::_nt!($e),
            0,
            0,
            0,
        ]
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        [
            $crate::_nt!($a),
            $crate::_nt!($b),
            $crate::_nt!($c),
            $crate::_nt!($d),
            $crate::_nt!($e),
            $crate::_nt!($f),
            0,
            0,
        ]
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        [
            $crate::_nt!($a),
            $crate::_nt!($b),
            $crate::_nt!($c),
            $crate::_nt!($d),
            $crate::_nt!($e),
            $crate::_nt!($f),
            $crate::_nt!($g),
            0,
        ]
    };
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident) => {
        [
            $crate::_nt!($a),
            $crate::_nt!($b),
            $crate::_nt!($c),
            $crate::_nt!($d),
            $crate::_nt!($e),
            $crate::_nt!($f),
            $crate::_nt!($g),
            $crate::_nt!($h),
        ]
    };
}

// ============================================================================
// Universal rpkg entry point
// ============================================================================

/// C-compatible plugin descriptor returned by `rayzor_rpkg_entry`.
/// Single entry point — no more guessing function names.
#[repr(C)]
pub struct RpkgPluginInfo {
    /// Runtime symbols: array of (name_ptr, name_len, fn_ptr) triples
    pub symbols_ptr: *const u8,
    pub symbols_count: usize,
    /// Method descriptors for compiler registration
    pub methods_ptr: *const NativeMethodDesc,
    pub methods_count: usize,
}

unsafe impl Send for RpkgPluginInfo {}
unsafe impl Sync for RpkgPluginInfo {}

/// Generate the universal `rayzor_rpkg_entry` export for an rpkg dylib.
///
/// Takes a method table name (from `declare_native_methods!`) and a
/// function that returns `Vec<(&'static str, *const u8)>` for runtime symbols.
///
/// ```rust,ignore
/// declare_native_methods! { MY_METHODS; ... }
/// fn get_symbols() -> Vec<(&'static str, *const u8)> { vec![...] }
/// rpkg_entry!(MY_METHODS, get_symbols);
/// ```
#[macro_export]
macro_rules! rpkg_entry {
    ($methods:ident, $symbols_fn:path) => {
        #[no_mangle]
        pub unsafe extern "C" fn rayzor_rpkg_entry() -> $crate::RpkgPluginInfo {
            // Build symbol entries as (name_ptr, name_len, fn_ptr) triples
            let symbols = $symbols_fn();
            let entries: Vec<(usize, usize, usize)> = symbols
                .iter()
                .map(|(name, ptr)| (name.as_ptr() as usize, name.len(), *ptr as usize))
                .collect();
            let count = entries.len();
            let leaked = Box::leak(entries.into_boxed_slice());

            $crate::RpkgPluginInfo {
                symbols_ptr: leaked.as_ptr() as *const u8,
                symbols_count: count,
                methods_ptr: $methods.as_ptr(),
                methods_count: $methods.len(),
            }
        }
    };
}
