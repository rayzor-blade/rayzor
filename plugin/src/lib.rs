//! Stable C-shaped ABI for Rayzor plugins.
//!
//! A plugin cdylib `use`s this crate (and nothing else from the
//! rayzor workspace). It calls the host through `extern "C"`
//! functions whose symbols are exported with `#[no_mangle]` by the
//! host binary; the dynamic linker resolves the plugin's externs
//! against the host's exports at `dlopen` time.
//!
//! Native binaries built from this workspace pass
//! `-Wl,-export_dynamic` (or platform equivalent) via
//! `.cargo/config.toml` so the host's `#[no_mangle]` symbols stay
//! visible to dlopen'd plugins.
//!
//! On wasm, plugins are rlib-linked into the host wasm runtime; the
//! same `extern "C"` decls resolve against the host exports at
//! static-link time. Identical surface, two link strategies.
//!
//! # Stability contract
//!
//! Every item in this crate is part of the plugin ABI. Adding a new
//! function is fine. Removing or re-shaping an existing one requires
//! bumping [`ABI_VERSION`] so the host's loader can refuse to load
//! mismatched plugins with a clear error.
//!
//! # Plugin shape
//!
//! A plugin cdylib must export:
//!
//! ```rust,ignore
//! // 1. ABI handshake — host checks this at dlopen and refuses
//! //    to bind any symbols on mismatch.
//! #[no_mangle]
//! pub extern "C" fn rayzor_plugin_abi_version() -> u32 {
//!     rayzor_plugin::ABI_VERSION
//! }
//!
//! // 2. Method descriptor table — declared via the macro below.
//! rayzor_plugin::declare_native_methods! { MY_METHODS; ... }
//!
//! // 3. `plugin_describe` + `plugin_init` exports — the
//! //    `rpkg_entry!` macro covers both for rpkg consumption,
//! //    or hand-roll if needed.
//! ```

// On wasm this crate is statically linked into the rayzor wasm runtime as
// part of a PIC side-module that links ONLY core (no alloc/libc/wasi), so it
// must be no_std there. The NATIVE build (compiler dep + dlopen'd dylib) keeps
// std. The only heap users here — the host-side `RuntimePlugin`/`PluginRegistry`
// abstractions — are gated to non-wasm (they're never used in the side-module),
// so the wasm build needs no `alloc` and pulls in no `core::fmt`.
#![cfg_attr(target_arch = "wasm32", no_std)]
#![allow(clippy::missing_safety_doc)]

// ============================================================================
// ABI version
// ============================================================================

/// Compiled-in ABI version. Plugins call
/// `rayzor_plugin_abi_version` at load time and compare against
/// this; mismatched versions abort with a "rebuild plugin against
/// rayzor ≥ X" message rather than the silent SIGSEGV that
/// signature drift used to produce.
///
/// Bump on any change to:
/// - The signature of a function declared in [`host_abi`]
/// - The numeric value of a constant in [`dtype`]
/// - The layout of a `#[repr(C)]` type exposed here
/// - Removal of any item
///
/// Adding new functions / constants does NOT require a bump — they
/// land as additive surface, and plugins that don't reference them
/// stay forward-compatible.
pub const ABI_VERSION: u32 = 1;

/// Symbol name plugins must export so the host can read their
/// compiled-in ABI version at dlopen time.
pub const ABI_VERSION_SYMBOL: &str = "rayzor_plugin_abi_version";

// ============================================================================
// Opaque handles
// ============================================================================

/// Opaque handle for a `rayzor.ds.Tensor`. Round-trips through the
/// host as an `i64` (raw pointer to the host's `RayzorTensor` struct).
/// Plugins must NOT dereference this directly — the struct layout
/// is private to the host crate. Reach for the accessor functions
/// in [`host_abi`] instead.
pub type TensorHandle = i64;

/// Opaque handle for a `rayzor.ds.QTensor`. Same shape as
/// `TensorHandle`.
pub type QTensorHandle = i64;

// ============================================================================
// dtype tags
// ============================================================================

/// Tensor dtype tags. Mirror of `rayzor_tensors::tensor::DTYPE_*` —
/// the values here are part of the ABI and bumping any one of them
/// requires bumping [`ABI_VERSION`].
pub mod dtype {
    pub const F32: u8 = 0;
    pub const F16: u8 = 1;
    pub const BF16: u8 = 2;
    pub const I32: u8 = 3;
    pub const I8: u8 = 4;
    pub const U8: u8 = 5;
    pub const FP8_E4M3: u8 = 6;
    pub const FP8_E5M2: u8 = 7;
}

// ============================================================================
// Host ABI — symbols the host binary exports for plugins to call
// ============================================================================

/// `extern "C"` declarations for every host symbol a plugin might
/// need to reach. The implementations live in `rayzor-runtime` (or
/// the rayzor binary) and are linked into the host process; the
/// dynamic linker resolves these references at `dlopen` time on
/// native, or at static-link time on wasm.
///
/// Plugins should prefer the safe wrappers below over calling these
/// directly — the wrappers handle the unsafe boundary and null /
/// error checks once.
pub mod host_abi {
    use super::TensorHandle;

    extern "C" {
        // -- Tensor lifecycle ---------------------------------------------

        /// Allocate a zero-initialised tensor with the given shape
        /// and dtype. Takes `*const usize` directly (no i64 cast
        /// required on the plugin side). `dtype` is one of
        /// [`super::dtype`]. Returns 0 on OOM / shape rejection.
        pub fn rayzor_plugin_tensor_alloc_zeros(
            shape_ptr: *const usize,
            ndim: usize,
            dtype: u8,
        ) -> i64;

        // -- Tensor accessors --------------------------------------------
        //
        // These are NEW additions to the host that round-trip a tensor
        // handle without the plugin needing to know the `RayzorTensor`
        // struct layout. Each one is a thin reader for one field.

        /// Returns the raw data pointer of a tensor. Plugin code is
        /// responsible for treating it as the right type (f32 for
        /// dtype=F32, etc.).
        pub fn rayzor_plugin_tensor_data(t: TensorHandle) -> *mut u8;

        /// Returns the dtype tag of a tensor (see [`super::dtype`]).
        pub fn rayzor_plugin_tensor_dtype(t: TensorHandle) -> u8;

        /// Returns the number of dimensions of a tensor.
        pub fn rayzor_plugin_tensor_ndim(t: TensorHandle) -> u32;

        /// Returns the shape pointer (read-only). Caller uses
        /// [`rayzor_plugin_tensor_ndim`] to know how many entries
        /// are valid.
        pub fn rayzor_plugin_tensor_shape(t: TensorHandle) -> *const usize;

        /// Returns 1 if the tensor's strides match row-major
        /// contiguous layout for its current shape, 0 otherwise.
        pub fn rayzor_plugin_tensor_is_contiguous(t: TensorHandle) -> u8;
    }
}

// ============================================================================
// Safe wrappers — idiomatic Rust API for plugin code
// ============================================================================

/// Inspectable view of a tensor handle. Constructed via
/// [`Tensor::from_handle`]; reads stay zero-copy (no data is moved,
/// the underlying buffer is still owned by the host).
pub struct Tensor {
    pub handle: TensorHandle,
}

impl Tensor {
    /// Wrap a raw handle. Returns `None` for the null handle (0).
    /// Plugin code should NOT manufacture handles from arbitrary
    /// integers — only from values returned by host APIs.
    #[inline]
    pub fn from_handle(handle: TensorHandle) -> Option<Self> {
        if handle == 0 {
            None
        } else {
            Some(Self { handle })
        }
    }

    /// Allocate a zero-initialised f32 tensor of the given shape.
    /// Returns `None` on OOM or shape rejection.
    #[inline]
    pub fn alloc_zeros_f32(shape: &[usize]) -> Option<Self> {
        Self::alloc_zeros(shape, dtype::F32)
    }

    /// Allocate a zero-initialised tensor with the given shape and
    /// dtype (see [`dtype`]). Returns `None` on OOM / shape rejection.
    #[inline]
    pub fn alloc_zeros(shape: &[usize], dtype: u8) -> Option<Self> {
        let handle = unsafe {
            host_abi::rayzor_plugin_tensor_alloc_zeros(shape.as_ptr(), shape.len(), dtype)
        };
        Self::from_handle(handle)
    }

    /// Read the raw data pointer. Returns `None` if the tensor's
    /// data pointer is null (shouldn't happen for a live handle).
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only as long as the host
    /// hasn't freed the tensor. Plugin code typically lives within
    /// a single method dispatch, during which the host keeps the
    /// tensor alive — so this is safe in practice for the common
    /// "read tensor in a foreign method body" use case.
    #[inline]
    pub fn data_ptr(&self) -> *mut u8 {
        unsafe { host_abi::rayzor_plugin_tensor_data(self.handle) }
    }

    /// Returns the dtype tag (matches [`dtype`]).
    #[inline]
    pub fn dtype(&self) -> u8 {
        unsafe { host_abi::rayzor_plugin_tensor_dtype(self.handle) }
    }

    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        unsafe { host_abi::rayzor_plugin_tensor_ndim(self.handle) as usize }
    }

    /// Returns the shape as a slice. The slice borrows the host's
    /// shape array; do not retain it past the current method call.
    ///
    /// # Safety
    ///
    /// Same as [`Tensor::data_ptr`] — the shape pointer is valid as
    /// long as the host keeps the tensor alive.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        let ndim = self.ndim();
        let ptr = unsafe { host_abi::rayzor_plugin_tensor_shape(self.handle) };
        if ptr.is_null() || ndim == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(ptr, ndim) }
        }
    }

    /// True if the tensor's strides are row-major contiguous for
    /// its current shape.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        unsafe { host_abi::rayzor_plugin_tensor_is_contiguous(self.handle) != 0 }
    }
}

// ============================================================================
// Runtime plugin trait (existing host-side API — kept as-is)
// ============================================================================

/// Trait for runtime plugins (host-side abstraction). Plugin
/// cdylibs don't implement this — they expose the `plugin_init` /
/// `plugin_describe` C exports and the host wraps them. This
/// trait is the host's internal representation.
///
/// Host-only: never used inside a wasm side-module, and gating it out
/// keeps the no_std wasm build free of `alloc`/`core::fmt`.
#[cfg(not(target_arch = "wasm32"))]
pub trait RuntimePlugin: Send + Sync {
    fn name(&self) -> &str;
    fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)>;
    fn on_load(&self) -> Result<(), String> {
        Ok(())
    }
    fn on_unload(&self) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct PluginRegistry {
    plugins: Vec<Box<dyn RuntimePlugin>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    pub fn register(&mut self, plugin: Box<dyn RuntimePlugin>) -> Result<(), String> {
        let name = plugin.name();
        if self.plugins.iter().any(|p| p.name() == name) {
            return Err(format!("Plugin '{}' is already registered", name));
        }
        plugin.on_load()?;
        self.plugins.push(plugin);
        Ok(())
    }

    pub fn collect_symbols(&self) -> Vec<(&'static str, *const u8)> {
        let mut symbols = Vec::new();
        for plugin in &self.plugins {
            symbols.extend(plugin.runtime_symbols());
        }
        symbols
    }

    pub fn list_plugins(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    pub fn get_plugin(&self, name: &str) -> Option<&dyn RuntimePlugin> {
        self.plugins
            .iter()
            .find(|p| p.name() == name)
            .map(|p| &**p as &dyn RuntimePlugin)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Method descriptor table — cross-dlopen surface
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
    pub return_type: u8,
    pub param_types: [u8; 8],
}

unsafe impl Send for NativeMethodDesc {}
unsafe impl Sync for NativeMethodDesc {}

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
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $i:ident) => {
        9u8
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
    // 9-arg signature: descriptor is fixed [u8; 8] so the 9th tag is
    // dropped; param_count still records 9 for the dispatcher.
    ($a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident, $h:ident, $i:ident) => {
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

#[repr(C)]
pub struct RpkgPluginInfo {
    pub symbols_ptr: *const u8,
    pub symbols_count: usize,
    pub methods_ptr: *const NativeMethodDesc,
    pub methods_count: usize,
}

unsafe impl Send for RpkgPluginInfo {}
unsafe impl Sync for RpkgPluginInfo {}

#[macro_export]
macro_rules! rpkg_entry {
    ($methods:ident, $symbols_fn:path) => {
        #[no_mangle]
        pub unsafe extern "C" fn rayzor_rpkg_entry() -> $crate::RpkgPluginInfo {
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

// ============================================================================
// Convenience macro for the ABI version handshake
// ============================================================================

/// Generate the `rayzor_plugin_abi_version` export. Every plugin
/// cdylib should invoke this exactly once at top level.
#[macro_export]
macro_rules! export_abi_version {
    () => {
        #[no_mangle]
        pub extern "C" fn rayzor_plugin_abi_version() -> u32 {
            $crate::ABI_VERSION
        }
    };
}
