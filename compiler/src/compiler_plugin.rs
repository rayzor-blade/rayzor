//! Compiler Plugin Architecture
//!
//! This module provides a trait-based plugin system for extending the compiler's
//! stdlib method mapping system. This is separate from the `rayzor_plugin` crate
//! which handles JIT runtime symbol registration.
//!
//! **Compiler plugins** (this module):
//! - Register method mappings (Haxe method -> runtime function)
//! - Declare extern function signatures in MIR
//! - Build MIR wrapper functions
//!
//! **Runtime plugins** (`rayzor_plugin` crate):
//! - Provide function pointers for JIT linking
//! - Handle symbol resolution at runtime
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────┐
//! │ CompilerCompilerPluginRegistry│ ── manages compiler-level plugins
//! └──────────┬───────────┘
//!            │
//!     ┌──────┴──────┐
//!     │             │
//! ┌───▼────┐  ┌────▼────┐
//! │ Builtin │  │  HDLL   │ ── implements CompilerPlugin trait
//! │ Plugin  │  │ Plugin  │
//! └─────────┘  └─────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use compiler::compiler_plugin::{CompilerPlugin, CompilerCompilerPluginRegistry};
//!
//! // Create a registry with plugins
//! let mut registry = CompilerCompilerPluginRegistry::new();
//! registry.register(Box::new(BuiltinPlugin));
//!
//! // Optionally add external plugins (e.g., HDLL)
//! // registry.register(Box::new(HdllPlugin::load("ssl.hdll")?));
//!
//! // Build combined stdlib mapping from all plugins
//! let mapping = registry.build_combined_mapping();
//! ```

use crate::ir::mir_builder::MirBuilder;
use crate::stdlib::{array, channel, memory, stdtypes, string, sync, thread, vec, vec_u8, map_iter};
use crate::stdlib::{MethodSignature, RuntimeFunctionCall, StdlibMapping};

/// Trait for compiler plugins that provide stdlib method mappings.
///
/// Plugins can be:
/// - **BuiltinPlugin**: The default rayzor stdlib (Rust runtime)
/// - **HdllPlugin**: Hashlink dynamic libraries (.hdll files)
/// - **Custom**: User-defined plugins for specialized libraries
///
/// # Lifecycle
///
/// 1. Plugin is registered with `CompilerPluginRegistry`
/// 2. During compilation, `method_mappings()` provides Haxe → runtime function mapping
/// 3. During MIR building, `declare_externs()` registers extern function signatures
/// 4. Optionally, `build_mir_wrappers()` creates MIR wrapper functions
pub trait CompilerPlugin: Send + Sync {
    /// Returns the plugin name for debugging and identification
    fn name(&self) -> &str;

    /// Returns method mappings from Haxe stdlib methods to runtime functions.
    ///
    /// These mappings tell the compiler how to translate Haxe method calls
    /// (like `str.charAt(0)`) to runtime function calls (like `haxe_string_char_at`).
    fn method_mappings(&self) -> Vec<(MethodSignature, RuntimeFunctionCall)>;

    /// Declare extern function signatures in the MIR builder.
    ///
    /// This is called during stdlib MIR construction to register extern functions
    /// that will be linked at JIT compilation time.
    fn declare_externs(&self, builder: &mut MirBuilder);

    /// Build MIR wrapper functions that forward to extern implementations.
    ///
    /// MIR wrappers are useful when the Haxe calling convention differs from
    /// the C calling convention of the runtime function.
    fn build_mir_wrappers(&self, builder: &mut MirBuilder);

    /// Returns the priority of this plugin (higher = loaded later, can override).
    ///
    /// Default priority is 0. Built-in plugins should use 0, while user plugins
    /// can use higher values to override built-in mappings.
    fn priority(&self) -> i32 {
        0
    }
}

/// Registry for managing multiple runtime plugins.
///
/// The registry aggregates mappings from all registered plugins and provides
/// a unified view for the compiler.
pub struct CompilerPluginRegistry {
    plugins: Vec<Box<dyn CompilerPlugin>>,
}

impl CompilerPluginRegistry {
    /// Create a new empty plugin registry.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Register a plugin with the registry.
    ///
    /// Plugins are stored in registration order. When building mappings,
    /// later plugins can override earlier ones based on priority.
    pub fn register(&mut self, plugin: Box<dyn CompilerPlugin>) {
        self.plugins.push(plugin);
    }

    /// Build a combined `StdlibMapping` from all registered plugins.
    ///
    /// Mappings are combined in priority order (lower priority first),
    /// so higher-priority plugins can override lower-priority ones.
    pub fn build_combined_mapping(&self) -> StdlibMapping {
        let mut mapping = StdlibMapping::new();

        // Sort plugins by priority (stable sort preserves registration order for equal priorities)
        let mut sorted_plugins: Vec<&Box<dyn CompilerPlugin>> = self.plugins.iter().collect();
        sorted_plugins.sort_by_key(|p| p.priority());

        // Collect all mappings from plugins
        for plugin in &sorted_plugins {
            mapping.register_mappings(plugin.method_mappings());
        }

        mapping
    }

    /// Declare all extern functions from all plugins.
    ///
    /// This should be called during stdlib MIR construction.
    pub fn declare_all_externs(&self, builder: &mut MirBuilder) {
        for plugin in &self.plugins {
            plugin.declare_externs(builder);
        }
    }

    /// Build all MIR wrappers from all plugins.
    ///
    /// This should be called during stdlib MIR construction.
    pub fn build_all_mir_wrappers(&self, builder: &mut MirBuilder) {
        for plugin in &self.plugins {
            plugin.build_mir_wrappers(builder);
        }
    }

    /// Get the names of all registered plugins.
    pub fn plugin_names(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// Get the number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for CompilerPluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// BuiltinPlugin - Default rayzor stdlib plugin
// ============================================================================

/// Built-in plugin providing the rayzor standard library.
///
/// This plugin wraps the existing `StdlibMapping` and stdlib MIR building
/// functions, providing them through the unified `CompilerPlugin` interface.
///
/// # Example
///
/// ```rust,ignore
/// use compiler::compiler_plugin::{CompilerPluginRegistry, BuiltinPlugin};
///
/// let mut registry = CompilerPluginRegistry::new();
/// registry.register(Box::new(BuiltinPlugin::new()));
///
/// let mapping = registry.build_combined_mapping();
/// ```
pub struct BuiltinPlugin {
    /// The standard library method mappings
    mapping: StdlibMapping,
}

impl BuiltinPlugin {
    /// Create a new builtin plugin with all standard library mappings.
    pub fn new() -> Self {
        Self {
            mapping: StdlibMapping::new(),
        }
    }
}

impl Default for BuiltinPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerPlugin for BuiltinPlugin {
    fn name(&self) -> &str {
        "builtin"
    }

    fn method_mappings(&self) -> Vec<(MethodSignature, RuntimeFunctionCall)> {
        self.mapping.all_mappings()
    }

    fn declare_externs(&self, builder: &mut MirBuilder) {
        // Extern declarations are handled by the MIR building functions below.
        // Memory extern functions (malloc, realloc, free)
        memory::build_memory_functions(builder);

        // Vec<u8> externs
        vec_u8::build_vec_u8_type(builder);

        // String externs
        string::build_string_type(builder);

        // Array externs
        array::build_array_type(builder);
        map_iter::build_map_iterators(builder);

        // Standard types externs
        stdtypes::build_std_types(builder);

        // Concurrent primitives externs
        thread::build_thread_type(builder);
        channel::build_channel_type(builder);
        sync::build_sync_types(builder);

        // Vec<T> monomorphized externs
        vec::build_vec_externs(builder);
    }

    fn build_mir_wrappers(&self, _builder: &mut MirBuilder) {
        // MIR wrappers are built as part of declare_externs() by the stdlib modules.
        // The stdlib building functions (like thread::build_thread_type) create both
        // extern declarations AND MIR wrapper functions in a single pass.
        //
        // This method exists for plugins that separate extern declaration from
        // MIR wrapper construction (e.g., HDLL plugins might declare externs first,
        // then build wrappers based on loaded library metadata).
    }

    fn priority(&self) -> i32 {
        // Built-in has lowest priority (0), allowing other plugins to override
        0
    }
}

// ============================================================================
// NativePlugin — auto-generated from NativeMethodDesc (no compiler core changes)
// ============================================================================

use crate::ir::{CallingConvention, IrType};
use crate::stdlib::IrTypeDescriptor;

/// A compiler plugin created dynamically from [`rayzor_plugin::NativeMethodDesc`]
/// descriptors loaded from a native package's cdylib.
///
/// This enables external packages (like rayzor-gpu) to register their methods
/// with the compiler **without modifying compiler source code**. The plugin
/// handles:
/// - Method mappings (Haxe method → extern C function)
/// - Extern function declarations in MIR
/// - No MIR wrappers needed (direct extern calls)
pub struct NativePlugin {
    plugin_name: String,
    methods: Vec<NativeMethodInfo>,
}

/// Parsed method info (owned strings, safe for compiler lifetime).
struct NativeMethodInfo {
    symbol_name: String,
    class_name: String,
    method_name: String,
    is_static: bool,
    param_count: u8,
    return_type: u8,
    param_types: Vec<u8>,
}

impl NativePlugin {
    /// Create a NativePlugin from raw descriptors read from a cdylib.
    ///
    /// Copies all string data from the descriptor pointers into owned Strings,
    /// so the plugin is independent of the dylib's memory after construction.
    ///
    /// # Safety
    ///
    /// The caller must ensure `descs` points to `count` valid `NativeMethodDesc`
    /// structs with valid string pointers.
    pub unsafe fn from_descriptors(
        name: &str,
        descs: *const rayzor_plugin::NativeMethodDesc,
        count: usize,
    ) -> Self {
        let mut methods = Vec::with_capacity(count);
        let slice = std::slice::from_raw_parts(descs, count);

        for desc in slice {
            let symbol_name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                desc.symbol_name,
                desc.symbol_name_len,
            ))
            .to_string();

            let class_name = normalize_class_name(std::str::from_utf8_unchecked(
                std::slice::from_raw_parts(desc.class_name, desc.class_name_len),
            ));

            let method_name = std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                desc.method_name,
                desc.method_name_len,
            ))
            .to_string();

            let param_types = desc.param_types[..desc.param_count as usize].to_vec();

            methods.push(NativeMethodInfo {
                symbol_name,
                class_name,
                method_name,
                is_static: desc.is_static != 0,
                param_count: desc.param_count,
                return_type: desc.return_type,
                param_types,
            });
        }

        NativePlugin {
            plugin_name: name.to_string(),
            methods,
        }
    }

    /// A plugin that provides only runtime symbols and no method descriptors.
    ///
    /// Valid for a plugin whose classes are dispatched by the compiler's own
    /// built-in mapping table rather than plugin-supplied descriptors: it
    /// exposes the kernel symbols (via `plugin_init`) for JIT linkage but has
    /// no method mappings of its own.
    pub fn empty(name: &str) -> Self {
        NativePlugin {
            plugin_name: name.to_string(),
            methods: Vec::new(),
        }
    }

    /// Create a NativePlugin from deserialized method entries (rpkg format).
    ///
    /// This is the safe counterpart to `from_descriptors` — takes owned Rust
    /// data from `MethodDescEntry` instead of raw C pointers.
    pub fn from_method_entries(name: &str, entries: Vec<crate::rpkg::MethodDescEntry>) -> Self {
        let methods = entries
            .into_iter()
            .map(|e| NativeMethodInfo {
                symbol_name: e.symbol_name,
                class_name: normalize_class_name(&e.class_name),
                method_name: e.method_name,
                is_static: e.is_static,
                param_count: e.param_count,
                return_type: e.return_type,
                param_types: e.param_types,
            })
            .collect();

        NativePlugin {
            plugin_name: name.to_string(),
            methods,
        }
    }
}

/// Convert a native_type tag to an IrTypeDescriptor.
fn native_type_to_descriptor(tag: u8) -> IrTypeDescriptor {
    match tag {
        0 => IrTypeDescriptor::Void,
        1 => IrTypeDescriptor::I64,
        2 => IrTypeDescriptor::F64,
        3 => IrTypeDescriptor::PtrVoid,
        4 => IrTypeDescriptor::Bool,
        _ => IrTypeDescriptor::I64, // fallback
    }
}

/// A plugin's class name in the mapping's key form.
///
/// The mapping is keyed by the Haxe dotted name and a class must register
/// under exactly one key, or a lookup through its other spelling finds a
/// different method set. Only the `::` path separator is rewritten: an
/// underscore is a legal character in a class name, so a plugin that means
/// a package path must declare it with separators.
fn normalize_class_name(name: &str) -> String {
    name.replace("::", ".")
}

/// Convert a native_type tag to an IrType.
fn native_type_to_ir(tag: u8) -> IrType {
    native_type_to_descriptor(tag).to_ir_type()
}

impl CompilerPlugin for NativePlugin {
    fn name(&self) -> &str {
        &self.plugin_name
    }

    fn method_mappings(&self) -> Vec<(MethodSignature, RuntimeFunctionCall)> {
        let mut mappings = Vec::with_capacity(self.methods.len());

        for m in &self.methods {
            // Leak strings to get 'static lifetime (same pattern as HdllPlugin)
            let class: &'static str = Box::leak(m.class_name.clone().into_boxed_str());
            let method: &'static str = Box::leak(m.method_name.clone().into_boxed_str());
            let runtime_name: &'static str = Box::leak(m.symbol_name.clone().into_boxed_str());

            // For instance methods: param_count includes self, but MethodSignature
            // wants the count EXCLUDING self.
            let user_param_count = if m.is_static {
                m.param_count as usize
            } else {
                (m.param_count as usize).saturating_sub(1)
            };

            let has_return = m.return_type != 0; // 0 = Void

            // Build param_types descriptor array (includes self for instance methods)
            let param_descs: Vec<IrTypeDescriptor> = m
                .param_types
                .iter()
                .map(|&t| native_type_to_descriptor(t))
                .collect();
            let param_types: &'static [IrTypeDescriptor] =
                Box::leak(param_descs.into_boxed_slice());

            let return_desc = if has_return {
                Some(native_type_to_descriptor(m.return_type))
            } else {
                None
            };

            let sig = MethodSignature {
                class,
                method,
                is_static: m.is_static,
                is_constructor: false,
                param_count: user_param_count,
            };

            let call = RuntimeFunctionCall {
                runtime_name,
                needs_out_param: false,
                has_self_param: !m.is_static,
                param_count: user_param_count,
                has_return,
                params_need_ptr_conversion: 0,
                raw_value_params: 0,
                returns_raw_value: false,
                extend_to_i64_params: 0,
                param_types: Some(param_types),
                return_type: return_desc,
                is_mir_wrapper: false,
                source: crate::stdlib::FunctionSource::ExternC,
            };

            mappings.push((sig, call));
        }

        mappings
    }

    fn declare_externs(&self, builder: &mut MirBuilder) {
        for m in &self.methods {
            let mut fb = builder.begin_function(&m.symbol_name);

            for (i, &ptype) in m.param_types.iter().enumerate() {
                fb = fb.param(&format!("p{}", i), native_type_to_ir(ptype));
            }

            fb = fb.returns(native_type_to_ir(m.return_type));
            fb = fb.calling_convention(CallingConvention::C);

            let func_id = fb.build();
            builder.mark_as_extern(func_id);
        }
    }

    fn build_mir_wrappers(&self, _builder: &mut MirBuilder) {
        // No MIR wrappers needed — the compiler calls externs directly.
        // Instance methods include self as the first parameter in the extern
        // signature, matching what the compiler passes.
    }

    fn priority(&self) -> i32 {
        // Higher than builtin (0), same as HDLL
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin {
        name: String,
        priority: i32,
    }

    impl CompilerPlugin for TestPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn method_mappings(&self) -> Vec<(MethodSignature, RuntimeFunctionCall)> {
            vec![]
        }

        fn declare_externs(&self, _builder: &mut MirBuilder) {
            // No-op for test
        }

        fn build_mir_wrappers(&self, _builder: &mut MirBuilder) {
            // No-op for test
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    #[test]
    fn test_plugin_registry() {
        let mut registry = CompilerPluginRegistry::new();
        assert!(registry.is_empty());

        registry.register(Box::new(TestPlugin {
            name: "test1".to_string(),
            priority: 0,
        }));
        registry.register(Box::new(TestPlugin {
            name: "test2".to_string(),
            priority: 10,
        }));

        assert_eq!(registry.len(), 2);
        assert_eq!(registry.plugin_names(), vec!["test1", "test2"]);
    }

    #[test]
    fn test_builtin_plugin() {
        let plugin = BuiltinPlugin::new();

        assert_eq!(plugin.name(), "builtin");
        assert_eq!(plugin.priority(), 0);

        // BuiltinPlugin should have many mappings from StdlibMapping
        let mappings = plugin.method_mappings();
        assert!(
            !mappings.is_empty(),
            "BuiltinPlugin should have method mappings"
        );

        // Verify some known mappings exist
        let has_string_method = mappings.iter().any(|(sig, _)| sig.class == "String");
        assert!(
            has_string_method,
            "BuiltinPlugin should have String methods"
        );

        let has_array_method = mappings.iter().any(|(sig, _)| sig.class == "Array");
        assert!(has_array_method, "BuiltinPlugin should have Array methods");
    }

    #[test]
    fn test_builtin_plugin_in_registry() {
        let mut registry = CompilerPluginRegistry::new();
        registry.register(Box::new(BuiltinPlugin::new()));

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.plugin_names(), vec!["builtin"]);

        // Build combined mapping should work
        let mapping = registry.build_combined_mapping();
        assert!(
            mapping.is_stdlib_class("String"),
            "Combined mapping should have String class"
        );
        assert!(
            mapping.is_stdlib_class("Array"),
            "Combined mapping should have Array class"
        );
    }
}
