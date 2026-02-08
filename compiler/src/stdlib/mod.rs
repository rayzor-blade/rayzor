//! Standard Library Implementation using MIR Builder
//!
//! This module provides the Haxe standard library built directly in MIR
//! without parsing source files. This approach provides:
//!
//! - **Fast compilation**: No parsing overhead
//! - **Type-safe**: Built using MIR builder API
//! - **Extern support**: Native runtime functions
//! - **Version control**: Stdlib is Rust code
//! - **Easy maintenance**: No complex Haxe parsing needed
//!
//! # Architecture
//!
//! The stdlib is organized into modules:
//! - `string` - String type with UTF-8 support and extern methods
//! - `array` - Array<T> dynamic array with extern methods
//! - `stdtypes` - Int, Float, Bool with extern methods
//! - `math` - Mathematical functions
//! - `sys` - System interactions
//!
//! # Extern Functions
//!
//! Extern functions are declared in MIR but implemented in the runtime.
//! The runtime provides native implementations for:
//! - String operations (concat, substring, indexOf)
//! - Array operations (push, pop, slice, sort)
//! - Type conversions (toString, parseInt)
//! - I/O operations (print, trace)

pub mod array;
pub mod ereg;
pub mod memory;
pub mod runtime_mapping;
pub mod stdtypes;
pub mod string;
pub mod vec;
pub mod vec_u8;

// Rayzor concurrent primitives
pub mod channel;
pub mod sync;
pub mod thread;

// Rayzor systems-level types (Box, Ptr, Ref, Usize)
pub mod systems;

// Rayzor data science types (Tensor)
pub mod tensor;

// Hashlink compatibility
pub mod hdll_plugin;
pub mod hl_types;

use crate::compiler_plugin::CompilerPluginRegistry;
use crate::ir::{mir_builder::MirBuilder, IrModule};

// Re-export runtime mapping types
pub use runtime_mapping::{
    FunctionSource, IrTypeDescriptor, MethodSignature, RuntimeFunctionCall, StdlibMapping,
};

// Re-export Hashlink compatibility types
pub use hdll_plugin::{HdllError, HdllManifest, HdllPlugin};
pub use hl_types::HlTypeKind;

/// Build the complete standard library as an MIR module
///
/// This creates all standard library types and functions using the MIR builder.
/// The stdlib includes:
/// - Memory management (malloc, realloc, free)
/// - String with extern methods
/// - Array<T> with extern methods
/// - Standard types (Int, Float, Bool)
/// - Built-in functions (trace, print)
///
/// # Returns
///
/// An IrModule containing the complete standard library
pub fn build_stdlib() -> IrModule {
    let mut builder = MirBuilder::new("haxe");

    // Memory management functions
    memory::build_memory_functions(&mut builder);

    // Build Vec<u8> type and methods
    vec_u8::build_vec_u8_type(&mut builder);

    // Build String type and methods
    string::build_string_type(&mut builder);

    // Build Array<T> type and methods
    array::build_array_type(&mut builder);

    // Build standard types and conversions
    stdtypes::build_std_types(&mut builder);

    // Build concurrent primitives
    thread::build_thread_type(&mut builder);
    channel::build_channel_type(&mut builder);
    sync::build_sync_types(&mut builder);

    // Build systems-level types (Box, Ptr, Ref, Usize)
    systems::build_systems_types(&mut builder);

    // Build tensor types (rayzor.ds.Tensor)
    tensor::build_tensor_types(&mut builder);

    // Build EReg (regular expressions)
    ereg::build_ereg_type(&mut builder);

    // Build Vec<T> extern declarations (monomorphized specializations)
    vec::build_vec_externs(&mut builder);

    builder.finish()
}

/// Build the standard library with additional plugin-provided functions.
///
/// This extends `build_stdlib()` by also declaring extern functions and
/// building MIR wrappers from all registered compiler plugins (including
/// HDLL plugins loaded from `@:hlNative` classes).
pub fn build_stdlib_with_plugins(registry: &CompilerPluginRegistry) -> IrModule {
    let mut builder = MirBuilder::new("haxe");

    // Built-in stdlib modules (same as build_stdlib)
    memory::build_memory_functions(&mut builder);
    vec_u8::build_vec_u8_type(&mut builder);
    string::build_string_type(&mut builder);
    array::build_array_type(&mut builder);
    stdtypes::build_std_types(&mut builder);
    thread::build_thread_type(&mut builder);
    channel::build_channel_type(&mut builder);
    sync::build_sync_types(&mut builder);
    systems::build_systems_types(&mut builder);
    tensor::build_tensor_types(&mut builder);
    ereg::build_ereg_type(&mut builder);
    vec::build_vec_externs(&mut builder);

    // Plugin-provided extern declarations and MIR wrappers
    registry.declare_all_externs(&mut builder);
    registry.build_all_mir_wrappers(&mut builder);

    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_builds() {
        let stdlib = build_stdlib();

        // Should have functions
        assert!(!stdlib.functions.is_empty(), "Stdlib should have functions");

        // Module should be named "haxe"
        assert_eq!(stdlib.name, "haxe");
    }

    #[test]
    fn test_stdlib_has_string_functions() {
        let stdlib = build_stdlib();

        // Should have string functions
        let has_string_concat = stdlib
            .functions
            .iter()
            .any(|(_, f)| f.name.contains("string"));

        assert!(has_string_concat, "Should have string functions");
    }
}
