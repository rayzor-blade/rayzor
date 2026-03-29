//! WIT (WebAssembly Interface Types) generator.
//!
//! Generates `.wit` interface definitions from @:export annotated Haxe classes.
//! These WIT files describe the Component Model interface, enabling interop
//! with other languages (Rust, Go, Python, C#) via wasmtime bindings.
//!
//! Example output:
//! ```wit
//! package rayzor:vec2;
//!
//! interface vec2 {
//!   /// Create a new Vec2 instance. Returns an opaque handle.
//!   vec2-new: func(x: f64, y: f64) -> s32;
//!   /// Compute the length of the vector.
//!   vec2-length: func(self: s32) -> f64;
//!   /// Compute the dot product with another vector.
//!   vec2-dot: func(self: s32, other: s32) -> f64;
//! }
//!
//! world vec2-world {
//!   export vec2;
//! }
//! ```

use super::wasm_bindgen::ExportedClass;
use crate::ir::IrType;

/// Generate WIT definitions from exported class metadata.
///
/// Produces a single `.wit` file containing interfaces for each exported class
/// and a world that exports all of them.
pub fn generate_wit(package_name: &str, classes: &[ExportedClass]) -> String {
    let mut wit = String::new();

    // Package declaration
    let sanitized_pkg = sanitize_wit_name(package_name);
    wit.push_str(&format!("package rayzor:{};\n\n", sanitized_pkg));

    // Generate an interface for each exported class
    for class in classes {
        generate_interface(&mut wit, class);
        wit.push('\n');
    }

    // Generate world that exports all interfaces
    wit.push_str(&format!("world {}-world {{\n", sanitized_pkg));
    for class in classes {
        let iface_name = sanitize_wit_name(&class.name);
        wit.push_str(&format!("  export {};\n", iface_name));
    }
    wit.push_str("}\n");

    wit
}

/// Generate a WIT interface for a single exported class.
fn generate_interface(wit: &mut String, class: &ExportedClass) {
    let iface_name = sanitize_wit_name(&class.name);
    let class_lower = class.name.to_lowercase();

    wit.push_str(&format!("interface {} {{\n", iface_name));

    // Constructor
    if let Some(ctor) = &class.constructor {
        let params = format_wit_params(&ctor.params, &ctor.param_types);
        wit.push_str(&format!(
            "  /// Create a new {} instance. Returns an opaque handle (pointer).\n",
            class.name
        ));
        wit.push_str(&format!(
            "  {}-new: func({}) -> s32;\n\n",
            class_lower, params
        ));
    }

    // Instance methods
    for method in &class.instance_methods {
        let method_wit_name = sanitize_wit_name(&method.name);
        let mut params = vec![("self".to_string(), "s32".to_string())];
        for (name, ty) in method.params.iter().zip(method.param_types.iter()) {
            params.push((sanitize_wit_name(name), ir_type_to_wit(ty).to_string()));
        }
        let params_str = params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect::<Vec<_>>()
            .join(", ");

        let ret = ir_type_to_wit(&method.return_type);
        if ret == "()" {
            wit.push_str(&format!(
                "  {}-{}: func({});\n",
                class_lower, method_wit_name, params_str
            ));
        } else {
            wit.push_str(&format!(
                "  {}-{}: func({}) -> {};\n",
                class_lower, method_wit_name, params_str, ret
            ));
        }
    }

    // Static methods
    for method in &class.static_methods {
        let method_wit_name = sanitize_wit_name(&method.name);
        let params = format_wit_params(&method.params, &method.param_types);
        let ret = ir_type_to_wit(&method.return_type);
        if ret == "()" {
            wit.push_str(&format!(
                "  {}-{}: func({});\n",
                class_lower, method_wit_name, params
            ));
        } else {
            wit.push_str(&format!(
                "  {}-{}: func({}) -> {};\n",
                class_lower, method_wit_name, params, ret
            ));
        }
    }

    // Malloc/free for constructors
    if class.constructor.is_some() {
        wit.push_str(&format!(
            "\n  /// Allocation size for {} instances (bytes).\n",
            class.name
        ));
        wit.push_str(&format!(
            "  {}-alloc-size: func() -> u32;\n",
            class_lower
        ));
    }

    wit.push_str("}\n");
}

/// Format WIT parameters from names and types.
fn format_wit_params(names: &[String], types: &[IrType]) -> String {
    names
        .iter()
        .zip(types.iter())
        .map(|(name, ty)| format!("{}: {}", sanitize_wit_name(name), ir_type_to_wit(ty)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Convert an IR type to a WIT type string.
fn ir_type_to_wit(ty: &IrType) -> &'static str {
    match ty {
        IrType::Void => "()",
        IrType::Bool => "bool",
        IrType::I8 => "s8",
        IrType::I16 => "s16",
        IrType::I32 => "s32",
        IrType::I64 => "s64",
        IrType::U8 => "u8",
        IrType::U16 => "u16",
        IrType::U32 => "u32",
        IrType::U64 => "u64",
        IrType::F32 => "f32",
        IrType::F64 => "f64",
        IrType::String => "string",
        IrType::Ptr(_) => "s32", // Opaque pointer as i32 handle
        _ => "s32",              // Default to s32 for unknown types
    }
}

/// Sanitize a name for WIT (lowercase kebab-case).
fn sanitize_wit_name(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('-');
            }
            result.push(ch.to_lowercase().next().unwrap());
        } else if ch == '_' {
            result.push('-');
        } else {
            result.push(ch);
        }
    }
    // WIT identifiers can't start with a digit
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert(0, 'x');
    }
    result
}
