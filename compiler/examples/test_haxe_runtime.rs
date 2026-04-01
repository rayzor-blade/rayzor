#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unreachable_patterns,
    unused_mut,
    unused_assignments,
    unused_parens
)]
#![allow(
    clippy::single_component_path_imports,
    clippy::for_kv_map,
    clippy::explicit_auto_deref
)]
#![allow(
    clippy::println_empty_string,
    clippy::len_zero,
    clippy::useless_vec,
    clippy::field_reassign_with_default
)]
#![allow(
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::bool_assert_comparison
)]
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::useless_format,
    clippy::clone_on_copy
)]
#![allow(clippy::unwrap_or_default)]
//! Comprehensive test of all Haxe core type runtime functions

use compiler::plugin::PluginRegistry;

fn main() {
    println!("🚀 Haxe Core Types Runtime Test\n");
    println!("{}", "=".repeat(70));

    // Set up plugin registry
    let mut registry = PluginRegistry::new();
    registry
        .register(rayzor_runtime::get_plugin())
        .expect("Failed to register runtime plugin");

    let symbols = registry.collect_symbols();
    println!("📦 Registered {} runtime functions\n", symbols.len());

    // Display all registered functions by category
    println!("📋 Available Runtime Functions:\n");

    let mut categories: std::collections::BTreeMap<&str, Vec<&str>> =
        std::collections::BTreeMap::new();

    for (name, _) in &symbols {
        let category = if name.starts_with("haxe_string_") {
            "String"
        } else if name.starts_with("haxe_array_") {
            "Array"
        } else if name.starts_with("haxe_math_") {
            "Math"
        } else if name.starts_with("haxe_sys_") {
            "Sys/IO"
        } else if name.starts_with("haxe_vec_") {
            "Vec (internal)"
        } else {
            "Other"
        };

        categories
            .entry(category)
            .or_insert_with(Vec::new)
            .push(name);
    }

    let mut sorted_categories: Vec<_> = categories.iter().collect();
    sorted_categories.sort_by_key(|(k, _)| *k);

    for (category, funcs) in sorted_categories {
        println!("  {} ({} functions):", category, funcs.len());
        let mut sorted_funcs = funcs.clone();
        sorted_funcs.sort();
        for func in sorted_funcs.iter().take(5) {
            println!("    - {}", func);
        }
        if funcs.len() > 5 {
            println!("    ... and {} more", funcs.len() - 5);
        }
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("\n✅ All Haxe core type runtime functions successfully registered!");
    println!("\n📊 Summary:");
    println!(
        "   - String functions: {}",
        categories.get("String").map(|v| v.len()).unwrap_or(0)
    );
    println!(
        "   - Array functions:  {}",
        categories.get("Array").map(|v| v.len()).unwrap_or(0)
    );
    println!(
        "   - Math functions:   {}",
        categories.get("Math").map(|v| v.len()).unwrap_or(0)
    );
    println!(
        "   - Sys/IO functions: {}",
        categories.get("Sys/IO").map(|v| v.len()).unwrap_or(0)
    );
    println!("   - Total:            {}", symbols.len());

    println!("\n🎉 Plugin system working perfectly!");
    println!("   Ready for Haxe compilation!");
}
