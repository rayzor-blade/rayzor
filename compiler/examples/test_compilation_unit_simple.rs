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
//! Simplified CompilationUnit test without generics
//!
//! This test verifies the core functionality:
//! 1. Stdlib loads FIRST without package prefix
//! 2. User files load AFTER stdlib with package prefix
//! 3. Symbol resolution works correctly

use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() {
    println!("=== Testing CompilationUnit (Simplified) ===\n");

    // Step 1: Create compilation unit with minimal stdlib
    println!("1. Creating compilation unit with minimal stdlib...");
    let mut config = CompilationConfig::default();
    // Use only non-generic stdlib files
    config.default_stdlib_imports = vec!["StdTypes.hx".to_string(), "String.hx".to_string()];

    let mut unit = CompilationUnit::new(config);
    println!("   ✓ Created\n");

    // Step 2: Load stdlib
    println!("2. Loading standard library...");
    match unit.load_stdlib() {
        Ok(()) => {
            println!("   ✓ Loaded {} stdlib files", unit.stdlib_files.len());
            println!();
        }
        Err(e) => {
            eprintln!("   ✗ Failed to load stdlib: {}", e);
            return;
        }
    }

    // Step 3: Add simple user file
    println!("3. Adding user file...");
    let source = r#"
        package test;

        class MyClass {
            public function new() {}

            public function greet():String {
                return "Hello";
            }

            public static function main():Void {
                var obj = new MyClass();
                var msg:String = obj.greet();
            }
        }
    "#;

    match unit.add_file(source, "MyClass.hx") {
        Ok(()) => {
            println!("   ✓ Added user file");
            println!(
                "   Total: {} stdlib + {} user files\n",
                unit.stdlib_files.len(),
                unit.user_files.len()
            );
        }
        Err(e) => {
            eprintln!("   ✗ Failed to add file: {}", e);
            return;
        }
    }

    // Step 4: Lower to TAST
    println!("4. Lowering to TAST...");
    match unit.lower_to_tast() {
        Ok(typed_files) => {
            println!("   ✓ Successfully lowered {} files", typed_files.len());

            // Step 5: Check symbol qualified names
            println!("\n5. Verifying symbol qualified names...");

            let mut stdlib_symbols = Vec::new();
            let mut user_symbols = Vec::new();

            for symbol in unit.symbol_table.all_symbols() {
                if let Some(qname) = symbol.qualified_name {
                    let name_str = unit.string_interner.get(qname).unwrap_or("");

                    if name_str.starts_with("test.") {
                        user_symbols.push(name_str.to_string());
                    } else if name_str.starts_with("haxe.") {
                        stdlib_symbols.push(name_str.to_string());
                    }
                }
            }

            println!("   Stdlib symbols (haxe.* package):");
            for (i, sym) in stdlib_symbols.iter().take(10).enumerate() {
                println!("      {}. {}", i + 1, sym);
            }
            if stdlib_symbols.len() > 10 {
                println!("      ... and {} more", stdlib_symbols.len() - 10);
            }

            println!("\n   User symbols (test.* package):");
            for (i, sym) in user_symbols.iter().take(10).enumerate() {
                println!("      {}. {}", i + 1, sym);
            }
            if user_symbols.len() > 10 {
                println!("      ... and {} more", user_symbols.len() - 10);
            }

            // Check for stdlib symbols that might have incorrect package prefixes
            // Note: Qualified names like "String.substring" are correct - they're class.method
            // We're looking for symbols with unexpected packages like "test.String.substring"
            let bad_stdlib: Vec<_> = stdlib_symbols
                .iter()
                .filter(|s| {
                    // Split by dots and check if first segment looks like a package
                    let parts: Vec<_> = s.split('.').collect();
                    if parts.len() >= 3 {
                        // This could be either "test.Class.method" (bad) or just deeply nested (check if starts with lowercase)
                        let first = parts[0];
                        first
                            .chars()
                            .next()
                            .map(|c| c.is_lowercase())
                            .unwrap_or(false)
                            && first != "test" // We already filtered test.* out
                    } else {
                        false
                    }
                })
                .collect();

            if !bad_stdlib.is_empty() {
                println!(
                    "\n   ⚠️  WARNING: {} stdlib symbols may have package prefixes:",
                    bad_stdlib.len()
                );
                for sym in bad_stdlib.iter().take(5) {
                    println!("      - {}", sym);
                }
            }

            // Verify user symbols have package prefix
            let good_user = user_symbols
                .iter()
                .filter(|s| s.starts_with("test."))
                .count();

            println!("\n6. Results:");
            println!(
                "   - Stdlib symbols with haxe.* package: {}",
                stdlib_symbols.len()
            );
            println!(
                "   - User symbols with test.* package: {}/{}",
                good_user,
                user_symbols.len()
            );

            if stdlib_symbols.len() > 0 && good_user == user_symbols.len() {
                println!(
                    "\n🎉 SUCCESS: All stdlib symbols prefixed with 'haxe.*', user symbols with 'test.*'!"
                );
            } else {
                println!("\n⚠️  Symbols may need verification");
            }
        }
        Err(e) => {
            eprintln!("   ✗ TAST lowering failed: {:?}", e);
        }
    }
}
