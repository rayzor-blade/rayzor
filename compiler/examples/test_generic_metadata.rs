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
/// Test for @:generic metadata extraction in TAST
use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() -> Result<(), String> {
    println!("=== Testing @:generic Metadata Extraction ===\n");

    let haxe_source = r#"
package test;

// Generic class with @:generic metadata - should trigger monomorphization
@:generic
class Container<T> {
    public var value:T;

    public function new(v:T) {
        this.value = v;
    }

    public function get():T {
        return this.value;
    }
}

// Non-generic class - no @:generic metadata
class SimpleClass {
    public var x:Int;

    public function new() {
        this.x = 0;
    }
}

// Final class with @:final
@:final
class FinalContainer<T> {
    public var item:T;

    public function new(i:T) {
        this.item = i;
    }
}

class Main {
    static function main() {
        // Test instantiation
        var intContainer = new Container<Int>(42);
        trace(intContainer.get());

        var stringContainer = new Container<String>("hello");
        trace(stringContainer.get());

        var simple = new SimpleClass();
        trace(simple.x);
    }
}
"#;

    let mut unit = CompilationUnit::new(CompilationConfig::default());

    println!("Loading stdlib...");
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    println!("Adding test file...");
    unit.add_file(haxe_source, "test_generic_metadata.hx")
        .map_err(|e| format!("Failed to add file: {}", e))?;

    println!("Compiling to TAST...");
    unit.lower_to_tast()
        .map_err(|errors| format!("TAST errors: {:?}", errors))?;

    // Check symbol flags for the classes
    println!("\n=== Symbol Flag Analysis ===");

    let symbol_table = &unit.symbol_table;
    let string_interner = &unit.string_interner;

    // Look for Container, SimpleClass, and FinalContainer symbols
    let classes_to_check = ["Container", "SimpleClass", "FinalContainer"];

    for class_name in &classes_to_check {
        let interned_name = string_interner.intern(class_name);

        // Search through all symbols
        for symbol in symbol_table.all_symbols() {
            if symbol.name == interned_name {
                let is_generic = symbol.flags.is_generic();
                let is_final = symbol
                    .flags
                    .contains(compiler::tast::symbols::SymbolFlags::FINAL);

                println!(
                    "Class '{}': @:generic={}, @:final={}",
                    class_name, is_generic, is_final
                );

                // Validation
                match *class_name {
                    "Container" if !is_generic => {
                        return Err(format!("Container should have @:generic flag"));
                    }
                    "SimpleClass" if is_generic => {
                        return Err(format!("SimpleClass should NOT have @:generic flag"));
                    }
                    "FinalContainer" if !is_final => {
                        return Err(format!("FinalContainer should have @:final flag"));
                    }
                    _ => {}
                }
                break;
            }
        }
    }

    println!("\n=== All tests passed! ===");
    Ok(())
}
