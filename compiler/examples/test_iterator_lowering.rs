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
//! Test for-in loop and iterator lowering
//!
//! Tests:
//! 1. Range operator: for (i in 0...n)
//! 2. Generic iterator pattern (currently returns error)
//!
//! This test uses the pipeline directly to avoid stdlib compilation overhead

use compiler::pipeline::{CompilationResult, compile_haxe_file};

fn main() {
    println!("=== Iterator/For-In Loop Lowering Tests ===\n");

    // Test 1: Range operator desugaring
    test_range_operator();

    // Test 2: Generic iterator (should error for now)
    test_generic_iterator();

    // Test 3: Range in function
    test_range_in_function();

    // Test 4: Nested range loops
    test_nested_ranges();

    println!("\n=== All tests completed ===");
}

fn test_range_operator() {
    println!("TEST: Range operator for-in loop");
    println!("for (i in 0...10) should desugar to while loop\n");

    let source = r#"
class RangeTest {
    public static function main() {
        var sum = 0;
        for (i in 0...10) {
            sum = sum + i;
        }
        return sum;
    }
}
"#;

    match compile_and_check(source, "RangeTest.hx") {
        Ok(()) => println!("✅ Range operator test PASSED\n"),
        Err(e) => println!("❌ Range operator test FAILED: {:?}\n", e),
    }
}

fn test_generic_iterator() {
    println!("TEST: Generic iterator (array)");
    println!("for (x in array) - should return error for now\n");

    let source = r#"
class ArrayIterTest {
    public static function main() {
        var arr = [1, 2, 3];
        var sum = 0;
        for (x in arr) {
            sum = sum + x;
        }
        return sum;
    }
}
"#;

    match compile_and_check(source, "ArrayIterTest.hx") {
        Ok(()) => println!("✅ Generic iterator test PASSED (unexpected!)\n"),
        Err(e) => {
            if e.contains("Generic for-in iterator") {
                println!("✅ Generic iterator correctly returns IncompleteImplementation error\n");
            } else {
                println!(
                    "❌ Generic iterator test FAILED with unexpected error: {}\n",
                    e
                );
            }
        }
    }
}

fn test_range_in_function() {
    println!("TEST: Range in function with variable end");
    println!("for (i in 0...n) where n is a variable\n");

    let source = r#"
class RangeFuncTest {
    public static function sumTo(n: Int): Int {
        var sum = 0;
        for (i in 0...n) {
            sum = sum + i;
        }
        return sum;
    }

    public static function main() {
        return sumTo(100);
    }
}
"#;

    match compile_and_check(source, "RangeFuncTest.hx") {
        Ok(()) => println!("✅ Range in function test PASSED\n"),
        Err(e) => println!("❌ Range in function test FAILED: {:?}\n", e),
    }
}

fn test_nested_ranges() {
    println!("TEST: Nested range loops");
    println!("for (i in 0...n) {{ for (j in 0...m) {{ ... }} }}\n");

    let source = r#"
class NestedRangeTest {
    public static function main() {
        var sum = 0;
        for (i in 0...5) {
            for (j in 0...5) {
                sum = sum + i * j;
            }
        }
        return sum;
    }
}
"#;

    match compile_and_check(source, "NestedRangeTest.hx") {
        Ok(()) => println!("✅ Nested ranges test PASSED\n"),
        Err(e) => println!("❌ Nested ranges test FAILED: {:?}\n", e),
    }
}

fn compile_and_check(source: &str, filename: &str) -> Result<(), String> {
    // Compile the source directly without stdlib
    let result: CompilationResult = compile_haxe_file(filename, source);

    if result.errors.is_empty() {
        println!("  Compilation succeeded!");
        println!("  Generated {} MIR module(s)", result.mir_modules.len());

        // Print some info about the generated MIR
        for module in &result.mir_modules {
            println!(
                "  Module '{}' has {} function(s)",
                module.name,
                module.functions.len()
            );
        }

        Ok(())
    } else {
        let error_messages: Vec<String> = result.errors.iter().map(|e| e.message.clone()).collect();
        Err(error_messages.join("; "))
    }
}
