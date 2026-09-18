//! Demo of integrated context-based error reporting
//!
//! This example shows how the enhanced diagnostic system integrates with
//! the parser's context errors to provide rich, actionable error messages.

use parser::{ErrorFormatter, parse_incrementally_enhanced};

fn main() {
    println!("🔧 Context Integration Demo");
    println!("============================\n");

    // Test 1: Missing semicolon (from parser context)
    demo_missing_semicolon();

    // Test 2: Missing braces (from parser context)
    demo_missing_braces();

    // Test 3: Invalid function declaration (from parser context)
    demo_invalid_function();

    // Test 4: Multiple context errors in one file
    demo_multiple_context_errors();
}

fn demo_missing_semicolon() {
    println!("📋 Demo 1: Missing Semicolon (Parser Context)");
    println!("----------------------------------------------\n");

    let test_code = r#"
class Test {
    var x = 1
    function test() {
        return x;
    }
}
"#;

    let result = parse_incrementally_enhanced("test.hx", test_code);

    if result.has_errors() {
        let formatter = ErrorFormatter::with_colors();
        println!("❌ Parse failed with enhanced diagnostics:");
        println!(
            "{}",
            formatter.format_diagnostics(&result.diagnostics, &result.source_map)
        );
    } else {
        println!("✅ Parsed successfully");
    }

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}

fn demo_missing_braces() {
    println!("📋 Demo 2: Missing Braces (Parser Context)");
    println!("-------------------------------------------\n");

    let test_code = r#"
class Test {
    function test() 
        return "hello";
    }
}
"#;

    let result = parse_incrementally_enhanced("test.hx", test_code);

    if result.has_errors() {
        let formatter = ErrorFormatter::with_colors();
        println!("❌ Parse failed with enhanced diagnostics:");
        println!(
            "{}",
            formatter.format_diagnostics(&result.diagnostics, &result.source_map)
        );
    } else {
        println!("✅ Parsed successfully");
    }

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}

fn demo_invalid_function() {
    println!("📋 Demo 3: Invalid Function Declaration (Parser Context)");
    println!("---------------------------------------------------------\n");

    let test_code = r#"
class Test {
    fucntion test() {
        var x = 1
        return x;
    }
}
"#;

    let result = parse_incrementally_enhanced("test.hx", test_code);

    if result.has_errors() {
        let formatter = ErrorFormatter::with_colors();
        println!("❌ Parse failed with enhanced diagnostics:");
        println!(
            "{}",
            formatter.format_diagnostics(&result.diagnostics, &result.source_map)
        );
    } else {
        println!("✅ Parsed successfully");
    }

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}

fn demo_multiple_context_errors() {
    println!("📋 Demo 4: Multiple Context Errors");
    println!("-----------------------------------\n");

    let test_code = r#"
package com.example

import haxe.ds.StringMap

class Calculator {
    var operations: StringMap<String->Float->Float>
    
    public function new() {
        operations = new StringMap();
    }
    
    public function calculate(op: String, a: Float, b: Float): Float 
        var fn = operations.get(op);
        return fn(a, b);
    }
}
"#;

    let result = parse_incrementally_enhanced("calculator.hx", test_code);

    if result.has_errors() {
        let formatter = ErrorFormatter::with_colors();
        println!("❌ Parse failed with enhanced diagnostics:");
        println!(
            "{}",
            formatter.format_diagnostics(&result.diagnostics, &result.source_map)
        );
    } else {
        println!("✅ Parsed successfully");
    }

    // Show the number of each type of diagnostic
    let error_count = result.diagnostics.errors().count();
    let warning_count = result.diagnostics.warnings().count();
    let info_count = result.diagnostics.infos().count();
    let hint_count = result.diagnostics.hints().count();

    println!("\n📊 Diagnostic Summary:");
    println!("   Errors: {}", error_count);
    println!("   Warnings: {}", warning_count);
    println!("   Info: {}", info_count);
    println!("   Hints: {}", hint_count);

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}
