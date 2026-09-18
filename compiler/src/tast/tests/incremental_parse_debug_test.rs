//! Test using incremental parser for granular error reporting

#[cfg(test)]
mod tests {
    use parser::incremental_parser::{ParsedElement, parse_incrementally};

    #[test]
    fn test_incremental_parse_with_errors() {
        // Test content with a problematic switch statement
        let test_content = r#"package com.example;

import haxe.ds.StringMap;

class TestClass {
    static function test() {
        var x = 1;
        // This switch is causing the parse error
        var result = switch (x) {
            case 0: "zero";
            case 1: "one";
            case _: "other";
        }
    }
}"#;

        let result = parse_incrementally("test.hx", test_content);

        println!("Parse complete: {}", result.complete);
        println!("Parsed elements: {}", result.parsed_elements.len());
        println!("Errors: {}", result.errors.len());

        // Print parsed elements
        for (i, element) in result.parsed_elements.iter().enumerate() {
            match element {
                ParsedElement::Package(pkg) => {
                    println!("  [{}] Package: {:?}", i, pkg);
                }
                ParsedElement::Import(imp) => {
                    println!("  [{}] Import: {:?}", i, imp);
                }
                ParsedElement::TypeDeclaration(td) => {
                    println!("  [{}] Type Declaration: {:?}", i, td);
                }
                _ => {
                    println!("  [{}] Other: {:?}", i, element);
                }
            }
        }

        // Print errors with details
        for (i, error) in result.errors.iter().enumerate() {
            println!("\n❌ Error [{}] at line {}:{}", i, error.line, error.column);
            println!("   Message: {}", error.message);
            println!(
                "   Remaining input (first 100 chars): {:?}",
                &error.remaining_input[..100.min(error.remaining_input.len())]
            );
        }
    }

    #[test]
    fn test_comprehensive_with_incremental() {
        let test_content = r#"package com.example;

import haxe.ds.StringMap;
import haxe.macro.Context;
using StringTools;
using Lambda;

// Metadata and type parameters
@:generic
@:final
class Container<T> {
    public var items:Array<T>;

    public function new() {
        items = [];
    }

    // Simple switch without null pattern
    static function complexFunction(optional:Int):String {
        var result = switch (optional) {
            case 0: "zero";
            case 1: "one";
            case _: "other";
        }
        return result;
    }
}"#;

        let result = parse_incrementally("test.hx", test_content);

        println!("\n=== Comprehensive Parse Test ===");
        println!("Parse complete: {}", result.complete);
        println!("Parsed elements: {}", result.parsed_elements.len());
        println!("Errors: {}", result.errors.len());

        if !result.errors.is_empty() {
            println!("\n=== Errors Found ===");
            for error in &result.errors {
                println!("Line {}:{} - {}", error.line, error.column, error.message);
                println!(
                    "Near: {:?}",
                    &error.remaining_input[..50.min(error.remaining_input.len())]
                );
            }
        }
    }
}
