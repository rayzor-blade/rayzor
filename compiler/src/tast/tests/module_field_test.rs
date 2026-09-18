//! Test module field parsing specifically

#[cfg(test)]
mod tests {
    use parser::incremental_parser::{ParsedElement, parse_incrementally};

    #[test]
    fn test_module_field_parsing() {
        // Test content with module-level fields
        let test_content = r#"package com.example;

import haxe.ds.StringMap;

// Module-level variable
var moduleVar:Int = 42;

// Module-level final
final MODULE_CONST:String = "hello";

// Module-level function
function moduleFunction():String {
    return "module function";
}

class TestClass {
    public function new() {}
}"#;

        let result = parse_incrementally("test.hx", test_content);

        println!("Parse complete: {}", result.complete);
        println!("Parsed elements: {}", result.parsed_elements.len());
        println!("Errors: {}", result.errors.len());

        let mut module_field_count = 0;

        // Print and count parsed elements
        for (i, element) in result.parsed_elements.iter().enumerate() {
            match element {
                ParsedElement::Package(pkg) => {
                    println!("  [{}] Package: {:?}", i, pkg);
                }
                ParsedElement::Import(imp) => {
                    println!("  [{}] Import: {:?}", i, imp);
                }
                ParsedElement::ModuleField(mf) => {
                    println!("  [{}] Module Field: {:?}", i, mf);
                    module_field_count += 1;
                }
                ParsedElement::TypeDeclaration(td) => {
                    println!("  [{}] Type Declaration: {:?}", i, td);
                }
                _ => {
                    println!("  [{}] Other: {:?}", i, element);
                }
            }
        }

        println!("Module fields found: {}", module_field_count);

        // Print errors if any
        for (i, error) in result.errors.iter().enumerate() {
            println!("\n❌ Error [{}] at line {}:{}", i, error.line, error.column);
            println!("   Message: {}", error.message);
            println!(
                "   Remaining input (first 100 chars): {:?}",
                &error.remaining_input[..100.min(error.remaining_input.len())]
            );
        }

        // We should have parsed at least 3 module fields (var, final, function)
        assert!(
            module_field_count >= 3,
            "Expected at least 3 module fields but found {}",
            module_field_count
        );
    }

    #[test]
    fn test_simple_module_field() {
        let test_content = r#"package test;

var x:Int = 10;

class Test {}"#;

        let result = parse_incrementally("test.hx", test_content);

        println!("\n=== Simple Module Field Test ===");
        println!("Complete: {}", result.complete);
        println!("Elements: {}", result.parsed_elements.len());
        println!("Errors: {}", result.errors.len());

        for (i, element) in result.parsed_elements.iter().enumerate() {
            println!("  [{}] {:?}", i, element);
        }

        let module_fields = result
            .parsed_elements
            .iter()
            .filter(|e| matches!(e, ParsedElement::ModuleField(_)))
            .count();

        println!("Module fields: {}", module_fields);

        if !result.errors.is_empty() {
            for error in &result.errors {
                println!(
                    "Error: {} at {}:{}",
                    error.message, error.line, error.column
                );
            }
        }
    }
}
