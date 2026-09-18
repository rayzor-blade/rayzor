//! Span tracking tests for the new Haxe parser

use parser::haxe_ast::{Span, TypeDeclaration};
use parser::parse_haxe_file;

#[test]
fn test_file_span_tracking() {
    let input = "package com.example;\nclass Test {}";

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            // File span should cover the entire input
            assert_eq!(haxe_file.span.start, 0);
            assert_eq!(haxe_file.span.end, input.len());
        }
        Err(e) => panic!("Should parse successfully, got: {}", e),
    }
}

#[test]
fn test_package_span() {
    let input = "package com.example.test;";

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            let package = haxe_file.package.unwrap();
            assert!(package.span.start < package.span.end);
            assert_eq!(package.span.start, 0);
            assert_eq!(package.span.end, input.len());

            // Extract the package text using span
            let package_text = &input[package.span.start..package.span.end];
            assert_eq!(package_text, "package com.example.test;");
        }
        Err(e) => panic!("Package parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_import_spans() {
    let input = r#"import haxe.Json;
import sys.io.File;"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            assert_eq!(haxe_file.imports.len(), 2);

            for import in &haxe_file.imports {
                assert!(import.span.start < import.span.end);
                let import_text = &input[import.span.start..import.span.end];
                assert!(import_text.starts_with("import"));
                assert!(import_text.ends_with(";"));
            }
        }
        Err(e) => panic!("Import parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_class_span() {
    let input = r#"class MyClass {
    var field: String;
    function method() {}
}"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert!(class.span.start < class.span.end);
                let class_text = &input[class.span.start..class.span.end];
                assert!(class_text.starts_with("class MyClass"));
                assert!(class_text.ends_with("}"));
            } else {
                panic!("Expected class declaration");
            }
        }
        Err(e) => panic!("Class parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_field_spans() {
    let input = r#"class Test {
    var field1: String;
    public var field2: Int = 42;
    function method(): Void {}
}"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                for field in &class.fields {
                    assert!(field.span.start < field.span.end);
                    let field_text = &input[field.span.start..field.span.end];

                    // Each field should contain its name based on kind
                    let field_name = match &field.kind {
                        parser::haxe_ast::ClassFieldKind::Var { name, .. } => name,
                        parser::haxe_ast::ClassFieldKind::Final { name, .. } => name,
                        parser::haxe_ast::ClassFieldKind::Property { name, .. } => name,
                        parser::haxe_ast::ClassFieldKind::Function(func) => &func.name,
                    };
                    assert!(field_text.contains(field_name));
                }
            } else {
                panic!("Expected class declaration");
            }
        }
        Err(e) => panic!("Field parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_expression_spans() {
    let input = r#"class Test {
    function test() {
        var x = 42;
        var y = "hello";
        var z = x + y.length;
    }
}"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0]
                && let Some(method) = class.fields.iter().find(|f| {
                    if let parser::haxe_ast::ClassFieldKind::Function(func) = &f.kind {
                        func.name == "test"
                    } else {
                        false
                    }
                })
                && let parser::haxe_ast::ClassFieldKind::Function(func) = &method.kind
                && let Some(body) = &func.body
            {
                assert!(body.span.start < body.span.end);

                // The body span should be within the input
                assert!(body.span.end <= input.len());
            }
        }
        Err(e) => panic!("Expression parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_nested_spans() {
    let input = r#"class Outer {
    class Inner {
        function method() {
            if (true) {
                trace("nested");
            }
        }
    }
}"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert!(class.span.start < class.span.end);

                // Verify all spans are within bounds
                fn check_span_bounds(span: &Span, input_len: usize) {
                    assert!(span.start <= span.end);
                    assert!(span.end <= input_len);
                }

                check_span_bounds(&class.span, input.len());
                for field in &class.fields {
                    check_span_bounds(&field.span, input.len());
                }
            }
        }
        Err(e) => panic!("Nested structure parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_metadata_spans() {
    let input = r#"@:native("MyClass")
@author("Developer")
class Test {
    @:optional
    var field: String;
}"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                // Check metadata spans
                for metadata in &class.meta {
                    assert!(metadata.span.start < metadata.span.end);
                    let meta_text = &input[metadata.span.start..metadata.span.end];
                    assert!(meta_text.starts_with('@'));
                }
            }
        }
        Err(e) => panic!("Metadata parsing should succeed, got: {}", e),
    }
}

#[test]
fn test_span_ordering() {
    let input = r#"package test;
import haxe.Json;
class Test {
    var field: String;
}"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            let mut last_end = 0;

            // Package should come first
            if let Some(package) = &haxe_file.package {
                assert!(package.span.start >= last_end);
                last_end = package.span.end;
            }

            // Imports should come after package
            for import in &haxe_file.imports {
                assert!(import.span.start >= last_end);
                last_end = import.span.end;
            }

            // Declarations should come after imports
            for decl in &haxe_file.declarations {
                let decl_span = match decl {
                    TypeDeclaration::Class(c) => &c.span,
                    TypeDeclaration::Interface(i) => &i.span,
                    TypeDeclaration::Enum(e) => &e.span,
                    TypeDeclaration::Typedef(t) => &t.span,
                    TypeDeclaration::Abstract(a) => &a.span,
                    TypeDeclaration::Conditional(c) => &c.span,
                };
                assert!(decl_span.start >= last_end);
            }
        }
        Err(e) => panic!("Span ordering test should succeed, got: {}", e),
    }
}

#[test]
fn test_empty_spans() {
    let input = "";

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            assert_eq!(haxe_file.span.start, 0);
            assert_eq!(haxe_file.span.end, 0);
        }
        Err(e) => panic!("Empty file should parse, got: {}", e),
    }
}

#[test]
fn test_whitespace_handling_in_spans() {
    let input = r#"
    
class   Test   {
    
    var    field :  String  ;
    
}

"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert!(class.span.start < class.span.end);

                // The span should include the actual content, not just whitespace
                let class_text = &input[class.span.start..class.span.end];
                assert!(class_text.contains("class"));
                assert!(class_text.contains("Test"));
            }
        }
        Err(e) => panic!("Whitespace handling should work, got: {}", e),
    }
}
