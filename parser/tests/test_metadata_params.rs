/// Test metadata parameter parsing
use parser::*;

#[test]
fn test_metadata_with_params() {
    let source = r#"
        @:safety(strict=true)
        class Main {
            static function main() {}
        }
    "#;

    match parse_haxe_file("test.hx", source, false) {
        Ok(file) => {
            assert_eq!(file.declarations.len(), 1);

            let class = match &file.declarations[0] {
                TypeDeclaration::Class(c) => c,
                other => panic!("Expected class, got: {:?}", other),
            };

            println!("Class metadata: {:#?}", class.meta);

            assert_eq!(class.meta.len(), 1);
            let meta = &class.meta[0];

            println!("Metadata name: {}", meta.name);
            println!("Metadata params count: {}", meta.params.len());

            if !meta.params.is_empty() {
                println!("First param: {:#?}", meta.params[0]);
            }

            assert_eq!(meta.name, "safety");
            assert_eq!(meta.params.len(), 1, "Expected 1 parameter for @:safety");

            // Check if the parameter is an assignment expression
            match &meta.params[0].kind {
                ExprKind::Assign { left, right, .. } => {
                    println!("✅ Parameter is an Assign expression");

                    if let ExprKind::Ident(name) = &left.kind {
                        assert_eq!(name, "strict");
                        println!("✅ Left side is 'strict'");
                    } else {
                        panic!("Left side should be an identifier");
                    }

                    if let ExprKind::Bool(value) = &right.kind {
                        assert!(*value);
                        println!("✅ Right side is 'true'");
                    } else {
                        panic!("Right side should be a bool literal, got: {:?}", right.kind);
                    }
                }
                other => panic!("Expected Assign expression, got: {:?}", other),
            }
        }
        Err(e) => panic!("Failed to parse: {:?}", e),
    }
}

#[test]
fn test_metadata_without_params() {
    let source = r#"
        @:safety
        class Main {}
    "#;

    match parse_haxe_file("test.hx", source, false) {
        Ok(file) => {
            assert_eq!(file.declarations.len(), 1);

            let class = match &file.declarations[0] {
                TypeDeclaration::Class(c) => c,
                other => panic!("Expected class, got: {:?}", other),
            };

            assert_eq!(class.meta.len(), 1);
            let meta = &class.meta[0];

            assert_eq!(meta.name, "safety");
            assert_eq!(meta.params.len(), 0, "Expected no parameters");

            println!("✅ @:safety without params parsed correctly");
        }
        Err(e) => panic!("Failed to parse: {:?}", e),
    }
}

#[test]
fn test_dotted_metadata_name() {
    let source = r#"
        class Main {
            @:haxe.warning("-WGenerator")
            static function f():Int return 1;
        }
    "#;

    match parse_haxe_file("test.hx", source, false) {
        Ok(file) => {
            let class = match &file.declarations[0] {
                TypeDeclaration::Class(c) => c,
                other => panic!("Expected class, got: {:?}", other),
            };
            let func = class
                .fields
                .iter()
                .find(|f| matches!(&f.kind, ClassFieldKind::Function(_)))
                .expect("expected a function field");
            assert_eq!(func.meta.len(), 1);
            assert_eq!(func.meta[0].name, "haxe.warning");
        }
        Err(e) => panic!("Dotted metadata name should parse, got: {}", e),
    }
}
