//! Comprehensive tests for multiple catch blocks implementation

use parser::parse_haxe_file;

#[test]
fn test_multiple_catch_with_different_types() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e: String) {
            trace("String error: " + e);
        } catch (e: Int) {
            trace("Int error: " + e);
        } catch (e: Float) {
            trace("Float error: " + e);
        } catch (e: Bool) {
            trace("Bool error: " + e);
        } catch (e: Dynamic) {
            trace("Dynamic error: " + e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(result) => {
            println!("Multiple catch blocks with different types parsed successfully");

            // Verify we have the expected number of catch blocks
            if let Some(parser::haxe_ast::TypeDeclaration::Class(class)) =
                result.declarations.first()
                && let Some(method) = class.fields.first()
                && let parser::haxe_ast::ClassFieldKind::Function(func) = &method.kind
                && let Some(body) = &func.body
                && let parser::haxe_ast::ExprKind::Block(block) = &body.kind
                && let Some(parser::haxe_ast::BlockElement::Expr(try_expr)) = block.first()
                && let parser::haxe_ast::ExprKind::Try { catches, .. } = &try_expr.kind
            {
                assert_eq!(catches.len(), 5, "Should have 5 catch blocks");

                // Verify the types are correct
                let expected_types = ["String", "Int", "Float", "Bool", "Dynamic"];
                for (i, catch_block) in catches.iter().enumerate() {
                    if let Some(parser::haxe_ast::Type::Path { path, .. }) = &catch_block.type_hint
                    {
                        assert_eq!(
                            path.name, expected_types[i],
                            "Catch block {} should have type {}",
                            i, expected_types[i]
                        );
                    }
                }
            }
        }
        Err(e) => {
            panic!(
                "Multiple catch blocks with different types should parse, got: {}",
                e
            );
        }
    }
}

#[test]
fn test_nested_try_catch() {
    let input = r#"
class Test {
    function test() {
        try {
            outerRiskyOperation();
        } catch (e: String) {
            try {
                innerRiskyOperation();
            } catch (inner: Int) {
                trace("Inner int error: " + inner);
            } catch (inner: Dynamic) {
                trace("Inner dynamic error: " + inner);
            }
        } catch (e: Dynamic) {
            trace("Outer dynamic error: " + e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("Nested try-catch parsed successfully");
        }
        Err(e) => {
            panic!("Nested try-catch should parse, got: {}", e);
        }
    }
}

#[test]
fn test_catch_with_complex_types() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e: Array<String>) {
            trace("Array error: " + e);
        } catch (e: Map<String, Int>) {
            trace("Map error: " + e);
        } catch (e: haxe.io.Error) {
            trace("IO error: " + e);
        } catch (e: Custom.Exception) {
            trace("Custom error: " + e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("Complex types in catch blocks parsed successfully");
        }
        Err(e) => {
            panic!("Complex types in catch blocks should parse, got: {}", e);
        }
    }
}

#[test]
fn test_catch_mixed_typed_and_untyped() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e: String) {
            trace("String error: " + e);
        } catch (e) {
            trace("Untyped error: " + e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("Mixed typed and untyped catch blocks parsed successfully");
        }
        Err(e) => {
            panic!(
                "Mixed typed and untyped catch blocks should parse, got: {}",
                e
            );
        }
    }
}

#[test]
fn test_catch_with_expressions() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e: String) {
            if (e.length > 0) {
                trace("Non-empty string error: " + e);
            }
        } catch (e: Int) {
            for (i in 0...e) {
                trace("Int error iteration: " + i);
            }
        } catch (e: Dynamic) {
            var handled = handleError(e);
            if (handled) {
                trace("Error handled successfully");
            } else {
                throw e;
            }
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("Catch blocks with complex expressions parsed successfully");
        }
        Err(e) => {
            panic!(
                "Catch blocks with complex expressions should parse, got: {}",
                e
            );
        }
    }
}

#[test]
fn test_try_catch_as_expression() {
    let input = r#"
class Test {
    function test() {
        var result = try {
            riskyOperation();
        } catch (e: String) {
            "String error: " + e;
        } catch (e: Int) {
            "Int error: " + e;
        } catch (e: Dynamic) {
            "Dynamic error: " + e;
        };
        
        trace(result);
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("Try-catch as expression parsed successfully");
        }
        Err(e) => {
            panic!("Try-catch as expression should parse, got: {}", e);
        }
    }
}

#[test]
fn test_catch_with_return_statements() {
    let input = r#"
class Test {
    function test():String {
        try {
            return riskyOperation();
        } catch (e: String) {
            return "String error: " + e;
        } catch (e: Int) {
            return "Int error: " + e;
        } catch (e: Dynamic) {
            return "Dynamic error: " + e;
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("Catch blocks with return statements parsed successfully");
        }
        Err(e) => {
            panic!(
                "Catch blocks with return statements should parse, got: {}",
                e
            );
        }
    }
}
