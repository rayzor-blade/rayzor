//! Test multiple catch blocks

use parser::parse_haxe_file;

#[test]
fn test_multiple_catch_blocks() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e: String) {
            handleStringError(e);
        } catch (e: Int) {
            handleIntError(e);
        } catch (e: Dynamic) {
            handleGenericError(e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(result) => {
            println!("Parsing successful!");
            // Check that we have 3 catch blocks
            if let Some(parser::haxe_ast::TypeDeclaration::Class(class)) =
                result.declarations.first()
                && let Some(method) = class.fields.first()
                && let parser::haxe_ast::ClassFieldKind::Function(func) = &method.kind
                && let Some(body) = &func.body
                && let parser::haxe_ast::ExprKind::Block(block) = &body.kind
                && let Some(parser::haxe_ast::BlockElement::Expr(try_expr)) = block.first()
                && let parser::haxe_ast::ExprKind::Try { catches, .. } = &try_expr.kind
            {
                println!("Found {} catch blocks", catches.len());
                assert_eq!(catches.len(), 3, "Should have 3 catch blocks");

                // Check types
                if let Some(type_hint) = &catches[0].type_hint {
                    println!("First catch type: {:?}", type_hint);
                }
                if let Some(type_hint) = &catches[1].type_hint {
                    println!("Second catch type: {:?}", type_hint);
                }
                if let Some(type_hint) = &catches[2].type_hint {
                    println!("Third catch type: {:?}", type_hint);
                }
            }
        }
        Err(e) => {
            panic!("Multiple catch blocks should parse, got: {}", e);
        }
    }
}

#[test]
fn test_single_catch_block() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e: Dynamic) {
            handleError(e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => {
            panic!("Single catch block should parse, got: {}", e);
        }
    }
}

#[test]
fn test_catch_without_type() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e) {
            handleError(e);
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => {
            panic!("Catch without type should parse, got: {}", e);
        }
    }
}
