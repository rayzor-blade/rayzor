//! Literal expression tests for the new Haxe parser

use parser::haxe_ast::{ExprKind, TypeDeclaration};
use parser::{ClassFieldKind, parse_haxe_file};

fn parse_simple_expr(expr: &str) -> ExprKind {
    let input = &format!("class Test {{ function test() {{ var x = {}; }} }}", expr);
    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                for field in &class.fields {
                    if let ClassFieldKind::Function(func) = &field.kind
                        && func.name == "test"
                        && field.modifiers.is_empty()
                        && let Some(body) = &func.body
                        && let ExprKind::Block(elements) = &body.kind
                        && let Some(parser::haxe_ast::BlockElement::Expr(expr)) = elements.first()
                        && let ExprKind::Var {
                            expr: Some(var_expr),
                            ..
                        } = &expr.kind
                    {
                        return var_expr.kind.clone();
                    }
                }
            }
            panic!("Could not extract expression from parsed result");
        }
        Err(e) => panic!("Failed to parse expression '{}': {}", expr, e),
    }
}

#[test]
fn test_integer_literals() {
    // Decimal
    match parse_simple_expr("42") {
        ExprKind::Int(42) => {}
        other => panic!("Expected Int(42), got {:?}", other),
    }

    // Hex
    match parse_simple_expr("0xFF") {
        ExprKind::Int(255) => {}
        other => panic!("Expected Int(255), got {:?}", other),
    }

    // Octal
    match parse_simple_expr("0755") {
        ExprKind::Int(493) => {}
        other => panic!("Expected Int(493), got {:?}", other),
    }
}

#[test]
#[allow(clippy::approx_constant)]
fn test_float_literals() {
    match parse_simple_expr("3.14") {
        ExprKind::Float(f) if (f - 3.14_f64).abs() < 0.001 => {}
        other => panic!("Expected Float(3.14), got {:?}", other),
    }

    match parse_simple_expr("1.23e-4") {
        ExprKind::Float(f) if (f - 0.000123).abs() < 0.0000001 => {}
        other => panic!("Expected Float(0.000123), got {:?}", other),
    }
}

#[test]
fn test_string_literals() {
    match parse_simple_expr("\"hello\"") {
        ExprKind::String(s) if s == "hello" => {}
        other => panic!("Expected String(\"hello\"), got {:?}", other),
    }

    match parse_simple_expr("'world'") {
        ExprKind::String(s) if s == "world" => {}
        other => panic!("Expected String(\"world\"), got {:?}", other),
    }
}

#[test]
fn test_boolean_literals() {
    match parse_simple_expr("true") {
        ExprKind::Bool(true) => {}
        other => panic!("Expected Bool(true), got {:?}", other),
    }

    match parse_simple_expr("false") {
        ExprKind::Bool(false) => {}
        other => panic!("Expected Bool(false), got {:?}", other),
    }
}

#[test]
fn test_null_literal() {
    match parse_simple_expr("null") {
        ExprKind::Null => {}
        other => panic!("Expected Null, got {:?}", other),
    }
}

#[test]
fn test_string_interpolation() {
    let input = r#"
class Test {
    function test() {
        var name = "world";
        var greeting = 'Hello $name!';
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {} // Just check it parses for now
        Err(e) => panic!("String interpolation should parse, got: {}", e),
    }
}

#[test]
fn test_array_literals() {
    let input = r#"
class Test {
    function test() {
        var empty = [];
        var numbers = [1, 2, 3];
        var mixed = [1, "hello", true];
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Array literals should parse, got: {}", e),
    }
}

#[test]
fn test_object_literals() {
    let input = r#"
class Test {
    function test() {
        var empty = {};
        var point = {x: 10, y: 20};
        var mixed = {name: "test", value: 42, active: true};
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Object literals should parse, got: {}", e),
    }
}
