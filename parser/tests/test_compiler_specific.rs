//! Tests for compiler-specific code blocks

use parser::{
    haxe_ast::{BlockElement, ClassFieldKind, ExprKind, TypeDeclaration},
    parse_haxe_file,
};

#[test]
fn test_js_code_block() {
    let input = r#"
        class Test {
            function platformSpecific():Void {
                __js__("console.log('Hello from JavaScript');");
            }
        }
    "#;

    let result = parse_haxe_file("test.hx", input, false);

    assert!(result.is_ok(), "Failed to parse: {:?}", result);

    let file = result.unwrap();
    assert_eq!(file.declarations.len(), 1);

    // Verify the compiler-specific code block was parsed correctly
    if let TypeDeclaration::Class(class) = &file.declarations[0] {
        assert_eq!(class.fields.len(), 1);
        if let ClassFieldKind::Function(func) = &class.fields[0].kind
            && let Some(body) = &func.body
            && let ExprKind::Block(elements) = &body.kind
            && let BlockElement::Expr(expr) = &elements[0]
        {
            match &expr.kind {
                ExprKind::CompilerSpecific { target, code, .. } => {
                    assert_eq!(target, "__js__");
                    if let ExprKind::String(s) = &code.kind {
                        assert_eq!(s, "console.log('Hello from JavaScript');");
                    } else {
                        panic!("Expected string literal for code");
                    }
                }
                _ => {
                    panic!("Expected CompilerSpecific expression, got {:?}", expr.kind)
                }
            }
        }
    }
}

#[test]
fn test_cpp_code_block() {
    let input = r#"
        class Test {
            function cppSpecific():Void {
                __cpp__("std::cout << 'Hello from C++' << std::endl;");
            }
        }
    "#;

    let result = parse_haxe_file("test.hx", input, false);

    assert!(result.is_ok(), "Failed to parse: {:?}", result);
}

#[test]
fn test_multiple_platform_blocks() {
    let input = r#"
        class Test {
            function multiPlatform():Void {
                __js__("console.log('JS');");
                __cpp__("printf('C++\\n');");
                __cs__("Console.WriteLine('C#');");
                __java__("System.out.println('Java');");
                __python__("print('Python')");
            }
        }
    "#;

    let result = parse_haxe_file("test.hx", input, false);

    assert!(result.is_ok(), "Failed to parse: {:?}", result);
}

#[test]
fn test_compiler_specific_with_interpolation() {
    let input = r#"
        class Test {
            function withInterpolation():Void {
                var name = "World";
                __js__('console.log("Hello, ${name}!");');
            }
        }
    "#;

    let result = parse_haxe_file("test.hx", input, false);

    assert!(result.is_ok(), "Failed to parse: {:?}", result);
}

#[test]
fn test_compiler_specific_in_expression() {
    let input = r#"
        class Test {
            function inExpression():Int {
                return __js__("Math.floor(3.14)");
            }
        }
    "#;

    let result = parse_haxe_file("test.hx", input, false);

    assert!(result.is_ok(), "Failed to parse: {:?}", result);
}

#[test]
fn test_direct_compiler_specific() {
    use parser::haxe_parser_expr::expression;

    // Test just the expression parsing
    let expr_input = r#"__js__("console.log('test')")"#;
    let full = expr_input;
    let expr_result = expression(full, expr_input);

    assert!(
        expr_result.is_ok(),
        "Failed to parse expression: {:?}",
        expr_result
    );

    if let Ok((_, expr)) = expr_result {
        match &expr.kind {
            ExprKind::CompilerSpecific { target, code, .. } => {
                assert_eq!(target, "__js__");
                match &code.kind {
                    ExprKind::String(s) => assert_eq!(s, "console.log('test')"),
                    _ => panic!("Expected string literal for code, got {:?}", code.kind),
                }
            }
            _ => panic!("Expected CompilerSpecific expression, got {:?}", expr.kind),
        }
    }
}
