//! Expression parsing tests for the new Haxe parser

use parser::parse_haxe_file;

fn test_expression_parsing(expr: &str) {
    let input = &format!("class Test {{ function test() {{ {}; }} }}", expr);
    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Failed to parse expression '{}': {}", expr, e),
    }
}

#[test]
fn test_binary_expressions() {
    test_expression_parsing("a + b");
    test_expression_parsing("x * y - z");
    test_expression_parsing("1 + 2 * 3");
    test_expression_parsing("a && b || c");
    test_expression_parsing("x << 2");
    test_expression_parsing("a >>> 4");
    test_expression_parsing("value == 42");
    test_expression_parsing("x != y");
    test_expression_parsing("a < b");
    test_expression_parsing("x >= y");
}

#[test]
fn test_unary_expressions() {
    test_expression_parsing("!flag");
    test_expression_parsing("-number");
    test_expression_parsing("~bits");
    test_expression_parsing("++counter");
    test_expression_parsing("--value");
    test_expression_parsing("counter++");
    test_expression_parsing("value--");
}

#[test]
fn test_ternary_expression() {
    test_expression_parsing("condition ? true_value : false_value");
    test_expression_parsing("a > b ? a : b");
    test_expression_parsing("x == 0 ? \"zero\" : \"non-zero\"");
}

#[test]
fn test_function_calls() {
    test_expression_parsing("trace(\"hello\")");
    test_expression_parsing("Math.max(1, 2)");
    test_expression_parsing("obj.method()");
    test_expression_parsing("func(a, b, c)");
}

#[test]
fn test_field_access() {
    test_expression_parsing("obj.field");
    test_expression_parsing("this.property");
    test_expression_parsing("super.method");
    test_expression_parsing("instance.nested.deep");
}

#[test]
fn test_array_access() {
    test_expression_parsing("arr[0]");
    test_expression_parsing("map[\"key\"]");
    test_expression_parsing("matrix[i][j]");
}

#[test]
fn test_assignments() {
    test_expression_parsing("x = 42");
    test_expression_parsing("obj.field = value");
    test_expression_parsing("arr[0] = item");
    test_expression_parsing("x += 10");
    test_expression_parsing("y -= 5");
    test_expression_parsing("z *= 2");
    test_expression_parsing("w /= 3");
}

#[test]
fn test_control_flow_expressions() {
    test_expression_parsing("if (condition) action");
    test_expression_parsing("if (x > 0) positive else negative");

    test_expression_parsing("while (running) update()");
    test_expression_parsing("for (i in 0...10) trace(i)");
    test_expression_parsing("do action while (condition)");

    test_expression_parsing("return value");
    test_expression_parsing("return");
    test_expression_parsing("break");
    test_expression_parsing("continue");
    test_expression_parsing("throw error");
}

#[test]
fn test_variable_declarations() {
    test_expression_parsing("var x = 42");
    test_expression_parsing("var name: String = \"test\"");
    test_expression_parsing("var flag: Bool");
    test_expression_parsing("final PI = 3.14159");
}

#[test]
fn test_new_expressions() {
    test_expression_parsing("new MyClass()");
    test_expression_parsing("new Array<String>()");
    test_expression_parsing("new Map<String, Int>()");
    test_expression_parsing("new Point(10, 20)");
}

#[test]
fn test_cast_expressions() {
    test_expression_parsing("cast value");
    test_expression_parsing("cast(obj, MyType)");
    test_expression_parsing("cast obj");
}

#[test]
fn test_switch_expression() {
    let input = r#"
class Test {
    function test() {
        switch (value) {
            case 1: "one";
            case 2 | 3: "two or three";
            case x if (x > 10): "big";
            default: "other";
        };
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Switch expression should parse, got: {}", e),
    }
}

#[test]
fn test_try_catch_expression() {
    let input = r#"
class Test {
    function test() {
        try {
            risky_operation();
        } catch (e: String) {
            trace("String error: " + e);
        } catch (e: Dynamic) {
            trace("Other error");
        };
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Try-catch expression should parse, got: {}", e),
    }
}

#[test]
fn test_block_expressions() {
    let input = r#"
class Test {
    function test() {
        {
            var local = 42;
            trace(local);
        };
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Block expression should parse, got: {}", e),
    }
}

#[test]
fn test_function_expressions() {
    let input = r#"
class Test {
    function test() {
        var fn1 = function(x) return x * 2;
        var fn2 = function(x: Int, y: Int): Int return x + y;
        var fn3 = x -> x * 2;
        var fn4 = (x, y) -> x + y;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Function expressions should parse, got: {}", e),
    }
}

#[test]
fn test_array_comprehensions() {
    let input = r#"
class Test {
    function test() {
        var squares = [for (i in 0...10) i * i];
        var evens = [for (i in 0...20) if (i % 2 == 0) i];
        var pairs = [for (i in 0...5) i => i * 2];
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Array comprehensions should parse, got: {}", e),
    }
}

#[test]
fn test_function_local_static_var() {
    let input = r#"
class Test {
    function test() {
        static var c = 10;
        static var d = c + 1;
        static var f = value -> value + 1;
        static final g = 7;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Function-local static var should parse, got: {}", e),
    }
}

#[test]
fn test_complex_expressions() {
    test_expression_parsing("a.method().field[index] += value");
    test_expression_parsing("obj?.optionalField?.optionalMethod()");
    test_expression_parsing("arr.filter(x -> x > 0).map(x -> x * 2)");
    test_expression_parsing("condition ? func(a, b) : default_value");
}

#[test]
fn test_precedence() {
    // These should parse without errors, testing operator precedence
    test_expression_parsing("1 + 2 * 3"); // Should be 1 + (2 * 3)
    test_expression_parsing("a && b || c"); // Should be (a && b) || c
    test_expression_parsing("x << 2 + 1"); // Should be x << (2 + 1)
    test_expression_parsing("!flag && other"); // Should be (!flag) && other
}
