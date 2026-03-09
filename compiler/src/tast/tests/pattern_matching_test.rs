#[cfg(test)]
mod pattern_matching_tests {
    use crate::pipeline::compile_haxe_source;

    /// Run compile_haxe_source on a thread with 16MB stack to handle the deep
    /// recursion of the full compilation pipeline (stdlib + TAST + HIR + MIR).
    /// The default test thread stack (~8MB) is insufficient for complex patterns.
    /// Returns error messages (empty vec on success).
    fn compile_with_stack(code: &str) -> Vec<String> {
        let code = code.to_string();
        std::thread::Builder::new()
            .name("compile-test".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let result = compile_haxe_source(&code);
                result
                    .errors
                    .iter()
                    .map(|e| e.message.clone())
                    .collect::<Vec<String>>()
            })
            .expect("failed to spawn thread")
            .join()
            .expect("compilation thread panicked")
    }

    #[test]
    fn test_const_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var x = 5;
        var result = switch (x) {
            case 1: "one";
            case 5: "five";
            case 10: "ten";
            default: "other";
        };
        trace(result); // Should be "five"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_variable_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var x = 5;
        var result = switch (x) {
            case n: 'number is $n';
        };
        trace(result); // Should be "number is 5"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_array_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var arr = [1, 2, 3];
        var result = switch (arr) {
            case []: "empty";
            case [x]: 'single: $x';
            case [x, y]: 'pair: $x, $y';
            case [x, y, z]: 'triple: $x, $y, $z';
            default: "many";
        };
        trace(result); // Should be "triple: 1, 2, 3"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_array_rest_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var arr = [1, 2, 3, 4, 5];
        var result = switch (arr) {
            case []: "empty";
            case [first, ...rest]: 'first: $first, rest: $rest';
            default: "no match";
        };
        trace(result); // Should be "first: 1, rest: [2, 3, 4, 5]"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_object_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var point = {x: 10, y: 20};
        var result = switch (point) {
            case {x: 0, y: 0}: "origin";
            case {x: x, y: 0}: 'x-axis at $x';
            case {x: 0, y: y}: 'y-axis at $y';
            case {x: x, y: y}: 'point at ($x, $y)';
        };
        trace(result); // Should be "point at (10, 20)"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_null_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var x:Null<Int> = null;
        var result = switch (x) {
            case null: "is null";
            case n: 'not null: $n';
        };
        trace(result); // Should be "is null"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_underscore_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var x = "hello";
        var result = switch (x) {
            case "bye": "farewell";
            case _: "anything else";
        };
        trace(result); // Should be "anything else"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_or_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var x = 2;
        var result = switch (x) {
            case 1 | 2 | 3: "small";
            case 4 | 5 | 6: "medium";
            default: "large";
        };
        trace(result); // Should be "small"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_type_pattern() {
        let haxe_code = r#"
class Test {
    static function main() {
        var x:Dynamic = "hello";
        var result = switch (x) {
            case (s:String): 'string: $s';
            case (i:Int): 'int: $i';
            case (f:Float): 'float: $f';
            default: "unknown type";
        };
        trace(result); // Should be "string: hello"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_nested_patterns() {
        let haxe_code = r#"
class Test {
    static function main() {
        var data = {name: "John", pos: {x: 10, y: 20}};
        var result = switch (data) {
            case {name: n, pos: {x: 0, y: 0}}: '$n is at origin';
            case {name: n, pos: {x: x, y: y}}: '$n is at ($x, $y)';
            default: "no match";
        };
        trace(result); // Should be "John is at (10, 20)"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_enum_constructor_pattern() {
        let haxe_code = r#"
enum Option<T> {
    None;
    Some(value:T);
}

class Test {
    static function main() {
        var opt = Some(42);
        var result = switch (opt) {
            case None: "nothing";
            case Some(v): 'value: $v';
        };
        trace(result); // Should be "value: 42"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_complex_constructor_pattern() {
        let haxe_code = r#"
enum Tree<T> {
    Leaf(value:T);
    Node(left:Tree<T>, right:Tree<T>);
}

class Test {
    static function main() {
        var tree = Node(Leaf(1), Node(Leaf(2), Leaf(3)));
        var result = switch (tree) {
            case Leaf(v): 'leaf: $v';
            case Node(Leaf(l), Leaf(r)): 'two leaves: $l and $r';
            case Node(Leaf(l), Node(_, _)): 'left leaf: $l, right node';
            case Node(Node(_, _), Leaf(r)): 'left node, right leaf: $r';
            case Node(_, _): "two nodes";
        };
        trace(result); // Should be "left leaf: 1, right node"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }

    #[test]
    fn test_mixed_patterns() {
        let haxe_code = r#"
enum Result<T, E> {
    Ok(value:T);
    Err(error:E);
}

class Test {
    static function main() {
        var results = [Ok(1), Err("failed"), Ok(3)];
        var result = switch (results) {
            case []: "empty";
            case [Ok(v)]: 'single success: $v';
            case [Ok(v1), Ok(v2)]: 'two successes: $v1, $v2';
            case [Ok(v), Err(e), ...rest]: 'success $v, then error: $e';
            default: "other pattern";
        };
        trace(result); // Should be "success 1, then error: failed"
    }
}
        "#;

        let errors = compile_with_stack(haxe_code);
        assert!(
            errors.is_empty(),
            "Expected no errors, but got: {:?}",
            errors
        );
    }
}
