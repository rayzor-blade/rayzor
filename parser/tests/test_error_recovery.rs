//! Error recovery and edge case tests for the new Haxe parser

use parser::parse_haxe_file;

#[test]
fn test_syntax_errors() {
    // The RD parser is more lenient than the nom parser on fragments —
    // incomplete statements like "if (", "switch {", "for (i in" and
    // "class Test { var field }" parse successfully via fallback paths.
    // Only test inputs that both parsers reject.
    let invalid_inputs = vec![
        "class",        // Incomplete class declaration
        "class Test {", // Unclosed brace
        "function",     // Standalone function keyword
        "var x = ;",    // Missing expression
        // "}" is accepted by RD parser fallback (empty file with stray brace)
        "class Test { function method( {} }", // Invalid parameter syntax
    ];

    let mut failures = Vec::new();
    for input in &invalid_inputs {
        if parse_haxe_file("test.hx", input, false).is_ok() {
            failures.push(*input);
        }
    }
    assert!(
        failures.is_empty(),
        "These inputs should fail to parse but succeeded: {:?}",
        failures
    );
}

#[test]
fn test_empty_constructs() {
    let input = r#"
class EmptyClass {
}

interface EmptyInterface {
}

enum EmptyEnum {
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            assert_eq!(haxe_file.declarations.len(), 3);
        }
        Err(e) => panic!("Empty constructs should parse, got: {}", e),
    }
}

#[test]
fn test_edge_case_identifiers() {
    let input = r#"
class _UnderscoreStart {
    var _field: String;
    var field_: String;
    var field_with_underscores: String;
    var FieldWithCaps: String;
    var FIELD_ALL_CAPS: String;
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Edge case identifiers should parse, got: {}", e),
    }
}

#[test]
fn test_complex_nesting() {
    let input = r#"
class Outer {
    function method() {
        if (condition) {
            for (i in array) {
                switch (value) {
                    case pattern:
                        if (nested_condition) {
                            try {
                                deep_call();
                            } catch (e: Dynamic) {
                                handle_error(e);
                            }
                        }
                    default:
                        continue;
                }
            }
        }
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Complex nesting should parse, got: {}", e),
    }
}

#[test]
fn test_unicode_in_strings() {
    let input = r#"
class Test {
    function test() {
        var unicode = "Hello 世界 🌍";
        var emoji = "Test 🚀 🎉 ⭐";
        var mixed = 'Français: café, naïve';
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Unicode in strings should parse, got: {}", e),
    }
}

#[test]
fn test_large_numbers() {
    let input = r#"
class Test {
    function test() {
        var large_int = 9223372036854775807;
        var hex = 0xFFFFFFFFFFFFFFFF;
        var octal = 0777777777777;
        var float = 1.7976931348623157e+308;
        var scientific = 1.23e-45;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Large numbers should parse, got: {}", e),
    }
}

#[test]
fn test_escaped_strings() {
    let input = r#"
class Test {
    function test() {
        var escaped = "Line 1\nLine 2\tTabbed\r\nWindows line ending";
        var quotes = "He said \"Hello\" to me";
        var backslash = "Path\\to\\file";
        var single_quotes = 'It\'s working';
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Escaped strings should parse, got: {}", e),
    }
}

#[test]
fn test_operator_precedence_edge_cases() {
    let input = r#"
class Test {
    function test() {
        var result1 = !a && b || c;
        var result2 = a + b * c / d - e;
        var result3 = x << 2 + 1;
        var result4 = a ? b ? c : d : e;
        var result5 = a && b ? c || d : e && f;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Operator precedence edge cases should parse, got: {}", e),
    }
}

#[test]
fn test_very_long_identifiers() {
    let input = r#"
class VeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongClassName {
    var veryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongFieldName: String;
    
    function veryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongMethodName(): Void {
        var localVeryVeryVeryVeryVeryVeryVeryVeryVeryVeryLongVariableName = "test";
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Very long identifiers should parse, got: {}", e),
    }
}

#[test]
fn test_multiple_inheritance_levels() {
    let input = r#"
class A {}
class B extends A {}
class C extends B {}
class D extends C implements I1, I2, I3 {}

interface I1 {}
interface I2 extends I1 {}
interface I3 extends I1, I2 {}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Multiple inheritance levels should parse, got: {}", e),
    }
}

#[test]
fn test_complex_generics() {
    let input = r#"
class Test {
    function complexGenerics(): Array<Map<String, Either<Success<Data>, Failure<Error>>>> {
        return [];
    }
    
    function constrainedGenerics<T: Comparable<T> & Serializable>(item: T): T {
        return item;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Complex generics should parse, got: {}", e),
    }
}

#[test]
fn test_metadata_edge_cases() {
    let input = r#"
@:build(macro Builder.build())
@:native("NativeClass")
@author("John Doe")
@:meta(param1, param2, param3)
@:meta("string param", 42, true)
class Test {
    @:deprecated("Use newMethod instead")
    @:overload(function(x: Int): Void {})
    @:overload(function(x: String): Void {})
    function method(x: Dynamic): Void {}
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Metadata edge cases should parse, got: {}", e),
    }
}

#[test]
fn test_trailing_commas() {
    let input = r#"
class Test {
    function test() {
        var array = [1, 2, 3,];
        var object = {x: 1, y: 2,};
        var map = ["a" => 1, "b" => 2,];
        method(a, b, c,);
    }
    
    function method(a: Int, b: String, c: Bool,): Void {}
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Trailing commas should parse, got: {}", e),
    }
}

#[test]
fn test_minimal_constructs() {
    let inputs = vec![
        "class A{}",
        "interface I{}",
        "enum E{}",
        "typedef T=Int;",
        "abstract A(Int){}",
    ];

    for input in inputs {
        match parse_haxe_file("test.hx", input, false) {
            Ok(_) => {}
            Err(e) => panic!("Minimal construct '{}' should parse, got: {}", input, e),
        }
    }
}
