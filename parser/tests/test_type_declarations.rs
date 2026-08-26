//! Type declaration tests for the new Haxe parser

use parser::haxe_ast::TypeDeclaration;
use parser::parse_haxe_file;

#[test]
fn test_simple_class() {
    let input = r#"
class MyClass {
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            assert_eq!(haxe_file.declarations.len(), 1);
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert_eq!(class.name, "MyClass");
                assert!(class.extends.is_none());
                assert!(class.implements.is_empty());
                assert!(class.fields.is_empty());
            } else {
                panic!("Expected class declaration");
            }
        }
        Err(e) => panic!("Simple class should parse, got: {}", e),
    }
}

#[test]
fn test_class_with_inheritance() {
    let input = r#"
class Child extends Parent implements IInterface1, IInterface2 {
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert_eq!(class.name, "Child");
                assert!(class.extends.is_some());
                assert_eq!(class.implements.len(), 2);
            } else {
                panic!("Expected class declaration");
            }
        }
        Err(e) => panic!("Class with inheritance should parse, got: {}", e),
    }
}

#[test]
fn test_class_with_fields() {
    let input = r#"
class MyClass {
    var field1: String;
    public var field2: Int = 42;
    private static var field3: Bool = true;
    
    public function new() {}
    
    public function method(param: String): Void {
        trace(param);
    }
    
    private static function staticMethod(): Int {
        return 42;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert_eq!(class.name, "MyClass");
                assert_eq!(class.fields.len(), 6);
            } else {
                panic!("Expected class declaration");
            }
        }
        Err(e) => panic!("Class with fields should parse, got: {}", e),
    }
}

#[test]
fn test_overload_between_function_modifiers() {
    let input = r#"
class Main {
    extern inline overload static function make(value:Int):Int {
        return value;
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, true);
    assert!(
        result.is_ok(),
        "Failed to parse overload between modifiers: {:?}",
        result.err()
    );
}

#[test]
fn test_class_with_properties() {
    let input = r#"
class MyClass {
    public var prop(get, set): Int;
    public var readOnly(get, never): String;
    public var writeOnly(never, set): Float;
    
    function get_prop(): Int return 42;
    function set_prop(value: Int): Int return value;
    function get_readOnly(): String return "test";
    function set_writeOnly(value: Float): Float return value;
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Class with properties should parse, got: {}", e),
    }
}

#[test]
fn test_interface() {
    let input = r#"
interface IMyInterface extends IParent {
    function method1(param: String): Void;
    function method2(): Int;
    var property(get, set): String;
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Interface(interface) = &haxe_file.declarations[0] {
                assert_eq!(interface.name, "IMyInterface");
                assert_eq!(interface.fields.len(), 3);
            } else {
                panic!("Expected interface declaration");
            }
        }
        Err(e) => panic!("Interface should parse, got: {}", e),
    }
}

#[test]
fn test_enum() {
    let input = r#"
enum Color {
    Red;
    Green;
    Blue;
    RGB(r: Int, g: Int, b: Int);
    HSV(h: Float, s: Float, v: Float);
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Enum(enum_decl) = &haxe_file.declarations[0] {
                assert_eq!(enum_decl.name, "Color");
                assert_eq!(enum_decl.constructors.len(), 5);
            } else {
                panic!("Expected enum declaration");
            }
        }
        Err(e) => panic!("Enum should parse, got: {}", e),
    }
}

#[test]
fn test_typedef() {
    let input = r#"
typedef Point = {
    x: Float,
    y: Float,
    ?z: Float
};

typedef StringMap<T> = Map<String, T>;

typedef Callback = String -> Int -> Void;
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            assert_eq!(haxe_file.declarations.len(), 3);
            for decl in &haxe_file.declarations {
                if let TypeDeclaration::Typedef(_) = decl {
                    // Good
                } else {
                    panic!("Expected typedef declaration");
                }
            }
        }
        Err(e) => panic!("Typedefs should parse, got: {}", e),
    }
}

#[test]
fn test_abstract() {
    let input = r#"
abstract Vec2(Point) from Point to Point {
    public function new(x: Float, y: Float) {
        this = {x: x, y: y};
    }
    
    @:op(A + B)
    public function add(other: Vec2): Vec2 {
        return new Vec2(this.x + other.x, this.y + other.y);
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            if let TypeDeclaration::Abstract(abstract_decl) = &haxe_file.declarations[0] {
                assert_eq!(abstract_decl.name, "Vec2");
                assert_eq!(abstract_decl.fields.len(), 2);
            } else {
                panic!("Expected abstract declaration");
            }
        }
        Err(e) => panic!("Abstract should parse, got: {}", e),
    }
}

#[test]
fn test_generic_classes() {
    let input = r#"
class Container<T> {
    var value: T;
    
    public function new(value: T) {
        this.value = value;
    }
    
    public function get(): T {
        return value;
    }
}

class Pair<T, U> {
    public var first: T;
    public var second: U;
    
    public function new(first: T, second: U) {
        this.first = first;
        this.second = second;
    }
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(haxe_file) => {
            assert_eq!(haxe_file.declarations.len(), 2);

            if let TypeDeclaration::Class(class) = &haxe_file.declarations[0] {
                assert_eq!(class.name, "Container");
                assert_eq!(class.type_params.len(), 1);
            } else {
                panic!("Expected class declaration");
            }

            if let TypeDeclaration::Class(class) = &haxe_file.declarations[1] {
                assert_eq!(class.name, "Pair");
                assert_eq!(class.type_params.len(), 2);
            } else {
                panic!("Expected class declaration");
            }
        }
        Err(e) => panic!("Generic classes should parse, got: {}", e),
    }
}

#[test]
fn test_extern_class() {
    let input = r#"
extern class JsArray<T> {
    var length: Int;
    function push(item: T): Int;
    function pop(): T;
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Extern class should parse, got: {}", e),
    }
}

#[test]
fn test_metadata_on_declarations() {
    let input = r#"
@:build(Builder.build())
@:native("MyNativeClass")
class MyClass {
    @:optional
    var field: String;
    
    @:overload(function(x: Int): Void {})
    public function method(x: String): Void {}
}
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {}
        Err(e) => panic!("Metadata on declarations should parse, got: {}", e),
    }
}
