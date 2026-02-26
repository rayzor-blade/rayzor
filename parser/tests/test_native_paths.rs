#[cfg(test)]
mod tests {
    use parser::haxe_ast::{ClassFieldKind, ExprKind, TypeDeclaration};
    use parser::parse_haxe_file;

    #[test]
    fn test_native_with_dotted_paths() {
        let input = r#"
package com.example;

// Java interop
@:native("java.util.ArrayList")
extern class ArrayList<T> {
    @:native("new")
    function new();
    
    @:native("add")
    function push(item:T):Bool;
    
    @:native("get")
    function at(index:Int):T;
    
    @:native("size")
    function length():Int;
}

// C# interop  
@:native("System.Collections.Generic.Dictionary")
extern class Dictionary<K, V> {
    @:native(".ctor")
    function new();
    
    @:native("TryGetValue")
    function tryGet(key:K, value:cs.Out<V>):Bool;
}

// JavaScript interop
@:native("window.console.log")
extern class Console {
    static function log(msg:Dynamic):Void;
}

// Complex nested paths
@:native("com.company.product.module.subsystem.ComponentFactory")
@:nativeGen
class ComponentFactory {
    @:native("com.company.product.module.subsystem.ComponentFactory.getInstance")
    static function getInstance():ComponentFactory;
    
    @:native("createComponent")
    function create(type:String):Dynamic;
}

// Python interop
@:native("numpy.ndarray")
extern class NDArray {
    @:native("shape")
    var shape:Array<Int>;
    
    @:native("dtype")
    var dtype:Dynamic;
    
    @:native("reshape")
    function reshape(shape:Array<Int>):NDArray;
}

// PHP namespaces
@:native("\\Vendor\\Package\\ClassName")
extern class PhpClass {
    @:native("__construct")
    function new();
    
    @:native("\\Vendor\\Package\\ClassName::staticMethod")
    static function staticMethod():Void;
}
"#;

        let ast =
            parse_haxe_file("test_native_paths.hx", input, false).expect("Parsing should succeed");

        // Verify all classes were parsed
        assert_eq!(ast.declarations.len(), 6);

        // Check ArrayList
        if let TypeDeclaration::Class(ref c) = &ast.declarations[0] {
            assert_eq!(c.name, "ArrayList");
            let native_meta = c
                .meta
                .iter()
                .find(|m| m.name == "native")
                .expect("Expected @:native");
            if let ExprKind::String(s) = &native_meta.params[0].kind {
                assert_eq!(s, "java.util.ArrayList");
            } else {
                panic!("Expected string parameter");
            }
        }

        // Check Dictionary with C# namespace
        if let TypeDeclaration::Class(ref c) = &ast.declarations[1] {
            assert_eq!(c.name, "Dictionary");
            let native_meta = c
                .meta
                .iter()
                .find(|m| m.name == "native")
                .expect("Expected @:native");
            if let ExprKind::String(s) = &native_meta.params[0].kind {
                assert_eq!(s, "System.Collections.Generic.Dictionary");
            } else {
                panic!("Expected string parameter");
            }
        }

        // Check Console with window.console.log
        if let TypeDeclaration::Class(ref c) = &ast.declarations[2] {
            assert_eq!(c.name, "Console");
            let native_meta = c
                .meta
                .iter()
                .find(|m| m.name == "native")
                .expect("Expected @:native");
            if let ExprKind::String(s) = &native_meta.params[0].kind {
                assert_eq!(s, "window.console.log");
            } else {
                panic!("Expected string parameter");
            }
        }

        // Check ComponentFactory with very long path
        if let TypeDeclaration::Class(ref c) = &ast.declarations[3] {
            assert_eq!(c.name, "ComponentFactory");
            let native_meta = c
                .meta
                .iter()
                .find(|m| m.name == "native")
                .expect("Expected @:native");
            if let ExprKind::String(s) = &native_meta.params[0].kind {
                assert_eq!(s, "com.company.product.module.subsystem.ComponentFactory");
            } else {
                panic!("Expected string parameter");
            }

            // Check static method with dotted native path
            let method = c
                .fields
                .iter()
                .find(|f| {
                    if let ClassFieldKind::Function(func) = &f.kind {
                        func.name == "getInstance"
                    } else {
                        false
                    }
                })
                .expect("Expected getInstance method");

            let method_native = method
                .meta
                .iter()
                .find(|m| m.name == "native")
                .expect("Expected @:native on method");
            if let ExprKind::String(s) = &method_native.params[0].kind {
                assert_eq!(
                    s,
                    "com.company.product.module.subsystem.ComponentFactory.getInstance"
                );
            } else {
                panic!("Expected string parameter");
            }
        }

        // Check PHP class with backslashes
        if let TypeDeclaration::Class(ref c) = &ast.declarations[5] {
            assert_eq!(c.name, "PhpClass");
            let native_meta = c
                .meta
                .iter()
                .find(|m| m.name == "native")
                .expect("Expected @:native");
            if let ExprKind::String(s) = &native_meta.params[0].kind {
                // Haxe `"\\"` is a single backslash — parser unescapes the string literal
                assert_eq!(s, "\\Vendor\\Package\\ClassName");
            } else {
                panic!("Expected string parameter");
            }
        }
    }

    #[test]
    fn test_native_edge_cases() {
        let edge_cases = vec![
            (r#"@:native("") class Empty {}"#, ""),
            (r#"@:native(".") class Dot {}"#, "."),
            (r#"@:native("..") class DoubleDot {}"#, ".."),
            (
                r#"@:native("a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p") class VeryLong {}"#,
                "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p",
            ),
            (
                r#"@:native("_internal.package.Class") class Internal {}"#,
                "_internal.package.Class",
            ),
            (
                r#"@:native("$special.chars.Class") class Special {}"#,
                "$special.chars.Class",
            ),
            (
                r#"@:native("数字.汉字.Class") class Unicode {}"#,
                "数字.汉字.Class",
            ),
            (
                r#"@:native("mixed.CASE.paTTern") class MixedCase {}"#,
                "mixed.CASE.paTTern",
            ),
        ];

        for (input, expected_native) in edge_cases {
            let ast = parse_haxe_file("edge_case.hx", input, false)
                .unwrap_or_else(|_| panic!("Failed to parse: {}", input));
            if let Some(TypeDeclaration::Class(ref c)) = ast.declarations.first() {
                let native_meta = c
                    .meta
                    .iter()
                    .find(|m| m.name == "native")
                    .expect("Expected @:native");
                if let ExprKind::String(s) = &native_meta.params[0].kind {
                    assert_eq!(
                        s, expected_native,
                        "Native path mismatch for input: {}",
                        input
                    );
                } else {
                    panic!("Expected string parameter for: {}", input);
                }
            } else {
                panic!("Expected class declaration for: {}", input);
            }
        }
    }

    #[test]
    fn test_native_invalid_syntax() {
        // These should still parse - the string content is not validated by the parser
        let invalid_cases = vec![
            r#"@:native("...") class TripleDot {}"#,
            r#"@:native("package..Class") class DoubleDotInside {}"#,
            r#"@:native(".package.Class") class LeadingDot {}"#,
            r#"@:native("package.Class.") class TrailingDot {}"#,
            r#"@:native("package .Class") class SpaceInPath {}"#,
            r#"@:native("package\n.Class") class NewlineInPath {}"#,
        ];

        for input in invalid_cases {
            let result = parse_haxe_file("invalid.hx", input, false);
            assert!(
                result.is_ok(),
                "Parser should accept any string content: {}",
                input
            );
        }
    }
}
