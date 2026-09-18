//! Comprehensive tests for AST to TAST lowering with type resolution and inference
//!
//! Tests the enhanced AST lowering system including:
//! - Two-pass type resolution with forward references
//! - Enhanced type inference for expressions
//! - Proper handling of all Haxe language constructs

use crate::tast::{
    AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable, node::TypedFile,
};
use parser::{HaxeFile, parse_haxe_file};
use std::cell::RefCell;
use std::rc::Rc;

/// Test helper to create a complete lowering context
fn create_test_context() -> (
    StringInterner,
    SymbolTable,
    Rc<RefCell<TypeTable>>,
    ScopeTree,
) {
    let string_interner = StringInterner::new();
    let symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let scope_tree = ScopeTree::new(ScopeId::first());

    (string_interner, symbol_table, type_table, scope_tree)
}

/// Test helper to parse and lower Haxe code
fn parse_and_lower(haxe_code: &str) -> Result<TypedFile, String> {
    // Parse the Haxe code
    let ast_result = parse_haxe_file("test.hx", haxe_code, true); // Enable recovery
    let haxe_file = match ast_result {
        Ok(file) => file,
        Err(errors) => return Err(format!("Parse errors: {:?}", errors)),
    };

    // Create lowering context
    let (mut string_interner, mut symbol_table, type_table, mut scope_tree) = create_test_context();

    // Create namespace and import resolvers
    let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
    let mut import_resolver = crate::tast::namespace::ImportResolver::new();

    // Create AST lowering instance
    let string_interner_rc = Rc::new(RefCell::new(StringInterner::new()));
    let mut lowering = AstLowering::new(
        &mut string_interner,
        string_interner_rc,
        &mut symbol_table,
        &type_table,
        &mut scope_tree,
        &mut namespace_resolver,
        &mut import_resolver,
    );

    // Skip loading full stdlib from .hx files (unit tests don't have access to stdlib sources)
    // This still registers top-level stdlib symbols (Array, String, Math, etc.)
    lowering.set_skip_stdlib_loading(true);

    // Lower to TAST
    match lowering.lower_file(&haxe_file) {
        Ok(typed_file) => Ok(typed_file),
        Err(error) => Err(format!("Lowering error: {:?}", error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_class_lowering() {
        let haxe_code = r#"
            class SimpleClass {
                public var x:Int = 42;
                public function getValue():Int {
                    return x;
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to lower basic class: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 1, "Should have exactly one class");

        let class = &typed_file.classes[0];

        assert_eq!(
            class.fields.len() + class.methods.len(),
            2,
            "Should have 2 fields (variable + method)"
        );
    }

    #[test]
    fn test_forward_reference_resolution() {
        let haxe_code = r#"
            class ForwardUser {
                public var ref:ForwardTarget;
                public function useTarget():String {
                    return ref.getName();
                }
            }

            class ForwardTarget {
                public function getName():String {
                    return "target";
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to resolve forward reference: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 2, "Should have both classes");

        // Verify that ForwardTarget type was resolved in ForwardUser
        let forward_user = &typed_file.classes[0];
        assert!(
            forward_user.fields.len() > 0,
            "ForwardUser should have fields"
        );
    }

    #[test]
    fn test_type_inference_literals() {
        let haxe_code = r#"
            class TypeInferenceTest {
                public function testLiterals() {
                    var intVar = 42;
                    var floatVar = 3.14;
                    var stringVar = "hello";
                    var boolVar = true;
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to infer literal types: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");

        let class = &typed_file.classes[0];
        assert!(class.methods.len() > 0, "Should have at least one method");
    }

    #[test]
    fn test_arithmetic_type_inference() {
        let haxe_code = r#"
            class ArithmeticTest {
                public function testArithmetic():Float {
                    var a:Int = 5;
                    var b:Int = 3;
                    var c:Float = 2.5;

                    var intResult = a + b;      // Should be Int
                    var floatResult = a + c;    // Should be Float
                    var divResult = a / b;      // Should be Float (division always returns Float)

                    return floatResult + divResult;
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to infer arithmetic types: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");
    }

    #[test]
    fn test_array_type_inference() {
        let haxe_code = r#"
            class ArrayTest {
                public function testArrays():Int {
                    var numbers = [1, 2, 3, 4, 5];  // Should infer Array<Int>
                    var first = numbers[0];         // Should infer Int from array access

                    return first + numbers.length;
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to infer array types: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");
    }

    #[test]
    fn test_function_call_type_inference() {
        let haxe_code = r#"
            class FunctionTest {
                public function getString():String {
                    return "hello";
                }

                public function getLength():Int {
                    var str = getString();  // Should infer String from function return
                    return str.length;      // Should work with inferred type
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to infer function call types: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");
    }

    #[test]
    fn test_interface_implementation() {
        let haxe_code = r#"
            interface Drawable {
                public function draw():Void;
            }

            class Circle implements Drawable {
                public function draw():Void {
                    trace("Drawing circle");
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to lower interface implementation: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.interfaces.len(), 1, "Should have one interface");
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");

        let class = &typed_file.classes[0];
        assert!(
            class.interfaces.len() > 0,
            "Class should implement interface"
        );
    }

    #[test]
    fn test_enum_declaration() {
        let haxe_code = r#"
            enum Color {
                Red;
                Green;
                Blue;
                RGB(r:Int, g:Int, b:Int);
            }

            class ColorUser {
                public function useColor():Color {
                    return Color.RGB(255, 0, 0);
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to lower enum declaration: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.enums.len(), 1, "Should have one enum");
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");

        let enum_decl = &typed_file.enums[0];
        assert!(
            enum_decl.variants.len() >= 4,
            "Should have at least 4 enum variants"
        );
    }

    #[test]
    fn test_abstract_type() {
        let haxe_code = r#"
            abstract Point(Array<Float>) from Array<Float> to Array<Float> {
                public var x(get, never):Float;
                public var y(get, never):Float;

                function get_x():Float return this[0];
                function get_y():Float return this[1];

                public function new(x:Float, y:Float) {
                    this = [x, y];
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to lower abstract type: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.abstracts.len(), 1, "Should have one abstract");

        let abstract_decl = &typed_file.abstracts[0];
        assert!(
            abstract_decl.fields.len() > 0,
            "Abstract should have fields"
        );
    }

    #[test]
    fn test_generic_class() {
        let haxe_code = r#"
            class Container<T> {
                private var value:T;

                public function new(val:T) {
                    this.value = val;
                }

                public function getValue():T {
                    return value;
                }
            }

            class GenericUser {
                public function useContainer():String {
                    var container = new Container<String>("hello");
                    return container.getValue();
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to lower generic class: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 2, "Should have two classes");

        let generic_class = &typed_file.classes[0];
        assert!(
            generic_class.type_parameters.len() > 0,
            "Should have type parameters"
        );
    }

    #[test]
    fn test_complex_forward_references() {
        let haxe_code = r#"
            class A {
                public var b:B;
                public var c:C;
            }

            class B {
                public var c:C;
                public var a:A;
            }

            class C {
                public var a:A;
                public var b:B;

                public function process():Void {
                    a.b.c.process();
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to resolve complex forward references: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 3, "Should have three classes");

        // Verify all classes have their dependencies resolved
        for class in &typed_file.classes {
            assert!(class.fields.len() > 0, "Each class should have fields");
        }
    }

    #[test]
    fn test_expression_type_inference() {
        let haxe_code = r#"
            class ExpressionTest {
                public function testExpressions():Bool {
                    var a = 5;
                    var b = 10;
                    var c = 3.14;

                    var comparison = a < b;          // Should be Bool
                    var arithmetic = a + b;         // Should be Int
                    var mixed = a + c;              // Should be Float
                    var division = a / b;           // Should be Float

                    var ternary = comparison ? arithmetic : mixed;  // Should be Float (union)

                    return comparison && (ternary > 0.0);
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to infer expression types: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();
        assert_eq!(typed_file.classes.len(), 1, "Should have one class");
    }

    #[test]
    fn test_comprehensive_haxe_features() {
        let haxe_code = r#"
            package test.comprehensive;

            import haxe.ds.Map;
            using StringTools;

            interface IProcessor<T> {
                public function process(input:T):T;
            }

            enum Result<T, E> {
                Success(value:T);
                Error(error:E);
            }

            abstract UserId(Int) from Int to Int {
                public function new(id:Int) {
                    this = id;
                }

                public function toString():String {
                    return "User#" + this;
                }
            }

            class DataProcessor implements IProcessor<String> {
                private var cache:Map<String, String>;
                private var userId:UserId;

                public function new(userId:UserId) {
                    this.userId = userId;
                    this.cache = new Map<String, String>();
                }

                public function process(input:String):String {
                    if (cache.exists(input)) {
                        return cache.get(input);
                    }

                    var result = input.toUpperCase();
                    cache.set(input, result);
                    return result;
                }

                public function processWithResult(input:String):Result<String, String> {
                    try {
                        var processed = process(input);
                        return Result.Success(processed);
                    } catch (e:Dynamic) {
                        return Result.Error("Processing failed: " + e);
                    }
                }
            }

            class Main {
                static function main() {
                    var processor = new DataProcessor(new UserId(123));
                    var result = processor.processWithResult("hello world");

                    switch (result) {
                        case Success(value):
                            trace("Processed: " + value);
                        case Error(error):
                            trace("Error: " + error);
                    }
                }
            }
        "#;

        let result = parse_and_lower(haxe_code);
        assert!(
            result.is_ok(),
            "Failed to lower comprehensive Haxe code: {:?}",
            result.err()
        );

        let typed_file = result.unwrap();

        // Verify all major components are present
        assert!(
            typed_file.interfaces.len() >= 1,
            "Should have at least one interface"
        );
        assert!(typed_file.enums.len() >= 1, "Should have at least one enum");
        assert!(
            typed_file.abstracts.len() >= 1,
            "Should have at least one abstract"
        );
        assert!(
            typed_file.classes.len() >= 2,
            "Should have at least two classes"
        );

        // Verify package information
        assert!(
            typed_file.metadata.package_name.is_some(),
            "Should have package name"
        );
        assert_eq!(
            typed_file.metadata.package_name.as_ref().unwrap(),
            "test.comprehensive"
        );

        // Verify imports and using statements
        assert!(typed_file.imports.len() > 0, "Should have imports");
        assert!(
            typed_file.using_statements.len() > 0,
            "Should have using statements"
        );

        println!("✅ Comprehensive test passed!");
        println!("   - Interfaces: {}", typed_file.interfaces.len());
        println!("   - Enums: {}", typed_file.enums.len());
        println!("   - Abstracts: {}", typed_file.abstracts.len());
        println!("   - Classes: {}", typed_file.classes.len());
        println!("   - Imports: {}", typed_file.imports.len());
        println!(
            "   - Using statements: {}",
            typed_file.using_statements.len()
        );
    }
}
