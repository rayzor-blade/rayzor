//! Minimal TAST validation test using only supported parser features

#[cfg(test)]
mod tests {
    use crate::tast::{
        AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable, node::TypedFile,
    };
    use parser::parse_haxe_file;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_minimal_tast_validation() {
        let test_content = r#"package com.example;

import haxe.ds.StringMap;
using StringTools;

// Generic class with constraints
@:generic
@:final
class Container<T> {
    public var items:Array<T>;
    private static final MAX_SIZE:Int = 100;

    public function new() {
        items = [];
    }

    public function add(item:T):Void {
        if (items.length < MAX_SIZE) {
            items.push(item);
        }
    }
}

// Interface
interface Comparable<T> {
    function compareTo(other:T):Int;
}

// Enum
enum Option<T> {
    None;
    Some(value:T);
}

// Abstract type
@:forward
abstract SafeInt(Int) from Int to Int {
    inline public function new(i:Int) {
        this = i;
    }

    @:from static public function fromString(s:String):SafeInt {
        return new SafeInt(Std.parseInt(s));
    }
}

// Typedef
typedef Point2D = {
    var x:Float;
    var y:Float;
}

// Main class
class Main {
    static function main() {
        // Basic expressions
        var container = new Container<String>();
        container.add("test");

        // Array comprehension
        var squares = [for (i in 0...10) i * i];

        // Map comprehension
        var map = [for (i in 0...5) i => i * i];

        // Null coalescing
        var nullable:Null<String> = null;
        var value = nullable ?? "default";

        // String interpolation
        var message = 'Value is ${value}';

        // Regular expression
        var regex = ~/[a-z]+/i;

        // Do-while loop
        var i = 0;
        do {
            i++;
        } while (i < 10);

        // For-in with key-value
        var kvMap = ["a" => 1, "b" => 2];
        for (key => val in kvMap) {
            trace('$key: $val');
        }

        // Try-catch
        try {
            throw "error";
        } catch (e:String) {
            trace("Error: " + e);
        }
    }
}"#;

        let ast = match parse_haxe_file("test.hx", test_content, false) {
            Ok(ast) => ast,
            Err(e) => panic!("Failed to parse test file: {:?}", e),
        };

        // Create TAST lowering context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

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

        // Lower to TAST
        let typed_file = lowering
            .lower_file(&ast)
            .expect("Failed to lower minimal test file");

        // Validate basic structure
        assert!(!typed_file.imports.is_empty(), "Should have imports");
        assert!(
            !typed_file.using_statements.is_empty(),
            "Should have using statements"
        );
        assert!(!typed_file.classes.is_empty(), "Should have classes");
        assert!(!typed_file.interfaces.is_empty(), "Should have interfaces");
        assert!(!typed_file.enums.is_empty(), "Should have enums");
        assert!(!typed_file.abstracts.is_empty(), "Should have abstracts");
        assert!(!typed_file.type_aliases.is_empty(), "Should have typedefs");

        // Validate Container class
        let container = typed_file
            .classes
            .iter()
            .find(|c| string_interner.get(c.name).unwrap() == "Container")
            .expect("Container class not found");

        assert!(
            !container.type_parameters.is_empty(),
            "Container should have type parameters"
        );
        assert!(!container.fields.is_empty(), "Container should have fields");
        assert!(
            !container.methods.is_empty(),
            "Container should have methods"
        );

        println!("✅ All minimal TAST validations passed!");
    }
}
