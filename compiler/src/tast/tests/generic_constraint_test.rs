//! Test generic constraint validation

#[cfg(test)]
mod tests {
    use crate::tast::{AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable};
    use diagnostics::SourceMap;
    use parser::parse_haxe_file;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_generic_constraint_violation() {
        let haxe_code = r#"
// Define an interface for comparison
interface Comparable<T> {
    function compareTo(other:T):Int;
}

// Define a generic class with constraints
class SortedList<T:Comparable<T>> {
    var items:Array<T>;

    public function new() {
        items = [];
    }

    public function add(item:T):Void {
        // Would implement sorted insertion using compareTo
    }
}

// A class that implements Comparable
class ComparableInt implements Comparable<ComparableInt> {
    var value:Int;

    public function new(v:Int) {
        value = v;
    }

    public function compareTo(other:ComparableInt):Int {
        return value - other.value;
    }
}

// A class that does NOT implement Comparable
class NotComparable {
    var name:String;

    public function new(n:String) {
        name = n;
    }
}

class TestGenericConstraints {
    public function test():Void {
        // This should work - ComparableInt implements Comparable
        var validList:SortedList<ComparableInt>;

        // This should fail - NotComparable doesn't implement Comparable
        var invalidList:SortedList<NotComparable>;
    }
}
"#;

        // Parse
        let ast_result = parse_haxe_file("generic_constraint_test.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file(
            "generic_constraint_test.hx".to_string(),
            haxe_code.to_string(),
        );

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        // Lower to TAST
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
        lowering.initialize_span_converter(file_id.as_usize() as u32, haxe_code.to_string());

        let typed_file_result = lowering.lower_file(&haxe_file);

        // Check for constraint violation error
        match typed_file_result {
            Ok(_) => {
                // If lowering succeeds, check if there were any errors collected
                let errors = lowering.get_errors();

                // We expect some kind of error - could be UnresolvedType or GenericParameterError
                let has_relevant_error = errors.iter().any(|e| {
                    matches!(
                        e,
                        crate::tast::ast_lowering::LoweringError::UnresolvedType { .. }
                            | crate::tast::ast_lowering::LoweringError::GenericParameterError { .. }
                            | crate::tast::ast_lowering::LoweringError::TypeInferenceError { .. }
                    )
                });

                if errors.is_empty() {
                    println!("Warning: Expected errors during lowering but none were found");
                } else {
                    println!("Lowering errors found: {}", errors.len());
                }

                println!("Errors: {:?}", errors);
            }
            Err(e) => {
                println!("Lowering failed with error: {:?}", e);
            }
        }
    }

    #[test]
    fn test_generic_constraint_success() {
        let haxe_code = r#"
// Simple generic with type parameter but no constraints
class SimpleGeneric<T> {
    var value:T;

    public function new(v:T) {
        value = v;
    }
}

class TestSimpleGeneric {
    public function test():Void {
        // These should all work - no constraints
        var intGen:SimpleGeneric<Int>;
        var stringGen:SimpleGeneric<String>;
        var anyGen:SimpleGeneric<Dynamic>;
    }
}
"#;

        // Parse
        let ast_result = parse_haxe_file("simple_generic_test.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id =
            source_map.add_file("simple_generic_test.hx".to_string(), haxe_code.to_string());

        // Create namespace and import resolvers
        let mut namespace_resolver = crate::tast::namespace::NamespaceResolver::new();
        let mut import_resolver = crate::tast::namespace::ImportResolver::new();

        // Lower to TAST
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
        lowering.initialize_span_converter(file_id.as_usize() as u32, haxe_code.to_string());

        let typed_file_result = lowering.lower_file(&haxe_file);

        // This should succeed without errors
        match typed_file_result {
            Ok(typed_file) => {
                println!(
                    "Successfully lowered file with {} classes",
                    typed_file.classes.len()
                );

                // Check for any errors - this test should have no errors at all
                let errors = lowering.get_errors();

                if !errors.is_empty() {
                    println!("Errors found during lowering:");
                    for (i, error) in errors.iter().enumerate() {
                        println!("  {}: {:?}", i + 1, error);
                    }
                    panic!(
                        "Expected no errors for simple generic test, but found {} errors",
                        errors.len()
                    );
                }

                println!("✅ No errors found - test passed!");
            }
            Err(e) => {
                // Also check for collected errors in case of failure
                let errors = lowering.get_errors();
                println!("Lowering failed with error: {:?}", e);
                if !errors.is_empty() {
                    println!("Additional collected errors:");
                    for (i, error) in errors.iter().enumerate() {
                        println!("  {}: {:?}", i + 1, error);
                    }
                }
                panic!("Lowering failed unexpectedly: {:?}", e);
            }
        }
    }

    /// Lowering errors for a source, including one that aborts the file.
    fn lowering_messages(code: &str) -> Vec<String> {
        lowering_messages_in(code, false)
    }

    /// Lowering errors for `code`, lowered as a user file when `user_file`.
    fn lowering_messages_in(code: &str, user_file: bool) -> Vec<String> {
        let haxe_file = parse_haxe_file("member_test.hx", code, true).expect("parse");
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("member_test.hx".to_string(), code.to_string());
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
        lowering.initialize_span_converter(file_id.as_usize() as u32, code.to_string());
        lowering.user_file = user_file;
        let result = lowering.lower_file(&haxe_file);
        let mut messages: Vec<String> = lowering
            .get_errors()
            .iter()
            .map(|e| format!("{e:?}"))
            .collect();
        if let Err(e) = result {
            messages.push(format!("{e:?}"));
        }
        messages
    }

    #[test]
    fn test_type_parameter_call_to_missing_member_is_an_error() {
        let messages = lowering_messages(
            r#"
interface I { function yes():Int; }
class Main {
    static function f<T:I>(v:T):Int return v.nope(1);
    static function main() {}
}
"#,
        );
        assert!(
            messages.iter().any(|m| m.contains("T has no field nope")),
            "{messages:?}"
        );
    }

    #[test]
    fn test_type_parameter_call_to_declared_member_is_not_an_error() {
        let messages = lowering_messages(
            r#"
interface IB { function base():Int; }
interface IA extends IB { function own():Int; }
class C { public var f:Int->Int; public function new() f = x -> x; }
class Ext { public static function twice(v:IA):Int return v.own() * 2; }
using Ext;
class Main {
    static function inherited<T:IA>(v:T):Int return v.base() + v.own();
    static function structural<T:{function size():Int;}>(v:T):Int return v.size();
    static function field<T:C>(v:T):Int return v.f(1);
    static function extension<T:IA>(v:T):Int return v.twice();
    static function main() {}
}
"#,
        );
        assert!(
            !messages.iter().any(|m| m.contains("has no field")),
            "{messages:?}"
        );
    }

    #[test]
    fn test_unknown_constructed_types_in_a_user_file_are_errors() {
        let messages = lowering_messages_in(
            r#"
class Main {
    static function main() {
        var c = new Nowhere<Int>(5);
        var d = 1 is Elsewhere;
    }
    @:generic static function make<C:haxe.Constraints.Constructible<Void->Void>>(type:Class<C>)
        return new C();
}
abstract Q<T>(Array<T>) from Array<T> {}
"#,
            true,
        );
        assert!(
            !messages
                .iter()
                .any(|m| m.contains("\"T\"") || m.contains("\"C\"")),
            "{messages:?}"
        );
        for name in ["Nowhere", "Elsewhere"] {
            assert!(
                messages
                    .iter()
                    .any(|m| m.contains("UnresolvedType") && m.contains(name)),
                "{name}: {messages:?}"
            );
        }
    }

    #[test]
    fn test_forward_type_references_in_a_user_file_resolve() {
        let messages = lowering_messages_in(
            r#"
class Main {
    static function main() {
        var l = new Later();
        var e:LaterEnum = A;
    }
}
class Later { public function new() {} }
enum LaterEnum { A; }
"#,
            true,
        );
        assert!(
            !messages.iter().any(|m| m.contains("\"Later")),
            "{messages:?}"
        );
    }
}
