//! Test complex generic type constraints parsing and lowering

use crate::tast::{
    AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable, node::TypedFile,
};
use parser::parse_haxe_file;
use std::cell::RefCell;
use std::rc::Rc;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper to parse and lower Haxe code
    fn parse_and_lower(haxe_code: &str) -> Result<TypedFile, String> {
        // Parse the Haxe code
        let ast_result = parse_haxe_file("test.hx", haxe_code, true); // Enable recovery
        let haxe_file = match ast_result {
            Ok(file) => file,
            Err(errors) => return Err(format!("Parse errors: {:?}", errors)),
        };

        // Create lowering context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());

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

        // Lower to TAST
        match lowering.lower_file(&haxe_file) {
            Ok(typed_file) => Ok(typed_file),
            Err(error) => Err(format!("Lowering error: {:?}", error)),
        }
    }

    #[test]
    fn test_complex_generic_constraints() {
        let haxe_code = r#"
            // Define Comparable interface since it's not built-in
            interface Comparable<T> {
                public function compareTo(other:T):Int;
            }

            class GenericClass<T, U:Comparable<String>> {
                public function new() {}
                public function process<V>(input: V): T {
                    return null;
                }
            }
        "#;

        println!("🔍 Testing complex generic constraints...");

        let result = parse_and_lower(haxe_code);
        match &result {
            Ok(typed_file) => {
                println!("✅ Parse and lower succeeded!");
                println!("   Classes: {}", typed_file.classes.len());
                if !typed_file.classes.is_empty() {
                    let class = &typed_file.classes[0];
                    println!("   Type parameters: {}", class.type_parameters.len());
                }
            }
            Err(error) => {
                println!("❌ Parse and lower failed: {}", error);
            }
        }

        assert!(
            result.is_ok(),
            "Failed to parse/lower complex generics: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_intersection_type_constraints() {
        let haxe_code = r#"
            // Define Comparable interface since it's not built-in
            interface Comparable<T> {
                public function compareTo(other:T):Int;
            }

            class IntersectionClass<T, U:Iterable<String> & Comparable<Int>> {
                public function new() {}
            }
        "#;

        println!("🔍 Testing intersection type constraints...");

        let result = parse_and_lower(haxe_code);
        match &result {
            Ok(typed_file) => {
                println!("✅ Parse and lower succeeded!");
                println!("   Classes: {}", typed_file.classes.len());
                if !typed_file.classes.is_empty() {
                    let class = &typed_file.classes[0];
                    println!("   Type parameters: {}", class.type_parameters.len());
                    // Check if intersection type constraints are preserved
                    for (i, param) in class.type_parameters.iter().enumerate() {
                        println!(
                            "   Type param {}: {} constraints",
                            i,
                            param.constraints.len()
                        );
                    }
                }
            }
            Err(error) => {
                println!("❌ Parse and lower failed: {}", error);
            }
        }

        assert!(
            result.is_ok(),
            "Failed to parse/lower intersection type constraints: {:?}",
            result.err()
        );
    }
}
