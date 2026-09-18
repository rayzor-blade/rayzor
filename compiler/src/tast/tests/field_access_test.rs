//! Test field access resolution

#[cfg(test)]
mod tests {
    use crate::tast::{AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable};
    use diagnostics::SourceMap;
    use parser::parse_haxe_file;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_field_access_resolution() {
        let haxe_code = r#"
class TestFieldAccess {
    var name:String;

    public function new() {
        name = "test";  // This should work now with field resolution
    }

    public function getName():String {
        return name;  // This should also work
    }
}
"#;

        // Parse
        let ast_result = parse_haxe_file("field_access_test.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id =
            source_map.add_file("field_access_test.hx".to_string(), haxe_code.to_string());

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

        // Check that lowering succeeds without field resolution errors
        match typed_file_result {
            Ok(typed_file) => {
                println!(
                    "Successfully lowered file with {} classes",
                    typed_file.classes.len()
                );

                // Check for any errors - this test should have no field resolution errors
                let errors = lowering.get_errors();

                if !errors.is_empty() {
                    println!("Errors found during lowering:");
                    for (i, error) in errors.iter().enumerate() {
                        println!("  {}: {:?}", i + 1, error);
                    }

                    // Check that there are no UnresolvedSymbol errors for "name"
                    let has_field_error = errors.iter().any(|e| {
                        if let crate::tast::ast_lowering::LoweringError::UnresolvedSymbol {
                            name,
                            ..
                        } = e
                        {
                            name == "name"
                        } else {
                            false
                        }
                    });

                    if has_field_error {
                        panic!(
                            "Found UnresolvedSymbol error for field 'name' - field resolution not working"
                        );
                    }

                    println!("No field resolution errors found - test passed!");
                } else {
                    println!("✅ No errors found - test passed!");
                }
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
}
