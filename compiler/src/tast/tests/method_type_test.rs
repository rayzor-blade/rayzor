//! Test method type resolution

#[cfg(test)]
mod tests {
    use crate::tast::{
        AstLowering, ScopeId, ScopeTree, StringInterner, SymbolTable, TypeTable,
        type_checking_pipeline::TypeCheckingPhase,
    };
    use diagnostics::{Diagnostics, ErrorFormatter, SourceMap};
    use parser::parse_haxe_file;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_method_type_resolution() {
        let haxe_code = r#"
class MethodTest {
    public function calculate(x:Int, y:Int):Int {
        return x + y;
    }

    public function test():Void {
        var result = calculate(10, 20);  // This should resolve to Int type
    }
}
"#;

        // Parse
        let ast_result = parse_haxe_file("method_test.hx", haxe_code, true);
        let haxe_file = ast_result.expect("Parse should succeed");

        // Create context
        let mut string_interner = StringInterner::new();
        let mut symbol_table = SymbolTable::new();
        let type_table = Rc::new(RefCell::new(TypeTable::new()));
        let mut scope_tree = ScopeTree::new(ScopeId::first());
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file("method_test.hx".to_string(), haxe_code.to_string());

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
        let typed_file = lowering
            .lower_file(&haxe_file)
            .expect("Lowering should succeed");

        // Find the test method and check the result variable type
        let test_class = &typed_file.classes[0];
        let test_method = test_class
            .methods
            .iter()
            .find(|m| string_interner.get(m.name).unwrap_or("") == "test")
            .expect("Should find test method");

        // The body contains a Block expression
        if let Some(crate::tast::node::TypedStatement::Expression { expression, .. }) =
            test_method.body.get(0)
        {
            if let crate::tast::node::TypedExpressionKind::Block { statements, .. } =
                &expression.kind
            {
                // Find the variable declaration in the block
                if let Some(crate::tast::node::TypedStatement::VarDeclaration {
                    initializer,
                    var_type,
                    ..
                }) = statements.get(0)
                {
                    if let Some(init_expr) = initializer {
                        // Check if the initializer is a method call
                        if let crate::tast::node::TypedExpressionKind::MethodCall { .. } =
                            &init_expr.kind
                        {
                            // Check the type of the method call expression
                            let type_name = match type_table.borrow().get(init_expr.expr_type) {
                                Some(type_info) => {
                                    use crate::tast::core::TypeKind;
                                    match &type_info.kind {
                                        TypeKind::Int => "Int",
                                        TypeKind::Dynamic => "Dynamic",
                                        TypeKind::Unknown => "Unknown",
                                        TypeKind::Void => "Void",
                                        _ => "Other",
                                    }
                                }
                                None => "Invalid",
                            };

                            println!("Method call result type: {}", type_name);
                            assert_eq!(
                                type_name, "Int",
                                "Method call should resolve to Int type, not {}",
                                type_name
                            );
                            return;
                        }
                    }
                }
            }
        }

        panic!("Could not find method call in test method body");
    }
}
