use parser::haxe_parser::parse_haxe_file_with_debug;

#[test]
fn test_catch_with_filter() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e:String) if (e.length > 0) {
            trace("Non-empty string error: " + e);
        }
    }
}
"#;

    match parse_haxe_file_with_debug("test.hx", input, true, true) {
        Ok(result) => {
            // Extract the try-catch from the AST
            if let Some(parser::haxe_ast::TypeDeclaration::Class(class)) =
                result.declarations.first()
            {
                {
                    if let Some(method) = class.fields.first()
                        && let parser::haxe_ast::ClassFieldKind::Function(func) = &method.kind
                        && let Some(body) = &func.body
                        && let parser::haxe_ast::ExprKind::Block(block) = &body.kind
                        && let Some(parser::haxe_ast::BlockElement::Expr(try_expr)) = block.first()
                        && let parser::haxe_ast::ExprKind::Try { catches, .. } = &try_expr.kind
                    {
                        assert_eq!(catches.len(), 1, "Should have 1 catch block");
                        let catch_block = &catches[0];
                        assert_eq!(catch_block.var, "e");
                        assert!(catch_block.type_hint.is_some());
                        assert!(catch_block.filter.is_some(), "Should have a filter");

                        // Check if the filter is a binary comparison
                        if let Some(filter) = &catch_block.filter {
                            // Handle parenthesized expressions - the filter is wrapped in parentheses
                            let actual_filter = match &filter.kind {
                                parser::haxe_ast::ExprKind::Paren(inner) => inner,
                                _ => filter,
                            };
                            match &actual_filter.kind {
                                parser::haxe_ast::ExprKind::Binary { left, op, right } => {
                                    assert_eq!(*op, parser::haxe_ast::BinaryOp::Gt);
                                    // left should be field access e.length
                                    match &left.kind {
                                        parser::haxe_ast::ExprKind::Field {
                                            expr, field, ..
                                        } => {
                                            assert_eq!(field, "length");
                                            match &expr.kind {
                                                parser::haxe_ast::ExprKind::Ident(name) => {
                                                    assert_eq!(name, "e");
                                                }
                                                _ => panic!("Expected identifier 'e'"),
                                            }
                                        }
                                        _ => panic!("Expected field access"),
                                    }
                                    // right should be literal 0
                                    match &right.kind {
                                        parser::haxe_ast::ExprKind::Int(val) => {
                                            assert_eq!(*val, 0);
                                        }
                                        _ => {
                                            panic!("Expected integer literal 0")
                                        }
                                    }
                                }
                                _ => panic!("Expected binary expression for filter"),
                            }
                        }
                        return;
                    }
                }
            }
            panic!("Could not find expected AST structure");
        }
        Err(e) => {
            panic!("Should parse successfully, got error: {}", e);
        }
    }
}

#[test]
fn test_catch_without_filter() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e:String) {
            trace("String error: " + e);
        }
    }
}
"#;

    match parse_haxe_file_with_debug("test.hx", input, true, true) {
        Ok(result) => {
            // Extract the try-catch from the AST
            if let Some(parser::haxe_ast::TypeDeclaration::Class(class)) =
                result.declarations.first()
            {
                {
                    if let Some(method) = class.fields.first()
                        && let parser::haxe_ast::ClassFieldKind::Function(func) = &method.kind
                        && let Some(body) = &func.body
                        && let parser::haxe_ast::ExprKind::Block(block) = &body.kind
                        && let Some(parser::haxe_ast::BlockElement::Expr(try_expr)) = block.first()
                        && let parser::haxe_ast::ExprKind::Try { catches, .. } = &try_expr.kind
                    {
                        assert_eq!(catches.len(), 1, "Should have 1 catch block");
                        let catch_block = &catches[0];
                        assert_eq!(catch_block.var, "e");
                        assert!(catch_block.type_hint.is_some());
                        assert!(catch_block.filter.is_none(), "Should not have a filter");
                        return;
                    }
                }
            }
            panic!("Could not find expected AST structure");
        }
        Err(e) => {
            panic!("Should parse successfully, got error: {}", e);
        }
    }
}

#[test]
fn test_multiple_catch_with_mixed_filters() {
    let input = r#"
class Test {
    function test() {
        try {
            riskyOperation();
        } catch (e:String) if (e.length > 0) {
            trace("Non-empty string error: " + e);
        } catch (e:Int) {
            trace("Any int error: " + e);
        } catch (e:Float) if (e > 0.0) {
            trace("Positive float error: " + e);
        }
    }
}
"#;

    match parse_haxe_file_with_debug("test.hx", input, true, true) {
        Ok(result) => {
            // Extract the try-catch from the AST
            if let Some(parser::haxe_ast::TypeDeclaration::Class(class)) =
                result.declarations.first()
            {
                {
                    if let Some(method) = class.fields.first()
                        && let parser::haxe_ast::ClassFieldKind::Function(func) = &method.kind
                        && let Some(body) = &func.body
                        && let parser::haxe_ast::ExprKind::Block(block) = &body.kind
                        && let Some(parser::haxe_ast::BlockElement::Expr(try_expr)) = block.first()
                        && let parser::haxe_ast::ExprKind::Try { catches, .. } = &try_expr.kind
                    {
                        assert_eq!(catches.len(), 3, "Should have 3 catch blocks");

                        // First catch: has filter
                        assert!(
                            catches[0].filter.is_some(),
                            "First catch should have filter"
                        );
                        assert_eq!(catches[0].var, "e");

                        // Second catch: no filter
                        assert!(
                            catches[1].filter.is_none(),
                            "Second catch should not have filter"
                        );
                        assert_eq!(catches[1].var, "e");

                        // Third catch: has filter
                        assert!(
                            catches[2].filter.is_some(),
                            "Third catch should have filter"
                        );
                        assert_eq!(catches[2].var, "e");

                        return;
                    }
                }
            }
            panic!("Could not find expected AST structure");
        }
        Err(e) => {
            panic!("Should parse successfully, got error: {}", e);
        }
    }
}
