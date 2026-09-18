//! Tests for dollar identifiers parsing

extern crate parser;
use parser::*;

#[test]
fn test_dollar_type_identifier() {
    let input = r#"
class Test {
    function test() {
        $type(expr);
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse $type identifier: {:?}",
        result.err()
    );

    let file = result.unwrap();
    assert_eq!(file.declarations.len(), 1);

    if let TypeDeclaration::Class(class) = &file.declarations[0] {
        assert_eq!(class.name, "Test");
        assert_eq!(class.fields.len(), 1);

        if let ClassFieldKind::Function(func) = &class.fields[0].kind {
            assert_eq!(func.name, "test");
            if let Some(body) = &func.body {
                if let ExprKind::Block(block) = &body.kind {
                    assert_eq!(block.len(), 1);
                    if let BlockElement::Expr(expr) = &block[0] {
                        if let ExprKind::Call {
                            expr: call_expr, ..
                        } = &expr.kind
                        {
                            if let ExprKind::DollarIdent { name, arg } = &call_expr.kind {
                                assert_eq!(name, "type");
                                assert!(arg.is_none());
                            } else {
                                panic!("Expected DollarIdent, got {:?}", call_expr.kind);
                            }
                        } else {
                            panic!("Expected Call, got {:?}", expr.kind);
                        }
                    } else {
                        panic!("Expected Expr, got {:?}", block[0]);
                    }
                } else {
                    panic!("Expected Block, got {:?}", body.kind);
                }
            } else {
                panic!("Expected function body");
            }
        } else {
            panic!("Expected Function, got {:?}", class.fields[0].kind);
        }
    } else {
        panic!("Expected Class, got {:?}", file.declarations[0]);
    }
}

#[test]
fn test_dollar_v_identifier() {
    let input = r#"
class Test {
    function test() {
        $v{someValue};
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse $v identifier: {:?}",
        result.err()
    );

    let file = result.unwrap();
    assert_eq!(file.declarations.len(), 1);

    if let TypeDeclaration::Class(class) = &file.declarations[0]
        && let ClassFieldKind::Function(func) = &class.fields[0].kind
        && let Some(body) = &func.body
        && let ExprKind::Block(block) = &body.kind
        && let BlockElement::Expr(expr) = &block[0]
    {
        if let ExprKind::DollarIdent { name, arg } = &expr.kind {
            assert_eq!(name, "v");
            assert!(arg.is_some());
            if let Some(arg_expr) = arg {
                if let ExprKind::Ident(ident) = &arg_expr.kind {
                    assert_eq!(ident, "someValue");
                } else {
                    panic!("Expected Ident, got {:?}", arg_expr.kind);
                }
            }
        } else {
            panic!("Expected DollarIdent, got {:?}", expr.kind);
        }
    }
}

#[test]
fn test_dollar_i_identifier() {
    let input = r#"
class Test {
    function test() {
        $i{fieldName};
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse $i identifier: {:?}",
        result.err()
    );

    let file = result.unwrap();
    if let TypeDeclaration::Class(class) = &file.declarations[0]
        && let ClassFieldKind::Function(func) = &class.fields[0].kind
        && let Some(body) = &func.body
        && let ExprKind::Block(block) = &body.kind
        && let BlockElement::Expr(expr) = &block[0]
        && let ExprKind::DollarIdent { name, arg } = &expr.kind
    {
        assert_eq!(name, "i");
        assert!(arg.is_some());
        if let Some(arg_expr) = arg
            && let ExprKind::Ident(ident) = &arg_expr.kind
        {
            assert_eq!(ident, "fieldName");
        }
    }
}

#[test]
fn test_dollar_a_identifier() {
    let input = r#"
class Test {
    function test() {
        $a{[expr1, expr2]};
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse $a identifier: {:?}",
        result.err()
    );

    let file = result.unwrap();
    if let TypeDeclaration::Class(class) = &file.declarations[0]
        && let ClassFieldKind::Function(func) = &class.fields[0].kind
        && let Some(body) = &func.body
        && let ExprKind::Block(block) = &body.kind
        && let BlockElement::Expr(expr) = &block[0]
        && let ExprKind::DollarIdent { name, arg } = &expr.kind
    {
        assert_eq!(name, "a");
        assert!(arg.is_some());
        if let Some(arg_expr) = arg
            && let ExprKind::Array(arr) = &arg_expr.kind
        {
            assert_eq!(arr.len(), 2);
            if let ExprKind::Ident(ident) = &arr[0].kind {
                assert_eq!(ident, "expr1");
            }
            if let ExprKind::Ident(ident) = &arr[1].kind {
                assert_eq!(ident, "expr2");
            }
        }
    }
}

#[test]
fn test_dollar_e_identifier() {
    let input = r#"
class Test {
    function test() {
        $e{myExpression};
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse $e identifier: {:?}",
        result.err()
    );

    let file = result.unwrap();
    if let TypeDeclaration::Class(class) = &file.declarations[0]
        && let ClassFieldKind::Function(func) = &class.fields[0].kind
        && let Some(body) = &func.body
        && let ExprKind::Block(block) = &body.kind
        && let BlockElement::Expr(expr) = &block[0]
        && let ExprKind::DollarIdent { name, arg } = &expr.kind
    {
        assert_eq!(name, "e");
        assert!(arg.is_some());
        if let Some(arg_expr) = arg
            && let ExprKind::Ident(ident) = &arg_expr.kind
        {
            assert_eq!(ident, "myExpression");
        }
    }
}

#[test]
fn test_regular_reification_still_works() {
    let input = r#"
class Test {
    function test() {
        return macro $someExpr;
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse regular reification: {:?}",
        result.err()
    );

    let file = result.unwrap();
    if let TypeDeclaration::Class(class) = &file.declarations[0]
        && let ClassFieldKind::Function(func) = &class.fields[0].kind
        && let Some(body) = &func.body
        && let ExprKind::Block(block) = &body.kind
        && let BlockElement::Expr(expr) = &block[0]
        && let ExprKind::Return(Some(return_expr)) = &expr.kind
        && let ExprKind::Macro(macro_expr) = &return_expr.kind
        && let ExprKind::Reify(reify_expr) = &macro_expr.kind
        && let ExprKind::Ident(ident) = &reify_expr.kind
    {
        assert_eq!(ident, "someExpr");
    }
}

#[test]
fn test_mixed_dollar_expressions() {
    let input = r#"
class Test {
    function test() {
        $type(expr);
        $v{value};
        var x = $someExpr;
    }
}
"#;

    let result = parse_haxe_file("test.hx", input, false);
    assert!(
        result.is_ok(),
        "Failed to parse mixed dollar expressions: {:?}",
        result.err()
    );

    let file = result.unwrap();
    if let TypeDeclaration::Class(class) = &file.declarations[0]
        && let ClassFieldKind::Function(func) = &class.fields[0].kind
        && let Some(body) = &func.body
        && let ExprKind::Block(block) = &body.kind
    {
        assert_eq!(block.len(), 3);

        // First: $type(expr) - should be a call to $type
        if let BlockElement::Expr(expr) = &block[0]
            && let ExprKind::Call {
                expr: call_expr, ..
            } = &expr.kind
            && let ExprKind::DollarIdent { name, arg } = &call_expr.kind
        {
            assert_eq!(name, "type");
            assert!(arg.is_none());
        }

        // Second: $v{value} - should be a dollar identifier
        if let BlockElement::Expr(expr) = &block[1]
            && let ExprKind::DollarIdent { name, arg } = &expr.kind
        {
            assert_eq!(name, "v");
            assert!(arg.is_some());
        }

        // Third: var x = $someExpr - should be a var declaration with reification
        if let BlockElement::Expr(expr) = &block[2]
            && let ExprKind::Var {
                name,
                expr: Some(var_expr),
                ..
            } = &expr.kind
        {
            assert_eq!(name, "x");
            if let ExprKind::Reify(reify_expr) = &var_expr.kind
                && let ExprKind::Ident(ident) = &reify_expr.kind
            {
                assert_eq!(ident, "someExpr");
            }
        }
    }
}
