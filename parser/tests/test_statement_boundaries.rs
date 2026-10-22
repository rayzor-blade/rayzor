use parser::{BlockElement, ClassFieldKind, ExprKind, TypeDeclaration};

#[test]
fn array_after_terminated_if_is_a_separate_statement() {
    let source = r#"class Main {
        static function value(flag:Bool) {
            if (flag) throw "invalid";
            [call(1, 2), call(flag ? 3 : 4, 5)];
        }
        static function arrow() return (flag:Bool) -> {
            if (flag) throw "invalid";
            [1, 2];
        };
    }"#;
    let file = parser::rd::rd_parse(source, "Main.hx", false, false).unwrap();
    let TypeDeclaration::Class(class) = &file.declarations[0] else {
        panic!()
    };
    let ClassFieldKind::Function(function) = &class.fields[0].kind else {
        panic!()
    };
    let ExprKind::Block(statements) = &function.body.as_ref().unwrap().kind else {
        panic!()
    };
    assert_eq!(statements.len(), 2);
    assert!(
        matches!(&statements[0], BlockElement::Expr(expr) if matches!(expr.kind, ExprKind::If { .. }))
    );
    assert!(
        matches!(&statements[1], BlockElement::Expr(expr) if matches!(&expr.kind, ExprKind::Array(items) if items.len() == 2))
    );
}

#[test]
fn semicolon_before_else_and_grouped_conditional_index_still_parse() {
    parser::rd::rd_parse(
        r#"class Main {
        static function value(flag:Bool) {
            if (flag) call(); else other();
            return (if (flag) [1] else [2])[0];
        }
    }"#,
        "Main.hx",
        false,
        false,
    )
    .unwrap();
}
