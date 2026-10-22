use parser::{ClassFieldKind, ExprKind, TypeDeclaration};

#[test]
fn inline_markup_is_verbatim_metadata_with_source_positions() {
    let markup = r#"<div title=\"a > b\">don't & 🦀 <!-- <ignored/> --> {f({x: \"}\"})}<text>${name}</text><br/></div>"#.replace("\\\"", "\"");
    let source = format!("class Main {{ static function view() return {markup}; }}");
    let file = parser::rd::rd_parse(&source, "Main.hx", false, false).unwrap();
    let TypeDeclaration::Class(class) = &file.declarations[0] else {
        panic!()
    };
    let ClassFieldKind::Function(function) = &class.fields[0].kind else {
        panic!()
    };
    let ExprKind::Return(Some(value)) = &function.body.as_ref().unwrap().kind else {
        panic!()
    };
    let ExprKind::Meta { meta, expr } = &value.kind else {
        panic!()
    };
    assert_eq!(meta.name, "markup");
    assert!(meta.compile_time);
    assert_eq!(expr.kind, ExprKind::String(markup.clone()));
    assert_eq!(&source[value.span.start..value.span.end], markup);
}

#[test]
fn markup_does_not_consume_comparisons_or_type_arguments() {
    parser::rd::rd_parse(
        "class Main { var a:Array<Int>; function f(a:Int,b:Int) { return a<b && b>0; } }",
        "Main.hx",
        false,
        false,
    )
    .unwrap();
    assert!(
        parser::rd::rd_parse(
            "class Main { function f() return <a><b/>; }",
            "Main.hx",
            false,
            false
        )
        .is_err()
    );
}

#[test]
fn markup_contents_are_validated_by_the_macro() {
    parser::rd::rd_parse(
        "class Main { function f() return <a><if {true}>yes<else>no</if><a>nested</a></a>; }",
        "Main.hx",
        false,
        false,
    )
    .unwrap();
}

#[test]
fn ternary_branches_accept_inline_markup() {
    let source = r#"class Main {
        static function view(flag:Bool) {
            var label = flag ? <text>Label</text> : <div class="row"><icon/></div>;
            return label;
        }
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
    let parser::BlockElement::Expr(declaration) = &statements[0] else {
        panic!()
    };
    let ExprKind::Var {
        expr: Some(value), ..
    } = &declaration.kind
    else {
        panic!()
    };
    let ExprKind::Ternary {
        then_expr,
        else_expr,
        ..
    } = &value.kind
    else {
        panic!()
    };
    assert!(matches!(&then_expr.kind, ExprKind::Meta { meta, .. } if meta.name == "markup"));
    assert!(matches!(&else_expr.kind, ExprKind::Meta { meta, .. } if meta.name == "markup"));
}

#[test]
fn control_bodies_accept_markup_without_consuming_parenthesized_comparisons() {
    let source = r#"class Main {
        static function view(items:Array<Int>, flag:Bool) {
            var first = [for (item in items) <text>${item}</text>];
            var second = [for (index => item in items) <text>${index}: ${item}</text>];
            var filtered = [for (item in items) if (check((item))) <text>${item}</text>];
            var choice = if (flag) <text>yes</text> else <text>no</text>;
            while (check((flag))) <text>waiting</text>;
            var comparison = (items.length)<first.length;
            return first;
        }
    }"#;
    let tokens = parser::lexer::Lexer::new(source).tokenize().unwrap();
    assert_eq!(
        tokens
            .iter()
            .filter(|token| token.kind == parser::token::TokenKind::MarkupLit)
            .count(),
        6
    );
    assert!(
        tokens
            .iter()
            .any(|token| token.kind == parser::token::TokenKind::Lt)
    );
    parser::rd::rd_parse(source, "Main.hx", false, false).unwrap();
}
