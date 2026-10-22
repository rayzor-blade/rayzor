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
