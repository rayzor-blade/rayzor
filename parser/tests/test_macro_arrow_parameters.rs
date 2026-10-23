use parser::{ClassFieldKind, ExprKind, TypeDeclaration};

#[test]
fn quoted_arrows_accept_spliced_parameter_names_and_types() {
    for value in [
        "($name:$hint) -> $body",
        "$name -> $body",
        "($name) -> $body",
        "($left:Int, $right:$hint) -> $body",
    ] {
        let source = format!("class Builder {{ static function make() return macro {value}; }}");
        let file = parser::rd::rd_parse(&source, "Builder.hx", false, false).unwrap();
        let TypeDeclaration::Class(class) = &file.declarations[0] else {
            panic!()
        };
        let ClassFieldKind::Function(function) = &class.fields[0].kind else {
            panic!()
        };
        let ExprKind::Return(Some(body)) = &function.body.as_ref().unwrap().kind else {
            panic!()
        };
        let ExprKind::Macro(arrow) = &body.kind else {
            panic!()
        };
        let ExprKind::Arrow { params, .. } = &arrow.kind else {
            panic!("{arrow:?}")
        };
        assert!(params[0].name.starts_with('$'), "{value}: {params:?}");
    }
}

#[test]
#[ignore = "requires the sibling ashui workspace"]
fn ashui_hxx_parses_without_legacy_fallback() {
    let file =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ashui/haxe/ashui/ui/Hxx.hx");
    let source = std::fs::read_to_string(&file).unwrap();
    let mut config = parser::preprocessor::PreprocessorConfig::default();
    config.define("macro");
    let source = parser::preprocessor::preprocess(&source, &config);
    parser::rd::rd_parse(&source, file.to_str().unwrap(), false, false).unwrap();
}
