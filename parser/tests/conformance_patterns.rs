#[test]
fn grouped_extractors_keep_the_pattern_structure() {
    let source = "class Main { static function main() { switch (1) { case (_.equals(2) => true) | (_.equals(3) => false): } } }";
    let file = parser::rd::rd_parse(source, "Main.hx", false, false).unwrap();
    let text = format!("{file:?}");
    assert!(text.contains("Or([Extractor"), "{text}");
    assert_eq!(text.matches("Extractor {").count(), 2);
}

#[test]
fn grouped_typed_pattern_remains_a_type_pattern() {
    let source = "class Main { static function main() { switch (null) { case (s:String): } } }";
    let file = parser::rd::rd_parse(source, "Main.hx", false, false).unwrap();
    assert!(format!("{file:?}").contains("Type { var: \"s\""));
}

#[test]
fn initialized_properties_keep_accessors_in_both_parsers() {
    use parser::{ClassFieldKind, ExprKind, PropertyAccess, TypeDeclaration};
    let source = "class Main { var value(get, set):Int = 7; }";
    let rd = parser::rd::rd_parse(source, "Main.hx", false, false).unwrap();
    let (_, legacy) = parser::haxe_parser::haxe_file("Main.hx", source, source).unwrap();
    for file in [rd, legacy] {
        let TypeDeclaration::Class(class) = &file.declarations[0] else {
            panic!("expected class");
        };
        let ClassFieldKind::Property {
            getter,
            setter,
            expr: Some(expr),
            ..
        } = &class.fields[0].kind
        else {
            panic!("initialized property lost its accessors or initializer");
        };
        assert_eq!(getter, &PropertyAccess::Custom("get".into()));
        assert_eq!(setter, &PropertyAccess::Custom("set".into()));
        assert!(matches!(expr.kind, ExprKind::Int(7)));
    }
}
