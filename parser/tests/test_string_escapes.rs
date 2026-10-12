use parser::{ClassFieldKind, ExprKind, TypeDeclaration};

#[test]
fn both_parsers_decode_fixed_braced_and_hexadecimal_escapes() {
    let source = r#"
        class Escapes {
            var emoji = "\u{1F602}";
            var maximum = "\u{10FFFF}";
            var mixed = "\u{41}\u0042\x43";
            var quoted = 'emoji \u{1F602} \x70';
            var literal = "\\u{1F602}";
        }
    "#;
    let files = [
        parser::parse_haxe_file("Escapes.hx", source, false).unwrap(),
        parser::rd::rd_parse(source, "Escapes.hx", false, false).unwrap(),
    ];
    for file in files {
        let TypeDeclaration::Class(class) = &file.declarations[0] else {
            panic!("expected class");
        };
        assert_eq!(class.fields.len(), 5);
        for (field, expected) in
            class
                .fields
                .iter()
                .zip(["😂", "\u{10ffff}", "ABC", "emoji 😂 p", r"\u{1F602}"])
        {
            let ClassFieldKind::Var {
                expr: Some(expr), ..
            } = &field.kind
            else {
                panic!("expected field initializer");
            };
            let ExprKind::String(actual) = &expr.kind else {
                panic!("expected string initializer: {expr:?}");
            };
            assert_eq!(actual, expected, "{field:?}");
        }
    }
}
