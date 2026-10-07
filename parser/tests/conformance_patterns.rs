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

fn expressions_in_both_parsers(expression: &str) -> Vec<parser::Expr> {
    use parser::{ClassFieldKind, TypeDeclaration};
    let source = format!("class Main {{ static var value = {expression}; }}");
    let rd = parser::rd::rd_parse(&source, "Main.hx", false, false).unwrap();
    let (_, legacy) = parser::haxe_parser::haxe_file("Main.hx", &source, &source).unwrap();
    [rd, legacy]
        .into_iter()
        .map(|file| {
            let TypeDeclaration::Class(class) = &file.declarations[0] else {
                panic!("expected class");
            };
            let ClassFieldKind::Var {
                expr: Some(expr), ..
            } = &class.fields[0].kind
            else {
                panic!("expected initialized field");
            };
            expr.clone()
        })
        .collect()
}

#[test]
fn metadata_leaves_binary_precedence_in_both_parsers() {
    use parser::{BinaryOp, ExprKind};
    for expr in expressions_in_both_parsers("5 * @foo 3 + 4") {
        let ExprKind::Binary {
            left,
            op: BinaryOp::Add,
            right,
        } = expr.kind
        else {
            panic!("addition must remain outside metadata");
        };
        assert!(matches!(right.kind, ExprKind::Int(4)));
        let ExprKind::Binary {
            right,
            op: BinaryOp::Mul,
            ..
        } = left.kind
        else {
            panic!("multiplication must keep its precedence");
        };
        let ExprKind::Meta { meta, expr } = right.kind else {
            panic!("expected metadata on the operand");
        };
        assert_eq!(meta.name, "foo");
        assert!(matches!(expr.kind, ExprKind::Int(3)));
    }
}

#[test]
fn metadata_wraps_assignments_in_both_parsers() {
    use parser::{AssignOp, ExprKind};
    for (source, expected_op) in [
        ("@foo @bar x = 1", AssignOp::Assign),
        ("@foo @bar x += 1", AssignOp::AddAssign),
    ] {
        for expr in expressions_in_both_parsers(source) {
            let ExprKind::Meta { meta, expr } = expr.kind else {
                panic!("assignment lost outer metadata");
            };
            assert_eq!(meta.name, "foo");
            let ExprKind::Meta { meta, expr } = expr.kind else {
                panic!("assignment lost inner metadata");
            };
            assert_eq!(meta.name, "bar");
            let ExprKind::Assign { left, op, right } = expr.kind else {
                panic!("metadata must cover the complete assignment");
            };
            assert_eq!(op, expected_op);
            assert!(matches!(left.kind, ExprKind::Ident(name) if name == "x"));
            assert!(matches!(right.kind, ExprKind::Int(1)));
        }
    }
}
