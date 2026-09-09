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
