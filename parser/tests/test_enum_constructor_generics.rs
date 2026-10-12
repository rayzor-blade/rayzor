use parser::{Type, TypeDeclaration};

#[test]
fn constructor_parameters_keep_their_scope_and_constraints() {
    let source = r#"
enum Wrapped<T> {
    Wrap<C:{value:Int}>(payload:Array<C>):Wrapped<C>;
    Other<C>(payload:C):Wrapped<C>;
}
"#;
    let files = [
        parser::parse_haxe_file("Wrapped.hx", source, false).unwrap(),
        parser::rd::rd_parse(source, "Wrapped.hx", false, false).unwrap(),
    ];
    for file in files {
        let TypeDeclaration::Enum(declaration) = &file.declarations[0] else {
            panic!("expected enum");
        };
        assert_eq!(declaration.type_params[0].name, "T");
        let constructor = &declaration.constructors[0];
        assert_eq!(constructor.type_params.len(), 1);
        assert_eq!(constructor.type_params[0].name, "C");
        let Type::Anonymous { fields, .. } = &constructor.type_params[0].constraints[0] else {
            panic!("expected structural constraint");
        };
        assert_eq!(fields[0].name, "value");
        assert_path(&fields[0].type_hint, "Int", &[]);
        assert_path(
            constructor.params[0].type_hint.as_ref().unwrap(),
            "Array",
            &["C"],
        );
        assert_path(constructor.return_type.as_ref().unwrap(), "Wrapped", &["C"]);
        let other = &declaration.constructors[1];
        assert_eq!(other.type_params[0].name, "C");
        assert!(other.type_params[0].constraints.is_empty());
        assert_path(other.params[0].type_hint.as_ref().unwrap(), "C", &[]);
    }
}

fn assert_path(ty: &Type, name: &str, parameters: &[&str]) {
    let Type::Path { path, params, .. } = ty else {
        panic!("expected type path: {ty:?}");
    };
    assert_eq!(path.name, name);
    assert_eq!(params.len(), parameters.len());
    for (parameter, expected) in params.iter().zip(parameters) {
        assert_path(parameter, expected, &[]);
    }
}
