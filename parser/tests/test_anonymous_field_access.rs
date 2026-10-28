use parser::{PropertyAccess, Type, TypeDeclaration};

#[test]
fn anonymous_types_preserve_final_and_property_access() {
    let source = "typedef View = { final value:Int; var readable(default, null):String; var writable(null, default):Int; };";
    let file = parser::rd::rd_parse(source, "View.hx", false, false).unwrap();
    let TypeDeclaration::Typedef(declaration) = &file.declarations[0] else {
        panic!("typedef")
    };
    let Type::Anonymous { fields, .. } = &declaration.type_def else {
        panic!("structure")
    };
    assert!(fields[0].access.is_final);
    assert_eq!(fields[1].access.getter, PropertyAccess::Default);
    assert_eq!(fields[1].access.setter, PropertyAccess::Null);
    assert_eq!(fields[2].access.getter, PropertyAccess::Null);
    assert_eq!(fields[2].access.setter, PropertyAccess::Default);
}
