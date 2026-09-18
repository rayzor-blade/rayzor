use parser::{haxe_ast::*, parse_haxe_file};

#[test]
fn test_typedef_intersection_basic() {
    let input = r#"
typedef ExtendedPoint = Point & {
    var z:Float;
};
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(ast) => {
            assert_eq!(ast.declarations.len(), 1);

            if let TypeDeclaration::Typedef(typedef) = &ast.declarations[0] {
                assert_eq!(typedef.name, "ExtendedPoint");

                // Check that it's an intersection type
                if let Type::Intersection { left, right, .. } = &typedef.type_def {
                    // Left should be Point
                    if let Type::Path { path, .. } = &**left {
                        assert_eq!(path.name, "Point");
                    } else {
                        panic!("Expected Path type on left side");
                    }

                    // Right should be anonymous type with z field
                    if let Type::Anonymous { fields, .. } = &**right {
                        assert_eq!(fields.len(), 1);
                        assert_eq!(fields[0].name, "z");
                    } else {
                        panic!("Expected Anonymous type on right side");
                    }
                } else {
                    panic!("Expected Intersection type");
                }
            } else {
                panic!("Expected typedef declaration");
            }
        }
        Err(e) => {
            panic!("Parse failed: {}", e);
        }
    }
}

#[test]
fn test_typedef_intersection_multiple_fields() {
    let input = r#"
typedef User = {
    var id:Int;
    var name:String;
} & {
    var email:String;
    var ?phone:String;
    function validate():Bool;
};
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(ast) => {
            assert_eq!(ast.declarations.len(), 1);

            if let TypeDeclaration::Typedef(typedef) = &ast.declarations[0] {
                assert_eq!(typedef.name, "User");

                if let Type::Intersection { left, right, .. } = &typedef.type_def {
                    // Check left side anonymous type
                    if let Type::Anonymous { fields, .. } = &**left {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].name, "id");
                        assert_eq!(fields[1].name, "name");
                    }

                    // Check right side anonymous type
                    if let Type::Anonymous { fields, .. } = &**right {
                        assert_eq!(fields.len(), 3);
                        assert_eq!(fields[0].name, "email");
                        assert_eq!(fields[1].name, "phone");
                        assert!(fields[1].optional);
                        assert_eq!(fields[2].name, "validate");
                    }
                } else {
                    panic!("Expected Intersection type");
                }
            }
        }
        Err(e) => {
            panic!("Parse failed: {}", e);
        }
    }
}

#[test]
fn test_typedef_intersection_with_type_params() {
    let input = r#"
typedef Container<T> = Array<T> & {
    var capacity:Int;
    function isFull():Bool;
};
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(ast) => {
            assert_eq!(ast.declarations.len(), 1);

            if let TypeDeclaration::Typedef(typedef) = &ast.declarations[0] {
                assert_eq!(typedef.name, "Container");
                assert_eq!(typedef.type_params.len(), 1);
                assert_eq!(typedef.type_params[0].name, "T");

                if let Type::Intersection { left, right, .. } = &typedef.type_def {
                    // Left should be Array<T>
                    if let Type::Path { path, params, .. } = &**left {
                        assert_eq!(path.name, "Array");
                        assert_eq!(params.len(), 1);
                    }

                    // Right should be anonymous type
                    if let Type::Anonymous { fields, .. } = &**right {
                        assert_eq!(fields.len(), 2);
                    }
                }
            }
        }
        Err(e) => {
            panic!("Parse failed: {}", e);
        }
    }
}

#[test]
fn test_typedef_intersection_nested() {
    let input = r#"
typedef ComplexType = BaseType & {var x:Int;} & {var y:String;};
"#;

    // This should parse as (BaseType & {var x:Int;}) & {var y:String;}
    // Due to left-associativity
    match parse_haxe_file("test.hx", input, false) {
        Ok(ast) => {
            assert_eq!(ast.declarations.len(), 1);

            if let TypeDeclaration::Typedef(typedef) = &ast.declarations[0] {
                assert_eq!(typedef.name, "ComplexType");

                // Should be an intersection type
                if let Type::Intersection { .. } = &typedef.type_def {
                    // Success - we parsed the nested intersection
                } else {
                    panic!("Expected Intersection type");
                }
            }
        }
        Err(e) => {
            panic!("Parse failed: {}", e);
        }
    }
}

#[test]
fn test_typedef_intersection_with_metadata() {
    let input = r#"
@:native("ExtendedNative")
typedef Extended = Native & {
    @:optional var extra:String;
};
"#;

    match parse_haxe_file("test.hx", input, false) {
        Ok(ast) => {
            assert_eq!(ast.declarations.len(), 1);

            if let TypeDeclaration::Typedef(typedef) = &ast.declarations[0] {
                assert_eq!(typedef.name, "Extended");
                assert_eq!(typedef.meta.len(), 1);
                assert_eq!(typedef.meta[0].name, "native");

                if let Type::Intersection { right, .. } = &typedef.type_def
                    && let Type::Anonymous { fields, .. } = &**right
                {
                    assert!(fields[0].optional);
                }
            }
        }
        Err(e) => {
            panic!("Parse failed: {}", e);
        }
    }
}
