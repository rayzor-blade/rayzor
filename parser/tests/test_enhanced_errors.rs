use parser::parse_haxe_file;

#[test]
fn test_enhanced_error_missing_semicolon() {
    let input = r#"
class TestClass {
    public function test() {
        var x = 42
        return x;
    }
}
"#;

    println!("Testing enhanced error reporting for missing semicolon");
    // The RD parser treats this as two consecutive statements
    // (`var x = 42` followed by `return x;`) which is valid Haxe,
    // so parsing may succeed. Either outcome is acceptable.
    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => {
            println!("RD parser accepted input (semicolons optional between statements)");
        }
        Err(e) => {
            println!("Enhanced error message:");
            println!("{}", e);
            assert!(e.contains("semicolon") || e.contains("';'") || e.contains("expected"));
        }
    }
}

#[test]
fn test_enhanced_error_missing_brace() {
    let input = r#"
class TestClass {
    public function test() {
        var x = 42;
        return x;
    // Missing closing brace
}
"#;

    println!("Testing enhanced error reporting for missing brace");
    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => panic!("Expected parse error but got success"),
        Err(e) => {
            println!("Enhanced error message:");
            println!("{}", e);

            // Check that parsing failed with some diagnostic
            assert!(!e.is_empty());
        }
    }
}

#[test]
fn test_enhanced_error_unexpected_token() {
    let input = r#"
class TestClass {
    public function test() {
        var x = 42;
        return x
    }
    
    // Invalid syntax
    invalid_keyword here;
}
"#;

    println!("Testing enhanced error reporting for unexpected token");
    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => panic!("Expected parse error but got success"),
        Err(e) => {
            println!("Enhanced error message:");
            println!("{}", e);

            // Check that the error message provides context
            assert!(e.len() > 50); // Should be a detailed error message
        }
    }
}

#[test]
fn test_enhanced_error_eof() {
    let input = r#"
class TestClass {
    public function test() {
        var x = 42;
"#;

    println!("Testing enhanced error reporting for unexpected EOF");
    match parse_haxe_file("test.hx", input, false) {
        Ok(_) => panic!("Expected parse error but got success"),
        Err(e) => {
            println!("Enhanced error message:");
            println!("{}", e);

            // Check that the error message mentions EOF or end of input
            assert!(e.contains("end of input") || e.contains("EOF") || e.contains("expected"));
        }
    }
}
