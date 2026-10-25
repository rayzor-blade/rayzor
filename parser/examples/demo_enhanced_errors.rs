use parser::parse_haxe_file;

fn main() {
    // Test 1: Missing semicolon
    println!("=== Test 1: Missing semicolon ===");
    let input1 = r#"
class TestClass {
    public function test() {
        var x = 42
        return x;
    }
}
"#;

    match parse_haxe_file("test.hx", input1, false) {
        Ok(_) => println!("✓ Unexpected success"),
        Err(e) => {
            println!("Enhanced error message:");
            println!("{}", e);
        }
    }

    println!("\n=== Test 2: Unexpected EOF ===");
    let input2 = r#"
class TestClass {
    public function test() {
        var x = 42;
"#;

    match parse_haxe_file("test.hx", input2, false) {
        Ok(_) => println!("✓ Unexpected success"),
        Err(e) => {
            println!("Enhanced error message:");
            println!("{}", e);
        }
    }

    println!("\n=== Test 3: Valid code (should work) ===");
    let input3 = r#"
class TestClass {
    public function test() {
        var x = 42;
        return x;
    }
}
"#;

    match parse_haxe_file("test.hx", input3, false) {
        Ok(_) => println!("✓ Parse succeeded as expected"),
        Err(e) => {
            println!("Unexpected error: {}", e);
        }
    }
}
