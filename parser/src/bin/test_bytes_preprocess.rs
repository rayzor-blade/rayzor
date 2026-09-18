//! Test preprocessor with Bytes.hx pattern
use parser::parse_haxe_file_with_diagnostics;
use parser::preprocessor::{PreprocessorConfig, preprocess};

fn main() {
    let config = PreprocessorConfig::default();

    // Test 1: Simple property syntax
    println!("=== Test 1: Simple property syntax ===");
    let simple_property = r#"
package test;

class Test {
    public var length(default, null):Int;
}
"#;
    let result1 = parse_haxe_file_with_diagnostics("Test.hx", simple_property);
    match &result1 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2: More complex properties with private vars
    println!("\n=== Test 2: Complex class with multiple fields ===");
    let complex_class = r#"
package test;

class Bytes {
    public var length(default, null):Int;
    var b:BytesData;

    function new(length, b) {
        this.length = length;
        this.b = b;
    }

    public inline function get(pos:Int):Int {
        return b[pos];
    }
}
"#;
    let result2 = parse_haxe_file_with_diagnostics("Test.hx", complex_class);
    match &result2 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2b: Class with untyped expression
    println!("\n=== Test 2b: Class with untyped expression ===");
    let untyped_class = r#"
package test;

class Bytes {
    var b:BytesData;

    public function compare(other:Bytes):Int {
        var b1 = b;
        var b2 = other.b;
        for (i in 0...10)
            if (b1[i] != b2[i])
                return untyped b1[i] - b2[i];
        return 0;
    }
}
"#;
    let result2b = parse_haxe_file_with_diagnostics("Test.hx", untyped_class);
    match &result2b {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2c: Class with throw Error.OutsideBounds
    println!("\n=== Test 2c: Class with throw Error.OutsideBounds ===");
    let throw_class = r#"
package test;

class Bytes {
    public var length:Int;

    public function blit(pos:Int, len:Int):Void {
        if (pos < 0 || len < 0 || pos + len > length)
            throw Error.OutsideBounds;
        // do something
    }
}
"#;
    let result2c = parse_haxe_file_with_diagnostics("Test.hx", throw_class);
    match &result2c {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2d: Constructor without typed params
    println!("\n=== Test 2d: Constructor without typed params ===");
    let untyped_constructor = r#"
package test;

class Bytes {
    public var length:Int;
    var b:BytesData;

    function new(length, b) {
        this.length = length;
        this.b = b;
    }
}
"#;
    let result2d = parse_haxe_file_with_diagnostics("Test.hx", untyped_constructor);
    match &result2d {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2e: Function with 4 params like blit
    println!("\n=== Test 2e: Function with 4 params (blit) ===");
    let blit_class = r#"
package test;

class Bytes {
    public var length:Int;
    var b:BytesData;

    public function blit(pos:Int, src:Bytes, srcpos:Int, len:Int):Void {
        if (pos < 0 || srcpos < 0 || len < 0)
            throw Error.OutsideBounds;
    }
}
"#;
    let result2e = parse_haxe_file_with_diagnostics("Test.hx", blit_class);
    match &result2e {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2f: While loop with decrement
    println!("\n=== Test 2f: While loop with decrement ===");
    let while_class = r#"
package test;

class Bytes {
    var b:Array<Int>;

    public function test():Void {
        var i = 10;
        while (i > 0) {
            i--;
            b[i] = 0;
        }
    }
}
"#;
    let result2f = parse_haxe_file_with_diagnostics("Test.hx", while_class);
    match &result2f {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2g: Accessing fields on returned struct (i.low, i.high)
    println!("\n=== Test 2g: Accessing fields on returned struct ===");
    let field_access = r#"
package test;

class Bytes {
    public function setDouble(pos:Int, v:Float):Void {
        var i = FPHelper.doubleToI64(v);
        setInt32(pos, i.low);
        setInt32(pos + 4, i.high);
    }
}
"#;
    let result2g = parse_haxe_file_with_diagnostics("Test.hx", field_access);
    match &result2g {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test 2h: Function call with two function call arguments (like line 122)
    println!("\n=== Test 2h: Two function calls as arguments ===");
    let two_calls = r#"
package test;

class Bytes {
    public function getDouble(pos:Int):Float {
        return FPHelper.i64ToDouble(getInt32(pos), getInt32(pos + 4));
    }
}
"#;
    let result2h = parse_haxe_file_with_diagnostics("Test.hx", two_calls);
    match &result2h {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test real file
    println!("\n=== Test 3: Real Bytes.hx ===");
    let real_bytes = std::fs::read_to_string("compiler/haxe-std/haxe/io/Bytes.hx")
        .expect("Failed to read Bytes.hx");

    // Preprocess to see what the parser receives
    let preprocessed = preprocess(&real_bytes, &config);

    println!("Total preprocessed lines: {}", preprocessed.lines().count());

    // Count how many lines until "class Bytes"
    let class_line = preprocessed
        .lines()
        .enumerate()
        .find(|(_, line)| line.contains("class Bytes"))
        .map(|(i, _)| i + 1);
    println!("\n'class Bytes' found at line: {:?}", class_line);

    // Test first 48 lines (ends after the inline get function closes)
    println!("\n=== Test first 48 lines of preprocessed Bytes.hx ===");
    let first_48: String = preprocessed.lines().take(48).collect::<Vec<_>>().join("\n") + "\n}\n";
    let result_48 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_48);
    match &result_48 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 48 lines");
            } else {
                println!("✅ PASS first 48 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test first 73 lines (ends after blit function)
    println!("\n=== Test first 73 lines of preprocessed Bytes.hx ===");
    let first_73: String = preprocessed.lines().take(73).collect::<Vec<_>>().join("\n") + "\n}\n";
    let result_73 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_73);
    match &result_73 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 73 lines");
            } else {
                println!("✅ PASS first 73 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Print preprocessed lines 115-155 to find the failing area
    println!("\n=== Preprocessed content lines 115-155 ===");
    for (i, line) in preprocessed.lines().enumerate() {
        if (114..155).contains(&i) {
            println!("{:4}: {}", i + 1, line);
        }
    }

    // Test first 82 lines (should include fill function)
    println!("\n=== Test first 82 lines of preprocessed Bytes.hx ===");
    let first_82: String = preprocessed.lines().take(82).collect::<Vec<_>>().join("\n") + "\n}\n";
    let result_82 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_82);
    match &result_82 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 82 lines");
            } else {
                println!("✅ PASS first 82 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test first 93 lines (should include sub function with b.slice())
    println!("\n=== Test first 93 lines of preprocessed Bytes.hx ===");
    let first_93: String = preprocessed.lines().take(93).collect::<Vec<_>>().join("\n") + "\n}\n";
    let result_93 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_93);
    match &result_93 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 93 lines");
            } else {
                println!("✅ PASS first 93 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test first 115 lines (should include compare function signature)
    println!("\n=== Test first 115 lines of preprocessed Bytes.hx ===");
    let first_115: String = preprocessed
        .lines()
        .take(115)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n}\n";
    let result_115 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_115);
    match &result_115 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 115 lines");
            } else {
                println!("✅ PASS first 115 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test first 150 lines
    println!("\n=== Test first 150 lines of preprocessed Bytes.hx ===");
    let first_150: String = preprocessed
        .lines()
        .take(150)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n}\n";
    let result_150 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_150);
    match &result_150 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 150 lines");
            } else {
                println!("✅ PASS first 150 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test first 100 lines of preprocessed content (no closing brace - expected to fail)
    println!("\n=== Test first 100 lines of preprocessed Bytes.hx (no brace) ===");
    let first_100: String = preprocessed
        .lines()
        .take(100)
        .collect::<Vec<_>>()
        .join("\n");
    let result_100 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_100);
    match &result_100 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 100 lines");
            } else {
                println!("✅ PASS first 100 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Test first 200 lines
    println!("\n=== Test first 200 lines of preprocessed Bytes.hx ===");
    let first_200: String = preprocessed
        .lines()
        .take(200)
        .collect::<Vec<_>>()
        .join("\n");
    let result_200 = parse_haxe_file_with_diagnostics("Bytes.hx", &first_200);
    match &result_200 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
                println!("❌ FAIL first 200 lines");
            } else {
                println!("✅ PASS first 200 lines");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }

    // Parse the preprocessed content directly (to simulate what the parser does)
    println!("\n=== Parsing preprocessed content ===");
    let result = parser::incremental_parser_enhanced::parse_incrementally_enhanced(
        "Bytes.hx",
        &preprocessed,
    );

    println!("Parsed elements: {}", result.parsed_elements.len());
    for (i, elem) in result.parsed_elements.iter().enumerate() {
        match elem {
            parser::incremental_parser_enhanced::ParsedElement::Package(pkg) => {
                println!("  {}: Package - {:?}", i, pkg.path);
            }
            parser::incremental_parser_enhanced::ParsedElement::Import(imp) => {
                println!("  {}: Import - {:?}", i, imp.path);
            }
            parser::incremental_parser_enhanced::ParsedElement::Using(u) => {
                println!("  {}: Using - {:?}", i, u.path);
            }
            parser::incremental_parser_enhanced::ParsedElement::TypeDeclaration(td) => {
                let name = match td {
                    parser::haxe_ast::TypeDeclaration::Class(c) => format!("class {}", c.name),
                    parser::haxe_ast::TypeDeclaration::Enum(e) => format!("enum {}", e.name),
                    parser::haxe_ast::TypeDeclaration::Interface(i) => {
                        format!("interface {}", i.name)
                    }
                    parser::haxe_ast::TypeDeclaration::Abstract(a) => {
                        format!("abstract {}", a.name)
                    }
                    parser::haxe_ast::TypeDeclaration::Typedef(t) => format!("typedef {}", t.name),
                    parser::haxe_ast::TypeDeclaration::Conditional(_) => "conditional".to_string(),
                };
                println!("  {}: TypeDecl - {}", i, name);
            }
            _ => {
                println!("  {}: Other", i);
            }
        }
    }

    if result.has_errors() {
        println!("\n=== Errors ===");
        println!("{}", result.format_diagnostics(false));
    }

    // Now parse through the normal flow
    println!("\n=== Parsing through parse_haxe_file_with_diagnostics ===");
    let result2 = parse_haxe_file_with_diagnostics("Bytes.hx", &real_bytes);
    match &result2 {
        Ok(parse_result) => {
            println!("Declarations: {}", parse_result.file.declarations.len());
            if parse_result.diagnostics.has_errors() {
                for error in parse_result.diagnostics.errors() {
                    println!("  Error: {}", error.message);
                }
            }
            if parse_result.file.declarations.is_empty() {
                println!("❌ FAIL");
            } else {
                println!("✅ PASS");
            }
        }
        Err(e) => println!("Parse error: {}", e),
    }
}
