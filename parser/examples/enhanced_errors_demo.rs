//! Demo of enhanced error reporting with Rust-style diagnostics
//!
//! This example shows how the new error reporting system provides:
//! - Multiple error types (errors, warnings, info, hints)
//! - Specific suggestions for fixes
//! - Help text and notes
//! - Beautiful formatting with source highlighting

use parser::enhanced_context::HaxeDiagnostics;
use parser::{
    DiagnosticBuilder, Diagnostics, ErrorFormatter, SourceMap, SourcePosition, SourceSpan,
    parse_haxe_file,
};

fn main() {
    println!("🚀 Enhanced Error Reporting Demo");
    println!("================================\n");

    // Test 1: Basic parsing error with suggestions
    demo_basic_errors();

    // Test 2: Multiple diagnostics
    demo_multiple_diagnostics();

    // Test 3: Advanced diagnostics with suggestions
    demo_advanced_diagnostics();

    // Test 4: Real parsing errors
    demo_real_parsing_errors();
}

fn demo_basic_errors() {
    println!("📋 Demo 1: Basic Parsing Errors");
    println!("-------------------------------\n");

    let test_code = r#"
class Test {
    fucntion test() {
        var x = 1
        return x;
    }
}
"#;

    match parse_haxe_file("demo.hx", test_code, false) {
        Ok(_) => println!("✅ Parsed successfully"),
        Err(e) => {
            println!("❌ Parse failed:");
            println!("{}", e);
        }
    }

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}

fn demo_multiple_diagnostics() {
    println!("📋 Demo 2: Multiple Diagnostics");
    println!("-------------------------------\n");

    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file(
        "demo.hx".to_string(),
        r#"
class Test {
    fucntion test() {
        var x = 1
        var result = switch (x) {
            case 0: "zero";
            case 1: "one";
        }
        return result
    }
}
"#
        .to_string(),
    );

    let mut diagnostics = Diagnostics::new();

    // Typo in function keyword
    let typo_span = SourceSpan::new(
        SourcePosition::new(3, 5, 18),
        SourcePosition::new(3, 13, 26),
        file_id,
    );
    diagnostics.push(HaxeDiagnostics::invalid_identifier(
        typo_span,
        "fucntion",
        "unknown keyword, did you mean something else?",
    ));

    // Missing semicolon after variable declaration
    let semicolon_span = SourceSpan::new(
        SourcePosition::new(4, 18, 44),
        SourcePosition::new(4, 19, 45),
        file_id,
    );
    diagnostics.push(HaxeDiagnostics::missing_semicolon(
        semicolon_span,
        "variable declaration",
    ));

    // Incomplete switch (missing default case)
    let switch_span = SourceSpan::new(
        SourcePosition::new(5, 22, 67),
        SourcePosition::new(8, 10, 130),
        file_id,
    );
    diagnostics.push(HaxeDiagnostics::incomplete_switch_expression(
        switch_span,
        &["default case".to_string()],
    ));

    // Missing semicolon after return
    let return_span = SourceSpan::new(
        SourcePosition::new(9, 22, 152),
        SourcePosition::new(9, 23, 153),
        file_id,
    );
    diagnostics.push(HaxeDiagnostics::missing_semicolon(
        return_span,
        "return statement",
    ));

    let formatter = ErrorFormatter::with_colors();
    println!(
        "{}",
        formatter.format_diagnostics(&diagnostics, &source_map)
    );

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}

fn demo_advanced_diagnostics() {
    println!("📋 Demo 3: Advanced Diagnostics with Rich Information");
    println!("----------------------------------------------------\n");

    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file(
        "demo.hx".to_string(),
        r#"
import haxe.ds.Map;

class Calculator {
    var operations: StringMap<String->Float->Float>;
    
    public function new() {
        operations = new StringMap();
        operations.set("add", function(a: Float, b: Float) a + b);
    }
    
    public function calculate(op: String, a: Float, b: Float): Float {
        var fn = operations.get(op);
        return fn(a, b);
    }
}
"#
        .to_string(),
    );

    let mut diagnostics = Diagnostics::new();

    // Info about missing import
    let type_span = SourceSpan::new(
        SourcePosition::new(5, 18, 67),
        SourcePosition::new(5, 27, 76),
        file_id,
    );
    diagnostics.push(HaxeDiagnostics::missing_import_suggestion(
        type_span,
        "StringMap",
    ));

    // Hint about return type annotation
    let function_span = SourceSpan::new(
        SourcePosition::new(12, 5, 200),
        SourcePosition::new(12, 17, 212),
        file_id,
    );
    diagnostics.push(
        DiagnosticBuilder::hint(
            "consider adding null check".to_string(),
            function_span.clone(),
        )
        .code("H0002")
        .label(function_span, "function might return null")
        .help("operations.get() can return null if the key doesn't exist")
        .note("use null-safe operators or explicit null checking")
        .suggestion(
            "add null check",
            SourceSpan::new(
                SourcePosition::new(14, 16, 290),
                SourcePosition::new(14, 25, 299),
                file_id,
            ),
            "if (fn != null) fn(a, b) else 0.0".to_string(),
        )
        .build(),
    );

    let formatter = ErrorFormatter::with_colors();
    println!(
        "{}",
        formatter.format_diagnostics(&diagnostics, &source_map)
    );

    let separator = "=".repeat(60);
    println!("\n{}\n", separator);
}

fn demo_real_parsing_errors() {
    println!("📋 Demo 4: Real Parsing Errors from Parser");
    println!("------------------------------------------\n");

    let test_cases = [
        // Missing semicolon after switch expression in variable assignment
        (
            "switch_semicolon.hx",
            r#"
class Test {
    static function test() {
        var x = 1;
        var result = switch (x) {
            case 0: "zero";
            case 1: "one";
            case _: "other";
        }
        trace(result);
    }
}
"#,
        ),
        // Unclosed parentheses
        (
            "unclosed_paren.hx",
            r#"
class Test {
    static function test() {
        var x = Math.max(1, 2;
        return x;
    }
}
"#,
        ),
        // Invalid function syntax
        (
            "invalid_function.hx",
            r#"
class Test {
    fucntion test {
        return "hello";
    }
}
"#,
        ),
    ];

    for (filename, code) in &test_cases {
        println!("🔍 Testing: {}", filename);
        println!("{}", "-".repeat(40));

        match parse_haxe_file(filename, code, false) {
            Ok(_) => println!("✅ Parsed successfully (unexpected!)"),
            Err(e) => {
                println!("❌ Parse error (as expected):");
                println!("{}", e);
            }
        }

        println!();
    }
}
