#![allow(
    unused_imports,
    unused_variables,
    dead_code,
    unreachable_patterns,
    unused_mut,
    unused_assignments,
    unused_parens
)]
#![allow(
    clippy::single_component_path_imports,
    clippy::for_kv_map,
    clippy::explicit_auto_deref
)]
#![allow(
    clippy::println_empty_string,
    clippy::len_zero,
    clippy::useless_vec,
    clippy::field_reassign_with_default
)]
#![allow(
    clippy::needless_borrow,
    clippy::redundant_closure,
    clippy::bool_assert_comparison
)]
#![allow(
    clippy::empty_line_after_doc_comments,
    clippy::useless_format,
    clippy::clone_on_copy
)]
//! Haxe Standard Library Test Suite
//!
//! Tests parsing, compilation and execution of Haxe stdlib files.

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use parser::preprocessor::PreprocessorConfig;

/// Root-level Haxe stdlib files (in compiler/haxe-std/)
const ROOT_STDLIB_FILES: &[&str] = &[
    "Any.hx",
    "Array.hx",
    "Class.hx",
    "Date.hx",
    "DateTools.hx",
    "EReg.hx",
    "Enum.hx",
    "EnumValue.hx",
    "IntIterator.hx",
    "Lambda.hx",
    "List.hx",
    "Map.hx",
    "Math.hx",
    "Reflect.hx",
    "Std.hx",
    "StdTypes.hx",
    "String.hx",
    "StringBuf.hx",
    "StringTools.hx",
    "Sys.hx",
    "Type.hx",
    "UInt.hx",
    "UnicodeString.hx",
    "Xml.hx",
];

fn main() -> Result<(), String> {
    println!("=== Haxe Standard Library Test Suite ===\n");

    // Phase 1: Parse each root-level stdlib file
    println!("Phase 1: Parsing root-level stdlib files...\n");

    let mut parse_pass = 0;
    let mut parse_fail = 0;
    let mut parse_failures: Vec<(String, String)> = Vec::new();

    for file in ROOT_STDLIB_FILES {
        let path = format!("compiler/haxe-std/{}", file);
        print!("  [PARSE] {}... ", file);

        match std::fs::read_to_string(&path) {
            Ok(content) => {
                // Preprocess before parsing
                let config = PreprocessorConfig::default();
                let preprocessed = parser::preprocessor::preprocess(&content, &config);

                match parser::haxe_parser::haxe_file(&path, &preprocessed, &preprocessed) {
                    Ok(_ast) => {
                        println!("OK");
                        parse_pass += 1;
                    }
                    Err(e) => {
                        let err_msg = format!("{:?}", e);
                        let short_err = if err_msg.len() > 60 {
                            format!("{}...", &err_msg[..60])
                        } else {
                            err_msg.clone()
                        };
                        println!("FAIL: {}", short_err);
                        parse_fail += 1;
                        parse_failures.push((file.to_string(), err_msg));
                    }
                }
            }
            Err(e) => {
                println!("FAIL: File not found - {}", e);
                parse_fail += 1;
                parse_failures.push((file.to_string(), format!("File not found: {}", e)));
            }
        }
    }

    println!(
        "\n  Parse Results: {}/{} passed\n",
        parse_pass,
        ROOT_STDLIB_FILES.len()
    );

    // Phase 2: Functional tests - compile and run simple programs using stdlib
    println!("Phase 2: Functional stdlib tests...\n");

    let functional_tests: Vec<(&str, &str)> = vec![
        // Math tests
        (
            "math_abs",
            r#"
class Main {
    static function main() {
        trace(Math.abs(-5));
    }
}
"#,
        ),
        (
            "math_floor",
            r#"
class Main {
    static function main() {
        trace(Math.floor(3.7));
    }
}
"#,
        ),
        (
            "math_ceil",
            r#"
class Main {
    static function main() {
        trace(Math.ceil(3.2));
    }
}
"#,
        ),
        (
            "math_max",
            r#"
class Main {
    static function main() {
        trace(Math.max(10, 20));
    }
}
"#,
        ),
        (
            "math_min",
            r#"
class Main {
    static function main() {
        trace(Math.min(10, 20));
    }
}
"#,
        ),
        (
            "math_sqrt",
            r#"
class Main {
    static function main() {
        trace(Math.sqrt(16));
    }
}
"#,
        ),
        // String tests
        (
            "string_length",
            r#"
class Main {
    static function main() {
        var s = "hello";
        trace(s.length);
    }
}
"#,
        ),
        (
            "string_charat",
            r#"
class Main {
    static function main() {
        var s = "hello";
        trace(s.charAt(1));
    }
}
"#,
        ),
        (
            "string_indexof",
            r#"
class Main {
    static function main() {
        var s = "hello world";
        trace(s.indexOf("world"));
    }
}
"#,
        ),
        (
            "string_substr",
            r#"
class Main {
    static function main() {
        var s = "hello world";
        trace(s.substr(0, 5));
    }
}
"#,
        ),
        (
            "string_touppercase",
            r#"
class Main {
    static function main() {
        var s = "hello";
        trace(s.toUpperCase());
    }
}
"#,
        ),
        (
            "string_tolowercase",
            r#"
class Main {
    static function main() {
        var s = "HELLO";
        trace(s.toLowerCase());
    }
}
"#,
        ),
        // Array tests
        (
            "array_push",
            r#"
class Main {
    static function main() {
        var arr = new Array<Int>();
        arr.push(1);
        arr.push(2);
        trace(arr.length);
    }
}
"#,
        ),
        (
            "array_pop",
            r#"
class Main {
    static function main() {
        var arr = [1, 2, 3];
        trace(arr.pop());
    }
}
"#,
        ),
        (
            "array_iteration",
            r#"
class Main {
    static function main() {
        var arr = [1, 2, 3];
        var sum = 0;
        for (x in arr) {
            sum += x;
        }
        trace(sum);
    }
}
"#,
        ),
        // Std tests
        (
            "std_int",
            r#"
class Main {
    static function main() {
        trace(Std.int(3.7));
    }
}
"#,
        ),
        (
            "std_parseint",
            r#"
class Main {
    static function main() {
        trace(Std.parseInt("42"));
    }
}
"#,
        ),
        (
            "std_parsefloat",
            r#"
class Main {
    static function main() {
        trace(Std.parseFloat("3.14"));
    }
}
"#,
        ),
        (
            "std_random",
            r#"
class Main {
    static function main() {
        var r = Std.random(100);
        trace(r >= 0 && r < 100);
    }
}
"#,
        ),
        // IntIterator tests
        (
            "intiterator_basic",
            r#"
class Main {
    static function main() {
        var sum = 0;
        for (i in 0...5) {
            sum += i;
        }
        trace(sum);
    }
}
"#,
        ),
        // Date tests
        (
            "date_now",
            r#"
class Main {
    static function main() {
        var d = Date.now();
        trace(d.getFullYear() >= 2024);
    }
}
"#,
        ),
        (
            "stringtools_startswith",
            r#"
using StringTools;       
class Main {
    static function main() {
        var s = "hello world";
        trace(s.startsWith("hello"));
    }
}
"#,
        ),
        (
            "stringtools_contains",
            r#"
class Main {
    static function main() {
        var s = "hello world";
        trace(StringTools.contains(s, "world"));
    }
}
"#,
        ),
        // sys.net.Host — DNS / hostname.
        (
            "host_localhost",
            r#"
import sys.net.Host;

class Main {
    static function main() {
        var name = Host.localhost();
    }
}
"#,
        ),
        // sys.net.Socket + Host — TCP echo over localhost.
        (
            "socket_host_basic",
            r#"
import sys.net.Socket;
import sys.net.Host;

class Main {
    static function main() {
        // Server: bind to localhost and listen.
        var server = new Socket();
        var host = new Host("127.0.0.1");
        server.bind(host, 19876);
        server.listen(1);

        // Client: connect to the server.
        var client = new Socket();
        client.connect(new Host("127.0.0.1"), 19876);

        // Server accepts; client writes; server reads.
        var conn = server.accept();
        client.write("hello");
        var data = conn.read();

        // Cleanup.
        conn.close();
        client.close();
        server.close();
    }
}
"#,
        ),
    ];

    let mut func_pass = 0;
    let mut func_fail = 0;
    let mut func_failures: Vec<(String, String)> = Vec::new();

    for (test_name, code) in &functional_tests {
        print!("  [EXEC] {}... ", test_name);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        match run_functional_test(code) {
            Ok(_output) => {
                println!("OK");
                func_pass += 1;
            }
            Err(e) => {
                let short_err = if e.len() > 50 {
                    format!("{}...", &e[..50])
                } else {
                    e.clone()
                };
                println!("FAIL: {}", short_err);
                func_fail += 1;
                func_failures.push((test_name.to_string(), e));
            }
        }
    }

    println!(
        "\n  Functional Results: {}/{} passed\n",
        func_pass,
        functional_tests.len()
    );

    // Summary
    println!("=== Summary ===");
    println!(
        "Parse tests:      {}/{}",
        parse_pass,
        ROOT_STDLIB_FILES.len()
    );
    println!("Functional tests: {}/{}", func_pass, functional_tests.len());

    let total = ROOT_STDLIB_FILES.len() + functional_tests.len();
    let passed = parse_pass + func_pass;

    if passed == total {
        println!("\nAll {} tests passed!", total);
        Ok(())
    } else {
        println!("\n{} tests failed", total - passed);

        if !parse_failures.is_empty() {
            println!("\nParse failures:");
            for (file, err) in &parse_failures {
                println!("  - {}: {}", file, err);
            }
        }

        if !func_failures.is_empty() {
            println!("\nFunctional failures:");
            for (name, err) in &func_failures {
                println!("  - {}: {}", name, err);
            }
        }

        Err(format!("{} tests failed", total - passed))
    }
}

fn run_functional_test(code: &str) -> Result<String, String> {
    // Create compilation unit with fast (lazy) config to ensure proper dependency ordering
    let mut unit = CompilationUnit::new(CompilationConfig::fast());

    // Load stdlib
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add test code
    unit.add_file(code, "test/Main.hx")
        .map_err(|e| format!("Failed to add file: {}", e))?;

    // Compile to TAST
    unit.lower_to_tast()
        .map_err(|errors| format!("TAST errors: {:?}", errors))?;

    // Get MIR modules
    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Create Cranelift backend with runtime symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend =
        CraneliftBackend::with_symbols(&symbols_ref).map_err(|e| format!("Backend: {}", e))?;

    // Compile modules
    for module in &mir_modules {
        backend
            .compile_module(module)
            .map_err(|e| format!("Compile: {}", e))?;
    }

    // Execute
    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            return Ok("executed".to_string());
        }
    }

    Err("No main found".to_string())
}
