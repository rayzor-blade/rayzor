//! End-to-end macro system tests.
//!
//! Tests the full macro pipeline: parse → register → expand → TAST → HIR → MIR → JIT.
//! Covers real metaprogramming patterns: reification, dollar-splicing, AST construction,
//! Context API, conditional compilation, and compile-time code generation.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Run a Haxe source file through `rayzor run` and return (stdout, stderr, success).
fn run_haxe_source(source: &str) -> (String, String, bool) {
    let tmp_dir = std::env::temp_dir().join("rayzor_macro_e2e");
    let _ = std::fs::create_dir_all(&tmp_dir);
    // Use atomic counter + thread ID + nanos for guaranteed unique filenames
    let counter = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    let thread_id = format!("{:?}", std::thread::current().id());
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp_file = tmp_dir.join(format!(
        "test_{}_{}{}.hx",
        counter,
        nanos,
        thread_id.replace(|c: char| !c.is_alphanumeric(), "")
    ));
    std::fs::write(&tmp_file, source).expect("failed to write temp file");

    let rayzor = find_rayzor_binary();

    let output = Command::new(&rayzor)
        .arg("run")
        .arg(tmp_file.to_str().unwrap())
        .output()
        .expect("failed to execute rayzor");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let _ = std::fs::remove_file(&tmp_file);
    (stdout, stderr, output.status.success())
}

fn extract_trace_lines(stdout: &str) -> Vec<&str> {
    stdout
        .lines()
        .filter(|line| {
            !line.starts_with('\u{1f680}') && !line.starts_with('\u{2713}') && !line.is_empty()
        })
        .collect()
}

fn find_rayzor_binary() -> String {
    // Try from workspace root first (typical cargo test cwd)
    for path in &[
        "target/release/rayzor",
        "../target/release/rayzor",
        "../../target/release/rayzor",
        "target/debug/rayzor",
        "../target/debug/rayzor",
    ] {
        if Path::new(path).exists() {
            return path.to_string();
        }
    }
    panic!("rayzor binary not found — run `cargo build --release` first");
}

// ================================================================
// REIFICATION: macro expr + $e{} expression splicing
// ================================================================

/// Test `macro trace($e{expr})` — wrap any expression in a trace call
#[test]
fn test_reification_expression_splice() {
    let source = r#"
class MacroTools {
    macro static function debugTrace(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro trace($e{e});
    }
}
class Main {
    static function main() {
        MacroTools.debugTrace(42);
        MacroTools.debugTrace("hello");
        MacroTools.debugTrace(10 + 20);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "hello", "30", "done"]);
}

// ================================================================
// REIFICATION: $v{} value splicing — inject computed constants
// ================================================================

/// Test `$v{computed_value}` — compile-time string manipulation injected as constant
#[test]
fn test_reification_value_splice() {
    let source = r#"
class MacroTools {
    macro static function shout(msg:haxe.macro.Expr):haxe.macro.Expr {
        var result = msg + "!!!";
        return macro $v{result};
    }
    macro static function compileTimeConcat(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        var combined = a + " " + b;
        return macro $v{combined};
    }
}
class Main {
    static function main() {
        trace(MacroTools.shout("hello"));
        trace(MacroTools.compileTimeConcat("foo", "bar"));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["hello!!!", "foo bar", "done"]);
}

// ================================================================
// REIFICATION: $i{} identifier splicing — dynamic function names
// ================================================================

/// Test `$i{name}` — splice a string as an identifier in generated code
#[test]
fn test_reification_identifier_splice() {
    let source = r#"
class MacroTools {
    macro static function callByName(name:haxe.macro.Expr, arg:haxe.macro.Expr):haxe.macro.Expr {
        return macro $i{name}($e{arg});
    }
}
class Main {
    static function main() {
        MacroTools.callByName("trace", 42);
        MacroTools.callByName("trace", "dynamic call");
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "dynamic call", "done"]);
}

// ================================================================
// CONTEXT API: Context.parse() — string-to-code generation
// ================================================================

/// Test Context.parse() — generate code from strings at compile time
#[test]
fn test_context_parse_code_generation() {
    let source = r#"
class MacroTools {
    macro static function genExpr(code:haxe.macro.Expr):haxe.macro.Expr {
        var parsed = Context.parse(code, Context.currentPos());
        return parsed;
    }
}
class Main {
    static function main() {
        var x = MacroTools.genExpr("1 + 2 + 3");
        trace(x);
        var y = MacroTools.genExpr("10 * 5");
        trace(y);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["6", "50", "done"]);
}

/// Test Context.parse() with dynamically constructed code strings
#[test]
fn test_context_parse_dynamic_code() {
    let source = r#"
class MacroTools {
    macro static function makeAdder(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        var code = a + " + " + b;
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        var x = MacroTools.makeAdder("100", "200");
        trace(x);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["300", "done"]);
}

// ================================================================
// COMPILE-TIME CODE GENERATION: arrays, loops, and control flow
// ================================================================

/// Test compile-time array generation — macro builds array at compile time
#[test]
fn test_compile_time_array_generation() {
    let source = r#"
class MacroTools {
    macro static function generateSquares(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = [];
        var i = 0;
        while (i < n) {
            result.push(i * i);
            i = i + 1;
        }
        return result;
    }
}
class Main {
    static function main() {
        var squares = MacroTools.generateSquares(5);
        trace(squares[0]);
        trace(squares[1]);
        trace(squares[2]);
        trace(squares[3]);
        trace(squares[4]);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["0", "1", "4", "9", "16", "done"]);
}

/// Test compile-time fibonacci — complex control flow in macro body
#[test]
fn test_compile_time_fibonacci() {
    let source = r#"
class MacroTools {
    macro static function fibonacci(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n <= 1) return n;
        var a = 0;
        var b = 1;
        var i = 2;
        while (i <= n) {
            var temp = a + b;
            a = b;
            b = temp;
            i = i + 1;
        }
        return b;
    }
}
class Main {
    static function main() {
        trace(MacroTools.fibonacci(0));
        trace(MacroTools.fibonacci(1));
        trace(MacroTools.fibonacci(10));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["0", "1", "55", "done"]);
}

// ================================================================
// MACRO-GENERATED TRACE WRAPPERS — real-world pattern
// ================================================================

/// Test macro that wraps expressions in trace calls with labels
#[test]
fn test_macro_trace_wrapper_with_label() {
    let source = r#"
class MacroTools {
    macro static function traceLabeled(label:haxe.macro.Expr, e:haxe.macro.Expr):haxe.macro.Expr {
        var prefix = label + ": ";
        return macro trace($v{prefix} + $e{e});
    }
}
class Main {
    static function main() {
        MacroTools.traceLabeled("result", 42);
        MacroTools.traceLabeled("name", "alice");
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["result: 42", "name: alice", "done"]);
}

// ================================================================
// CONDITIONAL COMPILATION (#if/#end)
// ================================================================

#[test]
fn test_conditional_compilation_rayzor_flag() {
    let source = r#"
class Main {
    static function main() {
        #if rayzor
        trace("rayzor");
        #end
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["rayzor", "done"]);
}

// ================================================================
// MACRO DEFINITION STRIPPING
// ================================================================

/// Macro definitions (with haxe.macro.Expr types) are stripped before TAST
#[test]
fn test_macro_definitions_stripped() {
    let source = r#"
class MacroTools {
    macro static function constVal():haxe.macro.Expr {
        return 42;
    }
    public static function helper():Int {
        return 10;
    }
}
class Main {
    static function main() {
        trace(MacroTools.constVal());
        trace(MacroTools.helper());
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "10", "done"]);
}

// ================================================================
// MULTIPLE MACRO CLASSES
// ================================================================

#[test]
fn test_multiple_macro_classes() {
    let source = r#"
class MathMacros {
    macro static function square(x:haxe.macro.Expr):haxe.macro.Expr {
        return x * x;
    }
}
class StringMacros {
    macro static function wrap(s:haxe.macro.Expr):haxe.macro.Expr {
        return "[" + s + "]";
    }
}
class Main {
    static function main() {
        trace(MathMacros.square(7));
        trace(StringMacros.wrap("hello"));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["49", "[hello]", "done"]);
}

// ================================================================
// MEMOIZATION: same args produce cached results
// ================================================================

#[test]
fn test_macro_memoization() {
    let source = r#"
class MacroTools {
    macro static function compute(x:haxe.macro.Expr):haxe.macro.Expr {
        return x * x + 1;
    }
}
class Main {
    static function main() {
        trace(MacroTools.compute(5));
        trace(MacroTools.compute(5));
        trace(MacroTools.compute(10));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["26", "26", "101", "done"]);
}

// ================================================================
// MIXED REIFICATION + CONSTANT FOLDING in same file
// ================================================================

/// Comprehensive test: reification, value splicing, identifier splicing,
/// Context.parse(), compile-time computation, and conditional compilation
#[test]
fn test_full_metaprogramming_integration() {
    let source = r#"
class MacroTools {
    macro static function add(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }
    macro static function wrapTrace(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro trace($e{e});
    }
    macro static function shout(msg:haxe.macro.Expr):haxe.macro.Expr {
        var result = msg + "!";
        return macro $v{result};
    }
    macro static function genFromString(code:haxe.macro.Expr):haxe.macro.Expr {
        return Context.parse(code, Context.currentPos());
    }
    macro static function fibonacci(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n <= 1) return n;
        var a = 0;
        var b = 1;
        var i = 2;
        while (i <= n) {
            var temp = a + b;
            a = b;
            b = temp;
            i = i + 1;
        }
        return b;
    }
}
class Main {
    static function main() {
        trace(MacroTools.add(100, 200));
        MacroTools.wrapTrace(42);
        trace(MacroTools.shout("hello"));
        var x = MacroTools.genFromString("7 * 8");
        trace(x);
        trace(MacroTools.fibonacci(10));
        #if rayzor
        trace("rayzor");
        #end
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(
        lines,
        vec!["300", "42", "hello!", "56", "55", "rayzor", "done"]
    );
}

// ================================================================
// MACRO WITH RESULT USED IN RUNTIME EXPRESSIONS
// ================================================================

#[test]
fn test_macro_result_in_expressions() {
    let source = r#"
class MacroTools {
    macro static function ten():haxe.macro.Expr {
        return 10;
    }
    macro static function add(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }
}
class Main {
    static function main() {
        var x = MacroTools.ten();
        var y = MacroTools.add(3, 4);
        trace(x);
        trace(y);
        trace(x + y);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["10", "7", "17", "done"]);
}

// ================================================================
// COMPILE-TIME STRING PROCESSING
// ================================================================

/// Test macro that does string manipulation at compile time
#[test]
fn test_compile_time_string_processing() {
    let source = r#"
class MacroTools {
    macro static function makeGreeting(name:haxe.macro.Expr):haxe.macro.Expr {
        return "Hello, " + name + "!";
    }
    macro static function repeatN(s:haxe.macro.Expr, n:haxe.macro.Expr):haxe.macro.Expr {
        var result = "";
        var i = 0;
        while (i < n) {
            result = result + s;
            i = i + 1;
        }
        return result;
    }
}
class Main {
    static function main() {
        trace(MacroTools.makeGreeting("World"));
        trace(MacroTools.repeatN("ab", 4));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["Hello, World!", "abababab", "done"]);
}

// ================================================================
// COMPILE-TIME CONDITIONAL CODE GENERATION
// ================================================================

/// Macro body uses if/else to generate different code paths at compile time
#[test]
fn test_compile_time_conditional_generation() {
    let source = r#"
class MacroTools {
    macro static function clampPositive(x:haxe.macro.Expr):haxe.macro.Expr {
        if (x < 0) return 0;
        return x;
    }
    macro static function maxVal(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        if (a > b) return a;
        return b;
    }
}
class Main {
    static function main() {
        trace(MacroTools.clampPositive(42));
        trace(MacroTools.clampPositive(-5));
        trace(MacroTools.maxVal(10, 20));
        trace(MacroTools.maxVal(99, 42));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "0", "20", "99", "done"]);
}

// ================================================================
// BLOCK-EXPRESSION-AS-VALUE: blocks return their last expression
// ================================================================

/// Test that block expressions return the value of their last expression
#[test]
#[ignore = "requires block-as-value support in tast_to_hir"]
fn test_block_expression_as_value() {
    let source = r#"
class Main {
    static function main() {
        var x = { 42; };
        trace(x);
        var y = {
            var a = 10;
            var b = 20;
            a + b;
        };
        trace(y);
        var z = {
            var msg = "hello";
            msg;
        };
        trace(z);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "30", "hello", "done"]);
}

// ================================================================
// MACRO BLOCK REIFICATION: `macro { ... $e{} ... }` as value
// ================================================================

/// Test macro that generates block expressions with $e{} splicing
#[test]
#[ignore = "requires block-as-value support in TAST-to-HIR"]
fn test_macro_block_reification_as_value() {
    let source = r#"
class MacroTools {
    macro static function assertPositive(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro {
            var _val = $e{e};
            _val;
        };
    }
}
class Main {
    static function main() {
        var x = MacroTools.assertPositive(42);
        trace(x);
        var y = MacroTools.assertPositive(100);
        trace(y);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "100", "done"]);
}

/// Test macro block with trace side-effects + value return
#[test]
#[ignore = "requires block-as-value support"]
fn test_macro_block_with_side_effects() {
    let source = r#"
class MacroTools {
    macro static function debugValue(label:haxe.macro.Expr, e:haxe.macro.Expr):haxe.macro.Expr {
        return macro {
            var _v = $e{e};
            trace($v{label} + ": " + _v);
            _v;
        };
    }
}
class Main {
    static function main() {
        var x = MacroTools.debugValue("answer", 42);
        trace(x + 1);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["answer: 42", "43", "done"]);
}

// ================================================================
// MACRO EXPRESSION WRAPPING: compile-time code transforms
// ================================================================

/// Test macro that wraps a value in a multiplication expression
#[test]
fn test_macro_expression_doubling() {
    let source = r#"
class MacroTools {
    macro static function double(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{e} * 2);
    }
    macro static function addOne(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{e} + 1);
    }
}
class Main {
    static function main() {
        trace(MacroTools.double(21));
        trace(MacroTools.addOne(99));
        trace(MacroTools.double(MacroTools.addOne(4)));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "100", "10", "done"]);
}

// ================================================================
// COMPILE-TIME LOOKUP TABLES — real-world macro pattern
// ================================================================

/// Test macro that generates lookup table values at compile time
#[test]
fn test_compile_time_lookup_table() {
    let source = r#"
class MacroTools {
    macro static function factorial(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = 1;
        var i = 2;
        while (i <= n) {
            result = result * i;
            i = i + 1;
        }
        return result;
    }
    macro static function isPrime(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n < 2) return 0;
        var i = 2;
        while (i * i <= n) {
            if (n - (n / i) * i == 0) return 0;
            i = i + 1;
        }
        return 1;
    }
}
class Main {
    static function main() {
        trace(MacroTools.factorial(5));
        trace(MacroTools.factorial(10));
        trace(MacroTools.isPrime(7));
        trace(MacroTools.isPrime(10));
        trace(MacroTools.isPrime(13));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["120", "3628800", "1", "0", "1", "done"]);
}

// ================================================================
// STRING BUILDER MACROS — compile-time code generation
// ================================================================

/// Test macro that builds string expressions via Context.parse
#[test]
fn test_context_parse_computed_expression() {
    let source = r#"
class MacroTools {
    macro static function sum3(a:haxe.macro.Expr, b:haxe.macro.Expr, c:haxe.macro.Expr):haxe.macro.Expr {
        var code = a + " + " + b + " + " + c;
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        var x = MacroTools.sum3("10", "20", "30");
        trace(x);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["60", "done"]);
}

// ================================================================
// COMPILE-TIME TYPE-LEVEL COMPUTATION
// ================================================================

/// Test macro that does compile-time power computation
#[test]
fn test_compile_time_power() {
    let source = r#"
class MacroTools {
    macro static function power(base:haxe.macro.Expr, exp:haxe.macro.Expr):haxe.macro.Expr {
        var result = 1;
        var i = 0;
        while (i < exp) {
            result = result * base;
            i = i + 1;
        }
        return result;
    }
}
class Main {
    static function main() {
        trace(MacroTools.power(2, 10));
        trace(MacroTools.power(3, 5));
        trace(MacroTools.power(10, 3));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["1024", "243", "1000", "done"]);
}

// ================================================================
// NESTED MACRO CALLS
// ================================================================

/// Test macro calls nested within each other
#[test]
fn test_nested_macro_calls() {
    let source = r#"
class MacroTools {
    macro static function add(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }
    macro static function mul(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a * b;
    }
}
class Main {
    static function main() {
        trace(MacroTools.add(MacroTools.mul(3, 4), MacroTools.mul(5, 6)));
        trace(MacroTools.mul(MacroTools.add(2, 3), MacroTools.add(4, 6)));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "50", "done"]);
}

// ================================================================
// MIXED RUNTIME + MACRO — macro result used in runtime computation
// ================================================================

/// Macro-expanded constants used in runtime loop
#[test]
fn test_macro_constants_in_runtime_loop() {
    let source = r#"
class MacroTools {
    macro static function sumUpTo(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = 0;
        var i = 1;
        while (i <= n) {
            result = result + i;
            i = i + 1;
        }
        return result;
    }
}
class Main {
    static function main() {
        var total = MacroTools.sumUpTo(10);
        trace(total);
        var sum = 0;
        var i = 0;
        while (i < total) {
            sum = sum + 1;
            i = i + 1;
        }
        trace(sum);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["55", "55", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time hash/checksum
// ================================================================

/// Macro computes a simple hash of a string at compile time
#[test]
fn test_compile_time_string_hash() {
    let source = r#"
class MacroTools {
    macro static function hashString(s:haxe.macro.Expr):haxe.macro.Expr {
        var hash = 0;
        var i = 0;
        var len = s.length;
        while (i < len) {
            hash = hash * 31 + i + 1;
            i = i + 1;
        }
        return hash;
    }
}
class Main {
    static function main() {
        trace(MacroTools.hashString("hello"));
        trace(MacroTools.hashString(""));
        trace(MacroTools.hashString("abc"));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // hash("hello"): len=5, loop 5 times: 1→33→1026→31810→986115
    // hash(""): len=0, loop 0 times: 0
    // hash("abc"): len=3, loop 3 times: 1→33→1026
    assert_eq!(lines, vec!["986115", "0", "1026", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time bit manipulation
// ================================================================

/// Macro computes bitmask at compile time from flag positions
#[test]
fn test_compile_time_bitmask() {
    let source = r#"
class MacroTools {
    macro static function bitmask(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = 1;
        var i = 0;
        while (i < n) {
            result = result * 2;
            i = i + 1;
        }
        return result;
    }
    macro static function combineMask(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        var va = 1;
        var i = 0;
        while (i < a) { va = va * 2; i = i + 1; }
        var vb = 1;
        var j = 0;
        while (j < b) { vb = vb * 2; j = j + 1; }
        return va + vb;
    }
}
class Main {
    static function main() {
        trace(MacroTools.bitmask(0));
        trace(MacroTools.bitmask(3));
        trace(MacroTools.bitmask(8));
        trace(MacroTools.combineMask(2, 4));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["1", "8", "256", "20", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time GCD (Euclidean algorithm)
// ================================================================

/// Macro computes GCD at compile time using Euclidean algorithm
#[test]
fn test_compile_time_gcd() {
    let source = r#"
class MacroTools {
    macro static function gcd(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        while (b != 0) {
            var temp = b;
            b = a - (a / b) * b;
            a = temp;
        }
        return a;
    }
    macro static function lcm(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        var ga = a;
        var gb = b;
        while (gb != 0) {
            var temp = gb;
            gb = ga - (ga / gb) * gb;
            ga = temp;
        }
        return (a / ga) * b;
    }
}
class Main {
    static function main() {
        trace(MacroTools.gcd(48, 18));
        trace(MacroTools.gcd(100, 75));
        trace(MacroTools.gcd(17, 13));
        trace(MacroTools.lcm(4, 6));
        trace(MacroTools.lcm(12, 8));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["6", "25", "1", "12", "24", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time Roman numeral converter
// ================================================================

/// Macro converts integer to Roman numeral string at compile time
#[test]
fn test_compile_time_roman_numerals() {
    let source = r#"
class MacroTools {
    macro static function toRoman(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = "";
        var remaining = n;
        while (remaining >= 1000) { result = result + "M"; remaining = remaining - 1000; }
        while (remaining >= 900) { result = result + "CM"; remaining = remaining - 900; }
        while (remaining >= 500) { result = result + "D"; remaining = remaining - 500; }
        while (remaining >= 400) { result = result + "CD"; remaining = remaining - 400; }
        while (remaining >= 100) { result = result + "C"; remaining = remaining - 100; }
        while (remaining >= 90) { result = result + "XC"; remaining = remaining - 90; }
        while (remaining >= 50) { result = result + "L"; remaining = remaining - 50; }
        while (remaining >= 40) { result = result + "XL"; remaining = remaining - 40; }
        while (remaining >= 10) { result = result + "X"; remaining = remaining - 10; }
        while (remaining >= 9) { result = result + "IX"; remaining = remaining - 9; }
        while (remaining >= 5) { result = result + "V"; remaining = remaining - 5; }
        while (remaining >= 4) { result = result + "IV"; remaining = remaining - 4; }
        while (remaining >= 1) { result = result + "I"; remaining = remaining - 1; }
        return result;
    }
}
class Main {
    static function main() {
        trace(MacroTools.toRoman(42));
        trace(MacroTools.toRoman(2024));
        trace(MacroTools.toRoman(3999));
        trace(MacroTools.toRoman(1));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["XLII", "MMXXIV", "MMMCMXCIX", "I", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time expression unrolling
// ================================================================

/// Macro unrolls a sum expression at compile time (simulates loop unrolling)
#[test]
fn test_compile_time_unrolled_sum() {
    let source = r#"
class MacroTools {
    macro static function sumRange(start:haxe.macro.Expr, end:haxe.macro.Expr):haxe.macro.Expr {
        var result = 0;
        var i = start;
        while (i <= end) {
            result = result + i;
            i = i + 1;
        }
        return result;
    }
    macro static function sumSquaresRange(start:haxe.macro.Expr, end:haxe.macro.Expr):haxe.macro.Expr {
        var result = 0;
        var i = start;
        while (i <= end) {
            result = result + i * i;
            i = i + 1;
        }
        return result;
    }
}
class Main {
    static function main() {
        trace(MacroTools.sumRange(1, 100));
        trace(MacroTools.sumSquaresRange(1, 10));
        trace(MacroTools.sumRange(5, 5));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // sum(1..100) = 5050, sumSquares(1..10) = 385, sum(5..5) = 5
    assert_eq!(lines, vec!["5050", "385", "5", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time string encoding
// ================================================================

/// Macro encodes string with simple Caesar cipher at compile time
#[test]
fn test_compile_time_string_encoding() {
    let source = r#"
class MacroTools {
    macro static function reverseString(s:haxe.macro.Expr):haxe.macro.Expr {
        var result = "";
        var i = s.length - 1;
        while (i >= 0) {
            result = result + s.charAt(i);
            i = i - 1;
        }
        return result;
    }
    macro static function padLeft(s:haxe.macro.Expr, width:haxe.macro.Expr, ch:haxe.macro.Expr):haxe.macro.Expr {
        var result = s;
        while (result.length < width) {
            result = ch + result;
        }
        return result;
    }
}
class Main {
    static function main() {
        trace(MacroTools.reverseString("hello"));
        trace(MacroTools.reverseString("abcdef"));
        trace(MacroTools.padLeft("42", 6, "0"));
        trace(MacroTools.padLeft("hi", 5, "."));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["olleh", "fedcba", "000042", "...hi", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: deeply nested macro composition
// ================================================================

/// Test 3+ levels of nested macro calls
#[test]
fn test_deep_nested_macro_composition() {
    let source = r#"
class M {
    macro static function add(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }
    macro static function mul(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a * b;
    }
    macro static function neg(x:haxe.macro.Expr):haxe.macro.Expr {
        return 0 - x;
    }
}
class Main {
    static function main() {
        // (3+4) * (5+6) = 7 * 11 = 77
        trace(M.mul(M.add(3, 4), M.add(5, 6)));
        // ((2*3) + (4*5)) = 6 + 20 = 26
        trace(M.add(M.mul(2, 3), M.mul(4, 5)));
        // neg(add(10, neg(3))) = neg(10 + -3) = neg(7) = -7
        trace(M.neg(M.add(10, M.neg(3))));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["77", "26", "-7", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time sequence generation
// ================================================================

/// Macro generates collatz sequence length at compile time
#[test]
fn test_compile_time_collatz() {
    let source = r#"
class MacroTools {
    macro static function collatzLength(n:haxe.macro.Expr):haxe.macro.Expr {
        var steps = 0;
        var current = n;
        while (current != 1) {
            if (current - (current / 2) * 2 == 0) {
                current = current / 2;
            } else {
                current = current * 3 + 1;
            }
            steps = steps + 1;
        }
        return steps;
    }
}
class Main {
    static function main() {
        trace(MacroTools.collatzLength(1));
        trace(MacroTools.collatzLength(6));
        trace(MacroTools.collatzLength(27));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // collatz(1)=0, collatz(6)=8 (6→3→10→5→16→8→4→2→1), collatz(27)=111
    assert_eq!(lines, vec!["0", "8", "111", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time base conversion
// ================================================================

/// Macro converts integer to binary string at compile time
#[test]
fn test_compile_time_base_conversion() {
    let source = r#"
class MacroTools {
    macro static function toBinary(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n == 0) return "0";
        var result = "";
        var remaining = n;
        while (remaining > 0) {
            var bit = remaining - (remaining / 2) * 2;
            if (bit == 0) {
                result = "0" + result;
            } else {
                result = "1" + result;
            }
            remaining = remaining / 2;
        }
        return result;
    }
    macro static function countBits(n:haxe.macro.Expr):haxe.macro.Expr {
        var count = 0;
        var remaining = n;
        while (remaining > 0) {
            count = count + remaining - (remaining / 2) * 2;
            remaining = remaining / 2;
        }
        return count;
    }
}
class Main {
    static function main() {
        trace(MacroTools.toBinary(0));
        trace(MacroTools.toBinary(10));
        trace(MacroTools.toBinary(255));
        trace(MacroTools.countBits(0));
        trace(MacroTools.countBits(7));
        trace(MacroTools.countBits(255));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["0", "1010", "11111111", "0", "3", "8", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: macro-generated expression with reification + nesting
// ================================================================

/// Test combining $e{} reification with nested macro calls
#[test]
fn test_reification_with_nested_macros() {
    let source = r#"
class M {
    macro static function add(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return a + b;
    }
    macro static function wrapTrace(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro trace($e{e});
    }
}
class Main {
    static function main() {
        M.wrapTrace(M.add(10, 20));
        M.wrapTrace(M.add(M.add(1, 2), M.add(3, 4)));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["30", "10", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time matrix determinant
// ================================================================

/// Macro computes 2x2 matrix determinant at compile time
#[test]
fn test_compile_time_determinant() {
    let source = r#"
class MacroTools {
    macro static function det2x2(a:haxe.macro.Expr, b:haxe.macro.Expr, c:haxe.macro.Expr, d:haxe.macro.Expr):haxe.macro.Expr {
        return a * d - b * c;
    }
}
class Main {
    static function main() {
        trace(MacroTools.det2x2(1, 0, 0, 1));
        trace(MacroTools.det2x2(2, 3, 1, 4));
        trace(MacroTools.det2x2(5, 7, 2, 3));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // det(I) = 1, det([[2,3],[1,4]]) = 8-3 = 5, det([[5,7],[2,3]]) = 15-14 = 1
    assert_eq!(lines, vec!["1", "5", "1", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: multiple macros from different classes in one expression
// ================================================================

/// Test cross-class macro composition
#[test]
fn test_cross_class_macro_composition() {
    let source = r#"
class MathMacros {
    macro static function square(x:haxe.macro.Expr):haxe.macro.Expr {
        return x * x;
    }
    macro static function cube(x:haxe.macro.Expr):haxe.macro.Expr {
        return x * x * x;
    }
}
class StringMacros {
    macro static function intToStr(n:haxe.macro.Expr):haxe.macro.Expr {
        return "" + n;
    }
}
class Main {
    static function main() {
        trace(StringMacros.intToStr(MathMacros.square(5)));
        trace(StringMacros.intToStr(MathMacros.cube(3)));
        trace(MathMacros.square(MathMacros.square(3)));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["25", "27", "81", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: compile-time digit sum and digit manipulation
// ================================================================

/// Macro computes digit sum and digit count at compile time
#[test]
fn test_compile_time_digit_operations() {
    let source = r#"
class MacroTools {
    macro static function digitSum(n:haxe.macro.Expr):haxe.macro.Expr {
        var sum = 0;
        var remaining = n;
        while (remaining > 0) {
            sum = sum + remaining - (remaining / 10) * 10;
            remaining = remaining / 10;
        }
        return sum;
    }
    macro static function digitCount(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n == 0) return 1;
        var count = 0;
        var remaining = n;
        while (remaining > 0) {
            count = count + 1;
            remaining = remaining / 10;
        }
        return count;
    }
    macro static function reverseDigits(n:haxe.macro.Expr):haxe.macro.Expr {
        var result = 0;
        var remaining = n;
        while (remaining > 0) {
            result = result * 10 + remaining - (remaining / 10) * 10;
            remaining = remaining / 10;
        }
        return result;
    }
}
class Main {
    static function main() {
        trace(MacroTools.digitSum(12345));
        trace(MacroTools.digitSum(999));
        trace(MacroTools.digitCount(12345));
        trace(MacroTools.digitCount(0));
        trace(MacroTools.reverseDigits(12345));
        trace(MacroTools.reverseDigits(1000));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["15", "27", "5", "1", "54321", "1", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: $v{} with compile-time string building
// ================================================================

/// Test $v{} value splicing with string built from loops
#[test]
fn test_value_splice_with_computed_strings() {
    let source = r#"
class MacroTools {
    macro static function makeCSV(a:haxe.macro.Expr, b:haxe.macro.Expr, c:haxe.macro.Expr):haxe.macro.Expr {
        var result = a + "," + b + "," + c;
        return macro $v{result};
    }
    macro static function repeatWithSep(s:haxe.macro.Expr, n:haxe.macro.Expr, sep:haxe.macro.Expr):haxe.macro.Expr {
        var result = "";
        var i = 0;
        while (i < n) {
            if (i > 0) result = result + sep;
            result = result + s;
            i = i + 1;
        }
        return macro $v{result};
    }
}
class Main {
    static function main() {
        trace(MacroTools.makeCSV("a", "b", "c"));
        trace(MacroTools.repeatWithSep("x", 4, "-"));
        trace(MacroTools.repeatWithSep("ha", 3, " "));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["a,b,c", "x-x-x-x", "ha ha ha", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: macro with early return branching
// ================================================================

/// Test macros with multiple return paths based on compile-time conditions
#[test]
fn test_macro_multi_return_paths() {
    let source = r#"
class MacroTools {
    macro static function classify(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n < 0) return "negative";
        if (n == 0) return "zero";
        if (n < 10) return "small";
        if (n < 100) return "medium";
        return "large";
    }
    macro static function sign(n:haxe.macro.Expr):haxe.macro.Expr {
        if (n < 0) return -1;
        if (n > 0) return 1;
        return 0;
    }
}
class Main {
    static function main() {
        trace(MacroTools.classify(-5));
        trace(MacroTools.classify(0));
        trace(MacroTools.classify(7));
        trace(MacroTools.classify(42));
        trace(MacroTools.classify(999));
        trace(MacroTools.sign(-100));
        trace(MacroTools.sign(0));
        trace(MacroTools.sign(42));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(
        lines,
        vec!["negative", "zero", "small", "medium", "large", "-1", "0", "1", "done"]
    );
}

// ================================================================
// REAL METAPROGRAMMING: Context.parse with complex generated code
// ================================================================

/// Test Context.parse generating arithmetic expressions from strings
#[test]
fn test_context_parse_complex_generation() {
    let source = r#"
class MacroTools {
    macro static function buildMulChain(a:haxe.macro.Expr, b:haxe.macro.Expr, c:haxe.macro.Expr):haxe.macro.Expr {
        var code = a + " * " + b + " * " + c;
        return Context.parse(code, Context.currentPos());
    }
    macro static function buildParenExpr(a:haxe.macro.Expr, b:haxe.macro.Expr, c:haxe.macro.Expr):haxe.macro.Expr {
        var code = "(" + a + " + " + b + ") * " + c;
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        trace(MacroTools.buildMulChain("2", "3", "7"));
        trace(MacroTools.buildParenExpr("3", "4", "5"));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["42", "35", "done"]);
}

// ================================================================
// REAL METAPROGRAMMING: macro result fed into runtime for-in loop
// ================================================================

/// Macro-computed constant used as loop bound and in arithmetic
#[test]
fn test_macro_computed_bounds_in_runtime_loop() {
    let source = r#"
class MacroTools {
    macro static function triangular(n:haxe.macro.Expr):haxe.macro.Expr {
        return n * (n + 1) / 2;
    }
}
class Main {
    static function main() {
        var bound = MacroTools.triangular(10);
        trace(bound);
        var product = 1;
        var i = 1;
        while (i <= 5) {
            product = product * i;
            i = i + 1;
        }
        trace(product);
        trace(bound + product);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["55", "120", "175", "done"]);
}
