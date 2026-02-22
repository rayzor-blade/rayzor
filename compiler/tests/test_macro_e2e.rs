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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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
import haxe.macro.Context;
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

// ================================================================
// CODE GENERATION: macros that produce runtime code via $e{} splicing
// ================================================================

/// Macro generates `(x * x)` expression — x is a RUNTIME variable, not a constant.
/// The macro produces code structure at compile time; values are computed at runtime.
#[test]
fn test_codegen_runtime_variable_square() {
    let source = r#"
class MacroTools {
    macro static function square(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{e} * $e{e});
    }
}
class Main {
    static function main() {
        var x = 7;
        trace(MacroTools.square(x));
        var y = 3;
        trace(MacroTools.square(y));
        trace(MacroTools.square(x + y));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // square(x) generates (x * x) = (7*7) = 49
    // square(y) generates (y * y) = (3*3) = 9
    // square(x + y) generates ((x+y) * (x+y)) = (10*10) = 100
    assert_eq!(lines, vec!["49", "9", "100", "done"]);
}

/// Macro generates `(0 - e)` negate and `(a + b)` sum on runtime variables.
#[test]
fn test_codegen_runtime_arithmetic_transforms() {
    let source = r#"
class M {
    macro static function negate(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro (0 - $e{e});
    }
    macro static function sum(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{a} + $e{b});
    }
    macro static function diff(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{a} - $e{b});
    }
}
class Main {
    static function main() {
        var x = 10;
        var y = 3;
        trace(M.negate(x));
        trace(M.sum(x, y));
        trace(M.diff(x, y));
        trace(M.sum(M.negate(x), y));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // negate(x) → (0 - x) = -10
    // sum(x, y) → (x + y) = 13
    // diff(x, y) → (x - y) = 7
    // sum(negate(x), y) → negate(x) generates (0 - x), then sum(that, y) → ((0-x) + y) = -7
    assert_eq!(lines, vec!["-10", "13", "7", "-7", "done"]);
}

/// Macro generates `(e * 2)` and `(e + 1)` — compose to produce complex runtime expressions.
/// Tests that macro-generated code composes correctly with runtime state.
#[test]
fn test_codegen_composed_runtime_transforms() {
    let source = r#"
class M {
    macro static function double(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{e} * 2);
    }
    macro static function inc(e:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{e} + 1);
    }
}
class Main {
    static function main() {
        var x = 5;
        trace(M.double(x));
        trace(M.inc(x));
        trace(M.double(M.inc(x)));
        trace(M.inc(M.double(x)));
        trace(M.double(M.double(x)));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // double(x) → (x * 2) = 10
    // inc(x) → (x + 1) = 6
    // double(inc(x)) → inc(x) = 6, double(6) = 12... no, nested:
    //   inc(x) expands to (x + 1), double(that) expands to ((x + 1) * 2) = 12
    // inc(double(x)) → double(x) = (x * 2), inc(that) = ((x * 2) + 1) = 11
    // double(double(x)) → double(x) = (x * 2), double(that) = ((x * 2) * 2) = 20
    assert_eq!(lines, vec!["10", "6", "12", "11", "20", "done"]);
}

// ================================================================
// CODE GENERATION: Context.parse producing runtime code from strings
// ================================================================

/// Context.parse generates a multiplication chain as runtime code.
/// The macro BUILDS a code string at compile time, then parses it into
/// an expression AST that runs at runtime.
#[test]
fn test_codegen_context_parse_multiply_chain() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function genFactorialExpr(n:haxe.macro.Expr):haxe.macro.Expr {
        var code = "1";
        var i = 1;
        while (i <= n) {
            code = code + " * " + i;
            i = i + 1;
        }
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        trace(MacroTools.genFactorialExpr(5));
        trace(MacroTools.genFactorialExpr(3));
        trace(MacroTools.genFactorialExpr(1));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // genFactorialExpr(5) → "1 * 1 * 2 * 3 * 4 * 5" → 120
    // genFactorialExpr(3) → "1 * 1 * 2 * 3" → 6
    // genFactorialExpr(1) → "1 * 1" → 1
    assert_eq!(lines, vec!["120", "6", "1", "done"]);
}

/// Context.parse generates a sum-of-additions chain as runtime code.
/// Demonstrates generating `x + x + x + ...` where x is a RUNTIME variable.
#[test]
fn test_codegen_context_parse_runtime_var_chain() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function repeatAdd(varName:haxe.macro.Expr, n:haxe.macro.Expr):haxe.macro.Expr {
        var code = varName;
        var i = 1;
        while (i < n) {
            code = code + " + " + varName;
            i = i + 1;
        }
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        var x = 10;
        trace(MacroTools.repeatAdd("x", 3));
        var y = 7;
        trace(MacroTools.repeatAdd("y", 4));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // repeatAdd("x", 3) → generates "x + x + x" = 30
    // repeatAdd("y", 4) → generates "y + y + y + y" = 28
    assert_eq!(lines, vec!["30", "28", "done"]);
}

/// Context.parse generates a complete trace call as runtime code.
/// The macro builds the ENTIRE `trace(expr)` call from a code string.
#[test]
fn test_codegen_context_parse_trace_call() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function genTraceCall(value:haxe.macro.Expr):haxe.macro.Expr {
        var code = "trace(" + value + ")";
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        MacroTools.genTraceCall("42");
        MacroTools.genTraceCall("100 + 200");
        MacroTools.genTraceCall("7 * 8");
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // genTraceCall("42") → generates `trace(42)` → 42
    // genTraceCall("100 + 200") → generates `trace(100 + 200)` → 300
    // genTraceCall("7 * 8") → generates `trace(7 * 8)` → 56
    assert_eq!(lines, vec!["42", "300", "56", "done"]);
}

// ================================================================
// CODE GENERATION: debug instrumentation — macros generating trace wrappers
// ================================================================

/// Macro generates `trace("label = " + expr)` where label is a compile-time
/// computed string and expr is runtime code. Demonstrates mixed compile-time
/// and runtime code generation.
#[test]
fn test_codegen_debug_var_instrumentation() {
    let source = r#"
class MacroTools {
    macro static function debugVar(label:haxe.macro.Expr, e:haxe.macro.Expr):haxe.macro.Expr {
        var prefix = label + " = ";
        return macro trace($v{prefix} + $e{e});
    }
}
class Main {
    static function main() {
        var x = 42;
        var y = 7;
        MacroTools.debugVar("x", x);
        MacroTools.debugVar("y", y);
        MacroTools.debugVar("x+y", x + y);
        MacroTools.debugVar("x*y", x * y);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // debugVar("x", x) → generates trace("x = " + x) → "x = 42"
    // debugVar("y", y) → generates trace("y = " + y) → "y = 7"
    // debugVar("x+y", x + y) → generates trace("x+y = " + (x+y)) → "x+y = 49"
    // debugVar("x*y", x * y) → generates trace("x*y = " + (x*y)) → "x*y = 294"
    assert_eq!(
        lines,
        vec!["x = 42", "y = 7", "x+y = 49", "x*y = 294", "done"]
    );
}

/// Macro generates trace calls with compile-time computed prefix strings.
/// The entire trace("prefix: " + value) is macro-generated runtime code.
#[test]
fn test_codegen_trace_with_compile_time_labels() {
    let source = r#"
class MacroTools {
    macro static function traceUpper(label:haxe.macro.Expr, e:haxe.macro.Expr):haxe.macro.Expr {
        var upperLabel = "[" + label + "] ";
        return macro trace($v{upperLabel} + $e{e});
    }
}
class Main {
    static function main() {
        var result = 100;
        MacroTools.traceUpper("RESULT", result);
        MacroTools.traceUpper("DOUBLED", result * 2);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["[RESULT] 100", "[DOUBLED] 200", "done"]);
}

// ================================================================
// CODE GENERATION: expression structure generation via $e{} splicing
// ================================================================

/// Macro generates Pythagorean distance expression: sqrt(a*a + b*b)
/// Demonstrates generating compound runtime expressions from $e{} splicing.
#[test]
fn test_codegen_pythagorean_expression() {
    let source = r#"
class MacroTools {
    macro static function sumOfSquares(a:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($e{a} * $e{a} + $e{b} * $e{b});
    }
}
class Main {
    static function main() {
        var x = 3;
        var y = 4;
        trace(MacroTools.sumOfSquares(x, y));
        trace(MacroTools.sumOfSquares(5, 12));
        var a = 1;
        var b = 1;
        trace(MacroTools.sumOfSquares(a, b));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // sumOfSquares(x, y) → (x*x + y*y) = (9 + 16) = 25
    // sumOfSquares(5, 12) → (5*5 + 12*12) = (25 + 144) = 169
    // sumOfSquares(a, b) → (a*a + b*b) = (1 + 1) = 2
    assert_eq!(lines, vec!["25", "169", "2", "done"]);
}

/// Macro generates linear expression `a*x + b` where a,b are compile-time
/// constants and x is a runtime variable. Classic code generation pattern.
#[test]
fn test_codegen_linear_expression() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function linear(a:haxe.macro.Expr, x:haxe.macro.Expr, b:haxe.macro.Expr):haxe.macro.Expr {
        return macro ($v{a} * $e{x} + $v{b});
    }
}
class Main {
    static function main() {
        var x = 10;
        trace(MacroTools.linear(2, x, 5));
        trace(MacroTools.linear(3, x, 1));
        var y = 0;
        trace(MacroTools.linear(7, y, 3));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // linear(2, x, 5) → (2 * x + 5) = (20 + 5) = 25
    // linear(3, x, 1) → (3 * x + 1) = (30 + 1) = 31
    // linear(7, y, 3) → (7 * y + 3) = (0 + 3) = 3
    assert_eq!(lines, vec!["25", "31", "3", "done"]);
}

// ================================================================
// CODE GENERATION: Context.parse generating expressions with variables
// ================================================================

/// Context.parse generates code string referencing runtime variables.
/// The generated code is a string built at compile time, parsed to AST,
/// then compiled as normal runtime code.
#[test]
fn test_codegen_context_parse_variable_expression() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function genExpr(code:haxe.macro.Expr):haxe.macro.Expr {
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        var a = 10;
        var b = 20;
        var c = 5;
        trace(MacroTools.genExpr("a + b"));
        trace(MacroTools.genExpr("a * b + c"));
        trace(MacroTools.genExpr("(a + b) * c"));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // genExpr("a + b") → parses to `a + b` → 30
    // genExpr("a * b + c") → parses to `a * b + c` → 205
    // genExpr("(a + b) * c") → parses to `(a + b) * c` → 150
    assert_eq!(lines, vec!["30", "205", "150", "done"]);
}

/// Context.parse generating code that calls trace with runtime computation.
/// The macro builds the ENTIRE function call as a code string.
#[test]
fn test_codegen_context_parse_generated_function_call() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function genCode(code:haxe.macro.Expr):haxe.macro.Expr {
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        var x = 42;
        MacroTools.genCode("trace(x)");
        MacroTools.genCode("trace(x + 1)");
        MacroTools.genCode("trace(x * 2)");
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // genCode("trace(x)") → generates `trace(x)` → 42
    // genCode("trace(x + 1)") → generates `trace(x + 1)` → 43
    // genCode("trace(x * 2)") → generates `trace(x * 2)` → 84
    assert_eq!(lines, vec!["42", "43", "84", "done"]);
}

// ================================================================
// CODE GENERATION: $i{} identifier splicing for dynamic dispatch
// ================================================================

/// Macro generates calls to different functions via $i{} identifier splicing.
/// The function name is determined at compile time, but the call executes at runtime.
#[test]
fn test_codegen_identifier_splice_function_dispatch() {
    let source = r#"
class MacroTools {
    macro static function call(name:haxe.macro.Expr, arg:haxe.macro.Expr):haxe.macro.Expr {
        return macro $i{name}($e{arg});
    }
}
class Main {
    static function double(x:Int):Int { return x * 2; }
    static function triple(x:Int):Int { return x * 3; }

    static function main() {
        var x = 10;
        trace(MacroTools.call("double", x));
        trace(MacroTools.call("triple", x));
        trace(MacroTools.call("double", 7));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // call("double", x) → generates `double(x)` → 20
    // call("triple", x) → generates `triple(x)` → 30
    // call("double", 7) → generates `double(7)` → 14
    assert_eq!(lines, vec!["20", "30", "14", "done"]);
}

// ================================================================
// CODE GENERATION: compile-time computation + runtime code generation
// ================================================================

/// Macro computes a coefficient at compile time, then generates runtime
/// multiplication code using that coefficient. Demonstrates the boundary
/// between compile-time and runtime: coefficient is computed by the macro,
/// but the multiplication with the runtime variable happens at runtime.
#[test]
fn test_codegen_compile_time_coefficient() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function scaleByPower(base:haxe.macro.Expr, exp:haxe.macro.Expr, e:haxe.macro.Expr):haxe.macro.Expr {
        var coeff = 1;
        var i = 0;
        while (i < exp) {
            coeff = coeff * base;
            i = i + 1;
        }
        return macro ($v{coeff} * $e{e});
    }
}
class Main {
    static function main() {
        var x = 3;
        trace(MacroTools.scaleByPower(2, 3, x));
        trace(MacroTools.scaleByPower(10, 2, x));
        var y = 7;
        trace(MacroTools.scaleByPower(2, 4, y));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // scaleByPower(2, 3, x) → coeff=8, generates (8 * x) = 24
    // scaleByPower(10, 2, x) → coeff=100, generates (100 * x) = 300
    // scaleByPower(2, 4, y) → coeff=16, generates (16 * y) = 112
    assert_eq!(lines, vec!["24", "300", "112", "done"]);
}

/// Macro builds an addition expression from a compile-time computed string
/// using Context.parse. The STRUCTURE is built at compile time via string
/// operations, then parsed into an AST that runs at runtime.
#[test]
fn test_codegen_context_parse_build_sum_expression() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function genSumTo(n:haxe.macro.Expr):haxe.macro.Expr {
        var code = "0";
        var i = 1;
        while (i <= n) {
            code = code + " + " + i;
            i = i + 1;
        }
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        trace(MacroTools.genSumTo(5));
        trace(MacroTools.genSumTo(10));
        trace(MacroTools.genSumTo(1));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // genSumTo(5) → "0 + 1 + 2 + 3 + 4 + 5" → 15
    // genSumTo(10) → "0 + 1 + 2 + ... + 10" → 55
    // genSumTo(1) → "0 + 1" → 1
    assert_eq!(lines, vec!["15", "55", "1", "done"]);
}

// ================================================================
// CODE GENERATION: complex compile+runtime hybrid patterns
// ================================================================

/// Macro generates a polynomial expression at compile time using Context.parse.
/// Given coefficients (compile time), generates `c0 + c1*x + c2*x*x + ...`
/// where x is a runtime variable.
#[test]
fn test_codegen_polynomial_expression() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function poly2(c0:haxe.macro.Expr, c1:haxe.macro.Expr, c2:haxe.macro.Expr, varName:haxe.macro.Expr):haxe.macro.Expr {
        var code = c0 + " + " + c1 + " * " + varName + " + " + c2 + " * " + varName + " * " + varName;
        return Context.parse(code, Context.currentPos());
    }
}
class Main {
    static function main() {
        var x = 3;
        trace(MacroTools.poly2(1, 2, 3, "x"));
        var y = 2;
        trace(MacroTools.poly2(5, 0, 1, "y"));
        trace(MacroTools.poly2(0, 0, 1, "x"));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // poly2(1, 2, 3, "x") → "1 + 2 * x + 3 * x * x" → 1 + 6 + 27 = 34
    // poly2(5, 0, 1, "y") → "5 + 0 * y + 1 * y * y" → 5 + 0 + 4 = 9
    // poly2(0, 0, 1, "x") → "0 + 0 * x + 1 * x * x" → 0 + 0 + 9 = 9
    assert_eq!(lines, vec!["34", "9", "9", "done"]);
}

/// Mixed pattern: compile-time string building + $v{} splice + $e{} runtime var.
/// Macro computes a string prefix at compile time, then generates runtime
/// string concatenation code.
#[test]
fn test_codegen_mixed_compile_runtime_string() {
    let source = r#"
class MacroTools {
    macro static function formatPrefix(prefix:haxe.macro.Expr, sep:haxe.macro.Expr, e:haxe.macro.Expr):haxe.macro.Expr {
        var fullPrefix = "[" + prefix + "]" + sep;
        return macro ($v{fullPrefix} + $e{e});
    }
}
class Main {
    static function main() {
        var msg = "world";
        trace(MacroTools.formatPrefix("INFO", " ", msg));
        trace(MacroTools.formatPrefix("ERR", ": ", msg));
        var num = 42;
        trace(MacroTools.formatPrefix("VAL", "=", num));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // formatPrefix("INFO", " ", msg) → "[INFO] " + msg → "[INFO] world"
    // formatPrefix("ERR", ": ", msg) → "[ERR]: " + msg → "[ERR]: world"
    // formatPrefix("VAL", "=", num) → "[VAL]=" + num → "[VAL]=42"
    assert_eq!(
        lines,
        vec!["[INFO] world", "[ERR]: world", "[VAL]=42", "done"]
    );
}

// ================================================================
// COMPILE-TIME I/O: sys.io.Process, sys.io.File, Sys
// ================================================================

/// Test sys.io.Process — run a subprocess at compile time to embed its output.
/// This is the classic Haxe macro pattern for embedding git commit hashes.
#[test]
fn test_sys_io_process_echo() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function embedCommand(cmd:haxe.macro.Expr, a:haxe.macro.Expr):haxe.macro.Expr {
        var p = new sys.io.Process(cmd, [a]);
        var out = p.stdout.readAll();
        p.close();
        return macro $v{out};
    }
}
class Main {
    static function main() {
        var result = MacroTools.embedCommand("echo", "hello_from_macro");
        trace(result);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // echo outputs with trailing newline, which gets trimmed or not depending on OS
    assert!(
        lines[0].contains("hello_from_macro"),
        "expected 'hello_from_macro' in output, got: {:?}",
        lines
    );
    assert_eq!(lines.last().unwrap(), &"done");
}

/// Test git rev-parse pattern — the canonical use case for compile-time Process.
/// Uses `git rev-parse --short HEAD` to embed current commit hash.
#[test]
fn test_sys_io_process_git_hash() {
    let source = r#"
import haxe.macro.Context;
class BuildInfo {
    macro static function getGitHash():haxe.macro.Expr {
        var p = new sys.io.Process("git", ["rev-parse", "--short", "HEAD"]);
        var hash = p.stdout.readAll();
        p.close();
        return macro $v{hash};
    }
}
class Main {
    static function main() {
        var hash = BuildInfo.getGitHash();
        trace(hash);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // Should be a 7-char hex string (git short hash) possibly with trailing whitespace
    let hash = lines[0].trim();
    assert!(
        hash.len() >= 7 && hash.chars().all(|c| c.is_ascii_hexdigit()),
        "expected git short hash, got: '{}'",
        hash
    );
    assert_eq!(lines.last().unwrap(), &"done");
}

/// Test sys.io.File.getContent — read a file at compile time and embed its content.
#[test]
fn test_sys_io_file_get_content() {
    // Create a temporary file to read at compile time
    let tmp_path = std::env::temp_dir().join("rayzor_macro_test_file.txt");
    std::fs::write(&tmp_path, "compile-time-content").expect("write temp file");
    let tmp_path_str = tmp_path.to_str().unwrap().replace('\\', "/");

    let source = format!(
        r#"
import haxe.macro.Context;
class MacroTools {{
    macro static function embedFile(path:haxe.macro.Expr):haxe.macro.Expr {{
        var content = sys.io.File.getContent(path);
        return macro $v{{content}};
    }}
}}
class Main {{
    static function main() {{
        var text = MacroTools.embedFile("{}");
        trace(text);
        trace("done");
    }}
}}
"#,
        tmp_path_str
    );
    let (stdout, stderr, success) = run_haxe_source(&source);
    // Clean up
    let _ = std::fs::remove_file(&tmp_path);

    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines, vec!["compile-time-content", "done"]);
}

/// Test Sys.getCwd() — embed current working directory at compile time.
#[test]
fn test_sys_get_cwd() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function embedCwd():haxe.macro.Expr {
        var cwd = Sys.getCwd();
        return macro $v{cwd};
    }
}
class Main {
    static function main() {
        var dir = MacroTools.embedCwd();
        trace(dir);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // Should output a valid directory path
    assert!(
        !lines[0].is_empty() && (lines[0].starts_with('/') || lines[0].contains(':')),
        "expected a directory path, got: '{}'",
        lines[0]
    );
    assert_eq!(lines.last().unwrap(), &"done");
}

/// Test Sys.systemName() — embed platform name at compile time.
#[test]
fn test_sys_system_name() {
    let source = r#"
import haxe.macro.Context;
class MacroTools {
    macro static function platform():haxe.macro.Expr {
        var name = Sys.systemName();
        return macro $v{name};
    }
}
class Main {
    static function main() {
        trace(MacroTools.platform());
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert!(
        ["Mac", "Linux", "Windows"].contains(&&*lines[0]),
        "expected platform name, got: '{}'",
        lines[0]
    );
    assert_eq!(lines.last().unwrap(), &"done");
}

/// Test Process with import — import sys.io.Process and use bare name.
#[test]
fn test_sys_io_process_with_import() {
    let source = r#"
import haxe.macro.Context;
import sys.io.Process;
class BuildInfo {
    macro static function runCommand(cmd:haxe.macro.Expr):haxe.macro.Expr {
        var p = new Process(cmd, []);
        var out = p.stdout.readAll();
        return macro $v{out};
    }
}
class Main {
    static function main() {
        var result = BuildInfo.runCommand("whoami");
        trace(result);
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    // whoami should return a non-empty username
    assert!(
        !lines[0].trim().is_empty(),
        "expected username from whoami, got: '{}'",
        lines[0]
    );
    assert_eq!(lines.last().unwrap(), &"done");
}

// =====================================================
// ClassRegistry E2E tests
// =====================================================

#[test]
fn test_class_registry_static_method() {
    // Macro calls a static method on a non-macro class in the same file
    let source = r#"
import haxe.macro.Context;
class MathHelper {
    static function square(x:Int):Int {
        return x * x;
    }
    static function add(a:Int, b:Int):Int {
        return a + b;
    }
}
class MacroTools {
    macro static function compileSquare(x:Int):haxe.macro.Expr {
        var result = MathHelper.square(x);
        return macro $v{result};
    }
    macro static function compileAdd(a:Int, b:Int):haxe.macro.Expr {
        var result = MathHelper.add(a, b);
        return macro $v{result};
    }
}
class Main {
    static function main() {
        trace(MacroTools.compileSquare(7));
        trace(MacroTools.compileAdd(10, 20));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines[0], "49");
    assert_eq!(lines[1], "30");
    assert_eq!(lines[2], "done");
}

#[test]
fn test_class_registry_constructor_and_instance_method() {
    // Macro constructs a user-defined class and calls an instance method
    let source = r#"
import haxe.macro.Context;
class Point {
    var x:Int;
    var y:Int;
    function new(x:Int, y:Int) {
        this.x = x;
        this.y = y;
    }
    function sum():Int {
        return this.x + this.y;
    }
}
class MacroTools {
    macro static function compilePointSum(px:Int, py:Int):haxe.macro.Expr {
        var p = new Point(px, py);
        var result = p.sum();
        return macro $v{result};
    }
}
class Main {
    static function main() {
        trace(MacroTools.compilePointSum(3, 4));
        trace(MacroTools.compilePointSum(10, 20));
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines[0], "7");
    assert_eq!(lines[1], "30");
    assert_eq!(lines[2], "done");
}

#[test]
fn test_class_registry_constructor_field_access() {
    // Macro constructs a class and reads its fields directly
    let source = r#"
import haxe.macro.Context;
class Config {
    var name:String;
    var value:Int;
    function new(name:String, value:Int) {
        this.name = name;
        this.value = value;
    }
}
class MacroTools {
    macro static function getConfigValue():haxe.macro.Expr {
        var c = new Config("test", 42);
        var v = c.value;
        return macro $v{v};
    }
}
class Main {
    static function main() {
        trace(MacroTools.getConfigValue());
        trace("done");
    }
}
"#;
    let (stdout, stderr, success) = run_haxe_source(source);
    assert!(success, "compilation failed: {}", stderr);
    let lines = extract_trace_lines(&stdout);
    assert_eq!(lines[0], "42");
    assert_eq!(lines[1], "done");
}
