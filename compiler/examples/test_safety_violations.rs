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
/// Test that safety violations are actually detected and reported with proper diagnostics
use compiler::pipeline::{CompilationError, ErrorCategory, compile_haxe_source};
use diagnostics::{ErrorFormatter, SourceMap};

fn print_diagnostics(source: &str, filename: &str, errors: &[CompilationError]) {
    let mut source_map = SourceMap::new();
    source_map.add_file(filename.to_string(), source.to_string());

    let formatter = ErrorFormatter::with_colors();

    for error in errors {
        let diagnostic = error.to_diagnostic(&source_map);
        let formatted = formatter.format_diagnostic(&diagnostic, &source_map);
        eprint!("{}", formatted);
    }
}

fn main() {
    println!("\n=== Testing Safety Violation Detection ===\n");

    // Test 1: Use-after-move violation
    let source1 = r#"@:safety(true)
class Box {
    public var value: Int;
    public function new(v: Int) {
        this.value = v;
    }
}

@:safety(true)
class Main {
    static function main() {
        var x = new Box(42);
        var y = x;  // Move x to y
        trace(x.value);   // ERROR: Use after move
    }
}
"#;

    println!("Test 1: Use-After-Move\n");
    let result = compile_haxe_source(source1);

    if !result.errors.is_empty() {
        print_diagnostics(source1, "test_use_after_move.hx", &result.errors);
    } else {
        println!("✓ No errors detected (unexpected!)");
    }

    println!("\n{}\n", "=".repeat(80));

    // Test 2: Double move violation
    let source2 = r#"@:safety(true)
class Box {
    public var value: Int;
    public function new(v: Int) {
        this.value = v;
    }
}

@:safety(true)
class Main {
    static function main() {
        var x = new Box(42);
        var y = x;  // First move
        var z = x;  // ERROR: Second move (double move)
    }
}
"#;

    println!("Test 2: Double Move\n");
    let result = compile_haxe_source(source2);

    if !result.errors.is_empty() {
        print_diagnostics(source2, "test_double_move.hx", &result.errors);
    } else {
        println!("✓ No errors detected (unexpected!)");
    }

    println!("\n{}\n", "=".repeat(80));

    // Test 3: Valid code - should compile successfully
    let source3 = r#"@:safety(true)
class Box {
    public var value: Int;
    public function new(v: Int) {
        this.value = v;
    }
}

@:safety(true)
class Main {
    static function getBox(): Box {
        var local = new Box(42);
        return local;  // OK - ownership transferred
    }

    static function main() {
        var x = getBox();
        trace(x.value);  // Should be OK
    }
}
"#;

    println!("Test 3: Valid Code (Should have NO ownership errors)\n");
    let result = compile_haxe_source(source3);

    // Filter out non-ownership errors
    let ownership_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| matches!(e.category, ErrorCategory::OwnershipError))
        .collect();

    if !ownership_errors.is_empty() {
        eprintln!("✗ Unexpected ownership errors:");
        let err_refs: Vec<CompilationError> = ownership_errors.iter().map(|&e| e.clone()).collect();
        print_diagnostics(source3, "test_valid_code.hx", &err_refs);
    } else {
        println!("✓ No ownership errors - code is valid!");
    }

    println!("\n=== Tests Complete ===\n");
}
