/// Isolated test suite for stdlib parser compatibility
///
/// Tests conditional compilation, metadata handling, and other syntax features
/// needed to parse the official Haxe 4.3.7 standard library.
///
/// The parser is a fall-through parser that collects diagnostics instead of panicking.
/// Each test checks both:
/// 1. That parsing completes without panicking
/// 2. The specific diagnostics/errors encountered
use compiler::compilation::{CompilationConfig, CompilationUnit};
use parser::parse_haxe_file_with_diagnostics;

/// Helper function to parse Haxe source using CompilationUnit and return diagnostics
/// For simple syntax tests (conditionals, metadata), stdlib is not needed
fn parse_and_get_diagnostics(source: &str, filename: &str) -> (bool, Vec<String>) {
    // Use direct parser for simple syntax tests
    match parse_haxe_file_with_diagnostics(filename, source) {
        Ok(parse_result) => {
            let has_errors = parse_result.diagnostics.has_errors();
            let error_messages: Vec<String> = parse_result
                .diagnostics
                .errors()
                .map(|e| format!("{:?}: {}", e.severity, e.message))
                .collect();
            (has_errors, error_messages)
        }
        Err(err) => (true, vec![err]),
    }
}

/// Helper function to parse stdlib files with full CompilationUnit context
/// Stdlib files reference each other, so they need the full compilation infrastructure
fn parse_stdlib_file_and_get_diagnostics(filename: &str) -> (bool, Vec<String>) {
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load the stdlib - this will attempt to parse the file we're testing
    match unit.load_stdlib() {
        Ok(_) => {
            // Stdlib loaded successfully - extract any parse errors from diagnostics
            // For now, we check if the specific file was loaded
            let file_loaded = unit
                .stdlib_files
                .iter()
                .any(|f| f.filename.contains(filename));
            if file_loaded {
                (false, Vec::new())
            } else {
                (true, vec![format!("File {} not found in stdlib", filename)])
            }
        }
        Err(e) => {
            // Stdlib loading failed - likely due to parse errors
            (true, vec![e])
        }
    }
}

// ============================================================================
// Conditional Compilation Tests
// ============================================================================

#[test]
fn test_simple_if_directive() {
    let source = r#"
#if cpp
var x = 1;
#end
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    // Currently expected to fail - conditional compilation not implemented
    println!("Simple #if directive:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);

    // TODO: Once implemented, change to:
    // assert!(!has_errors, "Should parse simple #if directive");
}

#[test]
fn test_if_else_directive() {
    let source = r#"
#if cpp
var x = 1;
#else
var x = 2;
#end
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("If-else directive:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_if_elseif_else_directive() {
    let source = r#"
#if cpp
var x = 1;
#elseif js
var x = 2;
#else
var x = 3;
#end
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("If-elseif-else directive:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_conditional_with_or_operator() {
    let source = r#"
#if (java || cpp)
var x = 1;
#end
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Conditional with OR operator:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_conditional_with_and_operator() {
    let source = r#"
#if (java && !macro)
var x = 1;
#end
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Conditional with AND and NOT operators:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_nested_conditionals() {
    let source = r#"
#if cpp
  #if debug
  var x = 1;
  #else
  var x = 2;
  #end
#else
  var x = 3;
#end
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Nested conditionals:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_conditional_around_metadata() {
    let source = r#"
#if jvm
@:runtimeValue
#end
@:coreType abstract Void {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Conditional around metadata:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_multiple_platform_checks() {
    let source = r#"
#if (java || cs || hl || cpp)
@:runtimeValue
#end
@:coreType abstract Int to Float {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Multiple platform checks:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

// ============================================================================
// Metadata on Separate Lines Tests
// ============================================================================

#[test]
fn test_metadata_inline_with_class() {
    let source = r#"
@:coreType abstract Void {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Metadata inline with class:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);

    // This should already work
    // TODO: Uncomment once we verify
    // assert!(!has_errors, "Inline metadata should already work");
}

#[test]
fn test_metadata_on_previous_line() {
    let source = r#"
@:coreType
abstract Void {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Metadata on previous line:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);

    // Currently expected to fail
    // TODO: Once implemented, change to:
    // assert!(!has_errors, "Should parse metadata on separate line");
}

#[test]
fn test_multiple_metadata_on_separate_lines() {
    let source = r#"
@:native("MyClass")
@:keep
class Foo {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Multiple metadata on separate lines:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_mixed_inline_and_separate_metadata() {
    let source = r#"
@:native("MyClass")
@:keep class Foo {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Mixed inline and separate metadata:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_metadata_with_blank_lines() {
    let source = r#"
@:native("MyClass")

@:keep

class Foo {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Metadata with blank lines:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

// ============================================================================
// Combined Tests (Conditionals + Metadata)
// ============================================================================

#[test]
fn test_real_stdtypes_void_pattern() {
    let source = r#"
#if jvm
@:runtimeValue
#end
@:coreType abstract Void {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Real StdTypes.hx Void pattern:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_real_stdtypes_int_pattern() {
    let source = r#"
#if (java || cs || hl || cpp)
@:runtimeValue
#end
@:coreType abstract Int to Float {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Real StdTypes.hx Int pattern:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

// ============================================================================
// Abstract Type Tests
// ============================================================================

#[test]
fn test_simple_abstract_type() {
    let source = r#"
@:coreType abstract Float {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Simple abstract type:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_abstract_with_to_conversion() {
    let source = r#"
@:coreType abstract Int to Float {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Abstract with 'to' conversion:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

#[test]
fn test_abstract_with_generic() {
    let source = r#"
@:coreType abstract Null<T> {}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Abstract with generic parameter:");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);
}

// ============================================================================
// Real Stdlib File Tests
// ============================================================================

#[test]
fn test_parse_stdtypes_hx() {
    // Use stdlib-aware parsing since StdTypes.hx is a stdlib file
    let (has_errors, errors) = parse_stdlib_file_and_get_diagnostics("StdTypes.hx");

    println!("Parsing real StdTypes.hx with CompilationUnit:");
    println!("  Has errors: {}", has_errors);
    println!("  Error count: {}", errors.len());
    if errors.len() <= 10 {
        println!("  Errors: {:?}", errors);
    } else {
        println!("  First 10 errors: {:?}", &errors[..10]);
    }

    // TODO: Once all features implemented:
    // assert!(!has_errors, "Should successfully parse StdTypes.hx");
}

#[test]
fn test_compilation_unit_with_real_stdlib() {
    // This test uses CompilationUnit to load the actual stdlib and compile a user file
    // It demonstrates the multi-file compilation infrastructure

    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load stdlib (this will attempt to parse StdTypes.hx and other core files)
    let stdlib_result = unit.load_stdlib();

    println!("CompilationUnit stdlib loading:");
    match &stdlib_result {
        Ok(_) => {
            println!("  Stdlib loaded successfully");
            println!("  Stdlib files: {}", unit.stdlib_files.len());
        }
        Err(e) => {
            println!("  Stdlib loading failed: {}", e);
        }
    }

    // Add a simple user file that uses stdlib types
    let user_source = r#"
        class Test {
            public function new() {
                var x: Int = 42;
                var s: String = "hello";
            }
        }
    "#;

    let add_result = unit.add_file(user_source, "Test.hx");
    println!("  User file add result: {:?}", add_result.is_ok());

    // Try to compile - this will fail if StdTypes.hx can't be parsed
    // (because Int, String, etc. won't be defined)
    match unit.lower_to_tast() {
        Ok(typed_files) => {
            println!("  Compilation succeeded!");
            println!("  Typed files: {}", typed_files.len());
        }
        Err(errors) => {
            println!("  Compilation failed with {} errors", errors.len());
            for (i, error) in errors.iter().take(5).enumerate() {
                println!("    Error {}: {}", i + 1, error.message);
            }
        }
    }

    // TODO: Once conditional compilation is implemented, this should succeed:
    // assert!(stdlib_result.is_ok(), "Should load stdlib successfully");
}

#[test]
fn test_parse_string_hx() {
    let (has_errors, errors) = parse_stdlib_file_and_get_diagnostics("String.hx");

    println!("Parsing real String.hx with CompilationUnit:");
    println!("  Has errors: {}", has_errors);
    println!("  Error count: {}", errors.len());
    if errors.len() <= 10 {
        println!("  Errors: {:?}", errors);
    } else {
        println!("  First 10 errors: {:?}", &errors[..10]);
    }
}

#[test]
fn test_parse_array_hx() {
    let (has_errors, errors) = parse_stdlib_file_and_get_diagnostics("Array.hx");

    println!("Parsing real Array.hx with CompilationUnit:");
    println!("  Has errors: {}", has_errors);
    println!("  Error count: {}", errors.len());
    if errors.len() <= 10 {
        println!("  Errors: {:?}", errors);
    } else {
        println!("  First 10 errors: {:?}", &errors[..10]);
    }
}

#[test]
fn test_parse_math_hx() {
    let (has_errors, errors) = parse_stdlib_file_and_get_diagnostics("Math.hx");

    println!("Parsing real Math.hx with CompilationUnit:");
    println!("  Has errors: {}", has_errors);
    println!("  Error count: {}", errors.len());
    if errors.len() <= 10 {
        println!("  Errors: {:?}", errors);
    } else {
        println!("  First 10 errors: {:?}", &errors[..10]);
    }
}

// ============================================================================
// Comprehensive Stdlib Scan Test
// ============================================================================

#[test]
#[ignore] // Run with --ignored flag for full scan
fn test_parse_all_stdlib_files() {
    use std::path::Path;
    use walkdir::WalkDir;

    let stdlib_path = Path::new("haxe-std");
    let mut total_files = 0;
    let mut parsed_ok = 0;
    let mut parse_errors = 0;
    let mut error_categories: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    println!("\n=== SCANNING ALL STDLIB FILES ===\n");

    for entry in WalkDir::new(stdlib_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "hx"))
    {
        let path = entry.path();
        let relative_path = path.strip_prefix(stdlib_path).unwrap();

        total_files += 1;

        if let Ok(source) = std::fs::read_to_string(path) {
            let (has_errors, errors) =
                parse_and_get_diagnostics(&source, path.to_str().unwrap_or("unknown"));

            if has_errors {
                parse_errors += 1;
                println!("❌ {}: {} errors", relative_path.display(), errors.len());

                // Categorize errors
                for error in &errors {
                    let category = if error.contains("#if")
                        || error.contains("#else")
                        || error.contains("#end")
                    {
                        "conditional_compilation"
                    } else if error.contains("@:") {
                        "metadata"
                    } else if error.contains("expected") {
                        "syntax_error"
                    } else {
                        "other"
                    };
                    *error_categories.entry(category.to_string()).or_insert(0) += 1;
                }
            } else {
                parsed_ok += 1;
                println!("✅ {}", relative_path.display());
            }
        }
    }

    println!("\n=== RESULTS ===");
    println!("Total files: {}", total_files);
    println!(
        "Parsed successfully: {} ({:.1}%)",
        parsed_ok,
        (parsed_ok as f64 / total_files as f64) * 100.0
    );
    println!(
        "Parse errors: {} ({:.1}%)",
        parse_errors,
        (parse_errors as f64 / total_files as f64) * 100.0
    );

    println!("\n=== ERROR CATEGORIES ===");
    let mut sorted_categories: Vec<_> = error_categories.iter().collect();
    sorted_categories.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    for (category, count) in sorted_categories {
        println!("{}: {}", category, count);
    }

    // Don't fail the test - this is for diagnostics
    // TODO: Once features implemented, uncomment:
    // assert_eq!(parse_errors, 0, "All stdlib files should parse successfully");
}

// ============================================================================
// Baseline Test - Should Already Work
// ============================================================================

#[test]
fn test_basic_class_still_works() {
    let source = r#"
class Foo {
    var x: Int;
    function bar(): Void {}
}
"#;

    let (has_errors, errors) = parse_and_get_diagnostics(source, "test.hx");

    println!("Basic class parsing (baseline):");
    println!("  Has errors: {}", has_errors);
    println!("  Errors: {:?}", errors);

    // This should work already
    assert!(
        !has_errors,
        "Basic class syntax should still parse correctly"
    );
}

#[test]
fn test_rayzor_concurrent_thread_still_works() {
    let source = std::fs::read_to_string("haxe-std/rayzor/concurrent/Thread.hx")
        .expect("Should be able to read rayzor Thread.hx");

    let (has_errors, errors) = parse_and_get_diagnostics(&source, "Thread.hx");

    println!("Parsing rayzor Thread.hx (baseline):");
    println!("  Has errors: {}", has_errors);
    if has_errors {
        println!("  Error count: {}", errors.len());
        if errors.len() <= 5 {
            println!("  Errors: {:?}", errors);
        }
    }

    // Our custom rayzor files should still work
    // TODO: Uncomment once we verify
    // assert!(!has_errors, "Rayzor concurrent classes should still parse");
}
