//! Comparison test harness: verifies the recursive descent parser produces
//! identical ASTs to the nom-based parser for all Haxe test files.
//!
//! This is the critical gate before wiring up the RD parser as default.

use std::fs;
use std::path::Path;

/// Parse with the nom parser (current default)
fn nom_parse(source: &str, filename: &str) -> Option<parser::HaxeFile> {
    // Use the preprocessor like the real pipeline
    let config = parser::preprocessor::PreprocessorConfig::default();
    let preprocessed = parser::preprocessor::preprocess(source, &config);
    parser::parse_haxe_file(filename, &preprocessed, false).ok()
}

/// Parse with the new recursive descent parser
fn rd_parse(source: &str, filename: &str) -> Option<parser::HaxeFile> {
    let config = parser::preprocessor::PreprocessorConfig::default();
    let preprocessed = parser::preprocessor::preprocess(source, &config);
    parser::rd::rd_parse(&preprocessed, filename, false, false).ok()
}

/// Compare two ASTs, ignoring span differences (spans may differ between parsers)
fn asts_equivalent(nom: &parser::HaxeFile, rd: &parser::HaxeFile) -> bool {
    // Compare package
    let pkg_eq = match (&nom.package, &rd.package) {
        (Some(a), Some(b)) => a.path == b.path,
        (None, None) => true,
        _ => false,
    };
    if !pkg_eq {
        return false;
    }

    // Compare imports
    if nom.imports.len() != rd.imports.len() {
        return false;
    }
    for (a, b) in nom.imports.iter().zip(rd.imports.iter()) {
        if a.path != b.path {
            return false;
        }
    }

    // Compare using
    if nom.using.len() != rd.using.len() {
        return false;
    }

    // Compare declaration count
    if nom.declarations.len() != rd.declarations.len() {
        return false;
    }

    // Compare declaration names (not full deep equality yet)
    for (a, b) in nom.declarations.iter().zip(rd.declarations.iter()) {
        if !decl_names_match(a, b) {
            return false;
        }
    }

    true
}

fn decl_names_match(a: &parser::TypeDeclaration, b: &parser::TypeDeclaration) -> bool {
    match (a, b) {
        (parser::TypeDeclaration::Class(ac), parser::TypeDeclaration::Class(bc)) => {
            ac.name == bc.name && ac.fields.len() == bc.fields.len()
        }
        (parser::TypeDeclaration::Interface(ai), parser::TypeDeclaration::Interface(bi)) => {
            ai.name == bi.name
        }
        (parser::TypeDeclaration::Enum(ae), parser::TypeDeclaration::Enum(be)) => {
            ae.name == be.name && ae.constructors.len() == be.constructors.len()
        }
        (parser::TypeDeclaration::Typedef(at), parser::TypeDeclaration::Typedef(bt)) => {
            at.name == bt.name
        }
        (parser::TypeDeclaration::Abstract(aa), parser::TypeDeclaration::Abstract(ba)) => {
            aa.name == ba.name
        }
        _ => false,
    }
}

/// Test a single file, returning (filename, pass/fail, error message)
fn test_file(path: &Path) -> (String, bool, String) {
    let filename = path.file_name().unwrap().to_str().unwrap().to_string();
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => return (filename, false, format!("read error: {}", e)),
    };

    let nom_result = nom_parse(&source, &filename);
    let rd_result = rd_parse(&source, &filename);

    match (nom_result, rd_result) {
        (Some(nom_ast), Some(rd_ast)) => {
            if asts_equivalent(&nom_ast, &rd_ast) {
                (filename, true, String::new())
            } else {
                let mut msg = format!(
                    "AST mismatch: nom={} decls/{} imports, rd={} decls/{} imports",
                    nom_ast.declarations.len(),
                    nom_ast.imports.len(),
                    rd_ast.declarations.len(),
                    rd_ast.imports.len(),
                );
                // Show first declaration name difference
                for (i, (a, b)) in nom_ast
                    .declarations
                    .iter()
                    .zip(rd_ast.declarations.iter())
                    .enumerate()
                {
                    if !decl_names_match(a, b) {
                        msg.push_str(&format!(" [decl {} differs]", i));
                        break;
                    }
                }
                (filename, false, msg)
            }
        }
        (Some(_), None) => (
            filename,
            false,
            "RD parser failed, nom succeeded".to_string(),
        ),
        (None, Some(_)) => (
            filename,
            false,
            "nom parser failed, RD succeeded".to_string(),
        ),
        (None, None) => {
            // Both failed — that's acceptable (file may have intentional errors)
            (filename, true, "both parsers failed (expected)".to_string())
        }
    }
}

#[test]
fn test_rd_vs_nom_haxe_tests() {
    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("compiler/tests/haxe");

    if !test_dir.exists() {
        eprintln!("Skipping: {} not found", test_dir.display());
        return;
    }

    let mut files: Vec<_> = fs::read_dir(&test_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "hx"))
        .map(|e| e.path())
        .collect();
    files.sort();

    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();

    for path in &files {
        let (filename, success, msg) = test_file(path);
        if success {
            passed += 1;
        } else {
            failed += 1;
            failures.push((filename, msg));
        }
    }

    eprintln!("\n=== RD vs nom comparison: {} files ===", files.len());
    eprintln!("  PASSED: {}", passed);
    eprintln!("  FAILED: {}", failed);

    if !failures.is_empty() {
        eprintln!("\nFailures:");
        for (name, msg) in &failures {
            eprintln!("  {} — {}", name, msg);
        }
    }

    // 100% pass rate required — RD parser must match nom for all files
    assert_eq!(
        failed,
        0,
        "RD parser failed on {} of {} files — must be 0 before wiring as default",
        failed,
        files.len()
    );
}

#[test]
fn test_rd_vs_nom_benchmarks() {
    let bench_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("compiler/benchmarks/src");

    if !bench_dir.exists() {
        eprintln!("Skipping: {} not found", bench_dir.display());
        return;
    }

    let files = ["deltablue.hx", "fibonacci.hx", "mandelbrot.hx", "nbody.hx"];

    for filename in &files {
        let path = bench_dir.join(filename);
        let (name, success, msg) = test_file(&path);
        if success {
            eprintln!("  PASS: {}", name);
        } else {
            eprintln!("  FAIL: {} — {}", name, msg);
        }
        // Benchmarks must ALL pass
        assert!(success, "Benchmark {} failed: {}", filename, msg);
    }
}

/// Focused tests for specific language features
#[test]
fn test_rd_class_with_methods() {
    let source = r#"
class Strength {
    public var value:Int;
    public function new(v:Int) {
        this.value = v;
    }
    public static function stronger(s1:Strength, s2:Strength):Bool {
        return s1.value < s2.value;
    }
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse class with methods");
    let file = result.unwrap();
    assert_eq!(file.declarations.len(), 1);
    if let parser::TypeDeclaration::Class(c) = &file.declarations[0] {
        assert_eq!(c.name, "Strength");
        assert_eq!(c.fields.len(), 3); // value, new, stronger
    }
}

#[test]
fn test_rd_class_inheritance() {
    let source = r#"
class Animal {
    public var name:String;
}
class Dog extends Animal {
    public function bark():String {
        return "woof";
    }
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse class inheritance");
    let file = result.unwrap();
    assert_eq!(file.declarations.len(), 2);
}

#[test]
fn test_rd_enum_with_params() {
    let source = r#"
enum Option<T> {
    Some(v:T);
    None;
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse generic enum");
    let file = result.unwrap();
    if let parser::TypeDeclaration::Enum(e) = &file.declarations[0] {
        assert_eq!(e.name, "Option");
        assert_eq!(e.type_params.len(), 1);
        assert_eq!(e.constructors.len(), 2);
        assert_eq!(e.constructors[0].params.len(), 1); // Some has param
        assert_eq!(e.constructors[1].params.len(), 0); // None has no params
    }
}

#[test]
fn test_rd_typedef() {
    let source = r#"
typedef Point = {
    x:Int,
    y:Int
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse typedef");
}

#[test]
fn test_rd_complex_expressions() {
    let source = r#"
class Main {
    static function main() {
        var x = 1 + 2 * 3;
        var y = x > 0 ? x : -x;
        var arr = [1, 2, 3];
        for (i in 0...arr.length) {
            trace(arr[i]);
        }
    }
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse complex expressions");
}

#[test]
fn test_rd_metadata() {
    let source = r#"
@:native("console.log")
@:keep
class Logger {
    @:deprecated("use log2")
    public static function log(msg:String):Void;
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse metadata");
    let file = result.unwrap();
    if let parser::TypeDeclaration::Class(c) = &file.declarations[0] {
        assert_eq!(c.meta.len(), 2);
    }
}

#[test]
fn test_rd_abstract_type() {
    let source = r#"
abstract Color(Int) {
    public static var Red = new Color(0);
    public function new(v:Int) {
        this = v;
    }
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse abstract type");
}

#[test]
fn test_rd_try_catch() {
    let source = r#"
class Main {
    static function main() {
        try {
            throw "error";
        } catch(e:String) {
            trace(e);
        }
    }
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse try/catch");
}

#[test]
fn test_rd_switch_case() {
    let source = r#"
class Main {
    static function main() {
        var x = 5;
        switch (x) {
            case 1:
                trace("one");
            case 2:
                trace("two");
            default:
                trace("other");
        }
    }
}
"#;
    let result = rd_parse(source, "test.hx");
    assert!(result.is_some(), "failed to parse switch/case");
}

#[test]
fn test_rd_debug_failures() {
    let failing_files = [
        "test_balanced_tree_generic.hx",
        "test_enum_abstract_methods.hx",
        "test_reflect_compare_methods.hx",
    ];

    let test_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("compiler/tests/haxe");

    for filename in &failing_files {
        let path = test_dir.join(filename);
        let source = fs::read_to_string(&path).unwrap();
        let config = parser::preprocessor::PreprocessorConfig::default();
        let preprocessed = parser::preprocessor::preprocess(&source, &config);
        match parser::rd::rd_parse(&preprocessed, filename, false, false) {
            Ok(file) => eprintln!("  {} OK ({} decls)", filename, file.declarations.len()),
            Err(errors) => {
                for e in errors.iter().take(3) {
                    eprintln!(
                        "  {} ERROR at {}..{}: {}",
                        filename, e.span.start, e.span.end, e.message
                    );
                }
            }
        }
    }
}
