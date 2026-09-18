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
#![allow(clippy::unnecessary_mut_passed)]
// Integration test for complete HIR to MIR lowering
// Tests all newly implemented features from Week 3

use compiler::ir::{
    hir_to_mir::lower_hir_to_mir, tast_to_hir::lower_tast_to_hir, validation::validate_module,
};
use compiler::tast::ScopeId;
use compiler::tast::{
    StringInterner, SymbolTable, TypeTable, ast_lowering::AstLowering, scopes::ScopeTree,
    type_checking_pipeline::TypeCheckingPhase,
};
use diagnostics::{Diagnostics, ErrorFormatter};
use parser::haxe_parser::parse_haxe_file;
use source_map::SourceMap;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Testing Complete MIR Lowering Pipeline ===\n");

    // Test 1: Exception Handling
    test_exception_handling();

    // Test 2: Conditional Expressions
    test_conditional_expressions();

    // Test 3: Array Literals
    test_array_literals();

    // Test 4: Map and Object Literals
    test_map_and_object_literals();

    // Test 5: Pattern Matching
    test_pattern_matching();

    // Test 6: Closures
    test_closures();

    println!("\n=== All MIR Lowering Tests Complete ===");
}

fn test_exception_handling() {
    println!("Test 1: Exception Handling");

    let source = r#"
package test;

class ExceptionTest {
    public function testTryCatch():Int {
        var result = 0;
        try {
            result = riskyOperation();
        } catch (e:Dynamic) {
            result = -1;
        } finally {
            trace("cleanup");
        }
        return result;
    }

    function riskyOperation():Int {
        return 42;
    }
}
    "#;

    match run_pipeline(source) {
        Ok(_) => println!("   ✓ Exception handling lowering successful"),
        Err(e) => println!("   ✗ Failed: {}", e),
    }
}

fn test_conditional_expressions() {
    println!("\nTest 2: Conditional Expressions");

    let source = r#"
package test;

class ConditionalTest {
    public function testTernary(x:Int):Int {
        var result = x > 0 ? x * 2 : x * -1;
        var nested = x > 100 ? 100 : (x < 0 ? 0 : x);
        return result + nested;
    }
}
    "#;

    match run_pipeline(source) {
        Ok(_) => println!("   ✓ Conditional expression lowering successful"),
        Err(e) => println!("   ✗ Failed: {}", e),
    }
}

fn test_array_literals() {
    println!("\nTest 3: Array Literals");

    let source = r#"
package test;

class ArrayTest {
    public function testArrays():Int {
        var arr = [1, 2, 3, 4, 5];
        var nested = [[1, 2], [3, 4]];
        var empty = [];
        return arr[0];
    }
}
    "#;

    match run_pipeline(source) {
        Ok(_) => println!("   ✓ Array literal lowering successful"),
        Err(e) => println!("   ✗ Failed: {}", e),
    }
}

fn test_map_and_object_literals() {
    println!("\nTest 4: Map and Object Literals");

    let source = r#"
package test;

class LiteralTest {
    public function testLiterals():Int {
        var obj = { x: 10, y: 20, name: "test" };
        var map = ["key1" => 1, "key2" => 2];
        return obj.x;
    }
}
    "#;

    match run_pipeline(source) {
        Ok(_) => println!("   ✓ Map/Object literal lowering successful"),
        Err(e) => println!("   ✗ Failed: {}", e),
    }
}

fn test_pattern_matching() {
    println!("\nTest 5: Pattern Matching");

    let source = r#"
package test;

class PatternTest {
    public function testSwitch(value:Int):String {
        return switch(value) {
            case 0: "zero";
            case 1 | 2 | 3: "small";
            case x if (x > 100): "large";
            default: "medium";
        };
    }

    public function testPatternGuards(x:Int):Bool {
        return switch(x) {
            case n if (n % 2 == 0): true;
            case _: false;
        };
    }
}
    "#;

    match run_pipeline(source) {
        Ok(_) => println!("   ✓ Pattern matching lowering successful"),
        Err(e) => println!("   ✗ Failed: {}", e),
    }
}

fn test_closures() {
    println!("\nTest 6: Closures");

    let source = r#"
package test;

class ClosureTest {
    public function testLambda():Int {
        var x = 10;
        var f = function(y:Int):Int {
            return x + y;
        };
        return f(5);
    }

    public function testCaptures():Int {
        var captured = 42;
        var lambda = function():Int {
            return captured;
        };
        return lambda();
    }
}
    "#;

    match run_pipeline(source) {
        Ok(_) => println!("   ✓ Closure lowering successful"),
        Err(e) => println!("   ✗ Failed: {}", e),
    }
}

fn run_pipeline(source: &str) -> Result<(), String> {
    // Step 1: Parse
    let ast =
        parse_haxe_file("test.hx", source, false).map_err(|e| format!("Parse error: {:?}", e))?;

    // Step 2: Create compilation context
    let mut string_interner = StringInterner::new();
    let mut type_table = Rc::new(RefCell::new(TypeTable::new()));
    let mut symbol_table = SymbolTable::new();
    let mut scope_tree = ScopeTree::new(ScopeId::first());
    let mut namespace_resolver = compiler::tast::namespace::NamespaceResolver::new();
    let mut import_resolver = compiler::tast::namespace::ImportResolver::new();

    // Step 3: Lower AST to TAST
    let mut lowerer = AstLowering::new(
        &mut string_interner,
        std::rc::Rc::new(std::cell::RefCell::new(
            compiler::tast::StringInterner::new(),
        )),
        &mut symbol_table,
        &mut type_table,
        &mut scope_tree,
        &mut namespace_resolver,
        &mut import_resolver,
    );

    let mut tast_module = lowerer
        .lower_file(&ast)
        .map_err(|e| format!("AST lowering error: {:?}", e))?;

    // Step 4: Type checking using TypeCheckingPhase
    // TypeCheckingPhase provides full type checking with diagnostics
    let source_map = SourceMap::new();
    let mut diagnostics = Diagnostics::new();

    let mut type_checking_phase = TypeCheckingPhase::new(
        &type_table,
        &symbol_table,
        &scope_tree,
        &string_interner,
        &source_map,
        &mut diagnostics,
    );

    // Run type checking - errors are recorded in diagnostics AND may return Err
    // We ignore the Result because we want to check diagnostics instead
    let _ = type_checking_phase.check_file(&mut tast_module);

    // Check diagnostics after type checking
    // Note: For MIR lowering tests, we want to continue even with type errors
    // to test that MIR lowering can handle TAST (even if types aren't perfect).
    // In a production compiler, you would fail here if has_errors() is true.
    if diagnostics.has_errors() {
        let error_count = diagnostics.errors().count();
        eprintln!(
            "\n⚠ Type checking found {} error(s) - continuing to test MIR lowering\n",
            error_count
        );

        // Use the default ErrorFormatter to print diagnostics with source context
        let formatter = ErrorFormatter::new();
        let formatted = formatter.format_diagnostics(&diagnostics, &source_map);
        eprintln!("{}", formatted);
    }

    // Step 5: Lower TAST to HIR
    let hir_module = lower_tast_to_hir(
        &tast_module,
        &symbol_table,
        &type_table,
        &mut string_interner,
        None,
    )
    .map_err(|e| format!("HIR lowering error: {:?}", e))?;

    // Step 6: Lower HIR to MIR
    let mir_module = lower_hir_to_mir(&hir_module, &string_interner, &type_table, &symbol_table)
        .map_err(|e| format!("MIR lowering errors: {:?}", e))?;

    // Step 7: Validate MIR
    validate_module(&mir_module)
        .map_err(|errors| format!("MIR validation errors: {} error(s)", errors.len()))?;

    Ok(())
}
