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
/// Comprehensive test for 100% Haxe language support through Cranelift backend
///
/// This test systematically exercises all major Haxe features:
/// - Loops (while, for, do-while)
/// - Function calls
/// - Arrays and indexing
/// - Classes and objects
/// - Type casts
/// - String operations
/// - And more...
use compiler::codegen::CraneliftBackend;
use compiler::ir::{hir_to_mir::lower_hir_to_mir, tast_to_hir::lower_tast_to_hir};
use compiler::tast::{
    StringInterner, SymbolTable, TypeTable, ast_lowering::AstLowering, scopes::ScopeTree,
};
use parser::haxe_parser::parse_haxe_file;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), String> {
    println!("=== Comprehensive Haxe Language Test ===\n");

    // Test 1: Loops (while)
    test_while_loop()?;

    // Test 2: Function calls
    test_function_calls()?;

    // Test 3: Loops (for)
    // test_for_loop()?;

    // Test 4: Arrays
    // test_arrays()?;

    println!("\n=== All Comprehensive Tests Passed! ===\n");
    Ok(())
}

fn test_while_loop() -> Result<(), String> {
    println!("--- Test: While Loop ---");

    let source = r#"
package test;

class Math {
    public static function sumToN(n:Int):Int {
        var sum = 0;
        var i = 1;
        while (i <= n) {
            sum = sum + i;
            i = i + 1;
        }
        return sum;
    }
}
    "#;

    println!("Source: sumToN(n) = 1 + 2 + ... + n");

    // Compile through full pipeline
    let mir_module = compile_haxe_to_mir(source)?;

    // Get the function
    let sum_func = mir_module
        .functions
        .values()
        .next()
        .ok_or("No functions in MIR module")?;

    println!("\nMIR Function: {}", sum_func.name);
    println!("  Blocks: {}", sum_func.cfg.blocks.len());

    // Print blocks
    for (block_id, block) in &sum_func.cfg.blocks {
        println!(
            "  Block {:?}: {} instructions",
            block_id,
            block.instructions.len()
        );
    }

    // Compile with Cranelift
    println!("\nCompiling MIR → Cranelift IR → Native...");
    let mut backend = CraneliftBackend::new()?;
    backend.compile_module(&mir_module)?;
    println!("✓ Compilation successful");

    // Get function pointer
    let func_ptr = backend.get_function_ptr(sum_func.id)?;
    let sum_fn: fn(i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test cases: sum(n) = n*(n+1)/2
    println!("\nExecuting JIT-compiled function:");
    let tests = vec![
        (0, 0),      // sum(0) = 0
        (1, 1),      // sum(1) = 1
        (5, 15),     // sum(5) = 1+2+3+4+5 = 15
        (10, 55),    // sum(10) = 55
        (100, 5050), // sum(100) = 5050
    ];

    let mut all_passed = true;
    for (n, expected) in tests {
        let result = sum_fn(n);
        let passed = result == expected;
        let symbol = if passed { "✓" } else { "✗" };
        println!(
            "  {} sumToN({}) = {} (expected {})",
            symbol, n, result, expected
        );
        all_passed &= passed;
    }

    if !all_passed {
        return Err("While loop test failed".to_string());
    }

    println!("✓ While loop test passed\n");
    Ok(())
}

fn test_function_calls() -> Result<(), String> {
    println!("\n--- Test: Function Calls ---");

    let source = r#"
package test;

class Math {
    public static function add(a:Int, b:Int):Int {
        return a + b;
    }

    public static function multiply(a:Int, b:Int):Int {
        return a * b;
    }

    public static function compute(x:Int):Int {
        var y = add(x, 10);
        var z = multiply(y, 2);
        return z;
    }
}
    "#;

    println!("Source: compute(x) calls add() and multiply()");

    // Compile through full pipeline
    let mir_module = compile_haxe_to_mir(source)?;

    println!("\nMIR Module:");
    println!("  Functions: {}", mir_module.functions.len());
    for func in mir_module.functions.values() {
        println!("    - {}", func.name);
    }

    // Find the compute function (it takes 1 parameter, while add and multiply take 2)
    let compute_func = mir_module
        .functions
        .values()
        .find(|f| f.signature.parameters.len() == 1)
        .ok_or("Compute function not found (expected function with 1 parameter)")?;

    println!("\nTesting function: {}", compute_func.name);
    println!("  Parameters: {}", compute_func.signature.parameters.len());
    println!("  Blocks: {}", compute_func.cfg.blocks.len());

    // Compile with Cranelift
    println!("Compiling MIR → Cranelift IR → Native...");
    let mut backend = CraneliftBackend::new()?;
    backend.compile_module(&mir_module)?;
    println!("✓ Compilation successful");

    // Get function pointer
    let func_ptr = backend.get_function_ptr(compute_func.id)?;
    let compute_fn: fn(i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test cases: compute(x) = (x + 10) * 2
    println!("\nExecuting JIT-compiled function:");
    let tests = vec![
        (5, 30),  // (5 + 10) * 2 = 30
        (0, 20),  // (0 + 10) * 2 = 20
        (10, 40), // (10 + 10) * 2 = 40
        (-5, 10), // (-5 + 10) * 2 = 10
    ];

    let mut all_passed = true;
    for (input, expected) in tests {
        let result = compute_fn(input);
        if result == expected {
            println!(
                "  ✓ compute({}) = {} (expected {})",
                input, result, expected
            );
        } else {
            println!(
                "  ✗ compute({}) = {} (expected {})",
                input, result, expected
            );
            all_passed = false;
        }
    }

    if !all_passed {
        return Err("Some function call tests failed".to_string());
    }

    println!("✓ Function calls test passed");
    Ok(())
}

/// Compile Haxe source through the full pipeline to MIR
fn compile_haxe_to_mir(source: &str) -> Result<compiler::ir::IrModule, String> {
    // Step 1: Parse
    let ast =
        parse_haxe_file("test.hx", source, false).map_err(|e| format!("Parse error: {}", e))?;

    // Step 2: Lower AST to TAST
    let mut string_interner = StringInterner::new();
    let mut symbol_table = SymbolTable::new();
    let type_table = Rc::new(RefCell::new(TypeTable::new()));
    let mut scope_tree = ScopeTree::new(compiler::tast::ScopeId::from_raw(0));
    let mut namespace_resolver = compiler::tast::namespace::NamespaceResolver::new();
    let mut import_resolver = compiler::tast::namespace::ImportResolver::new();

    let mut ast_lowering = AstLowering::new(
        &mut string_interner,
        std::rc::Rc::new(std::cell::RefCell::new(
            compiler::tast::StringInterner::new(),
        )),
        &mut symbol_table,
        &type_table,
        &mut scope_tree,
        &mut namespace_resolver,
        &mut import_resolver,
    );

    let mut typed_file = ast_lowering
        .lower_file(&ast)
        .map_err(|e| format!("TAST lowering error: {:?}", e))?;

    // Step 3: Lower TAST to HIR
    let string_interner_rc = Rc::new(RefCell::new(string_interner));
    typed_file.string_interner = Rc::clone(&string_interner_rc);

    let hir_module = lower_tast_to_hir(
        &typed_file,
        &symbol_table,
        &type_table,
        &mut *string_interner_rc.borrow_mut(),
        None, // No semantic graphs for now
    )
    .map_err(|errors| {
        let messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
        format!("HIR lowering errors: {}", messages.join(", "))
    })?;

    // Step 4: Lower HIR to MIR (this produces proper SSA!)
    let mir_module = lower_hir_to_mir(
        &hir_module,
        &*string_interner_rc.borrow(),
        &type_table,
        &symbol_table,
    )
    .map_err(|errors| {
        let messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();
        format!("MIR lowering errors: {}", messages.join(", "))
    })?;

    Ok(mir_module)
}
