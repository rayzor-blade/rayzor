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
#![allow(clippy::single_match)]
/// Complete pipeline test: Haxe Source → AST → TAST → HIR → MIR → Cranelift → Native Execution
///
/// This test demonstrates the full compilation pipeline with proper SSA form from HIR,
/// solving the SSA limitation we discovered in manual MIR construction.
///
/// Pipeline stages:
/// 1. Parse Haxe source code
/// 2. Lower AST to TAST (Typed AST)
/// 3. Lower TAST to HIR (High-level IR with semantic info)
/// 4. Lower HIR to MIR (SSA form with phi nodes)
/// 5. Compile MIR to Cranelift IR
/// 6. JIT compile to native code
/// 7. Execute and verify results
use compiler::codegen::CraneliftBackend;
use compiler::ir::{hir_to_mir::lower_hir_to_mir, tast_to_hir::lower_tast_to_hir};
use compiler::tast::{
    ast_lowering::AstLowering, scopes::ScopeTree, StringInterner, SymbolTable, TypeTable,
};
use parser::haxe_parser::parse_haxe_file;
use rayzor_plugin::RuntimePlugin;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), String> {
    println!("=== Math Operations: Full Pipeline Test ===\n");
    println!("Pipeline: Haxe → AST → TAST → HIR → MIR → Cranelift → JIT → Native Execution\n");

    // Test Math operations
    test_math_abs()?;

    println!("\n🎉 All Math pipeline tests passed!");
    println!("{}", "=".repeat(70));
    Ok(())
}

fn test_math_abs() -> Result<(), String> {
    println!("TEST: Math.abs(-5.5)");
    println!("{}", "─".repeat(70));

    let source = r#"
package test;

class MathTest {
    public static function testAbs():Float {
        var x:Float = -5.5;
        return Math.abs(x);
    }
}
    "#;

    println!("Haxe Source:");
    println!("  var x:Float = -5.5;");
    println!("  return Math.abs(x);\n");

    println!("Compiling through full pipeline...");
    let mir_module = compile_haxe_to_mir(source)?;

    let test_func = mir_module
        .functions
        .values()
        .next()
        .ok_or("No function found in MIR")?;

    println!("  ✓ Parser → TAST → HIR → MIR");
    println!("  ✓ Function: {}", test_func.name);
    println!("  ✓ Blocks: {}\n", test_func.cfg.blocks.len());

    println!("JIT Compiling with Cranelift...");

    // Get runtime symbols from the plugin system
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = CraneliftBackend::with_symbols(&symbols_ref)?;
    backend.compile_module(&mir_module)?;
    println!("  ✓ MIR → Cranelift IR → Native\n");

    let func_ptr = backend.get_function_ptr(test_func.id)?;
    let test_fn: fn() -> f64 = unsafe { std::mem::transmute(func_ptr) };

    println!("Executing JIT-compiled native code...");
    let result = test_fn();

    println!("  Output: {}", result);
    println!("  Expected: 5.5");

    if (result - 5.5).abs() < 0.0001 {
        println!("  ✅ PASSED");
        Ok(())
    } else {
        Err(format!(
            "Math.abs test failed: expected 5.5, got {}",
            result
        ))
    }
}

fn test_simple_arithmetic() -> Result<(), String> {
    println!("--- Test 1: Simple Arithmetic ---");

    let source = r#"
package test;

class TestMath {
    public static function add(a:Int, b:Int):Int {
        return a + b;
    }
}
    "#;

    println!("Source: add(a, b) = a + b");

    // Compile through full pipeline
    let mir_module = compile_haxe_to_mir(source)?;

    // Debug: Print all function names
    println!("\nAvailable functions in MIR:");
    for func in mir_module.functions.values() {
        println!("  - {}", func.name);
    }

    // Get the first (and only) function - it should be 'add'
    // TODO: Function names from HIR are not properly resolved (showing as InternedString(N))
    // This is a known issue in HIR→MIR lowering - needs string interner access
    let add_func = mir_module
        .functions
        .values()
        .next()
        .ok_or("No functions in MIR module")?;

    println!("\nMIR Function: {}", add_func.name);
    println!("  Blocks: {}", add_func.cfg.blocks.len());
    println!(
        "  Instructions: {}",
        add_func
            .cfg
            .blocks
            .values()
            .map(|b| b.instructions.len())
            .sum::<usize>()
    );

    // Debug: Print block details
    for (block_id, block) in &add_func.cfg.blocks {
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
    let func_ptr = backend.get_function_ptr(add_func.id)?;
    let add_fn: fn(i64, i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test cases
    println!("\nExecuting JIT-compiled function:");
    let tests = vec![(10, 20, 30), (100, 200, 300), (-5, 15, 10), (0, 0, 0)];

    let mut all_passed = true;
    for (a, b, expected) in tests {
        let result = add_fn(a, b);
        let passed = result == expected;
        let symbol = if passed { "✓" } else { "✗" };
        println!(
            "  {} add({}, {}) = {} (expected {})",
            symbol, a, b, result, expected
        );
        all_passed &= passed;
    }

    if !all_passed {
        return Err("Simple arithmetic test failed".to_string());
    }

    println!("✓ Simple arithmetic test passed\n");
    Ok(())
}

fn test_conditional() -> Result<(), String> {
    println!("--- Test 2: Conditional (if/else) ---");

    let source = r#"
package test;

class TestMath {
    public static function max(a:Int, b:Int):Int {
        if (a > b) {
            return a;
        } else {
            return b;
        }
    }
}
    "#;

    println!("Source: max(a, b) = if (a > b) then a else b");

    // Compile through full pipeline
    let mir_module = compile_haxe_to_mir(source)?;

    // Get the first (and only) function - it should be 'max'
    let max_func = mir_module
        .functions
        .values()
        .next()
        .ok_or("No functions in MIR module")?;

    println!("\nMIR Function: {}", max_func.name);
    println!(
        "  Blocks: {} (should have 3: entry, then, else)",
        max_func.cfg.blocks.len()
    );

    // Debug: Print all blocks
    for (block_id, block) in &max_func.cfg.blocks {
        println!(
            "  Block {:?}: {} instructions, terminator: {:?}",
            block_id,
            block.instructions.len(),
            std::mem::discriminant(&block.terminator)
        );
    }

    // Compile with Cranelift
    println!("\nCompiling MIR → Cranelift IR → Native...");
    let mut backend = CraneliftBackend::new()?;
    backend.compile_module(&mir_module)?;
    println!("✓ Compilation successful");

    // Get function pointer
    let func_ptr = backend.get_function_ptr(max_func.id)?;
    let max_fn: fn(i64, i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test cases
    println!("\nExecuting JIT-compiled function:");
    let tests = vec![
        (10, 5, 10),
        (5, 10, 10),
        (42, 42, 42),
        (-10, -20, -10),
        (100, 99, 100),
    ];

    let mut all_passed = true;
    for (a, b, expected) in tests {
        let result = max_fn(a, b);
        let passed = result == expected;
        let symbol = if passed { "✓" } else { "✗" };
        println!(
            "  {} max({}, {}) = {} (expected {})",
            symbol, a, b, result, expected
        );
        all_passed &= passed;
    }

    if !all_passed {
        return Err("Conditional test failed".to_string());
    }

    println!("✓ Conditional test passed\n");
    Ok(())
}

#[allow(dead_code)]
fn test_loop_ssa() -> Result<(), String> {
    println!("--- Test 3: Loop with SSA (while) ---");

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

    println!("Source: sumToN(n) = 1 + 2 + 3 + ... + n");

    // Compile through full pipeline
    let mir_module = compile_haxe_to_mir(source)?;

    // Find the 'sumToN' function
    let sum_func = mir_module
        .functions
        .values()
        .find(|f| f.name == "sumToN")
        .ok_or("Could not find 'sumToN' function")?;

    println!("\nMIR Function: {}", sum_func.name);
    println!(
        "  Blocks: {} (should have 4: entry, cond, body, exit)",
        sum_func.cfg.blocks.len()
    );

    // Verify SSA form: Check for phi nodes in loop header
    let has_phi_nodes = sum_func
        .cfg
        .blocks
        .values()
        .any(|block| !block.phi_nodes.is_empty());

    if has_phi_nodes {
        println!("  ✓ SSA form verified: phi nodes present for loop variables");
    } else {
        println!("  ⚠ Warning: No phi nodes found (may not be proper SSA)");
    }

    // Compile with Cranelift
    println!("\nCompiling MIR → Cranelift IR → Native...");
    let mut backend = CraneliftBackend::new()?;
    backend.compile_module(&mir_module)?;
    println!("✓ Compilation successful");

    // Get function pointer
    let func_ptr = backend.get_function_ptr(sum_func.id)?;
    let sum_fn: fn(i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test cases: sum_to_n(n) = n*(n+1)/2
    println!("\nExecuting JIT-compiled function:");
    let tests = vec![
        (0, 0),      // sum_to_n(0) = 0
        (1, 1),      // sum_to_n(1) = 1
        (5, 15),     // sum_to_n(5) = 1+2+3+4+5 = 15
        (10, 55),    // sum_to_n(10) = 55
        (100, 5050), // sum_to_n(100) = 5050
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
        return Err("Loop SSA test failed".to_string());
    }

    println!("✓ Loop SSA test passed (proper SSA from HIR!)");
    println!("  This validates that HIR→MIR produces correct SSA with phi nodes\n");

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

    // Debug: Print TAST functions and classes
    for func in &typed_file.functions {
        let func_name = string_interner.get(func.name).unwrap_or("?");
        eprintln!("  TAST function: {}", func_name);
    }
    for class in &typed_file.classes {
        let class_name = string_interner.get(class.name).unwrap_or("?");
        eprintln!(
            "  TAST class: {} with {} methods",
            class_name,
            class.methods.len()
        );
        for method in &class.methods {
            let method_name = string_interner.get(method.name).unwrap_or("?");
            eprintln!(
                "    Method: {} (interned ID: {:?})",
                method_name, method.name
            );
        }
    }

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

    // Debug: Print HIR functions and class methods
    for (symbol_id, func) in &hir_module.functions {
        eprintln!(
            "  HIR function: {:?} (symbol {:?}), has_body: {}",
            func.name,
            symbol_id,
            func.body.is_some()
        );
    }

    for (type_id, type_decl) in &hir_module.types {
        match type_decl {
            compiler::ir::hir::HirTypeDecl::Class(class) => {
                eprintln!(
                    "  HIR class: {:?} with {} methods",
                    class.name,
                    class.methods.len()
                );
                for method in &class.methods {
                    eprintln!(
                        "    Method: {:?}, has_body: {}",
                        method.function.name,
                        method.function.body.is_some()
                    );
                }
            }
            _ => {}
        }
    }

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

    for func in mir_module.functions.values() {
        eprintln!("  MIR function: {}", func.name);
    }

    Ok(mir_module)
}
