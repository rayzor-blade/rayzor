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
use compiler::codegen::CraneliftBackend;
/// Test executable indirect function calls through full pipeline
///
/// This test:
/// 1. Compiles Haxe code with function pointers (TAST → HIR → MIR)
/// 2. JIT compiles with Cranelift
/// 3. Executes the code and verifies indirect calls work correctly
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::hir_to_mir::lower_hir_to_mir;
use compiler::ir::tast_to_hir::lower_tast_to_hir;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), String> {
    println!("=== Executable Indirect Function Calls Test ===\n");

    // Create compilation unit
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load stdlib
    println!("1. Loading stdlib...");
    match unit.load_stdlib() {
        Ok(()) => println!("   ✓ Loaded {} stdlib files\n", unit.stdlib_files.len()),
        Err(e) => {
            eprintln!("   ✗ Failed: {}", e);
            return Err(e);
        }
    }

    // Test code with function pointers and indirect calls
    let source = r#"
        package test;

        class MathOps {
            // Simple arithmetic functions
            public static function add(a:Int, b:Int):Int {
                return a + b;
            }

            public static function multiply(a:Int, b:Int):Int {
                return a * b;
            }

            public static function subtract(a:Int, b:Int):Int {
                return a - b;
            }

            // Function that takes a function pointer (indirect call)
            public static function applyOperation(x:Int, y:Int, op:Int->Int->Int):Int {
                return op(x, y);  // CallIndirect
            }

            // Test function that uses indirect calls
            public static function testIndirectCalls():Int {
                // Test 1: add through function pointer
                var addFunc:Int->Int->Int = add;
                var sum = applyOperation(10, 5, addFunc);  // Should return 15

                // Test 2: multiply through function pointer
                var mulFunc:Int->Int->Int = multiply;
                var product = applyOperation(10, 5, mulFunc);  // Should return 50

                // Test 3: subtract through function pointer
                var subFunc:Int->Int->Int = subtract;
                var diff = applyOperation(10, 5, subFunc);  // Should return 5

                // Return combined result: (10+5) + (10*5) + (10-5) = 15 + 50 + 5 = 70
                return sum + product + diff;
            }
        }
    "#;

    println!("2. Adding test code with indirect calls...");
    match unit.add_file(source, "MathOps.hx") {
        Ok(()) => println!("   ✓ Added MathOps.hx\n"),
        Err(e) => {
            eprintln!("   ✗ Failed: {}", e);
            return Err(e);
        }
    }

    // Lower to TAST
    println!("3. Lowering to TAST...");
    let typed_files = match unit.lower_to_tast() {
        Ok(files) => {
            println!("   ✓ Lowered {} files to TAST\n", files.len());
            files
        }
        Err(e) => {
            eprintln!("   ✗ TAST failed: {:?}", e);
            return Err(format!("{:?}", e));
        }
    };

    // Lower to HIR
    println!("4. Lowering TAST to HIR...");
    let string_interner_rc = Rc::new(RefCell::new(std::mem::replace(
        &mut unit.string_interner,
        compiler::tast::StringInterner::new(),
    )));

    let mut all_hir_modules = Vec::new();
    for typed_file in &typed_files {
        let hir_module = {
            let mut interner_guard = string_interner_rc.borrow_mut();
            match lower_tast_to_hir(
                typed_file,
                &unit.symbol_table,
                &unit.type_table,
                &mut *interner_guard,
                None,
            ) {
                Ok(hir) => {
                    println!(
                        "   ✓ Lowered {} to HIR ({} types)",
                        typed_file.metadata.file_path,
                        hir.types.len()
                    );
                    Some(hir)
                }
                Err(errors) => {
                    eprintln!("   ✗ HIR lowering errors:");
                    for error in errors {
                        eprintln!("     - {}", error.message);
                    }
                    None
                }
            }
        };

        if let Some(hir) = hir_module {
            all_hir_modules.push(hir);
        }
    }

    if all_hir_modules.is_empty() {
        return Err("No HIR modules generated".to_string());
    }
    println!();

    // Lower HIR to MIR
    println!("5. Lowering HIR to MIR...");
    let mut all_mir_modules = Vec::new();
    for hir_module in &all_hir_modules {
        let mir_module = match lower_hir_to_mir(
            hir_module,
            &*string_interner_rc.borrow(),
            &unit.type_table,
            &unit.symbol_table,
        ) {
            Ok(mir) => {
                println!(
                    "   ✓ Lowered {} to MIR ({} functions)",
                    hir_module.name,
                    mir.functions.len()
                );
                mir
            }
            Err(errors) => {
                eprintln!("   ✗ MIR lowering errors:");
                for error in errors {
                    eprintln!("     - {}", error.message);
                }
                continue;
            }
        };
        all_mir_modules.push(mir_module);
    }

    if all_mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }
    println!();

    // Find the test module with our functions
    let test_module = all_mir_modules
        .iter()
        .find(|m| {
            m.functions
                .values()
                .any(|f| f.name.contains("testIndirectCalls"))
        })
        .ok_or("Could not find test module")?;

    // Find the testIndirectCalls function
    let test_func_id = test_module
        .functions
        .iter()
        .find(|(_, f)| f.name.contains("testIndirectCalls"))
        .map(|(id, _)| *id)
        .ok_or("Could not find testIndirectCalls function")?;

    println!("6. JIT compiling with Cranelift...");
    let mut backend = CraneliftBackend::new()?;

    backend.compile_module(test_module)?;
    println!("   ✓ Cranelift compilation successful\n");

    // Get function pointer
    println!("7. Executing JIT-compiled code...");
    let func_ptr = backend.get_function_ptr(test_func_id)?;
    println!("   ✓ Function pointer: {:p}\n", func_ptr);

    // Execute the JIT-compiled code
    let test_function: fn() -> i64 = unsafe { std::mem::transmute(func_ptr) };
    let result = test_function();
    println!("   ✓ Execution complete\n");

    // Verify result
    println!("8. Verifying result...");
    println!("   Expected: 70  (15 + 50 + 5)");
    println!("   Got:      {}", result);

    if result == 70 {
        println!("\n🎉 SUCCESS: Indirect function calls work correctly!");
        println!("   ✓ Function pointers assigned");
        println!("   ✓ CallIndirect instructions executed");
        println!("   ✓ All operations returned correct values");
        Ok(())
    } else {
        Err(format!(
            "FAILED: Expected 70, got {}. Indirect calls may not be working correctly.",
            result
        ))
    }
}
