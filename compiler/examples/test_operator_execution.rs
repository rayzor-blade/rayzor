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
//! Test that operator overloading actually executes correctly at runtime

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::hir_to_mir::lower_hir_to_mir;
use compiler::ir::tast_to_hir::lower_tast_to_hir;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), String> {
    println!("\n=== Testing Operator Overloading Execution ===\n");

    // Use a simpler example without constructor calls
    let source = r#"
        package test;

        abstract Counter(Int) from Int to Int {
            @:op(A + B)
            public inline function add(rhs:Counter):Int {
                // Simple addition that just returns the sum as Int
                return this + rhs;
            }
        }

        class Main {
            public static function main() {
                var a:Counter = 5;
                var b:Counter = 10;
                return a + b;  // Should call add() which inlines to: this + rhs = 5 + 10 = 15
            }
        }
    "#;

    let mut unit = CompilationUnit::new(CompilationConfig {
        load_stdlib: false,
        ..Default::default()
    });

    unit.add_file(source, "test.hx")?;

    // Lower to TAST
    let typed_files = unit.lower_to_tast().map_err(|e| format!("{:?}", e))?;
    println!("✓ TAST generated\n");

    // Lower to HIR → MIR
    let string_interner_rc = Rc::new(RefCell::new(std::mem::replace(
        &mut unit.string_interner,
        compiler::tast::StringInterner::new(),
    )));

    let mut mir_modules = Vec::new();
    for typed_file in &typed_files {
        let hir = {
            let mut interner = string_interner_rc.borrow_mut();
            lower_tast_to_hir(
                typed_file,
                &unit.symbol_table,
                &unit.type_table,
                &mut *interner,
                None,
            )
            .map_err(|e| format!("HIR error: {:?}", e))?
        };

        let mir = lower_hir_to_mir(
            &hir,
            &*string_interner_rc.borrow(),
            &unit.type_table,
            &unit.symbol_table,
        )
        .map_err(|e| format!("MIR error: {:?}", e))?;
        if !mir.functions.is_empty() {
            mir_modules.push(mir);
        }
    }
    println!("✓ HIR and MIR generated\n");

    // Find main function
    let test_module = mir_modules
        .into_iter()
        .find(|m| m.functions.iter().any(|(_, f)| f.name.contains("main")))
        .ok_or("No module with main function found")?;

    let (main_func_id, _main_func) = test_module
        .functions
        .iter()
        .find(|(_, f)| f.name.contains("main"))
        .ok_or("Main function not found")?;

    // Compile with Cranelift
    let mut backend =
        CraneliftBackend::new().map_err(|e| format!("Failed to create backend: {}", e))?;

    backend.compile_module(&test_module)?;
    println!("✓ Cranelift compilation complete\n");

    // Execute
    let func_ptr = backend.get_function_ptr(*main_func_id)?;
    let main_fn: fn() -> i64 = unsafe { std::mem::transmute(func_ptr) };
    let result = main_fn();

    println!("Result: {}", result);

    if result == 15 {
        println!("\n✅ TEST PASSED: Operator overloading works correctly at runtime!");
        println!("   a + b = 5 + 10 = 15 ✓\n");
        Ok(())
    } else {
        println!("\n❌ FAILED: Expected 15, got {}\n", result);
        Err(format!(
            "Operator overloading returned wrong value: {}",
            result
        ))
    }
}
