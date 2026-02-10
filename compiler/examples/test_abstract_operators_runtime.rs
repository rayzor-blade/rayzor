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
//! Test abstract type operator overloading at runtime with Cranelift
//!
//! This tests whether operator overloading (@:op) actually works at runtime

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::hir_to_mir::lower_hir_to_mir;
use compiler::ir::tast_to_hir::lower_tast_to_hir;
use std::cell::RefCell;
use std::rc::Rc;

fn main() -> Result<(), String> {
    println!("\n=== Testing Abstract Type Operators at Runtime ===\n");

    // Test 1: Basic abstract without operators
    test_basic_abstract()?;

    // Test 2: Abstract with operator overloading
    test_operator_addition()?;

    Ok(())
}

fn test_basic_abstract() -> Result<(), String> {
    println!("Test 1: Basic Abstract Type (without operators)\n");

    let source = r#"
        abstract Counter(Int) from Int to Int {
            public inline function new(value:Int) {
                this = value;
            }

            public inline function add(other:Counter):Counter {
                return new Counter(this + other.toInt());
            }

            public inline function toInt():Int {
                return this;
            }
        }

        class Main {
            public static function main() {
                var a:Counter = 5;
                var b:Counter = 10;
                var sum = a.add(b);
                trace(sum);  // Should print 15
                trace("done");
            }
        }
    "#;

    compile_and_run(source)?;
    println!("  ✓ TEST PASSED: Basic abstract type works!\n");
    Ok(())
}

fn test_operator_addition() -> Result<(), String> {
    println!("Test 2: Abstract with Operator Overloading (@:op)\n");

    let source = r#"
        abstract Counter(Int) from Int to Int {
            public inline function new(value:Int) {
                this = value;
            }

            @:op(A + B)
            public inline function add(rhs:Counter):Counter {
                return new Counter(this + rhs.toInt());
            }

            public inline function toInt():Int {
                return this;
            }
        }

        class Main {
            public static function main() {
                var a:Counter = 5;
                var b:Counter = 10;
                var sum = a + b;  // Uses @:op(A + B)
                trace(sum);       // Should print 15
                trace("done");
            }
        }
    "#;

    match compile_and_run(source) {
        Ok(()) => {
            println!("  ✓ TEST PASSED: Operator overloading works at runtime!\n");
            Ok(())
        }
        Err(e) => {
            println!("  ❌ FAILED: Operator overloading not working");
            println!("  Error: {}\n", e);
            Err(e)
        }
    }
}

fn compile_and_run(source: &str) -> Result<(), String> {
    // Step 1: Create compilation unit with stdlib for trace() support
    let mut unit = CompilationUnit::new(CompilationConfig::fast());

    unit.add_file(source, "test.hx")?;

    // Step 2: Lower to TAST
    let typed_files = unit.lower_to_tast().map_err(|e| format!("{:?}", e))?;
    println!("  ✓ TAST generated ({} files)", typed_files.len());

    // Step 3: Lower to HIR → MIR
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
    println!("  ✓ HIR and MIR generated");

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

    // Step 4: Compile with Cranelift (with runtime symbols for trace/string support)
    let plugin = rayzor_runtime::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = CraneliftBackend::with_symbols(&symbols_ref)
        .map_err(|e| format!("Failed to create backend: {}", e))?;

    backend.compile_module(&test_module)?;
    println!("  ✓ Cranelift compilation complete");

    // Step 5: Execute (main returns void)
    let func_ptr = backend.get_function_ptr(*main_func_id)?;
    let main_fn: fn() = unsafe { std::mem::transmute(func_ptr) };
    main_fn();
    println!("  ✓ Execution successful");

    Ok(())
}
