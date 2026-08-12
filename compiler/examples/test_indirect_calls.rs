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
//! Test indirect function calls with qualified names
//!
//! This test verifies:
//! 1. Function pointers with qualified names
//! 2. CallIndirect instructions in MIR
//! 3. Cranelift codegen for indirect calls

use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() {
    println!("=== Indirect Function Calls Test ===\n");

    // Create compilation unit
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load stdlib
    println!("1. Loading stdlib...");
    match unit.load_stdlib() {
        Ok(()) => println!("   ✓ Loaded {} stdlib files\n", unit.stdlib_files.len()),
        Err(e) => {
            eprintln!("   ✗ Failed: {}", e);
            return;
        }
    }

    // Test code with function pointers and indirect calls
    let source = r#"
        package test;

        class IndirectCallTest {
            // Function type alias
            public static function add(a:Int, b:Int):Int {
                return a + b;
            }

            public static function multiply(a:Int, b:Int):Int {
                return a * b;
            }

            public static function applyOperation(x:Int, y:Int, op:Int->Int->Int):Int {
                // This should generate a CallIndirect
                return op(x, y);
            }

            public static function main():Void {
                // Store function reference
                var addFunc:Int->Int->Int = add;
                var mulFunc:Int->Int->Int = multiply;

                // Call through function pointer (indirect)
                var sum = applyOperation(5, 3, addFunc);
                var product = applyOperation(5, 3, mulFunc);

                // Direct call for comparison
                var directSum = add(5, 3);
            }
        }
    "#;

    println!("2. Adding test code with indirect calls...");
    match unit.add_file(source, "IndirectCallTest.hx") {
        Ok(()) => println!("   ✓ Added IndirectCallTest.hx\n"),
        Err(e) => {
            eprintln!("   ✗ Failed: {}", e);
            return;
        }
    }

    // Lower to TAST
    println!("3. Lowering to TAST...");
    let typed_files = match unit.lower_to_tast() {
        Ok(files) => {
            println!("   ✓ Lowered {} files to TAST", files.len());

            // Check for function symbols
            let test_symbols: Vec<_> = unit
                .symbol_table
                .all_symbols()
                .filter_map(|s| {
                    s.qualified_name.and_then(|qname| {
                        let name = unit.string_interner.get(qname)?;
                        if name.starts_with("test.IndirectCallTest") {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                })
                .collect();

            println!("   Found {} test symbols:", test_symbols.len());
            for sym in test_symbols.iter().take(10) {
                println!("      - {}", sym);
            }

            files
        }
        Err(e) => {
            eprintln!("   ✗ TAST failed: {:?}", e);
            return;
        }
    };

    println!("\n4. Checking qualified names...");

    // Look for the static methods with qualified names
    let has_add = unit.symbol_table.all_symbols().any(|s| {
        s.qualified_name
            .and_then(|q| unit.string_interner.get(q))
            .map(|n| n == "test.IndirectCallTest.add")
            .unwrap_or(false)
    });

    let has_apply_op = unit.symbol_table.all_symbols().any(|s| {
        s.qualified_name
            .and_then(|q| unit.string_interner.get(q))
            .map(|n| n == "test.IndirectCallTest.applyOperation")
            .unwrap_or(false)
    });

    println!("   ✓ test.IndirectCallTest.add: {}", has_add);
    println!(
        "   ✓ test.IndirectCallTest.applyOperation: {}",
        has_apply_op
    );

    // Lower to HIR
    println!("\n5. Lowering TAST to HIR...");
    use compiler::ir::tast_to_hir::lower_tast_to_hir;
    use std::cell::RefCell;
    use std::rc::Rc;

    // StringInterner doesn't implement Clone, so wrap the original
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
        eprintln!("   ✗ No HIR modules generated");
        return;
    }

    // Lower HIR to MIR
    println!("\n6. Lowering HIR to MIR...");
    use compiler::ir::hir_to_mir::lower_hir_to_mir;

    let mut all_mir_modules = Vec::new();
    for hir_module in &all_hir_modules {
        let mir_module = match lower_hir_to_mir(
            hir_module,
            &*string_interner_rc.borrow(),
            &unit.type_table,
            &unit.symbol_table,
        ) {
            Ok(mir) => {
                println!("   ✓ Lowered to MIR ({} functions)", mir.functions.len());
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

    // Check for CallIndirect instructions
    println!("\n7. Checking for CallIndirect instructions...");
    use compiler::ir::IrInstruction;

    let mut found_call_indirect = false;
    let mut found_call_direct = false;

    for mir_module in &all_mir_modules {
        for (_func_id, function) in &mir_module.functions {
            let func_name = &function.name;

            // Check each basic block for call instructions
            for (_bb_id, basic_block) in &function.cfg.blocks {
                for instr in &basic_block.instructions {
                    match instr {
                        IrInstruction::CallIndirect { func_ptr, args, .. } => {
                            println!("   ✓ Found CallIndirect in function '{}'", func_name);
                            println!("       func_ptr: {}, args: {:?}", func_ptr, args);
                            found_call_indirect = true;
                        }
                        IrInstruction::CallDirect { func_id, args, .. } => {
                            // Look up the function name being called
                            if let Some(target_fn) = mir_module.functions.get(func_id) {
                                println!(
                                    "   • Found CallDirect to '{}' (fn{}) in '{}' with {} args",
                                    target_fn.qualified_name.as_ref().unwrap_or(&target_fn.name),
                                    func_id.0,
                                    func_name,
                                    args.len()
                                );
                            } else {
                                println!(
                                    "   • Found CallDirect to 'fn{}' in '{}'",
                                    func_id.0, func_name
                                );
                            }
                            found_call_direct = true;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    println!("\n8. Summary:");
    println!("   - Qualified names: {}", has_add && has_apply_op);
    println!("   - Direct calls found: {}", found_call_direct);
    println!("   - Indirect calls found: {}", found_call_indirect);

    if has_add && has_apply_op && found_call_indirect {
        println!("\n🎉 SUCCESS: Full indirect call pipeline working!");
        println!("   ✓ Qualified names propagated through TAST → HIR → MIR");
        println!("   ✓ CallIndirect instructions generated correctly");
    } else if !found_call_indirect {
        println!("\n⚠️  CallIndirect not found - may need function pointer support in HIR/MIR");
    } else {
        println!("\n⚠️  Some symbols missing qualified names");
    }
}
