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
//! End-to-end test for generic function compilation
//!
//! This tests the COMPLETE generics pipeline:
//! 1. Parse @:generic class
//! 2. Extract generic metadata in TAST
//! 3. Lower to HIR with type parameters
//! 4. Lower to MIR with TypeVar types
//! 5. Run monomorphization pass
//! 6. Verify specialized functions are generated

use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::ir::IrType;

fn main() {
    println!("=== Generic Function End-to-End Test ===\n");

    // Test 1: Simple generic function
    test_simple_generic_function();

    // Test 2: Generic class with methods
    test_generic_class();

    println!("\n=== All generic E2E tests completed ===");
}

fn test_simple_generic_function() {
    println!("TEST 1: Simple generic function");
    println!("{}", "-".repeat(50));

    let source = r#"
@:generic
class Identity<T> {
    public static function id(x: T): T {
        return x;
    }
}

class Main {
    static function main() {
        var intResult = Identity.id(42);
        var floatResult = Identity.id(3.14);
        trace("Int result: " + intResult);
        trace("Float result: " + floatResult);
    }
}
"#;

    match compile_and_check(source, "test_simple_generic") {
        Ok(stats) => {
            println!("  ✅ Compilation succeeded");
            println!("  📊 Monomorphization stats:");
            println!(
                "     - Generic functions found: {}",
                stats.generic_functions_found
            );
            println!(
                "     - Instantiations created: {}",
                stats.instantiations_created
            );
            println!(
                "     - Call sites rewritten: {}",
                stats.call_sites_rewritten
            );

            if stats.generic_functions_found > 0 {
                println!("  ✅ Generic functions detected");
            } else {
                println!("  ⚠️  No generic functions detected (may need HIR/MIR work)");
            }
        }
        Err(e) => {
            println!("  ❌ Compilation failed: {:?}", e);
        }
    }
    println!();
}

fn test_generic_class() {
    println!("TEST 2: Generic class with methods");
    println!("{}", "-".repeat(50));

    let source = r#"
@:generic
class Box<T> {
    var value: T;

    public function new(v: T) {
        this.value = v;
    }

    public function get(): T {
        return this.value;
    }

    public function set(v: T): Void {
        this.value = v;
    }
}

class Main {
    static function main() {
        var intBox = new Box<Int>(42);
        var val = intBox.get();
        intBox.set(100);
        trace("Box value: " + val);
    }
}
"#;

    match compile_and_check(source, "test_generic_class") {
        Ok(stats) => {
            println!("  ✅ Compilation succeeded");
            println!("  📊 Monomorphization stats:");
            println!(
                "     - Generic functions found: {}",
                stats.generic_functions_found
            );
            println!(
                "     - Instantiations created: {}",
                stats.instantiations_created
            );
            println!(
                "     - Call sites rewritten: {}",
                stats.call_sites_rewritten
            );
        }
        Err(e) => {
            println!("  ❌ Compilation failed: {:?}", e);
        }
    }
    println!();
}

#[derive(Debug, Clone, Default)]
struct MonoStats {
    generic_functions_found: usize,
    instantiations_created: usize,
    call_sites_rewritten: usize,
}

fn compile_and_check(source: &str, name: &str) -> Result<MonoStats, String> {
    // Create compilation unit
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load stdlib first (critical for proper resolution)
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add the source
    unit.add_file(source, &format!("{}.hx", name))
        .map_err(|e| format!("Failed to add file: {}", e))?;

    // Compile to TAST
    let _typed_files = unit
        .lower_to_tast()
        .map_err(|errors| format!("TAST errors: {:?}", errors))?;

    // Get MIR modules
    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Check for generic functions in MIR
    let mut stats = MonoStats::default();

    println!("  📋 User-defined functions in MIR:");
    println!(
        "  📦 Total functions in module: {}",
        mir_modules.iter().map(|m| m.functions.len()).sum::<usize>()
    );

    // Look for monomorphized functions (contain __ in name)
    for module in &mir_modules {
        for (_id, func) in &module.functions {
            if func.name.contains("__") {
                println!("  🎯 Monomorphized: {} (id={:?})", func.name, func.id);
                stats.instantiations_created += 1;
            }
        }
    }

    for module in &mir_modules {
        // Count functions with type parameters
        for (_id, func) in &module.functions {
            // Only show user-defined functions (not stdlib)
            if func.name == "main"
                || func.name == "get"
                || func.name == "set"
                || func.name == "new"
                || func.name == "id"
            {
                println!(
                    "    - {} (type_params: {:?}, return: {:?})",
                    func.name,
                    func.signature
                        .type_params
                        .iter()
                        .map(|p| &p.name)
                        .collect::<Vec<_>>(),
                    func.signature.return_type
                );
            }

            if !func.signature.type_params.is_empty() {
                stats.generic_functions_found += 1;
                println!(
                    "  Found generic function: {} with params {:?}",
                    func.name,
                    func.signature
                        .type_params
                        .iter()
                        .map(|p| &p.name)
                        .collect::<Vec<_>>()
                );
            }

            // Check for TypeVar in signature
            if contains_type_var(&func.signature.return_type) {
                println!("  Function {} returns TypeVar", func.name);
            }
            for param in &func.signature.parameters {
                if contains_type_var(&param.ty) {
                    println!("  Function {} has TypeVar param: {}", func.name, param.name);
                }
            }

            // Check for CallDirect instructions with type_args
            for block in func.cfg.blocks.values() {
                for instr in &block.instructions {
                    if let compiler::ir::IrInstruction::CallDirect {
                        func_id, type_args, ..
                    } = instr
                        && !type_args.is_empty()
                    {
                        println!(
                            "  📞 Call with type_args in {}: func_id={:?}, type_args={:?}",
                            func.name, func_id, type_args
                        );
                        stats.call_sites_rewritten += 1;
                    }
                }
            }
        }
    }

    // Run monomorphization (currently just for stats - the real work happens in compilation)
    // The monomorphization pass runs automatically during compilation

    Ok(stats)
}

fn contains_type_var(ty: &IrType) -> bool {
    match ty {
        IrType::TypeVar(_) => true,
        IrType::Ptr(inner) => contains_type_var(inner),
        IrType::Ref(inner) => contains_type_var(inner),
        IrType::Array(inner, _) => contains_type_var(inner),
        IrType::Slice(inner) => contains_type_var(inner),
        IrType::Generic { base, type_args } => {
            contains_type_var(base) || type_args.iter().any(contains_type_var)
        }
        _ => false,
    }
}
