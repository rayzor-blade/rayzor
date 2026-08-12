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
//! Test array_iteration
use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() -> Result<(), String> {
    println!("=== Testing array_iteration ===\n");

    let code = r#"
class Main {
    static function main() {
        var arr = [1, 2, 3];
        var sum = 0;
        for (x in arr) {
            sum += x;
        }
        trace(sum);
    }
}
"#;

    // Create compilation unit
    let mut unit = CompilationUnit::new(CompilationConfig::default());

    // Load stdlib
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add test code
    unit.add_file(code, "test/Main.hx")
        .map_err(|e| format!("Failed to add file: {}", e))?;

    // Compile to TAST
    unit.lower_to_tast()
        .map_err(|errors| format!("TAST errors: {:?}", errors))?;

    // Get MIR modules
    let mir_modules = unit.get_mir_modules();
    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Print MIR for the main function
    for module in &mir_modules {
        for (_func_id, func) in &module.functions {
            if func.name == "main" {
                println!("\n=== MIR for main ===");
                println!("Function: {}", func.name);
                println!("Blocks: {}", func.cfg.blocks.len());
                for (block_id, block) in &func.cfg.blocks {
                    println!("\nBlock {:?}:", block_id);
                    for instr in &block.instructions {
                        println!("  {:?}", instr);
                    }
                    println!("  TERM: {:?}", block.terminator);
                }
            }
        }
    }

    // Create Cranelift backend with runtime symbols
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = CraneliftBackend::with_symbols(&symbols_ref)?;

    // Compile modules
    for module in &mir_modules {
        backend.compile_module(module)?;
    }

    // Execute
    println!("\n=== Executing ===");
    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            println!("\n=== Execution Complete ===");
            return Ok(());
        }
    }

    Err("No main found".to_string())
}
