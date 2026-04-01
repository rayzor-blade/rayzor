#![allow(
    unused_imports,
    unused_variables,
    dead_code,
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
use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() {
    let mut unit = CompilationUnit::new(CompilationConfig::fast());
    unit.load_stdlib().expect("stdlib");

    let source = r#"
package test;

import rayzor.Box;
import rayzor.Ptr;
import rayzor.Usize;

class Main {
    static function main() {
        var boxed = Box.init(99);
        var ptr:Ptr<Int> = boxed.asPtr();
        trace(ptr.deref());    // 99
        trace(ptr.raw());      // pointer address
        var addr:Usize = 42;
        trace(addr);           // 42 via @:to Int
        boxed.free();
    }
}
"#;

    unit.add_file(source, "test_ptr.hx").expect("add file");

    match unit.lower_to_tast() {
        Ok(files) => println!("TAST: {} files", files.len()),
        Err(e) => {
            println!("TAST failed: {:?}", e);
            return;
        }
    }

    let mir_modules = unit.get_mir_modules();
    println!("MIR: {} modules", mir_modules.len());

    for (i, m) in mir_modules.iter().enumerate() {
        let has_main = m.functions.values().any(|f| f.name.contains("main"));
        println!(
            "  Module {}: {} functions, {} externs, has_main={}",
            i,
            m.functions.len(),
            m.extern_functions.len(),
            has_main
        );
        if has_main {
            // Build func_id -> name map
            let mut id_to_name: std::collections::BTreeMap<compiler::ir::IrFunctionId, String> =
                std::collections::BTreeMap::new();
            for (fid, f) in &m.functions {
                id_to_name.insert(*fid, f.name.clone());
            }
            for (fid, f) in &m.extern_functions {
                id_to_name.insert(*fid, format!("extern:{}", f.name));
            }
            for (_fid, f) in &m.functions {
                if f.name.contains("main") && !f.name.contains("__") {
                    println!("\n  === {} ===", f.name);
                    for (_bid, block) in &f.cfg.blocks {
                        for inst in &block.instructions {
                            if let compiler::ir::IrInstruction::CallDirect {
                                func_id,
                                dest,
                                args,
                                ..
                            } = inst
                            {
                                let name =
                                    id_to_name.get(func_id).map(|s| s.as_str()).unwrap_or("???");
                                println!(
                                    "      call {} ({:?}) args={:?} -> {:?}",
                                    name, func_id, args, dest
                                );
                            } else {
                                println!("      {:?}", inst);
                            }
                        }
                    }
                }
            }
        }
    }

    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = CraneliftBackend::with_symbols(&symbols_ref).expect("backend");

    for (i, module) in mir_modules.iter().enumerate() {
        println!("Compiling module {}...", i);
        match backend.compile_module(module) {
            Ok(_) => println!("  OK"),
            Err(e) => {
                println!("  FAILED: {}", e);
                return;
            }
        }
    }

    println!("Executing...");
    for module in mir_modules.iter().rev() {
        match backend.call_main(module) {
            Ok(()) => {
                println!("  Execution OK");
                return;
            }
            Err(e) => println!("  No main in this module: {}", e),
        }
    }
    println!("No main found!");
}
