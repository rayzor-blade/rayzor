#![allow(unused_imports, unused_variables, dead_code, unreachable_patterns)]
#![allow(clippy::single_component_path_imports)]

use compiler::codegen::CraneliftBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};

fn main() {
    let code = r#"
class Main {
    static function main() {
        var arr = [10, 20, 30];

        // Test manual iterator
        var it = arr.iterator();
        while (it.hasNext()) {
            trace(it.next());
        }

        // Test KV iterator
        var kvit = arr.keyValueIterator();
        while (kvit.hasNext()) {
            var kv = kvit.next();
            trace(kv.key);
            trace(kv.value);
        }

        trace("done");
    }
}
"#;

    let mut unit = CompilationUnit::new(CompilationConfig::fast());

    if let Err(e) = unit.load_stdlib() {
        println!("FAIL: load_stdlib: {}", e);
        return;
    }

    if let Err(e) = unit.add_file(code, "test.hx") {
        println!("FAIL: add_file: {}", e);
        return;
    }

    println!("Lowering to TAST...");
    if let Err(e) = unit.lower_to_tast() {
        println!("FAIL: TAST: {:?}", e);
        return;
    }

    println!("Getting MIR modules...");
    let mir_modules = unit.get_mir_modules();
    println!("Got {} MIR modules", mir_modules.len());

    if mir_modules.is_empty() {
        println!("FAIL: No MIR modules");
        return;
    }

    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    let symbols = plugin.runtime_symbols();
    let symbols_ref: Vec<(&str, *const u8)> = symbols.iter().map(|(n, p)| (*n, *p)).collect();

    let mut backend = match CraneliftBackend::with_symbols(&symbols_ref) {
        Ok(b) => b,
        Err(e) => {
            println!("FAIL: Backend: {}", e);
            return;
        }
    };

    // Dump all function names across all modules
    for (i, module) in mir_modules.iter().enumerate() {
        println!(
            "=== Module {} ({} functions) ===",
            i,
            module.functions.len()
        );
        for (fid, func) in &module.functions {
            println!("  {:?}: {}", fid, func.name);
        }
    }

    // Dump MIR of the last module (user code) to check for hasNext/next calls
    if let Some(last_module) = mir_modules.last() {
        for func in last_module.functions.values() {
            if func.name.contains("main") || func.name.contains("Main") {
                println!("\n=== MIR of {} ===", func.name);
                for (bid, block) in &func.cfg.blocks {
                    println!("  {:?}:", bid);
                    for inst in &block.instructions {
                        println!("    {:?}", inst);
                    }
                    println!("    terminator: {:?}", block.terminator);
                }
            }
        }
    }

    println!("Compiling {} modules...", mir_modules.len());
    for (i, module) in mir_modules.iter().enumerate() {
        println!(
            "  Compiling module {} ({} functions)...",
            i,
            module.functions.len()
        );
        if let Err(e) = backend.compile_module(module) {
            println!("  FAIL: Compile module {}: {}", i, e);
            return;
        }
    }

    println!("Calling main...");
    for module in mir_modules.iter().rev() {
        if backend.call_main(module).is_ok() {
            println!("PASS");
            return;
        }
    }

    println!("FAIL: No main found");
}
