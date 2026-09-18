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
#![allow(clippy::needless_return)]
//! Test LLVM JIT with mandelbrot

#[cfg(feature = "llvm-backend")]
use compiler::codegen::LLVMJitBackend;
use compiler::compilation::{CompilationConfig, CompilationUnit};
#[cfg(feature = "llvm-backend")]
use inkwell::context::Context;

fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    let plugin = rayzor_runtime::plugin_impl::get_plugin();
    plugin
        .runtime_symbols()
        .iter()
        .map(|(n, p)| (*n, *p))
        .collect()
}

fn main() {
    #[cfg(not(feature = "llvm-backend"))]
    {
        eprintln!(
            "LLVM backend not enabled. Run with: cargo run --release --features llvm-backend --example test_llvm_mandelbrot"
        );
        return;
    }

    #[cfg(feature = "llvm-backend")]
    {
        let source = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/benchmarks/src/mandelbrot_simple.hx"
        ))
        .expect("read");
        let symbols = get_runtime_symbols();

        println!("Compiling mandelbrot_simple with LLVM...");
        let mut unit = CompilationUnit::new(CompilationConfig::fast());
        unit.load_stdlib().expect("stdlib");
        unit.add_file(&source, "mandelbrot_simple.hx")
            .expect("parse");
        unit.lower_to_tast().expect("tast");
        let mir_modules = unit.get_mir_modules();

        println!("Got {} MIR modules", mir_modules.len());

        let context = Context::create();
        let mut backend = LLVMJitBackend::with_symbols(&context, &symbols).expect("backend");

        println!("Declaring modules...");
        for module in &mir_modules {
            backend.declare_module(module).expect("declare");
        }

        println!("Compiling bodies...");
        for module in &mir_modules {
            backend.compile_module_bodies(module).expect("compile");
        }

        println!("Finalizing (running -O3 optimization)...");
        backend.finalize().expect("finalize");

        println!("Calling main...");
        for module in mir_modules.iter().rev() {
            if backend.call_main(module).is_ok() {
                break;
            }
        }
        println!("Done!");
    }
}
