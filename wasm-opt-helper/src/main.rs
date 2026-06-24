//! `rayzor-wasm-opt` — a self-contained `wasm-opt -O3 -all` helper.
//!
//! This is a SEPARATE BINARY spawned as a subprocess by the main `rayzor`
//! binary (see `src/wasm_cmd.rs::find_wasm_opt` / `wasm_opt_bytes`). It exists
//! so Binaryen — pulled in by the `wasm-opt` crate, which compiles Binaryen
//! from source and exports ~1890 `llvm::` ADT symbols — never ends up in the
//! same process/link as LLVM 21 (llvm-sys 211). Linking both together makes
//! LLVM's `Module::print` bind to Binaryen's vendored-older-LLVM SmallVector
//! layout and SIGSEGV with a ~6 GB allocation. Keeping Binaryen behind a
//! process boundary is the entire point of this crate.
//!
//! Usage:  `rayzor-wasm-opt <input.wasm> <output.wasm>`
//!
//! Runs `OptimizationOptions::new_opt_level_3().all_features()` — the same
//! `-O3 -all` the old external `wasm-opt` invocation used. `all_features()`
//! sets `FeatureBaseline::All`, so RELAXED-SIMD stays enabled and
//! `i32x4.relaxed_dot` (the Q4 dot kernel → AArch64 SDOT) is NOT lowered away,
//! plus SIMD / THREADS / BULK-MEMORY / NONTRAPPING-FTOI / MUTABLE-GLOBALS /
//! SIGN-EXT.
//!
//! Exit codes: 0 on success; 2 on argv misuse; 1 on any optimization error.

use std::path::Path;
use std::process::ExitCode;

use wasm_opt::OptimizationOptions;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "rayzor-wasm-opt: usage: {} <input.wasm> <output.wasm>",
            args.first().map(String::as_str).unwrap_or("rayzor-wasm-opt")
        );
        return ExitCode::from(2);
    }

    let infile = Path::new(&args[1]);
    let outfile = Path::new(&args[2]);

    // `-O3 -all`. `run()` reads `infile` and writes `outfile` directly (the
    // crate API is file-path based, not in-memory bytes).
    match OptimizationOptions::new_opt_level_3()
        .all_features()
        .run(infile, outfile)
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rayzor-wasm-opt: optimization failed: {e}");
            ExitCode::FAILURE
        }
    }
}
