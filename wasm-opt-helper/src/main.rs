//! `rayzor-wasm-opt` — a self-contained `wasm-opt -O2 -all` helper.
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
//! Runs `OptimizationOptions::new_opt_level_2().all_features()` — `-O2 -all`
//! (NOT -O3: an O3-only aggressive pass miscompiles the relaxed-SIMD
//! relaxed_dot on constant inputs; O2 is correct, same Q4 perf). `all_features()`
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
            args.first()
                .map(String::as_str)
                .unwrap_or("rayzor-wasm-opt")
        );
        return ExitCode::from(2);
    }

    let infile = Path::new(&args[1]);
    let outfile = Path::new(&args[2]);

    // `-O2 -all`. NOT -O3: an O3-only aggressive pass miscompiles the
    // relaxed-SIMD `i32x4.relaxed_dot_i8x16_i7x16_add_s` on constant inputs
    // (test_simd16i8_bitops: and/or/shifts all fold to 0x04040404; xor and any
    // runtime-data dot are unaffected). -O2 fixes it with no measurable perf
    // loss on the Q4 kernel (0.96ms either way, bit-exact). `run()` reads
    // `infile` and writes `outfile` directly (the crate API is path-based).
    match OptimizationOptions::new_opt_level_2()
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
