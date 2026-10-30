//! rayzor-build: AOT compiler for Haxe -> native executables
//!
//! This is a thin wrapper around `compiler::tools::aot_build`.
//! Prefer using `rayzor aot` instead.
//!
//! Usage:
//!   `rayzor-build [OPTIONS] <SOURCE_FILES...>`

use compiler::codegen::aot_compiler::{LlvmOptimizationLevel, OutputFormat};
use compiler::ir::optimization::OptimizationLevel;
use compiler::tools::aot_build::{self, AotConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut target_triple: Option<String> = None;
    let mut output_format = OutputFormat::Executable;
    let mut opt_level = OptimizationLevel::O2;
    let mut llvm_opt_level = None;
    let mut mir_opt_level = None;
    let mut codegen_opt_level = None;
    let mut strip = true; // tree-shake by default for AOT
    let mut strip_symbols = false;
    let mut verbose = false;
    let mut linker: Option<String> = None;
    let mut runtime_dir: Option<PathBuf> = None;
    let mut sysroot: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut source_files: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" | "-o" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--target" => {
                i += 1;
                if i < args.len() {
                    target_triple = Some(args[i].clone());
                }
            }
            "--emit" => {
                i += 1;
                if i < args.len() {
                    output_format = match args[i].as_str() {
                        "exe" => OutputFormat::Executable,
                        "obj" => OutputFormat::ObjectFile,
                        "llvm-ir" => OutputFormat::LlvmIr,
                        "llvm-bc" => OutputFormat::LlvmBitcode,
                        "asm" => OutputFormat::Assembly,
                        other => {
                            eprintln!(
                                "Unknown emit format: {}. Use: exe, obj, llvm-ir, llvm-bc, asm",
                                other
                            );
                            std::process::exit(1);
                        }
                    };
                }
            }
            "--optimize" | "-O" => {
                i += 1;
                if i < args.len() {
                    let level: LlvmOptimizationLevel = args[i].parse().unwrap_or_else(|error| {
                        eprintln!("{error}");
                        std::process::exit(1)
                    });
                    opt_level = level.default_mir_level();
                    llvm_opt_level = Some(level);
                } else {
                    opt_level = OptimizationLevel::O2;
                }
            }
            "-O0" | "-O1" | "-O2" | "-O3" | "-Os" | "-Oz" => {
                let level: LlvmOptimizationLevel = args[i][2..].parse().unwrap();
                opt_level = level.default_mir_level();
                llvm_opt_level = Some(level);
            }
            "--mir-opt-level" | "--codegen-opt-level" => {
                let mir = args[i] == "--mir-opt-level";
                i += 1;
                let level = match args.get(i).map(String::as_str) {
                    Some("0") => OptimizationLevel::O0,
                    Some("1") => OptimizationLevel::O1,
                    Some("2") => OptimizationLevel::O2,
                    Some("3") => OptimizationLevel::O3,
                    _ => {
                        eprintln!("MIR/codegen optimization level must be 0, 1, 2 or 3");
                        std::process::exit(1);
                    }
                };
                if mir {
                    mir_opt_level = Some(level);
                } else {
                    codegen_opt_level = Some(level);
                }
            }
            "--no-strip" => strip = false,
            "--strip" => strip_symbols = true,
            "--runtime-dir" => {
                i += 1;
                if i < args.len() {
                    runtime_dir = Some(PathBuf::from(&args[i]));
                }
            }
            "--linker" => {
                i += 1;
                if i < args.len() {
                    linker = Some(args[i].clone());
                }
            }
            "--sysroot" => {
                i += 1;
                if i < args.len() {
                    sysroot = Some(PathBuf::from(&args[i]));
                }
            }
            "--verbose" | "-v" => verbose = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            arg if !arg.starts_with('-') => {
                source_files.push(arg.to_string());
            }
            _ => {
                eprintln!("Warning: Unknown argument: {}", args[i]);
            }
        }
        i += 1;
    }

    if source_files.is_empty() {
        eprintln!("Error: No source files specified");
        eprintln!("Usage: rayzor-build [OPTIONS] <SOURCE_FILES...>");
        std::process::exit(1);
    }

    let config = AotConfig {
        source_files,
        output: output_path,
        target_triple,
        output_format,
        opt_level: mir_opt_level.unwrap_or(opt_level),
        llvm_opt_level: Some(llvm_opt_level.unwrap_or_else(|| opt_level.into())),
        codegen_opt_level,
        strip,
        strip_symbols,
        verbose,
        linker,
        runtime_dir,
        sysroot,
        enable_cache: false,
        cache_dir: None,
        extra_source_dirs: Vec::new(),
        native_link_libs: Vec::new(),
    };

    if let Err(e) = aot_build::run_aot(config, Vec::new()) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    println!("rayzor-build: Compile Haxe to native executables");
    println!();
    println!("NOTE: Prefer using `rayzor aot` instead.");
    println!();
    println!("USAGE:");
    println!("    rayzor-build [OPTIONS] <SOURCE_FILES...>");
    println!();
    println!("OPTIONS:");
    println!("    -o, --output <FILE>       Output path (default: <source>.out)");
    println!("    --target <TRIPLE>         Target triple (default: host)");
    println!(
        "    --emit <FORMAT>           Output: exe, obj, llvm-ir, llvm-bc, asm (default: exe)"
    );
    println!("    -O0, -O1, -O2, -O3, -Os, -Oz  LLVM optimization level (default: O2)");
    println!("    --mir-opt-level <0..3>         Override MIR optimization");
    println!("    --codegen-opt-level <0..3>     Override machine-code effort");
    println!("    --no-strip                Disable dead-code stripping");
    println!("    --strip                   Strip debug symbols from binary");
    println!("    --runtime-dir <DIR>       Path to librayzor_runtime.a");
    println!("    --linker <PATH>           Override linker path");
    println!("    --sysroot <PATH>          Sysroot for cross-compilation");
    println!("    -v, --verbose             Verbose output");
    println!("    -h, --help                Show this help message");
}
