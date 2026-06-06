//! rayzor-build: AOT compiler for Haxe -> native executables
//!
//! This is a thin wrapper around `compiler::tools::aot_build`.
//! Prefer using `rayzor aot` instead.
//!
//! Usage:
//!   `rayzor-build [OPTIONS] <SOURCE_FILES...>`

use compiler::codegen::aot_compiler::OutputFormat;
use compiler::ir::optimization::OptimizationLevel;
use compiler::tools::aot_build::{self, AotConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut target_triple: Option<String> = None;
    let mut output_format = OutputFormat::Executable;
    let mut opt_level = OptimizationLevel::O2;
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
                    opt_level = aot_build::parse_opt_level(&args[i]);
                } else {
                    opt_level = OptimizationLevel::O2;
                }
            }
            "-O0" => opt_level = OptimizationLevel::O0,
            "-O1" => opt_level = OptimizationLevel::O1,
            "-O2" => opt_level = OptimizationLevel::O2,
            "-O3" => opt_level = OptimizationLevel::O3,
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
        opt_level,
        strip,
        strip_symbols,
        verbose,
        linker,
        runtime_dir,
        sysroot,
        enable_cache: false,
        cache_dir: None,
    };

    if let Err(e) = aot_build::run_aot(config) {
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
    println!("    -O0, -O1, -O2, -O3       Optimization level (default: O2)");
    println!("    --no-strip                Disable dead-code stripping");
    println!("    --strip                   Strip debug symbols from binary");
    println!("    --runtime-dir <DIR>       Path to librayzor_runtime.a");
    println!("    --linker <PATH>           Override linker path");
    println!("    --sysroot <PATH>          Sysroot for cross-compilation");
    println!("    -v, --verbose             Verbose output");
    println!("    -h, --help                Show this help message");
}
