#![allow(
    unused_imports,
    dead_code,
    clippy::redundant_closure,
    clippy::collapsible_str_replace,
    clippy::single_component_path_imports
)]

//! Pre-compile stdlib to BLADE format
//!
//! This is a thin wrapper around `compiler::tools::preblade`.
//! Prefer using `rayzor bundle` or `rayzor preblade` instead.
//!
//! Usage:
//!   cargo run --bin preblade -- --out .rayzor/blade/stdlib
//!   cargo run --bin preblade -- --list
//!   cargo run --bin preblade -- --bundle app.rzb Main.hx

use std::path::PathBuf;

use compiler::ir::optimization::OptimizationLevel;
use compiler::tools::preblade::{self, BundleConfig, PrebladeConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut out_path: Option<PathBuf> = None;
    let mut bundle_path: Option<PathBuf> = None;
    let mut source_files: Vec<String> = Vec::new();
    let mut list_only = false;
    let mut verbose = false;
    let mut opt_level: Option<OptimizationLevel> = None;
    let mut strip = false;
    let mut compress = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" | "-o" => {
                i += 1;
                if i < args.len() {
                    out_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--bundle" | "-b" => {
                i += 1;
                if i < args.len() {
                    bundle_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--optimize" | "-O" => {
                i += 1;
                if i < args.len() {
                    opt_level = Some(preblade::parse_opt_level(&args[i]));
                } else {
                    opt_level = Some(OptimizationLevel::O2);
                }
            }
            "-O0" => opt_level = Some(OptimizationLevel::O0),
            "-O1" => opt_level = Some(OptimizationLevel::O1),
            "-O2" => opt_level = Some(OptimizationLevel::O2),
            "-O3" => opt_level = Some(OptimizationLevel::O3),
            "--strip" => strip = true,
            "--no-strip" => strip = false,
            "--compress" => compress = true,
            "--no-compress" => compress = false,
            "--list" | "-l" => list_only = true,
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

    // Bundle mode
    if let Some(bundle_out) = bundle_path {
        if source_files.is_empty() {
            eprintln!("Error: No source files specified for bundle");
            eprintln!("Usage: preblade --bundle app.rzb Main.hx [other.hx ...]");
            std::process::exit(1);
        }

        let config = BundleConfig {
            output: bundle_out.clone(),
            source_files,
            verbose,
            opt_level,
            strip,
            compress,
            enable_cache: false,
            cache_dir: None,
            extra_source_dirs: Vec::new(),
            plugins: Vec::new(),
        };

        match preblade::create_bundle(config) {
            Ok(module_count) => {
                println!();
                println!("Bundle created: {}", bundle_out.display());
                println!("  Modules: {}", module_count);
            }
            Err(e) => {
                eprintln!("Bundle creation failed: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    // Standard symbol extraction mode
    let out_path = out_path.unwrap_or_else(|| PathBuf::from(".rayzor/blade/stdlib"));

    if !list_only {
        if let Err(e) = std::fs::create_dir_all(&out_path) {
            eprintln!("Error creating output directory: {}", e);
            std::process::exit(1);
        }
    }

    println!("Pre-BLADE: Extracting stdlib symbols");
    println!("  Output: {}", out_path.display());
    println!();

    let config = PrebladeConfig {
        out_path,
        list_only,
        verbose,
        cache_dir: None,
    };

    match preblade::extract_stdlib_symbols(&config) {
        Ok((classes, enums, aliases)) => {
            println!();
            println!("Pre-BLADE complete:");
            println!("  Classes: {}", classes);
            println!("  Enums:   {}", enums);
            println!("  Aliases: {}", aliases);
        }
        Err(e) => {
            eprintln!("Pre-BLADE failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("preblade - Pre-compile Haxe to BLADE format");
    println!();
    println!("NOTE: Prefer using `rayzor bundle` or `rayzor preblade` instead.");
    println!();
    println!("Usage:");
    println!("  preblade [OPTIONS] [SOURCE_FILES...]");
    println!();
    println!("Modes:");
    println!("  Symbol extraction (default):");
    println!("    preblade --out .rayzor/blade/stdlib");
    println!();
    println!("  Bundle creation:");
    println!("    preblade --bundle app.rzb Main.hx [other.hx ...]");
    println!();
    println!("Options:");
    println!("  --out, -o <PATH>      Output directory for .bsym files");
    println!("  --bundle, -b <FILE>   Create a .rzb bundle from source files");
    println!("  --optimize, -O <N>    Apply MIR optimizations (0-3, default: 2)");
    println!("  -O0, -O1, -O2, -O3   Shorthand for --optimize N");
    println!("  --strip               Enable dead-code stripping (for AOT/size-optimized bundles)");
    println!("  --no-compress         Disable zstd compression (on by default)");
    println!("  --list, -l            List types without generating files");
    println!("  --verbose, -v         Show detailed output");
    println!("  --help, -h            Show this help message");
}
