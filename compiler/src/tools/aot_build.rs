//! AOT build logic extracted for use as a library function.
//!
//! Wraps `AotCompiler` with a config struct for CLI integration.

#[cfg(feature = "llvm-backend")]
use crate::codegen::aot_compiler::{AotCompiler, OutputFormat};
use crate::ir::optimization::OptimizationLevel;
use std::path::PathBuf;

/// Configuration for AOT compilation via the unified CLI.
#[cfg(feature = "llvm-backend")]
pub struct AotConfig {
    /// Source files to compile
    pub source_files: Vec<String>,
    /// Output path
    pub output: Option<PathBuf>,
    /// Target triple (None = host)
    pub target_triple: Option<String>,
    /// Output format
    pub output_format: OutputFormat,
    /// MIR optimization level
    pub opt_level: OptimizationLevel,
    /// Tree-shake unreachable code
    pub strip: bool,
    /// Strip debug symbols from binary
    pub strip_symbols: bool,
    /// Verbose output
    pub verbose: bool,
    /// Custom linker path
    pub linker: Option<String>,
    /// Path to librayzor_runtime.a
    pub runtime_dir: Option<PathBuf>,
    /// Sysroot for cross-compilation
    pub sysroot: Option<PathBuf>,
    /// Enable BLADE incremental cache
    pub enable_cache: bool,
    /// Custom BLADE cache directory
    pub cache_dir: Option<PathBuf>,
}

/// Run AOT compilation with the given config.
///
/// Returns Ok(()) on success.
#[cfg(feature = "llvm-backend")]
pub fn run_aot(config: AotConfig) -> Result<(), String> {
    if config.source_files.is_empty() {
        return Err("No source files specified".to_string());
    }

    let mut compiler = AotCompiler::default();
    compiler.target_triple = config.target_triple;
    compiler.output_format = config.output_format;
    compiler.opt_level = config.opt_level;
    compiler.strip = !config.strip; // AotCompiler.strip means "don't tree-shake" when false
    compiler.strip_symbols = config.strip_symbols;
    compiler.verbose = config.verbose;
    compiler.linker = config.linker;
    compiler.runtime_dir = config.runtime_dir;
    compiler.sysroot = config.sysroot;

    // Default output path
    let output = config.output.unwrap_or_else(|| {
        let base = PathBuf::from(&config.source_files[0])
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        match compiler.output_format {
            OutputFormat::Executable => {
                if cfg!(target_os = "windows")
                    || compiler
                        .target_triple
                        .as_deref()
                        .is_some_and(|t| t.contains("windows"))
                {
                    PathBuf::from(format!("{}.exe", base))
                } else {
                    PathBuf::from(&base)
                }
            }
            OutputFormat::ObjectFile => PathBuf::from(format!("{}.o", base)),
            OutputFormat::LlvmIr => PathBuf::from(format!("{}.ll", base)),
            OutputFormat::LlvmBitcode => PathBuf::from(format!("{}.bc", base)),
            OutputFormat::Assembly => PathBuf::from(format!("{}.s", base)),
            OutputFormat::CSource => PathBuf::from(format!("{}.c", base)),
        }
    });

    println!("Rayzor AOT Compiler");
    if config.verbose {
        println!("  sources  {}", config.source_files.join(", "));
        println!(
            "  output   {} ({:?})",
            output.display(),
            compiler.output_format
        );
        println!(
            "  target   {}",
            compiler.target_triple.as_deref().unwrap_or("host")
        );
        println!("  opt      {:?}", compiler.opt_level);
    }

    // Check sources first (type-check pass, results cached as BLADE artifacts)
    if config.verbose {
        for src in &config.source_files {
            println!("  check    {}", src);
        }
    }

    let compile_result = if compiler.output_format == OutputFormat::CSource {
        compiler.compile_c(&config.source_files, &output)
    } else {
        compiler.compile(&config.source_files, &output)
    };
    match compile_result {
        Ok(result) => {
            println!(
                "  aot      {} ({} bytes, {})",
                result.path.display(),
                result.code_size,
                result.target_triple
            );
            println!("✓ Build succeeded");
            Ok(())
        }
        Err(e) => Err(format!("Build failed: {}", e)),
    }
}

/// Parse an optimization level string.
pub fn parse_opt_level(s: &str) -> OptimizationLevel {
    match s {
        "0" => OptimizationLevel::O0,
        "1" => OptimizationLevel::O1,
        "2" => OptimizationLevel::O2,
        "3" => OptimizationLevel::O3,
        _ => {
            eprintln!("Warning: Unknown optimization level '{}', using O2", s);
            OptimizationLevel::O2
        }
    }
}
