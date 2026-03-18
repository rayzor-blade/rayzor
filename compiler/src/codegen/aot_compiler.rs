//! AOT (Ahead-of-Time) Compiler for Rayzor
//!
//! Compiles Haxe source files to native executables via LLVM.
//! Supports cross-compilation to any LLVM target triple.

#[cfg(feature = "llvm-backend")]
use inkwell::context::Context;
#[cfg(feature = "llvm-backend")]
use inkwell::targets::RelocMode;

use crate::compilation::{CompilationConfig, CompilationUnit};
use crate::ir::optimization::{strip_stack_trace_updates, OptimizationLevel, PassManager};
use crate::ir::tree_shake;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Output format for AOT compilation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    /// Linked native executable (default)
    Executable,
    /// Object file (.o) — user links manually
    ObjectFile,
    /// LLVM IR text (.ll)
    LlvmIr,
    /// LLVM bitcode (.bc)
    LlvmBitcode,
    /// Native assembly (.s)
    Assembly,
    /// C source code compiled with gcc/g++ (no LLVM dependency)
    CSource,
}

/// Result of AOT compilation
pub struct AotOutput {
    pub path: PathBuf,
    pub format: OutputFormat,
    pub target_triple: String,
    pub code_size: u64,
}

/// AOT compiler configuration
pub struct AotCompiler {
    /// Target triple (None = host)
    pub target_triple: Option<String>,
    /// MIR optimization level
    pub opt_level: OptimizationLevel,
    /// Output format
    pub output_format: OutputFormat,
    /// Whether to tree-shake unused code
    pub strip: bool,
    /// Verbose output
    pub verbose: bool,
    /// Custom linker path
    pub linker: Option<String>,
    /// Path to librayzor_runtime.a
    pub runtime_dir: Option<PathBuf>,
    /// Sysroot for cross-compilation
    pub sysroot: Option<PathBuf>,
    /// Strip debug symbols from binary
    pub strip_symbols: bool,
}

impl Default for AotCompiler {
    fn default() -> Self {
        Self {
            target_triple: None,
            opt_level: OptimizationLevel::O2,
            output_format: OutputFormat::Executable,
            strip: true,
            verbose: false,
            linker: None,
            runtime_dir: None,
            sysroot: None,
            strip_symbols: false,
        }
    }
}

impl AotCompiler {
    /// Compile Haxe source files to native output.
    #[cfg(feature = "llvm-backend")]
    pub fn compile(
        &self,
        source_files: &[String],
        output_path: &Path,
    ) -> Result<AotOutput, String> {
        use crate::codegen::llvm_aot_backend;
        use crate::codegen::llvm_jit_backend::LLVMJitBackend;
        use std::time::Instant;

        let t0 = Instant::now();

        // --- Phase 1: Parse and compile to MIR ---
        if self.verbose {
            println!("  Parsing and lowering to MIR...");
        }

        let mut unit = CompilationUnit::new(CompilationConfig::default());
        unit.load_stdlib()
            .map_err(|e| format!("Failed to load stdlib: {}", e))?;

        for source_file in source_files {
            let source = std::fs::read_to_string(source_file)
                .map_err(|e| format!("Failed to read {}: {}", source_file, e))?;
            unit.add_file(&source, source_file)
                .map_err(|e| format!("Failed to add {}: {}", source_file, e))?;
        }

        unit.lower_to_tast()
            .map_err(|errors| format!("Compilation failed: {:?}", errors))?;

        let mir_modules = unit.get_mir_modules();
        if mir_modules.is_empty() {
            return Err("No MIR modules generated".to_string());
        }

        let mut modules: Vec<_> = mir_modules.iter().map(|m| (**m).clone()).collect();

        // --- Phase 2: MIR optimizations ---
        // Check if system LLVM tools are available for optimization.
        // When they are, cap MIR at O2 because MIR O3's GVN pass changes FP
        // operation ordering, and system LLVM (newer version) optimizes differently
        // with the reordered ops, producing different FP results. System opt -O3
        // handles GVN/vectorization/etc. natively anyway.
        let has_system_tools = (self.output_format == OutputFormat::Executable
            || self.output_format == OutputFormat::ObjectFile)
            && llvm_aot_backend::has_system_llvm_tools();
        let mir_opt = if has_system_tools && self.opt_level == OptimizationLevel::O3 {
            OptimizationLevel::O2
        } else {
            self.opt_level
        };
        if mir_opt != OptimizationLevel::O0 {
            if self.verbose {
                println!("  Applying MIR optimizations ({:?})...", mir_opt);
            }
            let mut pass_manager = PassManager::for_level(mir_opt);
            for module in &mut modules {
                let _ = pass_manager.run(module);
                let _ = strip_stack_trace_updates(module);
            }
        }

        // --- Phase 3: Find entry point ---
        let (entry_module_name, entry_function_name) = find_entry_point(&modules)?;
        if self.verbose {
            println!(
                "  Entry point: {}::{}",
                entry_module_name, entry_function_name
            );
        }

        // --- Phase 4: Tree-shake ---
        if self.strip {
            if self.verbose {
                println!("  Tree-shaking...");
            }
            let stats = tree_shake::tree_shake_bundle(
                &mut modules,
                &entry_module_name,
                &entry_function_name,
            );
            if self.verbose {
                println!(
                    "    Removed: {} functions, {} externs, {} globals, {} empty modules",
                    stats.functions_removed,
                    stats.extern_functions_removed,
                    stats.globals_removed,
                    stats.modules_removed
                );
                println!(
                    "    Kept: {} functions, {} externs",
                    stats.functions_kept, stats.extern_functions_kept
                );
            }
        }

        // --- Phase 5: LLVM compilation ---
        if self.verbose {
            println!("  Compiling to LLVM IR...");
        }

        llvm_aot_backend::init_llvm_aot();

        let llvm_opt = match self.opt_level {
            OptimizationLevel::O0 => inkwell::OptimizationLevel::None,
            OptimizationLevel::O1 => inkwell::OptimizationLevel::Less,
            OptimizationLevel::O2 => inkwell::OptimizationLevel::Default,
            OptimizationLevel::O3 => inkwell::OptimizationLevel::Aggressive,
        };

        let context = Context::create();
        let mut backend = LLVMJitBackend::with_aot_mode(&context, llvm_opt)?;

        // Two-pass: declare all, then compile all bodies
        for module in &modules {
            backend.declare_module(module)?;
        }
        for module in &modules {
            backend.compile_module_bodies(module)?;
        }

        // Find the LLVM function name for the entry point
        let entry_llvm_name = find_entry_llvm_name(&backend, &modules, &entry_function_name)?;
        let startup_llvm_names = find_startup_llvm_names(&backend, &modules);

        // --- Phase 6: AOT-specific emit via llvm_aot_backend ---
        let module = backend.get_module();
        let target_triple_str = self.target_triple.as_deref();

        // For executables and object files, try system LLVM tools (opt + llc) first.
        // System LLVM (typically v19-21) has better inlining heuristics than the
        // bundled LLVM 18, producing ~2.7x faster code for hot-loop benchmarks.
        let opt_flag = match self.opt_level {
            OptimizationLevel::O0 => "-O0",
            OptimizationLevel::O1 => "-O1",
            OptimizationLevel::O2 => "-O2",
            OptimizationLevel::O3 => "-O3",
        };

        if self.output_format == OutputFormat::Executable
            || self.output_format == OutputFormat::ObjectFile
        {
            // Dump IR WITHOUT main wrapper — optimization should see user code only.
            // The main wrapper will be linked separately as a tiny C file so that
            // system opt doesn't inline the entry into the C main() (which changes
            // the optimization context for inner functions and alters FP results).
            //
            // Set data layout + triple first so opt has accurate type size info.
            llvm_aot_backend::set_module_target(module, target_triple_str)?;
            let ir_text = module.print_to_string().to_string();

            let obj_path = if self.output_format == OutputFormat::Executable {
                output_path.with_extension("o")
            } else {
                output_path.to_path_buf()
            };

            if self.verbose {
                println!("  Optimizing and emitting object file...");
            }

            let rename_entry = if self.output_format == OutputFormat::Executable {
                Some(entry_llvm_name.as_str())
            } else {
                None
            };
            let used_system = llvm_aot_backend::compile_ir_with_system_tools(
                &ir_text,
                &obj_path,
                opt_flag,
                rename_entry,
            )?;

            if !used_system {
                // Fall back to inkwell optimization + codegen
                if self.verbose {
                    println!("  (using built-in LLVM optimization)");
                }
                llvm_aot_backend::optimize_module(module, target_triple_str, llvm_opt)?;
                // Generate main() wrapper after inkwell optimization
                if self.output_format == OutputFormat::Executable {
                    llvm_aot_backend::generate_main_wrapper(
                        module,
                        &entry_llvm_name,
                        &startup_llvm_names,
                    )?;
                }
                llvm_aot_backend::compile_to_object_file(
                    module,
                    &obj_path,
                    target_triple_str,
                    RelocMode::PIC,
                    llvm_opt,
                )?;
            }

            if self.output_format == OutputFormat::Executable {
                if self.verbose {
                    println!("  Linking...");
                }
                // When using system tools, link a C main() wrapper separately
                if used_system {
                    self.link_executable_with_entry(
                        &obj_path,
                        output_path,
                        &entry_llvm_name,
                        &startup_llvm_names,
                    )?;
                } else {
                    self.link_executable(&obj_path, output_path)?;
                }
                let _ = std::fs::remove_file(&obj_path);
            }
        } else {
            // For IR/bitcode/asm output, use inkwell directly
            llvm_aot_backend::optimize_module(module, target_triple_str, llvm_opt)?;

            if self.output_format == OutputFormat::Executable {
                llvm_aot_backend::generate_main_wrapper(
                    module,
                    &entry_llvm_name,
                    &startup_llvm_names,
                )?;
            }

            match self.output_format {
                OutputFormat::LlvmIr => {
                    if self.verbose {
                        println!("  Emitting LLVM IR...");
                    }
                    llvm_aot_backend::emit_llvm_ir(module, output_path)?;
                }
                OutputFormat::LlvmBitcode => {
                    if self.verbose {
                        println!("  Emitting LLVM bitcode...");
                    }
                    llvm_aot_backend::emit_llvm_bitcode(module, output_path)?;
                }
                OutputFormat::Assembly => {
                    if self.verbose {
                        println!("  Emitting assembly...");
                    }
                    llvm_aot_backend::emit_assembly(
                        module,
                        output_path,
                        target_triple_str,
                        llvm_opt,
                    )?;
                }
                _ => unreachable!(),
            }
        }

        let elapsed = t0.elapsed();
        let code_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

        let actual_triple = self.target_triple.clone().unwrap_or_else(|| {
            #[cfg(feature = "llvm-backend")]
            {
                use inkwell::targets::TargetMachine;
                TargetMachine::get_default_triple()
                    .as_str()
                    .to_string_lossy()
                    .to_string()
            }
            #[cfg(not(feature = "llvm-backend"))]
            {
                "unknown".to_string()
            }
        });

        if self.verbose {
            println!("  Done in {:?}", elapsed);
        }

        Ok(AotOutput {
            path: output_path.to_path_buf(),
            format: self.output_format,
            target_triple: actual_triple,
            code_size,
        })
    }

    /// Link an object file into a native executable
    fn link_executable(&self, obj_path: &Path, output_path: &Path) -> Result<(), String> {
        let linker = self.find_linker()?;
        let runtime_path = self.find_runtime()?;

        let mut cmd = Command::new(&linker);

        // Output path
        cmd.arg("-o").arg(output_path);

        // Object file
        cmd.arg(obj_path);

        // Runtime library (static)
        cmd.arg(&runtime_path);

        // Optimization level for linker (matches LLVM codegen optimization)
        let opt_flag = match self.opt_level {
            OptimizationLevel::O0 => "-O0",
            OptimizationLevel::O1 => "-O1",
            OptimizationLevel::O2 => "-O2",
            OptimizationLevel::O3 => "-O3",
        };
        cmd.arg(opt_flag);

        // Cross-compilation target
        if let Some(ref triple) = self.target_triple {
            cmd.arg(format!("--target={}", triple));
        }

        // Sysroot for cross-compilation
        if let Some(ref sysroot) = self.sysroot {
            cmd.arg(format!("--sysroot={}", sysroot.display()));
        }

        // Platform-specific linker flags
        let triple_str = self.target_triple.as_deref().unwrap_or("");
        if triple_str.contains("darwin") || triple_str.is_empty() && cfg!(target_os = "macos") {
            // macOS
            cmd.args(["-lSystem", "-lc", "-lm", "-lpthread"]);
            cmd.args(["-framework", "CoreFoundation", "-framework", "Security"]);
        } else if triple_str.contains("windows") {
            // Windows
            cmd.args(["kernel32.lib", "ws2_32.lib", "userenv.lib", "bcrypt.lib"]);
        } else {
            // Linux / other Unix
            cmd.args(["-lc", "-lm", "-lpthread", "-ldl"]);
        }

        // Strip debug symbols
        if self.strip_symbols {
            cmd.arg("-s");
        }

        if self.verbose {
            println!("    {}", format_command(&cmd));
        }

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to run linker: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Linking failed:\n{}", stderr));
        }

        Ok(())
    }

    /// Link an object file into a native executable with a C main() entry point.
    /// Used when system LLVM tools handle optimization (so the main wrapper isn't
    /// baked into the LLVM IR). The C main calls the Haxe entry function.
    fn link_executable_with_entry(
        &self,
        obj_path: &Path,
        output_path: &Path,
        entry_func_name: &str,
        startup_func_names: &[String],
    ) -> Result<(), String> {
        // Write a tiny C main() that calls the Haxe entry point.
        // If the entry was "main", it was renamed to "_haxe_main" in the IR.
        let actual_entry = if entry_func_name == "main" {
            "_haxe_main"
        } else {
            entry_func_name
        };
        let main_c_path = output_path.with_extension("_main.c");
        let mut main_c = String::from("extern void rayzor_init_args_from_argv(int, char**);\n");
        for startup in startup_func_names {
            main_c.push_str(&format!("extern void {}(long);\n", startup));
        }
        main_c.push_str(&format!("extern void {}(long);\n", actual_entry));
        main_c
            .push_str("int main(int argc, char** argv) { rayzor_init_args_from_argv(argc, argv);");
        for startup in startup_func_names {
            main_c.push_str(&format!(" {}(0);", startup));
        }
        main_c.push_str(&format!(" {}(0); return 0; }}\n", actual_entry));
        std::fs::write(&main_c_path, &main_c)
            .map_err(|e| format!("Failed to write main wrapper: {}", e))?;

        let linker = self.find_linker()?;
        let runtime_path = self.find_runtime()?;

        let mut cmd = Command::new(&linker);
        cmd.arg("-o").arg(output_path);
        cmd.arg(obj_path);
        cmd.arg(&main_c_path);
        cmd.arg(&runtime_path);

        let opt_flag = match self.opt_level {
            OptimizationLevel::O0 => "-O0",
            OptimizationLevel::O1 => "-O1",
            OptimizationLevel::O2 => "-O2",
            OptimizationLevel::O3 => "-O3",
        };
        cmd.arg(opt_flag);

        if let Some(ref triple) = self.target_triple {
            cmd.arg(format!("--target={}", triple));
        }
        if let Some(ref sysroot) = self.sysroot {
            cmd.arg(format!("--sysroot={}", sysroot.display()));
        }

        let triple_str = self.target_triple.as_deref().unwrap_or("");
        if triple_str.contains("darwin") || triple_str.is_empty() && cfg!(target_os = "macos") {
            cmd.args(["-lSystem", "-lc", "-lm", "-lpthread"]);
            cmd.args(["-framework", "CoreFoundation", "-framework", "Security"]);
        } else if triple_str.contains("windows") {
            cmd.args(["kernel32.lib", "ws2_32.lib", "userenv.lib", "bcrypt.lib"]);
        } else {
            cmd.args(["-lc", "-lm", "-lpthread", "-ldl"]);
        }

        if self.strip_symbols {
            cmd.arg("-s");
        }

        if self.verbose {
            println!("    {}", format_command(&cmd));
        }

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to run linker: {}", e))?;

        // Cleanup
        let _ = std::fs::remove_file(&main_c_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Linking failed:\n{}", stderr));
        }

        Ok(())
    }

    /// Find a suitable linker
    fn find_linker(&self) -> Result<String, String> {
        if let Some(ref linker) = self.linker {
            return Ok(linker.clone());
        }

        // Try common linkers in order of preference
        for candidate in &["clang", "gcc", "cc"] {
            if Command::new(candidate)
                .arg("--version")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return Ok(candidate.to_string());
            }
        }

        Err("No linker found. Install clang or gcc, or pass --linker <path>.".to_string())
    }

    /// Find the runtime static library
    fn find_runtime(&self) -> Result<PathBuf, String> {
        // 1. Explicit --runtime-dir
        if let Some(ref dir) = self.runtime_dir {
            let path = dir.join("librayzor_runtime.a");
            if path.exists() {
                return Ok(path);
            }
            return Err(format!(
                "librayzor_runtime.a not found in {}",
                dir.display()
            ));
        }

        // 2. RAYZOR_RUNTIME_DIR env var
        if let Ok(dir) = std::env::var("RAYZOR_RUNTIME_DIR") {
            let path = PathBuf::from(&dir).join("librayzor_runtime.a");
            if path.exists() {
                return Ok(path);
            }
        }

        // 3. Check relative to executable location first so release binaries prefer
        // the matching release runtime archive in Cargo's deps output.
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let mut candidates = vec![
                    exe_dir.join("librayzor_runtime.a"),
                    exe_dir.join("deps").join("librayzor_runtime.a"),
                ];
                if let Some(parent) = exe_dir.parent() {
                    candidates.push(parent.join("librayzor_runtime.a"));
                    candidates.push(parent.join("deps").join("librayzor_runtime.a"));
                }
                for path in candidates {
                    if path.exists() {
                        return Ok(path);
                    }
                }
            }
        }

        // 4. Check relative to cargo workspace (release first, then debug; deps
        // before root to match Cargo's normal staticlib layout).
        for profile in &["release", "debug"] {
            for path in [
                PathBuf::from(format!("target/{}/deps/librayzor_runtime.a", profile)),
                PathBuf::from(format!("target/{}/librayzor_runtime.a", profile)),
            ] {
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        Err("librayzor_runtime.a not found. Build it with:\n  \
             cargo build --release -p rayzor-runtime\n\
             Or pass --runtime-dir <path>"
            .to_string())
    }

    #[cfg(not(feature = "llvm-backend"))]
    pub fn compile(
        &self,
        _source_files: &[String],
        _output_path: &Path,
    ) -> Result<AotOutput, String> {
        Err("AOT compilation requires the llvm-backend feature. \
             Rebuild with: cargo build --features llvm-backend"
            .to_string())
    }
    /// Compile Haxe source files to C, then compile with gcc/g++.
    /// Does NOT require the LLVM backend.
    pub fn compile_c(
        &self,
        source_files: &[String],
        output_path: &Path,
    ) -> Result<AotOutput, String> {
        use crate::codegen::c_backend::CBackend;
        use std::process::Command;
        use std::time::Instant;

        let t0 = Instant::now();

        // --- Phase 1: Parse and compile to MIR ---
        if self.verbose {
            println!("  Parsing and lowering to MIR...");
        }

        let mut unit = CompilationUnit::new(CompilationConfig::default());
        unit.load_stdlib()
            .map_err(|e| format!("Failed to load stdlib: {}", e))?;

        for source_file in source_files {
            let source = std::fs::read_to_string(source_file)
                .map_err(|e| format!("Failed to read {}: {}", source_file, e))?;
            unit.add_file(&source, source_file)
                .map_err(|e| format!("Failed to add {}: {}", source_file, e))?;
        }

        unit.lower_to_tast()
            .map_err(|errors| format!("Compilation failed: {:?}", errors))?;

        let mir_modules = unit.get_mir_modules();
        if mir_modules.is_empty() {
            return Err("No MIR modules generated".to_string());
        }

        let mut modules: Vec<_> = mir_modules.iter().map(|m| (**m).clone()).collect();

        // --- Phase 2: MIR optimizations ---
        if self.opt_level != OptimizationLevel::O0 {
            if self.verbose {
                println!("  Applying MIR optimizations ({:?})...", self.opt_level);
            }
            let mut pass_manager = PassManager::for_level(self.opt_level);
            for module in &mut modules {
                let _ = pass_manager.run(module);
                let _ = strip_stack_trace_updates(module);
            }
        }

        // --- Phase 3: Find entry point ---
        let (_entry_module_name, entry_function_name) = find_entry_point(&modules)?;
        if self.verbose {
            println!("  Entry point: {}", entry_function_name);
        }

        // --- Phase 4: Tree-shake ---
        if self.strip {
            if self.verbose {
                println!("  Tree-shaking...");
            }
            let stats = tree_shake::tree_shake_bundle(
                &mut modules,
                &_entry_module_name,
                &entry_function_name,
            );
            if self.verbose {
                println!(
                    "    Removed: {} functions, {} externs",
                    stats.functions_removed, stats.extern_functions_removed
                );
            }
        }

        // --- Phase 5: Emit C source ---
        if self.verbose {
            println!("  Emitting C source...");
        }

        // Find startup functions
        let startup_funcs: Vec<String> = modules
            .iter()
            .flat_map(|m| {
                m.functions.values().filter_map(|f| {
                    if f.name == "__vtable_init__" || f.name == "__init__" {
                        Some(f.name.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        let c_source = CBackend::emit_modules(&modules, &entry_function_name, &startup_funcs)?;

        // If output format is CSource, just write the .c file
        if self.output_format == OutputFormat::CSource {
            let c_path = output_path.with_extension("c");
            std::fs::write(&c_path, &c_source)
                .map_err(|e| format!("Failed to write C source: {}", e))?;
            let code_size = c_source.len() as u64;
            return Ok(AotOutput {
                path: c_path,
                format: OutputFormat::CSource,
                target_triple: "native".to_string(),
                code_size,
            });
        }

        // --- Phase 6: Compile C with gcc ---
        if self.verbose {
            println!("  Compiling with gcc...");
        }

        let c_path = std::env::temp_dir().join("rayzor_aot_c.c");
        std::fs::write(&c_path, &c_source)
            .map_err(|e| format!("Failed to write temp C source: {}", e))?;

        let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let opt_flag = match self.opt_level {
            OptimizationLevel::O0 => "-O0",
            OptimizationLevel::O1 => "-O1",
            OptimizationLevel::O2 => "-O2",
            OptimizationLevel::O3 => "-O3",
        };

        // Find runtime library
        let runtime_lib = self.find_runtime()?;

        let mut cmd = Command::new(&cc);
        cmd.arg("-o").arg(output_path);
        cmd.arg(&c_path);
        cmd.arg(&runtime_lib);
        cmd.arg(opt_flag);
        cmd.arg("-lm");

        // Platform-specific flags
        #[cfg(target_os = "linux")]
        {
            cmd.arg("-lpthread").arg("-ldl");
        }
        #[cfg(target_os = "macos")]
        {
            cmd.arg("-framework").arg("CoreFoundation");
            cmd.arg("-framework").arg("Security");
        }

        let compile_out = cmd
            .output()
            .map_err(|e| format!("Failed to run {}: {}", cc, e))?;

        // Clean up temp file
        let _ = std::fs::remove_file(&c_path);

        if !compile_out.status.success() {
            let stderr = String::from_utf8_lossy(&compile_out.stderr);
            return Err(format!("C compilation failed:\n{}", stderr));
        }

        let code_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

        if self.verbose {
            println!("  Done in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        }

        Ok(AotOutput {
            path: output_path.to_path_buf(),
            format: OutputFormat::Executable,
            target_triple: "native".to_string(),
            code_size,
        })
    }
}

/// Find the entry point (module name, function name) from MIR modules
fn find_entry_point(modules: &[crate::ir::IrModule]) -> Result<(String, String), String> {
    // Search for a function named "main" in user modules (at the end)
    for module in modules.iter().rev() {
        for (_func_id, func) in &module.functions {
            if func.name == "main" || func.name.ends_with("_main") {
                return Ok((module.name.clone(), func.name.clone()));
            }
        }
    }
    Err("No entry point found. Define a main() function.".to_string())
}

/// Find the LLVM function name for the entry point
#[cfg(feature = "llvm-backend")]
fn find_entry_llvm_name(
    backend: &crate::codegen::llvm_jit_backend::LLVMJitBackend,
    modules: &[crate::ir::IrModule],
    entry_function_name: &str,
) -> Result<String, String> {
    // Get function symbols and find the one matching our entry point
    let symbols = backend.get_function_symbols();
    for (_id, name) in &symbols {
        if name == entry_function_name || name.ends_with(&format!("_{}", entry_function_name)) {
            return Ok(name.clone());
        }
    }

    // Also check if the mangled name exists directly in the module
    // The LLVM module may have mangled the name
    for module in modules.iter().rev() {
        for (func_id, func) in &module.functions {
            if func.name == entry_function_name || func.name.ends_with("_main") {
                if let Some(name) = symbols.get(func_id) {
                    return Ok(name.clone());
                }
            }
        }
    }

    Err(format!(
        "Entry function '{}' not found in compiled LLVM module",
        entry_function_name
    ))
}

#[cfg(feature = "llvm-backend")]
fn find_startup_llvm_names(
    backend: &crate::codegen::llvm_jit_backend::LLVMJitBackend,
    modules: &[crate::ir::IrModule],
) -> Vec<String> {
    use std::collections::HashSet;

    let symbols = backend.get_function_symbols();
    let mut startup_names = Vec::new();
    let mut seen = HashSet::new();

    for module in modules {
        for hook_name in ["__vtable_init__", "__init__"] {
            let Some((func_id, _)) = module.functions.iter().find(|(_, f)| f.name == hook_name)
            else {
                continue;
            };

            let Some(symbol_name) = symbols.get(func_id) else {
                continue;
            };

            if seen.insert(symbol_name.clone()) {
                startup_names.push(symbol_name.clone());
            }
        }
    }

    startup_names
}

fn format_command(cmd: &Command) -> String {
    let prog = cmd.get_program().to_string_lossy().to_string();
    let args: Vec<_> = cmd
        .get_args()
        .map(|a| a.to_string_lossy().to_string())
        .collect();
    format!("{} {}", prog, args.join(" "))
}
