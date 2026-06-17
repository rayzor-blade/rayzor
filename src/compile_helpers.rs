//! Shared helpers for compiling Haxe source through the full pipeline to MIR.
//!
//! Used by the `run`, `check`, `build`, `dump`, and WASM command modules.

use std::path::PathBuf;

/// Result of compiling Haxe source to MIR, including metadata needed for codegen.
pub struct MirCompilationResult {
    pub module: compiler::ir::IrModule,
    pub diagnostics: Vec<diagnostics::Diagnostic>,
    pub class_alloc_sizes: std::collections::BTreeMap<String, u64>,
    pub qualified_method_map: std::collections::BTreeMap<String, String>,
}

/// Compile Haxe source through the full pipeline to MIR.
///
/// Uses `CompilationUnit` for proper multi-file, stdlib-aware compilation.
/// Returns the primary MIR module (user code) and any diagnostics (warnings).
pub fn compile_haxe_to_mir(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
    safety_warnings: bool,
) -> Result<(compiler::ir::IrModule, Vec<diagnostics::Diagnostic>), String> {
    let result = compile_haxe_to_mir_full(
        source,
        filename,
        plugins,
        extra_source_dirs,
        safety_warnings,
    )?;
    Ok((result.module, result.diagnostics))
}

/// Compile Haxe source through the full pipeline with explicit cache controls.
pub fn compile_haxe_to_mir_with_cache(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
    safety_warnings: bool,
    enable_cache: bool,
    cache_dir: Option<PathBuf>,
) -> Result<(compiler::ir::IrModule, Vec<diagnostics::Diagnostic>), String> {
    let result = compile_haxe_to_mir_with_defines_and_cache(
        source,
        filename,
        plugins,
        extra_source_dirs,
        safety_warnings,
        &[],
        enable_cache,
        cache_dir,
    )?;
    Ok((result.module, result.diagnostics))
}

/// Same as `compile_haxe_to_mir` but returns the full `MirCompilationResult`.
pub fn compile_haxe_to_mir_full(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
    safety_warnings: bool,
) -> Result<MirCompilationResult, String> {
    compile_haxe_to_mir_with_defines(
        source,
        filename,
        plugins,
        extra_source_dirs,
        safety_warnings,
        &[],
    )
}

/// Compile with additional preprocessor defines (e.g. `#if wasm`).
pub fn compile_haxe_to_mir_with_defines(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
    safety_warnings: bool,
    extra_defines: &[&str],
) -> Result<MirCompilationResult, String> {
    compile_haxe_to_mir_with_defines_and_cache(
        source,
        filename,
        plugins,
        extra_source_dirs,
        safety_warnings,
        extra_defines,
        true,
        None,
    )
}

/// Compile with additional preprocessor defines and explicit cache controls.
pub fn compile_haxe_to_mir_with_defines_and_cache(
    source: &str,
    filename: &str,
    plugins: Vec<Box<dyn compiler::compiler_plugin::CompilerPlugin>>,
    extra_source_dirs: &[PathBuf],
    safety_warnings: bool,
    extra_defines: &[&str],
    enable_cache: bool,
    cache_dir: Option<PathBuf>,
) -> Result<MirCompilationResult, String> {
    use compiler::compilation::{CompilationConfig, CompilationUnit};

    let mut config = CompilationConfig {
        load_stdlib: true,
        enable_cache,
        cache_dir,
        emit_safety_warnings: safety_warnings,
        extra_defines: extra_defines.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    config.pipeline_config = config.pipeline_config.skip_analysis();
    if std::env::var("RAYZOR_TRACE_CACHE").is_ok() {
        eprintln!(
            "[cache] compile_haxe_to_mir enable_cache={} cache_dir={:?} defines={:?}",
            config.enable_cache, config.cache_dir, config.extra_defines
        );
    }

    let mut unit = CompilationUnit::new(config);

    for plugin in plugins {
        unit.register_compiler_plugin(plugin);
    }

    for dir in extra_source_dirs {
        unit.add_source_path(dir.clone());
    }

    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    unit.add_file(source, filename)?;

    if let Err(errors) = unit.lower_to_tast() {
        unit.print_compilation_errors(&errors);
        return Err(format!("Check failed with {} error(s)", errors.len()));
    }

    let mir_modules = unit.get_mir_modules();

    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    let mut module = (**mir_modules.last().unwrap()).clone();
    // L1: on the wasm/Cranelift-JIT path the stdlib value-type wrappers
    // (Ptr/Usize/Box/SIMD4f) are merged into the module AFTER the single
    // InliningPass runs and so each value-type op stays an un-inlined
    // CallDirect to a 1-3 instruction leaf. Native AOT re-runs its own
    // PassManager (aot_compiler.rs) so this only matters for wasm. Run one
    // InliningPass on the merged module, gated on the "wasm" define so native
    // callers (which pass &[]) are untouched. The wrappers are well under the
    // cost model's 50-instruction threshold; the forward-ref-stub guard
    // (d5f2b48) and fresh-pass reg-id recompute (655d7ac) both hold.
    if extra_defines.iter().any(|d| *d == "wasm") {
        use compiler::ir::inlining::InliningPass;
        use compiler::ir::optimization::OptimizationPass;
        let mut pass = InliningPass::new();
        let _ = pass.run_on_module(&mut module);
    }
    let diagnostics = unit.collected_diagnostics.clone();
    let class_alloc_sizes = unit.get_class_alloc_sizes_by_name().clone();
    let qualified_method_map = unit.get_qualified_method_map().clone();
    Ok(MirCompilationResult {
        module,
        diagnostics,
        class_alloc_sizes,
        qualified_method_map,
    })
}
