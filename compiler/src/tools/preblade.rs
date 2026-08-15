//! Pre-compile and bundle creation logic.
//!
//! Extracted from the `preblade` binary for use as library functions.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::compilation::{CompilationConfig, CompilationUnit};
use crate::ir::blade::{
    save_bundle, save_symbol_manifest, BladeAbstractInfo, BladeClassInfo, BladeEnumInfo,
    BladeEnumVariantInfo, BladeFieldInfo, BladeMethodInfo, BladeModuleSymbols, BladeParamInfo,
    BladeTypeAliasInfo, BladeTypeInfo, RayzorBundle,
};
use crate::ir::optimization::{strip_stack_trace_updates, OptimizationLevel, PassManager};
use crate::ir::tree_shake;

/// Configuration for bundle creation.
pub struct BundleConfig {
    /// Output .rzb path
    pub output: PathBuf,
    /// Source files to compile
    pub source_files: Vec<String>,
    /// Verbose output
    pub verbose: bool,
    /// MIR optimization level (None = no optimization)
    pub opt_level: Option<OptimizationLevel>,
    /// Tree-shake unreachable code
    pub strip: bool,
    /// Enable zstd compression
    pub compress: bool,
    /// Enable BLADE incremental cache
    pub enable_cache: bool,
    /// Custom BLADE cache directory
    pub cache_dir: Option<PathBuf>,
    /// Extra source directories (class paths)
    pub extra_source_dirs: Vec<PathBuf>,
    /// Compiler plugins to register
    pub plugins: Vec<Box<dyn crate::compiler_plugin::CompilerPlugin>>,
}

/// Configuration for symbol extraction.
pub struct PrebladeConfig {
    /// Output directory for .bsym files
    pub out_path: PathBuf,
    /// Only list types, don't generate files
    pub list_only: bool,
    /// Verbose output
    pub verbose: bool,
    /// Custom BLADE cache directory
    pub cache_dir: Option<PathBuf>,
}

/// Create a .rzb bundle from source files.
///
/// Returns the number of modules in the bundle.
pub fn create_bundle(mut config: BundleConfig) -> Result<usize, String> {
    use std::time::Instant;

    println!("Creating Rayzor Bundle: {}", config.output.display());

    let t0 = Instant::now();

    // Create compilation unit.
    //
    // A .rzb is a self-contained production artifact and MUST be built from
    // fresh, cross-module-coherent MIR — the BLADE cache is DELIBERATELY
    // ignored here regardless of `config.enable_cache`. Cached per-file MIR
    // carries session-specific function ids from the compile that populated
    // the cache; when those cached modules are merged into the single bundle
    // module, a cross-module CallDirect (e.g. GGUFLoader.loadWithTokenizer ->
    // fn 410017) can survive with an id that no longer exists and that the
    // name-based finalize pass cannot remap, so the bundle SIGILLs at the
    // trap-stub on first call. Compiling from source every time keeps the id
    // space internally consistent. (The cache still speeds up `rayzor run`.)
    let mut comp_config = CompilationConfig::default();
    comp_config.enable_cache = false;
    comp_config.cache_dir = config.cache_dir.clone();
    comp_config.pipeline_config = comp_config.pipeline_config.skip_analysis();
    comp_config.emit_safety_warnings = false;
    let _ = config.enable_cache; // bundle builds never read/write the cache

    let mut unit = CompilationUnit::new(comp_config);

    for dir in &config.extra_source_dirs {
        unit.add_source_path(dir.clone());
    }

    for plugin in config.plugins.drain(..) {
        unit.register_compiler_plugin(plugin);
    }

    // Load stdlib
    if config.verbose {
        println!("  stdlib   loading");
    }
    unit.load_stdlib()
        .map_err(|e| format!("Failed to load stdlib: {}", e))?;

    // Add source files and type-check (results cached as BLADE artifacts)
    for source_file in &config.source_files {
        if config.verbose {
            println!("  check    {}", source_file);
        }
        let source = std::fs::read_to_string(source_file)
            .map_err(|e| format!("Failed to read {}: {}", source_file, e))?;
        unit.add_file(&source, source_file)
            .map_err(|e| format!("Failed to add {}: {}", source_file, e))?;
    }

    // Type-check pass — caches successful checks as BLADE artifacts.
    // If check fails, errors are reported via diagnostics formatter and we do NOT build.
    if let Err(errors) = unit.lower_to_tast() {
        unit.print_compilation_errors(&errors);
        return Err(format!("Check failed with {} error(s)", errors.len()));
    }

    if config.verbose {
        println!("  check    passed");
    }

    // Coherency pass — MUST run before we clone/serialize the MIR. A bundle
    // is an artifact builder: like AOT (aot_compiler.rs), it clones MIR after
    // lower_to_tast() and persists it. Cross-module CallDirect targets are
    // only re-resolved to their final local ids by this pass, which consults
    // `import_mir_modules` as the id source-of-truth. That source-of-truth is
    // NOT carried in the bundle, so any stale ref left in the merged module is
    // frozen at serialize time and surfaces later as an intermittent
    // "function <id> not found" at JIT time (the id survives merge as a
    // dangling cross-module target only on some builds, per HashMap ordering).
    // The interactive `run` path tolerates it because it executes in the same
    // live session; the bundle cannot. Mirror AOT: finalize, THEN snapshot.
    unit.finalize_mir_references();

    // Get MIR modules (uses cached BLADE files when available)
    let mir_modules = unit.get_mir_modules();

    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Bundle exactly the module `rayzor run` compiles: the MERGED module,
    // which `get_mir_modules()` returns LAST (the earlier entries are the
    // per-file/stdlib modules, still un-merged). Bundling all of them was
    // the regression — it (a) double-included the whole stdlib as broken
    // standalone functions (bare `keys` extern with no runtime symbol,
    // `writeString` with an ABI-mismatched signature) that the merged path
    // resolves at the call site, and (b) made tree-shaking miss cross-module
    // CallDirect edges (a callee in another Vec entry was pruned as
    // "unreachable" even though the entry module calls it), producing
    // "function not found" at JIT time. Mirroring run_file here — single
    // merged module, then shake, then link stubs — keeps the two paths
    // byte-identical.
    let mut merged = (**mir_modules.last().unwrap()).clone();
    let module_count = 1usize;

    let entry_function = merged
        .functions
        .values()
        .find(|f| f.name == "main" || f.name == "Main_main" || f.name.ends_with("_main"))
        .map(|f| f.name.clone())
        .ok_or("No entry point found (no main function)")?;
    let entry_module = merged.name.clone();

    if config.verbose {
        println!("  entry    {}::{}", entry_module, entry_function);
    }

    // Tree-shake the SINGLE merged module (always, like run_file). Shaking a
    // multi-module Vec misses cross-module callees; shaking one merged module
    // is complete. Then link forward-declared self-contained wrapper stubs so
    // their calls don't degrade to no-ops (run_file does the same).
    {
        let mut modules = vec![merged];
        let stats = tree_shake::tree_shake_bundle(&mut modules, &entry_module, &entry_function);
        merged = modules.into_iter().next().unwrap();
        merged.link_selfcontained_wrapper_stubs();
        if config.verbose {
            println!(
                "  shake    -{} fn, -{} ext | kept {} fn, {} ext",
                stats.functions_removed,
                stats.extern_functions_removed,
                stats.functions_kept,
                stats.extern_functions_kept
            );
        }
    }

    let mut modules = vec![merged];

    // Apply MIR optimizations after tree-shaking
    if let Some(level) = config.opt_level {
        if level != OptimizationLevel::O0 {
            if config.verbose {
                println!("  opt      {:?} ({} modules)", level, modules.len());
            }
            let mut pass_manager = PassManager::for_level(level);
            for module in &mut modules {
                let _ = pass_manager.run(module);
            }
        }
    }

    // Strip call-frame location update hooks in stripped bundles.
    // This keeps precompiled execution parity with source-JIT benchmark paths.
    if config.strip {
        for module in &mut modules {
            let _ = strip_stack_trace_updates(module);
        }
    }

    // Create and save bundle
    let mut bundle = RayzorBundle::new(modules, &entry_module, &entry_function, None);
    if config.compress {
        bundle.flags.compressed = true;
    }

    save_bundle(&config.output, &bundle).map_err(|e| format!("Failed to save bundle: {}", e))?;

    let elapsed = t0.elapsed();
    println!("  bundle   {} modules in {:?}", module_count, elapsed);

    // Show bundle size
    if let Ok(meta) = std::fs::metadata(&config.output) {
        let size = meta.len();
        if size > 1024 * 1024 {
            println!("  Bundle size: {:.2} MB", size as f64 / (1024.0 * 1024.0));
        } else if size > 1024 {
            println!("  Bundle size: {:.2} KB", size as f64 / 1024.0);
        } else {
            println!("  Bundle size: {} bytes", size);
        }
    }

    Ok(module_count)
}

/// Extract symbols from stdlib.
///
/// Returns (classes, enums, aliases) counts.
pub fn extract_stdlib_symbols(config: &PrebladeConfig) -> Result<(usize, usize, usize), String> {
    // Find stdlib path using the same resolution as CompilationConfig
    let stdlib_path = crate::compilation::CompilationConfig::discover_stdlib_paths()
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| {
            "Could not find stdlib path. Set RAYZOR_STD_PATH or run from project root.".to_string()
        })?;

    if config.list_only {
        println!();
        for module in bsym::build_manifest(&stdlib_path) {
            println!("  {}", module.name);
        }
        return Ok((0, 0, 0));
    }

    let modules = bsym::build_manifest(&stdlib_path);

    let mut total_classes = 0;
    let mut total_enums = 0;
    let mut total_aliases = 0;
    for module in &modules {
        total_classes += module.types.classes.len();
        total_enums += module.types.enums.len();
        total_aliases += module.types.type_aliases.len();
        if config.verbose {
            println!(
                "  {}: {} classes, {} enums, {} aliases, {} abstracts",
                module.name,
                module.types.classes.len(),
                module.types.enums.len(),
                module.types.type_aliases.len(),
                module.types.abstracts.len()
            );
        }
    }

    let manifest_path = config.out_path.join("stdlib.bsym");
    println!();
    println!("Saving symbol manifest to {}...", manifest_path.display());
    println!("  {} modules with symbol information", modules.len());

    if let Err(e) = save_symbol_manifest(&manifest_path, modules) {
        eprintln!("Failed to save symbol manifest: {}", e);
    } else {
        println!("  Symbol manifest saved successfully");
    }

    Ok((total_classes, total_enums, total_aliases))
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

// --- Internal helpers (moved from preblade binary) ---
