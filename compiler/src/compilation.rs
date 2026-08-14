//! Multi-file Compilation Infrastructure
//!
//! This module provides the proper architecture for compiling multiple source files
//! together, including standard library loading, package management, and symbol resolution.

use crate::compiler_plugin::CompilerPluginRegistry;
use crate::dependency_graph::{CircularDependency, DependencyAnalysis, DependencyGraph};
use crate::ir::{
    blade::{
        load_blade, load_symbol_manifest, save_blade_with_state, BladeAbstractInfo, BladeAccessor,
        BladeCachedMaps, BladeClassInfo, BladeEnumInfo, BladeFieldEntry, BladeFuncEntry,
        BladeMetadata, BladeMethodInfo, BladePropertyEntry, BladeSymbolManifest,
        BladeTypeAliasInfo, BladeTypeInfo,
    },
    IrFunctionId, IrInstruction, IrModule, IrValue, Monomorphizer,
};
use crate::pipeline::{
    CompilationError, CompilationResult, ErrorCategory, HaxeCompilationPipeline, PipelineConfig,
};
use crate::stdlib::hdll_plugin::HdllPlugin;
use crate::tast::{
    namespace::{ImportResolver, NamespaceResolver},
    stdlib_loader::{StdLibConfig, StdLibLoader},
    symbols::SymbolFlags,
    AstLowering, ScopeId, ScopeTree, SourceLocation, StringInterner, SymbolId, SymbolTable, TypeId,
    TypeKind, TypeTable, TypedFile,
};
use log::{debug, info, trace, warn};
use parser::{parse_haxe_file, parse_haxe_file_with_debug, HaxeFile};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, Default)]
pub struct TypecheckStageTimings {
    pub hdll_ms: f64,
    pub dependency_ms: f64,
    pub import_scan_ms: f64,
    pub import_load_ms: f64,
    pub import_hx_ms: f64,
    pub user_files_ms: f64,
    pub result_stdlib_ms: f64,
    pub file_parse_ms: f64,
    pub macro_ms: f64,
    pub ast_lower_ms: f64,
    pub send_sync_ms: f64,
    pub ownership_ms: f64,
    pub hir_ms: f64,
    pub extern_check_ms: f64,
    pub mir_prep_ms: f64,
    pub mir_lower_core_ms: f64,
    pub mir_ms: f64,
    pub stdlib_merge_ms: f64,
    pub monomorphize_ms: f64,
    pub files_seen: usize,
    pub macro_skipped_files: usize,
    pub imports_collected: usize,
    pub import_cache_hits: usize,
    pub import_cache_misses: usize,
    pub import_fresh_compiles: usize,
    pub import_typedef_fresh: usize,
    pub import_already_compiled: usize,
    pub import_extern_skips: usize,
}

#[inline]
fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[inline]
fn profile_timer(enabled: bool) -> Option<Instant> {
    enabled.then(Instant::now)
}

#[inline]
fn add_profile_ms(slot: &mut f64, start: Option<Instant>) {
    if let Some(start) = start {
        *slot += elapsed_ms(start);
    }
}

#[inline]
fn finish_profile_ms(slot: &mut f64, start: Option<Instant>) -> f64 {
    if let Some(start) = start {
        let ms = elapsed_ms(start);
        *slot += ms;
        ms
    } else {
        0.0
    }
}

#[inline]
fn is_stdtypes_ambient_name(name: &str) -> bool {
    matches!(
        name,
        "Void"
            | "Float"
            | "Int"
            | "Single"
            | "Null"
            | "Bool"
            | "Dynamic"
            | "Iterator"
            | "Iterable"
            | "KeyValueIterator"
            | "KeyValueIterable"
            | "ArrayAccess"
    )
}

#[inline]
fn is_stdtypes_ambient_import(path: &str) -> bool {
    is_stdtypes_ambient_name(path)
        || path == "StdTypes"
        || path
            .strip_prefix("StdTypes.")
            .is_some_and(is_stdtypes_ambient_name)
}

#[inline]
fn is_extern_only_stdlib_base(base: &str) -> bool {
    matches!(
        base,
        "Math" | "Array" | "String" | "Std" | "Class" | "Enum" | "EnumValue" | "Any"
    )
}

#[inline]
fn is_extern_only_stdlib_import(path: &str) -> bool {
    is_extern_only_stdlib_base(path.rsplit('.').next().unwrap_or(path))
}

#[inline]
fn is_manifest_backed_ambient_import(path: &str, stdlib_manifest_loaded: bool) -> bool {
    is_stdtypes_ambient_import(path)
        || (stdlib_manifest_loaded && is_extern_only_stdlib_import(path))
}

#[inline]
fn haxe_file_source_has_macro_hook(file: &parser::HaxeFile) -> bool {
    let Some(source) = file.input.as_deref() else {
        return false;
    };
    source.contains("macro ")
        || source.contains("macro\t")
        || source.contains("@:build")
        || source.contains("@:autoBuild")
        || source.contains("@:genericBuild")
}

fn collect_ir_value_function_refs(value: &IrValue, out: &mut Vec<IrFunctionId>) {
    match value {
        IrValue::Function(func_id) => out.push(*func_id),
        IrValue::Closure {
            function,
            environment,
        } => {
            out.push(*function);
            collect_ir_value_function_refs(environment, out);
        }
        IrValue::Array(items) | IrValue::Struct(items) => {
            for item in items {
                collect_ir_value_function_refs(item, out);
            }
        }
        _ => {}
    }
}

fn collect_ir_instruction_function_refs(inst: &IrInstruction, out: &mut Vec<IrFunctionId>) {
    match inst {
        IrInstruction::CallDirect { func_id, .. }
        | IrInstruction::FunctionRef { func_id, .. }
        | IrInstruction::MakeClosure { func_id, .. } => out.push(*func_id),
        IrInstruction::Const { value, .. } => collect_ir_value_function_refs(value, out),
        _ => {}
    }
}

fn retain_referenced_stdlib_functions(
    stdlib_mir: &mut IrModule,
    root_names: &BTreeSet<String>,
) -> usize {
    use std::collections::VecDeque;

    let before = stdlib_mir.functions.len();
    let mut needed: BTreeSet<IrFunctionId> = BTreeSet::new();
    let mut queue = VecDeque::new();

    for (func_id, func) in &stdlib_mir.functions {
        if root_names.contains(&func.name) {
            needed.insert(*func_id);
            queue.push_back(*func_id);
        }
    }

    while let Some(func_id) = queue.pop_front() {
        let Some(func) = stdlib_mir.functions.get(&func_id) else {
            continue;
        };
        let mut refs = Vec::new();
        for block in func.cfg.blocks.values() {
            for inst in &block.instructions {
                collect_ir_instruction_function_refs(inst, &mut refs);
            }
        }
        for ref_id in refs {
            if stdlib_mir.functions.contains_key(&ref_id) && needed.insert(ref_id) {
                queue.push_back(ref_id);
            }
        }
    }

    stdlib_mir
        .functions
        .retain(|func_id, _| needed.contains(func_id));
    before.saturating_sub(stdlib_mir.functions.len())
}

/// Represents a complete compilation unit with multiple source files
pub struct CompilationUnit {
    /// Stdlib files (loaded first with haxe.* package)
    pub stdlib_files: Vec<HaxeFile>,

    /// Global import.hx files (loaded after stdlib, before user files)
    pub import_hx_files: Vec<HaxeFile>,

    /// User source files
    pub user_files: Vec<HaxeFile>,

    /// True when stdlib symbols were populated from a BLADE manifest.
    /// In that mode top-level extern-only stdlib imports are already known and
    /// should not be resolved back to source during cold import discovery.
    stdlib_manifest_loaded: bool,

    /// Macro expansion origins (for IDE hints showing expanded results)
    pub macro_expansions: Vec<crate::macro_system::expander::ExpansionOrigin>,

    /// Shared string interner
    pub string_interner: StringInterner,

    /// Symbol table (shared across all files)
    pub symbol_table: SymbolTable,

    /// Type table (shared across all files)
    pub type_table: Rc<RefCell<TypeTable>>,

    /// Scope tree (shared across all files)
    pub scope_tree: ScopeTree,

    /// Namespace resolver
    pub namespace_resolver: NamespaceResolver,

    /// Import resolver
    pub import_resolver: ImportResolver,

    /// Configuration
    pub config: CompilationConfig,

    /// Cache of types that failed to load on-demand (to avoid repeated attempts)
    pub failed_type_loads: BTreeSet<String>,

    /// Declared static-method signatures of every parsed class/abstract,
    /// shared into each per-file AstLowering so a call site can type a
    /// not-yet-lowered class's static from its declaration (no decay).
    static_sig_index: Rc<RefCell<crate::tast::sig_index::StaticSigIndex>>,

    /// Cache of files that have been successfully compiled (to avoid redundant recompilation)
    /// Maps filename to the TypedFile result
    compiled_files: BTreeMap<String, TypedFile>,

    /// Internal compilation pipeline (delegates to HaxeCompilationPipeline)
    pipeline: HaxeCompilationPipeline,

    /// MIR modules generated during compilation (collected from pipeline results)
    mir_modules: Vec<std::sync::Arc<crate::ir::IrModule>>,

    /// MIR modules from on-demand imported stdlib files (e.g., BalancedTree.hx).
    /// These are merged into the user module during stdlib renumbering rather than
    /// being stored as separate modules, because their function IDs would collide.
    import_mir_modules: Vec<crate::ir::IrModule>,

    /// Function IDs from import modules that correspond to source-level declarations
    /// (class methods, constructors, top-level functions — NOT generated MIR wrappers).
    /// These must be protected from stdlib merge name collisions.
    import_own_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId>,

    /// Stdlib typed files loaded on-demand (typedefs, etc. that need to be in HIR)
    loaded_stdlib_typed_files: Vec<TypedFile>,

    /// Raw ASTs of files loaded via import resolution (stdlib and user-package imports).
    /// Used for cross-file macro discovery — when a file is expanded, macros defined
    /// in its imports (e.g., `import tink.Json`) must be registered first. The compiled
    /// `TypedFile` form isn't usable here; macro expansion runs before TAST lowering.
    loaded_import_haxe_files: Vec<HaxeFile>,

    /// Monotonic file_id counter handed out to each compiled file in
    /// arrival order so per-statement spans can identify their source.
    /// Previously `compile_file_with_shared_state_ex_with_parsed`
    /// hardcoded `FileId::new(0)` for EVERY file, causing every
    /// TypedExpression / TypedStatement span in imported nue/* modules
    /// to carry file_id=0 — the root of bugs_diagnostic_span_file_id_always_zero.
    /// Bumped each time a file enters the lowering pass; mirrored into the
    /// span_converter via initialize_span_converter_with_filename.
    next_file_id: u32,

    /// Maps absolute (or as-provided) source file path → assigned
    /// file_id so the same file imported via multiple paths still
    /// resolves to the same compilation-level identity. Keyed by the
    /// filename string that was passed to compile_file_with_shared_state_ex.
    file_id_by_filename: BTreeMap<String, u32>,

    /// Per-file source bytes captured at the moment the file entered
    /// the lowering pipeline. The renderer in `build_full_source_map`
    /// uses THIS exact bytes blob for ariadne so byte_offsets stored
    /// on diagnostic spans (computed against the same bytes) resolve
    /// to the right line/column. Falling back to a fresh disk read
    /// can drift by a line or two on files whose on-disk version
    /// changed between lowering and rendering, or whose in-memory
    /// version diverged via macro expansion / line-ending
    /// normalisation.
    file_source_by_filename: BTreeMap<String, String>,

    /// Diagnostics collected during compilation (warnings, non-exhaustive switches, etc.)
    /// These are printed during compilation AND stored here for cache replay.
    pub collected_diagnostics: Vec<diagnostics::Diagnostic>,

    /// Accumulated class_fields from all compiled files.
    /// Allows cross-file static field access (e.g., BufferUsage.VERTEX from imported GraphicsTypes.hx).
    global_class_fields: BTreeMap<
        crate::tast::SymbolId,
        Vec<(crate::tast::InternedString, crate::tast::SymbolId, bool)>,
    >,

    /// Mapping from HIR function symbols to MIR function IDs for stdlib functions
    /// This allows user code to call pure Haxe stdlib functions (like StringTools)
    stdlib_function_map: BTreeMap<crate::tast::SymbolId, crate::ir::IrFunctionId>,

    /// Name-based mapping from qualified function names to MIR function IDs
    /// This is used for cross-file lookups where SymbolIds differ between compilation units
    /// e.g., "StringTools.startsWith" -> IrFunctionId(N)
    stdlib_function_name_map: BTreeMap<String, crate::ir::IrFunctionId>,

    /// Accumulated field index map from imported files (SymbolId -> (TypeId, field_index))
    /// Passed to user file's MIR lowering so it can resolve field access on imported classes
    import_field_index_map: BTreeMap<crate::tast::SymbolId, (crate::tast::TypeId, u32)>,

    /// Last-attempt import-compile errors per module name. try_compile_import runs
    /// once per retry pass, so its errors must NOT be printed inline (that
    /// over-reports transient ordering failures a later pass resolves). They are
    /// stashed here and only the genuine FINAL survivors are surfaced after the
    /// retry loop converges.
    last_import_errors: BTreeMap<String, Vec<String>>,

    /// Accumulated field class names from imported files (SymbolId -> qualified class name)
    /// Used by BLADE cache to serialize field entries with correct class names
    import_field_class_names: BTreeMap<crate::tast::SymbolId, String>,

    /// Accumulated property access map from imported files
    import_property_access_map: BTreeMap<crate::tast::SymbolId, crate::tast::PropertyAccessInfo>,

    /// Accumulated constructor name map from imported files (class name -> constructor IrFunctionId)
    /// Passed to user file's MIR lowering so it can resolve `new ClassName()` for imported classes
    import_constructor_name_map: BTreeMap<String, crate::ir::IrFunctionId>,

    /// Per-parameter qualified type names for imported functions, keyed by
    /// renumbered IrFunctionId. Recovers HIR-level info that MIR alone
    /// erases (Class and Interface both become `Ptr(Void)`), so the user
    /// file's `maybe_materialize_for_call` Path 3 can decide whether to
    /// wrap a class arg in an interface fat pointer when calling an
    /// imported constructor. None entries mean the slot isn't a
    /// class/interface (primitive, anonymous, etc.).
    import_function_param_iface_names: BTreeMap<crate::ir::IrFunctionId, Vec<Option<String>>>,

    /// Accumulated constructor arity for imported functions, keyed by the
    /// renumbered MIR function id. Kept incrementally so each import/user MIR
    /// lowering does not rescan every previously lowered import module.
    import_constructor_param_counts: BTreeMap<crate::ir::IrFunctionId, usize>,

    /// Accumulated lowered parameter types for imported functions, keyed by the
    /// renumbered MIR function id. Used for optional/default argument fill and
    /// kept incrementally to avoid O(imports^2) MIR-prep work.
    import_function_param_types: BTreeMap<crate::ir::IrFunctionId, Vec<crate::ir::IrType>>,

    /// Accumulated imported globals keyed by qualified name.
    import_external_globals: BTreeMap<String, (crate::ir::IrGlobalId, crate::ir::IrType)>,

    /// Accumulated class allocation sizes from imported files (SymbolId -> byte size).
    /// Keyed by the class declaration's SymbolId — stable across module contexts,
    /// unlike TypeIds. Passed to user file's MIR lowering so it knows how much
    /// memory to allocate for imported classes.
    import_class_alloc_sizes: BTreeMap<crate::tast::SymbolId, u64>,

    /// Name-keyed class allocation sizes (class_name → bytes).
    /// Stable across compilation contexts where TypeIds differ.
    import_class_alloc_sizes_by_name: BTreeMap<String, u64>,

    /// Accumulated class TypeId → SymbolId mapping from imported files.
    /// Used for field disambiguation when multiple classes share the same field name.
    import_class_type_to_symbol: BTreeMap<crate::tast::TypeId, crate::tast::SymbolId>,

    /// Accumulated class method symbols from imported files
    /// Passed to user file's MIR lowering for iterator protocol resolution
    import_class_method_symbols:
        BTreeMap<(crate::tast::SymbolId, crate::tast::InternedString), crate::tast::SymbolId>,

    /// Accumulated interface metadata from imported files. Cross-file
    /// interface dispatch (`var t:Iface = new Class(); t.method()` in a
    /// file that imports `Iface`) requires this — otherwise the
    /// per-file `HirToMirContext` starts empty and method lookup
    /// silently falls through, dropping the call expression.
    import_interface_method_names:
        BTreeMap<crate::tast::SymbolId, Vec<crate::tast::InternedString>>,
    import_interface_method_return_types:
        BTreeMap<(crate::tast::SymbolId, crate::tast::InternedString), crate::tast::TypeId>,
    import_interface_extends: BTreeMap<crate::tast::SymbolId, Vec<crate::tast::SymbolId>>,
    import_interface_vtables:
        BTreeMap<(crate::tast::SymbolId, crate::tast::SymbolId), Vec<crate::tast::SymbolId>>,

    /// Compiler plugin registry (builtin + HDLL plugins)
    compiler_plugin_registry: CompilerPluginRegistry,

    /// Function pointers collected from loaded HDLL plugins for JIT linking
    hdll_symbols: Vec<(String, *const u8)>,

    /// Set of already-loaded HDLL library names to avoid duplicate loading
    loaded_hdlls: BTreeSet<String>,

    /// Global inline var constants (name-keyed: "ClassName.fieldName" → value).
    /// Populated from both fresh compilation and BLADE cache restore.
    /// Passed to HIR lowering for cross-file static inline var resolution.
    global_inline_vars: BTreeMap<String, crate::ir::blade::BladeInlineValue>,

    /// Type info extracted from the last compiled file (for BLADE cache save)
    last_compiled_type_info: Option<BladeTypeInfo>,

    /// MIR cross-reference maps from the last compiled file (for BLADE cache save)
    last_compiled_cached_maps: Option<BladeCachedMaps>,

    /// Source-level function IDs from the last compiled file (methods + constructors).
    /// Used by try_compile_import to track which functions in import modules are
    /// genuine declarations vs generated MIR wrappers.
    last_compiled_own_func_ids: Option<std::collections::BTreeSet<crate::ir::IrFunctionId>>,

    /// Cached stdlib MIR module (built once, cloned for each user file merge)
    cached_stdlib_mir: Option<crate::ir::IrModule>,

    /// Map from extern function symbol name → JS module name.
    /// Populated by register_extern_methods_from_typed_file for @:jsImport classes.
    /// Used by WASM backend to import from correct JS module instead of "rayzor".
    pub extern_js_module_map: BTreeMap<String, String>,

    /// Map from qualified Haxe method name (e.g. "rayzor.gpu.Surface.getFormat")
    /// to native symbol name (e.g. "rayzor_gpu_gfx_surface_get_format").
    /// Used by WASM backend to resolve stub wrapper functions to their imports.
    pub qualified_method_map: BTreeMap<String, String>,

    /// Last `lower_to_tast` timing breakdown. This is deliberately coarse enough
    /// to stay cheap, but detailed enough to keep "typecheck" from hiding HIR,
    /// MIR, merge, and macro-expansion work.
    typecheck_timings: TypecheckStageTimings,
}

/// Configuration for compilation
#[derive(Clone)]
pub struct CompilationConfig {
    /// Paths to search for standard library files
    pub stdlib_paths: Vec<PathBuf>,

    /// Default stdlib imports to load
    pub default_stdlib_imports: Vec<String>,

    /// Whether to load stdlib
    pub load_stdlib: bool,

    /// Root package for stdlib (e.g., "haxe")
    pub stdlib_root_package: Option<String>,

    /// Global import.hx files to process (loaded before user files, after stdlib)
    pub global_import_hx_files: Vec<PathBuf>,

    /// Enable incremental compilation with BLADE cache
    pub enable_cache: bool,

    /// Directory for BLADE cache files
    pub cache_dir: Option<PathBuf>,

    /// Lazy stdlib loading - skip upfront symbol registration for faster cold start
    /// When enabled, stdlib symbols are loaded on-demand when first referenced
    /// This trades first-access latency for faster initial startup
    pub lazy_stdlib: bool,

    /// Pipeline configuration for analysis and optimization
    pub pipeline_config: PipelineConfig,

    /// Directories to search for .hdll files (referenced by @:hlNative metadata)
    pub hdll_search_paths: Vec<PathBuf>,

    /// Whether to emit safety warnings (use-after-move, etc.)
    pub emit_safety_warnings: bool,

    /// Extra preprocessor defines (e.g., "wasm" for WASM target builds).
    /// Added to the default set ("rayzor", "sys").
    pub extra_defines: Vec<String>,

    /// Collect detailed typecheck timing breakdowns. Off by default because the
    /// probes sit on cold-start paths and should only exist during profiling.
    pub profile_typecheck: bool,
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            stdlib_paths: Self::discover_stdlib_paths(),
            default_stdlib_imports: vec![
                "StdTypes.hx".to_string(), // Contains Iterator typedef
                "String.hx".to_string(),
                "Array.hx".to_string(),
                "Math.hx".to_string(), // Top-level Math functions (sqrt, sin, cos, etc.)
                "Std.hx".to_string(),  // Top-level conversion utilities
                "Type.hx".to_string(), // ValueType enum + Type reflection APIs
                // Concurrent types
                "rayzor/concurrent/Thread.hx".to_string(),
                "rayzor/concurrent/Channel.hx".to_string(),
                "rayzor/concurrent/Mutex.hx".to_string(),
                "rayzor/concurrent/Arc.hx".to_string(),
                // Array iterator classes (compiled as regular Haxe, not runtime-backed)
                "haxe/iterators/ArrayIterator.hx".to_string(),
                "haxe/iterators/ArrayKeyValueIterator.hx".to_string(),
            ],
            load_stdlib: true,
            stdlib_root_package: Some("haxe".to_string()), // Prefix stdlib with "haxe.*" namespace
            global_import_hx_files: Vec::new(),            // No global import.hx by default
            enable_cache: true, // Cache enabled - BLADE manifest now includes Math, Std, Date, etc.
            cache_dir: None,    // Auto-discover cache directory when needed
            lazy_stdlib: false, // Default to eager loading for compatibility
            pipeline_config: PipelineConfig::default(),
            hdll_search_paths: vec![PathBuf::from(".")],
            emit_safety_warnings: true,
            extra_defines: Vec::new(),
            profile_typecheck: false,
        }
    }
}

impl CompilationConfig {
    /// Discover standard library paths from environment and standard locations
    ///
    /// Search order:
    /// Discover rayzor's own stdlib (haxe-std).
    ///
    /// Resolution order:
    /// 1. RAYZOR_STD_PATH environment variable (explicit override)
    /// 2. Relative to the rayzor binary (../haxe-std, ../compiler/haxe-std)
    /// 3. Relative to cwd (compiler/haxe-std, ./haxe-std, ../haxe-std)
    ///
    /// NOTE: System Haxe installations (/usr/local/lib/haxe/std etc.) are NOT
    /// searched. Rayzor uses its own stdlib with rayzor-specific extensions.
    /// Mixing system Haxe stdlib causes subtle compilation errors.
    pub fn discover_stdlib_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // 1. Explicit override via RAYZOR_STD_PATH
        if let Ok(std_path) = std::env::var("RAYZOR_STD_PATH") {
            let path = PathBuf::from(&std_path);
            if path.exists() {
                info!("Found stdlib at RAYZOR_STD_PATH: {}", std_path);
                paths.push(path);
                return paths;
            } else {
                warn!(
                    "RAYZOR_STD_PATH set but directory doesn't exist: {}",
                    std_path
                );
            }
        }

        // 2. Walk up from the binary location looking for haxe-std/
        if let Ok(exe) = std::env::current_exe() {
            if let Some(mut dir) = exe.parent().map(|p| p.to_path_buf()) {
                for _ in 0..5 {
                    for name in &["haxe-std", "compiler/haxe-std"] {
                        let candidate = dir.join(name);
                        if candidate.is_dir() {
                            paths.push(candidate);
                        }
                    }
                    if !dir.pop() {
                        break;
                    }
                }
            }
        }

        // 3. Walk up from cwd looking for haxe-std/
        if let Ok(mut dir) = std::env::current_dir() {
            for _ in 0..5 {
                for name in &["haxe-std", "compiler/haxe-std"] {
                    let candidate = dir.join(name);
                    if candidate.is_dir() {
                        // Deduplicate by canonical path
                        let dominated = paths.iter().any(|p| {
                            matches!(
                                (p.canonicalize(), candidate.canonicalize()),
                                (Ok(a), Ok(b)) if a == b
                            )
                        });
                        if !dominated {
                            paths.push(candidate);
                        }
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }

        if paths.is_empty() {
            warn!("No rayzor stdlib found. Set RAYZOR_STD_PATH environment variable.");
            // Fallback for development
            paths.push(PathBuf::from("compiler/haxe-std"));
            paths.push(PathBuf::from("./haxe-std"));
        }

        paths
    }

    /// Get the current target triple (e.g., "x86_64-macos", "aarch64-linux")
    pub fn get_target_triple() -> String {
        let arch = std::env::consts::ARCH;
        let os = std::env::consts::OS;
        format!("{}-{}", arch, os)
    }

    /// Target discriminator for cache paths.
    ///
    /// The same `.hx` source lowers to DIFFERENT MIR depending on
    /// `extra_defines` (`#if wasm`, host-import vs native-runtime bindings,
    /// etc.). Keying the cache only by source path made native and wasm builds
    /// of the same file share one `.blade` slot — running native then wasm (or
    /// vice versa) served the wrong-target MIR until `.rayzor` was deleted by
    /// hand. Folding the sorted defines into the cache directory gives native
    /// and wasm fully separate cache trees.
    pub fn cache_discriminator(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        if self.extra_defines.is_empty() {
            return "native".to_string();
        }
        let mut defines = self.extra_defines.clone();
        defines.sort();
        let tag = if defines.iter().any(|d| d == "wasm") {
            "wasm"
        } else {
            "native"
        };
        let mut hasher = DefaultHasher::new();
        defines.hash(&mut hasher);
        format!("{}-{:08x}", tag, hasher.finish() as u32)
    }

    /// Get or create the cache directory (target-discriminated — see
    /// [`cache_discriminator`]).
    pub fn get_cache_dir(&self) -> PathBuf {
        // Base is `.rayzor/blade/cache` (separate from the Rust target folder),
        // or an explicit `--cache-dir`. Either way the per-target subdir keeps
        // native and wasm artifacts from colliding.
        let base = self
            .cache_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(".rayzor/blade/cache"));
        let dir = base.join(self.cache_discriminator());

        // Try to create it if it doesn't exist
        if !dir.exists() {
            let _ = std::fs::create_dir_all(&dir);
        }

        dir
    }

    /// Get the target directory for the given profile
    pub fn get_target_dir(profile: &str) -> PathBuf {
        let triple = Self::get_target_triple();
        PathBuf::from("target").join(triple).join(profile)
    }

    /// Get the build directory for intermediate artifacts
    pub fn get_build_dir(profile: &str) -> PathBuf {
        Self::get_target_dir(profile).join("build")
    }

    /// Get the cache directory for a specific profile
    pub fn get_profile_cache_dir(profile: &str) -> PathBuf {
        Self::get_target_dir(profile).join("cache")
    }

    /// Get the output directory for executables
    pub fn get_output_dir(profile: &str) -> PathBuf {
        Self::get_target_dir(profile)
    }

    /// Get the cache file path for a given source file
    pub fn get_cache_path(&self, source_path: &Path) -> PathBuf {
        let cache_dir = self.get_cache_dir();

        // Create a cache filename based on the source path
        // Convert path to a safe filename by replacing separators with underscores
        let source_str = source_path.to_string_lossy();
        let cache_name = source_str
            .replace(['/', '\\', ':'], "_")
            .replace(".hx", ".blade");

        cache_dir.join(cache_name)
    }

    /// Create a fast compilation config optimized for interpreter cold start
    ///
    /// This configuration prioritizes startup speed over type safety:
    /// - Lazy stdlib loading (symbols loaded on-demand)
    /// - Cache enabled for subsequent runs
    ///
    /// Ideal for REPL, development mode, and interpreted execution.
    pub fn fast() -> Self {
        Self {
            lazy_stdlib: true,
            ..Default::default()
        }
    }

    /// Create a strict compilation config with full type checking
    ///
    /// This is the default behavior - all symbols loaded upfront,
    /// full type analysis enabled.
    pub fn strict() -> Self {
        Self {
            lazy_stdlib: false,
            ..Default::default()
        }
    }
}

impl CompilationUnit {
    /// Create a new compilation unit with the given configuration
    pub fn new(config: CompilationConfig) -> Self {
        let string_interner = StringInterner::new();
        let namespace_resolver = NamespaceResolver::new();
        let import_resolver = ImportResolver::new();

        // Create pipeline with config
        let pipeline = HaxeCompilationPipeline::with_config(config.pipeline_config.clone());

        Self {
            stdlib_files: Vec::new(),
            import_hx_files: Vec::new(),
            user_files: Vec::new(),
            stdlib_manifest_loaded: false,
            macro_expansions: Vec::new(),
            string_interner,
            symbol_table: SymbolTable::new(),
            type_table: Rc::new(RefCell::new(TypeTable::new())),
            scope_tree: ScopeTree::new(ScopeId::from_raw(0)),
            namespace_resolver,
            import_resolver,
            config,
            failed_type_loads: BTreeSet::new(),
            static_sig_index: Rc::new(RefCell::new(
                crate::tast::sig_index::StaticSigIndex::default(),
            )),
            compiled_files: BTreeMap::new(),
            pipeline,
            mir_modules: Vec::new(),
            import_mir_modules: Vec::new(),
            import_own_func_ids: std::collections::BTreeSet::new(),
            loaded_stdlib_typed_files: Vec::new(),
            loaded_import_haxe_files: Vec::new(),
            next_file_id: 0,
            file_id_by_filename: BTreeMap::new(),
            file_source_by_filename: BTreeMap::new(),
            collected_diagnostics: Vec::new(),
            global_class_fields: BTreeMap::new(),
            stdlib_function_map: BTreeMap::new(),
            stdlib_function_name_map: BTreeMap::new(),
            import_field_index_map: BTreeMap::new(),
            last_import_errors: BTreeMap::new(),
            import_field_class_names: BTreeMap::new(),
            import_property_access_map: BTreeMap::new(),
            import_constructor_name_map: BTreeMap::new(),
            import_function_param_iface_names: BTreeMap::new(),
            import_constructor_param_counts: BTreeMap::new(),
            import_function_param_types: BTreeMap::new(),
            import_external_globals: BTreeMap::new(),
            import_class_alloc_sizes: BTreeMap::new(),
            import_class_alloc_sizes_by_name: BTreeMap::new(),
            import_class_type_to_symbol: BTreeMap::new(),
            import_class_method_symbols: BTreeMap::new(),
            import_interface_method_names: BTreeMap::new(),
            import_interface_method_return_types: BTreeMap::new(),
            import_interface_extends: BTreeMap::new(),
            import_interface_vtables: BTreeMap::new(),
            compiler_plugin_registry: CompilerPluginRegistry::new(),
            hdll_symbols: Vec::new(),
            loaded_hdlls: BTreeSet::new(),
            global_inline_vars: BTreeMap::new(),
            last_compiled_type_info: None,
            last_compiled_cached_maps: None,
            last_compiled_own_func_ids: None,
            cached_stdlib_mir: None,
            extern_js_module_map: BTreeMap::new(),
            qualified_method_map: BTreeMap::new(),
            typecheck_timings: TypecheckStageTimings::default(),
        }
    }

    /// Build the preprocessor config with extra_defines from compilation config.
    /// All file parsing should use this to ensure #if wasm etc. work consistently.
    fn preprocessor_config(&self) -> parser::preprocessor::PreprocessorConfig {
        let mut config = parser::preprocessor::PreprocessorConfig::default();
        // rayzor is a native/JIT target — masquerade as Haxe's `eval` (macro
        // interpreter) target for stdlib conditional compilation. `eval` is the
        // closest existing Haxe target to rayzor's model, and selecting it makes
        // the stdlib pick its native extern branches (`#if (flash||cpp||eval)`,
        // `#if (neko||eval)`) while skipping the JS/Flash dynamic-init branches
        // (`#if !eval`). Without it, e.g. Math.hx compiles its
        // `Math.NaN = Number["NaN"]` JS bootstrap, which references the JS global
        // `Number` and fails ("Cannot find name 'Number'") — Math then never
        // registers, starving FPHelper/Int32/StringTools and any imported module
        // that reads Math.POSITIVE_INFINITY or calls StringTools.*.
        if std::env::var_os("RAYZOR_NO_EVAL_DEFINE").is_none() {
            config.defines.insert("eval".to_string());
        }
        // Every Haxe target also defines its own name (`#if js`, `#if cpp`, …).
        // rayzor is not `eval` — the masquerade above only picks stdlib branches
        // — so stdlib and user code need a way to say "this is rayzor
        // specifically", e.g. `Single`, which the shared `#if` guards otherwise
        // exclude because rayzor is in none of their target lists.
        config.defines.insert("rayzor".to_string());
        for define in &self.config.extra_defines {
            config.defines.insert(define.clone());
        }
        config
    }

    /// Parse a file with the compilation unit's preprocessor defines.
    fn parse_file(&self, filename: &str, source: &str) -> Result<parser::HaxeFile, String> {
        let config = self.preprocessor_config();
        let file = parser::haxe_parser::parse_haxe_file_with_config(
            filename, source, true, true, &config,
        )?;
        // Every canonical parse feeds the static-signature index so call
        // sites can type not-yet-lowered statics from their declarations.
        {
            let mut index = self.static_sig_index.borrow_mut();
            index.preprocessor = config;
            index.index_file(&file);
        }
        Ok(file)
    }

    /// Load standard library files
    /// This should be called FIRST, before any user files are added
    pub fn load_stdlib(&mut self) -> Result<(), String> {
        if !self.config.load_stdlib {
            return Ok(());
        }

        // Configure stdlib loader
        let mut loader_config = StdLibConfig::default();
        loader_config.std_paths = self.config.stdlib_paths.clone();
        loader_config.default_imports = self.config.default_stdlib_imports.clone();

        let mut loader = StdLibLoader::new(loader_config);
        loader.set_preprocessor_config(self.preprocessor_config());

        // Configure namespace resolver with stdlib paths for on-demand loading
        self.namespace_resolver
            .set_stdlib_paths(self.config.stdlib_paths.clone());

        // Load pre-compiled symbols from BLADE manifest if caching is enabled
        // Skip if lazy_stdlib is enabled (for faster cold start)
        let bsym_loaded = if self.config.enable_cache && !self.config.lazy_stdlib {
            if self.load_stdlib_symbols() {
                debug!("BLADE symbols loaded, stdlib configured for cached resolution");
                true
            } else {
                debug!("No BLADE symbols available, falling back to on-demand loading");
                false
            }
        } else if self.config.lazy_stdlib {
            debug!("Lazy stdlib enabled - skipping upfront symbol registration for faster startup");
            self.register_builtin_globals();
            false
        } else {
            false
        };
        self.stdlib_manifest_loaded = bsym_loaded;

        // Load default stdlib imports (StdTypes, etc.) only when we have not
        // already registered them from the BLADE symbol manifest. StdTypes are
        // top-level compiler-visible types, not a user import; parsing the file
        // here on the bsym path was cold-start work with no semantic payload.
        if bsym_loaded {
            debug!("BLADE symbols loaded; skipping default stdlib source parse");
        } else {
            let default_files = loader.load_default_imports();
            for file in default_files {
                if let Some(source) = file.input.as_deref() {
                    self.pre_register_and_enums_from_source(&file.filename, source)?;
                }
                self.stdlib_files.push(file);
            }
            debug!("Loaded {} default stdlib imports", self.stdlib_files.len());
        }

        Ok(())
    }

    /// Set source paths for user code (for on-demand import loading)
    /// These paths are checked first when resolving imports
    pub fn set_source_paths(&mut self, paths: Vec<PathBuf>) {
        self.namespace_resolver.set_source_paths(paths);
    }

    // === BLADE Caching Methods ===

    /// Get the BLADE cache path for a source file.
    ///
    /// The cache filename is the module's fully-qualified name with `.blade`
    /// appended (e.g. `nue.transformer.GQAttention.blade`). Two strategies
    /// are tried in order:
    ///
    /// 1. **Known root strip** — `haxe-std/` or `/src/` in the path. Covers
    ///    bundled stdlib and the common `src/`-rooted layout.
    ///
    /// 2. **Project class-path strip** — match the source against each
    ///    configured source path (from rayzor.toml `class-paths`) and strip
    ///    the longest matching prefix. Without this, projects whose Haxe
    ///    sources don't live under `src/` (e.g. nue's `class-paths = ["."]`
    ///    with files at `nue/nue/transformer/GQAttention.hx`) cache by
    ///    bare filename only — `GQAttention.blade` — and short-name
    ///    collisions between packages silently overwrite each other.
    ///
    /// Falls back to the filename only if nothing matches; that path is the
    /// least desirable because it loses the package and risks collisions.
    fn blade_cache_path(&self, source_path: &str) -> Option<PathBuf> {
        let cache_dir = self.config.get_cache_dir();
        let normalized = source_path.replace('\\', "/");

        let module_part: String = if let Some(pos) = normalized.rfind("haxe-std/") {
            normalized[pos + 9..].to_string()
        } else if let Some(pos) = normalized.rfind("/src/") {
            normalized[pos + 5..].to_string()
        } else {
            // Try stripping a project class-path (rayzor.toml `class-paths`).
            // Pick the longest matching prefix so nested roots resolve in
            // favour of the more-specific one.
            let stripped = {
                let abs = std::path::Path::new(&normalized)
                    .canonicalize()
                    .ok()
                    .and_then(|p| p.to_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| normalized.clone());
                let mut best: Option<String> = None;
                for root in self.namespace_resolver.get_source_paths() {
                    let root_str = root
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(|s| s.to_string()))
                        .unwrap_or_else(|| root.to_string_lossy().to_string());
                    let root_with_slash = if root_str.ends_with('/') {
                        root_str.clone()
                    } else {
                        format!("{root_str}/")
                    };
                    if abs.starts_with(&root_with_slash) {
                        let candidate = abs[root_with_slash.len()..].to_string();
                        if best
                            .as_ref()
                            .map(|b| candidate.len() < b.len())
                            .unwrap_or(true)
                        {
                            best = Some(candidate);
                        }
                    }
                }
                best
            };
            stripped.unwrap_or_else(|| {
                normalized
                    .rsplit('/')
                    .next()
                    .unwrap_or(&normalized)
                    .to_string()
            })
        };

        let module_name = module_part.replace('/', ".").replace(".hx", "");

        if module_name.is_empty() {
            return None;
        }

        Some(cache_dir.join(format!("{}.blade", module_name)))
    }

    /// Compute the BLADE source fingerprint for this compilation context.
    ///
    /// Source text alone is not enough: `extra_defines` changes `#if`
    /// lowering, so native and wasm builds of the same file must not share
    /// the same per-module artifact.
    /// Identity hash of the USER PROGRAM being compiled (all user source
    /// files: filename + content). A `.blade` module cache stores PROGRAM-
    /// SPECIFIC state — the global function-id renumbering, class memory
    /// layout, GENERATED reflection ctor wrappers, and inherited-field tables
    /// are all assigned relative to the full module set of the program that
    /// produced it. Folding this into every module's cache key (below) makes
    /// a cached module reusable ONLY for the same program: re-running the same
    /// program hits; any edit to a user source file (or a different program
    /// entirely, e.g. the next test in the suite) misses and recompiles.
    fn user_program_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        "rayzor-program-identity-v1".hash(&mut h);
        for f in &self.user_files {
            f.filename.hash(&mut h);
            if let Some(src) = &f.input {
                src.hash(&mut h);
            }
        }
        h.finish()
    }

    fn hash_source_for_config(&self, source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        "rayzor-blade-source-v4".hash(&mut hasher);
        source.hash(&mut hasher);
        let mut defines = self.config.extra_defines.clone();
        defines.sort();
        defines.hash(&mut hasher);
        // F6 cache-coherence: a cached module carries state assigned relative
        // to the WHOLE program (id renumbering, class layout, reflection ctor
        // wrappers, inherited fields). Two different programs that both import
        // this module previously SHARED its cache and reused that stale state
        // — surfacing as `Cannot find name 'root'` (inherited field on a fresh
        // subclass of a cached parent), the `__reflect_ctor_wrap` W0020
        // (Cast source undefined), or a load SIGSEGV. Key on the program so the
        // cache is only reused when it is genuinely valid (same program, no
        // user-source edits). Sound; trades cross-program incremental reuse for
        // correctness (cache-on previously required a manual `.rayzor` scrub).
        self.user_program_hash().hash(&mut hasher);
        // Fold in a content hash of the TRANSITIVE import set. The cache key was
        // previously the entry file's own bytes + defines only, so editing a
        // DEPENDENCY (a `.hx` imported by this entry, directly or transitively)
        // left the key unchanged: the stale MIR — e.g. an interface vtable from
        // before a method was added — validated as current and was reused,
        // segfaulting at load. `compute_import_set_hash` walks the import graph
        // and hashes each resolvable file's bytes; it is intentionally
        // permissive (imports it can't resolve simply don't contribute).
        let import_hash = crate::ir::blade::compute_import_set_hash(
            source,
            self.namespace_resolver.get_source_paths(),
        );
        import_hash.hash(&mut hasher);
        hasher.finish()
    }

    /// Try to load a cached MIR module from BLADE cache
    /// Returns Some(IrModule) if cache is valid, None otherwise
    fn try_load_blade_cached(&self, source_path: &str, source: &str) -> Option<IrModule> {
        if !self.config.enable_cache {
            return None;
        }

        let blade_path = self.blade_cache_path(source_path)?;
        if !blade_path.exists() {
            trace!("[BLADE] Cache miss (no file): {}", source_path);
            return None;
        }

        match load_blade(&blade_path) {
            Ok((mir, metadata, _symbols, _cached_maps)) => {
                // Validate cache by checking source hash AND compiler cache
                // ABI id — see save_to_cache / matching check at the other
                // load site for why both are required.
                let current_hash = self.hash_source_for_config(source);
                let current_build_id = env!("RAYZOR_BUILD_ID");
                if metadata.source_hash != current_hash {
                    trace!("[BLADE] Cache stale (hash mismatch): {}", source_path);
                    None
                } else if metadata.build_id != current_build_id {
                    trace!("[BLADE] Cache stale (build-id mismatch): {}", source_path);
                    None
                } else {
                    debug!(
                        "[BLADE] Cache hit: {} -> {}",
                        source_path,
                        blade_path.display()
                    );
                    Some(mir)
                }
            }
            Err(e) => {
                trace!("[BLADE] Cache read error for {}: {}", source_path, e);
                None
            }
        }
    }

    /// Save a MIR module to BLADE cache with optional type info and cross-reference maps
    fn save_blade_cached(
        &self,
        source_path: &str,
        source: &str,
        mir: &IrModule,
        dependencies: Vec<String>,
        symbols: Option<BladeTypeInfo>,
        cached_maps: Option<BladeCachedMaps>,
    ) {
        if !self.config.enable_cache {
            return;
        }

        let blade_path = match self.blade_cache_path(source_path) {
            Some(p) => p,
            None => return,
        };

        // Ensure cache directory exists
        if let Some(parent) = blade_path.parent() {
            if !parent.exists() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    trace!("[BLADE] Failed to create cache dir: {}", e);
                    return;
                }
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let metadata = BladeMetadata {
            name: mir.name.clone(),
            source_path: source_path.to_string(),
            source_hash: self.hash_source_for_config(source),
            source_timestamp: now, // We use hash for validation, not timestamp
            compile_timestamp: now,
            dependencies,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            build_id: env!("RAYZOR_BUILD_ID").to_string(),
        };

        // Compute per-function granular invalidation hashes (§3.2)
        let cached_maps = cached_maps.map(|mut maps| {
            for entry in &mut maps.functions {
                if let Some(func) = mir
                    .functions
                    .values()
                    .find(|f| f.name.ends_with(&entry.method_name))
                {
                    entry.signature_hash = crate::ir::blade::compute_signature_hash(func);
                    entry.body_hash = crate::ir::blade::compute_body_hash(func);
                }
            }
            maps
        });

        match save_blade_with_state(&blade_path, mir, metadata, symbols, cached_maps) {
            Ok(()) => {
                debug!(
                    "[BLADE] Cached: {} -> {}",
                    source_path,
                    blade_path.display()
                );
            }
            Err(e) => {
                trace!("[BLADE] Failed to cache {}: {}", source_path, e);
            }
        }
    }

    /// Build name-keyed cached maps from MIR lowering result for BLADE cache storage.
    /// Converts SymbolId/TypeId-keyed maps to name-keyed maps that survive across compilations.
    fn build_cached_maps_from_mir_result(
        &self,
        function_map: &BTreeMap<crate::tast::SymbolId, crate::ir::IrFunctionId>,
        field_index_map: &BTreeMap<crate::tast::SymbolId, (crate::tast::TypeId, u32)>,
        constructor_name_map: &BTreeMap<String, crate::ir::IrFunctionId>,
        class_alloc_sizes: &BTreeMap<crate::tast::SymbolId, u64>,
        field_class_names: &BTreeMap<crate::tast::SymbolId, String>,
        property_access_map: &BTreeMap<crate::tast::SymbolId, crate::tast::PropertyAccessInfo>,
        function_param_hir_types: &BTreeMap<crate::ir::IrFunctionId, Vec<crate::tast::TypeId>>,
        interface_vtables: &BTreeMap<
            (crate::tast::SymbolId, crate::tast::SymbolId),
            Vec<crate::tast::SymbolId>,
        >,
        interface_method_names: &BTreeMap<crate::tast::SymbolId, Vec<crate::tast::InternedString>>,
        interface_method_return_types: &BTreeMap<
            (crate::tast::SymbolId, crate::tast::InternedString),
            crate::tast::TypeId,
        >,
        interface_extends: &BTreeMap<crate::tast::SymbolId, Vec<crate::tast::SymbolId>>,
    ) -> BladeCachedMaps {
        let mut functions = Vec::new();
        let mut fields = Vec::new();
        let mut class_sizes = Vec::new();

        // Resolve a HIR TypeId to a qualified-name string for the cache.
        // Only Class/Interface types matter to Path 3 of
        // `maybe_materialize_for_call`; everything else (primitives,
        // abstracts, anonymous, …) we encode as None so the restore
        // side leaves the corresponding param-type slot unwrapped.
        let resolve_hir_type_name = |ty: crate::tast::TypeId| -> Option<String> {
            let type_table = self.type_table.borrow();
            let info = type_table.get(ty)?;
            let symbol_id = match &info.kind {
                crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                crate::tast::TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                _ => None,
            }?;
            let sym = self.symbol_table.get_symbol(symbol_id)?;
            sym.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(sym.name))
                .map(|s| s.to_string())
        };
        let param_names_for = |func_id: crate::ir::IrFunctionId| -> Vec<Option<String>> {
            function_param_hir_types
                .get(&func_id)
                .map(|tys| tys.iter().copied().map(resolve_hir_type_name).collect())
                .unwrap_or_default()
        };

        // Convert function_map: SymbolId → IrFunctionId to (class_name, method_name, func_id)
        for (symbol_id, func_id) in function_map {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                // Constructors are stored separately in constructor_name_map.
                // Skip non-function symbols here (e.g. class symbols used as ctor keys),
                // otherwise cache restore will try to resolve bogus method names like `Exception`.
                if !matches!(sym.kind, crate::tast::SymbolKind::Function) {
                    continue;
                }

                let method_name = self
                    .string_interner
                    .get(sym.name)
                    .unwrap_or("<unknown>")
                    .to_string();

                // Find the class this method belongs to by looking at its scope
                let class_name = self.find_class_name_for_scope(sym.scope_id);

                functions.push(BladeFuncEntry {
                    class_name: class_name.unwrap_or_default(),
                    method_name,
                    func_id: func_id.0,
                    is_constructor: false,
                    signature_hash: 0, // computed at save time from MIR
                    body_hash: 0,
                    param_type_names: param_names_for(*func_id),
                });
            }
        }

        // Add constructors from constructor_name_map (already name-keyed)
        for (class_name, func_id) in constructor_name_map {
            functions.push(BladeFuncEntry {
                class_name: class_name.clone(),
                method_name: "new".to_string(),
                func_id: func_id.0,
                is_constructor: true,
                signature_hash: 0,
                body_hash: 0,
                param_type_names: param_names_for(*func_id),
            });
        }

        // Convert field_index_map: SymbolId → (TypeId, field_index) to (class_name, field_name, field_index)
        for (symbol_id, (_type_id, field_index)) in field_index_map {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                let field_name = self
                    .string_interner
                    .get(sym.name)
                    .unwrap_or("<unknown>")
                    .to_string();

                // Use field_class_names from MIR context (populated during register_class_metadata)
                // Fall back to accumulated import names for fields inherited from dependencies
                let class_name = field_class_names
                    .get(symbol_id)
                    .cloned()
                    .or_else(|| self.import_field_class_names.get(symbol_id).cloned())
                    .or_else(|| self.find_class_name_for_scope(sym.scope_id));

                fields.push(BladeFieldEntry {
                    class_name: class_name.unwrap_or_default(),
                    field_name,
                    field_index: *field_index,
                });
            }
        }

        // Convert class_alloc_sizes: SymbolId → u64 to (class_name, size)
        for (symbol_id, size) in class_alloc_sizes {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                let name = sym
                    .qualified_name
                    .and_then(|n| self.string_interner.get(n))
                    .or_else(|| self.string_interner.get(sym.name))
                    .unwrap_or("<unknown>")
                    .to_string();
                class_sizes.push((name, *size));
            }
        }

        // Convert property_access_map: SymbolId → PropertyAccessInfo to (class_name, field_name, getter, setter)
        let mut properties = Vec::new();
        for (symbol_id, prop_info) in property_access_map {
            if let Some(sym) = self.symbol_table.get_symbol(*symbol_id) {
                let field_name = self
                    .string_interner
                    .get(sym.name)
                    .unwrap_or("<unknown>")
                    .to_string();
                let class_name = field_class_names
                    .get(symbol_id)
                    .cloned()
                    .or_else(|| self.import_field_class_names.get(symbol_id).cloned())
                    .or_else(|| self.find_class_name_for_scope(sym.scope_id));
                let to_blade = |acc: &crate::tast::PropertyAccessor| -> BladeAccessor {
                    match acc {
                        crate::tast::PropertyAccessor::Default => BladeAccessor::Default,
                        crate::tast::PropertyAccessor::Null => BladeAccessor::Null,
                        crate::tast::PropertyAccessor::Never => BladeAccessor::Never,
                        crate::tast::PropertyAccessor::Dynamic => BladeAccessor::Dynamic,
                        crate::tast::PropertyAccessor::Method(n) => BladeAccessor::Method(
                            self.string_interner.get(*n).unwrap_or("").to_string(),
                        ),
                    }
                };
                // Skip orphan entries with no resolvable owning class — they
                // can't be looked up correctly on restore (the load side keys
                // by class_name) and just pollute the merged map. The
                // name-based fallback in lower_field_access used to surface
                // them first and shadow the real property (e.g. an empty
                // `length` from ArrayIterator stole the StringBuf.length
                // dispatch after a cross-test cache load).
                let Some(class_name) = class_name else {
                    continue;
                };
                properties.push(BladePropertyEntry {
                    class_name,
                    field_name,
                    getter: to_blade(&prop_info.getter),
                    setter: to_blade(&prop_info.setter),
                });
            }
        }

        // Convert interface_vtables: (class_sym, iface_sym) → Vec<method_sym>
        // into qualified-name triples so the SymbolIds survive
        // re-numbering on restore.
        let mut iface_vtable_entries: Vec<crate::ir::blade::BladeIfaceVtableEntry> = Vec::new();
        let qname_of = |sid: crate::tast::SymbolId| -> Option<String> {
            let sym = self.symbol_table.get_symbol(sid)?;
            sym.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(sym.name))
                .map(|s| s.to_string())
        };
        for ((class_sym, iface_sym), methods) in interface_vtables {
            let Some(class_name) = qname_of(*class_sym) else {
                continue;
            };
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let method_qnames: Vec<String> = methods.iter().filter_map(|m| qname_of(*m)).collect();
            // Skip entries that lost their methods on resolution — they
            // can't be reconstructed deterministically on restore.
            if method_qnames.len() != methods.len() {
                continue;
            }
            iface_vtable_entries.push(crate::ir::blade::BladeIfaceVtableEntry {
                class_name,
                iface_name,
                method_qnames,
            });
        }

        // Convert interface_method_names: iface_sym → Vec<InternedString>
        // into qname-keyed entries. `qname_of` from the iface_vtables
        // block above is closed over here too.
        //
        // Slot-alignment discipline: `maybe_materialize_for_call`'s
        // vtable-slot math indexes by position into this Vec. A silently
        // empty string from a failed interner lookup would shift every
        // following slot index by one and misroute dispatch. Mirror
        // c7a170d's all-or-nothing skip — drop the whole entry if ANY
        // method name fails to intern, rather than ship a slot-misaligned
        // Vec. Same discipline as `interface_extends` below.
        let mut iface_method_names_entries: Vec<crate::ir::blade::BladeInterfaceMethodNamesEntry> =
            Vec::new();
        for (iface_sym, method_names) in interface_method_names {
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let method_names_str: Vec<String> = method_names
                .iter()
                .filter_map(|n| self.string_interner.get(*n).map(|s| s.to_string()))
                .collect();
            if method_names_str.len() != method_names.len() {
                continue;
            }
            iface_method_names_entries.push(crate::ir::blade::BladeInterfaceMethodNamesEntry {
                iface_name,
                method_names: method_names_str,
            });
        }

        // Convert interface_method_return_types: (iface_sym,
        // method_name) → TypeId into qname-keyed entries. Drop entries
        // whose return type isn't a Class/Interface (those are
        // recoverable from MIR signature on the consumer side).
        let mut iface_method_return_type_entries: Vec<
            crate::ir::blade::BladeInterfaceMethodReturnTypeEntry,
        > = Vec::new();
        for ((iface_sym, method_name), ty) in interface_method_return_types {
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let Some(return_type_name) = resolve_hir_type_name(*ty) else {
                continue;
            };
            let method_name_str = self
                .string_interner
                .get(*method_name)
                .unwrap_or("")
                .to_string();
            if method_name_str.is_empty() {
                continue;
            }
            iface_method_return_type_entries.push(
                crate::ir::blade::BladeInterfaceMethodReturnTypeEntry {
                    iface_name,
                    method_name: method_name_str,
                    return_type_name,
                },
            );
        }

        // Convert interface_extends: iface_sym → Vec<parent_sym> into
        // qname-keyed entries. Skip entries where any parent qname
        // fails to resolve so partial-restore doesn't silently lose
        // the rest of the chain.
        let mut iface_extends_entries: Vec<crate::ir::blade::BladeInterfaceExtendsEntry> =
            Vec::new();
        for (iface_sym, parents) in interface_extends {
            let Some(iface_name) = qname_of(*iface_sym) else {
                continue;
            };
            let parent_names: Vec<String> = parents.iter().filter_map(|p| qname_of(*p)).collect();
            if parent_names.len() != parents.len() {
                continue;
            }
            iface_extends_entries.push(crate::ir::blade::BladeInterfaceExtendsEntry {
                iface_name,
                parent_names,
            });
        }

        BladeCachedMaps {
            functions,
            fields,
            class_sizes,
            properties,
            inline_vars: Vec::new(), // populated separately in try_compile_import
            interface_vtables: iface_vtable_entries,
            interface_method_names: iface_method_names_entries,
            interface_method_return_types: iface_method_return_type_entries,
            interface_extends: iface_extends_entries,
        }
    }

    /// Find the qualified class name that owns a given scope.
    /// Used to convert scope-based symbol lookups to name-based keys for cache.
    fn find_class_name_for_scope(&self, scope_id: ScopeId) -> Option<String> {
        // Search all symbols for a class whose scope_id matches
        // Class symbols have their scope_id set to the class member scope
        for i in 0..self.symbol_table.len() {
            let sym_id = crate::tast::SymbolId::from_raw(i as u32);
            if let Some(sym) = self.symbol_table.get_symbol(sym_id) {
                if matches!(sym.kind, crate::tast::SymbolKind::Class) && sym.scope_id == scope_id {
                    return sym
                        .qualified_name
                        .and_then(|n| self.string_interner.get(n))
                        .or_else(|| self.string_interner.get(sym.name))
                        .map(|s| s.to_string());
                }
            }
        }
        None
    }

    /// Extract static inline var constants from a TypedFile for BLADE cache storage.
    fn extract_inline_vars_from_typed_file(
        typed_file: &TypedFile,
        symbol_table: &crate::tast::SymbolTable,
        string_interner: &crate::tast::StringInterner,
    ) -> Vec<crate::ir::blade::BladeInlineVarEntry> {
        use crate::ir::blade::{BladeInlineValue, BladeInlineVarEntry};
        use crate::tast::node::LiteralValue;

        let mut entries = Vec::new();

        for class in &typed_file.classes {
            let class_name = symbol_table
                .get_symbol(class.symbol_id)
                .and_then(|sym| {
                    sym.qualified_name
                        .and_then(|n| string_interner.get(n))
                        .or_else(|| string_interner.get(sym.name))
                })
                .unwrap_or("")
                .to_string();
            if class_name.is_empty() {
                continue;
            }

            for field in &class.fields {
                if !field.is_static {
                    continue;
                }
                // Only inline/final fields
                let is_inline = symbol_table
                    .get_symbol(field.symbol_id)
                    .map(|s| s.is_inline())
                    .unwrap_or(false);
                if !is_inline && field.mutability == crate::tast::symbols::Mutability::Mutable {
                    continue;
                }
                let Some(init) = &field.initializer else {
                    continue;
                };

                let field_name = string_interner
                    .get(
                        symbol_table
                            .get_symbol(field.symbol_id)
                            .map(|s| s.name)
                            .unwrap_or(unsafe { crate::tast::InternedString::from_raw(0) }),
                    )
                    .unwrap_or("")
                    .to_string();

                // Try to evaluate the initializer to a constant
                let value = match &init.kind {
                    crate::tast::node::TypedExpressionKind::Literal { value: lit } => match lit {
                        LiteralValue::Int(v) => Some(BladeInlineValue::Int(*v)),
                        LiteralValue::Float(v) => Some(BladeInlineValue::Float(*v)),
                        LiteralValue::Bool(v) => Some(BladeInlineValue::Bool(*v)),
                        LiteralValue::String(v) => Some(BladeInlineValue::String(v.clone())),
                        _ => None,
                    },
                    _ => None,
                };

                if let Some(value) = value {
                    entries.push(BladeInlineVarEntry {
                        class_name: class_name.clone(),
                        field_name,
                        value,
                    });
                }
            }
        }

        // Also handle abstract fields (enum abstract constants)
        for abs in &typed_file.abstracts {
            let abs_name = symbol_table
                .get_symbol(abs.symbol_id)
                .and_then(|sym| {
                    sym.qualified_name
                        .and_then(|n| string_interner.get(n))
                        .or_else(|| string_interner.get(sym.name))
                })
                .unwrap_or("")
                .to_string();
            if abs_name.is_empty() {
                continue;
            }

            for field in &abs.fields {
                if !field.is_static {
                    continue;
                }
                let Some(init) = &field.initializer else {
                    continue;
                };
                let field_name = string_interner
                    .get(
                        symbol_table
                            .get_symbol(field.symbol_id)
                            .map(|s| s.name)
                            .unwrap_or(unsafe { crate::tast::InternedString::from_raw(0) }),
                    )
                    .unwrap_or("")
                    .to_string();

                let value = match &init.kind {
                    crate::tast::node::TypedExpressionKind::Literal { value: lit } => match lit {
                        LiteralValue::Int(v) => Some(BladeInlineValue::Int(*v)),
                        LiteralValue::Float(v) => Some(BladeInlineValue::Float(*v)),
                        LiteralValue::Bool(v) => Some(BladeInlineValue::Bool(*v)),
                        LiteralValue::String(v) => Some(BladeInlineValue::String(v.clone())),
                        _ => None,
                    },
                    _ => None,
                };

                if let Some(value) = value {
                    entries.push(BladeInlineVarEntry {
                        class_name: abs_name.clone(),
                        field_name,
                        value,
                    });
                }
            }
        }

        entries
    }

    /// Store inline vars from BladeInlineVarEntry into the global map.
    fn store_inline_vars(&mut self, entries: &[crate::ir::blade::BladeInlineVarEntry]) {
        for entry in entries {
            let key = format!("{}.{}", entry.class_name, entry.field_name);
            self.global_inline_vars.insert(key, entry.value.clone());
        }
    }

    // === BLADE Symbol Loading Methods ===

    /// Load pre-compiled stdlib symbols from .bsym manifest
    /// Returns true if symbols were loaded successfully
    pub fn load_stdlib_symbols(&mut self) -> bool {
        let manifest_path = PathBuf::from(".rayzor/blade/stdlib/stdlib.bsym");
        // Only a manifest generated for THIS stdlib may stand in for parsing it.
        // A copy compiled into the binary cannot: nothing regenerates the asset
        // when haxe-std changes, and the loader validates only magic and format
        // version, so the symbols it registers can describe a different stdlib
        // than the one being compiled against.
        if !manifest_path.exists() {
            debug!(
                "[BLADE] No symbol manifest at {}; parsing stdlib sources",
                manifest_path.display()
            );
            return false;
        }
        let manifest = load_symbol_manifest(&manifest_path);

        match manifest {
            Ok(manifest) => {
                info!(
                    "[BLADE] Loading {} modules from symbol manifest",
                    manifest.modules.len()
                );
                self.register_symbols_from_manifest(&manifest);
                // Also register builtin globals like 'trace' that aren't in the manifest
                self.register_builtin_globals();
                true
            }
            Err(e) => {
                debug!("[BLADE] Failed to load symbol manifest: {}", e);
                false
            }
        }
    }

    /// Register built-in global symbols like 'trace' that aren't in the BLADE manifest
    fn register_builtin_globals(&mut self) {
        use crate::tast::{
            LifetimeId, Mutability, SourceLocation, Symbol, SymbolFlags, SymbolKind, Visibility,
        };

        // Register built-in global functions
        let builtin_functions = [
            ("trace", vec!["Dynamic"], "Void"), // trace(value: Dynamic): Void
        ];

        for (func_name, param_types, return_type_str) in builtin_functions {
            let func_name_interned = self.string_interner.intern(func_name);

            // Create parameter types
            let param_type_ids: Vec<TypeId> = param_types
                .iter()
                .map(|param_type_name| match *param_type_name {
                    "Dynamic" => self.type_table.borrow().dynamic_type(),
                    "Int" => self.type_table.borrow().int_type(),
                    "String" => self.type_table.borrow().string_type(),
                    "Float" => self.type_table.borrow().float_type(),
                    "Bool" => self.type_table.borrow().bool_type(),
                    "Void" => self.type_table.borrow().void_type(),
                    _ => self.type_table.borrow().dynamic_type(),
                })
                .collect();

            // Create return type
            let return_type_id = match return_type_str {
                "Dynamic" => self.type_table.borrow().dynamic_type(),
                "Int" => self.type_table.borrow().int_type(),
                "String" => self.type_table.borrow().string_type(),
                "Float" => self.type_table.borrow().float_type(),
                "Bool" => self.type_table.borrow().bool_type(),
                "Void" => self.type_table.borrow().void_type(),
                _ => self.type_table.borrow().dynamic_type(),
            };

            // Create function type
            let function_type_id = self
                .type_table
                .borrow_mut()
                .create_function_type(param_type_ids, return_type_id);

            // Create function symbol
            let func_symbol_id = SymbolId::from_raw(self.symbol_table.len() as u32);
            let func_symbol = Symbol {
                id: func_symbol_id,
                name: func_name_interned,
                kind: SymbolKind::Function,
                type_id: function_type_id,
                scope_id: ScopeId::first(),
                lifetime_id: LifetimeId::invalid(),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                definition_location: SourceLocation::unknown(),
                is_used: false,
                is_exported: false,
                documentation: None,
                flags: SymbolFlags::NONE,
                package_id: None,
                qualified_name: None,
                native_name: None,
                frameworks: None,
                c_includes: None,
                c_sources: None,
                c_libs: None,
                js_import: None,
            };

            // Add symbol to symbol table
            self.symbol_table.add_symbol(func_symbol);

            // Add to root scope for global resolution
            if let Some(scope) = self.scope_tree.get_scope_mut(ScopeId::first()) {
                scope.add_symbol(func_symbol_id, func_name_interned);
            }

            trace!("[BLADE] Registered builtin: {}", func_name);
        }
    }

    /// Register all symbols from a loaded manifest
    fn register_symbols_from_manifest(&mut self, manifest: &BladeSymbolManifest) {
        let mut total_classes = 0;
        let mut total_enums = 0;
        let mut total_aliases = 0;
        let mut total_abstracts = 0;
        let mut total_methods = 0;

        for module in &manifest.modules {
            // Mark this file as "loaded" so load_import_file_recursive will skip it
            // This prevents redundant re-parsing of files whose symbols are already cached
            let source_path = PathBuf::from(&module.source_path);
            self.namespace_resolver.mark_file_loaded(source_path);

            for class_info in &module.types.classes {
                let method_count = class_info.methods.len() + class_info.static_methods.len();
                self.register_class_from_blade(class_info);
                total_classes += 1;
                total_methods += method_count;
            }
            for enum_info in &module.types.enums {
                self.register_enum_from_blade(enum_info);
                total_enums += 1;
            }
            for alias_info in &module.types.type_aliases {
                self.register_type_alias_from_blade(alias_info);
                total_aliases += 1;
            }
            for abstract_info in &module.types.abstracts {
                let method_count = abstract_info.methods.len() + abstract_info.static_methods.len();
                self.register_abstract_from_blade(abstract_info);
                total_abstracts += 1;
                total_methods += method_count;
            }
        }

        debug!("[BLADE] Registered {} classes, {} enums, {} aliases, {} abstracts ({} methods) from manifest",
            total_classes, total_enums, total_aliases, total_abstracts, total_methods);
    }

    /// Register a class from BLADE symbol info
    fn register_class_from_blade(&mut self, class_info: &BladeClassInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&class_info.name);
        let qualified_name = if class_info.package.is_empty() {
            class_info.name.clone()
        } else {
            format!("{}.{}", class_info.package.join("."), class_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create a scope for the class members
        let class_scope = self.scope_tree.create_scope(Some(ScopeId::first()));

        // Create class symbol using the existing helper method
        let symbol_id = self
            .symbol_table
            .create_class_in_scope(short_name, ScopeId::first());

        // Update symbol metadata including the class scope
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
            sym.scope_id = class_scope; // Set the scope where members are registered
            if class_info.is_extern {
                sym.flags = sym.flags.union(SymbolFlags::EXTERN);
            }
            if class_info.is_final {
                sym.flags = sym.flags.union(SymbolFlags::FINAL);
            }
            if class_info.is_abstract {
                sym.flags = sym.flags.union(SymbolFlags::ABSTRACT);
            }
            if let Some(ref native) = class_info.native_name {
                sym.flags = sym.flags.union(SymbolFlags::NATIVE);
                let native_interned = self.string_interner.intern(native);
                sym.native_name = Some(native_interned);
            }
        }

        // Create class type
        let class_type = self
            .type_table
            .borrow_mut()
            .create_class_type(symbol_id, vec![]);

        // Update symbol with type
        self.symbol_table.update_symbol_type(symbol_id, class_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(class_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        // Register instance methods
        for method in &class_info.methods {
            self.register_method_from_blade(method, symbol_id, class_scope, false);
        }

        // Register static methods
        for method in &class_info.static_methods {
            self.register_method_from_blade(method, symbol_id, class_scope, true);
        }

        // Register constructor if present
        if let Some(ctor) = &class_info.constructor {
            self.register_method_from_blade(ctor, symbol_id, class_scope, false);
        }

        // Register fields, and seed the same per-class field table that a
        // fresh AstLowering pass would export. BLADE symbol restore registers
        // fields into the class scope, but static field lowering consults
        // `global_class_fields` when a later file is lowered. Without this,
        // manifest-restored externs such as Math expose methods but not
        // constants like Math.POSITIVE_INFINITY.
        let mut restored_fields = Vec::new();
        for field in &class_info.fields {
            let field_symbol = self.register_field_from_blade(field, symbol_id, class_scope);
            let field_name = self.string_interner.intern(&field.name);
            restored_fields.push((field_name, field_symbol, field.is_static));
        }

        // Register static fields
        for field in &class_info.static_fields {
            let field_symbol = self.register_field_from_blade(field, symbol_id, class_scope);
            let field_name = self.string_interner.intern(&field.name);
            restored_fields.push((field_name, field_symbol, field.is_static));
        }

        if !restored_fields.is_empty() {
            self.global_class_fields
                .entry(symbol_id)
                .or_insert_with(|| restored_fields.clone());
        }

        trace!(
            "[BLADE] Registered class: {} ({} methods, {} fields) in scope {:?}",
            qualified_name,
            class_info.methods.len() + class_info.static_methods.len(),
            class_info.fields.len() + class_info.static_fields.len(),
            class_scope
        );

        symbol_id
    }

    /// Register a method from BLADE info into a class scope
    fn register_method_from_blade(
        &mut self,
        method: &BladeMethodInfo,
        class_symbol: SymbolId,
        class_scope: ScopeId,
        is_static: bool,
    ) -> SymbolId {
        let method_name = self.string_interner.intern(&method.name);
        let class_qualified_name = self.symbol_table.get_symbol(class_symbol).and_then(|sym| {
            sym.qualified_name
                .and_then(|n| self.string_interner.get(n))
                .or_else(|| self.string_interner.get(sym.name))
                .map(|s| s.to_string())
        });

        // Create the function symbol
        let method_symbol = self
            .symbol_table
            .create_function_in_scope(method_name, class_scope);

        // Parse parameter types and return type to create a function type
        let param_types: Vec<TypeId> = method
            .params
            .iter()
            .map(|p| self.parse_type_string(&p.param_type))
            .collect();
        let return_type = self.parse_type_string(&method.return_type);

        // Create function type
        let func_type = self
            .type_table
            .borrow_mut()
            .create_type(TypeKind::Function {
                params: param_types,
                return_type,
                effects: crate::tast::core::FunctionEffects::default(),
            });

        // Resolve native_name from the cached `@:native` metadata before we
        // open the borrow, so we can intern the string without aliasing
        // `self.symbol_table`.
        let native_name_interned = method
            .native_name
            .as_ref()
            .map(|n| self.string_interner.intern(n));

        // Update symbol with type and flags
        if let Some(sym) = self.symbol_table.get_symbol_mut(method_symbol) {
            sym.type_id = func_type;
            if is_static {
                sym.flags = sym.flags.union(SymbolFlags::STATIC);
            }
            if method.is_inline {
                sym.flags = sym.flags.union(SymbolFlags::INLINE);
            }
            if !method.is_public {
                sym.visibility = crate::tast::symbols::Visibility::Private;
            }
            if let Some(class_name) = &class_qualified_name {
                let method_qualified_name = self
                    .string_interner
                    .intern(&format!("{}.{}", class_name, method.name));
                sym.qualified_name = Some(method_qualified_name);
            }
            // Restore `@:native("foo")` from the BLADE cache. Without this,
            // stdlib runtime mappings have to be keyed by Haxe method name
            // (defeating the purpose of `@:native`), and FFI symbol lookup
            // through `sym.native_name` always finds None for cached types.
            if let Some(native_interned) = native_name_interned {
                sym.native_name = Some(native_interned);
                sym.flags = sym.flags.union(SymbolFlags::NATIVE);
            }
        }

        // Add to scope, updating both symbol list and name lookup cache.
        if let Some(scope) = self.scope_tree.get_scope_mut(class_scope) {
            scope.add_symbol(method_symbol, method_name);
        }

        method_symbol
    }

    /// Register a field from BLADE info into a class scope
    fn register_field_from_blade(
        &mut self,
        field: &crate::ir::blade::BladeFieldInfo,
        _class_symbol: SymbolId,
        class_scope: ScopeId,
    ) -> SymbolId {
        let field_name = self.string_interner.intern(&field.name);

        // Create the field symbol
        let field_symbol = self.symbol_table.create_field(field_name);

        // Parse field type
        let field_type = self.parse_type_string(&field.field_type);

        // Update symbol with type and flags
        if let Some(sym) = self.symbol_table.get_symbol_mut(field_symbol) {
            sym.type_id = field_type;
            sym.scope_id = class_scope;
            if field.is_static {
                sym.flags = sym.flags.union(SymbolFlags::STATIC);
            }
            if field.is_final {
                sym.mutability = crate::tast::symbols::Mutability::Immutable;
            }
            if !field.is_public {
                sym.visibility = crate::tast::symbols::Visibility::Private;
            }
        }

        // Add to scope (using add_symbol to update both symbols list and lookup cache)
        if let Some(scope) = self.scope_tree.get_scope_mut(class_scope) {
            scope.add_symbol(field_symbol, field_name);
        }

        field_symbol
    }

    /// Register an enum from BLADE symbol info
    fn register_enum_from_blade(&mut self, enum_info: &BladeEnumInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&enum_info.name);
        let qualified_name = if enum_info.package.is_empty() {
            enum_info.name.clone()
        } else {
            format!("{}.{}", enum_info.package.join("."), enum_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create enum symbol using the existing helper method
        let symbol_id = self
            .symbol_table
            .create_enum_in_scope(short_name, ScopeId::first());

        // Update symbol metadata
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
            if enum_info.is_extern {
                sym.flags = sym.flags.union(SymbolFlags::EXTERN);
            }
        }

        // Create type parameters for generic enums (e.g., Option<T>, Result<T, E>)
        let mut type_param_ids = Vec::new();
        let mut type_param_map: BTreeMap<String, TypeId> = BTreeMap::new();
        for tp_name in &enum_info.type_params {
            let tp_interned = self.string_interner.intern(tp_name);
            let tp_symbol = self.symbol_table.create_type_parameter(tp_interned, vec![]);
            let tp_type = self.type_table.borrow_mut().create_type_parameter(
                tp_symbol,
                vec![],
                crate::tast::core::Variance::Invariant,
            );
            type_param_ids.push(tp_type);
            type_param_map.insert(tp_name.clone(), tp_type);
        }

        // Create enum type with type parameters
        let enum_type = self
            .type_table
            .borrow_mut()
            .create_enum_type(symbol_id, type_param_ids);

        // Update symbol with type
        self.symbol_table.update_symbol_type(symbol_id, enum_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(enum_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        // Register enum variants in root scope so they can be resolved
        // during pattern matching and constructor calls
        for variant in &enum_info.variants {
            let variant_name = self.string_interner.intern(&variant.name);
            let variant_symbol = self.symbol_table.create_enum_variant_in_scope(
                variant_name,
                ScopeId::first(),
                symbol_id,
            );

            // For generic enum variants whose params reference type parameters,
            // create a Function type with proper TypeParameter TypeIds so
            // resolve_field_type() can substitute them (e.g., T → Int).
            // Only set this for variants where ALL params resolve to known type params;
            // non-generic variants (like TClass(c:Class<Dynamic>)) must NOT get a
            // Function type with invalid params, as that corrupts field type resolution.
            if !variant.params.is_empty() && !type_param_map.is_empty() {
                let param_type_ids: Vec<_> = variant
                    .params
                    .iter()
                    .map(|p| {
                        type_param_map
                            .get(&p.param_type)
                            .copied()
                            .unwrap_or(TypeId::invalid())
                    })
                    .collect();
                // Only set if all params resolved to valid type parameter TypeIds
                if param_type_ids.iter().all(|id| id.is_valid()) {
                    let fn_type = self
                        .type_table
                        .borrow_mut()
                        .create_function_type(param_type_ids, enum_type);
                    self.symbol_table
                        .update_symbol_type(variant_symbol, fn_type);
                }
            }

            // Add variant to root scope for global resolution
            self.scope_tree
                .get_scope_mut(ScopeId::first())
                .expect("Root scope should exist")
                .add_symbol(variant_symbol, variant_name);
        }

        trace!(
            "[BLADE] Registered enum: {} ({} variants)",
            qualified_name,
            enum_info.variants.len()
        );

        symbol_id
    }

    /// Pre-register type declarations from default stdlib files (e.g. StdTypes.hx).
    /// This is lightweight: it parses the files and registers enum/class symbols
    /// into the symbol table without full TAST lowering, preserving lazy stdlib performance.
    /// Register a type alias from BLADE symbol info
    fn register_type_alias_from_blade(&mut self, alias_info: &BladeTypeAliasInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&alias_info.name);
        let qualified_name = if alias_info.package.is_empty() {
            alias_info.name.clone()
        } else {
            format!("{}.{}", alias_info.package.join("."), alias_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create type alias symbol using the existing helper method
        let symbol_id = self
            .symbol_table
            .create_type_alias_in_scope(short_name, ScopeId::first());

        // Update symbol metadata
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
        }

        // Parse the target type string and create appropriate TypeId
        let target_type = self.parse_type_string(&alias_info.target_type);

        // Create type alias type
        let alias_type = self
            .type_table
            .borrow_mut()
            .create_type(TypeKind::TypeAlias {
                symbol_id,
                target_type,
                type_args: vec![],
            });

        // Update symbol with type
        self.symbol_table.update_symbol_type(symbol_id, alias_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(alias_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        trace!(
            "[BLADE] Registered type alias: {} -> {}",
            qualified_name,
            alias_info.target_type
        );

        symbol_id
    }

    /// Register an abstract type from BLADE symbol info
    fn register_abstract_from_blade(&mut self, abstract_info: &BladeAbstractInfo) -> SymbolId {
        let short_name = self.string_interner.intern(&abstract_info.name);
        let qualified_name = if abstract_info.package.is_empty() {
            abstract_info.name.clone()
        } else {
            format!("{}.{}", abstract_info.package.join("."), abstract_info.name)
        };
        let qualified_interned = self.string_interner.intern(&qualified_name);

        // Create a scope for the abstract's methods
        let abstract_scope = self.scope_tree.create_scope(Some(ScopeId::first()));

        // Create abstract symbol using the existing helper method
        let symbol_id = self
            .symbol_table
            .create_abstract_in_scope(short_name, ScopeId::first());

        // Parse the underlying type
        let underlying_type = self.parse_type_string(&abstract_info.underlying_type);

        // Update symbol metadata including the abstract scope
        if let Some(sym) = self.symbol_table.get_symbol_mut(symbol_id) {
            sym.qualified_name = Some(qualified_interned);
            sym.is_exported = true;
            sym.scope_id = abstract_scope; // Set the scope where methods are registered
            if let Some(ref native) = abstract_info.native_name {
                sym.flags = sym.flags.union(SymbolFlags::NATIVE);
                let native_interned = self.string_interner.intern(native);
                sym.native_name = Some(native_interned);
            }
        }

        // Create abstract type
        let abstract_type = self
            .type_table
            .borrow_mut()
            .create_type(TypeKind::Abstract {
                symbol_id,
                underlying: Some(underlying_type),
                type_args: vec![],
            });

        // Update symbol with type
        self.symbol_table
            .update_symbol_type(symbol_id, abstract_type);

        // Register type-symbol mapping
        self.symbol_table
            .register_type_symbol_mapping(abstract_type, symbol_id);

        // Register qualified name alias
        self.symbol_table
            .add_symbol_alias(symbol_id, ScopeId::first(), qualified_interned);

        // Register instance methods
        for method in &abstract_info.methods {
            self.register_method_from_blade(method, symbol_id, abstract_scope, false);
        }

        // Register static methods
        for method in &abstract_info.static_methods {
            self.register_method_from_blade(method, symbol_id, abstract_scope, true);
        }

        trace!(
            "[BLADE] Registered abstract: {} ({} methods) in scope {:?}",
            qualified_name,
            abstract_info.methods.len() + abstract_info.static_methods.len(),
            abstract_scope
        );

        symbol_id
    }

    /// Parse a type string (e.g., "Array<Int>", "String", "Null<Float>") and return a TypeId
    fn parse_type_string(&mut self, type_str: &str) -> TypeId {
        let type_str = type_str.trim();

        // Handle primitives
        match type_str {
            "Int" => return self.type_table.borrow().int_type(),
            "Float" => return self.type_table.borrow().float_type(),
            "Bool" => return self.type_table.borrow().bool_type(),
            "String" => return self.type_table.borrow().string_type(),
            "Void" => return self.type_table.borrow().void_type(),
            "Dynamic" => return self.type_table.borrow().dynamic_type(),
            _ => {}
        }

        // Handle Null<T>
        if let Some(inner) = type_str
            .strip_prefix("Null<")
            .and_then(|s| s.strip_suffix(">"))
        {
            let inner_type = self.parse_type_string(inner);
            return self
                .type_table
                .borrow_mut()
                .create_optional_type(inner_type);
        }

        // Handle Array<T>
        if let Some(inner) = type_str
            .strip_prefix("Array<")
            .and_then(|s| s.strip_suffix(">"))
        {
            let element_type = self.parse_type_string(inner);
            return self.type_table.borrow_mut().create_array_type(element_type);
        }

        // Handle function types: (A, B) -> C
        if type_str.starts_with("(") {
            if let Some((params_str, return_str)) = type_str.split_once(") -> ") {
                let params_str = params_str.trim_start_matches('(');
                let params: Vec<TypeId> = if params_str.is_empty() {
                    vec![]
                } else {
                    self.parse_type_list(params_str)
                };
                let return_type = self.parse_type_string(return_str);
                return self
                    .type_table
                    .borrow_mut()
                    .create_function_type(params, return_type);
            }
        }

        // Handle generic types: ClassName<T, U>
        // Need to find the matching close bracket, not just the last '>'
        if let Some(open) = type_str.find('<') {
            // Find the matching closing bracket
            let mut depth = 0;
            let mut close = None;
            for (i, ch) in type_str.char_indices() {
                match ch {
                    '<' => depth += 1,
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if let Some(close) = close {
                if open < close {
                    let base_name = &type_str[..open];
                    let args_str = &type_str[open + 1..close];
                    let type_args = self.parse_type_list(args_str);

                    // Look up the base type
                    if let Some(symbol_id) = self.lookup_type_symbol(base_name) {
                        return self
                            .type_table
                            .borrow_mut()
                            .create_class_type(symbol_id, type_args);
                    }
                }
            }
        }

        // Simple class/enum name
        if let Some(symbol_id) = self.lookup_type_symbol(type_str) {
            // If the resolved symbol is a TypeAlias, return its existing type_id
            // (the TypeAlias type) instead of wrapping it in a Class type with the
            // alias's symbol_id. Wrapping would synthesise a bogus Class { symbol_id:
            // <typedef-symbol> } whose `symbol_id` doesn't point at a class — every
            // downstream `resolve_type_to_class_symbol` then returns None and method
            // dispatch silently falls through to the wrong class's same-named
            // method (e.g., `bytes.set(...)` jumping into `VecI32.set`).
            let alias_type = self
                .symbol_table
                .get_symbol(symbol_id)
                .filter(|s| s.kind == crate::tast::symbols::SymbolKind::TypeAlias)
                .map(|s| s.type_id)
                .filter(|t| t.is_valid());
            if let Some(t) = alias_type {
                return t;
            }
            return self
                .type_table
                .borrow_mut()
                .create_class_type(symbol_id, vec![]);
        }

        // Create a placeholder for unresolved types
        let name = self.string_interner.intern(type_str);
        self.type_table
            .borrow_mut()
            .create_type(TypeKind::Placeholder { name })
    }

    /// Parse a comma-separated list of types, handling nested generics
    fn parse_type_list(&mut self, types_str: &str) -> Vec<TypeId> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut depth = 0;

        for ch in types_str.chars() {
            match ch {
                '<' => {
                    depth += 1;
                    current.push(ch);
                }
                '>' => {
                    depth -= 1;
                    current.push(ch);
                }
                ',' if depth == 0 => {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        result.push(self.parse_type_string(trimmed));
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }

        // Don't forget the last type
        let trimmed = current.trim();
        if !trimmed.is_empty() {
            result.push(self.parse_type_string(trimmed));
        }

        result
    }

    /// Look up a type symbol by name (checks short name in global scope)
    fn lookup_type_symbol(&self, name: &str) -> Option<SymbolId> {
        // Try short name lookup in global scope first.
        let interned = self.string_interner.intern(name);
        if let Some(symbol) = self.symbol_table.lookup_symbol(ScopeId::first(), interned) {
            return Some(symbol.id);
        }

        // Dotted names (e.g. "rayzor.Bytes"): split into bare short name and
        // verify the resolved symbol's qualified_name matches. Without this,
        // BLADE-preloaded typedefs like `haxe.io.Bytes = rayzor.Bytes` resolve
        // their target to a Placeholder because the symbol is registered as
        // bare name "Bytes".
        if let Some(last_dot) = name.rfind('.') {
            let short = &name[last_dot + 1..];
            let short_interned = self.string_interner.intern(short);
            if let Some(symbol) = self
                .symbol_table
                .lookup_symbol(ScopeId::first(), short_interned)
            {
                let qname_matches = symbol
                    .qualified_name
                    .and_then(|qn| self.string_interner.get(qn))
                    .map(|qn| qn == name)
                    .unwrap_or(false);
                if qname_matches {
                    return Some(symbol.id);
                }
            }
        }

        None
    }

    /// Extract all class references from a Haxe AST file.
    /// This includes explicit imports, using statements, new expressions, and type annotations.
    fn extract_all_dependencies(ast: &parser::HaxeFile) -> Vec<String> {
        use parser::{BlockElement, ClassFieldKind, ExprKind, Type, TypeDeclaration};

        let mut deps = std::collections::BTreeSet::new();

        // 1. Explicit imports
        for import in &ast.imports {
            if !import.path.is_empty() {
                deps.insert(import.path.join("."));
            }
        }

        // 2. Using statements
        for using in &ast.using {
            if !using.path.is_empty() {
                deps.insert(using.path.join("."));
            }
        }

        // Helper to extract type references from a Type
        fn extract_type_deps(ty: &Type, deps: &mut std::collections::BTreeSet<String>) {
            match ty {
                Type::Path { path, params, .. } => {
                    // Only add if it looks like a class name (starts with uppercase)
                    if path.package.is_empty() && !path.name.is_empty() {
                        let first_char = path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false)
                            && !is_stdtypes_ambient_name(&path.name)
                        {
                            deps.insert(path.name.clone());
                        }
                    } else if !path.package.is_empty() {
                        // Qualified type path like sys.io.File
                        let mut full_path = path.package.clone();
                        full_path.push(path.name.clone());
                        deps.insert(full_path.join("."));
                    }
                    // Recurse into type parameters
                    for param in params {
                        extract_type_deps(param, deps);
                    }
                }
                Type::Function { params, ret, .. } => {
                    for param in params {
                        extract_type_deps(param, deps);
                    }
                    extract_type_deps(ret, deps);
                }
                Type::Anonymous { fields, .. } => {
                    for field in fields {
                        extract_type_deps(&field.type_hint, deps);
                    }
                }
                _ => {}
            }
        }

        // Helper to extract dependencies from a block element
        fn extract_block_elem_deps(
            elem: &BlockElement,
            deps: &mut std::collections::BTreeSet<String>,
        ) {
            match elem {
                BlockElement::Expr(e) => extract_expr_deps(e, deps),
                BlockElement::Import(imp) => {
                    if !imp.path.is_empty() {
                        deps.insert(imp.path.join("."));
                    }
                }
                BlockElement::Using(u) => {
                    if !u.path.is_empty() {
                        deps.insert(u.path.join("."));
                    }
                }
                BlockElement::Conditional(cond) => {
                    // Handle #if branch
                    for elem in &cond.if_branch.content {
                        extract_block_elem_deps(elem, deps);
                    }
                    // Handle #elseif branches
                    for branch in &cond.elseif_branches {
                        for elem in &branch.content {
                            extract_block_elem_deps(elem, deps);
                        }
                    }
                    // Handle #else branch
                    if let Some(else_body) = &cond.else_branch {
                        for elem in else_body {
                            extract_block_elem_deps(elem, deps);
                        }
                    }
                }
            }
        }

        // Helper to extract dependencies from an expression
        fn extract_expr_deps(expr: &parser::Expr, deps: &mut std::collections::BTreeSet<String>) {
            match &expr.kind {
                ExprKind::New {
                    type_path,
                    params,
                    args,
                } => {
                    // Extract class name from new expression
                    if type_path.package.is_empty() && !type_path.name.is_empty() {
                        let first_char = type_path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false)
                            && !is_stdtypes_ambient_name(&type_path.name)
                        {
                            deps.insert(type_path.name.clone());
                        }
                    } else if !type_path.package.is_empty() {
                        let mut full_path = type_path.package.clone();
                        full_path.push(type_path.name.clone());
                        deps.insert(full_path.join("."));
                    }
                    // Recurse into type params and args
                    for param in params {
                        extract_type_deps(param, deps);
                    }
                    for arg in args {
                        extract_expr_deps(arg, deps);
                    }
                }
                ExprKind::Call { expr, args } => {
                    extract_expr_deps(expr, deps);
                    for arg in args {
                        extract_expr_deps(arg, deps);
                    }
                }
                ExprKind::Field { expr, field, .. } => {
                    // If the object is a capitalized identifier, it may be a class/module
                    // reference (e.g. NativeStackTrace.exceptionStack()). Add it as a
                    // potential unqualified dep so load_imports_efficiently can resolve it
                    // via package-prefix fallback (tries haxe.X, haxe.ds.X, etc.).
                    if let ExprKind::Ident(name) = &expr.kind {
                        if name
                            .chars()
                            .next()
                            .map(|c| c.is_uppercase())
                            .unwrap_or(false)
                            && !is_stdtypes_ambient_name(name)
                        {
                            deps.insert(name.clone());
                        }
                    }
                    // Also try to extract qualified paths from nested field chains
                    // e.g. haxe.io.Bytes.ofString(...) → "haxe.io.Bytes"
                    // We check the full chain including current field — if the last
                    // component is uppercase, it's a package.Class pattern
                    if field
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        // Current field is uppercase (e.g. "Bytes" in haxe.io.Bytes)
                        // Try to build full qualified path from the expression chain + this field
                        let mut parts = Vec::new();
                        fn collect_parts_for_dep(e: &parser::Expr, parts: &mut Vec<String>) {
                            match &e.kind {
                                ExprKind::Ident(name) => parts.push(name.clone()),
                                ExprKind::Field { expr, field, .. } => {
                                    collect_parts_for_dep(expr, parts);
                                    parts.push(field.clone());
                                }
                                _ => {}
                            }
                        }
                        collect_parts_for_dep(expr, &mut parts);
                        parts.push(field.clone());
                        if parts.len() >= 2 {
                            deps.insert(parts.join("."));
                        }
                    }
                    extract_expr_deps(expr, deps);
                }
                ExprKind::Index { expr, index } => {
                    extract_expr_deps(expr, deps);
                    extract_expr_deps(index, deps);
                }
                ExprKind::Unary { expr, .. } => {
                    extract_expr_deps(expr, deps);
                }
                ExprKind::Binary { left, right, .. } => {
                    extract_expr_deps(left, deps);
                    extract_expr_deps(right, deps);
                }
                ExprKind::Assign { left, right, .. } => {
                    extract_expr_deps(left, deps);
                    extract_expr_deps(right, deps);
                }
                ExprKind::Ternary {
                    cond,
                    then_expr,
                    else_expr,
                } => {
                    extract_expr_deps(cond, deps);
                    extract_expr_deps(then_expr, deps);
                    extract_expr_deps(else_expr, deps);
                }
                ExprKind::Array(elems) => {
                    for elem in elems {
                        extract_expr_deps(elem, deps);
                    }
                }
                ExprKind::Block(elems) => {
                    for elem in elems {
                        extract_block_elem_deps(elem, deps);
                    }
                }
                ExprKind::Var {
                    type_hint, expr, ..
                }
                | ExprKind::Final {
                    type_hint, expr, ..
                } => {
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                    if let Some(e) = expr {
                        extract_expr_deps(e, deps);
                    }
                }
                ExprKind::Return(Some(e)) | ExprKind::Throw(e) => {
                    extract_expr_deps(e, deps);
                }
                ExprKind::If {
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    extract_expr_deps(cond, deps);
                    extract_expr_deps(then_branch, deps);
                    if let Some(e) = else_branch {
                        extract_expr_deps(e, deps);
                    }
                }
                ExprKind::While { cond, body } | ExprKind::DoWhile { body, cond } => {
                    extract_expr_deps(cond, deps);
                    extract_expr_deps(body, deps);
                }
                ExprKind::For { iter, body, .. } => {
                    extract_expr_deps(iter, deps);
                    extract_expr_deps(body, deps);
                }
                ExprKind::Try {
                    expr,
                    catches,
                    finally_block,
                } => {
                    extract_expr_deps(expr, deps);
                    for catch in catches {
                        if let Some(ty) = &catch.type_hint {
                            extract_type_deps(ty, deps);
                        }
                        extract_expr_deps(&catch.body, deps);
                    }
                    if let Some(finally) = finally_block {
                        extract_expr_deps(finally, deps);
                    }
                }
                ExprKind::Cast { expr, type_hint } => {
                    extract_expr_deps(expr, deps);
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                }
                ExprKind::TypeCheck { expr, type_hint } => {
                    extract_expr_deps(expr, deps);
                    extract_type_deps(type_hint, deps);
                }
                ExprKind::Switch {
                    expr,
                    cases,
                    default,
                } => {
                    extract_expr_deps(expr, deps);
                    for case in cases {
                        // Extract from patterns (they may contain constructor references)
                        for pattern in &case.patterns {
                            extract_pattern_deps(pattern, deps);
                        }
                        if let Some(guard) = &case.guard {
                            extract_expr_deps(guard, deps);
                        }
                        extract_expr_deps(&case.body, deps);
                    }
                    if let Some(d) = default {
                        extract_expr_deps(d, deps);
                    }
                }
                ExprKind::Arrow { expr, .. } => {
                    extract_expr_deps(expr, deps);
                }
                ExprKind::Map(pairs) => {
                    for (k, v) in pairs {
                        extract_expr_deps(k, deps);
                        extract_expr_deps(v, deps);
                    }
                }
                ExprKind::Object(fields) => {
                    for field in fields {
                        extract_expr_deps(&field.expr, deps);
                    }
                }
                ExprKind::Function(func) => {
                    // Extract from function parameters and return type
                    for param in &func.params {
                        if let Some(ty) = &param.type_hint {
                            extract_type_deps(ty, deps);
                        }
                        if let Some(default) = &param.default_value {
                            extract_expr_deps(default, deps);
                        }
                    }
                    if let Some(ret) = &func.return_type {
                        extract_type_deps(ret, deps);
                    }
                    if let Some(body) = &func.body {
                        extract_expr_deps(body, deps);
                    }
                }
                ExprKind::Paren(e)
                | ExprKind::Untyped(e)
                | ExprKind::Meta { expr: e, .. }
                | ExprKind::Macro(e)
                | ExprKind::Inline(e)
                | ExprKind::Reify(e) => {
                    extract_expr_deps(e, deps);
                }
                ExprKind::Tuple(elements) => {
                    for e in elements {
                        extract_expr_deps(e, deps);
                    }
                }
                ExprKind::ArrayComprehension { for_parts, expr } => {
                    for part in for_parts {
                        extract_expr_deps(&part.iter, deps);
                    }
                    extract_expr_deps(expr, deps);
                }
                ExprKind::MapComprehension {
                    for_parts,
                    key,
                    value,
                } => {
                    for part in for_parts {
                        extract_expr_deps(&part.iter, deps);
                    }
                    extract_expr_deps(key, deps);
                    extract_expr_deps(value, deps);
                }
                ExprKind::StringInterpolation(parts) => {
                    for part in parts {
                        if let parser::StringPart::Interpolation(e) = part {
                            extract_expr_deps(e, deps);
                        }
                    }
                }
                _ => {}
            }
        }

        // Helper to extract dependencies from patterns (in switch cases)
        fn extract_pattern_deps(
            pattern: &parser::Pattern,
            deps: &mut std::collections::BTreeSet<String>,
        ) {
            match pattern {
                parser::Pattern::Const(e) => extract_expr_deps(e, deps),
                parser::Pattern::Constructor { path, params } => {
                    // Constructor patterns reference enum/class types
                    if path.package.is_empty() && !path.name.is_empty() {
                        let first_char = path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false)
                            && !is_stdtypes_ambient_name(&path.name)
                        {
                            deps.insert(path.name.clone());
                        }
                    } else if !path.package.is_empty() {
                        let mut full_path = path.package.clone();
                        full_path.push(path.name.clone());
                        deps.insert(full_path.join("."));
                    }
                    for param in params {
                        extract_pattern_deps(param, deps);
                    }
                }
                parser::Pattern::Array(patterns) | parser::Pattern::Or(patterns) => {
                    for p in patterns {
                        extract_pattern_deps(p, deps);
                    }
                }
                parser::Pattern::ArrayRest { elements, .. } => {
                    for p in elements {
                        extract_pattern_deps(p, deps);
                    }
                }
                parser::Pattern::Object { fields } => {
                    for (_, pattern) in fields {
                        extract_pattern_deps(pattern, deps);
                    }
                }
                parser::Pattern::Type { type_hint, .. } => {
                    extract_type_deps(type_hint, deps);
                }
                parser::Pattern::Extractor { expr, value } => {
                    extract_expr_deps(expr, deps);
                    extract_expr_deps(value, deps);
                }
                _ => {}
            }
        }

        // Helper to extract dependencies from class fields
        fn extract_field_deps(
            field: &parser::ClassField,
            deps: &mut std::collections::BTreeSet<String>,
        ) {
            match &field.kind {
                ClassFieldKind::Var {
                    type_hint, expr, ..
                }
                | ClassFieldKind::Final {
                    type_hint, expr, ..
                } => {
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                    if let Some(e) = expr {
                        extract_expr_deps(e, deps);
                    }
                }
                ClassFieldKind::Property { type_hint, .. } => {
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                }
                ClassFieldKind::Function(func) => {
                    for param in &func.params {
                        if let Some(ty) = &param.type_hint {
                            extract_type_deps(ty, deps);
                        }
                        if let Some(default) = &param.default_value {
                            extract_expr_deps(default, deps);
                        }
                    }
                    if let Some(ret) = &func.return_type {
                        extract_type_deps(ret, deps);
                    }
                    if let Some(body) = &func.body {
                        extract_expr_deps(body, deps);
                    }
                }
            }
        }

        // 3. Extract from type declarations (classes, interfaces, etc.)
        for decl in &ast.declarations {
            match decl {
                TypeDeclaration::Class(class_decl) => {
                    // Extract from extends clause
                    if let Some(extends) = &class_decl.extends {
                        extract_type_deps(extends, &mut deps);
                    }
                    // Extract from implements clause
                    for impl_type in &class_decl.implements {
                        extract_type_deps(impl_type, &mut deps);
                    }
                    // Extract from fields
                    for field in &class_decl.fields {
                        extract_field_deps(field, &mut deps);
                    }
                }
                TypeDeclaration::Interface(iface_decl) => {
                    // Extract from extends clause
                    for extends in &iface_decl.extends {
                        extract_type_deps(extends, &mut deps);
                    }
                    // Extract from fields
                    for field in &iface_decl.fields {
                        extract_field_deps(field, &mut deps);
                    }
                }
                TypeDeclaration::Typedef(typedef_decl) => {
                    extract_type_deps(&typedef_decl.type_def, &mut deps);
                }
                TypeDeclaration::Enum(enum_decl) => {
                    for ctor in &enum_decl.constructors {
                        for param in &ctor.params {
                            if let Some(ty) = &param.type_hint {
                                extract_type_deps(ty, &mut deps);
                            }
                        }
                    }
                }
                TypeDeclaration::Abstract(abstract_decl) => {
                    if let Some(ty) = &abstract_decl.underlying {
                        extract_type_deps(ty, &mut deps);
                    }
                    for ty in &abstract_decl.from {
                        extract_type_deps(ty, &mut deps);
                    }
                    for ty in &abstract_decl.to {
                        extract_type_deps(ty, &mut deps);
                    }
                    for field in &abstract_decl.fields {
                        extract_field_deps(field, &mut deps);
                    }
                }
                TypeDeclaration::Conditional(cond) => {
                    // Handle conditional compilation blocks
                    // Handle #if branch
                    for inner_decl in &cond.if_branch.content {
                        if let TypeDeclaration::Class(c) = inner_decl {
                            for field in &c.fields {
                                extract_field_deps(field, &mut deps);
                            }
                        }
                    }
                    // Handle #elseif branches
                    for branch in &cond.elseif_branches {
                        for inner_decl in &branch.content {
                            if let TypeDeclaration::Class(c) = inner_decl {
                                for field in &c.fields {
                                    extract_field_deps(field, &mut deps);
                                }
                            }
                        }
                    }
                    // Handle #else branch
                    if let Some(else_body) = &cond.else_branch {
                        for inner_decl in else_body {
                            if let TypeDeclaration::Class(c) = inner_decl {
                                for field in &c.fields {
                                    extract_field_deps(field, &mut deps);
                                }
                            }
                        }
                    }
                }
            }
        }

        // StdTypes.hx contributes top-level prelude types. They are not files
        // the user imports, so do not feed bare `Iterator`, `Null`, `Bool`, etc.
        // into load_imports_efficiently where they become failed path guesses.
        deps.retain(|d| !Self::is_bare_stdtypes_prelude_dependency(d));

        // Qualify bare type names with the file's package.
        // e.g., if File.hx has `package sys.io;` and references `FileInput`,
        // also add `sys.io.FileInput` so the import loader can find it.
        if let Some(package) = &ast.package {
            if !package.path.is_empty() {
                let package_prefix = package.path.join(".");
                let qualified: Vec<String> = deps
                    .iter()
                    .filter(|d| {
                        !d.contains('.')
                            && d.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    })
                    .map(|d| format!("{}.{}", package_prefix, d))
                    .collect();
                for q in qualified {
                    deps.insert(q);
                }
            }
        }

        // A type this file declares itself is not a dependency. Leaving it in
        // sends an undotted name to the import loader, which guesses a package
        // for it — so a file with its own `Resource` pulls in `haxe.Resource`
        // and reports that an unrelated module failed to compile.
        let own_types: std::collections::BTreeSet<String> = ast
            .declarations
            .iter()
            .filter_map(|decl| match decl {
                TypeDeclaration::Class(c) => Some(c.name.clone()),
                TypeDeclaration::Interface(i) => Some(i.name.clone()),
                TypeDeclaration::Enum(e) => Some(e.name.clone()),
                TypeDeclaration::Abstract(a) => Some(a.name.clone()),
                TypeDeclaration::Typedef(t) => Some(t.name.clone()),
                _ => None,
            })
            .collect();
        deps.retain(|d| !own_types.contains(d));
        deps.retain(|d| !Self::is_bare_stdtypes_prelude_dependency(d));

        let mut result: Vec<String> = deps.into_iter().collect();
        result.sort();
        result
    }

    /// Load imports efficiently by pre-collecting all dependencies and compiling in topological order.
    /// This avoids the fail-retry pattern that causes exponential recompilation.
    pub fn load_imports_efficiently(&mut self, imports: &[String]) -> Result<(), String> {
        use std::collections::{BTreeMap, BTreeSet, VecDeque};

        // Step 1: Collect all files and their dependencies by parsing (not compiling)
        // Use BTreeMap for deterministic iteration order
        // and causes non-deterministic import base offsets, leading to different function
        // IDs, different inlining decisions, and ultimately wrong optimized MIR.
        let mut all_files: BTreeMap<String, (PathBuf, String, Vec<String>)> = BTreeMap::new();
        let mut to_process: VecDeque<String> = VecDeque::new();
        for name in imports {
            if is_stdtypes_ambient_import(name) {
                continue;
            }
            if self.stdlib_manifest_loaded && is_extern_only_stdlib_import(name) {
                if self.config.profile_typecheck {
                    self.typecheck_timings.import_extern_skips += 1;
                }
                continue;
            }
            to_process.push_back(name.clone());
        }
        let mut visited: BTreeSet<String> = BTreeSet::new();

        while let Some(qualified_path) = to_process.pop_front() {
            if is_manifest_backed_ambient_import(&qualified_path, self.stdlib_manifest_loaded)
                || visited.contains(&qualified_path)
            {
                if self.stdlib_manifest_loaded && is_extern_only_stdlib_import(&qualified_path) {
                    if self.config.profile_typecheck {
                        self.typecheck_timings.import_extern_skips += 1;
                    }
                }
                continue;
            }
            visited.insert(qualified_path.clone());

            // Resolve to file path (use _force variant to bypass BLADE cache's
            // is_file_loaded check — BLADE pre-registers symbols but doesn't preserve
            // full TAST state needed for generic instantiation and method resolution)
            let resolved = self
                .namespace_resolver
                .resolve_qualified_path_to_file_force(&qualified_path);
            let file_path = if let Some(path) = resolved {
                path
            } else if !qualified_path.contains('.') {
                // Try common prefixes for unqualified names
                let prefixes = [
                    "haxe.iterators",
                    "haxe.ds",
                    "haxe",
                    "sys.thread",
                    "sys",
                    "haxe.exceptions",
                    "haxe.io",
                ];
                let mut found = None;
                for prefix in &prefixes {
                    let full = format!("{}.{}", prefix, qualified_path);
                    if let Some(path) = self
                        .namespace_resolver
                        .resolve_qualified_path_to_file_force(&full)
                    {
                        found = Some(path);
                        break;
                    }
                }
                match found {
                    Some(p) => p,
                    None => continue, // Skip unresolvable imports
                }
            } else {
                continue; // Skip unresolvable
            };

            // Deduplicate by file path — the same file can appear under different
            // qualified names (e.g., "BalancedTree" and "haxe.ds.BalancedTree")
            let file_path_str = file_path.to_string_lossy().to_string();
            if all_files
                .values()
                .any(|(p, _, _)| p.to_string_lossy() == file_path_str)
            {
                continue;
            }

            // BLADE manifests register extern-only stdlib classes up front.
            // They have no Haxe bodies we need to lower, so do not spend cold
            // start time reading/parsing them just to skip them later.
            let base = std::path::Path::new(&file_path_str)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if self.namespace_resolver.is_file_loaded(&file_path)
                && is_extern_only_stdlib_base(base)
            {
                if self.config.profile_typecheck {
                    self.typecheck_timings.import_extern_skips += 1;
                }
                continue;
            }

            // Read and parse to extract imports
            let source = match std::fs::read_to_string(&file_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let filename = file_path_str;
            let deps = match parser::parse_haxe_file(&filename, &source, false) {
                Ok(ast) => Self::extract_all_dependencies(&ast),
                Err(_) => Vec::new(),
            };
            // Queue dependencies for processing
            for dep in &deps {
                if is_manifest_backed_ambient_import(dep, self.stdlib_manifest_loaded) {
                    if self.stdlib_manifest_loaded && is_extern_only_stdlib_import(dep) {
                        if self.config.profile_typecheck {
                            self.typecheck_timings.import_extern_skips += 1;
                        }
                    }
                    continue;
                }
                if !visited.contains(dep) {
                    to_process.push_back(dep.clone());
                }
            }

            all_files.insert(qualified_path.clone(), (file_path, source, deps));
        }

        // Debug: log collected files
        if !all_files.is_empty() {
            debug!(
                "[IMPORT_LOAD] Collected {} files for import",
                all_files.len()
            );
        }
        if self.config.profile_typecheck {
            self.typecheck_timings.imports_collected += all_files.len();
        }

        // Step 2: Topological sort using Kahn's algorithm
        // Use BTreeMap for deterministic iteration order
        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();

        // Build reverse map: bare class name → qualified name for same-package deps
        let bare_to_qualified: BTreeMap<String, String> = all_files
            .keys()
            .filter_map(|qn| {
                let short = qn.rsplit('.').next()?;
                Some((short.to_string(), qn.clone()))
            })
            .collect();

        for (name, (_, _, deps)) in &all_files {
            in_degree.entry(name.clone()).or_insert(0);
            for dep in deps {
                // Skip self-dependencies (class referencing itself in its own file)
                if dep == name {
                    continue;
                }
                // Check both qualified name ("sim.Point2D") and bare name ("Point2D")
                let resolved_dep = if all_files.contains_key(dep) {
                    Some(dep.clone())
                } else {
                    bare_to_qualified.get(dep).cloned()
                };
                if let Some(resolved) = resolved_dep {
                    if resolved != *name {
                        graph
                            .entry(resolved.clone())
                            .or_default()
                            .push(name.clone());
                        *in_degree.entry(name.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut compile_order: Vec<String> = Vec::new();

        while let Some(name) = queue.pop_front() {
            compile_order.push(name.clone());
            if let Some(dependents) = graph.get(&name) {
                for dep in dependents {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep.clone());
                        }
                    }
                }
            }
        }

        // Handle cycle: if compile_order doesn't include all files, some are stuck in a cycle.
        // Append remaining files in any order (they'll still compile, just without guaranteed dep order).
        if compile_order.len() < all_files.len() {
            let stuck_count = all_files.len() - compile_order.len();
            if stuck_count > 0 {
                debug!(
                    "Cycle detected, {} files stuck in dependency cycle. Forcing compilation.",
                    stuck_count
                );
                let in_order: std::collections::BTreeSet<_> =
                    compile_order.iter().cloned().collect();
                // Mini topological sort for stuck files: repeatedly emit files
                // whose deps (among remaining stuck files) are all already emitted.
                let mut stuck: BTreeMap<String, Vec<String>> = BTreeMap::new();
                for name in all_files.keys() {
                    if in_order.contains(name) {
                        continue;
                    }
                    let deps_in_stuck: Vec<String> = all_files
                        .get(name)
                        .map(|(_, _, deps)| {
                            deps.iter()
                                .filter_map(|d| {
                                    let resolved = if all_files.contains_key(d) {
                                        Some(d.clone())
                                    } else {
                                        bare_to_qualified.get(d).cloned()
                                    };
                                    resolved.filter(|r| {
                                        r != name
                                            && !in_order.contains(r)
                                            && all_files.contains_key(r)
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    stuck.insert(name.clone(), deps_in_stuck);
                }
                let mut emitted: BTreeSet<String> = BTreeSet::new();
                loop {
                    let ready: Vec<String> = stuck
                        .iter()
                        .filter(|(_, deps)| deps.iter().all(|d| emitted.contains(d)))
                        .map(|(name, _)| name.clone())
                        .collect();
                    if !ready.is_empty() {
                        for name in &ready {
                            compile_order.push(name.clone());
                            emitted.insert(name.clone());
                            stuck.remove(name);
                        }
                        continue; // Check for more ready files
                    }
                    // No ready file — break a cycle by emitting the one with the
                    // FEWEST still-unemitted stuck-deps (ties: name order). This
                    // emits leaf-like dependencies first so a constructor's
                    // callee (e.g. `LlamaModel`, built by `LlamaArch.build` via
                    // `new LlamaModel()`) lands before its caller — otherwise the
                    // caller can't resolve the callee's constructor and leaves the
                    // object unconstructed.
                    let victim = stuck
                        .iter()
                        .map(|(name, deps)| {
                            let outstanding = deps.iter().filter(|d| !emitted.contains(*d)).count();
                            (outstanding, name.clone())
                        })
                        .min()
                        .map(|(_, name)| name);
                    if let Some(name) = victim {
                        compile_order.push(name.clone());
                        emitted.insert(name.clone());
                        stuck.remove(&name);
                        continue; // Re-check after cycle break
                    }
                    break; // All emitted
                }
            }
        }

        // Step 3: Compile in topological order with retry for files that fail
        // due to unresolved symbols (dependency ordering issues from cycles).
        debug!(
            "[IMPORT_LOAD] Compiling {} stdlib files: {:?}",
            compile_order.len(),
            compile_order
        );
        // Snapshot the diagnostics length before each first-pass attempt so
        // we can discard import errors that get resolved by the retry pass.
        // Without this, every transient dependency-ordering failure surfaces
        // (e.g. `Cannot find name 'FPHelper'` from haxe.io.Input compiled
        // before haxe.io.FPHelper) even though the retry succeeds.
        let first_pass_snapshot = self.collected_diagnostics.len();
        let mut retry_queue: Vec<(String, PathBuf, String, Vec<String>, usize)> = Vec::new();
        for name in compile_order {
            if let Some((file_path, source, deps)) = all_files.remove(&name) {
                let diag_snapshot = self.collected_diagnostics.len();
                if !self.try_compile_import(&name, &file_path, &source, deps.clone()) {
                    retry_queue.push((name, file_path, source, deps, diag_snapshot));
                } else {
                    // Success: any diagnostics pushed during this attempt were
                    // recovered (e.g. from a partial parse) — keep them.
                }
            }
        }

        // Retry failed files until a pass resolves nothing new. Their
        // dependencies get registered as earlier files in the queue succeed —
        // but a DEEP import chain (A imports B imports C, all failing the first
        // pass on ordering) needs as many passes as its depth. The prior code
        // did a SINGLE retry, so it cleared only one level: a long chain like
        // `Main -> GGUFLoader -> GGUFReader.TensorInfo -> ...` left the tail
        // stranded as an empty MIR module whose methods then trap as forward-ref
        // stubs (udf #0xc11f / wasm unreachable) at unrelated call sites. Each
        // pass discards transient ordering errors (truncate to
        // first_pass_snapshot); only the survivors of a no-progress pass are
        // surfaced as genuine failures. A failed attempt pushes no MIR (the
        // pop+push is in try_compile_import's Ok branch), so re-attempting is
        // safe and never duplicates a module.
        let mut pending: Vec<(String, PathBuf, String, Vec<String>)> = retry_queue
            .into_iter()
            .map(|(n, p, s, d, _snap)| (n, p, s, d))
            .collect();
        let final_failures: Vec<String> = loop {
            self.collected_diagnostics.truncate(first_pass_snapshot);
            let before = pending.len();
            let mut next = Vec::new();
            for (name, file_path, source, deps) in std::mem::take(&mut pending) {
                if !self.try_compile_import(&name, &file_path, &source, deps.clone()) {
                    next.push((name, file_path, source, deps));
                }
            }
            // No progress: the survivors are genuine failures; their errors
            // (pushed after first_pass_snapshot during this pass) are kept.
            if next.len() == before {
                break next.into_iter().map(|(n, ..)| n).collect();
            }
            pending = next;
        };

        // LOUD-FAIL only the genuine FINAL failures — modules still unresolved
        // after the retry loop converged. Transient first-pass ordering failures
        // (a module compiled before its dependency) are NOT reported here, since
        // a later pass resolved them. Each call into a truly-failed module lowers
        // to a forward-ref trap stub (udf #0xc11f / wasm `unreachable`), so this
        // is the difference between a clean run and a silent SIGILL.
        for name in &final_failures {
            if let Some(errs) = self.last_import_errors.get(name) {
                eprintln!(
                    "error[IMPORT]: imported module `{}` failed to compile ({} error(s)); \
                     calls into it will trap at runtime. First errors:",
                    name,
                    errs.len()
                );
                for line in errs.iter().take(8) {
                    eprintln!("    {}", line);
                }
            }
        }

        // Fixup pass: resolve stale cross-module refs that couldn't be resolved during
        // renumbering because the target module hadn't been loaded yet (ordering issue
        // with blade cache). Now that ALL modules are loaded, stdlib_function_name_map
        // is complete and we can resolve any remaining stale refs.
        self.fixup_stale_cross_module_refs();
        self.fixup_stale_constructor_ids();
        self.fixup_stale_method_ids();

        Ok(())
    }

    /// Try to compile a single import file. Returns true on success, false on failure.
    fn try_compile_import(
        &mut self,
        name: &str,
        file_path: &Path,
        source: &str,
        deps: Vec<String>,
    ) -> bool {
        let filename = file_path.to_string_lossy().to_string();

        // Skip if already compiled
        if self.compiled_files.contains_key(&filename) {
            if self.config.profile_typecheck {
                self.typecheck_timings.import_already_compiled += 1;
            }
            return true;
        }

        // Skip extern-only stdlib files whose types are already registered via bsym.
        // These files have no function bodies — their methods map to runtime externs
        // via MIR wrappers. Compiling them produces 0 MIR functions.
        // Files with real code (Exception, StringTools, BalancedTree, etc.) must still
        // be compiled to get field indices, constructors, and MIR.
        let is_loaded = self
            .namespace_resolver
            .is_file_loaded(&file_path.to_path_buf());
        if is_loaded {
            let base = std::path::Path::new(&filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if is_extern_only_stdlib_base(base) {
                if self.config.profile_typecheck {
                    self.typecheck_timings.import_extern_skips += 1;
                }
                return true;
            }
        }

        // Mark as loaded
        self.namespace_resolver
            .mark_file_loaded(file_path.to_path_buf());

        // Phase 2: capture raw AST of imports containing macros so that
        // cross-file macro discovery (e.g. `import tink.Json; tink.Json.parse(...)`)
        // works even when the import is served from BLADE cache. This is gated
        // by a cheap substring check to avoid parsing every stdlib file; only
        // files containing macro definitions need to be kept around for the
        // expander's dependency scan.
        if source.contains("macro ") || source.contains("@:build") {
            if let Ok(raw_ast) = self.parse_file(&filename, source) {
                self.loaded_import_haxe_files.push(raw_ast);
            }
        }

        // Try BLADE cache first. Typedef-only modules are cacheable now that
        // BLADE restores type aliases as first-class type-system symbols; their
        // dependencies are still walked by the import loader before consumers
        // resolve the alias target in the current compilation context.
        let source_has_typedef = source.contains("typedef ");
        let cache_hit = if self.config.enable_cache {
            let hit = self.try_load_import_from_cache(&filename, source);
            if self.config.profile_typecheck {
                if hit {
                    self.typecheck_timings.import_cache_hits += 1;
                } else {
                    self.typecheck_timings.import_cache_misses += 1;
                }
            }
            hit
        } else {
            false
        };

        if cache_hit {
            // BLADE cache hit means compile_file_with_shared_state_ex →
            // compile_ast_with_shared_state was NOT called for this file,
            // so file_id_by_filename / file_source_by_filename never got
            // an entry. Allocate them now from the in-memory `source` so
            // the renderer can still locate this file by file_id and
            // resolve its byte_offsets against the same bytes the cached
            // MIR's spans were computed against.
            let file_id_u32 = *self
                .file_id_by_filename
                .entry(filename.to_string())
                .or_insert_with(|| {
                    let id = self.next_file_id;
                    self.next_file_id += 1;
                    id
                });
            let _ = file_id_u32;
            self.file_source_by_filename
                .entry(filename.to_string())
                .or_insert_with(|| source.to_string());
            return true;
        }

        // Never skip pre-registration for import files — they always need
        // full type registration to make extern class fields visible.
        let is_stdlib = false;
        // User package imports skip the stdlib merge — it happens once in the main file's
        // compilation. This prevents stdlib functions from overwriting user functions by bare name.
        // Stdlib imports (EReg, StringTools, etc.) still need the merge because their Haxe
        // source has placeholder method bodies that must be replaced by MIR wrappers.
        let skip_stdlib_merge = !is_stdlib;
        match self.compile_file_with_shared_state_ex(
            &filename,
            source,
            is_stdlib,
            skip_stdlib_merge,
        ) {
            Ok(typed_file) => {
                if self.config.profile_typecheck {
                    self.typecheck_timings.import_fresh_compiles += 1;
                    if source_has_typedef {
                        self.typecheck_timings.import_typedef_fresh += 1;
                    }
                }
                // Extract inline var constants before consuming the TypedFile.
                // These are stored in BLADE cache and in global_inline_vars for
                // cross-file static inline var resolution (e.g., Key.ESCAPE).
                let inline_vars = Self::extract_inline_vars_from_typed_file(
                    &typed_file,
                    &self.symbol_table,
                    &self.string_interner,
                );
                self.store_inline_vars(&inline_vars);

                // Register extern class methods as plugin mappings so MIR lowerer
                // can resolve them (same as rpkg NativePlugin does).
                self.register_extern_methods_from_typed_file(&typed_file);

                self.loaded_stdlib_typed_files.push(typed_file);

                // Move the MIR from mir_modules to import_mir_modules.
                if let Some(mir_arc) = self.mir_modules.pop() {
                    // Save to BLADE cache before renumbering
                    if self.config.enable_cache {
                        let type_info = self.last_compiled_type_info.take();
                        let mut cached_maps = self.last_compiled_cached_maps.take();
                        // Append inline vars to cached maps for cache persistence
                        if let Some(ref mut maps) = cached_maps {
                            maps.inline_vars = inline_vars;
                        }
                        self.save_blade_cached(
                            &filename,
                            source,
                            &mir_arc,
                            deps,
                            type_info,
                            cached_maps,
                        );
                    }

                    // Track which import functions are source-level declarations
                    // (methods, constructors) vs generated MIR wrappers.
                    // Only for USER PACKAGES — stdlib files (EReg, StringTools, etc.) have
                    // placeholder method bodies that must be replaced by stdlib MIR wrappers.
                    let own_ids = self.last_compiled_own_func_ids.take().unwrap_or_default();
                    let is_user_package = !filename.contains("haxe-std");
                    if is_user_package {
                        let import_base: u32 =
                            100_000 + (self.import_mir_modules.len() as u32 * 10_000);
                        for old_id in &own_ids {
                            let new_id = crate::ir::IrFunctionId(old_id.0 + import_base);
                            self.import_own_func_ids.insert(new_id);
                        }
                    }

                    self.renumber_and_push_import_mir((*mir_arc).clone());
                } else if self.config.enable_cache {
                    let type_info = self.last_compiled_type_info.take();
                    let _ = self.last_compiled_cached_maps.take();
                    if let Some(type_info) = type_info {
                        let module_name = std::path::Path::new(&filename)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("stdlib")
                            .to_string();
                        let empty_mir = crate::ir::IrModule::new(module_name, filename.clone());
                        self.save_blade_cached(
                            &filename,
                            source,
                            &empty_mir,
                            deps,
                            Some(type_info),
                            None,
                        );
                    }
                }
                true
            }
            Err(errors) => {
                // LOUD-FAIL: an imported module that fails to compile yields an
                // empty MIR, so every call into it lowers to a forward-ref stub
                // that traps (udf #0xc11f / wasm `unreachable`) at call time with
                // NO diagnostic. Stash this attempt's errors keyed by module name
                // — try_compile_import runs once per retry pass, so printing here
                // would over-report transient ordering failures that a later pass
                // resolves. The retry loop surfaces only the genuine FINAL
                // survivors (see load_imports_efficiently) from this map.
                self.last_import_errors.insert(
                    name.to_string(),
                    errors
                        .iter()
                        .map(|e| {
                            format!("{} ({}:{})", e.message, e.location.line, e.location.column)
                        })
                        .collect(),
                );
                // Surface import failures via the regular diagnostic
                // pipeline. Previously these went only to debug-log and the
                // caller saw "false" — so a parse/type error in an imported
                // file (e.g. `var x = 1e-5` failing to lex as a Float)
                // silently produced an empty MIR module. Downstream lookups
                // for the imported class's methods then returned forward-ref
                // stubs that ended in `unreachable`, surfacing as a SIGILL
                // / silent exit at unrelated call sites in the importer.
                debug!(
                    "[IMPORT_LOAD] Failed to compile {}: {} error(s)",
                    name,
                    errors.len()
                );
                for e in &errors {
                    debug!("  - {}", e.message);
                    // Convert CompilationError → Diagnostic and add to the
                    // collected pool so the runner prints it.
                    let span = if e.location.is_valid() {
                        let pos = diagnostics::SourcePosition::new(
                            e.location.line as usize,
                            e.location.column as usize,
                            e.location.byte_offset as usize,
                        );
                        let end_pos = diagnostics::SourcePosition::new(
                            e.location.line as usize,
                            (e.location.column + 1) as usize,
                            (e.location.byte_offset + 1) as usize,
                        );
                        diagnostics::SourceSpan::new(pos, end_pos, diagnostics::FileId::new(0))
                    } else {
                        diagnostics::SourceSpan::new(
                            diagnostics::SourcePosition::new(0, 0, 0),
                            diagnostics::SourcePosition::new(0, 1, 1),
                            diagnostics::FileId::new(0),
                        )
                    };
                    self.collected_diagnostics.push(diagnostics::Diagnostic {
                        severity: diagnostics::DiagnosticSeverity::Error,
                        code: Some(format!("IMPORT[{}]", name)),
                        message: format!("imported file {}: {}", name, e.message),
                        span,
                        labels: Vec::new(),
                        suggestions: Vec::new(),
                        notes: Vec::new(),
                        help: Vec::new(),
                    });
                }
                false
            }
        }
    }

    /// Try to load an import file from BLADE cache.
    /// Returns true if cache hit (MIR loaded + symbols registered), false if miss.
    fn try_load_import_from_cache(&mut self, filename: &str, source: &str) -> bool {
        // Try to load from BLADE cache
        let (mir, _metadata, symbols, cached_maps) =
            match self.try_load_blade_cached_full(filename, source) {
                Some(data) => data,
                None => return false,
            };

        // We need both type info and cached maps for a full cache restore.
        // Typedef/extern-only imports can produce no MIR at all; for those,
        // symbols-only + empty MIR is enough to avoid redoing TAST/HIR work
        // when the entry module recompiles.
        let (symbols, cached_maps) = match (symbols, cached_maps) {
            (Some(s), Some(m)) => (s, Some(m)),
            (Some(s), None) if mir.functions.is_empty() => (s, None),
            _ => {
                debug!("[BLADE] Cache hit but missing type info/maps: {}", filename);
                return false;
            }
        };

        debug!(
            "[BLADE] Import cache hit: {} ({} functions, {} fields, {} class sizes)",
            filename,
            cached_maps.as_ref().map(|m| m.functions.len()).unwrap_or(0),
            cached_maps.as_ref().map(|m| m.fields.len()).unwrap_or(0),
            cached_maps
                .as_ref()
                .map(|m| m.class_sizes.len())
                .unwrap_or(0)
        );

        // Step 1: Register symbols from type info (restores type system state)
        let registered = self.register_symbols_from_type_info(&symbols);

        let Some(cached_maps) = cached_maps else {
            return true;
        };

        // Step 2: Rebuild MIR-level maps from cached maps using fresh IDs
        self.restore_cached_maps(&cached_maps, &registered);

        // Step 3: Build name-based function map from MIR
        // Use qualified names to avoid collisions (e.g., "current" matching
        // both ArrayIterator.current field and Thread.current method)
        for (func_id, func) in &mir.functions {
            if !func.cfg.blocks.is_empty() {
                // Prefer qualified_name (e.g., "ArrayIterator.hasNext") over bare name ("hasNext")
                let map_name = func.qualified_name.as_deref().unwrap_or(&func.name);
                self.stdlib_function_name_map
                    .insert(map_name.to_string(), *func_id);
            }
        }

        // Step 4: Restore inline var constants from cache
        if !cached_maps.inline_vars.is_empty() {
            self.store_inline_vars(&cached_maps.inline_vars);
        }

        // Step 4.5: Mark this import's SOURCE-DECLARED functions (methods +
        // constructors from cached_maps.functions) as OWNED, exactly like
        // the fresh-compile path does via last_compiled_own_func_ids.
        // Without this, a cache-loaded user module's methods (e.g.
        // KVCache.append) are unprotected at the stdlib merge: the merge
        // deletes/rewrites them, fresh modules lowering against them fall
        // through to a degenerate bare-name extern stub, and the JIT panics
        // at finalize with "can't resolve symbol append" (the AOT link
        // fails on the same bare symbol). This was the touch-entry-file /
        // edit-one-dep mixed-cache crash.
        //
        // Scope strictly to DECLARED functions: marking every non-empty-CFG
        // function also protects generated stdlib wrapper placeholders
        // inside the user module, which then shadow the real stdlib MIR
        // after the merge (first symptom: cached StringMap.get dispatching
        // into a stub — "missing meta key 'general.architecture'").
        // Stdlib files keep placeholder method bodies the stdlib wrappers
        // must replace, so the guard stays user-packages-only, mirroring
        // the fresh path.
        let is_user_package = !filename.contains("haxe-std");
        if is_user_package {
            let import_base: u32 = 100_000 + (self.import_mir_modules.len() as u32 * 10_000);
            for entry in &cached_maps.functions {
                let new_id = crate::ir::IrFunctionId(entry.func_id + import_base);
                self.import_own_func_ids.insert(new_id);
            }
        }

        // Step 5: Renumber and push to import_mir_modules
        self.renumber_and_push_import_mir(mir);

        true
    }

    /// Load a BLADE cached file and return all components including type info and cached maps
    fn try_load_blade_cached_full(
        &self,
        source_path: &str,
        source: &str,
    ) -> Option<(
        IrModule,
        BladeMetadata,
        Option<BladeTypeInfo>,
        Option<BladeCachedMaps>,
    )> {
        if !self.config.enable_cache {
            return None;
        }

        let blade_path = self.blade_cache_path(source_path)?;
        if !blade_path.exists() {
            return None;
        }

        match load_blade(&blade_path) {
            Ok((mir, metadata, symbols, cached_maps)) => {
                let current_hash = self.hash_source_for_config(source);
                let current_build_id = env!("RAYZOR_BUILD_ID");
                if metadata.source_hash != current_hash {
                    debug!("[BLADE] Cache stale (hash mismatch): {}", source_path);
                    None
                } else if metadata.build_id != current_build_id {
                    debug!("[BLADE] Cache stale (build-id mismatch): {}", source_path);
                    None
                } else {
                    Some((mir, metadata, symbols, cached_maps))
                }
            }
            Err(e) => {
                debug!("[BLADE] Cache read error for {}: {}", source_path, e);
                None
            }
        }
    }

    /// Register type system symbols from BladeTypeInfo (for cache restore).
    /// Returns a mapping of class names to their fresh IDs for map reconstruction.
    fn register_symbols_from_type_info(
        &mut self,
        symbols: &BladeTypeInfo,
    ) -> BTreeMap<String, (crate::tast::SymbolId, crate::tast::TypeId, ScopeId)> {
        let mut class_map = BTreeMap::new();

        for class_info in &symbols.classes {
            let symbol_id = self.register_class_from_blade(class_info);
            let qualified_name = if class_info.package.is_empty() {
                class_info.name.clone()
            } else {
                format!("{}.{}", class_info.package.join("."), class_info.name)
            };
            // Get the type ID and scope ID we just created
            if let Some(sym) = self.symbol_table.get_symbol(symbol_id) {
                let type_id = sym.type_id;
                let scope_id = sym.scope_id;
                // Insert both qualified name (haxe.Exception) and simple name (Exception)
                // so BLADE field entries using either convention can be restored
                if !class_info.package.is_empty() {
                    class_map.insert(class_info.name.clone(), (symbol_id, type_id, scope_id));
                }
                class_map.insert(qualified_name, (symbol_id, type_id, scope_id));
            }
        }

        for enum_info in &symbols.enums {
            self.register_enum_from_blade(enum_info);
        }

        for alias_info in &symbols.type_aliases {
            self.register_type_alias_from_blade(alias_info);
        }

        for abstract_info in &symbols.abstracts {
            self.register_abstract_from_blade(abstract_info);
        }

        class_map
    }

    /// Restore MIR-level cross-reference maps from cached data using fresh symbol IDs.
    fn restore_cached_maps(
        &mut self,
        cached_maps: &BladeCachedMaps,
        registered: &BTreeMap<String, (crate::tast::SymbolId, crate::tast::TypeId, ScopeId)>,
    ) {
        use crate::ir::IrFunctionId;

        // Restore function mappings: find method SymbolId in registered class scopes
        for entry in &cached_maps.functions {
            // Stash per-param interface-name info under the *cached* func_id;
            // renumber_and_push_import_mir later remaps the keys. Only store
            // entries that carry at least one name — empty Vecs are no-ops.
            if entry.param_type_names.iter().any(|name| name.is_some()) {
                self.import_function_param_iface_names
                    .insert(IrFunctionId(entry.func_id), entry.param_type_names.clone());
            }

            if entry.is_constructor {
                // Constructors are keyed by class name
                self.import_constructor_name_map
                    .insert(entry.class_name.clone(), IrFunctionId(entry.func_id));
                continue;
            }

            // Look up the class, then find the method symbol in its scope
            if let Some((_class_sym, _class_type, class_scope)) = registered.get(&entry.class_name)
            {
                let method_name_interned = self.string_interner.intern(&entry.method_name);
                if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                    if let Some(method_sym) = scope.get_symbol(method_name_interned) {
                        self.stdlib_function_map
                            .insert(method_sym, IrFunctionId(entry.func_id));
                    }
                }
            }
        }

        // Restore field index mappings
        for entry in &cached_maps.fields {
            if let Some((_class_sym, class_type, class_scope)) = registered.get(&entry.class_name) {
                let field_name_interned = self.string_interner.intern(&entry.field_name);
                if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                    if let Some(field_sym) = scope.get_symbol(field_name_interned) {
                        debug!(
                            "[BLADE_FIELD] Restored {}.{} {:?} -> (TypeId({:?}), index={})",
                            entry.class_name,
                            entry.field_name,
                            field_sym,
                            class_type,
                            entry.field_index
                        );
                        self.import_field_index_map
                            .insert(field_sym, (*class_type, entry.field_index));
                    } else {
                        debug!(
                            "[BLADE_FIELD] MISS: {}.{} not found in scope {:?}",
                            entry.class_name, entry.field_name, class_scope
                        );
                    }
                }
            } else {
                debug!(
                    "[BLADE_FIELD] MISS: class '{}' not in registered map",
                    entry.class_name
                );
            }
        }

        // Restore class allocation sizes
        for (class_name, size) in &cached_maps.class_sizes {
            // Name-based (stable across contexts)
            self.import_class_alloc_sizes_by_name
                .insert(class_name.clone(), *size);
            // SymbolId-based (stable across contexts)
            if let Some((class_sym, _class_type, _)) = registered.get(class_name) {
                self.import_class_alloc_sizes.insert(*class_sym, *size);
            }
        }

        // Derive class sizes from field entries for old caches that lack class_sizes.
        // For each class with fields but no size entry, compute (max_field_index + 1) * 8.
        {
            let mut class_max_idx: BTreeMap<&str, u32> = BTreeMap::new();
            for entry in &cached_maps.fields {
                if !entry.class_name.is_empty() {
                    let cur = class_max_idx.entry(&entry.class_name).or_insert(0);
                    if entry.field_index > *cur {
                        *cur = entry.field_index;
                    }
                }
            }
            for (class_name, max_idx) in &class_max_idx {
                if !self
                    .import_class_alloc_sizes_by_name
                    .contains_key(*class_name)
                {
                    let size = ((*max_idx as u64) + 1) * 8;
                    self.import_class_alloc_sizes_by_name
                        .insert(class_name.to_string(), size);
                    if let Some((class_sym, _class_type, _)) = registered.get(*class_name) {
                        self.import_class_alloc_sizes.insert(*class_sym, size);
                    }
                }
            }
        }

        // Restore interface_vtables entries — these survive across
        // compilations because they're keyed by qualified name. For
        // each (class_qname, iface_qname, method_qnames) entry, look
        // up the SymbolIds in the current context and insert into
        // import_interface_vtables so iface-to-iface casts can
        // emit `haxe_iface_vtable_set_slot` registrations from the
        // user-module's __vtable_init__.
        for entry in &cached_maps.interface_vtables {
            let Some((class_sym, _, _)) = registered.get(&entry.class_name) else {
                continue;
            };
            let Some((iface_sym, _, iface_scope)) = registered.get(&entry.iface_name) else {
                continue;
            };
            let Some(iface_scope) = self.scope_tree.get_scope(*iface_scope) else {
                continue;
            };
            // Resolve each method qname's local name (the trailing
            // segment after the last `.`) back to a SymbolId in the
            // interface's own scope.
            let mut method_syms: Vec<crate::tast::SymbolId> =
                Vec::with_capacity(entry.method_qnames.len());
            let mut all_found = true;
            for qname in &entry.method_qnames {
                let local = qname.rsplit('.').next().unwrap_or(qname.as_str());
                let interned = self.string_interner.intern(local);
                if let Some(method_sym) = iface_scope.get_symbol(interned) {
                    method_syms.push(method_sym);
                } else {
                    all_found = false;
                    break;
                }
            }
            if !all_found {
                continue;
            }
            self.import_interface_vtables
                .insert((*class_sym, *iface_sym), method_syms);
        }

        // Restore interface_method_names: qname → ordered method
        // names. Required so downstream files that pick up a BLADE
        // cached import can still resolve `t.encode(...)` on an
        // interface-typed local — `maybe_materialize_for_call`'s
        // interface-dispatch path keys the vtable slot by method
        // index from this map.
        for entry in &cached_maps.interface_method_names {
            let Some((iface_sym, _, _)) = registered.get(&entry.iface_name) else {
                continue;
            };
            let method_syms: Vec<crate::tast::InternedString> = entry
                .method_names
                .iter()
                .map(|n| self.string_interner.intern(n))
                .collect();
            self.import_interface_method_names
                .insert(*iface_sym, method_syms);
        }

        // Restore interface_method_return_types: (iface, method_name)
        // → return TypeId. Required for cross-context iface method
        // return-type re-resolution (W0014/W0015) on cached imports.
        for entry in &cached_maps.interface_method_return_types {
            let Some((iface_sym, _, _)) = registered.get(&entry.iface_name) else {
                continue;
            };
            let Some((_, return_type_id, _)) = registered.get(&entry.return_type_name) else {
                continue;
            };
            let method_name_interned = self.string_interner.intern(&entry.method_name);
            self.import_interface_method_return_types
                .insert((*iface_sym, method_name_interned), *return_type_id);
        }

        // Restore interface_extends: iface → parent ifaces. Required
        // so iface-to-iface upcasts / dispatch through a parent
        // interface resolves when both ifaces come from cached
        // imports. All-or-nothing — drop the entry if any parent
        // qname is unregistered in the consuming context.
        for entry in &cached_maps.interface_extends {
            let Some((iface_sym, _, _)) = registered.get(&entry.iface_name) else {
                continue;
            };
            let mut parent_syms: Vec<crate::tast::SymbolId> =
                Vec::with_capacity(entry.parent_names.len());
            let mut all_found = true;
            for parent_name in &entry.parent_names {
                if let Some((parent_sym, _, _)) = registered.get(parent_name) {
                    parent_syms.push(*parent_sym);
                } else {
                    all_found = false;
                    break;
                }
            }
            if !all_found {
                continue;
            }
            self.import_interface_extends
                .insert(*iface_sym, parent_syms);
        }

        // Restore property access mappings
        for entry in &cached_maps.properties {
            if let Some((_class_sym, _class_type, class_scope)) = registered.get(&entry.class_name)
            {
                let field_name_interned = self.string_interner.intern(&entry.field_name);
                if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                    if let Some(field_sym) = scope.get_symbol(field_name_interned) {
                        let from_blade = |acc: &BladeAccessor| -> crate::tast::PropertyAccessor {
                            match acc {
                                BladeAccessor::Default => crate::tast::PropertyAccessor::Default,
                                BladeAccessor::Null => crate::tast::PropertyAccessor::Null,
                                BladeAccessor::Never => crate::tast::PropertyAccessor::Never,
                                BladeAccessor::Dynamic => crate::tast::PropertyAccessor::Dynamic,
                                BladeAccessor::Method(n) => crate::tast::PropertyAccessor::Method(
                                    self.string_interner.intern(n),
                                ),
                            }
                        };
                        self.import_property_access_map.insert(
                            field_sym,
                            crate::tast::PropertyAccessInfo {
                                getter: from_blade(&entry.getter),
                                setter: from_blade(&entry.setter),
                            },
                        );
                    }
                }
            }
        }

        // Restore class_type_to_symbol and class_method_symbols mappings
        for (class_name, (class_sym, class_type, class_scope)) in registered {
            self.import_class_type_to_symbol
                .insert(*class_type, *class_sym);
            // Restore class_method_symbols by iterating symbols in the class scope
            if let Some(scope) = self.scope_tree.get_scope(*class_scope) {
                for &method_sym in &scope.symbols {
                    if let Some(sym) = self.symbol_table.get_symbol(method_sym) {
                        self.import_class_method_symbols
                            .insert((*class_sym, sym.name), method_sym);
                    }
                }
            }
        }
    }

    /// Post-load fixup: resolve stale cross-module function references in all import modules.
    /// During renumbering, some refs couldn't be resolved because the target module hadn't
    /// been loaded yet. Now all modules are loaded and stdlib_function_name_map is complete.
    fn fixup_stale_cross_module_refs(&mut self) {
        use crate::ir::{IrInstruction, IrTerminator};

        // Build a set of all valid function IDs across all import modules
        // PLUS the main user file (`mir_modules`).
        //
        // We deliberately exclude `cached_stdlib_mir` here. That cache holds
        // the pre-merge / pre-renumber stdlib MIR — its ids do not exist in
        // the final merged module the codegen will see, and folding them
        // into `all_func_ids` artificially "retains" stale CallDirect
        // targets in the Sweep 1 retain step, which then bypass the
        // name-fallback resolution and surface as missing-function errors
        // at codegen.
        let mut all_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            std::collections::BTreeSet::new();
        for m in &self.import_mir_modules {
            all_func_ids.extend(m.functions.keys().copied());
            all_func_ids.extend(m.extern_functions.keys().copied());
        }
        for m in &self.mir_modules {
            all_func_ids.extend(m.functions.keys().copied());
            all_func_ids.extend(m.extern_functions.keys().copied());
        }

        // Also build a "forward-ref stub → name" map. A stub is an
        // `IrFunction` registered by `register_stdlib_mir_forward_ref` while
        // lowering a user file whose dispatch target wasn't yet compiled
        // (e.g. `Caller.probe` calls `h.findMeta` before `Holder.hx`'s
        // retry pass produced the real findMeta MIR). The stub exists in
        // `module.functions` (so `all_func_ids` contains it) but has
        // exactly one empty entry block with an `Unreachable` terminator
        // — it's a placeholder, not a callable. Once `Holder.hx`'s real
        // findMeta is registered in `stdlib_function_name_map`, we can
        // rewrite any CallDirect to the stub so it targets the real
        // function instead. Without this, the call survives merge as a
        // dispatch into the empty stub and the runtime jumps into
        // uninitialised code (UD2 / SIGILL).
        //
        // Key: name carried by the stub IrFunction (the qualified name
        // passed to `register_stdlib_mir_forward_ref`, e.g.
        // `pkg.Holder.findMeta`). Value: the stub's renumbered func_id.
        // For every IrFunctionId in any loaded module, if the function
        // is an empty forward-ref stub, record its qualified name. The
        // earlier version of this map was keyed BY-NAME (one entry per
        // name, first-found wins) — but when the same stub name was
        // registered in multiple modules (e.g. `string_concat` in both
        // an import module via cached MIR AND a user module via a
        // fresh `register_stdlib_mir_forward_ref` call during user-file
        // lowering), only the first stub's id was retained. Any
        // CallDirect pointing at the OTHER stub's id missed the rewrite
        // and remained pointed at the eventual safety-net trap stub.
        //
        // Keying by ID (every stub id → its name) and looking up the
        // current CallDirect's func_id directly avoids the
        // first-stub-wins miss. The id space is unique per
        // post-renumber session so there's no key collision.
        let mut stub_by_id: std::collections::BTreeMap<
            crate::ir::IrFunctionId,
            (String, crate::ir::IrFunctionSignature),
        > = std::collections::BTreeMap::new();
        // Candidates carry their FULL qualified name (3rd tuple element) so the
        // stub->real match can disambiguate by qualified name. This is essential
        // for constructors: every class's constructor has the bare name "new" and
        // many share the signature `(*void)->void`, so a bare-name+signature match
        // is ambiguous and was silently giving up — leaving e.g. a real
        // `haxe.ds.BalancedTree.new` stranded behind its forward-ref trap stub.
        let mut real_funcs_by_bare_name: std::collections::BTreeMap<
            String,
            Vec<(
                crate::ir::IrFunctionId,
                crate::ir::IrFunctionSignature,
                String,
            )>,
        > = std::collections::BTreeMap::new();

        fn is_empty_forward_ref_stub(func: &crate::ir::IrFunction) -> bool {
            func.cfg.blocks.len() == 1
                && func.cfg.blocks.values().all(|b| {
                    b.instructions.is_empty() && matches!(b.terminator, IrTerminator::Unreachable)
                })
        }

        fn bare_function_name(name: &str) -> &str {
            name.rsplit('.').next().unwrap_or(name)
        }

        fn effective_name(func: &crate::ir::IrFunction) -> String {
            func.qualified_name
                .clone()
                .unwrap_or_else(|| func.name.clone())
        }

        for m in &self.import_mir_modules {
            for (id, func) in &m.functions {
                if is_empty_forward_ref_stub(func) {
                    let name = func
                        .qualified_name
                        .clone()
                        .unwrap_or_else(|| func.name.clone());
                    stub_by_id.insert(*id, (name, func.signature.clone()));
                }
                if !func.cfg.blocks.is_empty() {
                    let qname = func.qualified_name.as_deref().unwrap_or(&func.name);
                    real_funcs_by_bare_name
                        .entry(bare_function_name(qname).to_string())
                        .or_default()
                        .push((*id, func.signature.clone(), qname.to_string()));
                }
            }
        }
        for m in &self.mir_modules {
            for (id, func) in &m.functions {
                if is_empty_forward_ref_stub(func) {
                    let name = func
                        .qualified_name
                        .clone()
                        .unwrap_or_else(|| func.name.clone());
                    stub_by_id.insert(*id, (name, func.signature.clone()));
                }
                if !func.cfg.blocks.is_empty() {
                    let qname = func.qualified_name.as_deref().unwrap_or(&func.name);
                    real_funcs_by_bare_name
                        .entry(bare_function_name(qname).to_string())
                        .or_default()
                        .push((*id, func.signature.clone(), qname.to_string()));
                }
            }
        }

        // ---- Rebind runtime-intrinsic forward-ref stubs to their extern symbol ----
        //
        // `register_stdlib_mir_forward_ref` builds its stub with a *1-block*
        // `Unreachable` cfg — `IrControlFlowGraph::new()` is NOT empty; it seeds
        // one entry block whose default terminator is `Unreachable`. For a
        // stdlib MIR *wrapper* the real body merges in later and replaces it.
        // But some stubs name a C-ABI RUNTIME symbol (`haxe_bytes_sub`,
        // `haxe_bytes_get`, `haxe_string_char_code_at_ptr`, …) registered in
        // the JIT symbol table (runtime/src/plugin_impl.rs) — there is no Haxe
        // body to merge. Codegen keys `is_extern` off `cfg.blocks.is_empty()`,
        // so a 1-block stub is NOT recognised as an extern: it is skipped at
        // definition and the finalize safety net installs a `udf #0xc11f` trap.
        // A call into one (e.g. `GGUFReader.parse` → `Bytes.sub` →
        // `haxe_bytes_sub`) then SIGILLs during GGUF load. The very same symbol
        // also appears as a genuine 0-block extern in another module (that copy
        // binds fine via `declare_function`'s `Import <name>` path) — the stub
        // copy just needs the same shape. Clear the stub's cfg so `is_extern`
        // holds and it binds to the runtime symbol like any extern.
        //
        // Gate strictly: only when the name ALSO exists as a true 0-block
        // extern somewhere (proof it is a runtime-bound symbol) AND has no real
        // (non-stub, non-empty) body anywhere (a real body means "redirect to
        // it", handled by the CallDirect rewrite below — not "bind to symbol").
        let mut extern_only_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        let mut real_body_names: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        {
            let mut scan = |m: &crate::ir::IrModule| {
                for func in m.functions.values() {
                    if func.cfg.blocks.is_empty() {
                        extern_only_names.insert(effective_name(func));
                    } else if !is_empty_forward_ref_stub(func) {
                        real_body_names.insert(effective_name(func));
                    }
                }
                for ext in m.extern_functions.values() {
                    extern_only_names.insert(ext.name.clone());
                }
            };
            for m in &self.import_mir_modules {
                scan(m);
            }
            for m in &self.mir_modules {
                scan(m);
            }
        }
        let mut stub_ids_to_externize: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            std::collections::BTreeSet::new();
        for (id, (name, _sig)) in stub_by_id.iter() {
            if extern_only_names.contains(name) && !real_body_names.contains(name) {
                stub_ids_to_externize.insert(*id);
            }
        }
        // Drop the externized ids from the stub map so the CallDirect rewrite
        // leaves their call sites pointing at the (now extern) function instead
        // of trying to redirect them to another stub.
        for id in &stub_ids_to_externize {
            stub_by_id.remove(id);
        }
        if std::env::var("RAYZOR_DUMP_FN_PTRS").is_ok() && !stub_ids_to_externize.is_empty() {
            eprintln!(
                "[rebind-extern] {} runtime-intrinsic stub(s) rebound to extern symbol",
                stub_ids_to_externize.len()
            );
        }

        // Apply the rewrite to BOTH import_mir_modules and mir_modules
        // (the user modules). A user-file CallDirect like `Sys.println(...
        // + counter)` lowers to a CallDirect targeting the stub
        // `string_concat` registered via `register_stdlib_mir_forward_ref`
        // — when that stub's renumbered id lands on a value the codegen
        // backend's safety net then traps (see
        // bugs_sys_call_in_generation_method's continuation), the user-
        // module CallDirect site needs name-based resolution to point at
        // the real stdlib impl. The previous version rewrote only the
        // import side, leaving user CallDirects pointing at stubs that
        // would never get a body.
        let stub_by_id = &stub_by_id;
        let real_funcs_by_bare_name = &real_funcs_by_bare_name;
        let stdlib_map = &self.stdlib_function_name_map;
        let all_func_ids = &all_func_ids;
        fn signatures_match(
            a: &crate::ir::IrFunctionSignature,
            b: &crate::ir::IrFunctionSignature,
        ) -> bool {
            a.calling_convention == b.calling_convention
                && a.return_type == b.return_type
                && a.uses_sret == b.uses_sret
                && a.parameters.len() == b.parameters.len()
                && a.parameters
                    .iter()
                    .zip(b.parameters.iter())
                    .all(|(pa, pb)| pa.ty == pb.ty && pa.by_ref == pb.by_ref)
        }
        fn unique_bare_match(
            name: &str,
            sig: &crate::ir::IrFunctionSignature,
            real_funcs_by_bare_name: &std::collections::BTreeMap<
                String,
                Vec<(
                    crate::ir::IrFunctionId,
                    crate::ir::IrFunctionSignature,
                    String,
                )>,
            >,
            skip_id: Option<crate::ir::IrFunctionId>,
        ) -> Option<crate::ir::IrFunctionId> {
            let bare_name = bare_function_name(name);
            let candidates = real_funcs_by_bare_name.get(bare_name)?;
            // Qualified-name disambiguation FIRST. Constructors all share the bare
            // name "new", so bare+sig is ambiguous across classes; prefer a real
            // candidate whose FULL qualified name equals the stub's (the real
            // `haxe.ds.BalancedTree.new` rather than some other class's `new`).
            // The qname pins the class; the signature pins the overload. Take the
            // first such match — duplicates of one qname+sig are interchangeable.
            if let Some(real_id) = candidates.iter().find_map(|(cid, csig, cqname)| {
                if Some(*cid) != skip_id && cqname == name && signatures_match(csig, sig) {
                    Some(*cid)
                } else {
                    None
                }
            }) {
                return Some(real_id);
            }
            // Fall back to a UNIQUE bare-name + signature match.
            let mut matches = candidates
                .iter()
                .filter_map(|(candidate_id, candidate_sig, _)| {
                    if Some(*candidate_id) == skip_id {
                        return None;
                    }
                    if signatures_match(candidate_sig, sig) {
                        Some(*candidate_id)
                    } else {
                        None
                    }
                });
            let real_id = matches.next()?;
            if matches.next().is_none() {
                Some(real_id)
            } else {
                None
            }
        }

        let mut rewrite_module = |module: &mut crate::ir::IrModule| {
            let ext_names = module.external_function_names.clone();
            let ext_sigs: std::collections::BTreeMap<
                crate::ir::IrFunctionId,
                crate::ir::IrFunctionSignature,
            > = module
                .extern_functions
                .iter()
                .map(|(id, ext)| (*id, ext.signature.clone()))
                .collect();
            let ext_decl_by_id: std::collections::BTreeMap<
                crate::ir::IrFunctionId,
                (String, crate::ir::IrFunctionSignature),
            > = module
                .extern_functions
                .iter()
                .map(|(id, ext)| (*id, (ext.name.clone(), ext.signature.clone())))
                .collect();
            for func in module.functions.values_mut() {
                for block in func.cfg.blocks.values_mut() {
                    for inst in &mut block.instructions {
                        match inst {
                            IrInstruction::CallDirect { func_id, .. }
                            | IrInstruction::FunctionRef { func_id, .. }
                            | IrInstruction::MakeClosure { func_id, .. } => {
                                // If the cached MIR recorded this call site
                                // as an external reference (ext_names has an
                                // entry for the func_id), ALWAYS resolve it
                                // by name. The previous "only fix when id
                                // isn't valid" rule had a silent failure
                                // mode: the cached id from session A could
                                // happen to collide with a *different*
                                // function's id in session B (both at
                                // module-index 9, say) — the call would then
                                // dispatch into an unrelated function and
                                // SIGILL at runtime instead of resolving to
                                // the named target. Name-first resolution
                                // makes the fixup robust against any
                                // import-order-dependent id assignment.
                                if let Some(name) = ext_names.get(func_id) {
                                    if let Some(&current_id) = stdlib_map.get(name) {
                                        *func_id = current_id;
                                        continue;
                                    }
                                    if let Some(sig) = ext_sigs.get(func_id) {
                                        if let Some(real_id) = unique_bare_match(
                                            name,
                                            sig,
                                            real_funcs_by_bare_name,
                                            None,
                                        ) {
                                            *func_id = real_id;
                                        }
                                    }
                                    continue;
                                }
                                if let Some((name, sig)) = ext_decl_by_id.get(func_id) {
                                    if let Some(&current_id) = stdlib_map.get(name.as_str()) {
                                        *func_id = current_id;
                                        continue;
                                    }
                                    if let Some(real_id) =
                                        unique_bare_match(name, sig, real_funcs_by_bare_name, None)
                                    {
                                        *func_id = real_id;
                                    }
                                    continue;
                                }

                                // Not an external reference: leave the id
                                // alone if it's already valid. If not valid
                                // and the cached id corresponds to a known
                                // forward-ref stub by name, redirect to the
                                // real implementation. Otherwise leave as-is
                                // (will fault at runtime, surfacing the
                                // missing-impl error rather than silently
                                // dispatching to an unrelated function).
                                if all_func_ids.contains(func_id) {
                                    if let Some((stub_name, stub_sig)) = stub_by_id.get(func_id) {
                                        if let Some(&real_id) = stdlib_map.get(stub_name.as_str()) {
                                            if real_id != *func_id {
                                                *func_id = real_id;
                                            }
                                        } else {
                                            if let Some(real_id) = unique_bare_match(
                                                stub_name,
                                                stub_sig,
                                                real_funcs_by_bare_name,
                                                Some(*func_id),
                                            ) {
                                                *func_id = real_id;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        };

        // Clear the cfg of every runtime-intrinsic stub picked above so codegen
        // sees `is_extern` (empty blocks) and binds it to the runtime symbol.
        let externize = |module: &mut crate::ir::IrModule| {
            for (id, func) in module.functions.iter_mut() {
                if stub_ids_to_externize.contains(id) {
                    func.cfg.blocks.clear();
                }
            }
        };
        for module in &mut self.import_mir_modules {
            externize(module);
            rewrite_module(module);
        }
        // `mir_modules` is `Vec<Arc<IrModule>>` (sharable across the
        // pipeline). `Arc::make_mut` clones the inner module on demand if
        // anyone else holds a handle; in the codegen-prep window only the
        // CompilationContext owns these handles, so this is in-place.
        for module in self.mir_modules.iter_mut() {
            let m = std::sync::Arc::make_mut(module);
            externize(m);
            rewrite_module(m);
        }
    }

    /// Post-load fixup: rewrite stale constructor func_ids in
    /// `import_constructor_name_map`.
    ///
    /// `restore_cached_maps` seeds this map from BLADE cache with func_ids
    /// from the *saving* session — those ids encode that session's
    /// `import_base + local_id`, which doesn't necessarily match the
    /// current session's id assignment. Plain renumbering inside
    /// `renumber_and_push_import_mir` only rewrites entries whose ids live
    /// in *that* module's id_map, so a constructor cached at e.g. fn200007
    /// (StringBuf in a session where it was at import index 10) stays at
    /// fn200007 even when the current session puts StringBuf at index 2
    /// (real id fn120007). The user file then lowers `new StringBuf()` to
    /// `call fn200007` — an id that no longer exists — and SIGILLs.
    ///
    /// `stdlib_function_name_map` does carry the current ids of every
    /// merged function keyed by qualified name (`StringBuf.new`), so we
    /// rewrite each constructor map entry by name once all modules are in.
    fn fixup_stale_constructor_ids(&mut self) {
        // Snapshot the qualified lookup so we don't borrow self twice.
        let map_snapshot: std::collections::BTreeMap<String, crate::ir::IrFunctionId> =
            self.stdlib_function_name_map.clone();
        for (class_name, func_id) in self.import_constructor_name_map.iter_mut() {
            let qualified = format!("{}.new", class_name);
            if let Some(&current_id) = map_snapshot.get(&qualified) {
                if *func_id != current_id {
                    if let Some(count) = self.import_constructor_param_counts.remove(func_id) {
                        self.import_constructor_param_counts
                            .entry(current_id)
                            .or_insert(count);
                    } else if let Some(params) = self.import_function_param_types.get(&current_id) {
                        self.import_constructor_param_counts
                            .entry(current_id)
                            .or_insert(params.len());
                    }
                    *func_id = current_id;
                }
            }
        }
    }

    /// Post-load fixup: rewrite stale method func_ids in
    /// `stdlib_function_map` (SymbolId → IrFunctionId) by reverse-mapping
    /// each symbol back to its qualified name and re-resolving via
    /// `stdlib_function_name_map`.
    ///
    /// Same hazard as `fixup_stale_constructor_ids` but for non-constructors:
    /// `restore_cached_maps` seeds method-map values with the *cached*
    /// (pre-renumber) func_id; `renumber_and_push_import_mir`'s id_map-keyed
    /// rewrite only fires for entries whose ids live in the module currently
    /// being renumbered. Cross-session id drift caused by changes in the
    /// import-load order (e.g. adding/removing a top-level import shifts the
    /// 10_000-block assignment for every subsequent module) leaves
    /// stdlib_function_map pointing at a slot that no longer exists.
    ///
    /// `stdlib_function_name_map` carries the *current* ids of every merged
    /// function keyed by qualified name (e.g. "LlamaModel.forwardIds"), so we
    /// re-key each entry by its symbol's qualified name once all modules are
    /// loaded. Entries whose qualified name can't be reconstructed, or whose
    /// qualified name isn't present in the name map, are left untouched —
    /// the existing `fixup_stale_cross_module_refs` MIR walk catches stale
    /// CallDirect targets that survive here.
    fn fixup_stale_method_ids(&mut self) {
        let map_snapshot: std::collections::BTreeMap<String, crate::ir::IrFunctionId> =
            self.stdlib_function_name_map.clone();
        // Snapshot the (sym, id) pairs so we don't borrow self.symbol_table
        // while mutating self.stdlib_function_map.
        let entries: Vec<(crate::tast::SymbolId, crate::ir::IrFunctionId)> = self
            .stdlib_function_map
            .iter()
            .map(|(s, f)| (*s, *f))
            .collect();
        for (sym_id, _) in entries {
            // Resolve the symbol's qualified name. Symbols populated by
            // `restore_cached_maps` set qualified_name to the interned form
            // "Class.method"; freshly compiled methods do the same via
            // ast_lowering. Symbols without a qualified_name are not
            // safely re-resolvable by name and are skipped.
            let qname = {
                let Some(sym) = self.symbol_table.get_symbol(sym_id) else {
                    continue;
                };
                let Some(qn_interned) = sym.qualified_name else {
                    continue;
                };
                match self.string_interner.get(qn_interned) {
                    Some(s) => s.to_string(),
                    None => continue,
                }
            };
            if let Some(&current_id) = map_snapshot.get(&qname) {
                if let Some(slot) = self.stdlib_function_map.get_mut(&sym_id) {
                    if *slot != current_id {
                        *slot = current_id;
                    }
                }
            }
        }
    }

    /// Renumber import MIR function IDs to avoid collisions and push to import_mir_modules
    fn renumber_and_push_import_mir(&mut self, mut import_mir: IrModule) {
        use crate::ir::{IrFunctionId, IrGlobalId, IrInstruction};

        let import_base: u32 = 100_000 + (self.import_mir_modules.len() as u32 * 10_000);

        // Build old→new ID mapping (include both functions and extern_functions)
        let mut id_map: std::collections::BTreeMap<IrFunctionId, IrFunctionId> =
            std::collections::BTreeMap::new();
        for old_id in import_mir.functions.keys() {
            id_map.insert(*old_id, IrFunctionId(old_id.0 + import_base));
        }
        for old_id in import_mir.extern_functions.keys() {
            id_map
                .entry(*old_id)
                .or_insert(IrFunctionId(old_id.0 + import_base));
        }

        // Globals get the same disjoint-range treatment as functions: every
        // module numbers its globals densely from 0, so an unrenumbered
        // import's LoadGlobal/StoreGlobal aliases the main module's slots
        // 1:1 (observed: a user static read back Math's LN2 — each module's
        // __init__ wrote the same @g0/@g1). Backends key global storage by
        // raw id value, so sparse renumbered ids are safe.
        let mut global_id_map: std::collections::BTreeMap<IrGlobalId, IrGlobalId> =
            std::collections::BTreeMap::new();
        for old_id in import_mir.globals.keys() {
            global_id_map.insert(*old_id, IrGlobalId(old_id.0 + import_base));
        }

        // Renumber functions
        let old_functions: std::collections::BTreeMap<_, _> =
            std::mem::take(&mut import_mir.functions);
        for (old_id, mut func) in old_functions {
            let new_id = *id_map.get(&old_id).unwrap();
            func.id = new_id;

            // Update internal CallDirect/FunctionRef/MakeClosure. ext_names
            // takes priority over id_map: if the cached MIR recorded this
            // site as an external reference, resolve by name regardless of
            // whether the cached id happens to alias something in this
            // module's old id space. (Same robustness argument as the
            // fixup pass: integer ids are not stable across compilation
            // sessions with different import orderings.)
            for block in func.cfg.blocks.values_mut() {
                for inst in &mut block.instructions {
                    match inst {
                        IrInstruction::CallDirect { func_id, .. }
                        | IrInstruction::FunctionRef { func_id, .. }
                        | IrInstruction::MakeClosure { func_id, .. } => {
                            if let Some(name) = import_mir.external_function_names.get(func_id) {
                                if let Some(&current_id) = self.stdlib_function_name_map.get(name) {
                                    *func_id = current_id;
                                }
                                // If name lookup fails the post-pass
                                // fixup_stale_cross_module_refs will retry
                                // once all modules are loaded.
                            } else if let Some(new_func_id) = id_map.get(func_id) {
                                *func_id = *new_func_id;
                            }
                        }
                        IrInstruction::LoadGlobal { global_id, .. }
                        | IrInstruction::StoreGlobal { global_id, .. } => {
                            if let Some(&new_gid) = global_id_map.get(global_id) {
                                *global_id = new_gid;
                            }
                        }
                        _ => {}
                    }
                }
            }

            import_mir.functions.insert(new_id, func);
        }

        // Renumber extern_functions
        let old_externs: std::collections::BTreeMap<_, _> =
            std::mem::take(&mut import_mir.extern_functions);
        for (old_id, mut efunc) in old_externs {
            let new_id = id_map
                .get(&old_id)
                .copied()
                .unwrap_or(IrFunctionId(old_id.0 + import_base));
            efunc.id = new_id;
            import_mir.extern_functions.insert(new_id, efunc);
        }

        // Renumber the globals table to match the rewritten instructions.
        let old_globals: std::collections::BTreeMap<_, _> = std::mem::take(&mut import_mir.globals);
        for (old_id, mut g) in old_globals {
            let new_id = *global_id_map.get(&old_id).unwrap();
            g.id = new_id;
            import_mir.globals.insert(new_id, g);
        }

        // Re-key the module's name records to the renumbered ids. These
        // entries are the qualified-name ground truth for every later
        // name-first repair (fixup passes, merge verification); leaving them
        // keyed by pre-renumber ids detaches them from the instructions and
        // silently degrades cross-module resolution to raw-number trust.
        let old_ext_names = std::mem::take(&mut import_mir.external_function_names);
        for (old_id, name) in old_ext_names {
            let new_id = id_map.get(&old_id).copied().unwrap_or(old_id);
            import_mir.external_function_names.insert(new_id, name);
        }

        // Update all accumulated maps to point to renumbered IDs
        for (_sym, func_id) in self.stdlib_function_map.iter_mut() {
            if let Some(&new_id) = id_map.get(func_id) {
                *func_id = new_id;
            }
        }
        for (_name, func_id) in self.stdlib_function_name_map.iter_mut() {
            if let Some(&new_id) = id_map.get(func_id) {
                *func_id = new_id;
            }
        }
        for (_name, func_id) in self.import_constructor_name_map.iter_mut() {
            if let Some(&new_id) = id_map.get(func_id) {
                *func_id = new_id;
            }
        }
        // Re-key import_function_param_iface_names from pre-renumber to
        // post-renumber func_ids. Drain into a temp so we don't iterate
        // and mutate the same map.
        let stale_iface_names = std::mem::take(&mut self.import_function_param_iface_names);
        for (old_id, names) in stale_iface_names {
            let new_id = id_map.get(&old_id).copied().unwrap_or(old_id);
            self.import_function_param_iface_names.insert(new_id, names);
        }

        // Keep MIR-lowering lookup inputs incrementally accumulated. The old
        // path rebuilt these maps by scanning every previous import module for
        // every next import/user file, which made cold source compilation scale
        // quadratically on large graphs like nue/llama-chat.
        let constructor_ids: std::collections::BTreeSet<_> =
            self.import_constructor_name_map.values().copied().collect();
        for (func_id, func) in &import_mir.functions {
            self.import_function_param_types.insert(
                *func_id,
                func.signature
                    .parameters
                    .iter()
                    .map(|p| p.ty.clone())
                    .collect(),
            );
            if constructor_ids.contains(func_id) {
                self.import_constructor_param_counts
                    .entry(*func_id)
                    .or_insert(func.signature.parameters.len());
            }
        }
        for global in import_mir.globals.values() {
            self.import_external_globals
                .entry(global.name.clone())
                .or_insert((global.id, global.ty.clone()));
        }

        // Populate `stdlib_function_name_map` with this import's functions so
        // downstream callers can resolve them by qualified name. The cache-hit
        // path in `try_load_blade_cached_full` already does this (line ~2988);
        // the fresh-compile path (this function) used to skip it, leaving
        // user-package methods unreachable by name from callers in other
        // files. Manifested as `loader.someMethod(...)` silently lowering to
        // `unreachable` even though the function existed in MIR — the
        // resolver in `hir_to_mir.rs::resolve_function_id_with_qualified_fallback`
        // falls through to this map as a last resort.
        for (func_id, func) in &import_mir.functions {
            if func.cfg.blocks.is_empty() {
                continue;
            }
            let map_name = func.qualified_name.as_deref().unwrap_or(&func.name);
            // Don't overwrite an existing entry — first writer wins to keep
            // BLADE-cache and fresh-compile entries from clobbering each other.
            self.stdlib_function_name_map
                .entry(map_name.to_string())
                .or_insert(*func_id);
        }

        self.import_mir_modules.push(import_mir);
    }

    /// Load a single file on-demand for import resolution (legacy - uses retry pattern)
    /// Prefer load_imports_efficiently for batch loading
    pub fn load_import_file(&mut self, qualified_path: &str) -> Result<(), String> {
        self.load_import_file_recursive(qualified_path, 0)
    }

    /// Internal recursive function for loading files with dependency resolution
    /// Max depth prevents infinite loops in circular dependencies
    fn load_import_file_recursive(
        &mut self,
        qualified_path: &str,
        depth: usize,
    ) -> Result<(), String> {
        const MAX_DEPTH: usize = 10;

        if depth > MAX_DEPTH {
            return Err(format!(
                "Maximum dependency depth ({}) exceeded for: {}",
                MAX_DEPTH, qualified_path
            ));
        }

        // Resolve the qualified path to a filesystem path
        // If not found directly, try common stdlib package prefixes for unqualified names
        let file_path = if let Some(path) = self
            .namespace_resolver
            .resolve_qualified_path_to_file(qualified_path)
        {
            path
        } else if !qualified_path.contains('.') {
            // Unqualified name - try common stdlib packages
            let prefixes = vec![
                "haxe.iterators",
                "haxe.ds",
                "haxe",
                "sys.thread",
                "sys",
                "haxe.exceptions",
                "haxe.io",
            ];
            let mut found_path = None;
            for prefix in &prefixes {
                let qualified = format!("{}.{}", prefix, qualified_path);
                if let Some(path) = self
                    .namespace_resolver
                    .resolve_qualified_path_to_file(&qualified)
                {
                    found_path = Some(path);
                    break;
                }
            }
            found_path.ok_or_else(|| format!("Could not resolve import: {}", qualified_path))?
        } else {
            // Check if the file is already loaded (resolve returns None for loaded files)
            if self
                .namespace_resolver
                .is_qualified_path_loaded(qualified_path)
            {
                return Ok(());
            }
            return Err(format!("Could not resolve import: {}", qualified_path));
        };

        // Skip if already loaded - this prevents redundant re-compilation
        if self.namespace_resolver.is_file_loaded(&file_path) {
            return Ok(());
        }

        // Mark as loaded BEFORE compiling to prevent recursive loading
        self.namespace_resolver.mark_file_loaded(file_path.clone());

        // Read the file
        let source = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read {:?}: {}", file_path, e))?;

        let filename = file_path.to_string_lossy().to_string();

        // Try to compile - if it fails due to missing dependencies, extract and load them
        match self.compile_file_with_shared_state(&filename, &source) {
            Ok(typed_file) => {
                debug!(
                    "  ✓ Successfully compiled and registered: {}",
                    qualified_path
                );
                // Store typedef files so they're included in HIR conversion
                if !typed_file.type_aliases.is_empty() {
                    trace!(
                        "    (contains {} type aliases)",
                        typed_file.type_aliases.len()
                    );
                }

                // Check if any type aliases have Placeholder targets that need to be loaded
                // This handles cases like `typedef Bytes = rayzor.Bytes` where rayzor.Bytes hasn't been loaded yet
                let mut placeholder_targets = Vec::new();
                {
                    let type_table = self.type_table.borrow();
                    for alias in &typed_file.type_aliases {
                        if let Some(target_info) = type_table.get(alias.target_type) {
                            if let crate::tast::TypeKind::Placeholder { name } = &target_info.kind {
                                if let Some(placeholder_name) = self.string_interner.get(*name) {
                                    trace!(
                                        "    Found typedef with Placeholder target: {}",
                                        placeholder_name
                                    );
                                    placeholder_targets.push(placeholder_name.to_string());
                                }
                            }
                        }
                    }
                }

                // If we found Placeholder targets, try to load them and retry
                if !placeholder_targets.is_empty() {
                    let mut any_loaded = false;
                    for target in &placeholder_targets {
                        if let Ok(_) = self.load_import_file_recursive(target, depth + 1) {
                            debug!("    ✓ Loaded typedef target: {}", target);
                            any_loaded = true;
                        }
                    }

                    if any_loaded {
                        // Retry compilation after loading typedef targets
                        debug!(
                            "  Retrying compilation of {} after loading typedef targets...",
                            qualified_path
                        );
                        match self.compile_file_with_shared_state(&filename, &source) {
                            Ok(recompiled_file) => {
                                self.loaded_stdlib_typed_files.push(recompiled_file);
                                return Ok(());
                            }
                            Err(_) => {
                                // Fall through and push the original typed_file
                            }
                        }
                    }
                }

                self.loaded_stdlib_typed_files.push(typed_file);
                Ok(())
            }
            Err(errors) => {
                // Extract UnresolvedType errors and try to load those dependencies
                let mut missing_types = Vec::new();
                for error in &errors {
                    if let Some(type_name) =
                        Self::extract_unresolved_type_from_error(&error.message)
                    {
                        // Skip generic type parameters and built-in typedefs
                        if !Self::is_generic_type_parameter(&type_name)
                            && !self.failed_type_loads.contains(&type_name)
                        {
                            missing_types.push(type_name);
                        }
                    }
                }

                // If we found missing types, try to load them recursively
                if !missing_types.is_empty() {
                    debug!(
                        "  Detected {} missing dependencies for {}: {:?}",
                        missing_types.len(),
                        qualified_path,
                        missing_types
                    );

                    let mut load_success = false;
                    for missing_type in &missing_types {
                        // Check if this looks like a field reference (e.g., "haxe.SysTools.winMetaCharacters")
                        // If so, extract just the class part (e.g., "haxe.SysTools")
                        let type_to_load = if let Some(last_dot) = missing_type.rfind('.') {
                            let after_dot = &missing_type[last_dot + 1..];
                            // If the part after the last dot starts with lowercase, it's likely a field
                            if after_dot
                                .chars()
                                .next()
                                .map(|c| c.is_lowercase())
                                .unwrap_or(false)
                            {
                                &missing_type[..last_dot]
                            } else {
                                missing_type.as_str()
                            }
                        } else {
                            missing_type.as_str()
                        };

                        // Try loading with the (possibly adjusted) name first
                        let loaded = if let Ok(_) =
                            self.load_import_file_recursive(type_to_load, depth + 1)
                        {
                            debug!("    ✓ Loaded dependency: {}", type_to_load);
                            true
                        } else if !type_to_load.contains('.') {
                            // If unqualified name failed, try with common stdlib packages
                            let prefixes = vec!["haxe.exceptions.", "haxe.io.", "haxe.ds."];
                            let mut prefix_loaded = false;
                            for prefix in prefixes {
                                let qualified = format!("{}{}", prefix, type_to_load);
                                if let Ok(_) =
                                    self.load_import_file_recursive(&qualified, depth + 1)
                                {
                                    debug!(
                                        "    ✓ Loaded dependency: {} (as {})",
                                        type_to_load, qualified
                                    );
                                    prefix_loaded = true;
                                    break;
                                }
                            }
                            prefix_loaded
                        } else {
                            false
                        };

                        if loaded {
                            load_success = true;
                        } else {
                            debug!("    ✗ Could not load dependency: {}", missing_type);
                            self.failed_type_loads.insert(missing_type.clone());
                        }
                    }

                    // If we successfully loaded at least one dependency, retry compilation
                    if load_success {
                        debug!(
                            "  Retrying compilation of {} after loading dependencies...",
                            qualified_path
                        );
                        match self.compile_file_with_shared_state(&filename, &source) {
                            Ok(typed_file) => {
                                // Store typedef files so they're included in HIR conversion
                                if !typed_file.type_aliases.is_empty() {
                                    trace!(
                                        "    (contains {} type aliases after retry)",
                                        typed_file.type_aliases.len()
                                    );
                                }

                                // Check if any type aliases have Placeholder targets that need to be loaded
                                // This handles cases like `typedef Bytes = rayzor.Bytes` where rayzor.Bytes hasn't been loaded yet
                                let mut placeholder_targets = Vec::new();
                                {
                                    let type_table = self.type_table.borrow();
                                    for alias in &typed_file.type_aliases {
                                        if let Some(target_info) = type_table.get(alias.target_type)
                                        {
                                            if let crate::tast::TypeKind::Placeholder { name } =
                                                &target_info.kind
                                            {
                                                if let Some(placeholder_name) =
                                                    self.string_interner.get(*name)
                                                {
                                                    trace!("    Found typedef with Placeholder target (after deps): {}", placeholder_name);
                                                    placeholder_targets
                                                        .push(placeholder_name.to_string());
                                                }
                                            }
                                        }
                                    }
                                }

                                // If we found Placeholder targets, try to load them and retry again
                                if !placeholder_targets.is_empty() {
                                    let mut any_loaded = false;
                                    for target in &placeholder_targets {
                                        if let Ok(_) =
                                            self.load_import_file_recursive(target, depth + 1)
                                        {
                                            debug!(
                                                "    ✓ Loaded typedef target (after deps): {}",
                                                target
                                            );
                                            any_loaded = true;
                                        }
                                    }

                                    if any_loaded {
                                        // Retry compilation after loading typedef targets
                                        debug!("  Retrying compilation of {} after loading typedef targets...", qualified_path);
                                        match self
                                            .compile_file_with_shared_state(&filename, &source)
                                        {
                                            Ok(recompiled_file) => {
                                                self.loaded_stdlib_typed_files
                                                    .push(recompiled_file);
                                                return Ok(());
                                            }
                                            Err(_) => {
                                                // Fall through and push the original typed_file
                                            }
                                        }
                                    }
                                }

                                self.loaded_stdlib_typed_files.push(typed_file);
                                return Ok(());
                            }
                            Err(errors) => {
                                // Check if any errors are UnresolvedType that we can try to load
                                let mut additional_missing = Vec::new();
                                for error in &errors {
                                    if let Some(type_name) =
                                        Self::extract_unresolved_type_from_error(&error.message)
                                    {
                                        if !Self::is_generic_type_parameter(&type_name)
                                            && !self.failed_type_loads.contains(&type_name)
                                        {
                                            additional_missing.push(type_name);
                                        }
                                    }
                                }

                                if !additional_missing.is_empty() {
                                    let mut loaded_any = false;
                                    for missing in &additional_missing {
                                        if let Ok(_) =
                                            self.load_import_file_recursive(missing, depth + 1)
                                        {
                                            debug!(
                                                "    ✓ Loaded additional dependency: {}",
                                                missing
                                            );
                                            loaded_any = true;
                                        }
                                    }

                                    if loaded_any {
                                        // Try one more time
                                        debug!("  Retrying compilation of {} after loading additional dependencies...", qualified_path);
                                        match self
                                            .compile_file_with_shared_state(&filename, &source)
                                        {
                                            Ok(final_file) => {
                                                self.loaded_stdlib_typed_files.push(final_file);
                                                return Ok(());
                                            }
                                            Err(final_errors) => {
                                                let error_msgs: Vec<String> = final_errors
                                                    .iter()
                                                    .map(|e| e.message.clone())
                                                    .collect();
                                                return Err(format!("Errors compiling {} (after loading additional dependencies): {}", filename, error_msgs.join(", ")));
                                            }
                                        }
                                    }
                                }

                                let error_msgs: Vec<String> =
                                    errors.iter().map(|e| e.message.clone()).collect();
                                return Err(format!(
                                    "Errors compiling {} (after loading dependencies): {}",
                                    filename,
                                    error_msgs.join(", ")
                                ));
                            }
                        }
                    }
                }

                // No missing types found or couldn't load them - return original error
                let error_msgs: Vec<String> = errors.iter().map(|e| e.message.clone()).collect();
                Err(format!(
                    "Errors compiling {}: {}",
                    filename,
                    error_msgs.join(", ")
                ))
            }
        }
    }

    /// Extract type name from UnresolvedType error messages
    /// Returns Some(type_name) if this is an UnresolvedType error, None otherwise
    fn extract_unresolved_type_from_error(error_msg: &str) -> Option<String> {
        // Match pattern: "UnresolvedType { type_name: \"SomeType\", ..."
        if let Some(type_name_start) = error_msg.find("type_name: \"") {
            let after_marker = &error_msg[type_name_start + 12..]; // 12 = length of 'type_name: "'
            if let Some(end) = after_marker.find('"') {
                return Some(after_marker[..end].to_string());
            }
        }
        // Match pattern: "Cannot find type 'SomeType'"
        if let Some(start) = error_msg.find("Cannot find type '") {
            let after_marker = &error_msg[start + 18..]; // 18 = length of "Cannot find type '"
            if let Some(end) = after_marker.find('\'') {
                return Some(after_marker[..end].to_string());
            }
        }
        None
    }

    /// Check if a type name looks like a generic type parameter
    /// Returns true for single letters (T, K, V) or common parameter patterns
    fn is_generic_type_parameter(type_name: &str) -> bool {
        // Single uppercase letter
        if type_name.len() == 1
            && type_name
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        {
            return true;
        }
        // Common generic parameter patterns and StdTypes prelude names.
        if Self::is_stdtypes_prelude_type_name(type_name) {
            return true;
        }
        matches!(type_name, "Key" | "Value" | "Item" | "Element")
    }

    fn is_stdtypes_prelude_type_name(type_name: &str) -> bool {
        is_stdtypes_ambient_name(type_name)
    }

    fn is_bare_stdtypes_prelude_dependency(dep: &str) -> bool {
        !dep.contains('.') && Self::is_stdtypes_prelude_type_name(dep)
    }

    /// Pre-register type declarations from a file without full compilation
    /// This is the first pass that registers class/interface/enum names in the namespace
    /// so they can be referenced by other files during full compilation
    fn pre_register_file_types(&mut self, filename: &str, source: &str) -> Result<(), String> {
        use crate::tast::ast_lowering::AstLowering;

        let ast_file = self
            .parse_file(filename, source)
            .map_err(|e| format!("Parse error in {}: {}", filename, e))?;

        // Create a temporary AstLowering instance just for pre-registration
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Pre-register only - call the pre_register_file method
        lowering
            .pre_register_file(&ast_file)
            .map_err(|e| format!("Pre-registration error in {}: {:?}", filename, e))?;

        Ok(())
    }

    /// Register only enum declarations from source into the symbol table.
    ///
    /// Used when loading from BLADE cache — the cached MIR has the compiled code
    /// but the symbol table needs enum declarations registered so that user code
    /// can resolve imported enum types and their variants.
    fn register_enums_from_source(&mut self, filename: &str, source: &str) {
        use crate::tast::ast_lowering::AstLowering;

        let ast_file = match self.parse_file(filename, source) {
            Ok(f) => f,
            Err(_) => return,
        };
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Set package context from the parsed file
        if let Some(ref pkg) = ast_file.package {
            lowering.set_package_from_parts(&pkg.path);
        }

        // Lower enum and abstract declarations from cached files.
        // Class registration must go through the normal TAST pipeline
        // to avoid overwriting user imports.
        for decl in &ast_file.declarations {
            match decl {
                parser::TypeDeclaration::Enum(enum_decl) => {
                    let _ = lowering.lower_enum_declaration_public(enum_decl);
                }
                parser::TypeDeclaration::Abstract(_) => {
                    // Pre-register abstract declarations so import resolution
                    // creates Abstract symbols instead of Class placeholders
                    let _ = lowering.pre_register_declaration(decl);
                }
                _ => {}
            }
        }
    }

    /// Combined pre-register types + enum registration from a single parse.
    /// Eliminates the double-parse that occurred when pre_register_file_types
    /// and register_enums_from_source were called separately.
    fn pre_register_and_enums_from_source(
        &mut self,
        filename: &str,
        source: &str,
    ) -> Result<(), String> {
        use crate::tast::ast_lowering::AstLowering;

        let ast_file = self
            .parse_file(filename, source)
            .map_err(|e| format!("Parse error in {}: {}", filename, e))?;
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Pre-register types (classes, interfaces, enums, abstracts)
        lowering
            .pre_register_file(&ast_file)
            .map_err(|e| format!("Pre-registration error in {}: {:?}", filename, e))?;

        // Also register enums and abstracts from the same parsed AST
        if let Some(ref pkg) = ast_file.package {
            lowering.set_package_from_parts(&pkg.path);
        }
        for decl in &ast_file.declarations {
            match decl {
                parser::TypeDeclaration::Enum(enum_decl) => {
                    let _ = lowering.lower_enum_declaration_public(enum_decl);
                }
                parser::TypeDeclaration::Abstract(_) => {
                    let _ = lowering.pre_register_declaration(decl);
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Load global import.hx files
    /// These are processed AFTER stdlib but BEFORE user files
    /// They provide global imports available to all user code
    pub fn load_global_imports(&mut self) -> Result<(), String> {
        use std::fs;

        for import_path in &self.config.global_import_hx_files.clone() {
            let source = fs::read_to_string(import_path)
                .map_err(|e| format!("Failed to read import.hx at {:?}: {}", import_path, e))?;

            let haxe_file = self
                .parse_file(import_path.to_str().unwrap_or("import.hx"), &source)
                .map_err(|e| format!("Parse error in {:?}: {}", import_path, e))?;

            self.import_hx_files.push(haxe_file);
        }

        Ok(())
    }

    /// Add a user source file to the compilation unit
    pub fn add_file(&mut self, source: &str, file_path: &str) -> Result<(), String> {
        let haxe_file = self
            .parse_file(file_path, source)
            .map_err(|e| format!("Parse error in {}: {}", file_path, e))?;

        self.user_files.push(haxe_file);
        Ok(())
    }

    /// Add a file from filesystem path
    /// This resolves the file's path and loads it, making it easier to work with
    /// real projects on disk
    pub fn add_file_from_path(&mut self, path: &PathBuf) -> Result<(), String> {
        use std::fs;

        let source = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {:?}: {}", path, e))?;

        let file_path_str = path
            .to_str()
            .ok_or_else(|| format!("Invalid UTF-8 in path: {:?}", path))?;

        self.add_file(&source, file_path_str)
    }

    /// Add all .hx files from a directory (recursively)
    /// This is useful for loading entire source trees
    ///
    /// # Arguments
    /// * `dir_path` - The directory to scan for .hx files
    /// * `recursive` - Whether to scan subdirectories
    pub fn add_directory(&mut self, dir_path: &PathBuf, recursive: bool) -> Result<usize, String> {
        use std::fs;

        let mut added_count = 0;

        let mut paths: Vec<PathBuf> = fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory {:?}: {}", dir_path, e))?
            .map(|entry| {
                entry
                    .map(|e| e.path())
                    .map_err(|e| format!("Failed to read directory entry: {}", e))
            })
            .collect::<Result<_, _>>()?;
        paths.sort();

        for path in paths {
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "hx" {
                        self.add_file_from_path(&path)?;
                        added_count += 1;
                    }
                }
            } else if path.is_dir() && recursive {
                added_count += self.add_directory(&path, recursive)?;
            }
        }

        Ok(added_count)
    }

    /// Resolve an import path to a filesystem path
    /// For example: "com.example.model.User" -> "src/com/example/model/User.hx"
    ///
    /// # Arguments
    /// * `import_path` - The import path (e.g., "com.example.model.User")
    /// * `source_paths` - Directories to search for source files (e.g., ["src", "lib"])
    pub fn resolve_import_path(
        &self,
        import_path: &str,
        source_paths: &[PathBuf],
    ) -> Option<PathBuf> {
        // Convert import path to filesystem path
        // "com.example.model.User" -> "com/example/model/User.hx"
        let file_path = import_path.replace('.', "/") + ".hx";

        // Search in each source path
        for source_path in source_paths {
            let full_path = source_path.join(&file_path);
            if full_path.exists() {
                return Some(full_path);
            }
        }

        None
    }

    /// Add a file by import path (e.g., "com.example.model.User")
    /// This automatically searches source paths to find the file
    ///
    /// # Arguments
    /// * `import_path` - The import path
    /// * `source_paths` - Directories to search for source files
    pub fn add_file_by_import(
        &mut self,
        import_path: &str,
        source_paths: &[PathBuf],
    ) -> Result<(), String> {
        let path = self
            .resolve_import_path(import_path, source_paths)
            .ok_or_else(|| format!("Could not resolve import: {}", import_path))?;

        self.add_file_from_path(&path)
    }

    /// Analyze dependencies and get compilation order
    ///
    /// This builds a dependency graph from all user files and determines
    /// the correct compilation order. It also detects circular dependencies.
    ///
    /// Returns (compilation_order, circular_dependencies)
    pub fn analyze_dependencies(&self) -> Result<DependencyAnalysis, Vec<CompilationError>> {
        if self.user_files.is_empty() {
            return Ok(DependencyAnalysis {
                compilation_order: Vec::new(),
                circular_dependencies: Vec::new(),
            });
        }

        // Build dependency graph
        let graph = DependencyGraph::from_files(&self.user_files);

        // Analyze
        let analysis = graph.analyze();

        // Report circular dependencies as warnings (not errors)
        if !analysis.circular_dependencies.is_empty() {
            debug!("⚠️  Warning: Circular dependencies detected!");
            for (i, cycle) in analysis.circular_dependencies.iter().enumerate() {
                debug!("\nCycle #{}:", i + 1);
                debug!("{}", cycle.format_error());
            }
            debug!("\nCompilation will proceed with best-effort ordering.\n");
        }

        Ok(analysis)
    }

    /// Compile a single file using shared state (string interner, symbol table, namespace resolver, etc.)
    /// This ensures symbols from different files can see each other
    ///
    /// If `skip_pre_registration` is true, assumes types have already been pre-registered
    /// and skips the first pass in lower_file.

    fn compile_file_with_shared_state_ex(
        &mut self,
        filename: &str,
        source: &str,
        skip_pre_registration: bool,
        skip_stdlib_merge: bool,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        use parser::parse_haxe_file_with_diagnostics;

        // Skip if already successfully compiled - return cached TypedFile
        if let Some(cached) = self.compiled_files.get(filename) {
            return Ok(cached.clone());
        }

        // Parse the file
        let t_parse = profile_timer(self.config.profile_typecheck);
        let haxe_file = self.parse_file(filename, source).map_err(|e| {
            vec![CompilationError {
                message: format!("Parse error: {}", e),
                location: SourceLocation::unknown(),
                category: ErrorCategory::ParseError,
                suggestion: None,
                related_errors: Vec::new(),
            }]
        })?;
        add_profile_ms(&mut self.typecheck_timings.file_parse_ms, t_parse);
        // Wrap in ParseResult-like struct for compatibility
        struct ParseResultShim {
            file: parser::HaxeFile,
        }
        let parse_result = ParseResultShim { file: haxe_file };

        self.compile_ast_with_shared_state(
            filename,
            source,
            &parse_result.file,
            skip_pre_registration,
            skip_stdlib_merge,
        )
    }

    fn compile_ast_with_shared_state(
        &mut self,
        filename: &str,
        source: &str,
        ast_file: &parser::HaxeFile,
        skip_pre_registration: bool,
        skip_stdlib_merge: bool,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        use crate::tast::ast_lowering::AstLowering;
        if self.config.profile_typecheck {
            self.typecheck_timings.files_seen += 1;
        }
        let profile_file_detail = self.config.profile_typecheck
            && std::env::var_os("RAYZOR_PROFILE_TYPECHECK_FILES").is_some();
        let file_total = profile_timer(profile_file_detail);
        let mut file_ast_ms = 0.0;
        let mut file_hir_ms = 0.0;
        let mut file_mir_prep_ms = 0.0;
        let mut file_mir_ms = 0.0;
        let mut file_merge_ms = 0.0;

        // Allocate (or look up) a compilation-level file_id for this file.
        // Previously hardcoded to FileId(0), which caused every TypedExpression
        // span in every file (Main.hx, BPETokenizer.hx, GenerationLoop.hx,
        // …) to carry the same file_id=0 — see
        // bugs_diagnostic_span_file_id_always_zero. The counter is monotonic
        // in arrival order; the filename map dedupes when the same file is
        // re-entered (e.g. via `compiled_files` cache lookup downstream).
        let file_id_u32 = *self
            .file_id_by_filename
            .entry(filename.to_string())
            .or_insert_with(|| {
                let id = self.next_file_id;
                self.next_file_id += 1;
                id
            });
        // Capture the EXACT source bytes the span_converter sees so the
        // renderer's source_map can use the same bytes for ariadne's
        // byte_offset → line/column resolution.
        self.file_source_by_filename
            .entry(filename.to_string())
            .or_insert_with(|| source.to_string());
        let file_id = diagnostics::FileId::new(file_id_u32 as usize);

        // Extract type info from AST for BLADE cache (before macros may modify it)
        if self.config.enable_cache {
            let type_info = crate::tools::preblade::extract_type_info_from_ast(ast_file);
            self.last_compiled_type_info = Some(type_info);
        }

        // Stage 1.5: Macro expansion (if enabled)
        let t_macro = profile_timer(self.config.profile_typecheck);
        let macro_expansion_needed = self.config.pipeline_config.enable_macro_expansion
            && self.macro_expansion_may_apply(ast_file);
        let ast_file_owned;
        let ast_file = if macro_expansion_needed {
            let mut class_registry = crate::macro_system::ClassRegistry::new();
            class_registry.register_files(&self.stdlib_files);
            class_registry.register_files(&self.import_hx_files);
            class_registry.register_files(&self.loaded_import_haxe_files);
            class_registry.register_file(ast_file);
            // Phase 2 fix: pass user files AND macro-bearing import files as
            // "dependency" files so cross-file macros (e.g.
            // `import tink.Json` + `tink.Json.parse(...)`) are discovered.
            // Without this, only the current file's macros are in the registry
            // and cross-file calls silently fall through.
            let mut dep_files: Vec<HaxeFile> = Vec::new();
            dep_files.extend(self.user_files.iter().cloned());
            dep_files.extend(self.loaded_import_haxe_files.iter().cloned());
            let expansion = crate::macro_system::expand_macros_with_dependencies(
                ast_file.clone(),
                class_registry,
                &dep_files,
            );
            // Surface macro expansion diagnostics to the user, not just to
            // debug logs. A silent fallthrough is much worse than a loud
            // error — a failed macro call otherwise routes to a regular
            // method (often the stdlib namesake) with no indication the
            // macro didn't run.
            for diag in &expansion.diagnostics {
                // Skip Info-level diagnostics — they're used for per-macro
                // registration traces and would spam the output.
                if matches!(diag.severity, crate::macro_system::MacroSeverity::Info) {
                    if matches!(diag.severity, crate::macro_system::MacroSeverity::Error) {
                        debug!("Macro expansion error in {}: {}", filename, diag.message);
                    }
                    continue;
                }
                let severity = match diag.severity {
                    crate::macro_system::MacroSeverity::Error => {
                        diagnostics::DiagnosticSeverity::Error
                    }
                    crate::macro_system::MacroSeverity::Warning => {
                        diagnostics::DiagnosticSeverity::Warning
                    }
                    crate::macro_system::MacroSeverity::Info => {
                        diagnostics::DiagnosticSeverity::Info
                    }
                };
                let loc = &diag.location;
                let pos = diagnostics::SourcePosition::new(
                    loc.line.max(1) as usize,
                    loc.column.max(1) as usize,
                    loc.byte_offset as usize,
                );
                let end_pos = diagnostics::SourcePosition::new(
                    loc.line.max(1) as usize,
                    (loc.column.max(1) + 1) as usize,
                    (loc.byte_offset + 1) as usize,
                );
                let span = diagnostics::SourceSpan::new(pos, end_pos, diagnostics::FileId::new(0));
                self.collected_diagnostics.push(diagnostics::Diagnostic {
                    severity,
                    code: Some("MACRO".to_string()),
                    message: format!("macro expansion in {}: {}", filename, diag.message),
                    span,
                    labels: Vec::new(),
                    suggestions: Vec::new(),
                    notes: Vec::new(),
                    help: Vec::new(),
                });
                if matches!(diag.severity, crate::macro_system::MacroSeverity::Error) {
                    debug!("Macro expansion error in {}: {}", filename, diag.message);
                }
            }
            if expansion.expansions_count > 0 {
                debug!(
                    "Macro expansion: {} macros expanded in {}",
                    expansion.expansions_count, filename
                );
            }
            // Store expansion origins for LSP macro hints
            self.macro_expansions.extend(expansion.expansion_origins);
            ast_file_owned = expansion.file;
            &ast_file_owned
        } else {
            if self.config.profile_typecheck && self.config.pipeline_config.enable_macro_expansion {
                self.typecheck_timings.macro_skipped_files += 1;
            }
            ast_file
        };
        add_profile_ms(&mut self.typecheck_timings.macro_ms, t_macro);

        // Lower to TAST using the SHARED state
        // NOTE: AstLowering needs an Rc<RefCell<StringInterner>> for TypedFile
        // We create a dummy one here - the actual interning happens via the &mut reference
        // TODO: Refactor CompilationUnit to store string_interner as Rc<RefCell<>> from the start
        let dummy_interner_rc = Rc::new(RefCell::new(StringInterner::new()));

        let mut lowering = AstLowering::new(
            &mut self.string_interner,
            dummy_interner_rc,
            &mut self.symbol_table,
            &self.type_table,
            &mut self.scope_tree,
            &mut self.namespace_resolver,
            &mut self.import_resolver,
        );

        // Skip pre-registration if requested (types already registered by CompilationUnit)
        lowering.set_skip_pre_registration(skip_pre_registration);

        // CompilationUnit manages stdlib loading itself via load_stdlib() and the
        // later stdlib MIR merge. Re-loading the stdlib inside AstLowering causes
        // the uncached path to pull in a different symbol/method set than the
        // cached path, which changes bundle contents and breaks DeltaBlue parity.
        lowering.set_skip_stdlib_loading(true);

        // Declared-static-signature index: lets call sites type statics whose
        // declaring file lowers later (no untyped-placeholder decay).
        lowering.set_static_sig_index(Rc::clone(&self.static_sig_index));

        // Seed class_fields from previously compiled files.
        // Only seed classes that have actual fields — empty entries interfere with
        // static method resolution by making the class "exist" in class_fields but
        // with no matching field, causing the static method to fall through to a
        // generic path instead of the stdlib dispatch.
        if !self.global_class_fields.is_empty() {
            let non_empty: BTreeMap<_, _> = self
                .global_class_fields
                .iter()
                .filter(|(_, fields)| !fields.is_empty())
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            if !non_empty.is_empty() {
                lowering.seed_class_fields(&non_empty);
            }
        }

        lowering.initialize_span_converter_with_filename(
            file_id.as_usize() as u32,
            source.to_string(),
            filename.to_string(),
        );

        let t_ast_lower = profile_timer(self.config.profile_typecheck);
        let typed_file = lowering
            .lower_file(ast_file)
            .map_err(|e| vec![e.to_compilation_error()])?;
        file_ast_ms = finish_profile_ms(&mut self.typecheck_timings.ast_lower_ms, t_ast_lower);

        // Export class_fields for subsequent compilations
        for (class_sym, fields) in lowering.export_class_fields() {
            self.global_class_fields
                .entry(*class_sym)
                .or_insert_with(|| fields.clone());
        }

        // Normal (non-safety) warnings: untyped empty array literals whose
        // element type stayed uncertain (never bound by a push/assign), so they
        // remain Array<Dynamic>. Always emitted — unlike ownership/safety
        // warnings these are not gated by `emit_safety_warnings`. (Last use of
        // `lowering` so its &mut borrow ends before the diagnostics push.)
        let array_warnings = lowering.take_empty_array_warnings();
        for (loc, msg) in array_warnings {
            let pos = diagnostics::SourcePosition::new(
                loc.line.max(1) as usize,
                loc.column.max(1) as usize,
                loc.byte_offset as usize,
            );
            let end_pos = diagnostics::SourcePosition::new(
                loc.line.max(1) as usize,
                (loc.column.max(1) + 1) as usize,
                (loc.byte_offset + 1) as usize,
            );
            let span = diagnostics::SourceSpan::new(pos, end_pos, diagnostics::FileId::new(0));
            self.collected_diagnostics.push(diagnostics::Diagnostic {
                severity: diagnostics::DiagnosticSeverity::Warning,
                code: Some("W0110".to_string()),
                message: msg,
                span,
                labels: Vec::new(),
                suggestions: Vec::new(),
                notes: Vec::new(),
                help: Vec::new(),
            });
        }

        // Send/Sync validation — check thread safety constraints (user files only)
        let is_stdlib = filename.contains("haxe-std/") || filename.contains("haxe-std\\");
        let t_send_sync = profile_timer(self.config.profile_typecheck);
        if !is_stdlib {
            use crate::tast::send_sync_validator::SendSyncValidator;
            let validator = SendSyncValidator::new(
                &self.type_table,
                &self.symbol_table,
                &self.string_interner,
                &typed_file.classes,
            );
            let mut send_sync_errors: Vec<CompilationError> = Vec::new();
            let collect_error =
                |error: crate::tast::send_sync_validator::SendSyncError| CompilationError {
                    message: error.message.clone(),
                    location: error.source_location,
                    category: ErrorCategory::ConcurrencyError,
                    suggestion: Some(
                        "Add @:derive([Send]) or @:derive([Send, Sync]) to the type".to_string(),
                    ),
                    related_errors: Vec::new(),
                };
            for class in &typed_file.classes {
                if let Err(error) = validator.validate_class(class) {
                    send_sync_errors.push(collect_error(error));
                }
                // Soundness: a class explicitly deriving Send/Sync must have
                // fields that fulfill the trait (extern types skip — opaque).
                for error in validator.validate_derive_soundness(class) {
                    send_sync_errors.push(collect_error(error));
                }
            }
            for function in &typed_file.functions {
                if let Err(error) = validator.validate_function(function) {
                    send_sync_errors.push(collect_error(error));
                }
            }
            if !send_sync_errors.is_empty() {
                add_profile_ms(&mut self.typecheck_timings.send_sync_ms, t_send_sync);
                return Err(send_sync_errors);
            }
        } // end if !is_stdlib
        add_profile_ms(&mut self.typecheck_timings.send_sync_ms, t_send_sync);

        // Ownership analysis: use-after-move detection (user files only)
        // Skip if any class has @:safety annotation — opted out of ownership tracking
        let has_safety_opt_out = typed_file.classes.iter().any(|c| c.has_safety_annotation());
        let t_ownership = profile_timer(self.config.profile_typecheck);
        if !is_stdlib && self.config.emit_safety_warnings && !has_safety_opt_out {
            let ownership_diagnostics = self.check_ownership_violations(&typed_file);
            if !ownership_diagnostics.is_empty() {
                // Print everything (so the user sees warnings AND the error
                // labels/help text), then if any diagnostic was strict
                // (`@:move`) we fail compilation with a hard error.
                self.print_mir_diagnostics(&ownership_diagnostics);
                let strict_errors: Vec<CompilationError> = ownership_diagnostics
                    .iter()
                    .filter(|d| matches!(d.severity, diagnostics::DiagnosticSeverity::Error))
                    .map(|d| CompilationError {
                        message: d.message.clone(),
                        location: SourceLocation {
                            file_id: d.span.file_id.as_usize() as u32,
                            byte_offset: d.span.start.byte_offset as u32,
                            line: d.span.start.line as u32,
                            column: d.span.start.column as u32,
                        },
                        category: ErrorCategory::OwnershipError,
                        suggestion: d.help.first().cloned(),
                        related_errors: Vec::new(),
                    })
                    .collect();
                if !strict_errors.is_empty() {
                    add_profile_ms(&mut self.typecheck_timings.ownership_ms, t_ownership);
                    return Err(strict_errors);
                }
            }
        }
        add_profile_ms(&mut self.typecheck_timings.ownership_ms, t_ownership);

        // Lower to HIR — pass loaded stdlib typed files so cross-file
        // static inline var references can be resolved
        use crate::ir::tast_to_hir::lower_tast_to_hir_with_imports;
        let import_refs: Vec<&crate::tast::node::TypedFile> =
            self.loaded_stdlib_typed_files.iter().collect();
        // (import_refs passed for inline var seeding)
        let t_hir = profile_timer(self.config.profile_typecheck);
        let hir_module = match lower_tast_to_hir_with_imports(
            &typed_file,
            &self.symbol_table,
            &self.type_table,
            &mut self.string_interner,
            None, // No semantic graphs for now
            &import_refs,
            &self.global_inline_vars,
        ) {
            Ok(module) => module,
            Err(errors) => {
                add_profile_ms(&mut self.typecheck_timings.hir_ms, t_hir);
                return Err(errors
                    .into_iter()
                    .map(|e| CompilationError {
                        message: e.message,
                        location: e.location,
                        category: ErrorCategory::TypeError,
                        suggestion: None,
                        related_errors: Vec::new(),
                    })
                    .collect::<Vec<_>>());
            }
        };
        file_hir_ms = finish_profile_ms(&mut self.typecheck_timings.hir_ms, t_hir);

        // Set source file path on HIR module for stack trace source info
        let mut hir_module = hir_module;
        hir_module.metadata.source_file = filename.to_string();

        // Check if this file contains ONLY extern class declarations BEFORE MIR lowering.
        // Extern class files only need TAST+HIR for type system registration (symbol scopes,
        // method signatures). Their runtime code is provided by build_stdlib() from Rust
        // implementations. Generating MIR stubs here would create function entries with wrong
        // signatures (0-param stubs for methods that need a receiver), breaking codegen.
        let t_extern_check = profile_timer(self.config.profile_typecheck);
        {
            use crate::tast::symbols::SymbolFlags;
            let has_non_extern_class = typed_file.classes.iter().any(|c| {
                !self
                    .symbol_table
                    .get_symbol(c.symbol_id)
                    .map(|s| s.flags.contains(SymbolFlags::EXTERN))
                    .unwrap_or(false)
            });
            let has_non_extern_abstract = typed_file.abstracts.iter().any(|a| {
                !self
                    .symbol_table
                    .get_symbol(a.symbol_id)
                    .map(|s| s.flags.contains(SymbolFlags::EXTERN))
                    .unwrap_or(false)
            });
            let has_extern_decls =
                !typed_file.classes.is_empty() || !typed_file.abstracts.is_empty();
            // An extern class may still declare a concrete (non-`@:native`) method
            // with a body — e.g. a small helper over other extern methods. Those
            // bodies need MIR: the per-method guards in hir_to_mir already skip the
            // bodyless extern methods, so only the concrete ones get lowered.
            let has_concrete_method = typed_file
                .classes
                .iter()
                .flat_map(|c| c.methods.iter())
                .chain(typed_file.abstracts.iter().flat_map(|a| a.methods.iter()))
                .any(|m| !m.body.is_empty());
            let is_extern_only = has_extern_decls
                && !has_non_extern_class
                && !has_non_extern_abstract
                && !has_concrete_method
                && typed_file.functions.is_empty()
                && typed_file.enums.is_empty();
            if is_extern_only {
                debug!(
                    "[EXTERN_ONLY] Skipping MIR for extern-only file: {}",
                    filename
                );
                self.compiled_files
                    .insert(filename.to_string(), typed_file.clone());
                add_profile_ms(&mut self.typecheck_timings.extern_check_ms, t_extern_check);
                return Ok(typed_file);
            }
        }
        add_profile_ms(&mut self.typecheck_timings.extern_check_ms, t_extern_check);

        // Lower to MIR
        // Use lower_hir_to_mir_with_function_map to:
        // 1. Pass external function references from previously compiled stdlib files
        // 2. Collect function mappings for stdlib files so user code can call them
        use crate::ir::hir_to_mir::lower_hir_to_mir_with_function_map;

        // Check if this is a stdlib file BEFORE lowering so we can decide whether
        // to collect function mappings
        let is_stdlib_file = filename.contains("haxe-std")
            || filename.contains("/haxe-std/")
            || filename.contains("\\haxe-std\\");

        debug!(
            "[MIR_LOWER] filename='{}', is_stdlib_file={}, classes={}",
            filename,
            is_stdlib_file,
            typed_file.classes.len()
        );

        // For user files, pass the stdlib function map so they can call stdlib functions
        // For stdlib files, pass an empty map (they can call each other once we accumulate the map)
        let external_functions = if is_stdlib_file {
            // Stdlib files can call previously compiled stdlib functions
            self.stdlib_function_map.clone()
        } else {
            // User files can call all compiled stdlib functions
            self.stdlib_function_map.clone()
        };

        // Name-based external function map for cross-file lookups where SymbolIds differ
        let external_functions_by_name = self.stdlib_function_name_map.clone();

        let stdlib_mapping = self.compiler_plugin_registry.build_combined_mapping();

        let t_mir_prep = profile_timer(self.config.profile_typecheck);
        let constructor_param_counts = self.import_constructor_param_counts.clone();
        let external_function_param_types = self.import_function_param_types.clone();

        // Seed cross-file property accessors from loaded stdlib typed files.
        // Extern-only files like sys/thread/Tls.hx skip MIR generation (handled
        // by `is_extern_only` above) so their property fields never reach
        // MirContext::register_class_metadata. Without this seed, user code
        // like `tls.value` falls through to a "field not found" error.
        // Equivalent to the BLADE-cache restoration path at line ~3020 but
        // also covers fresh (uncached) stdlib loads.
        let stdlib_files: Vec<_> = self
            .loaded_stdlib_typed_files
            .iter()
            .map(|f| f as *const _)
            .collect();
        for tf_ptr in stdlib_files {
            let tf = unsafe { &*tf_ptr };
            self.seed_property_accessors_from_typed_file(tf);
        }

        // Save external constructor keys to filter them out of the result
        let external_constructor_keys: std::collections::BTreeSet<String> =
            self.import_constructor_name_map.keys().cloned().collect();

        // Globals from modules already lowered, keyed by qualified name. Their
        // ids are final: imports are renumbered into disjoint ranges before this
        // module is lowered.
        let external_globals = self.import_external_globals.clone();
        file_mir_prep_ms = finish_profile_ms(&mut self.typecheck_timings.mir_prep_ms, t_mir_prep);

        let t_mir = profile_timer(self.config.profile_typecheck);
        let mir_result = match lower_hir_to_mir_with_function_map(
            &hir_module,
            &self.string_interner,
            &self.type_table,
            &self.symbol_table,
            external_functions,
            external_functions_by_name,
            external_globals,
            &stdlib_mapping,
            self.import_field_index_map.clone(),
            self.import_property_access_map.clone(),
            self.import_constructor_name_map.clone(),
            self.import_class_alloc_sizes.clone(),
            self.import_class_method_symbols.clone(),
            self.import_class_type_to_symbol.clone(),
            constructor_param_counts,
            external_function_param_types,
            self.import_class_alloc_sizes_by_name.clone(),
            self.import_interface_method_names.clone(),
            self.import_interface_method_return_types.clone(),
            self.import_interface_extends.clone(),
            self.import_interface_vtables.clone(),
            self.import_function_param_iface_names.clone(),
            self.import_field_class_names.clone(),
            Some(Rc::clone(&self.static_sig_index)),
        ) {
            Ok(result) => result,
            Err(errors) => {
                add_profile_ms(&mut self.typecheck_timings.mir_ms, t_mir);
                add_profile_ms(&mut self.typecheck_timings.mir_lower_core_ms, t_mir);
                return Err(errors
                    .into_iter()
                    .map(|e| CompilationError {
                        message: e.message,
                        location: e.location,
                        category: ErrorCategory::TypeError,
                        suggestion: None,
                        related_errors: Vec::new(),
                    })
                    .collect::<Vec<_>>());
            }
        };
        file_mir_ms = finish_profile_ms(&mut self.typecheck_timings.mir_ms, t_mir);
        self.typecheck_timings.mir_lower_core_ms += file_mir_ms;

        // Print any diagnostics from MIR lowering (e.g., exhaustiveness warnings)
        if !mir_result.diagnostics.is_empty() {
            self.print_mir_diagnostics(&mir_result.diagnostics);
        }

        // Capture user-defined function IDs before module is consumed
        let mir_result_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            mir_result.function_map.values().copied().collect();
        let mir_result_ctor_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> =
            mir_result.constructor_name_map.values().copied().collect();

        let mut mir_module = mir_result.module;

        // (MIR dump moved to after stdlib merge)

        // Build BladeCachedMaps for BLADE cache (name-keyed, before ID-keyed accumulation consumes the data)
        if self.config.enable_cache {
            let cached_maps = self.build_cached_maps_from_mir_result(
                &mir_result.function_map,
                &mir_result.field_index_map,
                &mir_result.constructor_name_map,
                &mir_result.class_alloc_sizes,
                &mir_result.field_class_names,
                &mir_result.property_access_map,
                &mir_result.function_param_hir_types,
                &mir_result.interface_vtables,
                &mir_result.interface_method_names,
                &mir_result.interface_method_return_types,
                &mir_result.interface_extends,
            );
            self.last_compiled_cached_maps = Some(cached_maps);
        }

        // Collect SymbolId-based function mappings from ALL files (stdlib + imports)
        // This enables cross-file method calls: user file can call import file methods
        // via the shared symbol table (SymbolIds are consistent across files)
        debug!(
            "DEBUG: Collecting {} function mappings from file: {}",
            mir_result.function_map.len(),
            filename
        );
        // (max_own_func_id computed earlier before module was moved)
        for (symbol_id, func_id) in mir_result.function_map {
            self.stdlib_function_map.insert(symbol_id, func_id);
        }

        // Collect constructor name map — only include constructors NEW to this file,
        // not external ones that were passed in via import_constructor_name_map.
        for (class_name, func_id) in mir_result.constructor_name_map {
            if !external_constructor_keys.contains(&class_name) {
                self.import_constructor_name_map.insert(class_name, func_id);
            }
        }

        // Collect class allocation sizes from ALL files
        for (type_id, size) in mir_result.class_alloc_sizes {
            self.import_class_alloc_sizes.insert(type_id, size);
        }

        // Collect name-keyed class allocation sizes (stable across compilation contexts)
        for (name, size) in mir_result.class_alloc_sizes_by_name {
            self.import_class_alloc_sizes_by_name.insert(name, size);
        }

        // Collect class method symbols from ALL files
        for (key, sym) in mir_result.class_method_symbols {
            self.import_class_method_symbols.insert(key, sym);
        }

        // Collect name-based mappings for cross-file lookups.
        // Use qualified names to avoid collisions (e.g., "current" matching
        // both ArrayIterator.current field and Thread.current method).
        // For stdlib files: all functions with non-empty bodies.
        // For user packages: only functions with qualified names (to avoid
        // polluting the namespace with bare names like "new").
        if is_stdlib_file {
            for (func_id, func) in &mir_module.functions {
                if !func.cfg.blocks.is_empty() {
                    let map_name = func.qualified_name.as_deref().unwrap_or(&func.name);
                    self.stdlib_function_name_map
                        .insert(map_name.to_string(), *func_id);
                }
            }
        } else {
            // User packages: only add functions with qualified names
            for (func_id, func) in &mir_module.functions {
                if !func.cfg.blocks.is_empty() {
                    if let Some(qn) = func.qualified_name.as_deref() {
                        self.stdlib_function_name_map
                            .insert(qn.to_string(), *func_id);
                    }
                }
            }
        }

        // Accumulate field index and property access maps from all compiled files
        // (both stdlib and imports) so user files can resolve field access on imported classes
        for (sym, val) in mir_result.field_index_map {
            self.import_field_index_map.insert(sym, val);
        }
        for (sym, name) in mir_result.field_class_names {
            self.import_field_class_names.insert(sym, name);
        }
        for (sym, val) in mir_result.property_access_map {
            self.import_property_access_map.insert(sym, val);
        }
        for (ty, sym) in mir_result.class_type_to_symbol {
            self.import_class_type_to_symbol.insert(ty, sym);
        }

        // Accumulate interface metadata so subsequent files can resolve
        // cross-file interface dispatch (variable typed as the interface,
        // method calls on it, fat-pointer wrapping at construction).
        for (sym, methods) in mir_result.interface_method_names {
            self.import_interface_method_names.insert(sym, methods);
        }
        for (key, ty) in mir_result.interface_method_return_types {
            self.import_interface_method_return_types.insert(key, ty);
        }
        for (sym, parents) in mir_result.interface_extends {
            self.import_interface_extends.insert(sym, parents);
        }
        for (key, vtable) in mir_result.interface_vtables {
            self.import_interface_vtables.insert(key, vtable);
        }

        // Harvest per-function HIR param types so the user file's
        // `maybe_materialize_for_call` Path 3 can recover class→interface
        // wrap decisions for imported constructors. MIR alone erases both
        // Class and Interface to `Ptr(Void)`; without these names the
        // wrap is silently skipped and raw class pointers land in
        // interface-typed fields (SIGBUS on first vtable dispatch).
        {
            let type_table = self.type_table.borrow();
            let string_interner = &self.string_interner;
            let symbol_table = &self.symbol_table;
            let resolve_name = |ty: crate::tast::TypeId| -> Option<String> {
                let info = type_table.get(ty)?;
                let symbol_id = match &info.kind {
                    crate::tast::TypeKind::Class { symbol_id, .. } => Some(*symbol_id),
                    crate::tast::TypeKind::Interface { symbol_id, .. } => Some(*symbol_id),
                    // Constrained type parameter `<T:Iface>`: forward the
                    // interface constraint's name. The importer's context sees
                    // this param as a TypeParameter with an EMPTY constraint
                    // list (constraints don't survive the module boundary), so
                    // it can't recover the interface to wrap the class arg —
                    // resolve it here in the declaring module where the
                    // constraint is present, and the name-based call-site wrap
                    // (external_function_param_iface_names) fires cross-module.
                    crate::tast::TypeKind::TypeParameter { constraints, .. } => constraints
                        .iter()
                        .find_map(|cid| match type_table.get(*cid) {
                            Some(ct) => {
                                if let crate::tast::TypeKind::Interface { symbol_id, .. } = &ct.kind
                                {
                                    Some(*symbol_id)
                                } else {
                                    None
                                }
                            }
                            None => None,
                        }),
                    _ => None,
                }?;
                let sym = symbol_table.get_symbol(symbol_id)?;
                sym.qualified_name
                    .and_then(|n| string_interner.get(n))
                    .or_else(|| string_interner.get(sym.name))
                    .map(|s| s.to_string())
            };
            for (func_id, type_ids) in &mir_result.function_param_hir_types {
                let names: Vec<Option<String>> =
                    type_ids.iter().copied().map(resolve_name).collect();
                if names.iter().any(|n| n.is_some()) {
                    self.import_function_param_iface_names
                        .insert(*func_id, names);
                }
            }
        }

        // NOTE: extern-only files are handled above (before MIR generation).

        // Capture this file's user-defined function IDs (from function_map + constructor_map).
        // Exclude MIR wrapper stubs — these have trivial bodies (ret false/null) and must be
        // replaced by the stdlib merge with real implementations that delegate to runtime.
        // Without this filter, non-extern stdlib classes like EReg have their methods (match,
        // matched, etc.) incorrectly protected from replacement.
        let own_func_ids: std::collections::BTreeSet<crate::ir::IrFunctionId> = mir_result_func_ids
            .union(&mir_result_ctor_ids)
            .copied()
            .filter(|func_id| {
                mir_module
                    .functions
                    .get(func_id)
                    .map(|f| !matches!(f.kind, crate::ir::FunctionKind::MirWrapper))
                    .unwrap_or(true)
            })
            .collect();

        // Store for try_compile_import to pick up and track after renumbering
        self.last_compiled_own_func_ids = Some(own_func_ids.clone());

        // The stdlib merge (imports + stdlib MIR wrappers) should only happen for the
        // final user file, not for import files. Import files produce clean MIR with only
        // their own functions. When the main file is compiled, all imports are merged in
        // first, then a single stdlib merge resolves forward refs without corrupting
        // user package functions.
        let t_stdlib_merge = profile_timer(self.config.profile_typecheck);
        if !is_stdlib_file && !skip_stdlib_merge {
            // Merge stdlib MIR (extern functions for Thread, Channel, Mutex, Arc, etc.)
            // This ensures extern runtime functions are available.
            // Uses build_stdlib_with_plugins to include HDLL extern declarations from loaded plugins.
            // Cache the stdlib MIR module to avoid rebuilding it for each user file.
            use crate::stdlib::build_stdlib_with_plugins;
            let mut stdlib_mir = if let Some(ref cached) = self.cached_stdlib_mir {
                cached.clone()
            } else {
                let mir = build_stdlib_with_plugins(&self.compiler_plugin_registry);
                self.cached_stdlib_mir = Some(mir.clone());
                mir
            };

            // Merge on-demand imported MIR modules (e.g., BalancedTree.hx, Point2D.hx)
            // into the user module. These were already renumbered to high IDs (100000+)
            // during load_imports_efficiently, so they won't collide with user or stdlib IDs.
            // Import modules now skip the stdlib merge, so they contain both:
            // (a) source-level declarations (tracked in import_own_func_ids) — protect these
            // (b) generated MIR wrappers for stdlib calls — let stdlib merge replace these
            let mut merged_import_func_ids: std::collections::BTreeSet<IrFunctionId> =
                own_func_ids.clone();
            // Sort import modules by name for deterministic merge order.
            // Sorting ensures the merged MIR is identical regardless of resolver order.
            self.import_mir_modules.sort_by(|a, b| a.name.cmp(&b.name));
            for import_module in self.import_mir_modules.drain(..) {
                // Merge import type definitions so runtime RTTI registration includes
                // imported classes/enums (needed for uncaught exception formatting and
                // hierarchy-aware typed catches).
                for (_old_type_def_id, mut typedef) in import_module.types {
                    let new_type_def_id = mir_module.alloc_typedef_id();
                    typedef.id = new_type_def_id;
                    mir_module.types.insert(new_type_def_id, typedef);
                }
                // Imported globals were renumbered into a disjoint id range
                // (renumber_and_push_import_mir); carry them into the merged
                // module so its globals table matches the instructions.
                // Previously they were dropped entirely while the imports'
                // LoadGlobal/StoreGlobal kept their dense-from-0 ids —
                // aliasing the main module's statics slot-for-slot.
                for (global_id, global) in import_module.globals {
                    if let Some(prev) = mir_module.globals.get(&global_id) {
                        if prev.name != global.name {
                            panic!(
                                "MIR merge id collision: global id {:?} is '{}' but module '{}' \
                                 provides '{}' at the same id.",
                                global_id, prev.name, import_module.name, global.name
                            );
                        }
                    }
                    mir_module.globals.insert(global_id, global);
                }
                for (func_id, func) in import_module.functions {
                    // Only protect source-level declarations (methods, constructors).
                    // MIR wrappers (not in import_own_func_ids) can be replaced by stdlib.
                    //
                    // `import_own_func_ids` is keyed by renumbered IrFunctionId, computed
                    // independently per import module via its own local-id arithmetic
                    // (`old_id + import_base`). A stdlib MIR-wrapper stub for a bodyless
                    // `extern class` method (e.g. `Tensor.addInto` -> `Tensor_addInto`) can
                    // land on a renumbered id that coincidentally collides with an unrelated
                    // genuine "own" declaration from another import — confirmed via
                    // `RAYZOR_DBG_LOAD` tracing: `Tensor_addInto`'s renumbered id showed up
                    // `kind=MirWrapper` yet was still `protected=true`, and its name never
                    // appeared in the `own_func_ids` filter's input set at all, so the
                    // protection wasn't coming from THIS function's own membership. Guard
                    // against that class of ID collision with a NAME-based veto: never
                    // protect an id whose function is a known stdlib MIR-wrapper name,
                    // regardless of why the id ended up in `import_own_func_ids`.
                    let is_known_stdlib_wrapper =
                        stdlib_mapping.is_mir_wrapper_function(&func.name);
                    if self.import_own_func_ids.contains(&func_id) && !is_known_stdlib_wrapper {
                        merged_import_func_ids.insert(func_id);
                    }
                    // Import id ranges are disjoint by construction (per-module
                    // base + stride); an occupied slot with a DIFFERENT name is
                    // an id-space collision. Last-writer-wins here silently
                    // rebinds every call site of the loser to an unrelated
                    // function (observed: a SpinPool worker's stdlib-extern
                    // call dispatching into rayzor_channel_receive once the
                    // import count pushed a module's base into another range).
                    if let Some(prev) = mir_module.functions.get(&func_id) {
                        let prev_name = prev.qualified_name.as_deref().unwrap_or(&prev.name);
                        let new_name = func.qualified_name.as_deref().unwrap_or(&func.name);
                        if prev_name != new_name {
                            panic!(
                                "MIR merge id collision: function id {:?} is '{}' but module '{}' \
                                 provides '{}' at the same id. Import id ranges overlapped — this \
                                 build would silently misroute calls.",
                                func_id, prev_name, import_module.name, new_name
                            );
                        }
                    }
                    mir_module.functions.insert(func_id, func);
                }
                for (func_id, extern_func) in import_module.extern_functions {
                    if let Some(prev) = mir_module.extern_functions.get(&func_id) {
                        if prev.name != extern_func.name {
                            panic!(
                                "MIR merge id collision: extern id {:?} is '{}' but module '{}' \
                                 provides '{}' at the same id.",
                                func_id, prev.name, import_module.name, extern_func.name
                            );
                        }
                    }
                    mir_module.extern_functions.insert(func_id, extern_func);
                }
                // Carry the import's name records into the merged module so the
                // post-merge fixup/verification passes keep their ground truth.
                for (func_id, name) in import_module.external_function_names {
                    mir_module
                        .external_function_names
                        .entry(func_id)
                        .or_insert(name);
                }
            }

            // Resolve name-keyed forward-ref dispatch-thunk stubs. A file that
            // constructs an imported class as an interface (e.g.
            // `ArchRegistry.withDefaults` doing `new LlamaArch()` when
            // `LlamaArch` compiles AFTER it) emits an EMPTY
            // `__vtable_dispatch_thunk__<class>_<method>` stub as a placeholder,
            // trusting the real thunk (same name, from the class's own module)
            // to be merged in. Both now coexist by different ids; the stub's
            // `FunctionRef` in the fat-ptr slot still points at the EMPTY stub
            // (calling it traps — SIGTRAP). Redirect every ref from an empty
            // `__vtable_dispatch_thunk__*` stub to the real, non-empty
            // same-named thunk, then drop the stub. Order-independent: works
            // regardless of which file compiled first.
            {
                let mut thunk_real: std::collections::BTreeMap<String, IrFunctionId> =
                    std::collections::BTreeMap::new();
                let mut thunk_stubs: Vec<(String, IrFunctionId)> = Vec::new();
                for (func_id, func) in &mir_module.functions {
                    if !func.name.starts_with("__vtable_dispatch_thunk__") {
                        continue;
                    }
                    if func.cfg.blocks.is_empty()
                        || func.cfg.blocks.values().all(|b| b.instructions.is_empty())
                    {
                        thunk_stubs.push((func.name.clone(), *func_id));
                    } else {
                        // Prefer the first real (non-empty) definition per name.
                        thunk_real.entry(func.name.clone()).or_insert(*func_id);
                    }
                }
                let mut thunk_replacements: std::collections::BTreeMap<IrFunctionId, IrFunctionId> =
                    std::collections::BTreeMap::new();
                for (name, stub_id) in &thunk_stubs {
                    if let Some(&real_id) = thunk_real.get(name) {
                        if real_id != *stub_id {
                            thunk_replacements.insert(*stub_id, real_id);
                        }
                    }
                }
                if !thunk_replacements.is_empty() {
                    for (_, caller_func) in mir_module.functions.iter_mut() {
                        for block in caller_func.cfg.blocks.values_mut() {
                            for instr in &mut block.instructions {
                                match instr {
                                    IrInstruction::CallDirect { func_id, .. }
                                    | IrInstruction::FunctionRef { func_id, .. }
                                    | IrInstruction::MakeClosure { func_id, .. } => {
                                        if let Some(&new_id) = thunk_replacements.get(func_id) {
                                            *func_id = new_id;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    // Drop the now-unreferenced empty stubs.
                    for stub_id in thunk_replacements.keys() {
                        mir_module.functions.remove(stub_id);
                    }
                }
            }

            // CRITICAL FIX: Renumber stdlib function IDs to avoid collisions with user functions
            // Each MIR module starts function IDs from 0, so when merging stdlib and user modules,
            // IDs will collide. For example:
            //   - User module: IrFunctionId(2) = "indexOf"
            //   - Stdlib module: IrFunctionId(2) = "free"
            // Without renumbering, stdlib's "free" would be skipped, causing vec_u8_free to call "indexOf"!

            // Find the maximum function ID in the user module
            let max_user_func_id = mir_module
                .functions
                .keys()
                .map(|id| id.0)
                .max()
                .unwrap_or(0);

            let max_user_extern_id = mir_module
                .extern_functions
                .keys()
                .map(|id| id.0)
                .max()
                .unwrap_or(0);

            let offset = std::cmp::max(max_user_func_id, max_user_extern_id) + 1;

            debug!("DEBUG: Renumbering stdlib functions with offset {} (max_user_func={}, max_user_extern={})",
                  offset, max_user_func_id, max_user_extern_id);

            // Build map of function names to ALL IDs in the user module (before merging).
            // Multiple import modules can have duplicate extern declarations of the same function.
            // We need to track ALL of them to replace every copy.
            let mut user_func_name_to_ids: BTreeMap<String, Vec<IrFunctionId>> = BTreeMap::new();
            for (func_id, func) in &mir_module.functions {
                user_func_name_to_ids
                    .entry(func.name.clone())
                    .or_default()
                    .push(*func_id);
            }

            // The full stdlib/runtime MIR module is large, and most source cold
            // runs only need the wrappers that replace stubs already emitted in
            // the user/import MIR. Keep those roots plus their transitive direct
            // function references; tree-shake still runs later as the semantic
            // safety net.
            if std::env::var_os("RAYZOR_FULL_STDLIB_MERGE").is_none() {
                let needed_stdlib_names: BTreeSet<String> = user_func_name_to_ids
                    .iter()
                    .filter(|(_, existing_ids)| {
                        existing_ids
                            .iter()
                            .any(|id| !merged_import_func_ids.contains(id))
                    })
                    .map(|(name, _)| name.clone())
                    .collect();
                let dropped =
                    retain_referenced_stdlib_functions(&mut stdlib_mir, &needed_stdlib_names);
                debug!(
                    "DEBUG: Selective stdlib merge dropped {} unreferenced functions",
                    dropped
                );
            }

            // Build mapping of old stdlib IDs to new renumbered IDs
            use std::collections::BTreeMap;
            let mut id_mapping: BTreeMap<IrFunctionId, IrFunctionId> = BTreeMap::new();

            // Note: extern_functions is not used - externs are in the functions map with empty CFGs
            // So we only need to renumber the functions map

            // FIRST PASS: Build complete ID mapping for all stdlib functions
            // We must do this BEFORE updating CallDirect instructions so that all IDs are available
            for (old_id, _) in &stdlib_mir.functions {
                let new_id = IrFunctionId(old_id.0 + offset);
                id_mapping.insert(*old_id, new_id);
            }

            // SECOND PASS: Renumber functions and update their internal references
            let mut renumbered_functions = BTreeMap::new();
            for (old_id, mut func) in stdlib_mir.functions {
                let new_id = *id_mapping.get(&old_id).unwrap();

                // Update the function's own ID
                func.id = new_id;

                // Update all function ID references in instructions (CallDirect, FunctionRef, MakeClosure)
                use crate::ir::IrInstruction;
                for block in func.cfg.blocks.values_mut() {
                    for inst in &mut block.instructions {
                        match inst {
                            IrInstruction::CallDirect { func_id, .. } => {
                                if let Some(&new_func_id) = id_mapping.get(func_id) {
                                    debug!(
                                        "DEBUG: Updated CallDirect in {} from func_id {} -> {}",
                                        func.name, func_id.0, new_func_id.0
                                    );
                                    *func_id = new_func_id;
                                }
                            }
                            IrInstruction::FunctionRef { func_id, .. } => {
                                if let Some(&new_func_id) = id_mapping.get(func_id) {
                                    debug!(
                                        "DEBUG: Updated FunctionRef in {} from func_id {} -> {}",
                                        func.name, func_id.0, new_func_id.0
                                    );
                                    *func_id = new_func_id;
                                }
                            }
                            IrInstruction::MakeClosure { func_id, .. } => {
                                if let Some(&new_func_id) = id_mapping.get(func_id) {
                                    debug!(
                                        "DEBUG: Updated MakeClosure in {} from func_id {} -> {}",
                                        func.name, func_id.0, new_func_id.0
                                    );
                                    *func_id = new_func_id;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                renumbered_functions.insert(new_id, func);
                debug!(
                    "DEBUG: Renumbered function '{}': {} -> {}",
                    renumbered_functions[&new_id].name, old_id.0, new_id.0
                );
            }

            // Merge renumbered stdlib functions - no collisions possible now!
            // (Note: extern functions are included in the functions map with empty CFGs)
            //
            // IMPORTANT: Replace user functions that have the same NAME as stdlib functions
            // The user module might have extern declarations (e.g. rayzor_channel_init) from
            // the lowering process, but these might have incorrect signatures due to type
            // inference issues. The stdlib version is the source of truth, so we REPLACE
            // the user's version with the stdlib's version.

            // Build a map of old ID -> new ID for all replacements
            let mut id_replacements: BTreeMap<IrFunctionId, IrFunctionId> = BTreeMap::new();

            for (func_id, func) in &renumbered_functions {
                if let Some(existing_ids) = user_func_name_to_ids.get(&func.name) {
                    for &existing_id in existing_ids {
                        // Protect source-level import functions from stdlib replacement.
                        if !merged_import_func_ids.contains(&existing_id) {
                            id_replacements.insert(existing_id, *func_id);
                        } else if func.name == "match" || func.name == "matched" {
                            eprintln!(
                                "[MERGE_PROTECT] fn{}={} protected from fn{}",
                                existing_id.0, func.name, func_id.0
                            );
                        }
                    }
                }
            }

            // Now merge the stdlib functions
            for (func_id, func) in renumbered_functions {
                // If this function replaces existing ones, remove old copies
                // Remove stubs, but protect user package import functions
                if let Some(existing_ids) = user_func_name_to_ids.get(&func.name) {
                    for &existing_id in existing_ids {
                        if !merged_import_func_ids.contains(&existing_id) {
                            mir_module.functions.remove(&existing_id);
                        }
                    }
                }

                mir_module.functions.insert(func_id, func);
                // Keep next_function_id in sync so alloc_function_id() won't collide
                mir_module.next_function_id = mir_module.next_function_id.max(func_id.0 + 1);
            }

            // Update ALL instructions that reference replaced function IDs
            // This is done AFTER all merging to avoid ID conflicts
            if !id_replacements.is_empty() {
                for (_, caller_func) in mir_module.functions.iter_mut() {
                    for block in caller_func.cfg.blocks.values_mut() {
                        for instr in &mut block.instructions {
                            match instr {
                                IrInstruction::CallDirect {
                                    func_id: ref mut called_func_id,
                                    ..
                                } => {
                                    if let Some(&new_id) = id_replacements.get(called_func_id) {
                                        *called_func_id = new_id;
                                    }
                                }
                                IrInstruction::FunctionRef {
                                    func_id: ref mut ref_func_id,
                                    ..
                                } => {
                                    if let Some(&new_id) = id_replacements.get(ref_func_id) {
                                        debug!(
                                        "DEBUG: Updated FunctionRef in {} from func_id {} -> {}",
                                        caller_func.name, ref_func_id.0, new_id.0
                                    );
                                        *ref_func_id = new_id;
                                    }
                                }
                                IrInstruction::MakeClosure {
                                    func_id: ref mut closure_func_id,
                                    ..
                                } => {
                                    if let Some(&new_id) = id_replacements.get(closure_func_id) {
                                        debug!(
                                        "DEBUG: Updated MakeClosure in {} from func_id {} -> {}",
                                        caller_func.name, closure_func_id.0, new_id.0
                                    );
                                        *closure_func_id = new_id;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }

                // Refresh the global name map after the merge.
                //
                // When the user file lowered its own MIR, every pre-merge
                // function (including MirWrapper stubs like `array_length`,
                // `Tensor_mul`, `rayzor_anon_set_field_by_index`, …) was
                // seeded into `stdlib_function_name_map` at this file's
                // pre-merge id (e.g. fn12). The stdlib merge above just
                // deleted those stub IrFunctions and replaced them with
                // renumbered real bodies (e.g. fn640115), updating every
                // CallDirect / FunctionRef / MakeClosure in the module via
                // `id_replacements`. But the name map still points at the
                // dead pre-merge ids — so the later
                // `fixup_stale_cross_module_refs` pass, when it looks up a
                // cross-module CallDirect by external-function name, gets
                // back the dead id and writes it over a valid renumbered
                // target, surfacing as a codegen "undefined function fnN"
                // error. Apply the same renumber to the name map so it
                // tracks the merged module.
                for id in self.stdlib_function_name_map.values_mut() {
                    if let Some(&new_id) = id_replacements.get(id) {
                        *id = new_id;
                    }
                }
            }

            // Verify MIR wrapper forward refs were replaced during merge.
            // A MirWrapper function with an empty CFG means the stdlib merge failed
            // to find the implementation — this would cause wrong values at runtime.
            if cfg!(debug_assertions) {
                for (func_id, func) in &mir_module.functions {
                    if func.cfg.blocks.is_empty()
                        && !matches!(func.kind, crate::ir::FunctionKind::ExternC)
                    {
                        eprintln!(
                            "warning: empty function body after stdlib merge: '{}' (ID {}) kind={:?}",
                            func.name, func_id.0, func.kind
                        );
                    }
                }
            }
        } // end if !is_stdlib_file (stdlib merge + renumbering)
        file_merge_ms =
            finish_profile_ms(&mut self.typecheck_timings.stdlib_merge_ms, t_stdlib_merge);

        // Dump MIR after stdlib merge so wrapper bodies are visible
        if std::env::var("RAYZOR_DUMP_MIR").is_ok() {
            eprintln!("=== MIR DUMP (post-merge) for {} ===", filename);
            eprintln!("{}", crate::ir::dump::dump_module(&mir_module));
            eprintln!("=== END MIR DUMP ===");
        }

        // Run monomorphization pass to specialize generic functions
        let t_monomorphize = profile_timer(self.config.profile_typecheck);
        let mut monomorphizer = Monomorphizer::new();
        monomorphizer.monomorphize_module(&mut mir_module);
        add_profile_ms(&mut self.typecheck_timings.monomorphize_ms, t_monomorphize);
        // let mono_stats = monomorphizer.stats();
        // if mono_stats.generic_functions_found > 0 || mono_stats.instantiations_created > 0 {
        //     debug!("DEBUG: Monomorphization stats: {} generic functions, {} instantiations, {} call sites rewritten",
        //               mono_stats.generic_functions_found,
        //               mono_stats.instantiations_created,
        //               mono_stats.call_sites_rewritten);
        // }

        // // Debug: dump alloc sizes
        // for (name, size) in &self.import_class_alloc_sizes_by_name {
        //     if name.contains("Point") || name.contains("Particle") || name.contains("Simulation") {
        //         eprintln!("[ALLOC_SIZE] {} → {} bytes", name, size);
        //     }
        // }
        // // Debug: dump constructor name map
        // for (name, fid) in &self.import_constructor_name_map {
        //     eprintln!("[CTOR_MAP] {} → fn{}", name, fid.0);
        // }
        // // Debug: dump constructor signatures in final merged module
        // for (fid, func) in &mir_module.functions {
        //     if func.name == "new" && !func.cfg.blocks.is_empty() {
        //         let params: Vec<String> = func.signature.parameters.iter()
        //             .map(|p| format!("{}:{:?}", p.name, p.ty)).collect();
        //         let qn = func.qualified_name.as_deref().unwrap_or("?");
        //         let blocks = func.cfg.blocks.len();
        //         let insts: usize = func.cfg.blocks.values().map(|b| b.instructions.len()).sum();
        //         // eprintln!("[FINAL_MIR] fn{}={} qn={} blocks={} insts={} new({})", fid.0, func.name, qn, blocks, insts, params.join(", "));
        //         // Dump instructions for Point2D-sized constructors
        //         // if params.len() == 3 && params[1].contains("F64") {
        //         //     for block in func.cfg.blocks.values() {
        //         //         for inst in &block.instructions {
        //         //             eprintln!("[FINAL_MIR]   {:?}", inst);
        //         //         }
        //         //     }
        //         // }
        //     }
        // }

        // Store the MIR module
        self.mir_modules.push(std::sync::Arc::new(mir_module));

        // Mark as successfully compiled to prevent redundant recompilation
        self.compiled_files
            .insert(filename.to_string(), typed_file.clone());

        if profile_file_detail {
            let total_ms = file_total.map(elapsed_ms).unwrap_or(0.0);
            eprintln!(
                "  typecheck-file: total={:.2}ms ast={:.2}ms hir={:.2}ms mir_prep={:.2}ms mir={:.2}ms merge={:.2}ms file={}",
                total_ms,
                file_ast_ms,
                file_hir_ms,
                file_mir_prep_ms,
                file_mir_ms,
                file_merge_ms,
                filename
            );
        }

        Ok(typed_file)
    }

    /// Compile a single file using shared state (backward-compatible wrapper).
    /// This is used for the main/final user file — stdlib merge runs.
    fn compile_file_with_shared_state(
        &mut self,
        filename: &str,
        source: &str,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        self.compile_file_with_shared_state_ex(filename, source, false, false)
    }

    /// Compile using an already-parsed AST (avoids redundant re-parsing).
    fn compile_pre_parsed_file(
        &mut self,
        ast_file: &parser::HaxeFile,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        // Skip if already compiled
        if let Some(cached) = self.compiled_files.get(&ast_file.filename) {
            return Ok(cached.clone());
        }

        let source = ast_file.input.as_deref().unwrap_or_default();
        self.compile_ast_with_shared_state(&ast_file.filename, source, ast_file, false, false)
    }

    fn macro_expansion_may_apply(&self, ast_file: &parser::HaxeFile) -> bool {
        haxe_file_source_has_macro_hook(ast_file)
            || self.user_files.iter().any(haxe_file_source_has_macro_hook)
            || self
                .import_hx_files
                .iter()
                .any(haxe_file_source_has_macro_hook)
            || self
                .loaded_import_haxe_files
                .iter()
                .any(haxe_file_source_has_macro_hook)
    }

    fn compile_user_ast_collecting_errors(
        &mut self,
        ast_file: &parser::HaxeFile,
        all_typed_files: &mut Vec<TypedFile>,
        all_errors: &mut Vec<CompilationError>,
    ) {
        match self.compile_pre_parsed_file(ast_file) {
            Ok(typed_file) => {
                all_typed_files.push(typed_file);
            }
            Err(errors) => {
                // Check if any errors are unresolved types that we can try to load on-demand
                let (loadable, other): (Vec<_>, Vec<_>) = errors.into_iter().partition(|e| {
                    e.message.contains("Unresolved type")
                        || e.message.contains("UnresolvedType")
                        || e.message.contains("Cannot find type")
                });

                // Try to load unresolved types on-demand
                let mut any_loaded = false;
                for error in loadable {
                    if let Some(type_name) = self.extract_type_name_from_error(&error.message) {
                        // Skip if we already tried to load this type and it failed
                        if self.failed_type_loads.contains(&type_name) {
                            all_errors.push(error);
                            continue;
                        }
                        if let Err(load_err) = self.load_import_file(&type_name) {
                            debug!("On-demand load failed for {}: {}", type_name, load_err);
                            self.failed_type_loads.insert(type_name.clone());
                            all_errors.push(error);
                        } else {
                            // Successfully loaded! Mark that we should retry
                            any_loaded = true;
                        }
                    } else {
                        all_errors.push(error);
                    }
                }

                // If we successfully loaded any dependencies, retry compiling this file
                if any_loaded {
                    debug!(
                        "  Retrying {} after loading dependencies...",
                        ast_file.filename
                    );
                    match self.compile_pre_parsed_file(ast_file) {
                        Ok(typed_file) => {
                            all_typed_files.push(typed_file);
                        }
                        Err(retry_errors) => {
                            // Still failed after loading dependencies
                            // Check if retry revealed NEW unresolved types that need loading
                            let (retry_loadable, retry_other): (Vec<_>, Vec<_>) =
                                retry_errors.into_iter().partition(|e| {
                                    e.message.contains("Unresolved type")
                                        || e.message.contains("UnresolvedType")
                                        || e.message.contains("Cannot find type")
                                });

                            let mut retry_loaded = false;
                            for error in retry_loadable {
                                if let Some(type_name) =
                                    self.extract_type_name_from_error(&error.message)
                                {
                                    if !self.failed_type_loads.contains(&type_name) {
                                        if let Err(load_err) = self.load_import_file(&type_name) {
                                            debug!(
                                                "On-demand load failed for {}: {}",
                                                type_name, load_err
                                            );
                                            self.failed_type_loads.insert(type_name.clone());
                                            all_errors.push(error);
                                        } else {
                                            retry_loaded = true;
                                        }
                                    } else {
                                        all_errors.push(error);
                                    }
                                } else {
                                    all_errors.push(error);
                                }
                            }

                            // If we loaded more dependencies on retry, try ONE more time
                            if retry_loaded {
                                debug!(
                                    "  Second retry of {} after loading more dependencies...",
                                    ast_file.filename
                                );
                                match self.compile_pre_parsed_file(ast_file) {
                                    Ok(typed_file) => {
                                        all_typed_files.push(typed_file);
                                    }
                                    Err(final_errors) => {
                                        all_errors.extend(final_errors);
                                    }
                                }
                            } else {
                                all_errors.extend(retry_other);
                            }
                        }
                    }
                } else {
                    // No dependencies loaded, keep original errors
                    all_errors.extend(other);
                }
            }
        }
    }

    pub fn typecheck_timings(&self) -> TypecheckStageTimings {
        self.typecheck_timings
    }

    /// Lower all files (stdlib + user) to TAST with full pipeline analysis
    ///
    /// This method delegates to HaxeCompilationPipeline for each file to leverage
    /// the complete analysis infrastructure including:
    /// - Type checking with diagnostics
    /// - Flow-sensitive analysis
    /// - Ownership and lifetime analysis
    /// - Memory safety validation
    ///
    /// Order of compilation:
    /// 1. Stdlib files (with haxe.* package)
    /// 2. Import.hx files (for global imports)
    /// 3. User files (in dependency order - dependencies first)
    ///
    /// On-demand loading: If a type is unresolved, attempts to load and compile
    /// the file that should contain it based on qualified path resolution.
    ///
    /// IMPORTANT: On error, this automatically prints formatted diagnostics to stderr

    pub fn lower_to_tast(&mut self) -> Result<Vec<TypedFile>, Vec<CompilationError>> {
        self.typecheck_timings = TypecheckStageTimings::default();

        // Step 0: Discover @:hlNative metadata in user files and load HDLL plugins
        let t_hdll = profile_timer(self.config.profile_typecheck);
        if self.config.pipeline_config.enable_semantic_analysis {
            self.discover_and_load_hdlls();
        }
        add_profile_ms(&mut self.typecheck_timings.hdll_ms, t_hdll);

        // Step 1: Analyze dependencies for user files
        // Fast path: single-file compilations don't need dependency analysis
        let t_dependency = profile_timer(self.config.profile_typecheck);
        let analysis = if self.user_files.len() <= 1 {
            DependencyAnalysis {
                compilation_order: (0..self.user_files.len()).collect(),
                circular_dependencies: Vec::new(),
            }
        } else {
            match self.analyze_dependencies() {
                Ok(a) => a,
                Err(errors) => {
                    self.print_compilation_errors(&errors);
                    return Err(errors);
                }
            }
        };
        add_profile_ms(&mut self.typecheck_timings.dependency_ms, t_dependency);

        let mut all_typed_files = Vec::new();
        let mut all_errors = Vec::new();

        // Step 2: Pre-load stdlib files for explicit imports AND using statements in user files
        // This ensures typedefs like sys.FileStat are available before compilation
        // Also handles root-level imports like "import StringTools;" and "using StringTools;"
        // Extract imports from already-parsed user file ASTs (no re-parsing needed).
        let t_import_scan = profile_timer(self.config.profile_typecheck);
        let (imports_to_load, usings_to_load): (Vec<String>, Vec<String>) =
            self.user_files.iter().fold(
                (Vec::new(), Vec::new()),
                |(mut imports, mut usings), ast| {
                    for import in &ast.imports {
                        if !import.path.is_empty() {
                            imports.push(import.path.join("."));
                        }
                    }
                    for using in &ast.using {
                        if !using.path.is_empty() {
                            usings.push(using.path.join("."));
                        }
                    }
                    let mut discovered = Vec::new();
                    collect_qualified_type_refs_from_ast(ast, &mut discovered);
                    imports.extend(discovered);
                    let user_deps = Self::extract_all_dependencies(ast);
                    imports.extend(user_deps);
                    (imports, usings)
                },
            );
        add_profile_ms(&mut self.typecheck_timings.import_scan_ms, t_import_scan);

        // Pre-load imports using efficient topological loading (avoids retry loops)
        let mut all_imports = imports_to_load;
        all_imports.extend(usings_to_load);
        let t_import_load = profile_timer(self.config.profile_typecheck);
        let _ = self.load_imports_efficiently(&all_imports);
        add_profile_ms(&mut self.typecheck_timings.import_load_ms, t_import_load);

        // Step 3: Compile import.hx files using SHARED state
        let import_sources: Vec<(String, String)> = self
            .import_hx_files
            .iter()
            .filter_map(|f| f.input.as_ref().map(|s| (f.filename.clone(), s.clone())))
            .collect();

        let t_import_hx = profile_timer(self.config.profile_typecheck);
        for (filename, source) in import_sources {
            match self.compile_file_with_shared_state(&filename, &source) {
                Ok(typed_file) => {
                    all_typed_files.push(typed_file);
                }
                Err(errors) => {
                    all_errors.extend(errors);
                }
            }
        }
        add_profile_ms(&mut self.typecheck_timings.import_hx_ms, t_import_hx);

        // Step 4: Compile user files in dependency order using SHARED state.
        // Use pre-parsed ASTs from self.user_files to avoid re-parsing.
        let user_file_indices: Vec<usize> = analysis.compilation_order.clone();
        let user_context_has_macro_hooks =
            self.user_files.iter().any(haxe_file_source_has_macro_hook)
                || self
                    .import_hx_files
                    .iter()
                    .any(haxe_file_source_has_macro_hook)
                || self
                    .loaded_import_haxe_files
                    .iter()
                    .any(haxe_file_source_has_macro_hook);

        let t_user_files = profile_timer(self.config.profile_typecheck);
        if user_context_has_macro_hooks {
            for idx in user_file_indices {
                let ast_file = self.user_files[idx].clone();
                self.compile_user_ast_collecting_errors(
                    &ast_file,
                    &mut all_typed_files,
                    &mut all_errors,
                );
            }
        } else {
            let user_files = std::mem::take(&mut self.user_files);
            for idx in user_file_indices {
                self.compile_user_ast_collecting_errors(
                    &user_files[idx],
                    &mut all_typed_files,
                    &mut all_errors,
                );
            }
            self.user_files = user_files;
        }
        add_profile_ms(&mut self.typecheck_timings.user_files_ms, t_user_files);

        // Step 5: Report all errors if any were found
        if !all_errors.is_empty() {
            self.print_compilation_errors(&all_errors);
            return Err(all_errors);
        }

        // Step 6: Include loaded stdlib files (typedefs, etc.) in the result
        // These were loaded on-demand during import resolution and contain type aliases
        // that need to be processed by HIR
        let t_result_stdlib = profile_timer(self.config.profile_typecheck);
        for stdlib_file in std::mem::take(&mut self.loaded_stdlib_typed_files) {
            all_typed_files.push(stdlib_file);
        }
        add_profile_ms(
            &mut self.typecheck_timings.result_stdlib_ms,
            t_result_stdlib,
        );

        self.maybe_dump_file_table();

        Ok(all_typed_files)
    }

    /// When `RAYZOR_DUMP_JIT_MAP=1`, write `/tmp/rayzor_file_table.csv`
    /// containing `file_id,path` rows for every file seen by the compiler.
    /// Pairs with `/tmp/rayzor_jit_symbols.csv` (whose `file_id` column
    /// references rows in this table) so off-line resolvers can map a
    /// PC → `IrFunction` → `Haxe file:line`.
    pub fn maybe_dump_file_table(&self) {
        if std::env::var_os("RAYZOR_DUMP_JIT_MAP").as_deref() != Some(std::ffi::OsStr::new("1")) {
            return;
        }
        let path = "/tmp/rayzor_file_table.csv";
        let mut out = String::from("file_id,path\n");
        let mut rows: Vec<(&String, u32)> = self
            .file_id_by_filename
            .iter()
            .map(|(name, id)| (name, *id))
            .collect();
        rows.sort_by_key(|r| r.1);
        for (name, id) in rows {
            let safe = name.replace('"', "''");
            out.push_str(&format!("{},\"{}\"\n", id, safe));
        }
        let n_files = self.file_id_by_filename.len();
        match std::fs::write(path, out) {
            Ok(_) => eprintln!("[jit-map] wrote {} file paths to {}", n_files, path),
            Err(e) => eprintln!("[jit-map] failed to write {}: {}", path, e),
        }
    }

    /// Extract the type name from an unresolved type error message
    fn extract_type_name_from_error(&self, message: &str) -> Option<String> {
        // Try to extract type name from error message formats:
        // "UnresolvedType { type_name: \"haxe.iterators.ArrayIterator\", ... }"
        // "Unresolved type: haxe.iterators.ArrayIterator"
        let type_name = if let Some(start) = message.find("type_name: \"") {
            let start = start + "type_name: \"".len();
            if let Some(end) = message[start..].find('"') {
                Some(message[start..start + end].to_string())
            } else {
                None
            }
        } else if let Some(start) = message.find("Unresolved type: ") {
            let start = start + "Unresolved type: ".len();
            let end = message[start..]
                .find(|c: char| !c.is_alphanumeric() && c != '.')
                .unwrap_or(message.len() - start);
            Some(message[start..start + end].to_string())
        } else if let Some(start) = message.find("Cannot find type '") {
            let start = start + "Cannot find type '".len();
            if let Some(end) = message[start..].find('\'') {
                Some(message[start..start + end].to_string())
            } else {
                None
            }
        } else {
            None
        };

        // Filter out generic type parameters and built-in typedefs:
        // - Single uppercase letters (T, K, V, E, R, etc.)
        // - Short names like "TKey", "TValue", etc.
        // - Built-in typedefs from StdTypes.hx (Iterator, KeyValueIterator, etc.)
        // These should NOT be treated as importable types
        if let Some(ref name) = type_name {
            // Skip single uppercase letter type parameters
            if name.len() == 1
                && name
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_uppercase())
                    .unwrap_or(false)
            {
                return None;
            }
            // Skip common generic type parameter patterns
            if name == "Key" || name == "Value" || name == "Item" || name == "Element" {
                return None;
            }
            // Skip built-in top-level types from StdTypes.hx (these are already loaded).
            if Self::is_stdtypes_prelude_type_name(name) {
                debug!("  Filtering out StdTypes prelude type: {}", name);
                return None;
            }
        }

        type_name
    }

    /// Try to load a cached MIR module from a BLADE file
    ///
    /// Returns Some(IrModule) if cache is valid, None if cache doesn't exist or is stale
    pub fn try_load_cached(&self, source_path: &Path) -> Option<IrModule> {
        if !self.config.enable_cache {
            return None;
        }

        let cache_path = self.config.get_cache_path(source_path);
        if !cache_path.exists() {
            return None;
        }

        // Load BLADE file
        let (mir_module, metadata, _symbols, _cached_maps) = match load_blade(&cache_path) {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to load cache for {:?}: {}", source_path, e);
                return None;
            }
        };

        // Check if source file has been modified since cache was created
        if let Ok(source_meta) = std::fs::metadata(source_path) {
            if let Ok(modified) = source_meta.modified() {
                let source_timestamp = modified
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Cache is stale if source was modified after cache was created
                if source_timestamp > metadata.compile_timestamp {
                    if self.config.enable_cache {
                        debug!(
                            "Cache stale for {:?} (source: {}, cache: {})",
                            source_path, source_timestamp, metadata.compile_timestamp
                        );
                    }
                    return None;
                }
            }
        }

        if let Ok(source) = std::fs::read_to_string(source_path) {
            let current_hash = self.hash_source_for_config(&source);
            if metadata.source_hash != current_hash {
                if self.config.enable_cache {
                    debug!("Cache source hash mismatch for {:?}", source_path);
                }
                return None;
            }
        }

        // Check compiler version matches
        let current_version = env!("CARGO_PKG_VERSION");
        if metadata.compiler_version != current_version {
            if self.config.enable_cache {
                debug!(
                    "Cache version mismatch for {:?} (cache: {}, current: {})",
                    source_path, metadata.compiler_version, current_version
                );
            }
            return None;
        }

        // Check compiler cache ABI id matches. Parser/lowerer/MIR-shape
        // changes within the same semver bump the id (see compiler/build.rs)
        // and can silently shift function IDs or AST structure for the same
        // source — without this guard, MIR cached by an older compiler loads
        // into a newer compiler and surfaces as SIGILL at unrelated call sites.
        let current_build_id = env!("RAYZOR_BUILD_ID");
        if metadata.build_id != current_build_id {
            if self.config.enable_cache {
                debug!(
                    "Cache build-id mismatch for {:?} (cache: {}, current: {})",
                    source_path, metadata.build_id, current_build_id
                );
            }
            return None;
        }

        if self.config.enable_cache {
            debug!("Cache hit for {:?}", source_path);
        }

        Some(mir_module)
    }

    /// Save a compiled MIR module to the BLADE cache
    pub fn save_to_cache(&self, source_path: &Path, module: &IrModule) -> Result<(), String> {
        if !self.config.enable_cache {
            return Ok(());
        }

        let cache_path = self.config.get_cache_path(source_path);

        // Ensure cache directory exists
        if let Some(parent) = cache_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        }

        // Get source file timestamp and compute hash
        let source_timestamp = std::fs::metadata(source_path)
            .and_then(|m| m.modified())
            .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(0);

        // Read source for hash computation
        let source_hash = std::fs::read_to_string(source_path)
            .map(|s| self.hash_source_for_config(&s))
            .unwrap_or(0);

        let compile_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create metadata
        let metadata = BladeMetadata {
            name: module.name.clone(),
            source_path: source_path.to_string_lossy().to_string(),
            source_hash,
            source_timestamp,
            compile_timestamp,
            dependencies: Vec::new(), // TODO: Track dependencies for proper invalidation
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            build_id: env!("RAYZOR_BUILD_ID").to_string(),
        };

        // Save to BLADE file (no type info/maps for standalone compile command)
        save_blade_with_state(&cache_path, module, metadata, None, None)
            .map_err(|e| format!("Failed to save cache: {}", e))?;

        if self.config.enable_cache {
            debug!("Cached MIR for {:?} -> {:?}", source_path, cache_path);
        }

        Ok(())
    }

    /// Clear all cached BLADE files
    pub fn clear_cache(&self) -> Result<(), String> {
        let cache_dir = self.config.get_cache_dir();
        if cache_dir.exists() {
            std::fs::remove_dir_all(&cache_dir)
                .map_err(|e| format!("Failed to clear cache: {}", e))?;
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| format!("Failed to recreate cache directory: {}", e))?;
            debug!("Cache cleared: {:?}", cache_dir);
        }
        Ok(())
    }

    /// Print compilation errors with formatted diagnostics to stderr.
    /// Uses the diagnostics crate's ErrorFormatter for consistent formatting.
    pub fn print_compilation_errors(&self, errors: &[CompilationError]) {
        use diagnostics::ErrorFormatter;

        // Build the same full source_map the MIR-diagnostic path uses,
        // so hard-error spans on cross-file types resolve to the right
        // file. See `build_full_source_map` for the inclusion order.
        let source_map = self.build_full_source_map();

        let formatter = ErrorFormatter::with_colors();

        for error in errors {
            let diagnostic = error.to_diagnostic(&source_map);
            let formatted = formatter.format_diagnostic(&diagnostic, &source_map);
            eprint!("{}", formatted);
        }
    }

    /// Check for ownership violations (use-after-move) in the TAST.
    /// Returns diagnostics that can be printed via print_mir_diagnostics.
    fn check_ownership_violations(&self, typed_file: &TypedFile) -> Vec<diagnostics::Diagnostic> {
        use crate::semantic_graph::{MoveType, OwnershipGraph};
        use crate::tast::{ScopeId, TypedExpressionKind, TypedStatement};

        let mut ownership_graph = OwnershipGraph::new();

        // Walk all classes and standalone functions to populate the ownership graph
        for class in &typed_file.classes {
            for method in &class.methods {
                let scope = ScopeId::from_raw(method.symbol_id.as_raw());
                for param in &method.parameters {
                    ownership_graph.add_variable(param.symbol_id, param.param_type, scope);
                }
                Self::populate_ownership_stmts(&mut ownership_graph, &method.body);
            }
            for ctor in &class.constructors {
                let scope = ScopeId::from_raw(ctor.symbol_id.as_raw());
                for param in &ctor.parameters {
                    ownership_graph.add_variable(param.symbol_id, param.param_type, scope);
                }
                Self::populate_ownership_stmts(&mut ownership_graph, &ctor.body);
            }
        }
        for func in &typed_file.functions {
            let scope = ScopeId::from_raw(func.symbol_id.as_raw());
            for param in &func.parameters {
                ownership_graph.add_variable(param.symbol_id, param.param_type, scope);
            }
            Self::populate_ownership_stmts(&mut ownership_graph, &func.body);
        }

        // Check for use-after-move violations
        let violations = ownership_graph.check_use_after_move();
        let mut diagnostics = Vec::new();

        // Build a trait checker so we can drop violations on `Copy` types —
        // Int / Float / Bool and any class deriving Copy are pass-by-value
        // semantically (matches Haxe + Rust), so `i++` in a for-loop or
        // passing an Int to a function does NOT consume the variable. The
        // ownership graph conservatively records all variable references
        // as moves; this filter removes the false positives at diagnostic
        // emission time. Classes that genuinely need move semantics (no
        // Copy derive) still surface their warnings.
        // Stdlib classes (Tensor, QTensor, …) live in `loaded_stdlib_typed_files`,
        // not on the current `typed_file`. Without folding them into the trait
        // checker's class map, `@:move` annotations on stdlib types are silently
        // inert at user-call sites (requires_strict_move would return false
        // because the class wouldn't be found at all). Chain every loaded stdlib
        // file's classes into the lookup so cross-file move semantics fire.
        let mut trait_checker = crate::tast::trait_checker::TraitChecker::new(
            self.type_table.as_ref(),
            &self.symbol_table,
            &self.string_interner,
            &typed_file.classes,
        );
        for stdlib_file in &self.loaded_stdlib_typed_files {
            trait_checker = trait_checker.extend_classes(&stdlib_file.classes);
        }

        for violation in violations {
            if let crate::semantic_graph::OwnershipViolation::UseAfterMove {
                variable,
                use_location,
                move_location,
                ..
            } = violation
            {
                // Skip if the variable is Copy — `i` in `i++`, primitives
                // passed to functions, etc. shouldn't fire the warning.
                // Also decide up-front whether the variable's class is
                // `@:move`-annotated; if so, the diagnostic is a hard error
                // (linear/affine semantics) rather than a soft warning.
                // `@:move` (strict_q) takes precedence over auto-Copy: when the
                // user explicitly opts into move semantics, we must NOT silently
                // treat the value as Copy even if all its fields happen to be
                // Copy-able. `is_copy` only skips the diagnostic when there is
                // no `@:move` annotation.
                let mut strict = false;
                if let Some(node) = ownership_graph.variables.get(&variable) {
                    if node.variable_type.is_valid() {
                        // `@:shared` short-circuits the entire diagnostic.
                        // Bindings of shared classes (e.g. rayzor.ds.Tensor)
                        // are reference-counted at runtime; aliasing them
                        // after a `.clone()` (which is now an atomic
                        // refcount increment) is not a use-after-move and
                        // must not produce E0382 — neither error nor
                        // warning. Skip ahead of the is_copy check so we
                        // don't even traverse the per-callsite work.
                        if trait_checker.requires_shared(node.variable_type) {
                            continue;
                        }
                        let strict_q = trait_checker.requires_strict_move(node.variable_type);
                        if !strict_q && trait_checker.is_copy(node.variable_type) {
                            continue;
                        }
                        strict = strict_q;
                    }
                }

                let var_name = self.get_symbol_name(variable, typed_file);
                // Opt-in debug for triaging E0382 sites — set RAYZOR_DEBUG_E0382 to
                // print each violation's (var, symbol, file, line, col) so the
                // diagnostic's "Main.hx fallback" rendering can be cross-referenced
                // against the real source location.
                if std::env::var("RAYZOR_DEBUG_E0382").is_ok() {
                    eprintln!("[E0382-DEBUG] var={} sym={} typed_file={} severity={} move_file_id={} move_line={} move_col={} use_file_id={} use_line={} use_col={}",
                        var_name,
                        variable.as_raw(),
                        typed_file.metadata.file_path,
                        if strict { "Error" } else { "Warning" },
                        move_location.file_id,
                        move_location.line,
                        move_location.column,
                        use_location.file_id,
                        use_location.line,
                        use_location.column,
                    );
                }
                let file_id = diagnostics::FileId::new(use_location.file_id as usize);
                // Span the entire identifier — we know `var_name` so the
                // end position is start + len-bytes. Previously this
                // was start + 1, highlighting just the first character.
                // var_name is the source identifier as the parser saw it
                // (Haxe identifiers are ASCII-only so byte_len == char_len
                // for column math).
                let name_byte_len = var_name.len();
                let use_start = diagnostics::SourcePosition::new(
                    use_location.line as usize,
                    use_location.column as usize,
                    use_location.byte_offset as usize,
                );
                let use_end = diagnostics::SourcePosition::new(
                    use_location.line as usize,
                    use_location.column as usize + name_byte_len,
                    use_location.byte_offset as usize + name_byte_len,
                );
                let use_span = diagnostics::SourceSpan::new(use_start, use_end, file_id);

                let move_start = diagnostics::SourcePosition::new(
                    move_location.line as usize,
                    move_location.column as usize,
                    move_location.byte_offset as usize,
                );
                let move_end = diagnostics::SourcePosition::new(
                    move_location.line as usize,
                    move_location.column as usize + name_byte_len,
                    move_location.byte_offset as usize + name_byte_len,
                );
                let move_span = diagnostics::SourceSpan::new(move_start, move_end, file_id);

                let help = if strict {
                    vec![
                        format!(
                            "`{}` is declared `@:move`, so its values cannot be aliased after a move.",
                            var_name
                        ),
                        format!(
                            "Clone the value explicitly (`var copy = {}.clone();`) or restructure the code so the original binding is no longer reachable.",
                            var_name
                        ),
                    ]
                } else {
                    vec![format!(
                        "Consider cloning: `var copy = {}.clone();`",
                        var_name
                    )]
                };
                let diag = diagnostics::Diagnostic {
                    severity: if strict {
                        diagnostics::DiagnosticSeverity::Error
                    } else {
                        diagnostics::DiagnosticSeverity::Warning
                    },
                    code: Some("E0382".to_string()),
                    message: format!("use of moved value: `{}`", var_name),
                    span: use_span.clone(),
                    labels: vec![
                        diagnostics::Label::primary(use_span, "value used here after move"),
                        diagnostics::Label::secondary(move_span, "value moved here"),
                    ],
                    suggestions: vec![],
                    notes: vec![],
                    help,
                };
                diagnostics.push(diag);
            }
        }

        diagnostics
    }

    /// Walk statements to populate ownership graph (moves and uses).
    fn populate_ownership_stmts(
        graph: &mut crate::semantic_graph::OwnershipGraph,
        stmts: &[crate::tast::TypedStatement],
    ) {
        use crate::semantic_graph::MoveType;
        use crate::tast::{ScopeId, TypedExpressionKind, TypedStatement};

        for stmt in stmts {
            match stmt {
                TypedStatement::VarDeclaration {
                    symbol_id,
                    var_type,
                    initializer,
                    ..
                } => {
                    let scope = ScopeId::from_raw(symbol_id.as_raw());
                    graph.add_variable(*symbol_id, *var_type, scope);
                    if let Some(init) = initializer {
                        if let TypedExpressionKind::Variable { symbol_id: src } = &init.kind {
                            graph.add_move(
                                *src,
                                Some(*symbol_id),
                                init.source_location,
                                MoveType::Explicit,
                            );
                        }
                        Self::populate_ownership_expr(graph, init);
                    }
                }
                TypedStatement::Expression { expression, .. } => {
                    Self::populate_ownership_expr(graph, expression);
                }
                TypedStatement::Return { value, .. } => {
                    if let Some(expr) = value {
                        Self::populate_ownership_expr(graph, expr);
                    }
                }
                TypedStatement::If {
                    condition,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    Self::populate_ownership_expr(graph, condition);
                    Self::populate_ownership_stmts(
                        graph,
                        std::slice::from_ref(then_branch.as_ref()),
                    );
                    if let Some(else_stmt) = else_branch {
                        Self::populate_ownership_stmts(
                            graph,
                            std::slice::from_ref(else_stmt.as_ref()),
                        );
                    }
                }
                TypedStatement::While {
                    condition, body, ..
                } => {
                    Self::populate_ownership_expr(graph, condition);
                    Self::populate_ownership_stmts(graph, std::slice::from_ref(body.as_ref()));
                }
                TypedStatement::Block { statements, .. } => {
                    Self::populate_ownership_stmts(graph, statements);
                }
                _ => {}
            }
        }
    }

    /// Walk expressions to record moves (function call args) and uses (variable refs).
    fn populate_ownership_expr(
        graph: &mut crate::semantic_graph::OwnershipGraph,
        expr: &crate::tast::TypedExpression,
    ) {
        use crate::semantic_graph::MoveType;
        use crate::tast::TypedExpressionKind;

        match &expr.kind {
            TypedExpressionKind::Variable { symbol_id } => {
                graph.record_use(*symbol_id, expr.source_location);
            }
            TypedExpressionKind::FieldAccess { object, .. } => {
                Self::populate_ownership_expr(graph, object);
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                Self::populate_ownership_expr(graph, function);
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    Self::populate_ownership_expr(graph, arg);
                }
            }
            TypedExpressionKind::MethodCall {
                receiver,
                arguments,
                ..
            } => {
                Self::populate_ownership_expr(graph, receiver);
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    Self::populate_ownership_expr(graph, arg);
                }
            }
            TypedExpressionKind::StaticMethodCall { arguments, .. } => {
                for arg in arguments {
                    if let TypedExpressionKind::Variable { symbol_id } = &arg.kind {
                        graph.add_move(
                            *symbol_id,
                            None,
                            arg.source_location,
                            MoveType::FunctionCall,
                        );
                    }
                    Self::populate_ownership_expr(graph, arg);
                }
            }
            TypedExpressionKind::Block { statements, .. } => {
                Self::populate_ownership_stmts(graph, statements);
            }
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                Self::populate_ownership_expr(graph, condition);
                Self::populate_ownership_expr(graph, then_expr);
                if let Some(e) = else_expr {
                    Self::populate_ownership_expr(graph, e);
                }
            }
            TypedExpressionKind::BinaryOp { left, right, .. } => {
                Self::populate_ownership_expr(graph, left);
                Self::populate_ownership_expr(graph, right);
            }
            TypedExpressionKind::UnaryOp { operand, .. } => {
                Self::populate_ownership_expr(graph, operand);
            }
            TypedExpressionKind::ArrayAccess { array, index, .. } => {
                Self::populate_ownership_expr(graph, array);
                Self::populate_ownership_expr(graph, index);
            }
            _ => {}
        }
    }

    /// Get variable name from SymbolId via symbol table.
    fn get_symbol_name(&self, symbol: crate::tast::SymbolId, _typed_file: &TypedFile) -> String {
        if let Some(sym) = self.symbol_table.get_symbol(symbol) {
            if let Some(name) = self.string_interner.get(sym.name) {
                if !name.is_empty() {
                    return name.to_string();
                }
            }
        }
        format!("var_{}", symbol.as_raw())
    }

    /// Print diagnostics from MIR lowering using the diagnostics formatter.
    /// The source map is built with the user file at FileId 0 to match the
    /// compiler's SourceLocation.file_id convention (user file = 0).
    fn print_mir_diagnostics(&mut self, mir_diagnostics: &[diagnostics::Diagnostic]) {
        use diagnostics::ErrorFormatter;

        // Store for cache replay
        self.collected_diagnostics
            .extend_from_slice(mir_diagnostics);

        // Build the full source_map covering user files + stdlib + imports +
        // on-demand-compiled package files. Cross-file diagnostics (E0382 on
        // a moved Tensor binding inside nue.sampling.LocalTempSampler, W0014
        // on a cross-context iface return inside nue.transformer.*, etc.)
        // carry the file_id of the ACTUAL source file. Without the imported
        // files registered here, the formatter falls back to whichever
        // file happens to occupy that file_id slot in the under-populated
        // map — typically the entry Main.hx — and uses the diagnostic's
        // line number as an offset INTO Main.hx, so the warning ends up
        // cited against an unrelated comment or statement.
        let source_map = self.build_full_source_map();

        let formatter = ErrorFormatter::with_colors();
        for diagnostic in mir_diagnostics {
            if diagnostic.code.as_deref() == Some("W0014")
                && std::env::var_os("RAYZOR_SHOW_W0014").is_none()
                && std::env::var_os("RAYZOR_SHOW_ADVICE").is_none()
            {
                continue;
            }
            let formatted = formatter.format_diagnostic(diagnostic, &source_map);
            eprint!("{}", formatted);
        }
    }

    /// Construct a SourceMap that registers every Haxe file the
    /// current compilation knows about, **at the same file_id** the
    /// lowering pipeline assigned to it via `file_id_by_filename`.
    ///
    /// SourceMap::add_file auto-assigns FileIds sequentially from 0,
    /// so the order we insert here determines the file_ids in the
    /// rendered output. We sort the (filename, file_id) pairs by
    /// file_id and insert in that order — if there are no gaps, the
    /// resulting source_map's FileIds align with the diagnostic
    /// spans' file_ids. Where the lowering pipeline emitted a file_id
    /// without populating file_id_by_filename (parser cache, generated
    /// modules, etc.), we still see file_id=0 fallthrough for those
    /// specific spans, but every cross-file user import resolves
    /// correctly.
    fn build_full_source_map(&self) -> diagnostics::SourceMap {
        let mut source_map = diagnostics::SourceMap::new();

        // Collect every file source we know about, keyed by filename.
        // file_source_by_filename is AUTHORITATIVE — it holds the exact
        // bytes the span_converter saw at lowering time, so byte_offset
        // → line/column resolution at the renderer matches what was
        // stored on diagnostic spans. The other lists are FALLBACKS
        // for files that bypassed `compile_ast_with_shared_state`
        // (e.g. parser-cache hits, stdlib pre-loads); their input
        // fields are typically the same bytes anyway.
        let mut sources: BTreeMap<String, String> = BTreeMap::new();
        for user_file in &self.user_files {
            if let Some(ref source) = user_file.input {
                sources.insert(user_file.filename.clone(), source.clone());
            }
        }
        for stdlib_file in &self.stdlib_files {
            if let Some(ref source) = stdlib_file.input {
                sources.insert(stdlib_file.filename.clone(), source.clone());
            }
        }
        for import_file in &self.import_hx_files {
            if let Some(ref source) = import_file.input {
                sources.insert(import_file.filename.clone(), source.clone());
            }
        }
        for import_haxe_file in &self.loaded_import_haxe_files {
            if let Some(ref source) = import_haxe_file.input {
                sources.insert(import_haxe_file.filename.clone(), source.clone());
            }
        }
        // Authoritative source: bytes captured at lowering entry. These
        // OVERWRITE any input-field copy so the renderer ALWAYS sees
        // the exact bytes the byte_offset was computed against.
        for (filename, source) in &self.file_source_by_filename {
            sources.insert(filename.clone(), source.clone());
        }
        for (filename, _) in &self.compiled_files {
            if !sources.contains_key(filename) {
                if let Ok(source) = std::fs::read_to_string(filename) {
                    sources.insert(filename.clone(), source);
                }
            }
        }

        // Sort file_id_by_filename by file_id ascending so the
        // sequential add_file calls land each file at the expected
        // FileId. Fall back to reading from disk if we don't have an
        // in-memory copy.
        let mut id_pairs: Vec<(&String, u32)> = self
            .file_id_by_filename
            .iter()
            .map(|(name, id)| (name, *id))
            .collect();
        id_pairs.sort_by_key(|(_, id)| *id);

        let mut next_expected: u32 = 0;
        for (filename, file_id) in id_pairs {
            // Fill any gaps with empty placeholder entries so subsequent
            // files land at their correct FileId. Gaps can occur when a
            // file_id was reserved but the file failed to compile.
            while next_expected < file_id {
                source_map.add_file(format!("<unknown:{}>", next_expected), String::new());
                next_expected += 1;
            }
            let content = sources
                .get(filename)
                .cloned()
                .or_else(|| std::fs::read_to_string(filename).ok())
                .unwrap_or_default();
            source_map.add_file(filename.clone(), content);
            next_expected += 1;
        }

        source_map
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        let cache_dir = self.config.get_cache_dir();
        let mut stats = CacheStats::default();

        if !cache_dir.exists() {
            return stats;
        }

        // Count .blade files and calculate total size
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("blade") {
                        stats.cached_modules += 1;
                        stats.total_size_bytes += metadata.len();
                    }
                }
            }
        }

        stats
    }

    /// Get the MIR modules that were generated during compilation.
    /// Returns a vector of MIR modules corresponding to the compiled files.
    pub fn get_mir_modules(&self) -> Vec<std::sync::Arc<crate::ir::IrModule>> {
        self.mir_modules.clone()
    }

    /// Finalize cross-module MIR references after all imports and user files
    /// have been lowered. Artifact builders such as AOT clone MIR directly
    /// after `lower_to_tast()`, so they need the same coherency pass that the
    /// interactive compile path runs before codegen.
    pub fn finalize_mir_references(&mut self) {
        self.fixup_stale_cross_module_refs();
        self.fixup_stale_constructor_ids();
        self.fixup_stale_method_ids();
    }

    /// Get extern function → JS module name mappings (from @:jsImport classes).
    pub fn get_extern_js_module_map(&self) -> &BTreeMap<String, String> {
        &self.extern_js_module_map
    }

    pub fn get_qualified_method_map(&self) -> &BTreeMap<String, String> {
        &self.qualified_method_map
    }

    /// Get class allocation sizes keyed by class name.
    /// Used by WASM bindgen to generate JS constructors that call malloc(size).
    pub fn get_class_alloc_sizes_by_name(&self) -> &BTreeMap<String, u64> {
        &self.import_class_alloc_sizes_by_name
    }

    /// Get HDLL function pointers for JIT linking.
    ///
    /// Returns symbol name and pointer pairs collected from all loaded HDLL plugins.
    /// These should be merged with runtime symbols when creating the backend.
    pub fn get_hdll_symbols(&self) -> &[(String, *const u8)] {
        &self.hdll_symbols
    }

    /// Register extern class methods from a TypedFile as plugin mappings.
    ///
    /// When an imported file contains an extern class (e.g., GPUDevice with @:native methods),
    /// this extracts the method signatures and registers them as NativePlugin entries.
    /// This makes them visible to the MIR lowerer's StdlibMapping, which otherwise only
    /// knows about methods from rpkg NativePlugins.
    /// Seed `import_property_access_map` from any property fields declared in
    /// the given typed file. Without this, extern-only stdlib files (whose MIR
    /// generation is skipped via the `is_extern_only` shortcut in
    /// `compile_file`) never populate the property accessor map, so user code
    /// like `tls.value` falls through to a "field not found" error in MIR
    /// `lower_field_access`. Each property field's `PropertyAccessInfo` is
    /// keyed by the field's SymbolId.
    fn seed_property_accessors_from_typed_file(&mut self, typed_file: &TypedFile) {
        for class in &typed_file.classes {
            for field in &class.fields {
                if let Some(prop_info) = field.property_access.as_ref() {
                    self.import_property_access_map
                        .entry(field.symbol_id)
                        .or_insert_with(|| prop_info.clone());
                }
            }
        }
    }

    fn register_extern_methods_from_typed_file(&mut self, typed_file: &TypedFile) {
        use crate::compiler_plugin::NativePlugin;
        use crate::rpkg::MethodDescEntry;

        // Snapshot the builtin stdlib mapping so we can skip classes that already have
        // explicit MIR wrapper mappings (e.g., rayzor.concurrent.Thread, rayzor.Bytes).
        // Without this skip, naive @:native("spawn") auto-registration overrides the
        // correct Thread_spawn MIR wrapper with a bare "spawn" bare-name symbol, which
        // then fails to resolve at JIT time ("can't resolve symbol spawn").
        let builtin_mapping = crate::stdlib::runtime_mapping::StdlibMapping::new();

        let mut entries: Vec<MethodDescEntry> = Vec::new();

        for class in &typed_file.classes {
            // Check if this is an extern class by looking up the symbol's flags
            let is_extern = self
                .symbol_table
                .get_symbol(class.symbol_id)
                .map(|s| {
                    s.flags.contains(crate::tast::symbols::SymbolFlags::EXTERN)
                        || s.flags.is_native()
                })
                .unwrap_or(false);

            if !is_extern {
                continue;
            }

            // Get the class's native name (from @:native metadata)
            let class_native_name = self
                .symbol_table
                .get_symbol(class.symbol_id)
                .and_then(|s| {
                    s.native_name
                        .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                })
                .or_else(|| {
                    self.symbol_table.get_symbol(class.symbol_id).and_then(|s| {
                        s.qualified_name
                            .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                    })
                })
                .unwrap_or_default();

            if class_native_name.is_empty() {
                continue;
            }
            // Also get the dot-separated qualified name for stdlib mapping lookups
            // (MIR lowerer queries with dots, not ::)
            let class_dot_name = class_native_name.replace("::", ".");
            let underscore_class_name = class_native_name.replace("::", "_");

            // Extract method entries
            for method in &class.methods {
                // A concrete (body-bearing) method on an extern class is compiled
                // to MIR and dispatched as a direct call. Registering it as a
                // bare-name extern mapping would make the call resolve to a
                // runtime symbol that does not exist ("can't resolve symbol X").
                // Only the bodyless `@:native` methods belong in this table.
                if !method.body.is_empty() {
                    continue;
                }
                let method_name = self
                    .string_interner
                    .get(method.name)
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let symbol_name = self
                    .symbol_table
                    .get_symbol(method.symbol_id)
                    .and_then(|s| {
                        s.native_name
                            .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                    })
                    .unwrap_or_else(|| method_name.clone());

                if symbol_name.is_empty() {
                    continue;
                }

                // If the builtin stdlib already has a mapping for this (class,
                // method), use the builtin's canonical runtime_name instead of the
                // bare @:native value. The BuiltinPlugin (priority 0) carries the
                // correct symbol names for all stdlib methods:
                //   - MIR wrappers:      Thread_spawn, Channel_send, ...
                //   - Extern C functions: haxe_bytes_get, sys_thread_join, ...
                //
                // Without this override, the NativePlugin auto-registration
                // (priority 10) would replace them with bare @:native symbols
                // ("spawn", "join"), which then fail at JIT link time
                // ("can't resolve symbol spawn"). For MIR wrappers we also skip
                // the NativePlugin entry entirely since the builtin path handles
                // them via stdlib MIR module merging.
                // Match the builtin mapping by the qualified (underscored) class
                // name first, then by the simple class name. The builtin mapping
                // registers map_method! entries with simple names like "StringMap",
                // "IntMap", etc. — the `class_matches` helper supports suffix-
                // matching the other direction ("StringMap" matches
                // "haxe_ds_StringMap"), but NOT the lookup direction. Without the
                // simple-name fallback the override never fires for haxe.ds.*
                // classes and the auto-registration registers
                // `(class="haxe.ds.StringMap", method="set", runtime="set")` as a
                // bogus mapping that the dispatch then picks up — see the user's
                // "no silent dispatch fallthrough" feedback.
                // Try multiple naming conventions to find the builtin mapping.
                // `class_native_name` can come in as `haxe::ds::StringMap`,
                // `haxe.ds.StringMap`, or `haxe_ds_StringMap` depending on the
                // upstream source — and the simple bare name (`StringMap`) is
                // what the macro-registered map_method! entries use.
                let bare_name = underscore_class_name
                    .rsplit(['_', '.'])
                    .next()
                    .unwrap_or(&underscore_class_name);
                let builtin_match = builtin_mapping
                    .find_by_name(&underscore_class_name, &method_name)
                    .or_else(|| builtin_mapping.find_by_name(&class_dot_name, &method_name))
                    .or_else(|| builtin_mapping.find_by_name(bare_name, &method_name));
                if let Some((_sig, call)) = builtin_match {
                    // DECLARATION vs RUNTIME arity. The lookup above is three
                    // name-only attempts with no arity awareness, so a method
                    // whose .hx signature promises more parameters than the
                    // runtime implements still binds here — and every later call
                    // inherits that binding. A call passing the extra argument is
                    // then emitted against a native symbol of a different arity:
                    // LLVM rejects the module ("Incorrect number of arguments"),
                    // the tier falls back, and the enclosing function silently
                    // emits NOTHING. No compile error, no runtime message.
                    //
                    // Known live cases: `Sys.command(cmd, ?args)` maps to
                    // `haxe_sys_command` (params: 1 — `?args` is unimplemented),
                    // and `File.read(path, binary = true)` maps to
                    // `file_read_default` (params: 1). Both are real gaps between
                    // the .hx surface and the runtime, not compiler confusion.
                    //
                    // Loud rather than fatal by default; RAYZOR_STRICT_STDLIB_ARITY=1
                    // rejects the binding instead (the intended end state).
                    let declared = method.parameters.len();
                    // OFF by default: the six known stdlib mismatches would print
                    // on every compile, which is noise until they are fixed (and
                    // buries any NEW mismatch in it). RAYZOR_STDLIB_ARITY_WARN=1
                    // lists them; RAYZOR_STRICT_STDLIB_ARITY=1 still rejects the
                    // binding regardless of the warning setting.
                    let warn_arity = std::env::var_os("RAYZOR_STDLIB_ARITY_WARN").is_some();
                    if declared != call.param_count && warn_arity {
                        eprintln!(
                            "[stdlib-arity] {}.{} declares {} parameter(s) but runtime mapping '{}' takes {} — a call supplying the extra argument(s) will be emitted against a mismatched native signature and silently produce nothing. Either implement the parameter in the runtime or narrow the .hx declaration.",
                            bare_name, method_name, declared, call.runtime_name, call.param_count
                        );
                    }
                    if declared != call.param_count
                        && std::env::var_os("RAYZOR_STRICT_STDLIB_ARITY").is_some()
                    {
                        continue;
                    }
                    // Record the mapping under the Haxe-qualified name so the
                    // WASM backend stub redirect still finds a canonical symbol.
                    let class_haxe_name = self
                        .symbol_table
                        .get_symbol(class.symbol_id)
                        .and_then(|s| {
                            s.qualified_name
                                .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                        })
                        .unwrap_or_default();
                    if !class_haxe_name.is_empty() {
                        let qualified = format!("{}.{}", class_haxe_name, method_name);
                        self.qualified_method_map
                            .insert(qualified, call.runtime_name.to_string());
                    }
                    if call.is_mir_wrapper {
                        continue;
                    }
                    // Non-MIR-wrapper (extern C): skip the NativePlugin entry too so
                    // that MIR lowering uses the builtin's correctly-typed mapping.
                    continue;
                }

                // Build qualified Haxe name for WASM stub resolution.
                // The MIR lowerer creates wrapper functions named e.g. "rayzor.gpu.Surface.getFormat"
                // from the Haxe package + class + method name.
                let class_haxe_name = self
                    .symbol_table
                    .get_symbol(class.symbol_id)
                    .and_then(|s| {
                        s.qualified_name
                            .and_then(|n| self.string_interner.get(n).map(|s| s.to_string()))
                    })
                    .unwrap_or_default();
                if !class_haxe_name.is_empty() && !symbol_name.is_empty() {
                    let qualified = format!("{}.{}", class_haxe_name, method_name);
                    self.qualified_method_map
                        .insert(qualified, symbol_name.clone());
                }

                // Real declared native-type tags for the return AND each param,
                // so a scalar (`:Int`/`:Bool`/`:Float`) doesn't decay to a boxed
                // PtrVoid across the module boundary. A C-ABI extern takes its
                // args by value; the plugin_match lowering marshals per this
                // signature, so a scalar param declared here as PtrVoid would be
                // boxed (the arg arriving as a DynamicValue pointer the kernel
                // then misreads). The leading self slot on instance methods is a
                // real pointer (tag 3).
                let return_tag = self.haxe_type_to_native_tag(method.return_type);
                let mut param_tags: Vec<u8> = Vec::with_capacity(method.parameters.len() + 1);
                if !method.is_static {
                    param_tags.push(3); // self pointer
                }
                for p in &method.parameters {
                    param_tags.push(self.haxe_type_to_native_tag(p.param_type));
                }
                entries.push(MethodDescEntry {
                    symbol_name,
                    class_name: class_native_name.clone(),
                    method_name,
                    is_static: method.is_static,
                    param_count: param_tags.len() as u8,
                    return_type: return_tag,
                    param_types: param_tags,
                });
            }

            // Store JS module mapping for @:jsImport class methods
            if let Some(class_sym) = self.symbol_table.get_symbol(class.symbol_id) {
                if let Some((mod_is, _)) = class_sym.js_import {
                    if let Some(js_module) = self.string_interner.get(mod_is) {
                        for entry in &entries {
                            self.extern_js_module_map
                                .insert(entry.symbol_name.clone(), js_module.to_string());
                        }
                    }
                }
            }
        }

        // Also register entries under dot-separated and underscore-joined class names
        // so the MIR lowerer can find them regardless of class name format.
        let mut extra_entries: Vec<MethodDescEntry> = Vec::new();
        for entry in &entries {
            let parts: Vec<&str> = entry.class_name.split("::").collect();
            if parts.len() > 1 {
                let dot_class = parts.join(".");
                extra_entries.push(MethodDescEntry {
                    class_name: dot_class,
                    ..entry.clone()
                });
                let underscore_class = parts.join("_");
                extra_entries.push(MethodDescEntry {
                    class_name: underscore_class,
                    ..entry.clone()
                });
            }
        }
        entries.extend(extra_entries);

        if !entries.is_empty() {
            let plugin = NativePlugin::from_method_entries("extern_import", entries);
            self.compiler_plugin_registry.register(Box::new(plugin));
        }
    }

    /// Map a Haxe TypeId to an IrTypeDescriptor u8 value for MethodDescEntry.
    fn haxe_type_to_descriptor(&self, type_id: TypeId) -> u8 {
        use crate::tast::TypeKind;
        let tt = self.type_table.borrow();
        match tt.get(type_id).map(|t| &t.kind) {
            Some(TypeKind::Int) => 3,    // I32
            Some(TypeKind::Float) => 7,  // F64
            Some(TypeKind::Bool) => 1,   // Bool
            Some(TypeKind::String) => 8, // String
            Some(TypeKind::Void) => 0,   // Void
            _ => 9,                      // PtrVoid for class types, etc.
        }
    }

    /// Map a Haxe return/param TypeId to a `native_type` tag as decoded by
    /// `compiler_plugin::native_type_to_descriptor` (0=Void 1=I64 2=F64 3=PtrVoid
    /// 4=Bool). This is a DIFFERENT numbering than `haxe_type_to_descriptor`.
    /// Used to give `@:native` extern methods their real declared return type
    /// instead of the old blanket PtrVoid default — a PtrVoid return decays a
    /// scalar (e.g. `:Int`) to a boxed pointer (null) across the module boundary.
    fn haxe_type_to_native_tag(&self, type_id: TypeId) -> u8 {
        use crate::tast::TypeKind;
        let tt = self.type_table.borrow();
        match tt.get(type_id).map(|t| &t.kind) {
            Some(TypeKind::Void) => 0,
            Some(TypeKind::Int) => 1,   // I64
            Some(TypeKind::Float) => 2, // F64
            Some(TypeKind::Bool) => 4,
            _ => 3, // PtrVoid — objects, String, Tensor, Usize, etc.
        }
    }

    /// Register an external compiler plugin.
    ///
    /// This allows native packages (loaded via dlopen) to provide method mappings
    /// and extern declarations without modifying compiler source code. Must be
    /// called before `lower_to_tast()`.
    pub fn register_compiler_plugin(
        &mut self,
        plugin: Box<dyn crate::compiler_plugin::CompilerPlugin + 'static>,
    ) {
        self.compiler_plugin_registry.register(plugin);
    }

    /// Add external runtime symbols for JIT linking.
    ///
    /// These are merged with HDLL symbols and made available to the JIT backend.
    pub fn add_external_symbols(&mut self, symbols: Vec<(String, *const u8)>) {
        self.hdll_symbols.extend(symbols);
    }

    /// Add an additional source path for import resolution (e.g. from an rpkg package).
    pub fn add_source_path(&mut self, path: PathBuf) {
        self.namespace_resolver.add_source_path(path);
    }

    /// Scan parsed user files for `@:hlNative` metadata and load corresponding HDLL libraries.
    ///
    /// This should be called after user files have been added (so `user_files` is populated)
    /// but before MIR lowering (so the plugin registry has all HDLL mappings available).
    ///
    /// For each class with `@:hlNative("libname")`, this:
    /// 1. Extracts method names and static flags from the class declaration
    /// 2. Searches `hdll_search_paths` for `libname.hdll`
    /// 3. Loads the HDLL via `hlp_` symbol introspection
    /// 4. Registers the plugin and collects function pointers for JIT linking
    pub fn discover_and_load_hdlls(&mut self) {
        // Collect hlNative class info from user files before mutating self
        let mut hl_native_classes: Vec<(String, String, Vec<(String, bool)>)> = Vec::new();

        for file in &self.user_files {
            for decl in &file.declarations {
                if let parser::TypeDeclaration::Class(class_decl) = decl {
                    if let Some(lib_name) = Self::extract_hl_native_meta(&class_decl.meta) {
                        let methods: Vec<(String, bool)> = class_decl
                            .fields
                            .iter()
                            .filter_map(|field| {
                                if let parser::ClassFieldKind::Function(func) = &field.kind {
                                    let is_static =
                                        field.modifiers.contains(&parser::Modifier::Static);
                                    Some((func.name.clone(), is_static))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !methods.is_empty() {
                            info!(
                                "Found @:hlNative(\"{}\") on class '{}' with {} methods",
                                lib_name,
                                class_decl.name,
                                methods.len()
                            );
                            hl_native_classes.push((lib_name, class_decl.name.clone(), methods));
                        }
                    }
                }
            }
        }

        // Now load each HDLL
        for (lib_name, class_name, methods) in hl_native_classes {
            if self.loaded_hdlls.contains(&lib_name) {
                debug!("HDLL '{}' already loaded, skipping", lib_name);
                continue;
            }

            let method_refs: Vec<(&str, bool)> = methods
                .iter()
                .map(|(name, is_static)| (name.as_str(), *is_static))
                .collect();

            if let Some(hdll_path) = self.find_hdll(&lib_name) {
                match HdllPlugin::load_with_introspection(
                    &hdll_path,
                    &lib_name,
                    &class_name,
                    &method_refs,
                ) {
                    Ok(plugin) => {
                        for (name, ptr) in plugin.get_symbols() {
                            self.hdll_symbols.push((name.to_string(), ptr));
                        }
                        self.compiler_plugin_registry.register(Box::new(plugin));
                        self.loaded_hdlls.insert(lib_name);
                    }
                    Err(e) => {
                        warn!("Failed to load {}.hdll: {}", lib_name, e);
                    }
                }
            } else {
                warn!(
                    "HDLL '{}' not found in search paths: {:?}",
                    lib_name, self.config.hdll_search_paths
                );
            }
        }
    }

    /// Extract `@:hlNative("libname")` metadata from a class's metadata list.
    ///
    /// Returns `Some(lib_name)` if `@:hlNative` is found, `None` otherwise.
    fn extract_hl_native_meta(meta: &[parser::Metadata]) -> Option<String> {
        for m in meta {
            let name = m.name.strip_prefix(':').unwrap_or(&m.name);
            if name == "hlNative" {
                // Extract library name from first parameter
                if let Some(first_param) = m.params.first() {
                    if let parser::ExprKind::String(lib_name) = &first_param.kind {
                        return Some(lib_name.clone());
                    }
                }
                // @:hlNative with no parameters - use class name as fallback
                return None;
            }
        }
        None
    }

    /// Search for an HDLL file in the configured search paths.
    ///
    /// On macOS, HDLLs are `.dylib` files. On Linux, `.so`. On Windows, `.dll`.
    /// The Hashlink convention uses `.hdll` extension.
    fn find_hdll(&self, lib_name: &str) -> Option<PathBuf> {
        // Try platform-specific names and .hdll extension
        let candidates = if cfg!(target_os = "macos") {
            vec![
                format!("{}.hdll", lib_name),
                format!("lib{}.dylib", lib_name),
                format!("{}.dylib", lib_name),
            ]
        } else if cfg!(target_os = "windows") {
            vec![format!("{}.hdll", lib_name), format!("{}.dll", lib_name)]
        } else {
            vec![
                format!("{}.hdll", lib_name),
                format!("lib{}.so", lib_name),
                format!("{}.so", lib_name),
            ]
        };

        for dir in &self.config.hdll_search_paths {
            for candidate in &candidates {
                let path = dir.join(candidate);
                if path.exists() {
                    return Some(path);
                }
            }
        }

        None
    }

    /// Get the stdlib typed files that were loaded during compilation
    /// Returns a reference to the vector of TypedFiles from stdlib loading
    pub fn get_stdlib_typed_files(&self) -> &[TypedFile] {
        &self.loaded_stdlib_typed_files
    }
}

/// Cache statistics
#[derive(Debug, Default)]
pub struct CacheStats {
    pub cached_modules: usize,
    pub total_size_bytes: u64,
}

impl CacheStats {
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Collect qualified type references from a parsed AST.
/// Walks all type declarations and their type references, collecting any
/// TypePath with a non-empty package as an implicit import.
/// For example, `new haxe.ds.BalancedTree<Int, String>()` yields "haxe.ds.BalancedTree".
fn collect_qualified_type_refs_from_ast(ast: &parser::HaxeFile, out: &mut Vec<String>) {
    use parser::haxe_ast::{BlockElement, ClassFieldKind, Expr, ExprKind, Type, TypeDeclaration};
    use std::collections::BTreeSet;

    let mut seen = BTreeSet::new();

    fn collect_from_type(ty: &Type, seen: &mut BTreeSet<String>, out: &mut Vec<String>) {
        match ty {
            Type::Path { path, params, .. } => {
                if !path.package.is_empty() {
                    let qualified = format!("{}.{}", path.package.join("."), path.name);
                    if seen.insert(qualified.clone()) {
                        out.push(qualified);
                    }
                }
                for p in params {
                    collect_from_type(p, seen, out);
                }
            }
            Type::Function { params, ret, .. } => {
                for p in params {
                    collect_from_type(p, seen, out);
                }
                collect_from_type(ret, seen, out);
            }
            Type::Optional { inner, .. } | Type::Parenthesis { inner, .. } => {
                collect_from_type(inner, seen, out);
            }
            Type::Intersection { left, right, .. } => {
                collect_from_type(left, seen, out);
                collect_from_type(right, seen, out);
            }
            Type::Anonymous { fields, .. } => {
                for f in fields {
                    collect_from_type(&f.type_hint, seen, out);
                }
            }
            Type::Wildcard { .. } => {}
        }
    }

    fn collect_from_expr(expr: &Expr, seen: &mut BTreeSet<String>, out: &mut Vec<String>) {
        match &expr.kind {
            ExprKind::New {
                type_path,
                params,
                args,
            } => {
                if !type_path.package.is_empty() {
                    let qualified = format!("{}.{}", type_path.package.join("."), type_path.name);
                    if seen.insert(qualified.clone()) {
                        out.push(qualified);
                    }
                }
                for p in params {
                    collect_from_type(p, seen, out);
                }
                for a in args {
                    collect_from_expr(a, seen, out);
                }
            }
            ExprKind::Block(elements) => {
                for elem in elements {
                    if let BlockElement::Expr(e) = elem {
                        collect_from_expr(e, seen, out);
                    }
                }
            }
            ExprKind::Var {
                type_hint, expr, ..
            }
            | ExprKind::Final {
                type_hint, expr, ..
            } => {
                if let Some(ty) = type_hint {
                    collect_from_type(ty, seen, out);
                }
                if let Some(init) = expr {
                    collect_from_expr(init, seen, out);
                }
            }
            ExprKind::Call { expr: callee, args } => {
                collect_from_expr(callee, seen, out);
                for a in args {
                    collect_from_expr(a, seen, out);
                }
            }
            ExprKind::Field {
                expr: obj, field, ..
            } => {
                // Try to extract qualified type reference from field chains like
                // `rayzor.concurrent.Thread.spawn()` → import "rayzor.concurrent.Thread"
                // Walk the chain collecting segments; if the base is a lowercase ident
                // (package root), reconstruct the qualified path up to the last
                // capitalized segment (class name).
                let mut segments = vec![field.as_str()];
                let mut cur = obj.as_ref();
                loop {
                    match &cur.kind {
                        ExprKind::Field {
                            expr: inner,
                            field: seg,
                            ..
                        } => {
                            segments.push(seg.as_str());
                            cur = inner.as_ref();
                        }
                        ExprKind::Ident(name) => {
                            segments.push(name.as_str());
                            break;
                        }
                        _ => break,
                    }
                }
                segments.reverse();
                // Need at least 3 segments: package.Class.method
                // Find the last capitalized segment (class name)
                if segments.len() >= 3 {
                    if let Some(class_idx) = segments
                        .iter()
                        .rposition(|s| s.chars().next().map_or(false, |c| c.is_uppercase()))
                    {
                        // Everything before and including class_idx is the qualified type
                        if class_idx >= 1 {
                            let qualified = segments[..=class_idx].join(".");
                            if seen.insert(qualified.clone()) {
                                out.push(qualified);
                            }
                        }
                    }
                }
                collect_from_expr(obj, seen, out);
            }
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                collect_from_expr(cond, seen, out);
                collect_from_expr(then_branch, seen, out);
                if let Some(e) = else_branch {
                    collect_from_expr(e, seen, out);
                }
            }
            ExprKind::Return(Some(e)) => collect_from_expr(e, seen, out),
            ExprKind::Binary { left, right, .. } => {
                collect_from_expr(left, seen, out);
                collect_from_expr(right, seen, out);
            }
            ExprKind::Unary { expr: e, .. } => {
                collect_from_expr(e, seen, out);
            }
            ExprKind::Assign { left, right, .. } => {
                collect_from_expr(left, seen, out);
                collect_from_expr(right, seen, out);
            }
            ExprKind::While { cond, body, .. } | ExprKind::DoWhile { body, cond } => {
                collect_from_expr(cond, seen, out);
                collect_from_expr(body, seen, out);
            }
            ExprKind::For { iter, body, .. } => {
                collect_from_expr(iter, seen, out);
                collect_from_expr(body, seen, out);
            }
            ExprKind::Switch {
                expr: subject,
                cases,
                default,
            } => {
                collect_from_expr(subject, seen, out);
                for case in cases {
                    if let Some(guard) = &case.guard {
                        collect_from_expr(guard, seen, out);
                    }
                    collect_from_expr(&case.body, seen, out);
                }
                if let Some(d) = default {
                    collect_from_expr(d, seen, out);
                }
            }
            ExprKind::Try {
                expr: body,
                catches,
                ..
            } => {
                collect_from_expr(body, seen, out);
                for c in catches {
                    collect_from_expr(&c.body, seen, out);
                }
            }
            ExprKind::Cast { expr: e, type_hint } => {
                collect_from_expr(e, seen, out);
                if let Some(ty) = type_hint {
                    collect_from_type(ty, seen, out);
                }
            }
            ExprKind::Array(items) => {
                for item in items {
                    collect_from_expr(item, seen, out);
                }
            }
            ExprKind::Paren(e)
            | ExprKind::Throw(e)
            | ExprKind::Untyped(e)
            | ExprKind::Meta { expr: e, .. } => {
                collect_from_expr(e, seen, out);
            }
            ExprKind::Ternary {
                cond,
                then_expr,
                else_expr,
            } => {
                collect_from_expr(cond, seen, out);
                collect_from_expr(then_expr, seen, out);
                collect_from_expr(else_expr, seen, out);
            }
            ExprKind::Index { expr: e, index } => {
                collect_from_expr(e, seen, out);
                collect_from_expr(index, seen, out);
            }
            ExprKind::TypeCheck { expr: e, type_hint } => {
                collect_from_expr(e, seen, out);
                collect_from_type(type_hint, seen, out);
            }
            _ => {}
        }
    }

    // Walk class field helpers
    fn collect_from_class_field(
        field: &parser::haxe_ast::ClassField,
        seen: &mut BTreeSet<String>,
        out: &mut Vec<String>,
    ) {
        match &field.kind {
            ClassFieldKind::Var {
                type_hint, expr, ..
            }
            | ClassFieldKind::Final {
                type_hint, expr, ..
            } => {
                if let Some(ty) = type_hint {
                    collect_from_type(ty, seen, out);
                }
                if let Some(e) = expr {
                    collect_from_expr(e, seen, out);
                }
            }
            ClassFieldKind::Property { type_hint, .. } => {
                if let Some(ty) = type_hint {
                    collect_from_type(ty, seen, out);
                }
            }
            ClassFieldKind::Function(func) => {
                if let Some(ret) = &func.return_type {
                    collect_from_type(ret, seen, out);
                }
                for param in &func.params {
                    if let Some(ty) = &param.type_hint {
                        collect_from_type(ty, seen, out);
                    }
                    if let Some(def) = &param.default_value {
                        collect_from_expr(def, seen, out);
                    }
                }
                if let Some(body) = &func.body {
                    collect_from_expr(body, seen, out);
                }
            }
        }
    }

    // Walk all type declarations
    for decl in &ast.declarations {
        match decl {
            TypeDeclaration::Class(class) => {
                if let Some(extends) = &class.extends {
                    collect_from_type(extends, &mut seen, out);
                }
                for iface in &class.implements {
                    collect_from_type(iface, &mut seen, out);
                }
                for field in &class.fields {
                    collect_from_class_field(field, &mut seen, out);
                }
            }
            TypeDeclaration::Enum(en) => {
                for variant in &en.constructors {
                    for param in &variant.params {
                        if let Some(ty) = &param.type_hint {
                            collect_from_type(ty, &mut seen, out);
                        }
                    }
                }
            }
            TypeDeclaration::Typedef(td) => {
                collect_from_type(&td.type_def, &mut seen, out);
            }
            TypeDeclaration::Abstract(ab) => {
                if let Some(ty) = &ab.underlying {
                    collect_from_type(ty, &mut seen, out);
                }
                for field in &ab.fields {
                    collect_from_class_field(field, &mut seen, out);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_unit_with_stdlib() {
        let mut unit = CompilationUnit::new(CompilationConfig::default());

        // Load stdlib
        unit.load_stdlib().expect("Failed to load stdlib");

        // Verify stdlib symbols were loaded. With a BLADE symbol manifest,
        // top-level stdlib types do not require parsing source files into
        // stdlib_files.
        assert!(unit.symbol_table.len() > 0, "No stdlib symbols loaded");
        assert_eq!(unit.user_files.len(), 0, "Should have no user files");
    }

    #[test]
    fn test_compilation_unit_add_user_file() {
        let mut unit = CompilationUnit::new(CompilationConfig::default());

        let source = r#"
            package test;
            class MyClass {
                public function new() {}
            }
        "#;

        unit.add_file(source, "MyClass.hx")
            .expect("Failed to add file");

        assert_eq!(unit.user_files.len(), 1);
        assert_eq!(unit.stdlib_files.len(), 0);
    }

    #[test]
    fn test_compilation_unit_full_pipeline() {
        let mut unit = CompilationUnit::new(CompilationConfig::default());

        // Load stdlib first
        unit.load_stdlib().expect("Failed to load stdlib");

        // Add user file
        let source = r#"
            package test;
            class MyClass {
                public function new() {}

                public function useArray():Void {
                    var arr = [1, 2, 3];
                    arr.push(4);
                }
            }
        "#;

        unit.add_file(source, "MyClass.hx")
            .expect("Failed to add file");

        // Lower to TAST - this should succeed now with proper stdlib propagation
        let typed_files = unit.lower_to_tast().expect("Failed to lower to TAST");

        assert!(typed_files.len() > 0, "Should have typed files");
    }
}
