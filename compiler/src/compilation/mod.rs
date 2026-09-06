//! Multi-file Compilation Infrastructure
//!
//! This module provides the proper architecture for compiling multiple source files
//! together, including standard library loading, package management, and symbol resolution.

include!(concat!(env!("OUT_DIR"), "/stdlib_defines.rs"));

use crate::compiler_plugin::CompilerPluginRegistry;
use crate::dependency_graph::{CircularDependency, DependencyAnalysis, DependencyGraph};
use crate::ir::{
    blade::{
        load_blade, load_symbol_manifest, load_symbol_manifest_from_bytes, save_blade_with_state,
        BladeAbstractInfo, BladeAccessor, BladeCachedMaps, BladeClassInfo, BladeEnumInfo,
        BladeFieldEntry, BladeFuncEntry, BladeMetadata, BladeMethodInfo, BladePropertyEntry,
        BladeSymbolManifest, BladeTypeAliasInfo, BladeTypeInfo,
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
    AstLowering, InternedString, ScopeId, ScopeTree, SourceLocation, StringInterner, SymbolId,
    SymbolTable, TypeId, TypeKind, TypeTable, TypedFile,
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
    /// Reading and parsing every import purely to discover its dependencies,
    /// before any of them is compiled.
    pub import_discover_ms: f64,
    pub import_toposort_ms: f64,
    pub import_compile_ms: f64,
    /// Looking an import up in the BLADE cache, and writing it back after a miss.
    pub import_cache_load_ms: f64,
    pub import_cache_save_ms: f64,
    /// The import's own compile, separated from the bookkeeping around it.
    pub import_compile_call_ms: f64,
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

/// Root-scope name a manifest-restored type is published under.
///
/// Only a package-less type is ambiently visible by its bare name. `haxe.zip.Entry`
/// is `Entry` only inside a module that imports it, so a packaged type takes its
/// qualified name in the root scope and leaves the bare name free. Publishing the
/// bare name instead makes a user's own `class Entry` collide with the stdlib
/// symbol: pre-registration sees the name taken, skips the user declaration, and
/// the file's type silently resolves to the stdlib one.
#[inline]
fn manifest_root_name(
    package: &[String],
    short_name: InternedString,
    qualified_name: InternedString,
) -> InternedString {
    if package.is_empty() {
        short_name
    } else {
        qualified_name
    }
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

    /// Short name → symbol for manifest types published under their qualified
    /// name. A manifest records a field or parameter type as it was written in
    /// source, so `haxe.io.Bytes` may appear as bare `Bytes`; resolving those
    /// signatures needs a short-name route, and keeping it here instead of in
    /// the root scope leaves the bare name free for a user declaration.
    manifest_types_by_short_name: BTreeMap<InternedString, SymbolId>,

    /// Type parameters of the manifest type whose members are being registered.
    /// A signature naming `T` means that type's parameter, not a global type,
    /// and member registration goes through `parse_type_string`, which has no
    /// other way to see the enclosing declaration.
    manifest_type_params: BTreeMap<String, TypeId>,

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

mod artifact_cache;
mod config;
mod crossmodule;
mod driver;
mod errors;
mod externs;
mod imports;
mod manifest;
mod ownership;
mod sources;

/// The name of a type parameter a manifest type refers to, if that is all it
/// is — a bare name with no package and no arguments.
fn blade_type_param_name(ty: Option<&bsym::BladeType>) -> Option<&str> {
    match ty? {
        bsym::BladeType::Path {
            package,
            name,
            sub: None,
            params,
        } if package.is_empty() && params.is_empty() => Some(name),
        _ => None,
    }
}

/// What declaring a class from a manifest publishes: enough for any sibling's
/// signature to resolve it, and enough to register its members afterwards
/// without deriving any of it again.
pub(crate) struct DeclaredClass {
    symbol_id: SymbolId,
    class_scope: ScopeId,
    type_params: BTreeMap<String, TypeId>,
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
            manifest_types_by_short_name: BTreeMap::new(),
            manifest_type_params: BTreeMap::new(),
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
