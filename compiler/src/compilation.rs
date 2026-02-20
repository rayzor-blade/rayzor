//! Multi-file Compilation Infrastructure
//!
//! This module provides the proper architecture for compiling multiple source files
//! together, including standard library loading, package management, and symbol resolution.

use crate::compiler_plugin::CompilerPluginRegistry;
use crate::dependency_graph::{CircularDependency, DependencyAnalysis, DependencyGraph};
use crate::ir::{
    blade::{
        load_blade, load_symbol_manifest, save_blade, BladeAbstractInfo, BladeClassInfo,
        BladeEnumInfo, BladeMetadata, BladeMethodInfo, BladeSymbolManifest, BladeTypeAliasInfo,
    },
    IrInstruction, IrModule, Monomorphizer,
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
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Represents a complete compilation unit with multiple source files
pub struct CompilationUnit {
    /// Stdlib files (loaded first with haxe.* package)
    pub stdlib_files: Vec<HaxeFile>,

    /// Global import.hx files (loaded after stdlib, before user files)
    pub import_hx_files: Vec<HaxeFile>,

    /// User source files
    pub user_files: Vec<HaxeFile>,

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
    pub failed_type_loads: HashSet<String>,

    /// Cache of files that have been successfully compiled (to avoid redundant recompilation)
    /// Maps filename to the TypedFile result
    compiled_files: HashMap<String, TypedFile>,

    /// Internal compilation pipeline (delegates to HaxeCompilationPipeline)
    pipeline: HaxeCompilationPipeline,

    /// MIR modules generated during compilation (collected from pipeline results)
    mir_modules: Vec<std::sync::Arc<crate::ir::IrModule>>,

    /// MIR modules from on-demand imported stdlib files (e.g., BalancedTree.hx).
    /// These are merged into the user module during stdlib renumbering rather than
    /// being stored as separate modules, because their function IDs would collide.
    import_mir_modules: Vec<crate::ir::IrModule>,

    /// Stdlib typed files loaded on-demand (typedefs, etc. that need to be in HIR)
    loaded_stdlib_typed_files: Vec<TypedFile>,

    /// Mapping from HIR function symbols to MIR function IDs for stdlib functions
    /// This allows user code to call pure Haxe stdlib functions (like StringTools)
    stdlib_function_map: BTreeMap<crate::tast::SymbolId, crate::ir::IrFunctionId>,

    /// Name-based mapping from qualified function names to MIR function IDs
    /// This is used for cross-file lookups where SymbolIds differ between compilation units
    /// e.g., "StringTools.startsWith" -> IrFunctionId(N)
    stdlib_function_name_map: BTreeMap<String, crate::ir::IrFunctionId>,

    /// Compiler plugin registry (builtin + HDLL plugins)
    compiler_plugin_registry: CompilerPluginRegistry,

    /// Function pointers collected from loaded HDLL plugins for JIT linking
    hdll_symbols: Vec<(String, *const u8)>,

    /// Set of already-loaded HDLL library names to avoid duplicate loading
    loaded_hdlls: HashSet<String>,
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
                // Concurrent types
                "rayzor/concurrent/Thread.hx".to_string(),
                "rayzor/concurrent/Channel.hx".to_string(),
                "rayzor/concurrent/Mutex.hx".to_string(),
                "rayzor/concurrent/Arc.hx".to_string(),
            ],
            load_stdlib: true,
            stdlib_root_package: Some("haxe".to_string()), // Prefix stdlib with "haxe.*" namespace
            global_import_hx_files: Vec::new(),            // No global import.hx by default
            enable_cache: true, // Cache enabled - BLADE manifest now includes Math, Std, Date, etc.
            cache_dir: None,    // Auto-discover cache directory when needed
            lazy_stdlib: false, // Default to eager loading for compatibility
            pipeline_config: PipelineConfig::default(),
            hdll_search_paths: vec![PathBuf::from(".")],
        }
    }
}

impl CompilationConfig {
    /// Discover standard library paths from environment and standard locations
    ///
    /// Search order:
    /// 1. HAXE_STD_PATH environment variable
    /// 2. HAXE_HOME environment variable (looking for std/ subdirectory)
    /// 3. Current project's haxe-std directory
    /// 4. Parent directory's haxe-std
    /// 5. Standard installation locations (platform-specific)
    pub fn discover_stdlib_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // 1. Check HAXE_STD_PATH environment variable
        if let Ok(haxe_std_path) = std::env::var("HAXE_STD_PATH") {
            let path = PathBuf::from(&haxe_std_path);
            if path.exists() {
                info!("Found stdlib at HAXE_STD_PATH: {}", haxe_std_path);
                paths.push(path);
                return paths; // Use this path exclusively if set
            } else {
                warn!(
                    "HAXE_STD_PATH set but directory doesn't exist: {}",
                    haxe_std_path
                );
            }
        }

        // 2. Check HAXE_HOME/std
        if let Ok(haxe_home) = std::env::var("HAXE_HOME") {
            let std_path = PathBuf::from(&haxe_home).join("std");
            if std_path.exists() {
                info!("Found stdlib at HAXE_HOME/std: {:?}", std_path);
                paths.push(std_path);
            }
        }

        // 3. Check current project's haxe-std directory
        let project_stdlib = PathBuf::from("compiler/haxe-std");
        if project_stdlib.exists() {
            paths.push(project_stdlib);
        }

        // 4. Check parent directory's haxe-std
        let parent_stdlib = PathBuf::from("../haxe-std");
        if parent_stdlib.exists() {
            paths.push(parent_stdlib);
        }

        let current_dir_stdlib = PathBuf::from("./haxe-std");
        if current_dir_stdlib.exists() {
            paths.push(current_dir_stdlib);
        }

        // 5. Platform-specific standard installation locations
        #[cfg(target_os = "linux")]
        {
            let linux_paths = vec![
                PathBuf::from("/usr/share/haxe/std"),
                PathBuf::from("/usr/local/share/haxe/std"),
                PathBuf::from("/opt/haxe/std"),
            ];
            for path in linux_paths {
                if path.exists() {
                    paths.push(path);
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            let macos_paths = vec![
                PathBuf::from("/usr/local/lib/haxe/std"),
                PathBuf::from("/opt/homebrew/lib/haxe/std"),
                PathBuf::from("/Library/Haxe/std"),
            ];
            for path in macos_paths {
                if path.exists() {
                    paths.push(path);
                }
            }

            // Check user's home directory
            if let Ok(home) = std::env::var("HOME") {
                let user_haxe = PathBuf::from(home).join(".haxe/std");
                if user_haxe.exists() {
                    paths.push(user_haxe);
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            let windows_paths = vec![
                PathBuf::from("C:\\HaxeToolkit\\haxe\\std"),
                PathBuf::from("C:\\Program Files\\Haxe\\std"),
                PathBuf::from("C:\\Program Files (x86)\\Haxe\\std"),
            ];
            for path in windows_paths {
                if path.exists() {
                    paths.push(path);
                }
            }

            // Check user's AppData
            if let Ok(appdata) = std::env::var("APPDATA") {
                let user_haxe = PathBuf::from(appdata).join("Haxe\\std");
                if user_haxe.exists() {
                    paths.push(user_haxe);
                }
            }
        }

        if paths.is_empty() {
            warn!("No standard library found. Set HAXE_STD_PATH environment variable.");
            warn!("         or install Haxe to a standard location.");
            // Still provide fallback paths for development
            paths.push(PathBuf::from("compiler/haxe-std"));
            paths.push(PathBuf::from("../haxe-std"));
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

    /// Get or create the cache directory
    pub fn get_cache_dir(&self) -> PathBuf {
        if let Some(ref cache_dir) = self.cache_dir {
            return cache_dir.clone();
        }

        // Default: .rayzor/blade/cache (separate from Rust target folder)
        let default_cache = PathBuf::from(".rayzor/blade/cache");

        // Try to create it if it doesn't exist
        if !default_cache.exists() {
            let _ = std::fs::create_dir_all(&default_cache);
        }

        default_cache
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
            string_interner,
            symbol_table: SymbolTable::new(),
            type_table: Rc::new(RefCell::new(TypeTable::new())),
            scope_tree: ScopeTree::new(ScopeId::from_raw(0)),
            namespace_resolver,
            import_resolver,
            config,
            failed_type_loads: HashSet::new(),
            compiled_files: HashMap::new(),
            pipeline,
            mir_modules: Vec::new(),
            import_mir_modules: Vec::new(),
            loaded_stdlib_typed_files: Vec::new(),
            stdlib_function_map: BTreeMap::new(),
            stdlib_function_name_map: BTreeMap::new(),
            compiler_plugin_registry: CompilerPluginRegistry::new(),
            hdll_symbols: Vec::new(),
            loaded_hdlls: HashSet::new(),
        }
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

        // Configure namespace resolver with stdlib paths for on-demand loading
        self.namespace_resolver
            .set_stdlib_paths(self.config.stdlib_paths.clone());

        // Load pre-compiled symbols from BLADE manifest if caching is enabled
        // Skip if lazy_stdlib is enabled (for faster cold start)
        if self.config.enable_cache && !self.config.lazy_stdlib {
            if self.load_stdlib_symbols() {
                debug!("BLADE symbols loaded, stdlib configured for cached resolution");
            } else {
                debug!("No BLADE symbols available, falling back to on-demand loading");
            }
        } else if self.config.lazy_stdlib {
            debug!("Lazy stdlib enabled - skipping upfront symbol registration for faster startup");
            // Still register builtin globals like 'trace' which are always needed
            self.register_builtin_globals();
        }

        // Load default stdlib imports (Math, Std, Array, String, etc.)
        // These are core types that are always needed, even with lazy_stdlib
        let default_files = loader.load_default_imports();
        for file in default_files {
            debug!("Loading default import: {}", file.filename);
            // Add the file to the stdlib files for processing during lowering
            self.stdlib_files.push(file);
        }
        debug!("Loaded {} default stdlib imports", self.stdlib_files.len());

        Ok(())
    }

    /// Set source paths for user code (for on-demand import loading)
    /// These paths are checked first when resolving imports
    pub fn set_source_paths(&mut self, paths: Vec<PathBuf>) {
        self.namespace_resolver.set_source_paths(paths);
    }

    // === BLADE Caching Methods ===

    /// Get the BLADE cache path for a source file
    fn blade_cache_path(&self, source_path: &str) -> Option<PathBuf> {
        // Use get_cache_dir() to auto-discover the cache directory
        let cache_dir = self.config.get_cache_dir();

        // Convert source path to a cache-safe filename
        // e.g., "compiler/haxe-std/haxe/io/Bytes.hx" -> "haxe.io.Bytes.blade"
        let module_name = source_path
            .replace('/', ".")
            .replace('\\', ".")
            .replace(".hx", "")
            .split('.')
            .skip_while(|s| *s == "compiler" || *s == "haxe-std" || s.is_empty())
            .collect::<Vec<_>>()
            .join(".");

        if module_name.is_empty() {
            return None;
        }

        Some(cache_dir.join(format!("{}.blade", module_name)))
    }

    /// Compute hash of source content for cache validation
    fn hash_source(source: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
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
            Ok((mir, metadata)) => {
                // Validate cache by checking source hash
                let current_hash = Self::hash_source(source);
                if metadata.source_hash == current_hash {
                    debug!(
                        "[BLADE] Cache hit: {} -> {}",
                        source_path,
                        blade_path.display()
                    );
                    Some(mir)
                } else {
                    trace!("[BLADE] Cache stale (hash mismatch): {}", source_path);
                    None
                }
            }
            Err(e) => {
                trace!("[BLADE] Cache read error for {}: {}", source_path, e);
                None
            }
        }
    }

    /// Save a MIR module to BLADE cache
    fn save_blade_cached(
        &self,
        source_path: &str,
        source: &str,
        mir: &IrModule,
        dependencies: Vec<String>,
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
            source_hash: Self::hash_source(source),
            source_timestamp: now, // We use hash for validation, not timestamp
            compile_timestamp: now,
            dependencies,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        };

        match save_blade(&blade_path, mir, metadata) {
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

    // === BLADE Symbol Loading Methods ===

    /// Load pre-compiled stdlib symbols from .bsym manifest
    /// Returns true if symbols were loaded successfully
    pub fn load_stdlib_symbols(&mut self) -> bool {
        let manifest_path = PathBuf::from(".rayzor/blade/stdlib/stdlib.bsym");
        if !manifest_path.exists() {
            debug!(
                "[BLADE] No symbol manifest found at {}",
                manifest_path.display()
            );
            return false;
        }

        match load_symbol_manifest(&manifest_path) {
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

        // Register fields
        for field in &class_info.fields {
            self.register_field_from_blade(field, symbol_id, class_scope);
        }

        // Register static fields
        for field in &class_info.static_fields {
            self.register_field_from_blade(field, symbol_id, class_scope);
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
        _class_symbol: SymbolId,
        class_scope: ScopeId,
        is_static: bool,
    ) -> SymbolId {
        let method_name = self.string_interner.intern(&method.name);

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

        // Update symbol with type and flags
        if let Some(sym) = self.symbol_table.get_symbol_mut(method_symbol) {
            sym.type_id = func_type;
            if is_static {
                sym.flags = sym.flags.union(SymbolFlags::STATIC);
            }
            if !method.is_public {
                sym.visibility = crate::tast::symbols::Visibility::Private;
            }
        }

        // Add to scope
        let _ = self
            .scope_tree
            .add_symbol_to_scope(class_scope, method_symbol);

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

        // Add to scope
        let _ = self
            .scope_tree
            .add_symbol_to_scope(class_scope, field_symbol);

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

        // Create enum type
        let enum_type = self
            .type_table
            .borrow_mut()
            .create_enum_type(symbol_id, vec![]);

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
        // Try short name lookup in global scope
        let interned = self.string_interner.intern(name);
        if let Some(symbol) = self.symbol_table.lookup_symbol(ScopeId::first(), interned) {
            return Some(symbol.id);
        }

        None
    }

    /// Extract all class references from a Haxe AST file.
    /// This includes explicit imports, using statements, new expressions, and type annotations.
    fn extract_all_dependencies(ast: &parser::HaxeFile) -> Vec<String> {
        use parser::{BlockElement, ClassFieldKind, ExprKind, Type, TypeDeclaration};

        let mut deps = std::collections::HashSet::new();

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
        fn extract_type_deps(ty: &Type, deps: &mut std::collections::HashSet<String>) {
            match ty {
                Type::Path { path, params, .. } => {
                    // Only add if it looks like a class name (starts with uppercase)
                    if path.package.is_empty() && !path.name.is_empty() {
                        let first_char = path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false) {
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
            deps: &mut std::collections::HashSet<String>,
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
        fn extract_expr_deps(expr: &parser::Expr, deps: &mut std::collections::HashSet<String>) {
            match &expr.kind {
                ExprKind::New {
                    type_path,
                    params,
                    args,
                } => {
                    // Extract class name from new expression
                    if type_path.package.is_empty() && !type_path.name.is_empty() {
                        let first_char = type_path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false) {
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
                ExprKind::Field { expr, .. } => {
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
            deps: &mut std::collections::HashSet<String>,
        ) {
            match pattern {
                parser::Pattern::Const(e) => extract_expr_deps(e, deps),
                parser::Pattern::Constructor { path, params } => {
                    // Constructor patterns reference enum/class types
                    if path.package.is_empty() && !path.name.is_empty() {
                        let first_char = path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false) {
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
            deps: &mut std::collections::HashSet<String>,
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

        deps.into_iter().collect()
    }

    /// Load imports efficiently by pre-collecting all dependencies and compiling in topological order.
    /// This avoids the fail-retry pattern that causes exponential recompilation.
    pub fn load_imports_efficiently(&mut self, imports: &[String]) -> Result<(), String> {
        use std::collections::{HashMap, HashSet, VecDeque};

        // Step 1: Collect all files and their dependencies by parsing (not compiling)
        let mut all_files: HashMap<String, (PathBuf, String, Vec<String>)> = HashMap::new(); // path -> (filepath, source, deps)
        let mut to_process: VecDeque<String> = imports.iter().cloned().collect();
        let mut visited: HashSet<String> = HashSet::new();

        while let Some(qualified_path) = to_process.pop_front() {
            if visited.contains(&qualified_path) {
                continue;
            }
            visited.insert(qualified_path.clone());

            // Resolve to file path (use _force variant to bypass BLADE cache's
            // is_file_loaded check — BLADE pre-registers symbols but doesn't preserve
            // full TAST state needed for generic instantiation and method resolution)
            let file_path = if let Some(path) = self
                .namespace_resolver
                .resolve_qualified_path_to_file_force(&qualified_path)
            {
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

        // Step 2: Topological sort using Kahn's algorithm
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for (name, (_, _, deps)) in &all_files {
            in_degree.entry(name.clone()).or_insert(0);
            for dep in deps {
                // Skip self-dependencies (class referencing itself in its own file)
                if dep == name {
                    continue;
                }
                if all_files.contains_key(dep) {
                    graph.entry(dep.clone()).or_default().push(name.clone());
                    *in_degree.entry(name.clone()).or_insert(0) += 1;
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

        debug!("[IMPORT_LOAD] compile_order: {:?}", compile_order);

        // Step 3: Compile in topological order (no retries needed!)
        // With BLADE caching: check cache first, compile only if needed
        for name in compile_order {
            if let Some((file_path, source, deps)) = all_files.remove(&name) {
                // Skip if already compiled
                let filename = file_path.to_string_lossy().to_string();
                debug!(
                    "[IMPORT_LOAD] Compiling: {} (already_compiled={})",
                    filename,
                    self.compiled_files.contains_key(&filename)
                );
                if self.compiled_files.contains_key(&filename) {
                    continue;
                }

                // Mark as loaded
                self.namespace_resolver.mark_file_loaded(file_path.clone());

                // NOTE: BLADE cache is intentionally NOT used here.
                // The BLADE cache stores compiled MIR but does NOT preserve type system
                // state (symbol scopes, class_methods, method signatures). When importing
                // files, we need lower_file() to run so that class symbols get their
                // scope_id set and methods are registered in the shared symbol_table.
                // Without this, cross-file method resolution fails (e.g., Mutex.lock()
                // returns Dynamic instead of MutexGuard<T>).

                // Cache miss or caching disabled - compile normally
                match self.compile_file_with_shared_state(&filename, &source) {
                    Ok(typed_file) => {
                        self.loaded_stdlib_typed_files.push(typed_file);

                        // Move the MIR from mir_modules to import_mir_modules.
                        // CRITICAL: Renumber import function IDs to a high base offset
                        // to prevent collisions with the user module (which also starts
                        // IDs from 0). Update stdlib_function_map/name_map so the user
                        // module references these high IDs instead of the original ones.
                        if let Some(mir_arc) = self.mir_modules.pop() {
                            // Save to BLADE cache before renumbering
                            if self.config.enable_cache {
                                self.save_blade_cached(&filename, &source, &mir_arc, deps);
                            }

                            use crate::ir::IrFunctionId;
                            let mut import_mir = (*mir_arc).clone();

                            // Use a high base to avoid collision with user module IDs
                            let import_base: u32 =
                                100_000 + (self.import_mir_modules.len() as u32 * 10_000);

                            // Build old→new ID mapping and renumber functions
                            let mut id_map: std::collections::HashMap<IrFunctionId, IrFunctionId> =
                                std::collections::HashMap::new();
                            for old_id in import_mir.functions.keys() {
                                id_map.insert(*old_id, IrFunctionId(old_id.0 + import_base));
                            }

                            // Renumber functions in the import module
                            let old_functions: std::collections::BTreeMap<_, _> =
                                std::mem::take(&mut import_mir.functions);
                            for (old_id, mut func) in old_functions {
                                let new_id = *id_map.get(&old_id).unwrap();
                                func.id = new_id;

                                // Update internal CallDirect/FunctionRef
                                use crate::ir::IrInstruction;
                                for block in func.cfg.blocks.values_mut() {
                                    for inst in &mut block.instructions {
                                        match inst {
                                            IrInstruction::CallDirect { func_id, .. }
                                            | IrInstruction::FunctionRef { func_id, .. }
                                            | IrInstruction::MakeClosure { func_id, .. } => {
                                                if let Some(new_func_id) = id_map.get(func_id) {
                                                    *func_id = *new_func_id;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }

                                import_mir.functions.insert(new_id, func);
                            }

                            // Update stdlib_function_map to point to renumbered IDs
                            for (_sym, func_id) in self.stdlib_function_map.iter_mut() {
                                if let Some(&new_id) = id_map.get(func_id) {
                                    *func_id = new_id;
                                }
                            }

                            // Update stdlib_function_name_map to point to renumbered IDs
                            for (_name, func_id) in self.stdlib_function_name_map.iter_mut() {
                                if let Some(&new_id) = id_map.get(func_id) {
                                    *func_id = new_id;
                                }
                            }

                            self.import_mir_modules.push(import_mir);
                        }
                    }
                    Err(_e) => {
                        // If it still fails, fall back to old retry mechanism
                        // This handles edge cases like Placeholder typedefs
                        debug!(
                            "[IMPORT_LOAD] Failed to compile {}: {} error(s)",
                            filename,
                            _e.len()
                        );
                        for e in &_e {
                            debug!("  - {}", e.message);
                        }
                    }
                }
            }
        }

        Ok(())
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

        let load_start = std::time::Instant::now();

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
        // Find the start of type_name: \" marker
        if let Some(type_name_start) = error_msg.find("type_name: \"") {
            // Move past 'type_name: "' to get to the actual name
            let after_marker = &error_msg[type_name_start + 12..]; // 12 = length of 'type_name: "'
                                                                   // Find the closing quote
            if let Some(end) = after_marker.find('"') {
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
        // Common generic parameter patterns
        matches!(
            type_name,
            "Key"
                | "Value"
                | "Item"
                | "Element"
                | "Iterator"
                | "KeyValueIterator"
                | "Iterable"
                | "KeyValueIterable"
        )
    }

    /// Pre-register type declarations from a file without full compilation
    /// This is the first pass that registers class/interface/enum names in the namespace
    /// so they can be referenced by other files during full compilation
    fn pre_register_file_types(&mut self, filename: &str, source: &str) -> Result<(), String> {
        use crate::tast::ast_lowering::AstLowering;
        use parser::parse_haxe_file_with_diagnostics;

        // Parse the file
        let parse_result = parse_haxe_file_with_diagnostics(filename, source)
            .map_err(|e| format!("Parse error in {}: {}", filename, e))?;

        let ast_file = parse_result.file;

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
        use parser::parse_haxe_file_with_diagnostics;

        let parse_result = match parse_haxe_file_with_diagnostics(filename, source) {
            Ok(r) => r,
            Err(_) => return,
        };

        let ast_file = parse_result.file;
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

    /// Load global import.hx files
    /// These are processed AFTER stdlib but BEFORE user files
    /// They provide global imports available to all user code
    pub fn load_global_imports(&mut self) -> Result<(), String> {
        use std::fs;

        for import_path in &self.config.global_import_hx_files.clone() {
            let source = fs::read_to_string(import_path)
                .map_err(|e| format!("Failed to read import.hx at {:?}: {}", import_path, e))?;

            let haxe_file =
                parse_haxe_file(import_path.to_str().unwrap_or("import.hx"), &source, true)
                    .map_err(|e| format!("Parse error in {:?}: {}", import_path, e))?;

            self.import_hx_files.push(haxe_file);
        }

        Ok(())
    }

    /// Add a user source file to the compilation unit
    pub fn add_file(&mut self, source: &str, file_path: &str) -> Result<(), String> {
        // Parse the file (file_name, input, recovery mode=true, debug=true to preserve source)
        let haxe_file = parse_haxe_file_with_debug(file_path, source, true, true)
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

        let entries = fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory {:?}: {}", dir_path, e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();

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
    ) -> Result<TypedFile, Vec<CompilationError>> {
        use crate::tast::ast_lowering::AstLowering;
        use parser::parse_haxe_file_with_diagnostics;

        // Skip if already successfully compiled - return cached TypedFile
        if let Some(cached) = self.compiled_files.get(filename) {
            return Ok(cached.clone());
        }

        // Parse the file
        let parse_result = parse_haxe_file_with_diagnostics(filename, source).map_err(|e| {
            vec![CompilationError {
                message: format!("Parse error: {}", e),
                location: SourceLocation::unknown(),
                category: ErrorCategory::ParseError,
                suggestion: None,
                related_errors: Vec::new(),
            }]
        })?;

        let ast_file = parse_result.file;
        let _source_map = parse_result.source_map;
        let file_id = diagnostics::FileId::new(0);

        // Stage 1.5: Macro expansion (if enabled)
        let ast_file = if self.config.pipeline_config.enable_macro_expansion {
            let expansion = crate::macro_system::expand_macros(ast_file);
            // Log macro diagnostics as warnings (non-fatal in multi-file context)
            for diag in &expansion.diagnostics {
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
            expansion.file
        } else {
            ast_file
        };

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

        // Skip stdlib loading during lowering if BLADE cache is enabled
        // (methods and types were already registered from the BLADE manifest)
        if self.config.enable_cache {
            lowering.set_skip_stdlib_loading(true);
        }

        lowering.initialize_span_converter_with_filename(
            file_id.as_usize() as u32,
            source.to_string(),
            filename.to_string(),
        );

        let typed_file = lowering.lower_file(&ast_file).map_err(|e| {
            vec![CompilationError {
                message: format!("Lowering error: {:?}", e),
                location: SourceLocation::unknown(),
                category: ErrorCategory::TypeError,
                suggestion: None,
                related_errors: Vec::new(),
            }]
        })?;

        // Lower to HIR
        use crate::ir::tast_to_hir::lower_tast_to_hir;
        let hir_module = lower_tast_to_hir(
            &typed_file,
            &self.symbol_table,
            &self.type_table,
            &mut self.string_interner,
            None, // No semantic graphs for now
        )
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|e| CompilationError {
                    message: format!("HIR lowering error: {:?}", e),
                    location: SourceLocation::unknown(),
                    category: ErrorCategory::InternalError,
                    suggestion: None,
                    related_errors: Vec::new(),
                })
                .collect::<Vec<_>>()
        })?;

        // Check if this file contains ONLY extern class declarations BEFORE MIR lowering.
        // Extern class files only need TAST+HIR for type system registration (symbol scopes,
        // method signatures). Their runtime code is provided by build_stdlib() from Rust
        // implementations. Generating MIR stubs here would create function entries with wrong
        // signatures (0-param stubs for methods that need a receiver), breaking codegen.
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
            let is_extern_only = has_extern_decls
                && !has_non_extern_class
                && !has_non_extern_abstract
                && typed_file.functions.is_empty()
                && typed_file.enums.is_empty();
            if is_extern_only {
                self.compiled_files
                    .insert(filename.to_string(), typed_file.clone());
                return Ok(typed_file);
            }
        }

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
            "DEBUG: [MIR LOWERING] filename='{}', is_stdlib_file={}",
            filename, is_stdlib_file
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

        let mir_result = lower_hir_to_mir_with_function_map(
            &hir_module,
            &self.string_interner,
            &self.type_table,
            &self.symbol_table,
            external_functions,
            external_functions_by_name,
            stdlib_mapping,
        )
        .map_err(|errors| {
            errors
                .into_iter()
                .map(|e| CompilationError {
                    message: format!("MIR lowering error: {:?}", e),
                    location: SourceLocation::unknown(),
                    category: ErrorCategory::InternalError,
                    suggestion: None,
                    related_errors: Vec::new(),
                })
                .collect::<Vec<_>>()
        })?;

        let mut mir_module = mir_result.module;

        // If this is a stdlib file, collect its function mappings
        if is_stdlib_file {
            debug!(
                "DEBUG: Collecting {} function mappings from stdlib file: {}",
                mir_result.function_map.len(),
                filename
            );
            for (symbol_id, func_id) in mir_result.function_map {
                self.stdlib_function_map.insert(symbol_id, func_id);
            }

            // Also collect name-based mappings for cross-file lookups
            // The function `name` field contains the qualified name (e.g., "StringTools.startsWith")
            let mut added_count = 0;
            let mut skipped_count = 0;
            for (func_id, func) in &mir_module.functions {
                // Only add non-empty CFG functions (skip forward refs/stubs)
                if !func.cfg.blocks.is_empty() {
                    self.stdlib_function_name_map
                        .insert(func.name.clone(), *func_id);
                    debug!("DEBUG: [NAME MAP] Added '{}' -> {:?}", func.name, func_id);
                    added_count += 1;
                } else {
                    debug!(
                        "DEBUG: [NAME MAP SKIP] '{}' has empty CFG (forward ref/stub)",
                        func.name
                    );
                    skipped_count += 1;
                }
            }
            debug!(
                "DEBUG: [NAME MAP] {} added, {} skipped for {}",
                added_count, skipped_count, filename
            );
        }

        // NOTE: extern-only files are handled above (before MIR generation).

        // Merge stdlib MIR (extern functions for Thread, Channel, Mutex, Arc, etc.)
        // This ensures extern runtime functions are available.
        // Uses build_stdlib_with_plugins to include HDLL extern declarations from loaded plugins.
        use crate::stdlib::build_stdlib_with_plugins;
        let mut stdlib_mir = build_stdlib_with_plugins(&self.compiler_plugin_registry);

        // Merge on-demand imported MIR modules (e.g., BalancedTree.hx) into the
        // user module. These were already renumbered to high IDs (100000+) during
        // load_imports_efficiently, so they won't collide with either user or stdlib IDs.
        // The user module already references these high IDs via external_function_map.
        for import_module in self.import_mir_modules.drain(..) {
            for (func_id, func) in import_module.functions {
                mir_module.functions.insert(func_id, func);
            }
        }

        // CRITICAL FIX: Renumber stdlib function IDs to avoid collisions with user functions
        // Each MIR module starts function IDs from 0, so when merging stdlib and user modules,
        // IDs will collide. For example:
        //   - User module: IrFunctionId(2) = "indexOf"
        //   - Stdlib module: IrFunctionId(2) = "free"
        // Without renumbering, stdlib's "free" would be skipped, causing vec_u8_free to call "indexOf"!

        // DEBUG: Print user functions before merging
        debug!(
            "DEBUG: User module has {} functions before merging:",
            mir_module.functions.len()
        );
        let mut user_func_ids: Vec<_> = mir_module.functions.keys().collect();
        user_func_ids.sort_by_key(|id| id.0);
        for func_id in user_func_ids.iter().take(5) {
            let func = &mir_module.functions[func_id];
            debug!("  - User IrFunctionId({}) = '{}'", func_id.0, func.name);
        }

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

        // Build mapping of old stdlib IDs to new renumbered IDs
        use crate::ir::IrFunctionId;
        use std::collections::HashMap;
        let mut id_mapping: HashMap<IrFunctionId, IrFunctionId> = HashMap::new();

        // Note: extern_functions is not used - externs are in the functions map with empty CFGs
        // So we only need to renumber the functions map

        // FIRST PASS: Build complete ID mapping for all stdlib functions
        // We must do this BEFORE updating CallDirect instructions so that all IDs are available
        for (old_id, _) in &stdlib_mir.functions {
            let new_id = IrFunctionId(old_id.0 + offset);
            id_mapping.insert(*old_id, new_id);
        }

        // SECOND PASS: Renumber functions and update their internal references
        let mut renumbered_functions = HashMap::new();
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

        // Build map of function names to IDs in the user module (before merging)
        let mut user_func_name_to_id: HashMap<String, IrFunctionId> = HashMap::new();
        for (func_id, func) in &mir_module.functions {
            user_func_name_to_id.insert(func.name.clone(), *func_id);
        }

        // Build a map of old ID -> new ID for all replacements
        // This must be done BEFORE we start modifying the module
        let mut id_replacements: HashMap<IrFunctionId, IrFunctionId> = HashMap::new();

        for (func_id, func) in &renumbered_functions {
            if let Some(&existing_id) = user_func_name_to_id.get(&func.name) {
                debug!(
                    "DEBUG: Will replace user function '{}' (ID {}) with stdlib version (ID {})",
                    func.name, existing_id.0, func_id.0
                );
                id_replacements.insert(existing_id, *func_id);
            }
        }

        // Now merge the stdlib functions
        for (func_id, func) in renumbered_functions {
            // DEBUG: Check signature after renumbering
            if func.name == "rayzor_channel_init" {
                debug!(
                    "DEBUG: AFTER renumbering - rayzor_channel_init (ID {}): params={}, extern={}",
                    func_id.0,
                    func.signature.parameters.len(),
                    func.cfg.blocks.is_empty()
                );
            }

            // If this function replaces an existing one, remove the old one
            if let Some(&existing_id) = user_func_name_to_id.get(&func.name) {
                mir_module.functions.remove(&existing_id);
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
        }

        // DEBUG: Print all function IDs in the merged module
        debug!(
            "DEBUG: Merged module has {} functions:",
            mir_module.functions.len()
        );
        let mut func_ids: Vec<_> = mir_module.functions.keys().collect();
        func_ids.sort_by_key(|id| id.0);
        for func_id in func_ids.iter().take(10) {
            // Print first 10
            let func = &mir_module.functions[func_id];
            debug!(
                "  - IrFunctionId({}) = '{}' (extern: {})",
                func_id.0,
                func.name,
                func.cfg.blocks.is_empty()
            );
        }

        // Verify MIR wrapper forward refs were replaced during merge.
        // A MirWrapper function with an empty CFG means the stdlib merge failed
        // to find the implementation — this would cause wrong values at runtime.
        for (func_id, func) in &mir_module.functions {
            if matches!(func.kind, crate::ir::FunctionKind::MirWrapper)
                && func.cfg.blocks.is_empty()
            {
                debug!(
                    "Unreplaced MIR forward ref after stdlib merge: '{}' (ID {})",
                    func.name, func_id.0
                );
            }
        }

        // Run monomorphization pass to specialize generic functions
        let mut monomorphizer = Monomorphizer::new();
        monomorphizer.monomorphize_module(&mut mir_module);
        let mono_stats = monomorphizer.stats();
        if mono_stats.generic_functions_found > 0 || mono_stats.instantiations_created > 0 {
            debug!("DEBUG: Monomorphization stats: {} generic functions, {} instantiations, {} call sites rewritten",
                      mono_stats.generic_functions_found,
                      mono_stats.instantiations_created,
                      mono_stats.call_sites_rewritten);
        }

        // Store the MIR module
        self.mir_modules.push(std::sync::Arc::new(mir_module));

        // Mark as successfully compiled to prevent redundant recompilation
        self.compiled_files
            .insert(filename.to_string(), typed_file.clone());

        Ok(typed_file)
    }

    /// Compile a single file using shared state (backward-compatible wrapper)

    fn compile_file_with_shared_state(
        &mut self,
        filename: &str,
        source: &str,
    ) -> Result<TypedFile, Vec<CompilationError>> {
        self.compile_file_with_shared_state_ex(filename, source, false)
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
        // Step 0: Discover @:hlNative metadata in user files and load HDLL plugins
        self.discover_and_load_hdlls();

        // Step 1: Analyze dependencies for user files
        let analysis = match self.analyze_dependencies() {
            Ok(a) => a,
            Err(errors) => {
                self.print_compilation_errors(&errors);
                return Err(errors);
            }
        };

        let mut all_typed_files = Vec::new();
        let mut all_errors = Vec::new();

        // Step 2: Pre-load stdlib files for explicit imports AND using statements in user files
        // This ensures typedefs like sys.FileStat are available before compilation
        // Also handles root-level imports like "import StringTools;" and "using StringTools;"
        let (imports_to_load, usings_to_load): (Vec<String>, Vec<String>) = self
            .user_files
            .iter()
            .filter_map(|file| {
                file.input
                    .as_ref()
                    .map(|source| (file.filename.clone(), source.clone()))
            })
            .fold(
                (Vec::new(), Vec::new()),
                |(mut imports, mut usings), (filename, source)| {
                    if let Ok(ast) = parser::parse_haxe_file(&filename, &source, false) {
                        // Collect imports
                        for import in &ast.imports {
                            if !import.path.is_empty() {
                                imports.push(import.path.join("."));
                            }
                        }
                        // Collect using statements (static extensions)
                        for using in &ast.using {
                            if !using.path.is_empty() {
                                usings.push(using.path.join("."));
                            }
                        }
                        // Auto-discover qualified type references in the AST
                        // (e.g., `new haxe.ds.BalancedTree<K,V>()` without explicit import)
                        let mut discovered = Vec::new();
                        collect_qualified_type_refs_from_ast(&ast, &mut discovered);
                        imports.extend(discovered);
                    }
                    (imports, usings)
                },
            );

        // Pre-load imports using efficient topological loading (avoids retry loops)
        let mut all_imports = imports_to_load;
        all_imports.extend(usings_to_load);
        let _ = self.load_imports_efficiently(&all_imports);

        // Step 3: Compile import.hx files using SHARED state
        let import_sources: Vec<(String, String)> = self
            .import_hx_files
            .iter()
            .filter_map(|f| f.input.as_ref().map(|s| (f.filename.clone(), s.clone())))
            .collect();

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

        // Step 4: Compile user files in dependency order using SHARED state
        // This ensures user files can see symbols from stdlib and other user files
        let user_sources: Vec<(String, String)> = analysis
            .compilation_order
            .iter()
            .filter_map(|&idx| {
                let file = &self.user_files[idx];
                file.input
                    .as_ref()
                    .map(|s| (file.filename.clone(), s.clone()))
            })
            .collect();

        for (filename, source) in user_sources {
            match self.compile_file_with_shared_state(&filename, &source) {
                Ok(typed_file) => {
                    all_typed_files.push(typed_file);
                }
                Err(errors) => {
                    // Check if any errors are unresolved types that we can try to load on-demand
                    let (loadable, other): (Vec<_>, Vec<_>) = errors.into_iter().partition(|e| {
                        e.message.contains("Unresolved type")
                            || e.message.contains("UnresolvedType")
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
                        debug!("  Retrying {} after loading dependencies...", filename);
                        match self.compile_file_with_shared_state(&filename, &source) {
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
                                    });

                                let mut retry_loaded = false;
                                for error in retry_loadable {
                                    if let Some(type_name) =
                                        self.extract_type_name_from_error(&error.message)
                                    {
                                        if !self.failed_type_loads.contains(&type_name) {
                                            if let Err(load_err) = self.load_import_file(&type_name)
                                            {
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
                                        filename
                                    );
                                    match self.compile_file_with_shared_state(&filename, &source) {
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

        // Step 5: Report all errors if any were found
        if !all_errors.is_empty() {
            self.print_compilation_errors(&all_errors);
            return Err(all_errors);
        }

        // Step 6: Include loaded stdlib files (typedefs, etc.) in the result
        // These were loaded on-demand during import resolution and contain type aliases
        // that need to be processed by HIR
        for stdlib_file in std::mem::take(&mut self.loaded_stdlib_typed_files) {
            all_typed_files.push(stdlib_file);
        }

        Ok(all_typed_files)
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
            // Skip built-in typedefs from StdTypes.hx (these are already loaded)
            if name == "Iterator"
                || name == "KeyValueIterator"
                || name == "Iterable"
                || name == "KeyValueIterable"
            {
                debug!("  Filtering out StdTypes typedef: {}", name);
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
        let (mir_module, metadata) = match load_blade(&cache_path) {
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
            .map(|s| Self::hash_source(&s))
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
        };

        // Save to BLADE file
        save_blade(&cache_path, module, metadata)
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
        use diagnostics::{ErrorFormatter, SourceMap};

        // Build source map with all parsed files
        let mut source_map = SourceMap::new();

        // Add stdlib files
        for stdlib_file in &self.stdlib_files {
            if let Some(ref source) = stdlib_file.input {
                source_map.add_file(stdlib_file.filename.clone(), source.clone());
            }
        }

        // Add import.hx files
        for import_file in &self.import_hx_files {
            if let Some(ref source) = import_file.input {
                source_map.add_file(import_file.filename.clone(), source.clone());
            }
        }

        // Add user files
        for user_file in &self.user_files {
            if let Some(ref source) = user_file.input {
                source_map.add_file(user_file.filename.clone(), source.clone());
            }
        }

        let formatter = ErrorFormatter::with_colors();

        for error in errors {
            let diagnostic = error.to_diagnostic(&source_map);
            let formatted = formatter.format_diagnostic(&diagnostic, &source_map);
            eprint!("{}", formatted);
        }
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

    /// Get HDLL function pointers for JIT linking.
    ///
    /// Returns symbol name and pointer pairs collected from all loaded HDLL plugins.
    /// These should be merged with runtime symbols when creating the backend.
    pub fn get_hdll_symbols(&self) -> &[(String, *const u8)] {
        &self.hdll_symbols
    }

    /// Register an external compiler plugin.
    ///
    /// This allows native packages (loaded via dlopen) to provide method mappings
    /// and extern declarations without modifying compiler source code. Must be
    /// called before `lower_to_tast()`.
    pub fn register_compiler_plugin(
        &mut self,
        plugin: Box<dyn crate::compiler_plugin::CompilerPlugin>,
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
    use std::collections::HashSet;

    let mut seen = HashSet::new();

    fn collect_from_type(ty: &Type, seen: &mut HashSet<String>, out: &mut Vec<String>) {
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

    fn collect_from_expr(expr: &Expr, seen: &mut HashSet<String>, out: &mut Vec<String>) {
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
            ExprKind::Field { expr: obj, .. } => {
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
        seen: &mut HashSet<String>,
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

        // Verify stdlib files were loaded
        assert!(unit.stdlib_files.len() > 0, "No stdlib files loaded");
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
