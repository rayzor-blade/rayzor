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
pub fn create_bundle(config: &BundleConfig) -> Result<usize, String> {
    use std::time::Instant;

    println!("Creating Rayzor Bundle: {}", config.output.display());

    let t0 = Instant::now();

    // Create compilation unit
    let mut comp_config = CompilationConfig::default();
    comp_config.enable_cache = config.enable_cache;
    comp_config.cache_dir = config.cache_dir.clone();

    let mut unit = CompilationUnit::new(comp_config);

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

    // Get MIR modules (uses cached BLADE files when available)
    let mir_modules = unit.get_mir_modules();

    if mir_modules.is_empty() {
        return Err("No MIR modules generated".to_string());
    }

    // Convert Arc<IrModule> to IrModule for the bundle
    let mut modules: Vec<_> = mir_modules.iter().map(|m| (**m).clone()).collect();

    let module_count = modules.len();

    // Find entry module and function
    let entry_module = modules
        .iter()
        .rev()
        .find(|m| {
            m.functions
                .values()
                .any(|f| f.name == "main" || f.name == "Main_main" || f.name.ends_with("_main"))
        })
        .map(|m| m.name.clone())
        .ok_or("No entry point found (no main function)")?;

    let entry_function = modules
        .iter()
        .find(|m| m.name == entry_module)
        .and_then(|m| {
            m.functions
                .values()
                .find(|f| f.name == "main" || f.name == "Main_main" || f.name.ends_with("_main"))
                .map(|f| f.name.clone())
        })
        .ok_or("Entry function not found")?;

    if config.verbose {
        println!("  entry    {}::{}", entry_module, entry_function);
    }

    // Tree-shake BEFORE optimization
    if config.strip {
        let stats = tree_shake::tree_shake_bundle(&mut modules, &entry_module, &entry_function);
        if config.verbose {
            println!(
                "  shake    -{} fn, -{} ext, -{} glob, -{} mod | kept {} fn, {} ext",
                stats.functions_removed,
                stats.extern_functions_removed,
                stats.globals_removed,
                stats.modules_removed,
                stats.functions_kept,
                stats.extern_functions_kept
            );
        }
    }

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

    // Discover all modules with their file paths
    let all_files = discover_stdlib_files(&stdlib_path);
    println!("  Discovered {} modules in stdlib", all_files.len());

    if config.list_only {
        println!();
        for (module_name, _) in &all_files {
            println!("  {}", module_name);
        }
        return Ok((all_files.len(), 0, 0));
    }

    let mut total_classes = 0;
    let mut total_enums = 0;
    let mut total_aliases = 0;
    let mut all_module_symbols: Vec<BladeModuleSymbols> = Vec::new();

    println!(
        "  Parsing and extracting types from {} files...",
        all_files.len()
    );

    for (module_name, file_path) in &all_files {
        let source = match std::fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                if config.verbose {
                    println!("    Warning: Could not read {}: {}", file_path.display(), e);
                }
                continue;
            }
        };

        let filename = file_path.to_string_lossy().to_string();

        let haxe_file = match parser::parse_haxe_file(&filename, &source, true) {
            Ok(f) => f,
            Err(e) => {
                if config.verbose {
                    println!("    Warning: Parse error in {}: {}", module_name, e);
                }
                continue;
            }
        };

        let type_info = extract_type_info_from_ast(&haxe_file);

        let class_count = type_info.classes.len();
        let enum_count = type_info.enums.len();
        let alias_count = type_info.type_aliases.len();
        let abstract_count = type_info.abstracts.len();

        if config.verbose
            && (class_count > 0 || enum_count > 0 || alias_count > 0 || abstract_count > 0)
        {
            println!(
                "  {}: {} classes, {} enums, {} aliases, {} abstracts",
                module_name, class_count, enum_count, alias_count, abstract_count
            );
        }

        total_classes += class_count;
        total_enums += enum_count;
        total_aliases += alias_count;

        if !type_info.classes.is_empty()
            || !type_info.enums.is_empty()
            || !type_info.type_aliases.is_empty()
            || !type_info.abstracts.is_empty()
        {
            let source_hash = hash_string(&filename);

            all_module_symbols.push(BladeModuleSymbols {
                name: module_name.clone(),
                source_path: filename,
                source_hash,
                types: type_info,
                dependencies: Vec::new(),
            });
        }
    }

    // Save symbol manifest
    let manifest_path = config.out_path.join("stdlib.bsym");
    println!();
    println!("Saving symbol manifest to {}...", manifest_path.display());
    println!(
        "  {} modules with symbol information",
        all_module_symbols.len()
    );

    if let Err(e) = save_symbol_manifest(&manifest_path, all_module_symbols) {
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

fn discover_stdlib_files(stdlib_path: &Path) -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    discover_files_recursive(stdlib_path, stdlib_path, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    files
}

fn discover_files_recursive(
    base_path: &Path,
    current_path: &Path,
    files: &mut Vec<(String, PathBuf)>,
) {
    let mut dir_paths: Vec<std::path::PathBuf> = match std::fs::read_dir(current_path) {
        Ok(e) => e.filter_map(|entry| entry.ok().map(|e| e.path())).collect(),
        Err(_) => return,
    };
    dir_paths.sort(); // Deterministic ordering

    for path in dir_paths {

        if path.is_dir() {
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if dir_name.starts_with('.') || dir_name.starts_with('_') {
                continue;
            }
            let skip_dirs = [
                "cpp", "cs", "flash", "hl", "java", "js", "lua", "neko", "php", "python", "eval",
            ];
            if skip_dirs.contains(&dir_name) {
                continue;
            }

            discover_files_recursive(base_path, &path, files);
        } else if path.extension().map(|e| e == "hx").unwrap_or(false) {
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            if file_name == "import.hx" {
                continue;
            }

            if let Ok(relative) = path.strip_prefix(base_path) {
                let module_name = relative
                    .to_string_lossy()
                    .replace('/', ".")
                    .replace('\\', ".")
                    .replace(".hx", "");
                files.push((module_name, path.clone()));
            }
        }
    }
}

pub fn extract_type_info_from_ast(haxe_file: &parser::HaxeFile) -> BladeTypeInfo {
    let mut type_info = BladeTypeInfo::default();

    let package: Vec<String> = haxe_file
        .package
        .as_ref()
        .map(|p| p.path.clone())
        .unwrap_or_default();

    for decl in &haxe_file.declarations {
        match decl {
            parser::TypeDeclaration::Class(class) => {
                let is_extern = class.modifiers.contains(&parser::Modifier::Extern);
                let native_name = extract_native_meta(&class.meta);
                let extends = class.extends.as_ref().map(|t| type_to_string(t));
                let implements: Vec<String> =
                    class.implements.iter().map(|t| type_to_string(t)).collect();
                let type_params: Vec<String> =
                    class.type_params.iter().map(|tp| tp.name.clone()).collect();

                let mut fields: Vec<BladeFieldInfo> = Vec::new();
                let mut static_fields: Vec<BladeFieldInfo> = Vec::new();
                let mut methods: Vec<BladeMethodInfo> = Vec::new();
                let mut static_methods: Vec<BladeMethodInfo> = Vec::new();
                let mut constructor: Option<BladeMethodInfo> = None;

                for field in &class.fields {
                    let is_static = field.modifiers.contains(&parser::Modifier::Static);
                    let is_inline = field.modifiers.contains(&parser::Modifier::Inline);
                    let is_public = matches!(field.access, Some(parser::Access::Public));

                    match &field.kind {
                        parser::ClassFieldKind::Var {
                            name,
                            type_hint,
                            expr,
                        } => {
                            let field_info = BladeFieldInfo {
                                name: name.clone(),
                                field_type: type_hint
                                    .as_ref()
                                    .map(|t| type_to_string(t))
                                    .unwrap_or_else(|| "Dynamic".to_string()),
                                is_public,
                                is_static,
                                is_final: false,
                                has_default: expr.is_some(),
                            };
                            if is_static {
                                static_fields.push(field_info);
                            } else {
                                fields.push(field_info);
                            }
                        }
                        parser::ClassFieldKind::Final {
                            name,
                            type_hint,
                            expr,
                        } => {
                            let field_info = BladeFieldInfo {
                                name: name.clone(),
                                field_type: type_hint
                                    .as_ref()
                                    .map(|t| type_to_string(t))
                                    .unwrap_or_else(|| "Dynamic".to_string()),
                                is_public,
                                is_static,
                                is_final: true,
                                has_default: expr.is_some(),
                            };
                            if is_static {
                                static_fields.push(field_info);
                            } else {
                                fields.push(field_info);
                            }
                        }
                        parser::ClassFieldKind::Property {
                            name, type_hint, ..
                        } => {
                            let field_info = BladeFieldInfo {
                                name: name.clone(),
                                field_type: type_hint
                                    .as_ref()
                                    .map(|t| type_to_string(t))
                                    .unwrap_or_else(|| "Dynamic".to_string()),
                                is_public,
                                is_static,
                                is_final: false,
                                has_default: false,
                            };
                            if is_static {
                                static_fields.push(field_info);
                            } else {
                                fields.push(field_info);
                            }
                        }
                        parser::ClassFieldKind::Function(func) => {
                            let method_info =
                                extract_method_from_ast(func, is_public, is_static, is_inline);
                            if func.name == "new" {
                                constructor = Some(method_info);
                            } else if is_static {
                                static_methods.push(method_info);
                            } else {
                                methods.push(method_info);
                            }
                        }
                    }
                }

                type_info.classes.push(BladeClassInfo {
                    name: class.name.clone(),
                    package: package.clone(),
                    extends,
                    implements,
                    type_params,
                    is_extern,
                    is_abstract: false,
                    is_final: class.modifiers.contains(&parser::Modifier::Final),
                    fields,
                    methods,
                    static_fields,
                    static_methods,
                    constructor,
                    native_name,
                });
            }
            parser::TypeDeclaration::Enum(enum_decl) => {
                let type_params: Vec<String> = enum_decl
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect();

                let variants: Vec<BladeEnumVariantInfo> = enum_decl
                    .constructors
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| {
                        let params: Vec<BladeParamInfo> = v
                            .params
                            .iter()
                            .map(|p| BladeParamInfo {
                                name: p.name.clone(),
                                param_type: p
                                    .type_hint
                                    .as_ref()
                                    .map(|t| type_to_string(t))
                                    .unwrap_or_else(|| "Dynamic".to_string()),
                                has_default: p.default_value.is_some(),
                                is_optional: p.optional,
                            })
                            .collect();
                        BladeEnumVariantInfo {
                            name: v.name.clone(),
                            params,
                            index: idx,
                        }
                    })
                    .collect();

                type_info.enums.push(BladeEnumInfo {
                    name: enum_decl.name.clone(),
                    package: package.clone(),
                    type_params,
                    variants,
                    is_extern: false,
                });
            }
            parser::TypeDeclaration::Typedef(typedef) => {
                let type_params: Vec<String> = typedef
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect();

                type_info.type_aliases.push(BladeTypeAliasInfo {
                    name: typedef.name.clone(),
                    package: package.clone(),
                    type_params,
                    target_type: type_to_string(&typedef.type_def),
                });
            }
            parser::TypeDeclaration::Abstract(abstract_decl) => {
                let native_name = extract_native_meta(&abstract_decl.meta);
                let type_params: Vec<String> = abstract_decl
                    .type_params
                    .iter()
                    .map(|tp| tp.name.clone())
                    .collect();

                let underlying_type = abstract_decl
                    .underlying
                    .as_ref()
                    .map(|t| type_to_string(t))
                    .unwrap_or_else(|| "Dynamic".to_string());

                let from_types: Vec<String> = abstract_decl
                    .from
                    .iter()
                    .map(|t| type_to_string(t))
                    .collect();
                let to_types: Vec<String> =
                    abstract_decl.to.iter().map(|t| type_to_string(t)).collect();

                let mut methods: Vec<BladeMethodInfo> = Vec::new();
                let mut static_methods: Vec<BladeMethodInfo> = Vec::new();

                for field in &abstract_decl.fields {
                    if let parser::ClassFieldKind::Function(func) = &field.kind {
                        let is_static = field.modifiers.contains(&parser::Modifier::Static);
                        let is_inline = field.modifiers.contains(&parser::Modifier::Inline);
                        let is_public = matches!(field.access, Some(parser::Access::Public));
                        let method_info =
                            extract_method_from_ast(func, is_public, is_static, is_inline);
                        if is_static {
                            static_methods.push(method_info);
                        } else {
                            methods.push(method_info);
                        }
                    }
                }

                type_info.abstracts.push(BladeAbstractInfo {
                    name: abstract_decl.name.clone(),
                    package: package.clone(),
                    type_params,
                    underlying_type,
                    forward_fields: vec![],
                    from_types,
                    to_types,
                    methods,
                    static_methods,
                    native_name,
                });
            }
            parser::TypeDeclaration::Interface(iface) => {
                let extends: Option<String> = iface.extends.first().map(|t| type_to_string(t));
                let implements: Vec<String> = iface
                    .extends
                    .iter()
                    .skip(1)
                    .map(|t| type_to_string(t))
                    .collect();
                let type_params: Vec<String> =
                    iface.type_params.iter().map(|tp| tp.name.clone()).collect();

                let mut methods: Vec<BladeMethodInfo> = Vec::new();
                for field in &iface.fields {
                    if let parser::ClassFieldKind::Function(func) = &field.kind {
                        let is_public = true;
                        let is_static = field.modifiers.contains(&parser::Modifier::Static);
                        let is_inline = field.modifiers.contains(&parser::Modifier::Inline);
                        let method_info =
                            extract_method_from_ast(func, is_public, is_static, is_inline);
                        methods.push(method_info);
                    }
                }

                type_info.classes.push(BladeClassInfo {
                    name: iface.name.clone(),
                    package: package.clone(),
                    extends,
                    implements,
                    type_params,
                    is_extern: false,
                    is_abstract: true,
                    is_final: false,
                    fields: vec![],
                    methods,
                    static_fields: vec![],
                    static_methods: vec![],
                    constructor: None,
                    native_name: None,
                });
            }
            parser::TypeDeclaration::Conditional(_) => {}
        }
    }

    type_info
}

fn extract_native_meta(meta: &[parser::Metadata]) -> Option<String> {
    for m in meta {
        let name = m.name.strip_prefix(':').unwrap_or(&m.name);
        if name == "native" {
            if let Some(first_param) = m.params.first() {
                if let parser::ExprKind::String(native_name) = &first_param.kind {
                    return Some(native_name.clone());
                }
            }
        }
    }
    None
}

fn extract_method_from_ast(
    func: &parser::Function,
    is_public: bool,
    is_static: bool,
    is_inline: bool,
) -> BladeMethodInfo {
    let params: Vec<BladeParamInfo> = func
        .params
        .iter()
        .map(|p| BladeParamInfo {
            name: p.name.clone(),
            param_type: p
                .type_hint
                .as_ref()
                .map(|t| type_to_string(t))
                .unwrap_or_else(|| "Dynamic".to_string()),
            has_default: p.default_value.is_some(),
            is_optional: p.optional,
        })
        .collect();

    let type_params: Vec<String> = func.type_params.iter().map(|tp| tp.name.clone()).collect();

    BladeMethodInfo {
        name: func.name.clone(),
        params,
        return_type: func
            .return_type
            .as_ref()
            .map(|t| type_to_string(t))
            .unwrap_or_else(|| "Void".to_string()),
        is_public,
        is_static,
        is_inline,
        type_params,
    }
}

fn type_to_string(ty: &parser::Type) -> String {
    match ty {
        parser::Type::Path { path, params, .. } => {
            let mut base = if path.package.is_empty() {
                path.name.clone()
            } else {
                format!("{}.{}", path.package.join("."), path.name)
            };

            if let Some(sub) = &path.sub {
                base = format!("{}.{}", base, sub);
            }

            if params.is_empty() {
                base
            } else {
                let param_strs: Vec<String> = params.iter().map(|t| type_to_string(t)).collect();
                format!("{}<{}>", base, param_strs.join(", "))
            }
        }
        parser::Type::Function { params, ret, .. } => {
            let param_strs: Vec<String> = params.iter().map(|t| type_to_string(t)).collect();
            format!("({}) -> {}", param_strs.join(", "), type_to_string(ret))
        }
        parser::Type::Anonymous { fields, .. } => {
            let field_strs: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", f.name, type_to_string(&f.type_hint)))
                .collect();
            format!("{{ {} }}", field_strs.join(", "))
        }
        parser::Type::Optional { inner, .. } => format!("Null<{}>", type_to_string(inner)),
        parser::Type::Parenthesis { inner, .. } => type_to_string(inner),
        parser::Type::Intersection { left, right, .. } => {
            format!("{} & {}", type_to_string(left), type_to_string(right))
        }
        parser::Type::Wildcard { .. } => "?".to_string(),
    }
}

fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}
