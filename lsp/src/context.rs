//! Compilation context for the LSP server.
//!
//! Maintains open file state and provides compilation for diagnostics,
//! hover, goto-definition, and completions.

use crate::analysis::{self, CompletionEntry, FileSymbolIndex};
use compiler::compilation::{CompilationConfig, CompilationUnit};
use compiler::tast::SymbolId;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;

/// Persistent compilation context shared across LSP requests.
pub struct LspContext {
    pub editor_cache: crate::editor_cache::EditorCache,
    /// Root workspace directory.
    pub root: PathBuf,
    pub(crate) workspace_roots: Vec<PathBuf>,
    /// Open file contents (URI → source text).
    pub open_files: HashMap<String, String>,
    pub document_versions: HashMap<String, i32>,
    /// Where imports are searched: the project's class paths, its
    /// libraries', then the workspace root.
    pub class_paths: Vec<PathBuf>,
    /// The project's defines, seen by its own files only.
    pub defines: Vec<String>,
    pub initialization_macros: Vec<String>,
    pub(crate) build_options: crate::build_config::Options,
    projects: Vec<crate::build_config::Settings>,
    /// The build file the configuration came from, if any.
    pub hxml: Option<PathBuf>,
    requested_hxml: Option<String>,
    /// What could not be applied from the configuration.
    pub config_problems: Vec<String>,
    /// How long the macros of one compile may run in all.
    pub macro_budget: std::time::Duration,
    /// The last analysis of each open document that produced an index.
    analyses: HashMap<String, Analysis>,
    sessions: VecDeque<crate::session_cache::SessionCache>,
    failed_analyses: HashSet<String>,
    pub analysis_cache_capacity: usize,
    diagnostics: HashMap<String, Vec<LspDiagnostic>>,
    workspace_index: crate::workspace_index::WorkspaceIndex,
    pub(crate) index_progress: Option<crate::workspace_index::IndexProgress>,
}

/// A compiled document: the unit its symbols live in and its position index.
pub struct Analysis {
    pub unit: Rc<RefCell<CompilationUnit>>,
    pub index: FileSymbolIndex,
}

impl Analysis {
    /// Whether `symbol` is called `name`: an index of earlier text can hold
    /// another name at a position that has since been edited.
    fn names(&self, symbol: SymbolId, name: &str) -> bool {
        let unit = self.unit.borrow();
        unit.symbol_table
            .get_symbol(symbol)
            .and_then(|s| unit.string_interner.get(s.name))
            == Some(name)
    }
}

impl LspContext {
    /// Moves mutable compiler graphs into one task; suspended tasks never
    /// share them with a query or another compile.
    pub(crate) fn fork_analysis(&mut self) -> Self {
        // Each runnable task owns at most one graph. Older pages retain their
        // editor answers in BLADE rather than multiplying suspended heaps.
        while self.sessions.len() > 1 {
            let oldest = self.sessions.pop_front().unwrap();
            self.analyses
                .retain(|_, analysis| !Rc::ptr_eq(&analysis.unit, &oldest.unit));
        }
        let mut task = Self::new(self.root.clone());
        task.editor_cache = self.editor_cache.clone();
        task.workspace_roots = self.workspace_roots.clone();
        task.open_files = self.open_files.clone();
        task.document_versions = self.document_versions.clone();
        task.class_paths = self.class_paths.clone();
        task.defines = self.defines.clone();
        task.initialization_macros = self.initialization_macros.clone();
        task.build_options = self.build_options.clone();
        task.projects = self.projects.clone();
        task.hxml = self.hxml.clone();
        task.requested_hxml = self.requested_hxml.clone();
        task.config_problems = self.config_problems.clone();
        task.macro_budget = self.macro_budget;
        task.analysis_cache_capacity = 1;
        task.workspace_index = self.workspace_index.clone();
        task.analyses = std::mem::take(&mut self.analyses);
        task.sessions = std::mem::take(&mut self.sessions);
        task
    }

    pub(crate) fn adopt_analysis(&mut self, uri: &str, mut task: Self) -> Vec<LspDiagnostic> {
        let diagnostics = task.diagnostics.remove(uri).unwrap_or_default();
        self.failed_analyses.remove(uri);
        if task.failed_analyses.contains(uri) {
            self.failed_analyses.insert(uri.to_string());
        }
        self.analyses.extend(
            task.analyses
                .into_iter()
                .filter(|(uri, _)| self.open_files.contains_key(uri)),
        );
        self.sessions.append(&mut task.sessions);
        while self.sessions.len() > self.analysis_cache_capacity.clamp(1, 8) {
            let oldest = self.sessions.pop_front().unwrap();
            self.analyses
                .retain(|_, analysis| !Rc::ptr_eq(&analysis.unit, &oldest.unit));
        }
        diagnostics
    }

    pub(crate) fn mark_analysis_failed(&mut self, uri: &str) {
        self.failed_analyses.insert(uri.to_string());
    }

    pub fn new(root: PathBuf) -> Self {
        Self {
            editor_cache: Default::default(),
            workspace_roots: vec![root.clone()],
            root,
            open_files: HashMap::new(),
            document_versions: HashMap::new(),
            class_paths: Vec::new(),
            defines: Vec::new(),
            initialization_macros: Vec::new(),
            build_options: Default::default(),
            projects: Vec::new(),
            hxml: None,
            requested_hxml: None,
            config_problems: Vec::new(),
            macro_budget: std::time::Duration::from_secs(5),
            analyses: HashMap::new(),
            sessions: VecDeque::new(),
            failed_analyses: HashSet::new(),
            analysis_cache_capacity: 2,
            diagnostics: HashMap::new(),
            workspace_index: Default::default(),
            index_progress: None,
        }
    }

    pub fn load_workspace_config(&mut self, hxml: Option<&str>) {
        self.requested_hxml = hxml.map(str::to_owned);
        self.projects = self
            .workspace_roots
            .iter()
            .flat_map(|root| crate::build_config::load(root, hxml))
            .collect();
        self.defines.clear();
        self.initialization_macros.clear();
        self.hxml = None;
        self.build_options = Default::default();
        self.class_paths = self
            .projects
            .iter()
            .flat_map(|project| project.class_paths.clone())
            .collect();
        self.config_problems = self
            .projects
            .iter()
            .flat_map(|project| project.problems.clone())
            .collect();
        if let Some(project) = self.projects.first() {
            self.defines = project.defines.clone();
            self.initialization_macros = project.macros.clone();
            self.hxml = project.hxml.clone();
            self.build_options = project.options.clone();
        }
        self.workspace_index = crate::workspace_index::WorkspaceIndex::start(
            self.class_paths.clone(),
            self.defines
                .iter()
                .chain(&self.build_options.extra_defines)
                .cloned()
                .collect(),
            self.projects
                .iter()
                .map(|project| {
                    (
                        project
                            .root
                            .canonicalize()
                            .unwrap_or_else(|_| project.root.clone()),
                        project
                            .defines
                            .iter()
                            .chain(&project.options.extra_defines)
                            .cloned()
                            .collect(),
                    )
                })
                .collect(),
            self.index_progress.clone(),
        );
    }

    pub(crate) fn reload_workspace_config(&mut self) {
        let hxml = self.requested_hxml.clone();
        self.invalidate_analysis_cache();
        self.load_workspace_config(hxml.as_deref());
        self.configure_editor_cache();
    }

    pub(crate) fn configuration_watch_paths(&self) -> Vec<PathBuf> {
        self.projects
            .iter()
            .flat_map(|project| {
                project
                    .options
                    .configuration_files
                    .iter()
                    .chain(&project.options.native_libs)
                    .chain(&project.options.packages)
                    .chain(project.options.resources.iter().map(|(path, _)| path))
                    .cloned()
            })
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub(crate) fn settings_for_uri(&self, uri: &str) -> crate::build_config::Settings {
        let mut settings = self.project_settings_for_uri(uri);
        crate::build_config::include_source_library(
            &mut settings,
            &PathBuf::from(uri_to_path(uri)),
        );
        settings
    }

    fn project_settings_for_uri(&self, uri: &str) -> crate::build_config::Settings {
        let path = PathBuf::from(uri_to_path(uri));
        let path = path.canonicalize().unwrap_or(path);
        let owner = self
            .projects
            .iter()
            .filter_map(|project| {
                crate::build_config::source_depth(&path, &project.root, &project.class_paths)
                    .map(|depth| (depth, project))
            })
            .max_by_key(|(depth, _)| *depth)
            .map(|(_, project)| project)
            .or_else(|| {
                self.projects.iter().rev().find(|project| {
                    crate::stdlib::discover_for_project(&project.root, project.options.rayzor)
                        .is_ok_and(|paths| {
                            paths.iter().any(|root| {
                                path.starts_with(
                                    root.canonicalize().unwrap_or_else(|_| root.clone()),
                                )
                            })
                        })
                })
            });
        if let Some(project) = owner {
            if self.projects.len() > 1 {
                return project.clone();
            }
        } else if !self.projects.is_empty() {
            // An open file can outlive its workspace folder. It retains its
            // own project's settings instead of inheriting another folder's.
            let parent = path.parent().unwrap_or(&path);
            let root =
                compiler::workspace::find_project_root(parent).unwrap_or_else(|| parent.into());
            if let Some(project) = crate::build_config::load(&root, None).into_iter().next() {
                return project;
            }
        }
        crate::build_config::Settings {
            root: self
                .projects
                .first()
                .map_or_else(|| self.root.clone(), |project| project.root.clone()),
            class_paths: self.class_paths.clone(),
            defines: self.defines.clone(),
            macros: self.initialization_macros.clone(),
            hxml: self.hxml.clone(),
            options: self.build_options.clone(),
            problems: self.config_problems.clone(),
        }
    }

    pub(crate) fn configure_editor_cache(&self) {
        self.editor_cache.reset_projects();
        for project in &self.projects {
            if let Ok(stdlib_paths) =
                crate::stdlib::discover_for_project(&project.root, project.options.rayzor)
            {
                self.editor_cache.configure_project(
                    crate::session_cache::SessionKey {
                        class_paths: project.class_paths.clone(),
                        stdlib_paths,
                        defines: project.defines.clone(),
                        macros: project.macros.clone(),
                        macro_budget: self.macro_budget,
                        options: project.options.clone(),
                    },
                    &project.root,
                );
            }
        }
    }

    pub(crate) fn configuration_warnings(&self) -> Vec<LspDiagnostic> {
        let mut files = std::collections::BTreeSet::new();
        self.projects.iter()
            .filter(|project| project.options.hxml_target_missing)
            .filter_map(|project| project.hxml.clone())
            .filter(|path| files.insert(path.clone()))
            .map(|path| {
                let mut warning = LspDiagnostic::error(
                    &path.to_string_lossy(), 1, 1,
                    "No Haxe target is configured for the selected HXML build. Target-dependent types and #if branches may be inaccurate; select an explicit target such as --hl, --js, or --cpp.".into(),
                );
                warning.severity = DiagSeverity::Warning;
                warning
            }).collect()
    }

    /// Compile a file and return diagnostics. Also builds the symbol index.
    pub fn compile_file(&mut self, uri: &str, source: &str) -> Vec<LspDiagnostic> {
        let profiling = std::env::var_os("RAYZOR_LSP_PROFILE").is_some();
        let started = std::time::Instant::now();
        self.failed_analyses.insert(uri.to_string());
        let filename = uri_to_path(uri);
        let settings = self.settings_for_uri(uri);
        self.workspace_index
            .update(std::path::Path::new(&filename), Some(source));

        let mut config = CompilationConfig {
            load_stdlib: true,
            profile_typecheck: profiling,
            emit_safety_warnings: false,
            record_references: true,
            user_defines: settings.defines.clone(),
            continue_after_macro_errors: true,
            typecheck_only: true,
            source_overrides: std::sync::Arc::new(
                self.open_files
                    .iter()
                    .map(|(uri, source)| {
                        let path = PathBuf::from(uri_to_path(uri));
                        (path.canonicalize().unwrap_or(path), source.clone())
                    })
                    .collect(),
            ),
            ..Default::default()
        };
        config.stdlib_paths =
            match crate::stdlib::discover_for_project(&settings.root, settings.options.rayzor) {
                Ok(paths) => paths,
                Err(message) => return vec![LspDiagnostic::error(&filename, 0, 0, message)],
            };
        // Cached Rayzor library symbols do not describe a system Haxe library.
        config.enable_cache = settings.options.rayzor && settings.options.cache_enabled;
        config.cache_dir = settings.options.cache_dir.clone();
        config.extra_defines = settings.options.extra_defines.clone();
        config.resources = std::sync::Arc::new(
            settings
                .options
                .resources
                .iter()
                .filter_map(|(path, name)| {
                    std::fs::read(path).ok().map(|bytes| (name.clone(), bytes))
                })
                .collect(),
        );
        config.pipeline_config = config.pipeline_config.skip_analysis();

        // A syntax error is reported as such, and the last analysis is kept:
        // compiling text the parser rejects yields a partial program and no
        // error, because a fallback parser recovers by dropping what it cannot
        // read.
        let syntax = self.syntax_errors(
            &filename,
            source,
            &settings
                .defines
                .iter()
                .chain(&settings.options.extra_defines)
                .cloned()
                .collect::<Vec<_>>(),
        );
        if !syntax.is_empty() {
            return syntax;
        }

        let key = crate::session_cache::SessionKey {
            class_paths: settings.class_paths.clone(),
            stdlib_paths: config.stdlib_paths.clone(),
            defines: settings.defines.clone(),
            macros: settings.macros.clone(),
            macro_budget: self.macro_budget,
            options: settings.options.clone(),
        };
        let mut inputs = crate::session_cache::Inputs::capture(&key, &self.open_files);
        inputs.set_source(&filename, source);
        // Old document indices cannot answer against an outdated dependency
        // graph. Keep bounded snapshots for undo/reopen, but reindex those
        // documents on their next request using the current source generation.
        self.analyses.retain(|_, analysis| {
            self.sessions.iter().any(|session| {
                Rc::ptr_eq(&session.unit, &analysis.unit)
                    && (session.key != key
                        || (inputs.reusable
                            && session.inputs == inputs
                            && session.unit.borrow().macro_dependencies.is_current()))
            })
        });
        // BLADE keeps editor answers for old source generations; their compiler
        // graphs need not remain alive alongside the replacement graph.
        self.sessions.retain(|session| {
            session.key != key
                || (inputs.reusable
                    && session.inputs == inputs
                    && session.unit.borrow().macro_dependencies.is_current())
        });
        let reusable = self.sessions.iter().rposition(|session| {
            inputs.reusable
                && session.key == key
                && session.inputs == inputs
                && session.can_extend(&filename)
        });
        let reused = reusable.is_some();
        let mut session = match reusable {
            Some(index) => self.sessions.remove(index).unwrap(),
            None => {
                while self.sessions.len() >= self.analysis_cache_capacity.clamp(1, 8) {
                    let oldest = self.sessions.pop_front().unwrap();
                    self.analyses
                        .retain(|_, analysis| !Rc::ptr_eq(&analysis.unit, &oldest.unit));
                }
                let mut unit = CompilationUnit::new(config);
                for path in settings
                    .options
                    .configuration_files
                    .iter()
                    .chain(&settings.options.native_libs)
                    .chain(&settings.options.packages)
                    .chain(settings.options.resources.iter().map(|(path, _)| path))
                {
                    unit.macro_dependencies.register_read(path);
                }
                for cp in &settings.class_paths {
                    unit.add_source_path(cp.clone());
                }
                let mut native_libraries = Vec::new();
                for path in &settings.options.native_libs {
                    if let Ok((library, plugin)) =
                        compiler::native_libs::load_manifest_native_lib_for_analysis(path)
                    {
                        unit.register_compiler_plugin(Box::new(plugin));
                        native_libraries.push(library);
                    }
                }
                for path in &settings.options.packages {
                    if let Ok(package) = compiler::rpkg::load_rpkg(path) {
                        unit.register_compiler_plugin(Box::new(
                            compiler::compiler_plugin::NativePlugin::from_method_entries(
                                package
                                    .plugin_name
                                    .as_deref()
                                    .unwrap_or(&package.package_name),
                                package.methods,
                            ),
                        ));
                    }
                }
                if let Err(error) = unit.load_stdlib() {
                    return vec![LspDiagnostic::error(
                        &filename,
                        0,
                        0,
                        format!("Stdlib: {error}"),
                    )];
                }
                let settings_file = settings.hxml.clone().unwrap_or_else(|| {
                    settings.root.join(if settings.options.rayzor {
                        "rayzor.toml"
                    } else {
                        "haxelib.json"
                    })
                });
                let settings_source = std::fs::read_to_string(&settings_file).unwrap_or_default();
                let configuration_file = if settings.options.rayzor {
                    settings.root.join("rayzor.toml")
                } else {
                    settings_file.clone()
                };
                let mut initialization_errors: Vec<_> = settings
                    .problems
                    .iter()
                    .map(|message| {
                        LspDiagnostic::error(
                            &configuration_file.to_string_lossy(),
                            1,
                            1,
                            message.clone(),
                        )
                    })
                    .collect();
                for call in &settings.macros {
                    let (line, column) = settings_source
                        .find(call)
                        .map(|offset| line_column(&settings_source, offset))
                        .unwrap_or((1, 1));
                    initialization_errors.extend(
                        unit.run_initialization_macros(std::slice::from_ref(call))
                            .into_iter()
                            .map(|message| {
                                LspDiagnostic::error(
                                    &settings_file.to_string_lossy(),
                                    line,
                                    column,
                                    message,
                                )
                            }),
                    );
                }
                let mut session = crate::session_cache::SessionCache::new(
                    key,
                    inputs,
                    unit,
                    initialization_errors,
                );
                session.native_libraries = native_libraries;
                session
            }
        };
        if let Some(index) = session.indexes.get(uri).cloned() {
            let result = session.diagnostics.clone();
            self.analyses.insert(
                uri.to_string(),
                Analysis {
                    unit: session.unit.clone(),
                    index,
                },
            );
            self.failed_analyses.remove(uri);
            self.sessions.push_back(session);
            if profiling {
                eprintln!(
                    "lsp-analysis: reused=true indexed=true total_ms={:.2} file={}",
                    started.elapsed().as_secs_f64() * 1000.0,
                    filename
                );
            }
            return result;
        }
        let shared_unit = session.unit.clone();
        let mut unit = shared_unit.borrow_mut();
        let initialization_errors = session.initialization_errors.clone();
        // A source reached as an import already has its types and reference
        // identities in this graph. Index it without compiling its body again.
        let already_typed = unit.has_typed_file(&filename);
        let lowered = if already_typed {
            Ok(Vec::new())
        } else {
            unit.set_editor_entry(&filename);
            let already_root = unit.user_files.iter().any(|file| file.filename == filename);
            if !already_root && let Err(error) = unit.add_file(source, &filename) {
                return vec![LspDiagnostic::error(
                    &filename,
                    0,
                    0,
                    format!("Parse: {error}"),
                )];
            }
            unit.lower_to_tast()
        };

        let mut result = match lowered {
            Ok(_typed_files) => Vec::new(),
            Err(errors) => errors
                .iter()
                .filter(|e| e.category != compiler::pipeline::ErrorCategory::MacroExpansionError)
                .map(|e| LspDiagnostic {
                    file: unit
                        .file_path_of(e.location.file_id)
                        .unwrap_or(&filename)
                        .to_string(),
                    line: e.location.line,
                    column: e.location.column,
                    end_line: e.location.line,
                    end_column: e.location.column + 1,
                    severity: match e.category {
                        compiler::pipeline::ErrorCategory::ParseError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::TypeError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::ConcurrencyError => DiagSeverity::Error,
                        compiler::pipeline::ErrorCategory::MacroExpansionError => {
                            DiagSeverity::Error
                        }
                        _ => DiagSeverity::Warning,
                    },
                    message: e.message.clone(),
                    suggestion: e.suggestion.clone(),
                    related: e.related_errors.clone(),
                    related_information: Vec::new(),
                })
                .collect(),
        };
        result.extend(session.diagnostics.iter().cloned());
        result.extend(
            unit.check_typed_file(&filename)
                .diagnostics
                .into_iter()
                .filter_map(|diagnostic| {
                    if diagnostic.severity != diagnostics::DiagnosticSeverity::Error {
                        return None;
                    }
                    let file = unit
                        .file_path_of(diagnostic.span.file_id.as_usize() as u32)
                        .unwrap_or(&filename);
                    Some(LspDiagnostic::error(
                        file,
                        diagnostic.span.start.line as u32,
                        diagnostic.span.start.column as u32,
                        diagnostic.message,
                    ))
                }),
        );
        result.extend(unit.macro_diagnostics.iter().filter_map(|diagnostic| {
            use compiler::macro_system::MacroSeverity;
            let severity = match diagnostic.severity {
                MacroSeverity::Error => DiagSeverity::Error,
                MacroSeverity::Warning => DiagSeverity::Warning,
                MacroSeverity::Info => return None,
            };
            let file = unit
                .file_path_of(diagnostic.location.file_id)
                .unwrap_or(&filename);
            let mut item = LspDiagnostic::error(
                file,
                diagnostic.location.line,
                diagnostic.location.column,
                diagnostic.message.clone(),
            );
            item.severity = severity;
            if let Some(call) = diagnostic.call_site
                && let Some(file) = unit.file_path_of(call.file_id)
                && let Ok(uri) = file_uri(file).parse()
            {
                let at = lsp_types::Position::new(
                    call.line.saturating_sub(1),
                    call.column.saturating_sub(1),
                );
                item.related_information
                    .push(lsp_types::DiagnosticRelatedInformation {
                        location: lsp_types::Location {
                            uri,
                            range: lsp_types::Range::new(at, at),
                        },
                        message: "Macro invoked here".into(),
                    });
            }
            Some(item)
        }));
        result.extend(initialization_errors);

        let file_id = unit.file_id_of(&filename).unwrap_or(u32::MAX);
        let mut preprocessor = parser::preprocessor::PreprocessorConfig::default();
        preprocessor.add_defines(
            settings
                .defines
                .iter()
                .chain(&settings.options.extra_defines),
        );
        preprocessor.defines.insert("eval".into());
        let text = parser::preprocessor::preprocess(source, &preprocessor);
        if let Ok(file) = parser::rd::rd_parse(&text, &filename, false, false) {
            FileSymbolIndex::record_source_declarations(&file, &text, file_id, &mut unit);
            crate::structural_fields::record(&file, &text, file_id, &mut unit);
            crate::type_references::record(&file, &text, file_id, &mut unit, &self.workspace_index);
        }
        crate::documentation::record(&mut unit);
        for file in unit.stdlib_files.iter().chain(&unit.user_files) {
            self.workspace_index.ensure_parsed_file(file);
        }
        // Imported placeholders may still carry a provisional class kind.
        // Source declarations determine their editor presentation.
        let source_types = self.workspace_index.type_identities();
        let declaration_kinds: Vec<_> = unit
            .symbol_table
            .all_symbols()
            .filter(|symbol| {
                matches!(
                    symbol.kind,
                    compiler::tast::symbols::SymbolKind::Class
                        | compiler::tast::symbols::SymbolKind::Interface
                        | compiler::tast::symbols::SymbolKind::Abstract
                        | compiler::tast::symbols::SymbolKind::Enum
                        | compiler::tast::symbols::SymbolKind::TypeAlias
                )
            })
            .filter_map(|symbol| {
                let qualified = symbol.qualified_name.unwrap_or_else(|| {
                    unit.symbol_table.build_qualified_name(
                        symbol.id,
                        &unit.scope_tree,
                        &unit.string_interner,
                    )
                });
                let name = unit.string_interner.get(qualified)?;
                let kind = source_types.get(name)?.kind;
                Some((symbol.id, kind))
            })
            .collect();
        for (id, kind) in declaration_kinds {
            unit.symbol_table.get_symbol_mut(id).unwrap().kind = kind;
        }
        let index = FileSymbolIndex::build(
            &unit.symbol_table,
            &unit.string_interner,
            &unit.symbol_references,
            file_id,
            &text,
        );
        if profiling {
            let timings = unit.typecheck_timings();
            eprintln!(
                "lsp-analysis: reused={} already_typed={} total_ms={:.2} typed_files={} imports_reused={} imports_compiled={} symbols={} file={}",
                reused,
                already_typed,
                started.elapsed().as_secs_f64() * 1000.0,
                if already_typed { 0 } else { timings.files_seen },
                if already_typed {
                    0
                } else {
                    timings.import_already_compiled
                },
                if already_typed {
                    0
                } else {
                    timings.import_fresh_compiles
                },
                unit.symbol_table.len(),
                filename
            );
        }
        if profiling && !already_typed {
            eprintln!("lsp-typecheck: {:?}", unit.typecheck_timings());
        }
        session.indexes.insert(uri.to_string(), index.clone());
        drop(unit);
        self.analyses.insert(
            uri.to_string(),
            Analysis {
                unit: shared_unit,
                index,
            },
        );
        self.failed_analyses.remove(uri);
        // Keep one copy of diagnostics for modules served from the shared
        // typed graph on later pages. Bound this along with the graph itself.
        for diagnostic in &result {
            if !session.diagnostics.iter().any(|old| {
                old.file == diagnostic.file
                    && old.line == diagnostic.line
                    && old.column == diagnostic.column
                    && old.message == diagnostic.message
            }) {
                session.diagnostics.push(diagnostic.clone());
            }
        }
        self.sessions.push_back(session);
        while self.sessions.len() > self.analysis_cache_capacity.clamp(1, 8) {
            let oldest = self.sessions.pop_front().unwrap();
            self.analyses
                .retain(|_, analysis| !Rc::ptr_eq(&analysis.unit, &oldest.unit));
        }

        result
    }

    /// Publish errors on their own files, merging contributions from open documents.
    pub fn update_diagnostics(
        &mut self,
        uri: &str,
        mut diagnostics: Vec<LspDiagnostic>,
    ) -> Vec<(String, Vec<LspDiagnostic>)> {
        for diagnostic in &mut diagnostics {
            diagnostic.file = self.editor_path(&diagnostic.file);
        }
        let mut files = std::collections::BTreeSet::from([uri_to_path(uri)]);
        files.extend(diagnostics.iter().map(|d| d.file.clone()));
        if let Some(previous) = self.diagnostics.insert(uri.to_string(), diagnostics) {
            files.extend(previous.into_iter().map(|d| d.file));
        }
        files
            .into_iter()
            .map(|file| {
                let mut found = Vec::new();
                for diagnostic in self
                    .diagnostics
                    .values()
                    .flatten()
                    .filter(|d| d.file == file)
                {
                    if !found.iter().any(|d: &LspDiagnostic| {
                        d.line == diagnostic.line
                            && d.column == diagnostic.column
                            && d.message == diagnostic.message
                    }) {
                        found.push(diagnostic.clone());
                    }
                }
                (file, found)
            })
            .collect()
    }

    fn editor_path(&self, file: &str) -> String {
        if self.open_files.contains_key(&file_uri(file)) {
            return file.to_owned();
        }
        let path = PathBuf::from(file);
        let identity = path.canonicalize().unwrap_or(path);
        let canonical = identity.to_string_lossy();
        if self.open_files.contains_key(&file_uri(&canonical)) {
            return canonical.into_owned();
        }
        self.open_files
            .keys()
            .find_map(|uri| {
                let open = uri_to_path(uri);
                let path = PathBuf::from(&open);
                (path.canonicalize().unwrap_or(path) == identity).then_some(open)
            })
            .unwrap_or_else(|| file.to_owned())
    }

    /// Forget a closed document and its analysis.
    pub fn close(&mut self, uri: &str) {
        self.open_files.remove(uri);
        self.document_versions.remove(uri);
        self.analyses.remove(uri);
        self.failed_analyses.remove(uri);
        self.diagnostics.remove(uri);
        self.sessions.retain(|session| {
            session
                .indexes
                .keys()
                .any(|uri| self.open_files.contains_key(uri))
        });
        self.index_disk_file(uri);
    }

    pub fn needs_analysis(&mut self, uri: &str) -> bool {
        self.open_files.contains_key(uri)
            && !self.has_current_analysis(uri)
            && !self.failed_analyses.contains(uri)
    }

    pub fn has_current_analysis(&self, uri: &str) -> bool {
        self.analyses.get(uri).is_some_and(|analysis| {
            self.open_files.get(uri).is_some_and(|source| {
                let unit = analysis.unit.borrow();
                unit.source_matches(analysis.index.file_id, source)
                    && unit.macro_dependencies.is_current()
            })
        })
    }

    pub(crate) fn indexed_names(&self, uri: &str) -> Vec<(u32, u32, u32, SymbolId)> {
        self.analyses
            .get(uri)
            .map(|analysis| {
                analysis
                    .index
                    .names
                    .iter()
                    .filter_map(|(&(line, column), &(len, symbol))| {
                        Some((line.checked_sub(1)?, column.checked_sub(1)?, len, symbol))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn indexed_expansions(&self, uri: &str) -> Vec<serde_json::Value> {
        let Some(analysis) = self.analyses.get(uri) else {
            return Vec::new();
        };
        let unit = analysis.unit.borrow();
        let Some(source) = unit.source_of(analysis.index.file_id) else {
            return Vec::new();
        };
        unit.macro_expansions
            .iter()
            .filter(|origin| origin.call_site.file_id == analysis.index.file_id)
            .map(|origin| macro_expansion_response(origin, source))
            .collect()
    }

    pub(crate) fn macro_dependencies(
        &self,
        uri: &str,
    ) -> Vec<compiler::macro_system::dependencies::Dependency> {
        self.analyses
            .get(uri)
            .map(|analysis| analysis.unit.borrow().macro_dependencies.snapshot())
            .unwrap_or_default()
    }

    /// A changed dependency, newly created module, or compiler panic makes
    /// the workspace graph unsafe to reuse, including its negative lookups.
    pub fn invalidate_analysis_cache(&mut self) {
        self.sessions.clear();
        self.analyses.clear();
        self.failed_analyses.clear();
    }

    pub(crate) fn invalidate_changed_inputs(&mut self) {
        let mut generations = HashMap::new();
        self.sessions.retain(|session| {
            let inputs = generations.entry(session.key.clone()).or_insert_with(|| {
                crate::session_cache::Inputs::capture(&session.key, &self.open_files)
            });
            inputs.reusable
                && *inputs == session.inputs
                && session.unit.borrow().macro_dependencies.is_current()
        });
        self.analyses.retain(|_, analysis| {
            self.sessions
                .iter()
                .any(|session| Rc::ptr_eq(&session.unit, &analysis.unit))
        });
        self.failed_analyses.clear();
    }

    pub fn index_disk_file(&self, uri: &str) {
        let path = PathBuf::from(uri_to_path(uri));
        let source = self
            .open_files
            .get(uri)
            .cloned()
            .or_else(|| std::fs::read_to_string(&path).ok());
        self.workspace_index.update(&path, source.as_deref());
    }

    pub fn workspace_symbols(&self, query: &str) -> Vec<serde_json::Value> {
        self.workspace_index.search(query)
    }

    pub fn poll_workspace_index(&self) -> bool {
        self.workspace_index.pump()
    }

    fn syntax_errors(
        &self,
        filename: &str,
        source: &str,
        defines: &[String],
    ) -> Vec<LspDiagnostic> {
        let mut config = parser::preprocessor::PreprocessorConfig::default();
        config.defines.insert("eval".to_string());
        config.add_defines(defines.iter());
        let text = parser::preprocessor::preprocess(source, &config);
        let import_file = filename.ends_with("import.hx");
        let Err(errors) = parser::rd::rd_parse(&text, filename, import_file, false) else {
            return Vec::new();
        };
        errors
            .iter()
            .map(|e| {
                let (line, column) = line_column(&text, e.span.start);
                let (end_line, end_column) = line_column(&text, e.span.end.max(e.span.start + 1));
                LspDiagnostic {
                    file: filename.to_string(),
                    line,
                    column,
                    end_line,
                    end_column,
                    severity: DiagSeverity::Error,
                    message: e.message.clone(),
                    suggestion: None,
                    related: Vec::new(),
                    related_information: Vec::new(),
                }
            })
            .collect()
    }

    fn symbol_at(&self, uri: &str, line: u32, col: u32) -> Option<(&Analysis, SymbolId)> {
        let a = self.analyses.get(uri)?;
        let current = self.open_files.get(uri)?;
        let unit = a.unit.borrow();
        let original = unit.source_of(a.index.file_id)?;
        if current != original {
            let (&(row, start), &(len, _)) = a.index.names.range(..=(line, col)).next_back()?;
            let old = original.lines().nth(row.checked_sub(1)? as usize)?;
            let new = current.lines().nth(row.checked_sub(1)? as usize)?;
            let begins = start.saturating_sub(1) as usize;
            let is_name = |unit: Option<u16>| {
                unit.and_then(|c| char::from_u32(c as u32))
                    .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '$')
            };
            for boundary in [begins.checked_sub(1), Some(begins + len as usize)]
                .into_iter()
                .flatten()
            {
                if is_name(new.encode_utf16().nth(boundary))
                    && !is_name(old.encode_utf16().nth(boundary))
                {
                    return None;
                }
            }
            if !old
                .encode_utf16()
                .skip(start.saturating_sub(1) as usize)
                .take(len as usize)
                .eq(new
                    .encode_utf16()
                    .skip(start.saturating_sub(1) as usize)
                    .take(len as usize))
            {
                return None;
            }
        }
        Some((a, a.index.find_symbol_at(line, col)?))
    }

    /// Look up type/doc info for a symbol at a given position.
    pub fn hover_info(&self, uri: &str, line: u32, col: u32) -> Option<String> {
        let (a, sym_id) = self.symbol_at(uri, line, col)?;
        let unit = a.unit.borrow();
        let symbol = analysis::symbol_at_use(
            &unit,
            &a.index,
            self.open_files.get(uri)?,
            line,
            col,
            sym_id,
        )?;
        let mut constructor_signature = None;
        if let Some(source) = self.open_files.get(uri)
            && analysis::is_constructor_name(source, line, col)
            && let Some(constructor) = analysis::constructor_symbol(&unit, symbol.type_id)
                .and_then(|id| unit.symbol_table.get_symbol(id))
        {
            let ty = unit.receiver_member_type(constructor.type_id, symbol.type_id);
            let signature = analysis::format_function_type(
                ty,
                &unit.type_table.borrow(),
                &unit.string_interner,
                &unit.symbol_table,
            );
            constructor_signature = Some(format!(
                "new {}{}",
                unit.string_interner.get(symbol.name)?,
                signature
            ));
        }
        let mut hover = analysis::format_hover_symbol(
            &symbol,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
        )?;
        let qualified = symbol.qualified_name.unwrap_or_else(|| {
            unit.symbol_table
                .build_qualified_name(sym_id, &unit.scope_tree, &unit.string_interner)
        });
        if let Some(path) = unit.file_path_of(symbol.definition_location.file_id) {
            self.workspace_index
                .ensure_disk_file(std::path::Path::new(path));
        } else if let Some(path) = unit.string_interner.get(qualified).and_then(|name| {
            unit.namespace_resolver
                .resolve_qualified_path_to_file_force(name)
        }) {
            self.workspace_index.ensure_disk_file(&path);
        }
        if let Some(declaration) = unit
            .string_interner
            .get(qualified)
            .and_then(|name| self.workspace_index.declaration(name))
            .or_else(|| {
                let text = self.open_files.get(uri)?;
                let (rline, rcol, _) = analysis::receiver_before(text, line, col)?;
                let receiver = a.index.find_symbol_at(rline, rcol)?;
                let receiver_type = a
                    .index
                    .type_at(rline, rcol)
                    .unwrap_or(unit.symbol_table.get_symbol(receiver)?.type_id);
                let owner =
                    crate::structural_fields::alias_owner(receiver_type, &unit).or_else(|| {
                        analysis::class_of_type(
                            receiver_type,
                            &unit.symbol_table,
                            &unit.type_table.borrow(),
                            &unit.string_interner,
                        )
                    })?;
                let path = unit
                    .symbol_table
                    .display_type_path(owner, &unit.string_interner)?;
                if let Some(source) = unit
                    .namespace_resolver
                    .resolve_qualified_path_to_file_force(&path)
                {
                    self.workspace_index.ensure_disk_file(&source);
                }
                let member = unit.string_interner.get(symbol.name)?;
                self.workspace_index
                    .declaration(&format!("{path}.{member}"))
            })
        {
            if let Some(old) = symbol
                .documentation
                .and_then(|doc| unit.string_interner.get(doc))
                && hover.ends_with(old)
            {
                hover.truncate(hover.len() - old.len());
                hover = hover.trim_end().to_string();
            }
            if let Some(doc) = declaration.documentation {
                hover.push_str("\n\n");
                hover.push_str(&doc);
            }
            if (declaration.type_kind.is_some()
                || (symbol.kind == compiler::tast::symbols::SymbolKind::Function
                    && hover.contains("*Signature not yet resolved.*")
                    && declaration.signature.contains("function ")))
                && let Some(end) = hover.find("\n```")
            {
                hover.replace_range("```haxe\n".len()..end, &declaration.signature);
                hover = hover.replace("\n\n*Signature not yet resolved.*", "");
            }
        }
        if let Some(signature) = constructor_signature
            && let Some(end) = hover.find("\n```")
        {
            hover.replace_range("```haxe\n".len()..end, &signature);
        }
        Some(hover)
    }

    /// Find definition location for a symbol at a given position.
    pub fn goto_definition(&self, uri: &str, line: u32, col: u32) -> Option<(String, u32, u32)> {
        let (a, sym_id) = self.symbol_at(uri, line, col)?;
        let unit = a.unit.borrow();
        let symbol = analysis::symbol_at_use(
            &unit,
            &a.index,
            self.open_files.get(uri)?,
            line,
            col,
            sym_id,
        )?;
        let loc = symbol.definition_location;
        if loc.is_valid()
            && let Some(path) = unit.file_path_of(loc.file_id)
        {
            let path = PathBuf::from(path);
            let file = self.editor_path(&path.canonicalize().unwrap_or(path).to_string_lossy());
            return Some((file, loc.line, loc.column));
        }
        let qualified = symbol.qualified_name?;
        let qualified = unit.string_interner.get(qualified)?;
        if let Some(path) = unit
            .namespace_resolver
            .resolve_qualified_path_to_file_force(qualified)
        {
            self.workspace_index.ensure_disk_file(&path);
        }
        let (path, range) = self.workspace_index.definition(qualified)?;
        let file = self.editor_path(&path.canonicalize().unwrap_or(path).to_string_lossy());
        Some((file, range.start.line + 1, range.start.character + 1))
    }

    /// Every place the symbol at a position is named, across the compiled
    /// files: (path, line, column, length), 1-based.
    pub fn references(
        &self,
        uri: &str,
        line: u32,
        col: u32,
        include_declaration: bool,
    ) -> Vec<(String, u32, u32, u32)> {
        let Some((a, sym_id)) = self.symbol_at(uri, line, col) else {
            return Vec::new();
        };
        let unit = a.unit.borrow();
        let mut out = Vec::new();
        let place = |at: compiler::tast::SourceLocation, len: u32| {
            unit.file_path_of(at.file_id)
                .map(|path| (path.to_string(), at.line, at.column, len))
        };
        if include_declaration
            && let Some(sym) = unit.symbol_table.get_symbol(sym_id)
            && sym.definition_location.is_valid()
        {
            let len = unit
                .string_interner
                .get(sym.name)
                .map_or(1, |n| n.len() as u32);
            out.extend(place(sym.definition_location, len));
        }
        for r in unit.symbol_references.iter().filter(|r| r.symbol == sym_id) {
            out.extend(place(r.at, r.len));
        }
        out
    }

    /// Completions at (line, col), 1-based: the members of the receiver
    /// when the cursor follows `receiver.`, otherwise every visible name.
    pub fn completions(&self, uri: &str, line: u32, col: u32) -> Completions {
        let Some(a) = self.analyses.get(uri) else {
            return Completions::Names(Vec::new());
        };
        let unit = a.unit.borrow();
        if let Some(text) = self.open_files.get(uri)
            && let Some((rline, rcol, name)) = analysis::receiver_before(text, line, col)
        {
            let members = a
                .index
                .find_symbol_at(rline, rcol)
                .filter(|&s| a.names(s, &name))
                .or_else(|| {
                    let (st, interner) = (&unit.symbol_table, &unit.string_interner);
                    analysis::resolve_name(&name, a.index.file_id, rline, rcol, st, interner)
                })
                .map(|receiver| {
                    let receiver_type = a
                        .index
                        .type_at(rline, rcol)
                        .unwrap_or_else(|| unit.symbol_table.get_symbol(receiver).unwrap().type_id);
                    let mut members = analysis::member_completions(receiver, receiver_type, &unit);
                    if let Some(owner) = crate::structural_fields::alias_owner(receiver_type, &unit)
                        && let Some(path) = unit
                            .symbol_table
                            .display_type_path(owner, &unit.string_interner)
                    {
                        if let Some(source) = unit
                            .namespace_resolver
                            .resolve_qualified_path_to_file_force(&path)
                        {
                            self.workspace_index.ensure_disk_file(&source);
                        }
                        for member in &mut members {
                            member.documentation = self
                                .workspace_index
                                .declaration(&format!("{path}.{}", member.label))
                                .and_then(|declaration| declaration.documentation);
                        }
                    }
                    if members.is_empty() {
                        analysis::stdlib_member_completions(receiver, &unit)
                    } else {
                        members
                    }
                })
                .unwrap_or_default();
            return Completions::Members(members);
        }
        Completions::Names(analysis::collect_completions(
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            a.index.file_id,
            &unit.symbol_references,
        ))
    }

    /// Build semantic tokens for syntax highlighting.
    pub fn semantic_tokens(&self, uri: &str) -> Option<Vec<analysis::SemanticToken>> {
        let a = self.analyses.get(uri)?;
        let unit = a.unit.borrow();
        Some(analysis::build_semantic_tokens(
            &unit.symbol_table,
            &a.index,
        ))
    }

    /// Build document symbols for the outline panel.
    pub fn document_symbols(&self, uri: &str) -> Option<Vec<analysis::DocumentSymbolEntry>> {
        let a = self.analyses.get(uri)?;
        let unit = a.unit.borrow();
        Some(analysis::build_document_symbols(
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            a.index.file_id,
        ))
    }

    /// The signature of the call whose argument list holds (line, col),
    /// 1-based, with the argument the cursor is in.
    pub fn signature_help(
        &self,
        uri: &str,
        line: u32,
        col: u32,
    ) -> Option<analysis::SignatureInfo> {
        let text = self.open_files.get(uri)?;
        let call = analysis::enclosing_call(text, line, col)?;
        let a = self.analyses.get(uri)?;
        let unit = a.unit.borrow();
        let (st, interner) = (&unit.symbol_table, &unit.string_interner);
        let callee = a
            .index
            .find_symbol_at(call.callee_line, call.callee_col)
            .filter(|&s| a.names(s, &call.callee))
            .or_else(|| {
                let (l, c) = (call.callee_line, call.callee_col);
                analysis::resolve_name(&call.callee, a.index.file_id, l, c, st, interner)
            })?;
        let mut symbol = analysis::symbol_at_use(
            &unit,
            &a.index,
            text,
            call.callee_line,
            call.callee_col,
            callee,
        )?;
        if analysis::is_constructor_name(text, call.callee_line, call.callee_col) {
            let constructor = analysis::constructor_symbol(&unit, symbol.type_id)?;
            let receiver = symbol.type_id;
            symbol = st.get_symbol(constructor)?.clone();
            symbol.type_id = unit.receiver_member_type(symbol.type_id, receiver);
        }
        analysis::signature_for_symbol(
            &symbol,
            &unit.symbol_table,
            &unit.type_table,
            &unit.string_interner,
            call.argument,
        )
    }

    pub fn expand_macro(
        &self,
        uri: &str,
        position: lsp_types::Position,
    ) -> Option<serde_json::Value> {
        let analysis = self.analyses.get(uri)?;
        let source = self.open_files.get(uri)?;
        let unit = analysis.unit.borrow();
        if !unit.source_matches(analysis.index.file_id, source) {
            return None;
        }
        let source = unit.source_of(analysis.index.file_id)?;
        let line_start = source
            .split_inclusive('\n')
            .take(position.line as usize)
            .map(str::len)
            .sum::<usize>();
        let line = source.get(line_start..)?.split('\n').next()?;
        let mut utf16 = 0;
        let column = line
            .char_indices()
            .find_map(|(offset, c)| {
                if utf16 >= position.character {
                    Some(offset)
                } else {
                    utf16 += c.len_utf16() as u32;
                    None
                }
            })
            .unwrap_or(line.len());
        let offset = line_start + column;
        let origin = unit
            .macro_expansions
            .iter()
            .enumerate()
            .filter(|(_, o)| {
                o.call_site.file_id == analysis.index.file_id
                    && o.call_span.start <= offset
                    && offset < o.call_span.end
            })
            .min_by_key(|(index, o)| {
                (
                    o.call_span.end.saturating_sub(o.call_span.start),
                    std::cmp::Reverse(*index),
                )
            })?
            .1;
        Some(macro_expansion_response(origin, source))
    }

    /// The macro expansions in the document, for inlay hints.
    pub fn macro_expansion_hints(&self, uri: &str) -> Vec<MacroExpansionHint> {
        let Some(a) = self.analyses.get(uri) else {
            return Vec::new();
        };
        let unit = a.unit.borrow();
        if !self
            .open_files
            .get(uri)
            .is_some_and(|source| unit.source_matches(a.index.file_id, source))
        {
            return Vec::new();
        }
        unit.macro_expansions
            .iter()
            .filter_map(|origin| {
                let text = origin.expanded_text.as_ref()?;
                let loc = &origin.call_site;
                if !loc.is_valid() || loc.file_id != a.index.file_id {
                    return None;
                }
                let source = unit.source_of(a.index.file_id)?;
                let (line, column) = line_column(source, origin.call_span.end);
                Some(MacroExpansionHint {
                    line,
                    column,
                    macro_name: origin.macro_name.clone(),
                    expanded_text: text.clone(),
                })
            })
            .collect()
    }
}

/// What completion offers at a position.
pub enum Completions {
    /// After `receiver.`: only what the receiver has.
    Members(Vec<CompletionEntry>),
    /// Anywhere else: every name in reach, with keywords.
    Names(Vec<CompletionEntry>),
}

/// A macro expansion hint for display as an inlay hint.
pub struct MacroExpansionHint {
    pub line: u32,
    pub column: u32,
    pub macro_name: String,
    pub expanded_text: String,
}

/// A diagnostic ready for LSP conversion.
#[allow(dead_code)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LspDiagnostic {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub severity: DiagSeverity,
    pub message: String,
    pub suggestion: Option<String>,
    pub related: Vec<String>,
    pub related_information: Vec<lsp_types::DiagnosticRelatedInformation>,
}

impl LspDiagnostic {
    pub fn error(file: &str, line: u32, col: u32, message: String) -> Self {
        Self {
            file: file.to_string(),
            line,
            column: col,
            end_line: line,
            end_column: col + 1,
            severity: DiagSeverity::Error,
            message,
            suggestion: None,
            related: Vec::new(),
            related_information: Vec::new(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum DiagSeverity {
    Error,
    Warning,
    Info,
    Hint,
}

fn macro_expansion_response(
    origin: &compiler::macro_system::expander::ExpansionOrigin,
    source: &str,
) -> serde_json::Value {
    let pos = |at| {
        let (line, character) = line_column(source, at);
        serde_json::json!({"line":line-1,"character":character-1})
    };
    serde_json::json!({
        "name": origin.macro_name,
        "expansion": origin.expanded_text,
        "range": {"start":pos(origin.call_span.start),"end":pos(origin.call_span.end)},
        "definition": origin.definition_file.as_ref().zip(origin.definition_site).and_then(|(file, at)| {
            let (line, character) = if at.line > 0 && at.column > 0 { (at.line, at.column) } else {
                let text = std::fs::read_to_string(file).ok()?;
                line_column(&text, at.byte_offset as usize)
            };
            Some(serde_json::json!({"uri":file_uri(file),"range":{"start":{"line":line-1,"character":character-1},"end":{"line":line-1,"character":character}}}))
        })
    })
}

/// The 1-based (line, column) of byte `offset` in `text`, clamped to its end.
fn line_column(text: &str, offset: usize) -> (u32, u32) {
    let before = &text.as_bytes()[..offset.min(text.len())];
    let line = before.iter().filter(|&&b| b == b'\n').count() as u32 + 1;
    let start = before
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |p| p + 1);
    let column = String::from_utf8_lossy(&before[start..])
        .encode_utf16()
        .count() as u32
        + 1;
    (line, column)
}

pub(crate) fn uri_to_path(uri: &str) -> String {
    if let Some(path) = uri.strip_prefix("file://") {
        path.replace("%20", " ")
    } else {
        uri.to_string()
    }
}

pub(crate) fn file_uri(path: &str) -> String {
    format!(
        "file://{}",
        path.replace('%', "%25")
            .replace(' ', "%20")
            .replace('#', "%23")
            .replace('?', "%3F")
    )
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn diagnostics_use_the_open_documents_path_identity() {
        let workspace = Workspace::new(&[("Main.hx", "class Main {}")]);
        std::fs::create_dir(workspace.0.join("nested")).unwrap();
        let mut context = workspace.context();
        let path = workspace.0.join("Main.hx").to_string_lossy().into_owned();
        let uri = file_uri(&path);
        context
            .open_files
            .insert(uri.clone(), "class Main {}".into());
        let alias = workspace
            .0
            .join("nested/../Main.hx")
            .to_string_lossy()
            .into_owned();
        let published = context.update_diagnostics(
            &uri,
            vec![LspDiagnostic::error(&alias, 1, 7, "example".into())],
        );
        assert_eq!(published.len(), 1);
        assert_eq!(published[0].0, path);
        assert_eq!(published[0].1.len(), 1);
        let cleared = context.update_diagnostics(&uri, Vec::new());
        assert_eq!(cleared.len(), 1);
        assert_eq!(cleared[0].0, path);
        assert!(cleared[0].1.is_empty());
        let config = workspace
            .0
            .join("build.hxml")
            .to_string_lossy()
            .into_owned();
        std::fs::write(&config, "-cp .").unwrap();
        let published = context.update_diagnostics(
            &uri,
            vec![LspDiagnostic::error(&config, 1, 1, "configuration".into())],
        );
        assert!(
            published
                .iter()
                .any(|(path, diagnostics)| path == &config && diagnostics.len() == 1)
        );
    }

    struct Workspace(PathBuf);
    impl Workspace {
        fn new(files: &[(&str, &str)]) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "rayzor-session-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&root).unwrap();
            for (name, source) in files {
                std::fs::write(root.join(name), source).unwrap();
            }
            Self(root)
        }
        fn context(&self) -> LspContext {
            let mut context = LspContext::new(self.0.clone());
            context.class_paths.push(self.0.clone());
            context
        }
        fn compile(&self, context: &mut LspContext, name: &str, source: Option<&str>) -> String {
            let path = self.0.join(name);
            let uri = file_uri(path.to_str().unwrap());
            let source = source
                .map(str::to_string)
                .unwrap_or_else(|| std::fs::read_to_string(path).unwrap());
            context.open_files.insert(uri.clone(), source.clone());
            let errors = context.compile_file(&uri, &source);
            assert!(
                errors.is_empty(),
                "{name}: {:?}",
                errors.iter().map(|d| &d.message).collect::<Vec<_>>()
            );
            uri
        }
    }
    impl Drop for Workspace {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    #[ignore = "requires the sibling RayDE workspace and its libraries"]
    fn rayde_bcencoder_retains_its_imported_bytes_identity() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../RayDE")
                    .canonicalize()
                    .unwrap();
                let mut context = LspContext::new(root.clone());
                context.load_workspace_config(None);
                context.initialization_macros.clear();
                let path = root
                    .join("../ashui/haxe/ashui/draw3d/BcEncoder.hx")
                    .canonicalize()
                    .unwrap();
                let source = std::fs::read_to_string(&path).unwrap();
                let uri = crate::context::file_uri(path.to_str().unwrap());
                context.open_files.insert(uri.clone(), source.clone());
                let diagnostics = context.compile_file(&uri, &source);
                assert!(
                    diagnostics.is_empty(),
                    "{:?}",
                    diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires the sibling RayDE workspace and ashui components"]
    fn rayde_components_resolve_package_types() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                for workspace_root in ["../../RayDE", "../../ashui"] {
                    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join(workspace_root)
                        .canonicalize()
                        .unwrap();
                    let mut context = LspContext::new(root.clone());
                    context.load_workspace_config(None);
                    context.initialization_macros.clear();
                    context.macro_budget = std::time::Duration::from_millis(500);
                    for name in ["Accordion", "ContextMenu", "Library", "Input"] {
                        let path = root
                            .join(format!(
                                "../ashui/components/haxe/ashui/components/{name}.hx"
                            ))
                            .canonicalize()
                            .unwrap();
                        let source = std::fs::read_to_string(&path).unwrap();
                        let uri = file_uri(path.to_str().unwrap());
                        context.open_files.insert(uri.clone(), source.clone());
                        let diagnostics = context.compile_file(&uri, &source);
                        assert!(
                            !diagnostics
                                .iter()
                                .any(|d| d.message == "Cannot find name 'Library'"
                                    || d.message == "Cannot find type 'DropdownMenu'"
                                    || d.message == "Cannot find name 'Size'"),
                            "{name}: missing package type"
                        );
                    }
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires sibling ashui and RayDE projects"]
    fn ashui_dialog_editor_services() {
        std::thread::Builder::new()
            .stack_size(256 << 20)
            .spawn(|| {
                let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../RayDE")
                    .canonicalize()
                    .unwrap();
                let path = root
                    .join("../ashui/components/haxe/ashui/components/Dialog.hx")
                    .canonicalize()
                    .unwrap();
                let source = std::fs::read_to_string(&path).unwrap();
                let uri = file_uri(path.to_str().unwrap());
                let mut context = LspContext::new(root);
                context.load_workspace_config(None);
                context.defines.push("hl".into());
                context.build_options.extra_defines.push("hl".into());
                context.build_options.cache_enabled = false;
                for project in &mut context.projects {
                    project.defines.push("hl".into());
                    project.options.extra_defines.push("hl".into());
                    project.options.cache_enabled = false;
                }
                context.macro_budget = std::time::Duration::from_secs(5);
                context.open_files.insert(uri.clone(), source.clone());
                let diagnostics = context.compile_file(&uri, &source);
                let own: Vec<_> = diagnostics
                    .iter()
                    .filter(|diagnostic| {
                        PathBuf::from(&diagnostic.file).canonicalize().ok().as_ref() == Some(&path)
                    })
                    .collect();
                assert!(
                    own.is_empty(),
                    "{:?}",
                    own.iter()
                        .map(|diagnostic| &diagnostic.message)
                        .collect::<Vec<_>>()
                );
                let at = |needle: &str, name: &str| {
                    let offset = source.find(needle).unwrap() + needle.find(name).unwrap();
                    line_column(&source, offset)
                };
                for (needle, name, expected) in [
                    (
                        "switch props.open",
                        "open",
                        "open:ashui.layout.IntoReactive<Bool>",
                    ),
                    ("registry.add(root", "add", "ashui.components.Dialog"),
                    ("case Const(v)", "v", "v:Bool"),
                    ("new Button({", "Button", "new Button("),
                    (
                        "ashui.ui.Dialog.DialogCancelEvent",
                        "DialogCancelEvent",
                        "DialogCancelEvent",
                    ),
                    ("new ashui.ui.Dialog({", "Dialog", "new Dialog("),
                ] {
                    let (line, col) = at(needle, name);
                    let hover = context.hover_info(&uri, line, col).unwrap_or_default();
                    assert!(hover.contains(expected), "{needle}: {hover}");
                    assert!(!hover.contains("__switch_subject"), "{hover}");
                    if needle.starts_with("new ") {
                        assert!(hover.contains("Props"), "{needle}: {hover}");
                    }
                }
                let (line, col) = at("switch props.open", "open");
                let Completions::Members(members) = context.completions(&uri, line, col) else {
                    panic!("props members missing")
                };
                let labels: Vec<_> = members.iter().map(|member| member.label.as_str()).collect();
                assert!(
                    labels.contains(&"open")
                        && labels.contains(&"onOpenChange")
                        && !labels.contains(&"length"),
                    "{labels:?}"
                );
                let (line, col) = at("d.opened.set(true)", "set");
                let Completions::Members(members) = context.completions(&uri, line, col) else {
                    panic!("signal members missing")
                };
                let get = members.iter().find(|member| member.label == "get").unwrap();
                assert_eq!(get.detail, "() -> Bool");
                let (line, col) = at("return Centered", "Centered");
                let definition = context.goto_definition(&uri, line, col).unwrap();
                assert!(definition.0.ends_with("TopLayer.hx"), "{definition:?}");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires the sibling RayDE workspace and installed Ashui libraries"]
    fn rayde_markup_reaches_the_generated_div_type() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../RayDE")
                    .canonicalize()
                    .unwrap();
                let mut context = LspContext::new(root.clone());
                context.load_workspace_config(None);
                context.initialization_macros.clear();
                let path = root.join("src/MacroDslProbe.hx");
                let source = "@:build(ashui.ui.Markup.build()) class MacroDslProbe { public static function main() { var box = <div/>; var node = box.node; } }";
                let uri = file_uri(path.to_str().unwrap());
                context.open_files.insert(uri.clone(), source.into());
                context.compile_file(&uri, source);
                let column = source.find("box =").unwrap() as u32 + 2;
                let hover = context.hover_info(&uri, 1, column).unwrap_or_default();
                assert!(hover.contains("Div"), "box hover: {hover}");
                let at = source.find("<div/>").unwrap() as u32;
                let hover = context.hover_info(&uri, 1, at + 2).unwrap_or_default();
                assert!(hover.contains("class Div") && hover.contains("A box:"), "{hover}");
                let definition = context.goto_definition(&uri, 1, at + 2).unwrap();
                assert!(definition.0.ends_with("/ashui/ui/Div.hx"), "{definition:?}");
                let expansion = context
                    .expand_macro(&uri, lsp_types::Position::new(0, at))
                    .unwrap();
                assert!(
                    expansion["expansion"].as_str().is_some_and(|text| {
                        text.contains("new") && text.contains("Div") && !text.contains("Hxx.hxx(")
                    }),
                    "{expansion}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    #[ignore = "requires the sibling RayDE workspace and installed Ashui libraries"]
    fn rayde_desktop_menu_tags_resolve_to_component_classes() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../RayDE")
                    .canonicalize()
                    .unwrap();
                let mut context = LspContext::new(root.clone());
                context.load_workspace_config(None);
                let path = root.join("src/rayde/ui/DesktopMenus.hx");
                let source = std::fs::read_to_string(&path).unwrap();
                let uri = file_uri(path.to_str().unwrap());
                context.open_files.insert(uri.clone(), source.clone());
                let _budget = compiler::macro_system::budget::limit(Some(context.macro_budget));
                context.compile_file(&uri, &source);
                for (tag, class) in [
                    ("menubar-menu", "MenubarMenu"),
                    ("menubar-trigger", "MenubarTrigger"),
                    ("menubar-content", "MenubarContent"),
                    ("menubar-separator", "MenubarSeparator"),
                    ("menubar-item", "MenubarItem"),
                ] {
                    let offset = source.find(&format!("<{tag}")).unwrap() + 1;
                    let (line, column) = line_column(&source, offset);
                    let hover = context.hover_info(&uri, line, column).unwrap_or_default();
                    assert!(hover.contains(&format!("class {class}")), "{tag}: {hover}");
                    let definition = context.goto_definition(&uri, line, column).unwrap();
                    assert!(
                        definition.0.ends_with("/ashui/components/Menubar.hx"),
                        "{definition:?}"
                    );
                    let expansion = context
                        .expand_macro(&uri, lsp_types::Position::new(line - 1, column - 1))
                        .unwrap();
                    assert!(
                        expansion["expansion"]
                            .as_str()
                            .is_some_and(|text| text
                                .contains(&format!("new ashui.components.Menubar.{class}"))),
                        "{expansion}"
                    );
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn manifest_resources_reach_analysis_and_invalidate_reused_graphs() {
        let workspace = Workspace::new(&[
            (
                "rayzor.toml",
                "[project]\nname = 'resources'\n[build]\nresources = ['data.txt@settings']\n",
            ),
            ("data.txt", "first"),
            ("Main.hx", "class Main { public var value:Int; }"),
        ]);
        let mut context = workspace.context();
        context.load_workspace_config(None);
        let uri = workspace.compile(&mut context, "Main.hx", None);
        let first = context.analyses[&uri].unit.clone();
        assert_eq!(
            first.borrow().config.resources[0],
            ("settings".into(), b"first".to_vec())
        );
        std::fs::write(workspace.0.join("data.txt"), "updated resource").unwrap();
        workspace.compile(&mut context, "Main.hx", None);
        let second = &context.analyses[&uri].unit;
        assert!(!Rc::ptr_eq(&first, second));
        assert_eq!(
            second.borrow().config.resources[0],
            ("settings".into(), b"updated resource".to_vec())
        );
    }

    #[test]
    fn pages_share_import_types_and_reopening_reuses_the_graph() {
        let workspace = Workspace::new(&[
            (
                "Shared.hx",
                "class Shared { public static function value():Int return 1; }",
            ),
            (
                "First.hx",
                "class First { static function test():Int return Shared.value(); }",
            ),
            (
                "Second.hx",
                "class Second { static function test():Int return Shared.value(); }",
            ),
        ]);
        let mut context = workspace.context();
        let first = workspace.compile(&mut context, "First.hx", None);
        let graph = context.analyses[&first].unit.clone();
        let shared = graph
            .borrow()
            .file_id_of(workspace.0.join("Shared.hx").to_str().unwrap())
            .unwrap();
        let second = workspace.compile(&mut context, "Second.hx", None);
        assert!(Rc::ptr_eq(&graph, &context.analyses[&second].unit));
        assert_eq!(context.sessions.len(), 1);
        assert_eq!(
            graph
                .borrow()
                .file_id_of(workspace.0.join("Shared.hx").to_str().unwrap()),
            Some(shared)
        );
        let before = (
            graph.borrow().symbol_table.len(),
            graph.borrow().symbol_references.len(),
        );
        context.close(&first);
        let first = workspace.compile(&mut context, "First.hx", None);
        assert!(Rc::ptr_eq(&graph, &context.analyses[&first].unit));
        assert!(context.hover_info(&first, 1, 49).is_some());
        assert_eq!(
            before,
            (
                graph.borrow().symbol_table.len(),
                graph.borrow().symbol_references.len()
            )
        );
    }

    #[test]
    fn changed_dependencies_and_defines_invalidate_the_graph() {
        let workspace = Workspace::new(&[
            (
                "Shared.hx",
                "class Shared { public static function value():Int return 1; }",
            ),
            (
                "First.hx",
                "class First { static function test():Dynamic return Shared.value(); }",
            ),
            (
                "Second.hx",
                "class Second { static function test():Dynamic return Shared.value(); }",
            ),
        ]);
        let mut context = workspace.context();
        let first = workspace.compile(&mut context, "First.hx", None);
        let old = Rc::downgrade(&context.analyses[&first].unit);
        // Use an unsaved dependency: imports must see its current buffer.
        let shared_uri = file_uri(workspace.0.join("Shared.hx").to_str().unwrap());
        context.open_files.insert(
            shared_uri,
            "class Shared { public static function value():String return 'updated'; }".into(),
        );
        let second = workspace.compile(&mut context, "Second.hx", None);
        assert!(
            old.upgrade().is_none(),
            "obsolete generations must be released"
        );
        let unit = context.analyses[&second].unit.borrow();
        let source = unit
            .source_of(
                unit.file_id_of(workspace.0.join("Shared.hx").to_str().unwrap())
                    .unwrap(),
            )
            .unwrap();
        assert!(source.contains("updated"), "{source}");
        drop(unit);
        let before_define = context.analyses[&second].unit.clone();
        context.defines.push("new_feature".into());
        workspace.compile(&mut context, "Second.hx", None);
        assert!(!Rc::ptr_eq(&before_define, &context.analyses[&second].unit));
        drop(before_define);
        assert!(
            old.upgrade().is_none(),
            "old graph must be released at capacity"
        );
    }

    #[test]
    fn cached_graphs_are_bounded_and_evicted_pages_are_reanalyzed() {
        let workspace = Workspace::new(&[(
            "Main.hx",
            "class Main { static function test():Int return 1; }",
        )]);
        let mut context = workspace.context();
        context.analysis_cache_capacity = 1;
        let uri = workspace.compile(&mut context, "Main.hx", None);
        let first = Rc::downgrade(&context.analyses[&uri].unit);
        for version in 2..8 {
            let text = format!("class Main {{ static function test():Int return {version}; }}");
            workspace.compile(&mut context, "Main.hx", Some(&text));
            assert_eq!(context.sessions.len(), 1);
            assert_eq!(context.analyses.len(), 1);
        }
        assert!(first.upgrade().is_none());
        context.invalidate_analysis_cache();
        assert!(context.needs_analysis(&uri));
        let source = context.open_files[&uri].clone();
        assert!(context.compile_file(&uri, &source).is_empty());
        assert!(!context.needs_analysis(&uri));
        let unit = context.analyses[&uri].unit.borrow();
        assert!(
            unit.source_of(context.analyses[&uri].index.file_id)
                .unwrap()
                .contains("return 7")
        );
    }

    #[test]
    fn closing_the_last_page_releases_its_compiler_graph() {
        let workspace = Workspace::new(&[(
            "Main.hx",
            "class Main { static function test():Int return 1; }",
        )]);
        let mut context = workspace.context();
        let uri = workspace.compile(&mut context, "Main.hx", None);
        let graph = Rc::downgrade(&context.analyses[&uri].unit);
        context.close(&uri);
        assert!(graph.upgrade().is_none());
        assert!(context.sessions.is_empty());
    }
    #[test]
    fn opening_an_import_through_its_canonical_path_reuses_its_source_identity() {
        let workspace = Workspace::new(&[
            (
                "Shared.hx",
                "class Shared { public static function value():Int return 1; }",
            ),
            (
                "Main.hx",
                "class Main { static function main() trace(Shared.value()); }",
            ),
        ]);
        let mut context = workspace.context();
        context.class_paths = vec![workspace.0.join(".")];
        let main = workspace.compile(&mut context, "Main.hx", None);
        let graph = context.analyses[&main].unit.clone();
        let compiled = graph.borrow().typecheck_timings().files_seen;
        let roots = graph.borrow().user_files.len();
        let shared = workspace.compile(&mut context, "Shared.hx", None);
        assert!(Rc::ptr_eq(&graph, &context.analyses[&shared].unit));
        assert_eq!(graph.borrow().user_files.len(), roots);
        assert_eq!(graph.borrow().typecheck_timings().files_seen, compiled);
        assert_ne!(context.analyses[&shared].index.file_id, u32::MAX);
        let hover = context.hover_info(&shared, 1, 8).unwrap();
        assert!(hover.contains("class Shared"), "{hover}");
        assert!(!context.document_symbols(&shared).unwrap().is_empty());
    }

    #[test]
    fn typed_dependencies_remain_available_after_the_graph_reaches_its_page_limit() {
        let workspace = Workspace::new(&[
            (
                "Shared.hx",
                "class Shared { public static var value:Int = 1; }",
            ),
            (
                "Main.hx",
                "class Main { static function main() trace(Shared.value); }",
            ),
        ]);
        let mut context = workspace.context();
        let main = workspace.compile(&mut context, "Main.hx", None);
        let graph = context.analyses[&main].unit.clone();
        {
            let mut unit = graph.borrow_mut();
            let root = unit.user_files[0].clone();
            unit.user_files.resize(32, root);
        }
        let shared = workspace.compile(&mut context, "Shared.hx", None);
        assert!(Rc::ptr_eq(&graph, &context.analyses[&shared].unit));
        assert_eq!(context.sessions.len(), 1);
        assert!(
            context
                .hover_info(&shared, 1, 35)
                .unwrap()
                .contains("value:Int")
        );
    }
}
