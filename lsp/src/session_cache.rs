//! Compatible pages share compiler-local identities in one typed graph.
//! Graphs are never merged: a changed source/configuration starts a new one.

use crate::context::LspDiagnostic;
use compiler::compilation::CompilationUnit;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Duration;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SessionKey {
    pub class_paths: Vec<PathBuf>,
    pub stdlib_paths: Vec<PathBuf>,
    pub defines: Vec<String>,
    pub macros: Vec<String>,
    pub macro_budget: Option<Duration>,
    pub options: crate::build_config::Options,
}

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct Inputs {
    // Discovering/removing a module invalidates negative import lookups.
    // Contents of loaded sources and macro resources have individual stamps.
    files: BTreeSet<PathBuf>,
    overlays: BTreeMap<PathBuf, String>,
    configuration: BTreeMap<PathBuf, Option<(u64, std::time::SystemTime)>>,
    pub reusable: bool,
}

impl Inputs {
    pub fn capture(key: &SessionKey, open_files: &HashMap<String, String>) -> Self {
        let mut inputs = Self {
            files: BTreeSet::new(),
            overlays: BTreeMap::new(),
            configuration: key
                .options
                .configuration_files
                .iter()
                .chain(&key.options.native_libs)
                .chain(&key.options.packages)
                .chain(key.options.resources.iter().map(|(path, _)| path))
                .map(|path| {
                    (
                        path.clone(),
                        std::fs::metadata(path)
                            .ok()
                            .and_then(|metadata| Some((metadata.len(), metadata.modified().ok()?))),
                    )
                })
                .collect(),
            reusable: true,
        };
        let mut visited = BTreeSet::new();
        let mut pending: Vec<_> = key
            .class_paths
            .iter()
            .chain(&key.stdlib_paths)
            .cloned()
            .collect();
        while let Some(path) = pending.pop() {
            let is_source = path.extension().is_some_and(|extension| extension == "hx");
            let path = canonical(&path);
            if !visited.insert(path.clone()) {
                continue;
            }
            let Ok(metadata) = std::fs::metadata(&path) else {
                // A nonexistent optional class path is represented by its
                // absence; if it appears, the next capture differs.
                continue;
            };
            if !metadata.is_dir() && is_source {
                inputs.files.insert(path.clone());
            }
            if metadata.is_dir() {
                match std::fs::read_dir(path) {
                    Ok(entries) => {
                        for entry in entries {
                            let Ok(entry) = entry else {
                                inputs.reusable = false;
                                continue;
                            };
                            let Ok(kind) = entry.file_type() else {
                                inputs.reusable = false;
                                continue;
                            };
                            let name = entry.file_name();
                            let name = name.to_string_lossy();
                            if kind.is_dir()
                                && (name.starts_with('.')
                                    || matches!(
                                        name.as_ref(),
                                        "target"
                                            | "node_modules"
                                            | "bin"
                                            | "out"
                                            | "dist"
                                            | "vendor"
                                    ))
                            {
                                continue;
                            }
                            if kind.is_dir()
                                || kind.is_symlink()
                                || entry
                                    .path()
                                    .extension()
                                    .is_some_and(|extension| extension == "hx")
                            {
                                pending.push(entry.path());
                            }
                        }
                    }
                    Err(_) => inputs.reusable = false,
                }
            }
        }
        for (uri, source) in open_files {
            inputs.set_source(&crate::context::uri_to_path(uri), source);
        }
        inputs
    }

    pub fn set_source(&mut self, filename: &str, source: &str) {
        let path = canonical(Path::new(filename));
        if std::fs::read_to_string(&path).ok().as_deref() != Some(source) {
            self.overlays.insert(path, source.to_string());
        }
    }
}

pub(crate) struct SessionCache {
    pub key: SessionKey,
    pub inputs: Inputs,
    pub unit: Rc<RefCell<CompilationUnit>>,
    pub initialization_errors: Vec<LspDiagnostic>,
    pub diagnostics: Vec<LspDiagnostic>,
    pub native_libraries: Vec<compiler::native_libs::Library>,
    pub indexes: HashMap<String, crate::analysis::FileSymbolIndex>,
}

impl SessionCache {
    pub fn new(
        key: SessionKey,
        inputs: Inputs,
        unit: CompilationUnit,
        initialization_errors: Vec<LspDiagnostic>,
    ) -> Self {
        Self {
            key,
            inputs,
            unit: Rc::new(RefCell::new(unit)),
            initialization_errors,
            diagnostics: Vec::new(),
            native_libraries: Vec::new(),
            indexes: HashMap::new(),
        }
    }

    pub fn can_extend(&self, filename: &str) -> bool {
        let unit = self.unit.borrow();
        if !unit.macro_dependencies.is_current() {
            return false;
        }
        // One graph can contain a large dependency closure; bound how many
        // unrelated pages may keep extending it before starting a new graph.
        unit.has_typed_file(filename)
            || unit.user_files.iter().any(|file| file.filename == filename)
            || (unit.user_files.len() < 32 && unit.symbol_table.len() < 100_000)
    }
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}
