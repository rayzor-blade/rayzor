//! Compatible pages share compiler-local identities in one typed graph.
//! Graphs are never merged: a changed source/configuration starts a new one.

use crate::context::LspDiagnostic;
use compiler::compilation::CompilationUnit;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, SystemTime};

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SessionKey {
    pub class_paths: Vec<PathBuf>,
    pub stdlib_paths: Vec<PathBuf>,
    pub defines: Vec<String>,
    pub macros: Vec<String>,
    pub macro_budget: Duration,
}

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct Inputs {
    // Include directory entries too: creating a formerly missing dependency
    // invalidates negative lookups. Resources can be read by build macros.
    files: BTreeMap<PathBuf, (u64, Option<SystemTime>)>,
    overlays: BTreeMap<PathBuf, String>,
    pub reusable: bool,
}

impl Inputs {
    pub fn capture(key: &SessionKey, open_files: &HashMap<String, String>) -> Self {
        let mut inputs = Self {
            files: BTreeMap::new(),
            overlays: BTreeMap::new(),
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
            let path = canonical(&path);
            if !visited.insert(path.clone()) {
                continue;
            }
            let Ok(metadata) = std::fs::metadata(&path) else {
                // A nonexistent optional class path is represented by its
                // absence; if it appears, the next capture differs.
                continue;
            };
            inputs.files.insert(
                path.clone(),
                if metadata.is_dir() {
                    (0, None)
                } else {
                    (metadata.len(), metadata.modified().ok())
                },
            );
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
                            pending.push(entry.path());
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
