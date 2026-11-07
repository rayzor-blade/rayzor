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
    /// Size and modification time in whole seconds, which an image export or archive unpack
    /// keeps where it truncates the nanoseconds.
    configuration: BTreeMap<PathBuf, Option<(u64, u64)>>,
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
                        std::fs::metadata(path).ok().and_then(|metadata| {
                            let modified = metadata
                                .modified()
                                .ok()?
                                .duration_since(std::time::UNIX_EPOCH)
                                .ok()?;
                            Some((metadata.len(), modified.as_secs()))
                        }),
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
                match std::fs::read_dir(&path) {
                    Ok(entries) => {
                        for entry in entries {
                            let Ok(entry) = entry else {
                                inputs.unreusable(&path);
                                continue;
                            };
                            let Ok(kind) = entry.file_type() else {
                                inputs.unreusable(&path);
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
                    Err(_) => inputs.unreusable(&path),
                }
            }
        }
        for (uri, source) in open_files {
            inputs.set_source(&crate::context::uri_to_path(uri), source);
        }
        inputs
    }

    /// A directory that cannot be read leaves what a graph or index depends on unknown.
    fn unreusable(&mut self, path: &Path) {
        if std::env::var_os("RAYZOR_LSP_PROFILE").is_some() {
            eprintln!(
                "lsp-index: inputs not reusable, cannot read {}",
                path.display()
            );
        }
        self.reusable = false;
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
    /// Passes a display compile has made over its document in this graph.
    pub passes: usize,
    /// A display pass began and did not finish, which leaves the graph half-typed.
    pub interrupted: bool,
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
            passes: 0,
            interrupted: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_stamps_ignore_the_nanoseconds_of_a_timestamp() {
        let path =
            std::env::temp_dir().join(format!("rayzor-inputs-seconds-{}", std::process::id()));
        std::fs::write(&path, "config").unwrap();
        let set = |time: std::time::SystemTime| {
            std::fs::File::options()
                .write(true)
                .open(&path)
                .unwrap()
                .set_modified(time)
                .unwrap()
        };
        let key = SessionKey {
            class_paths: Vec::new(),
            stdlib_paths: Vec::new(),
            defines: Vec::new(),
            macros: Vec::new(),
            macro_budget: None,
            options: crate::build_config::Options {
                configuration_files: vec![path.clone()],
                ..Default::default()
            },
        };
        let base = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        set(base + std::time::Duration::from_nanos(123_456_789));
        let built = Inputs::capture(&key, &HashMap::new());
        set(base);
        assert!(built == Inputs::capture(&key, &HashMap::new()));
        set(base + std::time::Duration::from_secs(1));
        assert!(built != Inputs::capture(&key, &HashMap::new()));
        std::fs::remove_file(path).unwrap();
    }
}
