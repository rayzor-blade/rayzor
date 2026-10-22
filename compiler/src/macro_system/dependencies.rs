//! Source and resource reads used to validate editor graphs and macro results.

use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dependency {
    pub path: PathBuf,
    stamp: Option<(u64, u128)>,
}

impl Dependency {
    fn capture(path: &Path) -> Self {
        let stamp = std::fs::metadata(path).ok().and_then(|metadata| {
            let modified = metadata
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?;
            Some((metadata.len(), modified.as_nanos()))
        });
        Self {
            path: path.into(),
            stamp,
        }
    }

    pub fn is_current(&self) -> bool {
        *self == Self::capture(&self.path)
    }
}

#[derive(Clone, Default)]
pub struct Dependencies(Arc<Mutex<BTreeMap<String, BTreeMap<PathBuf, Dependency>>>>);

const SOURCE_INPUTS: &str = "<source>";

impl Dependencies {
    pub fn register(&self, module: String, path: PathBuf) {
        let path = path.canonicalize().unwrap_or(path);
        let dependency = Dependency::capture(&path);
        self.0
            .lock()
            .unwrap()
            .entry(module)
            .or_default()
            .insert(path, dependency);
    }

    /// Keep the first read's stamp so a change during analysis rejects its result.
    pub fn register_read(&self, path: impl AsRef<Path>) {
        self.capture_read("", path.as_ref());
    }

    pub fn register_source(&self, path: impl AsRef<Path>) {
        self.capture_read(SOURCE_INPUTS, path.as_ref());
    }

    fn capture_read(&self, group: &str, path: &Path) {
        let path = dependency_path(path);
        self.0
            .lock()
            .unwrap()
            .entry(group.into())
            .or_default()
            .entry(path.clone())
            .or_insert_with(|| Dependency::capture(&path));
    }

    /// A macro's own output must not invalidate the graph it just produced.
    pub fn refresh_written(&self, path: impl AsRef<Path>) {
        let path = dependency_path(path.as_ref());
        let parent = path.parent();
        for (group, files) in self.0.lock().unwrap().iter_mut() {
            if group == SOURCE_INPUTS {
                continue;
            }
            for dependency in files.values_mut() {
                if dependency.path == path || Some(dependency.path.as_path()) == parent {
                    *dependency = Dependency::capture(&dependency.path);
                }
            }
        }
    }

    pub fn snapshot(&self) -> Vec<Dependency> {
        self.0
            .lock()
            .unwrap()
            .values()
            .flat_map(|files| files.values().cloned())
            .collect()
    }

    pub fn is_current(&self) -> bool {
        self.0
            .lock()
            .unwrap()
            .values()
            .all(|files| files.values().all(Dependency::is_current))
    }
}

fn dependency_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| {
        let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
        match (absolute.parent(), absolute.file_name()) {
            (Some(parent), Some(name)) => parent
                .canonicalize()
                .map(|parent| parent.join(name))
                .unwrap_or(absolute),
            _ => absolute,
        }
    })
}
