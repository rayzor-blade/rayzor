//! Files and directories read by macros, including resources outside class paths.

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
