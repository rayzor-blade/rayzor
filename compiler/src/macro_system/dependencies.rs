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
    content_digest: Option<[u8; 16]>,
    /// A Haxe source the unit parsed, rather than a file a macro read.
    #[serde(default)]
    source: bool,
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
            content_digest: None,
            source: false,
        }
    }

    pub fn is_source(&self) -> bool {
        self.source
    }

    pub fn is_current(&self) -> bool {
        let current = Self::capture(&self.path);
        if self.stamp == current.stamp {
            return true;
        }
        let Some(expected) = self.content_digest else {
            return false;
        };
        if self.stamp.map(|(length, _)| length) != current.stamp.map(|(length, _)| length) {
            return false;
        }
        use md5::{Digest, Md5};
        use std::io::Read;
        let Ok(mut file) = std::fs::File::open(&self.path) else {
            return false;
        };
        let mut digest = Md5::new();
        let mut buffer = [0; 8192];
        loop {
            let Ok(length) = file.read(&mut buffer) else {
                return false;
            };
            if length == 0 {
                break;
            }
            digest.update(&buffer[..length]);
        }
        <[u8; 16]>::from(digest.finalize()) == expected
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

    /// Preserve the first bytes read, so rewriting identical generated output stays reusable.
    pub fn register_content(&self, path: impl AsRef<Path>, bytes: &[u8]) {
        use md5::{Digest, Md5};
        let path = dependency_path(path.as_ref());
        let mut groups = self.0.lock().unwrap();
        let entry = groups
            .entry(String::new())
            .or_default()
            .entry(path.clone())
            .or_insert_with(|| Dependency::capture(&path));
        entry
            .content_digest
            .get_or_insert_with(|| Md5::digest(bytes).into());
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
            .or_insert_with(|| Dependency {
                source: group == SOURCE_INPUTS,
                ..Dependency::capture(&path)
            });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_file_rewrites_keep_identical_content_current() {
        let path =
            std::env::temp_dir().join(format!("rayzor-macro-dependency-{}", std::process::id()));
        std::fs::write(&path, b"first").unwrap();
        let dependencies = Dependencies::default();
        dependencies.register_read(&path);
        dependencies.register_content(&path, b"first");
        let next = std::fs::metadata(&path).unwrap().modified().unwrap()
            + std::time::Duration::from_secs(1);
        std::fs::write(&path, b"first").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(next)
            .unwrap();
        assert!(dependencies.is_current());
        let snapshot = dependencies.snapshot();
        assert!(snapshot.iter().all(Dependency::is_current));
        std::fs::write(&path, b"other").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(next + std::time::Duration::from_secs(1))
            .unwrap();
        dependencies.register_content(&path, b"other");
        assert!(!dependencies.is_current());
        assert!(!snapshot.iter().all(Dependency::is_current));
        std::fs::remove_file(path).unwrap();
    }
}
