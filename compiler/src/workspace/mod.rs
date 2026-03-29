//! Workspace and project management for Rayzor.
//!
//! Provides cargo-like workspace support with `rayzor.toml` manifests,
//! multi-project workspaces, shared BLADE caches, and backwards
//! compatibility with `.hxml` build files.

pub mod init;
pub mod manifest;

use std::path::{Path, PathBuf};

pub use manifest::{
    BuildConfig, BundleConfig as ManifestBundleConfig, CacheConfig, ProjectManifest,
    RayzorManifest, WorkspaceCacheConfig, WorkspaceManifest,
};

/// A resolved workspace (may contain multiple projects).
#[derive(Debug)]
pub struct Workspace {
    /// Root directory containing the workspace rayzor.toml
    pub root: PathBuf,
    /// Member projects
    pub members: Vec<Project>,
    /// Shared cache configuration
    pub cache: CacheConfig,
}

/// A resolved single project.
#[derive(Debug)]
pub struct Project {
    /// Project root directory (contains rayzor.toml)
    pub root: PathBuf,
    /// Parsed manifest
    pub manifest: ProjectManifest,
}

impl Project {
    /// Resolve the entry file path relative to project root.
    pub fn entry_path(&self) -> Option<PathBuf> {
        self.manifest.entry.as_ref().map(|e| self.root.join(e))
    }

    /// Resolve class paths relative to project root.
    pub fn resolved_class_paths(&self) -> Vec<PathBuf> {
        self.manifest
            .build
            .as_ref()
            .map(|b| b.class_paths.iter().map(|cp| self.root.join(cp)).collect())
            .unwrap_or_default()
    }

    /// Resolve output path relative to project root.
    pub fn output_path(&self) -> Option<PathBuf> {
        self.manifest
            .build
            .as_ref()
            .and_then(|b| b.output.as_ref())
            .map(|o| self.root.join(o))
    }

    /// Resolve WASM JS host module paths from [wasm] config.
    /// Returns module_name → absolute path to .js file.
    pub fn resolved_wasm_hosts(&self) -> std::collections::HashMap<String, PathBuf> {
        self.manifest
            .wasm
            .as_ref()
            .map(|w| {
                w.hosts
                    .iter()
                    .map(|(name, path)| (name.clone(), self.root.join(path)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the BLADE cache directory for this project.
    pub fn cache_dir(&self) -> PathBuf {
        if let Some(ref cache) = self.manifest.cache {
            if let Some(ref dir) = cache.dir {
                return self.root.join(dir);
            }
        }
        self.root.join(".rayzor").join("cache")
    }

    /// Whether caching is enabled.
    pub fn cache_enabled(&self) -> bool {
        self.manifest
            .cache
            .as_ref()
            .map(|c| c.enabled.unwrap_or(true))
            .unwrap_or(true)
    }

    /// Get the MIR optimization level (0-3, default 2).
    pub fn opt_level(&self) -> u8 {
        self.manifest
            .build
            .as_ref()
            .and_then(|b| b.opt_level)
            .unwrap_or(2)
    }

    /// Get the JIT preset name (default "application").
    pub fn preset(&self) -> &str {
        self.manifest
            .build
            .as_ref()
            .and_then(|b| b.preset.as_deref())
            .unwrap_or("application")
    }

    /// Get defines as (key, optional value) pairs.
    pub fn defines(&self) -> Vec<(String, Option<String>)> {
        self.manifest
            .build
            .as_ref()
            .and_then(|b| b.defines.as_ref())
            .map(|defs| {
                defs.iter()
                    .map(|(k, v)| {
                        let val = match v {
                            toml::Value::Boolean(true) => None,
                            toml::Value::String(s) => Some(s.clone()),
                            other => Some(other.to_string()),
                        };
                        (k.clone(), val)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// The manifest file name.
pub const MANIFEST_FILE: &str = "rayzor.toml";

/// Find the project/workspace root by walking up from `start_dir`.
///
/// Returns the directory containing `rayzor.toml`, or None.
pub fn find_project_root(start_dir: &Path) -> Option<PathBuf> {
    let mut current = start_dir.to_path_buf();
    loop {
        if current.join(MANIFEST_FILE).exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Load and resolve a manifest from the given directory.
///
/// If `path` points to a file, uses that file. If it points to a directory,
/// looks for `rayzor.toml` inside it.
pub fn load_manifest(path: &Path) -> Result<RayzorManifest, String> {
    let manifest_path = if path.is_file() {
        path.to_path_buf()
    } else {
        path.join(MANIFEST_FILE)
    };

    if !manifest_path.exists() {
        return Err(format!(
            "No {} found at {}",
            MANIFEST_FILE,
            manifest_path.display()
        ));
    }

    let content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Failed to read {}: {}", manifest_path.display(), e))?;

    manifest::parse_manifest(&content)
}

/// Load a project from a directory containing rayzor.toml.
pub fn load_project(dir: &Path) -> Result<Project, String> {
    let manifest = load_manifest(dir)?;
    match manifest {
        RayzorManifest::SingleProject(pm) => Ok(Project {
            root: dir.to_path_buf(),
            manifest: pm,
        }),
        RayzorManifest::Workspace(_) => {
            Err("Expected a project manifest, found a workspace manifest".to_string())
        }
    }
}

/// Load a workspace from a directory containing a workspace rayzor.toml.
pub fn load_workspace(dir: &Path) -> Result<Workspace, String> {
    let manifest = load_manifest(dir)?;
    match manifest {
        RayzorManifest::Workspace(wm) => {
            let mut members = Vec::new();
            for member_path in &wm.members {
                let member_dir = dir.join(member_path);
                match load_project(&member_dir) {
                    Ok(project) => members.push(project),
                    Err(e) => {
                        eprintln!("Warning: Failed to load member '{}': {}", member_path, e);
                    }
                }
            }

            let cache = wm
                .cache
                .as_ref()
                .map(|wc| CacheConfig {
                    dir: wc.dir.clone(),
                    enabled: Some(true),
                })
                .unwrap_or_else(|| CacheConfig {
                    dir: Some(".rayzor/cache".to_string()),
                    enabled: Some(true),
                });

            Ok(Workspace {
                root: dir.to_path_buf(),
                members,
                cache,
            })
        }
        RayzorManifest::SingleProject(_) => {
            Err("Expected a workspace manifest, found a project manifest".to_string())
        }
    }
}

/// Auto-detect whether a directory is a workspace or single project, and load accordingly.
pub fn load_auto(dir: &Path) -> Result<LoadedConfig, String> {
    let manifest = load_manifest(dir)?;
    match manifest {
        RayzorManifest::Workspace(wm) => {
            let ws = load_workspace(dir)?;
            Ok(LoadedConfig::Workspace(ws))
        }
        RayzorManifest::SingleProject(pm) => {
            let project = Project {
                root: dir.to_path_buf(),
                manifest: pm,
            };
            Ok(LoadedConfig::Project(project))
        }
    }
}

/// Result of auto-detecting and loading config.
#[derive(Debug)]
pub enum LoadedConfig {
    Workspace(Workspace),
    Project(Project),
}
