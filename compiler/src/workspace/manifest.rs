//! TOML manifest parsing for `rayzor.toml`.

use serde::Deserialize;
use std::collections::HashMap;

/// Top-level manifest — either a single project or a workspace.
#[derive(Debug)]
pub enum RayzorManifest {
    SingleProject(ProjectManifest),
    Workspace(WorkspaceManifest),
}

/// The raw TOML structure (used for initial deserialization to detect type).
#[derive(Debug, Deserialize)]
struct RawManifest {
    project: Option<ProjectManifest>,
    workspace: Option<WorkspaceSection>,
    build: Option<BuildConfig>,
    cache: Option<CacheConfig>,
    bundle: Option<BundleConfig>,
    dependencies: Option<HashMap<String, DependencySpec>>,
}

/// A single dependency specification in `[dependencies]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    /// Simple version string: `mylib = "1.0"`
    Version(String),
    /// Table with fields: `mylib = { path = "../mylib" }`
    Table {
        /// Local path to the package directory or .rpkg file.
        path: Option<String>,
        /// Package name in the rpkg registry.
        rpkg: Option<String>,
        /// Git repository URL.
        git: Option<String>,
        /// Git branch (used with `git`).
        branch: Option<String>,
        /// Version constraint.
        version: Option<String>,
    },
}

#[derive(Debug, Deserialize)]
struct WorkspaceSection {
    members: Vec<String>,
    cache: Option<WorkspaceCacheConfig>,
}

/// Project manifest fields from `[project]`.
#[derive(Debug, Clone, Deserialize)]
pub struct ProjectManifest {
    /// Project name
    pub name: Option<String>,
    /// Project version
    pub version: Option<String>,
    /// Entry point file (e.g. "src/Main.hx")
    pub entry: Option<String>,
    /// Delegate to an HXML file for all config
    pub hxml: Option<String>,
    /// Build configuration
    #[serde(skip)]
    pub build: Option<BuildConfig>,
    /// Cache configuration
    #[serde(skip)]
    pub cache: Option<CacheConfig>,
    /// Bundle configuration
    #[serde(skip)]
    pub bundle: Option<BundleConfig>,
    /// Dependencies
    #[serde(skip)]
    pub dependencies: Option<HashMap<String, DependencySpec>>,
}

/// Workspace manifest fields.
#[derive(Debug, Clone)]
pub struct WorkspaceManifest {
    /// Member project directories (relative paths)
    pub members: Vec<String>,
    /// Shared cache configuration
    pub cache: Option<WorkspaceCacheConfig>,
}

/// `[build]` section.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct BuildConfig {
    /// Class paths (-cp equivalent)
    #[serde(default)]
    pub class_paths: Vec<String>,
    /// MIR optimization level (0-3)
    pub opt_level: Option<u8>,
    /// JIT preset name
    pub preset: Option<String>,
    /// Build target: "native", "jit", "bundle"
    pub target: Option<String>,
    /// Output path
    pub output: Option<String>,
    /// Defines (-D equivalent)
    pub defines: Option<HashMap<String, toml::Value>>,
}

/// `[cache]` section.
#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    /// Cache directory (relative to project root)
    pub dir: Option<String>,
    /// Whether caching is enabled
    pub enabled: Option<bool>,
}

/// `[bundle]` section.
#[derive(Debug, Clone, Deserialize)]
pub struct BundleConfig {
    /// Enable zstd compression
    pub compress: Option<bool>,
    /// Enable tree-shaking
    pub strip: Option<bool>,
}

/// `[workspace.cache]` section.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceCacheConfig {
    /// Shared cache directory
    pub dir: Option<String>,
}

/// Parse a `rayzor.toml` string into a `RayzorManifest`.
pub fn parse_manifest(content: &str) -> Result<RayzorManifest, String> {
    let raw: RawManifest =
        toml::from_str(content).map_err(|e| format!("Failed to parse rayzor.toml: {}", e))?;

    // Workspace takes precedence
    if let Some(ws) = raw.workspace {
        return Ok(RayzorManifest::Workspace(WorkspaceManifest {
            members: ws.members,
            cache: ws.cache,
        }));
    }

    // Single project
    if let Some(mut project) = raw.project {
        project.build = raw.build;
        project.cache = raw.cache;
        project.bundle = raw.bundle;
        project.dependencies = raw.dependencies;
        return Ok(RayzorManifest::SingleProject(project));
    }

    Err("rayzor.toml must contain either [project] or [workspace]".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_project() {
        let toml = r#"
[project]
name = "hello"
version = "0.1.0"
entry = "src/Main.hx"

[build]
class-paths = ["src"]
opt-level = 2

[cache]
enabled = true
"#;
        let manifest = parse_manifest(toml).unwrap();
        match manifest {
            RayzorManifest::SingleProject(p) => {
                assert_eq!(p.name.as_deref(), Some("hello"));
                assert_eq!(p.entry.as_deref(), Some("src/Main.hx"));
                assert_eq!(p.build.as_ref().unwrap().opt_level, Some(2));
            }
            _ => panic!("Expected SingleProject"),
        }
    }

    #[test]
    fn test_parse_workspace() {
        let toml = r#"
[workspace]
members = ["game", "engine"]

[workspace.cache]
dir = ".rayzor/cache"
"#;
        let manifest = parse_manifest(toml).unwrap();
        match manifest {
            RayzorManifest::Workspace(w) => {
                assert_eq!(w.members, vec!["game", "engine"]);
                assert_eq!(
                    w.cache.as_ref().unwrap().dir.as_deref(),
                    Some(".rayzor/cache")
                );
            }
            _ => panic!("Expected Workspace"),
        }
    }

    #[test]
    fn test_parse_hxml_delegation() {
        let toml = r#"
[project]
name = "legacy"
hxml = "build.hxml"
"#;
        let manifest = parse_manifest(toml).unwrap();
        match manifest {
            RayzorManifest::SingleProject(p) => {
                assert_eq!(p.hxml.as_deref(), Some("build.hxml"));
            }
            _ => panic!("Expected SingleProject"),
        }
    }
}
