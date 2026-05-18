//! Manifest dependency resolution to `.rpkg` file paths.
//!
//! Reads `[dependencies]` from a project manifest and resolves each entry to
//! an absolute path to a `.rpkg` archive. The compiler then dlopens each
//! resolved rpkg via [`crate::rpkg::install::RpkgPlugin::load`] and the
//! plugin's `declare_native_methods!` table auto-registers itself — no
//! `--rpkg` flag, no compiler core changes per plugin.
//!
//! Supported `[dependencies]` shapes:
//!
//! ```toml
//! [dependencies]
//! # All four of these resolve to the same rpkg. The `.rpkg` extension is
//! # inferred when the path points to a directory or a name without the
//! # extension — callers shouldn't have to spell it out.
//! my-lib = { path = "../my-lib/my-lib.rpkg" }   # explicit file
//! my-lib = { path = "../my-lib" }                # dir → looks for my-lib.rpkg / rayzor-my-lib.rpkg inside
//! my-lib = { path = "../my-lib/my-lib" }         # extensionless file path → .rpkg auto-appended
//!
//! # By name from the local registry (~/.rayzor/packages/).
//! other-lib = { rpkg = "other-lib" }             # explicit registry lookup
//! third-lib = "1.0.0"                            # version string → registry lookup by key
//! fourth-lib = { version = "0.2" }               # same, table form
//! ```
//!
//! Git / hosted-registry resolution is not yet implemented and produces a
//! clear error pointing the user at `path` or local-registry alternatives.
//! The version-tag shape today resolves through the local registry; once a
//! hosted registry exists it'll fall back to that.

use super::manifest::{DependencySpec, ProjectManifest};
use crate::rpkg::registry::LocalRegistry;
use std::path::{Path, PathBuf};

/// Resolve every `[dependencies]` entry in `project_manifest` to a `.rpkg`
/// path. `project_root` is the directory containing `rayzor.toml` — relative
/// path specs are joined against it.
///
/// Returns the resolved paths in dependency-table order. Each entry is
/// validated to exist on disk; missing files produce a clear error.
///
/// On platforms without a local registry directory (or when the registry
/// is empty), `rpkg`-by-name lookups fail loudly with a hint to run
/// `rayzor rpkg install <file.rpkg>` first.
pub fn resolve_dependencies(
    project_manifest: &ProjectManifest,
    project_root: &Path,
) -> Result<Vec<PathBuf>, String> {
    let Some(deps) = project_manifest.dependencies.as_ref() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(deps.len());
    for (name, spec) in deps {
        let path = resolve_one(name, spec, project_root)?;
        out.push(path);
    }
    Ok(out)
}

fn resolve_one(name: &str, spec: &DependencySpec, project_root: &Path) -> Result<PathBuf, String> {
    match spec {
        DependencySpec::Version(_) => {
            resolve_from_registry(name).ok_or_else(|| registry_miss_message(name, name))
        }
        DependencySpec::Table {
            path,
            rpkg,
            git: _,
            branch: _,
            version,
        } => {
            if let Some(p) = path {
                return resolve_path(name, p, project_root);
            }
            if let Some(reg_name) = rpkg {
                return resolve_from_registry(reg_name)
                    .ok_or_else(|| registry_miss_message(name, reg_name));
            }
            if version.is_some() {
                // No path, no rpkg, but version present → registry lookup by
                // dep name. Same path as the bare-version form.
                return resolve_from_registry(name)
                    .ok_or_else(|| registry_miss_message(name, name));
            }
            Err(format!(
                "Dependency '{}' has no `path`, `rpkg`, or `version` field. \
                 Add one of them to rayzor.toml.",
                name
            ))
        }
    }
}

/// Resolve a `path = "..."` field to an existing `.rpkg` file.
///
/// Accepts any of:
/// - explicit file ending in `.rpkg`
/// - directory containing one or more `.rpkg` files (uses `<dir>/<name>.rpkg`,
///   `<dir>/rayzor-<name>.rpkg`, or the lone `.rpkg` if exactly one exists)
/// - extensionless file path (`.rpkg` auto-appended)
fn resolve_path(dep_name: &str, raw: &str, project_root: &Path) -> Result<PathBuf, String> {
    let raw_path = Path::new(raw);
    let abs = if raw_path.is_absolute() {
        raw_path.to_path_buf()
    } else {
        project_root.join(raw_path)
    };

    // 1. Direct hit — explicit file path exists.
    if abs.is_file() {
        return Ok(abs);
    }

    // 2. Auto-append `.rpkg` if missing.
    if abs.extension().is_none() {
        let with_ext = abs.with_extension("rpkg");
        if with_ext.is_file() {
            return Ok(with_ext);
        }
    }

    // 3. Directory — look for `<dep_name>.rpkg`, `rayzor-<dep_name>.rpkg`,
    //    or fall back to a unique `*.rpkg` inside.
    if abs.is_dir() {
        let candidates = [
            abs.join(format!("{}.rpkg", dep_name)),
            abs.join(format!("rayzor-{}.rpkg", dep_name)),
        ];
        for cand in &candidates {
            if cand.is_file() {
                return Ok(cand.clone());
            }
        }
        if let Ok(entries) = std::fs::read_dir(&abs) {
            let rpkgs: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("rpkg"))
                .collect();
            if rpkgs.len() == 1 {
                return Ok(rpkgs.into_iter().next().unwrap());
            }
            if rpkgs.len() > 1 {
                return Err(format!(
                    "Dependency '{}' points at directory {} which contains multiple .rpkg files. \
                     Disambiguate by pointing `path` at the specific .rpkg.",
                    dep_name,
                    abs.display()
                ));
            }
        }
        return Err(format!(
            "Dependency '{}' points at directory {} but no .rpkg file was found inside.",
            dep_name,
            abs.display()
        ));
    }

    Err(format!(
        "Dependency '{}' path {} resolved to nothing on disk. Tried `{}`, `{}.rpkg`, \
         and looking inside as a directory.",
        dep_name,
        raw,
        abs.display(),
        abs.display()
    ))
}

fn resolve_from_registry(name: &str) -> Option<PathBuf> {
    LocalRegistry::open_default()
        .ok()
        .and_then(|reg| reg.rpkg_path(name))
}

fn registry_miss_message(dep_name: &str, reg_name: &str) -> String {
    let by_label = if dep_name == reg_name {
        String::new()
    } else {
        format!(" (registry name '{}')", reg_name)
    };
    format!(
        "Dependency '{}'{} not found in the local registry. \
         Install it with `rayzor rpkg install <file.rpkg>`, point `path` at the .rpkg, \
         or wait for hosted registries (not yet implemented).",
        dep_name, by_label
    )
}
