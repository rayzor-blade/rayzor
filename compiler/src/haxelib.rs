//! Finds `-lib` libraries in a haxelib repository, as `haxelib path` would,
//! without running haxelib.

use crate::hxml::HxmlConfig;
use std::path::{Path, PathBuf};

/// What one library adds to a build.
#[derive(Debug, Clone, Default)]
pub struct Library {
    pub name: String,
    pub version: Option<String>,
    pub class_paths: Vec<PathBuf>,
    /// `-D name=version`, plus the defines of its `extraParams.hxml`.
    pub defines: Vec<(String, Option<String>)>,
}

/// The libraries `requested` (`name` or `name:version`) and everything they
/// depend on, each once, with a message for every one that is missing.
pub fn resolve(requested: &[String], project_dir: &Path) -> (Vec<Library>, Vec<String>) {
    let Some(repo) = repository(project_dir) else {
        let missing = requested
            .iter()
            .map(|lib| format!("-lib {lib}: no haxelib repository found"))
            .collect();
        return (Vec::new(), missing);
    };
    let mut libraries: Vec<Library> = Vec::new();
    let mut problems = Vec::new();
    let mut queue: Vec<(String, Option<String>)> = requested
        .iter()
        .map(|lib| match lib.split_once(':') {
            Some((name, version)) => (name.to_string(), Some(version.to_string())),
            None => (lib.clone(), None),
        })
        .collect();
    while let Some((name, version)) = queue.pop() {
        if libraries.iter().any(|l| l.name == name) {
            continue;
        }
        match library(&repo, &name, version.as_deref()) {
            Ok((lib, dependencies)) => {
                libraries.push(lib);
                queue.extend(dependencies);
            }
            Err(e) => problems.push(format!("-lib {name}: {e}")),
        }
    }
    (libraries, problems)
}

/// The repository haxelib would use from `project_dir`: a local `.haxelib`
/// in it or above it, `HAXELIB_PATH`, the path `~/.haxelib` names, `~/haxelib`.
pub fn repository(project_dir: &Path) -> Option<PathBuf> {
    let local = project_dir
        .ancestors()
        .map(|d| d.join(".haxelib"))
        .find(|d| d.is_dir());
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    let configured = home
        .as_ref()
        .and_then(|h| std::fs::read_to_string(h.join(".haxelib")).ok())
        .map(|p| PathBuf::from(p.trim()));
    local
        .into_iter()
        .chain(std::env::var_os("HAXELIB_PATH").map(PathBuf::from))
        .chain(configured)
        .chain(home.map(|h| h.join("haxelib")))
        .find(|d| d.is_dir())
}

type Dependencies = Vec<(String, Option<String>)>;

fn library(
    repo: &Path,
    name: &str,
    version: Option<&str>,
) -> Result<(Library, Dependencies), String> {
    let folder = name.replace('.', ",");
    let dir = [folder.clone(), folder.to_lowercase()]
        .into_iter()
        .map(|f| repo.join(f))
        .find(|d| d.is_dir())
        .ok_or_else(|| format!("not installed in {}", repo.display()))?;
    let dev = std::fs::read_to_string(dir.join(".dev"))
        .ok()
        .map(|p| PathBuf::from(p.trim()))
        .filter(|p| version.is_none() && p.is_dir());
    let (root, version) = match dev {
        Some(path) => (path, None),
        None => {
            let version = match version {
                Some(v) => v.to_string(),
                None => std::fs::read_to_string(dir.join(".current"))
                    .map(|v| v.trim().to_string())
                    .map_err(|_| "no current version set".to_string())?,
            };
            let root = dir.join(version.replace('.', ","));
            if !root.is_dir() {
                return Err(format!("version {version} is not installed"));
            }
            (root, Some(version))
        }
    };

    let manifest: serde_json::Value = std::fs::read_to_string(root.join("haxelib.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default();
    let class_path = manifest["classPath"].as_str().unwrap_or("");
    let mut lib = Library {
        name: name.to_string(),
        version: version.clone(),
        class_paths: vec![root.join(class_path)],
        defines: vec![(name.to_string(), version)],
    };
    let mut dependencies: Dependencies = manifest["dependencies"]
        .as_object()
        .map(|deps| {
            deps.iter()
                .map(|(dep, v)| {
                    let v = v.as_str().filter(|v| !v.is_empty()).map(str::to_string);
                    (dep.clone(), v)
                })
                .collect()
        })
        .unwrap_or_default();

    let extra = root.join("extraParams.hxml");
    if extra.is_file()
        && let Ok(params) = HxmlConfig::builds_from_file(&extra)
        && let Some(params) = params.into_iter().next()
    {
        // Its relative paths name the library's own files.
        lib.class_paths.extend(params.class_paths);
        lib.defines.extend(params.defines);
        dependencies.extend(params.libraries.iter().map(|l| match l.split_once(':') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (l.clone(), None),
        }));
    }
    Ok((lib, dependencies))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_current_version_its_class_path_and_dependencies() {
        let repo = std::env::temp_dir().join(format!("rayzor-haxelib-{}", std::process::id()));
        let lib = repo.join(".haxelib");
        std::fs::create_dir_all(lib.join("foo/1,2,0/src")).unwrap();
        std::fs::write(lib.join("foo/.current"), "1.2.0\n").unwrap();
        std::fs::write(
            lib.join("foo/1,2,0/haxelib.json"),
            r#"{"name":"foo","classPath":"src","dependencies":{"bar":""}}"#,
        )
        .unwrap();
        std::fs::create_dir_all(lib.join("bar/dev")).unwrap();
        std::fs::write(
            lib.join("bar/.dev"),
            lib.join("bar/dev").display().to_string(),
        )
        .unwrap();

        let (libs, problems) = resolve(&["foo".into()], &repo);
        assert!(problems.is_empty(), "{problems:?}");
        let foo = libs.iter().find(|l| l.name == "foo").unwrap();
        assert_eq!(foo.class_paths, vec![lib.join("foo/1,2,0/src")]);
        assert_eq!(foo.defines, vec![("foo".into(), Some("1.2.0".into()))]);
        let bar = libs.iter().find(|l| l.name == "bar").unwrap();
        assert_eq!(bar.class_paths, vec![lib.join("bar/dev")]);

        let (_, problems) = resolve(&["missing".into()], &repo);
        assert_eq!(problems.len(), 1);
        std::fs::remove_dir_all(&repo).unwrap();
    }
}
