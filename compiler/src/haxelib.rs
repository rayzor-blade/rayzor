//! Finds `-lib` libraries in a haxelib repository, as `haxelib path` would,
//! without running haxelib.

use crate::hxml::HxmlConfig;
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};

/// What one library adds to a build.
#[derive(Debug, Clone, Default)]
pub struct Library {
    pub name: String,
    pub version: Option<String>,
    pub class_paths: Vec<PathBuf>,
    /// `-D name=version`, plus the defines of its `extraParams.hxml`.
    pub defines: Vec<(String, Option<String>)>,
    /// Initialization macros from the library's extraParams.hxml.
    pub macros: Vec<String>,
}

/// A dependency with paths already resolved against its declaring file.
#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    version: Option<String>,
    path: Option<PathBuf>,
}

impl Dependency {
    pub fn is_unpinned(&self) -> bool {
        self.version.is_none() && self.path.is_none()
    }

    pub fn is_local(&self) -> bool {
        self.path.is_some()
    }

    pub fn parse(spec: &str, base: &Path) -> Self {
        let (name, version) = spec.split_once(':').unwrap_or((spec, ""));
        Self::from_spec(name, &DependencySpec::Version(version.into()), base)
    }

    fn from_spec(name: &str, spec: &DependencySpec, base: &Path) -> Self {
        let (version, path) = match spec {
            DependencySpec::Path { path } => (None, Some(base.join(path))),
            DependencySpec::Version(value)
                if value.starts_with("./")
                    || value.starts_with("../")
                    || Path::new(value).is_absolute() =>
            {
                (None, Some(base.join(value)))
            }
            DependencySpec::Version(value) => ((!value.is_empty()).then(|| value.clone()), None),
        };
        Self {
            name: name.into(),
            version,
            path,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Version(String),
    Path { path: PathBuf },
}

/// The source and dependency fields of haxelib.json. Path tables extend the
/// usual version strings for local editor/workspace dependencies.
#[derive(Debug, Default, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default, rename = "classPath")]
    pub class_path: String,
    #[serde(default)]
    pub dependencies: BTreeMap<String, DependencySpec>,
}

impl Manifest {
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn dependencies(&self, root: &Path) -> Vec<Dependency> {
        self.dependencies
            .iter()
            .map(|(name, spec)| Dependency::from_spec(name, spec, root))
            .collect()
    }
}

/// The libraries `requested` (`name` or `name:version`) and everything they
/// depend on, each once, with a message for every one that is missing.
pub fn resolve(requested: &[String], project_dir: &Path) -> (Vec<Library>, Vec<String>) {
    let requested = requested
        .iter()
        .map(|spec| Dependency::parse(spec, project_dir))
        .collect::<Vec<_>>();
    resolve_dependencies(&requested, project_dir, &[])
}

/// `provided` names libraries already supplied by source paths. The caller
/// supplies their dependencies, so their installed copies and hooks are skipped.
pub fn resolve_dependencies(
    requested: &[Dependency],
    project_dir: &Path,
    provided: &[String],
) -> (Vec<Library>, Vec<String>) {
    resolve_in_repository(requested, repository(project_dir).as_deref(), provided)
}

fn resolve_in_repository(
    requested: &[Dependency],
    repo: Option<&Path>,
    provided: &[String],
) -> (Vec<Library>, Vec<String>) {
    let mut libraries: Vec<Library> = Vec::new();
    let mut problems = Vec::new();
    let mut queue: VecDeque<Dependency> = requested.iter().cloned().collect();
    let mut seen: HashSet<String> = provided.iter().cloned().collect();
    // Resolve direct requests before transitive dependencies so an explicit
    // version cannot be replaced by another library's dependency declaration.
    while let Some(Dependency {
        name,
        version,
        path,
    }) = queue.pop_front()
    {
        if !seen.insert(name.clone()) {
            continue;
        }
        let resolved = match path {
            Some(root) => library_at(&root, &name, None),
            None => repo
                .ok_or_else(|| "no haxelib repository found".to_string())
                .and_then(|repo| library(repo, &name, version.as_deref())),
        };
        match resolved {
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

type Dependencies = Vec<Dependency>;

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

    library_at(&root, name, version)
}

fn library_at(
    root: &Path,
    name: &str,
    version: Option<String>,
) -> Result<(Library, Dependencies), String> {
    if !root.is_dir() {
        return Err(format!(
            "local library directory does not exist: {}",
            root.display()
        ));
    }
    let manifest_path = root.join("haxelib.json");
    let manifest = if manifest_path.is_file() {
        Manifest::read(&manifest_path)?
    } else {
        Manifest::default()
    };
    let mut lib = Library {
        name: name.to_string(),
        version: version.clone(),
        class_paths: vec![root.join(&manifest.class_path)],
        defines: vec![(name.to_string(), version)],
        macros: Vec::new(),
    };
    let mut dependencies = manifest.dependencies(root);

    let extra = root.join("extraParams.hxml");
    if extra.is_file()
        && let Ok(params) = HxmlConfig::builds_from_file(&extra)
        && let Some(params) = params.into_iter().next()
    {
        // Its relative paths name the library's own files.
        lib.class_paths.extend(params.class_paths);
        lib.defines.extend(params.defines);
        lib.macros.extend(params.macros);
        dependencies.extend(params.libraries.iter().map(|l| Dependency::parse(l, root)));
    }
    Ok((lib, dependencies))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_libraries_resolve_transitive_paths_and_cycles_without_a_repository() {
        let root = std::env::temp_dir().join(format!("rayzor-local-libs-{}", std::process::id()));
        std::fs::create_dir_all(root.join("first library/src")).unwrap();
        std::fs::create_dir_all(root.join("second/src")).unwrap();
        std::fs::write(
            root.join("first library/haxelib.json"),
            r#"{"classPath":"src","dependencies":{"second":{"path":"../second"}}}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("second/haxelib.json"),
            r#"{"classPath":"src","dependencies":{"first":"../first library"}}"#,
        )
        .unwrap();
        std::fs::write(
            root.join("second/extraParams.hxml"),
            "--macro Second.enable()\n-D local_second",
        )
        .unwrap();
        let (libs, problems) = resolve_in_repository(
            &[Dependency::parse("first:./first library", &root)],
            None,
            &[],
        );
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(libs.len(), 2);
        assert_eq!(
            libs[1].class_paths[0].canonicalize().unwrap(),
            root.join("second/src").canonicalize().unwrap()
        );
        assert_eq!(libs[1].macros, ["Second.enable()"]);
        assert!(libs[1].defines.contains(&("local_second".into(), None)));
        let (_, problems) =
            resolve_in_repository(&[Dependency::parse("missing:./not-here", &root)], None, &[]);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("local library directory does not exist"));
        std::fs::remove_dir_all(root).unwrap();
    }

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
