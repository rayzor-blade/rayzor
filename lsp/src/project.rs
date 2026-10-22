//! Conservative fallbacks for workspaces without a complete editor build.

use compiler::haxelib::{Dependency, Manifest};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Prefer a root build, then root HXML files, then files at most two directories
/// below the root. Never merge candidates: their targets may be incompatible.
pub(crate) fn find_hxml(root: &Path, problems: &mut Vec<String>) -> Option<PathBuf> {
    let build = root.join("build.hxml");
    if build.is_file() {
        return Some(build);
    }
    let mut directories = vec![root.to_path_buf()];
    let mut visited = 0;
    for _depth in 0..=2 {
        let mut next = Vec::new();
        let mut candidates = Vec::new();
        for directory in directories {
            visited += 1;
            if visited > 256 {
                problems.push("HXML discovery stopped after 256 directories; select a build with initializationOptions.hxml".into());
                break;
            }
            let Ok(entries) = std::fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(kind) = entry.file_type() else {
                    continue;
                };
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with('.') {
                    continue;
                }
                // Do not follow symlinks or scan dependency/build caches.
                if kind.is_dir()
                    && !matches!(
                        name.as_ref(),
                        "target" | "node_modules" | "vendor" | "bin" | "out" | "dist"
                    )
                {
                    next.push(entry.path());
                } else if kind.is_file()
                    && entry.path().extension().is_some_and(|e| e == "hxml")
                    && name != "extraParams.hxml"
                {
                    candidates.push(entry.path());
                }
            }
        }
        candidates.sort_by_key(|path| {
            (
                path.file_name().is_none_or(|n| n != "build.hxml"),
                path.clone(),
            )
        });
        if let Some(selected) = candidates.first() {
            if candidates.len() > 1 {
                problems.push(format!(
                    "Multiple HXML builds found; using {}. Select another with initializationOptions.hxml",
                    selected.display()
                ));
            }
            return Some(selected.clone());
        }
        if visited > 256 {
            break;
        }
        next.sort();
        directories = next;
    }
    None
}

#[derive(Default)]
pub(crate) struct LibraryMetadata {
    pub input_files: Vec<PathBuf>,
    pub class_paths: Vec<PathBuf>,
    pub libraries: Vec<Dependency>,
    pub provided: Vec<String>,
}

/// Inspect the workspace manifest and the nearest manifest owning each
/// configured class path, including local libraries outside the workspace.
/// Class paths do not imply `-lib`: do not run these libraries' extraParams
/// hooks merely because their source directories are referenced.
pub(crate) fn library_metadata(
    root: &Path,
    class_paths: &[PathBuf],
    problems: &mut Vec<String>,
) -> LibraryMetadata {
    let mut manifests = vec![root.join("haxelib.json")];
    for class_path in class_paths {
        let path = class_path
            .canonicalize()
            .unwrap_or_else(|_| class_path.clone());
        if let Some(manifest) = path
            .ancestors()
            .map(|parent| parent.join("haxelib.json"))
            .find(|manifest| manifest.is_file())
        {
            manifests.push(manifest);
        }
    }
    let mut metadata = LibraryMetadata::default();
    let mut seen = HashSet::new();
    for path in manifests {
        metadata.input_files.push(path.clone());
        if !path.is_file() || !seen.insert(path.canonicalize().unwrap_or_else(|_| path.clone())) {
            continue;
        }
        match Manifest::read(&path) {
            Ok(manifest) => {
                let root = path.parent().unwrap();
                metadata.libraries.extend(manifest.dependencies(root));
                metadata.provided.extend(manifest.name);
                metadata.class_paths.push(root.join(manifest.class_path));
            }
            Err(error) => problems.push(error),
        }
    }
    metadata
}

#[cfg(test)]
mod tests {
    use crate::context::LspContext;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Workspace(PathBuf);

    impl Workspace {
        fn new(files: &[(&str, &str)]) -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let root = std::env::temp_dir().join(format!(
                "rayzor-project-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            for (name, text) in files {
                let path = root.join(name);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(path, text).unwrap();
            }
            Self(root)
        }

        fn load(&self, hxml: Option<&str>) -> LspContext {
            let mut context = LspContext::new(self.0.clone());
            context.load_workspace_config(hxml);
            context
        }
    }

    impl Drop for Workspace {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn finds_nested_build_but_explicit_selection_wins() {
        let workspace = Workspace::new(&[
            ("extraParams.hxml", "--macro Wrong.enable()"),
            ("target/build.hxml", "--macro Wrong.enable()"),
            (".cache/build.hxml", "--macro Wrong.enable()"),
            ("config/build.hxml", "-cp ../src\n--macro Nested.enable()"),
            ("config/web.hxml", "--macro Web.enable()"),
        ]);
        let context = workspace.load(None);
        assert_eq!(context.hxml, Some(workspace.0.join("config/build.hxml")));
        assert_eq!(context.initialization_macros, ["Nested.enable()"]);
        assert_eq!(context.config_problems.len(), 1); // ambiguity is visible
        assert_eq!(
            workspace
                .load(Some("config/web.hxml"))
                .initialization_macros,
            ["Web.enable()"]
        );
        let missing = workspace.load(Some("missing.hxml"));
        assert!(missing.initialization_macros.is_empty());
        assert_eq!(missing.config_problems.len(), 1);
    }

    #[test]
    fn discovers_manifest_dependencies_and_library_macros_without_hxml() {
        let workspace = Workspace::new(&[
            (
                "haxelib.json",
                r#"{"classPath":"src","dependencies":{"helper":"1.0.0"}}"#,
            ),
            ("extraParams.hxml", "--macro Local.enable()"),
            (
                ".haxelib/helper/1,0,0/haxelib.json",
                r#"{"classPath":"haxe"}"#,
            ),
            (
                ".haxelib/helper/1,0,0/extraParams.hxml",
                "--macro Helper.enable()",
            ),
        ]);
        let context = workspace.load(None);
        assert!(context.hxml.is_none());
        assert!(
            context.config_problems.is_empty(),
            "{:?}",
            context.config_problems
        );
        assert!(context.class_paths.contains(&workspace.0.join("src")));
        assert!(
            context
                .class_paths
                .contains(&workspace.0.join(".haxelib/helper/1,0,0/haxe"))
        );
        assert_eq!(
            context.initialization_macros,
            ["Local.enable()", "Helper.enable()"]
        );
    }

    #[test]
    fn class_path_metadata_fills_dependencies_without_overriding_build() {
        let workspace = Workspace::new(&[
            (
                "app/build.hxml",
                "-cp ../local/haxe\n-cp ../local/components\n-lib helper:2.0.0\n--macro Chosen.enable()",
            ),
            ("local/haxe/Local.hx", "class Local {}"),
            ("local/components/Component.hx", "class Component {}"),
            (
                "local/haxelib.json",
                r#"{"name":"local","classPath":"haxe","dependencies":{"helper":"1.0.0","transitive":""}}"#,
            ),
            ("local/extraParams.hxml", "--macro Unselected.enable()"),
            (".haxelib/helper/2,0,0/haxelib.json", "{}"),
            (".haxelib/transitive/.current", "1.0.0"),
            (
                ".haxelib/transitive/1,0,0/haxelib.json",
                r#"{"dependencies":{"helper":"1.0.0","local":"0.2.0"}}"#,
            ),
            (
                ".haxelib/transitive/1,0,0/extraParams.hxml",
                "--macro Dependency.enable()",
            ),
        ]);
        let mut context = LspContext::new(workspace.0.join("app"));
        context.load_workspace_config(None);
        assert!(
            context.config_problems.is_empty(),
            "{:?}",
            context.config_problems
        );
        assert_eq!(
            context.initialization_macros,
            ["Chosen.enable()", "Dependency.enable()"]
        );
        assert!(
            context
                .class_paths
                .iter()
                .any(|p| p.ends_with("helper/2,0,0/"))
        );
        assert_eq!(
            context
                .class_paths
                .iter()
                .filter(
                    |p| p.canonicalize().ok() == workspace.0.join("local/haxe").canonicalize().ok()
                )
                .count(),
            1
        );
    }

    #[test]
    fn local_path_supplies_an_unpinned_hxml_library_and_dev_dependencies() {
        let workspace = Workspace::new(&[
            ("app/build.hxml", "-lib local\n"),
            (
                "app/haxelib.json",
                r#"{"dependencies":{"local":{"path":"../local"}}}"#,
            ),
            (
                "local/haxelib.json",
                r#"{"name":"local","classPath":"src","dependencies":{"helper":""}}"#,
            ),
            ("local/src/Local.hx", "class Local {}"),
            ("local/extraParams.hxml", "--macro Local.enable()"),
            ("helper/haxelib.json", r#"{"classPath":"haxe"}"#),
            ("helper/extraParams.hxml", "--macro Helper.enable()"),
            (".haxelib/helper/.current", "not-installed"),
        ]);
        std::fs::write(
            workspace.0.join(".haxelib/helper/.dev"),
            workspace.0.join("helper").to_str().unwrap(),
        )
        .unwrap();
        for hxml in ["-lib local\n-cp ../local/src", "-cp ../local/src"] {
            std::fs::write(workspace.0.join("app/build.hxml"), hxml).unwrap();
            let mut context = LspContext::new(workspace.0.join("app"));
            context.load_workspace_config(None);
            assert!(
                context.config_problems.is_empty(),
                "{:?}",
                context.config_problems
            );
            assert_eq!(
                context.initialization_macros,
                ["Local.enable()", "Helper.enable()"]
            );
            assert!(
                context
                    .class_paths
                    .contains(&workspace.0.join("helper/haxe"))
            );
            assert!(context.defines.iter().any(|d| d == "helper"));
        }
    }

    #[test]
    fn broken_metadata_does_not_discard_explicit_settings() {
        let workspace = Workspace::new(&[
            ("build.hxml", "-cp src\n--macro Setup.enable()"),
            ("haxelib.json", "{ broken json"),
        ]);
        let context = workspace.load(None);
        assert_eq!(context.initialization_macros, ["Setup.enable()"]);
        assert!(context.class_paths.contains(&workspace.0.join("src")));
        assert_eq!(context.config_problems.len(), 1);
        assert!(context.config_problems[0].contains("haxelib.json"));
    }
}
