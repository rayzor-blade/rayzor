//! Standard library discovery for editor analysis.

use std::path::{Path, PathBuf};

fn valid(path: &Path) -> bool {
    path.join("StdTypes.hx").is_file()
}

fn beside_executable(exe: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for exe in [Some(exe.to_path_buf()), exe.canonicalize().ok()]
        .into_iter()
        .flatten()
    {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("std"));
            paths.push(dir.join("../share/haxe/std"));
            paths.push(dir.join("../lib/haxe/std"));
        }
    }
    paths
}

fn haxe_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    candidates.extend(std::env::var_os("HAXE_STD_PATH").map(PathBuf::from));
    if let Some(home) = std::env::var_os("HAXEPATH") {
        candidates.push(PathBuf::from(home).join("std"));
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let exe = dir.join(if cfg!(windows) { "haxe.exe" } else { "haxe" });
            if exe.is_file() {
                candidates.extend(beside_executable(&exe));
            }
        }
    }
    candidates.extend(
        [
            "/usr/share/haxe/std",
            "/usr/local/share/haxe/std",
            "/usr/lib/haxe/std",
            "/usr/local/lib/haxe/std",
            "/opt/homebrew/share/haxe/std",
            "C:/HaxeToolkit/haxe/std",
        ]
        .map(PathBuf::from),
    );
    candidates
}

/// An installed Haxe SDK precedes an inherited native override for HXML files.
fn preferred_library(
    rayzor_override: Option<&Path>,
    haxe_candidates: &[PathBuf],
) -> Result<Option<Vec<PathBuf>>, String> {
    if let Some(path) = haxe_candidates.iter().find(|path| valid(path)) {
        return Ok(Some(vec![
            path.canonicalize().unwrap_or_else(|_| path.clone()),
        ]));
    }
    if let Some(path) = rayzor_override {
        if !valid(path) {
            return Err(format!(
                "RAYZOR_STD_PATH does not contain StdTypes.hx: {}",
                path.display()
            ));
        }
        return Ok(Some(vec![path.into()]));
    }
    Ok(None)
}

/// Native manifests use Rayzor's library, including its rayzor.* packages.
pub(crate) fn discover_for_project(root: &Path, native: bool) -> Result<Vec<PathBuf>, String> {
    let rayzor_override = std::env::var_os("RAYZOR_STD_PATH")
        .filter(|path| !path.is_empty())
        .map(PathBuf::from);
    let haxe = if native {
        Vec::new()
    } else {
        haxe_candidates()
    };
    if let Some(paths) = preferred_library(rayzor_override.as_deref(), &haxe)? {
        return Ok(paths);
    }
    if native {
        for directory in root.ancestors() {
            for relative in ["compiler/haxe-std", "haxe-std"] {
                let path = directory.join(relative);
                if valid(&path) {
                    return Ok(vec![path.canonicalize().unwrap_or(path)]);
                }
            }
        }
    }
    let paths = compiler::compilation::CompilationConfig::discover_stdlib_paths();
    let paths: Vec<_> = paths.into_iter().filter(|path| valid(path)).collect();
    if paths.is_empty() {
        Err(if native {
            "Rayzor standard library not found for rayzor.toml; set RAYZOR_STD_PATH"
        } else {
            "No Haxe standard library found; set HAXE_STD_PATH or RAYZOR_STD_PATH"
        }
        .into())
    } else {
        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hxml_sdk_precedes_inherited_native_override() {
        let root =
            std::env::temp_dir().join(format!("rayzor-lsp-sdk-choice-{}", std::process::id()));
        let native = root.join("native");
        let stock = root.join("haxe");
        for directory in [&native, &stock] {
            std::fs::create_dir_all(directory).unwrap();
            std::fs::write(directory.join("StdTypes.hx"), "").unwrap();
        }
        assert_eq!(
            preferred_library(Some(&native), std::slice::from_ref(&stock)).unwrap(),
            Some(vec![stock.canonicalize().unwrap()])
        );
        assert_eq!(
            preferred_library(Some(&native), &[]).unwrap(),
            Some(vec![native.clone()])
        );
        let missing = root.join("missing");
        assert_eq!(
            preferred_library(Some(&missing), std::slice::from_ref(&stock)).unwrap(),
            Some(vec![stock.canonicalize().unwrap()])
        );
        assert!(preferred_library(Some(&missing), &[]).is_err());
        assert_eq!(preferred_library(None, &[]).unwrap(), None);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundled_symbols_cover_native_apis_and_reject_another_library() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../compiler/haxe-std");
        let config = compiler::compilation::CompilationConfig {
            typecheck_only: true,
            stdlib_paths: vec![root.clone()],
            ..Default::default()
        };
        let mut unit = compiler::compilation::CompilationUnit::new(config.clone());
        assert!(unit.load_stdlib_symbols());
        for name in [
            "rayzor.concurrent.Channel",
            "haxe.io.Bytes",
            "haxe.io.BytesData",
            "haxe.io.Path",
            "haxe.ds.StringMap",
        ] {
            assert!(
                unit.symbol_table.all_symbols().any(|symbol| {
                    unit.string_interner
                        .get(symbol.qualified_name.unwrap_or(symbol.name))
                        == Some(name)
                }),
                "missing snapshot type {name}"
            );
        }
        let other =
            std::env::temp_dir().join(format!("rayzor-other-stdlib-{}", std::process::id()));
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(
            other.join("StdTypes.hx"),
            "class DifferentStandardLibrary {}",
        )
        .unwrap();
        let mut unit =
            compiler::compilation::CompilationUnit::new(compiler::compilation::CompilationConfig {
                stdlib_paths: vec![other.clone(), root],
                ..config
            });
        assert!(
            !unit.load_stdlib_symbols(),
            "restored symbols from an unrelated library"
        );
        std::fs::remove_dir_all(other).unwrap();
    }

    #[test]
    fn discovers_a_symlinked_haxe_installation() {
        let root = std::env::temp_dir().join(format!("rayzor-lsp-stdlib-{}", std::process::id()));
        std::fs::create_dir_all(root.join("install/std")).unwrap();
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(root.join("install/haxe"), "").unwrap();
        std::fs::write(root.join("install/std/StdTypes.hx"), "").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(root.join("install/haxe"), root.join("bin/haxe")).unwrap();
            assert!(
                beside_executable(&root.join("bin/haxe"))
                    .iter()
                    .any(|p| valid(p))
            );
        }
        assert!(
            beside_executable(&root.join("install/haxe"))
                .iter()
                .any(|p| valid(p))
        );
        assert!(!valid(&root));
        std::fs::remove_dir_all(root).unwrap();
    }
}
