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

/// An explicit Rayzor library wins; otherwise an editor uses Haxe's library.
/// Bundled Rayzor discovery remains available when Haxe is not installed.
pub(crate) fn discover() -> Result<Vec<PathBuf>, String> {
    if let Some(path) = std::env::var_os("RAYZOR_STD_PATH").filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        return if valid(&path) {
            Ok(vec![path])
        } else {
            Err(format!(
                "RAYZOR_STD_PATH does not contain StdTypes.hx: {}",
                path.display()
            ))
        };
    }
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
    if let Some(path) = candidates.into_iter().find(|p| valid(p)) {
        return Ok(vec![path.canonicalize().unwrap_or(path)]);
    }
    let paths = compiler::compilation::CompilationConfig::discover_stdlib_paths();
    if paths.iter().any(|p| valid(p)) {
        Ok(paths)
    } else {
        Err("No Haxe standard library found; set HAXE_STD_PATH or RAYZOR_STD_PATH".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
