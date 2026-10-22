use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    path::Path,
};

fn hash_sources(path: &Path, hash: &mut DefaultHasher) {
    let mut paths: Vec<_> = std::fs::read_dir(path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            hash_sources(&path, hash);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            println!("cargo:rerun-if-changed={}", path.display());
            path.hash(hash);
            std::fs::read(path).unwrap().hash(hash);
        }
    }
}

fn main() {
    let mut hash = DefaultHasher::new();
    hash_sources(Path::new("src"), &mut hash);
    println!("cargo:rerun-if-changed=build.rs");
    std::fs::read("build.rs").unwrap().hash(&mut hash);
    println!("cargo:rustc-env=RAYZOR_LSP_BUILD_ID={:016x}", hash.finish());
}
