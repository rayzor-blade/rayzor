use std::path::{Path, PathBuf};

const CACHE_ABI_VERSION: &str = "rayzor-cache-abi-v1";

struct Fnv64(u64);

impl Fnv64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn write_str(&mut self, value: &str) {
        self.write(value.as_bytes());
        self.write(&[0]);
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn main() {
    // On Linux, export symbols for dynamically loaded shared libraries
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    }

    if std::env::var_os("CARGO_FEATURE_LLVM_BACKEND").is_some() {
        build_llvm21_const_compat();
    }

    generate_stdlib_manifest();

    // Emit a cache ABI build id. BLADE cache entries are tagged with this
    // value, so MIR cached by one compiler binary is only reused by another
    // binary built from the same compiler/parser/stdlib inputs. This keeps the
    // corruption guard from the old per-build timestamp without invalidating
    // every cache on a redundant relink of identical sources.
    println!("cargo:rustc-env=RAYZOR_BUILD_ID={}", cache_build_id());

    // Re-run this script (bumping the build id) whenever ANY compiler
    // source changes. The previous list covered only src/ir, src/tast,
    // and the parser — changes to src/codegen, src/compilation.rs,
    // src/stdlib, haxe-std, etc. kept the OLD build id, so BLADE caches
    // written by a meaningfully different compiler still validated and
    // poisoned imports ("can't resolve symbol append" Cranelift panics,
    // IMPORT[...] field errors). Over-invalidation is the right trade:
    // one warm-up compile (~6-8s) per compiler rebuild, versus silent
    // import corruption.
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=haxe-std");
    println!("cargo:rerun-if-changed=../parser/src");
    println!("cargo:rerun-if-changed=../diagnostics/src");
}

/// Extract the standard library's declarations into `OUT_DIR/stdlib.bsym`.
///
/// The compiler embeds this manifest, so building it here is what keeps the two
/// in step: `rerun-if-changed=haxe-std` already re-runs this script whenever the
/// standard library changes, and the manifest is rebuilt from those same
/// sources. A committed artifact could disagree with them; one produced here
/// cannot.
/// Reserve for the parse below. A recursive-descent parser's stack depth
/// follows how deeply the source nests, and the standard library has
/// expressions deep enough to exceed the 1 MB a thread gets by default on
/// Windows — where it shows up as the build script exiting with
/// STATUS_STACK_OVERFLOW rather than as a parse error.
const MANIFEST_PARSE_STACK: usize = 32 * 1024 * 1024;

fn generate_stdlib_manifest() {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let stdlib_root = PathBuf::from("haxe-std");

    // Parse on a thread with a stack sized for it, rather than on whichever
    // stack the host happened to give the build script.
    let root = stdlib_root.clone();
    let modules = std::thread::Builder::new()
        .stack_size(MANIFEST_PARSE_STACK)
        .spawn(move || bsym::build_manifest(&root))
        .expect("failed to spawn the manifest parse")
        .join()
        .expect("the stdlib manifest parse panicked");
    if modules.is_empty() {
        panic!(
            "no stdlib declarations found under {}; the embedded manifest would be empty",
            stdlib_root.display()
        );
    }

    if let Err(e) = bsym::save_symbol_manifest(out_dir.join("stdlib.bsym"), modules) {
        panic!("failed to write the stdlib symbol manifest: {e}");
    }
}

fn cache_build_id() -> String {
    let mut hash = Fnv64::new();
    hash.write_str(CACHE_ABI_VERSION);
    hash.write_str(env!("CARGO_PKG_VERSION"));

    // Target shape only. A cache entry holds MIR and type information, all of
    // it produced before a backend is chosen, so which backends a build links
    // does not change what a module lowers to — and including them would give
    // the same front end two identities, so artifacts produced by a
    // backend-free build of this crate could not be read by one that links a
    // backend.
    for key in [
        "CARGO_CFG_TARGET_ARCH",
        "CARGO_CFG_TARGET_OS",
        "CARGO_CFG_TARGET_ENV",
        "CARGO_CFG_TARGET_POINTER_WIDTH",
    ] {
        hash.write_str(key);
        hash.write_str(&std::env::var(key).unwrap_or_default());
    }

    for path in [
        PathBuf::from("Cargo.toml"),
        PathBuf::from("../Cargo.lock"),
        PathBuf::from("../parser/Cargo.toml"),
        PathBuf::from("../diagnostics/Cargo.toml"),
    ] {
        hash_path(&path, &mut hash);
    }

    for dir in [
        PathBuf::from("src"),
        PathBuf::from("haxe-std"),
        PathBuf::from("../parser/src"),
        PathBuf::from("../diagnostics/src"),
    ] {
        hash_tree(&dir, &mut hash);
    }

    format!("{:016x}", hash.finish())
}

fn hash_tree(root: &Path, hash: &mut Fnv64) {
    let mut files = Vec::new();
    collect_files(root, &mut files);
    files.sort();
    for file in files {
        hash_path(&file, hash);
    }
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    if meta.is_file() {
        files.push(path.to_path_buf());
        return;
    }
    if !meta.is_dir() {
        return;
    }

    let Ok(entries) = std::fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        collect_files(&entry.path(), files);
    }
}

fn hash_path(path: &Path, hash: &mut Fnv64) {
    let display = path.to_string_lossy();
    hash.write_str(&display);
    match std::fs::read(path) {
        Ok(bytes) => {
            hash.write_str("file");
            hash.write(&bytes);
            hash.write(&[0]);
        }
        Err(_) => hash.write_str("missing"),
    }
}

fn build_llvm21_const_compat() {
    println!("cargo:rerun-if-changed=src/llvm21_const_compat.cpp");

    let include_dir = std::env::var("LLVM_SYS_211_PREFIX")
        .ok()
        .or_else(|| std::env::var("LLVM_SYS_PREFIX").ok())
        .map(|prefix| std::path::PathBuf::from(prefix).join("include"))
        .filter(|path| path.exists())
        .or_else(|| llvm_config_arg("--includedir"))
        .or_else(|| {
            let path = std::path::PathBuf::from("/opt/homebrew/opt/llvm/include");
            path.exists().then_some(path)
        });

    let Some(include_dir) = include_dir else {
        // Let llvm-sys emit the real configuration error later; this shim is
        // only needed once LLVM headers are discoverable.
        return;
    };

    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .include(include_dir)
        .file("src/llvm21_const_compat.cpp")
        .compile("rayzor_llvm21_const_compat");
}

fn llvm_config_arg(arg: &str) -> Option<std::path::PathBuf> {
    let output = std::process::Command::new("llvm-config")
        .arg(arg)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let path = String::from_utf8(output.stdout).ok()?;
    let path = std::path::PathBuf::from(path.trim());
    path.exists().then_some(path)
}
