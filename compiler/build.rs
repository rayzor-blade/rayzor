use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // On Linux, export symbols for dynamically loaded shared libraries
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
    }

    if std::env::var_os("CARGO_FEATURE_LLVM_BACKEND").is_some() {
        build_llvm21_const_compat();
    }

    // Emit a build ID that changes on every rebuild. BLADE cache entries
    // are tagged with this ID, so MIR cached by one compiler build is
    // invalidated when the compiler itself is recompiled — protects
    // against silent miscompiles when a parser/lowerer change shifts
    // function IDs or AST shape for the same source.
    //
    // Use the compile-time clock (seconds since UNIX epoch). Two builds
    // in the same second would collide, which is fine for cache
    // invalidation in practice (rebuilding the compiler takes longer
    // than a second).
    let build_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    println!("cargo:rustc-env=RAYZOR_BUILD_ID={}", build_secs);

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
