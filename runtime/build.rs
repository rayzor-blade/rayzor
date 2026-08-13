fn main() {
    #[cfg(feature = "tcc-runtime")]
    build_tcc();
}

#[cfg(feature = "tcc-runtime")]
fn build_tcc() {
    // TCC source lives in the compiler crate's vendor directory
    let tcc_dir = std::path::Path::new("../compiler/vendor/tinycc");
    if !tcc_dir.exists() {
        panic!(
            "TCC source not found at ../compiler/vendor/tinycc. \
             Run: git clone --depth 1 https://github.com/TinyCC/tinycc.git compiler/vendor/tinycc"
        );
    }

    // Resolve absolute path so TCC can find its own includes (tccdefs.h) at runtime
    let tcc_abs = std::fs::canonicalize(tcc_dir).expect("Failed to resolve TCC vendor path");
    let tcc_dir_quoted = format!("\"{}\"", tcc_abs.display());

    let mut build = cc::Build::new();
    build
        .file(tcc_dir.join("libtcc.c"))
        .include(tcc_dir)
        .define("ONE_SOURCE", "1")
        .define("TCC_LIBTCC", "1")
        // NOTE: do NOT define CONFIG_TCC_STATIC — it replaces dlsym/dlopen
        // with dummies that only know 4 symbols. We need real dlsym so TCC
        // can resolve any libc/libm/system symbol during JIT relocation.
        .define("CONFIG_TCCDIR", tcc_dir_quoted.as_str())
        .warnings(false);

    // Describe the TARGET, not the host. A build script is compiled for the
    // host, so `cfg!` here answers the wrong question the moment anyone
    // cross-compiles.
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    match target_arch.as_str() {
        "x86_64" => {
            build.define("TCC_TARGET_X86_64", "1");
        }
        "aarch64" => {
            build.define("TCC_TARGET_ARM64", "1");
        }
        "x86" => {
            build.define("TCC_TARGET_I386", "1");
        }
        _ => {}
    }

    // The object format. Without it TCC assumes ELF, and on Windows that also
    // means it does not consider itself native — so `ONE_SOURCE` never pulls in
    // tccrun.c and `tcc_relocate`, the entry point the JIT is built on, simply
    // does not exist to link against.
    match target_os.as_str() {
        "macos" => {
            build.define("TCC_TARGET_MACHO", "1");
        }
        "windows" => {
            build.define("TCC_TARGET_PE", "1");
        }
        _ => {}
    }

    build.compile("tcc");

    println!("cargo:rerun-if-changed=../compiler/vendor/tinycc/libtcc.c");
    println!("cargo:rerun-if-changed=../compiler/vendor/tinycc/libtcc.h");
}
