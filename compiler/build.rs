use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    // On Linux, export symbols for dynamically loaded shared libraries
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-arg=-Wl,--export-dynamic");
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

    // Re-run this script if any source under src/ir/ changes so the
    // build-id moves and old caches get invalidated on the next build.
    println!("cargo:rerun-if-changed=src/ir");
    println!("cargo:rerun-if-changed=src/tast");
    println!("cargo:rerun-if-changed=../parser/src");
}
