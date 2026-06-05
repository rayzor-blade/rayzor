// macOS cdylibs need `-undefined dynamic_lookup` so host-process
// symbols (rayzor_tensor_zeros etc., statically linked into the
// `rayzor` binary) resolve at dlopen time instead of demanding them
// at static link time. Scoped to this crate via build.rs so it doesn't
// affect the rest of the workspace.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
