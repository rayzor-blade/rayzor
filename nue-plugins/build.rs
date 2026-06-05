// macOS cdylibs need `-undefined,dynamic_lookup` so the linker defers
// resolution of host-provided symbols (`rayzor_plugin_tensor_*` etc.)
// to dlopen time. The host binary side of the contract is the
// matching `-Wl,-export_dynamic` flag in `.cargo/config.toml`, which
// exports those symbols to dyld's global scope so they're reachable.
//
// Linux uses `-Wl,--export-dynamic` on the host binary alone — its
// ld.so resolves undefined cdylib refs against the host's exported
// symbol table without needing a cdylib-side flag.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
