fn main() {
    // The host supplies the `rayzor_plugin_*` carrier symbols when it loads
    // this library. Darwin requires opting into resolving them at load time;
    // ELF linkers defer them by default.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-undefined,dynamic_lookup");
    }
}
