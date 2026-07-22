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
        // In-process CoreML runtime for the nue graph engines (bert_graph ANE
        // encoder + prefill_graph Llama prefill). CoreML is the only Apple-
        // sanctioned path to the Neural Engine; the ObjC shim is compiled into
        // this cdylib and its `nue_coreml_*` symbols link intra-module.
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rerun-if-changed=src/coreml_shim.m");
        cc::Build::new()
            .file("src/coreml_shim.m")
            .flag("-fobjc-arc")
            .compile("coreml_shim");
    }
}
