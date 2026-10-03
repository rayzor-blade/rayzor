use std::path::{Path, PathBuf};

fn gpu_buffer_extensions(generated: String, source: &Path) -> Result<String, String> {
    let overlay = std::fs::read_to_string(source).map_err(|error| error.to_string())?;
    let body_start = overlay
        .find("extern class GpuBuffer")
        .and_then(|start| overlay[start..].find('{').map(|open| start + open + 1))
        .ok_or("Rayzor GpuBuffer extern has no class body")?;
    let body_end = overlay
        .rfind('}')
        .ok_or("Rayzor GpuBuffer extern is unclosed")?;
    let insert = generated
        .rfind('}')
        .ok_or("generated xgpu GpuBuffer extern is unclosed")?;
    let mut combined = generated;
    combined.insert_str(insert, &overlay[body_start..body_end]);
    Ok(combined)
}

fn write_haxe(root: &Path) -> Result<(), String> {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let committed = manifest.join("haxe/rayzor/gpu");
    let adapter = manifest.join("haxe-overrides/rayzor/gpu");
    let package = root.join("rayzor/gpu");
    std::fs::create_dir_all(&package).map_err(|error| error.to_string())?;
    // Copying a file onto itself truncates it, so the committed files are
    // copied only into a different directory.
    let same_dir = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same_dir(&committed, &package) {
        for entry in std::fs::read_dir(&committed).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            if entry.path().extension().and_then(|value| value.to_str()) == Some("hx") {
                std::fs::copy(entry.path(), package.join(entry.file_name()))
                    .map_err(|error| error.to_string())?;
            }
        }
    }

    for file in xgpu_bindgen::_haxe(xgpu_bindgen::haxe::Runtime::Rayzor)? {
        let path = root.join(file.path);
        std::fs::create_dir_all(path.parent().expect("generated extern has a parent"))
            .map_err(|error| error.to_string())?;
        let source = if path.ends_with("rayzor/gpu/GpuBuffer.hx") {
            gpu_buffer_extensions(file.source, &adapter.join("GpuBuffer.hx"))?
        } else {
            file.source
        };
        std::fs::write(path, source).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn main() {
    println!("cargo:rerun-if-env-changed=XGPU_HAXE_OUT");
    println!("cargo:rerun-if-env-changed=XGPU_HAXE_STAMP");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    xgpu_backend::install(&out).expect("xgpu backend installs");
    // The native package extends GpuBuffer with Rayzor's lazy-compute state.
    // The compiler's WebGPU-only feature shape does not compile that native
    // compute stack, so let xgpu emit its ordinary handle wrapper there.
    let adapter_resources: &[&str] = if std::env::var_os("CARGO_FEATURE_NATIVE").is_some() {
        &["GpuBuffer"]
    } else {
        &[]
    };
    // Method classes are named by the externs' package, as they declare them.
    let model = xgpu_bindgen::generate_rayzor_in(
        "gpu",
        xgpu_bindgen::RAYZOR_PACKAGE,
        xgpu_bindgen::gpu_api(),
        xgpu_bindgen::WEBGPU_IDL,
        adapter_resources,
    )
    .expect("xgpu Rayzor model generates");
    std::fs::write(out.join("xgpu_rayzor.rs"), model).expect("xgpu model writes");

    //if let Some(root) = std::env::var_os("XGPU_HAXE_OUT") {
    write_haxe(&PathBuf::from("haxe")).expect("xgpu Rayzor externs generate");
    // }
}
