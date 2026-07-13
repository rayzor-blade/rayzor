use std::path::Path;

/// Load a loose `.dylib`/`.so`/`.dll` referenced via `[build] native-libs`
/// in a project manifest. Mirrors the rpkg native-lib path:
/// 1. dlopen the file
/// 2. Locate `plugin_describe` (or `rayzor_plugin_describe`) -> method table
/// 3. Locate `plugin_init` (or `rayzor_plugin_init`) -> runtime symbol table
///
/// Returns the library handle, the constructed `NativePlugin`, and the
/// runtime symbols. Library must stay alive for the duration of the
/// compilation. Runtime symbols are registered with the JIT so call
/// sites in the dispatched methods can resolve to actual function ptrs.
#[repr(C)]
struct SymbolEntry {
    name_ptr: *const u8,
    name_len: usize,
    fn_ptr: *const std::ffi::c_void,
}

#[allow(clippy::type_complexity)]
pub fn load_manifest_native_lib(
    path: &Path,
) -> Result<
    (
        libloading::Library,
        compiler::compiler_plugin::NativePlugin,
        Vec<(String, *const u8)>,
    ),
    String,
> {
    // Manifests carry one platform's extension (e.g. `.dylib`), but the same
    // logical plugin builds to `.so` on Linux / `.dll` on Windows. If the exact
    // path is missing, retry with the host's dynamic-library extension so a
    // single `[build] native-libs` entry works cross-platform.
    let resolved: std::borrow::Cow<Path> = if path.exists() {
        std::borrow::Cow::Borrowed(path)
    } else {
        let host_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else if cfg!(target_os = "windows") {
            "dll"
        } else {
            "so"
        };
        let alt = path.with_extension(host_ext);
        if alt.exists() {
            std::borrow::Cow::Owned(alt)
        } else {
            return Err(format!(
                "file not found: {} (also tried {})",
                path.display(),
                alt.display()
            ));
        }
    };
    let path = resolved.as_ref();
    let lib =
        unsafe { libloading::Library::new(path) }.map_err(|e| format!("dlopen failed: {}", e))?;

    let plugin_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("native_lib");

    compiler::rpkg::install::check_plugin_abi(&lib, plugin_name)?;

    type DescribeFn = unsafe extern "C" fn(*mut usize) -> *const rayzor_plugin::NativeMethodDesc;
    let describe_names: &[&[u8]] = &[b"plugin_describe", b"rayzor_plugin_describe"];
    let mut plugin: Option<compiler::compiler_plugin::NativePlugin> = None;
    for name in describe_names {
        if let Ok(describe_fn) = unsafe { lib.get::<DescribeFn>(name) } {
            let mut count: usize = 0;
            let descs = unsafe { describe_fn(&mut count) };
            if !descs.is_null() && count > 0 {
                plugin = Some(unsafe {
                    compiler::compiler_plugin::NativePlugin::from_descriptors(
                        plugin_name,
                        descs,
                        count,
                    )
                });
            }
            break;
        }
    }
    // A plugin may omit `plugin_describe` entirely and instead provide only a
    // runtime symbol table (`plugin_init`) when its classes are dispatched by
    // the compiler's built-in mapping table (e.g. the tensor/quant kernels).
    // In that case the describe table is legitimately empty; fail only if the
    // symbol table below is also empty (the dylib provides nothing).
    let describe_was_empty = plugin.is_none();
    let plugin =
        plugin.unwrap_or_else(|| compiler::compiler_plugin::NativePlugin::empty(plugin_name));

    type InitFn = unsafe extern "C" fn(*mut usize) -> *const SymbolEntry;
    let init_names: &[&[u8]] = &[b"plugin_init", b"rayzor_plugin_init"];
    let mut runtime_symbols: Vec<(String, *const u8)> = Vec::new();
    for name in init_names {
        if let Ok(init_fn) = unsafe { lib.get::<InitFn>(name) } {
            let mut count: usize = 0;
            let ptr = unsafe { init_fn(&mut count) };
            if !ptr.is_null() && count > 0 {
                let entries = unsafe { std::slice::from_raw_parts(ptr, count) };
                for e in entries {
                    let name_bytes = unsafe { std::slice::from_raw_parts(e.name_ptr, e.name_len) };
                    let name = String::from_utf8_lossy(name_bytes).into_owned();
                    runtime_symbols.push((name, e.fn_ptr as *const u8));
                }
            }
            break;
        }
    }

    if describe_was_empty && runtime_symbols.is_empty() {
        return Err(
            "provides neither a `plugin_describe` method table nor a `plugin_init` symbol table"
                .to_string(),
        );
    }

    Ok((lib, plugin, runtime_symbols))
}
