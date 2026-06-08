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
    if !path.exists() {
        return Err(format!("file not found: {}", path.display()));
    }
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
    let plugin =
        plugin.ok_or_else(|| "no `plugin_describe` export with a non-empty table".to_string())?;

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

    Ok((lib, plugin, runtime_symbols))
}
