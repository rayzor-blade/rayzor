//! WASM Runner — executes compiled WASM binaries via embedded wasmtime.
//!
//! Used by `rayzor run --wasm` to run WASM programs without an external
//! wasmtime installation. Provides WASI P1 imports for stdout, filesystem,
//! environment, and clocks.

#[cfg(feature = "wasm-runtime")]
pub fn run_wasm(wasm_bytes: &[u8]) -> Result<(), String> {
    use wasmtime::*;

    let engine = Engine::default();
    let module =
        Module::new(&engine, wasm_bytes).map_err(|e| format!("WASM compilation failed: {}", e))?;

    // Build WASI P1 context with inherited stdio + current directory access
    let mut builder = wasi_common::WasiCtxBuilder::new();
    builder.inherit_stdio().inherit_env();

    // Grant access to current directory for file I/O
    if let Ok(cwd) = std::env::current_dir() {
        let _ = builder.preopened_dir(
            &cwd,
            ".",
            wasi_common::DirPerms::all(),
            wasi_common::FilePerms::all(),
        );
    }

    let wasi_ctx = builder.build_p1();

    let mut store = Store::new(&engine, wasi_ctx);

    // Create linker and add WASI P1 snapshot imports
    let mut linker = Linker::new(&engine);
    wasi_common::p1::add_to_linker_sync(&mut linker, |ctx| ctx)
        .map_err(|e| format!("WASI linker error: {}", e))?;

    // Provide stubs for "rayzor" module imports (thread/sync/future/arc/box/ereg).
    // These are preserved as real imports by the linker for browser JS runtime,
    // but wasmtime needs at least stub definitions.
    for import in module.imports() {
        if import.module() == "rayzor" {
            let name = import.name().to_string();
            let ty = import.ty();
            match ty {
                ExternType::Func(func_ty) => {
                    let results: Vec<ValType> = func_ty.results().collect();
                    let module_name = import.module().to_string();
                    linker
                        .func_new(&module_name, &name, func_ty.clone(), move |_caller, _params, out| {
                            // Return default values (0 for i32/i64, 0.0 for f32/f64)
                            for (i, r) in results.iter().enumerate() {
                                out[i] = match r {
                                    ValType::I32 => Val::I32(0),
                                    ValType::I64 => Val::I64(0),
                                    ValType::F32 => Val::F32(0),
                                    ValType::F64 => Val::F64(0),
                                    _ => Val::I32(0),
                                };
                            }
                            Ok(())
                        })
                        .map_err(|e| format!("Failed to stub {}: {}", name, e))?;
                }
                _ => {}
            }
        }
    }

    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| format!("WASM instantiation failed: {}", e))?;

    // Call _start
    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|e| format!("_start not found: {}", e))?;

    match start.call(&mut store, ()) {
        Ok(()) => Ok(()),
        Err(e) => {
            if let Some(exit) = e.downcast_ref::<wasi_common::I32Exit>() {
                if exit.0 == 0 {
                    return Ok(());
                }
                return Err(format!("process exited with code {}", exit.0));
            }
            Err(format!("WASM execution error: {}", e))
        }
    }
}

#[cfg(not(feature = "wasm-runtime"))]
pub fn run_wasm(_wasm_bytes: &[u8]) -> Result<(), String> {
    Err("WASM runtime not available. Install wasmtime or compile rayzor with --features wasm-runtime".to_string())
}
